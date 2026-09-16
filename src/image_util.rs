// ============================================================
// src/image_util.rs — VLM image decoding from OpenAI data URIs
// ============================================================
// OpenAI multimodal clients send images as data URIs:
//   data:image/jpeg;base64,<b64>
//
// This module: strip prefix → base64-decode → image crate decodes
// JPEG/PNG/WebP → to_rgb8() → flat NHWC Vec<u8> + dimensions.
//
// The result goes directly to the VLM bridge: the C++ side wraps the
// raw pixel pointer in an ov::Tensor({1, H, W, 3}, u8, data).
// ============================================================

use anyhow::Context;
use base64ct::{Base64, Encoding};

/// Decoded image ready for the VLM engine.
///
/// Layout: flat NHWC RGB uint8 — `height × width × 3` bytes, row-major.
/// Matches `ov::Tensor({1, height, width, 3}, ov::element::u8, data)`.
#[derive(Debug)]
pub struct DecodedImage {
    /// Raw pixel bytes: `height × width × 3` in RGB order.
    pub data: Vec<u8>,
    pub height: u32,
    pub width: u32,
}

/// Decode a `data:image/...;base64,<b64>` URI into an RGB pixel buffer.
///
/// Supports every format the `image` crate handles with our feature set:
/// JPEG, PNG, and WebP. GIF and other formats are silently rejected by
/// the underlying decoder rather than an explicit MIME check — this keeps
/// the code simple and avoids maintaining a format allowlist.
///
/// HTTP and HTTPS URLs are **not** fetched — returning them as an error
/// prevents accidental server-side request forgery in Phase 5v1.
///
/// # Errors
/// - Input is not a `data:` URI (HTTP/S URL, plain path, etc.)
/// - Missing `;base64,` delimiter in the URI
/// - Base64 payload is malformed
/// - Decoded bytes are not a recognisable / decodable image format
pub fn decode_data_uri(uri: &str) -> anyhow::Result<DecodedImage> {
    let rest = uri
        .strip_prefix("data:")
        .ok_or_else(|| anyhow::anyhow!("only data URIs are supported — got: {uri:.60}"))?;

    let (meta, b64) = rest.split_once(',').ok_or_else(|| {
        anyhow::anyhow!("malformed data URI: missing comma between metadata and payload")
    })?;

    if !meta.ends_with(";base64") {
        return Err(anyhow::anyhow!(
            "only base64-encoded data URIs are supported (got metadata: {meta})"
        ));
    }

    let bytes = Base64::decode_vec(b64).context("base64 decode failed")?;
    decode_image_bytes(&bytes)
}

/// Decode raw image file bytes (JPEG/PNG/WebP) into an RGB pixel buffer.
///
/// This is the format-decode tail shared with [`decode_data_uri`]; the
/// `/v1/images/edits` multipart handler calls it directly on uploaded file bytes
/// (which are not base64-wrapped, unlike chat data URIs).
///
/// # Errors
/// The bytes are not a recognisable / decodable image format.
pub fn decode_image_bytes(bytes: &[u8]) -> anyhow::Result<DecodedImage> {
    let img = image::load_from_memory(bytes).context("image decode failed")?;
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    Ok(DecodedImage {
        data: rgb.into_raw(),
        height,
        width,
    })
}

/// Decode an `OpenAI` edit **mask** (alpha PNG) into an OV inpaint mask buffer.
///
/// Conventions differ and must be translated:
/// - **`OpenAI`**: fully **transparent** pixels (alpha == 0) mark the region to
///   regenerate; any opacity is preserved.
/// - **OV `InpaintingPipeline`**: an RGB mask where **white (255)** marks the
///   region to regenerate and black is preserved.
///
/// So we map alpha → RGB: `alpha == 0` → white (edit), otherwise black (keep),
/// emitting `height × width × 3` RGB. A mask with no alpha channel decodes as
/// fully opaque → an all-black (no-op) mask, per the `OpenAI` contract.
///
/// # Errors
/// The bytes are not a recognisable / decodable image format.
pub fn decode_mask_bytes(bytes: &[u8]) -> anyhow::Result<DecodedImage> {
    let img = image::load_from_memory(bytes).context("mask decode failed")?;
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();

    let px_count = (width as usize)
        .checked_mul(height as usize)
        .context("mask dimensions overflow")?;
    let mut data = Vec::with_capacity(px_count * 3);
    for px in rgba.pixels() {
        // OpenAI transparent (alpha 0) ⇒ regenerate ⇒ OV white; else keep ⇒ black.
        let v = if px[3] == 0 { 255u8 } else { 0u8 };
        data.extend_from_slice(&[v, v, v]);
    }

    Ok(DecodedImage {
        data,
        height,
        width,
    })
}

/// Encode a flat NHWC RGB `u8` buffer (`width × height × 3`, row-major) as PNG
/// bytes — the inverse of [`decode_data_uri`]. The image generation pipeline
/// (`/v1/images/generations`) hands the SDXL output tensor here before base64.
///
/// # Errors
/// - `rgb.len()` does not equal `width × height × 3`
/// - the PNG encoder fails (effectively impossible for a valid buffer)
pub fn rgb_to_png(width: u32, height: u32, rgb: &[u8]) -> anyhow::Result<Vec<u8>> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|wh| wh.checked_mul(3))
        .context("image dimensions overflow")?;
    anyhow::ensure!(
        rgb.len() == expected,
        "pixel buffer length {} does not match {width}×{height}×3 = {expected}",
        rgb.len(),
    );
    // from_raw only returns None on a length mismatch, which we just ruled out.
    let img = image::RgbImage::from_raw(width, height, rgb.to_vec())
        .context("pixel buffer does not match dimensions")?;
    let mut buf = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .context("PNG encode failed")?;
    Ok(buf)
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use base64ct::{Base64, Encoding};
    use image::{DynamicImage, ImageFormat, RgbImage};

    use super::*;

    /// Encode an in-memory image to a data URI using the given MIME/format.
    fn to_data_uri(img: &DynamicImage, fmt: ImageFormat, mime: &str) -> String {
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), fmt)
            .unwrap();
        format!("data:{mime};base64,{}", Base64::encode_string(&buf))
    }

    /// JPEG round-trip: decode produces correct dimensions and 3 channels.
    #[test]
    fn decode_jpeg_data_uri_returns_rgb_pixels() {
        // x ∈ 0..4, y ∈ 0..3 — both always fit in u8; the cast cannot truncate.
        // (Newer clippy flags this fixture; older toolchains on the fleet did not.)
        #[allow(clippy::cast_possible_truncation)]
        let img = DynamicImage::ImageRgb8(RgbImage::from_fn(4, 3, |x, y| {
            image::Rgb([x as u8, y as u8, 128u8])
        }));
        let uri = to_data_uri(&img, ImageFormat::Jpeg, "image/jpeg");
        let decoded = decode_data_uri(&uri).unwrap();
        assert_eq!(decoded.width, 4);
        assert_eq!(decoded.height, 3);
        assert_eq!(
            decoded.data.len(),
            4 * 3 * 3,
            "RGB: 3 bytes per pixel × width × height"
        );
    }

    /// PNG round-trip: dimensions survive lossless encode/decode.
    #[test]
    fn decode_png_data_uri_returns_correct_dimensions() {
        let img =
            DynamicImage::ImageRgb8(RgbImage::from_fn(10, 7, |_, _| image::Rgb([42u8, 0, 255])));
        let uri = to_data_uri(&img, ImageFormat::Png, "image/png");
        let decoded = decode_data_uri(&uri).unwrap();
        assert_eq!(decoded.width, 10);
        assert_eq!(decoded.height, 7);
        assert_eq!(decoded.data.len(), 10 * 7 * 3);
    }

    /// HTTP URLs must be rejected — we don't fetch remote images.
    #[test]
    fn http_url_returns_error() {
        let err = decode_data_uri("https://example.com/photo.jpg").unwrap_err();
        assert!(
            err.to_string().contains("data URI"),
            "error must mention 'data URI': {err}"
        );
    }

    /// Malformed base64 payload → clear error mentioning base64.
    #[test]
    fn malformed_base64_returns_error() {
        let err = decode_data_uri("data:image/jpeg;base64,!!!not-valid-base64!!!").unwrap_err();
        assert!(
            err.to_string().contains("base64"),
            "error must mention 'base64': {err}"
        );
    }

    /// `rgb_to_png` emits valid PNG bytes (magic header) that decode back to the
    /// original dimensions — the encode side of the round-trip.
    #[test]
    fn rgb_to_png_emits_valid_png() {
        let (w, h) = (4u32, 3u32);
        let rgb = vec![200u8; (w * h * 3) as usize];
        let png = rgb_to_png(w, h, &rgb).unwrap();
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n", "PNG magic header");
        let decoded = image::load_from_memory(&png).unwrap();
        assert_eq!(decoded.width(), w);
        assert_eq!(decoded.height(), h);
    }

    /// A pixel buffer whose length disagrees with the dimensions is rejected.
    #[test]
    fn rgb_to_png_rejects_size_mismatch() {
        let err = rgb_to_png(2, 2, &[0u8; 3]).unwrap_err();
        assert!(
            err.to_string().contains("does not match"),
            "error must explain the mismatch: {err}"
        );
    }

    /// `decode_mask_bytes` maps `OpenAI` alpha → OV white/black: transparent pixels
    /// become white (edit), opaque pixels become black (keep).
    #[test]
    fn decode_mask_maps_alpha_to_white_edit_region() {
        use image::RgbaImage;
        // 2×1: left pixel transparent (edit), right pixel opaque (keep).
        let mask = DynamicImage::ImageRgba8(RgbaImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                image::Rgba([0, 0, 0, 0]) // transparent → edit
            } else {
                image::Rgba([0, 0, 0, 255]) // opaque → keep
            }
        }));
        let mut buf = Vec::new();
        mask.write_to(&mut std::io::Cursor::new(&mut buf), ImageFormat::Png)
            .unwrap();
        let decoded = decode_mask_bytes(&buf).unwrap();
        assert_eq!(decoded.width, 2);
        assert_eq!(decoded.height, 1);
        // Left pixel white (edit), right pixel black (keep).
        assert_eq!(&decoded.data[0..3], &[255, 255, 255]);
        assert_eq!(&decoded.data[3..6], &[0, 0, 0]);
    }

    /// A mask with no alpha channel decodes as fully opaque → all-black (no-op).
    #[test]
    fn decode_mask_without_alpha_is_all_keep() {
        let mask =
            DynamicImage::ImageRgb8(RgbImage::from_fn(3, 1, |_, _| image::Rgb([12, 200, 99])));
        let mut buf = Vec::new();
        mask.write_to(&mut std::io::Cursor::new(&mut buf), ImageFormat::Png)
            .unwrap();
        let decoded = decode_mask_bytes(&buf).unwrap();
        assert!(
            decoded.data.iter().all(|&b| b == 0),
            "no-alpha mask must be all-black (keep everything)"
        );
    }

    /// `decode_image_bytes` decodes raw file bytes (the multipart path) to RGB.
    #[test]
    fn decode_image_bytes_returns_rgb() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_fn(5, 4, |_, _| image::Rgb([1, 2, 3])));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), ImageFormat::Png)
            .unwrap();
        let decoded = decode_image_bytes(&buf).unwrap();
        assert_eq!((decoded.width, decoded.height), (5, 4));
        assert_eq!(decoded.data.len(), 5 * 4 * 3);
    }

    /// Valid base64 but the bytes are not a recognisable image → clear error.
    #[test]
    fn corrupt_image_payload_returns_error() {
        // "hello world" is valid base64 → valid bytes, but not a valid JPEG/PNG.
        let uri = format!(
            "data:image/jpeg;base64,{}",
            Base64::encode_string(b"hello world, definitely not an image")
        );
        let err = decode_data_uri(&uri).unwrap_err();
        assert!(
            err.to_string().contains("image decode"),
            "error must mention 'image decode': {err}"
        );
    }
}
