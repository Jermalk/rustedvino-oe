// ============================================================
// src/image_util.rs — VLM image decoding from OpenAI data URIs
// ============================================================
// OpenAI multimodal clients send images as data URIs:
//   data:image/jpeg;base64,<b64>
//
// This module: strip prefix → base64-decode → image crate decodes
// JPEG/PNG/WebP → into_rgb8() → flat NHWC Vec<u8> + dimensions. Chat
// images to a vision model are checked from the header first (S5c) and
// decoded later, then downscaled to `max_image_pixels` (C28).
//
// The result goes directly to the VLM bridge: the C++ side wraps the
// raw pixel pointer in an ov::Tensor({1, H, W, 3}, u8, data).
// ============================================================

use anyhow::Context;
use base64ct::{Base64, Encoding};

/// Per-request image limits (0.9.1 S5, outside review I-02), from
/// [`crate::model_manager::Config::max_images_per_request`] and
/// [`crate::model_manager::Config::max_image_pixels_per_request`]. The request-body limit bounds
/// only the compressed bytes; a tiny PNG can declare 16k x 16k pixels, so without these one
/// request could decode hundreds of images of up to ~1 GB each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageLimits {
    /// Most images one request may carry.
    pub max_images: usize,
    /// Most pixels (`width x height`, summed over the request's images) one request may decode.
    pub max_pixels: u64,
}

impl Default for ImageLimits {
    /// The config defaults: 16 images, 64 MP in total.
    fn default() -> Self {
        Self {
            max_images: crate::model_manager::config::default_max_images_per_request(),
            max_pixels: crate::model_manager::config::default_max_image_pixels_per_request(),
        }
    }
}

/// Running use of an [`ImageLimits`] across one request's images.
#[derive(Debug)]
pub struct ImageBudget {
    limits: ImageLimits,
    images: usize,
    pixels: u64,
}

impl ImageBudget {
    /// A fresh budget for one request.
    #[must_use]
    pub fn new(limits: ImageLimits) -> Self {
        Self {
            limits,
            images: 0,
            pixels: 0,
        }
    }
}

/// An image request over a server limit — the caller's error (a 400), carried inside the
/// `anyhow::Error` so the handler can `downcast_ref` it and answer with `code`.
#[derive(Debug)]
pub struct ImageLimitExceeded {
    /// `too_many_images` or `image_too_large`.
    pub code: &'static str,
    /// Names the limit and the config key; never any of the caller's bytes.
    pub message: String,
}

impl std::fmt::Display for ImageLimitExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ImageLimitExceeded {}

/// Image work that may run at once on the blocking pool (0.9.1 S5b, Fable review): header
/// passes (dimensions only; at most 64 MiB each, [`HEADER_READ_MAX_ALLOC`]) and the full decodes
/// of `/v1/images/edits`. Run on the async runtime, a handful of large decodes stalled every
/// other request (`/health` included).
///
/// Memory of one full decode, for a request at the 64 MP `max_image_pixels_per_request`
/// default (the pixel sum bounds one request's decodes): 8-bit RGB ~192 MiB; 8-bit RGBA PNG
/// ~0.45 GiB transient (RGBA output + the RGB copy); 16-bit PNG up to ~0.7 GiB (`max_alloc`
/// 8 B/px + the RGB copy); progressive or CMYK JPEG up to ~1.1 GiB (zune-jpeg's coefficient
/// buffers, ~14 B/px, are bounded by the header's dimensions, not by `max_alloc`). A vision
/// decode adds the downscale's working set (see [`VLM_DECODE_SLOTS`]).
///
/// Server-wide at most four full decodes run at once — two here (edits) and two behind
/// [`VLM_DECODE_SLOTS`] — so the worst case in flight is ~4.7 GiB (four 64 MP progressive
/// JPEGs, two of them with a downscale); a typical 12 MP 8-bit photo is ~36 MB.
static DECODE_SLOTS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(2);

/// Full decodes of chat images for a vision model, run while the request holds its engine slot
/// (0.9.1 S5c, review F3): a semaphore of their own, so a request holding a slot never queues
/// behind header passes or image edits. Two at a time. One costs the decode itself (as
/// [`DECODE_SLOTS`] describes: ~192 MiB for a 64 MP 8-bit image, up to ~1.1 GiB for a 64 MP
/// progressive JPEG) plus the downscale's working set, at most about 32 B x
/// `max_image_pixels` (~128 MB at the 4 MP default, see [`downscale_rgb`]).
static VLM_DECODE_SLOTS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(2);

/// Take every [`VLM_DECODE_SLOTS`] permit (test seam, review N3): a vision request that needs
/// a full decode then waits until the returned permit drops.
#[cfg(test)]
pub(crate) async fn hold_vlm_decode_slots() -> Option<tokio::sync::SemaphorePermit<'static>> {
    VLM_DECODE_SLOTS.acquire_many(2).await.ok()
}

/// Run an image decode (`f`) on the blocking pool, behind [`DECODE_SLOTS`]. Waiting for a
/// slot is async, so a queue of decodes never occupies a runtime worker.
///
/// # Errors
/// The decode task panicked or was cancelled (the semaphore is never closed).
pub async fn decode_off_runtime<T, F>(f: F) -> anyhow::Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    decode_off_runtime_with(&DECODE_SLOTS, f).await
}

/// [`decode_off_runtime`] behind [`VLM_DECODE_SLOTS`]: the full decode of a vision request's
/// images after it took its engine slot.
///
/// # Errors
/// As [`decode_off_runtime`].
pub async fn decode_for_vlm_off_runtime<T, F>(f: F) -> anyhow::Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    decode_off_runtime_with(&VLM_DECODE_SLOTS, f).await
}

/// Run `f` on the blocking pool behind a permit of `slots`. The permit moves INTO the blocking
/// closure (review F1): if the caller's future is dropped (a client disconnect) while `f` runs,
/// the decode goes on to the end and keeps its permit until then, so the next waiter does not
/// start a decode on top of it.
async fn decode_off_runtime_with<T, F>(
    slots: &'static tokio::sync::Semaphore,
    f: F,
) -> anyhow::Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let slot = slots.acquire().await.context("image decode slots closed")?;
    tokio::task::spawn_blocking(move || {
        let _slot = slot;
        f()
    })
    .await
    .context("image decode task failed")
}

/// Allocation allowed while only reading an image's header (dimensions).
const HEADER_READ_MAX_ALLOC: u64 = 64 << 20;

/// An image whose header passed the request's [`ImageBudget`], not yet decoded: the compressed
/// bytes and the dimensions the header declares (0.9.1 S5c). A vision request holds these — not
/// decoded RGB — while it waits for device admission and an engine slot; it decodes with
/// [`decode_pending_for_vlm`] once it holds the slot.
#[derive(Debug)]
pub struct PendingImage {
    bytes: Vec<u8>,
    width: u32,
    height: u32,
}

impl PendingImage {
    /// Width the header declares.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height the header declares.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }
}

/// Count the image in `budget`, read its dimensions from the header and refuse — before any
/// pixel is allocated — when the request would pass a limit. Returns the header's dimensions.
fn check_within_budget(
    bytes: &[u8],
    budget: &mut ImageBudget,
    what: &str,
) -> anyhow::Result<(u32, u32)> {
    let limits = budget.limits;
    if budget.images >= limits.max_images {
        return Err(anyhow::Error::new(ImageLimitExceeded {
            code: "too_many_images",
            message: format!(
                "too many images: at most {} per request (max_images_per_request)",
                limits.max_images
            ),
        }));
    }
    // The header read gets explicit limits too: building a decoder may buffer metadata
    // (PNG text/ICC chunks, EXIF), and the crate's default would allow 512 MiB for it.
    let mut probe = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .context("image read failed")?;
    let mut probe_limits = image::Limits::default();
    probe_limits.max_alloc = Some(HEADER_READ_MAX_ALLOC);
    probe.limits(probe_limits);
    let (width, height) = probe
        .into_dimensions()
        .with_context(|| format!("{what} decode failed"))?;
    let pixels = u64::from(width) * u64::from(height);
    let total = budget.pixels.saturating_add(pixels);
    if total > limits.max_pixels {
        return Err(anyhow::Error::new(ImageLimitExceeded {
            code: "image_too_large",
            message: format!(
                "image too large: {width}x{height} would bring this request to {total} pixels, \
                 over the limit of {} (max_image_pixels_per_request)",
                limits.max_pixels
            ),
        }));
    }
    budget.images += 1;
    budget.pixels = total;
    Ok((width, height))
}

/// Decode a header-checked image with explicit `image::Limits` (exact width/height, and an
/// allocation cap from them) so a header that lies cannot make the decoder allocate more.
fn decode_checked(
    bytes: &[u8],
    (width, height): (u32, u32),
    what: &str,
) -> anyhow::Result<image::DynamicImage> {
    #[cfg(test)]
    test_hook::record_full_decode(width, height);
    let pixels = u64::from(width) * u64::from(height);
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .context("image read failed")?;
    let mut decode_limits = image::Limits::default();
    decode_limits.max_image_width = Some(width);
    decode_limits.max_image_height = Some(height);
    // 8 bytes per pixel covers the widest decoder output (16-bit RGBA), plus slack for the
    // decoder's own scratch buffers.
    decode_limits.max_alloc = Some(pixels.saturating_mul(8).saturating_add(16 << 20));
    reader.limits(decode_limits);
    reader
        .decode()
        .with_context(|| format!("{what} decode failed"))
}

/// Decode `bytes` within `budget`: [`check_within_budget`] then [`decode_checked`].
fn decode_within_budget(
    bytes: &[u8],
    budget: &mut ImageBudget,
    what: &str,
) -> anyhow::Result<image::DynamicImage> {
    let dims = check_within_budget(bytes, budget, what)?;
    decode_checked(bytes, dims, what)
}

/// The size an image of `width x height` is resized to so that it has at most `max_pixels`
/// pixels (0.9.1 C28, `max_image_pixels`), or `None` when it already fits (it is then passed
/// through untouched). Both sides are scaled by the same factor (aspect ratio kept, up to
/// rounding), each side stays at least 1, and the result is never over `max_pixels` — the float
/// estimate is corrected in integer arithmetic. `max_pixels` of 0 is treated as 1 (the config
/// refuses 0 at load).
#[must_use]
pub fn downscaled_dims(width: u32, height: u32, max_pixels: u64) -> Option<(u32, u32)> {
    let cap = max_pixels.max(1);
    let (w, h) = (u64::from(width), u64::from(height));
    if w * h <= cap {
        return None;
    }
    // w*h > cap >= 1, so both sides are >= 1 and the ratio is in (0, 1).
    #[allow(clippy::cast_precision_loss)]
    let scale = (cap as f64 / (w * h) as f64).sqrt();
    // Truncation is the intent (floor); the values are in [0, side], side <= u32::MAX.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let fit = |side: u64| ((side as f64 * scale) as u64).clamp(1, side);
    let (mut nw, mut nh) = (fit(w), fit(h));
    // Float error, or a side clamped up to 1 (e.g. 5 000 000 x 1 under a 4 MP cap): shrink the
    // longer side until the product fits. `cap / other >= 1` because other <= cap here.
    if nw * nh > cap {
        if nw >= nh {
            nw = (cap / nh).max(1);
        } else {
            nh = (cap / nw).max(1);
        }
    }
    // Both are <= the original u32 sides.
    Some((
        u32::try_from(nw).unwrap_or(width),
        u32::try_from(nh).unwrap_or(height),
    ))
}

/// Who an image belongs to, for the downscale log line (review F8): the decode runs on the
/// blocking pool, outside any request context.
#[derive(Debug, Clone, Copy, Default)]
pub struct ImageOwner<'a> {
    /// The model the image is for.
    pub model: &'a str,
    /// The request's id (`chatcmpl-rv-…`).
    pub request_id: &'a str,
}

/// `rgb` resized to [`downscaled_dims`], or `rgb` itself — the same buffer, no copy — when it
/// already fits within `max_pixels`.
///
/// Two passes (review F2): when a side shrinks by 2x or more, an integer area-average
/// (`imageops::thumbnail`, by `floor(ratio)` per side) first, and the source is dropped; then
/// Catmull-Rom (bicubic) for the remaining ratio (< 2x per side), skipped when nothing is left.
/// `imageops::resize` alone keeps a `width x new_height` RGBA f32 intermediate (16 B/px), which
/// for a 64 MP source under a 4 MP cap is ~262 MB on top of the source; the pre-reduction caps
/// that intermediate at about 32 B x `max_pixels` whatever the source size.
#[must_use]
pub fn downscale_rgb(
    rgb: image::RgbImage,
    max_pixels: u64,
    owner: ImageOwner<'_>,
) -> image::RgbImage {
    let (width, height) = rgb.dimensions();
    let Some((nw, nh)) = downscaled_dims(width, height, max_pixels) else {
        return rgb;
    };
    // nw <= width and nh <= height, so both factors are >= 1, and width / kx >= nw.
    let (kx, ky) = ((width / nw).max(1), (height / nh).max(1));
    let src = if kx >= 2 || ky >= 2 {
        let reduced = image::imageops::thumbnail(&rgb, width / kx, height / ky);
        drop(rgb);
        reduced
    } else {
        rgb
    };
    let out = if src.dimensions() == (nw, nh) {
        src
    } else {
        image::imageops::resize(&src, nw, nh, image::imageops::FilterType::CatmullRom)
    };
    tracing::info!(
        model = owner.model,
        request_id = owner.request_id,
        from_width = width,
        from_height = height,
        to_width = nw,
        to_height = nh,
        max_image_pixels = max_pixels,
        "image downscaled to max_image_pixels"
    );
    out
}

/// Header pass of a chat `image_url` (0.9.1 S5c): parse the `data:` URI, base64-decode it and
/// check the image against `budget` from its header ([`ImageLimitExceeded`] or a format error
/// before any pixel exists). The returned [`PendingImage`] holds only the compressed bytes;
/// [`decode_pending_for_vlm`] decodes it later, once the request holds an engine slot.
///
/// # Errors
/// As [`decode_data_uri`], except that a body corrupt past a valid header is only found by
/// [`decode_pending_for_vlm`].
pub fn check_data_uri(uri: &str, budget: &mut ImageBudget) -> anyhow::Result<PendingImage> {
    let bytes = data_uri_bytes(uri)?;
    let (width, height) = check_within_budget(&bytes, budget, "image")?;
    Ok(PendingImage {
        bytes,
        width,
        height,
    })
}

/// Decode a header-checked chat image for a vision model: RGB, then downscaled to at most
/// `max_image_pixels` pixels (0.9.1 C28, [`downscale_rgb`]). The compressed bytes are dropped
/// as soon as the decode returns.
///
/// # Errors
/// The image body does not decode (corrupt past its header, or the header lied).
pub fn decode_pending_for_vlm(
    pending: PendingImage,
    max_image_pixels: u64,
    owner: ImageOwner<'_>,
) -> anyhow::Result<DecodedImage> {
    let img = decode_checked(&pending.bytes, (pending.width, pending.height), "image")?;
    drop(pending);
    let rgb = downscale_rgb(img.into_rgb8(), max_image_pixels, owner);
    let (width, height) = rgb.dimensions();
    Ok(DecodedImage {
        data: rgb.into_raw(),
        height,
        width,
    })
}

/// Test hook (0.9.1 S5c): every full decode, by its header dimensions, so a test can tell
/// whether a request decoded its pixels before or after it waited. Keyed by dimensions so tests
/// running in parallel (each with its own odd size) do not see each other.
#[cfg(test)]
pub(crate) mod test_hook {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex, PoisonError};

    static FULL_DECODES: Mutex<Vec<(u32, u32)>> = Mutex::new(Vec::new());

    pub(super) fn record_full_decode(width: u32, height: u32) {
        if let Ok(mut v) = FULL_DECODES.lock() {
            v.push((width, height));
        }
        let gate = GATES.lock().ok().and_then(|g| {
            g.iter()
                .find(|(d, _)| *d == (width, height))
                .map(|(_, gate)| Arc::clone(gate))
        });
        if let Some(gate) = gate {
            gate.entered.fetch_add(1, Ordering::SeqCst);
            let mut open = gate.open.lock().unwrap_or_else(PoisonError::into_inner);
            while !*open {
                open = gate.cv.wait(open).unwrap_or_else(PoisonError::into_inner);
            }
        }
    }

    /// Holds every full decode of one image size until [`DecodeGate::open`] (review F5: a
    /// client that goes away while its request decodes).
    #[derive(Default)]
    pub(crate) struct DecodeGate {
        open: Mutex<bool>,
        cv: Condvar,
        entered: AtomicUsize,
    }

    impl DecodeGate {
        /// Let every held and later decode of this size run.
        pub(crate) fn open(&self) {
            *self.open.lock().unwrap_or_else(PoisonError::into_inner) = true;
            self.cv.notify_all();
        }

        /// How many decodes have reached the gate.
        pub(crate) fn entered(&self) -> usize {
            self.entered.load(Ordering::SeqCst)
        }
    }

    type GateEntry = ((u32, u32), Arc<DecodeGate>);
    static GATES: Mutex<Vec<GateEntry>> = Mutex::new(Vec::new());

    /// From now on, every full decode of a `width x height` image waits at the returned gate.
    pub(crate) fn gate_decodes_of(width: u32, height: u32) -> Arc<DecodeGate> {
        let gate = Arc::new(DecodeGate::default());
        if let Ok(mut g) = GATES.lock() {
            g.push(((width, height), Arc::clone(&gate)));
        }
        gate
    }

    /// How many full decodes of a `width x height` image have run in this process.
    pub(crate) fn full_decodes_of(width: u32, height: u32) -> usize {
        FULL_DECODES
            .lock()
            .map_or(0, |v| v.iter().filter(|&&d| d == (width, height)).count())
    }

    /// A tiny PNG whose header declares `width x height`, with a valid 4x4 body behind it: the
    /// shape of the attack (outside review I-02) — kilobytes on the wire, gigapixels declared.
    pub(crate) fn png_declaring(width: u32, height: u32) -> Vec<u8> {
        let small = image::RgbImage::from_pixel(4, 4, image::Rgb([7, 7, 7]));
        let mut buf = Vec::new();
        let _ = image::DynamicImage::ImageRgb8(small)
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png);
        // IHDR data starts at byte 16 (8 signature + 4 length + 4 type); fix its CRC after.
        buf[16..20].copy_from_slice(&width.to_be_bytes());
        buf[20..24].copy_from_slice(&height.to_be_bytes());
        let crc = crc32(&buf[12..29]);
        buf[29..33].copy_from_slice(&crc.to_be_bytes());
        buf
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut c = 0xFFFF_FFFF_u32;
        for &b in data {
            c ^= u32::from(b);
            for _ in 0..8 {
                c = if c & 1 == 1 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
        }
        !c
    }
}

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
pub fn decode_data_uri(uri: &str, budget: &mut ImageBudget) -> anyhow::Result<DecodedImage> {
    let bytes = data_uri_bytes(uri)?;
    decode_image_bytes(&bytes, budget)
}

/// The payload of a `data:<mime>;base64,<b64>` URI, base64-decoded.
fn data_uri_bytes(uri: &str) -> anyhow::Result<Vec<u8>> {
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

    Base64::decode_vec(b64).context("base64 decode failed")
}

/// Decode raw image file bytes (JPEG/PNG/WebP) into an RGB pixel buffer.
///
/// This is the format-decode tail shared with [`decode_data_uri`]; the
/// `/v1/images/edits` multipart handler calls it directly on uploaded file bytes
/// (which are not base64-wrapped, unlike chat data URIs).
///
/// # Errors
/// The bytes are not a recognisable / decodable image format, or the image passes `budget`
/// ([`ImageLimitExceeded`]).
pub fn decode_image_bytes(bytes: &[u8], budget: &mut ImageBudget) -> anyhow::Result<DecodedImage> {
    let img = decode_within_budget(bytes, budget, "image")?;
    // `into_rgb8`: no copy when the decoder already produced 8-bit RGB (JPEG, most PNGs).
    let rgb = img.into_rgb8();
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
/// The bytes are not a recognisable / decodable image format, or the mask passes `budget`
/// ([`ImageLimitExceeded`]).
pub fn decode_mask_bytes(bytes: &[u8], budget: &mut ImageBudget) -> anyhow::Result<DecodedImage> {
    let img = decode_within_budget(bytes, budget, "mask")?;
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

    use super::test_hook::png_declaring;
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
        let decoded = decode_data_uri(&uri, &mut ImageBudget::new(ImageLimits::default())).unwrap();
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
        let decoded = decode_data_uri(&uri, &mut ImageBudget::new(ImageLimits::default())).unwrap();
        assert_eq!(decoded.width, 10);
        assert_eq!(decoded.height, 7);
        assert_eq!(decoded.data.len(), 10 * 7 * 3);
    }

    /// HTTP URLs must be rejected — we don't fetch remote images.
    #[test]
    fn http_url_returns_error() {
        let err = decode_data_uri(
            "https://example.com/photo.jpg",
            &mut ImageBudget::new(ImageLimits::default()),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("data URI"),
            "error must mention 'data URI': {err}"
        );
    }

    /// Malformed base64 payload → clear error mentioning base64.
    #[test]
    fn malformed_base64_returns_error() {
        let err = decode_data_uri(
            "data:image/jpeg;base64,!!!not-valid-base64!!!",
            &mut ImageBudget::new(ImageLimits::default()),
        )
        .unwrap_err();
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
        let decoded =
            decode_mask_bytes(&buf, &mut ImageBudget::new(ImageLimits::default())).unwrap();
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
        let decoded =
            decode_mask_bytes(&buf, &mut ImageBudget::new(ImageLimits::default())).unwrap();
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
        let decoded =
            decode_image_bytes(&buf, &mut ImageBudget::new(ImageLimits::default())).unwrap();
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
        let err = decode_data_uri(&uri, &mut ImageBudget::new(ImageLimits::default())).unwrap_err();
        assert!(
            err.to_string().contains("image decode"),
            "error must mention 'image decode': {err}"
        );
    }

    /// 0.9.1 S5: a PNG declaring 65535x65535 (4.3 GP, ~12.9 GB as RGB) is refused from its
    /// header, before any pixel buffer exists, with a typed error naming the limit.
    #[test]
    fn png_declaring_huge_dimensions_is_refused_before_decoding() {
        let png = png_declaring(65_535, 65_535);
        assert!(png.len() < 200, "the attack is tiny: {} bytes", png.len());
        let mut budget = ImageBudget::new(ImageLimits::default());
        let err = decode_image_bytes(&png, &mut budget).unwrap_err();
        let limit = err.downcast_ref::<ImageLimitExceeded>().unwrap();
        assert_eq!(limit.code, "image_too_large");
        assert!(
            limit.message.contains("max_image_pixels_per_request"),
            "{limit}"
        );
    }

    /// The count cap: the (n+1)th image is refused, whatever its size.
    #[test]
    fn images_over_the_count_are_refused() {
        let limits = ImageLimits {
            max_images: 2,
            max_pixels: 1 << 20,
        };
        let ok = {
            let img = RgbImage::from_pixel(4, 4, image::Rgb([1, 2, 3]));
            let mut buf = Vec::new();
            DynamicImage::ImageRgb8(img)
                .write_to(&mut std::io::Cursor::new(&mut buf), ImageFormat::Png)
                .unwrap();
            buf
        };
        let mut budget = ImageBudget::new(limits);
        decode_image_bytes(&ok, &mut budget).unwrap();
        decode_image_bytes(&ok, &mut budget).unwrap();
        let err = decode_image_bytes(&ok, &mut budget).unwrap_err();
        assert_eq!(
            err.downcast_ref::<ImageLimitExceeded>().unwrap().code,
            "too_many_images"
        );
    }

    /// The pixel budget is summed over the request: two images that each fit alone do not
    /// fit together; the mask counts like an image.
    #[test]
    fn pixels_are_summed_over_the_request_including_the_mask() {
        let img = RgbImage::from_pixel(300, 300, image::Rgb([1, 2, 3]));
        let mut buf = Vec::new();
        DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut buf), ImageFormat::Png)
            .unwrap();
        let limits = ImageLimits {
            max_images: 16,
            max_pixels: 150_000,
        };
        let mut budget = ImageBudget::new(limits);
        decode_image_bytes(&buf, &mut budget).unwrap(); // 90 000
        let err = decode_mask_bytes(&buf, &mut budget).unwrap_err(); // + 90 000 > 150 000
        let limit = err.downcast_ref::<ImageLimitExceeded>().unwrap();
        assert_eq!(limit.code, "image_too_large");
        assert!(limit.message.contains("180000"), "{limit}");
    }

    /// 0.9.1 C28: the downscale size table. Each result fits the cap, keeps each side >= 1 and
    /// never grows a side; at or under the cap the image is left alone.
    #[test]
    fn downscaled_dims_table() {
        const MP4: u64 = 4_000_000;
        // (width, height, cap, expected)
        type Case = (u32, u32, u64, Option<(u32, u32)>);
        let cases: &[Case] = &[
            (2000, 2000, MP4, None),               // exactly the cap
            (4000, 1000, MP4, None),               // exactly the cap, 4:1
            (2400, 1667, MP4, Some((2399, 1666))), // the measured "4 MP" step: 800 px over
            (4000, 3000, MP4, Some((2309, 1732))), // 12 MP phone photo
            (3264, 2448, MP4, Some((2309, 1732))), // 8 MP
            (1, 1, MP4, None),                     // tiny
            (3, 5, 1, Some((1, 1))),               // tiny over a tiny cap
            (20_000, 1, MP4, None),                // extreme aspect, under the cap
            (1, 20_000, MP4, None),
            (20_000, 1, 1000, Some((1000, 1))), // extreme aspect, over: short side stays 1
            (1, 20_000, 1000, Some((1, 1000))),
            (5_000_000, 1, MP4, Some((4_000_000, 1))), // a side clamped up to 1
            (60_000, 1000, MP4, Some((15_491, 258))),  // 60 MP, within the S5 sum
            (7, 3, 20, Some((6, 2))),                  // odd sizes
            (2001, 2000, MP4, Some((2000, 1999))),     // cap + 2000: rounding
        ];
        for &(w, h, cap, want) in cases {
            let got = downscaled_dims(w, h, cap);
            assert_eq!(got, want, "{w}x{h} cap {cap}");
            if let Some((nw, nh)) = got {
                assert!(
                    u64::from(nw) * u64::from(nh) <= cap,
                    "{w}x{h} -> {nw}x{nh} over {cap}"
                );
                assert!(
                    nw >= 1 && nh >= 1 && nw <= w && nh <= h,
                    "{w}x{h} -> {nw}x{nh}"
                );
            }
        }
    }

    /// The cap + 1 edge and a sweep: rounding never lands over the cap, and the result is never
    /// needlessly small (within one row/column of the exact area).
    #[test]
    fn downscaled_dims_never_over_the_cap() {
        assert_eq!(
            downscaled_dims(2000, 2000, 3_999_999)
                .map(|(w, h)| u64::from(w) * u64::from(h) <= 3_999_999),
            Some(true)
        );
        for cap in [65_536_u64, 1_000_000, 4_000_000, 4_194_304] {
            for (w, h) in [
                (4001_u32, 999_u32),
                (9999, 9999),
                (12_345, 6789),
                (65_535, 977),
                (977, 65_535),
            ] {
                if let Some((nw, nh)) = downscaled_dims(w, h, cap) {
                    let area = u64::from(nw) * u64::from(nh);
                    assert!(area <= cap, "{w}x{h} cap {cap} -> {nw}x{nh}");
                    let slack = u64::from(nw.max(nh)) * 2 + 2;
                    assert!(
                        area + slack >= cap,
                        "{w}x{h} cap {cap} -> {nw}x{nh}: too small"
                    );
                }
            }
        }
    }

    /// An image at or under the cap comes back as the same buffer (no copy); one over is resized
    /// to the table's size.
    #[test]
    fn downscale_rgb_passes_small_images_through_without_a_copy() {
        let img = RgbImage::from_pixel(100, 50, image::Rgb([9, 8, 7]));
        let ptr = img.as_raw().as_ptr();
        let out = downscale_rgb(img, 5000, ImageOwner::default());
        assert_eq!(out.dimensions(), (100, 50));
        assert_eq!(out.as_raw().as_ptr(), ptr, "under the cap: same buffer");
        let big = RgbImage::from_pixel(100, 50, image::Rgb([9, 8, 7]));
        let out = downscale_rgb(big, 4999, ImageOwner::default());
        assert_eq!(Some(out.dimensions()), downscaled_dims(100, 50, 4999));
        assert_eq!(out.dimensions(), (99, 49));
    }

    /// The VLM path: header check, then decode + downscale to `max_image_pixels`.
    #[test]
    fn checked_data_uri_decodes_downscaled_for_the_vlm() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(400, 300, image::Rgb([1, 2, 3])));
        let uri = to_data_uri(&img, ImageFormat::Png, "image/png");
        let mut budget = ImageBudget::new(ImageLimits::default());
        let pending = check_data_uri(&uri, &mut budget).unwrap();
        assert_eq!((pending.width(), pending.height()), (400, 300));
        let decoded = decode_pending_for_vlm(pending, 30_000, ImageOwner::default()).unwrap();
        assert_eq!((decoded.width, decoded.height), (200, 150));
        assert_eq!(decoded.data.len(), 200 * 150 * 3);
        // Under the cap: original size.
        let pending = check_data_uri(&uri, &mut budget).unwrap();
        let decoded = decode_pending_for_vlm(pending, 4_000_000, ImageOwner::default()).unwrap();
        assert_eq!((decoded.width, decoded.height), (400, 300));
    }

    /// The header pass refuses what S5 refuses — the per-request pixel sum counts the ORIGINAL
    /// header pixels, not the downscaled size — and finds format errors; a body corrupt past a
    /// valid header is only found by the decode.
    #[test]
    fn header_pass_refuses_limits_and_leaves_corrupt_bodies_to_the_decode() {
        let huge = format!(
            "data:image/png;base64,{}",
            Base64::encode_string(&png_declaring(65_535, 65_535))
        );
        let err = check_data_uri(&huge, &mut ImageBudget::new(ImageLimits::default())).unwrap_err();
        assert_eq!(
            err.downcast_ref::<ImageLimitExceeded>().unwrap().code,
            "image_too_large"
        );
        let junk = format!(
            "data:image/png;base64,{}",
            Base64::encode_string(b"not an image")
        );
        let err = check_data_uri(&junk, &mut ImageBudget::new(ImageLimits::default())).unwrap_err();
        assert!(err.to_string().contains("image"), "{err}");
        // 300x300 declared, 4x4 of data behind it: the header passes, the decode fails.
        let lying = format!(
            "data:image/png;base64,{}",
            Base64::encode_string(&png_declaring(300, 300))
        );
        let mut budget = ImageBudget::new(ImageLimits {
            max_images: 16,
            max_pixels: 150_000,
        });
        let pending = check_data_uri(&lying, &mut budget).unwrap();
        assert!(decode_pending_for_vlm(pending, 4_000_000, ImageOwner::default()).is_err());
        // Original pixels count: a second 300x300 (90 000) passes the 150 000 sum only if the
        // first counted at its downscaled size — it must not.
        assert!(check_data_uri(&lying, &mut budget).is_err());
    }

    /// Review F1: the decode slot lives in the blocking closure. Dropping the caller's future
    /// while the decode runs (a client disconnect) does not give the slot back until the decode
    /// ends, so the next waiter cannot start a second decode on top of it.
    #[tokio::test]
    async fn a_dropped_caller_keeps_the_decode_slot_until_the_decode_ends() {
        static SLOTS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(1);
        let (started_tx, started_rx) = std::sync::mpsc::channel::<()>();
        let (finish_tx, finish_rx) = std::sync::mpsc::channel::<()>();
        let task = tokio::spawn(decode_off_runtime_with(&SLOTS, move || {
            let _ = started_tx.send(());
            let _ = finish_rx.recv();
        }));
        tokio::task::spawn_blocking(move || started_rx.recv())
            .await
            .unwrap()
            .unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert_eq!(
            SLOTS.available_permits(),
            0,
            "the running decode still holds its slot"
        );
        finish_tx.send(()).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while SLOTS.available_permits() == 0 {
            assert!(std::time::Instant::now() < deadline, "slot never released");
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    }

    /// Review F2: the two-pass downscale (integer area pre-reduction, then Catmull-Rom) lands
    /// exactly on `downscaled_dims` for large and small ratios and keeps a flat colour flat.
    #[test]
    fn two_pass_downscale_hits_the_size_and_keeps_the_colour() {
        for (w, h, cap) in [
            (2000_u32, 1500_u32, 100_000_u64), // 5.5x per side: pre-reduce by 5, then bicubic
            (2000, 1000, 500_000),             // 2x exactly: pre-reduce only
            (1100, 1000, 1_000_000),           // < 2x: bicubic only
            (6000, 10, 20_000),                // wide: x only
        ] {
            let img = RgbImage::from_pixel(w, h, image::Rgb([200, 100, 50]));
            let out = downscale_rgb(img, cap, ImageOwner::default());
            assert_eq!(
                Some(out.dimensions()),
                downscaled_dims(w, h, cap),
                "{w}x{h} {cap}"
            );
            assert!(
                out.pixels().all(|p| p.0 == [200, 100, 50]),
                "{w}x{h} {cap}: colour changed"
            );
        }
    }

    /// Review N1: the downscale keeps the picture in place — a red/blue half split and a
    /// vertical gradient survive 8x and 3.3x downscales at the corners and on both sides of the
    /// split, so a shift, a crop or a smear fails this.
    #[test]
    fn downscale_keeps_the_picture_in_place() {
        // (width, height, cap): 800x400 -> 100x50 (8x); 660x330 -> ~200x100 (3.3x).
        for (w, h, cap) in [(800_u32, 400_u32, 5000_u64), (660, 330, 20_000)] {
            #[allow(clippy::cast_possible_truncation)]
            let img = RgbImage::from_fn(w, h, |x, y| {
                let g = (u64::from(y) * 255 / u64::from(h - 1)) as u8;
                if x < w / 2 {
                    image::Rgb([255, g, 0])
                } else {
                    image::Rgb([0, g, 255])
                }
            });
            let out = downscale_rgb(img, cap, ImageOwner::default());
            let (nw, nh) = out.dimensions();
            assert_eq!(Some((nw, nh)), downscaled_dims(w, h, cap));
            let near = |got: u8, want: u8, what: &str| {
                assert!(
                    got.abs_diff(want) <= 24,
                    "{w}x{h} -> {nw}x{nh}: {what} = {got}, want ~{want}"
                );
            };
            let px = |x: u32, y: u32| out.get_pixel(x, y).0;
            // Corners: colour of the half, gradient at its ends.
            for (x, y, r, g, b) in [
                (0, 0, 255, 0, 0),
                (0, nh - 1, 255, 255, 0),
                (nw - 1, 0, 0, 0, 255),
                (nw - 1, nh - 1, 0, 255, 255),
            ] {
                let p = px(x, y);
                near(p[0], r, &format!("red at ({x},{y})"));
                near(p[1], g, &format!("green at ({x},{y})"));
                near(p[2], b, &format!("blue at ({x},{y})"));
            }
            // Just either side of the split (two output pixels away, clear of the bicubic
            // ringing): still the half's colour; the middle row is mid-gradient.
            let mid = nh / 2;
            let left = px(nw / 2 - 2, mid);
            let right = px(nw / 2 + 1, mid);
            near(left[0], 255, "red left of the split");
            near(left[2], 0, "blue left of the split");
            near(right[0], 0, "red right of the split");
            near(right[2], 255, "blue right of the split");
            near(left[1], 128, "gradient at the middle row");
            near(right[1], 128, "gradient at the middle row");
        }
    }
}
