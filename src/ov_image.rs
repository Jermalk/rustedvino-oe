// ============================================================
// src/ov_image.rs — Safe Rust wrapper around the Text2Image C-API
// ============================================================
// Wraps `ov_image_create` / `ov_image_generate` / `ov_image_free` from
// `ov_bridge/ov_bridge.cpp` in a safe Rust API (Phase 5.3b).
//
// THREADING:
//   Text2ImagePipeline is single-threaded by contract (same as Whisper/VLM/embed).
//   OvImageEngine is !Sync; it is moved onto a dedicated OS thread at spawn time.
//   Send is implemented manually because we own the raw pointer and never alias
//   it from another thread.
//
// RESULTS:
//   The bridge delivers each generated image via `image_cb` (the same per-index
//   callback pattern as `ov_embed`): one call per image with a pointer to that
//   image's `height * width * 3` NHWC RGB u8 bytes. We copy them into an owned
//   `RgbImage`; PNG encoding happens in `crate::image_util`, not here.
// ============================================================

use std::ffi::{CString, c_void};

use anyhow::Context;

use crate::ov_pipeline::last_error;

// ── FFI callback typedef (matches ov_bridge.cpp) ─────────────────────────────

/// One generated image: `data` points to `height * width * 3` NHWC RGB u8 bytes,
/// valid only for the duration of the call.
#[allow(non_camel_case_types)]
type OvImageCallback = unsafe extern "C" fn(
    user_data: *mut c_void,
    index: usize,
    data: *const u8,
    height: u32,
    width: u32,
);

// ── FFI declarations ─────────────────────────────────────────────────────────

unsafe extern "C" {
    fn ov_image_create(
        model_path: *const std::ffi::c_char,
        device: *const std::ffi::c_char,
        ov_cache_dir: *const std::ffi::c_char,
    ) -> *mut c_void;

    fn ov_image_free(handle: *mut c_void);

    #[allow(clippy::too_many_arguments)]
    fn ov_image_generate(
        handle: *mut c_void,
        prompt: *const std::ffi::c_char,
        negative_prompt: *const std::ffi::c_char,
        width: u32,
        height: u32,
        num_inference_steps: u32,
        num_images: u32,
        use_seed: std::ffi::c_int,
        seed: u64,
        guidance_scale: f32,
        image_cb: Option<OvImageCallback>,
        user_data: *mut c_void,
    ) -> std::ffi::c_int;

    #[allow(clippy::too_many_arguments)]
    fn ov_image_edit(
        handle: *mut c_void,
        prompt: *const std::ffi::c_char,
        negative_prompt: *const std::ffi::c_char,
        init_data: *const u8,
        init_h: u32,
        init_w: u32,
        mask_data: *const u8,
        mask_h: u32,
        mask_w: u32,
        num_inference_steps: u32,
        num_images: u32,
        use_seed: std::ffi::c_int,
        seed: u64,
        guidance_scale: f32,
        strength: f32,
        image_cb: Option<OvImageCallback>,
        user_data: *mut c_void,
    ) -> std::ffi::c_int;
}

// ── Result + options types ────────────────────────────────────────────────────

/// One generated image: flat NHWC RGB u8 pixels (`width × height × 3`, row-major)
/// — the layout `crate::image_util::rgb_to_png` consumes directly.
#[derive(Debug, Clone)]
pub struct RgbImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Raw pixel bytes: `height × width × 3` in RGB order.
    pub data: Vec<u8>,
}

/// Generation parameters for one [`OvImageEngine::generate`] call. Mirrors the
/// subset of `ImageGenerationConfig` the bridge forwards as properties.
#[derive(Debug, Clone)]
pub struct ImageGenOptions {
    /// Negative prompt, or `None` to omit.
    pub negative_prompt: Option<String>,
    /// Output width in pixels (divisible by 8).
    pub width: u32,
    /// Output height in pixels (divisible by 8).
    pub height: u32,
    /// Denoising steps; `0` leaves the pipeline default.
    pub num_inference_steps: u32,
    /// Images to generate this call (`num_images_per_prompt`).
    pub num_images: u32,
    /// RNG seed for reproducibility, or `None` to leave the pipeline default.
    pub seed: Option<u64>,
    /// CFG guidance scale; `<= 0.0` leaves the pipeline default.
    pub guidance_scale: f32,
}

/// Parameters for one [`OvImageEngine::edit`] call (inpaint or img2img). The
/// source image and optional mask are passed separately; output size matches the
/// source (rounded down to the VAE scale factor by the pipeline), so there is no
/// `width`/`height` here.
#[derive(Debug, Clone)]
pub struct ImageEditOptions {
    /// Negative prompt, or `None` to omit.
    pub negative_prompt: Option<String>,
    /// Denoising steps; `0` leaves the pipeline default.
    pub num_inference_steps: u32,
    /// Images to generate this call (`num_images_per_prompt`).
    pub num_images: u32,
    /// RNG seed for reproducibility, or `None` to leave the pipeline default.
    pub seed: Option<u64>,
    /// CFG guidance scale; `<= 0.0` leaves the pipeline default.
    pub guidance_scale: f32,
    /// Denoise strength in `(0, 1]`; `<= 0.0` leaves the pipeline default. Lower
    /// preserves more of the source; `1.0` re-noises the whole frame.
    pub strength: f32,
}

// ── OvImageEngine ──────────────────────────────────────────────────────────────

/// A loaded `Text2ImagePipeline` — the Phase-5 text-to-image engine.
///
/// Lives on a single dedicated OS thread; all calls to `generate` must be from
/// that thread. `drop` frees the pipeline and its GPU memory synchronously.
pub struct OvImageEngine {
    handle: *mut c_void,
}

// SAFETY: OvImageEngine owns the Text2ImagePipeline exclusively. The raw pointer
// is moved onto the engine thread at spawn time and never shared or aliased. We
// do NOT implement Sync — the engine must never be shared by reference.
unsafe impl Send for OvImageEngine {}

impl OvImageEngine {
    /// Load the image model at `model_path` on `device`.
    ///
    /// `ov_cache_dir`: directory for the `OpenVINO` GPU blob cache (`""` disables
    /// it). Non-empty: first load writes a compiled blob; later loads read it.
    ///
    /// # Errors
    /// Returns an error if the model dir is invalid, the device is unavailable,
    /// or pipeline construction fails (OOM, unsupported hardware, etc.).
    ///
    /// # Performance
    /// Loading JITs every submodel (text encoders, `UNet`, VAE) on the GPU; call
    /// from the dedicated engine thread — never the async runtime.
    pub fn new(model_path: &str, device: &str, ov_cache_dir: &str) -> anyhow::Result<Self> {
        let c_path = CString::new(model_path).context("model_path contains interior NUL")?;
        let c_device = CString::new(device).context("device contains interior NUL")?;
        let c_cache = CString::new(ov_cache_dir).context("ov_cache_dir contains interior NUL")?;

        // SAFETY: the CStrings outlive the call; ov_image_create returns a valid
        // heap-allocated OvImageState or NULL on failure.
        let handle =
            unsafe { ov_image_create(c_path.as_ptr(), c_device.as_ptr(), c_cache.as_ptr()) };

        if handle.is_null() {
            anyhow::bail!("ov_image_create failed: {}", last_error());
        }

        tracing::info!(model = %model_path, device = %device, "image engine loaded");
        Ok(Self { handle })
    }

    /// Generate images from `prompt` using `opts`.
    ///
    /// # Errors
    /// Returns an error if the C++ pipeline call fails or a callback panics.
    ///
    /// # Threading
    /// Blocks the calling thread for the full denoising loop. Call only from the
    /// dedicated image engine thread.
    pub fn generate(&self, prompt: &str, opts: &ImageGenOptions) -> anyhow::Result<Vec<RgbImage>> {
        let c_prompt = CString::new(prompt).context("prompt contains interior NUL")?;
        let c_negative = match opts.negative_prompt.as_deref() {
            Some(np) => Some(CString::new(np).context("negative_prompt contains interior NUL")?),
            None => None,
        };
        let negative_ptr = c_negative.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());

        let mut ctx = ImageCbCtx {
            images: Vec::new(),
            panicked: false,
        };
        let user_data = std::ptr::addr_of_mut!(ctx).cast::<c_void>();

        // SAFETY: handle is non-null (invariant); the CStrings are alive for the
        // call; the trampoline matches the callback ABI; user_data points to a
        // live ctx for the call's duration.
        let ret = unsafe {
            ov_image_generate(
                self.handle,
                c_prompt.as_ptr(),
                negative_ptr,
                opts.width,
                opts.height,
                opts.num_inference_steps,
                opts.num_images,
                std::ffi::c_int::from(opts.seed.is_some()),
                opts.seed.unwrap_or(0),
                opts.guidance_scale,
                Some(image_trampoline),
                user_data,
            )
        };

        if ret != 0 {
            anyhow::bail!("ov_image_generate failed: {}", last_error());
        }
        if ctx.panicked {
            anyhow::bail!("image callback panicked during generation");
        }

        Ok(ctx.images)
    }

    /// Edit `init` toward `prompt`: inpaint when `mask` is `Some`, img2img when
    /// `None`. The mask is RGB where white (255) marks the region to regenerate;
    /// its dims should match `init`. Output size matches `init`.
    ///
    /// # Errors
    /// Returns an error if the model has no `vae_encoder` (edits unsupported), the
    /// C++ pipeline call fails, or a callback panics.
    ///
    /// # Threading
    /// Blocks the calling thread for the full denoising loop. Call only from the
    /// dedicated image engine thread.
    pub fn edit(
        &self,
        prompt: &str,
        init: &RgbImage,
        mask: Option<&RgbImage>,
        opts: &ImageEditOptions,
    ) -> anyhow::Result<Vec<RgbImage>> {
        let c_prompt = CString::new(prompt).context("prompt contains interior NUL")?;
        let c_negative = match opts.negative_prompt.as_deref() {
            Some(np) => Some(CString::new(np).context("negative_prompt contains interior NUL")?),
            None => None,
        };
        let negative_ptr = c_negative.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());

        // Mask pointer/dims, or null/0 for img2img.
        let (mask_ptr, mask_h, mask_w) = match mask {
            Some(m) => (m.data.as_ptr(), m.height, m.width),
            None => (std::ptr::null(), 0, 0),
        };

        let mut ctx = ImageCbCtx {
            images: Vec::new(),
            panicked: false,
        };
        let user_data = std::ptr::addr_of_mut!(ctx).cast::<c_void>();

        // SAFETY: handle is non-null (invariant); the CStrings and the pixel
        // buffers (`init.data`, and `m.data` when present) stay alive across the
        // call; the C++ side wraps them in non-owning tensors and reads them only
        // during generate(); the trampoline matches the callback ABI.
        let ret = unsafe {
            ov_image_edit(
                self.handle,
                c_prompt.as_ptr(),
                negative_ptr,
                init.data.as_ptr(),
                init.height,
                init.width,
                mask_ptr,
                mask_h,
                mask_w,
                opts.num_inference_steps,
                opts.num_images,
                std::ffi::c_int::from(opts.seed.is_some()),
                opts.seed.unwrap_or(0),
                opts.guidance_scale,
                opts.strength,
                Some(image_trampoline),
                user_data,
            )
        };

        if ret != 0 {
            anyhow::bail!("ov_image_edit failed: {}", last_error());
        }
        if ctx.panicked {
            anyhow::bail!("image callback panicked during edit");
        }

        Ok(ctx.images)
    }
}

impl Drop for OvImageEngine {
    /// Synchronously frees the image pipeline and its GPU allocations.
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle is the unique owner; this is the only free site.
            unsafe { ov_image_free(self.handle) };
            self.handle = std::ptr::null_mut();
            tracing::debug!("image engine freed");
        }
    }
}

/// Everything the trampoline needs, passed through `user_data` for one
/// `ov_image_generate` call.
struct ImageCbCtx {
    /// Accumulates generated images from `image_cb`, in batch order.
    images: Vec<RgbImage>,
    /// Set if a callback panicked; `generate` turns it into an error.
    panicked: bool,
}

/// C callback trampoline for one generated image. Runs under a panic firewall.
///
/// # Safety
/// Called by C inside `ov_image_generate`. `user_data` is a live `*mut
/// ImageCbCtx`; `data`..`data + height*width*3` is a live byte range owned by the
/// result tensor for the duration of the call.
unsafe extern "C" fn image_trampoline(
    user_data: *mut c_void,
    _index: usize,
    data: *const u8,
    height: u32,
    width: u32,
) {
    // SAFETY: user_data is the context pointer set up in generate().
    let ctx = unsafe { &mut *user_data.cast::<ImageCbCtx>() };
    if ctx.panicked {
        return;
    }
    // PANIC FIREWALL: a panic must never unwind across this extern "C" frame.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let len = (height as usize) * (width as usize) * 3;
        let pixels = if len == 0 || data.is_null() {
            Vec::new()
        } else {
            // SAFETY: the bridge passes a pointer to `height*width*3` contiguous
            // u8 pixels (NHWC RGB), alive for this call; we copy them out.
            unsafe { std::slice::from_raw_parts(data, len) }.to_vec()
        };
        ctx.images.push(RgbImage {
            width,
            height,
            data: pixels,
        });
    }));
    if result.is_err() {
        ctx.panicked = true;
    }
}

// ============================================================
// Tests — GPU-gated end-to-end smoke (ignored by default)
// ============================================================
#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    /// End-to-end FFI smoke test: load a real SDXL model and render one small
    /// image. Requires a real GPU + model, so it is `#[ignore]`d. Point
    /// `RV_SDXL_MODEL` at a converted `Text2Image` IR dir to run it:
    ///
    /// ```text
    /// RV_SDXL_MODEL=/opt/rustedvino/models/sdxl-fp16-ov \
    ///   cargo test --release sdxl_smoke -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a real GPU + SDXL model (set RV_SDXL_MODEL)"]
    fn sdxl_smoke_renders_one_image() {
        let Ok(model) = std::env::var("RV_SDXL_MODEL") else {
            eprintln!("RV_SDXL_MODEL unset — skipping");
            return;
        };
        let engine = OvImageEngine::new(&model, "GPU", "").expect("load sdxl");
        let opts = ImageGenOptions {
            negative_prompt: None,
            width: 512,
            height: 512,
            num_inference_steps: 4,
            num_images: 1,
            seed: Some(42),
            guidance_scale: 0.0,
        };
        let images = engine.generate("a red circle", &opts).expect("generate");
        assert_eq!(images.len(), 1);
        let img = &images[0];
        eprintln!(
            "image {}x{} ({} bytes)",
            img.width,
            img.height,
            img.data.len()
        );
        assert_eq!(
            img.data.len(),
            (img.width as usize) * (img.height as usize) * 3
        );
    }
}
