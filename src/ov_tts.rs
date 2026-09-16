// ============================================================
// src/ov_tts.rs — Safe Rust wrapper around the TTS C-API
// ============================================================
// Wraps `ov_tts_create` / `ov_tts_generate` / `ov_tts_free` from
// `ov_bridge/ov_bridge.cpp` in a safe Rust API (Phase 5.2a).
//
// THREADING:
//   Text2SpeechPipeline is single-threaded by contract (same as Whisper).
//   OvTtsEngine is !Sync; it is moved onto a dedicated OS thread at spawn
//   time. Send is implemented manually because we own the raw pointer and
//   never alias it from another thread.
//
// RESULTS:
//   The bridge delivers each waveform tensor via `samples_cb` — one call per
//   entry in Text2SpeechDecodedResults.speeches. For a single-string input
//   there is exactly one call (index 0). Samples are mono f32 at 16 kHz.
// ============================================================

use std::ffi::{CString, c_void};
use std::ptr;

use anyhow::Context;

use crate::ov_pipeline::last_error;

// ── FFI callback typedef (matches ov_bridge.cpp) ─────────────────────────────

/// Delivers one waveform: `data` is `num_samples` f32 at 16 kHz. `index` is
/// the position in `Text2SpeechDecodedResults.speeches` (0 for single-string).
#[allow(non_camel_case_types)]
type OvTtsSamplesCallback = unsafe extern "C" fn(
    user_data: *mut c_void,
    index: usize,
    data: *const f32,
    num_samples: usize,
);

// ── FFI declarations ─────────────────────────────────────────────────────────

unsafe extern "C" {
    fn ov_tts_create(
        model_path: *const std::ffi::c_char,
        device: *const std::ffi::c_char,
        ov_cache_dir: *const std::ffi::c_char,
    ) -> *mut c_void;

    fn ov_tts_free(handle: *mut c_void);

    fn ov_tts_generate(
        handle: *mut c_void,
        text: *const std::ffi::c_char,
        samples_cb: Option<OvTtsSamplesCallback>,
        user_data: *mut c_void,
    ) -> std::ffi::c_int;
}

// ── OvTtsEngine ──────────────────────────────────────────────────────────────

/// A loaded `Text2SpeechPipeline` — the Phase-5 text-to-speech engine.
///
/// Lives on a single dedicated OS thread; all calls to `synthesize` must be
/// from that thread. `drop` frees the pipeline and its device memory
/// synchronously.
pub struct OvTtsEngine {
    handle: *mut c_void,
}

// SAFETY: OvTtsEngine owns the Text2SpeechPipeline exclusively. The raw
// pointer is moved onto the engine thread at spawn time and never shared or
// aliased. We do NOT implement Sync — the engine must never be shared by ref.
unsafe impl Send for OvTtsEngine {}

impl OvTtsEngine {
    /// Load the TTS model at `model_path` on `device`.
    ///
    /// `ov_cache_dir`: directory for the `OpenVINO` GPU blob cache (empty string disables
    /// it). Non-empty: first load writes a compiled blob; later loads read it.
    ///
    /// # Errors
    /// Returns an error if the model dir is invalid, the device is unavailable,
    /// or pipeline construction fails (OOM, unsupported hardware, etc.).
    pub fn new(model_path: &str, device: &str, ov_cache_dir: &str) -> anyhow::Result<Self> {
        let c_path = CString::new(model_path).context("model_path contains interior NUL")?;
        let c_device = CString::new(device).context("device contains interior NUL")?;
        let c_cache = CString::new(ov_cache_dir).context("ov_cache_dir contains interior NUL")?;

        // SAFETY: the CStrings outlive the call; ov_tts_create returns a
        // valid heap-allocated OvTtsState or NULL on failure.
        let handle = unsafe { ov_tts_create(c_path.as_ptr(), c_device.as_ptr(), c_cache.as_ptr()) };

        if handle.is_null() {
            anyhow::bail!("ov_tts_create failed: {}", last_error());
        }

        tracing::info!(model = %model_path, device = %device, "TTS engine loaded");
        Ok(Self { handle })
    }

    /// Synthesise speech for `text`. Returns 16 kHz mono f32 PCM samples.
    ///
    /// # Errors
    /// Returns an error if the C++ pipeline call fails or the callback panics.
    ///
    /// # Threading
    /// Blocks the calling thread for the full synthesis. Call only from the
    /// dedicated TTS engine thread.
    pub fn synthesize(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let c_text = CString::new(text).context("text contains interior NUL")?;

        let mut ctx = TtsCbCtx {
            samples: Vec::new(),
            panicked: false,
        };
        let user_data = std::ptr::addr_of_mut!(ctx).cast::<c_void>();

        // SAFETY: handle is non-null (invariant); c_text is alive for the call;
        // the trampoline matches the callback ABI; user_data points to live ctx.
        let ret = unsafe {
            ov_tts_generate(
                self.handle,
                c_text.as_ptr(),
                Some(samples_trampoline),
                user_data,
            )
        };

        if ret != 0 {
            anyhow::bail!("ov_tts_generate failed: {}", last_error());
        }
        if ctx.panicked {
            anyhow::bail!("TTS callback panicked during synthesis");
        }

        Ok(ctx.samples)
    }
}

impl Drop for OvTtsEngine {
    /// Synchronously frees the TTS pipeline and its device allocations.
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle is the unique owner; this is the only free site.
            unsafe { ov_tts_free(self.handle) };
            self.handle = ptr::null_mut();
            tracing::debug!("TTS engine freed");
        }
    }
}

/// Everything the trampoline needs, passed through `user_data` for one
/// `ov_tts_generate` call.
struct TtsCbCtx {
    /// Accumulates samples from `samples_cb` (index 0 for single-string input).
    samples: Vec<f32>,
    /// Set if the callback panicked; `synthesize` turns it into an error.
    panicked: bool,
}

/// C callback trampoline for one waveform tensor. Panic-firewalled.
///
/// # Safety
/// Called by C inside `ov_tts_generate`. `user_data` is a live `*mut TtsCbCtx`;
/// `data`..`data+num_samples` is a valid f32 range owned by the bridge for the
/// duration of the callback.
unsafe extern "C" fn samples_trampoline(
    user_data: *mut c_void,
    _index: usize,
    data: *const f32,
    num_samples: usize,
) {
    // SAFETY: user_data is the context pointer set up in synthesize().
    let ctx = unsafe { &mut *user_data.cast::<TtsCbCtx>() };
    if ctx.panicked {
        return;
    }
    // PANIC FIREWALL: a panic must never unwind across this extern "C" frame.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if !data.is_null() && num_samples > 0 {
            // SAFETY: invariant from the bridge — data is a live ov::Tensor buffer.
            let slice = unsafe { std::slice::from_raw_parts(data, num_samples) };
            ctx.samples.extend_from_slice(slice);
        }
    }));
    if result.is_err() {
        ctx.panicked = true;
    }
}

// ============================================================
// Tests
// ============================================================
#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]
    use super::*;

    /// End-to-end FFI smoke test: load a real `SpeechT5` model and synthesise a
    /// short phrase. Requires a real device + model, so it is `#[ignore]`d.
    ///
    /// ```text
    /// RV_TTS_MODEL=/path/to/speecht5-tts-ov \
    ///   cargo test --release tts_smoke -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a real device + TTS model (set RV_TTS_MODEL)"]
    fn tts_smoke_synthesises_hello() {
        let Ok(model) = std::env::var("RV_TTS_MODEL") else {
            eprintln!("RV_TTS_MODEL unset — skipping");
            return;
        };
        let engine = OvTtsEngine::new(&model, "CPU", "").expect("load tts");
        let samples = engine.synthesize("Hello world.").expect("synthesize");
        eprintln!("samples={}", samples.len());
        assert!(!samples.is_empty(), "must produce samples");
        assert!(samples.iter().all(|s| s.is_finite()), "all samples finite");
    }
}
