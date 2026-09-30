// ============================================================
// src/ov_whisper.rs — Safe Rust wrapper around the Whisper C-API
// ============================================================
// Wraps `ov_whisper_create` / `ov_whisper_generate` / `ov_whisper_free` from
// `ov_bridge/ov_bridge.cpp` in a safe Rust API (Phase 5.1b).
//
// THREADING:
//   WhisperPipeline is single-threaded by contract (same as VLM/embed).
//   OvWhisperEngine is !Sync; it is moved onto a dedicated OS thread at spawn
//   time. Send is implemented manually because we own the raw pointer and never
//   alias it from another thread.
//
// RESULTS:
//   The bridge delivers the transcript via `text_cb` (once) and each
//   timestamped segment via `chunk_cb` (only when timestamps are requested) —
//   the same callback-collection pattern as `ov_embed`. The detected language
//   is written into a small caller-owned buffer.
// ============================================================

use std::ffi::{CStr, CString, c_void};
use std::ptr;

use anyhow::Context;

use crate::ov_pipeline::last_error;

// ── FFI callback typedefs (match ov_bridge.cpp) ──────────────────────────────

/// Full transcript, delivered once. `text` is NOT NUL-terminated — use `len`.
#[allow(non_camel_case_types)]
type OvWhisperTextCallback =
    unsafe extern "C" fn(user_data: *mut c_void, text: *const u8, len: usize);

/// One timestamped segment. `start_ts`/`end_ts` are seconds.
#[allow(non_camel_case_types)]
type OvWhisperChunkCallback = unsafe extern "C" fn(
    user_data: *mut c_void,
    start_ts: f32,
    end_ts: f32,
    text: *const u8,
    len: usize,
);

// ── FFI declarations ─────────────────────────────────────────────────────────

unsafe extern "C" {
    fn ov_whisper_create(
        model_path: *const std::ffi::c_char,
        device: *const std::ffi::c_char,
        ov_cache_dir: *const std::ffi::c_char,
    ) -> *mut c_void;

    fn ov_whisper_free(handle: *mut c_void);

    #[allow(clippy::too_many_arguments)]
    fn ov_whisper_generate(
        handle: *mut c_void,
        samples: *const f32,
        num_samples: usize,
        language: *const std::ffi::c_char,
        return_timestamps: std::ffi::c_int,
        translate: std::ffi::c_int,
        text_cb: Option<OvWhisperTextCallback>,
        chunk_cb: Option<OvWhisperChunkCallback>,
        user_data: *mut c_void,
        language_out: *mut std::ffi::c_char,
        language_cap: usize,
    ) -> std::ffi::c_int;
}

/// Whether the Whisper model in `model_dir` supports the translate task, from
/// its `generation_config.json`: `is_multilingual` and a `translate` entry in
/// `task_to_id`. English-only (`.en`) models have neither.
///
/// `None` when the file is missing or unreadable — unknown, so callers let
/// the request through rather than refuse a model that might work. Note what
/// this can't tell: `whisper-large-v3-turbo` declares translation but was
/// fine-tuned on transcription only, so its translations are weak.
#[must_use]
pub(crate) fn resolve_can_translate(model_dir: &std::path::Path) -> Option<bool> {
    let text = std::fs::read_to_string(model_dir.join("generation_config.json")).ok()?;
    let cfg: serde_json::Value = serde_json::from_str(&text).ok()?;
    let multilingual = cfg
        .get("is_multilingual")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let has_task = cfg
        .get("task_to_id")
        .and_then(|t| t.get("translate"))
        .is_some();
    Some(multilingual && has_task)
}

// ── Result types ─────────────────────────────────────────────────────────────

/// One timestamped transcription segment (seconds).
#[derive(Debug, Clone)]
pub struct WhisperChunk {
    /// Segment start offset, seconds.
    pub start: f32,
    /// Segment end offset, seconds.
    pub end: f32,
    /// Segment transcript text.
    pub text: String,
}

/// A decoded transcription: full text, detected language, optional per-segment
/// timings (populated only when `transcribe` is asked for timestamps).
#[derive(Debug, Clone)]
pub struct WhisperResult {
    /// The full transcript.
    pub text: String,
    /// Detected (or requested) language, as reported by the pipeline.
    pub language: String,
    /// Per-segment timings, empty unless timestamps were requested.
    pub chunks: Vec<WhisperChunk>,
}

// ── OvWhisperEngine ──────────────────────────────────────────────────────────

/// A loaded `WhisperPipeline` — the Phase-5 speech-to-text engine.
///
/// Lives on a single dedicated OS thread; all calls to `transcribe` must be from
/// that thread. `drop` frees the pipeline and its GPU memory synchronously.
pub struct OvWhisperEngine {
    handle: *mut c_void,
}

// SAFETY: OvWhisperEngine owns the WhisperPipeline exclusively. The raw pointer
// is moved onto the engine thread at spawn time and never shared or aliased. We
// do NOT implement Sync — the engine must never be shared by reference.
unsafe impl Send for OvWhisperEngine {}

impl OvWhisperEngine {
    /// Load the Whisper model at `model_path` on `device`.
    ///
    /// `ov_cache_dir`: directory for the `OpenVINO` GPU blob cache (`""` disables
    /// it). Non-empty: first load writes a compiled blob; later loads read it.
    ///
    /// # Errors
    /// Returns an error if the model dir is invalid, the device is unavailable,
    /// or pipeline construction fails (OOM, unsupported hardware, etc.).
    ///
    /// # Performance
    /// Loading JITs the encoder + decoder on the GPU; call from `spawn_blocking`
    /// or the dedicated engine thread — never the async runtime.
    pub fn new(model_path: &str, device: &str, ov_cache_dir: &str) -> anyhow::Result<Self> {
        let c_path = CString::new(model_path).context("model_path contains interior NUL")?;
        let c_device = CString::new(device).context("device contains interior NUL")?;
        let c_cache = CString::new(ov_cache_dir).context("ov_cache_dir contains interior NUL")?;

        // SAFETY: the CStrings outlive the call; ov_whisper_create returns a
        // valid heap-allocated OvWhisperState or NULL on failure.
        let handle =
            unsafe { ov_whisper_create(c_path.as_ptr(), c_device.as_ptr(), c_cache.as_ptr()) };

        if handle.is_null() {
            anyhow::bail!("ov_whisper_create failed: {}", last_error());
        }

        tracing::info!(model = %model_path, device = %device, "Whisper engine loaded");
        Ok(Self { handle })
    }

    /// Transcribe 16 kHz mono PCM `samples` (normalised ~[-1, 1]).
    ///
    /// `language`: a source-language code (`"en"`, `"pl"`, …) or `None` to
    /// autodetect. `timestamps`: when `true`, the result carries per-segment
    /// [`WhisperChunk`]s. `translate`: Whisper's translate task — English
    /// output whatever the spoken language; check [`resolve_can_translate`]
    /// first, it only works on multilingual models.
    ///
    /// # Errors
    /// Returns an error if the C++ pipeline call fails or a callback panics.
    ///
    /// # Threading
    /// Blocks the calling thread for the full transcription. Call only from the
    /// dedicated STT engine thread.
    pub fn transcribe(
        &self,
        samples: &[f32],
        language: Option<&str>,
        timestamps: bool,
        translate: bool,
    ) -> anyhow::Result<WhisperResult> {
        let c_language = match language {
            Some(l) => Some(CString::new(l).context("language contains interior NUL")?),
            None => None,
        };
        let language_ptr = c_language.as_ref().map_or(ptr::null(), |c| c.as_ptr());

        let mut ctx = WhisperCbCtx {
            text: String::new(),
            chunks: Vec::new(),
            panicked: false,
        };
        let user_data = std::ptr::addr_of_mut!(ctx).cast::<c_void>();

        // Detected language, written NUL-terminated by the bridge.
        let mut language_buf: [std::ffi::c_char; 64] = [0; 64];

        // SAFETY: handle is non-null (invariant); `samples` is alive for the call;
        // the trampolines match the callback ABIs; user_data points to a live ctx
        // for the call's duration; language_buf is a valid 64-byte out buffer.
        let ret = unsafe {
            ov_whisper_generate(
                self.handle,
                if samples.is_empty() {
                    ptr::null()
                } else {
                    samples.as_ptr()
                },
                samples.len(),
                language_ptr,
                std::ffi::c_int::from(timestamps),
                std::ffi::c_int::from(translate),
                Some(text_trampoline),
                Some(chunk_trampoline),
                user_data,
                language_buf.as_mut_ptr(),
                language_buf.len(),
            )
        };

        if ret != 0 {
            anyhow::bail!("ov_whisper_generate failed: {}", last_error());
        }
        if ctx.panicked {
            anyhow::bail!("whisper callback panicked during transcription");
        }

        // SAFETY: the bridge NUL-terminates language_buf within its capacity.
        let detected = unsafe { CStr::from_ptr(language_buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        Ok(WhisperResult {
            text: ctx.text,
            language: detected,
            chunks: ctx.chunks,
        })
    }
}

impl Drop for OvWhisperEngine {
    /// Synchronously frees the Whisper pipeline and its GPU allocations.
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle is the unique owner; this is the only free site.
            unsafe { ov_whisper_free(self.handle) };
            self.handle = ptr::null_mut();
            tracing::debug!("Whisper engine freed");
        }
    }
}

/// Everything the trampolines need, passed through `user_data` for one
/// `ov_whisper_generate` call.
struct WhisperCbCtx {
    /// Accumulates the transcript from `text_cb`.
    text: String,
    /// Accumulates timestamped segments from `chunk_cb`.
    chunks: Vec<WhisperChunk>,
    /// Set if a callback panicked; `transcribe` turns it into an error.
    panicked: bool,
}

/// Read `len` UTF-8 bytes at `ptr` into an owned `String` (lossy). Empty when
/// `len == 0` or `ptr` is null.
///
/// # Safety
/// `ptr` must point to `len` valid bytes alive for the call (upheld by the
/// bridge, which passes a live `std::string`'s buffer).
unsafe fn read_text(ptr: *const u8, len: usize) -> String {
    if len == 0 || ptr.is_null() {
        return String::new();
    }
    // SAFETY: caller guarantees `ptr`..`ptr+len` is a live byte range.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    String::from_utf8_lossy(bytes).into_owned()
}

/// C callback trampoline for the full transcript. Runs under a panic firewall.
///
/// # Safety
/// Called by C inside `ov_whisper_generate`. `user_data` is a live `*mut
/// WhisperCbCtx`; `text`..`text+len` is a live UTF-8 byte range.
unsafe extern "C" fn text_trampoline(user_data: *mut c_void, text: *const u8, len: usize) {
    // SAFETY: user_data is the context pointer set up in transcribe().
    let ctx = unsafe { &mut *user_data.cast::<WhisperCbCtx>() };
    if ctx.panicked {
        return;
    }
    // PANIC FIREWALL: a panic must never unwind across this extern "C" frame.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: invariant from the bridge — see fn doc.
        let s = unsafe { read_text(text, len) };
        ctx.text.push_str(&s);
    }));
    if result.is_err() {
        ctx.panicked = true;
    }
}

/// C callback trampoline for one timestamped segment. Panic-firewalled.
///
/// # Safety
/// Same invariants as [`text_trampoline`].
unsafe extern "C" fn chunk_trampoline(
    user_data: *mut c_void,
    start_ts: f32,
    end_ts: f32,
    text: *const u8,
    len: usize,
) {
    // SAFETY: user_data is the context pointer set up in transcribe().
    let ctx = unsafe { &mut *user_data.cast::<WhisperCbCtx>() };
    if ctx.panicked {
        return;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: invariant from the bridge — see fn doc.
        let s = unsafe { read_text(text, len) };
        ctx.chunks.push(WhisperChunk {
            start: start_ts,
            end: end_ts,
            text: s,
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

    /// Real `generation_config.json` shapes: a multilingual model with a
    /// `translate` task can translate; an English-only (.en) model can't; no
    /// file means unknown.
    #[test]
    fn resolve_can_translate_reads_generation_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        let write = |json: &str| {
            std::fs::write(dir.path().join("generation_config.json"), json).expect("write");
        };
        assert_eq!(resolve_can_translate(dir.path()), None, "no file");
        write(
            r#"{"is_multilingual": true, "task_to_id": {"transcribe": 50360, "translate": 50359}}"#,
        );
        assert_eq!(resolve_can_translate(dir.path()), Some(true));
        write(r#"{"is_multilingual": false}"#);
        assert_eq!(
            resolve_can_translate(dir.path()),
            Some(false),
            "English-only"
        );
        write(r#"{"is_multilingual": true, "task_to_id": {"transcribe": 1}}"#);
        assert_eq!(
            resolve_can_translate(dir.path()),
            Some(false),
            "no translate task"
        );
    }

    /// End-to-end FFI smoke test: load a real Whisper model and transcribe a
    /// short silence buffer. Requires a real GPU + model, so it is `#[ignore]`d.
    /// Point `RV_WHISPER_MODEL` at a converted Whisper IR dir to run it:
    ///
    /// ```text
    /// RV_WHISPER_MODEL=/opt/rustedvino/models/whisper-large-v3-int8-ov \
    ///   cargo test --release whisper_smoke -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a real GPU + Whisper model (set RV_WHISPER_MODEL)"]
    fn whisper_smoke_transcribes_silence() {
        let Ok(model) = std::env::var("RV_WHISPER_MODEL") else {
            eprintln!("RV_WHISPER_MODEL unset — skipping");
            return;
        };
        let engine = OvWhisperEngine::new(&model, "GPU", "").expect("load whisper");
        // 1 s of 16 kHz silence.
        let samples = vec![0.0_f32; 16_000];
        let result = engine
            .transcribe(&samples, Some("en"), true, false)
            .expect("transcribe");
        eprintln!(
            "text={:?} language={:?} chunks={}",
            result.text,
            result.language,
            result.chunks.len()
        );
    }
}
