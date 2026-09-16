// ============================================================
// src/ov_vlm.rs — Safe Rust wrapper around the VLM C-API
// ============================================================
// Wraps `ov_vlm_create` / `ov_vlm_generate` / `ov_vlm_free` from
// `ov_bridge/ov_bridge.cpp` in a safe Rust API.
//
// THREADING:
//   VLMPipeline is single-threaded by contract (same as CB).
//   OvVlmEngine is !Sync (never shared by reference); it is moved
//   onto a dedicated OS thread at spawn time. Send is implemented
//   manually because we explicitly own the raw pointer and never
//   alias it from another thread.
// ============================================================

use std::ffi::{CString, c_void};
use std::ptr;

use anyhow::Context;

use crate::image_util::DecodedImage;
use crate::ov_cb::GenParams;
use crate::ov_pipeline::last_error;

// ── OvGenParamsC (private FFI mirror — must match OvGenParams in ov_bridge.cpp) ──
//
// Duplicated here (vs. importing from ov_cb) because the CB struct is private
// to that module. Both are `#[repr(C)]` and have identical layouts.
#[repr(C)]
struct OvGenParamsC {
    max_new_tokens: usize,
    temperature: f32,
    top_p: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    repetition_penalty: f32,
    use_rng_seed: std::ffi::c_int,
    rng_seed: u64,
    stop_strings: *const *const std::ffi::c_char,
    stop_count: usize,
    /// G1: NUL-terminated JSON-Schema string, or NULL for free-form output.
    json_schema: *const std::ffi::c_char,
    /// Top-k sampling; `0` = not set (engine default). APPENDED —
    /// never insert mid-struct: the size-only drift guard cannot catch
    /// same-size field reordering.
    top_k: usize,
    /// `0` = plain decoding. VLM requests never set this (always `0`), but
    /// the struct is shared with the C++ side and must stay byte-identical.
    /// APPENDED (speculative-decoding plan) — never insert mid-struct: the
    /// size-only drift guard cannot catch same-size field reordering.
    num_assistant_tokens: usize,
}

// ── VLM token callback (same ABI as OvTokenCallback in ov_bridge.cpp) ───────
//
// Called once per decoded text fragment. Return 0 to continue, 1 to stop.
#[allow(non_camel_case_types)]
type OvVlmTokenCallback =
    unsafe extern "C" fn(token: *const u8, len: usize, user_data: *mut c_void) -> std::ffi::c_int;

// ── FFI declarations ─────────────────────────────────────────────────────────

unsafe extern "C" {
    fn ov_vlm_create(
        model_path: *const std::ffi::c_char,
        device: *const std::ffi::c_char,
        ov_cache_dir: *const std::ffi::c_char,
        cache_size_gb: f64,
        enable_prefix_caching: bool,
    ) -> *mut c_void;

    fn ov_vlm_free(handle: *mut c_void);

    #[allow(clippy::too_many_arguments)]
    fn ov_vlm_generate(
        handle: *mut c_void,
        messages_json: *const std::ffi::c_char,
        tools_json: *const std::ffi::c_char,
        extra_context_json: *const std::ffi::c_char,
        images: *const *const u8,
        heights: *const u32,
        widths: *const u32,
        num_images: u32,
        params: *const OvGenParamsC,
        callback: Option<OvVlmTokenCallback>,
        user_data: *mut c_void,
        finish_code_out: *mut std::ffi::c_int,
        input_tokens_out: *mut usize,
        // Diagnostic-only (2026-08-21 instant-EOS investigation, Part 5 —
        // see `generate`'s doc comment): the pipeline's own
        // `perf_metrics`-reported generated-token count and mean inference
        // duration, independent of the streamer callback's own tally.
        generated_tokens_out: *mut usize,
        inference_ms_out: *mut f64,
    ) -> std::ffi::c_int;

    fn ov_vlm_count_tokens(
        handle: *mut c_void,
        text: *const u8,
        len: usize,
        out_count: *mut usize,
    ) -> std::ffi::c_int;

    /// `sizeof(OvGenParams)` on the C++ side — see the size-check test below.
    #[cfg(test)]
    fn ov_gen_params_size() -> usize;
}

// Re-export from ov_cb so VLM callers don't need to import from both modules.
pub use crate::ov_cb::FinishReason;

// ── OvVlmEngine ──────────────────────────────────────────────────────────────

/// A loaded `VLMPipeline` — the Phase-5 vision-language engine.
///
/// Lives on a single dedicated OS thread. All calls to `generate` must be from
/// that same thread. `drop` frees the pipeline and its GPU memory synchronously.
pub struct OvVlmEngine {
    handle: *mut c_void,
}

// SAFETY: OvVlmEngine owns the VLMPipeline exclusively. The raw pointer is
// moved onto the engine thread at spawn time and never shared or aliased.
// We do NOT implement Sync — the engine must never be shared by reference.
unsafe impl Send for OvVlmEngine {}

/// Serialise an optional JSON argument to a `CString` for the FFI, or `None`
/// to pass NULL.
///
/// Generic over the payload so the `tools` slice and the `extra_context`
/// object share one implementation.
///
/// Shared by `tools` and `extra_context`, which have identical contracts: both
/// are optional per-request JSON blobs attached to the same `ChatHistory`, and
/// both must outlive the call. Returns `None` on a serialisation failure or an
/// interior NUL rather than propagating — an unsendable optional argument
/// degrades to "not supplied", which is the same as today's behaviour when the
/// caller passes nothing.
fn json_arg_cstring<T: serde::Serialize + ?Sized>(value: Option<&T>) -> Option<CString> {
    let json = serde_json::to_string(value?).ok()?;
    CString::new(json).ok()
}

impl OvVlmEngine {
    /// Load the VLM model at `model_path` on `device`.
    ///
    /// `ov_cache_dir`: directory for the `OpenVINO` GPU blob cache. `""` disables
    /// caching. Non-empty: first load writes a compiled blob; subsequent loads
    /// read it (~10–30 s) instead of recompiling from IR (~1–3 min per sub-model).
    ///
    /// `cache_size_gb`: KV cache budget in whole GB, applied via `OpenVINO`'s
    /// `scheduler_config` property — same semantics as [`crate::ov_cb::OvCbEngine`]'s
    /// parameter of the same name. `0.0` keeps `OpenVINO`'s own default (dynamic/
    /// unbounded), which measured ~2x this value in real VRAM at 75-100K context
    /// depth on `qwen3.5-4b-int8-ov` (dev/autotest/
    /// 20260820_nudge_fix_verification_qwen3.5-4b-int8-ov.md) — always pass a
    /// real budget in production. Hybrid attention architectures reserve a
    /// fixed floor for their linear-attention layers' state regardless of
    /// context length; too small a value fails construction outright rather
    /// than silently growing unbounded.
    ///
    /// `enable_prefix_caching`: same semantics as `OvCbEngine`'s parameter of
    /// the same name; only applied when `cache_size_gb > 0.0`. Measured
    /// critical, not optional, for this server's chat pattern (the client
    /// resends the full growing conversation every turn): with it off, every
    /// turn reprocesses the entire prior history from scratch — ~16s/turn at
    /// 50K token depth; with it on, ~0.4s/turn from the second turn onward
    /// (the project's internal engineering log).
    ///
    /// # Errors
    /// Returns an error if the model dir is invalid, the device is unavailable,
    /// `cache_size_gb` is too small for the model's fixed reservations, or
    /// pipeline construction otherwise fails (OOM, unsupported hardware, etc.).
    ///
    /// # Performance
    /// Loading takes 30–120 s (GPU JIT for all sub-models). Call from
    /// `spawn_blocking` or the dedicated engine thread — never the async runtime.
    pub fn new(
        model_path: &str,
        device: &str,
        ov_cache_dir: &str,
        cache_size_gb: f64,
        enable_prefix_caching: bool,
    ) -> anyhow::Result<Self> {
        let c_path = CString::new(model_path).context("model_path contains interior NUL")?;
        let c_device = CString::new(device).context("device contains interior NUL")?;
        let c_cache_dir =
            CString::new(ov_cache_dir).context("ov_cache_dir contains interior NUL")?;

        // SAFETY: CStrings outlive the call; ov_vlm_create returns a valid
        // heap-allocated OvVlmState or NULL on failure.
        let handle = unsafe {
            ov_vlm_create(
                c_path.as_ptr(),
                c_device.as_ptr(),
                c_cache_dir.as_ptr(),
                cache_size_gb,
                enable_prefix_caching,
            )
        };

        if handle.is_null() {
            anyhow::bail!("ov_vlm_create failed: {}", last_error());
        }

        tracing::info!(model = %model_path, device = %device, "VLM engine loaded");
        Ok(Self { handle })
    }

    /// Generate text from a chat history + decoded images.
    ///
    /// `messages` is the full conversation as JSON values (`role`/`content`,
    /// plus `tool_calls`/`tool_call_id`/`name` on any message that has them —
    /// see `handlers::chat::extract_vlm_messages`). Image-URL content parts
    /// are stripped to text before calling; pixel data is in `images`
    /// (decoded by [`crate::image_util::decode_data_uri`]).
    ///
    /// `tools` is the `OpenAI` `tools` array, when the request has any active
    /// (`None`/empty otherwise). Handed to `ChatHistory::set_tools()` on the
    /// C++ side — `VLMPipeline` renders its own chat template internally
    /// (there is no VLM equivalent of the text path's `build_prompt`), so
    /// this is the only way to get a `{% if tools %}` branch to fire for a
    /// vision model.
    ///
    /// `on_token` receives each decoded text fragment; returning `true` stops
    /// generation early (use this when the client disconnects).
    ///
    /// # Errors
    /// Returns an error if JSON serialisation or the C++ pipeline call fails.
    ///
    /// # Threading
    /// Blocks the calling thread for the full generation. Call only from the
    /// dedicated VLM engine thread — same contract as `OvCbEngine::step`.
    /// Returns `(finish_reason, prompt_token_count, generated_tokens,
    /// inference_ms)`. `prompt_token_count` comes from `VLMPipeline`'s
    /// `PerfMetrics::num_input_tokens` after generation; it is 0 if metrics
    /// are unavailable for this `OpenVINO` `GenAI` version. `generated_tokens`/
    /// `inference_ms` are diagnostic-only (2026-08-21 instant-EOS
    /// investigation, the project's internal engineering log
    /// Part 5): the pipeline's own `perf_metrics` count/timing, independent
    /// of `on_token`'s own callback tally — the streamer never receives the
    /// EOS token itself, so `generated_tokens == 1` with `on_token` called
    /// zero times distinguishes "sampled EOS immediately" (real work
    /// happened) from "never actually ran" (would read `0`), which the
    /// caller can't tell apart from callback count alone.
    pub fn generate(
        &self,
        messages: &[serde_json::Value],
        images: &[DecodedImage],
        tools: Option<&[serde_json::Value]>,
        extra_context: Option<&serde_json::Value>,
        params: &GenParams,
        mut on_token: impl FnMut(&str) -> bool,
    ) -> anyhow::Result<(FinishReason, usize, usize, f64)> {
        let messages_json =
            serde_json::to_string(messages).context("failed to serialise messages to JSON")?;
        let c_messages = CString::new(messages_json).context("messages JSON contains NUL byte")?;

        // Tools and extra template context share one NULL-when-absent
        // contract; both CStrings must outlive the FFI call, so they are kept
        // as locals on this frame.
        let tools_cstring = json_arg_cstring(tools.filter(|t| !t.is_empty()));
        let tools_ptr = tools_cstring.as_ref().map_or(ptr::null(), |cs| cs.as_ptr());
        let extra_context_cstring = json_arg_cstring(extra_context);
        let extra_context_ptr = extra_context_cstring
            .as_ref()
            .map_or(ptr::null(), |cs| cs.as_ptr());

        // Stop strings — CStrings must outlive the FFI call.
        let stop_cstrings: Vec<CString> = params
            .stop
            .iter()
            .filter_map(|s| CString::new(s.as_str()).ok())
            .collect();
        let stop_ptrs: Vec<*const std::ffi::c_char> = stop_cstrings
            .iter()
            .map(|cs| cs.as_c_str().as_ptr())
            .collect();

        // G1 structured output: keep the schema CString alive on this frame so
        // its pointer is valid for the whole FFI call. NULL = free-form.
        let schema_cstring: Option<CString> = params
            .json_schema
            .as_deref()
            .and_then(|s| CString::new(s).ok());
        let schema_ptr = schema_cstring
            .as_ref()
            .map_or(ptr::null(), |cs| cs.as_ptr());

        let c_params = OvGenParamsC {
            max_new_tokens: params.max_new_tokens,
            temperature: params.temperature.unwrap_or(f32::NAN),
            top_p: params.top_p.unwrap_or(f32::NAN),
            presence_penalty: params.presence_penalty.unwrap_or(f32::NAN),
            frequency_penalty: params.frequency_penalty.unwrap_or(f32::NAN),
            repetition_penalty: params.repetition_penalty.unwrap_or(f32::NAN),
            use_rng_seed: std::ffi::c_int::from(params.seed.is_some()),
            rng_seed: params.seed.unwrap_or(0),
            stop_strings: if stop_ptrs.is_empty() {
                ptr::null()
            } else {
                stop_ptrs.as_ptr()
            },
            stop_count: stop_ptrs.len(),
            json_schema: schema_ptr,
            top_k: params.top_k.unwrap_or(0),
            num_assistant_tokens: 0,
        };

        // Image arrays (zero-copy: Rust owns the pixel buffers).
        let image_ptrs: Vec<*const u8> = images.iter().map(|img| img.data.as_ptr()).collect();
        let heights: Vec<u32> = images.iter().map(|img| img.height).collect();
        let widths: Vec<u32> = images.iter().map(|img| img.width).collect();

        // Pass `on_token` into C via a context struct (same pattern as step()
        // in ov_cb.rs). The context lives on this stack frame for the entire
        // blocking call; `panicked` records a panic caught by the trampoline's
        // firewall so it can be surfaced as an error after the call returns.
        let mut ctx = VlmCbCtx {
            on_token: &mut on_token,
            panicked: false,
        };
        let user_data = std::ptr::addr_of_mut!(ctx).cast::<c_void>();

        let mut finish_code: std::ffi::c_int = 1; // default: STOP
        let mut input_tokens: usize = 0;
        let mut generated_tokens: usize = 0;
        let mut inference_ms: f64 = 0.0;

        // SAFETY: handle is non-null (invariant); all CStrings and slices are alive
        // for the duration of the call; the trampoline matches OvVlmTokenCallback;
        // user_data points to a live fat pointer for the call's duration.
        let num_images = u32::try_from(images.len()).unwrap_or(u32::MAX);
        let ret = unsafe {
            ov_vlm_generate(
                self.handle,
                c_messages.as_ptr(),
                tools_ptr,
                extra_context_ptr,
                if image_ptrs.is_empty() {
                    ptr::null()
                } else {
                    image_ptrs.as_ptr()
                },
                if heights.is_empty() {
                    ptr::null()
                } else {
                    heights.as_ptr()
                },
                if widths.is_empty() {
                    ptr::null()
                } else {
                    widths.as_ptr()
                },
                num_images,
                std::ptr::addr_of!(c_params),
                Some(token_trampoline),
                user_data,
                std::ptr::addr_of_mut!(finish_code),
                std::ptr::addr_of_mut!(input_tokens),
                std::ptr::addr_of_mut!(generated_tokens),
                std::ptr::addr_of_mut!(inference_ms),
            )
        };

        if ret != 0 {
            anyhow::bail!("ov_vlm_generate failed: {}", last_error());
        }
        if ctx.panicked {
            anyhow::bail!("token callback panicked during VLM generation");
        }

        let reason = match finish_code {
            2 => FinishReason::Length,
            _ => FinishReason::Stop,
        };
        Ok((reason, input_tokens, generated_tokens, inference_ms))
    }

    /// Count the tokens `text` encodes to with the VLM's tokenizer.
    ///
    /// Used by the L0 prompt-length gate (`chat.rs`) before submitting a VLM
    /// generation — without this, an oversized prompt reaches
    /// `VLMPipeline::generate` directly and can OOM the GPU (`CL_OUT_OF_RESOURCES`),
    /// which poisons the `OpenCL` context for the whole process, not just the one
    /// request (confirmed live: a ~75k-token prompt took down every GPU-backed
    /// model until a process restart).
    ///
    /// # Errors
    /// Returns an error if the tokenizer fails.
    ///
    /// # Threading
    /// The tokenizer wraps a non-thread-safe `InferRequest`; call only from the
    /// dedicated VLM engine thread — same contract as [`generate`](Self::generate).
    pub fn count_tokens(&self, text: &str) -> anyhow::Result<usize> {
        let mut count: usize = 0;
        // SAFETY: handle is non-null (invariant); `text` is a valid `len`-byte
        // buffer alive for the call; `out_count` points to a live usize.
        let ret = unsafe {
            ov_vlm_count_tokens(
                self.handle,
                text.as_ptr(),
                text.len(),
                std::ptr::addr_of_mut!(count),
            )
        };
        if ret != 0 {
            anyhow::bail!("ov_vlm_count_tokens failed: {}", last_error());
        }
        Ok(count)
    }
}

/// Everything [`token_trampoline`] needs, passed through `user_data` for one
/// `ov_vlm_generate` call.
struct VlmCbCtx<'a> {
    /// The per-token closure supplied to [`OvVlmEngine::generate`].
    on_token: &'a mut dyn FnMut(&str) -> bool,
    /// Set when the closure panicked — the trampoline stops generation and
    /// `generate` turns the flag into an error after the FFI call returns.
    panicked: bool,
}

/// C callback trampoline: reconstructs the context from `user_data` and runs
/// the Rust closure under a panic firewall.
///
/// # Safety
/// Called by C inside `ov_vlm_generate`. Invariants upheld by `OvVlmEngine::generate`:
/// - `user_data` is a valid `*mut VlmCbCtx` on the caller's stack, alive for
///   the whole blocking call.
/// - `token` points to `len` bytes of UTF-8; may be empty (`len == 0`).
unsafe extern "C" fn token_trampoline(
    token: *const u8,
    len: usize,
    user_data: *mut c_void,
) -> std::ffi::c_int {
    // SAFETY: user_data is the context pointer set up in OvVlmEngine::generate.
    let ctx = unsafe { &mut *user_data.cast::<VlmCbCtx>() };
    if ctx.panicked {
        return 1; // already poisoned — keep telling C to stop
    }

    let text = if len == 0 {
        ""
    } else {
        // SAFETY: token points to `len` bytes from the C++ string, alive until
        // this callback returns; the bridge guarantees UTF-8 boundary safety.
        let bytes = unsafe { std::slice::from_raw_parts(token, len) };
        std::str::from_utf8(bytes).unwrap_or("")
    };

    // PANIC FIREWALL: a panic must never unwind out of this extern "C" frame
    // into the C++ caller — that is a guaranteed process abort. AssertUnwindSafe
    // is sound: on panic we stop generation and `generate` reports an error, so
    // any half-mutated closure state is never reused.
    // NOTE the convention here is the inverse of ov_pipeline's: this closure
    // returns `true` to STOP (see `OvVlmEngine::generate` docs), so the bool
    // maps to the C stop code directly, without negation.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (ctx.on_token)(text)));
    match result {
        Ok(stop) => std::ffi::c_int::from(stop), // 0=continue, 1=stop
        Err(payload) => {
            tracing::error!(
                panic = crate::ov_pipeline::panic_message(payload.as_ref()),
                "VLM token callback panicked — stopping generation"
            );
            ctx.panicked = true;
            1 // stop generation
        }
    }
}

impl Drop for OvVlmEngine {
    /// Synchronously frees the VLM pipeline and all its GPU allocations.
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle is the unique owner; this is the only free site.
            unsafe { ov_vlm_free(self.handle) };
            self.handle = ptr::null_mut();
            tracing::debug!("VLM engine freed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{OvGenParamsC, ov_gen_params_size};

    /// Regression test for a real near-miss (2026-07-14): adding
    /// `repetition_penalty` to `OvGenParams` (C++) without updating BOTH of
    /// its private Rust `#[repr(C)]` mirrors (`ov_cb.rs`, `ov_vlm.rs`) silently
    /// misaligns every field after the missed one — no compile error, no
    /// panic, just garbage values crossing the FFI boundary (this module's
    /// copy was the one actually missed). This test would have caught it
    /// immediately: `size_of::<OvGenParamsC>()` must equal `sizeof(OvGenParams)`
    /// on the C++ side. See `ov_cb.rs` for the first copy's identical test.
    #[test]
    fn ov_gen_params_c_matches_bridge_size() {
        assert_eq!(
            std::mem::size_of::<OvGenParamsC>(),
            unsafe { ov_gen_params_size() },
            "OvGenParamsC (ov_vlm.rs) has drifted from OvGenParams (ov_bridge.cpp) — \
             update both #[repr(C)] Rust mirrors when changing the C++ struct"
        );
    }
}
