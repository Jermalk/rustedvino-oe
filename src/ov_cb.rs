// ============================================================
// src/ov_cb.rs — Safe Rust wrapper around the CB C-API in ov_bridge.cpp
// ============================================================
//
// This is the Phase-2 engine binding. Where `ov_pipeline.rs` wraps the
// one-shot blocking `LLMPipeline::generate`, this wraps the *granular*
// ContinuousBatchingPipeline API: many requests share one KV pool, advanced
// together one model iteration at a time.
//
// CRASH COURSE — why a different shape than OvPipeline:
//   OvPipeline::generate() blocks for a whole response. The CB engine instead
//   exposes a *step loop*: add_request() enqueues, step() advances ALL active
//   requests by one iteration and hands back whatever tokens landed, and
//   has_unfinished() is the loop condition. The owning engine thread (built in
//   the next step) drives this loop; this module is just the safe FFI surface.
//
// THREADING:
//   ContinuousBatchingPipeline is single-threaded by contract — exactly ONE
//   OS thread ever touches it. OvCbEngine holds a raw pointer (!Send by
//   default); we implement Send manually because we *move ownership* onto that
//   one engine thread and never share the pointer. There is no Mutex: unlike
//   OvPipeline (shared across tasks behind Arc<Mutex>), the CB engine is never
//   shared — tasks talk to it only through a command channel.
// ============================================================

use anyhow::{Context, bail, ensure};
use std::ffi::{CString, c_void};

use crate::ov_pipeline::last_error;

// ── GenParams ──────────────────────────────────────────────────────────────────

/// Sampling parameters forwarded from the HTTP request to the CB engine.
///
/// `None` on any field means "keep the engine default" (no override).
/// `stop` is the list of stop-strings (`OpenAI` `stop` field, string or array).
#[derive(Debug, Clone, Default)]
pub struct GenParams {
    /// Token budget; `0` = model default.
    pub max_new_tokens: usize,
    /// `OpenAI` `temperature`. `None` → engine default. `Some(0.0)` → greedy.
    pub temperature: Option<f32>,
    /// `OpenAI` `top_p`. `None` → engine default.
    pub top_p: Option<f32>,
    /// `OpenAI` `presence_penalty`. `None` → engine default (0.0).
    pub presence_penalty: Option<f32>,
    /// `OpenAI` `frequency_penalty`. `None` → engine default (0.0).
    pub frequency_penalty: Option<f32>,
    /// Repetition penalty (not a standard `OpenAI` field — accepted as a named
    /// extension on `ChatRequest`/`CompletionRequest`, same convention other
    /// `OpenAI`-compatible servers use). Distinct from `presence_penalty`/
    /// `frequency_penalty`: `GenerationConfig::repetition_penalty` scales a
    /// token's logit directly by this factor every time it has already
    /// appeared, `> 1.0` discourages repeats, `1.0` = off. `None` → engine
    /// default (1.0), but callers building this via
    /// [`crate::handlers::chat::gen_params_from_chat`] get a non-`None` safety
    /// net by default — see that function's doc comment for why.
    pub repetition_penalty: Option<f32>,
    /// `OpenAI` `seed`. `None` → non-deterministic (engine chooses).
    pub seed: Option<u64>,
    /// `OpenAI` `stop` — strings that end generation before the EOS token.
    pub stop: Vec<String>,
    /// G1 structured output: a JSON-Schema string. When `Some`, generation is
    /// constrained (via `GenAI`'s xgrammar backend) to produce output matching
    /// this schema. `None` → free-form. Set once per request, never per token.
    pub json_schema: Option<String>,
    /// Top-k sampling (extension field — Ollama/`llama.cpp` convention).
    /// `None` or `Some(0)` → disabled (engine keeps its default, effectively
    /// unlimited).
    pub top_k: Option<usize>,
}

/// C-side representation of [`GenParams`]. Must EXACTLY match `OvGenParams`
/// in `ov_bridge/ov_bridge.cpp` (field order, sizes, alignment).
///
/// `f32::NAN` signals "not set" for all float fields.
/// `use_rng_seed == 1` means `rng_seed` is meaningful.
/// `stop_strings` and `stop_count` are NULL/0 when there are no stop strings.
/// `json_schema` is NULL when no structured-output constraint is requested.
///
/// Not `pub` — used only inside this module to cross the FFI boundary.
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
    /// `0` = plain decoding. `>0` = assisted request (speculative decoding;
    /// requires a pipeline built with a draft model attached). APPENDED
    /// (speculative-decoding plan) — never insert mid-struct: the size-only
    /// drift guard cannot catch same-size field reordering.
    num_assistant_tokens: usize,
}

// ── Raw C declarations ────────────────────────────────────────────────────────
//
// Must exactly match the extern "C" CB surface in ov_bridge/ov_bridge.cpp.
// `*mut c_void` = the opaque `OvCbHandle` on the C side.

/// Per-step callback: invoked once per request that produced output (or
/// finished) during an [`OvCbEngine::step`] call.
///   `request_id`    – the id passed to [`OvCbEngine::add_request`]
///   `delta`/`len`   – new UTF-8 bytes (NOT null-terminated); may be empty
///   `finish_reason` – 0 running, 1 STOP (eos), 2 LENGTH (max tokens)
///   `new_tokens`    – real token ids reported as of this callback (not
///                     derived from `len`/byte counting), accumulated on the
///                     C++ side since this request's callback last fired —
///                     a token generated while its decoded text was held
///                     back (a multi-byte UTF-8 char split across steps) is
///                     still counted exactly once, whenever it's finally
///                     emitted. Plain decoding: normally 1 per callback.
///                     Speculative decoding: can be >1 — one verification
///                     step can accept several draft tokens at once, landing
///                     in the SAME callback. Do not assume 1 callback == 1
///                     token; that assumption silently undercounts
///                     `usage.completion_tokens` under speculative decoding
///                     (found live 2026-07-19, see the project's internal engineering log).
#[allow(non_camel_case_types)]
type OvCbTokenCallback = unsafe extern "C" fn(
    user_data: *mut c_void,
    request_id: u64,
    delta: *const u8,
    len: usize,
    finish_reason: std::ffi::c_int,
    new_tokens: usize,
);

/// One-shot text sink for [`ov_cb_decode`]: invoked exactly once with the full
/// decoded UTF-8 string (`text`/`len`, NOT null-terminated).
#[allow(non_camel_case_types)]
type OvTextCallback = unsafe extern "C" fn(user_data: *mut c_void, text: *const u8, len: usize);

unsafe extern "C" {
    fn ov_cb_create(
        model_path: *const std::ffi::c_char,
        device: *const std::ffi::c_char,
        max_num_seqs: usize,
        cache_size_gb: f64,
        kv_cache_precision: *const std::ffi::c_char,
        ov_cache_dir: *const std::ffi::c_char,
        enable_prefix_caching: bool,
        draft_model_path: *const std::ffi::c_char,
        draft_device: *const std::ffi::c_char,
    ) -> *mut c_void;

    fn ov_cb_free(handle: *mut c_void);

    fn ov_cb_add_request(
        handle: *mut c_void,
        request_id: u64,
        prompt: *const std::ffi::c_char,
        params: *const OvGenParamsC,
    ) -> std::ffi::c_int;

    fn ov_cb_add_request_ids(
        handle: *mut c_void,
        request_id: u64,
        ids: *const i64,
        count: usize,
        params: *const OvGenParamsC,
    ) -> std::ffi::c_int;

    fn ov_cb_drop_request(handle: *mut c_void, request_id: u64);

    fn ov_cb_has_unfinished(handle: *mut c_void) -> std::ffi::c_int;

    fn ov_cb_get_metrics(handle: *mut c_void, out_cache_usage_pct: *mut f64) -> std::ffi::c_int;

    fn ov_cb_step(
        handle: *mut c_void,
        callback: Option<OvCbTokenCallback>,
        user_data: *mut c_void,
    ) -> std::ffi::c_int;

    fn ov_cb_encode(
        handle: *mut c_void,
        text: *const u8,
        len: usize,
        out_ids: *mut i64,
        cap: usize,
        out_count: *mut usize,
    ) -> std::ffi::c_int;

    fn ov_cb_decode(
        handle: *mut c_void,
        ids: *const i64,
        count: usize,
        callback: Option<OvTextCallback>,
        user_data: *mut c_void,
    ) -> std::ffi::c_int;

    fn ov_device_available(device: *const std::ffi::c_char) -> std::ffi::c_int;

    fn ov_list_devices(out: *mut std::ffi::c_char, out_size: usize) -> std::ffi::c_int;

    fn ov_device_property(
        device: *const std::ffi::c_char,
        key: *const std::ffi::c_char,
        out: *mut std::ffi::c_char,
        out_size: usize,
    ) -> std::ffi::c_int;

    fn ov_get_openvino_version(out: *mut std::ffi::c_char, out_size: usize) -> std::ffi::c_int;

    /// `sizeof(OvGenParams)` on the C++ side — see the size-check test below.
    #[cfg(test)]
    fn ov_gen_params_size() -> usize;
}

/// Returns `true` if `device` is listed in `OpenVINO`'s available-devices set.
///
/// Calls `ov::Core::get_available_devices()` via the C bridge. Returns `false`
/// on any exception or if the device name contains an interior NUL byte.
/// Used by the startup assertion in `main.rs` to give a clear early error
/// before any model load attempt.
#[must_use]
pub fn device_available(device: &str) -> bool {
    let Ok(c_device) = CString::new(device) else {
        return false;
    };
    // SAFETY: `ov_device_available` reads `device` as a C string and has no
    // aliasing or lifetime requirements beyond the call. `c_device` outlives it.
    unsafe { ov_device_available(c_device.as_ptr()) != 0 }
}

/// Returns every device listed by `ov::Core::get_available_devices()`.
///
/// Wraps `ov_list_devices` in the C bridge. Returns an empty `Vec` on any
/// bridge error. Used by `--list-devices` in `main.rs` before config load.
#[must_use]
pub fn list_devices() -> Vec<String> {
    let mut buf = vec![0u8; 4096];
    // SAFETY: `buf` is valid for `buf.len()` bytes; `ov_list_devices` writes a
    // NUL-terminated newline-separated string and never writes past `out_size`.
    let ret = unsafe { ov_list_devices(buf.as_mut_ptr().cast::<std::ffi::c_char>(), buf.len()) };
    if ret < 0 {
        return Vec::new();
    }
    let nul_pos = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..nul_pos])
        .split('\n')
        .filter(|s| !s.is_empty())
        .map(std::borrow::ToOwned::to_owned)
        .collect()
}

/// Queries a single `OpenVINO` device property as a string.
///
/// Wraps `ov_device_property` in the C bridge. `key` is an `OpenVINO` property
/// name — e.g. `"FULL_DEVICE_NAME"`, `"DEVICE_TYPE"` (→ `"DISCRETE"` /
/// `"INTEGRATED"` / `"OTHER"`), `"DEVICE_ARCHITECTURE"`, or
/// `"GPU_DEVICE_TOTAL_MEM_SIZE"` (decimal bytes; GPU devices only).
///
/// Returns `None` if the device or property is unsupported, or on any bridge
/// error or interior NUL in the arguments. A property that simply does not
/// apply to a device (e.g. the GPU memory size on the `CPU`) reports `None` —
/// callers treat that as "this device does not expose this fact".
#[must_use]
pub fn device_property(device: &str, key: &str) -> Option<String> {
    let c_device = CString::new(device).ok()?;
    let c_key = CString::new(key).ok()?;
    let mut buf = vec![0u8; 1024];
    // SAFETY: both C strings outlive the call; `buf` is valid for `buf.len()`
    // bytes and `ov_device_property` never writes past `out_size` and always
    // NUL-terminates. No aliasing: the three pointers reference distinct memory.
    let ret = unsafe {
        ov_device_property(
            c_device.as_ptr(),
            c_key.as_ptr(),
            buf.as_mut_ptr().cast::<std::ffi::c_char>(),
            buf.len(),
        )
    };
    if ret < 0 {
        return None;
    }
    let nul_pos = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    Some(String::from_utf8_lossy(&buf[..nul_pos]).into_owned())
}

/// The running `OpenVINO` runtime's build-number string (e.g.
/// `"2026.2.1-19140-c01cd93e24d"`), or `None` on any bridge error.
///
/// Wraps `ov_get_openvino_version` in the C bridge — `ov::get_openvino_version()`
/// is static for the process lifetime (no `Core` instance, `noexcept`), so the
/// first call's result is cached and every later call returns it without
/// crossing the FFI boundary again (`PLAN_image_metadata_response.md` Tier 3:
/// "exposed once at server startup, not per-request").
#[must_use]
pub fn openvino_version() -> Option<String> {
    static VERSION: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    VERSION
        .get_or_init(|| {
            let mut buf = vec![0u8; 256];
            // SAFETY: `buf` is valid for `buf.len()` bytes; `ov_get_openvino_version`
            // never writes past `out_size` and always NUL-terminates.
            let ret = unsafe {
                ov_get_openvino_version(buf.as_mut_ptr().cast::<std::ffi::c_char>(), buf.len())
            };
            if ret < 0 {
                return None;
            }
            let nul_pos = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            let s = String::from_utf8_lossy(&buf[..nul_pos]).into_owned();
            // `ov_bridge.cpp` writes "" (ret == 0, not < 0) when
            // `ov::Version::buildNumber` is null — that's success-shaped at the
            // FFI layer but not a real version string. Treat empty the same as
            // the error path so callers get `None` (omit), never `Some("")`
            // (which would otherwise cache permanently in this OnceLock).
            if s.is_empty() { None } else { Some(s) }
        })
        .clone()
}

// ── FinishReason ───────────────────────────────────────────────────────────────

/// Why a request stopped generating. Maps the C `finish_reason` codes 1/2 to
/// the `OpenAI`-style `finish_reason` strings the chat handler will emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Reached an end-of-sequence token (`GenerationFinishReason::STOP`).
    Stop,
    /// Hit the `max_new_tokens` budget (`GenerationFinishReason::LENGTH`).
    Length,
}

impl FinishReason {
    /// The `OpenAI` `finish_reason` field value for this stop cause.
    #[must_use]
    pub fn as_openai(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

/// Maps the raw C `finish_reason` code to a Rust `Option<FinishReason>`.
/// `0` (still running) → `None`; any unknown nonzero code is treated as `Stop`
/// so the stream is always closed rather than hung. `3` (pool-exhausted,
/// see [`step_trampoline`]) also lands on `Stop` here deliberately — this
/// function only decides whether/how the request gets *finalized*, not
/// whether the client sees a genuine completion; `step_trampoline` computes
/// the separate `pool_exhausted` flag the caller uses for that.
fn finish_from_code(code: std::ffi::c_int) -> Option<FinishReason> {
    match code {
        0 => None,
        2 => Some(FinishReason::Length),
        _ => Some(FinishReason::Stop), // 1 = STOP, and any unexpected code
    }
}

// ── OvCbEngine ───────────────────────────────────────────────────────────────

/// A loaded `ContinuousBatchingPipeline` — the Phase-2 multi-request engine.
///
/// Lives on a single dedicated OS thread (the engine thread). All access is
/// from that thread; never shared behind `Arc`. `drop` frees the KV pool /
/// VRAM synchronously.
pub struct OvCbEngine {
    /// Opaque C++ pointer. Non-null after a successful [`OvCbEngine::new`].
    handle: *mut c_void,
    /// `0` = plain pipeline. `>0` = every request is stamped assisted; only
    /// valid when the pipeline was constructed with a draft model attached.
    num_assistant_tokens: usize,
}

// SAFETY: OvCbEngine owns the ContinuousBatchingPipeline exclusively. We move
// ownership onto the engine thread and never hand out the inner pointer or
// touch it from another thread. Sending ownership across the thread boundary
// once (at spawn) is therefore sound. We do NOT implement Sync — it must never
// be shared by reference between threads.
unsafe impl Send for OvCbEngine {}

impl OvCbEngine {
    /// Load the model and construct the CB pipeline on `device`.
    ///
    /// # Arguments
    /// * `model_path` – directory with `.xml`/`.bin` + tokenizer files
    /// * `device` – `OpenVINO` device string (`"GPU.1"`, `"CPU"`, …)
    /// * `max_num_seqs` – KV scheduler cap; `0` = `OpenVINO` default (256).
    ///   Keep small (e.g. 8) to avoid KV-pool exhaustion stalls.
    /// * `cache_size_gb` – fixed KV pool size in GB; `0.0` = dynamic (stall risk).
    /// * `kv_cache_precision` – `KV_CACHE_PRECISION` property: `""` keeps the plugin
    ///   default (`f16` on GPU), `"u8"` compresses the KV cache (~2× context per GB).
    /// * `ov_cache_dir` – directory for the `OpenVINO` GPU blob cache. `""` disables
    ///   caching. Non-empty: first load writes a compiled blob; subsequent loads
    ///   read the blob (~10–30 s) instead of recompiling from IR (~1–3 min).
    /// * `enable_prefix_caching` – reuse KV blocks across requests sharing an
    ///   identical prompt prefix. Only affects prefill/TTFT, never decode
    ///   tok/s. Retained blocks count against `cache_size_gb`'s pool — do not
    ///   combine with `cache_size_gb == 0.0` (dynamic allocation), where
    ///   retained blocks would grow unbounded and untracked by the VRAM budget.
    /// * `draft_model_path` – directory of a draft model for speculative
    ///   decoding. `""` = no draft (plain decoding, unchanged behaviour).
    /// * `draft_device` – `OpenVINO` device for the draft. `""` = same device
    ///   as `device`. Ignored when `draft_model_path` is empty.
    /// * `num_assistant_tokens` – candidate tokens per verification step for
    ///   every request this engine serves. Must be `0` iff `draft_model_path`
    ///   is empty — a nonzero value with no draft would poison every request.
    ///
    /// # Errors
    /// Returns an error if the model directory is invalid, the device is
    /// unavailable, pipeline construction fails, or `draft_model_path`
    /// emptiness disagrees with `num_assistant_tokens` being zero.
    ///
    /// # Performance
    /// Loading takes several seconds (GPU JIT). Call from `spawn_blocking` or
    /// the dedicated engine thread, never from an async runtime thread.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_path: &str,
        device: &str,
        max_num_seqs: usize,
        cache_size_gb: f64,
        kv_cache_precision: &str,
        ov_cache_dir: &str,
        enable_prefix_caching: bool,
        draft_model_path: &str,
        draft_device: &str,
        num_assistant_tokens: usize,
    ) -> anyhow::Result<Self> {
        ensure!(
            draft_model_path.is_empty() == (num_assistant_tokens == 0),
            "num_assistant_tokens must be 0 iff draft_model_path is empty \
             (got draft_model_path={draft_model_path:?}, num_assistant_tokens={num_assistant_tokens})"
        );

        let c_path = CString::new(model_path).context("model_path contains interior NUL byte")?;
        let c_device = CString::new(device).context("device contains interior NUL byte")?;
        let c_kv_prec = CString::new(kv_cache_precision)
            .context("kv_cache_precision contains interior NUL byte")?;
        let c_cache_dir =
            CString::new(ov_cache_dir).context("ov_cache_dir contains interior NUL byte")?;
        let c_draft_path = CString::new(draft_model_path)
            .context("draft_model_path contains interior NUL byte")?;
        let c_draft_device =
            CString::new(draft_device).context("draft_device contains interior NUL byte")?;

        // SAFETY: all CStrings outlive the call; ov_cb_create returns a valid
        // pointer or NULL.
        let handle = unsafe {
            ov_cb_create(
                c_path.as_ptr(),
                c_device.as_ptr(),
                max_num_seqs,
                cache_size_gb,
                c_kv_prec.as_ptr(),
                c_cache_dir.as_ptr(),
                enable_prefix_caching,
                c_draft_path.as_ptr(),
                c_draft_device.as_ptr(),
            )
        };

        if handle.is_null() {
            bail!("ov_cb_create failed: {}", last_error());
        }

        tracing::info!(model = %model_path, device = %device, "CB engine loaded");
        Ok(Self {
            handle,
            num_assistant_tokens,
        })
    }

    /// Enqueue a generation request. `request_id` must be unique among
    /// currently-live requests. `prompt` is sent raw — apply the chat template
    /// before calling. Sampling behaviour is controlled by `params`.
    ///
    /// # Errors
    /// Returns an error if `prompt` (or any stop string) contains a NUL byte,
    /// or if the pipeline rejects the request.
    pub fn add_request(
        &mut self,
        request_id: u64,
        prompt: &str,
        params: &GenParams,
    ) -> anyhow::Result<()> {
        let c_prompt = CString::new(prompt).context("prompt contains interior NUL byte")?;

        // SAFETY: handle is non-null (invariant); c_prompt and the C-params
        // owned by `with_c_params` both live until the FFI call returns.
        let ret = Self::with_c_params(params, self.num_assistant_tokens, |c_params| unsafe {
            ov_cb_add_request(self.handle, request_id, c_prompt.as_ptr(), c_params)
        });

        if ret != 0 {
            bail!("ov_cb_add_request failed: {}", last_error());
        }
        Ok(())
    }

    /// Enqueue a generation request from PRE-TOKENIZED prompt ids (T7.3).
    ///
    /// The chat/completions L0 length gate already encoded this prompt to check
    /// it against the KV-pool capacity. Passing those ids straight through skips
    /// a second full tokenize pass that the string [`add_request`] would do
    /// inside the pipeline — both encodes use the tokenizer's default
    /// `add_special_tokens`, so the ids are identical and only the redundant
    /// pass is removed. `ids` must be non-empty (the C side rejects an empty
    /// slice) and is copied into an owned tensor on the C side.
    ///
    /// # Errors
    /// Returns an error if `ids` is empty or the pipeline rejects the request.
    pub fn add_request_ids(
        &mut self,
        request_id: u64,
        ids: &[i64],
        params: &GenParams,
    ) -> anyhow::Result<()> {
        // SAFETY: handle is non-null (invariant); `ids` lives for the call (C
        // side copies it into an owned tensor); the C-params owned by
        // `with_c_params` live until the FFI call returns.
        let ret = Self::with_c_params(params, self.num_assistant_tokens, |c_params| unsafe {
            ov_cb_add_request_ids(self.handle, request_id, ids.as_ptr(), ids.len(), c_params)
        });

        if ret != 0 {
            bail!("ov_cb_add_request_ids failed: {}", last_error());
        }
        Ok(())
    }

    /// Marshal [`GenParams`] into the flat C `OvGenParamsC` layout and invoke
    /// `f` with a pointer to it. The CStrings/pointer arrays backing the C
    /// struct live on this function's stack frame, so the pointer is valid for
    /// exactly the duration of `f` — both `add_request` entry points share this
    /// to keep their sampling/stop/schema marshalling byte-identical.
    fn with_c_params<R>(
        params: &GenParams,
        num_assistant_tokens: usize,
        f: impl FnOnce(*const OvGenParamsC) -> R,
    ) -> R {
        // Build CString values for stop strings, then a pointer array.
        // Both vecs must outlive the FFI call (they are on this stack frame).
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
            .map_or(std::ptr::null(), |cs| cs.as_ptr());

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
                std::ptr::null()
            } else {
                stop_ptrs.as_ptr()
            },
            stop_count: stop_ptrs.len(),
            json_schema: schema_ptr,
            top_k: params.top_k.unwrap_or(0),
            num_assistant_tokens,
        };

        f(std::ptr::addr_of!(c_params))
    }

    /// Cancel a live request (e.g. its client disconnected). Idempotent: an
    /// unknown id is a no-op. The pipeline frees that request's KV blocks on
    /// the next [`OvCbEngine::step`].
    pub fn drop_request(&mut self, request_id: u64) {
        // SAFETY: handle is non-null (invariant); unknown ids are a C-side no-op.
        unsafe { ov_cb_drop_request(self.handle, request_id) };
    }

    /// `true` while any request is still generating — the engine-loop condition.
    #[must_use]
    pub fn has_unfinished(&self) -> bool {
        // SAFETY: handle is non-null (invariant).
        unsafe { ov_cb_has_unfinished(self.handle) != 0 }
    }

    /// Live KV-cache occupancy as a percentage (0–100) from the pipeline's last
    /// scheduler step. `None` if the FFI metrics read fails (the gauge then
    /// keeps its prior value rather than reporting a bogus 0).
    ///
    /// Cheap (a cached struct read); called once per step from the engine
    /// thread to publish `rustedvino_kv_cache_usage_percent` (Slice 3a).
    #[must_use]
    pub fn cache_usage_pct(&self) -> Option<f64> {
        let mut out = 0.0_f64;
        // SAFETY: handle is non-null (invariant); `out` is a valid f64 the C
        // side writes only when it returns 0 (success).
        let rc = unsafe { ov_cb_get_metrics(self.handle, &raw mut out) };
        (rc == 0).then_some(out)
    }

    /// Advance every active request by one model iteration, invoking `on_token`
    /// once per request that produced new text or finished this step.
    ///
    /// `on_token(request_id, delta, new_tokens, finish)`:
    /// - `delta` – new UTF-8 text (already UTF-8-boundary-safe; never splits a
    ///   multibyte char). May be empty on the final call.
    /// - `new_tokens` – real token ids reported as of this call, from the
    ///   engine's own count (not derived from `delta`'s byte length) —
    ///   accumulated since this request's callback last fired, so a token
    ///   generated while its text was held back (a multi-byte UTF-8 char
    ///   split across steps) is still counted exactly once. Plain decoding:
    ///   normally `1` per call. Speculative decoding (draft attached): can be
    ///   `>1` — one verification step can accept several draft tokens at
    ///   once, landing in this same callback. Callers computing
    ///   `usage.completion_tokens` must sum this field, not count callbacks —
    ///   counting callbacks silently undercounts under speculative decoding
    ///   (found live 2026-07-19; see the project's internal engineering log).
    /// - `finish` – `None` while running; `Some(reason)` on the last call for
    ///   that request (no further callbacks for that id will follow).
    ///
    /// Returns the ids of requests whose `on_token` invocation **panicked**
    /// this step. The panic is caught at the FFI boundary (it must never
    /// unwind into the C++ frames — guaranteed process abort), the request is
    /// already cancelled in the pipeline, and its closure will never be called
    /// again. The caller must discard its routing state for those ids and
    /// surface an error to their clients.
    ///
    /// # Errors
    /// Returns an error if the underlying model step fails (e.g. device error).
    ///
    /// # Threading
    /// Blocks the calling thread for one model iteration. Call only from the
    /// dedicated engine thread.
    pub fn step(
        &mut self,
        mut on_token: impl FnMut(u64, &str, usize, Option<FinishReason>, bool),
    ) -> anyhow::Result<Vec<u64>> {
        // Pass the step context to C as a pointer in user_data. The trampoline
        // casts it back, runs the closure under a panic firewall, and records
        // any request whose closure panicked. Lifetime is safe: `ctx` lives on
        // this stack frame for the entire ov_cb_step call (which blocks until
        // the step + drain completes), and the C side stores nothing.
        let mut ctx = StepCtx {
            on_token: &mut on_token,
            panicked: Vec::new(),
        };
        let user_data = std::ptr::addr_of_mut!(ctx).cast::<c_void>();

        // SAFETY: handle is non-null; the trampoline matches OvCbTokenCallback;
        // user_data points to a live `StepCtx` for the call's duration.
        let ret = unsafe { ov_cb_step(self.handle, Some(step_trampoline), user_data) };

        if ret != 0 {
            bail!("ov_cb_step failed: {}", last_error());
        }

        // A panicked callback leaves its request's downstream state unknown —
        // cancel it in the pipeline so it never produces another callback and
        // its KV blocks are freed on the next step.
        for &id in &ctx.panicked {
            self.drop_request(id);
        }
        Ok(ctx.panicked)
    }

    /// Count the tokens `text` encodes to with the model's tokenizer.
    ///
    /// This is the exact-`usage` path: it runs the tokenizer but copies no ids
    /// (the C side fills only the count). Cheaper than [`encode`](Self::encode)
    /// when the ids themselves are not needed (e.g. `prompt_tokens`).
    ///
    /// # Errors
    /// Returns an error if the tokenizer fails.
    ///
    /// # Threading
    /// The tokenizer wraps a non-thread-safe `InferRequest`; call only from the
    /// engine thread (same contract as [`step`](Self::step)).
    pub fn count_tokens(&self, text: &str) -> anyhow::Result<usize> {
        let mut count: usize = 0;
        // SAFETY: handle is non-null (invariant); `text` is a valid `len`-byte
        // buffer alive for the call; no id buffer requested (null/cap 0); the
        // C side writes only through `out_count`, which points to `count`.
        let ret = unsafe {
            ov_cb_encode(
                self.handle,
                text.as_ptr(),
                text.len(),
                std::ptr::null_mut(),
                0,
                std::ptr::addr_of_mut!(count),
            )
        };
        if ret != 0 {
            bail!("ov_cb_encode (count) failed: {}", last_error());
        }
        Ok(count)
    }

    /// Encode `text` into token ids using the model's tokenizer.
    ///
    /// Runs in two passes — count, then fill an exactly-sized buffer — so the
    /// returned `Vec` is never over-allocated. Used by `/tokenize`; not on the
    /// generation hot path.
    ///
    /// # Errors
    /// Returns an error if the tokenizer fails.
    ///
    /// # Threading
    /// Engine-thread only (see [`count_tokens`](Self::count_tokens)).
    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<i64>> {
        let n = self.count_tokens(text)?;
        let mut ids = vec![0i64; n];
        if n == 0 {
            return Ok(ids);
        }
        let mut count: usize = 0;
        // SAFETY: handle non-null; `text` valid for `len` bytes; `ids` has `n`
        // slots and `cap == n`; `out_count` points to a live usize.
        let ret = unsafe {
            ov_cb_encode(
                self.handle,
                text.as_ptr(),
                text.len(),
                ids.as_mut_ptr(),
                n,
                std::ptr::addr_of_mut!(count),
            )
        };
        if ret != 0 {
            bail!("ov_cb_encode failed: {}", last_error());
        }
        // Deterministic, but truncate defensively in case the second pass
        // produced fewer ids than the first reported.
        ids.truncate(count.min(n));
        Ok(ids)
    }

    /// Decode token `ids` back to text using the model's detokenizer.
    ///
    /// Used by `/detokenize`; not on the generation hot path.
    ///
    /// # Errors
    /// Returns an error if the detokenizer fails.
    ///
    /// # Threading
    /// Engine-thread only (see [`count_tokens`](Self::count_tokens)).
    pub fn decode(&self, ids: &[i64]) -> anyhow::Result<String> {
        let mut out = String::new();
        // The trampoline pushes the decoded string into `out`. Same context-
        // passing trick as `step`: the closure lives on this frame for the
        // whole (blocking) call, and C stores nothing.
        let mut sink = |s: &str| out.push_str(s);
        let mut ctx = TextCbCtx {
            sink: &mut sink,
            panicked: false,
        };
        let user_data = std::ptr::addr_of_mut!(ctx).cast::<c_void>();

        // SAFETY: handle non-null; `ids`/`len` describe a valid slice (possibly
        // empty — the C side guards `ids == null` only when count > 0); the
        // trampoline matches `OvTextCallback`; `user_data` points to a live
        // `TextCbCtx` for the call's duration.
        let ret = unsafe {
            ov_cb_decode(
                self.handle,
                ids.as_ptr(),
                ids.len(),
                Some(decode_trampoline),
                user_data,
            )
        };
        if ret != 0 {
            bail!("ov_cb_decode failed: {}", last_error());
        }
        if ctx.panicked {
            bail!("decode text callback panicked");
        }
        Ok(out)
    }
}

// ── FFI callback contexts (panic firewall) ───────────────────────────────────
//
// Each trampoline below is an `extern "C"` frame: a Rust panic unwinding out
// of it into the C++ caller is a guaranteed process abort (whole-node DoS —
// one engine thread serves every request). Every trampoline therefore runs
// its closure under `std::panic::catch_unwind` and records the panic in a
// context struct passed through `user_data`; the safe wrapper turns the
// record into a per-request cleanup (`step`) or an error (`decode`) after the
// FFI call returns. The C side can't be told "this request failed" directly —
// its callbacks return void — so the record is the only channel out.

/// Signature of the per-token closure passed to [`OvCbEngine::step`]:
/// `(request_id, delta, new_tokens, finish, pool_exhausted)`. `pool_exhausted`
/// is `true` only when the C side's `finish_reason == 3` — the CB scheduler's
/// `GenerationStatus::IGNORED` (KV-pool exhaustion, this request never ran to
/// completion), confirmed by direct instrumentation
/// (the project's internal engineering log). `finish` is still
/// `Some(FinishReason::Stop)` in that case too (finalization must still
/// happen — see [`finish_from_code`]); callers that care about the
/// distinction check `pool_exhausted` separately rather than inventing a new
/// `FinishReason` variant, which would force every other caller of the
/// shared `FinishReason` type (VLM, NPU, streaming) to handle a state that
/// only the CB path can produce. Factored into a named alias because the
/// 5-parameter `dyn FnMut` trips clippy's `type_complexity`.
type OnTokenFn<'a> = dyn FnMut(u64, &str, usize, Option<FinishReason>, bool) + 'a;

/// Context for [`step_trampoline`], one per [`OvCbEngine::step`] call.
struct StepCtx<'a> {
    /// The per-token closure supplied to [`OvCbEngine::step`].
    on_token: &'a mut OnTokenFn<'a>,
    /// Quarantine list: ids whose closure panicked this step. Their closures
    /// are never invoked again; `step` cancels them in the pipeline and
    /// returns the ids to the caller.
    panicked: Vec<u64>,
}

/// Context for [`decode_trampoline`], one per [`OvCbEngine::decode`] call.
struct TextCbCtx<'a> {
    /// Receives the decoded text.
    sink: &'a mut dyn FnMut(&str),
    /// Set when the sink panicked — `decode` turns it into an error.
    panicked: bool,
}

/// C callback that decodes the raw delta into a `&str`, maps the finish code,
/// and calls the Rust closure stored in `user_data` — under a panic firewall.
///
/// # Safety
/// Called by C inside `ov_cb_step`. Invariants upheld by [`OvCbEngine::step`]:
/// - `user_data` is a valid `*mut StepCtx` on the caller's stack, alive for
///   the whole call.
/// - `delta` points to `len` bytes of complete UTF-8 (the bridge guards the
///   boundary); `len == 0` means no bytes (`delta` may be a valid empty string).
unsafe extern "C" fn step_trampoline(
    user_data: *mut c_void,
    request_id: u64,
    delta: *const u8,
    len: usize,
    finish_reason: std::ffi::c_int,
    new_tokens: usize,
) {
    // SAFETY: user_data is the pointer set up in OvCbEngine::step; valid and
    // non-null for the duration of the ov_cb_step call.
    let ctx = unsafe { &mut *user_data.cast::<StepCtx>() };

    // Quarantine guard: once a request's closure has panicked, its downstream
    // state is unknown — never invoke the closure for that id again.
    if ctx.panicked.contains(&request_id) {
        return;
    }

    // Build the &str. Guard len == 0 so we never deref for an empty delta.
    // The bridge holds back partial multibyte chars, so from_utf8 should always
    // succeed; on the off chance it doesn't, emit nothing rather than panic.
    let text = if len == 0 {
        ""
    } else {
        // SAFETY: delta points to `len` valid bytes owned by the C++ string,
        // alive until this callback returns.
        let bytes = unsafe { std::slice::from_raw_parts(delta, len) };
        std::str::from_utf8(bytes).unwrap_or("")
    };
    let finish = finish_from_code(finish_reason);
    let pool_exhausted = finish_reason == 3;

    // PANIC FIREWALL: a panic must never unwind out of this extern "C" frame
    // (see the context-struct block comment). AssertUnwindSafe is sound: on
    // panic the request is quarantined here, cancelled in the pipeline by
    // `step`, and its routing state is discarded by the engine loop — any
    // half-mutated per-request state is never observed again.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (ctx.on_token)(request_id, text, new_tokens, finish, pool_exhausted);
    }));
    if let Err(payload) = result {
        tracing::error!(
            request_id,
            panic = crate::ov_pipeline::panic_message(payload.as_ref()),
            "token callback panicked — quarantining request"
        );
        ctx.panicked.push(request_id);
    }
}

/// C callback that hands the decoded UTF-8 string to the Rust sink stored in
/// `user_data` (a [`TextCbCtx`] set up by [`OvCbEngine::decode`]) — under a
/// panic firewall.
///
/// # Safety
/// Called by C inside `ov_cb_decode`. Invariants upheld by `decode`:
/// - `user_data` is a valid `*mut TextCbCtx` on the caller's stack, alive for
///   the whole call.
/// - `text` points to `len` bytes of UTF-8 from the detokenizer's `std::string`,
///   alive until this callback returns; `len == 0` means the empty string.
unsafe extern "C" fn decode_trampoline(user_data: *mut c_void, text: *const u8, len: usize) {
    // SAFETY: user_data is the pointer set up in OvCbEngine::decode; valid and
    // non-null for the duration of the ov_cb_decode call.
    let ctx = unsafe { &mut *user_data.cast::<TextCbCtx>() };
    if ctx.panicked {
        return;
    }

    let s = if len == 0 {
        ""
    } else {
        // SAFETY: text points to `len` valid bytes owned by the C++ string,
        // alive until this callback returns.
        let bytes = unsafe { std::slice::from_raw_parts(text, len) };
        std::str::from_utf8(bytes).unwrap_or("")
    };

    // PANIC FIREWALL: never unwind out of this extern "C" frame (see the
    // context-struct block comment). AssertUnwindSafe is sound: on panic the
    // partially-filled output is discarded — `decode` returns an error.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (ctx.sink)(s)));
    if let Err(payload) = result {
        tracing::error!(
            panic = crate::ov_pipeline::panic_message(payload.as_ref()),
            "decode text callback panicked"
        );
        ctx.panicked = true;
    }
}

impl Drop for OvCbEngine {
    /// Synchronously frees the pipeline and its KV pool — VRAM released now.
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle is the unique owner; this is the only free site.
            unsafe { ov_cb_free(self.handle) };
            self.handle = std::ptr::null_mut();
            tracing::debug!("CB engine freed");
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
//
// Real engine tests need a GPU + model dir (integration only). Unit tests here
// cover the pure mapping logic that needs no FFI.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{
        FinishReason, OvCbEngine, OvGenParamsC, StepCtx, TextCbCtx, c_void, decode_trampoline,
        finish_from_code, openvino_version, ov_gen_params_size, step_trampoline,
    };

    /// Regression test for a real near-miss (2026-07-14): adding
    /// `repetition_penalty` to `OvGenParams` (C++) without updating BOTH of
    /// its private Rust `#[repr(C)]` mirrors (`ov_cb.rs`, `ov_vlm.rs`) silently
    /// misaligns every field after the missed one — no compile error, no
    /// panic, just garbage values crossing the FFI boundary. This test would
    /// have caught it immediately: `size_of::<OvGenParamsC>()` must equal
    /// `sizeof(OvGenParams)` on the C++ side. See `ov_vlm.rs` for the second
    /// copy's identical test.
    #[test]
    fn ov_gen_params_c_matches_bridge_size() {
        assert_eq!(
            std::mem::size_of::<OvGenParamsC>(),
            unsafe { ov_gen_params_size() },
            "OvGenParamsC (ov_cb.rs) has drifted from OvGenParams (ov_bridge.cpp) — \
             update both #[repr(C)] Rust mirrors when changing the C++ struct"
        );
    }

    /// `openvino_version()` returns the real running runtime's build number
    /// (non-empty, e.g. `"2026.2.1-..."`) and is stable across calls — the
    /// `OnceLock` cache must not re-cross the FFI boundary and must not
    /// silently start returning `None` after the first successful call
    /// (`PLAN_image_metadata_response.md` Tier 3).
    #[test]
    fn openvino_version_is_stable_and_nonempty() {
        let a = openvino_version().expect("running OpenVINO runtime reports a version");
        let b = openvino_version();
        assert!(!a.is_empty());
        assert_eq!(Some(a), b, "cached value must be stable across calls");
    }

    /// T1.2 accept test: a panic inside the per-token closure is caught at the
    /// FFI boundary (thread survives — this test would abort the process
    /// otherwise), the request is quarantined (its closure never runs again),
    /// and other requests keep flowing.
    #[test]
    fn step_trampoline_catches_panic_and_quarantines_request() {
        let mut delivered: Vec<(u64, String)> = Vec::new();
        let mut on_token = |id: u64,
                            delta: &str,
                            _new_tokens: usize,
                            _finish: Option<FinishReason>,
                            _pool_exhausted: bool| {
            // The simulated closure bug: request 7's deliveries always panic.
            assert!(id != 7, "closure bug for request 7");
            delivered.push((id, delta.to_owned()));
        };
        let mut ctx = StepCtx {
            on_token: &mut on_token,
            panicked: Vec::new(),
        };
        let user_data = std::ptr::addr_of_mut!(ctx).cast::<c_void>();

        let delta = b"hi";
        // SAFETY: mimics the C side exactly — valid StepCtx pointer, valid
        // UTF-8 buffer of the stated length, alive across the calls.
        unsafe {
            // Request 7 panics — must be caught, not unwind out of the test.
            step_trampoline(user_data, 7, delta.as_ptr(), delta.len(), 0, 1);
            // Request 7 again — quarantined, closure must NOT run (no panic).
            step_trampoline(user_data, 7, delta.as_ptr(), delta.len(), 0, 1);
            // Request 3 — healthy, must be delivered normally.
            step_trampoline(user_data, 3, delta.as_ptr(), delta.len(), 1, 1);
        }

        assert_eq!(ctx.panicked, vec![7], "request 7 must be quarantined once");
        assert_eq!(
            delivered,
            vec![(3, "hi".to_owned())],
            "healthy request must be unaffected by the poisoned one"
        );
    }

    /// Same firewall on the decode path: a panicking sink is caught and the
    /// context is flagged so `decode` can return an error.
    #[test]
    fn decode_trampoline_catches_panic_and_flags_context() {
        let mut sink = |_s: &str| panic!("sink bug");
        let mut ctx = TextCbCtx {
            sink: &mut sink,
            panicked: false,
        };
        let user_data = std::ptr::addr_of_mut!(ctx).cast::<c_void>();

        let text = b"abc";
        // SAFETY: mimics the C side — valid TextCbCtx pointer, valid UTF-8
        // buffer of the stated length.
        unsafe {
            decode_trampoline(user_data, text.as_ptr(), text.len());
            // Second call: already flagged — the sink must be skipped (a
            // second panic here would prove the guard failed).
            decode_trampoline(user_data, text.as_ptr(), text.len());
        }
        assert!(ctx.panicked, "the sink panic must be recorded");
    }

    #[test]
    fn finish_code_mapping() {
        assert_eq!(finish_from_code(0), None);
        assert_eq!(finish_from_code(1), Some(FinishReason::Stop));
        assert_eq!(finish_from_code(2), Some(FinishReason::Length));
        // Code 3 (pool-exhausted) still finalizes as Stop here — callers
        // needing to distinguish it check the separate `pool_exhausted` flag
        // `step_trampoline` computes, not a different `finish_from_code` value.
        assert_eq!(finish_from_code(3), Some(FinishReason::Stop));
        // Unknown nonzero codes close the stream as Stop, never hang.
        assert_eq!(finish_from_code(99), Some(FinishReason::Stop));
    }

    /// `step_trampoline` sets `pool_exhausted` true only for code 3, and
    /// still finalizes the request (`finish` stays `Some(Stop)`) either way —
    /// verified end-to-end through the real trampoline, not just
    /// `finish_from_code` in isolation, since `pool_exhausted` is computed
    /// separately in `step_trampoline` itself.
    #[test]
    fn step_trampoline_flags_pool_exhausted_only_for_code_3() {
        let mut calls: Vec<(u64, Option<FinishReason>, bool)> = Vec::new();
        let mut on_token = |id: u64, _delta: &str, _new_tokens: usize, finish, pool_exhausted| {
            calls.push((id, finish, pool_exhausted));
        };
        let mut ctx = StepCtx {
            on_token: &mut on_token,
            panicked: Vec::new(),
        };
        let user_data = std::ptr::addr_of_mut!(ctx).cast::<c_void>();
        let delta = b"";
        // SAFETY: mimics the C side — valid StepCtx pointer, empty (still
        // valid) UTF-8 buffer, alive across the calls.
        unsafe {
            step_trampoline(user_data, 1, delta.as_ptr(), delta.len(), 1, 0); // genuine STOP
            step_trampoline(user_data, 2, delta.as_ptr(), delta.len(), 3, 0); // pool-exhausted
        }
        assert_eq!(
            calls,
            vec![
                (1, Some(FinishReason::Stop), false),
                (2, Some(FinishReason::Stop), true),
            ]
        );
    }

    #[test]
    fn finish_reason_openai_strings() {
        assert_eq!(FinishReason::Stop.as_openai(), "stop");
        assert_eq!(FinishReason::Length.as_openai(), "length");
    }

    /// Live tokenizer round-trip on the real model. Needs GPU.1 + the model
    /// dir, so it is ignored by default — run explicitly with:
    ///   `cargo test -p rustedvino --lib -- --ignored tokenizer_round_trip`
    /// Override the model path with `RV_TEST_MODEL` if it lives elsewhere.
    #[test]
    #[ignore = "needs GPU.1 + model dir; run with --ignored"]
    fn tokenizer_round_trip_on_real_model() {
        let model = std::env::var("RV_TEST_MODEL")
            .unwrap_or_else(|_| "/opt/rustedvino/models/qwen3-8b-int4-ov".to_owned());
        // One slot, tiny KV pool — we only exercise the tokenizer, not decoding.
        // Empty kv_cache_precision = plugin default.
        let engine = OvCbEngine::new(&model, "GPU.1", 1, 1.0, "", "", true, "", "", 0).unwrap();

        let text = "Hello, world! 你好";
        let ids = engine.encode(text).unwrap();
        assert!(!ids.is_empty(), "encode must produce token ids");

        // count_tokens must agree with the length encode returns.
        let count = engine.count_tokens(text).unwrap();
        assert_eq!(count, ids.len(), "count_tokens must match encode() length");

        // Decoding the ids must recover the meaningful content (the detokenizer
        // may normalise leading/trailing whitespace, so assert on a substring).
        let decoded = engine.decode(&ids).unwrap();
        assert!(
            decoded.contains("Hello, world!") && decoded.contains("你好"),
            "round-trip lost content: {decoded:?}"
        );

        // Empty edge cases must not error.
        assert_eq!(engine.decode(&[]).unwrap(), "", "decode([]) must be empty");
        let _ = engine.count_tokens("").unwrap();
    }
}
