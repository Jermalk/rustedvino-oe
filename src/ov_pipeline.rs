// ============================================================
// src/ov_pipeline.rs — Safe Rust wrapper around ov_bridge.cpp
// ============================================================
//
// CRASH COURSE — FFI (Foreign Function Interface):
//   Rust can call C functions directly. You declare the C function
//   signatures in an `extern "C"` block — Rust trusts you to get
//   them right. Every call to an `extern "C"` function is `unsafe`
//   because Rust cannot verify the C code's memory safety.
//
//   The pattern here:
//   1. Unsafe `extern "C"` block — raw C signatures (never call directly)
//   2. Safe `OvPipeline` struct — wraps the opaque pointer and enforces
//      the invariants (never null after creation, not Send+Sync alone)
//   3. Public API — `new()` + `generate()` — no `unsafe` at call site
//
// CRASH COURSE — Arc<Mutex<OvPipeline>>:
//   LLMPipeline is NOT thread-safe (OV does not guard it internally).
//   OvPipeline holds a raw pointer, which is neither Send nor Sync by
//   default. We implement Send manually (it's safe because we guarantee
//   single-threaded access through the Mutex). The Mutex itself is Sync,
//   so Arc<Mutex<OvPipeline>> can be shared across async tasks.
//
// CRASH COURSE — spawn_blocking:
//   `generate()` is a long-running synchronous call — it blocks the
//   OS thread for the entire generation duration. You must NEVER call
//   blocking code on the tokio async runtime thread pool. Use
//   `tokio::task::spawn_blocking` to run it on a dedicated thread.
//   The chat handler does this; ov_pipeline.rs just provides the call.
// ============================================================

use anyhow::{Context, bail};
use std::ffi::{CStr, CString};

use crate::ov_cb::FinishReason;

// ── Raw C declarations ────────────────────────────────────────────────────────
//
// These must exactly match the signatures in ov_bridge/ov_bridge.cpp.
// `*mut std::ffi::c_void` = the opaque `OvPipelineHandle` on the C side.
// `unsafe extern "C"` = "this is a C function; calling it is unsafe."

#[allow(non_camel_case_types)]
type OvTokenCallback = unsafe extern "C" fn(
    token: *const u8,
    len: usize,
    user_data: *mut std::ffi::c_void,
) -> std::ffi::c_int;

unsafe extern "C" {
    fn ov_pipeline_create(
        model_path: *const std::ffi::c_char,
        device: *const std::ffi::c_char,
        max_prompt_len: u32,
    ) -> *mut std::ffi::c_void;

    fn ov_pipeline_free(handle: *mut std::ffi::c_void);

    fn ov_pipeline_generate(
        handle: *mut std::ffi::c_void,
        prompt: *const std::ffi::c_char,
        max_new_tokens: usize,
        callback: Option<OvTokenCallback>,
        user_data: *mut std::ffi::c_void,
        finish_code_out: *mut std::ffi::c_int,
    ) -> std::ffi::c_int;

    fn ov_pipeline_encode(
        handle: *mut std::ffi::c_void,
        text: *const u8,
        len: usize,
        out_ids: *mut i64,
        cap: usize,
        out_count: *mut usize,
    ) -> std::ffi::c_int;

    fn ov_last_error() -> *const std::ffi::c_char;
}

// ── Helper: read the thread-local C error buffer ─────────────────────────────

/// Returns the last error set by any `ov_*` function on this thread,
/// or `"(no error)"` if the buffer is empty.
///
/// Shared with [`crate::ov_cb`] — the CB engine sets the same thread-local
/// buffer via the same `ov_last_error` symbol, so both wrappers read it here.
///
/// # Safety
/// Must only be called immediately after a failed `ov_*` call,
/// before any other `ov_*` call that might overwrite the buffer.
pub(crate) fn last_error() -> String {
    // SAFETY: ov_last_error returns a pointer to a thread-local static
    // buffer. It is always non-null and valid UTF-8 (or ASCII at worst).
    // We copy it into a Rust String before any further ov_ calls.
    unsafe {
        let ptr = ov_last_error();
        if ptr.is_null() {
            return "(no error)".to_owned();
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

// ── Helper: panic payload → message ──────────────────────────────────────────

/// Best-effort extraction of a human-readable message from a panic payload.
///
/// Shared by every FFI trampoline's panic firewall (`ov_cb`, `ov_vlm`,
/// `ov_embed`, this module): `std::panic::catch_unwind` yields a
/// `Box<dyn Any>`, which is a `&str` or `String` for ordinary panics.
pub(crate) fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload.downcast_ref::<&str>().copied().unwrap_or_else(|| {
        payload
            .downcast_ref::<String>()
            .map_or("<non-string panic payload>", String::as_str)
    })
}

// ── OvPipeline ───────────────────────────────────────────────────────────────

/// A loaded `OpenVINO` `GenAI` `LLMPipeline` — the actual GPU inference engine.
///
/// - Creation (`new`) happens once at startup, inside `spawn_blocking`.
/// - Generation (`generate`) blocks; always call from `spawn_blocking`.
/// - Destruction (`drop`) runs synchronously — VRAM is freed immediately.
///
/// Wrap in `Arc<Mutex<OvPipeline>>` for safe sharing across tasks.
pub struct OvPipeline {
    /// Opaque C++ pointer. Non-null after successful `new()`.
    handle: *mut std::ffi::c_void,
}

// SAFETY: OvPipeline owns the LLMPipeline exclusively. We never give out
// a reference to the inner pointer without holding the wrapping Mutex.
// Therefore it is safe to move the pointer across thread boundaries.
//
// WHY WE NEED THIS: raw pointers are !Send + !Sync by default because the
// compiler cannot know whether the pointee is thread-safe. We know it isn't
// (hence the Mutex), but we can safely send the *ownership* to another thread.
unsafe impl Send for OvPipeline {}

impl OvPipeline {
    /// Load the model and return a ready-to-use pipeline.
    ///
    /// # Arguments
    /// * `model_path` – directory containing `.xml`/`.bin` files and tokenizer
    ///   (e.g. `"/opt/rustedvino/models/qwen3-8b-int4-ov"`)
    /// * `device` – `OpenVINO` device string (`"GPU.1"`, `"CPU"`, …)
    /// * `max_prompt_len` – NPU-only compile-time `MAX_PROMPT_LEN` ceiling.
    ///   `None` keeps `OpenVINO`'s own default (1024 as of this writing);
    ///   ignored (harmlessly) for non-NPU devices. See the project's internal engineering log #6.
    ///
    /// # Errors
    /// Returns an error if the model directory is invalid, the device is
    /// unavailable, or model loading fails.
    ///
    /// # Performance
    /// Model loading takes several seconds (JIT compilation on GPU).
    /// Call from `tokio::task::spawn_blocking`, never from an async context.
    pub fn new(
        model_path: &str,
        device: &str,
        max_prompt_len: Option<u32>,
    ) -> anyhow::Result<Self> {
        let c_path = CString::new(model_path).context("model_path contains interior NUL byte")?;
        let c_device = CString::new(device).context("device contains interior NUL byte")?;

        // SAFETY: Both CStrings are alive for the duration of the call.
        // ov_pipeline_create returns either a valid pointer or NULL.
        let handle = unsafe {
            ov_pipeline_create(
                c_path.as_ptr(),
                c_device.as_ptr(),
                max_prompt_len.unwrap_or(0),
            )
        };

        if handle.is_null() {
            bail!("ov_pipeline_create failed: {}", last_error());
        }

        tracing::info!(model = %model_path, device = %device, max_prompt_len, "OV pipeline loaded");
        Ok(Self { handle })
    }

    /// Count `text`'s tokens using the pipeline's tokenizer, independent of
    /// `generate()`. Used to pre-flight a prompt against a (possibly
    /// NPU-limited) length ceiling before committing to generation — see
    /// `crate::handlers::chat::gate_npu_prompt`.
    ///
    /// # Errors
    /// Returns an error if the tokenizer fails.
    pub fn count_tokens(&self, text: &str) -> anyhow::Result<usize> {
        let mut count: usize = 0;
        // SAFETY: handle is non-null (invariant); `text` is a valid `len`-byte
        // buffer alive for the call; no id buffer requested (null/cap 0); the
        // C side writes only through `out_count`, which points to `count`.
        let ret = unsafe {
            ov_pipeline_encode(
                self.handle,
                text.as_ptr(),
                text.len(),
                std::ptr::null_mut(),
                0,
                std::ptr::addr_of_mut!(count),
            )
        };
        if ret != 0 {
            bail!("ov_pipeline_encode (count) failed: {}", last_error());
        }
        Ok(count)
    }

    /// Run inference synchronously, streaming tokens to `on_token`.
    ///
    /// # Arguments
    /// * `prompt` – the user's text; the model applies its chat template
    /// * `max_new_tokens` – token budget; `0` means use model default
    /// * `on_token` – closure called once per generated token;
    ///   return `true` to continue, `false` to stop early
    ///
    /// Returns the generation's [`FinishReason`]: `Length` when the token
    /// budget cut the generation, `Stop` otherwise (EOS or early stop).
    ///
    /// # Errors
    /// Returns an error if inference fails (e.g. OOM, device error).
    ///
    /// # Safety / Threading
    /// Blocks the calling thread. Call from `spawn_blocking` only.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        mut on_token: impl FnMut(&str) -> bool,
    ) -> anyhow::Result<FinishReason> {
        let c_prompt = CString::new(prompt).context("prompt contains interior NUL byte")?;

        // We pass `on_token` to C as a raw pointer in `user_data`.
        // The C callback casts it back and calls it.
        //
        // CRASH COURSE — trait object pointer as user_data:
        //   `on_token` is a closure — an unnamed type implementing FnMut(&str)->bool.
        //   We can't pass it to C directly, but we CAN pass a pointer to it.
        //   The C code stores nothing; it just calls the function pointer with
        //   the user_data pointer on every token. The lifetime is safe because
        //   `on_token` lives on the stack for the entire duration of
        //   `ov_pipeline_generate` (which blocks until generation is done).
        // The trampoline runs the closure under a panic firewall; `panicked`
        // records a caught panic so we can surface it as an error after the
        // blocking call returns (the C side only sees "stop generating").
        let mut ctx = TokenCbCtx {
            on_token: &mut on_token,
            panicked: false,
        };
        let user_data_ptr = std::ptr::addr_of_mut!(ctx).cast::<std::ffi::c_void>();

        let mut finish_code: std::ffi::c_int = 1; // STOP default

        // SAFETY: handle is non-null (invariant), c_prompt is valid,
        // trampoline has the correct signature, user_data points to a valid
        // FnMut on this stack frame, and finish_code lives on this frame —
        // all alive for the (blocking) call duration.
        let ret = unsafe {
            ov_pipeline_generate(
                self.handle,
                c_prompt.as_ptr(),
                max_new_tokens,
                Some(token_trampoline),
                user_data_ptr,
                &raw mut finish_code,
            )
        };

        if ret != 0 {
            bail!("ov_pipeline_generate failed: {}", last_error());
        }
        if ctx.panicked {
            bail!("token callback panicked during generation");
        }
        Ok(match finish_code {
            2 => FinishReason::Length,
            _ => FinishReason::Stop,
        })
    }
}

/// Everything [`token_trampoline`] needs, passed through `user_data` for one
/// `ov_pipeline_generate` call.
struct TokenCbCtx<'a> {
    /// The per-token closure supplied to [`OvPipeline::generate`].
    on_token: &'a mut dyn FnMut(&str) -> bool,
    /// Set when the closure panicked — the trampoline stops generation and
    /// `generate` turns the flag into an error after the FFI call returns.
    panicked: bool,
}

/// C callback that converts the raw token pointer into a `&str` and calls
/// the Rust closure stored in `user_data`.
///
/// # Safety
/// Called by C code inside `ov_pipeline_generate`. Invariants upheld by the
/// calling code:
/// - `token` is a valid pointer to `len` bytes of UTF-8 (`GenAI` guarantees this)
/// - `user_data` is a valid `*mut TokenCbCtx` on the caller's stack
/// - The context is alive for the entire call (stack lifetime)
unsafe extern "C" fn token_trampoline(
    token: *const u8,
    len: usize,
    user_data: *mut std::ffi::c_void,
) -> std::ffi::c_int {
    // CRASH COURSE — Rust 2024 requires unsafe blocks inside unsafe fn:
    //   Before 2024, `unsafe fn` made the entire body implicitly unsafe.
    //   Rust 2024 changed this: you must still write `unsafe {}` around each
    //   unsafe operation inside an unsafe function. The goal: make it clear
    //   exactly which *specific* operation is unsafe, not the whole function.

    // SAFETY: user_data is a valid *mut TokenCbCtx on the caller's stack.
    // The caller (ov_pipeline_generate) guarantees this.
    let ctx = unsafe { &mut *user_data.cast::<TokenCbCtx>() };
    if ctx.panicked {
        return 1; // already poisoned — keep telling C to stop
    }

    // Convert the raw bytes to a &str.
    // If OV returns non-UTF-8 (shouldn't happen), we skip the token silently.
    // SAFETY: token points to `len` valid bytes allocated by the C++ string.
    let bytes = unsafe { std::slice::from_raw_parts(token, len) };
    if let Ok(s) = std::str::from_utf8(bytes) {
        // PANIC FIREWALL: a panic must never unwind out of this extern "C"
        // frame into the C++ caller — that is a guaranteed process abort.
        // AssertUnwindSafe is sound: on panic we stop generation and `generate`
        // reports an error, so any half-mutated closure state is never reused.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (ctx.on_token)(s)));
        match result {
            Ok(cont) => i32::from(!cont),
            Err(payload) => {
                tracing::error!(
                    panic = panic_message(payload.as_ref()),
                    "token callback panicked — stopping generation"
                );
                ctx.panicked = true;
                1 // stop generation
            }
        }
    } else {
        // Bad UTF-8 — skip silently, continue generation
        0
    }
}

impl Drop for OvPipeline {
    /// Synchronously frees the pipeline, releasing VRAM immediately.
    ///
    /// CRASH COURSE — drop in Rust:
    ///   Unlike Python's GC, Rust's destructor runs *immediately* when the
    ///   last owner goes out of scope. For GPU memory this is critical:
    ///   `drop(pipeline)` → VRAM freed now, not at next GC cycle.
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle is the unique owner of the LLMPipeline.
            // This is the only place it is freed (no double-free possible).
            unsafe { ov_pipeline_free(self.handle) };
            self.handle = std::ptr::null_mut();
            tracing::debug!("OV pipeline freed");
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
//
// Real OV tests require a GPU and a model directory — they are integration
// tests and run only when explicitly enabled via environment variables.
// Unit tests here cover only the safe Rust wrapper logic.

#[cfg(test)]
mod tests {
    use super::last_error;

    /// Smoke test: `last_error()` returns something sensible when buffer is clean.
    #[test]
    fn last_error_returns_string() {
        let e = last_error();
        // After no OV call the thread-local buffer is clean — function
        // returns "(no error)" or an empty string. Either is acceptable;
        // nothing else is.
        assert!(
            e == "(no error)" || e.is_empty(),
            "expected \"(no error)\" or empty, got: {e:?}"
        );
    }
}
