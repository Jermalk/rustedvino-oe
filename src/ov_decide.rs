// ============================================================
// src/ov_decide.rs — Rust FFI wrapper for the decision (System One) bridge path
// ============================================================
// The C bridge exposes five calls:
// ov_decide_create / free / encode / prefill / vocab_size. There is no GenAI pipeline:
// one stateful InferRequest on a text-only decision IR, one fresh prefill per question,
// and only the k candidate logits come back over the FFI (never the 248 320-wide row).
//
// CRASH COURSE — why `&mut self` on `prefill` but `&self` on `encode`:
//   A prefill mutates the request's hidden states (reset, then run). Taking `&mut self`
//   lets the borrow checker prove no two prefills can overlap on one engine, even by
//   accident inside the engine thread. Encoding only reads the tokenizer.
// ============================================================

use std::ffi::{CString, c_char, c_int, c_void};

use anyhow::Context as _;

use crate::ov_pipeline::last_error;

unsafe extern "C" {
    fn ov_decide_create(
        model_dir: *const c_char,
        device: *const c_char,
        ov_cache_dir: *const c_char,
        max_tokens: usize,
    ) -> *mut c_void;

    fn ov_decide_free(handle: *mut c_void);

    fn ov_decide_encode(
        handle: *mut c_void,
        text: *const c_char,
        len: usize,
        out_ids: *mut i64,
        cap: usize,
        out_count: *mut usize,
    ) -> c_int;

    fn ov_decide_prefill(
        handle: *mut c_void,
        ids: *const i64,
        n: usize,
        cand: *const i64,
        k: usize,
        out_logits: *mut f32,
    ) -> c_int;

    fn ov_decide_vocab_size(handle: *mut c_void, out_vocab: *mut usize) -> c_int;
}

/// A loaded decision model — lives on a single dedicated OS thread.
///
/// `drop` frees the compiled model and its device memory synchronously.
pub struct OvDecideEngine {
    handle: *mut c_void,
    model_dir: String,
    device: String,
    cache_dir: String,
    max_tokens: usize,
    vocab: usize,
}

// SAFETY: OvDecideEngine owns its OvDecideState exclusively. The raw pointer is moved onto
// the engine thread at spawn time and never shared or aliased; OpenVINO's InferRequest is
// not documented as thread-safe, so we deliberately do NOT implement Sync.
unsafe impl Send for OvDecideEngine {}

fn create(
    model_dir: &str,
    device: &str,
    cache_dir: &str,
    max_tokens: usize,
) -> anyhow::Result<*mut c_void> {
    let c_dir = CString::new(model_dir).context("model_dir contains interior NUL")?;
    let c_device = CString::new(device).context("device contains interior NUL")?;
    let c_cache = CString::new(cache_dir).context("cache_dir contains interior NUL")?;
    // SAFETY: the CStrings outlive the call; ov_decide_create returns a heap-allocated
    // OvDecideState or NULL (message in ov_last_error on this thread).
    let handle = unsafe {
        ov_decide_create(
            c_dir.as_ptr(),
            c_device.as_ptr(),
            c_cache.as_ptr(),
            max_tokens,
        )
    };
    if handle.is_null() {
        anyhow::bail!("ov_decide_create failed: {}", last_error());
    }
    Ok(handle)
}

impl OvDecideEngine {
    /// Load and compile the gathered decision IR in `model_dir` on `device`.
    /// An empty `cache_dir` disables the `OpenVINO` model cache. Every prefill is capped at `max_tokens`.
    ///
    /// # Errors
    /// Fails if the IR is missing, not gathered to the last position, has non-f32 logits, or
    /// compilation fails on `device`.
    pub fn new(
        model_dir: &str,
        device: &str,
        cache_dir: &str,
        max_tokens: usize,
    ) -> anyhow::Result<Self> {
        let handle = create(model_dir, device, cache_dir, max_tokens)?;
        let mut vocab = 0usize;
        // SAFETY: handle is non-null and exclusively ours; `vocab` is a valid out-pointer.
        if unsafe { ov_decide_vocab_size(handle, &raw mut vocab) } != 0 {
            let err = last_error();
            // SAFETY: handle is valid and not yet stored anywhere else.
            unsafe { ov_decide_free(handle) };
            anyhow::bail!("ov_decide_vocab_size failed: {err}");
        }
        tracing::info!(model = %model_dir, device = %device, vocab, max_tokens, "decision engine loaded");
        Ok(Self {
            handle,
            model_dir: model_dir.to_owned(),
            device: device.to_owned(),
            cache_dir: cache_dir.to_owned(),
            max_tokens,
            vocab,
        })
    }

    /// Recreate the compiled model in place after a failed call (same recovery as the
    /// rerank engine: one failed request instead of a standing outage).
    ///
    /// # Errors
    /// Fails if the model cannot be recompiled; `self` then holds a null handle and every
    /// later call fails fast.
    pub fn reload(&mut self) -> anyhow::Result<()> {
        if !self.handle.is_null() {
            // SAFETY: handle is valid, exclusively owned, called only on the engine thread.
            unsafe { ov_decide_free(self.handle) };
            self.handle = std::ptr::null_mut();
        }
        self.handle = create(
            &self.model_dir,
            &self.device,
            &self.cache_dir,
            self.max_tokens,
        )?;
        tracing::info!(model = %self.model_dir, device = %self.device, "decision engine reloaded after a failed call");
        Ok(())
    }

    /// Vocabulary size (static logits width).
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab
    }

    /// Prefill cap given at load.
    #[must_use]
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Token ids of `text`, no special tokens added (chat control tokens inside the rendered
    /// prompt are parsed as specials by the tokenizer IR).
    ///
    /// # Errors
    /// Fails if the engine is not loaded or tokenisation fails.
    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<i64>> {
        if self.handle.is_null() {
            anyhow::bail!("decision engine is not loaded (a previous reload failed)");
        }
        let mut count = 0usize;
        // SAFETY: handle non-null; `text` is valid for `text.len()` bytes; count-only pass
        // (out_ids NULL, cap 0) writes only `count`.
        let rc = unsafe {
            ov_decide_encode(
                self.handle,
                text.as_ptr().cast(),
                text.len(),
                std::ptr::null_mut(),
                0,
                &raw mut count,
            )
        };
        if rc != 0 {
            anyhow::bail!("ov_decide_encode failed: {}", last_error());
        }
        let mut ids = vec![0i64; count];
        // SAFETY: `ids` has room for `count` i64s and the bridge copies at most `cap` of them.
        let rc = unsafe {
            ov_decide_encode(
                self.handle,
                text.as_ptr().cast(),
                text.len(),
                ids.as_mut_ptr(),
                ids.len(),
                &raw mut count,
            )
        };
        if rc != 0 {
            anyhow::bail!("ov_decide_encode failed: {}", last_error());
        }
        // The tokenizer is deterministic, so both passes agree; if they ever did not, a short
        // or zero-padded vector would be scored silently — fail instead.
        anyhow::ensure!(
            count == ids.len(),
            "ov_decide_encode: token count changed between passes ({} then {count})",
            ids.len()
        );
        Ok(ids)
    }

    /// One fresh prefill of `ids`; returns the logits of `candidates` at the last position.
    ///
    /// # Errors
    /// Fails on invalid arguments (empty, over `max_tokens`, candidate outside the vocabulary)
    /// or an inference error — after which the caller should [`Self::reload`].
    pub fn prefill(&mut self, ids: &[i64], candidates: &[i64]) -> anyhow::Result<Vec<f32>> {
        if self.handle.is_null() {
            anyhow::bail!("decision engine is not loaded (a previous reload failed)");
        }
        let mut out = vec![0f32; candidates.len()];
        // SAFETY: handle non-null and exclusively borrowed (&mut self); `ids`/`candidates`
        // are valid for their lengths; `out` has room for exactly `candidates.len()` floats.
        let rc = unsafe {
            ov_decide_prefill(
                self.handle,
                ids.as_ptr(),
                ids.len(),
                candidates.as_ptr(),
                candidates.len(),
                out.as_mut_ptr(),
            )
        };
        if rc != 0 {
            anyhow::bail!("ov_decide_prefill failed: {}", last_error());
        }
        Ok(out)
    }
}

impl Drop for OvDecideEngine {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle is valid; we own it exclusively; dropped on the engine thread.
            unsafe { ov_decide_free(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}
