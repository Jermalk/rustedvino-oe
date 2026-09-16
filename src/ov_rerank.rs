// ============================================================
// src/ov_rerank.rs — Rust FFI wrapper for TextRerankPipeline
// ============================================================
// Mirrors the structure of ov_embed.rs. The C bridge exposes
// ov_rerank_create / ov_rerank_documents / ov_rerank_free (the same
// opaque-handle + callback pattern as the embedding bridge).
//
// The pipeline is created with a large top_n so all documents are scored
// and sorted; Rust applies the per-request top_n trim.
// ============================================================

use std::ffi::{CString, c_void};
use std::panic::AssertUnwindSafe;

use anyhow::Context as _;

use crate::ov_pipeline::last_error;

// ── Top-N cap used at pipeline creation ──────────────────────────────────────
// Passes a large-enough top_n so the C++ pipeline scores all documents.
// Per-request top_n trimming happens in the Rust handler.
pub const RERANK_PIPELINE_TOP_N: usize = 10_000;

// ── Callback type ─────────────────────────────────────────────────────────────

type OvRerankCallback = unsafe extern "C" fn(user_data: *mut c_void, index: usize, score: f32);

// ── FFI declarations ──────────────────────────────────────────────────────────

unsafe extern "C" {
    fn ov_rerank_create(
        model_path: *const std::ffi::c_char,
        device: *const std::ffi::c_char,
        top_n: usize,
    ) -> *mut c_void;

    fn ov_rerank_free(handle: *mut c_void);

    fn ov_rerank_documents(
        handle: *mut c_void,
        query: *const std::ffi::c_char,
        texts: *const *const std::ffi::c_char,
        n: usize,
        callback: Option<OvRerankCallback>,
        user_data: *mut c_void,
    ) -> std::ffi::c_int;

    fn ov_rerank_count_tokens(
        handle: *mut c_void,
        query: *const std::ffi::c_char,
        document: *const std::ffi::c_char,
    ) -> usize;
}

// ── OvRerankEngine ────────────────────────────────────────────────────────────

/// A loaded `TextRerankPipeline` — lives on a single dedicated OS thread.
///
/// `drop` frees the pipeline and its GPU memory synchronously.
pub struct OvRerankEngine {
    handle: *mut c_void,
    model_path: String,
    device: String,
}

// SAFETY: OvRerankEngine owns the TextRerankPipeline exclusively. The raw
// pointer is moved onto the engine thread at spawn time and never shared or
// aliased. We deliberately do NOT implement Sync.
unsafe impl Send for OvRerankEngine {}

impl OvRerankEngine {
    /// Load the reranking model at `model_path` on `device`.
    ///
    /// # Errors
    /// Returns an error if the model dir is invalid, the device is unavailable,
    /// or pipeline construction fails.
    pub fn new(model_path: &str, device: &str) -> anyhow::Result<Self> {
        let c_path = CString::new(model_path).context("model_path contains interior NUL")?;
        let c_device = CString::new(device).context("device contains interior NUL")?;

        // SAFETY: CStrings outlive the call; ov_rerank_create returns a valid
        // heap-allocated OvRerankState or NULL on failure.
        let handle =
            unsafe { ov_rerank_create(c_path.as_ptr(), c_device.as_ptr(), RERANK_PIPELINE_TOP_N) };

        if handle.is_null() {
            anyhow::bail!("ov_rerank_create failed: {}", last_error());
        }

        tracing::info!(model = %model_path, device = %device, "reranking engine loaded");
        Ok(Self {
            handle,
            model_path: model_path.to_owned(),
            device: device.to_owned(),
        })
    }

    /// Recreate the underlying pipeline in place, freeing the current handle
    /// first.
    ///
    /// Self-healing recovery (2026-08-04,
    /// the project's internal engineering log): a failed `rerank()`
    /// call can leave the pipeline's internal inference-request tensor state
    /// permanently inconsistent — every subsequent call on the same load then
    /// fails identically until an operator manually evicts and reloads the
    /// model. Calling this after any `rerank()` error converts that standing
    /// outage into a single failed request.
    ///
    /// # Errors
    /// Returns an error if the new pipeline fails to construct. `self` is left
    /// with a null handle in that case, so a subsequent `rerank()` call fails
    /// fast (see the null check there) instead of reusing a freed pointer.
    pub fn reload(&mut self) -> anyhow::Result<()> {
        if !self.handle.is_null() {
            // SAFETY: handle is valid, exclusively owned, called only from the
            // dedicated engine thread.
            unsafe { ov_rerank_free(self.handle) };
            self.handle = std::ptr::null_mut();
        }

        let c_path =
            CString::new(self.model_path.as_str()).context("model_path contains interior NUL")?;
        let c_device =
            CString::new(self.device.as_str()).context("device contains interior NUL")?;

        // SAFETY: CStrings outlive the call; ov_rerank_create returns a valid
        // heap-allocated OvRerankState or NULL on failure.
        let handle =
            unsafe { ov_rerank_create(c_path.as_ptr(), c_device.as_ptr(), RERANK_PIPELINE_TOP_N) };
        if handle.is_null() {
            anyhow::bail!("ov_rerank_create failed during reload: {}", last_error());
        }

        self.handle = handle;
        tracing::info!(
            model = %self.model_path,
            device = %self.device,
            "reranking engine pipeline reloaded after a failed call"
        );
        Ok(())
    }

    /// Rerank `texts` against `query`.
    ///
    /// Returns `(original_index, relevance_score)` pairs sorted by score
    /// descending. An empty `texts` returns an empty `Vec` without an FFI call.
    ///
    /// # Errors
    /// Returns an error if the pipeline call fails, or if the handle is
    /// currently null (a previous [`Self::reload`] attempt failed).
    ///
    /// # Threading
    /// Blocks the calling thread. Call only from the dedicated rerank engine thread.
    pub fn rerank(&self, query: &str, texts: &[String]) -> anyhow::Result<Vec<(usize, f32)>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if self.handle.is_null() {
            anyhow::bail!("reranking pipeline is not loaded (a previous reload attempt failed)");
        }

        let c_query = CString::new(query).context("query contains interior NUL")?;
        let c_texts: Vec<CString> = texts
            .iter()
            .map(|t| CString::new(t.as_str()).context("document contains interior NUL"))
            .collect::<anyhow::Result<_>>()?;
        let c_ptrs: Vec<*const std::ffi::c_char> = c_texts.iter().map(|s| s.as_ptr()).collect();

        // Trampoline context — collects callback deliveries.
        let mut ctx = RerankCtx {
            results: Vec::with_capacity(texts.len()),
            panicked: false,
        };

        // SAFETY: handle is valid (non-null); all pointers outlive the call;
        // callback is a plain function pointer; user_data points to ctx which
        // is exclusively owned here and not moved during the call.
        let ret = unsafe {
            ov_rerank_documents(
                self.handle,
                c_query.as_ptr(),
                c_ptrs.as_ptr(),
                c_ptrs.len(),
                Some(rerank_trampoline),
                std::ptr::addr_of_mut!(ctx).cast::<c_void>(),
            )
        };

        if ctx.panicked {
            anyhow::bail!("panic in rerank callback — result collection aborted");
        }
        if ret != 0 {
            anyhow::bail!("ov_rerank_documents failed: {}", last_error());
        }

        Ok(ctx.results)
    }

    /// Count the tokens `query` and `document` need combined (summed from two
    /// independent single-text counts — see `ov_rerank_count_tokens`'s C++
    /// doc comment for why not a true paired encoding). For the
    /// pre-inference length gate
    /// (the project's internal engineering log).
    ///
    /// # Errors
    /// Returns an error if the tokenizer call fails, or if the handle is
    /// currently null (a previous [`Self::reload`] attempt failed).
    pub fn count_tokens(&self, query: &str, document: &str) -> anyhow::Result<usize> {
        if self.handle.is_null() {
            anyhow::bail!("reranking pipeline is not loaded (a previous reload attempt failed)");
        }
        let c_query = CString::new(query).context("query contains interior NUL")?;
        let c_document = CString::new(document).context("document contains interior NUL")?;
        // SAFETY: handle is non-null; both CStrings outlive the call.
        let count =
            unsafe { ov_rerank_count_tokens(self.handle, c_query.as_ptr(), c_document.as_ptr()) };
        if count == usize::MAX {
            anyhow::bail!("ov_rerank_count_tokens failed: {}", last_error());
        }
        Ok(count)
    }
}

impl Drop for OvRerankEngine {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle is valid; we own it exclusively; called from the
            // engine thread (the only thread that touches this pointer).
            unsafe { ov_rerank_free(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

// ── Callback trampoline ───────────────────────────────────────────────────────

struct RerankCtx {
    results: Vec<(usize, f32)>,
    panicked: bool,
}

unsafe extern "C" fn rerank_trampoline(user_data: *mut c_void, index: usize, score: f32) {
    // SAFETY: user_data is &mut RerankCtx cast to *mut c_void. Alive for the
    // full ov_rerank_documents call and exclusively owned by that call.
    let ctx = unsafe { &mut *(user_data.cast::<RerankCtx>()) };
    if std::panic::catch_unwind(AssertUnwindSafe(|| {
        ctx.results.push((index, score));
    }))
    .is_err()
    {
        ctx.panicked = true;
    }
}
