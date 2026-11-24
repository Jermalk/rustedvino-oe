// ============================================================
// src/ov_rerank.rs — Rust FFI wrapper for TextRerankPipeline
// ============================================================
// Mirrors the structure of ov_embed.rs. The C bridge exposes
// ov_rerank_create / ov_rerank_documents / ov_rerank_free (the same
// opaque-handle + callback pattern as the embedding bridge).
//
// The bridge scores each document with its own one-document rerank() call
// (C27: batched calls hit an upstream heap corruption, see
// ov_rerank_documents in ov_bridge.cpp), maps every score back to the
// document's position and sorts the set; `checked_rerank_order` re-checks
// that here. The pipeline is created with a large top_n so nothing is
// dropped; Rust applies the per-request top_n trim.
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

        checked_rerank_order(ctx.results, texts.len())
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

// ── Result check ──────────────────────────────────────────────────────────────

/// Check the bridge's `(original_index, score)` results for `n` documents and
/// return them sorted by score descending (equal scores: lower index first; a
/// NaN score last) — the order the bridge already delivers, so this sort is a
/// no-op on a correct result.
///
/// C27: the bridge scores one document per pipeline call, and the pipeline
/// numbers results within its own call (always index 0). A mapping slip there
/// would hand every result index 0 — the response would then repeat
/// `documents[0]`'s text under every score, with no error. So the indices must
/// be exactly a permutation of `0..n`; anything else is an error, never a
/// response.
///
/// # Errors
/// Returns an error when the result count is not `n`, or an index is out of
/// range or repeated.
pub(crate) fn checked_rerank_order(
    mut results: Vec<(usize, f32)>,
    n: usize,
) -> anyhow::Result<Vec<(usize, f32)>> {
    if results.len() != n {
        anyhow::bail!(
            "reranking returned {} results for {n} documents",
            results.len()
        );
    }
    let mut seen = vec![false; n];
    for &(index, _) in &results {
        match seen.get_mut(index) {
            Some(slot) if !*slot => *slot = true,
            Some(_) => anyhow::bail!("reranking returned document index {index} twice"),
            None => anyhow::bail!("reranking returned document index {index} for {n} documents"),
        }
    }
    let key = |score: f32| {
        if score.is_nan() {
            f32::NEG_INFINITY
        } else {
            score
        }
    };
    results.sort_by(|a, b| {
        key(b.1)
            .partial_cmp(&key(a.1))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    Ok(results)
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::checked_rerank_order;

    /// The C27 trap: one document per pipeline call, and every call's result
    /// says index 0. Passed through, the response would label every score with
    /// `documents[0]`. It must be an error instead.
    #[test]
    fn every_document_as_index_0_is_rejected() {
        let all_zero = vec![(0, 0.9), (0, 0.5), (0, 0.1)];
        let err = checked_rerank_order(all_zero, 3).unwrap_err().to_string();
        assert!(err.contains("index 0 twice"), "{err}");
    }

    #[test]
    fn per_document_scores_come_back_mapped_and_sorted_descending() {
        // Delivered in document order (as a bridge without its sort would).
        let in_doc_order = vec![(0, 0.2), (1, 0.9), (2, 0.05), (3, 0.6)];
        let out = checked_rerank_order(in_doc_order, 4).unwrap();
        assert_eq!(out, vec![(1, 0.9), (3, 0.6), (0, 0.2), (2, 0.05)]);
    }

    #[test]
    fn an_already_sorted_result_is_unchanged() {
        let sorted = vec![(2, 0.99), (0, 0.5), (1, 0.5), (3, 0.0)];
        assert_eq!(checked_rerank_order(sorted.clone(), 4).unwrap(), sorted);
    }

    #[test]
    fn equal_scores_keep_the_lower_index_first() {
        let ties = vec![(0, 0.5), (1, 0.7), (2, 0.5), (3, 0.5)];
        let out = checked_rerank_order(ties, 4).unwrap();
        assert_eq!(out, vec![(1, 0.7), (0, 0.5), (2, 0.5), (3, 0.5)]);
    }

    #[test]
    fn equal_scores_delivered_out_of_index_order_are_reordered_by_index() {
        let ties = vec![(3, 0.5), (1, 0.7), (2, 0.5), (0, 0.5)];
        let out = checked_rerank_order(ties, 4).unwrap();
        assert_eq!(out, vec![(1, 0.7), (0, 0.5), (2, 0.5), (3, 0.5)]);
    }

    #[test]
    fn a_nan_score_sorts_last() {
        let out = checked_rerank_order(vec![(0, f32::NAN), (1, 0.1)], 2).unwrap();
        assert_eq!(out[0], (1, 0.1));
        assert_eq!(out[1].0, 0);
        assert!(out[1].1.is_nan());
    }

    #[test]
    fn a_missing_or_out_of_range_index_is_rejected() {
        assert!(checked_rerank_order(vec![(0, 0.5)], 2).is_err());
        assert!(checked_rerank_order(vec![(0, 0.5), (2, 0.4)], 2).is_err());
        assert!(checked_rerank_order(vec![(0, 0.5), (1, 0.4), (1, 0.3)], 2).is_err());
    }

    #[test]
    fn no_documents_is_an_empty_result() {
        assert!(checked_rerank_order(Vec::new(), 0).unwrap().is_empty());
    }
}
