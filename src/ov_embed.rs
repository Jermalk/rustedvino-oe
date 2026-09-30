// ============================================================
// src/ov_embed.rs — Safe Rust wrapper around the embedding C-API
// ============================================================
// Wraps `ov_embed_create` / `ov_embed_documents` / `ov_embed_count_tokens` /
// `ov_embed_free` from `ov_bridge/ov_bridge.cpp` in a safe Rust API.
//
// THREADING:
//   TextEmbeddingPipeline is single-stream and blocking (same contract as the
//   CB and VLM engines). OvEmbedEngine is !Sync; it is moved onto a dedicated
//   OS thread at spawn time and never aliased. Send is implemented manually
//   because we own the raw pointer exclusively.
// ============================================================

use std::ffi::{CString, c_void};
use std::ptr;

use anyhow::Context;

use crate::ov_pipeline::last_error;

// ── Pooling strategy ──────────────────────────────────────────────────────────

/// Pooling strategy applied to the encoder output before (optional) L2
/// normalization. Codes match `TextEmbeddingPipeline::PoolingType` in the
/// bridge: `0=CLS`, `1=MEAN`, `2=LAST_TOKEN`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pooling {
    /// First-token (`[CLS]`) embedding.
    Cls,
    /// Mean of all token embeddings — the e5 / sentence-transformers default.
    Mean,
    /// Last-token embedding.
    LastToken,
}

impl Pooling {
    /// Parse a config string (`"cls"`/`"mean"`/`"last_token"`); unknown → MEAN.
    #[must_use]
    pub fn from_config(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "cls" => Self::Cls,
            "last_token" | "last" => Self::LastToken,
            _ => Self::Mean,
        }
    }

    /// The integer code passed across the FFI boundary.
    fn code(self) -> std::ffi::c_int {
        match self {
            Self::Cls => 0,
            Self::Mean => 1,
            Self::LastToken => 2,
        }
    }
}

// ── Embed callback (matches OvEmbedCallback in ov_bridge.cpp) ─────────────────
//
// Called once per embedded document with its index and float vector.
type OvEmbedCallback =
    unsafe extern "C" fn(user_data: *mut c_void, index: usize, vec: *const f32, dim: usize);

// ── FFI declarations ─────────────────────────────────────────────────────────

unsafe extern "C" {
    fn ov_embed_create(
        model_path: *const std::ffi::c_char,
        device: *const std::ffi::c_char,
        pooling_type: std::ffi::c_int,
        normalize: std::ffi::c_int,
    ) -> *mut c_void;

    fn ov_embed_free(handle: *mut c_void);

    fn ov_embed_documents(
        handle: *mut c_void,
        texts: *const *const std::ffi::c_char,
        n: usize,
        callback: Option<OvEmbedCallback>,
        user_data: *mut c_void,
    ) -> std::ffi::c_int;

    fn ov_embed_count_tokens(handle: *mut c_void, text: *const std::ffi::c_char) -> usize;
}

// ── OvEmbedEngine ─────────────────────────────────────────────────────────────

/// A loaded `TextEmbeddingPipeline` — the R4 text-embedding engine.
///
/// Lives on a single dedicated OS thread. All calls must be from that same
/// thread. `drop` frees the pipeline and its GPU memory synchronously.
pub struct OvEmbedEngine {
    handle: *mut c_void,
}

// SAFETY: OvEmbedEngine owns the TextEmbeddingPipeline exclusively. The raw
// pointer is moved onto the engine thread at spawn time and never shared or
// aliased. We deliberately do NOT implement Sync.
unsafe impl Send for OvEmbedEngine {}

impl OvEmbedEngine {
    /// Load the embedding model at `model_path` on `device`.
    ///
    /// # Errors
    /// Returns an error if the model dir is invalid, the device is unavailable,
    /// or pipeline construction fails (OOM, unsupported hardware, etc.).
    ///
    /// # Performance
    /// Loading compiles the encoder for the target device — call from
    /// `spawn_blocking` or the dedicated engine thread, never the async runtime.
    pub fn new(
        model_path: &str,
        device: &str,
        pooling: Pooling,
        normalize: bool,
    ) -> anyhow::Result<Self> {
        let c_path = CString::new(model_path).context("model_path contains interior NUL")?;
        let c_device = CString::new(device).context("device contains interior NUL")?;

        // SAFETY: CStrings outlive the call; ov_embed_create returns a valid
        // heap-allocated OvEmbedState or NULL on failure.
        let handle = unsafe {
            ov_embed_create(
                c_path.as_ptr(),
                c_device.as_ptr(),
                pooling.code(),
                std::ffi::c_int::from(normalize),
            )
        };

        if handle.is_null() {
            anyhow::bail!("ov_embed_create failed: {}", last_error());
        }

        tracing::info!(
            model = %model_path,
            device = %device,
            ?pooling,
            normalize,
            "embedding engine loaded"
        );
        Ok(Self { handle })
    }

    /// Embed a batch of documents, returning one float vector per input (in
    /// order). An empty `texts` returns an empty `Vec` without an FFI call.
    ///
    /// # Errors
    /// Returns an error if a text contains an interior NUL, or the C++ pipeline
    /// call fails.
    ///
    /// # Threading
    /// Blocks the calling thread for the full batch. Call only from the
    /// dedicated embed engine thread.
    pub fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // CStrings must outlive the FFI call; the ptr array borrows from them.
        let c_texts: Vec<CString> = texts
            .iter()
            .map(|t| CString::new(t.as_str()).context("embedding input contains NUL byte"))
            .collect::<anyhow::Result<_>>()?;
        let ptrs: Vec<*const std::ffi::c_char> = c_texts.iter().map(|c| c.as_ptr()).collect();

        // Pre-sized collector; the trampoline places each vector by its index.
        // `panicked` records a panic caught by the trampoline's firewall so it
        // can be surfaced as an error after the blocking call returns.
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); texts.len()];
        let mut ctx = EmbedCbCtx {
            out: &mut out,
            filled: 0,
            panicked: false,
        };
        let user_data = std::ptr::addr_of_mut!(ctx).cast::<c_void>();

        // SAFETY: handle is non-null (invariant); ptrs + CStrings are alive for
        // the call; the trampoline matches OvEmbedCallback; user_data points to
        // a live EmbedCbCtx for the call's duration.
        let ret = unsafe {
            ov_embed_documents(
                self.handle,
                ptrs.as_ptr(),
                ptrs.len(),
                Some(embed_trampoline),
                user_data,
            )
        };
        if ret != 0 {
            anyhow::bail!("ov_embed_documents failed: {}", last_error());
        }
        // Copy the flags out of `ctx` before touching `out` — `ctx` holds a
        // `&mut out`, so its borrow must end (last use here) before `out` is
        // read by the validator below or moved into `Ok`.
        let filled = ctx.filled;
        let panicked = ctx.panicked;
        if panicked {
            anyhow::bail!("embedding collector callback panicked");
        }
        // T7.2: the bridge returns 0 even when the pipeline yields fewer rows
        // than requested. Reject any short, empty, or ragged batch here so a
        // partial result is never presented as a successful 200.
        validate_embed_output(&out, filled, texts.len())?;
        Ok(out)
    }

    /// Count the tokens `text` encodes to (for `usage.prompt_tokens`).
    ///
    /// # Errors
    /// Returns an error if the tokenizer call fails.
    pub fn count_tokens(&self, text: &str) -> anyhow::Result<usize> {
        let c_text = CString::new(text).context("text contains interior NUL")?;
        // SAFETY: handle is non-null; c_text outlives the call.
        let count = unsafe { ov_embed_count_tokens(self.handle, c_text.as_ptr()) };
        if count == usize::MAX {
            anyhow::bail!("ov_embed_count_tokens failed: {}", last_error());
        }
        Ok(count)
    }
}

/// Everything [`embed_trampoline`] needs, passed through `user_data` for one
/// `ov_embed_documents` call.
struct EmbedCbCtx<'a> {
    /// Pre-sized collector; the trampoline places each vector by its index.
    out: &'a mut Vec<Vec<f32>>,
    /// Number of rows the bridge actually delivered (one increment per vector
    /// successfully placed). A shortfall vs the input count means the pipeline
    /// returned fewer rows than requested — caught as a data-integrity error
    /// rather than surfaced as empty vectors in a 200 (T7.2).
    filled: usize,
    /// Set when the collector panicked — `embed` turns the flag into an error
    /// (never a partial-success 200).
    panicked: bool,
}

/// C callback trampoline: places one document's float vector into the
/// collector, under a panic firewall.
///
/// # Safety
/// Called by C inside `ov_embed_documents`. Invariants upheld by
/// `OvEmbedEngine::embed`:
/// - `user_data` is a valid `*mut EmbedCbCtx` on the caller's stack,
///   alive (and correctly sized) for the whole blocking call.
/// - `vec` points to `dim` floats, valid for the duration of this call.
/// - `index` is in bounds (`< texts.len()`), guaranteed by the bridge iterating
///   over the input documents in order.
unsafe extern "C" fn embed_trampoline(
    user_data: *mut c_void,
    index: usize,
    vec: *const f32,
    dim: usize,
) {
    // SAFETY: user_data is the context pointer set up in OvEmbedEngine::embed.
    let ctx = unsafe { &mut *user_data.cast::<EmbedCbCtx>() };
    if ctx.panicked {
        return;
    }
    let slice = if dim == 0 || vec.is_null() {
        &[][..]
    } else {
        // SAFETY: vec points to `dim` floats, alive until this callback returns.
        unsafe { std::slice::from_raw_parts(vec, dim) }
    };
    // PANIC FIREWALL: a panic (e.g. allocation failure in `to_vec`) must never
    // unwind out of this extern "C" frame — that is a guaranteed process abort.
    // AssertUnwindSafe is sound: on panic the whole batch is discarded by
    // `embed` (error, never a partial result).
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if let Some(slot) = ctx.out.get_mut(index) {
            *slot = slice.to_vec();
            true
        } else {
            // Out-of-range index — a bridge contract violation. The slot is not
            // placed; the `filled != n` check in `embed` surfaces the shortfall.
            false
        }
    }));
    match result {
        Ok(true) => ctx.filled += 1,
        Ok(false) => {}
        Err(payload) => {
            tracing::error!(
                index,
                panic = crate::ov_pipeline::panic_message(payload.as_ref()),
                "embedding collector callback panicked"
            );
            ctx.panicked = true;
        }
    }
}

/// Data-integrity gate for an embedding batch (T7.2). The C bridge reports
/// success even when the pipeline returns fewer rows than requested, which
/// would otherwise leave unfilled slots as empty vectors in a 200. Reject any
/// batch that is short, has an empty (zero-dim/unfilled) row, or has ragged
/// dimensions — a corrupt result must be an error, never a partial 200.
///
/// `filled` is the number of rows the bridge actually delivered; `expected` is
/// the input count.
///
/// # Errors
/// Returns an error if `filled != expected`, any row is empty, or rows have
/// inconsistent dimensions.
fn validate_embed_output(out: &[Vec<f32>], filled: usize, expected: usize) -> anyhow::Result<()> {
    if filled != expected {
        anyhow::bail!(
            "embedding pipeline delivered {filled} rows for {expected} inputs — short-row response"
        );
    }
    if let Some(i) = out.iter().position(Vec::is_empty) {
        anyhow::bail!(
            "embedding row {i} is empty (zero-dim or unfilled) — corrupt embedding output"
        );
    }
    let Some(first) = out.first() else {
        return Ok(());
    };
    let dim = first.len();
    if let Some(i) = out.iter().position(|v| v.len() != dim) {
        anyhow::bail!(
            "embedding row {i} has dim {} != expected {dim} — inconsistent embedding output",
            out[i].len()
        );
    }
    Ok(())
}

/// Read `model_dir/config.json`'s `max_position_embeddings` — a BERT-family
/// model's hard token-length ceiling (a fixed position-embedding table sized
/// to this value). `OpenVINO`'s IR itself declares dynamic shapes at the graph
/// level, so nothing at the IR rejects an over-length input; feeding one to
/// `embed_documents` raises a C++ shape-inference exception instead of a clean
/// error (root-caused 2026-08-03,
/// the project's internal engineering log).
///
/// The usable length is not always the table size: RoBERTa-family models
/// (XLM-R, e.g. multilingual-e5, bge-reranker-v2-m3) offset position ids by
/// `padding_idx + 1 = 2`, so a 514-row table holds 512 tokens. Inputs of 513–514
/// tokens were *not* rejected by `OpenVINO` — they embedded silently with
/// untrained positions (measured 2026-09-29). So the ceiling is, in order: the
/// tokenizer's `model_max_length` (`tokenizer_config.json`) when it is a real
/// value no larger than the table; else the table size minus 2 for a
/// RoBERTa-family `model_type`; else the table size.
///
/// Returns `None` when `config.json` is absent, unparseable, or lacks the
/// field — fail-open, mirroring the NPU `max_prompt_len` gate's "config.json
/// absent/unparseable → skip the gate" policy: an unknown ceiling means no
/// gate, not a guessed one.
#[must_use]
pub(crate) fn resolve_max_seq_len(model_dir: &std::path::Path) -> Option<usize> {
    let read_json = |name: &str| -> Option<serde_json::Value> {
        let text = std::fs::read_to_string(model_dir.join(name)).ok()?;
        serde_json::from_str(&text).ok()
    };
    let config = read_json("config.json")?;
    let table = config
        .get("max_position_embeddings")?
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())?;
    // HF writes a huge sentinel (1e30, often as a float) when the tokenizer
    // has no limit; `as_u64` rejects floats and oversize values, and the
    // `<= table` check rejects anything else implausible.
    let tokenizer_limit = read_json("tokenizer_config.json")
        .and_then(|t| t.get("model_max_length")?.as_u64())
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n >= 1 && n <= table);
    if let Some(limit) = tokenizer_limit {
        return Some(limit);
    }
    let roberta_family = matches!(
        config.get("model_type").and_then(serde_json::Value::as_str),
        Some("xlm-roberta" | "roberta" | "camembert")
    );
    Some(if roberta_family {
        table.saturating_sub(2)
    } else {
        table
    })
}

impl Drop for OvEmbedEngine {
    /// Synchronously frees the embedding pipeline and its GPU allocations.
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle is the unique owner; this is the only free site.
            unsafe { ov_embed_free(self.handle) };
            self.handle = ptr::null_mut();
            tracing::debug!("embedding engine freed");
        }
    }
}

// ============================================================
// Unit tests (no GPU)
// ============================================================
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    // ── resolve_max_seq_len ───────────────────────────────────────────────────

    #[test]
    fn resolve_max_seq_len_reads_the_config_field() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"architectures":["BertModel"],"max_position_embeddings":512}"#,
        )
        .expect("write config.json");
        assert_eq!(resolve_max_seq_len(dir.path()), Some(512));
    }

    /// multilingual-e5's real files: table 514, tokenizer 512 → 512 (the gate
    /// used to admit 513–514, embedded silently with untrained positions).
    #[test]
    fn resolve_max_seq_len_prefers_tokenizer_limit() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type":"xlm-roberta","max_position_embeddings":514}"#,
        )
        .expect("write config.json");
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"model_max_length":512}"#,
        )
        .expect("write tokenizer_config.json");
        assert_eq!(resolve_max_seq_len(dir.path()), Some(512));
    }

    /// Without a usable tokenizer limit (absent, HF's 1e30 "unset" sentinel, or
    /// larger than the table), a RoBERTa-family table loses its 2 offset slots;
    /// a BERT table is used as-is.
    #[test]
    fn resolve_max_seq_len_roberta_offset_fallback() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = |model_type: &str, table: u32| {
            std::fs::write(
                dir.path().join("config.json"),
                format!(r#"{{"model_type":"{model_type}","max_position_embeddings":{table}}}"#),
            )
            .expect("write config.json");
        };
        cfg("xlm-roberta", 8194);
        assert_eq!(resolve_max_seq_len(dir.path()), Some(8192));
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"model_max_length":1e30}"#,
        )
        .expect("write tokenizer_config.json");
        assert_eq!(
            resolve_max_seq_len(dir.path()),
            Some(8192),
            "1e30 sentinel ignored"
        );
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"model_max_length":100000}"#,
        )
        .expect("write tokenizer_config.json");
        assert_eq!(
            resolve_max_seq_len(dir.path()),
            Some(8192),
            "> table ignored"
        );
        cfg("bert", 512);
        assert_eq!(resolve_max_seq_len(dir.path()), Some(512));
    }

    #[test]
    fn resolve_max_seq_len_none_when_config_json_missing() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert_eq!(resolve_max_seq_len(dir.path()), None);
    }

    #[test]
    fn resolve_max_seq_len_none_when_field_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"architectures":["BertModel"]}"#,
        )
        .expect("write config.json");
        assert_eq!(resolve_max_seq_len(dir.path()), None);
    }

    #[test]
    fn resolve_max_seq_len_none_when_config_json_is_corrupt() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.json"), "{not json").expect("write config.json");
        assert_eq!(resolve_max_seq_len(dir.path()), None);
    }

    #[test]
    fn pooling_parses_case_insensitively_and_defaults_to_mean() {
        assert_eq!(Pooling::from_config("cls"), Pooling::Cls);
        assert_eq!(Pooling::from_config("CLS"), Pooling::Cls);
        assert_eq!(Pooling::from_config("last_token"), Pooling::LastToken);
        assert_eq!(Pooling::from_config("mean"), Pooling::Mean);
        assert_eq!(Pooling::from_config("nonsense"), Pooling::Mean);
    }

    #[test]
    fn pooling_codes_match_bridge_contract() {
        assert_eq!(Pooling::Cls.code(), 0);
        assert_eq!(Pooling::Mean.code(), 1);
        assert_eq!(Pooling::LastToken.code(), 2);
    }

    // ── T7.2 embed data-integrity gate ───────────────────────────────────────

    #[test]
    fn validate_accepts_a_complete_batch() {
        let out = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        assert!(validate_embed_output(&out, 2, 2).is_ok());
    }

    #[test]
    fn validate_rejects_short_row_response() {
        // 3 inputs, only 2 rows delivered → error, never a partial-200.
        let out = vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![]];
        let err = validate_embed_output(&out, 2, 3).unwrap_err();
        assert!(
            err.to_string().contains("short-row"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn validate_rejects_empty_row_even_when_count_matches() {
        // The bridge delivered all rows, but one is empty (zero-dim/corrupt).
        let out = vec![vec![0.1, 0.2], vec![]];
        let err = validate_embed_output(&out, 2, 2).unwrap_err();
        assert!(
            err.to_string().contains("empty"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn validate_rejects_inconsistent_dimensions() {
        let out = vec![vec![0.1, 0.2], vec![0.3]];
        let err = validate_embed_output(&out, 2, 2).unwrap_err();
        assert!(err.to_string().contains("dim"), "unexpected message: {err}");
    }

    /// Live round-trip on a real embedding model (run with `--ignored`).
    /// Mirrors the `ov_cb` tokenizer round-trip test; needs the GPU + the
    /// `multilingual-e5-large-int8` IR on disk.
    #[test]
    #[ignore = "requires GPU + multilingual-e5-large-int8 on disk"]
    fn embed_round_trip_on_real_model() {
        let dir = "/opt/rustedvino/models/multilingual-e5-large-int8";
        let eng = OvEmbedEngine::new(dir, "GPU.0", Pooling::Mean, true)
            .or_else(|_| OvEmbedEngine::new(dir, "CPU", Pooling::Mean, true))
            .expect("load embedding model");
        let texts = vec!["hello world".to_owned(), "another sentence".to_owned()];
        let vecs = eng.embed(&texts).expect("embed");
        assert_eq!(vecs.len(), 2, "one vector per input");
        assert_eq!(vecs[0].len(), 1024, "e5-large hidden size is 1024");
        // L2-normalized → ‖v‖ ≈ 1.
        let norm: f32 = vecs[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "normalized vector, got norm {norm}"
        );
        assert!(eng.count_tokens("hello world").expect("count") > 0);
    }
}
