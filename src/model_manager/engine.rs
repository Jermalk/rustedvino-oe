// ============================================================
// src/model_manager/engine.rs — polymorphic engine seam
// ============================================================
// R0 of the engine-registry refactor. Introduces the *management*
// abstraction every engine kind will implement, decoupling
// `ModelManager` (the single registry / lifecycle / VRAM / metrics
// concern) from *execution* (the concrete pipeline, which stays typed
// and split per kind — a technical necessity).
//
// R0 scope: only the text-generation (CB) engine exists, so the enums
// carry a single `TextGen` variant. R1 folds in `Vision` (VLM); R3
// scaffolds the media kinds. Adding a variant later turns every
// `match` here non-exhaustive — the compiler then *forces* the
// routing / lifecycle edits that phase requires. That is the whole
// point of the seam: a new kind is "add a variant + a factory arm,"
// never a silent fallthrough.
//
// CRASH COURSE — why an enum AND a trait?
//   - `ManagedEngine` (trait): the UNIFORM concern. The observability
//     and lifecycle facts the manager reads identically for any kind
//     (kind / active / max_concurrency). Sync — no `async-trait` dep,
//     so the hot path stays untouched.
//   - `EngineHandleKind` (enum): the TYPED dispatch. Execution differs
//     per kind (CB `add_request(prompt)` vs VLM `generate(msgs,imgs)`),
//     so the handler matches the enum and calls the concrete async
//     method on the inner handle. Async lives on the concrete handle.
// ============================================================

use crate::cb_engine::EngineHandle;
use crate::embed_engine::EmbeddingHandle;
use crate::npu_engine::NpuHandle;
use crate::pipelines::image::ImageHandle;
use crate::pipelines::stt::SttHandle;
use crate::pipelines::tts::TtsHandle;
use crate::rerank_engine::RerankingHandle;
use crate::vlm_engine::VlmHandle;

/// What a model *does* — drives request routing, validation, the health
/// probe shape, and metrics labels.
///
/// R0 defined only [`ModelKind::TextGen`]; R1 folds in [`ModelKind::Vision`]
/// (the VLM). Later phases extend this further (`Embedding`/`Stt`/`Tts`/
/// `ImageGen` in R3). The kind stays an INTERNAL concept — it is never a
/// `/v1/models` wire field (`OpenAI`'s schema has no `kind`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ModelKind {
    /// Text generation (an LLM) served by the continuous-batching engine.
    TextGen,
    /// Vision-language generation (a VLM) served by the single-stream
    /// `VLMPipeline` engine. A VLM also serves text-only requests.
    Vision,
    /// Text embeddings (`/v1/embeddings`, the G3 compat flow). R3 seam.
    Embedding,
    /// Speech-to-text (Whisper). R3 seam.
    Stt,
    /// Text-to-speech (Kokoro / Piper). R3 seam.
    Tts,
    /// Text-to-image (`Text2Image`). R3 seam.
    ImageGen,
    /// Cross-encoder reranking (`/v1/rerank`).
    Reranking,
}

impl ModelKind {
    /// Stable lowercase identifier for the `kind` Prometheus label and
    /// structured-log fields. Internal observability only — never a
    /// `/v1/models` wire field. Kept as a `match` so a future variant forces
    /// an explicit label here rather than silently defaulting.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::TextGen => "text_gen",
            Self::Vision => "vision",
            Self::Embedding => "embedding",
            Self::Stt => "stt",
            Self::Tts => "tts",
            Self::ImageGen => "image_gen",
            Self::Reranking => "reranking",
        }
    }

    /// Parse a [`label`](Self::label) string back into a kind. Used to interpret
    /// the explicit `model_kinds` config hints. Returns `None` for an
    /// unrecognised value (the caller falls back to directory detection).
    #[must_use]
    pub fn from_label(s: &str) -> Option<Self> {
        match s {
            "text_gen" => Some(Self::TextGen),
            "vision" => Some(Self::Vision),
            "embedding" => Some(Self::Embedding),
            "stt" => Some(Self::Stt),
            "tts" => Some(Self::Tts),
            "image_gen" => Some(Self::ImageGen),
            "reranking" => Some(Self::Reranking),
            _ => None,
        }
    }
}

/// The management / observability facade EVERY engine kind implements.
///
/// This is the *uniform* half of the split: whatever the underlying
/// pipeline, `ModelManager` reads these the same way for metrics and
/// lifecycle bookkeeping. It is deliberately **sync** — async execution
/// lives on the concrete handle reached through [`EngineHandleKind`], so
/// no `async-trait` dependency and no change to the token hot path.
pub trait ManagedEngine: Send + Sync {
    /// The kind of model this engine serves (routing + metrics label).
    fn kind(&self) -> ModelKind;
    /// Accepted-but-unfinished requests right now (`rustedvino_requests_running`).
    fn active(&self) -> usize;
    /// Configured concurrency cap (`rustedvino_requests_max`).
    fn max_concurrency(&self) -> usize;
    /// Requests admitted but not yet actively processed
    /// (`rustedvino_requests_waiting`): CB counts callers parked at the
    /// admission semaphore; the single-stream VLM counts queued-but-not-running.
    ///
    /// Defaults to `0` for kinds with no waiting concept (e.g. a future
    /// synchronous engine), so adding a kind never silently mis-reports a queue.
    fn waiting(&self) -> usize {
        0
    }
    /// Net measured device-memory change (bytes) across this engine's requests
    /// since load — working memory the runtime keeps between requests, which
    /// the config-declared `vram_gb` doesn't see. Only the embedding engine
    /// measures it today (its batch cache is the measured case); `Some(0)`
    /// elsewhere. `None` = unknown for this load: a request's measurement
    /// overlapped another model's load or eviction.
    fn runtime_memory_growth_bytes(&self) -> Option<i64> {
        Some(0)
    }
    /// Live KV-cache pool occupancy as a percentage (0–100) from the engine's
    /// last step (`rustedvino_kv_cache_usage_percent`, co-residency Slice 3a).
    ///
    /// Defaults to `0.0`. For most kinds (media/embedding/reranking/NPU) this
    /// is honestly correct — no batched KV pool exists to report on. The VLM
    /// engine (`vlm_engine.rs`) is the one exception: `VLMPipeline` runs on a
    /// `ContinuousBatchingPipeline` internally (`VLMContinuousBatchingAdapter`,
    /// confirmed via a live gdb backtrace, 2026-08-20) exactly like the plain
    /// text-gen path, and does have a real batched KV pool — but its `0.0`
    /// here is NOT the true occupancy, just the unimplemented default: `ov::
    /// genai::VLMPipeline`'s public C++ API exposes no `get_metrics()`
    /// equivalent, and `VLMContinuousBatchingAdapter` is a private `pimpl`
    /// class, unreachable without reaching into private internals (unsound —
    /// not attempted). Confirmed live against a real running server
    /// (the project's internal engineering log): tokens
    /// flowing, real throughput/duration histograms, `kv_cache_usage_percent`
    /// pinned at 0 throughout. Not fixable from `RustedVINO`'s side without an
    /// upstream `OpenVINO` `GenAI` API addition. See
    /// [`Self::cache_usage_supported`] for how a scraper tells this apart
    /// from a genuinely idle pool. Only the CB engine overrides this today.
    fn cache_usage_pct(&self) -> f64 {
        0.0
    }

    /// Whether [`Self::cache_usage_pct`] reflects a real, trustworthy live
    /// reading for this engine instance (`rustedvino_kv_cache_usage_supported`)
    /// — as opposed to a structurally-unmeasurable `0.0` that would otherwise
    /// be indistinguishable from a genuinely empty pool. Defaults to `false`
    /// (safe: never silently claims a number is trustworthy when unverified).
    /// `true` only where the engine can actually query the underlying
    /// pipeline's real occupancy — today, only the plain CB path
    /// (`cb_engine::EngineHandle`). The VLM engine deliberately does NOT
    /// override this to `true`: it has a real pool (see
    /// [`Self::cache_usage_pct`]'s doc comment) but no way to query it, so its
    /// `0.0` is exactly the kind of value this flag exists to flag as
    /// unmeasurable rather than empty.
    fn cache_usage_supported(&self) -> bool {
        false
    }

    /// Wall-clock age of the longest currently in-flight generation, if any.
    ///
    /// Watchdog signal only (not a metric): the supervisor polls this
    /// (`ModelManager::oldest_generation_age`, aggregated across every Ready
    /// model) to detect a generation that has been running far longer than
    /// any legitimate request could — e.g. an `OpenVINO` FFI call spinning
    /// forever after a GPU driver engine-reset it never surfaced as an error
    /// (reproduced 2026-07-28, the project's internal engineering log).
    /// `/health` cannot see this: it stays `200 ok` the whole time a single
    /// engine thread is wedged, since the HTTP listener itself is unaffected.
    ///
    /// Defaults to `None` for kinds that don't track it yet — an engine that
    /// doesn't override this is simply invisible to the watchdog, never
    /// falsely reported as hung.
    fn oldest_generation_age(&self) -> Option<std::time::Duration> {
        None
    }
}

/// Typed dispatch handle — the value `ModelManager` stores per Ready model
/// and the chat handler matches on to reach the concrete engine API.
///
/// R1 carries [`EngineHandleKind::TextGen`] and [`EngineHandleKind::Vision`].
/// Cloning is O(1) (the inner handle clones a `Sender` + `Arc`s).
#[derive(Clone, Debug)]
pub enum EngineHandleKind {
    /// The continuous-batching text-generation engine (`cb_engine`).
    TextGen(EngineHandle),
    /// The single-stream NPU text-generation engine (`npu_engine`).
    /// Uses the static `LLMPipeline` (not CB); requires a channel-wise int4 IR.
    /// `ModelKind` is still `TextGen` — NPU is a placement/execution distinction.
    NpuTextGen(NpuHandle),
    /// The single-stream vision-language engine (`vlm_engine`).
    Vision(VlmHandle),
    /// The single-stream text-embedding engine (`embed_engine`). R4/G3: a real
    /// pipeline-backed handle (promoted from the R3 stub).
    Embedding(EmbeddingHandle),
    /// The single-stream speech-to-text engine (`pipelines::stt`). Phase 5.1c: a
    /// real pipeline-backed handle (promoted from the R3 stub).
    Stt(SttHandle),
    /// Text-to-speech engine — R3 seam.
    Tts(TtsHandle),
    /// Text-to-image engine — R3 seam.
    ImageGen(ImageHandle),
    /// Cross-encoder reranking engine (`/v1/rerank`).
    Reranking(RerankingHandle),
}

impl EngineHandleKind {
    /// The [`ModelKind`] of the wrapped engine.
    ///
    /// Kept as a `match` (not a constant) so that adding a variant in a
    /// later phase makes this non-exhaustive and the compiler flags the
    /// new arm — the seam's forcing function.
    #[must_use]
    pub fn kind(&self) -> ModelKind {
        match self {
            Self::TextGen(_) | Self::NpuTextGen(_) => ModelKind::TextGen,
            Self::Vision(_) => ModelKind::Vision,
            Self::Embedding(_) => ModelKind::Embedding,
            Self::Stt(_) => ModelKind::Stt,
            Self::Tts(_) => ModelKind::Tts,
            Self::ImageGen(_) => ModelKind::ImageGen,
            Self::Reranking(_) => ModelKind::Reranking,
        }
    }

    /// Ask the underlying engine to stop promptly, aborting in-flight work.
    ///
    /// Used ONLY on the process-shutdown path (not normal admin/LRU eviction,
    /// which is documented to drain in-flight work first): it flips the engine's
    /// shutdown flag so an in-flight request cannot pin the eviction join for the
    /// whole generation (#7). Only the continuous-batching text-generation engine
    /// implements an explicit flag today; the single-stream kinds rely on the
    /// command-channel drop plus the manager's bounded join to stop in time.
    pub fn request_shutdown(&self) {
        if let Self::TextGen(h) = self {
            h.request_shutdown();
        }
        // NpuTextGen: no explicit shutdown flag — channel drop on thread exit suffices.
    }

    /// Borrow the engine as its uniform [`ManagedEngine`] facade — what the
    /// manager uses for metrics and lifecycle, agnostic to the concrete kind.
    #[must_use]
    pub fn as_managed(&self) -> &dyn ManagedEngine {
        match self {
            Self::TextGen(h) => h,
            Self::NpuTextGen(h) => h,
            Self::Vision(h) => h,
            Self::Embedding(h) => h,
            Self::Stt(h) => h,
            Self::Tts(h) => h,
            Self::ImageGen(h) => h,
            Self::Reranking(h) => h,
        }
    }
}

// ============================================================
// Unit tests
// ============================================================
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::cb_engine::{EngineCommand, EngineHandle};

    /// Build a `TextGen` handle wrapping a fresh (idle) CB engine handle.
    /// `from_sender` uses a test cap of 1000 and no in-flight requests.
    fn text_gen_handle() -> EngineHandleKind {
        let (tx, _rx) = tokio::sync::mpsc::channel::<EngineCommand>(1);
        EngineHandleKind::TextGen(EngineHandle::from_sender(tx))
    }

    /// Each kind exposes a distinct, stable lowercase metrics/log label.
    #[test]
    fn model_kind_labels_are_distinct_and_stable() {
        assert_eq!(ModelKind::TextGen.label(), "text_gen");
        assert_eq!(ModelKind::Vision.label(), "vision");
        assert_eq!(ModelKind::Embedding.label(), "embedding");
        assert_eq!(ModelKind::Stt.label(), "stt");
        assert_eq!(ModelKind::Tts.label(), "tts");
        assert_eq!(ModelKind::ImageGen.label(), "image_gen");
        assert_eq!(ModelKind::Reranking.label(), "reranking");

        let labels = [
            ModelKind::TextGen.label(),
            ModelKind::Vision.label(),
            ModelKind::Embedding.label(),
            ModelKind::Stt.label(),
            ModelKind::Tts.label(),
            ModelKind::ImageGen.label(),
            ModelKind::Reranking.label(),
        ];
        let unique: std::collections::BTreeSet<&str> = labels.iter().copied().collect();
        assert_eq!(unique.len(), labels.len(), "labels must all be distinct");
    }

    /// The enum reports its kind directly and through the `ManagedEngine` facade.
    #[test]
    fn engine_handle_kind_reports_text_gen() {
        let handle = text_gen_handle();
        assert_eq!(handle.kind(), ModelKind::TextGen);
        assert_eq!(handle.as_managed().kind(), ModelKind::TextGen);
    }

    /// `as_managed()` mirrors the inherent CB accessors: a fresh handle has no
    /// active requests and exposes the test concurrency cap.
    #[test]
    fn managed_facade_exposes_active_and_max_concurrency() {
        let handle = text_gen_handle();
        let managed = handle.as_managed();
        assert_eq!(
            managed.active(),
            0,
            "fresh engine has no in-flight requests"
        );
        assert_eq!(
            managed.max_concurrency(),
            1000,
            "test handle uses the 1000-permit cap"
        );
        assert_eq!(
            managed.waiting(),
            0,
            "fresh engine has no callers parked at the admission gate"
        );
    }

    /// Build a `Vision` handle wrapping a fresh (idle) VLM handle.
    fn vision_handle() -> EngineHandleKind {
        let (tx, _rx) = tokio::sync::mpsc::channel::<crate::vlm_engine::VlmCommand>(8);
        EngineHandleKind::Vision(VlmHandle::from_sender(tx, "test-vlm"))
    }

    /// The Vision arm reports its kind both directly and through the facade.
    #[test]
    fn engine_handle_kind_reports_vision() {
        let handle = vision_handle();
        assert_eq!(handle.kind(), ModelKind::Vision);
        assert_eq!(handle.as_managed().kind(), ModelKind::Vision);
    }

    /// A fresh VLM handle has no in-flight requests and exposes its cap.
    /// `VlmHandle::from_sender` uses a generous test cap of 1000 (mirroring
    /// `EngineHandle::from_sender`'s CB equivalent) — production's real
    /// default is [`crate::vlm_engine::VLM_CHANNEL_CAP`], asserted instead via
    /// `spawn_mock_vlm` in `test_metrics_snapshot_includes_vision_model`.
    #[test]
    fn vision_facade_exposes_active_and_max_concurrency() {
        let handle = vision_handle();
        let managed = handle.as_managed();
        assert_eq!(managed.active(), 0, "fresh VLM has no in-flight requests");
        assert_eq!(
            managed.max_concurrency(),
            1000,
            "from_sender's test cap, not the production default"
        );
        assert_eq!(managed.waiting(), 0, "idle VLM has nothing queued");
    }
}
