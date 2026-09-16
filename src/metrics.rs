// ============================================================
// src/metrics.rs — Prometheus /metrics surface (Phase 3.3)
// ============================================================
// Establishes the global `metrics` recorder + the Prometheus text
// exporter, and exposes the state gauges that read values already
// tracked elsewhere in the server (engine capacity counters, ready
// models, configured VRAM). Names mirror vLLM for ecosystem
// compatibility; everything is served same-port on :11437.
//
// TWO FAMILIES OF METRICS LIVE HERE:
//
// 1. PULL MODEL (state gauges): reflect *current* state, so they are set at
//    scrape time by `record_state()` (called from the /metrics handler)
//    rather than pushed on the request hot path.
//
// 2. PUSH MODEL (hot-path counters/histograms, Phase 3.4): tokens, tok/s,
//    TTFT, request duration. These are emitted from inside the cb-engine
//    thread's per-token loop. To keep that loop cheap, the engine caches the
//    metric *handles* once at thread start (see [`HotMetrics`]) — each
//    handle resolves its labels exactly once, so a per-token emit is a single
//    atomic op, not a registry lookup. The handles are no-ops when no global
//    recorder is installed (unit tests), so constructing `HotMetrics` is
//    always safe.
// ============================================================

use metrics::{Counter, Gauge, Histogram, Unit};
use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};

use crate::model_manager::{MetricsSnapshot, ModelKind};

// ---- Metric names (vLLM-compatible) ---------------------------------

// State gauges (pull-at-scrape, Phase 3.3).
const REQUESTS_RUNNING: &str = "rustedvino_requests_running";
const REQUESTS_MAX: &str = "rustedvino_requests_max";
const REQUESTS_WAITING: &str = "rustedvino_requests_waiting";
const MODEL_LOADED: &str = "rustedvino_model_loaded";
const VRAM_TOTAL_BYTES: &str = "rustedvino_vram_total_bytes";
const VRAM_USED_BYTES: &str = "rustedvino_vram_used_bytes";

/// `OpenVINO` compile-cache disk usage (`dev/plans/ov-cache-self-management.md`)
/// — pushed from the background sweep (`ModelManager::prune_ov_cache_if_over_budget`,
/// every `ov_cache_sweep_interval_secs`), not pulled at scrape time: computing
/// it walks every file in `ov_cache_dir` (thousands of tiny `.cl_cache` files
/// on a long-lived box), too expensive to redo on every `/metrics` scrape.
/// Reported whenever `ov_cache_dir` is configured, regardless of whether
/// `ov_cache_max_gb` bounds it — the original gap this plan set out to close
/// was "no visibility at all," which existed independently of pruning ever
/// being configured.
const OV_CACHE_BYTES: &str = "rustedvino_ov_cache_bytes";
const OV_CACHE_PRUNED_BYTES_TOTAL: &str = "rustedvino_ov_cache_pruned_bytes_total";
const OV_CACHE_PRUNED_FILES_TOTAL: &str = "rustedvino_ov_cache_pruned_files_total";

// Co-residency Slice 3a sizing/policy gauges (pull-at-scrape).
const KV_CACHE_POOL_GB: &str = "rustedvino_kv_cache_pool_gb";
const MODEL_LOAD_DURATION_SECONDS: &str = "rustedvino_model_load_duration_seconds";
const MODEL_PINNED: &str = "rustedvino_model_pinned";
const MODEL_PRIORITY: &str = "rustedvino_model_priority";
const KV_CACHE_USAGE_PERCENT: &str = "rustedvino_kv_cache_usage_percent";
const KV_CACHE_USAGE_SUPPORTED: &str = "rustedvino_kv_cache_usage_supported";

/// KV-cache pressure flag (`dev/plans/kv-cache-pressure-detection.md`) — 1
/// while a model's occupancy has been continuously at/above
/// `kv_pressure_threshold_pct` for at least `kv_pressure_sustained_secs`, 0
/// otherwise. Pull-at-scrape via `record_state`/`MetricsSnapshot`, same as
/// its sibling `KV_CACHE_USAGE_PERCENT`/`KV_CACHE_USAGE_SUPPORTED` — the
/// underlying tracker is updated by a separate periodic sweep
/// (`ModelManager::run_kv_pressure_sweep_once`), but reading its current
/// derived state is a cheap in-memory lookup, unlike `OV_CACHE_BYTES` above
/// which would require a real directory walk if pulled at scrape time.
/// Always 0 while `kv_pressure_monitor_enabled` is off.
const KV_CACHE_PRESSURE_FLAGGED: &str = "rustedvino_kv_cache_pressure_flagged";

/// L0 prompt-length gate rejections (`400 context_length_exceeded`) — pushed
/// from the handler layer (`handlers/chat.rs`'s three gates), not the engine
/// thread, since a rejection never reaches `add_request`/the hot path at all.
/// Previously invisible server-side entirely (`dev/autotest/
/// 20260823_l0_gate_rejection_observability_gap.md`).
const CONTEXT_LENGTH_EXCEEDED_TOTAL: &str = "rustedvino_context_length_exceeded_total";

/// Concurrent-admission rejections: a request whose own length is within the
/// model's L0 gate ceiling, but that ceiling assumes single-tenant pool
/// ownership — rejected instead because admitting it alongside requests
/// already in flight would overcommit the model's real KV pool capacity
/// (the project's internal engineering log's root-cause entry). Deliberately a
/// separate metric from `CONTEXT_LENGTH_EXCEEDED_TOTAL`: that one means "this
/// request's own prompt is too long," this one means "this prompt is fine
/// alone, the pool is just busy right now" — conflating them would mislead
/// an operator debugging which failure mode they're actually seeing.
const POOL_CAPACITY_REJECTED_TOTAL: &str = "rustedvino_pool_capacity_rejected_total";

/// Own-budget admission rejections: a single-slot model (`max_concurrent_streams
/// == 1`) rejecting a request whose own `prompt_tokens + max_tokens` can't fit
/// the KV pool — no other in-flight request involved (`dev/decisions/
/// decisions-048.md`'s deferred "fix 3"). Deliberately a separate metric from
/// `POOL_CAPACITY_REJECTED_TOTAL`: that one means "the pool is busy with other
/// requests right now, retry shortly" (genuinely good advice); this one means
/// "this request's own generation budget can never fit here, retrying the same
/// request will fail identically" — conflating them would give an operator (or
/// a client's retry logic) the wrong advice for this case.
const OWN_BUDGET_REJECTED_TOTAL: &str = "rustedvino_own_budget_rejected_total";

// Per-key request attribution (labeled only by a Bearer key's last 6
// characters — see `record_key_usage`'s doc comment for the full safety
// contract). Two metrics, not one: the counter alone only answers "which
// key has the highest cumulative total since boot," not "which key is
// active *right now*" — this box has no Prometheus `rate()` query running
// against it, so the gauge is what actually makes the counter interpretable
// without a TSDB.
const KEY_USAGE_TOTAL: &str = "rustedvino_key_usage_total";
const KEY_LAST_SEEN_TIMESTAMP_SECONDS: &str = "rustedvino_key_last_seen_timestamp_seconds";

// Hot-path counters/histograms (push from the engine thread, Phase 3.4).
const TOKENS_GENERATED_TOTAL: &str = "rustedvino_tokens_generated_total";
const REQUESTS_TOTAL: &str = "rustedvino_requests_total";
const TOKENS_PER_SECOND: &str = "rustedvino_tokens_per_second";
const TTFT_SECONDS: &str = "rustedvino_ttft_seconds";
const REQUEST_DURATION_SECONDS: &str = "rustedvino_request_duration_seconds";

/// STT/TTS real-time factor (processing time ÷ audio duration, dimensionless).
/// Emitted by both offline endpoints (`/v1/audio/transcriptions`,
/// `/v1/audio/speech`) and the realtime voice pipeline — both paths route
/// through the same `SttHandle`/`TtsHandle` engine threads, so one metric
/// covers both without a `path` label.
const RTF_RATIO: &str = "rustedvino_rtf_ratio";

// Realtime voice pipeline metrics (push from realtime.rs, observability pass).
const RT_CONNECTIONS_TOTAL: &str = "rustedvino_realtime_connections_total";
const RT_TURNS_TOTAL: &str = "rustedvino_realtime_turns_total";
const RT_STT_DURATION_SECONDS: &str = "rustedvino_realtime_stt_duration_seconds";
const RT_LLM_DURATION_SECONDS: &str = "rustedvino_realtime_llm_duration_seconds";
const RT_TTS_SENTENCES_TOTAL: &str = "rustedvino_realtime_tts_sentences_total";

/// Histogram buckets for time-to-first-token (seconds). TTFT is dominated by
/// prefill; sub-second is the healthy range, with a long tail for big prompts.
const TTFT_BUCKETS: &[f64] = &[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0];

/// Histogram buckets for end-to-end request duration (seconds). Spans a short
/// chat reply through a multi-thousand-token generation.
const DURATION_BUCKETS: &[f64] = &[0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0];

/// Histogram buckets for Whisper STT latency (seconds). STT is roughly linear
/// with audio length; typical single-utterance times are 0.1 s–2 s on GPU.
const RT_STT_BUCKETS: &[f64] = &[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0];

/// Histogram buckets for realtime LLM turn duration (seconds). The voice
/// pipeline terminates at first-complete-response, not at stream start, so
/// this spans decode of the full reply (typically 1 s–20 s for 512 tok cap).
const RT_LLM_BUCKETS: &[f64] = &[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0];

/// Histogram buckets for STT/TTS real-time factor. GPU-accelerated inference
/// is typically well under 1.0 (faster than real time); CPU/NPU paths can
/// approach or exceed it.
const RTF_BUCKETS: &[f64] = &[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0];

/// Install the global Prometheus recorder and register metric descriptions.
///
/// Returns a [`PrometheusHandle`] whose `render()` produces the text
/// exposition format. Call **once** at startup (the recorder is process-global).
///
/// # Errors
/// Returns an error if a global recorder is already installed.
pub fn install() -> anyhow::Result<PrometheusHandle> {
    // Render the two latency metrics as Prometheus *histograms* (bucketed)
    // rather than the exporter's default *summary* (quantiles): buckets are
    // aggregatable across replicas and are what vLLM dashboards expect.
    let handle = PrometheusBuilder::new()
        .set_buckets_for_metric(Matcher::Full(TTFT_SECONDS.to_owned()), TTFT_BUCKETS)
        .map_err(|e| anyhow::anyhow!("failed to set TTFT buckets: {e}"))?
        .set_buckets_for_metric(
            Matcher::Full(REQUEST_DURATION_SECONDS.to_owned()),
            DURATION_BUCKETS,
        )
        .map_err(|e| anyhow::anyhow!("failed to set duration buckets: {e}"))?
        .set_buckets_for_metric(
            Matcher::Full(RT_STT_DURATION_SECONDS.to_owned()),
            RT_STT_BUCKETS,
        )
        .map_err(|e| anyhow::anyhow!("failed to set realtime STT buckets: {e}"))?
        .set_buckets_for_metric(
            Matcher::Full(RT_LLM_DURATION_SECONDS.to_owned()),
            RT_LLM_BUCKETS,
        )
        .map_err(|e| anyhow::anyhow!("failed to set realtime LLM buckets: {e}"))?
        .set_buckets_for_metric(Matcher::Full(RTF_RATIO.to_owned()), RTF_BUCKETS)
        .map_err(|e| anyhow::anyhow!("failed to set RTF buckets: {e}"))?
        .install_recorder()
        .map_err(|e| anyhow::anyhow!("failed to install Prometheus recorder: {e}"))?;
    register_metrics();
    Ok(handle)
}

/// Register `# HELP`/`# TYPE` metadata for every metric.
fn register_metrics() {
    register_state_gauges();
    register_hot_path_metrics();
    register_key_usage_metrics();
    register_ov_cache_metrics();
}

/// OV compile-cache disk usage — see [`OV_CACHE_BYTES`]'s doc comment.
fn register_ov_cache_metrics() {
    describe_gauge!(
        OV_CACHE_BYTES,
        Unit::Bytes,
        "Total on-disk size of ov_cache_dir (OpenVINO's compiled-blob cache plus its OpenCL/oneDNN kernel-cache files), as of the last background sweep. Absent entirely when ov_cache_dir is not configured; present but static between sweeps otherwise, regardless of whether ov_cache_max_gb bounds it"
    );
    describe_counter!(
        OV_CACHE_PRUNED_BYTES_TOTAL,
        Unit::Bytes,
        "Cumulative bytes reclaimed by ov_cache pruning since server start. Only increments when ov_cache_max_gb is configured and the cache was over budget at a sweep"
    );
    describe_counter!(
        OV_CACHE_PRUNED_FILES_TOTAL,
        Unit::Count,
        "Cumulative .blob files deleted by ov_cache pruning since server start. Same gating as rustedvino_ov_cache_pruned_bytes_total"
    );
}

/// Per-key request attribution — see [`KEY_USAGE_TOTAL`]'s doc comment.
fn register_key_usage_metrics() {
    describe_counter!(
        KEY_USAGE_TOTAL,
        Unit::Count,
        "Requests attributed to a specific Bearer key, labeled only by its last 6 characters \
         (never the full key — see require_bearer_auth's key_suffix helper, the only thing \
         that ever turns a full key into this fragment). Cumulative since server start, not a \
         rolling window — pair with rustedvino_key_last_seen_timestamp_seconds to tell \
         \"high total, quiet lately\" from \"active right now.\" Only successfully-authenticated \
         requests count: a key under 12 characters is skipped entirely (its last 6 chars would \
         be most/all of the actual secret), and rejected/401/403 attempts are deliberately not \
         counted — counting arbitrary presented (not necessarily valid) tokens here would turn \
         this metric's currently-bounded label cardinality (one series per *configured* key) \
         into an attacker-controlled unbounded one."
    );
    describe_gauge!(
        KEY_LAST_SEEN_TIMESTAMP_SECONDS,
        Unit::Seconds,
        "Unix timestamp of the most recent request attributed to this key suffix (see \
         rustedvino_key_usage_total). Same label, same safety contract, same 12-character floor."
    );
}

/// State gauges (pull-at-scrape) + Co-residency Slice 3a sizing/policy gauges.
fn register_state_gauges() {
    describe_gauge!(
        REQUESTS_RUNNING,
        Unit::Count,
        "Requests admitted to an engine slot and not yet finished (actively running). See requests_waiting for the admission-gate queue"
    );
    describe_gauge!(
        REQUESTS_MAX,
        Unit::Count,
        "Configured maximum concurrent request slots (the 429 capacity gate)"
    );
    describe_gauge!(
        REQUESTS_WAITING,
        Unit::Count,
        "Requests admitted but not yet actively processed: CB counts callers parked at the admission gate; the single-stream VLM counts queued-but-not-running"
    );
    describe_gauge!(
        MODEL_LOADED,
        Unit::Count,
        "1 if the model is loaded and Ready, 0 otherwise"
    );
    describe_gauge!(
        VRAM_TOTAL_BYTES,
        Unit::Bytes,
        "Total VRAM configured for the inference device"
    );
    describe_gauge!(
        VRAM_USED_BYTES,
        Unit::Bytes,
        "VRAM reserved across all loaded models of every kind (weights + KV pool)"
    );

    // Co-residency Slice 3a: sizing + policy gauges.
    describe_gauge!(
        KV_CACHE_POOL_GB,
        Unit::Gibibytes,
        "KV cache pool reserved for the model at load time (GB); 0 when the model is not loaded"
    );
    describe_gauge!(
        MODEL_PINNED,
        Unit::Count,
        "1 if the model is operator-pinned (never an eviction victim), 0 otherwise"
    );
    describe_gauge!(
        MODEL_PRIORITY,
        Unit::Count,
        "Operator soft eviction-order weight: higher is evicted later"
    );
    describe_gauge!(
        KV_CACHE_USAGE_PERCENT,
        Unit::Percent,
        "Live KV-cache pool occupancy (0-100) from the engine's last step; 0 when idle, unloaded, or (see rustedvino_kv_cache_usage_supported) unmeasurable for this engine kind"
    );
    describe_gauge!(
        KV_CACHE_USAGE_SUPPORTED,
        Unit::Count,
        "1 if rustedvino_kv_cache_usage_percent is a trustworthy live reading for this model, 0 if it's a structurally-unmeasurable value that can't be told apart from a genuinely empty pool (e.g. the VLM engine, which has a real KV pool but no public API to query it) or the model is unloaded"
    );
    describe_gauge!(
        KV_CACHE_PRESSURE_FLAGGED,
        Unit::Count,
        "1 if this model's KV occupancy has been continuously at/above kv_pressure_threshold_pct for at least kv_pressure_sustained_secs, 0 otherwise (including when kv_pressure_monitor_enabled is off). Detect-and-flag only — no eviction/resize is ever triggered by this alone; see POST /v1/admin/models/{id}/resize"
    );
    describe_gauge!(
        MODEL_LOAD_DURATION_SECONDS,
        Unit::Seconds,
        "Wall-clock duration of this model's most recent successful load (engine construction/JIT-compile time only, not admission-check or HTTP overhead). The device label (CPU/GPU/NPU/GPU.1...) is the model's actual resolved silicon, same as every other per-model gauge. Retained across eviction — reports the last-known figure for a model that's since been unloaded rather than the series disappearing"
    );
}

/// Hot-path counters/histograms (Phase 3.4) + realtime voice pipeline metrics.
fn register_hot_path_metrics() {
    describe_counter!(
        TOKENS_GENERATED_TOTAL,
        Unit::Count,
        "Output tokens generated across all requests (non-empty per-step deltas)"
    );
    describe_counter!(
        REQUESTS_TOTAL,
        Unit::Count,
        "Generation requests accepted by the engine (excludes 404/429/503 rejected before generation). The `modality` label splits text-only requests from image-bearing (multimodal) ones, even on a single VLM engine"
    );
    describe_counter!(
        CONTEXT_LENGTH_EXCEEDED_TOTAL,
        Unit::Count,
        "L0 prompt-length gate rejections (400 context_length_exceeded) before the request ever reached the engine — a clean, expected rejection, not a fault. The `gate` label (cb/vlm/npu) distinguishes which pipeline rejected, since NPU shares the cb path's text_gen kind label"
    );
    describe_counter!(
        POOL_CAPACITY_REJECTED_TOTAL,
        Unit::Count,
        "Concurrent-admission rejections: this request's own prompt is within the model's length gate, but admitting it now would overcommit the KV pool alongside requests already in flight — a clean, expected rejection distinct from context_length_exceeded, not a fault"
    );
    describe_counter!(
        OWN_BUDGET_REJECTED_TOTAL,
        Unit::Count,
        "Own-budget admission rejections on a single-slot model (max_concurrent_streams == 1): this request's own prompt_tokens + max_tokens exceeds the KV pool, with no other in-flight request involved — distinct from pool_capacity_rejected because retrying the identical request will fail identically; the client needs a smaller max_tokens, not a delay"
    );
    describe_gauge!(
        TOKENS_PER_SECOND,
        Unit::CountPerSecond,
        "Exponential moving average of aggregate engine decode throughput (tokens/sec across the live batch)"
    );
    describe_histogram!(
        TTFT_SECONDS,
        Unit::Seconds,
        "Time from request arrival (handler entry) to its first generated token. Carries a `modality` label (text vs multimodal)"
    );
    describe_histogram!(
        REQUEST_DURATION_SECONDS,
        Unit::Seconds,
        "End-to-end request duration from handler entry to final token. Carries a `modality` label (text vs multimodal)"
    );
    describe_histogram!(
        RTF_RATIO,
        Unit::Count,
        "STT/TTS real-time factor: processing time divided by audio duration. Below 1.0 is faster than real time. Covers both offline endpoints and the realtime voice pipeline (same engine thread either way); `kind` label is stt or tts"
    );

    // Realtime voice pipeline metrics.
    describe_counter!(
        RT_CONNECTIONS_TOTAL,
        Unit::Count,
        "WebSocket connections accepted on GET /v1/realtime"
    );
    describe_counter!(
        RT_TURNS_TOTAL,
        Unit::Count,
        "Completed voice pipeline turns. `result` label: ok | stt_error | llm_error | cancelled | \
         barge_in | admission_rejected"
    );
    describe_histogram!(
        RT_STT_DURATION_SECONDS,
        Unit::Seconds,
        "Whisper STT latency per voice turn (seconds from wav start to transcript)"
    );
    describe_histogram!(
        RT_LLM_DURATION_SECONDS,
        Unit::Seconds,
        "LLM decode latency per voice turn (seconds from add_request to final token)"
    );
    describe_counter!(
        RT_TTS_SENTENCES_TOTAL,
        Unit::Count,
        "TTS synthesis attempts per sentence. `result` label: ok | dropped"
    );
}

// ---- Hot-path metric handles (Phase 3.4) ----------------------------

/// Per-request input modality — the dimension that distinguishes a text-only
/// request from one carrying image content, **even on the same engine**. `kind`
/// labels the engine (a property of the loaded model); `modality` labels the
/// request (a property of its content). A VLM serves both, so its
/// `requests_total{kind="vision"}` splits into `modality="text"` and
/// `modality="multimodal"`; a text-gen engine only ever emits `modality="text"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modality {
    /// No image content parts — a pure text prompt. Every CB/text-gen request,
    /// and a text-only request served by a VLM.
    Text,
    /// At least one image content part (only reachable on a `Vision` engine).
    Multimodal,
}

impl Modality {
    /// Number of variants — the length of the cached per-request handle array.
    const COUNT: usize = 2;

    /// The Prometheus `modality` label value.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Modality::Text => "text",
            Modality::Multimodal => "multimodal",
        }
    }
}

/// Per-request metric handles for one `(kind, modality)` pair. Cached so a
/// per-request emit is a single handle call, not a label resolution.
struct PerRequestMetrics {
    /// `rustedvino_requests_total` — incremented when a request is accepted.
    requests_total: Counter,
    /// `rustedvino_ttft_seconds` — recorded once per request (first token).
    ttft_seconds: Histogram,
    /// `rustedvino_request_duration_seconds` — recorded once per request (finish).
    request_duration: Histogram,
}

/// Cached metric handles for the per-token hot path, owned by the engine
/// thread for its whole life.
///
/// Constructing this resolves each handle's `model`/`device`/`kind` (and, for
/// the per-request family, `modality`) labels exactly once at engine start, so
/// emitting a metric inside the per-token loop is a single atomic operation —
/// no registry lookup, no label rendering per token.
///
/// When no global recorder is installed (unit tests, or a build that never
/// called [`install`]), the `metrics` macros return no-op handles, so
/// [`HotMetrics::new`] is always safe to call.
pub struct HotMetrics {
    /// `rustedvino_tokens_generated_total` — incremented per non-empty token.
    /// Kind-labelled only: a batched step mixes requests, so there is no single
    /// per-token modality.
    tokens_generated: Counter,
    /// `rustedvino_tokens_per_second` — set with the throughput EMA per step.
    /// Kind-labelled only (batch-level, same reason as `tokens_generated`).
    tokens_per_second: Gauge,
    /// Per-request handles indexed by [`Modality`] (`Modality as usize`), so the
    /// `requests_total`/`ttft`/`duration` families split text vs multimodal.
    /// `None` for a modality this engine cannot serve (a text-gen engine has no
    /// `Multimodal` entry), so it never advertises a series it can't increment.
    per_request: [Option<PerRequestMetrics>; Modality::COUNT],
    /// `rustedvino_rtf_ratio` — recorded by STT/TTS engines only (`record_rtf`
    /// is simply never called by other kinds, so their series stay empty).
    rtf: Histogram,
}

impl HotMetrics {
    /// Build the handle cache for one engine (one model of one `kind` on one
    /// device). Every handle carries the same `model`/`device`/`kind` label set,
    /// so per-kind throughput is aggregatable in Prometheus (R2).
    #[must_use]
    pub fn new(kind: ModelKind, model_id: &str, device: &str) -> Self {
        let model = model_id.to_owned();
        let dev = device.to_owned();
        let kind_label = kind.label();
        // Resolve the per-request handles once per applicable modality, so an
        // emit is an array index + handle call — no per-request label
        // resolution. Only the modalities an engine can actually serve are
        // registered: every kind serves text; only a `Vision` engine also
        // serves `multimodal` (image-bearing) requests — so a text-gen model
        // never advertises a `modality="multimodal"` series it can't increment.
        let build = |modality: Modality| {
            let m = modality.label();
            PerRequestMetrics {
                requests_total: counter!(REQUESTS_TOTAL, "model" => model.clone(), "device" => dev.clone(), "kind" => kind_label, "modality" => m),
                ttft_seconds: histogram!(TTFT_SECONDS, "model" => model.clone(), "device" => dev.clone(), "kind" => kind_label, "modality" => m),
                request_duration: histogram!(REQUEST_DURATION_SECONDS, "model" => model.clone(), "device" => dev.clone(), "kind" => kind_label, "modality" => m),
            }
        };
        let mut per_request: [Option<PerRequestMetrics>; Modality::COUNT] = [None, None];
        per_request[Modality::Text as usize] = Some(build(Modality::Text));
        if matches!(kind, ModelKind::Vision) {
            per_request[Modality::Multimodal as usize] = Some(build(Modality::Multimodal));
        }
        Self {
            tokens_generated: counter!(TOKENS_GENERATED_TOTAL, "model" => model.clone(), "device" => dev.clone(), "kind" => kind_label),
            tokens_per_second: gauge!(TOKENS_PER_SECOND, "model" => model.clone(), "device" => dev.clone(), "kind" => kind_label),
            per_request,
            rtf: histogram!(RTF_RATIO, "model" => model.clone(), "device" => dev.clone(), "kind" => kind_label),
        }
    }

    /// Count `count` generated output tokens (per non-empty per-step delta).
    ///
    /// `count` is the real number of tokens the engine reports for that step
    /// — normally `1`, but a speculative-decoding (CB) request can accept
    /// several draft tokens in one verification step, all landing in a
    /// single delta. Callers must pass the engine's own count, not assume
    /// `1` per callback (found live 2026-07-19, see the project's internal engineering log).
    pub fn token_generated(&self, count: u64) {
        self.tokens_generated.increment(count);
    }

    /// Count one request accepted by the engine, tagged by input `modality`.
    /// A no-op if the engine doesn't serve this modality (defensive — the
    /// engines only ever pass a modality they can serve).
    pub fn request_accepted(&self, modality: Modality) {
        if let Some(pr) = &self.per_request[modality as usize] {
            pr.requests_total.increment(1);
        }
    }

    /// Publish the current throughput EMA (tokens/sec).
    pub fn set_tokens_per_second(&self, tps: f64) {
        self.tokens_per_second.set(tps);
    }

    /// Record a request's time-to-first-token, in seconds, tagged by `modality`.
    pub fn record_ttft(&self, modality: Modality, secs: f64) {
        if let Some(pr) = &self.per_request[modality as usize] {
            pr.ttft_seconds.record(secs);
        }
    }

    /// Record a request's end-to-end duration, in seconds, tagged by `modality`.
    pub fn record_duration(&self, modality: Modality, secs: f64) {
        if let Some(pr) = &self.per_request[modality as usize] {
            pr.request_duration.record(secs);
        }
    }

    /// Record a request's real-time factor (processing time ÷ audio duration).
    /// STT/TTS engines only.
    pub fn record_rtf(&self, ratio: f64) {
        self.rtf.record(ratio);
    }
}

/// Set the state gauges from a live [`MetricsSnapshot`].
///
/// Called by the `/metrics` handler immediately before rendering, so the
/// scrape reflects the current request counts.
// Request counts and slot caps are small; the usize→f64 casts cannot lose
// meaningful precision for any realistic value.
#[allow(clippy::cast_precision_loss)]
pub fn record_state(snap: &MetricsSnapshot) {
    // `snap.models` carries every model loaded at least once — Ready *and*
    // evicted. Evicted models report `loaded == false` with zero counters, so
    // emitting them here resets their gauges to 0 instead of leaving the stale
    // `1`/last-count a Ready scrape wrote. The series keeps the same
    // model/device/kind label set it had while loaded.
    //
    // Phase C1: the `device` label is per-model (`m.device`), not the global
    // `snap.device`. A model placed off the primary device (an iGPU embedder, a
    // CPU model) is now labelled with its *actual* device. The VRAM gauges below
    // remain keyed by the primary `snap.device` (the box-level pool figure).
    for m in &snap.models {
        let model = m.id.as_str();
        let device = m.device.as_str();
        let kind = m.kind.label();
        gauge!(REQUESTS_RUNNING, "model" => model.to_owned(), "device" => device.to_owned(), "kind" => kind)
            .set(m.active as f64);
        gauge!(REQUESTS_MAX, "model" => model.to_owned(), "device" => device.to_owned(), "kind" => kind)
            .set(m.max_seqs as f64);
        gauge!(REQUESTS_WAITING, "model" => model.to_owned(), "device" => device.to_owned(), "kind" => kind)
            .set(m.waiting as f64);
        gauge!(MODEL_LOADED, "model" => model.to_owned(), "device" => device.to_owned(), "kind" => kind)
            .set(f64::from(u8::from(m.loaded)));
        // Co-residency Slice 3a: sizing + policy gauges. Pool follows the live
        // reservation (0 when evicted); pinned/priority are static policy.
        gauge!(KV_CACHE_POOL_GB, "model" => model.to_owned(), "device" => device.to_owned(), "kind" => kind)
            .set(m.kv_cache_pool_gb);
        gauge!(MODEL_PINNED, "model" => model.to_owned(), "device" => device.to_owned(), "kind" => kind)
            .set(f64::from(u8::from(m.pinned)));
        gauge!(MODEL_PRIORITY, "model" => model.to_owned(), "device" => device.to_owned(), "kind" => kind)
            .set(f64::from(m.priority));
        gauge!(KV_CACHE_USAGE_PERCENT, "model" => model.to_owned(), "device" => device.to_owned(), "kind" => kind)
            .set(m.kv_cache_usage_pct);
        gauge!(KV_CACHE_USAGE_SUPPORTED, "model" => model.to_owned(), "device" => device.to_owned(), "kind" => kind)
            .set(f64::from(u8::from(m.kv_cache_usage_supported)));
        gauge!(KV_CACHE_PRESSURE_FLAGGED, "model" => model.to_owned(), "device" => device.to_owned(), "kind" => kind)
            .set(f64::from(u8::from(m.kv_pressure_flagged)));
        // `None` only in the narrow window before this model's first load has
        // ever completed — skip the series entirely rather than emit a
        // misleading 0.0 (a real load never completes in ~0s).
        if let Some(secs) = m.load_duration_secs {
            gauge!(MODEL_LOAD_DURATION_SECONDS, "model" => model.to_owned(), "device" => device.to_owned(), "kind" => kind)
                .set(secs);
        }
    }
    // Box-level aggregate VRAM gauges (backward-compat, single-domain view).
    let primary = snap.device.as_str();
    gauge!(VRAM_TOTAL_BYTES, "device" => primary.to_owned()).set(snap.total_vram_gb * 1e9);
    gauge!(VRAM_USED_BYTES, "device" => primary.to_owned()).set(snap.used_vram_gb * 1e9);
    // Per-domain gauges (multi-GPU: each domain gets its own labelled series).
    for (domain, total_gb, used_gb) in &snap.domain_vram {
        gauge!(VRAM_TOTAL_BYTES, "domain" => domain.clone()).set(*total_gb * 1e9);
        gauge!(VRAM_USED_BYTES, "domain" => domain.clone()).set(*used_gb * 1e9);
    }
}

/// Record one L0 prompt-length gate rejection (`400 context_length_exceeded`)
/// — see [`CONTEXT_LENGTH_EXCEEDED_TOTAL`]'s description. Called from the
/// handler layer (`handlers/chat.rs`'s `gate_prompt`/`gate_vlm_prompt`/
/// `gate_npu_prompt`), not the engine thread — a registry lookup per call is
/// fine here (these are rare relative to the token hot path `HotMetrics`
/// caches handles for). `gate` is `"cb"`/`"vlm"`/`"npu"`.
pub(crate) fn record_context_length_exceeded(model_id: &str, gate: &'static str) {
    counter!(CONTEXT_LENGTH_EXCEEDED_TOTAL, "model" => model_id.to_owned(), "gate" => gate)
        .increment(1);
}

/// See [`POOL_CAPACITY_REJECTED_TOTAL`]'s description.
pub(crate) fn record_pool_capacity_rejected(model_id: &str) {
    counter!(POOL_CAPACITY_REJECTED_TOTAL, "model" => model_id.to_owned()).increment(1);
}

/// See [`OWN_BUDGET_REJECTED_TOTAL`]'s description.
pub(crate) fn record_own_budget_rejected(model_id: &str) {
    counter!(OWN_BUDGET_REJECTED_TOTAL, "model" => model_id.to_owned()).increment(1);
}

/// Records one request attributed to `key_suffix` — see [`KEY_USAGE_TOTAL`]'s
/// doc comment for the full metric semantics. `key_suffix` must already be
/// exactly the last 6 characters of a key that matched a configured one;
/// this function trusts its caller rather than re-deriving or validating the
/// fragment, because `require_bearer_auth`'s `key_suffix` helper in `lib.rs`
/// is the *only* thing in this codebase that is ever allowed to turn a full
/// Bearer key into something shorter — duplicating that logic here would
/// create a second place a full key could accidentally leak through.
/// Called from `lib.rs` per matched request, not per token — a registry
/// lookup + `String` alloc here each time is the same tradeoff
/// `record_context_length_exceeded` above already makes for the same reason.
pub(crate) fn record_key_usage(key_suffix: &str) {
    counter!(KEY_USAGE_TOTAL, "key_suffix" => key_suffix.to_owned()).increment(1);
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());
    gauge!(KEY_LAST_SEEN_TIMESTAMP_SECONDS, "key_suffix" => key_suffix.to_owned()).set(now_secs);
}

/// Records `ov_cache_dir`'s current on-disk size — see [`OV_CACHE_BYTES`]'s
/// doc comment. Called once per background sweep
/// (`ModelManager::prune_ov_cache_if_over_budget`), unconditionally whenever
/// `ov_cache_dir` is configured — before any `ov_cache_max_gb` budget check,
/// so a box that never bounds its cache still gets basic visibility into it.
// Byte counts here are far under f64's exact-integer ceiling.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn record_ov_cache_bytes(total_bytes: u64) {
    gauge!(OV_CACHE_BYTES).set(total_bytes as f64);
}

/// Records one prune pass that actually deleted something — see
/// [`OV_CACHE_PRUNED_BYTES_TOTAL`]'s doc comment. Not called when a sweep
/// finds nothing to delete (unbounded cache, or already under budget), so
/// these counters only move on a real reclaim.
pub(crate) fn record_ov_cache_pruned(freed_bytes: u64, deleted_files: u64) {
    counter!(OV_CACHE_PRUNED_BYTES_TOTAL).increment(freed_bytes);
    counter!(OV_CACHE_PRUNED_FILES_TOTAL).increment(deleted_files);
}

// ============================================================
// Unit tests
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// Constructing `HotMetrics` and emitting through every handle must be safe
    /// with **no global recorder installed** — the engine thread always builds
    /// one, and unit/integration tests run without calling [`install`]. The
    /// `metrics` macros fall back to no-op handles, so none of this panics.
    #[test]
    fn hot_metrics_are_noop_safe_without_recorder() {
        let m = HotMetrics::new(ModelKind::TextGen, "qwen3-8b-int4-ov", "GPU.1");
        m.request_accepted(Modality::Text);
        m.request_accepted(Modality::Multimodal);
        m.token_generated(1);
        m.set_tokens_per_second(62.5);
        m.record_ttft(Modality::Text, 0.12);
        m.record_duration(Modality::Multimodal, 3.4);
        m.record_rtf(0.4);
        // Reaching here without a panic is the assertion.
    }

    /// `record_context_length_exceeded` actually increments the counter, with
    /// the `gate` label distinguishing which pipeline rejected — the fix for
    /// a clean L0-gate rejection being fully invisible server-side (`dev/
    /// autotest/20260823_l0_gate_rejection_observability_gap.md`).
    #[test]
    fn context_length_exceeded_counter_increments_per_gate() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            record_context_length_exceeded("qwen3-4b-int4-ov", "cb");
            record_context_length_exceeded("qwen3-4b-int4-ov", "cb");
            record_context_length_exceeded("qwen3.5-9B-int4-ov", "vlm");
        });
        let rendered = handle.render();
        assert!(
            rendered.contains(
                "rustedvino_context_length_exceeded_total{model=\"qwen3-4b-int4-ov\",gate=\"cb\"} 2"
            ),
            "cb gate must count 2 rejections for its model:\n{rendered}"
        );
        assert!(
            rendered.contains(
                "rustedvino_context_length_exceeded_total{model=\"qwen3.5-9B-int4-ov\",gate=\"vlm\"} 1"
            ),
            "vlm gate must count its own rejection separately:\n{rendered}"
        );
    }

    /// Hot-path metrics render with a `kind` label that distinguishes a
    /// text-generation engine from a vision engine (R2). Uses a *local*
    /// recorder so the test never touches the process-global recorder and can
    /// run in parallel with the rest of the suite.
    #[test]
    fn exposition_carries_kind_label_per_kind() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let text = HotMetrics::new(ModelKind::TextGen, "qwen3-8b-int4-ov", "GPU.1");
            text.request_accepted(Modality::Text);
            text.token_generated(1);
            let vision = HotMetrics::new(ModelKind::Vision, "qwen2.5-vl-7b-int4-ov", "GPU.1");
            vision.request_accepted(Modality::Multimodal);
            vision.token_generated(1);
        });
        let rendered = handle.render();
        assert!(
            rendered.contains("kind=\"text_gen\""),
            "text-gen metrics must carry kind=\"text_gen\":\n{rendered}"
        );
        assert!(
            rendered.contains("kind=\"vision\""),
            "vision metrics must carry kind=\"vision\":\n{rendered}"
        );
        // The two kinds are distinct series on the same metric name.
        assert!(rendered.contains("rustedvino_tokens_generated_total"));
        assert!(rendered.contains("model=\"qwen2.5-vl-7b-int4-ov\""));
    }

    /// A single VLM engine (`kind="vision"`) splits `requests_total` into
    /// `modality="text"` and `modality="multimodal"` — the per-request label
    /// that `kind` alone cannot express (a text-only query to a VLM is still a
    /// vision-engine request). TTFT/duration carry the same label.
    #[test]
    fn requests_split_by_modality_on_one_engine() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let vlm = HotMetrics::new(ModelKind::Vision, "qwen2.5-vl-7b-int4-ov", "GPU.1");
            vlm.request_accepted(Modality::Text);
            vlm.request_accepted(Modality::Text);
            vlm.request_accepted(Modality::Multimodal);
            vlm.record_ttft(Modality::Multimodal, 0.2);
            vlm.record_duration(Modality::Multimodal, 1.0);
        });
        let r = handle.render();

        // requests_total is two series on the same engine, distinguished by modality.
        let req_line = |modality: &str| {
            r.lines()
                .find(|l| {
                    l.starts_with("rustedvino_requests_total")
                        && l.contains(&format!("modality=\"{modality}\""))
                })
                .unwrap_or_else(|| panic!("no requests_total for modality={modality}:\n{r}"))
        };
        assert!(
            req_line("text").ends_with(" 2"),
            "two text requests: {}",
            req_line("text")
        );
        assert!(
            req_line("multimodal").ends_with(" 1"),
            "one multimodal request: {}",
            req_line("multimodal")
        );
        // Both carry kind="vision" (same engine).
        assert!(req_line("text").contains("kind=\"vision\""));
        // ttft histogram is also modality-tagged.
        assert!(
            r.lines().any(|l| l.starts_with("rustedvino_ttft_seconds")
                && l.contains("modality=\"multimodal\"")),
            "ttft must carry modality:\n{r}"
        );
    }

    /// A text-gen engine can never serve image content (the handler 400s it
    /// first), so it must NOT register a `modality="multimodal"` series — only
    /// `modality="text"`. Pins the kind-aware registration.
    #[test]
    fn text_gen_engine_has_no_multimodal_series() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let llm = HotMetrics::new(ModelKind::TextGen, "qwen3-8b-int4-ov", "GPU.1");
            llm.request_accepted(Modality::Text);
            // Even if a (buggy) caller passed Multimodal, it must be a silent
            // no-op — no series created.
            llm.request_accepted(Modality::Multimodal);
        });
        let r = handle.render();
        assert!(
            r.contains("modality=\"text\""),
            "text series must exist:\n{r}"
        );
        assert!(
            !r.contains("modality=\"multimodal\""),
            "a text-gen engine must NOT advertise a multimodal series:\n{r}"
        );
    }

    /// Build a one-model snapshot for the state-gauge tests.
    fn snap_with(loaded: bool, active: usize, max_seqs: usize) -> MetricsSnapshot {
        MetricsSnapshot {
            device: "GPU.1".to_owned(),
            total_vram_gb: 22.5,
            used_vram_gb: if loaded { 5.5 } else { 0.0 },
            domain_vram: vec![("GPU.1".to_owned(), 22.5, if loaded { 5.5 } else { 0.0 })],
            models: vec![crate::model_manager::ModelMetrics {
                id: "qwen3-8b-int4-ov".to_owned(),
                device: "GPU.1".to_owned(),
                kind: ModelKind::TextGen,
                loaded,
                active,
                max_seqs,
                waiting: 0,
                // Pool follows the live reservation (0 when evicted); pinned/
                // priority are static policy and report loaded or not.
                kv_cache_pool_gb: if loaded { 4.0 } else { 0.0 },
                pinned: true,
                priority: 100,
                // Live occupancy only reads non-zero on a Ready engine.
                kv_cache_usage_pct: if loaded { 42.0 } else { 0.0 },
                kv_cache_usage_supported: loaded,
                // Retained across eviction — reports the last-known load time
                // regardless of `loaded`, same as production behaviour.
                load_duration_secs: Some(12.5),
                // Monitor disabled in these tests — always false, same as
                // production default.
                kv_pressure_flagged: false,
            }],
        }
    }

    /// The `model="qwen3-8b-int4-ov"` line for `metric` in a Prometheus render.
    fn series_value(rendered: &str, metric: &str) -> f64 {
        rendered
            .lines()
            .find(|l| l.starts_with(metric) && l.contains("qwen3-8b-int4-ov"))
            .and_then(|l| l.rsplit(' ').next())
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("no `{metric}` series for qwen3-8b-int4-ov in:\n{rendered}"))
    }

    /// `record_state` must drive `model_loaded` (and the request gauges) back to
    /// 0 when a model that was Ready becomes evicted — the snapshot keeps
    /// reporting it with `loaded == false`, so the stale `1` is overwritten.
    /// (Regression: evicted models used to drop out of the snapshot entirely,
    /// leaving the gauge stuck at its last value.)
    #[test]
    fn model_loaded_gauge_resets_to_zero_on_eviction() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            // Scrape 1: model Ready, mid-flight requests.
            record_state(&snap_with(true, 3, 16));
            let loaded = handle.render();
            assert!((series_value(&loaded, MODEL_LOADED) - 1.0).abs() < f64::EPSILON);
            assert!((series_value(&loaded, REQUESTS_RUNNING) - 3.0).abs() < f64::EPSILON);
            assert!((series_value(&loaded, REQUESTS_MAX) - 16.0).abs() < f64::EPSILON);

            // Scrape 2: same model now evicted (loaded=false, zeroed counters).
            record_state(&snap_with(false, 0, 0));
            let evicted = handle.render();
            assert!(
                series_value(&evicted, MODEL_LOADED).abs() < f64::EPSILON,
                "model_loaded must read 0 after eviction:\n{evicted}"
            );
            assert!(
                series_value(&evicted, REQUESTS_RUNNING).abs() < f64::EPSILON,
                "requests_running must reset to 0 after eviction"
            );
            assert!(
                series_value(&evicted, REQUESTS_MAX).abs() < f64::EPSILON,
                "requests_max must reset to 0 after eviction"
            );
        });
    }

    /// Co-residency Slice 3a: `record_state` must emit the sizing/policy gauges
    /// per model. The pool follows the live reservation (0 once evicted) while
    /// `pinned`/`priority` are static policy and persist across eviction.
    #[test]
    fn coresidency_sizing_and_policy_gauges_render() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            // Ready: pool reserved, policy reported.
            record_state(&snap_with(true, 1, 16));
            let ready = handle.render();
            assert!((series_value(&ready, KV_CACHE_POOL_GB) - 4.0).abs() < f64::EPSILON);
            assert!((series_value(&ready, MODEL_PINNED) - 1.0).abs() < f64::EPSILON);
            assert!((series_value(&ready, MODEL_PRIORITY) - 100.0).abs() < f64::EPSILON);
            assert!((series_value(&ready, KV_CACHE_USAGE_PERCENT) - 42.0).abs() < f64::EPSILON);
            assert!(
                (series_value(&ready, KV_CACHE_USAGE_SUPPORTED) - 1.0).abs() < f64::EPSILON,
                "a Ready model whose engine supports the reading must report supported=1"
            );

            // Evicted: pool + live occupancy drop to 0; policy gauges hold their
            // static value.
            record_state(&snap_with(false, 0, 0));
            let evicted = handle.render();
            assert!(
                series_value(&evicted, KV_CACHE_POOL_GB).abs() < f64::EPSILON,
                "kv pool must read 0 once the model is evicted:\n{evicted}"
            );
            assert!(
                series_value(&evicted, KV_CACHE_USAGE_PERCENT).abs() < f64::EPSILON,
                "kv usage must read 0 once the model is evicted:\n{evicted}"
            );
            assert!(
                series_value(&evicted, KV_CACHE_USAGE_SUPPORTED).abs() < f64::EPSILON,
                "an evicted model's usage reading must report supported=0, not just the \
                 percent gauge dropping to 0:\n{evicted}"
            );
            assert!((series_value(&evicted, MODEL_PINNED) - 1.0).abs() < f64::EPSILON);
            assert!((series_value(&evicted, MODEL_PRIORITY) - 100.0).abs() < f64::EPSILON);
        });
    }

    /// `rustedvino_model_load_duration_seconds` reports the last-known load
    /// time and keeps reporting it after eviction — unlike `kv_cache_pool_gb`
    /// (resets to 0) it behaves like `pinned`/`priority` (static, persists),
    /// because it describes a past event, not current live reservation.
    #[test]
    fn load_duration_gauge_persists_across_eviction() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            record_state(&snap_with(true, 1, 16));
            let ready = handle.render();
            assert!(
                (series_value(&ready, MODEL_LOAD_DURATION_SECONDS) - 12.5).abs() < f64::EPSILON,
                "a Ready model must report its last load duration:\n{ready}"
            );

            record_state(&snap_with(false, 0, 0));
            let evicted = handle.render();
            assert!(
                (series_value(&evicted, MODEL_LOAD_DURATION_SECONDS) - 12.5).abs() < f64::EPSILON,
                "load duration must persist after eviction, not reset to 0:\n{evicted}"
            );
        });
    }

    /// `record_key_usage` renders both the counter and the last-seen gauge,
    /// labeled by the exact fragment it was given — this test only ever
    /// passes it a 6-character fragment, matching `key_suffix`'s contract
    /// (see `lib.rs`); this function itself does no length checking, that's
    /// the caller's job (see its own doc comment).
    #[test]
    fn key_usage_renders_counter_and_last_seen_gauge() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            record_key_usage("b7c4e9");
        });
        let rendered = handle.render();
        assert!(
            rendered.contains("rustedvino_key_usage_total"),
            "must render the counter:\n{rendered}"
        );
        assert!(
            rendered.contains("rustedvino_key_last_seen_timestamp_seconds"),
            "must render the last-seen gauge:\n{rendered}"
        );
        assert!(
            rendered.contains("key_suffix=\"b7c4e9\""),
            "both series must carry the exact suffix given:\n{rendered}"
        );
        let ts: f64 = rendered
            .lines()
            .find(|l| l.starts_with(KEY_LAST_SEEN_TIMESTAMP_SECONDS))
            .and_then(|l| l.rsplit(' ').next())
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| {
                panic!("no `{KEY_LAST_SEEN_TIMESTAMP_SECONDS}` series in:\n{rendered}")
            });
        assert!(
            ts > 1_700_000_000.0,
            "last-seen must be a real current Unix timestamp, not 0 or a placeholder: {ts}"
        );
    }

    /// Two different key suffixes are two distinct series, not merged into
    /// one — the whole point of labeling by suffix instead of a single
    /// unlabeled counter.
    #[test]
    fn key_usage_distinguishes_suffixes() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            record_key_usage("aaaaaa");
            record_key_usage("aaaaaa");
            record_key_usage("bbbbbb");
        });
        let rendered = handle.render();
        assert!(rendered.contains("key_suffix=\"aaaaaa\""));
        assert!(rendered.contains("key_suffix=\"bbbbbb\""));
    }
}
