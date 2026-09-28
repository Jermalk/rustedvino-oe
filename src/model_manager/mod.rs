// ============================================================
// src/model_manager/mod.rs — multi-model lifecycle manager
// ============================================================
// `ModelManager` is the Phase-2 replacement for the single
// hardcoded model in `main.rs`. It owns the set of *known*
// models (from config), tracks which are loaded, routes chat
// requests to the right engine, and evicts models by LRU when
// VRAM is tight.
//
// Key design points vs stormVINO's model_manager.py:
//
//   1. NO automatic fallback to a default model. If the client
//      asks for "model-x" and it is not loaded → 404/503.
//      stormVINO fell back to `default_model` silently, which
//      hid misconfiguration. Rust says what it means.
//
//   2. Rust `drop()` is synchronous. The Python version needed
//      `gc.collect()` after `del pipeline` because CPython's
//      reference counting is not guaranteed to run __del__ before
//      the next allocator call. In Rust, dropping the EngineHandle
//      + joining the thread gives deterministic VRAM release.
//
//   3. Eviction is safe under concurrency. We mark state →
//      Evicting under a write lock before dropping the handle,
//      so new requests to a being-evicted model see 503 rather
//      than getting a dangling handle.
//
//   4. VRAM is pre-allocated in the tracker before the model load
//      begins, preventing two concurrent loads from both believing
//      they fit. On load failure the pre-allocation is undone.
//
// CRASH COURSE — RwLock vs Mutex:
//   `RwLock<T>` allows EITHER many concurrent readers OR one writer.
//   `models` is read on every `get_handle()` call (hot path, many
//   concurrent requests). Taking a write lock for every read would
//   serialise all chat requests. RwLock gives read concurrency at
//   the cost of slightly more complex write logic.
// ============================================================

pub mod config;
pub mod engine;
pub(crate) mod lifecycle;
pub(crate) mod placement;
pub mod stub_engines;
pub(crate) mod template;
mod vram;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::{Duration, Instant};

use anyhow::Context as _;
use anyhow::Result;

pub use config::{Config, ReasoningParser, SupervisorConfig};
pub use engine::{EngineHandleKind, ManagedEngine, ModelKind};

/// True when an `OpenVINO`/`OpenCL` error string indicates the GPU context is
/// permanently wedged (OOM-class). Both markers below mean the `OpenCL` context
/// cannot recover for this process and every subsequent call will fail:
///   `CL_OUT_OF_RESOURCES` (-5)  — allocator failed (VRAM OOM)
///   `CL_INVALID_EVENT`    (-58) — context already poisoned; retry also fails
///
/// Shared by the load path (L3 gate) and the inference hot path so both
/// classify poison identically (T6.2). Kept a plain substring match on the C++
/// text — the bridge has no structured error code to inspect.
pub(crate) fn is_gpu_poison_error(msg: &str) -> bool {
    msg.contains("CL_OUT_OF_RESOURCES") || msg.contains("CL_INVALID_EVENT")
}

/// Substring marker for a CB-scheduler KV-pool-exhaustion error
/// (`GenerationStatus::IGNORED` — the project's internal engineering log).
/// Shared between `cb_engine.rs` (constructs the `StreamEvent::Error` message
/// containing it) and [`is_pool_exhausted_error`] (classifies it at the
/// response-shaping layer) so the two can't drift apart.
///
/// `cb_engine.rs` embeds `observed_prompt_tokens=N active_requests=M` after
/// this marker — `M` (the number of requests concurrently in flight against
/// this model's engine thread at the moment of failure, itself included) is
/// how the caller tells a genuinely transient multi-tenant capacity blip
/// (`M > 1` — other sequences are competing for the pool, expected to clear
/// once they finish, plain retryable 503, unchanged) apart from a single
/// conversation alone hitting the model's real ceiling (`M <= 1` — nothing
/// else can be crowding it out, so the static `compute_max_prompt_tokens`
/// formula estimate is simply wrong for this model and retrying the same
/// request can never succeed). See `dev/autotest/
/// 20260823_qwen3-4b-int4-ov_cb_pool_exhaustion_gap.md`, which found a dense
/// (non-hybrid) model's real capacity ~15% below the formula's advertised
/// ceiling with zero contention — the same class of static-formula miss the
/// VLM wedge below was built for, just on the plain CB path, where it
/// previously had no self-heal at all.
pub(crate) const POOL_EXHAUSTED_MARKER: &str = "KV cache pool exhausted for this request";

/// True when `msg` reports the CB scheduler's `GenerationStatus::IGNORED` —
/// this request's generation was never actually run because the KV pool had
/// no room for it. Distinct from [`is_gpu_poison_error`]: this is scoped to
/// the one request, not the whole `OpenCL` context, so callers must NOT treat
/// it as GPU poisoning (no `mark_gpu_poisoned`). Whether it is actually
/// retryable-as-is depends on [`pool_exhausted_active_requests`] — see
/// [`POOL_EXHAUSTED_MARKER`]'s doc comment.
pub(crate) fn is_pool_exhausted_error(msg: &str) -> bool {
    msg.contains(POOL_EXHAUSTED_MARKER)
}

/// Extract a `field=N` value embedded in a KV-wedge/pool-exhaustion marker
/// message — shared parsing for both [`kv_wedge_observed_prompt_tokens`] and
/// [`pool_exhausted_active_requests`] so the two can't drift on digit-parsing
/// rules. `None` on any parse failure (field absent, or the digits don't
/// parse) rather than guessing.
fn extract_marker_field(msg: &str, field: &str) -> Option<usize> {
    let after = msg.split(field).nth(1)?;
    let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// Extract the `observed_prompt_tokens=N` value embedded in a
/// [`VLM_KV_WEDGE_MARKER`] or [`POOL_EXHAUSTED_MARKER`] message, if present
/// and well-formed. `None` on any parse failure — the caller should skip the
/// ratchet write (still surface/handle the error) rather than act on a
/// guessed value.
pub(crate) fn kv_wedge_observed_prompt_tokens(msg: &str) -> Option<usize> {
    extract_marker_field(msg, "observed_prompt_tokens=")
}

/// Extract the `active_requests=N` value `cb_engine.rs` embeds in a
/// [`POOL_EXHAUSTED_MARKER`] message — see that constant's doc comment for
/// why this gates whether pool exhaustion gets treated as transient
/// congestion or a real static-formula miss worth self-healing. `None` on
/// any parse failure — callers must treat that as "not provably solo" and
/// skip recovery, since evicting a model on a misread is the wrong direction
/// to be wrong in.
pub(crate) fn pool_exhausted_active_requests(msg: &str) -> Option<usize> {
    extract_marker_field(msg, "active_requests=")
}

/// Substring marker for the VLM-path KV-admission wedge (`dev/autotest/
/// 20260821_omnicoder9b_qwen35_hybrid_stall.md`) — hybrid attention
/// architectures (Qwen3.5/3.6's `GatedDeltaNet`) whose real usable KV
/// capacity the static `compute_max_prompt_tokens` formula cannot precisely
/// predict (same root cause independently reported as
/// `vllm-project/vllm#37121`). Once real usage crosses the model's actual
/// (unknown in advance) ceiling, the pipeline permanently wedges: every
/// subsequent call returns zero tokens with a frozen `perf_metrics` reading,
/// silently "successful" (`finish_reason: "stop"`) unless surfaced as an
/// error here. `vlm_engine.rs::run_generate` embeds
/// `observed_prompt_tokens={N}` in the message — extract it with
/// [`kv_wedge_observed_prompt_tokens`] for
/// [`ModelManager::spawn_kv_wedge_recovery`]'s ratchet.
///
/// Distinct from both [`is_gpu_poison_error`] (whole-`OpenCL`-context
/// poisoning) and [`is_pool_exhausted_error`] (the CB path's analogous but
/// differently-triggered `GenerationStatus::IGNORED`): unlike either, the
/// pipeline itself never self-recovers from this, so the caller must call
/// [`ModelManager::spawn_kv_wedge_recovery`] (evict + reload), not just
/// surface a retryable error and hope the next request is luckier.
pub(crate) const VLM_KV_WEDGE_MARKER: &str =
    "VLM pipeline produced zero tokens internally (likely KV-admission capacity wedge)";

/// True when `msg` reports the VLM-path KV-admission wedge — see
/// [`VLM_KV_WEDGE_MARKER`]'s doc comment.
pub(crate) fn is_vlm_kv_wedge_error(msg: &str) -> bool {
    msg.contains(VLM_KV_WEDGE_MARKER)
}

/// Maximum time eviction waits for an engine thread to exit before abandoning
/// the join (#4). Kept well inside the stop script's 45s budget so a single
/// wedged engine thread — a stuck FFI step, a never-settling pipeline — cannot
/// hang eviction, and therefore process shutdown, indefinitely.
///
/// Deliberately short: by the time normal eviction reaches this join, it has
/// already polled the engine down to zero active requests (or force-aborted
/// it) via [`EVICT_DRAIN_DEADLINE`] below, so the thread should exit almost
/// immediately — this deadline is purely a backstop against a *wedged*
/// thread, not a busy one.
const JOIN_DEADLINE: Duration = Duration::from_secs(8);

/// Maximum time normal (non-abort) eviction waits for an engine's real
/// in-flight requests to reach their own natural completion before giving up
/// and forcing an abort (2026-07-18 drain fix). Generous relative to
/// `JOIN_DEADLINE`: an actual generation can legitimately run for a long time
/// (a large `max_tokens` on a slow device), and this is what the documented
/// "drains in-flight requests" eviction contract is actually waiting on — not
/// a symptom of anything wedged, which `JOIN_DEADLINE` guards separately.
const EVICT_DRAIN_DEADLINE: Duration = Duration::from_mins(2);

/// Poll interval while waiting out [`EVICT_DRAIN_DEADLINE`]. Cheap (an atomic
/// read via [`ManagedEngine::active`](crate::model_manager::engine::ManagedEngine::active)),
/// so a short interval costs nothing and keeps eviction latency close to the
/// request's actual remaining generation time rather than rounding up to a
/// coarse poll step.
const EVICT_DRAIN_POLL_INTERVAL: Duration = Duration::from_millis(50);
use lifecycle::{EngineFactory, OvEngineFactory};
use vram::MemoryTracker;

use crate::cb_engine::EngineHandle;
use crate::device_inventory::{DOMAIN_SYSTEM, DeviceInfo, DeviceInventory, DeviceTier};
use crate::os_memory;
use crate::prompt_builder::ModelFamily;

// ---- Error type -------------------------------------------------------

/// Reasons a [`ModelManager::get_handle`] or related call can fail.
///
/// The variants are fine-grained so HTTP handlers can map each to the
/// right status code without string matching.
#[derive(Debug, PartialEq, Eq)]
pub enum ModelError {
    /// The model ID is not in the known-model registry (not in `vram_gb`).
    /// → HTTP 404
    NotFound(String),
    /// The model IS registered, but its directory does not exist under
    /// `models_dir` — a stale/typo'd config entry (e.g. name drift between the
    /// registered `model_id` and what was actually downloaded/converted).
    /// Checked up front, before any VRAM reservation or LRU eviction, so a
    /// doomed load can't evict a healthy model for nothing.
    /// → HTTP 404
    FilesMissing(String),
    /// The model's directory exists, but is missing files its kind needs to
    /// actually run (`crate::model_completeness`) — e.g. a reranking model
    /// with no converted `openvino_tokenizer.*` (the live incident this
    /// check was built for, 2026-08-03: the model loaded fine and only
    /// failed on first real inference). Checked at the same point as
    /// [`Self::FilesMissing`], before any VRAM reservation or eviction — a
    /// doomed load fails cheaply instead of evicting a healthy model first.
    /// → HTTP 404
    FilesIncomplete(String, Vec<String>),
    /// The model exists but has not been loaded yet.
    /// → HTTP 503
    NotLoaded,
    /// A load for this model is currently in progress.
    /// → HTTP 503
    Loading,
    /// The model is being evicted; new requests must wait or use another model.
    /// → HTTP 503
    Evicting,
    /// The GPU's `OpenCL` context was poisoned by a previously failed load
    /// (`CL_OUT_OF_RESOURCES` or `CL_INVALID_EVENT`). New loads will fail until
    /// the process is restarted. Existing loaded models may still serve requests.
    /// → HTTP 503
    GpuPoisoned,
    /// The model is Ready but its kind does not support the requested operation
    /// (e.g. a [`ModelKind::Vision`] model on a text-only endpoint such as
    /// `/tokenize` or `/v1/completions`). The message names the mismatch.
    /// → HTTP 400
    WrongKind(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "model '{id}' not found"),
            Self::FilesMissing(id) => write!(
                f,
                "model '{id}' is registered but its directory was not found under models_dir — check the config entry and models_dir for a stale or misspelled model_id"
            ),
            Self::FilesIncomplete(id, missing) => write!(
                f,
                "model '{id}' is missing required files: {} — re-run the model conversion/export for this model",
                missing.join(", ")
            ),
            Self::NotLoaded => write!(f, "model is not loaded"),
            Self::Loading => write!(f, "model is currently loading"),
            Self::Evicting => write!(f, "model is being evicted"),
            Self::GpuPoisoned => write!(
                f,
                "GPU context poisoned by a previous failed load — process restart required"
            ),
            Self::WrongKind(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for ModelError {}

// ---- State machine ----------------------------------------------------

/// Lifecycle state for one model slot.
///
/// Valid transitions (#35 — reflects the real edges after T5.1/T5.2):
/// ```text
/// NotLoaded ──load_model──▶ Loading ──success─────────────▶ Ready
///                                   ├──failure────────────▶ NotLoaded
///                                   └──cancel (future drop)▶ NotLoaded   (T5.1 LoadCancelGuard)
/// Ready     ──evict_model─▶ Evicting ──done──────────────▶ NotLoaded
///                                    └──thread panic/join timeout▶ NotLoaded  (T5.2 finish_evict
///                                                                  always reconciles the tracker)
/// ```
/// Edge cases enforced in code, not drawn above:
/// - `Loading` blocks `evict_model` (`begin_evict` → "cannot evict a model that
///   is currently loading") — eviction waits for the load to settle.
/// - A cancelled load (HTTP client disconnect drops the `load_model` future)
///   resets `Loading → NotLoaded` via the `LoadCancelGuard` `Drop` (T5.1), so
///   there is no permanent `Loading` wedge.
/// - `GpuPoisoned`/`WrongKind` are [`ModelError`] variants, not slot states; a
///   poison-class load failure leaves the slot `NotLoaded` and trips a separate
///   process-level gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelState {
    /// Model is registered (in config) but not in VRAM.
    NotLoaded,
    /// Load is in progress; a `spawn_blocking` call is running.
    Loading,
    /// Model is in VRAM and accepting requests.
    Ready,
    /// Eviction is in progress; engine thread is draining and will exit.
    Evicting,
}

// ---- Internal record --------------------------------------------------

struct ModelRecord {
    state: ModelState,
    /// The typed submit handle — `Some` only when `state == Ready`.
    /// R1: [`EngineHandleKind::TextGen`] or [`EngineHandleKind::Vision`]; the
    /// enum is the seam that lets later phases store embedding / media handles
    /// in the same slot.
    handle: Option<EngineHandleKind>,
    /// The engine OS thread — `Some` only when `state == Ready`.
    thread: Option<std::thread::JoinHandle<()>>,
    /// Updated by `get_handle` to track least-recently-used order.
    last_used: Instant,
    /// VRAM estimate from config. `0.0` means unknown (skip gating).
    vram_gb: f64,
    /// KV cache pool actually allocated at load time (GB).
    /// Set by `execute_load`; used by the L0 prompt-length gate.
    kv_cache_gb: f64,
    /// Maximum prompt length in tokens before the L0 gate rejects the request.
    /// `0` means the gate is disabled (model's `config.json` could not be parsed).
    /// Formula: `kv_cache_gb × 1 GiB / bytes_per_token(precision)`.
    max_prompt_tokens: usize,
    /// Raw, unclamped KV-pool token capacity — `compute_max_prompt_tokens`'s
    /// `formula_result` before the native-context/ratchet clamps. `0` when
    /// the gate is disabled, same convention as `max_prompt_tokens`. Used by
    /// the concurrent-admission check ([`EngineHandle::try_admit_tokens`]),
    /// never by the per-request L0 gate — see `compute_max_prompt_tokens`'s
    /// doc comment for why the two must stay separate.
    pool_capacity_tokens: usize,
    /// The resolved `max_num_seqs` this model was last loaded with (see
    /// `resolve_runtime_params`). `0` until the first successful load. Used
    /// by the chat handler's own-budget admission check
    /// (the project's internal engineering log's deferred "fix 3") — a request's
    /// `prompt_tokens + max_new_tokens` is only gated against
    /// `pool_capacity_tokens` when this is exactly `1`: a single-slot model
    /// has no other request to preempt when its own generation exhausts the
    /// pool, unlike a multi-slot model where that's already handled more
    /// gracefully (clean finalization, not a hang — see the same decisions
    /// entry). Not read for anything else; not the live scheduler cap.
    max_concurrent_streams: usize,
    /// The model's Jinja chat template, loaded on first successful load.
    /// Empty until `state == Ready` (callers only read it via the Ready path).
    template: Arc<str>,
    /// Tool-call dialect detected from `template` at load time.
    family: ModelFamily,
    /// Special tokens from `tokenizer_config.json`, injected into the Jinja context.
    eos_token: Arc<str>,
    bos_token: Arc<str>,
    /// Kind resolved from `model_kinds` config at registration. Default:
    /// `TextGen`. Stable for the process lifetime — used in `ModelInfo` so the
    /// admin listing shows kind for models that have never been loaded.
    configured_kind: ModelKind,
    /// The kind this model was last loaded as — `None` until its first
    /// successful load, then retained across eviction. Drives `model_loaded`
    /// in `metrics_snapshot`: an evicted model keeps reporting `model_loaded 0`
    /// (with its known `kind` label) instead of leaving a stale `1` behind.
    last_kind: Option<ModelKind>,
    /// Wall-clock duration of this model's most recent successful load (the
    /// `spawn_blocking(factory.load(..))` window in `execute_load` — engine
    /// construction/JIT-compile time, not admission-check or HTTP overhead).
    /// `None` until the first successful load, then retained across eviction
    /// (same lifetime as `last_kind`) so `rustedvino_model_load_duration_seconds`
    /// keeps reporting the last-known figure for a model that's since been
    /// evicted, instead of the gauge disappearing.
    last_load_duration_secs: Option<f64>,
    /// Co-residency Slice 1: when `true`, this model is never an eviction victim
    /// (`select_eviction_victim` hard-excludes it). Set once at registration
    /// from `config.model_policies`; static for the process lifetime.
    pinned: bool,
    /// Co-residency Slice 1: soft eviction-order weight — higher is evicted
    /// later. Static, set at registration from `config.model_policies`.
    priority: i32,
    /// Co-residency Slice 3: when `false`, this model is never an eviction
    /// victim — a hard exclude alongside `pinned`. Set once at registration
    /// from `config.model_policies`; static for the process lifetime. Absent
    /// policy ⇒ `true` (the pre-feature behaviour: evictable).
    evictable: bool,
    /// Phase C1: the `OpenVINO` device this model loads on — the per-model
    /// `device` policy override if set, else the global `config.device`. Resolved
    /// once at registration; threaded into `execute_load`'s `factory.load` call
    /// and surfaced as the per-model `device` metrics label.
    device: String,
    /// Phase C1: the memory domain `device` belongs to (a discrete GPU's own
    /// name, or the shared `system` domain for iGPU/CPU/NPU). Resolved once at
    /// registration via [`resolve_device_domain`]. The domain `ensure_vram_for`
    /// admits into and `select_eviction_victim` evicts within for this model, so
    /// a model placed off the primary device is charged to the right pool.
    domain: String,
    /// Whether to run `extract_thinking` on buffered responses from this model
    /// (gap-2 vs OVMS). Set from [`config::ModelPolicy::reasoning_parser`] at
    /// registration; auto-detected from the chat template at first load if still
    /// `None` (a template containing `enable_thinking` → `Some(Qwen3)`). `None`
    /// means no scan — correct for non-thinking models and prevents false-positive
    /// strips on models that emit `<think>` as literal content.
    reasoning_parser: Option<config::ReasoningParser>,
    /// NPU-only `MAX_PROMPT_LEN` override from
    /// [`config::ModelPolicy::max_prompt_len`], set at registration. `None`
    /// means the NPU chat gate falls back to
    /// `crate::handlers::chat::NPU_DEFAULT_MAX_PROMPT_LEN`. Ignored for
    /// every non-NPU model. See the project's internal engineering log #6.
    configured_max_prompt_len: Option<u32>,
    /// The model's own shipped sampling defaults, from `generation_config.json`
    /// (see [`template::load_generation_defaults`]). Loaded once at first
    /// successful load, alongside `template`/`eos_token`/`bos_token` — empty
    /// (`Default::default()`, all fields `None`) until then and for
    /// media/embedding models.
    generation_defaults: template::GenerationDefaults,
    /// The model's native (trained/converted) context ceiling —
    /// `max_position_embeddings`, `rope_scaling`-adjusted — from
    /// [`template::read_native_context_limit`]. A hardware-independent model
    /// property, unrelated to `max_prompt_tokens`'s VRAM-derived formula:
    /// a KV pool can easily afford more tokens than the model was ever
    /// trained on (confirmed live 2026-08-23,
    /// the project's internal engineering log). Resolved
    /// once at registration (a static disk property, unlike `max_prompt_tokens`/
    /// `kv_cache_gb` which are load-dependent) — so it's visible via
    /// `ModelInfo` even for a `NotLoaded` model, and survives eviction.
    /// `None` when `config.json` is absent/unparseable/lacks the field
    /// (fail-open — same policy as [`crate::ov_embed::resolve_max_seq_len`]).
    native_context_limit: Option<usize>,
}

impl ModelRecord {
    /// The victim-ordering sort key for a Ready eviction candidate:
    /// `(busy, priority, last_used)`, ascending. A busy model (in-flight `> 0`,
    /// read live through the `ManagedEngine` facade) sorts LAST; then the
    /// lowest `priority`; then the least-recently-used `last_used`. Busy and
    /// priority are *soft* (weighted, not excluded) — hard exclusion is
    /// [`eviction_protected`](Self::eviction_protected).
    ///
    /// The single source of victim *ordering* (T4/F6): both the dry-run
    /// admission check ([`check_admission`](ModelManager::check_admission)) and
    /// the live picker ([`select_eviction_victim`](ModelManager::select_eviction_victim))
    /// rank by this, so the "what would evict" preview can never disagree with
    /// the model actually evicted.
    fn eviction_sort_key(&self) -> (bool, i32, Instant) {
        let busy = self
            .handle
            .as_ref()
            .is_some_and(|h| h.as_managed().active() > 0);
        (busy, self.priority, self.last_used)
    }
}

// ---- Public chat-routing context --------------------------------------

/// Everything the chat handler needs to serve one request for a model:
/// the engine submit handle plus the prompt-building inputs (chat template
/// and tool-call dialect). Returned by [`ModelManager::get_chat_context`].
pub struct ChatContext {
    /// Cloneable, typed submit handle for the model's engine. R1:
    /// [`EngineHandleKind::TextGen`] or [`EngineHandleKind::Vision`]; the chat
    /// handler matches on it for capability routing (text→CB, vision→VLM).
    pub handle: EngineHandleKind,
    /// The model's Jinja chat template (for `prompt_builder::build_prompt`).
    pub template: Arc<str>,
    /// The model's tool-call dialect (for `prompt_builder::parse_tool_calls`).
    pub family: ModelFamily,
    /// Special tokens from `tokenizer_config.json`, injected into the Jinja context.
    pub eos_token: Arc<str>,
    pub bos_token: Arc<str>,
    /// Maximum prompt length in tokens before the L0 gate rejects the request.
    /// `0` means the gate is disabled (model's `config.json` could not be parsed).
    pub max_prompt_tokens: usize,
    /// Raw, unclamped KV-pool token capacity — see
    /// `ModelRecord::pool_capacity_tokens`'s doc comment. Used by the
    /// concurrent-admission check, not the per-request L0 gate.
    pub pool_capacity_tokens: usize,
    /// See `ModelRecord::max_concurrent_streams`'s doc comment — only read
    /// by the chat handler's own-budget admission check, gated on `== 1`.
    pub max_concurrent_streams: usize,
    /// Whether the chat handler should run `extract_thinking` on this model's
    /// buffered responses. `None` = no scan; `Some(_)` = strip `<think>` and
    /// surface reasoning as `reasoning_content`. See [`config::ReasoningParser`].
    pub reasoning_parser: Option<config::ReasoningParser>,
    /// NPU-only `MAX_PROMPT_LEN` override (the project's internal engineering log #6). `None` means
    /// the NPU chat gate falls back to
    /// [`crate::handlers::chat::NPU_DEFAULT_MAX_PROMPT_LEN`]. Ignored for
    /// every non-NPU model.
    pub configured_max_prompt_len: Option<u32>,
    /// The model's own shipped sampling defaults (`generation_config.json`),
    /// resolved by [`crate::handlers::chat::resolve_sampling_defaults`]
    /// against the request's `tools_active` state and any client-explicit
    /// values. All-`None` for media/embedding models and for any model whose
    /// `generation_config.json` didn't pass [`template::load_generation_defaults`]'s
    /// gate.
    pub generation_defaults: template::GenerationDefaults,
}

// ---- Public info type (for /v1/models) --------------------------------

/// Snapshot of a model's state, returned by [`ModelManager::list_models`].
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub state: ModelState,
    pub kind: ModelKind,
    pub vram_gb: f64,
    /// The `OpenVINO` device this model is (or will be) placed on — e.g.
    /// `"GPU"`, `"CPU"`, `"NPU"`. Resolved once at registration (see
    /// `ModelRecord::device`), so this is accurate even before the model's
    /// first load, not just while `Ready`.
    pub device: String,
    /// The model's own resolved sampling defaults (`generation_config.json`,
    /// post sanity-check — see [`template::load_generation_defaults`]), if
    /// any. All `None` before the model's first successful load, for
    /// media/embedding models, or when the file didn't pass the loader's
    /// gate.
    pub generation_defaults: template::GenerationDefaults,
    /// The L0 prompt-length gate ceiling (`ModelRecord::max_prompt_tokens`) —
    /// the number a `400 context_length_exceeded` would cite. `0` before the
    /// model's first successful load, once evicted, for media/embedding
    /// models, or when the gate is disabled (`config.json` unparseable).
    /// Exposed so an external client/operator can discover the real ceiling
    /// proactively instead of only reactively via that error (`dev/autotest/
    /// 20260823_external_metrics_interface_audit.md`).
    pub max_prompt_tokens: usize,
    /// The KV cache pool actually reserved for this model at load time (GB),
    /// `0.0` before first load / once evicted. Companion figure to
    /// `max_prompt_tokens` — together they answer "how much context can I
    /// send, and how much pool is that costing."
    pub kv_cache_gb: f64,
    /// The model's native (trained/converted) context ceiling —
    /// `max_position_embeddings`, `rope_scaling`-adjusted (see
    /// [`template::read_native_context_limit`]). Contrast with
    /// `max_prompt_tokens`: this is what the model was *built for*; that is
    /// what this box's KV pool can currently *serve* — the two can differ
    /// wildly (confirmed live 2026-08-23, `dev/autotest/
    /// 20260823_max_position_embeddings_gate_gap.md`: a VRAM-derived ceiling
    /// of 142k-306k tokens against a real trained ceiling of 40,960).
    /// Resolved once at registration — a static disk property, so it's
    /// accurate even before the model's first load and survives eviction.
    /// `None` when `config.json` is absent/unparseable/lacks the field.
    pub native_context_limit: Option<usize>,
    /// Live KV-cache occupancy percentage (0-100), same source and same
    /// unloaded/unsupported-engine-kind caveats as
    /// [`ModelMetrics::kv_cache_usage_pct`] — surfaced here too
    /// (`dev/plans/kv-cache-pressure-detection.md`'s ops-review finding)
    /// so an operator not watching Prometheus can see it via
    /// `GET /v1/admin/models` directly.
    pub kv_cache_usage_pct: f64,
    /// Whether this model is currently flagged for sustained KV pressure —
    /// see [`ModelMetrics::kv_pressure_flagged`] for the exact semantics.
    /// Same reasoning for surfacing it here: a gauge nobody looks at is not
    /// a monitor.
    pub kv_pressure_flagged: bool,
}

/// Result of the dry-run admission check ("what fits"), returned by
/// The live system-RAM admission gate — the *second* of the two conditions
/// [`ModelManager::ensure_vram_for`] admits on, and the one the dry-run used to
/// ignore.
///
/// Inert on discrete-GPU domains: [`ModelManager::live_system_avail_gb`] returns
/// `None` for anything but the shared UMA `system` domain, so both fields are
/// `None` there and [`Self::admits`] is unconditionally `true`. That is not a
/// shortcut — on a discrete box a load genuinely is admitted on logical VRAM
/// alone, and charging it the full KV target would refuse models that load fine.
struct RamGate {
    /// Live `MemAvailable`-derived headroom in GB, or `None` off the `system`
    /// domain.
    avail: Option<f64>,
    /// `weights + conservative KV + margin`, in GB — `None` off the `system`
    /// domain.
    needed: Option<f64>,
    /// OS reservation, in GB, that the gate must leave untouched.
    floor: f64,
}

impl RamGate {
    /// Would this load be admitted once `freed` GB of residents are evicted?
    /// Eviction frees real RAM, which is why `ensure_vram_for` re-reads
    /// availability on every retry rather than deciding once.
    fn admits(&self, freed: f64) -> bool {
        match (self.avail, self.needed) {
            (Some(avail), Some(need)) => {
                os_memory::ram_floor_admits(avail + freed, need, self.floor)
            }
            _ => true,
        }
    }
}

/// [`ModelManager::check_admission`] (co-residency Slice 1).
///
/// Runs the [`ensure_vram_for`](ModelManager::ensure_vram_for) arithmetic
/// **without loading or evicting anything**: would the model fit, and which
/// residents would have to be evicted to make room? This is the admin-facing
/// "tell it what to pin; it tells you what fits" promise over the existing
/// deterministic engine — no GPU work.
#[derive(Debug, Clone)]
pub struct AdmissionCheck {
    /// `true` if the model can be admitted — either it fits in current free
    /// VRAM, or it fits after evicting the models listed in `would_evict`.
    /// `false` means even evicting every *non-protected* resident (not pinned
    /// and `evictable`, co-residency Slices 1/3) leaves too little.
    ///
    /// On the UMA `system` domain this folds in the **live system-RAM gate**
    /// too, because that is the gate that actually refuses the load there.
    /// Reporting only the logical-VRAM verdict is what let a model pass
    /// `rv check` and then fail to load.
    pub fits: bool,
    /// VRAM the load needs: `weights + min_kv_cache_gb + vram_safety_margin_gb`
    /// (the same `total_minimum` the real load gate computes).
    pub needed_gb: f64,
    /// VRAM currently unreserved in the model's inference domain, before any
    /// simulated eviction. `0.0` when gating is disabled for the domain.
    pub free_gb: f64,
    /// *Non-protected* residents — not pinned and `evictable` (co-residency
    /// Slices 1/3) — that would be evicted (in eviction order) to admit this
    /// model. Empty when it already fits, when gating is disabled, or when it
    /// cannot be admitted at all (`fits == false`).
    pub would_evict: Vec<String>,
    /// Live **system-RAM** need: `weights + conservative KV + margin`, the same
    /// `live_need` [`ensure_vram_for`](ModelManager::ensure_vram_for) gates on.
    ///
    /// `Some` **only on the shared UMA `system` domain** — the one place
    /// `live_system_avail_gb` returns a reading. On a discrete-GPU domain the
    /// RAM gate never fires, so a load really is admitted on `needed_gb` alone
    /// and reporting a RAM figure would be inventing a constraint that does not
    /// exist there. This asymmetry is the whole reason the dry-run cannot simply
    /// charge the full KV target unconditionally.
    ///
    /// Note this is a *different, larger* quantity than `needed_gb`: it counts
    /// the resolved per-model KV target (capped by `cache_size_gb`), not the
    /// `min_kv_cache_gb` floor.
    pub ram_needed_gb: Option<f64>,
    /// Live `MemAvailable`-derived headroom for the `system` domain, after the
    /// OS reservation floor, before any simulated eviction. `Some` exactly when
    /// [`Self::ram_needed_gb`] is.
    pub ram_avail_gb: Option<f64>,
    /// `true` if the model is already `Ready` — the check is moot (it is
    /// resident now), reported as `fits` with no eviction.
    pub already_loaded: bool,
}

/// Outcome of an on-demand load attempt from the chat path
/// ([`ModelManager::request_on_demand_load`], co-residency Slice 3c).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnDemandLoad {
    /// A background load was started (or one was already in progress): the
    /// caller should answer 503 `Loading` + `Retry-After` so the client retries.
    Loading,
    /// The model is **not** `load: on_demand` (it is eager) — no auto-load was
    /// started: the caller should answer the original `NotLoaded` 503, which a
    /// `Retry-After`-less response signals will not self-resolve.
    NotApplicable,
}

/// RTCC: which realtime slot a resolution request is for
/// (the project's internal engineering log's generalization of v2's
/// LLM-only [`ModelManager::resolve_realtime_llm`]). `Llm` covers both
/// text-gen and vision models (a VLM is still "the LLM slot" from the
/// session's point of view) — `Stt`/`Tts` are exact-kind matches, since
/// neither has a `realtime_viable_minimum` floor of its own (kind-matching
/// *is* their floor, D4). `Embed` is deliberately **grant-only** — v2's D2
/// and RTCC's framing decision 3 both exclude `embed_model` from the
/// courtesy floor ("convenience only, sessions degrade gracefully without
/// it"); `slot_default`/`clears_viable_minimum_for_slot` return `None`/
/// `false` unconditionally for it, so [`resolve_realtime_slot`]'s existing
/// step 3/4 (substitute / auto-load default) never fire for an embed
/// request — it only ever grants (already-`Ready`, or admits-and-loads) or
/// honestly reports the literal request failed. This variant exists purely
/// so `request_model_load` can classify an embed model id correctly instead
/// of falling into the `Llm` catch-all and risking an LLM model being
/// offered as a "substitute" for an embedding request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealtimeSlot {
    Llm,
    Stt,
    Tts,
    Embed,
}

/// D5: outcome of [`ModelManager::resolve_realtime_slot`] — what a realtime
/// session's slot actually gets, given what it asked for. See
/// the project's internal engineering log (LLM-only origin) and
/// the project's internal engineering log (STT/TTS generalization).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealtimeResolution {
    /// The model id the session should actually use. Never fabricated —
    /// either a real registered model, or (only in the no-default,
    /// nothing-viable edge case) the literal id the client requested.
    pub model_id: String,
    /// `"requested"` | `"default"` | `"substituted"` — matches D6's
    /// `model_selection` event `source` field verbatim, so callers can pass
    /// it straight through without translation.
    pub source: &'static str,
    /// Machine-usable reason code, present only when the literal request
    /// wasn't simply granted as-is — e.g. `"would_evict_protected"`,
    /// `"not_resident_no_headroom"`. `None` when `source == "requested"`
    /// with no background load needed.
    pub reason: Option<&'static str>,
    /// `true` when this resolution kicked off a background load (the
    /// resolved model is not yet `Ready`) — callers should expect turns to
    /// error until it becomes `Ready` (graceful per-turn degradation, not a
    /// failure of the resolution itself), and may want to say so to the
    /// client (D8's `model_loading` event).
    pub loading: bool,
}

// ---- Co-residency Slice 2: per-load runtime-param resolution -----------

/// Optional per-load overrides for a model's runtime parameters
/// (co-residency Slice 2 seam).
///
/// All fields `None` (the [`Default`]) means "use config" — what every
/// implicit load (chat-path, LRU, startup) passes. The
/// `POST /v1/admin/models/{id}/load` body is the one caller that fills these
/// (Slice 2b), via [`load_model_with_overrides`](ModelManager::load_model_with_overrides).
#[derive(Debug, Clone, Copy, Default)]
pub struct LoadOverrides {
    /// Override the KV-cache pool target (GB) for this load. `0.0` is not a
    /// valid override (use `None` for unbounded); the load endpoint validates.
    pub kv_cache_gb: Option<f64>,
    /// Override `max_num_seqs` (the CB scheduler + 429 gate) for this load.
    pub max_concurrent_streams: Option<usize>,
    /// Realtime voice arbitration v2 (D3): bypass the eviction-grace window
    /// (only the grace window — never the realtime serving set, `pinned`, or
    /// non-`evictable`) when this load needs to evict a recently-used model.
    /// Default `false`. The `POST /v1/admin/models/{id}/load` body is the
    /// only caller that sets this.
    pub force: bool,
}

/// A model's runtime parameters resolved at load time (co-residency Slice 2).
///
/// Produced by [`resolve_runtime_params`] from the precedence chain
/// **load override → per-model config policy → global config default →
/// builtin fallback**. Threaded through [`ModelManager::load_model`] into the
/// VRAM reservation ([`compute_kv_pool_gb`](ModelManager::compute_kv_pool_gb))
/// and the engine build ([`EngineFactory::load`](lifecycle::EngineFactory::load)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ResolvedRuntimeParams {
    /// KV-cache pool target in GB. `0.0` = unbounded (grab all remaining after
    /// weights + margin — the single-model default), otherwise the pool is
    /// clamped to this in `compute_kv_pool_gb`.
    pub kv_cache_gb: f64,
    /// `SchedulerConfig::max_num_seqs` for this model's engine and its HTTP 429
    /// admission cap. `0` keeps the `OpenVINO` default (256).
    pub max_num_seqs: usize,
}

/// Resolve a model's load-time runtime parameters via the Slice 2 precedence
/// chain: a load-request override wins, else the per-model
/// [`ModelPolicy`](config::ModelPolicy), else the global config default, else
/// the builtin fallback (`0.0` KV = unbounded, `config.max_num_seqs` streams).
///
/// Keeping this a free function over `&Config` (not a method) makes the
/// precedence directly unit-testable without a live `ModelManager`.
pub(crate) fn resolve_runtime_params(
    config: &Config,
    entry: Option<&config::ModelEntry>,
    overrides: &LoadOverrides,
) -> ResolvedRuntimeParams {
    let policy = entry.map(|e| &e.policy);
    let kv_cache_gb = overrides
        .kv_cache_gb
        .or_else(|| policy.and_then(|p| p.kv_cache_gb))
        .unwrap_or(config.default_kv_cache_gb);
    let max_num_seqs = overrides
        .max_concurrent_streams
        .or_else(|| policy.and_then(|p| p.max_concurrent_streams))
        .unwrap_or(config.max_num_seqs);
    ResolvedRuntimeParams {
        kv_cache_gb,
        max_num_seqs,
    }
}

// ---- MoE concurrency guard --------------------------------------------

/// Maximum `max_num_seqs` applied automatically to `MoE` models when no explicit
/// concurrency cap is set.  Qwen3-30B-A3B (128 experts) collapsed from
/// 62.7 → 25 tok/s at c=2 under CB's batched decode; c=1 avoids that.
const MOE_MAX_NUM_SEQS: usize = 1;

/// Cap `resolved` down to [`MOE_MAX_NUM_SEQS`] when `model_dir/config.json`
/// declares sparse expert routing and the cap was not set explicitly.
///
/// `cap_was_explicit` is `true` when either a load-request override or a
/// per-model policy already set `max_concurrent_streams` — in that case we
/// warn but respect the caller's choice.
fn apply_moe_cap(
    model_id: &str,
    resolved: usize,
    model_dir: &std::path::Path,
    cap_was_explicit: bool,
) -> usize {
    let Some(num_experts) = template::read_moe_num_experts(model_dir) else {
        return resolved;
    };
    // `resolved == 0` is the documented "let OpenVINO decide" sentinel
    // (CONFIG.md: `max_num_seqs` `0` = OV default 256) — NOT "already at or
    // below the cap." Treating it as already-safe here would silently hand
    // an MoE model 256-way concurrency, exactly the throughput collapse this
    // guard exists to prevent. Only a genuine 1..=MOE_MAX_NUM_SEQS value is
    // already safe.
    if resolved != 0 && resolved <= MOE_MAX_NUM_SEQS {
        return resolved; // already safe
    }
    if cap_was_explicit {
        tracing::warn!(
            model = %model_id,
            num_experts,
            max_num_seqs = resolved,
            "MoE model — CB throughput collapses above 1 concurrent sequence; \
             running at explicitly-requested cap {resolved}. \
             Expect degraded throughput at c>1.",
        );
        resolved
    } else {
        tracing::warn!(
            model = %model_id,
            num_experts,
            moe_cap = MOE_MAX_NUM_SEQS,
            "MoE model — capping max_num_seqs to {} to prevent CB throughput collapse. \
             Set model_policies.<id>.max_concurrent_streams to override.",
            MOE_MAX_NUM_SEQS,
        );
        MOE_MAX_NUM_SEQS
    }
}

// ---- Speculative-decoding concurrency guard ----------------------------

/// Maximum `max_num_seqs` applied automatically to speculative-decoding
/// models when no explicit concurrency cap is set. Multi-stream speculative
/// serving is completely unverified — every measurement in
/// the project's internal engineering log Part 0 used
/// `max_num_seqs=1`; batched-verification bandwidth-amortization math is an
/// open question (Part 0 fact #7).
const SPECULATIVE_MAX_NUM_SEQS: usize = 1;

/// Cap `resolved` down to [`SPECULATIVE_MAX_NUM_SEQS`] when `speculative` is
/// enabled and the cap was not set explicitly. Same warn-but-respect contract
/// as [`apply_moe_cap`] (Part 3 of the plan).
fn apply_speculative_cap(
    model_id: &str,
    resolved: usize,
    speculative: bool,
    cap_was_explicit: bool,
) -> usize {
    // Same `resolved == 0` sentinel caveat as `apply_moe_cap`: `0` means
    // "let OpenVINO decide" (256), not "already at or below the cap."
    if !speculative || (resolved != 0 && resolved <= SPECULATIVE_MAX_NUM_SEQS) {
        return resolved;
    }
    if cap_was_explicit {
        tracing::warn!(
            model = %model_id,
            max_num_seqs = resolved,
            "speculative decoding is unverified at c>1 — benefit may invert and \
             correctness is unvalidated under batching; running at explicitly- \
             requested cap {resolved}. Expect unpredictable throughput.",
        );
        resolved
    } else {
        tracing::warn!(
            model = %model_id,
            speculative_cap = SPECULATIVE_MAX_NUM_SEQS,
            "speculative decoding — capping max_num_seqs to {} (multi-stream \
             serving is completely unverified). Set \
             model_policies.<id>.max_concurrent_streams to override.",
            SPECULATIVE_MAX_NUM_SEQS,
        );
        SPECULATIVE_MAX_NUM_SEQS
    }
}

// ---- Speculative decoding: pairing validation (Layer B, disk-inspecting) --

/// Validate a draft/target pairing for speculative decoding — the
/// disk-inspecting checks that [`Config::validate`]'s pure-config Layer A
/// cannot make (they need the model directories on disk). See
/// the project's internal engineering log Part 4.
///
/// Not yet called anywhere in the load path (Migration step 2) — step 3 wires
/// it into both `build_not_loaded_record` (fail-fast at registration) and
/// `load_model_with_overrides` (the choke point every load, including startup
/// preloads, funnels through).
///
/// # Errors
/// Returns an error naming the specific gate that refused the pairing: target
/// not dense `TextGen`, target on NPU, target or draft is `MoE`, draft is not
/// a plain LLM directory (missing `openvino_model.xml` or is a VLM), tokenizer
/// (`vocab_size`) unreadable or mismatched, or an explicit `draft_device` that
/// disagrees with the target's resolved device.
pub(crate) fn validate_speculative_pairing(
    target_dir: &std::path::Path,
    draft_dir: &std::path::Path,
    target_kind: ModelKind,
    target_device: &str,
    spec: &config::SpeculativeConfig,
) -> anyhow::Result<()> {
    // Gate 1: target must be dense CB TextGen.
    anyhow::ensure!(
        target_kind == ModelKind::TextGen,
        "speculative decoding supports dense text-generation models only, got {target_kind:?}"
    );
    anyhow::ensure!(
        target_device != "NPU",
        "speculative decoding is not supported on NPU — the NPU TextGen branch \
         uses the static LLMPipeline, not ContinuousBatchingPipeline, and has no \
         draft support"
    );
    anyhow::ensure!(
        template::read_moe_num_experts(target_dir).is_none(),
        "speculative decoding target is a MoE model — measured as a net \
         throughput loss on the only MoE model tested (dev/DECISIONS.md \
         2026-07-17); blocked until separately verified"
    );

    // Gate 2: draft directory must be a plain dense LLM.
    anyhow::ensure!(
        draft_dir.is_dir() && draft_dir.join("openvino_model.xml").is_file(),
        "speculative decoding draft_model '{}' is not a plain LLM directory \
         (missing openvino_model.xml)",
        draft_dir.display()
    );
    anyhow::ensure!(
        !draft_dir
            .join("openvino_vision_embeddings_model.xml")
            .is_file(),
        "speculative decoding draft_model '{}' is a VLM directory — \
         draft_model()'s loader is hardcoded to the single-file LLM layout",
        draft_dir.display()
    );
    anyhow::ensure!(
        template::read_moe_num_experts(draft_dir).is_none(),
        "speculative decoding draft_model '{}' is a MoE model — unverified, \
         banned symmetrically with the target ban",
        draft_dir.display()
    );

    // Gate 3: tokenizer identity. `None` on either side is a hard refusal —
    // an unverifiable pairing is a garbage-output risk, not a pass.
    let target_vocab = template::read_vocab_size(target_dir);
    let draft_vocab = template::read_vocab_size(draft_dir);
    match (target_vocab, draft_vocab) {
        (Some(t), Some(d)) => anyhow::ensure!(
            t == d,
            "speculative decoding tokenizer mismatch: target vocab_size={t}, \
             draft vocab_size={d} — draft and target must tokenize identically"
        ),
        _ => anyhow::bail!(
            "speculative decoding cannot verify tokenizer compatibility — \
             vocab_size unreadable on target ({}) and/or draft ({})",
            target_dir.display(),
            draft_dir.display()
        ),
    }
    // model_type mismatch is a smell, not a refusal — a shared vocab_size is
    // the load-bearing check; families genuinely sharing a tokenizer would
    // otherwise be falsely refused.
    if let (Some(t), Some(d)) = (
        template::read_model_type(target_dir),
        template::read_model_type(draft_dir),
    ) && t != d
    {
        tracing::warn!(
            target_model_type = %t,
            draft_model_type = %d,
            "speculative decoding pairing has differing model_type — vocab_size \
             matched so the load proceeds, but cross-family pairings have shown \
             latency alone does not predict a win (dev/DECISIONS.md 2026-07-17 \
             \"First non-Qwen target tested\")",
        );
    }

    // Gate 4: device. v1 restricts draft_device to equal the target's
    // resolved device — implementation-simplicity scope decision, not
    // evidence cross-device drafting doesn't work (dev/DECISIONS.md
    // 2026-07-17 "Correction: cross-device drafting is direction-dependent").
    if let Some(d) = &spec.draft_device {
        anyhow::ensure!(
            d == target_device,
            "speculative decoding draft_device '{d}' must equal the target's \
             resolved device '{target_device}' — v1 does not support \
             cross-device drafting"
        );
    }

    Ok(())
}

// ---- Metrics snapshot (for /metrics) ----------------------------------

/// Per-Ready-model engine counters, captured by [`ModelManager::metrics_snapshot`].
// Independent flags reported as separate Prometheus series (loaded/pinned/
// usage-supported/pressure-flagged) — not alternative states of one thing,
// so an enum wouldn't reduce anything here. `kv_pressure_flagged` pushed
// this over the lint's default threshold.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone)]
pub struct ModelMetrics {
    /// Model ID (the `model` Prometheus label).
    pub id: String,
    /// Phase C1: the `OpenVINO` device this model is placed on (the per-series
    /// `device` Prometheus label). Resolved per-model at registration, so a model
    /// running off the primary device (an iGPU embedder, a CPU model) is labelled
    /// with its *actual* device — fixing the pre-C1 global-label bug where every
    /// series inherited `config.device`.
    pub device: String,
    /// What this model does (the `kind` Prometheus label: `text_gen`/`vision`).
    /// For an evicted model this is the kind it was last loaded as, so the
    /// zeroed series keeps the same label set it had while loaded.
    pub kind: ModelKind,
    /// `true` while the model is Ready; `false` once evicted. Drives
    /// `rustedvino_model_loaded` (1/0) and forces the request gauges to 0 when
    /// unloaded so they don't read stale values after eviction.
    pub loaded: bool,
    /// Accepted-but-unfinished requests (`rustedvino_requests_running`). `0` when unloaded.
    pub active: usize,
    /// Configured concurrency cap (`rustedvino_requests_max`). `0` when unloaded.
    pub max_seqs: usize,
    /// Requests admitted/queued but not yet actively processed
    /// (`rustedvino_requests_waiting`). `0` when unloaded.
    pub waiting: usize,
    /// KV cache pool reserved at load (GB) — the `compute_kv_pool_gb` result
    /// stored on the record (`rustedvino_kv_cache_pool_gb`). `0.0` when unloaded
    /// (eviction zeroes it), so the gauge tracks the live reservation.
    pub kv_cache_pool_gb: f64,
    /// Operator `pinned` policy (`rustedvino_model_pinned`, 1/0). Static for the
    /// process lifetime, so it reports the same value loaded or evicted.
    pub pinned: bool,
    /// Operator soft eviction-order weight (`rustedvino_model_priority`). Static
    /// for the process lifetime — higher is evicted later.
    pub priority: i32,
    /// Live KV-cache occupancy percentage (0–100) from the engine's last step
    /// (`rustedvino_kv_cache_usage_percent`). `0.0` when unloaded, idle, or —
    /// for the VLM engine specifically — always, regardless of real
    /// occupancy: it has a real batched pool but no public API to query it.
    /// See [`Self::kv_cache_usage_supported`] before trusting this at face
    /// value for a given model.
    pub kv_cache_usage_pct: f64,
    /// Whether `kv_cache_usage_pct` above is a trustworthy live reading for
    /// this model (`rustedvino_kv_cache_usage_supported`, 1/0) — `false` for
    /// an unloaded model or an engine kind whose `0.0` can't be told apart
    /// from a genuinely empty pool (see
    /// `ManagedEngine::cache_usage_supported`'s doc comment).
    pub kv_cache_usage_supported: bool,
    /// Wall-clock duration (seconds) of this model's most recent successful
    /// load (`rustedvino_model_load_duration_seconds`) — engine
    /// construction/JIT-compile time only, not admission-check or HTTP
    /// overhead. `None` only in the narrow window before a model's first
    /// load has ever completed; in practice always `Some` here, since this
    /// struct is only built for a model with a known `kind` (`last_kind`),
    /// set in the same `execute_load` block as this field. Retained across
    /// eviction — an evicted model keeps reporting its last-known load time
    /// instead of the gauge disappearing (same lifetime as `kind`/`loaded`).
    pub load_duration_secs: Option<f64>,
    /// Whether this model's KV occupancy has been continuously at/above
    /// `kv_pressure_threshold_pct` for at least `kv_pressure_sustained_secs`
    /// (`rustedvino_kv_cache_pressure_flagged`, 1/0) — see
    /// `dev/plans/kv-cache-pressure-detection.md`. Always `false` while
    /// `kv_pressure_monitor_enabled` is off, for an unloaded model, or for a
    /// `kind` where `kv_cache_usage_supported` is `false` (a permanently-0.0
    /// reading can never cross a positive threshold, so this never
    /// spuriously flags an engine kind that can't report real occupancy).
    pub kv_pressure_flagged: bool,
}

/// Live state for the `/metrics` endpoint: the inference device, total VRAM,
/// and one [`ModelMetrics`] per Ready model.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// `OpenVINO` device string (the `device` Prometheus label, e.g. `"GPU.1"`).
    pub device: String,
    /// Configured total VRAM in GB (`rustedvino_vram_total_bytes` = ×1e9).
    pub total_vram_gb: f64,
    /// VRAM currently reserved across all loaded models of every kind
    /// (`rustedvino_vram_used_bytes` = ×1e9). The sum tracked by `VramTracker`,
    /// so it returns to zero once everything is evicted (R2 alloc==free).
    pub used_vram_gb: f64,
    /// One entry per model that has been loaded at least once this process
    /// lifetime — Ready models (`loaded == true`) and evicted ones
    /// (`loaded == false`). Including evicted models lets `record_state` emit
    /// `model_loaded 0` for them instead of leaving a stale `1`. Models never
    /// loaded are omitted (no kind/series to report yet).
    pub models: Vec<ModelMetrics>,
    /// Per-domain VRAM breakdown: `(domain_id, total_gb, used_gb)` for every
    /// tracked memory domain. Emitted as domain-labelled Prometheus gauges so
    /// multi-GPU boxes expose per-device utilisation alongside the aggregate.
    pub domain_vram: Vec<(String, f64, f64)>,
}

/// Result of [`ModelManager::reload_config`] — what changed after re-reading
/// `config.json` from disk.
#[derive(Debug, Default)]
pub struct ConfigReloadReport {
    /// Model IDs newly present in the file and now registered `NotLoaded`.
    pub added: Vec<String>,
    /// Model IDs already registered `NotLoaded` whose config fields (`vram_gb`,
    /// kind, or device) changed and were refreshed in place.
    pub updated: Vec<String>,
    /// Model IDs the file declares but whose directory doesn't exist under
    /// `models_dir` — skipped, same guard as startup registration.
    pub skipped_missing_files: Vec<String>,
    /// Model IDs the file declares that reload deliberately left untouched
    /// because they are `Ready`, `Loading`, or `Evicting` — reload never
    /// disrupts a live or in-flight model.
    pub left_untouched: Vec<String>,
    /// Count of entries present in both the file and the live registry with
    /// no field differences — nothing to do.
    pub unchanged: usize,
    /// Names of **global** (non-model) config fields whose value in the file
    /// differs from the value this process booted with, sorted.
    ///
    /// Reload deliberately re-reads only the per-model registry;
    /// [`ModelManager::config`] is an owned boot-time snapshot and nothing
    /// swaps it. That was always the documented design, but it was invisible at
    /// runtime: an operator who edited `min_kv_cache_gb`, reloaded, and got a
    /// `200 OK` had no way to learn their edit did nothing. This field says so.
    ///
    /// **Non-empty means "not applied", not "will apply on restart".** Some of
    /// these (`bind_addr`, `port`, `cors_allowed_origins`, `ov_cache_dir`,
    /// `supervisor`) genuinely need a restart because boot latched them into a
    /// bound socket or built middleware; `total_vram_gb`, `domain_budgets` and
    /// `device_budgets` seed the `MemoryTracker` (and a second frozen copy in
    /// `AppState`) whose live accounting would desync if rewritten under
    /// resident models. Others (`min_kv_cache_gb`, `cache_size_gb`,
    /// `vram_safety_margin_gb`, …) are read at point of use on every load and
    /// *could* be refreshed live — that is simply not built. The distinction is
    /// deliberately not encoded here: reporting it would promise a taxonomy this
    /// code does not yet enforce.
    pub globals_not_applied: Vec<String>,
}

/// Result of [`ModelManager::patch_model`] — which supplied fields landed
/// where, plus the full merged entry (the only place a caller can currently
/// read back a model's complete policy — `GET /v1/admin/models` doesn't
/// carry it).
#[derive(Debug)]
pub struct PatchModelReport {
    /// Field names already in effect — immediately, for a `Ready` model too.
    pub applied_live: Vec<&'static str>,
    /// Field names persisted but only changing engine behaviour at this
    /// model's next load.
    pub effective_on_next_load: Vec<&'static str>,
    /// Field names supplied whose value already matched — accepted, a no-op.
    pub unchanged: Vec<&'static str>,
    /// The full entry after merging, as it now reads in `config.json`.
    pub entry: config::ModelEntry,
}

/// Where [`ModelManager::set_preload`] gets its new `preload` list from.
#[derive(Debug)]
pub enum PreloadSource {
    /// Replace `preload` with exactly this list (order preserved).
    Explicit(Vec<String>),
    /// Snapshot whichever registered models are currently `Ready`,
    /// alphabetically sorted for a deterministic result.
    FromLive,
}

/// Result of [`ModelManager::set_preload`] — the new list plus what changed
/// relative to the old one, and (for `from_live: true`) which `Ready`
/// models were candidates but got excluded.
#[derive(Debug, Default)]
pub struct SetPreloadReport {
    /// The `preload` list now in `config.json`.
    pub preload: Vec<String>,
    /// The list before this call.
    pub previous: Vec<String>,
    /// Model IDs newly present in `preload`.
    pub added: Vec<String>,
    /// Model IDs removed from `preload`.
    pub removed: Vec<String>,
    /// `from_live` only: registered-but-not-`Ready` model IDs that were
    /// therefore excluded from the snapshot (not an error — a preload list
    /// only ever describes *future* boot state, so "not loaded right now"
    /// just means "not in this snapshot," never a failure).
    pub skipped_not_ready: Vec<String>,
}

/// Result of [`ModelManager::reload_keys_file`] — what changed after
/// re-reading the keys file from disk. Deliberately independent of
/// [`ConfigReloadReport`] — see `reload_keys_file`'s doc comment for why key
/// rotation has its own reload path.
#[derive(Debug, Default)]
pub struct KeysReloadReport {
    /// `true` when `api_keys` on disk differed from the live value and was
    /// swapped in.
    pub keys_updated: bool,
    /// `true` when `admin_api_keys` on disk differed from the live value and
    /// was swapped in. Stays `false` when `admin_downgrade_refused` is `true`
    /// (the live value is deliberately left alone in that case).
    pub admin_keys_updated: bool,
    /// `true` when the file's new `admin_api_keys` was empty while a
    /// non-empty admin scope was live — that specific downgrade was refused,
    /// not applied. See `reload_keys_file`'s doc comment.
    pub admin_downgrade_refused: bool,
    /// Live inference-key count *after* this reload (whether or not it
    /// changed) — never the keys themselves.
    pub api_key_count: usize,
    /// Live admin-key count *after* this reload (whether or not it
    /// changed) — never the keys themselves.
    pub admin_key_count: usize,
}

/// One problem [`ModelManager::audit_config`] found for a single `model_id`.
#[derive(Debug, Clone)]
pub struct ConfigAuditEntry {
    /// The model ID from `config.json`'s `models` map.
    pub model_id: String,
    /// The directory that should exist under `models_dir` for this entry.
    pub expected_path: String,
    /// `true` if `expected_path` does not exist — this entry can never load
    /// (the class of bug `model_exists`/`FilesMissing` guards against).
    pub files_missing: bool,
    /// `true` if this `model_id` also appears in `config.preload` — a missing
    /// directory here is fatal at next startup, not just unloadable.
    pub in_preload: bool,
    /// `true` if the entry exists on disk but isn't in the live registry yet
    /// (declared in the file after the process last started or reloaded) —
    /// resolved by calling `POST /v1/admin/config/reload`.
    pub pending_reload: bool,
}

/// Result of [`ModelManager::audit_config`] — every config.json model entry
/// with at least one problem. Read-only: never mutates config or the
/// registry, unlike [`reload_config`](ModelManager::reload_config).
#[derive(Debug, Default)]
pub struct ConfigAuditReport {
    /// Total `model_id` entries declared in `config.json`, healthy or not.
    pub total_declared: usize,
    /// Entries with at least one problem (currently: `files_missing` and/or
    /// `pending_reload`) — a healthy entry is omitted entirely.
    pub problems: Vec<ConfigAuditEntry>,
}

// ---- ModelManager -----------------------------------------------------

/// Manages the lifecycle of all known models: loading, eviction, and routing.
///
/// `ModelManager` is `Send + Sync` and should be wrapped in an `Arc`
/// before being stored in `AppState`.
pub struct ModelManager {
    config: Config,
    models: Arc<RwLock<HashMap<String, ModelRecord>>>,
    vram: Arc<RwLock<MemoryTracker>>,
    /// The memory domain every model is charged to until the Phase-C placement
    /// engine routes per-model. Pre-placement all models load on `config.device`,
    /// so this is that device's domain id (a discrete GPU's own name today; the
    /// inventory-resolved `"system"` domain for iGPU/CPU/NPU once wired in B-2).
    /// `ensure_vram_for` admits and reserves within this domain.
    inference_domain: String,
    factory: Arc<dyn EngineFactory>,
    next_request_id: Arc<AtomicU64>,
    /// Set to `true` when an `ov_cb_create` call returns `CL_OUT_OF_RESOURCES`
    /// or `CL_INVALID_EVENT` — both indicate that the `OpenCL` context for the
    /// inference device is permanently wedged for this process lifetime. While
    /// poisoned, `load_model` refuses any new loads with `ModelError::GpuPoisoned`
    /// (HTTP 503) rather than attempting a doomed GPU call.
    gpu_poisoned: AtomicBool,
    /// Serialises the `load_model` critical section (T5.2, committee findings
    /// 20/38). Without it, two concurrent loads of different models can both
    /// pick the same LRU victim — the loser's `begin_evict` then races to
    /// "already being evicted" and the load fails with a confusing 500 — and two
    /// heavy GPU JIT compiles would contend on the single inference device. With
    /// it, victim selection → VRAM reservation → engine build run one load at a
    /// time; the next load re-evaluates VRAM with the prior reservation already
    /// committed. An async [`tokio::sync::Mutex`] because the section spans
    /// `.await` points (the eviction join, the `spawn_blocking` engine load).
    load_lock: tokio::sync::Mutex<()>,
    /// Device inventory captured at startup — used for device validation and
    /// `ModelRecord` construction when models are added dynamically at runtime.
    inventory: Arc<DeviceInventory>,
    /// Runtime model entries added via `POST /v1/admin/models/add`, overlaying
    /// `config.models` at every policy-lookup site. Checked before `config.models`
    /// so dynamically added models participate in KV-cache resolution and
    /// placement without mutating the startup `Config`.
    dynamic_entries: RwLock<HashMap<String, config::ModelEntry>>,
    /// Config file path for atomic persistence of dynamically added entries.
    /// `None` in test builds (no file to write to).
    config_path: Option<std::path::PathBuf>,
    /// Shared voice flow pin — notified on every eviction so a pin never
    /// outlives its LLM (see `voice_pin::VoicePinManager::clear_if_llm`).
    /// `None` in test builds and whenever `AppState` isn't wired up yet.
    voice_pin: Option<Arc<crate::voice_pin::VoicePinManager>>,
    /// Shared, hot-swappable Bearer-key state — the *same* `Arc` the router's
    /// auth middleware reads (`AppState::auth`). `reload_keys_file` writes
    /// through this handle so a key rotation is visible to the next request
    /// with no router rebuild. `None` in test builds and whenever `AppState`
    /// isn't wired up yet — `reload_keys_file` then errors rather than
    /// silently doing nothing (unlike `voice_pin`'s "absent means not wired
    /// up, not a fault" convention: a caller that explicitly asked to reload
    /// keys and got silent success when nothing actually happened is a worse
    /// failure mode than a clear error).
    auth: Option<Arc<arc_swap::ArcSwap<crate::AuthConfig>>>,
    /// Resolved path to the keys file (`config::resolve_keys_file_path`),
    /// re-read on every [`reload_keys_file`](Self::reload_keys_file) call.
    /// Resolved once in `startup::bootstrap` and never changes at runtime —
    /// repointing the credential source requires a restart, so a config
    /// reload can never silently redirect where keys come from. `None` in
    /// test builds and whenever `with_keys_file_path` was never called.
    keys_file_path: Option<std::path::PathBuf>,
    /// Model ids with a KV-wedge recovery (`spawn_kv_wedge_recovery`)
    /// currently in flight — either the VLM hybrid-attention wedge or a solo
    /// plain-CB pool exhaustion (see [`POOL_EXHAUSTED_MARKER`]'s doc
    /// comment). Every request against a wedged model fails with the same
    /// marker until the evict+reload completes, so without this set each of
    /// those queued failures would redundantly re-spawn a recovery and —
    /// worse — race to write the ratchet sidecar from its own (possibly
    /// stale, since a wedged VLM pipeline's `perf_metrics` freeze) observed
    /// token count. First request to insert its model id wins the claim; the
    /// spawned task removes it on completion (success or failure) so a
    /// genuinely new wedge later can trigger recovery again.
    kv_wedge_recovery_inflight: Mutex<HashSet<String>>,
    /// Serialises every mutation of `config.json` on disk, plus (for
    /// `patch_model`/`set_preload`) the read-merge-validate-write sequence
    /// that produces it. `persist_model_entry`/`persist_model_removal` hold
    /// it only for their own read/write; `patch_model`/`set_preload` hold it
    /// for their whole body — nothing in either critical section `.await`s,
    /// so a plain `std::sync::Mutex` is correct (no risk of holding it
    /// across a suspension point). Without this, two concurrent admin
    /// mutations race on the same `config.json.tmp` path and can lose one
    /// write entirely; `patch_model` additionally needs it to stop two
    /// concurrent `PATCH`es on the same model from each merging against a
    /// stale pre-lock read and silently discarding the other's change.
    config_mutation_lock: Mutex<()>,
    /// Per-model KV-cache pressure tracking
    /// (`dev/plans/kv-cache-pressure-detection.md`) — a plain
    /// `std::sync::Mutex` since every access is a short in-memory read/write,
    /// never held across an `.await`. Absent entry means "never sampled over
    /// threshold" / "never resized." `over_threshold_since` is cleared on
    /// eviction (meaningless for an unloaded model — a fresh load starts
    /// clean), but `last_resized_at` deliberately survives eviction: the
    /// resize cooldown must survive the resize's own evict+reload cycle, and
    /// throttles resize churn on a time basis regardless of what triggered
    /// any particular eviction in between.
    kv_pressure: Mutex<HashMap<String, KvPressureState>>,
}

/// One model's live KV-pressure bookkeeping — see [`ModelManager`]'s
/// `kv_pressure` field.
#[derive(Debug, Default, Clone, Copy)]
struct KvPressureState {
    /// First instant this model was seen continuously at/above
    /// `kv_pressure_threshold_pct`; cleared the moment a sample drops back
    /// below it.
    over_threshold_since: Option<Instant>,
    /// Whether this model was flagged as of the END of the last sweep tick —
    /// stored explicitly, not re-derived from `over_threshold_since` plus
    /// "now" on every read. This matters more than it looks: re-deriving it
    /// fresh each tick from the same stored `since` timestamp would compare
    /// (`since`, `now`) against itself and could never observe a
    /// false→true edge while usage stays continuously above threshold
    /// across multiple ticks — exactly the main case this feature exists
    /// for (caught by this feature's own unit tests before it shipped:
    /// `pressure_state_flags_once_sustained_duration_elapses` failed against
    /// the derive-fresh version). Persisting the bit and comparing against
    /// the *previous* tick's value is what makes the edge detection real.
    flagged: bool,
    /// Instant this model's most recent `POST .../resize` completed — the
    /// cooldown gate compares against `kv_resize_cooldown_secs`.
    last_resized_at: Option<Instant>,
}

/// What [`update_pressure_state`] did to a model's flagged status this tick
/// — distinguishes "just crossed into/out of sustained pressure" (log-worthy)
/// from "no change" (log nothing, every tick would otherwise spam).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PressureTransition {
    /// Still/newly under threshold or not yet sustained long enough — no log.
    Unchanged,
    /// Just crossed into sustained pressure this tick.
    JustFlagged,
    /// Was flagged, usage dropped back under threshold this tick.
    JustCleared,
}

/// Pure state-transition logic for one model, one sweep tick — extracted
/// from [`ModelManager::run_kv_pressure_sweep_once`] specifically so it's
/// testable without a live engine/`ModelManager` at all (`cache_usage_pct`
/// only comes from a real compiled pipeline; this function needs none of
/// that, just the numbers). Mutates `state` in place and reports which kind
/// of transition (if any) just happened.
fn update_pressure_state(
    state: &mut KvPressureState,
    usage_pct: f64,
    threshold_pct: f64,
    sustained: Duration,
    now: Instant,
) -> PressureTransition {
    let was_flagged = state.flagged;
    if usage_pct >= threshold_pct {
        state.over_threshold_since.get_or_insert(now);
    } else {
        state.over_threshold_since = None;
    }
    state.flagged = state
        .over_threshold_since
        .is_some_and(|t| now.duration_since(t) >= sustained);
    match (was_flagged, state.flagged) {
        (false, true) => PressureTransition::JustFlagged,
        (true, false) => PressureTransition::JustCleared,
        _ => PressureTransition::Unchanged,
    }
}

// CRASH COURSE — why no `unsafe impl Send/Sync` here?
//   All fields of `ModelManager` are automatically `Send + Sync`:
//   - `Arc<RwLock<HashMap<String, ModelRecord>>>` — RwLock<T> is Send+Sync
//     when T is Send; all ModelRecord fields (EngineHandle, JoinHandle<()>,
//     ModelState, Instant, f64) implement Send+Sync.
//   - `Arc<RwLock<VramTracker>>` — same reasoning.
//   - `Arc<dyn EngineFactory>` — trait bound requires Send+Sync+'static.
//   - `Arc<AtomicU64>` — AtomicU64 is Send+Sync.
//   - `tokio::sync::Mutex<()>` — Send+Sync (the T5.2 load_lock).
//   - `std::sync::Mutex<HashSet<String>>` — Send+Sync when the contained
//     type is Send (String is); held only for brief insert/remove, never
//     across an `.await`.
//   Rust derives Send+Sync automatically; no unsafe annotation needed.

/// Resolves the memory domain that models load into (pre-placement), from the
/// configured inference device and the startup `inventory`.
///
/// A discrete GPU reports its own name as its `domain_id` (a private VRAM pool);
/// an integrated GPU / CPU / NPU reports the shared [`DOMAIN_SYSTEM`] domain. If
/// the device is not in the inventory — an empty inventory in GPU-free tests, or
/// a device `OpenVINO` enumerated but did not fully probe — we fall back to the
/// device name itself, which for the discrete fleet is identical to its domain.
///
/// [`DOMAIN_SYSTEM`]: crate::device_inventory::DOMAIN_SYSTEM
fn resolve_inference_domain(config: &Config, inventory: &DeviceInventory) -> String {
    resolve_device_domain(&config.device, inventory)
}

/// Resolves the memory domain for an arbitrary `OpenVINO` device (Phase C1).
///
/// Generalizes [`resolve_inference_domain`] from the single `config.device` to
/// any per-model device override: a discrete GPU maps to its own private VRAM
/// pool (its name *is* the domain id); an integrated GPU / CPU / NPU maps to the
/// shared [`DOMAIN_SYSTEM`] domain. A device not in the inventory (empty
/// inventory in GPU-free tests, or one OV enumerated but did not fully probe)
/// falls back to the device name itself — for the discrete fleet that is
/// identical to its domain.
///
/// [`DOMAIN_SYSTEM`]: crate::device_inventory::DOMAIN_SYSTEM
fn resolve_device_domain(device: &str, inventory: &DeviceInventory) -> String {
    inventory
        .get(device)
        .map_or_else(|| device.to_owned(), |d| d.domain_id.clone())
}

/// Registration-time projected-occupancy fit check (T5/F4): does `domain` have
/// room for `gb` more gigabytes given the immovable footprint (`projected`)
/// already placed there against its budget (`domain_total`)?
///
/// Mirrors [`MemoryTracker::fits`](vram::MemoryTracker::fits): a gating-disabled
/// (`total == 0.0`) or unknown domain is unbounded → always `true`. Used only to
/// *prefer* a fitting domain during placement; it never hard-rejects (see
/// [`placement::place_with_fit`]).
fn fits_projected(
    domain_total: &HashMap<String, f64>,
    projected: &HashMap<String, f64>,
    domain: &str,
    gb: f64,
) -> bool {
    match domain_total.get(domain) {
        Some(&total) if total > 0.0 => {
            let used = projected.get(domain).copied().unwrap_or(0.0);
            (total - used) >= gb
        }
        _ => true, // disabled or unknown domain → unbounded
    }
}

/// Resolve one model's device at registration via the C2 precedence chain,
/// fit-aware (T5/F4). Pure given the `inventory` + the `projected` immovable
/// occupancy; returns the validated device name.
///
/// Precedence (first match wins):
///  1. per-model explicit `device` (C1) — always wins, engine skipped;
///  2. empty inventory (GPU-free unit tests) — the auto engine needs a real
///     inventory, so fall back to embedding auto-place / `config.device`,
///     exactly the pre-C2 test behaviour;
///  3. per-model `tier_preference` override → fit-aware engine with that order;
///  4. embedding with an explicit `config.embedding_device` → that device
///     (graceful fallback preserved via `resolve_embedding_device`);
///  5. else → the kind's size-aware built-in tier preference → fit-aware engine.
///
/// The kind comes from the `model_kinds` config hint (the factory's
/// directory-sniff `detect_kind` is a load-time concern); an un-hinted model is
/// placed as a heavy LLM (dGPU-first), the safe default for the fleet's primary
/// models. The factory loads on *exactly* this device (it no longer re-resolves
/// embeddings), so the record's device, domain, and metrics label all agree with
/// the real load — closing the embedding-mislabel edge.
///
/// # Errors
/// An explicit per-model `device` the running `OpenVINO` did not enumerate, or a
/// `tier_preference` no device satisfies, is rejected here.
#[allow(clippy::too_many_arguments)]
fn resolve_model_device(
    model_id: &str,
    vram_estimate: f64,
    kind: Option<ModelKind>,
    policy: Option<&config::ModelPolicy>,
    config: &Config,
    inventory: &DeviceInventory,
    domain_total: &HashMap<String, f64>,
    projected: &HashMap<String, f64>,
) -> Result<String> {
    let is_embedding = kind == Some(ModelKind::Embedding);
    let fits = |dom: &str, gb: f64| fits_projected(domain_total, projected, dom, gb);
    let device = if let Some(d) = policy.and_then(|p| p.device.clone()) {
        // (1) explicit override — validated against the inventory below.
        d
    } else if inventory.devices().is_empty() {
        // (2) GPU-free tests: no auto engine.
        if is_embedding {
            lifecycle::resolve_embedding_device(config.embedding_device.as_deref(), &config.device)
        } else {
            config.device.clone()
        }
    } else if let Some(p) = policy.filter(|p| !p.tier_preference.is_empty()) {
        // (3) per-model tier-preference override.
        let tiers = parse_tier_preference(model_id, &p.tier_preference)?;
        placement::place_with_fit(model_id, &tiers, vram_estimate, inventory, fits)?
    } else if is_embedding && config.embedding_device.is_some() {
        // (4) explicit operator embedding device (keeps graceful fallback).
        lifecycle::resolve_embedding_device(config.embedding_device.as_deref(), &config.device)
    } else {
        // (5) size-aware built-in default for the kind. STT uses its own (smaller)
        // light/heavy threshold; everything else uses the LLM threshold.
        let resolved_kind = kind.unwrap_or(ModelKind::TextGen);
        let light_max_gb = if resolved_kind == ModelKind::Stt {
            config.light_stt_max_gb
        } else {
            config.light_model_max_gb
        };
        let mut pref =
            placement::default_tier_preference(resolved_kind, vram_estimate, light_max_gb);
        // A model whose declared size alone is this close to the *entire* dGPU
        // budget can never safely coexist with anything else there — exclude
        // the dGPU tier outright rather than let per-registration VRAM budget
        // contention (which depends on registration order and what else
        // happens to be pinned) route it away only after the fact.
        let dgpu_ceiling_gb = config.total_vram_gb * config.dgpu_size_ceiling_fraction;
        if config.dgpu_size_ceiling_fraction < 1.0 && vram_estimate >= dgpu_ceiling_gb {
            pref.retain(|t| *t != DeviceTier::Heavy);
        }
        let resolved = placement::place_with_fit(model_id, &pref, vram_estimate, inventory, fits)?;
        // Budget contention from an *earlier*, immovable (pinned/non-evictable)
        // model's reservation can still reroute this model to a weaker tier —
        // a legitimate decision, but otherwise a silent one. Compare against
        // the unconstrained answer (ignoring fit) and warn on a mismatch, so
        // "why is my 9.5GB model on CPU?" shows up in the boot log instead of
        // requiring a load-then-inspect round trip to discover.
        if let Ok(unconstrained) = placement::place(model_id, &pref, inventory)
            && unconstrained != resolved
        {
            tracing::warn!(
                model_id,
                resolved_device = %resolved,
                unconstrained_device = %unconstrained,
                vram_gb = vram_estimate,
                "model placement changed by VRAM budget contention — likely an earlier \
                 pinned/non-evictable model's reservation left insufficient headroom on the \
                 preferred device; check `pinned`/`evictable` settings and per-model `vram_gb` \
                 values if this is unexpected",
            );
        }
        resolved
    };
    // Validate-and-reject an explicit per-model `device` the running OpenVINO did
    // not enumerate (skipped for an empty inventory in tests). The tier-preference
    // / default paths self-validate via `place` (a preference no device satisfies
    // is already an error).
    if let Some(p) = policy
        && let Some(dev) = p.device.as_deref()
        && !inventory.devices().is_empty()
    {
        // HETERO:A,B is a virtual composite — not itself enumerated, but valid when
        // every component device is. Check components for HETERO, literal name otherwise.
        let is_valid = if let Some(components) = dev.strip_prefix("HETERO:") {
            components
                .split(',')
                .all(|c| inventory.get(c.trim()).is_some())
        } else {
            inventory.get(dev).is_some()
        };
        if !is_valid {
            let available: Vec<&str> = inventory
                .devices()
                .iter()
                .map(|d| d.name.as_str())
                .collect();
            anyhow::bail!(
                "model_policies['{model_id}'].device = '{dev}' is not an available \
                 OpenVINO device (enumerated: {available:?})"
            );
        }
    }
    Ok(device)
}

/// Parse a per-model `tier_preference` config list (kebab-case [`DeviceTier`]
/// labels) into tiers, erroring on an unrecognised label — validate-and-reject
/// at startup rather than silently dropping a typo. Mirrors how `model_kinds`
/// strings map through [`ModelKind::from_label`].
fn parse_tier_preference(model_id: &str, labels: &[String]) -> Result<Vec<DeviceTier>> {
    labels
        .iter()
        .map(|l| {
            DeviceTier::from_label(l).ok_or_else(|| {
                anyhow::anyhow!(
                    "model_policies['{model_id}'].tier_preference contains unknown tier '{l}' \
                     (valid: heavy, strong-igpu, weak-igpu, npu, fallback)"
                )
            })
        })
        .collect()
}

/// The derived budget (GB) for the shared "system" memory domain: total OS RAM
/// minus the (per-OS or configured) reservation, floored at zero. `0.0` when RAM
/// is unreadable (e.g. the not-yet-implemented Windows reader) → that domain is
/// left gating-disabled rather than guessed. An explicit `domain_budgets`
/// override is applied later in [`assemble_domain_budgets`], so it is not
/// consulted here.
fn system_domain_budget_gb(config: &Config) -> f64 {
    let reservation = config
        .system_ram_reservation_gb
        .unwrap_or_else(os_memory::default_reservation_gb);
    os_memory::total_ram_gb().map_or(0.0, |total| (total - reservation).max(0.0))
}

/// Assembles the per-domain budgets for the [`MemoryTracker`], given the
/// resolved `inference_domain` and the derived `system_budget_gb`.
///
/// Pure — no OS or OV calls — so the budget-assignment *policy* is unit-testable
/// without real hardware. Precedence per domain:
/// 1. explicit `config.domain_budgets[domain]` override (wins always; `0.0`
///    disables that domain's gating),
/// 2. the **inference** domain → `config.total_vram_gb` (the legacy knob is, by
///    definition, the budget for the device we run on → single-dGPU unchanged),
/// 3. the **system** domain → `system_budget_gb` (RAM − reservation),
/// 4. any other (secondary discrete GPU) domain → that device's reported VRAM,
///    or `0.0` (disabled) when it exposed no memory figure.
///
/// The inference domain is always present even with an empty inventory (GPU-free
/// tests), so the tracker always has the domain `ensure_vram_for` reserves into.
fn assemble_domain_budgets(
    config: &Config,
    inventory: &DeviceInventory,
    inference_domain: &str,
    system_budget_gb: f64,
) -> Vec<(String, f64)> {
    let derived = |domain: &str| -> f64 {
        if let Some(&o) = config.domain_budgets.get(domain) {
            o
        } else if domain == inference_domain {
            config.total_vram_gb
        } else if domain == DOMAIN_SYSTEM {
            system_budget_gb
        } else {
            inventory
                .get(domain)
                .and_then(|d| d.total_mem_gb)
                .unwrap_or(0.0)
        }
    };

    let mut budgets: HashMap<String, f64> = inventory
        .domains()
        .into_iter()
        .map(|domain| {
            let b = derived(&domain);
            (domain, b)
        })
        .collect();

    // Empty inventory (tests) discovers no domains — guarantee the inference one.
    budgets
        .entry(inference_domain.to_owned())
        .or_insert_with(|| derived(inference_domain));

    budgets.into_iter().collect()
}

/// Resolves the full per-domain budget list from config + inventory, computing
/// the system-domain budget from OS RAM. Thin wrapper over the pure
/// [`assemble_domain_budgets`] (which is where the policy lives + is tested).
fn resolve_domain_budgets(
    config: &Config,
    inventory: &DeviceInventory,
    inference_domain: &str,
) -> Vec<(String, f64)> {
    let system_budget = system_domain_budget_gb(config);
    assemble_domain_budgets(config, inventory, inference_domain, system_budget)
}

impl ModelManager {
    /// Creates a `ModelManager` using the production `OpenVINO` engine factory.
    ///
    /// This is the constructor called from `main.rs`. Tests use the
    /// `pub(crate) fn new()` overload which accepts a mock factory.
    ///
    /// A `preload` model that fails to load does not fail startup — see
    /// [`new_with_inventory`](Self::new_with_inventory).
    ///
    /// # Errors
    /// Fails only on a config/registration problem (e.g. a `preload` entry
    /// with no `models` entry, an unresolvable device). See
    /// [`new_with_inventory`](Self::new_with_inventory).
    pub async fn new_production(config: Config, inventory: Arc<DeviceInventory>) -> Result<Self> {
        // Parse the explicit kind hints once, warning on (and dropping) any
        // unrecognised label rather than panicking — an unknown hint just falls
        // back to directory detection.
        let model_kinds: HashMap<String, ModelKind> = config
            .models
            .iter()
            .filter_map(|(id, entry)| {
                let label = entry.kind.as_deref()?;
                let kind = ModelKind::from_label(label);
                if kind.is_none() {
                    tracing::warn!(
                        model_id = %id,
                        label = %label,
                        "unknown models[id].kind value — ignoring, falling back to detection"
                    );
                }
                kind.map(|k| (id.clone(), k))
            })
            .collect();

        let factory = OvEngineFactory {
            // cache_size_gb and max_num_seqs are resolved per-load by
            // load_model (co-residency Slice 2) — not factory fields.
            kv_cache_precision: config.kv_cache_precision.clone(),
            admission_queue_timeout_ms: config.admission_queue_timeout_ms,
            model_kinds: std::sync::RwLock::new(model_kinds),
            concurrent_streams: std::sync::RwLock::new(HashMap::new()),
            max_prompt_len_hints: std::sync::RwLock::new(HashMap::new()),
            draft_hints: std::sync::RwLock::new(HashMap::new()),
            image_provenance_hints: std::sync::RwLock::new(HashMap::new()),
            embedding_pooling: crate::ov_embed::Pooling::from_config(&config.embedding_pooling),
            embedding_normalize: config.embedding_normalize,
            ov_cache_dir: config.ov_cache_dir.clone(),
            enable_prefix_caching: config.enable_prefix_caching,
        };
        Self::new_with_inventory(config, Arc::new(factory), inventory).await
    }

    /// Set the config file path used to persist dynamically-added model entries.
    ///
    /// Call on the result of [`new_production`](Self::new_production) before
    /// wrapping in `Arc`. Without this, `add_model` still succeeds but skips
    /// disk persistence (entries survive only until process exit).
    #[must_use]
    pub fn with_config_path(mut self, path: std::path::PathBuf) -> Self {
        self.config_path = Some(path);
        self
    }

    /// Attach the shared voice flow pin so eviction can clear it (see
    /// `voice_pin::VoicePinManager::clear_if_llm`).
    ///
    /// Call on the result of [`new_production`](Self::new_production) before
    /// wrapping in `Arc`, with the **same** `VoicePinManager` instance handed to
    /// `AppState` — without this, evicting the pinned LLM leaves the pin stale.
    #[must_use]
    pub fn with_voice_pin(mut self, voice_pin: Arc<crate::voice_pin::VoicePinManager>) -> Self {
        self.voice_pin = Some(voice_pin);
        self
    }

    /// Attach the shared, hot-swappable auth state so `reload_config` can
    /// rotate `api_keys`/`admin_api_keys` live.
    ///
    /// Call with the *same* `Arc` also handed to `AppState::with_auth` —
    /// mirroring [`with_voice_pin`](Self::with_voice_pin)'s shared-instance
    /// pattern — or a key rotation here would update an `ArcSwap` the auth
    /// middleware never reads from.
    #[must_use]
    pub fn with_auth(mut self, auth: Arc<arc_swap::ArcSwap<crate::AuthConfig>>) -> Self {
        self.auth = Some(auth);
        self
    }

    /// Attach the resolved keys-file path so `reload_keys_file` knows where
    /// to re-read from. Call with `config::resolve_keys_file_path`'s result.
    #[must_use]
    pub fn with_keys_file_path(mut self, path: std::path::PathBuf) -> Self {
        self.keys_file_path = Some(path);
        self
    }

    /// Test-only peek at the live auth state — `reload_keys_file`'s report
    /// already surfaces what a real caller needs (counts, whether anything
    /// changed), but a test verifying "nothing mutated on a failed reload"
    /// needs to see the actual values before/after.
    #[cfg(test)]
    pub(crate) fn auth_snapshot_for_test(&self) -> Option<(Vec<String>, Vec<String>)> {
        self.auth.as_ref().map(|a| {
            let g = a.load();
            (g.api_keys.clone(), g.admin_api_keys.clone())
        })
    }

    /// Resolve device/domain and build a fresh `NotLoaded` [`ModelRecord`] for
    /// `(model_id, entry)` — the shape both [`add_model`](Self::add_model) and
    /// [`reload_config`](Self::reload_config) need for a model that isn't live
    /// yet. No projected-occupancy map is passed to placement: a single
    /// newly-registered model has no startup-time reservation to route
    /// around — fit-checking is deferred to its first `load_model` call, same
    /// as `add_model` always did.
    fn build_not_loaded_record(
        &self,
        model_id: &str,
        entry: &config::ModelEntry,
    ) -> Result<ModelRecord> {
        // No explicit `kind` in config: fall back to the same file-sniffing
        // `detect_kind` uses at real load time (`OvEngineFactory::load`),
        // instead of blindly assuming `TextGen`. Closes the gap where a VLM
        // registered without an explicit `"kind": "vision"` reported
        // `configured_kind: TextGen` (wrong — metrics/API display and, worse,
        // `resolve_model_device` below, which received `None` and lost the
        // kind-aware tier preference) even though the *actual* engine
        // construction at load time detected Vision correctly via the same
        // file-sniffing, independently — see `dev/autotest/
        // 20260717_qwen3_4b_int8_sigsegv.md` Finding 3.
        let resolved_kind = entry
            .kind
            .as_deref()
            .and_then(ModelKind::from_label)
            .unwrap_or_else(|| {
                self.factory
                    .detect_kind(&self.config.models_dir.join(model_id))
            });
        // Teach the factory this model's kind *before* any load reaches it —
        // including the auto-detected case: harmless (load() would
        // independently re-derive the identical answer from the same files)
        // and keeps the factory and this record from ever disagreeing.
        self.factory.register_kind_hint(model_id, resolved_kind);
        if let Some(cap) = entry.policy.max_concurrent_streams {
            self.factory.register_concurrent_streams_hint(model_id, cap);
        }
        // NPU max_prompt_len is registered later, at the single load choke point
        // (`load_model_with_overrides`, right before `execute_load`) — not here.
        // This function isn't called for startup-preloaded models (only
        // `add_model`/`reload_config`), so registering it only here would miss
        // exactly the common case (an NPU model preloaded from config.models).
        let resolved_device = resolve_model_device(
            model_id,
            entry.vram_gb,
            Some(resolved_kind),
            Some(&entry.policy),
            &self.config,
            &self.inventory,
            &HashMap::new(),
            &HashMap::new(),
        )?;
        let domain = resolve_device_domain(&resolved_device, &self.inventory);
        // Speculative decoding: fail fast at registration (add_model /
        // reload_config paths) rather than only discovering a bad pairing on
        // the model's next load. `load_model_with_overrides` re-validates at
        // the actual load choke point regardless — this is belt-and-braces
        // for the paths that go through this function (see Part 4 of
        // dev/plans/speculative-decoding-integration.md).
        if let Some(spec) = entry.policy.speculative.as_ref() {
            let draft_dir = self.config.models_dir.join(&spec.draft_model);
            validate_speculative_pairing(
                &self.config.models_dir.join(model_id),
                &draft_dir,
                resolved_kind,
                &resolved_device,
                spec,
            )?;
        }
        Ok(ModelRecord {
            state: ModelState::NotLoaded,
            handle: None,
            thread: None,
            last_used: Instant::now(),
            vram_gb: entry.vram_gb,
            kv_cache_gb: 0.0,
            max_prompt_tokens: 0,
            pool_capacity_tokens: 0,
            max_concurrent_streams: 0,
            template: Arc::from(""),
            family: ModelFamily::Default,
            eos_token: Arc::from(""),
            bos_token: Arc::from(""),
            configured_kind: resolved_kind,
            last_kind: None,
            last_load_duration_secs: None,
            pinned: entry.policy.pinned,
            priority: entry.policy.priority,
            evictable: entry.policy.evictable,
            device: resolved_device,
            domain,
            reasoning_parser: entry.policy.reasoning_parser,
            configured_max_prompt_len: entry.policy.max_prompt_len,
            generation_defaults: template::GenerationDefaults::default(),
            native_context_limit: template::read_native_context_limit(
                &self.config.models_dir.join(model_id),
            ),
        })
    }

    /// Register a new model at runtime, load it immediately, and persist it to
    /// config only once that load succeeds.
    ///
    /// This is the add-only counterpart to editing `config.json` and restarting:
    /// it validates the request, inserts a `NotLoaded` record, and triggers an
    /// immediate load. **Config persistence happens last, after the load
    /// succeeds** — not before. A failed load rolls back the in-memory
    /// registration too, so a failed add leaves no trace anywhere: no ghost
    /// `NotLoaded` entry in `self.models`, nothing written to `config.json`.
    /// Earlier this instead persisted first, so a doomed load (bad VRAM
    /// budget, unavailable device, …) left a permanent, possibly-wrong
    /// registry entry behind even though the caller only ever saw a `400`/
    /// `500` — "the add failed" silently didn't mean "nothing happened". The
    /// model is **not** added to `preload` — the config change survives
    /// restarts but the model only auto-loads on next restart if the operator
    /// explicitly adds it to `preload` (`POST /v1/admin/config/preload`).
    ///
    /// `policy` carries the full [`ModelPolicy`](config::ModelPolicy) —
    /// everything a static `config.json` stanza can express (`pinned`,
    /// `kv_cache_gb`, `tier_preference`, image-gen `precision`/`model_source`/
    /// `model_revision`, …), not just `device`. Every field is optional
    /// because the operator may not know all of it yet at add time; pass
    /// [`ModelPolicy::default()`](config::ModelPolicy) for the pre-feature
    /// four-field behaviour (`model_id`/`vram_gb`/`kind`/`device`).
    ///
    /// # Errors
    /// - Model directory not found under `models_dir`.
    /// - `vram_gb` non-finite or negative.
    /// - Unknown `kind` label.
    /// - `device` empty or not enumerated by `OpenVINO`.
    /// - Any `policy` field fails [`config::validate_model_entry`]'s invariants
    ///   (e.g. `max_concurrent_streams: 0`, `speculative` without `vram_gb > 0.0`).
    /// - Model ID already registered.
    /// - Immediate load failure — registration is rolled back, nothing persists.
    /// - Config persistence failure — rare (disk write/rename), and only
    ///   possible once the load has already succeeded, so the model is
    ///   `Ready` and serving even though this call still returns `Err`; the
    ///   registration just won't survive a restart until retried or fixed.
    #[allow(clippy::too_many_lines)]
    pub async fn add_model(
        &self,
        model_id: String,
        vram_gb: f64,
        kind: Option<String>,
        policy: config::ModelPolicy,
    ) -> anyhow::Result<()> {
        // ── Validate inputs ────────────────────────────────────────────────
        let model_dir = self.config.models_dir.join(&model_id);
        anyhow::ensure!(
            model_dir.is_dir(),
            "model directory not found: '{}' — check models_dir '{}' and the model_id",
            model_dir.display(),
            self.config.models_dir.display()
        );
        if let Some(ref k) = kind {
            anyhow::ensure!(
                ModelKind::from_label(k).is_some(),
                "unknown kind '{k}' — valid values: \
                 text_gen, vision, embedding, stt, tts, image_gen, reranking"
            );
        }
        if let Some(ref dev) = policy.device {
            anyhow::ensure!(!dev.is_empty(), "device must be a non-empty string");
            if !self.inventory.devices().is_empty() {
                let available: Vec<&str> = self
                    .inventory
                    .devices()
                    .iter()
                    .map(|d| d.name.as_str())
                    .collect();
                anyhow::ensure!(
                    self.inventory.get(dev).is_some(),
                    "device '{dev}' is not an available OpenVINO device \
                     (enumerated: {available:?})"
                );
            }
        }

        // ── Build ModelEntry, then run the same per-model invariants a
        // static config.json entry must pass (config::Config::validate) ────
        let entry = config::ModelEntry {
            vram_gb,
            kind,
            policy,
        };
        config::validate_model_entry(&model_id, &entry)?;

        // ── Duplicate check (fast path) ──────────────────────────────────────
        // Cheap up-front rejection before doing any of `build_not_loaded_record`'s
        // side-effecting work (it registers kind/concurrency hints on the
        // factory) — a duplicate `add_model` call must not perturb an
        // already-registered model's factory hints on its way to failing.
        // This is a fast path, not the sole correctness guarantee: the atomic
        // recheck below is what actually closes the race (see its comment).
        let in_models = self
            .models
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&model_id);
        let in_dynamic = self
            .dynamic_entries
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&model_id);
        anyhow::ensure!(
            !in_models && !in_dynamic,
            "model '{model_id}' is already registered — \
             use POST /v1/admin/models/{model_id}/load to load it"
        );

        // ── Resolve device/domain and build the record ──────────────────────
        let record = self.build_not_loaded_record(&model_id, &entry)?;

        // ── Register in runtime maps ONLY — deliberately NOT persisted to
        // config.json yet. The record must exist in `self.models` before
        // `load_model` can run (it's the single load choke point every path,
        // including this one, funnels through), but persisting here — before
        // knowing whether the load actually succeeds — would leave a
        // permanent, possibly-wrong-metadata `NotLoaded` entry in config.json
        // for a load that never worked: "the add failed" would stop meaning
        // "nothing happened". See
        // dev/autotest/20260804_doomed_load_evicts_everything_first.md's
        // "separate, smaller finding" for the live repro that flagged this.
        //
        // The `models` insert below is guarded by an ATOMIC recheck (same
        // write-lock hold as the insert itself) — closing a race an
        // independent review (Fable) caught in this fix's own rollback path:
        // two concurrent `add_model` calls for the same `model_id` could
        // otherwise both pass the fast-path check above before either
        // registered, and the loser's on-failure rollback would then delete
        // the winner's just-succeeded live registration out from under it —
        // a Ready model silently vanishing from `/v1/admin/models` while its
        // engine and VRAM stayed resident and un-deregisterable through any
        // normal API path. Claiming the slot in `self.models` atomically
        // means a second racer for the same ID now always fails here, before
        // ever reaching `load_model` — so no rollback can ever target a model
        // another winning call is using.
        {
            let mut models = self.write_models("add_model:register");
            anyhow::ensure!(
                !models.contains_key(&model_id),
                "model '{model_id}' is already registered — \
                 use POST /v1/admin/models/{model_id}/load to load it"
            );
            models.insert(model_id.clone(), record);
        }
        self.dynamic_entries
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(model_id.clone(), entry.clone());

        tracing::info!(
            model_id = %model_id,
            vram_gb,
            "model registered dynamically — triggering immediate load"
        );

        // ── Load immediately. On failure, roll back the in-memory
        // registration too — a failed add must leave no trace anywhere, not
        // even an in-memory `NotLoaded` ghost the caller never asked to keep
        // around (unlike a *pre-existing* model's failed reload, which
        // correctly stays registered — that's `abort_load`'s job, not this
        // one's). ────────────────────────────────────────────────────────────
        if let Err(e) = self.load_model(&model_id).await {
            self.write_models("add_model:rollback").remove(&model_id);
            self.dynamic_entries
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&model_id);
            return Err(e);
        }

        // ── Persist to config only now that the load has actually
        // succeeded — a durable entry means a durably-working one. ─────────
        self.persist_model_entry(&model_id, &entry)
    }

    /// Partially update an *already-registered* model's [`config::ModelEntry`]
    /// — `PATCH /v1/admin/models/{id}`'s implementation. The counterpart to
    /// `add_model` (register) and `deregister_model` (destroy): this is the
    /// missing "update" of the CRUD set, closing the gap where the only way
    /// to change a live model's `vram_gb`/`load` policy/etc. was evict +
    /// hand-edit `config.json` + `config/reload` + reload — a multi-step
    /// dance for what should be one call.
    ///
    /// Unlike `add_model` (persist-last, since the load can fail),
    /// **persistence happens before the live apply** here, but only after
    /// anything that can still reject the merged entry has already run —
    /// `build_not_loaded_record` (placement-touching fields: validates
    /// `tier_preference` labels and re-checks the speculative pairing,
    /// neither of which `validate_model_entry` covers) is called *before*
    /// the write, not after, so a rejection there leaves `config.json`
    /// untouched instead of persisting a stanza that only fails at the next
    /// boot. The remaining live-apply step (updating a few `ModelRecord`
    /// fields and the `dynamic_entries` overlay) is then infallible *unless*
    /// the model's state moved out from under this call between the initial
    /// check and the insert (a concurrent load starting) — that race is
    /// caught and reported rather than silently applied or silently
    /// dropped; see the state re-check right before the placement-path
    /// insert, below.
    ///
    /// Field classification (see [`config::ModelEntryPatch::apply_to`] for
    /// the exact per-field rule):
    /// - `device`/`tier_preference` are placement-affecting — rejected while
    ///   the model is `Ready` (409: evict first), applied via a full record
    ///   rebuild ([`build_not_loaded_record`](Self::build_not_loaded_record))
    ///   while `NotLoaded`.
    /// - `vram_gb`/`kv_cache_gb`/`max_concurrent_streams`/`max_prompt_len`/
    ///   `speculative` persist immediately but only change engine behaviour
    ///   at the model's *next* load (`effective_on_next_load` in the report).
    /// - Everything else (`pinned`/`priority`/`evictable`/`load`/
    ///   `reasoning_parser`/`capabilities`/`eviction_grace_secs`/precision-
    ///   family fields) is live the moment this call returns, for a `Ready`
    ///   model included — each is read fresh at decision time (eviction,
    ///   on-demand-load check, capability check, chat's reasoning-parser
    ///   scan), never cached at load time. See [`effective_entry`]'s doc
    ///   comment for why that requires the dynamic overlay, not
    ///   `self.config` (immutable after startup), to be the one this writes.
    ///
    /// `kind` is immutable via PATCH (400) — it changes which engine factory
    /// builds the model, which needs a fresh registration, not a field tweak.
    ///
    /// Holds `config_mutation_lock` for the whole call (read current entry →
    /// merge → validate → persist → live-apply) — nothing here `.await`s, so
    /// two concurrent `PATCH`es on the same model can't each merge against a
    /// stale pre-lock read and silently discard the other's change.
    ///
    /// # Errors
    /// - Model not registered (404).
    /// - `patch.kind.is_some()` (400 — immutable).
    /// - `patch.is_empty()` (400 — no fields supplied).
    /// - Model is `Loading`/`Evicting` (409).
    /// - `device`/`tier_preference` supplied while `Ready` (409).
    /// - The merged entry fails [`config::validate_model_entry`], or an
    ///   explicit `device` is empty/not enumerated (400).
    /// - `config_path` not set (503 — nothing to persist to).
    /// - Config write/rename failure (rare — disk I/O).
    #[allow(clippy::too_many_lines)]
    pub fn patch_model(
        &self,
        model_id: &str,
        patch: &config::ModelEntryPatch,
    ) -> anyhow::Result<PatchModelReport> {
        anyhow::ensure!(
            patch.kind.is_none(),
            "kind is immutable via PATCH — deregister and POST /v1/admin/models/add \
             to re-register '{model_id}' under a different kind"
        );
        anyhow::ensure!(!patch.is_empty(), "no fields supplied");

        let Some(ref config_path) = self.config_path else {
            anyhow::bail!(
                "config_path not set — PATCH is unavailable \
                 (server was started without a durable config file)"
            );
        };

        // Held for the whole function: no `.await` below this point.
        let _guard = self
            .config_mutation_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let state = self
            .read_models("patch_model")
            .get(model_id)
            .map(|r| r.state)
            .ok_or_else(|| ModelError::NotFound(model_id.to_owned()))?;
        match state {
            ModelState::Loading => return Err(ModelError::Loading.into()),
            ModelState::Evicting => return Err(ModelError::Evicting.into()),
            ModelState::Ready if patch.touches_placement() => {
                anyhow::bail!(
                    "'{model_id}' is Ready — device/tier_preference would disagree with \
                     the resident engine's actual placement; evict it first"
                );
            }
            ModelState::Ready | ModelState::NotLoaded => {}
        }

        let current = self
            .effective_entry(model_id)
            .ok_or_else(|| ModelError::NotFound(model_id.to_owned()))?;
        let (merged, outcome) = patch.apply_to(&current);
        config::validate_model_entry(model_id, &merged)?;
        if let Some(ref dev) = merged.policy.device {
            anyhow::ensure!(!dev.is_empty(), "device must be a non-empty string");
            if !self.inventory.devices().is_empty() {
                let available: Vec<&str> = self
                    .inventory
                    .devices()
                    .iter()
                    .map(|d| d.name.as_str())
                    .collect();
                anyhow::ensure!(
                    self.inventory.get(dev).is_some(),
                    "device '{dev}' is not an available OpenVINO device \
                     (enumerated: {available:?})"
                );
            }
        }

        // Build the placement-path record — which validates `tier_preference`
        // labels and re-runs the speculative pairing check, neither of which
        // `validate_model_entry` covers — BEFORE persisting, not after. A
        // rejection here must leave `config.json` untouched; getting this
        // ordering backwards once meant a bad `tier_preference` label wrote
        // successfully and only failed on `build_not_loaded_record`,
        // producing a `config.json` that fails validation on the very next
        // boot even though this call itself returned an error.
        let new_record = if patch.touches_placement() {
            Some(self.build_not_loaded_record(model_id, &merged)?)
        } else {
            None
        };

        Self::write_config_json(config_path, |value| {
            value["models"][model_id] = serde_json::to_value(&merged)?;
            Ok(())
        })?;

        if let Some(record) = new_record {
            // Re-check under the same write-lock hold as the insert: nothing
            // stops a concurrent `POST .../load` (or an on-demand load) from
            // moving this model NotLoaded -> Loading during the validation/
            // persist window above (state was last confirmed NotLoaded
            // before that, not atomically with this insert). Inserting a
            // fresh NotLoaded record over an in-flight Loading one would
            // orphan that load: its completion writes into a record nobody
            // is coordinating with, and `abort_load` wouldn't find the
            // Loading state it expects if the load fails.
            let mut models = self.write_models("patch_model");
            match models.get(model_id).map(|r| r.state) {
                Some(ModelState::NotLoaded) => {
                    models.insert(model_id.to_owned(), record);
                }
                other => {
                    drop(models);
                    anyhow::bail!(
                        "'{model_id}' stopped being NotLoaded while this PATCH was in flight \
                         (now {other:?}) — persisted to config.json for next load, but the live \
                         record was not replaced; retry once the concurrent operation finishes"
                    );
                }
            }
        } else if let Some(r) = self.write_models("patch_model").get_mut(model_id) {
            r.vram_gb = merged.vram_gb;
            r.pinned = merged.policy.pinned;
            r.priority = merged.policy.priority;
            r.evictable = merged.policy.evictable;
            r.reasoning_parser = merged.policy.reasoning_parser;
        }
        self.dynamic_entries
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(model_id.to_owned(), merged.clone());

        // Factory hints re-registered so the *next* load picks up the new
        // values — mirrors `load_model_with_overrides`'s own refresh-on-load
        // choke point, done here too so a hint isn't stuck at its
        // registration-time value until some unrelated load happens to touch it.
        if let Some(cap) = merged.policy.max_concurrent_streams {
            self.factory.register_concurrent_streams_hint(model_id, cap);
        }
        if let Some(max_prompt_len) = merged.policy.max_prompt_len {
            self.factory
                .register_max_prompt_len_hint(model_id, max_prompt_len);
        }
        self.sync_image_provenance_hint(model_id, Some(&merged));

        tracing::info!(
            model_id,
            applied_live = ?outcome.applied_live,
            effective_on_next_load = ?outcome.effective_on_next_load,
            "model patched"
        );

        Ok(PatchModelReport {
            applied_live: outcome.applied_live,
            effective_on_next_load: outcome.effective_on_next_load,
            unchanged: outcome.unchanged,
            entry: merged,
        })
    }

    /// Write a new model entry into the JSON config file atomically.
    ///
    /// Reads the file as `serde_json::Value`, inserts under `models[model_id]`,
    /// writes to a `.tmp` sibling, then renames over the original (atomic on
    /// Linux same-filesystem). Serializes the *whole* `entry` — `vram_gb`,
    /// `kind`, and every `ModelPolicy` field the caller set — so a runtime
    /// `add_model` registration is durable across restarts exactly as
    /// configured, not just its `vram_gb`/`kind`/`device` subset.
    fn persist_model_entry(
        &self,
        model_id: &str,
        entry: &config::ModelEntry,
    ) -> anyhow::Result<()> {
        let Some(ref config_path) = self.config_path else {
            tracing::warn!(
                model_id,
                "config_path not set — skipping config persistence for add_model"
            );
            return Ok(());
        };
        let _guard = self
            .config_mutation_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entry = entry.clone();
        Self::write_config_json(config_path, |value| {
            value["models"][model_id] = serde_json::to_value(&entry)?;
            Ok(())
        })?;

        tracing::info!(
            model_id,
            path = %config_path.display(),
            "persisted new model entry to config"
        );
        Ok(())
    }

    /// Remove `model_id`'s entry (and any `preload` mention) from the JSON
    /// config file atomically. Mirrors [`persist_model_entry`](Self::persist_model_entry) —
    /// same read/rewrite/atomic-rename shape, in reverse. A no-op (not an
    /// error) if `config_path` is unset or the model wasn't in the file (e.g.
    /// it was only ever a same-process `add_model` addition that predates the
    /// last persist, or the file was hand-edited already).
    ///
    /// Does **not** take `config_mutation_lock` itself — its one caller,
    /// [`deregister_model`](Self::deregister_model), holds it across this
    /// call *and* the registry/overlay removals immediately before it, so
    /// the whole "remove from everywhere" sequence is one critical section.
    /// Without that, a `patch_model` racing a `deregister_model` for the
    /// same id could re-insert the entry into `dynamic_entries`/`config.json`
    /// after deregister removed it, resurrecting a model deregister was
    /// supposed to erase.
    fn persist_model_removal(&self, model_id: &str) -> anyhow::Result<()> {
        let Some(ref config_path) = self.config_path else {
            tracing::warn!(
                model_id,
                "config_path not set — skipping config persistence for deregister"
            );
            return Ok(());
        };

        let mut removed_from_models = false;
        let mut removed_from_preload = false;
        let wrote = Self::write_config_json_if_changed(config_path, |value| {
            removed_from_models = value
                .get_mut("models")
                .and_then(serde_json::Value::as_object_mut)
                .is_some_and(|models| models.remove(model_id).is_some());
            removed_from_preload = value
                .get_mut("preload")
                .and_then(serde_json::Value::as_array_mut)
                .is_some_and(|preload| {
                    let before = preload.len();
                    preload.retain(|v| v.as_str() != Some(model_id));
                    preload.len() != before
                });
            Ok(removed_from_models || removed_from_preload)
        })?;

        if wrote {
            tracing::info!(
                model_id,
                removed_from_preload,
                path = %config_path.display(),
                "removed model entry from config"
            );
        }
        Ok(())
    }

    /// Read `config_path` as a [`serde_json::Value`], apply `mutate` in
    /// place, validate the *whole* resulting file as a [`config::Config`]
    /// (not just the touched entry — a per-model change can still produce an
    /// invalid file, e.g. a `patch_model` clearing `vram_gb` on a model with
    /// `speculative` set), then write it via a temp-file-plus-rename (atomic
    /// on the same filesystem). Callers must hold `config_mutation_lock` —
    /// this function does not take it itself, so `patch_model`/`set_preload`
    /// can wrap a whole read-merge-validate-write sequence in one critical
    /// section without deadlocking on a second acquire.
    fn write_config_json(
        config_path: &std::path::Path,
        mutate: impl FnOnce(&mut serde_json::Value) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        Self::write_config_json_if_changed(config_path, |value| {
            mutate(value)?;
            Ok(true)
        })
        .map(|_| ())
    }

    /// Same as [`write_config_json`](Self::write_config_json), but `mutate`
    /// reports whether anything actually changed (`Ok(false)` skips the
    /// validate-and-write entirely — [`persist_model_removal`](Self::persist_model_removal)'s
    /// "nothing to remove" case shouldn't pay a whole-file re-validate for a
    /// no-op). Returns whether it wrote.
    fn write_config_json_if_changed(
        config_path: &std::path::Path,
        mutate: impl FnOnce(&mut serde_json::Value) -> anyhow::Result<bool>,
    ) -> anyhow::Result<bool> {
        let raw = std::fs::read_to_string(config_path)?;
        let mut value: serde_json::Value = serde_json::from_str(&raw)?;

        if !mutate(&mut value)? {
            return Ok(false);
        }

        let patched: config::Config = serde_json::from_value(value.clone())
            .context("patched config no longer deserializes as a valid Config")?;
        patched
            .validate()
            .context("patched config failed validation — nothing written")?;

        let tmp = config_path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(&value)?)?;
        std::fs::rename(&tmp, config_path)?;
        Ok(true)
    }

    /// Fully remove `model_id` from the live registry and, if it was durably
    /// persisted, from `config.json` — evicting it from VRAM first if it is
    /// currently `Ready`.
    ///
    /// Unlike [`evict_model`](Self::evict_model) (unload from VRAM, stay
    /// registered — the counterpart is `POST .../{id}/load`), this erases the
    /// model entirely: it will not reappear in `GET /v1/admin/models`, on the
    /// next restart, or after a [`reload_config`](Self::reload_config) call.
    /// The seal for the "stale config entry" class of bug this project has
    /// hit repeatedly — the operator's explicit way to say "this `model_id` is
    /// gone for good," rather than hand-editing `config.json` and restarting.
    ///
    /// Also accepts a `model_id` that was declared in `config.json` but never
    /// made it into the live registry at all — the entry a missing-directory
    /// (`FilesMissing`) guard skipped at startup or a prior reload. That is
    /// exactly the "typo'd/stale `model_id`" case this method exists to let
    /// an operator clean up, so it is not a 404 just because there was never
    /// anything to evict: this call checks the startup config snapshot, any
    /// dynamic `add_model` entries, and — as a last resort, since a manual
    /// file edit may postdate both — the config file itself, before giving up
    /// with `NotFound`.
    ///
    /// # Errors
    /// - `ModelError::NotFound` if `model_id` is unknown everywhere: not live,
    ///   not in the startup config snapshot, not a dynamic entry, and not in
    ///   the config file on disk.
    /// - `ModelError::Loading`/`Evicting` if a load or eviction is already in
    ///   flight for it — retry once that settles.
    /// - Whatever [`evict_model`](Self::evict_model) can fail with, if the
    ///   model is currently `Ready`.
    pub async fn deregister_model(&self, model_id: &str) -> Result<()> {
        let state = self
            .read_models("deregister_model")
            .get(model_id)
            .map(|r| r.state);
        match state {
            Some(ModelState::Loading) => return Err(ModelError::Loading.into()),
            Some(ModelState::Evicting) => return Err(ModelError::Evicting.into()),
            Some(ModelState::Ready) => self.evict_model(model_id).await?,
            Some(ModelState::NotLoaded) => {}
            None => {
                let known = self.config.models.contains_key(model_id)
                    || self
                        .dynamic_entries
                        .read()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .contains_key(model_id)
                    || self
                        .config_path
                        .as_ref()
                        .and_then(|p| config::Config::load(p).ok())
                        .is_some_and(|c| c.models.contains_key(model_id));
                if !known {
                    return Err(ModelError::NotFound(model_id.to_owned()).into());
                }
            }
        }
        // Everything above this point may `.await` (the `evict_model` call);
        // nothing below does, so it's safe to hold `config_mutation_lock`
        // (a plain `std::sync::Mutex`) for the rest — one critical section
        // covering the registry removal, the overlay removal, and the file
        // removal, so a same-id `patch_model` can't interleave and
        // resurrect the entry (see `persist_model_removal`'s doc comment).
        let _guard = self
            .config_mutation_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.write_models("deregister_model").remove(model_id);
        self.dynamic_entries
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(model_id);
        self.persist_model_removal(model_id)?;
        Ok(())
    }

    /// Re-read `config.json` from disk and register any newly-declared model
    /// (validated via `EngineFactory::model_exists`, same guard as startup),
    /// refreshing the config-derived fields of any existing `NotLoaded` entry
    /// whose file values changed.
    ///
    /// Deliberately narrow in scope, to keep a reload's blast radius zero for
    /// anything already live:
    /// - Never touches a `Ready`/`Loading`/`Evicting` model — its engine keeps
    ///   running with whatever settings it loaded under, even if the file
    ///   changed underneath it. `PATCH /v1/admin/models/{id}` handles most
    ///   fields live (immediately for non-placement fields, next-load for
    ///   `kv_cache_gb`/etc.); `device`/`tier_preference` still need an evict
    ///   first (409 otherwise) since they'd disagree with the resident engine.
    /// - Never removes an entry the file dropped — that is
    ///   [`deregister_model`](Self::deregister_model)'s job, an explicit
    ///   per-model action, not an implicit side effect of a reload.
    /// - Never triggers a load, even for a model newly added to `preload` —
    ///   reload only changes what's *registered*, never what's *resident*.
    /// - Ignores every global knob (`total_vram_gb`, `bind_addr`, …) — only
    ///   `models` entries are diffed.
    /// - **Never touches API keys** — those live in a separate keys file with
    ///   their own independent reload, [`reload_keys_file`](Self::reload_keys_file)
    ///   / `POST /v1/admin/keys/reload`, on purpose: a key rotation is a
    ///   security event that deserves its own audit line, not burial inside a
    ///   model-catalog diff, and the two have different failure semantics (a
    ///   missing model directory is `skipped_missing_files`; a malformed keys
    ///   file aborts the whole reload, mutating nothing).
    ///
    /// A non-empty `api_keys`/`admin_api_keys` still inline in the loaded
    /// file (the pre-keys-file, deprecated shape) is a **hard error** here
    /// too, same as at boot (`startup::bootstrap`) — without this check, an
    /// operator who accidentally re-adds keys inline and reloads would get
    /// silent non-application (the fields still exist on `Config`, so nothing
    /// warns), the mirror image of the silent-non-revocation failure the
    /// keys-file split was built to avoid.
    ///
    /// # Errors
    /// - `config_path` not set (server started without a durable config
    ///   file).
    /// - The file can't be read, or fails `Config::load`'s validation.
    /// - The file has non-empty `api_keys`/`admin_api_keys` inline (migration
    ///   tripwire — move them to the keys file).
    /// - Placement fails for a newly-added entry (explicit `device` the
    ///   running `OpenVINO` didn't enumerate, or similar).
    pub fn reload_config(&self) -> anyhow::Result<ConfigReloadReport> {
        let Some(ref config_path) = self.config_path else {
            anyhow::bail!(
                "config_path not set — reload is unavailable \
                 (server was started without a durable config file)"
            );
        };
        // Doesn't write config.json itself, but does mutate `dynamic_entries`
        // (the same overlay `patch_model`/`add_model` write to) — held for
        // the whole call so a concurrent `patch_model` on the same model
        // can't have its change overwritten by this reading stale
        // pre-patch file contents, or vice versa. Plain `fn`, no `.await`.
        let _guard = self
            .config_mutation_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let file_config = config::Config::load(config_path)?;
        anyhow::ensure!(
            file_config.api_keys.is_empty() && file_config.admin_api_keys.is_empty(),
            "config.json has inline api_keys/admin_api_keys — these are deprecated and no \
             longer read for auth (moved to a separate keys file, never git-tracked). Move \
             them to {} and remove them from config.json before reloading",
            config::resolve_keys_file_path(config_path, file_config.keys_file.as_deref()).display()
        );

        let mut report = ConfigReloadReport {
            // Computed up front, against the boot snapshot this process has been
            // running on all along. Reload never swaps `self.config`, so this is
            // the operator's only runtime signal that a global edit did nothing.
            globals_not_applied: config::globals_not_applied(&self.config, &file_config),
            ..ConfigReloadReport::default()
        };

        for (model_id, entry) in &file_config.models {
            let existing_state = self
                .read_models("reload_config")
                .get(model_id)
                .map(|r| r.state);

            match existing_state {
                None => {
                    let model_dir = self.config.models_dir.join(model_id);
                    if !self.factory.model_exists(&model_dir) {
                        tracing::warn!(
                            model_id = model_id.as_str(),
                            expected_path = %model_dir.display(),
                            "config reload: model directory does not exist — skipping registration"
                        );
                        report.skipped_missing_files.push(model_id.clone());
                        continue;
                    }
                    let record = self.build_not_loaded_record(model_id, entry)?;
                    self.write_models("reload_config")
                        .insert(model_id.clone(), record);
                    // Mirror into the overlay too — `effective_entry` is the
                    // only thing `eviction_grace_for`/`model_capabilities`/
                    // `request_on_demand_load` read for a model also present
                    // in `config.models` (which `self.config` itself never
                    // reflects after startup); without this, this file's
                    // `eviction_grace_secs`/`capabilities`/`load` would never
                    // reach those sites for a model reload just registered.
                    self.dynamic_entries
                        .write()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .insert(model_id.clone(), entry.clone());
                    report.added.push(model_id.clone());
                }
                Some(ModelState::Ready | ModelState::Loading | ModelState::Evicting) => {
                    report.left_untouched.push(model_id.clone());
                }
                Some(ModelState::NotLoaded) => {
                    let differs = {
                        let guard = self.read_models("reload_config");
                        let Some(current) = guard.get(model_id) else {
                            continue; // evicted between the check above and here
                        };
                        let new_kind = entry
                            .kind
                            .as_deref()
                            .and_then(ModelKind::from_label)
                            .unwrap_or(ModelKind::TextGen);
                        (current.vram_gb - entry.vram_gb).abs() > f64::EPSILON
                            || new_kind != current.configured_kind
                            || current.pinned != entry.policy.pinned
                            || current.priority != entry.policy.priority
                            || current.evictable != entry.policy.evictable
                            || current.reasoning_parser != entry.policy.reasoning_parser
                            || entry
                                .policy
                                .device
                                .as_deref()
                                .is_some_and(|d| d != current.device)
                    };
                    if differs {
                        let record = self.build_not_loaded_record(model_id, entry)?;
                        self.write_models("reload_config")
                            .insert(model_id.clone(), record);
                        // Same overlay mirror as the `added` branch above.
                        self.dynamic_entries
                            .write()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .insert(model_id.clone(), entry.clone());
                        report.updated.push(model_id.clone());
                    } else {
                        report.unchanged += 1;
                    }
                }
            }
        }

        Ok(report)
    }

    /// Replace the persisted `preload` array in `config.json` —
    /// `POST /v1/admin/config/preload`'s implementation. Never touches the
    /// `models` map itself, and never triggers a load — like every config
    /// mutation here, it changes what's *registered to auto-load next boot*,
    /// not what's *resident right now* (mirrors `reload_config`'s own
    /// "changes what's registered, never what's resident" rule).
    ///
    /// Every id in the resulting list — explicit or `from_live`-snapshotted
    /// — must already have a `models` entry *in the file being written*, not
    /// just in the live registry: `add_model`'s persist can fail after its
    /// load already succeeded (documented on `add_model`'s own doc comment),
    /// so a live-`Ready` model can exist with no file stanza. Snapshotting
    /// such a model into `preload` would write a file that fails
    /// [`config::Config::validate`]'s own "every preload model must have a
    /// `models` entry" invariant on the very next boot — checked and
    /// rejected here instead (400, naming the offender), before anything is
    /// written.
    ///
    /// Holds `config_mutation_lock` for the whole call, same reasoning as
    /// [`patch_model`](Self::patch_model) — nothing here `.await`s.
    ///
    /// # Errors
    /// - `config_path` not set (503).
    /// - [`PreloadSource::Explicit`] contains a duplicate id (400).
    /// - Any resulting id has no `models` entry in the file (400, naming
    ///   every offender at once rather than one at a time).
    /// - Config write/rename failure (rare — disk I/O).
    pub fn set_preload(&self, source: PreloadSource) -> anyhow::Result<SetPreloadReport> {
        let Some(ref config_path) = self.config_path else {
            anyhow::bail!(
                "config_path not set — preload management is unavailable \
                 (server was started without a durable config file)"
            );
        };
        let _guard = self
            .config_mutation_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let mut skipped_not_ready = Vec::new();
        let new_preload: Vec<String> = match source {
            PreloadSource::Explicit(ids) => {
                let mut seen = std::collections::HashSet::with_capacity(ids.len());
                for id in &ids {
                    anyhow::ensure!(
                        seen.insert(id.clone()),
                        "duplicate model id '{id}' in preload list"
                    );
                }
                ids
            }
            PreloadSource::FromLive => {
                let mut ready = Vec::new();
                for (id, r) in self.read_models("set_preload:from_live").iter() {
                    if r.state == ModelState::Ready {
                        ready.push(id.clone());
                    } else {
                        skipped_not_ready.push(id.clone());
                    }
                }
                ready.sort_unstable();
                skipped_not_ready.sort_unstable();
                ready
            }
        };

        let mut previous = Vec::new();
        Self::write_config_json(config_path, |value| {
            previous = value
                .get("preload")
                .and_then(serde_json::Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(str::to_owned))
                        .collect()
                })
                .unwrap_or_default();

            let known: std::collections::HashSet<&str> = value
                .get("models")
                .and_then(serde_json::Value::as_object)
                .map(|m| m.keys().map(String::as_str).collect())
                .unwrap_or_default();
            let unknown: Vec<&String> = new_preload
                .iter()
                .filter(|id| !known.contains(id.as_str()))
                .collect();
            anyhow::ensure!(
                unknown.is_empty(),
                "preload references model id(s) with no `models` entry: {unknown:?}"
            );

            value["preload"] = serde_json::to_value(&new_preload)?;
            Ok(())
        })?;

        let prev_set: std::collections::HashSet<&String> = previous.iter().collect();
        let new_set: std::collections::HashSet<&String> = new_preload.iter().collect();
        let added = new_preload
            .iter()
            .filter(|id| !prev_set.contains(id))
            .cloned()
            .collect();
        let removed = previous
            .iter()
            .filter(|id| !new_set.contains(id))
            .cloned()
            .collect();

        tracing::info!(preload = ?new_preload, "preload list updated");

        Ok(SetPreloadReport {
            preload: new_preload,
            previous,
            added,
            removed,
            skipped_not_ready,
        })
    }

    /// Re-read the keys file from disk and hot-swap `api_keys`/`admin_api_keys`
    /// — independent of [`reload_config`](Self::reload_config) on purpose
    /// (see its doc comment). `POST /v1/admin/keys/reload` is the only
    /// caller.
    ///
    /// **The hard invariant this whole mechanism rests on** (write it down
    /// once, here): key material only ever enters the live `AuthConfig` via
    /// a filesystem write to the keys file — this method, and every admin
    /// endpoint, can only ever say "re-read what's on disk," never accept
    /// key values in a request body. That's what keeps a stolen admin key
    /// from becoming self-perpetuating (mint a second key via the API,
    /// survive the operator rotating the first) — see the project's internal engineering log.
    /// No admin endpoint may ever gain a "write this key" body parameter;
    /// if one is ever tempting, that's a sign it belongs in the keys file
    /// instead.
    ///
    /// All-or-nothing: a missing or malformed file aborts with the live
    /// `AuthConfig` completely untouched — unlike `startup::bootstrap`'s
    /// boot-time posture (missing file → open, with a warning), a reload
    /// that can't read the file it was told to re-read is far more likely an
    /// operational mistake (wrong path, accidental delete) than "auth was
    /// never configured," so it errors loud instead of silently going open.
    ///
    /// Privilege-downgrade guard (unchanged from the pre-split design): if
    /// the file's new `admin_api_keys` is empty while the live value is
    /// non-empty, the admin list is left alone
    /// (`KeysReloadReport::admin_downgrade_refused: true`,
    /// `admin_keys_updated: false`, a warning logged) — otherwise a reload
    /// could silently reopen every `/v1/admin/*` route to any
    /// inference-scoped key, since an empty `admin_api_keys` falls back to
    /// the (usually broader) `api_keys` gate. `api_keys` itself has no
    /// equivalent guard: emptying it is a deliberate, symmetric "go open"
    /// action an operator can already take by editing the file. No key
    /// value is ever logged or returned, only counts.
    ///
    /// # Errors
    /// - No keys-file path wired (`with_keys_file_path` never called — server
    ///   started without a durable config file, or in a test build).
    /// - The file doesn't exist, can't be read, or fails to parse (
    ///   `AuthConfig`'s `deny_unknown_fields` included) — nothing is mutated.
    pub fn reload_keys_file(&self) -> anyhow::Result<KeysReloadReport> {
        let Some(auth) = &self.auth else {
            anyhow::bail!(
                "no auth handle wired (AppState::with_auth/ModelManager::with_auth never \
                 called) — reload is unavailable"
            );
        };
        let Some(ref keys_path) = self.keys_file_path else {
            anyhow::bail!(
                "no keys_file_path wired (server was started without a durable config file) \
                 — reload is unavailable"
            );
        };
        let file_keys = crate::AuthConfig::load_from_file(keys_path)?;
        let current = auth.load();

        let keys_changed = file_keys.api_keys != current.api_keys;

        let admin_downgrade_refused =
            file_keys.admin_api_keys.is_empty() && !current.admin_api_keys.is_empty();
        let new_admin_keys = if admin_downgrade_refused {
            current.admin_api_keys.clone()
        } else {
            file_keys.admin_api_keys.clone()
        };
        let admin_keys_changed =
            !admin_downgrade_refused && file_keys.admin_api_keys != current.admin_api_keys;

        if admin_downgrade_refused {
            tracing::warn!(
                live_admin_key_count = current.admin_api_keys.len(),
                "keys reload: file's admin_api_keys is now empty while a non-empty admin \
                 scope is live — refusing to downgrade (would silently reopen /v1/admin/* to \
                 any inference key); live admin keys left unchanged"
            );
        }

        let api_key_count = file_keys.api_keys.len();
        let admin_key_count = new_admin_keys.len();
        if keys_changed || admin_keys_changed {
            auth.store(Arc::new(crate::AuthConfig {
                api_keys: file_keys.api_keys,
                admin_api_keys: new_admin_keys,
            }));
            tracing::info!(
                keys_updated = keys_changed,
                admin_keys_updated = admin_keys_changed,
                api_key_count,
                admin_key_count,
                "keys reload: auth keys rotated"
            );
        }
        Ok(KeysReloadReport {
            keys_updated: keys_changed,
            admin_keys_updated: admin_keys_changed,
            admin_downgrade_refused,
            api_key_count,
            admin_key_count,
        })
    }

    /// Audit `config.json` on disk against the live registry — the
    /// stale/typo'd-`model_id` class of bug this project has repeatedly hit
    /// by hand, surfaced over the API instead of only in startup logs.
    ///
    /// Read-only: never mutates config or the registry (unlike
    /// [`reload_config`](Self::reload_config)). Re-reads the file fresh each
    /// call, so it also catches drift the live registry hasn't picked up yet
    /// (`pending_reload`).
    ///
    /// # Errors
    /// - `config_path` not set.
    /// - The file can't be read, or fails `Config::load`'s validation.
    pub fn audit_config(&self) -> anyhow::Result<ConfigAuditReport> {
        let Some(ref config_path) = self.config_path else {
            anyhow::bail!(
                "config_path not set — audit is unavailable \
                 (server was started without a durable config file)"
            );
        };
        let file_config = config::Config::load(config_path)?;
        let preload: std::collections::HashSet<&str> =
            file_config.preload.iter().map(String::as_str).collect();
        let live = self.read_models("audit_config");

        let mut report = ConfigAuditReport {
            total_declared: file_config.models.len(),
            problems: Vec::new(),
        };
        for model_id in file_config.models.keys() {
            let model_dir = self.config.models_dir.join(model_id);
            let files_missing = !self.factory.model_exists(&model_dir);
            let pending_reload = !files_missing && !live.contains_key(model_id);
            if files_missing || pending_reload {
                report.problems.push(ConfigAuditEntry {
                    model_id: model_id.clone(),
                    expected_path: model_dir.display().to_string(),
                    files_missing,
                    in_preload: preload.contains(model_id.as_str()),
                    pending_reload,
                });
            }
        }
        Ok(report)
    }

    /// Creates a `ModelManager` from a config and a (possibly mock) engine factory.
    ///
    /// Registers all models listed in `config.vram_gb` as `NotLoaded`, then
    /// loads each model in `config.preload` sequentially. Every preload entry
    /// must itself appear in `vram_gb` (T5.5/#34) — a preload model with no VRAM
    /// estimate is a hard error, not a silent uncounted 0.0-GB load.
    ///
    /// `pub(crate)` so tests inside this crate can inject a `MockEngineFactory`.
    /// External callers (e.g. `main.rs`) use [`new_production`](Self::new_production).
    ///
    /// This overload passes an **empty** device inventory: with no devices
    /// discovered, the inference domain falls back to `config.device` and the
    /// tracker is single-domain — exactly the pre-inventory behaviour, which is
    /// what GPU-free unit tests want. Production resolves a real inventory via
    /// [`new_with_inventory`](Self::new_with_inventory).
    ///
    /// # Errors
    /// Fails only on a config/registration problem (e.g. a `preload` entry
    /// with no `models` entry, an unresolvable device or tier). A `preload`
    /// model that fails to *load* (engine/factory failure at runtime — the
    /// 2026-07-14 VRAM-overcommit incident, the project's internal engineering log)
    /// does **not** fail startup: see [`new_with_inventory`](Self::new_with_inventory).
    ///
    /// Test-only: production always has an inventory and uses
    /// [`new_with_inventory`](Self::new_with_inventory) (via `new_production`).
    #[cfg(test)]
    pub(crate) async fn new(config: Config, factory: Arc<dyn EngineFactory>) -> Result<Self> {
        Self::new_with_inventory(config, factory, Arc::new(DeviceInventory::default())).await
    }

    /// As [`new`](Self::new), but resolves the inference memory domain (and, in
    /// Phase B-2, per-domain budgets) from a real device `inventory`.
    ///
    /// A `preload` model that fails to load (e.g. a `CL_OUT_OF_RESOURCES`-class
    /// engine failure) degrades gracefully instead of aborting the process —
    /// mirrors [`request_on_demand_load`](Self::request_on_demand_load)'s
    /// handling of the identical error path: logged `WARN`, the model is left
    /// `NotLoaded` (state reset already happens inside `load_model`'s own error
    /// path), and the rest of `preload` still runs. Before this, an eager
    /// preload failure was fatal at startup — one poisoned GPU turned into a
    /// permanent crash-loop-to-giveup under `--supervise`, since every respawn
    /// hit the same preload failure (2026-07-14 incident, see
    /// the project's internal engineering log).
    ///
    /// # Errors
    /// Fails only on a config/registration problem — see [`new`](Self::new).
    #[allow(clippy::too_many_lines)]
    pub(crate) async fn new_with_inventory(
        config: Config,
        factory: Arc<dyn EngineFactory>,
        inventory: Arc<DeviceInventory>,
    ) -> Result<Self> {
        // Per-domain budgets are needed *during* registration now (T5/F4): the
        // memory domain + its total drive fit-aware placement. They depend only
        // on config + inventory, so compute them before the loop (the tracker
        // reuses `budgets` after it).
        let inference_domain = resolve_inference_domain(&config, &inventory);
        let budgets = resolve_domain_budgets(&config, &inventory, &inference_domain);
        let domain_total: HashMap<String, f64> = budgets.iter().cloned().collect();
        // Running per-domain reservation by *immovable* (pinned / non-evictable)
        // models — later models route around a domain these have filled.
        let mut projected: HashMap<String, f64> = HashMap::new();

        let mut records: HashMap<String, ModelRecord> = HashMap::new();

        // T5/F4: deterministic, immovable-first registration order. `config.models`
        // is a HashMap (unspecified iteration order); once placement reads per-domain
        // occupancy the order is significant, so fix it — place pinned / non-evictable
        // models first (they hold the permanent reservations later models route
        // around), then by id for stability.
        let is_immovable = |e: &config::ModelEntry| e.policy.pinned || !e.policy.evictable;
        let mut entries: Vec<(&String, &config::ModelEntry)> = config.models.iter().collect();
        entries.sort_by(|a, b| {
            is_immovable(b.1)
                .cmp(&is_immovable(a.1))
                .then_with(|| a.0.cmp(b.0))
        });

        for (model_id, entry) in entries {
            // Skip a config entry whose directory doesn't exist under
            // `models_dir` — a stale/typo'd model_id (the exact class of bug
            // that used to silently register a phantom, unloadable entry that
            // showed up in `/v1/admin/models` and, worse, could trigger an LRU
            // eviction on load attempt before failing). Checked here, before
            // placement/reservation, so a phantom entry never occupies a domain
            // reservation slot either. `factory.model_exists` defaults to `true`
            // for test mocks (which never touch the real filesystem); only
            // `OvEngineFactory` performs the real check.
            let model_dir = config.models_dir.join(model_id);
            if !factory.model_exists(&model_dir) {
                tracing::warn!(
                    model_id = model_id.as_str(),
                    expected_path = %model_dir.display(),
                    "config registers a model whose directory does not exist — skipping registration (fix the config entry or provide the missing files)"
                );
                continue;
            }
            let vram_estimate = entry.vram_gb;
            let policy = &entry.policy;
            // No explicit `kind`: fall back to the same file-sniffing
            // `detect_kind` uses at real load time, instead of leaving this
            // `None` — an unhinted `None` here previously lost the kind-aware
            // tier preference for `resolve_model_device` below, and defaulted
            // `configured_kind` to a hardcoded `TextGen` even for a real VLM
            // (see `dev/autotest/20260717_qwen3_4b_int8_sigsegv.md` Finding 3).
            let kind = entry
                .kind
                .as_deref()
                .and_then(ModelKind::from_label)
                .unwrap_or_else(|| factory.detect_kind(&model_dir));
            // Startup-preloaded models with an explicit `kind` already reach
            // the factory via `new_production`'s bulk `model_kinds` hint
            // population; the auto-detected case doesn't, so teach it here —
            // harmless (load() would independently re-derive the identical
            // answer) and keeps the factory and this record in agreement.
            factory.register_kind_hint(model_id, kind);
            let device = resolve_model_device(
                model_id,
                vram_estimate,
                Some(kind),
                Some(policy),
                &config,
                &inventory,
                &domain_total,
                &projected,
            )?;
            let domain = resolve_device_domain(&device, &inventory);
            // T5/F4: reserve an immovable model's footprint in its domain so the
            // models placed after it route around the space it permanently holds.
            // Evictable models do NOT reserve — eviction reclaims them at load, so
            // they still stack on the preferred device and are sorted at runtime.
            if is_immovable(entry) {
                *projected.entry(domain.clone()).or_insert(0.0) += vram_estimate;
            }
            records.insert(
                model_id.clone(),
                ModelRecord {
                    state: ModelState::NotLoaded,
                    handle: None,
                    thread: None,
                    // Unloaded models start with current time — the LRU search
                    // filters by state==Ready, so this value is never used for
                    // eviction ordering until after a successful load_model().
                    last_used: Instant::now(),
                    vram_gb: vram_estimate,
                    kv_cache_gb: 0.0,
                    max_prompt_tokens: 0,
                    pool_capacity_tokens: 0,
                    max_concurrent_streams: 0,
                    template: Arc::from(""),
                    family: ModelFamily::Default,
                    eos_token: Arc::from(""),
                    bos_token: Arc::from(""),
                    configured_kind: kind,
                    last_kind: None,
                    last_load_duration_secs: None,
                    pinned: policy.pinned,
                    priority: policy.priority,
                    evictable: policy.evictable,
                    device,
                    domain,
                    reasoning_parser: policy.reasoning_parser,
                    configured_max_prompt_len: policy.max_prompt_len,
                    generation_defaults: template::GenerationDefaults::default(),
                    native_context_limit: template::read_native_context_limit(&model_dir),
                },
            );
        }

        // T5.5 (#34): every preload model must already be registered from
        // `config.models`. In production `Config::validate` guarantees this;
        // erroring here too closes the footgun for callers that bypass validation
        // (a 0.0-GB model loads uncounted, never evictable by weight → OOM risk).
        for model_id in &config.preload {
            anyhow::ensure!(
                records.contains_key(model_id),
                "preload model '{model_id}' is not registered — either it's missing from \
                 config.models, or its directory was not found under models_dir (see the \
                 startup warning above)"
            );
        }

        // The memory domain for each model came from the inventory above: a
        // discrete GPU's own name (private VRAM pool), or the shared "system"
        // domain for an iGPU/CPU/NPU. With an empty inventory (unit tests) it
        // falls back to `config.device` verbatim. The tracker is sized per domain
        // from the `budgets` computed before the loop — the inference domain keeps
        // the legacy `total_vram_gb`, so a single-dGPU box (and a CPU
        // `0.0`-disabled box) is byte-for-byte unchanged; other domains (system
        // RAM, secondary GPUs) get their own budgets.
        let mm = Self {
            vram: Arc::new(RwLock::new(MemoryTracker::new(budgets))),
            inference_domain,
            factory,
            next_request_id: Arc::new(AtomicU64::new(0)),
            models: Arc::new(RwLock::new(records)),
            gpu_poisoned: AtomicBool::new(false),
            load_lock: tokio::sync::Mutex::new(()),
            config,
            inventory: Arc::clone(&inventory),
            dynamic_entries: RwLock::new(HashMap::new()),
            config_path: None,
            voice_pin: None,
            auth: None,
            keys_file_path: None,
            kv_wedge_recovery_inflight: Mutex::new(HashSet::new()),
            config_mutation_lock: Mutex::new(()),
            kv_pressure: Mutex::new(HashMap::new()),
        };

        mm.load_preload_models().await;

        Ok(mm)
    }

    /// Loads every model in `config.preload`, sequentially.
    ///
    /// A load failure does not abort startup — mirrors on-demand's handling
    /// of the identical error path (see
    /// [`request_on_demand_load`](Self::request_on_demand_load)): log a WARN
    /// and leave the model `NotLoaded` (already done inside `load_model`'s
    /// own error path) rather than crashing the whole process. Keeps going so
    /// a single bad/exhausted model doesn't take every other preload model
    /// (and the server itself) down with it — see the 2026-07-14 incident,
    /// the project's internal engineering log.
    async fn load_preload_models(&self) {
        for model_id in self.config.preload.clone() {
            if let Err(e) = self.load_model(&model_id).await {
                tracing::warn!(
                    model = %model_id,
                    error = %e,
                    "preload failed — model left NotLoaded, server continues"
                );
            }
        }
    }

    // ---- Public API ---------------------------------------------------

    /// Returns a cloneable engine handle for `model_id` if it is Ready.
    ///
    /// Also updates `last_used` for LRU tracking.
    ///
    /// # Errors
    /// - [`ModelError::NotFound`] — model ID is not registered in config
    /// - [`ModelError::NotLoaded`] — model exists but is not in VRAM
    /// - [`ModelError::Loading`] — model is being loaded right now
    /// - [`ModelError::Evicting`] — model is being evicted
    pub fn get_handle(&self, model_id: &str) -> std::result::Result<EngineHandle, ModelError> {
        self.get_text_context(model_id).map(|(handle, _)| handle)
    }

    /// Like [`get_handle`](Self::get_handle) but also returns the model's L0
    /// prompt-length limit (`max_prompt_tokens`; `0` = gate disabled) — for
    /// text endpoints that gate raw prompts before submission
    /// (`/v1/completions`, T2.1 gate parity with the chat path).
    ///
    /// # Errors
    /// Same as [`get_handle`](Self::get_handle).
    pub fn get_text_context(
        &self,
        model_id: &str,
    ) -> std::result::Result<(EngineHandle, usize), ModelError> {
        let ctx = self.get_chat_context(model_id)?;
        // R1: the enum is now multi-variant, so this destructure is refutable.
        // The text-only callers (`/tokenize`, `/v1/completions`) reject non-text
        // kinds explicitly rather than silently mis-routing a VLM here. A future
        // kind (R3) makes this match non-exhaustive → the compiler flags it.
        let kind = ctx.handle.kind();
        match ctx.handle {
            EngineHandleKind::TextGen(handle) => Ok((handle, ctx.max_prompt_tokens)),
            // NpuTextGen uses LLMPipeline which has no standalone tokenize call
            // in the current bridge — reject for /tokenize and /v1/completions.
            EngineHandleKind::NpuTextGen(_)
            | EngineHandleKind::Vision(_)
            | EngineHandleKind::Embedding(_)
            | EngineHandleKind::Stt(_)
            | EngineHandleKind::Tts(_)
            | EngineHandleKind::ImageGen(_)
            | EngineHandleKind::Reranking(_) => Err(ModelError::WrongKind(format!(
                "model '{model_id}' is a {kind:?} model — this endpoint serves text-generation models only"
            ))),
        }
    }

    /// The configured `/v1/completions` prompt-array fan-out cap (T2.1).
    #[must_use]
    pub fn max_prompt_array(&self) -> usize {
        self.config.max_prompt_array
    }

    /// The configured server-wide `max_tokens` ceiling (T7.1/T2.2); `0` = uncapped.
    #[must_use]
    pub fn max_tokens_cap(&self) -> usize {
        self.config.max_tokens_cap
    }

    /// Default embedding model to seed into new realtime sessions, or `None`
    /// when no server-side default is configured.
    #[must_use]
    pub fn default_embed_model(&self) -> Option<&str> {
        self.config.default_embed_model.as_deref()
    }

    /// Returns the [`EmbeddingHandle`](crate::embed_engine::EmbeddingHandle) for
    /// a Ready embedding model, updating `last_used` for LRU tracking.
    ///
    /// The `/v1/embeddings` analogue of [`get_handle`](Self::get_handle): it
    /// triggers a lazy load (via the handler) and rejects every non-embedding
    /// kind with [`ModelError::WrongKind`] (400) — a text or vision model is the
    /// wrong engine for embeddings.
    ///
    /// # Errors
    /// Same as [`get_handle`](Self::get_handle), plus `WrongKind` for non-embedding models.
    pub fn get_embedding_handle(
        &self,
        model_id: &str,
    ) -> std::result::Result<crate::embed_engine::EmbeddingHandle, ModelError> {
        let ctx = self.get_chat_context(model_id)?;
        let kind = ctx.handle.kind();
        match ctx.handle {
            EngineHandleKind::Embedding(handle) => Ok(handle),
            EngineHandleKind::TextGen(_)
            | EngineHandleKind::NpuTextGen(_)
            | EngineHandleKind::Vision(_)
            | EngineHandleKind::Stt(_)
            | EngineHandleKind::Tts(_)
            | EngineHandleKind::ImageGen(_)
            | EngineHandleKind::Reranking(_) => Err(ModelError::WrongKind(format!(
                "model '{model_id}' is a {kind:?} model — /v1/embeddings serves embedding models only"
            ))),
        }
    }

    /// Returns the [`SttHandle`](crate::pipelines::stt::SttHandle) for a Ready
    /// speech-to-text model, updating `last_used` for LRU tracking.
    ///
    /// The `/v1/audio/transcriptions` analogue of
    /// [`get_embedding_handle`](Self::get_embedding_handle): it rejects every
    /// non-STT kind with [`ModelError::WrongKind`] (400) — a text/vision/embedding
    /// model is the wrong engine for transcription.
    ///
    /// # Errors
    /// Same as [`get_handle`](Self::get_handle), plus `WrongKind` for non-STT models.
    pub fn get_stt_handle(
        &self,
        model_id: &str,
    ) -> std::result::Result<crate::pipelines::stt::SttHandle, ModelError> {
        let ctx = self.get_chat_context(model_id)?;
        let kind = ctx.handle.kind();
        match ctx.handle {
            EngineHandleKind::Stt(handle) => Ok(handle),
            EngineHandleKind::TextGen(_)
            | EngineHandleKind::NpuTextGen(_)
            | EngineHandleKind::Vision(_)
            | EngineHandleKind::Embedding(_)
            | EngineHandleKind::Tts(_)
            | EngineHandleKind::ImageGen(_)
            | EngineHandleKind::Reranking(_) => Err(ModelError::WrongKind(format!(
                "model '{model_id}' is a {kind:?} model — /v1/audio/transcriptions serves STT models only"
            ))),
        }
    }

    /// Returns the [`TtsHandle`](crate::pipelines::tts::TtsHandle) for a Ready
    /// text-to-speech model, updating `last_used` for LRU tracking.
    ///
    /// Rejects every non-TTS kind with [`ModelError::WrongKind`] (400).
    ///
    /// # Errors
    /// Same as [`get_handle`](Self::get_handle), plus `WrongKind` for non-TTS models.
    pub fn get_tts_handle(
        &self,
        model_id: &str,
    ) -> std::result::Result<crate::pipelines::tts::TtsHandle, ModelError> {
        let ctx = self.get_chat_context(model_id)?;
        let kind = ctx.handle.kind();
        match ctx.handle {
            EngineHandleKind::Tts(handle) => Ok(handle),
            EngineHandleKind::TextGen(_)
            | EngineHandleKind::NpuTextGen(_)
            | EngineHandleKind::Vision(_)
            | EngineHandleKind::Embedding(_)
            | EngineHandleKind::Stt(_)
            | EngineHandleKind::ImageGen(_)
            | EngineHandleKind::Reranking(_) => Err(ModelError::WrongKind(format!(
                "model '{model_id}' is a {kind:?} model — /v1/audio/speech serves TTS models only"
            ))),
        }
    }

    /// Returns the [`ImageHandle`](crate::pipelines::image::ImageHandle) for a
    /// Ready text-to-image model, updating `last_used` for LRU tracking.
    ///
    /// The `/v1/images/generations` analogue of
    /// [`get_stt_handle`](Self::get_stt_handle): it rejects every non-image kind
    /// with [`ModelError::WrongKind`] (400) — a text/vision/embedding/STT model is
    /// the wrong engine for image generation.
    ///
    /// # Errors
    /// Same as [`get_handle`](Self::get_handle), plus `WrongKind` for non-image models.
    pub fn get_image_handle(
        &self,
        model_id: &str,
    ) -> std::result::Result<crate::pipelines::image::ImageHandle, ModelError> {
        let ctx = self.get_chat_context(model_id)?;
        let kind = ctx.handle.kind();
        match ctx.handle {
            EngineHandleKind::ImageGen(handle) => Ok(handle),
            EngineHandleKind::TextGen(_)
            | EngineHandleKind::NpuTextGen(_)
            | EngineHandleKind::Vision(_)
            | EngineHandleKind::Embedding(_)
            | EngineHandleKind::Stt(_)
            | EngineHandleKind::Tts(_)
            | EngineHandleKind::Reranking(_) => Err(ModelError::WrongKind(format!(
                "model '{model_id}' is a {kind:?} model — /v1/images/generations serves image models only"
            ))),
        }
    }

    /// Returns the resolved [`DeviceInfo`] (silicon identity, capability tier,
    /// total memory) for a registered model's placement device.
    ///
    /// `PLAN_image_metadata_response.md` §1a: `ModelRecord.device` already holds
    /// the resolved `OpenVINO` device name (e.g. `"GPU.1"`); this just looks it
    /// up in the startup [`DeviceInventory`] snapshot. Returns `None` if the
    /// model is unknown or its device isn't in the inventory (should not happen
    /// for a model that has ever loaded — `resolve_model_device` validates
    /// against this same inventory).
    #[must_use]
    pub fn device_info_for(&self, model_id: &str) -> Option<DeviceInfo> {
        let guard = self.read_models("device_info_for");
        let record = guard.get(model_id)?;
        self.inventory.get(&record.device).cloned()
    }

    /// Returns the [`RerankingHandle`](crate::rerank_engine::RerankingHandle) for
    /// a Ready reranking model, updating `last_used` for LRU tracking.
    ///
    /// The `/v1/rerank` analogue of [`get_embedding_handle`](Self::get_embedding_handle):
    /// rejects every non-reranking kind with [`ModelError::WrongKind`] (400).
    ///
    /// # Errors
    /// Same as [`get_handle`](Self::get_handle), plus `WrongKind` for non-reranking models.
    pub fn get_reranking_handle(
        &self,
        model_id: &str,
    ) -> std::result::Result<crate::rerank_engine::RerankingHandle, ModelError> {
        let ctx = self.get_chat_context(model_id)?;
        let kind = ctx.handle.kind();
        match ctx.handle {
            EngineHandleKind::Reranking(handle) => Ok(handle),
            EngineHandleKind::TextGen(_)
            | EngineHandleKind::NpuTextGen(_)
            | EngineHandleKind::Vision(_)
            | EngineHandleKind::Embedding(_)
            | EngineHandleKind::Stt(_)
            | EngineHandleKind::Tts(_)
            | EngineHandleKind::ImageGen(_) => Err(ModelError::WrongKind(format!(
                "model '{model_id}' is a {kind:?} model — /v1/rerank serves reranking models only"
            ))),
        }
    }

    /// Returns the full [`ChatContext`] (handle + chat template + family) for a
    /// Ready model, updating `last_used` for LRU tracking.
    ///
    /// This is the chat handler's entry point: it needs the template to build
    /// the prompt and the family to parse any tool calls, in addition to the
    /// submit handle. [`get_handle`](Self::get_handle) is a thin wrapper that
    /// drops everything but the handle.
    ///
    /// # Errors
    /// Same as [`get_handle`](Self::get_handle): `NotFound` / `NotLoaded` /
    /// `Loading` / `Evicting`.
    pub fn get_chat_context(&self, model_id: &str) -> std::result::Result<ChatContext, ModelError> {
        let mut guard = self.write_models("get_chat_context");
        match guard.get_mut(model_id) {
            None => Err(ModelError::NotFound(model_id.to_owned())),
            Some(r) => match r.state {
                ModelState::Ready => {
                    r.last_used = Instant::now();
                    // INVARIANT: handle is always Some when state == Ready.
                    // Set before Ready in execute_load; taken before Evicting
                    // in begin_evict. If somehow None, report NotLoaded.
                    if let Some(h) = r.handle.clone() {
                        Ok(ChatContext {
                            handle: h,
                            template: Arc::clone(&r.template),
                            family: r.family,
                            eos_token: Arc::clone(&r.eos_token),
                            bos_token: Arc::clone(&r.bos_token),
                            max_prompt_tokens: r.max_prompt_tokens,
                            pool_capacity_tokens: r.pool_capacity_tokens,
                            max_concurrent_streams: r.max_concurrent_streams,
                            reasoning_parser: r.reasoning_parser,
                            configured_max_prompt_len: r.configured_max_prompt_len,
                            generation_defaults: r.generation_defaults,
                        })
                    } else {
                        Err(ModelError::NotLoaded)
                    }
                }
                ModelState::NotLoaded => Err(ModelError::NotLoaded),
                ModelState::Loading => Err(ModelError::Loading),
                ModelState::Evicting => Err(ModelError::Evicting),
            },
        }
    }

    /// Loads a model from disk and makes it available for inference.
    ///
    /// If the model is already `Ready`, returns immediately (idempotent).
    /// If VRAM is insufficient, evicts the least-recently-used Ready model
    /// and retries until the model fits or no further eviction is possible.
    ///
    /// # Errors
    /// - Model ID not in registry → `ModelError::NotFound`
    /// - Model is already `Loading` → error (caller should retry after delay)
    /// - Insufficient VRAM and no evictable model → error
    /// - Factory/engine failure → propagated error
    pub async fn load_model(&self, model_id: &str) -> Result<()> {
        self.load_model_with_overrides(model_id, LoadOverrides::default())
            .await
    }

    /// Like [`load_model`](Self::load_model), but applies per-load runtime
    /// overrides (co-residency Slice 2b).
    ///
    /// The `overrides` win over the per-model [`ModelPolicy`](config::ModelPolicy)
    /// and the global config defaults in [`resolve_runtime_params`]. The
    /// `POST /v1/admin/models/{id}/load` body is the only caller that passes a
    /// non-default value; everything else routes through
    /// [`load_model`](Self::load_model) with `LoadOverrides::default()`.
    ///
    /// # Errors
    /// Same as [`load_model`](Self::load_model).
    pub async fn load_model_with_overrides(
        &self,
        model_id: &str,
        overrides: LoadOverrides,
    ) -> Result<()> {
        let weight_gb = self.begin_load(model_id)?;

        // Sentinel: already Ready (idempotent path).
        if weight_gb == f64::NEG_INFINITY {
            return Ok(());
        }

        // T5.1: cancellation guard. From here the record is `Loading`; if this
        // future is dropped at any await below (the HTTP client disconnecting
        // kills the handler task), neither the success path nor the error path
        // would run and the record would wedge in `Loading` forever (503
        // forever; reload and evict both rejected). The guard's `Drop` runs
        // `abort_load` — idempotent: resets only a still-`Loading` record and
        // VRAM-free is a no-op when nothing was reserved — unless `disarm()`ed
        // on a normal exit.
        let guard = LoadCancelGuard {
            mm: self,
            model_id,
            armed: true,
        };

        // T5.2: serialise the load critical section (victim selection → VRAM
        // reservation → engine build). `begin_load` above already set this
        // record to `Loading` (so a concurrent load of the SAME model fast-fails
        // with `Loading` without waiting here); this lock then serialises loads
        // of DIFFERENT models, which would otherwise race on LRU victim selection
        // and contend on the single GPU. The next load re-evaluates VRAM with
        // this load's reservation already committed. Held until the function
        // returns (covers `ensure_vram_for`'s eviction and `execute_load`).
        let _load_guard = self.load_lock.lock().await;

        // Resolve effective model entry (dynamic overlay wins — see
        // `effective_entry`'s doc comment). Cloned already, so nothing here
        // holds a lock across the `.await`s below.
        let entry = self.effective_entry(model_id);

        // Co-residency Slice 2: resolve this model's runtime parameters once,
        // through the precedence chain (load override → per-model policy →
        // global default → builtin). The load-request body fills `overrides`
        // (Slice 2b); every other caller passes `LoadOverrides::default()`.
        let params = resolve_runtime_params(&self.config, entry.as_ref(), &overrides);

        // NPU MAX_PROMPT_LEN hint: registered fresh on every load (not just at
        // registration, unlike register_kind_hint/register_concurrent_streams_hint
        // in build_not_loaded_record — those miss startup-preloaded models, which
        // never go through build_not_loaded_record). This is the single choke
        // point every load path (preload, on-demand, add_model, reload_config)
        // funnels through, using the freshest resolved `entry` (dynamic overlay
        // wins), so a config reload's new value is picked up on the model's next
        // load. Ignored for non-NPU devices inside `load()`.
        if let Some(max_prompt_len) = entry.as_ref().and_then(|e| e.policy.max_prompt_len) {
            self.factory
                .register_max_prompt_len_hint(model_id, max_prompt_len);
        }

        // Image-gen provenance hint (Tier 3, PLAN_image_metadata_response.md):
        // registered fresh on every load, like the MAX_PROMPT_LEN hint above —
        // ignored by every non-ImageGen load inside `load()`. NOTE (unlike that
        // hint's comment, which overstates this): `entry` is resolved from
        // `dynamic_entries` ∪ the immutable startup `Config` snapshot
        // (`load_model_with_overrides`, above) — `reload_config` never mutates
        // either, so an operator's config-file edit to these three fields does
        // NOT reach a subsequent reload+load; only a full restart picks it up.
        // Filed as a known gap, not fixed here — the same limitation predates
        // this hint (MAX_PROMPT_LEN has it too).
        self.sync_image_provenance_hint(model_id, entry.as_ref());

        // MoE guard: sparse-expert models collapse under CB at c>1.  Cap
        // max_num_seqs to MOE_MAX_NUM_SEQS unless the operator already set an
        // explicit limit (in which case we warn but respect their choice).
        let cap_was_explicit = overrides.max_concurrent_streams.is_some()
            || entry
                .as_ref()
                .and_then(|e| e.policy.max_concurrent_streams)
                .is_some();
        let model_dir = self.config.models_dir.join(model_id);

        // Validate the model's files actually exist BEFORE any VRAM reservation
        // or LRU eviction below — a registered-but-absent model_id (stale config,
        // typo, name drift from the real on-disk directory) must fail cheaply
        // here rather than evict a healthy Ready model to make room for a load
        // that was always going to fail once the factory tried to open the IR.
        // Delegated to the factory (not a raw `is_dir()` here) because test
        // factories never touch the real filesystem — only `OvEngineFactory`
        // performs a real check; `EngineFactory::model_exists`'s default is `true`.
        if !self.factory.model_exists(&model_dir) {
            guard.disarm();
            self.abort_load(model_id);
            return Err(ModelError::FilesMissing(model_id.to_owned()).into());
        }

        // Completeness check (`crate::model_completeness`, 2026-08-03): the
        // directory existing is not the same as the model actually having
        // every file its kind needs — same placement principle as the
        // `model_exists` check above, fail cheap before any VRAM
        // reservation or eviction. See `incomplete_files`'s doc.
        if let Some(missing) = self.incomplete_files(model_id, &model_dir) {
            guard.disarm();
            self.abort_load(model_id);
            return Err(ModelError::FilesIncomplete(model_id.to_owned(), missing).into());
        }

        let max_num_seqs =
            apply_moe_cap(model_id, params.max_num_seqs, &model_dir, cap_was_explicit);

        // Speculative decoding (Migration step 3): the single load choke point
        // every path funnels through, including startup preloads (which skip
        // build_not_loaded_record's fail-fast check above). Extracted to keep
        // this function under the 100-line clippy cap (same reason
        // `compute_max_prompt_tokens` was split out of `execute_load`).
        let speculative_spec = entry.as_ref().and_then(|e| e.policy.speculative.as_ref());
        let draft_vram_gb =
            match self.resolve_speculative_draft_vram_gb(model_id, &model_dir, speculative_spec) {
                Ok(gb) => gb,
                Err(e) => {
                    guard.disarm();
                    self.abort_load(model_id);
                    return Err(e);
                }
            };
        let max_num_seqs = apply_speculative_cap(
            model_id,
            max_num_seqs,
            speculative_spec.is_some(),
            cap_was_explicit,
        );

        // Non-LLM models (STT / embedding / image / TTS) have no KV cache pool;
        // grabbing all remaining VRAM for "KV" is bogus and blocks co-residency.
        // Read the record's `configured_kind` — resolved once at registration
        // via the explicit config `kind` when set, else `factory.detect_kind`
        // (see `new_with_inventory`) — rather than re-deriving from the raw
        // config field alone. The two used to disagree: a model with no
        // explicit `kind` in config but a real, auto-detected non-LLM kind
        // (e.g. an embedding/STT/TTS model registered without a `"kind"`
        // field) got `needs_kv_pool = true` here regardless, silently
        // reserving a full grab-all KV pool it never needed — see
        // `dev/autotest/20260807_needs_kv_pool_ignores_auto_detected_kind.md`.
        // Unknown/missing record still defaults to `true` (the original
        // LLM-assumption fallback — safe, and this path is unreachable in
        // practice since `begin_load` above already confirmed the record
        // exists).
        let needs_kv_pool = self
            .read_models("load_model_with_overrides:needs_kv_pool")
            .get(model_id)
            .is_none_or(|r| matches!(r.configured_kind, ModelKind::TextGen | ModelKind::Vision));

        // Compute KV pool size.  For models with a known weight footprint we
        // compute it dynamically from remaining VRAM, bounded by the resolved
        // per-model KV target (`0.0` = unbounded grab-all).  For models with
        // vram_gb=0.0 (unknown) we bypass VRAM gating: honour an explicit KV
        // target if one was resolved, else fall back to the configured cap.
        let kv_gb = if weight_gb > 0.0 {
            match self
                .ensure_vram_for(
                    model_id,
                    weight_gb + draft_vram_gb,
                    params.kv_cache_gb,
                    needs_kv_pool,
                    overrides.force,
                )
                .await
            {
                Ok(kv) => kv,
                Err(e) => {
                    guard.disarm();
                    self.abort_load(model_id);
                    return Err(e);
                }
            }
        } else if params.kv_cache_gb > 0.0 {
            params.kv_cache_gb
        } else {
            self.config.cache_size_gb
        };

        let result = self.execute_load(model_id, kv_gb, max_num_seqs).await;
        guard.disarm();
        if result.is_err() {
            self.abort_load(model_id);
        }
        result
    }

    /// `POST /v1/admin/models/{id}/resize`
    /// (`dev/plans/kv-cache-pressure-detection.md`): evict then reload a
    /// currently-`Ready` model with a new `kv_cache_gb`, carrying every
    /// other resolved runtime parameter forward unchanged so the caller only
    /// has to say what's changing. Exists because the obvious "just call
    /// `/load` again with a bigger number" does not work: this same
    /// function's already-Ready sentinel above returns `Ok(())` immediately
    /// without applying new overrides — confirmed by reading it, not
    /// assumed, while designing this.
    ///
    /// Requires the model to already be `Ready` (→ [`ModelError::NotLoaded`]
    /// otherwise) — this changes a *resident* model's size, it does not load
    /// one for the first time; use `/load` for that. Pinned models ARE
    /// resizable: self-eviction-for-resize goes through the same
    /// [`Self::evict_model`] every other eviction does, which never
    /// consults `pinned` at all — only *victim selection for someone else's
    /// load* does. Not a special case, just confirmed rather than assumed.
    ///
    /// Gated by `kv_resize_cooldown_secs` regardless of whether the pressure
    /// monitor is enabled: nothing else stops a script or a nervous operator
    /// from calling this back-to-back on a single-slot model, where each
    /// call is a real evict+reload — a genuine mini-outage for that model,
    /// not a free action. The cooldown is claimed (recorded) before
    /// evicting, not after the whole operation completes, so a rapid second
    /// call fails fast rather than evicting the model twice.
    ///
    /// If eviction succeeds but the reload fails (new size doesn't fit,
    /// files missing, whatever), the model ends up unloaded, **not reverted
    /// to its old size** — deliberately no automatic rollback-and-retry; the
    /// caller must decide whether to retry at the old size or a different
    /// one, per the plan's own explicit non-goal.
    ///
    /// # Errors
    /// - `kv_cache_gb` is not finite or not positive.
    /// - [`ModelError::NotFound`] — `model_id` is not registered.
    /// - [`ModelError::NotLoaded`] — the model is not currently `Ready`.
    /// - The model was resized within `kv_resize_cooldown_secs` (a plain
    ///   message, not a `ModelError` variant — the caller-facing handler
    ///   maps it to 429 by matching the message text).
    /// - Any error [`Self::evict_model`] or [`Self::load_model_with_overrides`]
    ///   can return, propagated as-is.
    ///
    /// # Returns
    /// The KV pool the model **actually** got (`ModelRecord::kv_cache_gb`, the
    /// stored [`compute_kv_pool_gb`](Self::compute_kv_pool_gb) result), which is
    /// **not** necessarily `kv_cache_gb`: the request is a *target*, clamped by
    /// free VRAM after weights, by the global `cache_size_gb` cap, and floored at
    /// `min_kv_cache_gb`. Reporting the request back would claim a success the
    /// allocator never delivered — the caller must be able to see a short pool.
    pub async fn resize_model_kv_cache(&self, model_id: &str, kv_cache_gb: f64) -> Result<f64> {
        anyhow::ensure!(
            kv_cache_gb.is_finite() && kv_cache_gb > 0.0,
            "kv_cache_gb must be a finite positive number, got {kv_cache_gb}"
        );

        let max_concurrent_streams = {
            let guard = self.read_models("resize_model_kv_cache");
            let record = guard
                .get(model_id)
                .ok_or_else(|| ModelError::NotFound(model_id.to_owned()))?;
            if record.state != ModelState::Ready {
                return Err(ModelError::NotLoaded.into());
            }
            record.max_concurrent_streams
        };

        // Cooldown gate, claimed (not just checked) before evicting: a
        // second call arriving mid-resize should fail fast on the cooldown
        // rather than race `evict_model`'s own Evicting-state guard.
        {
            let cooldown = Duration::from_secs(self.config.kv_resize_cooldown_secs);
            let mut tracker = self
                .kv_pressure
                .lock()
                .map_err(|_| anyhow::anyhow!("kv_pressure tracker lock poisoned"))?;
            let state = tracker.entry(model_id.to_owned()).or_default();
            if let Some(last) = state.last_resized_at {
                let elapsed = last.elapsed();
                anyhow::ensure!(
                    elapsed >= cooldown,
                    "model '{model_id}' was resized {}s ago — cooldown is {}s, retry shortly",
                    elapsed.as_secs(),
                    cooldown.as_secs()
                );
            }
            state.last_resized_at = Some(Instant::now());
        }

        self.evict_model(model_id).await?;
        self.load_model_with_overrides(
            model_id,
            LoadOverrides {
                kv_cache_gb: Some(kv_cache_gb),
                max_concurrent_streams: (max_concurrent_streams > 0)
                    .then_some(max_concurrent_streams),
                force: false,
            },
        )
        .await?;

        // Report what the allocator delivered, not what was asked for. The
        // load path stores `ensure_vram_for`'s return (the `compute_kv_pool_gb`
        // result) on the record; echoing the *request* instead would report a
        // clamped pool as a full-size success.
        // Unreachable in practice (the load above just succeeded), but the
        // fallback must not be `kv_cache_gb` — echoing the request is the exact
        // lie this function exists to stop telling.
        let achieved = self
            .read_models("resize_model_kv_cache:achieved")
            .get(model_id)
            .map(|r| r.kv_cache_gb)
            .ok_or_else(|| {
                anyhow::anyhow!("model '{model_id}' vanished from the registry during resize")
            })?;
        Ok(achieved)
    }

    /// Image-gen provenance hint (Tier 3, `PLAN_image_metadata_response.md`),
    /// extracted from [`load_model_with_overrides`](Self::load_model_with_overrides)
    /// to keep it under the 100-line clippy cap — same reason
    /// [`resolve_speculative_draft_vram_gb`](Self::resolve_speculative_draft_vram_gb)
    /// below was extracted.
    ///
    /// Always registers — including an all-`None` hint when `entry` is absent
    /// or carries no Tier 3 fields. This is deliberate, not a missed
    /// optimization: an *earlier* load may have registered a real hint for
    /// this `model_id` (e.g. before an operator removed `precision`/
    /// `model_source`/`model_revision` from config, or swapped the checkpoint
    /// via `deregister_model` + `add_model`), and `OvEngineFactory`'s hint map
    /// only ever grows via `insert` — nothing else ever clears an entry once
    /// written. Skipping registration here when the *current* config has
    /// nothing to say would leave that stale hint in place forever, so a
    /// later `/v1/images/generations` response would keep reporting the OLD
    /// checkpoint's provenance next to the NEW checkpoint's freshly-hashed
    /// `model_hash` — fabricated, self-contradicting metadata, exactly what
    /// the plan's "omit, don't fabricate" rule forbids. Always overwriting
    /// (with an all-`None` hint acting as a clear) closes that gap; the
    /// write-lock cost is negligible next to the GPU JIT compile every image
    /// load already pays.
    fn sync_image_provenance_hint(&self, model_id: &str, entry: Option<&config::ModelEntry>) {
        let policy = entry.map(|e| &e.policy);
        self.factory.register_image_provenance_hint(
            model_id,
            crate::pipelines::image::ImageProvenanceHint {
                precision: policy.and_then(|p| p.precision.clone()),
                model_source: policy.and_then(|p| p.model_source.clone()),
                model_revision: policy.and_then(|p| p.model_revision.clone()),
            },
        );
    }

    /// Speculative decoding (Migration step 3), extracted from
    /// [`load_model_with_overrides`](Self::load_model_with_overrides) to keep
    /// it under the 100-line clippy cap. Validates the pairing (Layer B,
    /// disk-inspecting — Part 4 of the project's internal engineering log),
    /// registers (or, for `None`, clears) the draft hint the factory consumes
    /// at `OvCbEngine` construction, and returns the draft's VRAM footprint to
    /// fold into this load's reservation (`0.0` when not speculative) — Part 6:
    /// one reservation, one `model_id`, so eviction frees the combined amount.
    ///
    /// # Errors
    /// Returns [`validate_speculative_pairing`]'s error when `spec` is `Some`
    /// and the pairing fails any gate. The draft hint is left unregistered on
    /// error — the caller aborts the load entirely.
    fn resolve_speculative_draft_vram_gb(
        &self,
        model_id: &str,
        model_dir: &std::path::Path,
        spec: Option<&config::SpeculativeConfig>,
    ) -> Result<f64> {
        let Some(spec) = spec else {
            self.factory.register_draft_hint(model_id, None);
            return Ok(0.0);
        };
        let draft_dir = self.config.models_dir.join(&spec.draft_model);
        let target_kind = self
            .read_models("load_model_with_overrides:speculative")
            .get(model_id)
            .map_or(ModelKind::TextGen, |r| r.configured_kind);
        let target_device = self.record_device(model_id);
        validate_speculative_pairing(model_dir, &draft_dir, target_kind, &target_device, spec)?;
        self.factory.register_draft_hint(
            model_id,
            Some(lifecycle::DraftHint {
                draft_path: draft_dir.to_string_lossy().into_owned(),
                draft_device: target_device,
                num_assistant_tokens: spec.num_assistant_tokens,
                verify_on_load: spec.verify_on_load,
            }),
        );
        Ok(spec.draft_vram_gb)
    }

    /// Start a background load for a `NotLoaded` model **iff** its policy is
    /// `load: on_demand`, then return immediately (co-residency Slice 3c).
    ///
    /// The chat path calls this when [`get_chat_context`](Self::get_chat_context)
    /// reports `NotLoaded`: an on-demand model (e.g. SDXL on a 16 GB B50) loads
    /// lazily on first use instead of failing fast. The load runs on a detached
    /// task via [`load_model`](Self::load_model), so it applies the model's
    /// **configured** [`ModelPolicy`](config::ModelPolicy) overrides — never
    /// anything from the chat request body (the `OpenAI` chat schema carries no
    /// kv/stream knobs). Idempotent: [`begin_load`](Self::begin_load) is the
    /// authoritative `NotLoaded → Loading` CAS under the write lock, so a second
    /// concurrent request whose task loses the race simply exits — at most one
    /// engine is ever built.
    ///
    /// Returns [`OnDemandLoad::Loading`] when a load was started (or one is
    /// already in flight) so the caller answers 503 + `Retry-After`;
    /// [`OnDemandLoad::NotApplicable`] for an eager (or unknown) model, leaving
    /// today's bare-503 behaviour intact.
    pub fn request_on_demand_load(self: &Arc<Self>, model_id: &str) -> OnDemandLoad {
        // Eager (or no policy) ⇒ no auto-load. Overlay-first via
        // `effective_entry` — this used to read `self.config.models` only,
        // so a model registered via `add_model` with `load: on_demand` (or
        // `patch_model`'d to it later) never actually got the on-demand
        // behaviour its own config declared. No `.await` below this line
        // needs the models lock either way.
        let on_demand = self
            .effective_entry(model_id)
            .map(|e| e.policy.load)
            .unwrap_or_default()
            == config::LoadPolicy::OnDemand;
        if !on_demand {
            return OnDemandLoad::NotApplicable;
        }

        // Only spawn from a genuine NotLoaded state. If the model has since moved
        // to Loading/Ready/Evicting, a retry is the right answer and no new task
        // is needed (begin_load would reject it anyway). This check is advisory —
        // begin_load remains the authoritative CAS — it just avoids a dead spawn.
        match self.read_models("on_demand:state").get(model_id) {
            Some(r) if r.state == ModelState::NotLoaded => {}
            // Already in flight / resident / evicting: tell the client to retry.
            Some(_) => return OnDemandLoad::Loading,
            // Unknown id cannot reach here (chat maps NotFound first); defensive.
            None => return OnDemandLoad::NotApplicable,
        }

        // Detached load. begin_load (inside load_model) flips NotLoaded→Loading
        // under the write lock and fast-fails a racing duplicate, so spawning
        // unconditionally from this advisory check is safe.
        let mm = Arc::clone(self);
        let id = model_id.to_owned();
        tokio::spawn(async move {
            if let Err(e) = mm.load_model(&id).await {
                tracing::warn!(model = %id, error = %e, "on-demand load failed");
            }
        });
        OnDemandLoad::Loading
    }

    /// D5/D8: start a background load for `model_id` if it is currently
    /// `NotLoaded`, **regardless of its configured [`LoadPolicy`](config::LoadPolicy)**
    /// — unlike [`request_on_demand_load`](Self::request_on_demand_load), which
    /// only fires for `load: on_demand` models. Realtime resolution decides for
    /// itself when a background load is appropriate (`resolve_realtime_llm`);
    /// a model's own eager/on-demand policy is a narrower, separate concern
    /// that stays untouched by this path. Idempotent for the same reason
    /// `request_on_demand_load` is: [`begin_load`](Self::begin_load) (inside
    /// [`load_model`](Self::load_model)) is the authoritative
    /// `NotLoaded → Loading` CAS under the write lock, so a second concurrent
    /// call that loses the advisory check below simply returns — at most one
    /// engine is ever built. A no-op (returns without spawning) for an unknown
    /// model id, or one already `Loading`/`Ready`/`Evicting`.
    fn spawn_background_load(self: &Arc<Self>, model_id: &str) {
        match self
            .read_models("resolve_realtime_llm:spawn_load")
            .get(model_id)
        {
            Some(r) if r.state == ModelState::NotLoaded => {}
            _ => return,
        }
        let mm = Arc::clone(self);
        let id = model_id.to_owned();
        tokio::spawn(async move {
            if let Err(e) = mm.load_model(&id).await {
                tracing::warn!(model = %id, error = %e, "realtime background load failed");
            }
        });
    }

    /// Resolve `id`'s effective [`config::ModelEntry`]: the dynamic overlay
    /// wins over the immutable startup [`Config`] snapshot. This is the one
    /// precedence rule every per-model policy lookup must use — `self.config`
    /// is never mutated after startup (a config-file edit only ever reaches
    /// the live server through this overlay, via `add_model`, `patch_model`,
    /// or `reload_config` mirroring a refreshed file entry into it), so a
    /// site that reads `self.config.models` first and the overlay only as a
    /// fallback can never observe a `patch_model` change, or a `reload_config`
    /// refresh, for a model that was *also* declared in `config.json` at
    /// boot — exactly the shape every real fleet model takes. Was previously
    /// duplicated ad hoc (three sites had the precedence backwards; see the
    /// project's internal engineering log) — this is now the single source.
    fn effective_entry(&self, id: &str) -> Option<config::ModelEntry> {
        self.dynamic_entries
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(id)
            .cloned()
            .or_else(|| self.config.models.get(id).cloned())
    }

    /// D5: `true` when `id` is a known model — either in the static
    /// `config.models` map or registered dynamically via
    /// `POST /v1/admin/models/add`.
    fn known_model(&self, id: &str) -> bool {
        self.config.models.contains_key(id)
            || self
                .dynamic_entries
                .read()
                .is_ok_and(|d| d.contains_key(id))
    }

    /// D5/D8: `true` when `id` is currently `Ready`. `pub` so the realtime
    /// handler can poll it from `spawn_model_ready_watcher` (D8's
    /// `model_loading` completion signal) without a load-completion
    /// notification primitive existing (a plain poll is cheap and this only
    /// runs while a background load is genuinely in flight).
    #[must_use]
    pub fn is_ready(&self, id: &str) -> bool {
        self.read_models("resolve_realtime_llm:is_ready")
            .get(id)
            .is_some_and(|r| r.state == ModelState::Ready)
    }

    /// RTCC: best-effort lookup of `id`'s registered [`ModelKind`] — prefers
    /// the live record's kind (set at registration, refined once loaded),
    /// falling back to the static config declaration for a model not yet
    /// registered as a record. `pub` so the realtime handler can classify a
    /// `request_model_load` target into the right [`RealtimeSlot`] before
    /// resolving it — mirrors [`is_ready`](Self::is_ready)'s reason for
    /// being `pub`. `None` only for a genuinely unknown id.
    #[must_use]
    pub fn known_model_kind(&self, id: &str) -> Option<ModelKind> {
        if let Some(r) = self.read_models("known_model_kind").get(id) {
            return Some(r.last_kind.unwrap_or(r.configured_kind));
        }
        self.effective_entry(id)
            .and_then(|e| e.kind.and_then(|k| ModelKind::from_label(&k)))
    }

    /// D4: declared `ModelPolicy::capabilities` for `id` — the dynamic
    /// overlay wins over static config (see [`effective_entry`]'s doc
    /// comment), so a `patch_model` change to this field is seen
    /// immediately. Empty if unknown or unset.
    fn model_capabilities(&self, id: &str) -> Vec<String> {
        self.effective_entry(id)
            .map(|e| e.policy.capabilities)
            .unwrap_or_default()
    }

    /// D4: does `id`/`r` (assumed `Ready`) clear the realtime
    /// viable-minimum floor? Absent `realtime_viable_minimum.llm` config ⇒
    /// every `text_gen`/`vision` model clears it — today's de-facto
    /// behavior, unchanged for nodes that don't configure a floor.
    fn clears_viable_minimum(&self, id: &str, r: &ModelRecord) -> bool {
        let kind = r.last_kind.unwrap_or(r.configured_kind);
        let Some(vm) = self
            .config
            .realtime_viable_minimum
            .as_ref()
            .and_then(|v| v.llm.as_ref())
        else {
            return matches!(kind, ModelKind::TextGen | ModelKind::Vision);
        };
        if !vm
            .kinds
            .iter()
            .any(|k| ModelKind::from_label(k) == Some(kind))
        {
            return false;
        }
        if vm.min_context_tokens > 0 && r.max_prompt_tokens < vm.min_context_tokens {
            return false;
        }
        if !vm.require_capabilities.is_empty() {
            let caps = self.model_capabilities(id);
            if !vm
                .require_capabilities
                .iter()
                .all(|req| caps.iter().any(|c| c == req))
            {
                return false;
            }
        }
        true
    }

    /// RTCC: `id`'s slot-appropriate `realtime_defaults` entry. STT/TTS map
    /// to their own dedicated fields; `Llm` covers text-gen and vision alike
    /// (one configured default serves both, matching `clears_viable_
    /// minimum`'s existing `TextGen | Vision` floor).
    fn slot_default(&self, slot: RealtimeSlot) -> Option<String> {
        let d = self.config.realtime_defaults.as_ref()?;
        match slot {
            RealtimeSlot::Llm => d.llm_model.clone(),
            RealtimeSlot::Stt => d.stt_model.clone(),
            RealtimeSlot::Tts => d.tts_model.clone(),
            // Deliberately not `d.embed_model` — embed is grant-only (see
            // RealtimeSlot::Embed's doc comment); a configured default here
            // would auto-load-in-background via resolve_realtime_slot step
            // 4, contradicting "convenience only, never forced."
            RealtimeSlot::Embed => None,
        }
    }

    /// RTCC: does `id`/`r` (assumed `Ready`) clear `slot`'s viable-minimum
    /// floor? `Llm` reuses [`clears_viable_minimum`](Self::clears_viable_minimum)
    /// verbatim (the only slot with a configurable floor, D4). STT/TTS have
    /// no `realtime_viable_minimum` section of their own — per D4, exact
    /// kind-matching (already enforced at use-time by `get_stt_handle`/
    /// `get_tts_handle`) *is* their entire floor. `Embed` always returns
    /// `false` — no Ready model is ever offered as a substitute for an
    /// embedding request (grant-only, see `RealtimeSlot::Embed`).
    fn clears_viable_minimum_for_slot(
        &self,
        slot: RealtimeSlot,
        id: &str,
        r: &ModelRecord,
    ) -> bool {
        match slot {
            RealtimeSlot::Llm => self.clears_viable_minimum(id, r),
            RealtimeSlot::Stt => r.last_kind.unwrap_or(r.configured_kind) == ModelKind::Stt,
            RealtimeSlot::Tts => r.last_kind.unwrap_or(r.configured_kind) == ModelKind::Tts,
            RealtimeSlot::Embed => false,
        }
    }

    /// D4+D5, generalized to any [`RealtimeSlot`] (RTCC): the best
    /// already-`Ready` model that clears the slot's viable-minimum floor, or
    /// `None` if nothing does. "Best" prefers the slot's configured default
    /// if it qualifies, else the floor-clearing model currently serving the
    /// most realtime sessions (D2's serving set) — convergence, so multiple
    /// concurrent voice clients collapse onto one resident model instead of
    /// each demanding a different one (the "optimal set" payoff
    /// `resolve_realtime_slot`'s doc comment describes).
    fn best_ready_viable(&self, slot: RealtimeSlot) -> Option<String> {
        let default_id = self.slot_default(slot);
        let guard = self.read_models("resolve_realtime_slot:best_viable");
        if let Some(ref id) = default_id
            && let Some(r) = guard.get(id.as_str())
            && r.state == ModelState::Ready
            && self.clears_viable_minimum_for_slot(slot, id, r)
        {
            return Some(id.clone());
        }
        guard
            .iter()
            .filter(|(id, r)| {
                r.state == ModelState::Ready && self.clears_viable_minimum_for_slot(slot, id, r)
            })
            .max_by_key(|(id, _)| self.voice_pin.as_ref().map_or(0, |vp| vp.serving_count(id)))
            .map(|(id, _)| id.clone())
    }

    /// D5, generalized to any [`RealtimeSlot`] (RTCC — was LLM-only in v2):
    /// resolve what a realtime session's `slot` should actually get, given
    /// what the client requested (`None` = nothing requested — e.g.
    /// blank-config discovery, or an unresolved STT/TTS slot before this
    /// generalization). Implements the "optimal set" decision rule from
    /// the project's internal engineering log: *serve every
    /// active voice session from floor-clearing models while minimizing
    /// loads/evictions; honor a client's literal preference exactly when
    /// honoring it disrupts nothing* — now applied uniformly to STT/LLM/TTS
    /// per the project's internal engineering log.
    ///
    /// Never depends on `/v1/admin/*` (requirement 5, raised after a voice
    /// client's direct admin-load call evicted an operator's just-loaded
    /// model live on this box) — every load this triggers goes through
    /// [`load_model`](Self::load_model) via a task spawned from *inside this
    /// process* ([`spawn_background_load`](Self::spawn_background_load)),
    /// exactly like [`request_on_demand_load`](Self::request_on_demand_load),
    /// never a call to `POST /v1/admin/models/{id}/load`.
    ///
    /// Decision order (D5 §1-4, all decided 2026-08-06; kind-agnostic parts
    /// — [`known_model`](Self::known_model), [`check_admission`](Self::check_admission),
    /// [`is_ready`](Self::is_ready) — were already slot-agnostic before this
    /// generalization, verified during the RTCC investigation):
    /// 1. Requested model is `Ready` → grant (`source: "requested"`).
    /// 2. Requested model is loadable without evicting anything protected
    ///    (checked via the existing side-effect-free `check_admission`
    ///    dry-run, which under D1 already accounts for the D2 serving set
    ///    and D3 grace protection) → grant, kick a background load
    ///    (`loading: true`).
    /// 3. Requested model infeasible (or nothing requested): substitute the
    ///    best already-`Ready` floor-clearing model for `slot`
    ///    ([`best_ready_viable`](Self::best_ready_viable)) — `source:
    ///    "default"` if it's the slot's configured default, else
    ///    `"substituted"`.
    /// 4. Nothing `Ready` clears the floor → fall back to the slot's
    ///    configured default with a background load (`loading: true`) —
    ///    **decided**: auto-load rather than refuse (open question 5).
    ///
    /// If no default is configured for `slot` and step 3 also found
    /// nothing, this reports the literal request (or an empty id when none
    /// was given) with `source: "substituted"` so the caller's resulting
    /// per-turn errors are honest about what they're waiting on — never
    /// panics, never fabricates a model id that isn't real.
    pub fn resolve_realtime_slot(
        self: &Arc<Self>,
        slot: RealtimeSlot,
        requested: Option<&str>,
    ) -> RealtimeResolution {
        if let Some(id) = requested
            && self.is_ready(id)
        {
            return RealtimeResolution {
                model_id: id.to_owned(),
                source: "requested",
                reason: None,
                loading: false,
            };
        }

        if let Some(id) = requested
            && self.known_model(id)
            && self.check_admission(id).is_ok_and(|c| c.fits)
        {
            self.spawn_background_load(id);
            return RealtimeResolution {
                model_id: id.to_owned(),
                source: "requested",
                reason: None,
                loading: true,
            };
        }

        if let Some(id) = self.best_ready_viable(slot) {
            let is_default = self.slot_default(slot).as_deref() == Some(id.as_str());
            return RealtimeResolution {
                model_id: id,
                source: if is_default { "default" } else { "substituted" },
                reason: requested.is_some().then_some("would_evict_protected"),
                loading: false,
            };
        }

        if let Some(default_id) = self.slot_default(slot) {
            self.spawn_background_load(&default_id);
            return RealtimeResolution {
                model_id: default_id,
                source: "default",
                reason: Some("not_resident_no_headroom"),
                loading: true,
            };
        }

        RealtimeResolution {
            model_id: requested.unwrap_or_default().to_owned(),
            source: "substituted",
            reason: Some("not_resident_no_headroom"),
            loading: false,
        }
    }

    /// Thin wrapper over [`resolve_realtime_slot`](Self::resolve_realtime_slot)
    /// for the LLM/VLM slot — kept so existing call sites and tests written
    /// against v2's original LLM-only resolver don't need to change.
    pub fn resolve_realtime_llm(self: &Arc<Self>, requested: Option<&str>) -> RealtimeResolution {
        self.resolve_realtime_slot(RealtimeSlot::Llm, requested)
    }

    /// Drains and evicts a loaded model, freeing its VRAM.
    ///
    /// Normal (admin / LRU) eviction: drains in-flight work first, then exits.
    /// The process-shutdown path uses [`shutdown`](Self::shutdown), which aborts
    /// in-flight work so a long generation cannot delay teardown.
    ///
    /// The eviction protocol:
    /// 1. Mark state → `Evicting` (new requests see 503 immediately).
    /// 2. Drop the `EngineHandle` — closes the command channel.
    /// 3. The engine thread finishes in-flight requests, then exits.
    /// 4. Join the thread — guarantees the pipeline is destroyed and VRAM freed.
    /// 5. Mark state → `NotLoaded`, update VRAM tracker.
    ///
    /// # Errors
    /// - Model ID not in registry → error
    /// - Model is not in `Ready` state → error (cannot evict what isn't loaded)
    pub async fn evict_model(&self, model_id: &str) -> Result<()> {
        self.evict_model_inner(model_id, false).await
    }

    /// Eviction core. When `abort_inflight` is true, flips the engine's shutdown
    /// flag before the join so it stops within one model step instead of
    /// finishing the current generation — the process-shutdown path (#7), where
    /// waiting out a long completion would blow the stop script's budget and
    /// provoke an operator reboot. Normal eviction passes `false` and instead
    /// polls the engine's own occupancy (see below) before dropping its handle.
    async fn evict_model_inner(&self, model_id: &str, abort_inflight: bool) -> Result<()> {
        let (handle, thread) = self.begin_evict(model_id)?;
        let kind = handle.kind();

        if abort_inflight {
            // On the shutdown path, tell the engine to stop promptly BEFORE
            // dropping this clone: an in-flight request may hold another clone
            // that keeps the command channel open, so closing this one alone
            // would not stop it (#7).
            handle.request_shutdown();
        } else {
            // Normal (admin/LRU) eviction: give any real in-flight request(s) a
            // genuine chance to reach their natural completion before we tear
            // the engine down. This is the caller-side half of the drain fix
            // (2026-07-18, live-caught on real hardware): dropping this handle
            // clone below only stops ABORTING a live streaming request once the
            // engine loop itself does the right thing on channel-disconnect
            // (see `cb_engine::engine_loop`) — but a real drain can legitimately
            // take as long as the generation itself, which JOIN_DEADLINE (an
            // 8s backstop against a *wedged* thread, not a busy one) is far too
            // short for. Poll instead of waiting on the join for this part.
            //
            // Bounded by EVICT_DRAIN_DEADLINE so a pathological runaway
            // generation cannot hang eviction forever: past that point, force
            // an abort (same as the shutdown path) rather than wait indefinitely.
            let deadline = Instant::now() + EVICT_DRAIN_DEADLINE;
            while handle.as_managed().active() > 0 && Instant::now() < deadline {
                tokio::time::sleep(EVICT_DRAIN_POLL_INTERVAL).await;
            }
            if handle.as_managed().active() > 0 {
                tracing::warn!(
                    model_id,
                    kind = kind.label(),
                    active = handle.as_managed().active(),
                    deadline_s = EVICT_DRAIN_DEADLINE.as_secs(),
                    "eviction drain deadline exceeded — aborting remaining in-flight request(s)"
                );
                handle.request_shutdown();
            }
        }
        // Close this clone of the command channel. With no surviving clone the
        // channel disconnects; the engine either has no live requests left (the
        // normal path, having just drained above) or was just told to abort
        // them (shutdown, or a drain-deadline fallback) — either way it now
        // exits promptly.
        drop(handle);

        // Blocking join — must run off the async runtime thread. The closure
        // collapses the join result to a bool so the panic payload
        // (`Box<dyn Any>`, awkward to carry across the await) is dropped on the
        // blocking thread. BOUNDED by JOIN_DEADLINE (#36): if the engine thread
        // does not exit in time (a wedged FFI step / never-settling pipeline),
        // abandon the wait rather than hang eviction — and therefore process
        // shutdown — forever. The detached blocking task frees the thread's VRAM
        // when it finally exits, or the OS reclaims it at process exit.
        let join = tokio::task::spawn_blocking(move || thread.join().is_ok());
        let clean_exit = if let Ok(res) = tokio::time::timeout(JOIN_DEADLINE, join).await {
            res.unwrap_or(false)
        } else {
            tracing::error!(
                model_id,
                kind = kind.label(),
                deadline_s = JOIN_DEADLINE.as_secs(),
                "engine thread did not exit within join deadline — abandoning join"
            );
            false
        };

        // T5.2: reconcile the tracker REGARDLESS of how the engine thread exited
        // (committee finding 38). A panicked engine thread still unwound its
        // stack — the pipeline `Drop` ran and freed the GPU memory — but it
        // skipped the normal exit. Without this unconditional `finish_evict`, a
        // panic would wedge the record in `Evicting` forever AND leave the VRAM
        // reservation in the tracker, which then permanently over-counts and
        // spuriously rejects every later load.
        self.finish_evict(model_id);

        if !clean_exit {
            tracing::error!(
                model_id,
                kind = kind.label(),
                "engine thread did not exit cleanly during eviction — tracker reconciled, \
                 record reset to NotLoaded"
            );
            return Err(anyhow::anyhow!(
                "engine thread did not exit cleanly during eviction (panic or join timeout)"
            ));
        }

        tracing::info!(model_id, kind = kind.label(), "model evicted");
        if let Some(vp) = &self.voice_pin {
            vp.clear_if_llm(model_id);
        }
        // Clear the pressure flag (it's meaningless for an unloaded model),
        // but deliberately KEEP `last_resized_at`: the resize cooldown must
        // survive the resize's own evict+reload cycle, and a normal LRU
        // eviction of a recently-resized model shouldn't reset its cooldown
        // either — the cooldown throttles resize churn on a time basis,
        // unrelated to why any particular eviction happened.
        if let Ok(mut kv_pressure) = self.kv_pressure.lock()
            && let Some(state) = kv_pressure.get_mut(model_id)
        {
            state.over_threshold_since = None;
        }
        Ok(())
    }

    /// Evict every `Ready` model — the process-shutdown path (T6.1).
    ///
    /// Each eviction runs the full quiesce protocol: the engine handle drops
    /// (closing the command channel), the engine thread terminates any
    /// still-live streams with terminal events (T1.3) and exits, the thread is
    /// joined, and the pipeline `Drop` frees its VRAM — so the GPU is clean
    /// before the process exits rather than relying on the driver to reap it.
    ///
    /// Best-effort: a failed eviction is logged and the remaining models are
    /// still evicted (shutdown must not wedge on one bad engine).
    pub async fn shutdown(&self) {
        for info in self.list_models() {
            // `abort_inflight = true`: process shutdown must not wait out a long
            // in-flight generation — terminate it cleanly (T1.3) and tear down (#7).
            if info.state == ModelState::Ready
                && let Err(e) = self.evict_model_inner(&info.id, true).await
            {
                tracing::warn!(model_id = %info.id, error = %e, "eviction during shutdown failed");
            }
        }
    }

    /// Returns a snapshot of all known models and their current states.
    ///
    /// Used by `GET /v1/models` (`OpenAI`: list available models) and the
    /// admin `GET /v1/admin/models` endpoint.
    #[must_use]
    pub fn list_models(&self) -> Vec<ModelInfo> {
        let guard = self.read_models("list_models");
        guard
            .iter()
            .map(|(id, r)| {
                let live = (r.state == ModelState::Ready)
                    .then_some(r.handle.as_ref())
                    .flatten()
                    .map(EngineHandleKind::as_managed);
                ModelInfo {
                    id: id.clone(),
                    state: r.state,
                    kind: r.configured_kind,
                    vram_gb: r.vram_gb,
                    device: r.device.clone(),
                    generation_defaults: r.generation_defaults,
                    max_prompt_tokens: r.max_prompt_tokens,
                    kv_cache_gb: r.kv_cache_gb,
                    native_context_limit: r.native_context_limit,
                    kv_cache_usage_pct: live.map_or(0.0, ManagedEngine::cache_usage_pct),
                    kv_pressure_flagged: self.kv_pressure_flagged_now(id),
                }
            })
            .collect()
    }

    /// Reads whether `model_id` is currently flagged for sustained KV
    /// pressure — a pure read of the tracker the periodic sweep maintains
    /// ([`Self::run_kv_pressure_sweep_once`]); never samples or mutates.
    /// `false` whenever the monitor is disabled, the model has no tracked
    /// state (never sampled, or evicted since — eviction clears its entry),
    /// or the sustained duration hasn't yet elapsed.
    fn kv_pressure_flagged_now(&self, model_id: &str) -> bool {
        if !self.config.kv_pressure_monitor_enabled {
            return false;
        }
        // Reads exactly what the last sweep tick decided — see
        // `KvPressureState::flagged`'s doc comment for why this must be the
        // persisted bit, not re-derived from `over_threshold_since` here too.
        self.kv_pressure
            .lock()
            .is_ok_and(|tracker| tracker.get(model_id).is_some_and(|s| s.flagged))
    }

    /// `load_model_with_overrides`'s completeness gate: `Some(blocking
    /// files)` when `model_id`'s directory is missing files its
    /// `configured_kind` needs to actually run — the exact class of bug
    /// this closes (2026-08-03): `ms-marco-MiniLM-L6-v2-int8-ov` was
    /// missing its converted `openvino_tokenizer.*` pair, loaded
    /// successfully, and only failed on the first real inference call.
    /// `None` when complete. `configured_kind` was already resolved at
    /// registration (`entry.kind` or `detect_kind`'s sniff) — reused here,
    /// not re-sniffed.
    fn incomplete_files(&self, model_id: &str, model_dir: &std::path::Path) -> Option<Vec<String>> {
        let kind = self
            .read_models("load_model_with_overrides:completeness")
            .get(model_id)
            .map_or(ModelKind::TextGen, |r| r.configured_kind);
        let report = self.factory.check_completeness(model_dir, kind);
        (!report.is_complete()).then_some(report.blocking_missing)
    }

    /// Filesystem-only completeness check for one registered model
    /// (`crate::model_completeness`) — safe to call regardless of the
    /// model's current state (`NotLoaded`/`Ready`/etc.), never touches VRAM
    /// or constructs an engine. Uses the model's already-resolved
    /// `configured_kind` (set at registration), not a fresh sniff. Returns
    /// a complete (empty) report for an unknown `model_id` — callers that
    /// need existence checking should use [`list_models`](Self::list_models)
    /// first, same contract this trait method's mock default already
    /// establishes for [`EngineFactory::check_completeness`](lifecycle::EngineFactory::check_completeness).
    #[must_use]
    pub fn check_model_completeness(
        &self,
        model_id: &str,
    ) -> crate::model_completeness::CompletenessReport {
        let Some(kind) = self
            .read_models("check_model_completeness")
            .get(model_id)
            .map(|r| r.configured_kind)
        else {
            return crate::model_completeness::CompletenessReport::default();
        };
        let model_dir = self.config.models_dir.join(model_id);
        self.factory.check_completeness(&model_dir, kind)
    }

    /// Returns model IDs that are currently in the `Ready` state.
    ///
    /// Used by `GET /v1/models` for the `OpenAI`-compatible model list.
    #[must_use]
    pub fn list_ready_models(&self) -> Vec<String> {
        let guard = self.read_models("list_ready_models");
        guard
            .iter()
            .filter(|(_, r)| r.state == ModelState::Ready)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Age of the longest generation currently in flight across every Ready
    /// model, or `None` if nothing is generating (or no Ready engine tracks
    /// it — see [`ManagedEngine::oldest_generation_age`]'s default).
    ///
    /// Backs the supervisor's watchdog: `/health` alone cannot see a wedged
    /// engine thread (the HTTP listener stays up regardless), so this is a
    /// second, independent signal — see
    /// the project's internal engineering log.
    #[must_use]
    pub fn oldest_generation_age(&self) -> Option<Duration> {
        let guard = self.read_models("oldest_generation_age");
        guard
            .values()
            .filter(|r| r.state == ModelState::Ready)
            .filter_map(|r| r.handle.as_ref())
            .filter_map(|h| h.as_managed().oldest_generation_age())
            .max()
    }

    /// Whether every `Ready` model currently has zero in-flight requests.
    ///
    /// **Not currently called by anything** — `run_cache_sweep_once_blocking`'s
    /// two passes (hash-precompute, prune) are both safe regardless of live
    /// traffic: precompute is pure file I/O with no load, and prune only
    /// deletes disk-cache files. This gate is reserved for the still-deferred
    /// blob-warming pass (load a never-loaded model just to capture its OV
    /// compile-cache blob, then evict it) — deferred specifically because
    /// `load_model` holds `load_lock` for the entire compile (measured 44s on
    /// a real box), so a real request could queue behind a speculative one
    /// even with this idle check in place; building blob-warming needs a
    /// design that closes that race, not just an idle gate at the start.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        let guard = self.read_models("is_idle");
        guard
            .values()
            .filter(|r| r.state == ModelState::Ready)
            .filter_map(|r| r.handle.as_ref())
            .all(|h| h.as_managed().active() == 0)
    }

    /// One pass of the OV-cache background sweep
    /// (the project's internal engineering log): hash-precompute, then
    /// prune. Deliberately does NOT include blob-warming (load a
    /// never-loaded model just to capture its OV compile-cache blob, then
    /// evict it) — see [`is_idle`](Self::is_idle)'s doc comment for why that
    /// was deferred rather than built alongside this.
    ///
    /// Blocking (file I/O, including a SHA-256 stream over multi-GB backbone
    /// files) — callers must run this via `spawn_blocking`, never directly on
    /// the async runtime.
    pub fn run_cache_sweep_once_blocking(&self) {
        self.precompute_missing_model_hashes();
        self.prune_ov_cache_if_over_budget();
    }

    /// One pass of the KV-cache pressure monitor
    /// (`dev/plans/kv-cache-pressure-detection.md`): samples every Ready
    /// model's live KV occupancy and updates how long each has been
    /// continuously at/above `kv_pressure_threshold_pct`. On a sustained
    /// trip, logs a warning — **does not evict, resize, or otherwise act**;
    /// this is "detect and flag" only, deliberately (see the plan's
    /// ops-review section on why that scope needs its own care too).
    ///
    /// Not blocking — every read here is a fast in-memory/atomic load
    /// (`cache_usage_pct`), unlike the OV-cache sweep above which streams
    /// file contents. Safe to call directly from an async task without
    /// `spawn_blocking`. No-op entirely when `kv_pressure_monitor_enabled`
    /// is false — repeated here even though the spawn site in `startup.rs`
    /// should already gate on it, so a stray direct call stays inert too.
    pub fn run_kv_pressure_sweep_once(&self) {
        if !self.config.kv_pressure_monitor_enabled {
            return;
        }
        let threshold = self.config.kv_pressure_threshold_pct;
        let sustained = Duration::from_secs(self.config.kv_pressure_sustained_secs);

        // One read-lock pass to gather live samples, released before we take
        // the (separate) kv_pressure lock below — mirrors metrics_snapshot's
        // own "read once, don't hold across other locks" shape.
        let samples: Vec<(String, String, ModelKind, f64)> = {
            let guard = self.read_models("run_kv_pressure_sweep_once");
            guard
                .iter()
                .filter_map(|(id, r)| {
                    if r.state != ModelState::Ready {
                        return None;
                    }
                    let kind = r.last_kind?;
                    let live = r.handle.as_ref()?.as_managed();
                    if !live.cache_usage_supported() {
                        return None;
                    }
                    Some((id.clone(), r.device.clone(), kind, live.cache_usage_pct()))
                })
                .collect()
        };

        let Ok(mut tracker) = self.kv_pressure.lock() else {
            return;
        };
        let now = Instant::now();
        for (model_id, device, kind, usage_pct) in samples {
            let state = tracker.entry(model_id.clone()).or_default();
            let transition = update_pressure_state(state, usage_pct, threshold, sustained, now);
            if transition == PressureTransition::JustFlagged {
                tracing::warn!(
                    model_id,
                    device,
                    kind = kind.label(),
                    usage_pct,
                    threshold_pct = threshold,
                    sustained_secs = sustained.as_secs(),
                    "KV cache pressure sustained — model flagged (no action taken; \
                     resize via POST /v1/admin/models/{{id}}/resize if warranted)"
                );
            } else if transition == PressureTransition::JustCleared {
                tracing::info!(
                    model_id,
                    device,
                    kind = kind.label(),
                    usage_pct,
                    "KV cache pressure cleared"
                );
            }
        }
    }

    /// Computes `model_hash` for every configured `ImageGen` model missing a
    /// manifest entry (or whose backbone changed since the last compute) —
    /// pure file I/O via [`resolve_model_hash`](crate::pipelines::image::resolve_model_hash),
    /// no model load, so unlike blob-warming this is safe to run regardless
    /// of live traffic (does not need [`is_idle`](Self::is_idle)).
    fn precompute_missing_model_hashes(&self) {
        let manifest_root =
            crate::cache_manifest::manifest_root(self.config.ov_cache_dir.as_deref());
        let image_gen_ids: Vec<String> = self
            .read_models("precompute_missing_model_hashes")
            .iter()
            .filter(|(_, r)| r.configured_kind == ModelKind::ImageGen)
            .map(|(id, _)| id.clone())
            .collect();
        for id in image_gen_ids {
            let model_dir = self.config.models_dir.join(&id);
            crate::pipelines::image::resolve_model_hash(&model_dir, &id, manifest_root.as_deref());
        }
    }

    /// If `ov_cache_dir`'s total on-disk size exceeds `ov_cache_max_gb`,
    /// deletes `.blob` files (never `.cl_cache`/`.onednn.cl_cache` — those are
    /// shared `OpenCL`/oneDNN kernel caches, not per-model, per the
    /// 2026-07-28 plan-file finding) until back under budget or no safe
    /// candidate remains.
    ///
    /// Priority, each tier oldest-mtime-first: (1) blobs owned by a model no
    /// longer in config, (2) blobs owned by a configured-but-idle model, (3)
    /// blobs the manifest doesn't attribute to any model. A blob owned by a
    /// currently-`Ready` **or** `Loading` model is **never** deleted, even if
    /// the cache stays over budget as a result — pruning must not force a
    /// live model's next load to recompile, and must not yank the cache file
    /// out from under a load that may be reading/reusing it mid-compile right
    /// now (`Loading` is a real, found-by-review gap fixed 2026-08-03: it is
    /// neither `Ready` nor "not in config", so it used to fall into tier 1 —
    /// a *higher* deletion priority than unattributed legacy junk).
    fn prune_ov_cache_if_over_budget(&self) {
        let Some(cache_dir) = self
            .config
            .ov_cache_dir
            .as_deref()
            .filter(|d| !d.is_empty())
            .map(std::path::Path::new)
        else {
            return;
        };

        let entries = crate::cache_manifest::list_cache_dir_entries(cache_dir);
        if entries.is_empty() {
            return;
        }
        let total_bytes: u64 = entries.iter().map(|e| e.size_bytes).sum();
        // Reported unconditionally, before the budget gate below: a box that
        // never configures `ov_cache_max_gb` (unbounded) still gets basic
        // size visibility — the gap this whole plan set out to close.
        crate::metrics::record_ov_cache_bytes(total_bytes);

        if self.config.ov_cache_max_gb <= 0.0 {
            return;
        }
        #[allow(clippy::cast_precision_loss)]
        // byte counts here are far under f64's exact-integer ceiling
        let total_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        if total_gb <= self.config.ov_cache_max_gb {
            return;
        }

        let Some(manifest_root) =
            crate::cache_manifest::manifest_root(self.config.ov_cache_dir.as_deref())
        else {
            return;
        };
        let owner_index = crate::cache_manifest::blob_owner_index(&manifest_root);
        // Protect Ready (serving requests) AND Loading (mid-compile, possibly
        // reading/reusing this exact file right now) — deleting either's
        // blob out from under it is the failure mode this function exists to
        // prevent. Evicting/NotLoaded do no cache_dir I/O, so they're
        // correctly left eligible.
        let protected_models: std::collections::HashSet<String> = self
            .read_models("prune_ov_cache_if_over_budget")
            .iter()
            .filter(|(_, r)| matches!(r.state, ModelState::Ready | ModelState::Loading))
            .map(|(id, _)| id.clone())
            .collect();

        let mut candidates: Vec<_> = entries
            .iter()
            .filter(|e| {
                std::path::Path::new(&e.file_name)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("blob"))
            })
            .filter(|e| {
                owner_index
                    .get(&e.file_name)
                    .is_none_or(|owner| !protected_models.contains(owner))
            })
            .collect();
        candidates.sort_by_key(|e| {
            let tier = match owner_index.get(&e.file_name) {
                None => 2, // unattributed — a pre-existing blob from before this feature shipped
                Some(owner) if !self.known_model(owner) => 0, // no longer configured
                Some(_) => 1, // configured, just not currently resident
            };
            (tier, e.mtime)
        });

        let mut freed_bytes = 0_u64;
        let mut deleted = 0_usize;
        for entry in candidates {
            #[allow(clippy::cast_precision_loss)]
            let remaining_gb =
                (total_bytes.saturating_sub(freed_bytes)) as f64 / (1024.0 * 1024.0 * 1024.0);
            if remaining_gb <= self.config.ov_cache_max_gb {
                break;
            }
            let path = cache_dir.join(&entry.file_name);
            match std::fs::remove_file(&path) {
                Ok(()) => {
                    freed_bytes += entry.size_bytes;
                    deleted += 1;
                }
                Err(err) => {
                    tracing::warn!(path = %path.display(), error = %err, "ov cache prune: could not delete blob, skipping");
                }
            }
        }

        if deleted > 0 {
            tracing::info!(
                deleted,
                freed_bytes,
                total_bytes,
                max_gb = self.config.ov_cache_max_gb,
                "ov cache prune: reclaimed space"
            );
            crate::metrics::record_ov_cache_pruned(freed_bytes, deleted as u64);
            // Refresh the gauge immediately rather than leaving the pre-prune
            // figure to stand until the next sweep (default 6h away).
            crate::metrics::record_ov_cache_bytes(total_bytes.saturating_sub(freed_bytes));
        }
        #[allow(clippy::cast_precision_loss)]
        let remaining_gb =
            (total_bytes.saturating_sub(freed_bytes)) as f64 / (1024.0 * 1024.0 * 1024.0);
        if remaining_gb > self.config.ov_cache_max_gb {
            tracing::warn!(
                remaining_gb,
                max_gb = self.config.ov_cache_max_gb,
                "ov cache prune: still over budget after exhausting safe candidates — \
                 all remaining blobs belong to Ready models"
            );
        }
    }

    /// Returns the next unique request ID for engine routing.
    ///
    /// IDs are globally unique across all engines managed by this instance.
    #[must_use]
    pub fn next_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Snapshot of live state for the `/metrics` endpoint.
    ///
    /// One read-lock pass over the records map, pulling each Ready model's
    /// engine counters. The `/metrics` handler turns this into Prometheus
    /// gauges at scrape time — nothing is pushed on the request hot path.
    #[must_use]
    pub fn metrics_snapshot(&self) -> MetricsSnapshot {
        let guard = self.read_models("metrics_snapshot");
        let models = guard
            .iter()
            .filter_map(|(id, r)| {
                // Report every model loaded at least once (kind known). Ready
                // models read live counters through the uniform `ManagedEngine`
                // facade; evicted models (handle gone) report `loaded == false`
                // and zero counters so `record_state` clears their stale gauges.
                let kind = r.last_kind?;
                let live = (r.state == ModelState::Ready)
                    .then_some(r.handle.as_ref())
                    .flatten()
                    .map(EngineHandleKind::as_managed);
                Some(ModelMetrics {
                    id: id.clone(),
                    device: r.device.clone(),
                    kind,
                    loaded: live.is_some(),
                    active: live.map_or(0, ManagedEngine::active),
                    max_seqs: live.map_or(0, ManagedEngine::max_concurrency),
                    waiting: live.map_or(0, ManagedEngine::waiting),
                    kv_cache_pool_gb: r.kv_cache_gb,
                    pinned: r.pinned,
                    priority: r.priority,
                    // Live occupancy: only a Ready engine reports it; evicted
                    // models and non-CB kinds fall through to 0.0.
                    kv_cache_usage_pct: live.map_or(0.0, ManagedEngine::cache_usage_pct),
                    kv_cache_usage_supported: live
                        .is_some_and(ManagedEngine::cache_usage_supported),
                    load_duration_secs: r.last_load_duration_secs,
                    kv_pressure_flagged: self.kv_pressure_flagged_now(id),
                })
            })
            .collect();
        // Legacy `device`-labelled total/used (T1/F7: both read from the same
        // single domain — the inference GPU's own — so used never exceeds
        // total for that figure) plus the full per-domain breakdown, read once
        // under the same scrape. Scoping to `inference_domain` (not summed
        // across every domain) matters as soon as co-residency spans more than
        // one: the old cross-domain sum silently double-counted the "system"
        // (CPU/iGPU, TTS/STT/embedding) domain into the GPU's own reported
        // total. A poisoned lock reports empty for all (the scrape degrades
        // rather than fails).
        let (total_vram_gb, used_vram_gb, domain_vram) =
            self.vram.read().map_or((0.0, 0.0, vec![]), |v| {
                let (total, used) = v.domain_gb(&self.inference_domain);
                (total, used, v.domain_snapshot())
            });
        MetricsSnapshot {
            device: self.config.device.clone(),
            total_vram_gb,
            used_vram_gb,
            domain_vram,
            models,
        }
    }

    /// Flag the GPU `OpenCL` context as poisoned after an OOM-class failure and
    /// log the raw diagnostics server-side (T6.2).
    ///
    /// Mirrors the load-path L3 gate for the inference hot path: once set,
    /// `begin_load` refuses new loads with [`ModelError::GpuPoisoned`] (503)
    /// until the process is restarted. `Release` pairs with the `Acquire` load
    /// in `begin_load` so a poisoning seen here is visible to later loads. The
    /// raw `err` is logged here only — callers return a scrubbed body. `err`
    /// takes any `Display` so both the load path (`&anyhow::Error`) and the
    /// streaming path (a raw `&str` from `StreamEvent::Error`) reuse it.
    pub(crate) fn mark_gpu_poisoned(&self, model_id: &str, err: impl std::fmt::Display) {
        self.gpu_poisoned.store(true, Ordering::Release);
        tracing::error!(
            model_id,
            error = %err,
            "GPU OpenCL context poisoned — process restart required before loading any model"
        );
    }

    /// Detect → Ratchet → Reload for a KV-admission wedge — originally built
    /// for the VLM hybrid-attention wedge (`dev/autotest/
    /// 20260821_omnicoder9b_qwen35_hybrid_stall.md`), and reused for a solo
    /// plain-CB pool exhaustion (`dev/autotest/
    /// 20260823_qwen3-4b-int4-ov_cb_pool_exhaustion_gap.md`) — both are the
    /// same root cause, a static `compute_max_prompt_tokens` estimate that
    /// doesn't match this model's real capacity. Called from the
    /// error-shaping path (`handlers/error.rs`) once either
    /// [`is_vlm_kv_wedge_error`] fires, or [`is_pool_exhausted_error`] fires
    /// with [`pool_exhausted_active_requests`] reading `<= 1` (no other
    /// request could have been crowding out the pool, so this isn't ordinary
    /// multi-tenant congestion), with an owned `Arc<Self>` so the actual
    /// evict+reload runs detached — the failing request's own response
    /// returns its 503 immediately; a request queued behind it gets the
    /// existing `Loading`/`Evicting` → 503 + `Retry-After` treatment from
    /// `model_error_response`, not a multi-second hang inside this call.
    ///
    /// `observed_prompt_tokens` is the calling request's own prompt length at
    /// the moment of failure — for the VLM path, whatever
    /// `vlm_engine.rs::run_generate` read from `perf_metrics` (see
    /// [`VLM_KV_WEDGE_MARKER`]'s doc comment for why that's only trustworthy
    /// from the request that actually tripped the wedge: the pipeline's
    /// `perf_metrics` freeze once wedged, so every other concurrently-failing
    /// request for the same model reads a stale, possibly much smaller,
    /// value); for the plain-CB path, `cb_engine.rs`'s own exact count for
    /// that request (no staleness concern — it's the CB scheduler's live
    /// figure, not a frozen VLM `perf_metrics` snapshot). Two safeguards
    /// against a victim corrupting the ratchet:
    /// - **First-wedge-wins claim** (`kv_wedge_recovery_inflight`): only the
    ///   first caller for a given `model_id` proceeds past this point; every
    ///   other concurrent caller returns immediately, no-op.
    /// - **Implausibility floor**: even the winning caller skips the ratchet
    ///   *write* (recovery still proceeds) when `observed_prompt_tokens` is
    ///   under 10% of the ceiling already in effect — a real first-contact
    ///   crossing lands near the true ceiling (~30% of the old, too-generous
    ///   formula estimate per the characterization doc), not at a tiny
    ///   fraction of it.
    pub(crate) fn spawn_kv_wedge_recovery(
        self: &Arc<Self>,
        model_id: &str,
        observed_prompt_tokens: usize,
    ) {
        {
            let mut inflight = self
                .kv_wedge_recovery_inflight
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if !inflight.insert(model_id.to_owned()) {
                tracing::debug!(
                    model_id,
                    "KV-admission wedge: recovery already in flight for this model, \
                     skipping duplicate trigger"
                );
                return;
            }
        }

        let current = self
            .read_models("spawn_kv_wedge_recovery")
            .get(model_id)
            .map(|r| (r.max_prompt_tokens, r.kv_cache_gb));

        match current {
            Some((current_max_prompt_tokens, kv_cache_gb)) => {
                let floor = current_max_prompt_tokens / 10;
                if observed_prompt_tokens > 0 && observed_prompt_tokens >= floor {
                    #[allow(
                        clippy::cast_precision_loss,
                        clippy::cast_sign_loss,
                        clippy::cast_possible_truncation
                    )]
                    let safe_tokens = (observed_prompt_tokens as f64 * 0.95) as usize;
                    template::write_kv_capacity_ratchet(
                        &self.config.models_dir.join(model_id),
                        safe_tokens,
                        kv_cache_gb,
                    );
                    tracing::warn!(
                        model_id,
                        observed_prompt_tokens,
                        safe_tokens,
                        kv_cache_gb,
                        "KV-admission wedge: learned a tighter capacity ceiling, \
                         recovering (evict+reload)"
                    );
                } else {
                    tracing::warn!(
                        model_id,
                        observed_prompt_tokens,
                        current_max_prompt_tokens,
                        "KV-admission wedge: observed token count too small to trust for \
                         the ratchet (stale metrics from a queued request?) — recovering \
                         without updating the learned ceiling"
                    );
                }
            }
            None => {
                tracing::warn!(
                    model_id,
                    "KV-admission wedge: model record was gone by the time recovery \
                     inspected it — recovering (evict+reload) without a ratchet update"
                );
            }
        }

        let mm = Arc::clone(self);
        let id = model_id.to_owned();
        tokio::spawn(async move {
            match mm.evict_model(&id).await {
                Ok(()) => {
                    if let Err(e) = mm.load_model(&id).await {
                        tracing::error!(
                            model_id = %id,
                            error = %e,
                            "KV-admission wedge recovery: reload failed — model left \
                             NotLoaded, next request will see model_unavailable"
                        );
                    } else {
                        tracing::info!(
                            model_id = %id,
                            "KV-admission wedge recovery: evict+reload complete"
                        );
                    }
                }
                Err(e) => {
                    tracing::error!(
                        model_id = %id,
                        error = %e,
                        "KV-admission wedge recovery: evict failed"
                    );
                }
            }
            mm.kv_wedge_recovery_inflight
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&id);
        });
    }

    // ---- Internal: state transitions ----------------------------------

    /// Phase 1 of `load_model`: validate state and transition to `Loading`.
    ///
    /// Returns the weight VRAM for this model (from config).
    /// Returns `f64::NEG_INFINITY` as a sentinel when the model is already Ready.
    fn begin_load(&self, model_id: &str) -> Result<f64> {
        // L3 gate: refuse any new load when the GPU context is poisoned.
        // Acquiring load order ensures we see any store from a prior poisoning.
        if self.gpu_poisoned.load(Ordering::Acquire) {
            return Err(ModelError::GpuPoisoned.into());
        }

        let mut guard = self.write_models("begin_load");
        let record = guard
            .get_mut(model_id)
            .ok_or_else(|| ModelError::NotFound(model_id.to_owned()))?;

        match record.state {
            ModelState::Ready => return Ok(f64::NEG_INFINITY), // sentinel: already loaded
            ModelState::Loading => return Err(ModelError::Loading.into()),
            ModelState::Evicting => {
                return Err(anyhow::anyhow!(
                    "cannot load model while it is being evicted"
                ));
            }
            ModelState::NotLoaded => record.state = ModelState::Loading,
        }
        Ok(record.vram_gb)
    }

    /// Conservative haircut applied to the raw K+V byte-math estimate in
    /// [`Self::compute_max_prompt_tokens`]. The formula counts only the raw
    /// K+V tensor bytes per token; it has no visibility into the `OpenVINO`
    /// `GenAI` CB scheduler's own block-allocation overhead (block-size
    /// rounding, per-`max_num_seqs` reservations, prefix-cache retention —
    /// mechanism unconfirmed, `OpenVINO` `GenAI`'s scheduler isn't vendored in
    /// this repo). the project's internal engineering log
    /// measured a dense (non-hybrid) model's real ceiling at ~53% of the raw
    /// formula's estimate — far more than this margin alone corrects — so
    /// this is deliberately modest, not a fitted correction: per the
    /// 2026-08-21 VLM-wedge decision, "the ratchet makes the static
    /// formula's precision non-critical." This margin only reduces how often
    /// a *first* load (before any ratchet exists) over-promises; correctness
    /// is [`Self::spawn_kv_wedge_recovery`]'s job once a real wedge or solo
    /// pool-exhaustion is observed.
    const FORMULA_SAFETY_MARGIN: f64 = 0.85;

    /// Phase 2 of `load_model`: factory call + state → `Ready`.
    ///
    /// `kv_gb` is the KV pool to request, already computed by `ensure_vram_for`
    /// (or the configured cap for unknown-VRAM models). `max_num_seqs` is the
    /// resolved per-model concurrency cap (co-residency Slice 2) — the CB
    /// scheduler limit and the HTTP 429 admission gate for this model's engine.
    /// Compute the L0 prompt-length gate ceiling for a loaded model.
    ///
    /// Returns `0` (gate disabled) for media/embedding models (`kv_gb == 0.0`) or
    /// when `config.json` is absent/unparseable — suppressing the WARN for the former
    /// since those models are not expected to carry KV-cache config.
    ///
    /// Applies any learned KV-capacity ratchet on top of the formula result
    /// (`template::read_kv_capacity_ratchet` — see its doc comment and
    /// `spawn_kv_wedge_recovery`): takes `min(formula_result, ratchet)`
    /// when a (pool-size-scaled) ratchet exists, so a real observed capacity
    /// limit always wins over a static formula's estimate, on every load
    /// (including a reload after `spawn_kv_wedge_recovery` evicts and
    /// reloads the model, and including a load at a *different* `kv_gb` than
    /// the one the ratchet was learned at), not just the process that first
    /// observed the wedge.
    /// Returns `(max_prompt_tokens, pool_capacity_tokens)`: the first is the
    /// existing L0 per-request gate ceiling (native-context/ratchet clamped —
    /// answers "is this ONE request's own prompt sane for this model"); the
    /// second is the raw, unclamped VRAM-derived formula estimate — this
    /// model's actual physical KV-pool token capacity, used by
    /// [`EngineHandle`]'s concurrent-admission check (answers "does the pool
    /// have room for this request ALONGSIDE what's already admitted"). The
    /// native/ratchet clamps below are deliberately NOT applied to the second
    /// value: they protect single-request quality/correctness, not shared
    /// physical capacity, and conflating them would let the pool-capacity
    /// check under-count real headroom on any model whose native context is
    /// much smaller than what its KV pool can physically hold (see
    /// the project's internal engineering log's Phi-4 case study).
    fn compute_max_prompt_tokens(
        model_id: &str,
        model_dir: &std::path::Path,
        kv_gb: f64,
        kv_cache_precision: &str,
    ) -> (usize, usize) {
        if kv_gb == 0.0 {
            return (0, 0); // media model: no KV pool — gate disabled, no warn
        }
        let bytes_f16 = template::read_kv_bytes_per_token_f16(model_dir);
        let precision_bytes: usize = match kv_cache_precision {
            "u8" => 1,
            "f32" => 4,
            _ => 2, // f16 plugin default
        };
        // bytes_f16 already encodes the ×2 f16 factor; adjust to actual precision.
        let bytes_per_token = if bytes_f16 > 0 {
            bytes_f16 * precision_bytes / 2
        } else {
            0
        };
        let formula_result = if bytes_per_token > 0 {
            // bytes_per_token < 1 MB; kv_gb < 100; result < 1M tokens —
            // all values fit in f64 and the result fits in usize.
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss
            )]
            {
                (kv_gb * 1_073_741_824.0 / bytes_per_token as f64 * Self::FORMULA_SAFETY_MARGIN)
                    as usize
            }
        } else {
            tracing::warn!(
                model_id,
                "config.json missing or unparseable — L0 prompt-length gate disabled"
            );
            return (0, 0); // gate disabled — no ratchet applies to "disabled"
        };
        // The VRAM-derived formula above has no idea what the model was
        // actually trained on — a KV pool can easily afford more tokens than
        // the model's own `max_position_embeddings` (confirmed live
        // 2026-08-23, `dev/autotest/20260823_max_position_embeddings_gate_gap.md`:
        // qwen3-4b-int4-ov's formula promised 142k-306k tokens depending on
        // pool size, against a real trained ceiling of 40,960). Clamp to it
        // when known and tighter — unmargined, since `FORMULA_SAFETY_MARGIN`
        // corrects an *estimate* and `max_position_embeddings` is a hard
        // model property, not one.
        let native_clamped = match template::read_native_context_limit(model_dir) {
            Some(native) if native > 0 && native < formula_result => {
                tracing::info!(
                    model_id,
                    formula_result,
                    native_context_limit = native,
                    "L0 prompt-length gate: clamping to the model's native context limit"
                );
                native
            }
            _ => formula_result,
        };
        let max_prompt_tokens = match template::read_kv_capacity_ratchet(model_dir, kv_gb) {
            Some(ratchet) if ratchet > 0 && ratchet < native_clamped => {
                tracing::info!(
                    model_id,
                    formula_result = native_clamped,
                    ratchet,
                    "L0 prompt-length gate: applying learned KV-capacity ratchet \
                     (tighter than the formula estimate)"
                );
                ratchet
            }
            _ => native_clamped,
        };
        (max_prompt_tokens, formula_result)
    }

    /// Absolute path of this model's `chat_template` override, if configured.
    ///
    /// A relative path resolves against the **config file's** own directory, so
    /// a repo-shipped template (e.g. `templates/lfm2.jinja`) can be referenced
    /// portably across fleet boxes whose checkouts live at different paths.
    /// `None` when no override is set, which leaves template loading exactly as
    /// it was.
    fn resolve_template_override(&self, model_id: &str) -> Option<std::path::PathBuf> {
        let rel = self
            .config
            .models
            .get(model_id)
            .and_then(|e| e.policy.chat_template.as_ref())?;
        if rel.is_absolute() {
            return Some(rel.clone());
        }
        Some(
            self.config_path
                .as_ref()
                .and_then(|c| c.parent())
                .map_or_else(|| rel.clone(), |dir| dir.join(rel)),
        )
    }

    async fn execute_load(&self, model_id: &str, kv_gb: f64, max_num_seqs: usize) -> Result<()> {
        let model_path = self.config.models_dir.join(model_id);
        // Phase C1: load on the model's resolved device (per-model `device`
        // override → else `config.device`), set once at registration. Pre-C1
        // every record's `device` was `config.device`, so a single-device box is
        // unchanged.
        let device = self.record_device(model_id);
        let factory = Arc::clone(&self.factory);

        // Measures exactly the engine-construction/JIT-compile window — not
        // admission-check or HTTP overhead — for `rustedvino_model_load_duration_seconds`.
        let load_started = Instant::now();
        let result = tokio::task::spawn_blocking(move || {
            factory.load(&model_path, &device, kv_gb, max_num_seqs)
        })
        .await
        .map_err(|e| anyhow::anyhow!("spawn_blocking panicked: {e}"))?;
        let load_duration_secs = load_started.elapsed().as_secs_f64();

        match result {
            Ok((handle, thread)) => {
                // Get kind first — gates template loading and L0 warn suppression.
                let kind = handle.kind();

                // Chat template: only meaningful for text/vision generation.
                // Media models (embedding, STT, TTS, image) don't ship
                // tokenizer_config.json — suppress the load and the warn.
                let template_override = self.resolve_template_override(model_id);
                let template = if matches!(kind, ModelKind::TextGen | ModelKind::Vision) {
                    template::load_chat_template_with_override(
                        &self.config.models_dir.join(model_id),
                        template_override.as_deref(),
                    )
                        .unwrap_or_else(|e| {
                            tracing::warn!(
                                model_id,
                                error = %e,
                                "no chat template found — using generic fallback (tool calling will be degraded)"
                            );
                            template::FALLBACK_TEMPLATE.to_owned()
                        })
                } else {
                    template::FALLBACK_TEMPLATE.to_owned()
                };
                let family = ModelFamily::detect(&template);
                let (eos_token, bos_token) =
                    template::load_special_tokens(&self.config.models_dir.join(model_id));
                // Model-shipped sampling defaults: same TextGen/Vision-only gate
                // as the chat template above — media/embedding models don't ship
                // a meaningful generation_config.json for this purpose.
                let generation_defaults = if matches!(kind, ModelKind::TextGen | ModelKind::Vision)
                {
                    template::load_generation_defaults(&self.config.models_dir.join(model_id))
                } else {
                    template::GenerationDefaults::default()
                };

                let mut guard = self.write_models("execute_load:commit");
                let record = guard
                    .get_mut(model_id)
                    .ok_or_else(|| anyhow::anyhow!("model record disappeared during load"))?;
                // L0 gate: max tokens before the prompt-length gate rejects, plus
                // the unclamped pool-capacity figure the concurrent-admission
                // check uses (see compute_max_prompt_tokens's doc comment).
                // Extracted to keep execute_load under the 100-line clippy cap.
                let (max_prompt_tokens, pool_capacity_tokens) = Self::compute_max_prompt_tokens(
                    model_id,
                    &self.config.models_dir.join(model_id),
                    kv_gb,
                    &self.config.kv_cache_precision,
                );
                record.last_kind = Some(kind);
                record.last_load_duration_secs = Some(load_duration_secs);
                record.handle = Some(handle);
                record.thread = Some(thread);
                record.template = Arc::from(template);
                record.family = family;
                // Auto-detect reasoning parser from the template when not set
                // explicitly in config (gap-2). A Qwen3-style template includes
                // `enable_thinking`; that marker is the only reliable signal we
                // have without probing the model. Non-thinking models and those
                // with an explicit policy override are untouched.
                if record.reasoning_parser.is_none()
                    && template::template_hints_thinking(&record.template)
                {
                    record.reasoning_parser = Some(config::ReasoningParser::Qwen3);
                    tracing::info!(
                        model_id,
                        "reasoning_parser auto-detected: qwen3 (template contains enable_thinking)"
                    );
                }
                record.eos_token = Arc::from(eos_token.as_str());
                record.bos_token = Arc::from(bos_token.as_str());
                record.generation_defaults = generation_defaults;
                record.kv_cache_gb = kv_gb;
                record.max_prompt_tokens = max_prompt_tokens;
                record.pool_capacity_tokens = pool_capacity_tokens;
                record.max_concurrent_streams = max_num_seqs;
                record.state = ModelState::Ready;
                record.last_used = Instant::now();
                tracing::info!(
                    model_id,
                    kind = kind.label(),
                    ?family,
                    kv_gb,
                    max_prompt_tokens,
                    pool_capacity_tokens,
                    load_duration_secs,
                    "model loaded"
                );
                Ok(())
            }
            Err(e) => {
                // L3: detect GPU context poisoning. Once set, gpu_poisoned blocks
                // all future load attempts with a 503 until a process restart.
                if is_gpu_poison_error(&e.to_string()) {
                    self.mark_gpu_poisoned(model_id, &e);
                }

                // Defence-in-depth: free VRAM allocation on engine failure.
                // `abort_load` in `load_model` also does this; VramTracker::free
                // is a no-op if the model was not allocated, so the double-call is safe.
                if let Ok(mut v) = self.vram.write() {
                    v.free(model_id);
                }
                Err(e)
            }
        }
    }

    /// Error path for `load_model`: reset state to `NotLoaded` and release any
    /// VRAM reservation that was made by `ensure_vram_for`. Idempotent — also
    /// run by [`LoadCancelGuard`] when the load future is dropped mid-await.
    /// Live system-RAM availability for the M1 admission gate, in GiB.
    ///
    /// `None` for any domain other than [`DOMAIN_SYSTEM`] (a discrete GPU's
    /// private VRAM is not reflected in `MemAvailable`, so the live gate must
    /// not run there) and `None` when `/proc/meminfo` is unreadable, which
    /// degrades to logical-only gating exactly as before.
    ///
    /// When `system_ram_budget_gb` is unset this is plain
    /// [`os_memory::available_ram_gb`] — byte-identical to the historical
    /// behaviour. When the operator has declared a budget, the reading is
    /// corrected by the reclaimable GPU page cache that `MemAvailable` omits
    /// (see [`os_memory::reclaimable_gpu_cache_gb`]) and then capped at the
    /// declared budget, so the server never plans against more system RAM than
    /// it was granted even if the machine happens to have more free.
    fn live_system_avail_gb(&self, domain: &str) -> Option<f64> {
        if domain != DOMAIN_SYSTEM {
            return None;
        }
        let avail = os_memory::available_ram_gb()?;
        let Some(budget) = self.config.system_ram_budget_gb else {
            return Some(avail);
        };
        // The raw credit is "memory no standard counter explains", which
        // includes the GPU pages backing models this server has ALREADY
        // loaded — those are live, not pooled, and crediting them would let
        // the gate admit a second copy of memory already in use. Subtract our
        // own tracked system-domain usage so only genuinely-freed pool memory
        // is credited. Measured why this matters: with a 12 GB model resident,
        // the uncorrected credit read 13.18 GB.
        let tracked_used = self
            .vram
            .read()
            .map_or(0.0, |v| v.domain_gb(DOMAIN_SYSTEM).1);
        let credit = (os_memory::reclaimable_gpu_cache_gb().unwrap_or(0.0) - tracked_used).max(0.0);
        Some((avail + credit).min(budget))
    }

    fn abort_load(&self, model_id: &str) {
        if let Ok(mut guard) = self.models.write()
            && let Some(r) = guard
                .get_mut(model_id)
                .filter(|r| r.state == ModelState::Loading)
        {
            r.state = ModelState::NotLoaded;
        }
        // Always attempt to free — VramTracker::free is a no-op if the model
        // was never allocated (e.g. failure before ensure_vram_for ran).
        if let Ok(mut v) = self.vram.write() {
            v.free(model_id);
        }
    }

    /// Dry-run admission check ("what fits") — co-residency Slice 1.
    ///
    /// Answers "could `model_id` load right now, and what would it evict?"
    /// **without touching the GPU or mutating any state.** Mirrors
    /// [`ensure_vram_for`](Self::ensure_vram_for)'s arithmetic
    /// (`needed = weights + min_kv + margin`) and
    /// [`select_eviction_victim`](Self::select_eviction_victim)'s ordering
    /// (non-pinned Ready candidates, `(busy, priority, last_used)` ascending),
    /// then greedily simulates eviction until the load fits. This is the
    /// admin-facing promise over the existing deterministic engine — no GPU work.
    ///
    /// The two locks are taken **sequentially, never nested** (models snapshot
    /// released before the vram read), so this cannot deadlock against a
    /// concurrent `ensure_vram_for` (which holds vram, then models).
    ///
    /// # Errors
    /// [`ModelError::NotFound`] if `model_id` is not a registered model.
    pub fn check_admission(&self, model_id: &str) -> Result<AdmissionCheck> {
        // Model weight + already-resident check + its domain, under the models
        // lock only. Phase C1: the dry-run mirrors `ensure_vram_for` — it must
        // answer against the *target model's* domain, not the primary one.
        let (weight_gb, already_loaded, domain, needs_kv_pool) = {
            let guard = self.read_models("check_admission");
            let record = guard
                .get(model_id)
                .ok_or_else(|| ModelError::NotFound(model_id.to_owned()))?;
            (
                record.vram_gb,
                record.state == ModelState::Ready,
                record.domain.clone(),
                // Mirror `load_model_with_overrides`'s (now-fixed) needs_kv_pool
                // logic so the dry-run admission prediction matches the real
                // load path — read the record's resolved `configured_kind`,
                // not the raw config field alone (see the fix note there).
                matches!(
                    record.configured_kind,
                    ModelKind::TextGen | ModelKind::Vision
                ),
            )
        };
        let needed_gb = weight_gb
            + if needs_kv_pool {
                self.config.min_kv_cache_gb
            } else {
                0.0
            }
            + self.config.vram_safety_margin_gb;

        // Second gate, UMA only — see `RamGate`.
        let ram_gate = self.ram_gate_for(model_id, &domain, weight_gb, needs_kv_pool);
        let (ram_needed_gb, ram_avail_gb) = (ram_gate.needed, ram_gate.avail);
        let ram_admits = |freed: f64| ram_gate.admits(freed);

        // Already Ready → resident now; the check is moot. Report it fits with no
        // eviction so callers get a clean "yes, it's here" answer.
        if already_loaded {
            let free_gb = self
                .vram
                .read()
                .map_err(|_| anyhow::anyhow!("vram lock poisoned"))?
                .free_gb(&domain);
            return Ok(AdmissionCheck {
                fits: true,
                needed_gb,
                free_gb,
                would_evict: Vec::new(),
                ram_needed_gb,
                ram_avail_gb,
                already_loaded: true,
            });
        }

        // Snapshot evictable Ready candidates in eviction order — the SAME
        // ordering select_eviction_victim uses. Taken under the models lock and
        // released here, before the vram lock is acquired (no nested hold).
        let candidates: Vec<String> = {
            let guard = self.read_models("check_admission:candidates");
            let mut ranked: Vec<((bool, i32, Instant), String)> = guard
                .iter()
                .filter(|(id, r)| {
                    id.as_str() != model_id
                        && r.state == ModelState::Ready
                        && r.domain == domain
                        && !self.eviction_protected(id, r, false)
                })
                .map(|(id, r)| (r.eviction_sort_key(), id.clone()))
                .collect();
            ranked.sort_by_key(|(key, _)| *key);
            ranked.into_iter().map(|(_, id)| id).collect()
        };

        let vram = self
            .vram
            .read()
            .map_err(|_| anyhow::anyhow!("vram lock poisoned"))?;
        let free_gb = vram.free_gb(&domain);

        // Fits as-is (covers gating-disabled / unknown domains, where fits() is
        // always true) → no eviction needed.
        if vram.fits(&domain, needed_gb) && ram_admits(0.0) {
            return Ok(AdmissionCheck {
                fits: true,
                needed_gb,
                free_gb,
                would_evict: Vec::new(),
                ram_needed_gb,
                ram_avail_gb,
                already_loaded: false,
            });
        }

        // Greedily evict candidates in order until BOTH gates are satisfied (or
        // we run out). Eviction frees real RAM as well as tracked VRAM — the
        // reason `ensure_vram_for` re-reads `live_system_avail_gb` on every
        // retry — so the same reclaimed total is credited to both.
        let mut freed = 0.0;
        let mut would_evict = Vec::new();
        for id in candidates {
            if free_gb + freed >= needed_gb && ram_admits(freed) {
                break;
            }
            freed += vram.allocated_gb(&id);
            would_evict.push(id);
        }

        Ok(AdmissionCheck {
            fits: free_gb + freed >= needed_gb && ram_admits(freed),
            needed_gb,
            free_gb,
            would_evict,
            ram_needed_gb,
            ram_avail_gb,
            already_loaded: false,
        })
    }

    /// Evict LRU models until `weight_gb + min_kv + safety_margin` of VRAM is
    /// free, then atomically compute and reserve `weight_gb + kv_gb`.
    ///
    /// Returns the computed `kv_gb` that was reserved.  The reservation covers
    /// the full allocation (weights + KV pool) so the VRAM tracker accurately
    /// reflects true GPU usage and prevents concurrent loads from both believing
    /// they fit.
    #[allow(clippy::too_many_lines)] // cohesive check/reserve/evict loop + M1 RAM gate
    async fn ensure_vram_for(
        &self,
        model_id: &str,
        weight_gb: f64,
        kv_target_gb: f64,
        needs_kv_pool: bool,
        force: bool,
    ) -> Result<f64> {
        // STT / embedding / image / TTS models have no KV cache; charge them
        // weight only. LLM/VLM (or unknown kind) get the min_kv floor so the
        // error message is honest about what they require.
        let min_kv = if needs_kv_pool {
            self.config.min_kv_cache_gb
        } else {
            0.0
        };
        let margin = self.config.vram_safety_margin_gb;
        // Minimum total VRAM we need visible before we can proceed.
        let total_minimum = weight_gb + min_kv + margin;
        // Phase C1: charge this model's *own* memory domain (its device's pool),
        // not always the primary inference domain. Pre-C1 every model resolved to
        // `inference_domain`, so a single-device box is unchanged.
        let domain = self.record_domain(model_id);

        // M1: live system-RAM admission floor (shared UMA "system" domain only).
        // The logical VRAM tracker cannot see real RSS, `OpenVINO` runtime/compile
        // overhead, or memory held by OTHER processes — on a UMA box a load the
        // accounting "fits" can still OOM the machine. Gate on a live MemAvailable
        // read against the OS reservation, using a CONSERVATIVE need (weights +
        // full KV target + margin, not just min_kv). Re-read each iteration so an
        // eviction — which frees real RAM — can rescue a retry. Discrete-GPU
        // domains are not gated (their private VRAM isn't reflected in MemAvailable).
        let ram_floor = self
            .config
            .system_ram_reservation_gb
            .unwrap_or_else(os_memory::default_reservation_gb);
        // Conservative KV estimate for the live gate: an LLM passes kv_target=0
        // ("grab-all"), so the pool will expand up to `cache_size_gb` (see
        // `compute_kv_pool_gb`). Count that cap, not the 0 target — otherwise the
        // gate undercounts an LLM's multi-GB KV pool (the original OOM's 4 GB).
        // Extracted to `conservative_kv_need_gb` so `check_admission`'s dry-run
        // charges the identical quantity — these two were computed separately
        // and diverged.
        let kv_need = if needs_kv_pool {
            Self::conservative_kv_need_gb(kv_target_gb, &self.config)
        } else {
            0.0
        };
        let live_need = weight_gb + kv_need + margin;

        // Upfront feasibility check — before evicting anything, verify this
        // load could ever fit even after freeing every evictable resident in
        // this domain. Without this, a genuinely oversized request destructively
        // evicts every co-resident model one at a time (real service disruption
        // for their in-flight clients) only to fail anyway once the last
        // evictable victim is gone — the eviction was doomed from the start. See
        // dev/autotest/20260804_doomed_load_evicts_everything_first.md.
        {
            let reclaimable_gb = self.max_reclaimable_gb(model_id, &domain, force);
            // `free_gb()` reports `0.0` for a gating-disabled (`total_gb == 0.0`)
            // or never-registered domain — unlike `fits()`, it has no "ungated"
            // special case, so it must not be compared against `total_minimum`
            // for such a domain. Detect that case the same way `fits()` does
            // internally: an infinite request always "fits" an ungated domain.
            let (free_gb, gating_disabled) = {
                let vram = self
                    .vram
                    .read()
                    .map_err(|_| anyhow::anyhow!("vram lock poisoned"))?;
                (vram.free_gb(&domain), vram.fits(&domain, f64::INFINITY))
            };
            let max_capacity = free_gb + reclaimable_gb;

            let live_avail = self.live_system_avail_gb(&domain);
            if let Some(avail) = live_avail {
                let max_avail = avail + reclaimable_gb;
                if !os_memory::ram_floor_admits(max_avail, live_need, ram_floor) {
                    return Err(anyhow::anyhow!(
                        "insufficient system memory: loading {model_id} needs ~{live_need:.1} GB \
                         (weights {weight_gb:.1} + KV {kv_need:.1} + margin {margin:.1}), but even \
                         evicting every evictable model in this domain would leave only \
                         ~{max_avail:.1} GB system RAM available and {ram_floor:.1} GB must stay \
                         free for the OS — this load can never fit, nothing was evicted"
                    ));
                }
            }

            if !gating_disabled && max_capacity < total_minimum {
                let tail = if reclaimable_gb <= 0.0 {
                    self.eviction_blocked_tail(model_id, &domain, force)
                } else {
                    "not enough even after evicting every evictable model in this domain — \
                     free more VRAM or load a smaller model"
                };
                return Err(anyhow::anyhow!(
                    "insufficient VRAM: need {total_minimum:.1} GB \
                     (weights {weight_gb:.1} + min KV {min_kv:.1} + margin {margin:.1}), at most \
                     {max_capacity:.1} GB could ever be freed in this domain, {tail} — this load \
                     can never fit, nothing was evicted"
                ));
            }
        }

        loop {
            // Live system-RAM check: `None` unless this is the gated system domain
            // with a readable /proc/meminfo (else degrade to logical-only gating,
            // matching how an unreadable system budget is left ungated).
            let live_avail = self.live_system_avail_gb(&domain);
            let ram_ok =
                live_avail.is_none_or(|a| os_memory::ram_floor_admits(a, live_need, ram_floor));

            // One write-lock acquisition per iteration: check, compute, and
            // allocate atomically — no TOCTOU window for a concurrent load.
            let kv_result = {
                let mut vram = self
                    .vram
                    .write()
                    .map_err(|_| anyhow::anyhow!("vram lock poisoned"))?;

                if ram_ok && vram.fits(&domain, total_minimum) {
                    if needs_kv_pool {
                        // Temporarily allocate just the weight to measure remaining VRAM.
                        vram.allocate(&domain, model_id, weight_gb);
                        // T5.4 seam: the KV-size policy lives in `compute_kv_pool_gb`
                        // — the single place a future bounded/co-residency policy
                        // changes, without touching this reserve/evict loop.
                        let kv_final = Self::compute_kv_pool_gb(
                            vram.free_gb(&domain),
                            kv_target_gb,
                            &self.config,
                        );
                        // Expand the reservation to cover the full allocation.
                        vram.allocate(&domain, model_id, weight_gb + kv_final);
                        Some(kv_final)
                    } else {
                        // No KV pool — STT / embedding / image / TTS reserve weight only.
                        vram.allocate(&domain, model_id, weight_gb);
                        Some(0.0)
                    }
                } else {
                    None
                }
            };

            if let Some(kv_gb) = kv_result {
                tracing::debug!(
                    model_id,
                    weight_gb,
                    kv_gb,
                    "VRAM reservation: weight + KV pool"
                );
                return Ok(kv_gb);
            }

            // Not enough VRAM — pick a victim (T5.4 seam: `select_eviction_victim`)
            // in the *same domain* and evict it. `None` → nothing evictable in
            // this domain → the hard error below.
            let lru_id = self.select_eviction_victim(model_id, &domain, force);

            match lru_id {
                None => {
                    // M1: if the LIVE system-RAM floor is the binding constraint
                    // (not the logical budget), name it — and nothing evictable is
                    // left to free, so report honestly instead of OOMing.
                    if let Some(avail) = live_avail
                        && !ram_ok
                    {
                        return Err(anyhow::anyhow!(
                            "insufficient system memory: loading {model_id} needs ~{live_need:.1} GB \
                             (weights {weight_gb:.1} + KV {kv_need:.1} + margin {margin:.1}), but only \
                             {avail:.1} GB system RAM is available and {ram_floor:.1} GB must stay free \
                             for the OS — free memory or load a smaller model"
                        ));
                    }
                    let free = self
                        .vram
                        .read()
                        .map_err(|_| anyhow::anyhow!("vram lock poisoned"))?
                        .free_gb(&domain);
                    let tail = self.eviction_blocked_tail(model_id, &domain, force);
                    return Err(anyhow::anyhow!(
                        "insufficient VRAM: need {total_minimum:.1} GB \
                         (weights {weight_gb:.1} + min KV {min_kv:.1} + margin {margin:.1}), \
                         only {free:.1} GB free, {tail}"
                    ));
                }
                Some(id) => {
                    tracing::info!(evicting = %id, loading = model_id, "LRU eviction for VRAM");
                    self.evict_model(&id).await?;
                }
            }
        }
    }

    /// True when `id`/`r` must **not** be chosen as an eviction victim.
    /// Four independent hard excludes, any one is sufficient:
    /// - operator-`pinned` or declared non-`evictable` (co-residency Slices 1/3,
    ///   the original two checks — always in force, `ignore_grace` never
    ///   touches these);
    /// - **D2 — realtime serving set:** `id` is in the resolved `{stt, llm,
    ///   tts}` set of at least one active realtime session
    ///   (`realtime_serving_set_contains`). Never bypassable by `force`
    ///   (the project's internal engineering log open question 3,
    ///   decided: `force` bypasses grace only).
    /// - **D3 — eviction grace window:** `id` was used (any channel — admin
    ///   load, plain API, realtime turn; `last_used` is bumped by all of
    ///   them identically) more recently than its grace window. Skipped when
    ///   `ignore_grace` is `true` — the admin load endpoint's `force: true`
    ///   escape hatch (D3), and *only* that escape hatch; every other caller
    ///   passes `false`.
    ///
    /// This is the single predicate every victim-selection path
    /// (`select_eviction_victim`, `max_reclaimable_gb`, `eviction_blocked_tail`,
    /// the `check_admission` dry-run) filters on, so they stay in lockstep —
    /// the dry-run preview can never disagree with what actually gets evicted.
    fn eviction_protected(&self, id: &str, r: &ModelRecord, ignore_grace: bool) -> bool {
        r.pinned
            || !r.evictable
            || self.realtime_serving_set_contains(id)
            || (!ignore_grace && r.last_used.elapsed() < self.eviction_grace_for(id))
    }

    /// D2: `true` when `id` is in the resolved `{stt, llm, tts}` set of at
    /// least one active realtime session. `false` in mock/test mode (no
    /// `voice_pin` wired) — matches every other voice-pin-gated behavior's
    /// "absent ⇒ inert" convention.
    fn realtime_serving_set_contains(&self, id: &str) -> bool {
        self.voice_pin
            .as_ref()
            .is_some_and(|vp| vp.serving_count(id) > 0)
    }

    /// D3: the eviction-grace window for `id` — the per-model
    /// `ModelPolicy::eviction_grace_secs` override if set, else the global
    /// `Config::eviction_grace_secs`. Both are validated finite/non-negative
    /// at config load (`config::validate_model_entry`, `Config::validate`),
    /// so this never needs to guard against a bad value here.
    fn eviction_grace_for(&self, id: &str) -> Duration {
        let secs = self
            .effective_entry(id)
            .and_then(|e| e.policy.eviction_grace_secs)
            .unwrap_or(self.config.eviction_grace_secs);
        Duration::from_secs_f64(secs.max(0.0))
    }

    /// Choose the best model to evict to free VRAM for `loading_id`, or `None`
    /// when no Ready model (other than the one loading) can be freed.
    ///
    /// **Phase C1 — domain-scoped.** Only Ready models in the *same memory
    /// `domain`* as the loading model are candidates: evicting a model from one
    /// GPU's pool frees nothing in another, so a cross-domain resident is never a
    /// victim. Pre-C1 every model shared the single inference domain, so this is
    /// a no-op on a single-device box.
    ///
    /// **T5.4 seam — eviction policy.** The single home for victim *quality*.
    /// Today: in-flight-aware LRU (#37). `last_used` is bumped only at request
    /// *admission*, so a model that admitted a long 2000-token stream 30s ago
    /// has an *older* timestamp than one that served a 1-token ping 1s ago and
    /// is now idle — naive `last_used` LRU would evict the actively-streaming
    /// model out from under its clients. We sort Ready candidates by
    /// `(busy, last_used)`: a busy model (in-flight > 0, read live through the
    /// `ManagedEngine` facade) is weighted LAST, so a genuinely idle model is
    /// always preferred. Busy models are weighted, not excluded — if *every*
    /// candidate is busy we still pick one (oldest) so the load makes progress.
    ///
    /// Co-residency Slices 1/3 + realtime arbitration v2 (D1-D3): an
    /// eviction-*protected* model (`eviction_protected`) is hard-excluded here,
    /// and `priority` is a secondary sort key — among the remaining candidates
    /// the eviction order is `(busy, priority, last_used)`: an in-flight model
    /// is the last resort, then the lowest-priority model, then the
    /// least-recently-used. Protection is a *hard* exclude; busy-weighting and
    /// priority are *soft* — if every candidate is busy (or low-priority) we
    /// still pick one so the load makes progress. If the only Ready models are
    /// protected, this returns `None` and the load fails in [`ensure_vram_for`]
    /// rather than evicting a protected model. `ignore_grace` is threaded from
    /// `ensure_vram_for`'s `force` parameter (D3's admin-only escape hatch).
    fn select_eviction_victim(
        &self,
        loading_id: &str,
        domain: &str,
        ignore_grace: bool,
    ) -> Option<String> {
        let guard = self.read_models("ensure_vram:victim");
        guard
            .iter()
            .filter(|(id, r)| {
                id.as_str() != loading_id
                    && r.state == ModelState::Ready
                    && r.domain == domain
                    && !self.eviction_protected(id, r, ignore_grace)
            })
            .min_by_key(|(_, r)| r.eviction_sort_key())
            .map(|(id, _)| id.clone())
    }

    /// Sum of every evictable Ready model's VRAM reservation in `domain` — the
    /// same candidate set [`select_eviction_victim`](Self::select_eviction_victim)
    /// draws from, but totalled rather than picked one at a time. This is the
    /// theoretical maximum a full eviction sweep of the domain could ever
    /// reclaim; used by [`ensure_vram_for`](Self::ensure_vram_for)'s upfront
    /// feasibility check to predict a doomed load without evicting anything.
    fn max_reclaimable_gb(&self, loading_id: &str, domain: &str, ignore_grace: bool) -> f64 {
        let candidates: Vec<String> = {
            let guard = self.read_models("ensure_vram:max_reclaimable");
            guard
                .iter()
                .filter(|(id, r)| {
                    id.as_str() != loading_id
                        && r.state == ModelState::Ready
                        && r.domain == domain
                        && !self.eviction_protected(id, r, ignore_grace)
                })
                .map(|(id, _)| id.clone())
                .collect()
        };
        let Ok(vram) = self.vram.read() else {
            return 0.0;
        };
        candidates.iter().map(|id| vram.allocated_gb(id)).sum()
    }

    /// Diagnostic tail for an insufficient-VRAM error: distinguishes "nothing
    /// else resident" from "everything else is protected" (pinned, non-evictable,
    /// in the realtime serving set, or within its eviction-grace window) — the
    /// latter is a policy/timing outcome, not an empty card, so name it. Scoped
    /// to `domain`: a protected resident in *another* domain cannot block this
    /// load. Shared between [`ensure_vram_for`](Self::ensure_vram_for)'s upfront
    /// feasibility check and its loop-exhausted fallback so both report
    /// identically.
    fn eviction_blocked_tail(
        &self,
        model_id: &str,
        domain: &str,
        ignore_grace: bool,
    ) -> &'static str {
        let guard = self.read_models("ensure_vram:pin_diag");
        let blocked_by_protected = guard.iter().any(|(id, r)| {
            r.state == ModelState::Ready
                && r.domain == domain
                && self.eviction_protected(id, r, ignore_grace)
        }) && !guard.iter().any(|(id, r)| {
            id.as_str() != model_id
                && r.state == ModelState::Ready
                && r.domain == domain
                && !self.eviction_protected(id, r, ignore_grace)
        });
        if blocked_by_protected {
            "only protected (pinned / non-evictable / in active realtime use / \
             within its eviction-grace window) models resident — unpin, mark \
             evictable, wait out the grace window, or free VRAM"
        } else {
            "no evictable model found"
        }
    }

    /// Compute the KV-cache pool size (GB) to reserve for a freshly-loaded model.
    ///
    /// **T5.4 seam — KV-size policy.** The single place that decides how much
    /// Builds this model's live system-RAM gate — the second of the two
    /// conditions [`ensure_vram_for`](Self::ensure_vram_for) admits on.
    fn ram_gate_for(
        &self,
        model_id: &str,
        domain: &str,
        weight_gb: f64,
        needs_kv_pool: bool,
    ) -> RamGate {
        let avail_gb = self.live_system_avail_gb(domain);
        let needed_gb = avail_gb.map(|_| {
            let kv_need = if needs_kv_pool {
                Self::conservative_kv_need_gb(self.resolved_kv_target_gb(model_id), &self.config)
            } else {
                0.0
            };
            weight_gb + kv_need + self.config.vram_safety_margin_gb
        });
        RamGate {
            avail: avail_gb,
            needed: needed_gb,
            floor: self
                .config
                .system_ram_reservation_gb
                .unwrap_or_else(os_memory::default_reservation_gb),
        }
    }

    /// The KV pool a load will *conservatively* be charged for by the live
    /// system-RAM gate: the resolved per-model target, capped by the global
    /// `cache_size_gb`, floored at `min_kv_cache_gb`.
    ///
    /// An LLM with no explicit target passes `kv_target_gb == 0.0` meaning
    /// "grab all remaining", so the pool expands to `cache_size_gb`; counting
    /// the literal `0.0` would undercount a multi-GB pool (the original OOM's
    /// 4 GB). When the target is unbounded *and* `cache_size_gb` is unset the
    /// result is genuinely unpredictable, so it degrades to the floor.
    ///
    /// Shared by [`ensure_vram_for`](Self::ensure_vram_for) (the real gate) and
    /// [`check_admission`](Self::check_admission) (the dry-run) so the two
    /// cannot drift — they did, and a model sized between the floor and its real
    /// target passed the check then failed the load.
    fn conservative_kv_need_gb(kv_target_gb: f64, config: &Config) -> f64 {
        let target = if kv_target_gb > 0.0 {
            kv_target_gb
        } else {
            f64::INFINITY
        };
        let cap = if config.cache_size_gb > 0.0 {
            config.cache_size_gb
        } else {
            f64::INFINITY
        };
        let bound = target.min(cap);
        if bound.is_finite() {
            bound.max(config.min_kv_cache_gb)
        } else {
            config.min_kv_cache_gb // fully uncapped grab-all -> can't predict
        }
    }

    /// The KV target a load of `model_id` would resolve to with no request-level
    /// override: per-model policy, else `default_kv_cache_gb`. `0.0` means
    /// unbounded. Used by the dry-run, which by definition has no overrides.
    fn resolved_kv_target_gb(&self, model_id: &str) -> f64 {
        let entry = self.effective_entry(model_id);
        resolve_runtime_params(&self.config, entry.as_ref(), &LoadOverrides::default()).kv_cache_gb
    }

    /// VRAM a model claims for its KV pool at load time. Today's policy: take all
    /// VRAM left after the weights (`free_after_weight_gb`), minus the safety
    /// margin, floored at `min_kv_cache_gb` and capped by `cache_size_gb`
    /// (`0` = uncapped — the historical "grab all remaining" behaviour).
    ///
    /// Isolating it here means a future co-residency policy (a bounded per-model
    /// reservation, or load-time sizing that reserves headroom for other
    /// registered models so they can co-reside) is a change to *this function
    /// body alone* — the reserve/evict loop in [`ensure_vram_for`] is unaffected.
    ///
    /// **Co-residency Slice 2:** `kv_target_gb` is the resolved per-model KV pool
    /// target (from [`resolve_runtime_params`] — load override → policy → global
    /// `default_kv_cache_gb`). `0.0` means *unbounded* (grab all remaining, the
    /// historical single-model behaviour); when `> 0.0` the pool is clamped to
    /// it so models co-reside. The global `cache_size_gb` cap and the
    /// `min_kv_cache_gb` floor still apply, so an absent target + unset
    /// `cache_size_gb` reproduces the pre-feature result byte-for-byte.
    fn compute_kv_pool_gb(free_after_weight_gb: f64, kv_target_gb: f64, config: &Config) -> f64 {
        // Grab-all baseline: everything left after the safety margin.
        let mut kv = free_after_weight_gb - config.vram_safety_margin_gb;
        // Bounded reservation: clamp to the resolved per-model target (0.0 = off).
        if kv_target_gb > 0.0 {
            kv = kv.min(kv_target_gb);
        }
        // Global hard cap still applies on top of the target (0.0 = uncapped).
        if config.cache_size_gb > 0.0 {
            kv = kv.min(config.cache_size_gb);
        }
        // Never go below the floor — ensure_vram_for guarantees this much is free.
        kv.max(config.min_kv_cache_gb)
    }

    /// Phase 1 of `evict_model`: validate state, transition to `Evicting`,
    /// extract the handle and thread.
    fn begin_evict(
        &self,
        model_id: &str,
    ) -> Result<(EngineHandleKind, std::thread::JoinHandle<()>)> {
        // L3 gate, eviction side (2026-08-04, dev/autotest/20260804_gpu_poisoned
        // _eviction_crash.md): refuse to evict *anything* while the GPU context
        // is poisoned. Dropping a resident pipeline runs its C++ destructor,
        // which can itself make an OpenCL call (e.g. clFinish) on the already
        // -broken context — a second exception thrown during that cleanup
        // escalates to std::terminate() and crashes the whole worker process,
        // not just the one poisoned model. `begin_load`'s L3 gate already
        // assumes poisoned models stay resident until a manual restart
        // ("existing loaded models may still serve requests" — see
        // `ModelError::GpuPoisoned`'s doc comment); this enforces that instead
        // of leaving it as an unguarded assumption.
        if self.gpu_poisoned.load(Ordering::Acquire) {
            return Err(ModelError::GpuPoisoned.into());
        }

        let mut guard = self.write_models("begin_evict");
        let record = guard
            .get_mut(model_id)
            .ok_or_else(|| ModelError::NotFound(model_id.to_owned()))?;

        match record.state {
            ModelState::Ready => {}
            ModelState::NotLoaded => return Err(anyhow::anyhow!(ModelError::NotLoaded)),
            ModelState::Loading => return Err(ModelError::Loading.into()),
            ModelState::Evicting => return Err(ModelError::Evicting.into()),
        }

        record.state = ModelState::Evicting;

        let handle = record
            .handle
            .take()
            .ok_or_else(|| anyhow::anyhow!("model marked Ready but handle is missing"))?;
        let thread = record
            .thread
            .take()
            .ok_or_else(|| anyhow::anyhow!("model marked Ready but thread is missing"))?;

        Ok((handle, thread))
    }

    /// Phase 2 of `evict_model` (post-join): update state and free VRAM.
    ///
    /// Frees the full reservation (weight + KV pool) that `ensure_vram_for`
    /// allocated — `VramTracker::free` removes the whole entry regardless of size.
    fn finish_evict(&self, model_id: &str) {
        if let Ok(mut guard) = self.models.write()
            && let Some(r) = guard.get_mut(model_id)
        {
            r.state = ModelState::NotLoaded;
            r.kv_cache_gb = 0.0;
            r.max_prompt_tokens = 0;
        }
        // Always safe — no-op if the model was never allocated (e.g. vram_gb=0.0).
        if let Ok(mut v) = self.vram.write() {
            v.free(model_id);
        }
    }

    // ---- Lock helpers --------------------------------------------------
    //
    // These helpers centralise poison recovery so it only appears once in
    // the source. RwLock::read/write return PoisonError if a thread panicked
    // while holding the lock — in practice this only happens if there is a
    // bug in the model_manager code itself.
    //
    // T5.3 (committee findings 39, 40): these helpers used to `panic!` on a
    // poisoned lock. Because the `models` lock sits on the path of EVERY
    // request (chat, completions, tokenize, /v1/models, /metrics), a single
    // first-panic under the lock turned every subsequent request into a
    // permanent panic loop — a whole-node cascade amplifier from one bug.
    // We now recover the guard via `PoisonError::into_inner` instead (the
    // graceful pattern the `vram` lock already uses): the node keeps serving
    // from the recovered map rather than panic-looping. The records are plain
    // data, so a recovered map is memory-safe and at worst logically stale for
    // the one record a panicking writer was mid-mutation on. We log the
    // recovery at WARN so a poisoned lock stays loud — it is still a
    // should-never-happen bug the operator must investigate.

    fn read_models(&self, ctx: &str) -> RwLockReadGuard<'_, HashMap<String, ModelRecord>> {
        self.models.read().unwrap_or_else(|e| {
            tracing::warn!(ctx, "models RwLock poisoned — recovering guard (T5.3)");
            e.into_inner()
        })
    }

    fn write_models(&self, ctx: &str) -> RwLockWriteGuard<'_, HashMap<String, ModelRecord>> {
        self.models.write().unwrap_or_else(|e| {
            tracing::warn!(ctx, "models RwLock poisoned — recovering guard (T5.3)");
            e.into_inner()
        })
    }

    /// Phase C1: the `OpenVINO` device a model loads on, resolved at registration
    /// (per-model `device` override → else `config.device`). Falls back to
    /// `config.device` for an unregistered id (should not happen on the load
    /// path — every loadable model is registered from `vram_gb`).
    ///
    /// Also the cross-pipeline admission middleware's live device lookup
    /// (the project's internal engineering log step 2): handlers
    /// call this fresh on every request — never cache it on a long-lived
    /// handle/context — because an admin `reload_config` can move a model to
    /// a different device on its *next* load (`reload_config` only updates
    /// `NotLoaded` entries; see its doc comment).
    pub(crate) fn record_device(&self, model_id: &str) -> String {
        self.read_models("record_device")
            .get(model_id)
            .map_or_else(|| self.config.device.clone(), |r| r.device.clone())
    }

    /// A model's own shipped sampling defaults (`generation_config.json`),
    /// resolved at its last successful load. `Default::default()` (all
    /// `None`) for an unregistered id, a never-loaded model, or a
    /// media/embedding model — same fallback shape as [`record_device`]'s
    /// unregistered-id case. Lighter than [`get_chat_context`] — no
    /// `last_used` bump, no `Ready`-state requirement — for callers (like
    /// `/v1/completions`) that only need this one field, not the full chat
    /// routing context.
    pub(crate) fn model_generation_defaults(&self, model_id: &str) -> template::GenerationDefaults {
        self.read_models("model_generation_defaults")
            .get(model_id)
            .map_or_else(template::GenerationDefaults::default, |r| {
                r.generation_defaults
            })
    }

    /// Phase C1: the memory domain a model is charged to (resolved at
    /// registration from its device). `ensure_vram_for` admits and reserves into
    /// this domain, and `select_eviction_victim` only evicts within it. Falls
    /// back to the primary `inference_domain` for an unregistered id.
    fn record_domain(&self, model_id: &str) -> String {
        self.read_models("record_domain")
            .get(model_id)
            .map_or_else(|| self.inference_domain.clone(), |r| r.domain.clone())
    }
}

/// RAII cancellation guard for [`ModelManager::load_model`] (T5.1, committee
/// finding 5 — mirrors the shared [`crate::in_flight::InFlightGuard`] pattern).
///
/// An async fn's cleanup code only runs if the future keeps being polled; when
/// the HTTP client disconnects, axum drops the handler task and every future
/// in it — mid-await. Without this guard a dropped `load_model` left the
/// record in `Loading` forever: chat 503s, reload 409s, evict 409s, until a
/// process restart. `Drop` while armed runs `abort_load` (reset to
/// `NotLoaded` + release any VRAM reservation); the normal exits `disarm()`.
///
/// The engine-load itself runs detached on the blocking pool, so a cancelled
/// load may still briefly hold real GPU memory until the orphaned engine
/// drops — the *tracker* is reset immediately, which is what gates new loads.
struct LoadCancelGuard<'a> {
    mm: &'a ModelManager,
    model_id: &'a str,
    armed: bool,
}

impl LoadCancelGuard<'_> {
    /// Defuse the guard on a normal exit (success or handled error).
    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for LoadCancelGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            tracing::warn!(
                model_id = self.model_id,
                "load_model future dropped mid-load — resetting state (cancellation guard)"
            );
            self.mm.abort_load(self.model_id);
        }
    }
}

// ============================================================
// Unit tests
// ============================================================
#[cfg(test)]
mod load_cancel_tests {
    #![allow(clippy::unwrap_used)]

    use std::path::Path;
    use std::sync::Arc;
    use std::task::{Context, Poll, Waker};

    use super::*;
    use crate::model_manager::lifecycle::{EngineFactory, MockEngineFactory};

    /// A mock factory whose `load` blocks long enough that the first poll of
    /// `load_model` reliably parks at the `spawn_blocking` await.
    struct SlowFactory(MockEngineFactory);

    impl EngineFactory for SlowFactory {
        fn detect_kind(&self, model_dir: &Path) -> engine::ModelKind {
            self.0.detect_kind(model_dir)
        }
        fn load(
            &self,
            model_path: &Path,
            device: &str,
            cache_size_gb: f64,
            max_num_seqs: usize,
        ) -> anyhow::Result<(EngineHandleKind, std::thread::JoinHandle<()>)> {
            std::thread::sleep(std::time::Duration::from_millis(200));
            self.0.load(model_path, device, cache_size_gb, max_num_seqs)
        }
    }

    /// T5.1 accept test: dropping the `load_model` future mid-flight (the
    /// HTTP client disconnected) must NOT wedge the record in `Loading` —
    /// the cancellation guard resets it to `NotLoaded`, the VRAM reservation
    /// is released, and a subsequent load succeeds.
    #[tokio::test]
    async fn dropped_load_future_resets_state_and_allows_reload() {
        let config = {
            let mut c = tests::make_config(&[("m", 1.0)], 10.0);
            c.vram_safety_margin_gb = 0.0;
            c
        };
        let mm = ModelManager::new(config, Arc::new(SlowFactory(MockEngineFactory::default())))
            .await
            .unwrap();

        // Poll the load future exactly once: it parks at the engine-load
        // await (the factory sleeps), leaving the record in `Loading`.
        let mut fut = Box::pin(mm.load_model("m"));
        let mut cx = Context::from_waker(Waker::noop());
        assert!(
            matches!(fut.as_mut().poll(&mut cx), Poll::Pending),
            "load must park at the engine-load await"
        );

        // Simulate the client disconnect: axum drops the handler future.
        drop(fut);

        // The guard must have reset the state — no permanent Loading wedge —
        // and released the tracker reservation.
        let state = mm
            .list_models()
            .into_iter()
            .find(|m| m.id == "m")
            .unwrap()
            .state;
        assert_eq!(
            state,
            ModelState::NotLoaded,
            "cancelled load must reset Loading → NotLoaded"
        );
        assert!(
            mm.metrics_snapshot().used_vram_gb.abs() < f64::EPSILON,
            "cancelled load must release its VRAM reservation"
        );

        // And the model is loadable again.
        mm.load_model("m").await.unwrap();
        let state = mm
            .list_models()
            .into_iter()
            .find(|m| m.id == "m")
            .unwrap()
            .state;
        assert_eq!(state, ModelState::Ready, "reload after cancellation works");
    }

    // ---- T5.2: load serialization + panic-safe VRAM tracker ------------

    /// A factory whose engine thread **panics** once its command channel
    /// closes — i.e. on the eviction path, right where the normal mock thread
    /// would exit cleanly. Used to force the `thread.join()` panic that the
    /// pre-T5.2 evict path mishandled.
    struct PanicOnEvictFactory;

    impl EngineFactory for PanicOnEvictFactory {
        fn detect_kind(&self, _model_dir: &Path) -> engine::ModelKind {
            engine::ModelKind::TextGen
        }
        fn load(
            &self,
            _model_path: &Path,
            _device: &str,
            _cache_size_gb: f64,
            _max_num_seqs: usize,
        ) -> anyhow::Result<(EngineHandleKind, std::thread::JoinHandle<()>)> {
            let (tx, mut rx) = tokio::sync::mpsc::channel::<crate::cb_engine::EngineCommand>(32);
            let thread = std::thread::spawn(move || {
                // Drain until the handle drops (eviction closes the channel),
                // then panic — simulating an engine thread that died on its way out.
                while rx.blocking_recv().is_some() {}
                panic!("simulated engine-thread panic during eviction");
            });
            Ok((
                EngineHandleKind::TextGen(crate::cb_engine::EngineHandle::from_sender(tx)),
                thread,
            ))
        }
    }

    /// T5.2 accept (finding 20): two concurrent loads of DIFFERENT models that
    /// each need an eviction resolve deterministically — both `load_model`
    /// futures return `Ok`, with no spurious "already being evicted" 500. The
    /// load-mutex serialises victim selection so the second load re-evaluates
    /// VRAM only after the first has committed its reservation and gone Ready.
    #[tokio::test]
    async fn concurrent_loads_of_different_models_resolve_deterministically() {
        // total=10, three 4 GB models, no KV cap → each load grabs all VRAM,
        // so a second resident model always requires evicting the first.
        let mm = Arc::new(
            ModelManager::new(
                tests::make_config(&[("v", 4.0), ("a", 4.0), ("b", 4.0)], 10.0),
                Arc::new(MockEngineFactory::default()),
            )
            .await
            .unwrap(),
        );
        // Preload the victim so both concurrent loads must evict to fit.
        mm.load_model("v").await.unwrap();

        let (ra, rb) = {
            let (ma, mb) = (Arc::clone(&mm), Arc::clone(&mm));
            let ta = tokio::spawn(async move { ma.load_model("a").await });
            let tb = tokio::spawn(async move { mb.load_model("b").await });
            tokio::join!(ta, tb)
        };
        assert!(ra.unwrap().is_ok(), "concurrent load 'a' must resolve Ok");
        assert!(rb.unwrap().is_ok(), "concurrent load 'b' must resolve Ok");

        // Exactly one of {a, b} is resident (the last to load evicted the other);
        // v was evicted to make room. VRAM = one model's full reservation.
        let ready: Vec<String> = mm.list_ready_models();
        assert_eq!(ready.len(), 1, "exactly one model resident, got {ready:?}");
        assert!(
            (mm.metrics_snapshot().used_vram_gb - 10.0).abs() < 1e-9,
            "the single resident model holds the full 10 GB reservation"
        );
    }

    /// Edge case requested 2026-08-03 while stress-testing the cache-management
    /// design: `evict_model` never takes `load_lock` (unlike LRU eviction inside
    /// `ensure_vram_for`, which does), so an admin/LRU eviction of one model can
    /// run fully concurrently with a `load_model` of an *unrelated* model. Both
    /// must resolve correctly with no VRAM-tracker corruption — plenty of
    /// headroom here so neither load needs to evict the other; this isolates
    /// the "two independent state machines mutating the shared tracker at once"
    /// race from the LRU-contention race the prior test already covers.
    #[tokio::test]
    async fn concurrent_evict_of_one_model_and_load_of_another_do_not_corrupt_tracker() {
        // cache_size_gb caps each KV pool (matches the established pattern,
        // e.g. test_vram_alloc_equals_free_across_text_and_vision below) so
        // both models genuinely have room to co-reside: weight 4 + KV 1 = 5GB
        // each, 10GB of 20GB total. Without this cap the mock's default
        // "grab all remaining VRAM" KV sizing means 'resident' alone consumes
        // the entire budget, so 'incoming' can never succeed regardless of
        // timing — that would test VRAM exhaustion, not the intended race.
        let mut config = tests::make_config(&[("resident", 4.0), ("incoming", 4.0)], 20.0);
        config.cache_size_gb = 1.0;
        let mm = Arc::new(
            ModelManager::new(config, Arc::new(MockEngineFactory::default()))
                .await
                .unwrap(),
        );
        mm.load_model("resident").await.unwrap();

        let (evict_res, load_res) = {
            let (m1, m2) = (Arc::clone(&mm), Arc::clone(&mm));
            let t1 = tokio::spawn(async move { m1.evict_model("resident").await });
            let t2 = tokio::spawn(async move { m2.load_model("incoming").await });
            tokio::join!(t1, t2)
        };
        assert!(
            evict_res.unwrap().is_ok(),
            "evicting 'resident' must succeed"
        );
        let load_res = load_res.unwrap();
        assert!(
            load_res.is_ok(),
            "loading 'incoming' must succeed — there is ample independent VRAM \
             for it regardless of 'resident's concurrent eviction: {:?}",
            load_res.err()
        );

        let models = mm.list_models();
        let resident_state = models.iter().find(|m| m.id == "resident").unwrap().state;
        let incoming_state = models.iter().find(|m| m.id == "incoming").unwrap().state;
        assert_eq!(
            resident_state,
            ModelState::NotLoaded,
            "resident was evicted"
        );
        assert_eq!(
            incoming_state,
            ModelState::Ready,
            "incoming finished loading"
        );
        assert!(
            (mm.metrics_snapshot().used_vram_gb - 5.0).abs() < 1e-9,
            "tracker must hold exactly 'incoming's 5GB (4 weight + 1 KV) — no \
             double-charge, no leak from the concurrent evict"
        );
    }

    /// Edge case requested 2026-08-03: evict and (re)load racing on the SAME
    /// model. The state machine (`begin_load`/`begin_evict`) must resolve this
    /// deterministically-consistent (not deterministically-*one-outcome*: the
    /// actual winner depends on unspecified tokio poll ordering between the two
    /// spawned tasks, which this test must not assume) in either interleaving —
    /// never a corrupted or wedged record. If evict's `begin_evict` (Ready→
    /// Evicting, a sync, non-yielding call) runs before load's `begin_load`
    /// checks state, the concurrent load sees `Evicting` and errors cleanly; if
    /// load's `begin_load` happens to run only after eviction has *fully*
    /// completed (state back to `NotLoaded`), it legitimately reloads the model
    /// from scratch instead of no-op'ing. Both outcomes are valid — the
    /// invariant that must hold regardless is: the model ends in a real,
    /// unwedged state (`Ready` or `NotLoaded`, never stuck `Loading`/`Evicting`),
    /// and the VRAM tracker exactly matches whichever state won (never a
    /// leak or double-charge).
    #[tokio::test]
    async fn evict_and_load_racing_on_the_same_model_resolve_without_corruption() {
        let mm = Arc::new(
            ModelManager::new(
                tests::make_config(&[("a", 4.0)], 10.0),
                Arc::new(MockEngineFactory::default()),
            )
            .await
            .unwrap(),
        );
        mm.load_model("a").await.unwrap();

        let (evict_res, load_res) = {
            let (m1, m2) = (Arc::clone(&mm), Arc::clone(&mm));
            let t1 = tokio::spawn(async move { m1.evict_model("a").await });
            let t2 = tokio::spawn(async move { m2.load_model("a").await });
            tokio::join!(t1, t2)
        };
        assert!(evict_res.unwrap().is_ok(), "evict must always succeed here");
        // load either no-ops (already Ready when it ran), errors cleanly
        // (Evicting when it ran), or genuinely reloads (NotLoaded when it ran,
        // i.e. it ran strictly after eviction fully finished) — all three are
        // correct; a panic is not.
        if let Ok(Err(e)) = &load_res {
            assert!(
                e.to_string().contains("being evicted"),
                "the only acceptable load error here is the Evicting rejection: {e}"
            );
        }

        let state = mm
            .list_models()
            .into_iter()
            .find(|m| m.id == "a")
            .unwrap()
            .state;
        let used_vram = mm.metrics_snapshot().used_vram_gb;
        match state {
            ModelState::NotLoaded => assert!(
                used_vram.abs() < f64::EPSILON,
                "NotLoaded must carry no VRAM reservation, got {used_vram}"
            ),
            ModelState::Ready => assert!(
                (used_vram - 4.0).abs() < 1e-9,
                "Ready must hold exactly the model's 4GB reservation, got {used_vram}"
            ),
            other => panic!("model must end Ready or NotLoaded, never wedged in {other:?}"),
        }
    }

    /// Edge case requested 2026-08-03: N concurrent `load_model` calls for the
    /// SAME never-loaded model ("load during two parallel loads" and "multiple
    /// loads in parallel" collapse to this one scenario when the model is
    /// identical). `begin_load`'s `NotLoaded → Loading` check-and-flip happens
    /// under a std (blocking, non-async) lock (`write_models`), so it is a true
    /// mutual-exclusion point regardless of tokio's scheduling order — exactly
    /// one of the N concurrent callers can ever observe `NotLoaded` and become
    /// the real loader; every other caller observes `Loading` (set by the
    /// winner) and fast-fails with `ModelError::Loading` without waiting or
    /// double-reserving VRAM.
    #[tokio::test]
    async fn concurrent_loads_of_the_same_model_exactly_one_wins() {
        // cache_size_gb caps the KV pool (established pattern, see the other
        // co-residency tests above) so the VRAM assertion below has a
        // predictable target — the mock's default is "grab all remaining
        // VRAM" for KV, which would otherwise charge 10GB (the whole budget)
        // instead of a value tied to this model's own 4GB weight.
        let mut config = tests::make_config(&[("a", 4.0)], 10.0);
        config.cache_size_gb = 1.0;
        let mm = Arc::new(
            ModelManager::new(config, Arc::new(MockEngineFactory::default()))
                .await
                .unwrap(),
        );

        let results = {
            let handles: Vec<_> = (0..3)
                .map(|_| {
                    let m = Arc::clone(&mm);
                    tokio::spawn(async move { m.load_model("a").await })
                })
                .collect();
            let mut results = Vec::with_capacity(handles.len());
            for h in handles {
                results.push(h.await.unwrap());
            }
            results
        };

        let oks = results.iter().filter(|r| r.is_ok()).count();
        let loading_errs = results
            .iter()
            .filter(|r| {
                r.as_ref()
                    .err()
                    .and_then(|e| e.downcast_ref::<ModelError>())
                    == Some(&ModelError::Loading)
            })
            .count();
        assert_eq!(
            oks, 1,
            "exactly one of 3 concurrent loads must succeed: {results:?}"
        );
        assert_eq!(
            loading_errs, 2,
            "the other 2 must fast-fail with ModelError::Loading, not some other \
             error or a hang: {results:?}"
        );

        let state = mm
            .list_models()
            .into_iter()
            .find(|m| m.id == "a")
            .unwrap()
            .state;
        assert_eq!(state, ModelState::Ready, "the winner's load completed");
        assert!(
            (mm.metrics_snapshot().used_vram_gb - 5.0).abs() < 1e-9,
            "VRAM charged exactly once (4 weight + 1 KV) — no double-reservation \
             from the losers"
        );
    }

    /// T5.2 accept (finding 38): an engine thread that panics on the eviction
    /// path must still leave the tracker consistent — the VRAM reservation is
    /// freed and the record returns to `NotLoaded`, never wedged in `Evicting`
    /// with a leaked reservation that would reject every later load.
    #[tokio::test]
    async fn evict_path_panic_leaves_tracker_consistent() {
        let mm = ModelManager::new(
            tests::make_config(&[("m", 4.0)], 10.0),
            Arc::new(PanicOnEvictFactory),
        )
        .await
        .unwrap();
        mm.load_model("m").await.unwrap();
        assert!(
            mm.metrics_snapshot().used_vram_gb > 0.0,
            "a loaded model holds a VRAM reservation"
        );

        // Evict: the engine thread panics on join. The call surfaces the unclean
        // exit (panic or, post-#4, a join timeout — same reconciliation either way)…
        let err = mm.evict_model("m").await.unwrap_err();
        assert!(
            err.to_string().contains("did not exit cleanly"),
            "evict must surface the unclean thread exit: {err}"
        );

        // …but the tracker is reconciled despite it (the core T5.2 guarantee).
        assert!(
            mm.metrics_snapshot().used_vram_gb.abs() < f64::EPSILON,
            "evict-path panic must still free the VRAM reservation"
        );
        let state = mm
            .list_models()
            .into_iter()
            .find(|m| m.id == "m")
            .unwrap()
            .state;
        assert_eq!(
            state,
            ModelState::NotLoaded,
            "panic must not wedge the record in Evicting"
        );
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::device_inventory::{DeviceInfo, DeviceKind, DeviceTier};
    use crate::model_manager::lifecycle::MockEngineFactory;

    // ---- KV-cache pressure state machine (dev/plans/kv-cache-pressure-detection.md) ----

    /// Usage below threshold never sets `over_threshold_since`, no transition.
    #[test]
    fn pressure_state_stays_clear_under_threshold() {
        let mut state = KvPressureState::default();
        let now = Instant::now();
        let t = update_pressure_state(&mut state, 50.0, 90.0, Duration::from_mins(1), now);
        assert_eq!(t, PressureTransition::Unchanged);
        assert!(state.over_threshold_since.is_none());
    }

    /// Crossing the threshold starts the clock but does NOT flag immediately
    /// — a single sample over threshold is exactly the "momentary spike"
    /// case this whole design exists to avoid reacting to.
    #[test]
    fn pressure_state_over_threshold_not_yet_sustained_is_unchanged() {
        let mut state = KvPressureState::default();
        let now = Instant::now();
        let t = update_pressure_state(&mut state, 95.0, 90.0, Duration::from_mins(1), now);
        assert_eq!(t, PressureTransition::Unchanged);
        assert_eq!(state.over_threshold_since, Some(now));
    }

    /// Once the sustained duration has genuinely elapsed since crossing, the
    /// next sample (still over threshold) reports `JustFlagged` exactly once.
    #[test]
    fn pressure_state_flags_once_sustained_duration_elapses() {
        let mut state = KvPressureState {
            over_threshold_since: Instant::now().checked_sub(Duration::from_secs(61)),
            ..Default::default()
        };
        let t = update_pressure_state(
            &mut state,
            95.0,
            90.0,
            Duration::from_mins(1),
            Instant::now(),
        );
        assert_eq!(t, PressureTransition::JustFlagged);

        // The very next tick, still over threshold: no new transition — this
        // is what keeps the warn-log from repeating every sweep.
        let t2 = update_pressure_state(
            &mut state,
            95.0,
            90.0,
            Duration::from_mins(1),
            Instant::now(),
        );
        assert_eq!(t2, PressureTransition::Unchanged);
    }

    /// Dropping back under threshold after being flagged clears the state and
    /// reports `JustCleared` exactly once, not a lingering flag.
    #[test]
    fn pressure_state_clears_when_usage_drops_back_down() {
        let mut state = KvPressureState {
            over_threshold_since: Instant::now().checked_sub(Duration::from_secs(61)),
            ..Default::default()
        };
        let sustained = Duration::from_mins(1);
        assert_eq!(
            update_pressure_state(&mut state, 95.0, 90.0, sustained, Instant::now()),
            PressureTransition::JustFlagged
        );
        let t = update_pressure_state(&mut state, 50.0, 90.0, sustained, Instant::now());
        assert_eq!(t, PressureTransition::JustCleared);
        assert!(state.over_threshold_since.is_none());
    }

    /// A brief dip back under threshold resets the clock entirely — this is
    /// "continuously" at/above threshold, not "cumulatively": a model that
    /// oscillates around the line never gets flagged, by design.
    #[test]
    fn pressure_state_dip_below_threshold_resets_the_clock() {
        let mut state = KvPressureState::default();
        let t0 = Instant::now();
        update_pressure_state(&mut state, 95.0, 90.0, Duration::from_mins(1), t0);
        assert_eq!(state.over_threshold_since, Some(t0));

        // Dips under threshold before the sustain window elapses — clock resets.
        update_pressure_state(
            &mut state,
            50.0,
            90.0,
            Duration::from_mins(1),
            t0 + Duration::from_secs(30),
        );
        assert!(state.over_threshold_since.is_none());

        // Back over threshold — the clock restarts from THIS instant, not t0.
        let t1 = t0 + Duration::from_secs(35);
        update_pressure_state(&mut state, 95.0, 90.0, Duration::from_mins(1), t1);
        assert_eq!(state.over_threshold_since, Some(t1));
    }

    // ---- Helpers -------------------------------------------------------

    /// Build a synthetic `DeviceInfo` for inventory tests. `domain_id` follows
    /// the real rule (discrete GPU → its own name, else the shared system domain)
    /// so callers exercise `resolve_inference_domain` exactly as production would.
    fn dev(name: &str, kind: DeviceKind) -> DeviceInfo {
        let domain_id = if kind == DeviceKind::DiscreteGpu {
            name.to_owned()
        } else {
            crate::device_inventory::DOMAIN_SYSTEM.to_owned()
        };
        DeviceInfo {
            name: name.to_owned(),
            full_name: name.to_owned(),
            architecture: None,
            kind,
            tier: DeviceTier::Fallback,
            domain_id,
            total_mem_gb: None,
        }
    }

    /// A discrete inference GPU resolves to its own name as the memory domain.
    #[test]
    fn resolve_inference_domain_discrete_gpu_is_own_name() {
        let mut cfg = make_config(&[], 22.5);
        cfg.device = "GPU.1".to_owned();
        let inv = DeviceInventory::from_devices(vec![
            dev("GPU.1", DeviceKind::DiscreteGpu),
            dev("GPU.0", DeviceKind::IntegratedGpu),
            dev("CPU", DeviceKind::Cpu),
        ]);
        assert_eq!(resolve_inference_domain(&cfg, &inv), "GPU.1");
    }

    /// An integrated-GPU / CPU inference device resolves to the shared "system"
    /// domain — so a CPU box's legacy `total_vram_gb` budget lands on "system".
    #[test]
    fn resolve_inference_domain_integrated_is_system() {
        let mut cfg = make_config(&[], 0.0);
        cfg.device = "GPU.0".to_owned();
        let inv = DeviceInventory::from_devices(vec![
            dev("GPU.1", DeviceKind::DiscreteGpu),
            dev("GPU.0", DeviceKind::IntegratedGpu),
        ]);
        assert_eq!(
            resolve_inference_domain(&cfg, &inv),
            crate::device_inventory::DOMAIN_SYSTEM
        );
    }

    /// A device absent from the inventory (empty inventory in GPU-free tests, or
    /// an un-probed device) falls back to the device name verbatim.
    #[test]
    fn resolve_inference_domain_unknown_device_falls_back_to_name() {
        let mut cfg = make_config(&[], 22.5);
        cfg.device = "GPU.1".to_owned();
        assert_eq!(
            resolve_inference_domain(&cfg, &DeviceInventory::default()),
            "GPU.1"
        );
    }

    // ---- assemble_domain_budgets (the per-domain budget policy) ------------

    /// As [`dev`], but with a reported total-memory figure (for secondary-GPU
    /// budget tests where the device's own VRAM is the source).
    fn dev_mem(name: &str, kind: DeviceKind, mem_gb: f64) -> DeviceInfo {
        DeviceInfo {
            total_mem_gb: Some(mem_gb),
            ..dev(name, kind)
        }
    }

    /// Look up one domain's assembled budget.
    fn budget_of(budgets: &[(String, f64)], domain: &str) -> Option<f64> {
        budgets.iter().find(|(d, _)| d == domain).map(|(_, b)| *b)
    }

    /// dGPU box: the inference GPU keeps `total_vram_gb`; the iGPU+CPU collapse
    /// into a "system" domain that takes the passed (RAM-derived) budget.
    #[test]
    fn assemble_dgpu_box_inference_keeps_vram_system_gets_ram() {
        let mut cfg = make_config(&[], 22.5);
        cfg.device = "GPU.1".to_owned();
        let inv = DeviceInventory::from_devices(vec![
            dev("GPU.1", DeviceKind::DiscreteGpu),
            dev("GPU.0", DeviceKind::IntegratedGpu),
            dev("CPU", DeviceKind::Cpu),
        ]);
        let b = assemble_domain_budgets(&cfg, &inv, "GPU.1", 11.0);
        assert_eq!(b.len(), 2, "two domains: GPU.1 + system");
        assert_eq!(budget_of(&b, "GPU.1"), Some(22.5));
        assert_eq!(budget_of(&b, DOMAIN_SYSTEM), Some(11.0));
    }

    /// A second discrete GPU (not the inference device) is budgeted from its own
    /// reported VRAM, not the inference `total_vram_gb`.
    #[test]
    fn assemble_secondary_discrete_gpu_uses_its_own_vram() {
        let mut cfg = make_config(&[], 22.5);
        cfg.device = "GPU.1".to_owned();
        let inv = DeviceInventory::from_devices(vec![
            dev_mem("GPU.1", DeviceKind::DiscreteGpu, 16.0),
            dev_mem("GPU.0", DeviceKind::DiscreteGpu, 8.0),
        ]);
        let b = assemble_domain_budgets(&cfg, &inv, "GPU.1", 99.0);
        assert_eq!(
            budget_of(&b, "GPU.1"),
            Some(22.5),
            "inference uses total_vram_gb"
        );
        assert_eq!(
            budget_of(&b, "GPU.0"),
            Some(8.0),
            "secondary uses its own VRAM"
        );
    }

    /// An explicit `domain_budgets` override wins over every derived budget —
    /// including the inference domain's `total_vram_gb` and the system RAM figure.
    #[test]
    fn assemble_explicit_override_wins() {
        let mut cfg = make_config(&[], 22.5);
        cfg.device = "GPU.1".to_owned();
        cfg.domain_budgets.insert("GPU.1".to_owned(), 20.0);
        cfg.domain_budgets.insert(DOMAIN_SYSTEM.to_owned(), 5.0);
        let inv = DeviceInventory::from_devices(vec![
            dev("GPU.1", DeviceKind::DiscreteGpu),
            dev("GPU.0", DeviceKind::IntegratedGpu),
        ]);
        let b = assemble_domain_budgets(&cfg, &inv, "GPU.1", 11.0);
        assert_eq!(
            budget_of(&b, "GPU.1"),
            Some(20.0),
            "override beats total_vram_gb"
        );
        assert_eq!(
            budget_of(&b, DOMAIN_SYSTEM),
            Some(5.0),
            "override beats RAM budget"
        );
    }

    /// Empty inventory (GPU-free tests): exactly one domain — the inference one
    /// at `total_vram_gb`. This is the back-compat single-domain shape.
    #[test]
    fn assemble_empty_inventory_is_single_inference_domain() {
        let mut cfg = make_config(&[], 22.5);
        cfg.device = "GPU.1".to_owned();
        let b = assemble_domain_budgets(&cfg, &DeviceInventory::default(), "GPU.1", 11.0);
        assert_eq!(b, vec![("GPU.1".to_owned(), 22.5)]);
    }

    /// Critical back-compat: when the inference device is itself in the system
    /// domain (a CPU box), that domain takes `total_vram_gb` — NOT the RAM-derived
    /// budget — so a CPU box's `total_vram_gb = 0.0` stays gating-disabled.
    #[test]
    fn assemble_cpu_inference_keeps_legacy_total_vram_not_ram() {
        let mut cfg = make_config(&[], 0.0);
        cfg.device = "CPU".to_owned();
        let inv = DeviceInventory::from_devices(vec![
            dev("CPU", DeviceKind::Cpu),
            dev("GPU.0", DeviceKind::IntegratedGpu),
        ]);
        let b = assemble_domain_budgets(&cfg, &inv, DOMAIN_SYSTEM, 11.0);
        assert_eq!(b.len(), 1, "CPU + iGPU collapse to one system domain");
        assert_eq!(
            budget_of(&b, DOMAIN_SYSTEM),
            Some(0.0),
            "legacy total_vram_gb wins for the inference (system) domain → stays disabled"
        );
    }

    /// Build a Config with the given known models and no preload.
    pub(super) fn make_config(models: &[(&str, f64)], total_vram_gb: f64) -> Config {
        Config {
            models_dir: std::path::PathBuf::from("/tmp/test-models"),
            device: "CPU".to_owned(),
            preload: vec![],
            models: models
                .iter()
                .map(|(id, gb)| {
                    (
                        id.to_string(),
                        config::ModelEntry {
                            vram_gb: *gb,
                            kind: None, // mock detects kind by name substring
                            policy: config::ModelPolicy::default(),
                        },
                    )
                })
                .collect(),
            total_vram_gb,
            max_num_seqs: 1000, // large test cap — MockEngineFactory has no KV pool
            cache_size_gb: 0.0, // no cap — use full dynamic sizing
            default_kv_cache_gb: 0.0, // unbounded — pre-Slice-2 grab-all
            vram_safety_margin_gb: 0.0, // no margin — keeps existing LRU test math simple
            min_kv_cache_gb: 0.0, // no minimum — allows any KV size including 0
            light_model_max_gb: 2.0, // C2 default; placement engine unused with empty inventory
            light_stt_max_gb: 1.0, // STT light/heavy threshold; engine unused with empty inventory
            dgpu_size_ceiling_fraction: 0.8, // production default; placement engine unused with empty inventory
            system_ram_reservation_gb: None, // per-OS default; no system domain in tests
            system_ram_budget_gb: None,      // opt-in; unset keeps the MemAvailable-only gate
            eviction_grace_secs: 0.0,        // no grace — keeps existing eviction test math simple
            realtime_defaults: None,         // irrelevant for mock
            realtime_viable_minimum: None,   // irrelevant for mock
            domain_budgets: HashMap::new(),  // no overrides — single inference domain
            device_budgets: HashMap::new(),  // disabled — irrelevant for mock
            kv_cache_precision: String::new(), // irrelevant for mock
            enable_prefix_caching: true,     // irrelevant for mock
            cors_allowed_origins: vec!["*".to_owned()], // irrelevant for mock
            api_keys: Vec::new(),            // auth off in tests
            admin_api_keys: Vec::new(),      // no admin scope in tests
            keys_file: None,
            admission_queue_timeout_ms: 5_000, // irrelevant for mock
            device_admission_queue_timeout_ms: 5_000, // irrelevant for mock
            embedding_pooling: "mean".to_owned(),
            embedding_normalize: true,
            default_embed_model: None,
            max_prompt_array: 16,              // the production default
            max_tokens_cap: 8192,              // the production default
            bind_addr: "127.0.0.1".to_owned(), // loopback — irrelevant for unit tests
            port: 11_437,
            allow_insecure_public_bind: false,
            supervisor: SupervisorConfig::default(),
            unknown_fields: HashMap::new(),
            embedding_device: None,
            ov_cache_dir: None,
            ov_cache_max_gb: 0.0,
            ov_cache_sweep_interval_secs: 21_600,
            kv_pressure_monitor_enabled: false,
            kv_pressure_threshold_pct: 0.0,
            kv_pressure_sustained_secs: 0,
            kv_pressure_sweep_interval_secs: 15,
            kv_resize_cooldown_secs: 30,
        }
    }

    /// Build a `ModelManager` synchronously (no preload) using `MockEngineFactory`.
    async fn make_mm(models: &[(&str, f64)], total_vram_gb: f64) -> ModelManager {
        let config = make_config(models, total_vram_gb);
        ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .expect("ModelManager::new failed in test")
    }

    // ---- resize_model_kv_cache (dev/plans/kv-cache-pressure-detection.md) ----

    /// Resizing an unregistered model → `NotFound`, same as every other
    /// model-scoped call.
    #[tokio::test]
    async fn resize_rejects_unknown_model() {
        let mm = make_mm(&[], 22.5).await;
        let err = mm
            .resize_model_kv_cache("no-such-model", 2.0)
            .await
            .unwrap_err();
        assert_eq!(
            err.downcast_ref::<ModelError>(),
            Some(&ModelError::NotFound("no-such-model".to_owned()))
        );
    }

    /// Resizing a registered-but-never-loaded model → `NotLoaded`: resize
    /// only applies to a resident model, it does not load one for the first
    /// time (that's what `/load` is for).
    #[tokio::test]
    async fn resize_rejects_not_ready_model() {
        let mm = make_mm(&[("qwen3-8b", 5.0)], 22.5).await;
        let err = mm.resize_model_kv_cache("qwen3-8b", 2.0).await.unwrap_err();
        assert_eq!(
            err.downcast_ref::<ModelError>(),
            Some(&ModelError::NotLoaded)
        );
    }

    /// Report-achieved: a resize whose target is clamped must return the pool
    /// the allocator actually delivered, not the number that was asked for.
    /// Here `cache_size_gb` caps the pool below the request, so a returned
    /// value equal to the request would be the exact bug this pins against —
    /// the endpoint used to echo `request.kv_cache_gb` unconditionally.
    #[tokio::test]
    async fn resize_returns_achieved_not_requested_when_clamped() {
        let mut config = make_config(&[("qwen3-8b", 5.0)], 22.5);
        config.cache_size_gb = 2.0;
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .expect("ModelManager::new failed in test");
        mm.load_model("qwen3-8b").await.unwrap();

        let achieved = mm.resize_model_kv_cache("qwen3-8b", 6.0).await.unwrap();

        assert!(
            (achieved - 2.0).abs() < 1e-9,
            "expected the cache_size_gb cap (2.0), got {achieved}"
        );
        // And it must agree with what the record/metrics actually report.
        let snap = mm.metrics_snapshot();
        let m = snap
            .models
            .iter()
            .find(|m| m.id == "qwen3-8b")
            .expect("resized model must still be reported");
        assert!(
            (m.kv_cache_pool_gb - achieved).abs() < 1e-9,
            "returned {achieved} but metrics report {}",
            m.kv_cache_pool_gb
        );
    }

    /// A resize on a `Ready` model evicts and reloads it at the new size —
    /// confirmed via `metrics_snapshot`'s live `kv_cache_pool_gb`, not just a
    /// successful `Ok(())`.
    #[tokio::test]
    async fn resize_applies_new_kv_cache_size() {
        let mm = make_mm(&[("qwen3-8b", 5.0)], 22.5).await;
        mm.load_model("qwen3-8b").await.unwrap();

        mm.resize_model_kv_cache("qwen3-8b", 3.0).await.unwrap();

        let snap = mm.metrics_snapshot();
        let m = snap
            .models
            .iter()
            .find(|m| m.id == "qwen3-8b")
            .expect("resized model must still be reported");
        assert!(
            m.loaded,
            "must be Ready again after the reload half of resize"
        );
        assert!(
            (m.kv_cache_pool_gb - 3.0).abs() < 1e-9,
            "expected the new 3.0 GB reservation, got {}",
            m.kv_cache_pool_gb
        );
    }

    /// A second resize call within `kv_resize_cooldown_secs` of the first is
    /// rejected — the guardrail the plan's ops-review added specifically so
    /// nothing (script or nervous operator) can hammer this endpoint on a
    /// single-slot model, where each call is a real evict+reload.
    #[tokio::test]
    async fn resize_second_call_within_cooldown_is_rejected() {
        let mm = make_mm(&[("qwen3-8b", 5.0)], 22.5).await;
        mm.load_model("qwen3-8b").await.unwrap();

        mm.resize_model_kv_cache("qwen3-8b", 3.0).await.unwrap();
        let err = mm.resize_model_kv_cache("qwen3-8b", 4.0).await.unwrap_err();
        assert!(
            err.to_string().contains("cooldown"),
            "must name the cooldown, not just fail silently: {err}"
        );

        // The rejected second call must not have evicted/reloaded anything —
        // the model should still be sitting at the FIRST resize's size.
        let snap = mm.metrics_snapshot();
        let m = snap.models.iter().find(|m| m.id == "qwen3-8b").unwrap();
        assert!(
            (m.kv_cache_pool_gb - 3.0).abs() < 1e-9,
            "cooldown-rejected resize must not have changed anything, got {}",
            m.kv_cache_pool_gb
        );
    }

    /// A non-positive or non-finite `kv_cache_gb` is rejected before any
    /// eviction happens — validated up front, same discipline as every other
    /// config/override knob in this codebase.
    #[tokio::test]
    async fn resize_rejects_non_positive_kv_cache_gb() {
        let mm = make_mm(&[("qwen3-8b", 5.0)], 22.5).await;
        mm.load_model("qwen3-8b").await.unwrap();

        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let err = mm.resize_model_kv_cache("qwen3-8b", bad).await.unwrap_err();
            assert!(
                err.to_string().contains("kv_cache_gb"),
                "must name the field for {bad}: {err}"
            );
        }
        // Still Ready, untouched by the rejected attempts.
        assert!(mm.metrics_snapshot().models[0].loaded);
    }

    // ---- get_handle state machine ----------------------------------------

    /// Requesting a model ID not in the registry → `NotFound`.
    #[tokio::test]
    async fn test_get_handle_not_found() {
        let mm = make_mm(&[], 22.5).await;
        let err = mm.get_handle("no-such-model").unwrap_err();
        assert_eq!(err, ModelError::NotFound("no-such-model".to_owned()));
    }

    /// `metrics_snapshot` reports models that have been loaded at least once
    /// (never-loaded models are omitted), plus the device and total VRAM.
    #[tokio::test]
    async fn test_metrics_snapshot_reports_ready_models() {
        let mm = make_mm(&[("qwen3-8b", 5.5), ("other-14b", 9.0)], 22.5).await;

        // Nothing loaded yet → no models in the snapshot, but device/vram present.
        let empty = mm.metrics_snapshot();
        assert_eq!(empty.device, "CPU");
        assert!((empty.total_vram_gb - 22.5).abs() < 1e-9);
        assert!(
            empty.models.is_empty(),
            "no models reported before any load"
        );

        // Load one → it appears loaded, active=0, the mock's test cap (1000).
        // The never-loaded `other-14b` is still omitted (no kind to report yet).
        mm.load_model("qwen3-8b").await.unwrap();
        let snap = mm.metrics_snapshot();
        assert_eq!(
            snap.models.len(),
            1,
            "only the ever-loaded model is reported"
        );
        let m = &snap.models[0];
        assert_eq!(m.id, "qwen3-8b");
        assert_eq!(m.kind, ModelKind::TextGen);
        assert!(m.loaded, "Ready model reports loaded == true");
        assert_eq!(m.active, 0);
        assert_eq!(m.max_seqs, 1000);
        assert_eq!(m.waiting, 0, "idle engine reports no waiting requests");
    }

    /// After eviction a model must STILL appear in the snapshot — with
    /// `loaded == false`, its kind retained, and all request counters zeroed —
    /// so `record_state` can reset `rustedvino_model_loaded` to 0 instead of
    /// leaving the stale `1` a prior Ready scrape wrote. (Regression: the old
    /// snapshot filtered to Ready models, so an evicted model dropped out and
    /// its gauge stuck at 1 forever.)
    #[tokio::test]
    async fn test_metrics_snapshot_reports_evicted_model_as_unloaded() {
        let mm = make_mm(&[("qwen3-8b", 5.5)], 22.5).await;

        mm.load_model("qwen3-8b").await.unwrap();
        assert!(mm.metrics_snapshot().models[0].loaded, "loaded after load");

        mm.evict_model("qwen3-8b").await.unwrap();
        let snap = mm.metrics_snapshot();
        assert_eq!(
            snap.models.len(),
            1,
            "evicted model is still reported (to zero it)"
        );
        let m = &snap.models[0];
        assert_eq!(m.id, "qwen3-8b");
        assert!(!m.loaded, "evicted model reports loaded == false");
        assert_eq!(
            m.kind,
            ModelKind::TextGen,
            "kind retained for a stable label set"
        );
        assert_eq!(m.active, 0, "evicted model has no handle → zero counters");
        assert_eq!(m.max_seqs, 0);
        assert_eq!(m.waiting, 0);
    }

    /// R2 VRAM invariant (one kind): a load reserves at least the model weight,
    /// and eviction frees the *full* reservation — `used_vram_gb` returns to
    /// exactly zero (alloc==free on evict).
    #[tokio::test]
    async fn test_vram_used_resets_to_zero_on_evict() {
        let mm = make_mm(&[("qwen3-8b", 5.5)], 22.5).await;
        assert!(
            mm.metrics_snapshot().used_vram_gb.abs() < 1e-9,
            "nothing reserved before any load"
        );

        mm.load_model("qwen3-8b").await.unwrap();
        let used = mm.metrics_snapshot().used_vram_gb;
        assert!(
            used >= 5.5,
            "reservation must cover at least the model weight, got {used}"
        );

        mm.evict_model("qwen3-8b").await.unwrap();
        assert!(
            mm.metrics_snapshot().used_vram_gb.abs() < 1e-9,
            "alloc==free: used VRAM returns to zero after eviction"
        );
    }

    /// R2 VRAM invariant (across kinds): a text model and a vision model are
    /// VRAM-tracked identically. With a small KV cap they sit co-resident; the
    /// snapshot sums both reservations and returns to zero once both evict —
    /// proving the registry accounts for *every* kind, not just `TextGen`.
    #[tokio::test]
    async fn test_vram_alloc_equals_free_across_text_and_vision() {
        // cache_size_gb caps each KV pool so both models fit (no LRU eviction):
        // weight 4 + KV 1 = 5 GB each, 10 GB of 22.5 GB total.
        let mut config = make_config(&[("qwen3-8b", 4.0), ("qwen2.5-vlm", 4.0)], 22.5);
        config.cache_size_gb = 1.0;
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();

        mm.load_model("qwen3-8b").await.unwrap();
        mm.load_model("qwen2.5-vlm").await.unwrap();

        let snap = mm.metrics_snapshot();
        assert_eq!(snap.models.len(), 2, "both kinds are co-resident");
        let kinds: Vec<ModelKind> = snap.models.iter().map(|m| m.kind).collect();
        assert!(
            kinds.contains(&ModelKind::TextGen) && kinds.contains(&ModelKind::Vision),
            "snapshot carries one TextGen and one Vision model, got {kinds:?}"
        );
        assert!(
            (snap.used_vram_gb - 10.0).abs() < 1e-6,
            "used VRAM is the sum of both reservations (5+5), got {}",
            snap.used_vram_gb
        );

        mm.evict_model("qwen3-8b").await.unwrap();
        mm.evict_model("qwen2.5-vlm").await.unwrap();
        assert!(
            mm.metrics_snapshot().used_vram_gb.abs() < 1e-9,
            "alloc==free across kinds: zero reserved after evicting both"
        );
    }

    /// R4/G3 keystone: the `Embedding` kind — the first real "media" flow —
    /// drives the ENTIRE manager: register → load(mock) → list → metrics →
    /// embed → routing rejection → evict, through only the uniform
    /// `ManagedEngine` facade. The manager core (load/evict/VRAM/LRU/metrics)
    /// needed zero edits to host it, proving the seam: a new kind is "a handle +
    /// a factory arm," not a manager rewrite.
    #[tokio::test]
    async fn embedding_kind_flows_through_the_registry_with_no_manager_edits() {
        // The "embed" substring makes MockEngineFactory classify this as Embedding.
        let mm = make_mm(&[("bge-embed-ov", 1.5)], 22.5).await;

        // Registered but not loaded → invisible to the Ready-only /v1/models.
        assert!(mm.list_ready_models().is_empty());

        // load(mock) → Ready via the real embedding handle. VRAM is reserved like
        // any kind.
        mm.load_model("bge-embed-ov").await.unwrap();
        assert_eq!(mm.list_ready_models(), vec!["bge-embed-ov".to_owned()]);

        // metrics: appears with kind=Embedding and the engine's real capacity.
        let snap = mm.metrics_snapshot();
        assert_eq!(snap.models.len(), 1);
        let m = &snap.models[0];
        assert_eq!(m.kind, ModelKind::Embedding);
        assert_eq!(m.active, 0);
        assert_eq!(
            m.max_seqs,
            crate::embed_engine::EMBED_CHANNEL_CAP,
            "a real embedding engine reports its channel cap"
        );
        assert_eq!(m.waiting, 0);
        // `used_vram_gb` is scoped to the primary inference domain only (T1/F7).
        // Now that registration correctly detects this model's kind up front
        // (the build_not_loaded_record/new_with_inventory fix — see
        // dev/autotest/20260717_qwen3_4b_int8_sigsegv.md Finding 3),
        // `resolve_model_device` routes it through the real
        // `resolve_embedding_device` probe like production does, which may
        // land it on a different device/domain than the primary one — so
        // check across every tracked domain instead of assuming a specific one.
        let total_used_across_domains: f64 = snap.domain_vram.iter().map(|(_, _, used)| used).sum();
        assert!(
            total_used_across_domains >= 1.5,
            "VRAM tracked for the embedding kind, in whichever domain it resolved to"
        );

        // The Ready record resolves to the Embedding handle, and it actually
        // embeds (the mock answers with a fixed vector per input).
        let handle = mm.get_embedding_handle("bge-embed-ov").unwrap();
        let out = handle
            .embed(vec!["a".to_owned(), "b".to_owned()])
            .await
            .unwrap();
        assert_eq!(out.vectors.len(), 2, "one vector per input");

        // A text-only endpoint (/tokenize, /v1/completions, chat) rejects it.
        assert!(matches!(
            mm.get_handle("bge-embed-ov").unwrap_err(),
            ModelError::WrongKind(_)
        ));

        // Drop our handle clone BEFORE evicting: eviction joins the engine
        // thread, which only exits once every handle (Sender) is dropped — the
        // quiesce protocol. A live request holding a handle keeps the engine up
        // until it finishes; here that means the test must release its clone or
        // the join would block forever.
        drop(handle);

        // evict → VRAM freed back to zero (alloc==free holds for the kind).
        mm.evict_model("bge-embed-ov").await.unwrap();
        assert!(mm.list_ready_models().is_empty());
        assert!(
            mm.metrics_snapshot().used_vram_gb.abs() < 1e-9,
            "alloc==free for the embedding kind too"
        );
    }

    /// A model added at runtime with no explicit `kind` must be correctly
    /// auto-detected (the same file/name-sniffing `detect_kind` uses at load
    /// time) at REGISTRATION, not silently defaulted to `TextGen` — the bug
    /// found and fixed in the project's internal engineering log
    /// Finding 3: `configured_kind` (what `GET /v1/admin/models` reports, and
    /// what feeds `resolve_model_device`'s kind-aware placement) previously
    /// hardcoded `TextGen` whenever `kind` was omitted, even for a real VLM —
    /// even though the actual engine construction at load time detected it
    /// correctly via the identical file-sniffing, independently. `add_model`
    /// loads synchronously, so this also proves the fix survives past load
    /// (`execute_load` only ever touches `last_kind`, never `configured_kind`).
    #[tokio::test]
    async fn add_model_auto_detects_kind_instead_of_defaulting_to_text_gen() {
        let mm = make_mm(&[], 22.5).await;
        // Unlike the static-config registration path (which only consults the
        // mock `factory.model_exists`, always `true`), `add_model` checks the
        // real filesystem directly — needs a real directory under `make_config`'s
        // fixed `models_dir` ("/tmp/test-models").
        let dir = std::path::Path::new("/tmp/test-models/qwen-vlm-8b");
        std::fs::create_dir_all(dir).unwrap();
        // The "vlm" segment makes MockEngineFactory classify this as Vision.
        mm.add_model(
            "qwen-vlm-8b".to_owned(),
            6.0,
            None,
            config::ModelPolicy::default(),
        )
        .await
        .unwrap();
        std::fs::remove_dir_all(dir).ok();

        let info = mm
            .list_models()
            .into_iter()
            .find(|m| m.id == "qwen-vlm-8b")
            .expect("model registered");
        assert_eq!(
            info.kind,
            ModelKind::Vision,
            "configured_kind must reflect the real detected kind, not a TextGen default"
        );
    }

    /// `add_model` must reject an invalid policy field with the same
    /// invariant `Config::validate` enforces for the static file — the
    /// runtime registration path shares `config::validate_model_entry`, not a
    /// laxer ad hoc check.
    #[tokio::test]
    async fn add_model_rejects_invalid_policy_field() {
        let mm = make_mm(&[], 22.5).await;
        let dir = std::path::Path::new("/tmp/test-models/bad-policy-model");
        std::fs::create_dir_all(dir).unwrap();
        let err = mm
            .add_model(
                "bad-policy-model".to_owned(),
                1.0,
                None,
                config::ModelPolicy {
                    max_concurrent_streams: Some(0),
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
        std::fs::remove_dir_all(dir).ok();
        assert!(
            err.to_string().contains("max_concurrent_streams"),
            "error must name the invalid field: {err}"
        );
        assert!(
            mm.list_models()
                .into_iter()
                .all(|m| m.id != "bad-policy-model"),
            "a validation failure must not leave the model half-registered"
        );
    }

    /// `add_model`'s full `ModelPolicy` — not just `device` — must be durably
    /// persisted to `config.json`, so a runtime-registered model survives a
    /// restart with the same policy the operator set at add time (the gap
    /// `persist_model_entry` used to have: it only wrote `vram_gb`/`kind`/
    /// `device`, silently dropping everything else, e.g. image-gen's
    /// `precision`/`model_source`/`model_revision`).
    #[tokio::test]
    async fn add_model_persists_full_policy_to_config_file() {
        let (mm, tmp_dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;
        let model_dir = std::path::Path::new("/tmp/test-models/flux-test-model");
        std::fs::create_dir_all(model_dir).unwrap();

        mm.add_model(
            "flux-test-model".to_owned(),
            8.5,
            Some("image_gen".to_owned()),
            config::ModelPolicy {
                pinned: true,
                priority: 5,
                precision: Some("int4".to_owned()),
                model_source: Some("OpenVINO/FLUX.1-schnell-int4-ov".to_owned()),
                model_revision: Some("abc123".to_owned()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        std::fs::remove_dir_all(model_dir).ok();

        let persisted = std::fs::read_to_string(tmp_dir.path().join("config.json")).unwrap();
        let value: serde_json::Value = serde_json::from_str(&persisted).unwrap();
        let entry = &value["models"]["flux-test-model"];
        assert_eq!(entry["vram_gb"], 8.5);
        assert_eq!(entry["kind"], "image_gen");
        assert_eq!(entry["pinned"], true);
        assert_eq!(entry["priority"], 5);
        assert_eq!(entry["precision"], "int4");
        assert_eq!(entry["model_source"], "OpenVINO/FLUX.1-schnell-int4-ov");
        assert_eq!(entry["model_revision"], "abc123");
    }

    /// A failed `add_model` load must roll back the in-memory registration
    /// too, and must never persist anything to `config.json` — "the add
    /// failed" must mean "nothing happened", not a permanent `NotLoaded`
    /// ghost entry with possibly-wrong metadata that survives until manually
    /// deregistered. Regression test for the persist-before-load ordering
    /// bug flagged as a separate, smaller finding in
    /// the project's internal engineering log.
    #[tokio::test]
    async fn add_model_failed_load_leaves_no_trace() {
        const SMALL_BUDGET_CONFIG: &str = r#"{
            "models_dir": "/tmp/test-models",
            "device": "CPU",
            "total_vram_gb": 2.0,
            "models": {}
        }"#;
        let (mm, tmp_dir) =
            make_mm_from_file(SMALL_BUDGET_CONFIG, Arc::new(MockEngineFactory::default())).await;
        let model_dir = std::path::Path::new("/tmp/test-models/doomed-add-model");
        std::fs::create_dir_all(model_dir).unwrap();

        let err = mm
            .add_model(
                "doomed-add-model".to_owned(),
                50.0, // way over the 2.0 GB budget, nothing else resident to evict
                None,
                config::ModelPolicy::default(),
            )
            .await
            .unwrap_err();
        std::fs::remove_dir_all(model_dir).ok();

        assert!(
            err.to_string().contains("insufficient VRAM"),
            "must fail on VRAM, not some earlier validation: {err}"
        );
        assert!(
            mm.list_models()
                .into_iter()
                .all(|m| m.id != "doomed-add-model"),
            "a failed add must not leave an in-memory ghost entry"
        );

        let persisted = std::fs::read_to_string(tmp_dir.path().join("config.json")).unwrap();
        let value: serde_json::Value = serde_json::from_str(&persisted).unwrap();
        assert!(
            value["models"].get("doomed-add-model").is_none(),
            "a failed add must not persist anything to config.json"
        );
    }

    /// Two concurrent `add_model` calls racing on the SAME `model_id` must
    /// resolve deterministically: exactly one wins (registers and loads
    /// successfully), the other fails cleanly with "already registered"
    /// *before* ever reaching `load_model` — so its on-failure rollback can
    /// never fire against, and delete, the winner's live registration.
    /// Regression test for a race an independent review (Fable) caught in
    /// this fix's own rollback path: with the old separate read-then-write
    /// duplicate check, both racers could pass the check before either
    /// registered, and the loser's rollback would then delete the winner's
    /// just-succeeded registration out from under it — a `Ready` model
    /// silently vanishing from `/v1/admin/models` while its engine and VRAM
    /// stayed resident and un-deregisterable through any normal API path.
    #[tokio::test]
    async fn add_model_concurrent_same_id_one_wins_cleanly() {
        let mm = Arc::new(make_mm(&[], 22.5).await);
        let dir = std::path::Path::new("/tmp/test-models/race-model");
        std::fs::create_dir_all(dir).unwrap();

        let mm_a = Arc::clone(&mm);
        let mm_b = Arc::clone(&mm);
        let a = tokio::spawn(async move {
            mm_a.add_model(
                "race-model".to_owned(),
                1.0,
                None,
                config::ModelPolicy::default(),
            )
            .await
        });
        let b = tokio::spawn(async move {
            mm_b.add_model(
                "race-model".to_owned(),
                1.0,
                None,
                config::ModelPolicy::default(),
            )
            .await
        });
        let (ra, rb) = tokio::join!(a, b);
        std::fs::remove_dir_all(dir).ok();
        let ra = ra.unwrap();
        let rb = rb.unwrap();

        let loser = match (ra, rb) {
            (Ok(()), Err(e)) | (Err(e), Ok(())) => e,
            other => panic!("exactly one racer must win, got: {other:?}"),
        };
        assert!(
            loser.to_string().contains("already registered"),
            "the loser must fail with 'already registered', not some other error: {loser}"
        );

        // The winner's registration must have survived intact — not deleted
        // by the loser's rollback.
        assert_eq!(
            mm.read_models("test").get("race-model").unwrap().state,
            ModelState::Ready,
            "the winning racer's model must be Ready and still registered"
        );
    }

    // ---- patch_model -----------------------------------------------------

    /// A PATCH with no fields supplied at all is rejected — accepting it
    /// would be a silent 200 no-op.
    #[tokio::test]
    async fn patch_model_rejects_empty_body() {
        let (mm, _dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;
        let err = mm
            .patch_model("existing", &config::ModelEntryPatch::default())
            .unwrap_err();
        assert!(err.to_string().contains("no fields supplied"));
    }

    /// `kind` is immutable via PATCH — supplying it is rejected before any
    /// other field is even looked at (changing kind changes which engine
    /// factory builds the model; that needs a fresh registration).
    #[tokio::test]
    async fn patch_model_rejects_kind_change() {
        let (mm, _dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;
        let patch = config::ModelEntryPatch {
            kind: Some("vision".to_owned()),
            pinned: Some(true),
            ..Default::default()
        };
        let err = mm.patch_model("existing", &patch).unwrap_err();
        assert!(err.to_string().contains("immutable"));
    }

    /// A `PATCH` of an unregistered model id is a clean `ModelError::NotFound`
    /// (→ 404), not a generic error string.
    #[tokio::test]
    async fn patch_model_unknown_id_is_not_found() {
        let (mm, _dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;
        let patch = config::ModelEntryPatch {
            pinned: Some(true),
            ..Default::default()
        };
        let err = mm.patch_model("does-not-exist", &patch).unwrap_err();
        assert!(matches!(
            err.downcast_ref::<ModelError>(),
            Some(ModelError::NotFound(_))
        ));
    }

    /// The core correctness fix this feature shipped with: before
    /// `patch_model` existed, `eviction_grace_for`/`request_on_demand_load`
    /// checked `self.config.models` *first* and the dynamic overlay only as
    /// a fallback — so for a model also declared in the static startup
    /// config (which every real fleet model is), the overlay was never
    /// consulted at all, and `self.config` is never mutated after startup.
    /// A `patch_model` call on such a model would return `200 OK` and
    /// silently do nothing where it mattered. This proves the fix: a `PATCH`
    /// of `eviction_grace_secs` and `load` on "existing" (declared in
    /// `BASE_CONFIG`) is actually visible at both call sites.
    #[tokio::test]
    async fn patch_model_on_startup_declared_model_is_visible_to_overlay_checked_sites() {
        let (mm, _dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;
        let mm = Arc::new(mm);

        // Sanity: before the patch, "existing" is Eager (BASE_CONFIG sets no
        // policy at all) — on-demand load never applies.
        assert_eq!(
            mm.request_on_demand_load("existing"),
            OnDemandLoad::NotApplicable
        );
        let before = mm.eviction_grace_for("existing");
        assert_ne!(
            before,
            Duration::from_secs_f64(42.0),
            "42s must not already be the global default, or the assertion \
             below would pass even without the fix"
        );

        let patch = config::ModelEntryPatch {
            eviction_grace_secs: Some(Some(42.0)),
            load: Some(config::LoadPolicy::OnDemand),
            ..Default::default()
        };
        mm.patch_model("existing", &patch).unwrap();

        assert_eq!(
            mm.eviction_grace_for("existing"),
            Duration::from_secs_f64(42.0),
            "eviction_grace_for must see the patched override, not just the \
             config-first (never-mutated) startup snapshot"
        );
        assert_ne!(
            mm.request_on_demand_load("existing"),
            OnDemandLoad::NotApplicable,
            "request_on_demand_load must see the patched `load: on_demand`, \
             not the config-only read that predated this fix"
        );
    }

    /// `device`/`tier_preference` are placement-affecting — rejected while
    /// the model is `Ready` rather than silently leaving the live engine on
    /// its old device while the record/config disagree.
    #[tokio::test]
    async fn patch_model_rejects_device_change_while_ready() {
        let (mm, _dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;
        mm.load_model("existing").await.unwrap();

        let patch = config::ModelEntryPatch {
            device: Some(Some("CPU".to_owned())),
            ..Default::default()
        };
        let err = mm.patch_model("existing", &patch).unwrap_err();
        assert!(err.to_string().contains("evict it first"));
    }

    /// The same `device` change succeeds while `NotLoaded` — no engine to
    /// disagree with, so it rebuilds the record immediately via the same
    /// path `reload_config`'s own differs-branch uses.
    #[tokio::test]
    async fn patch_model_device_change_while_not_loaded_rebuilds_record() {
        let (mm, dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;

        let patch = config::ModelEntryPatch {
            device: Some(Some("CPU".to_owned())),
            ..Default::default()
        };
        let report = mm.patch_model("existing", &patch).unwrap();
        assert!(report.applied_live.contains(&"device"));

        let persisted = std::fs::read_to_string(dir.path().join("config.json")).unwrap();
        let value: serde_json::Value = serde_json::from_str(&persisted).unwrap();
        assert_eq!(value["models"]["existing"]["device"], "CPU");
    }

    /// A full positive-path check: persisted file, live record, and the
    /// report's field classification all agree after one call touching a
    /// next-load field (`vram_gb`) and a live field (`pinned`/`priority`).
    #[tokio::test]
    async fn patch_model_persists_and_updates_live_record_and_report() {
        let (mm, dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;

        let patch = config::ModelEntryPatch {
            vram_gb: Some(3.5),
            pinned: Some(true),
            priority: Some(7),
            ..Default::default()
        };
        let report = mm.patch_model("existing", &patch).unwrap();
        assert!(report.effective_on_next_load.contains(&"vram_gb"));
        assert!(report.applied_live.contains(&"pinned"));
        assert!(report.applied_live.contains(&"priority"));
        assert!((report.entry.vram_gb - 3.5).abs() < f64::EPSILON);

        {
            let guard = mm.read_models("test");
            let record = guard.get("existing").unwrap();
            assert!((record.vram_gb - 3.5).abs() < f64::EPSILON);
            assert!(record.pinned);
            assert_eq!(record.priority, 7);
        }

        let persisted = std::fs::read_to_string(dir.path().join("config.json")).unwrap();
        let value: serde_json::Value = serde_json::from_str(&persisted).unwrap();
        assert_eq!(value["models"]["existing"]["vram_gb"], 3.5);
        assert_eq!(value["models"]["existing"]["pinned"], true);
        assert_eq!(value["models"]["existing"]["priority"], 7);
    }

    /// Regression test for the missing config-mutation lock: two concurrent
    /// `patch_model` calls on *different* models, run on real OS threads (not
    /// just concurrent futures — `patch_model` is synchronous), must both
    /// survive in the written file. Before `config_mutation_lock` existed,
    /// each wrote to the same fixed `config.json.tmp` path with no
    /// serialisation, so one could lose the other's write entirely.
    #[test]
    fn patch_model_concurrent_patches_on_different_models_both_persist() {
        const TWO_MODEL_CONFIG: &str = r#"{
            "models_dir": "/tmp/test-models",
            "device": "CPU",
            "total_vram_gb": 0.0,
            "models": {
                "model-a": { "vram_gb": 1.0 },
                "model-b": { "vram_gb": 1.0 }
            }
        }"#;
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (mm, dir) = rt.block_on(make_mm_from_file(
            TWO_MODEL_CONFIG,
            Arc::new(MockEngineFactory::default()),
        ));
        let mm = Arc::new(mm);

        let (mm1, mm2) = (Arc::clone(&mm), Arc::clone(&mm));
        let t1 = std::thread::spawn(move || {
            mm1.patch_model(
                "model-a",
                &config::ModelEntryPatch {
                    priority: Some(7),
                    ..Default::default()
                },
            )
        });
        let t2 = std::thread::spawn(move || {
            mm2.patch_model(
                "model-b",
                &config::ModelEntryPatch {
                    priority: Some(9),
                    ..Default::default()
                },
            )
        });
        t1.join().unwrap().unwrap();
        t2.join().unwrap().unwrap();

        let persisted = std::fs::read_to_string(dir.path().join("config.json")).unwrap();
        let value: serde_json::Value = serde_json::from_str(&persisted).unwrap();
        assert_eq!(value["models"]["model-a"]["priority"], 7);
        assert_eq!(value["models"]["model-b"]["priority"], 9);
    }

    /// Same as the different-models test above, but on the *same* model —
    /// the case `config_mutation_lock` being held for `patch_model`'s whole
    /// body (not just the file write) specifically exists for: without it,
    /// two concurrent `PATCH`es on one model could each merge their change
    /// against the same stale pre-lock read of "current effective entry",
    /// and the second write would silently discard the first's field.
    #[test]
    fn patch_model_concurrent_patches_on_the_same_model_dont_lose_either_change() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (mm, dir) = rt.block_on(make_mm_from_file(
            BASE_CONFIG,
            Arc::new(MockEngineFactory::default()),
        ));
        let mm = Arc::new(mm);
        let barrier = Arc::new(std::sync::Barrier::new(2));

        let (mm1, mm2) = (Arc::clone(&mm), Arc::clone(&mm));
        let (b1, b2) = (Arc::clone(&barrier), Arc::clone(&barrier));
        let t1 = std::thread::spawn(move || {
            b1.wait();
            mm1.patch_model(
                "existing",
                &config::ModelEntryPatch {
                    priority: Some(7),
                    ..Default::default()
                },
            )
        });
        let t2 = std::thread::spawn(move || {
            b2.wait();
            mm2.patch_model(
                "existing",
                &config::ModelEntryPatch {
                    pinned: Some(true),
                    ..Default::default()
                },
            )
        });
        t1.join().unwrap().unwrap();
        t2.join().unwrap().unwrap();

        let persisted = std::fs::read_to_string(dir.path().join("config.json")).unwrap();
        let value: serde_json::Value = serde_json::from_str(&persisted).unwrap();
        assert_eq!(
            value["models"]["existing"]["priority"], 7,
            "first patch's field must survive"
        );
        assert_eq!(
            value["models"]["existing"]["pinned"], true,
            "second patch's field must survive"
        );
    }

    /// `Loading`/`Evicting` reject the whole call (409) — a state PATCH must
    /// never disturb a load or eviction already in flight.
    #[tokio::test]
    async fn patch_model_rejects_loading_state() {
        let (mm, _dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;
        mm.write_models("test").get_mut("existing").unwrap().state = ModelState::Loading;
        let err = mm
            .patch_model(
                "existing",
                &config::ModelEntryPatch {
                    pinned: Some(true),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(
            err.downcast_ref::<ModelError>(),
            Some(ModelError::Loading)
        ));
    }

    #[tokio::test]
    async fn patch_model_rejects_evicting_state() {
        let (mm, _dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;
        mm.write_models("test").get_mut("existing").unwrap().state = ModelState::Evicting;
        let err = mm
            .patch_model(
                "existing",
                &config::ModelEntryPatch {
                    pinned: Some(true),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(
            err.downcast_ref::<ModelError>(),
            Some(ModelError::Evicting)
        ));
    }

    /// Regression test for the persist-before-validate ordering bug a review
    /// found: a `PATCH` with a bad `tier_preference` label must fail *and*
    /// leave `config.json` untouched. Before the fix, `write_config_json`
    /// ran before `build_not_loaded_record` — the thing that actually
    /// validates tier labels, since `validate_model_entry` doesn't cover
    /// them — so a rejected call still wrote the bad label to disk,
    /// producing a file that would fail `Config::validate` on the very next
    /// boot even though this call itself returned an error. Needs a
    /// *non-empty* device inventory: with an empty one,
    /// `resolve_model_device`'s "GPU-free tests" branch never even parses
    /// `tier_preference`, so this can't reuse `make_mm_from_file`'s default.
    #[tokio::test]
    async fn patch_model_rejects_bad_tier_preference_without_persisting() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg_path = dir.path().join("config.json");
        std::fs::write(&cfg_path, BASE_CONFIG).expect("write temp config");
        let config = Config::load(&cfg_path).expect("parse temp config");
        let inventory = Arc::new(DeviceInventory::from_devices(vec![dev(
            "CPU",
            DeviceKind::Cpu,
        )]));
        let mm = ModelManager::new_with_inventory(
            config,
            Arc::new(MockEngineFactory::default()),
            inventory,
        )
        .await
        .expect("ModelManager::new_with_inventory failed in test")
        .with_config_path(cfg_path);

        let before = std::fs::read_to_string(dir.path().join("config.json")).unwrap();
        let patch = config::ModelEntryPatch {
            tier_preference: Some(vec!["not-a-real-tier".to_owned()]),
            ..Default::default()
        };
        let err = mm.patch_model("existing", &patch).unwrap_err();
        assert!(
            err.to_string().contains("unknown tier"),
            "unexpected error: {err}"
        );

        let after = std::fs::read_to_string(dir.path().join("config.json")).unwrap();
        assert_eq!(
            before, after,
            "a rejected placement-path PATCH must not touch config.json"
        );
    }

    /// List-valued fields (`tier_preference`/`capabilities`) clear via an
    /// explicit empty list, and a nullable scalar field clears via explicit
    /// `null` — both distinct from the field being merely absent from the
    /// PATCH body (which leaves the current value untouched).
    #[tokio::test]
    async fn patch_model_clears_list_fields_and_nullable_device() {
        let (mm, dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;

        // First set them to something, so clearing is an observable change.
        mm.patch_model(
            "existing",
            &config::ModelEntryPatch {
                capabilities: Some(vec!["tool_calling".to_owned()]),
                device: Some(Some("CPU".to_owned())),
                ..Default::default()
            },
        )
        .unwrap();

        let report = mm
            .patch_model(
                "existing",
                &config::ModelEntryPatch {
                    capabilities: Some(vec![]),
                    device: Some(None),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(report.entry.policy.capabilities.is_empty());
        assert!(report.entry.policy.device.is_none());

        let persisted = std::fs::read_to_string(dir.path().join("config.json")).unwrap();
        let value: serde_json::Value = serde_json::from_str(&persisted).unwrap();
        assert_eq!(
            value["models"]["existing"]["capabilities"],
            serde_json::json!([])
        );
        assert_eq!(
            value["models"]["existing"]["device"],
            serde_json::Value::Null
        );
    }

    // ---- set_preload -------------------------------------------------------

    /// An explicit `preload` list naming an unregistered model id is
    /// rejected — writing it would fail `Config::validate`'s own invariant
    /// on the very next boot.
    #[tokio::test]
    async fn set_preload_explicit_rejects_unknown_id() {
        let (mm, _dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;
        let err = mm
            .set_preload(PreloadSource::Explicit(vec!["nope".to_owned()]))
            .unwrap_err();
        assert!(err.to_string().contains("nope"));
    }

    #[tokio::test]
    async fn set_preload_explicit_rejects_duplicate_id() {
        let (mm, _dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;
        let err = mm
            .set_preload(PreloadSource::Explicit(vec![
                "existing".to_owned(),
                "existing".to_owned(),
            ]))
            .unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }

    #[tokio::test]
    async fn set_preload_explicit_persists_and_reports_added() {
        let (mm, dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;
        let report = mm
            .set_preload(PreloadSource::Explicit(vec!["existing".to_owned()]))
            .unwrap();
        assert_eq!(report.preload, vec!["existing".to_owned()]);
        assert!(report.previous.is_empty());
        assert_eq!(report.added, vec!["existing".to_owned()]);
        assert!(report.removed.is_empty());

        let persisted = std::fs::read_to_string(dir.path().join("config.json")).unwrap();
        let value: serde_json::Value = serde_json::from_str(&persisted).unwrap();
        assert_eq!(value["preload"], serde_json::json!(["existing"]));
    }

    /// `from_live` snapshots only `Ready` models, reporting the rest as
    /// `skipped_not_ready` rather than as an error — a preload list only
    /// ever describes future boot state, so "not loaded right now" is
    /// completely normal, not a failure.
    #[tokio::test]
    async fn set_preload_from_live_excludes_not_ready_models() {
        const TWO_MODEL_CONFIG: &str = r#"{
            "models_dir": "/tmp/test-models",
            "device": "CPU",
            "total_vram_gb": 0.0,
            "models": {
                "loaded-one": { "vram_gb": 1.0 },
                "not-loaded-one": { "vram_gb": 1.0 }
            }
        }"#;
        let (mm, _dir) =
            make_mm_from_file(TWO_MODEL_CONFIG, Arc::new(MockEngineFactory::default())).await;
        mm.load_model("loaded-one").await.unwrap();

        let report = mm.set_preload(PreloadSource::FromLive).unwrap();
        assert_eq!(report.preload, vec!["loaded-one".to_owned()]);
        assert_eq!(report.skipped_not_ready, vec!["not-loaded-one".to_owned()]);
    }

    /// `from_live` must reject (not silently snapshot) a `Ready` model that
    /// has no `models` entry in the file being written — the documented
    /// `add_model` edge case where a load succeeds but the subsequent
    /// persist fails. Snapshotting it into `preload` would produce a file
    /// that fails `Config::validate`'s own invariant on the very next boot.
    #[tokio::test]
    async fn set_preload_from_live_rejects_a_ready_model_missing_from_the_file() {
        let (mm, dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;
        let model_dir = std::path::Path::new("/tmp/test-models/ghost-model");
        std::fs::create_dir_all(model_dir).unwrap();
        mm.add_model(
            "ghost-model".to_owned(),
            1.0,
            None,
            config::ModelPolicy::default(),
        )
        .await
        .unwrap();
        std::fs::remove_dir_all(model_dir).ok();
        // Simulate the file having lost the stanza (hand-edit, or the
        // documented add_model persist-can-fail-after-load-succeeds case)
        // while the live registry still has it Ready.
        std::fs::write(dir.path().join("config.json"), BASE_CONFIG).unwrap();

        let err = mm.set_preload(PreloadSource::FromLive).unwrap_err();
        assert!(err.to_string().contains("ghost-model"));
    }

    /// Same bug, the other registration path: a model in the *static startup*
    /// config (`new_with_inventory`'s registration loop, not `add_model`)
    /// with no explicit `kind` must also be correctly auto-detected before
    /// its first load, not defaulted to `TextGen`.
    #[tokio::test]
    async fn startup_registration_auto_detects_kind_instead_of_defaulting_to_text_gen() {
        // The "vlm" segment makes MockEngineFactory classify this as Vision.
        let mm = make_mm(&[("qwen-vlm-8b", 6.0)], 22.5).await;

        // Before any load_model() call: list_models() must already report the
        // correct kind, proving the fix lives in registration, not load.
        let info = mm
            .list_models()
            .into_iter()
            .find(|m| m.id == "qwen-vlm-8b")
            .expect("model registered");
        assert_eq!(
            info.kind,
            ModelKind::Vision,
            "configured_kind must reflect the real detected kind, not a TextGen default"
        );
    }

    /// Phase 5.1c: the `Stt` kind now flows end-to-end through the manager
    /// (register → load → metrics → resolve → transcribe → reject-other →
    /// evict), exactly like the embedding kind — a real channel-backed handle,
    /// not a zero-capacity stub. (`get_handle`/`get_embedding_handle` reject it:
    /// it is neither a text nor an embedding model.)
    #[tokio::test]
    async fn stt_kind_flows_through_the_registry_with_no_manager_edits() {
        // The "stt" segment makes MockEngineFactory classify this as Stt.
        let mm = make_mm(&[("whisper-stt-ov", 1.5)], 22.5).await;
        mm.load_model("whisper-stt-ov").await.unwrap();

        let snap = mm.metrics_snapshot();
        let m = &snap.models[0];
        assert_eq!(m.kind, ModelKind::Stt);
        assert_eq!(
            m.max_seqs,
            crate::pipelines::stt::STT_CHANNEL_CAP,
            "a real STT engine reports its channel cap"
        );

        // The temporary ChatContext is dropped at the end of this statement —
        // holding a clone past the evict below would keep the engine thread's
        // Sender alive and block the quiesce join.
        assert!(matches!(
            mm.get_chat_context("whisper-stt-ov").unwrap().handle,
            EngineHandleKind::Stt(_)
        ));

        // The Ready record resolves to the Stt handle, and it actually
        // transcribes (the mock answers with a canned transcription).
        let handle = mm.get_stt_handle("whisper-stt-ov").unwrap();
        let out = handle
            .transcribe(b"fake-audio".to_vec(), None, true)
            .await
            .unwrap();
        assert_eq!(out.text, "This is a mock transcription.");
        assert_eq!(out.segments.len(), 2);

        // Text-only and embedding endpoints reject it.
        assert!(matches!(
            mm.get_handle("whisper-stt-ov").unwrap_err(),
            ModelError::WrongKind(_)
        ));
        assert!(matches!(
            mm.get_embedding_handle("whisper-stt-ov").unwrap_err(),
            ModelError::WrongKind(_)
        ));

        // Drop our handle clone BEFORE evicting (the quiesce protocol — eviction
        // joins the engine thread, which exits only once every handle is dropped).
        drop(handle);

        mm.evict_model("whisper-stt-ov").await.unwrap();
        assert!(
            mm.metrics_snapshot().used_vram_gb.abs() < 1e-9,
            "alloc==free for the STT kind too"
        );
    }

    /// Phase 5.3c: the `ImageGen` kind now flows end-to-end through the manager
    /// (register → load → metrics → resolve → generate → reject-other → evict),
    /// exactly like the STT kind — a real channel-backed handle, not a
    /// zero-capacity stub.
    #[tokio::test]
    async fn image_kind_flows_through_the_registry_with_no_manager_edits() {
        // The "imagegen" segment makes MockEngineFactory classify this as ImageGen.
        let mm = make_mm(&[("sdxl-imagegen-ov", 6.5)], 22.5).await;
        mm.load_model("sdxl-imagegen-ov").await.unwrap();

        let snap = mm.metrics_snapshot();
        let m = &snap.models[0];
        assert_eq!(m.kind, ModelKind::ImageGen);
        assert_eq!(
            m.max_seqs,
            crate::pipelines::image::IMAGE_CHANNEL_CAP,
            "a real image engine reports its channel cap"
        );

        assert!(matches!(
            mm.get_chat_context("sdxl-imagegen-ov").unwrap().handle,
            EngineHandleKind::ImageGen(_)
        ));

        // The Ready record resolves to the ImageGen handle, and it actually
        // generates (the mock answers with n canned PNGs).
        let handle = mm.get_image_handle("sdxl-imagegen-ov").unwrap();
        let opts = crate::ov_image::ImageGenOptions {
            negative_prompt: None,
            width: 512,
            height: 512,
            num_inference_steps: 20,
            num_images: 2,
            seed: None,
            guidance_scale: 0.0,
        };
        let pngs = handle
            .generate("a red circle".to_owned(), opts)
            .await
            .unwrap();
        assert_eq!(pngs.len(), 2);
        assert_eq!(&pngs[0][..8], b"\x89PNG\r\n\x1a\n");

        // Text-only and STT endpoints reject it.
        assert!(matches!(
            mm.get_handle("sdxl-imagegen-ov").unwrap_err(),
            ModelError::WrongKind(_)
        ));
        assert!(matches!(
            mm.get_stt_handle("sdxl-imagegen-ov").unwrap_err(),
            ModelError::WrongKind(_)
        ));

        // Drop our handle clone BEFORE evicting (the quiesce protocol).
        drop(handle);

        mm.evict_model("sdxl-imagegen-ov").await.unwrap();
        assert!(
            mm.metrics_snapshot().used_vram_gb.abs() < 1e-9,
            "alloc==free for the image kind too"
        );
    }

    /// Known model in `NotLoaded` state → `NotLoaded` error.
    #[tokio::test]
    async fn test_get_handle_not_loaded() {
        let mm = make_mm(&[("qwen3-8b", 5.5)], 22.5).await;
        let err = mm.get_handle("qwen3-8b").unwrap_err();
        assert_eq!(err, ModelError::NotLoaded);
    }

    /// Known model in Loading state → Loading error.
    #[tokio::test]
    async fn test_get_handle_loading() {
        let mm = make_mm(&[("qwen3-8b", 5.5)], 22.5).await;
        // Force state to Loading manually.
        mm.write_models("test").get_mut("qwen3-8b").unwrap().state = ModelState::Loading;
        let err = mm.get_handle("qwen3-8b").unwrap_err();
        assert_eq!(err, ModelError::Loading);
    }

    /// Known model in Ready state → Ok(handle).
    #[tokio::test]
    async fn test_get_handle_ready() {
        let mm = make_mm(&[("qwen3-8b", 5.5)], 22.5).await;
        mm.load_model("qwen3-8b").await.unwrap();
        let result = mm.get_handle("qwen3-8b");
        assert!(
            result.is_ok(),
            "Ready model must return a handle: {result:?}"
        );
    }

    /// Known model in Evicting state → Evicting error.
    #[tokio::test]
    async fn test_get_handle_evicting() {
        let mm = make_mm(&[("qwen3-8b", 5.5)], 22.5).await;
        mm.load_model("qwen3-8b").await.unwrap();
        // Force state to Evicting without actually evicting.
        mm.write_models("test").get_mut("qwen3-8b").unwrap().state = ModelState::Evicting;
        let err = mm.get_handle("qwen3-8b").unwrap_err();
        assert_eq!(err, ModelError::Evicting);
    }

    /// `get_handle` updates `last_used` (needed for LRU ordering).
    #[tokio::test]
    async fn test_get_handle_updates_last_used() {
        let mm = make_mm(&[("qwen3-8b", 5.5)], 22.5).await;
        mm.load_model("qwen3-8b").await.unwrap();

        let before = mm.read_models("test").get("qwen3-8b").unwrap().last_used;
        std::thread::sleep(std::time::Duration::from_millis(5));
        mm.get_handle("qwen3-8b").unwrap();
        let after = mm.read_models("test").get("qwen3-8b").unwrap().last_used;

        assert!(
            after > before,
            "last_used must advance on each get_handle call"
        );
    }

    /// After load, `get_chat_context` returns the handle plus a usable template
    /// and a detected family. (The test model dir does not exist, so the
    /// generic fallback template is used — family Default.)
    #[tokio::test]
    async fn test_get_chat_context_after_load() {
        let mm = make_mm(&[("qwen3-8b", 5.5)], 22.5).await;
        mm.load_model("qwen3-8b").await.unwrap();
        let ctx = mm.get_chat_context("qwen3-8b").expect("Ready model");
        assert_eq!(ctx.family, ModelFamily::Default);
        assert!(
            !ctx.template.is_empty(),
            "template must be populated on load"
        );
    }

    /// `get_chat_context` surfaces the same errors as `get_handle`.
    #[tokio::test]
    async fn test_get_chat_context_not_loaded() {
        let mm = make_mm(&[("qwen3-8b", 5.5)], 22.5).await;
        assert!(matches!(
            mm.get_chat_context("qwen3-8b"),
            Err(ModelError::NotLoaded)
        ));
    }

    // ---- load_model -------------------------------------------------------

    /// Loading a model transitions it to Ready and makes its handle available.
    #[tokio::test]
    async fn test_load_model_transitions_to_ready() {
        let mm = make_mm(&[("qwen3-8b", 5.5)], 22.5).await;
        mm.load_model("qwen3-8b").await.unwrap();

        let state = mm.read_models("test").get("qwen3-8b").unwrap().state;
        assert_eq!(state, ModelState::Ready);
        assert!(
            mm.get_handle("qwen3-8b").is_ok(),
            "model must be reachable via get_handle after load"
        );
    }

    /// Loading a model that doesn't exist in the registry → `NotFound` error.
    #[tokio::test]
    async fn test_load_model_unknown_returns_error() {
        let mm = make_mm(&[], 22.5).await;
        let err = mm.load_model("ghost-model").await.unwrap_err();
        let model_err = err
            .downcast_ref::<ModelError>()
            .expect("must be ModelError");
        assert_eq!(*model_err, ModelError::NotFound("ghost-model".to_owned()));
    }

    /// Loading an already-Ready model is idempotent (no error).
    #[tokio::test]
    async fn test_load_model_idempotent_when_already_ready() {
        let mm = make_mm(&[("qwen3-8b", 5.5)], 22.5).await;
        mm.load_model("qwen3-8b").await.unwrap();
        // Second call must succeed without error.
        mm.load_model("qwen3-8b").await.unwrap();
        let state = mm.read_models("test").get("qwen3-8b").unwrap().state;
        assert_eq!(state, ModelState::Ready);
    }

    /// If the engine factory fails, the model resets to `NotLoaded`.
    #[tokio::test]
    async fn test_load_model_factory_failure_resets_to_not_loaded() {
        let config = make_config(&[("bad-model", 5.5)], 22.5);
        let factory = Arc::new(MockEngineFactory {
            should_fail: true,
            ..Default::default()
        });
        let mm = ModelManager::new(config, factory).await.unwrap();

        let err = mm.load_model("bad-model").await;
        assert!(err.is_err(), "factory failure must propagate as error");

        let state = mm.read_models("test").get("bad-model").unwrap().state;
        assert_eq!(
            state,
            ModelState::NotLoaded,
            "state must reset to NotLoaded on failure"
        );
    }

    /// 2026-07-14 incident regression: a `preload` model whose load fails
    /// (e.g. a `CL_OUT_OF_RESOURCES`-class engine failure) must not abort
    /// startup — `new_with_inventory` should still return `Ok`, with the
    /// failed model left `NotLoaded`, mirroring on-demand's graceful
    /// handling of the identical error path instead of taking the whole
    /// process down with it.
    #[tokio::test]
    async fn test_preload_failure_does_not_abort_startup() {
        let mut config = make_config(&[("bad-model", 5.5)], 22.5);
        config.preload = vec!["bad-model".to_owned()];
        let factory = Arc::new(MockEngineFactory {
            fail_with_oom: true,
            ..Default::default()
        });

        let mm = ModelManager::new(config, factory)
            .await
            .expect("a preload load failure must not fail startup");

        let state = mm.read_models("test").get("bad-model").unwrap().state;
        assert_eq!(
            state,
            ModelState::NotLoaded,
            "the failed preload model must be left NotLoaded, not panic/abort"
        );
    }

    // ---- evict_model ------------------------------------------------------

    /// Evicting a `Ready` model transitions it to `NotLoaded` and frees VRAM.
    #[tokio::test]
    async fn test_evict_model_ready_transitions_to_not_loaded() {
        let mm = make_mm(&[("qwen3-8b", 5.5)], 22.5).await;
        mm.load_model("qwen3-8b").await.unwrap();

        // With dynamic KV sizing (margin=0, min_kv=0, cap=0 in test config):
        // kv_gb = 22.5 - 5.5 - 0.0 = 17.0. Tracker = weight + kv = 22.5 GB → free = 0.
        let free_before = mm.vram.read().unwrap().free_gb(&mm.inference_domain);
        assert!(
            (free_before - 0.0).abs() < 0.01,
            "dynamic KV uses all remaining VRAM: expected 0 GB free, got {free_before:.2}"
        );

        mm.evict_model("qwen3-8b").await.unwrap();

        let state = mm.read_models("test").get("qwen3-8b").unwrap().state;
        assert_eq!(state, ModelState::NotLoaded);

        // VRAM must be restored.
        let free_after = mm.vram.read().unwrap().free_gb(&mm.inference_domain);
        assert!(
            (free_after - 22.5).abs() < 0.01,
            "VRAM must be fully restored after eviction"
        );
    }

    /// T6.1: `shutdown` evicts every Ready model (full quiesce per model),
    /// restores all VRAM, and skips not-loaded models without erroring.
    #[tokio::test]
    async fn test_shutdown_evicts_all_ready_models() {
        // Co-resident pair (VLM first — a text model's dynamic KV grabs all
        // remaining VRAM, so it must load second) + one never-loaded model.
        let mm = make_mm(&[("vlm-a", 4.0), ("model-b", 4.0), ("model-c", 4.0)], 22.5).await;
        mm.load_model("vlm-a").await.unwrap();
        mm.load_model("model-b").await.unwrap();

        mm.shutdown().await;

        let guard = mm.read_models("test");
        for id in ["vlm-a", "model-b", "model-c"] {
            assert_eq!(
                guard.get(id).unwrap().state,
                ModelState::NotLoaded,
                "{id} must be NotLoaded after shutdown"
            );
        }
        drop(guard);

        let free = mm.vram.read().unwrap().free_gb(&mm.inference_domain);
        assert!(
            (free - 22.5).abs() < 0.01,
            "all VRAM must be restored after shutdown: got {free:.2} GB free"
        );
    }

    /// Evicting a model that is not loaded returns an error.
    #[tokio::test]
    async fn test_evict_model_not_loaded_returns_error() {
        let mm = make_mm(&[("qwen3-8b", 5.5)], 22.5).await;
        let err = mm.evict_model("qwen3-8b").await.unwrap_err();
        assert!(
            err.to_string().contains("not loaded"),
            "error message must mention 'not loaded': {err}"
        );
    }

    /// Evicting a model ID not in the registry returns an error.
    #[tokio::test]
    async fn test_evict_model_unknown_returns_error() {
        let mm = make_mm(&[], 22.5).await;
        let err = mm.evict_model("ghost").await.unwrap_err();
        let model_err = err
            .downcast_ref::<ModelError>()
            .expect("must be ModelError");
        assert_eq!(*model_err, ModelError::NotFound("ghost".to_owned()));
    }

    // ---- LRU eviction ----------------------------------------------------

    /// When VRAM is tight, loading a new model evicts the LRU Ready model.
    #[tokio::test]
    async fn test_load_evicts_lru_when_vram_insufficient() {
        // 10 GB total; model-a = 6 GB, model-b = 6 GB (both don't fit).
        let mm = make_mm(&[("model-a", 6.0), ("model-b", 6.0)], 10.0).await;

        mm.load_model("model-a").await.unwrap();

        // model-a is now using 6 GB; model-b needs 6 GB but only 4 GB free.
        // Expectation: model-a is evicted (LRU) → model-b loads.
        mm.load_model("model-b").await.unwrap();

        let state_a = mm.read_models("test").get("model-a").unwrap().state;
        let state_b = mm.read_models("test").get("model-b").unwrap().state;

        assert_eq!(
            state_a,
            ModelState::NotLoaded,
            "model-a must have been evicted"
        );
        assert_eq!(
            state_b,
            ModelState::Ready,
            "model-b must be Ready after LRU eviction"
        );
    }

    /// T5.4/#37: a busy (in-flight) model is never the eviction victim while a
    /// genuinely idle model exists — even when the busy model is the LEAST
    /// recently used by `last_used`. Naive LRU evicts the active stream; the
    /// in-flight-aware selection must spare it and take the idle model instead.
    #[tokio::test]
    async fn test_lru_spares_busy_model_for_idle_one() {
        // cache_size_gb caps each KV pool at 2 GB so two 5 GB models co-reside
        // (5+2 each = 14 GB on a 17 GB card); a third 5 GB load then leaves only
        // 3 GB free and must evict exactly one model.
        let mut config = make_config(
            &[("model-a", 5.0), ("model-b", 5.0), ("model-c", 5.0)],
            17.0,
        );
        config.cache_size_gb = 2.0;
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();

        // Load a first (older last_used), then b (newer). Both Ready.
        mm.load_model("model-a").await.unwrap();
        mm.load_model("model-b").await.unwrap();

        // Make the OLDER model (a) busy — hold one in-flight slot. Pure LRU would
        // evict a (oldest); the in-flight-aware selection must skip it.
        let _permit = {
            let guard = mm.read_models("test");
            let handle = guard.get("model-a").unwrap().handle.as_ref().unwrap();
            match handle {
                EngineHandleKind::TextGen(h) => h.occupy_slot_for_test().unwrap(),
                other => panic!("model-a must be a TextGen mock, got {other:?}"),
            }
        };

        // Loading c forces one eviction.
        mm.load_model("model-c").await.unwrap();

        let state_a = mm.read_models("test").get("model-a").unwrap().state;
        let state_b = mm.read_models("test").get("model-b").unwrap().state;
        let state_c = mm.read_models("test").get("model-c").unwrap().state;
        assert_eq!(
            state_a,
            ModelState::Ready,
            "busy model-a (oldest) must NOT be evicted"
        );
        assert_eq!(
            state_b,
            ModelState::NotLoaded,
            "idle model-b is the correct victim"
        );
        assert_eq!(state_c, ModelState::Ready, "model-c loads after eviction");
    }

    /// No models loaded at all is the trivially-idle case — the gate the
    /// OV-cache background sweep (the project's internal engineering log)
    /// checks before it has anything to warm.
    #[tokio::test]
    async fn is_idle_true_with_no_models_loaded() {
        let config = make_config(&[("model-a", 5.0)], 17.0);
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();
        assert!(mm.is_idle());
    }

    /// A `Ready` model with zero in-flight requests is still idle — loading a
    /// model alone doesn't count as activity.
    #[tokio::test]
    async fn is_idle_true_with_a_ready_but_unoccupied_model() {
        let config = make_config(&[("model-a", 5.0)], 17.0);
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();
        mm.load_model("model-a").await.unwrap();
        assert!(mm.is_idle());
    }

    /// A held in-flight slot on any single resident model makes the whole
    /// server non-idle — the sweep must not warm a second model while the
    /// first is serving live traffic.
    #[tokio::test]
    async fn is_idle_false_when_any_ready_model_has_an_in_flight_request() {
        // cache_size_gb caps each KV pool at 2 GB so both 5 GB models co-reside
        // (5+2 each = 14 GB on a 17 GB card) — the default (dynamic KV sizing)
        // would otherwise evict model-a to make room for model-b.
        let mut config = make_config(&[("model-a", 5.0), ("model-b", 5.0)], 17.0);
        config.cache_size_gb = 2.0;
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();
        mm.load_model("model-a").await.unwrap();
        mm.load_model("model-b").await.unwrap();

        let _permit = {
            let guard = mm.read_models("test");
            let handle = guard.get("model-a").unwrap().handle.as_ref().unwrap();
            match handle {
                EngineHandleKind::TextGen(h) => h.occupy_slot_for_test().unwrap(),
                other => panic!("model-a must be a TextGen mock, got {other:?}"),
            }
        };

        assert!(
            !mm.is_idle(),
            "model-a's held slot must make the server busy"
        );
    }

    // ---- OV-cache background sweep (dev/plans/ov-cache-self-management.md) ----

    /// `run_cache_sweep_once_blocking` is a complete no-op on the default
    /// config shape: `ov_cache_max_gb: 0.0` (default, unbounded) and
    /// `ov_cache_dir: None` (default, unset). No models configured, so nothing
    /// for either pass to do; the test only needs to confirm it doesn't
    /// panic and creates nothing.
    #[tokio::test]
    async fn cache_sweep_is_a_no_op_on_default_config() {
        let config = make_config(&[], 17.0);
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();
        mm.run_cache_sweep_once_blocking();
    }

    /// Precompute writes a `model_hash` manifest entry for a configured
    /// `ImageGen` model, purely from its on-disk backbone file — no load
    /// needed, no engine involved (the mock factory's `model_exists` returns
    /// `true` unconditionally, but the manifest write itself does real
    /// filesystem I/O against a real backbone this test creates).
    #[tokio::test]
    async fn precompute_writes_model_hash_for_configured_image_gen_model() {
        let models_dir = tempfile::tempdir().expect("models tempdir");
        let cache_root = tempfile::tempdir().expect("cache root tempdir");
        std::fs::create_dir_all(models_dir.path().join("sdxl-imagegen-test/unet"))
            .expect("mkdir backbone dir");
        std::fs::write(
            models_dir
                .path()
                .join("sdxl-imagegen-test/unet/openvino_model.bin"),
            b"pretend-backbone-weights",
        )
        .expect("write backbone");

        let mut config = make_config(&[("sdxl-imagegen-test", 5.0)], 17.0);
        config.models_dir = models_dir.path().to_path_buf();
        config.ov_cache_dir = Some(
            cache_root
                .path()
                .join("ov_cache")
                .to_string_lossy()
                .into_owned(),
        );
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();

        mm.run_cache_sweep_once_blocking();

        let manifest_root = crate::cache_manifest::manifest_root(Some(
            &cache_root.path().join("ov_cache").to_string_lossy(),
        ))
        .expect("manifest root computed");
        let entry = crate::cache_manifest::read(&manifest_root, "sdxl-imagegen-test")
            .expect("manifest written")
            .model_hash
            .expect("model_hash entry present");
        assert_eq!(entry.backbone_relpath, "unet/openvino_model.bin");
    }

    /// Prune deletes `.blob` files but never `.cl_cache`/`.onednn.cl_cache` —
    /// those are shared `OpenCL`/oneDNN kernel caches, not per-model, so
    /// pruning them would degrade every other model's next compile for
    /// space that isn't actually the driver of disk usage (2026-07-28
    /// plan-file finding).
    #[tokio::test]
    async fn prune_never_deletes_cl_cache_files() {
        let cache_dir = tempfile::tempdir().expect("cache dir tempdir");
        // One 2MB orphan .blob (over budget alone) plus a .cl_cache file of
        // the same size — if cl_cache were eligible, deleting it would also
        // clear the budget, masking the bug this test guards against.
        std::fs::write(
            cache_dir.path().join("orphan.blob"),
            vec![0_u8; 2 * 1024 * 1024],
        )
        .expect("write orphan blob");
        std::fs::write(
            cache_dir.path().join("shared.cl_cache"),
            vec![0_u8; 2 * 1024 * 1024],
        )
        .expect("write cl_cache file");

        let mut config = make_config(&[], 17.0);
        config.ov_cache_dir = Some(cache_dir.path().to_string_lossy().into_owned());
        config.ov_cache_max_gb = 0.001; // ~1MB cap — both files together are well over
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();

        mm.run_cache_sweep_once_blocking();

        assert!(
            !cache_dir.path().join("orphan.blob").exists(),
            "orphan .blob over budget must be deleted"
        );
        assert!(
            cache_dir.path().join("shared.cl_cache").exists(),
            ".cl_cache must never be deleted by prune"
        );
    }

    /// A blob attributed to a currently-`Ready` model is never deleted, even
    /// if the cache stays over budget as a result — pruning must not force a
    /// live model's next load to recompile.
    #[tokio::test]
    async fn prune_never_deletes_a_ready_models_blob() {
        let cache_dir = tempfile::tempdir().expect("cache dir tempdir");
        std::fs::write(
            cache_dir.path().join("resident.blob"),
            vec![0_u8; 2 * 1024 * 1024],
        )
        .expect("write resident blob");

        let mut config = make_config(&[("model-a", 5.0)], 17.0);
        config.ov_cache_dir = Some(cache_dir.path().to_string_lossy().into_owned());
        config.ov_cache_max_gb = 0.001; // over budget on its own
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();
        mm.load_model("model-a").await.unwrap();

        let manifest_root =
            crate::cache_manifest::manifest_root(config_ov_cache_dir(&mm)).expect("manifest root");
        crate::cache_manifest::write_device_blobs_entry(
            &manifest_root,
            "model-a",
            "GPU",
            &["resident.blob".to_owned()],
        );

        mm.run_cache_sweep_once_blocking();

        assert!(
            cache_dir.path().join("resident.blob").exists(),
            "a Ready model's blob must survive pruning even over budget"
        );
    }

    /// A blob attributed to a model that is currently `Loading` must survive
    /// pruning, exactly like a `Ready` model's blob — the model is actively
    /// mid-compile and may be reading/reusing this exact file right now.
    /// `Loading` is neither `Ready` (so it isn't excluded from `candidates` by
    /// the `ready_models` filter) nor is the owner "not in config" (it's
    /// actively loading a configured model), so it fell into tier 1
    /// ("configured, just not currently resident") — a *higher* deletion
    /// priority than unattributed legacy junk (tier 2), the opposite of the
    /// intended protection. Found by code review (2026-08-03), confirmed here
    /// before the fix in `prune_ov_cache_if_over_budget`.
    #[tokio::test]
    async fn prune_never_deletes_a_loading_models_blob() {
        let cache_dir = tempfile::tempdir().expect("cache dir tempdir");
        std::fs::write(
            cache_dir.path().join("loading.blob"),
            vec![0_u8; 2 * 1024 * 1024],
        )
        .expect("write loading blob");

        let mut config = make_config(&[("model-a", 5.0)], 17.0);
        config.ov_cache_dir = Some(cache_dir.path().to_string_lossy().into_owned());
        config.ov_cache_max_gb = 0.001; // over budget on its own
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();

        let manifest_root =
            crate::cache_manifest::manifest_root(config_ov_cache_dir(&mm)).expect("manifest root");
        crate::cache_manifest::write_device_blobs_entry(
            &manifest_root,
            "model-a",
            "GPU",
            &["loading.blob".to_owned()],
        );

        // Put "model-a" into Loading without completing the load — begin_load
        // is exactly phase 1 of load_model, leaving the record mid-compile.
        mm.begin_load("model-a").expect("begin_load");

        mm.run_cache_sweep_once_blocking();

        assert!(
            cache_dir.path().join("loading.blob").exists(),
            "a Loading model's blob must survive pruning — it may be reading \
             this exact file right now"
        );
    }

    /// Priority ordering: a blob owned by a model no longer in config is
    /// deleted before a blob owned by a configured-but-not-`Ready` model,
    /// which is deleted before an unattributed blob — each only as needed to
    /// get back under budget, not all at once.
    #[tokio::test]
    async fn prune_prioritizes_not_in_config_over_not_ready_over_unattributed() {
        let cache_dir = tempfile::tempdir().expect("cache dir tempdir");
        let blob_size = 2 * 1024 * 1024; // 2MB each
        std::fs::write(cache_dir.path().join("gone.blob"), vec![0_u8; blob_size])
            .expect("write gone.blob");
        std::fs::write(
            cache_dir.path().join("not-ready.blob"),
            vec![0_u8; blob_size],
        )
        .expect("write not-ready.blob");
        std::fs::write(cache_dir.path().join("orphan.blob"), vec![0_u8; blob_size])
            .expect("write orphan.blob");

        // "model-a" is configured (NotLoaded, never loaded) — "gone-model" is
        // attributed but absent from config entirely.
        let mut config = make_config(&[("model-a", 5.0)], 17.0);
        config.ov_cache_dir = Some(cache_dir.path().to_string_lossy().into_owned());
        // Cap just under one blob's worth so exactly one deletion suffices.
        config.ov_cache_max_gb = 2.0 * 2.0 / 1024.0 + 0.0001; // 2 * 2MB, in GB
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();

        let manifest_root =
            crate::cache_manifest::manifest_root(config_ov_cache_dir(&mm)).expect("manifest root");
        crate::cache_manifest::write_device_blobs_entry(
            &manifest_root,
            "gone-model",
            "GPU",
            &["gone.blob".to_owned()],
        );
        crate::cache_manifest::write_device_blobs_entry(
            &manifest_root,
            "model-a",
            "GPU",
            &["not-ready.blob".to_owned()],
        );
        // orphan.blob deliberately gets no manifest entry at all.

        mm.run_cache_sweep_once_blocking();

        assert!(
            !cache_dir.path().join("gone.blob").exists(),
            "not-in-config blob must be deleted first"
        );
        assert!(
            cache_dir.path().join("not-ready.blob").exists(),
            "configured-but-not-ready blob must survive once the cap is met"
        );
        assert!(
            cache_dir.path().join("orphan.blob").exists(),
            "unattributed blob must survive once the cap is met"
        );
    }

    /// Test-only accessor: `ModelManager` doesn't expose its `Config` outside
    /// the crate, so prune/precompute tests that need to seed a manifest at
    /// the *same* root the sweep will compute go through the live config
    /// instead of recomputing the path independently.
    fn config_ov_cache_dir(mm: &ModelManager) -> Option<&str> {
        mm.config.ov_cache_dir.as_deref()
    }

    /// Regression test for the eviction-drain fix (2026-07-18, live-caught on
    /// real hardware): normal (non-abort) `evict_model` must WAIT for a real
    /// in-flight request to finish before it drops the engine handle and joins
    /// — it must not race ahead and reconcile the tracker while the request is
    /// still genuinely active. Simulated here via `occupy_slot_for_test`'s held
    /// permit (no live engine thread needed) rather than a real generation.
    #[tokio::test]
    async fn evict_model_waits_for_a_real_in_flight_request_before_completing() {
        let config = make_config(&[("model-a", 5.0)], 17.0);
        let mm = Arc::new(
            ModelManager::new(config, Arc::new(MockEngineFactory::default()))
                .await
                .unwrap(),
        );
        mm.load_model("model-a").await.unwrap();

        // Hold a permit — model-a looks genuinely busy (active() == 1) —
        // exactly as a real in-flight streaming request would.
        let permit = {
            let guard = mm.read_models("test");
            let handle = guard.get("model-a").unwrap().handle.as_ref().unwrap();
            match handle {
                EngineHandleKind::TextGen(h) => h.occupy_slot_for_test().unwrap(),
                other => panic!("model-a must be a TextGen mock, got {other:?}"),
            }
        };

        let evicted = tokio::spawn({
            let mm = Arc::clone(&mm);
            async move { mm.evict_model("model-a").await }
        });

        // Give the eviction task a moment to reach (and start polling in) the
        // drain wait, then confirm it has NOT completed — it must still be
        // waiting on the held permit, not racing ahead.
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(
            !evicted.is_finished(),
            "eviction must still be draining while the permit is held"
        );
        assert_eq!(
            mm.read_models("test").get("model-a").unwrap().state,
            ModelState::Evicting,
            "model must be marked Evicting (not yet NotLoaded) while genuinely busy"
        );

        // Release the simulated in-flight request — eviction should now
        // complete promptly (well within one EVICT_DRAIN_POLL_INTERVAL or two).
        drop(permit);
        evicted
            .await
            .expect("eviction task must not panic")
            .expect("eviction must succeed once the request finishes");

        assert_eq!(
            mm.read_models("test").get("model-a").unwrap().state,
            ModelState::NotLoaded,
            "model must be evicted once genuinely idle"
        );
    }

    /// The conservative KV estimate the live-RAM gate charges. Extracted from
    /// `ensure_vram_for` so `check_admission` charges the identical quantity —
    /// this test is what keeps the two from drifting apart again.
    #[test]
    fn test_conservative_kv_need_gb() {
        let mut cfg = make_config(&[], 24.0);
        cfg.min_kv_cache_gb = 2.0;
        cfg.cache_size_gb = 0.0;

        // Bounded target, no global cap → the target itself.
        assert!((ModelManager::conservative_kv_need_gb(6.0, &cfg) - 6.0).abs() < 1e-9);
        // Target below the floor → floored.
        assert!((ModelManager::conservative_kv_need_gb(1.0, &cfg) - 2.0).abs() < 1e-9);
        // Unbounded grab-all with no cap → genuinely unpredictable, degrade to
        // the floor (do NOT count 0.0, which would undercount a multi-GB pool).
        assert!((ModelManager::conservative_kv_need_gb(0.0, &cfg) - 2.0).abs() < 1e-9);

        cfg.cache_size_gb = 4.0;
        // Global cap bites before the target.
        assert!((ModelManager::conservative_kv_need_gb(6.0, &cfg) - 4.0).abs() < 1e-9);
        // Unbounded target now predictable: it expands to the cap. This is the
        // case the dry-run used to miss entirely, charging `min_kv` (2.0)
        // against a pool that really becomes 4.0.
        assert!((ModelManager::conservative_kv_need_gb(0.0, &cfg) - 4.0).abs() < 1e-9);
    }

    /// `RamGate` is inert off the UMA `system` domain — the property that keeps
    /// this fix from regressing the discrete-GPU boxes, where no live-RAM gate
    /// exists and charging the full KV target would refuse loads that succeed.
    #[test]
    fn test_ram_gate_inert_on_discrete_domains() {
        let discrete = RamGate {
            avail: None,
            needed: None,
            floor: 4.0,
        };
        assert!(discrete.admits(0.0), "a discrete domain has no RAM gate");
        assert!(discrete.admits(-999.0), "and nothing can make it refuse");
    }

    /// On UMA the gate refuses when the need does not clear the OS floor, and
    /// eviction (which frees real RAM) can rescue it — the retry behaviour
    /// `ensure_vram_for` relies on, mirrored in the dry-run.
    #[test]
    fn test_ram_gate_gates_and_credits_eviction_on_uma() {
        let gate = RamGate {
            avail: Some(10.0),
            needed: Some(24.0),
            floor: 4.0,
        };
        assert!(
            !gate.admits(0.0),
            "24 GB needed against 10 GB avail must refuse"
        );
        // Evicting enough residents frees real RAM and admits the load.
        assert!(
            gate.admits(20.0),
            "eviction credit must be counted, as ensure_vram_for's retry does"
        );
    }

    /// T5.4 seam: `compute_kv_pool_gb` pins the current KV-size policy so a
    /// future co-residency change is a deliberate, test-visible edit.
    /// `kv_target_gb = 0.0` is the unbounded (pre-Slice-2) path.
    #[test]
    fn test_compute_kv_pool_gb_current_policy() {
        let mut cfg = make_config(&[], 24.0);
        cfg.vram_safety_margin_gb = 1.0;
        cfg.min_kv_cache_gb = 2.0;

        // Uncapped (cache_size_gb == 0): grab all remaining minus the margin.
        cfg.cache_size_gb = 0.0;
        assert!((ModelManager::compute_kv_pool_gb(10.0, 0.0, &cfg) - 9.0).abs() < 1e-9);
        // Floored at min_kv when remaining-after-margin is tiny.
        assert!((ModelManager::compute_kv_pool_gb(2.0, 0.0, &cfg) - 2.0).abs() < 1e-9);
        // Capped by cache_size_gb when set and smaller than the optimal grab.
        cfg.cache_size_gb = 4.0;
        assert!((ModelManager::compute_kv_pool_gb(10.0, 0.0, &cfg) - 4.0).abs() < 1e-9);
        // Cap above the optimal grab does not inflate it.
        assert!((ModelManager::compute_kv_pool_gb(3.0, 0.0, &cfg) - 2.0).abs() < 1e-9);
    }

    /// Co-residency Slice 2: a non-zero `kv_target_gb` clamps the pool so a
    /// model leaves VRAM for co-residents, while the floor and the global
    /// `cache_size_gb` cap still apply.
    #[test]
    fn test_compute_kv_pool_gb_bounded_target() {
        let mut cfg = make_config(&[], 24.0);
        cfg.vram_safety_margin_gb = 1.0;
        cfg.min_kv_cache_gb = 2.0;
        cfg.cache_size_gb = 0.0;

        // Target below the grab-all baseline → clamped to the target.
        // free-margin = 19; target 6 → 6.
        assert!((ModelManager::compute_kv_pool_gb(20.0, 6.0, &cfg) - 6.0).abs() < 1e-9);
        // Target above the available baseline → cannot exceed what's free.
        // free-margin = 4; target 6 → 4.
        assert!((ModelManager::compute_kv_pool_gb(5.0, 6.0, &cfg) - 4.0).abs() < 1e-9);
        // Target below the floor is lifted to min_kv (never start below the floor).
        // free-margin = 19; target 1; floor 2 → 2.
        assert!((ModelManager::compute_kv_pool_gb(20.0, 1.0, &cfg) - 2.0).abs() < 1e-9);
        // Global cache_size_gb cap still binds on top of a larger target.
        cfg.cache_size_gb = 5.0;
        assert!((ModelManager::compute_kv_pool_gb(20.0, 8.0, &cfg) - 5.0).abs() < 1e-9);
    }

    // ---- Non-KV VRAM estimation (fix-1: STT / embed / image charge weight only) --

    /// A model with an explicit `stt` kind hint must reserve only its weight
    /// footprint — not weight + KV pool — so two STT models can co-reside on
    /// a box where a single LLM KV grab would evict the first.
    #[tokio::test]
    async fn stt_model_charges_weight_only() {
        // 16 GB box; STT model is 1 GB. Without the fix, ensure_vram_for would
        // compute a ~14.8 GB KV pool (all remaining after weight + margin), making
        // used_vram ≈ 15.8 GB instead of 1 GB.
        let mut cfg = make_config(&[("whisper-tiny-stt", 1.0)], 16.0);
        cfg.models
            .entry("whisper-tiny-stt".to_owned())
            .or_default()
            .kind = Some("stt".to_owned());
        let mm = ModelManager::new(cfg, Arc::new(MockEngineFactory::default()))
            .await
            .expect("make mm");

        mm.load_model("whisper-tiny-stt").await.unwrap();
        let used = mm.metrics_snapshot().used_vram_gb;
        assert!(
            (used - 1.0).abs() < 0.1,
            "STT model must charge weight only (~1 GB), got {used:.2} GB"
        );
    }

    /// Two STT models each weighing 1 GB must both load on a 16 GB box without
    /// evicting each other (they would if each grabbed ~14.8 GB KV pool).
    #[tokio::test]
    async fn two_stt_models_can_co_reside() {
        let mut cfg = make_config(
            &[("whisper-tiny-stt", 1.0), ("whisper-base-stt", 1.5)],
            16.0,
        );
        cfg.models
            .entry("whisper-tiny-stt".to_owned())
            .or_default()
            .kind = Some("stt".to_owned());
        cfg.models
            .entry("whisper-base-stt".to_owned())
            .or_default()
            .kind = Some("stt".to_owned());
        let mm = ModelManager::new(cfg, Arc::new(MockEngineFactory::default()))
            .await
            .expect("make mm");

        mm.load_model("whisper-tiny-stt").await.unwrap();
        mm.load_model("whisper-base-stt").await.unwrap();

        // Both must be Ready (not evicted).
        let snap = mm.metrics_snapshot();
        assert_eq!(
            snap.models.len(),
            2,
            "both STT models must appear in the snapshot: {snap:#?}"
        );
        let both_loaded = snap.models.iter().all(|m| m.loaded);
        assert!(
            both_loaded,
            "both STT models must be co-resident: {snap:#?}"
        );
        // Combined footprint ≈ 2.5 GB, not 2× (1 + 14.8) GB.
        assert!(
            mm.metrics_snapshot().used_vram_gb < 4.0,
            "combined STT footprint must be near 2.5 GB, not KV-inflated"
        );
    }

    /// A model WITHOUT an explicit kind hint must still get the LLM KV pool
    /// treatment (regression guard: `needs_kv_pool` defaults to `true`).
    #[tokio::test]
    async fn llm_without_kind_hint_gets_kv_pool() {
        // 16 GB box, 5 GB LLM, no kind hint → grab-all KV after weight.
        let mm = make_mm(&[("qwen3-8b", 5.0)], 16.0).await;
        mm.load_model("qwen3-8b").await.unwrap();
        let used = mm.metrics_snapshot().used_vram_gb;
        // Without a kind hint the full 16 GB domain is claimed (weight + KV grab-all).
        assert!(
            used > 5.0,
            "LLM with no kind hint must claim more than weight alone, got {used:.2} GB"
        );
    }

    /// Co-residency Slice 2: the precedence chain
    /// override → policy → global default → builtin.
    #[test]
    fn test_resolve_runtime_params_precedence() {
        let mut cfg = make_config(&[("chat", 5.0), ("embed", 1.0), ("bare", 2.0)], 24.0);
        cfg.max_num_seqs = 16;
        cfg.default_kv_cache_gb = 4.0;
        cfg.models.entry("chat".to_owned()).or_default().policy =
            crate::model_manager::config::ModelPolicy {
                kv_cache_gb: Some(8.0),
                max_concurrent_streams: Some(4),
                ..Default::default()
            };

        // Per-model policy wins over the global default.
        let chat = resolve_runtime_params(&cfg, cfg.models.get("chat"), &LoadOverrides::default());
        assert!((chat.kv_cache_gb - 8.0).abs() < 1e-9);
        assert_eq!(chat.max_num_seqs, 4);

        // No policy → global default KV + global max_num_seqs.
        let bare = resolve_runtime_params(&cfg, cfg.models.get("bare"), &LoadOverrides::default());
        assert!((bare.kv_cache_gb - 4.0).abs() < 1e-9);
        assert_eq!(bare.max_num_seqs, 16);

        // A load override beats both policy and global default.
        let over = LoadOverrides {
            kv_cache_gb: Some(2.0),
            max_concurrent_streams: Some(2),
            force: false,
        };
        let chat_over = resolve_runtime_params(&cfg, cfg.models.get("chat"), &over);
        assert!((chat_over.kv_cache_gb - 2.0).abs() < 1e-9);
        assert_eq!(chat_over.max_num_seqs, 2);
    }

    // ---- MoE cap (apply_moe_cap) ----------------------------------------

    fn write_config(dir: &std::path::Path, json: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), json).unwrap();
    }

    /// Dense model: cap is a no-op regardless of resolved value.
    #[test]
    fn moe_cap_noop_on_dense_model() {
        let dir = std::env::temp_dir().join(format!("rv-moecap-dense-{}", std::process::id()));
        write_config(&dir, r#"{"model_type":"qwen3","num_hidden_layers":36}"#);
        assert_eq!(apply_moe_cap("dense", 256, &dir, false), 256);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `MoE` model with no explicit cap → capped to `MOE_MAX_NUM_SEQS`.
    #[test]
    fn moe_cap_applied_when_no_explicit_override() {
        let dir = std::env::temp_dir().join(format!("rv-moecap-moe-{}", std::process::id()));
        write_config(
            &dir,
            r#"{"model_type":"qwen3_moe","num_experts":128,"num_experts_per_tok":8}"#,
        );
        assert_eq!(
            apply_moe_cap("moe-model", 256, &dir, false),
            MOE_MAX_NUM_SEQS
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `MoE` model with explicit operator override → cap respected, not overridden.
    #[test]
    fn moe_cap_explicit_override_respected() {
        let dir = std::env::temp_dir().join(format!("rv-moecap-exp-{}", std::process::id()));
        write_config(
            &dir,
            r#"{"model_type":"qwen3_moe","num_experts":128,"num_experts_per_tok":8}"#,
        );
        // cap_was_explicit=true → we warn but do NOT override
        assert_eq!(apply_moe_cap("moe-model", 4, &dir, true), 4);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `MoE` model already at or below the cap → no change.
    #[test]
    fn moe_cap_already_at_cap() {
        let dir = std::env::temp_dir().join(format!("rv-moecap-low-{}", std::process::id()));
        write_config(&dir, r#"{"num_experts":128}"#);
        assert_eq!(
            apply_moe_cap("moe-model", MOE_MAX_NUM_SEQS, &dir, false),
            MOE_MAX_NUM_SEQS
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `resolved == 0` (the documented "let `OpenVINO` decide" sentinel,
    /// CONFIG.md `max_num_seqs`) is NOT "already at or below the cap" — an
    /// `MoE` model with no explicit override must still be capped to
    /// `MOE_MAX_NUM_SEQS`, not handed 256-way concurrency.
    #[test]
    fn moe_cap_treats_zero_sentinel_as_uncapped_not_already_safe() {
        let dir = std::env::temp_dir().join(format!("rv-moecap-zero-{}", std::process::id()));
        write_config(
            &dir,
            r#"{"model_type":"qwen3_moe","num_experts":128,"num_experts_per_tok":8}"#,
        );
        assert_eq!(
            apply_moe_cap("moe-model", 0, &dir, false),
            MOE_MAX_NUM_SEQS,
            "max_num_seqs=0 must still be capped for an MoE model"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- Speculative-decoding concurrency guard (apply_speculative_cap) ----

    /// Not a speculative model: cap is a no-op regardless of resolved value.
    #[test]
    fn speculative_cap_noop_when_not_speculative() {
        assert_eq!(apply_speculative_cap("plain", 256, false, false), 256);
    }

    /// Speculative model with no explicit cap → capped to `SPECULATIVE_MAX_NUM_SEQS`.
    #[test]
    fn speculative_cap_applied_when_no_explicit_override() {
        assert_eq!(
            apply_speculative_cap("spec-model", 256, true, false),
            SPECULATIVE_MAX_NUM_SEQS
        );
    }

    /// Speculative model with an explicit operator override → cap respected,
    /// not overridden (warn-but-respect, same contract as `apply_moe_cap`).
    #[test]
    fn speculative_cap_explicit_override_respected() {
        assert_eq!(apply_speculative_cap("spec-model", 4, true, true), 4);
    }

    /// `resolved == 0` (the "let `OpenVINO` decide" sentinel) is NOT "already
    /// at or below the cap" — a speculative model with no explicit override
    /// must still be capped to `SPECULATIVE_MAX_NUM_SEQS`, matching
    /// `apply_moe_cap`'s equivalent fix.
    #[test]
    fn speculative_cap_treats_zero_sentinel_as_uncapped_not_already_safe() {
        assert_eq!(
            apply_speculative_cap("spec-model", 0, true, false),
            SPECULATIVE_MAX_NUM_SEQS,
            "max_num_seqs=0 must still be capped for a speculative model"
        );
    }

    /// Speculative model already at or below the cap → no change.
    #[test]
    fn speculative_cap_already_at_cap() {
        assert_eq!(
            apply_speculative_cap("spec-model", SPECULATIVE_MAX_NUM_SEQS, true, false),
            SPECULATIVE_MAX_NUM_SEQS
        );
    }

    // ---- Speculative decoding pairing validation (validate_speculative_pairing) --

    /// Builds a valid dense-LLM directory: `config.json` with the given
    /// `vocab_size`/`model_type`, plus the `openvino_model.xml` marker file
    /// `draft_model()`'s loader requires.
    fn write_dense_llm_dir(dir: &std::path::Path, model_type: &str, vocab_size: u64) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            format!(r#"{{"model_type":"{model_type}","vocab_size":{vocab_size}}}"#),
        )
        .unwrap();
        std::fs::write(dir.join("openvino_model.xml"), "").unwrap();
    }

    fn default_spec(draft_model: &str) -> config::SpeculativeConfig {
        config::SpeculativeConfig {
            draft_model: draft_model.to_owned(),
            draft_vram_gb: 0.8,
            draft_device: None,
            num_assistant_tokens: 5,
            verify_on_load: true,
        }
    }

    /// A valid dense/dense, matching-tokenizer, same-device pairing passes.
    #[test]
    fn spec_pairing_happy_path_passes() {
        let target = std::env::temp_dir().join(format!("rv-spec-happy-t-{}", std::process::id()));
        let draft = std::env::temp_dir().join(format!("rv-spec-happy-d-{}", std::process::id()));
        write_dense_llm_dir(&target, "qwen3", 151_936);
        write_dense_llm_dir(&draft, "qwen3", 151_936);
        assert!(
            validate_speculative_pairing(
                &target,
                &draft,
                ModelKind::TextGen,
                "GPU.1",
                &default_spec("draft"),
            )
            .is_ok()
        );
        std::fs::remove_dir_all(&target).ok();
        std::fs::remove_dir_all(&draft).ok();
    }

    /// Gate 1: a VLM target (`ModelKind::Vision`) is refused outright.
    #[test]
    fn spec_pairing_vlm_target_is_refused() {
        let target = std::env::temp_dir().join(format!("rv-spec-vlm-t-{}", std::process::id()));
        let draft = std::env::temp_dir().join(format!("rv-spec-vlm-d-{}", std::process::id()));
        write_dense_llm_dir(&target, "qwen3_vl", 151_936);
        write_dense_llm_dir(&draft, "qwen3", 151_936);
        let err = validate_speculative_pairing(
            &target,
            &draft,
            ModelKind::Vision,
            "GPU.1",
            &default_spec("draft"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("dense text-generation"));
        std::fs::remove_dir_all(&target).ok();
        std::fs::remove_dir_all(&draft).ok();
    }

    /// Gate 1: an NPU target device is refused (static `LLMPipeline`, no CB).
    #[test]
    fn spec_pairing_npu_device_is_refused() {
        let target = std::env::temp_dir().join(format!("rv-spec-npu-t-{}", std::process::id()));
        let draft = std::env::temp_dir().join(format!("rv-spec-npu-d-{}", std::process::id()));
        write_dense_llm_dir(&target, "qwen3", 151_936);
        write_dense_llm_dir(&draft, "qwen3", 151_936);
        let err = validate_speculative_pairing(
            &target,
            &draft,
            ModelKind::TextGen,
            "NPU",
            &default_spec("draft"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("NPU"));
        std::fs::remove_dir_all(&target).ok();
        std::fs::remove_dir_all(&draft).ok();
    }

    /// Gate 1: a `MoE` target (`num_experts > 1`) is refused on the throughput basis.
    #[test]
    fn spec_pairing_moe_target_is_refused() {
        let target = std::env::temp_dir().join(format!("rv-spec-moet-t-{}", std::process::id()));
        let draft = std::env::temp_dir().join(format!("rv-spec-moet-d-{}", std::process::id()));
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(
            target.join("config.json"),
            r#"{"model_type":"qwen3_moe","num_experts":128,"vocab_size":151936}"#,
        )
        .unwrap();
        std::fs::write(target.join("openvino_model.xml"), "").unwrap();
        write_dense_llm_dir(&draft, "qwen3", 151_936);
        let err = validate_speculative_pairing(
            &target,
            &draft,
            ModelKind::TextGen,
            "GPU.1",
            &default_spec("draft"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("MoE"));
        std::fs::remove_dir_all(&target).ok();
        std::fs::remove_dir_all(&draft).ok();
    }

    /// Gate 2: a `MoE` draft is refused symmetrically with the target ban.
    #[test]
    fn spec_pairing_moe_draft_is_refused() {
        let target = std::env::temp_dir().join(format!("rv-spec-moed-t-{}", std::process::id()));
        let draft = std::env::temp_dir().join(format!("rv-spec-moed-d-{}", std::process::id()));
        write_dense_llm_dir(&target, "qwen3", 151_936);
        std::fs::create_dir_all(&draft).unwrap();
        std::fs::write(
            draft.join("config.json"),
            r#"{"model_type":"qwen3_moe","num_experts":8,"vocab_size":151936}"#,
        )
        .unwrap();
        std::fs::write(draft.join("openvino_model.xml"), "").unwrap();
        let err = validate_speculative_pairing(
            &target,
            &draft,
            ModelKind::TextGen,
            "GPU.1",
            &default_spec("draft"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("MoE"));
        std::fs::remove_dir_all(&target).ok();
        std::fs::remove_dir_all(&draft).ok();
    }

    /// Gate 2: a draft directory missing `openvino_model.xml` is refused.
    #[test]
    fn spec_pairing_draft_missing_model_xml_is_refused() {
        let target = std::env::temp_dir().join(format!("rv-spec-nomxl-t-{}", std::process::id()));
        let draft = std::env::temp_dir().join(format!("rv-spec-nomxl-d-{}", std::process::id()));
        write_dense_llm_dir(&target, "qwen3", 151_936);
        std::fs::create_dir_all(&draft).unwrap();
        std::fs::write(draft.join("config.json"), r#"{"vocab_size":151936}"#).unwrap();
        // no openvino_model.xml written
        let err = validate_speculative_pairing(
            &target,
            &draft,
            ModelKind::TextGen,
            "GPU.1",
            &default_spec("draft"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("openvino_model.xml"));
        std::fs::remove_dir_all(&target).ok();
        std::fs::remove_dir_all(&draft).ok();
    }

    /// Gate 2: a draft directory shaped like a VLM (ships the vision-embeddings
    /// file) is refused even though `openvino_model.xml` is also present.
    #[test]
    fn spec_pairing_vlm_draft_is_refused() {
        let target = std::env::temp_dir().join(format!("rv-spec-vlmd-t-{}", std::process::id()));
        let draft = std::env::temp_dir().join(format!("rv-spec-vlmd-d-{}", std::process::id()));
        write_dense_llm_dir(&target, "qwen3", 151_936);
        write_dense_llm_dir(&draft, "qwen3_vl", 151_936);
        std::fs::write(draft.join("openvino_vision_embeddings_model.xml"), "").unwrap();
        let err = validate_speculative_pairing(
            &target,
            &draft,
            ModelKind::TextGen,
            "GPU.1",
            &default_spec("draft"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("VLM"));
        std::fs::remove_dir_all(&target).ok();
        std::fs::remove_dir_all(&draft).ok();
    }

    /// Gate 3: mismatched `vocab_size` is refused.
    #[test]
    fn spec_pairing_vocab_mismatch_is_refused() {
        let target = std::env::temp_dir().join(format!("rv-spec-vocm-t-{}", std::process::id()));
        let draft = std::env::temp_dir().join(format!("rv-spec-vocm-d-{}", std::process::id()));
        write_dense_llm_dir(&target, "qwen3", 151_936);
        write_dense_llm_dir(&draft, "mistral", 32_000);
        let err = validate_speculative_pairing(
            &target,
            &draft,
            ModelKind::TextGen,
            "GPU.1",
            &default_spec("draft"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("tokenizer mismatch"));
        std::fs::remove_dir_all(&target).ok();
        std::fs::remove_dir_all(&draft).ok();
    }

    /// Gate 3: an unreadable/missing `config.json` (so `vocab_size` cannot be
    /// verified) is a hard refusal, not a pass.
    #[test]
    fn spec_pairing_unreadable_config_is_refused() {
        let target = std::env::temp_dir().join(format!("rv-spec-noconf-t-{}", std::process::id()));
        let draft = std::env::temp_dir().join(format!("rv-spec-noconf-d-{}", std::process::id()));
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("openvino_model.xml"), "").unwrap();
        // no config.json written on the target — vocab_size unreadable
        write_dense_llm_dir(&draft, "qwen3", 151_936);
        let err = validate_speculative_pairing(
            &target,
            &draft,
            ModelKind::TextGen,
            "GPU.1",
            &default_spec("draft"),
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("cannot verify tokenizer compatibility")
        );
        std::fs::remove_dir_all(&target).ok();
        std::fs::remove_dir_all(&draft).ok();
    }

    /// Gate 4: an explicit `draft_device` differing from the target's resolved
    /// device is refused (v1 scope restriction, not a correctness finding).
    #[test]
    fn spec_pairing_cross_device_is_refused() {
        let target = std::env::temp_dir().join(format!("rv-spec-xdev-t-{}", std::process::id()));
        let draft = std::env::temp_dir().join(format!("rv-spec-xdev-d-{}", std::process::id()));
        write_dense_llm_dir(&target, "qwen3", 151_936);
        write_dense_llm_dir(&draft, "qwen3", 151_936);
        let mut spec = default_spec("draft");
        spec.draft_device = Some("CPU".to_owned());
        let err = validate_speculative_pairing(&target, &draft, ModelKind::TextGen, "GPU.1", &spec)
            .unwrap_err();
        assert!(err.to_string().contains("cross-device drafting"));
        std::fs::remove_dir_all(&target).ok();
        std::fs::remove_dir_all(&draft).ok();
    }

    /// Gate 4: an explicit `draft_device` equal to the target's device passes.
    #[test]
    fn spec_pairing_same_device_explicit_is_ok() {
        let target = std::env::temp_dir().join(format!("rv-spec-sdev-t-{}", std::process::id()));
        let draft = std::env::temp_dir().join(format!("rv-spec-sdev-d-{}", std::process::id()));
        write_dense_llm_dir(&target, "qwen3", 151_936);
        write_dense_llm_dir(&draft, "qwen3", 151_936);
        let mut spec = default_spec("draft");
        spec.draft_device = Some("GPU.1".to_owned());
        assert!(
            validate_speculative_pairing(&target, &draft, ModelKind::TextGen, "GPU.1", &spec)
                .is_ok()
        );
        std::fs::remove_dir_all(&target).ok();
        std::fs::remove_dir_all(&draft).ok();
    }

    /// When `model_type` differs but `vocab_size` matches, the pairing is
    /// allowed (a WARN, not a refusal) — see gate 3's rationale.
    #[test]
    fn spec_pairing_model_type_mismatch_warns_not_refuses() {
        let target = std::env::temp_dir().join(format!("rv-spec-mtw-t-{}", std::process::id()));
        let draft = std::env::temp_dir().join(format!("rv-spec-mtw-d-{}", std::process::id()));
        write_dense_llm_dir(&target, "mistral-small", 151_936);
        write_dense_llm_dir(&draft, "ministral", 151_936);
        assert!(
            validate_speculative_pairing(
                &target,
                &draft,
                ModelKind::TextGen,
                "GPU.1",
                &default_spec("draft"),
            )
            .is_ok()
        );
        std::fs::remove_dir_all(&target).ok();
        std::fs::remove_dir_all(&draft).ok();
    }

    // ---- Speculative decoding: end-to-end wiring (load_model_with_overrides) --

    /// Builds a `Config` for speculative-decoding wiring tests: a real
    /// tempdir `models_dir` with a `target` model (`vram_gb=15.0`) whose
    /// policy carries `speculative`. The draft is deliberately NOT its own
    /// `models` entry (Part 2's design: never independently loadable).
    fn speculative_wiring_config(
        models_dir: &std::path::Path,
        total_vram_gb: f64,
        spec: config::SpeculativeConfig,
    ) -> Config {
        let mut cfg = make_config(&[], total_vram_gb);
        cfg.models_dir = models_dir.to_path_buf();
        cfg.models.insert(
            "target".to_owned(),
            config::ModelEntry {
                vram_gb: 15.0,
                kind: None,
                policy: config::ModelPolicy {
                    speculative: Some(spec),
                    ..Default::default()
                },
            },
        );
        cfg
    }

    /// A valid speculative pairing loads successfully, and the VRAM
    /// tracker's reservation reflects the combined weight + draft footprint
    /// — not just the target's own `vram_gb` (Part 6: one reservation, one
    /// `model_id`). `cache_size_gb` bounds the KV pool so the total is exact:
    /// weight (15.0) + draft (0.8) + kv (1.0 cap) = 16.8 GB.
    #[tokio::test]
    async fn load_model_with_overrides_speculative_folds_draft_vram() {
        let models_dir =
            std::env::temp_dir().join(format!("rv-spec-wire-ok-{}", std::process::id()));
        write_dense_llm_dir(&models_dir.join("target"), "qwen3", 151_936);
        write_dense_llm_dir(&models_dir.join("draft"), "qwen3", 151_936);
        let mut cfg = speculative_wiring_config(&models_dir, 30.0, default_spec("draft"));
        cfg.cache_size_gb = 1.0;
        let mm = ModelManager::new(cfg, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();

        mm.load_model("target").await.unwrap();

        let used = mm.metrics_snapshot().used_vram_gb;
        assert!(
            (used - 16.8).abs() < 1e-9,
            "expected weight(15.0) + draft(0.8) + kv(1.0) = 16.8 GB reserved, got {used}"
        );

        mm.evict_model("target").await.unwrap();
        assert!(
            mm.metrics_snapshot().used_vram_gb.abs() < 1e-9,
            "eviction must free the combined reservation"
        );

        std::fs::remove_dir_all(&models_dir).ok();
    }

    /// An invalid pairing (tokenizer mismatch) fails the load at
    /// `load_model_with_overrides`'s choke point — proving the wiring
    /// actually calls `validate_speculative_pairing`, not just constructing
    /// a `DraftHint` unconditionally. This model was registered via startup
    /// config (not `add_model`/`reload_config`), so `build_not_loaded_record`
    /// never ran for it — the load choke point is the only gate that catches
    /// this, exactly the case the plan calls out for startup preloads.
    #[tokio::test]
    async fn load_model_with_overrides_speculative_invalid_pairing_fails_load() {
        let models_dir =
            std::env::temp_dir().join(format!("rv-spec-wire-bad-{}", std::process::id()));
        write_dense_llm_dir(&models_dir.join("target"), "qwen3", 151_936);
        write_dense_llm_dir(&models_dir.join("draft"), "mistral", 32_000); // vocab mismatch
        let cfg = speculative_wiring_config(&models_dir, 30.0, default_spec("draft"));
        let mm = ModelManager::new(cfg, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();

        let err = mm.load_model("target").await.unwrap_err();
        assert!(
            err.to_string().contains("tokenizer mismatch"),
            "load must fail via validate_speculative_pairing's gate 3, got: {err}"
        );
        assert!(
            mm.metrics_snapshot().used_vram_gb.abs() < 1e-9,
            "a failed load must not leave a phantom VRAM reservation"
        );

        std::fs::remove_dir_all(&models_dir).ok();
    }

    /// When VRAM is insufficient AND no model can be evicted → error.
    #[tokio::test]
    async fn test_load_fails_when_no_evictable_model() {
        // 3 GB total; model-a = 5 GB (doesn't fit, nothing to evict).
        let mm = make_mm(&[("model-a", 5.0)], 3.0).await;
        let err = mm.load_model("model-a").await.unwrap_err();
        assert!(
            err.to_string().contains("insufficient VRAM"),
            "error must mention insufficient VRAM: {err}"
        );
    }

    /// A doomed load — one that would still be too big even after evicting
    /// every other resident in its domain — must fail immediately via the
    /// upfront feasibility check, WITHOUT evicting any of those residents.
    /// Regression test for the project's internal engineering log:
    /// previously `ensure_vram_for` discovered infeasibility only after
    /// destructively evicting every co-resident model one at a time.
    #[tokio::test]
    async fn test_doomed_load_does_not_evict_anything() {
        // 10 GB total; two small residents (3 GB + 3 GB, 6 GB combined) plus a
        // doomed 100 GB request: even freeing every last GB can never satisfy
        // it — this can NEVER fit. The "-embed" substring makes MockEngineFactory
        // classify model-a/model-b as Embedding (no KV pool, so they reserve
        // exactly their declared weight and genuinely co-reside — an LLM-kind
        // model would grab-all remaining VRAM as KV and evict the other purely
        // from that, unrelated to the fix under test).
        let mm = make_mm(
            &[
                ("model-a-embed", 3.0),
                ("model-b-embed", 3.0),
                ("huge", 100.0),
            ],
            10.0,
        )
        .await;
        mm.load_model("model-a-embed").await.unwrap();
        mm.load_model("model-b-embed").await.unwrap();
        assert_eq!(
            mm.read_models("test").get("model-a-embed").unwrap().state,
            ModelState::Ready
        );
        assert_eq!(
            mm.read_models("test").get("model-b-embed").unwrap().state,
            ModelState::Ready
        );

        let err = mm.load_model("huge").await.unwrap_err();
        assert!(
            err.to_string().contains("insufficient VRAM"),
            "error must mention insufficient VRAM: {err}"
        );
        assert!(
            err.to_string().contains("nothing was evicted"),
            "error must state the upfront check fired before any eviction: {err}"
        );

        // The critical assertion: model-a-embed and model-b-embed must STILL
        // be Ready — a doomed request must not destructively evict co-resident
        // models it was never going to be able to use anyway.
        assert_eq!(
            mm.read_models("test").get("model-a-embed").unwrap().state,
            ModelState::Ready,
            "model-a-embed must survive a doomed load — it was never going to be enough"
        );
        assert_eq!(
            mm.read_models("test").get("model-b-embed").unwrap().state,
            ModelState::Ready,
            "model-b-embed must survive a doomed load — it was never going to be enough"
        );
    }

    // ---- co-residency Slice 1: pinning + priority + check_admission -------

    /// Build a 3-model manager with KV capped at 2 GB on a 17 GB card so models
    /// co-reside (each reserves weight + 2 GB), and apply per-model policies.
    /// Mirrors the `test_lru_spares_busy_model_for_idle_one` setup.
    async fn make_mm_with_policies(
        policies: &[(&str, bool, i32)], // (id, pinned, priority)
    ) -> ModelManager {
        let mut config = make_config(
            &[("model-a", 5.0), ("model-b", 5.0), ("model-c", 5.0)],
            17.0,
        );
        config.cache_size_gb = 2.0;
        for (id, pinned, priority) in policies {
            config.models.entry((*id).to_owned()).or_default().policy = config::ModelPolicy {
                pinned: *pinned,
                priority: *priority,
                ..Default::default()
            };
        }
        ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .expect("ModelManager::new failed in test")
    }

    /// `evict_model` clears the voice pin's LLM entry when the evicted model is
    /// the pinned LLM — proves the `ModelManager` ↔ `VoicePinManager` wiring,
    /// not just the isolated pin logic (unit-tested separately in `voice_pin.rs`).
    #[tokio::test]
    async fn test_evict_model_clears_matching_voice_pin() {
        let config = make_config(&[("model-a", 6.0)], 10.0);
        let voice_pin = Arc::new(crate::voice_pin::VoicePinManager::default());
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap()
            .with_voice_pin(Arc::clone(&voice_pin));

        mm.load_model("model-a").await.unwrap();
        voice_pin.try_set("stt".to_owned(), "model-a".to_owned(), "tts".to_owned());

        mm.evict_model("model-a").await.unwrap();

        assert!(
            voice_pin.get().is_none(),
            "pin must clear — its LLM (model-a) was just evicted"
        );
    }

    /// Eviction of a model that is NOT the pinned LLM leaves the pin untouched.
    #[tokio::test]
    async fn test_evict_model_leaves_unrelated_voice_pin_alone() {
        let config = make_config(&[("model-a", 6.0), ("model-b", 6.0)], 20.0);
        let voice_pin = Arc::new(crate::voice_pin::VoicePinManager::default());
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap()
            .with_voice_pin(Arc::clone(&voice_pin));

        mm.load_model("model-a").await.unwrap();
        mm.load_model("model-b").await.unwrap();
        voice_pin.try_set("stt".to_owned(), "model-a".to_owned(), "tts".to_owned());

        mm.evict_model("model-b").await.unwrap();

        assert_eq!(
            voice_pin.get().unwrap().llm_model,
            "model-a",
            "pin survives eviction of a model that isn't its pinned LLM"
        );
    }

    /// A pinned model is never the eviction victim: with only a pinned model
    /// resident, a load that needs its VRAM fails rather than evicting it — and
    /// the error names the pinning, not an empty card.
    #[tokio::test]
    async fn test_pinned_model_is_never_evicted() {
        // 10 GB card, KV uncapped: model-a (pinned) grabs the whole card.
        let mut config = make_config(&[("model-a", 6.0), ("model-b", 6.0)], 10.0);
        config
            .models
            .entry("model-a".to_owned())
            .or_default()
            .policy = config::ModelPolicy {
            pinned: true,
            priority: 0,
            ..Default::default()
        };
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();

        mm.load_model("model-a").await.unwrap();
        let err = mm.load_model("model-b").await.unwrap_err();

        assert!(
            err.to_string().contains("pinned"),
            "error must name the pinning: {err}"
        );
        assert_eq!(
            mm.read_models("test").get("model-a").unwrap().state,
            ModelState::Ready,
            "the pinned model must still be resident"
        );
    }

    /// D3 (the project's internal engineering log): a model
    /// within its eviction-grace window is protected exactly like a pinned
    /// one — a load that needs its VRAM fails rather than evicting it, even
    /// though nothing marks it `pinned`/non-`evictable`.
    #[tokio::test]
    async fn test_grace_window_protects_a_just_used_model() {
        let mut config = make_config(&[("model-a", 6.0), ("model-b", 6.0)], 10.0);
        config.eviction_grace_secs = 300.0; // long enough that "just loaded" stays within it
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();

        mm.load_model("model-a").await.unwrap();
        let err = mm.load_model("model-b").await.unwrap_err();

        assert!(
            err.to_string().contains("grace window"),
            "error must mention the grace window: {err}"
        );
        assert_eq!(
            mm.read_models("test").get("model-a").unwrap().state,
            ModelState::Ready,
            "the recently-used model must still be resident"
        );
    }

    /// D3: `eviction_grace_secs: 0.0` (global or per-model override) opts a
    /// model out of grace protection entirely — pure LRU, unchanged from
    /// pre-D3 behavior. Regression guard for the "every existing test uses
    /// grace 0.0" assumption `make_config` relies on.
    #[tokio::test]
    async fn test_zero_grace_secs_disables_grace_protection() {
        let mut config = make_config(&[("model-a", 6.0), ("model-b", 6.0)], 10.0);
        config.eviction_grace_secs = 0.0;
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();

        mm.load_model("model-a").await.unwrap();
        mm.load_model("model-b").await.unwrap(); // must evict model-a, not fail

        assert_eq!(
            mm.read_models("test").get("model-a").unwrap().state,
            ModelState::NotLoaded,
            "no grace configured — model-a is a normal LRU victim"
        );
    }

    /// D2: a model registered in the realtime serving set (an active
    /// session depends on it) is protected exactly like a pinned one,
    /// independent of the first-writer-wins `voice_pin` cold-start
    /// suggestion. Deregistering releases the protection.
    #[tokio::test]
    async fn test_serving_set_protects_a_model_an_active_session_depends_on() {
        let config = make_config(&[("model-a", 6.0), ("model-b", 6.0)], 10.0);
        let voice_pin = Arc::new(crate::voice_pin::VoicePinManager::default());
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap()
            .with_voice_pin(Arc::clone(&voice_pin));

        mm.load_model("model-a").await.unwrap();
        voice_pin.register_serving(["model-a"]);

        let err = mm.load_model("model-b").await.unwrap_err();
        assert!(
            err.to_string().contains("active realtime use"),
            "error must mention active realtime use: {err}"
        );
        assert_eq!(
            mm.read_models("test").get("model-a").unwrap().state,
            ModelState::Ready,
            "still resident — a live session depends on it"
        );

        // Deregistering releases the protection: the same load now succeeds.
        voice_pin.deregister_serving(["model-a"]);
        mm.load_model("model-b").await.unwrap();
        assert_eq!(
            mm.read_models("test").get("model-a").unwrap().state,
            ModelState::NotLoaded,
            "no longer protected — becomes the LRU victim"
        );
    }

    /// D5 step 1: an already-`Ready` requested model is granted outright,
    /// no substitution, no background load.
    #[tokio::test]
    async fn test_resolve_realtime_llm_grants_a_ready_requested_model() {
        let mm = Arc::new(make_mm(&[("model-a", 2.0)], 10.0).await);
        mm.load_model("model-a").await.unwrap();

        let res = mm.resolve_realtime_llm(Some("model-a"));
        assert_eq!(res.model_id, "model-a");
        assert_eq!(res.source, "requested");
        assert_eq!(res.reason, None);
        assert!(!res.loading);
    }

    /// D5 step 3: when the requested model can't be granted (it would evict
    /// a protected model) but another Ready model clears the
    /// viable-minimum floor (the default "any `text_gen`/vision" floor, no
    /// `realtime_viable_minimum` configured), substitute it instead of
    /// refusing outright.
    #[tokio::test]
    async fn test_resolve_realtime_llm_substitutes_when_request_would_evict_protected() {
        let mut config = make_config(&[("model-a", 6.0), ("model-b", 6.0)], 10.0);
        config
            .models
            .entry("model-a".to_owned())
            .or_default()
            .policy = config::ModelPolicy {
            pinned: true,
            ..Default::default()
        };
        let mm = Arc::new(
            ModelManager::new(config, Arc::new(MockEngineFactory::default()))
                .await
                .unwrap(),
        );
        mm.load_model("model-a").await.unwrap(); // pinned, grabs the whole 10 GB card

        let res = mm.resolve_realtime_llm(Some("model-b"));
        assert_eq!(
            res.model_id, "model-a",
            "substituted to the only Ready floor-clearing model"
        );
        assert_eq!(res.source, "substituted");
        assert_eq!(res.reason, Some("would_evict_protected"));
        assert!(!res.loading);
    }

    /// D5 step 4: nothing Ready clears the floor, but a configured
    /// `realtime_defaults.llm_model` exists — kick a background load for it
    /// (`loading: true`) rather than refusing, and the load actually
    /// completes (not a fire-and-forget that silently fails).
    #[tokio::test]
    async fn test_resolve_realtime_llm_background_loads_the_configured_default() {
        let mut config = make_config(&[("model-a", 2.0)], 10.0);
        config.realtime_defaults = Some(config::RealtimeDefaults {
            stt_model: None,
            llm_model: Some("model-a".to_owned()),
            tts_model: None,
            embed_model: None,
        });
        let mm = Arc::new(
            ModelManager::new(config, Arc::new(MockEngineFactory::default()))
                .await
                .unwrap(),
        );

        let res = mm.resolve_realtime_llm(None);
        assert_eq!(res.model_id, "model-a");
        assert_eq!(res.source, "default");
        assert_eq!(res.reason, Some("not_resident_no_headroom"));
        assert!(res.loading);

        for _ in 0..50 {
            if mm.is_ready("model-a") {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(
            mm.is_ready("model-a"),
            "the background load kicked off by resolve_realtime_llm should complete"
        );
    }

    /// RTCC (the project's internal engineering log): `resolve_
    /// realtime_slot` generalizes D5 step 1 to the STT slot — an
    /// already-`Ready` requested STT model is granted outright, same rule
    /// as the LLM slot.
    #[tokio::test]
    async fn test_resolve_realtime_slot_stt_grants_a_ready_requested_model() {
        let mm = Arc::new(make_mm(&[("stt-a", 1.0)], 10.0).await);
        mm.load_model("stt-a").await.unwrap();

        let res = mm.resolve_realtime_slot(RealtimeSlot::Stt, Some("stt-a"));
        assert_eq!(res.model_id, "stt-a");
        assert_eq!(res.source, "requested");
        assert_eq!(res.reason, None);
        assert!(!res.loading);
    }

    /// RTCC: `resolve_realtime_slot` generalizes D5 step 3 to the TTS
    /// slot — a pinned Ready TTS model substitutes for an infeasible
    /// request instead of refusing outright, same as the LLM slot.
    #[tokio::test]
    async fn test_resolve_realtime_slot_tts_substitutes_when_request_would_evict_protected() {
        let mut config = make_config(&[("tts-a", 6.0), ("tts-b", 6.0)], 10.0);
        config.models.entry("tts-a".to_owned()).or_default().policy = config::ModelPolicy {
            pinned: true,
            ..Default::default()
        };
        let mm = Arc::new(
            ModelManager::new(config, Arc::new(MockEngineFactory::default()))
                .await
                .unwrap(),
        );
        mm.load_model("tts-a").await.unwrap(); // pinned, grabs the whole 10 GB card

        let res = mm.resolve_realtime_slot(RealtimeSlot::Tts, Some("tts-b"));
        assert_eq!(
            res.model_id, "tts-a",
            "substituted to the only Ready TTS model"
        );
        assert_eq!(res.source, "substituted");
        assert_eq!(res.reason, Some("would_evict_protected"));
        assert!(!res.loading);
    }

    /// RTCC: the STT/TTS floor is exact kind-matching, D4's whole floor for
    /// those slots — a Ready `TextGen` model must never be offered as a
    /// substitute for an unresolvable STT request, even when it's the only
    /// Ready model on the box. Guards against `best_ready_viable`'s filter
    /// silently degrading to "any Ready model" for slots without a
    /// configured `realtime_viable_minimum` section.
    #[tokio::test]
    async fn test_resolve_realtime_slot_stt_never_substitutes_a_different_kind() {
        let mm = Arc::new(make_mm(&[("llm-a", 2.0)], 10.0).await);
        mm.load_model("llm-a").await.unwrap();

        let res = mm.resolve_realtime_slot(RealtimeSlot::Stt, Some("missing-stt"));
        assert_eq!(
            res.model_id, "missing-stt",
            "reports the literal request honestly — no cross-kind substitution"
        );
        assert_eq!(res.source, "substituted");
        assert_eq!(res.reason, Some("not_resident_no_headroom"));
    }

    /// RTCC: `known_model_kind` — the lookup `request_model_load` uses to
    /// classify an explicit load target into the right slot before
    /// resolving it — reports the mock factory's name-based kind detection
    /// for every registered model (present as a record from construction,
    /// before any load), and `None` for a genuinely unknown id.
    #[tokio::test]
    async fn test_known_model_kind_reports_registered_kind() {
        let mm = make_mm(&[("stt-a", 1.0), ("tts-a", 1.0), ("llm-a", 2.0)], 10.0).await;
        assert_eq!(mm.known_model_kind("stt-a"), Some(ModelKind::Stt));
        assert_eq!(mm.known_model_kind("tts-a"), Some(ModelKind::Tts));
        assert_eq!(mm.known_model_kind("llm-a"), Some(ModelKind::TextGen));
        assert_eq!(mm.known_model_kind("does-not-exist"), None);
    }

    /// RTCC: the `Embed` slot grants (already-`Ready`, or admits-and-loads)
    /// exactly like every other slot, but D2/framing decision 3 exclude
    /// `embed_model` from the courtesy floor — "convenience only." This is
    /// the load-bearing regression guard for that: even with a Ready
    /// `TextGen` model resident and a configured `realtime_defaults.
    /// llm_model`, an infeasible embed request must neither substitute a
    /// different model nor trigger a background load — it must honestly
    /// report the literal request failed.
    #[tokio::test]
    async fn test_resolve_realtime_slot_embed_never_substitutes_or_auto_loads() {
        let mut config = make_config(&[("llm-a", 2.0)], 10.0);
        config.realtime_defaults = Some(config::RealtimeDefaults {
            stt_model: None,
            llm_model: Some("llm-a".to_owned()),
            tts_model: None,
            embed_model: None,
        });
        let mm = Arc::new(
            ModelManager::new(config, Arc::new(MockEngineFactory::default()))
                .await
                .unwrap(),
        );
        mm.load_model("llm-a").await.unwrap();

        let res = mm.resolve_realtime_slot(RealtimeSlot::Embed, Some("missing-embed"));
        assert_eq!(
            res.model_id, "missing-embed",
            "no cross-slot substitution — reports the literal request honestly"
        );
        assert_eq!(res.source, "substituted");
        assert!(!res.loading, "grant-only — never auto-loads a default");
    }

    /// RTCC: the `Embed` slot still grants an already-`Ready` requested
    /// model outright — "grant-only" means no substitution/default, not "no
    /// grant at all."
    #[tokio::test]
    async fn test_resolve_realtime_slot_embed_grants_a_ready_requested_model() {
        let mm = Arc::new(make_mm(&[("embed-a", 1.0)], 10.0).await);
        mm.load_model("embed-a").await.unwrap();

        let res = mm.resolve_realtime_slot(RealtimeSlot::Embed, Some("embed-a"));
        assert_eq!(res.model_id, "embed-a");
        assert_eq!(res.source, "requested");
        assert!(!res.loading);
    }

    /// Pinning is a HARD exclude that overrides LRU: a pinned model is spared
    /// even when it is the least-recently-used; the idle non-pinned model is the
    /// victim instead.
    #[tokio::test]
    async fn test_pinning_overrides_lru_order() {
        // Pin model-a, the model loaded FIRST (oldest last_used). Pure LRU would
        // evict it; the pin must redirect eviction to model-b.
        let mm = make_mm_with_policies(&[("model-a", true, 0)]).await;
        mm.load_model("model-a").await.unwrap();
        mm.load_model("model-b").await.unwrap();
        mm.load_model("model-c").await.unwrap(); // forces one eviction

        assert_eq!(
            mm.read_models("test").get("model-a").unwrap().state,
            ModelState::Ready,
            "pinned model-a (oldest) must be spared"
        );
        assert_eq!(
            mm.read_models("test").get("model-b").unwrap().state,
            ModelState::NotLoaded,
            "idle non-pinned model-b is the victim"
        );
        assert_eq!(
            mm.read_models("test").get("model-c").unwrap().state,
            ModelState::Ready,
        );
    }

    /// A non-evictable model (Slice 3, `evictable: false`, NOT pinned) is
    /// excluded from eviction exactly like a pinned one: with only it resident,
    /// a load that needs its VRAM fails and the error names the protection.
    #[tokio::test]
    async fn test_non_evictable_model_is_never_evicted() {
        let mut config = make_config(&[("model-a", 6.0), ("model-b", 6.0)], 10.0);
        config
            .models
            .entry("model-a".to_owned())
            .or_default()
            .policy = config::ModelPolicy {
            evictable: false,
            ..Default::default()
        };
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();

        mm.load_model("model-a").await.unwrap();
        let err = mm.load_model("model-b").await.unwrap_err();

        assert!(
            err.to_string().contains("protected"),
            "error must name the protection (non-evictable): {err}"
        );
        assert_eq!(
            mm.read_models("test").get("model-a").unwrap().state,
            ModelState::Ready,
            "the non-evictable model must still be resident"
        );
    }

    /// `evictable: false` is a HARD exclude overriding LRU, just like pinning:
    /// the oldest non-evictable model is spared and the idle evictable one is
    /// the victim.
    #[tokio::test]
    async fn test_non_evictable_overrides_lru_order() {
        let mut config = make_config(
            &[("model-a", 5.0), ("model-b", 5.0), ("model-c", 5.0)],
            17.0,
        );
        config.cache_size_gb = 2.0;
        config
            .models
            .entry("model-a".to_owned())
            .or_default()
            .policy = config::ModelPolicy {
            evictable: false,
            ..Default::default()
        };
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();

        mm.load_model("model-a").await.unwrap(); // oldest
        mm.load_model("model-b").await.unwrap();
        mm.load_model("model-c").await.unwrap(); // forces one eviction

        assert_eq!(
            mm.read_models("test").get("model-a").unwrap().state,
            ModelState::Ready,
            "non-evictable model-a (oldest) must be spared"
        );
        assert_eq!(
            mm.read_models("test").get("model-b").unwrap().state,
            ModelState::NotLoaded,
            "idle evictable model-b is the victim"
        );
    }

    /// `priority` is a soft ordering ABOVE LRU: among idle non-pinned candidates,
    /// the lower-priority model is evicted first even when it is newer than a
    /// higher-priority one.
    #[tokio::test]
    async fn test_priority_evicts_low_before_high_over_lru() {
        // model-a high priority but loaded FIRST (oldest); model-b default
        // priority loaded second (newest). LRU alone evicts the oldest (a);
        // priority must spare the high-priority a and evict b.
        let mm = make_mm_with_policies(&[("model-a", false, 10), ("model-b", false, 0)]).await;
        mm.load_model("model-a").await.unwrap();
        mm.load_model("model-b").await.unwrap();
        mm.load_model("model-c").await.unwrap();

        assert_eq!(
            mm.read_models("test").get("model-a").unwrap().state,
            ModelState::Ready,
            "high-priority model-a must be spared despite being oldest"
        );
        assert_eq!(
            mm.read_models("test").get("model-b").unwrap().state,
            ModelState::NotLoaded,
            "low-priority model-b is evicted first"
        );
    }

    // ---- Co-residency Slice 3c: on-demand load from the chat path ------

    /// Helper: a 1-model manager (Arc, for `request_on_demand_load`) whose model
    /// carries the given load policy.
    async fn make_on_demand_mm(load: config::LoadPolicy) -> Arc<ModelManager> {
        let mut config = make_config(&[("model-a", 5.0)], 17.0);
        config
            .models
            .entry("model-a".to_owned())
            .or_default()
            .policy = config::ModelPolicy {
            load,
            ..Default::default()
        };
        Arc::new(
            ModelManager::new(config, Arc::new(MockEngineFactory::default()))
                .await
                .unwrap(),
        )
    }

    /// An eager (default) `NotLoaded` model is NOT auto-loaded: the chat path
    /// gets `NotApplicable` (→ bare 503) and the model stays `NotLoaded`.
    #[tokio::test]
    async fn test_on_demand_eager_model_is_not_auto_loaded() {
        let mm = make_on_demand_mm(config::LoadPolicy::Eager).await;
        assert_eq!(
            mm.request_on_demand_load("model-a"),
            OnDemandLoad::NotApplicable
        );
        assert_eq!(
            mm.read_models("test").get("model-a").unwrap().state,
            ModelState::NotLoaded,
            "an eager model must not be loaded on demand"
        );
    }

    /// An `on_demand` `NotLoaded` model is loaded lazily: the call returns
    /// `Loading` and the detached task drives the model to `Ready`.
    #[tokio::test]
    async fn test_on_demand_model_loads_in_background() {
        let mm = make_on_demand_mm(config::LoadPolicy::OnDemand).await;
        assert_eq!(mm.request_on_demand_load("model-a"), OnDemandLoad::Loading);

        // The load runs on a detached task; poll until Ready (mock load is fast).
        let mut ready = false;
        for _ in 0..200 {
            if mm.read_models("test").get("model-a").unwrap().state == ModelState::Ready {
                ready = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        assert!(ready, "on-demand background load must reach Ready");
    }

    /// An `on_demand` model already in flight (Loading) returns `Loading`
    /// without disturbing the state — no second load is started.
    #[tokio::test]
    async fn test_on_demand_already_loading_returns_loading_no_respawn() {
        let mm = make_on_demand_mm(config::LoadPolicy::OnDemand).await;
        mm.write_models("test").get_mut("model-a").unwrap().state = ModelState::Loading;
        assert_eq!(mm.request_on_demand_load("model-a"), OnDemandLoad::Loading);
        assert_eq!(
            mm.read_models("test").get("model-a").unwrap().state,
            ModelState::Loading,
            "state must be untouched — the advisory check skips the spawn"
        );
    }

    /// `check_admission` on an unknown model returns `NotFound`.
    #[tokio::test]
    async fn test_check_admission_unknown_model() {
        let mm = make_mm(&[("model-a", 5.0)], 17.0).await;
        let err = mm.check_admission("ghost").unwrap_err();
        assert_eq!(
            err.downcast_ref::<ModelError>(),
            Some(&ModelError::NotFound("ghost".to_owned()))
        );
    }

    /// A model that fits in free VRAM reports `fits` with no eviction and does
    /// not mutate any state (pure dry run).
    #[tokio::test]
    async fn test_check_admission_fits_without_eviction() {
        let mm = make_mm_with_policies(&[]).await;
        let check = mm.check_admission("model-a").unwrap();
        assert!(check.fits);
        assert!(
            check.would_evict.is_empty(),
            "empty card → nothing to evict"
        );
        assert!(!check.already_loaded);
        // needed = weight(5) + min_kv(0) + margin(0); free = full 17 GB card.
        assert!((check.needed_gb - 5.0).abs() < 1e-9);
        assert!((check.free_gb - 17.0).abs() < 1e-9);
        // Dry run: model-a is still NotLoaded.
        assert_eq!(
            mm.read_models("test").get("model-a").unwrap().state,
            ModelState::NotLoaded,
        );
    }

    /// When the card is full, the check reports the residents that WOULD be
    /// evicted — in eviction order — and changes nothing.
    #[tokio::test]
    async fn test_check_admission_reports_would_evict() {
        let mm = make_mm_with_policies(&[]).await;
        mm.load_model("model-a").await.unwrap(); // reserves 5 + 2 = 7
        mm.load_model("model-b").await.unwrap(); // reserves 7 → 14 used, 3 free

        // model-c needs 5 GB; only 3 free → evicting the LRU victim (model-a)
        // frees 7 GB, enough. would_evict lists exactly model-a.
        let check = mm.check_admission("model-c").unwrap();
        assert!(check.fits);
        assert_eq!(check.would_evict, vec!["model-a".to_owned()]);
        assert!((check.free_gb - 3.0).abs() < 1e-9);

        // Nothing was actually evicted — both residents stay Ready.
        assert_eq!(
            mm.read_models("test").get("model-a").unwrap().state,
            ModelState::Ready,
        );
        assert_eq!(
            mm.read_models("test").get("model-b").unwrap().state,
            ModelState::Ready,
        );
    }

    /// A pinned resident is excluded from the eviction plan: if freeing only the
    /// non-pinned residents is not enough, the check reports `fits: false`.
    #[tokio::test]
    async fn test_check_admission_pinned_blocks_fit() {
        // Pin model-a. On a 10 GB card (KV capped 2) it reserves 7, leaving 3.
        let mut config = make_config(&[("model-a", 5.0), ("model-b", 5.0)], 10.0);
        config.cache_size_gb = 2.0;
        config
            .models
            .entry("model-a".to_owned())
            .or_default()
            .policy = config::ModelPolicy {
            pinned: true,
            priority: 0,
            ..Default::default()
        };
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();
        mm.load_model("model-a").await.unwrap();

        // model-b needs 5; 3 free; the only resident (model-a) is pinned → no
        // eviction candidate → cannot fit.
        let check = mm.check_admission("model-b").unwrap();
        assert!(!check.fits, "pinned resident blocks the load");
        assert!(check.would_evict.is_empty());
    }

    /// `check_admission` on an already-loaded model reports `already_loaded` and
    /// a trivial fit.
    #[tokio::test]
    async fn test_check_admission_already_loaded() {
        let mm = make_mm_with_policies(&[]).await;
        mm.load_model("model-a").await.unwrap();
        let check = mm.check_admission("model-a").unwrap();
        assert!(check.already_loaded);
        assert!(check.fits);
        assert!(check.would_evict.is_empty());
    }

    /// With VRAM gating disabled (`total_vram_gb = 0.0`) every model fits and the
    /// check never proposes an eviction.
    #[tokio::test]
    async fn test_check_admission_gating_disabled_always_fits() {
        let mm = make_mm(&[("model-a", 5.0)], 0.0).await;
        let check = mm.check_admission("model-a").unwrap();
        assert!(check.fits, "gating disabled → always fits");
        assert!(check.would_evict.is_empty());
    }

    /// T5.5/#34: `new` rejects a preload model absent from `vram_gb` — the
    /// defense behind `Config::validate` for any caller that bypasses it. No
    /// silent 0.0-GB registration that would load uncounted.
    #[tokio::test]
    async fn test_new_rejects_preload_without_vram_estimate() {
        let mut config = make_config(&[("known", 5.0)], 22.5);
        config.preload = vec!["ghost".to_owned()];
        // `unwrap_err` would require `ModelManager: Debug` (it isn't) — match.
        let err = match ModelManager::new(config, Arc::new(MockEngineFactory::default())).await {
            Ok(_) => panic!("expected error for an unsized preload model"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("ghost"),
            "must name the unsized preload model: {err}"
        );
    }

    // ---- list_models / list_ready_models ----------------------------------

    /// `list_models` returns all known models regardless of state.
    #[tokio::test]
    async fn test_list_models_returns_all_known() {
        let mm = make_mm(&[("model-a", 6.0), ("model-b", 6.0)], 22.5).await;
        mm.load_model("model-a").await.unwrap();

        let infos = mm.list_models();
        assert_eq!(infos.len(), 2);
    }

    /// `list_models` reports each model's resolved device — the global
    /// `config.device` default for models with no override, the per-model
    /// policy override otherwise. Surfaced via `GET /v1/admin/models`.
    #[tokio::test]
    async fn test_list_models_reports_resolved_device() {
        let mut config = make_config(&[("model-a", 6.0), ("model-b", 6.0)], 22.5);
        config
            .models
            .entry("model-b".to_owned())
            .or_default()
            .policy = config::ModelPolicy {
            device: Some("NPU".to_owned()),
            ..Default::default()
        };
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();

        let infos = mm.list_models();
        let a = infos.iter().find(|i| i.id == "model-a").unwrap();
        let b = infos.iter().find(|i| i.id == "model-b").unwrap();
        assert_eq!(a.device, "CPU", "no override — falls back to config.device");
        assert_eq!(b.device, "NPU", "per-model policy override wins");
    }

    /// `list_ready_models` returns only Ready model IDs.
    #[tokio::test]
    async fn test_list_ready_models_returns_only_ready() {
        let mm = make_mm(&[("model-a", 6.0), ("model-b", 6.0)], 22.5).await;
        mm.load_model("model-a").await.unwrap();

        let ready = mm.list_ready_models();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0], "model-a");
    }

    // ---- next_id ---------------------------------------------------------

    /// `next_id` returns monotonically increasing values.
    #[tokio::test]
    async fn test_next_id_is_monotonic() {
        let mm = make_mm(&[], 0.0).await;
        let a = mm.next_id();
        let b = mm.next_id();
        let c = mm.next_id();
        assert!(a < b && b < c, "next_id must be strictly increasing");
    }

    // ---- L3: GPU context poisoning (gpu_poisoned) -------------------------

    /// After a `CL_OUT_OF_RESOURCES` load failure the poison flag is set and
    /// all subsequent `load_model` calls return `GpuPoisoned` immediately —
    /// no factory call, no GPU touch.
    #[tokio::test]
    async fn test_gpu_poisoned_after_oom_load_failure() {
        let config = make_config(&[("bad-model", 5.5)], 22.5);
        let factory = Arc::new(MockEngineFactory {
            fail_with_oom: true,
            ..Default::default()
        });
        let mm = ModelManager::new(config, factory).await.unwrap();

        let first = mm.load_model("bad-model").await;
        assert!(
            first.is_err(),
            "OOM factory failure must propagate as error"
        );

        assert!(
            mm.gpu_poisoned.load(Ordering::Acquire),
            "gpu_poisoned must be set after CL_OUT_OF_RESOURCES failure"
        );

        // Any subsequent load attempt — same or different model — is refused.
        let second = mm.load_model("bad-model").await.unwrap_err();
        let model_err = second
            .downcast_ref::<ModelError>()
            .expect("must be ModelError");
        assert_eq!(*model_err, ModelError::GpuPoisoned);
    }

    /// A plain (non-OOM) factory failure must NOT set the poison flag — the
    /// GPU context is still healthy; the failure was a bad model path or config.
    #[tokio::test]
    async fn test_regular_failure_does_not_poison_gpu() {
        let config = make_config(&[("bad-model", 5.5)], 22.5);
        let factory = Arc::new(MockEngineFactory {
            should_fail: true,
            ..Default::default()
        });
        let mm = ModelManager::new(config, factory).await.unwrap();

        let err = mm.load_model("bad-model").await;
        assert!(err.is_err(), "factory failure must propagate");

        assert!(
            !mm.gpu_poisoned.load(Ordering::Acquire),
            "gpu_poisoned must remain false after a non-OOM failure"
        );
    }

    /// The shared classifier recognises both OOM-class markers and nothing else
    /// — the load path and the inference path must agree on what "poison" means.
    #[test]
    fn is_gpu_poison_error_matches_only_oom_markers() {
        assert!(is_gpu_poison_error(
            "ov_embed_documents failed: CL_OUT_OF_RESOURCES (-5)"
        ));
        assert!(is_gpu_poison_error(
            "context wedged: CL_INVALID_EVENT (-58)"
        ));
        assert!(!is_gpu_poison_error("ov_embed_documents failed: bad input"));
        assert!(!is_gpu_poison_error(""));
    }

    /// Both wedge-marker parsers share one `extract_marker_field` helper —
    /// verify each reads its own field correctly and ignores the other's,
    /// and that malformed/absent input is `None`, never a guessed value.
    #[test]
    fn kv_wedge_field_parsers_read_their_own_field_only() {
        let vlm_msg = format!("{VLM_KV_WEDGE_MARKER} — observed_prompt_tokens=37847");
        assert_eq!(kv_wedge_observed_prompt_tokens(&vlm_msg), Some(37_847));
        assert_eq!(pool_exhausted_active_requests(&vlm_msg), None);

        let pool_msg = format!(
            "generation ignored by the CB scheduler — {POOL_EXHAUSTED_MARKER} \
             (observed_prompt_tokens=37166 active_requests=1)"
        );
        assert_eq!(kv_wedge_observed_prompt_tokens(&pool_msg), Some(37_166));
        assert_eq!(pool_exhausted_active_requests(&pool_msg), Some(1));

        let congested_msg = format!(
            "generation ignored by the CB scheduler — {POOL_EXHAUSTED_MARKER} \
             (observed_prompt_tokens=12000 active_requests=5)"
        );
        assert_eq!(pool_exhausted_active_requests(&congested_msg), Some(5));

        assert_eq!(kv_wedge_observed_prompt_tokens("no marker here"), None);
        assert_eq!(pool_exhausted_active_requests("no marker here"), None);
        assert_eq!(
            kv_wedge_observed_prompt_tokens("observed_prompt_tokens=abc"),
            None
        );
    }

    /// `compute_max_prompt_tokens` applies `FORMULA_SAFETY_MARGIN` to the raw
    /// K+V byte-math result before ever comparing it against a ratchet — no
    /// ratchet sidecar exists here, so the returned value must be exactly
    /// the margin-adjusted formula result, not the raw one.
    #[test]
    fn compute_max_prompt_tokens_applies_the_formula_safety_margin() {
        let dir = std::env::temp_dir().join(format!(
            "rv-max-prompt-margin-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // Qwen3-4B-int4-ov-style dense GQA config: layers=36, kv_heads=8,
        // hidden=2560, heads=32 → head_dim=80. Raw bytes/token (u8) =
        // 36*8*80*2*2/2 = 46_080.
        std::fs::write(
            dir.join("config.json"),
            r#"{"num_hidden_layers":36,"num_attention_heads":32,"num_key_value_heads":8,"hidden_size":2560}"#,
        )
        .unwrap();

        let kv_gb = 3.0_f64;
        let bytes_per_token = 46_080.0_f64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let expected = (kv_gb * 1_073_741_824.0 / bytes_per_token
            * ModelManager::FORMULA_SAFETY_MARGIN) as usize;

        let (result, pool_capacity) =
            ModelManager::compute_max_prompt_tokens("qwen3-4b-int4-ov", &dir, kv_gb, "u8");
        assert_eq!(result, expected);
        assert!(
            result < 69_905,
            "margin must actually shrink the raw (unmargined) estimate: {result}"
        );
        assert_eq!(
            pool_capacity, expected,
            "no native/ratchet clamp applies here — pool_capacity_tokens must equal \
             the same margin-adjusted formula result as max_prompt_tokens"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// `compute_max_prompt_tokens` clamps to the model's native
    /// `max_position_embeddings` when it's tighter than the (already
    /// margin-adjusted) formula result — the exact `qwen3-4b-int4-ov` shape
    /// that started this investigation: formula computed 142k-306k tokens
    /// depending on pool size, against a real trained ceiling of 40,960
    /// (the project's internal engineering log). The
    /// clamp itself must be unmargined — `FORMULA_SAFETY_MARGIN` never
    /// applies to a hard model property.
    #[test]
    fn compute_max_prompt_tokens_clamps_to_native_context_limit() {
        let dir = std::env::temp_dir().join(format!(
            "rv-max-prompt-native-clamp-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // Same dense GQA shape as the margin test above, plus a native ceiling
        // tighter than the margin-adjusted formula result (59,419).
        std::fs::write(
            dir.join("config.json"),
            r#"{"num_hidden_layers":36,"num_attention_heads":32,"num_key_value_heads":8,
                "hidden_size":2560,"max_position_embeddings":40960,"rope_scaling":null}"#,
        )
        .unwrap();

        let kv_gb = 3.0_f64;
        let bytes_per_token = 46_080.0_f64; // same dense GQA shape as the margin test above
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let raw_formula_result = (kv_gb * 1_073_741_824.0 / bytes_per_token
            * ModelManager::FORMULA_SAFETY_MARGIN) as usize;

        let (result, pool_capacity) =
            ModelManager::compute_max_prompt_tokens("qwen3-4b-int4-ov", &dir, kv_gb, "u8");
        assert_eq!(
            result, 40_960,
            "native ceiling is tighter than the formula estimate — must win, unmargined"
        );
        assert_eq!(
            pool_capacity, raw_formula_result,
            "pool_capacity_tokens (the concurrent-admission ceiling) must stay the RAW, \
             unclamped formula estimate — the native-context clamp protects one request's \
             own sanity, not shared physical pool capacity (the Phi-4 case is exactly \
             this: a tiny native context next to a much larger physical pool)"
        );
        assert!(
            pool_capacity > result,
            "this test's whole point: the two must diverge when native context is the \
             tighter clamp, not silently stay equal"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A native `max_position_embeddings` looser than the formula result must
    /// not loosen the gate beyond what the formula itself would allow — the
    /// clamp only ever tightens.
    #[test]
    fn compute_max_prompt_tokens_ignores_native_limit_above_formula() {
        let dir = std::env::temp_dir().join(format!(
            "rv-max-prompt-native-noop-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"num_hidden_layers":36,"num_attention_heads":32,"num_key_value_heads":8,
                "hidden_size":2560,"max_position_embeddings":262144}"#,
        )
        .unwrap();

        let kv_gb = 3.0_f64;
        let bytes_per_token = 46_080.0_f64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let expected = (kv_gb * 1_073_741_824.0 / bytes_per_token
            * ModelManager::FORMULA_SAFETY_MARGIN) as usize;

        let (result, pool_capacity) =
            ModelManager::compute_max_prompt_tokens("qwen3-4b-int4-ov", &dir, kv_gb, "u8");
        assert_eq!(result, expected);
        assert_eq!(
            pool_capacity, expected,
            "native limit looser than the formula — no clamp applies, both values equal"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// T6.2: an inference-path OOM (flagged via `mark_gpu_poisoned`) trips the
    /// same L3 gate as a load-path OOM — the next load is refused with 503.
    #[tokio::test]
    async fn mark_gpu_poisoned_blocks_subsequent_loads() {
        let mm = make_mm(&[("m", 5.5)], 22.5).await;
        assert!(!mm.gpu_poisoned.load(Ordering::Acquire));

        mm.mark_gpu_poisoned("m", anyhow::anyhow!("CL_OUT_OF_RESOURCES (-5)"));
        assert!(mm.gpu_poisoned.load(Ordering::Acquire));

        let err = mm.load_model("m").await.unwrap_err();
        let model_err = err
            .downcast_ref::<ModelError>()
            .expect("must be ModelError");
        assert_eq!(*model_err, ModelError::GpuPoisoned);
    }

    /// L3 gate, eviction side (`dev/autotest/20260804_gpu_poisoned_eviction
    /// _crash.md`): once the GPU context is poisoned, evicting a `Ready`
    /// model is refused with `GpuPoisoned` instead of proceeding to drop its
    /// handle — dropping a pipeline on a broken `OpenCL` context can crash
    /// the whole process, not just fail the one request.
    #[tokio::test]
    async fn mark_gpu_poisoned_blocks_eviction_of_a_ready_model() {
        let mm = make_mm(&[("m", 5.5)], 22.5).await;
        mm.load_model("m").await.unwrap();
        assert!(!mm.gpu_poisoned.load(Ordering::Acquire));

        mm.mark_gpu_poisoned("m", anyhow::anyhow!("CL_OUT_OF_RESOURCES (-5)"));

        let err = mm.evict_model("m").await.unwrap_err();
        let model_err = err
            .downcast_ref::<ModelError>()
            .expect("must be ModelError");
        assert_eq!(*model_err, ModelError::GpuPoisoned);

        // The model must still be reported Ready — eviction never proceeded.
        let ready = mm.list_ready_models();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0], "m");
    }

    /// T6.2b: an OOM seen on the **streaming** path (via the SSE error helper)
    /// poisons the GPU context exactly like the non-stream path — the next load
    /// is then refused. Proves the streaming-frame wiring reaches the same gate.
    #[tokio::test]
    async fn streaming_oom_frame_poisons_gpu() {
        let mm = Arc::new(make_mm(&[("m", 5.5)], 22.5).await);
        assert!(!mm.gpu_poisoned.load(Ordering::Acquire));

        // The frame itself is discarded — we assert the side effect.
        let _ = crate::handlers::error::sse_inference_error_event(
            Some(&mm),
            "m",
            "ov_cb generate failed: /opt/models/m CL_OUT_OF_RESOURCES (-5)",
        );
        assert!(
            mm.gpu_poisoned.load(Ordering::Acquire),
            "streaming OOM must poison the GPU context"
        );

        let err = mm.load_model("m").await.unwrap_err();
        assert_eq!(
            err.downcast_ref::<ModelError>(),
            Some(&ModelError::GpuPoisoned)
        );
    }

    /// A non-OOM streaming error must NOT poison the context (no false positive
    /// that would wedge the node into restart-required on a benign failure).
    #[tokio::test]
    async fn streaming_generic_frame_does_not_poison_gpu() {
        let mm = Arc::new(make_mm(&[("m", 5.5)], 22.5).await);
        let _ = crate::handlers::error::sse_inference_error_event(
            Some(&mm),
            "m",
            "ov_cb generate failed: ragged batch",
        );
        assert!(
            !mm.gpu_poisoned.load(Ordering::Acquire),
            "a benign streaming error must not poison the GPU"
        );
    }

    // ---- spawn_kv_wedge_recovery ---------------------------------------

    /// Waits (bounded) for a `spawn_kv_wedge_recovery` background task to
    /// release its claim on `model_id` — the detached `tokio::spawn`d task has
    /// no `JoinHandle` these tests can await directly, so this polls the
    /// (private, same-module-accessible) inflight set instead. `MockEngineFactory`
    /// evict/load are effectively instant, so the loop exits almost immediately
    /// when the recovery genuinely completes; a real hang exhausts the bound and
    /// fails the assertion that follows, rather than hanging the test suite.
    async fn wait_for_wedge_recovery_to_finish(mm: &ModelManager, model_id: &str) {
        for _ in 0..200 {
            if !mm
                .kv_wedge_recovery_inflight
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .contains(model_id)
            {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        panic!("VLM KV-wedge recovery for {model_id} did not release its claim in time");
    }

    /// First-wedge-wins: a second call for the SAME model, made before the
    /// first call's spawned recovery task has had a chance to run (no `.await`
    /// between them — `#[tokio::test]` defaults to a single-threaded runtime,
    /// so the spawned task cannot start until this test function itself
    /// yields), must be a no-op rather than double-claiming or re-triggering a
    /// second evict+reload race (Fable's "Hazard A").
    #[tokio::test]
    async fn spawn_kv_wedge_recovery_second_call_is_a_no_op_while_first_in_flight() {
        let mm = Arc::new(make_mm(&[("vlm-1", 5.5)], 22.5).await);
        mm.load_model("vlm-1").await.unwrap();

        mm.spawn_kv_wedge_recovery("vlm-1", 5_000);
        assert!(
            mm.kv_wedge_recovery_inflight
                .lock()
                .unwrap()
                .contains("vlm-1"),
            "first call must claim the model id synchronously, before any .await"
        );

        mm.spawn_kv_wedge_recovery("vlm-1", 1);
        assert_eq!(
            mm.kv_wedge_recovery_inflight.lock().unwrap().len(),
            1,
            "a second concurrent call for the same model must not double-claim"
        );

        wait_for_wedge_recovery_to_finish(&mm, "vlm-1").await;
    }

    /// A later, genuinely NEW wedge (after the first recovery finished and
    /// released its claim) must be able to trigger recovery again — the claim
    /// is per-episode, not a permanent lockout.
    #[tokio::test]
    async fn spawn_kv_wedge_recovery_claims_again_after_prior_episode_finishes() {
        let mm = Arc::new(make_mm(&[("vlm-1", 5.5)], 22.5).await);
        mm.load_model("vlm-1").await.unwrap();

        mm.spawn_kv_wedge_recovery("vlm-1", 5_000);
        wait_for_wedge_recovery_to_finish(&mm, "vlm-1").await;

        // Model was reloaded by the first episode — a second, later wedge is
        // a fresh claim, not blocked by the first's (already-released) one.
        mm.spawn_kv_wedge_recovery("vlm-1", 5_000);
        assert!(
            mm.kv_wedge_recovery_inflight
                .lock()
                .unwrap()
                .contains("vlm-1"),
            "a fresh wedge episode must be able to claim again after the prior one finished"
        );
        wait_for_wedge_recovery_to_finish(&mm, "vlm-1").await;
    }

    /// An implausibly small `observed_prompt_tokens` (far below 10% of the
    /// ceiling already in effect — Fable's "Hazard B": stale/frozen
    /// `perf_metrics` from a queued victim request, not a real observation)
    /// must NOT be written to the ratchet sidecar, even though recovery
    /// (evict+reload) still proceeds.
    #[tokio::test]
    async fn spawn_kv_wedge_recovery_skips_ratchet_write_below_floor() {
        let dir =
            std::env::temp_dir().join(format!("rv-wedge-floor-{}-{}", std::process::id(), line!()));
        std::fs::create_dir_all(dir.join("vlm-1")).unwrap();
        let mut config = make_config(&[("vlm-1", 5.5)], 22.5);
        config.models_dir = dir.clone();
        let mm = Arc::new(
            ModelManager::new(config, Arc::new(MockEngineFactory::default()))
                .await
                .unwrap(),
        );
        mm.load_model("vlm-1").await.unwrap();
        // The mock factory's `config.json`-less model gets max_prompt_tokens
        // == 0 from the L0-gate formula — force a realistic non-zero ceiling
        // so the 10%-of-ceiling floor check has something to compare against.
        mm.write_models("test setup")
            .get_mut("vlm-1")
            .unwrap()
            .max_prompt_tokens = 100_000;

        mm.spawn_kv_wedge_recovery("vlm-1", 50); // far below the 10,000 floor
        wait_for_wedge_recovery_to_finish(&mm, "vlm-1").await;

        assert_eq!(
            template::read_kv_capacity_ratchet(&dir.join("vlm-1"), 1.0),
            None,
            "an implausibly small observation must not be written to the ratchet"
        );
        // Recovery itself still ran despite skipping the ratchet write.
        assert_eq!(
            mm.read_models("test").get("vlm-1").unwrap().state,
            ModelState::Ready,
            "evict+reload must still happen even when the ratchet write is skipped"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A plausible `observed_prompt_tokens` (at/above the 10% floor) IS
    /// written to the ratchet sidecar, discounted by the 0.95 safety margin,
    /// tagged with the `kv_cache_gb` active at learn time (Fable's pool-size
    /// scaling fix).
    #[tokio::test]
    async fn spawn_kv_wedge_recovery_writes_ratchet_for_a_plausible_observation() {
        let dir =
            std::env::temp_dir().join(format!("rv-wedge-write-{}-{}", std::process::id(), line!()));
        std::fs::create_dir_all(dir.join("vlm-1")).unwrap();
        let mut config = make_config(&[("vlm-1", 5.5)], 22.5);
        config.models_dir = dir.clone();
        let mm = Arc::new(
            ModelManager::new(config, Arc::new(MockEngineFactory::default()))
                .await
                .unwrap(),
        );
        mm.load_model("vlm-1").await.unwrap();
        {
            let mut guard = mm.write_models("test setup");
            let record = guard.get_mut("vlm-1").unwrap();
            record.max_prompt_tokens = 100_000;
            record.kv_cache_gb = 7.5;
        }

        mm.spawn_kv_wedge_recovery("vlm-1", 37_847); // well above the 10,000 floor
        wait_for_wedge_recovery_to_finish(&mm, "vlm-1").await;

        let ratchet = template::read_kv_capacity_ratchet(&dir.join("vlm-1"), 7.5);
        assert_eq!(
            ratchet,
            Some(35_954), // 37_847 * 0.95, truncated
            "a plausible observation is written with the 0.95 safety margin, \
             at the pool size (7.5GB) active when it was learned"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- R1: Vision (VLM) as a managed model ------------------------------
    //
    // `MockEngineFactory` classifies any model id containing "vlm" as a Vision
    // model, so a single manager can hold mixed kinds with no GPU.

    /// A VLM loads through the manager like any model and reports `Vision`.
    #[tokio::test]
    async fn test_load_vision_model_ready_and_reports_kind() {
        let mm = make_mm(&[("qwen-vlm-8b", 5.5)], 22.5).await;
        mm.load_model("qwen-vlm-8b").await.unwrap();

        let state = mm.read_models("test").get("qwen-vlm-8b").unwrap().state;
        assert_eq!(state, ModelState::Ready);

        let ctx = mm.get_chat_context("qwen-vlm-8b").expect("Ready VLM");
        assert_eq!(ctx.handle.kind(), ModelKind::Vision);
        assert!(matches!(ctx.handle, EngineHandleKind::Vision(_)));
    }

    /// `get_handle` (the text-only resolver for /tokenize, /v1/completions)
    /// rejects a Vision model with `WrongKind` rather than mis-routing it.
    #[tokio::test]
    async fn test_get_handle_rejects_vision_model() {
        let mm = make_mm(&[("qwen-vlm-8b", 5.5)], 22.5).await;
        mm.load_model("qwen-vlm-8b").await.unwrap();

        let err = mm.get_handle("qwen-vlm-8b").unwrap_err();
        assert!(
            matches!(err, ModelError::WrongKind(_)),
            "a VLM on a text-only endpoint must be WrongKind, got {err:?}"
        );
    }

    /// A Ready VLM appears in the metrics snapshot with the VLM concurrency cap.
    #[tokio::test]
    async fn test_metrics_snapshot_includes_vision_model() {
        let mm = make_mm(&[("qwen-vlm-8b", 5.5)], 22.5).await;
        mm.load_model("qwen-vlm-8b").await.unwrap();

        let snap = mm.metrics_snapshot();
        assert_eq!(snap.models.len(), 1);
        let m = &snap.models[0];
        assert_eq!(m.id, "qwen-vlm-8b");
        assert_eq!(m.active, 0, "fresh VLM has no in-flight requests");
        assert_eq!(
            m.max_seqs,
            crate::vlm_engine::VLM_CHANNEL_CAP,
            "VLM reports its command-channel cap as max concurrency"
        );
    }

    /// Evicting a Ready VLM frees its full VRAM reservation, same as an LLM.
    #[tokio::test]
    async fn test_evict_vision_model_frees_vram() {
        let mm = make_mm(&[("qwen-vlm-8b", 5.5)], 22.5).await;
        mm.load_model("qwen-vlm-8b").await.unwrap();
        assert!(
            mm.vram.read().unwrap().free_gb(&mm.inference_domain) < 0.01,
            "VRAM reserved"
        );

        mm.evict_model("qwen-vlm-8b").await.unwrap();

        let state = mm.read_models("test").get("qwen-vlm-8b").unwrap().state;
        assert_eq!(state, ModelState::NotLoaded);
        assert!(
            (mm.vram.read().unwrap().free_gb(&mm.inference_domain) - 22.5).abs() < 0.01,
            "VRAM fully restored after VLM eviction"
        );
    }

    /// LRU eviction works across kinds: loading a VLM evicts an LRU LLM when
    /// VRAM is tight, and vice versa — proving the lifecycle is kind-agnostic.
    #[tokio::test]
    async fn test_lru_evicts_text_to_load_vision_mixed_kinds() {
        // 10 GB total; an LLM (6 GB) and a VLM (6 GB) cannot co-reside.
        let mm = make_mm(&[("llm-a", 6.0), ("qwen-vlm-b", 6.0)], 10.0).await;

        mm.load_model("llm-a").await.unwrap();
        mm.load_model("qwen-vlm-b").await.unwrap();

        let (a_state, b_state, b_kind) = {
            let guard = mm.read_models("test");
            let a = guard.get("llm-a").unwrap();
            let b = guard.get("qwen-vlm-b").unwrap();
            (a.state, b.state, b.handle.as_ref().unwrap().kind())
        };
        assert_eq!(a_state, ModelState::NotLoaded, "LLM evicted to fit the VLM");
        assert_eq!(b_state, ModelState::Ready);
        assert_eq!(
            b_kind,
            ModelKind::Vision,
            "the survivor is the Vision engine"
        );
    }

    /// Multiple VLMs can be registered and loaded subject to VRAM, just like LLMs.
    #[tokio::test]
    async fn test_multiple_vlms_register_and_load() {
        // Cap the KV pool so each model reserves only weight+cap (5+2 GB),
        // leaving room for both — otherwise full-dynamic KV sizing has the first
        // load grab all VRAM and the second evicts it.
        let mut config = make_config(&[("vlm-one", 5.0), ("vlm-two", 5.0)], 22.5);
        config.cache_size_gb = 2.0;
        let mm = ModelManager::new(config, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();
        mm.load_model("vlm-one").await.unwrap();
        mm.load_model("vlm-two").await.unwrap();

        let ready = mm.list_ready_models();
        assert_eq!(ready.len(), 2, "both VLMs co-resident: {ready:?}");
        for id in ["vlm-one", "vlm-two"] {
            let ctx = mm.get_chat_context(id).expect("Ready VLM");
            assert_eq!(ctx.handle.kind(), ModelKind::Vision);
        }
    }

    // ---- T5.3: poison-tolerant `models` lock -----------------------------

    /// T5.3 (committee findings 39, 40): a poisoned `models` lock must NOT turn
    /// every request into a permanent panic loop. After a thread panics while
    /// holding the write guard the lock is poisoned; the manager's read/write
    /// helpers must recover the guard instead of re-panicking, so the node
    /// keeps serving. Pre-fix, `read_models`/`write_models` called `panic!`
    /// here and every subsequent request would have cascaded.
    #[tokio::test]
    async fn test_poisoned_models_lock_recovers_not_panics() {
        let mm = make_mm(&[("qwen3-8b", 5.5)], 22.5).await;

        // Poison the lock: panic while holding the write guard. The guard drops
        // during unwind → the lock is marked poisoned; catch_unwind stops the
        // unwind so the test thread itself survives.
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = mm.models.write().expect("lock not yet poisoned");
            panic!("intentional poison for T5.3 test");
        }));
        assert!(panicked.is_err(), "the closure must have panicked");
        assert!(
            mm.models.is_poisoned(),
            "models lock must be poisoned after the panic"
        );

        // Read path recovers (would have panicked pre-fix).
        let ready = mm.list_ready_models();
        assert!(
            ready.is_empty(),
            "recovered read returns the registered map"
        );
        assert_eq!(
            mm.get_handle("qwen3-8b").unwrap_err(),
            ModelError::NotLoaded,
            "node still serves requests after the lock poisoned"
        );

        // Write path recovers too: mutating through the recovered guard does
        // not panic, and the change is observable.
        mm.write_models("test").get_mut("qwen3-8b").unwrap().state = ModelState::Loading;
        assert_eq!(
            mm.get_handle("qwen3-8b").unwrap_err(),
            ModelError::Loading,
            "recovered write guard applied the state change"
        );
    }

    // ---- Phase C1: per-model device placement ----------------------------

    /// A three-device fleet (dGPU + iGPU + CPU). The dGPU keeps its own VRAM
    /// domain; the iGPU and CPU share the `system` domain.
    fn fleet_inventory() -> DeviceInventory {
        DeviceInventory::from_devices(vec![
            dev("GPU.1", DeviceKind::DiscreteGpu),
            dev("GPU.0", DeviceKind::IntegratedGpu),
            dev("CPU", DeviceKind::Cpu),
        ])
    }

    /// `resolve_device_domain` maps any device to its domain: a discrete GPU to
    /// its own name, an iGPU/CPU to the shared system domain, an unknown device
    /// to its own name (the discrete-fleet fallback).
    #[test]
    fn resolve_device_domain_maps_each_device_kind() {
        let inv = fleet_inventory();
        assert_eq!(resolve_device_domain("GPU.1", &inv), "GPU.1");
        assert_eq!(resolve_device_domain("GPU.0", &inv), DOMAIN_SYSTEM);
        assert_eq!(resolve_device_domain("CPU", &inv), DOMAIN_SYSTEM);
        assert_eq!(
            resolve_device_domain("NPU", &inv),
            "NPU",
            "a device absent from the inventory falls back to its own name"
        );
    }

    /// Build a manager over the three-device fleet with `tiny` routed to the
    /// iGPU (`GPU.0` → system domain) and everything else on the primary dGPU
    /// (`GPU.1`). Bounded KV (1 GB) leaves observable free space in each domain.
    async fn fleet_mm(models: &[(&str, f64)], gpu1_gb: f64, system_gb: f64) -> ModelManager {
        let mut cfg = make_config(models, gpu1_gb);
        cfg.device = "GPU.1".to_owned();
        cfg.default_kv_cache_gb = 1.0; // bounded so domains don't grab-all to full
        cfg.domain_budgets
            .insert(DOMAIN_SYSTEM.to_owned(), system_gb);
        cfg.models.entry("tiny".to_owned()).or_default().policy = config::ModelPolicy {
            device: Some("GPU.0".to_owned()),
            ..Default::default()
        };
        ModelManager::new_with_inventory(
            cfg,
            Arc::new(MockEngineFactory::default()),
            Arc::new(fleet_inventory()),
        )
        .await
        .expect("fleet manager build")
    }

    /// A model with an explicit `device` override is charged to *its own* memory
    /// domain: loading the iGPU model draws down the system pool and leaves the
    /// dGPU pool untouched, and its metrics carry the actual `GPU.0` device.
    #[tokio::test]
    async fn per_model_device_charges_its_own_domain() {
        let mm = fleet_mm(&[("big", 10.0), ("tiny", 1.0)], 16.0, 8.0).await;

        mm.load_model("big").await.unwrap();
        let gpu1_free_before_tiny = mm.vram.read().unwrap().free_gb("GPU.1");

        mm.load_model("tiny").await.unwrap();
        let v = mm.vram.read().unwrap();
        // big: weight 10 + KV 1 = 11 on GPU.1 (budget 16) → 5 free.
        assert!(
            (v.free_gb("GPU.1") - 5.0).abs() < 1e-9,
            "GPU.1 reflects only the dGPU model (weight+KV = 11 of 16)"
        );
        // Loading the iGPU model did not touch the dGPU domain.
        assert!(
            (v.free_gb("GPU.1") - gpu1_free_before_tiny).abs() < 1e-9,
            "the iGPU load must not draw down the dGPU pool"
        );
        // tiny: weight 1 + KV 1 = 2 on the system domain (budget 8) → 6 free.
        assert!(
            (v.free_gb(DOMAIN_SYSTEM) - 6.0).abs() < 1e-9,
            "the system domain reflects only the iGPU model (weight+KV = 2 of 8)"
        );
        drop(v);

        // Metrics carry the per-model device, not the global config.device.
        let snap = mm.metrics_snapshot();
        let by_id = |id: &str| snap.models.iter().find(|m| m.id == id).cloned().unwrap();
        assert_eq!(by_id("big").device, "GPU.1");
        assert_eq!(
            by_id("tiny").device,
            "GPU.0",
            "the iGPU model is labelled with its actual device, not config.device"
        );
    }

    /// Eviction is domain-scoped: a dGPU load that needs room evicts a dGPU
    /// resident and never the iGPU model in the other domain.
    #[tokio::test]
    async fn eviction_is_domain_scoped() {
        // GPU.1 budget 12: one 10 GB model + 1 GB KV (=11) fits; a second does not.
        let mm = fleet_mm(&[("a", 10.0), ("b", 10.0), ("tiny", 1.0)], 12.0, 8.0).await;

        mm.load_model("tiny").await.unwrap();
        mm.load_model("a").await.unwrap();
        // Loading b onto GPU.1 must evict a (same domain), sparing tiny (system).
        mm.load_model("b").await.unwrap();

        let guard = mm.read_models("test");
        assert_eq!(
            guard.get("tiny").unwrap().state,
            ModelState::Ready,
            "the iGPU model in another domain is never evicted for a dGPU load"
        );
        assert_eq!(
            guard.get("a").unwrap().state,
            ModelState::NotLoaded,
            "the dGPU resident was the eviction victim"
        );
        assert_eq!(guard.get("b").unwrap().state, ModelState::Ready);
    }

    /// An explicit `device` override naming a device the running `OpenVINO` did
    /// not enumerate is rejected at startup with a clear error — no silent
    /// fallback.
    #[tokio::test]
    async fn unknown_device_override_rejected_at_startup() {
        let mut cfg = make_config(&[("m", 1.0)], 16.0);
        cfg.device = "GPU.1".to_owned();
        cfg.models.entry("m".to_owned()).or_default().policy = config::ModelPolicy {
            device: Some("GPU.9".to_owned()),
            ..Default::default()
        };
        let result = ModelManager::new_with_inventory(
            cfg,
            Arc::new(MockEngineFactory::default()),
            Arc::new(fleet_inventory()),
        )
        .await;
        // `ModelManager` is not `Debug`, so match rather than `expect_err`.
        let msg = match result {
            Ok(_) => panic!("an unenumerated device override must fail startup"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("GPU.9") && msg.contains("not an available"),
            "error names the bad device and the reason: {msg}"
        );
    }

    // ---- Phase C2: auto tier-preference placement engine ------------------

    /// A device with an explicit capability tier (the plain `dev` helper tiers
    /// everything `Fallback`, which can't exercise the placement engine).
    fn tiered_dev(name: &str, kind: DeviceKind, tier: DeviceTier) -> DeviceInfo {
        let mut d = dev(name, kind);
        d.tier = tier;
        d
    }

    /// A realistically-tiered Arc box: dGPU=heavy, iGPU=weak, CPU=fallback.
    fn tiered_fleet() -> DeviceInventory {
        DeviceInventory::from_devices(vec![
            tiered_dev("GPU.1", DeviceKind::DiscreteGpu, DeviceTier::Heavy),
            tiered_dev("GPU.0", DeviceKind::IntegratedGpu, DeviceTier::WeakIgpu),
            tiered_dev("CPU", DeviceKind::Cpu, DeviceTier::Fallback),
        ])
    }

    /// Build a manager over the tiered fleet (no preload, mock factory) so the
    /// auto engine resolves each model's device at registration.
    async fn tiered_mm(cfg: Config) -> ModelManager {
        ModelManager::new_with_inventory(
            cfg,
            Arc::new(MockEngineFactory::default()),
            Arc::new(tiered_fleet()),
        )
        .await
        .expect("tiered manager build")
    }

    /// A heavy LLM (no policy, no kind hint → treated as a heavy LLM) auto-places
    /// on the dGPU; a thin LLM (≤ `light_model_max_gb`) offloads to the weak iGPU.
    #[tokio::test]
    async fn auto_places_heavy_on_dgpu_thin_on_igpu() {
        let mut cfg = make_config(&[("qwen3-8b", 8.0), ("qwen3-0.6b", 1.0)], 16.0);
        cfg.device = "GPU.1".to_owned();
        let mm = tiered_mm(cfg).await;
        assert_eq!(mm.record_device("qwen3-8b"), "GPU.1");
        assert_eq!(
            mm.record_device("qwen3-0.6b"),
            "GPU.0",
            "a sub-threshold LLM offloads to the weak iGPU off the dGPU"
        );
    }

    /// An embedding model (kind-hinted) with no explicit `embedding_device`
    /// auto-routes to the weak iGPU via the engine's embedding default.
    #[tokio::test]
    async fn auto_places_embedding_on_weak_igpu() {
        let mut cfg = make_config(&[("e5", 0.5)], 16.0);
        cfg.device = "GPU.1".to_owned();
        cfg.models.entry("e5".to_owned()).or_default().kind = Some("embedding".to_owned());
        let mm = tiered_mm(cfg).await;
        assert_eq!(mm.record_device("e5"), "GPU.0");
    }

    /// An explicit `config.embedding_device` still wins over the engine default
    /// for an embedding model (operator override, precedence step 4).
    #[tokio::test]
    async fn explicit_embedding_device_overrides_engine() {
        let mut cfg = make_config(&[("e5", 0.5)], 16.0);
        cfg.device = "GPU.1".to_owned();
        cfg.embedding_device = Some("CPU".to_owned());
        cfg.models.entry("e5".to_owned()).or_default().kind = Some("embedding".to_owned());
        let mm = tiered_mm(cfg).await;
        assert_eq!(mm.record_device("e5"), "CPU");
    }

    /// A per-model `tier_preference` override is honoured by the engine and wins
    /// over the kind's built-in default (here forcing a heavy LLM onto the CPU).
    #[tokio::test]
    async fn tier_preference_override_routes_via_engine() {
        let mut cfg = make_config(&[("big", 8.0)], 16.0);
        cfg.device = "GPU.1".to_owned();
        cfg.models.entry("big".to_owned()).or_default().policy = config::ModelPolicy {
            tier_preference: vec!["fallback".to_owned()],
            ..Default::default()
        };
        let mm = tiered_mm(cfg).await;
        assert_eq!(mm.record_device("big"), "CPU");
    }

    /// An explicit per-model `device` still outranks `tier_preference` (C1 wins).
    #[tokio::test]
    async fn explicit_device_outranks_tier_preference() {
        let mut cfg = make_config(&[("big", 8.0)], 16.0);
        cfg.device = "GPU.1".to_owned();
        cfg.models.entry("big".to_owned()).or_default().policy = config::ModelPolicy {
            device: Some("GPU.0".to_owned()),
            tier_preference: vec!["fallback".to_owned()],
            ..Default::default()
        };
        let mm = tiered_mm(cfg).await;
        assert_eq!(mm.record_device("big"), "GPU.0");
    }

    /// An unrecognised tier label in `tier_preference` is rejected at startup.
    #[tokio::test]
    async fn unknown_tier_label_rejected_at_startup() {
        let mut cfg = make_config(&[("m", 1.0)], 16.0);
        cfg.device = "GPU.1".to_owned();
        cfg.models.entry("m".to_owned()).or_default().policy = config::ModelPolicy {
            tier_preference: vec!["super-gpu".to_owned()],
            ..Default::default()
        };
        let result = ModelManager::new_with_inventory(
            cfg,
            Arc::new(MockEngineFactory::default()),
            Arc::new(tiered_fleet()),
        )
        .await;
        let msg = match result {
            Ok(_) => panic!("an unknown tier label must fail startup"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("super-gpu") && msg.contains("unknown tier"),
            "error names the bad label: {msg}"
        );
    }

    /// A `tier_preference` no device on this box satisfies is rejected at startup.
    #[tokio::test]
    async fn unsatisfiable_tier_preference_rejected_at_startup() {
        // The tiered fleet has no NPU.
        let mut cfg = make_config(&[("m", 1.0)], 16.0);
        cfg.device = "GPU.1".to_owned();
        cfg.models.entry("m".to_owned()).or_default().policy = config::ModelPolicy {
            tier_preference: vec!["npu".to_owned()],
            ..Default::default()
        };
        let result = ModelManager::new_with_inventory(
            cfg,
            Arc::new(MockEngineFactory::default()),
            Arc::new(tiered_fleet()),
        )
        .await;
        let msg = match result {
            Ok(_) => panic!("an unsatisfiable tier preference must fail startup"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("no available OpenVINO device"),
            "error explains nothing satisfies the preference: {msg}"
        );
    }

    /// T5/F4: two **immovable** (pinned) models that both prefer the dGPU but
    /// can't both fit there — the second falls through to the next satisfiable
    /// tier whose domain has room, instead of resolving permanently onto the
    /// full dGPU domain. Placement order is immovable-first then by id, so the
    /// lexicographically-first model claims the dGPU.
    #[tokio::test]
    async fn fit_coupling_routes_second_immovable_model_off_a_full_domain() {
        // dGPU domain (GPU.1) budget = total_vram_gb = 16; the system domain
        // (GPU.0 + CPU) gets an explicit 16 GB so the fallthrough target fits.
        let mut cfg = make_config(&[("aaa", 10.0), ("bbb", 10.0)], 16.0);
        cfg.device = "GPU.1".to_owned();
        cfg.domain_budgets.insert(DOMAIN_SYSTEM.to_owned(), 16.0);
        let pinned_heavy = || config::ModelPolicy {
            tier_preference: vec!["heavy".to_owned(), "weak-igpu".to_owned()],
            pinned: true,
            ..Default::default()
        };
        cfg.models.entry("aaa".to_owned()).or_default().policy = pinned_heavy();
        cfg.models.entry("bbb".to_owned()).or_default().policy = pinned_heavy();
        let mm = tiered_mm(cfg).await;
        assert_eq!(
            mm.record_device("aaa"),
            "GPU.1",
            "first pinned model takes the dGPU"
        );
        assert_eq!(
            mm.record_device("bbb"),
            "GPU.0",
            "second pinned model can't fit the dGPU → falls through to the weak iGPU"
        );
    }

    /// T5/F4 contrast: the same over-subscription with **evictable** (default)
    /// models does NOT fall through — evictable models hold no permanent
    /// reservation (eviction reclaims them at load), so both stack on the
    /// preferred dGPU and the runtime eviction policy sorts them out.
    #[tokio::test]
    async fn fit_coupling_does_not_reserve_for_evictable_models() {
        let mut cfg = make_config(&[("aaa", 10.0), ("bbb", 10.0)], 16.0);
        cfg.device = "GPU.1".to_owned();
        cfg.domain_budgets.insert(DOMAIN_SYSTEM.to_owned(), 16.0);
        let evictable_heavy = || config::ModelPolicy {
            tier_preference: vec!["heavy".to_owned(), "weak-igpu".to_owned()],
            // pinned: false, evictable: true (defaults)
            ..Default::default()
        };
        cfg.models.entry("aaa".to_owned()).or_default().policy = evictable_heavy();
        cfg.models.entry("bbb".to_owned()).or_default().policy = evictable_heavy();
        let mm = tiered_mm(cfg).await;
        assert_eq!(mm.record_device("aaa"), "GPU.1");
        assert_eq!(
            mm.record_device("bbb"),
            "GPU.1",
            "evictable models do not reserve → both stay on the preferred dGPU"
        );
    }

    // ---- config reload / audit / deregister -------------------------------

    /// A mock factory whose `model_exists` reports `false` for exactly one
    /// `model_id` (everything else delegates to a real `MockEngineFactory`) —
    /// the seam the "declared in the file, but no files on disk" branch of
    /// `reload_config`/`audit_config` needs, without a real filesystem.
    struct MissingFilesFactory {
        inner: MockEngineFactory,
        missing_id: String,
    }

    impl crate::model_manager::lifecycle::EngineFactory for MissingFilesFactory {
        fn detect_kind(&self, model_dir: &std::path::Path) -> ModelKind {
            self.inner.detect_kind(model_dir)
        }
        fn model_exists(&self, model_dir: &std::path::Path) -> bool {
            model_dir.file_name().and_then(|s| s.to_str()) != Some(self.missing_id.as_str())
        }
        fn load(
            &self,
            model_path: &std::path::Path,
            device: &str,
            cache_size_gb: f64,
            max_num_seqs: usize,
        ) -> anyhow::Result<(EngineHandleKind, std::thread::JoinHandle<()>)> {
            self.inner
                .load(model_path, device, cache_size_gb, max_num_seqs)
        }
    }

    /// Build a `ModelManager` whose config is a *real* file at `config_path`,
    /// parsed with the same `Config::load` validation production uses — the
    /// fixture `reload_config`/`audit_config`/`deregister_model` need, since
    /// all three read or write that file directly rather than `self.config`.
    /// Returns the manager plus the still-alive `TempDir` (keep it in scope
    /// for the test's duration — its `Drop` deletes the directory).
    async fn make_mm_from_file(
        json: &str,
        factory: Arc<dyn crate::model_manager::lifecycle::EngineFactory>,
    ) -> (ModelManager, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.json");
        std::fs::write(&path, json).expect("write temp config");
        let config = Config::load(&path).expect("parse temp config");
        let mm = ModelManager::new(config, factory)
            .await
            .expect("ModelManager::new failed in test")
            .with_config_path(path);
        (mm, dir)
    }

    const BASE_CONFIG: &str = r#"{
        "models_dir": "/tmp/test-models",
        "device": "CPU",
        "total_vram_gb": 0.0,
        "models": { "existing": { "vram_gb": 1.0 } }
    }"#;

    /// `reload_config` registers a `model_id` newly added to the file since
    /// startup, as `NotLoaded`, without loading it — and leaves the
    /// already-registered entry alone (nothing about it changed).
    #[tokio::test]
    async fn reload_config_adds_new_entry() {
        let (mm, dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
                "models_dir": "/tmp/test-models",
                "device": "CPU",
                "total_vram_gb": 0.0,
                "models": {
                    "existing": { "vram_gb": 1.0 },
                    "new-model": { "vram_gb": 2.0 }
                }
            }"#,
        )
        .expect("rewrite temp config");

        let report = mm.reload_config().expect("reload_config");
        assert_eq!(report.added, vec!["new-model".to_owned()]);
        assert!(report.updated.is_empty());
        assert!(report.skipped_missing_files.is_empty());
        assert_eq!(report.unchanged, 1, "the untouched 'existing' entry");

        let ids: Vec<String> = mm.list_models().into_iter().map(|m| m.id).collect();
        assert!(ids.contains(&"new-model".to_owned()));
        let new_record = mm
            .list_models()
            .into_iter()
            .find(|m| m.id == "new-model")
            .expect("new-model registered");
        assert_eq!(new_record.state, ModelState::NotLoaded);
    }

    /// A `NotLoaded` entry whose `vram_gb` changed in the file is refreshed
    /// in place and reported as `updated`.
    #[tokio::test]
    async fn reload_config_updates_changed_not_loaded_entry() {
        let (mm, dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
                "models_dir": "/tmp/test-models",
                "device": "CPU",
                "total_vram_gb": 0.0,
                "models": { "existing": { "vram_gb": 9.0 } }
            }"#,
        )
        .expect("rewrite temp config");

        let report = mm.reload_config().expect("reload_config");
        assert_eq!(report.updated, vec!["existing".to_owned()]);
        assert!(report.added.is_empty());
        assert_eq!(report.unchanged, 0);

        let record = mm
            .list_models()
            .into_iter()
            .find(|m| m.id == "existing")
            .expect("still registered");
        assert!((record.vram_gb - 9.0).abs() < f64::EPSILON);
    }

    /// A `Ready` model is never touched by reload, even though its config
    /// entry changed on disk — reload's whole point is zero blast radius for
    /// anything live.
    #[tokio::test]
    async fn reload_config_never_touches_a_ready_model() {
        let (mm, dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;
        mm.load_model("existing").await.expect("load existing");

        std::fs::write(
            dir.path().join("config.json"),
            r#"{
                "models_dir": "/tmp/test-models",
                "device": "CPU",
                "total_vram_gb": 0.0,
                "models": { "existing": { "vram_gb": 9.0 } }
            }"#,
        )
        .expect("rewrite temp config");

        let report = mm.reload_config().expect("reload_config");
        assert_eq!(report.left_untouched, vec!["existing".to_owned()]);
        assert!(report.updated.is_empty());

        let record = mm
            .list_models()
            .into_iter()
            .find(|m| m.id == "existing")
            .expect("still registered");
        assert_eq!(record.state, ModelState::Ready, "still Ready, untouched");
        assert!(
            (record.vram_gb - 1.0).abs() < f64::EPSILON,
            "vram_gb kept its original loaded value, did not pick up the file's 9.0"
        );
    }

    /// A new entry whose directory doesn't exist is skipped and reported,
    /// not silently registered as an unloadable ghost.
    #[tokio::test]
    async fn reload_config_skips_entry_with_missing_files() {
        let factory = Arc::new(MissingFilesFactory {
            inner: MockEngineFactory::default(),
            missing_id: "ghost-model".to_owned(),
        });
        let (mm, dir) = make_mm_from_file(BASE_CONFIG, factory).await;
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
                "models_dir": "/tmp/test-models",
                "device": "CPU",
                "total_vram_gb": 0.0,
                "models": {
                    "existing": { "vram_gb": 1.0 },
                    "ghost-model": { "vram_gb": 2.0 }
                }
            }"#,
        )
        .expect("rewrite temp config");

        let report = mm.reload_config().expect("reload_config");
        assert_eq!(report.skipped_missing_files, vec!["ghost-model".to_owned()]);
        assert!(report.added.is_empty());

        let ids: Vec<String> = mm.list_models().into_iter().map(|m| m.id).collect();
        assert!(
            !ids.contains(&"ghost-model".to_owned()),
            "a model with no files must never be registered"
        );
    }

    /// `reload_config` refuses to run at all when `config.json` still has
    /// deprecated inline `api_keys`/`admin_api_keys` — the reload-tripwire
    /// half of the migration check (`check_no_inline_keys` in `startup.rs`
    /// is the boot-time half).
    #[tokio::test]
    async fn reload_config_rejects_deprecated_inline_keys() {
        let (mm, dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
                "models_dir": "/tmp/test-models",
                "device": "CPU",
                "total_vram_gb": 0.0,
                "models": { "existing": { "vram_gb": 1.0 } },
                "api_keys": ["sk-forgot-to-migrate"]
            }"#,
        )
        .expect("rewrite temp config");

        let err = mm.reload_config().unwrap_err().to_string();
        assert!(err.contains("deprecated"), "{err}");
    }

    // ---- reload_keys_file --------------------------------------------------

    /// Build a `ModelManager` wired with both a config file and a keys file
    /// (the split this session's redesign introduced) — mirrors
    /// `make_mm_from_file` but also seeds `auth`/`keys_file_path`.
    async fn make_mm_with_keys(
        config_json: &str,
        initial_keys_json: &str,
        factory: Arc<dyn crate::model_manager::lifecycle::EngineFactory>,
    ) -> (ModelManager, tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let config_path = dir.path().join("config.json");
        std::fs::write(&config_path, config_json).expect("write temp config");
        let keys_path = dir.path().join("test.keys.json");
        std::fs::write(&keys_path, initial_keys_json).expect("write temp keys");
        let initial_auth =
            crate::AuthConfig::load_from_file(&keys_path).expect("parse initial keys");
        let auth = Arc::new(arc_swap::ArcSwap::new(Arc::new(initial_auth)));
        let config = Config::load(&config_path).expect("parse temp config");
        let mm = ModelManager::new(config, factory)
            .await
            .expect("ModelManager::new failed in test")
            .with_config_path(config_path)
            .with_auth(auth)
            .with_keys_file_path(keys_path.clone());
        (mm, dir, keys_path)
    }

    /// Happy path: a changed keys file rotates the live keys and the report
    /// reflects exactly what changed.
    #[tokio::test]
    async fn reload_keys_file_rotates_and_reports() {
        let (mm, _dir, keys_path) = make_mm_with_keys(
            BASE_CONFIG,
            r#"{"api_keys": ["sk-old"], "admin_api_keys": []}"#,
            Arc::new(MockEngineFactory::default()),
        )
        .await;
        std::fs::write(
            &keys_path,
            r#"{"api_keys": ["sk-new"], "admin_api_keys": []}"#,
        )
        .expect("rewrite keys file");

        let report = mm.reload_keys_file().expect("reload_keys_file");
        assert!(report.keys_updated);
        assert!(!report.admin_keys_updated);
        assert!(!report.admin_downgrade_refused);
        assert_eq!(report.api_key_count, 1);
        assert_eq!(report.admin_key_count, 0);

        let (live_api, live_admin) = mm.auth_snapshot_for_test().expect("auth wired");
        assert_eq!(live_api, vec!["sk-new".to_string()]);
        assert!(live_admin.is_empty());
    }

    /// A malformed keys file aborts the reload with the live keys completely
    /// untouched — all-or-nothing, no partial apply.
    #[tokio::test]
    async fn reload_keys_file_malformed_mutates_nothing() {
        let (mm, _dir, keys_path) = make_mm_with_keys(
            BASE_CONFIG,
            r#"{"api_keys": ["sk-original"], "admin_api_keys": []}"#,
            Arc::new(MockEngineFactory::default()),
        )
        .await;
        std::fs::write(&keys_path, r#"{"api_keys": "not-an-array"}"#).expect("write broken keys");

        assert!(mm.reload_keys_file().is_err());

        let (live_api, live_admin) = mm.auth_snapshot_for_test().expect("auth wired");
        assert_eq!(
            live_api,
            vec!["sk-original".to_string()],
            "a failed reload must not mutate live keys"
        );
        assert!(live_admin.is_empty());
    }

    /// A missing keys file at reload time is an error, not "boot open" —
    /// reload's posture is deliberately stricter than boot's (see
    /// `reload_keys_file`'s doc comment).
    #[tokio::test]
    async fn reload_keys_file_missing_file_errors() {
        let (mm, _dir, keys_path) = make_mm_with_keys(
            BASE_CONFIG,
            r#"{"api_keys": ["sk-original"], "admin_api_keys": []}"#,
            Arc::new(MockEngineFactory::default()),
        )
        .await;
        std::fs::remove_file(&keys_path).expect("delete keys file");

        assert!(mm.reload_keys_file().is_err());
        let (live_api, _) = mm.auth_snapshot_for_test().expect("auth wired");
        assert_eq!(live_api, vec!["sk-original".to_string()]);
    }

    /// The privilege-downgrade guard: the file's new `admin_api_keys` going
    /// empty while a non-empty admin scope is live is refused, not applied —
    /// `api_keys` still rotates independently in the same reload.
    #[tokio::test]
    async fn reload_keys_file_refuses_admin_downgrade() {
        let (mm, _dir, keys_path) = make_mm_with_keys(
            BASE_CONFIG,
            r#"{"api_keys": ["sk-old"], "admin_api_keys": ["sk-admin"]}"#,
            Arc::new(MockEngineFactory::default()),
        )
        .await;
        std::fs::write(
            &keys_path,
            r#"{"api_keys": ["sk-new"], "admin_api_keys": []}"#,
        )
        .expect("rewrite keys file");

        let report = mm.reload_keys_file().expect("reload_keys_file");
        assert!(report.keys_updated, "api_keys still rotates");
        assert!(!report.admin_keys_updated, "admin downgrade must not apply");
        assert!(report.admin_downgrade_refused);
        assert_eq!(report.admin_key_count, 1, "old admin key count preserved");

        let (live_api, live_admin) = mm.auth_snapshot_for_test().expect("auth wired");
        assert_eq!(live_api, vec!["sk-new".to_string()], "api_keys did rotate");
        assert_eq!(
            live_admin,
            vec!["sk-admin".to_string()],
            "admin_api_keys must stay exactly as it was"
        );
    }

    /// `reload_keys_file` with no `auth`/`keys_file_path` wired (test builds
    /// that never called the builders) errors rather than silently doing
    /// nothing — see the field doc comments on `auth`/`keys_file_path` for
    /// why this differs from `voice_pin`'s "absent is fine" convention.
    #[tokio::test]
    async fn reload_keys_file_errors_when_not_wired() {
        let (mm, _dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;
        assert!(mm.reload_keys_file().is_err());
    }

    /// `.store()` on the shared `ArcSwap<AuthConfig>` is atomic for the
    /// *whole* struct — a concurrent reader's `.load()` always sees either
    /// fully-old or fully-new `{api_keys, admin_api_keys}`, never a mix of
    /// one field from each. This is the property `require_bearer_auth`'s
    /// single per-request snapshot relies on (see its doc comment) — a
    /// single-`ArcSwap` design (vs. two independently-swappable ones, one
    /// per key scope) is what makes it true by construction rather than by
    /// convention. Not a concurrency stress test (that would be flaky by
    /// nature); a direct check that a swap is all-or-nothing.
    #[test]
    fn auth_config_swap_is_never_torn() {
        let auth = arc_swap::ArcSwap::new(Arc::new(crate::AuthConfig {
            api_keys: vec!["old-api".to_string()],
            admin_api_keys: vec!["old-admin".to_string()],
        }));
        auth.store(Arc::new(crate::AuthConfig {
            api_keys: vec!["new-api".to_string()],
            admin_api_keys: vec!["new-admin".to_string()],
        }));
        let snapshot = auth.load();
        // Both fields must come from the SAME store() call — never old_api
        // paired with new_admin or vice versa.
        assert_eq!(snapshot.api_keys, vec!["new-api".to_string()]);
        assert_eq!(snapshot.admin_api_keys, vec!["new-admin".to_string()]);
    }

    /// `audit_config` flags a declared entry with no files on disk, and
    /// reports whether it's also in `preload` (fatal-at-restart severity).
    ///
    /// `ghost-model` is added to the file *after* construction, not in the
    /// startup config — putting a missing-files model in `preload` at
    /// startup is a hard construction error (T5.5), which is correct
    /// behaviour but not what this test is exercising.
    #[tokio::test]
    async fn audit_config_reports_missing_files_and_preload_membership() {
        let factory = Arc::new(MissingFilesFactory {
            inner: MockEngineFactory::default(),
            missing_id: "ghost-model".to_owned(),
        });
        let (mm, dir) = make_mm_from_file(BASE_CONFIG, factory).await;
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
                "models_dir": "/tmp/test-models",
                "device": "CPU",
                "total_vram_gb": 0.0,
                "preload": ["ghost-model"],
                "models": {
                    "existing": { "vram_gb": 1.0 },
                    "ghost-model": { "vram_gb": 2.0 }
                }
            }"#,
        )
        .expect("rewrite temp config");

        let report = mm.audit_config().expect("audit_config");
        assert_eq!(report.total_declared, 2);
        assert_eq!(report.problems.len(), 1);
        let problem = &report.problems[0];
        assert_eq!(problem.model_id, "ghost-model");
        assert!(problem.files_missing);
        assert!(problem.in_preload);
        assert!(
            !problem.pending_reload,
            "missing_files wins — not also double-reported as pending"
        );
    }

    /// `audit_config` flags an entry that exists on disk but was added to
    /// the file after the process started — `pending_reload`, resolved by
    /// calling `reload_config`.
    #[tokio::test]
    async fn audit_config_reports_pending_reload() {
        let (mm, dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
                "models_dir": "/tmp/test-models",
                "device": "CPU",
                "total_vram_gb": 0.0,
                "models": {
                    "existing": { "vram_gb": 1.0 },
                    "new-model": { "vram_gb": 2.0 }
                }
            }"#,
        )
        .expect("rewrite temp config");

        let report = mm.audit_config().expect("audit_config");
        assert_eq!(report.problems.len(), 1);
        assert_eq!(report.problems[0].model_id, "new-model");
        assert!(!report.problems[0].files_missing);
        assert!(report.problems[0].pending_reload);

        // Reloading resolves it — a second audit reports nothing left.
        mm.reload_config().expect("reload_config");
        let report = mm.audit_config().expect("audit_config");
        assert!(report.problems.is_empty());
    }

    /// Deregistering a `NotLoaded` model removes it from the live registry
    /// and persists the removal to `config.json`.
    #[tokio::test]
    async fn deregister_model_removes_not_loaded_entry_and_persists() {
        let (mm, dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;

        mm.deregister_model("existing").await.expect("deregister");

        let ids: Vec<String> = mm.list_models().into_iter().map(|m| m.id).collect();
        assert!(!ids.contains(&"existing".to_owned()));

        let on_disk = std::fs::read_to_string(dir.path().join("config.json")).unwrap();
        let value: serde_json::Value = serde_json::from_str(&on_disk).unwrap();
        assert!(
            value["models"].get("existing").is_none(),
            "removed entry must not survive in the persisted file"
        );
    }

    /// Deregistering a `Ready` model evicts it first (freeing VRAM) before
    /// removing it — no orphaned VRAM reservation left behind.
    #[tokio::test]
    async fn deregister_model_evicts_a_ready_model_first() {
        let (mm, _dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;
        mm.load_model("existing").await.expect("load existing");
        assert!(mm.metrics_snapshot().used_vram_gb > 0.0);

        mm.deregister_model("existing").await.expect("deregister");

        assert!(
            mm.metrics_snapshot().used_vram_gb.abs() < f64::EPSILON,
            "eviction during deregister must release its VRAM reservation"
        );
        let ids: Vec<String> = mm.list_models().into_iter().map(|m| m.id).collect();
        assert!(!ids.contains(&"existing".to_owned()));
    }

    /// Deregistering an unknown `model_id` is a clean `NotFound`, matching
    /// every other model-scoped call.
    #[tokio::test]
    async fn deregister_model_unknown_id_is_not_found() {
        let (mm, _dir) =
            make_mm_from_file(BASE_CONFIG, Arc::new(MockEngineFactory::default())).await;
        let err = mm.deregister_model("ghost").await.unwrap_err();
        assert_eq!(
            err.downcast_ref::<ModelError>(),
            Some(&ModelError::NotFound("ghost".to_owned()))
        );
    }

    /// The motivating case: a `model_id` declared in `config.json` but whose
    /// directory was missing, so it was skipped at startup registration and
    /// never made it into `self.models` at all — `deregister_model` must
    /// still be able to purge it from the file, not 404 just because there
    /// was never anything live to evict.
    #[tokio::test]
    async fn deregister_model_purges_a_never_registered_ghost_entry() {
        let factory = Arc::new(MissingFilesFactory {
            inner: MockEngineFactory::default(),
            missing_id: "ghost-model".to_owned(),
        });
        let (mm, dir) = make_mm_from_file(
            r#"{
                "models_dir": "/tmp/test-models",
                "device": "CPU",
                "total_vram_gb": 0.0,
                "models": {
                    "existing": { "vram_gb": 1.0 },
                    "ghost-model": { "vram_gb": 2.0 }
                }
            }"#,
            factory,
        )
        .await;
        // Confirm the premise: ghost-model was skipped, never registered.
        let ids: Vec<String> = mm.list_models().into_iter().map(|m| m.id).collect();
        assert!(!ids.contains(&"ghost-model".to_owned()));

        mm.deregister_model("ghost-model")
            .await
            .expect("deregister a never-registered but config-declared entry");

        let on_disk = std::fs::read_to_string(dir.path().join("config.json")).unwrap();
        let value: serde_json::Value = serde_json::from_str(&on_disk).unwrap();
        assert!(
            value["models"].get("ghost-model").is_none(),
            "ghost entry must be gone from the persisted file"
        );
        assert!(
            value["models"].get("existing").is_some(),
            "unrelated entry must survive"
        );
    }
}
