// ============================================================
// src/cb_engine.rs — the continuous-batching engine thread
// ============================================================
// The Phase-2 concurrency core. ONE dedicated OS thread owns the
// `OvCbEngine` (ContinuousBatchingPipeline is single-threaded by
// contract) and a route table mapping each live request to its SSE
// token channel. Async chat handlers never touch the engine directly —
// they speak to it only through a `tokio::sync::mpsc` command channel.
//
//   handler task ──AddRequest{id, prompt, tx}──▶ ┌─────────────────┐
//                                                │  cb-engine       │
//                                                │  thread:         │
//                                                │  OvCbEngine +    │
//                                                │  HashMap<id,tx>  │
//                                                └───────┬─────────┘
//                                                        │ per-token
//                                                        ▼
//                                          request's TokenSender → SSE
//
// There is no DropRequest command: a client disconnect is detected lazily on
// the next step (its closed TokenSender), which cancels the request and frees
// its slot — cancellation is eventual (one step), not immediate.
//
// CRASH COURSE — why a std::thread, not a tokio task:
//   engine.step() blocks the OS thread for a whole model iteration
//   (tens of ms). Running it on a tokio async worker would stall every
//   other task on that worker. A dedicated std::thread is isolated from
//   the async runtime entirely. The command channel is the only bridge.
//
// CRASH COURSE — why a command channel, not Arc<Mutex<engine>>:
//   A Mutex would let any task lock + step the engine, but that
//   serializes generation under the lock and throws away the whole point
//   of continuous batching (many requests advancing together in ONE
//   step). Funnelling every request through one owner thread keeps the
//   batch shared and needs no lock and no `unsafe` sharing.
// ============================================================

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio::sync::mpsc::error::{TryRecvError, TrySendError};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

use crate::metrics::{HotMetrics, Modality};
use crate::model_manager::POOL_EXHAUSTED_MARKER;
use crate::model_manager::engine::{ManagedEngine, ModelKind};
use crate::ov_cb::{FinishReason, GenParams, OvCbEngine};
use crate::streaming::{StreamEvent, TokenSender};

/// Capacity of the engine command channel.
///
/// Commands are tiny and the engine drains them between model steps, so
/// this only needs to absorb a burst of simultaneous request submissions.
const COMMAND_CAPACITY: usize = 128;

/// `OpenVINO`'s own `SchedulerConfig::max_num_seqs` default, used when a
/// caller passes `0` — the documented "let `OpenVINO` decide" sentinel
/// (`CONFIG.md` `max_num_seqs`; `ov_bridge.cpp` only overrides the C++
/// scheduler's own default when `max_num_seqs > 0`). [`spawn_engine`]
/// mirrors this same value for the Rust-side admission [`Semaphore`] so a
/// model configured with `0` gets *this* many concurrent slots, not a
/// 0-permit semaphore that 429s every request forever (the project's internal engineering log
/// 2026-08-01).
const OV_DEFAULT_MAX_NUM_SEQS: usize = 256;

/// Normalizes the `max_num_seqs` value used for the Rust-side admission
/// `Semaphore`: `0` (the "let `OpenVINO` decide" sentinel) becomes
/// [`OV_DEFAULT_MAX_NUM_SEQS`]; any other value passes through unchanged.
/// Extracted from [`spawn_engine`] so this normalization is directly
/// unit-testable without a real `OvCbEngine`/GPU.
fn resolve_admission_cap(max_seqs: usize) -> usize {
    if max_seqs == 0 {
        OV_DEFAULT_MAX_NUM_SEQS
    } else {
        max_seqs
    }
}

/// Upper bound on the pipeline-settle steps run at engine-loop exit.
///
/// Cancelled requests release their KV blocks on a subsequent `step`; this
/// caps that settle loop so a pipeline that never reports idle cannot hang
/// the engine thread (and the eviction join) forever.
const MAX_SETTLE_STEPS: usize = 16;

/// Smoothing factor for the tokens/sec exponential moving average.
///
/// Each model step contributes one throughput sample (tokens produced /
/// step wall-time); `0.3` weights the newest step at 30%, damping the
/// step-to-step jitter of a continuous-batching engine while still tracking
/// real throughput shifts within a few steps.
const TPS_EMA_ALPHA: f64 = 0.3;

/// Fold a fresh throughput sample into the running EMA.
///
/// The first sample (when `current == 0.0`) seeds the average directly, so the
/// gauge jumps to a real value on the first step instead of crawling up from 0.
fn fold_ema(current: f64, sample: f64) -> f64 {
    if current == 0.0 {
        sample
    } else {
        TPS_EMA_ALPHA * sample + (1.0 - TPS_EMA_ALPHA) * current
    }
}

/// A live request's routing + timing state, held in the engine thread's route
/// table for the request's lifetime.
struct ActiveRequest {
    /// Where this request's tokens are streamed (→ SSE).
    token_tx: TokenSender,
    /// When the request arrived at the handler — the zero point for both the
    /// time-to-first-token and end-to-end duration metrics.
    requested_at: Instant,
    /// Whether the first token has been emitted yet (so TTFT is recorded once).
    first_token_seen: bool,
    /// Semaphore permit held for the lifetime of this request.
    /// Dropped automatically when the route is removed → slot released.
    #[allow(dead_code)] // held for its Drop side-effect; never read directly
    permit: OwnedSemaphorePermit,
    /// Tail of the rendered prompt (last 500 chars), captured at submission
    /// time. Used only for the `pool_exhausted` warn log in `step_and_route`
    /// (the project's internal engineering log) — not shown to
    /// clients, just server-side context for a rare, worth-investigating event.
    prompt_tail: String,
    /// This request's `max_new_tokens` budget — same log, same rationale.
    max_new_tokens: usize,
    /// This request's exact prompt token count (0 if the count failed —
    /// see the `AddRequest` handler). Embedded in the `pool_exhausted` error
    /// message's `observed_prompt_tokens=` field so
    /// `ModelManager::spawn_kv_wedge_recovery` can ratchet on it the same
    /// way it already does for the VLM wedge marker
    /// (the project's internal engineering log).
    prompt_tokens: usize,
}

/// A message from an async handler to the engine thread.
pub enum EngineCommand {
    /// Enqueue a new generation request.
    AddRequest {
        /// Unique id (monotonic, from `AppState::next_request_id`).
        id: u64,
        /// The prompt text (chat template already applied by the caller). Used
        /// for the string-tokenize path when `prompt_ids` is `None`.
        prompt: String,
        /// Pre-tokenized prompt ids from the handler's L0 length gate (T7.3).
        /// `Some` → submit these directly (one tokenize pass total) and use the
        /// length for `usage.prompt_tokens`; `None` → the engine tokenizes the
        /// `prompt` string itself (gate disabled or its tokenize failed open).
        prompt_ids: Option<Vec<i64>>,
        /// Sampling parameters (`temperature`, `top_p`, stop strings, …).
        gen_params: GenParams,
        /// Where this request's tokens are streamed (→ SSE).
        token_tx: TokenSender,
        /// Handler-entry timestamp — the zero point for TTFT + duration metrics.
        requested_at: Instant,
        /// Capacity permit — dropped when the engine finishes or drops this request,
        /// returning the slot to the semaphore so waiting handlers can proceed.
        permit: OwnedSemaphorePermit,
    },
    /// Encode text → token ids on the engine thread (the only thread allowed to
    /// touch the tokenizer). Serves `POST /tokenize`. The result (or a stringified
    /// tokenizer error) is returned on `reply`.
    Tokenize {
        /// Text to encode.
        text: String,
        /// Where the engine thread sends the encoded ids (or an error string).
        reply: oneshot::Sender<Result<Vec<i64>, String>>,
    },
    /// Decode token ids → text on the engine thread. Serves `POST /detokenize`.
    /// The result (or a stringified detokenizer error) is returned on `reply`.
    Detokenize {
        /// Token ids to decode.
        ids: Vec<i64>,
        /// Where the engine thread sends the decoded text (or an error string).
        reply: oneshot::Sender<Result<String, String>>,
    },
}

/// Result of submitting a request to the engine.
#[derive(Debug, PartialEq, Eq)]
pub enum SubmitResult {
    /// Request accepted — the engine thread owns `token_tx` until done.
    Submitted,
    /// Engine is at its `max_num_seqs` limit — caller should return HTTP 429.
    AtCapacity,
    /// Command channel closed (engine thread exited) — caller should return HTTP 503.
    EngineDead,
}

/// Admission-gate outcome for **request/response** engines (embedding, STT,
/// TTS) — the same three outcomes as [`SubmitResult`], adapted for engines
/// whose call directly returns the inference result instead of handing it off
/// to a separate token stream. `SubmitResult` has no payload because CB/VLM/
/// NPU submission and result delivery are two different channels (the command
/// vs. `token_tx`); a request/response engine has only one return path, so its
/// success case must carry either the real output or the pipeline failure.
///
/// Not to be confused with the future cross-pipeline admission middleware
/// (the project's internal engineering log's `DeviceBudgets`/
/// `WorkLease`) — that is a *second*, device-wide gate sitting in front of
/// each engine's own admission gate; this type is purely per-engine, exactly
/// like `SubmitResult`.
#[derive(Debug)]
pub enum AdmitError {
    /// No permit acquired within `queue_timeout` — caller should return HTTP 429.
    AtCapacity,
    /// Command channel closed (engine thread exited) — caller should return HTTP 503.
    EngineDead,
    /// Admitted, but the pipeline call itself failed.
    Failed(anyhow::Error),
}

impl std::fmt::Display for AdmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AtCapacity => write!(f, "engine at capacity"),
            Self::EngineDead => write!(f, "engine unavailable (engine thread exited)"),
            Self::Failed(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for AdmitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Failed(e) => Some(e.as_ref()),
            Self::AtCapacity | Self::EngineDead => None,
        }
    }
}

/// A cheap, cloneable handle the chat handler uses to submit work to the
/// engine thread. Cloning clones the command `Sender` and `Arc`s (O(1)).
#[derive(Clone, Debug)]
pub struct EngineHandle {
    cmd_tx: mpsc::Sender<EngineCommand>,
    /// Semaphore with `max_seqs` permits. A permit is acquired on submit and
    /// held by the engine thread until the request finishes, decoupling the
    /// KV-safe cap from admission. When no permits are available, `add_request`
    /// waits up to `queue_timeout` before returning [`SubmitResult::AtCapacity`].
    sem: Arc<Semaphore>,
    /// `max_seqs` permits in the semaphore — also the HTTP `/metrics` cap label.
    max_seqs: usize,
    /// Maximum time to wait for a semaphore permit before returning 429.
    queue_timeout: Duration,
    /// Callers currently parked in [`add_request`](Self::add_request) awaiting a
    /// semaphore permit — the engine's honest `rustedvino_requests_waiting`
    /// (R2). Incremented before the permit wait, decremented the moment it
    /// resolves (granted or timed out). Distinct from `active()` (already
    /// running): a caller is *waiting* only while blocked at the admission gate.
    waiting: Arc<AtomicUsize>,
    /// Set by [`request_shutdown`](Self::request_shutdown) to tell the engine
    /// thread to stop promptly. The engine checks it at the top of every loop
    /// iteration, so it breaks out within one model step — even mid-generation
    /// and even when a *clone* of this handle is still held by an in-flight
    /// handler (which keeps the command channel open). Without it, eviction /
    /// process shutdown would block on the join until that generation finished
    /// naturally (up to minutes): the off-switch hang.
    shutting_down: Arc<AtomicBool>,
    /// Live KV-cache occupancy percentage (0–100), as `f32` bits, published by
    /// the engine thread after each step and read sync by `/metrics`
    /// (`rustedvino_kv_cache_usage_percent`, co-residency Slice 3a). `0.0` while
    /// idle (no live batch) and until the first step runs. Mirrors how
    /// `waiting`/`active` expose engine-thread state without a command
    /// round-trip — the metrics path stays fully synchronous.
    cache_usage: Arc<AtomicU64>,
    /// Sum of prompt-token counts for every request currently admitted on
    /// this engine (reserved at admission, released when the request
    /// finishes — see [`crate::in_flight::AdmittedTokensGuard`]). Unlike
    /// `cache_usage`, this is synchronous and current the instant a request
    /// is admitted, with no dependency on the engine having run a step yet —
    /// exactly the gap that made `cache_usage` unsafe as an admission signal
    /// (measured live: stays `0.0` for ~1.5s after a large request starts,
    /// the project's internal engineering log). Read by the concurrent-admission
    /// check in `handlers::chat::gate_prompt`'s caller, not by `/metrics`.
    admitted_prompt_tokens: Arc<AtomicUsize>,
}

impl EngineHandle {
    /// Creates an `EngineHandle` from a raw command sender, for tests.
    ///
    /// Only compiled in test builds — `MockEngineFactory` uses this to hand
    /// back a real handle without going through `OvCbEngine`. Production code
    /// always gets handles from [`spawn_engine`]. Uses a large test cap
    /// (`1000`) and a long timeout so tests never hit the admission gate.
    #[cfg(test)]
    pub(crate) fn from_sender(cmd_tx: mpsc::Sender<EngineCommand>) -> Self {
        let max_seqs = 1000;
        Self {
            cmd_tx,
            sem: Arc::new(Semaphore::new(max_seqs)),
            max_seqs,
            queue_timeout: Duration::from_secs(30),
            waiting: Arc::new(AtomicUsize::new(0)),
            shutting_down: Arc::new(AtomicBool::new(false)),
            cache_usage: Arc::new(AtomicU64::new(0)),
            admitted_prompt_tokens: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Creates a test handle with an explicit cap and timeout. Used by
    /// admission-gate tests that need to observe `AtCapacity` behaviour.
    #[cfg(test)]
    pub(crate) fn from_sender_with_cap(
        cmd_tx: mpsc::Sender<EngineCommand>,
        max_seqs: usize,
        queue_timeout_ms: u64,
    ) -> Self {
        Self {
            cmd_tx,
            sem: Arc::new(Semaphore::new(max_seqs)),
            max_seqs,
            queue_timeout: Duration::from_millis(queue_timeout_ms),
            waiting: Arc::new(AtomicUsize::new(0)),
            shutting_down: Arc::new(AtomicBool::new(false)),
            cache_usage: Arc::new(AtomicU64::new(0)),
            admitted_prompt_tokens: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Occupy one admission slot, returning the held permit, to simulate an
    /// in-flight request in tests. While the returned permit is alive,
    /// [`active`](Self::active) reports one more running request; dropping it
    /// releases the slot. Used by the T5.4 in-flight-aware LRU test to make a
    /// model "busy" without a live engine thread.
    #[cfg(test)]
    pub(crate) fn occupy_slot_for_test(&self) -> Option<tokio::sync::OwnedSemaphorePermit> {
        Arc::clone(&self.sem).try_acquire_owned().ok()
    }

    /// Signal the engine thread to stop promptly, regardless of whether other
    /// handle clones still hold the command channel open.
    ///
    /// The engine checks this flag at the top of every loop iteration, so it
    /// breaks out within one model step (~tens of ms) even mid-generation. Used
    /// by the process-shutdown path so an in-flight non-streaming / tool request
    /// cannot pin the eviction join for the whole completion (#7). In-flight
    /// requests are then terminated cleanly by the engine loop's T1.3 teardown.
    pub fn request_shutdown(&self) {
        self.shutting_down.store(true, Ordering::Relaxed);
    }

    /// Submit a generation request.
    ///
    /// Waits up to `queue_timeout` for a capacity slot before returning
    /// [`SubmitResult::AtCapacity`] (→ HTTP 429). This short wait absorbs
    /// the refire bursts that pipelined clients generate while the previous
    /// batch completes, avoiding spurious 429s at moderate load.
    ///
    /// Returns [`SubmitResult::EngineDead`] if the command channel is closed
    /// (engine thread exited) — caller should respond HTTP 503.
    ///
    /// On success, `token_tx` is consumed by the engine until completion.
    ///
    /// CRASH COURSE — `Semaphore`:
    ///   `Semaphore::new(n)` creates a pool of `n` permits. `acquire_owned()`
    ///   waits asynchronously until one is available and returns an
    ///   `OwnedSemaphorePermit`. When that permit is dropped, it's returned
    ///   automatically — no explicit release needed. This replaces the old
    ///   `AtomicUsize` CAS loop and, crucially, gives waiting callers a
    ///   lightweight async notification when a slot opens rather than requiring
    ///   them to poll or time-out immediately.
    pub async fn add_request(
        &self,
        id: u64,
        prompt: String,
        prompt_ids: Option<Vec<i64>>,
        gen_params: GenParams,
        token_tx: TokenSender,
        requested_at: Instant,
    ) -> SubmitResult {
        // Wait up to `queue_timeout` for a capacity permit. Count this caller as
        // `waiting` for exactly the span it is parked at the admission gate, so
        // `rustedvino_requests_waiting` reflects real queue pressure (R2).
        //
        // `InFlightGuard` (not just an increment) so a cancelled caller — an
        // HTTP client disconnecting while parked at the gate — still
        // decrements: its `Drop` runs even if this `.await` is torn down
        // mid-poll, unlike a bare `fetch_sub` placed after the await.
        let permit_result = {
            let _waiting_guard = crate::in_flight::InFlightGuard::new(&self.waiting);
            tokio::time::timeout(self.queue_timeout, Arc::clone(&self.sem).acquire_owned()).await
        };
        let permit = match permit_result {
            Ok(Ok(p)) => p,
            Ok(Err(_)) => return SubmitResult::EngineDead, // semaphore closed (shouldn't happen)
            Err(_) => return SubmitResult::AtCapacity,     // timed out waiting for a slot
        };

        // Permit acquired. Send to engine — if the channel is closed, the
        // permit drops here and the slot is returned to the semaphore.
        let sent = self
            .cmd_tx
            .send(EngineCommand::AddRequest {
                id,
                prompt,
                prompt_ids,
                gen_params,
                token_tx,
                requested_at,
                permit,
            })
            .await;

        if sent.is_err() {
            // The permit was moved into the EngineCommand; .send() returns it
            // via the Err variant (SentError<EngineCommand>), which is dropped
            // here — automatically releasing the slot.
            return SubmitResult::EngineDead;
        }

        SubmitResult::Submitted
    }

    /// Number of accepted-but-not-yet-finished requests on this engine.
    ///
    /// Derived from the semaphore: `max_seqs − available_permits`. Used by
    /// the `/metrics` endpoint (`rustedvino_requests_running`).
    #[must_use]
    pub fn active(&self) -> usize {
        self.max_seqs.saturating_sub(self.sem.available_permits())
    }

    /// Configured maximum concurrent sequences (the 429 capacity gate).
    ///
    /// Exposed for `/metrics` (`rustedvino_requests_max`).
    #[must_use]
    pub fn max_seqs(&self) -> usize {
        self.max_seqs
    }

    /// Callers currently blocked in [`add_request`](Self::add_request) awaiting a
    /// capacity permit. Exposed for `/metrics` (`rustedvino_requests_waiting`).
    #[must_use]
    pub fn waiting(&self) -> usize {
        self.waiting.load(Ordering::Relaxed)
    }

    /// Live KV-cache occupancy percentage (0–100) the engine thread published
    /// after its last step. Exposed for `/metrics`
    /// (`rustedvino_kv_cache_usage_percent`). `0.0` while idle or before the
    /// first step.
    #[must_use]
    pub fn cache_usage_pct(&self) -> f64 {
        f64::from_bits(self.cache_usage.load(Ordering::Relaxed))
    }

    /// Sum of prompt-token counts for every request currently admitted on
    /// this engine. See [`Self::admitted_prompt_tokens`] field doc for why
    /// this is used for admission instead of [`Self::cache_usage_pct`].
    #[must_use]
    pub fn admitted_prompt_tokens(&self) -> usize {
        self.admitted_prompt_tokens.load(Ordering::Relaxed)
    }

    /// Attempts to reserve `tokens` against this engine's pool-capacity
    /// budget, atomically accounting for every other request already
    /// admitted. Returns `None` (nothing reserved) when admitting `tokens`
    /// would push the total over `capacity` — the caller should reject the
    /// request (fast, honest 400/429) instead of submitting it to the engine
    /// and letting the scheduler thrash. On `Some`, hold the guard for the
    /// WHOLE request lifetime (through generation, not just submission) —
    /// see [`crate::in_flight::AdmittedTokensGuard`]'s doc comment.
    #[must_use]
    pub(crate) fn try_admit_tokens(
        &self,
        tokens: usize,
        capacity: usize,
    ) -> Option<crate::in_flight::AdmittedTokensGuard> {
        crate::in_flight::AdmittedTokensGuard::try_new(
            &self.admitted_prompt_tokens,
            tokens,
            capacity,
        )
    }

    /// Encode `text` into token ids using the model's tokenizer.
    ///
    /// The tokenizer lives on the engine thread (its `InferRequest` is not
    /// thread-safe), so this routes through the command channel and awaits a
    /// `oneshot` reply rather than calling the tokenizer directly. It does NOT
    /// consume a capacity slot — tokenization is not a generation request and is
    /// never rejected with 429.
    ///
    /// # Errors
    /// Returns an error if the engine thread is gone (channel closed or reply
    /// dropped) or the tokenizer itself fails.
    pub async fn tokenize(&self, text: String) -> anyhow::Result<Vec<i64>> {
        let (reply, reply_rx) = oneshot::channel();
        self.cmd_tx
            .send(EngineCommand::Tokenize { text, reply })
            .await
            .map_err(|_| anyhow::anyhow!("inference engine unavailable"))?;
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("engine dropped tokenize reply"))?
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Decode token `ids` back to text using the model's detokenizer.
    ///
    /// Same engine-thread routing and slot-free contract as
    /// [`tokenize`](Self::tokenize).
    ///
    /// # Errors
    /// Returns an error if the engine thread is gone or the detokenizer fails.
    pub async fn detokenize(&self, ids: Vec<i64>) -> anyhow::Result<String> {
        let (reply, reply_rx) = oneshot::channel();
        self.cmd_tx
            .send(EngineCommand::Detokenize { ids, reply })
            .await
            .map_err(|_| anyhow::anyhow!("inference engine unavailable"))?;
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("engine dropped detokenize reply"))?
            .map_err(|e| anyhow::anyhow!(e))
    }
}

/// The CB engine is a [`ModelKind::TextGen`] engine. This impl is the
/// `ManagedEngine` facade the `ModelManager` reads for metrics and lifecycle —
/// it mirrors the inherent [`active`](EngineHandle::active) /
/// [`max_seqs`](EngineHandle::max_seqs) accessors. The bodies touch the fields
/// directly (rather than self-calling the inherent methods) so the trait and
/// inherent names never shadow each other.
impl ManagedEngine for EngineHandle {
    fn kind(&self) -> ModelKind {
        ModelKind::TextGen
    }

    fn active(&self) -> usize {
        self.max_seqs.saturating_sub(self.sem.available_permits())
    }

    fn max_concurrency(&self) -> usize {
        self.max_seqs
    }

    fn waiting(&self) -> usize {
        self.waiting.load(Ordering::Relaxed)
    }

    fn cache_usage_pct(&self) -> f64 {
        f64::from_bits(self.cache_usage.load(Ordering::Relaxed))
    }

    /// The plain CB engine is the one kind that can actually query its real
    /// KV-pool occupancy (`ov_cb_get_metrics`, see [`Self::cache_usage_pct`])
    /// — unlike the NPU engine, which shares `ModelKind::TextGen`'s label but
    /// runs `OpenVINO`'s static `LLMPipeline` (no batched pool to query at
    /// all), or the VLM engine, which has a real pool but no public API to
    /// read it (see `ManagedEngine::cache_usage_pct`'s doc comment).
    fn cache_usage_supported(&self) -> bool {
        true
    }
}

/// Take ownership of a loaded [`OvCbEngine`], spawn the engine thread, and
/// return a handle for submitting work plus the thread's [`JoinHandle`].
///
/// `max_seqs` is the maximum number of concurrently active requests. Any
/// [`EngineHandle::add_request`] call that would exceed this limit returns
/// [`SubmitResult::AtCapacity`] immediately — no GPU work is started and the
/// caller can respond HTTP 429.
///
/// The engine thread runs until every [`EngineHandle`] clone is dropped (the
/// command channel closes), finishes any in-flight requests, then exits.
///
/// The [`std::thread::JoinHandle`] is returned so the model manager can
/// **join** the thread during model eviction — guaranteeing VRAM is freed
/// before a replacement model is loaded (the quiesce protocol).
///
/// # Errors
/// Returns an error only if the OS refuses to spawn the thread.
/// Take ownership of a loaded [`OvCbEngine`], spawn the engine thread, and
/// return a handle for submitting work plus the thread's [`JoinHandle`].
///
/// `max_seqs` is the KV-pool safety cap AND the semaphore size; `0` is
/// normalized to [`OV_DEFAULT_MAX_NUM_SEQS`] before either use.
/// `queue_timeout` is the maximum time `add_request` waits for a slot.
///
/// The engine thread runs until every [`EngineHandle`] clone is dropped,
/// finishes any in-flight requests, then exits.
///
/// # Errors
/// Returns an error only if the OS refuses to spawn the thread.
pub fn spawn_engine(
    engine: OvCbEngine,
    max_seqs: usize,
    queue_timeout: Duration,
    model_id: &str,
    device: &str,
) -> anyhow::Result<(EngineHandle, std::thread::JoinHandle<()>)> {
    // `max_seqs == 0` is the "let OpenVINO decide" sentinel, not "zero
    // concurrency." `OvCbEngine::new` (called separately, before this)
    // already passes the raw 0 through correctly — the C++ scheduler keeps
    // its own default in that case. Mirror that same effective value here so
    // the Rust-side admission gate doesn't become a stricter, undocumented
    // bottleneck of zero permits.
    let max_seqs = resolve_admission_cap(max_seqs);
    let (cmd_tx, cmd_rx) = mpsc::channel::<EngineCommand>(COMMAND_CAPACITY);
    let sem = Arc::new(Semaphore::new(max_seqs));
    // Shared shutdown flag: the handle sets it (request_shutdown), the engine
    // thread polls it every step so it can stop even while a clone of the
    // handle keeps the command channel open (#7).
    let shutting_down = Arc::new(AtomicBool::new(false));
    let shutting_down_engine = Arc::clone(&shutting_down);
    // Live KV occupancy gauge: the engine thread writes it each step, `/metrics`
    // reads it sync (Slice 3a). Shared the same way as `shutting_down`.
    let cache_usage = Arc::new(AtomicU64::new(0));
    let cache_usage_engine = Arc::clone(&cache_usage);

    // Resolve the metric label handles once, here, off the hot path.
    let metrics = HotMetrics::new(ModelKind::TextGen, model_id, device);

    // Owned copies for the engine thread's lifecycle logs (model_id + kind on
    // every path — R2). These are off the per-token hot path.
    let model_id_log = model_id.to_owned();
    let device_log = device.to_owned();

    let thread = std::thread::Builder::new()
        .name("cb-engine".to_owned())
        .spawn(move || {
            tracing::info!(
                model_id = model_id_log,
                kind = ModelKind::TextGen.label(),
                device = device_log,
                "CB engine thread started"
            );
            engine_loop(
                engine,
                cmd_rx,
                metrics,
                shutting_down_engine,
                cache_usage_engine,
            );
            tracing::info!(
                model_id = model_id_log,
                kind = ModelKind::TextGen.label(),
                "CB engine thread exiting — all handles dropped"
            );
        })?;

    Ok((
        EngineHandle {
            cmd_tx,
            sem,
            max_seqs,
            queue_timeout,
            waiting: Arc::new(AtomicUsize::new(0)),
            shutting_down,
            cache_usage,
            admitted_prompt_tokens: Arc::new(AtomicUsize::new(0)),
        },
        thread,
    ))
}

/// The subset of [`OvCbEngine`]'s interface the engine loop needs, extracted
/// so tests can drive [`engine_loop`] against a deterministic fake instead of
/// a real GPU pipeline (see `tests::FakeEngine`). Production code always uses
/// [`OvCbEngine`] via [`spawn_engine`] — this trait exists purely for that
/// substitution, not as a general abstraction layer.
trait CbEngineOps {
    /// Forwards to [`OvCbEngine::add_request`].
    fn add_request(
        &mut self,
        request_id: u64,
        prompt: &str,
        params: &GenParams,
    ) -> anyhow::Result<()>;
    /// Forwards to [`OvCbEngine::add_request_ids`].
    fn add_request_ids(
        &mut self,
        request_id: u64,
        ids: &[i64],
        params: &GenParams,
    ) -> anyhow::Result<()>;
    /// Forwards to [`OvCbEngine::drop_request`].
    fn drop_request(&mut self, request_id: u64);
    /// Forwards to [`OvCbEngine::has_unfinished`].
    fn has_unfinished(&self) -> bool;
    /// Forwards to [`OvCbEngine::cache_usage_pct`].
    fn cache_usage_pct(&self) -> Option<f64>;
    /// Forwards to [`OvCbEngine::step`].
    fn step(
        &mut self,
        on_token: impl FnMut(u64, &str, usize, Option<FinishReason>, bool),
    ) -> anyhow::Result<Vec<u64>>;
    /// Forwards to [`OvCbEngine::count_tokens`].
    fn count_tokens(&self, text: &str) -> anyhow::Result<usize>;
    /// Forwards to [`OvCbEngine::encode`].
    fn encode(&self, text: &str) -> anyhow::Result<Vec<i64>>;
    /// Forwards to [`OvCbEngine::decode`].
    fn decode(&self, ids: &[i64]) -> anyhow::Result<String>;
}

impl CbEngineOps for OvCbEngine {
    fn add_request(
        &mut self,
        request_id: u64,
        prompt: &str,
        params: &GenParams,
    ) -> anyhow::Result<()> {
        OvCbEngine::add_request(self, request_id, prompt, params)
    }
    fn add_request_ids(
        &mut self,
        request_id: u64,
        ids: &[i64],
        params: &GenParams,
    ) -> anyhow::Result<()> {
        OvCbEngine::add_request_ids(self, request_id, ids, params)
    }
    fn drop_request(&mut self, request_id: u64) {
        OvCbEngine::drop_request(self, request_id);
    }
    fn has_unfinished(&self) -> bool {
        OvCbEngine::has_unfinished(self)
    }
    fn cache_usage_pct(&self) -> Option<f64> {
        OvCbEngine::cache_usage_pct(self)
    }
    fn step(
        &mut self,
        on_token: impl FnMut(u64, &str, usize, Option<FinishReason>, bool),
    ) -> anyhow::Result<Vec<u64>> {
        OvCbEngine::step(self, on_token)
    }
    fn count_tokens(&self, text: &str) -> anyhow::Result<usize> {
        OvCbEngine::count_tokens(self, text)
    }
    fn encode(&self, text: &str) -> anyhow::Result<Vec<i64>> {
        OvCbEngine::encode(self, text)
    }
    fn decode(&self, ids: &[i64]) -> anyhow::Result<String> {
        OvCbEngine::decode(self, ids)
    }
}

/// The engine thread body. Owns the pipeline and the route table for its
/// entire life. Runs until the command channel closes (every `EngineHandle`
/// clone dropped — eviction or process shutdown) AND every in-flight request
/// has finished.
///
/// Draining vs. aborting is decided by `shutting_down`, not by the channel
/// alone: a routine admin/LRU eviction (`shutting_down` stays `false`) lets
/// any live requests finish naturally once the channel disconnects — the
/// loop keeps stepping until `routes` is empty, THEN exits. Process shutdown
/// (`abort_inflight`, `ModelManager::shutdown`) sets `shutting_down` first, so
/// the top-of-loop check below exits within one step regardless of live
/// requests (#7) — the T1.3 teardown then terminates them cleanly.
///
/// On exit, any request still in flight (only reachable via the abort path,
/// or a genuinely wedged pipeline) gets a terminal `Error` + `Done` on its
/// stream (T1.3, finding 44): eviction/shutdown must never leave an SSE
/// client hanging on a stream that just stops. The pipeline itself is dropped
/// when this function returns, cancelling the abandoned generations and
/// freeing VRAM.
///
/// Capacity slots are tracked by `OwnedSemaphorePermit` stored in each
/// `ActiveRequest` — they are returned to the semaphore automatically when
/// the route is removed, with no explicit counter management needed here.
#[allow(clippy::needless_pass_by_value)]
fn engine_loop<E: CbEngineOps>(
    mut engine: E,
    mut cmd_rx: mpsc::Receiver<EngineCommand>,
    metrics: HotMetrics,
    shutting_down: Arc<AtomicBool>,
    cache_usage: Arc<AtomicU64>,
) {
    // request_id → routing + timing state for that request.
    let mut routes: HashMap<u64, ActiveRequest> = HashMap::new();
    // Throughput EMA, carried across steps (0.0 = no sample yet).
    let mut tps_ema: f64 = 0.0;
    // Set once the command channel disconnects (every `EngineHandle` clone
    // dropped). From then on `try_recv`/`blocking_recv` can never yield a new
    // command again, but live routes still get to finish naturally below —
    // this flag just skips re-polling a channel that can only ever say
    // "disconnected" again.
    let mut channel_closed = false;

    'run: loop {
        // 0. Explicit shutdown signal (eviction / process shutdown). Checked
        //    first so the loop exits within one step even when an in-flight
        //    request still holds a handle clone keeping the channel open (#7).
        //    The T1.3 teardown below terminates any live request cleanly.
        if shutting_down.load(Ordering::Relaxed) {
            break 'run;
        }

        // 1. Drain every command currently queued, without blocking. New
        //    AddRequests here join the live batch on this iteration's step.
        if !channel_closed {
            loop {
                match cmd_rx.try_recv() {
                    Ok(cmd) => handle_command(cmd, &mut engine, &mut routes, &metrics),
                    Err(TryRecvError::Empty) => break,
                    // All handles dropped. Do NOT exit yet: routine eviction
                    // relies on this loop draining any live request to its
                    // natural completion (only `shutting_down`, checked
                    // above, forces an early abort). Falls through to step 2,
                    // which keeps advancing `routes` until empty, then the
                    // idle branch's `blocking_recv` returns `None` and exits.
                    Err(TryRecvError::Disconnected) => {
                        channel_closed = true;
                        break;
                    }
                }
            }
        }

        // Invariant guard: `routes` and the C++ pipeline's own live-request
        // set are two independently maintained views of "work in flight" and
        // can silently diverge (observed: KV-pool exhaustion under
        // GenerationStatus::IGNORED left a request un-finalized on both sides
        // at once — the project's internal engineering log 2026-08-19, nanbeige hang). If the
        // pipeline ever reports nothing left to do while Rust still tracks
        // live routes, those routes are orphaned — nothing will ever step
        // them again (has_unfinished() below would keep this branch skipped
        // forever) and they'd otherwise park this thread in blocking_recv
        // with their permits never released. Fail them now instead: a clean
        // error response beats a permanent wedge.
        if !engine.has_unfinished() && !routes.is_empty() {
            tracing::error!(
                orphaned = routes.len(),
                "engine reports no unfinished work but routes still tracks live requests \
                 — pipeline/Rust bookkeeping diverged, failing orphaned routes"
            );
            fail_all_routes(
                &mut routes,
                "engine lost track of this request (internal error) — please retry",
            );
        }

        // 2. If work is in flight, advance it one step and route the tokens.
        //    Otherwise block until the next command arrives (no busy-spin) —
        //    or, if the channel is already closed, exit: nothing left to wait
        //    for.
        if engine.has_unfinished() {
            step_and_route(&mut engine, &mut routes, &metrics, &mut tps_ema);
            // Slice 3a: publish live KV occupancy for /metrics. `get_metrics()`
            // is a cached-struct read, so this is negligible against the step
            // we just ran. On FFI failure keep the prior value (don't flap to 0).
            if let Some(pct) = engine.cache_usage_pct() {
                cache_usage.store(pct.to_bits(), Ordering::Relaxed);
            }
        } else if channel_closed {
            break 'run; // no live routes, no way to receive new commands
        } else {
            // Idle: no live batch holds KV blocks, so report 0 occupancy
            // (the plan's idle→0 clamp) rather than a stale last-step value.
            cache_usage.store(0.0_f64.to_bits(), Ordering::Relaxed);
            match cmd_rx.blocking_recv() {
                Some(cmd) => handle_command(cmd, &mut engine, &mut routes, &metrics),
                None => break 'run, // channel closed → shut down
            }
        }
    }

    // T1.3: the loop can exit with requests still live (eviction or shutdown
    // mid-generation) — cancel them in the pipeline and terminate each stream
    // cleanly instead of silently dropping it.
    if !routes.is_empty() {
        tracing::warn!(
            in_flight = routes.len(),
            "engine loop exiting with live requests — sending terminal events"
        );
        for id in routes.keys() {
            engine.drop_request(*id);
        }
        fail_all_routes(
            &mut routes,
            "engine shutting down — request aborted before completion",
        );
    }

    // Cancelled requests release their KV blocks only on a subsequent step.
    // Dropping the pipeline while sequences still hold blocks trips GenAI's
    // `m_block_table.empty()` destructor assertion → C++ `terminate` → SIGABRT
    // (observed live during the T6.1 smoke test). Drive the pipeline
    // until it reports idle; the cap guards against an engine that never
    // settles (a cancelled-only batch steps quickly — no tokens generated).
    let mut settle = 0;
    while engine.has_unfinished() && settle < MAX_SETTLE_STEPS {
        if let Err(e) = engine.step(|_, _, _, _, _| {}) {
            tracing::warn!(error = %e, "settle step failed during engine shutdown");
            break;
        }
        settle += 1;
    }
    if engine.has_unfinished() {
        // The pipeline still holds KV blocks after the capped settle loop.
        // Dropping it now runs ~ContinuousBatchingPipeline, which trips GenAI's
        // `m_block_table.empty()` destructor assertion → C++ `terminate` →
        // SIGABRT: it takes down the whole process AND skips clean L0/xe context
        // teardown, wedging GPU.1 so the next launch fails. Leaking the engine
        // is the lesser evil here: the OS reclaims its VRAM at process exit, and
        // for a live eviction the manager's bounded join still lets it proceed.
        tracing::error!(
            settle_steps = MAX_SETTLE_STEPS,
            "engine still reports unfinished work after settle — leaking pipeline to avoid \
             C++ destructor abort (VRAM reclaimed on process exit)"
        );
        std::mem::forget(engine);
    }
}

/// Fail every live route with a terminal pair — `Error(msg)` then
/// `Done(Stop)` — and remove it (dropping its `ActiveRequest` releases the
/// capacity permit). Used when the whole batch dies at once: a failed engine
/// step, or engine-loop exit with requests still in flight (T1.3).
///
/// `try_send`, best-effort: a stalled client's full channel must never block
/// the engine thread — dropping the route closes its stream either way.
fn fail_all_routes(routes: &mut HashMap<u64, ActiveRequest>, msg: &str) {
    for (_id, req) in routes.drain() {
        let _ = req.token_tx.try_send(StreamEvent::Error(msg.to_owned()));
        let _ = req.token_tx.try_send(StreamEvent::Done(FinishReason::Stop));
        // req drops here, releasing its permit automatically
    }
}

/// Apply one command to the engine + route table.
///
/// Capacity slots are managed via `OwnedSemaphorePermit`:
/// - On success: the permit is stored in `ActiveRequest` and released when
///   the route is eventually removed (finish, drop, or step error).
/// - On failure: the permit drops at the end of the `Err` arm, returning
///   the slot to the semaphore immediately — no explicit release needed.
fn handle_command<E: CbEngineOps>(
    cmd: EngineCommand,
    engine: &mut E,
    routes: &mut HashMap<u64, ActiveRequest>,
    metrics: &HotMetrics,
) {
    match cmd {
        EngineCommand::AddRequest {
            id,
            prompt,
            prompt_ids,
            gen_params,
            token_tx,
            requested_at,
            permit,
        } => {
            // T7.3: prefer the gate's pre-tokenized ids — submitting them skips
            // a redundant tokenize pass the string path would do inside the
            // pipeline. The handler's L0 gate already encoded this prompt; reuse
            // those ids both to submit AND for usage.prompt_tokens (no third
            // encode via count_tokens). `None` (gate disabled / failed open) →
            // fall back to the string path that tokenizes here.
            let submit = match &prompt_ids {
                Some(ids) => engine.add_request_ids(id, ids, &gen_params),
                None => engine.add_request(id, &prompt, &gen_params),
            };
            match submit {
                Ok(()) => {
                    // Counts only requests that reach generation — 404/429/503 are
                    // rejected in the handler before ever becoming a command. A CB
                    // engine is text-gen: image content is 400'd before it can reach
                    // here, so every request is `Modality::Text`.
                    metrics.request_accepted(Modality::Text);
                    // Ship usage.prompt_tokens ahead of the first token. Best-effort:
                    // omit on failure (the collector falls back to 0). When the gate
                    // supplied ids we already have the exact count for free; only the
                    // string fallback pays a count_tokens pass (engine-thread-only
                    // tokenizer + prompt text).
                    let prompt_tokens = match &prompt_ids {
                        Some(ids) => Some(ids.len()),
                        None => match engine.count_tokens(&prompt) {
                            Ok(n) => Some(n),
                            Err(e) => {
                                tracing::warn!(request_id = id, error = %e, "prompt token count failed");
                                None
                            }
                        },
                    };
                    if let Some(n) = prompt_tokens {
                        // try_send: the per-request channel is fresh (this is its
                        // first event), so Full is impossible; and the engine thread
                        // must never block on a client channel.
                        let _ = token_tx.try_send(StreamEvent::PromptTokens(n));
                    }
                    let prompt_tail: String = prompt
                        .chars()
                        .rev()
                        .take(500)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect();
                    routes.insert(
                        id,
                        ActiveRequest {
                            token_tx,
                            requested_at,
                            first_token_seen: false,
                            permit, // held until the route is removed → slot released then
                            prompt_tail,
                            max_new_tokens: gen_params.max_new_tokens,
                            prompt_tokens: prompt_tokens.unwrap_or(0),
                        },
                    );
                }
                Err(e) => {
                    // Submission failed — `permit` drops at end of this arm,
                    // returning the slot automatically. Report on this request's
                    // own stream; other requests are unaffected.
                    tracing::error!(request_id = id, error = %e, "CB add_request failed");
                    // try_send: fresh channel (see PromptTokens above) — and the
                    // engine thread must never block on a client channel.
                    let _ = token_tx.try_send(StreamEvent::Error(e.to_string()));
                    let _ = token_tx.try_send(StreamEvent::Done(FinishReason::Stop));
                    drop(permit); // explicit for clarity — slot released here
                }
            }
        }
        // There is no explicit per-request cancel command: a client disconnect
        // is detected *lazily* on the next step that produces a token for the
        // request (its closed `token_tx` → `deliver_or_drop` → the `disconnected`
        // path in `step_and_route`, which calls `engine.drop_request` and frees
        // the slot). Cancellation is therefore eventual, bounded by one step —
        // never claim it is immediate.
        // Tokenizer requests are processed inline between model steps. Encoding
        // is microseconds against a ~15 ms step, so the brief engine-thread
        // occupancy is negligible. The reply receiver may already be gone (the
        // handler's request was cancelled / client vanished) — ignore the send
        // error in that case.
        EngineCommand::Tokenize { text, reply } => {
            let _ = reply.send(engine.encode(&text).map_err(|e| e.to_string()));
        }
        EngineCommand::Detokenize { ids, reply } => {
            let _ = reply.send(engine.decode(&ids).map_err(|e| e.to_string()));
        }
    }
}

/// Deliver one stream event on a request's token channel WITHOUT ever blocking
/// the engine thread. Returns `true` if the request must be routed out
/// (client gone or not keeping up).
///
/// T1.1 (committee findings 1, 2): the engine thread drives the whole batch —
/// a `blocking_send` into one slow client's full channel stalled every other
/// request's tokens (network-triggerable head-of-line stall: just read your
/// SSE stream slowly). Backpressure policy (per the hardening plan): a client
/// whose channel is full is not keeping up — drop it via the same path as a
/// disconnected client, and let the batch advance. The channel buffers
/// `streaming::CHANNEL_CAPACITY` events, so only a genuinely stalled reader
/// is ever dropped, never one that is merely a few tokens behind.
///
/// Takes `req` (not just its `token_tx`) so the disconnect log below can
/// report how far the request had gotten — a mid-generation client drop
/// previously produced ZERO log output at any level (found in remote
/// verification: a real production disconnect was only reconstructible
/// after the fact from 5-second metric samples, not visible in the log at
/// all).
fn deliver_or_drop(req: &ActiveRequest, id: u64, event: StreamEvent) -> bool {
    match req.token_tx.try_send(event) {
        Ok(()) => false,
        Err(TrySendError::Full(_)) => {
            tracing::warn!(
                request_id = id,
                "client not keeping up (token channel full) — dropping its stream"
            );
            true
        }
        Err(TrySendError::Closed(_)) => {
            // Receiver dropped — the client (or its cancelled HTTP handler
            // future) disconnected. `AdmittedTokensGuard`/the semaphore
            // permit still release correctly via routes.remove's Drop chain
            // regardless of this log — this line exists purely so the event
            // is visible without reconstructing it from metrics after the
            // fact.
            tracing::info!(
                request_id = id,
                prompt_tokens = req.prompt_tokens,
                first_token_seen = req.first_token_seen,
                elapsed_secs = req.requested_at.elapsed().as_secs_f64(),
                "client disconnected mid-generation — releasing its KV reservation and slot"
            );
            true
        }
    }
}

/// Run one model step and fan the produced tokens out to their requests.
///
/// Two cleanups happen after the step (collected during it to avoid borrowing
/// `engine` while it is mutably borrowed by `step`):
/// - finished requests are removed from the route table; removing an
///   `ActiveRequest` drops its `OwnedSemaphorePermit`, releasing the slot
///   automatically — no explicit counter decrement needed;
/// - requests whose client has disconnected are cancelled in the pipeline.
///
/// HOT PATH: the callback runs once per produced token across the live batch.
/// Metric work here is restricted to cached-handle atomics (`metrics`) plus a
/// per-request bool flip; `Instant::now()` is called at most once per step
/// (throughput) and once per request (first token / finish), never per token.
// `tokens_this_step` is bounded by the batch's per-step output (a few hundred
// at most), so the u64→f64 cast for the throughput sample cannot lose precision.
#[allow(clippy::cast_precision_loss)]
fn step_and_route<E: CbEngineOps>(
    engine: &mut E,
    routes: &mut HashMap<u64, ActiveRequest>,
    metrics: &HotMetrics,
    tps_ema: &mut f64,
) {
    let mut finished: Vec<u64> = Vec::new();
    let mut disconnected: Vec<u64> = Vec::new();
    // Aggregate output of this single step, for the throughput sample.
    let mut tokens_this_step: u64 = 0;
    let step_start = Instant::now();

    let step_result = engine.step(|id, delta, new_tokens, finish, pool_exhausted| {
        // Captured before the `get_mut` borrow below (which would otherwise
        // conflict with a second borrow of `routes` for its length): how many
        // requests are concurrently in flight against *this model's* engine
        // thread right now, itself included. Embedded in the pool_exhausted
        // error message so `classify_inference_error` can tell genuine
        // multi-tenant congestion (>1 — expected to clear on its own) apart
        // from a single conversation alone hitting the real ceiling (<=1 —
        // nothing else could have crowded it out) — see
        // `model_manager::POOL_EXHAUSTED_MARKER`'s doc comment.
        let active_requests = routes.len();
        if let Some(req) = routes.get_mut(&id) {
            let mut client_gone = false;
            if !delta.is_empty() {
                // Count the engine's own reported token count, not 1 per
                // callback: a speculative-decoding verification step can
                // accept several draft tokens at once, all landing in this
                // same callback (found live 2026-07-19, see the project's internal engineering log).
                // On this request's very first token, record its
                // time-to-first-token exactly once.
                let new_tokens_u64 = new_tokens as u64;
                tokens_this_step += new_tokens_u64;
                metrics.token_generated(new_tokens_u64);
                if !req.first_token_seen {
                    req.first_token_seen = true;
                    metrics.record_ttft(Modality::Text, req.requested_at.elapsed().as_secs_f64());
                }
                if deliver_or_drop(req, id, StreamEvent::Token(delta.to_owned(), new_tokens)) {
                    client_gone = true;
                }
            }
            if let Some(reason) = finish {
                if pool_exhausted {
                    // 2026-08-21 (found live, instant-EOS-stall investigation):
                    // confirmed by direct instrumentation that requests here hit
                    // GenerationStatus::IGNORED — the CB scheduler's KV-pool ran
                    // out of room for this request and it never actually ran,
                    // despite what `reason` claims. Previously this silently
                    // became a fake successful empty completion
                    // (`finish_reason: "stop"`, zero tokens) — indistinguishable
                    // from genuine EOS at the API. Report it as a real error
                    // instead (mirrors `route_out_panicked`'s Error-then-Done
                    // shape): `inference_error_from_text` maps this specific
                    // message to a retryable 503 without poisoning the GPU
                    // (unlike a real OpenCL OOM, other requests are unaffected —
                    // this one just couldn't fit in the pool at this moment).
                    let _ = req.token_tx.try_send(StreamEvent::Error(format!(
                        "generation ignored by the CB scheduler — {POOL_EXHAUSTED_MARKER} \
                         (observed_prompt_tokens={} active_requests={active_requests})",
                        req.prompt_tokens
                    )));
                    tracing::warn!(
                        request_id = id,
                        max_new_tokens = req.max_new_tokens,
                        prompt_tokens = req.prompt_tokens,
                        active_requests,
                        prompt_tail = %req.prompt_tail,
                        "CB generation hit KV-pool exhaustion (GenerationStatus::IGNORED) \
                         — reporting as an error, not a fake completion"
                    );
                }
                if deliver_or_drop(req, id, StreamEvent::Done(reason)) {
                    client_gone = true;
                }
                // Generation completed — record the end-to-end duration.
                metrics.record_duration(Modality::Text, req.requested_at.elapsed().as_secs_f64());
            }
            if client_gone {
                disconnected.push(id);
            }
        }
        if finish.is_some() {
            finished.push(id);
        }
    });

    // Throughput sample: this step's aggregate tokens over its wall-time,
    // folded into the EMA gauge. One Instant read per step, off the token loop.
    let dt = step_start.elapsed().as_secs_f64();
    if tokens_this_step > 0 && dt > 0.0 {
        let inst_tps = tokens_this_step as f64 / dt;
        *tps_ema = fold_ema(*tps_ema, inst_tps);
        metrics.set_tokens_per_second(*tps_ema);
    }

    // Remove finished routes. Dropping `ActiveRequest` drops its permit → slot released.
    for id in finished {
        routes.remove(&id);
    }

    let panicked = match step_result {
        Ok(panicked) => panicked,
        Err(e) => {
            // A step failure is engine-wide: fail every in-flight request and
            // clear. The trailing Done's finish_reason is moot after Error;
            // Stop is the safe default for SSE clients. Cancel each request in
            // the pipeline too — their KV blocks must be reclaimable (a block
            // still held at pipeline drop aborts the process, see engine_loop)
            // and a failed step must not strand them.
            tracing::error!(error = %e, "CB engine step failed — failing all in-flight requests");
            for id in routes.keys() {
                engine.drop_request(*id);
            }
            fail_all_routes(routes, &e.to_string());
            return;
        }
    };

    // Route out any request whose token callback panicked (panic firewall in
    // ov_cb.rs). The pipeline half is already cancelled inside `step`; here we
    // surface a clean stream error and drop the route — releasing its permit —
    // so one poisoned request never takes the node down.
    route_out_panicked(routes, &panicked);

    // Cancel any requests whose client vanished mid-stream.
    for id in disconnected {
        engine.drop_request(id);
        routes.remove(&id); // drops ActiveRequest → permit released
    }
}

/// Remove the routes of requests whose token callback panicked and surface a
/// clean stream error (+ terminal `Done`) to each affected client. Other
/// requests are untouched — the firewall isolates the failure to its request.
fn route_out_panicked(routes: &mut HashMap<u64, ActiveRequest>, panicked: &[u64]) {
    for id in panicked {
        if let Some(req) = routes.remove(id) {
            tracing::error!(
                request_id = id,
                "token routing panicked — failing this request only"
            );
            // try_send, best-effort: never block the engine thread on a
            // client channel (the route is dropped either way, closing the
            // stream).
            let _ = req.token_tx.try_send(StreamEvent::Error(
                "internal error while streaming this response".to_owned(),
            ));
            let _ = req.token_tx.try_send(StreamEvent::Done(FinishReason::Stop));
            // req drops here, releasing its permit automatically
        }
    }
}

// ============================================================
// Unit tests
// ============================================================
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// Deterministic stand-in for `OvCbEngine`, used only to exercise
    /// `engine_loop`'s draining logic without a real GPU pipeline.
    ///
    /// A request's "prompt" is parsed as a decimal step count; `add_request`
    /// records it and `step` decrements it once per call, finishing (emitting
    /// `Done(Stop)`) when it reaches zero. `step` blocks on `gate` until the
    /// test sends a signal — this lets a test pace the fake generation one
    /// step at a time with no sleeps, so the drain-vs-abort race under test
    /// is deterministic rather than timing-dependent.
    struct FakeEngine {
        pending: HashMap<u64, u32>,
        gate: std::sync::mpsc::Receiver<()>,
    }

    impl FakeEngine {
        fn new(gate: std::sync::mpsc::Receiver<()>) -> Self {
            Self {
                pending: HashMap::new(),
                gate,
            }
        }
    }

    impl CbEngineOps for FakeEngine {
        fn add_request(
            &mut self,
            request_id: u64,
            prompt: &str,
            _params: &GenParams,
        ) -> anyhow::Result<()> {
            let steps: u32 = prompt.parse().unwrap_or(1);
            self.pending.insert(request_id, steps);
            Ok(())
        }

        fn add_request_ids(
            &mut self,
            request_id: u64,
            ids: &[i64],
            _params: &GenParams,
        ) -> anyhow::Result<()> {
            let steps = ids.first().map_or(1, |&n| u32::try_from(n).unwrap_or(1));
            self.pending.insert(request_id, steps);
            Ok(())
        }

        fn drop_request(&mut self, request_id: u64) {
            self.pending.remove(&request_id);
        }

        fn has_unfinished(&self) -> bool {
            !self.pending.is_empty()
        }

        fn cache_usage_pct(&self) -> Option<f64> {
            Some(0.0)
        }

        fn step(
            &mut self,
            mut on_token: impl FnMut(u64, &str, usize, Option<FinishReason>, bool),
        ) -> anyhow::Result<Vec<u64>> {
            // Block until the test says "go" — no sleeps, no flakiness.
            let _ = self.gate.recv();
            let mut finished = Vec::new();
            for (&id, remaining) in &mut self.pending {
                *remaining = remaining.saturating_sub(1);
                if *remaining == 0 {
                    on_token(id, "x", 1, Some(FinishReason::Stop), false);
                    finished.push(id);
                } else {
                    on_token(id, "x", 1, None, false);
                }
            }
            for id in &finished {
                self.pending.remove(id);
            }
            Ok(Vec::new())
        }

        fn count_tokens(&self, text: &str) -> anyhow::Result<usize> {
            Ok(text.split_whitespace().count())
        }

        fn encode(&self, _text: &str) -> anyhow::Result<Vec<i64>> {
            Ok(vec![1])
        }

        fn decode(&self, _ids: &[i64]) -> anyhow::Result<String> {
            Ok(String::new())
        }
    }

    /// Regression test for the streaming-eviction-abort bug (live-caught on
    /// real hardware, 2026-07-18): routine (non-abort) eviction drops every
    /// `EngineHandle` clone, but a genuinely in-flight request must still run
    /// to its NATURAL completion — never the "engine shutting down" abort
    /// path — because `shutting_down` (not the channel) is what decides abort
    /// vs. drain.
    #[tokio::test]
    async fn eviction_drains_a_real_in_flight_stream_instead_of_aborting_it() {
        use crate::streaming::stream_channel;

        let (cmd_tx, cmd_rx) = mpsc::channel::<EngineCommand>(4);
        let handle = EngineHandle::from_sender_with_cap(cmd_tx, 4, 1_000);
        let shutting_down = Arc::new(AtomicBool::new(false));
        let cache_usage = Arc::new(AtomicU64::new(0));
        let metrics = HotMetrics::new(ModelKind::TextGen, "fake-model", "CPU");

        let (gate_tx, gate_rx) = std::sync::mpsc::channel::<()>();
        let engine = FakeEngine::new(gate_rx);
        let thread = std::thread::spawn({
            let shutting_down = Arc::clone(&shutting_down);
            let cache_usage = Arc::clone(&cache_usage);
            move || engine_loop(engine, cmd_rx, metrics, shutting_down, cache_usage)
        });

        // A request that needs 3 fake steps to finish (see FakeEngine's doc:
        // the prompt string is parsed as the step count).
        let (tok_tx, mut tok_rx) = stream_channel();
        let submitted = handle
            .add_request(
                1,
                "3".to_owned(),
                None,
                GenParams::default(),
                tok_tx,
                Instant::now(),
            )
            .await;
        assert_eq!(submitted, SubmitResult::Submitted);

        // Let one step run (2 remaining) — the request is genuinely in flight
        // when eviction happens below, not merely queued.
        gate_tx.send(()).unwrap();
        loop {
            match tok_rx.recv().await.expect("stream ended before any token") {
                StreamEvent::Token(..) => break,
                StreamEvent::PromptTokens(_) => {} // ignore, not relevant here
                other => panic!("unexpected event before first token: {other:?}"),
            }
        }

        // Simulate routine (non-abort) admin/LRU eviction: drop every handle
        // clone. `shutting_down` stays false — this must NOT abort the
        // still-running request.
        drop(handle);

        // Let the remaining 2 steps run to natural completion.
        gate_tx.send(()).unwrap();
        gate_tx.send(()).unwrap();

        // The request must finish NATURALLY — Done(Stop) — never Error.
        let mut saw_done = false;
        while let Some(ev) = tok_rx.recv().await {
            match ev {
                StreamEvent::Error(e) => panic!("request was aborted, not drained: {e}"),
                StreamEvent::Done(reason) => {
                    assert_eq!(
                        reason,
                        FinishReason::Stop,
                        "must finish with a natural reason"
                    );
                    saw_done = true;
                }
                StreamEvent::Token(..)
                | StreamEvent::PromptTokens(_)
                | StreamEvent::CompletionTokens(_) => {}
            }
        }
        assert!(saw_done, "must see a natural Done(Stop) after the drain");

        // The engine thread must exit once routes are drained and the
        // channel is closed — bounded join so a regression hangs the test
        // instead of the whole suite.
        tokio::task::spawn_blocking(move || thread.join().unwrap())
            .await
            .unwrap();
    }

    /// Companion to the drain test above: process shutdown (`abort_inflight`)
    /// must still cut a live generation short within about one step, not wait
    /// for it to finish — `request_shutdown` must keep working after the
    /// drain fix.
    #[tokio::test]
    async fn shutdown_still_aborts_a_real_in_flight_stream_promptly() {
        use crate::streaming::stream_channel;

        let (cmd_tx, cmd_rx) = mpsc::channel::<EngineCommand>(4);
        let handle = EngineHandle::from_sender_with_cap(cmd_tx, 4, 1_000);
        let shutting_down = Arc::new(AtomicBool::new(false));
        let cache_usage = Arc::new(AtomicU64::new(0));
        let metrics = HotMetrics::new(ModelKind::TextGen, "fake-model", "CPU");

        let (gate_tx, gate_rx) = std::sync::mpsc::channel::<()>();
        let engine = FakeEngine::new(gate_rx);
        let thread = std::thread::spawn({
            let shutting_down = Arc::clone(&shutting_down);
            let cache_usage = Arc::clone(&cache_usage);
            move || engine_loop(engine, cmd_rx, metrics, shutting_down, cache_usage)
        });

        // A request that would need 100 fake steps to finish naturally.
        let (tok_tx, mut tok_rx) = stream_channel();
        let submitted = handle
            .add_request(
                1,
                "100".to_owned(),
                None,
                GenParams::default(),
                tok_tx,
                Instant::now(),
            )
            .await;
        assert_eq!(submitted, SubmitResult::Submitted);

        // Let one step run — genuinely in flight, 99 steps from a natural finish.
        gate_tx.send(()).unwrap();
        loop {
            match tok_rx.recv().await.expect("stream ended before any token") {
                StreamEvent::Token(..) => break,
                StreamEvent::PromptTokens(_) => {}
                other => panic!("unexpected event before first token: {other:?}"),
            }
        }

        // Process shutdown: flip the flag the engine thread actually polls.
        // NOTE: `handle.request_shutdown()` is not used here — the test
        // handle (`from_sender_with_cap`) owns its own independent
        // `shutting_down` Arc, separate from the one this test wired
        // directly into `engine_loop` above; only the latter is what the
        // engine thread checks. (In production, `spawn_engine` hands the
        // SAME Arc to both, so `EngineHandle::request_shutdown` works as
        // documented — this divergence is purely a test-harness fact.)
        shutting_down.store(true, Ordering::Relaxed);
        drop(handle);

        // Unblock whichever step the engine is currently parked in — the
        // `shutting_down` check only runs BETWEEN full steps, so at most one
        // more step can complete before abort takes effect.
        gate_tx.send(()).unwrap();

        let mut tokens_after_shutdown = 0u32;
        let mut saw_error = false;
        loop {
            match tok_rx
                .recv()
                .await
                .expect("stream must end with a terminal event, not silently close")
            {
                StreamEvent::Token(..) => {
                    tokens_after_shutdown += 1;
                    assert!(
                        tokens_after_shutdown <= 1,
                        "shutdown let more than one extra step run — not prompt"
                    );
                }
                StreamEvent::PromptTokens(_) | StreamEvent::CompletionTokens(_) => {}
                StreamEvent::Error(_) => saw_error = true,
                StreamEvent::Done(reason) => {
                    assert_eq!(reason, FinishReason::Stop);
                    break;
                }
            }
        }
        assert!(
            saw_error,
            "shutdown must surface the abort error, not a natural finish"
        );

        tokio::task::spawn_blocking(move || thread.join().unwrap())
            .await
            .unwrap();
    }

    /// The first throughput sample seeds the EMA directly (no crawl-up from 0).
    #[test]
    fn fold_ema_seeds_on_first_sample() {
        assert!(
            (fold_ema(0.0, 62.0) - 62.0).abs() < f64::EPSILON,
            "first sample must become the EMA value verbatim"
        );
    }

    /// Subsequent samples are blended by `TPS_EMA_ALPHA`, damping jitter.
    #[test]
    fn fold_ema_blends_subsequent_samples() {
        // 0.3 * 100 + 0.7 * 60 = 30 + 42 = 72
        let blended = fold_ema(60.0, 100.0);
        assert!(
            (blended - 72.0).abs() < 1e-9,
            "EMA must weight the new sample at alpha=0.3: got {blended}"
        );
        // The blend always lands strictly between the old EMA and the sample.
        assert!(blended > 60.0 && blended < 100.0);
    }

    /// `max_num_seqs == 0` (the documented "let `OpenVINO` decide" sentinel,
    /// CONFIG.md) must resolve to `OV_DEFAULT_MAX_NUM_SEQS`, not `0` — a
    /// 0-permit admission `Semaphore` would 429 every request forever
    /// (the project's internal engineering log 2026-08-01).
    #[test]
    fn resolve_admission_cap_treats_zero_sentinel_as_ov_default() {
        assert_eq!(resolve_admission_cap(0), OV_DEFAULT_MAX_NUM_SEQS);
    }

    /// Any non-zero value passes through unchanged.
    #[test]
    fn resolve_admission_cap_passes_through_nonzero_values() {
        assert_eq!(resolve_admission_cap(1), 1);
        assert_eq!(resolve_admission_cap(16), 16);
    }

    /// A fresh handle reports zero active requests and its configured cap.
    #[test]
    fn engine_handle_accessors_report_active_and_max() {
        let (tx, _rx) = mpsc::channel::<EngineCommand>(1);
        let handle = EngineHandle::from_sender(tx);
        // from_sender uses a test cap of 1000, all permits available → active = 0.
        assert_eq!(handle.active(), 0, "fresh handle has no active requests");
        assert_eq!(handle.max_seqs(), 1000, "test handle uses the test cap");
    }

    /// When a slot opens (permit released), a waiting `add_request` completes
    /// instead of returning `AtCapacity`.
    #[tokio::test]
    async fn add_request_waits_and_succeeds_when_slot_opens() {
        use crate::streaming::stream_channel;

        let (tx, mut rx) = mpsc::channel::<EngineCommand>(4);
        // cap=1, 500 ms timeout — enough for the test, tight enough to be fast
        let handle = EngineHandle::from_sender_with_cap(tx, 1, 500);

        let (tok_tx, _tok_rx) = stream_channel();
        let result1 = handle
            .add_request(
                1,
                "p".into(),
                None,
                GenParams::default(),
                tok_tx,
                Instant::now(),
            )
            .await;
        assert_eq!(
            result1,
            SubmitResult::Submitted,
            "first request must get the slot"
        );

        // Receive the first command — now we hold the permit inside `first_cmd`.
        let first_cmd = rx.recv().await.unwrap();

        // Release the slot after 50 ms (simulate a short generation).
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            drop(first_cmd); // drops OwnedSemaphorePermit → slot returned
        });

        // Second request should wait and then succeed.
        let (tok_tx2, _tok_rx2) = stream_channel();
        let result2 = handle
            .add_request(
                2,
                "p".into(),
                None,
                GenParams::default(),
                tok_tx2,
                Instant::now(),
            )
            .await;
        assert_eq!(
            result2,
            SubmitResult::Submitted,
            "second request must get slot after first releases it"
        );
    }

    /// When no slot opens within `queue_timeout`, `add_request` returns
    /// `AtCapacity` (→ HTTP 429) rather than waiting forever.
    #[tokio::test]
    async fn add_request_returns_at_capacity_after_timeout() {
        use crate::streaming::stream_channel;

        let (tx, _rx) = mpsc::channel::<EngineCommand>(4);
        // cap=1, 50 ms timeout — short so the test doesn't slow the suite
        let handle = EngineHandle::from_sender_with_cap(tx, 1, 50);

        let (tok_tx, _) = stream_channel();
        // First request gets the slot; the command stays in the channel buffer.
        let _ = handle
            .add_request(
                1,
                "p".into(),
                None,
                GenParams::default(),
                tok_tx,
                Instant::now(),
            )
            .await;

        // Second request should time out (nobody drains the channel, permit held).
        let (tok_tx2, _) = stream_channel();
        let result = handle
            .add_request(
                2,
                "p".into(),
                None,
                GenParams::default(),
                tok_tx2,
                Instant::now(),
            )
            .await;
        assert_eq!(result, SubmitResult::AtCapacity, "must 429 after timeout");
    }

    /// `waiting()` counts exactly the callers currently parked at the admission
    /// gate: zero when idle, one while a second caller blocks on a full
    /// semaphore, back to zero once a slot frees and that caller proceeds.
    #[tokio::test]
    async fn waiting_counts_callers_parked_at_the_admission_gate() {
        use crate::streaming::stream_channel;

        let (tx, mut rx) = mpsc::channel::<EngineCommand>(4);
        // cap=1, 1 s timeout — long enough that the parked caller stays parked
        // while we observe it, short enough not to hang the suite on failure.
        let handle = EngineHandle::from_sender_with_cap(tx, 1, 1000);
        assert_eq!(handle.waiting(), 0, "no parked callers initially");

        // First request takes the only slot; its permit rides inside the
        // buffered command we hold, so the slot stays occupied.
        let (tok_tx, _r) = stream_channel();
        assert_eq!(
            handle
                .add_request(
                    1,
                    "p".into(),
                    None,
                    GenParams::default(),
                    tok_tx,
                    Instant::now()
                )
                .await,
            SubmitResult::Submitted
        );
        let first_cmd = rx.recv().await.unwrap();

        // A second caller (a handle clone shares the waiting counter) parks at
        // the gate because no slot is free.
        let h2 = handle.clone();
        let parked = tokio::spawn(async move {
            let (tok_tx2, _r2) = stream_channel();
            h2.add_request(
                2,
                "p".into(),
                None,
                GenParams::default(),
                tok_tx2,
                Instant::now(),
            )
            .await
        });

        // Poll (bounded) until the blocked caller registers as waiting.
        let mut observed = false;
        for _ in 0..100 {
            if handle.waiting() == 1 {
                observed = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        assert!(observed, "the blocked caller must register as waiting=1");

        // Free the slot → the parked caller acquires it and stops waiting.
        drop(first_cmd);
        assert_eq!(
            parked.await.unwrap(),
            SubmitResult::Submitted,
            "parked caller proceeds once a slot frees"
        );
        assert_eq!(
            handle.waiting(),
            0,
            "waiting returns to 0 after the gate clears"
        );
    }

    /// Cancelling `add_request` while it is only *waiting for a permit* (not
    /// yet admitted) must not leak `waiting()` — regression test for a gap
    /// found in `embed_engine.rs`'s test suite (2026-07-14): a bare
    /// `fetch_add`/`fetch_sub` around the acquire `.await` never runs its
    /// `fetch_sub` if the awaiting future is dropped mid-wait (an HTTP client
    /// disconnecting while queued). Fixed via a block-scoped `InFlightGuard`.
    #[tokio::test]
    async fn cancelling_add_request_while_parked_does_not_leak_waiting() {
        use crate::streaming::stream_channel;

        let (tx, _rx) = mpsc::channel::<EngineCommand>(4);
        let handle = EngineHandle::from_sender_with_cap(tx, 1, 5_000);

        // Take the only permit directly so the next `add_request` call parks.
        let held = Arc::clone(&handle.sem).try_acquire_owned().unwrap();
        assert_eq!(handle.active(), 1);

        let h2 = handle.clone();
        let task = tokio::spawn(async move {
            let (tok_tx, _r) = stream_channel();
            h2.add_request(
                1,
                "p".into(),
                None,
                GenParams::default(),
                tok_tx,
                Instant::now(),
            )
            .await
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(handle.waiting(), 1, "parked awaiting the permit");

        task.abort();
        let _ = task.await;
        assert_eq!(
            handle.waiting(),
            0,
            "no leaked waiting count on cancellation"
        );
        drop(held);
    }

    /// T1.1: the engine thread's token delivery never blocks. A healthy
    /// client gets its event; a stalled client (full channel) and a
    /// disconnected client (receiver gone) are flagged for routing-out —
    /// without the engine thread waiting on either. If `deliver_or_drop`
    /// ever blocked, the full-channel case below would hang this test.
    #[test]
    fn deliver_or_drop_never_blocks_and_flags_stalled_or_gone_clients() {
        use crate::streaming::stream_channel;

        let sem = Arc::new(Semaphore::new(3));
        let test_req = |tx: TokenSender| ActiveRequest {
            token_tx: tx,
            requested_at: Instant::now(),
            first_token_seen: false,
            permit: Arc::clone(&sem).try_acquire_owned().unwrap(),
            prompt_tail: String::new(),
            max_new_tokens: 0,
            prompt_tokens: 0,
        };

        // Healthy client: delivered, not dropped.
        let (tx, mut rx) = stream_channel();
        assert!(
            !deliver_or_drop(&test_req(tx), 1, StreamEvent::Token("a".to_owned(), 1)),
            "healthy client must be kept"
        );
        assert!(
            matches!(rx.try_recv().unwrap(), StreamEvent::Token(t, _) if t == "a"),
            "healthy client must receive its token"
        );

        // Stalled client: fill the channel to capacity, then deliver once more.
        let (tx_full, _rx_full) = stream_channel();
        while tx_full
            .try_send(StreamEvent::Token("x".to_owned(), 1))
            .is_ok()
        {}
        assert!(
            deliver_or_drop(&test_req(tx_full), 2, StreamEvent::Token("y".to_owned(), 1)),
            "stalled client (full channel) must be dropped, not waited on"
        );

        // Disconnected client: receiver already dropped.
        let (tx_closed, _) = stream_channel();
        assert!(
            deliver_or_drop(
                &test_req(tx_closed),
                3,
                StreamEvent::Token("z".to_owned(), 1)
            ),
            "disconnected client must be dropped"
        );
    }

    /// T1.2: a request whose token callback panicked is routed out with a
    /// clean stream error + terminal `Done`, its capacity slot is released,
    /// and other live routes are untouched.
    #[test]
    fn route_out_panicked_fails_only_the_poisoned_request() {
        use crate::streaming::stream_channel;

        let sem = Arc::new(Semaphore::new(2));
        let mut routes: HashMap<u64, ActiveRequest> = HashMap::new();
        let (tx_bad, mut rx_bad) = stream_channel();
        let (tx_ok, mut rx_ok) = stream_channel();
        for (id, tx) in [(7u64, tx_bad), (3u64, tx_ok)] {
            routes.insert(
                id,
                ActiveRequest {
                    token_tx: tx,
                    requested_at: Instant::now(),
                    first_token_seen: false,
                    permit: Arc::clone(&sem).try_acquire_owned().unwrap(),
                    prompt_tail: String::new(),
                    max_new_tokens: 0,
                    prompt_tokens: 0,
                },
            );
        }
        assert_eq!(sem.available_permits(), 0, "both slots held");

        route_out_panicked(&mut routes, &[7]);

        // The poisoned request: error frame, then terminal Done, route gone,
        // slot released.
        assert!(matches!(
            rx_bad.try_recv().unwrap(),
            StreamEvent::Error(msg) if msg.contains("internal error")
        ));
        assert!(matches!(
            rx_bad.try_recv().unwrap(),
            StreamEvent::Done(FinishReason::Stop)
        ));
        assert!(!routes.contains_key(&7), "poisoned route must be removed");
        assert_eq!(sem.available_permits(), 1, "poisoned slot must be released");

        // The healthy request: no events, route intact, slot still held.
        assert!(rx_ok.try_recv().is_err(), "healthy stream gets no events");
        assert!(routes.contains_key(&3), "healthy route must survive");
    }

    /// A fake `CbEngineOps` whose `step` always reports `pool_exhausted` for
    /// one specific request id — used to test `step_and_route`'s
    /// `observed_prompt_tokens=`/`active_requests=` message embedding
    /// (the project's internal engineering log)
    /// without needing a real KV pool to actually run dry.
    struct PoolExhaustedFakeEngine {
        exhausted_id: u64,
    }

    impl CbEngineOps for PoolExhaustedFakeEngine {
        fn add_request(&mut self, _: u64, _: &str, _: &GenParams) -> anyhow::Result<()> {
            Ok(())
        }
        fn add_request_ids(&mut self, _: u64, _: &[i64], _: &GenParams) -> anyhow::Result<()> {
            Ok(())
        }
        fn drop_request(&mut self, _: u64) {}
        fn has_unfinished(&self) -> bool {
            false
        }
        fn cache_usage_pct(&self) -> Option<f64> {
            Some(1.0)
        }
        fn step(
            &mut self,
            mut on_token: impl FnMut(u64, &str, usize, Option<FinishReason>, bool),
        ) -> anyhow::Result<Vec<u64>> {
            on_token(self.exhausted_id, "", 0, Some(FinishReason::Stop), true);
            Ok(Vec::new())
        }
        fn count_tokens(&self, _: &str) -> anyhow::Result<usize> {
            Ok(0)
        }
        fn encode(&self, _: &str) -> anyhow::Result<Vec<i64>> {
            Ok(vec![])
        }
        fn decode(&self, _: &[i64]) -> anyhow::Result<String> {
            Ok(String::new())
        }
    }

    /// Builds a `routes` map with `other_count` unrelated in-flight requests
    /// plus one request (id 99) about to hit pool exhaustion, runs it through
    /// `step_and_route`, and returns the `StreamEvent::Error` message it
    /// produced for id 99 — so a test can assert on the embedded
    /// `observed_prompt_tokens=`/`active_requests=` fields.
    fn run_pool_exhausted_step(prompt_tokens: usize, other_count: usize) -> String {
        use crate::streaming::stream_channel;

        let sem = Arc::new(Semaphore::new(other_count + 1));
        let mut routes: HashMap<u64, ActiveRequest> = HashMap::new();
        let (tx, mut rx) = stream_channel();
        routes.insert(
            99,
            ActiveRequest {
                token_tx: tx,
                requested_at: Instant::now(),
                first_token_seen: false,
                permit: Arc::clone(&sem).try_acquire_owned().unwrap(),
                prompt_tail: String::new(),
                max_new_tokens: 1400,
                prompt_tokens,
            },
        );
        for i in 0..other_count {
            let (other_tx, _other_rx) = stream_channel();
            routes.insert(
                1000 + i as u64,
                ActiveRequest {
                    token_tx: other_tx,
                    requested_at: Instant::now(),
                    first_token_seen: false,
                    permit: Arc::clone(&sem).try_acquire_owned().unwrap(),
                    prompt_tail: String::new(),
                    max_new_tokens: 0,
                    prompt_tokens: 0,
                },
            );
        }

        let mut engine = PoolExhaustedFakeEngine { exhausted_id: 99 };
        let metrics = HotMetrics::new(ModelKind::TextGen, "fake-model", "CPU");
        let mut tps_ema = 0.0;
        step_and_route(&mut engine, &mut routes, &metrics, &mut tps_ema);

        match rx.try_recv().unwrap() {
            StreamEvent::Error(msg) => msg,
            other => panic!("expected an Error event, got {other:?}"),
        }
    }

    /// Solo failure (no other request in flight against this engine): the
    /// message must embed this request's own exact prompt token count and
    /// `active_requests=1` — the signal `classify_inference_error` uses to
    /// tell a real static-formula miss apart from ordinary congestion.
    #[test]
    fn pool_exhausted_message_embeds_observed_tokens_and_solo_active_requests() {
        let msg = run_pool_exhausted_step(37_166, 0);
        assert!(
            msg.contains("observed_prompt_tokens=37166"),
            "message must embed the exact prompt token count: {msg}"
        );
        assert!(
            msg.contains("active_requests=1"),
            "solo failure must report active_requests=1: {msg}"
        );
    }

    /// Congested failure (4 other requests concurrently in flight against
    /// the same engine): `active_requests` must count all of them, including
    /// the failing one itself — this is the signal that keeps a genuinely
    /// busy multi-tenant model from being evicted over an ordinary capacity
    /// blip.
    #[test]
    fn pool_exhausted_message_reports_active_requests_including_concurrent_ones() {
        let msg = run_pool_exhausted_step(12_000, 4);
        assert!(
            msg.contains("observed_prompt_tokens=12000"),
            "message must embed the exact prompt token count: {msg}"
        );
        assert!(
            msg.contains("active_requests=5"),
            "must count the failing request plus its 4 concurrent siblings: {msg}"
        );
    }

    /// T1.3: when the whole batch is failed at once (engine-loop exit or step
    /// failure), every live route gets a terminal `Error` + `Done(Stop)` pair,
    /// the route table empties, and all capacity permits are released —
    /// no stream is ever left hanging without a terminal event.
    #[test]
    fn fail_all_routes_terminates_every_stream_and_releases_permits() {
        use crate::streaming::stream_channel;

        let sem = Arc::new(Semaphore::new(2));
        let mut routes: HashMap<u64, ActiveRequest> = HashMap::new();
        let (tx_a, mut rx_a) = stream_channel();
        let (tx_b, mut rx_b) = stream_channel();
        for (id, tx) in [(1u64, tx_a), (2u64, tx_b)] {
            routes.insert(
                id,
                ActiveRequest {
                    token_tx: tx,
                    requested_at: Instant::now(),
                    first_token_seen: false,
                    permit: Arc::clone(&sem).try_acquire_owned().unwrap(),
                    prompt_tail: String::new(),
                    max_new_tokens: 0,
                    prompt_tokens: 0,
                },
            );
        }
        assert_eq!(sem.available_permits(), 0, "both slots held");

        fail_all_routes(&mut routes, "engine shutting down");

        for rx in [&mut rx_a, &mut rx_b] {
            assert!(matches!(
                rx.try_recv().unwrap(),
                StreamEvent::Error(msg) if msg.contains("shutting down")
            ));
            assert!(matches!(
                rx.try_recv().unwrap(),
                StreamEvent::Done(FinishReason::Stop)
            ));
        }
        assert!(routes.is_empty(), "all routes must be removed");
        assert_eq!(sem.available_permits(), 2, "all slots must be released");
    }

    /// The plain CB engine is the one `ManagedEngine` implementer that can
    /// actually query real KV-pool occupancy — confirm it overrides the
    /// trait's safe-by-default `false` (see `ManagedEngine::cache_usage_supported`'s
    /// doc comment; the project's internal engineering log).
    #[test]
    fn cb_engine_reports_cache_usage_as_supported() {
        let (tx, _rx) = mpsc::channel::<EngineCommand>(4);
        let handle = EngineHandle::from_sender(tx);
        assert!(ManagedEngine::cache_usage_supported(&handle));
    }

    /// `tokenize` sends a `Tokenize` command and returns the engine's reply.
    /// A fake "engine" task stands in for the real engine thread, proving the
    /// command + `oneshot` round-trip without a GPU.
    #[tokio::test]
    async fn tokenize_round_trips_through_command_channel() {
        let (tx, mut rx) = mpsc::channel::<EngineCommand>(4);
        let handle = EngineHandle::from_sender(tx);

        tokio::spawn(async move {
            match rx.recv().await {
                Some(EngineCommand::Tokenize { text, reply }) => {
                    assert_eq!(text, "hello");
                    let _ = reply.send(Ok(vec![1, 2, 3]));
                }
                other => panic!("expected Tokenize, got {:?}", other.is_some()),
            }
        });

        let ids = handle.tokenize("hello".to_owned()).await.unwrap();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    /// `detokenize` sends a `Detokenize` command and returns the engine's reply.
    #[tokio::test]
    async fn detokenize_round_trips_through_command_channel() {
        let (tx, mut rx) = mpsc::channel::<EngineCommand>(4);
        let handle = EngineHandle::from_sender(tx);

        tokio::spawn(async move {
            match rx.recv().await {
                Some(EngineCommand::Detokenize { ids, reply }) => {
                    assert_eq!(ids, vec![9906, 1879]);
                    let _ = reply.send(Ok("Hello world".to_owned()));
                }
                other => panic!("expected Detokenize, got {:?}", other.is_some()),
            }
        });

        let text = handle.detokenize(vec![9906, 1879]).await.unwrap();
        assert_eq!(text, "Hello world");
    }

    /// A tokenizer error on the engine side surfaces as an `Err` to the caller
    /// (the error string is carried back over the `oneshot`).
    #[tokio::test]
    async fn tokenize_propagates_engine_error() {
        let (tx, mut rx) = mpsc::channel::<EngineCommand>(4);
        let handle = EngineHandle::from_sender(tx);

        tokio::spawn(async move {
            if let Some(EngineCommand::Tokenize { reply, .. }) = rx.recv().await {
                let _ = reply.send(Err("bad tokenizer".to_owned()));
            }
        });

        let err = handle.tokenize("x".to_owned()).await.unwrap_err();
        assert!(
            err.to_string().contains("bad tokenizer"),
            "engine error must reach the caller: {err}"
        );
    }

    /// If the engine thread is gone (command channel closed), `tokenize` errors
    /// rather than hanging.
    #[tokio::test]
    async fn tokenize_errors_when_engine_dead() {
        let (tx, rx) = mpsc::channel::<EngineCommand>(1);
        drop(rx); // engine thread gone
        let handle = EngineHandle::from_sender(tx);
        assert!(
            handle.tokenize("x".to_owned()).await.is_err(),
            "tokenize must error when the engine channel is closed"
        );
    }

    /// If the engine accepts the command but drops the reply sender without
    /// answering, `detokenize` errors rather than hanging forever.
    #[tokio::test]
    async fn detokenize_errors_when_reply_dropped() {
        let (tx, mut rx) = mpsc::channel::<EngineCommand>(1);
        let handle = EngineHandle::from_sender(tx);

        tokio::spawn(async move {
            // Receive the command, then drop it (and its reply sender) unanswered.
            let _ = rx.recv().await;
        });

        assert!(
            handle.detokenize(vec![1]).await.is_err(),
            "detokenize must error when the reply sender is dropped"
        );
    }
}
