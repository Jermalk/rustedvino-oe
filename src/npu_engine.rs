// ============================================================
// src/npu_engine.rs — NPU LLM engine thread (lnl-a)
// ============================================================
// Same architecture as vlm_engine.rs but for the static LLMPipeline:
//   - One dedicated OS thread per NPU LLM model
//   - Commands arrive via an mpsc channel (capacity 8)
//   - Engine thread processes one Generate at a time (LLMPipeline
//     is single-threaded and blocking — no CB-style parallelism)
//   - NPU requires channel-wise (cw) int4 IR; group-wise fails compile
//     (proven empirically on a Lunar Lake NPU box)
//   - Token stream goes to the same StreamEvent / TokenSender as the
//     CB path, so the chat handler and SSE layer are reused unchanged
//   - Admission is a real `Semaphore` gate (mirrors `cb_engine`/`vlm_engine`,
//     the project's internal engineering log's recipe): a blocking channel send
//     with no timeout previously let a full queue park the caller forever
//     instead of returning a 429.
//   - NPU compiles a FIXED-SHAPE graph ahead of time (unlike GPU/CPU's
//     dynamic-shape kernels), so OpenVINO GenAI's static LLMPipeline takes a
//     compile-time `MAX_PROMPT_LEN` (default 1024 tokens) — a request whose
//     templated prompt exceeds it hard-fails. Configurable via the per-model
//     `max_prompt_len` config field (`ModelPolicy`), threaded through
//     `OvPipeline::new` → `ov_pipeline_create`'s AnyMap (the project's internal engineering log,
//     CLOSED 2026-07-16) — raising it grows the compiled KV-cache and compile
//     time; it is a build-time tradeoff, not a per-request setting. The chat
//     handler's `gate_npu_prompt` pre-flights the compiled limit (carried on
//     the handle as `NpuShape`, resolved at load) via `NpuCommand::
//     CountTokens`, independent of `generate()`, so an over-limit prompt gets
//     a clean 400 instead of OpenVINO's raw C++ exception text.
//   - `OvPipeline::new` receives `Config::ov_cache_dir` and `ov_pipeline_create`
//     sets it as `CACHE_DIR` (since 2026-09-28): one weightless compiled blob
//     per model, ~6 s cached load vs 30.7 s cold on Lunar Lake, identical
//     output (the project's internal engineering log). Without a cache dir,
//     the Level Zero driver's own cache (`~/.cache/ze_intel_npu_cache/`)
//     still keeps repeat loads at ~8–12 s.
//     Changing `max_prompt_len` / `min_response_len` between loads compiles
//     a new shape. OpenVINO keys the cached blob on it (verified live), so
//     every combination tried leaves its own ~0.66 GB blob behind.
// ============================================================

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::mpsc::{Sender, error::TrySendError};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

use crate::cb_engine::SubmitResult;
use crate::metrics::{HotMetrics, Modality};
use crate::model_manager::{ManagedEngine, ModelKind};
use crate::ov_cb::{FinishReason, GenParams};
use crate::ov_pipeline::{NpuGenOutcome, OvPipeline};
use crate::streaming::{StreamEvent, TokenSender};

/// NPU admission-gate size (also the reported `max_concurrency`): up to this
/// many requests may be queued/generating before the gate returns
/// [`SubmitResult::AtCapacity`]. The raw channel buffer is set to the same
/// value — a permit is always acquired before a command is sent, so the
/// channel itself never needs to absorb more than `NPU_CHANNEL_CAP` commands.
pub(crate) const NPU_CHANNEL_CAP: usize = 8;

/// `OpenVINO`'s default NPU `MIN_RESPONSE_LEN`: output room the static KV
/// cache reserves on top of `MAX_PROMPT_LEN`. Measured on Lunar Lake: an
/// 863-token prompt stopped after 290 generated tokens (863 + 290 = 1024 +
/// 128 + 1), and `MIN_RESPONSE_LEN = 512` lifted the cap.
pub(crate) const NPU_DEFAULT_MIN_RESPONSE_LEN: usize = 128;

/// Total tokens (prompt + generated) the compiled NPU KV cache holds:
/// `max_prompt_len + min_response_len`, each falling back to `OpenVINO`'s
/// default. Generation that reaches it ends without an EOS, so
/// [`run_generate`] reports it as `Length`, not `Stop`.
///
/// No alignment: the NPU does not round either value up (e.g. to 64).
/// Measured on Lunar Lake: `1000 / 100` filled at `1100 + 1` total tokens,
/// the `1024 / 128` default at `1152 + 1`.
#[must_use]
pub(crate) fn npu_kv_capacity(max_prompt_len: Option<u32>, min_response_len: Option<u32>) -> usize {
    let response = min_response_len.map_or(NPU_DEFAULT_MIN_RESPONSE_LEN, |v| v as usize);
    npu_effective_max_prompt_len(max_prompt_len) + response
}

/// The effective NPU `MAX_PROMPT_LEN`: the configured override, or
/// `OpenVINO`'s default when unset.
fn npu_effective_max_prompt_len(max_prompt_len: Option<u32>) -> usize {
    max_prompt_len.map_or(crate::handlers::chat::NPU_DEFAULT_MAX_PROMPT_LEN, |v| {
        v as usize
    })
}

/// The fixed shape a resident NPU LLM was compiled with.
///
/// Resolved once, at load, from the same `max_prompt_len` / `min_response_len`
/// values handed to `OvPipeline::new`, and carried on the [`NpuHandle`] so the
/// chat handler's prompt gate checks the ceiling the engine *actually has* —
/// not the registry's configured value, which a PATCH or config reload can
/// change without recompiling the resident graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NpuShape {
    /// Longest prompt (in tokens) the compiled graph accepts.
    pub max_prompt_len: usize,
    /// Prompt + generated tokens the static KV cache holds; see [`npu_kv_capacity`].
    pub kv_capacity: usize,
}

impl NpuShape {
    /// Resolve the compiled shape from the load-time overrides (`None` =
    /// `OpenVINO`'s default for that dimension).
    #[must_use]
    pub fn resolve(max_prompt_len: Option<u32>, min_response_len: Option<u32>) -> Self {
        Self {
            max_prompt_len: npu_effective_max_prompt_len(max_prompt_len),
            kv_capacity: npu_kv_capacity(max_prompt_len, min_response_len),
        }
    }
}

// ── Commands ─────────────────────────────────────────────────────────────────

pub(crate) enum NpuCommand {
    /// Run a single-stream NPU LLM generation.
    Generate {
        /// Pre-rendered prompt string (chat template already applied in Rust;
        /// the bridge disables `OpenVINO`'s own templating).
        prompt: String,
        /// Sampling / stop / structured-output settings and the token budget —
        /// the same [`GenParams`] the CB and VLM engines take. Boxed to keep
        /// the command small (mirrors `vlm_engine::VlmCommand`).
        params: Box<GenParams>,
        /// Token stream sink — receives `Token`, `Done`, and `Error` events.
        token_tx: TokenSender,
        /// Wall-clock start (handler entry) for TTFT / duration / tok/s metrics.
        started_at: Instant,
        /// Admission-gate permit, held until the generation call returns. Dropped
        /// by `finish_npu_generation` before the terminal `Done` / `Error` is
        /// sent (0.9.1 C29), releasing the slot for the next waiting caller.
        /// See [`NpuHandle::generate`].
        permit: OwnedSemaphorePermit,
    },
    /// Count `text`'s tokens via the pipeline's tokenizer, independent of
    /// generation — used to pre-flight a prompt's length before submitting
    /// it to [`Generate`](Self::Generate). Not admission-gated (mirrors
    /// `vlm_engine::VlmCommand::CountTokens`): cheap and shouldn't compete for
    /// the same generation slot.
    CountTokens {
        text: String,
        reply: oneshot::Sender<Result<usize, String>>,
    },
}

// ── Handle ───────────────────────────────────────────────────────────────────

/// Cloneable submit handle for the NPU engine thread.
///
/// Requests serialise in the command channel; the engine thread processes one
/// generation at a time. The admission `Semaphore` tracks accepted-but-
/// unfinished requests for `/metrics` — a permit is held from before the
/// command is sent until the engine thread finishes the generation.
#[derive(Clone, Debug)]
pub struct NpuHandle {
    tx: Sender<NpuCommand>,
    model_id: Arc<str>,
    /// Admission-gate permits — `cap` of them. Mirrors `cb_engine::EngineHandle`'s
    /// `sem` field exactly.
    sem: Arc<Semaphore>,
    /// Channel capacity = the semaphore's permit count = [`NPU_CHANNEL_CAP`].
    cap: usize,
    /// Maximum time [`generate`](Self::generate) waits for a permit before
    /// returning [`SubmitResult::AtCapacity`] (→ HTTP 429).
    queue_timeout: Duration,
    /// Callers currently parked in [`generate`](Self::generate) awaiting an
    /// admission permit — the engine's honest `rustedvino_requests_waiting`.
    waiting: Arc<AtomicUsize>,
    /// The shape the resident pipeline was compiled with.
    shape: NpuShape,
    /// Set (`SeqCst`) by `begin_evict` before it reads `active()`; checked by ``generate`` *after*
    /// it holds an admission permit. Shared by every clone of this handle, so a submit that
    /// slips in after the eviction started is refused (retryable) instead of being drained and
    /// aborted; a reload builds a new engine with a clear flag. See
    /// `cb_engine::EngineHandle::closing`. D3-S1.D2; nothing sets it yet.
    closing: Arc<AtomicBool>,
    /// This engine's liveness (0.9.1 S1, [`crate::stall_watch`]): the engine thread reports its
    /// calls and progress to it, its watcher declares a stall. Shared by every clone and the
    /// engine thread; a reload builds a new engine with a fresh one.
    liveness: crate::stall_watch::EngineLiveness,
}

impl NpuHandle {
    /// This engine instance's liveness (0.9.1 S1 stall watchdog).
    #[must_use]
    pub fn liveness(&self) -> &crate::stall_watch::EngineLiveness {
        &self.liveness
    }

    /// The model ID of the loaded NPU LLM.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Longest prompt (in tokens) the resident pipeline was compiled to
    /// accept — what the chat handler's prompt gate enforces.
    #[must_use]
    pub fn max_prompt_len(&self) -> usize {
        self.shape.max_prompt_len
    }

    /// The shape the resident pipeline was compiled with.
    #[must_use]
    pub fn shape(&self) -> NpuShape {
        self.shape
    }

    /// Test seam: replace the handle's compiled shape.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_shape(mut self, shape: NpuShape) -> Self {
        self.shape = shape;
        self
    }

    /// Test seam: build a handle around an existing channel with no GPU.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn from_sender(tx: Sender<NpuCommand>, model_id: &str) -> Self {
        let cap = 1000; // generous test cap — never hits the admission gate
        Self {
            tx,
            model_id: Arc::from(model_id),
            sem: Arc::new(Semaphore::new(cap)),
            cap,
            queue_timeout: Duration::from_secs(30),
            waiting: Arc::new(AtomicUsize::new(0)),
            shape: NpuShape::resolve(None, None),
            closing: Arc::new(AtomicBool::new(false)),
            liveness: crate::stall_watch::EngineLiveness::new(),
        }
    }

    /// Creates a test handle with an explicit cap and timeout. Used by
    /// admission-gate tests that need to observe `AtCapacity` behaviour.
    #[cfg(test)]
    pub(crate) fn from_sender_with_cap(
        tx: Sender<NpuCommand>,
        model_id: &str,
        cap: usize,
        queue_timeout_ms: u64,
    ) -> Self {
        Self {
            tx,
            model_id: Arc::from(model_id),
            sem: Arc::new(Semaphore::new(cap)),
            cap,
            queue_timeout: Duration::from_millis(queue_timeout_ms),
            waiting: Arc::new(AtomicUsize::new(0)),
            shape: NpuShape::resolve(None, None),
            closing: Arc::new(AtomicBool::new(false)),
            liveness: crate::stall_watch::EngineLiveness::new(),
        }
    }

    /// Enqueue an NPU generation request.
    ///
    /// Waits up to `queue_timeout` for an admission permit before returning
    /// [`SubmitResult::AtCapacity`] (→ HTTP 429) — mirrors
    /// `cb_engine::EngineHandle::add_request`'s admission gate exactly.
    ///
    /// Returns [`SubmitResult::EngineDead`] if the command channel is closed
    /// (engine thread exited) — caller should respond HTTP 503.
    pub(crate) async fn generate(
        &self,
        prompt: String,
        params: GenParams,
        token_tx: TokenSender,
        started_at: Instant,
    ) -> SubmitResult {
        // `InFlightGuard` (not just an increment) so a cancelled caller — an
        // HTTP client disconnecting while parked at the gate — still
        // decrements: its `Drop` runs even if this `.await` is torn down
        // mid-poll, unlike a bare `fetch_sub` placed after the await.
        let permit_result = {
            let _waiting_guard = crate::in_flight::InFlightGuard::new(&self.waiting);
            tokio::select! {
                biased;
                r = tokio::time::timeout(
                    self.queue_timeout,
                    Arc::clone(&self.sem).acquire_owned(),
                ) => r,
                // 0.9.1 S1: the engine stalled while this caller waited (as the CB engine's
                // `add_request`): the verdict pair goes on the caller's own channel.
                () = self.liveness.dead() => {
                    crate::stall_watch::send_stalled_pair_now(&token_tx);
                    return SubmitResult::Submitted;
                }
            }
        };
        let permit = match permit_result {
            Ok(Ok(p)) => p,
            Ok(Err(_)) => return SubmitResult::EngineDead, // semaphore closed (shouldn't happen)
            Err(_) => return SubmitResult::AtCapacity,     // timed out waiting for a permit
        };

        // The eviction may have started while this caller waited for (or just took) its permit.
        // Checked AFTER holding it: `begin_evict` sets `closing` before reading `active()`, so
        // either it counted this permit (and drains it) or this check sees the flag.
        if self.closing.load(Ordering::SeqCst) {
            drop(permit);
            return SubmitResult::Closing;
        }

        // Permit acquired. Send to the engine — if the channel is closed, the
        // permit (moved into the command) drops here, automatically releasing
        // the slot back to the semaphore.
        let sent = self
            .tx
            .send(NpuCommand::Generate {
                prompt,
                params: Box::new(params),
                token_tx,
                started_at,
                permit,
            })
            .await;
        if sent.is_err() {
            return SubmitResult::EngineDead;
        }
        SubmitResult::Submitted
    }

    /// Count `prompt`'s tokens via the NPU pipeline's tokenizer.
    ///
    /// Not admission-gated (mirrors `vlm_engine::VlmHandle::count_tokens`) —
    /// this is a cheap tokenizer call, not a generation, so it does not
    /// compete for the same single-stream slot [`generate`](Self::generate)
    /// does.
    ///
    /// # Errors
    /// Returns an error if the engine thread is gone or the tokenizer fails.
    pub(crate) async fn count_tokens(&self, text: String) -> anyhow::Result<usize> {
        let (reply, reply_rx) = oneshot::channel();
        self.liveness
            .unless_dead(self.tx.send(NpuCommand::CountTokens { text, reply }))
            .await?
            .map_err(|_| anyhow::anyhow!("NPU engine unavailable"))?;
        // 0.9.1 S1: a stuck engine thread never replies; the verdict releases the caller.
        self.liveness
            .unless_dead(reply_rx)
            .await?
            .map_err(|_| anyhow::anyhow!("NPU engine dropped tokenize reply"))?
            .map_err(|e| anyhow::anyhow!(e))
    }
}

/// The NPU LLM is `ModelKind::TextGen` — same routing as the CB engine,
/// different execution handle. The chat handler distinguishes them via
/// `EngineHandleKind::NpuTextGen` vs `EngineHandleKind::TextGen`.
impl ManagedEngine for NpuHandle {
    fn kind(&self) -> ModelKind {
        ModelKind::TextGen
    }

    fn set_closing(&self, closing: bool) {
        self.closing.store(closing, Ordering::SeqCst);
    }

    /// Accepted-but-unfinished requests (queued + the one generating) —
    /// derived from the semaphore, real occupancy rather than an
    /// approximation. See `vlm_engine::VlmHandle::active`'s doc comment for
    /// why this is honest even though `OvPipeline::generate` itself is
    /// strictly single-stream.
    fn active(&self) -> usize {
        self.cap.saturating_sub(self.sem.available_permits())
    }

    fn max_concurrency(&self) -> usize {
        self.cap
    }

    /// Callers currently parked in [`generate`](Self::generate) awaiting an
    /// admission permit. Mirrors `cb_engine::EngineHandle::waiting` exactly.
    fn waiting(&self) -> usize {
        self.waiting.load(Ordering::Relaxed)
    }
}

// ── Engine thread ─────────────────────────────────────────────────────────────

/// Spawn a dedicated OS thread owning `pipeline` and return an [`NpuHandle`].
///
/// The pipeline is the sole owner of the `LLMPipeline` C++ object; requests
/// serialise through the command channel, so no locking is needed. The thread
/// exits cleanly when all handle clones are dropped (channel closes).
///
/// # Errors
/// Propagates thread-spawn errors (rare OS resource exhaustion).
pub fn spawn_npu_engine(
    mut pipeline: OvPipeline,
    model_id: &str,
    device: &str,
    queue_timeout: Duration,
    shape: NpuShape,
) -> anyhow::Result<(NpuHandle, std::thread::JoinHandle<()>)> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<NpuCommand>(NPU_CHANNEL_CAP);
    let sem = Arc::new(Semaphore::new(NPU_CHANNEL_CAP));
    let model_id_arc: Arc<str> = Arc::from(model_id);
    let model_id_owned = model_id.to_owned();
    let device = device.to_owned();

    let metrics = HotMetrics::new(ModelKind::TextGen, model_id, &device);
    // 0.9.1 S1: shared by the handle and the engine thread, which reports its calls to it.
    let liveness = crate::stall_watch::EngineLiveness::new();
    let liveness_engine = liveness.clone();

    let thread = std::thread::Builder::new()
        .name(format!("npu-engine-{model_id_owned}"))
        .spawn(move || {
            tracing::info!(
                model_id = model_id_owned,
                device,
                "NPU engine thread started"
            );
            // 0.9.1 S1: however this thread exits (shutdown, panic), it is not left busy.
            let _idle_on_exit = liveness_engine.idle_on_drop();
            while let Some(cmd) = rx.blocking_recv() {
                match cmd {
                    NpuCommand::Generate {
                        prompt,
                        params,
                        token_tx,
                        started_at,
                        permit,
                    } => {
                        run_generate(
                            &mut pipeline,
                            &model_id_owned,
                            &prompt,
                            &params,
                            &token_tx,
                            &metrics,
                            started_at,
                            shape.kv_capacity,
                            &liveness_engine,
                            // Balance the handle's admission acquire, whatever the
                            // generation outcome — released by `run_generate` the
                            // moment the generation call returns, BEFORE the
                            // terminal event is sent (0.9.1 C29).
                            permit,
                        );
                    }
                    NpuCommand::CountTokens { text, reply } => {
                        let _ = reply.send(pipeline.count_tokens(&text).map_err(|e| e.to_string()));
                    }
                }
            }
            tracing::info!(
                model_id = model_id_owned,
                "NPU engine thread exiting — all handles dropped"
            );
        })?;

    Ok((
        NpuHandle {
            tx,
            model_id: model_id_arc,
            sem,
            cap: NPU_CHANNEL_CAP,
            queue_timeout,
            waiting: Arc::new(AtomicUsize::new(0)),
            shape,
            closing: Arc::new(AtomicBool::new(false)),
            liveness,
        },
        thread,
    ))
}

/// Marker text: a mock NPU request whose prompt contains it fails with this
/// text as its `Error` (test-only; no routing test can hit it by accident).
#[cfg(test)]
pub(crate) const MOCK_NPU_GENERATE_ERROR: &str = "__mock_npu_generate_error__";

/// `(model id, params)` of every `Generate` the mock NPU engines received (0.9.1
/// thinking-control handler tests). Process-wide and never cleared: it lives for the whole test
/// process, so every test that reads it must use a model id no other test uses.
#[cfg(test)]
static MOCK_SEEN_PARAMS: std::sync::Mutex<Vec<(String, GenParams)>> =
    std::sync::Mutex::new(Vec::new());

/// The mock NPU `Generate` params received so far for `model_id`.
#[cfg(test)]
pub(crate) fn mock_seen_params(model_id: &str) -> Vec<GenParams> {
    MOCK_SEEN_PARAMS
        .lock()
        .map(|v| {
            v.iter()
                .filter(|s| s.0 == model_id)
                .map(|s| s.1.clone())
                .collect()
        })
        .unwrap_or_default()
}

/// GPU-free mock NPU engine — test analogue of [`spawn_npu_engine`].
///
/// Answers each `Generate` with one token + `Done(Stop)` and drops the
/// permit per command, so `active()` is honest in routing tests. A prompt
/// containing [`MOCK_NPU_GENERATE_ERROR`] gets the error terminal instead
/// (`Error` + `Done(Stop)`, no token). Both orders match the real engine: the
/// permit is released before the terminal event (0.9.1 C29).
#[cfg(test)]
pub(crate) fn spawn_mock_npu(model_id: &str) -> (NpuHandle, std::thread::JoinHandle<()>) {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<NpuCommand>(NPU_CHANNEL_CAP);
    let sem = Arc::new(Semaphore::new(NPU_CHANNEL_CAP));
    let liveness = crate::stall_watch::EngineLiveness::new();
    let liveness_engine = liveness.clone();
    let mock_id = model_id.to_owned();

    let thread = std::thread::spawn(move || {
        while let Some(cmd) = rx.blocking_recv() {
            match cmd {
                NpuCommand::Generate {
                    prompt,
                    params,
                    token_tx,
                    permit,
                    ..
                } => {
                    if let Ok(mut seen) = MOCK_SEEN_PARAMS.lock() {
                        seen.push((mock_id.clone(), *params));
                    }
                    // 0.9.1 S1: a marked request stalls (or answers) while it holds its permit.
                    if crate::stall_watch::mock::run(&prompt, &token_tx, &liveness_engine) {
                        drop(permit);
                        continue;
                    }
                    if prompt.contains(MOCK_NPU_GENERATE_ERROR) {
                        drop(permit);
                        let _ = token_tx
                            .try_send(StreamEvent::Error(MOCK_NPU_GENERATE_ERROR.to_owned()));
                        let _ = token_tx.try_send(StreamEvent::Done(FinishReason::Stop));
                        continue;
                    }
                    let _ = token_tx.blocking_send(StreamEvent::Token("mock-npu".to_owned(), 1));
                    drop(permit);
                    let _ = token_tx.blocking_send(StreamEvent::Done(FinishReason::Stop));
                }
                NpuCommand::CountTokens { text, reply } => {
                    // Mock tokenizer: one "token" per whitespace-separated word,
                    // good enough for gate_npu_prompt's threshold tests.
                    let _ = reply.send(Ok(text.split_whitespace().count()));
                }
            }
        }
    });

    (
        NpuHandle {
            tx,
            model_id: Arc::from(model_id),
            sem,
            cap: NPU_CHANNEL_CAP,
            queue_timeout: Duration::from_secs(30),
            waiting: Arc::new(AtomicUsize::new(0)),
            shape: NpuShape::resolve(None, None),
            closing: Arc::new(AtomicBool::new(false)),
            liveness,
        },
        thread,
    )
}

/// Drive one generation request on the engine thread.
///
/// `slot` (the admission permit) is handed to [`finish_npu_generation`] right
/// after the generation call returns, which drops it before `CompletionTokens`,
/// the metrics and the terminal `Done` / `Error` — so a client woken by `Done`
/// never finds its own finished request in `active()` (0.9.1 C29, the streaming
/// form of C2). The token closure does not capture it.
#[allow(clippy::too_many_arguments)]
fn run_generate<S>(
    pipeline: &mut OvPipeline,
    model_id: &str,
    prompt: &str,
    params: &GenParams,
    token_tx: &TokenSender,
    metrics: &HotMetrics,
    started_at: Instant,
    kv_capacity: usize,
    liveness: &crate::stall_watch::EngineLiveness,
    slot: S,
) {
    metrics.request_accepted(Modality::Text);
    // 0.9.1 V1-usage (P2): the engine's start of this request (the command queue is behind it).
    let engine_started_at = Instant::now();
    // 0.9.1 S1: the call starts here, before the first OpenVINO call (the token count below).
    liveness.begin();

    // Same tokenizer call `NpuCommand::CountTokens`/`gate_npu_prompt` already
    // use, just made from inside this generation instead of a separate
    // round-trip — the prior "no prompt token count available from
    // LLMPipeline" comment here was stale: `OvPipeline::count_tokens` has
    // been callable on this exact pipeline the whole time. try_send as the
    // very first event, same convention as `cb_engine.rs`'s `PromptTokens`
    // send; fail-open (log and continue) rather than aborting the request,
    // matching `gate_npu_prompt`'s own `Ok(_) | Err(_) => Ok(())` policy.
    let counted_prompt_tokens = match pipeline.count_tokens(prompt) {
        Ok(n) => {
            let _ = token_tx.try_send(StreamEvent::PromptTokens(n));
            Some(n)
        }
        Err(e) => {
            tracing::warn!(model_id, error = %e, "NPU prompt token count failed");
            None
        }
    };

    // Chunk accounting and the budget: see `NpuStream`.
    let mut stream = NpuStream::new(model_id, started_at, params);
    let result = pipeline.generate(prompt, params, |text| {
        // 0.9.1 S1: every streamer callback is progress (it fires on decoded text only, so a long
        // run of tokens that decode to nothing reads as silence — a documented limit).
        liveness.progress();
        stream.on_chunk(text, token_tx, metrics)
    });
    let NpuStream {
        first_chunk_at,
        output,
        acct,
        stopped_by_client,
        budget,
        ..
    } = stream;
    liveness.end();
    finish_npu_generation(
        slot,
        result,
        &NpuGenTail {
            model_id,
            started_at,
            acct,
            stopped_by_client,
            counted_prompt_tokens,
            kv_capacity,
            prefill: crate::metrics::prefill_interval(
                started_at,
                engine_started_at,
                first_chunk_at,
            ),
            output: &output,
            reasoning: params.reasoning,
            budget: crate::thinking_budget::BudgetOutcome::of(budget.as_ref(), stopped_by_client),
        },
        token_tx,
        metrics,
        |text: &str| pipeline.count_tokens(text),
    );
}

/// One NPU generation's streamer state; [`Self::on_chunk`] is the streamer callback's body (a
/// method so a test drives the real callback).
struct NpuStream<'a> {
    model_id: &'a str,
    started_at: Instant,
    first_chunk_at: Option<Instant>,
    /// The whole output, for the reasoning recount after the generation (rule 2).
    output: String,
    /// The callback fires once per decoded text CHUNK, and a chunk can carry
    /// several tokens (the streamer holds back incomplete pieces), so counting
    /// chunks undercounts (seen live: 192 for a 200-token generation). Each
    /// chunk goes out immediately; the pipeline's own total follows as
    /// `CompletionTokens` just before `Done`, and consumers use it in place of
    /// their running sum.
    acct: ChunkAccounting,
    stopped_by_client: bool,
    /// 0.9.1 strict thinking budget: one per delivered chunk against the budget scaled to chunks
    /// (`chunk_budget` with `NPU_TOKENS_PER_CHUNK_PERCENT`, 1.04 tokens a chunk, from the note
    /// on `acct` above), like the VLM's: the stop lands near the budget, approximately.
    budget: Option<crate::thinking_budget::BudgetTracker>,
}

impl<'a> NpuStream<'a> {
    fn new(model_id: &'a str, started_at: Instant, params: &GenParams) -> Self {
        Self {
            model_id,
            started_at,
            first_chunk_at: None,
            output: String::new(),
            acct: ChunkAccounting::default(),
            stopped_by_client: false,
            budget: crate::thinking_budget::chunk_tracker_for(
                params.reasoning,
                params.thinking_budget,
                crate::thinking_budget::NPU_TOKENS_PER_CHUNK_PERCENT,
            ),
        }
    }

    /// The streamer callback for one decoded chunk.
    /// NOTE: `OvPipeline::generate` convention is true=continue, false=stop.
    /// This is the opposite of the VLM engine callback — do not copy `vlm_engine.rs` here.
    fn on_chunk(&mut self, text: &str, token_tx: &TokenSender, metrics: &HotMetrics) -> bool {
        if text.is_empty() {
            return true; // consumers skip empty deltas; nothing to send or count
        }
        // try_send, never blocking: same backpressure policy as the VLM and CB
        // engines — a stalled client must not wedge this engine thread and park
        // every queued NPU request behind it.
        match token_tx.try_send(StreamEvent::Token(text.to_owned(), 1)) {
            Ok(()) => {
                self.output.push_str(text);
                if self.acct.record_sent() {
                    // TTFT = the first token the client can actually receive.
                    self.first_chunk_at = Some(Instant::now());
                    metrics.record_ttft(Modality::Text, self.started_at.elapsed().as_secs_f64());
                }
                metrics.token_generated(1);
                // `false` stops the pipeline (this engine's convention).
                let Some(b) = self.budget.as_mut() else {
                    return true;
                };
                let already = b.stopped();
                if b.push(text, 1) == crate::thinking_budget::BudgetStep::Stop {
                    if !already {
                        tracing::debug!(
                            model_id = self.model_id,
                            reasoning_chunks = b.counted(),
                            "NPU thinking budget spent — stopping the request \
                             (finish_reason length)"
                        );
                    }
                    return false;
                }
                true
            }
            Err(TrySendError::Full(_)) => {
                tracing::warn!(
                    "NPU client not keeping up (token channel full) — stopping generation"
                );
                self.stopped_by_client = true;
                false
            }
            Err(TrySendError::Closed(_)) => {
                self.stopped_by_client = true;
                false
            }
        }
    }
}

/// What [`finish_npu_generation`] needs from the generation besides its result.
struct NpuGenTail<'a> {
    model_id: &'a str,
    started_at: Instant,
    /// Chunks delivered to the client during the generation.
    acct: ChunkAccounting,
    /// The token closure stopped the generation (client gone / not keeping up).
    stopped_by_client: bool,
    /// The tokenizer's prompt count, the fallback for the KV-full rule.
    counted_prompt_tokens: Option<usize>,
    kv_capacity: usize,
    /// Engine start of the request to its first delivered chunk (0.9.1 V1-usage, P2).
    prefill: Option<std::time::Duration>,
    /// The whole delivered output (the reasoning recount's input).
    output: &'a str,
    /// The request's reasoning split.
    reasoning: crate::prompt_builder::ReasoningSplit,
    /// Where the request's thinking budget ended (0.9.1): a budget stop (or `length` inside the
    /// reasoning) finishes as [`FinishReason::ThinkingBudget`], a finished request.
    budget: crate::thinking_budget::BudgetOutcome,
}

/// The tail of an NPU generation, after the generation call returned: release
/// `slot` (the admission permit) FIRST (0.9.1 C29), then the token-count
/// top-up, `CompletionTokens`, the duration / tok/s metrics and the terminal
/// `Done` (or `Error` + `Done`) — the order `run_generate` always had after the
/// release. A client woken by `Done` or `Error` therefore never finds its own
/// finished request in `active()`. Split out so the order is unit-tested
/// without a model.
///
/// `count_tokens` is the pipeline's tokenizer (engine thread), used once, after the slot is
/// released, to recount the reasoning text (`crate::reasoning_tokens`, rule 2: approximate — NPU
/// chunks carry no per-chunk token count). Its failure never fails the request (byte split + WARN).
/// Like the `CountTokens` command, the call runs outside the S1 call watch (`liveness.end()` has
/// run): a tokenizer that hung here would not be caught by the stall verdict (one short encode).
fn finish_npu_generation<S, C>(
    slot: S,
    result: anyhow::Result<NpuGenOutcome>,
    tail: &NpuGenTail<'_>,
    token_tx: &TokenSender,
    metrics: &HotMetrics,
    count_tokens: C,
) where
    C: FnOnce(&str) -> anyhow::Result<usize>,
{
    drop(slot);
    let NpuGenTail {
        model_id,
        started_at,
        ref acct,
        stopped_by_client,
        counted_prompt_tokens,
        kv_capacity,
        prefill,
        output,
        reasoning,
        budget,
    } = *tail;

    let generated = acct.total(result.as_ref().ok().map(|o| o.generated_tokens));
    metrics.token_generated(u64::try_from(generated - acct.sent()).unwrap_or(u64::MAX));
    if !stopped_by_client {
        let _ = token_tx.try_send(StreamEvent::CompletionTokens(generated));
    }
    let completion_tokens = u64::try_from(generated).unwrap_or(u64::MAX);

    let elapsed = started_at.elapsed().as_secs_f64();
    metrics.record_duration(Modality::Text, elapsed);
    if elapsed > 0.0 {
        #[allow(clippy::cast_precision_loss)]
        metrics.set_tokens_per_second(completion_tokens as f64 / elapsed);
    }

    match result {
        Ok(outcome) => {
            if outcome.input_tokens == 0 {
                tracing::debug!(
                    model_id,
                    counted_prompt_tokens,
                    "NPU perf metrics reported 0 input tokens — using the tokenizer count \
                     for the KV-full rule"
                );
            }
            let outcome = with_input_fallback(outcome, counted_prompt_tokens);
            // 0.9.1 V1-usage: the usage block's figures — prompt = the tokenizer count sent as
            // PromptTokens (the pipeline's own count when that failed), completion = `generated`.
            // A client-stopped request (gone, or not keeping up) is not a finished request: no
            // recount, no counters, no prefill sample — as on the text engine (review R-4, P6).
            let reasoning_tokens = (!stopped_by_client).then(|| {
                let prompt_tokens = counted_prompt_tokens.unwrap_or(outcome.input_tokens);
                if counted_prompt_tokens.is_none() && prompt_tokens > 0 {
                    // The tokenizer count failed, so no PromptTokens went out at the start: send
                    // the pipeline's own count now, so usage.prompt_tokens and the counter agree
                    // (review R-6).
                    let _ = token_tx.try_send(StreamEvent::PromptTokens(prompt_tokens));
                }
                let reasoning_count = crate::reasoning_tokens::recount_reasoning(
                    model_id,
                    output,
                    reasoning,
                    generated,
                    count_tokens,
                );
                metrics.record_request_tokens(
                    Modality::Text,
                    prompt_tokens,
                    reasoning_count.tokens,
                );
                if let Some(prefill) = prefill {
                    metrics.record_prefill(Modality::Text, prompt_tokens, prefill);
                }
                reasoning_count.tokens
            });
            if outcome.input_tokens != 0 {
                tracing::debug!(
                    model_id,
                    input_tokens = outcome.input_tokens,
                    generated_tokens = outcome.generated_tokens,
                    "NPU generation complete"
                );
            }
            let finish = npu_finish_reason(outcome, kv_capacity);
            if finish != outcome.finish {
                tracing::warn!(
                    model_id,
                    input_tokens = outcome.input_tokens,
                    generated_tokens = outcome.generated_tokens,
                    kv_capacity,
                    "NPU generation hit the compiled KV size (max_prompt_len + \
                     min_response_len) — reporting finish_reason length"
                );
            }
            // 0.9.1: a budget stop (or `length` inside the reasoning) is reported as
            // ThinkingBudget (`length`).
            let finish = budget.finish(finish);
            if let Some(n) = reasoning_tokens {
                let _ = token_tx.try_send(StreamEvent::ReasoningTokens(n));
            }
            let _ = token_tx.try_send(StreamEvent::Done(finish));
        }
        Err(e) => {
            tracing::error!(model_id, error = %e, "NPU generation failed");
            let _ = token_tx.try_send(StreamEvent::Error(e.to_string()));
            let _ = token_tx.try_send(StreamEvent::Done(FinishReason::Stop));
        }
    }
}

/// Per-request token accounting for NPU streaming, kept free of the pipeline
/// so its edge cases are unit-testable.
///
/// Counts the chunks actually delivered to the client; the request's total is
/// the pipeline's own generated-token count, which can exceed the chunk count
/// (one chunk may carry several tokens) but never legitimately fall below it.
#[derive(Debug, Default)]
struct ChunkAccounting {
    sent: usize,
}

impl ChunkAccounting {
    /// Record one delivered chunk; `true` when it was the first (TTFT point).
    fn record_sent(&mut self) -> bool {
        self.sent += 1;
        self.sent == 1
    }

    /// Chunks delivered so far.
    fn sent(&self) -> usize {
        self.sent
    }

    /// The request's completion-token total: the pipeline's count when it
    /// returned one (`None` on a generation error), floored at the chunks
    /// already delivered — each delivered chunk held at least one token.
    fn total(&self, pipeline_generated: Option<usize>) -> usize {
        pipeline_generated.map_or(self.sent, |g| g.max(self.sent))
    }
}

/// Fill a missing prompt count: when the pipeline's perf metrics report
/// `input_tokens == 0` (a build that doesn't populate them), use the
/// tokenizer count taken at the start of the request, so the KV-full rule in
/// [`npu_finish_reason`] doesn't silently weaken to "generated alone".
fn with_input_fallback(
    mut outcome: crate::ov_pipeline::NpuGenOutcome,
    counted_prompt_tokens: Option<usize>,
) -> crate::ov_pipeline::NpuGenOutcome {
    if outcome.input_tokens == 0
        && let Some(n) = counted_prompt_tokens
    {
        outcome.input_tokens = n;
    }
    outcome
}

/// The honest finish reason for an NPU generation. The pipeline reports
/// `Stop` whenever the token budget was not reached, including when the
/// static KV cache filled up and cut the answer short — that is a length
/// stop, not an EOS. A full cache always shows as `prompt + generated ==
/// kv_capacity + 1` (the final token is sampled from the last slot but never
/// stored), so the rule is `>` — a genuine EOS on the last slot (`==`) stays
/// `Stop`.
fn npu_finish_reason(
    outcome: crate::ov_pipeline::NpuGenOutcome,
    kv_capacity: usize,
) -> FinishReason {
    if outcome.finish == FinishReason::Stop
        && kv_capacity > 0
        && outcome.input_tokens + outcome.generated_tokens > kv_capacity
    {
        FinishReason::Length
    } else {
        outcome.finish
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::streaming::stream_channel;

    /// A fresh handle reports zero active/waiting requests and its configured cap.
    #[test]
    fn idle_handle_reports_zero_active_and_waiting() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<NpuCommand>(NPU_CHANNEL_CAP);
        let h = NpuHandle::from_sender(tx, "test-npu");
        assert_eq!(h.active(), 0);
        assert_eq!(h.waiting(), 0);
    }

    /// The NPU engine reports `TextGen` kind — it shares the text-gen routing slot.
    #[test]
    fn kind_is_text_gen() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<NpuCommand>(NPU_CHANNEL_CAP);
        let h = NpuHandle::from_sender(tx, "test-npu");
        assert_eq!(h.kind(), ModelKind::TextGen);
        assert_eq!(h.max_concurrency(), 1000, "test handle uses the test cap");
    }

    /// When a slot opens (permit released), a waiting `generate` completes
    /// instead of returning `AtCapacity`.
    #[tokio::test]
    async fn generate_waits_and_succeeds_when_slot_opens() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<NpuCommand>(4);
        let handle = NpuHandle::from_sender_with_cap(tx, "test-npu", 1, 500);

        let (tok_tx, _tok_rx) = stream_channel();
        let result1 = handle
            .generate(
                "hi".to_owned(),
                GenParams {
                    max_new_tokens: 8,
                    ..GenParams::default()
                },
                tok_tx,
                Instant::now(),
            )
            .await;
        assert_eq!(
            result1,
            SubmitResult::Submitted,
            "first request must get the slot"
        );

        let first_cmd = rx.recv().await.unwrap();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            drop(first_cmd); // drops the embedded OwnedSemaphorePermit → slot returned
        });

        let (tok_tx2, _tok_rx2) = stream_channel();
        let result2 = handle
            .generate(
                "hi again".to_owned(),
                GenParams {
                    max_new_tokens: 8,
                    ..GenParams::default()
                },
                tok_tx2,
                Instant::now(),
            )
            .await;
        assert_eq!(
            result2,
            SubmitResult::Submitted,
            "second request must get the slot after the first releases it"
        );
    }

    /// When no slot opens within `queue_timeout`, `generate` returns
    /// `AtCapacity` (→ HTTP 429) rather than waiting forever.
    #[tokio::test]
    async fn generate_returns_at_capacity_after_timeout() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<NpuCommand>(4);
        let handle = NpuHandle::from_sender_with_cap(tx, "test-npu", 1, 50);

        let (tok_tx, _) = stream_channel();
        let _ = handle
            .generate(
                "hi".to_owned(),
                GenParams {
                    max_new_tokens: 8,
                    ..GenParams::default()
                },
                tok_tx,
                Instant::now(),
            )
            .await;

        let (tok_tx2, _) = stream_channel();
        let result = handle
            .generate(
                "hi again".to_owned(),
                GenParams {
                    max_new_tokens: 8,
                    ..GenParams::default()
                },
                tok_tx2,
                Instant::now(),
            )
            .await;
        assert_eq!(result, SubmitResult::AtCapacity, "must 429 after timeout");
    }

    /// `waiting()` counts exactly the callers currently parked at the
    /// admission gate.
    #[tokio::test]
    async fn waiting_counts_callers_parked_at_the_admission_gate() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<NpuCommand>(4);
        let handle = NpuHandle::from_sender_with_cap(tx, "test-npu", 1, 1000);
        assert_eq!(handle.waiting(), 0, "no parked callers initially");

        let (tok_tx, _r) = stream_channel();
        assert_eq!(
            handle
                .generate(
                    "hi".to_owned(),
                    GenParams {
                        max_new_tokens: 8,
                        ..GenParams::default()
                    },
                    tok_tx,
                    Instant::now()
                )
                .await,
            SubmitResult::Submitted
        );
        assert_eq!(handle.active(), 1);

        let handle2 = handle.clone();
        let parked = tokio::spawn(async move {
            let (tok_tx2, _r2) = stream_channel();
            handle2
                .generate(
                    "hi".to_owned(),
                    GenParams {
                        max_new_tokens: 8,
                        ..GenParams::default()
                    },
                    tok_tx2,
                    Instant::now(),
                )
                .await
        });

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(handle.waiting(), 1, "second caller is parked at the gate");

        let first_cmd = rx.recv().await.unwrap();
        drop(first_cmd);

        let result = parked.await.unwrap();
        assert_eq!(result, SubmitResult::Submitted);
        assert_eq!(handle.waiting(), 0, "no callers parked once admitted");
    }

    /// A closed channel (engine thread gone) reports `EngineDead`, not
    /// `AtCapacity` — the caller should respond 503, not 429.
    #[tokio::test]
    async fn generate_returns_engine_dead_when_channel_closed() {
        let (tx, rx) = tokio::sync::mpsc::channel::<NpuCommand>(4);
        drop(rx);
        let handle = NpuHandle::from_sender(tx, "dead-npu");

        let (tok_tx, _) = stream_channel();
        let result = handle
            .generate(
                "hi".to_owned(),
                GenParams {
                    max_new_tokens: 8,
                    ..GenParams::default()
                },
                tok_tx,
                Instant::now(),
            )
            .await;
        assert_eq!(result, SubmitResult::EngineDead);
    }

    /// The measured Lunar Lake case: an 863-token prompt stopped after 290
    /// generated tokens with the default 1024 + 128 KV — the pipeline said
    /// `Stop`, the answer was cut short, so the server must say `Length`.
    #[test]
    fn kv_full_stop_is_reported_as_length() {
        let outcome = crate::ov_pipeline::NpuGenOutcome {
            finish: FinishReason::Stop,
            generated_tokens: 290,
            input_tokens: 863,
        };
        assert_eq!(
            npu_finish_reason(outcome, npu_kv_capacity(None, None)),
            FinishReason::Length
        );
    }

    /// A real EOS well inside the KV, and a budget stop, pass through unchanged.
    #[test]
    fn eos_and_budget_stops_pass_through() {
        let eos = crate::ov_pipeline::NpuGenOutcome {
            finish: FinishReason::Stop,
            generated_tokens: 29,
            input_tokens: 32,
        };
        assert_eq!(npu_finish_reason(eos, 1152), FinishReason::Stop);
        let budget = crate::ov_pipeline::NpuGenOutcome {
            finish: FinishReason::Length,
            generated_tokens: 200,
            input_tokens: 32,
        };
        assert_eq!(npu_finish_reason(budget, 1152), FinishReason::Length);
    }

    /// `input_tokens == 0` from the pipeline falls back to the tokenizer count
    /// (0.7.0 plan A5): without it the live KV-full case (40 + 1113 over a
    /// 1152 capacity) would read as a genuine EOS.
    #[test]
    fn zero_input_tokens_falls_back_to_counted_prompt() {
        let reported = crate::ov_pipeline::NpuGenOutcome {
            finish: FinishReason::Stop,
            generated_tokens: 1113,
            input_tokens: 0,
        };
        let cap = npu_kv_capacity(None, None);
        assert_eq!(
            npu_finish_reason(reported, cap),
            FinishReason::Stop,
            "weakened rule"
        );
        let filled = with_input_fallback(reported, Some(40));
        assert_eq!(filled.input_tokens, 40);
        assert_eq!(npu_finish_reason(filled, cap), FinishReason::Length);
        // A real pipeline count is never overridden; no count leaves it at 0.
        let real = crate::ov_pipeline::NpuGenOutcome {
            input_tokens: 41,
            ..reported
        };
        assert_eq!(with_input_fallback(real, Some(40)).input_tokens, 41);
        assert_eq!(with_input_fallback(reported, None).input_tokens, 0);
    }

    /// Chunk accounting edge cases (0.7.0 plan A4): multi-token chunks take the
    /// pipeline's larger count; a pipeline count below the delivered chunks
    /// (impossible in theory) is floored at them; an error falls back to the
    /// chunks delivered; nothing delivered + error is 0.
    #[test]
    fn chunk_accounting_totals() {
        let mut acct = ChunkAccounting::default();
        assert!(
            acct.record_sent(),
            "first delivered chunk is the TTFT point"
        );
        assert!(!acct.record_sent());
        assert!(!acct.record_sent());
        assert_eq!(acct.sent(), 3);
        assert_eq!(acct.total(Some(7)), 7, "multi-token chunks");
        assert_eq!(acct.total(Some(2)), 3, "floored at delivered chunks");
        assert_eq!(acct.total(None), 3, "error path: what was delivered");
        assert_eq!(ChunkAccounting::default().total(None), 0);
        assert_eq!(
            ChunkAccounting::default().total(Some(4)),
            4,
            "stopped before any send"
        );
    }

    #[test]
    fn kv_capacity_defaults_and_overrides() {
        assert_eq!(npu_kv_capacity(None, None), 1024 + 128);
        assert_eq!(npu_kv_capacity(Some(2048), Some(512)), 2560);
        // Unaligned values are used as-is (live-verified: 1000/100 filled at 1101).
        assert_eq!(npu_kv_capacity(Some(1000), Some(100)), 1100);
    }

    /// The KV-full boundary, from live data: a full cache shows as capacity + 1
    /// total tokens (`Length`); ending exactly at capacity is a genuine EOS.
    #[test]
    fn kv_full_boundary_is_capacity_plus_one() {
        let at = |total: usize| crate::ov_pipeline::NpuGenOutcome {
            finish: FinishReason::Stop,
            generated_tokens: total - 40,
            input_tokens: 40,
        };
        let cap = npu_kv_capacity(Some(1000), Some(100));
        assert_eq!(npu_finish_reason(at(1101), cap), FinishReason::Length);
        assert_eq!(npu_finish_reason(at(1100), cap), FinishReason::Stop);
    }

    /// `NpuShape::resolve` falls back to `OpenVINO`'s defaults per dimension
    /// — the shape a PATCH-to-`null` + reload actually compiles.
    #[test]
    fn npu_shape_resolves_defaults_and_overrides() {
        assert_eq!(
            NpuShape::resolve(None, None),
            NpuShape {
                max_prompt_len: 1024,
                kv_capacity: 1024 + 128
            }
        );
        assert_eq!(
            NpuShape::resolve(Some(2048), None),
            NpuShape {
                max_prompt_len: 2048,
                kv_capacity: 2048 + 128
            }
        );
        assert_eq!(NpuShape::resolve(None, Some(512)).max_prompt_len, 1024);
    }
}

// ── D3-S1.A0: `active()` contract tests ─────────────────────────────────────
// Characterization of the busy signal that eviction reads (`ManagedEngine::active`).
// Behaviour-neutral: tests only. See the project's internal engineering log.
#[cfg(test)]
mod active_contract_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::model_manager::engine::ManagedEngine;

    /// `active()` is `cap - available_permits`: a held admission permit counts, a released one does not.
    #[test]
    fn active_counts_held_permits_for_npu() {
        let (tx, _rx) = tokio::sync::mpsc::channel(8);
        let h = NpuHandle::from_sender(tx, "t-npu");
        assert_eq!(h.active(), 0, "idle");
        let permit = Arc::clone(&h.sem).try_acquire_owned().unwrap();
        assert_eq!(h.active(), 1, "one permit held");
        drop(permit);
        assert_eq!(h.active(), 0, "released");
    }

    /// D3-S1.D2: with the closing flag set, a submit that already holds its permit is refused
    /// (retryable) and the permit is released; cleared again, submits are accepted.
    #[tokio::test]
    async fn closing_refuses_a_submit_and_releases_its_permit() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<NpuCommand>(4);
        let h = NpuHandle::from_sender(tx, "t-npu");
        let submit = |h: &NpuHandle| {
            let (tok_tx, _tok_rx) = crate::streaming::stream_channel();
            let h = h.clone();
            async move {
                h.generate(
                    "hi".to_owned(),
                    GenParams::default(),
                    tok_tx,
                    Instant::now(),
                )
                .await
            }
        };

        ManagedEngine::set_closing(&h, true);
        assert_eq!(submit(&h).await, SubmitResult::Closing);
        assert_eq!(h.active(), 0, "the refused submit's permit is released");
        assert!(rx.try_recv().is_err(), "nothing reached the engine");

        ManagedEngine::set_closing(&h, false);
        assert_eq!(submit(&h).await, SubmitResult::Submitted);
        assert!(rx.try_recv().is_ok(), "the command reached the engine");
    }

    /// Submit one generation with `prompt` to a fresh mock NPU; return every
    /// event the client got, each with `active()` read the moment it had it.
    async fn mock_npu_events_with_active(prompt: &str) -> Vec<(StreamEvent, usize)> {
        let (h, _thread) = spawn_mock_npu("t-npu");
        let (tok_tx, mut tok_rx) = crate::streaming::stream_channel();
        let submitted = h
            .generate(
                prompt.to_owned(),
                GenParams::default(),
                tok_tx,
                Instant::now(),
            )
            .await;
        assert_eq!(submitted, SubmitResult::Submitted);
        let mut seen = Vec::new();
        while let Some(ev) = tok_rx.recv().await {
            let active = h.active();
            seen.push((ev, active));
        }
        seen
    }

    /// 0.9.1 C29: the admission slot is released before `Done` reaches the
    /// client — `active()` is 0 the moment the client has `Done`.
    #[tokio::test]
    async fn npu_releases_the_slot_before_done_reaches_the_client() {
        let seen = mock_npu_events_with_active("hi").await;
        assert_eq!(seen.len(), 2, "{seen:?}");
        assert!(matches!(&seen[0], (StreamEvent::Token(t, 1), _) if t == "mock-npu"));
        assert!(
            matches!(&seen[1], (StreamEvent::Done(FinishReason::Stop), 0)),
            "the slot must be free when Done arrives: {seen:?}"
        );
    }

    /// 0.9.1 C29, error terminal: the slot is already released when the
    /// `Error` (and the trailing `Done`) reach the client.
    #[tokio::test]
    async fn npu_releases_the_slot_before_an_error_reaches_the_client() {
        let seen = mock_npu_events_with_active(MOCK_NPU_GENERATE_ERROR).await;
        assert_eq!(seen.len(), 2, "{seen:?}");
        assert!(
            matches!(&seen[0], (StreamEvent::Error(e), 0) if e == MOCK_NPU_GENERATE_ERROR),
            "the slot must be free when Error arrives: {seen:?}"
        );
        assert!(
            matches!(&seen[1], (StreamEvent::Done(FinishReason::Stop), 0)),
            "{seen:?}"
        );
    }

    /// C29 probe slot: rides in the slot tuple AFTER a real admission permit
    /// (tuple fields drop in order), so its `Drop` runs right after the permit
    /// is released and records the semaphore's held count and how many events
    /// the client channel already holds at that moment. `queued` is the half that
    /// discriminates: `held` is 0 by construction (the probe drops after the
    /// permit) and only confirms that drop order.
    struct ReleaseProbe {
        sem: Arc<Semaphore>,
        tx: TokenSender,
        at_release: Arc<std::sync::Mutex<Option<(usize, usize)>>>,
    }

    impl Drop for ReleaseProbe {
        fn drop(&mut self) {
            let held = 1 - self.sem.available_permits();
            let queued = crate::streaming::CHANNEL_CAPACITY - self.tx.capacity();
            *self.at_release.lock().unwrap() = Some((held, queued));
        }
    }

    /// Runs [`finish_npu_generation`] with `result` and a (permit, probe) slot;
    /// returns (held permits, queued events) at release time and the events.
    fn run_npu_tail(result: anyhow::Result<NpuGenOutcome>) -> ((usize, usize), Vec<StreamEvent>) {
        let sem = Arc::new(Semaphore::new(1));
        let permit = Arc::clone(&sem).try_acquire_owned().unwrap();
        let (tx, mut rx) = crate::streaming::stream_channel();
        let at_release = Arc::new(std::sync::Mutex::new(None));
        let probe = ReleaseProbe {
            sem: Arc::clone(&sem),
            tx: tx.clone(),
            at_release: Arc::clone(&at_release),
        };
        let metrics = HotMetrics::new(ModelKind::TextGen, "t-npu", "NPU");
        finish_npu_generation(
            (permit, probe),
            result,
            &NpuGenTail {
                model_id: "t-npu",
                started_at: Instant::now(),
                acct: ChunkAccounting::default(),
                stopped_by_client: false,
                counted_prompt_tokens: Some(5),
                kv_capacity: 4096,
                prefill: None,
                output: "",
                reasoning: crate::prompt_builder::ReasoningSplit::default(),
                budget: crate::thinking_budget::BudgetOutcome::default(),
            },
            &tx,
            &metrics,
            |_: &str| Ok(0),
        );
        drop(tx);
        let mut events = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            events.push(ev);
        }
        let at = at_release.lock().unwrap().expect("slot was dropped");
        (at, events)
    }

    /// 0.9.1 C29 (F1): the REAL NPU tail releases the slot before `Done`: at
    /// release the permit is back (0 held) and no event has been sent yet.
    #[test]
    fn npu_tail_releases_the_slot_before_done() {
        let (at, events) = run_npu_tail(Ok(NpuGenOutcome {
            finish: FinishReason::Stop,
            generated_tokens: 3,
            input_tokens: 5,
        }));
        assert_eq!(at, (0, 0), "released before any event was sent: {events:?}");
        assert!(
            matches!(
                events.as_slice(),
                [
                    StreamEvent::CompletionTokens(3),
                    StreamEvent::ReasoningTokens(0),
                    StreamEvent::Done(FinishReason::Stop)
                ]
            ),
            "{events:?}"
        );
    }

    /// 0.9.1 C29 (F1), error terminal: released before `Error` + `Done`.
    #[test]
    fn npu_tail_releases_the_slot_before_an_error() {
        let (at, events) = run_npu_tail(Err(anyhow::anyhow!("boom")));
        assert_eq!(at, (0, 0), "released before any event was sent: {events:?}");
        assert!(
            matches!(
                events.as_slice(),
                [
                    StreamEvent::CompletionTokens(0),
                    StreamEvent::Error(e),
                    StreamEvent::Done(FinishReason::Stop)
                ] if e == "boom"
            ),
            "{events:?}"
        );
    }

    // ---- 0.9.1 V1-usage: reasoning recount + prompt / prefill series ----

    use crate::metrics::test_sample as sample;

    /// The NPU tail for a successful generation of `generated` tokens; events + `/metrics` text.
    fn run_npu_usage(
        output: &str,
        counted_prompt: Option<usize>,
        generated: usize,
        prefill: Option<std::time::Duration>,
        count: impl FnOnce(&str) -> anyhow::Result<usize>,
    ) -> (Vec<StreamEvent>, String) {
        run_npu_usage_with(
            (false, crate::thinking_budget::BudgetOutcome::default()),
            output,
            counted_prompt,
            generated,
            prefill,
            count,
        )
    }

    fn run_npu_usage_with(
        (stopped, budget): (bool, crate::thinking_budget::BudgetOutcome),
        output: &str,
        counted_prompt: Option<usize>,
        generated: usize,
        prefill: Option<std::time::Duration>,
        count: impl FnOnce(&str) -> anyhow::Result<usize>,
    ) -> (Vec<StreamEvent>, String) {
        use metrics_exporter_prometheus::PrometheusBuilder;
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let (tx, mut rx) = crate::streaming::stream_channel();
        metrics::with_local_recorder(&recorder, || {
            let metrics = HotMetrics::new(ModelKind::TextGen, "u-npu", "NPU");
            let mut acct = ChunkAccounting::default();
            acct.record_sent();
            finish_npu_generation(
                (),
                Ok(NpuGenOutcome {
                    finish: FinishReason::Stop,
                    generated_tokens: generated,
                    input_tokens: 700,
                }),
                &NpuGenTail {
                    model_id: "u-npu",
                    started_at: Instant::now(),
                    acct,
                    stopped_by_client: stopped,
                    counted_prompt_tokens: counted_prompt,
                    kv_capacity: 0,
                    prefill,
                    output,
                    reasoning: crate::prompt_builder::ReasoningSplit::new(
                        Some(crate::model_manager::ReasoningParser::Qwen3),
                        false,
                    ),
                    budget,
                },
                &tx,
                &metrics,
                count,
            );
        });
        drop(tx);
        let mut events = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            events.push(ev);
        }
        (events, handle.render())
    }

    fn reasoning_of(events: &[StreamEvent]) -> usize {
        events
            .iter()
            .find_map(|e| match e {
                StreamEvent::ReasoningTokens(n) => Some(*n),
                _ => None,
            })
            .expect("ReasoningTokens event")
    }

    #[test]
    fn npu_recounts_reasoning_and_records_the_three_series() {
        let (events, r) = run_npu_usage(
            "<think>hmm</think>Yes.",
            Some(512),
            9,
            Some(std::time::Duration::from_secs(2)),
            |s: &str| {
                assert_eq!(s, "<think>hmm</think>");
                Ok(3)
            },
        );
        assert_eq!(reasoning_of(&events), 3);
        assert!(reasoning_of(&events) <= 9);
        assert_eq!(
            sample(&r, "rustedvino_prompt_tokens_total", "u-npu"),
            Some(512.0)
        );
        assert_eq!(
            sample(&r, "rustedvino_reasoning_tokens_total", "u-npu"),
            Some(3.0)
        );
        let sum = sample(&r, "rustedvino_prefill_tokens_per_second_sum", "u-npu").unwrap();
        assert!((sum - 256.0).abs() < 0.01, "512 tokens over 2 s: {sum}");
    }

    #[test]
    fn npu_tokenizer_error_falls_back_and_thinking_off_is_zero() {
        let (events, _) = run_npu_usage("<think>ab</think>cd", Some(300), 4, None, |_: &str| {
            Err(anyhow::anyhow!("no tokenizer"))
        });
        // 17 of 19 bytes are reasoning: 4 × 17/19 = 3.58 → 4.
        assert_eq!(reasoning_of(&events), 4);
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Done(FinishReason::Stop))
        ));
        let (events, r) = run_npu_usage("plain", None, 2, None, |_: &str| Ok(7));
        assert_eq!(reasoning_of(&events), 0);
        // R-6: the tokenizer count failed, so the pipeline's count goes out as PromptTokens
        // (usage.prompt_tokens and the counter agree).
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::PromptTokens(700))),
            "{events:?}"
        );
        // Tokenizer prompt count missing: the pipeline's own input count is used.
        assert_eq!(
            sample(&r, "rustedvino_prompt_tokens_total", "u-npu"),
            Some(700.0)
        );
    }

    #[test]
    fn npu_prefill_floor_holds() {
        let (_, r) = run_npu_usage(
            "x",
            Some(255),
            1,
            Some(std::time::Duration::from_millis(5)),
            |_: &str| Ok(0),
        );
        assert_eq!(
            sample(&r, "rustedvino_prefill_tokens_per_second_count", "u-npu"),
            Some(0.0)
        );
        assert_eq!(
            sample(&r, "rustedvino_prompt_tokens_total", "u-npu"),
            Some(255.0)
        );
    }

    /// R-4 (P6 symmetric): a client-stopped NPU request is not counted — no recount, no
    /// `ReasoningTokens`, no prompt / reasoning counts and no prefill sample — as on CB.
    #[test]
    fn npu_client_stopped_request_counts_nothing() {
        let (events, r) = run_npu_usage_with(
            (true, crate::thinking_budget::BudgetOutcome::default()),
            "<think>half",
            Some(900),
            5,
            Some(std::time::Duration::from_millis(100)),
            |_: &str| panic!("a stopped request must not be recounted"),
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, StreamEvent::ReasoningTokens(_))),
            "{events:?}"
        );
        assert_eq!(
            sample(&r, "rustedvino_prompt_tokens_total", "u-npu"),
            Some(0.0)
        );
        assert_eq!(
            sample(&r, "rustedvino_reasoning_tokens_total", "u-npu"),
            Some(0.0)
        );
        assert_eq!(
            sample(&r, "rustedvino_prefill_tokens_per_second_count", "u-npu"),
            Some(0.0)
        );
    }

    /// 0.9.1 strict thinking budget: a budget stop is a FINISHED request — recounted, counted,
    /// `ReasoningTokens` then `Done(ThinkingBudget)` (`length`); the KV-full rule does not hide it.
    #[test]
    fn npu_budget_stop_is_length_and_counted_as_finished() {
        let stopped = crate::thinking_budget::BudgetOutcome {
            armed: true,
            stopped: true,
            inside: true,
        };
        let (events, r) = run_npu_usage_with(
            (false, stopped),
            "<think>still going",
            Some(700),
            6,
            Some(std::time::Duration::from_millis(100)),
            |_: &str| Ok(6),
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::ReasoningTokens(6))),
            "{events:?}"
        );
        assert!(
            matches!(
                events.last(),
                Some(StreamEvent::Done(FinishReason::ThinkingBudget))
            ),
            "{events:?}"
        );
        assert_eq!(
            sample(&r, "rustedvino_prompt_tokens_total", "u-npu"),
            Some(700.0)
        );
        assert_eq!(
            sample(&r, "rustedvino_reasoning_tokens_total", "u-npu"),
            Some(6.0)
        );
        assert_eq!(
            sample(&r, "rustedvino_prefill_tokens_per_second_count", "u-npu"),
            Some(1.0)
        );
    }

    /// Drives the real NPU streamer callback (`NpuStream::on_chunk`) the way `OvPipeline` does —
    /// one call per chunk until it answers `false` (stop: this engine's convention) — then the
    /// real tail. Returns the chunks delivered and the events.
    fn drive_npu_stream(chunks: &[&str], budget: Option<u32>) -> (usize, Vec<StreamEvent>) {
        let (tx, mut rx) = crate::streaming::stream_channel();
        let metrics = HotMetrics::new(ModelKind::TextGen, "s-npu", "NPU");
        let params = GenParams {
            reasoning: crate::prompt_builder::ReasoningSplit::new(
                Some(crate::model_manager::ReasoningParser::Qwen3),
                true,
            ),
            thinking_budget: budget,
            ..GenParams::default()
        };
        let started_at = Instant::now();
        let mut stream = NpuStream::new("s-npu", started_at, &params);
        let mut delivered = 0;
        for c in chunks {
            delivered += 1;
            if !stream.on_chunk(c, &tx, &metrics) {
                break;
            }
        }
        finish_npu_generation(
            (),
            Ok(NpuGenOutcome {
                finish: FinishReason::Stop,
                generated_tokens: delivered,
                input_tokens: 40,
            }),
            &NpuGenTail {
                model_id: "s-npu",
                started_at,
                acct: stream.acct,
                stopped_by_client: stream.stopped_by_client,
                counted_prompt_tokens: Some(40),
                kv_capacity: 0,
                prefill: None,
                output: &stream.output,
                reasoning: params.reasoning,
                budget: crate::thinking_budget::BudgetOutcome::of(
                    stream.budget.as_ref(),
                    stream.stopped_by_client,
                ),
            },
            &tx,
            &metrics,
            |s: &str| Ok(s.len()),
        );
        drop(tx);
        let mut events = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            events.push(ev);
        }
        (delivered, events)
    }

    /// The NPU streamer with a budget: 4 tokens = ceil(4 / 1.04) = 4 chunks; the stop (`false`)
    /// is asked once, at the fourth chunk, nothing after it is sent, and the request ends
    /// `Done(ThinkingBudget)`.
    #[test]
    fn npu_streamer_stops_at_the_chunk_budget() {
        let chunks = ["a", "", "b", "c", "d", "e"];
        let (delivered, events) = drive_npu_stream(&chunks, Some(4));
        assert_eq!(
            delivered, 5,
            "the empty chunk is skipped, the stop comes at `d`"
        );
        let tokens = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::Token(..)))
            .count();
        assert_eq!(tokens, 4);
        assert!(
            matches!(
                events.last(),
                Some(StreamEvent::Done(FinishReason::ThinkingBudget))
            ),
            "{events:?}"
        );
        let (delivered, events) = drive_npu_stream(&chunks, None);
        assert_eq!(delivered, 6);
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Done(FinishReason::Stop))
        ));
    }
}
