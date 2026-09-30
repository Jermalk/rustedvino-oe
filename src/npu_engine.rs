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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::mpsc::{Sender, error::TrySendError};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

use crate::cb_engine::SubmitResult;
use crate::metrics::{HotMetrics, Modality};
use crate::model_manager::{ManagedEngine, ModelKind};
use crate::ov_cb::{FinishReason, GenParams};
use crate::ov_pipeline::OvPipeline;
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
        /// Admission-gate permit, held until this generation finishes. Dropped
        /// at the end of the engine thread's match arm, releasing the slot for
        /// the next waiting caller. See [`NpuHandle::generate`].
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
}

impl NpuHandle {
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
            tokio::time::timeout(self.queue_timeout, Arc::clone(&self.sem).acquire_owned()).await
        };
        let permit = match permit_result {
            Ok(Ok(p)) => p,
            Ok(Err(_)) => return SubmitResult::EngineDead, // semaphore closed (shouldn't happen)
            Err(_) => return SubmitResult::AtCapacity,     // timed out waiting for a permit
        };

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
        self.tx
            .send(NpuCommand::CountTokens { text, reply })
            .await
            .map_err(|_| anyhow::anyhow!("NPU engine unavailable"))?;
        reply_rx
            .await
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

    let thread = std::thread::Builder::new()
        .name(format!("npu-engine-{model_id_owned}"))
        .spawn(move || {
            tracing::info!(
                model_id = model_id_owned,
                device,
                "NPU engine thread started"
            );
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
                        );
                        // Balance the handle's admission acquire, whatever the
                        // generation outcome — releases the slot for the next
                        // waiting caller.
                        drop(permit);
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
        },
        thread,
    ))
}

/// GPU-free mock NPU engine — test analogue of [`spawn_npu_engine`].
///
/// Answers each `Generate` with one token + `Done(Stop)` and drops the
/// permit per command, so `active()` is honest in routing tests.
#[cfg(test)]
#[allow(dead_code)] // Reserved for lifecycle tests that exercise NPU placement.
pub(crate) fn spawn_mock_npu(model_id: &str) -> (NpuHandle, std::thread::JoinHandle<()>) {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<NpuCommand>(NPU_CHANNEL_CAP);
    let sem = Arc::new(Semaphore::new(NPU_CHANNEL_CAP));

    let thread = std::thread::spawn(move || {
        while let Some(cmd) = rx.blocking_recv() {
            match cmd {
                NpuCommand::Generate {
                    token_tx, permit, ..
                } => {
                    let _ = token_tx.blocking_send(StreamEvent::Token("mock-npu".to_owned(), 1));
                    let _ = token_tx.blocking_send(StreamEvent::Done(FinishReason::Stop));
                    drop(permit);
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
        },
        thread,
    )
}

/// Drive one generation request on the engine thread.
#[allow(clippy::too_many_arguments)]
fn run_generate(
    pipeline: &mut OvPipeline,
    model_id: &str,
    prompt: &str,
    params: &GenParams,
    token_tx: &TokenSender,
    metrics: &HotMetrics,
    started_at: Instant,
    kv_capacity: usize,
) {
    metrics.request_accepted(Modality::Text);

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

    // The callback fires once per decoded text CHUNK, and a chunk can carry
    // several tokens (the streamer holds back incomplete pieces), so counting
    // chunks undercounts (seen live: 192 for a 200-token generation). Each
    // chunk goes out immediately; the pipeline's own total follows as
    // `CompletionTokens` just before `Done`, and consumers use it in place of
    // their running sum.
    let mut acct = ChunkAccounting::default();
    let mut stopped_by_client = false;
    let result = pipeline.generate(prompt, params, |text| {
        if text.is_empty() {
            return true; // consumers skip empty deltas; nothing to send or count
        }
        // try_send, never blocking: same backpressure policy as the VLM and CB
        // engines — a stalled client must not wedge this engine thread and park
        // every queued NPU request behind it.
        // NOTE: OvPipeline::generate convention is true=continue, false=stop.
        // This is the opposite of the VLM engine callback — do not copy vlm_engine.rs here.
        match token_tx.try_send(StreamEvent::Token(text.to_owned(), 1)) {
            Ok(()) => {
                if acct.record_sent() {
                    // TTFT = the first token the client can actually receive.
                    metrics.record_ttft(Modality::Text, started_at.elapsed().as_secs_f64());
                }
                metrics.token_generated(1);
                true
            }
            Err(TrySendError::Full(_)) => {
                tracing::warn!(
                    "NPU client not keeping up (token channel full) — stopping generation"
                );
                stopped_by_client = true;
                false
            }
            Err(TrySendError::Closed(_)) => {
                stopped_by_client = true;
                false
            }
        }
    });

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
