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
//     dev/plans/vlm-admission-gate-fix.md's recipe): a blocking channel send
//     with no timeout previously let a full queue park the caller forever
//     instead of returning a 429.
//   - NPU compiles a FIXED-SHAPE graph ahead of time (unlike GPU/CPU's
//     dynamic-shape kernels), so OpenVINO GenAI's static LLMPipeline takes a
//     compile-time `MAX_PROMPT_LEN` (default 1024 tokens) — a request whose
//     templated prompt exceeds it hard-fails. Configurable via the per-model
//     `max_prompt_len` config field (`ModelPolicy`), threaded through
//     `OvPipeline::new` → `ov_pipeline_create`'s AnyMap (`dev/ovms-gap.md` #6,
//     CLOSED 2026-07-16) — raising it grows the compiled KV-cache and compile
//     time; it is a build-time tradeoff, not a per-request setting. The chat
//     handler's `gate_npu_prompt` pre-flights the effective limit (config
//     override, else `NPU_DEFAULT_MAX_PROMPT_LEN`) via `NpuCommand::
//     CountTokens`, independent of `generate()`, so an over-limit prompt gets
//     a clean 400 instead of OpenVINO's raw C++ exception text.
//   - `OvPipeline::new` never receives (and `ov_pipeline_create` never sets)
//     `ov::cache_dir` — unlike every other engine kind in this codebase
//     (CB/VLM/TTS/STT/image, all of which thread `Config::ov_cache_dir`
//     through). Every NPU model load, including a plain server restart with
//     an unchanged config, recompiles from scratch (~20–40 s on Lunar Lake).
//     One consequence worth knowing: changing `max_prompt_len` between
//     restarts is always safe — there is no persisted blob that could go
//     stale against the new value, because nothing is ever persisted here.
// ============================================================

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::mpsc::{Sender, error::TrySendError};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

use crate::cb_engine::SubmitResult;
use crate::metrics::{HotMetrics, Modality};
use crate::model_manager::{ManagedEngine, ModelKind};
use crate::ov_cb::FinishReason;
use crate::ov_pipeline::OvPipeline;
use crate::streaming::{StreamEvent, TokenSender};

/// NPU admission-gate size (also the reported `max_concurrency`): up to this
/// many requests may be queued/generating before the gate returns
/// [`SubmitResult::AtCapacity`]. The raw channel buffer is set to the same
/// value — a permit is always acquired before a command is sent, so the
/// channel itself never needs to absorb more than `NPU_CHANNEL_CAP` commands.
pub(crate) const NPU_CHANNEL_CAP: usize = 8;

// ── Commands ─────────────────────────────────────────────────────────────────

pub(crate) enum NpuCommand {
    /// Run a single-stream NPU LLM generation.
    Generate {
        /// Pre-rendered prompt string (chat template already applied in Rust).
        prompt: String,
        /// Maximum new tokens to generate.
        max_new_tokens: usize,
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
}

impl NpuHandle {
    /// The model ID of the loaded NPU LLM.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
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
        max_new_tokens: usize,
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
                max_new_tokens,
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
                        max_new_tokens,
                        token_tx,
                        started_at,
                        permit,
                    } => {
                        run_generate(
                            &mut pipeline,
                            &model_id_owned,
                            &prompt,
                            max_new_tokens,
                            &token_tx,
                            &metrics,
                            started_at,
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
    max_new_tokens: usize,
    token_tx: &TokenSender,
    metrics: &HotMetrics,
    started_at: Instant,
) {
    metrics.request_accepted(Modality::Text);
    let mut completion_tokens: u64 = 0;
    let mut first_token_seen = false;

    // Same tokenizer call `NpuCommand::CountTokens`/`gate_npu_prompt` already
    // use, just made from inside this generation instead of a separate
    // round-trip — the prior "no prompt token count available from
    // LLMPipeline" comment here was stale: `OvPipeline::count_tokens` has
    // been callable on this exact pipeline the whole time. try_send as the
    // very first event, same convention as `cb_engine.rs`'s `PromptTokens`
    // send; fail-open (log and continue) rather than aborting the request,
    // matching `gate_npu_prompt`'s own `Ok(_) | Err(_) => Ok(())` policy.
    match pipeline.count_tokens(prompt) {
        Ok(n) => {
            let _ = token_tx.try_send(StreamEvent::PromptTokens(n));
        }
        Err(e) => {
            tracing::warn!(model_id, error = %e, "NPU prompt token count failed");
        }
    }

    let result = pipeline.generate(prompt, max_new_tokens, |text| {
        if !first_token_seen {
            first_token_seen = true;
            metrics.record_ttft(Modality::Text, started_at.elapsed().as_secs_f64());
        }
        completion_tokens += 1;
        metrics.token_generated(1);
        // try_send, never blocking: same backpressure policy as the VLM and CB
        // engines — a stalled client must not wedge this engine thread and park
        // every queued NPU request behind it.
        // NOTE: OvPipeline::generate convention is true=continue, false=stop.
        // This is the opposite of the VLM engine callback — do not copy vlm_engine.rs here.
        match token_tx.try_send(StreamEvent::Token(text.to_owned(), 1)) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) => {
                tracing::warn!(
                    "NPU client not keeping up (token channel full) — stopping generation"
                );
                false
            }
            Err(TrySendError::Closed(_)) => false,
        }
    });

    let elapsed = started_at.elapsed().as_secs_f64();
    metrics.record_duration(Modality::Text, elapsed);
    if elapsed > 0.0 {
        #[allow(clippy::cast_precision_loss)]
        metrics.set_tokens_per_second(completion_tokens as f64 / elapsed);
    }

    match result {
        Ok(finish) => {
            let _ = token_tx.try_send(StreamEvent::Done(finish));
        }
        Err(e) => {
            tracing::error!(model_id, error = %e, "NPU generation failed");
            let _ = token_tx.try_send(StreamEvent::Error(e.to_string()));
            let _ = token_tx.try_send(StreamEvent::Done(FinishReason::Stop));
        }
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
            .generate("hi".to_owned(), 8, tok_tx, Instant::now())
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
            .generate("hi again".to_owned(), 8, tok_tx2, Instant::now())
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
            .generate("hi".to_owned(), 8, tok_tx, Instant::now())
            .await;

        let (tok_tx2, _) = stream_channel();
        let result = handle
            .generate("hi again".to_owned(), 8, tok_tx2, Instant::now())
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
                .generate("hi".to_owned(), 8, tok_tx, Instant::now())
                .await,
            SubmitResult::Submitted
        );
        assert_eq!(handle.active(), 1);

        let handle2 = handle.clone();
        let parked = tokio::spawn(async move {
            let (tok_tx2, _r2) = stream_channel();
            handle2
                .generate("hi".to_owned(), 8, tok_tx2, Instant::now())
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
            .generate("hi".to_owned(), 8, tok_tx, Instant::now())
            .await;
        assert_eq!(result, SubmitResult::EngineDead);
    }
}
