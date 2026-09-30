// ============================================================
// src/vlm_engine.rs — VLM engine thread (Phase 5.2)
// ============================================================
// Same architecture as `cb_engine.rs` but for VLMPipeline:
//   - One dedicated OS thread per VLM model
//   - Commands arrive via an mpsc channel (capacity 8)
//   - Engine thread processes one Generate at a time (VLMPipeline
//     is single-threaded and blocking — no CB-style parallelism)
//   - Token stream goes to the same StreamEvent / TokenSender as the
//     CB path, so the chat handler and SSE layer are reused unchanged.
// ============================================================

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::mpsc::{Sender, error::TrySendError};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

use crate::cb_engine::SubmitResult;
use crate::image_util::DecodedImage;
use crate::metrics::{HotMetrics, Modality};
use crate::model_manager::{ManagedEngine, ModelKind, VLM_KV_WEDGE_MARKER};
use crate::ov_cb::{FinishReason, GenParams};
use crate::ov_vlm::OvVlmEngine;
use crate::streaming::{StreamEvent, TokenSender};

/// Last-resort VLM concurrency cap — used only when neither layer 2
/// (`max_concurrent_streams`) nor layer 1 (`max_num_seqs`) resolves to a value
/// (see `OvEngineFactory::load`'s `Vision` arm). Not the normal default in
/// practice: a configured pipeline `max_num_seqs > 0` already wins over this.
/// Sizes the admission [`Semaphore`] in [`spawn_vlm_engine`] — the real
/// concurrency gate. See [`VLM_COMMAND_CAPACITY`] for the (unrelated) raw
/// channel buffer size.
pub(crate) const VLM_CHANNEL_CAP: usize = 8;

/// Raw command-channel buffer size — deliberately decoupled from
/// [`VLM_CHANNEL_CAP`]/the configured cap. Concurrency is gated entirely by
/// the admission `Semaphore` (a permit is acquired *before* a command is
/// ever sent), so the channel itself only needs to absorb a burst of
/// simultaneous submissions — mirrors `cb_engine::COMMAND_CAPACITY`.
const VLM_COMMAND_CAPACITY: usize = 128;

// ── Commands ─────────────────────────────────────────────────────────────────

pub(crate) enum VlmCommand {
    /// Run a VLM generation.
    Generate {
        /// Full chat history as JSON values (image-URL parts stripped to text).
        messages: Vec<serde_json::Value>,
        /// Decoded NHWC RGB images from `image_url` content parts, in order.
        images: Vec<DecodedImage>,
        /// `OpenAI` `tools` array, when active for this request (suppressed by
        /// `tool_choice: "none"` same as the text path) — `None`/empty means
        /// no tools binding. See [`OvVlmEngine::generate`] for why this has
        /// to travel with the request instead of being baked into a prompt.
        tools: Option<Vec<serde_json::Value>>,
        /// Extra chat-template variables for this request (today:
        /// `{"enable_thinking": <bool>}`). Travels with the request for the
        /// same reason `tools` does — a VLM's chat template is applied inside
        /// `VLMPipeline`, so a per-request template variable cannot be baked
        /// into a prompt on our side. `None` renders exactly as before.
        extra_context: Option<serde_json::Value>,
        /// Sampling parameters. Boxed — adding `tools` above pushed this
        /// variant's size past `clippy::large_enum_variant`'s threshold
        /// relative to `CountTokens`; boxing this one field (clippy's own
        /// suggestion) is cheaper than boxing several smaller ones.
        params: Box<GenParams>,
        /// Token stream sink — receives `Token`, `Done`, and `Error` events.
        token_tx: TokenSender,
        /// Wall-clock start (handler entry) for TTFT / duration / tok/s metrics.
        started_at: Instant,
        /// Admission-gate permit, held until this generation finishes. Dropped
        /// at the end of the engine thread's match arm, releasing the slot for
        /// the next waiting caller. See [`VlmHandle::generate`].
        permit: OwnedSemaphorePermit,
        /// Watchdog ticket — dropped alongside `permit`, clearing this
        /// generation's entry in the [`GenerationClock`]. See
        /// [`VlmHandle::generate`].
        ticket: GenerationTicket,
    },
    /// Count tokens in `text` on the engine thread (the only thread allowed to
    /// touch the tokenizer). Serves the L0 prompt-length gate in `chat.rs`,
    /// checked before `Generate` is ever sent — see
    /// [`OvVlmEngine::count_tokens`](crate::ov_vlm::OvVlmEngine::count_tokens).
    CountTokens {
        /// Text to count.
        text: String,
        /// Where the engine thread sends the count (or a stringified error).
        reply: oneshot::Sender<Result<usize, String>>,
    },
}

// ── Generation watchdog clock ───────────────────────────────────────────────

/// Shared table of in-flight generation start times, keyed by a monotonic id.
///
/// Backs [`ManagedEngine::oldest_generation_age`] for the VLM path — the
/// supervisor's watchdog polls this (via `ModelManager::oldest_generation_age`
/// → an admin endpoint) to catch a generation stuck far longer than any real
/// request plausibly takes (2026-07-28 incident:
/// the project's internal engineering log; a GPU driver engine-reset
/// that `openvino_genai` never surfaced as an error, so the engine thread
/// spun forever with no way for this process to detect or cancel it).
///
/// A [`GenerationTicket`] is created on the *caller's* task, in
/// [`VlmHandle::generate`], before the command is dispatched — never on the
/// engine thread, which is exactly what might be wedged. Its `Drop` removes
/// the table entry unconditionally, whichever path runs: the engine thread
/// finishing the generation normally (the ticket travels inside
/// `VlmCommand::Generate` and drops alongside the admission `permit`), or the
/// caller's own future being torn down before the command ever sends (a
/// client disconnecting mid-submit). Either way there is exactly one Drop
/// site per ticket, so the table can't leak a stale entry that would later
/// make the watchdog kill a healthy worker.
#[derive(Clone, Default, Debug)]
pub(crate) struct GenerationClock(Arc<Mutex<HashMap<u64, Instant>>>);

static NEXT_GENERATION_ID: AtomicU64 = AtomicU64::new(0);

impl GenerationClock {
    /// Starts tracking a new generation, returning the ticket that must be
    /// held (directly or via the command it travels inside) for the
    /// generation's full duration.
    fn start(&self) -> GenerationTicket {
        let id = NEXT_GENERATION_ID.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut table) = self.0.lock() {
            table.insert(id, Instant::now());
        }
        GenerationTicket {
            table: Arc::clone(&self.0),
            id,
        }
    }

    /// Age of the longest currently in-flight generation, if any are active.
    fn oldest_age(&self) -> Option<Duration> {
        let now = Instant::now();
        let table = self.0.lock().ok()?;
        table.values().map(|start| now.duration_since(*start)).max()
    }
}

/// RAII handle for one [`GenerationClock`] entry — see the clock's doc for why
/// the Drop-based removal (not a statement after some await) is load-bearing.
pub(crate) struct GenerationTicket {
    table: Arc<Mutex<HashMap<u64, Instant>>>,
    id: u64,
}

impl Drop for GenerationTicket {
    fn drop(&mut self) {
        if let Ok(mut table) = self.table.lock() {
            table.remove(&self.id);
        }
    }
}

// ── Handle ───────────────────────────────────────────────────────────────────

/// Cloneable submit handle for the VLM engine thread.
///
/// The channel capacity (and therefore the queue depth) is set by
/// `max_concurrent_streams` in the model config; it defaults to
/// [`VLM_CHANNEL_CAP`] when not configured. The engine thread processes one
/// generation at a time — requests naturally serialise in the channel.
#[derive(Clone, Debug)]
pub struct VlmHandle {
    tx: Sender<VlmCommand>,
    /// The model ID of the loaded VLM, returned in the `model` response field.
    model_id: Arc<str>,
    /// Admission-gate permits — `cap` of them. A permit is acquired *before* a
    /// `Generate` command is sent and held by the engine thread until that
    /// generation finishes, so `available_permits()` is honest occupancy
    /// (queued + generating), not merely "callers who called `generate()`".
    /// Mirrors `cb_engine::EngineHandle`'s `sem` field exactly.
    sem: Arc<Semaphore>,
    /// Channel capacity = max queued + 1 running. Set from `max_concurrent_streams`
    /// config; defaults to [`VLM_CHANNEL_CAP`]. Also the semaphore's permit count.
    cap: usize,
    /// Maximum time [`generate`](Self::generate) waits for a permit before
    /// returning [`SubmitResult::AtCapacity`] (→ HTTP 429).
    queue_timeout: Duration,
    /// Callers currently parked in [`generate`](Self::generate) awaiting an
    /// admission permit — the engine's honest `rustedvino_requests_waiting`.
    /// Incremented before the permit wait, decremented the moment it resolves
    /// (granted or timed out). Mirrors `cb_engine::EngineHandle`'s `waiting`.
    waiting: Arc<AtomicUsize>,
    /// Watchdog clock — see [`GenerationClock`].
    clock: GenerationClock,
}

impl VlmHandle {
    /// The model ID of the loaded VLM.
    ///
    /// Used by the chat handler to set the `model` field in the response,
    /// so it reflects the actual engine that served the request rather than
    /// whatever model name the client sent in the request.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Build a handle around an existing command channel — test seam only.
    ///
    /// Mirrors [`crate::cb_engine::EngineHandle::from_sender`]: lets unit tests
    /// construct a `Vision` handle with no GPU. The in-flight counter starts at
    /// zero. (For a mock that also *responds*, see `spawn_mock_vlm`.)
    #[cfg(test)]
    #[must_use]
    pub(crate) fn from_sender(tx: Sender<VlmCommand>, model_id: &str) -> Self {
        let cap = 1000; // generous test cap — never hits the admission gate
        Self {
            tx,
            model_id: Arc::from(model_id),
            sem: Arc::new(Semaphore::new(cap)),
            cap,
            queue_timeout: Duration::from_secs(30),
            waiting: Arc::new(AtomicUsize::new(0)),
            clock: GenerationClock::default(),
        }
    }

    /// Creates a test handle with an explicit cap and timeout. Used by
    /// admission-gate tests that need to observe `AtCapacity` behaviour.
    /// Mirrors `cb_engine::EngineHandle::from_sender_with_cap`.
    #[cfg(test)]
    pub(crate) fn from_sender_with_cap(
        tx: Sender<VlmCommand>,
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
            clock: GenerationClock::default(),
        }
    }

    /// Enqueue a VLM generation request.
    ///
    /// Waits up to `queue_timeout` for an admission permit before returning
    /// [`SubmitResult::AtCapacity`] (→ HTTP 429) — mirrors
    /// `cb_engine::EngineHandle::add_request`'s admission gate exactly, so a
    /// VLM under sustained overload gives clients a real "busy" signal instead
    /// of queuing indefinitely (the previous behaviour: a blocking channel
    /// send that only ever waited, never rejected).
    ///
    /// Returns [`SubmitResult::EngineDead`] if the command channel is closed
    /// (engine thread exited) — caller should respond HTTP 503.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn generate(
        &self,
        messages: Vec<serde_json::Value>,
        images: Vec<DecodedImage>,
        tools: Option<Vec<serde_json::Value>>,
        extra_context: Option<serde_json::Value>,
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

        // Started here, on the caller's task, before dispatch — never on the
        // engine thread, which is exactly what a wedged FFI call would leave
        // unresponsive. Travels inside the command so its `Drop` always runs:
        // engine-thread completion (alongside `permit`) or a cancelled send
        // right here (this task torn down mid-`.await`) both clear it. See
        // `GenerationClock`.
        let ticket = self.clock.start();

        // Permit acquired. Send to the engine — if the channel is closed, the
        // permit (moved into the command) drops here, automatically releasing
        // the slot back to the semaphore.
        let sent = self
            .tx
            .send(VlmCommand::Generate {
                messages,
                images,
                tools,
                extra_context,
                params: Box::new(params),
                token_tx,
                started_at,
                permit,
                ticket,
            })
            .await;
        if sent.is_err() {
            return SubmitResult::EngineDead;
        }
        SubmitResult::Submitted
    }

    /// Count the tokens `text` encodes to with the VLM's tokenizer.
    ///
    /// The tokenizer lives on the engine thread (its `InferRequest` is not
    /// thread-safe), so this routes through the command channel and awaits a
    /// `oneshot` reply rather than calling the tokenizer directly. It does NOT
    /// consume an in-flight slot — counting is not a generation request.
    ///
    /// # Errors
    /// Returns an error if the engine thread is gone (channel closed or reply
    /// dropped) or the tokenizer itself fails.
    pub(crate) async fn count_tokens(&self, text: String) -> anyhow::Result<usize> {
        let (reply, reply_rx) = oneshot::channel();
        self.tx
            .send(VlmCommand::CountTokens { text, reply })
            .await
            .map_err(|_| anyhow::anyhow!("VLM engine unavailable"))?;
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("engine dropped count_tokens reply"))?
            .map_err(|e| anyhow::anyhow!(e))
    }
}

/// The VLM is a [`ModelKind::Vision`] engine. This is the `ManagedEngine`
/// facade `ModelManager` reads for metrics/lifecycle, uniform with the CB
/// engine's impl in `cb_engine.rs`.
impl ManagedEngine for VlmHandle {
    fn kind(&self) -> ModelKind {
        ModelKind::Vision
    }

    /// Accepted-but-unfinished requests (queued + the one generating) —
    /// derived from the semaphore: `cap - available_permits()`. Real
    /// occupancy, not an approximation: a permit is held for exactly the span
    /// a request is admitted, from before it is sent to the engine thread
    /// until that thread finishes the generation. `VLMPipeline` itself still
    /// processes exactly one request at a time — this counts *admitted*
    /// requests, matching what `max_concurrent_streams` has always meant for
    /// VLM (queue depth), not "executing on GPU right now" (always ≤ 1).
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

    fn oldest_generation_age(&self) -> Option<Duration> {
        self.clock.oldest_age()
    }
}

// ── Engine thread ─────────────────────────────────────────────────────────────

/// Spawn a dedicated OS thread owning `engine` and return a [`VlmHandle`].
///
/// The thread exits cleanly when all handle clones are dropped (channel closes).
///
/// # Errors
/// Only propagates thread-spawn errors (rare OS resource exhaustion).
pub fn spawn_vlm_engine(
    engine: OvVlmEngine,
    model_id: &str,
    device: &str,
    cap: Option<usize>,
    queue_timeout: Duration,
) -> anyhow::Result<(VlmHandle, std::thread::JoinHandle<()>)> {
    let cap = cap.unwrap_or(VLM_CHANNEL_CAP);
    let (tx, mut rx) = tokio::sync::mpsc::channel::<VlmCommand>(VLM_COMMAND_CAPACITY);
    let sem = Arc::new(Semaphore::new(cap));
    let model_id_arc: Arc<str> = Arc::from(model_id);
    let model_id_owned = model_id.to_owned();
    let device = device.to_owned();

    // Resolve the metric label handles once, off the per-token path — the
    // VLM's `Vision`-kind analogue of the cb-engine's `HotMetrics` (R2: VLMs
    // were previously invisible to the per-token metric family).
    let metrics = HotMetrics::new(ModelKind::Vision, model_id, &device);

    let thread = std::thread::Builder::new()
        .name(format!("vlm-engine-{model_id_owned}"))
        .spawn(move || {
            tracing::info!(
                model_id = model_id_owned,
                kind = ModelKind::Vision.label(),
                device,
                "VLM engine thread started"
            );
            while let Some(cmd) = rx.blocking_recv() {
                match cmd {
                    VlmCommand::Generate {
                        messages,
                        images,
                        tools,
                        extra_context,
                        params,
                        token_tx,
                        started_at,
                        permit,
                        ticket,
                    } => {
                        run_generate(
                            &engine,
                            &model_id_owned,
                            &messages,
                            &images,
                            tools.as_deref(),
                            extra_context.as_ref(),
                            &params,
                            &token_tx,
                            &metrics,
                            started_at,
                        );
                        // Balance the handle's admission acquire, whatever the
                        // generation outcome (success or error both finish
                        // here) — releases the slot for the next waiting caller.
                        // `ticket` clears this generation's watchdog entry the
                        // same way — see `GenerationClock`.
                        drop(permit);
                        drop(ticket);
                    }
                    // Processed inline between generations, same as the CB
                    // engine's Tokenize command — encoding is fast against a
                    // generation call, and the reply receiver may already be
                    // gone (caller cancelled / disconnected); ignore that.
                    VlmCommand::CountTokens { text, reply } => {
                        let _ = reply.send(engine.count_tokens(&text).map_err(|e| e.to_string()));
                    }
                }
            }
            tracing::info!(
                model_id = model_id_owned,
                kind = ModelKind::Vision.label(),
                "VLM engine thread exiting — all handles dropped"
            );
        })?;

    Ok((
        VlmHandle {
            tx,
            model_id: model_id_arc,
            sem,
            cap,
            queue_timeout,
            waiting: Arc::new(AtomicUsize::new(0)),
            clock: GenerationClock::default(),
        },
        thread,
    ))
}

/// Spawn a GPU-free mock VLM engine — the `Vision` analogue of
/// [`crate::cb_engine::EngineHandle::from_sender`]'s draining thread, used by
/// `MockEngineFactory` and routing tests.
///
/// The thread shares the handle's in-flight counter (so `active()` is honest)
/// and answers each `Generate` with one token + `Done(Stop)`, so a buffered
/// `collect_completion` yields a valid response without touching a GPU. When
/// the command carries `tools`, it emits a `<tool_call>` block instead of the
/// plain `"mock-vlm"` token — lets a routing test assert `tools` actually
/// reached the engine (real VLM tool-call plumbing) without a GPU.
#[cfg(test)]
pub(crate) fn spawn_mock_vlm(model_id: &str) -> (VlmHandle, std::thread::JoinHandle<()>) {
    use crate::ov_cb::FinishReason;

    // Real production default (nothing threads a config-driven cap into the
    // mock factory) — matches what `spawn_vlm_engine` falls back to.
    let cap = VLM_CHANNEL_CAP;
    let (tx, mut rx) = tokio::sync::mpsc::channel::<VlmCommand>(VLM_COMMAND_CAPACITY);
    let sem = Arc::new(Semaphore::new(cap));

    let thread = std::thread::spawn(move || {
        while let Some(cmd) = rx.blocking_recv() {
            match cmd {
                VlmCommand::Generate {
                    tools,
                    token_tx,
                    permit,
                    ..
                } => {
                    let text = if tools.is_some_and(|t| !t.is_empty()) {
                        r#"<tool_call>{"name": "mock_tool", "arguments": {}}</tool_call>"#
                    } else {
                        "mock-vlm"
                    };
                    let _ = token_tx.blocking_send(StreamEvent::Token(text.to_owned(), 1));
                    let _ = token_tx.blocking_send(StreamEvent::Done(FinishReason::Stop));
                    drop(permit);
                }
                // Deterministic word-count stand-in — no real tokenizer in
                // mock mode. Good enough for routing/gate tests, which only
                // care about crossing a threshold, not an exact count.
                VlmCommand::CountTokens { text, reply } => {
                    let _ = reply.send(Ok(text.split_whitespace().count()));
                }
            }
        }
    });

    (
        VlmHandle {
            tx,
            model_id: Arc::from(model_id),
            sem,
            cap,
            queue_timeout: Duration::from_secs(30),
            waiting: Arc::new(AtomicUsize::new(0)),
            clock: GenerationClock::default(),
        },
        thread,
    )
}

/// Drive one generation request on the engine thread.
///
/// Event ordering: Token… → PromptTokens(n) → Done(finish).
/// `PromptTokens` comes after all `Token` events but before `Done` so that
/// `ScanState` (SSE path) sees it before emitting the usage chunk.
///
/// Per-token metrics mirror the cb-engine's emission so a VLM shows up in the
/// same `rustedvino_*` families (now carrying `kind="vision"`): one
/// `request_accepted`, one `token_generated` per token, `ttft` on the first
/// token, and `duration` + a per-generation `tokens_per_second` on finish.
/// `VLMPipeline` is single-stream, so tok/s is simply this request's
/// tokens ÷ wall time — no cross-request EMA needed.
#[allow(clippy::too_many_arguments)]
fn run_generate(
    engine: &OvVlmEngine,
    model_id: &str,
    messages: &[serde_json::Value],
    images: &[DecodedImage],
    tools: Option<&[serde_json::Value]>,
    extra_context: Option<&serde_json::Value>,
    params: &GenParams,
    token_tx: &TokenSender,
    metrics: &HotMetrics,
    started_at: Instant,
) {
    // A VLM serves both text-only and image-bearing requests; the per-request
    // `modality` label records which this was (a text-only query to a VLM is
    // still served on the vision engine — `kind="vision"` alone can't tell them
    // apart). Determined from the decoded image vector already in hand.
    let modality = if images.is_empty() {
        Modality::Text
    } else {
        Modality::Multimodal
    };
    metrics.request_accepted(modality);
    let mut completion_tokens: u64 = 0;
    let mut first_token_seen = false;
    // Set only inside the two backpressure/disconnect branches below — i.e.
    // only ever reachable AFTER at least one real token already streamed
    // (`completion_tokens += 1` runs before this match on every invocation).
    // Kept as a defensive guard on the wedge check below rather than relied
    // on to distinguish anything today: a genuine wedge (never-invoked
    // closure) cannot set this flag, so it can never mask one.
    let mut aborted_early = false;

    let result = engine.generate(messages, images, tools, extra_context, params, |text| {
        if !first_token_seen {
            first_token_seen = true;
            metrics.record_ttft(modality, started_at.elapsed().as_secs_f64());
        }
        completion_tokens += 1;
        metrics.token_generated(1);
        // `try_send`, never a blocking send (T1.1): this VLM engine thread
        // serves its queue sequentially, so blocking on one stalled client's
        // full channel would wedge the thread mid-generate and park every
        // queued VLM request behind it — network-triggerable by reading an
        // SSE stream slowly. Same backpressure policy as the CB engine:
        // `Full` means the client is not keeping up → returning `true` stops
        // this generation early; `Closed` is the existing disconnect path.
        match token_tx.try_send(StreamEvent::Token(text.to_owned(), 1)) {
            Ok(()) => false,
            Err(TrySendError::Full(_)) => {
                tracing::warn!(
                    "VLM client not keeping up (token channel full) — stopping its generation"
                );
                aborted_early = true;
                true
            }
            Err(TrySendError::Closed(_)) => {
                aborted_early = true;
                true
            }
        }
    });

    // Record end-to-end duration + per-generation throughput regardless of
    // outcome (an error after N tokens still consumed N tokens of wall time).
    let elapsed = started_at.elapsed().as_secs_f64();
    metrics.record_duration(modality, elapsed);
    if elapsed > 0.0 {
        #[allow(clippy::cast_precision_loss)] // token counts are far below f64's exact range
        metrics.set_tokens_per_second(completion_tokens as f64 / elapsed);
    }

    match result {
        Ok((finish, prompt_tokens, generated_tokens, inference_ms)) => {
            tracing::debug!(
                model_id,
                finish = finish.as_openai(),
                prompt_tokens,
                completion_tokens,
                "VLM generation complete"
            );
            // KV-admission wedge detection
            // (the project's internal engineering log): `generated_tokens`
            // comes from the pipeline's OWN `perf_metrics`, independent of the
            // streamer callback above — the callback never receives the EOS
            // token itself, so a genuine single-token EOS decode reports
            // `completion_tokens == 0` but `generated_tokens == 1` (a real
            // decode step ran). `generated_tokens == 0` means the pipeline
            // never actually decoded anything — on a hybrid attention model
            // (Qwen3.5/3.6) that crossed its real (formula-overestimated) KV
            // capacity, this is permanent: every subsequent call to this
            // pipeline repeats it with a frozen `perf_metrics` reading. Two
            // guards against a false positive: `aborted_early` (the closure
            // requested an early stop — can only fire after a real token
            // already streamed, so cannot coincide with `generated_tokens ==
            // 0`, but kept as a defensive belt-and-suspenders check) and
            // `max_new_tokens == 0` (the client deliberately asked for no
            // completion — not a wedge).
            let is_kv_wedge = generated_tokens == 0 && !aborted_early && params.max_new_tokens != 0;
            if is_kv_wedge {
                tracing::error!(
                    model_id,
                    finish = finish.as_openai(),
                    prompt_tokens,
                    inference_ms,
                    message_count = messages.len(),
                    max_new_tokens = params.max_new_tokens,
                    tools_count = tools.map_or(0, <[serde_json::Value]>::len),
                    "VLM generation produced ZERO tokens internally — KV-admission capacity wedge"
                );
                // Error then a terminal Done — replacing the success path
                // below, not appending to it (mirrors the `Err(e)` arm and
                // the CB engine's pool-exhausted precedent): the collectors
                // treat the first terminal event as authoritative, so a
                // trailing "successful" Done here would silently win and the
                // wedge would never surface to the client.
                let _ = token_tx.try_send(StreamEvent::Error(format!(
                    "{VLM_KV_WEDGE_MARKER} — observed_prompt_tokens={prompt_tokens}"
                )));
                let _ = token_tx.try_send(StreamEvent::Done(FinishReason::Stop));
                return;
            }
            // PromptTokens must arrive before Done so the SSE ScanState
            // has the count when it emits the usage chunk on Done.
            // try_send, best-effort: never block this engine thread on a
            // client channel (a stalled client's stream closes when the
            // sender drops at the end of this generation either way).
            if prompt_tokens > 0 {
                let _ = token_tx.try_send(StreamEvent::PromptTokens(prompt_tokens));
            }
            let _ = token_tx.try_send(StreamEvent::Done(finish));
        }
        Err(e) => {
            tracing::error!(model_id, error = %e, "VLM generation failed");
            // Error then a terminal Done, mirroring the CB engine (T3.2):
            // every consumer — streaming or collecting — sees a terminal
            // event, never a stream that just stops.
            let _ = token_tx.try_send(StreamEvent::Error(e.to_string()));
            let _ = token_tx.try_send(StreamEvent::Done(FinishReason::Stop));
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
    use crate::streaming::stream_channel;

    /// A fresh handle reports zero active/waiting requests and its configured cap.
    #[test]
    fn vlm_handle_accessors_report_active_and_max() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<VlmCommand>(4);
        let handle = VlmHandle::from_sender(tx, "test-vlm");
        assert_eq!(handle.active(), 0, "fresh handle has no active requests");
        assert_eq!(handle.waiting(), 0, "fresh handle has no parked callers");
        assert_eq!(
            handle.max_concurrency(),
            1000,
            "test handle uses the test cap"
        );
    }

    /// The VLM engine has a real KV pool internally but no public `OpenVINO`
    /// `GenAI` API to query it — it must NOT claim `cache_usage_supported`,
    /// so `rustedvino_kv_cache_usage_percent`'s permanent `0.0` for this kind
    /// is correctly flagged as unmeasurable, not mistaken for a genuinely
    /// empty pool (the project's internal engineering log).
    #[test]
    fn vlm_handle_does_not_claim_cache_usage_is_supported() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<VlmCommand>(4);
        let handle = VlmHandle::from_sender(tx, "test-vlm");
        assert!(!ManagedEngine::cache_usage_supported(&handle));
    }

    /// A ticket's `Drop` clears its entry regardless of *why* it dropped —
    /// covers both the "engine finished" shape (ticket dropped explicitly)
    /// and the "caller cancelled" shape (ticket dropped via scope exit,
    /// mirroring what happens when `generate()`'s `.send().await` is torn
    /// down mid-poll). Watchdog correctness (2026-07-28 incident,
    /// the project's internal engineering log) depends on this: a
    /// leaked entry would eventually make the supervisor kill a healthy
    /// worker.
    #[test]
    fn generation_clock_reflects_only_still_held_tickets() {
        let clock = GenerationClock::default();
        assert_eq!(clock.oldest_age(), None, "idle clock tracks nothing");

        let ticket = clock.start();
        assert!(
            clock.oldest_age().is_some(),
            "a held ticket must be visible"
        );

        drop(ticket);
        assert_eq!(
            clock.oldest_age(),
            None,
            "a dropped ticket must not linger in the table"
        );
    }

    /// With two overlapping generations, the clock reports the *older* one's
    /// age — the watchdog cares about the longest-stuck request, not the
    /// most recent.
    #[test]
    fn generation_clock_reports_the_oldest_of_several_tickets() {
        let clock = GenerationClock::default();
        let older = clock.start();
        std::thread::sleep(Duration::from_millis(20));
        let _newer = clock.start();

        let oldest = clock.oldest_age().unwrap();
        assert!(
            oldest >= Duration::from_millis(20),
            "must report at least the older ticket's true age, got {oldest:?}"
        );

        drop(older);
        assert!(
            clock.oldest_age().is_some(),
            "the newer ticket is still held"
        );
    }

    /// `tools` survives the trip from `VlmHandle::generate()`'s call site to
    /// the `VlmCommand::Generate` the engine thread receives — regression
    /// guard for the VLM tool-calling wiring (easy to silently drop a field
    /// when threading something new through an mpsc command enum).
    #[tokio::test]
    async fn generate_threads_tools_through_the_command_channel() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<VlmCommand>(4);
        let handle = VlmHandle::from_sender(tx, "test-vlm");
        let (tok_tx, _tok_rx) = stream_channel();

        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {"name": "search_files", "parameters": {}}
        })];
        let result = handle
            .generate(
                vec![],
                vec![],
                Some(tools.clone()),
                None,
                GenParams::default(),
                tok_tx,
                Instant::now(),
            )
            .await;
        assert_eq!(result, SubmitResult::Submitted);

        let VlmCommand::Generate { tools: got, .. } = rx.recv().await.unwrap() else {
            panic!("expected a Generate command");
        };
        assert_eq!(got, Some(tools));
    }

    /// When a slot opens (permit released), a waiting `generate` completes
    /// instead of returning `AtCapacity`.
    #[tokio::test]
    async fn generate_waits_and_succeeds_when_slot_opens() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<VlmCommand>(4);
        // cap=1, 500 ms timeout — enough for the test, tight enough to be fast.
        let handle = VlmHandle::from_sender_with_cap(tx, "test-vlm", 1, 500);

        let (tok_tx, _tok_rx) = stream_channel();
        let result1 = handle
            .generate(
                vec![],
                vec![],
                None,
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
            tokio::time::sleep(Duration::from_millis(50)).await;
            drop(first_cmd); // drops the embedded OwnedSemaphorePermit → slot returned
        });

        // Second request should wait and then succeed.
        let (tok_tx2, _tok_rx2) = stream_channel();
        let result2 = handle
            .generate(
                vec![],
                vec![],
                None,
                None,
                GenParams::default(),
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
    /// `AtCapacity` (→ HTTP 429) rather than waiting forever — the fix for
    /// the 2026-07-14 finding (the project's internal engineering log):
    /// previously this awaited indefinitely on the bounded channel send.
    #[tokio::test]
    async fn generate_returns_at_capacity_after_timeout() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<VlmCommand>(4);
        // cap=1, 50 ms timeout — short so the test doesn't slow the suite.
        let handle = VlmHandle::from_sender_with_cap(tx, "test-vlm", 1, 50);

        let (tok_tx, _) = stream_channel();
        // First request gets the slot; the command stays in the channel buffer
        // (nobody drains it), so the permit stays held.
        let _ = handle
            .generate(
                vec![],
                vec![],
                None,
                None,
                GenParams::default(),
                tok_tx,
                Instant::now(),
            )
            .await;

        let (tok_tx2, _) = stream_channel();
        let result = handle
            .generate(
                vec![],
                vec![],
                None,
                None,
                GenParams::default(),
                tok_tx2,
                Instant::now(),
            )
            .await;
        assert_eq!(result, SubmitResult::AtCapacity, "must 429 after timeout");
    }

    /// `waiting()` counts exactly the callers currently parked at the
    /// admission gate: zero when idle, one while a second caller blocks on a
    /// full semaphore, back to zero once a slot frees and that caller proceeds.
    #[tokio::test]
    async fn waiting_counts_callers_parked_at_the_admission_gate() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<VlmCommand>(4);
        // cap=1, 1 s timeout — long enough that the parked caller stays parked
        // while we observe it, short enough not to hang the suite on failure.
        let handle = VlmHandle::from_sender_with_cap(tx, "test-vlm", 1, 1000);
        assert_eq!(handle.waiting(), 0, "no parked callers initially");

        // First request takes the only slot; its permit rides inside the
        // buffered command we hold, so the slot stays occupied.
        let (tok_tx, _r) = stream_channel();
        assert_eq!(
            handle
                .generate(
                    vec![],
                    vec![],
                    None,
                    None,
                    GenParams::default(),
                    tok_tx,
                    Instant::now()
                )
                .await,
            SubmitResult::Submitted
        );
        assert_eq!(
            handle.active(),
            1,
            "one request admitted and occupying the slot"
        );

        // Second caller blocks at the admission gate — spawn it and poll `waiting()`.
        let handle2 = handle.clone();
        let parked = tokio::spawn(async move {
            let (tok_tx2, _r2) = stream_channel();
            handle2
                .generate(
                    vec![],
                    vec![],
                    None,
                    None,
                    GenParams::default(),
                    tok_tx2,
                    Instant::now(),
                )
                .await
        });

        // Give the spawned task a moment to reach the semaphore wait.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(handle.waiting(), 1, "second caller is parked at the gate");

        // Free the slot — the parked caller should be admitted.
        let first_cmd = rx.recv().await.unwrap();
        drop(first_cmd);

        let result = parked.await.unwrap();
        assert_eq!(
            result,
            SubmitResult::Submitted,
            "parked caller gets the freed slot"
        );
        assert_eq!(handle.waiting(), 0, "no callers parked once admitted");
    }

    /// Cancelling `generate` while it is only *waiting for a permit* (not yet
    /// admitted) must not leak `waiting()` — regression test for a gap found
    /// in `embed_engine.rs`'s test suite (2026-07-14): a bare
    /// `fetch_add`/`fetch_sub` around the acquire `.await` never runs its
    /// `fetch_sub` if the awaiting future is dropped mid-wait (an HTTP client
    /// disconnecting while queued). Fixed via a block-scoped `InFlightGuard`.
    #[tokio::test]
    async fn cancelling_generate_while_parked_does_not_leak_waiting() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<VlmCommand>(4);
        let handle = VlmHandle::from_sender_with_cap(tx, "test-vlm", 1, 5_000);

        // Take the only permit directly so the next `generate()` call parks.
        let held = Arc::clone(&handle.sem).try_acquire_owned().unwrap();
        assert_eq!(handle.active(), 1);

        let handle2 = handle.clone();
        let task = tokio::spawn(async move {
            let (tok_tx, _r) = stream_channel();
            handle2
                .generate(
                    vec![],
                    vec![],
                    None,
                    None,
                    GenParams::default(),
                    tok_tx,
                    Instant::now(),
                )
                .await
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
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

    /// A closed channel (engine thread gone) reports `EngineDead`, not
    /// `AtCapacity` — the caller should respond 503, not 429.
    #[tokio::test]
    async fn generate_returns_engine_dead_when_channel_closed() {
        let (tx, rx) = tokio::sync::mpsc::channel::<VlmCommand>(4);
        drop(rx);
        let handle = VlmHandle::from_sender(tx, "dead-vlm");

        let (tok_tx, _) = stream_channel();
        let result = handle
            .generate(
                vec![],
                vec![],
                None,
                None,
                GenParams::default(),
                tok_tx,
                Instant::now(),
            )
            .await;
        assert_eq!(result, SubmitResult::EngineDead);
    }
}
