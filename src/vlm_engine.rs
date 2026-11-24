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
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
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
        /// Admission-gate permit, held until the generation call returns. Dropped
        /// by `finish_vlm_generation` before the terminal `Done` / `Error` is
        /// sent (0.9.1 C29), releasing the slot for the next waiting caller.
        /// See [`VlmHandle::generate`].
        permit: OwnedSemaphorePermit,
        /// Watchdog ticket — dropped alongside `permit`, clearing this
        /// generation's entry in the [`GenerationClock`]. See
        /// [`VlmHandle::generate`].
        ticket: GenerationTicket,
        /// Resources the request reserved that must outlive the HTTP request: dropped by the
        /// engine thread when the generation call returns, not when the client goes away (a
        /// disconnect does not stop a prefill already running on the device). Today: the
        /// transient device-memory lease of a long prompt.
        lease: Option<Box<dyn std::any::Any + Send>>,
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
    /// Encode `text` to token ids on the engine thread (`/tokenize`, D-09).
    Tokenize {
        /// Text to encode.
        text: String,
        /// Where the engine thread sends the ids (or a stringified error).
        reply: oneshot::Sender<Result<Vec<i64>, String>>,
    },
    /// Decode token ids back to text on the engine thread (`/detokenize`, D-09).
    Detokenize {
        /// Ids to decode.
        tokens: Vec<i64>,
        /// Where the engine thread sends the text (or a stringified error).
        reply: oneshot::Sender<Result<String, String>>,
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
pub(crate) struct GenerationClock(Arc<Mutex<ClockState>>);

/// One in-flight generation in the [`GenerationClock`].
#[derive(Debug)]
struct ClockEntry {
    /// When the caller took the ticket (queued behind whatever the engine thread is running).
    queued_at: Instant,
    /// How long it may run once the engine thread has started it ([`generation_allowance`]).
    allowance: Duration,
    /// When the engine thread started it; `None` while it still waits in the command queue.
    running_since: Option<Instant>,
}

/// The table plus the last moment the engine thread made progress (started or finished a
/// generation).
#[derive(Debug)]
struct ClockState {
    entries: HashMap<u64, ClockEntry>,
    last_progress: Instant,
}

impl Default for ClockState {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            last_progress: Instant::now(),
        }
    }
}

static NEXT_GENERATION_ID: AtomicU64 = AtomicU64::new(0);

/// Prefill speed the watchdog assumes at least, tokens per second. The 27B vision model measured
/// about 375 tok/s on the B60 (92k tokens in ~245 s); half of that is the floor.
pub(crate) const WATCHDOG_PREFILL_FLOOR_TOK_S: u64 = 200;
/// Decode speed the watchdog assumes at least, tokens per second. The 27B measured 13-18 tok/s.
pub(crate) const WATCHDOG_DECODE_FLOOR_TOK_S: u64 = 5;

/// How long a generation may take before the watchdog starts counting it as overdue: the prompt
/// at the prefill floor plus the answer budget at the decode floor. A fixed ceiling cannot serve
/// a 5-token chat and a 92k-token prompt with a 6k-token answer alike (Block B2a: the second
/// took over 600 s and was killed by a ceiling that was never meant for it).
#[must_use]
pub(crate) fn generation_allowance(prompt_tokens: usize, max_new_tokens: usize) -> Duration {
    let secs = (prompt_tokens as u64) / WATCHDOG_PREFILL_FLOOR_TOK_S
        + (max_new_tokens as u64) / WATCHDOG_DECODE_FLOOR_TOK_S;
    Duration::from_secs(secs)
}

impl GenerationClock {
    /// Starts tracking a new generation, returning the ticket that must be
    /// held (directly or via the command it travels inside) for the
    /// generation's full duration.
    ///
    /// `allowance` is how long this generation may reasonably take once it runs (see
    /// [`generation_allowance`]); the watchdog only counts time beyond it, and only from the
    /// moment the engine thread calls [`GenerationTicket::begin`].
    fn start(&self, allowance: Duration) -> GenerationTicket {
        let id = NEXT_GENERATION_ID.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut state) = self.0.lock() {
            state.entries.insert(
                id,
                ClockEntry {
                    queued_at: Instant::now(),
                    allowance,
                    running_since: None,
                },
            );
        }
        GenerationTicket {
            state: Arc::clone(&self.0),
            id,
        }
    }

    /// How far past its allowance the worst generation is: zero while every one is inside it,
    /// `None` when nothing is in flight. The supervisor compares this with
    /// `watchdog_hang_ceiling_secs`, so a request's ceiling is its allowance plus that slack.
    ///
    /// The engine thread runs one generation at a time. A **running** generation is measured from
    /// the moment it started. **Queued** ones wait behind it and are not counted while one runs
    /// (the running one's own deadline covers a wedged thread, and a short request queued behind a
    /// long one must not inherit the long one's time); when nothing is running, a queued one is
    /// measured from the later of its own queueing and the engine thread's last progress, so a
    /// thread that never picks anything up is still caught.
    fn oldest_age(&self) -> Option<Duration> {
        let now = Instant::now();
        let state = self.0.lock().ok()?;
        if state.entries.is_empty() {
            return None;
        }
        let running = state
            .entries
            .values()
            .filter_map(|e| {
                e.running_since
                    .map(|since| now.duration_since(since).saturating_sub(e.allowance))
            })
            .max();
        if running.is_some() {
            return running;
        }
        state
            .entries
            .values()
            .map(|e| now.duration_since(e.queued_at.max(state.last_progress)))
            .max()
    }
}

/// RAII handle for one [`GenerationClock`] entry — see the clock's doc for why
/// the Drop-based removal (not a statement after some await) is load-bearing.
pub(crate) struct GenerationTicket {
    state: Arc<Mutex<ClockState>>,
    id: u64,
}

impl GenerationTicket {
    /// The engine thread starts this generation now: its allowance runs from here.
    fn begin(&self) {
        if let Ok(mut state) = self.state.lock() {
            let now = Instant::now();
            state.last_progress = now;
            if let Some(e) = state.entries.get_mut(&self.id) {
                e.running_since = Some(now);
            }
        }
    }
}

impl Drop for GenerationTicket {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            state.entries.remove(&self.id);
            state.last_progress = Instant::now();
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
    /// Set (`SeqCst`, never cleared) when an eviction of this engine has committed — forced, or
    /// non-forced past the busy check — as opposed to the transient `closing` a refused
    /// non-forced eviction raises and clears. [`VlmSlot::submit`] checks this one (review F7):
    /// a held slot counts in `active()`, so any eviction after `acquire_slot` either refuses
    /// (busy) or commits.
    committed: Arc<AtomicBool>,
    /// Standalone tokenizer (D-13): `/tokenize` and `/detokenize` call it directly from a blocking
    /// thread instead of queueing behind the single-stream engine thread. `None` (mock engines, or
    /// a tokenizer that failed to load) falls back to the engine-thread commands.
    tokenizer: Option<Arc<crate::ov_tokenizer::OvTokenizer>>,
    /// Bounds how many standalone-tokenizer calls occupy blocking-pool threads at once. The C++
    /// tokenizer serialises on one mutex anyway, so without this a flood of `/tokenize` requests
    /// would park hundreds of OS threads (shared with every other `OpenVINO` call) behind that lock.
    tok_gate: Arc<Semaphore>,
}

/// Standalone-tokenizer calls allowed in flight at once per vision model (the rest wait async).
const VLM_TOKENIZER_CONCURRENCY: usize = 4;

impl VlmHandle {
    /// This engine instance's liveness (0.9.1 S1 stall watchdog).
    #[must_use]
    pub fn liveness(&self) -> &crate::stall_watch::EngineLiveness {
        &self.liveness
    }

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
            closing: Arc::new(AtomicBool::new(false)),
            liveness: crate::stall_watch::EngineLiveness::new(),
            committed: Arc::new(AtomicBool::new(false)),
            tokenizer: None,
            tok_gate: Arc::new(Semaphore::new(VLM_TOKENIZER_CONCURRENCY)),
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
            closing: Arc::new(AtomicBool::new(false)),
            liveness: crate::stall_watch::EngineLiveness::new(),
            committed: Arc::new(AtomicBool::new(false)),
            tokenizer: None,
            tok_gate: Arc::new(Semaphore::new(VLM_TOKENIZER_CONCURRENCY)),
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
        self.generate_with_lease(
            messages,
            images,
            tools,
            extra_context,
            params,
            token_tx,
            started_at,
            None,
            0,
        )
        .await
    }

    /// [`Self::generate`] that hands `lease` to the engine thread, which drops it when the
    /// generation call returns (success, error or a client that is long gone). On every early
    /// return here (`AtCapacity`, `Closing`, `EngineDead`) it drops with this call.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn generate_with_lease(
        &self,
        messages: Vec<serde_json::Value>,
        images: Vec<DecodedImage>,
        tools: Option<Vec<serde_json::Value>>,
        extra_context: Option<serde_json::Value>,
        params: GenParams,
        token_tx: TokenSender,
        started_at: Instant,
        lease: Option<Box<dyn std::any::Any + Send>>,
        prompt_tokens: usize,
    ) -> SubmitResult {
        match self.acquire_slot().await {
            Ok(slot) => {
                slot.submit(
                    messages,
                    images,
                    tools,
                    extra_context,
                    params,
                    token_tx,
                    started_at,
                    lease,
                    prompt_tokens,
                )
                .await
            }
            // 0.9.1 S1: the engine stalled while this caller waited — answered on its own
            // channel, like the requests the verdict reached (see `cb_engine`'s `add_request`).
            Err(SubmitResult::EngineDead) if self.liveness.is_dead() => {
                crate::stall_watch::send_stalled_pair_now(&token_tx);
                SubmitResult::Submitted
            }
            Err(refused) => refused,
        }
    }

    /// The waiting half of [`Self::generate_with_lease`] (0.9.1 S5c): wait up to `queue_timeout`
    /// for an admission permit and return it as a [`VlmSlot`], or the [`SubmitResult`] to answer
    /// with (`AtCapacity`, `Closing`, `EngineDead`). The chat handler decodes its images only
    /// after this returns, so a request parked here holds compressed bytes, not decoded pixels.
    ///
    /// # Errors
    /// The refusal, as the [`SubmitResult`] the caller maps to a response (never `Submitted`).
    pub(crate) async fn acquire_slot(&self) -> Result<VlmSlot<'_>, SubmitResult> {
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
                // 0.9.1 S1: the engine stalled while this caller waited — the stuck thread holds
                // the slot, it will not free. `EngineDead` with the liveness dead: callers answer
                // it as `generation_stalled`.
                () = self.liveness.dead() => return Err(SubmitResult::EngineDead),
            }
        };
        let permit = match permit_result {
            Ok(Ok(p)) => p,
            Ok(Err(_)) => return Err(SubmitResult::EngineDead), // semaphore closed (shouldn't happen)
            Err(_) => return Err(SubmitResult::AtCapacity),     // timed out waiting for a permit
        };

        // The eviction may have started while this caller waited for (or just took) its permit.
        // Checked AFTER holding it: `begin_evict` sets `closing` before reading `active()`, so
        // either it counted this permit (and drains it) or this check sees the flag.
        if self.closing.load(Ordering::SeqCst) {
            drop(permit);
            return Err(SubmitResult::Closing);
        }
        Ok(VlmSlot {
            handle: self,
            permit,
        })
    }

    /// Hold every admission permit of this engine (test seam, 0.9.1 S5c): requests then park in
    /// [`Self::acquire_slot`] until the returned permit drops.
    #[cfg(test)]
    pub(crate) async fn hold_all_slots(&self) -> Option<OwnedSemaphorePermit> {
        let n = u32::try_from(self.cap).unwrap_or(u32::MAX);
        Arc::clone(&self.sem).acquire_many_owned(n).await.ok()
    }
}

/// An admission permit of one [`VlmHandle`], taken by [`VlmHandle::acquire_slot`] and not yet
/// used (0.9.1 S5c). It counts in `active()` from the moment it is taken, so an eviction that
/// starts meanwhile drains it; dropping it unused releases the slot.
pub(crate) struct VlmSlot<'a> {
    handle: &'a VlmHandle,
    permit: OwnedSemaphorePermit,
}

impl VlmSlot<'_> {
    /// The sending half of [`VlmHandle::generate_with_lease`]: start the watchdog ticket and hand
    /// the generation to the engine thread. Refuses with `Closing` when an eviction has
    /// **committed** since the slot was taken (the caller may have spent a while decoding its
    /// images). Not on the transient `closing` flag: this slot counts in `active()`, so a
    /// non-forced eviction that raises it is refused as busy and clears it again (review F7).
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn submit(
        self,
        messages: Vec<serde_json::Value>,
        images: Vec<DecodedImage>,
        tools: Option<Vec<serde_json::Value>>,
        extra_context: Option<serde_json::Value>,
        params: GenParams,
        token_tx: TokenSender,
        started_at: Instant,
        lease: Option<Box<dyn std::any::Any + Send>>,
        prompt_tokens: usize,
    ) -> SubmitResult {
        let Self { handle, permit } = self;
        if handle.committed.load(Ordering::SeqCst) {
            drop(permit);
            return SubmitResult::Closing;
        }

        // Started here, on the caller's task, before dispatch — never on the
        // engine thread, which is exactly what a wedged FFI call would leave
        // unresponsive. Travels inside the command so its `Drop` always runs:
        // engine-thread completion (alongside `permit`) or a cancelled send
        // right here (this task torn down mid-`.await`) both clear it. See
        // `GenerationClock`.
        let ticket = handle
            .clock
            .start(generation_allowance(prompt_tokens, params.max_new_tokens));

        // Permit acquired. Send to the engine — if the channel is closed, the
        // permit (moved into the command) drops here, automatically releasing
        // the slot back to the semaphore.
        let sent = handle
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
                lease,
            })
            .await;
        if sent.is_err() {
            return SubmitResult::EngineDead;
        }
        SubmitResult::Submitted
    }
}

impl VlmHandle {
    /// Count the tokens `text` encodes to with the VLM's tokenizer.
    ///
    /// Uses the standalone tokenizer on a blocking thread when there is one (as
    /// [`tokenize`](Self::tokenize) does); otherwise the tokenizer on the engine thread (its
    /// `InferRequest` is not thread-safe), reached through the command channel with a `oneshot`
    /// reply, which waits behind a running generation. It does NOT consume an in-flight slot —
    /// counting is not a generation request.
    ///
    /// # Errors
    /// Returns an error if the engine thread is gone (channel closed or reply
    /// dropped) or the tokenizer itself fails.
    pub(crate) async fn count_tokens(&self, text: String) -> anyhow::Result<usize> {
        // The standalone tokenizer, when there is one, so counting does not queue behind a running
        // generation on the engine thread (Fable review 3, S-4: every other request's admission
        // waited for the whole of a long generation).
        if let Some(tok) = &self.tokenizer {
            let tok = Arc::clone(tok);
            let _slot = self.tok_gate.acquire().await;
            return tokio::task::spawn_blocking(move || tok.encode(&text).map(|ids| ids.len()))
                .await
                .map_err(|e| anyhow::anyhow!("count_tokens task failed: {e}"))?;
        }
        let (reply, reply_rx) = oneshot::channel();
        self.liveness
            .unless_dead(self.tx.send(VlmCommand::CountTokens { text, reply }))
            .await?
            .map_err(|_| anyhow::anyhow!("VLM engine unavailable"))?;
        // 0.9.1 S1: a stuck engine thread never replies; the verdict releases the caller.
        self.liveness
            .unless_dead(reply_rx)
            .await?
            .map_err(|_| anyhow::anyhow!("engine dropped count_tokens reply"))?
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Encode `text` to token ids with the VLM's tokenizer (`/tokenize`). Uses the standalone
    /// tokenizer on a blocking thread when there is one (D-13), else the engine thread like
    /// [`count_tokens`](Self::count_tokens). Takes no admission slot either way.
    ///
    /// # Errors
    /// Returns an error if the engine thread is gone or the tokenizer fails.
    pub(crate) async fn tokenize(&self, text: String) -> anyhow::Result<Vec<i64>> {
        if let Some(tok) = &self.tokenizer {
            let tok = Arc::clone(tok);
            let _slot = self.tok_gate.acquire().await;
            return tokio::task::spawn_blocking(move || tok.encode(&text))
                .await
                .map_err(|e| anyhow::anyhow!("tokenize task failed: {e}"))?;
        }
        let (reply, reply_rx) = oneshot::channel();
        self.liveness
            .unless_dead(self.tx.send(VlmCommand::Tokenize { text, reply }))
            .await?
            .map_err(|_| anyhow::anyhow!("VLM engine unavailable"))?;
        // 0.9.1 S1: a stuck engine thread never replies; the verdict releases the caller.
        self.liveness
            .unless_dead(reply_rx)
            .await?
            .map_err(|_| anyhow::anyhow!("engine dropped tokenize reply"))?
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Decode token ids to text with the VLM's tokenizer (`/detokenize`).
    ///
    /// # Errors
    /// Returns an error if the engine thread is gone or the detokenizer fails.
    pub(crate) async fn detokenize(&self, tokens: Vec<i64>) -> anyhow::Result<String> {
        if let Some(tok) = &self.tokenizer {
            let tok = Arc::clone(tok);
            let _slot = self.tok_gate.acquire().await;
            return tokio::task::spawn_blocking(move || tok.decode(&tokens))
                .await
                .map_err(|e| anyhow::anyhow!("detokenize task failed: {e}"))?;
        }
        let (reply, reply_rx) = oneshot::channel();
        self.liveness
            .unless_dead(self.tx.send(VlmCommand::Detokenize { tokens, reply }))
            .await?
            .map_err(|_| anyhow::anyhow!("VLM engine unavailable"))?;
        // 0.9.1 S1: a stuck engine thread never replies; the verdict releases the caller.
        self.liveness
            .unless_dead(reply_rx)
            .await?
            .map_err(|_| anyhow::anyhow!("engine dropped detokenize reply"))?
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

    fn set_closing(&self, closing: bool) {
        self.closing.store(closing, Ordering::SeqCst);
    }

    fn mark_eviction_committed(&self) {
        self.committed.store(true, Ordering::SeqCst);
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
    tokenizer: Option<Arc<crate::ov_tokenizer::OvTokenizer>>,
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
    let metrics =
        HotMetrics::new(ModelKind::Vision, model_id, &device).with_kv_scheduler_drops(model_id);
    // 0.9.1 S1: shared by the handle and the engine thread, which reports its calls to it.
    let liveness = crate::stall_watch::EngineLiveness::new();
    let liveness_engine = liveness.clone();

    let thread = std::thread::Builder::new()
        .name(format!("vlm-engine-{model_id_owned}"))
        .spawn(move || {
            tracing::info!(
                model_id = model_id_owned,
                kind = ModelKind::Vision.label(),
                device,
                "VLM engine thread started"
            );
            // 0.9.1 S1: however this thread exits (shutdown, panic), it is not left busy.
            let _idle_on_exit = liveness_engine.idle_on_drop();
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
                        lease,
                    } => {
                        // The allowance runs from here, not from the queueing.
                        ticket.begin();
                        // 0.9.1 S1: and the stall watchdog's call clock (before the call).
                        liveness_engine.begin();
                        // Counted for the exit decision even after the record leaves Ready
                        // (a failed shutdown eviction resets it while this thread is still in
                        // the plugin) — see `ffi_watch::FfiKind::VlmGenerate`.
                        let ffi = crate::ffi_watch::global().arm(
                            crate::ffi_watch::FfiKind::VlmGenerate,
                            &model_id_owned,
                            None,
                            std::time::Duration::ZERO,
                        );
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
                            // Balance the handle's admission acquire, whatever the
                            // generation outcome — released by `run_generate` the
                            // moment the generation call returns, BEFORE any
                            // terminal event is sent (0.9.1 C29). `ticket` clears
                            // this generation's watchdog entry the same way — see
                            // `GenerationClock`. A tuple drops its fields in order:
                            // ffi, permit, ticket, lease (the order used before C29).
                            (ffi, permit, ticket, lease),
                            &liveness_engine,
                        );
                    }
                    // Processed inline between generations, same as the CB
                    // engine's Tokenize command — encoding is fast against a
                    // generation call, and the reply receiver may already be
                    // gone (caller cancelled / disconnected); ignore that.
                    VlmCommand::CountTokens { text, reply } => {
                        let _ = reply.send(engine.count_tokens(&text).map_err(|e| e.to_string()));
                    }
                    VlmCommand::Tokenize { text, reply } => {
                        let _ = reply.send(engine.encode(&text).map_err(|e| e.to_string()));
                    }
                    VlmCommand::Detokenize { tokens, reply } => {
                        let _ = reply.send(engine.decode(&tokens).map_err(|e| e.to_string()));
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
            closing: Arc::new(AtomicBool::new(false)),
            liveness,
            committed: Arc::new(AtomicBool::new(false)),
            tokenizer,
            tok_gate: Arc::new(Semaphore::new(VLM_TOKENIZER_CONCURRENCY)),
        },
        thread,
    ))
}

/// Marker text: a mock VLM request whose messages contain it fails with this
/// text as its `Error` (test-only; no routing test can hit it by accident).
/// The `(width, height)` of the images in each `Generate` the mock VLM engines received (0.9.1
/// C28 tests: what actually reached the engine after the handler's decode and downscale).
#[cfg(test)]
static MOCK_SEEN_IMAGE_DIMS: std::sync::Mutex<Vec<Vec<(u32, u32)>>> =
    std::sync::Mutex::new(Vec::new());

/// A copy of [`MOCK_SEEN_IMAGE_DIMS`]: every mock `Generate`'s image sizes so far, process-wide.
#[cfg(test)]
pub(crate) fn mock_seen_image_dims() -> Vec<Vec<(u32, u32)>> {
    MOCK_SEEN_IMAGE_DIMS
        .lock()
        .map(|v| v.clone())
        .unwrap_or_default()
}

/// `(model id, params, extra context)` of every `Generate` the mock VLM engines received
/// (0.9.1 thinking-control handler tests). Process-wide and never cleared: it lives for the whole
/// test process, so every test that reads it must use a model id no other test uses.
#[cfg(test)]
pub(crate) type MockVlmSeen = (String, GenParams, Option<serde_json::Value>);

#[cfg(test)]
static MOCK_SEEN_PARAMS: std::sync::Mutex<Vec<MockVlmSeen>> = std::sync::Mutex::new(Vec::new());

/// `(model id, started_at)` of every `Generate` the mock VLM engines received: the metrics' zero
/// point (F1b: TTFT and duration start after an on-demand wait). Process-wide, like
/// [`MOCK_SEEN_PARAMS`]: read it with a model id no other test uses.
#[cfg(test)]
static MOCK_SEEN_STARTED_AT: std::sync::Mutex<Vec<(String, Instant)>> =
    std::sync::Mutex::new(Vec::new());

/// The `started_at` of each mock VLM `Generate` received so far for `model_id`.
#[cfg(test)]
pub(crate) fn mock_seen_started_at(model_id: &str) -> Vec<Instant> {
    MOCK_SEEN_STARTED_AT
        .lock()
        .map(|v| v.iter().filter(|s| s.0 == model_id).map(|s| s.1).collect())
        .unwrap_or_default()
}

/// The mock VLM `Generate`s received so far for `model_id`.
#[cfg(test)]
pub(crate) fn mock_seen_params(model_id: &str) -> Vec<MockVlmSeen> {
    MOCK_SEEN_PARAMS
        .lock()
        .map(|v| v.iter().filter(|s| s.0 == model_id).cloned().collect())
        .unwrap_or_default()
}

#[cfg(test)]
pub(crate) const MOCK_VLM_GENERATE_ERROR: &str = "__mock_vlm_generate_error__";

/// A message containing this marker (with `tools` present) makes the mock VLM
/// answer with a Qwen XML tool call carrying Python literals (`True`, `None`)
/// and a numeric-looking string, as Qwen3.5/3.6 do (QXL R4).
#[cfg(test)]
pub(crate) const MOCK_VLM_QWEN_XML_TOOL: &str = "__mock_vlm_qwen_xml_tool__";

/// The Qwen XML answer [`MOCK_VLM_QWEN_XML_TOOL`] selects.
#[cfg(test)]
const MOCK_VLM_QWEN_XML_ANSWER: &str = "<tool_call>\n<function=get_forecast>\n\
    <parameter=city>\nHanoi\n</parameter>\n<parameter=metric>\nTrue\n</parameter>\n\
    <parameter=note>\nNone\n</parameter>\n<parameter=zip>\n10000\n</parameter>\n\
    </function>\n</tool_call>";

/// Spawn a GPU-free mock VLM engine — the `Vision` analogue of
/// [`crate::cb_engine::EngineHandle::from_sender`]'s draining thread, used by
/// `MockEngineFactory` and routing tests.
///
/// The thread shares the handle's in-flight counter (so `active()` is honest)
/// and answers each `Generate` with one token + `Done(Stop)`, so a buffered
/// `collect_completion` yields a valid response without touching a GPU. When
/// the command carries `tools`, it emits a `<tool_call>` block instead of the
/// plain `"mock-vlm"` token — lets a routing test assert `tools` actually
/// reached the engine (real VLM tool-call plumbing) without a GPU. A message
/// containing [`MOCK_VLM_GENERATE_ERROR`] gets the error terminal instead
/// (`Error` + `Done(Stop)`, no token). Both orders match the real engine: the
/// admission permit, watchdog ticket and lease are released before the
/// terminal event (0.9.1 C29).
#[cfg(test)]
pub(crate) fn spawn_mock_vlm(model_id: &str) -> (VlmHandle, std::thread::JoinHandle<()>) {
    spawn_mock_vlm_with_timeout(model_id, Duration::from_secs(30))
}

/// [`spawn_mock_vlm`] with its admission `queue_timeout` (a test that needs the
/// `AtCapacity` → 429 path without waiting 30 s).
#[cfg(test)]
#[allow(clippy::too_many_lines)] // one mock command loop; test-only
pub(crate) fn spawn_mock_vlm_with_timeout(
    model_id: &str,
    queue_timeout: Duration,
) -> (VlmHandle, std::thread::JoinHandle<()>) {
    use crate::ov_cb::FinishReason;

    // Real production default (nothing threads a config-driven cap into the
    // mock factory) — matches what `spawn_vlm_engine` falls back to.
    let cap = VLM_CHANNEL_CAP;
    let (tx, mut rx) = tokio::sync::mpsc::channel::<VlmCommand>(VLM_COMMAND_CAPACITY);
    let sem = Arc::new(Semaphore::new(cap));
    let liveness = crate::stall_watch::EngineLiveness::new();
    let liveness_engine = liveness.clone();
    let mock_id = model_id.to_owned();

    let thread = std::thread::spawn(move || {
        while let Some(cmd) = rx.blocking_recv() {
            match cmd {
                VlmCommand::Generate {
                    messages,
                    images,
                    tools,
                    token_tx,
                    permit,
                    ticket,
                    lease,
                    params,
                    extra_context,
                    started_at,
                    ..
                } => {
                    if let Ok(mut seen) = MOCK_SEEN_STARTED_AT.lock() {
                        seen.push((mock_id.clone(), started_at));
                    }
                    if let Ok(mut seen) = MOCK_SEEN_PARAMS.lock() {
                        seen.push((mock_id.clone(), *params, extra_context));
                    }
                    if let Ok(mut seen) = MOCK_SEEN_IMAGE_DIMS.lock() {
                        seen.push(images.iter().map(|i| (i.width, i.height)).collect());
                    }
                    // Released together, in the real engine's order (permit, ticket,
                    // lease), before the terminal event — as `finish_vlm_generation`.
                    let slot = (permit, ticket, lease);
                    // 0.9.1 S1: a marked request stalls (or answers) while it holds its slot.
                    let joined: String = messages.iter().map(ToString::to_string).collect();
                    if crate::stall_watch::mock::run(&joined, &token_tx, &liveness_engine) {
                        drop(slot);
                        continue;
                    }
                    let text = if tools.is_some_and(|t| !t.is_empty()) {
                        if joined.contains(MOCK_VLM_QWEN_XML_TOOL) {
                            MOCK_VLM_QWEN_XML_ANSWER
                        } else {
                            r#"<tool_call>{"name": "mock_tool", "arguments": {}}</tool_call>"#
                        }
                    } else {
                        "mock-vlm"
                    };
                    if messages
                        .iter()
                        .any(|m| m.to_string().contains(MOCK_VLM_GENERATE_ERROR))
                    {
                        // Error terminal, same order as the real engine (C29): slot
                        // released first, then Error + Done.
                        drop(slot);
                        let _ = token_tx
                            .try_send(StreamEvent::Error(MOCK_VLM_GENERATE_ERROR.to_owned()));
                        let _ = token_tx.try_send(StreamEvent::Done(FinishReason::Stop));
                        continue;
                    }
                    let _ = token_tx.blocking_send(StreamEvent::Token(text.to_owned(), 1));
                    // Same order as the real engine (C29): slot first, then Done.
                    drop(slot);
                    let _ = token_tx.blocking_send(StreamEvent::Done(FinishReason::Stop));
                }
                // Deterministic word-count stand-in — no real tokenizer in
                // mock mode. Good enough for routing/gate tests, which only
                // care about crossing a threshold, not an exact count.
                VlmCommand::CountTokens { text, reply } => {
                    let _ = reply.send(Ok(text.split_whitespace().count()));
                }
                // Mock tokenizer for /tokenize tests: one id per character (its code point), so
                // encode/decode round-trip exactly.
                VlmCommand::Tokenize { text, reply } => {
                    let _ = reply.send(Ok(text.chars().map(|c| i64::from(u32::from(c))).collect()));
                }
                VlmCommand::Detokenize { tokens, reply } => {
                    let decoded = tokens
                        .iter()
                        .enumerate()
                        .map(|(i, &t)| {
                            u32::try_from(t)
                                .ok()
                                .and_then(char::from_u32)
                                .ok_or_else(|| {
                                    format!(
                                        "{} tokens[{i}] = {t} is outside the vocabulary",
                                        crate::handlers::tokenize::INVALID_TOKEN_ID_MARKER
                                    )
                                })
                        })
                        .collect::<Result<String, String>>();
                    let _ = reply.send(decoded);
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
            queue_timeout,
            waiting: Arc::new(AtomicUsize::new(0)),
            clock: GenerationClock::default(),
            closing: Arc::new(AtomicBool::new(false)),
            liveness,
            committed: Arc::new(AtomicBool::new(false)),
            tokenizer: None,
            tok_gate: Arc::new(Semaphore::new(VLM_TOKENIZER_CONCURRENCY)),
        },
        thread,
    )
}

/// The generated-token count to report: the pipeline's own figure when it gave one (the streamer
/// callback fires once per decoded chunk, and a chunk can hold several tokens), never less than
/// the callbacks seen.
fn authoritative_token_count(pipeline_generated: Option<usize>, callbacks: u64) -> u64 {
    match pipeline_generated {
        Some(n) if n > 0 => (n as u64).max(callbacks),
        _ => callbacks,
    }
}

/// Drive one generation request on the engine thread.
///
/// Event ordering: Token… → PromptTokens(n) → CompletionTokens(n) → Done(finish).
/// `PromptTokens` comes after all `Token` events but before `Done` so that
/// `ScanState` (SSE path) sees it before emitting the usage chunk.
///
/// Per-token metrics mirror the cb-engine's emission so a VLM shows up in the
/// same `rustedvino_*` families (now carrying `kind="vision"`): one
/// `request_accepted`, one `token_generated` per token, `ttft` on the first
/// token, and `duration` + a per-generation `tokens_per_second` on finish.
/// `VLMPipeline` is single-stream, so tok/s is simply this request's
/// tokens ÷ wall time — no cross-request EMA needed.
///
/// `slot` is everything the request holds that must be released when the
/// generation is over (admission permit, watchdog ticket, FFI arm, lease). It
/// is handed to [`finish_vlm_generation`] right after the generation call
/// returns, which drops it before the metrics and before any terminal event
/// (`Done` / `Error`) — so a client woken by `Done` never finds its own
/// finished request in `active()` (0.9.1 C29, the streaming form of C2). The
/// token closure does not capture it.
#[allow(clippy::too_many_arguments)]
fn run_generate<S>(
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
    slot: S,
    liveness: &crate::stall_watch::EngineLiveness,
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
    // 0.9.1 V1-usage (P2): the engine's start of this request — the command queue is behind it,
    // image preprocessing, the template and prefill are ahead of it.
    let engine_started_at = Instant::now();
    let mut stream = VlmStream::new(model_id, modality, started_at, params);
    let result = engine.generate(messages, images, tools, extra_context, params, |text| {
        // 0.9.1 S1: a streamer callback is progress. It fires on decoded text only, so a long
        // run of tokens that decode to nothing (skipped special tokens) reads as silence — the
        // documented limit (no bridge change in 0.9.1).
        liveness.progress();
        stream.on_chunk(text, token_tx, metrics)
    });
    liveness.end();
    let VlmStream {
        first_chunk_at,
        output,
        completion_tokens,
        aborted_early,
        budget,
        ..
    } = stream;
    finish_vlm_generation(
        slot,
        result,
        &VlmGenTail {
            model_id,
            modality,
            started_at,
            completion_tokens,
            aborted_early,
            max_new_tokens: params.max_new_tokens,
            message_count: messages.len(),
            tools_count: tools.map_or(0, <[serde_json::Value]>::len),
            prefill: crate::metrics::prefill_interval(
                started_at,
                engine_started_at,
                first_chunk_at,
            ),
            output: &output,
            reasoning: params.reasoning,
            budget: crate::thinking_budget::BudgetOutcome::of(budget.as_ref(), aborted_early),
        },
        token_tx,
        metrics,
        |text: &str| engine.count_tokens(text),
    );
}

/// One VLM generation's streamer state; [`Self::on_chunk`] is the streamer callback's body (a
/// method so a test drives the real callback).
struct VlmStream<'a> {
    model_id: &'a str,
    modality: Modality,
    started_at: Instant,
    first_chunk_at: Option<Instant>,
    /// The whole output, for the reasoning recount after the generation (rule 2).
    output: String,
    completion_tokens: u64,
    /// Set only inside the two backpressure/disconnect branches of [`Self::on_chunk`] — i.e.
    /// only ever reachable AFTER at least one real token already streamed
    /// (`completion_tokens += 1` runs before that match on every invocation).
    /// Kept as a defensive guard on the wedge check rather than relied
    /// on to distinguish anything today: a genuine wedge (never-invoked
    /// closure) cannot set this flag, so it can never mask one.
    aborted_early: bool,
    /// 0.9.1 strict thinking budget. A VLM streamer chunk carries no token count, so the running
    /// count is one per chunk against the budget scaled to chunks (`chunk_budget` with
    /// `VLM_TOKENS_PER_CHUNK_PERCENT`, 1.37 tokens a chunk): the stop lands near the budget,
    /// approximately. The reported `reasoning_tokens` is
    /// the recount after the generation, as for every VLM request.
    budget: Option<crate::thinking_budget::BudgetTracker>,
}

impl<'a> VlmStream<'a> {
    fn new(model_id: &'a str, modality: Modality, started_at: Instant, params: &GenParams) -> Self {
        Self {
            model_id,
            modality,
            started_at,
            first_chunk_at: None,
            output: String::new(),
            completion_tokens: 0,
            aborted_early: false,
            budget: crate::thinking_budget::chunk_tracker_for(
                params.reasoning,
                params.thinking_budget,
                crate::thinking_budget::VLM_TOKENS_PER_CHUNK_PERCENT,
            ),
        }
    }

    /// The streamer callback for one decoded chunk: `true` stops the pipeline (this engine's
    /// convention).
    fn on_chunk(&mut self, text: &str, token_tx: &TokenSender, metrics: &HotMetrics) -> bool {
        if self.first_chunk_at.is_none() {
            self.first_chunk_at = Some(Instant::now());
            metrics.record_ttft(self.modality, self.started_at.elapsed().as_secs_f64());
        }
        self.output.push_str(text);
        self.completion_tokens += 1;
        metrics.token_generated(1);
        // `try_send`, never a blocking send (T1.1): this VLM engine thread
        // serves its queue sequentially, so blocking on one stalled client's
        // full channel would wedge the thread mid-generate and park every
        // queued VLM request behind it — network-triggerable by reading an
        // SSE stream slowly. Same backpressure policy as the CB engine:
        // `Full` means the client is not keeping up → returning `true` stops
        // this generation early; `Closed` is the existing disconnect path.
        match token_tx.try_send(StreamEvent::Token(text.to_owned(), 1)) {
            Ok(()) => vlm_budget_says_stop(self.budget.as_mut(), self.model_id, text),
            Err(TrySendError::Full(_)) => {
                tracing::warn!(
                    "VLM client not keeping up (token channel full) — stopping its generation"
                );
                self.aborted_early = true;
                true
            }
            Err(TrySendError::Closed(_)) => {
                self.aborted_early = true;
                true
            }
        }
    }
}

/// 0.9.1: feed a delivered VLM chunk to the request's budget (one per chunk); `true` when the
/// pipeline must stop (the VLM streamer convention: `true` = stop).
fn vlm_budget_says_stop(
    budget: Option<&mut crate::thinking_budget::BudgetTracker>,
    model_id: &str,
    text: &str,
) -> bool {
    let Some(b) = budget else {
        return false;
    };
    let already = b.stopped();
    let stop = b.push(text, 1) == crate::thinking_budget::BudgetStep::Stop;
    if stop && !already {
        tracing::debug!(
            model_id,
            reasoning_chunks = b.counted(),
            "VLM thinking budget spent — stopping the request (finish_reason length)"
        );
    }
    stop
}

/// What [`finish_vlm_generation`] needs from the generation besides its result:
/// the per-request facts the metrics, the wedge check and its log use.
struct VlmGenTail<'a> {
    model_id: &'a str,
    modality: Modality,
    started_at: Instant,
    /// Streamer callbacks seen (one per decoded text chunk).
    completion_tokens: u64,
    /// The token closure asked the pipeline to stop (client gone / not keeping up).
    aborted_early: bool,
    max_new_tokens: usize,
    message_count: usize,
    tools_count: usize,
    /// Engine start of the request to its first text chunk (0.9.1 V1-usage, P2); `None` when no
    /// chunk came.
    prefill: Option<Duration>,
    /// The whole streamed output (the reasoning recount's input).
    output: &'a str,
    /// The request's reasoning split.
    reasoning: crate::prompt_builder::ReasoningSplit,
    /// Where the request's thinking budget ended (0.9.1): a budget stop (or `length` inside the
    /// reasoning) finishes as [`FinishReason::ThinkingBudget`], a finished request.
    budget: crate::thinking_budget::BudgetOutcome,
}

/// The tail of a VLM generation, after the generation call returned: release
/// `slot` FIRST (admission permit, watchdog ticket, FFI arm, lease — 0.9.1 C29),
/// then record the metrics, then send the terminal events (`PromptTokens` /
/// `CompletionTokens` / `Done`, or `Error` + `Done`). A client woken by `Done`
/// or `Error` therefore never finds its own finished request in `active()`.
/// Split out of [`run_generate`] so the order is unit-tested without a model.
///
/// `count_tokens` is the model's tokenizer (engine thread), used once, after the slot is released,
/// to recount the reasoning text (`crate::reasoning_tokens`, rule 2: approximate — VLM chunks carry
/// no per-chunk token count). Its failure never fails the request (byte split + WARN).
/// Like the `CountTokens` command, the call runs outside the S1 call watch (`liveness.end()` has
/// run): a tokenizer that hung here would not be caught by the stall verdict (one short encode).
fn finish_vlm_generation<S, C>(
    slot: S,
    result: anyhow::Result<(FinishReason, usize, usize, f64)>,
    tail: &VlmGenTail<'_>,
    token_tx: &TokenSender,
    metrics: &HotMetrics,
    count_tokens: C,
) where
    C: FnOnce(&str) -> anyhow::Result<usize>,
{
    drop(slot);
    let VlmGenTail {
        model_id,
        modality,
        started_at,
        completion_tokens,
        aborted_early,
        max_new_tokens,
        message_count,
        tools_count,
        prefill,
        output,
        reasoning,
        budget,
    } = *tail;

    // Record end-to-end duration + per-generation throughput regardless of
    // outcome (an error after N tokens still consumed N tokens of wall time).
    let elapsed = started_at.elapsed().as_secs_f64();
    metrics.record_duration(modality, elapsed);
    // The streamer callback fires once per decoded text chunk and a chunk can hold several tokens
    // (Block B2c: 4,678 callbacks for 6,425 tokens), so the pipeline's own count, when it
    // reports one, is the authority for the throughput and the token metric.
    let counted_tokens = authoritative_token_count(
        result.as_ref().ok().map(|(_, _, generated, _)| *generated),
        completion_tokens,
    );
    if counted_tokens > completion_tokens {
        metrics.token_generated(counted_tokens - completion_tokens);
    }
    if elapsed > 0.0 {
        #[allow(clippy::cast_precision_loss)] // token counts are far below f64's exact range
        metrics.set_tokens_per_second(counted_tokens as f64 / elapsed);
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
            // `max_new_tokens == 0` (an unbounded budget the handler could not
            // resolve: `max_tokens_cap: 0`, no `max_tokens`, and a pool the
            // exact gate does not know — the engine decides the length, so 0
            // tokens is not read as a wedge). An explicit `max_tokens: 0` is a
            // 400 on HTTP and never gets here; under the exact gate an
            // unbounded budget arrives resolved to the pool room (P-02), so
            // wedge detection applies to it.
            // 0.9.1: a thinking-budget stop is a deliberate streamer stop too, never a wedge.
            let is_kv_wedge =
                generated_tokens == 0 && !aborted_early && !budget.stopped && max_new_tokens != 0;
            if is_kv_wedge {
                metrics.kv_scheduler_drop();
                tracing::error!(
                    model_id,
                    finish = finish.as_openai(),
                    prompt_tokens,
                    inference_ms,
                    message_count,
                    max_new_tokens,
                    tools_count,
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
            // 0.9.1 V1-usage. A client-stopped request (gone, or not keeping up) is not a
            // finished request: no recount, no counters, no prefill sample — as on the text
            // engine (review R-4, P6). Otherwise the usage block's completion figure is the
            // pipeline's count when it sends one (below), else the chunk count the handler sums.
            let reasoning_tokens = (!aborted_early).then(|| {
                let usage_completion = if generated_tokens > 0 {
                    generated_tokens
                } else {
                    usize::try_from(completion_tokens).unwrap_or(usize::MAX)
                };
                let reasoning_count = crate::reasoning_tokens::recount_reasoning(
                    model_id,
                    output,
                    reasoning,
                    usage_completion,
                    count_tokens,
                );
                metrics.record_request_tokens(modality, prompt_tokens, reasoning_count.tokens);
                if let Some(prefill) = prefill {
                    metrics.record_prefill(modality, prompt_tokens, prefill);
                    // Cross-check (no bridge call): GenAI's own whole-generation time next to the
                    // measured first-chunk interval.
                    tracing::debug!(
                        model_id,
                        prompt_tokens,
                        prefill_ms = prefill.as_secs_f64() * 1000.0,
                        inference_ms,
                        "VLM effective prefill"
                    );
                }
                reasoning_count.tokens
            });
            // PromptTokens must arrive before Done so the SSE ScanState
            // has the count when it emits the usage chunk on Done.
            // try_send, best-effort: never block this engine thread on a
            // client channel (a stalled client's stream closes when the
            // sender drops at the end of this generation either way).
            if prompt_tokens > 0 {
                let _ = token_tx.try_send(StreamEvent::PromptTokens(prompt_tokens));
            }
            // Usage must use the pipeline's count, not the number of streamer callbacks.
            if generated_tokens > 0 {
                let _ = token_tx.try_send(StreamEvent::CompletionTokens(generated_tokens));
            }
            if let Some(n) = reasoning_tokens {
                let _ = token_tx.try_send(StreamEvent::ReasoningTokens(n));
            }
            // 0.9.1: a budget stop is reported as `length` (ThinkingBudget).
            let _ = token_tx.try_send(StreamEvent::Done(budget.finish(finish)));
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

        let ticket = clock.start(Duration::ZERO);
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

    /// Block B2c: 4,678 streamer callbacks carried 6,425 tokens; the pipeline's count wins.
    #[test]
    fn the_pipeline_count_beats_the_callback_count() {
        assert_eq!(authoritative_token_count(Some(6_425), 4_678), 6_425);
        assert_eq!(
            authoritative_token_count(Some(0), 7),
            7,
            "no figure: callbacks"
        );
        assert_eq!(authoritative_token_count(None, 7), 7, "an error: callbacks");
        assert_eq!(
            authoritative_token_count(Some(3), 7),
            7,
            "never fewer than seen"
        );
    }

    /// Block B2a: a generation is only overdue past its own allowance. A 92k-token prompt with a
    /// 6,425-token answer gets 460 s + 1,285 s; the ceiling the supervisor applies on top is slack.
    #[test]
    fn the_watchdog_counts_only_time_beyond_the_requests_allowance() {
        assert_eq!(
            generation_allowance(92_159, 6_425),
            Duration::from_secs(460 + 1_285)
        );
        assert_eq!(generation_allowance(0, 0), Duration::ZERO);
        assert_eq!(generation_allowance(5, 512), Duration::from_secs(102));
        let clock = GenerationClock::default();
        let long = clock.start(Duration::from_secs(3_600));
        long.begin();
        assert_eq!(
            clock.oldest_age(),
            Some(Duration::ZERO),
            "running inside its allowance: generating, but not overdue"
        );
        drop(long);
        let short = clock.start(Duration::ZERO);
        short.begin();
        std::thread::sleep(Duration::from_millis(20));
        assert!(
            clock
                .oldest_age()
                .is_some_and(|d| d >= Duration::from_millis(20)),
            "a running generation with no allowance is overdue from its first second"
        );
    }

    /// Fable review 3, B-1: a request queued behind a long running one must not be counted while
    /// the long one runs, and must not inherit the long one's elapsed time when it starts.
    #[test]
    fn a_queued_request_does_not_inherit_the_running_ones_time() {
        let clock = GenerationClock::default();
        let long = clock.start(Duration::from_secs(3_600));
        long.begin();
        // A health probe (allowance 0) is queued behind it.
        let probe = clock.start(Duration::ZERO);
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(
            clock.oldest_age(),
            Some(Duration::ZERO),
            "queued behind a running generation: not counted, the running one is inside its allowance"
        );
        // The long one finishes; the probe starts: its clock begins now, not at its queueing.
        drop(long);
        probe.begin();
        assert!(
            clock
                .oldest_age()
                .is_some_and(|d| d < Duration::from_millis(30)),
            "the probe's wait is not counted against it"
        );
        // A thread that never picks up what is queued is still caught.
        drop(probe);
        let _stuck = clock.start(Duration::from_secs(3_600));
        std::thread::sleep(Duration::from_millis(20));
        assert!(
            clock
                .oldest_age()
                .is_some_and(|d| d >= Duration::from_millis(20)),
            "nothing running and a request waiting: counted from the last progress"
        );
    }

    /// With two overlapping generations, the clock reports the *older* one's
    /// age — the watchdog cares about the longest-stuck request, not the
    /// most recent.
    #[test]
    fn generation_clock_reports_the_oldest_of_several_tickets() {
        let clock = GenerationClock::default();
        let older = clock.start(Duration::ZERO);
        std::thread::sleep(Duration::from_millis(20));
        let _newer = clock.start(Duration::ZERO);

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

    /// S2: the lease travels with the job. The submitting call returning (or the HTTP request
    /// being dropped) does not release it; only dropping the command the engine thread holds
    /// does, and a submission that never gets in releases it at once.
    #[tokio::test]
    async fn the_lease_lives_as_long_as_the_engine_job() {
        struct Flag(Arc<AtomicBool>);
        impl Drop for Flag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let (tx, mut rx) = tokio::sync::mpsc::channel::<VlmCommand>(4);
        let handle = VlmHandle::from_sender_with_cap(tx, "test-vlm", 1, 50);
        let released = Arc::new(AtomicBool::new(false));
        let (tok_tx, tok_rx) = stream_channel();
        let result = handle
            .generate_with_lease(
                vec![],
                vec![],
                None,
                None,
                GenParams::default(),
                tok_tx,
                Instant::now(),
                Some(Box::new(Flag(Arc::clone(&released)))),
                0,
            )
            .await;
        assert_eq!(result, SubmitResult::Submitted);
        // The client goes away; the job is still running on the engine thread.
        drop(tok_rx);
        assert!(!released.load(Ordering::SeqCst), "held while the job runs");
        let job = rx.recv().await.unwrap();
        assert!(!released.load(Ordering::SeqCst), "held by the command");
        drop(job); // the engine thread finishing the generation
        assert!(released.load(Ordering::SeqCst), "released with the job");

        // Take the only slot again; a second submission then times out and drops its lease.
        let (tok_tx, _rx1) = stream_channel();
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
        let _busy = rx.recv().await.unwrap();
        let rejected = Arc::new(AtomicBool::new(false));
        let (tok_tx, _rx2) = stream_channel();
        let result = handle
            .generate_with_lease(
                vec![],
                vec![],
                None,
                None,
                GenParams::default(),
                tok_tx,
                Instant::now(),
                Some(Box::new(Flag(Arc::clone(&rejected)))),
                0,
            )
            .await;
        assert_eq!(result, SubmitResult::AtCapacity);
        assert!(
            rejected.load(Ordering::SeqCst),
            "a rejected submission frees its lease"
        );
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
    fn active_counts_held_permits_for_vlm() {
        let (tx, _rx) = tokio::sync::mpsc::channel(8);
        let h = VlmHandle::from_sender(tx, "t-vlm");
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
        let (tx, mut rx) = tokio::sync::mpsc::channel::<VlmCommand>(4);
        let h = VlmHandle::from_sender(tx, "t-vlm");
        let submit = |h: &VlmHandle| {
            let (tok_tx, _tok_rx) = crate::streaming::stream_channel();
            let h = h.clone();
            async move {
                h.generate(
                    vec![],
                    vec![],
                    None,
                    None,
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

    /// 0.9.1 S5c: a slot taken with `acquire_slot` counts in `active()` (an eviction drains it)
    /// while the caller decodes; an eviction that commits meanwhile makes `submit` refuse with
    /// `Closing` and release the permit, a refused one does not; dropping an unused slot
    /// releases it too.
    #[tokio::test]
    async fn a_slot_counts_while_held_and_submit_rechecks_closing() {
        async fn submit(slot: VlmSlot<'_>) -> SubmitResult {
            let (tok_tx, _tok_rx) = crate::streaming::stream_channel();
            slot.submit(
                vec![],
                vec![],
                None,
                None,
                GenParams::default(),
                tok_tx,
                Instant::now(),
                None,
                0,
            )
            .await
        }
        let (tx, mut rx) = tokio::sync::mpsc::channel::<VlmCommand>(4);
        let h = VlmHandle::from_sender(tx, "t-vlm");

        let slot = h.acquire_slot().await.unwrap();
        assert_eq!(h.active(), 1, "a held slot counts");
        drop(slot);
        assert_eq!(h.active(), 0, "an unused slot releases on drop");

        // A refused (busy) non-forced eviction raises `closing` and clears it: a submit landing
        // in between is NOT refused (review F7).
        let slot = h.acquire_slot().await.unwrap();
        ManagedEngine::set_closing(&h, true);
        assert_eq!(submit(slot).await, SubmitResult::Submitted);
        assert!(rx.try_recv().is_ok(), "the command reached the engine");
        ManagedEngine::set_closing(&h, false);

        // A committed eviction refuses the submit and releases the permit.
        let slot = h.acquire_slot().await.unwrap();
        ManagedEngine::mark_eviction_committed(&h);
        assert_eq!(submit(slot).await, SubmitResult::Closing);
        assert_eq!(h.active(), 0, "the refused submit's permit is released");
        assert!(rx.try_recv().is_err(), "nothing reached the engine");
    }

    /// D-09: the VLM handle serves tokenize/detokenize through the engine thread. The mock
    /// tokenizer is one id per character, so the round trip is exact and the ids are checkable.
    #[tokio::test]
    async fn tokenize_and_detokenize_round_trip_through_the_engine_thread() {
        let (h, _thread) = spawn_mock_vlm("t-vlm");
        let ids = h.tokenize("Hi z".to_owned()).await.unwrap();
        assert_eq!(ids, vec![72, 105, 32, 122]);
        assert_eq!(h.detokenize(ids).await.unwrap(), "Hi z");
        assert_eq!(h.detokenize(vec![]).await.unwrap(), "");
        // Tokenizing consumes no admission slot.
        assert_eq!(h.active(), 0);
    }

    /// Submit one generation to a fresh mock VLM whose single user message is
    /// `content`; return every event the client got, each with `active()` read
    /// the moment the client had it.
    async fn mock_vlm_events_with_active(content: &str) -> Vec<(StreamEvent, usize)> {
        let (h, _thread) = spawn_mock_vlm("t-vlm");
        let (tok_tx, mut tok_rx) = crate::streaming::stream_channel();
        let submitted = h
            .generate(
                vec![serde_json::json!({"role": "user", "content": content})],
                vec![],
                None,
                None,
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
    async fn vlm_releases_the_slot_before_done_reaches_the_client() {
        let seen = mock_vlm_events_with_active("hi").await;
        assert_eq!(seen.len(), 2, "{seen:?}");
        assert!(matches!(&seen[0], (StreamEvent::Token(t, 1), _) if t == "mock-vlm"));
        assert!(
            matches!(
                &seen[1],
                (StreamEvent::Done(crate::ov_cb::FinishReason::Stop), 0)
            ),
            "the slot must be free when Done arrives: {seen:?}"
        );
    }

    /// 0.9.1 C29, error terminal: the slot is already released when the
    /// `Error` (and the trailing `Done`) reach the client.
    #[tokio::test]
    async fn vlm_releases_the_slot_before_an_error_reaches_the_client() {
        let seen = mock_vlm_events_with_active(MOCK_VLM_GENERATE_ERROR).await;
        assert_eq!(seen.len(), 2, "{seen:?}");
        assert!(
            matches!(&seen[0], (StreamEvent::Error(e), 0) if e == MOCK_VLM_GENERATE_ERROR),
            "the slot must be free when Error arrives: {seen:?}"
        );
        assert!(
            matches!(
                &seen[1],
                (StreamEvent::Done(crate::ov_cb::FinishReason::Stop), 0)
            ),
            "{seen:?}"
        );
    }

    /// C29 probe slot: rides in the slot tuple AFTER a real admission permit
    /// (tuple fields drop in order), so its `Drop` runs right after the permit
    /// is released and records, at that moment, the semaphore's held count and
    /// how many events the client channel already holds. 0 events = nothing
    /// (no `Done`, no `Error`) reached the client before the release.
    /// `queued` is the half that discriminates: `held` is 0 by construction (the probe
    /// drops after the permit) and only confirms that drop order.
    struct ReleaseProbe {
        sem: Arc<Semaphore>,
        cap: usize,
        tx: TokenSender,
        at_release: Arc<Mutex<Option<(usize, usize)>>>,
    }

    impl Drop for ReleaseProbe {
        fn drop(&mut self) {
            let held = self.cap - self.sem.available_permits();
            let queued = crate::streaming::CHANNEL_CAPACITY - self.tx.capacity();
            *self.at_release.lock().unwrap() = Some((held, queued));
        }
    }

    /// Runs [`finish_vlm_generation`] with `result` and a (permit, probe) slot;
    /// returns (held permits, queued events) at release time and the events.
    fn run_vlm_tail(
        result: anyhow::Result<(FinishReason, usize, usize, f64)>,
    ) -> ((usize, usize), Vec<StreamEvent>) {
        run_vlm_tail_with(result, 16)
    }

    /// [`run_vlm_tail`] with an explicit `max_new_tokens` budget.
    fn run_vlm_tail_with(
        result: anyhow::Result<(FinishReason, usize, usize, f64)>,
        max_new_tokens: usize,
    ) -> ((usize, usize), Vec<StreamEvent>) {
        let sem = Arc::new(Semaphore::new(1));
        let permit = Arc::clone(&sem).try_acquire_owned().unwrap();
        let (tx, mut rx) = crate::streaming::stream_channel();
        let at_release = Arc::new(Mutex::new(None));
        let probe = ReleaseProbe {
            sem: Arc::clone(&sem),
            cap: 1,
            tx: tx.clone(),
            at_release: Arc::clone(&at_release),
        };
        let metrics = HotMetrics::new(ModelKind::Vision, "t-vlm", "CPU");
        finish_vlm_generation(
            (permit, probe),
            result,
            &VlmGenTail {
                model_id: "t-vlm",
                modality: Modality::Text,
                started_at: Instant::now(),
                completion_tokens: 3,
                aborted_early: false,
                max_new_tokens,
                message_count: 1,
                tools_count: 0,
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

    /// 0.9.1 C29 (F1): the REAL VLM tail releases the slot before `Done`:
    /// at release the permit is back (0 held) and no event has been sent yet.
    #[test]
    fn vlm_tail_releases_the_slot_before_done() {
        let (at, events) = run_vlm_tail(Ok((FinishReason::Stop, 5, 3, 1.0)));
        assert_eq!(at, (0, 0), "released before any event was sent: {events:?}");
        assert!(
            matches!(
                events.as_slice(),
                [
                    StreamEvent::PromptTokens(5),
                    StreamEvent::CompletionTokens(3),
                    StreamEvent::ReasoningTokens(0),
                    StreamEvent::Done(FinishReason::Stop)
                ]
            ),
            "{events:?}"
        );
    }

    /// 0.9.1 C29 (F1), error terminals: a failed generation and a KV wedge
    /// (0 tokens generated) — the slot is released before `Error` + `Done`.
    #[test]
    fn vlm_tail_releases_the_slot_before_an_error() {
        let (at, events) = run_vlm_tail(Err(anyhow::anyhow!("boom")));
        assert_eq!(at, (0, 0), "released before any event was sent: {events:?}");
        assert!(
            matches!(
                events.as_slice(),
                [StreamEvent::Error(e), StreamEvent::Done(FinishReason::Stop)] if e == "boom"
            ),
            "{events:?}"
        );

        let (at, events) = run_vlm_tail(Ok((FinishReason::Stop, 5, 0, 1.0)));
        assert_eq!(at, (0, 0), "released before any event was sent: {events:?}");
        assert!(
            matches!(
                events.as_slice(),
                [StreamEvent::Error(e), StreamEvent::Done(FinishReason::Stop)]
                    if e.contains(VLM_KV_WEDGE_MARKER)
            ),
            "{events:?}"
        );
    }

    /// P-02 / E1 F1: a cap-0 request with no `max_tokens` whose budget the handler resolved to the
    /// pool room is a bounded request, so 0 generated tokens is the KV wedge; an unresolved 0
    /// (pool unknown) is not.
    #[test]
    fn a_resolved_room_budget_with_no_tokens_is_a_wedge() {
        let room =
            crate::handlers::chat::clamp_vlm_max_tokens("t-vlm", (true, 10_000), 4_000, 0).unwrap();
        assert_eq!(room, 5_900);
        let (_, events) = run_vlm_tail_with(Ok((FinishReason::Stop, 4_000, 0, 1.0)), room);
        assert!(
            matches!(
                events.as_slice(),
                [StreamEvent::Error(e), StreamEvent::Done(FinishReason::Stop)]
                    if e.contains(VLM_KV_WEDGE_MARKER)
            ),
            "{events:?}"
        );
        let (_, events) = run_vlm_tail_with(Ok((FinishReason::Stop, 4_000, 0, 1.0)), 0);
        assert!(
            !events.iter().any(|e| matches!(e, StreamEvent::Error(_))),
            "an unresolved unbounded budget is not a wedge: {events:?}"
        );
    }

    // ---- 0.9.1 V1-usage: reasoning recount + prompt / prefill series ----

    use crate::metrics::test_sample as sample;

    /// Runs the VLM tail for a successful generation of `generated` tokens over `prompt` prompt
    /// tokens with `output` as the streamed text; returns the events and the `/metrics` text.
    fn run_vlm_usage(
        output: &str,
        prompt: usize,
        generated: usize,
        prefill: Option<Duration>,
        count: impl FnOnce(&str) -> anyhow::Result<usize>,
    ) -> (Vec<StreamEvent>, String) {
        run_vlm_usage_with(
            (false, crate::thinking_budget::BudgetOutcome::default()),
            output,
            prompt,
            generated,
            prefill,
            count,
        )
    }

    fn run_vlm_usage_with(
        (stopped, budget): (bool, crate::thinking_budget::BudgetOutcome),
        output: &str,
        prompt: usize,
        generated: usize,
        prefill: Option<Duration>,
        count: impl FnOnce(&str) -> anyhow::Result<usize>,
    ) -> (Vec<StreamEvent>, String) {
        use metrics_exporter_prometheus::PrometheusBuilder;
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let (tx, mut rx) = crate::streaming::stream_channel();
        metrics::with_local_recorder(&recorder, || {
            let metrics = HotMetrics::new(ModelKind::Vision, "u-vlm", "GPU.0");
            finish_vlm_generation(
                (),
                Ok((FinishReason::Stop, prompt, generated, 1.0)),
                &VlmGenTail {
                    model_id: "u-vlm",
                    modality: Modality::Multimodal,
                    started_at: Instant::now(),
                    completion_tokens: 3,
                    aborted_early: stopped,
                    max_new_tokens: 64,
                    message_count: 1,
                    tools_count: 0,
                    prefill,
                    output,
                    reasoning: crate::prompt_builder::ReasoningSplit::new(
                        Some(crate::model_manager::ReasoningParser::Qwen3),
                        true,
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
    fn vlm_recounts_reasoning_and_records_the_three_series() {
        let (events, r) = run_vlm_usage(
            "look at the image</think>A cat.",
            1_200,
            12,
            Some(Duration::from_millis(400)),
            |s: &str| {
                assert_eq!(s, "look at the image</think>");
                Ok(6)
            },
        );
        assert_eq!(reasoning_of(&events), 6);
        assert!(reasoning_of(&events) <= 12);
        assert_eq!(
            sample(
                &r,
                "rustedvino_prompt_tokens_total",
                "modality=\"multimodal\""
            ),
            Some(1_200.0)
        );
        assert_eq!(
            sample(
                &r,
                "rustedvino_reasoning_tokens_total",
                "modality=\"multimodal\""
            ),
            Some(6.0)
        );
        let sum = sample(
            &r,
            "rustedvino_prefill_tokens_per_second_sum",
            "modality=\"multimodal\"",
        )
        .unwrap();
        assert!(
            (sum - 3_000.0).abs() < 1.0,
            "1,200 tokens over 400 ms: {sum}"
        );
    }

    #[test]
    fn vlm_recount_is_clamped_and_a_tokenizer_error_falls_back_to_bytes() {
        let (events, _) = run_vlm_usage("all thinking", 300, 4, None, |_: &str| Ok(99));
        assert_eq!(reasoning_of(&events), 4, "clamped to the generated total");
        // Tokenizer error: the byte split of the exact total.
        let (events, r) = run_vlm_usage("abcd</think>abcdefgh", 300, 10, None, |_: &str| {
            Err(anyhow::anyhow!("tokenizer failed"))
        });
        assert_eq!(reasoning_of(&events), 6, "12 of 20 bytes → 6 of 10 tokens");
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Done(FinishReason::Stop))
        ));
        assert_eq!(
            sample(
                &r,
                "rustedvino_reasoning_tokens_total",
                "modality=\"multimodal\""
            ),
            Some(6.0)
        );
        // No first chunk: no prefill sample, counters still count.
        assert_eq!(
            sample(
                &r,
                "rustedvino_prefill_tokens_per_second_count",
                "modality=\"multimodal\""
            ),
            Some(0.0)
        );
        assert_eq!(
            sample(
                &r,
                "rustedvino_prompt_tokens_total",
                "modality=\"multimodal\""
            ),
            Some(300.0)
        );
    }

    #[test]
    fn vlm_prefill_floor_holds() {
        let (_, r) = run_vlm_usage("x", 255, 1, Some(Duration::from_millis(10)), |_: &str| {
            Ok(1)
        });
        assert_eq!(
            sample(
                &r,
                "rustedvino_prefill_tokens_per_second_count",
                "modality=\"multimodal\""
            ),
            Some(0.0)
        );
        assert_eq!(
            sample(
                &r,
                "rustedvino_prompt_tokens_total",
                "modality=\"multimodal\""
            ),
            Some(255.0)
        );
    }

    /// R-4 (P6 symmetric): a client-stopped VLM request is not counted — no recount, no
    /// `ReasoningTokens`, no prompt / reasoning counts and no prefill sample — as on CB.
    #[test]
    fn vlm_client_stopped_request_counts_nothing() {
        let (events, r) = run_vlm_usage_with(
            (true, crate::thinking_budget::BudgetOutcome::default()),
            "<think>half",
            900,
            5,
            Some(Duration::from_millis(100)),
            |_: &str| panic!("a stopped request must not be recounted"),
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, StreamEvent::ReasoningTokens(_))),
            "{events:?}"
        );
        let m = "modality=\"multimodal\"";
        assert_eq!(sample(&r, "rustedvino_prompt_tokens_total", m), Some(0.0));
        assert_eq!(
            sample(&r, "rustedvino_reasoning_tokens_total", m),
            Some(0.0)
        );
        assert_eq!(
            sample(&r, "rustedvino_prefill_tokens_per_second_count", m),
            Some(0.0)
        );
    }

    /// 0.9.1 strict thinking budget: a budget stop is a FINISHED request — recounted, the three
    /// series counted, `ReasoningTokens` before a terminal `Done(ThinkingBudget)` (`length`).
    #[test]
    fn vlm_budget_stop_is_length_and_counted_as_finished() {
        let stopped = crate::thinking_budget::BudgetOutcome {
            armed: true,
            stopped: true,
            inside: true,
        };
        let (events, r) = run_vlm_usage_with(
            (false, stopped),
            "still thinking about the image",
            900,
            8,
            Some(Duration::from_millis(100)),
            |_: &str| Ok(8),
        );
        assert_eq!(reasoning_of(&events), 8);
        assert!(
            matches!(
                events.last(),
                Some(StreamEvent::Done(FinishReason::ThinkingBudget))
            ),
            "{events:?}"
        );
        assert_eq!(FinishReason::ThinkingBudget.as_openai(), "length");
        let m = "modality=\"multimodal\"";
        assert_eq!(sample(&r, "rustedvino_prompt_tokens_total", m), Some(900.0));
        assert_eq!(
            sample(&r, "rustedvino_reasoning_tokens_total", m),
            Some(8.0)
        );
        assert_eq!(
            sample(&r, "rustedvino_prefill_tokens_per_second_count", m),
            Some(1.0)
        );
        // Ran out of max_tokens inside the reasoning (budget armed, not reached): also the
        // budget finish. A natural EOS stays `stop`.
        let inside = crate::thinking_budget::BudgetOutcome {
            armed: true,
            stopped: false,
            inside: true,
        };
        assert_eq!(
            inside.finish(FinishReason::Length),
            FinishReason::ThinkingBudget
        );
        assert_eq!(inside.finish(FinishReason::Stop), FinishReason::Stop);
        let unarmed = crate::thinking_budget::BudgetOutcome::default();
        assert_eq!(unarmed.finish(FinishReason::Length), FinishReason::Length);
    }

    /// Drives the real streamer callback (`VlmStream::on_chunk`) the way the pipeline does — one
    /// call per chunk until it answers `true` (stop) — then the real tail. Returns the chunks the
    /// pipeline delivered, the events and the stream.
    fn drive_vlm_stream(
        chunks: &[&str],
        budget: Option<u32>,
        generated_tokens: usize,
    ) -> (usize, Vec<StreamEvent>) {
        let (tx, mut rx) = crate::streaming::stream_channel();
        let metrics = HotMetrics::new(ModelKind::Vision, "s-vlm", "GPU.0");
        let params = GenParams {
            reasoning: crate::prompt_builder::ReasoningSplit::new(
                Some(crate::model_manager::ReasoningParser::Qwen3),
                true,
            ),
            thinking_budget: budget,
            max_new_tokens: 64,
            ..GenParams::default()
        };
        let started_at = Instant::now();
        let mut stream = VlmStream::new("s-vlm", Modality::Text, started_at, &params);
        let mut delivered = 0;
        for c in chunks {
            delivered += 1;
            if stream.on_chunk(c, &tx, &metrics) {
                break;
            }
        }
        finish_vlm_generation(
            (),
            Ok((FinishReason::Stop, 40, generated_tokens, 1.0)),
            &VlmGenTail {
                model_id: "s-vlm",
                modality: Modality::Text,
                started_at,
                completion_tokens: stream.completion_tokens,
                aborted_early: stream.aborted_early,
                max_new_tokens: params.max_new_tokens,
                message_count: 1,
                tools_count: 0,
                prefill: None,
                output: &stream.output,
                reasoning: params.reasoning,
                budget: crate::thinking_budget::BudgetOutcome::of(
                    stream.budget.as_ref(),
                    stream.aborted_early,
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

    /// The VLM streamer with a budget: a 4-token budget is 3 chunks (ceil(4 / 1.37)); the
    /// callback asks for the stop exactly once, at the third chunk, no chunk after it reaches the
    /// client, and the request ends `Done(ThinkingBudget)` — also when the pipeline reports 0
    /// generated tokens after the stop (not a KV wedge).
    #[test]
    fn vlm_streamer_stops_at_the_chunk_budget_and_is_not_a_wedge() {
        let chunks = ["a", "b", "c", "d", "e"];
        for generated in [3, 0] {
            let (delivered, events) = drive_vlm_stream(&chunks, Some(4), generated);
            assert_eq!(delivered, 3, "stop asked at the third chunk");
            let tokens = events
                .iter()
                .filter(|e| matches!(e, StreamEvent::Token(..)))
                .count();
            assert_eq!(tokens, 3, "no chunk after the stop");
            assert!(
                !events.iter().any(|e| matches!(e, StreamEvent::Error(_))),
                "generated {generated}: {events:?}"
            );
            assert!(
                matches!(
                    events.last(),
                    Some(StreamEvent::Done(FinishReason::ThinkingBudget))
                ),
                "generated {generated}: {events:?}"
            );
        }
        // No budget: the same chunks run to the end, a plain finish.
        let (delivered, events) = drive_vlm_stream(&chunks, None, 5);
        assert_eq!(delivered, 5);
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Done(FinishReason::Stop))
        ));
        // A natural close before the budget: no stop.
        let (delivered, events) =
            drive_vlm_stream(&["a", "</think>", "x", "y", "z", "w"], Some(4), 6);
        assert_eq!(delivered, 6);
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Done(FinishReason::Stop))
        ));
    }
}
