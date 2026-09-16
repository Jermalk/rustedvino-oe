// ============================================================
// src/embed_engine.rs — embedding engine thread (R4/G3)
// ============================================================
// The first non-stub "media" engine in the registry. Same dedicated-thread +
// command-channel architecture as `vlm_engine.rs`, but the work is
// request/response (embed a batch → return vectors) rather than a token
// stream, so the command carries a `oneshot` reply channel (like the CB
// engine's `Tokenize`/`Detokenize` commands) instead of a `TokenSender`.
//
//   - One dedicated OS thread owns the `OvEmbedEngine` (TextEmbeddingPipeline).
//   - Commands arrive via an mpsc channel (capacity `EMBED_CHANNEL_CAP`).
//   - The thread processes one Embed at a time (the pipeline is single-stream
//     and blocking), replying over the per-request oneshot.
//   - Admission is a real `Semaphore` gate (mirrors `cb_engine`/`vlm_engine`,
//     dev/plans/vlm-admission-gate-fix.md's recipe): a permit is acquired
//     *before* a command is sent and held by the engine thread until it
//     replies, so `active()` is honest occupancy, not merely "callers who
//     called `embed()`". Replaces the old `in_flight: AtomicUsize` +
//     `InFlightGuard` approximation.
// ============================================================

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc::Sender, oneshot};

use crate::cb_engine::AdmitError;
use crate::metrics::{HotMetrics, Modality};
use crate::model_manager::{ManagedEngine, ModelKind};
use crate::ov_embed::OvEmbedEngine;

/// Embedding admission-gate size (also the reported `max_concurrency`): up to
/// this many requests may be queued/embedding before the gate returns
/// [`AdmitError::AtCapacity`].
pub(crate) const EMBED_CHANNEL_CAP: usize = 8;

/// The result of one embedding request: one float vector per input document,
/// plus the total prompt-token count for `usage`.
#[derive(Clone, Debug)]
pub struct EmbedOutput {
    /// One embedding per input text, in input order.
    pub vectors: Vec<Vec<f32>>,
    /// Total tokens across all inputs (for `usage.prompt_tokens`/`total_tokens`).
    pub prompt_tokens: usize,
}

// ── Commands ─────────────────────────────────────────────────────────────────

pub(crate) enum EmbedCommand {
    /// Embed a batch of documents and reply with the vectors + token count.
    Embed {
        /// Input documents, already normalized to a non-empty `Vec<String>`.
        texts: Vec<String>,
        /// One-shot reply channel — carries the result back to the handler.
        reply: oneshot::Sender<anyhow::Result<EmbedOutput>>,
        /// Wall-clock start (handler entry) for the duration metric.
        started_at: Instant,
        /// Admission-gate permit, held until this batch finishes. Dropped
        /// after the engine thread sends its reply, releasing the slot for
        /// the next waiting caller. See [`EmbeddingHandle::embed`].
        permit: OwnedSemaphorePermit,
    },
    /// Count `text`'s tokens via the pipeline's tokenizer. Not
    /// admission-gated (mirrors `npu_engine::NpuCommand::CountTokens` /
    /// `vlm_engine::VlmCommand::CountTokens`): a cheap tokenizer call, not an
    /// embed, so it doesn't compete for [`EMBED_CHANNEL_CAP`] slots. Used by
    /// the L0 length gate (`handlers::embeddings::gate_embed_text`) to reject
    /// an over-length input before it reaches `embed_documents` and raises an
    /// OV shape-inference exception.
    CountTokens {
        text: String,
        reply: oneshot::Sender<Result<usize, String>>,
    },
}

// ── Handle ───────────────────────────────────────────────────────────────────

/// Cloneable submit handle for the embedding engine thread.
///
/// `embed` is request/response: it sends the batch and awaits the reply. The
/// engine thread processes one batch at a time — requests serialise in the
/// channel.
#[derive(Clone, Debug)]
pub struct EmbeddingHandle {
    tx: Sender<EmbedCommand>,
    /// The model ID of the loaded embedding model (the `model` response field).
    model_id: Arc<str>,
    /// Admission-gate permits — `cap` of them. Mirrors `cb_engine::EngineHandle`'s
    /// `sem` field exactly.
    sem: Arc<Semaphore>,
    /// Channel capacity = the semaphore's permit count = [`EMBED_CHANNEL_CAP`].
    cap: usize,
    /// Maximum time [`embed`](Self::embed) waits for a permit before
    /// returning [`AdmitError::AtCapacity`] (→ HTTP 429).
    queue_timeout: Duration,
    /// Callers currently parked in [`embed`](Self::embed) awaiting an
    /// admission permit — the engine's honest `rustedvino_requests_waiting`.
    waiting: Arc<AtomicUsize>,
    /// The model's `config.json`-declared `max_position_embeddings`, resolved
    /// once at load time ([`crate::ov_embed::resolve_max_seq_len`]). `None`
    /// when the field wasn't declared/parseable — the L0 length gate is
    /// skipped in that case, same fail-open policy as the NPU prompt gate.
    max_seq_len: Option<usize>,
}

impl EmbeddingHandle {
    /// The model ID of the loaded embedding model.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Build a handle around an existing command channel — test seam only.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn from_sender(tx: Sender<EmbedCommand>, model_id: &str) -> Self {
        let cap = 1000; // generous test cap — never hits the admission gate
        Self {
            tx,
            model_id: Arc::from(model_id),
            sem: Arc::new(Semaphore::new(cap)),
            cap,
            queue_timeout: Duration::from_secs(30),
            waiting: Arc::new(AtomicUsize::new(0)),
            max_seq_len: None,
        }
    }

    /// Build a handle with an explicit `max_seq_len` — test seam for the L0
    /// length-gate tests, which need a handle that reports a real ceiling.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn from_sender_with_max_seq_len(
        tx: Sender<EmbedCommand>,
        model_id: &str,
        max_seq_len: usize,
    ) -> Self {
        Self {
            max_seq_len: Some(max_seq_len),
            ..Self::from_sender(tx, model_id)
        }
    }

    /// Creates a test handle with an explicit cap and timeout. Used by
    /// admission-gate tests that need to observe `AtCapacity` behaviour.
    #[cfg(test)]
    pub(crate) fn from_sender_with_cap(
        tx: Sender<EmbedCommand>,
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
            max_seq_len: None,
        }
    }

    /// Embed a batch of documents.
    ///
    /// Waits up to `queue_timeout` for an admission permit before returning
    /// [`AdmitError::AtCapacity`] (→ HTTP 429) — mirrors
    /// `cb_engine::EngineHandle::add_request`'s admission gate. Returns
    /// [`AdmitError::EngineDead`] if the engine thread has exited (→ HTTP
    /// 503), or [`AdmitError::Failed`] if the pipeline call itself failed.
    ///
    /// # Errors
    /// See the variants above: [`AdmitError::AtCapacity`],
    /// [`AdmitError::EngineDead`], [`AdmitError::Failed`].
    pub async fn embed(&self, texts: Vec<String>) -> Result<EmbedOutput, AdmitError> {
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
            Ok(Err(_)) => return Err(AdmitError::EngineDead), // semaphore closed (shouldn't happen)
            Err(_) => return Err(AdmitError::AtCapacity),     // timed out waiting for a permit
        };

        let started_at = Instant::now();
        let (reply_tx, reply_rx) = oneshot::channel();

        // Permit acquired. Send to the engine — if the channel is closed, the
        // permit (moved into the command) drops here, automatically releasing
        // the slot back to the semaphore.
        let sent = self
            .tx
            .send(EmbedCommand::Embed {
                texts,
                reply: reply_tx,
                started_at,
                permit,
            })
            .await;
        if sent.is_err() {
            return Err(AdmitError::EngineDead);
        }

        let result = reply_rx.await.map_err(|_| AdmitError::EngineDead)?;
        result.map_err(AdmitError::Failed)
    }

    /// The model's `max_position_embeddings` ceiling (resolved once at load
    /// time from `config.json`), or `None` when it wasn't declared/parseable.
    #[must_use]
    pub(crate) fn max_seq_len(&self) -> Option<usize> {
        self.max_seq_len
    }

    /// Count `text`'s tokens via the pipeline's tokenizer.
    ///
    /// Not admission-gated (mirrors `npu_engine::NpuHandle::count_tokens` /
    /// `vlm_engine::VlmHandle::count_tokens`) — a cheap tokenizer call, not an
    /// embed, so it doesn't compete for a real embed slot.
    ///
    /// # Errors
    /// Returns an error if the engine thread is gone or the tokenizer fails.
    pub(crate) async fn count_tokens(&self, text: String) -> anyhow::Result<usize> {
        let (reply, reply_rx) = oneshot::channel();
        self.tx
            .send(EmbedCommand::CountTokens { text, reply })
            .await
            .map_err(|_| anyhow::anyhow!("embedding engine unavailable"))?;
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("embedding engine dropped tokenize reply"))?
            .map_err(|e| anyhow::anyhow!(e))
    }
}

/// The embedding engine is a [`ModelKind::Embedding`] engine. Uniform with the
/// CB/VLM `ManagedEngine` impls so `ModelManager` reads metrics/lifecycle the
/// same way for every kind.
impl ManagedEngine for EmbeddingHandle {
    fn kind(&self) -> ModelKind {
        ModelKind::Embedding
    }

    /// Accepted-but-unfinished requests (queued + the one embedding) —
    /// derived from the semaphore: real occupancy, not an approximation.
    fn active(&self) -> usize {
        self.cap.saturating_sub(self.sem.available_permits())
    }

    fn max_concurrency(&self) -> usize {
        self.cap
    }

    /// Callers currently parked in [`embed`](Self::embed) awaiting an
    /// admission permit. Mirrors `cb_engine::EngineHandle::waiting` exactly.
    fn waiting(&self) -> usize {
        self.waiting.load(Ordering::Relaxed)
    }
}

// ── Engine thread ─────────────────────────────────────────────────────────────

/// Spawn a dedicated OS thread owning `engine` and return an [`EmbeddingHandle`].
///
/// The thread exits cleanly when all handle clones are dropped (channel closes).
///
/// # Errors
/// Only propagates thread-spawn errors (rare OS resource exhaustion).
pub fn spawn_embed_engine(
    engine: OvEmbedEngine,
    model_id: &str,
    device: &str,
    queue_timeout: Duration,
    max_seq_len: Option<usize>,
) -> anyhow::Result<(EmbeddingHandle, std::thread::JoinHandle<()>)> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<EmbedCommand>(EMBED_CHANNEL_CAP);
    let sem = Arc::new(Semaphore::new(EMBED_CHANNEL_CAP));
    let model_id_arc: Arc<str> = Arc::from(model_id);
    let model_id_owned = model_id.to_owned();
    let device = device.to_owned();

    // Embeddings join the `kind="embedding"` metric family. There is no token
    // stream, so only request-count and end-to-end duration are emitted (ttft /
    // tok-per-second are generation concepts that do not apply).
    let metrics = HotMetrics::new(ModelKind::Embedding, model_id, &device);

    let thread = std::thread::Builder::new()
        .name(format!("embed-engine-{model_id_owned}"))
        .spawn(move || {
            tracing::info!(
                model_id = model_id_owned,
                kind = ModelKind::Embedding.label(),
                device,
                "embedding engine thread started"
            );
            while let Some(command) = rx.blocking_recv() {
                match command {
                    EmbedCommand::Embed {
                        texts,
                        reply,
                        started_at,
                        permit,
                    } => {
                        // Embedding input is always text → `Modality::Text`.
                        metrics.request_accepted(Modality::Text);
                        let result = run_embed(&engine, &model_id_owned, &texts);
                        metrics.record_duration(Modality::Text, started_at.elapsed().as_secs_f64());
                        // The receiver may be gone if the client disconnected — ignore.
                        let _ = reply.send(result);
                        // Balance the handle's admission acquire, whatever the
                        // outcome — releases the slot for the next waiting caller.
                        drop(permit);
                    }
                    EmbedCommand::CountTokens { text, reply } => {
                        let _ = reply.send(engine.count_tokens(&text).map_err(|e| e.to_string()));
                    }
                }
            }
            tracing::info!(
                model_id = model_id_owned,
                kind = ModelKind::Embedding.label(),
                "embedding engine thread exiting — all handles dropped"
            );
        })?;

    Ok((
        EmbeddingHandle {
            tx,
            model_id: model_id_arc,
            sem,
            cap: EMBED_CHANNEL_CAP,
            queue_timeout,
            waiting: Arc::new(AtomicUsize::new(0)),
            max_seq_len,
        },
        thread,
    ))
}

/// Run one embedding batch on the engine thread: embed the documents, then
/// count tokens across all inputs for `usage`. A token-count failure is
/// non-fatal — it degrades `prompt_tokens` to 0 rather than failing the request.
fn run_embed(
    engine: &OvEmbedEngine,
    model_id: &str,
    texts: &[String],
) -> anyhow::Result<EmbedOutput> {
    let vectors = engine.embed(texts)?;
    let prompt_tokens = texts
        .iter()
        .map(|t| engine.count_tokens(t).unwrap_or(0))
        .sum();
    tracing::debug!(
        model_id,
        n = texts.len(),
        prompt_tokens,
        "embedding batch complete"
    );
    Ok(EmbedOutput {
        vectors,
        prompt_tokens,
    })
}

/// Spawn a GPU-free mock embedding engine — the `Embedding` analogue of
/// `spawn_mock_vlm`, used by `MockEngineFactory` and routing tests.
///
/// Answers each batch with a fixed 4-dim vector per input and `prompt_tokens`
/// equal to the input count, so a handler test yields a valid response shape
/// without touching a GPU.
#[cfg(test)]
pub(crate) fn spawn_mock_embed(model_id: &str) -> (EmbeddingHandle, std::thread::JoinHandle<()>) {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<EmbedCommand>(EMBED_CHANNEL_CAP);
    let sem = Arc::new(Semaphore::new(EMBED_CHANNEL_CAP));

    let thread = std::thread::spawn(move || {
        while let Some(command) = rx.blocking_recv() {
            match command {
                EmbedCommand::Embed {
                    texts,
                    reply,
                    permit,
                    ..
                } => {
                    let vectors = texts.iter().map(|_| vec![0.1, 0.2, 0.3, 0.4]).collect();
                    let _ = reply.send(Ok(EmbedOutput {
                        vectors,
                        prompt_tokens: texts.len(),
                    }));
                    drop(permit);
                }
                // No real tokenizer behind the mock — answer with a rough
                // word-count stand-in so a caller exercising the length gate
                // against a mock handle doesn't hang waiting for a reply.
                EmbedCommand::CountTokens { text, reply } => {
                    let _ = reply.send(Ok(text.split_whitespace().count()));
                }
            }
        }
    });

    (
        EmbeddingHandle {
            tx,
            model_id: Arc::from(model_id),
            sem,
            cap: EMBED_CHANNEL_CAP,
            queue_timeout: Duration::from_secs(30),
            waiting: Arc::new(AtomicUsize::new(0)),
            max_seq_len: None,
        },
        thread,
    )
}

// ============================================================
// Unit tests (no GPU)
// ============================================================
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// A fresh handle reports zero active/waiting requests and its configured cap.
    #[test]
    fn idle_handle_reports_zero_active_and_waiting() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<EmbedCommand>(EMBED_CHANNEL_CAP);
        let h = EmbeddingHandle::from_sender(tx, "test-embed");
        assert_eq!(h.active(), 0);
        assert_eq!(h.waiting(), 0);
        assert_eq!(h.max_concurrency(), 1000, "test handle uses the test cap");
    }

    /// The mock engine round-trips a batch: one vector per input, token count.
    #[tokio::test]
    async fn mock_embed_round_trips_a_batch() {
        let (handle, _thread) = spawn_mock_embed("mock-embed");
        let out = handle
            .embed(vec!["a".to_owned(), "b".to_owned()])
            .await
            .expect("mock embed");
        assert_eq!(out.vectors.len(), 2, "one vector per input");
        assert_eq!(out.vectors[0].len(), 4, "mock returns 4-dim vectors");
        assert_eq!(out.prompt_tokens, 2);
        assert_eq!(
            handle.active(),
            0,
            "counter settled back to zero after reply"
        );
    }

    /// When a slot opens (permit released after the first reply), a waiting
    /// `embed` completes instead of returning `AtCapacity`.
    #[tokio::test]
    async fn embed_waits_and_succeeds_when_slot_opens() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<EmbedCommand>(4);
        let handle = EmbeddingHandle::from_sender_with_cap(tx, "test-embed", 1, 500);

        // Minimal engine-thread stand-in: replies to each command in order,
        // holding the first one's permit for 50 ms to simulate a slow embed.
        tokio::spawn(async move {
            let mut first = true;
            while let Some(EmbedCommand::Embed {
                texts,
                reply,
                permit,
                ..
            }) = rx.recv().await
            {
                if first {
                    first = false;
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                let _ = reply.send(Ok(EmbedOutput {
                    vectors: texts.iter().map(|_| vec![]).collect(),
                    prompt_tokens: 0,
                }));
                drop(permit);
            }
        });

        let h2 = handle.clone();
        let first_call = tokio::spawn(async move { h2.embed(vec!["a".to_owned()]).await });
        // Give the first call time to be admitted before the second competes for the slot.
        tokio::time::sleep(Duration::from_millis(10)).await;

        let second = handle.embed(vec!["b".to_owned()]).await;
        assert!(
            second.is_ok(),
            "second call must wait for the slot and then succeed, got {second:?}"
        );
        assert!(first_call.await.unwrap().is_ok());
    }

    /// When no slot opens within `queue_timeout`, `embed` returns
    /// `AtCapacity` (→ HTTP 429) rather than waiting forever.
    #[tokio::test]
    async fn embed_returns_at_capacity_after_timeout() {
        let (tx, rx) = tokio::sync::mpsc::channel::<EmbedCommand>(4);
        let handle = EmbeddingHandle::from_sender_with_cap(tx, "test-embed", 1, 50);

        let h2 = handle.clone();
        let _first = tokio::spawn(async move { h2.embed(vec!["a".to_owned()]).await });
        // Give the first request time to acquire the only permit and park in
        // the channel buffer (nobody drains it), so the slot stays held.
        tokio::time::sleep(Duration::from_millis(20)).await;

        let result = handle.embed(vec!["b".to_owned()]).await;
        assert!(
            matches!(result, Err(AdmitError::AtCapacity)),
            "must 429 after timeout, got {result:?}"
        );
        drop(rx);
    }

    /// A closed channel (engine thread gone) reports `EngineDead`, not
    /// `AtCapacity` — the caller should respond 503, not 429.
    #[tokio::test]
    async fn embed_returns_engine_dead_when_channel_closed() {
        let (tx, rx) = tokio::sync::mpsc::channel::<EmbedCommand>(4);
        drop(rx);
        let handle = EmbeddingHandle::from_sender(tx, "dead-embed");
        let result = handle.embed(vec!["x".to_owned()]).await;
        assert!(matches!(result, Err(AdmitError::EngineDead)));
        assert_eq!(handle.active(), 0, "no permit was ever held");
    }

    /// `max_seq_len()` reports `None` for a handle built without one, and
    /// `Some(n)` for one built via `from_sender_with_max_seq_len`.
    #[test]
    fn max_seq_len_reports_the_configured_ceiling() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<EmbedCommand>(4);
        let no_ceiling = EmbeddingHandle::from_sender(tx.clone(), "test-embed");
        assert_eq!(no_ceiling.max_seq_len(), None);

        let with_ceiling = EmbeddingHandle::from_sender_with_max_seq_len(tx, "test-embed", 512);
        assert_eq!(with_ceiling.max_seq_len(), Some(512));
    }

    /// `count_tokens` round-trips through `CountTokens` independent of the
    /// admission semaphore — sending it does not consume a permit.
    #[tokio::test]
    async fn count_tokens_round_trips_without_touching_the_admission_gate() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<EmbedCommand>(4);
        let handle = EmbeddingHandle::from_sender_with_cap(tx, "test-embed", 1, 500);

        tokio::spawn(async move {
            if let Some(EmbedCommand::CountTokens { text, reply }) = rx.recv().await {
                let _ = reply.send(Ok(text.split_whitespace().count()));
            }
        });

        let count = handle
            .count_tokens("one two three".to_owned())
            .await
            .expect("count_tokens");
        assert_eq!(count, 3);
        assert_eq!(
            handle.active(),
            0,
            "count_tokens must not consume an admission permit"
        );
    }

    /// A closed channel surfaces as an error from `count_tokens`, not a hang.
    #[tokio::test]
    async fn count_tokens_errors_when_channel_closed() {
        let (tx, rx) = tokio::sync::mpsc::channel::<EmbedCommand>(4);
        drop(rx);
        let handle = EmbeddingHandle::from_sender(tx, "dead-embed");
        assert!(handle.count_tokens("x".to_owned()).await.is_err());
    }

    /// A genuine pipeline failure (reply carries `Err`) surfaces as
    /// `AdmitError::Failed`, distinguishable from an admission-gate rejection.
    #[tokio::test]
    async fn embed_reports_failed_when_engine_replies_with_an_error() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<EmbedCommand>(4);
        let handle = EmbeddingHandle::from_sender(tx, "test-embed");

        let h2 = handle.clone();
        let call = tokio::spawn(async move { h2.embed(vec!["a".to_owned()]).await });

        let EmbedCommand::Embed { reply, permit, .. } = rx.recv().await.unwrap() else {
            panic!("expected an Embed command");
        };
        let _ = reply.send(Err(anyhow::anyhow!("pipeline exploded")));
        drop(permit);

        let result = call.await.unwrap();
        assert!(matches!(result, Err(AdmitError::Failed(_))));
    }

    /// Cancelling the `embed` future while it is only *waiting for a permit*
    /// (not yet admitted) must not leak a slot — `Semaphore::acquire` is
    /// cancel-safe by construction, unlike the old `InFlightGuard` scheme this
    /// replaces.
    #[tokio::test]
    async fn cancelling_embed_while_parked_does_not_leak_a_permit() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<EmbedCommand>(4);
        let handle = EmbeddingHandle::from_sender_with_cap(tx, "test-embed", 1, 5_000);

        // Take the only permit directly so the next `embed()` call parks.
        let held = Arc::clone(&handle.sem).try_acquire_owned().unwrap();
        assert_eq!(handle.active(), 1);

        let h2 = handle.clone();
        let task = tokio::spawn(async move { h2.embed(vec!["a".to_owned()]).await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(handle.waiting(), 1, "parked awaiting the permit");

        task.abort();
        let _ = task.await;
        assert_eq!(
            handle.waiting(),
            0,
            "no leaked waiting count on cancellation"
        );
        assert_eq!(
            handle.active(),
            1,
            "the held permit is still ours, untouched"
        );
        drop(held);
        assert_eq!(
            handle.active(),
            0,
            "releasing the held permit settles active()"
        );
    }
}
