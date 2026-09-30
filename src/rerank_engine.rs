// ============================================================
// src/rerank_engine.rs — reranking engine thread (/v1/rerank)
// ============================================================
// Mirrors the architecture of embed_engine.rs: a dedicated OS thread owns
// the OvRerankEngine (TextRerankPipeline). Commands arrive via an mpsc
// channel; each carries a oneshot reply. Single-stream and blocking.
// ============================================================

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use tokio::sync::{mpsc::Sender, oneshot};

use crate::in_flight::InFlightGuard;
use crate::metrics::{HotMetrics, Modality};
use crate::model_manager::{ManagedEngine, ModelKind};
use crate::ov_rerank::OvRerankEngine;

/// Reranking command-channel capacity.
pub(crate) const RERANK_CHANNEL_CAP: usize = 8;

/// The result of one reranking request.
#[derive(Clone, Debug)]
pub struct RerankOutput {
    /// `(original_index, relevance_score)` sorted by score descending.
    /// Already trimmed to the requested `top_n` by the handler.
    pub results: Vec<(usize, f32)>,
}

// ── Commands ──────────────────────────────────────────────────────────────────

pub(crate) enum RerankCommand {
    Rerank {
        query: String,
        documents: Vec<String>,
        reply: oneshot::Sender<anyhow::Result<RerankOutput>>,
        started_at: Instant,
    },
    /// Count the tokens `query` and `document` need combined, via the
    /// pipeline's tokenizer. Not admission-gated (mirrors
    /// `embed_engine::EmbedCommand::CountTokens`) — a cheap tokenizer call,
    /// not a rerank, so it doesn't compete for [`RERANK_CHANNEL_CAP`] slots.
    /// Used by the L0 length gate (`handlers::reranking::gate_rerank_pair`)
    /// to reject an over-length pair before it reaches the pipeline and
    /// raises an OV shape-inference exception
    /// (the project's internal engineering log).
    CountTokens {
        query: String,
        document: String,
        reply: oneshot::Sender<Result<usize, String>>,
    },
}

// ── Handle ────────────────────────────────────────────────────────────────────

/// Cloneable submit handle for the reranking engine thread.
#[derive(Clone, Debug)]
pub struct RerankingHandle {
    tx: Sender<RerankCommand>,
    model_id: Arc<str>,
    in_flight: Arc<AtomicUsize>,
    /// The model's usable input length in tokens, resolved once at load time
    /// ([`crate::ov_embed::resolve_max_seq_len`] — a generic BERT-family
    /// `config.json` reader, not embedding-specific despite its module).
    /// `None` when `config.json` didn't declare it — fail-open, no gate.
    max_seq_len: Option<usize>,
}

impl RerankingHandle {
    /// The model ID of the loaded reranking model.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// The model's usable input length in tokens, if known. Used by the
    /// `/v1/rerank` handler's pre-inference length gate
    /// (the project's internal engineering log).
    #[must_use]
    pub(crate) fn max_seq_len(&self) -> Option<usize> {
        self.max_seq_len
    }

    /// Build a handle around an existing command channel — test seam only.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn from_sender(tx: Sender<RerankCommand>, model_id: &str) -> Self {
        Self {
            tx,
            model_id: Arc::from(model_id),
            in_flight: Arc::new(AtomicUsize::new(0)),
            max_seq_len: None,
        }
    }

    /// Build a handle with an explicit `max_seq_len` — test seam for the
    /// length-gate unit tests.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn from_sender_with_max_seq_len(
        tx: Sender<RerankCommand>,
        model_id: &str,
        max_seq_len: usize,
    ) -> Self {
        Self {
            tx,
            model_id: Arc::from(model_id),
            in_flight: Arc::new(AtomicUsize::new(0)),
            max_seq_len: Some(max_seq_len),
        }
    }

    /// Rerank `documents` against `query`.
    ///
    /// # Errors
    /// Returns an error if the engine thread has exited or the pipeline call failed.
    pub async fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
    ) -> anyhow::Result<RerankOutput> {
        let started_at = Instant::now();
        let (reply_tx, reply_rx) = oneshot::channel();
        let _guard = InFlightGuard::new(&self.in_flight);

        self.tx
            .send(RerankCommand::Rerank {
                query,
                documents,
                reply: reply_tx,
                started_at,
            })
            .await
            .map_err(|_| anyhow::anyhow!("reranking engine is not available"))?;

        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("reranking engine dropped the request"))?
    }

    /// Count the tokens `query` and `document` need combined, via the
    /// pipeline's tokenizer.
    ///
    /// Not admission-gated (mirrors `embed_engine::EmbeddingHandle::count_tokens`)
    /// — a cheap tokenizer call, not a rerank, so it doesn't compete for a
    /// real rerank slot.
    ///
    /// # Errors
    /// Returns an error if the engine thread is gone or the tokenizer fails.
    pub(crate) async fn count_tokens(
        &self,
        query: String,
        document: String,
    ) -> anyhow::Result<usize> {
        let (reply, reply_rx) = oneshot::channel();
        self.tx
            .send(RerankCommand::CountTokens {
                query,
                document,
                reply,
            })
            .await
            .map_err(|_| anyhow::anyhow!("reranking engine unavailable"))?;
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("reranking engine dropped tokenize reply"))?
            .map_err(|e| anyhow::anyhow!(e))
    }
}

impl ManagedEngine for RerankingHandle {
    fn kind(&self) -> ModelKind {
        ModelKind::Reranking
    }

    fn active(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }

    fn max_concurrency(&self) -> usize {
        RERANK_CHANNEL_CAP
    }

    fn waiting(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed).saturating_sub(1)
    }
}

// ── Engine thread ──────────────────────────────────────────────────────────────

/// Spawn a dedicated OS thread owning `engine` and return a [`RerankingHandle`].
///
/// # Errors
/// Only propagates thread-spawn errors.
pub fn spawn_rerank_engine(
    mut engine: OvRerankEngine,
    model_id: &str,
    device: &str,
    max_seq_len: Option<usize>,
) -> anyhow::Result<(RerankingHandle, std::thread::JoinHandle<()>)> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<RerankCommand>(RERANK_CHANNEL_CAP);
    let model_id_arc: Arc<str> = Arc::from(model_id);
    let model_id_owned = model_id.to_owned();
    let device = device.to_owned();

    let metrics = HotMetrics::new(ModelKind::Reranking, model_id, &device);

    let thread = std::thread::Builder::new()
        .name(format!("rerank-engine-{model_id_owned}"))
        .spawn(move || {
            tracing::info!(
                model_id = model_id_owned,
                kind = ModelKind::Reranking.label(),
                device,
                "reranking engine thread started"
            );
            while let Some(command) = rx.blocking_recv() {
                match command {
                    RerankCommand::Rerank {
                        query,
                        documents,
                        reply,
                        started_at,
                    } => {
                        metrics.request_accepted(Modality::Text);
                        let result = engine
                            .rerank(&query, &documents)
                            .map(|results| RerankOutput { results });
                        // Self-healing recovery (the project's internal engineering log):
                        // a failed call can leave the pipeline's internal tensor state
                        // permanently inconsistent, so every later call on this same
                        // load would otherwise fail identically until an admin
                        // manually evicts and reloads the model. Reload immediately
                        // instead, so only this one request is lost.
                        if result.is_err() {
                            if let Err(reload_err) = engine.reload() {
                                tracing::error!(
                                    model_id = model_id_owned,
                                    error = %reload_err,
                                    "reranking engine reload after a failed call also failed — \
                                     pipeline unusable until the model is manually evicted and reloaded"
                                );
                            } else {
                                tracing::warn!(
                                    model_id = model_id_owned,
                                    "reranking engine self-healed: reloaded the pipeline after a failed call"
                                );
                            }
                        }
                        metrics.record_duration(Modality::Text, started_at.elapsed().as_secs_f64());
                        let _ = reply.send(result);
                    }
                    RerankCommand::CountTokens {
                        query,
                        document,
                        reply,
                    } => {
                        let _ = reply.send(
                            engine
                                .count_tokens(&query, &document)
                                .map_err(|e| e.to_string()),
                        );
                    }
                }
            }
            tracing::info!(
                model_id = model_id_owned,
                kind = ModelKind::Reranking.label(),
                "reranking engine thread exiting — all handles dropped"
            );
        })?;

    Ok((
        RerankingHandle {
            tx,
            model_id: model_id_arc,
            in_flight: Arc::new(AtomicUsize::new(0)),
            max_seq_len,
        },
        thread,
    ))
}

/// Spawn a GPU-free mock reranking engine for integration tests.
#[cfg(test)]
pub(crate) fn spawn_mock_rerank(model_id: &str) -> (RerankingHandle, std::thread::JoinHandle<()>) {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<RerankCommand>(RERANK_CHANNEL_CAP);
    let thread = std::thread::spawn(move || {
        while let Some(command) = rx.blocking_recv() {
            match command {
                RerankCommand::Rerank {
                    documents, reply, ..
                } => {
                    // Mock: return documents in reverse order with descending
                    // scores. `rank` is bounded by the test's own document
                    // count (never remotely near f32's 2^24 exact-integer
                    // ceiling), so the usize->f32 cast is lossless in practice.
                    #[allow(clippy::cast_precision_loss)]
                    let results = documents
                        .iter()
                        .enumerate()
                        .rev()
                        .enumerate()
                        .map(|(rank, (orig_idx, _))| (orig_idx, 1.0_f32 - rank as f32 * 0.1))
                        .collect();
                    let _ = reply.send(Ok(RerankOutput { results }));
                }
                // No real tokenizer behind the mock — answer with a rough
                // word-count stand-in so a caller exercising the length gate
                // against a mock handle doesn't hang waiting for a reply.
                RerankCommand::CountTokens {
                    query,
                    document,
                    reply,
                } => {
                    let count =
                        query.split_whitespace().count() + document.split_whitespace().count();
                    let _ = reply.send(Ok(count));
                }
            }
        }
    });
    (
        RerankingHandle {
            tx,
            model_id: Arc::from(model_id),
            in_flight: Arc::new(AtomicUsize::new(0)),
            max_seq_len: None,
        },
        thread,
    )
}

// ============================================================
// Unit tests
// ============================================================
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn waiting_is_in_flight_minus_running() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<RerankCommand>(RERANK_CHANNEL_CAP);
        let h = RerankingHandle::from_sender(tx, "test-rerank");
        assert_eq!(h.active(), 0);
        assert_eq!(h.waiting(), 0);
        h.in_flight.store(3, Ordering::Relaxed);
        assert_eq!(h.waiting(), 2);
    }

    #[tokio::test]
    async fn mock_rerank_round_trips() {
        let (handle, _thread) = spawn_mock_rerank("mock-rerank");
        let out = handle
            .rerank(
                "test query".to_owned(),
                vec!["doc0".to_owned(), "doc1".to_owned(), "doc2".to_owned()],
            )
            .await
            .expect("mock rerank");
        assert_eq!(out.results.len(), 3);
        assert_eq!(handle.active(), 0);
    }

    #[tokio::test]
    async fn rerank_errors_when_engine_gone() {
        let (tx, rx) = tokio::sync::mpsc::channel::<RerankCommand>(RERANK_CHANNEL_CAP);
        let handle = RerankingHandle::from_sender(tx, "dead-rerank");
        drop(rx);
        let err = handle.rerank("q".to_owned(), vec!["d".to_owned()]).await;
        assert!(err.is_err());
        assert_eq!(handle.active(), 0);
    }

    /// `max_seq_len()` reports `None` for a handle built without one, and
    /// `Some(n)` for one built via `from_sender_with_max_seq_len`
    /// (the project's internal engineering log's length gate).
    #[test]
    fn max_seq_len_reports_the_configured_ceiling() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<RerankCommand>(RERANK_CHANNEL_CAP);
        let no_ceiling = RerankingHandle::from_sender(tx.clone(), "test-rerank");
        assert_eq!(no_ceiling.max_seq_len(), None);

        let with_ceiling = RerankingHandle::from_sender_with_max_seq_len(tx, "test-rerank", 512);
        assert_eq!(with_ceiling.max_seq_len(), Some(512));
    }
}
