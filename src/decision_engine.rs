// ============================================================
// src/decision_engine.rs — decision engine thread (/v1/systemone)
// ============================================================
// Mirrors rerank_engine.rs: a dedicated OS thread owns the OvDecideEngine (one stateful
// InferRequest on a gathered decision IR). Commands arrive via an mpsc channel, each with a
// oneshot reply. One request at a time; its questions run sequentially, each as one fresh
// prefill (v1: no shared-prefix priming).
//
// A request is checked in full before any inference, as Ollama's `Score` does: every prompt
// is tokenised and length-checked first, so a too-long question 5 fails the request without
// spending prefills on questions 1–4.
//
// The model's decision metadata lives in a sidecar `decision.json` next to the IR (Q3):
// the prompt format, the system prompt, the candidate token ids for the codes A…Z, and the
// context length Ollama uses (`num_ctx`, 8194 for nimble).
// ============================================================

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde::Deserialize;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc::Sender, oneshot};

use crate::in_flight::InFlightGuard;
use crate::metrics::{HotMetrics, Modality};
use crate::model_manager::{ManagedEngine, ModelKind};
use crate::ov_decide::OvDecideEngine;

/// Decision command-channel capacity.
pub(crate) const DECIDE_CHANNEL_CAP: usize = 8;

/// Sidecar file name, next to `openvino_model.xml`.
pub const SIDECAR: &str = "decision.json";

/// The decision metadata a model directory must carry (`decision.json`).
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct DecisionMeta {
    /// Prompt builder: only `"ollama"` (Ollama v0.35.0 `decision.Compile` + qwen3.5) in v1.
    pub prompt_format: String,
    /// The model's system prompt (Ollama: the Modelfile `SYSTEM`).
    pub system_prompt: String,
    /// Token id of each candidate code, "A" first; at least 26 entries.
    pub candidate_ids: Vec<i64>,
    /// Context length the length rule is checked against (Ollama `num_ctx`; 8194 for nimble:
    /// 8192 prompt tokens + 2 scoring positions).
    pub context_length: usize,
    /// Licence of the weights, for the registry (e.g. "Apache-2.0").
    #[serde(default)]
    pub licence: Option<String>,
}

impl DecisionMeta {
    /// Read and validate `decision.json` in `model_dir`. Unknown fields are logged, not fatal.
    ///
    /// # Errors
    /// Missing or malformed sidecar, an unsupported `prompt_format`, fewer than 26 candidate
    /// ids, or a `context_length` below 3.
    pub fn load(model_dir: &Path) -> anyhow::Result<Self> {
        let path = model_dir.join(SIDECAR);
        let text = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("decision model needs {}: {e}", path.display()))?;
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("{} is not valid JSON: {e}", path.display()))?;
        if let Some(obj) = value.as_object() {
            const KNOWN: [&str; 5] = [
                "prompt_format",
                "system_prompt",
                "candidate_ids",
                "context_length",
                "licence",
            ];
            for key in obj.keys().filter(|k| !KNOWN.contains(&k.as_str())) {
                tracing::warn!(file = %path.display(), field = %key, "unknown field in decision sidecar (ignored)");
            }
        }
        let meta: Self = serde_json::from_value(value)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        meta.validate()?;
        Ok(meta)
    }

    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.prompt_format == "ollama",
            "decision.json prompt_format {:?} is not supported (v1 supports \"ollama\")",
            self.prompt_format
        );
        anyhow::ensure!(
            self.candidate_ids.len() >= crate::systemone::compile::MAX_CANDIDATES,
            "decision.json needs {} candidate_ids (A–Z), has {}",
            crate::systemone::compile::MAX_CANDIDATES,
            self.candidate_ids.len()
        );
        anyhow::ensure!(
            self.context_length >= 3,
            "decision.json context_length must be >= 3"
        );
        Ok(())
    }

    /// Check the sidecar against the loaded model: every code letter must encode to exactly
    /// its candidate id, every id must be inside the vocabulary, and — Ollama's per-request
    /// check, done once here — appending the code to a rendered prompt must add exactly that
    /// one token (no merge across the boundary). Every prompt ends in the same assistant
    /// prefix, so one probe prompt covers all requests.
    ///
    /// # Errors
    /// The first mismatch.
    pub fn verify(&self, engine: &OvDecideEngine) -> anyhow::Result<()> {
        let vocab = engine.vocab_size();
        let probe = crate::systemone::render_qwen35(
            &self.system_prompt,
            "{\"context\":\"probe\",\"schema\":[]}\n\nRequested field: \"a\"",
        );
        let base = engine.encode(&probe)?;
        for (i, &id) in self
            .candidate_ids
            .iter()
            .take(crate::systemone::compile::MAX_CANDIDATES)
            .enumerate()
        {
            let code = char::from(b'A' + u8::try_from(i)?);
            anyhow::ensure!(
                usize::try_from(id).is_ok_and(|v| v < vocab),
                "decision.json candidate id {id} for {code} is outside the vocabulary (0..{vocab})"
            );
            let ids = engine.encode(&code.to_string())?;
            anyhow::ensure!(
                ids == [id],
                "decision.json says {code} = {id}, but the tokenizer encodes {code} as {ids:?}"
            );
            let joined = engine.encode(&format!("{probe}{code}"))?;
            anyhow::ensure!(
                joined.len() == base.len() + 1
                    && joined.starts_with(&base)
                    && joined.last() == Some(&id),
                "candidate {code} does not append exactly one token ({id}) to a rendered prompt"
            );
        }
        Ok(())
    }
}

/// Refuse the NPU for a decision model, whatever the config says. The default placement
/// already leaves it out, but an explicit per-model `device` (or an `AUTO:`/`HETERO:` list
/// naming it) bypasses that — and the NPU plugin's LLM mode corrupts the hybrid model's
/// conv/SSM states: wrong probabilities, not a load error.
///
/// # Errors
/// When `device` names the NPU.
pub fn check_device(device: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !device.to_ascii_uppercase().contains("NPU"),
        "decision models cannot run on the NPU (device {device:?}): its LLM mode corrupts the \
         hybrid model's recurrent states — use GPU or CPU"
    );
    Ok(())
}

/// One scored request: candidate logits per question and the prompt token counts.
#[derive(Clone, Debug)]
pub struct DecideOutput {
    /// Candidate logits per question, in request order.
    pub logits: Vec<Vec<f32>>,
    /// Prompt tokens per question (Ollama `usage.input_tokens` is their sum).
    pub prompt_tokens: Vec<usize>,
    /// Prefill time per question, milliseconds.
    pub prefill_ms: Vec<f64>,
}

/// Why a decision request failed on the engine thread.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecideError {
    /// A prompt breaks Ollama's length rule (HTTP 400, Ollama's message verbatim).
    Length(String),
    /// Tokenisation or inference failed (HTTP 500); the engine has reloaded itself.
    Engine(String),
    /// The client disconnected; the remaining questions were not scored (nobody receives this).
    Cancelled,
}

impl std::fmt::Display for DecideError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Length(m) | Self::Engine(m) => f.write_str(m),
            Self::Cancelled => f.write_str("request cancelled by the client"),
        }
    }
}

/// Ollama's length rule (`llm/llama_server_score.go` @ v0.35.0), checked per prompt `index`
/// with `ctx` = the model's context length (`MaxTokens = ContextLength()`).
///
/// # Errors
/// Ollama's two 400 messages, in its order.
pub fn check_length(index: usize, n: usize, ctx: usize) -> Result<(), DecideError> {
    if n == 0 || n > ctx {
        return Err(DecideError::Length(format!(
            "prompt {index} has {n} tokens; expected 1–{ctx} (input is never truncated)"
        )));
    }
    if n + 2 > ctx {
        return Err(DecideError::Length(format!(
            "prompt {index} requires {} context tokens for scoring; model has {ctx}",
            n + 2
        )));
    }
    Ok(())
}

pub(crate) enum DecideCommand {
    Decide {
        prompts: Vec<String>,
        /// Number of candidates per question (codes A.. taken from the sidecar ids).
        widths: Vec<usize>,
        reply: oneshot::Sender<Result<DecideOutput, DecideError>>,
        started_at: Instant,
        /// Held by the engine thread until it is done with this request, NOT by the handler's
        /// future: a disconnected client drops that future while the engine may still be
        /// mid-prefill, and `active()` reaching 0 then would let eviction drop the handle and
        /// race the join deadline against a still-resident pipeline. Dropped with the command
        /// on every path (send failure, cancelled send, queued-then-skipped).
        turn: DecideTurn,
    },
}

/// One request's place at the engine: the in-flight count (so `active()` covers a request from
/// the moment it starts waiting) and, when the device is budgeted, the model's turn gate.
/// Moved into the command so the engine thread — not the handler's future — releases both.
pub struct DecideTurn {
    _guard: InFlightGuard,
    _gate: Option<OwnedSemaphorePermit>,
}

/// The turn gate was not free within the queue timeout.
#[derive(Debug, PartialEq, Eq)]
pub struct TurnTimeout;

/// Cloneable submit handle for the decision engine thread.
#[derive(Clone, Debug)]
pub struct DecisionHandle {
    tx: Sender<DecideCommand>,
    model_id: Arc<str>,
    in_flight: Arc<AtomicUsize>,
    meta: Arc<DecisionMeta>,
    /// One permit: the engine scores one request at a time, so only the request at the front
    /// needs a device lease. Requests behind it wait here, holding nothing, instead of each
    /// taking a lease they cannot use while the cross-pipeline budget starves chat.
    gate: Arc<Semaphore>,
    /// Set by the engine thread when the reload after a failed prefill also failed: the
    /// engine can no longer serve, and the model must be evicted (and reloaded) to recover.
    dead: Arc<AtomicBool>,
}

impl DecisionHandle {
    /// The model ID of the loaded decision model.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// The model's decision metadata (system prompt, candidate ids, context length).
    #[must_use]
    pub fn meta(&self) -> &DecisionMeta {
        &self.meta
    }

    /// `true` once the engine failed to reload after a failed prefill — it will fail every
    /// request until the model is evicted and loaded again.
    #[must_use]
    pub fn is_dead(&self) -> bool {
        self.dead.load(Ordering::Acquire)
    }

    /// Test seam: simulate a failed reload.
    #[cfg(test)]
    pub(crate) fn mark_dead_for_test(&self) {
        self.dead.store(true, Ordering::Release);
    }

    /// Score `prompts` (one per question); `widths[i]` is question i's candidate count.
    ///
    /// # Errors
    /// [`DecideError::Length`] for Ollama's length rule; [`DecideError::Engine`] when the
    /// engine thread is gone or inference failed.
    pub async fn decide(
        &self,
        prompts: Vec<String>,
        widths: Vec<usize>,
    ) -> Result<DecideOutput, DecideError> {
        self.decide_in_turn(self.ungated_turn(), prompts, widths)
            .await
    }

    /// A turn that takes no gate permit: the request only counts as in flight. For devices
    /// without a configured budget, where requests queue in the engine channel as before.
    #[must_use]
    pub fn ungated_turn(&self) -> DecideTurn {
        DecideTurn {
            _guard: InFlightGuard::new(&self.in_flight),
            _gate: None,
        }
    }

    /// Wait for this model's turn gate, for at most `timeout`. The request counts as in
    /// flight from the moment it starts waiting, so eviction drains it like any other.
    ///
    /// # Errors
    /// [`TurnTimeout`] when the gate stayed busy for the whole `timeout`.
    pub async fn wait_turn(&self, timeout: Duration) -> Result<DecideTurn, TurnTimeout> {
        let guard = InFlightGuard::new(&self.in_flight);
        let permit = tokio::time::timeout(timeout, Arc::clone(&self.gate).acquire_owned())
            .await
            .map_err(|_| TurnTimeout)?
            .map_err(|_| TurnTimeout)?;
        Ok(DecideTurn {
            _guard: guard,
            _gate: Some(permit),
        })
    }

    /// [`Self::decide`] for a request that already holds its `turn`.
    ///
    /// # Errors
    /// As [`Self::decide`].
    pub async fn decide_in_turn(
        &self,
        turn: DecideTurn,
        prompts: Vec<String>,
        widths: Vec<usize>,
    ) -> Result<DecideOutput, DecideError> {
        let started_at = Instant::now();
        let (reply, reply_rx) = oneshot::channel();
        self.tx
            .send(DecideCommand::Decide {
                prompts,
                widths,
                reply,
                started_at,
                turn,
            })
            .await
            .map_err(|_| DecideError::Engine("decision engine is not available".into()))?;
        reply_rx
            .await
            .map_err(|_| DecideError::Engine("decision engine dropped the request".into()))?
    }
}

impl ManagedEngine for DecisionHandle {
    fn kind(&self) -> ModelKind {
        ModelKind::Decision
    }

    fn active(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }

    fn max_concurrency(&self) -> usize {
        DECIDE_CHANNEL_CAP
    }

    fn waiting(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed).saturating_sub(1)
    }
}

/// The per-request work, on the engine thread: encode + length-check every prompt, then one
/// fresh prefill each. Generic over the scorer so the mock runs the same checks.
fn run_decide<E, P, C>(
    prompts: &[String],
    widths: &[usize],
    meta: &DecisionMeta,
    mut encode: E,
    mut prefill: P,
    cancelled: C,
) -> Result<DecideOutput, DecideError>
where
    E: FnMut(&str) -> anyhow::Result<Vec<i64>>,
    P: FnMut(&[i64], &[i64]) -> anyhow::Result<Vec<f32>>,
    C: Fn() -> bool,
{
    let mut encoded = Vec::with_capacity(prompts.len());
    for (i, p) in prompts.iter().enumerate() {
        let ids = encode(p).map_err(|e| DecideError::Engine(e.to_string()))?;
        check_length(i, ids.len(), meta.context_length)?;
        encoded.push(ids);
    }
    let mut out = DecideOutput {
        logits: Vec::with_capacity(prompts.len()),
        prompt_tokens: encoded.iter().map(Vec::len).collect(),
        prefill_ms: Vec::with_capacity(prompts.len()),
    };
    for (ids, &k) in encoded.iter().zip(widths) {
        // The client went away (its request future, and so the reply receiver, was dropped):
        // stop before the next prefill instead of scoring up to 64 questions for nobody.
        if cancelled() {
            return Err(DecideError::Cancelled);
        }
        let cand = meta.candidate_ids.get(..k).ok_or_else(|| {
            DecideError::Engine(format!(
                "{k} candidates but the sidecar has {}",
                meta.candidate_ids.len()
            ))
        })?;
        let t = Instant::now();
        out.logits
            .push(prefill(ids, cand).map_err(|e| DecideError::Engine(e.to_string()))?);
        out.prefill_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(out)
}

/// Spawn a dedicated OS thread owning `engine` and return a [`DecisionHandle`].
///
/// # Errors
/// Only propagates thread-spawn errors.
pub fn spawn_decision_engine(
    mut engine: OvDecideEngine,
    meta: DecisionMeta,
    model_id: &str,
    device: &str,
) -> anyhow::Result<(DecisionHandle, std::thread::JoinHandle<()>)> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<DecideCommand>(DECIDE_CHANNEL_CAP);
    let meta = Arc::new(meta);
    let thread_meta = Arc::clone(&meta);
    let model_id_owned = model_id.to_owned();
    let device = device.to_owned();
    let metrics = HotMetrics::new(ModelKind::Decision, model_id, &device);
    let dead = Arc::new(AtomicBool::new(false));
    let thread_dead = Arc::clone(&dead);

    let thread = std::thread::Builder::new()
        .name(format!("decide-engine-{model_id_owned}"))
        .spawn(move || {
            tracing::info!(model_id = model_id_owned, kind = ModelKind::Decision.label(), device, "decision engine thread started");
            while let Some(DecideCommand::Decide { prompts, widths, reply, started_at, turn }) = rx.blocking_recv() {
                metrics.request_accepted(Modality::Text);
                let mut failed = false;
                let result = {
                    let engine_cell = std::cell::RefCell::new(&mut engine);
                    run_decide(
                        &prompts,
                        &widths,
                        &thread_meta,
                        |p| engine_cell.borrow().encode(p),
                        |ids, cand| {
                            let r = engine_cell.borrow_mut().prefill(ids, cand);
                            failed |= r.is_err();
                            r
                        },
                        || reply.is_closed(),
                    )
                };
                // Same self-healing as the rerank engine: a failed inference may leave the
                // request's state inconsistent, so reload now and lose only this request.
                if failed {
                    match engine.reload() {
                        Ok(()) => tracing::warn!(model_id = model_id_owned, "decision engine reloaded after a failed prefill"),
                        Err(e) => {
                            thread_dead.store(true, Ordering::Release);
                            tracing::error!(model_id = model_id_owned, error = %e, "decision engine reload failed — marked dead; the next request evicts it");
                        }
                    }
                }
                metrics.record_duration(Modality::Text, started_at.elapsed().as_secs_f64());
                let _ = reply.send(result);
                // Only now is the request really finished: `active()` stays above 0 for the
                // whole prefill, so eviction waits for it instead of racing the join deadline.
                drop(turn);
            }
            tracing::info!(model_id = model_id_owned, kind = ModelKind::Decision.label(), "decision engine thread exiting — all handles dropped");
        })?;

    Ok((
        DecisionHandle {
            tx,
            model_id: Arc::from(model_id),
            in_flight: Arc::new(AtomicUsize::new(0)),
            meta,
            gate: Arc::new(Semaphore::new(1)),
            dead,
        },
        thread,
    ))
}

/// Metadata the GPU-free mock uses: 26 candidate ids 0..26, the nimble system prompt
/// stand-in, and nimble's context length.
#[cfg(test)]
pub(crate) fn mock_meta() -> DecisionMeta {
    DecisionMeta {
        prompt_format: "ollama".into(),
        system_prompt: "Mock decision system prompt.".into(),
        candidate_ids: (0..26).collect(),
        context_length: 8194,
        licence: Some("Apache-2.0".into()),
    }
}

/// Spawn a GPU-free mock decision engine: one "token" per character of the prompt (so tests
/// can hit the length boundaries exactly), and logit `j` for candidate `j` (the last
/// candidate always wins).
#[cfg(test)]
pub(crate) fn spawn_mock_decision(model_id: &str) -> (DecisionHandle, std::thread::JoinHandle<()>) {
    spawn_mock(model_id, None)
}

/// [`spawn_mock_decision`] whose every prefill signals `started` and then blocks until the
/// test sends on the paired gate sender — a prefill the test can hold open.
#[cfg(test)]
pub(crate) fn spawn_gated_mock_decision(
    model_id: &str,
    started: std::sync::mpsc::Sender<()>,
    gate: std::sync::mpsc::Receiver<()>,
) -> (DecisionHandle, std::thread::JoinHandle<()>) {
    spawn_mock(model_id, Some((started, gate)))
}

#[cfg(test)]
fn spawn_mock(
    model_id: &str,
    gate: Option<(std::sync::mpsc::Sender<()>, std::sync::mpsc::Receiver<()>)>,
) -> (DecisionHandle, std::thread::JoinHandle<()>) {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<DecideCommand>(DECIDE_CHANNEL_CAP);
    let meta = Arc::new(mock_meta());
    let thread_meta = Arc::clone(&meta);
    let thread = std::thread::spawn(move || {
        while let Some(DecideCommand::Decide {
            prompts,
            widths,
            reply,
            turn,
            ..
        }) = rx.blocking_recv()
        {
            let result = run_decide(
                &prompts,
                &widths,
                &thread_meta,
                |p| Ok(p.chars().map(|_| 1i64).collect()),
                // Candidate ids are 0..26, so the id doubles as the logit.
                #[allow(clippy::cast_precision_loss)]
                |_, cand| {
                    if let Some((started, gate)) = &gate {
                        let _ = started.send(());
                        let _ = gate.recv();
                    }
                    Ok(cand.iter().map(|&c| c as f32).collect())
                },
                || reply.is_closed(),
            );
            let _ = reply.send(result);
            drop(turn);
        }
    });
    (
        DecisionHandle {
            tx,
            model_id: Arc::from(model_id),
            in_flight: Arc::new(AtomicUsize::new(0)),
            meta,
            gate: Arc::new(Semaphore::new(1)),
            dead: Arc::new(AtomicBool::new(false)),
        },
        thread,
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use std::time::Duration;

    /// Review R13: a disconnected client stops the request before the next prefill.
    #[test]
    fn cancellation_stops_before_the_next_prefill() {
        let meta = mock_meta();
        let prompts: Vec<String> = (0..5).map(|i| format!("q{i}")).collect();
        let prefills = std::cell::Cell::new(0);
        let result = run_decide(
            &prompts,
            &[2; 5],
            &meta,
            |p| Ok(p.chars().map(|_| 1i64).collect()),
            |_, cand| {
                prefills.set(prefills.get() + 1);
                Ok(vec![0.0; cand.len()])
            },
            || prefills.get() >= 2, // the client "disconnects" after the second question
        );
        assert_eq!(result.unwrap_err(), DecideError::Cancelled);
        assert_eq!(prefills.get(), 2, "no prefill after the disconnect");
    }

    /// Review R13 follow-up: a client that disconnects mid-prefill must not make `active()`
    /// read 0 while the engine thread is still running, or eviction (which waits only while
    /// `active() > 0`) drops the handle and races the 8 s join deadline against a pipeline
    /// that is still resident. `active()` must hold until the engine is really done.
    #[tokio::test]
    async fn active_holds_until_the_engine_finishes_after_a_disconnect() {
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (gate_tx, gate_rx) = std::sync::mpsc::channel();
        let (h, thread) = spawn_gated_mock_decision("mock-decide", started_tx, gate_rx);

        let request = tokio::spawn({
            let h = h.clone();
            async move { h.decide(vec!["a".into(), "b".into()], vec![2, 2]).await }
        });
        // Wait (off the runtime thread) until the engine is inside the first prefill.
        tokio::task::spawn_blocking(move || {
            started_rx.recv().unwrap();
            started_rx
        })
        .await
        .unwrap();
        assert_eq!(h.active(), 1);

        // The client disconnects: its request future is dropped mid-await.
        request.abort();
        let _ = request.await;
        assert_eq!(
            h.active(),
            1,
            "the engine is still mid-prefill — active() must not drop to 0 yet"
        );

        // Let the prefill finish: the engine sees the closed reply, cancels before the
        // second question, and only then releases the request.
        gate_tx.send(()).unwrap();
        for _ in 0..200 {
            if h.active() == 0 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert_eq!(
            h.active(),
            0,
            "released once the engine finished the request"
        );

        drop(h);
        drop(gate_tx);
        tokio::task::spawn_blocking(move || thread.join().unwrap())
            .await
            .unwrap();
    }

    /// The turn gate lets one request through at a time; a request that is waiting counts as
    /// in flight (so eviction drains it) and one that gives up stops counting.
    #[tokio::test]
    async fn turn_gate_admits_one_and_counts_waiters() {
        let (h, _t) = spawn_mock_decision("mock-decide");
        let first = h.wait_turn(Duration::from_millis(50)).await.unwrap();
        assert_eq!(h.active(), 1);

        assert_eq!(
            h.wait_turn(Duration::from_millis(30)).await.err(),
            Some(TurnTimeout),
            "the gate is held"
        );
        assert_eq!(h.active(), 1, "a request that timed out no longer counts");

        let waiter = tokio::spawn({
            let h = h.clone();
            async move { h.wait_turn(Duration::from_secs(2)).await.is_ok() }
        });
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(h.active(), 2, "a request waiting at the gate is in flight");

        drop(first);
        assert!(
            waiter.await.unwrap(),
            "the waiter got the gate once it freed"
        );
        assert_eq!(h.active(), 0);
    }

    #[test]
    fn npu_is_refused_in_any_spelling() {
        for bad in [
            "NPU",
            "npu",
            "NPU.0",
            "AUTO:NPU,GPU",
            "HETERO:GPU,NPU",
            "MULTI:npu",
        ] {
            assert!(check_device(bad).is_err(), "{bad} must be refused");
        }
        for ok in ["GPU", "GPU.1", "CPU", "AUTO:GPU,CPU", "HETERO:GPU,CPU"] {
            assert!(check_device(ok).is_ok(), "{ok} must be allowed");
        }
    }

    #[test]
    fn length_rule_matches_ollama_boundaries() {
        // ctx 8194 (nimble num_ctx): 8192 fits; 8193/8194 need 2 more; 8195 is over.
        assert_eq!(check_length(0, 8192, 8194), Ok(()));
        assert_eq!(
            check_length(3, 8193, 8194),
            Err(DecideError::Length(
                "prompt 3 requires 8195 context tokens for scoring; model has 8194".into()
            ))
        );
        assert_eq!(
            check_length(1, 8195, 8194),
            Err(DecideError::Length(
                "prompt 1 has 8195 tokens; expected 1–8194 (input is never truncated)".into()
            ))
        );
        assert!(matches!(
            check_length(0, 0, 8194),
            Err(DecideError::Length(_))
        ));
    }

    #[tokio::test]
    async fn mock_scores_every_question_and_checks_all_lengths_first() {
        let (h, _t) = spawn_mock_decision("mock-decide");
        let out = h
            .decide(vec!["abc".into(), "de".into()], vec![3, 2])
            .await
            .unwrap();
        assert_eq!(out.logits, vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0]]);
        assert_eq!(out.prompt_tokens, vec![3, 2]);
        // The second prompt is too long: the whole request fails, naming prompt 1.
        let err = h
            .decide(vec!["ok".into(), "x".repeat(8195)], vec![2, 2])
            .await
            .unwrap_err();
        assert_eq!(
            err,
            DecideError::Length(
                "prompt 1 has 8195 tokens; expected 1–8194 (input is never truncated)".into()
            )
        );
        assert_eq!(h.active(), 0);
    }

    #[test]
    fn sidecar_validation() {
        let dir = tempfile::tempdir().unwrap();
        assert!(DecisionMeta::load(dir.path()).is_err(), "missing sidecar");
        let good = serde_json::json!({"prompt_format": "ollama", "system_prompt": "s",
            "candidate_ids": (32..58).collect::<Vec<i64>>(), "context_length": 8194, "licence": "Apache-2.0", "extra": 1});
        std::fs::write(dir.path().join(SIDECAR), good.to_string()).unwrap();
        let meta = DecisionMeta::load(dir.path()).unwrap();
        assert_eq!(meta.candidate_ids[0], 32);
        let bad = serde_json::json!({"prompt_format": "bespoke", "system_prompt": "s",
            "candidate_ids": (32..58).collect::<Vec<i64>>(), "context_length": 8194});
        std::fs::write(dir.path().join(SIDECAR), bad.to_string()).unwrap();
        assert!(
            DecisionMeta::load(dir.path()).is_err(),
            "unsupported format"
        );
    }
}
