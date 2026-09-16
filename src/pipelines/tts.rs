// ============================================================
// src/pipelines/tts.rs — text-to-speech engine threads + shared handle (Phase 5.2)
// ============================================================
// Three TTS backends share the same `TtsHandle` submit surface:
//
//   SpeechT5 (5.2a) — OpenVINO Text2SpeechPipeline, 16 kHz mono f32.
//     `spawn_tts_engine(engine, …)` — dedicated OS thread; OV C++ calls block it.
//
//   Kokoro-82M (5.2b) — ONNX Runtime via `kokoro_tts::KokoroTts`, 24 kHz mono f32.
//     `spawn_kokoro_engine(onnx_path, voices_path, …)` — dedicated OS thread with a
//     mini tokio runtime (Kokoro synthesis is async-native via ORT).
//
//   Coqui `pl/mai_female/vits` (5.2c, Polish) — direct `ort` session, 22.05 kHz
//     mono f32. `spawn_coqui_vits_engine(onnx_path, tokens_path, …)` — dedicated
//     OS thread, fully synchronous ORT calls (no embedded runtime needed, unlike
//     Kokoro). Character-based model (no phonemizer) — see `dev/DECISIONS.md` for
//     why this replaced an earlier Piper/espeak-ng prototype (GPL-3.0, incompatible
//     with RustedVINO's BSL→Apache-2.0 licensing plan).
//
// All three return `(TtsHandle, std::thread::JoinHandle<()>)`.  The handle carries
// the backend's sample rate so the HTTP layer writes the correct WAV header
// without knowing which engine is behind it.
// ============================================================

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::Context as _;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc::Sender, oneshot};

use crate::cb_engine::AdmitError;
use crate::metrics::{HotMetrics, Modality};
use crate::model_manager::{ManagedEngine, ModelKind};
use crate::ov_tts::OvTtsEngine;

/// TTS admission-gate size (also the reported `max_concurrency`): up to this
/// many requests may be queued/synthesizing before the gate returns
/// [`AdmitError::AtCapacity`]. Mirrors `cb_engine`/`vlm_engine`'s admission
/// pattern (the project's internal engineering log) — a permit is acquired
/// *before* a command is sent and held by the engine thread until it replies,
/// so `active()` is honest occupancy rather than the old `in_flight`
/// approximation.
pub(crate) const TTS_CHANNEL_CAP: usize = 4;

/// Output sample rate of `SpeechT5` (and the mock engine). 16 kHz mono f32.
pub const TTS_SAMPLE_RATE: u32 = 16_000;

/// Output sample rate of Kokoro-82M. 24 kHz mono f32.
pub const KOKORO_SAMPLE_RATE: u32 = 24_000;

/// Output sample rate of the Coqui `pl/mai_female/vits` model. 22.05 kHz mono f32.
pub const COQUI_VITS_PL_SAMPLE_RATE: u32 = 22_050;

// ── Commands ────────────────────────────────────────────────────────────────

pub(crate) enum TtsCommand {
    /// Synthesise speech for `text`.
    Synthesize {
        text: String,
        /// Voice name — unused by `SpeechT5` (default speaker embedding);
        /// mapped to a `kokoro_tts::Voice` variant for Kokoro.
        voice: Option<String>,
        /// Playback speed multiplier (0.25–4.0).  `SpeechT5` ignores this;
        /// Kokoro passes it as the per-voice speed parameter; Coqui VITS maps
        /// it to `length_scale = 1.0 / speed` (VITS's own rate control —
        /// smaller `length_scale` stretches phoneme duration *down*, i.e.
        /// faster speech, so the mapping inverts).
        speed: f32,
        reply: oneshot::Sender<anyhow::Result<Vec<f32>>>,
        started_at: Instant,
        /// Admission-gate permit, held until this synthesis finishes. Dropped
        /// after the engine thread sends its reply, releasing the slot for
        /// the next waiting caller. See [`TtsHandle::synthesize`].
        permit: OwnedSemaphorePermit,
    },
}

// ── Handle ──────────────────────────────────────────────────────────────────

/// Cloneable submit handle for any TTS engine thread.
///
/// `synthesize` is request/response: sends the text and awaits the PCM reply.
/// The engine thread processes one job at a time.
#[derive(Clone, Debug)]
pub struct TtsHandle {
    tx: Sender<TtsCommand>,
    model_id: Arc<str>,
    /// Admission-gate permits — `cap` of them. Mirrors `cb_engine::EngineHandle`'s
    /// `sem` field exactly.
    sem: Arc<Semaphore>,
    /// Channel capacity = the semaphore's permit count = [`TTS_CHANNEL_CAP`].
    cap: usize,
    /// Maximum time [`synthesize`](Self::synthesize) waits for a permit before
    /// returning [`AdmitError::AtCapacity`] (→ HTTP 429). Distinct from the
    /// fixed 10-minute ceiling on the synthesis reply itself, below.
    queue_timeout: Duration,
    /// Callers currently parked in [`synthesize`](Self::synthesize) awaiting an
    /// admission permit — the engine's honest `rustedvino_requests_waiting`.
    waiting: Arc<AtomicUsize>,
    /// PCM sample rate of this engine's output (`16_000` for `SpeechT5`, `24_000` for Kokoro).
    sample_rate: u32,
}

impl TtsHandle {
    /// The model ID of the loaded TTS model.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Sample rate of the PCM output (Hz). Use this to build the WAV header —
    /// different backends output different rates.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Build a handle around an existing command channel — test seam only.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn from_sender(tx: Sender<TtsCommand>, model_id: &str) -> Self {
        let cap = 1000; // generous test cap — never hits the admission gate
        Self {
            tx,
            model_id: Arc::from(model_id),
            sem: Arc::new(Semaphore::new(cap)),
            cap,
            queue_timeout: Duration::from_secs(30),
            waiting: Arc::new(AtomicUsize::new(0)),
            sample_rate: TTS_SAMPLE_RATE,
        }
    }

    /// Creates a test handle with an explicit cap and timeout. Used by
    /// admission-gate tests that need to observe `AtCapacity` behaviour.
    #[cfg(test)]
    pub(crate) fn from_sender_with_cap(
        tx: Sender<TtsCommand>,
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
            sample_rate: TTS_SAMPLE_RATE,
        }
    }

    /// Synthesise speech for `text`. Returns mono f32 PCM at `self.sample_rate()`.
    ///
    /// `voice` selects the Kokoro voice (e.g. `"af_heart"`); `SpeechT5` ignores it.
    /// `speed` is the playback rate multiplier (0.25–4.0); `SpeechT5` ignores it.
    ///
    /// Waits up to `queue_timeout` for an admission permit before returning
    /// [`AdmitError::AtCapacity`] (→ HTTP 429). Returns [`AdmitError::EngineDead`]
    /// if the engine thread has exited (→ HTTP 503) or the 10-minute synthesis
    /// ceiling below lapses, or [`AdmitError::Failed`] if the pipeline call itself
    /// failed.
    ///
    /// # Errors
    /// See the variants above: [`AdmitError::AtCapacity`],
    /// [`AdmitError::EngineDead`], [`AdmitError::Failed`].
    pub async fn synthesize(
        &self,
        text: String,
        voice: Option<String>,
        speed: f32,
    ) -> Result<Vec<f32>, AdmitError> {
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

        let sent = self
            .tx
            .send(TtsCommand::Synthesize {
                text,
                voice,
                speed,
                reply: reply_tx,
                started_at,
                permit,
            })
            .await;
        if sent.is_err() {
            return Err(AdmitError::EngineDead);
        }

        // 10-minute hard ceiling — SpeechT5 decoder can loop indefinitely on
        // pathological input (no EOS). First call on a new device triggers OV
        // kernel compilation (~6 min); subsequent calls are 2–8 s. The engine
        // thread stays stuck until the C++ call returns, but the realtime
        // handler gets an error and moves on.
        let result = tokio::time::timeout(Duration::from_mins(10), reply_rx)
            .await
            .map_err(|_| AdmitError::EngineDead)?
            .map_err(|_| AdmitError::EngineDead)?;
        result.map_err(AdmitError::Failed)
    }
}

impl ManagedEngine for TtsHandle {
    fn kind(&self) -> ModelKind {
        ModelKind::Tts
    }

    /// Accepted-but-unfinished requests (queued + the one synthesizing) —
    /// derived from the semaphore: real occupancy, not an approximation.
    fn active(&self) -> usize {
        self.cap.saturating_sub(self.sem.available_permits())
    }

    fn max_concurrency(&self) -> usize {
        self.cap
    }

    /// Callers currently parked in [`synthesize`](Self::synthesize) awaiting an
    /// admission permit. Mirrors `cb_engine::EngineHandle::waiting` exactly.
    fn waiting(&self) -> usize {
        self.waiting.load(Ordering::Relaxed)
    }
}

// ── SpeechT5 engine thread ────────────────────────────────────────────────────

/// Spawn a dedicated OS thread owning `engine` and return a [`TtsHandle`].
///
/// The thread exits cleanly when all handle clones are dropped (channel closes).
///
/// # Errors
/// Only propagates thread-spawn errors (rare OS resource exhaustion).
pub fn spawn_tts_engine(
    engine: OvTtsEngine,
    model_id: &str,
    device: &str,
    queue_timeout: Duration,
) -> anyhow::Result<(TtsHandle, std::thread::JoinHandle<()>)> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<TtsCommand>(TTS_CHANNEL_CAP);
    let sem = Arc::new(Semaphore::new(TTS_CHANNEL_CAP));
    let model_id_arc: Arc<str> = Arc::from(model_id);
    let model_id_owned = model_id.to_owned();
    let device = device.to_owned();

    let metrics = HotMetrics::new(ModelKind::Tts, model_id, &device);

    let thread = std::thread::Builder::new()
        .name(format!("tts-engine-{model_id_owned}"))
        .spawn(move || {
            tracing::info!(
                model_id = %model_id_owned,
                kind = ModelKind::Tts.label(),
                device = %device,
                "SpeechT5 TTS engine thread started"
            );
            while let Some(TtsCommand::Synthesize {
                text,
                voice: _voice,
                speed: _speed,
                reply,
                started_at,
                permit,
            }) = rx.blocking_recv()
            {
                metrics.request_accepted(Modality::Text);
                let result = engine.synthesize(&text);
                let elapsed = started_at.elapsed();
                metrics.record_duration(Modality::Text, elapsed.as_secs_f64());
                if let Ok(ref samples) = result {
                    #[allow(clippy::cast_precision_loss)]
                    // len() << 2^53; precision loss negligible
                    let duration_s = samples.len() as f64 / f64::from(TTS_SAMPLE_RATE);
                    let rtf = elapsed.as_secs_f64() / duration_s.max(0.001);
                    metrics.record_rtf(rtf);
                    tracing::info!(
                        model_id = %model_id_owned,
                        elapsed_ms = elapsed.as_millis(),
                        audio_s = format!("{duration_s:.2}"),
                        rtf = format!("{rtf:.2}"),
                        "SpeechT5 synthesis complete"
                    );
                }
                // Client may have disconnected — ignore a dropped receiver.
                let _ = reply.send(result);
                // Balance the handle's admission acquire, whatever the outcome —
                // releases the slot for the next waiting caller.
                drop(permit);
            }
            tracing::info!(
                model_id = %model_id_owned,
                kind = ModelKind::Tts.label(),
                "SpeechT5 TTS engine thread exiting — all handles dropped"
            );
        })?;

    Ok((
        TtsHandle {
            tx,
            model_id: model_id_arc,
            sem,
            cap: TTS_CHANNEL_CAP,
            queue_timeout,
            waiting: Arc::new(AtomicUsize::new(0)),
            sample_rate: TTS_SAMPLE_RATE,
        },
        thread,
    ))
}

// ── Kokoro-82M engine thread ──────────────────────────────────────────────────

/// Spawn a dedicated OS thread owning a mini tokio runtime + Kokoro TTS engine.
///
/// Blocks the caller until the model finishes loading (the `KokoroTts::new` call
/// completes inside the spawned thread and signals back via a sync channel), then
/// returns the handle. Load errors are propagated back to the caller.
///
/// The mini-runtime lives for the lifetime of the engine thread; synthesis is
/// `async` via ORT internally, so the thread needs its own runtime — it must not
/// share the server's runtime to avoid blocking async worker threads.
///
/// # Errors
/// Propagates Kokoro model-load errors and thread-spawn errors.
pub fn spawn_kokoro_engine(
    onnx_path: String,
    voices_path: String,
    model_id: &str,
    device: &str,
    queue_timeout: Duration,
) -> anyhow::Result<(TtsHandle, std::thread::JoinHandle<()>)> {
    let (cmd_tx, cmd_rx) = tokio::sync::mpsc::channel::<TtsCommand>(TTS_CHANNEL_CAP);
    let sem = Arc::new(Semaphore::new(TTS_CHANNEL_CAP));
    // Sync channel: the engine thread signals load success/failure to this thread.
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<anyhow::Result<()>>();

    let model_id_arc: Arc<str> = Arc::from(model_id);
    let model_id_owned = model_id.to_owned();
    let device_owned = device.to_owned();
    let metrics = HotMetrics::new(ModelKind::Tts, model_id, device);

    let thread = std::thread::Builder::new()
        .name(format!("kokoro-tts-{model_id_owned}"))
        .spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(r) => r,
                Err(e) => {
                    let _ = ready_tx.send(Err(anyhow::anyhow!("tokio runtime: {e}")));
                    return;
                }
            };

            rt.block_on(async move {
                let tts = match kokoro_tts::KokoroTts::new(&onnx_path, &voices_path).await {
                    Ok(t) => {
                        tracing::info!(
                            model_id = %model_id_owned,
                            device = %device_owned,
                            "Kokoro TTS engine loaded"
                        );
                        let _ = ready_tx.send(Ok(()));
                        t
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(anyhow::anyhow!("Kokoro load: {e}")));
                        return;
                    }
                };

                let mut rx = cmd_rx;
                while let Some(TtsCommand::Synthesize {
                    text,
                    voice,
                    speed,
                    reply,
                    started_at,
                    permit,
                }) = rx.recv().await
                {
                    metrics.request_accepted(Modality::Text);
                    let v = voice_from_str(voice.as_deref(), speed);
                    let result = tts
                        .synth(&text, v)
                        .await
                        .map(|(samples, _dur)| samples)
                        .map_err(|e| anyhow::anyhow!("Kokoro synthesis: {e}"));
                    let elapsed = started_at.elapsed();
                    metrics.record_duration(Modality::Text, elapsed.as_secs_f64());
                    if let Ok(ref samples) = result {
                        #[allow(clippy::cast_precision_loss)]
                        let duration_s = samples.len() as f64 / f64::from(KOKORO_SAMPLE_RATE);
                        let rtf = elapsed.as_secs_f64() / duration_s.max(0.001);
                        metrics.record_rtf(rtf);
                        tracing::info!(
                            model_id = %model_id_owned,
                            elapsed_ms = elapsed.as_millis(),
                            audio_s = format!("{duration_s:.2}"),
                            rtf = format!("{rtf:.2}"),
                            "Kokoro synthesis complete"
                        );
                    }
                    let _ = reply.send(result);
                    drop(permit);
                }
                tracing::info!(
                    model_id = %model_id_owned,
                    "Kokoro TTS engine thread exiting — all handles dropped"
                );
            });
        })?;

    // Block until load succeeds or fails.  `recv` errors if the thread panicked
    // before sending — treat that as a load failure.
    ready_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("Kokoro engine exited before signaling ready"))??;

    Ok((
        TtsHandle {
            tx: cmd_tx,
            model_id: model_id_arc,
            sem,
            cap: TTS_CHANNEL_CAP,
            queue_timeout,
            waiting: Arc::new(AtomicUsize::new(0)),
            sample_rate: KOKORO_SAMPLE_RATE,
        },
        thread,
    ))
}

/// Map an `Option<&str>` `OpenAI` voice name + speed to a [`kokoro_tts::Voice`] variant.
///
/// Defaults to `af_heart` (American Female, natural) when the name is absent or
/// unrecognised. Unknown names emit a `warn` log rather than an error so callers
/// get audio even with a typo.
fn voice_from_str(name: Option<&str>, speed: f32) -> kokoro_tts::Voice {
    match name.unwrap_or("af_heart") {
        "af_heart" => kokoro_tts::Voice::AfHeart(speed),
        "af_bella" => kokoro_tts::Voice::AfBella(speed),
        "af_nicole" => kokoro_tts::Voice::AfNicole(speed),
        "af_sky" => kokoro_tts::Voice::AfSky(speed),
        "af_kore" => kokoro_tts::Voice::AfKore(speed),
        "af_aoede" => kokoro_tts::Voice::AfAoede(speed),
        "af_nova" => kokoro_tts::Voice::AfNova(speed),
        "af_sarah" => kokoro_tts::Voice::AfSarah(speed),
        "af_jessica" => kokoro_tts::Voice::AfJessica(speed),
        "af_alloy" => kokoro_tts::Voice::AfAlloy(speed),
        "af_river" => kokoro_tts::Voice::AfRiver(speed),
        "bf_emma" => kokoro_tts::Voice::BfEmma(speed),
        "bf_isabella" => kokoro_tts::Voice::BfIsabella(speed),
        "bf_alice" => kokoro_tts::Voice::BfAlice(speed),
        "bf_lily" => kokoro_tts::Voice::BfLily(speed),
        "am_adam" => kokoro_tts::Voice::AmAdam(speed),
        "am_michael" => kokoro_tts::Voice::AmMichael(speed),
        "am_echo" => kokoro_tts::Voice::AmEcho(speed),
        "am_eric" => kokoro_tts::Voice::AmEric(speed),
        "am_liam" => kokoro_tts::Voice::AmLiam(speed),
        "am_onyx" => kokoro_tts::Voice::AmOnyx(speed),
        "am_puck" => kokoro_tts::Voice::AmPuck(speed),
        "am_fenrir" => kokoro_tts::Voice::AmFenrir(speed),
        "bm_george" => kokoro_tts::Voice::BmGeorge(speed),
        "bm_lewis" => kokoro_tts::Voice::BmLewis(speed),
        "bm_daniel" => kokoro_tts::Voice::BmDaniel(speed),
        "bm_fable" => kokoro_tts::Voice::BmFable(speed),
        other => {
            tracing::warn!(voice = %other, "unknown Kokoro voice — falling back to af_heart");
            kokoro_tts::Voice::AfHeart(speed)
        }
    }
}

// ── Coqui `pl/mai_female/vits` engine thread (Polish) ─────────────────────────

/// VITS `add_blank` token — interleaved between every input token (and at the
/// start), matching this model's own export convention (`add_blank=1` in the
/// ONNX's embedded metadata) and empirically verified against the real model.
const COQUI_VITS_BLANK_ID: i64 = 3;

/// Default inference scales for the `pl/mai_female/vits` checkpoint, taken from
/// its own training config (`model_args.inference_noise_scale{,_dp}`, `length_scale`)
/// — not the generic VITS/Piper defaults, which differ per voice.
///
/// `length_scale` is the checkpoint's *default* pace, applied when the caller
/// asks for `speed: 1.0`; the request's `speed` field scales it at call time
/// (see [`coqui_vits_synthesize`]) rather than baking in one fixed rate.
const COQUI_VITS_NOISE_SCALE: f32 = 0.3;
const COQUI_VITS_LENGTH_SCALE: f32 = 1.0;
const COQUI_VITS_NOISE_SCALE_DP: f32 = 0.3;

/// Coqui's own `multilingual_cleaners(text)` (the cleaner this model was trained
/// with, `lang=None`), reimplemented: lowercase; `;`/`:` → `,`; `-` → ` `; strip
/// `<>()[]"`; collapse and trim whitespace. See `TTS/tts/utils/text/cleaners.py`
/// in `coqui-ai/TTS` (MPL-2.0, referenced only as documentation of the exact
/// preprocessing contract — no code taken from it).
fn clean_text(text: &str) -> String {
    let normalized: String = text
        .to_lowercase()
        .chars()
        .map(|c| match c {
            ';' | ':' => ',',
            '-' => ' ',
            other => other,
        })
        .filter(|c| !matches!(c, '<' | '>' | '(' | ')' | '[' | ']' | '"'))
        .collect();
    normalized.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Load the `tokens.txt` vocab (`"<char-or-tag> <id>"` per line, one entry per
/// line) shipped alongside the ONNX model. Multi-character entries (`<PAD>`,
/// `<EOS>`, `<BOS>`, `<BLNK>`) are skipped — this model's `add_blank` scheme only
/// ever needs [`COQUI_VITS_BLANK_ID`], not the other special tokens.
fn load_coqui_vits_vocab(path: &Path) -> anyhow::Result<HashMap<char, i64>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut map = HashMap::new();
    for line in text.lines() {
        let Some((token, id_str)) = line.rsplit_once(' ') else {
            continue;
        };
        let Ok(id) = id_str.parse::<i64>() else {
            continue;
        };
        let mut chars = token.chars();
        if let (Some(ch), None) = (chars.next(), chars.next()) {
            map.insert(ch, id);
        }
    }
    anyhow::ensure!(
        !map.is_empty(),
        "no character entries found in {}",
        path.display()
    );
    Ok(map)
}

/// Map cleaned text to token ids, interleaving [`COQUI_VITS_BLANK_ID`] the same
/// way this model's own ONNX export does (verified against the real model:
/// `[blank, id_0, blank, id_1, blank, …]`). Characters absent from the vocab are
/// dropped silently (matches Coqui's own tokenizer behavior for out-of-vocab
/// characters).
fn coqui_vits_text_to_ids(vocab: &HashMap<char, i64>, text: &str) -> Vec<i64> {
    let cleaned = clean_text(text);
    let mut ids = Vec::with_capacity(cleaned.len() * 2 + 1);
    ids.push(COQUI_VITS_BLANK_ID);
    for ch in cleaned.chars() {
        if let Some(&id) = vocab.get(&ch) {
            ids.push(id);
            ids.push(COQUI_VITS_BLANK_ID);
        }
    }
    ids
}

/// Run one synthesis pass through the loaded ORT session.
///
/// `speed` is the OpenAI-style request multiplier (0.25–4.0; 1.0 = the
/// checkpoint's trained default pace). VITS's own rate knob is
/// `length_scale`, which moves *opposite* to speed — doubling `length_scale`
/// halves the rate — so it's derived as `COQUI_VITS_LENGTH_SCALE / speed`.
/// Clamped defensively in case a caller sends 0 or a negative value (the
/// public field is `f32`, not validated at the HTTP boundary).
fn coqui_vits_synthesize(
    session: &mut ort::session::Session,
    vocab: &HashMap<char, i64>,
    text: &str,
    speed: f32,
) -> anyhow::Result<Vec<f32>> {
    let ids = coqui_vits_text_to_ids(vocab, text);
    anyhow::ensure!(!ids.is_empty(), "no recognized characters in input text");
    let input_len =
        i64::try_from(ids.len()).context("input text too long (token count overflows i64)")?;

    let length_scale = COQUI_VITS_LENGTH_SCALE / speed.clamp(0.25, 4.0);
    let input_t = ort::value::Tensor::<i64>::from_array(([1_i64, input_len], ids))
        .context("building `input` tensor")?;
    let lengths_t = ort::value::Tensor::<i64>::from_array(([1_i64], vec![input_len]))
        .context("building `input_lengths` tensor")?;
    let scales_t = ort::value::Tensor::<f32>::from_array((
        [3_i64],
        vec![
            COQUI_VITS_NOISE_SCALE,
            length_scale,
            COQUI_VITS_NOISE_SCALE_DP,
        ],
    ))
    .context("building `scales` tensor")?;
    // Single-speaker, single-language model (`n_speakers=0`) — the graph still
    // declares `sid`/`langid` as required inputs, so both are always supplied
    // as index 0 (verified against the real model).
    let sid_t = ort::value::Tensor::<i64>::from_array(([1_i64], vec![0_i64]))
        .context("building `sid` tensor")?;
    let langid_t = ort::value::Tensor::<i64>::from_array(([1_i64], vec![0_i64]))
        .context("building `langid` tensor")?;

    let outputs = session
        .run(ort::inputs![input_t, lengths_t, scales_t, sid_t, langid_t])
        .context("ORT session run failed")?;
    let (_shape, audio) = outputs[0]
        .try_extract_tensor::<f32>()
        .context("extracting `output` tensor")?;
    Ok(audio.to_vec())
}

/// Spawn a dedicated OS thread owning an `ort` session for the Coqui
/// `pl/mai_female/vits` model.
///
/// Blocks the caller until the model finishes loading (mirrors
/// [`spawn_kokoro_engine`]'s ready-signal pattern), then returns the handle.
/// Unlike Kokoro, synthesis here is fully synchronous — no embedded tokio
/// runtime is needed inside the thread.
///
/// # Errors
/// Propagates vocab/model load errors and thread-spawn errors.
pub fn spawn_coqui_vits_engine(
    onnx_path: String,
    tokens_path: String,
    model_id: &str,
    device: &str,
    queue_timeout: Duration,
) -> anyhow::Result<(TtsHandle, std::thread::JoinHandle<()>)> {
    let (cmd_tx, mut cmd_rx) = tokio::sync::mpsc::channel::<TtsCommand>(TTS_CHANNEL_CAP);
    let sem = Arc::new(Semaphore::new(TTS_CHANNEL_CAP));
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<anyhow::Result<()>>();

    let model_id_arc: Arc<str> = Arc::from(model_id);
    let model_id_owned = model_id.to_owned();
    let device_owned = device.to_owned();
    let metrics = HotMetrics::new(ModelKind::Tts, model_id, device);

    let thread = std::thread::Builder::new()
        .name(format!("coqui-vits-tts-{model_id_owned}"))
        .spawn(move || {
            let vocab = match load_coqui_vits_vocab(Path::new(&tokens_path)) {
                Ok(v) => v,
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            let mut session = match ort::session::Session::builder()
                .and_then(|mut b| b.commit_from_file(&onnx_path))
                .with_context(|| format!("loading Coqui VITS ONNX model {onnx_path}"))
            {
                Ok(s) => s,
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            tracing::info!(
                model_id = %model_id_owned,
                device = %device_owned,
                "Coqui VITS (Polish) TTS engine loaded"
            );
            let _ = ready_tx.send(Ok(()));

            while let Some(TtsCommand::Synthesize {
                text,
                voice: _voice,
                speed,
                reply,
                started_at,
                permit,
            }) = cmd_rx.blocking_recv()
            {
                metrics.request_accepted(Modality::Text);
                let result = coqui_vits_synthesize(&mut session, &vocab, &text, speed);
                let elapsed = started_at.elapsed();
                metrics.record_duration(Modality::Text, elapsed.as_secs_f64());
                if let Ok(ref samples) = result {
                    #[allow(clippy::cast_precision_loss)]
                    let duration_s = samples.len() as f64 / f64::from(COQUI_VITS_PL_SAMPLE_RATE);
                    let rtf = elapsed.as_secs_f64() / duration_s.max(0.001);
                    metrics.record_rtf(rtf);
                    tracing::info!(
                        model_id = %model_id_owned,
                        elapsed_ms = elapsed.as_millis(),
                        audio_s = format!("{duration_s:.2}"),
                        rtf = format!("{rtf:.2}"),
                        "Coqui VITS synthesis complete"
                    );
                }
                let _ = reply.send(result);
                drop(permit);
            }
            tracing::info!(
                model_id = %model_id_owned,
                "Coqui VITS TTS engine thread exiting — all handles dropped"
            );
        })?;

    ready_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("Coqui VITS engine exited before signaling ready"))??;

    Ok((
        TtsHandle {
            tx: cmd_tx,
            model_id: model_id_arc,
            sem,
            cap: TTS_CHANNEL_CAP,
            queue_timeout,
            waiting: Arc::new(AtomicUsize::new(0)),
            sample_rate: COQUI_VITS_PL_SAMPLE_RATE,
        },
        thread,
    ))
}

// ── Mock engine (tests only) ──────────────────────────────────────────────────

/// Spawn a GPU-free mock TTS engine — the `Tts` analogue of `spawn_mock_stt`.
///
/// Answers each request with one second of 16 kHz silence so that handler
/// tests yield a valid PCM → WAV response without touching a GPU.
#[cfg(test)]
pub(crate) fn spawn_mock_tts(model_id: &str) -> (TtsHandle, std::thread::JoinHandle<()>) {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<TtsCommand>(TTS_CHANNEL_CAP);
    let sem = Arc::new(Semaphore::new(TTS_CHANNEL_CAP));

    let thread = std::thread::spawn(move || {
        while let Some(TtsCommand::Synthesize { reply, permit, .. }) = rx.blocking_recv() {
            // One second of 16 kHz silence.
            let _ = reply.send(Ok(vec![0.0_f32; TTS_SAMPLE_RATE as usize]));
            drop(permit);
        }
    });

    (
        TtsHandle {
            tx,
            model_id: Arc::from(model_id),
            sem,
            cap: TTS_CHANNEL_CAP,
            queue_timeout: Duration::from_secs(30),
            waiting: Arc::new(AtomicUsize::new(0)),
            sample_rate: TTS_SAMPLE_RATE,
        },
        thread,
    )
}

// ============================================================
// Unit tests (no GPU)
// ============================================================
#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::cast_possible_truncation
    )]
    use super::*;

    #[test]
    fn clean_text_lowercases_and_maps_symbols() {
        assert_eq!(clean_text("Cześć; jak-się: masz?"), "cześć, jak się, masz?");
    }

    #[test]
    fn clean_text_strips_aux_symbols_and_collapses_whitespace() {
        assert_eq!(clean_text("Hello  <world>   (test)"), "hello world test");
    }

    #[test]
    fn coqui_vits_text_to_ids_interleaves_blank_and_skips_unknown_chars() {
        let mut vocab = HashMap::new();
        vocab.insert('a', 10);
        vocab.insert('b', 11);
        // '#' is not in the vocab and must be dropped silently.
        let ids = coqui_vits_text_to_ids(&vocab, "a#b");
        assert_eq!(
            ids,
            vec![
                COQUI_VITS_BLANK_ID,
                10,
                COQUI_VITS_BLANK_ID,
                11,
                COQUI_VITS_BLANK_ID,
            ]
        );
    }

    #[test]
    fn load_coqui_vits_vocab_parses_chars_and_skips_multi_char_tokens() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tokens.txt");
        std::fs::write(&path, "<PAD> 0\n<EOS> 1\na 4\nA 4\nb 5\n").expect("write tokens.txt");
        let vocab = load_coqui_vits_vocab(&path).expect("load vocab");
        assert_eq!(vocab.get(&'a'), Some(&4));
        assert_eq!(vocab.get(&'A'), Some(&4));
        assert_eq!(vocab.get(&'b'), Some(&5));
        assert_eq!(vocab.len(), 3, "special multi-char tokens must be skipped");
    }

    #[test]
    fn idle_handle_reports_zero_active_and_waiting() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<TtsCommand>(TTS_CHANNEL_CAP);
        let h = TtsHandle::from_sender(tx, "test-tts");
        assert_eq!(h.active(), 0);
        assert_eq!(h.waiting(), 0, "idle engine has nothing queued");
        assert_eq!(h.max_concurrency(), 1000, "test handle uses the test cap");
    }

    /// `from_sender` is the mock/test-seam constructor (used by SpeechT5/mock
    /// tests) and always reports [`TTS_SAMPLE_RATE`] — it never touches
    /// `spawn_kokoro_engine`/`spawn_coqui_vits_engine`, which assign
    /// [`KOKORO_SAMPLE_RATE`]/[`COQUI_VITS_PL_SAMPLE_RATE`] at their own
    /// `TtsHandle` construction sites. Those two functions block on a real
    /// ONNX model load before returning a handle at all (`ready_rx.recv()`,
    /// e.g. `spawn_kokoro_engine` above), so their sample-rate wiring can't
    /// be exercised without live model fixtures — verified instead by a live
    /// smoke test (README "Setup" §5), same as every other real OV/ONNX
    /// backend in this codebase.
    #[test]
    fn from_sender_handle_reports_speecht5_sample_rate() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<TtsCommand>(TTS_CHANNEL_CAP);
        let h = TtsHandle::from_sender(tx, "test-tts");
        assert_eq!(h.sample_rate(), TTS_SAMPLE_RATE, "mock/SpeechT5 = 16 kHz");
    }

    /// The three backend sample-rate constants must be pairwise distinct —
    /// a duplicate would silently produce a wrong WAV header for whichever
    /// backend shares another's rate.
    #[test]
    fn backend_sample_rate_constants_are_distinct() {
        let rates = [
            TTS_SAMPLE_RATE,
            KOKORO_SAMPLE_RATE,
            COQUI_VITS_PL_SAMPLE_RATE,
        ];
        let unique: std::collections::BTreeSet<u32> = rates.iter().copied().collect();
        assert_eq!(unique.len(), rates.len(), "sample rates must all differ");
    }

    #[tokio::test]
    async fn mock_tts_round_trips() {
        let (handle, _thread) = spawn_mock_tts("mock-tts");
        let samples = handle
            .synthesize("Hello world.".to_owned(), None, 1.0)
            .await
            .expect("mock synthesize");
        assert_eq!(samples.len(), TTS_SAMPLE_RATE as usize);
        assert!(samples.iter().all(|s| *s == 0.0), "mock returns silence");
        assert_eq!(handle.active(), 0, "counter settled back to zero");
    }

    /// A dropped engine (closed channel) reports `EngineDead`, not a generic
    /// error — the caller should respond 503, not 429.
    #[tokio::test]
    async fn synthesize_returns_engine_dead_when_channel_closed() {
        let (tx, rx) = tokio::sync::mpsc::channel::<TtsCommand>(TTS_CHANNEL_CAP);
        let handle = TtsHandle::from_sender(tx, "dead-tts");
        drop(rx);
        let result = handle.synthesize("hi".to_owned(), None, 1.0).await;
        assert!(matches!(result, Err(AdmitError::EngineDead)));
        assert_eq!(handle.active(), 0, "no permit was ever held");
    }

    /// When a slot opens (permit released after the first reply), a waiting
    /// `synthesize` completes instead of returning `AtCapacity`.
    #[tokio::test]
    async fn synthesize_waits_and_succeeds_when_slot_opens() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<TtsCommand>(4);
        let handle = TtsHandle::from_sender_with_cap(tx, "test-tts", 1, 500);

        tokio::spawn(async move {
            let mut first = true;
            while let Some(TtsCommand::Synthesize { reply, permit, .. }) = rx.recv().await {
                if first {
                    first = false;
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                let _ = reply.send(Ok(vec![0.0_f32; 10]));
                drop(permit);
            }
        });

        let h2 = handle.clone();
        let first_call =
            tokio::spawn(async move { h2.synthesize("a".to_owned(), None, 1.0).await });
        tokio::time::sleep(Duration::from_millis(10)).await;

        let second = handle.synthesize("b".to_owned(), None, 1.0).await;
        assert!(
            second.is_ok(),
            "second call must wait for the slot and then succeed, got {second:?}"
        );
        assert!(first_call.await.unwrap().is_ok());
    }

    /// When no slot opens within `queue_timeout`, `synthesize` returns
    /// `AtCapacity` (→ HTTP 429) rather than waiting forever.
    #[tokio::test]
    async fn synthesize_returns_at_capacity_after_timeout() {
        let (tx, rx) = tokio::sync::mpsc::channel::<TtsCommand>(4);
        let handle = TtsHandle::from_sender_with_cap(tx, "test-tts", 1, 50);

        let h2 = handle.clone();
        let _first = tokio::spawn(async move { h2.synthesize("a".to_owned(), None, 1.0).await });
        tokio::time::sleep(Duration::from_millis(20)).await;

        let result = handle.synthesize("b".to_owned(), None, 1.0).await;
        assert!(
            matches!(result, Err(AdmitError::AtCapacity)),
            "must 429 after timeout, got {result:?}"
        );
        drop(rx);
    }
}
