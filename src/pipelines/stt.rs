// ============================================================
// src/pipelines/stt.rs — speech-to-text engine thread + handle (Phase 5.1c)
// ============================================================
// The convergence step for STT: wires the Whisper FFI (`crate::ov_whisper`, 5.1b)
// to the HTTP handler through a real, channel-backed engine handle — the STT
// analogue of `crate::embed_engine`. Request/response (one audio file in → one
// transcription out, no token stream), so the command carries a `oneshot` reply
// channel.
//
//   - One dedicated OS thread owns the `OvWhisperEngine` (`WhisperPipeline`).
//   - The thread DECODES the uploaded audio (`symphonia`) and RESAMPLES it to
//     Whisper's required 16 kHz mono f32 (`rubato`), then transcribes — all on
//     the engine thread so the !Sync pipeline is never aliased.
//   - Commands arrive via an mpsc channel (capacity `STT_CHANNEL_CAP`); one job
//     runs at a time (the pipeline is single-stream and blocking).
//   - Admission is a real `Semaphore` gate (mirrors `cb_engine`/`vlm_engine`,
//     dev/plans/vlm-admission-gate-fix.md's recipe): a permit is acquired
//     *before* a command is sent and held by the engine thread until it
//     replies, so `active()` is honest occupancy. Replaces the old
//     `in_flight: AtomicUsize` + `InFlightGuard` approximation.
// ============================================================

use std::io::Cursor;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::Context as _;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc::Sender, oneshot};

use crate::cb_engine::AdmitError;
use crate::metrics::{HotMetrics, Modality};
use crate::model_manager::{ManagedEngine, ModelKind};
use crate::ov_whisper::OvWhisperEngine;

/// Whisper's required input sample rate (mono f32). All decoded audio is
/// resampled to this before transcription.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// STT admission-gate size (also the reported `max_concurrency`): up to this
/// many requests may be queued/transcribing before the gate returns
/// [`AdmitError::AtCapacity`]. Smaller than the embedding cap — a
/// transcription holds the engine far longer than a single embedding batch.
pub(crate) const STT_CHANNEL_CAP: usize = 4;

// ── Result types ──────────────────────────────────────────────────────────────

/// One transcription segment with its time span (seconds).
#[derive(Debug, Clone)]
pub struct Segment {
    /// Segment start offset, seconds.
    pub start: f32,
    /// Segment end offset, seconds.
    pub end: f32,
    /// Segment transcript text.
    pub text: String,
}

/// A decoded transcription: full text, detected/requested language, and
/// per-segment timings (populated only when timestamps were requested).
#[derive(Debug, Clone)]
pub struct Transcription {
    /// The full transcript.
    pub text: String,
    /// Detected (or requested) language, as reported by the pipeline.
    pub language: String,
    /// Per-segment timings; empty unless timestamps were requested.
    pub segments: Vec<Segment>,
}

impl Transcription {
    /// Total audio duration (seconds) = end of the last segment, or `0.0` when
    /// there are no timestamped segments.
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.segments.last().map_or(0.0, |s| s.end)
    }
}

// ── Commands ────────────────────────────────────────────────────────────────

pub(crate) enum SttCommand {
    /// Decode + resample + transcribe one uploaded audio file.
    Transcribe {
        /// Raw uploaded audio bytes (any `symphonia`-supported container).
        audio_bytes: Vec<u8>,
        /// Optional source-language hint (`"en"`, `"pl"`, …); `None` = autodetect.
        language: Option<String>,
        /// When `true`, the result carries per-segment timings.
        timestamps: bool,
        /// One-shot reply channel — carries the transcription back to the handler.
        reply: oneshot::Sender<anyhow::Result<Transcription>>,
        /// Wall-clock start (handler entry) for the duration metric.
        started_at: Instant,
        /// Admission-gate permit, held until this transcription finishes.
        /// Dropped after the engine thread sends its reply, releasing the slot
        /// for the next waiting caller. See [`SttHandle::transcribe`].
        permit: OwnedSemaphorePermit,
    },
}

// ── Handle ──────────────────────────────────────────────────────────────────

/// Cloneable submit handle for the STT engine thread.
///
/// `transcribe` is request/response: it sends the audio and awaits the reply.
/// The engine thread processes one job at a time — requests serialise in the
/// channel.
#[derive(Clone, Debug)]
pub struct SttHandle {
    tx: Sender<SttCommand>,
    /// The model ID of the loaded Whisper model (the response `model` field).
    model_id: Arc<str>,
    /// Admission-gate permits — `cap` of them. Mirrors `cb_engine::EngineHandle`'s
    /// `sem` field exactly.
    sem: Arc<Semaphore>,
    /// Channel capacity = the semaphore's permit count = [`STT_CHANNEL_CAP`].
    cap: usize,
    /// Maximum time [`transcribe`](Self::transcribe) waits for a permit before
    /// returning [`AdmitError::AtCapacity`] (→ HTTP 429).
    queue_timeout: Duration,
    /// Callers currently parked in [`transcribe`](Self::transcribe) awaiting an
    /// admission permit — the engine's honest `rustedvino_requests_waiting`.
    waiting: Arc<AtomicUsize>,
}

impl SttHandle {
    /// The model ID of the loaded Whisper model.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Build a handle around an existing command channel — test seam only.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn from_sender(tx: Sender<SttCommand>, model_id: &str) -> Self {
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
        tx: Sender<SttCommand>,
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

    /// Transcribe one uploaded audio file.
    ///
    /// `audio_bytes` is the raw upload (wav/mp3/ogg/flac/m4a); decode + resample
    /// happen on the engine thread. `language` is a source-language hint or
    /// `None` to autodetect; `timestamps` requests per-segment timings.
    ///
    /// Waits up to `queue_timeout` for an admission permit before returning
    /// [`AdmitError::AtCapacity`] (→ HTTP 429). Returns [`AdmitError::EngineDead`]
    /// if the engine thread has exited (→ HTTP 503), or [`AdmitError::Failed`]
    /// if the audio could not be decoded or the pipeline call failed.
    ///
    /// # Errors
    /// See the variants above: [`AdmitError::AtCapacity`],
    /// [`AdmitError::EngineDead`], [`AdmitError::Failed`].
    pub async fn transcribe(
        &self,
        audio_bytes: Vec<u8>,
        language: Option<String>,
        timestamps: bool,
    ) -> Result<Transcription, AdmitError> {
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
            .send(SttCommand::Transcribe {
                audio_bytes,
                language,
                timestamps,
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
}

/// The STT engine is a [`ModelKind::Stt`] engine. Uniform with the other
/// `ManagedEngine` impls so `ModelManager` reads metrics/lifecycle identically.
impl ManagedEngine for SttHandle {
    fn kind(&self) -> ModelKind {
        ModelKind::Stt
    }

    /// Accepted-but-unfinished requests (queued + the one transcribing) —
    /// derived from the semaphore: real occupancy, not an approximation.
    fn active(&self) -> usize {
        self.cap.saturating_sub(self.sem.available_permits())
    }

    fn max_concurrency(&self) -> usize {
        self.cap
    }

    /// Callers currently parked in [`transcribe`](Self::transcribe) awaiting an
    /// admission permit. Mirrors `cb_engine::EngineHandle::waiting` exactly.
    fn waiting(&self) -> usize {
        self.waiting.load(Ordering::Relaxed)
    }
}

// ── Engine thread ─────────────────────────────────────────────────────────────

/// Spawn a dedicated OS thread owning `engine` and return an [`SttHandle`].
///
/// The thread exits cleanly when all handle clones are dropped (channel closes).
///
/// # Errors
/// Only propagates thread-spawn errors (rare OS resource exhaustion).
pub fn spawn_stt_engine(
    engine: OvWhisperEngine,
    model_id: &str,
    device: &str,
    queue_timeout: Duration,
) -> anyhow::Result<(SttHandle, std::thread::JoinHandle<()>)> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<SttCommand>(STT_CHANNEL_CAP);
    let sem = Arc::new(Semaphore::new(STT_CHANNEL_CAP));
    let model_id_arc: Arc<str> = Arc::from(model_id);
    let model_id_owned = model_id.to_owned();
    let device = device.to_owned();

    // STT joins the `kind="stt"` metric family. There is no token stream, so
    // only request-count and end-to-end duration are emitted. Audio input has no
    // dedicated `Modality` yet → `Modality::Text` (NOW: extend the enum later).
    let metrics = HotMetrics::new(ModelKind::Stt, model_id, &device);

    let thread = std::thread::Builder::new()
        .name(format!("stt-engine-{model_id_owned}"))
        .spawn(move || {
            tracing::info!(
                model_id = model_id_owned,
                kind = ModelKind::Stt.label(),
                device,
                "STT engine thread started"
            );
            while let Some(SttCommand::Transcribe {
                audio_bytes,
                language,
                timestamps,
                reply,
                started_at,
                permit,
            }) = rx.blocking_recv()
            {
                metrics.request_accepted(Modality::Text);
                let elapsed_result = run_transcribe(
                    &engine,
                    &model_id_owned,
                    &audio_bytes,
                    language.as_deref(),
                    timestamps,
                );
                let elapsed = started_at.elapsed();
                metrics.record_duration(Modality::Text, elapsed.as_secs_f64());
                if let Ok((_, n_samples)) = &elapsed_result {
                    #[allow(clippy::cast_precision_loss)]
                    // n_samples << 2^53; precision loss negligible
                    let audio_s = *n_samples as f64 / f64::from(WHISPER_SAMPLE_RATE);
                    metrics.record_rtf(elapsed.as_secs_f64() / audio_s.max(0.001));
                }
                let result = elapsed_result.map(|(t, _)| t);
                // The receiver may be gone if the client disconnected — ignore.
                let _ = reply.send(result);
                // Balance the handle's admission acquire, whatever the outcome —
                // releases the slot for the next waiting caller.
                drop(permit);
            }
            tracing::info!(
                model_id = model_id_owned,
                kind = ModelKind::Stt.label(),
                "STT engine thread exiting — all handles dropped"
            );
        })?;

    Ok((
        SttHandle {
            tx,
            model_id: model_id_arc,
            sem,
            cap: STT_CHANNEL_CAP,
            queue_timeout,
            waiting: Arc::new(AtomicUsize::new(0)),
        },
        thread,
    ))
}

/// Run one transcription on the engine thread: decode + resample the upload to
/// 16 kHz mono, then transcribe and map the [`crate::ov_whisper::WhisperResult`]
/// to a [`Transcription`].
fn run_transcribe(
    engine: &OvWhisperEngine,
    model_id: &str,
    audio_bytes: &[u8],
    language: Option<&str>,
    timestamps: bool,
) -> anyhow::Result<(Transcription, usize)> {
    let samples = prepare_samples(audio_bytes)?;
    let n = samples.len();
    let result = engine.transcribe(&samples, language, timestamps)?;
    tracing::debug!(
        model_id,
        samples = n,
        chars = result.text.len(),
        chunks = result.chunks.len(),
        "transcription complete"
    );
    Ok((
        Transcription {
            text: result.text,
            language: result.language,
            segments: result
                .chunks
                .into_iter()
                .map(|c| Segment {
                    start: c.start,
                    end: c.end,
                    text: c.text,
                })
                .collect(),
        },
        n,
    ))
}

// ── Audio decode + resample (pure Rust, no GPU) ───────────────────────────────

/// Decode an uploaded audio file and resample it to Whisper's 16 kHz mono f32.
///
/// # Errors
/// Returns an error if the container/codec is unsupported, the bytes are
/// malformed, or no audio track is present.
pub fn prepare_samples(audio_bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    let (samples, sample_rate) = decode_to_mono(audio_bytes)?;
    resample_to_16k(&samples, sample_rate)
}

/// Decode any `symphonia`-supported container to mono f32 PCM, returning the
/// samples and their source sample rate. Multi-channel audio is downmixed by
/// averaging channels.
fn decode_to_mono(audio_bytes: &[u8]) -> anyhow::Result<(Vec<f32>, u32)> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
    use symphonia::core::errors::Error as SymErr;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    // symphonia needs an owned, seekable source; copy the upload into a cursor.
    let owned = audio_bytes.to_vec();
    let mss = MediaSourceStream::new(
        Box::new(Cursor::new(owned)),
        MediaSourceStreamOptions::default(),
    );

    let probed = symphonia::default::get_probe()
        .format(
            &Hint::new(),
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("unsupported or malformed audio container")?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .context("audio file has no decodable track")?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .context("audio track has no sample rate")?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("no decoder for this audio codec")?;

    // Any error ends decoding (EOF is reported as an IoError) — we keep whatever
    // we decoded so far and stop.
    let mut mono: Vec<f32> = Vec::new();
    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }
        let pcm_ref = match decoder.decode(&packet) {
            Ok(d) => d,
            // A single bad packet is non-fatal; skip it.
            Err(SymErr::DecodeError(_)) => continue,
            Err(e) => return Err(anyhow::anyhow!("audio decode failed: {e}")),
        };
        let spec = *pcm_ref.spec();
        let channels = spec.channels.count().max(1);
        let mut buf = SampleBuffer::<f32>::new(pcm_ref.capacity() as u64, spec);
        buf.copy_interleaved_ref(pcm_ref);
        for frame in buf.samples().chunks(channels) {
            let sum: f32 = frame.iter().sum();
            #[allow(clippy::cast_precision_loss)]
            mono.push(sum / channels as f32);
        }
    }

    if mono.is_empty() {
        anyhow::bail!("audio file decoded to zero samples");
    }
    Ok((mono, sample_rate))
}

/// Resample mono f32 PCM from `in_rate` to [`WHISPER_SAMPLE_RATE`] using rubato's
/// high-quality sinc interpolator. A no-op when `in_rate` already matches.
///
/// # Errors
/// Returns an error if the resampler rejects the ratio or a processing step fails.
pub(crate) fn resample_to_16k(samples: &[f32], in_rate: u32) -> anyhow::Result<Vec<f32>> {
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    const CHUNK: usize = 1024;

    if in_rate == WHISPER_SAMPLE_RATE {
        return Ok(samples.to_vec());
    }

    let ratio = f64::from(WHISPER_SAMPLE_RATE) / f64::from(in_rate);
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, CHUNK, 1)
        .context("failed to construct resampler")?;

    // Capacity is a hint only; the lossy ratio scaling is harmless here.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let cap = (samples.len() as f64 * ratio) as usize + CHUNK;
    let mut out: Vec<f32> = Vec::with_capacity(cap);
    let mut pos = 0;
    // Feed full fixed-size chunks; SincFixedIn::process requires exactly
    // `input_frames_next()` (== CHUNK) frames per call.
    while pos + CHUNK <= samples.len() {
        let chunk_in = [samples[pos..pos + CHUNK].to_vec()];
        let block = resampler
            .process(&chunk_in, None)
            .context("resampler chunk failed")?;
        out.extend_from_slice(&block[0]);
        pos += CHUNK;
    }
    // Flush the final partial chunk (process_partial pads internally).
    if pos < samples.len() {
        let tail = [samples[pos..].to_vec()];
        let block = resampler
            .process_partial(Some(&tail), None)
            .context("resampler tail failed")?;
        out.extend_from_slice(&block[0]);
    }

    Ok(out)
}

// ── Mock engine (tests only) ──────────────────────────────────────────────────

/// Spawn a GPU-free mock STT engine — the `Stt` analogue of `spawn_mock_embed`,
/// used by `MockEngineFactory` and routing tests.
///
/// Answers each request with a fixed transcription (canned text + two segments)
/// without decoding the audio or touching a GPU, so a handler test yields a
/// valid response shape.
#[cfg(test)]
pub(crate) fn spawn_mock_stt(model_id: &str) -> (SttHandle, std::thread::JoinHandle<()>) {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<SttCommand>(STT_CHANNEL_CAP);
    let sem = Arc::new(Semaphore::new(STT_CHANNEL_CAP));

    let thread = std::thread::spawn(move || {
        while let Some(SttCommand::Transcribe {
            language,
            reply,
            permit,
            ..
        }) = rx.blocking_recv()
        {
            let _ = reply.send(Ok(Transcription {
                text: "This is a mock transcription.".to_owned(),
                language: language.unwrap_or_else(|| "english".to_owned()),
                segments: vec![
                    Segment {
                        start: 0.0,
                        end: 1.5,
                        text: "This is a mock".to_owned(),
                    },
                    Segment {
                        start: 1.5,
                        end: 2.8,
                        text: "transcription.".to_owned(),
                    },
                ],
            }));
            drop(permit);
        }
    });

    (
        SttHandle {
            tx,
            model_id: Arc::from(model_id),
            sem,
            cap: STT_CHANNEL_CAP,
            queue_timeout: Duration::from_secs(30),
            waiting: Arc::new(AtomicUsize::new(0)),
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
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]

    use super::*;

    /// Build a minimal 16-bit PCM WAV in memory: `sample_rate` Hz, mono, holding
    /// `samples` (clamped to i16). Enough for symphonia's WAV reader to decode.
    fn wav_mono_i16(sample_rate: u32, samples: &[f32]) -> Vec<u8> {
        let data_len = (samples.len() * 2) as u32;
        let mut v = Vec::with_capacity(44 + data_len as usize);
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&(36 + data_len).to_le_bytes());
        v.extend_from_slice(b"WAVE");
        v.extend_from_slice(b"fmt ");
        v.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
        v.extend_from_slice(&1u16.to_le_bytes()); // PCM
        v.extend_from_slice(&1u16.to_le_bytes()); // mono
        v.extend_from_slice(&sample_rate.to_le_bytes());
        v.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
        v.extend_from_slice(&2u16.to_le_bytes()); // block align
        v.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        v.extend_from_slice(b"data");
        v.extend_from_slice(&data_len.to_le_bytes());
        for s in samples {
            let i = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
            v.extend_from_slice(&i.to_le_bytes());
        }
        v
    }

    /// A fresh handle reports zero active/waiting requests and its configured cap.
    #[test]
    fn idle_handle_reports_zero_active_and_waiting() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<SttCommand>(STT_CHANNEL_CAP);
        let h = SttHandle::from_sender(tx, "test-stt");
        assert_eq!(h.active(), 0);
        assert_eq!(h.waiting(), 0, "idle engine has nothing queued");
        assert_eq!(h.max_concurrency(), 1000, "test handle uses the test cap");
    }

    /// The mock engine round-trips a request: canned text, echoed language, and
    /// the admission gate settles back to zero occupancy.
    #[tokio::test]
    async fn mock_stt_round_trips() {
        let (handle, _thread) = spawn_mock_stt("mock-stt");
        let out = handle
            .transcribe(b"fake-audio".to_vec(), Some("pl".to_owned()), true)
            .await
            .expect("mock transcribe");
        assert_eq!(out.text, "This is a mock transcription.");
        assert_eq!(out.language, "pl");
        assert_eq!(out.segments.len(), 2);
        assert!((out.duration() - 2.8).abs() < f32::EPSILON);
        assert_eq!(handle.active(), 0, "counter settled back to zero");
    }

    /// A dropped engine (closed channel) reports `EngineDead`, not a generic
    /// error — the caller should respond 503, not 429.
    #[tokio::test]
    async fn transcribe_returns_engine_dead_when_channel_closed() {
        let (tx, rx) = tokio::sync::mpsc::channel::<SttCommand>(STT_CHANNEL_CAP);
        let handle = SttHandle::from_sender(tx, "dead-stt");
        drop(rx);
        let result = handle.transcribe(b"x".to_vec(), None, false).await;
        assert!(matches!(result, Err(AdmitError::EngineDead)));
        assert_eq!(handle.active(), 0, "no permit was ever held");
    }

    /// When a slot opens (permit released after the first reply), a waiting
    /// `transcribe` completes instead of returning `AtCapacity`.
    #[tokio::test]
    async fn transcribe_waits_and_succeeds_when_slot_opens() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<SttCommand>(4);
        let handle = SttHandle::from_sender_with_cap(tx, "test-stt", 1, 500);

        tokio::spawn(async move {
            let mut first = true;
            while let Some(SttCommand::Transcribe { reply, permit, .. }) = rx.recv().await {
                if first {
                    first = false;
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                let _ = reply.send(Ok(Transcription {
                    text: "ok".to_owned(),
                    language: "en".to_owned(),
                    segments: vec![],
                }));
                drop(permit);
            }
        });

        let h2 = handle.clone();
        let first_call =
            tokio::spawn(async move { h2.transcribe(b"a".to_vec(), None, false).await });
        tokio::time::sleep(Duration::from_millis(10)).await;

        let second = handle.transcribe(b"b".to_vec(), None, false).await;
        assert!(
            second.is_ok(),
            "second call must wait for the slot and then succeed, got {second:?}"
        );
        assert!(first_call.await.unwrap().is_ok());
    }

    /// When no slot opens within `queue_timeout`, `transcribe` returns
    /// `AtCapacity` (→ HTTP 429) rather than waiting forever.
    #[tokio::test]
    async fn transcribe_returns_at_capacity_after_timeout() {
        let (tx, rx) = tokio::sync::mpsc::channel::<SttCommand>(4);
        let handle = SttHandle::from_sender_with_cap(tx, "test-stt", 1, 50);

        let h2 = handle.clone();
        let _first = tokio::spawn(async move { h2.transcribe(b"a".to_vec(), None, false).await });
        tokio::time::sleep(Duration::from_millis(20)).await;

        let result = handle.transcribe(b"b".to_vec(), None, false).await;
        assert!(
            matches!(result, Err(AdmitError::AtCapacity)),
            "must 429 after timeout, got {result:?}"
        );
        drop(rx);
    }

    /// Decoding a mono WAV recovers the right sample count at the native rate.
    #[test]
    fn decode_mono_wav_recovers_samples() {
        let samples: Vec<f32> = (0..1600).map(|i| ((i as f32) * 0.01).sin() * 0.5).collect();
        let wav = wav_mono_i16(16_000, &samples);
        let (decoded, rate) = decode_to_mono(&wav).expect("decode wav");
        assert_eq!(rate, 16_000);
        assert_eq!(decoded.len(), 1600, "one mono sample per input frame");
    }

    /// A 16 kHz input passes through resampling unchanged (the no-op fast path).
    #[test]
    fn resample_passthrough_at_16k() {
        let samples = vec![0.1_f32; 500];
        let out = resample_to_16k(&samples, 16_000).expect("resample");
        assert_eq!(out, samples);
    }

    /// Resampling 8 kHz → 16 kHz roughly doubles the sample count and stays finite.
    #[test]
    fn resample_8k_to_16k_doubles_length() {
        let samples: Vec<f32> = (0..4000).map(|i| ((i as f32) * 0.02).sin() * 0.3).collect();
        let out = resample_to_16k(&samples, 8_000).expect("resample");
        // ~2× the input (sinc transients shift the exact count slightly).
        let ratio = out.len() as f32 / samples.len() as f32;
        assert!(
            (1.8..2.2).contains(&ratio),
            "8k→16k should ~double length, got ratio {ratio}"
        );
        assert!(out.iter().all(|s| s.is_finite()), "all samples finite");
    }

    /// End-to-end pure-Rust path: WAV at 8 kHz → decode → resample to 16 kHz.
    #[test]
    fn prepare_samples_decodes_and_resamples() {
        let samples: Vec<f32> = (0..2000).map(|i| ((i as f32) * 0.03).sin() * 0.4).collect();
        let wav = wav_mono_i16(8_000, &samples);
        let prepared = prepare_samples(&wav).expect("prepare");
        let ratio = prepared.len() as f32 / samples.len() as f32;
        assert!((1.8..2.2).contains(&ratio), "8k→16k ~doubles, got {ratio}");
    }

    /// Garbage bytes are rejected with an error, not a panic.
    #[test]
    fn prepare_samples_rejects_garbage() {
        let err = prepare_samples(b"not audio at all");
        assert!(err.is_err(), "non-audio bytes must error");
    }
}
