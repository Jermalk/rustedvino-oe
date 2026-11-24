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
//     the project's internal engineering log's recipe): a permit is acquired
//     *before* a command is sent and held by the engine thread until it
//     replies, so `active()` is honest occupancy. Replaces the old
//     `in_flight: AtomicUsize` + `InFlightGuard` approximation.
// ============================================================

use std::io::Cursor;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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

// ── Language hint ───────────────────────────────────────────────────────────

/// The language codes Whisper's tokenizer has a `<|xx|>` token for (`openai/whisper`
/// `tokenizer.py` `LANGUAGES`, the same set every multilingual Whisper checkpoint, fine-tunes
/// included, carries). The bridge builds `"<|" + language + "|>"`; a code outside this set has no
/// such token and used to fail inside the pipeline as a 500 (FM-3).
pub const WHISPER_LANGUAGES: [&str; 100] = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
];

/// Normalises a client's STT `language` hint before it reaches the bridge (FM-3).
///
/// Trimmed and lower-cased; a region or script suffix after `-` or `_` is dropped (`pl-PL`,
/// `PL`, `zh_TW` become `pl`, `pl`, `zh`). Empty or `auto` means "detect": `Ok(None)`, the same as
/// no field. The result must be one of [`WHISPER_LANGUAGES`].
///
/// # Errors
/// Any other value: a client-facing message naming the accepted form (a 400 on the HTTP routes).
pub fn normalize_language(raw: &str) -> Result<Option<String>, String> {
    let lowered = raw.trim().to_ascii_lowercase();
    if lowered.is_empty() || lowered == "auto" {
        return Ok(None);
    }
    let base = lowered.split(['-', '_']).next().unwrap_or_default();
    if WHISPER_LANGUAGES.contains(&base) {
        return Ok(Some(base.to_owned()));
    }
    let shown: String = raw.trim().chars().take(32).collect();
    Err(format!(
        "unsupported language '{shown}' — use a language code Whisper knows (ISO 639-1, such as \
         \"en\" or \"pl\"; a region suffix such as \"pl-PL\" is accepted), or \"auto\" or no \
         language field to detect the language"
    ))
}

// ── Commands ────────────────────────────────────────────────────────────────
// ── Commands ────────────────────────────────────────────────────────────────

/// What the Whisper model does with the audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SttTask {
    /// Text in the spoken language (`/v1/audio/transcriptions`).
    Transcribe,
    /// English text whatever the spoken language (`/v1/audio/translations`).
    Translate,
}

pub(crate) enum SttCommand {
    /// Decode + resample + transcribe one uploaded audio file.
    Transcribe {
        /// Raw uploaded audio bytes (any `symphonia`-supported container).
        audio_bytes: Vec<u8>,
        /// Optional source-language hint (`"en"`, `"pl"`, …); `None` = autodetect.
        language: Option<String>,
        /// When `true`, the result carries per-segment timings.
        timestamps: bool,
        /// Transcribe or translate to English.
        task: SttTask,
        /// One-shot reply channel — carries the transcription back to the handler.
        reply: oneshot::Sender<anyhow::Result<Transcription>>,
        /// Wall-clock start (handler entry) for the duration metric.
        started_at: Instant,
        /// Admission-gate permit, held until this transcription finishes.
        /// Dropped by the engine thread just before it sends its reply, releasing
        /// the slot for the next waiting caller, so a woken caller never counts
        /// itself in `active()`. See [`SttHandle::transcribe`].
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
    /// Whether the model supports the translate task (`None` = unknown, see
    /// [`crate::ov_whisper::resolve_can_translate`]).
    can_translate: Option<bool>,
    /// Set (`SeqCst`) by `begin_evict` before it reads `active()`; checked after the admission
    /// permit is held. Shared by every clone of this handle; a reload builds a new engine with a
    /// clear flag. See `cb_engine::EngineHandle::closing`. D3-S1.D3a; nothing sets it yet.
    closing: Arc<AtomicBool>,
}

impl SttHandle {
    /// Occupy one admission slot, returning the held permit, to simulate an in-flight request in
    /// tests (as `cb_engine::EngineHandle::occupy_slot_for_test`): while it is alive,
    /// `active()` reports one more running request.
    #[cfg(test)]
    pub(crate) fn occupy_slot_for_test(&self) -> Option<tokio::sync::OwnedSemaphorePermit> {
        Arc::clone(&self.sem).try_acquire_owned().ok()
    }

    /// `false` only when the model is known not to translate (English-only);
    /// `true` when it can, or when that's unknown.
    #[must_use]
    pub fn can_translate(&self) -> bool {
        self.can_translate != Some(false)
    }

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
            closing: Arc::new(AtomicBool::new(false)),
            can_translate: Some(true),
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
            closing: Arc::new(AtomicBool::new(false)),
            can_translate: Some(true),
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
        self.submit(audio_bytes, language, timestamps, SttTask::Transcribe)
            .await
    }

    /// Translate one uploaded audio file into English — same contract as
    /// [`transcribe`](Self::transcribe); `language` is the *spoken* language
    /// hint. Check [`can_translate`](Self::can_translate) first.
    ///
    /// # Errors
    /// As [`transcribe`](Self::transcribe).
    pub async fn translate(
        &self,
        audio_bytes: Vec<u8>,
        language: Option<String>,
        timestamps: bool,
    ) -> Result<Transcription, AdmitError> {
        self.submit(audio_bytes, language, timestamps, SttTask::Translate)
            .await
    }

    async fn submit(
        &self,
        audio_bytes: Vec<u8>,
        language: Option<String>,
        timestamps: bool,
        task: SttTask,
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

        // The eviction may have started while this caller waited for (or just took) its permit.
        // Checked AFTER holding it: `begin_evict` sets `closing` before reading `active()`, so
        // either it counted this permit (and drains it) or this check sees the flag.
        if self.closing.load(Ordering::SeqCst) {
            drop(permit);
            return Err(AdmitError::Closing);
        }

        let started_at = Instant::now();
        let (reply_tx, reply_rx) = oneshot::channel();

        let sent = self
            .tx
            .send(SttCommand::Transcribe {
                audio_bytes,
                language,
                timestamps,
                task,
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

    fn set_closing(&self, closing: bool) {
        self.closing.store(closing, Ordering::SeqCst);
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
    can_translate: Option<bool>,
    max_audio_duration_secs: u64,
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
                task,
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
                    task,
                    max_audio_duration_secs,
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
                // Balance the handle's admission acquire, whatever the outcome —
                // releases the slot for the next waiting caller — BEFORE the reply
                // (0.9.1 C2, see `in_flight::finish`).
                crate::in_flight::finish(permit, reply, result);
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
            closing: Arc::new(AtomicBool::new(false)),
            can_translate,
        },
        thread,
    ))
}

/// How long one transcription of `n_samples` (16 kHz) may run before the watchdog counts it:
/// twice the audio length, at least 60 s.
#[must_use]
pub(crate) fn transcribe_allowance(n_samples: usize) -> Duration {
    let audio_secs = (n_samples as u64).div_ceil(u64::from(WHISPER_SAMPLE_RATE));
    Duration::from_secs((2 * audio_secs).max(60))
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
    task: SttTask,
    max_audio_duration_secs: u64,
) -> anyhow::Result<(Transcription, usize)> {
    let samples = prepare_samples(audio_bytes, max_audio_duration_secs)?;
    let n = samples.len();
    // Step 6c: visible to the watchdog (past its allowance) and to the exit decision while the
    // OpenVINO call runs, also after the caller gave up or the record left Ready.
    let ffi = crate::ffi_watch::global().arm(
        crate::ffi_watch::FfiKind::Transcribe,
        model_id,
        None,
        transcribe_allowance(n),
    );
    let result = engine.transcribe(&samples, language, timestamps, task == SttTask::Translate);
    drop(ffi);
    let result = result?;
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

/// The uploaded audio could not be decoded (garbage, unsupported container,
/// no track, zero samples). The text is one of this module's own fixed
/// phrases — never any of the caller's bytes — so it is safe to show.
/// Carried inside the `anyhow::Error` so the handler can `downcast_ref` it.
#[derive(Debug)]
pub struct BadAudio(pub String);

impl std::fmt::Display for BadAudio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BadAudio {}

/// Decode an uploaded audio file and resample it to Whisper's 16 kHz mono f32.
///
/// `max_secs` is [`crate::model_manager::Config::max_audio_duration_secs`]: longer audio is
/// refused (0.9.1 S4) — up front when the container declares its length, otherwise as soon
/// as decoding passes it, so memory stays bounded whatever the compressed size.
///
/// # Errors
/// Returns an error if the container/codec is unsupported, the bytes are
/// malformed, no audio track is present, or the audio is longer than `max_secs`.
pub fn prepare_samples(audio_bytes: &[u8], max_secs: u64) -> anyhow::Result<Vec<f32>> {
    // Every decode failure is a property of the upload, not of the server:
    // typed so the HTTP layer answers 400, not 500 (Windows pass 2, D9).
    //
    // A malformed header can also make the demuxer PANIC (a WAV declaring sample rate 0 panics in
    // symphonia's `TimeBase::new`). On the engine thread that kills the thread and with it every later
    // transcription (503 "engine unavailable" until the model is reloaded) - found by the overnight
    // torture run. Treat a panic in the decoder exactly like any other undecodable upload.
    let decoded = std::panic::catch_unwind(|| decode_to_mono(audio_bytes, max_secs));
    let (samples, sample_rate) = match decoded {
        Ok(result) => result.map_err(|e| anyhow::Error::new(BadAudio(e.to_string())))?,
        Err(_) => {
            return Err(anyhow::Error::new(BadAudio(
                "audio file has a malformed header".to_owned(),
            )));
        }
    };
    resample_to_16k(&samples, sample_rate)
}

/// ISO BMFF (MP4/M4A/MOV) sample tables whose entry count the demuxer pre-allocates for, with
/// the bytes per entry. symphonia's isomp4 reader calls `Vec::with_capacity(count)` on these
/// during `probe()` — before any check of ours can run — so a 3 KB file declaring
/// 0xFFFFFFFF `stsc` entries asks for 48 GiB and the allocator aborts the process (0.9.1 S4b,
/// Fable review; reproduced: "memory allocation of 68719476720 bytes failed", SIGABRT).
/// `stsz` counts only when its `sample_size` is 0 (a per-sample table).
const BMFF_TABLES: [(&[u8; 4], u64); 5] = [
    (b"stts", 8),
    (b"stsc", 12),
    (b"stco", 4),
    (b"co64", 8),
    (b"stsz", 4),
];

/// Plain container boxes the walk descends into: the path to the sample tables, metadata
/// (`udta`, `ilst` and its items), the `QuickTime` `wave` decoder-parameter box.
const BMFF_CONTAINERS: [&[u8; 4]; 13] = [
    b"moov", b"trak", b"mdia", b"minf", b"stbl", b"mvex", b"moof", b"traf", b"edts", b"dinf",
    b"udta", b"wave", b"ilst",
];

/// Audio sample entries whose child boxes (`esds`, `alac`, `dfLa`, `dOps`, `wave`, ...) the
/// demuxer reads after a version-dependent header.
const BMFF_AUDIO_ENTRIES: [&[u8; 4]; 14] = [
    b"mp4a", b"alac", b"fLaC", b"Opus", b"lpcm", b"sowt", b"twos", b"in24", b"in32", b"fl32",
    b"fl64", b"raw ", b"ulaw", b"alaw",
];

/// Refuse an MP4 the demuxer would allocate more for than RAM before reading it (0.9.1 S4b/S4c,
/// Fable reviews; both reproduced as process aborts):
/// - a sample table ([`BMFF_TABLES`]) declaring more entries than its box holds (a 3 KB M4A
///   with 0xFFFFFFFF `stsc` entries asked for 64 GiB);
/// - a box declaring a **64-bit** size larger than its parent (an 84-byte M4A with a 2^50-byte
///   `moov/udta/meta/ilst/©ART/data` box: symphonia never clamps a size to the stream and
///   reads some payloads with `read_boxed_slice_exact`, i.e. `vec![0; size]` up front).
///
/// Every other length a reachable symphonia reader allocates from is 32 bits or less: at most
/// 4 GiB, lazily zeroed and filled from at most the 50 MiB body, so it cannot abort. The walk
/// visits every box the demuxer reads on the way: plain containers ([`BMFF_CONTAINERS`]),
/// `meta` (a full box), `stsd` and the audio sample entries in it. symphonia's probe finds a
/// file by an `ftyp` marker anywhere in its search window, so every `ftyp` occurrence is
/// tried as a root. A walk that meets a 32-bit size running past its parent just stops
/// (symphonia fails there too, and a chance `ftyp` in audio data must not reject a good file);
/// only the two impossible shapes above are refused.
fn check_bmff_tables(bytes: &[u8]) -> anyhow::Result<()> {
    let mut from = 0;
    while let Some(i) = bytes[from..].windows(4).position(|w| w == b"ftyp") {
        let at = from + i;
        if at >= 4 {
            walk_bmff(bytes, at - 4, bytes.len(), 0, false)?;
        }
        from = at + 4;
    }
    Ok(())
}

fn be_u32(b: &[u8], at: usize) -> Option<u64> {
    b.get(at..at + 4)
        .map(|s| u64::from(u32::from_be_bytes([s[0], s[1], s[2], s[3]])))
}

fn be_u16(b: &[u8], at: usize) -> Option<u16> {
    b.get(at..at + 2).map(|s| u16::from_be_bytes([s[0], s[1]]))
}

/// `items`: every box at this level is a container (the children of `ilst`: `©ART`, `trkn`,
/// ..., each holding `data`/`name`/`mean`).
fn walk_bmff(
    bytes: &[u8],
    mut off: usize,
    end: usize,
    depth: u8,
    items: bool,
) -> anyhow::Result<()> {
    // Real files nest a few levels (moov/trak/mdia/minf/stbl/stsd/mp4a/esds); stop well past it.
    if depth > 16 {
        return Ok(());
    }
    while off + 8 <= end {
        let Some(size32) = be_u32(bytes, off) else {
            return Ok(());
        };
        let typ = &bytes[off + 4..off + 8];
        let room = (end - off) as u64;
        let (size, header) = match size32 {
            0 => (room, 8_usize),
            1 => {
                let (Some(hi), Some(lo)) = (be_u32(bytes, off + 8), be_u32(bytes, off + 12)) else {
                    return Ok(());
                };
                let size = (hi << 32) | lo;
                if size > room {
                    anyhow::bail!(
                        "malformed MP4: a '{}' box declares {size} bytes, more than the {room} its parent holds",
                        String::from_utf8_lossy(typ)
                    );
                }
                (size, 16)
            }
            n => (n, 8),
        };
        let Ok(size_us) = usize::try_from(size) else {
            return Ok(());
        };
        if size_us < header || size > room {
            return Ok(()); // malformed 32-bit size: the demuxer stops here too
        }
        let body = off + header;
        let box_end = off + size_us;
        let payload = (size_us - header) as u64;
        if let Some((name, entry)) = BMFF_TABLES.iter().find(|(t, _)| typ == &t[..]) {
            // Full box: version/flags (4), then for stsz sample_size (4) + count, else count.
            let (count_at, fixed) = if *name == b"stsz" {
                (body + 8, 12)
            } else {
                (body + 4, 8)
            };
            let per_sample_table = *name != b"stsz" || be_u32(bytes, body + 4) == Some(0);
            if let Some(count) = be_u32(bytes, count_at)
                && per_sample_table
                && count.saturating_mul(*entry) > payload.saturating_sub(fixed)
            {
                anyhow::bail!(
                    "malformed MP4: the '{}' table declares {count} entries, more than its box holds",
                    String::from_utf8_lossy(&name[..])
                );
            }
        } else if items || BMFF_CONTAINERS.iter().any(|c| typ == &c[..]) {
            walk_bmff(bytes, body, box_end, depth + 1, typ == b"ilst")?;
        } else if typ == b"meta" {
            walk_bmff(bytes, body + 4, box_end, depth + 1, false)?; // full box: version/flags first
        } else if typ == b"stsd" {
            walk_bmff(bytes, body + 8, box_end, depth + 1, false)?; // version/flags + entry count
        } else if BMFF_AUDIO_ENTRIES.iter().any(|c| typ == &c[..]) {
            // 6 reserved + 2 data-ref index, then the version: v0 header 28 bytes, v1 44, v2 64.
            let skip = match be_u16(bytes, body + 8) {
                Some(0) => 28,
                Some(1) => 44,
                Some(2) => 64,
                _ => 0,
            };
            if skip > 0 && body + skip <= box_end {
                walk_bmff(bytes, body + skip, box_end, depth + 1, false)?;
            }
        }
        off = box_end;
    }
    Ok(())
}

/// Highest source sample rate accepted. The duration cap is counted in samples at the
/// source rate, so the worst-case decode memory is `max_audio_duration_secs` x this x 4 bytes
/// (0.9.1 S4b): 48 kHz keeps 2 h to ~1.4 GB and covers every common speech rate (8-48 kHz).
/// Whisper hears 16 kHz, so a higher rate gains nothing; such a file is a 400.
const MAX_SOURCE_SAMPLE_RATE: u32 = 48_000;

/// Most frames per packet a track may declare before its decoder is built (0.9.1 S4c).
const MAX_FRAMES_PER_PACKET: u64 = 1 << 20;

/// Lowest source sample rate accepted (0.9.1 S4c). 4 kHz is below any real speech recording
/// (telephony is 8 kHz).
const MIN_SOURCE_SAMPLE_RATE: u32 = 4_000;

/// The refusal for audio longer than the server's limit — the caller's error (a 400).
fn too_long(max_secs: u64) -> anyhow::Error {
    anyhow::anyhow!(
        "audio is longer than this server's limit of {max_secs} s (max_audio_duration_secs)"
    )
}

/// Decode any `symphonia`-supported container to mono f32 PCM, returning the
/// samples and their source sample rate. Multi-channel audio is downmixed by
/// averaging channels. Stops with an error past `max_secs` of audio.
fn decode_to_mono(audio_bytes: &[u8], max_secs: u64) -> anyhow::Result<(Vec<f32>, u32)> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
    use symphonia::core::errors::Error as SymErr;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    // Before the probe: the MP4 demuxer pre-allocates its sample tables while probing.
    check_bmff_tables(audio_bytes)?;
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
    // A floor too: resampling 1 Hz to 16 kHz is a 16 000x expansion inside rubato (Fable nit).
    anyhow::ensure!(
        sample_rate >= MIN_SOURCE_SAMPLE_RATE,
        "audio sample rate {sample_rate} Hz is below the supported {MIN_SOURCE_SAMPLE_RATE} Hz"
    );
    anyhow::ensure!(
        sample_rate <= MAX_SOURCE_SAMPLE_RATE,
        "audio sample rate {sample_rate} Hz is above the supported {MAX_SOURCE_SAMPLE_RATE} Hz"
    );
    // Mono samples at the source rate the cap allows. usize: ≤ 384 000 × max_secs, and a
    // config large enough to overflow would saturate to "no practical cap", never wrap.
    let max_samples =
        usize::try_from(max_secs.saturating_mul(u64::from(sample_rate))).unwrap_or(usize::MAX);
    // Refuse before decoding anything when the container declares a longer track.
    let declared = track
        .codec_params
        .n_frames
        .map(|n| usize::try_from(n).unwrap_or(usize::MAX));
    if declared.is_some_and(|n| n > max_samples) {
        return Err(too_long(max_secs));
    }

    // A decoder sizes its packet buffer from `max_frames_per_packet` when it is built: the
    // PCM decoder takes an MP4 LPCM sample entry's u32 `frames_per_packet` as is, so a file
    // could ask for frames x channels x 8 bytes (~1 TiB) and abort the process (0.9.1 S4c).
    // Real codecs use at most 65 535 (FLAC); 2^20 frames is ~22 s at 48 kHz in one packet.
    if let Some(frames) = track.codec_params.max_frames_per_packet {
        anyhow::ensure!(
            frames <= MAX_FRAMES_PER_PACKET,
            "audio declares {frames} frames per packet, above the supported {MAX_FRAMES_PER_PACKET}"
        );
    }
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("no decoder for this audio codec")?;

    // Any error ends decoding (EOF is reported as an IoError) — we keep whatever
    // we decoded so far and stop.
    // Reserved once from the declared length (already ≤ the cap) so growth never doubles the
    // peak; an undeclared or wrong length still stops at the cap in the loop below.
    let mut mono: Vec<f32> = Vec::with_capacity(declared.unwrap_or(0));
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
        // Checked before the packet's frames are kept: a declared length can be absent or
        // wrong, so this bound is the one that holds memory to `max_samples`.
        if mono.len().saturating_add(pcm_ref.frames()) > max_samples {
            return Err(too_long(max_secs));
        }
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
            task,
            reply,
            permit,
            ..
        }) = rx.blocking_recv()
        {
            // Released, then replied, as the real engine thread does.
            crate::in_flight::finish(
                permit,
                reply,
                Ok(Transcription {
                    text: match task {
                        SttTask::Transcribe => "This is a mock transcription.",
                        SttTask::Translate => "This is a mock translation.",
                    }
                    .to_owned(),
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
                }),
            );
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
            closing: Arc::new(AtomicBool::new(false)),
            // Mirrors Whisper's naming: an `.en` model is English-only.
            can_translate: Some(!model_id.contains(".en")),
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

    /// The task reaches the engine thread (the mock answers per task), and an
    /// English-only model reports it can't translate.
    #[tokio::test]
    async fn translate_task_reaches_engine_and_en_models_cannot_translate() {
        let (handle, _thread) = spawn_mock_stt("mock-stt");
        assert!(handle.can_translate());
        let t = handle
            .translate(b"audio".to_vec(), Some("pl".to_owned()), false)
            .await
            .unwrap();
        assert_eq!(t.text, "This is a mock translation.");
        let t = handle
            .transcribe(b"audio".to_vec(), None, false)
            .await
            .unwrap();
        assert_eq!(t.text, "This is a mock transcription.");
        let (en_only, _thread) = spawn_mock_stt("whisper-tiny.en");
        assert!(!en_only.can_translate());
    }

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

    /// When a slot opens (the first request's permit is released), a waiting
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
        let (decoded, rate) = decode_to_mono(&wav, 60).expect("decode wav");
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
    /// 6c: the transcription allowance is twice the audio, never under a minute.
    #[test]
    fn transcribe_allowance_is_twice_the_audio_at_least_a_minute() {
        let sr = WHISPER_SAMPLE_RATE as usize;
        assert_eq!(transcribe_allowance(0), Duration::from_secs(60));
        assert_eq!(transcribe_allowance(10 * sr), Duration::from_secs(60));
        assert_eq!(transcribe_allowance(30 * sr), Duration::from_secs(60));
        assert_eq!(transcribe_allowance(31 * sr), Duration::from_secs(62));
        assert_eq!(transcribe_allowance(600 * sr), Duration::from_mins(20));
        assert_eq!(
            transcribe_allowance(600 * sr + 1),
            Duration::from_secs(1202)
        );
    }

    #[test]
    fn prepare_samples_decodes_and_resamples() {
        let samples: Vec<f32> = (0..2000).map(|i| ((i as f32) * 0.03).sin() * 0.4).collect();
        let wav = wav_mono_i16(8_000, &samples);
        let prepared = prepare_samples(&wav, 60).expect("prepare");
        let ratio = prepared.len() as f32 / samples.len() as f32;
        assert!((1.8..2.2).contains(&ratio), "8k→16k ~doubles, got {ratio}");
    }

    /// A WAV that declares sample rate 0 made symphonia panic inside the demuxer, which killed the STT
    /// engine thread (overnight torture run). It must now come back as a typed `BadAudio` error.
    #[test]
    fn wav_with_sample_rate_zero_is_bad_audio_not_a_panic() {
        let wav = wav_mono_i16(0, &[0.1_f32; 64]);
        let err = prepare_samples(&wav, 60).expect_err("sample rate 0 must be rejected");
        assert!(
            err.downcast_ref::<BadAudio>().is_some(),
            "typed, so the API answers 400: {err}"
        );
    }

    /// A WAV header that patches the data-chunk size: the container then declares `frames`
    /// frames whatever the bytes that follow (S4 test helper).
    fn with_declared_frames(mut wav: Vec<u8>, frames: u32) -> Vec<u8> {
        wav[4..8].copy_from_slice(&(36 + frames * 2).to_le_bytes());
        wav[40..44].copy_from_slice(&(frames * 2).to_le_bytes());
        wav
    }

    /// 0.9.1 S4 (I-01): a container declaring 10 h of audio is refused before decoding, with
    /// a typed error (400), and the message names the limit.
    #[test]
    fn declared_length_over_the_cap_is_refused_before_decoding() {
        let wav = with_declared_frames(wav_mono_i16(8_000, &[0.1_f32; 64]), 8_000 * 36_000);
        let err = prepare_samples(&wav, 7_200).expect_err("10 h must be refused");
        assert!(err.downcast_ref::<BadAudio>().is_some(), "typed: {err}");
        assert!(
            err.to_string().contains("limit of 7200 s"),
            "names the limit: {err}"
        );
    }

    /// The cap is inclusive: exactly `max_secs` decodes, one sample more does not.
    #[test]
    fn audio_exactly_at_the_cap_decodes_and_one_sample_more_does_not() {
        let at_cap = wav_mono_i16(8_000, &vec![0.1_f32; 8_000]);
        let (decoded, _) = decode_to_mono(&at_cap, 1).expect("1 s at a 1 s cap");
        assert_eq!(decoded.len(), 8_000);
        let over = wav_mono_i16(8_000, &vec![0.1_f32; 8_001]);
        let err = decode_to_mono(&over, 1).expect_err("1 s + 1 sample over a 1 s cap");
        assert!(err.to_string().contains("limit of 1 s"), "{err}");
    }

    /// The bound inside the decode loop holds when the container's length is wrong. Fixture:
    /// 0.3 s of 160 kb/s MP3 followed by 3 s of 8 kb/s silence, no Xing header. symphonia
    /// estimates the length from the first frames' bitrate — about 0.83 s — so the up-front
    /// check passes a 1 s cap, while the file decodes to about 3.5 s. A low-bitrate tail is the
    /// shape of the attack (outside review I-01), so this is the bound that matters.
    #[test]
    fn decode_loop_stops_at_the_cap_when_the_declared_length_understates() {
        let mp3 = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/stt_vbr_understated_3s.mp3"
        ));
        let (full, rate) = decode_to_mono(mp3, 60).expect("decodes under a generous cap");
        assert_eq!(rate, 16_000);
        assert!(full.len() > 3 * 16_000, "fixture is > 3 s: {}", full.len());
        let err = decode_to_mono(mp3, 1).expect_err("3.5 s over a 1 s cap");
        assert!(err.to_string().contains("limit of 1 s"), "{err}");
    }

    /// The cap is counted in source samples, so the rate ceiling bounds the memory: above
    /// 48 kHz is a 400 (0.9.1 S4b), 48 kHz itself decodes.
    #[test]
    fn sample_rate_above_48_khz_is_bad_audio() {
        let wav = wav_mono_i16(96_000, &[0.1_f32; 64]);
        let err = prepare_samples(&wav, 7_200).expect_err("96 kHz must be refused");
        assert!(err.downcast_ref::<BadAudio>().is_some(), "typed: {err}");
        let (_, rate) = decode_to_mono(&wav_mono_i16(48_000, &[0.1_f32; 64]), 7_200).unwrap();
        assert_eq!(rate, 48_000);
    }

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(format!(
            "{}/tests/fixtures/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap()
    }

    /// 0.9.1 S4b: a 3 KB M4A whose `stsc` declares 0xFFFFFFFF entries made symphonia's probe
    /// ask for 64 GiB and abort the process (reproduced). It is now a 400 before the probe; the
    /// same file unpatched still decodes.
    #[test]
    fn m4a_with_an_impossible_sample_table_is_refused_before_the_probe() {
        let (decoded, rate) = decode_to_mono(&fixture("stt_1s_aac.m4a"), 60).expect("clean M4A");
        assert_eq!(rate, 16_000);
        assert!(decoded.len() >= 16_000);
        let err = prepare_samples(&fixture("stt_m4a_stsc_count_overflow.m4a"), 60)
            .expect_err("must be refused");
        assert!(err.downcast_ref::<BadAudio>().is_some(), "typed: {err}");
        assert!(err.to_string().contains("'stsc' table declares"), "{err}");
    }

    /// The guard tries every `ftyp` as a root, as symphonia's probe scans for it: junk in
    /// front of the bad file does not get it past.
    #[test]
    fn the_mp4_guard_also_finds_a_root_behind_leading_junk() {
        let mut shifted = vec![0x55_u8; 37];
        shifted.extend_from_slice(&fixture("stt_m4a_stsc_count_overflow.m4a"));
        let err = check_bmff_tables(&shifted).expect_err("must be refused");
        assert!(err.to_string().contains("'stsc'"), "{err}");
        // A chance "ftyp" in other data, with no valid boxes behind it, is not a refusal.
        let mut noise = vec![0_u8; 64];
        noise[20..24].copy_from_slice(b"ftyp");
        assert!(check_bmff_tables(&noise).is_ok());
    }

    /// 0.9.1 S4c (Fable adversarial review, reproduced): an 84-byte M4A whose
    /// `moov/udta/meta/ilst/©ART/data` box declares a 2^50-byte 64-bit size made symphonia's
    /// probe allocate it and abort the process. Refused before the probe now.
    #[test]
    fn m4a_with_an_impossible_64_bit_box_size_is_refused_before_the_probe() {
        let bad = fixture("stt_m4a_metadata_largesize.m4a");
        assert_eq!(bad.len(), 84);
        let err = prepare_samples(&bad, 60).expect_err("must be refused");
        assert!(err.downcast_ref::<BadAudio>().is_some(), "typed: {err}");
        assert!(err.to_string().contains("'data' box declares"), "{err}");
        // The clean M4A (with its own boxes) still passes the scan.
        assert!(check_bmff_tables(&fixture("stt_1s_aac.m4a")).is_ok());
    }

    /// 0.9.1 S4c: a `QuickTime` LPCM (v2) sample entry declaring 2^32 - 1 frames per packet;
    /// the PCM decoder would allocate frames x channels x sample size when built. Refused
    /// before the decoder exists. Built from an ffmpeg MOV with the sample entry replaced.
    #[test]
    fn lpcm_entry_with_huge_frames_per_packet_is_refused_before_the_decoder() {
        let err = prepare_samples(&fixture("stt_mov_lpcm_frames_per_packet_overflow.mov"), 60)
            .expect_err("must be refused");
        assert!(err.downcast_ref::<BadAudio>().is_some(), "typed: {err}");
        assert!(err.to_string().contains("frames per packet"), "{err}");
    }

    /// The MP4 walk must not refuse ordinary files: ffmpeg MOVs with each PCM sample-entry
    /// version (v0 `sowt`, v1 `fl32`) and the AAC M4A decode. (A byte scan for 64-bit sizes
    /// was tried first and refused an ffmpeg MOV: table data looks like box headers.)
    #[test]
    fn the_mp4_walk_passes_ordinary_files() {
        for f in [
            "stt_1s_aac.m4a",
            "stt_mov_sowt_v0.mov",
            "stt_mov_fl32_v1.mov",
        ] {
            let (decoded, _) =
                decode_to_mono(&fixture(f), 60).unwrap_or_else(|e| panic!("{f}: {e}"));
            assert!(!decoded.is_empty(), "{f}");
        }
    }

    /// A sample rate under 4 kHz is a 400 (it would expand thousands of times when resampled).
    #[test]
    fn sample_rate_below_4_khz_is_bad_audio() {
        let err = prepare_samples(&wav_mono_i16(1, &[0.1_f32; 64]), 7_200).expect_err("1 Hz");
        assert!(err.downcast_ref::<BadAudio>().is_some(), "typed: {err}");
        decode_to_mono(&wav_mono_i16(8_000, &[0.1_f32; 64]), 7_200).expect("8 kHz decodes");
    }

    /// ALAC is not decoded any more (its decoder sizes a buffer from the file's magic cookie
    /// before any packet, 0.9.1 S4b): an ALAC M4A is a 400, not a decode.
    #[test]
    fn alac_is_refused() {
        let err = prepare_samples(&fixture("stt_1s_alac.m4a"), 60).expect_err("ALAC is off");
        assert!(err.downcast_ref::<BadAudio>().is_some(), "typed: {err}");
    }

    /// Garbage bytes are rejected with an error, not a panic.
    #[test]
    fn prepare_samples_rejects_garbage() {
        let err = prepare_samples(b"not audio at all", 60);
        let err = err.expect_err("non-audio bytes must error");
        assert!(
            err.downcast_ref::<BadAudio>().is_some(),
            "typed, so the API answers 400: {err}"
        );
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

    /// A request that reached the engine keeps `active()` at 1 (its admission permit travels in
    /// the command) even after the caller is abandoned, and releases when the engine drops the command.
    #[tokio::test]
    async fn active_holds_for_a_stt_request_until_the_engine_is_done() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<SttCommand>(8);
        let h = SttHandle::from_sender(tx, "t-stt");
        assert_eq!(h.active(), 0, "idle");

        let caller = tokio::spawn({
            let h = h.clone();
            async move {
                h.transcribe(b"a".to_vec(), None, false).await.ok();
            }
        });
        let cmd = rx.recv().await.unwrap();
        assert_eq!(h.active(), 1, "request delivered to the engine");

        caller.abort();
        let _ = caller.await;
        assert_eq!(
            h.active(),
            1,
            "abandoned caller: the engine still holds the permit"
        );

        drop(cmd);
        assert_eq!(h.active(), 0, "engine finished (command dropped)");
    }

    /// D3-S1.D3a: with the closing flag set, a submit that already holds its permit is refused
    /// (retryable `AdmitError::Closing`) and the permit is released; cleared again, it is accepted.
    #[tokio::test]
    async fn closing_refuses_a_submit_and_releases_its_permit() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<SttCommand>(8);
        let h = SttHandle::from_sender(tx, "t-stt");

        ManagedEngine::set_closing(&h, true);
        let refused = h.transcribe(b"a".to_vec(), None, false).await;
        assert!(matches!(refused, Err(AdmitError::Closing)), "{refused:?}");
        assert_eq!(h.active(), 0, "the refused submit's permit is released");
        assert!(rx.try_recv().is_err(), "nothing reached the engine");

        ManagedEngine::set_closing(&h, false);
        let caller = tokio::spawn({
            let h = h.clone();
            async move { h.transcribe(b"a".to_vec(), None, false).await.ok() }
        });
        let cmd = rx.recv().await.unwrap();
        assert_eq!(h.active(), 1, "accepted: the command reached the engine");
        drop(cmd);
        let _ = caller.await;
    }
}

#[cfg(test)]
mod language_tests {
    #![allow(clippy::unwrap_used)]
    use super::normalize_language;

    /// FM-3: the unambiguous forms normalise; `auto` and empty mean detect.
    #[test]
    fn language_hint_normalises_unambiguous_forms() {
        for (raw, want) in [
            ("pl", Some("pl")),
            ("PL", Some("pl")),
            (" pl-PL ", Some("pl")),
            ("en_US", Some("en")),
            ("zh-Hant-TW", Some("zh")),
            ("haw", Some("haw")),
            ("yue", Some("yue")),
            ("auto", None),
            ("AUTO", None),
            ("", None),
            ("  ", None),
        ] {
            assert_eq!(normalize_language(raw).unwrap().as_deref(), want, "{raw:?}");
        }
    }

    #[test]
    fn language_hint_refuses_what_whisper_has_no_token_for() {
        for raw in [
            "xx",
            "polish",
            "p",
            "-pl",
            "pl!",
            "<|pl|>",
            "e\u{301}n",
            "x\0y",
        ] {
            let msg = normalize_language(raw).unwrap_err();
            assert!(
                msg.contains("pl-PL") && msg.contains("auto"),
                "{raw:?}: {msg}"
            );
        }
    }
}
