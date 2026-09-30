// ============================================================
// src/handlers/realtime.rs — GET /v1/realtime (WebSocket)
// ============================================================
// Bidirectional realtime voice pipeline:
//   Client binary frames (PCM16 16 kHz mono i16 LE)
//     → server energy VAD
//     → Whisper STT (batch, final transcript only in v1)
//     → LLM token stream (thinking stripped)
//     → sentence-boundary Kokoro/SpeechT5 TTS
//   → Client binary frames (PCM16 16 kHz mono i16 LE)
//
// Control messages are JSON text frames in both directions.
// See INTERFACE.md (`GET /v1/realtime`) for the full protocol spec.
// ============================================================

use std::sync::{
    Arc, RwLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::{
    extract::{
        State,
        ws::{Message as WsMsg, WebSocket, WebSocketUpgrade},
    },
    response::Response,
};
use futures_util::{SinkExt as _, StreamExt as _};
use metrics::{counter, histogram};
use serde::{Deserialize, Serialize};
use tokio::{
    sync::{Notify, mpsc},
    time,
};

use crate::{
    app_state::AppState,
    cb_engine::SubmitResult,
    model_manager::EngineHandleKind,
    model_manager::ModelKind,
    model_manager::RealtimeSlot,
    model_manager::ReasoningParser,
    ov_cb::GenParams,
    pipelines::stt::resample_to_16k,
    prompt_builder::{ThinkFilter, ThinkPiece},
    realtime_types::{ClientHistoryEntry, SessionSnapshot},
    streaming::{StreamEvent, stream_channel},
};

// ── Request / response token budget ──────────────────────────────────────────
// Voice responses should be short and snappy. 120 tokens ≈ 15 s at normal
// speaking pace — 3–4 sentences, enough for conversational turns.
const VOICE_MAX_TOKENS: usize = 120;

// ── PCM16 audio constants ─────────────────────────────────────────────────────
const PCM_SAMPLE_RATE: u32 = 16_000; // Whisper + SpeechT5 native rate
const PCM_BYTES_PER_SAMPLE: usize = 2; // i16 LE

// ── VAD tuning ────────────────────────────────────────────────────────────────
// RMS threshold in normalised float (0.0–1.0). 0.015 ≈ –36 dBFS, comfortably
// above typical microphone noise floor and below quiet speech.
const VAD_THRESHOLD: f32 = 0.015;
// Consecutive samples above threshold before we declare speech has started.
const VAD_ONSET_SAMPLES: usize = (PCM_SAMPLE_RATE as usize * 100) / 1000; // 100 ms
// Consecutive samples below threshold before we commit the utterance.
const VAD_OFFSET_SAMPLES: usize = (PCM_SAMPLE_RATE as usize * 500) / 1000; // 500 ms

// ── Safety bounds ─────────────────────────────────────────────────────────────
// Whisper's context window is 30 s. Auto-commit the utterance at this limit so
// the VAD audio buffer cannot grow without bound if speech never stops.
const MAX_UTTERANCE_SAMPLES: usize = PCM_SAMPLE_RATE as usize * 30; // 480 000 samples ≈ 960 KB

// Trim conversation history to this many user+assistant turn pairs.
// Bounds both memory and the LLM prompt length for long sessions.
const MAX_HISTORY_TURNS: usize = 20;

// Client-supplied history cap — prevents unbounded memory from a history replay.
// 200 entries covers a very long multi-session conversation; embedding retrieval
// keeps LLM context cost bounded regardless of pool size.
const MAX_CLIENT_HISTORY: usize = 200;

// Embedding-based retrieval tuning.
const MEMORY_SIM_THRESHOLD: f32 = 0.5;
const MEMORY_TOP_K: usize = 3;

// Barge-in: responses synthesised when the user utters the configured phrase.
// Rotated by turn_id % len so the response varies across turns.
const STOP_RESPONSES: &[&str] = &["Calming down.", "Going shut up mode."];

// Disconnect an idle client after this duration of no incoming audio.
const IDLE_TIMEOUT: Duration = Duration::from_mins(5);

// ── Sentence splitter tuning ──────────────────────────────────────────────────
// Don't split until the buffer has at least this many chars, so we don't chop
// "Mr." or "U.S." into single-word TTS calls.
// 15: abbreviations like "Mr.", "Dr.", "U.S." all appear at byte offsets < 10
// so they are still protected. Lowered from 25 — the original 25 was causing
// the first TTS call to receive 60-80 char sentences (7+ s audio, ~14 s
// SpeechT5 synthesis), resulting in 11-14 s tts_first_ms and users re-speaking
// before hearing any audio, triggering premature cancel.
const SENTENCE_MIN_LEN: usize = 15;

// ─────────────────────────────────────────────────────────────────────────────
// Protocol types
// ─────────────────────────────────────────────────────────────────────────────

/// JSON text frames the client sends to the server.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
// Config carries many Option fields; the size difference vs Cancel is intentional
// and short-lived (one frame per turn, immediately destructured).
#[allow(clippy::large_enum_variant)]
enum ClientEvent {
    /// Configure or reconfigure the session. Fields are all optional — only
    /// the provided fields are updated; absent fields keep their current value.
    Config {
        language: Option<String>,
        stt_model: Option<String>,
        llm_model: Option<String>,
        /// TTS model id (e.g. `"speecht5-tts-ov"`). Absent → text-only output.
        tts_model: Option<String>,
        /// Voice name passed to `TtsHandle::synthesize` (e.g. `"af_nicole"`).
        tts_voice: Option<String>,
        /// Optional system prompt injected as the first message in every LLM
        /// turn. Replaces any previously set value; send `""` to clear.
        system_prompt: Option<String>,
        /// If the transcript contains this phrase (normalised substring match),
        /// the server synthesises a brief stop response and skips the LLM turn.
        barge_in_phrase: Option<String>,
        /// Embedding model used for memory retrieval and back-filling history
        /// entries that lack a pre-computed vector.
        embed_model: Option<String>,
        /// Client-supplied prior conversation history (stateless server mode).
        /// Replaces the current `history_entries` when present.
        history: Option<Vec<ClientHistoryEntry>>,
        /// Hard cap on LLM output tokens for this session. `None` → server
        /// default (`VOICE_MAX_TOKENS`). Set low (e.g. 30) to enforce brevity
        /// on hardware where TTS latency is the bottleneck.
        max_tokens: Option<u32>,
        /// Stop speaking after this many TTS sentences per turn. `None` →
        /// unlimited. Useful for enforcing concise answers (e.g. 2–3 sentences)
        /// without a hard token cap that can cut off mid-sentence.
        max_sentences: Option<u32>,
    },
    /// Interrupt the current LLM generation and TTS synthesis.
    Cancel,
    /// Append past-session history entries into the retrieval pool without
    /// replacing the current session history. Entries go into `history_entries`
    /// only — they do not enter the rolling LLM context window. The pipeline
    /// task embeds them and fires a `TurnHistory` event per entry so the client
    /// can update its local vectors.
    InjectHistory { entries: Vec<ClientHistoryEntry> },
    /// Destructively replace the retrieval pool. Clears all current entries and
    /// processes the provided batch (embedding any that lack vectors). Responds
    /// with a single `HistoryReloaded` event. Empty `entries` = clear only.
    /// The rolling LLM context window is not affected.
    ReloadHistory { entries: Vec<ClientHistoryEntry> },
    /// D8 (the project's internal engineering log): explicitly
    /// ask the server to start loading `model_id`, independent of `Config` —
    /// for pre-warming before a session starts talking, or to get load
    /// progress feedback distinct from the per-turn `Error`/`Done` events.
    /// Arbitrated identically to a `Config`-driven request through the same
    /// `resolve_realtime_llm` chokepoint (never a bypass with authority of
    /// its own). Deliberately does **not** itself register into the
    /// realtime serving set (D2) — only an actual `Config` resolution that
    /// names this model does; otherwise a client pre-warming several
    /// candidates "just in case" would eviction-protect all of them and
    /// undermine the "minimize resident footprint" half of the optimal-set
    /// goal. Progress reported via `ModelLoading` events.
    RequestModelLoad { model_id: String },
}

/// JSON text frames the server sends to the client.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ServerEvent<'a> {
    /// Energy VAD detected speech onset.
    SpeechStarted,
    /// Energy VAD detected end of utterance — STT is starting.
    SpeechStopped,
    /// Whisper result for the committed utterance.
    Transcript {
        text: &'a str,
        /// Always `true` in v1 (no streaming partial ASR).
        #[serde(rename = "final")]
        final_: bool,
    },
    /// One LLM content token (thinking stripped).
    Text { content: &'a str },
    /// LLM generation complete for this turn. `turn_id` joins server and
    /// client JSONL logs — both sides emit it so events can be correlated.
    Done { turn_id: u64 },
    /// Error in one of the pipeline stages.
    Error { stage: &'a str, message: &'a str },
    /// History entry — sent after each turn (`turn_id` = `Some`) or when back-filling
    /// embeddings for client-supplied history entries (`turn_id` = `None`).
    TurnHistory {
        /// `Some(id)` = new turn just completed; `None` = re-embed of existing entry.
        turn_id: Option<u64>,
        user: &'a str,
        assistant: &'a str,
        /// Embedding vector (user+"\n"+assistant). Absent when no embed model is
        /// configured or embedding failed.
        #[serde(skip_serializing_if = "Option::is_none")]
        embedding: Option<Vec<f32>>,
        /// Unix timestamp in milliseconds matching the corresponding history entry.
        timestamp: u64,
        /// Model that produced `embedding` — lets the client persist and
        /// round-trip provenance alongside the vector (see `ClientHistoryEntry`).
        #[serde(skip_serializing_if = "Option::is_none")]
        embedding_model: Option<&'a str>,
        /// STT model that transcribed `user`.
        #[serde(skip_serializing_if = "Option::is_none")]
        stt_model: Option<&'a str>,
        /// LLM model that generated `assistant`.
        #[serde(skip_serializing_if = "Option::is_none")]
        llm_model: Option<&'a str>,
    },
    /// Barge-in phrase was detected — server synthesised a stop response and
    /// skipped the normal LLM turn.
    Stopped { turn_id: u64 },
    /// Sent once after a `reload_history` command completes. `count` is the
    /// number of entries now in the retrieval pool (0 if entries was empty).
    HistoryReloaded { count: usize },
    /// D6 (the project's internal engineering log), extended to
    /// all three slots by RTCC
    /// (the project's internal engineering log): what the session's most recent `Config` resolution
    /// actually produced. Sent once after every `Config` event (initial or
    /// mid-session re-resolve). STT/LLM/TTS all go through real
    /// server-side arbitration (`resolve_realtime_slot`) — `stt_source`/
    /// `tts_source` mirror `llm_source`'s vocabulary exactly, no longer a
    /// plain echo of what was requested.
    ModelSelection {
        #[serde(skip_serializing_if = "Option::is_none")]
        stt_model: Option<&'a str>,
        /// `"requested"` | `"default"` | `"substituted"`. Absent only in
        /// mock/test mode (no `ModelManager` — nothing to arbitrate).
        #[serde(skip_serializing_if = "Option::is_none")]
        stt_source: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        stt_reason: Option<&'a str>,
        llm_model: &'a str,
        /// `"requested"` | `"default"` | `"substituted"`.
        llm_source: &'a str,
        /// Machine-usable reason, present only when `llm_source !=
        /// "requested"` or a background load was needed — e.g.
        /// `"would_evict_protected"`, `"not_resident_no_headroom"`.
        #[serde(skip_serializing_if = "Option::is_none")]
        llm_reason: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tts_model: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tts_source: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tts_reason: Option<&'a str>,
    },
    /// D8: progress on a background model load — either an explicit
    /// `request_model_load`, or an implicit one `resolve_realtime_llm`
    /// kicked off during `Config` resolution. May fire more than once per
    /// `model_id` across a session's lifetime (e.g. pre-warmed, evicted by
    /// grace-window expiry, reloaded).
    ModelLoading {
        model_id: &'a str,
        /// `"started"` | `"ready"` | `"failed"`.
        stage: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<&'a str>,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Session state
// ─────────────────────────────────────────────────────────────────────────────

/// Per-connection session: configured model IDs, voice, language, and the
/// growing conversation history for multi-turn LLM context.
#[derive(Clone, Default)]
struct Session {
    stt_model: Option<String>,
    llm_model: Option<String>,
    tts_model: Option<String>,
    tts_voice: Option<String>,
    language: Option<String>,
    /// System prompt prepended to every LLM turn. `None` = no system message.
    /// Empty string clears a previously set prompt.
    system_prompt: Option<String>,
    /// Conversation history — grows each completed turn, shared across
    /// all utterances on this WebSocket connection.
    history: Vec<RtMessage>,
    /// Phrase that triggers barge-in when detected in the transcript.
    barge_in_phrase: Option<String>,
    /// Embedding model used for memory retrieval.
    embed_model: Option<String>,
    /// Full history for embedding-based retrieval (up to `MAX_CLIENT_HISTORY`).
    history_entries: Vec<ClientHistoryEntry>,
    /// Per-session LLM output token cap. `None` → `VOICE_MAX_TOKENS`.
    max_tokens: Option<u32>,
    /// Stop speaking after this many TTS sentences per turn. `None` → unlimited.
    max_sentences: Option<u32>,
}

/// Minimal chat message serialisable into the chat template context.
#[derive(Clone, Serialize)]
struct RtMessage {
    role: String,
    content: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Utterance context — VAD → pipeline channel payload
// ─────────────────────────────────────────────────────────────────────────────

/// What the VAD commits to the pipeline: the audio plus timing and quality
/// metadata captured at speech-end time.
struct UtteranceCtx {
    audio: Vec<i16>,
    /// RMS energy of the committed audio (normalised 0.0–1.0).
    energy: f32,
    /// Wall-clock ms from `SpeechStarted` to `SpeechEnded`.
    utterance_duration_ms: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Turn context — per-turn telemetry accumulator
// ─────────────────────────────────────────────────────────────────────────────

/// Telemetry data accumulated across all pipeline stages for one voice turn.
/// Passed by `&mut` through `run_response` → `run_llm_tts` →
/// `drain_voice_tokens` / `synthesize_and_send`. Emitted as a structured log
/// line at Done — when using a JSON tracing subscriber this produces JSONL
/// that can be joined with the client log on `turn_id`.
struct TurnCtx {
    conn_id: u64,
    turn_id: u64,
    // VAD metrics (set from UtteranceCtx at construction)
    audio_samples: usize,
    utterance_duration_ms: u64,
    utterance_energy: f32,
    // STT metrics (set after run_stt returns)
    whisper_ms: u64,
    transcript_empty: bool,
    // LLM metrics (set in run_llm_tts / drain_voice_tokens)
    llm_start: Option<Instant>,
    llm_ttft_ms: u64,
    llm_tokens: usize,
    think_tokens: usize,
    // TTS metrics (updated by synthesize_and_send / computed at Done)
    tts_sentences: usize,
    tts_first_at: Option<Instant>, // raw instant: anchors tts_first_ms computation
    tts_first_ms: u64,             // ms from llm_start to first audio sent (set at Done)
    tts_total_ms: u64,             // ms from llm_start to Done (set at Done)
    audio_sent_samples: usize,
    // Cancel tracking (set when a cancel guard fires)
    cancel_stage: &'static str,
}

impl TurnCtx {
    fn new(conn_id: u64, turn_id: u64, u: &UtteranceCtx) -> Self {
        Self {
            conn_id,
            turn_id,
            audio_samples: u.audio.len(),
            utterance_duration_ms: u.utterance_duration_ms,
            utterance_energy: u.energy,
            whisper_ms: 0,
            transcript_empty: false,
            llm_start: None,
            llm_ttft_ms: 0,
            llm_tokens: 0,
            think_tokens: 0,
            tts_sentences: 0,
            tts_first_at: None,
            tts_first_ms: 0,
            tts_total_ms: 0,
            audio_sent_samples: 0,
            cancel_stage: "none",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VoicePolicy — per-model-family prompt/output treatment
// ─────────────────────────────────────────────────────────────────────────────

/// Encapsulates all model-family-specific behaviour for the realtime voice
/// pipeline. Derived once from the model's `ReasoningParser` tag after the
/// chat context is acquired; callers check the policy fields instead of
/// inspecting the parser variant directly.
#[derive(Default)]
struct VoicePolicy {
    /// Token appended to the raw prompt after `build_prompt` to steer the
    /// model directly into its "final answer" channel (gpt-oss only).
    channel_prefill: Option<&'static str>,
    /// When `true`, `drain_voice_tokens` buffers the first ≤10 chars and
    /// strips any leaked channel-name prefix before emitting content.
    strip_output_prefix: bool,
    /// When `true`, the session `system_prompt` is passed as `model_identity`
    /// (overriding the model's built-in identity block) instead of being
    /// injected as a `role=system` message.
    system_as_identity: bool,
}

impl VoicePolicy {
    fn from_parser(parser: Option<&ReasoningParser>) -> Self {
        match parser {
            Some(ReasoningParser::GptOss) => Self {
                channel_prefill: Some("<|channel|>final<|message|>"),
                strip_output_prefix: true,
                system_as_identity: true,
            },
            // Qwen3: template handles <think>\n\n</think> suppression — no channel
            // magic or identity override.
            // Mistral/Mixtral: standard [INST] format — no reasoning blocks or
            // channel tokens.
            // Phi family: standard chat; Phi-4-mini reasoning is suppressed by
            // build_prompt's enable_thinking=false — no additional voice quirks yet.
            // None (unknown / non-family-tagged): safe defaults.
            Some(ReasoningParser::Qwen3 | ReasoningParser::Mistral | ReasoningParser::Phi)
            | None => Self::default(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Energy VAD
// ─────────────────────────────────────────────────────────────────────────────

enum VadEvent {
    SpeechStarted,
    /// The accumulated utterance audio (raw i16 LE samples, 16 kHz mono).
    SpeechEnded(Vec<i16>),
}

enum VadState {
    Silence,
    /// Above threshold but not long enough yet to commit to speech.
    /// Accumulates samples so the onset audio is included in the utterance.
    Onset {
        count: usize,
        audio: Vec<i16>,
    },
    /// Confirmed speech — accumulating samples.
    Speech {
        audio: Vec<i16>,
        /// Samples accumulated below threshold so far (candidate close).
        silence_count: usize,
    },
}

struct EnergyVad {
    state: VadState,
}

impl EnergyVad {
    fn new() -> Self {
        Self {
            state: VadState::Silence,
        }
    }

    /// Feed raw PCM16 LE bytes; returns 0–2 events per call.
    fn push_bytes(&mut self, bytes: &[u8]) -> Vec<VadEvent> {
        let samples = bytes_to_i16(bytes);
        let is_speech = rms(&samples) > VAD_THRESHOLD;
        let n = samples.len();
        let mut events = Vec::new();

        self.state = match std::mem::replace(&mut self.state, VadState::Silence) {
            VadState::Silence => {
                if is_speech {
                    // Fast-path: a single large chunk already satisfies onset duration.
                    if n >= VAD_ONSET_SAMPLES {
                        events.push(VadEvent::SpeechStarted);
                        let mut audio = Vec::with_capacity(n + 8192);
                        audio.extend_from_slice(&samples);
                        VadState::Speech {
                            audio,
                            silence_count: 0,
                        }
                    } else {
                        // Accumulate onset samples so they are included in the utterance.
                        VadState::Onset {
                            count: n,
                            audio: samples,
                        }
                    }
                } else {
                    VadState::Silence
                }
            }
            VadState::Onset { count, mut audio } => {
                if is_speech {
                    audio.extend_from_slice(&samples);
                    let total = count + n;
                    if total >= VAD_ONSET_SAMPLES {
                        events.push(VadEvent::SpeechStarted);
                        audio.reserve(8192);
                        VadState::Speech {
                            audio,
                            silence_count: 0,
                        }
                    } else {
                        VadState::Onset {
                            count: total,
                            audio,
                        }
                    }
                } else {
                    // Speech didn't sustain — discard onset audio.
                    VadState::Silence
                }
            }
            VadState::Speech {
                mut audio,
                silence_count,
            } => {
                audio.extend_from_slice(&samples);
                // Auto-commit at Whisper's 30 s context limit to bound memory usage.
                if audio.len() >= MAX_UTTERANCE_SAMPLES {
                    events.push(VadEvent::SpeechEnded(audio));
                    return events;
                }
                if is_speech {
                    VadState::Speech {
                        audio,
                        silence_count: 0,
                    }
                } else {
                    let total_silence = silence_count + n;
                    if total_silence >= VAD_OFFSET_SAMPLES {
                        events.push(VadEvent::SpeechEnded(audio));
                        VadState::Silence
                    } else {
                        VadState::Speech {
                            audio,
                            silence_count: total_silence,
                        }
                    }
                }
            }
        };

        events
    }
}

/// Root mean square of i16 samples, normalised to 0.0–1.0.
fn rms(samples: &[i16]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples
        .iter()
        .map(|&s| {
            let f = f64::from(s) / f64::from(i16::MAX);
            f * f
        })
        .sum();
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    let r = (sum_sq / samples.len() as f64).sqrt() as f32;
    r
}

// ─────────────────────────────────────────────────────────────────────────────
// PCM16 utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Interpret raw bytes as little-endian i16 samples (drop any trailing odd byte).
fn bytes_to_i16(bytes: &[u8]) -> Vec<i16> {
    bytes
        .as_chunks::<PCM_BYTES_PER_SAMPLE>()
        .0
        .iter()
        .map(|b| i16::from_le_bytes(*b))
        .collect()
}

/// Reinterpret i16 samples as raw LE bytes.
fn i16_to_bytes(samples: &[i16]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

/// Convert TTS f32 output to PCM16 LE bytes for the wire.
fn f32_to_pcm16_bytes(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|&s| {
            #[allow(clippy::cast_possible_truncation)]
            let i = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
            i.to_le_bytes()
        })
        .collect()
}

/// Wrap raw PCM16 LE bytes in a minimal WAV container so symphonia can decode
/// them inside `SttHandle::transcribe`.
fn pcm16_to_wav(pcm_bytes: &[u8], sample_rate: u32) -> Vec<u8> {
    #[allow(clippy::cast_possible_truncation)]
    let data_len = pcm_bytes.len() as u32;
    let mut v = Vec::with_capacity(44 + pcm_bytes.len());
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
    v.extend_from_slice(pcm_bytes);
    v
}

/// `Duration::as_millis()` returns `u128`; cap at `u64::MAX` (584 million years).
fn to_ms(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

// ─────────────────────────────────────────────────────────────────────────────
// Sentence splitter
// ─────────────────────────────────────────────────────────────────────────────

/// Buffers LLM tokens and emits a complete sentence each time a sentence
/// boundary is detected after `SENTENCE_MIN_LEN` characters have accumulated.
///
/// This lets TTS start on sentence 1 while the LLM is still generating
/// sentence 2, giving low apparent latency to the user.
struct SentenceSplitter {
    buf: String,
}

impl SentenceSplitter {
    fn new() -> Self {
        Self { buf: String::new() }
    }

    /// Push a token; returns a sentence if a boundary was detected.
    fn push(&mut self, token: &str) -> Option<String> {
        self.buf.push_str(token);
        if self.buf.len() < SENTENCE_MIN_LEN {
            return None;
        }
        // Double newline → paragraph boundary.
        if let Some(pos) = self.buf.find("\n\n") {
            let sentence = self.buf[..pos].trim().to_owned();
            self.buf = self.buf[pos + 2..].trim_start().to_owned();
            if !sentence.is_empty() {
                return Some(sentence);
            }
        }
        // [.!?] followed by a space or end-of-buffer.
        // Find the split index in a contained scope so the byte-slice borrow
        // ends before we mutate self.buf.
        let split_at = {
            let b = self.buf.as_bytes();
            let mut found = None;
            for i in (SENTENCE_MIN_LEN - 1)..b.len() {
                if matches!(b[i], b'.' | b'!' | b'?') {
                    let next = b.get(i + 1).copied();
                    // `next.is_none()` means "end of buffer", not "end of
                    // stream" — the LLM's next token may still land right
                    // after this byte. A digit immediately before a trailing
                    // '.' (e.g. "...pi is 3.", next token "14") is very
                    // likely a decimal point, not a sentence end — hold the
                    // boundary back and let `flush()` (true end-of-stream)
                    // or a later space/newline confirm it (see
                    // the project's internal engineering log).
                    let trailing_decimal_like = next.is_none()
                        && b[i] == b'.'
                        && b.get(i.wrapping_sub(1)).is_some_and(u8::is_ascii_digit);
                    if !trailing_decimal_like
                        && (next.is_none() || matches!(next, Some(b' ' | b'\n')))
                    {
                        found = Some(i);
                        break;
                    }
                }
            }
            found
        };
        if let Some(i) = split_at {
            let sentence = self.buf[..=i].trim().to_owned();
            self.buf = self.buf[i + 1..].trim_start().to_owned();
            if !sentence.is_empty() {
                return Some(sentence);
            }
        }
        None
    }

    /// Flush whatever remains in the buffer as the final sentence.
    fn flush(&mut self) -> Option<String> {
        let s = std::mem::take(&mut self.buf);
        let s = s.trim().to_owned();
        if s.is_empty() { None } else { Some(s) }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Current wall-clock time as Unix milliseconds.
fn unix_now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// Return `true` if `transcript` contains `phrase` after both are normalised
/// (lowercase, strip non-alphanumeric, collapse whitespace).
fn is_barge_in_phrase(transcript: &str, phrase: &str) -> bool {
    fn normalise(s: &str) -> String {
        s.chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
    let norm_t = normalise(transcript);
    let norm_p = normalise(phrase);
    !norm_p.is_empty() && norm_t.contains(&norm_p)
}

/// Cosine similarity between two equal-length float vectors.
/// Returns 0.0 if either vector is zero-magnitude or lengths differ.
#[allow(clippy::cast_precision_loss)]
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a < f32::EPSILON || mag_b < f32::EPSILON {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

/// Embed `query`, compute cosine similarity against entries that have embeddings,
/// filter by `MEMORY_SIM_THRESHOLD`, and return the top-`MEMORY_TOP_K` entries.
async fn retrieve_memories<'a>(
    query: &str,
    entries: &'a [ClientHistoryEntry],
    handle: &crate::embed_engine::EmbeddingHandle,
) -> Vec<&'a ClientHistoryEntry> {
    let Ok(output) = handle.embed(vec![query.to_owned()]).await else {
        return Vec::new();
    };
    let Some(query_vec) = output.vectors.into_iter().next() else {
        return Vec::new();
    };
    // L2: observability for incompatible-embedding pools. An entry embedded by a
    // different model lives in a different vector space — cosine similarity
    // against it is meaningless even when the vector *length* happens to match,
    // so a length-only check would let that pair silently score a bogus
    // similarity instead of being recognised as incompatible. Entries with a
    // recorded `embedding_model` are gated on an exact match against the query's
    // model; entries with no recorded model (pre-provenance / older client)
    // fall back to the previous length-only check. Either way an incompatible
    // entry scores 0.0 (dropped by the threshold filter below) and is counted
    // so operators can see degraded recall.
    let query_model = handle.model_id();
    let query_dim = query_vec.len();
    let mut model_mismatch = 0usize;
    let mut dim_mismatch = 0usize;
    let mut scored: Vec<(f32, &ClientHistoryEntry)> = entries
        .iter()
        .filter_map(|e| {
            let emb = e.embedding.as_deref()?;
            let compatible = if let Some(entry_model) = e.embedding_model.as_deref() {
                let same = entry_model == query_model;
                if !same {
                    model_mismatch += 1;
                }
                same
            } else {
                let same_dim = emb.len() == query_dim;
                if !same_dim {
                    dim_mismatch += 1;
                }
                same_dim
            };
            let score = if compatible {
                cosine_sim(&query_vec, emb)
            } else {
                0.0
            };
            Some((score, e))
        })
        .filter(|(score, _)| *score >= MEMORY_SIM_THRESHOLD)
        .collect();
    if model_mismatch > 0 || dim_mismatch > 0 {
        tracing::warn!(
            query_model,
            query_dim,
            model_mismatch,
            dim_mismatch,
            "realtime: memory retrieval degraded — entries embedded by a \
             different model (or, absent that provenance, a different \
             dimension) score 0 and are never retrieved"
        );
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    scored
        .into_iter()
        .take(MEMORY_TOP_K)
        .map(|(_, e)| e)
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Outgoing message helpers
// ─────────────────────────────────────────────────────────────────────────────

type OutTx = mpsc::Sender<WsMsg>;

async fn send_event(out: &OutTx, event: &ServerEvent<'_>) {
    if let Ok(json) = serde_json::to_string(event) {
        let _ = out.send(WsMsg::Text(json.into())).await;
    }
}

async fn send_audio(out: &OutTx, pcm16_bytes: Vec<u8>) {
    let _ = out.send(WsMsg::Binary(pcm16_bytes.into())).await;
}

/// D8: watches `model_id` until it becomes `Ready` (or the wait times out)
/// and emits the matching `model_loading` completion event. Polling, not
/// event-driven — `ModelManager` has no load-completion notification
/// primitive; a 200ms poll is cheap and this only runs while a background
/// load from `resolve_realtime_llm`/`request_model_load` is genuinely in
/// flight, never per-turn. `mm.is_ready` is a plain `HashMap` read behind a
/// short-held `RwLock`, so polling it does not contend with the load itself.
fn spawn_model_ready_watcher(
    mm: Arc<crate::model_manager::ModelManager>,
    model_id: String,
    out_tx: OutTx,
) {
    const POLL_INTERVAL: Duration = Duration::from_millis(200);
    // Cold GPU load ceiling — matches the supervisor's own
    // `health_timeout_secs` default (90s) rather than inventing a new number.
    const WATCH_TIMEOUT: Duration = Duration::from_secs(90);
    tokio::spawn(async move {
        let start = Instant::now();
        loop {
            if mm.is_ready(&model_id) {
                send_event(
                    &out_tx,
                    &ServerEvent::ModelLoading {
                        model_id: &model_id,
                        stage: "ready",
                        detail: None,
                    },
                )
                .await;
                return;
            }
            if start.elapsed() > WATCH_TIMEOUT {
                send_event(
                    &out_tx,
                    &ServerEvent::ModelLoading {
                        model_id: &model_id,
                        stage: "failed",
                        detail: Some("load did not complete within the watch window"),
                    },
                )
                .await;
                return;
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Response pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Run one full STT → LLM → TTS pipeline turn.
///
/// Returns `(user_text, assistant_text)`. Both are empty on error or
/// cancellation — caller should not add empty pairs to history.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
async fn run_response(
    audio_i16: Vec<i16>,
    session: &Session,
    out: &OutTx,
    cancel: &AtomicBool,
    cancel_notify: &Notify,
    state: &AppState,
    ctx: &mut TurnCtx,
    embed_handle: Option<&crate::embed_engine::EmbeddingHandle>,
) -> (String, String) {
    // Pipeline owns the cancel reset. The receiver task only ever sets it to
    // true; resetting here (before any await) ensures the prior turn has
    // fully exited before the flag is cleared.
    cancel.store(false, Ordering::Release);

    // ── Cross-pipeline device admission (step 3) ─────────────────────────────
    // The turn's whole multi-device cost, acquired atomically before touching
    // any per-engine gate — see the project's internal engineering log
    // Part 3. STT/LLM/TTS genuinely nest within one turn (the LLM engine's own
    // permit is held across every inline TTS call), so acquiring per-stage
    // would recreate the hold-and-wait risk the design exists to avoid.
    // Embed's device is deliberately excluded: `retrieve_memories` runs inside
    // `run_llm_tts`, not here, so including it would mean threading the lease
    // down another layer for a resource that's already gated by its own
    // per-engine semaphore — see the project's internal engineering log 2026-07-17 "realtime turn
    // admission lease".
    let mm_ref = state.model_manager.as_ref();
    let stt_device = mm_ref
        .zip(session.stt_model.as_deref())
        .map(|(mm, m)| mm.record_device(m));
    let llm_device = mm_ref
        .zip(session.llm_model.as_deref())
        .map(|(mm, m)| mm.record_device(m));
    let tts_device = mm_ref
        .zip(session.tts_model.as_deref())
        .map(|(mm, m)| mm.record_device(m));
    let mut turn_wants: Vec<(String, u32)> = Vec::new();
    for device in [&stt_device, &llm_device, &tts_device]
        .into_iter()
        .flatten()
    {
        turn_wants.push((device.clone(), 1));
    }
    let mut turn_lease = match state
        .device_budgets
        .admit(&turn_wants, crate::admission::WorkClass::RealtimeTurn)
        .await
    {
        Ok(lease) => lease,
        Err(e) => {
            counter!("rustedvino_realtime_turns_total", "result" => "admission_rejected")
                .increment(1);
            tracing::warn!(
                conn_id = ctx.conn_id,
                turn_id = ctx.turn_id,
                err = %e,
                "realtime: device admission rejected"
            );
            send_event(
                out,
                &ServerEvent::Error {
                    stage: "admission",
                    message: &e.to_string(),
                },
            )
            .await;
            send_event(
                out,
                &ServerEvent::Done {
                    turn_id: ctx.turn_id,
                },
            )
            .await;
            return (String::new(), String::new());
        }
    };

    // ── STT ───────────────────────────────────────────────────────────────────
    let stt_start = Instant::now();
    let wav = pcm16_to_wav(&i16_to_bytes(&audio_i16), PCM_SAMPLE_RATE);
    let transcript = match run_stt(wav, session, state).await {
        Ok(t) => t,
        Err(e) => {
            counter!("rustedvino_realtime_turns_total", "result" => "stt_error").increment(1);
            tracing::warn!(
                conn_id = ctx.conn_id,
                turn_id = ctx.turn_id,
                err = %e,
                "realtime: stt error"
            );
            send_event(
                out,
                &ServerEvent::Error {
                    stage: "asr",
                    message: &e.to_string(),
                },
            )
            .await;
            send_event(
                out,
                &ServerEvent::Done {
                    turn_id: ctx.turn_id,
                },
            )
            .await;
            return (String::new(), String::new());
        }
    };
    // STT is done: give back its device token before the LLM stage starts —
    // holding it any longer wastes budget for no safety benefit (Part 3).
    if let Some(ref device) = stt_device {
        turn_lease.release(device, 1);
    }

    ctx.whisper_ms = to_ms(stt_start.elapsed());
    histogram!("rustedvino_realtime_stt_duration_seconds")
        .record(stt_start.elapsed().as_secs_f64());
    tracing::info!(
        conn_id = ctx.conn_id,
        turn_id = ctx.turn_id,
        stt_ms = ctx.whisper_ms,
        "realtime: stt complete"
    );

    let transcript = transcript.trim().to_owned();
    tracing::debug!(
        conn_id = ctx.conn_id,
        turn_id = ctx.turn_id,
        transcript = %transcript,
        "realtime: stt transcript"
    );
    if transcript.is_empty() {
        ctx.transcript_empty = true;
        tracing::debug!(
            conn_id = ctx.conn_id,
            turn_id = ctx.turn_id,
            "realtime: empty transcript — skipping turn"
        );
        send_event(
            out,
            &ServerEvent::Done {
                turn_id: ctx.turn_id,
            },
        )
        .await;
        return (String::new(), String::new());
    }
    // Whisper hallucinates short phrases ("Thank you.", "Hmm.", etc.) from
    // ambient noise, especially at session start or during near-silence.
    // Require at least 3 words before sending to the LLM.
    if transcript.split_whitespace().count() < 3 {
        ctx.transcript_empty = true;
        tracing::debug!(
            conn_id = ctx.conn_id,
            turn_id = ctx.turn_id,
            transcript = %transcript,
            "realtime: transcript too short — likely hallucination, skipping"
        );
        send_event(
            out,
            &ServerEvent::Done {
                turn_id: ctx.turn_id,
            },
        )
        .await;
        return (String::new(), String::new());
    }

    // ── Barge-in check ───────────────────────────────────────────────────────
    // Must run after hallucination filter, before the cancel check and before
    // sending the Transcript event (barge-in is a control interaction, not a
    // content turn — the client gets Stopped instead of Transcript+Done).
    if let Some(ref phrase) = session.barge_in_phrase
        && is_barge_in_phrase(&transcript, phrase)
    {
        ctx.cancel_stage = "barge_in";
        // Anchor tts timing so settle_tts_timing can compute tts_first_ms.
        ctx.llm_start = Some(Instant::now());
        let response_idx = usize::try_from(ctx.turn_id).unwrap_or(0) % STOP_RESPONSES.len();
        let response_text = STOP_RESPONSES[response_idx];
        synthesize_and_send(
            response_text,
            session,
            out,
            state,
            ctx,
            cancel,
            cancel_notify,
        )
        .await;
        counter!("rustedvino_realtime_turns_total", "result" => "barge_in").increment(1);
        send_event(
            out,
            &ServerEvent::Stopped {
                turn_id: ctx.turn_id,
            },
        )
        .await;
        return (String::new(), String::new());
    }

    send_event(
        out,
        &ServerEvent::Transcript {
            text: &transcript,
            final_: true,
        },
    )
    .await;

    if cancel.load(Ordering::Acquire) {
        ctx.cancel_stage = "stt";
        counter!("rustedvino_realtime_turns_total", "result" => "cancelled").increment(1);
        send_event(
            out,
            &ServerEvent::Done {
                turn_id: ctx.turn_id,
            },
        )
        .await;
        return (transcript, String::new());
    }

    // ── LLM + TTS ─────────────────────────────────────────────────────────────
    let assistant_text = match run_llm_tts(
        &transcript,
        session,
        out,
        cancel,
        cancel_notify,
        state,
        ctx,
        embed_handle,
    )
    .await
    {
        Ok(text) => text,
        Err(e) => {
            counter!("rustedvino_realtime_turns_total", "result" => "llm_error").increment(1);
            tracing::warn!(
                conn_id = ctx.conn_id,
                turn_id = ctx.turn_id,
                err = %e,
                "realtime: llm error"
            );
            send_event(
                out,
                &ServerEvent::Error {
                    stage: "llm",
                    message: &e.to_string(),
                },
            )
            .await;
            send_event(
                out,
                &ServerEvent::Done {
                    turn_id: ctx.turn_id,
                },
            )
            .await;
            return (transcript, String::new());
        }
    };

    (transcript, assistant_text)
}

async fn run_stt(
    wav_bytes: Vec<u8>,
    session: &Session,
    state: &AppState,
) -> anyhow::Result<String> {
    let Some(mm) = state.model_manager.as_ref() else {
        return Ok("test transcription".to_owned()); // no-GPU test path
    };
    let model = session
        .stt_model
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("no STT model configured — send a config event first"))?;
    let handle = mm
        .get_stt_handle(model)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let t = handle
        .transcribe(wav_bytes, session.language.clone(), false)
        .await?;
    Ok(t.text)
}

/// Drive the LLM token stream through the think filter and sentence splitter,
/// synthesising each complete sentence via TTS and streaming PCM16 to the client.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
async fn run_llm_tts(
    user_text: &str,
    session: &Session,
    out: &OutTx,
    cancel: &AtomicBool,
    cancel_notify: &Notify,
    state: &AppState,
    ctx: &mut TurnCtx,
    embed_handle: Option<&crate::embed_engine::EmbeddingHandle>,
) -> anyhow::Result<String> {
    let Some(mm) = state.model_manager.as_ref() else {
        // No-GPU test path — emit one canned text event and done.
        let mock = "This is a test response from the realtime pipeline.";
        send_event(out, &ServerEvent::Text { content: mock }).await;
        send_event(
            out,
            &ServerEvent::Done {
                turn_id: ctx.turn_id,
            },
        )
        .await;
        return Ok(mock.to_owned());
    };

    let llm_model = session
        .llm_model
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("no LLM model configured — send a config event first"))?;

    let chat_ctx = mm
        .get_chat_context(llm_model)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    // Retrieve relevant past context via embedding similarity (if available).
    let memories: Vec<&ClientHistoryEntry> = if let Some(handle) = embed_handle {
        if session.history_entries.is_empty() {
            Vec::new()
        } else {
            retrieve_memories(user_text, &session.history_entries, handle).await
        }
    } else {
        Vec::new()
    };

    let policy = VoicePolicy::from_parser(chat_ctx.reasoning_parser.as_ref());

    // Build full conversation: optional system prompt + history + this user turn.
    // When memories are available, prepend a [Relevant past context] block to
    // the system prompt content so the LLM can reference prior exchanges.
    let mut messages: Vec<RtMessage> = Vec::new();
    let effective_system: Option<String> = if memories.is_empty() {
        session.system_prompt.clone()
    } else {
        let memory_block = {
            use std::fmt::Write as _;
            memories.iter().fold(String::new(), |mut acc, m| {
                // Field accesses in format args don't trigger uninlined_format_args.
                let _ = write!(acc, "User: {}\nAssistant: {}\n\n", m.user, m.assistant);
                acc
            })
        };
        let prefix = format!("[Relevant past context]\n{memory_block}");
        Some(match &session.system_prompt {
            Some(sp) => format!("{prefix}{sp}"),
            None => prefix,
        })
    };
    // gpt-oss: route system_prompt to model_identity so it overrides the
    // built-in "You are ChatGPT…" block instead of being injected as a
    // developer message alongside it. All other families: role=system message.
    let model_identity = if policy.system_as_identity {
        effective_system.as_deref()
    } else {
        if let Some(ref sp) = effective_system {
            messages.push(RtMessage {
                role: "system".to_owned(),
                content: sp.clone(),
            });
        }
        None
    };
    messages.extend(session.history.iter().cloned());
    messages.push(RtMessage {
        role: "user".to_owned(),
        content: user_text.to_owned(),
    });

    let (token_tx, token_rx) = stream_channel();
    let gen_params = GenParams {
        max_new_tokens: session.max_tokens.map_or(VOICE_MAX_TOKENS, |n| n as usize),
        temperature: None,
        top_p: None,
        presence_penalty: None,
        frequency_penalty: None,
        // Same safety net chat/completions get — a realtime voice turn has no
        // per-request override surface, so it always gets the default.
        repetition_penalty: Some(crate::handlers::chat::DEFAULT_REPETITION_PENALTY),
        seed: None,
        stop: vec![],
        json_schema: None,
        // No per-request override surface on a voice turn; engine default.
        top_k: None,
    };

    ctx.llm_start = Some(Instant::now());
    match chat_ctx.handle {
        EngineHandleKind::TextGen(engine) => {
            let mut prompt = crate::prompt_builder::build_prompt(
                &messages,
                None, // no tools for voice
                &chat_ctx.template,
                &chat_ctx.eos_token,
                &chat_ctx.bos_token,
                Some(false), // disable thinking — chain-of-thought silence is fatal for voice UX
                model_identity,
            )?;
            if let Some(pf) = policy.channel_prefill {
                prompt.push_str(pf);
            }
            match engine
                .add_request(
                    next_request_id(),
                    prompt,
                    None,
                    gen_params,
                    token_tx,
                    Instant::now(),
                )
                .await
            {
                SubmitResult::Submitted => {}
                SubmitResult::AtCapacity => return Err(anyhow::anyhow!("LLM engine at capacity")),
                SubmitResult::EngineDead => return Err(anyhow::anyhow!("LLM engine unavailable")),
            }
        }
        // VLM path: text-only voice chat through a Vision-kind model — no image
        // ingestion over the WS protocol (no frame type for it today). No
        // VLM in this codebase's catalog sets `reasoning_parser: gpt_oss`, so
        // `policy.system_as_identity`/`channel_prefill` never fire here in
        // practice — `messages` (built above, unconditionally) already carries
        // the system prompt as a plain role=system entry, which is exactly
        // what a VLM's own internal chat templating expects. If a
        // gpt-oss-flavoured VLM is ever added, this arm would silently drop
        // the model_identity override — not handled speculatively.
        EngineHandleKind::Vision(vlm) => {
            let vlm_messages: Vec<serde_json::Value> = messages
                .iter()
                .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
                .collect();
            // L0 gate — mandatory, not optional. Reused as-is from the HTTP
            // Vision arm: this is the exact check standing between an
            // oversized prompt and a confirmed live CL_OUT_OF_RESOURCES
            // GPU-context-poisoning crash (see `gate_vlm_prompt`'s doc
            // comment in handlers/chat.rs). The boxed HTTP Response in its
            // Err isn't unpacked — wrong shape for a WS error — just mapped
            // to a plain message; the caller already turns any Err here into
            // ServerEvent::Error{stage:"llm"} + Done.
            // KNOWN TEST GAP (Fable review, 2026-08-04): no test in this file's
            // `mod tests` can currently observe this call firing or being
            // deleted — `run_llm_tts_serves_a_vision_kind_llm` loads its mock
            // model with no on-disk `config.json`, so `ModelManager` computes
            // `max_prompt_tokens = 0` (`compute_max_prompt_tokens` returns 0
            // whenever `read_kv_bytes_per_token_f16` finds nothing to parse),
            // which self-disables the gate before it ever reaches here. Closing
            // this needs either a real on-disk config.json test fixture under a
            // temp `models_dir`, or a `ModelManager` test seam to set
            // `max_prompt_tokens` directly — bigger than this pass; `gate_vlm_
            // prompt` itself has 4 direct unit tests in `chat.rs` (byte
            // pre-check, over-limit, within-limit, fail-open) that don't share
            // this gap.
            let gate_text: String = messages
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            if crate::handlers::chat::gate_vlm_prompt(&vlm, &gate_text, chat_ctx.max_prompt_tokens)
                .await
                .is_err()
            {
                return Err(anyhow::anyhow!(
                    "prompt too long for model '{llm_model}' — exceeds its context window"
                ));
            }
            match vlm
                .generate(
                    vlm_messages,
                    vec![],
                    None,
                    // No extra template context on the realtime voice path —
                    // it has no `enable_thinking` request field of its own.
                    None,
                    gen_params,
                    token_tx,
                    Instant::now(),
                )
                .await
            {
                SubmitResult::Submitted => {}
                SubmitResult::AtCapacity => return Err(anyhow::anyhow!("LLM engine at capacity")),
                SubmitResult::EngineDead => return Err(anyhow::anyhow!("LLM engine unavailable")),
            }
        }
        other => {
            return Err(anyhow::anyhow!(
                "model '{llm_model}' is {:?} — realtime needs a text-gen or vision model",
                other.kind()
            ));
        }
    }
    tracing::info!(
        conn_id = ctx.conn_id,
        turn_id = ctx.turn_id,
        llm_model,
        "realtime: llm request submitted"
    );

    drain_voice_tokens(
        token_rx,
        cancel,
        cancel_notify,
        out,
        session,
        state,
        ctx,
        &policy,
    )
    .await
}

// gpt-oss channel names that may appear as the first plain-text token when the
// model regenerates its own channel prefix despite the pre-fill. Special-token
// wrappers (<|channel|> / <|message|>) are stripped by the OV engine; only the
// channel name word leaks through. Max length: "commentary" = 10.
const GPT_OSS_CHANNEL_NAMES: &[&str] = &["final", "analysis", "commentary"];

/// Strip a leading gpt-oss channel name ("final", "analysis", "commentary") from
/// `text` if present, and return the cleaned string with a flag indicating whether
/// stripping occurred.
///
/// Matching is case-insensitive and word-boundary-guarded: "Finally" and "finally"
/// are both caught, but "finally" as a prefix of "finally," is stripped while
/// "finally" inside "finalize" is not (next char must be non-alphabetic or EOL).
fn strip_gpt_oss_channel_prefix(text: &str) -> (&str, bool) {
    let lower = text.to_ascii_lowercase();
    for &name in GPT_OSS_CHANNEL_NAMES {
        if lower.starts_with(name) {
            let rest = &text[name.len()..];
            // Strip when the channel name is followed by non-alpha (e.g. "final answer")
            // or an uppercase letter (model concatenated channel name directly with the
            // first sentence word: "finalI'm" → strip to "I'm").
            // A lowercase continuation means a natural word: "finally" → 'l' → keep.
            let should_strip = rest
                .chars()
                .next()
                .is_none_or(|c| !c.is_alphabetic() || c.is_uppercase());
            if should_strip {
                return (rest, true);
            }
        }
    }
    (text, false)
}

/// Returns `true` when the per-turn sentence cap has been reached.
fn sentence_cap_reached(session: &Session, ctx: &TurnCtx) -> bool {
    session
        .max_sentences
        .is_some_and(|max| ctx.tts_sentences >= max as usize)
}

/// Graceful early-stop: emit telemetry + `Done` and return `full_text`.
/// Called when the sentence cap fires mid-stream; dropping `token_rx`
/// (when `drain_voice_tokens` returns) signals the LLM engine to cancel.
async fn early_done(out: &OutTx, ctx: &mut TurnCtx, full_text: String) -> anyhow::Result<String> {
    settle_tts_timing(ctx);
    emit_turn_telemetry(ctx);
    counter!("rustedvino_realtime_turns_total", "result" => "ok").increment(1);
    send_event(
        out,
        &ServerEvent::Done {
            turn_id: ctx.turn_id,
        },
    )
    .await;
    Ok(full_text)
}

/// Drain the LLM token stream, apply thinking-mode filter, split into sentences
/// for TTS, and emit text events. Returns the full assistant text.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
async fn drain_voice_tokens(
    mut token_rx: crate::streaming::TokenReceiver,
    cancel: &AtomicBool,
    cancel_notify: &Notify,
    out: &OutTx,
    session: &Session,
    state: &AppState,
    ctx: &mut TurnCtx,
    policy: &VoicePolicy,
) -> anyhow::Result<String> {
    let mut splitter = SentenceSplitter::new();
    let mut think_filter = ThinkFilter::default();
    let mut full_text = String::new();
    // Buffer the first ≤10 content chars to detect and strip a leaked channel
    // name prefix (only active when the policy requests it, i.e. gpt-oss).
    // Max channel name = "commentary" (10 chars); buffer until > 10.
    let mut prefix_buf: Option<String> = if policy.strip_output_prefix {
        Some(String::new())
    } else {
        None
    };

    loop {
        if cancel.load(Ordering::Acquire) {
            // Dropping token_rx closes the channel → CB engine cancels generation.
            ctx.cancel_stage = if ctx.tts_sentences > 0 { "tts" } else { "llm" };
            tracing::debug!(
                conn_id = ctx.conn_id,
                turn_id = ctx.turn_id,
                stage = ctx.cancel_stage,
                "realtime: cancel took effect — drain stopped"
            );
            settle_tts_timing(ctx);
            emit_turn_telemetry(ctx);
            counter!("rustedvino_realtime_turns_total", "result" => "cancelled").increment(1);
            send_event(
                out,
                &ServerEvent::Done {
                    turn_id: ctx.turn_id,
                },
            )
            .await;
            break;
        }
        match token_rx.recv().await {
            // `new_tokens` (real per-event token count) is not threaded into
            // `ctx.llm_tokens`/`ctx.think_tokens` below: those already count
            // ThinkFilter *pieces*, not raw tokens (one delta can split into
            // several pieces across a `<think>` boundary), so summing
            // `new_tokens` per piece would double-count. This telemetry-only
            // counter is a pre-existing approximation, not an OpenAI `usage`
            // field — left as-is; unlike `usage.completion_tokens`, it does
            // not need the project's internal engineering log 2026-07-19 fix applied here.
            Some(StreamEvent::Token(delta, _new_tokens)) => {
                for piece in think_filter.process(&delta) {
                    match piece {
                        ThinkPiece::Content(text) => {
                            // For gpt-oss: buffer until prefix determination is settled.
                            let text = if let Some(ref mut buf) = prefix_buf {
                                buf.push_str(&text);
                                if buf.len() <= 10 {
                                    continue; // still accumulating
                                }
                                let resolved = buf.clone();
                                prefix_buf = None;
                                let (clean, did_strip) = strip_gpt_oss_channel_prefix(&resolved);
                                if did_strip {
                                    tracing::debug!(
                                        conn_id = ctx.conn_id,
                                        "realtime: stripped gpt-oss channel prefix"
                                    );
                                }
                                clean.to_owned()
                            } else {
                                text
                            };
                            // Record TTFT on first content token.
                            if ctx.llm_tokens == 0 {
                                ctx.llm_ttft_ms = ctx.llm_start.map_or(0, |t| to_ms(t.elapsed()));
                            }
                            ctx.llm_tokens += 1;
                            send_event(out, &ServerEvent::Text { content: &text }).await;
                            full_text.push_str(&text);
                            if let Some(sentence) = splitter.push(&text)
                                && !cancel.load(Ordering::Acquire)
                            {
                                synthesize_and_send(
                                    &sentence,
                                    session,
                                    out,
                                    state,
                                    ctx,
                                    cancel,
                                    cancel_notify,
                                )
                                .await;
                                if sentence_cap_reached(session, ctx) {
                                    return early_done(out, ctx, full_text).await;
                                }
                            }
                        }
                        ThinkPiece::Reasoning(_) => {
                            // Count filtered think tokens but do not emit them.
                            ctx.think_tokens += 1;
                        }
                    }
                }
            }
            Some(StreamEvent::Done(_)) => {
                // Flush any buffered prefix — short responses may never exceed 10 chars.
                if let Some(buf) = prefix_buf.take() {
                    let (clean, _) = strip_gpt_oss_channel_prefix(&buf);
                    if !clean.is_empty() {
                        full_text.push_str(clean);
                        if let Some(sentence) = splitter.push(clean)
                            && !cancel.load(Ordering::Acquire)
                        {
                            synthesize_and_send(
                                &sentence,
                                session,
                                out,
                                state,
                                ctx,
                                cancel,
                                cancel_notify,
                            )
                            .await;
                            if sentence_cap_reached(session, ctx) {
                                return early_done(out, ctx, full_text).await;
                            }
                        }
                    }
                }
                for piece in think_filter.flush() {
                    match piece {
                        ThinkPiece::Content(text) => {
                            full_text.push_str(&text);
                            if let Some(sentence) = splitter.push(&text)
                                && !cancel.load(Ordering::Acquire)
                            {
                                synthesize_and_send(
                                    &sentence,
                                    session,
                                    out,
                                    state,
                                    ctx,
                                    cancel,
                                    cancel_notify,
                                )
                                .await;
                                if sentence_cap_reached(session, ctx) {
                                    return early_done(out, ctx, full_text).await;
                                }
                            }
                        }
                        ThinkPiece::Reasoning(_) => {
                            ctx.think_tokens += 1;
                        }
                    }
                }
                if let Some(sentence) = splitter.flush()
                    && !cancel.load(Ordering::Acquire)
                {
                    synthesize_and_send(&sentence, session, out, state, ctx, cancel, cancel_notify)
                        .await;
                    if sentence_cap_reached(session, ctx) {
                        return early_done(out, ctx, full_text).await;
                    }
                }
                // Finalise timing fields before emitting telemetry.
                settle_tts_timing(ctx);
                emit_turn_telemetry(ctx);
                if let Some(t) = ctx.llm_start {
                    histogram!("rustedvino_realtime_llm_duration_seconds")
                        .record(t.elapsed().as_secs_f64());
                }
                counter!("rustedvino_realtime_turns_total", "result" => "ok").increment(1);
                send_event(
                    out,
                    &ServerEvent::Done {
                        turn_id: ctx.turn_id,
                    },
                )
                .await;
                break;
            }
            Some(StreamEvent::PromptTokens(_) | StreamEvent::CompletionTokens(_)) => {}
            Some(StreamEvent::Error(e)) => return Err(anyhow::anyhow!("{e}")),
            None => break,
        }
    }
    Ok(full_text)
}

/// Compute `tts_first_ms` (`llm_start` → first audio) and `tts_total_ms`
/// (`llm_start` → now). Must be called once before `emit_turn_telemetry`.
fn settle_tts_timing(ctx: &mut TurnCtx) {
    ctx.tts_first_ms = ctx
        .llm_start
        .zip(ctx.tts_first_at)
        .and_then(|(llm, tts)| tts.checked_duration_since(llm))
        .map_or(0, to_ms);
    ctx.tts_total_ms = ctx.llm_start.map_or(0, |t| to_ms(t.elapsed()));
}

/// Emit the structured per-turn telemetry log. Fired once per turn at Done or
/// Cancel. When using a JSON tracing subscriber this produces a JSONL record
/// that can be joined with the client log on `(conn_id, turn_id)`.
fn emit_turn_telemetry(ctx: &TurnCtx) {
    tracing::info!(
        conn_id = ctx.conn_id,
        turn_id = ctx.turn_id,
        audio_samples = ctx.audio_samples,
        utterance_duration_ms = ctx.utterance_duration_ms,
        utterance_energy = ctx.utterance_energy,
        transcript_empty = ctx.transcript_empty,
        whisper_ms = ctx.whisper_ms,
        llm_ttft_ms = ctx.llm_ttft_ms,
        llm_tokens = ctx.llm_tokens,
        think_tokens = ctx.think_tokens,
        tts_sentences = ctx.tts_sentences,
        tts_first_ms = ctx.tts_first_ms,
        tts_total_ms = ctx.tts_total_ms,
        audio_sent_samples = ctx.audio_sent_samples,
        cancel_stage = ctx.cancel_stage,
        "realtime: turn complete"
    );
}

/// Prepare `text` for speech synthesis: expand digits/symbols into spoken
/// words (`crate::tts_normalize`), then strip markdown decoration and other
/// characters that confuse the TTS phonemizer (`*`, `_`, `#`, `` ` ``, `~`,
/// `|`) or emoji. The LLM occasionally leaks these into voice output despite
/// the system prompt. The two failure modes differ: an unknown markdown
/// character is silently skipped, garbling the surrounding word, while an
/// emoji has been observed to make the engine stall for a long pause rather
/// than skip cleanly (`🛒`, reported live) — either way the fix is the same,
/// strip it before synthesis ever sees it.
///
/// Normalization runs *before* the strip, while punctuation that matters to
/// it (decimal points, `&`, `%`, ...) is still intact — see
/// the project's internal engineering log. `lang` is the session's
/// configured language (`Session.language`), reused here rather than adding
/// a second language field.
fn sanitize_for_tts(text: &str, lang: Option<&str>) -> String {
    let normalized = crate::tts_normalize::normalize_for_speech(text, lang);
    let cleaned: String = normalized
        .chars()
        .filter(|c| !matches!(c, '*' | '_' | '#' | '`' | '~' | '|') && !is_emoji(*c))
        .collect();
    // Collapse any whitespace runs left by the removal.
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// True for characters in the common emoji/pictograph Unicode blocks.
/// Range-based rather than a dependency — this only needs to catch the
/// LLM's actual emoji usage, not achieve exhaustive Unicode emoji-property
/// coverage (skin-tone modifiers, ZWJ sequences, etc. are out of scope;
/// they'd still leave the base emoji character stripped, just possibly a
/// stray modifier byte behind — acceptable, not seen in practice yet).
fn is_emoji(c: char) -> bool {
    matches!(c as u32,
        0x2600..=0x27BF     // Misc Symbols, Dingbats (☀ ✂ ✔ ...)
        | 0x1F300..=0x1F5FF // Misc Symbols and Pictographs
        | 0x1F600..=0x1F64F // Emoticons
        | 0x1F680..=0x1F6FF // Transport and Map Symbols (includes 🛒 U+1F6D2)
        | 0x1F900..=0x1F9FF // Supplemental Symbols and Pictographs
        | 0x1FA70..=0x1FAFF // Symbols and Pictographs Extended-A
        | 0x1F1E6..=0x1F1FF // Regional indicator letters (flag emoji)
        | 0xFE0F // Variation Selector-16 (forces emoji presentation)
    )
}

/// Synthesise `sentence` via the configured TTS model and stream PCM16 bytes
/// to the client. Errors are logged and the sentence is silently dropped —
/// the LLM text stream is already on the wire, so the user still sees the text.
async fn synthesize_and_send(
    sentence: &str,
    session: &Session,
    out: &OutTx,
    state: &AppState,
    ctx: &mut TurnCtx,
    cancel: &AtomicBool,
    cancel_notify: &Notify,
) {
    let Some(mm) = state.model_manager.as_ref() else {
        return;
    };
    let Some(tts_model) = session.tts_model.as_deref() else {
        return;
    };

    let handle = match mm.get_tts_handle(tts_model) {
        Ok(h) => h,
        Err(e) => {
            counter!("rustedvino_realtime_tts_sentences_total", "result" => "dropped").increment(1);
            tracing::warn!(
                conn_id = ctx.conn_id,
                err = %e,
                "realtime: TTS handle unavailable — sentence dropped"
            );
            return;
        }
    };

    let tts_sample_rate = handle.sample_rate();
    let clean = sanitize_for_tts(sentence, session.language.as_deref());
    if clean.is_empty() {
        return;
    }
    // NEW-H(B): subscribe to the cancel notification BEFORE checking the flag
    // so a notify_waiters() that fires between subscription and select! is not
    // missed. The explicit load() catches a cancel that was already set before
    // we subscribed (notify_waiters does not store a permit).
    let notified = cancel_notify.notified();
    if cancel.load(Ordering::Acquire) {
        return;
    }
    let tts_start = Instant::now();
    let synth_result = tokio::select! {
        biased;
        () = notified => return,
        r = handle.synthesize(clean, session.tts_voice.clone(), 1.0) => r,
    };
    match synth_result {
        Ok(samples) => {
            // Resample to the wire rate (16 kHz) when the TTS backend outputs a
            // different rate. Kokoro outputs 24 kHz; SpeechT5 outputs 16 kHz.
            let wire_samples = if tts_sample_rate == PCM_SAMPLE_RATE {
                samples
            } else {
                match resample_to_16k(&samples, tts_sample_rate) {
                    Ok(r) => r,
                    Err(e) => {
                        counter!("rustedvino_realtime_tts_sentences_total", "result" => "dropped")
                            .increment(1);
                        tracing::warn!(
                            conn_id = ctx.conn_id,
                            err = %e,
                            src_rate = tts_sample_rate,
                            "realtime: TTS resample failed — sentence dropped"
                        );
                        return;
                    }
                }
            };
            let tts_ms = to_ms(tts_start.elapsed());
            ctx.tts_sentences += 1;
            ctx.audio_sent_samples += wire_samples.len();
            // Record first-audio timestamp (for tts_first_ms in telemetry).
            if ctx.tts_first_at.is_none() {
                ctx.tts_first_at = Some(Instant::now());
            }
            tracing::debug!(
                conn_id = ctx.conn_id,
                turn_id = ctx.turn_id,
                sentence_len = sentence.len(),
                tts_ms,
                "realtime: tts sentence sent"
            );
            counter!("rustedvino_realtime_tts_sentences_total", "result" => "ok").increment(1);
            send_audio(out, f32_to_pcm16_bytes(&wire_samples)).await;
        }
        Err(e) => {
            counter!("rustedvino_realtime_tts_sentences_total", "result" => "dropped").increment(1);
            tracing::warn!(
                conn_id = ctx.conn_id,
                err = %e,
                "realtime: TTS synthesis failed — sentence dropped"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Request / connection ID counters
// ─────────────────────────────────────────────────────────────────────────────

static REQUEST_ID: AtomicU64 = AtomicU64::new(0);
/// Separate from `REQUEST_ID`: each WebSocket connection gets a stable ID for
/// the lifetime of the session — not consumed per-LLM-request.
static CONN_ID: AtomicU64 = AtomicU64::new(0);

fn next_request_id() -> u64 {
    REQUEST_ID.fetch_add(1, Ordering::Relaxed)
}

fn next_conn_id() -> u64 {
    CONN_ID.fetch_add(1, Ordering::Relaxed)
}

// ─────────────────────────────────────────────────────────────────────────────
// WebSocket handler
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a `Config` event to the session, updating only the provided fields.
#[allow(clippy::too_many_arguments)]
async fn apply_client_config(
    session: &Arc<tokio::sync::Mutex<Session>>,
    language: Option<String>,
    stt_model: Option<String>,
    llm_model: Option<String>,
    tts_model: Option<String>,
    tts_voice: Option<String>,
    system_prompt: Option<String>,
    barge_in_phrase: Option<String>,
    embed_model: Option<String>,
    history: Option<Vec<ClientHistoryEntry>>,
    max_tokens: Option<u32>,
    max_sentences: Option<u32>,
) {
    let mut s = session.lock().await;
    if let Some(v) = language {
        s.language = Some(v);
    }
    if let Some(v) = stt_model {
        s.stt_model = Some(v);
    }
    if let Some(v) = llm_model {
        s.llm_model = Some(v);
    }
    if let Some(v) = tts_model {
        s.tts_model = Some(v);
    }
    if let Some(v) = tts_voice {
        s.tts_voice = Some(v);
    }
    if let Some(v) = system_prompt {
        // Empty string sent by client clears the prompt.
        s.system_prompt = if v.is_empty() { None } else { Some(v) };
    }
    if let Some(v) = barge_in_phrase {
        s.barge_in_phrase = if v.is_empty() { None } else { Some(v) };
    }
    if let Some(v) = embed_model {
        s.embed_model = Some(v);
    }
    if let Some(v) = max_tokens {
        s.max_tokens = Some(v);
    }
    if let Some(v) = max_sentences {
        s.max_sentences = Some(v);
    }
    if let Some(mut entries) = history {
        // Trim to cap — keep the most recent entries.
        if entries.len() > MAX_CLIENT_HISTORY {
            entries.drain(..entries.len() - MAX_CLIENT_HISTORY);
        }
        // Rebuild the LLM context window from the most recent turns.
        let start = entries.len().saturating_sub(MAX_HISTORY_TURNS);
        s.history = entries[start..]
            .iter()
            .flat_map(|e| {
                [
                    RtMessage {
                        role: "user".to_owned(),
                        content: e.user.clone(),
                    },
                    RtMessage {
                        role: "assistant".to_owned(),
                        content: e.assistant.clone(),
                    },
                ]
            })
            .collect();
        s.history_entries = entries;
    }
}

/// Handle an `InjectHistory` batch from the client.
///
/// Appends entries to the retrieval pool (`history_entries`), embeds any that
/// lack a vector (if an embed handle is available), and fires a `TurnHistory`
/// event per entry so the client can store the computed vectors locally.
/// Entries are NOT added to the rolling LLM context window — they are
/// available for similarity retrieval only.
async fn handle_inject_history(
    mut entries: Vec<ClientHistoryEntry>,
    session: &Arc<tokio::sync::Mutex<Session>>,
    session_snap: &Arc<RwLock<SessionSnapshot>>,
    out: &OutTx,
    embed_handle: Option<&crate::embed_engine::EmbeddingHandle>,
    conn_id: u64,
) {
    // Embed entries that lack vectors (best-effort — no embed handle → raw text only).
    if let Some(handle) = embed_handle {
        let (texts, indices): (Vec<String>, Vec<usize>) = entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.embedding.is_none())
            .map(|(i, e)| (format!("{}\n{}", e.user, e.assistant), i))
            .unzip();
        if !texts.is_empty()
            && let Ok(output) = handle.embed(texts).await
        {
            for (&idx, vec) in indices.iter().zip(output.vectors.iter()) {
                if let Some(e) = entries.get_mut(idx) {
                    e.embedding = Some(vec.clone());
                    e.embedding_model = Some(handle.model_id().to_owned());
                }
            }
        }
    }

    // Append to session pool, respecting the cap (drop oldest on overflow).
    let new_entries = {
        let mut guard = session.lock().await;
        for entry in entries {
            guard.history_entries.push(entry);
        }
        if guard.history_entries.len() > MAX_CLIENT_HISTORY {
            let overflow = guard.history_entries.len() - MAX_CLIENT_HISTORY;
            guard.history_entries.drain(..overflow);
        }
        guard.history_entries.clone()
    };

    // Mirror into admin snapshot.
    if let Ok(mut snap) = session_snap.write() {
        snap.history_entries.clone_from(&new_entries);
    }

    // Fire TurnHistory events so the client can persist the computed vectors.
    let injected_count = new_entries.len();
    for entry in &new_entries[new_entries.len().saturating_sub(injected_count)..] {
        send_event(
            out,
            &ServerEvent::TurnHistory {
                turn_id: None,
                user: &entry.user,
                assistant: &entry.assistant,
                embedding: entry.embedding.clone(),
                timestamp: entry.timestamp,
                embedding_model: entry.embedding_model.as_deref(),
                stt_model: entry.stt_model.as_deref(),
                llm_model: entry.llm_model.as_deref(),
            },
        )
        .await;
    }

    tracing::info!(
        conn_id,
        count = injected_count,
        "realtime: inject_history applied"
    );
}

/// Handle a `ReloadHistory` command from the client.
///
/// Clears the retrieval pool (`history_entries`) entirely, then processes the
/// provided entries exactly as `handle_inject_history` does (embed missing
/// vectors, append). Responds with a single `HistoryReloaded` event — no
/// per-entry `TurnHistory` echo. The rolling LLM context window is unchanged.
async fn handle_reload_history(
    mut entries: Vec<ClientHistoryEntry>,
    session: &Arc<tokio::sync::Mutex<Session>>,
    session_snap: &Arc<RwLock<SessionSnapshot>>,
    out: &OutTx,
    embed_handle: Option<&crate::embed_engine::EmbeddingHandle>,
    conn_id: u64,
) {
    // Embed entries that lack vectors (best-effort — no embed handle → raw text only).
    if let Some(handle) = embed_handle {
        let (texts, indices): (Vec<String>, Vec<usize>) = entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.embedding.is_none())
            .map(|(i, e)| (format!("{}\n{}", e.user, e.assistant), i))
            .unzip();
        if !texts.is_empty()
            && let Ok(output) = handle.embed(texts).await
        {
            for (&idx, vec) in indices.iter().zip(output.vectors.iter()) {
                if let Some(e) = entries.get_mut(idx) {
                    e.embedding = Some(vec.clone());
                    e.embedding_model = Some(handle.model_id().to_owned());
                }
            }
        }
    }

    // Clear pool and replace with provided entries, respecting the cap.
    let new_entries = {
        let mut guard = session.lock().await;
        guard.history_entries.clear();
        for entry in entries {
            guard.history_entries.push(entry);
        }
        if guard.history_entries.len() > MAX_CLIENT_HISTORY {
            let overflow = guard.history_entries.len() - MAX_CLIENT_HISTORY;
            guard.history_entries.drain(..overflow);
        }
        guard.history_entries.clone()
    };

    let count = new_entries.len();

    // Mirror into admin snapshot.
    if let Ok(mut snap) = session_snap.write() {
        snap.history_entries.clone_from(&new_entries);
    }

    send_event(out, &ServerEvent::HistoryReloaded { count }).await;
    tracing::info!(conn_id, count, "realtime: reload_history applied");
}

/// WebSocket receiver task: reads frames, runs VAD, routes Config/Cancel events.
/// Dropping `audio_tx` at exit signals the pipeline to shut down.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
async fn run_receiver(
    // Generic over the frame stream (not the concrete `SplitStream<WebSocket>`)
    // so unit tests can drive it with a mock stream. The real call site passes a
    // `SplitStream<WebSocket>`, which satisfies these bounds.
    mut stream: impl futures_util::Stream<Item = Result<WsMsg, axum::Error>> + Unpin + Send,
    out_tx: OutTx,
    cancel: Arc<AtomicBool>,
    cancel_notify: Arc<Notify>,
    session: Arc<tokio::sync::Mutex<Session>>,
    audio_tx: mpsc::Sender<UtteranceCtx>,
    inject_tx: mpsc::Sender<Vec<ClientHistoryEntry>>,
    reload_tx: mpsc::Sender<Vec<ClientHistoryEntry>>,
    conn_id: u64,
    session_snap: Arc<RwLock<SessionSnapshot>>,
    shutdown: tokio_util::sync::CancellationToken,
    // D5/D6/D8 (the project's internal engineering log): needed
    // to resolve/arbitrate the LLM slot and update the realtime serving set
    // on `Config`/`RequestModelLoad`. `model_manager` mirrors `AppState`'s
    // own `Option` — `None` in mock/test mode leaves Config resolution as a
    // no-op, unchanged from pre-D5 behavior.
    model_manager: Option<Arc<crate::model_manager::ModelManager>>,
    voice_pin: Arc<crate::voice_pin::VoicePinManager>,
) {
    let mut vad = EnergyVad::new();
    let mut speech_start: Option<Instant> = None;
    // NEW-H(A): suppress the per-drop "pipeline busy" Error storm. During a barge-in
    // the cap-bounded audio channel can overflow repeatedly; emit at most one client
    // Error per backlog episode and count the rest at debug. The latch resets when an
    // utterance is accepted again — a drained pipeline starts a fresh episode.
    let mut busy_notified = false;
    let mut dropped_busy = 0u64;
    loop {
        // `biased` so a pending shutdown wins over a ready frame — teardown is
        // never starved by an always-streaming client.
        let msg = tokio::select! {
            biased;
            () = shutdown.cancelled() => break,
            m = stream.next() => match m {
                Some(Ok(m)) => m,
                Some(Err(e)) => {
                    tracing::warn!(conn_id, err = %e, "realtime: ws recv error");
                    break;
                }
                None => break,
            },
        };
        match msg {
            WsMsg::Binary(bytes) => {
                for event in vad.push_bytes(&bytes) {
                    match event {
                        VadEvent::SpeechStarted => {
                            tracing::debug!(conn_id, "realtime: speech started");
                            speech_start = Some(Instant::now());
                            send_event(&out_tx, &ServerEvent::SpeechStarted).await;
                        }
                        VadEvent::SpeechEnded(audio) => {
                            let start = speech_start.take().unwrap_or_else(Instant::now);
                            let utterance_duration_ms = to_ms(start.elapsed());
                            let energy = rms(&audio);
                            tracing::debug!(
                                conn_id,
                                utterance_duration_ms,
                                "realtime: speech ended"
                            );
                            send_event(&out_tx, &ServerEvent::SpeechStopped).await;
                            // The pipeline resets the cancel flag at utterance start
                            // — the receiver only ever sets it to true.
                            // try_send keeps this task live for Cancel events even
                            // when the pipeline is backlogged.
                            let utterance = UtteranceCtx {
                                audio,
                                energy,
                                utterance_duration_ms,
                            };
                            if audio_tx.try_send(utterance).is_err() {
                                dropped_busy += 1;
                                if busy_notified {
                                    tracing::debug!(
                                        conn_id,
                                        dropped_busy,
                                        "realtime: utterance discarded — pipeline busy (suppressed)"
                                    );
                                } else {
                                    busy_notified = true;
                                    send_event(
                                        &out_tx,
                                        &ServerEvent::Error {
                                            stage: "pipeline",
                                            message: "utterance discarded — pipeline busy",
                                        },
                                    )
                                    .await;
                                }
                            } else if busy_notified {
                                // Pipeline accepted again — backlog episode over.
                                tracing::debug!(
                                    conn_id,
                                    dropped_busy,
                                    "realtime: pipeline drained — backlog cleared"
                                );
                                busy_notified = false;
                                dropped_busy = 0;
                            }
                        }
                    }
                }
            }
            WsMsg::Text(text) => match serde_json::from_str::<ClientEvent>(&text) {
                Ok(ClientEvent::Config {
                    language,
                    stt_model,
                    llm_model,
                    tts_model,
                    tts_voice,
                    system_prompt,
                    barge_in_phrase,
                    embed_model,
                    history,
                    max_tokens,
                    max_sentences,
                }) => {
                    // D2: snapshot what this session currently depends on
                    // *before* applying the new config, so the serving-set
                    // update below can deregister exactly what's being
                    // replaced (not everything, not nothing).
                    let old_serving: Vec<String> = {
                        let s = session.lock().await;
                        [
                            s.stt_model.clone(),
                            s.llm_model.clone(),
                            s.tts_model.clone(),
                        ]
                        .into_iter()
                        .flatten()
                        .collect()
                    };

                    apply_client_config(
                        &session,
                        language,
                        stt_model,
                        llm_model,
                        tts_model,
                        tts_voice,
                        system_prompt,
                        barge_in_phrase,
                        embed_model,
                        history,
                        max_tokens,
                        max_sentences,
                    )
                    .await;

                    // D5, generalized to all three slots (RTCC) — "server
                    // shall choose the optimal set of models... client can
                    // ask for more than is on the menu but server decides if
                    // it's possible." Overwrites session.{stt,llm,tts}_model
                    // with the *resolved* ids, not the raw request —
                    // mock/test mode (no model_manager) leaves them as
                    // literally requested, unchanged from pre-D5 behavior.
                    let (requested_stt, requested_llm, requested_tts) = {
                        let s = session.lock().await;
                        (
                            s.stt_model.clone(),
                            s.llm_model.clone(),
                            s.tts_model.clone(),
                        )
                    };
                    let stt_resolution = model_manager.as_ref().map(|mm| {
                        mm.resolve_realtime_slot(RealtimeSlot::Stt, requested_stt.as_deref())
                    });
                    let llm_resolution = model_manager.as_ref().map(|mm| {
                        mm.resolve_realtime_slot(RealtimeSlot::Llm, requested_llm.as_deref())
                    });
                    let tts_resolution = model_manager.as_ref().map(|mm| {
                        mm.resolve_realtime_slot(RealtimeSlot::Tts, requested_tts.as_deref())
                    });
                    {
                        let mut s = session.lock().await;
                        if let Some(ref res) = stt_resolution {
                            s.stt_model = Some(res.model_id.clone());
                        }
                        if let Some(ref res) = llm_resolution {
                            s.llm_model = Some(res.model_id.clone());
                        }
                        if let Some(ref res) = tts_resolution {
                            s.tts_model = Some(res.model_id.clone());
                        }
                    }
                    tracing::debug!(conn_id, "realtime: session config updated");
                    // Mirror updated config into the admin-visible snapshot.
                    let (stt, llm, tts, voice, lang, sys, barge, embed, hist) = {
                        let s = session.lock().await;
                        (
                            s.stt_model.clone(),
                            s.llm_model.clone(),
                            s.tts_model.clone(),
                            s.tts_voice.clone(),
                            s.language.clone(),
                            s.system_prompt.clone(),
                            s.barge_in_phrase.clone(),
                            s.embed_model.clone(),
                            s.history_entries.clone(),
                        )
                    }; // tokio lock released
                    if let Ok(mut snap) = session_snap.write() {
                        snap.stt_model.clone_from(&stt);
                        snap.llm_model.clone_from(&llm);
                        snap.tts_model.clone_from(&tts);
                        snap.tts_voice = voice;
                        snap.language = lang;
                        snap.system_prompt = sys;
                        snap.barge_in_phrase = barge;
                        snap.embed_model = embed;
                        snap.history_entries = hist;
                    }

                    // D2: swap the realtime serving set — deregister what
                    // this session used to depend on, register what it
                    // depends on now (resolved values, post-D5).
                    let new_serving: Vec<String> = [stt.clone(), llm.clone(), tts.clone()]
                        .into_iter()
                        .flatten()
                        .collect();
                    voice_pin.deregister_serving(old_serving.iter().map(String::as_str));
                    voice_pin.register_serving(new_serving.iter().map(String::as_str));

                    if let Some(ref llm_res) = llm_resolution {
                        for resolution in [
                            stt_resolution.as_ref(),
                            Some(llm_res),
                            tts_resolution.as_ref(),
                        ]
                        .into_iter()
                        .flatten()
                        {
                            if resolution.loading
                                && let Some(mm) = model_manager.as_ref()
                            {
                                send_event(
                                    &out_tx,
                                    &ServerEvent::ModelLoading {
                                        model_id: &resolution.model_id,
                                        stage: "started",
                                        detail: None,
                                    },
                                )
                                .await;
                                spawn_model_ready_watcher(
                                    Arc::clone(mm),
                                    resolution.model_id.clone(),
                                    out_tx.clone(),
                                );
                            }
                        }
                        send_event(
                            &out_tx,
                            &ServerEvent::ModelSelection {
                                stt_model: stt.as_deref(),
                                stt_source: stt_resolution.as_ref().map(|r| r.source),
                                stt_reason: stt_resolution.as_ref().and_then(|r| r.reason),
                                llm_model: &llm_res.model_id,
                                llm_source: llm_res.source,
                                llm_reason: llm_res.reason,
                                tts_model: tts.as_deref(),
                                tts_source: tts_resolution.as_ref().map(|r| r.source),
                                tts_reason: tts_resolution.as_ref().and_then(|r| r.reason),
                            },
                        )
                        .await;
                    }
                }
                Ok(ClientEvent::Cancel) => {
                    cancel.store(true, Ordering::Release);
                    cancel_notify.notify_waiters();
                    tracing::debug!(conn_id, "realtime: cancel requested");
                }
                Ok(ClientEvent::InjectHistory { entries }) => {
                    if entries.is_empty() {
                        tracing::debug!(conn_id, "realtime: inject_history — empty batch, ignored");
                    } else {
                        tracing::info!(
                            conn_id,
                            count = entries.len(),
                            "realtime: inject_history received"
                        );
                        // H1: try_send — never block the receiver on a full/closed
                        // channel. Injection is idempotent-ish and infrequent;
                        // dropping a flooded one is fine, wedging the receiver is not.
                        if let Err(e) = inject_tx.try_send(entries) {
                            tracing::debug!(conn_id, err = %e, "realtime: inject_history dropped");
                        }
                    }
                }
                Ok(ClientEvent::ReloadHistory { entries }) => {
                    tracing::info!(
                        conn_id,
                        count = entries.len(),
                        "realtime: reload_history received"
                    );
                    // H1: try_send — see inject_history above.
                    if let Err(e) = reload_tx.try_send(entries) {
                        tracing::debug!(conn_id, err = %e, "realtime: reload_history dropped");
                    }
                }
                Ok(ClientEvent::RequestModelLoad { model_id }) => {
                    // D8, kind-aware since RTCC's slot generalization —
                    // arbitrated identically to a Config-driven request for
                    // whichever slot `model_id` actually belongs to (was
                    // unconditionally the LLM slot pre-RTCC, which happened
                    // to work for STT/TTS ids too via resolve_realtime_llm's
                    // kind-agnostic first two branches — an undocumented
                    // side effect, not a designed path; a rejected STT/TTS
                    // request would previously fall back through the
                    // LLM-shaped substitution branch, which this fixes).
                    // Same resolver chokepoint either way, never a bypass
                    // with authority of its own. Deliberately does NOT
                    // register into the D2 serving set (see the variant's
                    // doc comment) — only an actual `Config` resolution
                    // naming this model does that.
                    let Some(mm) = model_manager.as_ref() else {
                        send_event(
                            &out_tx,
                            &ServerEvent::ModelLoading {
                                model_id: &model_id,
                                stage: "failed",
                                detail: Some("model manager not initialised"),
                            },
                        )
                        .await;
                        continue;
                    };
                    let slot = match mm.known_model_kind(&model_id) {
                        Some(ModelKind::Stt) => RealtimeSlot::Stt,
                        Some(ModelKind::Tts) => RealtimeSlot::Tts,
                        Some(ModelKind::Embedding) => RealtimeSlot::Embed,
                        _ => RealtimeSlot::Llm,
                    };
                    let resolution = mm.resolve_realtime_slot(slot, Some(&model_id));
                    if resolution.model_id == model_id {
                        if resolution.loading {
                            send_event(
                                &out_tx,
                                &ServerEvent::ModelLoading {
                                    model_id: &model_id,
                                    stage: "started",
                                    detail: None,
                                },
                            )
                            .await;
                            spawn_model_ready_watcher(
                                Arc::clone(mm),
                                model_id.clone(),
                                out_tx.clone(),
                            );
                        } else {
                            // Already Ready — nothing to wait for.
                            send_event(
                                &out_tx,
                                &ServerEvent::ModelLoading {
                                    model_id: &model_id,
                                    stage: "ready",
                                    detail: None,
                                },
                            )
                            .await;
                        }
                    } else {
                        // The literal request wasn't grantable — say why
                        // rather than silently loading something else on
                        // the client's behalf (that's Config's job, not
                        // this explicit command's).
                        send_event(
                            &out_tx,
                            &ServerEvent::ModelLoading {
                                model_id: &model_id,
                                stage: "failed",
                                detail: resolution.reason.or(Some(
                                    "would evict a protected model, or is not resident \
                                     and cannot fit",
                                )),
                            },
                        )
                        .await;
                    }
                }
                Err(e) => {
                    send_event(
                        &out_tx,
                        &ServerEvent::Error {
                            stage: "protocol",
                            message: &e.to_string(),
                        },
                    )
                    .await;
                }
            },
            WsMsg::Close(_) => break,
            // Ping/Pong handled automatically by axum; ignore here.
            _ => {}
        }
    }
    // Signal the pipeline to abort any in-flight turn — same path as the
    // explicit Cancel event, but fired automatically on connection drop.
    cancel.store(true, Ordering::Release);
    cancel_notify.notify_waiters();
}

/// C2: bridge an admin kill into an in-flight turn.
///
/// The response loop only polls `kill.cancelled()` *between* turns; during
/// `run_response` it consults only the `cancel` `AtomicBool`. So an admin kill
/// that lands mid-generation would otherwise keep producing tokens/audio until
/// the turn finishes. This task sets `cancel` (the drain loop aborts within
/// ~one token, emitting its `Done` + cancel telemetry) and cancels `shutdown`
/// (waking the receiver — same teardown path as C1).
///
/// It also selects on `shutdown`: a *normal* close (idle / audio-None) calls
/// `shutdown.cancel()` at teardown, which lets this task exit. Without that arm
/// the task would park on `kill.cancelled()` forever — dropping a
/// `CancellationToken` without cancelling never resolves `cancelled()` — leaking
/// one task per non-killed session.
fn spawn_kill_bridge(
    cancel: Arc<AtomicBool>,
    cancel_notify: Arc<Notify>,
    kill: tokio_util::sync::CancellationToken,
    shutdown: tokio_util::sync::CancellationToken,
) {
    tokio::spawn(async move {
        tokio::select! {
            () = kill.cancelled() => {
                cancel.store(true, Ordering::Release);
                cancel_notify.notify_waiters();
                shutdown.cancel();
            }
            () = shutdown.cancelled() => {} // normal teardown — nothing to do
        }
    });
}

/// `GET /v1/realtime` — upgrade to WebSocket, then run the voice pipeline.
pub async fn realtime_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(|socket| realtime_session(socket, state))
}

#[allow(clippy::too_many_lines)]
async fn realtime_session(socket: WebSocket, state: AppState) {
    let conn_id = next_conn_id();
    counter!("rustedvino_realtime_connections_total").increment(1);
    tracing::info!(conn_id, "realtime: connection opened");
    state.voice_pin.session_opened();

    let (sink, stream) = socket.split();

    // Outgoing channel — lets both the receiver task and the response pipeline
    // write to the WebSocket without sharing the sink directly.
    let (out_tx, mut out_rx) = mpsc::channel::<WsMsg>(64);

    // Sender task: drains out_rx → WebSocket sink.
    // Exits when all out_tx clones are dropped.
    tokio::spawn(async move {
        let mut sink = sink;
        while let Some(msg) = out_rx.recv().await {
            if sink.send(msg).await.is_err() {
                break;
            }
        }
    });

    // Channel from VAD (receiver task) to the response pipeline (main body).
    // Capacity 4 — absorbs a brief barge-in burst (user keeps talking while a slow
    // TTS synth holds the pipeline) without dropping the first follow-up utterance.
    let (audio_tx, mut audio_rx) = mpsc::channel::<UtteranceCtx>(4);

    // Channel for InjectHistory events: receiver task → pipeline task.
    // Small capacity — injection is infrequent (user-driven), never bursts.
    let (inject_tx, mut inject_rx) = mpsc::channel::<Vec<ClientHistoryEntry>>(4);

    // Channel for ReloadHistory events: receiver task → pipeline task.
    let (reload_tx, mut reload_rx) = mpsc::channel::<Vec<ClientHistoryEntry>>(4);

    // Shared cancel flag: receiver task sets it on `cancel` event;
    // response pipeline checks it each token and resets it at utterance start.
    let cancel = Arc::new(AtomicBool::new(false));
    // NEW-H(B): paired Notify so synthesize_and_send can select! on cancel
    // instead of blocking the full ~3 s TTS synth. notify_waiters() wakes all
    // current waiters but stores no permit — safe across utterances (unlike
    // CancellationToken, which cannot be reset and would poison future turns).
    let cancel_notify = Arc::new(Notify::new());

    // Session state shared between receiver task (config updates) and the
    // response pipeline (reads model IDs / history / updates history).
    let default_embed = state
        .model_manager
        .as_ref()
        .and_then(|mm| mm.default_embed_model().map(str::to_owned));
    let session = Arc::new(tokio::sync::Mutex::new(Session {
        embed_model: default_embed.clone(),
        ..Session::default()
    }));

    // Admin-visible session snapshot: registered globally on connect, updated
    // on config changes and after each turn, removed on close.
    let session_id = uuid::Uuid::new_v4();
    let connected_at = unix_now_ms();
    let session_snap = Arc::new(RwLock::new(SessionSnapshot {
        id: session_id,
        conn_id,
        connected_at,
        stt_model: None,
        llm_model: None,
        tts_model: None,
        tts_voice: None,
        language: None,
        system_prompt: None,
        barge_in_phrase: None,
        embed_model: default_embed,
        turn_count: 0,
        history_entries: Vec::new(),
    }));
    let kill = tokio_util::sync::CancellationToken::new();
    // Cooperative shutdown for the receiver task. The always-streaming client
    // never closes the socket itself, so `run_receiver` would loop forever on
    // `stream.next()` and `receiver_task.await` (below) would hang — leaking the
    // registry entry and the voice pin (the "zombie session" bug). Cancelling
    // this token on every break path lets the receiver exit and the socket close.
    let shutdown = tokio_util::sync::CancellationToken::new();
    if let Ok(mut reg) = state.realtime_sessions.write() {
        reg.insert(
            session_id,
            crate::realtime_types::RealtimeSessionEntry {
                snapshot: Arc::clone(&session_snap),
                kill: kill.clone(),
            },
        );
    }

    // Receiver task: reads WebSocket frames, runs VAD, fires config/cancel.
    let receiver_task = {
        let out_tx = out_tx.clone();
        let cancel = Arc::clone(&cancel);
        let session = Arc::clone(&session);
        let session_snap = Arc::clone(&session_snap);
        let shutdown = shutdown.clone();
        let model_manager = state.model_manager.clone();
        let voice_pin = Arc::clone(&state.voice_pin);
        tokio::spawn(run_receiver(
            stream,
            out_tx,
            cancel,
            Arc::clone(&cancel_notify),
            session,
            audio_tx,
            inject_tx,
            reload_tx,
            conn_id,
            session_snap,
            shutdown,
            model_manager,
            voice_pin,
        ))
    };

    // C2: bridge admin-kill into an in-flight turn (see `spawn_kill_bridge`).
    spawn_kill_bridge(
        Arc::clone(&cancel),
        Arc::clone(&cancel_notify),
        kill.clone(),
        shutdown.clone(),
    );

    // ── Embedding handle ──────────────────────────────────────────────────────
    // Acquired before the main loop. Typically None at session start because no
    // Config event has been received yet; re-tried each turn when still None.
    let mut embed_handle: Option<crate::embed_engine::EmbeddingHandle> = {
        let guard = session.lock().await;
        guard.embed_model.as_deref().and_then(|model| {
            state
                .model_manager
                .as_ref()?
                .get_embedding_handle(model)
                .ok()
        })
        // guard dropped here — no MutexGuard held across any await
    };

    // Initial back-fill: embed history entries that have no vector yet.
    // This is a no-op for fresh sessions (history_entries is empty) but
    // handles the case where the client sends Config+history before the first
    // audio frame and some entries lack pre-computed embeddings.
    if let Some(ref handle) = embed_handle {
        let model_id = handle.model_id().to_owned();
        let (texts, indices): (Vec<String>, Vec<usize>) = {
            let guard = session.lock().await;
            guard
                .history_entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.embedding.is_none())
                .map(|(i, e)| (format!("{}\n{}", e.user, e.assistant), i))
                .unzip()
        }; // guard dropped
        if !texts.is_empty()
            && let Ok(output) = handle.embed(texts).await
        {
            // Apply vectors to entries and collect event data (lock scope).
            let events: Vec<(String, String, Vec<f32>, u64)> = {
                let mut guard = session.lock().await;
                indices
                    .iter()
                    .zip(output.vectors.iter())
                    .filter_map(|(&idx, vec)| {
                        guard.history_entries.get_mut(idx).map(|entry| {
                            entry.embedding = Some(vec.clone());
                            entry.embedding_model = Some(model_id.clone());
                            (
                                entry.user.clone(),
                                entry.assistant.clone(),
                                vec.clone(),
                                entry.timestamp,
                            )
                        })
                    })
                    .collect()
            }; // guard dropped
            for (user, assistant, embedding, timestamp) in events {
                send_event(
                    &out_tx,
                    &ServerEvent::TurnHistory {
                        turn_id: None,
                        user: &user,
                        assistant: &assistant,
                        embedding: Some(embedding),
                        timestamp,
                        embedding_model: Some(model_id.as_str()),
                        stt_model: None,
                        llm_model: None,
                    },
                )
                .await;
            }
        }
    }

    // Response pipeline (runs in this task, sequentially — one utterance at a
    // time). Receives committed audio from the VAD via audio_rx.
    let mut turn_id: u64 = 0;
    loop {
        let utterance = tokio::select! {
            // `biased`: a pending admin-kill must win over an already-queued
            // utterance. Without it the kill arm and a ready `audio_rx` race
            // ~50/50, so after the C2 bridge aborts the in-flight turn the loop
            // could pick up the NEXT queued utterance and generate it in full —
            // the session would survive the kill by ~one extra turn per queued
            // utterance (observed live: a barge-in-queued second turn ran ~30 s
            // past the kill). Checking kill first drops the queue and tears down.
            biased;
            () = kill.cancelled() => {
                tracing::info!(conn_id, "realtime: session killed by admin");
                break;
            }
            Some(entries) = inject_rx.recv() => {
                handle_inject_history(
                    entries,
                    &session,
                    &session_snap,
                    &out_tx,
                    embed_handle.as_ref(),
                    conn_id,
                )
                .await;
                continue;
            }
            Some(entries) = reload_rx.recv() => {
                handle_reload_history(
                    entries,
                    &session,
                    &session_snap,
                    &out_tx,
                    embed_handle.as_ref(),
                    conn_id,
                )
                .await;
                continue;
            }
            result = time::timeout(IDLE_TIMEOUT, audio_rx.recv()) => match result {
                Ok(Some(u)) => u,
                Ok(None) => break, // receiver task exited → normal shutdown
                Err(_) => {
                    tracing::debug!(conn_id, "realtime: idle timeout — closing session");
                    break;
                }
            }
        };

        let snapshot = {
            let guard = session.lock().await;
            guard.clone()
        };

        // If embed_handle hasn't been acquired yet (embed_model was set by a
        // Config event after session start), try to acquire it now.
        if embed_handle.is_none() {
            embed_handle = snapshot.embed_model.as_deref().and_then(|model| {
                state
                    .model_manager
                    .as_ref()?
                    .get_embedding_handle(model)
                    .ok()
            });
        }

        let mut ctx = TurnCtx::new(conn_id, turn_id, &utterance);
        let (user_text, assistant_text) = run_response(
            utterance.audio,
            &snapshot,
            &out_tx,
            &cancel,
            &cancel_notify,
            &state,
            &mut ctx,
            embed_handle.as_ref(),
        )
        .await;

        turn_id += 1;

        // Only extend history for fully-completed (non-cancelled) turns.
        if !user_text.is_empty() && !assistant_text.is_empty() {
            // Embed the turn (user+"\n"+assistant) if an embed handle is available.
            let (embedding, embedding_model): (Option<Vec<f32>>, Option<String>) =
                if let Some(ref handle) = embed_handle {
                    let text = format!("{user_text}\n{assistant_text}");
                    let vec = handle
                        .embed(vec![text])
                        .await
                        .ok()
                        .and_then(|mut o| o.vectors.pop());
                    let model = vec.is_some().then(|| handle.model_id().to_owned());
                    (vec, model)
                } else {
                    (None, None)
                };
            let timestamp = unix_now_ms();
            let completed_turn_id = ctx.turn_id; // saved before turn_id is incremented

            // Notify the client: full turn history entry with optional embedding.
            send_event(
                &out_tx,
                &ServerEvent::TurnHistory {
                    turn_id: Some(completed_turn_id),
                    user: &user_text,
                    assistant: &assistant_text,
                    embedding: embedding.clone(),
                    timestamp,
                    embedding_model: embedding_model.as_deref(),
                    stt_model: snapshot.stt_model.as_deref(),
                    llm_model: snapshot.llm_model.as_deref(),
                },
            )
            .await;

            // Try to pin this session's model set on first successful turn
            // (first-writer-wins; no-op if pin already set). Doing this after
            // a proven turn ensures the models are confirmed working before
            // being pinned — prevents an explicit-override session that is
            // still loading a model from pinning an unverified set.
            if let (Some(stt), Some(llm), Some(tts)) = (
                snapshot.stt_model.as_ref(),
                snapshot.llm_model.as_ref(),
                snapshot.tts_model.as_ref(),
            ) && state
                .voice_pin
                .try_set(stt.clone(), llm.clone(), tts.clone())
            {
                tracing::info!(
                    conn_id,
                    stt_model = %stt,
                    llm_model = %llm,
                    tts_model = %tts,
                    "realtime: voice flow pinned"
                );
            }

            let mut guard = session.lock().await;
            // Push the full entry for embedding-based retrieval.
            guard.history_entries.push(ClientHistoryEntry {
                user: user_text.clone(),
                assistant: assistant_text.clone(),
                embedding,
                timestamp,
                embedding_model,
                stt_model: snapshot.stt_model.clone(),
                llm_model: snapshot.llm_model.clone(),
            });
            // Trim the oldest turn pair when the LLM context cap is reached.
            if guard.history.len() >= MAX_HISTORY_TURNS * 2 {
                guard.history.drain(..2);
            }
            guard.history.push(RtMessage {
                role: "user".to_owned(),
                content: user_text,
            });
            guard.history.push(RtMessage {
                role: "assistant".to_owned(),
                content: assistant_text,
            });
            // Snapshot for admin API — clone before releasing the tokio lock.
            let new_entries = guard.history_entries.clone();
            drop(guard);
            if let Ok(mut snap) = session_snap.write() {
                snap.turn_count += 1;
                snap.history_entries = new_entries;
            }
        }
    }

    tracing::info!(conn_id, turns = turn_id, "realtime: connection closed");
    // Wake the receiver so it exits its frame loop on every break path (kill,
    // idle, audio-None). Idempotent. Without this, the always-streaming client
    // keeps the receiver blocked on `stream.next()` and this await never returns.
    shutdown.cancel();
    receiver_task.await.ok();
    // D2: release whatever this session's last Config resolution registered
    // into the realtime serving set — otherwise a closed session's models
    // would stay eviction-protected forever.
    {
        let final_serving: Vec<String> = {
            let s = session.lock().await;
            [
                s.stt_model.clone(),
                s.llm_model.clone(),
                s.tts_model.clone(),
            ]
            .into_iter()
            .flatten()
            .collect()
        };
        state
            .voice_pin
            .deregister_serving(final_serving.iter().map(String::as_str));
    }
    state.voice_pin.session_closed();
    if let Ok(mut reg) = state.realtime_sessions.write() {
        reg.remove(&session_id);
    }
    // out_tx (main body's clone) drops here; once receiver_task's clone
    // also dropped, the sender task's out_rx closes and the sender exits.
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests (no GPU / no WebSocket required)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    // ── SentenceSplitter ─────────────────────────────────────────────────────

    #[test]
    fn splitter_holds_back_short_fragment() {
        let mut s = SentenceSplitter::new();
        assert!(s.push("Hello.").is_none(), "too short to split");
    }

    #[test]
    fn splitter_emits_on_period_after_min_len() {
        let mut s = SentenceSplitter::new();
        // Push tokens until we cross SENTENCE_MIN_LEN with a period.
        assert!(
            s.push("The quick brown fox jumps over the lazy dog")
                .is_none()
        );
        let result = s.push(".");
        assert_eq!(
            result.as_deref(),
            Some("The quick brown fox jumps over the lazy dog.")
        );
    }

    #[test]
    fn splitter_emits_on_exclamation() {
        let mut s = SentenceSplitter::new();
        // "What a wonderful day it is today!" is 33 chars (> SENTENCE_MIN_LEN)
        // and ends with '!' — the splitter finds the boundary immediately.
        let result = s.push("What a wonderful day it is today!");
        assert_eq!(result.as_deref(), Some("What a wonderful day it is today!"));
        // "More text follows." is 18 chars (> SENTENCE_MIN_LEN=15) and ends
        // with '.' — push splits it immediately; flush returns nothing.
        let result2 = s.push("More text follows.");
        assert_eq!(result2.as_deref(), Some("More text follows."));
        assert!(s.flush().is_none(), "buffer should be empty after split");
    }

    #[test]
    fn splitter_flush_returns_remainder() {
        let mut s = SentenceSplitter::new();
        s.push("This sentence has no terminal punctuation");
        assert_eq!(
            s.flush().as_deref(),
            Some("This sentence has no terminal punctuation")
        );
        assert!(s.flush().is_none(), "buffer should be empty after flush");
    }

    #[test]
    fn splitter_paragraph_break_splits() {
        let mut s = SentenceSplitter::new();
        s.push("First paragraph content here");
        let r = s.push("\n\nSecond paragraph");
        assert_eq!(r.as_deref(), Some("First paragraph content here"));
    }

    /// A `.` that is the last buffered byte with a digit right before it must
    /// NOT split — the LLM's tokenizer can emit "3", ".", "14" as three
    /// separate tokens, and splitting after "3." would hand "14" to the next
    /// sentence, making decimal reconstruction impossible downstream. See
    /// the project's internal engineering log.
    #[test]
    fn splitter_holds_back_trailing_decimal_point() {
        let mut s = SentenceSplitter::new();
        assert!(s.push("The value of pi is ").is_none());
        assert!(
            s.push("3").is_none(),
            "still short of a boundary, no period yet"
        );
        assert!(
            s.push(".").is_none(),
            "trailing '.' after a digit must be held back as a likely decimal point"
        );
        assert!(
            s.push("14").is_none(),
            "the held-back buffer should now contain the full decimal, still no space/EOS"
        );
        let r = s.push(" done.");
        assert_eq!(r.as_deref(), Some("The value of pi is 3.14 done."));
    }

    /// A genuine sentence-ending period with no preceding digit still splits
    /// immediately at end-of-buffer, same as before this fix.
    #[test]
    fn splitter_still_splits_on_non_digit_trailing_period() {
        let mut s = SentenceSplitter::new();
        let r = s.push("This is a genuine sentence end.");
        assert_eq!(r.as_deref(), Some("This is a genuine sentence end."));
    }

    // ── sanitize_for_tts ────────────────────────────────────────────────────

    #[test]
    fn sanitize_strips_markdown_decoration() {
        assert_eq!(
            sanitize_for_tts("**bold** and _italic_ and `code`", None),
            "bold and italic and code"
        );
    }

    /// The bug report this pins: an emoji mid-sentence was observed to make
    /// the TTS engine stall for a long pause rather than skip cleanly —
    /// strip it before synthesis ever sees it.
    #[test]
    fn sanitize_strips_emoji() {
        assert_eq!(
            sanitize_for_tts("Add it to your cart 🛒 now", None),
            "Add it to your cart now"
        );
    }

    #[test]
    fn sanitize_strips_emoji_at_boundaries_and_runs() {
        assert_eq!(sanitize_for_tts("🛒", None), "");
        assert_eq!(sanitize_for_tts("🛒🎉 great news 🎉🛒", None), "great news");
    }

    #[test]
    fn sanitize_runs_number_normalization_first() {
        // Confirms the full pipeline: normalize_for_speech expands "1,000"
        // before the markdown/emoji strip ever runs.
        assert_eq!(
            sanitize_for_tts("over 1,000 years 🛒", Some("en")),
            "over one thousand years"
        );
    }

    // ── PCM16 utilities ───────────────────────────────────────────────────────

    #[test]
    fn pcm16_round_trip() {
        let samples: Vec<i16> = vec![0, 1000, -1000, i16::MAX, i16::MIN];
        let bytes = i16_to_bytes(&samples);
        let back = bytes_to_i16(&bytes);
        assert_eq!(back, samples);
    }

    #[test]
    fn f32_to_pcm16_clamps() {
        let samples = vec![0.0_f32, 1.0, -1.0, 2.0, -2.0];
        let bytes = f32_to_pcm16_bytes(&samples);
        let i16s = bytes_to_i16(&bytes);
        assert_eq!(i16s[0], 0);
        assert_eq!(i16s[1], i16::MAX);
        assert_eq!(i16s[2], i16::MIN + 1); // clamp(-1.0, 1.0) * 32767
        // Values > 1.0 and < -1.0 are clamped.
        assert_eq!(i16s[3], i16::MAX);
        assert_eq!(i16s[4], i16::MIN + 1);
    }

    #[test]
    fn wav_header_starts_with_riff() {
        let pcm = i16_to_bytes(&[0i16; 16]);
        let wav = pcm16_to_wav(&pcm, 16_000);
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(wav.len(), 44 + pcm.len());
    }

    // ── ClientEvent parsing ───────────────────────────────────────────────────

    #[test]
    fn parse_config_event() -> anyhow::Result<()> {
        let json = r#"{"type":"config","llm_model":"qwen3-8b","stt_model":"whisper-base"}"#;
        let ev: ClientEvent = serde_json::from_str(json)?;
        assert!(matches!(ev, ClientEvent::Config { .. }));
        Ok(())
    }

    #[test]
    fn parse_config_event_new_fields() -> anyhow::Result<()> {
        let json = r#"{
            "type": "config",
            "llm_model": "qwen3-8b",
            "barge_in_phrase": "Ruvi stop talking",
            "embed_model": "bge-embed-ov",
            "history": [
                {
                    "user": "Hello",
                    "assistant": "Hi there",
                    "timestamp": 1700000000000
                }
            ]
        }"#;
        let ev: ClientEvent = serde_json::from_str(json)?;
        if let ClientEvent::Config {
            barge_in_phrase,
            embed_model,
            history,
            ..
        } = ev
        {
            assert_eq!(barge_in_phrase.as_deref(), Some("Ruvi stop talking"));
            assert_eq!(embed_model.as_deref(), Some("bge-embed-ov"));
            let entries = history.expect("history should be Some");
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].user, "Hello");
            assert!(entries[0].embedding.is_none());
        } else {
            panic!("expected Config variant");
        }
        Ok(())
    }

    #[test]
    fn parse_cancel_event() -> anyhow::Result<()> {
        let json = r#"{"type":"cancel"}"#;
        let ev: ClientEvent = serde_json::from_str(json)?;
        assert!(matches!(ev, ClientEvent::Cancel));
        Ok(())
    }

    #[test]
    fn unknown_event_type_errors() {
        let json = r#"{"type":"unknown"}"#;
        assert!(serde_json::from_str::<ClientEvent>(json).is_err());
    }

    /// `Done` events carry a `turn_id` field — clients use it to join their
    /// local JSONL log with the server-side telemetry log.
    #[test]
    fn done_event_serialises_with_turn_id() -> anyhow::Result<()> {
        let json = serde_json::to_string(&ServerEvent::Done { turn_id: 7 })?;
        assert!(json.contains("\"turn_id\":7"), "missing turn_id: {json}");
        assert!(json.contains("\"type\":\"done\""), "missing type: {json}");
        Ok(())
    }

    // ── sentence_cap_reached ─────────────────────────────────────────────────

    #[test]
    fn sentence_cap_reached_none_never_fires() {
        let session = Session::default(); // max_sentences: None
        let u = UtteranceCtx {
            audio: vec![],
            energy: 0.0,
            utterance_duration_ms: 0,
        };
        let mut ctx = TurnCtx::new(0, 0, &u);
        ctx.tts_sentences = 100;
        assert!(!sentence_cap_reached(&session, &ctx));
    }

    #[test]
    fn sentence_cap_reached_fires_at_limit() {
        let session = Session {
            max_sentences: Some(3),
            ..Session::default()
        };
        let u = UtteranceCtx {
            audio: vec![],
            energy: 0.0,
            utterance_duration_ms: 0,
        };
        let mut ctx = TurnCtx::new(0, 0, &u);
        ctx.tts_sentences = 2;
        assert!(
            !sentence_cap_reached(&session, &ctx),
            "should not fire before limit"
        );
        ctx.tts_sentences = 3;
        assert!(sentence_cap_reached(&session, &ctx), "should fire at limit");
        ctx.tts_sentences = 4;
        assert!(
            sentence_cap_reached(&session, &ctx),
            "should fire past limit"
        );
    }

    // ── Helper functions ──────────────────────────────────────────────────────

    #[test]
    fn barge_in_normalises_and_matches() {
        assert!(is_barge_in_phrase(
            "Hey Ruvi, stop talking please!",
            "ruvi stop talking"
        ));
        // Punctuation and case are stripped.
        assert!(is_barge_in_phrase(
            "RUVI, STOP TALKING!",
            "ruvi stop talking"
        ));
        // Unrelated phrase doesn't match.
        assert!(!is_barge_in_phrase("hello world", "ruvi stop talking"));
        // Empty phrase never matches.
        assert!(!is_barge_in_phrase("whatever", ""));
    }

    #[test]
    fn cosine_sim_identical_vectors() {
        let v = vec![1.0_f32, 0.0, 0.0];
        assert!((cosine_sim(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        assert!(cosine_sim(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_zero_vector() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 0.0];
        assert!(cosine_sim(&a, &b).abs() < 1e-6);
    }

    // ── retrieve_memories: embedding-model provenance gate ──────────────────

    fn history_entry(
        embedding: Option<Vec<f32>>,
        embedding_model: Option<&str>,
    ) -> ClientHistoryEntry {
        ClientHistoryEntry {
            user: "u".to_owned(),
            assistant: "a".to_owned(),
            embedding,
            timestamp: 0,
            embedding_model: embedding_model.map(str::to_owned),
            stt_model: None,
            llm_model: None,
        }
    }

    /// `spawn_mock_embed` always answers `[0.1, 0.2, 0.3, 0.4]`, so an entry
    /// carrying that exact vector scores a perfect 1.0 against any query on raw
    /// cosine similarity alone — the model-provenance gate is the only thing
    /// that can tell it apart from a genuinely compatible entry embedded by a
    /// different model that happens to produce a same-length vector.
    #[tokio::test]
    async fn retrieve_memories_excludes_an_entry_from_a_different_embedding_model() {
        let (handle, _thread) = crate::embed_engine::spawn_mock_embed("mock-embed");
        let same_vector = vec![0.1, 0.2, 0.3, 0.4];
        let entries = vec![
            history_entry(Some(same_vector.clone()), Some("mock-embed")),
            history_entry(Some(same_vector), Some("some-other-embed-model")),
        ];

        let hits = retrieve_memories("query", &entries, &handle).await;

        assert_eq!(hits.len(), 1, "only the matching-model entry should score");
        assert_eq!(hits[0].embedding_model.as_deref(), Some("mock-embed"));
    }

    /// An entry with no recorded `embedding_model` (older client / pre-fix data)
    /// falls back to the previous length-only check rather than being dropped
    /// outright — same-length passes, different-length is excluded.
    #[tokio::test]
    async fn retrieve_memories_falls_back_to_dim_check_without_provenance() {
        let (handle, _thread) = crate::embed_engine::spawn_mock_embed("mock-embed");
        let entries = vec![
            history_entry(Some(vec![0.1, 0.2, 0.3, 0.4]), None),
            history_entry(Some(vec![0.1, 0.2, 0.3]), None),
        ];

        let hits = retrieve_memories("query", &entries, &handle).await;

        assert_eq!(hits.len(), 1, "only the same-dimension entry should score");
        assert_eq!(
            hits[0].embedding.as_deref(),
            Some(&[0.1, 0.2, 0.3, 0.4][..])
        );
    }

    // ── strip_gpt_oss_channel_prefix ─────────────────────────────────────────

    #[test]
    fn channel_strip_lowercase_bare_name() {
        assert_eq!(
            strip_gpt_oss_channel_prefix("final answer"),
            (" answer", true)
        );
        assert_eq!(
            strip_gpt_oss_channel_prefix("analysis: here"),
            (": here", true)
        );
        assert_eq!(strip_gpt_oss_channel_prefix("commentary on"), (" on", true));
    }

    #[test]
    fn channel_strip_uppercase_caught() {
        // Case-insensitive match when followed by non-alpha.
        assert_eq!(
            strip_gpt_oss_channel_prefix("Final answer"),
            (" answer", true)
        );
        assert_eq!(
            strip_gpt_oss_channel_prefix("FINAL answer"),
            (" answer", true)
        );
    }

    #[test]
    fn channel_strip_direct_concat_stripped() {
        // Model emits "final" then message with no space: "finalI'm" → strip to "I'm".
        assert_eq!(
            strip_gpt_oss_channel_prefix("finalI'm here"),
            ("I'm here", true)
        );
        assert_eq!(strip_gpt_oss_channel_prefix("finalSure"), ("Sure", true));
    }

    #[test]
    fn channel_strip_finally_not_stripped() {
        // "finally"/"Finally" → char after "final" is lowercase 'l' → natural word, keep.
        assert_eq!(
            strip_gpt_oss_channel_prefix("finally, "),
            ("finally, ", false)
        );
        assert_eq!(
            strip_gpt_oss_channel_prefix("Finally, "),
            ("Finally, ", false)
        );
    }

    #[test]
    fn channel_strip_no_match() {
        assert_eq!(
            strip_gpt_oss_channel_prefix("sure, let me"),
            ("sure, let me", false)
        );
        assert_eq!(strip_gpt_oss_channel_prefix(""), ("", false));
    }

    // ── VoicePolicy ───────────────────────────────────────────────────────────

    #[test]
    fn voice_policy_gpt_oss() {
        let p = VoicePolicy::from_parser(Some(&ReasoningParser::GptOss));
        assert_eq!(p.channel_prefill, Some("<|channel|>final<|message|>"));
        assert!(p.strip_output_prefix);
        assert!(p.system_as_identity);
    }

    #[test]
    fn voice_policy_qwen3_is_default() {
        let p = VoicePolicy::from_parser(Some(&ReasoningParser::Qwen3));
        assert!(p.channel_prefill.is_none());
        assert!(!p.strip_output_prefix);
        assert!(!p.system_as_identity);
    }

    #[test]
    fn voice_policy_no_parser_is_default() {
        let p = VoicePolicy::from_parser(None);
        assert!(p.channel_prefill.is_none());
        assert!(!p.strip_output_prefix);
        assert!(!p.system_as_identity);
    }

    #[test]
    fn voice_policy_mistral_is_default() {
        let p = VoicePolicy::from_parser(Some(&ReasoningParser::Mistral));
        assert!(p.channel_prefill.is_none());
        assert!(!p.strip_output_prefix);
        assert!(!p.system_as_identity);
    }

    #[test]
    fn voice_policy_phi_is_default() {
        let p = VoicePolicy::from_parser(Some(&ReasoningParser::Phi));
        assert!(p.channel_prefill.is_none());
        assert!(!p.strip_output_prefix);
        assert!(!p.system_as_identity);
    }

    // ── EnergyVad ─────────────────────────────────────────────────────────────

    #[test]
    fn vad_silence_emits_nothing() {
        let mut vad = EnergyVad::new();
        // All-zero PCM = silence.
        let silence = i16_to_bytes(&[0i16; 1600]);
        let events = vad.push_bytes(&silence);
        assert!(events.is_empty());
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn vad_speech_above_threshold_starts_onset() {
        let mut vad = EnergyVad::new();
        // 0.1 normalised amplitude → RMS 0.1, well above VAD_THRESHOLD (0.015).
        // Value 3276 is well within i16 range; cast_possible_truncation is a false positive.
        let loud_sample: i16 = (0.1 * f32::from(i16::MAX)) as i16;
        let loud = i16_to_bytes(&[loud_sample; 100]);
        let events = vad.push_bytes(&loud);
        // 100 samples < VAD_ONSET_SAMPLES (1600) — still in Onset, no event yet.
        assert!(events.is_empty());
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn vad_sustained_speech_emits_started() {
        let mut vad = EnergyVad::new();
        // Value 3276 is well within i16 range; cast_possible_truncation is a false positive.
        let loud_sample: i16 = (0.1 * f32::from(i16::MAX)) as i16;
        // Push enough loud samples to cross VAD_ONSET_SAMPLES.
        let loud = i16_to_bytes(&vec![loud_sample; VAD_ONSET_SAMPLES + 100]);
        let events = vad.push_bytes(&loud);
        assert!(
            events.iter().any(|e| matches!(e, VadEvent::SpeechStarted)),
            "expected SpeechStarted after sustained loud audio"
        );
    }

    // ── run_receiver shutdown (C1 zombie-session fix) ────────────────────────

    /// The always-streaming client never closes the socket, so the frame stream
    /// never yields. Before C1 this made `run_receiver` loop forever and the
    /// session leaked. Assert the receiver exits promptly once `shutdown` fires.
    #[tokio::test]
    async fn receiver_exits_on_shutdown_token() {
        use std::sync::atomic::AtomicBool;

        // `pending` never yields a frame — models the always-streaming client.
        let stream = futures_util::stream::pending::<Result<WsMsg, axum::Error>>();
        let (out_tx, _out_rx) = mpsc::channel(1);
        let cancel = Arc::new(AtomicBool::new(false));
        let session = Arc::new(tokio::sync::Mutex::new(Session::default()));
        let (audio_tx, _audio_rx) = mpsc::channel(1);
        let (inject_tx, _inject_rx) = mpsc::channel(1);
        let (reload_tx, _reload_rx) = mpsc::channel(1);
        let snap = Arc::new(RwLock::new(SessionSnapshot {
            id: uuid::Uuid::new_v4(),
            conn_id: 0,
            connected_at: 0,
            stt_model: None,
            llm_model: None,
            tts_model: None,
            tts_voice: None,
            language: None,
            system_prompt: None,
            barge_in_phrase: None,
            embed_model: None,
            turn_count: 0,
            history_entries: Vec::new(),
        }));
        let shutdown = tokio_util::sync::CancellationToken::new();

        let handle = tokio::spawn(run_receiver(
            stream,
            out_tx,
            cancel,
            Arc::new(Notify::new()),
            session,
            audio_tx,
            inject_tx,
            reload_tx,
            0,
            snap,
            shutdown.clone(),
            None,
            Arc::new(crate::voice_pin::VoicePinManager::default()),
        ));

        shutdown.cancel();
        let join = tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("run_receiver did not exit within 1s of shutdown.cancel()");
        assert!(join.is_ok(), "run_receiver task panicked");
    }

    // ── spawn_kill_bridge (C2 in-flight kill) ────────────────────────────────

    /// An admin kill must set `cancel` (so the in-flight drain aborts) and cancel
    /// `shutdown` (so the receiver wakes).
    #[tokio::test]
    async fn kill_bridge_sets_cancel_and_shutdown() {
        use std::sync::atomic::AtomicBool;
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_notify = Arc::new(Notify::new());
        let kill = tokio_util::sync::CancellationToken::new();
        let shutdown = tokio_util::sync::CancellationToken::new();
        spawn_kill_bridge(
            Arc::clone(&cancel),
            Arc::clone(&cancel_notify),
            kill.clone(),
            shutdown.clone(),
        );

        assert!(!cancel.load(Ordering::Acquire), "cancel must start false");
        kill.cancel();
        // The bridge runs on its own task; poll briefly for it to react.
        for _ in 0..100 {
            if cancel.load(Ordering::Acquire) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        assert!(
            cancel.load(Ordering::Acquire),
            "kill should set cancel=true"
        );
        assert!(shutdown.is_cancelled(), "kill should cancel shutdown");
    }

    /// A normal close fires `shutdown` (not `kill`); the bridge must exit without
    /// touching `cancel` — otherwise it would spuriously abort and leak.
    #[tokio::test]
    async fn kill_bridge_normal_close_leaves_cancel_untouched() {
        use std::sync::atomic::AtomicBool;
        let cancel = Arc::new(AtomicBool::new(false));
        let kill = tokio_util::sync::CancellationToken::new();
        let shutdown = tokio_util::sync::CancellationToken::new();
        spawn_kill_bridge(
            Arc::clone(&cancel),
            Arc::new(Notify::new()),
            kill.clone(),
            shutdown.clone(),
        );

        shutdown.cancel(); // normal teardown path
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(
            !cancel.load(Ordering::Acquire),
            "normal close must not set cancel"
        );
    }

    // ── NEW-H(B): cancel_notify fires on admin kill ───────────────────────────

    /// `kill_bridge` must fire `cancel_notify` alongside cancel=true so an in-flight
    /// `synthesize_and_send` can abort via select! instead of waiting up to ~3 s.
    #[tokio::test]
    async fn kill_bridge_fires_cancel_notify() {
        use std::sync::atomic::AtomicBool;
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_notify = Arc::new(Notify::new());
        let kill = tokio_util::sync::CancellationToken::new();
        let shutdown = tokio_util::sync::CancellationToken::new();
        spawn_kill_bridge(
            Arc::clone(&cancel),
            Arc::clone(&cancel_notify),
            kill.clone(),
            shutdown.clone(),
        );

        // Subscribe to the notification BEFORE firing kill (matches production pattern).
        let notified = cancel_notify.notified();
        kill.cancel();
        // notified() must resolve — i.e. notify_waiters() was called by the bridge.
        tokio::time::timeout(Duration::from_millis(200), notified)
            .await
            .expect("cancel_notify must fire within 200 ms of admin kill");
    }

    // ── NEW-H(A): pipeline-busy Error storm suppression ──────────────────────

    /// When the audio channel backlogs (barge-in burst), the receiver must emit
    /// at most ONE "pipeline busy" Error per episode — not one per dropped
    /// utterance — so it can't storm the client. Drive four committed utterances
    /// through a cap-1 channel that is never drained: the first fills it, the next
    /// three overflow, and exactly one Error must reach the client.
    #[tokio::test]
    #[allow(clippy::cast_possible_truncation)]
    async fn pipeline_busy_error_emitted_once_per_episode() {
        use std::sync::atomic::AtomicBool;

        // Each utterance = loud frame (crosses onset → SpeechStarted, enters Speech)
        // + silent frame (crosses offset → SpeechEnded → one try_send attempt).
        let loud_sample: i16 = (0.1 * f32::from(i16::MAX)) as i16;
        let loud = i16_to_bytes(&vec![loud_sample; VAD_ONSET_SAMPLES + 100]);
        let silence = i16_to_bytes(&vec![0i16; VAD_OFFSET_SAMPLES + 100]);
        let mut frames: Vec<Result<WsMsg, axum::Error>> = Vec::new();
        for _ in 0..4 {
            frames.push(Ok(WsMsg::Binary(loud.clone().into())));
            frames.push(Ok(WsMsg::Binary(silence.clone().into())));
        }
        // Stream terminates (None) → receiver breaks on its own; no shutdown needed.
        let stream = futures_util::stream::iter(frames);

        let (out_tx, mut out_rx) = mpsc::channel::<WsMsg>(64);
        let cancel = Arc::new(AtomicBool::new(false));
        let session = Arc::new(tokio::sync::Mutex::new(Session::default()));
        // Cap 1, kept alive but never read: utterance 1 fills it, 2..4 overflow.
        let (audio_tx, _audio_rx) = mpsc::channel::<UtteranceCtx>(1);
        let (inject_tx, _inject_rx) = mpsc::channel(1);
        let (reload_tx, _reload_rx) = mpsc::channel(1);
        let snap = Arc::new(RwLock::new(SessionSnapshot {
            id: uuid::Uuid::new_v4(),
            conn_id: 0,
            connected_at: 0,
            stt_model: None,
            llm_model: None,
            tts_model: None,
            tts_voice: None,
            language: None,
            system_prompt: None,
            barge_in_phrase: None,
            embed_model: None,
            turn_count: 0,
            history_entries: Vec::new(),
        }));
        let shutdown = tokio_util::sync::CancellationToken::new();

        run_receiver(
            stream,
            out_tx,
            cancel,
            Arc::new(Notify::new()),
            session,
            audio_tx,
            inject_tx,
            reload_tx,
            0,
            snap,
            shutdown,
            None,
            Arc::new(crate::voice_pin::VoicePinManager::default()),
        )
        .await;

        let mut busy_errors = 0;
        while let Ok(msg) = out_rx.try_recv() {
            if let WsMsg::Text(t) = msg
                && t.contains("pipeline busy")
            {
                busy_errors += 1;
            }
        }
        assert_eq!(
            busy_errors, 1,
            "expected exactly one pipeline-busy Error per backlog episode, got {busy_errors}"
        );
    }

    // ── run_response device admission (step 3) ───────────────────────────────
    //
    // Residual not covered here, deferred to step 4 (the project's internal engineering log
    // 2026-07-17 "realtime turn admission lease"): whether the STT device
    // token is released *promptly after run_stt* rather than merely *by the
    // time run_response returns* is not distinguishable by any test that
    // awaits the whole call — RAII drops everything on return either way, so
    // early-vs-late release look identical from outside. Proving the early
    // release needs a concurrency probe (spawn run_response, synchronize on
    // the post-release `Transcript` event, admit the STT device from outside
    // while the turn is still in its LLM/TTS stage) — deferred to step 4,
    // when a real `EngineFactory` double that can be held open on demand is
    // worth building alongside the live device-cap calibration work. What IS
    // covered: the up-front atomic admit happens before STT is touched at all
    // (`admission_rejected_turn_never_reaches_stt` below), the release
    // mechanism itself (`admission::tests::admit_accumulates_repeated_wants_
    // entries_on_the_same_device`), and RAII scope-holding (compiler-enforced,
    // not test-enforced — `turn_lease` is never moved out of `run_response`).

    /// Minimal `Config` for building a real `ModelManager` + `DeviceBudgets`
    /// pair without touching disk — no model needs to be registered for
    /// `record_device` to resolve a device string (it falls back to
    /// `config.device` for any unregistered id), so these tests only need
    /// `session.{stt,llm,tts}_model` to be `Some`, never a loaded model.
    fn admission_test_config(
        device_budgets: std::collections::HashMap<String, u32>,
    ) -> crate::model_manager::Config {
        crate::model_manager::Config {
            models_dir: std::path::PathBuf::from("/tmp/test-models"),
            device: "CPU".to_owned(),
            preload: vec![],
            models: std::collections::HashMap::new(),
            total_vram_gb: 22.5,
            max_num_seqs: 1000,
            cache_size_gb: 0.0,
            default_kv_cache_gb: 0.0,
            vram_safety_margin_gb: 0.0,
            min_kv_cache_gb: 0.0,
            light_model_max_gb: 2.0,
            light_stt_max_gb: 1.0,
            dgpu_size_ceiling_fraction: 0.8,
            system_ram_reservation_gb: None,
            system_ram_budget_gb: None,
            eviction_grace_secs: 0.0,
            realtime_defaults: None,
            realtime_viable_minimum: None,
            domain_budgets: std::collections::HashMap::new(),
            device_budgets,
            kv_cache_precision: String::new(),
            enable_prefix_caching: true,
            cors_allowed_origins: vec!["*".to_owned()],
            api_keys: Vec::new(),
            admin_api_keys: Vec::new(),
            keys_file: None,
            admission_queue_timeout_ms: 5_000,
            // Short: a reject-path test must not wait out a real 5 s timeout.
            device_admission_queue_timeout_ms: 50,
            embedding_pooling: "mean".to_owned(),
            embedding_normalize: true,
            default_embed_model: None,
            max_prompt_array: 16,
            max_embedding_inputs: 256,
            max_embedding_batch_tokens: 32_768,
            max_tokens_cap: 8192,
            bind_addr: "127.0.0.1".to_owned(),
            port: 11_437,
            allow_insecure_public_bind: false,
            supervisor: crate::model_manager::SupervisorConfig::default(),
            unknown_fields: std::collections::HashMap::new(),
            embedding_device: None,
            ov_cache_dir: None,
            ov_cache_max_gb: 0.0,
            ov_cache_sweep_interval_secs: 21_600,
            kv_pressure_monitor_enabled: false,
            kv_pressure_threshold_pct: 0.0,
            kv_pressure_sustained_secs: 0,
            kv_pressure_sweep_interval_secs: 15,
            kv_resize_cooldown_secs: 30,
        }
    }

    /// A device saturated before the turn starts rejects the whole turn
    /// *before* STT is ever touched: exactly `Error{stage:"admission"}` then
    /// `Done` reach the client, no `Transcript` (proof STT never ran), and
    /// `run_response` returns the empty pair early-return callers rely on to
    /// skip adding a history entry.
    #[tokio::test]
    async fn admission_rejected_turn_never_reaches_stt() {
        use crate::model_manager::ModelManager;
        use crate::model_manager::lifecycle::MockEngineFactory;

        let cfg = admission_test_config(std::collections::HashMap::from([("CPU".to_owned(), 1)]));
        let budgets = Arc::new(crate::admission::DeviceBudgets::from_config(&cfg));
        let mm = ModelManager::new(cfg, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();
        let state = AppState::new()
            .with_model_manager(Arc::new(mm))
            .with_device_budgets(Arc::clone(&budgets));

        // Pre-exhaust the only CPU permit. Bound to `_held`, not `_` — a
        // bare `_` drops immediately and the turn's own admit would succeed.
        let _held = budgets
            .admit(&[("CPU".to_owned(), 1)], crate::admission::WorkClass::Chat)
            .await
            .unwrap();

        let session = Session {
            // Not registered — record_device() falls back to config.device
            // ("CPU"), which is exactly the saturated device above.
            stt_model: Some("phantom-stt".to_owned()),
            ..Default::default()
        };
        let cancel = AtomicBool::new(false);
        let cancel_notify = Notify::new();
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let utterance = UtteranceCtx {
            audio: vec![],
            energy: 0.0,
            utterance_duration_ms: 0,
        };
        let mut ctx = TurnCtx::new(0, 1, &utterance);

        let result = run_response(
            vec![],
            &session,
            &out_tx,
            &cancel,
            &cancel_notify,
            &state,
            &mut ctx,
            None,
        )
        .await;

        assert_eq!(
            result,
            (String::new(), String::new()),
            "a rejected turn must return the empty pair, like every other early-return path"
        );

        let WsMsg::Text(t) = out_rx.try_recv().expect("expected an Error event") else {
            panic!("expected a text frame");
        };
        assert!(t.contains(r#""type":"error"#), "got: {t}");
        assert!(t.contains(r#""stage":"admission"#), "got: {t}");

        let WsMsg::Text(t) = out_rx.try_recv().expect("expected a Done event") else {
            panic!("expected a text frame");
        };
        assert!(t.contains(r#""type":"done"#), "got: {t}");
        assert!(t.contains(r#""turn_id":1"#), "got: {t}");

        assert!(
            out_rx.try_recv().is_err(),
            "no further events — a Transcript event here would mean STT ran \
             despite the rejected admission"
        );
    }

    /// The no-`ModelManager` mock test path (used by every WS-level test
    /// above) never touches `device_budgets` at all — `mm_ref` is `None`, so
    /// `stt_device`/`llm_device`/`tts_device` are all `None` regardless of
    /// what `session` configures, `wants` stays empty, and `admit(&[], ..)`
    /// is a guaranteed no-op. Confirms step 3's wiring didn't add a new
    /// failure mode to the path every other realtime test already exercises.
    #[tokio::test]
    async fn no_model_manager_path_is_inert_to_admission() {
        let state = AppState::default(); // model_manager: None
        let session = Session {
            stt_model: Some("whatever".to_owned()),
            llm_model: Some("whatever".to_owned()),
            tts_model: Some("whatever".to_owned()),
            ..Default::default()
        };
        let cancel = AtomicBool::new(false);
        let cancel_notify = Notify::new();
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let utterance = UtteranceCtx {
            audio: vec![],
            energy: 0.0,
            utterance_duration_ms: 0,
        };
        let mut ctx = TurnCtx::new(0, 1, &utterance);

        let result = run_response(
            vec![],
            &session,
            &out_tx,
            &cancel,
            &cancel_notify,
            &state,
            &mut ctx,
            None,
        )
        .await;

        // The mm=None mock STT always answers "test transcription" (2 words),
        // which trips the < 3 word hallucination filter — the turn ends
        // there, never reaching the mm=None mock LLM response.
        assert_eq!(result, (String::new(), String::new()));
        let WsMsg::Text(t) = out_rx.try_recv().expect("expected a Done event") else {
            panic!("expected a text frame");
        };
        assert!(t.contains(r#""type":"done"#), "got: {t}");
        assert!(
            out_rx.try_recv().is_err(),
            "only Done: the short-transcript path sends nothing else"
        );
    }

    // ── VLM-in-voice-session support (the project's internal engineering log) ────

    /// `admission_test_config` plus one registered (not yet loaded) model —
    /// reuses the 30-odd-field `Config` literal rather than duplicating it.
    fn config_with_model(model_id: &str, vram_gb: f64) -> crate::model_manager::Config {
        let mut cfg = admission_test_config(std::collections::HashMap::new());
        cfg.models.insert(
            model_id.to_owned(),
            crate::model_manager::config::ModelEntry {
                vram_gb,
                kind: None,
                policy: crate::model_manager::config::ModelPolicy::default(),
            },
        );
        cfg
    }

    /// A voice turn against a `Vision`-kind (VLM) LLM streams text and
    /// completes normally — regression test for lifting realtime.rs's former
    /// `EngineHandleKind::TextGen`-only restriction. `run_llm_tts` is called
    /// directly (same technique as the admission tests above): it needs no
    /// STT/VAD plumbing, and `session.tts_model: None` makes
    /// `synthesize_and_send` a no-op, so only `Text`/`Done` events matter here.
    #[tokio::test]
    async fn run_llm_tts_serves_a_vision_kind_llm() {
        use crate::model_manager::ModelManager;
        use crate::model_manager::lifecycle::MockEngineFactory;

        let cfg = config_with_model("vlm-test-model", 5.0);
        let mm = ModelManager::new(cfg, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();
        mm.load_model("vlm-test-model").await.unwrap();
        let state = AppState::new().with_model_manager(Arc::new(mm));

        let session = Session {
            llm_model: Some("vlm-test-model".to_owned()),
            ..Default::default()
        };
        let cancel = AtomicBool::new(false);
        let cancel_notify = Notify::new();
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let utterance = UtteranceCtx {
            audio: vec![],
            energy: 0.0,
            utterance_duration_ms: 0,
        };
        let mut ctx = TurnCtx::new(0, 1, &utterance);

        let result = run_llm_tts(
            "hello there, how are you today",
            &session,
            &out_tx,
            &cancel,
            &cancel_notify,
            &state,
            &mut ctx,
            None,
        )
        .await;

        assert_eq!(result.unwrap(), "mock-vlm");

        let WsMsg::Text(t) = out_rx.try_recv().expect("expected a Text event") else {
            panic!("expected a text frame");
        };
        assert!(t.contains(r#""type":"text"#), "got: {t}");
        assert!(t.contains("mock-vlm"), "got: {t}");

        let WsMsg::Text(t) = out_rx.try_recv().expect("expected a Done event") else {
            panic!("expected a text frame");
        };
        assert!(t.contains(r#""type":"done"#), "got: {t}");
    }

    /// A kind that is neither `TextGen` nor `Vision` still errors — regression
    /// guard proving the `other` catch-all (and its updated message) survived
    /// the restructure that added the `Vision` arm.
    #[tokio::test]
    async fn run_llm_tts_rejects_a_non_text_non_vision_kind() {
        use crate::model_manager::ModelManager;
        use crate::model_manager::lifecycle::MockEngineFactory;

        let cfg = config_with_model("embed-test-model", 1.0);
        let mm = ModelManager::new(cfg, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();
        mm.load_model("embed-test-model").await.unwrap();
        let state = AppState::new().with_model_manager(Arc::new(mm));

        let session = Session {
            llm_model: Some("embed-test-model".to_owned()),
            ..Default::default()
        };
        let cancel = AtomicBool::new(false);
        let cancel_notify = Notify::new();
        let (out_tx, _out_rx) = mpsc::channel(8);
        let utterance = UtteranceCtx {
            audio: vec![],
            energy: 0.0,
            utterance_duration_ms: 0,
        };
        let mut ctx = TurnCtx::new(0, 1, &utterance);

        let err = run_llm_tts(
            "hello there, how are you today",
            &session,
            &out_tx,
            &cancel,
            &cancel_notify,
            &state,
            &mut ctx,
            None,
        )
        .await
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("realtime needs a text-gen or vision model"),
            "got: {err}"
        );
    }
}
