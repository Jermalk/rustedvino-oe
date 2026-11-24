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
// Whisper's context window is 30 s. Auto-commit the utterance just below it so
// the VAD audio buffer cannot grow without bound if speech never stops. Not at
// exactly 30 s: whisper-large-v3-turbo (GenAI 2026.4.1) returns 1-2 words for a
// clip of exactly 480 000 samples, while 29.0-29.99 s clips transcribe in full,
// so the committed turn was dropped.
const MAX_UTTERANCE_SAMPLES: usize = PCM_SAMPLE_RATE as usize * 59 / 2; // 29.5 s = 472 000 samples ≈ 944 KB

// Trim conversation history to this many user+assistant turn pairs.
// Bounds both memory and the LLM prompt length for long sessions.
const MAX_HISTORY_TURNS: usize = 20;

// Client-supplied history cap — prevents unbounded memory from a history replay.
// 200 entries covers a very long multi-session conversation; embedding retrieval
// keeps LLM context cost bounded regardless of pool size.
const MAX_CLIENT_HISTORY: usize = 200;

/// Keeps the newest [`MAX_CLIENT_HISTORY`] entries of a retrieval pool. Applied
/// wherever the pool grows: `config.history`, `inject_history`, `reload_history`
/// and every completed turn (0.9.1 Q5: 0.9.0 skipped the cap for turns, so a long
/// session grew the pool, and the admin snapshot copy of it, without bound).
fn cap_retrieval_pool(entries: &mut Vec<ClientHistoryEntry>) {
    if entries.len() > MAX_CLIENT_HISTORY {
        entries.drain(..entries.len() - MAX_CLIENT_HISTORY);
    }
}

// Embedding-based retrieval tuning.
const MEMORY_SIM_THRESHOLD: f32 = 0.5;
const MEMORY_TOP_K: usize = 3;

// Barge-in: responses synthesised when the user utters the configured phrase.
// Rotated by turn_id % len so the response varies across turns.
const STOP_RESPONSES: &[&str] = &["Calming down.", "Going shut up mode."];

// The idle timeout (no committed utterance; an inject/reload batch also
// restarts it) is `AppState::realtime_idle_timeout`,
// default `realtime_types::REALTIME_IDLE_TIMEOUT` (5 min).

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
        /// TTS model id (e.g. `"speecht5-tts-ov"`). Absent → keep the current
        /// choice (the resolver picks a loaded TTS model when none was named).
        /// `""` → speech off for the session (0.9.1 Q11).
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
    /// plain echo of what was requested. `tts_source` can also be `"off"`:
    /// the client sent `"tts_model": ""` (0.9.1 Q11).
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
        /// "requested"` (a granted request has none, even with a background
        /// load: watch `model_loading` for that) — e.g.
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
    /// The client turned speech off (`"tts_model": ""`, 0.9.1 Q11). Kept apart
    /// from `tts_model`, which holds the *resolved* id: the resolver also
    /// stores `""` when no TTS model could be chosen, and that must not read
    /// as "off" on the next `config`.
    tts_off: bool,
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
    /// D2: the ids this session has registered in the realtime serving set.
    /// Changed only by [`swap_serving`], under the session lock, so a `config`
    /// and a C1b clear cannot both deregister the same id.
    registered_serving: Vec<String>,
    /// Set by the session teardown, in the critical section that deregisters
    /// `registered_serving`: from then on [`swap_serving`] and the C1b clear do
    /// nothing, so a late watcher cannot re-register the session's ids.
    closed: bool,
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
                // Auto-commit just inside Whisper's 30 s window to bound memory usage.
                // Commit exactly MAX_UTTERANCE_SAMPLES: anything over 30 s puts the
                // utterance on GenAI's long-form path, which on the NPU pipeline
                // has never been exercised. The rest of this frame, if speech,
                // starts the next utterance.
                if audio.len() >= MAX_UTTERANCE_SAMPLES {
                    let tail = audio.split_off(MAX_UTTERANCE_SAMPLES);
                    events.push(VadEvent::SpeechEnded(audio));
                    if is_speech && !tail.is_empty() {
                        events.push(VadEvent::SpeechStarted);
                        VadState::Speech {
                            audio: tail,
                            silence_count: 0,
                        }
                    } else {
                        VadState::Silence
                    }
                } else if is_speech {
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

/// Return `true` if `transcript` contains `phrase` as whole words after both
/// are normalised (lowercase, strip non-alphanumeric, collapse whitespace):
/// phrase "stop" matches "please stop now" but not "stopwatch" or "stopped".
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
    !norm_p.is_empty() && format!(" {norm_t} ").contains(&format!(" {norm_p} "))
}

/// `source` reported in `model_selection` when the client turned speech off.
const TTS_OFF_SOURCE: &str = "off";

/// `reason` reported in `model_selection` when no TTS model can serve the
/// session (0.9.1 C1): speech is silent, `tts_model` is `""`.
const TTS_NONE_AVAILABLE: &str = "none_available";

/// The TTS resolution for a session no TTS model can serve: `tts_model: ""`
/// (the field stays a string), `tts_source: "substituted"` (REALTIME.md: "also
/// used when nothing could be chosen"), `tts_reason: "none_available"`.
fn tts_none_available() -> crate::model_manager::RealtimeResolution {
    crate::model_manager::RealtimeResolution {
        model_id: String::new(),
        source: "substituted",
        reason: Some(TTS_NONE_AVAILABLE),
        loading: false,
    }
}

/// Resolve the TTS slot for a `config`. `off` (the client sent `"tts_model": ""`,
/// [`Session::tts_off`]) turns speech off for the session (0.9.1 Q11): no TTS
/// model is chosen or loaded, and replies are text only. Otherwise the resolver
/// runs, also for a stored `""` it produced itself, and picks a loaded TTS model
/// when none was named.
///
/// 0.9.1 C1: when the resolver names no model that
/// is loaded, loading, or being loaded for this session — nothing requested and
/// no TTS available; a configured default that is not loaded and does not fit;
/// a named registered id without room; a named unknown id — the slot resolves
/// to none ([`tts_none_available`]), so no sentence is attempted, warned about
/// or counted dropped, and a default that cannot load is not stored for the
/// session. Done here, not in the shared resolver, so the STT and LLM slots and
/// the `request_model_load` reply (whose `failed` detail is the resolver's
/// reason) are unchanged.
fn resolve_tts_slot(
    mm: &Arc<crate::model_manager::ModelManager>,
    off: bool,
    requested: Option<&str>,
) -> crate::model_manager::RealtimeResolution {
    if off {
        return crate::model_manager::RealtimeResolution {
            model_id: String::new(),
            source: TTS_OFF_SOURCE,
            reason: None,
            loading: false,
        };
    }
    let requested = requested.filter(|m| !m.is_empty());
    let res = mm.resolve_realtime_slot(RealtimeSlot::Tts, requested);
    if res.loading {
        return res;
    }
    if res.model_id.is_empty() {
        return tts_none_available();
    }
    // One read of the state (F1): a load finishing between two reads must not
    // look like "neither loaded nor loading".
    match mm.model_state(&res.model_id) {
        Some(crate::model_manager::ModelState::Ready) => res,
        // Already being loaded (by another session or `request_model_load`):
        // a loading resolution for this session too, so it gets `model_loading`
        // `started` and the load watcher (C1b covers a failure). Same values a
        // loading resolution gets from the resolver today.
        Some(crate::model_manager::ModelState::Loading) => {
            let id = res.model_id.as_str();
            let (source, reason) = if requested == Some(id) {
                ("requested", None)
            } else if mm.slot_default(RealtimeSlot::Tts).as_deref() == Some(id) {
                ("default", Some("not_resident_no_headroom"))
            } else {
                // Another model picked for the session: keep the resolver's labels.
                (res.source, res.reason)
            };
            crate::model_manager::RealtimeResolution {
                source,
                reason,
                loading: true,
                model_id: res.model_id,
            }
        }
        _ => tts_none_available(),
    }
}

/// A stored model id as a resolver request: `""` is no request.
fn non_empty(id: Option<&String>) -> Option<&str> {
    id.map(String::as_str).filter(|m| !m.is_empty())
}

/// The model ids a session depends on, for the D2 serving set. An empty id
/// (speech off, or a slot nothing could fill) names no model.
fn serving_ids(s: &Session) -> Vec<String> {
    [
        s.stt_model.clone(),
        s.llm_model.clone(),
        s.tts_model.clone(),
    ]
    .into_iter()
    .flatten()
    .filter(|m| !m.is_empty())
    .collect()
}

/// D2: make the serving set match what `s` depends on now ([`serving_ids`]):
/// deregister what it registered before, register the current ids, and record
/// them. Called with the session lock held (the voice-pin calls take a short
/// std lock and never await), so the old/new diff and the counts stay in step.
///
/// Registers the new set before deregistering the old one, so an id in both
/// never drops to zero in between. Does nothing once the session is closed.
fn swap_serving(s: &mut Session, voice_pin: &crate::voice_pin::VoicePinManager) {
    if s.closed {
        return;
    }
    let now = serving_ids(s);
    voice_pin.register_serving(now.iter().map(String::as_str));
    let before = std::mem::replace(&mut s.registered_serving, now);
    voice_pin.deregister_serving(before.iter().map(String::as_str));
}

/// D2 at session teardown: mark the session closed and deregister what it
/// registered, in one critical section (the caller holds the session lock), so
/// a C1b clear that runs later is a no-op instead of re-registering its ids.
fn close_serving(s: &mut Session, voice_pin: &crate::voice_pin::VoicePinManager) {
    s.closed = true;
    let registered = std::mem::take(&mut s.registered_serving);
    voice_pin.deregister_serving(registered.iter().map(String::as_str));
}

/// Whether a session's model set may become the voice-flow pin: every id set.
fn pinnable_voice_flow(stt: &str, llm: &str, tts: &str) -> bool {
    !stt.is_empty() && !llm.is_empty() && !tts.is_empty()
}

/// What a trimmed STT transcript is, for the turn that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TranscriptKind {
    /// Nothing was transcribed.
    Empty,
    /// The configured barge-in phrase matched.
    BargeIn,
    /// Fewer than [`MIN_TRANSCRIPT_WORDS`] words: Whisper hallucinates short
    /// phrases ("Thank you.", "Hmm.") from ambient noise, so these are dropped.
    TooShort,
    /// A content turn for the LLM.
    Content,
}

/// Content turns need at least this many words (see [`TranscriptKind::TooShort`]).
const MIN_TRANSCRIPT_WORDS: usize = 3;

/// Classify a trimmed transcript. The barge-in phrase is checked before the
/// word-count filter, so a one- or two-word phrase can match (Q8a).
fn classify_transcript(transcript: &str, barge_in_phrase: Option<&str>) -> TranscriptKind {
    if transcript.is_empty() {
        TranscriptKind::Empty
    } else if barge_in_phrase.is_some_and(|p| is_barge_in_phrase(transcript, p)) {
        TranscriptKind::BargeIn
    } else if transcript.split_whitespace().count() < MIN_TRANSCRIPT_WORDS {
        TranscriptKind::TooShort
    } else {
        TranscriptKind::Content
    }
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

/// The immediate answer to a `request_model_load` (0.9.1 Q3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoadReply {
    /// The model is loaded now.
    Ready,
    /// A load of the model is running; the watcher reports the end.
    Started,
    /// The model cannot be used; the reason code goes in `detail`.
    Failed(&'static str),
}

/// Resolves `request_model_load` for `model_id` and decides the reply.
///
/// An id the server does not know is refused before the resolver runs: the
/// resolver would otherwise substitute (or start loading the slot default) and
/// the reply would carry that substitution's reason instead of `unknown_model`.
fn request_model_load_reply(
    mm: &Arc<crate::model_manager::ModelManager>,
    model_id: &str,
) -> LoadReply {
    let Some(kind) = mm.known_model_kind(model_id) else {
        return LoadReply::Failed("unknown_model");
    };
    let slot = match kind {
        ModelKind::Stt => RealtimeSlot::Stt,
        ModelKind::Tts => RealtimeSlot::Tts,
        ModelKind::Embedding => RealtimeSlot::Embed,
        _ => RealtimeSlot::Llm,
    };
    let resolution = mm.resolve_realtime_slot(slot, Some(model_id));
    model_load_reply(
        model_id,
        &resolution,
        mm.is_ready(model_id),
        mm.is_loading(model_id),
    )
}

/// Decides the reply to `request_model_load` for a known `requested` model.
///
/// The resolver's fallback steps name the requested id even when nothing is
/// loaded or loading (to report what the session waits on), so "the resolver
/// named it" alone is not "ready": 0.9.0 answered `ready` there, for a default
/// that does not fit. `ready` now needs `ready_now`, the model's real state, and
/// a model already loading is `started` even when the resolver, which counts its
/// reservation against a second copy, would have substituted.
fn model_load_reply(
    requested: &str,
    resolution: &crate::model_manager::RealtimeResolution,
    ready_now: bool,
    loading_now: bool,
) -> LoadReply {
    const CANNOT_FIT: &str = "not_resident_no_headroom";
    if resolution.model_id == requested && resolution.loading {
        return LoadReply::Started;
    }
    if ready_now {
        return LoadReply::Ready;
    }
    if loading_now {
        return LoadReply::Started;
    }
    // Not loaded and not loading: say why rather than load something else on
    // the client's behalf (that is `config`'s job).
    LoadReply::Failed(resolution.reason.unwrap_or(CANNOT_FIT))
}

/// D8: watches `model_id` until it becomes `Ready` (or the wait times out)
/// and emits the matching `model_loading` completion event. Polling, not
/// event-driven — `ModelManager` has no load-completion notification
/// primitive; a 200ms poll is cheap and this only runs while a background
/// load from `resolve_realtime_llm`/`request_model_load` is genuinely in
/// flight, never per-turn. `mm.is_ready` is a plain `HashMap` read behind a
/// short-held `RwLock`, so polling it does not contend with the load itself.
///
/// The watcher stops when `shutdown` is cancelled (session end, 0.9.1 Q10):
/// it holds an `out_tx` clone, and the sender task — so the socket — lives
/// until every clone is dropped.
///
/// `tts_clear` (0.9.1 C1b) is set for the session's TTS load from a `config`:
/// once the load is seen to have failed (from the second poll on, the model is
/// neither ready nor loading), the session's TTS id is cleared to none
/// ([`clear_failed_tts`]), so later sentences are not attempted, warned about
/// or counted dropped. The events are unchanged: `failed` still comes when the
/// watch window is up (REALTIME.md), and no second `model_selection` is sent.
fn spawn_model_ready_watcher(
    mm: Arc<crate::model_manager::ModelManager>,
    model_id: String,
    out_tx: OutTx,
    shutdown: tokio_util::sync::CancellationToken,
    tts_clear: Option<TtsLoadFailClear>,
) {
    const POLL_INTERVAL: Duration = Duration::from_millis(200);
    // Cold GPU load ceiling — matches the supervisor's own
    // `health_timeout_secs` default (90s) rather than inventing a new number.
    const WATCH_TIMEOUT: Duration = Duration::from_secs(90);
    tokio::spawn(async move {
        let start = Instant::now();
        let mut tts_clear = tts_clear;
        loop {
            // One read per poll (F1): "ready" and "failed" come from the same state.
            let state = mm.model_state(&model_id);
            // C1b: not on the first poll, where the spawned load may not have
            // marked the record `Loading` yet.
            if start.elapsed() >= POLL_INTERVAL
                && !matches!(
                    state,
                    Some(
                        crate::model_manager::ModelState::Ready
                            | crate::model_manager::ModelState::Loading
                    )
                )
                && let Some(clear) = tts_clear.take()
            {
                clear_failed_tts(&clear, &model_id).await;
            }
            if state == Some(crate::model_manager::ModelState::Ready) {
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
            tokio::select! {
                () = shutdown.cancelled() => return,
                () = tokio::time::sleep(POLL_INTERVAL) => {}
            }
        }
    });
}

/// What [`spawn_model_ready_watcher`] needs to clear a session's TTS id when
/// the TTS load it watches fails (0.9.1 C1b).
struct TtsLoadFailClear {
    session: Arc<tokio::sync::Mutex<Session>>,
    session_snap: Arc<RwLock<SessionSnapshot>>,
    voice_pin: Arc<crate::voice_pin::VoicePinManager>,
    conn_id: u64,
}

/// C1b: the TTS load for `model_id` failed after the slot resolved to it with
/// `loading: true` (C1 case 5). If the session still uses `model_id` (a later
/// `config` may have changed it), its TTS id becomes `""` — the same effect as
/// a `none_available` resolution from then on — in the session and the admin
/// snapshot; the id leaves the D2 serving set (the next `config` computes what
/// to deregister from the session, so it would never be removed otherwise);
/// and one WARN names the model and the session. Returns whether it cleared
/// (never for a closed session).
async fn clear_failed_tts(clear: &TtsLoadFailClear, model_id: &str) -> bool {
    const MSG: &str =
        "realtime: TTS model failed to load — speech off for this session until the next config";
    {
        let mut s = clear.session.lock().await;
        if s.closed || s.tts_model.as_deref() != Some(model_id) {
            return false;
        }
        s.tts_model = Some(String::new());
        swap_serving(&mut s, &clear.voice_pin);
    }
    let session_id = clear.session_snap.write().ok().map(|mut snap| {
        snap.tts_model = Some(String::new());
        snap.id
    });
    if let Some(session_id) = session_id {
        tracing::warn!(conn_id = clear.conn_id, %session_id, model = %model_id, "{MSG}");
    } else {
        tracing::warn!(conn_id = clear.conn_id, model = %model_id, "{MSG}");
    }
    #[cfg(test)]
    if let Ok(mut seen) = TTS_CLEARS.lock() {
        *seen.entry(model_id.to_owned()).or_insert(0) += 1;
    }
    true
}

/// Completed C1b clears by [`clear_failed_tts`] (each logs its one WARN), by
/// model id. A test that counts uses a model id of its own (tests run in
/// parallel); a thread-local log capture races with tracing's process-wide
/// callsite cache, so the clears are counted instead of log lines.
#[cfg(test)]
static TTS_CLEARS: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<String, usize>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

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
    embed_model: Option<&str>,
) -> (String, String) {
    // The cancel reset is the caller's (`begin_turn`, in the session loop), so
    // it can refuse to start a turn once the client is gone (0.9.1 Q9).

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
        .zip(session.stt_model.as_deref().filter(|m| !m.is_empty()))
        .map(|(mm, m)| mm.record_device(m));
    let llm_device = mm_ref
        .zip(session.llm_model.as_deref().filter(|m| !m.is_empty()))
        .map(|(mm, m)| mm.record_device(m));
    let tts_device = mm_ref
        .zip(session.tts_model.as_deref().filter(|m| !m.is_empty()))
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
            let message = gpu_poison_client_message(state, session.stt_model.as_deref(), &e)
                .unwrap_or_else(|| e.to_string());
            send_event(
                out,
                &ServerEvent::Error {
                    stage: "asr",
                    message: &message,
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
    let kind = classify_transcript(&transcript, session.barge_in_phrase.as_deref());
    if matches!(kind, TranscriptKind::Empty | TranscriptKind::TooShort) {
        ctx.transcript_empty = true;
        tracing::debug!(
            conn_id = ctx.conn_id,
            turn_id = ctx.turn_id,
            transcript = %transcript,
            empty = kind == TranscriptKind::Empty,
            "realtime: transcript empty or too short (likely hallucination), skipping turn"
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

    // ── Barge-in ─────────────────────────────────────────────────────────────
    // Classified before the 3-word filter (Q8a), so a short phrase such as
    // "stop" still matches. Runs before the cancel check and before the
    // Transcript event (barge-in is a control interaction, not a content turn —
    // the client gets Stopped instead of Transcript+Done).
    if kind == TranscriptKind::BargeIn {
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
        embed_model,
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
            let message = stalled_client_message(&e)
                .or_else(|| gpu_poison_client_message(state, session.llm_model.as_deref(), &e))
                .unwrap_or_else(|| e.to_string());
            send_event(
                out,
                &ServerEvent::Error {
                    stage: "llm",
                    message: &message,
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

/// A GPU out-of-resources failure inside a realtime turn poisons the `OpenCL` context for the whole
/// process, exactly as on the HTTP routes: flag it so the supervisor restarts the worker at once
/// (instead of the server serving a dead GPU until the next request trips over it), and return the
/// message to send the client. The raw C++ diagnostics are never put on the wire for it (T6.2).
/// The client text for an LLM turn ended by the engine stall watchdog (0.9.1 S1, review F7): a
/// plain sentence, not the internal marker.
fn stalled_client_message(err: &anyhow::Error) -> Option<String> {
    crate::stall_watch::is_generation_stalled_error(&err.to_string())
        .then(|| REALTIME_STALLED_MESSAGE.to_owned())
}

/// What a realtime client is told when the model's engine stalled during its turn.
const REALTIME_STALLED_MESSAGE: &str =
    "the model stopped responding and was taken out of service; the server needs a restart";

fn gpu_poison_client_message(
    state: &AppState,
    model: Option<&str>,
    err: &anyhow::Error,
) -> Option<String> {
    let text = err.to_string();
    if !crate::model_manager::is_gpu_poison_error(&text) {
        return None;
    }
    if let Some(mm) = state.model_manager.as_ref() {
        mm.mark_gpu_poisoned(model.unwrap_or("unknown"), err);
    }
    // G1 (step 7): "restarting — retry shortly" only when a supervised worker really restarts.
    Some(super::error::gpu_poisoned_reply_now().0.to_owned())
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
    // FM-3: the same normalisation as the HTTP routes (`pl-PL` -> `pl`, `auto` -> detect).
    let language = match session.language.as_deref() {
        None => None,
        Some(raw) => crate::pipelines::stt::normalize_language(raw).map_err(anyhow::Error::msg)?,
    };
    let t = handle.transcribe(wav_bytes, language, false).await?;
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
    embed_model: Option<&str>,
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
        .get_chat_context_for_request(llm_model)
        .await
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    // Retrieve relevant past context via embedding similarity (if available).
    // The handle is fetched for this lookup only and dropped right after it (Q6).
    let memories: Vec<&ClientHistoryEntry> = match embedding_handle_for(state, embed_model) {
        Some(handle) if !session.history_entries.is_empty() => {
            retrieve_memories(user_text, &session.history_entries, &handle).await
        }
        _ => Vec::new(),
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
    // 0.9.1 V5b: realtime follows the model's sampling profile (operator `sampling`
    // over its generation_config.json over greedy + penalty 1.1). A voice turn has
    // no client sampling fields and no tools, so the profile applies as is.
    let mut gen_params = GenParams {
        max_new_tokens: session.max_tokens.map_or(VOICE_MAX_TOKENS, |n| n as usize),
        ..GenParams::default()
    };
    crate::handlers::chat::apply_sampling_profile(
        &mut gen_params,
        crate::handlers::chat::ClientSampling::default(),
        false,
        &chat_ctx.sampling,
        chat_ctx.handle.is_npu(),
        llm_model,
    );
    let gen_params = gen_params;

    ctx.llm_start = Some(Instant::now());
    // 0.9.1 S1: the same engine stall verdict as the HTTP routes reaches this turn's channel; it
    // arrives as the stream's error, which the caller sends as `error{stage:"llm"}` + `done`.
    if let Some(liveness) = chat_ctx.handle.liveness() {
        liveness.register(
            &token_tx,
            &format!("realtime-{}-{}", ctx.conn_id, ctx.turn_id),
            llm_model,
            mm,
        );
    }
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
            // 0.9.1 V1-usage: the engine's reasoning count (`rustedvino_reasoning_tokens_total`)
            // uses the same split as the chat route. Thinking is disabled here, so it is
            // normally 0.
            let mut gen_params = gen_params;
            gen_params.reasoning = crate::prompt_builder::ReasoningSplit::new(
                chat_ctx.reasoning_parser,
                crate::prompt_builder::prompt_ends_in_thinking(&prompt),
            );
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
                SubmitResult::Closing => {
                    return Err(anyhow::anyhow!(
                        "LLM model is being unloaded — retry shortly"
                    ));
                }
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
            // GPU-context-poisoning crash (see `gate_vlm_admission`'s doc
            // comment in handlers/chat.rs). The boxed HTTP Response in its
            // Err isn't unpacked — wrong shape for a WS error — just mapped
            // to a plain message; the caller already turns any Err here into
            // ServerEvent::Error{stage:"llm"} + Done.
            // KNOWN TEST GAP (Fable review, 2026-08-04): no test in this file's
            // `mod tests` can currently observe this call firing or being
            // deleted — `run_llm_tts_serves_a_vision_kind_llm` loads its mock
            // model with no on-disk `config.json`, so `ModelManager` computes
            // `max_prompt_tokens = 0` (`compute_max_prompt_tokens` returns 0
            // whenever `legacy_bytes_per_token_f16` finds nothing to parse),
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
            let Ok(admitted) = crate::handlers::chat::gate_vlm_admission(
                &vlm,
                &gate_text,
                messages.len(),
                None,
                false,
                (chat_ctx.max_prompt_tokens, chat_ctx.gate_detail.as_ref()),
            )
            .await
            else {
                return Err(anyhow::anyhow!(
                    "prompt too long for model '{llm_model}' — exceeds its context window"
                ));
            };
            let mut gen_params = gen_params;
            let (max_new_tokens, transient_lease) = vlm_clamp_and_lease(
                mm,
                llm_model,
                (
                    chat_ctx.kv_gate_mode == "exact",
                    chat_ctx.pool_capacity_tokens,
                ),
                admitted,
                gen_params.max_new_tokens,
            )?;
            gen_params.max_new_tokens = max_new_tokens;
            match vlm
                .generate_with_lease(
                    vlm_messages,
                    vec![],
                    None,
                    Some(voice_vlm_template_context()),
                    gen_params,
                    token_tx,
                    Instant::now(),
                    transient_lease,
                    admitted,
                )
                .await
            {
                SubmitResult::Submitted => {}
                SubmitResult::AtCapacity => return Err(anyhow::anyhow!("LLM engine at capacity")),
                SubmitResult::Closing => {
                    return Err(anyhow::anyhow!(
                        "LLM model is being unloaded — retry shortly"
                    ));
                }
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

/// End a cancelled turn: cancel telemetry and `done`. Returns an empty reply,
/// so the caller records nothing: a cancelled exchange does not enter the
/// history, and no `turn_history` is sent (0.9.1 Q1).
///
/// `cancel_stage` is the telemetry field: `"llm"` or `"tts"`.
async fn cancelled_done(
    out: &OutTx,
    ctx: &mut TurnCtx,
    cancel_stage: &'static str,
) -> anyhow::Result<String> {
    ctx.cancel_stage = cancel_stage;
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
    Ok(String::new())
}

/// Where final reply text goes: the client, and the session's TTS.
struct Emit<'a> {
    out: &'a OutTx,
    session: &'a Session,
    state: &'a AppState,
    cancel: &'a AtomicBool,
    cancel_notify: &'a Notify,
}

/// Count one content piece of the reply: TTFT at the first, `llm_tokens` for each. Called as a
/// piece arrives, also while gpt-oss's channel prefix is still buffered (C4), so TTFT is the first
/// content piece and buffered pieces are not undercounted.
fn note_content_piece(ctx: &mut TurnCtx) {
    if ctx.llm_tokens == 0 {
        ctx.llm_ttft_ms = ctx.llm_start.map_or(0, |t| to_ms(t.elapsed()));
    }
    ctx.llm_tokens += 1;
}

/// Send one final piece of reply text: a `text` event, the full text, and the sentence splitter
/// (a finished sentence goes to TTS). One path for the streaming arm and for what is flushed at
/// `Done`, so a reply that ends inside a buffer is sent as text too (C4: a gpt-oss reply of 10
/// bytes or less was spoken but never sent). `true` when the sentence cap ended the turn.
async fn emit_content(
    text: &str,
    full_text: &mut String,
    splitter: &mut SentenceSplitter,
    emit: &Emit<'_>,
    ctx: &mut TurnCtx,
) -> bool {
    if text.is_empty() {
        return false;
    }
    send_event(emit.out, &ServerEvent::Text { content: text }).await;
    full_text.push_str(text);
    if let Some(sentence) = splitter.push(text)
        && !emit.cancel.load(Ordering::Acquire)
    {
        synthesize_and_send(
            &sentence,
            emit.session,
            emit.out,
            emit.state,
            ctx,
            emit.cancel,
            emit.cancel_notify,
        )
        .await;
        return sentence_cap_reached(emit.session, ctx);
    }
    false
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
    let emit = Emit {
        out,
        session,
        state,
        cancel,
        cancel_notify,
    };
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
            let at = if ctx.tts_sentences > 0 { "tts" } else { "llm" };
            return cancelled_done(out, ctx, at).await;
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
                            // Counted as it arrives, also while gpt-oss's prefix is buffered (C4).
                            note_content_piece(ctx);
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
                            if emit_content(&text, &mut full_text, &mut splitter, &emit, ctx).await
                            {
                                return early_done(out, ctx, full_text).await;
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
                // Its pieces were counted as they arrived; sent like any other content (C4).
                if let Some(buf) = prefix_buf.take() {
                    let (clean, _) = strip_gpt_oss_channel_prefix(&buf);
                    if emit_content(clean, &mut full_text, &mut splitter, &emit, ctx).await {
                        return early_done(out, ctx, full_text).await;
                    }
                }
                for piece in think_filter.flush() {
                    match piece {
                        ThinkPiece::Content(text) => {
                            note_content_piece(ctx);
                            if emit_content(&text, &mut full_text, &mut splitter, &emit, ctx).await
                            {
                                return early_done(out, ctx, full_text).await;
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
                // A cancel that landed while the last sentences were spoken still
                // cancels the turn: the client stopped the reply (0.9.1 Q1). The LLM
                // had finished, so the stage is `tts` even with speech off.
                // KNOWN TEST GAP (Fable review, 2026-11-07): no test reaches this
                // check — with no TTS seam, nothing yields between `Done` and here
                // for a test to set the flag in. Verified by reading.
                if cancel.load(Ordering::Acquire) {
                    return cancelled_done(out, ctx, "tts").await;
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
            Some(
                StreamEvent::PromptTokens(_)
                | StreamEvent::CompletionTokens(_)
                | StreamEvent::ReasoningTokens(_),
            ) => {}
            Some(StreamEvent::Error(e)) => return Err(anyhow::anyhow!("{e}")),
            // The engine closed the stream without a final event: it dropped a
            // request whose channel was full, or its terminal pair was lost to a
            // full channel. The reply is cut off, so this is an LLM error (the
            // caller sends `error` + `done` and records no turn), not a finish
            // (0.9.1 Q4; 0.9.0 sent no `done` and kept the cut-off text).
            // A cancel that landed while `recv` was pending wins: the loop top
            // takes the cancelled path (its telemetry, `done`, `result=cancelled`).
            None if cancel.load(Ordering::Acquire) => {}
            None => {
                return Err(anyhow::anyhow!(
                    "the LLM stream ended without a finish; the reply was cut off"
                ));
            }
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
    // Unicode spaces become plain ones before anything reads the text (0.9.1 C6). Belt and
    // braces: `normalize_for_speech` already folds all whitespace to plain spaces, so this
    // keeps the mapping when that internal detail changes.
    let text = crate::tts_normalize::plain_spaces(text);
    let normalized = crate::tts_normalize::normalize_for_speech(&text, lang);
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
    // `""` is speech off (0.9.1 Q11): no TTS call, no dropped-sentence count.
    let Some(tts_model) = session.tts_model.as_deref().filter(|m| !m.is_empty()) else {
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
        s.tts_off = v.is_empty();
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
        cap_retrieval_pool(&mut entries);
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

/// Texts per embedding batch sent by the realtime session (pool back-fill,
/// `inject_history`, `reload_history`). `EmbeddingHandle::embed` has no input or
/// token cap of its own (those live in the `/v1/embeddings` handler), so a pool of
/// 200 long entries is never sent as one padded batch.
const EMBED_CHUNK_TEXTS: usize = 32;

/// Turns the pool back-fill is skipped for after one of its batches failed.
const BACKFILL_BACKOFF_TURNS: u64 = 5;

/// Embeds `texts` in chunks of [`EMBED_CHUNK_TEXTS`], in order. Stops at the first
/// chunk that fails (logged): returns the vectors of the chunks before it (a prefix
/// of `texts`) and `false`; `true` when every chunk succeeded.
async fn embed_in_chunks(
    handle: &crate::embed_engine::EmbeddingHandle,
    texts: Vec<String>,
    conn_id: u64,
) -> (Vec<Vec<f32>>, bool) {
    let total = texts.len();
    let mut vectors = Vec::with_capacity(total);
    for chunk in texts.chunks(EMBED_CHUNK_TEXTS) {
        match handle.embed(chunk.to_vec()).await {
            Ok(output) if output.vectors.len() == chunk.len() => vectors.extend(output.vectors),
            Ok(output) => {
                tracing::warn!(
                    conn_id,
                    model = handle.model_id(),
                    expected = chunk.len(),
                    got = output.vectors.len(),
                    "realtime: embedding batch returned the wrong number of vectors"
                );
                return (vectors, false);
            }
            Err(e) => {
                tracing::warn!(
                    conn_id,
                    model = handle.model_id(),
                    error = ?e,
                    embedded = vectors.len(),
                    total,
                    "realtime: embedding batch failed; the rest of the history stays unembedded"
                );
                return (vectors, false);
            }
        }
    }
    (vectors, true)
}

/// `true` when `entry` must be embedded by the session's model `model`: it has no
/// vector, or its vector comes from another model (vectors from different models
/// cannot be compared, so retrieval skips such an entry until it is re-embedded).
/// A vector with no recorded model is kept as it is.
fn needs_embedding(entry: &ClientHistoryEntry, model: &str) -> bool {
    entry.embedding.is_none() || entry.embedding_model.as_deref().is_some_and(|m| m != model)
}

/// Which pool entry a back-fill vector belongs to: its position when the batch was
/// read, plus enough of the entry to tell whether it is still there.
struct BackfillKey {
    index: usize,
    user: String,
    assistant: String,
    timestamp: u64,
}

/// Applies back-fill `vectors` from `model` to `pool`, one per key, and returns the
/// entries that got one (for the `turn_history` echo). A key whose position now
/// holds another entry, or one that no longer needs embedding, is skipped: the pool
/// can be replaced (`config.history`) while the batch is being embedded.
fn apply_backfill(
    pool: &mut [ClientHistoryEntry],
    keys: &[BackfillKey],
    vectors: Vec<Vec<f32>>,
    model: &str,
) -> Vec<ClientHistoryEntry> {
    keys.iter()
        .zip(vectors)
        .filter_map(|(key, vector)| {
            let entry = pool.get_mut(key.index)?;
            let same = entry.user == key.user
                && entry.assistant == key.assistant
                && entry.timestamp == key.timestamp;
            if !same || !needs_embedding(entry, model) {
                return None;
            }
            entry.embedding = Some(vector);
            entry.embedding_model = Some(model.to_owned());
            Some(entry.clone())
        })
        .collect()
}

/// Embeds the pool entries that [`needs_embedding`] for `handle`'s model, then echoes
/// each one as a `turn_history` with `turn_id: null` so the client can store the
/// vector. Runs at connect and at every turn start (0.9.1 Q7); a scan of at most
/// [`MAX_CLIENT_HISTORY`] entries when nothing is missing.
///
/// Returns `false` when an embedding batch failed (the entries before it are still
/// applied), so the caller can back off instead of paying a failing batch every turn.
async fn backfill_embeddings(
    session: &Arc<tokio::sync::Mutex<Session>>,
    session_snap: &Arc<RwLock<SessionSnapshot>>,
    out: &OutTx,
    handle: &crate::embed_engine::EmbeddingHandle,
    conn_id: u64,
) -> bool {
    let model = handle.model_id().to_owned();
    let (texts, keys): (Vec<String>, Vec<BackfillKey>) = {
        let guard = session.lock().await;
        guard
            .history_entries
            .iter()
            .enumerate()
            .filter(|(_, e)| needs_embedding(e, &model))
            .map(|(index, e)| {
                (
                    format!("{}\n{}", e.user, e.assistant),
                    BackfillKey {
                        index,
                        user: e.user.clone(),
                        assistant: e.assistant.clone(),
                        timestamp: e.timestamp,
                    },
                )
            })
            .unzip()
    };
    if texts.is_empty() {
        return true;
    }
    let (vectors, complete) = embed_in_chunks(handle, texts, conn_id).await;
    let (embedded, pool) = {
        let mut guard = session.lock().await;
        let embedded = apply_backfill(&mut guard.history_entries, &keys, vectors, &model);
        (embedded, guard.history_entries.clone())
    };
    if embedded.is_empty() {
        return complete;
    }
    if let Ok(mut snap) = session_snap.write() {
        snap.history_entries = pool;
    }
    for e in &embedded {
        send_event(
            out,
            &ServerEvent::TurnHistory {
                turn_id: None,
                user: &e.user,
                assistant: &e.assistant,
                embedding: e.embedding.clone(),
                timestamp: e.timestamp,
                embedding_model: Some(model.as_str()),
                stt_model: None,
                llm_model: None,
            },
        )
        .await;
    }
    complete
}

/// A handle for the session's embedding model `model`, for one use: callers drop
/// it when that use ends. `None` when there is no model, no model manager, or the
/// model is not loaded as an embedding model.
fn embedding_handle_for(
    state: &AppState,
    model: Option<&str>,
) -> Option<crate::embed_engine::EmbeddingHandle> {
    state
        .model_manager
        .as_ref()?
        .get_embedding_handle(model?)
        .ok()
}

/// Adds a completed turn to both histories and returns the retrieval pool (a copy, for the
/// admin snapshot). The pool is capped like every other way of filling it (0.9.1 Q5); the
/// rolling LLM context keeps the last [`MAX_HISTORY_TURNS`] exchanges.
fn record_completed_turn(s: &mut Session, entry: ClientHistoryEntry) -> Vec<ClientHistoryEntry> {
    let (user, assistant) = (entry.user.clone(), entry.assistant.clone());
    s.history_entries.push(entry);
    cap_retrieval_pool(&mut s.history_entries);
    // Trim the oldest turn pair when the LLM context cap is reached.
    if s.history.len() >= MAX_HISTORY_TURNS * 2 {
        s.history.drain(..2);
    }
    s.history.push(RtMessage {
        role: "user".to_owned(),
        content: user,
    });
    s.history.push(RtMessage {
        role: "assistant".to_owned(),
        content: assistant,
    });
    s.history_entries.clone()
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
            .filter(|(_, e)| needs_embedding(e, handle.model_id()))
            .map(|(i, e)| (format!("{}\n{}", e.user, e.assistant), i))
            .unzip();
        if !texts.is_empty() {
            let vectors = embed_in_chunks(handle, texts, conn_id).await.0;
            for (&idx, vec) in indices.iter().zip(vectors) {
                if let Some(e) = entries.get_mut(idx) {
                    e.embedding = Some(vec);
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
        cap_retrieval_pool(&mut guard.history_entries);
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
            .filter(|(_, e)| needs_embedding(e, handle.model_id()))
            .map(|(i, e)| (format!("{}\n{}", e.user, e.assistant), i))
            .unzip();
        if !texts.is_empty() {
            let vectors = embed_in_chunks(handle, texts, conn_id).await.0;
            for (&idx, vec) in indices.iter().zip(vectors) {
                if let Some(e) = entries.get_mut(idx) {
                    e.embedding = Some(vec);
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
        cap_retrieval_pool(&mut guard.history_entries);
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
///
/// Returns the frame stream, so the session teardown can read the client's
/// close reply after a server-initiated close (0.9.1 C5) — and, after a close
/// the client began, let the WebSocket flush its automatic close reply.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
async fn run_receiver<S>(
    // Generic over the frame stream (not the concrete `SplitStream<WebSocket>`)
    // so unit tests can drive it with a mock stream. The real call site passes a
    // `SplitStream<WebSocket>`, which satisfies these bounds.
    mut stream: S,
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
) -> S
where
    S: futures_util::Stream<Item = Result<WsMsg, axum::Error>> + Unpin + Send,
{
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
                            // The session loop resets the cancel flag at turn start
                            // (`begin_turn`) — the receiver only ever sets it to true.
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
                    let (requested_stt, requested_llm, requested_tts, tts_off) = {
                        let s = session.lock().await;
                        (
                            s.stt_model.clone(),
                            s.llm_model.clone(),
                            s.tts_model.clone(),
                            s.tts_off,
                        )
                    };
                    // A stored `""` (the resolver found nothing last time) is no
                    // request: resolving it as one would report a reason such as
                    // `would_evict_protected` for a model nobody asked for.
                    let stt_resolution = model_manager.as_ref().map(|mm| {
                        mm.resolve_realtime_slot(
                            RealtimeSlot::Stt,
                            non_empty(requested_stt.as_ref()),
                        )
                    });
                    let llm_resolution = model_manager.as_ref().map(|mm| {
                        mm.resolve_realtime_slot(
                            RealtimeSlot::Llm,
                            non_empty(requested_llm.as_ref()),
                        )
                    });
                    let tts_resolution = model_manager
                        .as_ref()
                        .map(|mm| resolve_tts_slot(mm, tts_off, requested_tts.as_deref()));
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
                        // D2: swap the realtime serving set — deregister what
                        // this session registered before, register what it
                        // depends on now (resolved values, post-D5), in the
                        // same critical section as the write (F5).
                        swap_serving(&mut s, &voice_pin);
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

                    if let Some(ref llm_res) = llm_resolution {
                        for (resolution, is_tts) in [
                            (stt_resolution.as_ref(), false),
                            (Some(llm_res), false),
                            (tts_resolution.as_ref(), true),
                        ]
                        .into_iter()
                        .filter_map(|(r, is_tts)| r.map(|r| (r, is_tts)))
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
                                // C1b: a failed TTS load clears the session's TTS id.
                                let tts_clear = is_tts.then(|| TtsLoadFailClear {
                                    session: Arc::clone(&session),
                                    session_snap: Arc::clone(&session_snap),
                                    voice_pin: Arc::clone(&voice_pin),
                                    conn_id,
                                });
                                spawn_model_ready_watcher(
                                    Arc::clone(mm),
                                    resolution.model_id.clone(),
                                    out_tx.clone(),
                                    shutdown.clone(),
                                    tts_clear,
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
                    let reply = request_model_load_reply(mm, &model_id);
                    let (stage, detail) = match reply {
                        LoadReply::Ready => ("ready", None),
                        LoadReply::Started => ("started", None),
                        LoadReply::Failed(why) => ("failed", Some(why)),
                    };
                    send_event(
                        &out_tx,
                        &ServerEvent::ModelLoading {
                            model_id: &model_id,
                            stage,
                            detail,
                        },
                    )
                    .await;
                    if reply == LoadReply::Started {
                        spawn_model_ready_watcher(
                            Arc::clone(mm),
                            model_id.clone(),
                            out_tx.clone(),
                            shutdown.clone(),
                            None,
                        );
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
    // The client is gone (close, error, end of stream) or teardown began: end
    // the session (0.9.1 Q9). `shutdown` first, then `cancel` — `begin_turn`
    // relies on that order, so a turn about to start sees one or the other.
    // Queued utterances are then never run, and the in-flight turn aborts —
    // same path as the explicit Cancel event.
    shutdown.cancel();
    cancel.store(true, Ordering::Release);
    cancel_notify.notify_waiters();
    stream
}

/// Clear the per-turn `cancel` flag for the turn about to start, unless the
/// session is ending. Returns `false` (and leaves `cancel` set) once
/// `shutdown` is cancelled: the client hung up, an admin kill landed, or
/// teardown began (0.9.1 Q9). The receiver cancels `shutdown` before it sets
/// `cancel`. If the swap reads the receiver's `true`, the acquire makes the
/// cancelled `shutdown` visible; if it reads an older value, the receiver's
/// `true` lands after the swap and the turn aborts at its first cancel check.
fn begin_turn(cancel: &AtomicBool, shutdown: &tokio_util::sync::CancellationToken) -> bool {
    cancel.swap(false, Ordering::AcqRel);
    if shutdown.is_cancelled() {
        cancel.store(true, Ordering::Release);
        return false;
    }
    true
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
/// The same bridge carries a server stop (0.9.1 C5): `realtime_session`
/// spawns it a second time with the session's `stopping` token as `kill`, so a
/// graceful shutdown aborts the in-flight turn and the 1001 close goes out
/// while the process is still draining.
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
    // Exits when all out_tx clones are dropped (or a send fails — e.g. after a
    // close frame went out). Aborted after a bounded wait at teardown.
    let sender_task = tokio::spawn(async move {
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
    // response pipeline checks it each token; `begin_turn` resets it at turn start.
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
    // Graceful server shutdown (0.9.1 C5): cancelled by `stop_realtime_sessions`
    // from the shutdown path; the session then closes with 1001.
    let stopping = tokio_util::sync::CancellationToken::new();
    // `register` starts the session stopped if the server is already stopping
    // (an upgrade that raced the stop signal, 0.9.1 C5).
    if let Ok(mut reg) = state.realtime_sessions.write() {
        reg.register(
            session_id,
            crate::realtime_types::RealtimeSessionEntry {
                snapshot: Arc::clone(&session_snap),
                kill: kill.clone(),
                stopping: stopping.clone(),
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
    // C5: a server stop aborts an in-flight turn the same way, so the 1001
    // close goes out while the process is still draining, not after the turn.
    spawn_kill_bridge(
        Arc::clone(&cancel),
        Arc::clone(&cancel_notify),
        stopping.clone(),
        shutdown.clone(),
    );

    // ── Embedding model ───────────────────────────────────────────────────────
    // The session's embedding model is pinned at its first successful use and
    // never changes: vectors from different models are not comparable, so a
    // change would need the whole pool re-embedded. Only the id is kept; a
    // handle is fetched for each use and dropped after it (0.9.1 Q6). A handle
    // kept for the session kept the engine thread alive, so evicting the model
    // ran into the join timeout, and the kept handle failed (`Closing`) on every
    // later call, even after the model was loaded again.
    let embed_handle = {
        let guard = session.lock().await;
        let model = guard.embed_model.clone();
        drop(guard); // no MutexGuard held across any await
        embedding_handle_for(&state, model.as_deref())
    };
    let mut embed_pin: Option<String> = embed_handle.as_ref().map(|h| h.model_id().to_owned());

    // Back-fill at connect (also run at every turn start, below).
    if let Some(ref handle) = embed_handle {
        backfill_embeddings(&session, &session_snap, &out_tx, handle, conn_id).await;
    }

    drop(embed_handle);

    // Response pipeline (runs in this task, sequentially — one utterance at a
    // time). Receives committed audio from the VAD via audio_rx.
    let mut turn_id: u64 = 0;
    // First turn at which the pool back-fill may run again after a failure.
    let mut backfill_after_turn: u64 = 0;
    let idle_timeout = state.realtime_idle_timeout;
    // Set when the idle timeout ended the loop (0.9.1 C5: close 1000).
    let mut idle = false;
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
            () = stopping.cancelled() => {
                tracing::info!(conn_id, "realtime: server stopping — closing session");
                break;
            }
            // The client hung up (the receiver cancels `shutdown` on exit):
            // drop the queued utterances instead of running them (0.9.1 Q9).
            () = shutdown.cancelled() => {
                tracing::debug!(conn_id, "realtime: client gone — dropping queued utterances");
                break;
            }
            Some(entries) = inject_rx.recv() => {
                handle_inject_history(
                    entries,
                    &session,
                    &session_snap,
                    &out_tx,
                    embedding_handle_for(&state, embed_pin.as_deref()).as_ref(),
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
                    embedding_handle_for(&state, embed_pin.as_deref()).as_ref(),
                    conn_id,
                )
                .await;
                continue;
            }
            result = time::timeout(idle_timeout, audio_rx.recv()) => match result {
                Ok(Some(u)) => u,
                Ok(None) => break, // receiver gone; normally the `shutdown` arm wins first
                Err(_) => {
                    tracing::debug!(conn_id, "realtime: idle timeout — closing session");
                    idle = true;
                    break;
                }
            }
        };

        // Not pinned yet (the model was not loaded, or `embed_model` came from a
        // later `config`): pin it now if it is usable.
        if embed_pin.is_none() {
            let model = session.lock().await.embed_model.clone();
            if let Some(model) = model
                && embedding_handle_for(&state, Some(&model)).is_some()
            {
                embed_pin = Some(model);
            }
        }
        // Embed what the pool still lacks (0.9.1 Q7): entries from `config.history`
        // (applied by the receiver task, so the connect back-fill never saw them),
        // batches injected before the model was pinned, and entries embedded by
        // another model. Before the snapshot, so this turn's retrieval sees them.
        // After a failed batch, skip the back-fill for BACKFILL_BACKOFF_TURNS turns:
        // a persistent failure must not add its admission wait to every turn.
        if turn_id >= backfill_after_turn
            && let Some(handle) = embedding_handle_for(&state, embed_pin.as_deref())
            && !backfill_embeddings(&session, &session_snap, &out_tx, &handle, conn_id).await
        {
            backfill_after_turn = turn_id + BACKFILL_BACKOFF_TURNS;
        }

        let snapshot = {
            let guard = session.lock().await;
            guard.clone()
        };

        // The back-fill above can take a while: re-check that the client is
        // still there right before the turn starts (0.9.1 Q9).
        if !begin_turn(&cancel, &shutdown) {
            tracing::debug!(conn_id, "realtime: client gone — turn not started");
            break;
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
            embed_pin.as_deref(),
        )
        .await;

        turn_id += 1;

        // Only extend history for fully-completed (non-cancelled) turns.
        if !user_text.is_empty() && !assistant_text.is_empty() {
            // Embed the turn (user+"\n"+assistant) if an embed handle is available.
            let (embedding, embedding_model): (Option<Vec<f32>>, Option<String>) =
                if let Some(handle) = embedding_handle_for(&state, embed_pin.as_deref()) {
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
            // A set with an empty id (a text-only session, `tts_model: ""`, or a
            // slot nothing could fill) is not a voice flow another client can
            // copy: it is never pinned (0.9.1 Q11).
            if let (Some(stt), Some(llm), Some(tts)) = (
                snapshot.stt_model.as_ref(),
                snapshot.llm_model.as_ref(),
                snapshot.tts_model.as_ref(),
            ) && pinnable_voice_flow(stt, llm, tts)
                && state
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
            let new_entries = record_completed_turn(
                &mut guard,
                ClientHistoryEntry {
                    user: user_text,
                    assistant: assistant_text,
                    embedding,
                    timestamp,
                    embedding_model,
                    stt_model: snapshot.stt_model.clone(),
                    llm_model: snapshot.llm_model.clone(),
                },
            );
            drop(guard);
            if let Ok(mut snap) = session_snap.write() {
                snap.turn_count += 1;
                snap.history_entries = new_entries;
            }
        }
    }

    // C5: one close path for every exit. A server-side cause sends its close
    // frame; a client close or a dropped connection has no cause and no frame.
    let cause = crate::realtime_types::CloseCause::from_signals(
        kill.is_cancelled(),
        stopping.is_cancelled(),
        idle,
    );
    let close_deadline = time::Instant::now() + crate::realtime_types::CLOSE_REPLY_WAIT;
    if let Some(cause) = cause {
        send_close_frame(&out_tx, cause, close_deadline, conn_id).await;
    }
    tracing::info!(
        conn_id,
        turns = turn_id,
        cause = cause.map_or("client", crate::realtime_types::CloseCause::reason),
        "realtime: connection closed"
    );
    // Wake the receiver so it exits its frame loop on every break path (kill,
    // idle, stop, audio-None). Idempotent. Without this, the always-streaming
    // client keeps the receiver blocked on `stream.next()` and this await never
    // returns. It also stops the `model_loading` watchers (Q10).
    shutdown.cancel();
    // Bounded (0.9.1 C5 review F5): a receiver parked in `send_event` on a full
    // outgoing queue — the sender blocked by a client that stopped reading —
    // never sees `shutdown`. Unbounded, this await hung forever: the registry
    // entry and the voice-pin count leaked, and no close frame went out.
    let mut receiver_task = receiver_task;
    match time::timeout_at(close_deadline, &mut receiver_task).await {
        Ok(Ok(stream)) => drain_until_close(stream, close_deadline, conn_id).await,
        Ok(Err(e)) => tracing::warn!(conn_id, err = %e, "realtime: receiver task failed"),
        Err(_) => {
            receiver_task.abort();
            tracing::warn!(
                conn_id,
                wait_ms = u64::try_from(crate::realtime_types::CLOSE_REPLY_WAIT.as_millis())
                    .unwrap_or(u64::MAX),
                cause = cause.map_or("client", crate::realtime_types::CloseCause::reason),
                "realtime: receiver did not stop within the close wait — aborted"
            );
        }
    }
    // D2: release whatever this session's last Config resolution registered
    // into the realtime serving set — otherwise a closed session's models
    // would stay eviction-protected forever.
    {
        close_serving(&mut *session.lock().await, &state.voice_pin);
    }
    state.voice_pin.session_closed();
    if let Ok(mut reg) = state.realtime_sessions.write() {
        reg.remove(&session_id);
    }
    // Last `out_tx` clone (the receiver's and the watchers' are gone): the
    // sender drains what is queued and exits. Bounded — a client that stopped
    // reading must not keep the socket and the task alive.
    drop(out_tx);
    let mut sender_task = sender_task;
    if time::timeout_at(close_deadline, &mut sender_task)
        .await
        .is_err()
    {
        sender_task.abort();
    }
}

/// C5: queue the close frame for `cause`. Bounded by `deadline`: the queue is
/// full only when the client stopped reading, and then the frame cannot be
/// delivered anyway.
async fn send_close_frame(
    out_tx: &OutTx,
    cause: crate::realtime_types::CloseCause,
    deadline: time::Instant,
    conn_id: u64,
) {
    let frame = WsMsg::Close(Some(axum::extract::ws::CloseFrame {
        code: cause.code(),
        reason: cause.reason().into(),
    }));
    if !matches!(
        time::timeout_at(deadline, out_tx.send(frame)).await,
        Ok(Ok(()))
    ) {
        tracing::debug!(
            conn_id,
            reason = cause.reason(),
            "realtime: close frame not queued"
        );
    }
}

/// C5: read the remaining frames until the client's close (or the end of the
/// stream), at most until `deadline`. After a server close this is the close
/// handshake reply; after a client close, reading on lets the WebSocket flush
/// its automatic close reply. Audio still in flight is discarded.
async fn drain_until_close<S>(mut stream: S, deadline: time::Instant, conn_id: u64)
where
    S: futures_util::Stream<Item = Result<WsMsg, axum::Error>> + Unpin,
{
    loop {
        match time::timeout_at(deadline, stream.next()).await {
            Ok(Some(Ok(WsMsg::Close(_)) | Err(_)) | None) => return,
            Ok(Some(Ok(_))) => {}
            Err(_) => {
                tracing::debug!(conn_id, "realtime: no close reply within the wait");
                return;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests (no GPU / no WebSocket required)
// ─────────────────────────────────────────────────────────────────────────────

/// The protections the realtime vision path shares with the HTTP one (F3-S-6): the answer
/// may not outgrow the KV pool (`max_new_tokens` clamped by
/// [`crate::handlers::chat::clamp_vlm_max_tokens`]; no room left is an error), and a long
/// prompt takes its transient device-memory lease, which the engine thread holds until the
/// job finishes (a dropped socket does not stop a running prefill). Returns the clamped
/// `max_new_tokens` and the lease (`None` below the in-flight threshold or without headroom).
pub(crate) fn vlm_clamp_and_lease(
    mm: &crate::model_manager::ModelManager,
    llm_model: &str,
    pool: (bool, usize),
    admitted: usize,
    requested: usize,
) -> anyhow::Result<(usize, Option<Box<dyn std::any::Any + Send>>)> {
    let Ok(max_new_tokens) =
        crate::handlers::chat::clamp_vlm_max_tokens(llm_model, pool, admitted, requested)
    else {
        anyhow::bail!(
            "prompt too long for model '{llm_model}' — no room left to answer in its KV pool"
        );
    };
    let lease = mm
        .acquire_transient_lease(llm_model, admitted)
        .map(|l| Box::new(l) as Box<dyn std::any::Any + Send>);
    Ok((max_new_tokens, lease))
}

/// Chat-template context for a vision LLM on the voice path (0.9.1 C3): thinking off, as the
/// text-gen path gets through `build_prompt(.., Some(false), ..)`. `VLMPipeline` applies the
/// template itself, so the flag reaches it only as a template variable. Built server-side with
/// this one key, like the chat VLM path, so nothing can override `bos_token`/`eos_token`.
fn voice_vlm_template_context() -> serde_json::Value {
    serde_json::json!({ "enable_thinking": false })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    // ── Voice-path template context (0.9.1 C3) ───────────────────────────────

    #[test]
    fn voice_vlm_context_disables_thinking_and_carries_nothing_else() {
        assert_eq!(
            voice_vlm_template_context(),
            serde_json::json!({ "enable_thinking": false })
        );
    }

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

    /// 0.9.1 C6: the realtime TTS path maps Unicode spaces before the normaliser, so the
    /// digits before a narrow no-break space are spelled out and no odd space survives.
    #[test]
    fn sanitize_maps_unicode_spaces_before_normalising() {
        assert_eq!(
            sanitize_for_tts("It is 328\u{202f}feet\u{a0}tall.", Some("en")),
            "It is three hundred twenty eight feet tall."
        );
        assert_eq!(sanitize_for_tts("x\u{3000}y\u{205f}z", None), "x y z");
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
    fn a_short_barge_in_phrase_is_not_dropped_as_noise() {
        assert_eq!(
            classify_transcript("Stop.", Some("stop")),
            TranscriptKind::BargeIn
        );
        assert_eq!(
            classify_transcript("Ruvi, stop", Some("ruvi stop")),
            TranscriptKind::BargeIn
        );
        // Without a matching phrase, short transcripts are still noise.
        assert_eq!(classify_transcript("Stop.", None), TranscriptKind::TooShort);
        assert_eq!(
            classify_transcript("Thank you.", Some("stop")),
            TranscriptKind::TooShort
        );
        assert_eq!(classify_transcript("", Some("stop")), TranscriptKind::Empty);
        assert_eq!(
            classify_transcript("tell me a story", Some("stop")),
            TranscriptKind::Content
        );
        assert_eq!(
            classify_transcript("please stop talking now", Some("stop talking")),
            TranscriptKind::BargeIn
        );
    }

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
        // Whole words only: no match inside a longer word.
        assert!(is_barge_in_phrase("Please stop.", "stop"));
        assert!(is_barge_in_phrase("Stop", "stop"));
        assert!(!is_barge_in_phrase("Stopwatch?", "stop"));
        assert!(!is_barge_in_phrase("Unstoppable.", "stop"));
        assert!(!is_barge_in_phrase("it stopped working", "stop"));
        assert!(!is_barge_in_phrase("Now.", "no"));
        assert!(!is_barge_in_phrase("ruvi stopping", "ruvi stop"));
    }

    #[test]
    fn a_word_containing_the_phrase_is_still_noise() {
        assert_eq!(
            classify_transcript("Stopwatch.", Some("stop")),
            TranscriptKind::TooShort
        );
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

    /// Feed `deltas` then `Done` through `drain_voice_tokens` with `policy`, no TTS model; return
    /// the `text` contents sent, the returned full text and the turn context.
    async fn drain_for_test(
        deltas: &[&str],
        policy: &VoicePolicy,
    ) -> (Vec<String>, String, TurnCtx) {
        let (tx, rx) = crate::streaming::stream_channel();
        for d in deltas {
            tx.send(StreamEvent::Token((*d).to_owned(), 1))
                .await
                .unwrap();
        }
        tx.send(StreamEvent::Done(crate::ov_cb::FinishReason::Stop))
            .await
            .unwrap();
        drop(tx);
        let (out_tx, mut out_rx) = mpsc::channel(64);
        let u = UtteranceCtx {
            audio: vec![],
            energy: 0.0,
            utterance_duration_ms: 0,
        };
        let mut ctx = TurnCtx::new(0, 1, &u);
        // A start a little in the past, so a stamped TTFT reads > 0 ms.
        ctx.llm_start = Instant::now().checked_sub(std::time::Duration::from_millis(5));
        let full = drain_voice_tokens(
            rx,
            &AtomicBool::new(false),
            &Notify::new(),
            &out_tx,
            &Session::default(),
            &AppState::new(),
            &mut ctx,
            policy,
        )
        .await
        .unwrap();
        drop(out_tx);
        let mut texts = Vec::new();
        while let Some(m) = out_rx.recv().await {
            if let WsMsg::Text(t) = m {
                let v: serde_json::Value = serde_json::from_str(&t).unwrap();
                if v["type"] == "text" {
                    texts.push(v["content"].as_str().unwrap().to_owned());
                }
            }
        }
        (texts, full, ctx)
    }

    /// C4: a gpt-oss reply that never grows past the 10-byte prefix buffer ("Paris.") is sent as
    /// `text`, counted, and stamps TTFT, not only spoken.
    #[tokio::test]
    async fn a_short_gpt_oss_reply_is_sent_as_text() {
        let gpt_oss = VoicePolicy::from_parser(Some(&ReasoningParser::GptOss));
        let (texts, full, ctx) = drain_for_test(&["Par", "is."], &gpt_oss).await;
        assert_eq!(texts.concat(), "Paris.");
        assert_eq!(full, "Paris.");
        assert_eq!(ctx.llm_tokens, 2, "each content piece counts");
        assert!(ctx.llm_ttft_ms > 0, "TTFT stamped");
    }

    /// C4: a short reply that starts with the leaked channel name is sent without it.
    #[tokio::test]
    async fn a_short_gpt_oss_reply_with_the_channel_prefix_is_sent_without_it() {
        let gpt_oss = VoicePolicy::from_parser(Some(&ReasoningParser::GptOss));
        let (texts, full, ctx) = drain_for_test(&["final", "8."], &gpt_oss).await;
        assert_eq!(texts.concat(), "8.");
        assert_eq!(full, "8.");
        assert_eq!(ctx.llm_tokens, 2);
        assert!(ctx.llm_ttft_ms > 0);
    }

    /// C4: on a longer gpt-oss reply the buffered pieces count too, and the text sent equals the
    /// full text.
    #[tokio::test]
    async fn a_long_gpt_oss_reply_counts_its_buffered_pieces() {
        let gpt_oss = VoicePolicy::from_parser(Some(&ReasoningParser::GptOss));
        let (texts, full, ctx) =
            drain_for_test(&["The ", "capital ", "is ", "Paris."], &gpt_oss).await;
        assert_eq!(texts.concat(), "The capital is Paris.");
        assert_eq!(full, "The capital is Paris.");
        assert_eq!(ctx.llm_tokens, 4);
    }

    /// C4 (Fable review): TTFT is stamped at the first content piece, not when the gpt-oss prefix
    /// buffer resolves. The first piece arrives at once, the one that resolves the buffer 80 ms
    /// later; before the fix TTFT read at least 80 ms.
    #[tokio::test]
    async fn gpt_oss_ttft_is_the_first_content_piece() {
        let (tx, rx) = crate::streaming::stream_channel();
        let producer = tokio::spawn(async move {
            tx.send(StreamEvent::Token("The ".to_owned(), 1))
                .await
                .unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(80)).await;
            for d in ["capital ", "is ", "Paris."] {
                tx.send(StreamEvent::Token(d.to_owned(), 1)).await.unwrap();
            }
            tx.send(StreamEvent::Done(crate::ov_cb::FinishReason::Stop))
                .await
                .unwrap();
        });
        let (out_tx, _out_rx) = mpsc::channel(64);
        let u = UtteranceCtx {
            audio: vec![],
            energy: 0.0,
            utterance_duration_ms: 0,
        };
        let mut ctx = TurnCtx::new(0, 1, &u);
        ctx.llm_start = Some(Instant::now());
        let gpt_oss = VoicePolicy::from_parser(Some(&ReasoningParser::GptOss));
        drain_voice_tokens(
            rx,
            &AtomicBool::new(false),
            &Notify::new(),
            &out_tx,
            &Session::default(),
            &AppState::new(),
            &mut ctx,
            &gpt_oss,
        )
        .await
        .unwrap();
        producer.await.unwrap();
        assert!(ctx.llm_ttft_ms < 60, "TTFT {} ms", ctx.llm_ttft_ms);
        assert_eq!(ctx.llm_tokens, 4);
    }

    /// C4: content the think filter still holds at `Done` (a trailing `<th` that never became a
    /// tag) is sent as `text` as well (any model).
    #[tokio::test]
    async fn content_flushed_at_done_is_sent_as_text() {
        let (texts, full, ctx) = drain_for_test(&["It is 3 <th"], &VoicePolicy::default()).await;
        assert_eq!(full, "It is 3 <th", "the held tail is kept");
        assert_eq!(texts.concat(), full, "and also sent");
        assert_eq!(ctx.llm_tokens, 2, "the flushed tail counts as a piece");
    }

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

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn vad_auto_commit_is_exactly_one_window() {
        let mut vad = EnergyVad::new();
        // Value 3276 is well within i16 range; cast_possible_truncation is a false positive.
        let loud_sample: i16 = (0.1 * f32::from(i16::MAX)) as i16;
        // ~31 s of continuous speech in 1 700-sample (~106 ms) frames: 472 000 is
        // not a multiple of the frame size, so a naive commit overshoots the window.
        let frame = i16_to_bytes(&[loud_sample; 1_700]);
        let mut committed = Vec::new();
        let mut restarted = false;
        for _ in 0..(31 * 16_000 / 1_700) {
            for ev in vad.push_bytes(&frame) {
                match ev {
                    VadEvent::SpeechEnded(audio) => committed.push(audio.len()),
                    VadEvent::SpeechStarted if !committed.is_empty() => restarted = true,
                    VadEvent::SpeechStarted => {}
                }
            }
        }
        assert_eq!(committed, vec![MAX_UTTERANCE_SAMPLES]);
        assert!(restarted, "overshoot should start the next utterance");
        match vad.state {
            VadState::Speech { ref audio, .. } => {
                assert_eq!(
                    audio.len(),
                    1_700 * (31 * 16_000 / 1_700) - MAX_UTTERANCE_SAMPLES
                );
            }
            _ => panic!("expected Speech state carrying the overshoot"),
        }
    }

    #[test]
    fn vad_auto_commit_stays_clear_of_whisper_exact_window() {
        // A clip of exactly 30 s makes Whisper turbo return 1-2 words (and the
        // turn is dropped as empty); over 30 s is the long-form path. Commit at
        // 29.5 s: inside the window, with margin from the bad exact length.
        let whisper_window = PCM_SAMPLE_RATE as usize * 30;
        assert_eq!(MAX_UTTERANCE_SAMPLES, 472_000);
        assert!(whisper_window - MAX_UTTERANCE_SAMPLES >= PCM_SAMPLE_RATE as usize / 4);
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

    /// 0.9.1 Q9: when the client's stream ends, the receiver must mark the
    /// session as ending (`shutdown`) and abort the in-flight turn (`cancel`),
    /// so the session loop runs none of the queued utterances.
    #[tokio::test]
    async fn a_client_hang_up_ends_the_session_and_cancels_the_turn() {
        use std::sync::atomic::AtomicBool;

        // An empty stream: the client closed the connection.
        let stream = futures_util::stream::empty::<Result<WsMsg, axum::Error>>();
        let (out_tx, _out_rx) = mpsc::channel(1);
        let cancel = Arc::new(AtomicBool::new(false));
        let (audio_tx, _audio_rx) = mpsc::channel(1);
        let (inject_tx, _inject_rx) = mpsc::channel(1);
        let (reload_tx, _reload_rx) = mpsc::channel(1);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let handle = tokio::spawn(run_receiver(
            stream,
            out_tx,
            Arc::clone(&cancel),
            Arc::new(Notify::new()),
            Arc::new(tokio::sync::Mutex::new(Session::default())),
            audio_tx,
            inject_tx,
            reload_tx,
            0,
            empty_snapshot(),
            shutdown.clone(),
            None,
            Arc::new(crate::voice_pin::VoicePinManager::default()),
        ));
        let _stream = tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("run_receiver did not exit when the stream ended")
            .expect("run_receiver task panicked");
        assert!(shutdown.is_cancelled(), "a hang-up must end the session");
        assert!(
            cancel.load(Ordering::Acquire),
            "a hang-up must cancel the turn"
        );
        // And the next queued turn is refused, with `cancel` left set.
        assert!(!begin_turn(&cancel, &shutdown));
        assert!(cancel.load(Ordering::Acquire));
    }

    #[test]
    fn begin_turn_clears_cancel_while_the_client_is_there() {
        let cancel = AtomicBool::new(true); // a client `cancel` from the last turn
        let shutdown = tokio_util::sync::CancellationToken::new();
        assert!(begin_turn(&cancel, &shutdown));
        assert!(!cancel.load(Ordering::Acquire));
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

        let _stream = run_receiver(
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
            npu_footprint_factor: 1.0,
            cpu_image_gen_max_pixels: 1024 * 1024,
            max_audio_duration_secs: 7_200,
            max_images_per_request: 16,
            max_image_pixels_per_request: 64 * 1024 * 1024,
            max_image_pixels: 4_000_000,
            min_kv_cache_gb: 0.0,
            light_model_max_gb: 2.0,
            light_stt_max_gb: 1.0,
            default_placement: crate::model_manager::config::DefaultPlacement::Auto,
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
            stream_first_token_timeout_secs: 600,
            stream_token_gap_timeout_secs: 120,
            bind_addr: "127.0.0.1".to_owned(),
            port: 11_437,
            allow_insecure_public_bind: false,
            supervisor: crate::model_manager::SupervisorConfig::default(),
            unknown_fields: std::collections::HashMap::new(),
            embedding_device: None,
            ov_cache_dir: None,
            ov_cache_layout: "keyed".to_owned(),
            kv_precision_pass: "explicit".to_owned(),
            kv_gate_mode: "exact".to_owned(),
            kv_gate_exact_on_unknown_engine: false,
            kv_headroom_mode: "dynamic".to_owned(),
            kv_transient_factor: 2.0,
            kv_headroom_margin_gib: 0.5,
            kv_headroom_min_tokens: 4096,
            kv_headroom_inflight_tokens: 32_768,
            kv_warmup: true,
            ov_cache_max_gb: 0.0,
            ov_cache_sweep_interval_secs: 21_600,
            kv_pressure_monitor_enabled: false,
            kv_pressure_threshold_pct: 0.0,
            kv_pressure_sustained_secs: 0,
            kv_pressure_sweep_interval_secs: 15,
            kv_resize_cooldown_secs: 30,
            load_timeout_secs: 600,
            on_demand_wait_secs: 0,
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

    /// 0.9.1 S1: a realtime LLM turn on an engine that stalls ends with the stall verdict (the
    /// same engine watchdog as the HTTP routes), the model is booked as stalled, and the client
    /// is told in a plain sentence — no marker, no config key (review F7).
    #[tokio::test]
    async fn a_stalled_llm_turn_ends_with_a_clean_message() {
        use crate::model_manager::ModelManager;
        use crate::model_manager::lifecycle::MockEngineFactory;

        let mut cfg = config_with_model("vlm-test-model", 5.0);
        cfg.stream_first_token_timeout_secs = 10;
        cfg.stream_token_gap_timeout_secs = 10;
        let mm = Arc::new(
            ModelManager::new(cfg, Arc::new(MockEngineFactory::default()))
                .await
                .unwrap(),
        );
        mm.load_model("vlm-test-model").await.unwrap();
        tokio::time::pause();
        let state = AppState::new().with_model_manager(Arc::clone(&mm));
        let session = Session {
            llm_model: Some("vlm-test-model".to_owned()),
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
        let t0 = tokio::time::Instant::now();
        let bound = crate::stall_watch::mock::RealTimeBound::start("realtime stall");
        let err = run_llm_tts(
            crate::stall_watch::mock::STALL_FIRST,
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
        drop(bound);
        assert!(t0.elapsed() >= std::time::Duration::from_secs(10));
        assert!(
            crate::stall_watch::is_generation_stalled_error(&err.to_string()),
            "{err}"
        );
        let message = stalled_client_message(&err).unwrap();
        assert!(!message.contains("_secs") && !message.contains("generation_stalled:"));
        assert!(
            mm.list_models()
                .iter()
                .any(|i| i.id == "vlm-test-model" && i.generation_stalled)
        );
    }

    /// 0.9.1 V5b: a realtime turn follows the model's sampling profile (here the operator's
    /// `sampling`) instead of the former hard-coded greedy + 1.1. The mock text engine records
    /// the request; it sends no final event, so the turn itself ends as an LLM error.
    #[tokio::test]
    async fn run_llm_tts_uses_the_models_sampling_profile() {
        use crate::model_manager::ModelManager;
        use crate::model_manager::lifecycle::MockEngineFactory;

        let mut cfg = config_with_model("llm-test-model", 5.0);
        if let Some(e) = cfg.models.get_mut("llm-test-model") {
            e.policy.sampling = Some(crate::model_manager::config::SamplingConfig {
                temperature: Some(0.7),
                top_p: Some(0.8),
                top_k: Some(20),
                repetition_penalty: Some(1.0),
                presence_penalty: Some(1.5),
                frequency_penalty: None,
            });
        }
        let factory = Arc::new(MockEngineFactory::default());
        let recorded = Arc::clone(&factory.recorded_params);
        let mm = ModelManager::new(cfg, factory).await.unwrap();
        mm.load_model("llm-test-model").await.unwrap();
        let state = AppState::new().with_model_manager(Arc::new(mm));
        let session = Session {
            llm_model: Some("llm-test-model".to_owned()),
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
        let _ = run_llm_tts(
            "hello",
            &session,
            &out_tx,
            &cancel,
            &cancel_notify,
            &state,
            &mut ctx,
            None,
        )
        .await;
        let rec = recorded.lock().unwrap();
        assert_eq!(rec.len(), 1, "one LLM request");
        let p = &rec[0];
        assert_eq!(
            (p.temperature, p.top_p, p.top_k, p.repetition_penalty),
            (Some(0.7), Some(0.8), Some(20), Some(1.0))
        );
        assert_eq!(p.presence_penalty, Some(1.5));
    }

    /// 0.9.1 V1-usage (review R-2): a realtime text turn hands the engine the model's reasoning
    /// parser; thinking is disabled on voice, so the output does not start in thinking.
    #[tokio::test]
    async fn run_llm_tts_hands_the_engine_its_reasoning_split() {
        use crate::model_manager::ModelManager;
        use crate::model_manager::lifecycle::MockEngineFactory;

        let mut cfg = config_with_model("llm-test-model", 5.0);
        if let Some(e) = cfg.models.get_mut("llm-test-model") {
            e.policy.reasoning_parser = Some(crate::model_manager::ReasoningParser::Qwen3);
        }
        let factory = Arc::new(MockEngineFactory::default());
        let recorded = Arc::clone(&factory.recorded_params);
        let mm = ModelManager::new(cfg, factory).await.unwrap();
        mm.load_model("llm-test-model").await.unwrap();
        let state = AppState::new().with_model_manager(Arc::new(mm));
        let session = Session {
            llm_model: Some("llm-test-model".to_owned()),
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
        let _ = run_llm_tts(
            "hello",
            &session,
            &out_tx,
            &cancel,
            &cancel_notify,
            &state,
            &mut ctx,
            None,
        )
        .await;
        let rec = recorded.lock().unwrap();
        assert_eq!(rec.len(), 1, "one LLM request");
        assert_eq!(
            rec[0].reasoning,
            crate::prompt_builder::ReasoningSplit::new(
                Some(crate::model_manager::ReasoningParser::Qwen3),
                false
            )
        );
    }

    /// 0.9.1 V5b (review F7): with no operator `sampling`, a realtime turn on a model whose
    /// `generation_config.json` has `do_sample: true` samples with the file's values and its file
    /// penalty (0.9.0 ran voice greedy + 1.1) — the ruled behaviour change.
    #[tokio::test]
    async fn run_llm_tts_follows_a_file_only_profile() {
        use crate::model_manager::ModelManager;
        use crate::model_manager::lifecycle::MockEngineFactory;

        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("llm-file-model");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(
            model_dir.join("generation_config.json"),
            r#"{"do_sample": true, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                "repetition_penalty": 1.05}"#,
        )
        .unwrap();
        let mut cfg = config_with_model("llm-file-model", 5.0);
        cfg.models_dir = dir.path().to_path_buf();
        let factory = Arc::new(MockEngineFactory::default());
        let recorded = Arc::clone(&factory.recorded_params);
        let mm = ModelManager::new(cfg, factory).await.unwrap();
        mm.load_model("llm-file-model").await.unwrap();
        let state = AppState::new().with_model_manager(Arc::new(mm));
        let session = Session {
            llm_model: Some("llm-file-model".to_owned()),
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
        let _ = run_llm_tts(
            "hello",
            &session,
            &out_tx,
            &cancel,
            &cancel_notify,
            &state,
            &mut ctx,
            None,
        )
        .await;
        let rec = recorded.lock().unwrap();
        assert_eq!(rec.len(), 1, "one LLM request");
        let p = &rec[0];
        assert_eq!(
            (p.temperature, p.top_p, p.top_k, p.repetition_penalty),
            (Some(0.6), Some(0.95), Some(20), Some(1.05))
        );
        assert_eq!(p.presence_penalty, None);
    }

    /// E3 (F3): a realtime turn on a vision model with a device-headroom limit, after an idle gap
    /// (readings 31 s old), takes exactly one fresh reading off the runtime before deciding.
    /// Reverting the turn to the cached `get_chat_context` fails this.
    #[tokio::test]
    async fn run_llm_tts_reads_an_old_device_reading_first_on_a_vision_llm() {
        use crate::model_manager::tests::{headroom_of, thread_probe_vision_manager};
        let (mm, probe) = thread_probe_vision_manager(512 << 20).await;
        headroom_of(&mm).age_readings(std::time::Duration::from_secs(31));
        let state = AppState::new().with_model_manager(Arc::new(mm));
        let session = Session {
            llm_model: Some("vlm-a".to_owned()),
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
        let sync = probe.synchronous_calls();
        assert_eq!(sync.len(), 1, "one synchronous probe: {sync:?}");
        assert_ne!(sync[0].0, std::thread::current().id(), "not on the runtime");
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

    /// A GPU out-of-resources failure inside a realtime turn poisons the context (so the
    /// supervisor restarts the worker) and is reported to the client without the raw C++ text; any
    /// other error is left alone. This path used to classify nothing: the worker kept serving a
    /// dead GPU after the first failure (live, 2026-10-08).
    #[tokio::test]
    async fn gpu_poison_in_a_realtime_turn_flags_the_context_and_hides_the_raw_text() {
        use crate::model_manager::lifecycle::MockEngineFactory;
        let cfg: crate::model_manager::Config = serde_json::from_value(serde_json::json!({
            "models_dir": "/tmp/test-rt-poison-models", "device": "CPU", "total_vram_gb": 22.5,
            "models": { "m": {"vram_gb": 1.0} }
        }))
        .unwrap();
        let mm = std::sync::Arc::new(
            crate::model_manager::ModelManager::new(
                cfg,
                std::sync::Arc::new(MockEngineFactory::default()),
            )
            .await
            .unwrap(),
        );
        let state = AppState::new().with_model_manager(std::sync::Arc::clone(&mm));

        let benign = anyhow::anyhow!("ov_cb_step failed: bad input");
        assert_eq!(gpu_poison_client_message(&state, Some("m"), &benign), None);
        assert!(
            !mm.is_gpu_poisoned(),
            "an ordinary error must not poison the GPU"
        );

        let oom = anyhow::anyhow!(
            "ov_cb_step failed: Exception from src/plugins/intel_gpu/src/graph/impls/onednn/\
             primitive_onednn_base.h:550:\ncould not execute a primitive"
        );
        let shown = gpu_poison_client_message(&state, Some("m"), &oom).unwrap();
        assert!(
            mm.is_gpu_poisoned(),
            "the oneDNN OOM form poisons the context"
        );
        assert!(
            !shown.contains("Exception from") && !shown.contains("primitive"),
            "{shown}"
        );
    }

    /// The -14 form (2026-11-15, ext-091): a realtime turn that fails with only
    /// `CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST` poisons the context like the other markers.
    #[tokio::test]
    async fn exec_status_minus_14_in_a_realtime_turn_flags_the_context() {
        use crate::model_manager::lifecycle::MockEngineFactory;
        let cfg: crate::model_manager::Config = serde_json::from_value(serde_json::json!({
            "models_dir": "/tmp/test-rt-poison14-models", "device": "CPU", "total_vram_gb": 22.5,
            "models": { "m": {"vram_gb": 1.0} }
        }))
        .unwrap();
        let mm = std::sync::Arc::new(
            crate::model_manager::ModelManager::new(
                cfg,
                std::sync::Arc::new(MockEngineFactory::default()),
            )
            .await
            .unwrap(),
        );
        let state = AppState::new().with_model_manager(std::sync::Arc::clone(&mm));

        let oom = anyhow::anyhow!(
            "ov_vlm_generate failed: Exception from src/inference/src/cpp/infer_request.cpp:224:\n\
                 Exception from src/plugins/intel_gpu/src/runtime/ocl/ocl_memory.cpp:593:\n\
                 [GPU] clWaitForEvents, error code: -14 CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST\n\n"
        );
        let shown = gpu_poison_client_message(&state, Some("m"), &oom).unwrap();
        assert!(mm.is_gpu_poisoned(), "the -14 form poisons the context");
        assert!(
            !shown.contains("CL_EXEC_STATUS") && !shown.contains("ocl_memory"),
            "{shown}"
        );
    }

    // ── Retrieval pool cap (0.9.1 Q5) ────────────────────────────────────────

    #[test]
    fn the_retrieval_pool_keeps_the_newest_entries_up_to_the_cap() {
        let entry = |n: usize| ClientHistoryEntry {
            user: format!("u{n}"),
            assistant: format!("a{n}"),
            embedding: None,
            timestamp: 0,
            embedding_model: None,
            stt_model: None,
            llm_model: None,
        };
        let mut pool: Vec<_> = (0..MAX_CLIENT_HISTORY).map(entry).collect();
        cap_retrieval_pool(&mut pool);
        assert_eq!(
            pool.len(),
            MAX_CLIENT_HISTORY,
            "at the cap nothing is dropped"
        );
        pool.push(entry(MAX_CLIENT_HISTORY));
        cap_retrieval_pool(&mut pool);
        assert_eq!(pool.len(), MAX_CLIENT_HISTORY);
        assert_eq!(pool[0].user, "u1", "the oldest entry goes");
        assert_eq!(
            pool[MAX_CLIENT_HISTORY - 1].user,
            format!("u{MAX_CLIENT_HISTORY}")
        );
    }

    /// 0.9.0 pushed each completed turn without the cap: the pool grew by one per turn.
    #[test]
    fn a_completed_turn_keeps_the_retrieval_pool_at_the_cap() {
        let entry = |n: usize| ClientHistoryEntry {
            user: format!("u{n}"),
            assistant: format!("a{n}"),
            embedding: None,
            timestamp: 0,
            embedding_model: None,
            stt_model: None,
            llm_model: None,
        };
        let mut s = Session {
            history_entries: (0..MAX_CLIENT_HISTORY).map(entry).collect(),
            ..Session::default()
        };
        let pool = record_completed_turn(&mut s, entry(MAX_CLIENT_HISTORY));
        assert_eq!(pool.len(), MAX_CLIENT_HISTORY);
        assert_eq!(s.history_entries.len(), MAX_CLIENT_HISTORY);
        assert_eq!(
            pool[MAX_CLIENT_HISTORY - 1].user,
            format!("u{MAX_CLIENT_HISTORY}")
        );
        assert_eq!(
            s.history.len(),
            2,
            "the turn also enters the rolling LLM context"
        );
    }

    // ── LLM stream closed without a final event (0.9.1 Q4) ───────────────────

    /// 0.9.0 broke out of the loop and returned the cut-off text as a finished reply: no
    /// `done`, and the caller recorded it as a turn. It is now an error, so the caller's
    /// error path sends `error` + `done` and records nothing.
    #[tokio::test]
    async fn a_stream_closed_without_a_finish_is_an_llm_error() {
        let (tx, rx) = crate::streaming::stream_channel();
        tx.send(StreamEvent::Token("The answer is forty".to_owned(), 1))
            .await
            .unwrap();
        drop(tx);
        let (out_tx, _out_rx) = mpsc::channel(64);
        let u = UtteranceCtx {
            audio: vec![],
            energy: 0.0,
            utterance_duration_ms: 0,
        };
        let mut ctx = TurnCtx::new(0, 1, &u);
        ctx.llm_start = Some(Instant::now());
        let result = drain_voice_tokens(
            rx,
            &AtomicBool::new(false),
            &Notify::new(),
            &out_tx,
            &Session::default(),
            &AppState::new(),
            &mut ctx,
            &VoicePolicy::default(),
        )
        .await;
        let err = result.expect_err("a cut-off stream must not count as a finished reply");
        assert!(err.to_string().contains("without a finish"), "{err}");
    }

    /// A cancel that arrives while the drain waits, followed by the engine closing the stream,
    /// is a cancelled turn (`done`, `Ok`), not an LLM error.
    #[tokio::test]
    async fn a_cancel_before_the_stream_closes_is_a_cancelled_turn() {
        let (tx, rx) = crate::streaming::stream_channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let c = Arc::clone(&cancel);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            c.store(true, Ordering::Release);
            drop(tx);
        });
        let (out_tx, mut out_rx) = mpsc::channel(64);
        let u = UtteranceCtx {
            audio: vec![],
            energy: 0.0,
            utterance_duration_ms: 0,
        };
        let mut ctx = TurnCtx::new(0, 1, &u);
        ctx.llm_start = Some(Instant::now());
        let result = drain_voice_tokens(
            rx,
            &cancel,
            &Notify::new(),
            &out_tx,
            &Session::default(),
            &AppState::new(),
            &mut ctx,
            &VoicePolicy::default(),
        )
        .await;
        assert!(result.is_ok(), "{:?}", result.err());
        drop(out_tx);
        let mut done = 0;
        while let Some(m) = out_rx.recv().await {
            if let WsMsg::Text(t) = m {
                let v: serde_json::Value = serde_json::from_str(&t).unwrap();
                done += usize::from(v["type"] == "done");
            }
        }
        assert_eq!(done, 1);
    }

    /// 0.9.1 Q1: a turn cancelled after some reply text was generated returns an
    /// empty reply, so the caller records no history entry and sends no
    /// `turn_history`. The text already streamed to the client stays sent.
    #[tokio::test]
    async fn a_cancelled_turn_returns_no_reply_for_the_history() {
        let (tx, rx) = crate::streaming::stream_channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let c = Arc::clone(&cancel);
        tokio::spawn(async move {
            tx.send(StreamEvent::Token("The answer is forty".to_owned(), 1))
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(30)).await;
            c.store(true, Ordering::Release);
            // The engine keeps producing: the next token wakes the drain, whose
            // loop-top cancel check (the route a live turn takes) ends the turn.
            tx.send(StreamEvent::Token(" two".to_owned(), 1))
                .await
                .unwrap();
            drop(tx);
        });
        let (out_tx, mut out_rx) = mpsc::channel(64);
        let u = UtteranceCtx {
            audio: vec![],
            energy: 0.0,
            utterance_duration_ms: 0,
        };
        let mut ctx = TurnCtx::new(0, 1, &u);
        ctx.llm_start = Some(Instant::now());
        let reply = drain_voice_tokens(
            rx,
            &cancel,
            &Notify::new(),
            &out_tx,
            &Session::default(),
            &AppState::new(),
            &mut ctx,
            &VoicePolicy::default(),
        )
        .await
        .unwrap();
        assert_eq!(reply, "", "a cancelled turn must not be recorded");
        assert_eq!(ctx.cancel_stage, "llm");
        drop(out_tx);
        let (mut texts, mut done) = (0, 0);
        while let Some(m) = out_rx.recv().await {
            if let WsMsg::Text(t) = m {
                let v: serde_json::Value = serde_json::from_str(&t).unwrap();
                texts += usize::from(v["type"] == "text");
                done += usize::from(v["type"] == "done");
            }
        }
        assert!(texts >= 1, "the partial text was streamed");
        assert_eq!(done, 1);
    }

    // ── Embedding handle held by a session (0.9.1 Q6) ─────────────────────────

    /// 0.9.0 kept one handle for the session: evicting the embedding model then ran into the
    /// 8 s join timeout ("engine thread did not exit cleanly") and the kept handle answered
    /// `Closing` on every call, even after a reload (probed on this mock, 2026-11-07). A handle
    /// fetched per use and dropped after it lets the eviction finish at once, and the next use
    /// reaches the reloaded engine.
    #[tokio::test]
    async fn an_embedding_handle_fetched_per_use_lets_the_model_be_evicted_and_reloaded() {
        let mm = q3_mm(q3_config(&[("embed-a", 1.0, false)]), &["embed-a"]).await;
        let state = AppState::new().with_model_manager(Arc::clone(&mm));
        let embed_once = || async {
            let handle = embedding_handle_for(&state, Some("embed-a"))?;
            handle
                .embed(vec!["x".to_owned()])
                .await
                .ok()
                .map(|o| o.vectors.len())
        };
        assert_eq!(embed_once().await, Some(1));

        let t = Instant::now();
        mm.evict_model("embed-a").await.unwrap();
        assert!(t.elapsed() < Duration::from_secs(2), "{:?}", t.elapsed());
        assert_eq!(embed_once().await, None, "not loaded: no handle");

        mm.load_model("embed-a").await.unwrap();
        assert_eq!(
            embed_once().await,
            Some(1),
            "the reloaded engine is reached"
        );
    }

    // ── Embedding back-fill (0.9.1 Q7) ───────────────────────────────────────

    fn pooled(user: &str, embedding: Option<Vec<f32>>, model: Option<&str>) -> ClientHistoryEntry {
        ClientHistoryEntry {
            user: user.to_owned(),
            ..history_entry(embedding, model)
        }
    }

    fn empty_snapshot() -> Arc<RwLock<SessionSnapshot>> {
        Arc::new(RwLock::new(SessionSnapshot {
            id: uuid::Uuid::nil(),
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
        }))
    }

    /// Runs one back-fill over `pool`; returns the pool after it, the snapshot's copy and
    /// the users of the echoed `turn_history` events.
    async fn backfill(
        pool: Vec<ClientHistoryEntry>,
    ) -> (
        Vec<ClientHistoryEntry>,
        Vec<ClientHistoryEntry>,
        Vec<String>,
    ) {
        let (handle, _thread) = crate::embed_engine::spawn_mock_embed("mock-embed");
        let session = Arc::new(tokio::sync::Mutex::new(Session {
            history_entries: pool,
            ..Session::default()
        }));
        let snap = empty_snapshot();
        let (out_tx, mut out_rx) = mpsc::channel(256); // every echo buffered: read afterwards
        assert!(backfill_embeddings(&session, &snap, &out_tx, &handle, 0).await);
        drop(out_tx);
        let mut echoed = Vec::new();
        while let Some(m) = out_rx.recv().await {
            if let WsMsg::Text(t) = m {
                let v: serde_json::Value = serde_json::from_str(&t).unwrap();
                if v["type"] == "turn_history" {
                    assert!(v["turn_id"].is_null());
                    assert_eq!(v["embedding_model"], "mock-embed");
                    echoed.push(v["user"].as_str().unwrap().to_owned());
                }
            }
        }
        let pool = session.lock().await.history_entries.clone();
        let snapped = snap.read().unwrap().history_entries.clone();
        (pool, snapped, echoed)
    }

    /// 0.9.0 ran the back-fill once, at connect, before `config.history` could arrive: such
    /// entries were never embedded and so never retrieved. The per-turn back-fill embeds them.
    #[tokio::test]
    async fn the_backfill_embeds_entries_without_a_vector_and_echoes_them() {
        let (pool, snapped, echoed) = backfill(vec![
            pooled("a", None, None),
            pooled("b", Some(vec![1.0; 4]), Some("mock-embed")),
        ])
        .await;
        assert_eq!(pool[0].embedding_model.as_deref(), Some("mock-embed"));
        assert!(pool[0].embedding.is_some());
        assert_eq!(
            pool[1].embedding,
            Some(vec![1.0; 4]),
            "an entry from the same model is kept"
        );
        assert_eq!(echoed, ["a"]);
        assert_eq!(snapped.len(), 2, "the admin snapshot sees the new vectors");
        assert!(snapped[0].embedding.is_some());
    }

    /// An entry embedded by another model is skipped by retrieval; it is re-embedded. A vector
    /// with no recorded model is kept.
    #[tokio::test]
    async fn the_backfill_reembeds_entries_from_another_model() {
        let (pool, _, echoed) = backfill(vec![
            pooled("other", Some(vec![9.0; 8]), Some("other-embed")),
            pooled("unknown", Some(vec![1.0; 4]), None),
        ])
        .await;
        assert_eq!(pool[0].embedding_model.as_deref(), Some("mock-embed"));
        assert_eq!(pool[0].embedding.as_ref().map(Vec::len), Some(4));
        assert_eq!(pool[1].embedding, Some(vec![1.0; 4]));
        assert_eq!(echoed, ["other"]);
    }

    /// More entries than one chunk: every entry is embedded, in order.
    #[tokio::test]
    async fn the_backfill_embeds_a_pool_larger_than_one_chunk() {
        let pool: Vec<_> = (0..EMBED_CHUNK_TEXTS * 2 + 5)
            .map(|n| pooled(&format!("e{n}"), None, None))
            .collect();
        let (pool, _, echoed) = backfill(pool).await;
        assert!(pool.iter().all(|e| e.embedding.is_some()));
        assert_eq!(echoed.len(), EMBED_CHUNK_TEXTS * 2 + 5);
        assert_eq!(echoed[EMBED_CHUNK_TEXTS], format!("e{EMBED_CHUNK_TEXTS}"));
    }

    /// A failing batch is reported, so the session loop backs off.
    #[tokio::test]
    async fn a_failing_backfill_reports_failure() {
        use crate::model_manager::engine::ManagedEngine;
        let (handle, _thread) = crate::embed_engine::spawn_mock_embed("mock-embed");
        ManagedEngine::set_closing(&handle, true);
        let session = Arc::new(tokio::sync::Mutex::new(Session {
            history_entries: vec![pooled("a", None, None)],
            ..Session::default()
        }));
        let (out_tx, _out_rx) = mpsc::channel(64);
        assert!(!backfill_embeddings(&session, &empty_snapshot(), &out_tx, &handle, 0).await);
        assert!(session.lock().await.history_entries[0].embedding.is_none());
    }

    #[test]
    fn needs_embedding_by_vector_and_model() {
        assert!(needs_embedding(&pooled("x", None, None), "m"));
        assert!(needs_embedding(
            &pooled("x", Some(vec![1.0]), Some("n")),
            "m"
        ));
        assert!(!needs_embedding(
            &pooled("x", Some(vec![1.0]), Some("m")),
            "m"
        ));
        assert!(!needs_embedding(&pooled("x", Some(vec![1.0]), None), "m"));
    }

    /// The pool can be replaced while a batch is embedded: a vector is applied only to the
    /// entry it was computed for.
    #[test]
    fn a_backfill_vector_is_not_applied_to_an_entry_that_replaced_its_own() {
        let key = |index, user: &str| BackfillKey {
            index,
            user: user.to_owned(),
            assistant: "a".to_owned(),
            timestamp: 0,
        };
        let mut pool = vec![pooled("new", None, None), pooled("kept", None, None)];
        let applied = apply_backfill(
            &mut pool,
            &[key(0, "old"), key(1, "kept")],
            vec![vec![0.5; 4], vec![0.7; 4]],
            "m",
        );
        assert!(
            pool[0].embedding.is_none(),
            "the replaced position gets no stale vector"
        );
        assert_eq!(pool[1].embedding, Some(vec![0.7; 4]));
        assert_eq!(applied.len(), 1);
    }

    // ── request_model_load reply (0.9.1 Q3) ──────────────────────────────────

    fn resolution(
        model_id: &str,
        reason: Option<&'static str>,
        loading: bool,
    ) -> crate::model_manager::RealtimeResolution {
        crate::model_manager::RealtimeResolution {
            model_id: model_id.to_owned(),
            source: "default",
            reason,
            loading,
        }
    }

    #[test]
    fn load_reply_is_ready_only_for_a_loaded_model() {
        let r = resolution("m", None, false);
        assert_eq!(model_load_reply("m", &r, true, false), LoadReply::Ready);
        assert_eq!(
            model_load_reply("m", &r, false, false),
            LoadReply::Failed("not_resident_no_headroom")
        );
    }

    /// The resolver counts a loading model's reservation against a second copy and may
    /// substitute; the client must still be told to wait for the load that is running.
    #[test]
    fn load_reply_is_started_for_a_model_already_loading() {
        let substituted = resolution("other", Some("would_evict_protected"), false);
        assert_eq!(
            model_load_reply("m", &substituted, false, true),
            LoadReply::Started
        );
    }

    #[test]
    fn load_reply_keeps_started_and_substitution_failures() {
        let started = resolution("m", None, true);
        assert_eq!(
            model_load_reply("m", &started, false, false),
            LoadReply::Started
        );
        let other = resolution("x", Some("would_evict_busy"), false);
        assert_eq!(
            model_load_reply("m", &other, false, false),
            LoadReply::Failed("would_evict_busy")
        );
    }

    /// 0.9.1 Q11: with a TTS model loaded, an omitted `tts_model` still gets it,
    /// and `""` turns speech off instead of being substituted.
    #[tokio::test]
    async fn an_empty_tts_model_turns_speech_off() {
        let mut cfg = q3_config(&[("tts-a", 1.0, false)]);
        if let Some(e) = cfg.models.get_mut("tts-a") {
            e.kind = Some("tts".to_owned());
        }
        let mm = q3_mm(cfg, &["tts-a"]).await;
        assert_eq!(resolve_tts_slot(&mm, false, None).model_id, "tts-a");
        // A `""` the resolver stored itself (nothing was loaded then) is not "off",
        // and not a request either: no reason is reported for it.
        let again = resolve_tts_slot(&mm, false, Some(""));
        assert_eq!(again.model_id, "tts-a");
        assert_eq!(again.reason, None);
        let off = resolve_tts_slot(&mm, true, Some(""));
        assert_eq!(off.model_id, "");
        assert_eq!(off.source, TTS_OFF_SOURCE);
        assert!(!off.loading);
        assert_eq!(off.reason, None);
    }

    /// Only a client-sent `""` turns speech off; naming a model turns it back on,
    /// and a `config` that omits `tts_model` leaves the choice as it was.
    #[tokio::test]
    async fn only_a_client_sent_empty_tts_model_is_off() {
        let session = Arc::new(tokio::sync::Mutex::new(Session::default()));
        let cfg = |tts: Option<&str>| {
            let session = Arc::clone(&session);
            let tts = tts.map(str::to_owned);
            async move {
                apply_client_config(
                    &session, None, None, None, tts, None, None, None, None, None, None, None,
                )
                .await;
            }
        };
        cfg(Some("")).await;
        assert!(session.lock().await.tts_off);
        cfg(None).await;
        assert!(
            session.lock().await.tts_off,
            "an omitted field keeps speech off"
        );
        cfg(Some("tts-a")).await;
        assert!(
            !session.lock().await.tts_off,
            "naming a model turns speech on"
        );
    }

    /// The C1 none resolution: `""`, `substituted`, `none_available`, not loading.
    fn assert_tts_none_available(res: &crate::model_manager::RealtimeResolution) {
        assert_eq!(res.model_id, "", "no TTS id is stored for the session");
        assert_eq!(res.source, "substituted");
        assert_eq!(res.reason, Some("none_available"));
        assert!(!res.loading, "nothing is loading");
    }

    /// 0.9.1 C1 cases 1 and 4 (option C): with no TTS model on the server, nothing requested
    /// and a named unknown id both resolve to none (`none_available`), not to an echoed id.
    #[tokio::test]
    async fn c1_no_tts_available_resolves_to_none_available() {
        let mm = q3_mm(q3_config(&[("llm-a", 1.0, false)]), &["llm-a"]).await;
        assert_tts_none_available(&resolve_tts_slot(&mm, false, None));
        assert_tts_none_available(&resolve_tts_slot(&mm, false, Some("ghost-tts")));
        // Speech off is still `off`, not `none_available`.
        let off = resolve_tts_slot(&mm, true, Some(""));
        assert_eq!((off.source, off.reason), (TTS_OFF_SOURCE, None));
    }

    /// The `model_selection` a none resolution produces keeps `tts_model` as the string `""`
    /// (wire-safe C1: the field is not omitted).
    #[test]
    fn c1_model_selection_keeps_an_empty_tts_model_string() {
        let none = tts_none_available();
        let ev = ServerEvent::ModelSelection {
            stt_model: None,
            stt_source: None,
            stt_reason: None,
            llm_model: "llm-a",
            llm_source: "requested",
            llm_reason: None,
            tts_model: Some(none.model_id.as_str()),
            tts_source: Some(none.source),
            tts_reason: none.reason,
        };
        let v = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["tts_model"], serde_json::json!(""), "{v}");
        assert_eq!(v["tts_source"], "substituted", "{v}");
        assert_eq!(v["tts_reason"], "none_available", "{v}");
    }

    /// Dropped-sentence count (`rustedvino_realtime_tts_sentences_total{result="dropped"}`) for
    /// two sentences sent through `synthesize_and_send` with the session's TTS id set to `tts`.
    fn c1_dropped_sentences(rt: &tokio::runtime::Runtime, state: &AppState, tts: &str) -> f64 {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            rt.block_on(async {
                let session = Session {
                    tts_model: Some(tts.to_owned()),
                    ..Session::default()
                };
                let (out_tx, _out_rx) = mpsc::channel(8);
                let utterance = UtteranceCtx {
                    audio: vec![],
                    energy: 0.0,
                    utterance_duration_ms: 0,
                };
                let mut ctx = TurnCtx::new(0, 1, &utterance);
                let cancel = AtomicBool::new(false);
                let notify = Notify::new();
                for sentence in ["One sentence.", "Another one."] {
                    synthesize_and_send(
                        sentence, &session, &out_tx, state, &mut ctx, &cancel, &notify,
                    )
                    .await;
                }
            });
        });
        handle
            .render()
            .lines()
            .find(|l| {
                l.starts_with("rustedvino_realtime_tts_sentences_total") && l.contains("dropped")
            })
            .and_then(|l| l.rsplit(' ').next())
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0)
    }

    fn c1_runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    /// 0.9.1 C1 cases 1 and 4: the id the resolver stores is what the turn uses, so per
    /// sentence nothing is attempted or counted dropped. A raw unprovidable id (what the
    /// resolver stored before C1) still counts every sentence: the resolution is the fix.
    #[test]
    fn c1_no_tts_available_counts_no_dropped_sentences() {
        let rt = c1_runtime();
        let mm = rt.block_on(q3_mm(q3_config(&[("llm-a", 1.0, false)]), &["llm-a"]));
        let none = resolve_tts_slot(&mm, false, None).model_id;
        let ghost = resolve_tts_slot(&mm, false, Some("ghost-tts")).model_id;
        let state = AppState::new().with_model_manager(mm);
        for id in [none, ghost] {
            assert!(
                c1_dropped_sentences(&rt, &state, &id).abs() < f64::EPSILON,
                "{id:?} is silent"
            );
        }
        assert!(
            (c1_dropped_sentences(&rt, &state, "ghost-tts") - 2.0).abs() < f64::EPSILON,
            "control: an unprovidable id itself still drops every sentence"
        );
    }

    /// A 10 GB server with a pinned, loaded 9 GB LLM and a 5 GB TTS model that cannot fit.
    fn c1_no_room_for_tts(default_tts: bool) -> crate::model_manager::Config {
        let mut cfg = q3_config(&[("llm-a", 9.0, true), ("tts-big", 5.0, false)]);
        if let Some(e) = cfg.models.get_mut("tts-big") {
            e.kind = Some("tts".to_owned());
        }
        if default_tts {
            cfg.realtime_defaults = Some(crate::model_manager::config::RealtimeDefaults {
                stt_model: None,
                llm_model: None,
                tts_model: Some("tts-big".to_owned()),
                embed_model: None,
            });
        }
        cfg
    }

    /// 0.9.1 C1 case 2: a configured `realtime_defaults.tts_model` that is not loaded and does
    /// not fit resolves to none: the default id is not stored for the session, and no sentence
    /// is attempted or counted dropped. The STT and LLM slots are unchanged (no STT model and
    /// no STT default: the resolver's empty id with `not_resident_no_headroom`, as before C1).
    #[test]
    fn c1_a_default_tts_that_does_not_fit_resolves_to_none_available() {
        let rt = c1_runtime();
        let mm = rt.block_on(q3_mm(c1_no_room_for_tts(true), &["llm-a"]));
        let res = resolve_tts_slot(&mm, false, None);
        assert_tts_none_available(&res);
        let stt = mm.resolve_realtime_slot(RealtimeSlot::Stt, None);
        assert_eq!(
            (stt.model_id.as_str(), stt.source, stt.reason, stt.loading),
            ("", "substituted", Some("not_resident_no_headroom"), false),
            "STT keeps its resolution"
        );
        let llm = mm.resolve_realtime_slot(RealtimeSlot::Llm, None);
        assert_eq!(llm.model_id, "llm-a");
        // The shared resolver itself is unchanged for TTS too (request_model_load uses it).
        let raw = mm.resolve_realtime_slot(RealtimeSlot::Tts, None);
        assert_eq!(
            (raw.model_id.as_str(), raw.source, raw.reason),
            ("tts-big", "default", Some("not_resident_no_headroom"))
        );
        let state = AppState::new().with_model_manager(mm);
        assert!(c1_dropped_sentences(&rt, &state, &res.model_id).abs() < f64::EPSILON);
    }

    /// 0.9.1 C1 case 3: a registered TTS id that has no room now, with no default, resolves to
    /// none: no sentence is attempted or counted dropped.
    #[test]
    fn c1_a_named_tts_without_room_resolves_to_none_available() {
        let rt = c1_runtime();
        let mm = rt.block_on(q3_mm(c1_no_room_for_tts(false), &["llm-a"]));
        let res = resolve_tts_slot(&mm, false, Some("tts-big"));
        assert_tts_none_available(&res);
        let state = AppState::new().with_model_manager(mm);
        assert!(c1_dropped_sentences(&rt, &state, &res.model_id).abs() < f64::EPSILON);
    }

    /// C1 leaves a usable TTS alone: a loaded one is granted as requested, and a named one that
    /// fits is granted with a background load (`loading: true`), as before.
    #[tokio::test]
    async fn c1_a_tts_that_is_loaded_or_fits_is_still_granted() {
        let mut cfg = q3_config(&[
            ("llm-a", 1.0, false),
            ("tts-a", 1.0, false),
            ("tts-b", 1.0, false),
        ]);
        for id in ["tts-a", "tts-b"] {
            if let Some(e) = cfg.models.get_mut(id) {
                e.kind = Some("tts".to_owned());
            }
        }
        let mm = q3_mm(cfg, &["llm-a", "tts-a"]).await;
        let ready = resolve_tts_slot(&mm, false, Some("tts-a"));
        assert_eq!(
            (ready.model_id.as_str(), ready.source),
            ("tts-a", "requested")
        );
        let fits = resolve_tts_slot(&mm, false, Some("tts-b"));
        assert_eq!(
            (fits.model_id.as_str(), fits.source, fits.loading),
            ("tts-b", "requested", true)
        );
    }

    /// An empty admin snapshot for a C1b test session.
    fn c1b_snapshot() -> Arc<RwLock<SessionSnapshot>> {
        Arc::new(RwLock::new(SessionSnapshot {
            id: uuid::Uuid::new_v4(),
            conn_id: 7,
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
        }))
    }

    /// 0.9.1 C1b (C1 case 5): a TTS slot resolved `loading: true` whose load then fails. The
    /// watcher clears the session's TTS id (and the admin snapshot's) and takes it out of the
    /// D2 serving set, sends no event before the watch window is up (no second
    /// `model_selection`), and later sentences are not attempted or counted dropped.
    #[test]
    fn c1b_a_tts_load_that_fails_after_resolve_clears_the_session_tts() {
        const TTS: &str = "tts-fail";
        let rt = c1_runtime();
        let mut cfg = q3_config(&[("llm-a", 1.0, false), (TTS, 1.0, false)]);
        if let Some(e) = cfg.models.get_mut(TTS) {
            e.kind = Some("tts".to_owned());
        }
        let factory = crate::model_manager::lifecycle::MockEngineFactory {
            fail_message_for: Some((TTS.to_owned(), "mock TTS load failed".to_owned())),
            ..Default::default()
        };
        let mm = rt.block_on(async {
            let mm = Arc::new(
                crate::model_manager::ModelManager::new(cfg, Arc::new(factory))
                    .await
                    .unwrap(),
            );
            mm.load_model("llm-a").await.unwrap();
            mm
        });
        let session = Arc::new(tokio::sync::Mutex::new(Session::default()));
        let snap = c1b_snapshot();
        let voice_pin = Arc::new(crate::voice_pin::VoicePinManager::default());
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let tts_id = rt.block_on(async {
            // As the `config` path does: resolve (spawns the load), store, register, watch.
            let res = resolve_tts_slot(&mm, false, Some(TTS));
            assert_eq!((res.model_id.as_str(), res.loading), (TTS, true));
            {
                let mut s = session.lock().await;
                s.tts_model = Some(res.model_id.clone());
                swap_serving(&mut s, &voice_pin);
            }
            if let Ok(mut s) = snap.write() {
                s.tts_model = Some(res.model_id.clone());
            }
            assert_eq!(voice_pin.serving_count(TTS), 1);
            spawn_model_ready_watcher(
                Arc::clone(&mm),
                res.model_id.clone(),
                out_tx.clone(),
                shutdown.clone(),
                Some(TtsLoadFailClear {
                    session: Arc::clone(&session),
                    session_snap: Arc::clone(&snap),
                    voice_pin: Arc::clone(&voice_pin),
                    conn_id: 7,
                }),
            );
            for _ in 0..100 {
                if session.lock().await.tts_model.as_deref() == Some("") {
                    break;
                }
                time::sleep(Duration::from_millis(20)).await;
            }
            session.lock().await.tts_model.clone()
        });
        assert_eq!(tts_id.as_deref(), Some(""), "the failed id is cleared");
        assert!(!mm.is_ready(TTS) && !mm.is_loading(TTS), "the load failed");
        assert_eq!(snap.read().unwrap().tts_model.as_deref(), Some(""));
        assert_eq!(voice_pin.serving_count(TTS), 0, "left the serving set");
        assert!(
            out_rx.try_recv().is_err(),
            "no event before the watch window is up"
        );
        shutdown.cancel();
        let state = AppState::new().with_model_manager(Arc::clone(&mm));
        let id = tts_id.unwrap_or_default();
        assert!(
            c1_dropped_sentences(&rt, &state, &id).abs() < f64::EPSILON,
            "later sentences are not attempted or counted dropped"
        );
        assert!(
            (c1_dropped_sentences(&rt, &state, TTS) - 2.0).abs() < f64::EPSILON,
            "control: the failed id itself drops every sentence"
        );
    }

    /// C1b: the watcher clears only the id it watched — a later `config` that moved the
    /// session to another TTS model is left alone, and so is its serving-set entry.
    #[tokio::test]
    async fn c1b_clear_leaves_a_session_that_moved_on_alone() {
        let session = Arc::new(tokio::sync::Mutex::new(Session {
            tts_model: Some("tts-other".to_owned()),
            ..Session::default()
        }));
        let voice_pin = Arc::new(crate::voice_pin::VoicePinManager::default());
        swap_serving(&mut *session.lock().await, &voice_pin);
        let clear = TtsLoadFailClear {
            session: Arc::clone(&session),
            session_snap: c1b_snapshot(),
            voice_pin: Arc::clone(&voice_pin),
            conn_id: 7,
        };
        assert!(!clear_failed_tts(&clear, "tts-fail").await);
        assert_eq!(session.lock().await.tts_model.as_deref(), Some("tts-other"));
        assert_eq!(voice_pin.serving_count("tts-other"), 1);
    }

    // ── C1/C1b through the real `config` path (fix round F3) ──────────────

    /// A manager over `cfg` with `factory`, `preload` loaded.
    async fn c1_mm(
        cfg: crate::model_manager::Config,
        factory: crate::model_manager::lifecycle::MockEngineFactory,
        preload: &[&str],
    ) -> Arc<crate::model_manager::ModelManager> {
        let mm = Arc::new(
            crate::model_manager::ModelManager::new(cfg, Arc::new(factory))
                .await
                .unwrap(),
        );
        for m in preload {
            mm.load_model(m).await.unwrap();
        }
        mm
    }

    /// `q3_config` with every id in `tts` marked a TTS model.
    fn c1_cfg(models: &[(&str, f64, bool)], tts: &[&str]) -> crate::model_manager::Config {
        let mut cfg = q3_config(models);
        for id in tts {
            if let Some(e) = cfg.models.get_mut(*id) {
                e.kind = Some("tts".to_owned());
            }
        }
        cfg
    }

    /// A receiver running on `mm` that has been sent one `config` frame.
    struct C1Run {
        session: Arc<tokio::sync::Mutex<Session>>,
        snap: Arc<RwLock<SessionSnapshot>>,
        voice_pin: Arc<crate::voice_pin::VoicePinManager>,
        out_rx: mpsc::Receiver<WsMsg>,
        shutdown: tokio_util::sync::CancellationToken,
    }

    fn c1_start_receiver(
        mm: &Arc<crate::model_manager::ModelManager>,
        config: &serde_json::Value,
    ) -> C1Run {
        let frames = futures_util::stream::iter(vec![Ok::<_, axum::Error>(WsMsg::Text(
            config.to_string().into(),
        ))])
        .chain(futures_util::stream::pending());
        let (out_tx, out_rx) = mpsc::channel(16);
        let (audio_tx, _audio_rx) = mpsc::channel(1);
        let (inject_tx, _inject_rx) = mpsc::channel(1);
        let (reload_tx, _reload_rx) = mpsc::channel(1);
        let session = Arc::new(tokio::sync::Mutex::new(Session::default()));
        let snap = c1b_snapshot();
        let voice_pin = Arc::new(crate::voice_pin::VoicePinManager::default());
        let shutdown = tokio_util::sync::CancellationToken::new();
        tokio::spawn(run_receiver(
            frames,
            out_tx,
            Arc::new(AtomicBool::new(false)),
            Arc::new(Notify::new()),
            Arc::clone(&session),
            audio_tx,
            inject_tx,
            reload_tx,
            7,
            Arc::clone(&snap),
            shutdown.clone(),
            Some(Arc::clone(mm)),
            Arc::clone(&voice_pin),
        ));
        C1Run {
            session,
            snap,
            voice_pin,
            out_rx,
            shutdown,
        }
    }

    /// The next JSON event within `ms`, or `None`.
    async fn c1_next_event(
        out_rx: &mut mpsc::Receiver<WsMsg>,
        ms: u64,
    ) -> Option<serde_json::Value> {
        loop {
            match time::timeout(Duration::from_millis(ms), out_rx.recv()).await {
                Ok(Some(WsMsg::Text(t))) => return serde_json::from_str(&t).ok(),
                Ok(Some(_)) => {}
                Ok(None) | Err(_) => return None,
            }
        }
    }

    /// C1 through `config`: with no TTS on the server, `model_selection` carries
    /// `tts_model: ""`, `substituted`, `none_available`, and the session stores `""`.
    #[tokio::test]
    async fn c1_config_with_no_tts_on_the_server_selects_none_available() {
        let mm = c1_mm(
            c1_cfg(&[("llm-a", 1.0, false)], &[]),
            crate::model_manager::lifecycle::MockEngineFactory::default(),
            &["llm-a"],
        )
        .await;
        let mut run = c1_start_receiver(&mm, &serde_json::json!({"type":"config"}));
        let ev = c1_next_event(&mut run.out_rx, 2000).await.unwrap();
        assert_eq!(ev["type"], "model_selection", "{ev}");
        assert_eq!(ev["tts_model"], serde_json::json!(""), "{ev}");
        assert_eq!(ev["tts_source"], "substituted", "{ev}");
        assert_eq!(ev["tts_reason"], "none_available", "{ev}");
        assert_eq!(run.session.lock().await.tts_model.as_deref(), Some(""));
        run.shutdown.cancel();
    }

    /// C1b through `config` (proves the config path hands the TTS watcher its clear context):
    /// a named TTS that fits but fails to load is cleared from the session, the snapshot and
    /// the serving set, cleared exactly once (one WARN), and no event beyond `model_loading started` +
    /// `model_selection` before the watch window is up.
    #[test]
    fn c1b_config_path_clears_a_tts_whose_load_fails() {
        const TTS: &str = "tts-cfgfail";
        let rt = c1_runtime();
        let run = rt.block_on(async {
            let factory = crate::model_manager::lifecycle::MockEngineFactory {
                fail_message_for: Some((TTS.to_owned(), "mock TTS load failed".to_owned())),
                ..Default::default()
            };
            let mm = c1_mm(
                c1_cfg(&[("llm-a", 1.0, false), (TTS, 1.0, false)], &[TTS]),
                factory,
                &["llm-a"],
            )
            .await;
            let mut run =
                c1_start_receiver(&mm, &serde_json::json!({"type":"config","tts_model":TTS}));
            let started = c1_next_event(&mut run.out_rx, 2000).await.unwrap();
            assert_eq!(
                (
                    started["type"].as_str(),
                    started["model_id"].as_str(),
                    started["stage"].as_str()
                ),
                (Some("model_loading"), Some(TTS), Some("started")),
                "{started}"
            );
            let sel = c1_next_event(&mut run.out_rx, 2000).await.unwrap();
            assert_eq!(sel["type"], "model_selection", "{sel}");
            assert_eq!(
                (sel["tts_model"].as_str(), sel["tts_source"].as_str()),
                (Some(TTS), Some("requested")),
                "{sel}"
            );
            for _ in 0..100 {
                if run.session.lock().await.tts_model.as_deref() == Some("") {
                    break;
                }
                time::sleep(Duration::from_millis(20)).await;
            }
            assert_eq!(run.session.lock().await.tts_model.as_deref(), Some(""));
            assert!(
                c1_next_event(&mut run.out_rx, 400).await.is_none(),
                "no further event"
            );
            run
        });
        assert_eq!(run.snap.read().unwrap().tts_model.as_deref(), Some(""));
        assert_eq!(run.voice_pin.serving_count(TTS), 0, "left the serving set");
        run.shutdown.cancel();
        let clears = TTS_CLEARS
            .lock()
            .map_or(0, |m| m.get(TTS).copied().unwrap_or(0));
        assert_eq!(clears, 1, "cleared exactly once");
    }

    /// F3(a): a TTS load that takes longer than one poll but succeeds is not mistaken for a
    /// failure: the session keeps the id and `ready` is sent.
    #[tokio::test]
    async fn c1b_a_slow_tts_load_that_succeeds_keeps_the_session_tts() {
        const TTS: &str = "tts-slow";
        let factory = crate::model_manager::lifecycle::MockEngineFactory {
            load_delay: Some(Duration::from_millis(600)),
            ..Default::default()
        };
        let mm = c1_mm(
            c1_cfg(&[("llm-a", 1.0, false), (TTS, 1.0, false)], &[TTS]),
            factory,
            &["llm-a"],
        )
        .await;
        let mut run = c1_start_receiver(&mm, &serde_json::json!({"type":"config","tts_model":TTS}));
        let started = c1_next_event(&mut run.out_rx, 2000).await.unwrap();
        assert_eq!(started["stage"], "started", "{started}");
        let sel = c1_next_event(&mut run.out_rx, 2000).await.unwrap();
        assert_eq!(sel["type"], "model_selection", "{sel}");
        let ready = c1_next_event(&mut run.out_rx, 3000).await.unwrap();
        assert_eq!(
            (ready["model_id"].as_str(), ready["stage"].as_str()),
            (Some(TTS), Some("ready")),
            "{ready}"
        );
        assert_eq!(run.session.lock().await.tts_model.as_deref(), Some(TTS));
        assert_eq!(run.voice_pin.serving_count(TTS), 1);
        run.shutdown.cancel();
    }

    /// F2: a TTS id another party is already loading — the resolver echoes it with
    /// `loading: false` (its reservation leaves no room for a second copy) — is a loading
    /// resolution for the session: `requested`, `model_loading started`, and the watcher
    /// (so a failure would be cleared); `ready` follows when that load lands.
    #[tokio::test]
    async fn c1_a_tts_another_party_is_loading_is_a_loading_resolution() {
        const TTS: &str = "tts-slow";
        let factory = crate::model_manager::lifecycle::MockEngineFactory {
            load_delay: Some(Duration::from_millis(800)),
            ..Default::default()
        };
        let mm = c1_mm(
            // Room (10 GB) for one copy of the 6 GB TTS, not for the second the
            // resolver's dry-run counts while the first is still loading.
            c1_cfg(&[(TTS, 6.0, false)], &[TTS]),
            factory,
            &[],
        )
        .await;
        let other = Arc::clone(&mm);
        let load = tokio::spawn(async move { other.load_model(TTS).await });
        for _ in 0..100 {
            if mm.is_loading(TTS) {
                break;
            }
            time::sleep(Duration::from_millis(5)).await;
        }
        assert!(mm.is_loading(TTS), "the other party's load is running");
        let raw = mm.resolve_realtime_slot(RealtimeSlot::Tts, Some(TTS));
        assert_eq!(
            (raw.model_id.as_str(), raw.loading),
            (TTS, false),
            "the scenario"
        );
        let res = resolve_tts_slot(&mm, false, Some(TTS));
        assert_eq!(
            (res.model_id.as_str(), res.source, res.reason, res.loading),
            (TTS, "requested", None, true)
        );
        let mut run = c1_start_receiver(&mm, &serde_json::json!({"type":"config","tts_model":TTS}));
        let started = c1_next_event(&mut run.out_rx, 2000).await.unwrap();
        assert_eq!(
            (started["model_id"].as_str(), started["stage"].as_str()),
            (Some(TTS), Some("started")),
            "{started}"
        );
        let sel = c1_next_event(&mut run.out_rx, 2000).await.unwrap();
        assert_eq!(
            (sel["tts_model"].as_str(), sel["tts_source"].as_str()),
            (Some(TTS), Some("requested")),
            "{sel}"
        );
        assert!(sel.get("tts_reason").is_none(), "{sel}");
        let ready = c1_next_event(&mut run.out_rx, 3000).await.unwrap();
        assert_eq!(ready["stage"], "ready", "{ready}");
        load.await.unwrap().unwrap();
        assert_eq!(run.session.lock().await.tts_model.as_deref(), Some(TTS));
        run.shutdown.cancel();
    }

    /// N1: a C1b clear that runs after the session's teardown changes nothing —
    /// the session's STT/LLM ids are not registered again.
    #[tokio::test]
    async fn c1b_a_clear_after_teardown_does_not_re_register_the_session() {
        let session = Arc::new(tokio::sync::Mutex::new(Session {
            stt_model: Some("stt-a".to_owned()),
            llm_model: Some("llm-a".to_owned()),
            tts_model: Some("tts-late".to_owned()),
            ..Session::default()
        }));
        let voice_pin = Arc::new(crate::voice_pin::VoicePinManager::default());
        swap_serving(&mut *session.lock().await, &voice_pin);
        assert_eq!(voice_pin.serving_count("llm-a"), 1);
        close_serving(&mut *session.lock().await, &voice_pin);
        let clear = TtsLoadFailClear {
            session: Arc::clone(&session),
            session_snap: c1b_snapshot(),
            voice_pin: Arc::clone(&voice_pin),
            conn_id: 7,
        };
        assert!(
            !clear_failed_tts(&clear, "tts-late").await,
            "no clear after teardown"
        );
        for id in ["stt-a", "llm-a", "tts-late"] {
            assert_eq!(voice_pin.serving_count(id), 0, "{id}");
        }
    }

    /// N2: an id in both the old and the new set keeps its count through a swap.
    #[test]
    fn swap_serving_keeps_an_id_that_stays() {
        let mut s = Session {
            llm_model: Some("llm-a".to_owned()),
            ..Session::default()
        };
        let voice_pin = crate::voice_pin::VoicePinManager::default();
        swap_serving(&mut s, &voice_pin);
        s.tts_model = Some("tts-a".to_owned());
        swap_serving(&mut s, &voice_pin);
        assert_eq!(voice_pin.serving_count("llm-a"), 1);
        assert_eq!(voice_pin.serving_count("tts-a"), 1);
    }

    #[test]
    fn serving_ids_skip_empty_ids() {
        let s = Session {
            stt_model: Some("stt".to_owned()),
            llm_model: Some(String::new()),
            tts_model: Some(String::new()),
            ..Session::default()
        };
        assert_eq!(serving_ids(&s), vec!["stt".to_owned()]);
    }

    #[test]
    fn a_set_with_an_empty_id_is_not_pinned() {
        assert!(pinnable_voice_flow("stt", "llm", "tts"));
        assert!(!pinnable_voice_flow("stt", "llm", ""));
        assert!(!pinnable_voice_flow("", "llm", "tts"));
        assert!(!pinnable_voice_flow("stt", "", "tts"));
    }

    fn q3_config(models: &[(&str, f64, bool)]) -> crate::model_manager::Config {
        let mut cfg = admission_test_config(std::collections::HashMap::new());
        cfg.total_vram_gb = 10.0;
        for (id, gb, pinned) in models {
            cfg.models.insert(
                (*id).to_owned(),
                crate::model_manager::config::ModelEntry {
                    vram_gb: *gb,
                    kind: None,
                    policy: crate::model_manager::config::ModelPolicy {
                        pinned: *pinned,
                        ..Default::default()
                    },
                },
            );
        }
        cfg
    }

    fn with_default_llm(
        mut cfg: crate::model_manager::Config,
        id: &str,
    ) -> crate::model_manager::Config {
        cfg.realtime_defaults = Some(crate::model_manager::config::RealtimeDefaults {
            stt_model: None,
            llm_model: Some(id.to_owned()),
            tts_model: None,
            embed_model: None,
        });
        cfg
    }

    async fn q3_mm(
        cfg: crate::model_manager::Config,
        preload: &[&str],
    ) -> Arc<crate::model_manager::ModelManager> {
        use crate::model_manager::ModelManager;
        use crate::model_manager::lifecycle::MockEngineFactory;
        let mm = Arc::new(
            ModelManager::new(cfg, Arc::new(MockEngineFactory::default()))
                .await
                .unwrap(),
        );
        for m in preload {
            mm.load_model(m).await.unwrap();
        }
        mm
    }

    /// 0.9.0 answered `ready`: the resolver falls back to the slot default, which is the
    /// requested model, and reports it with `loading: false` because it does not fit.
    #[tokio::test]
    async fn a_default_llm_that_does_not_fit_is_not_reported_ready() {
        let cfg = with_default_llm(
            q3_config(&[("stt-pin", 6.0, true), ("llm-b", 6.0, false)]),
            "llm-b",
        );
        let mm = q3_mm(cfg, &["stt-pin"]).await;
        assert_eq!(
            request_model_load_reply(&mm, "llm-b"),
            LoadReply::Failed("not_resident_no_headroom")
        );
    }

    /// 0.9.0 answered `ready` for an id the server does not know, when no default and no
    /// loaded model exist for the slot.
    #[tokio::test]
    async fn an_unknown_model_is_not_reported_ready() {
        let mm = q3_mm(q3_config(&[("llm-a", 2.0, false)]), &[]).await;
        assert_eq!(
            request_model_load_reply(&mm, "llm-typo"),
            LoadReply::Failed("unknown_model")
        );
    }

    /// With a model of the slot loaded, the resolver would substitute it and the reply would
    /// carry `would_evict_protected`; an unknown id is named as such instead.
    #[tokio::test]
    async fn an_unknown_model_is_named_even_with_a_loaded_llm() {
        let mm = q3_mm(q3_config(&[("llm-a", 2.0, false)]), &["llm-a"]).await;
        assert_eq!(
            request_model_load_reply(&mm, "llm-typo"),
            LoadReply::Failed("unknown_model")
        );
    }

    /// With a default and nothing loaded, the resolver would start loading the default; an
    /// unknown id is refused before that, so nothing is loaded on its behalf.
    #[tokio::test]
    async fn an_unknown_model_starts_no_load_of_the_default() {
        let cfg = with_default_llm(q3_config(&[("llm-a", 2.0, false)]), "llm-a");
        let mm = q3_mm(cfg, &[]).await;
        assert_eq!(
            request_model_load_reply(&mm, "llm-typo"),
            LoadReply::Failed("unknown_model")
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!mm.is_ready("llm-a") && !mm.is_loading("llm-a"));
    }

    #[tokio::test]
    async fn a_loaded_model_is_reported_ready() {
        let mm = q3_mm(q3_config(&[("llm-a", 2.0, false)]), &["llm-a"]).await;
        assert_eq!(request_model_load_reply(&mm, "llm-a"), LoadReply::Ready);
    }
}
