// ============================================================
// src/handlers/media.rs — Phase 5 media endpoints (STT / TTS / Image)
// ============================================================
// The OpenAI media routes:
//
//   POST /v1/audio/transcriptions → STT/Whisper
//   POST /v1/audio/translations   → STT/Whisper, translate task (English out)
//   POST /v1/audio/speech         → TTS
//   POST /v1/images/generations   → Image/SDXL
//   POST /v1/images/edits         → Image/SDXL img2img
//
// 5.1a scope: parse the multipart request, validate it, and render all four
// OpenAI `response_format`s from a CANNED transcription — no GPU, no model
// resolution. The real Whisper pipeline + model resolution (404/WrongKind) land
// in 5.1b/5.1c.
// ============================================================

use axum::{
    Extension, Json,
    extract::{Multipart, State},
    http::StatusCode,
    response::{IntoResponse as _, Response},
};
use serde::{Deserialize, Serialize};

use super::chat::unix_now;
use super::error::{
    JsonBody, MultipartBody, device_admission_error_response, inference_error_response,
    openai_error, openai_error_with_param,
};
use crate::Authenticated;
use crate::app_state::AppState;
use crate::device_inventory::DeviceInfo;
use crate::model_manager::ModelManager;
use crate::ov_image::{ImageEditOptions, ImageGenOptions, RgbImage};
use crate::pipelines::stt::{Segment, SttTask, Transcription};
use crate::pipelines::tts::TTS_SAMPLE_RATE;
use crate::tts_split::{TTS_SEGMENT_CAP_HEADER, TTS_SEGMENTS_HEADER};
use axum::http::{HeaderName, HeaderValue};

// ---- Request types --------------------------------------------------

/// The `response_format` of a transcription (`OpenAI`-compatible). Absent →
/// [`ResponseFormat::Json`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResponseFormat {
    /// `{"text": "…"}` — the default.
    Json,
    /// The transcript as a `text/plain` body, no JSON wrapper.
    Text,
    /// `SubRip` subtitles (`.srt`) as `text/plain`.
    Srt,
    /// `{task, language, duration, text, segments:[…]}` with per-segment timings.
    VerboseJson,
}

impl ResponseFormat {
    /// Parse the wire value; `None`/absent → `Json`. Returns `None` for an
    /// unrecognised value (the caller turns that into a 400).
    fn parse(value: Option<&str>) -> Option<Self> {
        match value.unwrap_or("json") {
            "json" => Some(Self::Json),
            "text" => Some(Self::Text),
            "srt" => Some(Self::Srt),
            "verbose_json" => Some(Self::VerboseJson),
            _ => None,
        }
    }
}

/// Fields parsed from the `multipart/form-data` transcription request. Only the
/// fields the server acts on; `prompt`/`temperature` are accepted-and-ignored.
struct TranscriptionRequest {
    /// Raw uploaded audio bytes (consumed by the Whisper pipeline in 5.1c).
    file: Vec<u8>,
    /// Target STT model id (resolved through the manager in 5.1c).
    model: String,
    /// Optional source-language hint (`"en"`, `"pl"`, …).
    language: Option<String>,
    /// How to serialise the result.
    response_format: ResponseFormat,
}

// ---- Response types --------------------------------------------------
//
// The decoded [`Transcription`]/[`Segment`] result types live in
// `crate::pipelines::stt` (single source of truth, shared with the engine).

/// `response_format=json` body.
#[derive(Serialize)]
struct JsonResponse<'a> {
    text: &'a str,
}

/// `response_format=verbose_json` body (`OpenAI` shape).
#[derive(Serialize)]
struct VerboseResponse<'a> {
    task: &'static str,
    language: &'a str,
    duration: f32,
    text: &'a str,
    segments: Vec<VerboseSegment<'a>>,
}

/// One entry of `verbose_json.segments`.
#[derive(Serialize)]
struct VerboseSegment<'a> {
    id: usize,
    start: f32,
    end: f32,
    text: &'a str,
}

// ---- Handler --------------------------------------------------------

/// `POST /v1/audio/transcriptions` — `OpenAI`-compatible speech-to-text.
///
/// 5.1c: parses + validates the multipart request, resolves the `model` through
/// the manager (404 if absent / 400 `WrongKind` if not an STT model), and runs
/// the real Whisper pipeline (audio decode + resample + transcription) before
/// rendering the requested `response_format`. With no model manager attached
/// (the no-GPU integration-test path) it renders a canned transcription.
pub async fn transcriptions(
    State(state): State<AppState>,
    MultipartBody(multipart): MultipartBody,
) -> Response {
    speech_to_text(&state, multipart, SttTask::Transcribe).await
}

/// `POST /v1/audio/translations` — `OpenAI`-compatible speech translation:
/// English text whatever the spoken language (Whisper's translate task).
///
/// Same request and response shapes as
/// [`transcriptions`](transcriptions); `language`, not in `OpenAI`'s spec
/// for this route, is accepted as a hint for the *spoken* language. A model
/// known not to translate (English-only, from its `generation_config.json`)
/// is refused with 400 `model_cannot_translate` before any work.
pub async fn translations(
    State(state): State<AppState>,
    MultipartBody(multipart): MultipartBody,
) -> Response {
    speech_to_text(&state, multipart, SttTask::Translate).await
}

/// Shared body of the two Whisper routes.
async fn speech_to_text(state: &AppState, multipart: Multipart, task: SttTask) -> Response {
    let req = match parse_request(multipart).await {
        Ok(r) => r,
        Err(resp) => return resp,
    };

    // srt/verbose_json carry per-segment timings; json/text only need the text,
    // so we only pay for timestamps when the chosen format will use them.
    let timestamps = matches!(
        req.response_format,
        ResponseFormat::Srt | ResponseFormat::VerboseJson
    );

    tracing::debug!(
        model = %req.model,
        bytes = req.file.len(),
        language = req.language.as_deref().unwrap_or("auto"),
        timestamps,
        ?task,
        "speech-to-text request"
    );

    let transcription = if let Some(mm) = state.model_manager.as_ref() {
        // Slice 3c parity: a NotLoaded model declared `load: on_demand` loads lazily on first
        // use; F1b: the request waits for it up to `on_demand_wait_secs`, else 503 Loading +
        // Retry-After. Eager models keep failing fast with a bare NotLoaded 503. Mirrors the
        // chat path (`chat.rs`).
        let handle = match super::on_demand::resolve_with_wait(mm, &req.model, || {
            mm.get_stt_handle(&req.model)
        })
        .await
        {
            Ok(h) => h,
            Err(u) => return u.into_response(&req.model),
        };
        if task == SttTask::Translate && !handle.can_translate() {
            return bad_request(
                "this Whisper model is English-only and cannot translate — use a \
                 multilingual Whisper model (not an .en one)",
                "model_cannot_translate",
            );
        }

        // Cross-pipeline device admission (step 2): a second gate in front of
        // the engine's own per-engine gate below, capping concurrency across
        // different engine kinds sharing this model's device.
        let device = mm.record_device(&req.model);
        let _device_lease = match state
            .device_budgets
            .admit(&[(device, 1)], crate::admission::WorkClass::Stt)
            .await
        {
            Ok(lease) => lease,
            Err(e) => return device_admission_error_response(&e),
        };

        let result = match task {
            SttTask::Transcribe => handle.transcribe(req.file, req.language, timestamps).await,
            SttTask::Translate => handle.translate(req.file, req.language, timestamps).await,
        };
        match result {
            Ok(t) => t,
            // Admission gate: 429 when the engine's at capacity, 503 when its
            // thread is gone — same mapping the chat handler uses for CB/VLM/NPU.
            Err(crate::cb_engine::AdmitError::AtCapacity) => {
                return openai_error(
                    StatusCode::TOO_MANY_REQUESTS,
                    "server at capacity — all inference slots busy, retry shortly",
                    "rate_limit_error",
                    Some("server_overloaded"),
                );
            }
            Err(crate::cb_engine::AdmitError::Closing) => {
                return super::error::model_closing_response();
            }
            Err(crate::cb_engine::AdmitError::EngineDead) => {
                return openai_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "STT engine unavailable (engine thread exited)",
                    "server_error",
                    Some("engine_unavailable"),
                );
            }
            // T6.2: never reflect raw C++ diagnostics into the body — an OOM-class
            // error becomes a retryable 503, anything else an opaque 500.
            Err(crate::cb_engine::AdmitError::Failed(e)) => {
                // The upload itself is undecodable: the caller's fault, and
                // retrying cannot help (D9). Anything else stays an engine error.
                if let Some(resp) = bad_audio_response(&e) {
                    return resp;
                }
                return inference_error_response(Some(mm), handle.model_id(), &e);
            }
        }
    } else {
        // Mock path (no GPU): a canned transcription so the route and all four
        // response formats are exercisable in integration tests without a model.
        let mut t = mock_transcription(req.language.as_deref());
        if task == SttTask::Translate {
            "This is a mock translation.".clone_into(&mut t.text);
        }
        t
    };

    render(&transcription, req.response_format, task)
}

/// Drain the multipart body into a validated [`TranscriptionRequest`], or an
/// error `Response` (400) on a malformed body / missing required field.
// Response is the idiomatic "early-return a ready-built HTTP error" type used
// throughout this handler module; called once per request, not a hot loop.
#[allow(clippy::result_large_err)]
async fn parse_request(mut multipart: Multipart) -> Result<TranscriptionRequest, Response> {
    let mut file: Option<Vec<u8>> = None;
    let mut model: Option<String> = None;
    let mut language: Option<String> = None;
    let mut response_format_raw: Option<String> = None;

    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(e) => {
                return Err(multipart_error(
                    &e,
                    "malformed multipart/form-data body",
                    "invalid_multipart",
                ));
            }
        };
        // Copy the name before consuming the field's body.
        let name = field.name().map(ToOwned::to_owned);
        match name.as_deref() {
            Some("file") => match field.bytes().await {
                Ok(bytes) => file = Some(bytes.to_vec()),
                Err(e) => {
                    return Err(multipart_error(
                        &e,
                        "could not read field 'file'",
                        "invalid_field",
                    ));
                }
            },
            Some("model") => model = field.text().await.ok(),
            Some("language") => language = field.text().await.ok(),
            Some("response_format") => response_format_raw = field.text().await.ok(),
            // prompt, temperature, and any unknown field: drain and ignore.
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    let file = file
        .filter(|f| !f.is_empty())
        .ok_or_else(|| bad_request("missing required field 'file'", "missing_file"))?;
    let model = model
        .filter(|m| !m.is_empty())
        .ok_or_else(|| bad_request("missing required field 'model'", "missing_model"))?;
    // FM-3: normalise the hint (`PL`, `pl-PL` -> `pl`; `auto` -> detect) and refuse a code Whisper
    // has no token for here, as a 400, instead of a 500 from inside the pipeline.
    let language = match language
        .as_deref()
        .map(crate::pipelines::stt::normalize_language)
    {
        None => None,
        Some(Ok(l)) => l,
        Some(Err(message)) => {
            return Err(openai_error_with_param(
                StatusCode::BAD_REQUEST,
                message,
                "invalid_request_error",
                Some("invalid_language"),
                Some("language"),
            ));
        }
    };
    let response_format =
        ResponseFormat::parse(response_format_raw.as_deref()).ok_or_else(|| {
            bad_request(
                "unsupported response_format — use json, text, srt, or verbose_json",
                "invalid_response_format",
            )
        })?;

    Ok(TranscriptionRequest {
        file,
        model,
        language,
        response_format,
    })
}

/// Render a [`Transcription`] in the requested format. `json`/`verbose_json` are
/// JSON; `text`/`srt` are `text/plain` bodies. `verbose_json`'s `task` names the
/// route that produced it (`"transcribe"` / `"translate"`), as `OpenAI`'s does.
fn render(t: &Transcription, format: ResponseFormat, task: SttTask) -> Response {
    match format {
        ResponseFormat::Json => Json(JsonResponse { text: &t.text }).into_response(),
        ResponseFormat::Text => t.text.clone().into_response(),
        ResponseFormat::Srt => to_srt(&t.segments).into_response(),
        ResponseFormat::VerboseJson => Json(VerboseResponse {
            task: match task {
                SttTask::Transcribe => "transcribe",
                SttTask::Translate => "translate",
            },
            language: &t.language,
            duration: t.duration(),
            text: &t.text,
            segments: t
                .segments
                .iter()
                .enumerate()
                .map(|(id, s)| VerboseSegment {
                    id,
                    start: s.start,
                    end: s.end,
                    text: &s.text,
                })
                .collect(),
        })
        .into_response(),
    }
}

/// Render segments as `SubRip` (`.srt`): 1-indexed cues with `HH:MM:SS,mmm` spans.
fn to_srt(segments: &[Segment]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for (i, seg) in segments.iter().enumerate() {
        // Writing to a String is infallible; ignore the formatter Result.
        let _ = write!(
            out,
            "{}\n{} --> {}\n{}\n\n",
            i + 1,
            srt_timestamp(seg.start),
            srt_timestamp(seg.end),
            seg.text.trim(),
        );
    }
    out
}

/// Format a second offset as an SRT timestamp `HH:MM:SS,mmm`.
fn srt_timestamp(seconds: f32) -> String {
    // Negative offsets clamp to zero; after the clamp the value is a small
    // non-negative millisecond count, so the round-then-cast neither truncates
    // meaningfully nor loses a sign.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let total_ms = (f64::from(seconds).max(0.0) * 1000.0).round() as u64;
    let hours = total_ms / 3_600_000;
    let minutes = (total_ms % 3_600_000) / 60_000;
    let secs = (total_ms % 60_000) / 1000;
    let millis = total_ms % 1000;
    format!("{hours:02}:{minutes:02}:{secs:02},{millis:03}")
}

/// A canned transcription for the no-GPU path (no model manager attached).
/// `language` echoes the requested hint, else defaults to `"english"`.
fn mock_transcription(language: Option<&str>) -> Transcription {
    Transcription {
        text: "This is a mock transcription.".to_owned(),
        language: language.unwrap_or("english").to_owned(),
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
    }
}

/// Longest text `POST /v1/audio/speech` accepts, in characters (the same limit as the `OpenAI` API).
const MAX_SPEECH_INPUT_CHARS: usize = 4096;

/// The 400 for text the speech endpoint cannot take: empty, containing a NUL byte, or longer than
/// [`MAX_SPEECH_INPUT_CHARS`]. The text encoder's attention is quadratic in the input: 70,000 characters
/// asked ONNX Runtime for a 78 GB allocation (70,000^2 x 4 B x 4 heads) in the overnight torture run.
fn speech_input_error(input: &str) -> Option<Response> {
    if input.trim().is_empty() {
        return Some(bad_request("input must not be empty", "missing_input"));
    }
    if input.contains('\0') {
        return Some(bad_request(
            "input must not contain NUL bytes",
            "invalid_input",
        ));
    }
    if input.chars().count() > MAX_SPEECH_INPUT_CHARS {
        return Some(bad_request(
            "input is too long: at most 4096 characters per request",
            "input_too_long",
        ));
    }
    None
}

/// The 400 for `SpeechT5` input that splits into more than
/// [`crate::tts_split::SPEECHT5_MAX_PIECES`] pieces (0.9.1 T5), naming the
/// limit, the count and the per-piece `cap`. `None` when the request is within it.
fn speech_pieces_error(pieces: usize, cap: usize) -> Option<Response> {
    let max = crate::tts_split::SPEECHT5_MAX_PIECES;
    (pieces > max).then(|| {
        bad_request_owned(
            format!(
                "input is too long for this model: it splits into {pieces} pieces after \
                 normalisation, at most {max} pieces of {cap} characters are accepted per request"
            ),
            "input_too_long",
        )
    })
}

/// The 400 for an edit `image`/`mask` that could not be decoded, or that passes the request's
/// image limits (0.9.1 S5: `too_many_images` / `image_too_large`, naming the limit).
fn image_decode_error(field: &str, e: &anyhow::Error) -> Response {
    if let Some(limit) = e.downcast_ref::<crate::image_util::ImageLimitExceeded>() {
        return bad_request_owned(format!("'{field}': {}", limit.message), limit.code);
    }
    let code = if field == "mask" {
        "invalid_mask"
    } else {
        "invalid_image"
    };
    bad_request_owned(format!("could not decode '{field}': {e}"), code)
}

/// The 400 for an upload that could not be decoded as audio, or `None` when
/// `e` is any other (engine) failure. Retrying the same bytes cannot help, so
/// this is the caller's error, not a 500.
fn bad_audio_response(e: &anyhow::Error) -> Option<Response> {
    let bad = e.downcast_ref::<crate::pipelines::stt::BadAudio>()?;
    Some(bad_request_owned(
        format!("the uploaded file is not usable audio: {bad}"),
        "invalid_audio",
    ))
}

/// Build a 400 `invalid_request_error` with the shared `OpenAI` envelope.
/// A multipart read error as a response. Exceeding the server's request-body
/// limit is an honest 413 that names the limit — it used to surface as a 400
/// "could not read field 'file'", which hid the cause (a 123 MB podcast
/// upload, found by the rustedvino-www session). Anything else keeps the
/// caller's 400.
fn multipart_error(
    e: &axum::extract::multipart::MultipartError,
    message: &'static str,
    code: &'static str,
) -> Response {
    if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
        return super::error::request_too_large();
    }
    bad_request(message, code)
}

fn bad_request(message: &'static str, code: &'static str) -> Response {
    openai_error(
        StatusCode::BAD_REQUEST,
        message,
        "invalid_request_error",
        Some(code),
    )
}

// ── TTS request type ─────────────────────────────────────────────────────────

/// `/v1/audio/speech` request (`OpenAI`-compatible JSON).
#[derive(Debug, serde::Deserialize)]
pub struct SpeechRequest {
    /// Target TTS model id.
    model: String,
    /// Text to synthesise.
    input: String,
    /// Voice name. `SpeechT5` ignores this (uses the built-in default speaker
    /// embedding); reserved for Stage 2 Kokoro voice dispatch.
    #[serde(default)]
    voice: Option<String>,
    /// Output format: `"wav"` (default) or `"pcm"` (raw i16 LE).
    /// `mp3` / `opus` / `flac` → 400 (not yet).
    #[serde(default)]
    response_format: Option<String>,
    /// Playback speed multiplier (0.25–4.0). `SpeechT5` ignores this; Kokoro
    /// and Coqui VITS (`pl-mai_female`) both honor it — `1.0` is each
    /// checkpoint's own trained default pace, not a fixed absolute rate.
    #[serde(default = "default_speed")]
    speed: f32,
    /// Language of `input` (`"en"`, `"pl"`, `"vi"`; a region suffix such as
    /// `"pl-PL"` is fine). Not in `OpenAI`'s spec: clients that omit it get the
    /// language the model speaks. Selects the number words used when
    /// `normalize` is on; an unsupported language leaves digits untouched.
    #[serde(default)]
    language: Option<String>,
    /// Expand digits and symbols (`%`, `&`, `$`, ...) into spoken words before
    /// synthesis. Default `true`; send `false` if the text is already
    /// speakable. Not in `OpenAI`'s spec.
    #[serde(default = "default_true")]
    normalize: bool,
}

fn default_true() -> bool {
    true
}

/// The text to synthesise: `input` with digits and symbols spoken out, unless
/// the request opted out or names a language with no number words. The
/// request's `language` wins over `model_language` (what the loaded model
/// speaks); with neither, English (as realtime does). Unicode spaces become
/// plain spaces first, in every case ([`crate::tts_normalize::plain_spaces`],
/// 0.9.1 C6).
fn speech_text(
    input: String,
    normalize: bool,
    request_language: Option<&str>,
    model_language: Option<&str>,
) -> String {
    let input = match crate::tts_normalize::plain_spaces(&input) {
        std::borrow::Cow::Borrowed(_) => input,
        std::borrow::Cow::Owned(mapped) => mapped,
    };
    if !normalize {
        return input;
    }
    let language = request_language.or(model_language);
    if language.is_some_and(|l| !crate::tts_normalize::supports_language(l)) {
        return input;
    }
    crate::tts_normalize::normalize_for_speech(&input, language)
}

fn default_speed() -> f32 {
    1.0
}

// ── WAV encoder ───────────────────────────────────────────────────────────────

/// Encode mono f32 PCM at `sample_rate` into a minimal 16-bit PCM WAV.
///
/// Samples are clamped to [-1, 1] and scaled to i16 range. The returned
/// `Vec<u8>` starts with the 44-byte RIFF/WAVE/fmt/data header.
fn encode_wav_i16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    #[allow(clippy::cast_possible_truncation)]
    let data_len = (samples.len() * 2) as u32;
    let mut v = Vec::with_capacity(44 + samples.len() * 2);
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
        #[allow(clippy::cast_possible_truncation)]
        let i = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
        v.extend_from_slice(&i.to_le_bytes());
    }
    v
}

// ── Handler ──────────────────────────────────────────────────────────────────

/// `POST /v1/audio/speech` — `OpenAI`-compatible text-to-speech (`SpeechT5`).
///
/// 5.2a: validates the JSON request, resolves the `model` through the manager
/// (404 if absent / 400 `WrongKind` if not a TTS model), runs the real
/// `Text2SpeechPipeline`, and returns 16-bit PCM WAV (or raw PCM) bytes.
#[allow(clippy::too_many_lines)]
pub async fn speech(
    State(state): State<AppState>,
    JsonBody(req): JsonBody<SpeechRequest>,
) -> Response {
    use axum::http::header::CONTENT_TYPE;

    // Validate response_format first — cheap, before any model resolution.
    let fmt = match req.response_format.as_deref().unwrap_or("wav") {
        "wav" => "wav",
        "pcm" => "pcm",
        _ => {
            return bad_request(
                "unsupported response_format — use 'wav' or 'pcm'",
                "invalid_response_format",
            );
        }
    };

    if !(0.25..=4.0).contains(&req.speed) {
        return bad_request("speed must be between 0.25 and 4.0", "invalid_speed");
    }

    if let Some(resp) = speech_input_error(&req.input) {
        return resp;
    }

    // `sample_rate` is split from `samples` so the WAV header uses the correct rate
    // per backend (SpeechT5 = 16 kHz, Kokoro = 24 kHz). The mock path defaults to
    // `TTS_SAMPLE_RATE` (SpeechT5) since it doesn't have a live handle.
    let sample_rate;
    // `(pieces, cap)` when the engine splits its input (SpeechT5): sent back as
    // the `x-rustedvino-tts-segments` / `-segment-cap` headers.
    let mut segments: Option<(usize, usize)> = None;
    let samples = if let Some(mm) = state.model_manager.as_ref() {
        // Slice 3c / F1b: an `on_demand` TTS model loads on first use, the request waiting up
        // to `on_demand_wait_secs` (see the STT path above).
        let handle = match super::on_demand::resolve_with_wait(mm, &req.model, || {
            mm.get_tts_handle(&req.model)
        })
        .await
        {
            Ok(h) => h,
            Err(u) => return u.into_response(&req.model),
        };

        tracing::debug!(
            model = %req.model,
            input_chars = req.input.len(),
            fmt,
            "TTS request"
        );

        sample_rate = handle.sample_rate();

        let text = speech_text(
            req.input,
            req.normalize,
            req.language.as_deref(),
            handle.language(),
        );
        // 0.9.1 T1: SpeechT5 drops input past ~600 characters with a 200, so its
        // (normalised) text is split into sentence pieces of at most `cap`
        // characters, synthesized in order and joined. Text within the cap is one
        // piece, unchanged. Other engines take the whole text in one call.
        // Split before the device admission below, so the T5 piece cap refuses
        // an oversized request without queueing it or touching the engine.
        let pieces: Vec<String> = match handle.segment_cap() {
            Some(cap) => {
                let pieces: Vec<String> = crate::tts_split::split_for_speech(&text, cap)
                    .into_iter()
                    .map(str::to_owned)
                    .collect();
                tracing::info!(
                    model = %req.model,
                    segments = pieces.len(),
                    cap,
                    input_chars = text.chars().count(),
                    "TTS segments"
                );
                if let Some(resp) = speech_pieces_error(pieces.len(), cap) {
                    tracing::info!(
                        model = %req.model,
                        pieces = pieces.len(),
                        limit = crate::tts_split::SPEECHT5_MAX_PIECES,
                        "TTS request refused: too many pieces"
                    );
                    return resp;
                }
                segments = Some((pieces.len(), cap));
                pieces
            }
            None => vec![text],
        };

        // Cross-pipeline device admission (step 2): a second gate in front of
        // the engine's own per-engine gate below, capping concurrency across
        // different engine kinds sharing this model's device.
        let device = mm.record_device(&req.model);
        let _device_lease = match state
            .device_budgets
            .admit(&[(device, 1)], crate::admission::WorkClass::Tts)
            .await
        {
            Ok(lease) => lease,
            Err(e) => return device_admission_error_response(&e),
        };

        // One admission for the whole request, all pieces (T1 F1/F2). SpeechT5's
        // own output starts and ends near silence (~40 ms lead, ~150-190 ms tail,
        // edges about -52 dBFS, measured on the T1 sweep), so the pieces are
        // joined as they are: no inserted gap, no fade.
        match handle.synthesize_pieces(pieces, req.voice, req.speed).await {
            Ok(s) => s,
            Err(e) => return speech_admit_error_response(e, mm, &handle),
        }
    } else {
        // Mock path (no GPU): one second of silence so the route and both
        // response formats are exercisable in integration tests without a model.
        sample_rate = TTS_SAMPLE_RATE;
        vec![0.0_f32; TTS_SAMPLE_RATE as usize]
    };

    let mut response = match fmt {
        "wav" => {
            let bytes = encode_wav_i16(&samples, sample_rate);
            (StatusCode::OK, [(CONTENT_TYPE, "audio/wav")], bytes).into_response()
        }
        "pcm" => {
            // Raw i16 LE — same quantisation as the WAV path, no header.
            let mut bytes = Vec::with_capacity(samples.len() * 2);
            for s in &samples {
                // Clamp guarantees [-32767, 32767] — safe to truncate.
                #[allow(clippy::cast_possible_truncation)]
                let i = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
                bytes.extend_from_slice(&i.to_le_bytes());
            }
            (
                StatusCode::OK,
                [(CONTENT_TYPE, "audio/octet-stream")],
                bytes,
            )
                .into_response()
        }
        _ => unreachable!("fmt already validated to 'wav' or 'pcm'"),
    };
    if let Some((n, cap)) = segments {
        let headers = response.headers_mut();
        headers.insert(
            HeaderName::from_static(TTS_SEGMENTS_HEADER),
            HeaderValue::from(n),
        );
        headers.insert(
            HeaderName::from_static(TTS_SEGMENT_CAP_HEADER),
            HeaderValue::from(cap),
        );
    }
    response
}

/// The response for a failed TTS synthesis call (one piece of a request).
fn speech_admit_error_response(
    e: crate::cb_engine::AdmitError,
    mm: &std::sync::Arc<ModelManager>,
    handle: &crate::pipelines::tts::TtsHandle,
) -> Response {
    match e {
        crate::cb_engine::AdmitError::AtCapacity => openai_error(
            StatusCode::TOO_MANY_REQUESTS,
            "server at capacity — all inference slots busy, retry shortly",
            "rate_limit_error",
            Some("server_overloaded"),
        ),
        crate::cb_engine::AdmitError::Closing => super::error::model_closing_response(),
        crate::cb_engine::AdmitError::EngineDead => openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "TTS engine unavailable (engine thread exited)",
            "server_error",
            Some("engine_unavailable"),
        ),
        crate::cb_engine::AdmitError::Failed(e) => {
            inference_error_response(Some(mm), handle.model_id(), &e)
        }
    }
}

// ============================================================
// Image generation — POST /v1/images/generations (SDXL, the phase-5 IMAGE plan)
// ============================================================
//
// 5.3a scope: parse + validate the JSON request and render a CANNED image — no
// GPU, no model resolution. The real Text2Image pipeline + model resolution
// (404 / WrongKind) land in 5.3c. `response_format` is **`b64_json` only**: we
// host no external URL, so `url` is a 400, not a silent fallback.

// ---- Request / response types ---------------------------------------

/// `/v1/images/generations` request (`OpenAI`-compatible JSON). Only the fields
/// the server acts on; `quality`/`style`/`user` are accepted-and-ignored.
#[derive(Debug, Deserialize)]
pub struct ImageGenerationRequest {
    /// Target image model id (resolved through the manager in 5.3c).
    model: String,
    /// Text prompt to render.
    prompt: String,
    /// Number of images to generate (1–4). Absent → 1.
    #[serde(default = "default_n")]
    n: u32,
    /// Output size — square or portrait/landscape (see [`parse_size`]). Absent → 1024×1024.
    size: Option<String>,
    /// Output encoding. Only `"b64_json"` is supported (no URL host). Absent → `b64_json`.
    response_format: Option<String>,
    /// Denoising steps (superset of the `OpenAI` surface; absent → 20). Distilled
    /// models want far fewer — FLUX.1-schnell is tuned for ~4. Bounded 1..=50.
    steps: Option<u32>,
    /// RNG seed for reproducibility. Absent → pipeline default (seed `42`, deterministic).
    seed: Option<u64>,
    /// Classifier-free guidance scale. Absent → pipeline built-in default (FLUX ignores
    /// this; SD3.5 default ~4.5). Higher = stricter prompt adherence, more saturated.
    /// Bounded 0.0–20.0; 0.0 means "use pipeline default".
    guidance_scale: Option<f32>,
    /// Negative prompt — describe what to avoid. Absent → no negative guidance.
    /// FLUX ignores this (distilled); SD3.5 supports it.
    negative_prompt: Option<String>,
}

/// `OpenAI` default when `n` is omitted.
fn default_n() -> u32 {
    1
}

/// A validated request: dimensions resolved, `n` bounded. Carries `model`/`prompt`
/// through to the engine call (used in 5.3c).
struct ValidatedImageRequest {
    width: u32,
    height: u32,
    n: u32,
    steps: u32,
    seed: Option<u64>,
    guidance_scale: f32,
    negative_prompt: Option<String>,
}

/// `/v1/images/generations` response (`OpenAI` shape, plus `RustedVINO`'s additive
/// `generation_metadata` — the image-metadata plan): `{ created,
/// data: [{ b64_json }], generation_metadata }`.
#[derive(Serialize)]
struct ImageGenerationResponse {
    created: u64,
    elapsed_ms: u128,
    data: Vec<ImageData>,
    generation_metadata: GenerationMetadata,
}

/// One generated image, base64-encoded PNG.
#[derive(Serialize)]
struct ImageData {
    b64_json: String,
}

/// RustedVINO-specific generation metadata, additive to the `OpenAI` response
/// shape — all three tiers of the image-metadata plan (model
/// identity, engine, device/host for hardware correlation, model hash,
/// sampler, and operator-supplied precision/source/revision). Unknown/
/// inapplicable fields are omitted, never emitted as `null`.
#[derive(Serialize)]
struct GenerationMetadata {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_family: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_hash_short: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sampler: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scheduler_config: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    precision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    device: Option<DeviceMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    host: Option<String>,
    engine: &'static str,
    engine_version: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    openvino_version: Option<String>,
}

/// The first 10 hex chars of a `sha256:<hex>`-prefixed digest — the A1111-style
/// short form (the Pyramu image-metadata spec §5.1: "first 10 hex chars of the
/// model file's SHA-256"). `None` if `full` isn't the expected shape (defensive
/// only — [`crate::pipelines::image::ImageHandle::model_hash`] always produces
/// `sha256:` + 64 hex chars).
fn model_hash_short(full: &str) -> Option<String> {
    full.strip_prefix("sha256:")
        .and_then(|hex| hex.get(..10))
        .map(str::to_owned)
}

/// The placement device's silicon identity, capability tier, and memory —
/// what makes cross-box hardware correlation (Arc B50 vs B60 vs B70, and
/// eventually which physical box) possible from the response alone.
#[derive(Serialize)]
struct DeviceMetadata {
    id: String,
    full_name: String,
    kind: String,
    tier: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    architecture: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_mem_gb: Option<f64>,
}

impl From<DeviceInfo> for DeviceMetadata {
    fn from(info: DeviceInfo) -> Self {
        Self {
            id: info.name,
            full_name: info.full_name,
            kind: info.kind.to_string(),
            tier: info.tier.to_string(),
            architecture: info.architecture,
            total_mem_gb: info.total_mem_gb,
        }
    }
}

/// The `ImageHandle`-derived subset of `GenerationMetadata` (Tiers 1 and 2,
/// plus Tier 3's operator-supplied provenance) — everything reachable once a
/// handle is acquired, before the model/device-agnostic fields (`host`,
/// `engine`, `openvino_version`) or the mock-path all-`None` case. Factored
/// out because `image_generations`/`image_edits` both need it identically.
#[derive(Default)]
struct HandleMetadata {
    model_family: Option<String>,
    model_hash: Option<String>,
    sampler: Option<String>,
    scheduler_config: Option<serde_json::Value>,
    precision: Option<String>,
    model_source: Option<String>,
    model_revision: Option<String>,
}

impl HandleMetadata {
    fn from_handle(handle: &crate::pipelines::image::ImageHandle) -> Self {
        Self {
            model_family: handle.model_family().map(str::to_owned),
            model_hash: handle.model_hash().map(str::to_owned),
            sampler: handle.sampler().map(str::to_owned),
            scheduler_config: handle.scheduler_config().cloned(),
            precision: handle.precision().map(str::to_owned),
            model_source: handle.model_source().map(str::to_owned),
            model_revision: handle.model_revision().map(str::to_owned),
        }
    }
}

// ---- Handler --------------------------------------------------------

/// `POST /v1/images/generations` — `OpenAI`-compatible text-to-image (SDXL).
///
/// 5.3c: validates the request, resolves the `model` through the manager (404 if
/// absent / 400 `WrongKind` if not an image model), and runs the real SDXL
/// `Text2Image` pipeline before base64-encoding each PNG. With no model manager
/// attached (the no-GPU integration-test path) it renders a canned image.
pub async fn image_generations(
    State(state): State<AppState>,
    authenticated: Option<Extension<Authenticated>>,
    JsonBody(req): JsonBody<ImageGenerationRequest>,
) -> Response {
    let valid = match validate_image_request(&req) {
        Ok(v) => v,
        Err(resp) => return *resp,
    };

    tracing::debug!(
        model = %req.model,
        prompt_chars = req.prompt.len(),
        n = valid.n,
        width = valid.width,
        height = valid.height,
        "image generation request"
    );

    let mut t0 = std::time::Instant::now();
    let mut hmeta = HandleMetadata::default();
    let mut device = None;
    let pngs = if let Some(mm) = state.model_manager.as_ref() {
        let pixels = u64::from(valid.width) * u64::from(valid.height) * u64::from(valid.n);
        if let Some(resp) = cpu_pixel_limit_refusal(mm, &req.model, pixels) {
            return resp;
        }
        // Slice 3c parity (see the STT path above): lazy-load an `on_demand` image model on
        // first request rather than failing flat; F1b: waiting up to `on_demand_wait_secs`.
        let handle = match super::on_demand::resolve_with_wait(mm, &req.model, || {
            mm.get_image_handle(&req.model)
        })
        .await
        {
            Ok(h) => h,
            Err(u) => return u.into_response(&req.model),
        };
        // `elapsed_ms` measures the generation, not an on-demand wait.
        t0 = std::time::Instant::now();
        hmeta = HandleMetadata::from_handle(&handle);
        device = mm
            .device_info_for(handle.model_id())
            .map(DeviceMetadata::from);
        let opts = ImageGenOptions {
            negative_prompt: valid.negative_prompt,
            width: valid.width,
            height: valid.height,
            num_inference_steps: valid.steps,
            num_images: valid.n,
            seed: valid.seed,
            guidance_scale: valid.guidance_scale,
        };
        match handle.generate(req.prompt.clone(), opts).await {
            Ok(p) => p,
            // T6.2: never reflect raw C++ diagnostics into the body — an OOM-class
            // error becomes a retryable 503, anything else an opaque 500.
            Err(e) => return inference_error_response(Some(mm), handle.model_id(), &e),
        }
    } else {
        // Mock path (no GPU): one canned PNG per requested image so the route and
        // the response shape are exercisable without a model. No real device/
        // pipeline behind it, so `model_family`/`device` stay omitted.
        match mock_images(valid.n) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!(error = %e, "failed to encode mock image");
                return openai_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "failed to encode image",
                    "internal_error",
                    None,
                );
            }
        }
    };
    let elapsed_ms = t0.elapsed().as_millis();

    tracing::info!(
        model = %req.model,
        width = valid.width,
        height = valid.height,
        n = valid.n,
        elapsed_ms,
        "image generation complete"
    );

    Json(ImageGenerationResponse {
        created: unix_now(),
        elapsed_ms,
        data: encode_pngs(&pngs),
        generation_metadata: GenerationMetadata {
            model: req.model.clone(),
            model_hash_short: hmeta.model_hash.as_deref().and_then(model_hash_short),
            model_family: hmeta.model_family,
            model_hash: hmeta.model_hash,
            sampler: hmeta.sampler,
            scheduler_config: hmeta.scheduler_config,
            precision: hmeta.precision,
            model_source: hmeta.model_source,
            model_revision: hmeta.model_revision,
            device,
            // Hostname only for key-validated callers (see `Authenticated`).
            host: authenticated.and_then(|_| crate::os_memory::host_name()),
            engine: "RustedVINO",
            engine_version: env!("CARGO_PKG_VERSION"),
            openvino_version: crate::ov_cb::openvino_version(),
        },
    })
    .into_response()
}

/// Default denoising steps when the request does not pin one — matches stormVINO.
const DEFAULT_NUM_INFERENCE_STEPS: u32 = 20;

/// Maximum total pixel count per request (width × height × n). Guards against the
/// Measured incident: n=4 @ 1024² caused an `OpenCL` context poison / OOM on the B50.
/// At 2×1024² the B50 (16 GB, ~6.5 GB for weights) comfortably fits n=2 @ 1024².
const MAX_IMAGE_PIXEL_BUDGET: u64 = 2 * 1024 * 1024;

/// Minimum width/height (px) `image_edits` accepts for a user-uploaded source
/// image. Below this, the SDXL/LCM img2img/inpaint pipeline's own
/// downsampling stack can run out of spatial room mid-convolution — `OpenVINO`
/// throws a shape-inference exception, but the C++ pipeline is left corrupted
/// and the process SEGFAULTS on the very next operation against it (
/// confirmed live at 64×64; 256×256 confirmed safe). Matches `parse_size`'s
/// own smallest supported *generation* size (`"256x256"`) — not a new number,
/// just enforced for edit's user-supplied input the same way generation's
/// enum-constrained `size` already enforces it by construction.
const MIN_EDIT_IMAGE_EDGE: u32 = 256;

/// Validate the request: `prompt` non-empty, `n` in `1..=4`, `size` one of the
/// supported values, `response_format` `b64_json` (URL host unsupported), and
/// `n × width × height` within the pixel budget. Returns a 400 `Response` on
/// any violation.
///
/// The error half is boxed: a full `Response` is large (>128 bytes) and
/// `clippy::result_large_err` flags returning it by value (same pattern as
/// `tokenize::resolve_handle`).
fn validate_image_request(
    req: &ImageGenerationRequest,
) -> Result<ValidatedImageRequest, Box<Response>> {
    if req.prompt.trim().is_empty() {
        return Err(Box::new(bad_request(
            "missing required field 'prompt'",
            "missing_prompt",
        )));
    }
    if !(1..=4).contains(&req.n) {
        return Err(Box::new(bad_request(
            "'n' must be between 1 and 4",
            "invalid_n",
        )));
    }
    match req.response_format.as_deref() {
        None | Some("b64_json") => {}
        Some("url") => {
            return Err(Box::new(bad_request(
                "response_format 'url' is not supported — this server returns b64_json only",
                "invalid_response_format",
            )));
        }
        Some(_) => {
            return Err(Box::new(bad_request(
                "unsupported response_format — use b64_json",
                "invalid_response_format",
            )));
        }
    }
    let (width, height) = parse_size(req.size.as_deref()).ok_or_else(|| {
        Box::new(bad_request(
            "unsupported size — use 256x256, 512x512, 512x768, 768x512, \
             768x1024, 1024x768, 1024x1024, 1024x1536, 1536x1024, or 1536x864",
            "invalid_size",
        ))
    })?;
    let steps = match req.steps {
        None => DEFAULT_NUM_INFERENCE_STEPS,
        Some(s) if (1..=50).contains(&s) => s,
        Some(_) => {
            return Err(Box::new(bad_request(
                "'steps' must be between 1 and 50",
                "invalid_steps",
            )));
        }
    };

    // Pixel-budget guard: prevent n×w×h combos that OOM the GPU and poison the
    // OpenCL context (observed at n=4 @ 1024² on B50 16 GB).
    let total_pixels = u64::from(width) * u64::from(height) * u64::from(req.n);
    if total_pixels > MAX_IMAGE_PIXEL_BUDGET {
        let max_n = MAX_IMAGE_PIXEL_BUDGET / (u64::from(width) * u64::from(height));
        return Err(Box::new(bad_request_owned(
            format!(
                "n={} at {}x{} exceeds the per-request pixel budget; \
                 try n≤{}",
                req.n, width, height, max_n
            ),
            "n_exceeds_pixel_budget",
        )));
    }

    let guidance_scale = match req.guidance_scale {
        None | Some(0.0) => 0.0,
        Some(g) if (0.0..=20.0).contains(&g) => g,
        Some(_) => {
            return Err(Box::new(bad_request(
                "'guidance_scale' must be between 0.0 and 20.0",
                "invalid_guidance_scale",
            )));
        }
    };

    Ok(ValidatedImageRequest {
        width,
        height,
        n: req.n,
        steps,
        seed: req.seed,
        guidance_scale,
        negative_prompt: req.negative_prompt.clone(),
    })
}

/// Map a `size` string to `(width, height)`. Absent → 1024×1024. Returns `None`
/// for an unsupported value (the caller turns that into a 400).
///
/// Supported sizes span squares and common portrait/landscape pairs (128-pixel
/// steps, SDXL-compatible aspect ratios). Portrait sizes unlock full-body
/// subjects that square crops cut off.
fn parse_size(size: Option<&str>) -> Option<(u32, u32)> {
    match size.unwrap_or("1024x1024") {
        "256x256" => Some((256, 256)),
        "512x512" => Some((512, 512)),
        "512x768" => Some((512, 768)),
        "768x512" => Some((768, 512)),
        "768x1024" => Some((768, 1024)),
        "1024x768" => Some((1024, 768)),
        "1024x1024" => Some((1024, 1024)),
        "1024x1536" => Some((1024, 1536)),
        "1536x1024" => Some((1536, 1024)),
        "1536x864" => Some((1536, 864)),
        _ => None,
    }
}

/// `n` canned 1×1 PNGs — the no-GPU response shape. A solid black pixel, just
/// enough to be a valid, decodable PNG.
fn mock_images(n: u32) -> anyhow::Result<Vec<Vec<u8>>> {
    let png = crate::image_util::rgb_to_png(1, 1, &[0, 0, 0])?;
    Ok((0..n).map(|_| png.clone()).collect())
}

/// Base64-encode each PNG blob into the `OpenAI` `{ b64_json }` response shape.
fn encode_pngs(pngs: &[Vec<u8>]) -> Vec<ImageData> {
    use base64ct::Encoding as _;
    pngs.iter()
        .map(|png| ImageData {
            b64_json: base64ct::Base64::encode_string(png),
        })
        .collect()
}

// ============================================================
// Image edits — POST /v1/images/edits (SDXL inpaint + img2img, sdxl-compat-plan)
// ============================================================
//
// The OpenAI image workhorse: `multipart/form-data` with an `image` file, an
// optional `mask` file (alpha PNG, same dims), and a `prompt`. Backing:
//   - mask present → SDXL `InpaintingPipeline` (regenerate the masked region)
//   - mask absent  → SDXL `Image2ImagePipeline` (reimagine the whole frame)
// Single source image only (gpt-image multi-image compositing is a documented
// gap → 400). Response shape is identical to /v1/images/generations.

/// Fields parsed from the `multipart/form-data` edit request. Only the fields the
/// server acts on; `size`/`quality`/`background`/`user` are accepted-and-ignored
/// (output size matches the source image).
struct ImageEditMultipart {
    /// Target image model id (resolved through the manager).
    model: String,
    /// Prompt describing the desired result.
    prompt: String,
    /// Raw source-image file bytes (JPEG/PNG/WebP).
    image: Vec<u8>,
    /// Raw mask file bytes (alpha PNG), or `None` for img2img.
    mask: Option<Vec<u8>>,
    /// Number of images to generate (1–4). Absent → 1.
    n: u32,
    /// Denoise strength in `(0, 1]`; `0.0` → pipeline default. Lower preserves
    /// more of the source (img2img: stays closer to the input). Maps loosely to
    /// `OpenAI`'s `input_fidelity` (inverse). Absent → `0.0`.
    strength: f32,
    /// RNG seed for reproducibility. Absent → pipeline samples fresh noise each call.
    seed: Option<u64>,
}

/// `POST /v1/images/edits` — `OpenAI`-compatible image edit (SDXL inpaint/img2img).
///
/// Parses + validates the multipart request, resolves the `model` through the
/// manager (404 if absent / 400 `WrongKind` if not an image model, 503 + lazy load
/// for an `on_demand` model), decodes the source image (and mask, if present), then
/// runs the SDXL inpaint (mask) or img2img (no mask) pipeline before base64. With
/// no model manager attached (no-GPU test path) it renders canned images.
#[allow(clippy::too_many_lines)]
pub async fn image_edits(
    State(state): State<AppState>,
    authenticated: Option<Extension<Authenticated>>,
    MultipartBody(multipart): MultipartBody,
) -> Response {
    let mut req = match parse_edit_request(multipart).await {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    // Decode the source image (and translate the OpenAI alpha mask → OV mask), image and mask
    // within one request budget (0.9.1 S5). No manager (no-GPU test path): the config defaults.
    let limits = state
        .model_manager
        .as_ref()
        .map_or_else(crate::image_util::ImageLimits::default, |mm| {
            mm.image_limits()
        });
    // Both decoded on the blocking pool, two requests at a time server-wide (0.9.1 S5b).
    let image_bytes = std::mem::take(&mut req.image);
    let mask_bytes = req.mask.take();
    let decoded = crate::image_util::decode_off_runtime(move || {
        let mut budget = crate::image_util::ImageBudget::new(limits);
        let image = crate::image_util::decode_image_bytes(&image_bytes, &mut budget)
            .map_err(|e| ("image", e))?;
        let mask = mask_bytes
            .map(|b| crate::image_util::decode_mask_bytes(&b, &mut budget))
            .transpose()
            .map_err(|e| ("mask", e))?;
        Ok::<_, (&'static str, anyhow::Error)>((image, mask))
    })
    .await;
    let (image, mask) = match decoded {
        Ok(Ok(pair)) => pair,
        Ok(Err((field, e))) => return image_decode_error(field, &e),
        Err(e) => return image_decode_error("image", &e),
    };
    let init = RgbImage {
        width: image.width,
        height: image.height,
        data: image.data,
    };
    // Measured incident: a source image smaller than the pipeline's own downsampling
    // stack can tolerate makes OpenVINO's shape-inference throw ("Kernel after
    // dilation has size ... larger than the data shape after padding") — but the
    // C++ pipeline state is left corrupted by that aborted encode, and the very
    // next operation on it SEGFAULTS the whole process (crashed live at 64×64,
    // confirmed safe at 256×256 — the same floor `parse_size` already treats as
    // the smallest supported *generation* size, so this isn't a new number, just
    // applying the existing one to user-uploaded edit images too). Reject here,
    // before ever reaching the pipeline — this cannot be caught safely once
    // triggered; only preventing it is reliable (full writeup in
    // the project's internal engineering log).
    if init.width < MIN_EDIT_IMAGE_EDGE || init.height < MIN_EDIT_IMAGE_EDGE {
        return bad_request_owned(
            format!(
                "'image' is {}x{}, below the {MIN_EDIT_IMAGE_EDGE}x{MIN_EDIT_IMAGE_EDGE} minimum \
                 this pipeline requires — a smaller source has crashed the server in the past \
                 (OpenVINO shape-inference failure leaves the pipeline unusable)",
                init.width, init.height
            ),
            "image_too_small",
        );
    }
    let mask = match mask {
        Some(d) => {
            if (d.width, d.height) != (init.width, init.height) {
                return bad_request_owned(
                    format!(
                        "'mask' dimensions {}x{} must match 'image' {}x{}",
                        d.width, d.height, init.width, init.height
                    ),
                    "mask_size_mismatch",
                );
            }
            Some(RgbImage {
                width: d.width,
                height: d.height,
                data: d.data,
            })
        }
        None => None,
    };

    tracing::debug!(
        model = %req.model,
        prompt_chars = req.prompt.len(),
        n = req.n,
        width = init.width,
        height = init.height,
        inpaint = mask.is_some(),
        "image edit request"
    );

    // Same pixel-budget guard as /generations: source image dims drive the budget.
    let total_pixels = u64::from(init.width) * u64::from(init.height) * u64::from(req.n);
    if total_pixels > MAX_IMAGE_PIXEL_BUDGET {
        let max_n = MAX_IMAGE_PIXEL_BUDGET / (u64::from(init.width) * u64::from(init.height));
        return bad_request_owned(
            format!(
                "n={} at {}x{} exceeds the per-request pixel budget; try n≤{}",
                req.n, init.width, init.height, max_n
            ),
            "n_exceeds_pixel_budget",
        );
    }

    let mut t0 = std::time::Instant::now();
    let mut hmeta = HandleMetadata::default();
    let mut device = None;
    let pngs = if let Some(mm) = state.model_manager.as_ref() {
        let pixels = u64::from(init.width) * u64::from(init.height) * u64::from(req.n);
        if let Some(resp) = cpu_pixel_limit_refusal(mm, &req.model, pixels) {
            return resp;
        }
        // Slice 3c parity (see image_generations): lazy-load an `on_demand` image model on
        // first request rather than failing flat; F1b: waiting up to `on_demand_wait_secs`.
        let handle = match super::on_demand::resolve_with_wait(mm, &req.model, || {
            mm.get_image_handle(&req.model)
        })
        .await
        {
            Ok(h) => h,
            Err(u) => return u.into_response(&req.model),
        };
        // `elapsed_ms` measures the generation, not an on-demand wait.
        t0 = std::time::Instant::now();
        hmeta = HandleMetadata::from_handle(&handle);
        device = mm
            .device_info_for(handle.model_id())
            .map(DeviceMetadata::from);
        let opts = ImageEditOptions {
            negative_prompt: None,
            num_inference_steps: DEFAULT_NUM_INFERENCE_STEPS,
            num_images: req.n,
            seed: req.seed,
            // 0.0 → the pipeline keeps its built-in guidance / strength defaults.
            guidance_scale: 0.0,
            strength: req.strength,
        };
        match handle.edit(req.prompt.clone(), init, mask, opts).await {
            Ok(p) => p,
            Err(e) => return inference_error_response(Some(mm), handle.model_id(), &e),
        }
    } else {
        // Mock path (no GPU): one canned PNG per requested image. No real
        // device/pipeline behind it, so `model_family`/`device` stay omitted.
        match mock_images(req.n) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!(error = %e, "failed to encode mock image");
                return openai_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "failed to encode image",
                    "internal_error",
                    None,
                );
            }
        }
    };
    let elapsed_ms = t0.elapsed().as_millis();

    tracing::info!(
        model = %req.model,
        n = req.n,
        elapsed_ms,
        "image edit complete"
    );

    Json(ImageGenerationResponse {
        created: unix_now(),
        elapsed_ms,
        data: encode_pngs(&pngs),
        generation_metadata: GenerationMetadata {
            model: req.model.clone(),
            model_hash_short: hmeta.model_hash.as_deref().and_then(model_hash_short),
            model_family: hmeta.model_family,
            model_hash: hmeta.model_hash,
            sampler: hmeta.sampler,
            scheduler_config: hmeta.scheduler_config,
            precision: hmeta.precision,
            model_source: hmeta.model_source,
            model_revision: hmeta.model_revision,
            device,
            // Hostname only for key-validated callers (see `Authenticated`).
            host: authenticated.and_then(|_| crate::os_memory::host_name()),
            engine: "RustedVINO",
            engine_version: env!("CARGO_PKG_VERSION"),
            openvino_version: crate::ov_cb::openvino_version(),
        },
    })
    .into_response()
}

/// Drain the multipart body into a validated [`ImageEditMultipart`], or a boxed
/// 400 `Response` on a malformed body / missing-or-invalid field. Multiple `image`
/// fields (gpt-image multi-image compositing) are rejected — a documented gap.
async fn parse_edit_request(mut multipart: Multipart) -> Result<ImageEditMultipart, Box<Response>> {
    let mut raw = RawEditFields::default();

    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(e) => {
                return Err(Box::new(multipart_error(
                    &e,
                    "malformed multipart/form-data body",
                    "invalid_multipart",
                )));
            }
        };
        let name = field.name().map(ToOwned::to_owned);
        match name.as_deref() {
            // `image` or `image[]` (the array form). A second one = multi-image.
            Some("image" | "image[]") => match field.bytes().await {
                Ok(bytes) => {
                    if raw.image.is_some() {
                        raw.multi_image = true;
                    } else {
                        raw.image = Some(bytes.to_vec());
                    }
                }
                Err(e) => {
                    return Err(Box::new(multipart_error(
                        &e,
                        "could not read field 'image'",
                        "invalid_field",
                    )));
                }
            },
            Some("mask") => match field.bytes().await {
                Ok(bytes) => raw.mask = Some(bytes.to_vec()),
                Err(e) => {
                    return Err(Box::new(multipart_error(
                        &e,
                        "could not read field 'mask'",
                        "invalid_field",
                    )));
                }
            },
            Some("model") => raw.model = field.text().await.ok(),
            Some("prompt") => raw.prompt = field.text().await.ok(),
            Some("n") => raw.n_raw = field.text().await.ok(),
            Some("strength") => raw.strength_raw = field.text().await.ok(),
            Some("seed") => raw.seed_raw = field.text().await.ok(),
            Some("response_format") => raw.response_format = field.text().await.ok(),
            // size, quality, background, user, and unknown fields: drain + ignore.
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    raw.finish()
}

/// Raw multipart fields, collected by the drain loop before validation.
#[derive(Default)]
struct RawEditFields {
    model: Option<String>,
    prompt: Option<String>,
    image: Option<Vec<u8>>,
    mask: Option<Vec<u8>>,
    n_raw: Option<String>,
    response_format: Option<String>,
    strength_raw: Option<String>,
    seed_raw: Option<String>,
    /// Set when a second `image`/`image[]` field arrives (a documented gap).
    multi_image: bool,
}

impl RawEditFields {
    /// Validate the collected fields into an [`ImageEditMultipart`], or a boxed
    /// 400 `Response` on a missing-or-invalid field.
    fn finish(self) -> Result<ImageEditMultipart, Box<Response>> {
        if self.multi_image {
            return Err(Box::new(bad_request(
                "multiple input images are not supported — provide a single source 'image' \
                 (multi-image compositing is unsupported on this server)",
                "multi_image_unsupported",
            )));
        }

        let image = self.image.filter(|f| !f.is_empty()).ok_or_else(|| {
            Box::new(bad_request(
                "missing required field 'image'",
                "missing_image",
            ))
        })?;
        let model = self.model.filter(|m| !m.is_empty()).ok_or_else(|| {
            Box::new(bad_request(
                "missing required field 'model'",
                "missing_model",
            ))
        })?;
        let prompt = self
            .prompt
            .filter(|p| !p.trim().is_empty())
            .ok_or_else(|| {
                Box::new(bad_request(
                    "missing required field 'prompt'",
                    "missing_prompt",
                ))
            })?;

        // `n`: absent → 1; present must parse and fall in 1..=4.
        let n = match self.n_raw.as_deref() {
            None => 1,
            Some(s) => s
                .trim()
                .parse::<u32>()
                .ok()
                .filter(|n| (1..=4).contains(n))
                .ok_or_else(|| Box::new(bad_request("'n' must be between 1 and 4", "invalid_n")))?,
        };

        // `strength`: absent → 0.0 (pipeline default); present must parse and fall
        // in (0, 1]. The C++ side treats <= 0.0 as "leave the pipeline default".
        let strength = match self.strength_raw.as_deref() {
            None => 0.0,
            Some(s) => s
                .trim()
                .parse::<f32>()
                .ok()
                .filter(|v| *v > 0.0 && *v <= 1.0)
                .ok_or_else(|| {
                    Box::new(bad_request(
                        "'strength' must be a number in (0, 1]",
                        "invalid_strength",
                    ))
                })?,
        };

        // Reuse the generations response_format policy: b64_json only, url → 400.
        match self.response_format.as_deref() {
            None | Some("b64_json") => {}
            Some("url") => {
                return Err(Box::new(bad_request(
                    "response_format 'url' is not supported — this server returns b64_json only",
                    "invalid_response_format",
                )));
            }
            Some(_) => {
                return Err(Box::new(bad_request(
                    "unsupported response_format — use b64_json",
                    "invalid_response_format",
                )));
            }
        }

        // `seed`: absent → None (fresh noise); present must parse as u64.
        let seed = match self.seed_raw.as_deref() {
            None => None,
            Some(s) => Some(s.trim().parse::<u64>().map_err(|_| {
                Box::new(bad_request(
                    "'seed' must be a non-negative integer",
                    "invalid_seed",
                ))
            })?),
        };

        Ok(ImageEditMultipart {
            model,
            prompt,
            image,
            mask: self.mask.filter(|m| !m.is_empty()),
            n,
            strength,
            seed,
        })
    }
}

/// A 400 with an owned (runtime-formatted) message — `bad_request` takes a
/// `&'static str`; the decode/dimension errors here need interpolation.
/// D-06: a 400 `size_exceeds_cpu_limit` when an image model on the CPU is asked for more
/// pixels (`n x width x height`) than its memory charge was sized for
/// ([`ModelManager::cpu_image_pixel_limit`]); `None` when the request fits or the model is
/// not a CPU image model.
fn cpu_pixel_limit_refusal(mm: &ModelManager, model: &str, pixels: u64) -> Option<Response> {
    let limit = mm.cpu_image_pixel_limit(model)?;
    (pixels > limit).then(|| cpu_pixel_limit_response(pixels, limit))
}

/// The response of [`cpu_pixel_limit_refusal`] (separate so the wording is unit-testable).
fn cpu_pixel_limit_response(pixels: u64, limit: u64) -> Response {
    bad_request_owned(
        format!(
            "this request asks for {pixels} pixels (n x width x height) but an image model on \
             the CPU serves at most {limit} (cpu_image_gen_max_pixels, the size its memory charge \
             is sized for): lower n or the size, or raise cpu_image_gen_max_pixels and restart"
        ),
        "size_exceeds_cpu_limit",
    )
}

fn bad_request_owned(message: String, code: &'static str) -> Response {
    openai_error(
        StatusCode::BAD_REQUEST,
        message,
        "invalid_request_error",
        Some(code),
    )
}

// ============================================================
// Unit tests
// ============================================================
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    // ── speech_text: number handling on /v1/audio/speech ──────────────────

    #[test]
    fn speech_text_uses_the_model_language_by_default() {
        assert_eq!(
            speech_text("5%".into(), true, None, Some("pl")),
            "pięć procent"
        );
        assert_eq!(
            speech_text("5%".into(), true, None, Some("en")),
            "five percent"
        );
        // No language anywhere: English, as realtime does.
        assert_eq!(speech_text("5%".into(), true, None, None), "five percent");
    }

    #[test]
    fn speech_text_request_language_wins_over_the_model() {
        assert_eq!(
            speech_text("5".into(), true, Some("pl-PL"), Some("en")),
            "pięć"
        );
    }

    /// 0.9.1 C6: Unicode spaces reach no backend, with normalisation on, off, or skipped for a
    /// language without number words; with it on, the digits are then spelled out.
    #[test]
    fn speech_text_maps_unicode_spaces_before_anything_else() {
        assert_eq!(
            speech_text("328\u{202f}feet".into(), true, Some("en"), None),
            "three hundred twenty eight feet"
        );
        assert_eq!(
            speech_text("328\u{202f}feet\u{a0}high".into(), false, None, None),
            "328 feet high"
        );
        assert_eq!(
            speech_text("328\u{2009}Fu\u{2007}ss".into(), true, Some("de"), None),
            "328 Fu ss"
        );
    }

    #[test]
    fn speech_text_opt_out_leaves_the_input_alone() {
        assert_eq!(
            speech_text("W 1948 roku 5%".into(), false, Some("pl"), None),
            "W 1948 roku 5%"
        );
    }

    #[test]
    fn speech_text_unsupported_language_is_not_guessed() {
        // German digits must not be read as English words.
        assert_eq!(
            speech_text("Es sind 5 Äpfel".into(), true, Some("de"), Some("en")),
            "Es sind 5 Äpfel"
        );
    }

    #[test]
    fn speech_request_defaults_normalize_on_and_language_unset() {
        let r: SpeechRequest = serde_json::from_str(r#"{"model":"m","input":"hi"}"#).unwrap();
        assert!(r.normalize);
        assert!(r.language.is_none());
        let r: SpeechRequest =
            serde_json::from_str(r#"{"model":"m","input":"hi","normalize":false,"language":"pl"}"#)
                .unwrap();
        assert!(!r.normalize);
        assert_eq!(r.language.as_deref(), Some("pl"));
    }

    use super::*;

    /// Windows pass 2 D9: an undecodable upload maps to 400 `invalid_audio`; an
    /// engine failure does not (it stays a 500/503 via `inference_error_response`).
    #[test]
    fn bad_audio_is_a_400_and_engine_errors_are_not() {
        let bad = anyhow::Error::new(crate::pipelines::stt::BadAudio(
            "unsupported or malformed audio container".to_owned(),
        ));
        let resp = bad_audio_response(&bad).expect("typed bad audio maps");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert!(bad_audio_response(&anyhow::anyhow!("npu compile failed")).is_none());
    }

    #[test]
    fn response_format_parses_default_and_variants() {
        assert_eq!(ResponseFormat::parse(None), Some(ResponseFormat::Json));
        assert_eq!(
            ResponseFormat::parse(Some("json")),
            Some(ResponseFormat::Json)
        );
        assert_eq!(
            ResponseFormat::parse(Some("text")),
            Some(ResponseFormat::Text)
        );
        assert_eq!(
            ResponseFormat::parse(Some("srt")),
            Some(ResponseFormat::Srt)
        );
        assert_eq!(
            ResponseFormat::parse(Some("verbose_json")),
            Some(ResponseFormat::VerboseJson)
        );
        assert_eq!(ResponseFormat::parse(Some("xml")), None);
    }

    #[test]
    fn srt_timestamp_formats_hms_millis() {
        assert_eq!(srt_timestamp(0.0), "00:00:00,000");
        assert_eq!(srt_timestamp(1.5), "00:00:01,500");
        assert_eq!(srt_timestamp(3661.5), "01:01:01,500");
        // Negative clamps to zero.
        assert_eq!(srt_timestamp(-2.0), "00:00:00,000");
    }

    #[test]
    fn to_srt_emits_indexed_cues() {
        let segments = vec![
            Segment {
                start: 0.0,
                end: 1.5,
                text: " hello ".to_owned(),
            },
            Segment {
                start: 1.5,
                end: 2.0,
                text: "world".to_owned(),
            },
        ];
        let srt = to_srt(&segments);
        // 1-indexed, arrow span, trimmed text, blank-line separated.
        assert!(srt.starts_with("1\n00:00:00,000 --> 00:00:01,500\nhello\n\n"));
        assert!(srt.contains("2\n00:00:01,500 --> 00:00:02,000\nworld\n\n"));
    }

    #[test]
    fn duration_is_last_segment_end() {
        let t = mock_transcription(None);
        assert!((t.duration() - 2.8).abs() < f32::EPSILON);
        assert_eq!(t.language, "english");
    }

    #[test]
    fn mock_language_echoes_hint() {
        let t = mock_transcription(Some("pl"));
        assert_eq!(t.language, "pl");
    }

    // ---- Image generation (5.3a) ------------------------------------

    fn img_req(n: u32, size: Option<&str>, fmt: Option<&str>) -> ImageGenerationRequest {
        ImageGenerationRequest {
            model: "sdxl".to_owned(),
            prompt: "a red circle".to_owned(),
            n,
            size: size.map(ToOwned::to_owned),
            response_format: fmt.map(ToOwned::to_owned),
            steps: None,
            seed: None,
            guidance_scale: None,
            negative_prompt: None,
        }
    }

    /// D-06: the CPU pixel-limit refusal is a 400 `size_exceeds_cpu_limit` naming both
    /// numbers and the config key.
    #[tokio::test]
    async fn cpu_pixel_limit_response_names_the_limit_and_the_key() {
        let resp = cpu_pixel_limit_response(2 * 1024 * 1024, 1024 * 1024);
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "size_exceeds_cpu_limit");
        let msg = json["error"]["message"].as_str().unwrap();
        assert!(msg.contains("2097152") && msg.contains("1048576"), "{msg}");
        assert!(msg.contains("cpu_image_gen_max_pixels"), "{msg}");
    }

    #[test]
    fn parse_size_maps_supported_sizes_and_default() {
        assert_eq!(parse_size(None), Some((1024, 1024)));
        // Squares.
        assert_eq!(parse_size(Some("256x256")), Some((256, 256)));
        assert_eq!(parse_size(Some("512x512")), Some((512, 512)));
        assert_eq!(parse_size(Some("1024x1024")), Some((1024, 1024)));
        // Non-square portrait.
        assert_eq!(parse_size(Some("512x768")), Some((512, 768)));
        assert_eq!(parse_size(Some("768x1024")), Some((768, 1024)));
        assert_eq!(parse_size(Some("1024x1536")), Some((1024, 1536)));
        // Non-square landscape.
        assert_eq!(parse_size(Some("768x512")), Some((768, 512)));
        assert_eq!(parse_size(Some("1024x768")), Some((1024, 768)));
        assert_eq!(parse_size(Some("1536x1024")), Some((1536, 1024)));
        assert_eq!(parse_size(Some("1536x864")), Some((1536, 864)));
        // Unsupported.
        assert_eq!(parse_size(Some("123x456")), None);
        assert_eq!(parse_size(Some("1024")), None);
    }

    #[test]
    fn validate_accepts_non_square_sizes() {
        // Portrait sizes pass through without error.
        let req = img_req(1, Some("512x768"), None);
        let v = validate_image_request(&req).map_err(|_| ()).unwrap();
        assert_eq!((v.width, v.height), (512, 768));

        let req = img_req(1, Some("1024x1536"), None);
        let v = validate_image_request(&req).map_err(|_| ()).unwrap();
        assert_eq!((v.width, v.height), (1024, 1536));
    }

    #[test]
    fn validate_pixel_budget_blocks_n4_at_1024sq() {
        // n=4 @ 1024² = 4M pixels — over budget.
        assert!(validate_image_request(&img_req(4, Some("1024x1024"), None)).is_err());
        // n=3 @ 1024² = 3M pixels — over budget.
        assert!(validate_image_request(&img_req(3, Some("1024x1024"), None)).is_err());
        // n=2 @ 1024² = 2M pixels — exactly at budget.
        assert!(validate_image_request(&img_req(2, Some("1024x1024"), None)).is_ok());
    }

    #[test]
    fn validate_pixel_budget_allows_small_sizes_at_n4() {
        // 512² × 4 = 1M pixels — well under budget.
        assert!(validate_image_request(&img_req(4, Some("512x512"), None)).is_ok());
        // 512×768 × 4 = 1.57M pixels — under budget.
        assert!(validate_image_request(&img_req(4, Some("512x768"), None)).is_ok());
    }

    #[test]
    fn validate_seed_passes_through() {
        let mut req = img_req(1, None, None);
        req.seed = Some(42);
        let v = validate_image_request(&req).map_err(|_| ()).unwrap();
        assert_eq!(v.seed, Some(42));
        // Absent seed → None.
        let v2 = validate_image_request(&img_req(1, None, None))
            .map_err(|_| ())
            .unwrap();
        assert_eq!(v2.seed, None);
    }

    #[test]
    fn validate_accepts_defaults() {
        let v = validate_image_request(&img_req(1, None, None))
            .map_err(|_| ())
            .unwrap();
        assert_eq!((v.width, v.height, v.n), (1024, 1024, 1));
    }

    #[test]
    fn validate_rejects_n_out_of_range() {
        assert!(validate_image_request(&img_req(0, None, None)).is_err());
        assert!(validate_image_request(&img_req(5, None, None)).is_err());
    }

    #[test]
    fn validate_rejects_empty_prompt() {
        let mut req = img_req(1, None, None);
        req.prompt = "   ".to_owned();
        assert!(validate_image_request(&req).is_err());
    }

    #[test]
    fn validate_rejects_url_and_unknown_format() {
        assert!(validate_image_request(&img_req(1, None, Some("url"))).is_err());
        assert!(validate_image_request(&img_req(1, None, Some("xml"))).is_err());
        assert!(validate_image_request(&img_req(1, None, Some("b64_json"))).is_ok());
    }

    #[test]
    fn validate_rejects_bad_size() {
        assert!(validate_image_request(&img_req(1, Some("777x777"), None)).is_err());
    }

    #[test]
    fn validate_steps_default_and_bounds() {
        // Absent → the SDXL-matching default (distilled models pin their own).
        let v = validate_image_request(&img_req(1, None, None))
            .map_err(|_| ())
            .unwrap();
        assert_eq!(v.steps, DEFAULT_NUM_INFERENCE_STEPS);
        // In-range pins through (FLUX schnell wants ~4).
        let mut req = img_req(1, None, None);
        req.steps = Some(4);
        assert_eq!(
            validate_image_request(&req).map_err(|_| ()).unwrap().steps,
            4
        );
        // 0 and >50 are rejected.
        req.steps = Some(0);
        assert!(validate_image_request(&req).is_err());
        req.steps = Some(51);
        assert!(validate_image_request(&req).is_err());
    }

    #[test]
    fn mock_images_emit_n_valid_pngs() {
        let pngs = mock_images(3).expect("mock images");
        assert_eq!(pngs.len(), 3);
        for png in &pngs {
            assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n", "each blob is a PNG");
        }
    }

    #[test]
    fn encode_pngs_wraps_each_as_b64_json() {
        use base64ct::{Base64, Encoding as _};
        let pngs = mock_images(2).expect("mock images");
        let data = encode_pngs(&pngs);
        assert_eq!(data.len(), 2);
        let bytes = Base64::decode_vec(&data[0].b64_json).expect("valid base64");
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "decodes back to a PNG");
    }

    /// `model_hash_short` takes the first 10 hex chars after the `sha256:`
    /// prefix — the A1111 `Model hash:` convention
    /// (the Pyramu image-metadata spec §5.1).
    #[test]
    fn model_hash_short_takes_first_ten_hex_chars_after_prefix() {
        let full = "sha256:1f4a2b3c4d5e6f708192a3b4c5d6e7f8";
        assert_eq!(model_hash_short(full).as_deref(), Some("1f4a2b3c4d"));
    }

    /// A hash without the expected `sha256:` prefix — should never happen given
    /// `compute_model_hash`'s contract, but the defensive `None` path must not
    /// panic or fabricate a short form from the wrong bytes.
    #[test]
    fn model_hash_short_returns_none_for_unexpected_shape() {
        assert_eq!(model_hash_short("not-a-hash"), None);
        assert_eq!(
            model_hash_short("sha256:short"),
            None,
            "fewer than 10 hex chars"
        );
    }

    /// PLAN §5 acceptance criterion: every unset optional field is entirely
    /// absent from the serialized object, never present as `null`. Locks the
    /// `skip_serializing_if` attributes against a future refactor (e.g.
    /// swapping to a derive macro, or a copy-paste that drops one) silently
    /// breaking the pyramu-panel "omit, don't fabricate" contract undetected.
    #[test]
    fn generation_metadata_omits_every_unset_optional_field() {
        let meta = GenerationMetadata {
            model: "test-model".to_owned(),
            model_family: None,
            model_hash: None,
            model_hash_short: None,
            sampler: None,
            scheduler_config: None,
            precision: None,
            model_source: None,
            model_revision: None,
            device: None,
            host: None,
            engine: "RustedVINO",
            engine_version: "0.1.0",
            openvino_version: None,
        };
        let value = serde_json::to_value(&meta).expect("serializes");
        let obj = value.as_object().expect("object");
        let keys: std::collections::BTreeSet<&str> =
            obj.keys().map(std::string::String::as_str).collect();
        assert_eq!(
            keys,
            std::collections::BTreeSet::from(["model", "engine", "engine_version"]),
            "every optional field must be entirely absent when None, never present as null: {obj:?}"
        );
    }

    /// The `device` sub-object follows the same rule independently — an unset
    /// `architecture`/`total_mem_gb` is absent, not `null`, while the
    /// always-present fields still serialize.
    #[test]
    fn device_metadata_omits_unset_optional_sub_fields() {
        let device = DeviceMetadata {
            id: "GPU.1".to_owned(),
            full_name: "Intel(R) Arc(TM) Pro B60 Graphics".to_owned(),
            kind: "discrete-gpu".to_owned(),
            tier: "heavy".to_owned(),
            architecture: None,
            total_mem_gb: None,
        };
        let value = serde_json::to_value(&device).expect("serializes");
        let obj = value.as_object().expect("object");
        let keys: std::collections::BTreeSet<&str> =
            obj.keys().map(std::string::String::as_str).collect();
        assert_eq!(
            keys,
            std::collections::BTreeSet::from(["id", "full_name", "kind", "tier"]),
            "architecture/total_mem_gb must be absent when None, never null: {obj:?}"
        );
    }

    // ── 0.9.1 T1: SpeechT5 input split into pieces, audio joined ──────────

    /// A manager-backed app with one mock TTS model loaded. A `speecht5` name
    /// gets the SpeechT5-like mock (segment cap 200, one sample per character
    /// encoding it); any other name the plain mock (no cap).
    async fn tts_app(model: &str) -> axum::Router {
        tts_app_and_manager(model).await.0
    }

    async fn tts_app_and_manager(model: &str) -> (axum::Router, std::sync::Arc<ModelManager>) {
        let json = format!(
            r#"{{"models_dir":"/tmp/test-t1-tts-models","device":"CPU","preload":[],
                "models":{{"{model}":{{"vram_gb":0.5}}}},"total_vram_gb":22.5,
                "vram_safety_margin_gb":0.0,"min_kv_cache_gb":0.0,"eviction_grace_secs":0.0}}"#
        );
        let cfg: crate::model_manager::Config = serde_json::from_str(&json).unwrap();
        let mm = ModelManager::new(
            cfg,
            std::sync::Arc::new(crate::model_manager::lifecycle::MockEngineFactory::default()),
        )
        .await
        .unwrap();
        mm.load_model(model).await.unwrap();
        let mm = std::sync::Arc::new(mm);
        (
            crate::create_router(AppState::new().with_model_manager(std::sync::Arc::clone(&mm))),
            mm,
        )
    }

    async fn post_speech(
        app: axum::Router,
        body: serde_json::Value,
    ) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
        use tower::ServiceExt as _;
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/audio/speech")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let headers = resp.headers().clone();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec();
        (status, headers, bytes)
    }

    /// The i16 PCM the handler writes for `samples` (same quantisation).
    fn pcm_of(samples: &[f32]) -> Vec<u8> {
        let mut v = Vec::new();
        for s in samples {
            #[allow(clippy::cast_possible_truncation)]
            let i = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
            v.extend_from_slice(&i.to_le_bytes());
        }
        v
    }

    fn header<'a>(h: &'a axum::http::HeaderMap, name: &str) -> Option<&'a str> {
        h.get(name).map(|v| v.to_str().unwrap())
    }

    #[tokio::test]
    async fn speecht5_long_input_is_split_and_the_audio_joined_in_order() {
        let text =
            "The keeper climbed the narrow stairs and lit the great lamp before the sun went down. "
                .repeat(6) + "Ships passed in the night, trusting its steady beam.";
        let text = text.trim().to_owned();
        assert!(text.chars().count() > 400);
        let pieces = crate::tts_split::split_for_speech(&text, 200);
        assert!(pieces.len() >= 3, "{pieces:?}");

        let app = tts_app("speecht5-tts-ov").await;
        let (status, headers, body) = post_speech(
            app,
            serde_json::json!({"model":"speecht5-tts-ov","input":text,"response_format":"pcm"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            header(&headers, "x-rustedvino-tts-segments"),
            Some(pieces.len().to_string().as_str())
        );
        assert_eq!(
            header(&headers, "x-rustedvino-tts-segment-cap"),
            Some("200")
        );
        // The mock answers each piece with one sample per character encoding it:
        // the body is the per-piece outputs concatenated, in order.
        let expected: Vec<f32> = pieces
            .iter()
            .flat_map(|p| crate::pipelines::tts::mock_samples_for(p))
            .collect();
        assert_eq!(body, pcm_of(&expected));
        // And not the single-call output (that would be one long piece).
        assert_ne!(
            body,
            pcm_of(&crate::pipelines::tts::mock_samples_for(&text))
        );
    }

    #[tokio::test]
    async fn speecht5_short_input_is_one_unchanged_piece_and_still_gets_the_headers() {
        // Leading/trailing spaces survive: the text handed to the engine is unchanged.
        let text = "  Hello there, keeper.  ";
        let app = tts_app("speecht5-tts-ov").await;
        let (status, headers, body) = post_speech(
            app,
            serde_json::json!({"model":"speecht5-tts-ov","input":text,"response_format":"pcm"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(header(&headers, "x-rustedvino-tts-segments"), Some("1"));
        assert_eq!(
            header(&headers, "x-rustedvino-tts-segment-cap"),
            Some("200")
        );
        let normalised = speech_text(text.to_owned(), true, None, Some("en"));
        assert_eq!(
            body,
            pcm_of(&crate::pipelines::tts::mock_samples_for(&normalised))
        );
    }

    #[tokio::test]
    async fn speecht5_split_runs_after_normalisation() {
        // 40 four-digit numbers: ~200 chars raw, far more once spelled out.
        let text = "1234 ".repeat(40);
        let normalised = speech_text(text.clone(), true, None, Some("en"));
        let expect_n = crate::tts_split::split_for_speech(&normalised, 200).len();
        assert!(expect_n > 1);
        let app = tts_app("speecht5-tts-ov").await;
        let (status, headers, _) = post_speech(
            app,
            serde_json::json!({"model":"speecht5-tts-ov","input":text}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            header(&headers, "x-rustedvino-tts-segments"),
            Some(expect_n.to_string().as_str())
        );
    }

    #[tokio::test]
    async fn an_engine_without_a_cap_gets_no_split_and_no_headers() {
        let app = tts_app("kokoro-tts-ov").await;
        let (status, headers, body) = post_speech(
            app,
            serde_json::json!({"model":"kokoro-tts-ov","input":"word ".repeat(200),"response_format":"pcm"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(headers.get("x-rustedvino-tts-segments").is_none());
        assert!(headers.get("x-rustedvino-tts-segment-cap").is_none());
        // The plain mock answers its one call with 1 s of 16 kHz silence.
        assert_eq!(body.len(), 16_000 * 2);
    }

    #[tokio::test]
    async fn speecht5_a_failing_second_piece_is_an_error_with_no_headers_and_no_held_slot() {
        let marker = crate::pipelines::tts::MOCK_FAIL_MARKER;
        let first = "The keeper climbed the narrow stairs and lit the great lamp. ".repeat(3);
        let text = format!("{first}Then the {marker} came in the night. And the end came too.");
        let pieces = crate::tts_split::split_for_speech(&text, 200);
        assert!(pieces.len() >= 2 && !pieces[0].contains(marker) && pieces[1].contains(marker));

        let (app, mm) = tts_app_and_manager("speecht5-tts-ov").await;
        let (status, headers, _) = post_speech(
            app,
            serde_json::json!({"model":"speecht5-tts-ov","input":text}),
        )
        .await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(headers.get("x-rustedvino-tts-segments").is_none());
        assert!(headers.get("x-rustedvino-tts-segment-cap").is_none());
        let handle = mm.get_tts_handle("speecht5-tts-ov").unwrap();
        let mut active = usize::MAX;
        for _ in 0..200 {
            active = crate::model_manager::ManagedEngine::active(&handle);
            if active == 0 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        assert_eq!(active, 0, "the request's admission slot is released");
    }

    /// T1 F3 worst case: a 4096-character run of digits is read digit by digit,
    /// so normalisation makes it about six times longer and `SpeechT5` gets well
    /// over 100 pieces — far past the T5 cap of 50, so it is refused (0.9.1 T5).
    /// The numbers here are the ones INTERFACE.md quotes.
    #[test]
    fn worst_case_digit_input_piece_count() {
        let digits = "7".repeat(MAX_SPEECH_INPUT_CHARS);
        let normalised = speech_text(digits, true, None, Some("en"));
        let chars = normalised.chars().count();
        let n = crate::tts_split::split_for_speech(&normalised, 200).len();
        eprintln!("worst case: {chars} chars normalised, {n} pieces");
        assert!((20_000..=30_000).contains(&chars), "{chars}");
        assert!((100..=160).contains(&n), "{n}");
    }

    // ── 0.9.1 T5: SpeechT5 refuses more than 50 pieces per request ────────

    /// `n` sentences that each become one piece of their own after
    /// normalisation (each spells out to more than half the 200 cap, so no two
    /// pack), short enough raw that 51 fit the 4096-character input limit.
    fn speech_text_of_pieces(n: usize) -> String {
        let text = (0..n)
            .map(|i| format!("Call {} 4444 4444 now.", 1000 + i))
            .collect::<Vec<_>>()
            .join(" ");
        assert!(text.chars().count() <= MAX_SPEECH_INPUT_CHARS);
        let normalised = speech_text(text.clone(), true, None, Some("en"));
        let got = crate::tts_split::split_for_speech(&normalised, 200).len();
        assert_eq!(got, n, "the helper must produce exactly {n} pieces");
        text
    }

    #[tokio::test]
    async fn speecht5_input_over_50_pieces_is_refused_before_any_synthesis() {
        const MODEL: &str = "speecht5-t5cap51-tts-ov";
        assert_eq!(crate::tts_split::SPEECHT5_MAX_PIECES, 50);
        let (app, mm) = tts_app_and_manager(MODEL).await;
        let (status, headers, body) = post_speech(
            app,
            serde_json::json!({"model":MODEL,"input":speech_text_of_pieces(51)}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
        assert_eq!(v["error"]["code"], "input_too_long", "{v}");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains("51 pieces"), "{msg}");
        assert!(msg.contains("at most 50 pieces"), "{msg}");
        assert!(headers.get("x-rustedvino-tts-segments").is_none());
        assert_eq!(
            crate::pipelines::tts::mock_pieces_seen(MODEL),
            0,
            "no piece reached the engine"
        );
        let handle = mm.get_tts_handle(MODEL).unwrap();
        assert_eq!(crate::model_manager::ManagedEngine::active(&handle), 0);
    }

    #[tokio::test]
    async fn speecht5_input_of_exactly_50_pieces_is_synthesized() {
        const MODEL: &str = "speecht5-t5cap50-tts-ov";
        let app = tts_app(MODEL).await;
        let (status, headers, _) = post_speech(
            app,
            serde_json::json!({"model":MODEL,"input":speech_text_of_pieces(50),"response_format":"pcm"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(header(&headers, "x-rustedvino-tts-segments"), Some("50"));
        assert_eq!(crate::pipelines::tts::mock_pieces_seen(MODEL), 50);
    }

    #[tokio::test]
    async fn the_piece_cap_does_not_apply_to_an_engine_without_a_segment_cap() {
        // The same 51-piece input, to a non-SpeechT5 engine: one call, 200 OK.
        let app = tts_app("kokoro-t5cap-tts-ov").await;
        let (status, headers, body) = post_speech(
            app,
            serde_json::json!({"model":"kokoro-t5cap-tts-ov","input":speech_text_of_pieces(51),"response_format":"pcm"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(headers.get("x-rustedvino-tts-segments").is_none());
        assert_eq!(body.len(), 16_000 * 2);
    }

    /// A split request whose every piece takes `MOCK_SLOW_PIECE_MS` in the mock.
    fn slow_split_text() -> (String, usize) {
        let marker = crate::pipelines::tts::MOCK_SLOW_MARKER;
        let sentence = format!(
            "The keeper climbed the stairs and the {marker} lit the lamp before night came down."
        );
        let text = vec![sentence; 16].join(" ");
        let n = crate::tts_split::split_for_speech(&text, 200).len();
        assert!(n >= 8, "{n}");
        (text, n)
    }

    /// T1 N1: a forced eviction during a split request stops the request
    /// between pieces with the retryable 503 and completes promptly (no wait for
    /// the remaining pieces, no `EvictionIncomplete`).
    #[tokio::test]
    async fn speecht5_a_forced_eviction_mid_request_ends_it_and_the_eviction_completes() {
        // A model id of its own, so the mock's piece count is this test's alone.
        const MODEL: &str = "speecht5-forced-evict-tts-ov";
        let (text, n) = slow_split_text();
        let (app, mm) = tts_app_and_manager(MODEL).await;
        mm.set_forced_join_deadline_for_test(std::time::Duration::from_secs(2));
        let request = tokio::spawn(post_speech(
            app,
            serde_json::json!({"model":MODEL,"input":text}),
        ));
        // Wait (bounded) until piece 1 is with the engine.
        for _ in 0..1000 {
            if crate::pipelines::tts::mock_pieces_seen(MODEL) >= 1 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        assert!(
            crate::pipelines::tts::mock_pieces_seen(MODEL) >= 1,
            "the request started"
        );

        // Ok, not EvictionIncomplete (the forced join deadline is 2 s here).
        mm.evict_model_forced(MODEL).await.unwrap();
        let (status, headers, _) = request.await.unwrap();

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(headers.get("x-rustedvino-tts-segments").is_none());
        // The piece in flight at the commit finishes; no later piece is sent.
        // Counted, not timed, so a loaded box cannot flake it.
        let seen = crate::pipelines::tts::mock_pieces_seen(MODEL);
        assert!(seen <= 2, "the engine got {seen} of {n} pieces");
        assert!(mm.get_tts_handle(MODEL).is_err(), "the model is gone");
    }

    /// T1 N1: a non-forced eviction refused because the request is running
    /// does not cut it: every piece is synthesized and the response is a 200.
    #[tokio::test]
    async fn speecht5_a_refused_eviction_mid_request_does_not_cut_it() {
        let (text, n) = slow_split_text();
        let piece = std::time::Duration::from_millis(crate::pipelines::tts::MOCK_SLOW_PIECE_MS);
        let (app, mm) = tts_app_and_manager("speecht5-tts-ov").await;
        let request = tokio::spawn(post_speech(
            app,
            serde_json::json!({"model":"speecht5-tts-ov","input":text}),
        ));
        tokio::time::sleep(piece + piece / 3).await; // between / inside pieces 1-2
        let refused = mm.evict_model("speecht5-tts-ov").await.unwrap_err();
        assert!(
            matches!(
                refused.downcast_ref::<crate::model_manager::ModelError>(),
                Some(crate::model_manager::ModelError::TargetBusy(_))
            ),
            "{refused}"
        );
        let (status, headers, _) = request.await.unwrap();
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            header(&headers, "x-rustedvino-tts-segments"),
            Some(n.to_string().as_str())
        );
    }
}
