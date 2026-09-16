// ============================================================
// src/handlers/media.rs — Phase 5 media endpoints (STT / TTS / Image)
// ============================================================
// The three OpenAI media routes. STT (`/v1/audio/transcriptions`) is the first
// to grow real request/response handling; TTS and Image stay 501 stubs until
// their sub-plans land (dev/plans/phase5/{TTS,IMAGE}.md).
//
//   POST /v1/audio/transcriptions → STT/Whisper  (phase5/STT.md) — 5.1a here
//   POST /v1/audio/speech         → TTS          (phase5/TTS.md)  — 501 stub
//   POST /v1/images/generations   → Image/SDXL   (phase5/IMAGE.md)— 501 stub
//
// 5.1a scope: parse the multipart request, validate it, and render all four
// OpenAI `response_format`s from a CANNED transcription — no GPU, no model
// resolution. The real Whisper pipeline + model resolution (404/WrongKind) land
// in 5.1b/5.1c.
// ============================================================

use axum::{
    Json,
    extract::{Multipart, State},
    http::StatusCode,
    response::{IntoResponse as _, Response},
};
use serde::{Deserialize, Serialize};

use super::chat::unix_now;
use super::error::{
    JsonBody, device_admission_error_response, inference_error_response, model_error_response,
    openai_error,
};
use crate::app_state::AppState;
use crate::device_inventory::DeviceInfo;
use crate::model_manager::{ModelError, OnDemandLoad};
use crate::ov_image::{ImageEditOptions, ImageGenOptions, RgbImage};
use crate::pipelines::stt::{Segment, Transcription};
use crate::pipelines::tts::TTS_SAMPLE_RATE;

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
pub async fn transcriptions(State(state): State<AppState>, multipart: Multipart) -> Response {
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
        "transcription request"
    );

    let transcription = if let Some(mm) = state.model_manager.as_ref() {
        let handle = match mm.get_stt_handle(&req.model) {
            Ok(h) => h,
            // Slice 3c parity: a NotLoaded model declared `load: on_demand` loads
            // lazily on first use — kick off a background load and tell the client
            // to retry (503 Loading + Retry-After). Eager models keep failing fast
            // with a bare NotLoaded 503. Mirrors the chat path (`chat.rs`).
            Err(ModelError::NotLoaded) => {
                return match mm.request_on_demand_load(&req.model) {
                    OnDemandLoad::Loading => model_error_response(ModelError::Loading),
                    OnDemandLoad::NotApplicable => model_error_response(ModelError::NotLoaded),
                };
            }
            Err(e) => return model_error_response(e),
        };

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

        match handle.transcribe(req.file, req.language, timestamps).await {
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
                return inference_error_response(Some(mm), handle.model_id(), &e);
            }
        }
    } else {
        // Mock path (no GPU): a canned transcription so the route and all four
        // response formats are exercisable in integration tests without a model.
        mock_transcription(req.language.as_deref())
    };

    render(&transcription, req.response_format)
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
            Err(_) => {
                return Err(bad_request(
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
                Err(_) => return Err(bad_request("could not read field 'file'", "invalid_field")),
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
/// JSON; `text`/`srt` are `text/plain` bodies.
fn render(t: &Transcription, format: ResponseFormat) -> Response {
    match format {
        ResponseFormat::Json => Json(JsonResponse { text: &t.text }).into_response(),
        ResponseFormat::Text => t.text.clone().into_response(),
        ResponseFormat::Srt => to_srt(&t.segments).into_response(),
        ResponseFormat::VerboseJson => Json(VerboseResponse {
            task: "transcribe",
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

/// Build a 400 `invalid_request_error` with the shared `OpenAI` envelope.
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

    if req.input.trim().is_empty() {
        return bad_request("input must not be empty", "missing_input");
    }

    // `sample_rate` is split from `samples` so the WAV header uses the correct rate
    // per backend (SpeechT5 = 16 kHz, Kokoro = 24 kHz). The mock path defaults to
    // `TTS_SAMPLE_RATE` (SpeechT5) since it doesn't have a live handle.
    let sample_rate;
    let samples = if let Some(mm) = state.model_manager.as_ref() {
        let handle = match mm.get_tts_handle(&req.model) {
            Ok(h) => h,
            Err(crate::model_manager::ModelError::NotLoaded) => {
                return match mm.request_on_demand_load(&req.model) {
                    OnDemandLoad::Loading => {
                        model_error_response(crate::model_manager::ModelError::Loading)
                    }
                    OnDemandLoad::NotApplicable => {
                        model_error_response(crate::model_manager::ModelError::NotLoaded)
                    }
                };
            }
            Err(e) => return model_error_response(e),
        };

        tracing::debug!(
            model = %req.model,
            input_chars = req.input.len(),
            fmt,
            "TTS request"
        );

        sample_rate = handle.sample_rate();

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

        match handle.synthesize(req.input, req.voice, req.speed).await {
            Ok(s) => s,
            Err(crate::cb_engine::AdmitError::AtCapacity) => {
                return openai_error(
                    StatusCode::TOO_MANY_REQUESTS,
                    "server at capacity — all inference slots busy, retry shortly",
                    "rate_limit_error",
                    Some("server_overloaded"),
                );
            }
            Err(crate::cb_engine::AdmitError::EngineDead) => {
                return openai_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "TTS engine unavailable (engine thread exited)",
                    "server_error",
                    Some("engine_unavailable"),
                );
            }
            Err(crate::cb_engine::AdmitError::Failed(e)) => {
                return inference_error_response(Some(mm), handle.model_id(), &e);
            }
        }
    } else {
        // Mock path (no GPU): one second of silence so the route and both
        // response formats are exercisable in integration tests without a model.
        sample_rate = TTS_SAMPLE_RATE;
        vec![0.0_f32; TTS_SAMPLE_RATE as usize]
    };

    match fmt {
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
    }
}

// ============================================================
// Image generation — POST /v1/images/generations (SDXL, phase5/IMAGE.md)
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
/// `generation_metadata` — `PLAN_image_metadata_response.md`): `{ created,
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
/// shape — all three tiers of `PLAN_image_metadata_response.md` (model
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
/// short form (`pyramu-image-metadata-spec.md` §5.1: "first 10 hex chars of the
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

    let t0 = std::time::Instant::now();
    let mut hmeta = HandleMetadata::default();
    let mut device = None;
    let pngs = if let Some(mm) = state.model_manager.as_ref() {
        let handle = match mm.get_image_handle(&req.model) {
            Ok(h) => h,
            // Slice 3c parity (see the STT path above): lazy-load an `on_demand`
            // image model on first request rather than failing flat.
            Err(ModelError::NotLoaded) => {
                return match mm.request_on_demand_load(&req.model) {
                    OnDemandLoad::Loading => model_error_response(ModelError::Loading),
                    OnDemandLoad::NotApplicable => model_error_response(ModelError::NotLoaded),
                };
            }
            Err(e) => return model_error_response(e),
        };
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
            host: crate::os_memory::host_name(),
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
/// (output size matches the source image — see sdxl-compat-plan §Gaps).
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
pub async fn image_edits(State(state): State<AppState>, multipart: Multipart) -> Response {
    let req = match parse_edit_request(multipart).await {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    // Decode the source image (and translate the OpenAI alpha mask → OV mask).
    let init = match crate::image_util::decode_image_bytes(&req.image) {
        Ok(d) => RgbImage {
            width: d.width,
            height: d.height,
            data: d.data,
        },
        Err(e) => {
            return bad_request_owned(format!("could not decode 'image': {e}"), "invalid_image");
        }
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
    // triggered; only preventing it is reliable. Full writeup:
    // dev/autotest/20260804_image_edit_small_image_segfault.md.
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
    let mask = match req.mask.as_deref() {
        Some(bytes) => match crate::image_util::decode_mask_bytes(bytes) {
            Ok(d) => {
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
            Err(e) => {
                return bad_request_owned(format!("could not decode 'mask': {e}"), "invalid_mask");
            }
        },
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

    let t0 = std::time::Instant::now();
    let mut hmeta = HandleMetadata::default();
    let mut device = None;
    let pngs = if let Some(mm) = state.model_manager.as_ref() {
        let handle = match mm.get_image_handle(&req.model) {
            Ok(h) => h,
            // Slice 3c parity (see image_generations): lazy-load an `on_demand`
            // image model on first request rather than failing flat.
            Err(ModelError::NotLoaded) => {
                return match mm.request_on_demand_load(&req.model) {
                    OnDemandLoad::Loading => model_error_response(ModelError::Loading),
                    OnDemandLoad::NotApplicable => model_error_response(ModelError::NotLoaded),
                };
            }
            Err(e) => return model_error_response(e),
        };
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
            host: crate::os_memory::host_name(),
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
            Err(_) => {
                return Err(Box::new(bad_request(
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
                Err(_) => {
                    return Err(Box::new(bad_request(
                        "could not read field 'image'",
                        "invalid_field",
                    )));
                }
            },
            Some("mask") => match field.bytes().await {
                Ok(bytes) => raw.mask = Some(bytes.to_vec()),
                Err(_) => {
                    return Err(Box::new(bad_request(
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

    use super::*;

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
    /// (`pyramu-image-metadata-spec.md` §5.1).
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
}
