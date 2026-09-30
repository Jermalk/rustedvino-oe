// ============================================================
// src/handlers/error.rs — OpenAI-compatible error envelope
// ============================================================
// Every error a handler returns is shaped as the OpenAI error body
// `{"error": {"message", "type", "code"}}`. Clients (LiteLLM fallback,
// the OpenAI SDKs, gateways) branch on `type`/`code`, so a bare string
// body is unparseable to them.
//
// This helper is shared by every route that can fail with a client- or
// server-side error (chat completions, tokenize/detokenize, and the
// legacy completions endpoint), so the envelope shape is defined once.
// ============================================================

use std::sync::Arc;

use axum::{
    Json,
    extract::Request,
    http::{HeaderValue, StatusCode, header},
    response::IntoResponse,
    response::Response,
};
use serde::Serialize;

use crate::model_manager::{
    ModelError, ModelManager, is_gpu_poison_error, is_pool_exhausted_error, is_vlm_kv_wedge_error,
    kv_wedge_observed_prompt_tokens, pool_exhausted_active_requests,
};

// ── JsonBody extractor ────────────────────────────────────────────────────────

/// A drop-in replacement for `axum::Json<T>` that maps deserialization errors
/// to the `OpenAI` error envelope (`{"error":{…}}`) instead of axum's default
/// plain-text 400/422.
///
/// Usage in handlers:
/// ```rust,ignore
/// pub async fn my_handler(JsonBody(req): JsonBody<MyRequest>) -> Response { … }
/// ```
pub(crate) struct JsonBody<T>(pub(crate) T);

impl<T, S> axum::extract::FromRequest<S> for JsonBody<T>
where
    T: serde::de::DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(v)) => Ok(Self(v)),
            // Over the request-body limit: say so, instead of calling a valid
            // (just too large) body malformed.
            Err(rejection) if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE => {
                Err(request_too_large())
            }
            Err(rejection) => {
                // T4.4 (committee finding 52): the serde rejection text reflects
                // the caller's input back and fingerprints our struct layout
                // (field names, expected types, byte offsets). Log it server-side
                // for operators; return a fixed generic message to the client so
                // nothing internal is disclosed and no input is echoed.
                tracing::debug!(detail = %rejection.body_text(), "rejected malformed request body");
                Err(openai_error(
                    StatusCode::BAD_REQUEST,
                    "malformed request body: expected valid JSON matching the endpoint schema",
                    "invalid_request_error",
                    Some("invalid_body"),
                ))
            }
        }
    }
}

/// 413 for a request body over the server's limit (`MAX_REQUEST_BODY`),
/// naming the limit. Shared by the JSON extractor and the multipart
/// (audio/image upload) parsers, which used to answer 400 "malformed body" /
/// "could not read field" and hide the cause.
pub(crate) fn request_too_large() -> Response {
    openai_error(
        StatusCode::PAYLOAD_TOO_LARGE,
        format!(
            "request body exceeds this server's {} MiB limit — send a smaller request \
             (for audio, 16 kHz mono is all speech-to-text needs)",
            crate::MAX_REQUEST_BODY >> 20
        ),
        "invalid_request_error",
        Some("request_too_large"),
    )
}

/// OpenAI-compatible error wrapper: `{"error": {...}}`.
#[derive(Serialize)]
struct ErrorEnvelope {
    error: ErrorDetail,
}

/// The error body. `type` is a Rust keyword, hence the serde rename. Field
/// order (`message, type, param, code`) matches the real `OpenAI` envelope.
#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    err_type: &'static str,
    /// The offending request field, when known. Unlike `code`, `OpenAI`
    /// always emits this key — present as JSON `null` when there is no
    /// specific field to blame, never omitted — so it is not
    /// `skip_serializing_if`-gated.
    param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<&'static str>,
}

/// Build an `OpenAI`-shaped error response with the given HTTP status.
///
/// `err_type` is the coarse `OpenAI` category (`invalid_request_error`,
/// `rate_limit_error`, `server_error`, …); `code` is the optional fine-grained
/// machine code (`model_not_found`, `server_overloaded`, …), omitted from the
/// JSON entirely when `None`. `param` is always `null` here — call
/// [`openai_error_with_param`] for errors that can name the offending field.
pub(crate) fn openai_error(
    status: StatusCode,
    message: impl Into<String>,
    err_type: &'static str,
    code: Option<&'static str>,
) -> Response {
    openai_error_with_param(status, message, err_type, code, None)
}

/// [`openai_error`] plus an explicit `param` — the request field the error is
/// about (e.g. `"dimensions"`), matching `OpenAI`'s error envelope exactly.
pub(crate) fn openai_error_with_param(
    status: StatusCode,
    message: impl Into<String>,
    err_type: &'static str,
    code: Option<&'static str>,
    param: Option<&str>,
) -> Response {
    (
        status,
        Json(ErrorEnvelope {
            error: ErrorDetail {
                message: message.into(),
                err_type,
                param: param.map(str::to_owned),
                code,
            },
        }),
    )
        .into_response()
}

/// Build one SSE `data:` frame for a mid-stream inference error — typed,
/// valid-JSON, **and scrubbed** of raw `OpenVINO` C++ diagnostics (T6.2b,
/// committee findings 17/29/30 on the streaming path).
///
/// This is the streaming twin of [`inference_error_from_text`]. Two differences
/// from the non-stream path, both forced by SSE: the response status was already
/// committed as `200` when the first frame went out, so an OOM here **cannot**
/// become a 503 — the best the frame can do is carry a generic, retryable-shaped
/// message. But the side effect still matters: on an OOM-class failure it flags
/// the GPU context as poisoned via `mm`, so the *next* load/request fails fast on
/// the same recovery path the non-stream paths use (`begin_load`'s L3 gate).
///
/// The raw `err` is never reflected into the frame body — `is_gpu_poison_error`
/// classifies it, `mark_gpu_poisoned` logs it server-side, and the client only
/// ever sees one of two fixed messages. One helper for all three streaming sites
/// (chat CB, chat VLM, legacy completions), so there is no second copy to drift.
pub(crate) fn sse_inference_error_event(
    mm: Option<&Arc<ModelManager>>,
    model_id: &str,
    raw: &str,
) -> axum::response::sse::Event {
    match classify_inference_error(mm, model_id, raw) {
        InferenceErrorKind::PoolExhausted => sse_error_frame(
            "the model is momentarily at capacity — retry shortly",
            "pool_exhausted",
        ),
        InferenceErrorKind::VlmKvWedge => sse_error_frame(
            "model temporarily unavailable — recovering from a KV-cache capacity \
             issue; retry shortly",
            "kv_capacity_wedge",
        ),
        InferenceErrorKind::GpuPoisoned => sse_error_frame(
            "model temporarily unavailable — the GPU is recovering; retry shortly",
            "model_unavailable",
        ),
        InferenceErrorKind::Opaque => sse_error_frame("inference failed", "mid_stream_error"),
    }
}

/// Internal: build an SSE error frame from a **constant** message (never
/// interpolated raw text). Typed serialization is the T3.1 guarantee — the
/// message can be re-shaped freely without risking a malformed `data:` line.
fn sse_error_frame(message: &str, code: &'static str) -> axum::response::sse::Event {
    let body = ErrorEnvelope {
        error: ErrorDetail {
            message: message.to_owned(),
            err_type: "server_error",
            param: None,
            code: Some(code),
        },
    };
    // Serializing this struct cannot realistically fail; the fallback is a
    // hand-written *constant* (nothing interpolated), so it is always valid.
    axum::response::sse::Event::default().data(serde_json::to_string(&body).unwrap_or_else(|_| {
        r#"{"error":{"message":"inference error","type":"server_error"}}"#.to_owned()
    }))
}

/// Build the standard 400 for a known `OpenAI` parameter this server cannot
/// honor.
///
/// Per the OpenAI-compat policy, an unsupported but
/// well-known param is **rejected explicitly** rather than silently
/// accepted-and-dropped (the "A1 silent-drop" trap): a client that asked for
/// `logprobs`/`n>1`/`logit_bias` must learn it was not honored. `param` is the
/// JSON field name; `why` is a short human reason. Type/code are fixed
/// (`invalid_request_error` / `unsupported_parameter`).
pub(crate) fn unsupported_param(param: &str, why: &str) -> Response {
    openai_error_with_param(
        StatusCode::BAD_REQUEST,
        format!("parameter '{param}' is not supported by this server: {why}"),
        "invalid_request_error",
        Some("unsupported_parameter"),
        Some(param),
    )
}

/// Map a cross-pipeline device-admission rejection
/// ([`crate::admission::AdmitError`]) to `HTTP 429` — the device-wide analogue
/// of the `rate_limit_error` a per-engine `SubmitResult::AtCapacity` already
/// returns. Shared by every call site that wraps an engine call with
/// `DeviceBudgets::admit` (chat, completions, embeddings, standalone STT/TTS),
/// so the response shape is identical everywhere and no handler file needs to
/// import `crate::admission::AdmitError` directly (it stays fully qualified
/// here only, avoiding any collision with the unrelated `cb_engine::AdmitError`
/// some of these same handler files already reference).
pub(crate) fn device_admission_error_response(e: &crate::admission::AdmitError) -> Response {
    openai_error(
        StatusCode::TOO_MANY_REQUESTS,
        format!(
            "server at capacity — device '{}' busy, retry shortly",
            e.device
        ),
        "rate_limit_error",
        Some("device_at_capacity"),
    )
}

/// `Retry-After` (seconds) advertised on a 503 for a *transient* model state
/// (`Loading`/`Evicting`): the state will change on its own, so a client should
/// retry rather than fail over. A small constant — model JIT load is seconds.
const RETRY_AFTER_TRANSIENT: HeaderValue = HeaderValue::from_static("5");

/// Map a [`ModelError`] to its OpenAI-shaped HTTP response.
///
/// Shared by every endpoint that resolves a model by id (tokenize, detokenize,
/// legacy completions, embeddings, and chat): `NotFound` → 404
/// `model_not_found`; any not-ready state → 503 `model_unavailable`. The
/// *transient* states (`Loading`/`Evicting`) additionally carry a `Retry-After`
/// header so clients retry instead of failing over (co-residency Slice 3b);
/// `NotLoaded` omits it because it will not resolve on its own — an eager model
/// never self-loads.
///
/// On-demand caveat (Slice 3c): `chat`, `embeddings`, `reranking`, and the
/// image/STT media handlers all intercept a `NotLoaded` `on_demand` model to
/// kick off a background load (converting its response to `Loading` +
/// `Retry-After`) before ever reaching here. Every other endpoint routes
/// `NotLoaded` straight here, so an `on_demand` model addressed *first* via
/// `/v1/completions` or `/tokenize` gets a bare 503 and is NOT auto-loaded.
pub(crate) fn model_error_response(e: ModelError) -> Response {
    match e {
        ModelError::NotFound(id) => openai_error(
            StatusCode::NOT_FOUND,
            format!("model '{id}' not found — check /v1/models for available models"),
            "invalid_request_error",
            Some("model_not_found"),
        ),
        ModelError::FilesMissing(id) => openai_error(
            StatusCode::NOT_FOUND,
            format!("model '{id}' is registered but its files were not found on disk"),
            "invalid_request_error",
            Some("model_not_found"),
        ),
        ModelError::FilesIncomplete(id, missing) => openai_error(
            StatusCode::NOT_FOUND,
            format!(
                "model '{id}' is missing required files: {} — re-run the model \
                 conversion/export for this model",
                missing.join(", ")
            ),
            "invalid_request_error",
            Some("model_not_found"),
        ),
        // All three not-ready states share the 503 `model_unavailable` envelope;
        // only the *transient* ones (`Loading`/`Evicting`) advertise
        // `Retry-After` — they resolve on their own, whereas `NotLoaded` will not
        // (an eager model never self-loads).
        ModelError::NotLoaded | ModelError::Loading | ModelError::Evicting => {
            let mut resp = openai_error(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("model unavailable: {e}"),
                "server_error",
                Some("model_unavailable"),
            );
            if matches!(e, ModelError::Loading | ModelError::Evicting) {
                resp.headers_mut()
                    .insert(header::RETRY_AFTER, RETRY_AFTER_TRANSIENT);
            }
            resp
        }
        ModelError::GpuPoisoned => openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!(
                "{e} — restart rustedvino to recover (a graceful stop via SIGTERM/systemctl drains in-flight requests and frees VRAM first)"
            ),
            "server_error",
            Some("gpu_poisoned"),
        ),
        ModelError::WrongKind(msg) => openai_error(
            StatusCode::BAD_REQUEST,
            msg,
            "invalid_request_error",
            Some("unsupported_operation"),
        ),
    }
}

/// Map an inference-path failure (`embed` / generate) to an `OpenAI` error
/// response **without ever reflecting the raw C++ diagnostics** into the client
/// body (T6.2, committee findings 17/29/30).
///
/// `OpenVINO` surfaces failures as raw C++ strings that leak model paths and
/// `OpenCL` internals; interpolating `{e}` into the body (as `embeddings.rs`
/// did) is a disclosure. Instead:
/// - OOM-class errors (`CL_OUT_OF_RESOURCES` / `CL_INVALID_EVENT`) become a
///   retryable 503 `model_unavailable` and flag the GPU context as poisoned via
///   `mm` so later loads fail fast (mirrors the load-path L3 gate);
/// - CB-scheduler KV-pool exhaustion (`GenerationStatus::IGNORED` — one
///   request's generation never ran because the pool had no room) becomes a
///   retryable 503 `pool_exhausted` — no `mark_gpu_poisoned`, since unlike a
///   real `OpenCL` OOM this is scoped to the one request, not the whole
///   context (the project's internal engineering log). When no
///   *other* request was in flight against the model at the time (so
///   nothing else could have caused the shortfall), this also triggers the
///   same evict+reload+ratchet recovery as the VLM wedge below
///   (the project's internal engineering log) — a
///   genuinely busy model with real concurrent load does not get evicted for
///   an ordinary capacity blip;
/// - anything else is an opaque 500 `server_error`.
///
/// The raw `err` is logged server-side only. `mm` is `None` on the mock path
/// (no GPU), where this is never reached in practice.
pub(crate) fn inference_error_response(
    mm: Option<&Arc<ModelManager>>,
    model_id: &str,
    err: &anyhow::Error,
) -> Response {
    inference_error_from_text(mm, model_id, &err.to_string())
}

/// `&str` form of [`inference_error_response`] for the streaming collectors,
/// whose `drain_channel` returns the raw engine error as a `String`
/// (`StreamEvent::Error`). Same mapping, same scrub: callers pass the raw text
/// and get back a body that never contains it.
pub(crate) fn inference_error_from_text(
    mm: Option<&Arc<ModelManager>>,
    model_id: &str,
    raw: &str,
) -> Response {
    match classify_inference_error(mm, model_id, raw) {
        InferenceErrorKind::PoolExhausted => openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "the model is momentarily at capacity — retry shortly",
            "server_error",
            Some("pool_exhausted"),
        ),
        InferenceErrorKind::VlmKvWedge => openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "model temporarily unavailable — recovering from a KV-cache capacity \
             issue; retry shortly",
            "server_error",
            Some("kv_capacity_wedge"),
        ),
        InferenceErrorKind::GpuPoisoned => openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "model temporarily unavailable — the GPU is recovering; retry shortly",
            "server_error",
            Some("model_unavailable"),
        ),
        InferenceErrorKind::Opaque => openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "inference failed",
            "server_error",
            None,
        ),
    }
}

/// Which class of inference failure `raw` represents, after applying its
/// side effect (poisoning the GPU context, spawning VLM-wedge recovery, or
/// nothing) — shared by [`inference_error_from_text`] and
/// [`sse_inference_error_event`] so the two response shapers (JSON body vs.
/// SSE frame) can never drift on which markers they recognize or what side
/// effect each one triggers.
enum InferenceErrorKind {
    /// CB-scheduler KV-pool exhaustion (`GenerationStatus::IGNORED`). Usually
    /// scoped to the one request — other requests against this model are
    /// fine, and the pool just fluctuates under real multi-tenant load. But
    /// when [`pool_exhausted_active_requests`] reads `<= 1` (this was the
    /// only request against the model), nothing else could have crowded out
    /// the pool — that's not congestion, it's the static formula being
    /// wrong, so `spawn_kv_wedge_recovery` is triggered the same as
    /// [`InferenceErrorKind::VlmKvWedge`]
    /// (the project's internal engineering log).
    PoolExhausted,
    /// VLM-path KV-admission wedge (`VLM_KV_WEDGE_MARKER`) — the pipeline
    /// itself never self-recovers; `spawn_kv_wedge_recovery` was triggered
    /// (evict + reload the model, learn a tighter ceiling).
    VlmKvWedge,
    /// Whole-`OpenCL`-context OOM (`is_gpu_poison_error`) — every model on
    /// this device is affected, not just this one.
    GpuPoisoned,
    /// Anything else — an opaque 500, raw text logged server-side only.
    Opaque,
}

fn classify_inference_error(
    mm: Option<&Arc<ModelManager>>,
    model_id: &str,
    raw: &str,
) -> InferenceErrorKind {
    if is_pool_exhausted_error(raw) {
        // Deliberately NOT mark_gpu_poisoned: this is one request's own
        // scheduling failure (KV pool momentarily full), not a poisoned
        // OpenCL context — other requests against this model are fine.
        //
        // `active_requests <= 1` (this was the only request against the
        // model when it failed) means nothing else could have been crowding
        // out the pool — that's not the transient multi-tenant congestion
        // this 503 is framed for, it's the static formula being wrong for
        // this model, same root cause as the VLM wedge below. A parse
        // failure (message shape drifted) conservatively does NOT trigger
        // recovery: unlike the VLM wedge, plain pool exhaustion usually
        // *is* legitimate and retryable, so misreading "solo" is the wrong
        // direction to be wrong in (see `POOL_EXHAUSTED_MARKER`'s doc
        // comment and the project's internal engineering log).
        let active_requests = pool_exhausted_active_requests(raw);
        if active_requests.is_some_and(|n| n <= 1) {
            let observed_prompt_tokens = kv_wedge_observed_prompt_tokens(raw).unwrap_or(0);
            if let Some(mm) = mm {
                mm.spawn_kv_wedge_recovery(model_id, observed_prompt_tokens);
            }
            tracing::warn!(
                model_id,
                error = %raw,
                observed_prompt_tokens,
                "generation ignored — KV pool exhausted with no other requests in flight, \
                 treating as a static-formula miss and recovering"
            );
        } else {
            tracing::warn!(model_id, error = %raw, "generation ignored — KV pool exhausted, retryable");
        }
        return InferenceErrorKind::PoolExhausted;
    }
    if is_vlm_kv_wedge_error(raw) {
        // Parse failure (message shape drifted) still triggers recovery —
        // spawn_kv_wedge_recovery treats an implausibly small/zero token
        // count as "skip the ratchet write, still evict+reload" (see its own
        // doc comment), so Detect+Error+Reload never depend on this parse
        // succeeding, only the Ratchet refinement does.
        let observed_prompt_tokens = kv_wedge_observed_prompt_tokens(raw).unwrap_or(0);
        if let Some(mm) = mm {
            mm.spawn_kv_wedge_recovery(model_id, observed_prompt_tokens);
        }
        tracing::error!(
            model_id,
            error = %raw,
            observed_prompt_tokens,
            "VLM pipeline produced zero tokens — KV-admission capacity wedge"
        );
        return InferenceErrorKind::VlmKvWedge;
    }
    if is_gpu_poison_error(raw) {
        // mark_gpu_poisoned logs the raw diagnostics server-side.
        if let Some(mm) = mm {
            mm.mark_gpu_poisoned(model_id, raw);
        }
        return InferenceErrorKind::GpuPoisoned;
    }
    tracing::error!(model_id, error = %raw, "inference failed");
    InferenceErrorKind::Opaque
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    /// `openai_error` serialises the `{error:{message,type,code}}` envelope.
    #[tokio::test]
    async fn openai_error_serialises_envelope() {
        let resp = openai_error(
            StatusCode::NOT_FOUND,
            "model 'gpt-4' not found",
            "invalid_request_error",
            Some("model_not_found"),
        );
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["message"], "model 'gpt-4' not found");
        // `type` is the serialised key (Rust keyword renamed via serde).
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["code"], "model_not_found");
    }

    /// T6.2: an OOM-class inference failure maps to a retryable 503 and the raw
    /// C++ diagnostics (model paths, `OpenCL` codes) never reach the client body.
    #[tokio::test]
    async fn inference_oom_is_retryable_503_with_scrubbed_body() {
        let raw = anyhow::anyhow!(
            "ov_embed_documents failed: /opt/models/secret-bge CL_OUT_OF_RESOURCES (-5)"
        );
        let resp = inference_error_response(None, "bge-embed-ov", &raw);
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "model_unavailable");
        let msg = json["error"]["message"].as_str().unwrap();
        assert!(
            !msg.contains("CL_OUT_OF_RESOURCES"),
            "OpenCL code leaked: {msg}"
        );
        assert!(!msg.contains("secret-bge"), "model path leaked: {msg}");
    }

    /// T6.2: a non-OOM inference failure is an opaque 500 — the raw error text
    /// is not reflected into the body.
    #[tokio::test]
    async fn inference_generic_error_is_opaque_500() {
        let raw = anyhow::anyhow!("ov_embed_documents failed: /opt/models/x ragged batch");
        let resp = inference_error_response(None, "bge-embed-ov", &raw);
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let msg = json["error"]["message"].as_str().unwrap();
        assert_eq!(msg, "inference failed");
        assert!(!msg.contains("ragged batch"), "raw error leaked: {msg}");
    }

    /// T6.2b: an OOM-class mid-stream error produces a scrubbed SSE frame — the
    /// raw `OpenCL` code and model path never reach the body — shaped as a
    /// retryable `model_unavailable` envelope. (`mm` is `None` here: the scrub is
    /// what we assert; the poison side effect reuses the load-path-tested
    /// `mark_gpu_poisoned`.)
    #[test]
    fn sse_inference_oom_frame_is_scrubbed() {
        let raw = "ov_cb generate failed: /opt/models/secret-llm CL_OUT_OF_RESOURCES (-5)";
        let frame = format!("{:?}", sse_inference_error_event(None, "qwen3-8b-ov", raw));
        assert!(
            !frame.contains("CL_OUT_OF_RESOURCES"),
            "OpenCL code leaked into SSE frame: {frame}"
        );
        assert!(
            !frame.contains("secret-llm"),
            "model path leaked into SSE frame: {frame}"
        );
        assert!(
            frame.contains("model_unavailable"),
            "OOM frame must carry the retryable code: {frame}"
        );
    }

    /// T6.2b: a non-OOM mid-stream error is an opaque "inference failed" SSE
    /// frame — the raw engine text is not reflected.
    #[test]
    fn sse_inference_generic_frame_is_scrubbed() {
        let raw = "ov_cb generate failed: /opt/models/x ragged batch boom";
        let frame = format!("{:?}", sse_inference_error_event(None, "qwen3-8b-ov", raw));
        assert!(
            frame.contains("inference failed"),
            "generic frame must use the fixed message: {frame}"
        );
        assert!(
            !frame.contains("ragged batch"),
            "raw error leaked into SSE frame: {frame}"
        );
        assert!(
            frame.contains("mid_stream_error"),
            "generic frame must carry the mid_stream_error code: {frame}"
        );
    }

    /// A `None` code is omitted from the envelope (not serialised as null).
    #[tokio::test]
    async fn openai_error_omits_code_when_none() {
        let resp = openai_error(
            StatusCode::BAD_REQUEST,
            "messages must not be empty",
            "invalid_request_error",
            None,
        );
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            json["error"].get("code").is_none(),
            "code key must be omitted when None: {json}"
        );
    }

    /// Unlike `code`, `param` must always be present — `null` when unset —
    /// matching the real `OpenAI` envelope, which never omits the key.
    #[tokio::test]
    async fn openai_error_param_key_present_as_null_by_default() {
        let resp = openai_error(
            StatusCode::BAD_REQUEST,
            "messages must not be empty",
            "invalid_request_error",
            None,
        );
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            json["error"]
                .get("param")
                .is_some_and(serde_json::Value::is_null),
            "param key must be present as null, not omitted: {json}"
        );
    }

    /// `unsupported_param` names the offending field in both `message` and
    /// the structured `param` key.
    #[tokio::test]
    async fn unsupported_param_populates_param_field() {
        let resp = unsupported_param("dimensions", "vector truncation is not supported");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["param"], "dimensions");
        assert_eq!(json["error"]["code"], "unsupported_parameter");
    }

    /// Slice 3b: the *transient* model states (`Loading`/`Evicting`) carry a
    /// `Retry-After` header so clients retry instead of failing over, while
    /// `NotLoaded` (no auto-load for an eager model) deliberately omits it. All
    /// three remain 503 `model_unavailable`.
    #[tokio::test]
    async fn transient_states_carry_retry_after_notloaded_does_not() {
        for (e, want_header) in [
            (ModelError::Loading, true),
            (ModelError::Evicting, true),
            (ModelError::NotLoaded, false),
        ] {
            let resp = model_error_response(e);
            assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
            let has_retry = resp.headers().contains_key(header::RETRY_AFTER);
            assert_eq!(
                has_retry, want_header,
                "Retry-After presence mismatch for a 503 state (want {want_header})"
            );
            if want_header {
                assert_eq!(
                    resp.headers().get(header::RETRY_AFTER).unwrap(),
                    "5",
                    "Retry-After must advertise the transient-state delay"
                );
            }
        }
    }
}
