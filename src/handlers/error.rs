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
    ModelError, ModelManager, is_genai_memory_budget_error, is_gpu_poison_error,
    is_pool_exhausted_error, is_vlm_kv_wedge_error, kv_wedge_observed_prompt_tokens,
    pool_exhausted_active_requests, pool_exhausted_generated_tokens,
};

/// Marker of `GenAI`'s chat-template failure (the vision pipeline renders the template itself).
const TEMPLATE_ERROR_MARKER: &str = "Minja's error: ";

/// The template's own reason from a `GenAI` chat-template failure (SF-5 D-03), e.g.
/// `Unexpected message role.` from a `raise_exception` — text written by the template's author,
/// cut before the row/column and template excerpt, one line, at most 200 characters. `None`
/// when `raw` is not a template failure.
fn template_rejection_reason(raw: &str) -> Option<String> {
    let after = &raw[raw.find(TEMPLATE_ERROR_MARKER)? + TEMPLATE_ERROR_MARKER.len()..];
    let line = after.lines().next().unwrap_or_default();
    let reason = line.find(" at row ").map_or(line, |i| &line[..i]).trim();
    let reason: String = reason.chars().take(200).collect();
    Some(if reason.is_empty() {
        "(no reason given)".to_owned()
    } else {
        reason
    })
}

/// The client text for [`InferenceErrorKind::TemplateRejected`] (both response shapes).
fn template_rejected_message(reason: &str) -> String {
    format!("the model's chat template rejected the messages: {reason}")
}

/// The client text for [`InferenceErrorKind::DeviceMemory`] (both response shapes).
const DEVICE_MEMORY_MESSAGE: &str = "the device has too little free memory for this model's KV \
     cache; the model is being reloaded. Retry in a few seconds, or unload another model";

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
                let detail = rejection.body_text();
                tracing::debug!(%detail, "rejected malformed request body");
                // 0.8.3 (F1): name the offending field, never the value or the
                // serde type detail. Field names are the public API surface.
                let message = field_hint(&detail).map_or_else(
                    || GENERIC_BODY_MESSAGE.to_owned(),
                    |hint| format!("malformed request body: {hint}"),
                );
                Err(openai_error(
                    StatusCode::BAD_REQUEST,
                    &message,
                    "invalid_request_error",
                    Some("invalid_body"),
                ))
            }
        }
    }
}

const GENERIC_BODY_MESSAGE: &str =
    "malformed request body: expected valid JSON matching the endpoint schema";

// ── Admin JSON bodies (P-C, P8) ──────────────────────────────────────────────

/// Whether an admin route keeps the 422 for an unknown field (`PATCH /v1/admin/models/{id}`
/// does, at any depth: its body is `deny_unknown_fields`) or answers it like any other bad
/// body (400 `invalid_body`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum UnknownField {
    /// The pre-0.9.1 axum answer, unchanged: 422, plain text.
    Keep422,
    /// 400 `invalid_body`, like a syntax error or a wrong type.
    InvalidBody,
}

/// The answer to an admin route's JSON body that axum's `Json` refused (P-C,
/// P8): malformed JSON, a wrong type and a missing field are the public 400 `invalid_body`
/// envelope; a missing `Content-Type` (415) and an over-limit body (413) are axum's own
/// answers, as before; an unknown field is a 422 or a 400 per `unknown`.
///
/// Unlike the inference routes ([`JsonBody`]), the message keeps serde's detail (the field path,
/// and the unknown field's name where there is one): these routes need an admin key, and the
/// operator needs the typo spelled out (`sampling.temprature`).
pub(crate) fn admin_json_rejection(
    rejection: axum::extract::rejection::JsonRejection,
    unknown: UnknownField,
) -> Response {
    use axum::extract::rejection::JsonRejection;
    match rejection {
        JsonRejection::JsonDataError(_)
            if unknown == UnknownField::Keep422
                && is_unknown_field_error(&rejection.body_text()) =>
        {
            rejection.into_response()
        }
        JsonRejection::JsonDataError(_) | JsonRejection::JsonSyntaxError(_) => {
            admin_invalid_body(&rejection.body_text())
        }
        other => other.into_response(),
    }
}

/// [`admin_json_rejection`] for a route that reads its body as bytes (`/load`, `/resize`: any
/// `Content-Type`, and `/load` also an empty body): parse `body` as `T`, or the 400
/// `invalid_body` envelope (an unknown field included). The error is boxed: a `Response` is
/// large (`clippy::result_large_err`).
pub(crate) fn admin_json_from_bytes<T: serde::de::DeserializeOwned>(
    body: &[u8],
) -> Result<T, Box<Response>> {
    Json::<T>::from_bytes(body)
        .map(|Json(v)| v)
        .map_err(|r| Box::new(admin_json_rejection(r, UnknownField::InvalidBody)))
}

/// axum's prefix of a [`JsonRejection::JsonDataError`](axum::extract::rejection::JsonRejection)
/// text (a body that is JSON but does not match the type).
const AXUM_DATA_PREFIX: &str = "Failed to deserialize the JSON body into the target type: ";
/// axum's prefix of a `JsonSyntaxError` text (a body that is not JSON).
const AXUM_SYNTAX_PREFIX: &str = "Failed to parse the request body as JSON: ";

/// Whether axum's data-error text `detail` is serde's unknown-field error, by its shape (P-C
/// review R-3), not by a substring a string value could echo (`{"vram_gb": "unknown field
/// `x`"}` is a wrong type): after axum's prefix comes the field path, `: `, then serde's
/// message, which starts with ``unknown field `<name>`, `` followed by `expected` or
/// `there are no fields`.
fn is_unknown_field_error(detail: &str) -> bool {
    let Some(rest) = detail.strip_prefix(AXUM_DATA_PREFIX) else {
        return false;
    };
    // The path comes first (`sampling`, `.`); serde's message after the first `: `.
    let message = rest.split_once(": ").map_or(rest, |(_, m)| m);
    message
        .strip_prefix("unknown field `")
        .and_then(|m| m.split_once("`, "))
        .is_some_and(|(_, after)| {
            after.starts_with("expected") || after.starts_with("there are no fields")
        })
}

/// The 400 `invalid_body` envelope with serde's detail, axum's prefix dropped.
fn admin_invalid_body(detail: &str) -> Response {
    let detail = detail
        .strip_prefix(AXUM_DATA_PREFIX)
        .or_else(|| detail.strip_prefix(AXUM_SYNTAX_PREFIX))
        .unwrap_or(detail);
    openai_error(
        StatusCode::BAD_REQUEST,
        format!("invalid JSON body: {detail}"),
        "invalid_request_error",
        Some("invalid_body"),
    )
}

/// A path segment is safe to echo only if it is a plain identifier/index path.
fn is_safe_path(p: &str) -> bool {
    !p.is_empty()
        && p.len() <= 64
        && p.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '[' | ']'))
}

/// Extracts "which field" from axum's serde rejection text, e.g.
/// `Failed to deserialize ...: messages[0].role: invalid type: ...` or
/// `... missing field `model` at line 1 column 2`. Returns only a field path
/// (charset-checked, length-capped); values and type detail are never echoed.
fn field_hint(detail: &str) -> Option<String> {
    let body = detail.strip_prefix("Failed to deserialize the JSON body into the target type: ")?;
    let path = body
        .split_once(": ")
        .map(|(p, _)| p)
        .filter(|p| *p != "." && is_safe_path(p));
    let missing = body.find("missing field `").and_then(|i| {
        let rest = &body[i + "missing field `".len()..];
        rest.split_once('`')
            .map(|(n, _)| n)
            .filter(|n| is_safe_path(n))
    });
    match (path, missing) {
        (Some(p), Some(m)) => Some(format!("missing required field `{p}.{m}`")),
        (None, Some(m)) => Some(format!("missing required field `{m}`")),
        (Some(p), None) => Some(format!("invalid value for field `{p}`")),
        (None, None) => None,
    }
}

/// Body for a request whose method the route does not accept (installed with
/// `Router::method_not_allowed_fallback`). axum still adds the `Allow` header
/// (checked live: `Allow: POST`); this only supplies the body.
pub(crate) async fn method_not_allowed() -> Response {
    openai_error(
        StatusCode::METHOD_NOT_ALLOWED,
        "method not allowed for this route",
        "invalid_request_error",
        Some("method_not_allowed"),
    )
}

// ── MultipartBody extractor ───────────────────────────────────────────────────

/// `axum::extract::Multipart`, but a request that is not valid
/// `multipart/form-data` (wrong `Content-Type`, missing boundary) is rejected
/// with the same OpenAI-style JSON error every other route gives, instead of
/// axum's plain-text "Invalid `boundary` for `multipart/form-data` request"
/// (Windows pass 2, D11). Use it in place of `Multipart` in handler signatures.
pub(crate) struct MultipartBody(pub(crate) axum::extract::Multipart);

impl<S> axum::extract::FromRequest<S> for MultipartBody
where
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match axum::extract::Multipart::from_request(req, state).await {
            Ok(m) => Ok(Self(m)),
            Err(rejection) if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE => {
                Err(request_too_large())
            }
            Err(rejection) => {
                tracing::debug!(detail = %rejection.body_text(), "rejected non-multipart request");
                Err(openai_error(
                    StatusCode::BAD_REQUEST,
                    "expected a multipart/form-data body (Content-Type: multipart/form-data; boundary=...)",
                    "invalid_request_error",
                    Some("invalid_multipart"),
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
        // 0.9.1 S1: an SSE frame cannot carry `Retry-After`; the text says what to do.
        InferenceErrorKind::GenerationStalled => {
            sse_error_frame(GENERATION_STALLED_MESSAGE, "generation_stalled")
        }
        InferenceErrorKind::PoolExhausted(None) => {
            sse_error_frame(POOL_EXHAUSTED_MESSAGE, "pool_exhausted")
        }
        InferenceErrorKind::PoolExhausted(Some(generated)) => {
            sse_error_frame(&output_growth_message(generated), "pool_exhausted")
        }
        InferenceErrorKind::VlmKvWedge => sse_error_frame(
            "model temporarily unavailable — recovering from a KV-cache capacity \
             issue; retry shortly",
            "kv_capacity_wedge",
        ),
        // G1: an SSE frame cannot carry `Retry-After`; the text says it.
        InferenceErrorKind::GpuPoisoned => {
            sse_error_frame(gpu_poisoned_reply_now().0, "model_unavailable")
        }
        InferenceErrorKind::ModelUnloading => sse_error_frame(
            "the model was unloaded while this request was running — retry shortly",
            "model_unloading",
        ),
        InferenceErrorKind::DeviceMemory => {
            sse_error_frame(DEVICE_MEMORY_MESSAGE, "insufficient_memory")
        }
        InferenceErrorKind::TemplateRejected(reason) => {
            sse_error_frame(&template_rejected_message(&reason), "template_rejected")
        }
        InferenceErrorKind::Opaque => sse_error_frame("inference failed", "mid_stream_error"),
    }
}

/// One SSE `data:` frame carrying `value` as JSON (K-02d). Serialising the server's own frame
/// structs cannot realistically fail, but if it ever does the stream gets a typed error frame
/// (`frame_serialize_failed`) and an error log, not an empty `data:` line a client fails to
/// parse without a word in the log.
pub(crate) fn sse_json_event<T: Serialize>(value: &T) -> axum::response::sse::Event {
    match serde_json::to_string(value) {
        Ok(json) => axum::response::sse::Event::default().data(json),
        Err(e) => {
            tracing::error!(error = %e, "an SSE frame could not be serialised; sent an error frame");
            sse_error_frame(
                "internal error: a stream frame could not be serialised",
                "frame_serialize_failed",
            )
        }
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
/// `Retry-After` for transient not-ready states (loading / evicting), seconds.
pub(crate) const RETRY_AFTER_TRANSIENT: HeaderValue = HeaderValue::from_static("5");

/// `Retry-After` while a supervised worker restarts after a GPU poisoning (G1, step 7): the first
/// respawn's backoff (5 s) + worker start + a typical model reload (a fixed 30 s).
pub(crate) const RETRY_AFTER_WORKER_RESTART: HeaderValue = HeaderValue::from_static("30");

/// What a client is told about a poisoned GPU context (G1). `restarting` = this worker is
/// supervised and on its way to exit 75 (`startup::gpu_poison_exit_requested`): the supervisor
/// brings a fresh one up, so a retry shortly WILL help. Otherwise only a manual restart does.
fn gpu_poisoned_reply(restarting: bool) -> (&'static str, Option<HeaderValue>) {
    if restarting {
        (
            "model unavailable — the GPU context failed (out of resources); the worker is \
             restarting — retry shortly",
            Some(RETRY_AFTER_WORKER_RESTART),
        )
    } else {
        (
            "model unavailable — the GPU context failed (out of resources) and the server must be \
             restarted; retrying will not help",
            None,
        )
    }
}

/// [`gpu_poisoned_reply`] for the current process state.
pub(crate) fn gpu_poisoned_reply_now() -> (&'static str, Option<HeaderValue>) {
    gpu_poisoned_reply(crate::startup::gpu_poison_exit_requested())
}

/// A 503 for a poisoned GPU context (G1): `code` as before, text and `Retry-After` from
/// [`gpu_poisoned_reply_now`].
fn gpu_poisoned_response(code: &'static str) -> Response {
    let (msg, retry) = gpu_poisoned_reply_now();
    let mut resp = openai_error(
        StatusCode::SERVICE_UNAVAILABLE,
        msg,
        "server_error",
        Some(code),
    );
    if let Some(v) = retry {
        resp.headers_mut().insert(header::RETRY_AFTER, v);
    }
    resp
}

/// A load that did not return (step 6b): `504 load_timeout` for the load that hit its deadline,
/// `503 device_stalled` + `Retry-After` for a load refused because another load on its device is
/// stalled. Shared by the inference routes and the admin load routes.
pub(crate) fn load_stall_response(e: &ModelError) -> Response {
    if matches!(e, ModelError::LoadTimeout { .. }) {
        return openai_error(
            StatusCode::GATEWAY_TIMEOUT,
            format!("{e}"),
            "server_error",
            Some("load_timeout"),
        );
    }
    let mut resp = openai_error(
        StatusCode::SERVICE_UNAVAILABLE,
        format!("{e}"),
        "server_error",
        Some("device_stalled"),
    );
    resp.headers_mut()
        .insert(header::RETRY_AFTER, RETRY_AFTER_TRANSIENT);
    resp
}

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
/// On-demand (Slice 3c, F1b): every inference route (chat, completions, embeddings, reranking,
/// System One, the media routes) sends a `NotLoaded` `on_demand` model through
/// `handlers::on_demand` first, which starts its load and waits for it up to
/// `on_demand_wait_secs`, answering `Loading` + `Retry-After` when it does not become ready in
/// time. A `NotLoaded` that reaches here is an eager model's (`/tokenize` and `/detokenize`, which
/// never load, answer `NotLoaded` with their own text before this).
pub(crate) fn model_error_response(e: ModelError) -> Response {
    match e {
        ModelError::NotFound(id) => openai_error(
            StatusCode::NOT_FOUND,
            unknown_model_message(&id),
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
        // Same envelope as `FilesIncomplete`. The path is left out: this body can reach an
        // inference client (T6.2), and the admin answer and the server log already name it.
        ModelError::TemplateUnreadable { model, .. } => openai_error(
            StatusCode::NOT_FOUND,
            format!("model '{model}': its chat_template override is unreadable (see the log)"),
            "invalid_request_error",
            Some("model_not_found"),
        ),
        // The not-ready states (and a stalled CPU/NPU model, 0.9.1 S1) share the 503
        // `model_unavailable` envelope; only the transient `Loading`/`Evicting` advertise
        // `Retry-After` — they resolve on their own, the others do not.
        ModelError::NotLoaded
        | ModelError::Loading
        | ModelError::Evicting
        | ModelError::GenerationStalled(_) => model_unavailable_response(&e),
        // G1: a supervised worker draining toward its restart says so; otherwise as before.
        ModelError::GpuPoisoned if crate::startup::gpu_poison_exit_requested() => {
            gpu_poisoned_response("gpu_poisoned")
        }
        ModelError::GpuPoisoned => openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!(
                "{e} — restart rustedvino to recover (a graceful stop via SIGTERM/systemctl drains in-flight requests and frees VRAM first)"
            ),
            "server_error",
            Some("gpu_poisoned"),
        ),
        // D-04 S2: no Retry-After — this process will not take the load; its successor might.
        ModelError::ShuttingDown => openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            e.to_string(),
            "server_error",
            Some("server_stopping"),
        ),
        ModelError::InsufficientMemory(msg) => openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            msg,
            "server_error",
            Some("insufficient_memory"),
        ),
        // D3: busy models are not evicted. `TargetBusy` (an admin eviction of a model with
        // work in flight) and `BlockedByBusy` (a load that would have to evict busy models)
        // both resolve when the work ends, so they advertise `Retry-After`. Neither is
        // produced until the policy flip (D3-S1.E2).
        ModelError::TargetBusy(_) | ModelError::BlockedByBusy(_) => {
            let mut resp = openai_error(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("model busy: {e}"),
                "server_error",
                Some("model_busy"),
            );
            resp.headers_mut()
                .insert(header::RETRY_AFTER, RETRY_AFTER_TRANSIENT);
            resp
        }
        ModelError::LoadTimeout { .. } | ModelError::DeviceStalled { .. } => {
            load_stall_response(&e)
        }
        // A forced eviction's engine is still finishing its request: its memory is not free
        // yet, so the load that needed it is refused; retryable.
        ModelError::EvictionIncomplete(_) => {
            let mut resp = openai_error(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("{e}"),
                "server_error",
                Some("eviction_incomplete"),
            );
            resp.headers_mut()
                .insert(header::RETRY_AFTER, RETRY_AFTER_TRANSIENT);
            resp
        }
        ModelError::WrongKind(msg) => openai_error(
            StatusCode::BAD_REQUEST,
            msg,
            "invalid_request_error",
            Some("unsupported_operation"),
        ),
        ModelError::ConfigFileInvalid(_) => config_file_invalid_response(&e),
    }
}

/// F1a: the 404 `model_not_found` text for an id that is not registered. It does not send the
/// client to `/v1/models` alone: that list shows what is usable now, not every registered model.
pub(crate) fn unknown_model_message(id: &str) -> String {
    format!("model '{id}' not found among the registered models; check the model id")
}

/// F1a: the one neutral sentence for a model that is registered but not loaded (the 503
/// `model_unavailable` body for `NotLoaded`, and the `GET /v1/models/{id}` 404 for a registered
/// model that is not listed). It names no `rv` command: the body reaches inference clients, who
/// may hold no admin rights. Since F1b an `on_demand` model never gets it from an inference route:
/// every route but `/tokenize` and `/detokenize` starts its load, and those two answer with their
/// own sentence (`handlers::tokenize::NOT_LOADED_MESSAGE`).
pub(crate) fn not_loaded_message(id: Option<&str>) -> String {
    let subject = id.map_or_else(|| "the model".to_owned(), |id| format!("model '{id}'"));
    format!(
        "{subject} is registered but not loaded; an administrator can load it, or set it to load \
         on demand"
    )
}

/// The 503 `model_unavailable` of [`model_error_response`]'s not-ready arm: `Retry-After` only for
/// the transient `Loading` / `Evicting`.
fn model_unavailable_response(e: &ModelError) -> Response {
    let message = if matches!(e, ModelError::NotLoaded) {
        format!("model unavailable: {}", not_loaded_message(None))
    } else {
        format!("model unavailable: {e}")
    };
    let mut resp = openai_error(
        StatusCode::SERVICE_UNAVAILABLE,
        message,
        "server_error",
        Some("model_unavailable"),
    );
    if matches!(e, ModelError::Loading | ModelError::Evicting) {
        resp.headers_mut()
            .insert(header::RETRY_AFTER, RETRY_AFTER_TRANSIENT);
    }
    resp
}

/// 409 `config_file_invalid`: `config.json` was already invalid, so a change that persists to it
/// was refused before anything was loaded or evicted. The code is what tells `rv` this is not a
/// bare 409 ("already registered").
pub(crate) fn config_file_invalid_response(e: &ModelError) -> Response {
    openai_error(
        StatusCode::CONFLICT,
        e.to_string(),
        "invalid_request_error",
        Some("config_file_invalid"),
    )
}

/// The answer for a request an engine refused because its model is being evicted (the engine's
/// `closing` flag, D3): exactly what a request gets when it finds the model `Evicting` — 503
/// `model_unavailable` with `Retry-After`.
pub(crate) fn model_closing_response() -> Response {
    model_error_response(ModelError::Evicting)
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
        // 0.9.1 S1: 504 + `Retry-After: 30` — the same wait a supervised worker
        // restart advertises; the request may be served by another model or after the restart.
        InferenceErrorKind::GenerationStalled => {
            let mut resp = openai_error(
                StatusCode::GATEWAY_TIMEOUT,
                GENERATION_STALLED_MESSAGE,
                "server_error",
                Some("generation_stalled"),
            );
            resp.headers_mut()
                .insert(header::RETRY_AFTER, RETRY_AFTER_WORKER_RESTART);
            resp
        }
        InferenceErrorKind::PoolExhausted(None) => openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            POOL_EXHAUSTED_MESSAGE,
            "server_error",
            Some("pool_exhausted"),
        ),
        InferenceErrorKind::PoolExhausted(Some(generated)) => openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            output_growth_message(generated),
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
        InferenceErrorKind::GpuPoisoned => gpu_poisoned_response("model_unavailable"),
        InferenceErrorKind::ModelUnloading => {
            let mut resp = openai_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "the model was unloaded while this request was running — retry shortly",
                "server_error",
                Some("model_unloading"),
            );
            resp.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                axum::http::HeaderValue::from_static("5"),
            );
            resp
        }
        InferenceErrorKind::DeviceMemory => {
            let mut resp = openai_error(
                StatusCode::SERVICE_UNAVAILABLE,
                DEVICE_MEMORY_MESSAGE,
                "server_error",
                Some("insufficient_memory"),
            );
            resp.headers_mut()
                .insert(axum::http::header::RETRY_AFTER, RETRY_AFTER_TRANSIENT);
            resp
        }
        InferenceErrorKind::TemplateRejected(reason) => openai_error(
            StatusCode::BAD_REQUEST,
            template_rejected_message(&reason),
            "invalid_request_error",
            Some("template_rejected"),
        ),
        InferenceErrorKind::Opaque => openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "inference failed",
            "server_error",
            None,
        ),
    }
}

/// What a client is told when the engine stall watchdog ended its generation (0.9.1 S1). Constant:
/// the deadline and the elapsed time are in the server's ERROR line, not in the body.
const GENERATION_STALLED_MESSAGE: &str = "the generation stalled: the model's engine made no \
     progress within the server's deadline and was marked dead; the model is refused until the \
     server restarts";

/// The `pool_exhausted` message for every case but the one below: the pool is momentarily full.
const POOL_EXHAUSTED_MESSAGE: &str = "the model is momentarily at capacity — retry shortly";

/// The `pool_exhausted` message when the only request on the model ran out of KV-cache room
/// after generating `generated` tokens (E2 amendment): retrying the same request
/// fails the same way, so the client is told what to change instead.
fn output_growth_message(generated: usize) -> String {
    format!(
        "the answer filled the model's KV cache after {generated} generated tokens; retrying the \
         same request will fail the same way — lower max_tokens or shorten the prompt"
    )
}

/// Prefix of the terminal error the CB engine sends to requests still live when its loop exits
/// (`cb_engine.rs`); maps to a retryable 503 rather than an opaque 500.
pub(crate) const ENGINE_SHUTDOWN_MARKER: &str = "engine shutting down";

/// Which class of inference failure `raw` represents, after applying its
/// side effect (poisoning the GPU context, spawning VLM-wedge recovery, or
/// nothing) — shared by [`inference_error_from_text`] and
/// [`sse_inference_error_event`] so the two response shapers (JSON body vs.
/// SSE frame) can never drift on which markers they recognize or what side
/// effect each one triggers.
enum InferenceErrorKind {
    /// The engine stall watchdog (`stall_watch`) ended the generation: its engine made no progress
    /// within `stream_first_token_timeout_secs` / `stream_token_gap_timeout_secs`. Its side effects
    /// (the engine marked dead, the GPU-poisoned flag for a GPU model, the ERROR line) already ran,
    /// once, in the watcher; none here.
    GenerationStalled,
    /// CB-scheduler KV-pool exhaustion (`GenerationStatus::IGNORED`). Usually
    /// scoped to the one request — other requests against this model are
    /// fine, and the pool just fluctuates under real multi-tenant load. But
    /// when [`pool_exhausted_active_requests`] reads `<= 1` (this was the
    /// only request against the model), nothing else could have crowded out
    /// the pool — that's not congestion, it's the static formula being
    /// wrong, so `spawn_kv_wedge_recovery` is triggered the same as
    /// [`InferenceErrorKind::VlmKvWedge`]
    /// (the project's internal engineering log) — unless the request had
    /// already generated tokens (`generated_tokens >= 1`): then the pool
    /// filled from the answer, and only the 503 is returned (E2,
    /// not a sizing miss), carrying that count (`Some`) so the message says that a
    /// retry of the same request fails again (E2 amendment); `None` for every
    /// other case (today's "momentarily at capacity" text).
    PoolExhausted(Option<usize>),
    /// VLM-path KV-admission wedge (`VLM_KV_WEDGE_MARKER`) — the pipeline
    /// itself never self-recovers; `spawn_kv_wedge_recovery` was triggered
    /// (evict + reload the model, learn a tighter ceiling).
    VlmKvWedge,
    /// Whole-`OpenCL`-context OOM (`is_gpu_poison_error`) — every model on
    /// this device is affected, not just this one.
    GpuPoisoned,
    /// The engine was stopped under a live request (eviction drain deadline,
    /// shutdown) — the request is fine, the model went away; retryable 503.
    ModelUnloading,
    /// `GenAI` refused to build the KV pool at request time for device memory
    /// ([`is_genai_memory_budget_error`], MR2-1): a vision model's pool is built at its first
    /// generate, which is a request when the load-time warm-up did not run (`kv_warmup: false`,
    /// a non-`GPU…` device string). The model is evicted and reloaded (the wedge recovery with
    /// no observed tokens, so no ceiling is learned); retryable 503 `insufficient_memory`.
    DeviceMemory,
    /// The model's chat template refused the messages (`raise_exception`, rendered by `GenAI` on
    /// the vision path; the text path already answers this with a 400 from its own renderer).
    /// Carries [`template_rejection_reason`]. A client error: 400 `template_rejected`.
    TemplateRejected(String),
    /// Anything else — an opaque 500, raw text logged server-side only.
    Opaque,
}

fn classify_inference_error(
    mm: Option<&Arc<ModelManager>>,
    model_id: &str,
    raw: &str,
) -> InferenceErrorKind {
    // First, and with no side effect: the watchdog's own text, never an engine's.
    if crate::stall_watch::is_generation_stalled_error(raw) {
        return InferenceErrorKind::GenerationStalled;
    }
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
        //
        // E2: a solo request that had already generated tokens filled the pool
        // with its answer (admission reserves the prompt only), so the prompt-side formula was
        // not wrong: same 503, but no evict+reload and no ratchet. An absent count (older
        // message shape) is unknown and keeps the recovery.
        let active_requests = pool_exhausted_active_requests(raw);
        let generated_tokens = pool_exhausted_generated_tokens(raw);
        let output_growth = if active_requests.is_some_and(|n| n <= 1)
            && let Some(generated_tokens) = generated_tokens.filter(|&g| g >= 1)
        {
            tracing::warn!(
                model_id,
                prompt_tokens = kv_wedge_observed_prompt_tokens(raw).unwrap_or(0),
                generated_tokens,
                "generation stopped — KV pool filled from output growth; no ratchet, no reload"
            );
            Some(generated_tokens)
        } else if active_requests.is_some_and(|n| n <= 1) {
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
            None
        } else {
            tracing::warn!(model_id, error = %raw, "generation ignored — KV pool exhausted, retryable");
            None
        };
        return InferenceErrorKind::PoolExhausted(output_growth);
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
    if is_genai_memory_budget_error(raw) {
        // Not a poison and not a ceiling miss: the pool never got built. Evict and reload
        // (first caller wins; `0` observed tokens skips the ratchet write) so the model does
        // not sit Ready and fail every request.
        if let Some(mm) = mm {
            mm.spawn_kv_wedge_recovery(model_id, 0);
        }
        tracing::warn!(
            model_id,
            error = %raw,
            "GenAI refused the KV cache at request time (device memory); reloading the model"
        );
        return InferenceErrorKind::DeviceMemory;
    }
    if let Some(reason) = template_rejection_reason(raw) {
        tracing::warn!(model_id, error = %raw, "the model's chat template rejected the request");
        return InferenceErrorKind::TemplateRejected(reason);
    }
    if raw.contains(ENGINE_SHUTDOWN_MARKER) {
        tracing::warn!(model_id, error = %raw, "request aborted: its engine was stopped (eviction or shutdown)");
        return InferenceErrorKind::ModelUnloading;
    }
    if crate::model_manager::is_kv_blob_mismatch_error(raw)
        && let Some(mm) = mm
    {
        mm.note_kv_blob_mismatch(model_id);
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

    /// G1 (step 7): a supervised worker on its way to a restart says "retry shortly" and sends
    /// `Retry-After: 30`; otherwise the old "retrying will not help" text, no header.
    #[test]
    fn gpu_poisoned_reply_is_retryable_only_while_restarting() {
        let (msg, retry) = gpu_poisoned_reply(true);
        assert!(msg.contains("retry shortly"), "{msg}");
        assert!(!msg.contains("will not help"), "{msg}");
        assert_eq!(retry, Some(HeaderValue::from_static("30")));
        let (msg, retry) = gpu_poisoned_reply(false);
        assert!(msg.contains("retrying will not help"), "{msg}");
        assert_eq!(retry, None);
    }

    /// Step 6b: a load past its deadline is a 504 `load_timeout`; a load refused because its
    /// device has a stalled load is a retryable 503 `device_stalled`.
    #[test]
    fn load_stall_errors_map_to_504_and_retryable_503() {
        let r = load_stall_response(&ModelError::LoadTimeout {
            model: "m".to_owned(),
            secs: 600,
        });
        assert_eq!(r.status(), StatusCode::GATEWAY_TIMEOUT);
        let r = load_stall_response(&ModelError::DeviceStalled {
            device: "GPU.1".to_owned(),
            model: "m".to_owned(),
            secs: 700,
        });
        assert_eq!(r.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(r.headers().contains_key(header::RETRY_AFTER));
    }

    /// C8: an unreadable `chat_template` override maps like `FilesIncomplete` (404
    /// `model_not_found`) and the client-facing body names the model but not the path (T6.2).
    #[tokio::test]
    async fn template_unreadable_is_404_model_not_found_without_the_path() {
        let resp = model_error_response(ModelError::TemplateUnreadable {
            model: "lfm2".to_owned(),
            path: "/srv/private/lfm2.jinja".into(),
            cause: "No such file or directory".to_owned(),
        });
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8_lossy(&bytes);
        assert!(
            body.contains("model_not_found") && body.contains("lfm2"),
            "{body}"
        );
        assert!(!body.contains("/srv/private"), "{body}");
    }

    /// D-04 S2: a load refused because the server is stopping is a 503 `server_stopping` with no
    /// `Retry-After` (this process will not take it).
    #[tokio::test]
    async fn shutting_down_is_503_server_stopping_without_retry_after() {
        let resp = model_error_response(ModelError::ShuttingDown);
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(!resp.headers().contains_key(header::RETRY_AFTER));
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "server_stopping");
    }

    #[test]
    fn field_hint_names_path_but_never_value() {
        let p = "Failed to deserialize the JSON body into the target type: ";
        assert_eq!(
            field_hint(&format!(
                "{p}max_tokens: invalid value: integer `-5`, expected u32 at line 1 column 30"
            ))
            .as_deref(),
            Some("invalid value for field `max_tokens`")
        );
        assert_eq!(
            field_hint(&format!(
                "{p}messages[0]: missing field `role` at line 1 column 20"
            ))
            .as_deref(),
            Some("missing required field `messages[0].role`")
        );
        assert_eq!(
            field_hint(&format!("{p}missing field `model` at line 1 column 2")).as_deref(),
            Some("missing required field `model`")
        );
        assert!(
            field_hint(&format!(
                "{p}temperature: invalid type: string \"hot\", expected f32"
            ))
            .unwrap()
            .contains("temperature")
        );
        assert!(
            !field_hint(&format!(
                "{p}temperature: invalid type: string \"hot\", expected f32"
            ))
            .unwrap()
            .contains("hot")
        );
        // Unsafe path characters or a non-serde rejection fall back to generic.
        assert_eq!(field_hint(&format!("{p}a b<script>: invalid type")), None);
        assert_eq!(
            field_hint("Failed to parse the request body as JSON: EOF"),
            None
        );
    }

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

    /// The -14 form (2026-11-15, ext-091): an inference failure whose text carries only
    /// `CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST` is the same scrubbed 503 `model_unavailable`
    /// as the other poison markers, not an opaque 500 (non-stream body and SSE frame). The
    /// `Retry-After` header depends on the process-wide poison-exit flag, not on the
    /// classification; `gpu_poisoned_reply_is_retryable_only_while_restarting` covers it.
    #[tokio::test]
    async fn inference_exec_status_minus_14_is_503_model_unavailable_scrubbed() {
        let raw = "ov_vlm_generate failed: Exception from src/inference/src/cpp/infer_request.cpp:224:\n\
             Exception from src/plugins/intel_gpu/src/runtime/ocl/ocl_memory.cpp:593:\n\
             [GPU] clWaitForEvents, error code: -14 CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST\n\n";
        let resp = inference_error_from_text(None, "qwen3.5-9B-int4-ov", raw);
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "model_unavailable");
        let body = String::from_utf8_lossy(&bytes);
        assert!(
            !body.contains("CL_EXEC_STATUS") && !body.contains("ocl_memory"),
            "{body}"
        );
        let frame = format!("{:?}", sse_inference_error_event(None, "m", raw));
        assert!(frame.contains("model_unavailable"), "{frame}");
        assert!(!frame.contains("CL_EXEC_STATUS"), "{frame}");
    }

    /// E2 + its amendment: the `pool_exhausted` JSON body and SSE frame keep
    /// today's bytes for every case but one — a solo request with `generated_tokens >= 1`, whose
    /// message says a retry fails the same way (same 503, type and code). The raw field itself
    /// never reaches an HTTP client. `mm` is `None`, so no recovery runs here.
    #[tokio::test]
    async fn pool_exhausted_http_text_is_unchanged_by_the_generated_count() {
        let marker = crate::model_manager::POOL_EXHAUSTED_MARKER;
        let raw =
            |fields: &str| format!("generation ignored by the CB scheduler — {marker} ({fields})");
        let body = |raw: String| async move {
            let resp = inference_error_from_text(None, "m", &raw);
            assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
            axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap()
        };
        let sse = |raw: &str| format!("{:?}", sse_inference_error_event(None, "m", raw));
        let today = r#"{"error":{"message":"the model is momentarily at capacity — retry shortly","type":"server_error","param":null,"code":"pool_exhausted"}}"#;
        let today_sse = sse(&raw("observed_prompt_tokens=900 active_requests=3"));
        for fields in [
            "observed_prompt_tokens=900 active_requests=3",
            "observed_prompt_tokens=900 active_requests=3 generated_tokens=41",
            "observed_prompt_tokens=900 active_requests=1 generated_tokens=0",
            "observed_prompt_tokens=900 active_requests=1",
        ] {
            let b = body(raw(fields)).await;
            assert_eq!(std::str::from_utf8(&b).unwrap(), today, "{fields}");
            assert_eq!(sse(&raw(fields)), today_sse, "{fields}");
        }

        // Solo, after 41 generated tokens: the amended message, same status/type/code.
        let grown = raw("observed_prompt_tokens=900 active_requests=1 generated_tokens=41");
        let b = body(grown.clone()).await;
        assert_eq!(
            std::str::from_utf8(&b).unwrap(),
            r#"{"error":{"message":"the answer filled the model's KV cache after 41 generated tokens; retrying the same request will fail the same way — lower max_tokens or shorten the prompt","type":"server_error","param":null,"code":"pool_exhausted"}}"#
        );
        let frame = sse(&grown);
        assert_ne!(frame, today_sse);
        assert!(frame.contains("after 41 generated tokens"), "{frame}");
        assert!(frame.contains("pool_exhausted"), "{frame}");
        assert!(!frame.contains("generated_tokens="), "{frame}");
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

    /// SF-5 D-03: a chat-template `raise_exception` on the vision path is a 400 naming the
    /// template's reason, without the template excerpt; other text is not matched.
    #[tokio::test]
    async fn a_template_rejection_is_a_400_with_the_templates_reason() {
        let raw = anyhow::anyhow!(
            "ov_vlm generate failed: Minja failed to apply chat template. Possible solutions \
             are\n* Provide a simplified chat template\nMinja's error: Unexpected message role. \
             at row 144, column 28:\n    {{%- else %}}\n        {{{{- raise_exception('x') }}}}"
        );
        let resp = inference_error_response(None, "qwen3.5", &raw);
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "template_rejected");
        assert_eq!(
            json["error"]["message"],
            "the model's chat template rejected the messages: Unexpected message role."
        );
        assert_eq!(template_rejection_reason("plain failure"), None);
        assert_eq!(
            template_rejection_reason("Minja's error:  at row 1").as_deref(),
            Some("(no reason given)")
        );
    }

    /// K-02d: a frame that cannot be serialised becomes a typed error frame, not an empty
    /// `data:` line; a normal value is sent as its JSON.
    #[test]
    fn sse_json_event_sends_an_error_frame_when_serialisation_fails() {
        struct Fails;
        impl Serialize for Fails {
            fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
                Err(serde::ser::Error::custom("boom"))
            }
        }
        let frame = |e: axum::response::sse::Event| format!("{e:?}");
        let bad = frame(sse_json_event(&Fails));
        assert!(bad.contains("frame_serialize_failed"), "{bad}");
        let good = frame(sse_json_event(&serde_json::json!({"a": 1})));
        assert!(
            good.contains(r#"{\"a\":1}"#) || good.contains(r#"{"a":1}"#),
            "{good}"
        );
    }

    /// MR2-1: `GenAI` refusing the KV pool at request time (a vision model whose warm-up did not
    /// run) is a retryable 503 `insufficient_memory` with `Retry-After`, the C++ text and CI
    /// path scrubbed — not an opaque 500 on every request.
    #[tokio::test]
    async fn genai_memory_budget_at_request_time_is_retryable_503() {
        let raw = anyhow::anyhow!(
            "ov_vlm generate failed: Check 'budget_in_bytes <= total_available_memory' failed \
             at /home/jenkins/agent/src/cache_orchestrator.hpp:759"
        );
        let resp = inference_error_response(None, "qwen3-vl", &raw);
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(resp.headers().get("retry-after").unwrap(), "5");
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "insufficient_memory");
        let msg = json["error"]["message"].as_str().unwrap();
        for leak in ["budget_in_bytes", "jenkins", "cache_orchestrator"] {
            assert!(!msg.contains(leak), "leaks {leak}: {msg}");
        }
    }

    /// An engine stopped under a live request (eviction drain deadline) is a retryable 503
    /// with `Retry-After`, not an opaque 500.
    #[tokio::test]
    async fn engine_shutdown_is_retryable_503() {
        let raw = anyhow::anyhow!("engine shutting down — request aborted before completion");
        let resp = inference_error_response(None, "bielik", &raw);
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(resp.headers().get("retry-after").unwrap(), "5");
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "model_unloading");
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

    /// D3-S1.B2: the busy-eviction refusals are 503 `model_busy` with a `Retry-After`, and name
    /// the cause (the models whose eviction was refused).
    #[tokio::test]
    async fn busy_refusals_are_503_model_busy_with_retry_after() {
        for e in [
            ModelError::TargetBusy(2),
            ModelError::BlockedByBusy(vec!["qwen".to_owned(), "bielik".to_owned()]),
        ] {
            let text = e.to_string();
            let resp = model_error_response(e);
            assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(resp.headers().get(header::RETRY_AFTER).unwrap(), "5");
            let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(json["error"]["code"], "model_busy");
            assert!(json["error"]["message"].as_str().unwrap().contains(&text));
        }
        let many =
            ModelError::BlockedByBusy(vec!["qwen".to_owned(), "bielik".to_owned()]).to_string();
        assert!(
            many.contains("qwen, bielik") && many.contains("have requests running"),
            "{many}"
        );
        let one = ModelError::BlockedByBusy(vec!["qwen".to_owned()]).to_string();
        assert!(one.contains("has requests running"), "{one}");
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

    async fn error_json(resp: Response) -> (StatusCode, serde_json::Value) {
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    /// F1a: `NotLoaded` keeps its 503 `model_unavailable` (no `Retry-After`), and its body is
    /// one neutral sentence: registered, not loaded, an administrator can load it or set it to
    /// load on demand. No `rv` command reaches an inference client.
    #[tokio::test]
    async fn not_loaded_body_is_the_neutral_sentence() {
        let (status, json) = error_json(model_error_response(ModelError::NotLoaded)).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(json["error"]["code"], "model_unavailable");
        assert_eq!(json["error"]["type"], "server_error");
        assert_eq!(
            json["error"]["message"],
            "model unavailable: the model is registered but not loaded; an administrator can \
             load it, or set it to load on demand"
        );
        let msg = json["error"]["message"].as_str().unwrap();
        assert!(
            !msg.contains("rv "),
            "no rv command in a client body: {msg}"
        );
        // The other not-ready states keep their old text.
        let (_, loading) = error_json(model_error_response(ModelError::Loading)).await;
        assert_eq!(
            loading["error"]["message"],
            format!("model unavailable: {}", ModelError::Loading)
        );
    }

    /// F1a: an unknown id keeps 404 `model_not_found`; the hint no longer points only at
    /// `/v1/models` (which lists what is usable now, not every registered model).
    #[tokio::test]
    async fn unknown_id_body_says_not_registered() {
        let (status, json) =
            error_json(model_error_response(ModelError::NotFound("nope".into()))).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(json["error"]["code"], "model_not_found");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(
            json["error"]["message"],
            "model 'nope' not found among the registered models; check the model id"
        );
        assert_eq!(
            not_loaded_message(Some("m")),
            "model 'm' is registered but not loaded; an administrator can load it, or set it to \
             load on demand"
        );
    }
}
