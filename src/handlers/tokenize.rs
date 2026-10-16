// ============================================================
// src/handlers/tokenize.rs — POST /tokenize and POST /detokenize
// ============================================================
// Non-OpenAI helper routes that expose the model's tokenizer directly:
//
//   POST /tokenize    {model, prompt}      → {tokens, count}
//   POST /detokenize  {model, tokens}      → {text}
//
// Why they exist (PLAN_cap-for-gap §2.3/2.4): agent frameworks and RL
// pipelines need token counts *before* sending a request (to stay under a
// context window) and need ids→text without a generation round-trip.
//
// THREADING — the keystone deferral, now paid off:
//   The tokenizer is owned by the continuous-batching engine thread and is
//   NOT thread-safe (its `InferRequest` is single-threaded). So these handlers
//   cannot call it directly. They route through the engine command channel via
//   `EngineHandle::tokenize` / `detokenize`, which send a `oneshot`-reply
//   command the engine thread services between model steps. Tokenization does
//   not consume an inference slot, so it is never rejected with 429.
// ============================================================

use axum::{Json, extract::State, http::StatusCode, response::IntoResponse, response::Response};
use serde::{Deserialize, Serialize};

use super::error::{JsonBody, model_error_response, openai_error, openai_error_with_param};
use crate::app_state::AppState;
use crate::model_manager::engine::TokenizerHandle;

/// Text the bridge puts in the error for an id outside the model's vocabulary
/// (`ov_cb_decode` / `ov_vlm_decode`); the handler turns it into a 400.
pub(crate) const INVALID_TOKEN_ID_MARKER: &str = "invalid token id:";

// ---- Request / response shapes --------------------------------------

/// `POST /tokenize` body. `prompt` is encoded verbatim — no chat template is
/// applied (callers wanting templated counts build the prompt themselves).
#[derive(Debug, Deserialize)]
pub struct TokenizeRequest {
    /// Which loaded model's tokenizer to use.
    pub model: String,
    /// The text to encode.
    pub prompt: String,
}

/// `POST /tokenize` response: the token ids and their count.
#[derive(Serialize)]
struct TokenizeResponse {
    tokens: Vec<i64>,
    count: usize,
}

/// `POST /detokenize` body.
#[derive(Debug, Deserialize)]
pub struct DetokenizeRequest {
    /// Which loaded model's detokenizer to use.
    pub model: String,
    /// The token ids to decode back to text.
    pub tokens: Vec<i64>,
}

/// `POST /detokenize` response: the decoded text.
#[derive(Serialize)]
struct DetokenizeResponse {
    text: String,
}

// ---- Shared model-resolution -----------------------------------------

/// Resolve the engine handle for `model`, or an OpenAI-shaped error response.
///
/// Mirrors the chat handler's model-routing: `NotFound` → 404, any
/// not-ready state → 503, and mock/test mode (no model manager) → 503 since
/// there is no tokenizer to call.
///
/// The error half is boxed: a full `Response` is large (>128 bytes), and
/// `clippy::result_large_err` flags returning it by value in a `Result`. The
/// happy path (an `EngineHandle`) stays unboxed and cheap.
fn resolve_handle(state: &AppState, model: &str) -> Result<TokenizerHandle, Box<Response>> {
    let Some(mm) = state.model_manager.as_ref() else {
        return Err(Box::new(openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "tokenizer unavailable: no model loaded (server running without a model manager)",
            "server_error",
            Some("model_unavailable"),
        )));
    };
    mm.get_tokenizer(model)
        .map_err(|e| Box::new(model_error_response(e)))
}

// ---- Handlers --------------------------------------------------------

/// `POST /tokenize` — encode `prompt` to token ids using the model's tokenizer.
pub async fn tokenize(
    State(state): State<AppState>,
    JsonBody(req): JsonBody<TokenizeRequest>,
) -> Response {
    let handle = match resolve_handle(&state, &req.model) {
        Ok(h) => h,
        Err(resp) => return *resp,
    };

    match handle.tokenize(req.prompt).await {
        Ok(tokens) => {
            let count = tokens.len();
            Json(TokenizeResponse { tokens, count }).into_response()
        }
        Err(e) => openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("tokenization failed: {e}"),
            "server_error",
            None,
        ),
    }
}

/// `POST /detokenize` — decode token ids back to text via the model's detokenizer.
pub async fn detokenize(
    State(state): State<AppState>,
    JsonBody(req): JsonBody<DetokenizeRequest>,
) -> Response {
    // A negative id is invalid for every model. The bridge range-checks against the vocabulary, but
    // a detokenizer without a vocabulary table (SentencePiece) cannot, so catch this one here.
    if let Some(i) = req.tokens.iter().position(|&t| t < 0) {
        return openai_error_with_param(
            StatusCode::BAD_REQUEST,
            format!(
                "{INVALID_TOKEN_ID_MARKER} tokens[{i}] = {} is negative",
                req.tokens[i]
            ),
            "invalid_request_error",
            Some("invalid_value"),
            Some("tokens"),
        );
    }
    let handle = match resolve_handle(&state, &req.model) {
        Ok(h) => h,
        Err(resp) => return *resp,
    };

    match handle.detokenize(req.tokens).await {
        Ok(text) => Json(DetokenizeResponse { text }).into_response(),
        // The bridge refuses an id outside the vocabulary and says so; that is the caller's
        // mistake, not a server failure.
        Err(e) if e.to_string().contains(INVALID_TOKEN_ID_MARKER) => {
            // Drop the internal "ov_xx_decode failed: " prefix; the API user needs only the rest.
            let full = e.to_string();
            let message = full
                .find(INVALID_TOKEN_ID_MARKER)
                .map_or(full.as_str(), |i| &full[i..]);
            openai_error_with_param(
                StatusCode::BAD_REQUEST,
                message,
                "invalid_request_error",
                Some("invalid_value"),
                Some("tokens"),
            )
        }
        Err(e) => openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("detokenization failed: {e}"),
            "server_error",
            None,
        ),
    }
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    /// In mock mode (no model manager) `/tokenize` returns a 503 `OpenAI`
    /// envelope rather than panicking — there is no tokenizer to call.
    #[tokio::test]
    async fn tokenize_without_model_manager_is_503() {
        let state = AppState::mock();
        let resp = tokenize(
            State(state),
            JsonBody(TokenizeRequest {
                model: "qwen3-8b-int4-ov".to_owned(),
                prompt: "Hello world".to_owned(),
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["type"], "server_error");
        assert_eq!(json["error"]["code"], "model_unavailable");
    }

    /// A negative token id is invalid for every model, so it is a 400 before any model lookup (a
    /// model with no vocabulary table cannot range-check ids in the bridge).
    #[tokio::test]
    async fn detokenize_negative_id_is_400_before_the_model_lookup() {
        let resp = detokenize(
            State(AppState::mock()),
            JsonBody(DetokenizeRequest {
                model: "anything".to_owned(),
                tokens: vec![5, -1],
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "invalid_value");
        assert_eq!(json["error"]["param"], "tokens");
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("tokens[1]")
        );
    }

    /// Same for `/detokenize` in mock mode.
    #[tokio::test]
    async fn detokenize_without_model_manager_is_503() {
        let state = AppState::mock();
        let resp = detokenize(
            State(state),
            JsonBody(DetokenizeRequest {
                model: "qwen3-8b-int4-ov".to_owned(),
                tokens: vec![9906, 1879],
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    /// `TokenizeResponse` serialises to the documented `{tokens, count}` shape.
    #[test]
    fn tokenize_response_shape() {
        let json = serde_json::to_value(TokenizeResponse {
            tokens: vec![9906, 1879],
            count: 2,
        })
        .unwrap();
        assert_eq!(json["tokens"][0], 9906);
        assert_eq!(json["tokens"][1], 1879);
        assert_eq!(json["count"], 2);
    }

    /// `DetokenizeResponse` serialises to the documented `{text}` shape.
    #[test]
    fn detokenize_response_shape() {
        let json = serde_json::to_value(DetokenizeResponse {
            text: "Hello world".to_owned(),
        })
        .unwrap();
        assert_eq!(json["text"], "Hello world");
    }
}
