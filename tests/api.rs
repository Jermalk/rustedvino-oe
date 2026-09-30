// ============================================================
// tests/api.rs — integration tests for the HTTP API
// ============================================================
// All tests use tower::ServiceExt::oneshot() — fires one request
// through the Router without binding a TCP port.
// Fast, deterministic, fully parallel.
// ============================================================

#![allow(clippy::unwrap_used)]

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt as _;
use rustedvino::app_state::AppState;
use tower::ServiceExt as _;

/// Helper: drain an async body into a `serde_json::Value`.
async fn body_json(body: Body) -> serde_json::Value {
    let bytes = body.collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

/// Helper: drain an async body into a String.
async fn body_string(body: Body) -> String {
    let bytes = body.collect().await.unwrap().to_bytes();
    String::from_utf8_lossy(&bytes).into_owned()
}

// ------------------------------------------------------------------
// GET /health
// ------------------------------------------------------------------

#[tokio::test]
async fn health_returns_ok() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let json = body_json(response.into_body()).await;
    assert_eq!(json["status"], "ok");
    assert_eq!(json["version"], env!("CARGO_PKG_VERSION"));
}

// ------------------------------------------------------------------
// GET /v1/models
// ------------------------------------------------------------------

/// Phase 0 empty list — shape test survives when Phase 2 populates data.
#[tokio::test]
async fn models_list_returns_empty_list() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let json = body_json(response.into_body()).await;
    assert_eq!(json["object"], "list");
    assert!(json["data"].is_array());
    assert_eq!(json["data"].as_array().unwrap().len(), 0);
}

// ------------------------------------------------------------------
// Unknown route
// ------------------------------------------------------------------

#[tokio::test]
async fn unknown_route_returns_404() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/this-route-does-not-exist")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

// ------------------------------------------------------------------
// POST /v1/chat/completions
// ------------------------------------------------------------------

/// Streaming response: status 200, Content-Type text/event-stream.
#[tokio::test]
async fn chat_stream_returns_sse_content_type() {
    let app = rustedvino::create_router(AppState::mock());

    let body = r#"{"model":"mock","messages":[{"role":"user","content":"hi"}],"stream":true}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let content_type = response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(
        content_type.starts_with("text/event-stream"),
        "expected text/event-stream, got: {content_type}"
    );
}

/// Streaming response body contains [DONE] sentinel and at least one token.
#[tokio::test]
async fn chat_stream_body_contains_done_and_tokens() {
    let app = rustedvino::create_router(AppState::mock());

    let body = r#"{"model":"mock","messages":[{"role":"user","content":"hi"}],"stream":true}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    let text = body_string(response.into_body()).await;

    // Every SSE line starts with "data: "
    assert!(
        text.contains("data: "),
        "SSE body must contain data: lines\n---\n{text}"
    );

    // Must end with the OpenAI sentinel
    assert!(
        text.contains("[DONE]"),
        "SSE body must contain [DONE]\n---\n{text}"
    );

    // Must include at least one content token (our mock sends "Hello")
    assert!(
        text.contains("Hello"),
        "SSE body must contain mock token 'Hello'\n---\n{text}"
    );

    // First data chunk must carry role: assistant
    assert!(
        text.contains("assistant"),
        "first SSE chunk must include role:assistant\n---\n{text}"
    );
}

/// Non-streaming request returns a buffered `chat.completion` JSON (Phase 3).
#[tokio::test]
async fn chat_nonstreaming_returns_completion() {
    let app = rustedvino::create_router(AppState::mock());

    let body = r#"{"model":"mock","messages":[{"role":"user","content":"hi"}],"stream":false}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response.into_body()).await;

    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["model"], "mock");
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    // The mock generator emits the full RustedVINO greeting as one message.
    assert!(
        json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .contains("RustedVINO"),
        "buffered content must contain the mock greeting\n---\n{json}"
    );
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    // 14 non-empty mock tokens → completion_tokens = 14; prompt not yet tracked.
    assert_eq!(json["usage"]["completion_tokens"], 14);
    assert_eq!(json["usage"]["prompt_tokens"], 0);
    assert_eq!(json["usage"]["total_tokens"], 14);
}

/// Missing stream field defaults to non-streaming → `chat.completion` JSON.
#[tokio::test]
async fn chat_missing_stream_field_returns_completion() {
    let app = rustedvino::create_router(AppState::mock());

    let body = r#"{"model":"mock","messages":[{"role":"user","content":"hi"}]}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
}

// ------------------------------------------------------------------
// G2 tool_choice (chat path)
// ------------------------------------------------------------------

/// `tool_choice:"none"` suppresses the buffered tool branch even when `tools`
/// are supplied → a plain `chat.completion` (no `tool_calls`).
#[tokio::test]
async fn chat_tool_choice_none_with_tools_is_plain_completion() {
    let app = rustedvino::create_router(AppState::mock());

    let body = r#"{"model":"mock","messages":[{"role":"user","content":"hi"}],
        "tools":[{"type":"function","function":{"name":"get_weather"}}],
        "tool_choice":"none","stream":false}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["object"], "chat.completion");
    // Plain answer: content present (mock greeting), no tool_calls field.
    assert!(
        json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .contains("RustedVINO"),
        "none must return plain content\n---\n{json}"
    );
    assert!(
        json["choices"][0]["message"]["tool_calls"].is_null(),
        "none must not produce tool_calls\n---\n{json}"
    );
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
}

/// An unknown `tool_choice` string is rejected up front with a 400 envelope.
#[tokio::test]
async fn chat_invalid_tool_choice_is_400() {
    let app = rustedvino::create_router(AppState::mock());

    let body = r#"{"model":"mock","messages":[{"role":"user","content":"hi"}],
        "tool_choice":"banana"}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["error"]["code"], "invalid_tool_choice");
}

/// A non-text `response_format` combined with a forcing `tool_choice` is a 400
/// (mutually-exclusive structured-output modes — not a silent drop).
#[tokio::test]
async fn chat_response_format_plus_required_tool_choice_is_400() {
    let app = rustedvino::create_router(AppState::mock());

    let body = r#"{"model":"mock","messages":[{"role":"user","content":"hi"}],
        "tools":[{"type":"function","function":{"name":"get_weather"}}],
        "tool_choice":"required","response_format":{"type":"json_object"}}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["error"]["code"], "incompatible_parameters");
}

// ------------------------------------------------------------------
// Unsupported-param 400 policy (explicit reject, not silent-drop)
// ------------------------------------------------------------------

/// `n > 1` on chat is rejected with an `unsupported_parameter` envelope.
#[tokio::test]
async fn chat_n_greater_than_one_is_unsupported_param_400() {
    let app = rustedvino::create_router(AppState::mock());
    let body = r#"{"model":"mock","messages":[{"role":"user","content":"hi"}],"n":2}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["error"]["code"], "unsupported_parameter");
}

/// #47 (T7.6): an explicit `max_tokens: 0` is a 400, not "no limit" — distinct
/// from omitting the field (which means "server default").
#[tokio::test]
async fn chat_max_tokens_zero_is_400() {
    let app = rustedvino::create_router(AppState::mock());
    let body = r#"{"model":"mock","messages":[{"role":"user","content":"hi"}],"max_tokens":0}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["error"]["code"], "invalid_max_tokens");
}

/// #49 (T7.6): two completions returned in the same wall-clock second carry
/// distinct response `id`s — the collision the whole-second `created` id caused.
#[tokio::test]
async fn chat_response_ids_are_unique_across_requests() {
    let body = r#"{"model":"mock","messages":[{"role":"user","content":"hi"}],"stream":false}"#;
    let send = || async {
        let app = rustedvino::create_router(AppState::mock());
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        body_json(response.into_body()).await["id"]
            .as_str()
            .unwrap()
            .to_owned()
    };

    let id1 = send().await;
    let id2 = send().await;
    assert_ne!(
        id1, id2,
        "concurrent same-second ids must differ: {id1} {id2}"
    );
}

/// `logprobs: true` on chat is rejected (the bridge exposes no logits).
#[tokio::test]
async fn chat_logprobs_is_unsupported_param_400() {
    let app = rustedvino::create_router(AppState::mock());
    let body = r#"{"model":"mock","messages":[{"role":"user","content":"hi"}],"logprobs":true}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["error"]["code"], "unsupported_parameter");
}

/// A clean chat request (`n: 1`, empty `logit_bias`) is accepted, proving the
/// policy rejects only genuinely-unhonored params.
#[tokio::test]
async fn chat_accepted_params_still_complete() {
    let app = rustedvino::create_router(AppState::mock());
    let body = r#"{"model":"mock","messages":[{"role":"user","content":"hi"}],"n":1,"logit_bias":{},"stream":false}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["object"], "chat.completion");
}

/// `logprobs` on legacy completions is rejected with the same envelope.
#[tokio::test]
async fn completions_logprobs_is_unsupported_param_400() {
    let app = rustedvino::create_router(AppState::mock());
    let body = r#"{"model":"mock","prompt":"hi","logprobs":2}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["error"]["code"], "unsupported_parameter");
}

// ------------------------------------------------------------------
// POST /v1/completions (legacy text completions, Phase 3.7)
// ------------------------------------------------------------------

/// Non-streaming legacy completion via the mock path: `text_completion` object,
/// `choices[].text` carries the mock generation, null `logprobs`, and `usage`.
#[tokio::test]
async fn completions_nonstreaming_returns_text_completion() {
    let app = rustedvino::create_router(AppState::mock());

    let body = r#"{"model":"mock","prompt":"The capital of Poland is","max_tokens":16}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response.into_body()).await;

    assert_eq!(json["object"], "text_completion");
    assert_eq!(json["model"], "mock");
    assert_eq!(json["choices"][0]["index"], 0);
    assert!(
        json["choices"][0]["text"]
            .as_str()
            .unwrap_or_default()
            .contains("RustedVINO"),
        "completion text must contain the mock generation\n---\n{json}"
    );
    assert!(
        json["choices"][0]["logprobs"].is_null(),
        "logprobs must be present and null"
    );
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["usage"]["completion_tokens"], 14);
    assert_eq!(json["usage"]["total_tokens"], 14);
}

/// An array `prompt` produces one indexed choice per element (mock path).
#[tokio::test]
async fn completions_array_prompt_yields_one_choice_each() {
    let app = rustedvino::create_router(AppState::mock());

    let body = r#"{"model":"mock","prompt":["alpha","beta"]}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["choices"][0]["index"], 0);
    assert_eq!(json["choices"][1]["index"], 1);
    // Two prompts → 2 × 14 mock tokens.
    assert_eq!(json["usage"]["completion_tokens"], 28);
}

/// Streaming legacy completion emits `text_completion` chunks + `[DONE]`, with
/// no leading role frame.
#[tokio::test]
async fn completions_streaming_emits_text_completion_chunks() {
    let app = rustedvino::create_router(AppState::mock());

    let body = r#"{"model":"mock","prompt":"hi","stream":true}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let text = body_string(response.into_body()).await;
    assert!(
        text.contains(r#""object":"text_completion""#),
        "must stream text_completion chunks: {text}"
    );
    assert!(text.contains("[DONE]"), "must end with [DONE]: {text}");
    assert!(
        !text.contains(r#""role""#),
        "legacy completions has no role frame: {text}"
    );
}

/// A multi-prompt streaming request is rejected with 400 (single-prompt only).
#[tokio::test]
async fn completions_multi_prompt_streaming_is_400() {
    let app = rustedvino::create_router(AppState::mock());

    let body = r#"{"model":"mock","prompt":["a","b"],"stream":true}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

// ------------------------------------------------------------------
// Phase 2 — Admin model management routes
// These tests use AppState::mock() (no model manager) to verify
// the HTTP contract: routes exist, return the right status codes,
// and degrade gracefully when the model manager is absent.
// ------------------------------------------------------------------

/// POST /v1/admin/models/{id}/load returns 503 when no model manager.
///
/// The route must exist (not 404) but cannot serve without a manager.
#[tokio::test]
async fn admin_load_returns_503_without_model_manager() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/models/qwen3-8b-int4-ov/load")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "must return 503 when model manager is absent"
    );
}

/// POST .../load with an out-of-range override returns 400 — and does so
/// *before* the model-manager check, so a malformed request is rejected
/// regardless of server state (mock state has no manager).
#[tokio::test]
async fn admin_load_rejects_invalid_override_with_400() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/models/qwen3-8b-int4-ov/load")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"kv_cache_gb": 0.0}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "kv_cache_gb <= 0 must be a 400"
    );
}

/// POST .../load with malformed JSON returns 400 (not 503/500).
#[tokio::test]
async fn admin_load_rejects_malformed_json_with_400() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/models/qwen3-8b-int4-ov/load")
                .header("content-type", "application/json")
                .body(Body::from("{not json"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

/// POST .../load with an unknown field returns 400 — a typo in an override key
/// must not be silently ignored.
#[tokio::test]
async fn admin_load_rejects_unknown_field_with_400() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/models/qwen3-8b-int4-ov/load")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"kv_cache": 4.0}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

/// A valid override body still reaches the manager check (503 here) — proves a
/// well-formed body is accepted and threaded past validation.
#[tokio::test]
async fn admin_load_valid_override_passes_validation() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/models/qwen3-8b-int4-ov/load")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"kv_cache_gb": 4.0, "max_concurrent_streams": 8}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "valid body must pass validation and hit the (absent) manager → 503"
    );
}

/// DELETE /v1/admin/models/{id} returns 503 when no model manager.
#[tokio::test]
async fn admin_unload_returns_503_without_model_manager() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/admin/models/qwen3-8b-int4-ov")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "must return 503 when model manager is absent"
    );
}

// ------------------------------------------------------------------
// POST /tokenize and POST /detokenize (Phase 3.7)
// ------------------------------------------------------------------

/// POST /tokenize returns 503 (route exists, not 404) when no model manager —
/// there is no tokenizer to call in mock mode. Body is an `OpenAI` error envelope.
#[tokio::test]
async fn tokenize_returns_503_without_model_manager() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"qwen3-8b-int4-ov","prompt":"Hello world"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "must return 503 when model manager is absent (route exists)"
    );
    let json = body_json(response.into_body()).await;
    assert_eq!(json["error"]["type"], "server_error");
    assert_eq!(json["error"]["code"], "model_unavailable");
}

/// POST /detokenize returns 503 when no model manager (route exists, not 404).
#[tokio::test]
async fn detokenize_returns_503_without_model_manager() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/detokenize")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"qwen3-8b-int4-ov","tokens":[9906,1879]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "must return 503 when model manager is absent (route exists)"
    );
}

/// /tokenize uses POST, not GET — wrong method returns 405 (proves the route is
/// registered; a missing route would be 404).
#[tokio::test]
async fn tokenize_wrong_method_returns_405() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/tokenize")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

/// The admin load route uses POST, not GET — wrong method returns 405.
#[tokio::test]
async fn admin_load_wrong_method_returns_405() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/admin/models/qwen3-8b-int4-ov/load")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::METHOD_NOT_ALLOWED,
        "GET on a POST-only route must return 405"
    );
}

/// The admin unload route uses DELETE — wrong method returns 405.
#[tokio::test]
async fn admin_unload_wrong_method_returns_405() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/admin/models/qwen3-8b-int4-ov")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::METHOD_NOT_ALLOWED,
        "GET on a DELETE-only route must return 405"
    );
}

/// GET /v1/models returns an empty list when no model manager is present.
///
/// Explicitly tests the Phase 2 empty-manager case (the existing Phase 0
/// test also covers this but this one names the intent clearly).
#[tokio::test]
async fn models_list_empty_without_model_manager() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["data"].as_array().unwrap().len(), 0);
}

/// Chat with any model when no model manager is configured uses the mock path.
///
/// The mock path ignores the model name entirely and returns mock tokens.
/// This preserves backwards-compatible behaviour for tests.
#[tokio::test]
async fn chat_without_model_manager_uses_mock_path() {
    let app = rustedvino::create_router(AppState::mock());

    let body = r#"{"model":"any-model","messages":[{"role":"user","content":"hi"}],"stream":true}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    // Mock path always returns 200 SSE regardless of model name.
    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(content_type.starts_with("text/event-stream"));
}

// ------------------------------------------------------------------
// Phase 3.3 — GET /metrics
// ------------------------------------------------------------------

/// In mock mode the recorder is not installed → /metrics is gated with 503.
/// The populated happy-path (gauges with values) is wire-verified on the box,
/// since it needs a real `ModelManager` with a loaded model.
#[tokio::test]
async fn metrics_returns_503_without_recorder() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

// ------------------------------------------------------------------
// Phase 3.1 middleware — x-request-id + CORS
// ------------------------------------------------------------------

/// Every response carries a generated `x-request-id` when the client sent none.
#[tokio::test]
async fn response_has_generated_request_id() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let id = response
        .headers()
        .get("x-request-id")
        .unwrap()
        .to_str()
        .unwrap();
    // A UUIDv4 string is 36 chars (8-4-4-4-12 + hyphens).
    assert_eq!(id.len(), 36, "generated id should be a UUIDv4: {id}");
}

// ------------------------------------------------------------------
// x-ruvi-host / generation_metadata.host — hostname only for callers who
// passed a key check (inference or admin key); never on an open server,
// the exempt `/health`, or any rejected (401/403) request.
// ------------------------------------------------------------------

/// GET `uri` (optionally with a bearer key) and return the response.
async fn get_with_key(
    app: axum::Router,
    uri: &str,
    key: Option<&str>,
) -> axum::http::Response<Body> {
    let mut req = Request::builder().uri(uri);
    if let Some(key) = key {
        req = req.header("authorization", format!("Bearer {key}"));
    }
    app.oneshot(req.body(Body::empty()).unwrap()).await.unwrap()
}

fn keyed_app() -> axum::Router {
    rustedvino::create_router_with_cors(
        AppState::mock(),
        &["*".to_owned()],
        &["sk-secret".to_owned()],
    )
}

/// A valid inference key gets the box's real hostname (not the old
/// hardcoded `"ruvi-host"` placeholder).
#[tokio::test]
async fn ruvi_host_header_sent_for_valid_inference_key() {
    let resp = get_with_key(keyed_app(), "/v1/models", Some("sk-secret")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let host = resp.headers().get("x-ruvi-host").unwrap().to_str().unwrap();
    assert_eq!(host, rustedvino::os_memory::host_name().unwrap());
}

/// An admin key on an admin-scoped server also counts as key-validated.
#[tokio::test]
async fn ruvi_host_header_sent_for_valid_admin_key() {
    let resp = get_with_key(admin_scope_app(), "/v1/models", Some("sk-admin")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(resp.headers().get("x-ruvi-host").is_some());
}

#[tokio::test]
async fn ruvi_host_header_absent_without_valid_key() {
    for (key, want) in [
        (None, StatusCode::UNAUTHORIZED),
        (Some("sk-wrong"), StatusCode::UNAUTHORIZED),
    ] {
        let resp = get_with_key(keyed_app(), "/v1/models", key).await;
        assert_eq!(resp.status(), want);
        assert!(
            resp.headers().get("x-ruvi-host").is_none(),
            "key {key:?} must not see the host"
        );
    }
    // Authenticated but out of scope (inference key on an admin route) → 403, still no host.
    let resp = get_with_key(admin_scope_app(), "/v1/admin/models", Some("sk-inference")).await;
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    assert!(resp.headers().get("x-ruvi-host").is_none());
}

/// The exempt probe never carries it, even when a valid key is presented.
#[tokio::test]
async fn ruvi_host_header_absent_on_health() {
    for key in [None, Some("sk-secret")] {
        let resp = get_with_key(keyed_app(), "/health", key).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers().get("x-ruvi-host").is_none());
    }
}

/// A server with no keys configured never discloses its hostname.
#[tokio::test]
async fn ruvi_host_header_absent_on_open_server() {
    for uri in ["/health", "/v1/models"] {
        let resp = get_with_key(rustedvino::create_router(AppState::mock()), uri, None).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers().get("x-ruvi-host").is_none(), "{uri}");
    }
}

/// The admin-scope branch of the auth gate (an admin key on an admin route —
/// `/metrics` shares that gate) marks the request too; `/v1/models` with an
/// admin key only exercises the inference gate's superset branch.
#[tokio::test]
async fn ruvi_host_header_sent_on_admin_branch() {
    let resp = get_with_key(admin_scope_app(), "/metrics", Some("sk-admin")).await;
    // Past the gate (the mock state's handler answers 503: no metrics recorder).
    assert!(![StatusCode::UNAUTHORIZED, StatusCode::FORBIDDEN].contains(&resp.status()));
    assert!(resp.headers().get("x-ruvi-host").is_some());
}

/// Admin keys only (no inference keys): inference routes are open, so no host
/// there even with the admin key; admin routes validate the key and get it.
#[tokio::test]
async fn ruvi_host_on_admin_keys_only_server() {
    let app = || {
        rustedvino::create_router_with_auth(
            AppState::mock(),
            &["*".to_owned()],
            &[],
            &["sk-admin".to_owned()],
        )
    };
    for key in [None, Some("sk-admin")] {
        let resp = get_with_key(app(), "/v1/models", key).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(
            resp.headers().get("x-ruvi-host").is_none(),
            "open inference route, key {key:?}"
        );
    }
    let resp = get_with_key(app(), "/metrics", Some("sk-admin")).await;
    assert!(![StatusCode::UNAUTHORIZED, StatusCode::FORBIDDEN].contains(&resp.status()));
    assert!(resp.headers().get("x-ruvi-host").is_some());
}

/// Image `generation_metadata.host` follows the same gate as the header.
#[tokio::test]
async fn image_metadata_host_only_for_valid_key() {
    let body = r#"{"model":"sdxl","prompt":"a red circle","n":1,"size":"512x512"}"#;
    let post = |app: axum::Router, key: Option<&'static str>| async move {
        let mut req = Request::builder()
            .method("POST")
            .uri("/v1/images/generations")
            .header("content-type", "application/json");
        if let Some(key) = key {
            req = req.header("authorization", format!("Bearer {key}"));
        }
        let resp = app
            .oneshot(req.body(Body::from(body)).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        body_json(resp.into_body()).await
    };
    let open = post(rustedvino::create_router(AppState::mock()), None).await;
    assert!(
        open["generation_metadata"].get("host").is_none(),
        "open server: {open}"
    );
    assert!(
        open["generation_metadata"].get("engine").is_some(),
        "metadata itself still present"
    );
    let keyed = post(keyed_app(), Some("sk-secret")).await;
    assert_eq!(
        keyed["generation_metadata"]["host"],
        rustedvino::os_memory::host_name().unwrap()
    );
}

/// `x-client-request-id` from the client is echoed back as `x-request-id`.
#[tokio::test]
async fn response_echoes_client_request_id() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .header("x-client-request-id", "trace-abc-123")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response
            .headers()
            .get("x-request-id")
            .unwrap()
            .to_str()
            .unwrap(),
        "trace-abc-123",
        "server must echo the client's request id"
    );
}

/// Default (`["*"]`) CORS policy answers a preflight with a wildcard origin.
#[tokio::test]
async fn cors_default_allows_any_origin() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .method("OPTIONS")
                .uri("/v1/chat/completions")
                .header("origin", "https://anything.example.com")
                .header("access-control-request-method", "POST")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response
            .headers()
            .get("access-control-allow-origin")
            .unwrap()
            .to_str()
            .unwrap(),
        "*",
        "default policy must allow any origin"
    );
}

/// An explicit allowlist echoes a permitted origin and omits a non-listed one.
#[tokio::test]
async fn cors_explicit_allowlist_echoes_only_listed_origin() {
    let origins = ["https://app.example.com".to_owned()];

    // Listed origin → echoed back.
    let app = rustedvino::create_router_with_cors(AppState::mock(), &origins, &[]);
    let allowed = app
        .oneshot(
            Request::builder()
                .method("OPTIONS")
                .uri("/v1/chat/completions")
                .header("origin", "https://app.example.com")
                .header("access-control-request-method", "POST")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        allowed
            .headers()
            .get("access-control-allow-origin")
            .unwrap()
            .to_str()
            .unwrap(),
        "https://app.example.com",
    );

    // Non-listed origin → no allow-origin header echoed.
    let app = rustedvino::create_router_with_cors(AppState::mock(), &origins, &[]);
    let denied = app
        .oneshot(
            Request::builder()
                .method("OPTIONS")
                .uri("/v1/chat/completions")
                .header("origin", "https://evil.example.com")
                .header("access-control-request-method", "POST")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        denied
            .headers()
            .get("access-control-allow-origin")
            .is_none(),
        "non-listed origin must not be allowed"
    );
}

// ------------------------------------------------------------------
// G4 — Bearer authentication
// ------------------------------------------------------------------

/// With no `api_keys` configured, the server is open: a credential-free
/// request to a guarded route is NOT rejected with 401.
#[tokio::test]
async fn auth_open_mode_allows_unauthenticated_request() {
    let app = rustedvino::create_router_with_cors(AppState::mock(), &["*".to_owned()], &[]);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_ne!(
        resp.status(),
        StatusCode::UNAUTHORIZED,
        "open mode must not 401"
    );
}

/// With keys configured, a missing `Authorization` header → 401 with the
/// `OpenAI` envelope (`code:invalid_api_key`).
#[tokio::test]
async fn auth_missing_header_is_rejected() {
    let keys = ["sk-secret".to_owned()];
    let app = rustedvino::create_router_with_cors(AppState::mock(), &["*".to_owned()], &keys);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    // A 401 still carries the x-request-id (request-id wraps auth).
    assert!(
        resp.headers().get("x-request-id").is_some(),
        "401 must still carry x-request-id"
    );
    let json = body_json(resp.into_body()).await;
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert_eq!(json["error"]["code"], "invalid_api_key");
}

/// A wrong Bearer key → 401.
#[tokio::test]
async fn auth_wrong_key_is_rejected() {
    let keys = ["sk-secret".to_owned()];
    let app = rustedvino::create_router_with_cors(AppState::mock(), &["*".to_owned()], &keys);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header("authorization", "Bearer sk-wrong")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let json = body_json(resp.into_body()).await;
    assert_eq!(json["error"]["code"], "invalid_api_key");
}

/// A correct Bearer key → request is allowed through (not 401). Scheme match
/// is case-insensitive.
#[tokio::test]
async fn auth_correct_key_is_allowed() {
    let keys = ["sk-secret".to_owned()];
    let app = rustedvino::create_router_with_cors(AppState::mock(), &["*".to_owned()], &keys);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header("authorization", "bearer sk-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_ne!(
        resp.status(),
        StatusCode::UNAUTHORIZED,
        "a valid key must pass auth"
    );
}

/// `/health` is exempt: reachable uncredentialed even when keys are
/// configured — it's the only path in `AUTH_EXEMPT_PATHS`. `/metrics` used
/// to be exempt too; see `metrics_requires_key_when_keys_configured` below
/// for why that changed.
#[tokio::test]
async fn auth_health_is_exempt() {
    let keys = ["sk-secret".to_owned()];
    let app = rustedvino::create_router_with_cors(AppState::mock(), &["*".to_owned()], &keys);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_ne!(
        resp.status(),
        StatusCode::UNAUTHORIZED,
        "/health must be exempt from auth"
    );
}

/// `/metrics` leaks the loaded model catalog, VRAM capacity/usage, and
/// per-model request-rate data — real operational intelligence, not a
/// content-free ping like `/health`. It is gated exactly like an admin
/// route (see `METRICS_PATH`'s doc comment): a missing key is `401` once
/// any keys are configured, same as `/v1/admin/*`.
#[tokio::test]
async fn metrics_requires_key_when_keys_configured() {
    let keys = ["sk-secret".to_owned()];
    let app = rustedvino::create_router_with_cors(AppState::mock(), &["*".to_owned()], &keys);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::UNAUTHORIZED,
        "/metrics must require a key once auth is on"
    );
}

/// With a distinct admin scope configured, an inference-only key is `403`
/// on `/metrics` (authenticated, but lacks admin scope) — the same
/// `insufficient_scope` rule `/v1/admin/*` gets, since `/metrics` shares its
/// gate despite the different URL.
#[tokio::test]
async fn metrics_inference_key_is_forbidden() {
    let resp = admin_scope_app()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .header("authorization", "Bearer sk-inference")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let json = body_json(resp.into_body()).await;
    assert_eq!(json["error"]["code"], "insufficient_scope");
}

/// An admin key passes `/metrics`'s gate (mirrors
/// `admin_scope_admin_key_passes_admin_route`).
#[tokio::test]
async fn metrics_admin_key_passes() {
    let resp = admin_scope_app()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .header("authorization", "Bearer sk-admin")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_ne!(resp.status(), StatusCode::UNAUTHORIZED);
    assert_ne!(resp.status(), StatusCode::FORBIDDEN);
}

/// `admin_locked: true` (no keys file has ever existed) blocks `/metrics`
/// with `503` too, same as any `/v1/admin/*` route — mirrors
/// `admin_locked_blocks_admin_route_with_no_key`.
#[tokio::test]
async fn metrics_blocked_when_admin_locked() {
    let resp = admin_locked_app()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let json = body_json(resp.into_body()).await;
    assert_eq!(json["error"]["code"], "admin_not_configured");
}

/// Backward compat: with no keys configured anywhere (the common
/// `create_router`/mock-test default), `/metrics` clears the auth gate the
/// same "auth off entirely" way every other route does — not a
/// metrics-specific carve-out. The request still ends in `503` (asserted
/// separately by `metrics_returns_503_without_recorder`), but for an
/// unrelated reason: `AppState::mock()` has no metrics recorder, so the
/// *handler itself* 503s. Distinguished here from the auth-locked `503`
/// (`metrics_blocked_when_admin_locked`, `code: admin_not_configured`) by
/// checking the body — auth was never reached to reject anything.
#[tokio::test]
async fn metrics_clears_auth_gate_when_no_keys_configured_anywhere() {
    let app = rustedvino::create_router(AppState::mock());
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_ne!(resp.status(), StatusCode::UNAUTHORIZED);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        !body_str.contains("admin_not_configured"),
        "the 503 here must be metrics_handler's \"not enabled\" (no recorder in the mock \
         state), not the auth layer's admin-locked rejection — got: {body_str}"
    );
}

/// `/health_generate` is NOT exempt: it runs a real GPU generation, so with
/// keys configured a credential-free probe must be rejected 401 rather than
/// burning an inference slot uncredentialed.
#[tokio::test]
async fn auth_health_generate_requires_key() {
    let keys = ["sk-secret".to_owned()];
    let app = rustedvino::create_router_with_cors(AppState::mock(), &["*".to_owned()], &keys);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health_generate")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::UNAUTHORIZED,
        "/health_generate must require a key when auth is on"
    );
    let json = body_json(resp.into_body()).await;
    assert_eq!(json["error"]["code"], "invalid_api_key");
}

// ------------------------------------------------------------------
// G5 — T4.3 admin scope separation
// ------------------------------------------------------------------

/// Build a router with distinct inference and admin Bearer allowlists.
fn admin_scope_app() -> axum::Router {
    rustedvino::create_router_with_auth(
        AppState::mock(),
        &["*".to_owned()],
        &["sk-inference".to_owned()],
        &["sk-admin".to_owned()],
    )
}

/// The T4.3 accept criterion: an inference-only key → **403** on `/v1/admin/*`
/// (authenticated, but lacks admin scope), with the `insufficient_scope` code.
#[tokio::test]
async fn admin_scope_inference_key_is_forbidden_on_admin_route() {
    let resp = admin_scope_app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/models/qwen3-8b-int4-ov/load")
                .header("authorization", "Bearer sk-inference")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let json = body_json(resp.into_body()).await;
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert_eq!(json["error"]["code"], "insufficient_scope");
}

/// The unload route (DELETE) shares the admin prefix → an inference key is
/// likewise 403 there (covers both admin routes, not just load).
#[tokio::test]
async fn admin_scope_inference_key_is_forbidden_on_unload_route() {
    let resp = admin_scope_app()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/admin/models/qwen3-8b-int4-ov")
                .header("authorization", "Bearer sk-inference")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

/// An admin key passes the admin gate: the request reaches the handler (which
/// returns 503 without a model manager) — never 401/403 from auth.
#[tokio::test]
async fn admin_scope_admin_key_passes_admin_route() {
    let resp = admin_scope_app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/models/qwen3-8b-int4-ov/load")
                .header("authorization", "Bearer sk-admin")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_ne!(resp.status(), StatusCode::UNAUTHORIZED);
    assert_ne!(
        resp.status(),
        StatusCode::FORBIDDEN,
        "admin key must clear the admin gate"
    );
}

/// Admin keys are a superset credential: an admin key also satisfies the
/// inference gate on a normal route.
#[tokio::test]
async fn admin_scope_admin_key_passes_inference_route() {
    let resp = admin_scope_app()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .header("authorization", "Bearer sk-admin")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_ne!(
        resp.status(),
        StatusCode::UNAUTHORIZED,
        "an admin key must also pass inference routes"
    );
}

/// A missing key on an admin route with admin scope → 401 (not 403): there is
/// no authenticated identity to deny scope to.
#[tokio::test]
async fn admin_scope_missing_key_on_admin_route_is_401() {
    let resp = admin_scope_app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/models/qwen3-8b-int4-ov/load")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let json = body_json(resp.into_body()).await;
    assert_eq!(json["error"]["code"], "invalid_api_key");
}

/// An entirely unknown key on an admin route → 401 (not in either allowlist).
#[tokio::test]
async fn admin_scope_unknown_key_on_admin_route_is_401() {
    let resp = admin_scope_app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/models/qwen3-8b-int4-ov/load")
                .header("authorization", "Bearer sk-bogus")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let json = body_json(resp.into_body()).await;
    assert_eq!(json["error"]["code"], "invalid_api_key");
}

/// Backward compat: with NO admin scope configured (empty `admin_api_keys`,
/// the 3-arg `create_router_with_cors` path), an inference key reaches the
/// admin handler — admin falls back to the inference gate.
#[tokio::test]
async fn admin_scope_empty_falls_back_to_inference_gate() {
    let app = rustedvino::create_router_with_cors(
        AppState::mock(),
        &["*".to_owned()],
        &["sk-inference".to_owned()],
    );
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/models/qwen3-8b-int4-ov/load")
                .header("authorization", "Bearer sk-inference")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_ne!(resp.status(), StatusCode::UNAUTHORIZED);
    assert_ne!(
        resp.status(),
        StatusCode::FORBIDDEN,
        "with no admin scope, the inference key gates admin too"
    );
}

// ------------------------------------------------------------------
// admin_locked — no keys file existed at boot (the project's internal engineering log,
// "shall not allow inference open but admin also open when keys.json is
// not present")
// ------------------------------------------------------------------

/// Build a router with `AppState::admin_locked: true` and no keys
/// configured — the exact state `startup::bootstrap` produces when the
/// keys file doesn't exist.
fn admin_locked_app() -> axum::Router {
    let state = AppState::mock().with_admin_locked(true);
    rustedvino::create_router_with_auth(state, &["*".to_owned()], &[], &[])
}

/// `admin_locked: true` → every admin route is `503`, even with no key
/// presented at all.
#[tokio::test]
async fn admin_locked_blocks_admin_route_with_no_key() {
    let resp = admin_locked_app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/models/qwen3-8b-int4-ov/load")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let json = body_json(resp.into_body()).await;
    assert_eq!(json["error"]["code"], "admin_not_configured");
}

/// `admin_locked: true` blocks admin routes unconditionally — even a key
/// that would (if it existed in some `admin_api_keys` list) be a valid
/// admin credential elsewhere cannot get through, because there is no
/// `admin_api_keys` list to check against in this state at all.
#[tokio::test]
async fn admin_locked_blocks_admin_route_even_with_a_bearer_token_present() {
    let resp = admin_locked_app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/models/qwen3-8b-int4-ov/load")
                .header("authorization", "Bearer sk-anything")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

/// `admin_locked: true` does NOT affect inference routes — they still
/// follow the normal `api_keys`-empty-means-open rule.
#[tokio::test]
async fn admin_locked_leaves_inference_open() {
    let resp = admin_locked_app()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_ne!(
        resp.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "admin_locked must not lock inference routes"
    );
    assert_ne!(resp.status(), StatusCode::UNAUTHORIZED);
}

/// `/health` stays exempt even when admin is locked — infra probes are
/// checked before the admin-lock branch.
#[tokio::test]
async fn admin_locked_leaves_health_exempt() {
    let resp = admin_locked_app()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// ------------------------------------------------------------------
// A2: max_completion_tokens alias
// ------------------------------------------------------------------

/// `max_completion_tokens` is accepted and treated as the token budget when
/// `max_tokens` is absent. The mock path just streams tokens — the important
/// thing is that the request is not rejected with 400.
#[tokio::test]
async fn chat_max_completion_tokens_is_accepted() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":50}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    // The mock path returns 200 (not 400/422) — the field is accepted.
    assert_eq!(response.status(), StatusCode::OK);
}

// ------------------------------------------------------------------
// A3: malformed body → OpenAI error envelope
// ------------------------------------------------------------------

/// A completely invalid JSON body on POST /v1/chat/completions must return
/// HTTP 400 with the `OpenAI` `{error:{message,type,code}}` envelope.
#[tokio::test]
async fn malformed_json_body_returns_openai_error_envelope() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from("not valid json {{{"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let json = body_json(response.into_body()).await;
    assert_eq!(
        json["error"]["type"], "invalid_request_error",
        "must be invalid_request_error: {json}"
    );
    assert_eq!(
        json["error"]["code"], "invalid_body",
        "must carry invalid_body code: {json}"
    );
    // T4.4 (finding 52): the message must be generic — it must NOT reflect the
    // caller's input back nor leak serde's struct-fingerprinting rejection text.
    let msg = json["error"]["message"].as_str().unwrap_or_default();
    assert!(!msg.is_empty(), "error message must not be empty: {json}");
    assert!(
        !msg.contains("{{{") && !msg.contains("expected value"),
        "error message must not reflect input or serde internals: {msg:?}"
    );
}

/// A well-typed JSON body but with a wrong-typed field (e.g. `model: 42`
/// instead of a string) must also return the `OpenAI` error envelope.
#[tokio::test]
async fn wrong_type_field_returns_openai_error_envelope() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":42,"messages":[]}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert_eq!(json["error"]["code"], "invalid_body");
}

/// The same envelope must appear on POST /v1/completions.
#[tokio::test]
async fn completions_malformed_body_returns_openai_envelope() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from("{broken"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

// ------------------------------------------------------------------
// GET /health_generate  (Phase 3.8)
// ------------------------------------------------------------------

/// Without a `ModelManager` (mock state), the deep health probe returns 503
/// with a JSON error body — no ready models available.
#[tokio::test]
async fn health_generate_no_model_manager_returns_503() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health_generate")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["status"], "error");
    assert!(
        json["message"].is_string(),
        "error body must carry a message field"
    );
}

#[tokio::test]
async fn vlm_image_request_in_mock_mode_is_served() {
    let app = rustedvino::create_router(AppState::mock());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::json!({
                    "model": "any",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is in this image?"},
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}}
                        ]
                    }]
                }).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    // R1: capability routing lives behind the ModelManager. In mock mode (no
    // manager) the request falls through to the mock token generator, so it is
    // served (200) rather than rejected. The real text-vs-vision routing matrix
    // is exercised by the in-crate `handlers::chat::tests` against a mock-factory
    // manager (image→text=400, text/image→vision=served).
    assert_eq!(response.status(), StatusCode::OK);
}

// ------------------------------------------------------------------
// POST /v1/embeddings (R4/G3)
// ------------------------------------------------------------------

/// Mock-path embeddings: `object:"list"`, one `data` entry per input with a
/// float vector + index, and `usage` populated.
#[tokio::test]
async fn embeddings_float_returns_list_with_vectors() {
    let app = rustedvino::create_router(AppState::mock());

    let body = r#"{"model":"mock-embed","input":"hello world"}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response.into_body()).await;

    assert_eq!(json["object"], "list");
    assert_eq!(json["model"], "mock-embed");
    assert_eq!(json["data"][0]["object"], "embedding");
    assert_eq!(json["data"][0]["index"], 0);
    assert!(
        json["data"][0]["embedding"].is_array(),
        "float encoding → JSON array\n---\n{json}"
    );
    assert_eq!(json["usage"]["prompt_tokens"], 1);
    assert_eq!(json["usage"]["total_tokens"], 1);
}

/// An array `input` produces one indexed embedding per element (mock path).
#[tokio::test]
async fn embeddings_array_input_yields_one_entry_each() {
    let app = rustedvino::create_router(AppState::mock());

    let body = r#"{"model":"mock-embed","input":["a","b","c"]}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["data"].as_array().unwrap().len(), 3);
    assert_eq!(json["data"][0]["index"], 0);
    assert_eq!(json["data"][1]["index"], 1);
    assert_eq!(json["data"][2]["index"], 2);
    assert_eq!(json["usage"]["prompt_tokens"], 3);
}

/// `encoding_format:"base64"` → the embedding is a base64 string, not an array.
#[tokio::test]
async fn embeddings_base64_encoding_returns_string() {
    let app = rustedvino::create_router(AppState::mock());

    let body = r#"{"model":"mock-embed","input":"hi","encoding_format":"base64"}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response.into_body()).await;
    assert!(
        json["data"][0]["embedding"].is_string(),
        "base64 encoding → JSON string\n---\n{json}"
    );
}

/// An unknown `encoding_format` is a 400 with the `OpenAI` error envelope.
#[tokio::test]
async fn embeddings_bad_encoding_format_is_400() {
    let app = rustedvino::create_router(AppState::mock());

    let body = r#"{"model":"mock-embed","input":"hi","encoding_format":"weird"}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

/// A missing `input` field is a 400 (the `JsonBody` envelope), not a panic.
#[tokio::test]
async fn embeddings_missing_input_is_400() {
    let app = rustedvino::create_router(AppState::mock());

    let body = r#"{"model":"mock-embed"}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

/// A `dimensions` request is rejected explicitly (400 `unsupported_parameter`,
/// `param:"dimensions"`) rather than silently returning a full-size vector the
/// caller didn't ask for — this server has no vector-truncation support.
#[tokio::test]
async fn embeddings_dimensions_param_is_rejected() {
    let app = rustedvino::create_router(AppState::mock());

    let body = r#"{"model":"mock-embed","input":"hi","dimensions":256}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert_eq!(json["error"]["code"], "unsupported_parameter");
    assert_eq!(json["error"]["param"], "dimensions");
}

/// An `input` array beyond the embeddings cap (`max_embedding_inputs`, default
/// 256) is rejected with a clean 400 rather than accepted unbounded.
#[tokio::test]
async fn embeddings_input_array_beyond_cap_is_rejected() {
    let app = rustedvino::create_router(AppState::mock());

    let inputs: Vec<String> = (0..257).map(|i| format!("\"input {i}\"")).collect();
    let body = format!(r#"{{"model":"mock-embed","input":[{}]}}"#, inputs.join(","));
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert_eq!(json["error"]["code"], "too_many_inputs");
}

/// An `input` array at (not beyond) the cap is accepted — 256, well past the
/// `/v1/completions` `max_prompt_array` (16) embeddings used to share.
#[tokio::test]
async fn embeddings_input_array_at_cap_is_accepted() {
    let app = rustedvino::create_router(AppState::mock());

    let inputs: Vec<String> = (0..256).map(|i| format!("\"input {i}\"")).collect();
    let body = format!(r#"{{"model":"mock-embed","input":[{}]}}"#, inputs.join(","));
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["data"].as_array().unwrap().len(), 256);
}

// ------------------------------------------------------------------
// Phase 5.2a — /v1/audio/speech (TTS)
// ------------------------------------------------------------------

/// A valid TTS request in mock mode (no model manager) returns 200 + WAV bytes
/// starting with the RIFF header.
#[tokio::test]
async fn tts_mock_returns_wav() {
    let app = rustedvino::create_router(AppState::mock());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"speecht5-tts","input":"Hello world."}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers().get("content-type").unwrap(), "audio/wav");
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(
        bytes.starts_with(b"RIFF"),
        "response must start with RIFF header"
    );
    assert!(
        bytes.len() > 44,
        "WAV must have data beyond the 44-byte header"
    );
}

/// `response_format=pcm` returns raw i16 bytes (no RIFF header).
#[tokio::test]
async fn tts_mock_pcm_format() {
    let app = rustedvino::create_router(AppState::mock());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"speecht5-tts","input":"Hi.","response_format":"pcm"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "audio/octet-stream"
    );
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(!bytes.starts_with(b"RIFF"), "PCM must not have WAV header");
    assert!(!bytes.is_empty(), "PCM must have sample bytes");
}

/// Empty `input` → 400.
#[tokio::test]
async fn tts_empty_input_is_400() {
    let app = rustedvino::create_router(AppState::mock());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"speecht5-tts","input":"  "}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

/// Unsupported `response_format` → 400.
#[tokio::test]
async fn tts_unknown_response_format_is_400() {
    let app = rustedvino::create_router(AppState::mock());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"speecht5-tts","input":"hi","response_format":"mp3"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

/// `speed` out of range → 400.
#[tokio::test]
async fn tts_speed_out_of_range_is_400() {
    let app = rustedvino::create_router(AppState::mock());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"speecht5-tts","input":"hi","speed":10.0}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

/// A media route exists for POST only — a GET is 405, not 404. Proves the route
/// is registered (an unregistered path would be 404).
#[tokio::test]
async fn media_route_wrong_method_is_405() {
    let app = rustedvino::create_router(AppState::mock());
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/audio/transcriptions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

// ------------------------------------------------------------------
// POST /v1/audio/transcriptions — STT 5.1a (mock transcription, no GPU)
// ------------------------------------------------------------------

const MP_BOUNDARY: &str = "RVTESTBOUNDARY";

/// Build a `multipart/form-data` body. Each field is `(name, optional filename,
/// content)`; a filename marks it as a file part.
fn build_multipart(fields: &[(&str, Option<&str>, &[u8])]) -> Vec<u8> {
    let mut body = Vec::new();
    for (name, filename, content) in fields {
        body.extend_from_slice(format!("--{MP_BOUNDARY}\r\n").as_bytes());
        match filename {
            Some(fname) => body.extend_from_slice(
                format!(
                    "Content-Disposition: form-data; name=\"{name}\"; filename=\"{fname}\"\r\n\
                     Content-Type: application/octet-stream\r\n\r\n"
                )
                .as_bytes(),
            ),
            None => body.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            ),
        }
        body.extend_from_slice(content);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{MP_BOUNDARY}--\r\n").as_bytes());
    body
}

/// POST a multipart transcription request and return the response.
async fn post_transcription(fields: &[(&str, Option<&str>, &[u8])]) -> axum::http::Response<Body> {
    let app = rustedvino::create_router(AppState::mock());
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header(
                "content-type",
                format!("multipart/form-data; boundary={MP_BOUNDARY}"),
            )
            .body(Body::from(build_multipart(fields)))
            .unwrap(),
    )
    .await
    .unwrap()
}

/// An upload over the 50 MiB request-body limit is an honest 413
/// `request_too_large` naming the limit — it used to be a 400
/// "could not read field 'file'" (a 123 MB podcast hit it).
#[tokio::test]
async fn transcription_over_body_limit_is_413() {
    let big = vec![0u8; 51 * 1024 * 1024];
    let response = post_transcription(&[
        ("model", None, b"whisper-stt"),
        ("file", Some("big.wav"), &big),
    ])
    .await;
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["error"]["code"], "request_too_large");
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("50 MiB")
    );
}

/// The same honest 413 for an oversized JSON body (a chat request with a huge
/// inline image, say) — it used to be 400 "malformed request body".
#[tokio::test]
async fn json_body_over_limit_is_413() {
    let app = rustedvino::create_router(AppState::mock());
    let body = format!(
        r#"{{"model":"m","messages":[{{"role":"user","content":"{}"}}]}}"#,
        "x".repeat(51 * 1024 * 1024)
    );
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["error"]["code"], "request_too_large");
}

/// POST a multipart request to `/v1/audio/translations`.
async fn post_translation(fields: &[(&str, Option<&str>, &[u8])]) -> axum::http::Response<Body> {
    let app = rustedvino::create_router(AppState::mock());
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/audio/translations")
            .header(
                "content-type",
                format!("multipart/form-data; boundary={MP_BOUNDARY}"),
            )
            .body(Body::from(build_multipart(fields)))
            .unwrap(),
    )
    .await
    .unwrap()
}

/// `/v1/audio/translations` is routed, shares the transcription request
/// parsing and response formats, and runs the translate task.
#[tokio::test]
async fn translation_route_returns_translated_text_in_each_format() {
    let json_resp = post_translation(&[
        ("model", None, b"whisper-stt"),
        ("file", Some("a.wav"), b"fake-audio"),
    ])
    .await;
    assert_eq!(json_resp.status(), StatusCode::OK);
    let json = body_json(json_resp.into_body()).await;
    assert_eq!(json["text"], "This is a mock translation.");

    let srt = post_translation(&[
        ("model", None, b"whisper-stt"),
        ("response_format", None, b"srt"),
        ("file", Some("a.wav"), b"fake-audio"),
    ])
    .await;
    assert_eq!(srt.status(), StatusCode::OK);

    // verbose_json names the task that ran, as OpenAI's does.
    let verbose = post_translation(&[
        ("model", None, b"whisper-stt"),
        ("response_format", None, b"verbose_json"),
        ("file", Some("a.wav"), b"fake-audio"),
    ])
    .await;
    assert_eq!(verbose.status(), StatusCode::OK);
    assert_eq!(body_json(verbose.into_body()).await["task"], "translate");

    let missing = post_translation(&[("model", None, b"whisper-stt")]).await;
    assert_eq!(missing.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn transcription_missing_file_is_400() {
    // model only, no file part.
    let response = post_transcription(&[("model", None, b"whisper-stt")]).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert_eq!(json["error"]["code"], "missing_file");
}

#[tokio::test]
async fn transcription_json_default_returns_text() {
    // No response_format → json default.
    let response = post_transcription(&[
        ("file", Some("a.wav"), b"RIFFsome-fake-audio"),
        ("model", None, b"whisper-stt"),
    ])
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response.into_body()).await;
    assert!(
        json["text"].as_str().is_some_and(|s| !s.is_empty()),
        "json format → non-empty text field"
    );
}

#[tokio::test]
async fn transcription_text_format_is_plain() {
    let response = post_transcription(&[
        ("file", Some("a.wav"), b"RIFFsome-fake-audio"),
        ("model", None, b"whisper-stt"),
        ("response_format", None, b"text"),
    ])
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let ctype = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();
    assert!(
        ctype.starts_with("text/plain"),
        "text format → text/plain, got {ctype}"
    );
    let body = body_string(response.into_body()).await;
    assert!(!body.is_empty());
    assert!(!body.trim_start().starts_with('{'), "must not be JSON");
}

#[tokio::test]
async fn transcription_srt_format_has_cue_header() {
    let response = post_transcription(&[
        ("file", Some("a.wav"), b"RIFFsome-fake-audio"),
        ("model", None, b"whisper-stt"),
        ("response_format", None, b"srt"),
    ])
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_string(response.into_body()).await;
    assert!(
        body.starts_with("1\n00:00:00,000 --> "),
        "srt format → first cue header, got: {body:?}"
    );
}

#[tokio::test]
async fn transcription_verbose_json_has_segments() {
    let response = post_transcription(&[
        ("file", Some("a.wav"), b"RIFFsome-fake-audio"),
        ("model", None, b"whisper-stt"),
        ("response_format", None, b"verbose_json"),
        ("language", None, b"pl"),
    ])
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["task"], "transcribe");
    assert_eq!(json["language"], "pl");
    let segments = json["segments"].as_array().unwrap();
    assert!(!segments.is_empty());
    let first = &segments[0];
    assert!(first["start"].is_number());
    assert!(first["end"].is_number());
    assert!(first["text"].is_string());
}

#[tokio::test]
async fn transcription_unknown_response_format_is_400() {
    let response = post_transcription(&[
        ("file", Some("a.wav"), b"RIFFsome-fake-audio"),
        ("model", None, b"whisper-stt"),
        ("response_format", None, b"xml"),
    ])
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["error"]["code"], "invalid_response_format");
}

// ------------------------------------------------------------------
// POST /v1/images/generations — Image 5.3a (mock canned PNG, no GPU)
// ------------------------------------------------------------------

/// POST a JSON image-generation request and return the response.
async fn post_image(body: &str) -> axum::http::Response<Body> {
    let app = rustedvino::create_router(AppState::mock());
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/images/generations")
            .header("content-type", "application/json")
            .body(Body::from(body.to_owned()))
            .unwrap(),
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn image_happy_path_returns_b64_png() {
    use base64ct::{Base64, Encoding as _};
    let response =
        post_image(r#"{"model":"sdxl","prompt":"a red circle","n":2,"size":"512x512"}"#).await;
    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response.into_body()).await;
    let data = json["data"].as_array().unwrap();
    assert_eq!(data.len(), 2, "n=2 yields two images");
    let b64 = data[0]["b64_json"].as_str().unwrap();
    let bytes = Base64::decode_vec(b64).unwrap();
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "decodes to a PNG");
}

#[tokio::test]
async fn image_n_out_of_range_is_400() {
    let response = post_image(r#"{"model":"sdxl","prompt":"x","n":9}"#).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["error"]["code"], "invalid_n");
}

#[tokio::test]
async fn image_url_response_format_is_400() {
    let response = post_image(r#"{"model":"sdxl","prompt":"x","response_format":"url"}"#).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["error"]["code"], "invalid_response_format");
}

#[tokio::test]
async fn image_bad_size_is_400() {
    let response = post_image(r#"{"model":"sdxl","prompt":"x","size":"800x600"}"#).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["error"]["code"], "invalid_size");
}

// ------------------------------------------------------------------
// POST /v1/images/edits: reject a too-small source image
// ------------------------------------------------------------------

/// Encode a solid-color `width`×`height` PNG for `image_edits` test uploads.
fn encode_test_png(width: u32, height: u32) -> Vec<u8> {
    let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(width, height, |_, _| {
        image::Rgb([200, 150, 100])
    }));
    let mut buf = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .unwrap();
    buf
}

/// POST a multipart image-edit request and return the response.
async fn post_image_edit(fields: &[(&str, Option<&str>, &[u8])]) -> axum::http::Response<Body> {
    let app = rustedvino::create_router(AppState::mock());
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/images/edits")
            .header(
                "content-type",
                format!("multipart/form-data; boundary={MP_BOUNDARY}"),
            )
            .body(Body::from(build_multipart(fields)))
            .unwrap(),
    )
    .await
    .unwrap()
}

/// A source image below the pipeline's real minimum-safe size must be
/// rejected with a clean 400, not passed through — regression test for the
/// live SIGSEGV this closes: an undersized `image_edits` upload used to
/// reach `OpenVINO`'s img2img pipeline directly, whose shape-inference failure
/// left the C++ pipeline corrupted and crashed the whole process on the next
/// operation against it. Full writeup:
/// the project's internal engineering log.
#[tokio::test]
async fn image_edit_too_small_source_is_400_not_a_crash() {
    let png = encode_test_png(64, 64);
    let response = post_image_edit(&[
        ("image", Some("source.png"), png.as_slice()),
        ("model", None, b"sdxl"),
        ("prompt", None, b"test"),
    ])
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["error"]["code"], "image_too_small");
}

/// A source image at exactly the minimum-safe size is accepted (mock path
/// renders a canned image, same as `/v1/images/generations`'s happy path) —
/// the fix must not reject legitimately-sized uploads.
#[tokio::test]
async fn image_edit_minimum_size_source_is_accepted() {
    let png = encode_test_png(256, 256);
    let response = post_image_edit(&[
        ("image", Some("source.png"), png.as_slice()),
        ("model", None, b"sdxl"),
        ("prompt", None, b"test"),
    ])
    .await;
    assert_eq!(response.status(), StatusCode::OK);
}

/// `image_edits`' `generation_metadata.host` is gated like the header: none on
/// an open server, the real hostname for a key-validated caller.
#[tokio::test]
async fn image_edit_metadata_host_only_for_valid_key() {
    let png = encode_test_png(256, 256);
    let body = build_multipart(&[
        ("image", Some("source.png"), png.as_slice()),
        ("model", None, b"sdxl"),
        ("prompt", None, b"test"),
    ]);
    let post = |app: axum::Router, key: Option<&'static str>, body: Vec<u8>| async move {
        let mut req = Request::builder()
            .method("POST")
            .uri("/v1/images/edits")
            .header(
                "content-type",
                format!("multipart/form-data; boundary={MP_BOUNDARY}"),
            );
        if let Some(key) = key {
            req = req.header("authorization", format!("Bearer {key}"));
        }
        let resp = app
            .oneshot(req.body(Body::from(body)).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        body_json(resp.into_body()).await
    };
    let open = post(
        rustedvino::create_router(AppState::mock()),
        None,
        body.clone(),
    )
    .await;
    assert!(
        open["generation_metadata"].get("host").is_none(),
        "open server: {open}"
    );
    let keyed = post(keyed_app(), Some("sk-secret"), body).await;
    assert_eq!(
        keyed["generation_metadata"]["host"],
        rustedvino::os_memory::host_name().unwrap()
    );
}
