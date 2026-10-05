// ============================================================
// src/handlers/systemone.rs — POST /v1/systemone (Ollama v0.35.0-compatible)
// ============================================================
// Typed decisions: every question is answered by one prefill of a decision model, reading
// the logits of its candidate code tokens. Wire format and messages are Ollama's
// (`server/routes.go SystemOneHandler`), and so is the ORDER of checks — clients see the
// same status for the same mistake:
//
//   1. first JSON value runs past 64 KiB       → 413 "request body must not exceed 64 KiB"
//   2. body not a decodable request           → 400 (decoder message)
//   3. request invalid (decision.Compile)     → 400 (Ollama's message)
//   4. model unknown                          → 404 "model '<m>' not found"
//   5. model is not a decision model          → 400 (Ollama's message)
//   6. device budget busy (if configured)      → 429 (shared with chat etc. on that device)
//   7. any prompt breaks the length rule      → 400 (checked for ALL prompts before any prefill)
//   8. inference / scorer failure             → 500
//
// Errors are Ollama's `{"error": "..."}`, not the OpenAI envelope. The body is taken as raw
// bytes: Ollama (gin `ShouldBindJSON`) ignores Content-Type and trailing data, and axum's
// `Json` extractor would reject both.
// ============================================================

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::app_state::AppState;
use crate::decision_engine::DecideError;
use crate::model_manager::{ModelError, OnDemandLoad};
use crate::systemone::{SystemOneError, answer, compile, decode_request};

fn ollama_error(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(serde_json::json!({ "error": message.into() }))).into_response()
}

fn status_of(e: &SystemOneError) -> StatusCode {
    StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
}

/// Resolve the decision model — Ollama's 404/400 for unknown or non-decision models, and
/// 503 (with `Retry-After` while loading) for one that is not ready, starting an on-demand
/// load if its policy allows.
fn resolve_decision_handle(
    mm: &std::sync::Arc<crate::model_manager::ModelManager>,
    model: &str,
) -> Result<crate::decision_engine::DecisionHandle, Box<Response>> {
    match mm.get_decision_handle(model) {
        Ok(h) => Ok(h),
        Err(ModelError::NotFound(_)) => Err(Box::new(ollama_error(
            StatusCode::NOT_FOUND,
            format!("model '{model}' not found"),
        ))),
        Err(ModelError::WrongKind(msg)) => {
            Err(Box::new(ollama_error(StatusCode::BAD_REQUEST, msg)))
        }
        Err(ModelError::NotLoaded) => Err(Box::new(match mm.request_on_demand_load(model) {
            OnDemandLoad::Loading => not_ready_transient(model, &ModelError::Loading),
            OnDemandLoad::NotApplicable => ollama_error(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("model '{model}' is not ready: {}", ModelError::NotLoaded),
            ),
        })),
        Err(e @ (ModelError::Loading | ModelError::Evicting)) => {
            Err(Box::new(not_ready_transient(model, &e)))
        }
        Err(e) => Err(Box::new(ollama_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("model '{model}' is not ready: {e}"),
        ))),
    }
}

/// A model that will be ready soon (loading, or evicting before a reload): 503 in Ollama's
/// `{"error": "..."}` shape — not the `OpenAI` envelope the shared helper builds — plus the
/// server-wide `Retry-After`. (Ollama itself blocks until the load finishes instead.)
fn not_ready_transient(model: &str, e: &ModelError) -> Response {
    let mut resp = ollama_error(
        StatusCode::SERVICE_UNAVAILABLE,
        format!("model '{model}' is not ready: {e}"),
    );
    resp.headers_mut().insert(
        axum::http::header::RETRY_AFTER,
        crate::handlers::error::RETRY_AFTER_TRANSIENT,
    );
    resp
}

/// A decision engine whose reload failed cannot serve again on this load. Evict the model
/// in the background (so it shows as not loaded, and an on-demand model reloads on the next
/// request) and answer 503 instead of a permanent 500.
fn evict_dead_engine(
    mm: &std::sync::Arc<crate::model_manager::ModelManager>,
    model: &str,
) -> Response {
    let (mm, id) = (std::sync::Arc::clone(mm), model.to_owned());
    tokio::spawn(async move {
        if let Err(e) = mm.evict_model(&id).await {
            tracing::error!(model = %id, error = %e, "evicting a dead decision engine failed");
        }
    });
    ollama_error(
        StatusCode::SERVICE_UNAVAILABLE,
        format!("model '{model}' failed and is being unloaded; retry to load it again"),
    )
}

/// `POST /v1/systemone` — answer typed questions about `state` with calibrated probabilities.
pub async fn systemone(State(state): State<AppState>, body: Bytes) -> Response {
    let req = match decode_request(&body) {
        Ok(r) => r,
        Err(e) => return ollama_error(status_of(&e), e.to_string()),
    };
    let compiled = match compile(&req) {
        Ok(c) => c,
        Err(e) => return ollama_error(status_of(&e), e.to_string()),
    };
    let Some(mm) = state.model_manager.as_ref() else {
        return ollama_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "model manager is not available",
        );
    };
    let handle = match resolve_decision_handle(mm, &req.model) {
        Ok(h) => h,
        Err(resp) => return *resp,
    };

    if handle.is_dead() {
        return evict_dead_engine(mm, &req.model);
    }
    // Cross-pipeline device admission, like every other inference endpoint: a decision request
    // is up to 64 prefills of up to 8K tokens, and on a GPU shared with chat it must queue
    // behind the same per-device budget instead of bypassing it. A pure no-op when the
    // device has no configured budget. The lease is held until all questions are scored.
    //
    // The engine scores one request at a time, so on a budgeted device only the request at
    // the front can use a lease. The rest wait at the model's turn gate holding nothing:
    // taking a lease first would park every queued request on a slot it cannot use and starve
    // chat and embeddings of the same budget (RTH `systemone_budget_control_*`).
    let device = mm.record_device(&req.model);
    let too_busy = |device: &str| {
        ollama_error(
            StatusCode::TOO_MANY_REQUESTS,
            format!("server at capacity — device '{device}' busy, retry shortly"),
        )
    };
    let turn = if state.device_budgets.is_budgeted(&device) {
        match handle.wait_turn(state.device_budgets.queue_timeout()).await {
            Ok(turn) => turn,
            Err(_) => return too_busy(&device),
        }
    } else {
        handle.ungated_turn()
    };
    let _device_lease = match state
        .device_budgets
        .admit(&[(device, 1)], crate::admission::WorkClass::Decide)
        .await
    {
        Ok(lease) => lease,
        Err(e) => return too_busy(&e.device),
    };
    let prompts = compiled.prompts(&handle.meta().system_prompt);
    let widths: Vec<usize> = compiled.fields.iter().map(|f| f.choices.len()).collect();
    let out = match handle.decide_in_turn(turn, prompts, widths).await {
        Ok(o) => o,
        Err(DecideError::Length(m)) => return ollama_error(StatusCode::BAD_REQUEST, m),
        // Only produced after the client disconnected, so nobody reads this response.
        Err(DecideError::Cancelled) => {
            return ollama_error(
                StatusCode::SERVICE_UNAVAILABLE,
                DecideError::Cancelled.to_string(),
            );
        }
        Err(DecideError::Engine(m)) => {
            tracing::error!(model = %req.model, err = %m, "systemone scoring failed");
            if handle.is_dead() {
                // This failure also broke the engine's reload: start recovery now.
                let _ = evict_dead_engine(mm, &req.model);
            }
            return ollama_error(StatusCode::INTERNAL_SERVER_ERROR, m);
        }
    };
    let input_tokens: usize = out.prompt_tokens.iter().sum();
    // Ollama counts one predicted token per scored row (+1 when it primes a shared prefix;
    // v1 never primes).
    let output_tokens = out.logits.len();
    match answer(
        &req.model,
        &compiled,
        &out.logits,
        input_tokens,
        output_tokens,
    ) {
        Ok(body) => {
            tracing::info!(
                model = %req.model,
                questions = out.logits.len(),
                input_tokens,
                prefill_ms_total = out.prefill_ms.iter().sum::<f64>(),
                "systemone decision complete"
            );
            (StatusCode::OK, Json(body)).into_response()
        }
        Err(e) => ollama_error(status_of(&e), e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    use crate::app_state::AppState;
    use crate::create_router;
    use crate::decision_engine::mock_meta;
    use crate::model_manager::lifecycle::MockEngineFactory;
    use crate::model_manager::{Config, ModelManager};
    use crate::systemone::wire::MAX_BODY_BYTES;
    use crate::systemone::{compile, decode_request};

    const DECIDE: &str = "mock-decide";
    const CHAT: &str = "mock-chat";

    async fn app() -> axum::Router {
        let cfg: Config = serde_json::from_value(serde_json::json!({
            "models_dir": "/tmp/test-systemone-models", "device": "CPU", "total_vram_gb": 22.5,
            "models": { DECIDE: {"vram_gb": 1.0}, CHAT: {"vram_gb": 1.0} }
        }))
        .unwrap();
        let mm = ModelManager::new(cfg, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();
        mm.load_model(DECIDE).await.unwrap();
        mm.load_model(CHAT).await.unwrap();
        create_router(AppState::new().with_model_manager(Arc::new(mm)))
    }

    async fn post(
        app: axum::Router,
        body: impl Into<Body>,
        content_type: Option<&str>,
    ) -> (StatusCode, serde_json::Value) {
        let mut req = Request::builder().method("POST").uri("/v1/systemone");
        if let Some(ct) = content_type {
            req = req.header("content-type", ct);
        }
        let resp = app.oneshot(req.body(body.into()).unwrap()).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    fn body(model: &str, state: &str) -> String {
        serde_json::json!({
            "model": model, "state": state,
            "questions": {
                "label": {"type": "choice", "instructions": "Which label?", "criteria": {"billing": null, "bug": null, "account": null}},
                "refund": {"type": "noul", "instructions": "Refund?"},
                "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Routine", "Soon", "Urgent"]}
            }
        })
        .to_string()
    }

    #[tokio::test]
    async fn answers_all_three_types_in_ollama_shape() {
        let (status, v) = post(
            app().await,
            body(DECIDE, "checkout is down"),
            Some("application/json"),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        // Mock logits = candidate index, so the last candidate wins everywhere.
        assert_eq!(v["model"], DECIDE);
        assert_eq!(v["answers"]["label"]["type"], "choice");
        assert_eq!(v["answers"]["label"]["choice"], "account");
        assert!(v["answers"]["refund"]["noul"].as_f64().unwrap() > 0.5);
        assert_eq!(v["answers"]["urgency"]["legend"]["2"], "Urgent");
        let keys: Vec<&String> = v["answers"].as_object().unwrap().keys().collect();
        assert_eq!(
            keys,
            ["label", "refund", "urgency"],
            "request order is kept"
        );
        assert_eq!(v["usage"]["output_tokens"], 3);
        assert!(v["usage"]["input_tokens"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn content_type_and_trailing_data_are_ignored_like_gin() {
        let (status, _) = post(app().await, format!("{}x", body(DECIDE, "s")), None).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn body_cap_is_64_kib_exactly() {
        // Ollama caps what its single Decode reads: the first JSON value must end within 64 KiB,
        // trailing whitespace/garbage may run past it (200); a value crossing the limit is 413.
        let base = body(DECIDE, "s");
        let trailing = format!("{base}{}", " ".repeat(MAX_BODY_BYTES + 5000));
        assert_eq!(post(app().await, trailing, None).await.0, StatusCode::OK);
        let one = |state: &str| {
            serde_json::json!({"model": DECIDE, "state": state,
                "questions": {"a": {"type": "noul", "instructions": "i"}}})
            .to_string()
        };
        let fits = one(&"y".repeat(MAX_BODY_BYTES - one("").len()));
        assert_eq!(fits.len(), MAX_BODY_BYTES);
        assert_ne!(
            post(app().await, fits, None).await.0,
            StatusCode::PAYLOAD_TOO_LARGE
        );
        let crosses = one(&"y".repeat(MAX_BODY_BYTES + 1 - one("").len()));
        let (status, v) = post(app().await, crosses, None).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(v["error"], "request body must not exceed 64 KiB");
    }

    #[tokio::test]
    async fn validation_runs_before_model_lookup() {
        // Invalid request naming an unknown model: Ollama answers 400, not 404.
        let bad = r#"{"model":"nope","state":"s","questions":{"a":{"type":"noul"}}}"#;
        let (status, v) = post(app().await, bad, None).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            v["error"],
            r#"question "a": instructions must be a nonempty string, object, or array"#
        );
        let (status, v) = post(app().await, "not json", None).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(v["error"].is_string());
    }

    #[tokio::test]
    async fn unknown_and_wrong_kind_models() {
        let (status, v) = post(app().await, body("nope", "s"), None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(v["error"], "model 'nope' not found");
        let (status, v) = post(app().await, body(CHAT, "s"), None).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            v["error"],
            r#"model "mock-chat" is not supported by System One; use a local Nimble or Tev GGUF model"#
        );
    }

    /// Review R9 (= the B70 test's F3): a registered but *unloaded* non-decision model must be refused
    /// with Ollama's 400 without being loaded — before the fix, readiness was checked first,
    /// the handler saw `NotLoaded` and started an on-demand load of the chat model.
    #[tokio::test]
    async fn unloaded_wrong_kind_model_is_refused_without_loading_it() {
        const IDLE_CHAT: &str = "mock-idlechat";
        let cfg: Config = serde_json::from_value(serde_json::json!({
            "models_dir": "/tmp/test-systemone-models", "device": "CPU", "total_vram_gb": 22.5,
            "models": { DECIDE: {"vram_gb": 1.0}, IDLE_CHAT: {"vram_gb": 1.0, "load": "on_demand"} }
        }))
        .unwrap();
        let mm = Arc::new(
            ModelManager::new(cfg, Arc::new(MockEngineFactory::default()))
                .await
                .unwrap(),
        );
        mm.load_model(DECIDE).await.unwrap();
        let app = create_router(AppState::new().with_model_manager(Arc::clone(&mm)));
        let (status, v) = post(app, body(IDLE_CHAT, "s"), None).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
        assert_eq!(
            v["error"],
            r#"model "mock-idlechat" is not supported by System One; use a local Nimble or Tev GGUF model"#
        );
        // Give a (wrongly) spawned background load time to start, then prove none did.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let state = mm
            .list_models()
            .into_iter()
            .find(|m| m.id == IDLE_CHAT)
            .map(|m| format!("{:?}", m.state));
        assert_eq!(
            state.as_deref(),
            Some("NotLoaded"),
            "the chat model must not have been loaded"
        );
    }

    /// Review R15: a decision engine whose reload failed must not stay "Ready" and fail every
    /// request with a 500 — the handler answers 503 and evicts it, so it can be reloaded.
    #[tokio::test]
    async fn dead_engine_is_evicted_not_left_failing() {
        let cfg: Config = serde_json::from_value(serde_json::json!({
            "models_dir": "/tmp/test-systemone-models", "device": "CPU", "total_vram_gb": 22.5,
            "models": { DECIDE: {"vram_gb": 1.0} }
        }))
        .unwrap();
        let mm = Arc::new(
            ModelManager::new(cfg, Arc::new(MockEngineFactory::default()))
                .await
                .unwrap(),
        );
        mm.load_model(DECIDE).await.unwrap();
        mm.get_decision_handle(DECIDE).unwrap().mark_dead_for_test();
        let app = create_router(AppState::new().with_model_manager(Arc::clone(&mm)));
        let (status, v) = post(app, body(DECIDE, "s"), None).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{v}");
        assert_eq!(
            v["error"],
            "model 'mock-decide' failed and is being unloaded; retry to load it again"
        );
        for _ in 0..50 {
            let state = mm
                .list_models()
                .into_iter()
                .find(|m| m.id == DECIDE)
                .map(|m| format!("{:?}", m.state));
            if state.as_deref() == Some("NotLoaded") {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        panic!("the dead decision engine was not evicted");
    }

    /// Review R10: the "model is loading" answer must use Ollama's `{"error": "..."}` shape
    /// (a string, like every other error on this route) plus `Retry-After`, not the `OpenAI`
    /// envelope — an Ollama client reading `error` as a string broke on exactly this path.
    #[tokio::test]
    async fn loading_answer_is_ollama_shaped_with_retry_after() {
        const LAZY: &str = "mock-decide-lazy";
        let cfg: Config = serde_json::from_value(serde_json::json!({
            "models_dir": "/tmp/test-systemone-models", "device": "CPU", "total_vram_gb": 22.5,
            "models": { LAZY: {"vram_gb": 1.0, "load": "on_demand"} }
        }))
        .unwrap();
        let mm = Arc::new(
            ModelManager::new(cfg, Arc::new(MockEngineFactory::default()))
                .await
                .unwrap(),
        );
        let app = create_router(AppState::new().with_model_manager(Arc::clone(&mm)));
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/systemone")
                    .body(Body::from(body(LAZY, "s")))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok()),
            Some("5")
        );
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            v["error"].is_string(),
            "Ollama shape: error must be a string, got {v}"
        );
        assert_eq!(
            v["error"],
            "model 'mock-decide-lazy' is not ready: model is currently loading"
        );
        // The background load completes and the same request is then served.
        for _ in 0..50 {
            let (status, _) = post(app.clone(), body(LAZY, "s"), None).await;
            if status == StatusCode::OK {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        panic!("the on-demand decision model never became ready");
    }

    /// A decision request takes a lease on its device's cross-pipeline budget, the same one chat
    /// uses: with the only slot held by chat it waits and gets 429; once chat is done it runs.
    #[tokio::test]
    async fn decision_shares_the_device_budget_with_chat() {
        let cfg_json = serde_json::json!({
            "models_dir": "/tmp/test-systemone-models", "device": "CPU", "total_vram_gb": 22.5,
            "device_budgets": {"CPU": 1}, "device_admission_queue_timeout_ms": 50,
            "models": { DECIDE: {"vram_gb": 1.0} }
        });
        let cfg: Config = serde_json::from_value(cfg_json.clone()).unwrap();
        let budgets = Arc::new(crate::admission::DeviceBudgets::from_config(&cfg));
        let mm = Arc::new(
            ModelManager::new(cfg, Arc::new(MockEngineFactory::default()))
                .await
                .unwrap(),
        );
        mm.load_model(DECIDE).await.unwrap();
        let device = mm.record_device(DECIDE);
        assert_eq!(
            device, "CPU",
            "the budget below is keyed by the model's device"
        );
        let app = create_router(
            AppState::new()
                .with_model_manager(Arc::clone(&mm))
                .with_device_budgets(Arc::clone(&budgets)),
        );
        let chat_lease = budgets
            .admit(&[(device.clone(), 1)], crate::admission::WorkClass::Chat)
            .await
            .unwrap();
        let (status, v) = post(app.clone(), body(DECIDE, "s"), None).await;
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS, "{v}");
        assert_eq!(
            v["error"],
            format!("server at capacity — device '{device}' busy, retry shortly")
        );
        drop(chat_lease);
        let (status, v) = post(app, body(DECIDE, "s"), None).await;
        assert_eq!(status, StatusCode::OK, "{v}");
    }

    /// Decision requests queued behind a running one wait at the model's turn gate holding NO
    /// device lease, so the whole budget stays available to chat. (Before the gate every queued
    /// request took a lease it could not use: with a budget of 2, two queued decisions starved
    /// chat and embeddings into 429s — RTH `systemone_budget_control_*`.)
    #[tokio::test]
    async fn queued_decision_requests_hold_no_device_lease() {
        let cfg_json = serde_json::json!({
            "models_dir": "/tmp/test-systemone-models", "device": "CPU", "total_vram_gb": 22.5,
            "device_budgets": {"CPU": 2}, "device_admission_queue_timeout_ms": 2000,
            "models": { DECIDE: {"vram_gb": 1.0} }
        });
        let cfg: Config = serde_json::from_value(cfg_json).unwrap();
        let budgets = Arc::new(crate::admission::DeviceBudgets::from_config(&cfg));
        let mm = Arc::new(
            ModelManager::new(cfg, Arc::new(MockEngineFactory::default()))
                .await
                .unwrap(),
        );
        mm.load_model(DECIDE).await.unwrap();
        let app = create_router(
            AppState::new()
                .with_model_manager(Arc::clone(&mm))
                .with_device_budgets(Arc::clone(&budgets)),
        );
        // Another decision request is "running": it holds the model's turn.
        let handle = super::resolve_decision_handle(&mm, DECIDE).ok().unwrap();
        let running = handle
            .wait_turn(std::time::Duration::from_secs(1))
            .await
            .unwrap();

        // A second request queues behind it, at the gate.
        let queued = tokio::spawn({
            let app = app.clone();
            async move { post(app, body(DECIDE, "s"), None).await }
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Chat can still take the entire budget at once.
        let mut leases = Vec::new();
        for n in 0..2 {
            let lease = tokio::time::timeout(
                std::time::Duration::from_millis(200),
                budgets.admit(&[("CPU".to_owned(), 1)], crate::admission::WorkClass::Chat),
            )
            .await
            .unwrap_or_else(|_| panic!("chat lease {n} blocked behind a queued decision request"))
            .unwrap();
            leases.push(lease);
        }
        drop(leases);

        drop(running);
        let (status, v) = queued.await.unwrap();
        assert_eq!(status, StatusCode::OK, "{v}");
    }

    /// A request that cannot get the model's turn within the queue timeout is shed with the
    /// same Ollama-shaped 429 as a saturated device budget.
    #[tokio::test]
    async fn turn_gate_timeout_is_a_429() {
        let cfg_json = serde_json::json!({
            "models_dir": "/tmp/test-systemone-models", "device": "CPU", "total_vram_gb": 22.5,
            "device_budgets": {"CPU": 2}, "device_admission_queue_timeout_ms": 50,
            "models": { DECIDE: {"vram_gb": 1.0} }
        });
        let cfg: Config = serde_json::from_value(cfg_json).unwrap();
        let budgets = Arc::new(crate::admission::DeviceBudgets::from_config(&cfg));
        let mm = Arc::new(
            ModelManager::new(cfg, Arc::new(MockEngineFactory::default()))
                .await
                .unwrap(),
        );
        mm.load_model(DECIDE).await.unwrap();
        let app = create_router(
            AppState::new()
                .with_model_manager(Arc::clone(&mm))
                .with_device_budgets(budgets),
        );
        let handle = super::resolve_decision_handle(&mm, DECIDE).ok().unwrap();
        let _running = handle
            .wait_turn(std::time::Duration::from_secs(1))
            .await
            .unwrap();
        let (status, v) = post(app, body(DECIDE, "s"), None).await;
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS, "{v}");
        assert_eq!(
            v["error"],
            "server at capacity — device 'CPU' busy, retry shortly"
        );
    }

    /// The mock counts one token per prompt character, so a `state` of the right length puts
    /// the (single) prompt on each side of Ollama's two boundaries (ctx 8194).
    #[tokio::test]
    async fn length_boundaries_match_ollama() {
        let one = |state: &str| {
            serde_json::json!({"model": DECIDE, "state": state,
                "questions": {"a": {"type": "noul", "instructions": "i"}}})
            .to_string()
        };
        let system = mock_meta().system_prompt;
        let prompt_len = |state: &str| {
            let c = compile(&decode_request(one(state).as_bytes()).unwrap()).unwrap();
            c.prompts(&system)[0].chars().count()
        };
        let overhead = prompt_len("x") - 1; // one ASCII char of state = one prompt char
        for (n, want) in [
            (8192, None),
            (
                8193,
                Some("prompt 0 requires 8195 context tokens for scoring; model has 8194"),
            ),
            (
                8194,
                Some("prompt 0 requires 8196 context tokens for scoring; model has 8194"),
            ),
            (
                8195,
                Some("prompt 0 has 8195 tokens; expected 1–8194 (input is never truncated)"),
            ),
        ] {
            let state = "x".repeat(n - overhead);
            assert_eq!(prompt_len(&state), n);
            let (status, v) = post(app().await, one(&state), None).await;
            match want {
                None => assert_eq!(status, StatusCode::OK, "n={n}: {v}"),
                Some(msg) => {
                    assert_eq!(status, StatusCode::BAD_REQUEST, "n={n}: {v}");
                    assert_eq!(v["error"], msg, "n={n}");
                }
            }
        }
    }
}
