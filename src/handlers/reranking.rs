// ============================================================
// src/handlers/reranking.rs — POST /v1/rerank
// ============================================================
// Cohere-compatible reranking endpoint. Accepts a query + a list
// of documents, returns results sorted by relevance score descending,
// trimmed to `top_n`. Mirrors the structure of handlers/embeddings.rs.
// ============================================================

use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::Response;
use axum::{Json, extract::State, response::IntoResponse};
use serde::{Deserialize, Serialize};

use crate::app_state::AppState;
use crate::handlers::error::{model_error_response, openai_error};
use crate::model_manager::{ModelError, ModelManager, OnDemandLoad};
use crate::rerank_engine::RerankingHandle;

/// Conservative worst-case bytes-per-token, mirrors `handlers::embeddings`'s
/// `MAX_BYTES_PER_TOKEN` — a cheap pre-tokenize length check so a
/// multi-megabyte pair doesn't pay for a tokenizer round-trip just to be
/// rejected. The real precision comes from the tokenizer-backed check below;
/// this is only the fast, generous first pass.
const MAX_BYTES_PER_TOKEN: usize = 32;

// ── Request ────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct RerankRequest {
    /// Model to use for reranking.
    pub model: String,
    /// The search query.
    pub query: String,
    /// The documents to rank against the query.
    pub documents: Vec<String>,
    /// Maximum number of results to return. Defaults to all documents.
    pub top_n: Option<usize>,
}

// ── Response ───────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct RerankResponse {
    pub object: &'static str,
    pub model: String,
    pub results: Vec<RerankResult>,
    pub usage: RerankUsage,
}

#[derive(Debug, Serialize)]
pub struct RerankResult {
    /// Zero-based index into the original `documents` array.
    pub index: usize,
    pub relevance_score: f32,
    pub document: RerankDocument,
}

#[derive(Debug, Serialize)]
pub struct RerankDocument {
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct RerankUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

// ── Handler ────────────────────────────────────────────────────────────────────

/// `POST /v1/rerank` — rerank documents against a query.
///
/// Returns results sorted by relevance score descending, trimmed to `top_n`.
/// Token-count fields in `usage` are always 0 (the cross-encoder does not
/// expose per-request token counts).
pub async fn rerank(
    State(state): State<AppState>,
    Json(req): Json<RerankRequest>,
) -> impl IntoResponse {
    if req.query.is_empty() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "Field 'query' must not be empty.",
            "invalid_request_error",
            None,
        );
    }
    if req.documents.is_empty() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "Field 'documents' must not be empty.",
            "invalid_request_error",
            None,
        );
    }

    let n_docs = req.documents.len();
    let top_n = req.top_n.unwrap_or(n_docs).min(n_docs);

    let Some(mm) = state.model_manager.as_ref() else {
        return openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Model manager is not available.",
            "server_error",
            None,
        );
    };

    let handle = match resolve_rerank_handle(mm, &req.model) {
        Ok(h) => h,
        Err(resp) => return *resp,
    };

    for (index, document) in req.documents.iter().enumerate() {
        if let Some(resp) = gate_rerank_pair(&handle, &req.query, document, index).await {
            return resp;
        }
    }

    let query = req.query.clone();
    let documents = req.documents.clone();

    let raw_results = match handle.rerank(query, documents).await {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(model = %req.model, err = %e, "reranking failed");
            return openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Reranking failed: {e}"),
                "server_error",
                None,
            );
        }
    };

    // Trim to top_n and build the response items, preserving the
    // original document text at each result's original index.
    let results: Vec<RerankResult> = raw_results
        .results
        .into_iter()
        .take(top_n)
        .map(|(orig_idx, score)| RerankResult {
            index: orig_idx,
            relevance_score: score,
            document: RerankDocument {
                text: req.documents[orig_idx].clone(),
            },
        })
        .collect();

    tracing::info!(
        model = %req.model,
        n_docs = n_docs,
        top_n = top_n,
        top_score = results.first().map_or(0.0, |r| f64::from(r.relevance_score)),
        "reranking complete"
    );

    (
        StatusCode::OK,
        Json(RerankResponse {
            object: "list",
            model: req.model,
            results,
            usage: RerankUsage {
                prompt_tokens: 0,
                total_tokens: 0,
            },
        }),
    )
        .into_response()
}

/// L0 length gate for one (query, document) pair
/// (the project's internal engineering log): without this, a pair
/// that tokenizes past the model's `max_position_embeddings` reaches
/// `TextRerankPipeline::rerank` directly and raises an OV shape-inference
/// exception — surfaced as an opaque 500, and (per the linked investigation)
/// leaves the pipeline's internal tensor state corrupted for every
/// subsequent request on the same load, not just this one.
///
/// Skipped entirely when the model's `config.json` didn't declare
/// `max_position_embeddings` — fail-open, same policy as every other length
/// gate in this codebase. A tokenizer-call error also fails open rather than
/// becoming a false 400 (mirrors `handlers::embeddings::gate_embed_text`).
async fn gate_rerank_pair(
    handle: &RerankingHandle,
    query: &str,
    document: &str,
    index: usize,
) -> Option<Response> {
    let max_seq_len = handle.max_seq_len()?;

    // Cheap byte pre-check first — skips a tokenizer round-trip for a
    // pathologically oversized pair without needing real precision.
    let combined_len = query.len() + document.len();
    if combined_len > max_seq_len.saturating_mul(MAX_BYTES_PER_TOKEN) {
        return Some(openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "documents[{index}] combined with the query is {combined_len} bytes — too \
                 large to fit this model's {max_seq_len}-token position-embedding capacity \
                 under even the most generous tokenizer compression; shorten the query or \
                 this document"
            ),
            "invalid_request_error",
            Some("context_length_exceeded"),
        ));
    }

    // Real tokenizer-backed check: sums the query's and document's own token
    // counts (not a true paired encoding — see `ov_rerank_count_tokens`'s
    // doc comment for why — but still far more precise than the byte
    // heuristic above, and never undercounts), catching dense
    // ordinary-language pairs the byte heuristic is deliberately too
    // generous to catch.
    match handle
        .count_tokens(query.to_owned(), document.to_owned())
        .await
    {
        Ok(count) if count > max_seq_len => Some(openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "documents[{index}] combined with the query has {count} tokens — exceeds \
                 this model's {max_seq_len}-token position-embedding capacity; shorten the \
                 query or this document"
            ),
            "invalid_request_error",
            Some("context_length_exceeded"),
        )),
        Ok(_) | Err(_) => None,
    }
}

/// Resolve the reranking-kind model handle for `model`, mapping every
/// `ModelError` to its HTTP response. Extracted from `rerank` to keep the
/// handler under clippy's 100-line function cap.
///
/// Includes the Slice 3c on-demand-load lazy path for `NotLoaded` (mirrors
/// chat/media/embeddings; a gap found by stress testing, 2026-08-04 — reranking previously
/// fell straight through to the generic "not ready" 503 with no auto-load).
/// An `Eager` model (today's only configured reranking model on any fleet
/// box) takes the `NotApplicable` arm, which reproduces byte-for-byte the
/// message the old fallthrough `Err(e)` arm used to produce for
/// `NotLoaded` — no behaviour change for the models actually in service.
fn resolve_rerank_handle(
    mm: &Arc<ModelManager>,
    model: &str,
) -> Result<RerankingHandle, Box<Response>> {
    match mm.get_reranking_handle(model) {
        Ok(h) => Ok(h),
        Err(ModelError::NotFound(_)) => Err(Box::new(openai_error(
            StatusCode::NOT_FOUND,
            format!("Model '{model}' not found. Load it first via /v1/admin/models."),
            "invalid_request_error",
            Some("model_not_found"),
        ))),
        Err(ModelError::WrongKind(msg)) => Err(Box::new(openai_error(
            StatusCode::BAD_REQUEST,
            msg,
            "invalid_request_error",
            None,
        ))),
        Err(ModelError::NotLoaded) => Err(Box::new(match mm.request_on_demand_load(model) {
            OnDemandLoad::Loading => model_error_response(ModelError::Loading),
            OnDemandLoad::NotApplicable => openai_error(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Model '{model}' is not ready: {}", ModelError::NotLoaded),
                "server_error",
                None,
            ),
        })),
        Err(e) => Err(Box::new(openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("Model '{model}' is not ready: {e}"),
            "server_error",
            None,
        ))),
    }
}

// ============================================================
// Unit tests
// ============================================================
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    use super::{MAX_BYTES_PER_TOKEN, gate_rerank_pair};
    use crate::app_state::AppState;
    use crate::create_router;
    use crate::model_manager::lifecycle::MockEngineFactory;
    use crate::model_manager::{Config, ModelManager};
    use crate::rerank_engine::RerankingHandle;

    fn test_config(model_name: &str) -> Config {
        Config {
            models_dir: std::path::PathBuf::from("/tmp/test-rerank-models"),
            device: "CPU".to_owned(),
            preload: vec![],
            models: [(
                model_name.to_owned(),
                crate::model_manager::config::ModelEntry {
                    vram_gb: 1.0,
                    kind: None,
                    policy: crate::model_manager::config::ModelPolicy::default(),
                },
            )]
            .into_iter()
            .collect(),
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
            device_budgets: std::collections::HashMap::new(),
            kv_cache_precision: String::new(),
            enable_prefix_caching: true,
            cors_allowed_origins: vec!["*".to_owned()],
            api_keys: Vec::new(),
            admin_api_keys: Vec::new(),
            keys_file: None,
            admission_queue_timeout_ms: 5_000,
            device_admission_queue_timeout_ms: 5_000,
            embedding_pooling: "mean".to_owned(),
            embedding_normalize: true,
            default_embed_model: None,
            max_prompt_array: 16,
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

    async fn state_with_model(model_name: &str) -> AppState {
        let mm = ModelManager::new(
            test_config(model_name),
            Arc::new(MockEngineFactory::default()),
        )
        .await
        .unwrap();
        mm.load_model(model_name).await.unwrap();
        AppState::new().with_model_manager(Arc::new(mm))
    }

    async fn body_json(resp: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        serde_json::from_slice(&bytes).expect("json")
    }

    async fn post_rerank(app: axum::Router, body: serde_json::Value) -> axum::response::Response {
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/rerank")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn rerank_model_not_found() {
        // Use a manager with no registered models so the lookup returns NotFound.
        let mm = ModelManager::new(
            Config {
                models_dir: std::path::PathBuf::from("/tmp/test-rerank-models"),
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
                device_budgets: std::collections::HashMap::new(),
                kv_cache_precision: String::new(),
                enable_prefix_caching: true,
                cors_allowed_origins: vec!["*".to_owned()],
                api_keys: Vec::new(),
                admin_api_keys: Vec::new(),
                keys_file: None,
                admission_queue_timeout_ms: 5_000,
                device_admission_queue_timeout_ms: 5_000,
                embedding_pooling: "mean".to_owned(),
                embedding_normalize: true,
                default_embed_model: None,
                max_prompt_array: 16,
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
            },
            Arc::new(MockEngineFactory::default()),
        )
        .await
        .unwrap();
        let state = AppState::new().with_model_manager(Arc::new(mm));
        let app = create_router(state);
        let resp = post_rerank(
            app,
            serde_json::json!({
                "model": "no-such-model",
                "query": "test query",
                "documents": ["doc0", "doc1"]
            }),
        )
        .await;
        assert_eq!(resp.status(), 404);
    }

    #[tokio::test]
    async fn rerank_wrong_kind_rejected() {
        let state = state_with_model("text-model").await;
        let app = create_router(state);
        let resp = post_rerank(
            app,
            serde_json::json!({
                "model": "text-model",
                "query": "test query",
                "documents": ["doc0", "doc1"]
            }),
        )
        .await;
        assert_eq!(resp.status(), 400);
        let j = body_json(resp).await;
        assert!(
            j["error"]["message"]
                .as_str()
                .unwrap()
                .contains("reranking models")
        );
    }

    #[tokio::test]
    async fn rerank_empty_query_rejected() {
        let state = state_with_model("mock-rerank").await;
        let app = create_router(state);
        let resp = post_rerank(
            app,
            serde_json::json!({
                "model": "mock-rerank",
                "query": "",
                "documents": ["doc0"]
            }),
        )
        .await;
        assert_eq!(resp.status(), 400);
    }

    #[tokio::test]
    async fn rerank_empty_documents_rejected() {
        let state = state_with_model("mock-rerank").await;
        let app = create_router(state);
        let resp = post_rerank(
            app,
            serde_json::json!({
                "model": "mock-rerank",
                "query": "hello",
                "documents": []
            }),
        )
        .await;
        assert_eq!(resp.status(), 400);
    }

    #[tokio::test]
    async fn rerank_returns_sorted_results() {
        let state = state_with_model("mock-rerank").await;
        let app = create_router(state);
        let resp = post_rerank(
            app,
            serde_json::json!({
                "model": "mock-rerank",
                "query": "test query",
                "documents": ["doc0", "doc1", "doc2"],
                "top_n": 2
            }),
        )
        .await;
        assert_eq!(resp.status(), 200);
        let j = body_json(resp).await;
        assert_eq!(j["object"], "list");
        let results = j["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        // Results must be in descending relevance_score order.
        let s0 = results[0]["relevance_score"].as_f64().unwrap();
        let s1 = results[1]["relevance_score"].as_f64().unwrap();
        assert!(s0 >= s1, "results not sorted: {s0} < {s1}");
        // Each result must carry back the document text.
        assert!(results[0]["document"]["text"].as_str().is_some());
    }

    #[tokio::test]
    async fn rerank_top_n_clamps_to_n_docs() {
        let state = state_with_model("mock-rerank").await;
        let app = create_router(state);
        let resp = post_rerank(
            app,
            serde_json::json!({
                "model": "mock-rerank",
                "query": "test query",
                "documents": ["doc0", "doc1"],
                "top_n": 99
            }),
        )
        .await;
        assert_eq!(resp.status(), 200);
        let j = body_json(resp).await;
        assert_eq!(j["results"].as_array().unwrap().len(), 2);
    }

    // ── L0 length gate (dev/autotest/20260804_rerank_shape_poisoning.md) ──

    /// No configured ceiling → the gate never rejects, no matter how long.
    #[tokio::test]
    async fn gate_is_a_no_op_when_max_seq_len_unknown() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<crate::rerank_engine::RerankCommand>(4);
        let handle = RerankingHandle::from_sender(tx, "test-rerank");
        assert!(
            gate_rerank_pair(&handle, &"word ".repeat(10_000), "doc", 0)
                .await
                .is_none()
        );
    }

    /// A (query, document) pair within the byte-length ceiling passes. No
    /// live responder needed: the byte pre-check passes, and the subsequent
    /// tokenizer call fails open against this dead channel (no receiver) —
    /// exercised for real, with a live mock responder, by
    /// `gate_accepts_a_pair_whose_real_token_count_is_within_the_ceiling` below.
    #[tokio::test]
    async fn gate_accepts_a_pair_within_the_ceiling() {
        let (tx, rx) = tokio::sync::mpsc::channel::<crate::rerank_engine::RerankCommand>(4);
        drop(rx);
        let handle = RerankingHandle::from_sender_with_max_seq_len(tx, "test-rerank", 512);
        assert!(
            gate_rerank_pair(&handle, "short query", "short document", 0)
                .await
                .is_none()
        );
    }

    /// A pair whose combined byte length exceeds the ceiling is rejected with
    /// a clean 400 `context_length_exceeded`, never reaching the pipeline —
    /// this is the exact trigger that poisoned the pipeline before this gate
    /// existed. Caught by the byte pre-check alone, before any tokenizer call.
    #[tokio::test]
    async fn gate_rejects_an_over_length_pair_with_400() {
        let (tx, rx) = tokio::sync::mpsc::channel::<crate::rerank_engine::RerankCommand>(4);
        drop(rx);
        let handle = RerankingHandle::from_sender_with_max_seq_len(tx, "test-rerank", 512);
        let huge_document = "x".repeat(512 * MAX_BYTES_PER_TOKEN + 1);
        let resp = gate_rerank_pair(&handle, "query", &huge_document, 3)
            .await
            .expect("must reject an over-length pair");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// The query's own length counts toward the combined ceiling, not just
    /// the document's — a cross-encoder feeds both into one sequence.
    #[tokio::test]
    async fn gate_counts_the_query_toward_the_combined_length() {
        let (tx, rx) = tokio::sync::mpsc::channel::<crate::rerank_engine::RerankCommand>(4);
        drop(rx);
        let handle = RerankingHandle::from_sender_with_max_seq_len(tx, "test-rerank", 512);
        let huge_query = "x".repeat(512 * MAX_BYTES_PER_TOKEN + 1);
        let resp = gate_rerank_pair(&handle, &huge_query, "short doc", 0)
            .await
            .expect("a huge query alone must trip the gate");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// A pair whose *real*, tokenizer-backed count exceeds the ceiling is
    /// rejected even when it's well under the byte pre-check's deliberately
    /// generous threshold — the exact class of gap the byte-only gate had
    /// before this session's `ov_rerank_count_tokens` FFI addition (dense
    /// ordinary-language text tokenizes far below the 32-bytes/token
    /// worst case the pre-check assumes).
    #[tokio::test]
    async fn gate_rejects_a_pair_whose_real_token_count_exceeds_the_ceiling_even_within_byte_budget()
     {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<crate::rerank_engine::RerankCommand>(4);
        let handle = RerankingHandle::from_sender_with_max_seq_len(tx, "test-rerank", 512);

        tokio::spawn(async move {
            if let Some(crate::rerank_engine::RerankCommand::CountTokens { reply, .. }) =
                rx.recv().await
            {
                let _ = reply.send(Ok(600));
            }
        });

        let resp = gate_rerank_pair(&handle, "short query", "short document", 0)
            .await
            .expect("must reject a pair whose real token count exceeds the ceiling");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// A pair whose real token count is within the ceiling passes — proves
    /// the tokenizer-backed check doesn't over-reject a legitimate pair.
    #[tokio::test]
    async fn gate_accepts_a_pair_whose_real_token_count_is_within_the_ceiling() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<crate::rerank_engine::RerankCommand>(4);
        let handle = RerankingHandle::from_sender_with_max_seq_len(tx, "test-rerank", 512);

        tokio::spawn(async move {
            if let Some(crate::rerank_engine::RerankCommand::CountTokens { reply, .. }) =
                rx.recv().await
            {
                let _ = reply.send(Ok(100));
            }
        });

        assert!(
            gate_rerank_pair(&handle, "short query", "short document", 0)
                .await
                .is_none()
        );
    }
}
