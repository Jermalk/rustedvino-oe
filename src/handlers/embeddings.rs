// ============================================================
// src/handlers/embeddings.rs — POST /v1/embeddings (R4/G3)
// ============================================================
// The OpenAI embeddings endpoint: text in, float vectors out. This is the
// first real "media" flow to land in the engine registry — it routes to a
// `ModelKind::Embedding` model via `ModelManager::get_embedding_handle`, which
// reaches the single-stream `TextEmbeddingPipeline` on its dedicated thread.
//
// We target the OpenAI spec (stormVINO has the same shape):
//   request : { model, input: string | [string,…], encoding_format }
//   response: { object:"list", model, data:[{object:"embedding",index,embedding}],
//               usage:{prompt_tokens, total_tokens} }
//
// `encoding_format` defaults to `"float"`; `"base64"` is supported because the
// openai-python client requests base64 by default (little-endian f32 bytes,
// standard base64). `dimensions` / `user` are accepted-and-ignored for now.
// ============================================================

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse as _, Response},
};
use base64ct::{Base64, Encoding as _};
use serde::{Deserialize, Serialize};

use super::error::{
    JsonBody, device_admission_error_response, inference_error_response, model_error_response,
    openai_error, unsupported_param,
};
use crate::app_state::AppState;
use crate::embed_engine::EmbeddingHandle;
use crate::model_manager::{ModelError, OnDemandLoad};

/// Conservative worst-case bytes-per-token, mirrors `handlers::chat`'s
/// `MAX_BYTES_PER_TOKEN` — a cheap pre-tokenize length check so a
/// multi-megabyte input doesn't pay for a tokenizer round-trip just to be
/// rejected.
const MAX_BYTES_PER_TOKEN: usize = 32;

// ---- Request types --------------------------------------------------

/// Incoming `POST /v1/embeddings` body (`OpenAI`-compatible). Only the fields
/// the server acts on are declared and enforced; `user` is any other
/// `OpenAI` param — opaque client-side tracking metadata with no effect on
/// the response, so it's ignored by serde rather than rejected. `dimensions`
/// is declared (unlike `user`) purely so the handler can detect it was sent
/// and reject it explicitly — see `unsupported_param` — because honoring it
/// would change the response shape, unlike `user`.
#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    /// Which loaded embedding model to use. No default fallback.
    pub model: String,

    /// The text(s) to embed: a single string or an array of strings (→ one
    /// embedding per element). Required — an absent `input` is a 400.
    pub input: EmbeddingInput,

    /// `"float"` (default) → numeric arrays; `"base64"` → little-endian f32
    /// bytes, base64-encoded (the openai-python default).
    pub encoding_format: Option<String>,

    /// `OpenAI` v3-model param to truncate the output vector to a smaller
    /// size. Not supported by this server's pipeline — captured here only so
    /// the handler can reject it explicitly instead of silently returning a
    /// full-size vector the caller didn't ask for.
    pub dimensions: Option<serde_json::Value>,
}

/// The `input` field's polymorphic shape: a single string or an array of them.
///
/// `#[serde(untagged)]` tries `Single` (a JSON string) first, then `Multi` (a
/// JSON array of strings). Array-of-token-id inputs are not supported — they
/// fail both arms and surface as a 400, the honest answer until the pipeline
/// accepts pre-tokenized input. Mirrors `completions::Prompt`.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// One text to embed.
    Single(String),
    /// Several texts → several embeddings.
    Multi(Vec<String>),
}

impl EmbeddingInput {
    /// Normalise to a vector of input strings.
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::Multi(v) => v,
        }
    }
}

// ---- Response types --------------------------------------------------

/// A full embeddings response (`object: "list"`).
#[derive(Serialize)]
struct EmbeddingResponse<'a> {
    object: &'static str,
    data: Vec<EmbeddingData>,
    model: &'a str,
    usage: EmbeddingUsage,
}

/// One embedding entry. `embedding` is either a float array or a base64 string,
/// depending on the request's `encoding_format`.
#[derive(Serialize)]
struct EmbeddingData {
    object: &'static str,
    // Field order is the wire order (serde serialises structs in declaration
    // order): `object, index, embedding` matches the OpenAI embeddings schema
    // and the stormVINO reference.
    index: usize,
    embedding: EmbeddingVector,
}

/// The embedding payload: numeric (`float`) or `base64`-encoded f32 bytes.
/// `#[serde(untagged)]` serialises a `Float` as a JSON array and a `Base64` as
/// a JSON string — exactly the two `OpenAI` `encoding_format` wire shapes.
#[derive(Serialize)]
#[serde(untagged)]
enum EmbeddingVector {
    Float(Vec<f32>),
    Base64(String),
}

/// Embeddings `usage` — only `prompt_tokens` + `total_tokens` (no completion,
/// nothing is generated). `total == prompt` per the `OpenAI` contract.
#[derive(Serialize)]
struct EmbeddingUsage {
    prompt_tokens: usize,
    total_tokens: usize,
}

// ---- Handler --------------------------------------------------------

/// L0 length gate for one embedding input, mirroring
/// `handlers::chat::gate_prompt`/`gate_npu_prompt`: without this, a text that
/// tokenizes past the model's `max_position_embeddings` reaches
/// `TextEmbeddingPipeline::embed_documents` directly and raises an OV
/// shape-inference exception, surfaced as an opaque 500 (root-caused
/// 2026-08-03, the project's internal engineering log).
///
/// Skipped entirely when the model's `config.json` didn't declare
/// `max_position_embeddings` (`handle.max_seq_len() == None` — fail-open, same
/// policy as the NPU gate's "config.json absent/unparseable" case). A cheap
/// byte-length pre-check runs before tokenizing (same shape as
/// `gate_prompt_bytes`); a tokenizer-call error also fails open rather than
/// becoming a false 400.
// Response is the idiomatic "early-return a ready-built HTTP error" type used
// throughout this handler module; called once per request, not a hot loop.
#[allow(clippy::result_large_err)]
async fn gate_embed_text(
    handle: &EmbeddingHandle,
    text: &str,
    index: usize,
) -> Result<Option<usize>, Response> {
    let Some(max_seq_len) = handle.max_seq_len() else {
        return Ok(None);
    };
    if text.len() > max_seq_len.saturating_mul(MAX_BYTES_PER_TOKEN) {
        return Err(openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "input[{index}] is {} bytes — too large to fit this model's \
                 {max_seq_len}-token position-embedding capacity under even the \
                 most generous tokenizer compression; shorten the input",
                text.len(),
            ),
            "invalid_request_error",
            Some("context_length_exceeded"),
        ));
    }
    match handle.count_tokens(text.to_owned()).await {
        Ok(count) if count > max_seq_len => Err(openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "input[{index}] has {count} tokens — exceeds this model's \
                 maximum input length of {max_seq_len} tokens; shorten the input"
            ),
            "invalid_request_error",
            Some("context_length_exceeded"),
        )),
        Ok(count) => Ok(Some(count)),
        Err(_) => Ok(None),
    }
}

/// The batch's padded size: inputs × the longest input's token count — what
/// the pipeline actually allocates, since a batch pads to its longest input.
/// `None` when any count is unknown (gate skipped or tokenizer error), so the
/// budget fails open like the length gate does.
fn padded_batch_tokens(counts: &[Option<usize>]) -> Option<usize> {
    let longest = counts
        .iter()
        .copied()
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .max()?;
    Some(longest.saturating_mul(counts.len()))
}

/// Reject a batch whose padded size exceeds `max_embedding_batch_tokens`
/// (`0` = no budget). `None` when within budget or the size is unknown.
fn reject_over_token_budget(budget: usize, counts: &[Option<usize>]) -> Option<Response> {
    let padded = padded_batch_tokens(counts)?;
    (budget > 0 && padded > budget).then(|| {
        let longest = padded / counts.len().max(1);
        openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "input array needs {padded} padded tokens ({} inputs × the longest input's \
                 {longest} tokens) — this server accepts at most {budget} per request; split \
                 it into smaller batches",
                counts.len()
            ),
            "invalid_request_error",
            Some("batch_too_large"),
        )
    })
}

/// Reject a `dimensions` request explicitly rather than silently returning a
/// full-size vector the caller didn't ask for — this server has no
/// vector-truncation support. `None` when the request is clean. Closes the
/// same "A1 silent-drop" trap `chat::unsupported_chat_param` closes for
/// `n`/`logprobs`/`logit_bias`.
fn reject_dimensions(req: &EmbeddingRequest) -> Option<Response> {
    req.dimensions.is_some().then(|| {
        unsupported_param(
            "dimensions",
            "this server does not support truncating output vectors — omit \
             the parameter and use the model's native dimensionality",
        )
    })
}

/// Cap the input-array fan-out: the whole array runs as one pipeline batch,
/// so an unbounded array lets a single call monopolize the model and its
/// memory. Its own knob (`max_embedding_inputs`, default 256) — embeddings
/// used to share `/v1/completions`' `max_prompt_array` (16), which sized for
/// one full generation per element. `None` when within the cap.
fn reject_oversized_input_array(state: &AppState, n_inputs: usize) -> Option<Response> {
    let max_inputs = state.model_manager.as_ref().map_or_else(
        crate::model_manager::config::default_max_embedding_inputs,
        |mm| mm.max_embedding_inputs(),
    );
    (n_inputs > max_inputs).then(|| {
        openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "input array has {n_inputs} elements — this server accepts at most \
                 {max_inputs} inputs per request"
            ),
            "invalid_request_error",
            Some("too_many_inputs"),
        )
    })
}

/// Resolve `encoding_format` to "is base64?" — `None`/`"float"` → `false`,
/// `"base64"` → `true`, anything else → a 400 so an unknown value fails fast.
///
/// The `Err` half is boxed to dodge `clippy::result_large_err` (a `Response`
/// is large), same pattern as `completions::submit_prompts`.
fn resolve_encoding_format(encoding_format: Option<&str>) -> Result<bool, Box<Response>> {
    match encoding_format {
        None | Some("float") => Ok(false),
        Some("base64") => Ok(true),
        Some(other) => Err(Box::new(openai_error(
            StatusCode::BAD_REQUEST,
            format!("unsupported encoding_format '{other}' — use 'float' or 'base64'"),
            "invalid_request_error",
            Some("invalid_encoding_format"),
        ))),
    }
}

/// `POST /v1/embeddings` — `OpenAI`-compatible text embeddings.
///
/// Routes to a `ModelKind::Embedding` model and returns one vector per input.
/// A non-embedding model (text/vision) is rejected with 400 (`WrongKind`); an
/// unknown model with 404; a not-ready model with 503.
pub async fn embeddings(
    State(state): State<AppState>,
    JsonBody(req): JsonBody<EmbeddingRequest>,
) -> Response {
    if let Some(resp) = reject_dimensions(&req) {
        return resp;
    }

    let texts = req.input.into_vec();
    if texts.is_empty() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "input must not be empty",
            "invalid_request_error",
            Some("empty_input"),
        );
    }
    if let Some(resp) = reject_oversized_input_array(&state, texts.len()) {
        return resp;
    }

    let as_base64 = match resolve_encoding_format(req.encoding_format.as_deref()) {
        Ok(b) => b,
        Err(resp) => return *resp,
    };

    // Resolve vectors + token count + the model name to report.
    let (vectors, prompt_tokens, model_name) = if let Some(mm) = state.model_manager.as_ref() {
        let handle = match mm.get_embedding_handle(&req.model) {
            Ok(h) => h,
            // Slice 3c parity (mirrors the chat + media paths): lazy-load an
            // `on_demand` embedding model on first request — 503 Loading +
            // Retry-After while it loads — rather than failing flat with NotLoaded.
            Err(ModelError::NotLoaded) => {
                return match mm.request_on_demand_load(&req.model) {
                    OnDemandLoad::Loading => model_error_response(ModelError::Loading),
                    OnDemandLoad::NotApplicable => model_error_response(ModelError::NotLoaded),
                };
            }
            Err(e) => return model_error_response(e),
        };

        // L0 length gate — before the device-admission lease below, so an
        // over-length input is rejected without occupying an admission slot.
        let mut counts = Vec::with_capacity(texts.len());
        for (index, text) in texts.iter().enumerate() {
            match gate_embed_text(&handle, text, index).await {
                Ok(count) => counts.push(count),
                Err(resp) => return resp,
            }
        }
        // Batch working memory follows the padded size and stays allocated
        // until eviction — bound it per request (measured 2026-09-29).
        if let Some(resp) = reject_over_token_budget(mm.max_embedding_batch_tokens(), &counts) {
            return resp;
        }

        // Cross-pipeline device admission (step 2): a second gate in front of
        // the engine's own per-engine gate below, capping concurrency across
        // different engine kinds sharing this model's device.
        let device = mm.record_device(&req.model);
        let _device_lease = match state
            .device_budgets
            .admit(&[(device, 1)], crate::admission::WorkClass::Embed)
            .await
        {
            Ok(lease) => lease,
            Err(e) => return device_admission_error_response(&e),
        };

        match handle.embed(texts).await {
            Ok(out) => (out.vectors, out.prompt_tokens, handle.model_id().to_owned()),
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
                    "embedding engine unavailable (engine thread exited)",
                    "server_error",
                    Some("engine_unavailable"),
                );
            }
            // T6.2: never reflect raw C++ diagnostics into the body. An OOM-class
            // error becomes a retryable 503 and poisons the context; anything
            // else is an opaque 500. Raw error logged server-side.
            Err(crate::cb_engine::AdmitError::Failed(e)) => {
                return inference_error_response(Some(mm), handle.model_id(), &e);
            }
        }
    } else {
        // Mock path (no GPU): a fixed small vector per input so the route is
        // exercisable in integration tests without a model.
        let vectors: Vec<Vec<f32>> = texts.iter().map(|_| vec![0.0, 0.1, 0.2, 0.3]).collect();
        (vectors, texts.len(), req.model.clone())
    };

    let data = vectors
        .into_iter()
        .enumerate()
        .map(|(index, v)| EmbeddingData {
            object: "embedding",
            index,
            embedding: if as_base64 {
                EmbeddingVector::Base64(encode_base64(&v))
            } else {
                EmbeddingVector::Float(v)
            },
        })
        .collect();

    Json(EmbeddingResponse {
        object: "list",
        data,
        model: &model_name,
        usage: EmbeddingUsage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    })
    .into_response()
}

/// Encode a float vector as little-endian f32 bytes, base64 (the `OpenAI`
/// `encoding_format: "base64"` wire form the openai-python client decodes).
fn encode_base64(vec: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(vec.len() * 4);
    for f in vec {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    Base64::encode_string(&bytes)
}

// ============================================================
// Unit tests
// ============================================================
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// Padded size = inputs × longest input; unknown if any count is unknown.
    #[test]
    fn padded_batch_tokens_uses_longest_input() {
        assert_eq!(
            padded_batch_tokens(&[Some(10), Some(512), Some(3)]),
            Some(1536)
        );
        assert_eq!(padded_batch_tokens(&[Some(10), None]), None, "fail open");
        assert_eq!(padded_batch_tokens(&[]), None);
    }

    /// Over budget → 400 `batch_too_large`; at budget, unknown size, or a `0`
    /// (disabled) budget → accepted. 64 full chunks fit the 32768 default; 65 don't.
    #[test]
    fn token_budget_rejects_only_over_budget() {
        let full = |n: usize| vec![Some(512); n];
        assert!(reject_over_token_budget(32_768, &full(64)).is_none());
        let resp = reject_over_token_budget(32_768, &full(65)).unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert!(
            reject_over_token_budget(0, &full(1000)).is_none(),
            "0 = no budget"
        );
        assert!(reject_over_token_budget(32_768, &[Some(512), None]).is_none());
        // 256 short queries are cheap: 256 × 12 = 3072 padded tokens.
        assert!(reject_over_token_budget(32_768, &vec![Some(12); 256]).is_none());
    }

    // ── L0 length gate (the project's internal engineering log) ──

    /// A handle with no declared `max_position_embeddings` skips the gate
    /// entirely — no channel round-trip, never rejects.
    #[tokio::test]
    async fn gate_is_a_no_op_when_max_seq_len_unknown() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<crate::embed_engine::EmbedCommand>(4);
        let handle = EmbeddingHandle::from_sender(tx, "test-embed");
        assert!(
            gate_embed_text(&handle, &"word ".repeat(10_000), 0)
                .await
                .is_ok()
        );
    }

    /// A text that tokenizes within the ceiling passes.
    #[tokio::test]
    async fn gate_accepts_a_text_within_the_ceiling() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<crate::embed_engine::EmbedCommand>(4);
        let handle = EmbeddingHandle::from_sender_with_max_seq_len(tx, "test-embed", 512);

        tokio::spawn(async move {
            if let Some(crate::embed_engine::EmbedCommand::CountTokens { reply, .. }) =
                rx.recv().await
            {
                let _ = reply.send(Ok(100));
            }
        });

        assert!(gate_embed_text(&handle, "short text", 0).await.is_ok());
    }

    /// A text whose token count exceeds the ceiling is rejected with a clean
    /// 400 `context_length_exceeded`, never reaching the pipeline.
    #[tokio::test]
    async fn gate_rejects_a_text_over_the_ceiling_with_400() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<crate::embed_engine::EmbedCommand>(4);
        let handle = EmbeddingHandle::from_sender_with_max_seq_len(tx, "test-embed", 512);

        tokio::spawn(async move {
            if let Some(crate::embed_engine::EmbedCommand::CountTokens { reply, .. }) =
                rx.recv().await
            {
                let _ = reply.send(Ok(515));
            }
        });

        let err = gate_embed_text(&handle, "a text that tokenizes long", 2)
            .await
            .expect_err("must reject an over-length input");
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    }

    /// A text long enough to trip the byte pre-check is rejected without ever
    /// touching the tokenizer channel (an unattended channel would otherwise
    /// hang the gate forever).
    #[tokio::test]
    async fn gate_rejects_on_byte_pre_check_without_a_tokenize_round_trip() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<crate::embed_engine::EmbedCommand>(4);
        let handle = EmbeddingHandle::from_sender_with_max_seq_len(tx, "test-embed", 512);

        let huge = "x".repeat(512 * MAX_BYTES_PER_TOKEN + 1);
        let err = gate_embed_text(&handle, &huge, 0)
            .await
            .expect_err("must reject on the byte pre-check");
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    }

    /// A tokenizer-call error (engine channel closed) fails open — a transient
    /// hiccup must not become a false 400.
    #[tokio::test]
    async fn gate_fails_open_when_tokenize_errors() {
        let (tx, rx) = tokio::sync::mpsc::channel::<crate::embed_engine::EmbedCommand>(4);
        drop(rx);
        let handle = EmbeddingHandle::from_sender_with_max_seq_len(tx, "test-embed", 512);
        assert!(gate_embed_text(&handle, "short text", 0).await.is_ok());
    }

    #[test]
    fn input_deserialises_single_and_array() {
        let single: EmbeddingRequest =
            serde_json::from_str(r#"{"model":"m","input":"hi"}"#).unwrap();
        assert_eq!(single.input.into_vec(), vec!["hi".to_owned()]);

        let multi: EmbeddingRequest =
            serde_json::from_str(r#"{"model":"m","input":["a","b"]}"#).unwrap();
        assert_eq!(multi.input.into_vec(), vec!["a".to_owned(), "b".to_owned()]);
    }

    #[test]
    fn absent_input_is_a_deserialisation_error() {
        let r: Result<EmbeddingRequest, _> = serde_json::from_str(r#"{"model":"m"}"#);
        assert!(r.is_err(), "input is required");
    }

    #[test]
    fn unknown_params_ignored() {
        let r: EmbeddingRequest =
            serde_json::from_str(r#"{"model":"m","input":"x","dimensions":256,"user":"u"}"#)
                .unwrap();
        assert_eq!(r.model, "m");
    }

    /// base64 of a known vector round-trips back to the same little-endian f32s.
    #[test]
    fn base64_encodes_little_endian_f32() {
        let v = vec![1.0_f32, -2.5, 0.0];
        let b64 = encode_base64(&v);
        let bytes = Base64::decode_vec(&b64).unwrap();
        assert_eq!(bytes.len(), 12, "3 floats × 4 bytes");
        let decoded: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(decoded, v);
    }
}
