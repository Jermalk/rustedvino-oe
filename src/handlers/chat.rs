// ============================================================
// src/handlers/chat.rs — POST /v1/chat/completions
// ============================================================
// Phase 1b: streaming (stream:true) via ContinuousBatchingPipeline.
// Phase 3:  non-streaming (stream:false) — collect_completion() drains
//           the same engine channel into a single chat.completion JSON.
//           Tool calling, sampling params, <think> strip, include_usage
//           added in Phases 3.2–3.8 / A1–A4.
// Mock token generator used as fallback when no ModelManager (tests).
//
// CRASH COURSE — axum handler signature rules:
//   Handlers are async functions where every argument is an
//   "extractor" — a type that implements FromRequest or
//   FromRequestParts. axum calls them in order left-to-right.
//
//   State(s): State<T>  → clones T from router state
//   Json(body): Json<T> → deserialises request body as T
//
//   The return type must implement IntoResponse. Using
//   `-> axum::response::Response` (the erased type) lets us
//   return different response shapes from the same function.
// ============================================================

use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{
        Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use futures_util::StreamExt as _;
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;

use std::sync::Arc;

use super::error::{
    JsonBody, device_admission_error_response, inference_error_from_text, model_error_response,
    openai_error, sse_inference_error_event, unsupported_param,
};
use crate::{
    app_state::AppState,
    cb_engine::{EngineHandle, SubmitResult},
    image_util::DecodedImage,
    model_manager::{
        EngineHandleKind, ModelError, ModelManager, OnDemandLoad, ReasoningParser,
        template::GenerationDefaults,
    },
    ov_cb::{FinishReason, GenParams},
    prompt_builder::{
        HarmonyFilter, ModelFamily, ThinkFilter, ThinkPiece, ToolCall, extract_reasoning_gpt_oss,
        extract_thinking, parse_tool_calls, prompt_ends_in_thinking, template_prefills_thinking,
    },
    streaming::{StreamEvent, TokenReceiver, stream_channel},
};

/// Generous upper bound on how many bytes a single tokenizer-vocab entry can
/// realistically span (the longest merged BPE/SentencePiece tokens — repeated
/// whitespace/newline runs, long dictionary words — top out well under this
/// across every tokenizer this server has loaded). Used only to size the
/// byte-length pre-check shared by [`gate_prompt`] and [`gate_vlm_prompt`]:
/// deliberately generous so it can never reject a prompt that would have
/// actually passed the real tokenizer-based check below.
const MAX_BYTES_PER_TOKEN: usize = 32;

/// The 400 `context_length_exceeded` envelope for an exact, tokenizer-reported
/// overflow — shared by [`gate_prompt`], [`gate_vlm_prompt`], and
/// [`gate_npu_prompt`] so all three gates produce byte-identical error shapes.
///
/// A clean L0-gate rejection is fully visible to the client (the 400 body
/// cites exact numbers) but was previously invisible server-side — no log
/// line, no metric, nothing an operator watching `/metrics` or the log file
/// could see (found live watching a real production run hit this exact path).
/// `gate`
/// (`"cb"`/`"vlm"`/`"npu"`) distinguishes which pipeline rejected, since NPU
/// shares `ModelKind::TextGen`'s label with the plain CB path.
fn prompt_too_long_error(
    model_id: &str,
    gate: &'static str,
    token_count: usize,
    max_prompt_tokens: usize,
) -> Response {
    tracing::warn!(
        model_id,
        gate,
        token_count,
        max_prompt_tokens,
        "L0 prompt-length gate rejected request — context_length_exceeded"
    );
    crate::metrics::record_context_length_exceeded(model_id, gate);
    openai_error(
        StatusCode::BAD_REQUEST,
        format!(
            "prompt is {token_count} tokens — exceeds this model's KV-pool capacity \
             of {max_prompt_tokens} tokens; shorten the prompt or increase cache_size_gb",
        ),
        "invalid_request_error",
        Some("context_length_exceeded"),
    )
}

/// Byte-length pre-check shared by [`gate_prompt`] and [`gate_vlm_prompt`],
/// run before either ever calls into a tokenizer.
///
/// Session 2026-07-02: a long-enough raw prompt exhausted host RAM *inside*
/// the tokenize call itself, before the exact token-count check downstream
/// ever got a chance to reject it — the resulting OOM took down the whole
/// desktop session, not just the request (the project's internal engineering log).
/// `prompt.len()` (bytes) only bounds the true token count from below when
/// it's small; the useful direction here is the reverse — `MAX_BYTES_PER_TOKEN`
/// bounds how few tokens a given byte length could possibly produce, so
/// crossing `max_prompt_tokens * MAX_BYTES_PER_TOKEN` bytes guarantees the
/// prompt is over budget regardless of how compressible its content is, with
/// no tokenizer call required to know that.
///
/// # Errors
/// The over-limit 400 response, boxed, when the byte ceiling is exceeded.
fn gate_prompt_bytes(
    model_id: &str,
    gate: &'static str,
    prompt: &str,
    max_prompt_tokens: usize,
) -> Result<(), Box<Response>> {
    if prompt.len() > max_prompt_tokens.saturating_mul(MAX_BYTES_PER_TOKEN) {
        tracing::warn!(
            model_id,
            gate,
            prompt_bytes = prompt.len(),
            max_prompt_tokens,
            "L0 prompt-length gate rejected request (byte pre-check) — \
             context_length_exceeded"
        );
        crate::metrics::record_context_length_exceeded(model_id, gate);
        return Err(Box::new(openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "prompt is {} bytes — too large to fit this model's KV-pool capacity \
                 of {max_prompt_tokens} tokens under even the most generous tokenizer \
                 compression; shorten the prompt or increase cache_size_gb",
                prompt.len(),
            ),
            "invalid_request_error",
            Some("context_length_exceeded"),
        )));
    }
    Ok(())
}

/// The `429` response for a concurrent-KV-admission rejection — shared by
/// both the cheap byte-length pre-check and the exact tokenized check in the
/// chat handler, so the two call sites can't drift into differently-worded
/// errors for the same failure mode. `token_estimate` is exact when the
/// caller has real token ids, or the conservative `min_possible_tokens`
/// lower bound when rejecting off the pre-check alone.
fn pool_capacity_exceeded_response(model_id: &str, token_estimate: usize) -> Response {
    openai_error(
        StatusCode::TOO_MANY_REQUESTS,
        format!(
            "model '{model_id}' is at KV-pool capacity with other in-flight requests \
             — this prompt (~{token_estimate} tokens) can't be admitted right now, retry shortly",
        ),
        "rate_limit_error",
        Some("pool_capacity_exceeded"),
    )
}

/// Same shape as [`pool_capacity_exceeded_response`], but a different cause:
/// a single-slot model (`max_concurrent_streams == 1`) rejecting a request
/// whose own `prompt_tokens + max_tokens` can't fit the pool, with no other
/// in-flight request involved (the project's internal engineering log's deferred
/// "fix 3"). Deliberately a distinct message: "retry shortly" is actively
/// bad advice here — the identical request will fail identically on retry,
/// since nothing about the pool's occupancy changes. `total_estimate`
/// already includes both the prompt and the requested `max_tokens`.
fn own_budget_exceeded_response(
    model_id: &str,
    total_estimate: usize,
    pool_capacity_tokens: usize,
) -> Response {
    openai_error(
        StatusCode::TOO_MANY_REQUESTS,
        format!(
            "model '{model_id}' allows only one request at a time, and this request's prompt \
             plus requested max_tokens (~{total_estimate} tokens total) exceeds its KV-pool \
             capacity (~{pool_capacity_tokens} tokens) — reduce max_tokens and retry",
        ),
        "rate_limit_error",
        Some("own_budget_exceeded"),
    )
}

/// L0 prompt-length gate, shared by `/v1/chat/completions` and the legacy
/// `/v1/completions` (T2.1 gate parity — one gate function, no second copy to
/// drift).
///
/// A cheap byte-length pre-check ([`gate_prompt_bytes`]) runs first, before
/// any tokenizer is touched. If that passes, the gate tokenizes the final
/// prompt on the engine thread (slot-free, same path as `POST /tokenize`) and
/// rejects it with 400 `context_length_exceeded` **before** `add_request` — a
/// prompt larger than the KV pool causes `engine.step()` to block forever,
/// requiring a process restart (Session 28).
///
/// The gate is skipped when `max_prompt_tokens == 0` (model `config.json`
/// absent/unparseable) and fails open when the tokenize round-trip itself
/// errors — a transient tokenizer hiccup must not become a false 400.
///
/// On success returns the gate's token ids so the caller can feed them straight
/// into `add_request` (T7.3 — encode once, not twice): `Some(ids)` when the
/// prompt was tokenized and is within the limit; `None` when the gate was
/// disabled or its tokenize failed open (the engine then tokenizes the string).
///
/// # Errors
/// The over-limit 400 response, boxed (large `Response` in `Err` position).
pub(crate) async fn gate_prompt(
    handle: &EngineHandle,
    model_id: &str,
    prompt: &str,
    max_prompt_tokens: usize,
) -> Result<Option<Vec<i64>>, Box<Response>> {
    if max_prompt_tokens == 0 {
        return Ok(None);
    }
    gate_prompt_bytes(model_id, "cb", prompt, max_prompt_tokens)?;
    match handle.tokenize(prompt.to_owned()).await {
        Ok(ids) if ids.len() > max_prompt_tokens => Err(Box::new(prompt_too_long_error(
            model_id,
            "cb",
            ids.len(),
            max_prompt_tokens,
        ))),
        Ok(ids) => Ok(Some(ids)), // within limit — reuse for add_request
        Err(_) => Ok(None),       // tokenize error — fail open, engine re-encodes
    }
}

/// Build the text [`gate_vlm_prompt`] measures from `vlm_messages` (the JSON
/// objects [`extract_vlm_messages`] produces).
///
/// Must include `tool_calls`, not just `content`: an assistant turn that
/// only made a tool call has empty `content` and all its bulk (function
/// name + JSON arguments) in `tool_calls` — the exact shape
/// `extract_vlm_messages` preserves so the model can see its own prior
/// calls. Counting `content` alone undercounts a tool-heavy conversation by
/// however much of its history is tool-call JSON, letting the gate admit a
/// prompt it should have rejected (confirmed live 2026-08-23,
/// the project's internal engineering log: a rejection
/// cited 262,298 gate-measured tokens for a conversation the engine had
/// reported as 293,471 tokens one turn earlier — an ~11% undercount).
fn vlm_gate_text(vlm_messages: &[serde_json::Value]) -> String {
    vlm_messages
        .iter()
        .filter_map(serde_json::Value::as_object)
        .flat_map(|m| {
            let content = m.get("content").and_then(|c| c.as_str()).map(str::to_owned);
            let tool_calls = m.get("tool_calls").map(serde_json::Value::to_string);
            [content, tool_calls].into_iter().flatten()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// L0 prompt-length gate for the VLM path.
///
/// Mirrors [`gate_prompt`], including its byte-length pre-check
/// ([`gate_prompt_bytes`]) — but `VLMPipeline::generate` always takes raw
/// messages/images (never pre-tokenized ids the way the CB engine can), so
/// this only needs a token *count*, not the ids themselves. `gate_text`
/// ([`vlm_gate_text`]) is the concatenated text content (and tool-call JSON)
/// of the request's messages (see `extract_vlm_messages`) — an approximation
/// of what `VLMPipeline` will actually template-render and tokenize
/// internally, close enough to catch a grossly oversized prompt before it
/// reaches the GPU.
///
/// Without this gate, an oversized prompt reached `VLMPipeline::generate`
/// directly and could exhaust GPU memory (`CL_OUT_OF_RESOURCES`), which
/// poisons the `OpenCL` context for the **whole process** — every GPU-backed
/// model, not just this request — until a manual restart (confirmed live: a
/// ~75k-token prompt against a 49,932-token limit took the server down this
/// way). The CB path already guards against the ids-only version of this
/// failure (`gate_prompt`'s doc comment: "a prompt larger than the KV pool
/// causes `engine.step()` to block forever, requiring a process restart") —
/// the VLM path just never got the same treatment when it was added.
///
/// Same fail-open policy as `gate_prompt`: disabled when `max_prompt_tokens
/// == 0`, and a tokenizer-call error does not become a false 400 (better to
/// let a rare tokenizer hiccup through than reject a legitimate request).
///
/// # Errors
/// The over-limit 400 response, boxed (large `Response` in `Err` position).
pub(crate) async fn gate_vlm_prompt(
    vlm: &crate::vlm_engine::VlmHandle,
    gate_text: &str,
    max_prompt_tokens: usize,
) -> Result<(), Box<Response>> {
    if max_prompt_tokens == 0 {
        return Ok(());
    }
    gate_prompt_bytes(vlm.model_id(), "vlm", gate_text, max_prompt_tokens)?;
    match vlm.count_tokens(gate_text.to_owned()).await {
        Ok(count) if count > max_prompt_tokens => Err(Box::new(prompt_too_long_error(
            vlm.model_id(),
            "vlm",
            count,
            max_prompt_tokens,
        ))),
        Ok(_) | Err(_) => Ok(()),
    }
}

/// `OpenVINO` `GenAI`'s own default `MAX_PROMPT_LEN` for the NPU static
/// `LLMPipeline` when no per-model override is configured (the project's internal engineering log
/// #6). Not discovered dynamically — there is no query API for it — so this
/// constant must be updated if `OpenVINO` ever changes its own default.
pub(crate) const NPU_DEFAULT_MAX_PROMPT_LEN: usize = 1024;

/// L0 prompt-length gate for the NPU text-gen path.
///
/// The NPU compiles a **fixed-shape** graph ahead of time (unlike GPU/CPU's
/// dynamic-shape kernels), so its context ceiling is a hard compile-time
/// limit, not a soft KV-pool budget — but until this gate, nothing checked it
/// before calling `generate()`: a too-long prompt surfaced as `OpenVINO`'s raw
/// C++ exception text instead of a clean 400 (found 2026-07-16 via an
/// Ollama-comparison benchmark; the project's internal engineering log #6).
///
/// Mirrors [`gate_vlm_prompt`] exactly (count-only — `NpuHandle::generate`
/// takes a plain string prompt, never pre-tokenized ids, so there is nothing
/// to reuse downstream the way `gate_prompt` reuses CB ids). `max_prompt_len`
/// is the resolved effective ceiling: the model's configured
/// `max_prompt_len` override, or [`NPU_DEFAULT_MAX_PROMPT_LEN`] when unset.
///
/// Same fail-open policy as `gate_prompt`/`gate_vlm_prompt`: a tokenizer-call
/// error does not become a false 400.
///
/// # Errors
/// The over-limit 400 response, boxed (large `Response` in `Err` position).
pub(crate) async fn gate_npu_prompt(
    npu: &crate::npu_engine::NpuHandle,
    prompt: &str,
    max_prompt_len: usize,
) -> Result<(), Box<Response>> {
    gate_prompt_bytes(npu.model_id(), "npu", prompt, max_prompt_len)?;
    match npu.count_tokens(prompt.to_owned()).await {
        Ok(count) if count > max_prompt_len => Err(Box::new(prompt_too_long_error(
            npu.model_id(),
            "npu",
            count,
            max_prompt_len,
        ))),
        Ok(_) | Err(_) => Ok(()),
    }
}

// ---- Request types --------------------------------------------------
//
// CRASH COURSE — #[derive(Deserialize)]:
//   serde automatically implements JSON → struct conversion.
//   Unknown fields in the JSON are silently ignored (serde default).
//   Missing fields with no default → deserialization error (400).
//   Fields with `Option<T>` default to None if absent.

/// Incoming POST /v1/chat/completions body (`OpenAI`-compatible).
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ChatRequest {
    /// Which model to use. Passed through to the response.
    pub model: String,

    /// Conversation history. Last message is the current user turn.
    pub messages: Vec<ChatMessage>,

    /// If true, respond with an SSE token stream; if false or absent,
    /// respond with a single `chat.completion` JSON object (Phase 3).
    /// Defaults to `false`, matching the `OpenAI` spec.
    pub stream: Option<bool>,

    /// Cap on generated tokens (`OpenAI` legacy name). Takes precedence over
    /// `max_completion_tokens` when both are present.
    pub max_tokens: Option<u32>,

    /// Cap on generated tokens (`OpenAI` v1 name). Used when `max_tokens` is absent.
    pub max_completion_tokens: Option<u32>,

    /// Sampling temperature. `0` = greedy; `> 0` = multinomial. `OpenAI` default: 1.0.
    pub temperature: Option<f32>,

    /// Nucleus sampling. Keep only the smallest set of tokens whose cumulative
    /// probability ≥ `top_p`. `OpenAI` default: 1.0 (no filtering).
    pub top_p: Option<f32>,

    /// Presence penalty. Positive values penalise tokens that have appeared at
    /// all, encouraging the model to talk about new topics. `OpenAI` range: −2..2.
    pub presence_penalty: Option<f32>,

    /// Frequency penalty. Positive values penalise tokens in proportion to how
    /// often they have already appeared. `OpenAI` range: −2..2.
    pub frequency_penalty: Option<f32>,

    /// Repetition penalty (not a standard `OpenAI` field — accepted as a named
    /// extension, the same convention other `OpenAI`-compatible servers use,
    /// e.g. Ollama/`llama.cpp`). Scales a token's logit by this factor every
    /// time it has already appeared; `> 1.0` discourages repeats, `1.0` = off.
    /// Absent → the server applies [`DEFAULT_REPETITION_PENALTY`] as a safety
    /// net (see [`resolve_repetition_penalty`]) — send `1.0` explicitly to
    /// opt out.
    #[serde(default)]
    pub repetition_penalty: Option<f32>,

    /// Top-k sampling (extension field, not standard `OpenAI` — the Ollama /
    /// `llama.cpp` convention): restrict sampling to the `k` highest-probability
    /// tokens. Absent or `0` → disabled (engine keeps its default, effectively
    /// unlimited). Previously this field did not exist and was
    /// silently dropped by serde on every path.
    #[serde(default)]
    pub top_k: Option<usize>,

    /// Deterministic seed. When set, the model uses the given seed for random
    /// sampling, making responses reproducible.
    pub seed: Option<u64>,

    /// Stop sequence(s). Generation halts (and the stop string is NOT included
    /// in the output) when any stop string appears. Accepts a single string or an
    /// array of up to four strings.
    pub stop: Option<serde_json::Value>,

    /// `OpenAI` tool/function schemas. When present, they are injected into the
    /// prompt via the model's chat template and the response is buffered so any
    /// `<tool_call>` blocks can be parsed (Phase 3 tool calling). Opaque JSON —
    /// passed straight to the template's `tools` variable.
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,

    /// Controls SSE stream extensions. Only meaningful when `stream: true`.
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,

    /// `OpenAI` `response_format` — constrains output to JSON (G1). `text`
    /// (default) leaves output free-form; `json_object` forces any valid JSON
    /// object; `json_schema` forces output matching the supplied JSON Schema.
    /// Enforced engine-side via `OpenVINO` `GenAI`'s structured-output (xgrammar)
    /// backend. Applies to chat only (not legacy `/v1/completions`).
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,

    /// `OpenAI` `tool_choice` — controls whether/which tool the model may call (G2).
    /// Opaque JSON: the string `"none"` | `"auto"` | `"required"`, or an object
    /// `{"type":"function","function":{"name":"…"}}` to force a specific function.
    /// `none` suppresses tool injection (plain chat even when `tools` are present);
    /// `auto` (the default) lets the model decide; `required` / a named function are
    /// enforced best-effort by appending a steering instruction to the prompt.
    /// Applies to chat only and only on the text path (where tool calling lives).
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,

    /// Whether the model should emit a `<think>` reasoning block before answering.
    /// `false` injects `<think>\n\n</think>` into the generation prompt so the model
    /// skips thinking entirely. `true` or absent lets the model decide (default for
    /// Qwen3 hybrid-thinking models). Passed as `enable_thinking` to the Jinja
    /// chat template.
    #[serde(default)]
    pub enable_thinking: Option<bool>,

    // ---- Known-but-unsupported params (rejected with 400, not silently dropped).
    // These are parsed only so the handler can detect their presence and return an
    // explicit `unsupported_parameter` error — see `unsupported_chat_param`. The
    // engine cannot honor any of them today. Cosmetic params the server safely
    // ignores (parallel_tool_calls, legacy functions/function_call, user, metadata,
    // store, service_tier, image `detail`) are intentionally NOT listed: serde
    // drops unknown fields, which is the correct accept-and-ignore behavior.
    /// Number of choices to generate. The CB engine serves a single sequence per
    /// request, so only `n: 1` (or absent) is honored; `n > 1` → 400.
    #[serde(default)]
    pub n: Option<u32>,

    /// Whether to return token log-probabilities. The bridge does not expose
    /// logits, so `logprobs: true` → 400 (`false`/absent is the no-op default).
    #[serde(default)]
    pub logprobs: Option<bool>,

    /// Number of top alternatives per token. Implies `logprobs`; unsupported → 400.
    #[serde(default)]
    pub top_logprobs: Option<u32>,

    /// Per-token logit bias map. No bias hook in the bridge; a non-empty map → 400.
    #[serde(default)]
    pub logit_bias: Option<serde_json::Value>,
}

/// `OpenAI` `response_format` request field (G1 structured outputs).
///
/// Three shapes are accepted:
/// - `{"type":"text"}` (or absent) — no constraint.
/// - `{"type":"json_object"}` — output must be a valid JSON object.
/// - `{"type":"json_schema","json_schema":{"name":…,"schema":{…}}}` — output
///   must conform to the embedded JSON Schema (`json_schema.schema`).
#[derive(Debug, Deserialize)]
pub struct ResponseFormat {
    /// `text` | `json_object` | `json_schema`. Unknown values are rejected 400.
    #[serde(rename = "type")]
    pub r#type: String,

    /// Present only when `type == "json_schema"`. The `OpenAI` wrapper object;
    /// the actual schema lives under its `schema` key.
    #[serde(default)]
    pub json_schema: Option<serde_json::Value>,
}

/// Controls optional SSE stream extensions (`stream_options` request field).
#[derive(Debug, Deserialize, Default)]
pub struct StreamOptions {
    /// When `true`, a final usage chunk is emitted before `[DONE]`.
    #[serde(default)]
    pub include_usage: bool,
}

/// One turn in the conversation (`OpenAI`-compatible).
///
/// `content` is optional because an assistant turn that *calls* a tool carries
/// `tool_calls` with `content: null`, and is followed by `tool`-role result
/// Message content — either a plain string (text-only) or an array of typed
/// `parts` (`OpenAI` multimodal format).  `#[serde(untagged)]` means serde matches
/// on shape: a JSON string → `Text`, a JSON array → `Parts`.
/// `Text("hi")` serialises back as `"hi"` — identical to the pre-VLM wire format.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// One element of a multimodal content array.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrlData },
}

/// The `image_url` field inside an `image_url` content part.
/// `url` is either `data:<mime>;base64,<b64>` or an HTTP URL.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageUrlData {
    pub url: String,
}

/// turns. All tool fields default to absent so plain chat requests are unchanged.
#[derive(Debug, Default, Deserialize, Serialize)]
pub struct ChatMessage {
    /// `"system"`, `"user"`, `"assistant"`, or `"tool"`.
    pub role: String,

    /// The message text (or multimodal parts). Absent on assistant turns that
    /// only carry `tool_calls`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,

    /// Tool calls the assistant chose to make (echoed back on the next turn).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,

    /// On a `tool`-role turn: the id of the call this message answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    /// Optional tool/function name (some clients set this on `tool` turns).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Return true if any message contains at least one `image_url` content part.
/// Used to route to the VLM path vs the CB text path.
pub fn has_images(messages: &[ChatMessage]) -> bool {
    messages.iter().any(|m| {
        matches!(&m.content, Some(MessageContent::Parts(parts))
            if parts.iter().any(|p| matches!(p, ContentPart::ImageUrl { .. })))
    })
}

/// Extract the text string from a `MessageContent`, concatenating text parts
/// and dropping image parts.  Used when passing messages to the CB engine.
pub fn extract_text_content(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(s) => s.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.as_str()),
                ContentPart::ImageUrl { .. } => None,
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}

/// Convert messages to plain `serde_json::Value` objects with string content,
/// suitable for the CB engine's chat-template renderer.  Image parts are
/// dropped; text parts are concatenated.
pub fn messages_to_template_values(messages: &[ChatMessage]) -> Vec<serde_json::Value> {
    messages
        .iter()
        .map(|m| {
            let mut obj = serde_json::json!({ "role": m.role });
            if let Some(c) = &m.content {
                obj["content"] = serde_json::Value::String(extract_text_content(c));
            }
            if let Some(tc) = &m.tool_calls {
                obj["tool_calls"] = serde_json::json!(tc);
            }
            if let Some(tcid) = &m.tool_call_id {
                obj["tool_call_id"] = serde_json::Value::String(tcid.clone());
            }
            if let Some(name) = &m.name {
                obj["name"] = serde_json::Value::String(name.clone());
            }
            obj
        })
        .collect()
}

// ---- Response types (SSE chunks) ------------------------------------
//
// CRASH COURSE — #[serde(skip_serializing_if = "Option::is_none")]:
//   By default, serde serialises Option::None as JSON `null`.
//   The OpenAI spec omits null fields entirely.
//   This attribute makes serde skip the field when it is None.
//   Result: `{"content":"Hello"}` not `{"content":"Hello","role":null}`.

/// One SSE chunk in the `OpenAI` streaming chat format.
#[derive(Serialize)]
struct ChatChunk<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: Vec<ChunkChoice>,
    /// Non-standard `OpenAI` extension, same spirit as `generation_metadata`
    /// on the image-gen endpoints (`PLAN_image_metadata_response.md`): the
    /// tool-call dialect detected from this model's chat template at load
    /// time ([`ModelFamily`]) — useful for debugging why tool-call parsing
    /// did or didn't happen for a given model. Always present (`Default` is a
    /// real, non-fabricated answer — not a sentinel for "unknown").
    model_family: &'static str,
    /// The sampling parameters actually used for this request, after
    /// [`resolve_sampling_defaults`] composes client-explicit values against
    /// `tools_active` and the model's own `generation_config.json` defaults —
    /// distinct from whatever the client sent, which may have specified none
    /// of these. Debugging aid: "why did this completion sample the way it
    /// did."
    sampling: EffectiveSampling,
}

/// See [`ChatChunk::sampling`] / [`CompletionChoice`]'s sibling on
/// [`ChatCompletion`]. Each field individually omitted (never `null`) when
/// that parameter fell through to the engine's own built-in default rather
/// than being resolved to a concrete value.
#[derive(Serialize)]
struct EffectiveSampling {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<usize>,
}

/// Bundles [`ChatChunk::model_family`]/[`ChatChunk::sampling`]'s source facts
/// so every response-chunk/completion builder takes one extra `Copy` argument
/// instead of four. Resolved once per request, before any response is built —
/// `temperature`/`top_p`/`top_k` come straight off the already-resolved
/// [`GenParams`](crate::ov_cb::GenParams) (`resolve_sampling_defaults` already
/// composed client-vs-model-default precedence; no separate resolution here).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResponseMeta {
    pub(crate) family: ModelFamily,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) top_k: Option<usize>,
}

impl ResponseMeta {
    fn sampling(self) -> EffectiveSampling {
        EffectiveSampling {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
        }
    }
}

#[derive(Serialize)]
struct ChunkChoice {
    index: u32,
    delta: Delta,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize, Default)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    /// Streaming `<think>` reasoning fragment (DeepSeek-style extension).
    /// Omitted on normal tokens so plain models are unaffected.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<StreamToolCall>>,
}

/// A tool call as it appears in a **streaming** `delta.tool_calls` array.
///
/// `OpenAI` requires an `index` field on every streaming tool-call delta so the
/// client can merge fragments by position; the non-streaming message shape
/// ([`ToolCall`]) omits it. We emit each call whole in a single chunk, so
/// `index` is simply its position in the array. `#[serde(flatten)]` splices the
/// `ToolCall` fields (`id`/`type`/`function`) up alongside `index`.
#[derive(Serialize)]
struct StreamToolCall {
    index: u32,
    #[serde(flatten)]
    call: ToolCall,
}

// ---- Response types (non-streaming chat.completion) -----------------
//
// The non-streaming reply is a single JSON object, not an SSE stream.
// Schema ported from stormVINO `chat_handler.py` (the behavioural spec):
//   {id, object:"chat.completion", created, model,
//    choices:[{index, message:{role, content}, finish_reason}],
//    usage:{prompt_tokens, completion_tokens, total_tokens}}

/// A full non-streaming chat completion (`OpenAI`-compatible).
#[derive(Serialize)]
struct ChatCompletion<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: Vec<CompletionChoice>,
    usage: Usage,
    /// See [`ChatChunk::model_family`].
    model_family: &'static str,
    /// See [`ChatChunk::sampling`].
    sampling: EffectiveSampling,
}

/// One choice in a non-streaming completion. Carries the *whole* message,
/// not a delta.
#[derive(Serialize)]
struct CompletionChoice {
    index: u32,
    message: ResponseMessage,
    finish_reason: &'static str,
}

/// The assembled assistant message (full content, not a delta).
///
/// `content` is `Option` and **not** skipped when `None`, so it serialises as
/// an explicit `"content": null` — the `OpenAI` shape for a tool-call message.
/// A normal completion sets `content: Some(text)` and omits `tool_calls`.
#[derive(Serialize)]
struct ResponseMessage {
    role: &'static str,
    content: Option<String>,
    /// The model's `<think>…</think>` reasoning, separated out of `content`
    /// (DeepSeek-style `reasoning_content`). Non-standard `OpenAI` — omitted
    /// when absent so plain models are unaffected. Non-streaming only; the
    /// streaming path does not yet split reasoning (needs a stateful buffer).
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
}

/// Token accounting (`OpenAI` `usage` block). `prompt_tokens` is the engine's
/// exact tokenizer count (Phase 3.6); `completion_tokens` is the count of
/// non-empty per-step deltas, which ≈ generated tokens. See DECISIONS.md
/// 2026-05-29. Shared by the chat and legacy-completions endpoints.
// Field names are fixed by the OpenAI wire format; the shared `_tokens`
// suffix is required, so the struct-field-names lint does not apply here.
#[derive(Serialize)]
#[allow(clippy::struct_field_names)]
pub(crate) struct Usage {
    pub(crate) prompt_tokens: usize,
    pub(crate) completion_tokens: usize,
    pub(crate) total_tokens: usize,
}

// ---- SSE event construction -----------------------------------------

/// Returns Unix timestamp in seconds. Used as the `created` field.
pub(crate) fn unix_now() -> u64 {
    // unwrap_or_default: only fails if system clock is before 1970.
    // On a running Linux box this is physically impossible, but the
    // type system can't prove it, so we handle it gracefully.
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Returns a strictly-increasing, process-unique sequence number for response
/// object ids.
///
/// `created` (whole-second Unix time, [`unix_now`]) is the only entropy the id
/// previously carried, so two requests landing in the same second produced the
/// *same* `chat.completion`/`text_completion` id. That violates the `OpenAI`
/// contract — clients dedupe, log, and trace on the id — and silently corrupts
/// any client keyed on it under concurrency (the exact load this server is
/// built for). Appending this counter guarantees uniqueness without a `uuid`
/// dependency or any clock-resolution assumption. Shared by both the chat and
/// legacy-completions handlers. (#49 / T7.6)
pub(crate) fn next_response_seq() -> u64 {
    static RESPONSE_SEQ: AtomicU64 = AtomicU64::new(0);
    RESPONSE_SEQ.fetch_add(1, Ordering::Relaxed)
}

/// Builds the first SSE frame: an empty delta that announces the role.
///
/// The `OpenAI` spec requires the first chunk to carry `role: "assistant"`
/// so clients know who is speaking before any content arrives.
fn role_event(id: &str, model: &str, created: u64, meta: ResponseMeta) -> Event {
    let chunk = ChatChunk {
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: Some("assistant"),
                ..Delta::default()
            },
            finish_reason: None,
        }],
        model_family: meta.family.label(),
        sampling: meta.sampling(),
    };
    // CRASH COURSE — serde_json::to_string:
    //   Serialises any Serialize value to a JSON string.
    //   Returns Result<String, Error>. We use unwrap_or_default()
    //   so a serialisation failure produces an empty string (the
    //   SSE frame is dropped) rather than panicking.
    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
}

/// Builds a token SSE frame.
fn token_event(token: &str, id: &str, model: &str, created: u64, meta: ResponseMeta) -> Event {
    let chunk = ChatChunk {
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                content: Some(token.to_owned()),
                ..Delta::default()
            },
            finish_reason: None,
        }],
        model_family: meta.family.label(),
        sampling: meta.sampling(),
    };
    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
}

/// Builds a reasoning SSE frame (`delta.reasoning_content`).
fn reasoning_event(token: &str, id: &str, model: &str, created: u64, meta: ResponseMeta) -> Event {
    let chunk = ChatChunk {
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                reasoning_content: Some(token.to_owned()),
                ..Delta::default()
            },
            finish_reason: None,
        }],
        model_family: meta.family.label(),
        sampling: meta.sampling(),
    };
    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
}

/// Builds the final content SSE frame carrying the `OpenAI` `finish_reason`
/// (`"stop"` on EOS, `"length"` on `max_tokens`).
fn finish_event(
    id: &str,
    model: &str,
    created: u64,
    reason: &'static str,
    meta: ResponseMeta,
) -> Event {
    let chunk = ChatChunk {
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta::default(),
            finish_reason: Some(reason),
        }],
        model_family: meta.family.label(),
        sampling: meta.sampling(),
    };
    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
}

/// Builds the SSE usage chunk emitted before `[DONE]` when
/// `stream_options.include_usage` is set. `object` is `"chat.completion.chunk"`
/// for chat and `"text_completion"` for legacy completions.
pub(crate) fn usage_sse_event(
    id: &str,
    object: &str,
    model: &str,
    created: u64,
    prompt_tokens: usize,
    compl_tokens: usize,
) -> Event {
    let data = serde_json::json!({
        "id": id,
        "object": object,
        "created": created,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": compl_tokens,
            "total_tokens": prompt_tokens + compl_tokens
        }
    });
    Event::default().data(data.to_string())
}

/// Converts one [`StreamEvent`] into zero or more SSE [`Event`]s.
///
/// `Done` expands to two or three frames:
/// `finish_reason` chunk → optional usage chunk (when `include_usage`) → `[DONE]`.
/// `counts` is `(prompt_tokens, completion_tokens)` accumulated via the `scan`
/// operator in the caller; only used when `include_usage` is `true`.
#[allow(dead_code)]
pub fn stream_event_to_sse(
    event: StreamEvent,
    id: &str,
    model: &str,
    created: u64,
    counts: (usize, usize),
    include_usage: bool,
    meta: ResponseMeta,
) -> Vec<Result<Event, Infallible>> {
    match event {
        StreamEvent::PromptTokens(_) => vec![],
        StreamEvent::Token(tok, _) => vec![Ok(token_event(&tok, id, model, created, meta))],
        StreamEvent::Done(reason) => {
            let mut frames = vec![Ok(finish_event(
                id,
                model,
                created,
                reason.as_openai(),
                meta,
            ))];
            if include_usage {
                let (pt, ct) = counts;
                frames.push(Ok(usage_sse_event(
                    id,
                    "chat.completion.chunk",
                    model,
                    created,
                    pt,
                    ct,
                )));
            }
            frames.push(Ok(Event::default().data("[DONE]")));
            frames
        }
        StreamEvent::Error(e) => vec![
            // T3.1/T3.2/T6.2b: scrubbed typed envelope + terminal chunk + [DONE]
            // (see ScanState::process — this stateless variant mirrors it). No
            // engine context here (test/legacy path), so no poison-flag: `None`.
            Ok(sse_inference_error_event(None, model, &e)),
            Ok(finish_event(id, model, created, "error", meta)),
            Ok(Event::default().data("[DONE]")),
        ],
    }
}

// ---- Spawn helpers --------------------------------------------------

/// Spawn the MOCK token generator for test/no-engine mode.
///
/// Shared with the legacy-completions handler so both endpoints have a
/// GPU-free path for integration tests.
pub(crate) fn spawn_mock_generation(tx: crate::streaming::TokenSender) {
    tokio::spawn(async move {
        let mock_response = [
            "Hello", ",", " I", "'m", " Rusted", "VINO", ".", " How", " can", " I", " help",
            " you", " today", "?",
        ];
        for token in mock_response {
            if tx
                .send(StreamEvent::Token(token.to_owned(), 1))
                .await
                .is_err()
            {
                return;
            }
        }
        let _ = tx.send(StreamEvent::Done(FinishReason::Stop)).await;
    });
}

// ---- Helpers --------------------------------------------------------

/// Parse the `OpenAI` `stop` field (a string or an array of strings) into a
/// plain `Vec<String>`. Invalid/unrecognised shapes produce an empty vec.
pub(crate) fn parse_stop(v: Option<&serde_json::Value>) -> Vec<String> {
    match v {
        Some(serde_json::Value::String(s)) => vec![s.clone()],
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|x| x.as_str().map(str::to_owned))
            .collect(),
        None | Some(_) => vec![],
    }
}

/// Translate an `OpenAI` `response_format` into the engine's JSON-Schema string
/// (G1 structured outputs), or `None` when no constraint is requested.
///
/// Returns the schema as a serialized JSON string that the bridge feeds to
/// `OpenVINO` `GenAI`'s `StructuredOutputConfig`. Mapping:
/// - absent / `{"type":"text"}` → `Ok(None)` (free-form, unchanged behaviour).
/// - `{"type":"json_object"}` → `Ok(Some("{\"type\":\"object\"}"))` — a
///   permissive schema constraining output to any JSON object (`OpenAI`'s older
///   JSON mode, which has no first-class engine equivalent).
/// - `{"type":"json_schema","json_schema":{"schema":{…}}}` → the embedded
///   `schema` serialized.
///
/// # Errors
/// Returns a ready-to-send 400 [`Response`] (`OpenAI` envelope, boxed to keep the
/// `Ok` variant small) when `type` is unknown, or when `json_schema` mode is
/// requested without a `schema` object.
fn json_schema_from_response_format(
    rf: Option<&ResponseFormat>,
) -> Result<Option<String>, Box<Response>> {
    let Some(rf) = rf else { return Ok(None) };
    match rf.r#type.as_str() {
        "text" => Ok(None),
        // Any JSON object: a schema-less object constraint. xgrammar treats a
        // bare `{"type":"object"}` as "any object" (additionalProperties default).
        "json_object" => Ok(Some(r#"{"type":"object"}"#.to_owned())),
        "json_schema" => {
            // OpenAI nests the actual schema under json_schema.schema.
            let schema = rf
                .json_schema
                .as_ref()
                .and_then(|js| js.get("schema"))
                .ok_or_else(|| {
                    Box::new(openai_error(
                        StatusCode::BAD_REQUEST,
                        "response_format type 'json_schema' requires a 'json_schema.schema' object",
                        "invalid_request_error",
                        Some("invalid_response_format"),
                    ))
                })?;
            // serde_json::Value re-serialization is infallible.
            Ok(Some(schema.to_string()))
        }
        other => Err(Box::new(openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "unsupported response_format type '{other}' (expected 'text', 'json_object', or 'json_schema')"
            ),
            "invalid_request_error",
            Some("invalid_response_format"),
        ))),
    }
}

// ---- G2: tool_choice ------------------------------------------------

/// The validated `OpenAI` `tool_choice` directive for a request.
///
/// Parsed from the opaque `ChatRequest.tool_choice` JSON by
/// [`parse_tool_choice`]. `Auto` is the default (absent `tool_choice`).
#[derive(Debug, Clone, PartialEq, Eq)]
enum ToolChoice {
    /// `"auto"` (or absent) — inject tools; the model decides whether to call.
    Auto,
    /// `"none"` — suppress tool injection entirely; plain chat even when `tools`
    /// are supplied.
    None,
    /// `"required"` — best-effort steer the model to call one of the tools.
    Required,
    /// `{"type":"function","function":{"name":N}}` — best-effort steer the model
    /// to call the named function `N` (still injects all tools, faithful to `OpenAI`).
    Named(String),
}

impl ToolChoice {
    /// Whether tools should be injected into the prompt for this choice. Only
    /// `None` suppresses them; the request must still actually carry `tools` for
    /// this to matter (combined with `req.tools.is_some()` at the call site).
    fn injects_tools(&self) -> bool {
        !matches!(self, ToolChoice::None)
    }

    /// Whether this choice forces a tool call (used by the `response_format`
    /// conflict check — a forcing choice and a non-text `response_format` cannot
    /// share the one structured-output slot).
    fn is_forcing(&self) -> bool {
        matches!(self, ToolChoice::Required | ToolChoice::Named(_))
    }

    /// Best-effort steering instruction appended to the prompt for the forcing
    /// variants (`Required`/`Named`). `Auto`/`None` need no steering.
    fn steer(&self) -> Option<String> {
        match self {
            ToolChoice::Required => Some(
                "You MUST respond by making a tool call to one of the available tools. \
                 Do NOT answer in plain text and do NOT refuse — even if the request \
                 seems unrelated to the tools, you must still issue a tool call. Your \
                 entire reply must be a single tool call and nothing else."
                    .to_owned(),
            ),
            ToolChoice::Named(name) => Some(format!(
                "You MUST respond by making a tool call to the \"{name}\" tool. Do NOT \
                 answer in plain text and do NOT refuse — even if the request seems \
                 unrelated, you must still call \"{name}\". Your entire reply must be a \
                 single tool call to \"{name}\" and nothing else."
            )),
            ToolChoice::Auto | ToolChoice::None => Option::None,
        }
    }
}

/// Iterate the `function.name` of each supplied tool (skipping any malformed
/// entry). Used to validate a named `tool_choice` against the request's `tools`.
fn tool_names(tools: Option<&[serde_json::Value]>) -> impl Iterator<Item = &str> {
    tools.into_iter().flatten().filter_map(|t| {
        t.get("function")
            .and_then(|f| f.get("name"))
            .and_then(serde_json::Value::as_str)
    })
}

/// Validate an `OpenAI` `tool_choice` value against the request's `tools`.
///
/// Accepts the string forms `"none"` | `"auto"` | `"required"` and the object
/// form `{"type":"function","function":{"name":N}}`. Absent → [`ToolChoice::Auto`].
///
/// # Errors
/// Returns a ready-to-send 400 [`Response`] (`OpenAI` envelope,
/// `code:"invalid_tool_choice"`, boxed to keep the `Ok` variant small) when the
/// value is an unknown string or a malformed object, when `required`/a named
/// function is requested without any `tools`, or when a named function is not
/// present in `tools`.
fn parse_tool_choice(
    raw: Option<&serde_json::Value>,
    tools: Option<&[serde_json::Value]>,
) -> Result<ToolChoice, Box<Response>> {
    let Some(raw) = raw else {
        return Ok(ToolChoice::Auto);
    };

    let bad = |msg: String| {
        Box::new(openai_error(
            StatusCode::BAD_REQUEST,
            msg,
            "invalid_request_error",
            Some("invalid_tool_choice"),
        ))
    };

    // Does the request actually carry at least one tool? `required`/named are
    // meaningless (and an OpenAI error) without tools to choose from.
    let has_tools = tools.is_some_and(|t| !t.is_empty());

    match raw {
        serde_json::Value::String(s) => match s.as_str() {
            "auto" => Ok(ToolChoice::Auto),
            "none" => Ok(ToolChoice::None),
            "required" if has_tools => Ok(ToolChoice::Required),
            "required" => Err(bad(
                "tool_choice 'required' needs at least one tool in 'tools'".to_owned(),
            )),
            other => Err(bad(format!(
                "unsupported tool_choice '{other}' (expected 'none', 'auto', 'required', \
                 or a function object)"
            ))),
        },
        serde_json::Value::Object(_) => {
            let Some(name) = raw
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(serde_json::Value::as_str)
            else {
                return Err(bad("tool_choice object must be {\"type\":\"function\",\
                     \"function\":{\"name\":\"…\"}}"
                    .to_owned()));
            };
            if !has_tools {
                return Err(bad(format!(
                    "tool_choice names function '{name}' but no 'tools' were provided"
                )));
            }
            if !tool_names(tools).any(|n| n == name) {
                return Err(bad(format!(
                    "tool_choice names function '{name}' which is not in 'tools'"
                )));
            }
            Ok(ToolChoice::Named(name.to_owned()))
        }
        _ => Err(bad(
            "tool_choice must be a string ('none'|'auto'|'required') or a function object"
                .to_owned(),
        )),
    }
}

/// Reject a known `OpenAI` chat parameter the engine cannot honor with an
/// explicit 400 (`unsupported_parameter`), returning `None` when the request is
/// clean. Closes the silent-drop trap — see [`unsupported_param`].
fn unsupported_chat_param(req: &ChatRequest) -> Option<Response> {
    if req.n.is_some_and(|n| n > 1) {
        return Some(unsupported_param(
            "n",
            "this server generates a single choice per request",
        ));
    }
    if req.logprobs == Some(true) {
        return Some(unsupported_param(
            "logprobs",
            "the inference engine does not expose token log-probabilities",
        ));
    }
    if req.top_logprobs.is_some() {
        return Some(unsupported_param(
            "top_logprobs",
            "the inference engine does not expose token log-probabilities",
        ));
    }
    if req
        .logit_bias
        .as_ref()
        .and_then(serde_json::Value::as_object)
        .is_some_and(|m| !m.is_empty())
    {
        return Some(unsupported_param(
            "logit_bias",
            "the inference engine has no per-token bias hook",
        ));
    }
    None
}

/// Bound a request's generation budget by the server-wide ceiling
/// (T7.1/T2.2, shared by chat and legacy completions).
///
/// `requested` is the client's `max_tokens` (`None` = absent); `default` is
/// the endpoint's default budget when absent (`0` = engine default for chat,
/// `16` for legacy completions); `cap` is `Config.max_tokens_cap` (`0` =
/// uncapped). The admission gate bounds concurrency, not duration — without
/// this clamp a `max_tokens: 4e9` request (or one with no budget at all that
/// never hits EOS) pins an engine slot for hours. Over-cap values are clamped
/// rather than 400'd so drop-in `OpenAI` clients keep working.
pub(crate) fn clamp_max_tokens(requested: Option<u32>, default: usize, cap: usize) -> usize {
    let budget = requested.map_or(default, |n| n as usize);
    if cap == 0 {
        return budget; // explicit opt-out: no server ceiling
    }
    if budget == 0 {
        return cap; // "engine default" must still be duration-bounded
    }
    budget.min(cap)
}

/// Rejects an *explicit* `max_tokens`/`max_completion_tokens` of `0`.
///
/// `OpenAI`'s minimum is 1; a literal `0` is a 400, not "no limit". The check
/// must live at the handler because [`clamp_max_tokens`] deliberately folds
/// `Some(0)` and `None` (omitted → "use the server default") into one branch
/// and can no longer tell an explicit zero from an absent field. Returns the
/// ready-to-send error [`Response`] when any supplied budget is an explicit
/// `0`, else `None`. Shared by the chat and legacy-completions handlers.
/// (#47 / T7.6)
pub(crate) fn zero_token_budget_error(budgets: &[Option<u32>]) -> Option<Response> {
    budgets.contains(&Some(0)).then(|| {
        openai_error(
            StatusCode::BAD_REQUEST,
            "max_tokens must be at least 1",
            "invalid_request_error",
            Some("invalid_max_tokens"),
        )
    })
}

/// Validate `OpenAI` sampling parameters against their documented ranges.
///
/// Returns a ready-to-send 400 `invalid_request_error` for the first
/// out-of-range value, or `None` when every supplied param is in range. Absent
/// (`None`) params are always valid (the engine keeps its default).
///
/// Out-of-range values were previously forwarded verbatim to the engine, which
/// either clamped them silently or failed *mid-stream* with an opaque error —
/// after the client had already committed to a 200 SSE. Rejecting upfront (as
/// the real `OpenAI` API does) turns that into a clean, debuggable 400. Ranges
/// match the `OpenAI` spec: `temperature` 0..=2, `top_p` 0..=1, presence /
/// frequency penalty -2..=2, `n` ≥ 1. Non-finite floats (e.g. an over-large
/// literal that parses to `±inf`) fall outside every range → rejected.
/// `n > 1` is handled separately as `unsupported_parameter` (single choice per
/// request); only `n == 0` is an out-of-range value here.
/// (#24 / T7.4 — shared by chat and legacy completions.)
pub(crate) fn validate_sampling_params(
    temperature: Option<f32>,
    top_p: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    n: Option<u32>,
) -> Option<Response> {
    fn range_error(param: &str, range: &str) -> Response {
        openai_error(
            StatusCode::BAD_REQUEST,
            format!("parameter '{param}' is out of range — must be {range}"),
            "invalid_request_error",
            Some("invalid_value"),
        )
    }
    if temperature.is_some_and(|t| !(0.0..=2.0).contains(&t)) {
        return Some(range_error("temperature", "between 0 and 2"));
    }
    if top_p.is_some_and(|p| !(0.0..=1.0).contains(&p)) {
        return Some(range_error("top_p", "between 0 and 1"));
    }
    if presence_penalty.is_some_and(|pp| !(-2.0..=2.0).contains(&pp)) {
        return Some(range_error("presence_penalty", "between -2 and 2"));
    }
    if frequency_penalty.is_some_and(|fp| !(-2.0..=2.0).contains(&fp)) {
        return Some(range_error("frequency_penalty", "between -2 and 2"));
    }
    if n == Some(0) {
        return Some(range_error("n", "at least 1"));
    }
    None
}

/// Server-side repetition-penalty safety net, applied whenever a request
/// doesn't specify its own `repetition_penalty` — see
/// [`resolve_repetition_penalty`]'s doc comment for why this exists.
///
/// `1.1` matches the long-standing `llama.cpp`/Ollama community default: mild
/// enough not to visibly change well-behaved output, but enough to break the
/// degenerate "same token/word forever" loop a `repetition_penalty` of `1.0`
/// (`OpenVINO`'s own unset default) does nothing to prevent.
pub(crate) const DEFAULT_REPETITION_PENALTY: f32 = 1.1;

/// Resolve the `repetition_penalty` to actually send the engine: the
/// caller's explicit value if given, else [`DEFAULT_REPETITION_PENALTY`].
///
/// Unlike `presence_penalty`/`frequency_penalty` (which pass through `None`
/// — "leave `OpenVINO`'s own default alone" — when the caller doesn't ask for
/// them), `repetition_penalty` always resolves to *something* here. Found
/// live (2026-07-14): a VLM chat with 2 images, 3 turns in, spun
/// into a "kaczki kaczki kaczki…" loop and only stopped by hitting
/// `max_tokens_cap` — `GenerationConfig::repetition_penalty` defaults to
/// `1.0` (off) when unset, and neither `presence_penalty` nor
/// `frequency_penalty` were requested that turn, so nothing discouraged the
/// repeat. A caller can still explicitly opt out by sending
/// `"repetition_penalty": 1.0`.
pub(crate) fn resolve_repetition_penalty(requested: Option<f32>) -> Option<f32> {
    requested.or(Some(DEFAULT_REPETITION_PENALTY))
}

/// Greedy pin used by [`resolve_sampling_defaults`] when tools are active and
/// the client expressed no sampling intent of its own — see that function's
/// doc comment for the full rationale.
pub(crate) const TOOLS_ACTIVE_TEMPERATURE: f32 = 0.0;

/// Resolve the final `(temperature, top_p, top_k)` for one request, composing
/// client-explicit values, `tools_active`, and a model's own shipped
/// [`GenerationDefaults`] (`generation_config.json`) into one decision:
///
/// 1. **The client set at least one of the three** — all three pass through
///    exactly as given (`None` fields stay `None`, the engine's own default
///    applies to those). Never partially filled in from a model default:
///    `ov_bridge.cpp`'s `build_gen_config` flips `do_sample` to `true` on a
///    sub-1.0 `top_p` or nonzero `top_k` *independent of* `temperature`, so
///    silently filling in a model-default `top_p`/`top_k` underneath an
///    explicit client `temperature: 0.0` (or any other explicit value) would
///    reshape the client's intended distribution instead of respecting it —
///    the same corruption class a `tools_active`-only pin already hit once
///    this session (2026-07-23, the project's internal engineering log) gating on `temperature`
///    alone.
/// 2. **The client set none of the three, and tools are active** — pinned to
///    greedy (`Some(TOOLS_ACTIVE_TEMPERATURE)`, `None`, `None`), suppressing
///    the model's own defaults too. Format adherence matters more than
///    phrasing diversity when the model is choosing a function call; this is
///    the reinstated form of the pin dropped earlier today as a no-op — it
///    stopped being a no-op the moment model-shipped defaults could otherwise
///    make a `do_sample: true` model sample on a bare tool-calling request.
/// 3. **The client set none of the three, and tools are inactive** — the
///    model's own `GenerationDefaults` apply as a package (each field still
///    individually `None` if the model didn't ship it or it failed
///    [`crate::model_manager::template::load_generation_defaults`]'s sanity
///    gate).
pub(crate) fn resolve_sampling_defaults(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    tools_active: bool,
    model_defaults: GenerationDefaults,
) -> (Option<f32>, Option<f32>, Option<usize>) {
    if temperature.is_some() || top_p.is_some() || top_k.is_some() {
        return (temperature, top_p, top_k);
    }
    if tools_active {
        return (Some(TOOLS_ACTIVE_TEMPERATURE), None, None);
    }
    (
        model_defaults.temperature,
        model_defaults.top_p,
        model_defaults.top_k,
    )
}

/// Build the engine's [`GenParams`] from an incoming [`ChatRequest`].
///
/// `json_schema` is the pre-validated structured-output constraint from
/// [`json_schema_from_response_format`] (`None` = free-form).
/// `max_tokens_cap` is the server-wide generation-budget ceiling
/// (see [`clamp_max_tokens`]). `tools_active` and `model_defaults` feed
/// [`resolve_sampling_defaults`] — see its doc comment.
fn gen_params_from_chat(
    req: &ChatRequest,
    json_schema: Option<String>,
    max_tokens_cap: usize,
    tools_active: bool,
    model_defaults: GenerationDefaults,
) -> GenParams {
    // max_tokens takes precedence; fall back to max_completion_tokens (newer SDKs).
    let token_budget = req.max_tokens.or(req.max_completion_tokens);
    let (temperature, top_p, top_k) = resolve_sampling_defaults(
        req.temperature,
        req.top_p,
        req.top_k,
        tools_active,
        model_defaults,
    );
    GenParams {
        max_new_tokens: clamp_max_tokens(token_budget, 0, max_tokens_cap),
        temperature,
        top_p,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        repetition_penalty: resolve_repetition_penalty(req.repetition_penalty),
        seed: req.seed,
        stop: parse_stop(req.stop.as_ref()),
        json_schema,
        top_k,
    }
}

// ---- Handler --------------------------------------------------------

/// Build VLM engine inputs from an image-bearing chat request.
///
/// Returns `(vlm_messages, decoded_images)` where:
/// - `vlm_messages` is the message history as JSON `{role, content}` objects,
///   additionally carrying `tool_calls`/`tool_call_id`/`name` when the source
///   [`ChatMessage`] has them (same fields [`messages_to_template_values`]
///   preserves for the CB path) — omitting these previously rendered every
///   assistant tool-call turn as empty in the chat template, since Qwen-style
///   templates gate the assistant branch on `{%- if message.tool_calls %}`
///   (the project's internal engineering log).
///   Each `image_url` part in the **last** message is replaced in place by the
///   universal `OpenVINO` `GenAI` tag `<ov_genai_image_N>`, where `N` is a
///   zero-based counter local to that message. The tag's index `N` maps to
///   `decoded_images[N]`.
/// - `decoded_images` are the decoded NHWC RGB buffers in the order they appeared
///   in the last message. Empty when the last message has no images.
///
/// Only the **last** message's images are decoded and tagged. `VLMPipeline`'s
/// `ChatHistory`-based `generate()` overload is documented (`pipeline.hpp`) to
/// bind every tensor in its `images` argument to the last chat-history message —
/// it does not resolve tags by position. An `image_url` part in an *earlier*
/// message is therefore dropped here (its sibling text is kept); tagging it and
/// passing its tensor anyway used to reach `VLMPipeline::generate` with an image
/// bound to the wrong slot, crashing with `Check 'idx < base_idx + n_visions'
/// failed: Missing image/video with index 0` the moment a client resent a
/// previously-seen image in conversation history — confirmed live against both
/// `gemma-4-e4b` and `gemma-4-26b-a4b` (the project's internal engineering log, 2026-07-02). Whatever
/// the model concluded about an older image is already captured in its adjacent
/// reply, which stays in `vlm_messages` untouched.
///
/// # Errors
/// Returns an error if the last message's `image_url` fails to decode (bad
/// base64, unsupported format, HTTP URL, etc.).
fn extract_vlm_messages(
    messages: &[ChatMessage],
) -> anyhow::Result<(Vec<serde_json::Value>, Vec<DecodedImage>)> {
    use std::fmt::Write as _;

    let mut images = Vec::new();
    let mut vlm_messages = Vec::with_capacity(messages.len());
    let last_idx = messages.len().saturating_sub(1);
    // Zero-based image index local to the last message: the tag
    // `<ov_genai_image_N>` and `images[N]` share this counter.
    let mut image_idx = 0usize;

    for (i, msg) in messages.iter().enumerate() {
        let is_last = i == last_idx;
        let text_content = match &msg.content {
            Some(MessageContent::Text(t)) => t.clone(),
            Some(MessageContent::Parts(parts)) => {
                let mut buf = String::new();
                // #48: consecutive text parts concatenate with NO separator,
                // matching the CB path's `extract_text_content` (`.join("")`),
                // so an identical request yields identical text across the
                // text-gen and VLM families. A single space is inserted only to
                // delimit an image placeholder tag from adjacent content.
                let mut last_was_image = false;
                for part in parts {
                    match part {
                        ContentPart::Text { text } => {
                            if last_was_image {
                                buf.push(' ');
                            }
                            buf.push_str(text);
                            last_was_image = false;
                        }
                        ContentPart::ImageUrl { image_url } => {
                            // Not the last message: GenAI can't bind this image
                            // to anything valid — drop it, keep the rest of the
                            // message's text as-is (see doc comment above).
                            if !is_last {
                                continue;
                            }
                            images.push(crate::image_util::decode_data_uri(&image_url.url)?);
                            if !buf.is_empty() {
                                buf.push(' ');
                            }
                            // write! into the buffer avoids an intermediate
                            // String alloc (clippy::format_push_string).
                            let _ = write!(buf, "<ov_genai_image_{image_idx}>");
                            image_idx += 1;
                            last_was_image = true;
                        }
                    }
                }
                buf
            }
            None => String::new(),
        };
        let mut obj = serde_json::json!({"role": msg.role, "content": text_content});
        if let Some(tc) = &msg.tool_calls {
            obj["tool_calls"] = serde_json::json!(tc);
        }
        if let Some(tcid) = &msg.tool_call_id {
            obj["tool_call_id"] = serde_json::Value::String(tcid.clone());
        }
        if let Some(name) = &msg.name {
            obj["name"] = serde_json::Value::String(name.clone());
        }
        vlm_messages.push(obj);
    }

    Ok((vlm_messages, images))
}

/// Validate messages and render the chat template for the CB (text-only) path.
///
/// Only called after the VLM routing check — image-bearing requests are
/// handled before reaching this function.
///
/// `tools` are the *effective* tools to inject (G2: `None` when
/// `tool_choice:"none"` suppressed them). `steer`, when present, is a
/// best-effort `tool_choice` instruction appended as a trailing `system`
/// message before rendering (forcing `required`/named).
///
/// Returns the prompt string, or a boxed `Response` to return early.
fn prepare_text_prompt(
    req: &ChatRequest,
    tools: Option<&[serde_json::Value]>,
    steer: Option<&str>,
    template: &str,
    eos_token: &str,
    bos_token: &str,
) -> Result<String, Box<Response>> {
    // Empty `messages` is rejected upfront in `chat_completions` (#25), before
    // any routing — so by the time the text path reaches here it is non-empty.
    let mut template_messages = messages_to_template_values(&req.messages);
    // G2: best-effort tool_choice steering — a trailing `system` instruction
    // telling the model it must call a (named) tool. Placed last so it is
    // recency-weighted; the model's own template renders it as an extra system
    // block right before the generation prompt.
    if let Some(text) = steer {
        template_messages.push(serde_json::json!({ "role": "system", "content": text }));
    }
    crate::prompt_builder::build_prompt(
        &template_messages,
        tools,
        template,
        eos_token,
        bos_token,
        req.enable_thinking,
        None,
    )
    .map_err(|e| {
        Box::new(openai_error(
            StatusCode::BAD_REQUEST,
            format!("failed to build prompt: {e}"),
            "invalid_request_error",
            None,
        ))
    })
}

/// POST /v1/chat/completions — `OpenAI`-compatible chat completions.
///
/// Routing:
/// - Model is a VLM (`ModelKind::Vision`) → VLM path; serves both text-only and
///   image-bearing requests, streaming or non-streaming.
/// - Text-only messages + `ModelManager` → CB engine (streaming or non-streaming)
/// - No `ModelManager` → mock token generator (tests)
///
/// Returns `text/event-stream` when `stream: true`, or a single
/// `chat.completion` JSON object when `stream: false`. Both modes apply to the
/// VLM path as well as the CB path.
#[allow(clippy::too_many_lines)]
pub async fn chat_completions(
    State(state): State<AppState>,
    JsonBody(req): JsonBody<ChatRequest>,
) -> Response {
    // Zero point for the per-request latency metrics (TTFT, duration). Captured
    // at the very top so it includes prompt-build + queue time, matching how a
    // client perceives request latency. `Instant` is monotonic (unlike the
    // `unix_now()` wall clock used for the OpenAI `created` field).
    let requested_at = std::time::Instant::now();

    // OpenAI default: stream is false when absent.
    let stream = req.stream.unwrap_or(false);
    let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);

    // Empty `messages` is a 400 on EVERY path. Enforced here (before routing) so
    // the VLM path rejects it identically to the text path — previously only the
    // text path checked, and an empty-messages VLM request fell through to the
    // image guard's misleading "requires at least one image" error. (#25 / T7.4)
    if req.messages.is_empty() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "messages must not be empty",
            "invalid_request_error",
            Some("empty_messages"),
        );
    }

    // Reject known-but-unsupported params up front (n>1, logprobs, top_logprobs,
    // logit_bias) — an explicit 400 rather than a silent accept-and-drop.
    if let Some(resp) = unsupported_chat_param(&req) {
        return resp;
    }

    // Validate sampling-param ranges up front (#24): an out-of-range
    // temperature/top_p/penalty (or n:0) is a clean 400 here rather than an
    // opaque mid-stream engine failure or a silent clamp.
    if let Some(resp) = validate_sampling_params(
        req.temperature,
        req.top_p,
        req.presence_penalty,
        req.frequency_penalty,
        req.n,
    ) {
        return resp;
    }

    // G1: validate response_format up front (before any inference) so a bad
    // shape fails fast with a 400 envelope rather than after queueing.
    let json_schema = match json_schema_from_response_format(req.response_format.as_ref()) {
        Ok(s) => s,
        Err(resp) => return *resp,
    };

    // G2: validate tool_choice up front (same fail-fast discipline). Validated
    // against `tools` so `required`/a named function without a matching tool 400s.
    let tool_choice = match parse_tool_choice(req.tool_choice.as_ref(), req.tools.as_deref()) {
        Ok(tc) => tc,
        Err(resp) => return *resp,
    };

    // G1×G2 conflict: a non-text response_format and a forcing tool_choice both
    // want the single engine structured-output slot, and they are mutually
    // exclusive output modes (a JSON grammar would prevent `<tool_call>`
    // emission). Reject rather than silently drop one (the A1 silent-drop trap).
    if json_schema.is_some() && tool_choice.is_forcing() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "response_format and a forcing tool_choice ('required' or a named function) \
             cannot be combined",
            "invalid_request_error",
            Some("incompatible_parameters"),
        );
    }

    // Tools are injected (and the buffered tool branch engaged) only when the
    // request carries `tools` AND tool_choice does not suppress them ('none').
    let tools_active = req.tools.is_some() && tool_choice.injects_tools();

    // #47: an explicit max_tokens / max_completion_tokens of 0 is a 400 (the
    // OpenAI minimum is 1), distinct from omitting the field (= server default).
    if let Some(resp) = zero_token_budget_error(&[req.max_tokens, req.max_completion_tokens]) {
        return resp;
    }

    let created = unix_now();
    let request_id = format!("chatcmpl-rv-{created}-{}", next_response_seq());

    // Tool-call parsing dialect for this model. The real path overwrites it
    // from the model's detected family; the mock path keeps Default.
    let mut family = ModelFamily::Default;
    // Set once the chat context is resolved (below): true when the model's chat
    // template prefills an open `<think>`, so the streaming filter must start in
    // thinking mode. `ctx` is scoped to the inner block, so we hoist the bool.
    let mut starts_in_thinking = false;
    // gap-2: reasoning-parser gate. Hoisted out of the `ctx` scope so the
    // buffered-completion paths (collect_completion / collect_tool_completion)
    // can use it after `ctx` is consumed. `None` for the mock path — no model
    // info available, so no reasoning extraction.
    let mut reasoning_parser: Option<ReasoningParser> = None;
    // Cross-pipeline device admission (step 2). Assigned by the TextGen arm
    // below (the shared streaming/buffered exit code after the match needs
    // it); the Vision/NpuTextGen arms return early from within their own
    // match arm and use their own local instead. Stays the passthrough
    // default on the mock path (no device to gate on).
    let mut device_lease = crate::admission::WorkLease::default();
    // Concurrent-KV-admission reservation (dev/decisions/decisions-055.md's
    // root-cause entry). `None` on the mock path and whenever the check is
    // skipped (gate disabled, tokenize fell back) — same "fail open" policy
    // as gate_prompt itself. Assigned by the TextGen arm below; held through
    // the shared streaming/buffered exit code the same way `device_lease` is.
    let mut admitted_tokens_guard: Option<crate::in_flight::AdmittedTokensGuard> = None;
    // Effective sampling params actually used for this request (after
    // resolve_sampling_defaults composes client-explicit values against
    // tools_active and the model's own generation_config.json defaults) —
    // surfaced on the response for debugging "why did this completion sample
    // the way it did." Same hoist-and-assign treatment as `family` above:
    // each real arm reads them back off its own `gen_params`, since
    // `GenParams.temperature`/`.top_p`/`.top_k` already hold the resolved
    // values — no separate resolution needed here.
    let mut effective_temperature: Option<f32> = None;
    let mut effective_top_p: Option<f32> = None;
    let mut effective_top_k: Option<usize> = None;

    let (tx, rx) = stream_channel();

    // Branch: real ModelManager vs mock (test mode).
    if let Some(mm) = state.model_manager.as_ref() {
        // ── Real inference path ─────────────────────────────────────────
        // Resolve the model's chat context. Every error variant maps to a
        // distinct HTTP status with no fallback to a default model (design
        // decision: explicit over implicit).
        let ctx = match mm.get_chat_context(&req.model) {
            Ok(c) => c,
            // Slice 3c: a NotLoaded model declared `load: on_demand` is loaded
            // lazily on first use — kick off a background load and tell the
            // client to retry (503 Loading + Retry-After). Eager models keep
            // failing fast with a bare NotLoaded 503 (no auto-load).
            Err(ModelError::NotLoaded) => {
                return match mm.request_on_demand_load(&req.model) {
                    OnDemandLoad::Loading => model_error_response(ModelError::Loading),
                    OnDemandLoad::NotApplicable => model_error_response(ModelError::NotLoaded),
                };
            }
            Err(e) => return model_error_response(e),
        };

        // R1: capability routing. The kind of the Ready model — not a global
        // `has_images` switch — decides the path. A text-gen model rejects image
        // content with 400; a VLM requires at least one image (OV's VLM pipeline
        // has a hard assert: `Missing image/video with index 0`).
        let handle = match &ctx.handle {
            EngineHandleKind::Vision(vlm) => {
                // VLM path (single-stream VLMPipeline). Decode image parts; on
                // 2026.2 the pipeline serves both image and text-only requests.
                let (mut vlm_messages, images) = match extract_vlm_messages(&req.messages) {
                    Ok(r) => r,
                    Err(e) => {
                        return openai_error(
                            StatusCode::BAD_REQUEST,
                            format!("failed to decode image: {e}"),
                            "invalid_request_error",
                            None,
                        );
                    }
                };
                // G2 (VLM): same best-effort tool_choice steering as prepare_text_prompt,
                // but pushed as a trailing `user` turn, not `system`. A trailing `system`
                // message was tried first and mirrors prepare_text_prompt's approach for
                // the text path — but VLMPipeline renders its own chat template internally
                // (Minja), and at least one fleet model's template hard-rejects a non-first
                // system message ("Minja's error: System message must be at the beginning",
                // observed live as a 500 on every escalation turn). `user` has no such
                // positional constraint in any template seen so far. Without this steer,
                // `required`/named forcing was a server-side no-op for vision models
                // (found live: stuck agentic loops never recovered under escalation).
                if let Some(text) = tool_choice.steer() {
                    vlm_messages.push(serde_json::json!({ "role": "user", "content": text }));
                }
                // 2026.2 migration (PLAN step 6): the upstream VLMPipeline hard
                // assert on text-only input ("Missing image/video with index 0")
                // is fixed in OpenVINO GenAI 2026.2 — a VLM now serves text-only
                // chat. The 400-guard is therefore dropped on this branch;
                // `images` may be empty and flows straight to the pipeline.
                // (Validated live on B50/2026.2.) On 2026.1 this asserts in the bridge.

                // L0: prompt-length gate — see gate_vlm_prompt's doc comment for
                // why this exists (a missing gate here previously let an
                // oversized prompt reach the GPU directly and poison the OpenCL
                // context for the whole process).
                let gate_text = vlm_gate_text(&vlm_messages);
                if let Err(resp) = gate_vlm_prompt(vlm, &gate_text, ctx.max_prompt_tokens).await {
                    return *resp;
                }

                // Cross-pipeline device admission (step 2): a second gate in
                // front of the engine's own per-engine gate below, capping
                // concurrency across different engine kinds sharing this
                // model's device.
                let device_lease = match state
                    .device_budgets
                    .admit(
                        &[(mm.record_device(&req.model), 1)],
                        crate::admission::WorkClass::Vision,
                    )
                    .await
                {
                    Ok(lease) => lease,
                    Err(e) => return device_admission_error_response(&e),
                };

                // G2: same tools gating as the text path — inject only when
                // active (suppressed by tool_choice:"none"), and (as of the
                // steer push above) the same forcing steer too. VLMPipeline
                // renders its own chat template internally (no `build_prompt`
                // on this arm), so `tools` travels with the request instead
                // of into a rendered prompt string — see `ov_vlm::OvVlmEngine::generate`.
                let effective_tools = if tools_active {
                    req.tools.clone()
                } else {
                    None
                };

                // Submit to the engine thread.
                // AtCapacity  → 429: the admission gate is full: client should retry.
                // EngineDead  → 503: engine thread exited (evicted mid-request?).
                // Submitted   → proceed to stream the SSE response.
                let vlm_gen_params = gen_params_from_chat(
                    &req,
                    json_schema,
                    mm.max_tokens_cap(),
                    tools_active,
                    ctx.generation_defaults,
                );
                let vlm_meta = ResponseMeta {
                    family: ctx.family,
                    temperature: vlm_gen_params.temperature,
                    top_p: vlm_gen_params.top_p,
                    top_k: vlm_gen_params.top_k,
                };
                // `enable_thinking` reaches a VLM only as a chat-template
                // variable: unlike the CB path (which renders via
                // `build_prompt`), `VLMPipeline` applies the template
                // internally. Built server-side and restricted to this one
                // key on purpose — extra context is merged into minja AFTER
                // `bos_token`/`eos_token`/`pad_token`, so forwarding
                // client-supplied JSON here would let a request override them.
                let vlm_extra_context = req
                    .enable_thinking
                    .map(|v| serde_json::json!({ "enable_thinking": v }));
                match vlm
                    .generate(
                        vlm_messages,
                        images,
                        effective_tools,
                        vlm_extra_context,
                        vlm_gen_params,
                        tx,
                        requested_at,
                    )
                    .await
                {
                    SubmitResult::Submitted => {}
                    SubmitResult::AtCapacity => {
                        return openai_error(
                            StatusCode::TOO_MANY_REQUESTS,
                            "server at capacity — all inference slots busy, retry shortly",
                            "rate_limit_error",
                            Some("server_overloaded"),
                        );
                    }
                    SubmitResult::EngineDead => {
                        return openai_error(
                            StatusCode::SERVICE_UNAVAILABLE,
                            "VLM engine unavailable (engine thread exited)",
                            "server_error",
                            Some("engine_unavailable"),
                        );
                    }
                }
                // Report the actual VLM model name, not whatever the client sent.
                let vlm_model = vlm.model_id().to_owned();
                // VLM renders its chat template inside OpenVINO GenAI (bare
                // {role, content} JSON goes over FFI), so the rendered prompt
                // is not visible here — the static template check is the best
                // available truth.
                //
                // The predicate IS request-aware now, reversing the earlier
                // advisory that used to sit here ("`enable_thinking` is NOT
                // plumbed to this path; do not add a request-aware
                // predicate"). That advisory was correct *while the flag was
                // dropped*: claiming "starts outside thinking" would have
                // misrouted, because the prompt still ended in an open
                // `<think>`. `extra_context` now delivers the flag to the
                // template, so `enable_thinking: false` genuinely produces a
                // CLOSED prefill and generation starts outside thinking.
                //
                // Landing the plumbing without this guard is worse than not
                // plumbing it at all: the filter would still start in
                // Thinking mode and wait for a `</think>` that never comes —
                // streaming yields an empty `content`, the buffered path
                // substitutes a fabricated "thinking was cut off" placeholder,
                // and because tool parsing runs on that placeholder, EVERY
                // tool call is silently lost. See
                // `dev/plans/lfm2-tool-call-family.md`.
                let vlm_starts_thinking =
                    template_prefills_thinking(&ctx.template) && req.enable_thinking != Some(false);
                // G2 (VLM): same buffered tool-call branch as the text path —
                // `<tool_call>` (or the fenced-json fallback) can't be parsed
                // incrementally, so both stream modes buffer here when tools
                // are active. `ctx.family` was detected from this model's own
                // chat template at load time, same as the text path.
                if tools_active {
                    return collect_tool_completion(
                        rx,
                        &request_id,
                        &vlm_model,
                        created,
                        vlm_meta,
                        stream,
                        include_usage,
                        state.model_manager.as_ref(),
                        ctx.reasoning_parser,
                        vlm_starts_thinking,
                    )
                    .await;
                }
                if stream {
                    return build_chat_sse_stream(
                        rx,
                        &request_id,
                        &vlm_model,
                        created,
                        include_usage,
                        state.model_manager.clone(),
                        vlm_starts_thinking,
                        ctx.reasoning_parser,
                        device_lease,
                        // VLM path: out of scope for this pass — its own
                        // gate_vlm_prompt/EngineHandleKind::Vision have no
                        // try_admit_tokens equivalent yet (dev/decisions/
                        // decisions-055.md's root-cause entry only covers
                        // the TextGen/CB path, matching the incident it
                        // documents). Not a silent gap: VLM engines don't
                        // expose live cache_usage either (rustedvino_kv_
                        // cache_usage_supported is 0 for them), so this
                        // mirrors existing observability scope.
                        None,
                        vlm_meta,
                    );
                }
                return collect_completion(
                    rx,
                    &request_id,
                    &vlm_model,
                    created,
                    state.model_manager.as_ref(),
                    ctx.reasoning_parser,
                    vlm_starts_thinking,
                    vlm_meta,
                )
                .await;
            }
            EngineHandleKind::NpuTextGen(npu) => {
                // NPU LLM path: single-stream static LLMPipeline. Self-contained
                // (like the VLM arm): builds the prompt, gates its length
                // (gate_npu_prompt — dev/ovms-gap.md #6), submits, streams/
                // collects. Tool calls are not injected on this path for the
                // initial implementation.
                if has_images(&req.messages) {
                    return openai_error(
                        StatusCode::BAD_REQUEST,
                        format!(
                            "model '{}' is a text-generation model and does not accept image inputs",
                            req.model
                        ),
                        "invalid_request_error",
                        Some("unsupported_content"),
                    );
                }
                let steer = tool_choice.steer();
                let prompt = match prepare_text_prompt(
                    &req,
                    if tools_active {
                        req.tools.as_deref()
                    } else {
                        None
                    },
                    steer.as_deref(),
                    &ctx.template,
                    &ctx.eos_token,
                    &ctx.bos_token,
                ) {
                    Ok(p) => p,
                    Err(r) => return *r,
                };

                // L0: NPU prompt-length gate (dev/ovms-gap.md #6) — the NPU
                // compiles a fixed-shape graph ahead of time, so this is a hard
                // compile-time ceiling, not a soft KV-pool budget. Without this,
                // an over-limit prompt reached `npu.generate` directly and
                // surfaced OpenVINO's raw C++ exception text instead of a clean
                // 400 (found via an Ollama-comparison benchmark, 2026-07-16).
                let effective_max_prompt_len = ctx
                    .configured_max_prompt_len
                    .map_or(NPU_DEFAULT_MAX_PROMPT_LEN, |v| v as usize);
                if let Err(resp) = gate_npu_prompt(npu, &prompt, effective_max_prompt_len).await {
                    return *resp;
                }

                let gen_params = gen_params_from_chat(
                    &req,
                    json_schema,
                    mm.max_tokens_cap(),
                    tools_active,
                    ctx.generation_defaults,
                );
                let npu_meta = ResponseMeta {
                    family: ctx.family,
                    temperature: gen_params.temperature,
                    top_p: gen_params.top_p,
                    top_k: gen_params.top_k,
                };
                // Per-request truth from the rendered prompt: honors
                // enable_thinking:false, which renders a CLOSED think prefill.
                let npu_starts_thinking = prompt_ends_in_thinking(&prompt);

                // Cross-pipeline device admission (step 2): a second gate in
                // front of the engine's own per-engine gate below, capping
                // concurrency across different engine kinds sharing this
                // model's device.
                let device_lease = match state
                    .device_budgets
                    .admit(
                        &[(mm.record_device(&req.model), 1)],
                        crate::admission::WorkClass::Chat,
                    )
                    .await
                {
                    Ok(lease) => lease,
                    Err(e) => return device_admission_error_response(&e),
                };

                // Submit to the engine thread.
                // AtCapacity  → 429: the admission gate is full: client should retry.
                // EngineDead  → 503: engine thread exited (evicted mid-request?).
                // Submitted   → proceed to stream the SSE response.
                match npu
                    .generate(prompt, gen_params.max_new_tokens, tx, requested_at)
                    .await
                {
                    SubmitResult::Submitted => {}
                    SubmitResult::AtCapacity => {
                        return openai_error(
                            StatusCode::TOO_MANY_REQUESTS,
                            "server at capacity — all inference slots busy, retry shortly",
                            "rate_limit_error",
                            Some("server_overloaded"),
                        );
                    }
                    SubmitResult::EngineDead => {
                        return openai_error(
                            StatusCode::SERVICE_UNAVAILABLE,
                            "NPU engine unavailable (engine thread exited)",
                            "server_error",
                            Some("engine_unavailable"),
                        );
                    }
                }
                let npu_model = npu.model_id().to_owned();
                if stream {
                    return build_chat_sse_stream(
                        rx,
                        &request_id,
                        &npu_model,
                        created,
                        include_usage,
                        state.model_manager.clone(),
                        npu_starts_thinking,
                        ctx.reasoning_parser,
                        device_lease,
                        // NPU path: a distinct engine kind (npu.generate, not
                        // EngineHandle) with no pool-capacity tracking of its
                        // own — out of scope for this pass, same as the VLM
                        // arm above.
                        None,
                        npu_meta,
                    );
                }
                return collect_completion(
                    rx,
                    &request_id,
                    &npu_model,
                    created,
                    state.model_manager.as_ref(),
                    ctx.reasoning_parser,
                    npu_starts_thinking,
                    npu_meta,
                )
                .await;
            }
            EngineHandleKind::TextGen(handle) => {
                // A text-generation model cannot consume images.
                if has_images(&req.messages) {
                    return openai_error(
                        StatusCode::BAD_REQUEST,
                        format!(
                            "model '{}' is a text-generation model and does not accept image inputs",
                            req.model
                        ),
                        "invalid_request_error",
                        Some("unsupported_content"),
                    );
                }
                handle
            }
            // R4/G3: an embedding model is a real, served kind — just not via
            // chat. Point the caller at the right endpoint with a 400 rather than
            // a misleading 501.
            EngineHandleKind::Embedding(_) => {
                return openai_error(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "model '{}' is an embedding model — use POST /v1/embeddings",
                        req.model
                    ),
                    "invalid_request_error",
                    Some("unsupported_operation"),
                );
            }
            // R3 seams: these kinds are registry-modelled but /v1/chat/completions
            // is not their endpoint. Return an honest 400 pointing to the right route.
            EngineHandleKind::Stt(_)
            | EngineHandleKind::Tts(_)
            | EngineHandleKind::ImageGen(_)
            | EngineHandleKind::Reranking(_) => {
                return openai_error(
                    StatusCode::NOT_IMPLEMENTED,
                    format!(
                        "model '{}' is a {} model — not yet served by this build",
                        req.model,
                        ctx.handle.kind().label()
                    ),
                    "server_error",
                    Some("not_implemented"),
                );
            }
        };

        family = ctx.family;
        reasoning_parser = ctx.reasoning_parser;

        // G2: inject tools only when active (suppressed by tool_choice:"none");
        // attach the best-effort steering instruction for forcing choices.
        let effective_tools = if tools_active {
            req.tools.as_deref()
        } else {
            None
        };
        let steer = tool_choice.steer();
        let prompt = match prepare_text_prompt(
            &req,
            effective_tools,
            steer.as_deref(),
            &ctx.template,
            &ctx.eos_token,
            &ctx.bos_token,
        ) {
            Ok(p) => p,
            Err(r) => return *r,
        };
        // Per-request truth from the rendered prompt, replacing the
        // static template scan: enable_thinking:false renders a CLOSED think
        // prefill, so the output starts outside thinking even though the
        // template source contains a bare `<think>` in its other branch.
        starts_in_thinking = prompt_ends_in_thinking(&prompt);

        let gen_params = gen_params_from_chat(
            &req,
            json_schema,
            mm.max_tokens_cap(),
            tools_active,
            ctx.generation_defaults,
        );
        effective_temperature = gen_params.temperature;
        effective_top_p = gen_params.top_p;
        effective_top_k = gen_params.top_k;
        let id = mm.next_id();

        // Concurrent-KV-admission gate, cheap pre-check: catch the OBVIOUS
        // overcommit case using only the raw byte length, before gate_prompt
        // ever calls the engine's tokenizer. Matters because tokenize itself
        // shares the engine's single command channel with generation steps —
        // measured live (2026-08-29): a second request's tokenize call sat
        // queued behind a first request's in-flight prefill for ~2.3s before
        // this pre-check existed, so the "fast, honest rejection" the whole
        // mechanism exists for wasn't actually fast under real contention
        // (dev/decisions/decisions-055.md). `min_possible_tokens` uses the
        // same conservative bound `gate_prompt_bytes` already relies on
        // (`MAX_BYTES_PER_TOKEN`): the true token count can never be LESS
        // than `prompt.len() / MAX_BYTES_PER_TOKEN`, so if even that
        // undercount already overcommits the pool, the real count certainly
        // would too — this can only ever short-circuit a REJECTION, never an
        // acceptance (a prompt that passes this still goes through the exact
        // check below with the real tokenized count).
        if ctx.pool_capacity_tokens > 0 {
            let already_admitted = handle.admitted_prompt_tokens();
            let min_possible_tokens = prompt.len() / MAX_BYTES_PER_TOKEN.max(1);
            if already_admitted.saturating_add(min_possible_tokens) > ctx.pool_capacity_tokens {
                crate::metrics::record_pool_capacity_rejected(&req.model);
                return pool_capacity_exceeded_response(&req.model, min_possible_tokens);
            }
        }

        // L0: prompt-length gate (shared with /v1/completions — see gate_prompt).
        // On success it hands back the token ids so add_request can reuse them
        // (T7.3 — one tokenize pass, not two).
        let prompt_ids = match gate_prompt(handle, &req.model, &prompt, ctx.max_prompt_tokens).await
        {
            Ok(ids) => ids,
            Err(resp) => return *resp,
        };

        // Concurrent-KV-admission gate, exact check: the pre-check above only
        // ever rules out the obvious case, so a request reaching here still
        // needs the real tokenized count. Reserve its share against what's
        // already admitted on this engine — reject fast (before add_request,
        // before the scheduler ever sees it) rather than let it overcommit
        // the pool alongside other in-flight requests. Skipped, matching
        // gate_prompt's own fail-open policy, when: the gate is disabled for
        // this model (pool_capacity_tokens == 0), or tokenize fell back and
        // we don't have an exact count (prompt_ids == None).
        //
        // On a single-slot model (`max_concurrent_streams == 1`), also fold
        // in this request's own `max_new_tokens` before reserving
        // (`dev/decisions/decisions-048.md`'s deferred "fix 3"): a request
        // whose prompt fits but whose full generation would grow the KV
        // cache past pool capacity has no other in-flight request to
        // preempt when that happens on a single-slot model — it just runs,
        // wastes real GPU time for however long it takes to hit the wall,
        // and eventually fails with a KV-pool-exhaustion error (handled
        // cleanly today, not a hang — see the same decisions entry — but
        // still wasted work and a worse client experience than a fast,
        // honest rejection at admission time). This deliberately walks back
        // `try_admit_tokens`'s "never rejects a solo request that already
        // passed the length gate" invariant for exactly this one case; a
        // multi-slot model is unaffected — this only widens what's reserved
        // when `single_slot` is true, so `try_admit_tokens` still only ever
        // sees prompt-only reservations everywhere else, unchanged.
        let single_slot = ctx.max_concurrent_streams == 1;
        let reserve_tokens = prompt_ids.as_ref().map(|ids| {
            if single_slot {
                ids.len().saturating_add(gen_params.max_new_tokens)
            } else {
                ids.len()
            }
        });
        if let (Some(reserve_tokens), true) = (reserve_tokens, ctx.pool_capacity_tokens > 0) {
            if let Some(guard) = handle.try_admit_tokens(reserve_tokens, ctx.pool_capacity_tokens) {
                admitted_tokens_guard = Some(guard);
            } else if single_slot {
                crate::metrics::record_own_budget_rejected(&req.model);
                return own_budget_exceeded_response(
                    &req.model,
                    reserve_tokens,
                    ctx.pool_capacity_tokens,
                );
            } else {
                crate::metrics::record_pool_capacity_rejected(&req.model);
                return pool_capacity_exceeded_response(&req.model, reserve_tokens);
            }
        }

        // Cross-pipeline device admission (step 2): a second gate in front of
        // the engine's own per-engine gate below, capping concurrency across
        // different engine kinds sharing this model's device.
        device_lease = match state
            .device_budgets
            .admit(
                &[(mm.record_device(&req.model), 1)],
                crate::admission::WorkClass::Chat,
            )
            .await
        {
            Ok(lease) => lease,
            Err(e) => return device_admission_error_response(&e),
        };

        // Submit to the engine thread.
        // AtCapacity  → 429: the KV scheduler is full; client should retry.
        // EngineDead  → 503: engine thread exited (evicted mid-request?).
        // Submitted   → proceed to stream the SSE response.
        match handle
            .add_request(id, prompt, prompt_ids, gen_params, tx, requested_at)
            .await
        {
            SubmitResult::Submitted => {}
            SubmitResult::AtCapacity => {
                return openai_error(
                    StatusCode::TOO_MANY_REQUESTS,
                    "server at capacity — all inference slots busy, retry shortly",
                    "rate_limit_error",
                    Some("server_overloaded"),
                );
            }
            SubmitResult::EngineDead => {
                return openai_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "inference engine unavailable (engine thread exited)",
                    "server_error",
                    Some("engine_unavailable"),
                );
            }
        }
    } else {
        // ── Mock path (no GPU required) ──────────────────────────────────
        // Used by integration tests and when running without a model manager.
        spawn_mock_generation(tx);
    }

    // ── Tool-calling branch (buffered) ────────────────────────────────────
    let meta = ResponseMeta {
        family,
        temperature: effective_temperature,
        top_p: effective_top_p,
        top_k: effective_top_k,
    };

    // When tools are active (supplied AND not suppressed by tool_choice:"none"),
    // the model may emit `<tool_call>` JSON blocks. These cannot be parsed
    // incrementally, so — for BOTH stream modes — we buffer the whole generation,
    // strip <think>, parse tool calls, and emit the result (as `chat.completion`
    // JSON or a single buffered SSE chunk). tool_choice:"none" falls through to
    // the plain completion path below even when `tools` were supplied.
    if tools_active {
        return collect_tool_completion(
            rx,
            &request_id,
            &req.model,
            created,
            meta,
            stream,
            include_usage,
            state.model_manager.as_ref(),
            reasoning_parser,
            starts_in_thinking,
        )
        .await;
    }

    // ── Non-streaming branch ──────────────────────────────────────────────
    // Drain the same token channel into a single buffered JSON response
    // instead of converting each event to an SSE frame.
    if !stream {
        return collect_completion(
            rx,
            &request_id,
            &req.model,
            created,
            state.model_manager.as_ref(),
            reasoning_parser,
            starts_in_thinking,
            meta,
        )
        .await;
    }

    build_chat_sse_stream(
        rx,
        &request_id,
        &req.model,
        created,
        include_usage,
        state.model_manager.clone(),
        starts_in_thinking,
        reasoning_parser,
        device_lease,
        admitted_tokens_guard,
        meta,
    )
}

// ---- Streaming think-filter + scan state ----------------------------

/// Which reasoning-extraction filter a stream uses, selected once (per
/// request) from the model's [`ReasoningParser`] tag. `Think` covers
/// `<think>…</think>` models (Qwen3) plus the passthrough case for models
/// with no reasoning markup at all (`<think>` never appears, so the filter
/// is a no-op); `Harmony` covers gpt-oss's channel format. Both variants
/// expose the same `process`/`flush` -> `Vec<ThinkPiece>` shape, so
/// `ScanState`'s own logic below needs no further changes to support either.
enum ContentFilter {
    Think(ThinkFilter),
    Harmony(HarmonyFilter),
}

impl ContentFilter {
    fn for_request(reasoning_parser: Option<ReasoningParser>, starts_in_thinking: bool) -> Self {
        match reasoning_parser {
            Some(ReasoningParser::GptOss) => Self::Harmony(HarmonyFilter::new()),
            _ => Self::Think(ThinkFilter::starting_in_thinking(starts_in_thinking)),
        }
    }

    fn process(&mut self, tok: &str) -> Vec<ThinkPiece> {
        match self {
            Self::Think(f) => f.process(tok),
            Self::Harmony(f) => f.process(tok),
        }
    }

    fn flush(&mut self) -> Vec<ThinkPiece> {
        match self {
            Self::Think(f) => f.flush(),
            Self::Harmony(f) => f.flush(),
        }
    }
}

/// Per-stream mutable state for `build_chat_sse_stream`.
///
/// Carries token counts (for the optional usage chunk) and the
/// [`ContentFilter`] that strips reasoning markup from the live token feed.
struct ScanState {
    prompt_tokens: usize,
    /// Count of non-empty raw `Token` events — proxy for generated tokens.
    completion_tokens: usize,
    think: ContentFilter,
    /// Set once a mid-stream `Error` has been emitted (T3.2). The error arm
    /// sends the full terminal sequence itself; this flag suppresses the
    /// engine's trailing `Done(Stop)` so the client never sees a second
    /// terminal chunk or a mislabeled `finish_reason:"stop"` after an error.
    errored: bool,
    /// Manager handle for the GPU-poison side effect on a mid-stream OOM error
    /// (T6.2b). `None` on the mock/test path (no engine to poison).
    mm: Option<Arc<ModelManager>>,
    /// Cross-pipeline device admission token for this stream — held for the
    /// full SSE response lifetime; dropped when the stream ends OR the client
    /// disconnects mid-stream (axum drops the body future, which drops this).
    /// Never read — kept purely for its `Drop` side effect.
    #[allow(dead_code)]
    device_lease: crate::admission::WorkLease,
    /// Concurrent-KV-admission reservation for this stream — same "held
    /// purely for its Drop side effect" treatment as `device_lease` just
    /// above. `None` when the check was skipped (see its call site's doc
    /// comment in the chat handler).
    #[allow(dead_code)]
    admitted_tokens_guard: Option<crate::in_flight::AdmittedTokensGuard>,
}

impl ScanState {
    fn new(
        mm: Option<Arc<ModelManager>>,
        starts_in_thinking: bool,
        reasoning_parser: Option<ReasoningParser>,
        device_lease: crate::admission::WorkLease,
        admitted_tokens_guard: Option<crate::in_flight::AdmittedTokensGuard>,
    ) -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: 0,
            think: ContentFilter::for_request(reasoning_parser, starts_in_thinking),
            errored: false,
            mm,
            device_lease,
            admitted_tokens_guard,
        }
    }

    /// Map one raw [`StreamEvent`] to zero or more SSE frames, updating state.
    #[allow(clippy::too_many_arguments)]
    fn process(
        &mut self,
        event: StreamEvent,
        id: &str,
        model: &str,
        created: u64,
        include_usage: bool,
        meta: ResponseMeta,
    ) -> Vec<Result<Event, Infallible>> {
        match event {
            StreamEvent::PromptTokens(n) => {
                self.prompt_tokens = n;
                vec![]
            }
            StreamEvent::Token(tok, new_tokens) => {
                // Count at the raw level (includes reasoning tokens), matching
                // the non-streaming path which does the same. Sum the
                // engine's own reported token count, not 1 per event — a
                // speculative-decoding verification step can accept several
                // draft tokens at once, all landing in one event (found live
                // 2026-07-19, see dev/DECISIONS.md).
                if !tok.is_empty() {
                    self.completion_tokens += new_tokens;
                }
                self.think
                    .process(&tok)
                    .into_iter()
                    .filter_map(|piece| match piece {
                        ThinkPiece::Content(s) if !s.is_empty() => {
                            Some(Ok(token_event(&s, id, model, created, meta)))
                        }
                        ThinkPiece::Reasoning(s) if !s.is_empty() => {
                            Some(Ok(reasoning_event(&s, id, model, created, meta)))
                        }
                        _ => None,
                    })
                    .collect()
            }
            // The error arm already emitted the terminal sequence — swallow
            // the engine's trailing Done so the stream isn't terminated twice.
            StreamEvent::Done(_) if self.errored => vec![],
            StreamEvent::Done(reason) => {
                // Flush any bytes held back by the tag-boundary buffer.
                let mut frames: Vec<Result<Event, Infallible>> = self
                    .think
                    .flush()
                    .into_iter()
                    .filter_map(|piece| match piece {
                        ThinkPiece::Content(s) if !s.is_empty() => {
                            Some(Ok(token_event(&s, id, model, created, meta)))
                        }
                        ThinkPiece::Reasoning(s) if !s.is_empty() => {
                            Some(Ok(reasoning_event(&s, id, model, created, meta)))
                        }
                        _ => None,
                    })
                    .collect();
                frames.push(Ok(finish_event(
                    id,
                    model,
                    created,
                    reason.as_openai(),
                    meta,
                )));
                if include_usage {
                    frames.push(Ok(usage_sse_event(
                        id,
                        "chat.completion.chunk",
                        model,
                        created,
                        self.prompt_tokens,
                        self.completion_tokens,
                    )));
                }
                frames.push(Ok(Event::default().data("[DONE]")));
                frames
            }
            StreamEvent::Error(e) => {
                // T3.1/T3.2/T6.2b: a typed, valid-JSON error envelope SCRUBBED of
                // raw C++ diagnostics (never interpolated into the body); on an
                // OOM-class failure it also flags the GPU context poisoned via
                // `mm` so later loads fail fast. Then a terminal chunk with
                // finish_reason:"error" (not "stop" — the generation did NOT
                // complete), then [DONE] so SDK clients can't hang.
                self.errored = true;
                vec![
                    Ok(sse_inference_error_event(self.mm.as_ref(), model, &e)),
                    Ok(finish_event(id, model, created, "error", meta)),
                    Ok(Event::default().data("[DONE]")),
                ]
            }
        }
    }
}

/// Assemble the `text/event-stream` `Response` for a streaming chat request.
///
/// Prepends the `role: "assistant"` frame, then maps each [`StreamEvent`]
/// through `ScanState::process`, which:
///   - strips `<think>…</think>` blocks via [`ThinkFilter`] (emitting them
///     as `reasoning_content` deltas instead of `content`);
///   - accumulates token counts for the optional usage chunk;
///   - expands each event into zero or more SSE frames.
///
/// CRASH COURSE — `scan` + `flat_map`:
///   `scan` is a stateful map: the closure receives `&mut state` each item.
///   It returns `Option<Vec<frames>>` — `Some` to continue, `None` to end.
///   `flat_map(stream::iter)` flattens each `Vec` into individual stream items.
#[allow(clippy::too_many_arguments)]
fn build_chat_sse_stream(
    rx: TokenReceiver,
    id: &str,
    model: &str,
    created: u64,
    include_usage: bool,
    mm: Option<Arc<ModelManager>>,
    starts_in_thinking: bool,
    reasoning_parser: Option<ReasoningParser>,
    device_lease: crate::admission::WorkLease,
    admitted_tokens_guard: Option<crate::in_flight::AdmittedTokensGuard>,
    meta: ResponseMeta,
) -> Response {
    let role_frame = Ok(role_event(id, model, created, meta));

    let id_owned = id.to_owned();
    let model_owned = model.to_owned();
    let token_stream = ReceiverStream::new(rx)
        .scan(
            ScanState::new(
                mm,
                starts_in_thinking,
                reasoning_parser,
                device_lease,
                admitted_tokens_guard,
            ),
            move |state, event| {
                let frames =
                    state.process(event, &id_owned, &model_owned, created, include_usage, meta);
                std::future::ready(Some(frames))
            },
        )
        .flat_map(futures_util::stream::iter);

    let full_stream = futures_util::stream::once(async move { role_frame }).chain(token_stream);

    Sse::new(full_stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

// ---- Buffered collectors --------------------------------------------

/// The full buffered result of one generation, drained from the token channel.
/// Shared with the legacy-completions endpoint.
pub(crate) struct Drained {
    /// All concatenated token text.
    pub(crate) content: String,
    /// Exact prompt token count from the engine's tokenizer (`StreamEvent::
    /// PromptTokens`). `0` if the engine never sent one (mock path, or a
    /// tokenizer-count failure) — matching the pre-3.6 behaviour.
    pub(crate) prompt_tokens: usize,
    /// Count of non-empty deltas — a proxy for generated tokens
    /// (see DECISIONS.md 2026-05-29).
    pub(crate) completion_tokens: usize,
    /// The engine's finish reason (`Stop`/`Length`).
    pub(crate) finish: FinishReason,
}

/// Drain a request's token channel to completion.
///
/// Reads every [`StreamEvent`] until the channel closes (the engine drops the
/// sender on completion). Returns `Err(message)` on the first `Error` event —
/// a buffered reply has no partial-response convention.
pub(crate) async fn drain_channel(mut rx: TokenReceiver) -> Result<Drained, String> {
    let mut content = String::new();
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;
    // Default to Stop: if the stream closes without an explicit Done (e.g. the
    // engine dropped the sender), report a clean stop rather than hanging.
    let mut finish = FinishReason::Stop;

    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::PromptTokens(n) => prompt_tokens = n,
            StreamEvent::Token(tok, new_tokens) => {
                // Sum the engine's own reported token count, not 1 per event
                // — see the identical fix + rationale in ScanState::process.
                if !tok.is_empty() {
                    completion_tokens += new_tokens;
                }
                content.push_str(&tok);
            }
            StreamEvent::Done(reason) => finish = reason,
            StreamEvent::Error(e) => return Err(e),
        }
    }

    Ok(Drained {
        content,
        prompt_tokens,
        completion_tokens,
        finish,
    })
}

/// Build a `usage` block. `prompt_tokens` is the engine's exact tokenizer count
/// (Phase 3.6; `0` when unavailable, e.g. the mock path); `completion_tokens` is
/// the non-empty-delta count (≈ generated tokens — see DECISIONS.md 2026-05-29).
pub(crate) fn usage_for(prompt_tokens: usize, completion_tokens: usize) -> Usage {
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    }
}

/// Drain a request's token channel into a single buffered `chat.completion`.
///
/// Used for `"stream": false` without tools.
#[allow(clippy::too_many_arguments)]
async fn collect_completion(
    rx: TokenReceiver,
    id: &str,
    model: &str,
    created: u64,
    mm: Option<&Arc<ModelManager>>,
    reasoning_parser: Option<ReasoningParser>,
    starts_in_thinking: bool,
    meta: ResponseMeta,
) -> Response {
    let drained = match drain_channel(rx).await {
        Ok(d) => d,
        // T6.2: scrub raw C++ engine text; OOM → retryable 503 + poison-flag.
        Err(e) => return inference_error_from_text(mm, model, &e),
    };

    // gap-2: run extract_thinking only when the model is configured (or
    // auto-detected) as a reasoning model. Non-thinking models skip the scan
    // entirely — no false-positive strip if a model emits <think> as literal
    // content, and no wasted regex pass on every Mistral/Whisper/SDXL response.
    // `starts_in_thinking`: the prompt prefilled an open `<think>`,
    // so tagless output is reasoning, not answer — parity with streaming.
    // gpt-oss doesn't emit `<think>` at all (harmony channels instead) — routed
    // to `extract_reasoning_gpt_oss` so its analysis/final split actually fires.
    let (reasoning, answer) = match reasoning_parser {
        Some(ReasoningParser::GptOss) => extract_reasoning_gpt_oss(&drained.content),
        Some(_) => extract_thinking(&drained.content, starts_in_thinking),
        None => (None, drained.content.clone()),
    };

    let body = ChatCompletion {
        id,
        object: "chat.completion",
        created,
        model,
        choices: vec![CompletionChoice {
            index: 0,
            message: ResponseMessage {
                role: "assistant",
                content: Some(answer),
                reasoning_content: reasoning,
                tool_calls: None,
            },
            finish_reason: drained.finish.as_openai(),
        }],
        usage: usage_for(drained.prompt_tokens, drained.completion_tokens),
        model_family: meta.family.label(),
        sampling: meta.sampling(),
    };

    Json(body).into_response()
}

/// Buffered tool-calling collector for requests that supplied `tools`.
///
/// Drains the full generation, strips `<think>`, then parses tool calls in the
/// model's [`ModelFamily`] dialect. When calls are found the message carries
/// `content: null` + `tool_calls` and `finish_reason: "tool_calls"`; otherwise
/// it is a normal text completion. Emits a `chat.completion` JSON object when
/// `stream == false`, or a single buffered SSE chunk + finish + `[DONE]` when
/// `stream == true`.
// Eight genuinely-distinct buffered-shaper inputs (channel, ids, family, stream
// flags, and the manager for T6.2 poison-flagging); a params struct would only
// add ceremony for a single internal call site.
#[allow(clippy::too_many_arguments)]
async fn collect_tool_completion(
    rx: TokenReceiver,
    id: &str,
    model: &str,
    created: u64,
    meta: ResponseMeta,
    stream: bool,
    include_usage: bool,
    mm: Option<&Arc<ModelManager>>,
    reasoning_parser: Option<ReasoningParser>,
    starts_in_thinking: bool,
) -> Response {
    let drained = match drain_channel(rx).await {
        Ok(d) => d,
        // T6.2: scrub raw C++ engine text; OOM → retryable 503 + poison-flag.
        Err(e) => return inference_error_from_text(mm, model, &e),
    };

    // Strip reasoning before tool-call parsing, but only for reasoning models
    // (gap-2). When reasoning_parser is None, pass the full content to
    // parse_tool_calls unchanged — the tool-call content never contains <think>.
    // `starts_in_thinking`: tagless output from a prefilled `<think>`
    // is all reasoning — parse_tool_calls then sees the placeholder and finds
    // nothing, which is correct (a model that never exited thinking never
    // emitted a post-think tool call). gpt-oss: see collect_completion above —
    // same GptOss/other/None split, same reasoning.
    let (thinking, answer) = match reasoning_parser {
        Some(ReasoningParser::GptOss) => extract_reasoning_gpt_oss(&drained.content),
        Some(_) => extract_thinking(&drained.content, starts_in_thinking),
        None => (None, drained.content.clone()),
    };
    // gpt-oss's tool call lives inside the `commentary` channel, which
    // `HarmonyFilter` routes into `thinking` alongside any genuine analysis
    // reasoning (see extract_reasoning_gpt_oss) — scan there, not `answer`
    // (the `final`-channel text, real regardless of whether a tool call is
    // also present, so left untouched). Every other family's tool-call
    // syntax lives inside the answer itself, scanned as before.
    let (tool_calls, remaining, thinking) = if meta.family == ModelFamily::GptOss {
        let (tool_calls, cleaned_thinking) =
            parse_tool_calls(thinking.as_deref().unwrap_or(""), meta.family);
        let thinking = (!cleaned_thinking.trim().is_empty()).then_some(cleaned_thinking);
        (tool_calls, answer, thinking)
    } else {
        let (tool_calls, remaining) = parse_tool_calls(&answer, meta.family);
        (tool_calls, remaining, thinking)
    };

    // A tool call overrides the engine's stop/length finish reason.
    let finish_reason: &'static str = if tool_calls.is_some() {
        "tool_calls"
    } else {
        drained.finish.as_openai()
    };

    if stream {
        // This path is fully buffered (the whole generation is already drained
        // and split), so — unlike the incremental token stream — we have the
        // complete reasoning here and surface it as `reasoning_content`, at
        // parity with the non-stream branch below.
        tool_completion_sse(
            id,
            model,
            created,
            tool_calls,
            remaining,
            thinking,
            finish_reason,
            include_usage,
            drained.prompt_tokens,
            drained.completion_tokens,
            meta,
        )
    } else {
        tool_completion_json(
            id,
            model,
            created,
            tool_calls,
            remaining,
            finish_reason,
            drained.prompt_tokens,
            drained.completion_tokens,
            thinking,
            meta,
        )
    }
}

/// Shape a buffered tool result as a non-streaming `chat.completion` JSON.
// Eight distinct response fields, each independent — bundling them into a
// struct would add indirection without clarifying this private shaper.
#[allow(clippy::too_many_arguments)]
fn tool_completion_json(
    id: &str,
    model: &str,
    created: u64,
    tool_calls: Option<Vec<ToolCall>>,
    remaining: String,
    finish_reason: &'static str,
    prompt_tokens: usize,
    completion_tokens: usize,
    reasoning: Option<String>,
    meta: ResponseMeta,
) -> Response {
    // With tool calls: content is explicit null. Otherwise: the text answer.
    // reasoning_content carries the stripped <think> block in either case.
    let message = if tool_calls.is_some() {
        ResponseMessage {
            role: "assistant",
            content: None,
            reasoning_content: reasoning,
            tool_calls,
        }
    } else {
        ResponseMessage {
            role: "assistant",
            content: Some(remaining),
            reasoning_content: reasoning,
            tool_calls: None,
        }
    };

    let body = ChatCompletion {
        id,
        object: "chat.completion",
        created,
        model,
        choices: vec![CompletionChoice {
            index: 0,
            message,
            finish_reason,
        }],
        usage: usage_for(prompt_tokens, completion_tokens),
        model_family: meta.family.label(),
        sampling: meta.sampling(),
    };

    Json(body).into_response()
}

/// Shape a buffered tool result as a single SSE payload chunk, a finish chunk,
/// an optional usage chunk (when `include_usage`), and the `[DONE]` sentinel.
#[allow(clippy::too_many_arguments)]
fn tool_completion_sse(
    id: &str,
    model: &str,
    created: u64,
    tool_calls: Option<Vec<ToolCall>>,
    remaining: String,
    reasoning: Option<String>,
    finish_reason: &'static str,
    include_usage: bool,
    prompt_tokens: usize,
    compl_tokens: usize,
    meta: ResponseMeta,
) -> Response {
    // Payload chunk: role + reasoning_content + (tool_calls) or (content),
    // finish_reason null. Each streaming tool call carries its array `index`
    // (OpenAI requirement) — without it openai-python mis-merges the call.
    let payload_delta = if let Some(calls) = tool_calls {
        let indexed = calls
            .into_iter()
            .enumerate()
            .map(|(i, call)| StreamToolCall {
                index: u32::try_from(i).unwrap_or(u32::MAX),
                call,
            })
            .collect();
        Delta {
            role: Some("assistant"),
            reasoning_content: reasoning,
            tool_calls: Some(indexed),
            ..Delta::default()
        }
    } else {
        Delta {
            role: Some("assistant"),
            reasoning_content: reasoning,
            content: Some(remaining),
            ..Delta::default()
        }
    };

    let payload = ChatChunk {
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: vec![ChunkChoice {
            index: 0,
            delta: payload_delta,
            finish_reason: None,
        }],
        model_family: meta.family.label(),
        sampling: meta.sampling(),
    };
    let finish = ChatChunk {
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta::default(),
            finish_reason: Some(finish_reason),
        }],
        model_family: meta.family.label(),
        sampling: meta.sampling(),
    };

    let mut frames: Vec<Result<Event, Infallible>> = vec![
        Ok(Event::default().data(serde_json::to_string(&payload).unwrap_or_default())),
        Ok(Event::default().data(serde_json::to_string(&finish).unwrap_or_default())),
    ];
    if include_usage {
        frames.push(Ok(usage_sse_event(
            id,
            "chat.completion.chunk",
            model,
            created,
            prompt_tokens,
            compl_tokens,
        )));
    }
    frames.push(Ok(Event::default().data("[DONE]")));

    Sse::new(futures_util::stream::iter(frames))
        .keep_alive(KeepAlive::default())
        .into_response()
}

// ---- Response helper ------------------------------------------------

/// Extension trait to call `.into_response()` on tuples and simple types.
/// Already provided by axum — imported here for clarity.
use axum::response::IntoResponse;

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    // ---- R1 capability-routing harness ------------------------------------
    //
    // Builds an `AppState` backed by a `MockEngineFactory` ModelManager (no
    // GPU). The mock classifies any model id containing "vlm" as Vision, so the
    // routing matrix can be exercised end-to-end through `chat_completions`.

    use crate::model_manager::ModelManager;
    use crate::model_manager::lifecycle::MockEngineFactory;
    use std::sync::Arc;

    /// Default `ResponseMeta` for tests that don't care about `model_family`/
    /// effective-sampling reporting itself — `ModelFamily::Default`, no
    /// sampling params resolved. Tests that specifically exercise those
    /// fields build a `ResponseMeta` directly instead of using this.
    fn test_meta() -> ResponseMeta {
        ResponseMeta {
            family: ModelFamily::Default,
            temperature: None,
            top_p: None,
            top_k: None,
        }
    }

    // ---- T2.1 shared L0 prompt-length gate ---------------------------------

    /// Spawn a fake engine thread that answers every `Tokenize` command with
    /// `n_tokens` ids, and return a handle wired to it.
    fn handle_with_token_count(n_tokens: usize) -> EngineHandle {
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        tokio::spawn(async move {
            while let Some(cmd) = rx.recv().await {
                if let crate::cb_engine::EngineCommand::Tokenize { reply, .. } = cmd {
                    let _ = reply.send(Ok(vec![0i64; n_tokens]));
                }
            }
        });
        EngineHandle::from_sender(tx)
    }

    /// A grossly oversized prompt must be rejected by the byte-length
    /// pre-check before `tokenize` is ever called — proven with a handle
    /// whose channel is already closed, so `tokenize` can only ever fail open
    /// (`Ok(None)`, never a 400). Seeing a 400 here means the byte gate, not
    /// the tokenizer, produced it.
    #[tokio::test]
    async fn gate_prompt_rejects_oversized_prompt_before_tokenizing() {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        drop(rx); // any call to tokenize() on this handle fails immediately
        let handle = EngineHandle::from_sender(tx);
        // max_prompt_tokens=10 -> byte ceiling is 10*32=320 bytes.
        let huge_prompt = "a".repeat(1_000);
        let resp = *gate_prompt(&handle, "test-model", &huge_prompt, 10)
            .await
            .unwrap_err();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "context_length_exceeded");
    }

    /// Over-limit prompt → 400 `context_length_exceeded`, never submitted.
    #[tokio::test]
    async fn gate_prompt_rejects_over_limit_prompt() {
        let handle = handle_with_token_count(101);
        let resp = *gate_prompt(&handle, "test-model", "long prompt", 100)
            .await
            .unwrap_err();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "context_length_exceeded");
    }

    /// Within-limit prompt passes the gate AND returns the token ids so
    /// `add_request` can reuse them (T7.3 — encode once). The forwarded ids
    /// must match the count the gate tokenized.
    #[tokio::test]
    async fn gate_prompt_passes_within_limit_and_forwards_ids() {
        let handle = handle_with_token_count(100);
        let ids = gate_prompt(&handle, "test-model", "ok prompt", 100)
            .await
            .unwrap();
        assert_eq!(ids.map(|v| v.len()), Some(100));
    }

    /// A real gate rejection increments the observability counter end to
    /// end — closes the gap where a clean 400 `context_length_exceeded` was
    /// fully visible to the client but invisible server-side (`dev/autotest/
    /// 20260823_l0_gate_rejection_observability_gap.md`). Exercises
    /// `gate_prompt_bytes` directly (sync, unlike the full `gate_prompt`)
    /// since `metrics::with_local_recorder`'s thread-local scoping can't
    /// safely wrap an `.await` that hops onto a background `tokio::spawn`
    /// task the way the tokenizer round-trip in the async gates does.
    #[test]
    fn gate_rejection_increments_context_length_exceeded_counter() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        let recorder = PrometheusBuilder::new().build_recorder();
        let prom_handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            gate_prompt_bytes("over-limit-model", "cb", &"a".repeat(1_000), 10)
        })
        .unwrap_err();
        let rendered = prom_handle.render();
        assert!(
            rendered.contains(
                "rustedvino_context_length_exceeded_total{model=\"over-limit-model\",gate=\"cb\"} 1"
            ),
            "a real gate rejection must increment the counter:\n{rendered}"
        );
    }

    /// `max_prompt_tokens == 0` disables the gate; a tokenize failure fails
    /// open (transient tokenizer trouble must not become a false 400). Both
    /// cases return `None` so the engine falls back to tokenizing the string
    /// (no ids to forward).
    #[tokio::test]
    async fn gate_prompt_fails_open_when_disabled_or_tokenizer_errors() {
        // Disabled gate: a dead engine channel would error if consulted, but
        // max == 0 short-circuits first.
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        let dead = EngineHandle::from_sender(tx);
        assert_eq!(
            gate_prompt(&dead, "test-model", "p", 0).await.unwrap(),
            None
        );

        // Enabled gate + tokenize error (engine gone) → fail open.
        let (tx2, rx2) = tokio::sync::mpsc::channel(1);
        drop(rx2);
        let dead2 = EngineHandle::from_sender(tx2);
        assert_eq!(
            gate_prompt(&dead2, "test-model", "p", 100).await.unwrap(),
            None
        );
    }

    // ---- gate_vlm_prompt (VLM L0 gate — the fix for the GPU-poisoning bug) --

    fn vlm_with_token_count(n_tokens: usize) -> crate::vlm_engine::VlmHandle {
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        tokio::spawn(async move {
            while let Some(cmd) = rx.recv().await {
                if let crate::vlm_engine::VlmCommand::CountTokens { reply, .. } = cmd {
                    let _ = reply.send(Ok(n_tokens));
                }
            }
        });
        crate::vlm_engine::VlmHandle::from_sender(tx, "test-vlm")
    }

    /// A grossly oversized VLM prompt must be rejected by the byte-length
    /// pre-check before `count_tokens` is ever called — same proof shape as
    /// `gate_prompt_rejects_oversized_prompt_before_tokenizing`: a handle
    /// whose channel is already closed can only ever fail open, so a 400 here
    /// can only have come from the byte gate.
    #[tokio::test]
    async fn gate_vlm_prompt_rejects_oversized_prompt_before_counting() {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        drop(rx);
        let dead = crate::vlm_engine::VlmHandle::from_sender(tx, "dead-vlm");
        // max_prompt_tokens=10 -> byte ceiling is 10*32=320 bytes.
        let huge_prompt = "a".repeat(1_000);
        let resp = *gate_vlm_prompt(&dead, &huge_prompt, 10).await.unwrap_err();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "context_length_exceeded");
    }

    /// Over-limit VLM prompt → 400 `context_length_exceeded`, same envelope as
    /// the CB path's `gate_prompt` — this is the exact case that previously
    /// reached `vlm.generate` unchecked and poisoned the GPU context.
    #[tokio::test]
    async fn gate_vlm_prompt_rejects_over_limit_prompt() {
        let vlm = vlm_with_token_count(75_000);
        let resp = *gate_vlm_prompt(&vlm, "a very long document", 49_932)
            .await
            .unwrap_err();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "context_length_exceeded");
    }

    /// Within-limit VLM prompt passes the gate silently.
    #[tokio::test]
    async fn gate_vlm_prompt_passes_within_limit() {
        let vlm = vlm_with_token_count(100);
        assert!(gate_vlm_prompt(&vlm, "ok prompt", 49_932).await.is_ok());
    }

    /// Same fail-open policy as `gate_prompt`: disabled gate (`max == 0`) and a
    /// dead engine (tokenize errors) both let the request through rather than
    /// producing a false 400.
    #[tokio::test]
    async fn gate_vlm_prompt_fails_open_when_disabled_or_engine_errors() {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        let dead = crate::vlm_engine::VlmHandle::from_sender(tx, "dead-vlm");
        assert!(gate_vlm_prompt(&dead, "p", 0).await.is_ok());

        let (tx2, rx2) = tokio::sync::mpsc::channel(1);
        drop(rx2);
        let dead2 = crate::vlm_engine::VlmHandle::from_sender(tx2, "dead-vlm-2");
        assert!(gate_vlm_prompt(&dead2, "p", 100).await.is_ok());
    }

    // ---- NPU L0 prompt-length gate (dev/ovms-gap.md #6) --------------------

    fn npu_with_token_count(n_tokens: usize) -> crate::npu_engine::NpuHandle {
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        tokio::spawn(async move {
            while let Some(cmd) = rx.recv().await {
                if let crate::npu_engine::NpuCommand::CountTokens { reply, .. } = cmd {
                    let _ = reply.send(Ok(n_tokens));
                }
            }
        });
        crate::npu_engine::NpuHandle::from_sender(tx, "test-npu")
    }

    /// A grossly oversized NPU prompt must be rejected by the byte-length
    /// pre-check before `count_tokens` is ever called — same proof shape as
    /// `gate_vlm_prompt_rejects_oversized_prompt_before_counting`.
    #[tokio::test]
    async fn gate_npu_prompt_rejects_oversized_prompt_before_counting() {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        drop(rx);
        let dead = crate::npu_engine::NpuHandle::from_sender(tx, "dead-npu");
        // max_prompt_len=10 -> byte ceiling is 10*32=320 bytes.
        let huge_prompt = "a".repeat(1_000);
        let resp = *gate_npu_prompt(&dead, &huge_prompt, 10).await.unwrap_err();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "context_length_exceeded");
    }

    /// Over-limit NPU prompt → 400 `context_length_exceeded` — the clean-error
    /// fix for the bug found 2026-07-16 (a too-long prompt used to reach
    /// `npu.generate` unchecked and surface `OpenVINO`'s raw exception text).
    #[tokio::test]
    async fn gate_npu_prompt_rejects_over_limit_prompt() {
        let npu = npu_with_token_count(1_061);
        let resp = *gate_npu_prompt(&npu, "a chat-templated prompt", 1_024)
            .await
            .unwrap_err();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "context_length_exceeded");
    }

    /// Within-limit NPU prompt passes the gate silently.
    #[tokio::test]
    async fn gate_npu_prompt_passes_within_limit() {
        let npu = npu_with_token_count(100);
        assert!(gate_npu_prompt(&npu, "ok prompt", 1_024).await.is_ok());
    }

    /// A dead engine (tokenize errors) fails open rather than producing a
    /// false 400 — same policy as `gate_prompt`/`gate_vlm_prompt`.
    #[tokio::test]
    async fn gate_npu_prompt_fails_open_when_engine_errors() {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        let dead = crate::npu_engine::NpuHandle::from_sender(tx, "dead-npu");
        assert!(gate_npu_prompt(&dead, "p", 1_024).await.is_ok());
    }

    // ---- T7.1/T2.2 shared max_tokens clamp ---------------------------------

    /// The server ceiling bounds explicit, absent, and absurd budgets alike;
    /// `cap == 0` is the explicit opt-out.
    #[test]
    fn clamp_max_tokens_bounds_every_budget_shape() {
        // Explicit budget under the cap: untouched.
        assert_eq!(clamp_max_tokens(Some(100), 0, 8192), 100);
        // Slot-pinning abuse (finding 8): clamped to the ceiling.
        assert_eq!(clamp_max_tokens(Some(u32::MAX), 0, 8192), 8192);
        // Absent budget + chat default (0 = engine default): still bounded.
        assert_eq!(clamp_max_tokens(None, 0, 8192), 8192);
        // Absent budget + legacy default 16: the OpenAI default survives.
        assert_eq!(clamp_max_tokens(None, 16, 8192), 16);
        // cap == 0: explicit opt-out, budgets pass through unchanged.
        assert_eq!(clamp_max_tokens(Some(1_000_000), 0, 0), 1_000_000);
        assert_eq!(clamp_max_tokens(None, 0, 0), 0);
    }

    /// The repetition-penalty safety net (2026-07-14, found live —
    /// a VLM chat looped generating "kaczki" forever with no anti-repeat
    /// sampling applied): absent → the server default; an explicit value,
    /// including the opt-out `1.0`, always wins.
    #[test]
    fn resolve_repetition_penalty_applies_default_only_when_absent() {
        assert_eq!(
            resolve_repetition_penalty(None),
            Some(DEFAULT_REPETITION_PENALTY)
        );
        assert_eq!(
            resolve_repetition_penalty(Some(1.0)),
            Some(1.0),
            "explicit opt-out wins"
        );
        assert_eq!(
            resolve_repetition_penalty(Some(1.3)),
            Some(1.3),
            "explicit value wins"
        );
    }

    /// `resolve_sampling_defaults` (2026-07-23): composes client-explicit
    /// values, `tools_active`, and a model's own `GenerationDefaults`. The
    /// all-or-nothing gate is load-bearing — a `top_p`/`top_k` collision with
    /// an explicit client `temperature` was a real bug caught earlier this
    /// session (the project's internal engineering log), so the "client set exactly one of the
    /// three" cases below are the regression tests for that bug class, not
    /// just coverage padding.
    #[test]
    fn resolve_sampling_defaults_all_or_nothing_gate() {
        let defaults = GenerationDefaults {
            temperature: Some(0.7),
            top_p: Some(0.8),
            top_k: Some(20),
        };

        // Client sets none, tools inactive → model defaults apply as a package.
        assert_eq!(
            resolve_sampling_defaults(None, None, None, false, defaults),
            (Some(0.7), Some(0.8), Some(20)),
            "no client sampling intent, no tools → model defaults"
        );

        // Client sets none, tools active → greedy pin, model defaults suppressed.
        assert_eq!(
            resolve_sampling_defaults(None, None, None, true, defaults),
            (Some(TOOLS_ACTIVE_TEMPERATURE), None, None),
            "no client sampling intent, tools active → greedy pin wins over model defaults"
        );

        // Client sets temperature only (even the bug-triggering 0.0) → nothing
        // else gets filled in, regardless of tools_active.
        assert_eq!(
            resolve_sampling_defaults(Some(0.0), None, None, false, defaults),
            (Some(0.0), None, None),
            "explicit temperature:0.0 alone must not pull in model top_p/top_k"
        );
        assert_eq!(
            resolve_sampling_defaults(Some(0.0), None, None, true, defaults),
            (Some(0.0), None, None),
            "explicit temperature wins over the tools_active pin too"
        );

        // Client sets top_p only → nothing else filled in (this is the exact
        // shape of the collision bug: an explicit top_p with no temperature
        // must never come back paired with a filled-in temperature).
        assert_eq!(
            resolve_sampling_defaults(None, Some(0.9), None, false, defaults),
            (None, Some(0.9), None),
            "explicit top_p alone must not pull in model temperature/top_k"
        );

        // Client sets top_k only → same rule.
        assert_eq!(
            resolve_sampling_defaults(None, None, Some(40), false, defaults),
            (None, None, Some(40)),
            "explicit top_k alone must not pull in model temperature/top_p"
        );

        // The literal 2026-07-23 incident shape: client sets top_p (or top_k)
        // only, AND tools are active. The reverted bug gated on
        // `temperature.is_none() && tools_active` alone, which would have
        // clobbered this exact case with the greedy pin. These two assert the
        // "client set at least one field" branch still wins even with
        // tools_active — the passthrough must never mix with the pin.
        assert_eq!(
            resolve_sampling_defaults(None, Some(0.9), None, true, defaults),
            (None, Some(0.9), None),
            "explicit top_p alone must win over the tools_active pin too"
        );
        assert_eq!(
            resolve_sampling_defaults(None, None, Some(40), true, defaults),
            (None, None, Some(40)),
            "explicit top_k alone must win over the tools_active pin too"
        );

        // Client sets all three → pure passthrough.
        assert_eq!(
            resolve_sampling_defaults(Some(1.2), Some(0.95), Some(50), false, defaults),
            (Some(1.2), Some(0.95), Some(50)),
            "all three explicit → passthrough, model defaults never consulted"
        );

        // No model defaults available (media model / unloaded / sanity-check
        // failed) → client-set-none resolves to all-None when tools inactive.
        assert_eq!(
            resolve_sampling_defaults(None, None, None, false, GenerationDefaults::default()),
            (None, None, None),
            "no model defaults + no client intent + no tools → all None, today's behavior"
        );
    }

    /// T7.4/#24: sampling params in range (or absent) pass; out-of-range → 400.
    #[test]
    fn validate_sampling_params_rejects_out_of_range() {
        // All absent → accepted (engine keeps its defaults).
        assert!(validate_sampling_params(None, None, None, None, None).is_none());
        // Boundary values (inclusive) on every param → accepted.
        assert!(
            validate_sampling_params(Some(0.0), Some(0.0), Some(-2.0), Some(-2.0), Some(1))
                .is_none()
        );
        assert!(
            validate_sampling_params(Some(2.0), Some(1.0), Some(2.0), Some(2.0), None).is_none()
        );
        // n > 1 is NOT this function's concern (it is an unsupported_parameter).
        assert!(validate_sampling_params(None, None, None, None, Some(5)).is_none());

        // Each out-of-range value (one at a time) → 400, including ±inf from an
        // over-large JSON literal and the silent n:0 → 1 trap.
        for resp in [
            validate_sampling_params(Some(2.5), None, None, None, None),
            validate_sampling_params(Some(-0.1), None, None, None, None),
            validate_sampling_params(Some(f32::INFINITY), None, None, None, None),
            validate_sampling_params(None, Some(1.5), None, None, None),
            validate_sampling_params(None, None, Some(3.0), None, None),
            validate_sampling_params(None, None, None, Some(-3.0), None),
            validate_sampling_params(None, None, None, None, Some(0)),
        ] {
            assert_eq!(resp.unwrap().status(), StatusCode::BAD_REQUEST);
        }
    }

    // ---- G1 response_format → json_schema mapping -------------------------

    /// Deserialize a `response_format` JSON value into the typed struct, the way
    /// serde does when parsing a `ChatRequest`.
    fn rf(v: serde_json::Value) -> ResponseFormat {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn response_format_absent_or_text_is_free_form() {
        assert!(json_schema_from_response_format(None).unwrap().is_none());
        let text = rf(serde_json::json!({"type": "text"}));
        assert!(
            json_schema_from_response_format(Some(&text))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn response_format_json_object_is_permissive_object_schema() {
        let obj = rf(serde_json::json!({"type": "json_object"}));
        let schema = json_schema_from_response_format(Some(&obj)).unwrap();
        assert_eq!(schema.as_deref(), Some(r#"{"type":"object"}"#));
    }

    #[test]
    fn response_format_json_schema_extracts_inner_schema() {
        let inner = serde_json::json!({"type": "object", "properties": {"x": {"type": "integer"}}});
        let req = rf(serde_json::json!({
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": inner.clone()}
        }));
        let schema = json_schema_from_response_format(Some(&req))
            .unwrap()
            .unwrap();
        // The extracted schema must round-trip to the same JSON value the client sent.
        let parsed: serde_json::Value = serde_json::from_str(&schema).unwrap();
        assert_eq!(parsed, inner);
    }

    #[test]
    fn response_format_json_schema_without_schema_is_rejected() {
        // Missing the inner `schema` object → 400, not a silent free-form fall-through.
        let req = rf(serde_json::json!({"type": "json_schema", "json_schema": {"name": "x"}}));
        let err = json_schema_from_response_format(Some(&req)).unwrap_err();
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn response_format_unknown_type_is_rejected() {
        let req = rf(serde_json::json!({"type": "yaml"}));
        let err = json_schema_from_response_format(Some(&req)).unwrap_err();
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    }

    // ---- G2 tool_choice parsing + validation ------------------------------

    /// A `tools` array of `{"type":"function","function":{"name":N}}` objects.
    fn tools_with(names: &[&str]) -> Vec<serde_json::Value> {
        names
            .iter()
            .map(|n| serde_json::json!({"type": "function", "function": {"name": n}}))
            .collect()
    }

    #[test]
    fn tool_choice_absent_is_auto() {
        assert_eq!(parse_tool_choice(None, None).unwrap(), ToolChoice::Auto);
    }

    #[test]
    fn tool_choice_string_auto_and_none() {
        let auto = serde_json::json!("auto");
        let none = serde_json::json!("none");
        assert_eq!(
            parse_tool_choice(Some(&auto), None).unwrap(),
            ToolChoice::Auto
        );
        assert_eq!(
            parse_tool_choice(Some(&none), None).unwrap(),
            ToolChoice::None
        );
    }

    #[test]
    fn tool_choice_required_with_tools_is_required() {
        let req = serde_json::json!("required");
        let tools = tools_with(&["get_weather"]);
        assert_eq!(
            parse_tool_choice(Some(&req), Some(&tools)).unwrap(),
            ToolChoice::Required
        );
    }

    #[test]
    fn tool_choice_required_without_tools_is_rejected() {
        let req = serde_json::json!("required");
        let err = parse_tool_choice(Some(&req), None).unwrap_err();
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
        // Empty tools array is also "no tools".
        let empty: Vec<serde_json::Value> = vec![];
        let err = parse_tool_choice(Some(&req), Some(&empty)).unwrap_err();
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn tool_choice_named_function_in_tools_is_named() {
        let req = serde_json::json!({"type": "function", "function": {"name": "get_weather"}});
        let tools = tools_with(&["get_time", "get_weather"]);
        assert_eq!(
            parse_tool_choice(Some(&req), Some(&tools)).unwrap(),
            ToolChoice::Named("get_weather".to_owned())
        );
    }

    #[test]
    fn tool_choice_named_function_not_in_tools_is_rejected() {
        let req = serde_json::json!({"type": "function", "function": {"name": "nope"}});
        let tools = tools_with(&["get_weather"]);
        let err = parse_tool_choice(Some(&req), Some(&tools)).unwrap_err();
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn tool_choice_named_function_without_tools_is_rejected() {
        let req = serde_json::json!({"type": "function", "function": {"name": "get_weather"}});
        let err = parse_tool_choice(Some(&req), None).unwrap_err();
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn tool_choice_unknown_string_is_rejected() {
        let req = serde_json::json!("banana");
        let err = parse_tool_choice(Some(&req), None).unwrap_err();
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn tool_choice_malformed_object_is_rejected() {
        // Object without function.name → 400 (not a silent fall-through to auto).
        let req = serde_json::json!({"type": "function"});
        let tools = tools_with(&["get_weather"]);
        let err = parse_tool_choice(Some(&req), Some(&tools)).unwrap_err();
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn tool_choice_wrong_json_type_is_rejected() {
        // A bare number is neither a known string nor a function object.
        let req = serde_json::json!(7);
        let err = parse_tool_choice(Some(&req), None).unwrap_err();
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn tool_choice_injects_tools_only_when_not_none() {
        assert!(ToolChoice::Auto.injects_tools());
        assert!(ToolChoice::Required.injects_tools());
        assert!(ToolChoice::Named("f".to_owned()).injects_tools());
        assert!(!ToolChoice::None.injects_tools());
    }

    #[test]
    fn tool_choice_is_forcing_only_for_required_and_named() {
        assert!(ToolChoice::Required.is_forcing());
        assert!(ToolChoice::Named("f".to_owned()).is_forcing());
        assert!(!ToolChoice::Auto.is_forcing());
        assert!(!ToolChoice::None.is_forcing());
    }

    #[test]
    fn tool_choice_steer_only_for_forcing_variants() {
        assert!(ToolChoice::Auto.steer().is_none());
        assert!(ToolChoice::None.steer().is_none());
        assert!(ToolChoice::Required.steer().is_some());
        // Named steering must mention the function name.
        let s = ToolChoice::Named("get_weather".to_owned()).steer().unwrap();
        assert!(s.contains("get_weather"), "steer must name the tool: {s}");
    }

    // ---- Unsupported-param 400 policy (chat) ------------------------------

    /// Deserialize a full `ChatRequest` from JSON the way the handler receives it.
    fn cr(v: serde_json::Value) -> ChatRequest {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn chat_clean_request_has_no_unsupported_param() {
        let req = cr(serde_json::json!({"model": "m", "messages": []}));
        assert!(unsupported_chat_param(&req).is_none());
    }

    #[test]
    fn chat_n_greater_than_one_is_rejected() {
        let req = cr(serde_json::json!({"model": "m", "messages": [], "n": 2}));
        assert_eq!(
            unsupported_chat_param(&req).unwrap().status(),
            StatusCode::BAD_REQUEST
        );
        // n == 1 (the served default) is fine.
        let ok = cr(serde_json::json!({"model": "m", "messages": [], "n": 1}));
        assert!(unsupported_chat_param(&ok).is_none());
    }

    #[test]
    fn chat_logprobs_true_is_rejected_but_false_is_ok() {
        let bad = cr(serde_json::json!({"model": "m", "messages": [], "logprobs": true}));
        assert_eq!(
            unsupported_chat_param(&bad).unwrap().status(),
            StatusCode::BAD_REQUEST
        );
        let ok = cr(serde_json::json!({"model": "m", "messages": [], "logprobs": false}));
        assert!(unsupported_chat_param(&ok).is_none());
    }

    #[test]
    fn chat_top_logprobs_present_is_rejected() {
        let req = cr(serde_json::json!({"model": "m", "messages": [], "top_logprobs": 5}));
        assert_eq!(
            unsupported_chat_param(&req).unwrap().status(),
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn chat_nonempty_logit_bias_is_rejected_but_empty_is_ok() {
        let bad =
            cr(serde_json::json!({"model": "m", "messages": [], "logit_bias": {"123": -100}}));
        assert_eq!(
            unsupported_chat_param(&bad).unwrap().status(),
            StatusCode::BAD_REQUEST
        );
        // An empty map biases nothing → accepted.
        let ok = cr(serde_json::json!({"model": "m", "messages": [], "logit_bias": {}}));
        assert!(unsupported_chat_param(&ok).is_none());
    }

    fn routing_config(models: &[(&str, f64)]) -> crate::model_manager::Config {
        crate::model_manager::Config {
            models_dir: std::path::PathBuf::from("/tmp/test-models"),
            device: "CPU".to_owned(),
            preload: vec![],
            models: models
                .iter()
                .map(|(id, gb)| {
                    (
                        (*id).to_owned(),
                        crate::model_manager::config::ModelEntry {
                            vram_gb: *gb,
                            kind: None,
                            policy: crate::model_manager::config::ModelPolicy::default(),
                        },
                    )
                })
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
            max_prompt_array: 16,              // the production default
            max_tokens_cap: 8192,              // the production default
            bind_addr: "127.0.0.1".to_owned(), // loopback — irrelevant for unit tests
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

    /// Build an `AppState` with the given models loaded Ready (via the mock).
    async fn routed_state(models: &[(&str, f64)]) -> AppState {
        let mm = ModelManager::new(
            routing_config(models),
            Arc::new(MockEngineFactory::default()),
        )
        .await
        .unwrap();
        for (id, _) in models {
            mm.load_model(id).await.unwrap();
        }
        AppState::new().with_model_manager(Arc::new(mm))
    }

    /// Deserialize a `ChatRequest` from a JSON literal (all fields default-filled).
    fn chat_req(v: serde_json::Value) -> ChatRequest {
        serde_json::from_value(v).unwrap()
    }

    /// 2026.2 migration (PLAN step 6): a VLM now SERVES text-only requests. The
    /// upstream hard assert in `inputs_embedder.cpp` ("Missing image/video with
    /// index 0") was fixed in OV 2026.2, so the 400-guard is gone —
    /// switching models mid-chat in a client like `AnythingLLM` (history messages
    /// with no image) no longer 400s. Verified live on B50/2026.2.
    #[tokio::test]
    async fn text_only_request_on_vision_model_is_served() {
        let state = routed_state(&[("qwen-vlm-8b", 5.5)]).await;
        let req = chat_req(serde_json::json!({
            "model": "qwen-vlm-8b",
            "messages": [{"role": "user", "content": "describe the weather"}]
        }));
        let resp = chat_completions(axum::extract::State(state), JsonBody(req)).await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "a VLM must serve text-only requests on 2026.2"
        );
    }

    /// T7.4/#25: empty `messages` is a 400 on BOTH the VLM and the text path —
    /// the upfront guard fires before routing, so the VLM no longer falls through
    /// to its misleading "requires at least one image" error.
    #[tokio::test]
    async fn empty_messages_returns_400_on_both_paths() {
        for model in ["qwen-vlm-8b", "qwen3-8b"] {
            let state = routed_state(&[(model, 5.5)]).await;
            let req = chat_req(serde_json::json!({ "model": model, "messages": [] }));
            let resp = chat_completions(axum::extract::State(state), JsonBody(req)).await;
            assert_eq!(
                resp.status(),
                StatusCode::BAD_REQUEST,
                "empty messages must 400 on the {model} path",
            );
        }
    }

    /// T7.4/#24: an out-of-range sampling param 400s before any inference, on the
    /// real routed path (not just the pure validator).
    #[tokio::test]
    async fn out_of_range_temperature_returns_400() {
        let state = routed_state(&[("qwen3-8b", 5.5)]).await;
        let req = chat_req(serde_json::json!({
            "model": "qwen3-8b",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 5.0
        }));
        let resp = chat_completions(axum::extract::State(state), JsonBody(req)).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// Image content on a text-generation model is rejected with 400.
    #[tokio::test]
    async fn image_request_on_text_model_returns_400() {
        let state = routed_state(&[("qwen3-8b", 5.5)]).await;
        let req = chat_req(serde_json::json!({
            "model": "qwen3-8b",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}}
            ]}]
        }));
        let resp = chat_completions(axum::extract::State(state), JsonBody(req)).await;
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "a text model must reject image content"
        );
    }

    /// A valid 2×2 PNG encoded as a `data:` URI — so the VLM image path decodes
    /// it instead of 400-ing on a corrupt payload.
    fn tiny_png_data_uri() -> String {
        use base64ct::{Base64, Encoding};
        use image::{DynamicImage, ImageFormat, RgbImage};
        let img = DynamicImage::ImageRgb8(RgbImage::from_fn(2, 2, |_, _| image::Rgb([10, 20, 30])));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), ImageFormat::Png)
            .unwrap();
        format!("data:image/png;base64,{}", Base64::encode_string(&buf))
    }

    /// Image content on a VLM is routed to the vision engine and served.
    #[tokio::test]
    async fn image_request_on_vision_model_is_served() {
        let state = routed_state(&[("qwen-vlm-8b", 5.5)]).await;
        let req = chat_req(serde_json::json!({
            "model": "qwen-vlm-8b",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": tiny_png_data_uri()}}
            ]}]
        }));
        let resp = chat_completions(axum::extract::State(state), JsonBody(req)).await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "a VLM must serve image requests"
        );
    }

    /// End-to-end: a VLM request with `tools` active gets its `tools` array
    /// all the way to the engine (the mock VLM echoes a `<tool_call>` block
    /// only when it receives a non-empty `tools`) and the response comes back
    /// OpenAI-shaped — `tool_calls` populated, `content: null`,
    /// `finish_reason: "tool_calls"`. Regression guard for the VLM
    /// tool-calling wiring: `req.tools` → `vlm.generate()` → `ChatHistory` (via
    /// FFI, unexercised by this mock) → `tools_active` routing this arm to
    /// `collect_tool_completion` instead of the plain completion path.
    #[tokio::test]
    async fn vlm_request_with_tools_returns_tool_calls() {
        let state = routed_state(&[("qwen-vlm-8b", 5.5)]).await;
        let req = chat_req(serde_json::json!({
            "model": "qwen-vlm-8b",
            "messages": [{"role": "user", "content": "list files in my home directory"}],
            "tools": [{
                "type": "function",
                "function": {"name": "search_files", "parameters": {"type": "object"}}
            }]
        }));
        let resp = chat_completions(axum::extract::State(state), JsonBody(req)).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            json["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "mock_tool"
        );
        assert_eq!(
            json["choices"][0]["message"]["content"],
            serde_json::Value::Null
        );
        assert_eq!(json["choices"][0]["finish_reason"], "tool_calls");
    }

    /// Same request shape but WITHOUT `tools` — the mock must not emit a
    /// `<tool_call>` block, so the response stays a plain completion.
    /// Guards against `tools_active` mis-firing (e.g. a missing `tool_choice:
    /// "none"` check) and routing every VLM request through the tool-call
    /// branch regardless of the request.
    #[tokio::test]
    async fn vlm_request_without_tools_is_a_plain_completion() {
        let state = routed_state(&[("qwen-vlm-8b", 5.5)]).await;
        let req = chat_req(serde_json::json!({
            "model": "qwen-vlm-8b",
            "messages": [{"role": "user", "content": "describe the weather"}]
        }));
        let resp = chat_completions(axum::extract::State(state), JsonBody(req)).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json["choices"][0]["message"]["tool_calls"].is_null());
        assert_eq!(json["choices"][0]["message"]["content"], "mock-vlm");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
    }

    /// Text-only request on a text model still works (no regression).
    #[tokio::test]
    async fn text_request_on_text_model_is_served() {
        let state = routed_state(&[("qwen3-8b", 5.5)]).await;
        let req = chat_req(serde_json::json!({
            "model": "qwen3-8b",
            "messages": [{"role": "user", "content": "hello"}]
        }));
        let resp = chat_completions(axum::extract::State(state), JsonBody(req)).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// `EffectiveSampling` omits every unset field — never emits `null` —
    /// same "omit, don't fabricate" convention `generation_metadata` uses on
    /// the image-gen endpoints.
    #[test]
    fn effective_sampling_omits_unset_fields() {
        let all_none = EffectiveSampling {
            temperature: None,
            top_p: None,
            top_k: None,
        };
        let value = serde_json::to_value(&all_none).unwrap();
        assert_eq!(
            value.as_object().unwrap().len(),
            0,
            "every field unset must serialize to an empty object: {value:?}"
        );

        let partial = EffectiveSampling {
            temperature: Some(0.7),
            top_p: None,
            top_k: Some(40),
        };
        let value = serde_json::to_value(&partial).unwrap();
        let obj = value.as_object().unwrap();
        let keys: std::collections::BTreeSet<&str> =
            obj.keys().map(std::string::String::as_str).collect();
        assert_eq!(
            keys,
            std::collections::BTreeSet::from(["temperature", "top_k"]),
            "top_p must be absent when None, temperature/top_k present when Some: {obj:?}"
        );
    }

    /// `ChatCompletion`/`ChatChunk` carry `model_family` and `sampling` —
    /// the tool-call dialect and effective sampling params actually used for
    /// this request, mirroring image-gen's `generation_metadata` transparency.
    #[test]
    fn chat_completion_reports_model_family_and_effective_sampling() {
        let meta = ResponseMeta {
            family: ModelFamily::Mistral,
            temperature: Some(0.8),
            top_p: Some(0.9),
            top_k: None,
        };
        let body = ChatCompletion {
            id: "id-1",
            object: "chat.completion",
            created: 0,
            model: "test-model",
            choices: vec![],
            usage: usage_for(0, 0),
            model_family: meta.family.label(),
            sampling: meta.sampling(),
        };
        let value = serde_json::to_value(&body).unwrap();
        assert_eq!(value["model_family"], "mistral");
        // f32 -> JSON f64 widening means 0.8f32 isn't bit-identical to the
        // literal 0.8f64 — compare with tolerance, not assert_eq!.
        assert!((value["sampling"]["temperature"].as_f64().unwrap() - 0.8).abs() < 1e-6);
        assert!((value["sampling"]["top_p"].as_f64().unwrap() - 0.9).abs() < 1e-6);
        assert!(
            value["sampling"]
                .as_object()
                .unwrap()
                .get("top_k")
                .is_none(),
            "top_k must be absent (None), not null: {value:?}"
        );
    }

    /// First SSE frame carries `role: "assistant"` and no content.
    #[test]
    fn role_event_has_assistant_role_and_no_content() {
        let event = role_event("id-1", "test-model", 1_000_000, test_meta());
        // Event's Debug output embeds the raw serialised SSE `data:` payload
        // (see `Event { buffer: Active(b"data: {...}\n"), .. }`), so checking
        // substrings here inspects the function's real output, not a
        // separately hand-built duplicate.
        let formatted = format!("{event:?}");
        assert!(
            formatted.contains(r#"\"delta\":{\"role\":\"assistant\"}"#),
            "role frame must carry role=assistant and nothing else in delta: {formatted}"
        );
        assert!(
            !formatted.contains(r#"\"content\":"#),
            "role frame must not carry content: {formatted}"
        );
        assert!(
            formatted.contains(r#"\"finish_reason\":null"#),
            "role frame must have a null finish_reason: {formatted}"
        );
    }

    /// Token event carries content but no role or `finish_reason`.
    #[test]
    fn token_event_carries_content_only() {
        let event = token_event("Hello", "id-2", "test-model", 1_000_000, test_meta());
        let formatted = format!("{event:?}");
        assert!(
            formatted.contains(r#"\"delta\":{\"content\":\"Hello\"}"#),
            "token frame must carry only content=Hello in delta: {formatted}"
        );
        assert!(
            !formatted.contains(r#"\"role\":"#),
            "token chunk must not include role key at all: {formatted}"
        );
    }

    /// Done event expands to exactly two SSE frames.
    #[test]
    fn done_event_produces_finish_and_done_frames() {
        let frames = stream_event_to_sse(
            StreamEvent::Done(FinishReason::Stop),
            "id-3",
            "m",
            0,
            (0, 0),
            false,
            test_meta(),
        );
        assert_eq!(frames.len(), 2, "Done must produce exactly 2 SSE frames");
        // Second frame must be the [DONE] sentinel
        let second = frames[1].as_ref().unwrap();
        let debug = format!("{second:?}");
        assert!(
            debug.contains("[DONE]"),
            "second frame must be [DONE]: {debug}"
        );
    }

    /// Token event produces exactly one SSE frame.
    #[test]
    fn token_event_produces_one_frame() {
        let frames = stream_event_to_sse(
            StreamEvent::Token("word".into(), 1),
            "id-4",
            "m",
            0,
            (0, 0),
            false,
            test_meta(),
        );
        assert_eq!(frames.len(), 1, "Token must produce exactly 1 SSE frame");
    }

    /// Regression test for the 2026-07-19 undercount (the project's internal engineering log):
    /// `ScanState::process` (the streaming chat path) must sum each `Token`
    /// event's reported count, not count events — speculative decoding's
    /// verification step can accept several draft tokens at once, all
    /// landing in one event/delta.
    #[test]
    fn scan_state_sums_multi_token_events_not_event_count() {
        let mut state = ScanState::new(
            None,
            false,
            None,
            crate::admission::WorkLease::default(),
            None,
        );
        state.process(
            StreamEvent::Token("assisted".into(), 4),
            "id",
            "m",
            0,
            false,
            test_meta(),
        );
        state.process(
            StreamEvent::Token(" chunk".into(), 1),
            "id",
            "m",
            0,
            false,
            test_meta(),
        );
        state.process(
            StreamEvent::Token(" more".into(), 2),
            "id",
            "m",
            0,
            false,
            test_meta(),
        );
        assert_eq!(
            state.completion_tokens, 7,
            "must sum 4+1+2=7 real tokens across 3 events, not count 3 events"
        );
    }

    /// `Done(Stop)` emits `finish_reason: "stop"`.
    ///
    /// `Event`'s `Debug` repr renders the body as a byte-string literal, so the
    /// JSON's `"` characters appear backslash-escaped (`\"`). We assert on that
    /// escaped form to pin the exact field value, not just the substring `stop`.
    #[test]
    fn finish_event_has_stop_reason_and_empty_delta() {
        let frames = stream_event_to_sse(
            StreamEvent::Done(FinishReason::Stop),
            "id-5",
            "test-model",
            0,
            (0, 0),
            false,
            test_meta(),
        );
        let finish_frame = frames[0].as_ref().unwrap();
        let debug = format!("{finish_frame:?}");
        assert!(
            debug.contains(r#"\"finish_reason\":\"stop\""#),
            "finish frame must carry finish_reason:\"stop\": {debug}"
        );
    }

    /// `Done(Length)` emits `finish_reason: "length"` — the token-budget path.
    #[test]
    fn finish_event_has_length_reason_when_budget_hit() {
        let frames = stream_event_to_sse(
            StreamEvent::Done(FinishReason::Length),
            "id-6",
            "test-model",
            0,
            (0, 0),
            false,
            test_meta(),
        );
        let finish_frame = frames[0].as_ref().unwrap();
        let debug = format!("{finish_frame:?}");
        assert!(
            debug.contains(r#"\"finish_reason\":\"length\""#),
            "finish frame must carry finish_reason:\"length\": {debug}"
        );
        // And must NOT also carry the "stop" value.
        assert!(
            !debug.contains(r#"\"finish_reason\":\"stop\""#),
            "finish frame must not also carry stop: {debug}"
        );
    }

    /// Non-streaming collector buffers all content, counts non-empty deltas
    /// as completion tokens, and reflects the engine's `finish_reason`.
    #[tokio::test]
    async fn collect_completion_buffers_content_and_counts_tokens() {
        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            tx.send(StreamEvent::Token("Hello".into(), 1))
                .await
                .unwrap();
            tx.send(StreamEvent::Token(" world".into(), 1))
                .await
                .unwrap();
            // Empty delta (e.g. multibyte char held back) must NOT count.
            tx.send(StreamEvent::Token(String::new(), 1)).await.unwrap();
            tx.send(StreamEvent::Done(FinishReason::Length))
                .await
                .unwrap();
        });

        let resp = collect_completion(
            rx,
            "chatcmpl-x",
            "test-model",
            42,
            None,
            None,
            false,
            test_meta(),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["model"], "test-model");
        assert_eq!(json["created"], 42);
        assert_eq!(json["choices"][0]["index"], 0);
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
        assert_eq!(json["choices"][0]["message"]["content"], "Hello world");
        // No <think> in the stream → reasoning_content must be omitted entirely.
        assert!(
            json["choices"][0]["message"]
                .get("reasoning_content")
                .is_none(),
            "reasoning_content must be absent for non-reasoning output"
        );
        assert_eq!(json["choices"][0]["finish_reason"], "length");
        assert_eq!(json["usage"]["completion_tokens"], 2);
        assert_eq!(json["usage"]["prompt_tokens"], 0);
        assert_eq!(json["usage"]["total_tokens"], 2);
    }

    /// Regression test for the 2026-07-19 undercount (the project's internal engineering log):
    /// a single `Token` event can represent MORE than one real token —
    /// speculative decoding's verification step can accept several draft
    /// tokens at once, all landing in one callback/delta. The collector must
    /// sum each event's reported count, not count events themselves (a
    /// single 3-event stream with counts 4/1/2 must total 7, never 3).
    #[tokio::test]
    async fn collect_completion_sums_multi_token_events_not_event_count() {
        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            tx.send(StreamEvent::Token("assisted chunk".into(), 4))
                .await
                .unwrap();
            tx.send(StreamEvent::Token(" one".into(), 1)).await.unwrap();
            tx.send(StreamEvent::Token(" more chunk".into(), 2))
                .await
                .unwrap();
            tx.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });

        let resp = collect_completion(
            rx,
            "chatcmpl-x",
            "test-model",
            42,
            None,
            None,
            false,
            test_meta(),
        )
        .await;
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(
            json["usage"]["completion_tokens"], 7,
            "must sum 4+1+2=7 real tokens across 3 events, not count 3 events"
        );
    }

    /// A `PromptTokens` event is folded into `usage.prompt_tokens`, and
    /// `total_tokens` becomes prompt + completion (Phase 3.6).
    #[tokio::test]
    async fn collect_completion_uses_exact_prompt_tokens() {
        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            // The engine sends the prompt count first, before any token.
            tx.send(StreamEvent::PromptTokens(11)).await.unwrap();
            tx.send(StreamEvent::Token("Hi".into(), 1)).await.unwrap();
            tx.send(StreamEvent::Token(" there".into(), 1))
                .await
                .unwrap();
            tx.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });

        let resp = collect_completion(rx, "id", "m", 0, None, None, false, test_meta()).await;
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(json["usage"]["prompt_tokens"], 11, "exact prompt count");
        assert_eq!(json["usage"]["completion_tokens"], 2);
        assert_eq!(json["usage"]["total_tokens"], 13, "prompt + completion");
    }

    /// Without a `PromptTokens` event (mock path), `prompt_tokens` stays 0 and
    /// `total_tokens` equals the completion count — the pre-3.6 behaviour.
    #[tokio::test]
    async fn collect_completion_prompt_tokens_default_zero() {
        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            tx.send(StreamEvent::Token("x".into(), 1)).await.unwrap();
            tx.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });

        let resp = collect_completion(rx, "id", "m", 0, None, None, false, test_meta()).await;
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["usage"]["prompt_tokens"], 0);
        assert_eq!(json["usage"]["total_tokens"], 1);
    }

    /// `<think>` reasoning is split out of `content` into `reasoning_content`.
    #[tokio::test]
    async fn collect_completion_splits_reasoning_content() {
        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            tx.send(StreamEvent::Token(
                "<think>weighing options</think>".into(),
                1,
            ))
            .await
            .unwrap();
            tx.send(StreamEvent::Token("The answer is 42.".into(), 1))
                .await
                .unwrap();
            tx.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });

        let resp = collect_completion(
            rx,
            "id",
            "m",
            0,
            None,
            Some(ReasoningParser::Qwen3),
            false,
            test_meta(),
        )
        .await;
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(
            json["choices"][0]["message"]["content"], "The answer is 42.",
            "content must be the answer only, with <think> removed"
        );
        assert_eq!(
            json["choices"][0]["message"]["reasoning_content"], "weighing options",
            "reasoning_content must carry the <think> text"
        );
    }

    /// gpt-oss's harmony channels (not `<think>` tags) are split into
    /// `reasoning_content`/`content` the same way, gated on
    /// `ReasoningParser::GptOss` specifically. Content is the live-verified
    /// raw response shape for `gpt-oss-20b-int4-ov` (2026-07-27).
    #[tokio::test]
    async fn collect_completion_splits_gpt_oss_harmony_reasoning() {
        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            tx.send(StreamEvent::Token(
                "analysisThe user asks a question.\n\nassistantfinalThe answer is 42.".into(),
                1,
            ))
            .await
            .unwrap();
            tx.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });

        let resp = collect_completion(
            rx,
            "id",
            "m",
            0,
            None,
            Some(ReasoningParser::GptOss),
            false,
            test_meta(),
        )
        .await;
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(
            json["choices"][0]["message"]["content"], "The answer is 42.",
            "content must be the final-channel answer only"
        );
        assert_eq!(
            json["choices"][0]["message"]["reasoning_content"], "The user asks a question.",
            "reasoning_content must carry the analysis-channel text, channel markers stripped"
        );
    }

    /// T3.1/T3.2/T6.2b: a mid-stream error reaches the SSE client as a *valid
    /// JSON* `OpenAI` error envelope **scrubbed** of the raw C++ text (here an
    /// OOM-class failure → the retryable `model_unavailable` message, with the
    /// `CL_OUT_OF_RESOURCES`/device string never reflected), followed by a
    /// terminal chunk with `finish_reason:"error"` and exactly one `[DONE]` —
    /// the engine's trailing `Done(Stop)` is suppressed (no second terminal, no
    /// "stop" mislabel).
    #[tokio::test]
    async fn mid_stream_error_emits_valid_envelope_terminal_and_single_done() {
        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            tx.send(StreamEvent::Token("Hello".into(), 1))
                .await
                .unwrap();
            tx.send(StreamEvent::Error(
                "ov_cb_step failed: \"GPU.1\" \\ CL_OUT_OF_RESOURCES\nat step()".into(),
            ))
            .await
            .unwrap();
            // The CB engine always follows Error with Done(Stop).
            tx.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });

        let resp = build_chat_sse_stream(
            rx,
            "id",
            "m",
            0,
            false,
            None,
            false,
            None,
            crate::admission::WorkLease::default(),
            None,
            test_meta(),
        );
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();

        // The error frame parses as JSON and matches the error.rs envelope.
        let err_line = body
            .lines()
            .find(|l| l.starts_with("data:") && l.contains("\"error\""))
            .unwrap(); // an error frame must be present
        let json: serde_json::Value =
            serde_json::from_str(err_line.trim_start_matches("data:").trim()).unwrap(); // valid JSON: the message is a fixed constant, never interpolated
        assert_eq!(json["error"]["type"], "server_error");
        // OOM-class → retryable code, and the raw diagnostics are scrubbed.
        assert_eq!(json["error"]["code"], "model_unavailable");
        let msg = json["error"]["message"].as_str().unwrap();
        assert!(
            !msg.contains("CL_OUT_OF_RESOURCES") && !msg.contains("GPU.1"),
            "raw C++ diagnostics leaked into the SSE error frame: {msg}"
        );

        // Terminal: finish_reason "error", exactly one [DONE], never "stop".
        assert!(
            body.contains(r#""finish_reason":"error""#),
            "errored stream must report finish_reason error: {body}"
        );
        assert_eq!(
            body.matches("[DONE]").count(),
            1,
            "exactly one [DONE]: {body}"
        );
        assert!(
            !body.contains(r#""finish_reason":"stop""#),
            "the engine's trailing Done(Stop) must be suppressed: {body}"
        );
    }

    /// A non-OOM `Error` event during non-streaming collection maps to an opaque
    /// HTTP 500 — the raw engine text is scrubbed, never reflected (T6.2).
    #[tokio::test]
    async fn collect_completion_error_event_maps_to_500() {
        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            // Raw text that, pre-T6.2, would have been echoed into the body.
            tx.send(StreamEvent::Error("boom /opt/models/secret".into()))
                .await
                .unwrap();
            tx.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });

        let resp = collect_completion(rx, "id", "m", 0, None, None, false, test_meta()).await;
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);

        // Body must be the OpenAI error envelope with a generic, scrubbed message.
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["type"], "server_error");
        let msg = json["error"]["message"].as_str().unwrap_or_default();
        assert_eq!(msg, "inference failed");
        assert!(!msg.contains("boom"), "raw engine text leaked: {json}");
    }

    /// An OOM-class error event maps to a retryable 503 `model_unavailable`
    /// (not a flat 500) with no raw diagnostics in the body (T6.2).
    #[tokio::test]
    async fn collect_completion_oom_maps_to_retryable_503() {
        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            tx.send(StreamEvent::Error(
                "ov_cb step failed: CL_OUT_OF_RESOURCES (-5)".into(),
            ))
            .await
            .unwrap();
            tx.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });

        let resp = collect_completion(rx, "id", "m", 0, None, None, false, test_meta()).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "model_unavailable");
        let msg = json["error"]["message"].as_str().unwrap_or_default();
        assert!(
            !msg.contains("CL_OUT_OF_RESOURCES"),
            "OpenCL code leaked: {json}"
        );
    }

    // ---- Tool-calling buffered collector --------------------------------

    /// A `<tool_call>` block in the output becomes an `OpenAI` tool-call message:
    /// `content: null`, `tool_calls[…]`, `finish_reason: "tool_calls"`.
    #[tokio::test]
    async fn collect_tool_completion_json_parses_tool_call() {
        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            tx.send(StreamEvent::Token(
                r#"<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>"#
                    .into(),
                1,
            ))
            .await
            .unwrap();
            tx.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });

        let resp = collect_tool_completion(
            rx,
            "id-1",
            "m",
            1,
            test_meta(),
            false,
            false,
            None,
            None,
            false,
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["finish_reason"], "tool_calls");
        assert!(
            json["choices"][0]["message"]["content"].is_null(),
            "tool-call message content must be explicit null"
        );
        let tc = &json["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "get_weather");
        assert_eq!(tc["function"]["arguments"], r#"{"city":"Paris"}"#);
    }

    /// gpt-oss's harmony tool-call shape, exercised end to end: raw engine
    /// text (leading `analysis`, the `assistantcommentary` transition
    /// marker, exactly as `HarmonyFilter` sees it) all the way through to
    /// the `OpenAI` `tool_calls` message shape. Confirms `collect_tool_completion`
    /// scans `thinking` (not `answer`) for `ModelFamily::GptOss`, and that the
    /// harmony transition marker + `to=functions...` header never leak into
    /// either `content` or `reasoning_content`.
    #[tokio::test]
    async fn collect_tool_completion_parses_gpt_oss_harmony_tool_call() {
        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            tx.send(StreamEvent::Token(
                "analysisWe need to call the weather tool.\
                 assistantcommentary to=functions.get_weather json{\"location\":\"SF\"}"
                    .into(),
                1,
            ))
            .await
            .unwrap();
            tx.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });

        let resp = collect_tool_completion(
            rx,
            "id-1",
            "m",
            1,
            ResponseMeta {
                family: ModelFamily::GptOss,
                ..test_meta()
            },
            false,
            false,
            None,
            Some(ReasoningParser::GptOss),
            false,
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(json["choices"][0]["finish_reason"], "tool_calls");
        assert!(json["choices"][0]["message"]["content"].is_null());
        let tc = &json["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tc["function"]["name"], "get_weather");
        assert_eq!(tc["function"]["arguments"], r#"{"location":"SF"}"#);
        assert_eq!(
            json["choices"][0]["message"]["reasoning_content"], "We need to call the weather tool.",
            "genuine analysis reasoning must survive, tool-call header stripped out of it"
        );
    }

    /// Output with no tool call is a normal text completion, carrying the
    /// engine's finish reason and no `tool_calls` key.
    #[tokio::test]
    async fn collect_tool_completion_json_without_call_is_text() {
        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            tx.send(StreamEvent::Token("Just a plain answer.".into(), 1))
                .await
                .unwrap();
            tx.send(StreamEvent::Done(FinishReason::Length))
                .await
                .unwrap();
        });

        let resp = collect_tool_completion(
            rx,
            "id",
            "m",
            0,
            test_meta(),
            false,
            false,
            None,
            None,
            false,
        )
        .await;
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(json["choices"][0]["finish_reason"], "length");
        assert_eq!(
            json["choices"][0]["message"]["content"],
            "Just a plain answer."
        );
        assert!(
            json["choices"][0]["message"].get("tool_calls").is_none(),
            "no tool_calls key when there is no call"
        );
    }

    /// `<think>` reasoning is stripped before tool parsing.
    #[tokio::test]
    async fn collect_tool_completion_strips_think_block() {
        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            tx.send(StreamEvent::Token(
                r#"<think>I should call it</think><tool_call>{"name": "f", "arguments": {}}</tool_call>"#
                    .into(),
                1,
            ))
            .await
            .unwrap();
            tx.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });

        let resp = collect_tool_completion(
            rx,
            "id",
            "m",
            0,
            test_meta(),
            false,
            false,
            None,
            Some(ReasoningParser::Qwen3),
            false,
        )
        .await;
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            json["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "f"
        );
        // The stripped <think> is now surfaced as reasoning_content.
        assert_eq!(
            json["choices"][0]["message"]["reasoning_content"],
            "I should call it"
        );
    }

    /// Streaming tool form emits a `tool_calls` chunk, a `tool_calls` finish
    /// chunk, and the `[DONE]` sentinel.
    #[tokio::test]
    async fn collect_tool_completion_sse_emits_tool_calls_and_done() {
        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            // Leading <think> block exercises streaming reasoning_content parity.
            tx.send(StreamEvent::Token(
                r#"<think>pick a tool</think><tool_call>{"name":"f","arguments":{}}</tool_call>"#
                    .into(),
                1,
            ))
            .await
            .unwrap();
            tx.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });

        let resp = collect_tool_completion(
            rx,
            "id",
            "m",
            0,
            test_meta(),
            true,
            false,
            None,
            Some(ReasoningParser::Qwen3),
            false,
        )
        .await;
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            body.contains(r#""finish_reason":"tool_calls""#),
            "must finish with tool_calls: {body}"
        );
        assert!(
            body.contains("[DONE]"),
            "must terminate with [DONE]: {body}"
        );

        // Parse the payload chunk (the data frame carrying tool_calls) and
        // inspect the delta directly — substring checks can't tell the tool-call
        // index apart from the choice index (both render `"index":0`).
        let payload_json = body
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter(|d| d.contains(r#""tool_calls""#))
            .map(|d| serde_json::from_str::<serde_json::Value>(d).unwrap())
            .next()
            .unwrap();
        let delta = &payload_json["choices"][0]["delta"];

        assert_eq!(delta["tool_calls"][0]["function"]["name"], "f");
        // OpenAI requires the array index on every streaming tool-call delta —
        // openai-python merges fragments by it.
        assert_eq!(
            delta["tool_calls"][0]["index"], 0,
            "streaming tool_calls must carry index: {body}"
        );
        // Reasoning surfaces in the buffered tool stream, at parity with the
        // non-stream path.
        assert_eq!(
            delta["reasoning_content"], "pick a tool",
            "must surface reasoning_content: {body}"
        );
    }

    // ---- stream_options.include_usage -----------------------------------

    /// With `include_usage`, `Done` expands to 3 frames: finish → usage → [DONE].
    #[test]
    fn include_usage_done_emits_usage_chunk_before_done() {
        let frames = stream_event_to_sse(
            StreamEvent::Done(FinishReason::Stop),
            "id-7",
            "test-model",
            42,
            (11, 5),
            true,
            test_meta(),
        );
        assert_eq!(frames.len(), 3, "include_usage Done must produce 3 frames");

        // Frame 0: finish_reason chunk.
        let f0 = format!("{:?}", frames[0].as_ref().unwrap());
        assert!(
            f0.contains(r#"\"finish_reason\":\"stop\""#),
            "frame 0 must be finish: {f0}"
        );

        // Frame 1: usage chunk — choices:[], usage object.
        let f1 = format!("{:?}", frames[1].as_ref().unwrap());
        assert!(
            f1.contains(r#"\"prompt_tokens\":11"#),
            "frame 1 must have prompt_tokens: {f1}"
        );
        assert!(
            f1.contains(r#"\"completion_tokens\":5"#),
            "frame 1 must have completion_tokens: {f1}"
        );
        assert!(
            f1.contains(r#"\"total_tokens\":16"#),
            "frame 1 must have total_tokens: {f1}"
        );
        assert!(
            f1.contains(r#"\"choices\":[]"#),
            "frame 1 must have empty choices: {f1}"
        );

        // Frame 2: [DONE] sentinel.
        let f2 = format!("{:?}", frames[2].as_ref().unwrap());
        assert!(f2.contains("[DONE]"), "frame 2 must be [DONE]: {f2}");
    }

    /// Without `include_usage`, `Done` still produces exactly 2 frames.
    #[test]
    fn no_include_usage_done_still_two_frames() {
        let frames = stream_event_to_sse(
            StreamEvent::Done(FinishReason::Stop),
            "id-8",
            "m",
            0,
            (11, 5),
            false,
            test_meta(),
        );
        assert_eq!(frames.len(), 2, "no include_usage must keep 2 frames");
    }

    // ---- A4 streaming <think> strip --------------------------------

    /// `<think>` content must NOT appear in `delta.content`; it must go to
    /// `delta.reasoning_content` instead. The answer text must be clean.
    #[tokio::test]
    async fn streaming_think_is_stripped_from_content() {
        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            tx.send(StreamEvent::Token("<think>".into(), 1))
                .await
                .unwrap();
            tx.send(StreamEvent::Token("step by step".into(), 1))
                .await
                .unwrap();
            tx.send(StreamEvent::Token("</think>".into(), 1))
                .await
                .unwrap();
            tx.send(StreamEvent::Token("answer".into(), 1))
                .await
                .unwrap();
            tx.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });

        let resp = build_chat_sse_stream(
            rx,
            "id",
            "m",
            0,
            false,
            None,
            false,
            None,
            crate::admission::WorkLease::default(),
            None,
            test_meta(),
        );
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            body.contains("reasoning_content"),
            "reasoning must appear in a delta: {body}"
        );
        assert!(
            body.contains(r#""content":"answer""#),
            "answer must appear in content delta: {body}"
        );
        assert!(
            !body.contains("<think>") && !body.contains("</think>"),
            "think tags must not appear in SSE output: {body}"
        );
        assert!(
            body.contains("[DONE]"),
            "stream must end with [DONE]: {body}"
        );
    }

    /// Same split, gpt-oss harmony shape: `reasoning_parser: GptOss` must
    /// select the `HarmonyFilter` path in `ScanState`, not the `<think>`-tag
    /// one — analysis-channel text goes to `reasoning_content`, the
    /// "assistantfinal" transition marker is stripped, not leaked as content.
    #[tokio::test]
    async fn streaming_gpt_oss_harmony_channel_split() {
        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            tx.send(StreamEvent::Token("analysis".into(), 1))
                .await
                .unwrap();
            // Capitalized, matching real gpt-oss output (reasoning always
            // starts a fresh sentence) — lowercase would fail the leading-
            // word boundary guard by design, see `match_leading_channel_word`.
            tx.send(StreamEvent::Token("Step by step".into(), 1))
                .await
                .unwrap();
            tx.send(StreamEvent::Token("assistant".into(), 1))
                .await
                .unwrap();
            tx.send(StreamEvent::Token("final".into(), 1))
                .await
                .unwrap();
            tx.send(StreamEvent::Token("answer".into(), 1))
                .await
                .unwrap();
            tx.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });

        let resp = build_chat_sse_stream(
            rx,
            "id",
            "m",
            0,
            false,
            None,
            false,
            Some(ReasoningParser::GptOss),
            crate::admission::WorkLease::default(),
            None,
            test_meta(),
        );
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            body.contains("reasoning_content"),
            "reasoning must appear in a delta: {body}"
        );
        assert!(
            body.contains(r#""content":"answer""#),
            "answer must appear in content delta: {body}"
        );
        assert!(
            !body.contains(r#""content":"assistant"#) && !body.contains(r#""content":"final"#),
            "channel transition marker must not leak into a content delta: {body}"
        );
        assert!(
            body.contains("[DONE]"),
            "stream must end with [DONE]: {body}"
        );
    }

    /// When `<think>` and `</think>` are split across token boundaries the
    /// filter must still strip the block correctly.
    #[tokio::test]
    async fn streaming_think_tag_split_across_tokens() {
        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            // open tag split: "<thi" then "nk>"
            tx.send(StreamEvent::Token("<thi".into(), 1)).await.unwrap();
            tx.send(StreamEvent::Token("nk>".into(), 1)).await.unwrap();
            tx.send(StreamEvent::Token("thought".into(), 1))
                .await
                .unwrap();
            // close tag split: "</thi" then "nk>"
            tx.send(StreamEvent::Token("</thi".into(), 1))
                .await
                .unwrap();
            tx.send(StreamEvent::Token("nk>".into(), 1)).await.unwrap();
            tx.send(StreamEvent::Token("clean".into(), 1))
                .await
                .unwrap();
            tx.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });

        let resp = build_chat_sse_stream(
            rx,
            "id",
            "m",
            0,
            false,
            None,
            false,
            None,
            crate::admission::WorkLease::default(),
            None,
            test_meta(),
        );
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            body.contains("reasoning_content"),
            "split-tag reasoning must appear: {body}"
        );
        assert!(
            body.contains(r#""content":"clean""#),
            "answer after split close-tag must be in content: {body}"
        );
        assert!(
            !body.contains("<think>") && !body.contains("</think>"),
            "reconstructed tags must not appear in output: {body}"
        );
    }

    /// A stream with no `<think>` block must be unaffected — no
    /// `reasoning_content` key anywhere in the output.
    #[tokio::test]
    async fn streaming_no_think_block_is_passthrough() {
        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            tx.send(StreamEvent::Token("plain answer".into(), 1))
                .await
                .unwrap();
            tx.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });

        let resp = build_chat_sse_stream(
            rx,
            "id",
            "m",
            0,
            false,
            None,
            false,
            None,
            crate::admission::WorkLease::default(),
            None,
            test_meta(),
        );
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            !body.contains("reasoning_content"),
            "plain output must not carry reasoning_content: {body}"
        );
        assert!(
            body.contains("plain answer"),
            "content must pass through unchanged: {body}"
        );
    }

    // ---- VLM scaffold: MessageContent types + routing helpers ----

    #[test]
    fn content_string_deserialises_as_text_variant() {
        let msg: ChatMessage =
            serde_json::from_str(r#"{"role":"user","content":"hello"}"#).unwrap();
        assert!(matches!(msg.content, Some(MessageContent::Text(s)) if s == "hello"));
    }

    #[test]
    fn content_array_deserialises_as_parts_variant() {
        let msg: ChatMessage = serde_json::from_str(
            r#"{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,abc"}}]}"#,
        )
        .unwrap();
        let Some(MessageContent::Parts(parts)) = &msg.content else {
            panic!("expected Parts variant");
        };
        assert_eq!(parts.len(), 2);
        assert!(matches!(&parts[0], ContentPart::Text { text } if text == "hi"));
        assert!(matches!(&parts[1], ContentPart::ImageUrl { .. }));
    }

    #[test]
    fn has_images_true_when_image_url_present() {
        let messages: Vec<ChatMessage> = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}}
            ]}
        ]))
        .unwrap();
        assert!(has_images(&messages));
    }

    #[test]
    fn has_images_false_for_text_only_string() {
        let messages: Vec<ChatMessage> =
            serde_json::from_value(serde_json::json!([{"role": "user", "content": "hi"}])).unwrap();
        assert!(!has_images(&messages));
    }

    #[test]
    fn has_images_false_for_text_only_parts() {
        let messages: Vec<ChatMessage> = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        ]))
        .unwrap();
        assert!(!has_images(&messages));
    }

    #[test]
    fn extract_text_content_joins_text_parts_and_drops_images() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "hello ".into(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrlData {
                    url: "data:image/jpeg;base64,abc".into(),
                },
            },
            ContentPart::Text {
                text: "world".into(),
            },
        ]);
        assert_eq!(extract_text_content(&content), "hello world");
    }

    // ---- T7.6: response-id uniqueness + ride-alongs (#47, #48) ----

    #[test]
    fn next_response_seq_is_strictly_increasing_and_unique() {
        // Two ids built in the same wall-clock second must still differ — the
        // collision the whole-second `created` field caused under concurrency.
        let a = next_response_seq();
        let b = next_response_seq();
        let c = next_response_seq();
        assert!(
            a < b && b < c,
            "sequence must strictly increase: {a} {b} {c}"
        );
        let created = unix_now();
        let id1 = format!("chatcmpl-rv-{created}-{}", next_response_seq());
        let id2 = format!("chatcmpl-rv-{created}-{}", next_response_seq());
        assert_ne!(id1, id2, "same-second ids must be distinct");
    }

    #[test]
    fn zero_token_budget_is_rejected_else_passes() {
        // Explicit 0 → error; omitted (None) or any positive value → None.
        assert!(zero_token_budget_error(&[Some(0)]).is_some());
        assert!(zero_token_budget_error(&[None, Some(0)]).is_some());
        assert!(zero_token_budget_error(&[None, None]).is_none());
        assert!(zero_token_budget_error(&[Some(1), None]).is_none());
        assert!(zero_token_budget_error(&[]).is_none());
    }

    #[test]
    fn vlm_text_part_separator_matches_cb_path() {
        // #48: consecutive text parts must concatenate identically on both
        // families. `extract_text_content` (CB) uses an empty join; the VLM
        // extractor must produce the same text when no image intervenes.
        let parts = vec![
            ContentPart::Text { text: "foo".into() },
            ContentPart::Text { text: "bar".into() },
        ];
        let cb = extract_text_content(&MessageContent::Parts(parts.clone()));
        let (vlm, images) = extract_vlm_messages(&[user_parts(parts)]).unwrap();
        assert_eq!(cb, "foobar");
        assert!(images.is_empty());
        assert_eq!(
            vlm[0]["content"].as_str().unwrap(),
            cb,
            "VLM and CB text-part joins must agree"
        );
    }

    /// `extract_vlm_messages` must carry `tool_calls` through to the rendered
    /// JSON, mirroring `messages_to_template_values` (the CB path) — dropping
    /// it renders every assistant tool-call turn as empty in a Qwen-style
    /// template (`{%- if message.tool_calls %}`), which the model then
    /// imitates instead of ever emitting a real tool call
    /// (the project's internal engineering log).
    #[test]
    fn extract_vlm_messages_preserves_tool_calls() {
        let tool_call = serde_json::json!({
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup_record", "arguments": "{\"record_id\":\"ZQ-7741\"}"}
        });
        let msgs = vec![ChatMessage {
            role: "assistant".into(),
            tool_calls: Some(vec![tool_call.clone()]),
            ..Default::default()
        }];

        let (vlm_messages, _images) = extract_vlm_messages(&msgs).unwrap();

        assert_eq!(
            vlm_messages[0]["tool_calls"],
            serde_json::json!([tool_call]),
            "tool_calls must reach the rendered message, not be dropped"
        );
    }

    /// Same as above for `tool_call_id`/`name` — the fields a `tool`-role
    /// reply carries to answer a specific prior call.
    #[test]
    fn extract_vlm_messages_preserves_tool_call_id_and_name() {
        let msgs = vec![ChatMessage {
            role: "tool".into(),
            content: Some(MessageContent::Text("42".into())),
            tool_call_id: Some("call_1".into()),
            name: Some("lookup_record".into()),
            ..Default::default()
        }];

        let (vlm_messages, _images) = extract_vlm_messages(&msgs).unwrap();

        assert_eq!(vlm_messages[0]["tool_call_id"], "call_1");
        assert_eq!(vlm_messages[0]["name"], "lookup_record");
    }

    /// A message with none of `tool_calls`/`tool_call_id`/`name` set must not
    /// grow spurious keys — additive fields, omitted when absent, same
    /// contract as `messages_to_template_values`.
    #[test]
    fn extract_vlm_messages_omits_tool_fields_when_absent() {
        let msgs = vec![user_parts(vec![ContentPart::Text { text: "hi".into() }])];

        let (vlm_messages, _images) = extract_vlm_messages(&msgs).unwrap();
        let obj = vlm_messages[0].as_object().unwrap();

        assert!(!obj.contains_key("tool_calls"));
        assert!(!obj.contains_key("tool_call_id"));
        assert!(!obj.contains_key("name"));
    }

    /// `vlm_gate_text` must include `tool_calls` JSON, not just `content` —
    /// the L0-gate-undercount bug (`dev/autotest/
    /// 20260823_l0_gate_token_count_discrepancy.md`): an assistant turn
    /// that only made a tool call has empty `content`, so a gate measuring
    /// `content` alone sees nothing from it at all.
    #[test]
    fn vlm_gate_text_includes_tool_calls_not_just_content() {
        let msgs = vec![ChatMessage {
            role: "assistant".into(),
            tool_calls: Some(vec![serde_json::json!({
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup_record", "arguments": "{\"record_id\":\"ZQ-7741-KAPPA\"}"}
            })]),
            ..Default::default()
        }];
        let (vlm_messages, _images) = extract_vlm_messages(&msgs).unwrap();

        let gate_text = vlm_gate_text(&vlm_messages);

        assert!(
            gate_text.contains("lookup_record") && gate_text.contains("ZQ-7741-KAPPA"),
            "gate_text must see the tool call's substance, not just an empty content field: {gate_text:?}"
        );
    }

    /// A plain text-only conversation must still gate correctly — the fix
    /// is additive, not a regression on the common case.
    #[test]
    fn vlm_gate_text_still_includes_plain_content() {
        let msgs = vec![user_parts(vec![ContentPart::Text {
            text: "hello world".into(),
        }])];
        let (vlm_messages, _images) = extract_vlm_messages(&msgs).unwrap();

        assert_eq!(vlm_gate_text(&vlm_messages), "hello world");
    }

    // ---- V1: multi-image positional tagging (extract_vlm_messages) ----
    //
    // These exercise the real decode path, so they need a genuinely decodable
    // image. `tiny_png_uri` builds a 1×1 PNG data URI via the same `image` +
    // `base64ct` crates `image_util` uses.
    fn tiny_png_uri() -> String {
        use base64ct::{Base64, Encoding};
        use image::{DynamicImage, ImageFormat, RgbImage};

        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(1, 1, image::Rgb([1, 2, 3])));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), ImageFormat::Png)
            .unwrap();
        format!("data:image/png;base64,{}", Base64::encode_string(&buf))
    }

    fn user_parts(parts: Vec<ContentPart>) -> ChatMessage {
        ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Parts(parts)),
            ..Default::default()
        }
    }

    #[test]
    fn extract_vlm_two_images_one_turn_get_aligned_tags() {
        let uri = tiny_png_uri();
        let msgs = vec![user_parts(vec![
            ContentPart::Text {
                text: "compare".into(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrlData { url: uri.clone() },
            },
            ContentPart::ImageUrl {
                image_url: ImageUrlData { url: uri },
            },
        ])];

        let (vlm_messages, images) = extract_vlm_messages(&msgs).unwrap();
        assert_eq!(images.len(), 2, "both images decoded");
        let content = vlm_messages[0]["content"].as_str().unwrap();
        assert_eq!(
            content, "compare <ov_genai_image_0> <ov_genai_image_1>",
            "tags injected in order, zero-based, aligned with images[N]"
        );
    }

    #[test]
    fn extract_vlm_interleaved_text_and_images_preserve_order() {
        let uri = tiny_png_uri();
        let msgs = vec![user_parts(vec![
            ContentPart::Text { text: "a".into() },
            ContentPart::ImageUrl {
                image_url: ImageUrlData { url: uri.clone() },
            },
            ContentPart::Text { text: "b".into() },
            ContentPart::ImageUrl {
                image_url: ImageUrlData { url: uri },
            },
        ])];

        let (vlm_messages, images) = extract_vlm_messages(&msgs).unwrap();
        assert_eq!(images.len(), 2);
        assert_eq!(
            vlm_messages[0]["content"].as_str().unwrap(),
            "a <ov_genai_image_0> b <ov_genai_image_1>",
        );
    }

    /// The crash this guards against: `VLMPipeline`'s `ChatHistory` overload
    /// binds `images` to the *last* message only (documented in
    /// `pipeline.hpp`), so an image tagged in an earlier turn used to reach
    /// `ov_vlm_generate` bound to the wrong slot and abort the process with
    /// `Missing image/video with index 0` — reproduced live 2026-07-02 by
    /// resending the same image a client had already sent in turn 1. Only the
    /// last message's image may be tagged/decoded; an earlier turn's image is
    /// dropped (its sibling text — here none — is kept, and whatever the model
    /// said about it already lives in the adjacent reply).
    #[test]
    fn extract_vlm_only_last_message_images_are_tagged() {
        let uri = tiny_png_uri();
        let msgs = vec![
            user_parts(vec![ContentPart::ImageUrl {
                image_url: ImageUrlData { url: uri.clone() },
            }]),
            ChatMessage {
                role: "assistant".into(),
                content: Some(MessageContent::Text("ok".into())),
                ..Default::default()
            },
            user_parts(vec![ContentPart::ImageUrl {
                image_url: ImageUrlData { url: uri },
            }]),
        ];

        let (vlm_messages, images) = extract_vlm_messages(&msgs).unwrap();
        assert_eq!(images.len(), 1, "only the last message's image is decoded");
        assert_eq!(
            vlm_messages[0]["content"].as_str().unwrap(),
            "",
            "the earlier turn's image is dropped, not tagged"
        );
        assert_eq!(vlm_messages[1]["content"].as_str().unwrap(), "ok");
        assert_eq!(
            vlm_messages[2]["content"].as_str().unwrap(),
            "<ov_genai_image_0>",
            "the last message's image gets a local, zero-based index"
        );
    }

    /// A client resending a turn-1 image alongside new turn-2 text (exactly
    /// the client behavior that surfaced the crash) must not have its old
    /// image counted or tagged, while the new text-only-from-images
    /// perspective last message still renders cleanly.
    #[test]
    fn extract_vlm_resent_older_image_dropped_with_sibling_text_kept() {
        let uri = tiny_png_uri();
        let msgs = vec![
            user_parts(vec![
                ContentPart::Text {
                    text: "what breed is this?".into(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrlData { url: uri.clone() },
                },
            ]),
            ChatMessage {
                role: "assistant".into(),
                content: Some(MessageContent::Text("looks like a boxer".into())),
                ..Default::default()
            },
            ChatMessage {
                role: "user".into(),
                content: Some(MessageContent::Text("what color is it?".into())),
                ..Default::default()
            },
        ];

        let (vlm_messages, images) = extract_vlm_messages(&msgs).unwrap();
        assert!(
            images.is_empty(),
            "no image in the last (text-only) message"
        );
        assert_eq!(
            vlm_messages[0]["content"].as_str().unwrap(),
            "what breed is this?",
            "sibling text survives; only the image part is dropped"
        );
        assert_eq!(
            vlm_messages[2]["content"].as_str().unwrap(),
            "what color is it?"
        );
    }

    #[test]
    fn extract_vlm_text_only_has_no_tags_or_images() {
        let msgs = vec![
            ChatMessage {
                role: "user".into(),
                content: Some(MessageContent::Text("hi".into())),
                ..Default::default()
            },
            user_parts(vec![ContentPart::Text { text: "two".into() }]),
        ];

        let (vlm_messages, images) = extract_vlm_messages(&msgs).unwrap();
        assert!(images.is_empty(), "no images decoded");
        assert_eq!(vlm_messages[0]["content"].as_str().unwrap(), "hi");
        assert_eq!(vlm_messages[1]["content"].as_str().unwrap(), "two");
        assert!(
            !vlm_messages[0]["content"]
                .as_str()
                .unwrap()
                .contains("ov_genai_image"),
            "no tag leaks into text-only content"
        );
    }

    #[test]
    fn messages_to_template_values_preserves_string_content() {
        let messages: Vec<ChatMessage> =
            serde_json::from_value(serde_json::json!([{"role": "user", "content": "hello"}]))
                .unwrap();
        let vals = messages_to_template_values(&messages);
        assert_eq!(vals[0]["content"], "hello");
        assert_eq!(vals[0]["role"], "user");
    }

    #[test]
    fn messages_to_template_values_converts_parts_to_string() {
        let messages: Vec<ChatMessage> = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": [
                {"type": "text", "text": "what is "},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
                {"type": "text", "text": "this?"}
            ]}
        ]))
        .unwrap();
        let vals = messages_to_template_values(&messages);
        assert_eq!(vals[0]["content"], "what is this?");
    }
}
