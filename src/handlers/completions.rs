// ============================================================
// src/handlers/completions.rs — POST /v1/completions (legacy)
// ============================================================
// The OpenAI *legacy* text-completion endpoint: raw `prompt` text in, raw
// `text` out. Distinct from /v1/chat/completions in three ways that matter:
//
//   1. NO chat template. The prompt is fed to the engine verbatim — this is a
//      raw continuation, not a chat turn. (A client that wants chat formatting
//      uses /v1/chat/completions.)
//   2. NO <think>/reasoning_content split. Legacy completions returns exactly
//      what the model generated; there is no `message` object to carry
//      reasoning separately.
//   3. Response object is `text_completion`, ids are `cmpl-…`, and each choice
//      carries `text` (not `message`) plus an always-null `logprobs`.
//
// We target the OpenAI spec, not stormVINO (which has no such route). Supported:
// `prompt` as a string or an array of strings (→ one choice each), `max_tokens`
// (OpenAI default 16), `stream`. Unsupported request params (n, echo, suffix,
// logprobs, best_of, stop, token-id prompts) are silently ignored by serde —
// accept-and-ignore keeps drop-in clients working rather than 400-ing them.
// ============================================================

use std::convert::Infallible;
use std::time::Instant;

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use futures_util::StreamExt as _;
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;

use super::chat::{
    StreamOptions, Usage, clamp_max_tokens, drain_channel, gate_prompt, next_response_seq,
    parse_stop, resolve_repetition_penalty, resolve_sampling_defaults, spawn_mock_generation,
    unix_now, usage_for, usage_sse_event, validate_sampling_params, zero_token_budget_error,
};
use std::sync::Arc;

use super::error::{
    JsonBody, device_admission_error_response, inference_error_from_text, model_error_response,
    openai_error, sse_inference_error_event, unsupported_param,
};
use crate::{
    app_state::AppState,
    cb_engine::SubmitResult,
    model_manager::ModelManager,
    ov_cb::GenParams,
    streaming::{StreamEvent, TokenReceiver, stream_channel},
};

// ---- Request types --------------------------------------------------

/// Incoming `POST /v1/completions` body (`OpenAI`-compatible). Only the fields
/// the server acts on are declared; any other `OpenAI` param is ignored by
/// serde rather than rejected, so drop-in clients are not broken.
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    /// Which loaded model to use. No phrase/tag routing, no default fallback.
    pub model: String,

    /// The prompt(s) to complete: a single string, or an array of strings
    /// (one completion `choice` per element). Fed to the engine verbatim.
    #[serde(default)]
    pub prompt: Prompt,

    /// Cap on generated tokens. `OpenAI`'s legacy default is **16**.
    pub max_tokens: Option<u32>,

    /// If true, stream `text_completion` chunks over SSE; else a single JSON
    /// object. Defaults to `false`.
    pub stream: Option<bool>,

    /// Controls SSE stream extensions. Only meaningful when `stream: true`.
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,

    /// Sampling temperature. `0` = greedy. `OpenAI` default: 1.0.
    pub temperature: Option<f32>,
    /// Nucleus sampling cutoff. `OpenAI` default: 1.0.
    pub top_p: Option<f32>,
    /// Presence penalty. `OpenAI` range: −2..2. Default: 0.
    pub presence_penalty: Option<f32>,
    /// Frequency penalty. `OpenAI` range: −2..2. Default: 0.
    pub frequency_penalty: Option<f32>,
    /// Repetition penalty (extension field, not standard `OpenAI` — see
    /// `ChatRequest::repetition_penalty`'s doc comment). Absent → the server
    /// applies a safety-net default; send `1.0` to opt out.
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    /// Top-k sampling (extension field — see `ChatRequest::top_k`).
    /// Absent or `0` → disabled.
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Deterministic seed. When set, makes sampling reproducible.
    pub seed: Option<u64>,
    /// Stop sequence(s): a string or an array of strings.
    pub stop: Option<serde_json::Value>,

    // ---- Known-but-unsupported params (rejected with 400, not silently dropped).
    // Parsed only to detect presence; the engine cannot honor any. See
    // `unsupported_completion_param`. Cosmetic params (`user`, `suffix`, …) are
    // left to serde's accept-and-ignore (unknown fields are dropped).
    /// Number of completions to generate. One sequence per request → `n > 1` → 400.
    #[serde(default)]
    pub n: Option<u32>,
    /// Server-side over-generation count. Same constraint as `n` → `best_of > 1` → 400.
    #[serde(default)]
    pub best_of: Option<u32>,
    /// Legacy logprobs (an integer count). The bridge exposes no logits → present → 400.
    #[serde(default)]
    pub logprobs: Option<serde_json::Value>,
    /// Per-token logit bias map. No bias hook in the bridge → a non-empty map → 400.
    #[serde(default)]
    pub logit_bias: Option<serde_json::Value>,
}

/// Reject a known `OpenAI` legacy-completions parameter the engine cannot honor
/// with an explicit 400 (`unsupported_parameter`); `None` when the request is
/// clean. Mirrors the chat-path policy.
fn unsupported_completion_param(req: &CompletionRequest) -> Option<Response> {
    if req.n.is_some_and(|n| n > 1) {
        return Some(unsupported_param(
            "n",
            "this server generates a single choice per request",
        ));
    }
    if req.best_of.is_some_and(|b| b > 1) {
        return Some(unsupported_param(
            "best_of",
            "this server generates a single choice per request",
        ));
    }
    if req.logprobs.is_some() {
        return Some(unsupported_param(
            "logprobs",
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

/// The `prompt` field's polymorphic shape: a single string or an array of them.
///
/// `#[serde(untagged)]` makes serde try `Single` (a JSON string) first, then
/// `Multi` (a JSON array of strings). Array-of-token-id prompts are not
/// supported — they fail both arms and surface as a 400, which is the honest
/// answer until the engine accepts pre-tokenized input.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Prompt {
    /// One prompt string.
    Single(String),
    /// Several prompt strings → several completion choices.
    Multi(Vec<String>),
}

impl Default for Prompt {
    /// An absent `prompt` defaults to a single empty string (the engine then
    /// generates from the model's BOS, matching `OpenAI`'s permissive default).
    fn default() -> Self {
        Self::Single(String::new())
    }
}

impl Prompt {
    /// Normalise to a vector of prompt strings.
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::Multi(v) => v,
        }
    }
}

// ---- Response types --------------------------------------------------

/// A full non-streaming legacy completion (`object: "text_completion"`).
#[derive(Serialize)]
struct Completion<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

/// One streaming chunk (`object: "text_completion"`). Usage is emitted as a
/// separate final chunk when `stream_options.include_usage` is set.
#[derive(Serialize)]
struct CompletionChunk<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: Vec<CompletionChoice>,
}

/// One completion choice: the generated `text`, an always-null `logprobs`
/// (logprobs are not supported), and the finish reason. `finish_reason` is
/// `None` (→ JSON null) on intermediate streaming chunks and `Some` on the
/// final chunk and on every non-streaming choice.
#[derive(Serialize)]
struct CompletionChoice {
    index: u32,
    text: String,
    /// Always `null` — logprobs are not implemented. Present (not skipped)
    /// because the `OpenAI` shape includes the key.
    logprobs: Option<u8>,
    finish_reason: Option<&'static str>,
}

// ---- Handler --------------------------------------------------------

/// `POST /v1/completions` — `OpenAI`-compatible legacy text completion.
///
/// Feeds each prompt to the engine **without a chat template** and returns the
/// raw generated text. Streaming (`stream: true`) emits `text_completion`
/// chunks; non-streaming returns a single `text_completion` object. Multiple
/// prompts are supported for non-streaming only.
#[allow(clippy::too_many_lines)]
pub async fn completions(
    State(state): State<AppState>,
    JsonBody(req): JsonBody<CompletionRequest>,
) -> Response {
    // Zero point for per-request latency metrics, captured at handler entry
    // (matches the chat handler).
    let requested_at = Instant::now();

    // Reject known-but-unsupported params up front (n>1, best_of>1, logprobs,
    // logit_bias) — explicit 400 rather than a silent accept-and-drop.
    if let Some(resp) = unsupported_completion_param(&req) {
        return resp;
    }

    // Validate sampling-param ranges up front (#24, shared with chat): an
    // out-of-range temperature/top_p/penalty (or n:0) is a clean 400 rather
    // than an opaque mid-stream engine failure or a silent clamp.
    if let Some(resp) = validate_sampling_params(
        req.temperature,
        req.top_p,
        req.presence_penalty,
        req.frequency_penalty,
        req.n,
    ) {
        return resp;
    }

    // #47: an explicit max_tokens of 0 is a 400 (OpenAI minimum is 1), distinct
    // from omitting the field (= the legacy default of 16). Shared with chat.
    if let Some(resp) = zero_token_budget_error(&[req.max_tokens]) {
        return resp;
    }

    let stream = req.stream.unwrap_or(false);
    let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);
    let created = unix_now();
    let id = format!("cmpl-rv-{created}-{}", next_response_seq());
    let prompts = req.prompt.into_vec();

    if prompts.is_empty() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "prompt must not be empty",
            "invalid_request_error",
            Some("empty_prompt"),
        );
    }

    // T2.1: cap the prompt-array fan-out. Each element occupies an engine
    // slot for its entire generation, so an unbounded array lets a single
    // connection saturate every slot (committee finding 3). Rejected before
    // any engine work starts.
    let max_prompts = state.model_manager.as_ref().map_or_else(
        crate::model_manager::config::default_max_prompt_array,
        |mm| mm.max_prompt_array(),
    );
    if prompts.len() > max_prompts {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "prompt array has {} elements — this server accepts at most \
                 {max_prompts} prompts per request",
                prompts.len()
            ),
            "invalid_request_error",
            Some("too_many_prompts"),
        );
    }

    // OpenAI legacy default max_tokens is 16 (chat uses the model default);
    // both are bounded by the server-wide ceiling (T2.2, shared clamp).
    let max_tokens_cap = state
        .model_manager
        .as_ref()
        .map_or_else(crate::model_manager::config::default_max_tokens_cap, |mm| {
            mm.max_tokens_cap()
        });
    // Model-shipped sampling defaults (generation_config.json), same
    // resolution `/v1/chat/completions` uses — see `resolve_sampling_defaults`'s
    // doc comment. `tools_active` is always `false` here: legacy completions
    // has no tool-calling concept at all, so only the "client set nothing →
    // model defaults apply" branch of that function is ever reachable on this
    // endpoint.
    let model_defaults = state
        .model_manager
        .as_ref()
        .map_or_else(Default::default, |mm| {
            mm.model_generation_defaults(&req.model)
        });
    let (temperature, top_p, top_k) =
        resolve_sampling_defaults(req.temperature, req.top_p, req.top_k, false, model_defaults);
    let gen_params = GenParams {
        max_new_tokens: clamp_max_tokens(req.max_tokens, 16, max_tokens_cap),
        temperature,
        top_p,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        repetition_penalty: resolve_repetition_penalty(req.repetition_penalty),
        seed: req.seed,
        stop: parse_stop(req.stop.as_ref()),
        // G1 structured output is chat-only (per OpenAI); legacy completions
        // stay free-form.
        json_schema: None,
        top_k,
    };

    if stream && prompts.len() > 1 {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "streaming supports a single prompt; send prompts separately or set stream=false",
            "invalid_request_error",
            Some("multi_prompt_stream"),
        );
    }

    // Submit every prompt (real engine or mock), collecting a receiver each.
    let receivers =
        match submit_prompts(&state, &req.model, prompts, &gen_params, requested_at).await {
            Ok(r) => r,
            Err(resp) => return *resp,
        };

    if stream {
        // Single prompt guaranteed by the check above.
        let Some((rx, device_lease)) = receivers.into_iter().next() else {
            return openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "no inference stream was created",
                "server_error",
                None,
            );
        };
        return completion_stream(
            rx,
            id,
            req.model,
            created,
            include_usage,
            state.model_manager.clone(),
            device_lease,
        );
    }

    // `_device_leases` stays alive (NLL) until `collect_completions`'s
    // `.await` resolves — releasing every prompt's device token only once
    // its generation is fully drained, not merely submitted.
    let (rxs, _device_leases): (Vec<TokenReceiver>, Vec<crate::admission::WorkLease>) =
        receivers.into_iter().unzip();
    collect_completions(rxs, &id, &req.model, created, state.model_manager.as_ref()).await
}

/// Submit each prompt to the engine (or the mock generator) and return one
/// token receiver per prompt, in order.
///
/// The `Err` half is boxed to dodge `clippy::result_large_err` (a `Response`
/// is large). On a mid-batch capacity/engine error, receivers already pushed
/// are dropped on return — which cancels those in-flight generations via the
/// engine's client-disconnect path, so no work is orphaned.
async fn submit_prompts(
    state: &AppState,
    model: &str,
    prompts: Vec<String>,
    gen_params: &GenParams,
    requested_at: Instant,
) -> Result<Vec<(TokenReceiver, crate::admission::WorkLease)>, Box<Response>> {
    let mut receivers = Vec::with_capacity(prompts.len());

    match state.model_manager.as_ref() {
        Some(mm) => {
            let (handle, max_prompt_tokens) = mm
                .get_text_context(model)
                .map_err(|e| Box::new(model_error_response(e)))?;
            // Live device lookup — see `record_device`'s doc comment on why
            // this must not be cached across requests (admin reload can move
            // a model on its next load). Resolved once here since it cannot
            // change mid-request-array.
            let device = mm.record_device(model);

            for prompt in prompts {
                // T2.1: L0 prompt-length gate — the same gate_prompt the chat
                // path uses (an over-long prompt wedges engine.step() forever;
                // the Session-28 fix never propagated to this endpoint). T7.3:
                // reuse the gate's ids so add_request skips a second encode.
                let prompt_ids = gate_prompt(&handle, model, &prompt, max_prompt_tokens).await?;

                // Cross-pipeline device admission (step 2): a second gate in
                // front of the engine's own per-engine gate below, capping
                // concurrency across different engine kinds sharing this
                // model's device.
                let device_lease = match state
                    .device_budgets
                    .admit(&[(device.clone(), 1)], crate::admission::WorkClass::Chat)
                    .await
                {
                    Ok(lease) => lease,
                    Err(e) => return Err(Box::new(device_admission_error_response(&e))),
                };

                let (tx, rx) = stream_channel();
                let req_id = mm.next_id();
                match handle
                    .add_request(
                        req_id,
                        prompt,
                        prompt_ids,
                        gen_params.clone(),
                        tx,
                        requested_at,
                    )
                    .await
                {
                    SubmitResult::Submitted => receivers.push((rx, device_lease)),
                    SubmitResult::AtCapacity => {
                        return Err(Box::new(openai_error(
                            StatusCode::TOO_MANY_REQUESTS,
                            "server at capacity — all inference slots busy, retry shortly",
                            "rate_limit_error",
                            Some("server_overloaded"),
                        )));
                    }
                    SubmitResult::EngineDead => {
                        return Err(Box::new(openai_error(
                            StatusCode::SERVICE_UNAVAILABLE,
                            "inference engine unavailable (engine thread exited)",
                            "server_error",
                            Some("engine_unavailable"),
                        )));
                    }
                }
            }
        }
        None => {
            // Mock path (no GPU): one mock token stream per prompt.
            for _ in prompts {
                let (tx, rx) = stream_channel();
                spawn_mock_generation(tx);
                receivers.push((rx, crate::admission::WorkLease::default()));
            }
        }
    }

    Ok(receivers)
}

/// Drain every prompt's stream and assemble the non-streaming `text_completion`
/// object, summing token usage across all choices.
async fn collect_completions(
    receivers: Vec<TokenReceiver>,
    id: &str,
    model: &str,
    created: u64,
    mm: Option<&Arc<ModelManager>>,
) -> Response {
    let mut choices = Vec::with_capacity(receivers.len());
    let mut prompt_total = 0usize;
    let mut completion_total = 0usize;

    for (index, rx) in receivers.into_iter().enumerate() {
        let drained = match drain_channel(rx).await {
            Ok(d) => d,
            // T6.2: scrub raw C++ engine text; OOM → retryable 503 + poison-flag.
            Err(e) => return inference_error_from_text(mm, model, &e),
        };
        prompt_total += drained.prompt_tokens;
        completion_total += drained.completion_tokens;
        choices.push(CompletionChoice {
            index: u32::try_from(index).unwrap_or(u32::MAX),
            text: drained.content,
            logprobs: None,
            finish_reason: Some(drained.finish.as_openai()),
        });
    }

    let body = Completion {
        id,
        object: "text_completion",
        created,
        model,
        choices,
        usage: usage_for(prompt_total, completion_total),
    };
    Json(body).into_response()
}

/// Scan state for `completion_stream`'s SSE conversion.
struct CompletionScanState {
    prompt_tokens: usize,
    /// Count of non-empty raw `Token` events — proxy for generated tokens.
    completion_tokens: usize,
    /// Set once a mid-stream `Error` has been emitted (T3.2). Mirrors the
    /// chat path's `ScanState::errored`.
    errored: bool,
    /// Cross-pipeline device admission token for this stream — held for the
    /// full SSE response lifetime; dropped when the stream ends OR the client
    /// disconnects mid-stream. Never read — kept purely for its `Drop` side
    /// effect (mirrors `chat::ScanState::device_lease`).
    #[allow(dead_code)]
    device_lease: crate::admission::WorkLease,
}

/// Build the streaming SSE response for a single prompt: a run of
/// `text_completion` chunks, then a final finish chunk, an optional usage
/// chunk (when `include_usage`), then `[DONE]`.
fn completion_stream(
    rx: TokenReceiver,
    id: String,
    model: String,
    created: u64,
    include_usage: bool,
    mm: Option<Arc<ModelManager>>,
    device_lease: crate::admission::WorkLease,
) -> Response {
    // `errored` is latched by a mid-stream Error — the error arm emits the
    // full terminal sequence itself, and the flag suppresses the engine's
    // trailing Done(Stop) (T3.2; mirrors the chat path's ScanState).
    let token_stream = ReceiverStream::new(rx)
        .scan(
            CompletionScanState {
                prompt_tokens: 0,
                completion_tokens: 0,
                errored: false,
                device_lease,
            },
            |state, event| {
                // Capture whether an Error had already been emitted BEFORE
                // this event, so the converter can suppress the Done that
                // follows it.
                let was_errored = state.errored;
                match &event {
                    crate::streaming::StreamEvent::PromptTokens(n) => state.prompt_tokens = *n,
                    crate::streaming::StreamEvent::Token(t, new_tokens) if !t.is_empty() => {
                        // Sum the engine's own reported token count, not 1 per
                        // event — see the identical fix in chat.rs's
                        // ScanState::process (dev/DECISIONS.md 2026-07-19).
                        state.completion_tokens += *new_tokens;
                    }
                    crate::streaming::StreamEvent::Error(_) => state.errored = true,
                    _ => {}
                }
                std::future::ready(Some((
                    event,
                    (state.prompt_tokens, state.completion_tokens),
                    was_errored,
                )))
            },
        )
        .flat_map(move |(event, counts, was_errored)| {
            futures_util::stream::iter(completion_event_to_sse(
                event,
                &id,
                &model,
                created,
                counts,
                include_usage,
                was_errored,
                mm.as_ref(),
            ))
        });

    Sse::new(token_stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Convert one [`StreamEvent`] into zero or more `text_completion` SSE frames.
///
/// Unlike the chat path there is no leading `role` frame. `Done` expands to
/// the finish chunk, an optional usage chunk (when `include_usage`), and `[DONE]`.
/// `counts` is `(prompt_tokens, completion_tokens)` accumulated via `scan`;
/// `was_errored` is true when a previous event already terminated the stream
/// with an error (the trailing `Done` is then swallowed — T3.2).
// A pure event→frames mapper: every argument is a distinct render input
// (event, response identity, counts, flags, the manager for the T6.2b poison
// side effect). Bundling them into a struct would only relocate the same fields
// — the chat path uses `ScanState` because it carries *mutable* state; this one
// does not, so a flat signature is clearer.
#[allow(clippy::too_many_arguments)]
fn completion_event_to_sse(
    event: StreamEvent,
    id: &str,
    model: &str,
    created: u64,
    counts: (usize, usize),
    include_usage: bool,
    was_errored: bool,
    mm: Option<&Arc<ModelManager>>,
) -> Vec<Result<Event, Infallible>> {
    match event {
        StreamEvent::PromptTokens(_) => vec![],
        StreamEvent::Token(tok, _) => {
            vec![Ok(completion_chunk_event(id, model, created, tok, None))]
        }
        // The error arm already emitted the terminal sequence — swallow the
        // engine's trailing Done so the stream isn't terminated twice.
        StreamEvent::Done(_) if was_errored => vec![],
        StreamEvent::Done(reason) => {
            let mut frames = vec![Ok(completion_chunk_event(
                id,
                model,
                created,
                String::new(),
                Some(reason.as_openai()),
            ))];
            if include_usage {
                let (pt, ct) = counts;
                frames.push(Ok(usage_sse_event(
                    id,
                    "text_completion",
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
            // T3.1/T3.2/T6.2b: typed, valid-JSON error envelope SCRUBBED of raw
            // C++ diagnostics (never string-interpolated); on an OOM-class error
            // it also poison-flags the GPU via `mm`. Then a terminal chunk with
            // finish_reason:"error", then [DONE] so SDK clients can't hang.
            Ok(sse_inference_error_event(mm, model, &e)),
            Ok(completion_chunk_event(
                id,
                model,
                created,
                String::new(),
                Some("error"),
            )),
            Ok(Event::default().data("[DONE]")),
        ],
    }
}

/// Build one `text_completion` SSE chunk carrying `text` (and an optional
/// `finish_reason` on the final chunk).
fn completion_chunk_event(
    id: &str,
    model: &str,
    created: u64,
    text: String,
    finish_reason: Option<&'static str>,
) -> Event {
    let chunk = CompletionChunk {
        id,
        object: "text_completion",
        created,
        model,
        choices: vec![CompletionChoice {
            index: 0,
            text,
            logprobs: None,
            finish_reason,
        }],
    };
    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::ov_cb::FinishReason;

    /// `prompt` accepts a bare string → one prompt.
    #[test]
    fn prompt_deserialises_single_string() {
        let req: CompletionRequest =
            serde_json::from_str(r#"{"model":"m","prompt":"hello"}"#).unwrap();
        assert_eq!(req.prompt.into_vec(), vec!["hello".to_owned()]);
    }

    /// `prompt` accepts an array of strings → many prompts.
    #[test]
    fn prompt_deserialises_string_array() {
        let req: CompletionRequest =
            serde_json::from_str(r#"{"model":"m","prompt":["a","b"]}"#).unwrap();
        assert_eq!(req.prompt.into_vec(), vec!["a".to_owned(), "b".to_owned()]);
    }

    /// An absent `prompt` defaults to a single empty string (never an empty vec).
    #[test]
    fn prompt_absent_defaults_to_single_empty() {
        let req: CompletionRequest = serde_json::from_str(r#"{"model":"m"}"#).unwrap();
        assert_eq!(req.prompt.into_vec(), vec![String::new()]);
    }

    /// Truly-unknown `OpenAI` params (`echo`, `suffix`, …) are ignored, not
    /// rejected; the now-parsed `n: 1` is the accepted single-choice default.
    #[test]
    fn unknown_params_are_ignored() {
        let req: CompletionRequest = serde_json::from_str(
            r#"{"model":"m","prompt":"hi","temperature":0.7,"n":1,"echo":true}"#,
        )
        .unwrap();
        assert_eq!(req.model, "m");
        assert!(unsupported_completion_param(&req).is_none());
    }

    // ---- Unsupported-param 400 policy (legacy completions) ----------------

    fn cmpl(json: &str) -> CompletionRequest {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn completion_clean_request_has_no_unsupported_param() {
        assert!(unsupported_completion_param(&cmpl(r#"{"model":"m","prompt":"hi"}"#)).is_none());
    }

    #[test]
    fn completion_n_greater_than_one_is_rejected() {
        let req = cmpl(r#"{"model":"m","prompt":"hi","n":2}"#);
        assert_eq!(
            unsupported_completion_param(&req).unwrap().status(),
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn completion_best_of_greater_than_one_is_rejected() {
        let req = cmpl(r#"{"model":"m","prompt":"hi","best_of":2}"#);
        assert_eq!(
            unsupported_completion_param(&req).unwrap().status(),
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn completion_logprobs_present_is_rejected() {
        let req = cmpl(r#"{"model":"m","prompt":"hi","logprobs":2}"#);
        assert_eq!(
            unsupported_completion_param(&req).unwrap().status(),
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn completion_nonempty_logit_bias_is_rejected() {
        let req = cmpl(r#"{"model":"m","prompt":"hi","logit_bias":{"50256":-100}}"#);
        assert_eq!(
            unsupported_completion_param(&req).unwrap().status(),
            StatusCode::BAD_REQUEST
        );
    }

    /// T2.1: a prompt array beyond the fan-out cap is rejected with 400
    /// `too_many_prompts` before any engine work (mock mode uses the default
    /// cap of 16); a single prompt sails past the cap check.
    #[tokio::test]
    async fn prompt_array_beyond_cap_is_rejected() {
        use crate::app_state::AppState;
        use axum::extract::State;

        let prompts: Vec<String> = (0..17).map(|i| format!("p{i}")).collect();
        let req: CompletionRequest =
            serde_json::from_value(serde_json::json!({"model": "m", "prompt": prompts})).unwrap();

        let resp = completions(State(AppState::new()), JsonBody(req)).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "too_many_prompts");
    }

    /// Non-streaming collector buffers content into `text`, sets `text_completion`
    /// + null logprobs + finish reason, and reports usage.
    #[tokio::test]
    async fn collect_completions_builds_text_completion() {
        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            tx.send(StreamEvent::PromptTokens(5)).await.unwrap();
            tx.send(StreamEvent::Token("Warsaw".into(), 1))
                .await
                .unwrap();
            tx.send(StreamEvent::Token(".".into(), 1)).await.unwrap();
            tx.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });

        let resp = collect_completions(vec![rx], "cmpl-x", "m", 7, None).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(json["object"], "text_completion");
        assert_eq!(json["created"], 7);
        assert_eq!(json["choices"][0]["index"], 0);
        assert_eq!(json["choices"][0]["text"], "Warsaw.");
        assert!(
            json["choices"][0]["logprobs"].is_null(),
            "logprobs must be present and null"
        );
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["usage"]["prompt_tokens"], 5);
        assert_eq!(json["usage"]["completion_tokens"], 2);
        assert_eq!(json["usage"]["total_tokens"], 7);
    }

    /// Multiple prompts produce one indexed choice each, with summed usage.
    #[tokio::test]
    async fn collect_completions_handles_multiple_prompts() {
        let (tx0, rx0) = stream_channel();
        let (tx1, rx1) = stream_channel();
        tokio::spawn(async move {
            tx0.send(StreamEvent::PromptTokens(2)).await.unwrap();
            tx0.send(StreamEvent::Token("one".into(), 1)).await.unwrap();
            tx0.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });
        tokio::spawn(async move {
            tx1.send(StreamEvent::PromptTokens(3)).await.unwrap();
            tx1.send(StreamEvent::Token("two".into(), 1)).await.unwrap();
            tx1.send(StreamEvent::Token("!".into(), 1)).await.unwrap();
            tx1.send(StreamEvent::Done(FinishReason::Length))
                .await
                .unwrap();
        });

        let resp = collect_completions(vec![rx0, rx1], "id", "m", 0, None).await;
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(json["choices"][0]["index"], 0);
        assert_eq!(json["choices"][0]["text"], "one");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["choices"][1]["index"], 1);
        assert_eq!(json["choices"][1]["text"], "two!");
        assert_eq!(json["choices"][1]["finish_reason"], "length");
        // Usage sums across prompts: prompt 2+3=5, completion 1+2=3.
        assert_eq!(json["usage"]["prompt_tokens"], 5);
        assert_eq!(json["usage"]["completion_tokens"], 3);
        assert_eq!(json["usage"]["total_tokens"], 8);
    }

    /// The streaming form emits `text_completion` chunks, a finish chunk, and
    /// the `[DONE]` sentinel — and never a leading role frame.
    #[tokio::test]
    async fn completion_stream_emits_chunks_then_done() {
        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            tx.send(StreamEvent::Token("Warsaw".into(), 1))
                .await
                .unwrap();
            tx.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });

        let resp = completion_stream(
            rx,
            "id".to_owned(),
            "m".to_owned(),
            0,
            false,
            None,
            crate::admission::WorkLease::default(),
        );
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            body.contains(r#""object":"text_completion""#),
            "chunks must be text_completion: {body}"
        );
        assert!(
            body.contains(r#""text":"Warsaw""#),
            "must stream text: {body}"
        );
        assert!(
            body.contains(r#""finish_reason":"stop""#),
            "must finish with stop: {body}"
        );
        assert!(
            body.contains("[DONE]"),
            "must terminate with [DONE]: {body}"
        );
        assert!(
            !body.contains(r#""role""#),
            "legacy completions has no role frame: {body}"
        );
    }

    /// T3.1/T3.2 on the legacy path: mid-stream error → valid-JSON envelope,
    /// terminal chunk with `finish_reason:"error"`, exactly one `[DONE]`; the
    /// engine's trailing `Done(Stop)` is suppressed.
    #[tokio::test]
    async fn completion_stream_error_emits_envelope_terminal_and_single_done() {
        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            tx.send(StreamEvent::Token("par".into(), 1)).await.unwrap();
            tx.send(StreamEvent::Error("engine \"died\"\nmid-step".into()))
                .await
                .unwrap();
            tx.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });

        let resp = completion_stream(
            rx,
            "id".to_owned(),
            "m".to_owned(),
            0,
            false,
            None,
            crate::admission::WorkLease::default(),
        );
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();

        let err_line = body
            .lines()
            .find(|l| l.starts_with("data:") && l.contains("\"error\""))
            .unwrap(); // an error frame must be present
        let json: serde_json::Value =
            serde_json::from_str(err_line.trim_start_matches("data:").trim()).unwrap(); // must be valid JSON despite quotes/newlines
        assert_eq!(json["error"]["type"], "server_error");

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
            "trailing Done(Stop) must be suppressed: {body}"
        );
    }

    /// With `include_usage`, the stream emits a usage chunk (choices:[], usage:{…})
    /// between the finish chunk and `[DONE]`.
    #[tokio::test]
    async fn completion_stream_include_usage_emits_usage_chunk() {
        use crate::ov_cb::FinishReason;
        use crate::streaming::StreamEvent;

        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            tx.send(StreamEvent::PromptTokens(7)).await.unwrap();
            tx.send(StreamEvent::Token("Hello".into(), 1))
                .await
                .unwrap();
            tx.send(StreamEvent::Token(" world".into(), 1))
                .await
                .unwrap();
            tx.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });

        let resp = completion_stream(
            rx,
            "cmpl-x".to_owned(),
            "m".to_owned(),
            0,
            true,
            None,
            crate::admission::WorkLease::default(),
        );
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            body.contains("\"prompt_tokens\":7"),
            "must include prompt_tokens: {body}"
        );
        assert!(
            body.contains("\"completion_tokens\":2"),
            "must include completion_tokens: {body}"
        );
        assert!(
            body.contains("\"total_tokens\":9"),
            "must include total_tokens: {body}"
        );
        assert!(
            body.contains("\"choices\":[]"),
            "usage chunk must have empty choices: {body}"
        );
        // [DONE] must still terminate the stream.
        assert!(
            body.contains("[DONE]"),
            "must still terminate with [DONE]: {body}"
        );
    }

    /// Regression test for the 2026-07-19 undercount (the project's internal engineering log):
    /// the streaming completions path must sum each `Token` event's reported
    /// count, not count events — speculative decoding's verification step
    /// can accept several draft tokens at once, all landing in one event.
    #[tokio::test]
    async fn completion_stream_sums_multi_token_events_not_event_count() {
        use crate::ov_cb::FinishReason;
        use crate::streaming::StreamEvent;

        let (tx, rx) = stream_channel();
        tokio::spawn(async move {
            tx.send(StreamEvent::Token("assisted chunk".into(), 4))
                .await
                .unwrap();
            tx.send(StreamEvent::Token(" more".into(), 3))
                .await
                .unwrap();
            tx.send(StreamEvent::Done(FinishReason::Stop))
                .await
                .unwrap();
        });

        let resp = completion_stream(
            rx,
            "cmpl-x".to_owned(),
            "m".to_owned(),
            0,
            true,
            None,
            crate::admission::WorkLease::default(),
        );
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            body.contains("\"completion_tokens\":7"),
            "must sum 4+3=7 real tokens across 2 events, not count 2 events: {body}"
        );
    }
}
