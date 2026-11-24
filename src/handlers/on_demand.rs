//! F1b: the one on-demand wait every inference route uses, except `/tokenize` and
//! `/detokenize`, which never load a model (P-C).
//!
//! A route resolves its model as before. When that says `NotLoaded` (or `Loading`, for an
//! `on_demand` model while a wait is configured), [`resolve_waiting`] hands the request to
//! [`ModelManager::wait_on_demand`]: with `on_demand_wait_secs: 0` that is exactly the old
//! answer (start the load, `503` + `Retry-After`); with a wait the request holds until the model
//! is Ready, the load fails, or the wait expires. After a Ready the model is resolved again,
//! once: if it is gone again (evicted in between) its load is started again and the request gets
//! the `503` — never a second wait, never a loop.
//!
//! Streaming chat and completions start their `200` SSE response before the wait
//! ([`sse_after_wait`]): the client gets SSE comment keep-alives every [`SSE_KEEP_ALIVE`] while
//! it waits, and any error after that is an SSE error frame, as for every streaming error.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;

use super::error::{model_error_response, openai_error};
use crate::model_manager::on_demand::{OnDemandWait, WaitPlace, WaiterSlot};
use crate::model_manager::{ModelError, ModelManager, OnDemandLoad};

/// Why a route's model is not usable for this request (the route maps it to its own answer).
#[derive(Debug)]
pub(crate) enum Unready {
    /// The on-demand path's answer, as before F1b: `Loading` (load started or running, also on
    /// wait expiry and at the waiter cap: `503` + `Retry-After`), `ShuttingDown`, or
    /// `NotApplicable` (an eager model: the plain not-loaded `503`).
    OnDemand(OnDemandLoad),
    /// The model lookup's own error (unknown id, wrong kind, evicting, ...).
    Model(ModelError),
    /// The load this request waited for failed: its [`ModelError`] (answered through the route's
    /// usual mapping, e.g. `504 load_timeout`, `503 insufficient_memory`), or `None` for any
    /// other failure ([`load_failed_response`]).
    LoadFailed(Option<ModelError>),
}

impl Unready {
    /// The `OpenAI`-shaped answer: the same mapping the route used before F1b
    /// ([`model_error_response`]), plus [`load_failed_response`] for a load that failed with an
    /// error that is not a [`ModelError`].
    pub(crate) fn into_response(self, model: &str) -> Response {
        match self {
            Self::OnDemand(OnDemandLoad::Loading) => model_error_response(ModelError::Loading),
            Self::OnDemand(OnDemandLoad::ShuttingDown) => {
                model_error_response(ModelError::ShuttingDown)
            }
            Self::OnDemand(OnDemandLoad::NotApplicable) => {
                model_error_response(ModelError::NotLoaded)
            }
            Self::Model(e) | Self::LoadFailed(Some(e)) => model_error_response(e),
            Self::LoadFailed(None) => load_failed_response(model),
        }
    }
}

/// `500 server_error` (no code) for a waited-for load that failed with an error that is not a
/// [`ModelError`] (an engine build failure). The raw error stays in the server log (the load's
/// WARN line): it can carry `OpenVINO` diagnostics that never go to a client (T6.2).
pub(crate) fn load_failed_response(model: &str) -> Response {
    openai_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("model '{model}' failed to load; see the server log"),
        "server_error",
        None,
    )
}

/// Resolve `model` for a request, waiting for its on-demand load when one applies.
///
/// `first` is the route's first lookup; `again` repeats it (once, after the wait). A lookup that
/// succeeded returns at once (a `slot` passed in is released). Only `NotLoaded`, and `Loading`
/// of an `on_demand` model while a wait is configured, go to the wait; every other error is
/// returned as [`Unready::Model`], as the route answered it before.
///
/// `place`: where the wait gets its waiter place ([`WaitPlace::Held`]: the one a streaming
/// route took before it started its response; [`WaitPlace::NoWait`]: a streaming route that
/// found none).
///
/// Cancel-safe: dropping this future (the client went away) drops the wait — its waiter place
/// is released, the load keeps running.
pub(crate) async fn resolve_waiting<T, F, Fut>(
    mm: &Arc<ModelManager>,
    model: &str,
    first: Result<T, ModelError>,
    place: WaitPlace,
    mut again: F,
) -> Result<T, Unready>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, ModelError>>,
{
    match first {
        Ok(t) => return Ok(t),
        Err(ModelError::NotLoaded) => {}
        Err(ModelError::Loading) if mm.on_demand_wait_applies(model) => {}
        Err(e) => return Err(Unready::Model(e)),
    }
    match mm.wait_on_demand(model, place).await {
        OnDemandWait::Settled => match again().await {
            Ok(t) => Ok(t),
            // Gone again before the request got in (evicted, or the load was abandoned): start
            // the load again and answer as without a wait. Never a second wait.
            Err(ModelError::NotLoaded) => Err(Unready::OnDemand(mm.request_on_demand_load(model))),
            Err(e) => Err(Unready::Model(e)),
        },
        OnDemandWait::Refused(r) => Err(Unready::OnDemand(r)),
        OnDemandWait::LoadFailed(e) => Err(Unready::LoadFailed(e)),
    }
}

/// [`resolve_waiting`] for a synchronous lookup (every on-demand route but chat; `/tokenize`
/// and `/detokenize` do not use it).
pub(crate) async fn resolve_with_wait<T>(
    mm: &Arc<ModelManager>,
    model: &str,
    lookup: impl Fn() -> Result<T, ModelError>,
) -> Result<T, Unready> {
    let first = lookup();
    resolve_waiting(mm, model, first, WaitPlace::Take, || {
        std::future::ready(lookup())
    })
    .await
}

/// For a streaming request whose first lookup is `first`: a waiter place when the request is to
/// wait for an on-demand load (the route then answers with [`sse_after_wait`]), else `None`:
/// no wait applies, a stalled load is in the way (answered at once, as an HTTP error), or the
/// waiter cap is reached. With `None` the route resolves with [`WaitPlace::NoWait`].
pub(crate) fn streaming_wait_slot<T>(
    mm: &ModelManager,
    model: &str,
    first: &Result<T, ModelError>,
) -> Option<WaiterSlot> {
    if matches!(first, Err(ModelError::NotLoaded | ModelError::Loading))
        && mm.on_demand_wait_applies(model)
        && mm.on_demand_stall_answer(model).is_none()
    {
        mm.try_on_demand_waiter_slot()
    } else {
        None
    }
}

/// Keep-alive interval of a streaming response that waits for an on-demand load: axum's
/// `KeepAlive::default()` interval, which the SSE responses use once generation runs.
pub(crate) const SSE_KEEP_ALIVE: Duration = Duration::from_secs(15);

/// The keep-alive frame: an empty SSE comment, the same bytes axum's `KeepAlive` sends.
const KEEP_ALIVE_FRAME: &[u8] = b":\n\n";

/// The keep-alive interval of a waited stream: [`SSE_KEEP_ALIVE`] (a test shortens it on its
/// own manager).
pub(crate) fn keep_alive_every(mm: Option<&ModelManager>) -> Duration {
    mm.map_or(SSE_KEEP_ALIVE, |mm| mm.on_demand_keep_alive(SSE_KEEP_ALIVE))
}

/// The answer once `fut` resolved: the `200` SSE body to forward, or the frames of an error.
enum Spliced {
    Body(axum::body::BodyDataStream),
    Frames(Bytes),
}

/// What [`sse_after_wait`]'s body is doing.
enum Phase {
    Waiting(
        Pin<Box<dyn Future<Output = Response> + Send>>,
        tokio::time::Interval,
    ),
    Forwarding(axum::body::BodyDataStream),
    Done,
}

/// A `200` `text/event-stream` response that sends SSE keep-alive comments (every `every`)
/// while `fut` runs
/// (the wait for an on-demand load, then the route's own handling), then:
/// - `fut` answered a streaming `200`: its body, frame for frame;
/// - `fut` answered an error: that error's JSON envelope as one SSE `data:` frame (the shape of
///   every streaming error frame, `{"error": {...}}`), then `data: [DONE]`. A `Retry-After` the
///   error carried cannot travel in an SSE body; the message says what to do.
///
/// Built in place (no spawned task): when the client goes away the body is dropped, and with it
/// `fut` and the wait inside it, which releases the waiter place at once.
pub(crate) fn sse_after_wait<F>(every: Duration, fut: F) -> Response
where
    F: Future<Output = Response> + Send + 'static,
{
    let mut tick = tokio::time::interval_at(tokio::time::Instant::now() + every, every);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let start = Phase::Waiting(Box::pin(fut), tick);
    let stream = futures_util::stream::unfold(start, |phase| async move {
        match phase {
            Phase::Waiting(mut fut, mut tick) => {
                tokio::select! {
                    resp = &mut fut => match splice(resp).await {
                        Spliced::Body(mut body) => {
                            let next = body.next().await?;
                            Some((next, Phase::Forwarding(body)))
                        }
                        Spliced::Frames(bytes) => Some((Ok(bytes), Phase::Done)),
                    },
                    _ = tick.tick() => Some((
                        Ok(Bytes::from_static(KEEP_ALIVE_FRAME)),
                        Phase::Waiting(fut, tick),
                    )),
                }
            }
            Phase::Forwarding(mut body) => {
                let next = body.next().await?;
                Some((next, Phase::Forwarding(body)))
            }
            Phase::Done => None,
        }
    });
    let mut resp = Body::from_stream(stream).into_response();
    let headers = resp.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(EVENT_STREAM));
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    resp
}

/// The SSE media type.
const EVENT_STREAM: &str = "text/event-stream";

/// Most bytes of an error body re-sent as an SSE frame (error envelopes are a few hundred).
const ERROR_BODY_LIMIT: usize = 64 * 1024;

/// The error frame used when an error body cannot be read back as JSON (not expected: every
/// error here is an `openai_error` envelope). A constant, valid JSON.
const FALLBACK_ERROR_FRAME: &str = r#"{"error":{"message":"the request failed before streaming started","type":"server_error","param":null,"code":null}}"#;

/// See [`sse_after_wait`].
async fn splice(resp: Response) -> Spliced {
    let is_stream = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with(EVENT_STREAM));
    if resp.status() == StatusCode::OK && is_stream {
        return Spliced::Body(resp.into_body().into_data_stream());
    }
    let status = resp.status();
    // Re-serialised, not forwarded raw: one compact line is one valid SSE `data:` field.
    let json = axum::body::to_bytes(resp.into_body(), ERROR_BODY_LIMIT)
        .await
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .map_or_else(|| FALLBACK_ERROR_FRAME.to_owned(), |v| v.to_string());
    tracing::debug!(%status, "streaming request ended before it started: sent as an SSE error frame");
    Spliced::Frames(Bytes::from(format!("data: {json}\n\ndata: [DONE]\n\n")))
}

// ============================================================
// Unit tests: every route, through the router, with mock engines
// ============================================================

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use axum::Router;
    use axum::http::{HeaderMap, Request};
    use http_body_util::BodyExt as _;
    use tower::ServiceExt as _;

    use super::*;
    use crate::app_state::AppState;
    use crate::model_manager::config::LoadPolicy;
    use crate::model_manager::lifecycle::{LoadGate, MockEngineFactory};
    use crate::model_manager::on_demand::MAX_ON_DEMAND_WAITERS;

    const TEXT: &str = "m-text";
    const LONG: Duration = Duration::from_secs(30);
    const BOUNDARY: &str = "F1BTESTBOUNDARY";

    /// One model per kind (the mock detects the kind from the name), all `on_demand` unless
    /// `eager`; loads held at `gate` when given; the wait set to `wait`.
    async fn app(
        wait: Duration,
        gate: Option<Arc<LoadGate>>,
        fail: bool,
        eager: bool,
    ) -> (Router, Arc<ModelManager>) {
        let ids = [
            TEXT,
            "m-embed",
            "m-rerank",
            "m-decide",
            "m-stt",
            "m-tts",
            "m-imagegen",
            "m-vlm",
        ];
        let (app, mm, _) = app_with(&ids, wait, gate, fail, eager, None).await;
        (app, mm)
    }

    /// The `requested_at` of every request the mock text engines received.
    type RequestedAt = Arc<std::sync::Mutex<Vec<std::time::Instant>>>;

    /// [`app`] with its own model ids; `kind_of` (when given) declares each model's `kind` in
    /// the config (the mock otherwise detects it from the name).
    async fn app_with(
        ids: &[&str],
        wait: Duration,
        gate: Option<Arc<LoadGate>>,
        fail: bool,
        eager: bool,
        kind_of: Option<fn(&str) -> &'static str>,
    ) -> (Router, Arc<ModelManager>, RequestedAt) {
        let models: Vec<(&str, f64)> = ids.iter().map(|id| (*id, 1.0)).collect();
        let mut cfg = crate::model_manager::tests::make_config(&models, 40.0);
        for (id, e) in &mut cfg.models {
            e.policy.load = if eager {
                LoadPolicy::Eager
            } else {
                LoadPolicy::OnDemand
            };
            if let Some(kind_of) = kind_of {
                e.kind = Some(kind_of(id).to_owned());
            }
        }
        let factory = MockEngineFactory {
            load_gate: gate,
            should_fail: fail,
            ..Default::default()
        };
        let requested_at = Arc::clone(&factory.recorded_requested_at);
        let mm = Arc::new(ModelManager::new(cfg, Arc::new(factory)).await.unwrap());
        mm.set_on_demand_wait_for_test(wait);
        mm.set_keep_alive_for_test(Duration::from_millis(20));
        let app = crate::create_router(AppState::new().with_model_manager(Arc::clone(&mm)));
        (app, mm, requested_at)
    }

    fn multipart(fields: &[(&str, Option<&str>, &[u8])]) -> Vec<u8> {
        let mut b = Vec::new();
        for &(name, file, content) in fields {
            b.extend_from_slice(format!("--{BOUNDARY}\r\n").as_bytes());
            let disposition = file.map_or_else(
                || format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n"),
                |f| {
                    format!(
                        "Content-Disposition: form-data; name=\"{name}\"; filename=\"{f}\"\r\n\
                         Content-Type: application/octet-stream\r\n\r\n"
                    )
                },
            );
            b.extend_from_slice(disposition.as_bytes());
            b.extend_from_slice(content);
            b.extend_from_slice(b"\r\n");
        }
        b.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
        b
    }

    /// The routes that never start a load, at any wait (a token count is not a reason to
    /// load a model).
    const NEVER_LOAD: [&str; 2] = ["/tokenize", "/detokenize"];

    /// Every route on the on-demand path: [`all_routes`] without [`NEVER_LOAD`].
    fn routes() -> Vec<(&'static str, &'static str, Request<Body>)> {
        all_routes()
            .into_iter()
            .filter(|(n, _, _)| !NEVER_LOAD.contains(n))
            .collect()
    }

    /// Every route that resolves a model: (path, model, request).
    #[allow(clippy::too_many_lines)] // one table
    fn all_routes() -> Vec<(&'static str, &'static str, Request<Body>)> {
        let json = |path: &str, v: serde_json::Value| {
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(v.to_string()))
                .unwrap()
        };
        // The mock text engine answers a prompt carrying this marker with two tokens and `Done`
        // (without it the request's channel just closes).
        let ok = crate::stall_watch::mock::OK;
        let user = serde_json::json!([{"role": "user", "content": ok}]);
        let audio: [(&str, Option<&str>, &[u8]); 2] = [
            ("model", None, b"m-stt"),
            ("file", Some("a.wav"), b"RIFF0000WAVE"),
        ];
        let image = png();
        let edit: [(&str, Option<&str>, &[u8]); 3] = [
            ("model", None, b"m-imagegen"),
            ("prompt", None, b"a cat"),
            ("image", Some("source.png"), image.as_slice()),
        ];
        vec![
            (
                "/v1/chat/completions",
                TEXT,
                json(
                    "/v1/chat/completions",
                    serde_json::json!({"model": TEXT, "messages": user}),
                ),
            ),
            (
                "/v1/chat/completions (stream)",
                TEXT,
                json(
                    "/v1/chat/completions",
                    serde_json::json!({"model": TEXT, "messages": user, "stream": true}),
                ),
            ),
            (
                "/v1/completions",
                TEXT,
                json(
                    "/v1/completions",
                    serde_json::json!({"model": TEXT, "prompt": ok}),
                ),
            ),
            (
                "/v1/completions (stream)",
                TEXT,
                json(
                    "/v1/completions",
                    serde_json::json!({"model": TEXT, "prompt": ok, "stream": true}),
                ),
            ),
            (
                "/tokenize",
                TEXT,
                json(
                    "/tokenize",
                    serde_json::json!({"model": TEXT, "prompt": "hi"}),
                ),
            ),
            (
                "/detokenize",
                TEXT,
                json(
                    "/detokenize",
                    serde_json::json!({"model": TEXT, "tokens": [1, 2]}),
                ),
            ),
            (
                "/v1/embeddings",
                "m-embed",
                json(
                    "/v1/embeddings",
                    serde_json::json!({"model": "m-embed", "input": "hi"}),
                ),
            ),
            (
                "/v1/rerank",
                "m-rerank",
                json(
                    "/v1/rerank",
                    serde_json::json!({"model": "m-rerank", "query": "q", "documents": ["a"]}),
                ),
            ),
            (
                "/v1/systemone",
                "m-decide",
                json(
                    "/v1/systemone",
                    serde_json::json!({
                        "model": "m-decide", "state": "s",
                        "questions": {"refund": {"type": "noul", "instructions": "Refund?"}}
                    }),
                ),
            ),
            (
                "/v1/audio/speech",
                "m-tts",
                json(
                    "/v1/audio/speech",
                    serde_json::json!({"model": "m-tts", "input": "hi"}),
                ),
            ),
            (
                "/v1/images/generations",
                "m-imagegen",
                json(
                    "/v1/images/generations",
                    serde_json::json!({"model": "m-imagegen", "prompt": "a cat", "size": "256x256"}),
                ),
            ),
            (
                "/v1/audio/transcriptions",
                "m-stt",
                form("/v1/audio/transcriptions", &audio),
            ),
            (
                "/v1/audio/translations",
                "m-stt",
                form("/v1/audio/translations", &audio),
            ),
            (
                "/v1/images/edits",
                "m-imagegen",
                form("/v1/images/edits", &edit),
            ),
        ]
    }

    fn form(path: &str, fields: &[(&str, Option<&str>, &[u8])]) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(path)
            .header(
                "content-type",
                format!("multipart/form-data; boundary={BOUNDARY}"),
            )
            .body(Body::from(multipart(fields)))
            .unwrap()
    }

    /// A 256 x 256 PNG (the smallest source image edits accept).
    fn png() -> Vec<u8> {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            256,
            256,
            image::Rgb([200, 150, 100]),
        ));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    }

    async fn send(app: &Router, req: Request<Body>) -> (StatusCode, HeaderMap, String) {
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let headers = resp.headers().clone();
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        (status, headers, String::from_utf8_lossy(&body).into_owned())
    }

    fn state_of(mm: &ModelManager, id: &str) -> Option<String> {
        mm.list_models()
            .into_iter()
            .find(|m| m.id == id)
            .map(|m| format!("{:?}", m.state))
    }

    async fn until(what: &str, mut cond: impl FnMut() -> bool) {
        for _ in 0..1000 {
            if cond() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("timed out waiting for: {what}");
    }

    /// Wait 0 (the shipped default): every on-demand route answers a not-loaded `on_demand`
    /// model with 503 + `Retry-After` at once and starts its load — as before F1b, and now also
    /// `/v1/completions` (which used to answer the plain not-loaded 503 and load nothing). No
    /// route waits. `/tokenize` and `/detokenize` never load: see
    /// [`tokenize_and_detokenize_never_load_or_wait`].
    #[tokio::test]
    async fn wait_zero_answers_503_retry_after_on_every_route_and_starts_the_load() {
        for (name, model, req) in routes() {
            let gate = Arc::new(LoadGate::default());
            let (app, mm) = app(Duration::ZERO, Some(Arc::clone(&gate)), false, false).await;
            let (status, headers, body) =
                tokio::time::timeout(Duration::from_secs(5), send(&app, req))
                    .await
                    .unwrap_or_else(|_| panic!("{name}: waited at wait 0"));
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{name}: {body}");
            assert_eq!(
                headers
                    .get(header::RETRY_AFTER)
                    .map(|v| v.to_str().unwrap()),
                Some("5"),
                "{name}: {body}"
            );
            // The same body as before F1b: the route's own "loading" answer (Ollama's shape on
            // System One, the OpenAI envelope with `model_unavailable` elsewhere).
            assert!(body.contains("loading"), "{name}: {body}");
            if name != "/v1/systemone" {
                assert!(body.contains("\"model_unavailable\""), "{name}: {body}");
            }
            until(&format!("{name}: the load started"), || {
                state_of(&mm, model).is_some_and(|s| s.contains("Loading"))
            })
            .await;
            assert_eq!(mm.on_demand_waiters(), 0, "{name}");
            gate.release();
        }
    }

    /// `/tokenize` and `/detokenize` never load a model. A not-loaded
    /// `on_demand` model gets the plain not-loaded 503 (no `Retry-After`) at once at every
    /// wait value; no load is started and no waiter place is taken. A model whose load another
    /// route started is answered at once too (the transient `Loading` 503), never waited for.
    #[tokio::test]
    async fn tokenize_and_detokenize_never_load_or_wait() {
        for wait in [Duration::ZERO, Duration::from_secs(1), LONG] {
            for (name, model, req) in all_routes()
                .into_iter()
                .filter(|(n, _, _)| NEVER_LOAD.contains(n))
            {
                let (app, mm) = app(wait, None, false, false).await;
                let (status, headers, body) =
                    tokio::time::timeout(Duration::from_secs(5), send(&app, req))
                        .await
                        .unwrap_or_else(|_| panic!("{name} at {wait:?}: waited"));
                assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{name}: {body}");
                assert!(headers.get(header::RETRY_AFTER).is_none(), "{name}: {body}");
                assert!(body.contains("\"model_unavailable\""), "{name}: {body}");
                assert!(body.contains("not loaded"), "{name}: {body}");
                // Review R-6: no advice to set the model to load on demand (it may already be).
                assert!(
                    body.contains("never load a model") && !body.contains("on demand"),
                    "{name}: {body}"
                );
                assert_eq!(mm.on_demand_waiters(), 0, "{name}");
                // Give a wrongly started load time to show.
                tokio::time::sleep(Duration::from_millis(50)).await;
                assert!(
                    state_of(&mm, model).is_some_and(|s| s.contains("NotLoaded")),
                    "{name} at {wait:?}: no load started"
                );
            }
        }
        // A load started by chat: the tokenizer routes answer at once while it runs.
        let gate = Arc::new(LoadGate::default());
        let (app, mm) = app(LONG, Some(Arc::clone(&gate)), false, false).await;
        assert!(matches!(
            mm.request_on_demand_load(TEXT),
            OnDemandLoad::Loading
        ));
        // Review R-7: the transient answers carry `Retry-After` (INTERFACE.md, tokenize section),
        // for a running load, a stalled one (its caller went away) and an eviction alike; none
        // waits, and none starts another load.
        let transient = |app: Router, mm: Arc<ModelManager>, what: &'static str| async move {
            for (name, _, req) in all_routes()
                .into_iter()
                .filter(|(n, _, _)| NEVER_LOAD.contains(n))
            {
                let (status, headers, body) =
                    tokio::time::timeout(Duration::from_secs(5), send(&app, req))
                        .await
                        .unwrap_or_else(|_| panic!("{name} {what}: waited"));
                assert_eq!(
                    status,
                    StatusCode::SERVICE_UNAVAILABLE,
                    "{name} {what}: {body}"
                );
                assert_eq!(
                    headers
                        .get(header::RETRY_AFTER)
                        .map(|v| v.to_str().unwrap()),
                    Some("5"),
                    "{name} {what}: {body}"
                );
                assert_eq!(mm.on_demand_waiters(), 0, "{name} {what}");
            }
        };
        // The started load's task sets `Loading` when it runs.
        until("loading", || {
            state_of(&mm, TEXT).is_some_and(|s| s.contains("Loading"))
        })
        .await;
        transient(app.clone(), Arc::clone(&mm), "loading").await;
        assert!(state_of(&mm, TEXT).unwrap().contains("Loading"));
        gate.release();

        // Stalled: the admin caller went away during the engine build.
        let gate = Arc::new(LoadGate::default());
        // `self::app`: the local `app` above shadows the helper.
        let (app, mm) = self::app(LONG, Some(Arc::clone(&gate)), false, false).await;
        let admin = tokio::spawn({
            let mm = Arc::clone(&mm);
            async move { mm.load_model(TEXT).await }
        });
        until("loading", || {
            state_of(&mm, TEXT).is_some_and(|s| s.contains("Loading"))
        })
        .await;
        tokio::time::sleep(Duration::from_millis(30)).await;
        admin.abort();
        let _ = admin.await;
        until("booked as stalled", || !mm.stalled_loads().is_empty()).await;
        transient(app.clone(), Arc::clone(&mm), "stalled").await;
        gate.release();

        // Evicting.
        let (app, mm) = self::app(LONG, None, false, false).await;
        mm.set_state_for_test(TEXT, crate::model_manager::ModelState::Evicting);
        transient(app.clone(), Arc::clone(&mm), "evicting").await;
        assert!(state_of(&mm, TEXT).unwrap().contains("Evicting"), "no load");
    }

    /// Wait 0 and an eager model: unchanged — the plain not-loaded 503, no `Retry-After`, no
    /// load. (Also with a wait configured: an eager model never waits.)
    #[tokio::test]
    async fn an_eager_model_is_never_loaded_or_waited_for() {
        for wait in [Duration::ZERO, LONG] {
            for (name, model, req) in all_routes() {
                let (app, mm) = app(wait, None, false, true).await;
                let (status, headers, body) =
                    tokio::time::timeout(Duration::from_secs(5), send(&app, req))
                        .await
                        .unwrap_or_else(|_| panic!("{name}: waited for an eager model"));
                assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{name}: {body}");
                assert!(headers.get(header::RETRY_AFTER).is_none(), "{name}");
                assert!(
                    state_of(&mm, model).is_some_and(|s| s.contains("NotLoaded")),
                    "{name}"
                );
            }
        }
    }

    /// Wait > 0: every route holds while the load runs and then serves the request (200).
    #[tokio::test]
    async fn with_a_wait_every_route_serves_once_the_model_is_ready() {
        for (name, model, req) in routes() {
            let gate = Arc::new(LoadGate::default());
            let (app, mm) = app(LONG, Some(Arc::clone(&gate)), false, false).await;
            let call = tokio::spawn({
                let app = app.clone();
                async move { send(&app, req).await }
            });
            until(&format!("{name}: waiting"), || {
                state_of(&mm, model).is_some_and(|s| s.contains("Loading"))
            })
            .await;
            gate.release();
            let (status, _, body) = tokio::time::timeout(Duration::from_secs(10), call)
                .await
                .unwrap_or_else(|_| panic!("{name}: no answer after Ready"))
                .unwrap();
            assert_eq!(status, StatusCode::OK, "{name}: {body}");
            assert_eq!(mm.on_demand_waiters(), 0, "{name}");
        }
    }

    /// Wait > 0, the load fails: a non-streaming route answers the load's failure (here not a
    /// `ModelError`: 500 "failed to load", no raw engine text); a streaming one the same error
    /// as an SSE error frame after its keep-alives, then `[DONE]`.
    #[tokio::test]
    async fn a_failed_load_reaches_the_waiting_request() {
        for (name, _, req) in routes() {
            let gate = Arc::new(LoadGate::default());
            let (app, mm) = app(LONG, Some(Arc::clone(&gate)), true, false).await;
            let call = tokio::spawn({
                let app = app.clone();
                async move { send(&app, req).await }
            });
            until(&format!("{name}: waiting"), || mm.on_demand_waiters() == 1).await;
            tokio::time::sleep(Duration::from_millis(60)).await;
            gate.release();
            let (status, _, body) = tokio::time::timeout(Duration::from_secs(10), call)
                .await
                .unwrap()
                .unwrap();
            assert!(body.contains("failed to load"), "{name}: {body}");
            assert!(
                !body.contains("should_fail"),
                "{name}: raw text leaked: {body}"
            );
            if name.ends_with("(stream)") {
                assert_eq!(status, StatusCode::OK, "{name}");
                assert!(
                    body.starts_with(":\n\n"),
                    "{name}: keep-alive first: {body}"
                );
                assert!(
                    body.contains("data: {\"error\":") && body.ends_with("data: [DONE]\n\n"),
                    "{name}: {body}"
                );
            } else {
                assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{name}: {body}");
            }
            assert_eq!(mm.on_demand_waiters(), 0, "{name}");
        }
    }

    /// Streaming chat and completions: keep-alive comments while the load runs, then the
    /// normal stream (role/text chunks and `[DONE]`).
    #[tokio::test]
    async fn streaming_sends_keep_alives_then_the_stream() {
        for (name, _, req) in routes()
            .into_iter()
            .filter(|(n, _, _)| n.ends_with("(stream)"))
        {
            let gate = Arc::new(LoadGate::default());
            let (app, mm) = app(LONG, Some(Arc::clone(&gate)), false, false).await;
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "{name}: headers before Ready"
            );
            assert_eq!(
                resp.headers().get(header::CONTENT_TYPE).unwrap(),
                "text/event-stream",
                "{name}"
            );
            let mut body = resp.into_body();
            let first = body.frame().await.unwrap().unwrap().into_data().unwrap();
            assert_eq!(&first[..], b":\n\n", "{name}: a keep-alive while waiting");
            assert_eq!(mm.on_demand_waiters(), 1, "{name}");
            gate.release();
            let tail = body.collect().await.unwrap().to_bytes();
            let tail = String::from_utf8_lossy(&tail);
            // The engine's own frames follow: text chunks, the finish chunk, `[DONE]`.
            let object = if name.contains("chat") {
                "chat.completion.chunk"
            } else {
                "text_completion"
            };
            assert!(tail.contains(object), "{name}: {tail}");
            assert!(tail.contains("tok0"), "{name}: {tail}");
            assert!(
                tail.contains("\"finish_reason\":\"stop\""),
                "{name}: {tail}"
            );
            assert!(tail.ends_with("data: [DONE]\n\n"), "{name}: {tail}");
            assert!(!tail.contains("\"error\""), "{name}: {tail}");
            assert_eq!(mm.on_demand_waiters(), 0, "{name}");
        }
    }

    /// A whole streaming chat after the wait, on an engine that ends its stream (the mock vision
    /// engine): keep-alives, the role and text chunks, the finish chunk, `[DONE]`.
    #[tokio::test]
    async fn a_waited_streaming_chat_runs_to_done() {
        let gate = Arc::new(LoadGate::default());
        let (app, mm) = app(LONG, Some(Arc::clone(&gate)), false, false).await;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "model": "m-vlm", "stream": true,
                    "messages": [{"role": "user", "content": "hi"}]
                })
                .to_string(),
            ))
            .unwrap();
        let call = tokio::spawn({
            let app = app.clone();
            async move { send(&app, req).await }
        });
        until("waiting", || mm.on_demand_waiters() == 1).await;
        tokio::time::sleep(Duration::from_millis(60)).await;
        gate.release();
        let (status, _, body) = call.await.unwrap();
        assert_eq!(status, StatusCode::OK);
        assert!(body.starts_with(":\n\n"), "{body}");
        assert!(body.contains("\"finish_reason\":\"stop\""), "{body}");
        assert!(body.ends_with("data: [DONE]\n\n"), "{body}");
        assert_eq!(mm.on_demand_waiters(), 0);
    }

    /// Expiry: 503 + `Retry-After` (non-streaming), the load still running; the message says
    /// the model is loading.
    #[tokio::test]
    async fn an_expired_wait_answers_503_retry_after() {
        let gate = Arc::new(LoadGate::default());
        let (app, mm) = app(
            Duration::from_millis(100),
            Some(Arc::clone(&gate)),
            false,
            false,
        )
        .await;
        let req = routes().swap_remove(0).2;
        let (status, headers, body) = send(&app, req).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
        assert_eq!(headers.get(header::RETRY_AFTER).unwrap(), "5");
        assert!(body.contains("loading"), "{body}");
        assert_eq!(mm.on_demand_waiters(), 0);
        assert!(state_of(&mm, TEXT).unwrap().contains("Loading"));
        gate.release();
    }

    /// The cap: with every waiter place taken a request (streaming included) gets the 503 +
    /// `Retry-After` at once, as an HTTP answer (no SSE response is started).
    #[tokio::test]
    async fn at_the_waiter_cap_requests_are_answered_at_once() {
        let gate = Arc::new(LoadGate::default());
        let (app, mm) = app(LONG, Some(Arc::clone(&gate)), false, false).await;
        let slots: Vec<_> = (0..MAX_ON_DEMAND_WAITERS)
            .map(|_| mm.try_on_demand_waiter_slot().unwrap())
            .collect();
        for (name, _, req) in routes() {
            let (status, headers, body) =
                tokio::time::timeout(Duration::from_secs(5), send(&app, req))
                    .await
                    .unwrap_or_else(|_| panic!("{name}: waited past the cap"));
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{name}: {body}");
            assert!(headers.get(header::RETRY_AFTER).is_some(), "{name}");
        }
        drop(slots);
        assert_eq!(mm.on_demand_waiters(), 0);
        gate.release();
    }

    /// A streaming client that goes away while waiting releases its place at once; the load
    /// is not cancelled.
    #[tokio::test]
    async fn a_streaming_client_that_leaves_releases_its_place() {
        let gate = Arc::new(LoadGate::default());
        let (app, mm) = app(LONG, Some(Arc::clone(&gate)), false, false).await;
        let req = routes()
            .into_iter()
            .find(|(n, _, _)| *n == "/v1/chat/completions (stream)")
            .unwrap()
            .2;
        let resp = app.clone().oneshot(req).await.unwrap();
        let mut body = resp.into_body();
        let _keep_alive = body.frame().await.unwrap().unwrap();
        assert_eq!(mm.on_demand_waiters(), 1);
        drop(body);
        assert_eq!(mm.on_demand_waiters(), 0, "released with the body");
        assert!(state_of(&mm, TEXT).unwrap().contains("Loading"));
        gate.release();
        until("Ready", || {
            state_of(&mm, TEXT).is_some_and(|s| s.contains("Ready"))
        })
        .await;
    }

    /// Ready, then evicted before the request got in: the request re-checks once, starts the
    /// load again and answers 503 + `Retry-After` — no second wait.
    #[tokio::test]
    async fn ready_then_evicted_rechecks_once_then_503() {
        let (_, mm) = app(LONG, None, false, false).await;
        let t0 = std::time::Instant::now();
        let out = resolve_waiting(
            &mm,
            TEXT,
            Err(ModelError::NotLoaded),
            WaitPlace::Take,
            || async {
                // The wait ended Ready; the model is evicted before this lookup.
                mm.evict_model(TEXT).await.unwrap();
                mm.get_text_context(TEXT).map(drop)
            },
        )
        .await;
        assert!(
            matches!(out, Err(Unready::OnDemand(OnDemandLoad::Loading))),
            "{out:?}"
        );
        assert!(t0.elapsed() < Duration::from_secs(5), "no second wait");
        until("loaded again", || {
            state_of(&mm, TEXT).is_some_and(|s| s.contains("Ready"))
        })
        .await;
        assert_eq!(mm.on_demand_waiters(), 0);
    }

    /// Both chat modes with a wait, the second request finding the model Ready (no wait).
    #[tokio::test]
    async fn chat_streaming_and_not_streaming_after_the_wait() {
        let (app, mm) = app(LONG, None, false, false).await;
        let mut reqs = routes().into_iter().filter(|(n, _, _)| n.contains("chat"));
        let (_, _, first) = reqs.next().unwrap();
        let (_, _, second) = reqs.next().unwrap();
        let (status, _, body) = send(&app, first).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains("chat.completion"), "{body}");
        let (status, headers, body) = send(&app, second).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            headers.get(header::CONTENT_TYPE).unwrap(),
            "text/event-stream"
        );
        assert!(body.contains("chat.completion.chunk"), "{body}");
        assert!(
            !body.starts_with(':'),
            "Ready: no wait, no keep-alive first: {body}"
        );
        assert_eq!(mm.on_demand_waiters(), 0);
    }

    /// Review R-6: every route refuses an on-demand model whose registered
    /// kind it cannot serve with its own wrong-kind 400, before any load or wait.
    #[tokio::test]
    async fn every_route_refuses_a_wrong_kind_before_loading() {
        let ids = [
            TEXT,
            "m-embed",
            "m-rerank",
            "m-decide",
            "m-stt",
            "m-tts",
            "m-imagegen",
        ];
        // Each model declared as a kind its routes do not serve.
        let wrong = |id: &str| if id == TEXT { "embedding" } else { "text_gen" };
        // `/tokenize` and `/detokenize` included: they never load, but keep this refusal.
        for (name, model, req) in all_routes() {
            let (app, mm, _) = app_with(&ids, LONG, None, false, false, Some(wrong)).await;
            let (status, headers, body) =
                tokio::time::timeout(Duration::from_secs(5), send(&app, req))
                    .await
                    .unwrap_or_else(|_| panic!("{name}: waited for a wrong kind"));
            assert_eq!(status, StatusCode::BAD_REQUEST, "{name}: {body}");
            assert!(headers.get(header::RETRY_AFTER).is_none(), "{name}");
            assert!(
                state_of(&mm, model).is_some_and(|s| s.contains("NotLoaded")),
                "{name}: no load started"
            );
            assert_eq!(mm.on_demand_waiters(), 0, "{name}");
        }
    }

    /// A factory whose registration-time detection finds no marker (the files were not there
    /// yet): `TextGen` by fallback, not certain. Its loads are the mock's, which detect the real
    /// kind from the name, as a real load detects it from the files that arrived.
    struct FallbackKindFactory(MockEngineFactory);

    impl crate::model_manager::lifecycle::EngineFactory for FallbackKindFactory {
        fn detect_kind(&self, _: &std::path::Path) -> crate::model_manager::ModelKind {
            crate::model_manager::ModelKind::TextGen
        }
        fn detect_kind_certain(
            &self,
            _: &std::path::Path,
        ) -> Option<crate::model_manager::ModelKind> {
            None
        }
        fn load(
            &self,
            model_path: &std::path::Path,
            device: &str,
            cache_size_gb: f64,
            max_num_seqs: usize,
        ) -> anyhow::Result<(
            crate::model_manager::EngineHandleKind,
            std::thread::JoinHandle<()>,
        )> {
            crate::model_manager::lifecycle::EngineFactory::load(
                &self.0,
                model_path,
                device,
                cache_size_gb,
                max_num_seqs,
            )
        }
    }

    /// Review N-1: the pre-load refusal needs a certain kind. An on-demand STT model registered
    /// with no `kind` and no marker (fallback `TextGen`) is loaded by its STT route and served;
    /// a marker kind (the mock's name segment) is still refused before any load.
    #[tokio::test]
    async fn a_fallback_kind_is_loaded_and_a_marker_kind_refused() {
        let stt = || {
            Request::builder()
                .method("POST")
                .uri("/v1/audio/transcriptions")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={BOUNDARY}"),
                )
                .body(Body::from(multipart(&[
                    ("model", None, b"m-stt"),
                    ("file", Some("a.wav"), b"RIFF0000WAVE"),
                ])))
                .unwrap()
        };
        // Fallback kind: loaded, then served by the kind its load found.
        let mut cfg = crate::model_manager::tests::make_config(&[("m-stt", 1.0)], 40.0);
        cfg.models.get_mut("m-stt").unwrap().policy.load = LoadPolicy::OnDemand;
        let gate = Arc::new(LoadGate::default());
        let factory = FallbackKindFactory(MockEngineFactory {
            load_gate: Some(Arc::clone(&gate)),
            ..Default::default()
        });
        let mm = Arc::new(ModelManager::new(cfg, Arc::new(factory)).await.unwrap());
        mm.set_on_demand_wait_for_test(LONG);
        let router = crate::create_router(AppState::new().with_model_manager(Arc::clone(&mm)));
        assert_eq!(
            mm.known_model_kind("m-stt"),
            Some(crate::model_manager::ModelKind::TextGen),
            "registered by fallback"
        );
        let call = tokio::spawn({
            let router = router.clone();
            async move { send(&router, stt()).await }
        });
        until("the load started", || {
            state_of(&mm, "m-stt").is_some_and(|s| s.contains("Loading"))
        })
        .await;
        gate.release();
        let (status, _, body) = call.await.unwrap();
        assert_eq!(status, StatusCode::OK, "{body}");

        // Marker kind (`stt` segment) on another route: refused, nothing loads.
        let (app, mm) = app(LONG, None, false, false).await;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({"model": "m-stt", "input": "hi"}).to_string(),
            ))
            .unwrap();
        let (status, _, body) = send(&app, req).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(
            state_of(&mm, "m-stt").is_some_and(|s| s.contains("NotLoaded")),
            "no load"
        );
        assert_eq!(mm.on_demand_waiters(), 0);
    }

    /// Review R-4: a streaming request that found no free place before its response started
    /// does not wait later, even when a place is free by then: the load is started and the
    /// immediate `503` + `Retry-After` answer comes back.
    #[tokio::test]
    async fn a_streaming_request_without_a_place_never_waits() {
        let gate = Arc::new(LoadGate::default());
        let (_, mm) = app(LONG, Some(Arc::clone(&gate)), false, false).await;
        let t0 = std::time::Instant::now();
        let out = resolve_waiting(
            &mm,
            TEXT,
            Err::<(), _>(ModelError::NotLoaded),
            WaitPlace::NoWait,
            || std::future::ready(mm.get_text_context(TEXT).map(drop)),
        )
        .await;
        assert!(
            matches!(out, Err(Unready::OnDemand(OnDemandLoad::Loading))),
            "{out:?}"
        );
        assert!(t0.elapsed() < Duration::from_secs(1));
        assert_eq!(mm.on_demand_waiters(), 0);
        until("the load started", || {
            state_of(&mm, TEXT).is_some_and(|s| s.contains("Loading"))
        })
        .await;
        gate.release();
    }

    /// Review R-5: a waited request's metrics zero point (TTFT, duration)
    /// is taken after the wait — on completions (text engine) and on a streaming chat (vision
    /// engine), whose engine also receives the request exactly once (its metrics are recorded
    /// once).
    #[tokio::test]
    async fn a_waited_requests_metrics_start_after_the_wait() {
        // The vision model's own id: the mock's record is process-wide.
        const VLM: &str = "m-vlm-f1bttft";
        // Completions, non-streaming.
        let gate = Arc::new(LoadGate::default());
        let (app, mm, requested_at) =
            app_with(&[TEXT], LONG, Some(Arc::clone(&gate)), false, false, None).await;
        let req = routes()
            .into_iter()
            .find(|(n, _, _)| *n == "/v1/completions")
            .unwrap()
            .2;
        let call = tokio::spawn({
            let app = app.clone();
            async move { send(&app, req).await }
        });
        until("waiting", || mm.on_demand_waiters() == 1).await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        let released = std::time::Instant::now();
        gate.release();
        let (status, _, body) = call.await.unwrap();
        assert_eq!(status, StatusCode::OK, "{body}");
        let seen = requested_at.lock().unwrap().clone();
        assert_eq!(seen.len(), 1);
        assert!(seen[0] >= released, "the zero point is after the wait");

        // Chat, streaming, on a vision model.
        let gate = Arc::new(LoadGate::default());
        let (app, mm, _) =
            app_with(&[VLM], LONG, Some(Arc::clone(&gate)), false, false, None).await;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "model": VLM, "stream": true,
                    "messages": [{"role": "user", "content": "hi"}]
                })
                .to_string(),
            ))
            .unwrap();
        let call = tokio::spawn({
            let app = app.clone();
            async move { send(&app, req).await }
        });
        until("waiting", || mm.on_demand_waiters() == 1).await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        let released = std::time::Instant::now();
        gate.release();
        let (status, _, body) = call.await.unwrap();
        assert_eq!(status, StatusCode::OK);
        assert!(body.starts_with(":\n\n"), "{body}");
        assert!(body.ends_with("data: [DONE]\n\n"), "{body}");
        let seen = crate::vlm_engine::mock_seen_started_at(VLM);
        assert_eq!(seen.len(), 1, "the engine got the request once");
        assert!(seen[0] >= released, "the zero point is after the wait");
    }
}
