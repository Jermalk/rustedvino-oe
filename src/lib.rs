// ============================================================
// src/lib.rs — `RustedVINO` library root
// ============================================================
// Declares all modules and assembles the axum Router.
// main.rs calls create_router(state) and binds the port.
// tests/ calls create_router(AppState::mock()) — no port, no GPU.
// ============================================================

pub mod admission;
pub mod app_state;
pub mod cache_manifest;
pub mod cb_engine;
#[cfg(unix)]
pub mod crash_handler;
pub mod device_inventory;
pub mod embed_engine;
mod handlers;
pub mod image_util;
pub mod in_flight;
pub mod log_ring;
pub mod metrics;
pub mod model_completeness;
pub mod model_manager;
pub mod npu_engine;
pub mod os_memory;
pub mod ov_cb;
pub mod ov_embed;
pub mod ov_image;
pub mod ov_pipeline;
pub mod ov_rerank;
pub mod ov_tts;
pub mod ov_vlm;
pub mod ov_whisper;
pub mod pipelines;
pub mod prompt_builder;
pub mod realtime_types;
pub mod rerank_engine;
pub mod startup;
pub mod streaming;
#[cfg(unix)]
pub mod supervisor;
pub mod tts_normalize;
pub mod vlm_engine;
pub mod voice_pin;

use std::sync::{Arc, OnceLock};

use app_state::AppState;
use arc_swap::ArcSwap;
use axum::{
    Router,
    extract::{DefaultBodyLimit, Request, State},
    http::{HeaderName, HeaderValue, StatusCode, header::AUTHORIZATION},
    middleware::Next,
    response::Response,
    routing::{delete, get, post},
};
use handlers::admin::{
    admin_add_model, admin_audit_config, admin_check_model, admin_deregister_model,
    admin_kill_realtime_session, admin_list_models, admin_load_model, admin_logs_tail,
    admin_models_completeness, admin_patch_model, admin_realtime_session_by_id,
    admin_realtime_sessions, admin_reload_config, admin_reload_keys, admin_resize_model,
    admin_set_preload, admin_unload_model, admin_voice_pin, admin_watchdog, health,
    health_generate, metrics_handler, models_list,
};
use handlers::chat::chat_completions;
use handlers::completions::completions;
use handlers::embeddings::embeddings;
use handlers::error::openai_error;
use handlers::media::{image_edits, image_generations, speech, transcriptions};
use handlers::realtime::realtime_handler;
use handlers::reranking::rerank;
use handlers::tokenize::{detokenize, tokenize};
use serde::Deserialize;
use subtle::ConstantTimeEq;
use tower_http::cors::{Any, CorsLayer};

/// 50 MB — covers full-res smartphone photos (base64 adds ~33 % overhead,
/// so this accepts ~37 MB of raw image data). Raised from axum's 2 MB default
/// to support VLM image payloads sent by clients like `AnythingLLM`.
const MAX_REQUEST_BODY: usize = 50 * 1024 * 1024;

/// Assembles the axum [`Router`] for the full `RustedVINO` API with a
/// **permissive** CORS policy (`Access-Control-Allow-Origin: *`).
///
/// This is the convenience constructor used by integration tests, which call
/// `rustedvino::create_router(AppState::mock()).oneshot(request)` without
/// binding a port or loading any GPU resources. Production code uses
/// [`create_router_with_cors`] to honour the configured origin allowlist.
///
/// Routes added per phase:
/// - Phase 0: `GET /health`, `GET /v1/models`
/// - Phase 1: `POST /v1/chat/completions`
/// - Phase 2: `POST /v1/admin/models/{id}/load`, `DELETE /v1/admin/models/{id}`
/// - Phase 3.3: `GET /metrics` (Prometheus)
/// - Phase 3.7: `POST /v1/completions`, `POST /tokenize`, `POST /detokenize`
/// - Phase 3.8: `GET /health_generate` (deep GPU readiness probe)
/// - Phase 5: STT / TTS / image routes
pub fn create_router(state: AppState) -> Router {
    create_router_with_cors(state, &["*".to_owned()], &[])
}

/// Assembles the axum [`Router`] with an explicit CORS origin allowlist and a
/// single (inference) Bearer allowlist — **no separate admin scope**.
///
/// This is a thin wrapper over [`create_router_with_auth`] passing an empty
/// `admin_api_keys`, so the `/v1/admin/*` routes are gated by the same
/// `api_keys` list as everything else (or fully open when it is empty). It is
/// the constructor most integration tests use. Production code calls
/// [`create_router_with_auth`] to honour `Config::admin_api_keys` (T4.3).
pub fn create_router_with_cors(
    state: AppState,
    cors_origins: &[String],
    api_keys: &[String],
) -> Router {
    create_router_with_auth(state, cors_origins, api_keys, &[])
}

/// Assembles the axum [`Router`] with CORS, an inference Bearer allowlist, and a
/// distinct **admin** Bearer allowlist (T4.3 scope separation).
///
/// `cors_origins` comes from `Config::cors_allowed_origins`. When any entry is
/// `"*"` the policy is fully permissive (`allow_origin(Any)`); otherwise only
/// the listed origins are echoed back. Every route also gets an
/// `x-request-id` response header (see [`attach_request_id`]).
///
/// `api_keys` comes from `Config::api_keys`. When **empty**, inference is open
/// (no auth — trusted alpha). When **non-empty**, every request except the infra
/// probes (`/health`, `/metrics`) must present a matching
/// `Authorization: Bearer <key>` or it is rejected `401`.
///
/// `admin_api_keys` comes from `Config::admin_api_keys`. When **non-empty**, the
/// `/v1/admin/*` routes require a key from *this* list; a valid inference key
/// that is not an admin key is rejected `403` (`code:insufficient_scope`). When
/// **empty**, the admin routes fall back to the `api_keys` gate. See
/// [`require_bearer_auth`].
///
/// Thin wrapper over [`create_router_with_extension`] passing `extension: None`
/// — see that function's doc for the extension-point mechanism.
pub fn create_router_with_auth(
    state: AppState,
    cors_origins: &[String],
    api_keys: &[String],
    admin_api_keys: &[String],
) -> Router {
    create_router_with_extension(state, cors_origins, api_keys, admin_api_keys, None)
}

/// [`create_router_with_auth`], plus an optional externally-built `Router` to
/// merge onto the finished result.
///
/// This is the extension point a separate crate (e.g. a closed-source
/// module that isn't part of this Apache 2.0 tree — see
/// the project's internal engineering log) uses to attach its own
/// routes onto a standard `rustedvino` server without forking this file.
///
/// `extension` must be a **fully-built, fully-stated** `Router` (axum's
/// `Router` defaults its state parameter to `()`, i.e. `.with_state(...)`
/// has already been called on it with whatever state *that* router needs —
/// not necessarily [`AppState`]). A finished `Router<()>` merges into a
/// router of any state type, which is what makes this composable without
/// this crate knowing anything about the extension's own state.
///
/// Deliberately merged in **after** every layer below (auth, CORS,
/// request-id, server headers) is applied to the base routes — the merge
/// happens on the fully-layered result, so `extension`'s routes are **not**
/// wrapped by [`require_bearer_auth`] or any other base middleware. An
/// extension implementing its own auth story (for example RTCC's
/// browser-safe WS handshake, which a browser's `WebSocket` API cannot
/// satisfy via the `Authorization` header this crate's middleware checks)
/// needs exactly this: full control over its own routes, composed with —
/// never gated by — the base server's inference-key auth.
pub fn create_router_with_extension(
    state: AppState,
    cors_origins: &[String],
    api_keys: &[String],
    admin_api_keys: &[String],
    extension: Option<Router>,
) -> Router {
    let base = Router::new()
        // ── Core API ───────────────────────────────────────────────────
        .route("/health", get(health))
        .route("/health_generate", get(health_generate))
        .route("/metrics", get(metrics_handler))
        .route("/v1/models", get(models_list))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/rerank", post(rerank))
        // ── Media endpoints (Phase 5) ─────────────────────────────────
        // 5.1a: /v1/audio/transcriptions (Whisper STT) — live
        // 5.2a: /v1/audio/speech (SpeechT5 TTS) — live
        // 5.3a: /v1/images/generations (Text2Image) — live
        .route("/v1/audio/transcriptions", post(transcriptions))
        .route("/v1/audio/speech", post(speech))
        .route("/v1/images/generations", post(image_generations))
        .route("/v1/images/edits", post(image_edits))
        // ── Tokenizer helpers (Phase 3.7, non-OpenAI) ─────────────────
        // Expose the model's tokenizer for agent frameworks: token counts
        // before sending, and ids→text without a generation round-trip.
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/v1/realtime", get(realtime_handler))
        // ── Admin model management (Phase 2) ──────────────────────────
        // These routes expose dynamic model loading/eviction over HTTP.
        // They are not part of the OpenAI API spec — prefix with
        // /v1/admin/ to keep them namespaced and easy to firewall.
        .route("/v1/admin/models", get(admin_list_models))
        .route(
            "/v1/admin/models/completeness",
            get(admin_models_completeness),
        )
        .route("/v1/admin/models/add", post(admin_add_model))
        .route("/v1/admin/models/{model_id}/load", post(admin_load_model))
        .route(
            "/v1/admin/models/{model_id}/resize",
            post(admin_resize_model),
        )
        .route("/v1/admin/models/{model_id}/check", post(admin_check_model))
        .route(
            "/v1/admin/models/{model_id}",
            delete(admin_unload_model).patch(admin_patch_model),
        )
        .route(
            "/v1/admin/models/{model_id}/register",
            delete(admin_deregister_model),
        )
        .route("/v1/admin/config/reload", post(admin_reload_config))
        .route("/v1/admin/config/preload", post(admin_set_preload))
        .route("/v1/admin/keys/reload", post(admin_reload_keys))
        .route("/v1/admin/config/audit", get(admin_audit_config))
        .route("/v1/admin/logs/tail", get(admin_logs_tail))
        .route("/v1/admin/health/watchdog", get(admin_watchdog))
        .route("/v1/admin/voice-pin", get(admin_voice_pin))
        .route("/v1/admin/realtime/sessions", get(admin_realtime_sessions))
        .route(
            "/v1/admin/realtime/sessions/{id}",
            get(admin_realtime_session_by_id).delete(admin_kill_realtime_session),
        )
        // ── Middleware ─────────────────────────────────────────────────
        // Applied to every route above. axum runs layers outermost-first,
        // and the OUTERMOST layer is the one added LAST. We add auth first
        // (innermost), then request-id, then CORS (outermost), giving the
        // request order: CORS → request-id → auth → handler. request-id
        // therefore wraps auth, so even a 401 from `require_bearer_auth`
        // carries an `x-request-id`. CORS is outermost so preflight
        // `OPTIONS` is answered by the CorsLayer before auth ever sees it.
        .layer(axum::middleware::from_fn_with_state(
            {
                // Seed (or reseed) `state.auth` from the constructor's own
                // params rather than trusting it was already populated —
                // covers both the production path (bootstrap already seeded
                // it identically, so this is a harmless no-op re-store) and
                // the ~10 test call sites that build a fresh `AppState` and
                // pass explicit keys straight into this constructor. Reusing
                // the *same* `ArcSwap` instance (not a fresh `Arc<AuthConfig>`
                // each call) is what lets `ModelManager::reload_keys_file` —
                // which holds its own clone of this exact `Arc` — rotate
                // keys the middleware observes with no router rebuild.
                state.auth.store(Arc::new(AuthConfig {
                    api_keys: api_keys.to_vec(),
                    admin_api_keys: admin_api_keys.to_vec(),
                }));
                AuthMiddlewareState {
                    keys: Arc::clone(&state.auth),
                    admin_locked: state.admin_locked,
                }
            },
            require_bearer_auth,
        ))
        .layer(axum::middleware::from_fn(attach_request_id))
        .layer(axum::middleware::from_fn(attach_ruvi_host))
        .layer(axum::middleware::from_fn(attach_server_headers))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY))
        .layer(build_cors_layer(cors_origins))
        // CRASH COURSE — with_state:
        //   Attaches AppState to the router. axum clones it once per
        //   request and injects it wherever State<AppState> appears in
        //   handler arguments.
        .with_state(state);

    match extension {
        Some(extension) => base.merge(extension),
        None => base,
    }
}

/// Builds the CORS layer from a configured origin list.
///
/// `["*"]` (the default) → allow any origin. Any other list → echo only those
/// origins. Methods and headers are permitted broadly; credentials are not
/// enabled (which would be incompatible with a wildcard origin anyway).
fn build_cors_layer(origins: &[String]) -> CorsLayer {
    let layer = CorsLayer::new().allow_methods(Any).allow_headers(Any);
    if origins.iter().any(|o| o == "*") {
        layer.allow_origin(Any)
    } else {
        let allowed: Vec<HeaderValue> = origins
            .iter()
            .filter_map(|o| HeaderValue::from_str(o).ok())
            .collect();
        layer.allow_origin(allowed)
    }
}

/// Middleware: attach an `x-ruvi-host` response header carrying this box's
/// hostname — every response on every pipeline (chat, VLM, STT, TTS,
/// embeddings, rerank, images), not just the image-gen `generation_metadata`
/// Tier 1 added. Lets fleet tooling attribute a response to the physical
/// machine that served it without an extra admin round-trip.
///
/// `os_memory::host_name()` reads `/proc/sys/kernel/hostname` on Linux — cheap,
/// but static for the process lifetime, so it's read once and cached rather
/// than on every request. Omitted entirely (no header at all) when the OS
/// doesn't report a hostname, rather than sending a placeholder — this
/// previously sent the literal string `"ruvi-host"` unconditionally, which
/// was never a real value.
async fn attach_ruvi_host(req: Request, next: Next) -> Response {
    static HOST: OnceLock<Option<String>> = OnceLock::new();
    let host = HOST.get_or_init(crate::os_memory::host_name);

    let mut response = next.run(req).await;
    if let Some(host) = host
        && let Ok(value) = HeaderValue::from_str(host)
    {
        response
            .headers_mut()
            .insert(HeaderName::from_static("x-ruvi-host"), value);
    }
    response
}

/// API-contract version (request/response schema), independent of the
/// server's own build/release number below. Fixed until the server has a
/// public distribution to version against.
const API_VERSION: &str = "1.0.0";

/// Middleware: attach `x-server` / `x-api-version` to every response, and
/// `x-server-version` (the build's `CARGO_PKG_VERSION`, same source
/// `/health` uses — [`crate::handlers::admin`]) only on a successful
/// (`200`) response to an admin route (T4.3's [`ADMIN_PATH_PREFIX`]).
///
/// The build number is real fleet-ops value on an authenticated admin call
/// ("which release is this box on") but unnecessary fingerprinting surface
/// everywhere else — OWASP flags version disclosure as reconnaissance
/// fodder, and unlike `x-server`/`x-api-version` it can't be pinned to like
/// an API-contract version, so there is no client-facing reason to expose
/// it broadly.
async fn attach_server_headers(req: Request, next: Next) -> Response {
    let is_admin_route = req.uri().path().starts_with(ADMIN_PATH_PREFIX);
    let mut response = next.run(req).await;
    let is_authorized_ok = response.status() == StatusCode::OK;
    let headers = response.headers_mut();
    headers.insert(
        HeaderName::from_static("x-server"),
        HeaderValue::from_static("RustedVINO"),
    );
    headers.insert(
        HeaderName::from_static("x-api-version"),
        HeaderValue::from_static(API_VERSION),
    );
    if is_admin_route && is_authorized_ok {
        headers.insert(
            HeaderName::from_static("x-server-version"),
            HeaderValue::from_static(env!("CARGO_PKG_VERSION")),
        );
    }
    response
}

async fn attach_request_id(req: Request, next: Next) -> Response {
    let request_id = req
        .headers()
        .get("x-client-request-id")
        .and_then(|v| v.to_str().ok())
        .map_or_else(|| uuid::Uuid::new_v4().to_string(), str::to_owned);

    let mut response = next.run(req).await;
    if let Ok(value) = HeaderValue::from_str(&request_id) {
        response
            .headers_mut()
            .insert(HeaderName::from_static("x-request-id"), value);
    }
    response
}

/// Paths that bypass Bearer auth even when `api_keys` is non-empty.
///
/// Only `/health` — the cheap, content-free liveness probe orchestrators
/// (k8s liveness/readiness) poll without credentials. Keeping it open is the
/// standard posture and avoids wedging health checks behind the API key.
/// Everything else requires a key once auth is enabled.
///
/// `/health_generate` is **deliberately excluded** despite being a probe: it
/// runs a real 1-token generation that consumes a GPU inference slot, so an
/// uncredentialed caller could drive GPU load / exhaust admission slots. It
/// requires a key like any other generation endpoint.
///
/// `/metrics` is **deliberately excluded** too (was exempt here until an
/// internet-facing deployment surfaced the risk): it leaks the loaded model
/// catalog, VRAM capacity/usage, and per-model request-rate data — real
/// operational intelligence, not a content-free ping like `/health`. Combined
/// with a permissive CORS policy (`cors_allowed_origins: ["*"]`, this
/// project's own default), an unauthenticated `/metrics` is a drive-by
/// information-disclosure vector: any website's JS can cross-origin `fetch`
/// and read it from a visitor's browser with no credentials needed, not just
/// "someone has to know your IP and curl it directly." `/metrics` is instead
/// folded into [`ADMIN_PATH_PREFIX`]'s gate below (same admin-key
/// requirement, same admin-locked 503, same superset-credential rule) even
/// though its URL isn't under `/v1/admin/` — moving the path would break
/// every existing Prometheus scrape config for no benefit.
const AUTH_EXEMPT_PATHS: [&str; 1] = ["/health"];

/// Path prefix for the admin model-management routes (load / unload). Requests
/// here are gated by `admin_api_keys` when that list is configured (T4.3).
const ADMIN_PATH_PREFIX: &str = "/v1/admin/";

/// `/metrics` is gated exactly like an admin route (see [`AUTH_EXEMPT_PATHS`]'s
/// doc comment on `/metrics` for why) despite not sharing [`ADMIN_PATH_PREFIX`].
const METRICS_PATH: &str = "/metrics";

/// The two Bearer allowlists injected into [`require_bearer_auth`].
///
/// `api_keys` gates inference and general routes; `admin_api_keys` adds a
/// distinct scope for the `/v1/admin/*` routes (T4.3). See the middleware for
/// the precedence rules.
///
/// Held behind `Arc<ArcSwap<AuthConfig>>` (on [`app_state::AppState::auth`],
/// shared with [`model_manager::ModelManager`]) rather than a bare `Arc`, so
/// `POST /v1/admin/keys/reload` can rotate keys with no server restart — see
/// `ModelManager::reload_keys_file`. Deliberately does not derive `Debug`:
/// there is then no accidental `{:?}` path that could print key material.
///
/// Also the on-disk shape of the keys file itself (`model_manager::config::
/// resolve_keys_file_path`) — deserialized directly, no separate DTO, so the
/// no-`Debug` property covers the file-parsing path too. `deny_unknown_fields`
/// is a deliberate, narrow exception to this project's usual
/// warn-on-unknown-config-key convention: a typo'd key in a credential file
/// (`admin_keys` for `admin_api_keys`) must be loud, not silently "no admin
/// keys configured".
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct AuthConfig {
    #[serde(default)]
    pub api_keys: Vec<String>,
    #[serde(default)]
    pub admin_api_keys: Vec<String>,
}

impl AuthConfig {
    /// Read and parse the keys file at `path` — the on-disk shape is exactly
    /// this struct's JSON, `deny_unknown_fields` included.
    ///
    /// Also warns (does not fail — see this struct's doc comment) when the
    /// file is readable by group or other on Unix; a no-op check on other
    /// platforms.
    ///
    /// # Errors
    /// The file doesn't exist, can't be read, or isn't valid JSON matching
    /// this shape. Callers that want "missing file means open, not an
    /// error" (`startup::bootstrap`'s boot-time posture) should check
    /// `path.exists()` themselves *before* calling this — deliberately not
    /// handled inside this function, so every caller's missing-file policy
    /// is visible at its own call site rather than buried in a shared
    /// error-kind check.
    pub fn load_from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading keys file {}: {e}", path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(meta) = std::fs::metadata(path) {
                let mode = meta.permissions().mode() & 0o777;
                if mode & 0o077 != 0 {
                    tracing::warn!(
                        path = %path.display(),
                        mode = format!("{mode:o}"),
                        "keys file is readable by group/other — recommend chmod 600"
                    );
                }
            }
        }
        let config: Self = serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("parsing keys file {}: {e}", path.display()))?;
        warn_on_key_suffix_collisions(&config);
        Ok(config)
    }
}

/// Warns (never fails a load — same "loud but non-fatal" posture as this
/// file's readable-by-group/other check above) when two or more configured
/// keys share the same [`key_suffix`]. `rustedvino_key_usage_total` would
/// then silently merge two different keys' activity under one series, which
/// reads as "one tester is very active" instead of "two testers happen to
/// share a suffix" — a genuinely wrong number, not just a missing one.
///
/// Both `startup::bootstrap` and `ModelManager::reload_keys_file` funnel
/// through [`AuthConfig::load_from_file`], so hooking the check in here
/// (rather than duplicating it at each call site) covers every path a keys
/// file can enter the live server through, including a hot rotation via
/// `POST /v1/admin/keys/reload`.
///
/// Only the colliding suffix and how many keys share it are logged — never
/// which keys, since that would defeat the entire point of a metric that
/// only ever handles a 6-character fragment.
fn warn_on_key_suffix_collisions(config: &AuthConfig) {
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for key in config.api_keys.iter().chain(config.admin_api_keys.iter()) {
        if let Some(suffix) = key_suffix(key) {
            *counts.entry(suffix).or_insert(0) += 1;
        }
    }
    for (suffix, count) in counts {
        if count > 1 {
            tracing::warn!(
                key_suffix = suffix,
                colliding_key_count = count,
                "keys file: multiple configured keys share the same last-6-character suffix — \
                 rustedvino_key_usage_total will merge their activity under one series; \
                 regenerate one of them if per-key attribution matters"
            );
        }
    }
}

/// State for [`require_bearer_auth`]'s layer: the hot-swappable keys plus a
/// fixed-at-boot flag for the "no keys file exists at all" admin lockdown.
/// Bundled into one `Clone` state (rather than a second `from_fn_with_state`
/// layer) so the two are always read together in the same request — see
/// [`admin_locked`](Self::admin_locked)'s doc comment for why they can't be
/// two independent booleans checked separately.
#[derive(Clone)]
struct AuthMiddlewareState {
    keys: Arc<ArcSwap<AuthConfig>>,
    /// Mirrors [`app_state::AppState::admin_locked`] — copied in once at
    /// router-build time, never itself hot-reloaded (there is deliberately
    /// no code path that can flip it after boot; see that field's doc
    /// comment for why).
    admin_locked: bool,
}

/// Middleware: enforce `Authorization: Bearer <key>` with optional admin scope.
///
/// [`AuthMiddlewareState`] is injected as the layer's state. The infra
/// probes in [`AUTH_EXEMPT_PATHS`] always pass through uncredentialed. For
/// everything else:
///
/// **Admin routes (`/v1/admin/*`, plus [`METRICS_PATH`] — see its doc
/// comment for why `/metrics` gets the admin gate despite the different
/// URL) when [`AuthMiddlewareState::admin_locked`] is `true`** (no keys file
/// existed at boot): **always `503`** — no bearer token, valid or not, can
/// ever satisfy an admin route in this state. This
/// check runs before anything else admin-related, deliberately: the whole
/// point is that a server that boots with no keys file configured yet must
/// never let `/v1/admin/*` fall through to the (usually wide open)
/// inference gate just because `admin_api_keys` also happens to be empty —
/// see `app_state::AppState::admin_locked`'s doc comment for the reasoning
/// and the the project's internal engineering log entry for the full design discussion.
///
/// **Admin routes with `admin_api_keys` configured (T4.3, `admin_locked`
/// false):**
/// - a key in `admin_api_keys` → allowed;
/// - a key valid for inference but NOT an admin key → `403`
///   (`code:insufficient_scope`) — authenticated, but lacks admin scope;
/// - an unknown key → `401`; a missing key → `401`.
///
/// **All other cases** (inference/general routes, and admin routes when
/// `admin_locked` is false and `admin_api_keys` is empty — an operator's
/// deliberate "share the inference scope" choice, made by creating a keys
/// file with `admin_api_keys` explicitly empty rather than not creating one
/// at all — they then fall back to the inference gate):
/// - `api_keys` empty → open, request passes through;
/// - else a key in `api_keys` *or* `admin_api_keys` (admin keys are a superset
///   credential) → allowed; unknown → `401`; missing → `401`.
///
/// The scheme match is case-insensitive (`Bearer`/`bearer`); the token is
/// trimmed of surrounding whitespace. Per-key comparison is constant-time
/// ([`key_matches`]) — a plain `==` short-circuits on the first mismatched
/// byte, letting response timing leak how many leading characters of a
/// guessed key were correct. Which key in the list matched (if any) is not
/// itself hidden — only each individual comparison is.
///
/// Every branch that lets a request through with a matched key also records
/// it via [`key_suffix`]/`crate::metrics::record_key_usage` — per-key
/// activity attribution, labeled only by the last 6 characters. Rejected
/// requests (401/403) and the fully-open case (no keys configured) record
/// nothing — there is no key to attribute a rejection to, and recording
/// arbitrary presented tokens (valid or not) would turn this metric's
/// currently-bounded cardinality into an attacker-controlled one.
async fn require_bearer_auth(
    State(auth): State<AuthMiddlewareState>,
    req: Request,
    next: Next,
) -> Response {
    // Infra probes are always reachable uncredentialed.
    if AUTH_EXEMPT_PATHS.contains(&req.uri().path()) {
        return next.run(req).await;
    }

    let is_admin_route =
        req.uri().path().starts_with(ADMIN_PATH_PREFIX) || req.uri().path() == METRICS_PATH;
    if is_admin_route && auth.admin_locked {
        return openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "This endpoint requires admin authentication, and no keys file has ever been \
             configured for this server. Create the keys file (see CONFIG.md's keys_file \
             entry) and restart to enable it.",
            "server_error",
            Some("admin_not_configured"),
        );
    }

    // One snapshot for the whole check — both the admin and inference
    // branches below read this same `Guard`, so a concurrent
    // `ModelManager::reload_keys_file` swap can never be observed as a torn
    // read (new `admin_api_keys` paired with old `api_keys` or vice versa).
    let keys = auth.keys.load();
    let provided = bearer_token(&req);

    // T4.3: admin routes with a distinct admin scope configured.
    if is_admin_route && !keys.admin_api_keys.is_empty() {
        return match provided {
            Some(key) if keys.admin_api_keys.iter().any(|k| key_matches(k, key)) => {
                if let Some(suffix) = key_suffix(key) {
                    crate::metrics::record_key_usage(suffix);
                }
                next.run(req).await
            }
            // A valid inference key reaching admin → authenticated but unauthorized.
            Some(key) if keys.api_keys.iter().any(|k| key_matches(k, key)) => openai_error(
                StatusCode::FORBIDDEN,
                "This API key does not have admin scope. Admin routes require an admin key.",
                "invalid_request_error",
                Some("insufficient_scope"),
            ),
            Some(_) => invalid_api_key(),
            None => missing_api_key(),
        };
    }

    // Inference / general gate. Admin keys also satisfy it (superset credential).
    if keys.api_keys.is_empty() {
        return next.run(req).await;
    }
    match provided {
        Some(key)
            if keys.api_keys.iter().any(|k| key_matches(k, key))
                || keys.admin_api_keys.iter().any(|k| key_matches(k, key)) =>
        {
            if let Some(suffix) = key_suffix(key) {
                crate::metrics::record_key_usage(suffix);
            }
            next.run(req).await
        }
        Some(_) => invalid_api_key(),
        None => missing_api_key(),
    }
}

/// Constant-time equality for one configured key against one presented
/// token — prevents a byte-at-a-time timing attack on the bearer-key check.
/// Different-length inputs short-circuit to `false` without a constant-time
/// byte comparison (length alone isn't the secret; the key's content is).
fn key_matches(configured: &str, presented: &str) -> bool {
    configured.as_bytes().ct_eq(presented.as_bytes()).into()
}

/// Extracts the last 6 characters of a presented Bearer token, for the
/// low-cardinality, non-identifying key-suffix label used by
/// `rustedvino_key_usage_total`/`rustedvino_key_last_seen_timestamp_seconds`
/// (`crate::metrics::record_key_usage`) — this is the *only* place in the
/// codebase allowed to turn a full key into a shorter fragment, and the
/// fragment it produces is never logged, stored, or passed anywhere as more
/// than these 6 characters.
///
/// Returns `None` for a key under 12 characters: for a short key, "last 6
/// characters" could be most or all of the actual secret, so the safe
/// behavior is to skip attribution entirely rather than let an activity
/// metric double as a partial key leak — `/metrics` is admin-gated, but
/// admin-gated data still routinely ends up pasted into issues and
/// screenshots.
///
/// Uses `char_indices` rather than a raw byte slice: `Authorization` header
/// values are guaranteed valid UTF-8 (ASCII, in practice) by
/// `HeaderValue::to_str`, so a byte-index slice would be safe today, but
/// this makes that guarantee unconditional rather than borrowed from a
/// caller that could change.
fn key_suffix(key: &str) -> Option<&str> {
    if key.chars().count() < 12 {
        return None;
    }
    let split_at = key.char_indices().rev().nth(5).map_or(0, |(i, _)| i);
    Some(&key[split_at..])
}

/// Parse `Authorization: Bearer <token>` (scheme case-insensitive, token
/// trimmed). Returns `None` when the header is absent, non-ASCII, or not a
/// Bearer scheme.
fn bearer_token(req: &Request) -> Option<&str> {
    req.headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split_once(' '))
        .and_then(|(scheme, token)| scheme.eq_ignore_ascii_case("Bearer").then(|| token.trim()))
}

/// `401` for a present-but-unrecognised key.
fn invalid_api_key() -> Response {
    openai_error(
        StatusCode::UNAUTHORIZED,
        "Incorrect API key provided.",
        "invalid_request_error",
        Some("invalid_api_key"),
    )
}

/// `401` for a missing `Authorization` header.
fn missing_api_key() -> Response {
    openai_error(
        StatusCode::UNAUTHORIZED,
        "Missing API key. Provide it via the 'Authorization: Bearer <key>' header.",
        "invalid_request_error",
        Some("invalid_api_key"),
    )
}

#[cfg(test)]
mod key_suffix_tests {
    use super::*;

    /// A key exactly at the 12-character floor is kept — the boundary is
    /// "under 12," not "under or equal to."
    #[test]
    fn twelve_chars_is_kept() {
        assert_eq!(key_suffix("abcdefghijkl"), Some("ghijkl"));
    }

    /// One character under the floor is skipped entirely, not truncated to
    /// whatever's left — a short key's "last 6 chars" would be most/all of
    /// the real secret.
    #[test]
    fn eleven_chars_is_skipped() {
        assert_eq!(key_suffix("abcdefghijk"), None);
    }

    #[test]
    fn empty_string_is_skipped() {
        assert_eq!(key_suffix(""), None);
    }

    #[test]
    fn three_chars_is_skipped() {
        assert_eq!(key_suffix("abc"), None);
    }

    /// A real-shaped key returns exactly its last 6 characters. Deliberately
    /// NOT a real key value (even a since-rotated one) — this is a test
    /// fixture that ends up in git history, and "it's just a test string"
    /// is exactly how a real secret accidentally becomes a permanent one.
    #[test]
    fn realistic_key_returns_last_six() {
        assert_eq!(
            key_suffix("sk-infer-examplehost-0000000000000000000000000000000000abcdef"),
            Some("abcdef")
        );
    }

    /// Two keys that only differ before their last 6 characters must not be
    /// distinguishable by this function alone — that's exactly the
    /// collision case `warn_on_key_suffix_collisions` exists to flag, not
    /// something `key_suffix` itself is expected to prevent.
    #[test]
    fn colliding_keys_produce_the_same_suffix() {
        assert_eq!(
            key_suffix("sk-alpha-000000abc123"),
            key_suffix("sk-beta-111111abc123")
        );
    }

    /// `warn_on_key_suffix_collisions` must not panic or do anything
    /// observable-in-a-test-harness-sense for the non-colliding, empty, or
    /// short-key cases — this just exercises those paths for a crash, since
    /// its actual output is a `tracing::warn!` this test doesn't capture.
    #[test]
    fn collision_warning_does_not_panic_on_edge_cases() {
        warn_on_key_suffix_collisions(&AuthConfig::default());
        warn_on_key_suffix_collisions(&AuthConfig {
            api_keys: vec!["short".to_owned()],
            admin_api_keys: vec![],
        });
        warn_on_key_suffix_collisions(&AuthConfig {
            api_keys: vec!["sk-alpha-000000abc123".to_owned()],
            admin_api_keys: vec!["sk-beta-111111abc123".to_owned()],
        });
    }
}
