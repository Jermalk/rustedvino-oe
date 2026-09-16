// ============================================================
// src/app_state.rs — shared application state
// ============================================================
// AppState is cloned cheaply into every request handler via
// axum's State extractor. Each field must be O(1) to clone:
// either a small Copy value or an `Arc<T>`.
//
// CRASH COURSE — Arc<T>:
//   "Atomically Reference-Counted" shared pointer. Multiple
//   owners; data lives until the last Arc drops. Clone is O(1)
//   — it increments an atomic counter, never copies T itself.
//   Python objects work this way under the hood (CPython refcount);
//   Rust just makes you explicit about it.
//
// Phase 2: AppState holds an optional Arc<ModelManager>.
//   When None  → test/mock mode (no GPU, mock token stream).
//   When Some  → production mode (get_handle routes to the right engine).
//
// Note: `engine: Option<EngineHandle>` from Phase 1 is gone.
//   The model manager owns all engine handles now.
// ============================================================

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use arc_swap::ArcSwap;
use metrics_exporter_prometheus::PrometheusHandle;

use crate::AuthConfig;
use crate::admission::DeviceBudgets;
use crate::log_ring::LogRingBuffer;
use crate::model_manager::ModelManager;
use crate::realtime_types::RealtimeSessionRegistry;
use crate::voice_pin::VoicePinManager;

/// Shared application state injected by axum into every handler.
///
/// Cloning is O(1). Constructed once in `main`, cloned per request.
#[derive(Clone)]
pub struct AppState {
    /// The model lifecycle manager, or `None` in mock/test mode.
    ///
    /// When `None`, chat completions fall back to the mock token generator
    /// (fast, deterministic, no GPU). All admin model routes return 503.
    ///
    /// When `Some`, `get_handle(model_id)` routes each chat request to
    /// the appropriate continuous-batching engine.
    pub model_manager: Option<Arc<ModelManager>>,

    /// Handle to the Prometheus recorder, or `None` in mock/test mode.
    ///
    /// `render()` produces the `/metrics` text. `None` means metrics are not
    /// installed (tests, or a build that never called `metrics::install()`) —
    /// the `/metrics` route then returns 503. `Option::default()` is `None`,
    /// so `#[derive(Default)]` still holds despite `PrometheusHandle: !Default`.
    pub metrics_handle: Option<PrometheusHandle>,

    /// Shared voice flow pin for realtime sessions.
    ///
    /// Records the first confirmed {stt, llm, tts} model set so subsequent
    /// blank-config clients can reuse it without rediscovery. Always present
    /// (even in mock/test mode — just empty). See `voice_pin.rs`.
    pub voice_pin: Arc<VoicePinManager>,

    /// Live realtime session registry for the admin API.
    ///
    /// Keyed by session UUID; each entry is an `Arc<RwLock<SessionSnapshot>>`
    /// updated as the session progresses. Inserted on WS connect, removed on close.
    pub realtime_sessions: RealtimeSessionRegistry,

    /// Cross-pipeline device admission ceiling (`dev/plans/cross-pipeline-
    /// admission-middleware.md` step 2). Always present (even mock/test
    /// mode) as a passthrough-by-default instance, mirroring `voice_pin`'s
    /// "always present, just empty" convention — see `admission.rs`.
    pub device_budgets: Arc<DeviceBudgets>,

    /// Recent-log tail for the admin API (`GET /v1/admin/logs/tail`).
    ///
    /// Always present, mirroring `voice_pin`'s "always present, just empty"
    /// convention — in mock/test mode it simply never receives any lines
    /// (nothing installs it as the tracing writer). In production, `main`
    /// builds one instance, installs it as the `tracing_subscriber` writer,
    /// and attaches the *same* instance here via `with_log_buffer` so the
    /// admin endpoint reads what the subscriber is writing.
    pub log_buffer: LogRingBuffer,

    /// Live, hot-swappable Bearer-key allowlists — the state the auth
    /// middleware (`require_bearer_auth`) reads on every request.
    ///
    /// Always present (even mock/test mode — starts as two empty lists, the
    /// open/no-auth posture). In production, `bootstrap` builds one instance
    /// and hands the *same* `Arc` to both this field and
    /// `ModelManager::with_auth`, so `POST /v1/admin/config/reload` writing
    /// through the `ModelManager`-held clone is immediately visible to the
    /// middleware's clone — no router rebuild, no dropped connections. See
    /// `crate::create_router_with_extension`, which (re)seeds this field from
    /// its own `api_keys`/`admin_api_keys` params on every router build (the
    /// test call sites' route into this, since they never go through
    /// `bootstrap`).
    pub auth: Arc<ArcSwap<AuthConfig>>,

    /// `true` only when the keys file did not exist at boot (`startup::
    /// load_keys_file_for_boot`'s "missing → open" branch) — admin routes
    /// (`/v1/admin/*`) are then **permanently locked** for this process's
    /// lifetime: no bearer token can ever satisfy them, regardless of what
    /// `auth` later gets hot-reloaded to (there is nothing to reload from —
    /// `ModelManager::reload_keys_file` errors on a still-missing file
    /// rather than transitioning out of this state). Inference stays keyed
    /// to `auth.api_keys` as normal (empty → open) — this only hardens the
    /// management surface, not the trusted-alpha inference default. An
    /// operator creates the keys file and restarts to lift it; there is no
    /// API-driven way out by design (see the project's internal engineering log — the same
    /// "key material only enters via a filesystem write" invariant the
    /// keys-file split itself rests on).
    pub admin_locked: bool,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            model_manager: None,
            metrics_handle: None,
            voice_pin: Arc::new(VoicePinManager::default()),
            realtime_sessions: Arc::new(RwLock::new(HashMap::new())),
            device_budgets: Arc::new(DeviceBudgets::default()),
            log_buffer: LogRingBuffer::default(),
            auth: Arc::new(ArcSwap::new(Arc::new(AuthConfig {
                api_keys: Vec::new(),
                admin_api_keys: Vec::new(),
            }))),
            admin_locked: false,
        }
    }
}

impl AppState {
    /// Creates production state without a model manager attached.
    ///
    /// Call `with_model_manager()` after model loading to attach one.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Attaches an already-initialised [`ModelManager`].
    ///
    /// # Example (in main.rs):
    /// ```ignore
    /// let config = Config::load(Path::new("/opt/rustedvino/config.json"))?;
    /// let mm = Arc::new(ModelManager::new(config, Arc::new(OvEngineFactory)).await?);
    /// let state = AppState::new().with_model_manager(mm);
    /// ```
    #[must_use]
    pub fn with_model_manager(mut self, mm: Arc<ModelManager>) -> Self {
        self.model_manager = Some(mm);
        self
    }

    /// Attaches the Prometheus recorder handle (from `metrics::install()`).
    ///
    /// Without it, `GET /metrics` returns 503.
    #[must_use]
    pub fn with_metrics_handle(mut self, handle: PrometheusHandle) -> Self {
        self.metrics_handle = Some(handle);
        self
    }

    /// Overrides the default-constructed voice pin with an externally-built
    /// one — used in production so the **same** instance is also handed to
    /// `ModelManager::with_voice_pin`, letting eviction clear a stale pin
    /// (`voice_pin::VoicePinManager::clear_if_llm`). Without this, `AppState`
    /// and `ModelManager` would each hold their own independent pin and
    /// eviction could never reach the one handlers actually read.
    #[must_use]
    pub fn with_voice_pin(mut self, voice_pin: Arc<VoicePinManager>) -> Self {
        self.voice_pin = voice_pin;
        self
    }

    /// Overrides the default-constructed (passthrough) `DeviceBudgets` with
    /// one built from the loaded config (`DeviceBudgets::from_config`).
    #[must_use]
    pub fn with_device_budgets(mut self, device_budgets: Arc<DeviceBudgets>) -> Self {
        self.device_budgets = device_budgets;
        self
    }

    /// Overrides the default-constructed `LogRingBuffer` with the *same*
    /// instance `main` installed as the `tracing_subscriber` writer — without
    /// this, the admin endpoint would read an empty buffer that never
    /// receives any lines.
    #[must_use]
    pub fn with_log_buffer(mut self, log_buffer: LogRingBuffer) -> Self {
        self.log_buffer = log_buffer;
        self
    }

    /// Overrides the default-constructed (empty) auth state with an
    /// externally-built one — used in production so the **same** `Arc` is
    /// also handed to `ModelManager::with_auth`, mirroring `with_voice_pin`'s
    /// shared-instance pattern. Without this, a key rotation written through
    /// the `ModelManager`-held clone would update a *different* `ArcSwap`
    /// than the one the auth middleware reads, and reload would silently do
    /// nothing.
    #[must_use]
    pub fn with_auth(mut self, auth: Arc<ArcSwap<AuthConfig>>) -> Self {
        self.auth = auth;
        self
    }

    /// Sets the boot-time admin lockdown flag — pass `true` only when
    /// `bootstrap` found no keys file at all. See [`admin_locked`](Self::admin_locked)'s
    /// doc comment.
    #[must_use]
    pub fn with_admin_locked(mut self, admin_locked: bool) -> Self {
        self.admin_locked = admin_locked;
        self
    }

    /// Creates test/mock state with no model manager (no GPU required).
    ///
    /// Integration tests use this so they never load a model or touch GPU.
    /// Chat requests fall through to the mock token generator.
    #[must_use]
    pub fn mock() -> Self {
        Self::new()
    }
}
