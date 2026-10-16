// ============================================================
// src/startup.rs — reusable server bootstrap + serve/shutdown
// ============================================================
// Extracted from main.rs so a second binary linking against this crate
// (e.g. RustedVINO Bear Edition, a private crate — see
// the project's internal engineering log) can build its
// own Router via create_router_with_extension and still get the exact same
// config load, device probe, model-manager construction, and graceful-
// shutdown behaviour as the stock binary, without copying main.rs by hand.
//
// Deliberately NOT covered here (stays each binary's own main.rs):
// tracing/log-buffer init (must run before anything logs), the
// `--supervise` and `--list-devices`/`--device-info` early-exit CLI flags,
// and installing the Unix crash handler — all process-entry decisions a
// binary makes for itself, not part of "build me an AppState and run it."
// ============================================================

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use crate::app_state::AppState;
use crate::device_inventory::DeviceInventory;
use crate::log_ring::LogRingBuffer;
use crate::model_manager::config::resolve_keys_file_path;
use crate::model_manager::{Config, ModelManager};

/// When this process finished starting up, set once by [`bootstrap`]. Backs the
/// `uptime_secs` the admin `server` block reports, so `rv status` gets the PID
/// and uptime from the server itself instead of probing the OS (the probes are
/// Unix-only).
static PROCESS_START: OnceLock<Instant> = OnceLock::new();

/// Records the process start time (first call wins; later calls are no-ops).
/// [`bootstrap`] calls it first, so a second binary linking this crate gets it
/// for free.
pub fn mark_process_start() {
    let _ = PROCESS_START.set(Instant::now());
}

/// Whole seconds since [`mark_process_start`], or `None` if it never ran
/// (unit tests, or a binary that skipped [`bootstrap`]).
#[must_use]
pub fn process_uptime_secs() -> Option<u64> {
    PROCESS_START.get().map(|t| t.elapsed().as_secs())
}

/// Compiled-in default config path for Linux production installs.
const DEFAULT_CONFIG: &str = "/opt/rustedvino/config.json";

/// How long after a shutdown signal in-flight requests may keep running
/// before the engines are torn down underneath them (T6.1).
///
/// Most generations finish well inside this window and their clients see a
/// normal completion. Anything still running when it expires gets a terminal
/// SSE error + `[DONE]` from the engine-loop exit path (T1.3) instead of a
/// severed connection. Keep this under systemd's `TimeoutStopSec` (default
/// 90 s) so the clean path always wins over SIGKILL.
pub const DRAIN_DEADLINE: Duration = Duration::from_secs(30);

/// Hard cap on engine teardown so process shutdown stays inside the stop
/// script's 45s budget even with several engines or a slow eviction (#5). Each
/// per-engine join is itself bounded (`model_manager` `JOIN_DEADLINE`); this
/// caps the sequential total: 30s drain + 10s teardown + a short server-close
/// window stays under 45s.
pub const SHUTDOWN_DEADLINE: Duration = Duration::from_secs(10);

/// The per-user config location on Windows: `%LOCALAPPDATA%\RustedVINO\config.json`
/// (no elevation to write, survives upgrading the program folder). Takes the
/// `LOCALAPPDATA` value as a parameter so the rule is testable on any OS; `None`
/// when it is unset or empty. [`resolve_config_path`] only consults it on
/// Windows, and `tools/rv` carries a copy pinned to this one by
/// `tests/rv_keys_path_parity.rs`.
#[must_use]
pub fn per_user_config_path(local_app_data: Option<&std::ffi::OsStr>) -> Option<PathBuf> {
    let base = local_app_data.filter(|v| !v.is_empty())?;
    Some(Path::new(base).join("RustedVINO").join("config.json"))
}

/// Resolves the config file path using four-level priority:
///
/// 1. `RUSTEDVINO_CONFIG` env var — explicit override, always wins.
/// 2. The compiled-in default `/opt/rustedvino/config.json` if it exists
///    (Linux system install — unchanged production behaviour).
/// 3. Windows only: `%LOCALAPPDATA%\RustedVINO\config.json` if it exists (the
///    per-user location an installer or `rv` setup writes to).
/// 4. `{exe_dir}/config.json` — portable path for Windows bundles and any
///    install that doesn't use `/opt/rustedvino/`.
///
/// # Errors
///
/// Returns an error listing every path tried when none is found, or when
/// `RUSTEDVINO_CONFIG` is set but points at a path that doesn't exist.
pub fn resolve_config_path() -> anyhow::Result<PathBuf> {
    if let Ok(p) = std::env::var("RUSTEDVINO_CONFIG") {
        let path = PathBuf::from(&p);
        anyhow::ensure!(
            path.exists(),
            "RUSTEDVINO_CONFIG={p:?} does not exist — check the path"
        );
        return Ok(path);
    }
    let default = Path::new(DEFAULT_CONFIG);
    if default.exists() {
        return Ok(default.to_path_buf());
    }
    if cfg!(windows)
        && let Some(p) = per_user_config_path(std::env::var_os("LOCALAPPDATA").as_deref())
        && p.exists()
    {
        return Ok(p);
    }
    let exe_dir_config = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|d| d.join("config.json")));
    if let Some(ref p) = exe_dir_config
        && p.exists()
    {
        return Ok(p.clone());
    }
    let exe_hint = exe_dir_config.map_or_else(
        || "<exe-dir>/config.json".to_string(),
        |p| p.display().to_string(),
    );
    anyhow::bail!(
        "no config file found — tried:\n  \
         • RUSTEDVINO_CONFIG env var: not set\n  \
         • {DEFAULT_CONFIG}: not found\n  \
         • (Windows) %LOCALAPPDATA%\\RustedVINO\\config.json: not found\n  \
         • {exe_hint}: not found\n\
         \nFix: set RUSTEDVINO_CONFIG=/path/to/config.json"
    )
}

/// Exit code of a worker that stopped itself because the GPU context was
/// poisoned (`EX_TEMPFAIL`). The supervisor recognises it in its log; it is respawned.
pub const EXIT_GPU_POISONED: u8 = 75;

/// Exit code of a worker that cannot start because of a deterministic
/// configuration or filesystem error (`EX_CONFIG`) - an unreadable or invalid
/// config file, a `preload` model that is not registered. Retrying the same
/// inputs can never succeed, so the supervisor does NOT respawn on it.
pub const EXIT_CONFIG_FATAL: u8 = 78;

/// A startup error that no restart can fix. Wrap it in the `anyhow::Error` a
/// startup step returns; `main` finds it with `downcast_ref` and exits with
/// [`EXIT_CONFIG_FATAL`] instead of the generic failure code.
#[derive(Debug)]
pub struct ConfigFatal(pub String);

impl std::fmt::Display for ConfigFatal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigFatal {}

/// A shutdown signal arrived while the server was still starting (preloading
/// models). `bootstrap` returns it; `main` treats it as a clean stop (exit 0),
/// not an error.
#[derive(Debug)]
pub struct ShutdownDuringStartup;

impl std::fmt::Display for ShutdownDuringStartup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("shutdown requested during startup")
    }
}

impl std::error::Error for ShutdownDuringStartup {}

/// Tags `e` as [`ConfigFatal`], keeping its full cause chain in the message.
#[must_use]
pub fn config_fatal(e: &anyhow::Error) -> anyhow::Error {
    anyhow::Error::new(ConfigFatal(format!("{e:#}")))
}

/// Set by the supervisor on the worker it spawns. A worker without it has no
/// one to restart it, so it stays up (degraded, and says so) instead of exiting.
pub const SUPERVISED_ENV: &str = "RUSTEDVINO_SUPERVISED";

/// Wakes every [`shutdown_signal`] waiter without an OS signal (the same drain
/// path as SIGTERM). `shutdown_signal` is awaited by `serve` AND by background
/// loops (cache sweep, KV monitor), so the request must reach all of them:
/// `notify_one` woke only one (usually a background loop, which then quietly
/// returned and left `serve` running - review finding). The request is a sticky
/// flag plus `notify_waiters`; see [`wait_for_flag`].
static SHUTDOWN_REQUEST: tokio::sync::Notify = tokio::sync::Notify::const_new();

/// Set once, never cleared: a shutdown has been requested.
static SHUTDOWN_REQUESTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Set by [`serve`] once the listener is bound.
static SERVING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the shutdown in progress was asked for by [`on_gpu_poisoned`].
static GPU_POISON_EXIT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// True when the process is stopping because the GPU context was poisoned;
/// `main` turns it into [`EXIT_GPU_POISONED`].
#[must_use]
pub fn gpu_poison_exit_requested() -> bool {
    GPU_POISON_EXIT.load(std::sync::atomic::Ordering::Acquire)
}

/// Resolves once `flag` is set. Any number of tasks may wait on the same
/// `notify`/`flag` pair and all of them resolve, including ones that start
/// waiting after the flag was set. The `Notified` future is registered
/// (`enable`) BEFORE the flag is read, so a request landing between the read
/// and the await is not lost.
async fn wait_for_flag(notify: &tokio::sync::Notify, flag: &std::sync::atomic::AtomicBool) {
    loop {
        let notified = notify.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if flag.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        notified.await;
    }
}

/// Sets `flag` and wakes every waiter of `notify` (see [`wait_for_flag`]).
fn raise_flag(notify: &tokio::sync::Notify, flag: &std::sync::atomic::AtomicBool) {
    flag.store(true, std::sync::atomic::Ordering::Release);
    notify.notify_waiters();
}

/// Asks the server to shut down gracefully - the same drain path SIGTERM takes.
/// Safe to call from any thread, including one outside the tokio runtime (the
/// dedicated signal thread in `shutdown_signals`).
pub fn request_shutdown() {
    raise_flag(&SHUTDOWN_REQUEST, &SHUTDOWN_REQUESTED);
}

/// Records that the server is accepting requests ([`serve`] calls it once the
/// listener is bound). Before this, a GPU poisoning does not stop the process
/// (see [`on_gpu_poisoned`]).
pub fn mark_serving() {
    SERVING.store(true, std::sync::atomic::Ordering::Release);
}

/// Reaction to the FIRST GPU-context poisoning. `from_serving_model` is true
/// when a model that was already serving failed (the inference path); false
/// when the failing operation was a model LOAD. The context is wedged for the
/// life of the process (decisions 2026-06-02: a restart is the only proven
/// cure), so under `--supervise` the worker drains and exits and the
/// supervisor respawns it. It stays up, degraded, when there is no supervisor
/// (nothing would restart it), when the poisoning happened during startup, or
/// when a model LOAD caused it (a preload or on-demand/admin load OOM: the same
/// load fails again after a restart, a client that retries it would turn that
/// into a restart loop, and the CPU/NPU models can keep serving). Always logged at ERROR - an operator reading the log must be able
/// to see why the server went away or why it did not.
pub fn on_gpu_poisoned(model_id: &str, from_serving_model: bool) {
    let supervised = std::env::var_os(SUPERVISED_ENV).is_some();
    let serving = SERVING.load(std::sync::atomic::Ordering::Acquire);
    if supervised && serving && from_serving_model {
        tracing::error!(
            model_id,
            exit_code = EXIT_GPU_POISONED,
            "GPU context poisoned - draining in-flight requests and exiting so the supervisor \
             restarts the worker (engine teardown is skipped: a poisoned context refuses it)"
        );
        GPU_POISON_EXIT.store(true, std::sync::atomic::Ordering::Release);
        raise_flag(&SHUTDOWN_REQUEST, &SHUTDOWN_REQUESTED);
    } else if supervised && serving {
        tracing::error!(
            model_id,
            "GPU context poisoned while LOADING a model - NOT exiting: the same load would fail \
             again after a restart (usually a wrong vram_gb / KV estimate), and the CPU/NPU \
             models keep serving; GPU models are unusable until the server is restarted"
        );
    } else if supervised {
        tracing::error!(
            model_id,
            "GPU context poisoned during startup - NOT exiting (a restart would repeat the \
             failure); staying up degraded, GPU models are unusable until a manual restart"
        );
    } else {
        tracing::error!(
            model_id,
            "GPU context poisoned - NOT exiting (no --supervise parent to restart the worker); \
             every GPU model is unusable until the server is restarted manually"
        );
    }
}

/// Terminates the process immediately with `code`, running no `atexit` handlers, static destructors or
/// thread teardown. Used only for the GPU-poisoned exit, where the normal teardown crashes (see [`serve`]).
#[cfg(unix)]
fn exit_without_destructors(code: u8) -> ! {
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    // SAFETY: `_exit` only reads an integer and ends the process; it is async-signal-safe and touches no
    // Rust state. Skipping destructors is the intent: they would run pipeline teardown on a poisoned context.
    unsafe { libc::_exit(i32::from(code)) }
}

/// Non-Unix fallback (the supervisor and the poison exit are Unix-only today): a normal exit.
#[cfg(not(unix))]
fn exit_without_destructors(code: u8) -> ! {
    std::process::exit(i32::from(code))
}

/// Proves `dir` exists (creating it) and accepts a new file, then removes the
/// probe. `OpenVINO` skips its compile cache without a word when it cannot write
/// there, so an unwritable `ov_cache_dir` looks like a working server that
/// recompiles every model on every load; this makes the cause visible at boot.
///
/// # Errors
///
/// The I/O error that stopped it (permission denied, a file in the way, ...).
pub fn probe_cache_dir_writable(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let probe = dir.join(".write-probe");
    std::fs::write(&probe, b"x")?;
    std::fs::remove_file(&probe)
}

/// Proves an existing file can be opened for writing, without changing it
/// (append mode, nothing written). Used for the files `rv` and the admin API
/// rewrite in place (`config.json`, the keys file): an ACL or read-only
/// attribute that denies the write otherwise shows up only later, as an `rv
/// patch` failing with "access denied". A missing file is not an error here.
///
/// # Errors
///
/// The I/O error that stopped it (permission denied, ...).
pub fn probe_file_writable(path: &Path) -> std::io::Result<()> {
    if !path.exists() {
        return Ok(());
    }
    std::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .map(|_| ())
}

/// Result of [`bootstrap`] — everything needed to build a [`axum::Router`]
/// (via [`crate::create_router_with_auth`] or
/// [`crate::create_router_with_extension`]) and then hand it to [`serve`].
pub struct Startup {
    pub state: AppState,
    pub cors_origins: Vec<String>,
    /// Loaded from the resolved keys file, not from `Config` — see
    /// `resolve_keys_file_path`/`AuthConfig::load_from_file` in `bootstrap`.
    pub api_keys: Vec<String>,
    /// Loaded from the resolved keys file, not from `Config` — see
    /// `resolve_keys_file_path`/`AuthConfig::load_from_file` in `bootstrap`.
    pub admin_api_keys: Vec<String>,
    pub addr: SocketAddr,
    pub model_manager: Arc<ModelManager>,
}

/// Loads config, probes the device inventory, builds the model manager
/// (preloading configured models), and assembles [`AppState`]. Call after
/// tracing is initialised and any early-exit CLI flags have been handled.
///
/// `log_buffer` must be the same instance already installed as the tracing
/// writer, so `GET /v1/admin/logs/tail` sees the same ring buffer the
/// process has actually been logging into.
///
/// # Errors
///
/// Returns an error if the config file can't be resolved/loaded/validated
/// (including the T4.2 security gate on a public bind with no API keys), if
/// no `OpenVINO` device enumerates at all, if the Prometheus recorder is
/// already installed, or if the model manager fails to construct (e.g. a
/// preload model fails to load).
#[allow(clippy::too_many_lines)]
pub async fn bootstrap(log_buffer: LogRingBuffer) -> anyhow::Result<Startup> {
    mark_process_start();
    let config_path = resolve_config_path().map_err(|e| config_fatal(&e))?;
    tracing::info!(path = %config_path.display(), "loading config");
    let config = Config::load(&config_path).map_err(|e| config_fatal(&e))?;
    if let Some(dir) = config.ov_cache_dir.as_deref().filter(|d| !d.is_empty())
        && let Err(e) = probe_cache_dir_writable(Path::new(dir))
    {
        tracing::warn!(
            path = dir,
            error = %e,
            "ov_cache_dir is not writable — OpenVINO will silently skip its compile cache and \
             recompile every model on every load (on Windows: check the folder's ACL and \
             integrity level; staging it from another OS can leave it read-only to this user)"
        );
    }

    // Dynamic KV allocation plus prefix caching: retained prefix blocks then grow
    // outside the VRAM budget (see `Config::enable_prefix_caching`).
    if config.cache_size_gb == 0.0 && config.enable_prefix_caching {
        tracing::warn!(
            "cache_size_gb is 0.0 (dynamic KV allocation) with enable_prefix_caching on - \
             retained prefix blocks can grow without bound outside the memory budget; set a \
             fixed cache_size_gb or turn enable_prefix_caching off"
        );
    }

    // The compile cache's sidecar directory and the config file are written at
    // runtime too (model-hash memo / provenance; `rv patch`, admin edits). Same
    // Windows failure as above, different symptom, so same early warning.
    if let Some(root) = crate::cache_manifest::manifest_root(config.ov_cache_dir.as_deref())
        && let Err(e) = probe_cache_dir_writable(&root)
    {
        tracing::warn!(
            path = %root.display(), error = %e,
            "cache_manifest directory is not writable — model-hash memoisation and provenance \
             records will not be saved (check the folder's ACL and integrity level)"
        );
    }
    if let Err(e) = probe_file_writable(&config_path) {
        tracing::warn!(
            path = %config_path.display(), error = %e,
            "config file is not writable — `rv patch` and admin config edits will fail \
             (check the file's ACL; staging from another OS can leave it read-only)"
        );
    }

    // ── Probe the device inventory (Phase A) ────────────────────────────
    // Snapshot every OpenVINO device's kind / tier / memory-domain at startup
    // and log the topology. The domain-keyed memory tracker (Phase B) consumes
    // it to resolve the inference memory domain and per-domain budgets;
    // placement (Phase C) will read the tiers. Probed once and shared.
    let inventory = Arc::new(DeviceInventory::probe());
    inventory.log();

    // ── Capability-based startup assertion (Phase D) ────────────────────
    // Fail fast, before any model load, only when OpenVINO enumerated *no*
    // inference device at all (GPU driver or runtime not accessible). This
    // replaces the pre-C2 single-device check (`device_available(config.device)`,
    // historically `devices.contains("GPU.1")`): it names no device, so a box
    // without a discrete GPU — Lunar Lake (CPU + iGPU + NPU) — still boots.
    // Per-model serveability is enforced separately at model registration, where
    // each preloaded model's resolved device is validated against this inventory
    // and an unplaceable model is rejected with a clear error.
    anyhow::ensure!(
        inventory.has_serveable_device(),
        "OpenVINO enumerated no inference devices — verify the GPU driver and \
         OpenVINO runtime are installed and accessible"
    );
    // `config.device` is the memory-domain *budget anchor* (resolve_inference_domain),
    // not necessarily where any model loads (C2 places each model on its own
    // resolved device). If it isn't enumerated, warn but keep booting: the anchor
    // domain falls back to the device name and registration is the real gate.
    if inventory.get(&config.device).is_some() {
        tracing::info!(device = %config.device, "budget-anchor device enumerated");
    } else if inventory.spec_known(&config.device) {
        // A HETERO list whose members all exist: usable for placement, but the memory domain
        // lookup is by plain device name and falls back to the literal spec.
        tracing::info!(
            device = %config.device,
            "global device is a HETERO list of enumerated devices; its memory domain falls back to \
             the literal name"
        );
    } else {
        let enumerated: Vec<&str> = inventory
            .devices()
            .iter()
            .map(|d| d.name.as_str())
            .collect();
        // With default_placement "device" every model without its own `device` is placed here
        // and will be refused, so say so; otherwise the device is only the budget anchor.
        let consequence = if config.default_placement
            == crate::model_manager::config::DefaultPlacement::Device
        {
            "every model without its own `device` will fail to load (default_placement = \"device\")"
        } else {
            "the inference memory domain falls back to the device name; each model is still placed on its own resolved device"
        };
        tracing::warn!(
            device = %config.device,
            enumerated = ?enumerated,
            "configured global `device` is not an enumerated OpenVINO device: {consequence}"
        );
    }

    // ── Start the model manager (preloads models) ───────────────────────
    // `ModelManager::new()` calls `tokio::task::spawn_blocking` internally
    // for each preload model — the async runtime is not stalled.
    tracing::info!(
        models_dir = %config.models_dir.display(),
        device = %config.device,
        preload = ?config.preload,
        "initialising model manager",
    );

    // Read fields out of config before it is moved into the ModelManager.
    let cors_origins = config.cors_allowed_origins.clone();

    let keys_path = resolve_keys_file_path(&config_path, config.keys_file.as_deref());
    if let Err(e) = probe_file_writable(&keys_path) {
        tracing::warn!(
            path = %keys_path.display(), error = %e,
            "keys file is not writable — key rotation through the CLI will fail (check the \
             file's ACL)"
        );
    }
    check_no_inline_keys(&config, &keys_path)?;
    let (loaded_keys, admin_locked) = load_keys_file_for_boot(&keys_path)?;
    let api_keys = loaded_keys.api_keys.clone();
    let admin_api_keys = loaded_keys.admin_api_keys.clone();
    let has_keys = !api_keys.is_empty() || !admin_api_keys.is_empty();

    // Built before the model manager and shared with it below (same pattern
    // as `voice_pin` just below), so `POST /v1/admin/keys/reload` can
    // rotate keys through `ModelManager`'s clone and have the router's auth
    // middleware — reading `AppState::auth`, the *same* `Arc` — see the
    // change on the very next request. No server restart, no dropped
    // connections, no VRAM-resident models evicted.
    let auth = Arc::new(arc_swap::ArcSwap::new(Arc::new(loaded_keys)));
    let ov_cache_sweep_interval = Duration::from_secs(config.ov_cache_sweep_interval_secs);
    // Read before `config` moves into `ModelManager::new_production` below —
    // both `Copy` types, no clone needed. `kv_pressure_monitor_enabled`
    // gates whether the sweep task is spawned at all
    // (the project's internal engineering log's ops-review finding: default off, and
    // when off there should be zero periodic cost, not just an inert check).
    let kv_pressure_monitor_enabled = config.kv_pressure_monitor_enabled;
    let kv_pressure_sweep_interval = Duration::from_secs(config.kv_pressure_sweep_interval_secs);
    let device_budgets = Arc::new(crate::admission::DeviceBudgets::from_config(&config));

    // T4.2: resolve the bind address, then enforce the public-bind-requires-
    // auth gate BEFORE any model load — no point spending minutes of GPU JIT
    // to then abort. `has_keys` comes from the keys file loaded just above,
    // not from `Config` (see `Config::enforce_open_bind_gate`'s doc comment
    // for why the gate split in two).
    let addr = config.validated_bind()?;
    config.enforce_open_bind_gate(addr, has_keys)?;

    // Install the Prometheus recorder before any metric is touched (it is
    // process-global and can only be installed once).
    let metrics_handle = crate::metrics::install()?;

    // Built before the model manager and shared with it below, so eviction
    // can clear a stale pin (voice_pin::VoicePinManager::clear_if_llm) —
    // the same instance AppState hands to realtime handlers.
    let voice_pin = Arc::new(crate::voice_pin::VoicePinManager::default());

    // R1: VLMs are managed models. The manager's engine factory classifies each
    // preload entry (`detect_kind`) and builds a CB or VLM engine accordingly —
    // no caller special-casing. LLM + VLM preload uniformly here.
    // D-02 (0.8.3): a preload can sit inside one GPU compile for minutes, and a
    // blocking OpenVINO call cannot be cancelled. Race the construction against
    // the shutdown signal so SIGTERM/ctrl-C is honoured at once; dropping the
    // future abandons the compile, and process exit tears its thread down.
    let constructed = tokio::select! {
        r = ModelManager::new_production(config, Arc::clone(&inventory)) => r?,
        () = shutdown_signal() => {
            tracing::warn!(
                "shutdown signal received while models were still loading - abandoning \
                 startup (nothing was serving yet)"
            );
            return Err(anyhow::Error::new(ShutdownDuringStartup));
        }
    };
    let model_manager = Arc::new(
        constructed
            .with_config_path(config_path.clone())
            .with_voice_pin(Arc::clone(&voice_pin))
            .with_auth(Arc::clone(&auth))
            .with_keys_file_path(keys_path),
    );
    tracing::info!("model manager ready");

    // ── OV-cache background sweep (the project's internal engineering log) ──
    // hash-precompute + prune, once at startup then every
    // `ov_cache_sweep_interval_secs`. Deliberately does NOT include
    // blob-warming (see `ModelManager::is_idle`'s doc comment) — both passes
    // here are safe regardless of live traffic, so no idle gate is needed.
    // `run_cache_sweep_once_blocking` is blocking file I/O (including a
    // SHA-256 stream over multi-GB backbone files), hence `spawn_blocking`
    // rather than running it directly on this task.
    {
        let model_manager = Arc::clone(&model_manager);
        tokio::spawn(async move {
            loop {
                let mm = Arc::clone(&model_manager);
                if let Err(e) =
                    tokio::task::spawn_blocking(move || mm.run_cache_sweep_once_blocking()).await
                {
                    tracing::error!(error = %e, "ov cache sweep task panicked");
                }
                tokio::select! {
                    () = tokio::time::sleep(ov_cache_sweep_interval) => {}
                    () = shutdown_signal() => return,
                }
            }
        });
    }

    // ── KV-cache pressure monitor (the project's internal engineering log) ──
    // Detect-and-flag only — never evicts, resizes, or otherwise acts. Not
    // spawned at all when disabled (the fleet-wide default), per the plan's
    // ops-review finding: no periodic cost, not just an inert per-tick check.
    // Unlike the OV-cache sweep above, every read here is a fast in-memory/
    // atomic load, so this runs directly on the task, no `spawn_blocking`.
    if kv_pressure_monitor_enabled {
        let model_manager = Arc::clone(&model_manager);
        tokio::spawn(async move {
            loop {
                model_manager.run_kv_pressure_sweep_once();
                tokio::select! {
                    () = tokio::time::sleep(kv_pressure_sweep_interval) => {}
                    () = shutdown_signal() => return,
                }
            }
        });
    }

    let state = AppState::new()
        .with_model_manager(Arc::clone(&model_manager))
        .with_metrics_handle(metrics_handle)
        .with_voice_pin(voice_pin)
        .with_device_budgets(device_budgets)
        .with_log_buffer(log_buffer)
        .with_auth(auth)
        .with_admin_locked(admin_locked);
    if admin_locked {
        tracing::warn!(
            "/v1/admin/* is locked for this process's lifetime — no keys file existed at boot"
        );
    }
    tracing::info!(?cors_origins, "CORS policy");
    tracing::info!(
        auth = if api_keys.is_empty() {
            "open"
        } else {
            "bearer"
        },
        keys = api_keys.len(),
        admin_scope = if admin_api_keys.is_empty() {
            "shared"
        } else {
            "separate"
        },
        admin_keys = admin_api_keys.len(),
        "API auth policy"
    );

    Ok(Startup {
        state,
        cors_origins,
        api_keys,
        admin_api_keys,
        addr,
        model_manager,
    })
}

/// Migration tripwire (the project's internal engineering log — the keys-file split): a non-empty
/// `config.api_keys`/`config.admin_api_keys` almost certainly means an
/// operator forgot to migrate to the separate keys file — these fields are
/// deprecated and never read for auth (see [`Config::api_keys`]'s doc
/// comment). Silently ignoring it would be the same silent-non-revocation
/// failure the split exists to prevent: the operator believes a key is
/// active; it authenticates nothing. `reload_config`'s doc comment on
/// `ModelManager` runs this same check again on every reload, not just boot.
///
/// # Errors
/// `config.api_keys` or `config.admin_api_keys` is non-empty.
fn check_no_inline_keys(config: &Config, keys_path: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        config.api_keys.is_empty() && config.admin_api_keys.is_empty(),
        "config.json has deprecated inline api_keys/admin_api_keys — these are no longer \
         read for auth. Move them to {} (never git-tracked, see CONFIG.md's keys_file entry) \
         and remove them from config.json",
        keys_path.display()
    );
    Ok(())
}

/// Boot-time keys loading — the one place this project's usual "fail-fast"
/// instinct is deliberately relaxed: a **missing** keys file boots the
/// server open for *inference* (empty [`crate::AuthConfig`]), with a loud
/// warning naming the resolved path, the same posture the old empty-
/// `api_keys` default had (a fresh box that hasn't had a keys file created
/// yet shouldn't hard-fail to boot) — but the returned `bool` is `true` in
/// exactly this case, which the caller uses to permanently lock
/// `/v1/admin/*` for the process's lifetime (`AppState::admin_locked` — see
/// its doc comment for why inference and admin get different defaults
/// here). A **present but malformed** file still bails —
/// [`crate::AuthConfig::load_from_file`]'s error. `ModelManager::
/// reload_keys_file` treats a *reload's* missing file as an error, not open
/// — see its doc comment for why boot and reload differ here.
///
/// # Errors
/// The file exists but can't be read or fails to parse.
fn load_keys_file_for_boot(keys_path: &Path) -> anyhow::Result<(crate::AuthConfig, bool)> {
    if keys_path.exists() {
        Ok((crate::AuthConfig::load_from_file(keys_path)?, false))
    } else {
        tracing::warn!(
            path = %keys_path.display(),
            "keys file not found — server boots with authentication OFF for inference, but \
             /v1/admin/* is permanently locked until one is created and the server restarts \
             (see CONFIG.md's keys_file entry)"
        );
        Ok((crate::AuthConfig::default(), true))
    }
}

/// Resolves when the process receives SIGTERM (`systemctl stop`, rolling
/// deploy) or SIGINT (ctrl-C). Never resolves if neither handler can be
/// installed — the server then simply runs until killed, which is the
/// pre-T6.1 behaviour, not a new failure mode.
///
/// `tokio::signal::unix` does not exist on Windows (E0433 on the MSVC
/// target), so the SIGTERM arm is Unix-only. On Windows, ctrl-C alone covers
/// foreground operation; the SCM stop event funnels into the same drain path
/// when the service wrapper lands.
async fn shutdown_signal() {
    // CRASH COURSE — why not `?` here: this future is handed to axum as a
    // plain `Future<Output = ()>`; it has no Result to propagate into. A
    // handler-installation failure is logged and replaced with a future that
    // never resolves (`pending()`), so the other signal still works.
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!(error = %e, "failed to install ctrl-c handler");
            std::future::pending::<()>().await;
        }
    };
    #[cfg(unix)]
    let sigterm = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => {
                tracing::error!(error = %e, "failed to install SIGTERM handler");
                std::future::pending::<()>().await;
            }
        }
    };
    // CRASH COURSE — cfg on an expression: both `sigterm` bindings have the
    // same name, but only one exists per platform. The select! below refers
    // to whichever survived compilation — no runtime branch, no dead code.
    #[cfg(not(unix))]
    let sigterm = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {},
        () = sigterm => {},
        () = wait_for_flag(&SHUTDOWN_REQUEST, &SHUTDOWN_REQUESTED) => {},
    }
}

/// Tear down all engines, bounded by [`SHUTDOWN_DEADLINE`] so process exit can
/// never hang past the stop script's budget (#5). Each per-engine join inside
/// `shutdown()` is itself bounded; this caps the sequential total.
async fn teardown_engines(model_manager: &ModelManager) {
    if tokio::time::timeout(SHUTDOWN_DEADLINE, model_manager.shutdown())
        .await
        .is_err()
    {
        tracing::error!("engine teardown exceeded shutdown deadline — exiting anyway");
    }
}

/// Serves `app` on `addr` with graceful shutdown (T6.1): stop accepting new
/// connections on SIGTERM/ctrl-C, give in-flight requests up to
/// [`DRAIN_DEADLINE`] to finish naturally, then tear down the engines — which
/// terminates any remaining stream with a clean SSE error + `[DONE]` (T1.3),
/// joins the engine threads, and frees VRAM via `Drop` before returning.
///
/// # Errors
///
/// Returns an error if `addr` can't be bound, or if the server task itself
/// ends with an error (still falls through to engine teardown first — VRAM
/// release is never skipped on a server error).
pub async fn serve(
    app: axum::Router,
    addr: SocketAddr,
    model_manager: &ModelManager,
) -> anyhow::Result<()> {
    tracing::info!(%addr, "RustedVINO listening");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    mark_serving();

    // The oneshot tells the select below *when* the signal fired so the drain
    // deadline starts then, not at process start.
    let (sig_tx, sig_rx) = tokio::sync::oneshot::channel::<()>();
    // CRASH COURSE — `into_future()`: axum's `WithGracefulShutdown` is not
    // itself a `Future`, only `IntoFuture` (a builder that *yields* one). A
    // bare `.await` converts implicitly, but `select!` must poll it by
    // `&mut` reference — so the conversion has to be explicit.
    let server = std::future::IntoFuture::into_future(
        axum::serve(listener, app).with_graceful_shutdown(async move {
            shutdown_signal().await;
            tracing::info!("shutdown signal received — draining in-flight requests");
            let _ = sig_tx.send(());
        }),
    );

    // CRASH COURSE — `pin!`: `select!` needs `&mut server` so the future can
    // be resumed after the deadline branch wins, but polling through a `&mut`
    // requires the future to be pinned (guaranteed never to move in memory,
    // because it may hold self-references across await points). `pin!` pins
    // it to this stack frame.
    let mut server = std::pin::pin!(server);
    let deadline = async {
        // If the oneshot errors the server ended without a signal (bind/accept
        // error) — park forever and let the other select branch report it.
        if sig_rx.await.is_err() {
            std::future::pending::<()>().await;
        }
        tokio::time::sleep(DRAIN_DEADLINE).await;
    };

    tokio::select! {
        // Normal path: every connection closed within the drain window (or the
        // server failed). Log a serve-task error but still fall through to engine
        // teardown — never skip VRAM release on a server error (#1).
        res = server.as_mut() => {
            if let Err(e) = res {
                tracing::error!(error = %e, "server task ended with error — proceeding to teardown");
            }
        }
        // Drain deadline expired with requests still running: tear the engines
        // down now; the terminal events (T1.3) close the remaining streams, after
        // which the server future resolves. Both the teardown and the final
        // server close are bounded so a wedged stream cannot hang exit (#5).
        () = deadline => {
            tracing::warn!(
                deadline_s = DRAIN_DEADLINE.as_secs(),
                "drain deadline expired — terminating remaining requests"
            );
            teardown_engines(model_manager).await;
            let _ = tokio::time::timeout(Duration::from_secs(3), server.as_mut()).await;
        }
    }

    // Idempotent on the deadline path (everything is already evicted); on the
    // normal path this is where engines drop and VRAM is released.
    teardown_engines(model_manager).await;
    tracing::info!("shutdown complete — engines drained and VRAM released");

    if gpu_poison_exit_requested() {
        // Do NOT return: returning drops the model manager, which lets every engine thread destroy its
        // pipeline, and destroying a pipeline on a poisoned GPU context segfaults inside the GPU plugin
        // (seen in an overnight-test rehearsal: SIGSEGV instead of exit code 75, with a crash dump each time).
        tracing::info!(
            exit_code = EXIT_GPU_POISONED,
            "GPU context poisoned: leaving now without running destructors; the supervisor restarts the worker"
        );
        exit_without_destructors(EXIT_GPU_POISONED);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn minimal_config(extra: &str) -> Config {
        let json =
            format!(r#"{{"models_dir": "/tmp/m", "device": "CPU", "total_vram_gb": 0.0{extra}}}"#);
        serde_json::from_str(&json).unwrap()
    }

    // ---- check_no_inline_keys (boot-tripwire) ----------------------------

    /// The common case: no inline keys, nothing to complain about.
    #[test]
    fn check_no_inline_keys_passes_when_both_empty() {
        let cfg = minimal_config("");
        assert!(check_no_inline_keys(&cfg, Path::new("/tmp/x.keys.json")).is_ok());
    }

    /// A non-empty `api_keys` is the migration tripwire — refuse to boot,
    /// and name the resolved keys path in the error so the operator knows
    /// exactly where to move it.
    #[test]
    fn check_no_inline_keys_rejects_inline_api_keys() {
        let cfg = minimal_config(r#", "api_keys": ["sk-old"]"#);
        let err = check_no_inline_keys(&cfg, Path::new("/tmp/x.keys.json"))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("/tmp/x.keys.json"),
            "names the target path: {err}"
        );
        assert!(err.contains("deprecated"), "{err}");
    }

    /// Same tripwire for `admin_api_keys` alone.
    #[test]
    fn check_no_inline_keys_rejects_inline_admin_api_keys() {
        let cfg = minimal_config(r#", "admin_api_keys": ["sk-old-admin"]"#);
        assert!(check_no_inline_keys(&cfg, Path::new("/tmp/x.keys.json")).is_err());
    }

    // ---- load_keys_file_for_boot (boot-missing/malformed) -----------------

    /// A missing keys file boots open for inference (empty `AuthConfig`, not
    /// an error) — but reports `admin_locked: true`, the caller's signal to
    /// lock `/v1/admin/*` for the process's lifetime.
    #[test]
    fn load_keys_file_for_boot_missing_returns_open_and_locks_admin() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nonexistent.keys.json");
        let (auth, admin_locked) = load_keys_file_for_boot(&path).unwrap();
        assert!(auth.api_keys.is_empty());
        assert!(auth.admin_api_keys.is_empty());
        assert!(admin_locked, "missing file must lock admin");
    }

    /// A present, valid keys file loads correctly and leaves admin unlocked
    /// — even when its `admin_api_keys` is itself empty (a deliberate
    /// "share the inference scope" choice, distinct from no file at all).
    #[test]
    fn load_keys_file_for_boot_valid_file_loads_and_unlocks_admin() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("valid.keys.json");
        std::fs::write(
            &path,
            r#"{"api_keys": ["sk-a"], "admin_api_keys": ["sk-admin"]}"#,
        )
        .unwrap();
        let (auth, admin_locked) = load_keys_file_for_boot(&path).unwrap();
        assert_eq!(auth.api_keys, vec!["sk-a".to_string()]);
        assert_eq!(auth.admin_api_keys, vec!["sk-admin".to_string()]);
        assert!(!admin_locked);
    }

    /// A present file with an explicitly empty `admin_api_keys` still
    /// unlocks admin — the "file exists" fact, not the emptiness of any one
    /// field, is what `admin_locked` tracks.
    #[test]
    fn load_keys_file_for_boot_empty_admin_scope_still_unlocks_admin() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shared-scope.keys.json");
        std::fs::write(&path, r#"{"api_keys": ["sk-a"], "admin_api_keys": []}"#).unwrap();
        let (_, admin_locked) = load_keys_file_for_boot(&path).unwrap();
        assert!(
            !admin_locked,
            "an explicit empty admin scope is not the same as no file"
        );
    }

    /// A present but malformed keys file bails — boot's leniency is only for
    /// "missing", never for "present but broken".
    #[test]
    fn load_keys_file_for_boot_malformed_bails() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.keys.json");
        std::fs::write(&path, r#"{"api_keys": "not-an-array"}"#).unwrap();
        assert!(load_keys_file_for_boot(&path).is_err());
    }

    /// A typo'd key (`admin_keys` for `admin_api_keys`) is caught by
    /// `deny_unknown_fields`, not silently treated as "no admin keys".
    #[test]
    fn load_keys_file_for_boot_rejects_unknown_field() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("typo.keys.json");
        std::fs::write(
            &path,
            r#"{"api_keys": ["sk-a"], "admin_keys": ["sk-admin"]}"#,
        )
        .unwrap();
        let err = match load_keys_file_for_boot(&path) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an unknown-field error"),
        };
        assert!(
            err.contains("admin_keys") || err.contains("unknown"),
            "{err}"
        );
    }
    #[test]
    fn per_user_config_path_is_under_localappdata() {
        let p = per_user_config_path(Some(std::ffi::OsStr::new("base"))).unwrap();
        assert!(
            p.ends_with(Path::new("RustedVINO").join("config.json")),
            "{p:?}"
        );
        assert!(p.starts_with("base"));
        assert_eq!(per_user_config_path(None), None);
        assert_eq!(per_user_config_path(Some(std::ffi::OsStr::new(""))), None);
    }

    /// The probe passes on a writable directory (creating it), leaves nothing
    /// behind, and fails where a file blocks the path.
    #[test]
    fn cache_dir_probe_accepts_writable_and_rejects_blocked_paths() {
        let d = tempfile::tempdir().unwrap();
        let target = d.path().join("a/b/ov_cache");
        probe_cache_dir_writable(&target).unwrap();
        assert!(target.is_dir());
        assert_eq!(
            std::fs::read_dir(&target).unwrap().count(),
            0,
            "probe cleans up"
        );

        let file = d.path().join("plain_file");
        std::fs::write(&file, b"x").unwrap();
        assert!(probe_cache_dir_writable(&file.join("sub")).is_err());
        assert!(probe_cache_dir_writable(&file).is_err());
    }

    /// A present, writable file passes untouched; a read-only one fails; a
    /// missing one is not an error (an optional keys file).
    #[test]
    fn file_probe_accepts_writable_and_missing_and_rejects_read_only() {
        let d = tempfile::tempdir().unwrap();
        let f = d.path().join("config.json");
        std::fs::write(&f, b"{}").unwrap();
        probe_file_writable(&f).unwrap();
        assert_eq!(
            std::fs::read(&f).unwrap(),
            b"{}",
            "probe must not alter the file"
        );
        probe_file_writable(&d.path().join("absent.json")).unwrap();

        let mut perms = std::fs::metadata(&f).unwrap().permissions();
        perms.set_readonly(true);
        std::fs::set_permissions(&f, perms).unwrap();
        // Running as root bypasses the permission bit; only assert when it bites.
        if std::fs::OpenOptions::new().append(true).open(&f).is_err() {
            assert!(probe_file_writable(&f).is_err());
        }
    }

    /// Several tasks wait on one shutdown request (`serve` plus the sweep and
    /// KV-monitor loops); ALL of them must wake, and one that starts waiting
    /// after the request must resolve at once. `notify_one` woke only the first.
    /// Local flag/notify, so no process-global state or signal handler is touched.
    #[tokio::test]
    async fn shutdown_request_wakes_every_waiter_including_late_ones() {
        let notify = Arc::new(tokio::sync::Notify::new());
        let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let waiters: Vec<_> = (0..3)
            .map(|_| {
                let (n, f) = (Arc::clone(&notify), Arc::clone(&flag));
                tokio::spawn(async move { wait_for_flag(&n, &f).await })
            })
            .collect();
        tokio::task::yield_now().await;
        assert!(
            waiters.iter().all(|w| !w.is_finished()),
            "nobody wakes early"
        );

        raise_flag(&notify, &flag);
        for w in waiters {
            let woke = tokio::time::timeout(Duration::from_secs(5), w).await;
            assert!(woke.is_ok(), "every waiter must wake on one request");
        }
        let late =
            tokio::time::timeout(Duration::from_secs(5), wait_for_flag(&notify, &flag)).await;
        assert!(
            late.is_ok(),
            "a waiter arriving after the request resolves at once"
        );
    }
}
