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
/// severed connection. After it, a stop may still wait up to [`LOAD_SETTLE_CAP`] for a running
/// model load (D-04 S4) before the teardown, so a stopper's budget (`TimeoutStopSec`, `rv stop`)
/// must cover drain + that wait + teardown, not this window alone (D-04 S8).
pub const DRAIN_DEADLINE: Duration = Duration::from_secs(30);

/// Hard cap on engine teardown, even with several engines or a slow eviction (#5). Each
/// per-engine join is itself bounded (`model_manager` `JOIN_DEADLINE`); this caps the sequential
/// total. It does not include the wait for running loads before the teardown (D-04 S4,
/// [`LOAD_SETTLE_CAP`]), so a whole stop is drain + load wait + this, not a fixed 45 s.
///
/// D-04 S6: the floor of [`teardown_budget`], which grows with the number of engines.
pub const SHUTDOWN_DEADLINE: Duration = Duration::from_secs(10);

/// D-04 S6: the stop's teardown budget for `ready` engines. `ModelManager::shutdown` evicts them
/// one after another, each join bounded by the engine join deadline (8 s), so a flat
/// [`SHUTDOWN_DEADLINE`] could not cover two slow engines: `ready` x that deadline + 2 s, at least
/// [`SHUTDOWN_DEADLINE`]. Past it the remaining joins are left to their threads — still tracked
/// as `ffi_watch` teardowns, so `main` leaves without destructors (76 if one is still running).
#[must_use]
pub fn teardown_budget(ready: usize) -> Duration {
    let per_engine = crate::model_manager::JOIN_DEADLINE;
    let n = u32::try_from(ready).unwrap_or(u32::MAX);
    per_engine
        .saturating_mul(n)
        .saturating_add(Duration::from_secs(2))
        .max(SHUTDOWN_DEADLINE)
}

/// How long the serving stop waits for the axum server to close after a drain-deadline
/// teardown (the terminal events close the remaining streams first).
pub const SERVER_CLOSE_WAIT: Duration = Duration::from_secs(3);

/// D-04 S8: what is left of a stop from a step that may still wait `load_wait` for running loads
/// and then tears down `engines` engines (the serving close and the exit wait follow).
#[must_use]
pub fn stop_remaining(load_wait: Duration, engines: usize) -> Duration {
    load_wait + teardown_budget(engines) + SERVER_CLOSE_WAIT + EXIT_QUIESCE + EXIT_WAIT
}

/// D-04 S8: the longest a stop of a serving worker with `engines` engines (Ready plus loading)
/// takes before the process is gone: the drain runs inside the load wait (both start at the stop
/// request), then the teardown, the server close and the exit wait:
/// what a stopper's timeout must cover (`TimeoutStopSec`, `rv stop`).
#[must_use]
pub fn serving_stop_bound(engines: usize) -> Duration {
    stop_remaining(LOAD_SETTLE_CAP.max(DRAIN_DEADLINE), engines)
}

/// D-04 S8: the longest a stop during preload takes with `engines` engines (preloaded plus the
/// running one). `load_waits` = the running load has no load deadline, so the stop waits up to
/// [`LOAD_SETTLE_CAP`] for it; otherwise it is overdue (the supervisor only kills a preload past
/// its allowance) and the stop gives it [`PRELOAD_OVERDUE_GRACE`]. No drain: nothing is served yet.
#[must_use]
pub fn preload_stop_bound(engines: usize, load_waits: bool) -> Duration {
    let wait = if load_waits {
        LOAD_SETTLE_CAP
    } else {
        PRELOAD_OVERDUE_GRACE + LOAD_SETTLE_POLL
    };
    wait + teardown_budget(engines) + EXIT_QUIESCE + EXIT_WAIT
}

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

/// A configuration problem found at boot that did not stop the server (step 8, G-g1): logged as
/// WARN, kept for the admin server block, printed by `rv start` and `rv status`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct StartupWarning {
    /// Stable machine-readable code (`snake_case`).
    pub code: &'static str,
    /// One sentence for the operator.
    pub message: String,
}

static STARTUP_WARNINGS: std::sync::Mutex<Vec<StartupWarning>> = std::sync::Mutex::new(Vec::new());

/// Record a startup warning (the caller logs it with its structured fields).
pub fn push_startup_warning(code: &'static str, message: impl Into<String>) {
    if let Ok(mut w) = STARTUP_WARNINGS.lock() {
        w.push(StartupWarning {
            code,
            message: message.into(),
        });
    }
}

/// Every startup warning of this process, in order.
#[must_use]
pub fn startup_warnings() -> Vec<StartupWarning> {
    STARTUP_WARNINGS
        .lock()
        .map(|w| w.clone())
        .unwrap_or_default()
}

/// The GPU/NPU driver one device runs on (step 8), for the admin server block and `rv status`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct DeviceDriver {
    /// `OpenVINO` device name (`GPU.1`, `NPU`).
    pub device: String,
    /// Full device name.
    pub name: String,
    /// Driver version (`OpenCL` `CL_DRIVER_VERSION` for GPUs, `NPU_DRIVER_VERSION` for the NPU).
    pub driver_version: String,
}

/// Host platform facts collected once at boot.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct PlatformInfo {
    /// Driver per accelerator device.
    pub drivers: Vec<DeviceDriver>,
    /// Kernel release (Linux).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel: Option<String>,
}

static PLATFORM_INFO: OnceLock<PlatformInfo> = OnceLock::new();

/// The boot-time [`PlatformInfo`] (empty before `bootstrap` collected it).
#[must_use]
pub fn platform_info() -> PlatformInfo {
    PLATFORM_INFO.get().cloned().unwrap_or_default()
}

/// Match `OpenCL` GPUs to inventory devices by PCI address (falling back to the full name) and add
/// the NPU's driver version. Pure, for the unit test.
#[must_use]
pub fn match_drivers(
    devices: &[crate::device_inventory::DeviceInfo],
    ocl: &[crate::gpu_driver::OclGpu],
    npu_driver: Option<&str>,
) -> Vec<DeviceDriver> {
    use crate::device_inventory::DeviceKind;
    let mut out = Vec::new();
    for d in devices {
        match d.kind {
            DeviceKind::DiscreteGpu | DeviceKind::IntegratedGpu => {
                let hit = ocl
                    .iter()
                    .find(|g| g.pci.is_some() && g.pci == d.pdev)
                    .or_else(|| ocl.iter().find(|g| g.name == d.full_name));
                if let Some(g) = hit {
                    out.push(DeviceDriver {
                        device: d.name.clone(),
                        name: d.full_name.clone(),
                        driver_version: g.driver_version.clone(),
                    });
                }
            }
            DeviceKind::Npu => {
                if let Some(v) = npu_driver {
                    out.push(DeviceDriver {
                        device: d.name.clone(),
                        name: d.full_name.clone(),
                        driver_version: v.to_owned(),
                    });
                }
            }
            DeviceKind::Cpu | DeviceKind::Other => {}
        }
    }
    out
}

/// What to do about the configured global `device` (G-g2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlobalDeviceCheck {
    /// Enumerated.
    Ok,
    /// A HETERO list of enumerated devices.
    Hetero,
    /// Not enumerated; boots with a warning (it is only the memory budget anchor).
    Warn(String),
    /// Not enumerated with `default_placement: "device"`: no model without its own `device` could
    /// ever load, so the boot is refused (`EXIT_CONFIG_FATAL`).
    Fatal(String),
}

/// Pure decision for the global `device` (unit-tested with a fake inventory).
#[must_use]
pub fn global_device_check(inventory: &DeviceInventory, config: &Config) -> GlobalDeviceCheck {
    if inventory.get(&config.device).is_some() {
        return GlobalDeviceCheck::Ok;
    }
    if inventory.spec_known(&config.device) {
        return GlobalDeviceCheck::Hetero;
    }
    let enumerated: Vec<&str> = inventory
        .devices()
        .iter()
        .map(|d| d.name.as_str())
        .collect();
    if config.default_placement == crate::model_manager::config::DefaultPlacement::Device {
        GlobalDeviceCheck::Fatal(format!(
            "configured global `device` {:?} is not an enumerated OpenVINO device (enumerated: {}) \
             and default_placement is \"device\": no model without its own `device` could ever \
             load — set `device` to one of the enumerated devices or use default_placement \"auto\"",
            config.device,
            enumerated.join(", ")
        ))
    } else {
        GlobalDeviceCheck::Warn(format!(
            "configured global `device` {:?} is not an enumerated OpenVINO device (enumerated: {}); \
             the inference memory domain falls back to the device name and each model is still \
             placed on its own resolved device",
            config.device,
            enumerated.join(", ")
        ))
    }
}

/// Env var naming the supervisor's per-spawn progress file (first-GPU-load deadline, see
/// `SupervisorConfig::gpu_first_load_deadline_secs`).
pub const PROGRESS_FILE_ENV: &str = "RUSTEDVINO_PROGRESS_FILE";

/// A milestone of this worker's first GPU model load, for the supervisor's deadline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuLoadMilestone {
    /// The first GPU load started its engine construction, under this load deadline in seconds
    /// (`0` = none). D-04 S8: the supervisor's first-load deadline follows it.
    Started(u64),
    /// A GPU load returned, committed or failed: the device schedules work.
    Done,
}

/// G-1: a startup preload milestone for the supervisor's health deadline (see
/// `supervisor::health_deadline`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreloadNote {
    /// A preload model's load starts; the load deadline it runs under, in seconds (`0` = none).
    Started(u64),
    /// That load ended — committed, failed or timed out alike.
    Done,
}

/// Record a preload milestone in the supervisor's progress file (every call, unlike
/// [`note_gpu_load`]; best-effort; no-op when not supervised). One whole line per write, so the
/// supervisor never acts on a torn one (it ignores a last line without its newline).
pub fn note_preload(note: PreloadNote) {
    use std::io::Write as _;
    let Some(path) = std::env::var_os(PROGRESS_FILE_ENV) else {
        return;
    };
    let line = match note {
        PreloadNote::Started(secs) => format!("preload_started {secs}\n"),
        PreloadNote::Done => "preload_done\n".to_owned(),
    };
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = f.write_all(line.as_bytes());
    }
}

/// Record `m` in the supervisor's progress file (once per milestone, best-effort; no-op when
/// not supervised). Called from the load path.
pub fn note_gpu_load(m: GpuLoadMilestone) {
    use std::io::Write as _;
    static STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let (flag, line) = match m {
        GpuLoadMilestone::Started(secs) => (&STARTED, format!("gpu_load_started {secs}\n")),
        GpuLoadMilestone::Done => (&DONE, "gpu_load_done\n".to_owned()),
    };
    let Some(path) = std::env::var_os(PROGRESS_FILE_ENV) else {
        return;
    };
    if flag.swap(true, std::sync::atomic::Ordering::AcqRel) {
        return;
    }
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = f.write_all(line.as_bytes());
    }
}

/// D-04 S8: tell the supervisor how much longer this stop may take ([`stop_remaining`]), so its
/// hang kill does not SIGKILL a worker that is still inside its own stop wait or teardown
/// (best-effort; no-op when not supervised). Written where the stop starts a long step: the first
/// wait for a running load, and each teardown. The supervisor counts each from when it reads it.
pub fn note_stop_budget(budget: Duration) {
    use std::io::Write as _;
    let Some(path) = std::env::var_os(PROGRESS_FILE_ENV) else {
        return;
    };
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = f.write_all(format!("stop_budget {}\n", budget.as_secs()).as_bytes());
    }
}

/// Exit code of a worker that left while `OpenVINO` calls were still running (`ffi_watch`): it skipped
/// the C++ destructors, so device memory is released by the process exit, not by the pipelines. The
/// supervisor respawns it and counts it like a watchdog kill.
pub const EXIT_FFI_STALLED: u8 = 76;

/// How long [`exit_without_destructors`] waits before `_exit` so the GPU runtime's copy-engine
/// direct-submission ring has stopped (its controller stops it ~5 ms after the last submission;
/// 250 ms = 50× that). Inside the 5 s slack of the supervisor's watchdog kill grace.
pub const EXIT_QUIESCE: Duration = Duration::from_millis(250);

/// How long process exit waits for in-flight `OpenVINO` calls to return before leaving without
/// destructors. Part of the supervisor's watchdog kill grace (`supervisor::watchdog_kill_grace`).
pub const EXIT_WAIT: Duration = Duration::from_secs(3);

/// How `main` leaves the process (see [`exit_decision`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExitDecision {
    /// The process exit code.
    pub code: u8,
    /// Leave with [`exit_without_destructors`] instead of returning from `main`.
    pub skip_destructors: bool,
}

/// Set once [`bootstrap`] starts building the model manager (preload). From then on an engine
/// thread may exist, so `main` leaves without the C++ static destructors (D-04 S7).
static ENGINES_STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Records that model construction has started (see [`engines_started`]). [`bootstrap`] calls it;
/// another binary building a `ModelManager` itself must call it too.
pub fn mark_engines_started() {
    ENGINES_STARTED.store(true, std::sync::atomic::Ordering::Release);
}

/// Whether model construction has started in this process ([`mark_engines_started`]).
#[must_use]
pub fn engines_started() -> bool {
    ENGINES_STARTED.load(std::sync::atomic::Ordering::Acquire)
}

/// The one exit predicate (D-01, step 6a; D-04 S7). `normal_code` is what the run's result maps
/// to (0, 1, 75, 78). Returning from `main` runs the C++ static destructors, which corrupt the
/// heap when an engine thread is still inside a pipeline (D-01) and segfault on a poisoned GPU
/// context. So once `engines_started` — model construction began, whether the run then stopped
/// during preload, failed after a partial preload, or served and tore down cleanly — every exit
/// skips them (decided 2026-10-29: `_exit` after a clean teardown too). `teardown_incomplete` alone
/// also skips them. With any `OpenVINO` call still in flight a clean 0 becomes 76; other codes
/// are kept (more telling). A GPU-poisoned run that otherwise succeeded exits 75 (`main` maps an
/// error to 1/78 even then). Destructors run only for an exit before the mark — config errors
/// found before preload, a port clash, the device-query flags; a config-fatal raised during model
/// construction (e.g. an unregistered preload model) is after it and leaves via `_exit(78)`. Unix
/// only: see the non-Unix [`exit_without_destructors`].
#[must_use]
pub fn exit_decision(
    normal_code: u8,
    engines_started: bool,
    teardown_incomplete: bool,
    poison: bool,
    in_flight: usize,
) -> ExitDecision {
    if in_flight > 0 && !poison {
        return ExitDecision {
            code: if normal_code == 0 {
                EXIT_FFI_STALLED
            } else {
                normal_code
            },
            skip_destructors: true,
        };
    }
    ExitDecision {
        code: normal_code,
        skip_destructors: engines_started || teardown_incomplete || poison,
    }
}

/// Waits up to [`EXIT_WAIT`] for in-flight `OpenVINO` calls to return, logs the shutdown outcome
/// and returns how many are still in flight. Returns at once when none is. Blocking: `main` calls
/// it after `async_main` returned. `ok` = the run ended without an error; "shutdown complete" is
/// logged only when the server also really served (not for `--supervise`, `--print-default-*`
/// or any other early exit, whose stdout must stay clean).
#[must_use]
pub fn settle_ffi_calls(ok: bool) -> usize {
    let served = ok && SERVING.load(std::sync::atomic::Ordering::Acquire);
    let watch = crate::ffi_watch::global();
    let n = watch.wait_idle(EXIT_WAIT, Duration::from_millis(50));
    if n > 0 {
        let kinds: Vec<String> = watch
            .in_flight_by_kind()
            .iter()
            .map(|(k, c)| format!("{} {c}", k.label()))
            .collect();
        tracing::error!(
            in_flight = n,
            kinds = %kinds.join(", "),
            calls = ?watch.in_flight_models(),
            exit_code = EXIT_FFI_STALLED,
            "{n} OpenVINO calls still in flight after {}s — leaving without destructors; \
             process exit releases device memory",
            EXIT_WAIT.as_secs()
        );
    } else if served && teardown_incomplete() {
        tracing::warn!(
            "shutdown finished without a full engine teardown (an engine thread did not exit \
             cleanly, or the GPU context is poisoned) — leaving without destructors; process exit \
             releases device memory"
        );
    } else if served {
        tracing::info!("shutdown complete — engines drained and VRAM released");
    }
    n
}

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

/// Set by the supervisor on the worker it spawns: a fresh random token per spawn that admits the
/// supervisor's probe of [`crate::WATCHDOG_PATH`] whatever the admin-key setup is. Without it the
/// probe met the admin gate (401 with admin keys, 503 with no keys file) and read that as "no
/// hang", so the generation watchdog never fired on a keyed server (live sweep 2026-10-27).
pub const WATCHDOG_TOKEN_ENV: &str = "RUSTEDVINO_WATCHDOG_TOKEN";

/// The supervisor's watchdog token from [`WATCHDOG_TOKEN_ENV`], if this worker was given one.
fn watchdog_token_from_env() -> Option<Arc<str>> {
    std::env::var(WATCHDOG_TOKEN_ENV)
        .ok()
        .filter(|t| !t.is_empty())
        .map(Arc::from)
}

/// Wakes every [`shutdown_signal`] waiter without an OS signal (the same drain
/// path as SIGTERM). `shutdown_signal` is awaited by `serve` AND by background
/// loops (cache sweep, KV monitor), so the request must reach all of them:
/// `notify_one` woke only one (usually a background loop, which then quietly
/// returned and left `serve` running - review finding). The request is a sticky
/// flag plus `notify_waiters`; see [`wait_for_flag`].
static SHUTDOWN_REQUEST: tokio::sync::Notify = tokio::sync::Notify::const_new();

/// Set once, never cleared: a shutdown has been requested.
///
/// Process-wide and never reset: a unit test that raises it ([`request_shutdown`], or a
/// [`shutdown_signal`] that sees a signal) makes every later `ModelManager` load in the same test
/// binary fail with `ShuttingDown`. Tests use `ModelManager::begin_shutdown` instead.
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
    STOP_REQUESTED_AT.get_or_init(Instant::now);
    raise_flag(&SHUTDOWN_REQUEST, &SHUTDOWN_REQUESTED);
}

/// When the first stop request arrived — the one start of every stop-path wait (D-04 S3/S4).
static STOP_REQUESTED_AT: OnceLock<Instant> = OnceLock::new();

/// When the stop was requested ([`request_shutdown`]); now, if it was not.
#[must_use]
pub fn stop_requested_at() -> Instant {
    STOP_REQUESTED_AT
        .get()
        .copied()
        .unwrap_or_else(Instant::now)
}

/// The longest a stop waits for running model loads before it gives up and leaves without
/// destructors (D-04 S3/S4, decided 2026-10-29: 600 s). Counted from [`stop_requested_at`], once
/// for the whole stop. A load past its own `load_timeout_secs` ends the wait earlier.
pub const LOAD_SETTLE_CAP: Duration = Duration::from_mins(10);

/// How often the stop-path wait re-checks the loads.
pub const LOAD_SETTLE_POLL: Duration = Duration::from_millis(250);

/// How often the stop-path wait logs that it is still waiting.
pub const LOAD_SETTLE_LOG_EVERY: Duration = Duration::from_secs(10);

/// One step of the stop-path wait for model loads (D-04 S3/S4); see [`settle_step`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum Settle {
    /// Nothing is loading or evicting: tear down now.
    Done,
    /// A load is still running inside its allowance: keep waiting.
    Wait,
    /// A load is past its allowance — the device has stalled; waiting longer does not help.
    Overdue,
    /// [`LOAD_SETTLE_CAP`] passed.
    Cap,
}

/// The stop-path wait rule (D-04 S5): wait while something is `unsettled` (a load or eviction
/// running), nothing is past its allowance (`overdue`, from `ffi_watch::stop_wait_overdue`: a
/// load still inside its allowance counts as not overdue even beside one that stalled earlier —
/// Fable S8 #10) and `waited` is under `cap`.
pub fn settle_step(
    unsettled: bool,
    overdue: Option<Duration>,
    waited: Duration,
    cap: Duration,
) -> Settle {
    if !unsettled {
        Settle::Done
    } else if overdue.is_some_and(|d| !d.is_zero()) {
        Settle::Overdue
    } else if waited >= cap {
        Settle::Cap
    } else {
        Settle::Wait
    }
}

/// The message of the stop path's "still waiting" line, which `rv stop`/`rv start` follow
/// (pinned by `tests/rv_stop_contract.rs`).
pub const STOP_WAIT_LOG: &str = "stop requested while a model load is running - waiting for it \
                                 to finish so the GPU is released cleanly";

/// The message of the preload line (with a `load_timeout_s` field) that `rv start` follows to
/// extend its wait (pinned by `tests/rv_stop_contract.rs`).
pub const PRELOAD_LOG: &str = "preloading model - the port opens once the preload is done";

/// The "still waiting" line of the stop path ([`STOP_WAIT_LOG`]).
pub fn log_stop_wait(models: &[String], waited: Duration) {
    tracing::warn!(
        models = ?models,
        waited_s = waited.as_secs(),
        cap_s = LOAD_SETTLE_CAP.as_secs(),
        "{STOP_WAIT_LOG}"
    );
}

/// The end of a stop-path wait that did not settle ([`Settle::Overdue`] or [`Settle::Cap`]).
pub fn log_settle_given_up(outcome: Settle, models: &[String], waited: Duration) {
    let why = if outcome == Settle::Overdue {
        "a load is past its load_timeout_secs - the device has stalled"
    } else {
        "the stop wait cap passed"
    };
    tracing::error!(
        models = ?models,
        waited_s = waited.as_secs(),
        cap_s = LOAD_SETTLE_CAP.as_secs(),
        why,
        "stop: not waiting any longer for the model load; leaving without destructors"
    );
}

/// The first give-up of the stop-path wait ([`Settle::Overdue`]/[`Settle::Cap`]), if any. A later
/// wait in the same stop returns it at once instead of waiting (and logging) again.
static STOP_WAIT_GAVE_UP: OnceLock<Settle> = OnceLock::new();

/// D-04 S4: before the engines are torn down, wait for running loads and evictions to finish
/// ([`ModelManager::wait_for_loads_to_settle`] under the stop-path rule).
///
/// A give-up marks the teardown incomplete ([`TEARDOWN_INCOMPLETE`]): the load it stopped
/// waiting for may return during `main`'s exit wait, and its result is then destroyed on the
/// engine thread — returning through the C++ destructors at that moment is the D-01 race, so
/// `main` must leave without them even when nothing is in flight any more. On a poisoned GPU
/// context there is nothing to wait for: the teardown is skipped anyway (`ModelManager::shutdown`).
async fn settle_loads(model_manager: &ModelManager) -> Settle {
    if let Some(&outcome) = STOP_WAIT_GAVE_UP.get() {
        return outcome;
    }
    if model_manager.is_gpu_poisoned() {
        return Settle::Done;
    }
    let outcome = model_manager
        .wait_for_loads_to_settle(stop_requested_at(), LOAD_SETTLE_CAP, || {
            crate::ffi_watch::global().stop_wait_overdue()
        })
        .await;
    if outcome != Settle::Done {
        let _ = STOP_WAIT_GAVE_UP.set(outcome);
        TEARDOWN_INCOMPLETE.store(true, std::sync::atomic::Ordering::SeqCst);
    }
    outcome
}

/// How long [`stop_during_preload`] keeps awaiting the construction after `ffi_watch` first
/// reports an overdue call. A preload past its `load_timeout_secs` returns by itself (the load is
/// booked as stalled) at about the same moment the call turns overdue; this grace lets that
/// return win, so the engines preloaded so far are torn down instead of left live (D-04 review).
pub const PRELOAD_OVERDUE_GRACE: Duration = Duration::from_secs(5);

/// D-04 S3: a stop arrived while `construct` (the model manager, preloading) was still running.
/// Keep awaiting it — S2 skips the remaining preload entries, so this is the current load —
/// then wait for loads to settle, tear the engines down, and report a stop before ready.
/// On [`Settle::Overdue`]/[`Settle::Cap`] the construction is given up but **not dropped**:
/// dropping would destroy the engines preloaded so far while the load still runs inside
/// `OpenVINO` (the D-01 race); `main` then leaves without destructors.
async fn stop_during_preload<F>(
    mut construct: std::pin::Pin<Box<F>>,
    preload: usize,
) -> anyhow::Error
where
    F: std::future::Future<Output = anyhow::Result<ModelManager>>,
{
    let since = stop_requested_at();
    let mut last_log: Option<Instant> = None;
    let mut overdue_since: Option<Instant> = None;
    let mut declared = false;
    let constructed = loop {
        tokio::select! {
            r = &mut construct => break r,
            () = tokio::time::sleep(LOAD_SETTLE_POLL) => {
                let watch = crate::ffi_watch::global();
                let models = watch.in_flight_models();
                // `unsettled` is always true here: the construction has not returned.
                let step = settle_step(true, watch.stop_wait_overdue(), since.elapsed(), LOAD_SETTLE_CAP);
                let give_up = match step {
                    Settle::Cap => true,
                    Settle::Overdue => {
                        overdue_since.get_or_insert_with(Instant::now).elapsed()
                            >= PRELOAD_OVERDUE_GRACE
                    }
                    Settle::Wait | Settle::Done => false,
                };
                if give_up {
                    log_settle_given_up(step, &models, since.elapsed());
                    let _ = STOP_WAIT_GAVE_UP.set(step);
                    std::mem::forget(construct);
                    return anyhow::Error::new(ShutdownDuringStartup);
                }
                if step == Settle::Wait && !declared {
                    // D-04 S8: the supervisor waits this out before any SIGKILL. At most every
                    // preload entry becomes an engine.
                    declared = true;
                    let left = LOAD_SETTLE_CAP.saturating_sub(since.elapsed());
                    note_stop_budget(stop_remaining(left, preload));
                }
                if step == Settle::Wait
                    && last_log.is_none_or(|t| t.elapsed() >= LOAD_SETTLE_LOG_EVERY)
                {
                    log_stop_wait(&models, since.elapsed());
                    last_log = Some(Instant::now());
                }
            }
        }
    };
    match constructed {
        Ok(model_manager) => {
            let settled = settle_loads(&model_manager).await;
            teardown_engines(&model_manager).await;
            // `settle_ffi_calls` logs no outcome before serving; say how this stop ended here.
            if settled == Settle::Done && !teardown_incomplete() {
                tracing::info!(
                    "stopped before serving - model loads finished, engines torn down and VRAM \
                     released"
                );
            }
        }
        Err(e) => tracing::warn!(error = %e, "model manager failed after the stop request"),
    }
    anyhow::Error::new(ShutdownDuringStartup)
}

/// Whether a stop was requested: a signal, or [`request_shutdown`]. With the dedicated signal
/// thread the flag is raised the moment the signal arrives; with tokio's fallback handler, once
/// any [`shutdown_signal`] waiter sees it (D-04 S2: `ModelManager` refuses new loads from then).
#[must_use]
pub fn shutdown_requested() -> bool {
    SHUTDOWN_REQUESTED.load(std::sync::atomic::Ordering::Acquire)
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
        // Through `request_shutdown`, so the stop-path wait has its one start time (D-04).
        request_shutdown();
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
/// thread teardown. Since D-04 S7 `main` leaves through it on every exit once model construction
/// started ([`exit_decision`]) — clean stops included — besides the GPU-poisoned exit (see [`serve`]):
/// libc `exit()` would run the C++ static destructors, which corrupt the heap when an engine thread
/// is still inside a pipeline (D-01) and segfault on a poisoned context.
///
/// Waits [`EXIT_QUIESCE`] first (B60, 2026-10-27): the GPU runtime's direct-submission ring on the
/// copy engine keeps running ~5 ms after the last submission, and a process that exits inside that
/// window faults the engine at teardown — the next process's first GPU work can then hang until a
/// GT reset. With the wait: 0 faults in 3 reps vs 3/3 without
/// (Arc Pro B60, xe driver).
#[cfg(unix)]
pub fn exit_without_destructors(code: u8) -> ! {
    use std::io::Write as _;
    std::thread::sleep(EXIT_QUIESCE);
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    // SAFETY: `_exit` only reads an integer and ends the process; it is async-signal-safe and touches no
    // Rust state. Skipping destructors is the intent (D-01, D-04 S7): they could race an engine thread
    // still inside a pipeline, or run pipeline teardown on a poisoned context.
    unsafe { libc::_exit(i32::from(code)) }
}

/// Non-Unix fallback: a normal exit. `process::exit` runs the C++ static destructors, so on Windows
/// the D-01 / D-04 S7 protection does not exist yet — every exit there behaves like returning from
/// `main` (known issue, D-04 S9).
#[cfg(not(unix))]
pub fn exit_without_destructors(code: u8) -> ! {
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
    /// The port, bound before preload but not yet listening — [`serve`] calls `listen` on it.
    pub socket: tokio::net::TcpSocket,
    pub model_manager: Arc<ModelManager>,
}

/// Binds `addr` without listening on it (D-04 S1). Called before preload so a port already in
/// use fails while no engine exists: the same clash after preload returned into the C++
/// destructors with live engines (the QA-084-MR-1 crash class). Not listening keeps today's
/// behaviour during preload — a connect is refused at once instead of hanging in the backlog.
///
/// Socket options match `tokio::net::TcpListener::bind`: `SO_REUSEADDR` on Unix only (a quick
/// restart must not fail on `TIME_WAIT`; on Windows the option would allow port stealing).
/// Limitation: on Linux two bound-but-not-listening sockets with `SO_REUSEADDR` may share a port,
/// so this catches a server that is already *listening*; two instances preloading at the same
/// moment still clash only at `listen`, which [`serve`] handles by tearing down first.
///
/// # Errors
///
/// Returns an error if the socket cannot be created or `addr` cannot be bound.
pub fn reserve_port(addr: SocketAddr) -> std::io::Result<tokio::net::TcpSocket> {
    let socket = if addr.is_ipv4() {
        tokio::net::TcpSocket::new_v4()?
    } else {
        tokio::net::TcpSocket::new_v6()?
    };
    #[cfg(unix)]
    socket.set_reuseaddr(true)?;
    socket.bind(addr)?;
    Ok(socket)
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
        push_startup_warning(
            "ov_cache_dir_unwritable",
            format!(
                "ov_cache_dir {dir} is not writable ({e}): every model is recompiled on every load"
            ),
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
        push_startup_warning(
            "dynamic_kv_with_prefix_caching",
            "cache_size_gb 0.0 with enable_prefix_caching: retained prefix blocks can grow outside \
             the memory budget",
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
        push_startup_warning(
            "cache_manifest_unwritable",
            format!(
                "{} is not writable ({e}): model hashes and provenance are not saved",
                root.display()
            ),
        );
    }
    if let Err(e) = probe_file_writable(&config_path) {
        tracing::warn!(
            path = %config_path.display(), error = %e,
            "config file is not writable — `rv patch` and admin config edits will fail \
             (check the file's ACL; staging from another OS can leave it read-only)"
        );
        push_startup_warning(
            "config_unwritable",
            format!(
                "{} is not writable ({e}): rv patch and admin config edits will fail",
                config_path.display()
            ),
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
    match global_device_check(&inventory, &config) {
        GlobalDeviceCheck::Ok => {
            tracing::info!(device = %config.device, "budget-anchor device enumerated");
        }
        // A HETERO list whose members all exist: usable for placement, but the memory domain
        // lookup is by plain device name and falls back to the literal spec.
        GlobalDeviceCheck::Hetero => tracing::info!(
            device = %config.device,
            "global device is a HETERO list of enumerated devices; its memory domain falls back to \
             the literal name"
        ),
        GlobalDeviceCheck::Warn(msg) => {
            tracing::warn!(device = %config.device, "{msg}");
            push_startup_warning("global_device_unknown", msg);
        }
        // G-g2: with default_placement "device" nothing without its own `device` could ever load.
        GlobalDeviceCheck::Fatal(msg) => {
            return Err(config_fatal(&anyhow::anyhow!(msg)));
        }
    }

    // Step 8: which GPU/NPU driver this process runs on (OpenVINO does not report it).
    let npu_driver = inventory
        .devices()
        .iter()
        .find(|d| d.kind == crate::device_inventory::DeviceKind::Npu)
        .and_then(|d| crate::ov_cb::device_property(&d.name, "NPU_DRIVER_VERSION"));
    let platform = PlatformInfo {
        drivers: match_drivers(
            inventory.devices(),
            &crate::gpu_driver::opencl_gpus(),
            npu_driver.as_deref(),
        ),
        kernel: crate::gpu_driver::kernel_release(),
    };
    for d in &platform.drivers {
        tracing::info!(
            device = %d.device,
            name = %d.name,
            driver = %d.driver_version,
            kernel = platform.kernel.as_deref().unwrap_or("?"),
            "device driver"
        );
    }
    let _ = PLATFORM_INFO.set(platform);

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
    let socket = reserve_port(addr)
        .map_err(|e| anyhow::anyhow!("cannot bind {addr} (is another server running?): {e}"))?;

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
    // D-04 S3 (reverses D-02's "SIGTERM honoured at once"): a preload can sit inside one GPU
    // compile for minutes, and a blocking OpenVINO call cannot be cancelled. Abandoning it left
    // the process to `_exit` with a live GPU context (the xe wedge trigger), so a stop now waits
    // for the current load (`stop_during_preload`) and tears down before leaving. Boxed so the
    // give-up path can keep the future alive instead of dropping it.
    // D-04 S7: from here on an engine may exist — `main` leaves without destructors.
    // Do not add a fallible step (`?`) between this mark and `serve` that can return while engines
    // are Ready: `main` would skip the destructors but nothing would tear the engines down or track
    // their threads (S7 review, item 2). `new_production` fails only before its preload, and `serve`
    // tears down before returning an error.
    mark_engines_started();
    let preload_len = config.preload.len();
    let mut construct = Box::pin(ModelManager::new_production(config, Arc::clone(&inventory)));
    let constructed = tokio::select! {
        r = &mut construct => r?,
        () = shutdown_signal() => {
            tracing::warn!(
                "shutdown signal received while models were still loading - finishing the \
                 current load, then stopping (nothing was serving yet)"
            );
            return Err(stop_during_preload(construct, preload_len).await);
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
        .with_admin_locked(admin_locked)
        .with_watchdog_token(watchdog_token_from_env());
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
        socket,
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
    // A signal seen here raises the shared flag too, so `shutdown_requested` holds on the
    // tokio-handler fallback path as well (idempotent; wakes the other waiters).
    tokio::select! {
        () = ctrl_c => request_shutdown(),
        () = sigterm => request_shutdown(),
        () = wait_for_flag(&SHUTDOWN_REQUEST, &SHUTDOWN_REQUESTED) => {},
    }
}

/// Tear down all engines, bounded by [`teardown_budget`] (#5, D-04 S6). Each per-engine join
/// inside `shutdown()` is itself bounded; this caps the sequential total. Callers wait for
/// running loads first ([`settle_loads`]); that wait is outside this bound.
async fn teardown_engines(model_manager: &ModelManager) {
    // `shutdown()` commits a stalled load that has returned and then evicts it; count it too
    // (D-04 S6 review: a late result arriving after a stop-wait give-up). Not on a poisoned
    // context: `shutdown()` skips it there, and committing could run a KV warm-up on it.
    if !model_manager.is_gpu_poisoned() {
        model_manager.reap_stalled_loads();
    }
    let ready = model_manager.ready_count();
    let budget = teardown_budget(ready);
    note_stop_budget(stop_remaining(Duration::ZERO, ready));
    match tokio::time::timeout(budget, model_manager.shutdown()).await {
        Err(_) => {
            TEARDOWN_INCOMPLETE.store(true, std::sync::atomic::Ordering::SeqCst);
            tracing::error!(
                budget_s = budget.as_secs(),
                "engine teardown exceeded its budget — exiting anyway"
            );
        }
        // A join that timed out inside the deadline leaves the thread in the plugin just the same
        // (Test 5, 2026-10-27: the VLM eviction join expired, the whole shutdown still fit).
        Ok(report) if !report.unclean.is_empty() => {
            TEARDOWN_INCOMPLETE.store(true, std::sync::atomic::Ordering::SeqCst);
            tracing::error!(
                models = ?report.unclean,
                "engine teardown: an engine thread did not exit cleanly — it may still be inside \
                 OpenVINO"
            );
        }
        Ok(_) => {}
    }
    // A poisoned context skips the engine teardown (`ModelManager::shutdown`): nothing was torn
    // down, and returning from `main` would run the pipeline destructors on that context (the
    // segfault the poison exit avoids). Same predicate as a failed join: leave without destructors.
    if model_manager.is_gpu_poisoned() {
        TEARDOWN_INCOMPLETE.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Set when an engine teardown hit [`SHUTDOWN_DEADLINE`], an engine join failed inside it, the
/// teardown was skipped on a poisoned GPU context, or the stop gave up waiting for a model load
/// ([`settle_loads`], D-04): an engine thread may still be inside the pipeline (or the pipelines
/// were never torn down, or a late load result is being destroyed), so `main` must leave with
/// [`exit_without_destructors`] instead of returning (D-01).
static TEARDOWN_INCOMPLETE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Whether an engine teardown ran out of time (see [`TEARDOWN_INCOMPLETE`]).
#[must_use]
pub fn teardown_incomplete() -> bool {
    TEARDOWN_INCOMPLETE.load(std::sync::atomic::Ordering::SeqCst)
}

/// Serves `app` on `socket` (bound by [`reserve_port`] in [`bootstrap`]) with graceful shutdown
/// (T6.1): stop accepting new
/// connections on SIGTERM/ctrl-C, give in-flight requests up to
/// [`DRAIN_DEADLINE`] to finish naturally, then tear down the engines — which
/// terminates any remaining stream with a clean SSE error + `[DONE]` (T1.3),
/// joins the engine threads, and frees VRAM via `Drop` before returning.
///
/// # Errors
///
/// Returns an error if `socket` can't listen, or if the server task itself
/// ends with an error (both fall through to engine teardown first — VRAM
/// release is never skipped on a server error).
pub async fn serve(
    app: axum::Router,
    addr: SocketAddr,
    socket: tokio::net::TcpSocket,
    model_manager: &ModelManager,
) -> anyhow::Result<()> {
    // Same backlog as `TcpListener::bind`. A failure here comes after preload (see
    // `reserve_port`'s limitation): tear the engines down before returning, never into the
    // destructors with live engines (D-04 S1).
    let listener = match socket.listen(1024) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(%addr, error = %e, "cannot listen — tearing down the engines");
            teardown_engines(model_manager).await;
            return Err(anyhow::anyhow!("cannot listen on {addr}: {e}"));
        }
    };
    tracing::info!(%addr, "RustedVINO listening");
    #[cfg(feature = "fault-injection")]
    crate::fault_injection::log_arming();
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
                "drain deadline expired — terminating remaining requests once running model loads finish"
            );
            // A give-up is recorded inside (`TEARDOWN_INCOMPLETE`); nothing to decide here.
            let _ = settle_loads(model_manager).await;
            teardown_engines(model_manager).await;
            let _ = tokio::time::timeout(SERVER_CLOSE_WAIT, server.as_mut()).await;
        }
    }

    // Idempotent on the deadline path (everything is already evicted); on the
    // normal path this is where engines drop and VRAM is released.
    // The outcome line ("shutdown complete …" or the in-flight warning) is logged by
    // `settle_ffi_calls`, after the exit wait, where the exit decision is made.
    // D-04 S4: first let running loads and evictions finish (outside SHUTDOWN_DEADLINE); returns
    // at once when nothing is running, e.g. after the deadline branch already waited.
    let _ = settle_loads(model_manager).await;
    teardown_engines(model_manager).await;

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

    // ---- step 8: global device check (G-g2), driver matching ----------------

    fn dev(
        name: &str,
        full: &str,
        kind: crate::device_inventory::DeviceKind,
        pdev: Option<&str>,
    ) -> crate::device_inventory::DeviceInfo {
        crate::device_inventory::DeviceInfo {
            name: name.to_owned(),
            full_name: full.to_owned(),
            architecture: None,
            kind,
            tier: crate::device_inventory::DeviceTier::Fallback,
            domain_id: name.to_owned(),
            total_mem_gb: None,
            pdev: pdev.map(str::to_owned),
        }
    }

    #[test]
    fn global_device_unknown_warns_or_is_fatal_with_device_placement() {
        use crate::device_inventory::DeviceKind;
        let inv = DeviceInventory::from_devices(vec![
            dev("CPU", "cpu", DeviceKind::Cpu, None),
            dev("GPU.0", "igpu", DeviceKind::IntegratedGpu, None),
        ]);
        let mut cfg = minimal_config("");
        cfg.device = "GPU.0".to_owned();
        assert_eq!(global_device_check(&inv, &cfg), GlobalDeviceCheck::Ok);
        cfg.device = "GPU.7".to_owned();
        assert!(matches!(
            global_device_check(&inv, &cfg),
            GlobalDeviceCheck::Warn(m) if m.contains("GPU.7") && m.contains("CPU, GPU.0")
        ));
        cfg.default_placement = crate::model_manager::config::DefaultPlacement::Device;
        assert!(matches!(
            global_device_check(&inv, &cfg),
            GlobalDeviceCheck::Fatal(m) if m.contains("default_placement")
        ));
    }

    #[test]
    fn drivers_match_by_pci_then_name_and_include_the_npu() {
        use crate::device_inventory::DeviceKind;
        use crate::gpu_driver::OclGpu;
        let devices = vec![
            dev("CPU", "cpu", DeviceKind::Cpu, None),
            dev(
                "GPU.0",
                "Intel(R) UHD Graphics 770",
                DeviceKind::IntegratedGpu,
                None,
            ),
            dev(
                "GPU.1",
                "Intel(R) Graphics [0xe211]",
                DeviceKind::DiscreteGpu,
                Some("0000:03:00.0"),
            ),
            dev("NPU", "Intel(R) AI Boost", DeviceKind::Npu, None),
        ];
        let ocl = vec![
            OclGpu {
                name: "Intel(R) Arc(TM) Pro B60 Graphics".into(),
                driver_version: "26.35.39758".into(),
                pci: Some("0000:03:00.0".into()),
            },
            OclGpu {
                name: "Intel(R) UHD Graphics 770".into(),
                driver_version: "26.35.39758".into(),
                pci: None,
            },
        ];
        let got = match_drivers(&devices, &ocl, Some("1.10.0"));
        let pairs: Vec<(&str, &str)> = got
            .iter()
            .map(|d| (d.device.as_str(), d.driver_version.as_str()))
            .collect();
        assert_eq!(
            pairs,
            vec![
                ("GPU.0", "26.35.39758"),
                ("GPU.1", "26.35.39758"),
                ("NPU", "1.10.0")
            ]
        );
    }

    // ---- exit_decision (D-01, step 6a) ------------------------------------

    /// The exit predicate table: (normal code, abandoned, teardown incomplete, poison, in flight)
    /// → (code, skip destructors).
    #[test]
    fn exit_decision_table() {
        // (normal, engines_started, incomplete, poison, in_flight) -> (code, skip_destructors)
        let rows: &[(u8, bool, bool, bool, usize, u8, bool)] = &[
            // before any engine (config error, port clash, early stop): normal return
            (0, false, false, false, 0, 0, false),
            (1, false, false, false, 0, 1, false),
            (
                EXIT_CONFIG_FATAL,
                false,
                false,
                false,
                0,
                EXIT_CONFIG_FATAL,
                false,
            ),
            // D-04 S7: once engines started, every exit is `_exit` with its own code —
            // clean stop, stop during preload, an error after a partial preload (review 2a)
            (0, true, false, false, 0, 0, true),
            (1, true, false, false, 0, 1, true),
            (
                EXIT_CONFIG_FATAL,
                true,
                false,
                false,
                0,
                EXIT_CONFIG_FATAL,
                true,
            ),
            // failed engine join / stop-wait give-up, nothing in flight: _exit 0 (decided 2026-10-29)
            (0, true, true, false, 0, 0, true),
            (0, false, true, false, 0, 0, true),
            // calls in flight: a clean 0 becomes 76, other codes are kept; always _exit
            (0, true, false, false, 2, EXIT_FFI_STALLED, true),
            (0, true, true, false, 1, EXIT_FFI_STALLED, true),
            (1, true, false, false, 1, 1, true),
            (
                EXIT_CONFIG_FATAL,
                false,
                false,
                false,
                1,
                EXIT_CONFIG_FATAL,
                true,
            ),
            // GPU poison keeps 75 and never returns through the destructors
            (
                EXIT_GPU_POISONED,
                true,
                true,
                true,
                0,
                EXIT_GPU_POISONED,
                true,
            ),
            (
                EXIT_GPU_POISONED,
                true,
                false,
                true,
                3,
                EXIT_GPU_POISONED,
                true,
            ),
        ];
        for &(normal, started, incomplete, poison, in_flight, code, skip) in rows {
            assert_eq!(
                exit_decision(normal, started, incomplete, poison, in_flight),
                ExitDecision {
                    code,
                    skip_destructors: skip
                },
                "row ({normal}, {started}, {incomplete}, {poison}, {in_flight})"
            );
        }
    }

    /// The new exit code stays distinct from the existing ones.
    #[test]
    fn exit_ffi_stalled_is_76_and_distinct() {
        assert_eq!(EXIT_FFI_STALLED, 76);
        assert_ne!(EXIT_FFI_STALLED, EXIT_GPU_POISONED);
        assert_ne!(EXIT_FFI_STALLED, EXIT_CONFIG_FATAL);
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

    /// D-04 S6: the teardown budget grows by one engine join deadline per Ready engine and never
    /// drops below `SHUTDOWN_DEADLINE`.
    #[test]
    fn teardown_budget_scales_with_the_engines() {
        let join = crate::model_manager::JOIN_DEADLINE;
        assert_eq!(teardown_budget(0), SHUTDOWN_DEADLINE);
        assert_eq!(
            teardown_budget(1),
            SHUTDOWN_DEADLINE.max(join + Duration::from_secs(2))
        );
        assert_eq!(teardown_budget(6), join * 6 + Duration::from_secs(2));
        assert!(teardown_budget(2) >= join * 2, "two slow engines both fit");
    }

    /// D-04 S5: the stop-path wait rule. Nothing running ends the wait whatever else holds; an
    /// overdue call (device stalled) ends it before the cap; a zero overdue (every call inside its
    /// allowance, or an unbounded `load_timeout_secs = 0` load) keeps waiting until the cap.
    #[test]
    fn settle_step_table() {
        let s = Duration::from_secs;
        let cap = s(600);
        let rows = [
            (false, None, s(0), Settle::Done),
            (false, Some(s(5)), s(900), Settle::Done),
            (true, None, s(0), Settle::Wait),
            (true, Some(Duration::ZERO), s(599), Settle::Wait),
            (true, Some(Duration::ZERO), s(600), Settle::Cap),
            (true, None, s(601), Settle::Cap),
            (true, Some(Duration::from_millis(1)), s(1), Settle::Overdue),
            (true, Some(s(5)), s(900), Settle::Overdue),
        ];
        for (unsettled, overdue, waited, want) in rows {
            assert_eq!(
                settle_step(unsettled, overdue, waited, cap),
                want,
                "unsettled={unsettled} overdue={overdue:?} waited={waited:?}"
            );
        }
    }

    /// D-04 S1: a port another server already listens on fails at `reserve_port` (before preload).
    /// The occupant must be *listening* — two bound-only sockets may share a port on Linux.
    #[tokio::test]
    async fn reserve_port_fails_on_a_listening_port_and_refuses_connects_until_listen() {
        let occupant = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let taken = occupant.local_addr().unwrap();
        assert!(
            reserve_port(taken).is_err(),
            "a listening port must not be reservable"
        );

        let socket = reserve_port("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = socket.local_addr().unwrap();
        assert!(
            tokio::net::TcpStream::connect(addr).await.is_err(),
            "reserved but not listening: a connect is refused, not queued"
        );
        let listener = socket.listen(16).unwrap();
        let (client, accepted) =
            tokio::join!(tokio::net::TcpStream::connect(addr), listener.accept());
        assert!(client.is_ok() && accepted.is_ok(), "serves once listening");
    }
}
