// ============================================================
// src/supervisor.rs — crash auto-restart wrapper (`--supervise`, Unix only)
// ============================================================
// A thin process-lifecycle wrapper: spawns the real server as a child,
// watches it, and respawns with backoff+cap on crash. Never touches
// OpenVINO/GPU state itself — that is entirely the worker's job, run in a
// fresh process each time. Linux-only by design (see `src/lib.rs`'s
// `#[cfg(unix)]` module gate): the Windows build never compiles this module
// at all, so it carries zero risk to the Windows cross-build.
//
// See the project's internal engineering log for why this design
// (vs. systemd `Restart=` or a shell supervisor loop) was chosen, and why a
// worker crashing with SIGSEGV/SIGABRT is safe to catch this way even inside
// a `systemd-run --user` scope (verified live, both signals, both plain and
// scoped).
// ============================================================

use std::net::SocketAddr;
use std::os::unix::process::ExitStatusExt;
use std::path::Path;
use std::process::ExitStatus;
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::process::{Child, Command};
use tokio::signal::unix::{Signal, SignalKind, signal};

use crate::model_manager::{Config, SupervisorConfig};

/// Entry point for `rustedvino --supervise` (dispatched from `main.rs`,
/// which resolves `config_path` the same way it does for a normal worker
/// startup). Loads the config independently of the worker — which loads it
/// again in its own process — purely to read `bind_addr`/`port` (for health
/// polling) and the optional `supervisor` tuning block.
///
/// # Errors
/// Propagates config-load failure, `current_exe()` failure, or a hard
/// give-up after exhausting [`SupervisorConfig::max_restarts`].
pub async fn run(config_path: &Path) -> anyhow::Result<()> {
    // MR-5c: a config that does not load is the worker's 78 too, as with a plain start (the
    // supervisor validates it before any worker is spawned).
    let config = Config::load(config_path).map_err(|e| crate::startup::config_fatal(&e))?;
    let health_addr = config.validated_bind()?;
    let sup_cfg = config.supervisor.clone();

    let exe = std::env::current_exe()?;
    let worker_args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| a != "--supervise")
        .collect();

    let probe = ProbeCmd {
        exe: exe.clone(),
        args: vec!["--gpu-probe".to_owned()],
        settle: PROBE_RESET_SETTLE,
    };
    run_supervised_with_probe(&exe, &worker_args, health_addr, &sup_cfg, Some(&probe)).await
}

/// [`gpu_readiness`] when a no-teardown exit armed it and a probe is set; `true` = SIGTERM came
/// in while probing (stop, no worker).
async fn probe_says_shut_down(
    armed: bool,
    probe: Option<&ProbeCmd>,
    cfg: &SupervisorConfig,
    sigterm: &mut Signal,
) -> bool {
    match probe.filter(|_| armed) {
        Some(p) => gpu_readiness(p, cfg, sigterm).await == Readiness::Shutdown,
        None => false,
    }
}

/// Probe attempts after a no-teardown exit before the worker is started anyway.
const PROBE_ATTEMPTS: u32 = 3;

/// Wait after killing a hung probe: on the B60 the exit of the process holding the stuck work
/// triggers xe's ~5 s schedule-disable timeout and the GT reset (10-27 report). Not always: after
/// a SIGKILL on 2026-10-29 the next process stayed stuck for 3 h 37 min until a manual GT reset,
/// so the exit is a hope here, not a fix (D-04 S9 review #5; the recovery text says so).
pub const PROBE_RESET_SETTLE: Duration = Duration::from_secs(10);

/// What [`gpu_readiness`] ended with.
#[derive(Debug, PartialEq, Eq)]
enum Readiness {
    /// Start the worker (the probe passed, failed fast, or every attempt hung — logged).
    Continue,
    /// SIGTERM arrived while probing: stop, no worker.
    Shutdown,
}

/// The GPU readiness probe after a no-teardown exit. The probe child
/// does a tiny compile + inference on every GPU; a GPU whose first work is never scheduled makes
/// it hang. Past [`SupervisorConfig::gpu_probe_deadline_secs`] the child is killed (its exit is
/// what makes the driver reset the stuck engine), the supervisor waits `probe.settle`
/// and tries again, up to [`PROBE_ATTEMPTS`]. A fast failure (an error, not a hang) is logged
/// and the worker starts. The worker's first-GPU-load backstop stays armed either way (until a
/// destructive test shows the probe catches the stuck state). `0` disables the probe.
async fn gpu_readiness(
    probe: &ProbeCmd,
    cfg: &SupervisorConfig,
    sigterm: &mut Signal,
) -> Readiness {
    if cfg.gpu_probe_deadline_secs == 0 {
        return Readiness::Continue;
    }
    let limit = Duration::from_secs(cfg.gpu_probe_deadline_secs);
    for attempt in 1..=PROBE_ATTEMPTS {
        let mut cmd = Command::new(&probe.exe);
        cmd.args(&probe.args)
            .stdout(std::process::Stdio::piped())
            .kill_on_drop(true);
        terminate_with_supervisor(&mut cmd);
        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, "GPU readiness probe could not start — starting the worker");
                return Readiness::Continue;
            }
        };
        let started = Instant::now();
        let waited = tokio::select! {
            r = tokio::time::timeout(limit, child.wait_with_output()) => Some(r),
            _ = sigterm.recv() => None,
        };
        match waited {
            None => {
                tracing::info!(
                    "SIGTERM received during the GPU readiness probe — exiting, no respawn"
                );
                return Readiness::Shutdown;
            }
            Some(Ok(Ok(out))) => {
                let report = String::from_utf8_lossy(&out.stdout)
                    .trim()
                    .replace('\n', "; ");
                if out.status.success() {
                    tracing::info!(attempt, ms = started.elapsed().as_millis(), %report, "GPU readiness probe passed");
                } else {
                    tracing::warn!(attempt, status = %out.status, %report, "GPU readiness probe failed (not a hang) — starting the worker");
                }
                return Readiness::Continue;
            }
            Some(Ok(Err(e))) => {
                tracing::warn!(error = %e, "GPU readiness probe: wait failed — starting the worker");
                return Readiness::Continue;
            }
            Some(Err(_elapsed)) => {
                // The timed-out future owned the child; dropping it killed the probe
                // (`kill_on_drop`), which is the point: its exit triggers the driver's reset.
                tracing::error!(
                    attempt,
                    of = PROBE_ATTEMPTS,
                    deadline_secs = limit.as_secs(),
                    "GPU readiness probe hung — killed it; waiting for the GPU to reset"
                );
                tokio::select! {
                    () = tokio::time::sleep(probe.settle) => {}
                    _ = sigterm.recv() => return Readiness::Shutdown,
                }
            }
        }
    }
    tracing::error!(
        attempts = PROBE_ATTEMPTS,
        "GPU readiness probe hung every time — starting the worker with its first-GPU-load \
         deadline armed"
    );
    Readiness::Continue
}

/// Outcome of one spawn attempt — decides whether the caller loops again.
#[derive(Debug, PartialEq, Eq)]
enum AttemptOutcome {
    /// The supervisor's own `SIGTERM` fired; the worker was signalled and
    /// waited on. Exit for good, no respawn.
    ShuttingDown,
    /// The worker exited on its own, or was killed for hanging — evaluate
    /// backoff/cap and try again.
    Failed,
    /// The generation watchdog killed a healthy worker for a request over the hang ceiling.
    /// Counted like [`Self::Failed`] and also in its own, longer window
    /// ([`watchdog_kill_window`]): one watchdog cycle outlasts `restart_window_secs`, so
    /// the plain cap never trips on a request that wedges every respawn.
    WatchdogKilled,
    /// The worker left on its own with [`crate::startup::EXIT_FFI_STALLED`]: an `OpenVINO` call
    /// was still running at exit. Respawned, and counted in the watchdog-kill window too — a
    /// call that wedges every respawn must hit the same cap as one the watchdog kills.
    FfiStalled,
    /// Killed by the first-GPU-load deadline ([`SupervisorConfig::gpu_first_load_deadline_secs`]).
    /// Counted like [`Self::Failed`]; does not arm the deadline again.
    GpuDeadlineKilled,
    /// The worker exited with [`crate::startup::EXIT_CONFIG_FATAL`]: a
    /// deterministic config error that a respawn cannot fix. Stop for good.
    ConfigFatal,
}

/// Core supervise loop — spawns `exe` with `args`, watches it, respawns with
/// backoff+cap on failure. Generic over the spawn target so a test can point
/// it at a small fake binary instead of the real (slow, GPU-dependent)
/// server; see `tests/supervisor.rs`.
///
/// # Errors
/// Returns an error once [`SupervisorConfig::max_restarts`] is exhausted
/// within [`SupervisorConfig::restart_window_secs`] — a deterministic crash
/// must not loop forever against a slow cold GPU init.
pub async fn run_supervised(
    exe: &Path,
    args: &[String],
    health_addr: SocketAddr,
    cfg: &SupervisorConfig,
) -> anyhow::Result<()> {
    run_supervised_with_probe(exe, args, health_addr, cfg, None).await
}

/// The command of the GPU readiness probe child: the server binary
/// with `--gpu-probe` in production; any command in tests.
#[derive(Clone, Debug)]
pub struct ProbeCmd {
    /// Executable.
    pub exe: std::path::PathBuf,
    /// Arguments.
    pub args: Vec<String>,
    /// Wait after killing a hung probe before the next attempt: [`PROBE_RESET_SETTLE`] in
    /// production (the stuck B60 resets ~5 s after the probe's exit), shorter in tests.
    pub settle: Duration,
}

/// [`run_supervised`] with the GPU readiness probe: after a worker left without teardown, and
/// before the next one is spawned, `probe` must exit within
/// [`SupervisorConfig::gpu_probe_deadline_secs`] (see [`gpu_readiness`]). `None` skips it.
///
/// # Errors
/// As [`run_supervised`].
#[allow(clippy::too_many_lines)] // one linear respawn loop; each step is a call or a guarded exit
pub async fn run_supervised_with_probe(
    exe: &Path,
    args: &[String],
    health_addr: SocketAddr,
    cfg: &SupervisorConfig,
    probe: Option<&ProbeCmd>,
) -> anyhow::Result<()> {
    let mut attempts: Vec<Instant> = Vec::new();
    let mut watchdog_kills: Vec<Instant> = Vec::new();
    let mut sigterm = signal(SignalKind::terminate())?;
    // Set after a worker left without teardown; arms the next worker's first-GPU-load deadline.
    let mut arm_gpu_deadline = false;
    // The readiness probe runs after every no-teardown exit, including one by the first-load
    // deadline kill (0.9.0 review E, shutdown-1); only the deadline itself is once per episode.
    let mut run_probe = false;
    let mut spawn_no: u64 = 0;
    // C27: the set the next worker targets (`None` for the first spawn: the full preload).
    let mut restore = RestoreState::default();

    loop {
        let now = Instant::now();
        prune_old_attempts(&mut attempts, cfg, now);
        let kill_window = watchdog_kill_window(cfg);
        watchdog_kills.retain(|&t| now.duration_since(t) < kill_window);
        if attempts.len() >= cfg.max_restarts {
            anyhow::bail!(
                "gave up after {} restarts within {}s — worker is crash-looping, not respawning again",
                attempts.len(),
                cfg.restart_window_secs
            );
        }
        if watchdog_kills.len() >= cfg.max_restarts {
            anyhow::bail!(
                "gave up after the generation watchdog killed the worker (or it left with OpenVINO \
                 calls stuck, exit {}) {} times within {}s — a request (or a client retrying it) \
                 wedges every respawn; not respawning again. Find the model in the 'generation \
                 watchdog' / 'still in flight' log lines above",
                crate::startup::EXIT_FFI_STALLED,
                watchdog_kills.len(),
                kill_window.as_secs()
            );
        }

        if let Some(delay) = backoff_delay(attempts.len(), cfg) {
            tracing::warn!(
                attempt = attempts.len() + 1,
                max = cfg.max_restarts,
                delay_secs = delay.as_secs(),
                "respawning worker after backoff"
            );
            tokio::select! {
                () = tokio::time::sleep(delay) => {}
                _ = sigterm.recv() => {
                    tracing::info!("SIGTERM received while backing off — exiting, no respawn");
                    return Ok(());
                }
            }
        }

        if probe_says_shut_down(run_probe, probe, cfg, &mut sigterm).await {
            return Ok(());
        }

        // Fresh per spawn (OS CSPRNG via uuid v4): admits only this worker's watchdog read.
        let watchdog_token = uuid::Uuid::new_v4().simple().to_string();
        spawn_no += 1;
        // Random per spawn: unique even when several supervisors share a process (the tests).
        let progress = std::env::temp_dir().join(format!(
            "rustedvino-progress-{}-{spawn_no}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        let _ = std::fs::remove_file(&progress);
        let mut child = worker_command(
            exe,
            args,
            &watchdog_token,
            &progress,
            restore.handed.as_deref(),
        )
        .spawn()?;
        let deadline = (arm_gpu_deadline && cfg.gpu_first_load_deadline_secs > 0)
            .then(|| Duration::from_secs(cfg.gpu_first_load_deadline_secs));
        tracing::info!(
            pid = child.id().unwrap_or(0),
            gpu_first_load_deadline_secs = deadline.map(|d| d.as_secs()),
            "worker spawned"
        );

        let mut no_teardown = false;
        let outcome = run_one_attempt(
            &mut child,
            health_addr,
            cfg,
            &watchdog_token,
            &mut sigterm,
            WorkerProgress {
                path: &progress,
                gpu_deadline: deadline,
            },
            &mut no_teardown,
        )
        .await;
        end_attempt_progress(&outcome, &progress, &mut restore, cfg, no_teardown);
        // Once per episode: a worker this deadline killed does not arm it again (no loop), but the
        // next worker is still probed first.
        run_probe = no_teardown;
        arm_gpu_deadline = no_teardown && outcome != AttemptOutcome::GpuDeadlineKilled;
        match outcome {
            AttemptOutcome::ShuttingDown => return Ok(()),
            AttemptOutcome::Failed | AttemptOutcome::GpuDeadlineKilled => {
                attempts.push(Instant::now());
            }
            AttemptOutcome::WatchdogKilled | AttemptOutcome::FfiStalled => {
                attempts.push(Instant::now());
                watchdog_kills.push(Instant::now());
            }
            // MR-5c: a `ConfigFatal`, so the supervisor exits with the worker's 78 (as a plain
            // start does), not 1.
            AttemptOutcome::ConfigFatal => {
                return Err(anyhow::Error::new(crate::startup::ConfigFatal(format!(
                    "worker exited with a fatal configuration error (exit {}) — not respawning; \
                     fix the config (see the worker's error above) and start again",
                    crate::startup::EXIT_CONFIG_FATAL
                ))));
            }
        }
    }
}

/// The worker's per-spawn progress file and the first-GPU-load deadline armed for this spawn
/// (`None` = not armed).
struct WorkerProgress<'a> {
    path: &'a Path,
    gpu_deadline: Option<Duration>,
}

/// Runs one spawn-to-exit attempt: races the worker becoming healthy against
/// it exiting on its own and against the supervisor's own `SIGTERM`. Once
/// healthy, keeps racing the exit against `SIGTERM` until one happens.
///
/// `gpu_deadline` (armed after a no-teardown exit, see
/// [`SupervisorConfig::gpu_first_load_deadline_secs`]) also races the first GPU load. Sets
/// `no_teardown` when this worker itself left without teardown (exit 75/76, a signal, or a kill
/// by this supervisor) — the caller arms the deadline for the next one.
/// The worker's command line and environment, set to end with the supervisor (MR-5d).
/// `restore` (C27) goes in [`crate::startup::RESTORE_MODELS_ENV`], `Some([])` as `[]` (a respawn
/// whose crashed set was empty: no preload); for `None` (a fresh start) the variable is removed,
/// so a value inherited by the supervisor never reaches a worker.
fn worker_command(
    exe: &Path,
    args: &[String],
    watchdog_token: &str,
    progress: &Path,
    restore: Option<&[String]>,
) -> Command {
    let mut cmd = Command::new(exe);
    cmd.args(args)
        .env(crate::startup::SUPERVISED_ENV, "1")
        .env(crate::startup::WATCHDOG_TOKEN_ENV, watchdog_token)
        .env(crate::startup::PROGRESS_FILE_ENV, progress);
    match restore.map(serde_json::to_string) {
        Some(Ok(json)) => {
            cmd.env(crate::startup::RESTORE_MODELS_ENV, json);
        }
        _ => {
            cmd.env_remove(crate::startup::RESTORE_MODELS_ENV);
        }
    }
    terminate_with_supervisor(&mut cmd);
    cmd
}

/// MR-5d: the worker gets SIGTERM (a graceful drain that frees VRAM — never a hard kill, which
/// leaves a B60 context hanging) when the supervisor dies, even by `kill -9`; it used to keep
/// serving orphaned (PPID 1) with nothing saying supervision was gone. Linux only
/// (`PR_SET_PDEATHSIG`); elsewhere a no-op.
///
/// The signal fires when the *thread* that spawned the child exits. The worker is spawned from
/// an async task on a tokio runtime worker thread, which lives as long as the runtime, i.e.
/// the supervisor; never spawn it from `spawn_blocking` (those threads exit when idle).
fn terminate_with_supervisor(cmd: &mut Command) {
    #[cfg(target_os = "linux")]
    {
        let supervisor = std::process::id();
        // SAFETY: the closure runs in the forked child before `exec`, where only
        // async-signal-safe calls are allowed: `prctl`, `getppid` and `raise` are. It captures
        // only a `u32` (no allocation, no locks) and touches no shared state.
        unsafe {
            cmd.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM, 0, 0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                // The supervisor may have died between fork and prctl: then nothing will signal.
                #[allow(clippy::cast_sign_loss)]
                if libc::getppid() as u32 != supervisor {
                    libc::raise(libc::SIGTERM);
                }
                Ok(())
            });
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cmd;
}

async fn run_one_attempt(
    child: &mut Child,
    health_addr: SocketAddr,
    cfg: &SupervisorConfig,
    watchdog_token: &str,
    sigterm: &mut Signal,
    progress: WorkerProgress<'_>,
    no_teardown: &mut bool,
) -> AttemptOutcome {
    let gpu_deadline = progress.gpu_deadline.map(|d| (d, progress.path));
    let progress = progress.path;
    let health_timeout = Duration::from_secs(cfg.health_timeout_secs);
    // G-1: a preload with no load deadline of its own is bounded by the hang ceiling (none when
    // that is off too).
    let uncapped_preload_limit = (cfg.watchdog_hang_ceiling_secs > 0)
        .then(|| Duration::from_secs(cfg.watchdog_hang_ceiling_secs));
    let became_healthy = tokio::select! {
        healthy = wait_for_health_or_preload(
            health_addr,
            health_timeout,
            progress,
            uncapped_preload_limit,
        ) => healthy,
        status = child.wait() => {
            *no_teardown = left_without_teardown(&status);
            return exit_outcome(status);
        }
        _ = sigterm.recv() => {
            return shut_down_child(child).await;
        }
        allowance_secs = first_gpu_load_overdue(gpu_deadline, uncapped_preload_limit) => {
            return gpu_deadline_kill(child, cfg, progress, gpu_deadline, allowance_secs, no_teardown)
                .await;
        }
    };

    if !became_healthy {
        tracing::error!(
            timeout_secs = cfg.health_timeout_secs,
            "worker did not become healthy in time — treating as hung"
        );
        let grace = preload_kill_grace(cfg, progress, false);
        *no_teardown = hang_kill_child(child, grace, progress).await;
        return AttemptOutcome::Failed;
    }

    tracing::info!("worker healthy");

    // Watchdog disabled (`watchdog_hang_ceiling_secs: 0`): unchanged two-way
    // race. Kept as a separate branch rather than folding a no-op future into
    // the `select!` below — a `0`-second ceiling would otherwise fire
    // immediately on the first real tick.
    // The first-GPU-load deadline keeps running once the worker is healthy: with no GPU preload
    // the first GPU load comes from a client (0.9.0 review E, shutdown-3).
    if cfg.watchdog_hang_ceiling_secs == 0 {
        return tokio::select! {
            status = child.wait() => {
                *no_teardown = left_without_teardown(&status);
                exit_outcome(status)
            }
            _ = sigterm.recv() => shut_down_child(child).await,
            allowance_secs = first_gpu_load_overdue(gpu_deadline, uncapped_preload_limit) => {
                gpu_deadline_kill(child, cfg, progress, gpu_deadline, allowance_secs, no_teardown)
                    .await
            }
        };
    }

    tokio::select! {
        status = child.wait() => {
            *no_teardown = left_without_teardown(&status);
            exit_outcome(status)
        }
        _ = sigterm.recv() => shut_down_child(child).await,
        allowance_secs = first_gpu_load_overdue(gpu_deadline, uncapped_preload_limit) => {
            gpu_deadline_kill(child, cfg, progress, gpu_deadline, allowance_secs, no_teardown).await
        }
        () = generation_watchdog(health_addr, cfg, watchdog_token) => {
            tracing::error!(
                ceiling_secs = cfg.watchdog_hang_ceiling_secs,
                "generation watchdog: oldest in-flight generation exceeded ceiling — treating as hung"
            );
            *no_teardown = hang_kill_child(child, watchdog_kill_grace(cfg), progress).await;
            AttemptOutcome::WatchdogKilled
        }
    }
}

/// The first GPU load did not return within the armed deadline after a no-teardown exit: kill
/// this worker (its exit is what may make the driver reset the stuck engine) and respawn.
async fn gpu_deadline_kill(
    child: &mut Child,
    cfg: &SupervisorConfig,
    progress: &Path,
    gpu_deadline: Option<(Duration, &Path)>,
    allowance_secs: u64,
    no_teardown: &mut bool,
) -> AttemptOutcome {
    tracing::error!(
        deadline_secs = gpu_deadline.map_or(0, |(d, _)| d.as_secs()),
        load_timeout_secs = allowance_secs,
        "first GPU model load did not finish within the deadline after a no-teardown exit — the \
         device is likely stuck from the previous worker; killing this worker so the driver \
         resets the engine, then respawning"
    );
    // A load with no deadline of its own is never overdue to the worker: its stop waits.
    let grace = preload_kill_grace(cfg, progress, allowance_secs == 0);
    *no_teardown = hang_kill_child(child, grace, progress).await;
    AttemptOutcome::GpuDeadlineKilled
}

/// Whether a worker that exited on its own left without its normal teardown: the GPU-poison
/// exit (75), the in-flight exit (76), or death by a signal (crash, OOM killer). Those leave a
/// live GPU context to the kernel, which on the B60 can stall the next process's first GPU work.
fn left_without_teardown(status: &std::io::Result<ExitStatus>) -> bool {
    match status {
        Ok(s) => s.signal().is_some() || is_poison_exit(s.code()) || is_ffi_stalled_exit(s.code()),
        Err(_) => false,
    }
}

/// The complete lines of a progress file the worker may still be appending to: a last line
/// without its `\n` is still being written and is left out.
fn complete_lines(contents: &str) -> std::str::Lines<'_> {
    contents[..contents.rfind('\n').unwrap_or(0)].lines()
}

/// C27: how many times one model is restored after a crash respawn within
/// [`SupervisorConfig::restart_window_secs`]. A model loaded again (by an operator) and resident
/// at a later crash inside the window is not restored a second time.
pub const MAX_RESTORES_PER_WINDOW: usize = 1;

/// C27 respawn restore bookkeeping, kept across the spawns of one supervisor (a new supervisor
/// starts empty: a clean service restart restores nothing).
#[derive(Debug, Default)]
pub struct RestoreState {
    /// The set handed to the current worker in [`crate::startup::RESTORE_MODELS_ENV`]: `None` =
    /// a fresh start (variable absent, full `preload`); `Some` = a respawn targeting that set
    /// (`preload` models included; `Some([])` = nothing, no preload).
    pub handed: Option<Vec<String>>,
    /// When each of `handed` was first handed over.
    handed_at: std::collections::HashMap<String, Instant>,
    /// Restores that really started (a `restore_started` line), stamped with their hand-over
    /// time; pruned to the restart window. Counts toward [`MAX_RESTORES_PER_WINDOW`].
    history: std::collections::HashMap<String, Vec<Instant>>,
}

/// Why a model is not handed to the next worker although the crashed one had it (or was asked
/// to restore it). Each is logged as a WARN.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestoreDrop {
    /// Its restore started in the crashed worker but never reached Ready (the load may be what
    /// crashed it).
    RestoreNeverReady(String),
    /// It was restored, and the worker crashed again while it was resident, within the restart
    /// window of its hand-over.
    ResidentAtSecondCrash(String),
    /// It was already restored [`MAX_RESTORES_PER_WINDOW`] time(s) within the restart window.
    RestoredRecently(String),
}

/// What the crashed worker's progress file says: resident models at the crash (in first-ready
/// order), restores started, restores that reached Ready after their start, the models its
/// preload phase started, whether it crashed inside that phase, and whether any model was ever
/// Ready in it.
///
/// M1: a `model_unloaded` inside an open load window (`load_started` …
/// `load_ended`) is that load's possible victim. It stays in `resident` only while a load open at
/// its eviction is still open at the end of the file (the worker crashed during it). It is out
/// once those loads all returned — Ready or not (a load that failed without crashing really did
/// evict it, and the worker kept serving without it) — or when a later `model_ready` /
/// `model_unloaded` of the victim itself decides. A file with no window lines reads as before.
struct ProgressModels {
    resident: Vec<String>,
    started: std::collections::HashSet<String>,
    reached_ready: std::collections::HashSet<String>,
    preload_reached: std::collections::HashSet<String>,
    /// `preload_phase_started` without a later `preload_phase_done`. A file with neither (a
    /// worker that does not write them) reads as a finished phase.
    in_preload_phase: bool,
    any_ready: bool,
    /// Loads whose window is still open at the end of the file (the crash came during them).
    open_at_end: std::collections::HashSet<String>,
    /// The victims the last load window released when it closed without Ready, if no model
    /// event came after it (the worker may have left because of that load: a poisoned GPU, a
    /// stalled call). Kept only after an abnormal exit, see [`plan_restore_after`].
    released_at_end: Vec<String>,
}

fn progress_models(progress: &str) -> ProgressModels {
    let mut p = ProgressModels {
        resident: Vec::new(),
        started: std::collections::HashSet::new(),
        reached_ready: std::collections::HashSet::new(),
        preload_reached: std::collections::HashSet::new(),
        in_preload_phase: false,
        any_ready: false,
        open_at_end: std::collections::HashSet::new(),
        released_at_end: Vec::new(),
    };
    // Open load windows (nested ones of one id counted), and each victim's loads.
    let mut open: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut victims: std::collections::HashMap<String, std::collections::HashSet<String>> =
        std::collections::HashMap::new();
    for line in complete_lines(progress) {
        let model_event = [
            "load_started ",
            "load_ended ",
            "model_ready ",
            "model_unloaded ",
            "restore_started ",
            "preload_model ",
            "preload_phase_",
        ]
        .iter()
        .any(|k| line.starts_with(k));
        if model_event {
            // Something happened after the last released window: the worker went on serving.
            p.released_at_end.clear();
        }
        if let Some(id) = line.strip_prefix("load_started ") {
            *open.entry(id.to_owned()).or_default() += 1;
        } else if let Some(id) = line.strip_prefix("load_ended ") {
            if let Some(n) = open.get_mut(id) {
                *n -= 1;
                if *n == 0 {
                    open.remove(id);
                    // That load returned: a victim of no other open load is really out.
                    let out: Vec<String> = victims
                        .iter_mut()
                        .filter_map(|(v, loads)| {
                            loads.remove(id);
                            loads.is_empty().then(|| v.clone())
                        })
                        .collect();
                    for v in &out {
                        victims.remove(v);
                        p.resident.retain(|r| r != v);
                    }
                    p.released_at_end = out;
                }
            }
        } else if line == "preload_phase_started" {
            p.in_preload_phase = true;
        } else if line == "preload_phase_done" {
            p.in_preload_phase = false;
        } else if let Some(id) = line.strip_prefix("preload_model ") {
            p.preload_reached.insert(id.to_owned());
        } else if let Some(id) = line.strip_prefix("model_ready ") {
            p.any_ready = true;
            if p.started.contains(id) {
                p.reached_ready.insert(id.to_owned());
            }
            if !p.resident.iter().any(|r| r == id) {
                p.resident.push(id.to_owned());
            }
            victims.remove(id);
            // The load reached Ready: the models it evicted are really out.
            let out: Vec<String> = victims
                .iter()
                .filter(|(_, loads)| loads.contains(id))
                .map(|(v, _)| v.clone())
                .collect();
            for v in out {
                victims.remove(&v);
                p.resident.retain(|r| *r != v);
            }
        } else if let Some(id) = line.strip_prefix("model_unloaded ") {
            if open.is_empty() || !p.resident.iter().any(|r| r == id) {
                victims.remove(id);
                p.resident.retain(|r| r != id);
            } else {
                victims.insert(id.to_owned(), open.keys().cloned().collect());
            }
        } else if let Some(id) = line.strip_prefix("restore_started ") {
            p.started.insert(id.to_owned());
        }
    }
    p.open_at_end = open.into_keys().collect();
    p
}

/// C27: the set the next worker targets, from the crashed worker's progress file
/// (`model_ready` / `model_unloaded` / `restore_started` / `preload_model` /
/// `preload_phase_*` / `load_started` / `load_ended` lines, last event per id wins; a load's victim
/// counts as resident while that load has not reached Ready, see `ProgressModels`) and the set that worker itself was handed
/// (`state.handed`). Updates `state` and returns the drops to log.
///
/// A fresh crashed worker (handed `None`) that crashed inside its preload phase, or before any
/// model was Ready, hands `None` on: the next worker runs the full `preload`, as a fresh start.
/// Otherwise the next set (`Some`, possibly empty) = models resident at the crash, plus handed
/// models not reached yet: a restore not started, or a `preload` model its preload phase had not
/// started (or was loading) when it crashed inside that phase. A handed `preload` model its
/// finished preload phase did not get Ready is not carried (it was not in the set at the crash).
/// Dropped: a restore that started but never reached Ready; a restored model resident again
/// at a crash inside `window` of its hand-over (after the window it is an ordinary resident
/// model); a model whose restore already started [`MAX_RESTORES_PER_WINDOW`] time(s) within
/// `window`. A model handed over but loaded by an operator before its restore ran counts no
/// restore. `preload` models load in the preload phase, which writes no `restore_started`, so
/// these drops never apply to them: a `preload` model Ready at every crash is handed on every
/// time (as every respawn reloaded `preload` before). The worker writes `model_ready` only for
/// `Eager` and `preload` models, so nothing else is ever handed over.
pub fn plan_restore(
    progress: &str,
    state: &mut RestoreState,
    now: Instant,
    window: Duration,
) -> Vec<RestoreDrop> {
    plan_restore_after(progress, state, now, window, false)
}

/// [`plan_restore`] knowing how the crashed worker left. `abnormal` = it left without its
/// teardown (exit 75 poisoned GPU, 76 a stalled call, a signal, a kill by this supervisor): then
/// the victims of the last load window that closed without Ready, with no model event after it,
/// are kept too — that load may be what took the worker down (a poisoned GPU load returns an error
/// and closes its window before the exit). After a clean exit, or once anything happened after
/// that window, they are out (M1 review, N1).
pub fn plan_restore_after(
    progress: &str,
    state: &mut RestoreState,
    now: Instant,
    window: Duration,
    abnormal: bool,
) -> Vec<RestoreDrop> {
    let mut p = progress_models(progress);
    if abnormal {
        for v in std::mem::take(&mut p.released_at_end) {
            if !p.resident.contains(&v) {
                p.resident.push(v);
            }
        }
    }
    let handed_some = state.handed.take();
    let fresh = handed_some.is_none();
    let handed = handed_some.unwrap_or_default();
    let mut handed_at = std::mem::take(&mut state.handed_at);
    // A restore counts when it really started (F4), stamped with its hand-over time.
    for id in handed.iter().filter(|id| p.started.contains(*id)) {
        let at = handed_at.get(id).copied().unwrap_or(now);
        state.history.entry(id.clone()).or_default().push(at);
    }
    for times in state.history.values_mut() {
        times.retain(|&t| now.duration_since(t) < window);
    }
    state.history.retain(|_, times| !times.is_empty());
    if fresh && (p.in_preload_phase || !p.any_ready) {
        // Nothing was serving yet (no runtime load before the listener), or nothing was ever
        // Ready: the next worker starts as fresh, with the full `preload`.
        return Vec::new();
    }

    let mut drops = Vec::new();
    let mut next: Vec<String> = Vec::new();
    let mut next_at = std::collections::HashMap::new();
    let mut dropped: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for id in &handed {
        let at = handed_at.remove(id).unwrap_or(now);
        if !p.started.contains(id) {
            // Not reached before the crash: carried over — unless it is resident anyway (an
            // operator loaded it first, or its preload phase did), then it is an ordinary
            // resident model below; or it is a `preload` model a finished preload phase did not
            // get Ready (refused, failed): not in the set at the crash.
            // A `preload` model whose own load the crash interrupted (its window still open: a
            // deferred one, M2, loads after the phase) is carried, as one loading
            // inside the phase is.
            let preload_missed = p.preload_reached.contains(id)
                && !p.in_preload_phase
                && !p.open_at_end.contains(id);
            if !p.resident.contains(id) && !preload_missed {
                next.push(id.clone());
                next_at.insert(id.clone(), at);
            }
        } else if !p.reached_ready.contains(id) {
            drops.push(RestoreDrop::RestoreNeverReady(id.clone()));
            dropped.insert(id);
        } else if p.resident.contains(id) && now.duration_since(at) < window {
            drops.push(RestoreDrop::ResidentAtSecondCrash(id.clone()));
            dropped.insert(id);
        }
    }
    for id in &p.resident {
        if dropped.contains(id.as_str()) || next.contains(id) {
            continue;
        }
        if state.history.get(id).map_or(0, Vec::len) >= MAX_RESTORES_PER_WINDOW {
            drops.push(RestoreDrop::RestoredRecently(id.clone()));
            continue;
        }
        next.push(id.clone());
        next_at.insert(id.clone(), now);
    }
    state.handed = Some(next);
    state.handed_at = next_at;
    drops
}

/// The progress file's text (lossy: a torn multibyte last line must not cost the whole file;
/// [`complete_lines`] drops that line anyway). Empty when it cannot be read.
fn read_progress(path: &Path) -> String {
    std::fs::read(path)
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default()
}

/// The end of one attempt's progress file: C27 [`plan_restore`] from it (the models the next
/// worker restores) — only after a crash (the outcomes that respawn; a clean stop or a
/// config-fatal exit ends the loop, and the set is cleared) — then delete it.
fn end_attempt_progress(
    outcome: &AttemptOutcome,
    progress: &Path,
    restore: &mut RestoreState,
    cfg: &SupervisorConfig,
    abnormal: bool,
) {
    if matches!(
        outcome,
        AttemptOutcome::Failed
            | AttemptOutcome::WatchdogKilled
            | AttemptOutcome::FfiStalled
            | AttemptOutcome::GpuDeadlineKilled
    ) {
        let contents = read_progress(progress);
        let drops = plan_restore_after(
            &contents,
            restore,
            Instant::now(),
            Duration::from_secs(cfg.restart_window_secs),
            abnormal || *outcome == AttemptOutcome::GpuDeadlineKilled,
        );
        log_restore_plan(&drops, restore.handed.as_deref());
    } else {
        restore.handed = None;
        restore.handed_at.clear();
    }
    let _ = std::fs::remove_file(progress);
}

/// Log [`plan_restore`]'s drops and hand-over (one WARN per drop).
fn log_restore_plan(drops: &[RestoreDrop], handed: Option<&[String]>) {
    for drop in drops {
        match drop {
            RestoreDrop::RestoreNeverReady(id) => tracing::warn!(
                model = %id,
                reason = "restore_never_ready",
                "respawn restore: not restoring the model again - its restore started in the \
                 crashed worker and never reached Ready"
            ),
            RestoreDrop::ResidentAtSecondCrash(id) => tracing::warn!(
                model = %id,
                reason = "resident_at_second_crash",
                "respawn restore: not restoring the model again - the worker crashed again while \
                 the restored model was loaded"
            ),
            RestoreDrop::RestoredRecently(id) => tracing::warn!(
                model = %id,
                reason = "restored_recently",
                max = MAX_RESTORES_PER_WINDOW,
                "respawn restore: not restoring the model - already restored within the restart \
                 window"
            ),
        }
    }
    match handed {
        None => tracing::info!(
            "respawn restore: the crashed worker had not finished its preload (or had nothing \
             Ready) - the next worker loads the full preload"
        ),
        Some([]) => tracing::info!(
            "respawn restore: nothing was loaded at the crash - the next worker restores nothing \
             and skips the preload"
        ),
        Some(models) => tracing::info!(
            models = ?models,
            "respawn restore: the next worker brings back these models (preload models in its \
             preload phase, the rest after listening)"
        ),
    }
}

/// What the worker's progress file says about its first GPU load: (started, with the load
/// deadline it runs under in seconds — `0` = none; done).
fn gpu_load_progress(contents: &str) -> (Option<u64>, bool) {
    let mut started = None;
    let mut done = false;
    for l in complete_lines(contents) {
        if l == "gpu_load_done" {
            done = true;
        } else if let Some(n) = l.strip_prefix("gpu_load_started ") {
            started = started.or_else(|| n.trim().parse().ok());
        } else if l == "gpu_load_started" {
            // A worker from before D-04 S8 (binary replaced under a running supervisor): no
            // deadline reported, so the configured one applies.
            started = started.or(Some(0));
        }
    }
    (started, done)
}

/// D-04 S8: the stop budgets the worker declared, in order (`stop_budget <secs>`, each the time
/// its stop may still take from that step; `startup::note_stop_budget`).
fn declared_stop_budgets(contents: &str) -> Vec<Duration> {
    complete_lines(contents)
        .filter_map(|l| l.strip_prefix("stop_budget "))
        .filter_map(|n| n.trim().parse().ok())
        .map(Duration::from_secs)
        .collect()
}

/// `SIGTERM`→`SIGKILL` grace for a worker killed before it became healthy (preload
/// overran the health deadline, or the first-GPU-load deadline): while a load runs, the worker's
/// own stop bound for the engines its progress file shows (preloads done + the running one) plus
/// slack, at least `hang_kill_grace_secs`; with nothing loading, `hang_kill_grace_secs`.
/// `load_waits`: the running load has no deadline of its own, so the worker's stop waits for it up to `LOAD_SETTLE_CAP` — a SIGKILL inside
/// that wait lands on a live GPU load, the D-04 wedge. A capped preload is already overdue when
/// these kills fire, and the worker leaves within seconds.
fn preload_kill_grace(cfg: &SupervisorConfig, progress: &Path, load_waits: bool) -> Duration {
    let contents = std::fs::read_to_string(progress).unwrap_or_default();
    let seen = preload_milestones(&contents);
    let done = seen
        .iter()
        .filter(|m| **m == PreloadMilestone::Done)
        .count();
    let running = match seen.last() {
        Some(PreloadMilestone::Started { allowance_secs }) => Some(*allowance_secs == 0),
        _ => None,
    };
    match running {
        // Nothing loading: nothing for the worker's stop to wait for.
        None if !load_waits => Duration::from_secs(cfg.hang_kill_grace_secs),
        uncapped => preload_kill_grace_for(cfg, done + 1, load_waits || uncapped == Some(true)),
    }
}

/// [`preload_kill_grace`] once the progress file is read.
fn preload_kill_grace_for(cfg: &SupervisorConfig, engines: usize, load_waits: bool) -> Duration {
    Duration::from_secs(cfg.hang_kill_grace_secs)
        .max(crate::startup::preload_stop_bound(engines, load_waits) + KILL_SLACK)
}

/// Slack on top of a worker's own stop bound before the supervisor sends `SIGKILL`.
const KILL_SLACK: Duration = Duration::from_secs(5);

/// The shortest `SIGTERM`→`SIGKILL` gap: room for the worker's first stop-budget declaration
/// (its stop-path wait polls every 250 ms) to be read.
const MIN_KILL_GRACE: Duration = Duration::from_secs(1);

/// G-1: a preload milestone the worker writes to the progress file (see
/// `startup::note_preload`): a preload model started, with the load deadline it runs under (`0` =
/// none), or one finished (any outcome).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreloadMilestone {
    /// `preload_started <allowance_secs>`.
    Started { allowance_secs: u64 },
    /// `preload_done`.
    Done,
}

/// The preload milestones in `contents`, in order. The file is read while the worker appends to
/// it: a last line without its `\n` is still being written and is ignored (a torn
/// `preload_started 6` of `600` must not read as 6 s).
fn preload_milestones(contents: &str) -> Vec<PreloadMilestone> {
    complete_lines(contents)
        .filter_map(|l| {
            if l == "preload_done" {
                Some(PreloadMilestone::Done)
            } else {
                l.strip_prefix("preload_started ")
                    .and_then(|n| n.trim().parse().ok())
                    .map(|allowance_secs| PreloadMilestone::Started { allowance_secs })
            }
        })
        .collect()
}

/// G-1: when a worker that has not become healthy counts as hung. `health` from the spawn, moved
/// later — never earlier — by each preload milestone, as first seen at its instant: a started
/// preload gets its own load allowance plus `health` (the worker reacts to its own deadline and
/// binds), a finished one `health` again. A started preload with no load deadline gets
/// `uncapped_limit` plus `health`; `None` (no deadline at all) when that is unset too.
fn health_deadline(
    spawned: Instant,
    health: Duration,
    seen: &[(PreloadMilestone, Instant)],
    uncapped_limit: Option<Duration>,
) -> Option<Instant> {
    let mut bounded = spawned + health;
    let mut unbounded = false;
    for (milestone, at) in seen {
        let allowance = match milestone {
            PreloadMilestone::Started { allowance_secs: 0 } => uncapped_limit,
            PreloadMilestone::Started { allowance_secs } => {
                Some(Duration::from_secs(*allowance_secs))
            }
            PreloadMilestone::Done => Some(Duration::ZERO),
        };
        match allowance {
            Some(a) => {
                bounded = bounded.max(*at + a + health);
                unbounded = false;
            }
            None => unbounded = true,
        }
    }
    (!unbounded).then_some(bounded)
}

/// G-1: wait for `/health` (polled every 500 ms), but a worker that reports preload progress
/// in `progress` is not hung while that progress keeps its deadline ahead (see
/// [`health_deadline`]). The preload runs before the worker listens on its port, so `/health`
/// cannot answer during it; a fixed `health_timeout_secs` killed every preload longer than that.
async fn wait_for_health_or_preload(
    addr: SocketAddr,
    health: Duration,
    progress: &Path,
    uncapped_limit: Option<Duration>,
) -> bool {
    let spawned = Instant::now();
    let mut seen: Vec<(PreloadMilestone, Instant)> = Vec::new();
    loop {
        if tokio::time::timeout(HEALTH_PROBE_TIMEOUT, probe_health_once(addr))
            .await
            .unwrap_or(false)
        {
            return true;
        }
        let contents = std::fs::read_to_string(progress).unwrap_or_default();
        for milestone in preload_milestones(&contents).into_iter().skip(seen.len()) {
            tracing::info!(
                ?milestone,
                "worker is preloading - health deadline follows its progress"
            );
            seen.push((milestone, Instant::now()));
        }
        if health_deadline(spawned, health, &seen, uncapped_limit)
            .is_some_and(|d| Instant::now() >= d)
        {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// Upper bound of one `/health` probe inside [`wait_for_health_or_preload`].
const HEALTH_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// Resolves when an armed deadline passes with the first GPU load started but not done; never
/// resolves when unarmed, before a GPU load starts, or once one is done (a worker with no GPU
/// models is never killed by it). Returns the load's own deadline in seconds (`0` = none).
///
/// The deadline is the configured one or the load's own `load_timeout_secs`, whichever is
/// longer, so it never fires while the worker still counts the load as running normally (a 30B
/// coder took 169 s against a fixed 180 s). A load with no deadline of its own (`0`) gets the
/// longer of the configured deadline and `uncapped` (the hang ceiling, as for a preload without
/// one); with the ceiling off it keeps the configured deadline, the backstop this exists for.
async fn first_gpu_load_overdue(
    deadline: Option<(Duration, &Path)>,
    uncapped: Option<Duration>,
) -> u64 {
    let Some((limit, path)) = deadline else {
        return std::future::pending().await;
    };
    let mut started_at: Option<Instant> = None;
    loop {
        tokio::time::sleep(Duration::from_millis(500)).await;
        let contents = std::fs::read_to_string(path).unwrap_or_default();
        match gpu_load_progress(&contents) {
            (_, true) => return std::future::pending().await,
            (Some(allowance), false) => {
                let own = if allowance == 0 {
                    uncapped.unwrap_or_default()
                } else {
                    Duration::from_secs(allowance)
                };
                let limit = limit.max(own);
                if started_at.get_or_insert_with(Instant::now).elapsed() >= limit {
                    return allowance;
                }
            }
            (None, false) => {}
        }
    }
}

/// Polls `GET /v1/admin/health/watchdog` every [`SupervisorConfig::watchdog_poll_secs`];
/// resolves once the oldest in-flight generation's age exceeds
/// [`SupervisorConfig::watchdog_hang_ceiling_secs`] — the signal `/health`
/// cannot provide, since a wedged engine thread leaves the HTTP listener (and
/// therefore `/health`) completely unaffected (reproduced 2026-07-28,
/// the project's internal engineering log).
///
/// Never resolves on a probe failure (connection hiccup, non-200, malformed body) or
/// on an explicit `null` (nothing generating) — only a *confirmed* age over
/// the ceiling counts, so a transient blip can never kill a healthy worker.
/// A failure is not silent, though: the first one of a streak logs an ERROR (the
/// watchdog is blind until it clears) and the recovery logs once. Silence here is
/// what hid a watchdog that never worked on a keyed server (2026-10-27).
/// Caller must not race this in when `watchdog_hang_ceiling_secs == 0`
/// (disabled).
async fn generation_watchdog(addr: SocketAddr, cfg: &SupervisorConfig, token: &str) {
    let mut ticker = tokio::time::interval(Duration::from_secs(cfg.watchdog_poll_secs));
    ticker.tick().await; // fires immediately; the worker only just became healthy
    let mut blind = false;
    loop {
        ticker.tick().await;
        // Config seconds are always small (a sane ceiling is minutes, not
        // millennia) — never near f64's 52-bit exact-integer boundary.
        #[allow(clippy::cast_precision_loss)]
        let ceiling_secs = cfg.watchdog_hang_ceiling_secs as f64;
        match probe_watchdog_once(addr, token).await {
            Ok(age) => {
                if blind {
                    blind = false;
                    tracing::info!("generation watchdog can read the worker's state again");
                }
                if age.is_some_and(|age_secs| age_secs > ceiling_secs) {
                    return;
                }
            }
            Err(reason) => {
                if !blind {
                    blind = true;
                    tracing::error!(
                        %reason,
                        "generation watchdog cannot read the worker's state — it cannot detect \
                         a hung generation until this clears"
                    );
                }
            }
        }
    }
}

/// Upper bound of one whole watchdog probe (connect, write, read) inside
/// [`generation_watchdog`] — the same 2 s as [`HEALTH_PROBE_TIMEOUT`].
const WATCHDOG_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// Single probe of [`crate::WATCHDOG_PATH`], authenticated with this worker's
/// per-spawn `token`: `Ok(Some(age))`, `Ok(None)` for an explicit `null` (nothing
/// generating), or `Err` with the reason the state could not be read. The whole probe is
/// bounded by [`WATCHDOG_PROBE_TIMEOUT`] (P-01): a worker that accepts the connection but
/// never answers (a stuck registry lock, every async thread busy) is an `Err`
/// ("timed out after 2 s"), so the watchdog logs itself blind instead of waiting forever.
async fn probe_watchdog_once(addr: SocketAddr, token: &str) -> Result<Option<f64>, String> {
    probe_watchdog_within(addr, token, WATCHDOG_PROBE_TIMEOUT).await
}

/// [`probe_watchdog_once`] with an explicit `bound` (tests use a short one).
async fn probe_watchdog_within(
    addr: SocketAddr,
    token: &str,
    bound: Duration,
) -> Result<Option<f64>, String> {
    tokio::time::timeout(bound, probe_watchdog_unbounded(addr, token))
        .await
        .unwrap_or_else(|_| Err(format!("timed out after {} s", bound.as_secs_f64())))
}

/// The probe itself, with no time bound — only [`probe_watchdog_within`] calls it.
async fn probe_watchdog_unbounded(addr: SocketAddr, token: &str) -> Result<Option<f64>, String> {
    let mut stream = TcpStream::connect(addr)
        .await
        .map_err(|e| format!("connect: {e}"))?;
    let request = format!(
        "GET {} HTTP/1.0\r\nHost: {}\r\n{}: {token}\r\nConnection: close\r\n\r\n",
        crate::WATCHDOG_PATH,
        addr.ip(),
        crate::WATCHDOG_TOKEN_HEADER,
    );
    stream
        .write_all(request.as_bytes())
        .await
        .map_err(|e| format!("write: {e}"))?;
    let mut buf = Vec::new();
    stream
        .read_to_end(&mut buf)
        .await
        .map_err(|e| format!("read: {e}"))?;
    parse_watchdog_response(&buf)
}

/// Parses a raw HTTP/1.x watchdog response — see [`probe_watchdog_once`]. A non-200
/// status is an error (it was a 401/503 from the admin gate that blinded the watchdog).
fn parse_watchdog_response(raw: &[u8]) -> Result<Option<f64>, String> {
    let text = std::str::from_utf8(raw).map_err(|_| "response is not UTF-8".to_owned())?;
    let (head, body) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| "truncated response".to_owned())?;
    let status = head.lines().next().unwrap_or_default();
    if status.split_whitespace().nth(1) != Some("200") {
        return Err(format!("unexpected status: {status}"));
    }
    let value: serde_json::Value =
        serde_json::from_str(body.trim()).map_err(|e| format!("body is not JSON: {e}"))?;
    match value.get("oldest_generation_secs") {
        Some(serde_json::Value::Null) => Ok(None),
        Some(v) => v
            .as_f64()
            .map(Some)
            .ok_or_else(|| format!("oldest_generation_secs is not a number: {v}")),
        None => Err("no oldest_generation_secs field".to_owned()),
    }
}

/// A single `/health` probe: connect, send a minimal HTTP/1.0 request, check
/// the status line for `200`. Any connection/parse failure just means "not
/// up yet" — not an error worth propagating.
async fn probe_health_once(addr: SocketAddr) -> bool {
    let Ok(mut stream) = TcpStream::connect(addr).await else {
        return false;
    };
    let request = format!(
        "GET /health HTTP/1.0\r\nHost: {}\r\nConnection: close\r\n\r\n",
        addr.ip()
    );
    if stream.write_all(request.as_bytes()).await.is_err() {
        return false;
    }
    let mut buf = [0_u8; 32];
    let Ok(n) = stream.read(&mut buf).await else {
        return false;
    };
    std::str::from_utf8(&buf[..n]).is_ok_and(|s| s.contains(" 200 "))
}

/// Operator-initiated stop: forward `SIGTERM` to the worker and wait for it
/// to exit. Never escalates to `SIGKILL` — the worker's own drain/shutdown
/// path owns that budget (`startup::serving_stop_bound`: drain inside a load wait of up to
/// `LOAD_SETTLE_CAP`, then `teardown_budget(n)` and the exit wait); killing here would skip
/// `OvCbEngine::Drop` for a worker that was shutting down cleanly.
async fn shut_down_child(child: &mut Child) -> AttemptOutcome {
    send_signal(child, libc::SIGTERM);
    match child.wait().await {
        Ok(status) if status.success() => {
            tracing::info!("SIGTERM received — worker stopped cleanly, exiting (no respawn)");
        }
        Ok(status) if is_ffi_stalled_exit(status.code()) => {
            tracing::warn!(
                code = ?status.code(),
                "SIGTERM received — worker stopped, but OpenVINO calls were still running; it \
                 left without destructors (no respawn)"
            );
        }
        Ok(status) => {
            // A non-zero code or a fatal signal (e.g. SIGSEGV from teardown)
            // means the worker's shutdown path crashed; record it rather than
            // reporting a clean stop.
            tracing::error!(
                code = ?status.code(),
                signal = ?status.signal(),
                "SIGTERM received — worker did NOT exit cleanly during shutdown (no respawn)"
            );
        }
        Err(e) => {
            tracing::warn!(error = %e, "error waiting for worker to exit during shutdown");
        }
    }
    AttemptOutcome::ShuttingDown
}

/// Hang-recovery: a worker that never became healthy is presumed wedged
/// (GPU.1 left dirty by a prior crash that bypassed `OvCbEngine::Drop`).
/// `SIGTERM` first, then `SIGKILL` after `grace` if it's still alive — the
/// one sanctioned `SIGKILL` in this codebase: the worker never reached a
/// state worth preserving, and the goal shifts to freeing the port/GPU for
/// the next attempt.
///
/// D-04 S8: each stop budget the worker declares after the `SIGTERM` ([`declared_stop_budgets`]
/// in `progress`: it is waiting for a running load, or tearing down) moves the `SIGKILL` to that
/// budget plus slack from when it is read, never earlier; a worker that never declares (wedged
/// beyond answering) keeps `grace`, at least [`MIN_KILL_GRACE`] so a declaration is read even with
/// `hang_kill_grace_secs: 0` (Fable S8 review, 1).
///
/// Returns whether the worker left without its teardown (`SIGKILL`ed, or its own exit says so —
/// [`left_without_teardown`]): a worker that stopped cleanly inside the grace does not arm the
/// next spawn's first-GPU-load deadline (Fable S8 review, 4).
async fn hang_kill_child(child: &mut Child, grace: Duration, progress: &Path) -> bool {
    let mut seen = 0;
    send_signal(child, libc::SIGTERM);
    let sent = Instant::now();
    let mut kill_at = sent + grace.max(MIN_KILL_GRACE);
    loop {
        let left = kill_at.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break;
        }
        if let Ok(status) =
            tokio::time::timeout(left.min(Duration::from_millis(500)), child.wait()).await
        {
            let no_teardown = left_without_teardown(&status);
            log_exit(status);
            return no_teardown;
        }
        let budgets = declared_stop_budgets(&std::fs::read_to_string(progress).unwrap_or_default());
        for budget in budgets.iter().skip(seen) {
            kill_at = kill_at.max(Instant::now() + *budget + KILL_SLACK);
            tracing::warn!(
                stop_budget_secs = budget.as_secs(),
                "worker is inside its own stop budget - waiting for it before SIGKILL"
            );
        }
        seen = seen.max(budgets.len());
    }
    tracing::error!(
        waited_secs = sent.elapsed().as_secs(),
        "hung worker still alive after grace — SIGKILL"
    );
    send_signal(child, libc::SIGKILL);
    let _ = child.wait().await;
    true
}

/// Sends `sig` to `child` if it's still known to be running.
fn send_signal(child: &Child, sig: libc::c_int) {
    if let Some(pid) = child.id() {
        // Safety: `pid` comes from `Child::id()` on our own live child handle,
        // read immediately before the call — this process is the parent that
        // spawned it, so the PID cannot have been reused by an unrelated
        // process out from under us between the read and the `kill(2)`.
        #[allow(clippy::cast_possible_wrap)]
        let pid = pid as libc::pid_t;
        unsafe {
            libc::kill(pid, sig);
        }
    }
}

/// Whether a worker exit code is the deliberate GPU-poisoned exit.
fn is_poison_exit(code: Option<i32>) -> bool {
    code == Some(i32::from(crate::startup::EXIT_GPU_POISONED))
}

/// Whether a worker exit code is the "left with `OpenVINO` calls in flight" exit.
fn is_ffi_stalled_exit(code: Option<i32>) -> bool {
    code == Some(i32::from(crate::startup::EXIT_FFI_STALLED))
}

/// Whether a worker exit code is the deliberate fatal-config exit.
fn is_config_fatal_exit(code: Option<i32>) -> bool {
    code == Some(i32::from(crate::startup::EXIT_CONFIG_FATAL))
}

/// Logs a worker exit and decides whether the supervisor may respawn it.
fn exit_outcome(status: std::io::Result<ExitStatus>) -> AttemptOutcome {
    let code = status.as_ref().ok().and_then(ExitStatus::code);
    log_exit(status);
    classify_exit(code)
}

/// The respawn decision for a worker that exited on its own with `code`.
fn classify_exit(code: Option<i32>) -> AttemptOutcome {
    if is_config_fatal_exit(code) {
        AttemptOutcome::ConfigFatal
    } else if is_ffi_stalled_exit(code) {
        AttemptOutcome::FfiStalled
    } else {
        AttemptOutcome::Failed
    }
}

fn log_exit(status: std::io::Result<ExitStatus>) {
    match status {
        Ok(status) if is_poison_exit(status.code()) => {
            tracing::error!(
                code = ?status.code(),
                "worker exited on purpose: its GPU context was poisoned (out of resources) — \
                 restarting it"
            );
        }
        Ok(status) if is_ffi_stalled_exit(status.code()) => {
            tracing::error!(
                code = ?status.code(),
                "worker exited with OpenVINO calls still running (it left without destructors) — \
                 restarting it; counted against the watchdog restart cap"
            );
        }
        Ok(status) if is_config_fatal_exit(status.code()) => {
            tracing::error!(
                code = ?status.code(),
                "worker exited on a fatal configuration error — NOT restarting (a respawn \
                 would fail identically)"
            );
        }
        Ok(status) => {
            tracing::error!(code = ?status.code(), signal = ?status.signal(), "worker exited");
        }
        Err(e) => tracing::error!(error = %e, "error waiting for worker"),
    }
}

/// `SIGTERM`→`SIGKILL` grace for a watchdog kill of a *healthy* worker: at least the worker's
/// drain + the teardown floor (`SHUTDOWN_DEADLINE`) + exit wait plus slack. A longer stop (a load
/// still running, several engines) is covered by the worker's stop-budget declarations, which
/// [`hang_kill_child`] honours (D-04 S8). Its other models may be streaming; killing it
/// mid-drain with `hang_kill_grace_secs` (10 s by default) SIGKILLs a process with healthy GPU
/// work in flight, the act that left the B60 hung on 2026-10-23
/// (a copy-engine CAT error, then the next GPU load hung until an xe reset). A worker that never became healthy
/// has nothing to drain and keeps the configured grace. Includes the worker's exit wait for
/// in-flight `OpenVINO` calls (`EXIT_WAIT`, step 6a).
fn watchdog_kill_grace(cfg: &SupervisorConfig) -> Duration {
    let drain = crate::startup::DRAIN_DEADLINE
        + crate::startup::SHUTDOWN_DEADLINE
        + crate::startup::EXIT_WAIT;
    Duration::from_secs(cfg.hang_kill_grace_secs).max(drain + Duration::from_secs(5))
}

/// Window over which watchdog kills are capped at `max_restarts`: room for that many full
/// watchdog cycles (backoff, health wait, the hang ceiling, a poll, the kill grace, and the
/// worker's stop wait for a running load — D-04 S8: a declared budget can move the kill by up to
/// `LOAD_SETTLE_CAP`). The plain
/// `restart_window_secs` (600 s by default) is shorter than one such cycle, so every attempt
/// aged out before the next and a wedging request looped forever.
fn watchdog_kill_window(cfg: &SupervisorConfig) -> Duration {
    let cycle = Duration::from_secs(
        cfg.backoff_max_secs
            + cfg.health_timeout_secs
            + cfg.watchdog_hang_ceiling_secs
            + cfg.watchdog_poll_secs,
    ) + watchdog_kill_grace(cfg)
        + crate::startup::LOAD_SETTLE_CAP;
    let n = u32::try_from(cfg.max_restarts).unwrap_or(u32::MAX);
    cycle
        .saturating_mul(n)
        .max(Duration::from_secs(cfg.restart_window_secs))
}

/// Drops restart timestamps older than the rolling window — an old,
/// unrelated crash shouldn't count against a fresh incident. `now` is
/// injected so the decision is testable without real sleeps.
fn prune_old_attempts(attempts: &mut Vec<Instant>, cfg: &SupervisorConfig, now: Instant) {
    let window = Duration::from_secs(cfg.restart_window_secs);
    attempts.retain(|&t| now.duration_since(t) < window);
}

/// Backoff before the attempt following `prior_failures` failures — `None`
/// for the very first attempt (no delay), doubling up to
/// [`SupervisorConfig::backoff_max_secs`] afterward.
fn backoff_delay(prior_failures: usize, cfg: &SupervisorConfig) -> Option<Duration> {
    if prior_failures == 0 {
        return None;
    }
    let shift = (prior_failures - 1).min(32);
    let secs = cfg.backoff_base_secs.saturating_mul(1_u64 << shift);
    Some(Duration::from_secs(secs.min(cfg.backoff_max_secs)))
}

#[cfg(test)]
mod tests {
    use super::{
        AttemptOutcome, KILL_SLACK, backoff_delay, classify_exit, declared_stop_budgets,
        gpu_load_progress, health_deadline, is_config_fatal_exit, is_ffi_stalled_exit,
        is_poison_exit, left_without_teardown, parse_watchdog_response, preload_kill_grace_for,
        preload_milestones, probe_watchdog_within, prune_old_attempts, watchdog_kill_grace,
        watchdog_kill_window,
    };

    /// P-01: a worker that accepts the connection and then never answers makes the probe an
    /// `Err` within its bound (the watchdog's "blind" ERROR path), not an endless wait.
    #[tokio::test]
    #[allow(clippy::unwrap_used)] // test-only: a failed bind/accept is a broken test host
    async fn watchdog_probe_times_out_on_a_silent_worker() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        // Accept and hold the socket open without ever writing a byte.
        let holder = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            std::future::pending::<()>().await;
            drop(socket);
        });
        let bound = std::time::Duration::from_millis(200);
        let started = std::time::Instant::now();
        let outcome = probe_watchdog_within(addr, "token", bound).await;
        let waited = started.elapsed();
        holder.abort();
        let Err(reason) = outcome else {
            panic!("a silent worker must be an error, got {outcome:?}");
        };
        assert!(reason.contains("timed out after 0.2 s"), "{reason}");
        assert!(waited >= bound, "returned before the bound: {waited:?}");
        assert!(
            waited < std::time::Duration::from_secs(5),
            "bound not honoured: {waited:?}"
        );
    }

    /// A 200 with an age or an explicit `null` is readable; anything else — the admin gate's
    /// 401/503 that blinded the watchdog, a body without the field, garbage — is an error, not
    /// a silent "no hang".
    #[test]
    fn watchdog_response_parsing_separates_unreadable_from_idle() {
        let ok =
            |body: &str| format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n{body}");
        assert_eq!(
            parse_watchdog_response(ok(r#"{"oldest_generation_secs":409.5}"#).as_bytes()),
            Ok(Some(409.5))
        );
        assert_eq!(
            parse_watchdog_response(ok(r#"{"oldest_generation_secs":null}"#).as_bytes()),
            Ok(None)
        );
        let unauthorized = "HTTP/1.1 401 Unauthorized\r\n\r\n{\"error\":{}}";
        assert!(parse_watchdog_response(unauthorized.as_bytes()).is_err());
        let locked = "HTTP/1.1 503 Service Unavailable\r\n\r\n{\"error\":{}}";
        assert!(parse_watchdog_response(locked.as_bytes()).is_err());
        assert!(parse_watchdog_response(ok("{}").as_bytes()).is_err());
        assert!(parse_watchdog_response(ok("not json").as_bytes()).is_err());
        assert!(
            parse_watchdog_response(b"HTTP/1.1 200 OK").is_err(),
            "truncated"
        );
    }

    /// The supervisor recognises the worker's deliberate GPU-poisoned exit (75)
    /// and nothing else (a crash, a signal kill, a clean 0).
    #[test]
    fn only_exit_code_75_is_the_poison_exit() {
        assert!(is_poison_exit(Some(75)));
        assert!(!is_poison_exit(Some(0)));
        assert!(!is_poison_exit(Some(1)));
        assert!(!is_poison_exit(None), "killed by a signal has no code");
    }
    /// Only exit code 78 stops the supervisor; 75 (poison), 1 and signal kills
    /// are all still respawned.
    /// The progress-file protocol: one marker per line.
    #[test]
    fn gpu_load_progress_reads_the_markers() {
        assert_eq!(gpu_load_progress(""), (None, false));
        assert_eq!(
            gpu_load_progress("gpu_load_started 600\n"),
            (Some(600), false)
        );
        assert_eq!(
            gpu_load_progress("gpu_load_started 0\ngpu_load_done\n"),
            (Some(0), true)
        );
        // D-04 S8: a torn `gpu_load_started 6` (of `600`) is not read as 6 s.
        assert_eq!(gpu_load_progress("gpu_load_started 6"), (None, false));
        // A pre-S8 worker's bare marker: started, no deadline reported.
        assert_eq!(gpu_load_progress("gpu_load_started\n"), (Some(0), false));
    }

    /// D-04 S8: the worker's declared stop budgets, in order, complete lines only.
    #[test]
    fn declared_stop_budgets_read_complete_lines() {
        assert!(declared_stop_budgets("").is_empty());
        assert!(declared_stop_budgets("preload_done\nstop_budget 6").is_empty());
        assert_eq!(
            declared_stop_budgets("gpu_load_started 600\nstop_budget 642\nstop_budget 16\n"),
            vec![Duration::from_secs(642), Duration::from_secs(16)]
        );
    }

    /// D-04 S8: a preload kill waits the worker's whole stop wait only when the running load has
    /// no deadline of its own; an overdue one leaves in seconds. Never below the configured grace.
    #[test]
    fn preload_kill_grace_follows_the_workers_stop_bound() {
        let cfg = test_cfg();
        // Overdue: 5 s overdue grace + 0.25 s poll + teardown_budget(1) 10 s + 0.25 + 3 + 5 slack.
        assert_eq!(
            preload_kill_grace_for(&cfg, 1, false),
            crate::startup::preload_stop_bound(1, false) + KILL_SLACK
        );
        assert_eq!(preload_kill_grace_for(&cfg, 1, false).as_secs(), 23);
        // Uncapped: the 600 s stop wait, so SIGKILL only past the worker's own cap.
        let waits = preload_kill_grace_for(&cfg, 3, true);
        assert!(waits > crate::startup::LOAD_SETTLE_CAP + crate::startup::teardown_budget(3));
        let long = SupervisorConfig {
            hang_kill_grace_secs: 120,
            ..test_cfg()
        };
        assert_eq!(
            preload_kill_grace_for(&long, 1, false),
            Duration::from_secs(120)
        );
    }

    /// G-1: complete lines only — a torn `preload_started 6` (of `600`) is not read as 6 s.
    #[test]
    fn preload_milestones_ignore_a_torn_last_line() {
        use super::PreloadMilestone::{Done, Started};
        assert!(preload_milestones("").is_empty());
        assert!(preload_milestones("preload_started 6").is_empty());
        assert_eq!(
            preload_milestones(
                "gpu_load_started\npreload_started 600\npreload_done\npreload_started 6"
            ),
            vec![
                Started {
                    allowance_secs: 600
                },
                Done
            ]
        );
        assert_eq!(
            preload_milestones("preload_started 0\ngarbage\n"),
            vec![Started { allowance_secs: 0 }]
        );
    }

    /// G-1: the deadline only ever moves later; a finished preload gives `health` again; a preload
    /// with no load deadline follows the uncapped limit, or has none when that is unset.
    #[test]
    fn health_deadline_follows_preload_progress() {
        use super::PreloadMilestone::{Done, Started};
        let t0 = Instant::now();
        let s = Duration::from_secs;
        let h = s(90);
        assert_eq!(health_deadline(t0, h, &[], None), Some(t0 + h));
        // A later preload with a smaller allowance never shortens an earlier extension.
        let big_then_small = [
            (
                Started {
                    allowance_secs: 600,
                },
                t0 + s(5),
            ),
            (Started { allowance_secs: 60 }, t0 + s(10)),
        ];
        assert_eq!(
            health_deadline(t0, h, &big_then_small, None),
            Some(t0 + s(695))
        );
        assert_eq!(
            health_deadline(
                t0,
                h,
                &[(
                    Started {
                        allowance_secs: 600
                    },
                    t0 + s(5)
                )],
                None
            ),
            Some(t0 + s(5) + s(600) + h)
        );
        // Done moves it to `health` after the done instant — but never back before an earlier
        // extension.
        let seen = [
            (
                Started {
                    allowance_secs: 600,
                },
                t0 + s(5),
            ),
            (Done, t0 + s(100)),
        ];
        assert_eq!(health_deadline(t0, h, &seen, None), Some(t0 + s(695)));
        let quick = [
            (Started { allowance_secs: 60 }, t0 + s(1)),
            (Done, t0 + s(200)),
        ];
        assert_eq!(health_deadline(t0, h, &quick, None), Some(t0 + s(290)));
        // No load deadline: the uncapped limit, or none at all.
        let open = [(Started { allowance_secs: 0 }, t0 + s(2))];
        assert_eq!(
            health_deadline(t0, h, &open, Some(s(600))),
            Some(t0 + s(692))
        );
        assert_eq!(health_deadline(t0, h, &open, None), None);
        let open_then_done = [
            (Started { allowance_secs: 0 }, t0 + s(2)),
            (Done, t0 + s(300)),
        ];
        assert_eq!(
            health_deadline(t0, h, &open_then_done, None),
            Some(t0 + s(390))
        );
    }

    /// G-1: the first-GPU-load deadline still reads only its own markers.
    #[test]
    fn gpu_load_progress_ignores_preload_lines() {
        assert_eq!(
            gpu_load_progress("preload_started 600\ngpu_load_started 600\npreload_done\n"),
            (Some(600), false)
        );
    }

    /// Only 75, 76 and a signal count as a no-teardown exit; a clean 0, an error 1 and 78 do not.
    #[test]
    fn no_teardown_exits_are_75_76_and_signals() {
        use std::os::unix::process::ExitStatusExt as _;
        let code = |c: i32| Ok(std::process::ExitStatus::from_raw(c << 8));
        assert!(left_without_teardown(&code(75)));
        assert!(left_without_teardown(&code(76)));
        assert!(!left_without_teardown(&code(0)));
        assert!(!left_without_teardown(&code(1)));
        assert!(!left_without_teardown(&code(78)));
        // Killed by SIGKILL (9): raw wait status = signal number.
        assert!(left_without_teardown(&Ok(
            std::process::ExitStatus::from_raw(9)
        )));
    }

    #[test]
    fn exit_code_76_is_respawned_and_counted_as_a_watchdog_kill() {
        assert_eq!(classify_exit(Some(76)), AttemptOutcome::FfiStalled);
        assert_eq!(classify_exit(Some(78)), AttemptOutcome::ConfigFatal);
        assert_eq!(classify_exit(Some(75)), AttemptOutcome::Failed);
        assert_eq!(classify_exit(Some(1)), AttemptOutcome::Failed);
        assert_eq!(classify_exit(Some(0)), AttemptOutcome::Failed);
        assert_eq!(classify_exit(None), AttemptOutcome::Failed);
        assert!(!is_ffi_stalled_exit(Some(75)));
    }

    #[test]
    fn only_exit_code_78_is_the_config_fatal_exit() {
        assert!(is_config_fatal_exit(Some(78)));
        assert!(!is_config_fatal_exit(Some(75)));
        assert!(!is_config_fatal_exit(Some(1)));
        assert!(!is_config_fatal_exit(Some(0)));
        assert!(
            !is_config_fatal_exit(None),
            "killed by a signal has no code"
        );
    }

    use crate::model_manager::SupervisorConfig;
    use std::time::{Duration, Instant};

    fn test_cfg() -> SupervisorConfig {
        SupervisorConfig {
            backoff_base_secs: 5,
            backoff_max_secs: 60,
            max_restarts: 5,
            restart_window_secs: 600,
            health_timeout_secs: 90,
            hang_kill_grace_secs: 10,
            watchdog_poll_secs: 30,
            watchdog_hang_ceiling_secs: 0,
            gpu_first_load_deadline_secs: 0,
            gpu_probe_deadline_secs: 0,
        }
    }

    #[test]
    fn first_attempt_has_no_backoff() {
        assert_eq!(backoff_delay(0, &test_cfg()), None);
    }

    #[test]
    fn backoff_doubles_up_to_the_cap() {
        let cfg = test_cfg();
        assert_eq!(backoff_delay(1, &cfg), Some(Duration::from_secs(5)));
        assert_eq!(backoff_delay(2, &cfg), Some(Duration::from_secs(10)));
        assert_eq!(backoff_delay(3, &cfg), Some(Duration::from_secs(20)));
        assert_eq!(backoff_delay(4, &cfg), Some(Duration::from_secs(40)));
        // Would be 80s uncapped — clamped to backoff_max_secs.
        assert_eq!(backoff_delay(5, &cfg), Some(Duration::from_mins(1)));
        assert_eq!(backoff_delay(50, &cfg), Some(Duration::from_mins(1)));
    }

    /// Fable review N1: a watchdog kill waits out the worker's own drain + teardown before
    /// SIGKILL, whatever `hang_kill_grace_secs` says; a larger configured grace still wins.
    #[test]
    fn watchdog_kill_grace_covers_the_workers_drain() {
        let cfg = test_cfg();
        // DRAIN 30 + SHUTDOWN 10 + EXIT_WAIT 3 + 5 s slack.
        assert_eq!(
            watchdog_kill_grace(&cfg),
            crate::startup::DRAIN_DEADLINE
                + crate::startup::SHUTDOWN_DEADLINE
                + crate::startup::EXIT_WAIT
                + Duration::from_secs(5)
        );
        assert_eq!(watchdog_kill_grace(&cfg), Duration::from_secs(48));
        let long = SupervisorConfig {
            hang_kill_grace_secs: 120,
            ..test_cfg()
        };
        assert_eq!(watchdog_kill_grace(&long), Duration::from_secs(120));
    }

    /// Fable review N3: with the shipped 600 s ceiling, `max_restarts` watchdog cycles fit in
    /// the kill window, so a request that wedges every respawn is stopped; the plain 600 s
    /// window could never hold two of them.
    #[test]
    fn watchdog_kill_window_holds_max_restarts_cycles() {
        let cfg = SupervisorConfig {
            watchdog_hang_ceiling_secs: 600,
            ..test_cfg()
        };
        let window = watchdog_kill_window(&cfg);
        // One cycle: backoff 60 + health 90 + ceiling 600 + poll 30 + grace 48 + load wait 600
        // = 1428 s.
        assert_eq!(window, Duration::from_mins(119)); // 5 × 1428 s
        assert!(window > Duration::from_secs(cfg.restart_window_secs) * 5);
        // Disabled watchdog: never below the plain window.
        assert_eq!(
            watchdog_kill_window(&SupervisorConfig {
                max_restarts: 0,
                ..test_cfg()
            }),
            Duration::from_secs(600)
        );
    }

    #[test]
    fn prune_drops_attempts_outside_the_window() {
        let cfg = test_cfg();
        let now = Instant::now();
        let mut attempts = vec![
            now.checked_sub(Duration::from_secs(700)).unwrap_or(now), // outside the 600s window
            now.checked_sub(Duration::from_secs(100)).unwrap_or(now), // inside
            now.checked_sub(Duration::from_secs(1)).unwrap_or(now),   // inside
        ];
        prune_old_attempts(&mut attempts, &cfg, now);
        assert_eq!(attempts.len(), 2);
    }

    #[test]
    fn prune_keeps_empty_history_empty() {
        let cfg = test_cfg();
        let mut attempts = Vec::new();
        prune_old_attempts(&mut attempts, &cfg, Instant::now());
        assert!(attempts.is_empty());
    }

    // ── C27 respawn restore ──────────────────────────────────────────────────

    const W: Duration = Duration::from_secs(600);

    #[test]
    fn restore_plan_hands_over_what_was_ready_at_the_crash() {
        let mut st = super::RestoreState::default();
        let drops = super::plan_restore(
            "model_ready a\nmodel_ready b\nmodel_unloaded b\nmodel_ready c\n",
            &mut st,
            Instant::now(),
            W,
        );
        assert!(drops.is_empty());
        assert_eq!(st.handed, Some(vec!["a".to_owned(), "c".to_owned()]));
    }

    #[test]
    fn restore_plan_ignores_a_torn_last_line_and_other_lines() {
        let mut st = super::RestoreState::default();
        super::plan_restore(
            "preload_started 600\npreload_done\nmodel_ready a\nmodel_ready b",
            &mut st,
            Instant::now(),
            W,
        );
        assert_eq!(st.handed, Some(vec!["a".to_owned()]));
    }

    #[test]
    fn restore_plan_drops_a_restore_that_never_reached_ready_and_carries_the_rest() {
        let mut st = super::RestoreState {
            handed: Some(vec!["a".to_owned(), "b".to_owned()]),
            ..Default::default()
        };
        let drops = super::plan_restore("restore_started a\n", &mut st, Instant::now(), W);
        assert_eq!(
            drops,
            vec![super::RestoreDrop::RestoreNeverReady("a".to_owned())]
        );
        assert_eq!(st.handed, Some(vec!["b".to_owned()]));
    }

    #[test]
    fn restore_plan_drops_a_restored_model_resident_at_a_second_crash() {
        let mut st = super::RestoreState::default();
        let t0 = Instant::now();
        super::plan_restore("model_ready a\n", &mut st, t0, W);
        let drops = super::plan_restore("restore_started a\nmodel_ready a\n", &mut st, t0, W);
        assert_eq!(
            drops,
            vec![super::RestoreDrop::ResidentAtSecondCrash("a".to_owned())]
        );
        assert_eq!(st.handed, Some(Vec::new()));
    }

    #[test]
    fn restore_plan_restores_a_model_once_per_window() {
        let mut st = super::RestoreState::default();
        let t0 = Instant::now();
        super::plan_restore("model_ready a\n", &mut st, t0, W);
        // Restored, then unloaded on purpose: not handed over.
        super::plan_restore(
            "restore_started a\nmodel_ready a\nmodel_unloaded a\n",
            &mut st,
            t0,
            W,
        );
        assert_eq!(st.handed, Some(Vec::new()));
        // Loaded again by an operator and resident at a third crash, inside the window.
        let drops =
            super::plan_restore("model_ready a\n", &mut st, t0 + Duration::from_secs(60), W);
        assert_eq!(
            drops,
            vec![super::RestoreDrop::RestoredRecently("a".to_owned())]
        );
        assert_eq!(st.handed, Some(Vec::new()));
        // After the window it may be restored again.
        let drops = super::plan_restore(
            "model_ready a\n",
            &mut st,
            t0 + W + Duration::from_secs(1),
            W,
        );
        assert!(drops.is_empty());
        assert_eq!(st.handed, Some(vec!["a".to_owned()]));
    }

    /// C27 F1: a restored model resident at a crash after the restart window of its hand-over is
    /// an ordinary resident model: restored again.
    #[test]
    fn restore_plan_restores_a_model_resident_at_a_crash_after_the_window() {
        let mut st = super::RestoreState::default();
        let t0 = Instant::now();
        super::plan_restore("model_ready a\n", &mut st, t0, W);
        let drops = super::plan_restore(
            "restore_started a\nmodel_ready a\n",
            &mut st,
            t0 + W + Duration::from_secs(1),
            W,
        );
        assert!(drops.is_empty(), "{drops:?}");
        assert_eq!(st.handed, Some(vec!["a".to_owned()]));
    }

    /// C27 F4: a handed-over model an operator loaded before its restore ran counts no restore:
    /// resident at the next crash, it is handed over again, not dropped as restored recently.
    #[test]
    fn restore_plan_counts_only_restores_that_started() {
        let mut st = super::RestoreState::default();
        let t0 = Instant::now();
        super::plan_restore("model_ready a\n", &mut st, t0, W);
        let drops = super::plan_restore("model_ready a\n", &mut st, t0 + Duration::from_secs(5), W);
        assert!(drops.is_empty(), "{drops:?}");
        assert_eq!(st.handed, Some(vec!["a".to_owned()]));
    }

    /// The ids as a handover (`Some`).
    #[allow(clippy::unnecessary_wraps)]
    fn handover(ids: &[&str]) -> Option<Vec<String>> {
        Some(ids.iter().map(|&id| id.to_owned()).collect())
    }

    /// A fresh worker that finished its preload: the preload models Ready at the crash are in the
    /// handed set, with the runtime ones.
    #[test]
    fn restore_plan_hands_over_preload_models_ready_at_the_crash() {
        let mut st = super::RestoreState::default();
        let drops = super::plan_restore(
            "preload_phase_started\npreload_model p\nmodel_ready p\npreload_model q\n\
             model_ready q\npreload_phase_done\nmodel_ready r\n",
            &mut st,
            Instant::now(),
            W,
        );
        assert!(drops.is_empty(), "{drops:?}");
        assert_eq!(st.handed, handover(&["p", "q", "r"]));
    }

    /// The 2026-11-16 16:40 shape: preload 9b + tts, 9b unloaded and 4b loaded at runtime (e5
    /// too) — the next worker targets tts, e5, 4b, not 9b.
    #[test]
    fn restore_plan_the_1640_shape_hands_over_the_runtime_swap_not_the_unloaded_preload() {
        let mut st = super::RestoreState::default();
        super::plan_restore(
            "preload_phase_started\npreload_model 9b\nmodel_ready 9b\npreload_model tts\n\
             model_ready tts\npreload_phase_done\nmodel_ready e5\nmodel_unloaded 9b\n\
             model_ready 4b\n",
            &mut st,
            Instant::now(),
            W,
        );
        assert_eq!(st.handed, handover(&["tts", "e5", "4b"]));
    }

    /// A fresh worker that crashed inside its preload phase hands nothing on: the next worker is
    /// a fresh start (full preload), not one targeting the half-loaded set.
    #[test]
    fn restore_plan_a_crash_during_a_fresh_preload_means_a_full_preload() {
        let mut st = super::RestoreState::default();
        let drops = super::plan_restore(
            "preload_phase_started\npreload_model p\nmodel_ready p\npreload_model q\n",
            &mut st,
            Instant::now(),
            W,
        );
        assert!(drops.is_empty());
        assert_eq!(st.handed, None);
    }

    /// A fresh worker that never had a model Ready hands nothing on (full preload), however its
    /// preload ended.
    #[test]
    fn restore_plan_a_fresh_worker_with_nothing_ever_ready_means_a_full_preload() {
        let mut st = super::RestoreState::default();
        super::plan_restore(
            "preload_phase_started\npreload_model p\npreload_phase_done\n",
            &mut st,
            Instant::now(),
            W,
        );
        assert_eq!(st.handed, None);
        // Without any preload lines (a worker with an empty preload) too.
        super::plan_restore("", &mut st, Instant::now(), W);
        assert_eq!(st.handed, None);
    }

    /// Every model unloaded by the operator before the crash: the next worker targets nothing
    /// (`Some([])`: no preload), not a fresh start.
    #[test]
    fn restore_plan_everything_unloaded_before_the_crash_hands_over_an_empty_set() {
        let mut st = super::RestoreState::default();
        super::plan_restore(
            "preload_phase_started\npreload_model p\nmodel_ready p\npreload_phase_done\n\
             model_unloaded p\n",
            &mut st,
            Instant::now(),
            W,
        );
        assert_eq!(st.handed, Some(Vec::new()));
        // A respawned worker handed `[]` that crashes with nothing Ready stays at `[]` (never
        // falls back to the full preload once the set is known).
        super::plan_restore(
            "preload_phase_started\npreload_phase_done\n",
            &mut st,
            Instant::now(),
            W,
        );
        assert_eq!(st.handed, Some(Vec::new()));
    }

    /// A respawned worker's finished preload phase: a handed preload model it got Ready stays, one
    /// it did not (refused, failed) is not carried, a runtime restore not started yet is.
    #[test]
    fn restore_plan_a_handed_preload_model_not_ready_after_the_phase_is_not_carried() {
        let mut st = super::RestoreState {
            handed: handover(&["p", "q", "r"]),
            ..Default::default()
        };
        let drops = super::plan_restore(
            "preload_phase_started\npreload_model p\nmodel_ready p\npreload_model q\n\
             preload_phase_done\n",
            &mut st,
            Instant::now(),
            W,
        );
        assert!(drops.is_empty(), "{drops:?}");
        assert_eq!(st.handed, handover(&["r", "p"]));
    }

    /// A respawned worker that crashed inside its preload phase carries its whole handed set (the
    /// preload model being loaded and those not reached included).
    #[test]
    fn restore_plan_a_crash_during_a_respawn_preload_carries_the_handed_set() {
        let mut st = super::RestoreState {
            handed: handover(&["p", "q", "s", "r"]),
            ..Default::default()
        };
        let drops = super::plan_restore(
            "preload_phase_started\npreload_model p\nmodel_ready p\npreload_model q\n",
            &mut st,
            Instant::now(),
            W,
        );
        assert!(drops.is_empty(), "{drops:?}");
        assert_eq!(st.handed, handover(&["q", "s", "r", "p"]));
    }

    /// No crash-loop rule for preload models: one Ready at a second crash inside the window is
    /// handed over again (its preload phase writes no `restore_started`), while a restored
    /// runtime model resident at that crash is dropped as before.
    #[test]
    fn restore_plan_a_preload_model_ready_at_a_second_crash_is_handed_over_again() {
        let mut st = super::RestoreState::default();
        let t0 = Instant::now();
        let first = "preload_phase_started\npreload_model p\nmodel_ready p\npreload_phase_done\n\
                     model_ready r\n";
        super::plan_restore(first, &mut st, t0, W);
        assert_eq!(st.handed, handover(&["p", "r"]));
        let drops = super::plan_restore(
            "preload_phase_started\npreload_model p\nmodel_ready p\npreload_phase_done\n\
             restore_started r\nmodel_ready r\n",
            &mut st,
            t0 + Duration::from_secs(5),
            W,
        );
        assert_eq!(
            drops,
            vec![super::RestoreDrop::ResidentAtSecondCrash("r".to_owned())]
        );
        assert_eq!(st.handed, handover(&["p"]));
    }

    /// Preload P and runtime r loaded; an `rv load` of big evicts P and the worker crashes before
    /// big is Ready: P stays in the set (a respawn must not come back with less), big is not in it.
    #[test]
    fn restore_plan_keeps_the_victim_of_a_load_that_crashed_before_ready() {
        let mut st = super::RestoreState::default();
        super::plan_restore(
            "preload_phase_started\npreload_model p\nmodel_ready p\npreload_phase_done\n\
             model_ready r\nload_started big\nmodel_unloaded p\n",
            &mut st,
            Instant::now(),
            W,
        );
        assert_eq!(st.handed, handover(&["p", "r"]));
    }

    /// The same load reaching Ready: its victim is really out.
    #[test]
    fn restore_plan_drops_the_victim_of_a_load_that_reached_ready() {
        let mut st = super::RestoreState::default();
        super::plan_restore(
            "model_ready p\nload_started big\nmodel_unloaded p\nmodel_ready big\n\
             load_ended big\n",
            &mut st,
            Instant::now(),
            W,
        );
        assert_eq!(st.handed, handover(&["big"]));
    }

    /// A resize (and a KV-wedge evict + reload, which writes the same lines) evicts the model
    /// itself: a crash before the reload is Ready keeps it; once Ready it is resident as before.
    #[test]
    fn restore_plan_keeps_a_model_whose_resize_or_wedge_reload_crashed() {
        let mut st = super::RestoreState::default();
        // Outer window (resize / wedge), self-eviction, inner window of the reload, crash.
        super::plan_restore(
            "model_ready m\nload_started m\nmodel_unloaded m\nload_started m\n",
            &mut st,
            Instant::now(),
            W,
        );
        assert_eq!(st.handed, handover(&["m"]));
        let mut st = super::RestoreState::default();
        super::plan_restore(
            "model_ready m\nload_started m\nmodel_unloaded m\nload_started m\nmodel_ready m\n\
             load_ended m\nload_ended m\n",
            &mut st,
            Instant::now(),
            W,
        );
        assert_eq!(st.handed, handover(&["m"]));
    }

    /// A load that returned without Ready (failed, no crash) really evicted its victim: out (the
    /// worker kept serving without it); an unload after the window closed is an ordinary unload;
    /// a later event of the victim decides.
    #[test]
    fn restore_plan_victims_after_a_failed_load_and_later_events() {
        let mut st = super::RestoreState::default();
        super::plan_restore(
            "model_ready v\nmodel_ready w\nmodel_ready x\nload_started big\n\
             model_unloaded v\nmodel_unloaded x\nload_ended big\nmodel_unloaded w\n\
             model_ready x\n",
            &mut st,
            Instant::now(),
            W,
        );
        assert_eq!(st.handed, handover(&["x"]));
        // Nested windows (resize / wedge): the inner reload failed, the outer is still open at
        // the crash: kept. Once the outer one returned too: out.
        let mut st = super::RestoreState::default();
        super::plan_restore(
            "model_ready m\nload_started m\nmodel_unloaded m\nload_started m\nload_ended m\n",
            &mut st,
            Instant::now(),
            W,
        );
        assert_eq!(st.handed, handover(&["m"]));
        let mut st = super::RestoreState::default();
        super::plan_restore(
            "model_ready m\nload_started m\nmodel_unloaded m\nload_started m\nload_ended m\n\
             load_ended m\n",
            &mut st,
            Instant::now(),
            W,
        );
        assert_eq!(st.handed, Some(Vec::new()));
        // The victim unloaded again on purpose after the window: out.
        let mut st = super::RestoreState::default();
        super::plan_restore(
            "model_ready v\nload_started big\nmodel_unloaded v\nload_ended big\n\
             model_ready v\nmodel_unloaded v\n",
            &mut st,
            Instant::now(),
            W,
        );
        assert_eq!(st.handed, Some(Vec::new()));
    }

    /// A kept victim persists across a second crash: the respawned worker crashing before its
    /// preload phase even started carries the whole handed set, the victim included.
    #[test]
    fn restore_plan_a_kept_victim_survives_a_second_crash() {
        let mut st = super::RestoreState::default();
        let t0 = Instant::now();
        super::plan_restore(
            "preload_phase_started\npreload_model p\nmodel_ready p\npreload_phase_done\n\
             load_started big\nmodel_unloaded p\n",
            &mut st,
            t0,
            W,
        );
        assert_eq!(st.handed, handover(&["p"]));
        let drops = super::plan_restore("", &mut st, t0 + Duration::from_secs(5), W);
        assert!(drops.is_empty(), "{drops:?}");
        assert_eq!(st.handed, handover(&["p"]));
    }

    /// A file without load-window lines (an older worker) reads as before: an unload is out.
    #[test]
    fn restore_plan_without_window_lines_an_unload_is_out() {
        let mut st = super::RestoreState::default();
        super::plan_restore(
            "model_ready p\nmodel_unloaded p\nmodel_ready big\n",
            &mut st,
            Instant::now(),
            W,
        );
        assert_eq!(st.handed, handover(&["big"]));
    }

    /// N1: a load that failed and closed its window just before an abnormal exit (a poisoned
    /// GPU: exit 75) keeps its victim; a later model event or a clean exit drops it (P14).
    #[test]
    fn restore_plan_keeps_the_victim_of_the_last_failed_load_after_an_abnormal_exit() {
        let lines = "model_ready p\nload_started big\nmodel_unloaded p\nload_ended big\n";
        let mut st = super::RestoreState::default();
        super::plan_restore_after(lines, &mut st, Instant::now(), W, true);
        assert_eq!(st.handed, handover(&["p"]));
        // A clean exit: out.
        let mut st = super::RestoreState::default();
        super::plan_restore_after(lines, &mut st, Instant::now(), W, false);
        assert_eq!(st.handed, Some(Vec::new()));
        // A later unrelated model event (the worker went on serving): out, even after 75.
        let mut st = super::RestoreState::default();
        super::plan_restore_after(
            &format!("{lines}model_ready other\n"),
            &mut st,
            Instant::now(),
            W,
            true,
        );
        assert_eq!(st.handed, handover(&["other"]));
        // A non-model line (the stop's budget) is not an event: still kept.
        let mut st = super::RestoreState::default();
        super::plan_restore_after(
            &format!("{lines}stop_budget 48\n"),
            &mut st,
            Instant::now(),
            W,
            true,
        );
        assert_eq!(st.handed, handover(&["p"]));
    }

    /// M2: a deferred preload model (loaded after the preload phase) whose own
    /// load the crash interrupted is carried, like one loading inside the phase; one whose load
    /// returned without Ready (refused, failed) is not.
    #[test]
    fn restore_plan_a_deferred_preload_interrupted_by_the_crash_is_carried() {
        let mut st = super::RestoreState {
            handed: handover(&["vlm"]),
            ..Default::default()
        };
        super::plan_restore(
            "preload_phase_started\npreload_phase_done\npreload_model vlm\nload_started vlm\n",
            &mut st,
            Instant::now(),
            W,
        );
        assert_eq!(st.handed, handover(&["vlm"]));
        let mut st = super::RestoreState {
            handed: handover(&["vlm"]),
            ..Default::default()
        };
        super::plan_restore(
            "preload_phase_started\npreload_phase_done\npreload_model vlm\nload_started vlm\n\
             load_ended vlm\n",
            &mut st,
            Instant::now(),
            W,
        );
        assert_eq!(st.handed, Some(Vec::new()));
    }

    /// The env handover: `Some([])` is written as `[]`, `None` removes the variable.
    #[test]
    fn the_worker_command_writes_an_empty_set_and_removes_the_variable_for_a_fresh_start() {
        let env_of = |restore: Option<&[String]>| {
            let cmd = super::worker_command(
                std::path::Path::new("/bin/true"),
                &[],
                "t",
                std::path::Path::new("/nonexistent"),
                restore,
            );
            cmd.as_std()
                .get_envs()
                .find(|(k, _)| *k == crate::startup::RESTORE_MODELS_ENV)
                .map(|(_, v)| v.map(|v| v.to_string_lossy().into_owned()))
        };
        assert_eq!(env_of(Some(&[])), Some(Some("[]".to_owned())));
        assert_eq!(
            env_of(Some(&["a".to_owned()])),
            Some(Some(r#"["a"]"#.to_owned()))
        );
        // Removed (an explicit removal shows as a `None` value).
        assert_eq!(env_of(None), Some(None));
    }

    /// C27 F6: a torn multibyte last line costs only that line, not the whole file.
    #[test]
    #[allow(clippy::unwrap_used)]
    fn a_progress_file_with_a_torn_multibyte_last_line_is_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("progress");
        let mut bytes = b"model_ready a\nmodel_ready \xc3".to_vec();
        std::fs::write(&path, &bytes).unwrap();
        let mut st = super::RestoreState::default();
        super::plan_restore(&super::read_progress(&path), &mut st, Instant::now(), W);
        assert_eq!(st.handed, Some(vec!["a".to_owned()]));
        // The same line completed later is read in full.
        bytes.extend_from_slice(b"\xa9\n");
        std::fs::write(&path, &bytes).unwrap();
        let mut st = super::RestoreState::default();
        super::plan_restore(&super::read_progress(&path), &mut st, Instant::now(), W);
        assert_eq!(st.handed, Some(vec!["a".to_owned(), "\u{e9}".to_owned()]));
    }
}
