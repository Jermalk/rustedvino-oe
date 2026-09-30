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
    let config = Config::load(config_path)?;
    let health_addr = config.validated_bind()?;
    let sup_cfg = config.supervisor.clone();

    let exe = std::env::current_exe()?;
    let worker_args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| a != "--supervise")
        .collect();

    run_supervised(&exe, &worker_args, health_addr, &sup_cfg).await
}

/// Outcome of one spawn attempt — decides whether the caller loops again.
enum AttemptOutcome {
    /// The supervisor's own `SIGTERM` fired; the worker was signalled and
    /// waited on. Exit for good, no respawn.
    ShuttingDown,
    /// The worker exited on its own, or was killed for hanging — evaluate
    /// backoff/cap and try again.
    Failed,
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
    let mut attempts: Vec<Instant> = Vec::new();
    let mut sigterm = signal(SignalKind::terminate())?;

    loop {
        prune_old_attempts(&mut attempts, cfg, Instant::now());
        if attempts.len() >= cfg.max_restarts {
            anyhow::bail!(
                "gave up after {} restarts within {}s — worker is crash-looping, not respawning again",
                attempts.len(),
                cfg.restart_window_secs
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

        let mut child = Command::new(exe).args(args).spawn()?;
        tracing::info!(pid = child.id().unwrap_or(0), "worker spawned");

        match run_one_attempt(&mut child, health_addr, cfg, &mut sigterm).await {
            AttemptOutcome::ShuttingDown => return Ok(()),
            AttemptOutcome::Failed => attempts.push(Instant::now()),
        }
    }
}

/// Runs one spawn-to-exit attempt: races the worker becoming healthy against
/// it exiting on its own and against the supervisor's own `SIGTERM`. Once
/// healthy, keeps racing the exit against `SIGTERM` until one happens.
async fn run_one_attempt(
    child: &mut Child,
    health_addr: SocketAddr,
    cfg: &SupervisorConfig,
    sigterm: &mut Signal,
) -> AttemptOutcome {
    let health_timeout = Duration::from_secs(cfg.health_timeout_secs);
    let became_healthy = tokio::select! {
        healthy = wait_for_health(health_addr, health_timeout) => healthy,
        status = child.wait() => {
            log_exit(status);
            return AttemptOutcome::Failed;
        }
        _ = sigterm.recv() => {
            return shut_down_child(child).await;
        }
    };

    if !became_healthy {
        tracing::error!(
            timeout_secs = cfg.health_timeout_secs,
            "worker did not become healthy in time — treating as hung"
        );
        hang_kill_child(child, Duration::from_secs(cfg.hang_kill_grace_secs)).await;
        return AttemptOutcome::Failed;
    }

    tracing::info!("worker healthy");

    // Watchdog disabled (`watchdog_hang_ceiling_secs: 0`): unchanged two-way
    // race. Kept as a separate branch rather than folding a no-op future into
    // the `select!` below — a `0`-second ceiling would otherwise fire
    // immediately on the first real tick.
    if cfg.watchdog_hang_ceiling_secs == 0 {
        return tokio::select! {
            status = child.wait() => {
                log_exit(status);
                AttemptOutcome::Failed
            }
            _ = sigterm.recv() => shut_down_child(child).await,
        };
    }

    tokio::select! {
        status = child.wait() => {
            log_exit(status);
            AttemptOutcome::Failed
        }
        _ = sigterm.recv() => shut_down_child(child).await,
        () = generation_watchdog(health_addr, cfg) => {
            tracing::error!(
                ceiling_secs = cfg.watchdog_hang_ceiling_secs,
                "generation watchdog: oldest in-flight generation exceeded ceiling — treating as hung"
            );
            hang_kill_child(child, Duration::from_secs(cfg.hang_kill_grace_secs)).await;
            AttemptOutcome::Failed
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
/// Never resolves on a probe failure (connection hiccup, malformed body) or
/// on an explicit `null` (nothing generating) — only a *confirmed* age over
/// the ceiling counts, so a transient blip can never kill a healthy worker.
/// Caller must not race this in when `watchdog_hang_ceiling_secs == 0`
/// (disabled).
async fn generation_watchdog(addr: SocketAddr, cfg: &SupervisorConfig) {
    let mut ticker = tokio::time::interval(Duration::from_secs(cfg.watchdog_poll_secs));
    ticker.tick().await; // fires immediately; the worker only just became healthy
    loop {
        ticker.tick().await;
        // Config seconds are always small (a sane ceiling is minutes, not
        // millennia) — never near f64's 52-bit exact-integer boundary.
        #[allow(clippy::cast_precision_loss)]
        let ceiling_secs = cfg.watchdog_hang_ceiling_secs as f64;
        if let Some(age_secs) = probe_watchdog_once(addr).await
            && age_secs > ceiling_secs
        {
            return;
        }
    }
}

/// Single probe of `GET /v1/admin/health/watchdog`: returns
/// `oldest_generation_secs` from the JSON body, or `None` on any
/// connection/parse failure OR an explicit `null` body (nothing generating) —
/// both cases mean "no confirmed hang," same as [`probe_health_once`]'s
/// "any failure just means not-yet-known" philosophy.
async fn probe_watchdog_once(addr: SocketAddr) -> Option<f64> {
    let mut stream = TcpStream::connect(addr).await.ok()?;
    let request = format!(
        "GET /v1/admin/health/watchdog HTTP/1.0\r\nHost: {}\r\nConnection: close\r\n\r\n",
        addr.ip()
    );
    stream.write_all(request.as_bytes()).await.ok()?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await.ok()?;
    let text = std::str::from_utf8(&buf).ok()?;
    let body = text.split_once("\r\n\r\n")?.1;
    let value: serde_json::Value = serde_json::from_str(body.trim()).ok()?;
    value.get("oldest_generation_secs")?.as_f64()
}

/// Polls `/health` on `addr` every 500ms (a plain hand-rolled HTTP/1.0 GET —
/// no HTTP client dependency needed for a one-line status check) until it
/// returns `200`, or `timeout` elapses.
async fn wait_for_health(addr: SocketAddr, timeout: Duration) -> bool {
    tokio::time::timeout(timeout, async {
        loop {
            if probe_health_once(addr).await {
                return;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    })
    .await
    .is_ok()
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
/// path (`startup.rs`'s `shutdown_signal`, bounded by `DRAIN_DEADLINE` +
/// `SHUTDOWN_DEADLINE`) owns that budget; killing here would skip
/// `OvCbEngine::Drop` for a worker that was shutting down cleanly.
async fn shut_down_child(child: &mut Child) -> AttemptOutcome {
    send_signal(child, libc::SIGTERM);
    if let Err(e) = child.wait().await {
        tracing::warn!(error = %e, "error waiting for worker to exit during shutdown");
    }
    tracing::info!("SIGTERM received — worker stopped, exiting (no respawn)");
    AttemptOutcome::ShuttingDown
}

/// Hang-recovery: a worker that never became healthy is presumed wedged
/// (GPU.1 left dirty by a prior crash that bypassed `OvCbEngine::Drop`).
/// `SIGTERM` first, then `SIGKILL` after `grace` if it's still alive — the
/// one sanctioned `SIGKILL` in this codebase: the worker never reached a
/// state worth preserving, and the goal shifts to freeing the port/GPU for
/// the next attempt.
async fn hang_kill_child(child: &mut Child, grace: Duration) {
    send_signal(child, libc::SIGTERM);
    if let Ok(status) = tokio::time::timeout(grace, child.wait()).await {
        log_exit(status);
        return;
    }
    tracing::error!(
        grace_secs = grace.as_secs(),
        "hung worker still alive after grace — SIGKILL"
    );
    send_signal(child, libc::SIGKILL);
    let _ = child.wait().await;
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

fn log_exit(status: std::io::Result<ExitStatus>) {
    match status {
        Ok(status) => {
            tracing::error!(code = ?status.code(), signal = ?status.signal(), "worker exited");
        }
        Err(e) => tracing::error!(error = %e, "error waiting for worker"),
    }
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
    use super::{backoff_delay, prune_old_attempts};
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
}
