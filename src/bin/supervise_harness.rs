// ============================================================
// src/bin/supervise_harness.rs — test fixture for tests/supervisor.rs
// ============================================================
// A thin CLI wrapper around `rustedvino::supervisor::run_supervised`,
// letting a test drive the generic supervisor engine against an arbitrary
// worker binary (fake_worker) in its own OS process. This isolation is
// needed for the SIGTERM/stop test specifically: `run_supervised` installs
// a process-wide SIGTERM handler, so a test raising a real SIGTERM against
// the shared `cargo test` process would risk affecting other concurrently
// running tests. Sending SIGTERM to this harness's own PID is safe.
//
// Usage: supervise_harness <worker-exe> [worker-args...]
// Config via env vars (defaults match `SupervisorConfig::default()`):
//   HARNESS_HEALTH_ADDR (required, e.g. "127.0.0.1:18123")
//   HARNESS_BACKOFF_BASE_SECS, HARNESS_BACKOFF_MAX_SECS, HARNESS_MAX_RESTARTS,
//   HARNESS_RESTART_WINDOW_SECS, HARNESS_HEALTH_TIMEOUT_SECS,
//   HARNESS_HANG_KILL_GRACE_SECS
// ============================================================

use std::path::PathBuf;

use rustedvino::model_manager::SupervisorConfig;

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(worker_exe) = args.get(1) else {
        eprintln!("usage: supervise_harness <worker-exe> [worker-args...]");
        std::process::exit(2);
    };
    let worker_exe = PathBuf::from(worker_exe);
    let worker_args = args[2..].to_vec();

    let Ok(health_addr_str) = std::env::var("HARNESS_HEALTH_ADDR") else {
        eprintln!("HARNESS_HEALTH_ADDR must be set");
        std::process::exit(2);
    };
    let Ok(health_addr) = health_addr_str.parse::<std::net::SocketAddr>() else {
        eprintln!("HARNESS_HEALTH_ADDR is not a valid socket address: {health_addr_str}");
        std::process::exit(2);
    };

    let cfg = SupervisorConfig {
        backoff_base_secs: env_u64("HARNESS_BACKOFF_BASE_SECS", 5),
        backoff_max_secs: env_u64("HARNESS_BACKOFF_MAX_SECS", 60),
        max_restarts: env_usize("HARNESS_MAX_RESTARTS", 5),
        restart_window_secs: env_u64("HARNESS_RESTART_WINDOW_SECS", 600),
        health_timeout_secs: env_u64("HARNESS_HEALTH_TIMEOUT_SECS", 90),
        hang_kill_grace_secs: env_u64("HARNESS_HANG_KILL_GRACE_SECS", 10),
        watchdog_poll_secs: env_u64("HARNESS_WATCHDOG_POLL_SECS", 30),
        // Off by default in the test harness — existing supervisor tests don't
        // expose `/v1/admin/health/watchdog` and shouldn't have to; a test that
        // wants to exercise the generation watchdog sets this explicitly.
        watchdog_hang_ceiling_secs: env_u64("HARNESS_WATCHDOG_HANG_CEILING_SECS", 0),
    };

    // `rustedvino::supervisor` is `#[cfg(unix)]` (supervisor mode is
    // Linux-only) — gating the call here, rather than skipping this whole
    // bin on Windows, keeps `cargo build`'s exit code green there even
    // though this fixture has nothing to do on that platform.
    #[cfg(unix)]
    let result =
        rustedvino::supervisor::run_supervised(&worker_exe, &worker_args, health_addr, &cfg).await;
    #[cfg(not(unix))]
    let result: anyhow::Result<()> = {
        let _ = (worker_exe, worker_args, health_addr, cfg);
        Err(anyhow::anyhow!(
            "supervise_harness is Unix-only (supervisor mode is Linux-only)"
        ))
    };

    match result {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("supervise_harness: {e}");
            std::process::exit(1);
        }
    }
}
