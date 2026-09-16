// ============================================================
// tests/supervisor.rs — crash auto-restart wrapper integration tests
// ============================================================
// Drives `rustedvino::supervisor::run_supervised` against `fake_worker`
// (a fast, dependency-free fixture — see src/bin/fake_worker.rs) instead of
// the real server, which needs 30-60s and real GPU hardware to cold-init.
//
// The SIGTERM/stop test runs through `supervise_harness` (a separate OS
// process) rather than calling `run_supervised` directly in this test
// binary: `run_supervised` installs a process-wide SIGTERM handler, and
// `cargo test` runs multiple tests concurrently in one process — raising a
// real SIGTERM against the shared test binary would risk affecting other
// tests running at the same time. Sending SIGTERM to a dedicated child
// process's own PID is safe.
// ============================================================
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use rustedvino::model_manager::SupervisorConfig;
use rustedvino::supervisor::run_supervised;

fn fake_worker() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_fake_worker"))
}

fn harness() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_supervise_harness"))
}

#[tokio::test]
async fn crash_loop_gives_up_after_max_restarts() {
    let cfg = SupervisorConfig {
        backoff_base_secs: 0,
        backoff_max_secs: 0,
        max_restarts: 3,
        restart_window_secs: 600,
        health_timeout_secs: 2,
        hang_kill_grace_secs: 1,
        watchdog_poll_secs: 30,
        watchdog_hang_ceiling_secs: 0,
    };
    // Health address is never actually contacted — the worker crashes
    // before it could bind anything — but `run_supervised` still needs one.
    let health_addr = "127.0.0.1:18191".parse().unwrap();

    let start = Instant::now();
    let result = run_supervised(&fake_worker(), &["crash".to_string()], health_addr, &cfg).await;

    assert!(
        result.is_err(),
        "expected a give-up error after exhausting max_restarts"
    );
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "a crash loop with zero backoff should fail fast, took {:?}",
        start.elapsed()
    );
}

#[tokio::test]
async fn hung_worker_is_actively_killed() {
    let cfg = SupervisorConfig {
        backoff_base_secs: 0,
        backoff_max_secs: 0,
        max_restarts: 1,
        restart_window_secs: 600,
        health_timeout_secs: 1,
        hang_kill_grace_secs: 1,
        watchdog_poll_secs: 30,
        watchdog_hang_ceiling_secs: 0,
    };
    let health_addr = "127.0.0.1:18192".parse().unwrap();

    let start = Instant::now();
    // `fake_worker hang` ignores SIGTERM — if the supervisor didn't
    // escalate to SIGKILL, this call would never return.
    let result = tokio::time::timeout(
        Duration::from_secs(10),
        run_supervised(&fake_worker(), &["hang".to_string()], health_addr, &cfg),
    )
    .await;

    let result = result.expect("run_supervised did not return — hung worker was not killed");
    assert!(
        result.is_err(),
        "single hang attempt with max_restarts=1 should give up, not succeed"
    );
    // health_timeout(1s) + hang_kill_grace(1s) should bound this well under
    // the outer 10s timeout if the SIGKILL escalation fired promptly.
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "hang-kill should resolve within a couple seconds, took {:?}",
        start.elapsed()
    );
}

#[tokio::test]
async fn sigterm_stops_cleanly_without_respawn() {
    let health_addr = "127.0.0.1:18193";

    let mut harness_proc = tokio::process::Command::new(harness())
        .arg(fake_worker())
        .arg("healthy")
        .arg(health_addr)
        .env("HARNESS_HEALTH_ADDR", health_addr)
        .env("HARNESS_MAX_RESTARTS", "5")
        .env("HARNESS_HEALTH_TIMEOUT_SECS", "5")
        .env("HARNESS_HANG_KILL_GRACE_SECS", "2")
        .spawn()
        .expect("spawn supervise_harness");

    // Wait for the worker to come up before testing the stop path.
    let became_healthy = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if tokio::net::TcpStream::connect(health_addr).await.is_ok() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .is_ok();
    assert!(became_healthy, "worker never came up through the harness");

    let Some(harness_pid) = harness_proc.id() else {
        panic!("harness process has no PID");
    };
    // Safety: `harness_pid` is our own just-spawned child's PID, read
    // immediately before the call — no PID-reuse window.
    #[allow(clippy::cast_possible_wrap)]
    unsafe {
        libc::kill(harness_pid as libc::pid_t, libc::SIGTERM);
    }

    let start = Instant::now();
    let status = tokio::time::timeout(Duration::from_secs(10), harness_proc.wait())
        .await
        .expect("harness did not exit after SIGTERM — stop-vs-crash disambiguation failed")
        .expect("error waiting for harness");

    assert!(
        status.success(),
        "harness should exit 0 on a clean SIGTERM stop, got {status:?}"
    );
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "SIGTERM stop should be prompt, took {:?}",
        start.elapsed()
    );
}
