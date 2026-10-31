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
        gpu_first_load_deadline_secs: 0,
        gpu_probe_deadline_secs: 0,
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

/// A worker that exits 78 (fatal config error) is NOT respawned: the supervisor
/// returns at once, even with a long backoff and a generous restart budget.
#[tokio::test]
async fn config_fatal_exit_is_not_respawned() {
    let cfg = SupervisorConfig {
        backoff_base_secs: 30,
        backoff_max_secs: 60,
        max_restarts: 5,
        restart_window_secs: 600,
        health_timeout_secs: 5,
        hang_kill_grace_secs: 1,
        watchdog_poll_secs: 30,
        watchdog_hang_ceiling_secs: 0,
        gpu_first_load_deadline_secs: 0,
        gpu_probe_deadline_secs: 0,
    };
    let health_addr = "127.0.0.1:18193".parse().unwrap();

    let start = Instant::now();
    let result = run_supervised(
        &fake_worker(),
        &["exit".to_string(), "78".to_string()],
        health_addr,
        &cfg,
    )
    .await;

    let err = result.expect_err("a config-fatal exit must end the supervisor with an error");
    assert!(
        err.to_string().contains("fatal configuration error"),
        "unexpected message: {err}"
    );
    // MR-5c: typed as ConfigFatal, so `main` exits 78 like a plain start.
    assert!(
        err.downcast_ref::<rustedvino::startup::ConfigFatal>()
            .is_some(),
        "the supervisor's error must be a ConfigFatal"
    );
    // One spawn, no 30 s backoff: a respawn path would take >= 30 s.
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "supervisor should stop immediately, took {:?}",
        start.elapsed()
    );
}

/// Control: an ordinary failure exit (code 1) still goes through the respawn
/// path - it burns the whole restart budget before giving up.
#[tokio::test]
async fn ordinary_failure_exit_is_still_respawned() {
    let cfg = SupervisorConfig {
        backoff_base_secs: 0,
        backoff_max_secs: 0,
        max_restarts: 3,
        restart_window_secs: 600,
        health_timeout_secs: 5,
        hang_kill_grace_secs: 1,
        watchdog_poll_secs: 30,
        watchdog_hang_ceiling_secs: 0,
        gpu_first_load_deadline_secs: 0,
        gpu_probe_deadline_secs: 0,
    };
    let health_addr = "127.0.0.1:18194".parse().unwrap();

    let result = run_supervised(
        &fake_worker(),
        &["exit".to_string(), "1".to_string()],
        health_addr,
        &cfg,
    )
    .await;

    let err = result.expect_err("exit 1 loops until the restart cap");
    assert!(
        err.to_string().contains("crash-looping"),
        "exit 1 must reach the give-up path, got: {err}"
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
        gpu_first_load_deadline_secs: 0,
        gpu_probe_deadline_secs: 0,
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

/// D-04 S8: a worker that declares a stop budget (it is waiting for a running load) is not
/// `SIGKILL`ed before that budget plus slack, though the configured grace is 1 s.
#[tokio::test]
async fn a_declared_stop_budget_defers_the_sigkill() {
    let cfg = SupervisorConfig {
        backoff_base_secs: 0,
        backoff_max_secs: 0,
        max_restarts: 1,
        restart_window_secs: 600,
        health_timeout_secs: 1,
        hang_kill_grace_secs: 1,
        watchdog_poll_secs: 30,
        watchdog_hang_ceiling_secs: 0,
        gpu_first_load_deadline_secs: 0,
        gpu_probe_deadline_secs: 0,
    };
    let health_addr = "127.0.0.1:18195".parse().unwrap();
    let start = Instant::now();
    let args = ["hang-declaring".to_string(), "2".to_string()];
    let result = tokio::time::timeout(
        Duration::from_secs(30),
        run_supervised(&fake_worker(), &args, health_addr, &cfg),
    )
    .await
    .expect("the declared budget must still end in a SIGKILL");
    assert!(
        result.is_err(),
        "max_restarts=1: one killed attempt gives up"
    );
    // health 1-1.5 s + first read 0.5 s + declared 2 s + slack 5 s (about 8.5 s); the bare grace
    // would have killed at ~2 s.
    let elapsed = start.elapsed();
    assert!(
        elapsed >= Duration::from_secs(8) && elapsed < Duration::from_secs(20),
        "{elapsed:?}"
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

/// A worker that leaves with exit 76 (`OpenVINO` calls still in flight, step 6a) is respawned like
/// any failure until the cap — never treated as a clean stop or a fatal config error. That it also
/// counts in the watchdog-kill window is unit-tested (`classify_exit`): both counters grow together
/// and the plain cap is checked first, so this loop always ends on the crash-loop text.
#[tokio::test]
async fn ffi_stalled_exit_is_respawned_until_the_cap() {
    let cfg = SupervisorConfig {
        backoff_base_secs: 0,
        backoff_max_secs: 0,
        max_restarts: 3,
        restart_window_secs: 600,
        health_timeout_secs: 5,
        hang_kill_grace_secs: 1,
        watchdog_poll_secs: 30,
        watchdog_hang_ceiling_secs: 0,
        gpu_first_load_deadline_secs: 0,
        gpu_probe_deadline_secs: 0,
    };
    let health_addr = "127.0.0.1:18197".parse().unwrap();

    let result = run_supervised(
        &fake_worker(),
        &["exit".to_string(), "76".to_string()],
        health_addr,
        &cfg,
    )
    .await;

    let err = result.expect_err("exit 76 is respawned until the restart cap");
    assert!(
        err.to_string().contains("crash-looping"),
        "exit 76 must reach the give-up path, got: {err}"
    );
}

/// Step 7: after a worker exits 76 (no teardown), the next worker's first GPU load must finish
/// within `gpu_first_load_deadline_secs`; the fake's second spawn marks a GPU load started and
/// never finishes it, so the supervisor kills it well before the 60 s health timeout, and the
/// third spawn (healthy) is reached. A clean SIGTERM then ends the run.
#[tokio::test]
async fn stuck_first_gpu_load_after_a_no_teardown_exit_is_killed_by_the_deadline() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = SupervisorConfig {
        backoff_base_secs: 0,
        backoff_max_secs: 0,
        max_restarts: 5,
        restart_window_secs: 600,
        health_timeout_secs: 60,
        hang_kill_grace_secs: 1,
        watchdog_poll_secs: 30,
        watchdog_hang_ceiling_secs: 0,
        gpu_first_load_deadline_secs: 2,
        gpu_probe_deadline_secs: 0,
    };
    let health_addr: std::net::SocketAddr = "127.0.0.1:18198".parse().unwrap();
    let args = vec![
        "gpu-stuck-after-76".to_string(),
        dir.path().display().to_string(),
        health_addr.to_string(),
    ];
    let t0 = std::time::Instant::now();
    let sup = tokio::spawn({
        let exe = fake_worker();
        async move { run_supervised(&exe, &args, health_addr, &cfg).await }
    });
    // Wait until the third spawn is healthy.
    let mut healthy = false;
    for _ in 0..200 {
        if std::fs::read_to_string(dir.path().join("n"))
            .ok()
            .as_deref()
            == Some("3")
            && tokio::net::TcpStream::connect(health_addr).await.is_ok()
        {
            healthy = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(
        healthy,
        "third spawn never came up (spawn counter: {:?}, after {:?})",
        std::fs::read_to_string(dir.path().join("n")).ok(),
        t0.elapsed()
    );
    assert!(
        t0.elapsed() < std::time::Duration::from_secs(30),
        "the deadline (2 s), not the 60 s health timeout, must have ended spawn 2: {:?}",
        t0.elapsed()
    );
    // End the run: stop spawn 3; spawn 4 exits 78 and the supervisor gives up for good.
    let pid: i32 = std::fs::read_to_string(dir.path().join("pid"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    // SAFETY: `pid` is the fake worker this test's supervisor spawned (it wrote its own pid).
    unsafe {
        libc::kill(pid, libc::SIGTERM);
    }
    let err = tokio::time::timeout(std::time::Duration::from_secs(30), sup)
        .await
        .expect("supervisor did not stop")
        .unwrap()
        .expect_err("spawn 4 exits 78");
    assert!(err.to_string().contains("fatal configuration"), "{err}");
}

/// G-1: a worker whose preload outlasts `health_timeout_secs` but reports its progress is not
/// killed as hung: the supervisor waits and the worker becomes healthy on its first spawn.
/// 0.9.0 review E (shutdown-3): the first-GPU-load deadline also covers a load that starts after
/// the worker is healthy (no GPU preload; a client's load). Spawn 2 is healthy, then its first GPU
/// load hangs; the deadline kills it and spawn 3 comes up. shutdown-1: the deadline kill is a
/// no-teardown exit, so the readiness probe runs before spawn 3 too.
#[tokio::test]
async fn stuck_first_gpu_load_after_health_is_killed_by_the_deadline() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = SupervisorConfig {
        backoff_base_secs: 0,
        backoff_max_secs: 0,
        max_restarts: 5,
        restart_window_secs: 600,
        health_timeout_secs: 60,
        hang_kill_grace_secs: 1,
        watchdog_poll_secs: 30,
        watchdog_hang_ceiling_secs: 0,
        gpu_first_load_deadline_secs: 2,
        gpu_probe_deadline_secs: 5,
    };
    let health_addr: std::net::SocketAddr = "127.0.0.1:18196".parse().unwrap();
    let args = vec![
        "gpu-stuck-after-healthy".to_string(),
        dir.path().display().to_string(),
        health_addr.to_string(),
    ];
    let t0 = std::time::Instant::now();
    let probe = rustedvino::supervisor::ProbeCmd {
        exe: fake_worker(),
        args: vec![
            "probe".to_string(),
            dir.path().display().to_string(),
            "ok".to_string(),
        ],
        settle: std::time::Duration::from_millis(50),
    };
    let sup = tokio::spawn({
        let exe = fake_worker();
        async move {
            rustedvino::supervisor::run_supervised_with_probe(
                &exe,
                &args,
                health_addr,
                &cfg,
                Some(&probe),
            )
            .await
        }
    });
    // Wait until the third spawn is healthy.
    let mut healthy = false;
    for _ in 0..200 {
        if std::fs::read_to_string(dir.path().join("n"))
            .ok()
            .as_deref()
            == Some("3")
            && tokio::net::TcpStream::connect(health_addr).await.is_ok()
        {
            healthy = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(
        healthy,
        "third spawn never came up (spawn counter: {:?}, after {:?})",
        std::fs::read_to_string(dir.path().join("n")).ok(),
        t0.elapsed()
    );
    assert!(
        t0.elapsed() < std::time::Duration::from_secs(30),
        "the 2 s deadline must have ended spawn 2: {:?}",
        t0.elapsed()
    );
    let probes = std::fs::read_to_string(dir.path().join("probes")).unwrap_or_default();
    assert_eq!(
        probes.trim(),
        "2",
        "probed after spawn 1's exit 76 and again after the deadline kill of spawn 2"
    );
    // End the run: stop spawn 3; spawn 4 exits 78 and the supervisor gives up for good.
    let pid: i32 = std::fs::read_to_string(dir.path().join("pid"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    // SAFETY: `pid` is the fake worker this test's supervisor spawned (it wrote its own pid).
    unsafe {
        libc::kill(pid, libc::SIGTERM);
    }
    let err = tokio::time::timeout(std::time::Duration::from_secs(30), sup)
        .await
        .expect("supervisor did not stop")
        .unwrap()
        .expect_err("spawn 4 exits 78");
    assert!(err.to_string().contains("fatal configuration"), "{err}");
}

/// G-1: a worker whose preload outlasts `health_timeout_secs` but reports its progress is not
/// killed as hung: the supervisor waits and the worker becomes healthy on its first spawn.
#[tokio::test]
async fn a_slow_preload_that_reports_progress_is_not_killed() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = SupervisorConfig {
        backoff_base_secs: 0,
        backoff_max_secs: 0,
        max_restarts: 3,
        restart_window_secs: 600,
        health_timeout_secs: 2,
        hang_kill_grace_secs: 1,
        watchdog_poll_secs: 30,
        watchdog_hang_ceiling_secs: 0,
        gpu_first_load_deadline_secs: 0,
        gpu_probe_deadline_secs: 0,
    };
    let health_addr: std::net::SocketAddr = "127.0.0.1:18201".parse().unwrap();
    let args = vec![
        "slow-preload".to_string(),
        dir.path().display().to_string(),
        health_addr.to_string(),
        "60".to_string(),
        "5".to_string(),
    ];
    let sup = tokio::spawn({
        let exe = fake_worker();
        async move { run_supervised(&exe, &args, health_addr, &cfg).await }
    });
    let mut healthy = false;
    for _ in 0..150 {
        if tokio::net::TcpStream::connect(health_addr).await.is_ok() {
            healthy = true;
            break;
        }
        if sup.is_finished() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let spawns = std::fs::read_to_string(dir.path().join("n")).unwrap_or_default();
    assert!(
        healthy && spawns == "1",
        "the 5 s preload (health timeout 2 s) must reach healthy on spawn 1 (spawns: {spawns:?})"
    );
    let pid: i32 = std::fs::read_to_string(dir.path().join("pid"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    // SAFETY: `pid` is the fake worker this test's supervisor spawned (it wrote its own pid).
    unsafe {
        libc::kill(pid, libc::SIGTERM);
    }
    let _ = tokio::time::timeout(Duration::from_secs(30), sup).await;
}

/// G-1: a preload that starts and never finishes is still killed — after its own load allowance
/// plus the health timeout, not never.
#[tokio::test]
async fn a_preload_that_never_finishes_is_killed_after_its_allowance() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = SupervisorConfig {
        backoff_base_secs: 0,
        backoff_max_secs: 0,
        max_restarts: 3,
        restart_window_secs: 600,
        health_timeout_secs: 2,
        hang_kill_grace_secs: 1,
        watchdog_poll_secs: 30,
        watchdog_hang_ceiling_secs: 0,
        gpu_first_load_deadline_secs: 0,
        gpu_probe_deadline_secs: 0,
    };
    let health_addr: std::net::SocketAddr = "127.0.0.1:18202".parse().unwrap();
    let args = vec![
        "preload-stuck".to_string(),
        dir.path().display().to_string(),
        "2".to_string(),
    ];
    let t0 = Instant::now();
    let result = tokio::time::timeout(
        Duration::from_secs(30),
        run_supervised(&fake_worker(), &args, health_addr, &cfg),
    )
    .await
    .expect("the supervisor must not wait forever on a stuck preload");
    let elapsed = t0.elapsed();
    assert!(
        result.is_err(),
        "spawn 2 exits 78, so the run ends in an error"
    );
    assert_eq!(
        std::fs::read_to_string(dir.path().join("n")).unwrap(),
        "2",
        "the stuck preload was killed and respawned once"
    );
    // Killed after allowance 2 s + health 2 s (+ kill grace 1 s), not at the bare 2 s health timeout.
    assert!(
        elapsed >= Duration::from_secs(4) && elapsed < Duration::from_secs(15),
        "{elapsed:?}"
    );
}

// ---- GPU readiness probe ------------------------------------------------------------------

/// A config for the probe tests: no backoff, a 1 s probe deadline, the first-load backstop off.
fn probe_cfg() -> SupervisorConfig {
    SupervisorConfig {
        backoff_base_secs: 0,
        backoff_max_secs: 0,
        max_restarts: 5,
        restart_window_secs: 600,
        health_timeout_secs: 5,
        hang_kill_grace_secs: 1,
        watchdog_poll_secs: 30,
        watchdog_hang_ceiling_secs: 0,
        gpu_first_load_deadline_secs: 0,
        gpu_probe_deadline_secs: 1,
    }
}

/// Runs the "76, then 78" worker under a probe following `script`; returns (probe calls, worker
/// spawns, elapsed). The run ends with the 2nd worker's config-fatal exit.
async fn run_probe_scenario(script: &str, port: u16) -> (u32, u32, Duration) {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().display().to_string();
    let probe = rustedvino::supervisor::ProbeCmd {
        exe: fake_worker(),
        args: vec!["probe".to_string(), d.clone(), script.to_string()],
        settle: Duration::from_millis(50),
    };
    let start = Instant::now();
    let result = rustedvino::supervisor::run_supervised_with_probe(
        &fake_worker(),
        &["no-teardown-then-fatal".to_string(), d],
        format!("127.0.0.1:{port}").parse().unwrap(),
        &probe_cfg(),
        Some(&probe),
    )
    .await;
    assert!(result.is_err(), "the 2nd worker's 78 ends the run");
    let read = |name: &str| {
        std::fs::read_to_string(dir.path().join(name))
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    };
    (read("probes"), read("workers"), start.elapsed())
}

/// A no-teardown exit makes the supervisor probe once before the next worker; a passing probe
/// lets it start at once.
#[tokio::test]
async fn a_passing_probe_runs_once_after_a_no_teardown_exit() {
    let (probes, workers, elapsed) = run_probe_scenario("ok", 18210).await;
    assert_eq!(probes, 1);
    assert_eq!(workers, 2);
    assert!(elapsed < Duration::from_secs(5), "{elapsed:?}");
}

/// A hung probe is killed at its deadline and run again; the next attempt passes.
#[tokio::test]
async fn a_hung_probe_is_killed_and_retried() {
    let (probes, workers, elapsed) = run_probe_scenario("hang,ok", 18211).await;
    assert_eq!(probes, 2, "hung once, then passed");
    assert_eq!(workers, 2);
    assert!(
        elapsed >= Duration::from_secs(1),
        "waited out the 1 s deadline: {elapsed:?}"
    );
}

/// A probe that hangs every time does not block recovery: after the attempts the worker
/// starts anyway (with its first-load backstop, when configured).
#[tokio::test]
async fn a_probe_hanging_every_time_still_starts_the_worker() {
    let (probes, workers, _) = run_probe_scenario("hang", 18212).await;
    assert_eq!(probes, 3, "PROBE_ATTEMPTS");
    assert_eq!(workers, 2, "the worker still started");
}

/// A probe that fails fast (an error, not a hang) is not retried and does not block the worker.
#[tokio::test]
async fn a_fast_failing_probe_does_not_block_the_worker() {
    let (probes, workers, elapsed) = run_probe_scenario("fail", 18213).await;
    assert_eq!(probes, 1);
    assert_eq!(workers, 2);
    assert!(elapsed < Duration::from_secs(5), "{elapsed:?}");
}

/// No probe without a no-teardown exit: a worker that stops cleanly-fatal at once is never
/// preceded by one.
#[tokio::test]
async fn no_probe_without_a_no_teardown_exit() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path().display().to_string();
    let probe = rustedvino::supervisor::ProbeCmd {
        exe: fake_worker(),
        args: vec!["probe".to_string(), d, "ok".to_string()],
        settle: Duration::from_millis(50),
    };
    let result = rustedvino::supervisor::run_supervised_with_probe(
        &fake_worker(),
        &["exit".to_string(), "78".to_string()],
        "127.0.0.1:18214".parse().unwrap(),
        &probe_cfg(),
        Some(&probe),
    )
    .await;
    assert!(result.is_err());
    assert!(!dir.path().join("probes").exists(), "the probe never ran");
}
