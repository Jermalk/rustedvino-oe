// ============================================================
// src/bin/fake_worker.rs — test fixture for tests/supervisor.rs
// ============================================================
// A fast, dependency-free stand-in for the real `rustedvino` worker, so the
// supervisor's respawn/backoff/hang-detection logic can be tested without a
// 30-60s cold model load and real GPU hardware. Controlled by positional CLI
// args (not env vars — those are process-global and would race across
// `cargo test`'s parallel test threads, since each test spawns this binary
// with a different desired behavior):
//
//   fake_worker crash
//   fake_worker hang
//   fake_worker healthy <bind-addr>   (e.g. "127.0.0.1:18123")
//   fake_worker exit <code>           (exit immediately with that status)
//   fake_worker gpu-stuck-after-76 <dir> <bind-addr>   (per-spawn sequence, see the fn)
//   fake_worker slow-preload <dir> <bind-addr> <allowance> <secs>   (G-1, see the fn)
//   fake_worker preload-stuck <dir> <allowance>                      (G-1, see the fn)
//   fake_worker hang-declaring <secs>          (D-04 S8: `hang` + a declared stop budget)
//   fake_worker no-teardown-then-fatal <dir>   (1st spawn exits 76, later ones log + exit 78)
//   fake_worker probe <dir> <script>           (GPU probe stand-in: `ok,hang,fail` per call)
//
// - "crash": raises a real, hardware-fault SIGSEGV immediately — not a
//   software `raise()` or Rust's null-check panic->SIGABRT path, both of
//   which behave differently (see
//   the project's internal engineering log for why this
//   distinction mattered when building the original test harness).
// - "hang": ignores SIGTERM and sleeps — simulates a worker wedged in a
//   dirty-GPU cold init, so a test can exercise the supervisor's SIGKILL
//   escalation (without this, the default SIGTERM disposition would just
//   exit immediately and the escalation path would never be reached).
// - "healthy": binds FAKE_WORKER_HEALTH_ADDR and answers every connection
//   with `200 OK`, then idles until signalled (default SIGTERM disposition
//   — the process just exits, simulating a graceful stop).
// ============================================================

fn main() {
    #[cfg(unix)]
    unix::run();
    #[cfg(not(unix))]
    {
        eprintln!(
            "fake_worker: this test fixture is Unix-only (supervisor mode is Linux-only) \
             — nothing to run on this platform"
        );
        std::process::exit(2);
    }
}

// The real fixture logic is Unix-only (it exists purely to test
// src/supervisor.rs, which is itself `#[cfg(unix)]`) and uses `libc`, a
// Unix-only target dependency — gating the whole module keeps the Windows
// cross-build's overall `cargo build` exit code green even though this bin
// has nothing to do there.
#[cfg(unix)]
mod unix {
    use std::hint::black_box;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    pub(super) fn run() {
        let args: Vec<String> = std::env::args().collect();
        match args.get(1).map(String::as_str) {
            Some("crash") => crash(),
            Some("hang") => hang(),
            Some("healthy") => healthy(args.get(2)),
            Some("exit") => exit_with(args.get(2)),
            Some("gpu-stuck-after-76") => gpu_stuck_after_76(args.get(2), args.get(3)),
            Some("gpu-stuck-after-healthy") => gpu_stuck_after_healthy(args.get(2), args.get(3)),
            Some("slow-preload") => slow_preload(&args),
            Some("preload-stuck") => preload_stuck(&args),
            Some("hang-declaring") => hang_declaring(args.get(2)),
            Some("no-teardown-then-fatal") => no_teardown_then_fatal(args.get(2)),
            Some("probe") => probe(args.get(2), args.get(3)),
            other => {
                eprintln!("fake_worker: unknown or missing behavior arg: {other:?}");
                std::process::exit(2);
            }
        }
    }

    fn crash() {
        // Safety: this is the whole point of the fixture — a deliberate fault,
        // not a real invariant to preserve. `black_box` hides the address so the
        // optimizer can't prove UB and substitute a trap instruction;
        // `write_volatile` forces a genuine hardware SIGSEGV.
        unsafe {
            let addr: usize = black_box(1);
            std::ptr::write_volatile(addr as *mut i32, 42);
        }
    }

    fn exit_with(code: Option<&String>) {
        let code = code.and_then(|c| c.parse::<i32>().ok()).unwrap_or(2);
        std::process::exit(code);
    }

    fn hang() {
        // Safety: installing SIG_IGN for SIGTERM on our own process, so the test
        // can exercise the supervisor's SIGKILL escalation path — otherwise the
        // default SIGTERM disposition would exit immediately and the escalation
        // would never be reached.
        unsafe {
            libc::signal(libc::SIGTERM, libc::SIG_IGN);
        }
        loop {
            std::thread::sleep(std::time::Duration::from_hours(1));
        }
    }

    /// The spawn number of this run (counted in `<dir>/n`).
    fn next_spawn(dir: &str) -> u32 {
        let counter = std::path::Path::new(dir).join("n");
        let n: u32 = std::fs::read_to_string(&counter)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
            + 1;
        let _ = std::fs::write(&counter, n.to_string());
        n
    }

    /// Append one line to the supervisor's progress file, as the real worker does.
    fn progress_line(line: &str) {
        if let Some(p) = std::env::var_os("RUSTEDVINO_PROGRESS_FILE")
            && let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
        {
            let _ = f.write_all(line.as_bytes());
        }
    }

    /// G-1: the 1st spawn preloads for `<secs>` (longer than the test's health timeout) while
    /// reporting it like the real worker (`preload_started <allowance>`, then `preload_done`), then
    /// becomes healthy on `<addr>` (pid in `<dir>/pid`). Any later spawn exits 78 — reaching it
    /// means the slow preload was killed.
    fn slow_preload(args: &[String]) {
        let (Some(dir), Some(addr)) = (args.get(2), args.get(3)) else {
            std::process::exit(2);
        };
        let allowance = args.get(4).map_or("60", String::as_str);
        let secs: u64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(4);
        if next_spawn(dir) > 1 {
            std::process::exit(78);
        }
        progress_line(&format!("preload_started {allowance}\n"));
        std::thread::sleep(std::time::Duration::from_secs(secs));
        progress_line("preload_done\n");
        let _ = std::fs::write(
            std::path::Path::new(dir).join("pid"),
            std::process::id().to_string(),
        );
        healthy(Some(addr));
    }

    /// D-04 S8: like `hang` (ignores SIGTERM, never healthy), but declares a stop budget of
    /// `<secs>` in the supervisor's progress file, as a worker inside its stop wait does.
    fn hang_declaring(secs: Option<&String>) {
        // Safety: installing SIG_IGN for SIGTERM on our own process, as `hang` does; no other
        // thread exists yet and no handler state is shared.
        unsafe {
            libc::signal(libc::SIGTERM, libc::SIG_IGN);
        }
        progress_line(&format!(
            "stop_budget {}\n",
            secs.map_or("1", String::as_str)
        ));
        loop {
            std::thread::sleep(std::time::Duration::from_secs(60));
        }
    }

    /// G-1: the 1st spawn starts a preload with `<allowance>` and never finishes or binds (a hung
    /// load); the supervisor must still kill it, after the allowance. Later spawns exit 78.
    fn preload_stuck(args: &[String]) {
        let Some(dir) = args.get(2) else {
            std::process::exit(2);
        };
        let allowance = args.get(3).map_or("1", String::as_str);
        if next_spawn(dir) > 1 {
            std::process::exit(78);
        }
        progress_line(&format!("preload_started {allowance}\n"));
        loop {
            std::thread::sleep(std::time::Duration::from_secs(60));
        }
    }

    /// Step 7 deadline scenario, one behaviour per spawn (counted in `<dir>/n`): 1st exits 76
    /// (no teardown); 2nd marks its first GPU load started in the supervisor's progress file and
    /// never finishes it (stuck device); 3rd is healthy on `<addr>` (pid in `<dir>/pid`); 4th
    /// exits 78 so the supervisor stops.
    /// One more call of a per-`dir` counter (`<dir>/<name>`), starting at 1.
    fn bump(dir: &str, name: &str) -> u32 {
        let counter = std::path::Path::new(dir).join(name);
        let n: u32 = std::fs::read_to_string(&counter)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
            + 1;
        let _ = std::fs::write(&counter, n.to_string());
        n
    }

    /// Readiness-probe scenario worker: the 1st spawn leaves without teardown (76); every later
    /// spawn records itself in `<dir>/workers` and exits 78, which stops the supervisor.
    fn no_teardown_then_fatal(dir: Option<&String>) {
        let Some(dir) = dir else {
            std::process::exit(2);
        };
        match bump(dir, "workers") {
            1 => std::process::exit(76),
            _ => std::process::exit(78),
        }
    }

    /// The GPU readiness probe stand-in: call k (counted in `<dir>/probes`) follows the k-th
    /// entry of the comma-separated `script` (the last one repeats): `ok` prints a probe line
    /// and exits 0, `fail` exits 2, `hang` never returns.
    fn probe(dir: Option<&String>, script: Option<&String>) {
        let (Some(dir), Some(script)) = (dir, script) else {
            std::process::exit(2);
        };
        let n = bump(dir, "probes");
        let steps: Vec<&str> = script.split(',').collect();
        let idx = usize::try_from(n).unwrap_or(usize::MAX).saturating_sub(1);
        match steps.get(idx).or(steps.last()).copied() {
            Some("ok") => {
                println!("GPU_PROBE device=FAKE ok=true ms=1");
                std::process::exit(0);
            }
            Some("hang") => loop {
                std::thread::sleep(std::time::Duration::from_secs(60));
            },
            _ => std::process::exit(2),
        }
    }

    fn gpu_stuck_after_76(dir: Option<&String>, addr: Option<&String>) {
        let Some(dir) = dir else {
            std::process::exit(2);
        };
        let counter = std::path::Path::new(dir).join("n");
        let n: u32 = std::fs::read_to_string(&counter)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
            + 1;
        let _ = std::fs::write(&counter, n.to_string());
        match n {
            1 => std::process::exit(76),
            2 => {
                if let Some(p) = std::env::var_os("RUSTEDVINO_PROGRESS_FILE") {
                    let _ = std::fs::write(p, "gpu_load_started 0\n");
                }
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(60));
                }
            }
            3 => {
                let _ = std::fs::write(
                    std::path::Path::new(dir).join("pid"),
                    std::process::id().to_string(),
                );
                healthy(addr);
            }
            // The test ends the run: a config-fatal exit stops the supervisor for good.
            _ => std::process::exit(78),
        }
    }

    /// 0.9.0 review E (shutdown-3): spawn 1 exits 76; spawn 2 becomes healthy, then its first GPU
    /// load (a client's, after `/health`) starts and never returns; spawn 3 is healthy.
    fn gpu_stuck_after_healthy(dir: Option<&String>, addr: Option<&String>) {
        let Some(dir) = dir else {
            std::process::exit(2);
        };
        let counter = std::path::Path::new(dir).join("n");
        let n: u32 = std::fs::read_to_string(&counter)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
            + 1;
        let _ = std::fs::write(&counter, n.to_string());
        match n {
            1 => std::process::exit(76),
            2 => {
                if let Some(p) = std::env::var_os("RUSTEDVINO_PROGRESS_FILE") {
                    std::thread::spawn(move || {
                        std::thread::sleep(std::time::Duration::from_secs(1));
                        let _ = std::fs::write(p, "gpu_load_started 0\n");
                    });
                }
                healthy(addr);
            }
            3 => {
                let _ = std::fs::write(
                    std::path::Path::new(dir).join("pid"),
                    std::process::id().to_string(),
                );
                healthy(addr);
            }
            _ => std::process::exit(78),
        }
    }

    fn healthy(addr: Option<&String>) {
        let Some(addr) = addr else {
            eprintln!("fake_worker: healthy behavior requires a bind address argument");
            std::process::exit(2);
        };
        let Ok(listener) = TcpListener::bind(addr) else {
            eprintln!("fake_worker: failed to bind {addr}");
            std::process::exit(2);
        };
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut buf = [0_u8; 512];
            let _ = stream.read(&mut buf);
            let _ = stream.write_all(b"HTTP/1.0 200 OK\r\nContent-Length: 0\r\n\r\n");
        }
    }
} // mod unix
