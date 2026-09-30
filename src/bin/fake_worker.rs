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
