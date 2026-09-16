// ============================================================
// RustedVINO — binary entry point
// ============================================================
// Bootstraps the Tokio runtime, loads configuration, initialises
// the ModelManager (which preloads models at startup), then
// starts the axum HTTP server.
//
// All routes, handlers, and business logic live in src/lib.rs
// and src/handlers/. Keeping main.rs thin means integration
// tests can import create_router() without OS resources.
// ============================================================

use std::time::Duration;

use rustedvino::log_ring::LogRingBuffer;

/// Hard cap on waiting for outstanding Tokio tasks once [`async_main`] has
/// already returned. Every OTHER shutdown stage above this one is bounded —
/// but each model's engine lives on its own dedicated `std::thread` (never
/// `spawn_blocking`, per the pipeline-ownership rule in the project's internal engineering log), driven by
/// an mpsc command channel. If that thread wedges inside the `OpenVINO` FFI call
/// (e.g. a GPU driver engine-reset that `openvino_genai` doesn't surface as an
/// error, so the call spins forever waiting for a completion that never comes
/// — reproduced 2026-07-28, the project's internal engineering log),
/// it never sends a reply — so the axum-spawned Tokio task servicing that
/// HTTP connection stays parked awaiting one forever. That task IS
/// Tokio-tracked, and eviction already gave up waiting on the raw thread
/// (dropped its own `JoinHandle`) by the time shutdown runs, so
/// `teardown_engines` never sees it either. Left unbounded, `Runtime::drop()`
/// (what `#[tokio::main]` inserts implicitly) waits for that task forever —
/// verified live: `teardown_engines` logged `"shutdown complete"` and the
/// process still didn't exit 25+ minutes later. `Runtime::shutdown_timeout`
/// bounds that wait explicitly; process exit right after forces an
/// `exit_group`, which tears down every thread — including a spinning one —
/// regardless of whether it cooperates.
const PROCESS_EXIT_TIMEOUT: Duration = Duration::from_secs(5);

/// Handles the device-query CLI flags that print and exit before any config
/// load or model work. Returns `true` if a flag was present and handled (the
/// caller then exits cleanly), `false` for a normal server startup.
///
/// - `--list-devices`: prints the bare `OpenVINO` device names (exits `1` if
///   none enumerate — a driver/runtime failure).
/// - `--device-info`: prints the full Phase-A device inventory (name, kind,
///   tier, memory domain, memory size) — the device-probe diagnostic view.
fn handle_device_query_flags() -> bool {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--list-devices") {
        let devices = rustedvino::ov_cb::list_devices();
        if devices.is_empty() {
            tracing::error!("failed to enumerate OpenVINO devices — check driver and runtime");
            std::process::exit(1);
        }
        tracing::info!(devices = %devices.join(", "), "available OpenVINO devices");
        for dev in &devices {
            println!("{dev}");
        }
        return true;
    }
    if args.iter().any(|a| a == "--device-info") {
        rustedvino::device_inventory::DeviceInventory::probe().print_report();
        return true;
    }
    false
}

/// Real process entry point. Builds the Tokio runtime explicitly (instead of
/// `#[tokio::main]`'s implicit one) so shutdown can be bounded — see
/// [`PROCESS_EXIT_TIMEOUT`]. `async_main` is unchanged behaviourally; only
/// what happens *after* it returns is different.
fn main() -> std::process::ExitCode {
    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("failed to build the Tokio runtime: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };

    let result = rt.block_on(async_main());

    // Consumes `rt` — bounds the wait for any outstanding `spawn_blocking`
    // task instead of blocking forever. See `PROCESS_EXIT_TIMEOUT` doc.
    rt.shutdown_timeout(PROCESS_EXIT_TIMEOUT);

    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e:?}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[allow(clippy::too_many_lines)]
async fn async_main() -> anyhow::Result<()> {
    // Recent-log tail for `GET /v1/admin/logs/tail` (e.g. a Pyramu panel) —
    // built here so it can be installed as the tracing writer below *and*
    // handed to `AppState` later via `with_log_buffer`, the same instance
    // either way. ANSI colour is disabled globally: it makes the buffered
    // lines unreadable without stripping escape codes client-side, and a
    // `nohup`-redirected log file benefits from plain text too.
    let log_buffer = LogRingBuffer::new();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_ansi(false)
        .with_writer(log_buffer.clone())
        .init();

    // ── Crash auto-restart wrapper ───────────────────────────────────────
    // `--supervise`: spawn the real server as a child, watch it, respawn
    // with backoff+cap on crash (src/supervisor.rs). Checked before the
    // crash handler installs below — that handler is for the worker's
    // OpenVINO threads, not this thin parent, which never touches OpenVINO.
    // Linux-only: the module doesn't exist in a Windows build, so a
    // Windows binary fails loud here instead of silently ignoring the flag
    // or hitting an unresolved path.
    if std::env::args().any(|a| a == "--supervise") {
        #[cfg(unix)]
        {
            let config_path = rustedvino::startup::resolve_config_path()?;
            return rustedvino::supervisor::run(&config_path).await;
        }
        #[cfg(not(unix))]
        {
            anyhow::bail!(
                "--supervise is not supported on this platform yet — run the binary directly"
            );
        }
    }

    // Catch native SIGSEGV/SIGABRT/SIGBUS/SIGILL (e.g. an OpenVINO C++ crash
    // on a model's engine thread) and log a marker + backtrace before the
    // process dies, instead of vanishing with nothing but a `dmesg` line.
    // Installed before any engine thread spawns so every one of them is
    // covered. Unix-only for now (see crash_handler module doc).
    #[cfg(unix)]
    rustedvino::crash_handler::install();

    // ── Early-exit device-query flags ───────────────────────────────────
    // Run before config load so they work on a fresh install with no config.
    if handle_device_query_flags() {
        return Ok(());
    }

    // ── Load config, probe devices, build the model manager + AppState ──
    // See src/startup.rs — shared with any other binary linking against
    // this crate (e.g. RustedVINO Bear Edition) so both get identical
    // startup validation and shutdown behaviour.
    let setup = rustedvino::startup::bootstrap(log_buffer).await?;
    let app = rustedvino::create_router_with_auth(
        setup.state,
        &setup.cors_origins,
        &setup.api_keys,
        &setup.admin_api_keys,
    );

    // Bind address/port come from config (`bind_addr`/`port`; default
    // loopback:11437 — secure by default, T4.2). stormVINO owns 11435;
    // cutover day: RustedVINO takes 11435 (Part 9).
    rustedvino::startup::serve(app, setup.addr, &setup.model_manager).await
}
