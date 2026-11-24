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
/// regardless of whether it cooperates. Careful: returning from `main` is NOT a bare
/// `exit_group`; it first runs libc `exit()` and with it the C++ static destructors, which race an
/// engine thread still inside the pipeline. `main` therefore leaves through
/// `startup::exit_without_destructors` on every exit once model construction started (D-01, D-04 S7),
/// so this bounded wait is reached only by exits before any engine existed (config errors, a port
/// clash, the device-query flags) — kept as the backstop for a task that never finishes.
const PROCESS_EXIT_TIMEOUT: Duration = Duration::from_secs(5);

/// Handles the device-query CLI flags that print and exit before any config
/// load or model work. Returns `true` if a flag was present and handled (the
/// caller then exits cleanly), `false` for a normal server startup.
///
/// - `--list-devices`: prints the bare `OpenVINO` device names (exits `1` if
///   none enumerate — a driver/runtime failure).
/// - `--device-info`: prints the full Phase-A device inventory (name, kind,
///   tier, memory domain, memory size) — the device-probe diagnostic view.
/// - `--print-default-config`: prints every global setting with its default
///   as JSON (`models_dir`/`device`/`total_vram_gb` are placeholders), ready to
///   paste into a `config.json`. `--print-default-model` does the same for one
///   `models.<id>` stanza. See `CONFIG.md`.
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
    if args.iter().any(|a| a == "--print-default-config") {
        print_template(rustedvino::model_manager::Config::default_template());
        return true;
    }
    if args.iter().any(|a| a == "--print-default-model") {
        print_template(rustedvino::model_manager::Config::default_model_template());
        return true;
    }
    if args.iter().any(|a| a == "--device-info") {
        rustedvino::device_inventory::DeviceInventory::probe().print_report();
        return true;
    }
    false
}

/// Marker that `scripts/fill-config-defaults.py` looks for in a binary before
/// running it: only a build that has `--print-default-config` contains it. (The
/// flag's own text is not reliable for this: the optimiser splits short string
/// comparisons.) A server older than 0.9.1 ignores unknown flags and would start for real;
/// since 0.9.1 an unknown flag exits 64 (see `rustedvino::cli`).
#[used]
static TEMPLATE_FLAGS_MARKER: [u8; 31] = *b"RUSTEDVINO-TEMPLATE-FLAGS-V1-OK";

/// Prints a settings template as pretty JSON, or the error and exit code 1.
fn print_template(template: anyhow::Result<serde_json::Value>) {
    match template.and_then(|v| Ok(serde_json::to_string_pretty(&v)?)) {
        Ok(text) => println!("{text}"),
        Err(e) => {
            eprintln!("could not build the settings template: {e}");
            std::process::exit(1);
        }
    }
}

/// Real process entry point. Builds the Tokio runtime explicitly (instead of
/// `#[tokio::main]`'s implicit one) so shutdown can be bounded — see
/// [`PROCESS_EXIT_TIMEOUT`]. `async_main` is unchanged behaviourally; only
/// what happens *after* it returns is different.
fn main() -> std::process::ExitCode {
    // FM-1: classify every argument first. `--help` prints the usage; an argument this binary
    // does not know (a typo, `/?`, `--config x.json`) prints it with the usage and exits 64
    // instead of starting the server with whatever config the search finds. After this every
    // argument is a known, valid-UTF-8 flag, so the `std::env::args()` calls below cannot panic.
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    match rustedvino::cli::classify_args(&args) {
        rustedvino::cli::CliAction::Proceed => {}
        rustedvino::cli::CliAction::Help => {
            print!("{}", rustedvino::cli::usage());
            return std::process::ExitCode::SUCCESS;
        }
        rustedvino::cli::CliAction::Unknown(arg) => {
            eprint!(
                "rustedvino: unknown argument {arg:?}\n\n{}",
                rustedvino::cli::usage()
            );
            return std::process::ExitCode::from(rustedvino::cli::EXIT_USAGE);
        }
    }
    // `--version` / `-V`: answer before anything else (no config, no signal
    // thread, no runtime) so it works on a fresh unpacked bundle.
    if std::env::args()
        .skip(1)
        .any(|a| a == "--version" || a == "-V")
    {
        println!(
            "rustedvino {} ({})",
            env!("CARGO_PKG_VERSION"),
            env!("RV_GIT_HASH")
        );
        return std::process::ExitCode::SUCCESS;
    }
    // `--gpu-probe`: the supervisor's GPU readiness probe child: no config, no
    // signal thread, no runtime; prints one line per GPU and exits 0 / 2.
    if std::env::args().skip(1).any(|a| a == "--gpu-probe") {
        return std::process::ExitCode::from(rustedvino::gpu_probe::run());
    }
    // Take SIGTERM/SIGINT on a dedicated sigwait thread so a foreign
    // (IGC/NEO) signal handler can never run for them - see
    // `shutdown_signals`. Must come before the runtime builds any thread, and
    // not in `--supervise` mode (the supervisor uses tokio's own handler).
    #[cfg(unix)]
    if !std::env::args().any(|a| a == "--supervise")
        && let Err(e) = rustedvino::shutdown_signals::install()
    {
        eprintln!(
            "warning: dedicated shutdown-signal thread unavailable ({e}); using tokio's handler"
        );
    }

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

    let poison = rustedvino::startup::gpu_poison_exit_requested();
    let abandoned = matches!(&result, Err(e)
        if e.downcast_ref::<rustedvino::startup::ShutdownDuringStartup>().is_some());
    let normal_code: u8 = match &result {
        Ok(()) if poison => rustedvino::startup::EXIT_GPU_POISONED,
        Ok(()) => 0,
        // A stop requested while models were still loading is a clean stop.
        Err(_) if abandoned => 0,
        Err(e) => {
            eprintln!("Error: {e:?}");
            if e.downcast_ref::<rustedvino::startup::ConfigFatal>()
                .is_some()
            {
                rustedvino::startup::EXIT_CONFIG_FATAL
            } else {
                1
            }
        }
    };

    // D-01: returning from `main` runs libc `exit()`, which runs the C++ static destructors; with an
    // engine thread still inside a pipeline the two overlap and corrupt the heap (SIGABRT, seen 7 of
    // 7 times in QA). D-04 S7: once model construction has started every exit leaves without them
    // (`engines_started`), also after a clean teardown (decided 2026-10-29). `exit_decision` is the one
    // predicate; with OpenVINO calls still in flight (`ffi_watch`) a clean stop exits 76.
    let in_flight = if poison {
        0
    } else {
        rustedvino::startup::settle_ffi_calls(result.is_ok())
    };
    let decision = rustedvino::startup::exit_decision(
        normal_code,
        rustedvino::startup::engines_started(),
        rustedvino::startup::teardown_incomplete(),
        poison,
        in_flight,
    );
    if decision.skip_destructors {
        rt.shutdown_background();
        rustedvino::startup::exit_without_destructors(decision.code);
    }

    // Consumes `rt` — bounds the wait for any outstanding `spawn_blocking`
    // task instead of blocking forever. See `PROCESS_EXIT_TIMEOUT` doc.
    rt.shutdown_timeout(PROCESS_EXIT_TIMEOUT);
    std::process::ExitCode::from(decision.code)
}

#[allow(clippy::too_many_lines)]
async fn async_main() -> anyhow::Result<()> {
    // Recent-log tail for `GET /v1/admin/logs/tail` (e.g. a UI panel) —
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
            let config_path = rustedvino::startup::resolve_config_path()
                .map_err(|e| rustedvino::startup::config_fatal(&e))?;
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
    let realtime_sessions = std::sync::Arc::clone(&setup.state.realtime_sessions);
    let app = rustedvino::create_router_with_auth(
        setup.state,
        &setup.cors_origins,
        &setup.api_keys,
        &setup.admin_api_keys,
    );

    // Bind address/port come from config (`bind_addr`/`port`; default
    // loopback:11437 — secure by default, T4.2). stormVINO owns 11435;
    // cutover day: RustedVINO takes 11435 (Part 9).
    rustedvino::startup::serve(
        app,
        setup.addr,
        setup.socket,
        &setup.model_manager,
        realtime_sessions,
    )
    .await
}
