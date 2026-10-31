//! `rv version`, `rv start`, `rv stop`, `rv restart` — run the server by hand.
//!
//! These cover the release-bundle case: the server binary sits next to `rv`
//! and nothing else (systemd, a supervisor script) owns it. A server that
//! systemd owns is left to `systemctl`: stopping it behind systemd's back
//! would just get it restarted, or leave the unit in a failed state.
//!
//! Errors follow the compiler's shape: one `error:` line saying what is
//! wrong, then `= note:` / `= help:` lines saying why and what to do.

use std::path::{Path, PathBuf};
use std::time::Duration;

// D-04 S8: what `rv start`/`rv stop` follow in the server's log, and how long. rv cannot depend
// on the server crate (it would link OpenVINO), so these are copies, pinned to the server's
// values by `tests/rv_stop_contract.rs`.

/// How long `stop` waits for the process to exit after SIGTERM when no model load holds the stop
/// up: the server's drain (30 s), the teardown of six engines (8 s each + 2 s) and its exit wait.
pub const STOP_TIMEOUT: Duration = Duration::from_secs(90);
/// How long `stop` waits once the server logs that it is waiting for a model load: its stop
/// wait (`startup::LOAD_SETTLE_CAP`, 600 s) plus drain, teardown and exit; also the systemd
/// example's `TimeoutStopSec`.
pub const STOP_LOAD_TIMEOUT: Duration = Duration::from_mins(12);
/// How long `start` gives a preload with no load deadline of its own (the server's stop wait
/// gives it as long: `startup::LOAD_SETTLE_CAP`).
pub const UNCAPPED_PRELOAD: Duration = Duration::from_mins(10);
/// Part of the server's stop-wait line (`startup::STOP_WAIT_LOG`).
pub const STOP_WAIT_LINE: &str = "stop requested while a model load is running";
/// The server's preload line (`startup::PRELOAD_LOG`), with a `load_timeout_s=` field.
pub const PRELOAD_LINE: &str = "preloading model - the port opens once the preload is done";

use crate::ctx::Ctx;

/// What `GET /health` reported.
struct Health {
    version: String,
    git_hash: String,
}

fn health(ctx: &Ctx) -> Option<Health> {
    let r = ctx.api.keyless().get("/health");
    let v = r.json()?;
    Some(Health {
        version: v.get("version")?.as_str()?.to_owned(),
        git_hash: v
            .get("git_hash")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown")
            .to_owned(),
    })
}

/// Prints a compiler-style error: the headline, then `(label, text)` lines
/// such as `("note", ..)` and `("help", ..)`. Always returns `1`.
fn fail(ctx: &Ctx, headline: &str, lines: &[(&str, String)]) -> i32 {
    eprintln!("{}", ctx.style.red(&format!("error: {headline}")));
    for (label, text) in lines {
        eprintln!("  = {label}: {text}");
    }
    1
}

#[cfg(unix)]
/// The `port` the *server* will bind: the config's own value, else the
/// built-in default. `rv`'s resolved port can differ (`--port`, `RV_PORT`),
/// but the server only ever reads the config.
fn config_port(config: &Path) -> u16 {
    std::fs::read_to_string(config)
        .ok()
        .and_then(|t| {
            serde_json::from_str::<serde_json::Value>(t.trim_start_matches('\u{feff}')).ok()
        })
        .and_then(|v| v.get("port")?.as_u64())
        .and_then(|p| u16::try_from(p).ok())
        .unwrap_or(crate::resolve::DEFAULT_PORT)
}

/// The server executable shipped next to `rv`: the launcher script (sets the
/// library paths) in a bundle, else a bare binary.
fn server_exe() -> Option<PathBuf> {
    let dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    ["rustedvino", "rustedvino-bin"]
        .iter()
        .map(|n| dir.join(n))
        .find(|p| p.is_file())
}

/// `<exe> --version`, trimmed. `None` when it cannot be run.
fn binary_version(exe: &Path) -> Option<String> {
    let out = std::process::Command::new(exe)
        .arg("--version")
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .filter(|s| !s.is_empty())
}

/// Runs `rv version`: the server binary on disk and the server that answers.
#[must_use]
pub fn version(ctx: &Ctx) -> i32 {
    ctx.warn_if_open();
    let on_disk = server_exe().and_then(|e| binary_version(&e));
    let running = health(ctx);
    if ctx.json {
        println!(
            "{}",
            serde_json::json!({
                "binary": on_disk,
                "running": running.as_ref().map(|h| serde_json::json!({
                    "version": h.version, "git_hash": h.git_hash,
                })),
            })
        );
        return 0;
    }
    println!(
        "server binary : {}",
        on_disk.unwrap_or_else(|| "not found next to rv".to_owned())
    );
    match running {
        Some(h) => println!(
            "running server: rustedvino {} ({}) on port {}",
            h.version, h.git_hash, ctx.res.port
        ),
        None => println!("running server: none answering on port {}", ctx.res.port),
    }
    0
}

/// Runs `rv start`; `supervise` passes `--supervise` (respawn after a crash).
#[must_use]
pub fn start(ctx: &Ctx, supervise: bool) -> i32 {
    run::start(ctx, supervise)
}

/// Runs `rv stop`. "Nothing running" is a success, so `restart` works either
/// way. `force` signals a server that systemd runs.
#[must_use]
pub fn stop(ctx: &Ctx, force: bool) -> i32 {
    run::stop(ctx, force)
}

/// Runs `rv restart`: stop (if running), then start, keeping `--supervise`
/// if the old server had it.
#[must_use]
pub fn restart(ctx: &Ctx, force: bool) -> i32 {
    run::restart(ctx, force)
}

#[cfg(not(unix))]
mod run {
    use super::{Ctx, fail};

    fn unsupported(ctx: &Ctx, verb: &str) -> i32 {
        fail(
            ctx,
            &format!("rv {verb} is not supported on this platform yet"),
            &[("help", "run rustedvino.exe directly".to_owned())],
        )
    }

    pub fn start(ctx: &Ctx, _supervise: bool) -> i32 {
        unsupported(ctx, "start")
    }

    pub fn stop(ctx: &Ctx, _force: bool) -> i32 {
        unsupported(ctx, "stop")
    }

    pub fn restart(ctx: &Ctx, _force: bool) -> i32 {
        unsupported(ctx, "restart")
    }
}

#[cfg(unix)]
mod run {
    use std::fmt::Write as _;
    use std::io::{Read, Seek, SeekFrom};
    use std::os::unix::process::CommandExt;
    use std::path::Path;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    use super::{Ctx, config_port, fail, health, server_exe};
    use crate::proc;

    /// How long `start` waits for the server to answer `/health`. Preloaded
    /// models load before the port opens, so this has to be generous; each preload the log
    /// reports extends it (D-04 S8).
    const START_TIMEOUT: Duration = Duration::from_secs(180);
    use super::{PRELOAD_LINE, STOP_LOAD_TIMEOUT, STOP_TIMEOUT, STOP_WAIT_LINE, UNCAPPED_PRELOAD};
    /// Slack after a preload's own load deadline before `start` stops waiting for it.
    const PRELOAD_SLACK: Duration = Duration::from_mins(1);
    const POLL: Duration = Duration::from_millis(500);
    /// How much of the log's end `start` shows when the server dies.
    const LOG_TAIL_BYTES: u64 = 64 * 1024;

    /// True when the process lives in a systemd *service* cgroup. (The
    /// `INVOCATION_ID` environment variable would not do: a server started
    /// from a shell inherits it whenever that shell itself runs under
    /// systemd.) A process started from a terminal sits in a `.scope`.
    /// Linux only; unreadable → false.
    #[cfg(target_os = "linux")]
    fn managed_by_systemd(pid: u32) -> bool {
        std::fs::read_to_string(format!("/proc/{pid}/cgroup")).is_ok_and(|t| cgroup_is_service(&t))
    }

    #[cfg(not(target_os = "linux"))]
    fn managed_by_systemd(_pid: u32) -> bool {
        false
    }

    /// Whether a `/proc/<pid>/cgroup` listing places the process in a `.service`.
    #[cfg(target_os = "linux")]
    fn cgroup_is_service(text: &str) -> bool {
        text.lines()
            .filter_map(|l| l.rsplit(':').next())
            .any(|path| {
                path.rsplit('/')
                    .next()
                    .is_some_and(|c| c.ends_with(".service"))
            })
    }

    #[cfg(target_os = "linux")]
    fn pid_alive(pid: u32) -> bool {
        Path::new(&format!("/proc/{pid}")).exists()
    }

    #[cfg(not(target_os = "linux"))]
    fn pid_alive(_pid: u32) -> bool {
        false
    }

    /// Whether the process was started with `--supervise`.
    #[cfg(target_os = "linux")]
    fn was_supervised(pid: u32) -> bool {
        std::fs::read(format!("/proc/{pid}/cmdline"))
            .is_ok_and(|raw| raw.split(|b| *b == 0).any(|a| a == b"--supervise"))
    }

    #[cfg(not(target_os = "linux"))]
    fn was_supervised(_pid: u32) -> bool {
        false
    }

    /// PIDs of server processes whose working directory is `dir`. The bundle
    /// launcher `cd`s into its own directory, so this finds a server run by
    /// hand from the same bundle even while it is still loading models and
    /// has not opened its port yet. Linux only.
    #[cfg(target_os = "linux")]
    fn servers_running_in(dir: &Path) -> Vec<u32> {
        let Ok(entries) = std::fs::read_dir("/proc") else {
            return Vec::new();
        };
        let want = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
        entries
            .flatten()
            .filter_map(|e| e.file_name().to_str()?.parse::<u32>().ok())
            .filter(|pid| {
                let argv0 = std::fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
                let argv0 = String::from_utf8_lossy(argv0.split(|b| *b == 0).next().unwrap_or(&[]))
                    .into_owned();
                let base = argv0.rsplit('/').next().unwrap_or("");
                (base == "rustedvino" || base == "rustedvino-bin")
                    && std::fs::read_link(format!("/proc/{pid}/cwd")).is_ok_and(|cwd| cwd == want)
            })
            .collect()
    }

    #[cfg(not(target_os = "linux"))]
    fn servers_running_in(_dir: &Path) -> Vec<u32> {
        Vec::new()
    }

    /// The process that holds this port. Only a port holder counts: the
    /// PID-file and command-line fallbacks can name another server on the
    /// same machine.
    fn port_holder(ctx: &Ctx) -> Option<proc::ProcState> {
        let state = proc::probe(ctx.res.port, &ctx.pid_file);
        (state.alive && state.source == proc::Source::Port && state.main_pid.is_some())
            .then_some(state)
    }

    /// The help line of the "already running" refusal. `rv stop` finds a still-loading server only
    /// when `rv start` launched it with this config (PID file); for one started by hand or by
    /// another script it cannot, so then only `kill` works.
    fn refusal_help(stop_works: bool, pid: u32) -> String {
        if stop_works {
            "wait and run `rv status`, watch the log, or stop it with `rv stop` (that also works \
             while models are still loading)"
                .to_owned()
        } else {
            format!(
                "wait and run `rv status`, watch the log, or end that process by hand with \
                 `kill {pid}` (`rv stop` cannot identify a server it did not start with this config)"
            )
        }
    }

    pub fn start(ctx: &Ctx, supervise: bool) -> i32 {
        if let Some(h) = health(ctx) {
            println!(
                "already running: rustedvino {} ({}) on port {}",
                h.version, h.git_hash, ctx.res.port
            );
            return 0;
        }
        let config = &ctx.res.config_file;
        if !config.is_file() {
            return fail(
                ctx,
                &format!("no config file at {}", config.display()),
                &[
                    (
                        "note",
                        "rv looks for $RUSTEDVINO_CONFIG, then a config.json next to rv, then \
                         /opt/rustedvino/config.json"
                            .to_owned(),
                    ),
                    (
                        "help",
                        "copy config.example.json to config.json next to rv and edit it".to_owned(),
                    ),
                    (
                        "help",
                        "or point RUSTEDVINO_CONFIG at your config file".to_owned(),
                    ),
                ],
            );
        }
        let server_port = config_port(config);
        if server_port != ctx.res.port {
            return fail(
                ctx,
                &format!(
                    "port {} was requested, but the server would listen on {server_port}",
                    ctx.res.port
                ),
                &[
                    (
                        "note",
                        format!(
                            "the server takes its port from {} only; --port and RV_PORT steer rv, \
                             not the server",
                            config.display()
                        ),
                    ),
                    (
                        "help",
                        format!(
                            "set \"port\": {} in the config, or drop --port / RV_PORT",
                            ctx.res.port
                        ),
                    ),
                ],
            );
        }
        let Some(exe) = server_exe() else {
            return fail(
                ctx,
                "no rustedvino executable next to rv",
                &[(
                    "help",
                    "run rv from the unpacked bundle directory, or start the server by hand"
                        .to_owned(),
                )],
            );
        };
        if let Some(dir) = exe.parent() {
            let others = servers_running_in(dir);
            if let Some(pid) = others.first() {
                return fail(
                    ctx,
                    &format!("a rustedvino from this directory is already running (pid {pid})"),
                    &[
                        (
                            "note",
                            format!(
                                "it has not opened port {} yet - models load before the port \
                                 opens, which can take minutes",
                                ctx.res.port
                            ),
                        ),
                        ("help", refusal_help(starting_server(ctx).is_some(), *pid)),
                    ],
                );
            }
        }
        ensure_keys_file(ctx);
        spawn_and_wait(ctx, &exe, supervise)
    }

    /// A fresh random admin key: `rv-` plus 48 hex digits from the kernel's random source.
    fn random_admin_key() -> std::io::Result<String> {
        let mut bytes = [0_u8; 24];
        std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
        let mut key = String::from("rv-");
        for b in bytes {
            let _ = write!(key, "{b:02x}");
        }
        Ok(key)
    }

    /// The body of a freshly created keys file: no inference keys (inference stays open, as in the
    /// shipped example) and ONE random admin key, so the admin API and every browser tab that can
    /// reach the port are not an open door. `rv` reads `admin_api_keys[0]` from this file itself.
    fn new_keys_file_text(admin_key: &str) -> String {
        format!("{{\n  \"api_keys\": [],\n  \"admin_api_keys\": [\"{admin_key}\"]\n}}\n")
    }

    /// Creates `path` (mode 0600) with a random admin key unless something already sits there.
    /// `Ok(true)` when it was created. The server reads the file only at start, and without it every
    /// `/v1/admin/*` route answers 503 for the whole run, so `rv start` makes sure it exists first.
    fn create_keys_file_if_missing(path: &Path) -> std::io::Result<bool> {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let text = new_keys_file_text(&random_admin_key()?);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
        {
            Ok(mut f) => {
                f.write_all(text.as_bytes())?;
                Ok(true)
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Makes sure the keys file exists before the server starts, and says so when it had to create it.
    fn ensure_keys_file(ctx: &Ctx) {
        let path = &ctx.res.keys_file;
        match create_keys_file_if_missing(path) {
            Ok(true) => println!(
                "created {} with a random admin key (rv reads it from there by itself; inference \
                 needs no key until you add `api_keys`)",
                path.display()
            ),
            Ok(false) => {}
            Err(e) => eprintln!(
                "warning: could not create {} ({e}); /v1/admin/* and `rv add` will answer 503 until \
                 the file exists and the server is restarted",
                path.display()
            ),
        }
    }

    /// Opens the log, spawns the server detached, and waits for it.
    fn spawn_and_wait(ctx: &Ctx, exe: &Path, supervise: bool) -> i32 {
        let config = &ctx.res.config_file;
        let log_path = config.with_file_name("rustedvino.log");
        let log = match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            Ok(f) => f,
            Err(e) => {
                return fail(
                    ctx,
                    &format!("cannot open the log file {}", log_path.display()),
                    &[
                        ("note", e.to_string()),
                        ("help", "make that directory writable".to_owned()),
                    ],
                );
            }
        };
        let Ok(log_err) = log.try_clone() else {
            return fail(ctx, "cannot duplicate the log file handle", &[]);
        };
        // The log is append-only across runs: remember where this run starts so
        // a failure shows only this run's lines (0.8.3, F3).
        let run_start = log.metadata().map_or(0, |m| m.len());
        let mut cmd = Command::new(exe);
        cmd.env("RUSTEDVINO_CONFIG", config)
            .stdin(Stdio::null())
            .stdout(log)
            .stderr(log_err)
            // Own process group: a Ctrl-C in this terminal, or rv exiting,
            // must not take the server with it.
            .process_group(0);
        if supervise {
            cmd.arg("--supervise");
        }
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return fail(
                    ctx,
                    &format!("cannot run {}", exe.display()),
                    &[("note", e.to_string())],
                );
            }
        };
        // Best effort: lets `rv stop` find the server while it is still loading
        // models and not yet listening (D-02).
        let _ = std::fs::write(&ctx.pid_file, child.id().to_string());
        println!(
            "starting rustedvino (pid {}){}, log: {}",
            child.id(),
            if supervise { " under --supervise" } else { "" },
            log_path.display()
        );
        wait_until_up(ctx, &mut child, &log_path, run_start)
    }

    /// True when the server exited cleanly while `rv start` was still waiting for it. The server
    /// exits 0 during start only when it was asked to shut down while loading models, so this is
    /// a stop, not a failure to report as an error (D-04).
    fn stopped_before_ready(status: std::process::ExitStatus) -> bool {
        status.success()
    }

    /// Waits for the freshly spawned server to answer `/health`, or to die.
    fn wait_until_up(
        ctx: &Ctx,
        child: &mut std::process::Child,
        log_path: &Path,
        run_start: u64,
    ) -> i32 {
        let t0 = Instant::now();
        let mut deadline = t0 + START_TIMEOUT;
        let mut offset = run_start;
        let mut stopping = false;
        while Instant::now() < deadline {
            if let Ok(Some(status)) = child.try_wait() {
                if stopped_before_ready(status) {
                    eprintln!(
                        "rustedvino was stopped before it was ready (it was asked to shut down \
                         while loading models)"
                    );
                    return 1;
                }
                let tail = read_run_log(log_path, run_start);
                let mut notes = Vec::new();
                if let Some(reason) = startup_reason(&tail) {
                    notes.push(("reason", reason));
                }
                notes.push(("note", "the last lines of its log follow".to_owned()));
                let code = fail(
                    ctx,
                    &format!("the server exited during start ({status})"),
                    &notes,
                );
                print_lines_tail(&tail);
                return code;
            }
            if let Some(h) = health(ctx) {
                println!(
                    "{} rustedvino {} ({}) is up on port {} after {:.0} s",
                    ctx.style.ok(),
                    h.version,
                    h.git_hash,
                    ctx.res.port,
                    t0.elapsed().as_secs_f64()
                );
                // Step 8: a server that booted with configuration problems says so here.
                if ctx.res.admin_usable()
                    && let Some(a) = ctx.api.get("/v1/admin/models").json()
                {
                    let (warnings, _) = crate::commands::status::startup_report_lines(&a["server"]);
                    for w in warnings {
                        println!("{} {}", ctx.style.warn(), ctx.style.yellow(&w));
                    }
                }
                return 0;
            }
            // D-04 S8: a preload the log reports, or a stop waiting for one, moves the deadline.
            let lines = new_log_lines(log_path, &mut offset);
            for secs in preload_allowances(&lines) {
                let allowance = if secs == 0 {
                    UNCAPPED_PRELOAD
                } else {
                    Duration::from_secs(secs)
                };
                deadline = deadline.max(Instant::now() + allowance + PRELOAD_SLACK);
            }
            if !stopping && lines.contains(STOP_WAIT_LINE) {
                stopping = true;
                deadline = deadline.max(Instant::now() + STOP_LOAD_TIMEOUT);
                println!(
                    "rustedvino was asked to stop and is waiting for a model load to finish so \
                     the GPU is released cleanly (up to 10 min)"
                );
            }
            std::thread::sleep(POLL);
        }
        fail(
            ctx,
            &format!(
                "no answer on port {} after {:.0} s",
                ctx.res.port,
                t0.elapsed().as_secs_f64()
            ),
            &[
                (
                    "note",
                    "the server is still running; it may be loading models".to_owned(),
                ),
                (
                    "help",
                    format!("watch {} or run `rv status`", log_path.display()),
                ),
            ],
        )
    }

    /// This run's part of the log (from `run_start`), at most the final
    /// [`LOG_TAIL_BYTES`] - the file is append-only and never rotated.
    fn read_run_log(path: &Path, run_start: u64) -> String {
        let Ok(mut f) = std::fs::File::open(path) else {
            return String::new();
        };
        let len = f.metadata().map_or(0, |m| m.len());
        let from = len.saturating_sub(LOG_TAIL_BYTES).max(run_start.min(len));
        let mut buf = Vec::new();
        if f.seek(SeekFrom::Start(from)).is_err() || f.read_to_end(&mut buf).is_err() {
            return String::new();
        }
        let text = String::from_utf8_lossy(&buf).into_owned();
        // Truncated by the byte cap rather than at the run's start: the first
        // line is partial.
        if from > run_start
            && let Some((_, rest)) = text.split_once('\n')
        {
            return rest.to_owned();
        }
        text
    }

    /// The complete lines appended to `path` since `*offset`; moves `*offset` past them (a last
    /// line still being written is left for the next call).
    fn new_log_lines(path: &Path, offset: &mut u64) -> String {
        let Ok(mut f) = std::fs::File::open(path) else {
            return String::new();
        };
        // Truncated or rotated under us (copytruncate): follow it from its new start.
        if f.metadata().is_ok_and(|m| m.len() < *offset) {
            *offset = 0;
        }
        let mut buf = Vec::new();
        if f.seek(SeekFrom::Start(*offset)).is_err() || f.read_to_end(&mut buf).is_err() {
            return String::new();
        }
        let Some(end) = buf.iter().rposition(|&b| b == b'\n') else {
            return String::new();
        };
        *offset += u64::try_from(end + 1).unwrap_or(0);
        String::from_utf8_lossy(&buf[..=end]).into_owned()
    }

    /// The load deadline (`load_timeout_s=`, `0` = none) of each preload `lines` report.
    fn preload_allowances(lines: &str) -> Vec<u64> {
        lines
            .lines()
            .filter(|l| l.contains(PRELOAD_LINE))
            .map(|l| {
                l.split_whitespace()
                    .find_map(|w| w.strip_prefix("load_timeout_s="))
                    .and_then(|n| n.parse().ok())
                    .unwrap_or(0)
            })
            .collect()
    }

    /// The one-sentence cause of a failed start: the fatal `Error: ...` line the
    /// binary prints on exit, else the last ERROR-level log line.
    fn startup_reason(text: &str) -> Option<String> {
        if let Some(l) = text.lines().rev().find(|l| l.starts_with("Error: ")) {
            return Some(l.trim_start_matches("Error: ").trim().to_owned());
        }
        text.lines().rev().find(|l| l.contains(" ERROR ")).map(|l| {
            l.split_once(" ERROR ")
                .map_or(l, |(_, m)| m)
                .trim()
                .to_owned()
        })
    }

    fn print_lines_tail(text: &str) {
        let lines: Vec<&str> = text.lines().collect();
        for l in &lines[lines.len().saturating_sub(15)..] {
            eprintln!("  {l}");
        }
    }

    /// A server `rv start` launched that is alive but not listening yet (it is
    /// still preloading models, which can take minutes): named by the PID file
    /// and confirmed to be running THIS config, so another server on the same
    /// machine is never taken for it.
    fn starting_server(ctx: &Ctx) -> Option<proc::ProcState> {
        let state = proc::probe(ctx.res.port, &ctx.pid_file);
        (state.alive
            && state.source == proc::Source::PidFile
            && state
                .main_pid
                .is_some_and(|p| proc::runs_config(p, &ctx.res.config_file)))
        .then_some(state)
    }

    pub fn stop(ctx: &Ctx, force: bool) -> i32 {
        let port_state = port_holder(ctx).or_else(|| {
            if health(ctx).is_some() {
                return None;
            }
            let s = starting_server(ctx)?;
            println!(
                "{}",
                ctx.style.dim(&format!(
                    "the server (pid {}) is still loading models and not listening yet - stopping it",
                    s.main_pid.unwrap_or_default()
                ))
            );
            Some(s)
        });
        let Some(state) = port_state else {
            if health(ctx).is_some() {
                return fail(
                    ctx,
                    &format!(
                        "a server answers on port {} but its process cannot be identified",
                        ctx.res.port
                    ),
                    &[(
                        "help",
                        "it belongs to another user or container; stop it from there".to_owned(),
                    )],
                );
            }
            println!("not running (nothing on port {})", ctx.res.port);
            return 0;
        };
        let Some(pid) = state.main_pid else {
            return 1;
        };
        if managed_by_systemd(pid) || state.worker_pid.is_some_and(managed_by_systemd) {
            if !force {
                return fail(
                    ctx,
                    &format!("pid {pid} runs under a systemd service"),
                    &[
                        (
                            "note",
                            "stopping it behind systemd's back usually gets it restarted"
                                .to_owned(),
                        ),
                        (
                            "help",
                            "`systemctl --user stop rustedvino` (or without --user for a system \
                             unit)"
                                .to_owned(),
                        ),
                        (
                            "help",
                            "if rv itself was started from some other service (tmux, cron, ssh), \
                             `rv stop --force` signals it anyway"
                                .to_owned(),
                        ),
                    ],
                );
            }
            eprintln!(
                "{}",
                ctx.style.yellow(
                    "warning: pid is under a systemd service; --force given, signalling it"
                )
            );
        }
        signal_and_wait(ctx, pid)
    }

    /// SIGTERM `pid` (via `kill`, which keeps `unsafe` and libc out of rv) and
    /// wait for it to exit.
    fn signal_and_wait(ctx: &Ctx, pid: u32) -> i32 {
        // D-04 S8: the server says when its stop waits for a model load; follow its log from here.
        let log_path = ctx.res.config_file.with_file_name("rustedvino.log");
        let mut offset = std::fs::metadata(&log_path).map_or(0, |m| m.len());
        let sent = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status();
        if !sent.is_ok_and(|s| s.success()) {
            return fail(
                ctx,
                &format!("could not signal pid {pid}"),
                &[(
                    "note",
                    "it may belong to another user, or have just exited".to_owned(),
                )],
            );
        }
        println!("stopping rustedvino (pid {pid}) ...");
        let t0 = Instant::now();
        let mut deadline = t0 + STOP_TIMEOUT;
        let mut waiting_for_load = false;
        while Instant::now() < deadline {
            if !pid_alive(pid) {
                println!(
                    "{} stopped after {:.0} s",
                    ctx.style.ok(),
                    t0.elapsed().as_secs_f64()
                );
                return 0;
            }
            if !waiting_for_load && new_log_lines(&log_path, &mut offset).contains(STOP_WAIT_LINE) {
                waiting_for_load = true;
                deadline = t0 + STOP_LOAD_TIMEOUT;
                println!(
                    "{} {}",
                    ctx.style.warn(),
                    ctx.style.yellow(
                        "the server is waiting for a model load to finish so the GPU is released \
                         cleanly (up to 10 min) - do not kill it"
                    )
                );
            }
            std::thread::sleep(POLL);
        }
        fail(
            ctx,
            &format!(
                "pid {pid} is still running {:.0} s after SIGTERM",
                t0.elapsed().as_secs_f64()
            ),
            &[
                (
                    "note",
                    "a stop waits up to 10 min for a model load that is still running (the \
                     server's log says \"waiting for it to finish\"; rv follows only the log next \
                     to config.json) - give it that time. Otherwise a request may be stuck in a \
                     device call (an NPU Whisper hang, or a GPU driver hang after an earlier kill)"
                        .to_owned(),
                ),
                (
                    "help",
                    "do NOT kill -9 a server with a GPU model loaded: on 2026-10-23 that left \
                     the B60 hung for ~55 min. Check `rv status` and the kernel log \
                     (`journalctl -k | grep xe`), wait for the driver to reset, or reboot"
                        .to_owned(),
                ),
                (
                    "help",
                    format!(
                        "CPU/NPU-only server, no GPU model loaded or loading: kill -9 {pid} is safe"
                    ),
                ),
            ],
        )
    }

    pub fn restart(ctx: &Ctx, force: bool) -> i32 {
        let supervised = port_holder(ctx).is_some_and(|s| {
            [s.main_pid, s.worker_pid]
                .into_iter()
                .flatten()
                .any(was_supervised)
        });
        let code = stop(ctx, force);
        if code != 0 {
            return code;
        }
        start(ctx, supervised)
    }

    #[cfg(all(test, target_os = "linux"))]
    #[allow(clippy::unwrap_used)]
    mod tests {
        use super::{
            cgroup_is_service, create_keys_file_if_missing, new_log_lines, preload_allowances,
            read_run_log, startup_reason, stopped_before_ready,
        };
        use std::os::unix::fs::PermissionsExt;
        use std::os::unix::process::ExitStatusExt;
        use std::process::ExitStatus;

        /// Opus review of 0.8.4: the refusal must not promise an `rv stop` that cannot find the
        /// server (one started by hand or by another script).
        #[test]
        fn refusal_help_only_promises_rv_stop_when_it_works() {
            assert!(super::refusal_help(true, 7).contains("`rv stop`"));
            assert!(!super::refusal_help(true, 7).contains("kill"));
            let h = super::refusal_help(false, 4242);
            assert!(h.contains("kill 4242"), "{h}");
            assert!(h.contains("cannot identify"), "{h}");
        }

        /// D-04 S8: `start` reads each preload's own deadline off the server's preload line; a line
        /// without the field (or `0`) is a preload with no deadline.
        #[test]
        fn preload_lines_carry_their_load_deadline() {
            let log = "2026-10-29T09:30:01Z  INFO rustedvino::model_manager: preloading model - the \
                       port opens once the preload is done model=coder-30b load_timeout_s=600\n\
                       2026-10-29T09:30:02Z  INFO rustedvino::startup: unrelated load_timeout_s=5\n\
                       x INFO preloading model - the port opens once the preload is done model=a\n";
            assert_eq!(preload_allowances(log), vec![600, 0]);
        }

        /// D-04 S8: the log follower returns complete lines only and resumes after them.
        #[test]
        fn new_log_lines_leaves_a_partial_line_for_later() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("rustedvino.log");
            std::fs::write(&path, "old\n").unwrap();
            let mut offset = 4;
            assert_eq!(new_log_lines(&path, &mut offset), "");
            std::fs::write(&path, "old\nstop requested\npart").unwrap();
            assert_eq!(new_log_lines(&path, &mut offset), "stop requested\n");
            assert_eq!(offset, 19);
            std::fs::write(&path, "old\nstop requested\npartial\n").unwrap();
            assert_eq!(new_log_lines(&path, &mut offset), "partial\n");
            // Truncated under us: read from the start again.
            std::fs::write(&path, "new\n").unwrap();
            assert_eq!(new_log_lines(&path, &mut offset), "new\n");
        }

        /// D-04: a clean exit during start means the server was told to shut down while it was
        /// loading models (the server exits 0 only then); any other status is a real failure.
        #[test]
        fn exit_zero_during_start_is_a_stop_not_a_failure() {
            assert!(stopped_before_ready(ExitStatus::from_raw(0)));
            assert!(!stopped_before_ready(ExitStatus::from_raw(1 << 8)));
            // killed by a signal (SIGKILL)
            assert!(!stopped_before_ready(ExitStatus::from_raw(9)));
        }

        #[test]
        fn startup_reason_prefers_the_fatal_error_line() {
            let log = "2026 INFO device inventory\n2026 ERROR load failed x\nError: config fatal: bad vram\n";
            assert_eq!(
                startup_reason(log).as_deref(),
                Some("config fatal: bad vram")
            );
            assert_eq!(
                startup_reason("2026 INFO a\n2026-10-14T09:00:00Z ERROR boom\n").as_deref(),
                Some("boom")
            );
            assert_eq!(startup_reason("2026 INFO only info\n"), None);
        }

        #[test]
        fn run_log_skips_earlier_runs() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("rustedvino.log");
            std::fs::write(&path, "old run Error: stale\n").unwrap();
            let start = std::fs::metadata(&path).unwrap().len();
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            std::io::Write::write_all(&mut f, b"new run line\n").unwrap();
            assert_eq!(read_run_log(&path, start), "new run line\n");
        }

        #[test]
        fn a_missing_keys_file_is_created_private_and_never_overwritten() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("config.keys.json");
            assert!(create_keys_file_if_missing(&path).unwrap());
            let text = std::fs::read_to_string(&path).unwrap();
            let v: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!(v["api_keys"], serde_json::json!([]));
            let keys = v["admin_api_keys"].as_array().unwrap();
            assert_eq!(keys.len(), 1);
            let key = keys[0].as_str().unwrap();
            assert!(key.starts_with("rv-") && key.len() == 3 + 48, "{key}");
            assert!(key[3..].chars().all(|c| c.is_ascii_hexdigit()));
            assert_ne!(
                super::random_admin_key().unwrap(),
                key,
                "keys must not repeat"
            );
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
            std::fs::write(&path, "{\"api_keys\":[\"k\"],\"admin_api_keys\":[]}").unwrap();
            assert!(!create_keys_file_if_missing(&path).unwrap());
            assert!(std::fs::read_to_string(&path).unwrap().contains("\"k\""));
        }

        #[test]
        fn a_service_cgroup_is_systemd_managed() {
            assert!(cgroup_is_service(
                "0::/user.slice/user-1000.slice/user@1000.service/app.slice/rustedvino.service\n"
            ));
        }

        #[test]
        fn a_terminal_scope_is_not() {
            // The user manager is itself `user@1000.service`, but the process
            // sits in the terminal's scope beneath it.
            assert!(!cgroup_is_service(
                "0::/user.slice/user-1000.slice/user@1000.service/app.slice/ptyxis-spawn-x.scope\n"
            ));
            assert!(!cgroup_is_service(""));
        }
    }
}
