//! `rv` binary entry point: the admin CLI.

// Doc comments on clap items become `--help` text, where markdown backticks
// would show literally — so doc_markdown is relaxed for this file only.
#![allow(clippy::doc_markdown)]

use std::process::ExitCode;

use clap::{CommandFactory, Parser, Subcommand};
use rv::ansi::Style;
use rv::commands::config::{self, PatchArgs, PreloadArgs};
use rv::commands::lifecycle::{self, AddArgs, LoadArgs};
use rv::commands::{check, logs, metrics, models, ops, service, status};
use rv::ctx::{Common, Ctx};
use rv::resolve::{Inputs, resolve};

/// RustedVINO management CLI.
#[derive(Parser)]
#[command(name = "rv", version, about = "RustedVINO management CLI")]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// the server binary on disk and the server that is answering
    Version(Common),
    /// start the server from the bundle (detached, log next to config.json)
    Start {
        /// respawn the server after a crash (passes --supervise)
        #[arg(long)]
        supervise: bool,
        #[command(flatten)]
        common: Common,
    },
    /// stop a server started by hand (refuses one that systemd runs)
    Stop {
        /// signal it even though it sits in a systemd service cgroup
        #[arg(long)]
        force: bool,
        #[command(flatten)]
        common: Common,
    },
    /// stop, then start (keeps --supervise if the old server had it)
    Restart {
        /// signal the old server even though it sits in a systemd service cgroup
        #[arg(long)]
        force: bool,
        #[command(flatten)]
        common: Common,
    },
    /// full server report; with a model id, that model's state and sampling profile
    Status {
        /// model id (optional): show only this model, with its sampling profile
        model: Option<String>,
        #[command(flatten)]
        common: Common,
    },
    /// every registered model, any state
    List(Common),
    /// loaded models only, with live load
    Ps(Common),
    /// dry-run admission check for a model
    Check {
        /// model id
        model: String,
        #[command(flatten)]
        common: Common,
    },
    /// the server's /metrics (admin key), with a speed line per loaded model first
    #[command(
        long_about = "Fetches /metrics with rv's admin key. First one line per loaded model: \
                      effective prefill p50 (tok/s, from the rustedvino_prefill_tokens_per_second \
                      buckets summed over modality; prefix-cache hits included, prompts of 256+ \
                      tokens only) and decode tok/s; '-' when the model has no sample yet, \
                      'unknown' against a server older than 0.9.1. Then the raw samples, \
                      unlabelled ones included. key_suffix label values are masked to their \
                      last 2 characters. --json prints {models: [...], samples: [{name, labels, \
                      value}]}."
    )]
    Metrics {
        /// show only the samples whose line contains this text (plain substring, not a regex;
        /// matched after masking; the per-model lines are always shown)
        #[arg(long)]
        grep: Option<String>,
        #[command(flatten)]
        common: Common,
    },
    /// tail recent server log lines
    Logs {
        /// number of lines
        #[arg(short, default_value_t = 200)]
        n: u32,
        #[command(flatten)]
        common: Common,
    },
    /// register an on-disk model (rv list) and load it
    Add {
        #[command(flatten)]
        args: AddArgs,
        #[command(flatten)]
        common: Common,
    },
    /// load a model into VRAM
    Load {
        #[command(flatten)]
        args: LoadArgs,
        #[command(flatten)]
        common: Common,
    },
    /// evict a model from VRAM (stays registered)
    #[command(
        long_about = "A model with a request running is not evicted: the server refuses (409) \
                      and the request finishes undisturbed. --force drains it (up to 2 minutes) \
                      and then aborts what is left; rv prints the running requests first, then \
                      whether they finished during the drain or were still running at the deadline (\"unknown\" \
                      against a server older than 0.9.1)."
    )]
    Unload {
        /// model id
        model: String,
        /// evict even if a request is running (it is drained, then aborted); prints what it did
        #[arg(long)]
        force: bool,
        #[command(flatten)]
        common: Common,
    },
    /// remove a model from the registry + config.json
    #[command(alias = "forget")]
    Deregister {
        /// model id
        model: String,
        /// don't prompt for confirmation
        #[arg(short = 'y', long)]
        yes: bool,
        /// evict even if a request is running (it is drained, then aborted); prints what it did
        #[arg(long)]
        force: bool,
        #[command(flatten)]
        common: Common,
    },
    /// pick up config.json edits — registers new/changed entries as
    /// NotLoaded, loads/evicts nothing
    ReloadConfig(Common),
    /// update fields of an already-registered model
    Patch {
        #[command(flatten)]
        args: PatchArgs,
        #[command(flatten)]
        common: Common,
    },
    /// replace the persisted preload list
    #[command(
        long_about = "Either --from-live (snapshot the currently-loaded set) or an explicit list \
                      of model ids. Persists to config.json; takes effect on the next restart, \
                      loads/evicts nothing now."
    )]
    Preload {
        #[command(flatten)]
        args: PreloadArgs,
        #[command(flatten)]
        common: Common,
    },
    /// snapshot the currently-loaded set as the new boot default (shortcut for
    /// `rv preload --from-live`)
    SaveDefaults {
        /// don't prompt for confirmation
        #[arg(short = 'y', long)]
        yes: bool,
        #[command(flatten)]
        common: Common,
    },
    /// evict and reload a loaded model with a new KV-cache size
    Resize {
        /// model id
        model: String,
        /// the new KV-cache pool target in GiB
        #[arg(long, allow_hyphen_values = true)]
        kv_cache_gb: f64,
        /// don't prompt for confirmation
        #[arg(short = 'y', long)]
        yes: bool,
        /// resize even if a request is running (it is drained, then aborted); prints what it did
        #[arg(long)]
        force: bool,
        #[command(flatten)]
        common: Common,
    },
    /// audit every registered model's file completeness (loads nothing)
    Completeness(Common),
    /// re-read the keys file and hot-swap the API/admin keys
    ReloadKeys(Common),
    /// the realtime voice-flow model pin
    VoicePin(Common),
    /// list live realtime WebSocket sessions
    Sessions(Common),
    /// one realtime session's full snapshot, as JSON
    Session {
        /// session id (UUID, see `rv sessions`)
        id: String,
        #[command(flatten)]
        common: Common,
    },
    /// force-close a live realtime session
    SessionKill {
        /// session id (UUID, see `rv sessions`)
        id: String,
        /// don't prompt for confirmation
        #[arg(short = 'y', long)]
        yes: bool,
        #[command(flatten)]
        common: Common,
    },
    /// Print what rv would use: config, keys, port, admin-key presence.
    /// Never prints the key itself.
    #[command(hide = true)]
    Resolve {
        /// Server port (overrides RV_PORT and the config's `port`).
        #[arg(long)]
        port: Option<u16>,
        /// Admin key (overrides RV_ADMIN_KEY and the keys file).
        #[arg(long)]
        admin_key: Option<String>,
    },
}

fn print_resolved(port: Option<u16>, admin_key: Option<String>) -> i32 {
    match resolve(&Inputs::from_process(port, admin_key)) {
        Ok(r) => {
            println!(
                "config      : {} ({})",
                r.config_file.display(),
                if r.config_readable {
                    "read"
                } else {
                    "missing or unreadable"
                }
            );
            println!("models_dir  : {}", r.models_dir.as_deref().unwrap_or("-"));
            println!(
                "keys        : {} ({}; {} inference, {} admin)",
                r.keys_file.display(),
                if r.keys_present { "present" } else { "absent" },
                r.api_key_count,
                r.admin_key_count
            );
            println!(
                "admin key   : {}",
                if r.admin_key.is_some() {
                    "available"
                } else if r.keys_file_is_empty() {
                    "none needed - the keys file holds no keys, so the server is OPEN and NOT SECURE"
                } else {
                    "none"
                }
            );
            println!("server      : {}", r.base_url());
            0
        }
        Err(e) => {
            eprintln!("error: {e}");
            2
        }
    }
}

/// Runs a verb inside the vertical margin the Python script prints around
/// every command's output (skipped with `--json`, which must stay one clean
/// parseable value).
fn with_margin(common: &Common, f: impl FnOnce(&Ctx) -> i32) -> i32 {
    let style = Style::detect();
    let ctx = match Ctx::new(common, style) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{}", style.red(&format!("error: {e}")));
            return 2;
        }
    };
    if !ctx.json {
        println!();
    }
    let code = f(&ctx);
    if !ctx.json {
        println!();
    }
    code
}

/// Restores the default SIGPIPE action (Rust's runtime ignores it), so `rv ps | head`
/// ends quietly like any Unix tool instead of panicking on a broken pipe.
#[cfg(unix)]
fn restore_sigpipe() {
    // SAFETY: `signal` with SIG_DFL only resets the disposition for this signal; it runs
    // first thing in `main`, before any thread exists and before any handler is installed.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

fn main() -> ExitCode {
    #[cfg(unix)]
    restore_sigpipe();
    let cli = Cli::parse();
    let code = match cli.cmd {
        // Bare `rv` prints help instead of erroring, like the Python script.
        None => {
            let _ = Cli::command().print_help();
            println!();
            0
        }
        Some(Cmd::Resolve { port, admin_key }) => print_resolved(port, admin_key),
        Some(Cmd::Version(c)) => with_margin(&c, service::version),
        Some(Cmd::Start { supervise, common }) => {
            with_margin(&common, |ctx| service::start(ctx, supervise))
        }
        Some(Cmd::Stop { force, common }) => with_margin(&common, |ctx| service::stop(ctx, force)),
        Some(Cmd::Restart { force, common }) => {
            with_margin(&common, |ctx| service::restart(ctx, force))
        }
        Some(Cmd::Status {
            model: None,
            common,
        }) => with_margin(&common, status::run),
        Some(Cmd::Status {
            model: Some(id),
            common,
        }) => with_margin(&common, |ctx| status::model(ctx, &id)),
        Some(Cmd::List(c)) => with_margin(&c, models::list),
        Some(Cmd::Ps(c)) => with_margin(&c, models::ps),
        Some(Cmd::Check { model, common }) => with_margin(&common, |ctx| check::run(ctx, &model)),
        Some(Cmd::Metrics { grep, common }) => {
            with_margin(&common, |ctx| metrics::run(ctx, grep.as_deref()))
        }
        Some(Cmd::Logs { n, common }) => with_margin(&common, |ctx| logs::run(ctx, n)),
        Some(Cmd::Add { args, common }) => with_margin(&common, |ctx| lifecycle::add(ctx, &args)),
        Some(Cmd::Load { args, common }) => with_margin(&common, |ctx| lifecycle::load(ctx, &args)),
        Some(Cmd::Unload {
            model,
            force,
            common,
        }) => with_margin(&common, |ctx| lifecycle::unload(ctx, &model, force)),
        Some(Cmd::Deregister {
            model,
            yes,
            force,
            common,
        }) => with_margin(&common, |ctx| {
            lifecycle::deregister(ctx, &model, yes, force)
        }),
        Some(Cmd::ReloadConfig(c)) => with_margin(&c, config::reload_config),
        Some(Cmd::Patch { args, common }) => with_margin(&common, |ctx| config::patch(ctx, &args)),
        Some(Cmd::Preload { args, common }) => {
            with_margin(&common, |ctx| config::preload(ctx, &args))
        }
        Some(Cmd::Resize {
            model,
            kv_cache_gb,
            yes,
            force,
            common,
        }) => with_margin(&common, |ctx| {
            ops::resize(ctx, &model, kv_cache_gb, yes, force)
        }),
        Some(Cmd::Completeness(c)) => with_margin(&c, ops::completeness),
        Some(Cmd::ReloadKeys(c)) => with_margin(&c, ops::reload_keys),
        Some(Cmd::VoicePin(c)) => with_margin(&c, ops::voice_pin),
        Some(Cmd::Sessions(c)) => with_margin(&c, ops::sessions),
        Some(Cmd::Session { id, common }) => with_margin(&common, |ctx| ops::session(ctx, &id)),
        Some(Cmd::SessionKill { id, yes, common }) => {
            with_margin(&common, |ctx| ops::session_kill(ctx, &id, yes))
        }
        Some(Cmd::SaveDefaults { yes, common }) => {
            with_margin(&common, |ctx| config::save_defaults(ctx, yes))
        }
    };
    ExitCode::from(u8::try_from(code).unwrap_or(1))
}
