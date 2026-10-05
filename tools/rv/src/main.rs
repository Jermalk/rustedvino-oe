//! `rv` binary entry point: the admin CLI.

// Doc comments on clap items become `--help` text, where markdown backticks
// would show literally — so doc_markdown is relaxed for this file only.
#![allow(clippy::doc_markdown)]

use std::process::ExitCode;

use clap::{CommandFactory, Parser, Subcommand};
use rv::ansi::Style;
use rv::commands::config::{self, PatchArgs, PreloadArgs};
use rv::commands::lifecycle::{self, AddArgs, LoadArgs};
use rv::commands::{check, logs, models, ops, status};
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
    /// full server report
    Status(Common),
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
    Unload {
        /// model id
        model: String,
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
        /// the new KV-cache pool target in GB
        #[arg(long, allow_hyphen_values = true)]
        kv_cache_gb: f64,
        /// don't prompt for confirmation
        #[arg(short = 'y', long)]
        yes: bool,
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

fn main() -> ExitCode {
    let cli = Cli::parse();
    let code = match cli.cmd {
        // Bare `rv` prints help instead of erroring, like the Python script.
        None => {
            let _ = Cli::command().print_help();
            println!();
            0
        }
        Some(Cmd::Resolve { port, admin_key }) => print_resolved(port, admin_key),
        Some(Cmd::Status(c)) => with_margin(&c, status::run),
        Some(Cmd::List(c)) => with_margin(&c, models::list),
        Some(Cmd::Ps(c)) => with_margin(&c, models::ps),
        Some(Cmd::Check { model, common }) => with_margin(&common, |ctx| check::run(ctx, &model)),
        Some(Cmd::Logs { n, common }) => with_margin(&common, |ctx| logs::run(ctx, n)),
        Some(Cmd::Add { args, common }) => with_margin(&common, |ctx| lifecycle::add(ctx, &args)),
        Some(Cmd::Load { args, common }) => with_margin(&common, |ctx| lifecycle::load(ctx, &args)),
        Some(Cmd::Unload { model, common }) => {
            with_margin(&common, |ctx| lifecycle::unload(ctx, &model))
        }
        Some(Cmd::Deregister { model, yes, common }) => {
            with_margin(&common, |ctx| lifecycle::deregister(ctx, &model, yes))
        }
        Some(Cmd::ReloadConfig(c)) => with_margin(&c, config::reload_config),
        Some(Cmd::Patch { args, common }) => with_margin(&common, |ctx| config::patch(ctx, &args)),
        Some(Cmd::Preload { args, common }) => {
            with_margin(&common, |ctx| config::preload(ctx, &args))
        }
        Some(Cmd::Resize {
            model,
            kv_cache_gb,
            yes,
            common,
        }) => with_margin(&common, |ctx| ops::resize(ctx, &model, kv_cache_gb, yes)),
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
