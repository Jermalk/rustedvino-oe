//! Local state shared by every subcommand. Port of `Ctx` in `scripts/rv`.

use std::path::PathBuf;

use serde_json::Value;

use crate::ansi::Style;
use crate::api::Api;
use crate::resolve::{Inputs, Resolved, resolve};

/// Flags every subcommand accepts (per-subcommand, as in the Python script:
/// `rv list --json`, not `rv --json list`).
#[derive(clap::Args, Debug, Clone, Default)]
pub struct Common {
    /// machine-readable output where supported
    #[arg(long)]
    pub json: bool,
    /// override the auto-loaded admin key
    #[arg(long)]
    pub admin_key: Option<String>,
    /// server port (default: the config's "port", else 11437)
    #[arg(long)]
    pub port: Option<u16>,
}

/// Everything a subcommand needs.
pub struct Ctx {
    /// Colour decision.
    pub style: Style,
    /// This machine's hostname.
    pub host: String,
    /// Resolved config/keys/port/admin key.
    pub res: Resolved,
    /// Admin API client (bearer = the admin key, when there is one).
    pub api: Api,
    /// Supervisor PID file (`RV_PID_FILE`, default `/tmp/rv.pid`).
    pub pid_file: PathBuf,
    /// `--json` was given.
    pub json: bool,
}

impl Ctx {
    /// Resolves local state and builds the client.
    ///
    /// # Errors
    ///
    /// An unusable `RV_PORT`, or an HTTP client that cannot be built.
    pub fn new(common: &Common, style: Style) -> anyhow::Result<Self> {
        let res = resolve(&Inputs::from_process(common.port, common.admin_key.clone()))?;
        let api = Api::new(res.base_url(), res.admin_key.clone())?;
        let pid_file = std::env::var_os("RV_PID_FILE")
            .map_or_else(|| PathBuf::from("/tmp/rv.pid"), PathBuf::from);
        Ok(Self {
            style,
            host: hostname(),
            res,
            api,
            pid_file,
            json: common.json,
        })
    }

    /// Prints a clear reason and returns `false` when there is no admin key.
    /// Callers keep control of their own exit code.
    #[must_use]
    pub fn require_admin_key(&self) -> bool {
        if self.res.admin_key.is_some() {
            return true;
        }
        if self.server_is_open() {
            self.warn_if_open();
            return true;
        }
        eprintln!(
            "{}",
            self.style.red(&format!(
                "error: no admin key available — set RV_ADMIN_KEY, pass --admin-key, or populate {}",
                self.res.keys_file.display()
            ))
        );
        eprintln!(
            "  = help: `rv start` creates that file when it is missing; a server that was already \
             running reads it only at start, so `rv restart` after creating or filling it"
        );
        false
    }

    /// Prints the OPEN-server warning when the keys file exists and holds no
    /// keys: the server is open, so no bearer is needed. Said on every call that talks to the server,
    /// on purpose - an open admin API is not a state to forget about. Does nothing when a key was
    /// given (flag or environment) or the file holds keys.
    pub fn warn_if_open(&self) {
        if !self.server_is_open() {
            return;
        }
        eprintln!(
            "{}",
            self.style.yellow(&format!(
                "warning: this server is OPEN and NOT SECURE - {} holds no keys, so admin calls \
                 go out with no authentication and anyone who can reach port {} can load, \
                 unload and delete models. Add keys to enable authentication.",
                self.res.keys_file.display(),
                self.res.port
            ))
        );
    }

    /// No key was given and the keys file exists with empty lists: admin calls need no bearer.
    fn server_is_open(&self) -> bool {
        self.res.admin_key.is_none() && self.res.keys_file_is_empty()
    }

    /// Directory names under `models_dir` — a best-effort local scan. `None`
    /// when `models_dir` is unknown or unreadable (not an empty list, so
    /// callers can tell "no models" from "couldn't check").
    #[must_use]
    pub fn list_model_dirs(&self) -> Option<Vec<String>> {
        let dir = self.res.models_dir.as_deref().filter(|d| !d.is_empty())?;
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .ok()?
            .flatten()
            .filter(|e| e.path().is_dir())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| !n.starts_with('.'))
            .collect();
        names.sort();
        Some(names)
    }

    /// Compact one-line summary of the `PATCH`-settable fields worth flagging
    /// at a glance; empty when everything is at its default.
    #[must_use]
    pub fn policy_flags(&self, model_id: &str) -> String {
        let Some(entry) = self.res.raw_models.get(model_id).and_then(Value::as_object) else {
            return String::new();
        };
        let mut flags: Vec<String> = Vec::new();
        if entry.get("pinned").is_some_and(truthy) {
            flags.push("pinned".into());
        }
        if let Some(p) = entry.get("priority").filter(|p| truthy(p)) {
            flags.push(format!("prio:{}", display_number(p)));
        }
        if entry.get("load").and_then(Value::as_str) == Some("on_demand") {
            flags.push("on_demand".into());
        }
        if entry.get("evictable") == Some(&Value::Bool(false)) {
            flags.push("not_evictable".into());
        }
        if let Some(g) = entry.get("eviction_grace_secs").and_then(Value::as_f64) {
            flags.push(format!("grace:{g:.0}s"));
        }
        flags.join(", ")
    }

    /// Asks the SERVER whether it enforces auth, rather than guessing from the
    /// local keys file. Returns `(inference_enforced, admin_enforced)`; either
    /// is `None` when it could not be determined. A deliberately key-less
    /// request to a protected route answers it: 401 means guarded, 2xx means
    /// open.
    #[must_use]
    pub fn probe_auth_enforcement(&self) -> (Option<bool>, Option<bool>) {
        let keyless = self.api.keyless();
        (
            classify_auth(keyless.get("/v1/models").code),
            classify_auth(keyless.get("/v1/admin/models").code),
        )
    }
}

/// What a key-less request's status says about enforcement: 401 means the route
/// is guarded, 2xx means anyone can call it, anything else (5xx, no response)
/// is "don't claim".
#[must_use]
pub fn classify_auth(code: Option<u16>) -> Option<bool> {
    match code {
        Some(401) => Some(true),
        Some(c) if (200..300).contains(&c) => Some(false),
        _ => None,
    }
}

/// Python truthiness for the JSON values a config can hold.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn display_number(v: &Value) -> String {
    v.as_i64().map_or_else(|| v.to_string(), |i| i.to_string())
}

/// Best-effort hostname without extra dependencies or `unsafe`.
fn hostname() -> String {
    for var in ["HOSTNAME", "COMPUTERNAME"] {
        if let Ok(h) = std::env::var(var)
            && !h.is_empty()
        {
            return h;
        }
    }
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .or_else(|_| std::fs::read_to_string("/etc/hostname"))
        .map(|s| s.trim().to_owned())
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

#[cfg(test)]
#[allow(clippy::expect_used)] // fixtures: a failed setup should fail the test loudly
mod tests {
    use super::*;

    fn ctx_with_models(models: &serde_json::Value) -> Ctx {
        let mut res = resolve(&Inputs {
            default_config: Some(PathBuf::from("/nonexistent")),
            ..Inputs::default()
        })
        .expect("resolves");
        res.raw_models = models.as_object().cloned().unwrap_or_default();
        Ctx {
            style: Style::plain(),
            host: "h".into(),
            api: Api::new("http://localhost:1".into(), None).expect("client"),
            res,
            pid_file: PathBuf::from("/tmp/x"),
            json: false,
        }
    }

    #[test]
    fn auth_classification() {
        assert_eq!(classify_auth(Some(401)), Some(true));
        assert_eq!(classify_auth(Some(200)), Some(false));
        assert_eq!(classify_auth(Some(204)), Some(false));
        assert_eq!(classify_auth(Some(503)), None);
        assert_eq!(classify_auth(Some(403)), None);
        assert_eq!(classify_auth(None), None);
    }

    #[test]
    fn policy_flags_summarise_non_default_fields_only() {
        let ctx = ctx_with_models(&serde_json::json!({
            "a": {"pinned": true, "priority": 3, "load": "on_demand",
                  "evictable": false, "eviction_grace_secs": 30.0},
            "b": {"pinned": false, "priority": 0, "load": "eager", "evictable": true},
            "c": {},
        }));
        assert_eq!(
            ctx.policy_flags("a"),
            "pinned, prio:3, on_demand, not_evictable, grace:30s"
        );
        assert_eq!(ctx.policy_flags("b"), "");
        assert_eq!(ctx.policy_flags("c"), "");
        assert_eq!(ctx.policy_flags("missing"), "");
    }

    #[test]
    fn model_dirs_skip_dotdirs_and_files_and_sort() {
        let d = tempfile::tempdir().expect("tempdir");
        for n in ["zeta", "alpha", ".hidden"] {
            std::fs::create_dir(d.path().join(n)).expect("mkdir");
        }
        std::fs::write(d.path().join("file.txt"), "x").expect("write");
        let mut ctx = ctx_with_models(&serde_json::json!({}));
        ctx.res.models_dir = Some(d.path().display().to_string());
        assert_eq!(
            ctx.list_model_dirs(),
            Some(vec!["alpha".into(), "zeta".into()])
        );
        ctx.res.models_dir = Some("/definitely/not/here".into());
        assert_eq!(ctx.list_model_dirs(), None);
        ctx.res.models_dir = None;
        assert_eq!(ctx.list_model_dirs(), None);
    }
}
