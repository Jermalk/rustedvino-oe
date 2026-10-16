//! Local-state resolution: which config file, which keys file, which port,
//! which admin key. Port of `Ctx.__init__` in `scripts/rv`.
//!
//! The Python script is the spec, with two deliberate omissions:
//!
//! * no per-host shortcut (`HOST_PREFIX`) — the public build of the Python
//!   script ships it empty (`scripts/sanitize-rv.py`) and the bundle check
//!   greps binaries for fleet hostnames, so a compiled-in map would leak them;
//! * no strict config validation — the server's `Config::load` rejects unknown
//!   fields and invalid values, and a config the server refuses must not also
//!   break `rv status`. Only `port`, `models_dir` and `keys_file` are read,
//!   leniently.
//!
//! Everything environmental is passed in via [`Inputs`], so tests never touch
//! the process environment.

use std::path::{Path, PathBuf};

use anyhow::{Context, bail};

/// The server's own default port (`src/model_manager/config.rs`), used when
/// neither a flag, `RV_PORT` nor the config names one.
pub const DEFAULT_PORT: u16 = 11437;

/// The per-user config location on Windows, `%LOCALAPPDATA%\RustedVINO\config.json`.
/// A copy of the server's `startup::per_user_config_path` (this crate cannot
/// depend on the server), pinned to it by the server crate's
/// `tests/rv_keys_path_parity.rs`. Takes the `LOCALAPPDATA` value as a parameter
/// so it is testable on any OS.
#[must_use]
pub fn per_user_config_path(local_app_data: Option<&std::ffi::OsStr>) -> Option<PathBuf> {
    let base = local_app_data.filter(|v| !v.is_empty())?;
    Some(Path::new(base).join("RustedVINO").join("config.json"))
}

/// Compiled-in default config location: on Unix the server's `DEFAULT_CONFIG`
/// (`/opt/rustedvino/config.json`); on Windows the per-user
/// `%LOCALAPPDATA%\RustedVINO\config.json`. `None` when `LOCALAPPDATA` is unset.
#[must_use]
pub fn platform_default_config() -> Option<PathBuf> {
    #[cfg(unix)]
    {
        Some(PathBuf::from("/opt/rustedvino/config.json"))
    }
    #[cfg(not(unix))]
    {
        per_user_config_path(std::env::var_os("LOCALAPPDATA").as_deref())
    }
}

/// Everything resolution reads from the environment, made explicit.
#[derive(Debug, Default, Clone)]
pub struct Inputs {
    /// `RUSTEDVINO_CONFIG`.
    pub env_config: Option<String>,
    /// `RV_PORT`.
    pub env_port: Option<String>,
    /// `RV_ADMIN_KEY`.
    pub env_admin_key: Option<String>,
    /// `--port`.
    pub flag_port: Option<u16>,
    /// `--admin-key`.
    pub flag_admin_key: Option<String>,
    /// Directory holding the `rv` executable (release-bundle layout).
    pub exe_dir: Option<PathBuf>,
    /// Fallback config path; normally [`platform_default_config`].
    pub default_config: Option<PathBuf>,
}

impl Inputs {
    /// Reads the real process environment and executable location.
    #[must_use]
    pub fn from_process(flag_port: Option<u16>, flag_admin_key: Option<String>) -> Self {
        Self {
            env_config: non_empty(std::env::var("RUSTEDVINO_CONFIG").ok()),
            env_port: non_empty(std::env::var("RV_PORT").ok()),
            env_admin_key: non_empty(std::env::var("RV_ADMIN_KEY").ok()),
            flag_port,
            flag_admin_key: non_empty(flag_admin_key),
            exe_dir: std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(Path::to_path_buf)),
            default_config: platform_default_config(),
        }
    }
}

/// Drops a UTF-8 byte-order mark: Windows PowerShell 5.1 writes one with
/// `Set-Content -Encoding UTF8`, and `serde_json` rejects it.
fn strip_bom(s: &str) -> &str {
    s.strip_prefix('\u{feff}').unwrap_or(s)
}

fn non_empty(v: Option<String>) -> Option<String> {
    v.filter(|s| !s.is_empty())
}

/// The resolved local state every subcommand shares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    /// The config path chosen (may not exist — a missing config is ordinary).
    pub config_file: PathBuf,
    /// Whether `config_file` existed and parsed as a JSON object.
    pub config_readable: bool,
    /// `models_dir` from the config, if present and a string.
    pub models_dir: Option<String>,
    /// The config's raw `models` object (empty when absent). `rv list`/`ps`
    /// read `pinned`/`priority`/... from it because `GET /v1/admin/models`
    /// does not echo them back.
    pub raw_models: serde_json::Map<String, serde_json::Value>,
    /// The keys file path (`keys_file` in the config, else the sibling).
    pub keys_file: PathBuf,
    /// Whether `keys_file` exists on disk.
    pub keys_present: bool,
    /// Whether `keys_file` was read and parsed as JSON. An unreadable or
    /// malformed file is never treated as an empty (open) one.
    pub keys_parsed: bool,
    /// Number of inference keys in the keys file.
    pub api_key_count: usize,
    /// Number of admin keys in the keys file.
    pub admin_key_count: usize,
    /// Admin key to use: `--admin-key`, else `RV_ADMIN_KEY`, else the first
    /// `admin_api_keys` entry.
    pub admin_key: Option<String>,
    /// Server port.
    pub port: u16,
}

impl Resolved {
    /// The keys file parsed and holds no keys at all. The server treats that as
    /// **open**: its admin and inference routes need no bearer, so anyone who can
    /// reach the port can load, unload and delete models.
    #[must_use]
    pub fn keys_file_is_empty(&self) -> bool {
        self.keys_parsed && self.api_key_count == 0 && self.admin_key_count == 0
    }

    /// Whether admin calls can be made: an admin key is known, or the server is
    /// open (see [`keys_file_is_empty`](Self::keys_file_is_empty)) and needs none.
    #[must_use]
    pub fn admin_usable(&self) -> bool {
        self.admin_key.is_some() || self.keys_file_is_empty()
    }

    /// `http://127.0.0.1:<port>` — the host is always this machine: the verbs
    /// read the PID file, models dir and keys off it, so a remote target would
    /// make half the output describe the wrong box. The numeric address, not
    /// `localhost`: on Windows `localhost` often resolves to `::1` first, and
    /// the server binds IPv4 loopback by default, so another local user could
    /// listen on `[::1]:PORT` and be sent the admin key.
    #[must_use]
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
}

/// Picks the config path. Order, strongest signal first (same as `Ctx` in
/// `scripts/rv`, whose comment explains why the bundle outranks the default):
///
/// 1. `RUSTEDVINO_CONFIG` — an explicit request always wins;
/// 2. `config.json` beside the executable — the release-bundle layout, a
///    deliberate placement;
/// 3. the platform default, if any.
///
/// Falls back to the platform default (or the bundle path) even when nothing
/// exists, so the caller can name what it looked for.
#[must_use]
pub fn pick_config_path(inputs: &Inputs) -> PathBuf {
    if let Some(p) = &inputs.env_config {
        // Absolute: the launcher `cd`s into the bundle before the server reads this variable, so a
        // relative path would name a different file for the server than for rv.
        let p = PathBuf::from(p);
        return std::path::absolute(&p).unwrap_or(p);
    }
    let bundle = inputs.exe_dir.as_ref().map(|d| d.join("config.json"));
    if let Some(b) = &bundle
        && b.is_file()
    {
        return b.clone();
    }
    if let Some(d) = &inputs.default_config {
        return d.clone();
    }
    bundle.unwrap_or_else(|| PathBuf::from("config.json"))
}

/// Where a config file's companion keys file lives.
///
/// Copy of `resolve_keys_file_path` in the server
/// (`src/model_manager/config.rs`), which this crate cannot depend on. The
/// server crate's `tests/rv_keys_path_parity.rs` pins the two together.
///
/// `explicit` is the config's `keys_file` verbatim: relative paths resolve
/// against the config's parent directory (never the cwd), absolute paths are
/// used as-is. `None` derives the sibling: strip a trailing `.config.json` (or
/// bare `.json`) from the file name and append `.keys.json`.
#[must_use]
pub fn resolve_keys_file_path(config_path: &Path, explicit: Option<&Path>) -> PathBuf {
    if let Some(explicit) = explicit {
        return if explicit.is_absolute() {
            explicit.to_path_buf()
        } else {
            config_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(explicit)
        };
    }
    let dir = config_path.parent().unwrap_or_else(|| Path::new("."));
    let name = config_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("config.json");
    let stem = name
        .strip_suffix(".config.json")
        .or_else(|| name.strip_suffix(".json"))
        .unwrap_or(name);
    dir.join(format!("{stem}.keys.json"))
}

/// The directory `rv` lives in, when a server launcher sits next to it (a release bundle). The
/// launcher `cd`s there before it starts the server, so the server's relative paths mean "relative
/// to this directory", not to wherever `rv` happens to be run from.
fn bundle_dir() -> Option<PathBuf> {
    let dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    // A release bundle has the launcher AND its `lib/`; a cargo `target/debug` dir has a binary
    // called `rustedvino` too, but no `lib/`.
    (dir.join("rustedvino").is_file() && dir.join("lib").is_dir()).then_some(dir)
}

/// A relative `models_dir` anchored to the bundle directory (see [`bundle_dir`]); absolute paths and
/// non-bundle installs are returned unchanged.
fn anchor_to_bundle(models_dir: &str, bundle: Option<&Path>) -> String {
    let p = Path::new(models_dir);
    match bundle {
        Some(b) if p.is_relative() && !models_dir.is_empty() => b.join(p).display().to_string(),
        _ => models_dir.to_owned(),
    }
}

/// Resolves config, keys, admin key and port.
///
/// # Errors
///
/// Fails only for an unusable `RV_PORT` (not a port number). A missing,
/// unreadable or malformed config or keys file is not an error — a bundle
/// before `cp config.example.json config.json` is an ordinary case — the
/// corresponding fields just stay empty.
pub fn resolve(inputs: &Inputs) -> anyhow::Result<Resolved> {
    let config_file = pick_config_path(inputs);

    let cfg: Option<serde_json::Value> = std::fs::read_to_string(&config_file)
        .ok()
        .and_then(|t| serde_json::from_str(strip_bom(&t)).ok())
        .filter(serde_json::Value::is_object);

    let models_dir = cfg
        .as_ref()
        .and_then(|c| c.get("models_dir"))
        .and_then(|v| v.as_str())
        .map(|d| anchor_to_bundle(d, bundle_dir().as_deref()));
    let raw_models = cfg
        .as_ref()
        .and_then(|c| c.get("models"))
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let explicit_keys = cfg
        .as_ref()
        .and_then(|c| c.get("keys_file"))
        .and_then(|v| v.as_str())
        .map(PathBuf::from);
    let keys_file = resolve_keys_file_path(&config_file, explicit_keys.as_deref());

    let keys_present = keys_file.is_file();
    let mut api_key_count = 0;
    let mut admin_key_count = 0;
    let mut first_admin_key: Option<String> = None;
    let mut keys_parsed = false;
    if keys_present
        && let Some(kf) = std::fs::read_to_string(&keys_file)
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(strip_bom(&t)).ok())
    {
        keys_parsed = true;
        api_key_count = kf
            .get("api_keys")
            .and_then(|v| v.as_array())
            .map_or(0, Vec::len);
        let admin = kf.get("admin_api_keys").and_then(|v| v.as_array());
        admin_key_count = admin.map_or(0, Vec::len);
        first_admin_key = non_empty(
            admin
                .and_then(|a| a.first())
                .and_then(|v| v.as_str())
                .map(str::to_owned),
        );
    }
    let admin_key = inputs
        .flag_admin_key
        .clone()
        .or_else(|| inputs.env_admin_key.clone())
        .or(first_admin_key);

    let port = resolve_port(inputs, cfg.as_ref())?;

    Ok(Resolved {
        config_file,
        config_readable: cfg.is_some(),
        models_dir,
        raw_models,
        keys_file,
        keys_present,
        keys_parsed,
        api_key_count,
        admin_key_count,
        admin_key,
        port,
    })
}

/// Port: `--port`, then `RV_PORT`, then the config's `port`, then
/// [`DEFAULT_PORT`]. Reading it from the config is the point: the tool and the
/// server must not disagree about where the server is.
fn resolve_port(inputs: &Inputs, cfg: Option<&serde_json::Value>) -> anyhow::Result<u16> {
    if let Some(p) = inputs.flag_port {
        return Ok(p);
    }
    if let Some(raw) = &inputs.env_port {
        let Ok(p) = raw.trim().parse::<u16>() else {
            bail!("RV_PORT is not a port number: {raw:?}");
        };
        return Ok(p);
    }
    let cfg_port = cfg
        .and_then(|c| c.get("port"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|p| u16::try_from(p).ok());
    Ok(cfg_port.unwrap_or(DEFAULT_PORT))
}

/// Reads a file to a string with the path in the error — used by later
/// phases; kept here so the context message stays uniform.
///
/// # Errors
///
/// Propagates the I/O error with the path attached.
pub fn read_text(path: &Path) -> anyhow::Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))
}

#[cfg(test)]
#[allow(clippy::expect_used)] // fixtures: a failed setup should fail the test loudly
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).expect("write fixture");
        p
    }

    fn base(dir: &Path) -> Inputs {
        Inputs {
            exe_dir: Some(dir.to_path_buf()),
            default_config: Some(dir.join("nonexistent-default.json")),
            ..Inputs::default()
        }
    }

    #[test]
    fn per_user_config_path_is_under_localappdata() {
        let p = per_user_config_path(Some(std::ffi::OsStr::new("base"))).expect("path");
        assert!(
            p.ends_with(Path::new("RustedVINO").join("config.json")),
            "{p:?}"
        );
        assert_eq!(per_user_config_path(None), None);
        assert_eq!(per_user_config_path(Some(std::ffi::OsStr::new(""))), None);
    }

    #[test]
    fn keys_path_matches_the_server_doc_examples() {
        assert_eq!(
            resolve_keys_file_path(Path::new("scripts/myserver.config.json"), None),
            PathBuf::from("scripts/myserver.keys.json")
        );
        assert_eq!(
            resolve_keys_file_path(Path::new("/opt/rustedvino/config.json"), None),
            PathBuf::from("/opt/rustedvino/config.keys.json")
        );
        assert_eq!(
            resolve_keys_file_path(Path::new("/etc/rv/odd"), None),
            PathBuf::from("/etc/rv/odd.keys.json")
        );
    }

    #[test]
    fn explicit_keys_path_is_relative_to_the_config_not_the_cwd() {
        assert_eq!(
            resolve_keys_file_path(Path::new("/a/b/c.json"), Some(Path::new("k.json"))),
            PathBuf::from("/a/b/k.json")
        );
        assert_eq!(
            resolve_keys_file_path(Path::new("/a/b/c.json"), Some(Path::new("/z/k.json"))),
            PathBuf::from("/z/k.json")
        );
    }

    #[test]
    fn env_config_beats_the_bundle_config() {
        let d = tempfile::tempdir().expect("tempdir");
        write(d.path(), "config.json", "{}");
        let other = write(d.path(), "other.json", "{}");
        let mut i = base(d.path());
        i.env_config = Some(other.display().to_string());
        assert_eq!(pick_config_path(&i), other);
    }

    #[test]
    fn a_relative_env_config_is_made_absolute() {
        let d = tempfile::tempdir().expect("tempdir");
        let mut i = base(d.path());
        i.env_config = Some("sub/x.json".to_owned());
        let got = pick_config_path(&i);
        assert!(got.is_absolute(), "{}", got.display());
        assert!(got.ends_with("sub/x.json"));
    }

    #[test]
    fn bundle_config_beats_the_platform_default() {
        let d = tempfile::tempdir().expect("tempdir");
        let bundle = write(d.path(), "config.json", "{}");
        let default = write(d.path(), "default.json", "{}");
        let mut i = base(d.path());
        i.default_config = Some(default);
        assert_eq!(pick_config_path(&i), bundle);
    }

    #[test]
    fn falls_back_to_the_default_when_no_bundle_config() {
        let d = tempfile::tempdir().expect("tempdir");
        let i = base(d.path());
        assert_eq!(
            pick_config_path(&i),
            d.path().join("nonexistent-default.json")
        );
    }

    #[test]
    fn missing_config_is_not_an_error() {
        let d = tempfile::tempdir().expect("tempdir");
        let r = resolve(&base(d.path())).expect("resolves");
        assert!(!r.config_readable);
        assert_eq!(r.port, DEFAULT_PORT);
        assert!(r.admin_key.is_none());
        assert!(!r.keys_present);
    }

    #[test]
    fn a_relative_models_dir_is_anchored_to_the_bundle_only() {
        let b = Path::new("/bundle");
        assert_eq!(anchor_to_bundle("./models", Some(b)), "/bundle/./models");
        assert_eq!(anchor_to_bundle("/abs/models", Some(b)), "/abs/models");
        assert_eq!(anchor_to_bundle("./models", None), "./models");
        assert_eq!(anchor_to_bundle("", Some(b)), "");
    }

    #[test]
    fn config_port_models_dir_and_keys_are_read_leniently() {
        let d = tempfile::tempdir().expect("tempdir");
        // Unknown field + a value the server would reject: rv must not care.
        write(
            d.path(),
            "config.json",
            r#"{"port": 12345, "models_dir": "/m", "totally_unknown": 1, "total_vram_gb": "oops"}"#,
        );
        write(
            d.path(),
            "config.keys.json",
            r#"{"api_keys": ["a", "b"], "admin_api_keys": ["adm1", "adm2"]}"#,
        );
        let r = resolve(&base(d.path())).expect("resolves");
        assert!(r.config_readable);
        assert_eq!(r.port, 12345);
        assert_eq!(r.models_dir.as_deref(), Some("/m"));
        assert!(r.raw_models.is_empty());
        assert_eq!((r.api_key_count, r.admin_key_count), (2, 2));
        assert_eq!(r.admin_key.as_deref(), Some("adm1"));
        assert_eq!(r.base_url(), "http://127.0.0.1:12345");
    }

    #[test]
    fn empty_first_admin_key_counts_as_no_key() {
        let d = tempfile::tempdir().expect("tempdir");
        write(d.path(), "config.json", "{}");
        write(d.path(), "config.keys.json", r#"{"admin_api_keys": [""]}"#);
        let r = resolve(&base(d.path())).expect("resolves");
        assert!(r.admin_key.is_none());
    }

    #[test]
    fn a_utf8_bom_does_not_make_a_file_unreadable() {
        let d = tempfile::tempdir().expect("tempdir");
        write(d.path(), "config.json", "\u{feff}{\"port\": 4242}");
        write(
            d.path(),
            "config.keys.json",
            "\u{feff}{\"admin_api_keys\": [\"k\"]}",
        );
        let r = resolve(&base(d.path())).expect("resolves");
        assert!(r.config_readable);
        assert_eq!((r.port, r.admin_key.as_deref()), (4242, Some("k")));
    }

    #[test]
    fn malformed_config_and_keys_are_tolerated() {
        let d = tempfile::tempdir().expect("tempdir");
        write(d.path(), "config.json", "{ not json");
        write(d.path(), "config.keys.json", "[1,2");
        let r = resolve(&base(d.path())).expect("resolves");
        assert!(!r.config_readable);
        assert!(r.keys_present);
        assert_eq!(r.admin_key_count, 0);
    }

    #[test]
    fn keys_file_field_in_config_is_honoured() {
        let d = tempfile::tempdir().expect("tempdir");
        write(
            d.path(),
            "config.json",
            r#"{"keys_file": "elsewhere.json"}"#,
        );
        write(d.path(), "elsewhere.json", r#"{"admin_api_keys": ["k"]}"#);
        let r = resolve(&base(d.path())).expect("resolves");
        assert_eq!(r.keys_file, d.path().join("elsewhere.json"));
        assert_eq!(r.admin_key.as_deref(), Some("k"));
    }

    #[test]
    fn admin_key_precedence_flag_then_env_then_file() {
        let d = tempfile::tempdir().expect("tempdir");
        write(d.path(), "config.json", "{}");
        write(
            d.path(),
            "config.keys.json",
            r#"{"admin_api_keys": ["file"]}"#,
        );
        let mut i = base(d.path());
        assert_eq!(resolve(&i).expect("r").admin_key.as_deref(), Some("file"));
        i.env_admin_key = Some("env".into());
        assert_eq!(resolve(&i).expect("r").admin_key.as_deref(), Some("env"));
        i.flag_admin_key = Some("flag".into());
        assert_eq!(resolve(&i).expect("r").admin_key.as_deref(), Some("flag"));
    }

    #[test]
    fn port_precedence_flag_then_env_then_config_then_default() {
        let d = tempfile::tempdir().expect("tempdir");
        write(d.path(), "config.json", r#"{"port": 2000}"#);
        let mut i = base(d.path());
        assert_eq!(resolve(&i).expect("r").port, 2000);
        i.env_port = Some("3000".into());
        assert_eq!(resolve(&i).expect("r").port, 3000);
        i.flag_port = Some(4000);
        assert_eq!(resolve(&i).expect("r").port, 4000);
    }

    #[test]
    fn bad_rv_port_is_an_error_and_out_of_range_config_port_is_ignored() {
        let d = tempfile::tempdir().expect("tempdir");
        write(d.path(), "config.json", r#"{"port": 99999}"#);
        let mut i = base(d.path());
        assert_eq!(resolve(&i).expect("r").port, DEFAULT_PORT);
        i.env_port = Some("abc".into());
        assert!(resolve(&i).is_err());
    }
}
