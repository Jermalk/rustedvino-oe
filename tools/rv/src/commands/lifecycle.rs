//! `rv add`, `load`, `unload`, `deregister`: verbs that change what is loaded
//! or registered.

use std::path::Path;

use serde_json::{Map, Value, json};

use crate::api::{LOAD_TIMEOUT, Payload, READ_TIMEOUT, encode_segment};
use crate::commands::prompt::confirm;
use crate::commands::{error_code, print_busy_hint, print_error};
use crate::ctx::Ctx;

/// The estimate is the plain on-disk size (D-03, 0.8.4). No single ratio fits the models measured so
/// far: ministral-3b INT8 works at about 0.96 of its size, Intel's mixed INT4/INT8 qwen3-14b measured
/// 1.11 (9.7 GB `.bin`, 10.8 GB resident), qwen3.6-35b-a3b INT4 measured 1.005. A 1.12 margin
/// (0.8.3, F7) fixed the 14B but made `rv add` refuse the 35B, which runs. Jerzy chose the
/// optimistic side: the server still adds its own minimum KV and safety margin, and warns when a
/// model measures well above its charge. A measured-footprint run is planned.
const VRAM_ESTIMATE_MARGIN: f64 = 1.0;

/// What the size estimate leaves out of a model directory (SF-2), matched case-insensitively on
/// the entry name at any depth, each with the label `rv add` prints:
/// - backups left beside the weights by hand edits (0.8.4 D-08), files or directories;
/// - partial or temporary downloads and editor leftovers;
/// - another server's compile cache (`OpenArc`'s `model_cache/`, about 3x the weights of an
///   embedding model measured on one box), a `git lfs` clone's second copy of the weights
///   (`.git/`) and Python/Jupyter caches (directories only).
const SKIPPED_SUFFIXES: [(&str, &str); 7] = [
    (".bak", "backups"),
    (".orig", "backups"),
    (".old", "backups"),
    (".part", "partial downloads"),
    (".incomplete", "partial downloads"),
    (".tmp", "partial downloads"),
    ("~", "editor backups"),
];
const SKIPPED_DIRS: [(&str, &str); 4] = [
    ("model_cache", "model_cache/"),
    (".git", ".git/"),
    ("__pycache__", "__pycache__/"),
    (".ipynb_checkpoints", ".ipynb_checkpoints/"),
];

/// The label of an entry the estimate does not count, or `None` when it counts.
fn skipped_by_estimate(name: &std::ffi::OsStr, is_dir: bool) -> Option<&'static str> {
    let name = name.to_string_lossy().to_ascii_lowercase();
    if is_dir && let Some((_, label)) = SKIPPED_DIRS.iter().find(|(d, _)| name == *d) {
        return Some(label);
    }
    SKIPPED_SUFFIXES
        .iter()
        .find(|(suffix, _)| name.ends_with(suffix))
        .map(|(_, label)| *label)
}

/// A model directory's size as `rv add` estimates it.
#[derive(Debug, Default)]
pub struct ModelDirSize {
    /// `(raw_disk_gb, estimated_vram_gb)` of the counted files; `None` when nothing counts
    /// (missing, empty, or only skipped entries).
    pub estimate: Option<(f64, f64)>,
    /// GB of the entries left out ([`skipped_by_estimate`]).
    pub skipped_gb: f64,
    /// Their labels, deduplicated, in first-seen order.
    pub skipped_kinds: Vec<&'static str>,
}

/// The size estimate from the files under `model_dir`. Sums the whole tree, not just
/// `openvino_model.bin`: some kinds (image-gen's `unet`/`vae`/`text_encoder`, a VLM's vision
/// tower) split weights across several files with no single dominant one, and the small
/// metadata files are noise at the KB scale against GB-scale weights. Backups, partial
/// downloads and foreign caches ([`skipped_by_estimate`]) are left out and reported.
#[must_use]
pub fn estimate_vram_gb(model_dir: &Path) -> ModelDirSize {
    struct Walk {
        total: u64,
        skipped: u64,
        found: bool,
        kinds: Vec<&'static str>,
    }
    fn walk(dir: &Path, w: &mut Walk, skipped_as: Option<&'static str>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let Ok(ft) = e.file_type() else { continue };
            let label = skipped_as.or_else(|| skipped_by_estimate(&e.file_name(), ft.is_dir()));
            if ft.is_dir() {
                walk(&e.path(), w, label);
            } else if let Ok(md) = std::fs::metadata(e.path()) {
                // `metadata` follows file symlinks (like os.path.getsize);
                // directory symlinks are not descended (like os.walk).
                if md.is_file() {
                    if let Some(label) = label {
                        w.skipped += md.len();
                        if !w.kinds.contains(&label) {
                            w.kinds.push(label);
                        }
                    } else {
                        w.total += md.len();
                        w.found = true;
                    }
                }
            }
        }
    }
    let mut w = Walk {
        total: 0,
        skipped: 0,
        found: false,
        kinds: Vec::new(),
    };
    walk(model_dir, &mut w, None);
    #[allow(clippy::cast_precision_loss)]
    let (raw_gb, skipped_gb) = (w.total as f64 / 1e9, w.skipped as f64 / 1e9);
    // `{:.1}` rounds the exact binary value (half-even on exact ties), like
    // Python's `round(x, 1)`; rounding `x * 10` first would round twice.
    let estimate = w
        .found
        .then(|| format!("{:.1}", raw_gb * VRAM_ESTIMATE_MARGIN).parse().ok())
        .flatten()
        .map(|est| (raw_gb, est));
    ModelDirSize {
        estimate,
        skipped_gb,
        skipped_kinds: w.kinds,
    }
}

/// Ids of the models currently `ready`, read quietly (empty when the list is
/// unavailable). Used to tell the user what a load evicted (0.8.3, F2).
fn ready_ids(ctx: &Ctx) -> Vec<String> {
    let r = ctx.api.get("/v1/admin/models");
    r.json()
        .filter(|_| r.code == Some(200))
        .and_then(|b| b.get("models"))
        .and_then(Value::as_array)
        .map(|ms| {
            ms.iter()
                .filter(|m| m.get("state").and_then(Value::as_str) == Some("ready"))
                .filter_map(|m| m.get("id").and_then(Value::as_str).map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

/// Prints which models were resident before a load and are gone after it.
fn report_evicted(ctx: &Ctx, before: &[String]) {
    let after = ready_ids(ctx);
    let gone: Vec<&str> = before
        .iter()
        .filter(|id| !after.contains(id))
        .map(String::as_str)
        .collect();
    if !gone.is_empty() {
        println!(
            "{}",
            ctx.style
                .dim(&format!("  evicted to make room: {}", gone.join(", ")))
        );
    }
}

/// Success text: the server's own message when it sent a string, else a default.
fn message(body: Option<&Payload>, default: &str) -> String {
    body.and_then(Payload::text).unwrap_or(default).to_owned()
}

/// `true` when an add answered 200 with `"persisted": false` (the model loaded but its
/// config.json entry was not written).
fn not_persisted(body: Option<&Payload>) -> bool {
    body.and_then(Payload::json)
        .and_then(|v| v.get("persisted"))
        .and_then(serde_json::Value::as_bool)
        == Some(false)
}

/// Flags of `rv add`. Doc comments here are `--help` text, so doc_markdown is
/// relaxed.
#[allow(clippy::doc_markdown)]
#[derive(clap::Args, Debug, Clone)]
pub struct AddArgs {
    /// must match a directory name under models_dir (see `rv list`)
    pub model: String,
    /// VRAM estimate in GB (0.0 to skip VRAM gating). Omit to auto-detect from
    /// the model directory's on-disk size
    #[arg(long)]
    pub vram_gb: Option<f64>,
    /// text_gen|vision|embedding|stt|tts|image_gen|reranking (default: auto-detect)
    #[arg(long)]
    pub kind: Option<String>,
    /// e.g. GPU.1, CPU, NPU (default: global config device)
    #[arg(long)]
    pub device: Option<String>,
    /// KV cache size in GB
    #[arg(long, allow_hyphen_values = true)]
    pub kv_cache_gb: Option<f64>,
    /// concurrent stream cap
    #[arg(long)]
    pub max_concurrent_streams: Option<u32>,
    /// never evict this model once loaded
    #[arg(long)]
    pub pinned: bool,
    /// force prefix caching ON for this model (it also follows the server's `enable_prefix_caching`
    /// setting; hybrid Qwen3.5/3.6/3.8 models on a GPU run with it on)
    #[arg(long)]
    pub enable_prefix_caching: bool,
    /// don't prompt for confirmation
    #[arg(short = 'y', long)]
    pub yes: bool,
}

/// The VRAM figure for `rv add`: `--vram-gb`, else an estimate from the model
/// directory's size. `Err(code)` after printing why neither is available.
fn vram_for_add(ctx: &Ctx, a: &AddArgs) -> Result<f64, i32> {
    if let Some(v) = a.vram_gb {
        return Ok(v);
    }
    let st = ctx.style;
    let Some(dir) = ctx.res.models_dir.as_deref().filter(|d| !d.is_empty()) else {
        eprintln!(
            "{}",
            st.red(&format!(
                "error: --vram-gb not given and models_dir is unknown (couldn't read it from {}) \
                 — pass --vram-gb explicitly",
                ctx.res.config_file.display()
            ))
        );
        return Err(1);
    };
    let model_dir = Path::new(dir).join(&a.model);
    let size = estimate_vram_gb(&model_dir);
    let skipped_note = (!size.skipped_kinds.is_empty()).then(|| {
        format!(
            "{:.1} GB of {}",
            size.skipped_gb,
            size.skipped_kinds.join(", ")
        )
    });
    let Some((_, est)) = size.estimate else {
        let why = skipped_note.as_ref().map_or_else(
            || "is missing or empty. Check the model_id against `rv list`".to_owned(),
            |note| format!("holds no weights the estimate counts (only {note})"),
        );
        eprintln!(
            "{}",
            st.red(&format!(
                "error: --vram-gb not given and couldn't estimate it — '{}' {why}, or pass \
                 --vram-gb explicitly",
                model_dir.display()
            ))
        );
        return Err(1);
    };
    // Printed unconditionally — NOT folded into the confirm prompt, which -y
    // skips printing entirely. An auto-detected number that only shows up when
    // someone happens to read the prompt is a coin flip on whether they
    // noticed.
    println!(
        "{}",
        st.dim(&format!(
            "auto-detected vram_gb: {est:.1} GB (the size on disk; a mixed-precision export can \
             measure up to about 10 % more; override with --vram-gb)"
        ))
    );
    if let Some(note) = skipped_note {
        println!("{}", st.dim(&format!("  not counted: {note}")));
    }
    Ok(est)
}

/// The `POST /v1/admin/models/add` body. Pure, so it is unit-testable.
#[must_use]
pub fn add_body(a: &AddArgs, vram_gb: f64) -> Value {
    let mut body = Map::new();
    body.insert("model_id".into(), json!(a.model));
    body.insert("vram_gb".into(), json!(vram_gb));
    if let Some(k) = &a.kind {
        body.insert("kind".into(), json!(k));
    }
    if let Some(d) = &a.device {
        body.insert("device".into(), json!(d));
    }
    if let Some(k) = a.kv_cache_gb {
        body.insert("kv_cache_gb".into(), json!(k));
    }
    if let Some(m) = a.max_concurrent_streams {
        body.insert("max_concurrent_streams".into(), json!(m));
    }
    if a.pinned {
        body.insert("pinned".into(), json!(true));
    }
    if a.enable_prefix_caching {
        body.insert("enable_prefix_caching".into(), json!(true));
    }
    Value::Object(body)
}

/// `--kv-cache-gb` must be a finite positive number. Checked before anything is sent: JSON has no
/// NaN, so a `NaN` would travel as `null` and the server would read it as "not given".
fn check_kv_cache_gb(v: Option<f64>) -> Result<(), String> {
    match v {
        Some(k) if !k.is_finite() || k <= 0.0 => Err(format!(
            "error: --kv-cache-gb must be a finite positive number, got {k}"
        )),
        _ => Ok(()),
    }
}

/// Runs `rv add`: register an on-disk model and load it.
#[must_use]
pub fn add(ctx: &Ctx, a: &AddArgs) -> i32 {
    if !ctx.require_admin_key() {
        return 1;
    }
    let st = ctx.style;
    if let Err(msg) = check_kv_cache_gb(a.kv_cache_gb) {
        eprintln!("{}", st.red(&msg));
        return 1;
    }
    let vram_gb = match vram_for_add(ctx, a) {
        Ok(v) => v,
        Err(code) => return code,
    };

    // `add_model` funnels through the same evicting load path `rv load` uses —
    // it can evict serving models to make room. Unlike `load` there is no
    // `/check` dry-run (the model is not in the registry yet), so it needs the
    // confirmation gate `deregister` uses instead.
    if !confirm(
        st,
        &format!(
            "register '{}' as {vram_gb:.1} GB (persists to config.json) and load it now — may \
             evict other resident models to make room, with no dry-run preview available for a \
             not-yet-registered model. Proceed?",
            a.model
        ),
        a.yes,
    ) {
        println!("aborted");
        return 2;
    }

    println!(
        "{}",
        st.dim(&format!(
            "registering + loading '{}' — can take 10-60s...",
            a.model
        ))
    );
    let before = ready_ids(ctx);
    let r = ctx.api.post(
        "/v1/admin/models/add",
        Some(&add_body(a, vram_gb)),
        LOAD_TIMEOUT,
    );
    match r.code {
        // Loaded and serving, but its config.json entry was not written: say it loudly — the
        // model is gone after a restart.
        Some(200) if not_persisted(r.body.as_ref()) => {
            let msg = r
                .body
                .as_ref()
                .and_then(|b| b.json()?.get("message")?.as_str().map(str::to_owned))
                .unwrap_or_else(|| "loaded, but NOT persisted to config.json".to_owned());
            eprintln!("{} {}", st.warn(), st.yellow(&msg));
            report_evicted(ctx, &before);
            0
        }
        Some(200) => {
            println!(
                "{} {}",
                st.ok(),
                message(r.body.as_ref(), "added and loaded")
            );
            report_evicted(ctx, &before);
            0
        }
        // 409 means "already registered" — unless the server says it is busy, or that
        // config.json is already invalid: a different cause with a different remedy.
        Some(409) if error_code(&r).is_none() => {
            eprintln!(
                "{}",
                st.red(&format!(
                    "'{0}' is already registered — use `rv load {0}` instead",
                    a.model
                ))
            );
            1
        }
        _ => {
            print_error(ctx, &r);
            1
        }
    }
}

/// Flags of `rv load`.
#[allow(clippy::doc_markdown)]
#[derive(clap::Args, Debug, Clone)]
pub struct LoadArgs {
    /// model id
    pub model: String,
    /// KV cache size in GB
    #[arg(long, allow_hyphen_values = true)]
    pub kv_cache_gb: Option<f64>,
    /// concurrent stream cap
    #[arg(long)]
    pub max_concurrent_streams: Option<u32>,
    /// bypass the eviction-grace window AND evict a model that has a request running (it is
    /// drained for up to 15 s, then aborted)
    #[arg(long)]
    pub force: bool,
    /// skip the pre-load admission check
    #[arg(long)]
    pub no_check: bool,
    /// don't prompt before an evicting load
    #[arg(short = 'y', long)]
    pub yes: bool,
}

/// Path of the dry-run admission check. `--force` is passed on so the preview
/// is the load that will actually run (grace window and busy rule bypassed); without
/// it the pre-check refused loads that `--force` was meant to allow.
fn check_path(encoded_id: &str, force: bool) -> String {
    let q = if force { "?force=true" } else { "" };
    format!("/v1/admin/models/{encoded_id}/check{q}")
}

/// The confirmation-prompt suffix naming evictions that will abort a running
/// request (only a FORCED load evicts a busy model: the server drains it for 15 s,
/// then aborts what is left). Empty when `busy_victims` is absent (an older
/// server) or empty.
fn busy_note(busy_victims: &Value) -> String {
    let busy: Vec<&str> = busy_victims
        .as_array()
        .map(|v| v.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if busy.is_empty() {
        String::new()
    } else {
        format!(
            " ({} busy: its running request is drained for 15 s, then aborted)",
            busy.join(", ")
        )
    }
}

/// The "does not fit" verdict of `rv load`: either nothing more can be evicted, or the room is
/// held by models that were left alone only because a request is running on them (D3).
fn does_not_fit_message(st: crate::ansi::Style, model: &str, check: &Value) -> String {
    let needed = check["needed_gb"].as_f64().unwrap_or(0.0);
    let free = check["free_gb"].as_f64().unwrap_or(0.0);
    let busy: Vec<&str> = check["blocked_by_busy"]
        .as_array()
        .map(|v| v.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    // F1: the device-headroom check (vision models on a discrete GPU) can be what refuses even
    // though the booked figures would fit; older servers omit the key.
    if let Some(why) = check["headroom_refusal"].as_str() {
        return format!(
            "{} '{model}' does not fit — device headroom: {why}; nothing was evicted",
            st.bad()
        );
    }
    if busy.is_empty() {
        format!(
            "{} '{model}' does not fit — needs {needed:.1} GB, {free:.1} GB free, and nothing \
             more can be evicted (pinned, non-evictable, in a realtime session or inside the \
             eviction-grace window; `--force` bypasses the grace window and the busy rule, \
             never the others)",
            st.bad()
        )
    } else {
        format!(
            "{} '{model}' needs {needed:.1} GB and only {free:.1} GB is free; the room is held \
             by {} which has a request running, so nothing was evicted — retry when it \
             finishes, or `rv load {model} --force` to drain it (15 s) and abort what is left",
            st.bad(),
            busy.join(", ")
        )
    }
}

/// F1/A2: a vision model not loaded on this server yet may need more evictions than the check
/// could see; say so before asking to confirm them. Older servers omit the key.
fn headroom_not_previewed_note(st: crate::ansi::Style, check: &Value) {
    if let Some(why) = check["headroom_not_previewed"].as_str() {
        println!(
            "{}",
            st.yellow(&format!("note: device headroom not previewed: {why}"))
        );
    }
}

/// Runs `rv load`: an admission check first (unless `--no-check`), a
/// confirmation when the load would evict, then the load.
#[must_use]
pub fn load(ctx: &Ctx, a: &LoadArgs) -> i32 {
    if !ctx.require_admin_key() {
        return 1;
    }
    let st = ctx.style;
    if let Err(msg) = check_kv_cache_gb(a.kv_cache_gb) {
        eprintln!("{}", st.red(&msg));
        return 1;
    }
    let seg = encode_segment(&a.model);
    if !a.no_check {
        let c = ctx.api.post(&check_path(&seg, a.force), None, READ_TIMEOUT);
        if c.code == Some(404) {
            // Not in the registry at all — distinct from "registered but does
            // not fit". Say so now, before hitting the same 404 on the real
            // /load call (two errors for one cause, in a misleading order).
            eprintln!(
                "{bad} '{m}' is not registered — `rv list` shows on-disk-but-unregistered models; \
                 register it first with `rv add {m} --vram-gb ...`",
                bad = st.bad(),
                m = a.model
            );
            return 1;
        }
        if let Some(check) = c
            .json()
            .filter(|_| c.code == Some(200))
            .filter(|v| v.is_object())
        {
            if check["already_loaded"].as_bool().unwrap_or(false) {
                println!("{} '{}' is already loaded", st.ok(), a.model);
                return 0;
            }
            if !check["fits"].as_bool().unwrap_or(false) {
                println!("{}", does_not_fit_message(st, &a.model, check));
                return 1;
            }
            let evict: Vec<&str> = check["would_evict"]
                .as_array()
                .map(|v| v.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            let busy_note = busy_note(&check["busy_victims"]);
            headroom_not_previewed_note(st, check);
            if !evict.is_empty()
                && !confirm(
                    st,
                    &format!(
                        "loading '{}' will evict: {}{busy_note}. Proceed?",
                        a.model,
                        evict.join(", ")
                    ),
                    a.yes,
                )
            {
                println!("aborted");
                return 2;
            }
        } else if c.code != Some(200) {
            let why = c
                .err
                .clone()
                .unwrap_or_else(|| crate::commands::code_text(c.code));
            eprintln!(
                "{}",
                st.yellow(&format!(
                    "warning: pre-load check failed ({why}) — attempting the load anyway"
                ))
            );
        }
    }

    let mut body = Map::new();
    if let Some(k) = a.kv_cache_gb {
        body.insert("kv_cache_gb".into(), json!(k));
    }
    if let Some(m) = a.max_concurrent_streams {
        body.insert("max_concurrent_streams".into(), json!(m));
    }
    if a.force {
        body.insert("force".into(), json!(true));
    }

    println!(
        "{}",
        st.dim(&format!(
            "loading '{}' — can take 10-60s (JIT compile on first load)...",
            a.model
        ))
    );
    let body = (!body.is_empty()).then(|| Value::Object(body));
    let before = ready_ids(ctx);
    let r = ctx.api.post(
        &format!("/v1/admin/models/{seg}/load"),
        body.as_ref(),
        LOAD_TIMEOUT,
    );
    if r.code == Some(200) {
        println!("{} {}", st.ok(), message(r.body.as_ref(), "loaded"));
        report_evicted(ctx, &before);
        0
    } else {
        print_error(ctx, &r);
        print_busy_hint(ctx, &r, &format!("load {}", a.model));
        1
    }
}

/// Runs `rv unload`: evict a model from VRAM (it stays registered).
#[must_use]
pub fn unload(ctx: &Ctx, model: &str, force: bool) -> i32 {
    if !ctx.require_admin_key() {
        return 1;
    }
    println!("{}", ctx.style.dim(&format!("unloading '{model}'...")));
    let r = ctx.api.delete(&eviction_path(
        &format!("/v1/admin/models/{}", encode_segment(model)),
        force,
    ));
    if r.code == Some(200) {
        println!("{} {}", ctx.style.ok(), message(r.body.as_ref(), "evicted"));
        0
    } else {
        print_error(ctx, &r);
        print_busy_hint(ctx, &r, &format!("unload {model}"));
        1
    }
}

/// `path` with `?force=true` when forcing (D3: a model with a request running is only evicted
/// then).
fn eviction_path(path: &str, force: bool) -> String {
    if force {
        format!("{path}?force=true")
    } else {
        path.to_owned()
    }
}

/// Runs `rv deregister` (alias `forget`): evict and remove from the registry
/// and from config.json.
#[must_use]
pub fn deregister(ctx: &Ctx, model: &str, yes: bool, force: bool) -> i32 {
    if !ctx.require_admin_key() {
        return 1;
    }
    if !confirm(
        ctx.style,
        &format!(
            "deregister '{model}' — evicts if loaded AND removes it from config.json (gone on \
             next restart, not just VRAM). Proceed?"
        ),
        yes,
    ) {
        println!("aborted");
        return 2;
    }
    let r = ctx.api.delete(&eviction_path(
        &format!("/v1/admin/models/{}/register", encode_segment(model)),
        force,
    ));
    if r.code == Some(200) {
        println!(
            "{} {}",
            ctx.style.ok(),
            message(r.body.as_ref(), "deregistered")
        );
        0
    } else {
        print_error(ctx, &r);
        print_busy_hint(ctx, &r, &format!("deregister {model}"));
        1
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)] // fixtures: a failed setup should fail the test loudly
mod tests {

    /// S281 defect (a): a 200 with `"persisted": false` is told apart from a plain success.
    #[test]
    fn not_persisted_reads_the_flag_only() {
        let off = Payload::Json(serde_json::json!({"message": "m", "persisted": false}));
        let on = Payload::Json(serde_json::json!({"persisted": true}));
        let text = Payload::Text("model 'x' added to config and loaded".into());
        assert!(not_persisted(Some(&off)));
        assert!(!not_persisted(Some(&on)));
        assert!(!not_persisted(Some(&text)));
        assert!(!not_persisted(None));
    }
    #[test]
    fn kv_cache_gb_must_be_finite_and_positive() {
        assert!(check_kv_cache_gb(None).is_ok());
        assert!(check_kv_cache_gb(Some(2.5)).is_ok());
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let Err(err) = check_kv_cache_gb(Some(bad)) else {
                panic!("{bad} must be refused");
            };
            assert!(err.contains("--kv-cache-gb"), "{err}");
        }
    }

    use super::*;

    #[test]
    fn estimate_sums_the_whole_tree_with_headroom() {
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(d.path().join("unet")).expect("mkdir");
        std::fs::write(d.path().join("unet/w.bin"), vec![0u8; 2_000_000]).expect("write");
        std::fs::write(d.path().join("tok.json"), vec![0u8; 1_000_000]).expect("write");
        let (raw, est) = estimate_vram_gb(d.path()).estimate.expect("estimate");
        assert!((raw - 0.003).abs() < 1e-9, "{raw}");
        // 0.003 GB -> rounds to one decimal: 0.0
        assert!(est.abs() < f64::EPSILON, "{est}");
    }

    #[test]
    fn estimate_rounds_the_exact_value_not_a_pre_rounded_one() {
        // 13.75e9 bytes = 13.75 GB, an exact tie: half-even gives 13.8, as Python's round does.
        let d = tempfile::tempdir().expect("tempdir");
        let f = std::fs::File::create(d.path().join("w.bin")).expect("create");
        f.set_len(13_750_000_000).expect("set_len");
        let (_, est) = estimate_vram_gb(d.path()).estimate.expect("estimate");
        assert!((est - 13.8).abs() < 1e-9, "{est}");
    }

    #[test]
    fn estimate_is_the_plain_disk_size_even_for_a_model_that_measures_above_it() {
        // Intel's 14B: 9.7 GB .bin, 10.8 GB resident (an Arc Pro B60, 2026-10-14). The estimate stays at
        // the disk size (D-03); the server's over-budget warning is what reports the overshoot.
        let d = tempfile::tempdir().expect("tempdir");
        let f = std::fs::File::create(d.path().join("openvino_model.bin")).expect("create");
        f.set_len(9_700_000_000).expect("set_len");
        let (raw, est) = estimate_vram_gb(d.path()).estimate.expect("estimate");
        assert!((est - raw).abs() < 1e-9, "{est} vs {raw}");
        assert!((est - 9.7).abs() < 1e-9, "{est}");
    }

    #[test]
    fn estimate_fits_the_35b_on_a_24gb_card() {
        // qwen3.6-35b-a3b: 19.7 GB on disk. With the server's 1 GB minimum KV and 1 GB margin it
        // needs 21.7 GB, under the 22.7 GB that can be freed on a B60 (it was refused at 24.0).
        let d = tempfile::tempdir().expect("tempdir");
        let f = std::fs::File::create(d.path().join("w.bin")).expect("create");
        f.set_len(19_700_000_000).expect("set_len");
        let (_, est) = estimate_vram_gb(d.path()).estimate.expect("estimate");
        assert!(est + 1.0 + 1.0 <= 22.7, "{est}");
    }

    #[test]
    fn estimate_rounds_to_one_decimal() {
        let d = tempfile::tempdir().expect("tempdir");
        // sparse 3.2 GB file: cheap to create, big enough to exercise the rounding.
        let f = std::fs::File::create(d.path().join("openvino_model.bin")).expect("create");
        f.set_len(3_200_000_000).expect("set_len");
        let (raw, est) = estimate_vram_gb(d.path()).estimate.expect("estimate");
        assert!((raw - 3.2).abs() < 1e-9);
        assert!((est - 3.2).abs() < 1e-9, "{est}");
    }

    /// SF-2: backups (files and directories, any case) and a foreign `model_cache/` are not
    /// counted, but reported; a weights file merely containing "old" in its name still counts.
    #[test]
    fn estimate_skips_backups_and_foreign_compile_caches() {
        let d = tempfile::tempdir().expect("tempdir");
        let p = d.path();
        let sized = |rel: &str, len: u64| {
            let f = std::fs::File::create(p.join(rel)).expect("create");
            f.set_len(len).expect("set_len");
        };
        std::fs::create_dir(p.join("model_cache")).expect("mkdir");
        std::fs::create_dir(p.join("unet.OLD")).expect("mkdir");
        sized("openvino_model.bin", 2_000_000_000);
        sized("golden_old_weights.bin", 100_000_000);
        sized("openvino_model.bin.bak", 2_000_000_000);
        sized("tokenizer.json.orig", 100_000_000);
        sized("model_cache/blob.blob", 3_000_000_000);
        sized("unet.OLD/w.bin", 1_000_000_000);
        std::fs::create_dir_all(p.join(".git/lfs/objects")).expect("mkdir");
        std::fs::create_dir_all(p.join("sub/model_cache")).expect("mkdir");
        sized(".git/lfs/objects/w", 2_000_000_000);
        sized("sub/model_cache/b.blob", 500_000_000);
        sized("openvino_model.bin.part", 1_000_000_000);
        sized("sub/model_cache.json", 1_000_000); // only a DIRECTORY named model_cache is skipped
        let size = estimate_vram_gb(p);
        let (raw, est) = size.estimate.expect("estimate");
        assert!((raw - 2.101).abs() < 1e-9, "{raw}");
        assert!((est - 2.1).abs() < 1e-9, "{est}");
        assert!((size.skipped_gb - 9.6).abs() < 1e-9, "{}", size.skipped_gb);
        for kind in ["backups", "model_cache/", ".git/", "partial downloads"] {
            assert!(
                size.skipped_kinds.contains(&kind),
                "{:?}",
                size.skipped_kinds
            );
        }
        assert_eq!(size.skipped_kinds.len(), 4, "{:?}", size.skipped_kinds);
    }

    /// Only backups present: nothing to estimate from, as for an empty directory.
    #[test]
    fn estimate_of_a_dir_with_only_backups_is_none() {
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::write(d.path().join("openvino_model.bin.bak"), b"x").expect("write");
        let size = estimate_vram_gb(d.path());
        assert!(size.estimate.is_none());
        assert_eq!(size.skipped_kinds, ["backups"], "the reason is reported");
    }

    #[test]
    fn estimate_of_missing_or_empty_dir_is_none() {
        let d = tempfile::tempdir().expect("tempdir");
        assert!(estimate_vram_gb(d.path()).estimate.is_none());
        std::fs::create_dir(d.path().join("empty_sub")).expect("mkdir");
        assert!(estimate_vram_gb(d.path()).estimate.is_none());
        assert!(estimate_vram_gb(&d.path().join("nope")).estimate.is_none());
    }

    #[test]
    fn add_body_carries_only_what_was_given() {
        let a = AddArgs {
            model: "m".into(),
            vram_gb: None,
            kind: Some("vision".into()),
            device: None,
            kv_cache_gb: Some(2.0),
            max_concurrent_streams: Some(4),
            pinned: true,
            enable_prefix_caching: false,
            yes: true,
        };
        assert_eq!(
            add_body(&a, 5.5),
            json!({"model_id": "m", "vram_gb": 5.5, "kind": "vision",
                   "kv_cache_gb": 2.0, "max_concurrent_streams": 4, "pinned": true})
        );
        let bare = AddArgs {
            kind: None,
            kv_cache_gb: None,
            max_concurrent_streams: None,
            pinned: false,
            ..a
        };
        assert_eq!(
            add_body(&bare, 0.0),
            json!({"model_id": "m", "vram_gb": 0.0})
        );
        let pc = AddArgs {
            enable_prefix_caching: true,
            ..bare
        };
        assert_eq!(
            add_body(&pc, 0.0),
            json!({"model_id": "m", "vram_gb": 0.0, "enable_prefix_caching": true})
        );
    }

    #[test]
    fn success_message_prefers_the_servers_own_text() {
        assert_eq!(message(Some(&Payload::Text("done!".into())), "x"), "done!");
        assert_eq!(message(Some(&Payload::Json(json!({"a": 1}))), "x"), "x");
        assert_eq!(message(None, "x"), "x");
    }
}

#[cfg(test)]
mod check_path_tests {
    use super::check_path;

    #[test]
    fn busy_victims_are_named_in_the_prompt_only_when_present() {
        use serde_json::json;
        assert_eq!(super::busy_note(&json!([])), "");
        assert_eq!(super::busy_note(&json!(null)), "");
        let note = super::busy_note(&json!(["a", "b"]));
        assert!(
            note.contains("a, b busy") && note.contains("15 s"),
            "{note}"
        );
    }

    #[test]
    fn force_is_forwarded_to_the_pre_check() {
        assert_eq!(check_path("m", false), "/v1/admin/models/m/check");
        assert_eq!(check_path("m", true), "/v1/admin/models/m/check?force=true");
    }
}
