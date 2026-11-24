//! `rv add`, `load`, `unload`, `deregister`: verbs that change what is loaded
//! or registered.

use std::path::Path;

use serde_json::{Map, Value, json};

use crate::api::{LOAD_TIMEOUT, Payload, READ_TIMEOUT, encode_segment};
use crate::commands::outcome;
use crate::commands::prompt::confirm;
use crate::commands::refusal;
use crate::commands::{OldServerNote, admin_entry_quiet, print_entry_profile};
use crate::commands::{error_code, print_busy_hint, print_error};
use crate::ctx::Ctx;
use crate::format::GIB;
use crate::kind::detect_kind_from_markers;
use crate::metrics;
use crate::sampling::{
    self, EntryProfile, Origin, ProfileText, SamplingFlags, kind_has_sampling, preview_json,
    read_file_layer, render,
};

/// The margin on the on-disk size, in GiB (2^30 bytes, the unit the server reads `vram_gb` in since
/// 0.9.0). An explicit 1.07 replaces the accidental ~7.4 % the old decimal (10^9) divisor added, so
/// every booking stays what it was (R-12). No single ratio fits the models measured so
/// far: ministral-3b INT8 works at about 0.96 of its size, Intel's mixed INT4/INT8 qwen3-14b measured
/// 1.11 (9.7 GB `.bin`, 10.8 GB resident), qwen3.6-35b-a3b INT4 measured 1.005. A 1.12 margin
/// (0.8.3, F7) fixed the 14B but made `rv add` refuse the 35B, which runs. Jerzy chose the
/// optimistic side (D-03, 0.8.4): the server still adds its own minimum KV and safety margin, and
/// warns when a model measures well above its charge. The value is revisited after the planned
/// measured-footprint run.
const VRAM_ESTIMATE_MARGIN: f64 = 1.07;

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
    /// `(raw_disk_gib, estimated_vram_gib)` of the counted files; `None` when nothing counts
    /// (missing, empty, or only skipped entries).
    pub estimate: Option<(f64, f64)>,
    /// GiB of the entries left out ([`skipped_by_estimate`]).
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
    // GiB (2^30), the unit the server reads `vram_gb` in since 0.9.0 and `rv status` shows. The
    // headroom is VRAM_ESTIMATE_MARGIN, stated, no longer hidden in a decimal divisor.
    let (raw_gb, skipped_gb) = (w.total as f64 / GIB, w.skipped as f64 / GIB);
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
    /// VRAM estimate in GiB (0.0 to skip VRAM gating). Omit to auto-detect from
    /// the model directory's on-disk size
    #[arg(long)]
    pub vram_gb: Option<f64>,
    /// text_gen|vision|embedding|stt|tts|image_gen|reranking|decision (default: auto-detect;
    /// decision is never auto-detected and must be given)
    #[arg(long)]
    pub kind: Option<String>,
    /// e.g. GPU.1, CPU, NPU (default: the server places the model; the global config
    /// device is used only with default_placement: "device")
    #[arg(long)]
    pub device: Option<String>,
    /// KV cache size in GiB
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
    /// default thinking budget in tokens (0 = thinking off). STRICT: a request that thinks past
    /// it ends with finish_reason "length" and normally no answer
    #[arg(long)]
    pub thinking_budget_tokens: Option<u32>,
    /// don't prompt for confirmation
    #[arg(short = 'y', long)]
    pub yes: bool,
    #[command(flatten)]
    pub sampling: SamplingFlags,
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
            "{:.1} GiB of {}",
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
            "auto-detected vram_gb: {est:.1} GiB (the size on disk plus a 7 % margin; a \
             mixed-precision export can measure a little more; override with --vram-gb)"
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
    if let Some(t) = a.thinking_budget_tokens {
        body.insert("thinking_budget_tokens".into(), json!(t));
    }
    if a.sampling.any() {
        body.insert("sampling".into(), a.sampling.operator().to_json());
    }
    Value::Object(body)
}

/// The sampling checks of `rv add` that need no request: each flag's range, and a sampling flag
/// on a `--kind` that has no sampling profile.
fn add_sampling_precheck(a: &AddArgs) -> Result<(), String> {
    if !a.sampling.any() {
        return Ok(());
    }
    if let Some(kind) = a.kind.as_deref().filter(|k| !kind_has_sampling(k)) {
        return Err(format!(
            "error: sampling flags apply to text_gen and vision models only, not --kind {kind}"
        ));
    }
    a.sampling.operator().validate()
}

/// The local preview of `rv add`: the profile the server will resolve from the model's own
/// `generation_config.json` and the sampling flags (V5c). `Ok(None)` when there is nothing to
/// preview (a non-text `--kind` or a non-text kind the directory's files prove, an unknown
/// `models_dir`, a missing directory).
///
/// # Errors
/// Sampling flags on a directory whose files prove a non-text kind (review F2), or flags that
/// break the server's `do_sample` rule against the local file (the server would 400).
fn add_preview(ctx: &Ctx, a: &AddArgs) -> Result<Option<ProfileText>, String> {
    if a.kind.as_deref().is_some_and(|k| !kind_has_sampling(k)) {
        return Ok(None);
    }
    let Some(dir) = ctx.res.models_dir.as_deref().filter(|d| !d.is_empty()) else {
        return Ok(None);
    };
    let model_dir = Path::new(dir).join(&a.model);
    if a.kind.is_none()
        && let Some(kind) = detect_kind_from_markers(&model_dir).filter(|k| !kind_has_sampling(k))
    {
        if a.sampling.any() {
            return Err(format!(
                "error: sampling flags apply to text_gen and vision models only, and '{}' is a \
                 model of kind {kind} (from its files; pass --kind to override)",
                a.model
            ));
        }
        return Ok(None);
    }
    let Some(file) = read_file_layer(&model_dir) else {
        return Ok(None);
    };
    let op = a.sampling.operator();
    op.check_do_sample(file.samples_with_temperature())
        .map_err(|e| format!("{e}. Add --temperature, or drop --top-p/--top-k"))?;
    let npu = a
        .device
        .as_deref()
        .is_some_and(|d| d.to_ascii_uppercase().starts_with("NPU"));
    let origin = Origin::Local {
        file_found: file.found,
    };
    Ok(Some(render(
        &a.model,
        &preview_json(&op, &file, npu),
        origin,
    )))
}

/// After a successful add: the server's effective profile (V5c), or why there is none. Returns
/// the exit code: `1` when sampling flags were sent to a server that dropped them.
fn print_added_profile(ctx: &Ctx, a: &AddArgs) -> i32 {
    let st = ctx.style;
    let Some(entry) = admin_entry_quiet(ctx, &a.model) else {
        println!(
            "{}",
            st.dim("  (could not read the model's sampling profile back from the server)")
        );
        return 0;
    };
    match print_entry_profile(ctx, &a.model, &entry, OldServerNote::Warning) {
        EntryProfile::Profile(_) => 0,
        EntryProfile::OldServer => i32::from(a.sampling.any()),
        EntryProfile::NotApplicable => {
            if a.sampling.any() {
                eprintln!(
                    "{} {}",
                    st.warn(),
                    st.yellow(&format!(
                        "'{0}' was detected as a model of kind {1}: the sampling settings were \
                         stored but apply to text_gen and vision models only — `rv patch {0} \
                         --clear-sampling` removes them",
                        a.model,
                        entry["kind"].as_str().unwrap_or("non-text")
                    ))
                );
            }
            0
        }
    }
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
    if let Err(msg) = check_kv_cache_gb(a.kv_cache_gb).and_then(|()| add_sampling_precheck(a)) {
        eprintln!("{}", st.red(&msg));
        return 1;
    }
    let vram_gb = match vram_for_add(ctx, a) {
        Ok(v) => v,
        Err(code) => return code,
    };
    // Printed before the confirm question (so `-y` shows it too), like the vram line above.
    match add_preview(ctx, a) {
        Ok(Some(p)) => sampling::print(st, &p),
        Ok(None) => {}
        Err(msg) => {
            eprintln!("{}", st.red(&msg));
            return 1;
        }
    }

    // `add_model` funnels through the same evicting load path `rv load` uses —
    // it can evict serving models to make room. Unlike `load` there is no
    // `/check` dry-run (the model is not in the registry yet), so it needs the
    // confirmation gate `deregister` uses instead.
    if !confirm(
        st,
        &format!(
            "register '{}' as {vram_gb:.1} GiB (persists to config.json) and load it now — may \
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
            print_added_profile(ctx, a)
        }
        Some(200) => {
            println!(
                "{} {}",
                st.ok(),
                message(r.body.as_ref(), "added and loaded")
            );
            report_evicted(ctx, &before);
            print_added_profile(ctx, a)
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
    /// KV cache size in GiB
    #[arg(long, allow_hyphen_values = true)]
    pub kv_cache_gb: Option<f64>,
    /// concurrent stream cap
    #[arg(long)]
    pub max_concurrent_streams: Option<u32>,
    /// bypass the eviction-grace window AND evict a model that has a request running (it is
    /// drained for up to 15 s, then aborted; prints what the evictions did)
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

/// The "does not fit" verdict of `rv load`: device headroom, or the room is held by models that
/// were left alone only because a request is running on them (D3), or (RVR) the refusing gate —
/// the GPU budget, system RAM or both — with why nothing was evicted and the `config.json` cap
/// when it is proven to be the limit (see [`refusal`]). `caps`: the config's caps (`None`:
/// unreadable); `nothing_loaded`: whether the server lists no loaded model (`None`: unknown).
fn does_not_fit_message(
    st: crate::ansi::Style,
    model: &str,
    check: &Value,
    caps: Option<&refusal::Caps>,
    nothing_loaded: Option<bool>,
) -> String {
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
        // On a refusal `would_evict` is the server's whole sweep: evicting all of it is not enough.
        let sweep = refusal::sweep_len(check);
        if let Some(r) = refusal::analyse(check, caps, nothing_loaded) {
            let mut lines = vec![
                format!(
                    "{} '{model}' does not fit ({}) — {}",
                    st.bad(),
                    r.gates,
                    r.eviction
                        .as_deref()
                        .unwrap_or("nothing more can be evicted")
                ),
                format!("  {}", r.figures),
            ];
            if let Some(l) = r.config_line {
                lines.push(format!("  {l}"));
            } else if r.eviction.is_none() {
                lines.push(GENERIC_EVICTION_CAUSES.to_owned());
            } else if sweep > 0 && !r.cap_proven {
                lines.push(GENERIC_REST_CAUSES.to_owned());
            }
            return lines.join("\n");
        }
        if sweep > 0 {
            return format!(
                "{} '{model}' does not fit — needs {needed:.1} GB, {free:.1} GB free, {}",
                st.bad(),
                refusal::sweep_phrase(sweep)
            );
        }
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

/// Why "nothing more can be evicted", when `rv` cannot tell which applies.
const GENERIC_EVICTION_CAUSES: &str = "  (pinned, non-evictable, realtime or in the \
                                       eviction-grace window; `--force` skips only the grace)";

/// After the evictable sweep was not enough and no config cap explains it: what else may hold the
/// room (protected residents, or memory outside this server on the RAM gate).
const GENERIC_REST_CAUSES: &str = "  (what is left is pinned, non-evictable, realtime, in the \
                                   grace window or outside rustedvino)";

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
    // The models the pre-check says this load evicts (`None`: not checked, or it said nothing).
    let mut victims: Option<Vec<String>> = None;
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
                let caps = refusal::Caps::read(&ctx.res.config_file);
                let nothing = refusal::nothing_loaded_now(ctx);
                println!(
                    "{}",
                    does_not_fit_message(st, &a.model, check, caps.as_ref(), nothing)
                );
                return 1;
            }
            let evict: Vec<&str> = check["would_evict"]
                .as_array()
                .map(|v| v.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            victims = Some(evict.iter().map(|m| (*m).to_owned()).collect());
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
    send_load(ctx, a, &seg, victims.as_deref())
}

/// The load itself, after the pre-check: the forced pre-call line, the timed POST, the outcome
/// and the load-time line.
fn send_load(ctx: &Ctx, a: &LoadArgs, seg: &str, victims: Option<&[String]>) -> i32 {
    let st = ctx.style;
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
    let running = if a.force {
        let running = victims.and_then(|v| {
            let ids: Vec<&str> = v.iter().map(String::as_str).collect();
            outcome::running(outcome::scrape(ctx).as_deref(), &ids)
        });
        // R-9: a check that names no victim — the load evicts nothing, so no drain to announce.
        let line = if victims.is_some_and(<[String]>::is_empty) {
            "the pre-load check says this load evicts nothing".to_owned()
        } else {
            format!(
                "{} (each model it evicts)",
                outcome::precall_line(running, outcome::LOAD_EVICT_DRAIN_S)
            )
        };
        println!("{}", st.dim(&line));
        running
    } else {
        None
    };
    let path = format!("/v1/admin/models/{seg}/load");
    let started = std::time::Instant::now();
    let r = if a.force {
        ctx.api.post_accept_json(&path, body.as_ref(), LOAD_TIMEOUT)
    } else {
        ctx.api.post(&path, body.as_ref(), LOAD_TIMEOUT)
    };
    let took = started.elapsed();
    if r.code == Some(200) && before.contains(&a.model) {
        // Already Ready (the idempotent 200, reached with --no-check): nothing was loaded, so
        // there is no load time to show — the metric still holds an earlier load's (R-5).
        println!("{} '{}' is already loaded", st.ok(), a.model);
        0
    } else if r.code == Some(200) {
        let got = a.force.then(|| outcome::parse(r.body.as_ref())).flatten();
        if let Some(o) = &got {
            println!(
                "{} {}",
                st.ok(),
                json_message(r.body.as_ref(), &format!("model '{}' loaded", a.model))
            );
            print_load_outcome(ctx, o, running);
        } else {
            println!("{} {}", st.ok(), message(r.body.as_ref(), "loaded"));
            report_evicted(ctx, &before);
            if a.force {
                println!(
                    "{}",
                    st.dim(&format!("  {}", outcome::clause(None, running)))
                );
            }
        }
        println!(
            "{}",
            st.dim(&format!("  {}", load_time_line(ctx, &a.model, took)))
        );
        0
    } else {
        print_error(ctx, &r);
        print_busy_hint(ctx, &r, &format!("load {}", a.model));
        1
    }
}

/// The success sentence of an outcome-JSON answer: its `message` (the plain-text body,
/// verbatim, since 0.9.1), else `default`.
pub(crate) fn json_message(body: Option<&Payload>, default: &str) -> String {
    body.and_then(Payload::json)
        .and_then(|v| v.get("message"))
        .and_then(Value::as_str)
        .map_or_else(|| default.to_owned(), str::to_owned)
}

/// What a forced load's evictions did (V1-rv P1).
fn print_load_outcome(ctx: &Ctx, o: &outcome::Outcome, running: Option<u64>) {
    if o.evicted.is_empty() {
        println!("{}", ctx.style.dim("  nothing was evicted"));
        return;
    }
    println!(
        "{}",
        ctx.style.dim(&format!(
            "  evicted to make room: {} \u{2014} {}",
            o.evicted.join(", "),
            outcome::clause(Some(o), running)
        ))
    );
    outcome::warn_if_still_running(ctx, Some(o), "the forced load");
}

/// `loaded in 13.1 s on GPU.1 (engine 12.8 s)` (V1-rv P5): the POST timed here, beside the
/// server's `rustedvino_model_load_duration_seconds` for the model on the device it is loaded on
/// now (the registry entry's `device`: the gauge keeps a series per device it was ever loaded on,
/// R-5). The gap is admission, the evictions and HTTP. "engine unknown" when the registry or the
/// metric is not there (an older server, no admin scrape).
fn load_time_line(ctx: &Ctx, model: &str, took: std::time::Duration) -> String {
    let device = admin_entry_quiet(ctx, model)
        .and_then(|e| e.get("device").and_then(Value::as_str).map(str::to_owned));
    let engine = device.as_deref().and_then(|d| {
        outcome::scrape(ctx).and_then(|rows| {
            metrics::by_label(
                &rows,
                "rustedvino_model_load_duration_seconds",
                &[("model", model), ("device", d)],
            )
        })
    });
    load_time_text(took.as_secs_f64(), device.as_deref(), engine)
}

/// The text of [`load_time_line`].
fn load_time_text(took_s: f64, device: Option<&str>, engine_s: Option<f64>) -> String {
    let on = device.map_or_else(String::new, |d| format!(" on {d}"));
    let engine = engine_s.map_or_else(
        || "engine unknown".to_owned(),
        |e| format!("engine {e:.1} s"),
    );
    format!("loaded in {took_s:.1} s{on} ({engine})")
}

/// Runs `rv unload`: evict a model from VRAM (it stays registered).
#[must_use]
pub fn unload(ctx: &Ctx, model: &str, force: bool) -> i32 {
    if !ctx.require_admin_key() {
        return 1;
    }
    println!("{}", ctx.style.dim(&format!("unloading '{model}'...")));
    let path = eviction_path(
        &format!("/v1/admin/models/{}", encode_segment(model)),
        force,
    );
    if force {
        return forced_eviction(
            ctx,
            model,
            &path,
            "unload",
            &format!("model '{model}' evicted"),
        );
    }
    let r = ctx.api.delete(&path);
    if r.code == Some(200) {
        println!("{} {}", ctx.style.ok(), message(r.body.as_ref(), "evicted"));
        0
    } else {
        print_error(ctx, &r);
        print_busy_hint(ctx, &r, &format!("unload {model}"));
        1
    }
}

/// A forced unload or deregister (V1-rv P1): the pre-call line from the model's running requests,
/// then the DELETE with `Accept: application/json`, then what the force did. `done` is the
/// success text (the server's own on an older server that answers plain text).
fn forced_eviction(ctx: &Ctx, model: &str, path: &str, verb: &str, done: &str) -> i32 {
    let running = outcome::running(outcome::scrape(ctx).as_deref(), &[model]);
    println!(
        "{}",
        ctx.style
            .dim(&outcome::precall_line(running, outcome::EVICT_DRAIN_S))
    );
    let r = ctx.api.delete_accept_json(path);
    if r.code == Some(200) {
        let got = outcome::parse(r.body.as_ref());
        // A plain-text answer (older server) keeps the server's own sentence; a JSON answer of a
        // deregister of a model that was not loaded evicted nothing.
        let head = if got.is_some() {
            json_message(r.body.as_ref(), done)
        } else {
            message(r.body.as_ref(), done)
        };
        let what = match &got {
            Some(o) if o.evicted.is_empty() => "it was not loaded, nothing was evicted".to_owned(),
            _ => outcome::clause(got.as_ref(), running),
        };
        println!("{} {head} \u{2014} {what}", ctx.style.ok());
        outcome::warn_if_still_running(ctx, got.as_ref(), &format!("{verb} {model}"));
        0
    } else {
        print_error(ctx, &r);
        print_busy_hint(ctx, &r, &format!("{verb} {model}"));
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
    let path = eviction_path(
        &format!("/v1/admin/models/{}/register", encode_segment(model)),
        force,
    );
    if force {
        return forced_eviction(
            ctx,
            model,
            &path,
            "deregister",
            &format!("model '{model}' deregistered"),
        );
    }
    let r = ctx.api.delete(&path);
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
        assert!((raw - 3_000_000.0 / GIB).abs() < 1e-12, "{raw}");
        // 0.0028 GiB -> rounds to one decimal: 0.0
        assert!(est.abs() < f64::EPSILON, "{est}");
    }

    #[test]
    fn estimate_rounds_the_exact_value_not_a_pre_rounded_one() {
        // 13.75e9 bytes = 12.806 GiB; x 1.07 = 13.702 -> 13.7 (the old 1e9 divisor booked 13.8).
        // Then 12.85 GiB: x 1.07 = 13.7495 -> 13.7, where rounding the GiB figure first (12.9 x 1.07
        // = 13.803) would give 13.8. One rounding, at the end.
        let d = tempfile::tempdir().expect("tempdir");
        let f = std::fs::File::create(d.path().join("w.bin")).expect("create");
        f.set_len(13_750_000_000).expect("set_len");
        let (_, est) = estimate_vram_gb(d.path()).estimate.expect("estimate");
        assert!((est - 13.7).abs() < 1e-9, "{est}");
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let bytes = (12.85 * GIB) as u64;
        f.set_len(bytes).expect("set_len");
        let (raw, est) = estimate_vram_gb(d.path()).estimate.expect("estimate");
        assert!((raw - 12.85).abs() < 1e-6, "{raw}");
        assert!((est - 13.7).abs() < 1e-9, "{est}");
    }

    /// R-12: GiB x 1.07 is the old decimal-GB figure x 0.9965 (1.07 / 1.0737). So
    /// it never books more than before, and books within 0.1 of it up to about 28.7e9 bytes on
    /// disk; above that the gap grows with the size (28.75e9 bytes: 28.8 -> 28.6; 80e9: 80.0 ->
    /// 79.7), staying within 0.35 % plus the rounding.
    #[test]
    fn estimate_in_gib_books_at_most_the_old_decimal_figure_and_close_to_it() {
        for bytes in [
            3_200_000_000_u64,
            9_700_000_000,
            13_750_000_000,
            19_700_000_000,
            28_700_000_000,
            28_750_000_000,
            80_000_000_000,
        ] {
            let d = tempfile::tempdir().expect("tempdir");
            let f = std::fs::File::create(d.path().join("w.bin")).expect("create");
            f.set_len(bytes).expect("set_len");
            let (_, est) = estimate_vram_gb(d.path()).estimate.expect("estimate");
            #[allow(clippy::cast_precision_loss)]
            let old: f64 = format!("{:.1}", bytes as f64 / 1e9).parse().expect("parse");
            assert!(est <= old + 1e-9, "{bytes}: {est} books more than {old}");
            assert!(
                old - est <= old * 0.0035 + 0.1 + 1e-9,
                "{bytes}: {est} vs {old}"
            );
            if bytes <= 28_700_000_000 {
                assert!(old - est <= 0.1 + 1e-9, "{bytes}: {est} vs {old}");
            }
        }
    }

    #[test]
    fn estimate_is_the_disk_size_plus_the_margin_even_for_a_model_that_measures_above_it() {
        // Intel's 14B: 9.7e9-byte .bin, 10.8 GB resident (an Arc Pro B60, 2026-10-14). The estimate is
        // the disk size in GiB (9.03) plus the 1.07 margin = 9.67 -> 9.7, the same booking as before
        // R-12 (D-03); the server's over-budget warning is what reports the overshoot.
        let d = tempfile::tempdir().expect("tempdir");
        let f = std::fs::File::create(d.path().join("openvino_model.bin")).expect("create");
        f.set_len(9_700_000_000).expect("set_len");
        let (raw, est) = estimate_vram_gb(d.path()).estimate.expect("estimate");
        assert!((raw - 9_700_000_000.0 / GIB).abs() < 1e-9, "{raw}");
        assert!((est - 9.7).abs() < 1e-9, "{est} vs {raw}");
    }

    #[test]
    fn estimate_fits_the_35b_on_a_24gb_card() {
        // qwen3.6-35b-a3b: 19.7e9 bytes on disk = 18.35 GiB, x 1.07 -> 19.6. With the server's 1 GiB
        // minimum KV and 1 GiB margin it needs 21.6, under the 22.7 that can be freed on a B60 (it
        // was refused at 24.0).
        let d = tempfile::tempdir().expect("tempdir");
        let f = std::fs::File::create(d.path().join("w.bin")).expect("create");
        f.set_len(19_700_000_000).expect("set_len");
        let (_, est) = estimate_vram_gb(d.path()).estimate.expect("estimate");
        assert!(est + 1.0 + 1.0 <= 22.7, "{est}");
    }

    #[test]
    fn estimate_rounds_to_one_decimal() {
        let d = tempfile::tempdir().expect("tempdir");
        // sparse 3.2e9-byte file: cheap to create, big enough to exercise the rounding.
        // 3.2e9 / 2^30 = 2.980 GiB; x 1.07 = 3.189 -> 3.2, today's booking (1e9 divisor: 3.2).
        let f = std::fs::File::create(d.path().join("openvino_model.bin")).expect("create");
        f.set_len(3_200_000_000).expect("set_len");
        let (raw, est) = estimate_vram_gb(d.path()).estimate.expect("estimate");
        assert!((raw - 2.980_232_238_769_531_2).abs() < 1e-9, "{raw}");
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
        assert!((raw - 2_101_000_000.0 / GIB).abs() < 1e-9, "{raw}");
        assert!((est - 2.1).abs() < 1e-9, "{est}");
        assert!(
            (size.skipped_gb - 9_600_000_000.0 / GIB).abs() < 1e-9,
            "{}",
            size.skipped_gb
        );
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
            thinking_budget_tokens: None,
            yes: true,
            sampling: SamplingFlags::default(),
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
        let tb = AddArgs {
            enable_prefix_caching: false,
            thinking_budget_tokens: Some(0),
            ..pc
        };
        assert_eq!(
            add_body(&tb, 0.0),
            json!({"model_id": "m", "vram_gb": 0.0, "thinking_budget_tokens": 0})
        );
    }

    fn add_args(kind: Option<&str>, sampling: SamplingFlags) -> AddArgs {
        AddArgs {
            model: "m".into(),
            vram_gb: Some(1.0),
            kind: kind.map(str::to_owned),
            device: None,
            kv_cache_gb: None,
            max_concurrent_streams: None,
            pinned: false,
            enable_prefix_caching: false,
            thinking_budget_tokens: None,
            yes: true,
            sampling,
        }
    }

    #[test]
    fn add_body_carries_the_sampling_flags() {
        let a = add_args(
            None,
            SamplingFlags {
                temperature: Some(0.15),
                top_p: Some(0.95),
                presence_penalty: Some(-0.5),
                ..SamplingFlags::default()
            },
        );
        assert_eq!(
            add_body(&a, 1.0),
            json!({"model_id": "m", "vram_gb": 1.0,
                   "sampling": {"temperature": 0.15, "top_p": 0.95, "presence_penalty": -0.5}})
        );
        assert!(add_sampling_precheck(&a).is_ok());
    }

    #[test]
    fn add_sampling_flags_are_checked_for_kind_and_range() {
        let t = SamplingFlags {
            temperature: Some(0.2),
            ..SamplingFlags::default()
        };
        let e = add_sampling_precheck(&add_args(Some("embedding"), t.clone())).expect_err("kind");
        assert!(e.contains("text_gen and vision"), "{e}");
        assert!(add_sampling_precheck(&add_args(Some("vision"), t)).is_ok());
        let bad = SamplingFlags {
            top_k: Some(0),
            ..SamplingFlags::default()
        };
        assert!(add_sampling_precheck(&add_args(None, bad)).is_err());
        // No flags: any kind is fine.
        assert!(add_sampling_precheck(&add_args(Some("stt"), SamplingFlags::default())).is_ok());
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
    use super::{check_path, load_time_text};

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
    fn load_time_text_shows_both_times_or_says_unknown() {
        assert_eq!(
            load_time_text(13.06, Some("GPU.1"), Some(12.84)),
            "loaded in 13.1 s on GPU.1 (engine 12.8 s)"
        );
        assert_eq!(
            load_time_text(2.0, Some("CPU"), None),
            "loaded in 2.0 s on CPU (engine unknown)"
        );
        assert_eq!(
            load_time_text(2.0, None, None),
            "loaded in 2.0 s (engine unknown)"
        );
    }

    #[test]
    fn force_is_forwarded_to_the_pre_check() {
        assert_eq!(check_path("m", false), "/v1/admin/models/m/check");
        assert_eq!(check_path("m", true), "/v1/admin/models/m/check?force=true");
    }

    // ---- RVR: the does-not-fit verdict names the refusing gate ----

    use super::{GENERIC_EVICTION_CAUSES, GENERIC_REST_CAUSES, does_not_fit_message, refusal};
    use crate::ansi::Style;
    use serde_json::{Value, json};

    const ID_35B: &str = "qwen3.6-35b-a3b-int4-ov";

    fn live_35b() -> Value {
        json!({"fits": false, "needed_gb": 22.0, "free_gb": 18.0, "vram_gated": true,
               "would_evict": [], "busy_victims": [], "blocked_by_busy": [],
               "ram_needed_gb": 24.0, "ram_avail_gb": 17.0, "already_loaded": false})
    }

    fn live_caps() -> refusal::Caps {
        refusal::Caps::from_value(&json!({"total_vram_gb": 18.0, "system_ram_budget_gb": 17.0,
                                          "system_ram_reservation_gb": 2.0}))
    }

    #[test]
    fn live_35b_verdict_names_both_gates_the_caps_and_fits_100_columns() {
        let m = does_not_fit_message(
            Style::plain(),
            ID_35B,
            &live_35b(),
            Some(&live_caps()),
            Some(true),
        );
        let lines: Vec<&str> = m.lines().collect();
        assert_eq!(
            lines,
            vec![
                "● 'qwen3.6-35b-a3b-int4-ov' does not fit (GPU budget and system RAM) — nothing is \
                 loaded to evict",
                "  GPU budget: needs 22.0 GB, 18.0 GB free · RAM: needs 24.0 GB + 2.0 GB reserve, \
                 17.0 GB available",
                "  limit: config.json sets total_vram_gb 18.0 / system_ram_budget_gb 17.0 — raise \
                 to load this model",
            ]
        );
        assert!(!m.contains("pinned"), "{m}");
        for l in &lines {
            assert!(l.chars().count() <= 100, "{} cols: {l}", l.chars().count());
        }
    }

    #[test]
    fn unknown_evictability_keeps_the_generic_causes_within_100_columns() {
        let c = json!({"fits": false, "needed_gb": 9.0, "free_gb": 3.0});
        let m = does_not_fit_message(Style::plain(), "m", &c, None, None);
        assert_eq!(
            m,
            format!(
                "● 'm' does not fit (GPU budget) — nothing more can be evicted\n  GPU budget: \
                 needs 9.0 GB, 3.0 GB free\n{GENERIC_EVICTION_CAUSES}"
            )
        );
        assert!(GENERIC_EVICTION_CAUSES.chars().count() <= 100);
    }

    #[test]
    fn ram_gate_only_and_nothing_loaded_drops_the_generic_causes() {
        let c = json!({"fits": false, "needed_gb": 12.0, "free_gb": 20.0,
                       "ram_needed_gb": 15.0, "ram_avail_gb": 16.0});
        let caps = refusal::Caps::from_value(&json!({"system_ram_reservation_gb": 2.0}));
        let m = does_not_fit_message(Style::plain(), "m", &c, Some(&caps), Some(true));
        assert_eq!(
            m,
            "● 'm' does not fit (system RAM) — nothing is loaded to evict\n  RAM: \
             needs 15.0 GB + 2.0 GB reserve, 16.0 GB available"
        );
    }

    #[test]
    fn busy_branch_is_unchanged() {
        let c = json!({"fits": false, "needed_gb": 9.0, "free_gb": 2.0,
                       "blocked_by_busy": ["bielik"]});
        let m = does_not_fit_message(Style::plain(), "big", &c, Some(&live_caps()), Some(false));
        assert_eq!(
            m,
            "● 'big' needs 9.0 GB and only 2.0 GB is free; the room is held by bielik which has a \
             request running, so nothing was evicted — retry when it finishes, or `rv load big \
             --force` to drain it (15 s) and abort what is left"
        );
    }

    #[test]
    fn headroom_branch_is_unchanged() {
        let mut c = live_35b();
        c["headroom_refusal"] = json!("needs 3 GiB beyond the pool");
        let m = does_not_fit_message(Style::plain(), "v", &c, Some(&live_caps()), Some(true));
        assert_eq!(
            m,
            "● 'v' does not fit — device headroom: needs 3 GiB beyond the pool; nothing was \
             evicted"
        );
    }

    /// The coordinator's live check (2026-11-21): five models loaded, all swept, still no fit.
    fn live_loaded_35b() -> Value {
        json!({"fits": false, "needed_gb": 22.0, "free_gb": 1.0068, "vram_gated": true,
               "would_evict": ["multilingual-e5-large-int8", "bge-reranker-v2-m3-int8-ov",
                               "whisper-large-v3-turbo-int8-ov", "speecht5-tts-ov",
                               "qwen3.5-9B-int4-ov"],
               "busy_victims": [], "blocked_by_busy": [],
               "ram_needed_gb": 24.0, "ram_avail_gb": 10.96, "already_loaded": false})
    }

    #[test]
    fn live_loaded_35b_verdict_names_the_sweep_and_the_proven_caps() {
        let m = does_not_fit_message(
            Style::plain(),
            ID_35B,
            &live_loaded_35b(),
            Some(&live_caps()),
            Some(false),
        );
        let lines: Vec<&str> = m.lines().collect();
        assert_eq!(
            lines,
            vec![
                "● 'qwen3.6-35b-a3b-int4-ov' does not fit (GPU budget and system RAM) — even after \
                 evicting all 5 evictable models",
                "  GPU budget: needs 22.0 GB, 1.0 GB free · RAM: needs 24.0 GB + 2.0 GB reserve, \
                 11.0 GB available",
                "  limit: config.json sets total_vram_gb 18.0 / system_ram_budget_gb 17.0 — raise \
                 to load this model",
            ]
        );
        assert!(!m.contains("nothing more can be evicted"), "{m}");
        // The detail lines fit 100 columns; the verdict does with a model id up to 10 characters
        // (this 23-character id takes it to 113).
        for l in &lines[1..] {
            assert!(l.chars().count() <= 100, "{} cols: {l}", l.chars().count());
        }
        assert_eq!(lines[0].chars().count() - ID_35B.chars().count(), 90);
    }

    #[test]
    fn a_sweep_short_of_room_under_the_caps_names_what_else_may_hold_it() {
        // 12 > 1 free, all under the 18 cap: the two swept models were not enough, so protected
        // residents (pinned, grace, …) hold the rest; no config line.
        let c = json!({"fits": false, "needed_gb": 12.0, "free_gb": 1.0, "vram_gated": true,
                       "would_evict": ["a", "b"], "busy_victims": [], "blocked_by_busy": []});
        let m = does_not_fit_message(Style::plain(), "m", &c, Some(&live_caps()), Some(false));
        assert_eq!(
            m,
            format!(
                "● 'm' does not fit (GPU budget) — even after evicting all 2 evictable models\n  \
                 GPU budget: needs 12.0 GB, 1.0 GB free\n{GENERIC_REST_CAUSES}"
            )
        );
        assert!(GENERIC_REST_CAUSES.chars().count() <= 100);
    }

    #[test]
    fn no_gate_by_rvs_arithmetic_keeps_the_old_sentence() {
        let c = json!({"fits": false, "needed_gb": 4.0, "free_gb": 9.0});
        let m = does_not_fit_message(Style::plain(), "m", &c, None, None);
        assert!(m.contains("needs 4.0 GB, 9.0 GB free, and nothing more can be evicted"));
    }
}
