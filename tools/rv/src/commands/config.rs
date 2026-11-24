//! `rv reload-config`, `patch`, `preload`, `save-defaults`: verbs that change
//! configuration rather than what is loaded.

use serde_json::{Map, Value, json};

use crate::api::{READ_TIMEOUT, Reply, encode_segment};
use crate::commands::prompt::confirm;
use crate::commands::{admin_entry, print_error};
use crate::ctx::Ctx;
use crate::sampling::{
    EntryProfile, OLD_SERVER_WARNING, Operator, SamplingFlags, entry_profile, merge_for_patch,
};

fn strings(v: &Value, key: &str) -> Vec<String> {
    v.get(key)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// Runs `rv reload-config`. No confirmation gate, unlike add/deregister: it
/// only ever registers new/changed entries as `NotLoaded` or leaves
/// Ready/Loading/Evicting ones untouched — it never loads or evicts anything,
/// so it carries none of their blast radius.
#[must_use]
pub fn reload_config(ctx: &Ctx) -> i32 {
    if !ctx.require_admin_key() {
        return 1;
    }
    let st = ctx.style;
    let r = ctx.api.post("/v1/admin/config/reload", None, READ_TIMEOUT);
    let Some(body) = r.json().filter(|v| r.code == Some(200) && v.is_object()) else {
        print_plain_error(ctx, &r);
        return 1;
    };
    if ctx.json {
        println!("{}", serde_json::to_string_pretty(body).unwrap_or_default());
        return 0;
    }
    for (key, label) in [
        ("added", "added"),
        ("updated", "updated"),
        ("left_untouched", "left untouched (Ready/Loading/Evicting)"),
        (
            "skipped_missing_files",
            "skipped (directory missing under models_dir)",
        ),
    ] {
        let items = strings(body, key);
        if !items.is_empty() {
            let glyph = if key == "skipped_missing_files" {
                st.warn()
            } else {
                st.ok()
            };
            println!("{glyph} {} {label}: {}", items.len(), items.join(", "));
        }
    }
    let unchanged = body.get("unchanged").and_then(Value::as_i64).unwrap_or(0);
    println!("{}", st.dim(&format!("{unchanged} unchanged")));
    // Reload re-reads only the per-model registry; global knobs are a boot
    // snapshot. Without this line a 200 OK reads as "everything landed", which
    // is how a changed min_kv_cache_gb silently does nothing. An older server
    // omits the key, which degrades quietly. Deliberately does NOT say "these
    // need a restart": some are read live on every load and could be
    // refreshed, that is just not built — the server promises no taxonomy
    // beyond "not applied", so neither does this line.
    let skipped = strings(body, "globals_not_applied");
    if !skipped.is_empty() {
        println!(
            "{}",
            st.yellow(&format!(
                "{} {} global setting(s) not applied by reload — restart to pick them up: {}",
                st.warn(),
                skipped.len(),
                skipped.join(", ")
            ))
        );
    }
    // C8: a loaded model's edited stanza is not applied, and unload + load alone would still use
    // the old entry (reload stages nothing for a live model). An older server omits the key.
    for (model, fields) in models_not_applied(body) {
        println!(
            "{}",
            st.yellow(&format!(
                "{} {model} is loaded: changed {} not applied — to apply: rv unload {model}, \
                 rv reload-config, rv load {model}",
                st.warn(),
                fields.join(", ")
            ))
        );
    }
    // V5: a changed `sampling` on a loaded model is applied to its next request (0.9.1).
    for line in applied_live_lines(body) {
        println!("{} {line}", st.ok());
    }
    0
}

/// `<model>: sampling applied live`, one per entry of the reload response's `applied_live`
/// (0.9.1; an older server omits the key).
fn applied_live_lines(body: &Value) -> Vec<String> {
    body.get("applied_live")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| {
            let model = m.get("model")?.as_str()?;
            Some(format!(
                "{model}: {} applied live",
                strings(m, "fields").join(", ")
            ))
        })
        .collect()
}

/// `(model, changed fields)` of the reload response's `models_not_applied`, in server order.
fn models_not_applied(body: &Value) -> Vec<(String, Vec<String>)> {
    body.get("models_not_applied")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| {
            let model = m.get("model")?.as_str()?.to_owned();
            Some((model, strings(m, "fields")))
        })
        .collect()
}

/// `error: ...` form (no status code), used by the verbs whose Python original
/// prints `error: {err or code}`.
fn print_plain_error(ctx: &Ctx, r: &Reply) {
    let detail = r
        .err
        .clone()
        .unwrap_or_else(|| crate::commands::code_text(r.code));
    eprintln!("{}", ctx.style.red(&format!("error: {detail}")));
}

/// Flags of `rv patch`. Doc comments here are `--help` text, so doc_markdown is
/// relaxed.
// The bools are independent `--clear-*` CLI switches, not a state machine.
#[allow(clippy::doc_markdown, clippy::struct_excessive_bools)]
#[derive(clap::Args, Debug, Clone)]
pub struct PatchArgs {
    /// model id
    pub model: String,
    /// VRAM estimate in GiB
    #[arg(long)]
    pub vram_gb: Option<f64>,
    /// never evict this model once loaded
    #[arg(long, value_parser = ["true", "false"])]
    pub pinned: Option<String>,
    /// eviction priority
    #[arg(long, allow_hyphen_values = true)]
    pub priority: Option<i64>,
    /// eager|on_demand
    #[arg(long, value_parser = ["eager", "on_demand"])]
    pub load: Option<String>,
    /// placement-affecting — rejected while the model is Ready (evict first)
    #[arg(long)]
    pub device: Option<String>,
    /// clear the device override: the server places the model again (the global config
    /// device only with default_placement: "device")
    #[arg(long)]
    pub clear_device: bool,
    /// eviction grace window in seconds
    #[arg(long)]
    pub eviction_grace_secs: Option<f64>,
    /// clear the override back to the global eviction_grace_secs default
    #[arg(long)]
    pub clear_eviction_grace_secs: bool,
    /// default thinking budget in tokens (0 = thinking off). STRICT: a request that thinks past
    /// it ends with finish_reason "length" and normally no answer
    #[arg(long)]
    pub thinking_budget_tokens: Option<u32>,
    /// clear the model's default thinking budget (no budget)
    #[arg(long)]
    pub clear_thinking_budget_tokens: bool,
    #[command(flatten)]
    pub sampling: SamplingFlags,
    /// drop this model's operator sampling values (back to generation_config.json / server
    /// defaults)
    #[arg(long)]
    pub clear_sampling: bool,
}

impl PatchArgs {
    /// The patch touches `sampling` (a value flag or `--clear-sampling`).
    #[must_use]
    pub fn touches_sampling(&self) -> bool {
        self.clear_sampling || self.sampling.any()
    }
}

/// Checks of the sampling flags that need no request: `--clear-sampling` against the value
/// flags, and each value's range (the server's rules, so the error comes before the request).
///
/// # Errors
/// A conflict or an out-of-range value.
pub fn sampling_precheck(a: &PatchArgs) -> Result<(), String> {
    if a.clear_sampling && a.sampling.any() {
        return Err(
            "error: --clear-sampling and the sampling value flags (--temperature/--top-p/--top-k/\
             --repetition-penalty/--presence-penalty/--frequency-penalty) are mutually exclusive"
                .to_owned(),
        );
    }
    a.sampling.operator().validate()
}

/// Builds the `PATCH` body, or the reason it cannot be built. Pure, so the
/// flag combinations are unit-testable. `sampling` is the merged object (or
/// `null`) from [`merge_for_patch`] when the patch touches sampling.
///
/// # Errors
///
/// A conflicting flag pair, an out-of-range sampling flag, or no field at all.
pub fn patch_body(a: &PatchArgs, sampling: Option<&Value>) -> Result<Value, String> {
    if a.clear_device && a.device.is_some() {
        return Err("error: --device and --clear-device are mutually exclusive".to_owned());
    }
    if a.clear_eviction_grace_secs && a.eviction_grace_secs.is_some() {
        return Err(
            "error: --eviction-grace-secs and --clear-eviction-grace-secs are mutually exclusive"
                .to_owned(),
        );
    }
    if a.clear_thinking_budget_tokens && a.thinking_budget_tokens.is_some() {
        return Err(
            "error: --thinking-budget-tokens and --clear-thinking-budget-tokens are mutually \
             exclusive"
                .to_owned(),
        );
    }
    sampling_precheck(a)?;
    let mut body = Map::new();
    if let Some(v) = a.vram_gb {
        body.insert("vram_gb".into(), json!(v));
    }
    if let Some(p) = &a.pinned {
        body.insert("pinned".into(), json!(p == "true"));
    }
    if let Some(p) = a.priority {
        body.insert("priority".into(), json!(p));
    }
    if let Some(l) = &a.load {
        body.insert("load".into(), json!(l));
    }
    if let Some(d) = &a.device {
        body.insert("device".into(), json!(d));
    } else if a.clear_device {
        body.insert("device".into(), Value::Null);
    }
    if let Some(g) = a.eviction_grace_secs {
        body.insert("eviction_grace_secs".into(), json!(g));
    } else if a.clear_eviction_grace_secs {
        body.insert("eviction_grace_secs".into(), Value::Null);
    }
    if let Some(t) = a.thinking_budget_tokens {
        body.insert("thinking_budget_tokens".into(), json!(t));
    } else if a.clear_thinking_budget_tokens {
        body.insert("thinking_budget_tokens".into(), Value::Null);
    }
    if let Some(s) = sampling {
        body.insert("sampling".into(), s.clone());
    }
    if body.is_empty() {
        return Err(
            "error: no fields supplied -- pass at least one of --vram-gb/--pinned/--priority/\
             --load/--device/--clear-device/--eviction-grace-secs/--clear-eviction-grace-secs/\
             --thinking-budget-tokens/--clear-thinking-budget-tokens/\
             --temperature/--top-p/--top-k/--repetition-penalty/--presence-penalty/\
             --frequency-penalty/--clear-sampling (anything else -- kv_cache_gb, \
             tier_preference, capabilities, etc. -- needs a raw curl PATCH; see INTERFACE.md)"
                .to_owned(),
        );
    }
    Ok(Value::Object(body))
}

/// Runs `rv patch`.
#[must_use]
pub fn patch(ctx: &Ctx, a: &PatchArgs) -> i32 {
    if !ctx.require_admin_key() {
        return 1;
    }
    let st = ctx.style;
    // Everything that needs no request first: conflicts, ranges, an empty patch.
    let precheck = if a.touches_sampling() {
        patch_body(a, Some(&Value::Null))
    } else {
        patch_body(a, None)
    };
    if let Err(msg) = precheck {
        eprintln!("{}", st.red(&msg));
        return 1;
    }
    let merged = if a.touches_sampling() {
        match sampling_for_patch(ctx, a) {
            Ok(m) => Some(m),
            Err(code) => return code,
        }
    } else {
        None
    };
    let body = match patch_body(a, merged.as_ref().map(|(_, _, v)| v)) {
        Ok(b) => b,
        Err(msg) => {
            eprintln!("{}", st.red(&msg));
            return 1;
        }
    };
    let r = ctx.api.patch(
        &format!("/v1/admin/models/{}", encode_segment(&a.model)),
        &body,
    );
    if let Some(resp) = r.json().filter(|v| r.code == Some(200) && v.is_object()) {
        if ctx.json {
            println!("{}", serde_json::to_string_pretty(resp).unwrap_or_default());
            return 0;
        }
        let (applied, next_load, unchanged) = (
            strings(resp, "applied_live"),
            strings(resp, "effective_on_next_load"),
            strings(resp, "unchanged"),
        );
        let mut parts = vec![format!("{} patched '{}'", st.ok(), a.model)];
        if !applied.is_empty() {
            parts.push(format!(" -- live now: {}", applied.join(", ")));
        }
        if !next_load.is_empty() {
            parts.push(format!("; next load: {}", next_load.join(", ")));
        }
        if !unchanged.is_empty() {
            parts.push(format!("; unchanged: {}", unchanged.join(", ")));
        }
        let msg = parts.concat();
        println!("{msg}");
        if let Some((before, after, _)) = &merged {
            let before = before.as_ref().map_or_else(
                || "(not reported for this kind)".to_owned(),
                Operator::summary,
            );
            println!("  sampling: {before} → {}", after.summary());
        }
        return 0;
    }
    print_error(ctx, &r);
    1
}

/// `rv patch` with sampling flags: GET the model, refuse value flags on a non-text kind or a
/// server that would drop `sampling`, then merge the flags over the operator values it reports
/// (all are resent). `--clear-sampling` passes on a non-text kind (the server stores `sampling`
/// for any kind, so an add that auto-detected a non-text kind must be undoable — review F1); its
/// `before` is then unknown (`None`). `Err(code)` after printing why; nothing has been sent then.
fn sampling_for_patch(ctx: &Ctx, a: &PatchArgs) -> Result<Merged, i32> {
    let st = ctx.style;
    let entry = admin_entry(ctx, &a.model).map_err(|(code, _)| code)?;
    match entry_profile(&entry) {
        EntryProfile::NotApplicable if a.clear_sampling => {
            Ok((None, Operator::default(), Value::Null))
        }
        EntryProfile::NotApplicable => {
            eprintln!(
                "{}",
                st.red(&format!(
                    "error: '{}' is a model of kind {} — sampling settings apply to text_gen \
                     and vision models only; nothing was sent",
                    a.model,
                    entry["kind"].as_str().unwrap_or("non-text")
                ))
            );
            Err(1)
        }
        EntryProfile::OldServer => {
            eprintln!(
                "{}",
                st.red(&format!(
                    "{} {OLD_SERVER_WARNING} — nothing was sent",
                    st.warn()
                ))
            );
            Err(1)
        }
        EntryProfile::Profile(current) => merge_for_patch(current, &a.sampling, a.clear_sampling)
            .map(|(before, after, v)| (Some(before), after, v))
            .map_err(|e| {
                eprintln!("{}", st.red(&e));
                1
            }),
    }
}

/// `(before, after, the sampling value sent)` of a patch that touches sampling.
type Merged = (Option<Operator>, Operator, Value);

/// Flags of `rv preload`.
#[allow(clippy::doc_markdown)]
#[derive(clap::Args, Debug, Clone)]
pub struct PreloadArgs {
    /// explicit model ids (mutually exclusive with --from-live)
    pub models: Vec<String>,
    /// snapshot whichever models are currently loaded as the new list
    #[arg(long)]
    pub from_live: bool,
    /// don't prompt for confirmation
    #[arg(short = 'y', long)]
    pub yes: bool,
}

/// Runs `rv preload`: replace the persisted preload list. Takes effect on the
/// next restart and loads/evicts nothing now.
#[must_use]
pub fn preload(ctx: &Ctx, a: &PreloadArgs) -> i32 {
    if !ctx.require_admin_key() {
        return 1;
    }
    let st = ctx.style;
    // `models` is an empty Vec (not absent) on a bare `rv preload`, so test
    // emptiness, not presence — the Python version once wiped the preload list
    // on a bare `rv preload` by checking identity instead.
    let body = if a.from_live && !a.models.is_empty() {
        eprintln!(
            "{}",
            st.red("error: --from-live and an explicit model list are mutually exclusive")
        );
        return 1;
    } else if a.from_live {
        json!({"from_live": true})
    } else if !a.models.is_empty() {
        json!({"model_ids": a.models})
    } else {
        eprintln!(
            "{}",
            st.red("error: pass either --from-live or a list of model ids")
        );
        return 1;
    };

    if !a.yes {
        let desc = if a.from_live {
            "the currently-loaded set".to_owned()
        } else {
            a.models.join(", ")
        };
        if !confirm(
            st,
            &format!(
                "replace the persisted preload list with {desc} (takes effect on next restart, \
                 changes nothing running now). Proceed?"
            ),
            a.yes,
        ) {
            println!("aborted");
            return 2;
        }
    }

    let r = ctx
        .api
        .post("/v1/admin/config/preload", Some(&body), READ_TIMEOUT);
    if let Some(resp) = r.json().filter(|v| r.code == Some(200) && v.is_object()) {
        if ctx.json {
            println!("{}", serde_json::to_string_pretty(resp).unwrap_or_default());
            return 0;
        }
        let list = strings(resp, "preload");
        println!(
            "{} preload is now: {}",
            st.ok(),
            if list.is_empty() {
                "(empty)".to_owned()
            } else {
                list.join(", ")
            }
        );
        let skipped = strings(resp, "skipped_not_ready");
        if !skipped.is_empty() {
            println!(
                "{}",
                st.dim(&format!(
                    "not currently loaded, excluded: {}",
                    skipped.join(", ")
                ))
            );
        }
        return 0;
    }
    print_error(ctx, &r);
    1
}

/// Runs `rv save-defaults`: a named shortcut for `rv preload --from-live`.
#[must_use]
pub fn save_defaults(ctx: &Ctx, yes: bool) -> i32 {
    preload(
        ctx,
        &PreloadArgs {
            models: Vec::new(),
            from_live: true,
            yes,
        },
    )
}

#[cfg(test)]
#[allow(clippy::expect_used)] // fixtures: a failed setup should fail the test loudly
mod tests {
    use super::*;

    fn args() -> PatchArgs {
        PatchArgs {
            model: "m".into(),
            vram_gb: None,
            pinned: None,
            priority: None,
            load: None,
            device: None,
            clear_device: false,
            eviction_grace_secs: None,
            clear_eviction_grace_secs: false,
            thinking_budget_tokens: None,
            clear_thinking_budget_tokens: false,
            sampling: SamplingFlags::default(),
            clear_sampling: false,
        }
    }

    /// `--thinking-budget-tokens N` sets it, `--clear-thinking-budget-tokens` sends `null`, both
    /// together are refused.
    #[test]
    fn patch_body_thinking_budget_tokens() {
        let mut a = args();
        a.thinking_budget_tokens = Some(2048);
        assert_eq!(
            patch_body(&a, None).expect("body"),
            json!({"thinking_budget_tokens": 2048})
        );
        let mut a = args();
        a.clear_thinking_budget_tokens = true;
        assert_eq!(
            patch_body(&a, None).expect("body"),
            json!({"thinking_budget_tokens": null})
        );
        a.thinking_budget_tokens = Some(0);
        let e = patch_body(&a, None).expect_err("conflict");
        assert!(e.contains("mutually exclusive"), "{e}");
    }

    /// C8: the reload response's `models_not_applied` becomes `(model, fields)`; an older
    /// server's body (no key) gives none.
    #[test]
    fn models_not_applied_reads_the_reload_response() {
        let body = serde_json::json!({
            "models_not_applied": [{"model": "lfm2", "fields": ["chat_template", "pinned"]}]
        });
        assert_eq!(
            models_not_applied(&body),
            vec![(
                "lfm2".to_owned(),
                vec!["chat_template".to_owned(), "pinned".to_owned()]
            )]
        );
        assert!(models_not_applied(&serde_json::json!({"added": []})).is_empty());
    }

    #[test]
    fn applied_live_reads_the_reload_response() {
        let body = serde_json::json!({
            "applied_live": [{"model": "mistral", "fields": ["sampling"]}],
            "models_not_applied": []
        });
        assert_eq!(
            applied_live_lines(&body),
            vec!["mistral: sampling applied live"]
        );
        assert!(applied_live_lines(&serde_json::json!({"added": []})).is_empty());
    }

    #[test]
    fn patch_with_no_fields_is_refused() {
        assert!(patch_body(&args(), None).is_err());
    }

    #[test]
    fn patch_conflicting_flags_are_refused() {
        let mut a = args();
        a.device = Some("GPU".into());
        a.clear_device = true;
        assert!(patch_body(&a, None).is_err());
        let mut a = args();
        a.eviction_grace_secs = Some(5.0);
        a.clear_eviction_grace_secs = true;
        assert!(patch_body(&a, None).is_err());
    }

    #[test]
    fn patch_body_carries_exactly_the_given_fields() {
        let mut a = args();
        a.pinned = Some("true".into());
        a.priority = Some(-2);
        a.load = Some("on_demand".into());
        a.clear_device = true;
        a.clear_eviction_grace_secs = true;
        assert_eq!(
            patch_body(&a, None).expect("body"),
            json!({"pinned": true, "priority": -2, "load": "on_demand",
                   "device": null, "eviction_grace_secs": null})
        );
        let mut a = args();
        a.pinned = Some("false".into());
        a.vram_gb = Some(3.5);
        assert_eq!(
            patch_body(&a, None).expect("body"),
            json!({"vram_gb": 3.5, "pinned": false})
        );
    }

    #[test]
    fn sampling_flags_conflict_and_ranges_before_any_request() {
        let mut a = args();
        a.sampling.temperature = Some(0.2);
        a.clear_sampling = true;
        let e = patch_body(&a, Some(&Value::Null)).expect_err("conflict");
        assert!(e.contains("mutually exclusive"), "{e}");
        let mut a = args();
        a.sampling.top_p = Some(1.5);
        let e = patch_body(&a, Some(&Value::Null)).expect_err("range");
        assert!(e.contains("--top-p"), "{e}");
        let mut a = args();
        a.sampling.presence_penalty = Some(-2.5);
        assert!(patch_body(&a, Some(&Value::Null)).is_err());
    }

    #[test]
    fn sampling_body_carries_the_merged_object_or_null() {
        let mut a = args();
        a.sampling.temperature = Some(0.2);
        a.pinned = Some("true".into());
        let merged = json!({"temperature": 0.2, "top_p": 0.8});
        assert_eq!(
            patch_body(&a, Some(&merged)).expect("body"),
            json!({"pinned": true, "sampling": {"temperature": 0.2, "top_p": 0.8}})
        );
        let mut a = args();
        a.clear_sampling = true;
        assert!(a.touches_sampling());
        assert_eq!(
            patch_body(&a, Some(&Value::Null)).expect("body"),
            json!({"sampling": null})
        );
        assert!(!args().touches_sampling());
    }
}
