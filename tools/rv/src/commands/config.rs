//! `rv reload-config`, `patch`, `preload`, `save-defaults`: verbs that change
//! configuration rather than what is loaded.

use serde_json::{Map, Value, json};

use crate::api::{READ_TIMEOUT, Reply, encode_segment};
use crate::commands::print_error;
use crate::commands::prompt::confirm;
use crate::ctx::Ctx;

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
    0
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
#[allow(clippy::doc_markdown)]
#[derive(clap::Args, Debug, Clone)]
pub struct PatchArgs {
    /// model id
    pub model: String,
    /// VRAM estimate in GB
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
    /// clear the device override back to the global config default
    #[arg(long)]
    pub clear_device: bool,
    /// eviction grace window in seconds
    #[arg(long)]
    pub eviction_grace_secs: Option<f64>,
    /// clear the override back to the global eviction_grace_secs default
    #[arg(long)]
    pub clear_eviction_grace_secs: bool,
}

/// Builds the `PATCH` body, or the reason it cannot be built. Pure, so the
/// flag combinations are unit-testable.
///
/// # Errors
///
/// A conflicting flag pair, or no field at all.
pub fn patch_body(a: &PatchArgs) -> Result<Value, &'static str> {
    if a.clear_device && a.device.is_some() {
        return Err("error: --device and --clear-device are mutually exclusive");
    }
    if a.clear_eviction_grace_secs && a.eviction_grace_secs.is_some() {
        return Err(
            "error: --eviction-grace-secs and --clear-eviction-grace-secs are mutually exclusive",
        );
    }
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
    if body.is_empty() {
        return Err(
            "error: no fields supplied -- pass at least one of --vram-gb/--pinned/--priority/\
             --load/--device/--clear-device/--eviction-grace-secs/--clear-eviction-grace-secs \
             (anything else -- kv_cache_gb, tier_preference, capabilities, etc. -- needs a raw \
             curl PATCH; see INTERFACE.md)",
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
    let body = match patch_body(a) {
        Ok(b) => b,
        Err(msg) => {
            eprintln!("{}", st.red(msg));
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
        return 0;
    }
    print_error(ctx, &r);
    1
}

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
        }
    }

    #[test]
    fn patch_with_no_fields_is_refused() {
        assert!(patch_body(&args()).is_err());
    }

    #[test]
    fn patch_conflicting_flags_are_refused() {
        let mut a = args();
        a.device = Some("GPU".into());
        a.clear_device = true;
        assert!(patch_body(&a).is_err());
        let mut a = args();
        a.eviction_grace_secs = Some(5.0);
        a.clear_eviction_grace_secs = true;
        assert!(patch_body(&a).is_err());
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
            patch_body(&a).expect("body"),
            json!({"pinned": true, "priority": -2, "load": "on_demand",
                   "device": null, "eviction_grace_secs": null})
        );
        let mut a = args();
        a.pinned = Some("false".into());
        a.vram_gb = Some(3.5);
        assert_eq!(
            patch_body(&a).expect("body"),
            json!({"vram_gb": 3.5, "pinned": false})
        );
    }
}
