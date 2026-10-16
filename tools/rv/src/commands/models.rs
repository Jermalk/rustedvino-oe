//! `rv list` and `rv ps`.

use serde_json::Value;

use crate::ctx::Ctx;
use crate::format::table;
use crate::metrics::{self, Sample};

/// Raw fetch of the registry. No "missing key" messaging: that is the caller's
/// job (`list` and `ps` word the same condition differently). `None` silently
/// when there is no key to try; prints and returns `None` when a key was sent
/// but the call still failed, since that is a real, distinct error.
fn fetch_admin_models(ctx: &Ctx) -> Option<Vec<Value>> {
    if !ctx.res.admin_usable() {
        return None;
    }
    let r = ctx.api.get("/v1/admin/models");
    if let Some(models) = r
        .json()
        .filter(|_| r.code == Some(200))
        .and_then(|b| b.get("models"))
        .and_then(Value::as_array)
    {
        if r.json()
            .is_some_and(|b| b["server"]["gpu_poisoned"].as_bool().unwrap_or(false))
        {
            eprintln!(
                "{}",
                ctx.style.red(
                    "GPU CONTEXT POISONED — an earlier out-of-resources failure wedged the GPU; \
                     every GPU model below is unusable and loads/unloads are refused. \
                     Restart the server (graceful stop) to recover."
                )
            );
        }
        return Some(models.clone());
    }
    eprintln!(
        "{}",
        ctx.style.red(&format!(
            "error: could not reach /v1/admin/models — {}",
            r.err.unwrap_or_else(|| crate::commands::code_text(r.code))
        ))
    );
    None
}

fn s<'a>(m: &'a Value, k: &str) -> &'a str {
    m.get(k).and_then(Value::as_str).unwrap_or("?")
}

fn gb(m: &Value, k: &str) -> String {
    format!("{:.1} GB", m.get(k).and_then(Value::as_f64).unwrap_or(0.0))
}

/// The memory-budget cell: on a CPU placement the figure is a system-RAM
/// budget, not VRAM, so say so (0.8.3, F6).
pub(crate) fn mem_cell(device: &str, gb_text: String) -> String {
    if device.eq_ignore_ascii_case("cpu") {
        format!("{gb_text} (RAM)")
    } else {
        gb_text
    }
}

/// `running/waiting/max`, or `n/a` unless all three metrics exist.
fn req_triplet(rows: &[Sample], model: &str) -> String {
    let get = |name| metrics::by_label(rows, name, &[("model", model)]);
    match (
        get("rustedvino_requests_running"),
        get("rustedvino_requests_waiting"),
        get("rustedvino_requests_max"),
    ) {
        #[allow(clippy::cast_possible_truncation)]
        (Some(r), Some(w), Some(m)) => format!("{}/{}/{}", r as i64, w as i64, m as i64),
        _ => "n/a".into(),
    }
}

fn kv_percent(rows: &[Sample], model: &str) -> String {
    metrics::by_label(
        rows,
        "rustedvino_kv_cache_usage_percent",
        &[("model", model)],
    )
    .map_or_else(|| "n/a".into(), |kv| format!("{kv:.0}%"))
}

/// Runs `rv list`: the registry (needs an admin key) beside a local scan of
/// `models_dir` (needs none). Neither alone answers "what is available".
#[must_use]
pub fn list(ctx: &Ctx) -> i32 {
    ctx.warn_if_open();
    let registered = fetch_admin_models(ctx);
    let disk_dirs = ctx.list_model_dirs();
    let st = ctx.style;

    if ctx.json {
        let v = serde_json::json!({
            "registered": registered,
            "on_disk": disk_dirs,
        });
        println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
        return i32::from(registered.is_none() && disk_dirs.is_none());
    }

    let reg_ids: Vec<&str> = registered
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|m| s(m, "id"))
        .collect();
    if let Some(reg) = &registered {
        let rows: Vec<Vec<String>> = reg
            .iter()
            .map(|m| {
                vec![
                    s(m, "id").to_owned(),
                    s(m, "state").to_owned(),
                    s(m, "device").to_owned(),
                    mem_cell(s(m, "device"), gb(m, "vram_gb")),
                    s(m, "kind").to_owned(),
                    ctx.policy_flags(s(m, "id")),
                ]
            })
            .collect();
        println!("{}", st.bold(&format!("{} model(s) registered", reg.len())));
        print!(
            "{}",
            table(
                st,
                &["ID", "STATE", "DEVICE", "VRAM", "KIND", "POLICY"],
                &rows,
                Some(1)
            )
        );
    } else {
        println!(
            "{} registered-model state unavailable (no admin key) — showing on-disk directories \
             only, load state unknown for all of them",
            st.warn()
        );
    }

    println!();
    let models_dir = ctx.res.models_dir.as_deref().unwrap_or("");
    if let Some(dirs) = &disk_dirs {
        if registered.is_some() {
            let unregistered: Vec<&String> = dirs
                .iter()
                .filter(|d| !reg_ids.contains(&d.as_str()))
                .collect();
            println!(
                "{}{}",
                st.bold(&format!(
                    "{} more on disk, not registered",
                    unregistered.len()
                )),
                st.dim(&format!("  ({models_dir})"))
            );
            for d in &unregistered {
                println!("  {d}");
            }
            if unregistered.is_empty() {
                println!(
                    "{}",
                    st.dim("  (every directory under models_dir is already registered)")
                );
            }
        } else {
            let noun = if dirs.len() == 1 { "y" } else { "ies" };
            println!(
                "{}{}",
                st.bold(&format!("{} model director{noun} on disk", dirs.len())),
                st.dim(&format!("  ({models_dir})"))
            );
            for d in dirs {
                println!("  {d}");
            }
        }
    } else {
        println!(
            "{} models_dir unknown or unreadable — couldn't read 'models_dir' from {}, or the \
             path doesn't exist",
            st.warn(),
            ctx.res.config_file.display()
        );
    }
    i32::from(registered.is_none() && disk_dirs.is_none())
}

/// Runs `rv ps`: loaded (`ready`) models with live KV and request load.
#[must_use]
pub fn ps(ctx: &Ctx) -> i32 {
    if !ctx.require_admin_key() {
        return 1;
    }
    let Some(all) = fetch_admin_models(ctx) else {
        return 1;
    };
    let ready: Vec<Value> = all
        .into_iter()
        .filter(|m| s(m, "state") == "ready")
        .collect();
    let rows_m = ctx
        .api
        .get("/metrics")
        .body
        .and_then(|b| b.text().map(metrics::parse))
        .unwrap_or_default();
    if ctx.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&ready).unwrap_or_default()
        );
        return 0;
    }
    let rows: Vec<Vec<String>> = ready
        .iter()
        .map(|m| {
            let id = s(m, "id");
            vec![
                id.to_owned(),
                s(m, "device").to_owned(),
                s(m, "kind").to_owned(),
                mem_cell(s(m, "device"), gb(m, "vram_gb")),
                gb(m, "cache_size_gb"),
                kv_percent(&rows_m, id),
                req_triplet(&rows_m, id),
                ctx.policy_flags(id),
            ]
        })
        .collect();
    println!(
        "{}",
        ctx.style.bold(&format!("{} model(s) loaded", ready.len()))
    );
    print!(
        "{}",
        table(
            ctx.style,
            &[
                "ID",
                "DEVICE",
                "KIND",
                "VRAM",
                "KV GB",
                "KV%",
                "RUN/WAIT/MAX",
                "POLICY"
            ],
            &rows,
            None
        )
    );
    0
}

/// Per-model request/KV cells shared with `status`.
#[must_use]
pub fn load_cells(rows: &[Sample], model: &str) -> (String, String) {
    (kv_percent(rows, model), req_triplet(rows, model))
}

#[cfg(test)]
mod mem_cell_tests {
    use super::mem_cell;

    #[test]
    fn cpu_budget_is_labelled_ram() {
        assert_eq!(mem_cell("CPU", "3.3 GB".into()), "3.3 GB (RAM)");
        assert_eq!(mem_cell("GPU.1", "3.3 GB".into()), "3.3 GB");
    }
}
