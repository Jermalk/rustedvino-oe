//! `rv list` and `rv ps`.

use serde_json::Value;

use crate::ctx::Ctx;
use crate::format::table;
use crate::metrics::{self, Sample};
use crate::order;

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

/// The context-limit cell: `max_prompt_tokens` as a plain number below 10,000,
/// else rounded thousands with `k`. `*` marks a limit that follows free device
/// memory (`device_headroom`), `~` an estimate (legacy gate), and a trailing `!` a vision
/// model that has admitted image requests whose images the limit did not count
/// (`kv_admission_low_confidence`, F2-S-5). `-` when unknown.
pub(crate) fn ctx_cell(m: &Value) -> String {
    let n = m
        .get("max_prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if n == 0 {
        return "-".to_owned();
    }
    let mut out = if n < 10_000 {
        n.to_string()
    } else {
        format!("{}k", (n + 500) / 1000)
    };
    if m.get("limit_clamped_by").and_then(Value::as_str) == Some("device_headroom") {
        out.push('*');
    } else if m.get("kv_gate_mode").and_then(Value::as_str) == Some("legacy") {
        out.push('~');
    }
    if m.get("kv_admission_low_confidence")
        .and_then(Value::as_bool)
        == Some(true)
    {
        out.push('!');
    }
    out
}

/// The KV pool cell: GiB with the whole MiB the server booked (`1.21 GiB (1239 MiB)`), from the
/// admin `kv_pool_mib`; the old one-decimal `cache_size_gb` figure when the server does not report
/// it, and `-` for a model with no pool.
pub(crate) fn kv_pool_cell(m: &Value) -> String {
    match m.get("kv_pool_mib").and_then(Value::as_u64) {
        Some(0) => "-".to_owned(),
        #[allow(clippy::cast_precision_loss)] // a pool is far below 2^52 MiB
        Some(mib) => format!("{:.2} GiB ({mib} MiB)", mib as f64 / 1024.0),
        None => gb(m, "cache_size_gb"),
    }
}

/// The KV-cache precision cell, e.g. `u4 (auto)`; `-` when not reported.
pub(crate) fn kv_prec_cell(m: &Value) -> String {
    let Some(prec) = m.get("kv_cache_precision").and_then(Value::as_str) else {
        return "-".to_owned();
    };
    let tag = match m.get("kv_precision_source").and_then(Value::as_str) {
        Some("auto_rule") => " (auto)",
        Some("global") => " (global)",
        Some("per_model") => " (model)",
        Some("ir_hint") => " (ir)",
        _ => "",
    };
    format!("{prec}{tag}")
}

/// `running/waiting/max` on the model's current `device`, or `n/a` unless all three metrics exist.
fn req_triplet(rows: &[Sample], model: &str, device: Option<&str>) -> String {
    let get = |name| metrics::by_model_on_device(rows, name, model, device);
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

/// The KV% cell. `n/a` when the percent metric is missing, and also when the
/// server says the reading is not measurable for this model
/// (`rustedvino_kv_cache_usage_supported == 0`, e.g. the VLM engine): that
/// percent is a permanent `0` that cannot be told from an empty pool (C21). A
/// server without the `supported` series keeps the plain reading. Both read the model's current
/// `device` ([`metrics::by_model_on_device`]).
fn kv_percent(rows: &[Sample], model: &str, device: Option<&str>) -> String {
    let get = |name| metrics::by_model_on_device(rows, name, model, device);
    // A 0/1 gauge: compare against the midpoint, not with float equality.
    if get("rustedvino_kv_cache_usage_supported").is_some_and(|v| v < 0.5) {
        return "n/a".into();
    }
    get("rustedvino_kv_cache_usage_percent").map_or_else(|| "n/a".into(), |kv| format!("{kv:.0}%"))
}

/// The device a model's metric cells read: its admin-list `device`, else the device of its only
/// `rustedvino_model_loaded == 1` series (as `rv metrics` does), else `None`.
fn cell_device(rows: &[Sample], model: &str, admin_device: Option<&str>) -> Option<String> {
    admin_device
        .filter(|d| !d.is_empty() && *d != "?")
        .map(str::to_owned)
        .or_else(|| crate::commands::metrics::sole_loaded_device(rows, model))
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
        let mut sorted: Vec<&Value> = reg.iter().collect();
        order::sort_values(&mut sorted);
        let rows: Vec<Vec<String>> = sorted
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
            let mut unregistered: Vec<&String> = dirs
                .iter()
                .filter(|d| !reg_ids.contains(&d.as_str()))
                .collect();
            unregistered.sort_by(|a, b| order::cmp_ids(a, b));
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
            let mut sorted: Vec<&String> = dirs.iter().collect();
            sorted.sort_by(|a, b| order::cmp_ids(a, b));
            for d in sorted {
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
    let mut ready: Vec<Value> = all
        .into_iter()
        // A GPU model on a poisoned context ("unavailable (gpu_poisoned)") is still
        // resident in VRAM, so it stays in the table; the state column says why.
        .filter(|m| s(m, "state") == "ready" || s(m, "state").starts_with("unavailable"))
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
    // After the `--json` print: that keeps the server's order.
    order::sort_models(&mut ready, |m| {
        (
            m.get("kind").and_then(Value::as_str),
            m.get("id").and_then(Value::as_str).unwrap_or(""),
        )
    });
    let rows: Vec<Vec<String>> = ready
        .iter()
        .map(|m| {
            let id = s(m, "id");
            let (kv, req) = load_cells(&rows_m, id, Some(s(m, "device")));
            vec![
                id.to_owned(),
                s(m, "device").to_owned(),
                s(m, "kind").to_owned(),
                mem_cell(s(m, "device"), gb(m, "vram_gb")),
                kv_pool_cell(m),
                ctx_cell(m),
                kv_prec_cell(m),
                kv,
                req,
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
                "KV POOL",
                "CTX",
                "KV PREC",
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

/// Per-model request/KV cells shared with `status`, read on the model's current device:
/// `admin_device` (the admin list's `device`; `None` or `?` when unknown), else the device of its
/// only loaded series.
#[must_use]
pub fn load_cells(rows: &[Sample], model: &str, admin_device: Option<&str>) -> (String, String) {
    let device = cell_device(rows, model, admin_device);
    (
        kv_percent(rows, model, device.as_deref()),
        req_triplet(rows, model, device.as_deref()),
    )
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

#[cfg(test)]
mod kv_cell_tests {
    use super::{ctx_cell, kv_percent, kv_pool_cell, kv_prec_cell, load_cells};
    use crate::metrics;
    use serde_json::json;

    /// C21: an unmeasurable pool reads `n/a`, not `0%`; a measurable one keeps
    /// its percentage; a server without the `supported` series keeps the reading.
    #[test]
    fn kv_percent_is_na_when_unsupported() {
        let rows = metrics::parse(
            "rustedvino_kv_cache_usage_percent{model=\"vlm\",device=\"GPU\",kind=\"vlm\"} 0\n\
             rustedvino_kv_cache_usage_supported{model=\"vlm\",device=\"GPU\",kind=\"vlm\"} 0\n\
             rustedvino_kv_cache_usage_percent{model=\"cb\",device=\"GPU\",kind=\"text_gen\"} 23.8\n\
             rustedvino_kv_cache_usage_supported{model=\"cb\",device=\"GPU\",kind=\"text_gen\"} 1\n\
             rustedvino_kv_cache_usage_percent{model=\"old\"} 7\n",
        );
        assert_eq!(kv_percent(&rows, "vlm", None), "n/a");
        assert_eq!(kv_percent(&rows, "cb", None), "24%");
        assert_eq!(kv_percent(&rows, "cb", Some("GPU")), "24%");
        assert_eq!(kv_percent(&rows, "old", None), "7%");
        assert_eq!(kv_percent(&rows, "old", Some("GPU")), "7%");
        assert_eq!(kv_percent(&rows, "absent", None), "n/a");
    }

    /// A model that moved CPU → `GPU.0`: the server keeps the CPU series (listed first here, so a
    /// first-match read would show it). Round LFO P4.
    const MOVED: &str = "\
rustedvino_model_loaded{model=\"m\",device=\"CPU\",kind=\"text_gen\"} 0\n\
rustedvino_model_loaded{model=\"m\",device=\"GPU.0\",kind=\"text_gen\"} 1\n\
rustedvino_kv_cache_usage_percent{model=\"m\",device=\"CPU\",kind=\"text_gen\"} 81\n\
rustedvino_kv_cache_usage_supported{model=\"m\",device=\"CPU\",kind=\"text_gen\"} 1\n\
rustedvino_kv_cache_usage_percent{model=\"m\",device=\"GPU.0\",kind=\"text_gen\"} 12\n\
rustedvino_kv_cache_usage_supported{model=\"m\",device=\"GPU.0\",kind=\"text_gen\"} 1\n\
rustedvino_requests_running{model=\"m\",device=\"CPU\",kind=\"text_gen\"} 0\n\
rustedvino_requests_waiting{model=\"m\",device=\"CPU\",kind=\"text_gen\"} 0\n\
rustedvino_requests_max{model=\"m\",device=\"CPU\",kind=\"text_gen\"} 4\n\
rustedvino_requests_running{model=\"m\",device=\"GPU.0\",kind=\"text_gen\"} 3\n\
rustedvino_requests_waiting{model=\"m\",device=\"GPU.0\",kind=\"text_gen\"} 1\n\
rustedvino_requests_max{model=\"m\",device=\"GPU.0\",kind=\"text_gen\"} 16\n";

    #[test]
    fn two_devices_the_current_device_picks_the_cells() {
        let rows = metrics::parse(MOVED);
        assert_eq!(
            load_cells(&rows, "m", Some("GPU.0")),
            ("12%".to_owned(), "3/1/16".to_owned())
        );
        assert_eq!(
            load_cells(&rows, "m", Some("CPU")),
            ("81%".to_owned(), "0/0/4".to_owned())
        );
        // No admin device: the only `model_loaded == 1` series names it.
        assert_eq!(
            load_cells(&rows, "m", None),
            ("12%".to_owned(), "3/1/16".to_owned())
        );
        assert_eq!(
            load_cells(&rows, "m", Some("?")),
            ("12%".to_owned(), "3/1/16".to_owned())
        );
    }

    #[test]
    fn two_devices_without_a_series_on_the_current_one_read_na() {
        let rows = metrics::parse(MOVED);
        // Just moved to GPU.1, nothing exported there yet: not the old devices' figures.
        assert_eq!(
            load_cells(&rows, "m", Some("GPU.1")),
            ("n/a".to_owned(), "n/a".to_owned())
        );
        // Unknown current device and two loaded series: ambiguous, so `n/a`, never one at random.
        let both = metrics::parse(&MOVED.replace(
            "model_loaded{model=\"m\",device=\"CPU\",kind=\"text_gen\"} 0",
            "model_loaded{model=\"m\",device=\"CPU\",kind=\"text_gen\"} 1",
        ));
        assert_eq!(
            load_cells(&both, "m", None),
            ("n/a".to_owned(), "n/a".to_owned())
        );
    }

    #[test]
    fn kv_pool_shows_gib_and_mib() {
        assert_eq!(
            kv_pool_cell(&json!({"kv_pool_mib": 1239})),
            "1.21 GiB (1239 MiB)"
        );
        assert_eq!(kv_pool_cell(&json!({"kv_pool_mib": 0})), "-");
        assert_eq!(kv_pool_cell(&json!({"cache_size_gb": 2.0})), "2.0 GB");
    }

    #[test]
    fn ctx_formats() {
        assert_eq!(ctx_cell(&json!({})), "-");
        assert_eq!(ctx_cell(&json!({"max_prompt_tokens": 0})), "-");
        assert_eq!(ctx_cell(&json!({"max_prompt_tokens": 4096})), "4096");
        assert_eq!(ctx_cell(&json!({"max_prompt_tokens": 10000})), "10k");
        assert_eq!(ctx_cell(&json!({"max_prompt_tokens": 193_116})), "193k");
        assert_eq!(ctx_cell(&json!({"max_prompt_tokens": 86381})), "86k");
        // F2-S-5: images admitted uncounted mark the cell, after the headroom mark.
        assert_eq!(
            ctx_cell(
                &json!({"max_prompt_tokens": 86381, "limit_clamped_by": "device_headroom",
                             "kv_admission_low_confidence": true})
            ),
            "86k*!"
        );
        assert_eq!(ctx_cell(&json!({"max_prompt_tokens": 104_500})), "105k");
    }

    #[test]
    fn ctx_flags() {
        let hr = json!({"max_prompt_tokens": 104_500, "limit_clamped_by": "device_headroom"});
        assert_eq!(ctx_cell(&hr), "105k*");
        let lg = json!({"max_prompt_tokens": 27_000, "kv_gate_mode": "legacy"});
        assert_eq!(ctx_cell(&lg), "27k~");
        let pool = json!({"max_prompt_tokens": 27_000, "limit_clamped_by": "pool", "kv_gate_mode": "exact"});
        assert_eq!(ctx_cell(&pool), "27k");
    }

    #[test]
    fn kv_prec_formats() {
        let c = |p: &str, s: &str| {
            kv_prec_cell(&json!({"kv_cache_precision": p, "kv_precision_source": s}))
        };
        assert_eq!(c("u4", "auto_rule"), "u4 (auto)");
        assert_eq!(c("i8", "global"), "i8 (global)");
        assert_eq!(c("f16", "ir_hint"), "f16 (ir)");
        assert_eq!(c("u4", "per_model"), "u4 (model)");
        assert_eq!(c("auto", "passthrough"), "auto");
        assert_eq!(kv_prec_cell(&json!({"kv_cache_precision": "i8"})), "i8");
        assert_eq!(kv_prec_cell(&json!({})), "-");
    }
}
