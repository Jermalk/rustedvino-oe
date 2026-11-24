//! `rv status` — the full server report.

use std::collections::BTreeSet;

use serde_json::{Value, json};

use crate::commands::models::load_cells;
use crate::ctx::Ctx;
use crate::format::{bar, fmt_duration, human_bytes, table};
use crate::metrics::{self, Sample};
use crate::proc::{self, ProcState, Source};

/// The Auth row. Reports each route class independently and states ONLY what
/// was observed: an undetermined route is never described as enforced (saying
/// "admin enforced" on the strength of a 503 would be an unearned claim). The
/// local keys file appears only as supporting detail, never as evidence about
/// the server.
fn auth_line(ctx: &Ctx, inf: Option<bool>, adm: Option<bool>) -> String {
    let st = ctx.style;
    let label = format!("{:<10}", "Auth");
    let r = &ctx.res;
    let local = if r.keys_present {
        format!(
            "{} inference · {} admin key(s) in {}",
            r.api_key_count,
            r.admin_key_count,
            r.keys_file.display()
        )
    } else {
        format!("no keys file at {}", r.keys_file.display())
    };
    let phrase = |name: &str, state: Option<bool>| match state {
        Some(true) => format!("{name} enforced"),
        Some(false) => format!("{name} OPEN"),
        None => format!("{name} undetermined"),
    };
    let summary = format!("{}, {}", phrase("inference", inf), phrase("admin", adm));
    // The glyph tracks the worst OBSERVED state: an open route is bad, an
    // undetermined one merely unknown, and "no key here" is an operator
    // problem rather than a server problem.
    let glyph = if inf == Some(false) || adm == Some(false) {
        st.bad()
    } else if inf.is_none() || adm.is_none() {
        st.warn()
    } else if !r.admin_usable() {
        return format!(
            "{} {label}{summary} — but no admin key available here; set RV_ADMIN_KEY or \
             --admin-key; {}",
            st.warn(),
            st.dim(&local)
        );
    } else {
        st.ok()
    };
    format!("{glyph} {label}{summary}; {}", st.dim(&local))
}

fn process_line(ctx: &Ctx, p: &ProcState, health_ok: bool) -> String {
    let st = ctx.style;
    let label = format!("{:<10}", "Process");
    let port = ctx.res.port;
    if !p.alive {
        if health_ok {
            return format!(
                "{} {label} running — PID unknown  {}",
                st.warn(),
                st.dim("(the server did not report it — needs an admin key or a newer server — and no OS probe found it)")
            );
        }
        let why = if cfg!(target_os = "linux") {
            format!(
                "(nothing holds :{port}, no live PID at {}, no rustedvino process found)",
                ctx.pid_file.display()
            )
        } else {
            format!("(nothing is answering on :{port})")
        };
        return format!("{} {label} not running  {}", st.bad(), st.dim(&why));
    }
    let up = p.uptime_secs.map_or_else(|| "unknown".into(), fmt_duration);
    let pf = ctx.pid_file.display();
    let note = match p.source {
        // The server's uptime is its own (the worker's). Under a supervisor that
        // differs from the supervisor's, which the OS probes report — so say
        // which one this is.
        Source::Server if p.worker_pid.is_some() => {
            format!("(worker up {up}; reported by the server itself)")
        }
        Source::Server => format!("(up {up}; reported by the server itself)"),
        // Confirmed: this process holds the port, so it IS the server the
        // rest of this report describes.
        Source::Port => format!("(up {up}; holds :{port})"),
        // Found by command line, not from the PID file — the stale file is a
        // real (if minor) inconsistency worth repairing.
        Source::Scan => {
            format!(
                "(up {up}; found by scan — {pf} is stale/absent; NOT confirmed to be the server on :{port})"
            )
        }
        // Named by the PID file but not confirmed against the port. With two
        // instances sharing one PID file this can be a DIFFERENT server, so it
        // must not be presented as fact.
        Source::PidFile => {
            format!("(up {up}; from {pf} — NOT confirmed to be the server on :{port})")
        }
    };
    let glyph = if matches!(p.source, Source::Server | Source::Port) {
        st.ok()
    } else {
        st.warn()
    };
    let main = p.main_pid.map_or_else(|| "?".into(), |v| v.to_string());
    // No worker means an unsupervised single process (no --supervise), which is
    // how the release bundle's launcher starts it. Calling that "supervisor N /
    // worker unknown" would invent a process tree that is not there.
    let shape = match p.worker_pid {
        Some(w) => format!("supervisor {main} / worker {w}"),
        None => format!("pid {main} {}", st.dim("(single process, not supervised)")),
    };
    format!("{glyph} {label} running — {shape}  {}", st.dim(&note))
}

/// Python's `x or 0` on a metric: `None` and `-0.0` both become `0.0`. The
/// server exports a literal `-0` for an idle device, which would otherwise
/// print as `-0.0 GB` / `-0%`.
fn zero_if_none_or_zero(v: Option<f64>) -> f64 {
    v.filter(|x| *x != 0.0).unwrap_or(0.0)
}

/// A memory budget as text. `0` means the gate is off for that domain (the release bundle ships
/// `total_vram_gb: 0.0`), which `0.0 GB` would read as "over budget".
fn budget_text(total: f64) -> String {
    if total > 0.0 {
        human_bytes(total)
    } else {
        "no limit".to_owned()
    }
}

fn vram_section(ctx: &Ctx, rows: &[Sample], headroom: Option<&Value>) {
    let st = ctx.style;
    let devices: BTreeSet<&str> = rows
        .iter()
        .filter(|s| s.name == "rustedvino_vram_total_bytes")
        .filter_map(|s| s.labels.get("device").map(String::as_str))
        .collect();
    if devices.is_empty() {
        return;
    }
    println!("{}", st.bold("VRAM"));
    for dev in devices {
        let total = zero_if_none_or_zero(metrics::by_label(
            rows,
            "rustedvino_vram_total_bytes",
            &[("device", dev)],
        ));
        let used = zero_if_none_or_zero(metrics::by_label(
            rows,
            "rustedvino_vram_used_bytes",
            &[("device", dev)],
        ));
        println!(
            "  {dev:<8} {:>9} / {:<9} {}",
            human_bytes(used),
            budget_text(total),
            bar(st, used, total)
        );
    }
    if let Some(total) =
        metrics::by_label(rows, "rustedvino_vram_total_bytes", &[("domain", "system")])
    {
        let used = zero_if_none_or_zero(metrics::by_label(
            rows,
            "rustedvino_vram_used_bytes",
            &[("domain", "system")],
        ))
        .max(0.0);
        println!(
            "  {:<8} {:>9} / {:<9} {}",
            "system",
            human_bytes(used),
            budget_text(total),
            bar(st, used, total)
        );
    }
    headroom_lines(ctx, headroom);
    println!();
}

/// The device-headroom state of each discrete GPU (`server.headroom` of the admin API): what a
/// vision model's long prompt can still be given, and where the figure comes from.
fn headroom_lines(ctx: &Ctx, headroom: Option<&Value>) {
    let st = ctx.style;
    let Some(h) = headroom else {
        return;
    };
    let devices = h["devices"].as_array().map_or(&[][..], Vec::as_slice);
    let mode = h["mode"].as_str().unwrap_or("?");
    let probe = h["probe"].as_str().unwrap_or("none");
    if mode == "off" {
        println!("  {}", st.dim("headroom: off (kv_headroom_mode)"));
        return;
    }
    if probe == "none" {
        println!(
            "  {}",
            st.dim("headroom: unknown (the server cannot read device memory on this platform)")
        );
        return;
    }
    if devices.is_empty() {
        return;
    }
    for d in devices {
        let g = |k: &str| d[k].as_f64().unwrap_or(0.0);
        let outside = d["outside_mib"].as_f64().map_or_else(
            || "outside unknown".to_owned(),
            |m| format!("{:.2} GiB others", m / 1024.0),
        );
        let leased = d["leases_mib"]
            .as_f64()
            .map_or_else(String::new, |m| format!("{m:.0} MiB leased, "));
        println!(
            "  {:<8} headroom: {:.1} GiB total, {outside}, {:.2} GiB mine (booked or measured, \
             the larger), {leased}{:.2} GiB left for a prompt (probe: {probe})",
            d["device"].as_str().unwrap_or("?"),
            g("total_gib"),
            g("own_gib"),
            g("free_gib"),
        );
    }
}

fn models_section(ctx: &Ctx, models: &Value, admin_models: Option<&Value>, rows_m: &[Sample]) {
    let by_id = |id: &str| {
        admin_models
            .and_then(|a| a.get("models"))
            .and_then(Value::as_array)
            .and_then(|a| a.iter().find(|m| m["id"] == id))
    };
    let Some(data) = models.get("data").and_then(Value::as_array) else {
        return;
    };
    // `/v1/models` has no `kind` (it is not an `OpenAI` field): take it from the admin list. No
    // admin list → every kind is unknown, so the rows come out alphabetically by id.
    let mut data: Vec<&Value> = data.iter().collect();
    crate::order::sort_models(&mut data, |m| {
        let id = m["id"].as_str().unwrap_or("");
        (by_id(id).and_then(|a| a["kind"].as_str()), id)
    });
    let table_rows: Vec<Vec<String>> = data
        .iter()
        .map(|m| {
            let id = m["id"].as_str().unwrap_or("?");
            let a = by_id(id);
            let get = |k: &str| a.and_then(|a| a.get(k));
            let vram = get("vram_gb")
                .and_then(Value::as_f64)
                .map_or_else(|| "n/a".into(), |v| format!("{v:.1} GB"));
            let device = get("device").and_then(Value::as_str).unwrap_or("?");
            let (kv, req) = load_cells(rows_m, id, Some(device));
            let vram = crate::commands::models::mem_cell(device, vram);
            vec![
                id.to_owned(),
                // Without an admin key, the `/v1/models` `state` (since 0.9.1: `not_loaded` /
                // `loading` for an `on_demand` model not loaded yet; absent when Ready).
                get("state")
                    .or_else(|| m.get("state"))
                    .and_then(Value::as_str)
                    .unwrap_or("?")
                    .to_owned(),
                get("device")
                    .and_then(Value::as_str)
                    .unwrap_or("?")
                    .to_owned(),
                vram,
                a.map_or_else(|| "-".to_owned(), crate::commands::models::ctx_cell),
                a.map_or_else(|| "-".to_owned(), crate::commands::models::kv_prec_cell),
                kv,
                req,
            ]
        })
        .collect();
    println!("{}", ctx.style.bold("Models"));
    print!(
        "{}",
        table(
            ctx.style,
            &[
                "ID",
                "STATE",
                "DEVICE",
                "VRAM",
                "CTX",
                "KV PREC",
                "KV%",
                "RUN/WAIT/MAX"
            ],
            &table_rows,
            Some(1)
        )
    );
    println!();
}

/// Runs `rv status <id>`: one model's state line plus its sampling profile (V5c). Exit 0 when the
/// model is registered, 1 otherwise. `--json` prints the model's admin entry.
#[must_use]
pub fn model(ctx: &Ctx, id: &str) -> i32 {
    if !ctx.require_admin_key() {
        return 1;
    }
    let st = ctx.style;
    let entry = match crate::commands::admin_entry(ctx, id) {
        Ok(e) => e,
        Err((code, not_registered)) => {
            if not_registered {
                eprintln!(
                    "  = help: `rv check {id}` previews a model on disk that is not registered; \
                     `rv list` shows them"
                );
            }
            return code;
        }
    };
    if ctx.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&entry).unwrap_or_default()
        );
        return 0;
    }
    println!("{}", model_line(st, id, &entry));
    crate::commands::print_entry_profile(ctx, id, &entry, crate::commands::OldServerNote::Neutral);
    0
}

/// `<id> · <state> · <device> · <vram> · <kind>` of one admin model entry.
fn model_line(st: crate::ansi::Style, id: &str, m: &Value) -> String {
    let get = |k: &str| m.get(k).and_then(Value::as_str).unwrap_or("?");
    let vram = m
        .get("vram_gb")
        .and_then(Value::as_f64)
        .map_or_else(|| "n/a".into(), |v| format!("{v:.1} GB"));
    format!(
        "{} · {} · {} · {} · {}",
        st.bold(id),
        crate::format::state_glyph(st, get("state")),
        get("device"),
        crate::commands::models::mem_cell(get("device"), vram),
        get("kind")
    )
}

/// Runs `rv status`. Exit 0 when `/health` answered 200, else 1.
#[must_use]
#[allow(clippy::too_many_lines)] // one linear report, mirrors the Python verb
pub fn run(ctx: &Ctx) -> i32 {
    ctx.warn_if_open();
    let st = ctx.style;
    let health = ctx.api.get("/health");
    let models = ctx.api.get("/v1/models");
    let health_ok = health.is_200();

    // Only probe when the server actually answered — otherwise the extra
    // requests just burn the timeout to learn nothing.
    let (inf, adm) = if health_ok {
        ctx.probe_auth_enforcement()
    } else {
        (None, None)
    };

    let mut admin_models: Option<Value> = None;
    let mut audit: Option<Value> = None;
    let mut watchdog: Option<Value> = None;
    let mut rows_m: Vec<Sample> = Vec::new();
    if ctx.res.admin_usable() {
        admin_models = ctx.api.get("/v1/admin/models").json().cloned();
        audit = ctx.api.get("/v1/admin/config/audit").json().cloned();
        watchdog = ctx.api.get("/v1/admin/health/watchdog").json().cloned();
        if let Some(t) = ctx.api.get("/metrics").body.as_ref().and_then(|b| b.text()) {
            rows_m = metrics::parse(t);
        }
    }

    // The server's own PID/uptime is authoritative (it is the process that
    // just answered). Older servers, keyless servers and a dead server fall
    // back to OS probing.
    let reported = admin_models.as_ref().and_then(proc::server_report);
    let pstate = reported.map_or_else(
        || proc::probe(ctx.res.port, &ctx.pid_file),
        |(pid, up)| proc::from_server_report(pid, up),
    );

    if ctx.json {
        let doc = json!({
            "host": ctx.host,
            "prefix": Value::Null,
            "process": {
                "alive": pstate.alive,
                "supervisor_pid": pstate.main_pid,
                "worker_pid": pstate.worker_pid,
            },
            "auth": {
                "api_key_count": ctx.res.api_key_count,
                "admin_key_count": ctx.res.admin_key_count,
                "keys_file": ctx.res.keys_file.display().to_string(),
                "keys_file_present": ctx.res.keys_present,
                // Observed from the server; null when undetermined. The counts
                // above describe the LOCAL file and may not be what the running
                // server loaded.
                "inference_auth_enforced": inf,
                "admin_auth_enforced": adm,
            },
            "health": health.body.as_ref().map(crate::api::Payload::to_value),
            "models": models.body.as_ref().map(crate::api::Payload::to_value),
            "admin_models": admin_models,
            "admin_config_audit": audit,
            "admin_watchdog": watchdog,
        });
        println!("{}", serde_json::to_string_pretty(&doc).unwrap_or_default());
        return i32::from(!health_ok);
    }

    println!("{}", st.bold(&format!("RustedVINO — {}", ctx.host)));
    println!();
    println!("{}", process_line(ctx, &pstate, health_ok));

    if let Some(h) = health.json().filter(|_| health_ok) {
        let v = h.get("version").and_then(Value::as_str).unwrap_or("?");
        // E-12: /health is a liveness probe and stays 200 on a poisoned GPU context;
        // its body (public, no admin key needed) and the admin server block say so.
        let poisoned = h["gpu_poisoned"].as_bool().unwrap_or(false)
            || admin_models
                .as_ref()
                .is_some_and(|a| a["server"]["gpu_poisoned"].as_bool().unwrap_or(false));
        if poisoned {
            println!(
                "{} {:<10} alive — v{v}, but {}",
                st.bad(),
                "Health",
                st.red(GPU_POISONED_BANNER)
            );
        } else {
            println!("{} {:<10} ok — v{v}", st.ok(), "Health");
        }
        // Step 6e: public flag; the device/model detail needs the admin watchdog read below.
        if h["device_stalled"].as_bool().unwrap_or(false) {
            println!(
                "{} {:<10} {}",
                st.warn(),
                "Loads",
                st.yellow("a model load has not returned — its device takes no new loads")
            );
        }
    } else {
        println!(
            "{} {:<10} unreachable  {}",
            st.bad(),
            "Health",
            st.dim(health.err.as_deref().unwrap_or(""))
        );
    }
    // Step 8: driver versions and startup warnings (admin server block).
    if let Some(server) = admin_models.as_ref().map(|a| &a["server"]) {
        let (warnings, drivers) = startup_report_lines(server);
        for d in drivers {
            println!("{} {:<10} {}", st.ok(), "Driver", st.dim(&d));
        }
        for w in warnings {
            println!("{} {:<10} {}", st.warn(), "Startup", st.yellow(&w));
        }
    }
    // Report what the SERVER does, not what the local keys file implies.
    println!("{}", auth_line(ctx, inf, adm));
    println!();

    if let Some(m) = models.json().filter(|m| {
        m.get("data")
            .and_then(Value::as_array)
            .is_some_and(|d| !d.is_empty())
    }) {
        models_section(ctx, m, admin_models.as_ref(), &rows_m);
    } else if models.code != Some(200) {
        println!(
            "{} Models: unreachable/unauthorized  {}",
            st.bad(),
            st.dim(models.err.as_deref().unwrap_or(""))
        );
        println!();
    }

    if !rows_m.is_empty() {
        vram_section(
            ctx,
            &rows_m,
            admin_models.as_ref().map(|a| &a["server"]["headroom"]),
        );
    }

    if let Some(a) = audit.as_ref().filter(|a| a.is_object()) {
        let problems = a
            .get("problems")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let total = a
            .get("total_declared")
            .map_or_else(|| "?".into(), Value::to_string);
        if problems.is_empty() {
            println!(
                "{} {:<10} no drift — {total} model(s) declared, all healthy",
                st.ok(),
                "Config"
            );
        } else {
            println!(
                "{} {:<10} {} problem(s) of {total} declared model(s):",
                st.warn(),
                "Config",
                problems.len()
            );
            for p in problems.iter().take(AUDIT_LINES_SHOWN) {
                println!("    - {}", problem_line(p));
            }
            if problems.len() > AUDIT_LINES_SHOWN {
                println!(
                    "    … and {} more (`rv status --json` lists them all)",
                    problems.len() - AUDIT_LINES_SHOWN
                );
            }
        }
    } else if ctx.res.admin_usable() {
        println!(
            "{} {:<10} audit unreachable/unauthorized",
            st.warn(),
            "Config"
        );
    }

    if let Some(w) = watchdog.as_ref().filter(|w| w.is_object()) {
        for line in stalled_lines(w) {
            println!("{} {:<10} {}", st.bad(), "Stalled", st.red(&line));
        }
        for line in recovery_lines(w) {
            println!("{} {:<10} {}", st.warn(), "Recovery", st.yellow(&line));
        }
        if let Some(n) = w
            .get("in_flight")
            .and_then(Value::as_u64)
            .filter(|n| *n > 0)
        {
            println!(
                "{} {:<10} {n} OpenVINO call(s) in flight",
                st.ok(),
                "In flight"
            );
        }
        match w.get("oldest_generation_secs").and_then(Value::as_f64) {
            None => println!(
                "{} {:<10} idle — no in-flight generations",
                st.ok(),
                "Watchdog"
            ),
            Some(age) => println!(
                "{} {:<10} oldest in-flight generation: {age:.1}s",
                if age < 30.0 { st.ok() } else { st.warn() },
                "Watchdog"
            ),
        }
    }

    if !ctx.res.admin_usable() {
        println!();
        println!(
            "{}",
            st.dim(&format!(
                "(admin-gated sections skipped — no admin key available; set RV_ADMIN_KEY or \
                 populate {})",
                ctx.res.keys_file.display()
            ))
        );
    }
    i32::from(!health_ok)
}

/// `rv status` / `rv start` lines for the admin `server` block's step-8 fields: one per startup
/// warning (`code: message`) and one per device driver (`GPU.1  <name>  driver <v>`, kernel last).
/// Empty for an older server or a caller without an admin key.
#[must_use]
pub fn startup_report_lines(server: &Value) -> (Vec<String>, Vec<String>) {
    let warnings = server
        .get("startup_warnings")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|w| {
                    format!(
                        "{}: {}",
                        w["code"].as_str().unwrap_or("?"),
                        w["message"].as_str().unwrap_or("")
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    let mut drivers: Vec<String> = server["platform"]["drivers"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|d| {
                    format!(
                        "{}  {}  driver {}",
                        d["device"].as_str().unwrap_or("?"),
                        d["name"].as_str().unwrap_or("?"),
                        d["driver_version"].as_str().unwrap_or("?")
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    if let Some(k) = server["platform"]["kernel"].as_str() {
        drivers.push(format!("kernel {k}"));
    }
    (warnings, drivers)
}

/// One line per stalled load in a `/v1/admin/health/watchdog` body (step 6e); empty for an
/// older server without the field.
fn stalled_lines(watchdog: &Value) -> Vec<String> {
    watchdog
        .get("stalled")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|s| {
                    let dev = s["device"].as_str().unwrap_or("?");
                    format!(
                        "{} of {} on {dev} has not returned for {} s — no new loads on {dev} \
                         until it does",
                        s["kind"].as_str().unwrap_or("load"),
                        s["model"].as_str().unwrap_or("?"),
                        s["secs"].as_u64().unwrap_or(0),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The distinct recovery steps of the stalled loads in a `/v1/admin/health/watchdog` body (D-04
/// S9), in order; empty for an older server without the field.
fn recovery_lines(watchdog: &Value) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for r in watchdog
        .get("stalled")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|s| s["recovery"].as_str())
    {
        if !out.iter().any(|o| o == r) {
            out.push(r.to_owned());
        }
    }
    out
}

/// The `rv status` banner for a poisoned GPU context (E-12); `rv ps` / `rv list`
/// print their own text from `models.rs`.
const GPU_POISONED_BANNER: &str = "GPU CONTEXT POISONED — restart required; GPU models are unavailable \
     until the server restarts (a `--supervise` worker restarts itself after a serving failure; \
     otherwise restart it by hand)";

/// How many config-audit problems `rv status` prints before summarising the rest.
const AUDIT_LINES_SHOWN: usize = 8;

/// One config-audit problem as a readable line. The server sends an object per
/// problem (`model_id`, `files_missing`, `in_preload`, `pending_reload`); `rv
/// status` used to print it as raw JSON (Windows pass 2, D14). A bare string
/// (an older server) is shown as is.
fn problem_line(p: &Value) -> String {
    if let Some(text) = p.as_str() {
        return text.to_owned();
    }
    let Some(id) = p.get("model_id").and_then(Value::as_str) else {
        return p.to_string();
    };
    let flag = |k: &str| p.get(k).and_then(Value::as_bool).unwrap_or(false);
    let mut notes = Vec::new();
    if flag("files_missing") {
        notes.push("files missing");
    }
    if flag("in_preload") {
        notes.push("in preload - fatal at next start");
    }
    if flag("pending_reload") {
        notes.push("on disk, needs a config reload");
    }
    if notes.is_empty() {
        id.to_owned()
    } else {
        format!("{id}: {}", notes.join(", "))
    }
}

#[cfg(test)]
mod problem_line_tests {
    use super::{problem_line, recovery_lines, stalled_lines, startup_report_lines};
    use serde_json::json;

    #[test]
    fn object_problems_read_as_a_sentence() {
        let p = json!({"model_id":"m","expected_path":"/x/m","files_missing":true,"in_preload":false,"pending_reload":false});
        assert_eq!(problem_line(&p), "m: files missing");
        let p =
            json!({"model_id":"n","files_missing":true,"in_preload":true,"pending_reload":false});
        assert_eq!(
            problem_line(&p),
            "n: files missing, in preload - fatal at next start"
        );
    }

    #[test]
    fn strings_and_unknown_shapes_pass_through() {
        assert_eq!(problem_line(&json!("old server text")), "old server text");
        assert_eq!(problem_line(&json!({"x":1})), r#"{"x":1}"#);
    }

    /// 6e: each stalled load in the watchdog body becomes one line; an older server's body
    /// (no `stalled` field) gives none.
    /// Step 8: warnings and drivers from the admin server block; nothing from an older server.
    #[test]
    fn startup_report_lines_render_warnings_and_drivers() {
        let server = json!({
            "startup_warnings": [{"code": "global_device_unknown", "message": "device GPU.7 …"}],
            "platform": {"drivers": [{"device": "GPU.1", "name": "B60", "driver_version": "25.18.33578"}],
                         "kernel": "7.0.0-38-generic"}
        });
        let (w, d) = startup_report_lines(&server);
        assert_eq!(w, vec!["global_device_unknown: device GPU.7 …".to_owned()]);
        assert_eq!(
            d,
            vec![
                "GPU.1  B60  driver 25.18.33578".to_owned(),
                "kernel 7.0.0-38-generic".to_owned()
            ]
        );
        let (w, d) = startup_report_lines(&json!({"engine": "RustedVINO"}));
        assert!(w.is_empty() && d.is_empty());
    }

    #[test]
    fn stalled_lines_render_each_stalled_load() {
        let w = json!({"stalled": [
            {"device": "GPU.1", "model": "big", "kind": "load", "secs": 700}
        ]});
        assert_eq!(
            stalled_lines(&w),
            vec![
                "load of big on GPU.1 has not returned for 700 s — no new loads on GPU.1 until it \
                 does"
                    .to_owned()
            ]
        );
        assert!(stalled_lines(&json!({"oldest_generation_secs": null})).is_empty());
    }

    /// D-04 S9: one line per distinct recovery step; an older server without the field shows none.
    #[test]
    fn recovery_lines_are_distinct_and_optional() {
        let w = json!({"stalled": [
            {"device": "GPU.1", "model": "a", "kind": "load", "secs": 700, "recovery": "reset GPU.1"},
            {"device": "GPU.1", "model": "b", "kind": "load", "secs": 650, "recovery": "reset GPU.1"},
            {"device": "GPU.0", "model": "c", "kind": "load", "secs": 640, "recovery": "reboot"}
        ]});
        assert_eq!(
            recovery_lines(&w),
            vec!["reset GPU.1".to_owned(), "reboot".to_owned()]
        );
        let old =
            json!({"stalled": [{"device": "GPU.1", "model": "a", "kind": "load", "secs": 1}]});
        assert!(recovery_lines(&old).is_empty());
    }
}
