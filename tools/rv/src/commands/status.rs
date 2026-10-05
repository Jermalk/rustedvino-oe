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
    } else if r.admin_key.is_none() {
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

fn vram_section(ctx: &Ctx, rows: &[Sample]) {
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
            human_bytes(total),
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
            human_bytes(total),
            bar(st, used, total)
        );
    }
    println!();
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
    let table_rows: Vec<Vec<String>> = data
        .iter()
        .map(|m| {
            let id = m["id"].as_str().unwrap_or("?");
            let a = by_id(id);
            let get = |k: &str| a.and_then(|a| a.get(k));
            let vram = get("vram_gb")
                .and_then(Value::as_f64)
                .map_or_else(|| "n/a".into(), |v| format!("{v:.1} GB"));
            let (kv, req) = load_cells(rows_m, id);
            vec![
                id.to_owned(),
                get("state")
                    .and_then(Value::as_str)
                    .unwrap_or("?")
                    .to_owned(),
                get("device")
                    .and_then(Value::as_str)
                    .unwrap_or("?")
                    .to_owned(),
                vram,
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
            &["ID", "STATE", "DEVICE", "VRAM", "KV%", "RUN/WAIT/MAX"],
            &table_rows,
            Some(1)
        )
    );
    println!();
}

/// Runs `rv status`. Exit 0 when `/health` answered 200, else 1.
#[must_use]
#[allow(clippy::too_many_lines)] // one linear report, mirrors the Python verb
pub fn run(ctx: &Ctx) -> i32 {
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
    if ctx.res.admin_key.is_some() {
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
        println!("{} {:<10} ok — v{v}", st.ok(), "Health");
    } else {
        println!(
            "{} {:<10} unreachable  {}",
            st.bad(),
            "Health",
            st.dim(health.err.as_deref().unwrap_or(""))
        );
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
        vram_section(ctx, &rows_m);
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
    } else if ctx.res.admin_key.is_some() {
        println!(
            "{} {:<10} audit unreachable/unauthorized",
            st.warn(),
            "Config"
        );
    }

    if let Some(w) = watchdog.as_ref().filter(|w| w.is_object()) {
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

    if ctx.res.admin_key.is_none() {
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
    use super::problem_line;
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
}
