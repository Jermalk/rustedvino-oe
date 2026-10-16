//! Verbs for admin routes the Python `rv` never wrapped: `resize`,
//! `completeness`, `reload-keys`, `voice-pin`, `sessions`, `session`,
//! `session-kill`. The router in `src/lib.rs` is the source of truth for what
//! exists; re-read it before adding to this list.

use serde_json::{Value, json};

use crate::api::{MUTATE_TIMEOUT, Payload, READ_TIMEOUT, encode_segment};
use crate::commands::print_error;
use crate::commands::prompt::confirm;
use crate::ctx::Ctx;
use crate::format::table;

fn pretty(v: &Value) -> String {
    serde_json::to_string_pretty(v).unwrap_or_default()
}

fn text<'a>(v: &'a Value, k: &str) -> &'a str {
    v.get(k).and_then(Value::as_str).unwrap_or("-")
}

/// `rv resize MODEL --kv-cache-gb F`: evict then reload a resident model with
/// a new KV-cache pool. Needs confirmation: it drains in-flight work, and on a
/// failed reload the model may be left unloaded (the server deliberately does
/// not roll back).
#[must_use]
pub fn resize(ctx: &Ctx, model: &str, kv_cache_gb: f64, yes: bool, force: bool) -> i32 {
    if !ctx.require_admin_key() {
        return 1;
    }
    let st = ctx.style;
    if !kv_cache_gb.is_finite() || kv_cache_gb <= 0.0 {
        eprintln!(
            "{}",
            st.red(&format!(
                "error: --kv-cache-gb must be a finite positive number, got {kv_cache_gb}"
            ))
        );
        return 1;
    }
    if !confirm(
        st,
        &format!(
            "resize '{model}' to a {kv_cache_gb} GB KV cache — evicts and reloads it (in-flight \
             requests drain first; takes 10-60s; if the reload fails the model is left unloaded). \
             Proceed?"
        ),
        yes,
    ) {
        println!("aborted");
        return 2;
    }
    println!(
        "{}",
        st.dim(&format!(
            "resizing '{model}' — evict + reload, can take 10-60s..."
        ))
    );
    let r = ctx.api.post(
        &format!("/v1/admin/models/{}/resize", encode_segment(model)),
        Some(&if force {
            json!({ "kv_cache_gb": kv_cache_gb, "force": true })
        } else {
            json!({ "kv_cache_gb": kv_cache_gb })
        }),
        MUTATE_TIMEOUT,
    );
    if r.code == Some(200) {
        let msg = r.body.as_ref().and_then(Payload::text).unwrap_or("resized");
        println!("{} {msg}", st.ok());
        return 0;
    }
    print_error(ctx, &r);
    if crate::commands::error_code(&r).as_deref() == Some("model_busy") {
        crate::commands::print_busy_hint(
            ctx,
            &r,
            &format!("resize {model} --kv-cache-gb {kv_cache_gb}"),
        );
    } else if r.code == Some(409) {
        eprintln!(
            "{}",
            st.dim(&format!(
                "(resize only applies to a loaded model — `rv load {model}` first)"
            ))
        );
    }
    1
}

/// `rv completeness`: every registered model's file-completeness report,
/// without loading anything. Exit 1 when any model has a blocking gap.
#[must_use]
pub fn completeness(ctx: &Ctx) -> i32 {
    if !ctx.require_admin_key() {
        return 1;
    }
    let st = ctx.style;
    let r = ctx.api.get("/v1/admin/models/completeness");
    let Some(body) = r.json().filter(|v| r.code == Some(200) && v.is_object()) else {
        print_error(ctx, &r);
        return 1;
    };
    let incomplete = body
        .get("incomplete_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if ctx.json {
        println!("{}", pretty(body));
        return i32::from(incomplete > 0);
    }
    let empty = Vec::new();
    let models = body
        .get("models")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    let join = |m: &Value, k: &str| -> String {
        let v: Vec<&str> = m
            .get(k)
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        v.join(", ")
    };
    let rows: Vec<Vec<String>> = models
        .iter()
        .map(|m| {
            let complete = m.get("complete").and_then(Value::as_bool).unwrap_or(false);
            vec![
                text(m, "id").to_owned(),
                text(m, "kind").to_owned(),
                if complete { "complete" } else { "INCOMPLETE" }.to_owned(),
                join(m, "blocking_missing"),
                join(m, "advisory_missing"),
            ]
        })
        .collect();
    let glyph = if incomplete == 0 { st.ok() } else { st.bad() };
    println!(
        "{glyph} {}",
        st.bold(&format!(
            "{} model(s) checked — {incomplete} with a blocking gap",
            models.len()
        ))
    );
    print!(
        "{}",
        table(
            st,
            &["ID", "KIND", "FILES", "BLOCKING", "ADVISORY"],
            &rows,
            None
        )
    );
    i32::from(incomplete > 0)
}

/// `rv reload-keys`: re-read the keys file and hot-swap the keys. Counts only.
#[must_use]
pub fn reload_keys(ctx: &Ctx) -> i32 {
    if !ctx.require_admin_key() {
        return 1;
    }
    let st = ctx.style;
    let r = ctx.api.post("/v1/admin/keys/reload", None, READ_TIMEOUT);
    let Some(b) = r.json().filter(|v| r.code == Some(200) && v.is_object()) else {
        print_error(ctx, &r);
        return 1;
    };
    if ctx.json {
        println!("{}", pretty(b));
        return 0;
    }
    let flag = |k: &str| b.get(k).and_then(Value::as_bool).unwrap_or(false);
    let count = |k: &str| b.get(k).and_then(Value::as_u64).unwrap_or(0);
    let word = |changed: bool| if changed { "updated" } else { "unchanged" };
    println!(
        "{} keys reloaded — inference keys {} ({} live), admin keys {} ({} live)",
        st.ok(),
        word(flag("keys_updated")),
        count("api_key_count"),
        word(flag("admin_keys_updated")),
        count("admin_key_count")
    );
    if flag("admin_downgrade_refused") {
        println!(
            "{}",
            st.yellow(&format!(
                "{} the file's admin_api_keys is empty while admin keys are live — that downgrade \
                 was refused, the live admin keys are kept",
                st.warn()
            ))
        );
    }
    0
}

/// `rv voice-pin`: the realtime voice-flow model pin.
#[must_use]
pub fn voice_pin(ctx: &Ctx) -> i32 {
    if !ctx.require_admin_key() {
        return 1;
    }
    let st = ctx.style;
    let r = ctx.api.get("/v1/admin/voice-pin");
    let Some(b) = r.json().filter(|v| r.code == Some(200) && v.is_object()) else {
        print_error(ctx, &r);
        return 1;
    };
    if ctx.json {
        println!("{}", pretty(b));
        return 0;
    }
    let sessions = b
        .get("active_sessions")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let ttl = b.get("ttl_secs").and_then(Value::as_u64).unwrap_or(0);
    match b.get("pin").filter(|p| p.is_object()) {
        Some(p) => {
            println!("{} voice pin active", st.ok());
            println!("  stt  {}", text(p, "stt_model"));
            println!("  llm  {}", text(p, "llm_model"));
            println!("  tts  {}", text(p, "tts_model"));
        }
        None => println!("{} no voice pin active", st.dim("●")),
    }
    println!(
        "{}",
        st.dim(&format!(
            "{sessions} realtime session(s) open; pin clears {ttl}s after the last one closes"
        ))
    );
    0
}

/// `rv sessions`: live realtime WebSocket sessions.
#[must_use]
pub fn sessions(ctx: &Ctx) -> i32 {
    if !ctx.require_admin_key() {
        return 1;
    }
    let r = ctx.api.get("/v1/admin/realtime/sessions");
    let Some(b) = r.json().filter(|v| r.code == Some(200) && v.is_object()) else {
        print_error(ctx, &r);
        return 1;
    };
    if ctx.json {
        println!("{}", pretty(b));
        return 0;
    }
    let empty = Vec::new();
    let list = b
        .get("sessions")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    let cell = |s: &Value, k: &str| match s.get(k) {
        Some(Value::String(t)) => t.clone(),
        Some(Value::Null) | None => "-".to_owned(),
        Some(v) => v.to_string(),
    };
    let rows: Vec<Vec<String>> = list
        .iter()
        .map(|s| {
            [
                "id",
                "connected_at",
                "stt_model",
                "llm_model",
                "tts_model",
                "turn_count",
            ]
            .iter()
            .map(|k| cell(s, k))
            .collect()
        })
        .collect();
    println!(
        "{}",
        ctx.style
            .bold(&format!("{} realtime session(s)", list.len()))
    );
    print!(
        "{}",
        table(
            ctx.style,
            &["ID", "CONNECTED", "STT", "LLM", "TTS", "TURNS"],
            &rows,
            None
        )
    );
    0
}

/// Whether `s` is a hex UUID, dashed (`8-4-4-4-12`) or not. The server's path
/// extractor wants a UUID; checking here gives a clear message instead of a
/// bare 400.
#[must_use]
pub fn is_uuid(s: &str) -> bool {
    // The server's `Path<Uuid>` also accepts the 32-hex form without dashes.
    if s.len() == 32 {
        return s.chars().all(|c| c.is_ascii_hexdigit());
    }
    let parts: Vec<&str> = s.split('-').collect();
    parts.len() == 5
        && parts
            .iter()
            .zip([8, 4, 4, 4, 12])
            .all(|(p, n)| p.len() == n && p.chars().all(|c| c.is_ascii_hexdigit()))
}

fn bad_uuid(ctx: &Ctx, id: &str) -> i32 {
    eprintln!(
        "{}",
        ctx.style.red(&format!(
            "error: '{id}' is not a session id (expected a UUID — see `rv sessions`)"
        ))
    );
    1
}

/// `rv session ID`: one session's full snapshot (config, history, turns), as
/// JSON — it is a nested document, not a table.
#[must_use]
pub fn session(ctx: &Ctx, id: &str) -> i32 {
    if !ctx.require_admin_key() {
        return 1;
    }
    if !is_uuid(id) {
        return bad_uuid(ctx, id);
    }
    let r = ctx.api.get(&format!("/v1/admin/realtime/sessions/{id}"));
    if let Some(b) = r.json().filter(|_| r.code == Some(200)) {
        println!("{}", pretty(b));
        return 0;
    }
    print_error(ctx, &r);
    1
}

/// `rv session-kill ID`: force-close a live realtime session. The WebSocket
/// closes cleanly on the pipeline's next iteration.
#[must_use]
pub fn session_kill(ctx: &Ctx, id: &str, yes: bool) -> i32 {
    if !ctx.require_admin_key() {
        return 1;
    }
    if !is_uuid(id) {
        return bad_uuid(ctx, id);
    }
    if !confirm(
        ctx.style,
        &format!("force-close realtime session {id}? The client's live call ends. Proceed?"),
        yes,
    ) {
        println!("aborted");
        return 2;
    }
    let r = ctx.api.delete(&format!("/v1/admin/realtime/sessions/{id}"));
    if matches!(r.code, Some(200 | 204)) {
        println!("{} close requested for session {id}", ctx.style.ok());
        0
    } else {
        print_error(ctx, &r);
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_check() {
        assert!(is_uuid("123e4567-e89b-12d3-a456-426614174000"));
        assert!(is_uuid("123E4567-E89B-12D3-A456-426614174000"));
        assert!(is_uuid("123e4567e89b12d3a456426614174000"));
        for bad in [
            "",
            "abc",
            "123e4567e89b12d3a45642661417400",
            "123e4567-e89b-12d3-a456-42661417400",
            "zzzzzzzz-e89b-12d3-a456-426614174000",
            "../../etc/passwd",
        ] {
            assert!(!is_uuid(bad), "{bad:?}");
        }
    }
}
