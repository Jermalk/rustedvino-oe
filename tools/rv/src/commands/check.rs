//! `rv check MODEL` — dry-run admission check.

use crate::api::{READ_TIMEOUT, encode_segment};
use crate::ctx::Ctx;

/// Runs `rv check`. Exit 0 when the model fits (or is loaded), 1 otherwise.
#[must_use]
pub fn run(ctx: &Ctx, model: &str) -> i32 {
    if !ctx.require_admin_key() {
        return 1;
    }
    let r = ctx.api.post(
        &format!("/v1/admin/models/{}/check", encode_segment(model)),
        None,
        READ_TIMEOUT,
    );
    let Some(body) = r.json().filter(|_| r.code == Some(200)) else {
        eprintln!(
            "{}",
            ctx.style.red(&format!(
                "error: {}",
                r.err.unwrap_or_else(|| crate::commands::code_text(r.code))
            ))
        );
        return 1;
    };
    if ctx.json {
        println!("{}", serde_json::to_string_pretty(body).unwrap_or_default());
        return 0;
    }
    let s = ctx.style;
    if body["already_loaded"].as_bool().unwrap_or(false) {
        println!("{} '{model}' is already loaded", s.ok());
        return 0;
    }
    let fits = body["fits"].as_bool().unwrap_or(false);
    let glyph = if fits { s.ok() } else { s.bad() };
    println!(
        "{glyph} '{model}' — needs {:.1} GB, {:.1} GB free",
        body["needed_gb"].as_f64().unwrap_or(0.0),
        body["free_gb"].as_f64().unwrap_or(0.0)
    );
    // Shared-memory (UMA) boxes gate on live system RAM as well as VRAM, and it
    // is the RAM gate that refuses loads there. Absent on discrete-GPU servers
    // and on servers older than this field pair.
    if let (Some(need), Some(avail)) = (
        body["ram_needed_gb"].as_f64(),
        body["ram_avail_gb"].as_f64(),
    ) {
        let line = format!("  system RAM: needs {need:.1} GB, {avail:.1} GB available");
        println!("{}", if fits { line } else { s.yellow(&line) });
    }
    let evict: Vec<&str> = body["would_evict"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    if !evict.is_empty() {
        println!("  would evict: {}", evict.join(", "));
        let busy: Vec<&str> = body["busy_victims"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        if !busy.is_empty() {
            println!(
                "{}",
                s.yellow(&format!(
                    "  BUSY (a request is running; drained for 15 s, then aborted): {}",
                    busy.join(", ")
                ))
            );
        }
    } else if !fits {
        println!(
            "{}",
            s.yellow("  does not fit even after evicting everything evictable")
        );
    }
    i32::from(!fits)
}
