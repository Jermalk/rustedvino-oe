//! `rv logs` — tail the server's in-memory log ring buffer.

use crate::ctx::Ctx;

/// Runs `rv logs -n N`.
#[must_use]
pub fn run(ctx: &Ctx, n: u32) -> i32 {
    if !ctx.require_admin_key() {
        return 1;
    }
    let r = ctx.api.get(&format!("/v1/admin/logs/tail?lines={n}"));
    if r.code != Some(200) {
        eprintln!(
            "{}",
            ctx.style.red(&format!(
                "error: {}",
                r.err.unwrap_or_else(|| crate::commands::code_text(r.code))
            ))
        );
        return 1;
    }
    match r.body {
        Some(crate::api::Payload::Text(t)) => println!("{t}"),
        Some(crate::api::Payload::Json(v)) => {
            println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
        }
        None => {}
    }
    0
}
