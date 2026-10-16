//! The verbs. One module per verb family; each returns the process exit code.

use crate::api::Reply;
use crate::ctx::Ctx;

pub mod check;
pub mod config;
pub mod lifecycle;
pub mod logs;
pub mod models;
pub mod ops;
pub mod prompt;
pub mod service;
pub mod status;

/// `200`, or `no response` when the request never got one.
#[must_use]
pub fn code_text(code: Option<u16>) -> String {
    code.map_or_else(|| "no response".to_owned(), |c| c.to_string())
}

/// The most useful text of a failed reply: the server's `{"error": ...}` /
/// `{"error": {"message": ...}}` message when the body is such JSON, else the
/// raw body, else the transport error.
#[must_use]
pub fn error_detail(r: &Reply) -> String {
    let raw = r
        .err
        .clone()
        .or_else(|| r.body.as_ref().map(|b| b.to_value().to_string()))
        .unwrap_or_default();
    serde_json::from_str::<serde_json::Value>(&raw)
        .ok()
        .and_then(|v| {
            let e = v.get("error")?;
            e.as_str()
                .map(str::to_owned)
                .or_else(|| e.get("message")?.as_str().map(str::to_owned))
        })
        .unwrap_or(raw)
}

/// The machine-readable `error.code` of a JSON error reply (`model_busy`, ...), if it has one.
pub fn error_code(r: &Reply) -> Option<String> {
    let raw = r
        .err
        .clone()
        .or_else(|| r.body.as_ref().map(|b| b.to_value().to_string()))?;
    serde_json::from_str::<serde_json::Value>(&raw)
        .ok()?
        .get("error")?
        .get("code")?
        .as_str()
        .map(str::to_owned)
}

/// After a refused eviction (`model_busy`): say why and how to override it. The server refuses to
/// evict a model that has a request running, so the request finishes undisturbed; `--force`
/// drains it (bounded) and then aborts what is left.
pub fn print_busy_hint(ctx: &Ctx, r: &Reply, verb: &str) {
    if error_code(r).as_deref() == Some("model_busy") {
        eprintln!(
            "{}",
            ctx.style.dim(&format!(
                "(a request is running on it, so it was left alone — retry when it finishes, or \
                 `rv {verb} --force` to drain it and abort what is left)"
            ))
        );
    }
}

/// Prints `error (CODE): detail` to stderr, in red.
pub fn print_error(ctx: &Ctx, r: &Reply) {
    eprintln!(
        "{}",
        ctx.style.red(&format!(
            "error ({}): {}",
            code_text(r.code),
            error_detail(r)
        ))
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reply(err: &str) -> Reply {
        Reply {
            code: Some(404),
            body: None,
            err: Some(err.to_owned()),
        }
    }

    #[test]
    fn error_code_reads_the_json_envelope_only() {
        assert_eq!(
            error_code(&reply(r#"{"error":{"message":"m","code":"model_busy"}}"#)).as_deref(),
            Some("model_busy")
        );
        assert_eq!(error_code(&reply(r#"{"error":"plain"}"#)), None);
        assert_eq!(error_code(&reply("not json")), None);
    }

    #[test]
    fn error_detail_unwraps_json_error_shapes() {
        assert_eq!(
            error_detail(&reply(r#"{"error":"session not found"}"#)),
            "session not found"
        );
        assert_eq!(
            error_detail(&reply(
                r#"{"error":{"message":"Missing API key","code":"x"}}"#
            )),
            "Missing API key"
        );
        assert_eq!(
            error_detail(&reply("model 'x' not found")),
            "model 'x' not found"
        );
        assert_eq!(error_detail(&reply(r#"{"other":1}"#)), r#"{"other":1}"#);
    }
}
