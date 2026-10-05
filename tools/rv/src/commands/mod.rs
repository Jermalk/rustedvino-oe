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
