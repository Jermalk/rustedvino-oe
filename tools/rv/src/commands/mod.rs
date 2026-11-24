//! The verbs. One module per verb family; each returns the process exit code.

use crate::api::Reply;
use crate::ctx::Ctx;

pub mod check;
pub mod config;
pub mod lifecycle;
pub mod logs;
pub mod metrics;
pub mod models;
pub mod ops;
pub mod outcome;
pub mod prompt;
mod refusal;
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

/// One model's admin entry, looked up in `GET /v1/admin/models` (there is no single-model GET).
pub(crate) enum EntryLookup {
    /// The entry.
    Found(serde_json::Value),
    /// The list came back and the model is not in it.
    NotRegistered,
    /// The list is not available (no response, 401, ...): the reply says why.
    Unavailable(Reply),
}

/// Looks `model` up in the registry.
pub(crate) fn lookup_entry(ctx: &Ctx, model: &str) -> EntryLookup {
    let r = ctx.api.get("/v1/admin/models");
    let Some(models) = r
        .json()
        .filter(|_| r.code == Some(200))
        .and_then(|b| b.get("models"))
        .and_then(serde_json::Value::as_array)
    else {
        return EntryLookup::Unavailable(r);
    };
    models
        .iter()
        .find(|m| m.get("id").and_then(serde_json::Value::as_str) == Some(model))
        .cloned()
        .map_or(EntryLookup::NotRegistered, EntryLookup::Found)
}

/// The entry of `model`, or `None` quietly when the list or the model is not there.
pub(crate) fn admin_entry_quiet(ctx: &Ctx, model: &str) -> Option<serde_json::Value> {
    match lookup_entry(ctx, model) {
        EntryLookup::Found(e) => Some(e),
        _ => None,
    }
}

/// The entry of `model`, or `Err(1)` after printing why there is none (`not registered`, or the
/// failed list request). `not_registered` is set when the list answered without the model.
pub(crate) fn admin_entry(ctx: &Ctx, model: &str) -> Result<serde_json::Value, (i32, bool)> {
    match lookup_entry(ctx, model) {
        EntryLookup::Found(e) => Ok(e),
        EntryLookup::NotRegistered => {
            eprintln!(
                "{}",
                ctx.style
                    .red(&format!("error: model '{model}' is not registered"))
            );
            Err((1, true))
        }
        EntryLookup::Unavailable(r) => {
            print_error(ctx, &r);
            Err((1, false))
        }
    }
}

/// How a verb words a text model whose entry has no `sampling` field (a server < 0.9.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OldServerNote {
    /// `add` / `patch`: sampling flags would be dropped — the warning (brief item 6).
    Warning,
    /// `check` / `status <id>`: nothing was sent — a dim neutral line (review F3).
    Neutral,
}

/// The neutral line `rv check` / `rv status <id>` print for a server without the field.
pub(crate) const NOT_REPORTED: &str = "sampling profile: not reported by this server (< 0.9.1)";

/// Prints a registered model's sampling profile as the server reports it (V5c): the profile for a
/// text model, the old-server note when its entry has no `sampling` field, nothing for other
/// kinds (P9/F2). Returns what it found.
pub(crate) fn print_entry_profile<'a>(
    ctx: &Ctx,
    model: &str,
    entry: &'a serde_json::Value,
    old: OldServerNote,
) -> crate::sampling::EntryProfile<'a> {
    use crate::sampling::{EntryProfile, OLD_SERVER_WARNING, Origin, entry_profile, print, render};
    let found = entry_profile(entry);
    match (&found, old) {
        (EntryProfile::Profile(s), _) => print(ctx.style, &render(model, s, Origin::Server)),
        (EntryProfile::OldServer, OldServerNote::Warning) => {
            eprintln!(
                "{} {}",
                ctx.style.warn(),
                ctx.style.yellow(OLD_SERVER_WARNING)
            );
        }
        (EntryProfile::OldServer, OldServerNote::Neutral) => {
            println!("{}", ctx.style.dim(NOT_REPORTED));
        }
        (EntryProfile::NotApplicable, _) => {}
    }
    found
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
