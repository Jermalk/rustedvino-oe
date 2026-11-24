//! What a forced unload, deregister, resize or load did (V1-rv P1): the line rv
//! prints before the call (requests running, drain deadline) and the outcome line after it, from
//! the server's six-key `{evicted, forced, drained_s, aborted, teardown_ms, message}` answer to
//! `Accept: application/json` (0.9.1+; rv reads `evicted`, `drained_s` and `aborted`). An older server answers plain text: the facts are then
//! "unknown".

use serde_json::Value;

use crate::api::Payload;
use crate::ctx::Ctx;
use crate::metrics::{self, Sample};

/// The drain bound of a forced unload, deregister or resize, in seconds: the server's
/// `EVICT_DRAIN_DEADLINE` (`src/model_manager/mod.rs`, 2 minutes). rv cannot ask the server for
/// it; keep the two in step.
pub const EVICT_DRAIN_S: u64 = 120;
/// The drain bound of a model a forced load evicts for room, in seconds: the server's
/// `LOAD_EVICT_DRAIN_DEADLINE` (`src/model_manager/mod.rs`, 15 s).
pub const LOAD_EVICT_DRAIN_S: u64 = 15;

/// One `/metrics` scrape, quietly: `None` when it is not available.
#[must_use]
pub fn scrape(ctx: &Ctx) -> Option<Vec<Sample>> {
    let r = ctx.api.get("/metrics");
    r.body
        .as_ref()
        .filter(|_| r.code == Some(200))
        .and_then(Payload::text)
        .map(metrics::parse)
}

/// `rustedvino_requests_running` summed over `models`; `None` when the scrape is missing or any
/// of them has no such series (a kind that does not report it, or an older server).
#[must_use]
pub fn running(samples: Option<&[Sample]>, models: &[&str]) -> Option<u64> {
    let samples = samples?;
    models.iter().try_fold(0u64, |acc, m| {
        let v = metrics::by_label(samples, "rustedvino_requests_running", &[("model", m)])?;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // a small gauge
        Some(acc + v.max(0.0) as u64)
    })
}

/// The line printed before a forced call: `N request(s) running — will drain up to <deadline>
/// then abort`.
#[must_use]
pub fn precall_line(running: Option<u64>, deadline_s: u64) -> String {
    let n = running.map_or_else(|| "unknown number of".to_owned(), |n| n.to_string());
    format!("{n} request(s) running \u{2014} will drain up to {deadline_s} s then abort")
}

/// The facts of the outcome JSON.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    /// Every model the call evicted, in order.
    pub evicted: Vec<String>,
    /// Seconds spent draining (0 = nothing was running).
    pub drained_s: f64,
    /// Requests still running at the drain deadline, on any engine kind: a text model aborts them,
    /// other engines finish or fail them during teardown.
    pub aborted: u64,
}

/// The outcome JSON of a 200, or `None` for a plain-text answer (an older server).
#[must_use]
pub fn parse(body: Option<&Payload>) -> Option<Outcome> {
    let v = body.and_then(Payload::json)?;
    Some(Outcome {
        evicted: v
            .get("evicted")?
            .as_array()?
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        drained_s: v.get("drained_s")?.as_f64()?,
        aborted: v.get("aborted")?.as_u64()?,
    })
}

/// The outcome clause: nothing was running / its request(s) finished during the drain (X s) /
/// drained X s; N request(s) still running at the drain deadline; "unknown" for `None` (an older server).
/// `running` is the count read before the call, used only to word the drained case.
#[must_use]
pub fn clause(outcome: Option<&Outcome>, running: Option<u64>) -> String {
    let Some(o) = outcome else {
        return "drain/abort unknown (the server does not report them; older than 0.9.1)"
            .to_owned();
    };
    if o.aborted > 0 {
        format!("drained {:.1} s; {}", o.drained_s, still_running(o.aborted))
    } else if o.drained_s > 0.0 {
        let its = match running {
            Some(n) if n > 0 => format!("its {n} running request(s)"),
            _ => "its running request(s)".to_owned(),
        };
        format!("{its} finished during the drain ({:.1} s)", o.drained_s)
    } else {
        "nothing was running".to_owned()
    }
}

/// What `aborted` means: the requests still running when the drain deadline
/// passed, on every engine kind.
fn still_running(n: u64) -> String {
    format!(
        "{n} request(s) still running at the drain deadline: a text model aborts them (client sees \
         model_unloading); other engines finish or fail them during teardown"
    )
}

/// Prints a warning on stderr when requests were still running at the drain deadline.
pub fn warn_if_still_running(ctx: &Ctx, outcome: Option<&Outcome>, what: &str) {
    if let Some(o) = outcome.filter(|o| o.aborted > 0) {
        eprintln!(
            "{}",
            ctx.style
                .yellow(&format!("warning: {what}: {}", still_running(o.aborted)))
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn outcome(drained_s: f64, aborted: u64) -> Outcome {
        Outcome {
            evicted: vec!["m".into()],
            drained_s,
            aborted,
        }
    }

    #[test]
    fn precall_line_names_the_count_and_the_deadline() {
        assert_eq!(
            precall_line(Some(2), EVICT_DRAIN_S),
            "2 request(s) running \u{2014} will drain up to 120 s then abort"
        );
        assert_eq!(
            precall_line(None, LOAD_EVICT_DRAIN_S),
            "unknown number of request(s) running \u{2014} will drain up to 15 s then abort"
        );
    }

    #[test]
    fn the_three_outcomes_and_the_older_server() {
        assert_eq!(
            clause(Some(&outcome(0.0, 0)), Some(0)),
            "nothing was running"
        );
        assert_eq!(
            clause(Some(&outcome(81.04, 0)), Some(1)),
            "its 1 running request(s) finished during the drain (81.0 s)"
        );
        assert_eq!(
            clause(Some(&outcome(3.0, 0)), None),
            "its running request(s) finished during the drain (3.0 s)"
        );
        assert_eq!(
            clause(Some(&outcome(120.02, 1)), Some(1)),
            "drained 120.0 s; 1 request(s) still running at the drain deadline: a text model \
             aborts them (client sees model_unloading); other engines finish or fail them during \
             teardown"
        );
        assert!(clause(None, Some(1)).contains("unknown"));
    }

    #[test]
    fn parse_reads_the_json_and_not_the_text() {
        let j = Payload::Json(
            json!({"evicted": ["a", "b"], "forced": true, "drained_s": 1.5,
                                     "aborted": 2, "teardown_ms": 700}),
        );
        assert_eq!(
            parse(Some(&j)),
            Some(Outcome {
                evicted: vec!["a".into(), "b".into()],
                drained_s: 1.5,
                aborted: 2
            })
        );
        assert_eq!(
            parse(Some(&Payload::Text("model 'a' evicted".into()))),
            None
        );
        assert_eq!(
            parse(Some(&Payload::Json(json!({"persisted": true})))),
            None
        );
        assert_eq!(parse(None), None);
    }

    #[test]
    fn running_sums_the_models_and_is_unknown_when_any_is_missing() {
        let rows = metrics::parse(
            "rustedvino_requests_running{model=\"a\"} 1\nrustedvino_requests_running{model=\"b\"} 2\n",
        );
        assert_eq!(running(Some(&rows), &["a", "b"]), Some(3));
        assert_eq!(running(Some(&rows), &[]), Some(0));
        assert_eq!(running(Some(&rows), &["a", "zz"]), None);
        assert_eq!(running(None, &["a"]), None);
    }
}
