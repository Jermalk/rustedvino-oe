//! `rv check MODEL` — dry-run admission check.

use std::path::Path;

use crate::api::{READ_TIMEOUT, encode_segment};
use crate::commands::{OldServerNote, admin_entry_quiet, print_entry_profile, refusal};
use crate::ctx::Ctx;
use crate::kind::detect_kind_from_markers;
use crate::sampling::{
    self, Operator, Origin, kind_has_sampling, preview_json, read_file_layer, render,
};

/// The local preview for a model the server does not know (V5c): the profile its own
/// `generation_config.json` gives, when the directory is on disk under `models_dir` and its
/// files do not prove a non-text kind (review F2).
fn print_local_preview(ctx: &Ctx, model: &str) {
    let Some(dir) = ctx.res.models_dir.as_deref().filter(|d| !d.is_empty()) else {
        return;
    };
    let model_dir = Path::new(dir).join(model);
    if detect_kind_from_markers(&model_dir).is_some_and(|k| !kind_has_sampling(k)) {
        return;
    }
    if let Some(file) = read_file_layer(&model_dir) {
        let origin = Origin::Local {
            file_found: file.found,
        };
        let p = render(
            model,
            &preview_json(&Operator::default(), &file, false),
            origin,
        );
        sampling::print(ctx.style, &p);
    }
}

/// The registered model's profile as the server reports it.
fn print_server_profile(ctx: &Ctx, model: &str) {
    if let Some(entry) = admin_entry_quiet(ctx, model) {
        print_entry_profile(ctx, model, &entry, OldServerNote::Neutral);
    }
}

/// The no-fit line when `rv` cannot say more.
const PLAIN: &str = "does not fit even after evicting everything evictable";

/// Why a model does not fit, after the figures: the refusing gate, why nothing was evicted, and
/// the `config.json` cap when it is proven to be the limit (RVR, see [`refusal`]). A device-headroom
/// refusal or a busy-held room keeps the plain line (their own lines follow / the load says it).
/// `caps`: the config's caps (`None`: unreadable).
fn no_fit_lines(ctx: &Ctx, body: &serde_json::Value, caps: Option<&refusal::Caps>) -> Vec<String> {
    let busy = body["blocked_by_busy"]
        .as_array()
        .is_some_and(|a| !a.is_empty());
    if body["headroom_refusal"].is_string() || busy {
        return plain_lines(body);
    }
    no_fit_text(body, caps, refusal::nothing_loaded_now(ctx))
}

/// The plain no-fit line, and (the check's sweep list is non-empty) that evicting all of it would
/// not be enough — on a refusal `would_evict` is not what a load would evict.
fn plain_lines(body: &serde_json::Value) -> Vec<String> {
    let mut lines = vec![format!("  {PLAIN}")];
    let n = refusal::sweep_len(body);
    if n == 1 {
        lines.push("  evicting the only evictable model would still not be enough".to_owned());
    } else if n > 1 {
        lines.push(format!(
            "  evicting all {n} evictable models would still not be enough"
        ));
    }
    lines
}

/// [`no_fit_lines`] without the I/O.
fn no_fit_text(
    body: &serde_json::Value,
    caps: Option<&refusal::Caps>,
    nothing_loaded: Option<bool>,
) -> Vec<String> {
    let Some(r) = refusal::analyse(body, caps, nothing_loaded) else {
        return plain_lines(body);
    };
    // The eviction phrase carries the sweep ("even after evicting all 5 evictable models").
    let mut lines = vec![format!(
        "  refused by: {} — {}",
        r.gates,
        r.eviction.as_deref().unwrap_or(PLAIN)
    )];
    if let Some(l) = r.config_line {
        lines.push(format!("  {l}"));
    }
    lines
}

/// Shared-memory (UMA) boxes gate on live system RAM as well as VRAM, and it is the RAM gate that
/// refuses loads there. Absent on discrete-GPU servers and on servers older than this field pair.
/// On a refusal the line shows the reservation the gate keeps free after the load (RVR), so
/// "needs 15, 17 available" refused follows from the figures printed.
fn print_ram_line(
    s: crate::ansi::Style,
    body: &serde_json::Value,
    fits: bool,
    caps: Option<&refusal::Caps>,
) {
    let (Some(need), Some(avail)) = (
        body["ram_needed_gb"].as_f64(),
        body["ram_avail_gb"].as_f64(),
    ) else {
        return;
    };
    if fits {
        println!("  system RAM: needs {need:.1} GB, {avail:.1} GB available");
    } else {
        let reserve = refusal::reserve_text(caps);
        println!(
            "{}",
            s.yellow(&format!(
                "  system RAM: needs {need:.1} GB + {reserve}, {avail:.1} GB available"
            ))
        );
    }
}

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
        // 0.8.3 (F5): an on-disk model that is not registered yet answers 404.
        if r.code == Some(404) {
            eprintln!(
                "  = help: if '{model}' is on disk, register it with `rv add {model}` \
                 (`rv list` shows models on disk that are not registered)"
            );
            if !ctx.json {
                print_local_preview(ctx, model);
            }
        }
        return 1;
    };
    if ctx.json {
        println!("{}", serde_json::to_string_pretty(body).unwrap_or_default());
        return 0;
    }
    let s = ctx.style;
    if body["already_loaded"].as_bool().unwrap_or(false) {
        println!("{} '{model}' is already loaded", s.ok());
        print_server_profile(ctx, model);
        return 0;
    }
    let fits = body["fits"].as_bool().unwrap_or(false);
    let glyph = if fits { s.ok() } else { s.bad() };
    // `vram_gated: false` = no VRAM budget on that domain (`total_vram_gb: 0.0`): the "free" figure
    // is a meaningless 0.0 and only system RAM decides. Older servers omit the key: gated.
    let free_text = if body["vram_gated"].as_bool().unwrap_or(true) {
        format!("{:.1} GB free", body["free_gb"].as_f64().unwrap_or(0.0))
    } else {
        "no VRAM limit set".to_owned()
    };
    println!(
        "{glyph} '{model}' — needs {:.1} GB, {free_text}",
        body["needed_gb"].as_f64().unwrap_or(0.0),
    );
    // RVR: on a refusal the config's caps explain it (read on this path only, so an edit since
    // `rv` started is seen; `None`: unreadable).
    let caps = if fits {
        None
    } else {
        refusal::Caps::read(&ctx.res.config_file)
    };
    print_ram_line(s, body, fits, caps.as_ref());
    let evict: Vec<&str> = body["would_evict"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    // On a refusal `would_evict` is the server's whole sweep, which a load would NOT evict (the
    // check still does not fit after it): the no-fit lines say so instead (RVR, Fault A).
    if fits && !evict.is_empty() {
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
        for line in no_fit_lines(ctx, body, caps.as_ref()) {
            println!("{}", s.yellow(&line));
        }
    }
    // The device-headroom gate (vision models on a discrete GPU). Older servers omit both keys.
    if let Some(why) = body["headroom_refusal"].as_str() {
        println!("{}", s.yellow(&format!("  device headroom: {why}")));
    }
    if let Some(why) = body["headroom_not_previewed"].as_str() {
        println!(
            "{}",
            s.yellow(&format!("  device headroom not previewed: {why}"))
        );
    }
    print_server_profile(ctx, model);
    i32::from(!fits)
}

#[cfg(test)]
mod tests {
    use super::{PLAIN, no_fit_text, refusal};
    use serde_json::json;

    #[test]
    fn live_35b_no_fit_lines_name_both_gates_and_the_caps() {
        let body = json!({"fits": false, "needed_gb": 22.0, "free_gb": 18.0, "vram_gated": true,
                          "would_evict": [], "busy_victims": [], "blocked_by_busy": [],
                          "ram_needed_gb": 24.0, "ram_avail_gb": 17.0, "already_loaded": false});
        let caps = refusal::Caps::from_value(&json!({"total_vram_gb": 18.0,
                                                     "system_ram_budget_gb": 17.0,
                                                     "system_ram_reservation_gb": 2.0}));
        assert_eq!(
            no_fit_text(&body, Some(&caps), Some(true)),
            vec![
                "  refused by: GPU budget and system RAM — nothing is loaded to evict".to_owned(),
                "  limit: config.json sets total_vram_gb 18.0 / system_ram_budget_gb 17.0 — raise \
                 to load this model"
                    .to_owned(),
            ]
        );
    }

    #[test]
    fn unknown_evictability_keeps_the_plain_sentence_after_the_gate() {
        let body = json!({"fits": false, "needed_gb": 40.0, "free_gb": 3.0});
        assert_eq!(
            no_fit_text(&body, None, None),
            vec![format!("  refused by: GPU budget — {PLAIN}")]
        );
    }

    #[test]
    fn live_loaded_35b_no_fit_lines_name_the_sweep_and_the_caps() {
        let body = json!({"fits": false, "needed_gb": 22.0, "free_gb": 1.0068, "vram_gated": true,
                          "would_evict": ["e5", "bge", "whisper", "speecht5", "qwen9b"],
                          "busy_victims": [], "blocked_by_busy": [],
                          "ram_needed_gb": 24.0, "ram_avail_gb": 10.96, "already_loaded": false});
        let caps = refusal::Caps::from_value(&json!({"total_vram_gb": 18.0,
                                                     "system_ram_budget_gb": 17.0,
                                                     "system_ram_reservation_gb": 2.0}));
        assert_eq!(
            no_fit_text(&body, Some(&caps), Some(false)),
            vec![
                "  refused by: GPU budget and system RAM — even after evicting all 5 evictable \
                 models"
                    .to_owned(),
                "  limit: config.json sets total_vram_gb 18.0 / system_ram_budget_gb 17.0 — raise \
                 to load this model"
                    .to_owned(),
            ]
        );
    }

    #[test]
    fn a_plain_refusal_with_a_sweep_says_evicting_it_is_not_enough() {
        let body = json!({"fits": false, "needed_gb": 4.0, "free_gb": 9.0,
                          "would_evict": ["a", "b", "c"]});
        assert_eq!(
            no_fit_text(&body, None, Some(false)),
            vec![
                format!("  {PLAIN}"),
                "  evicting all 3 evictable models would still not be enough".to_owned(),
            ]
        );
    }

    #[test]
    fn no_gate_by_rvs_arithmetic_is_the_plain_line() {
        let body = json!({"fits": false, "needed_gb": 4.0, "free_gb": 9.0});
        assert_eq!(
            no_fit_text(&body, None, Some(true)),
            vec![format!("  {PLAIN}")]
        );
    }
}
