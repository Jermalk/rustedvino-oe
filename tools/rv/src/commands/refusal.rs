//! Why a dry-run admission check (`/v1/admin/models/{id}/check`) said "does not fit": which gate
//! refuses (the device memory budget, live system RAM, or both), whether anything could have been
//! evicted, and whether a `config.json` cap is what stands in the way. Shared by `rv load` and
//! `rv check`; pure, so every branch is unit-tested.
//!
//! The relations mirror the server (`src/model_manager/mod.rs`, rc.3):
//! - `free_gb` = the domain's total − what is booked in it; the total is a `domain_budgets` entry,
//!   else `total_vram_gb` (never more than the device reports). Evicting everything frees at most
//!   the booked part, so `needed_gb` above the configured total means no eviction can make room,
//!   whatever is loaded.
//! - `ram_avail_gb` = `min(live available RAM + credit, system_ram_budget_gb)`; the gate admits
//!   when `available − ram_needed_gb ≥` the reservation (`system_ram_reservation_gb`, else the OS
//!   default), and evictions never lift availability above `min(MemTotal, budget)`. So
//!   `ram_needed_gb + reservation > system_ram_budget_gb` means the budget is the limit.
//! - The check's figures are taken before any simulated eviction. On a refusal, `would_evict` is
//!   the whole sweep (every evictable resident of the domain; the server's doc comment saying it
//!   is empty then is stale): even evicting all of them is not enough.

use std::path::Path;

use serde_json::Value;

/// Two figures this close are "the same number" (the server rounds nothing, but a JSON
/// round-trip of a config value can drift in the last bits).
const SAME_GB: f64 = 0.05;

/// The server's default OS reservation when `system_ram_reservation_gb` is unset
/// (`os_memory::default_reservation_gb`, which this crate cannot depend on). `rv` talks to a server
/// on this machine only, so its own OS is the server's.
const DEFAULT_RESERVATION_GB: f64 = if cfg!(windows) { 6.0 } else { 4.0 };

/// The memory caps `config.json` sets (absent or `null` keys are `None`).
#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct Caps {
    /// `total_vram_gb`.
    pub total_vram_gb: Option<f64>,
    /// `system_ram_budget_gb`.
    pub system_ram_budget_gb: Option<f64>,
    /// `system_ram_reservation_gb`.
    pub system_ram_reservation_gb: Option<f64>,
    /// `domain_budgets` entries, by domain.
    pub domain_budgets: Vec<(String, f64)>,
}

impl Caps {
    /// The caps of a parsed config object.
    pub(crate) fn from_value(cfg: &Value) -> Self {
        let num = |k: &str| cfg.get(k).and_then(Value::as_f64);
        let mut domain_budgets: Vec<(String, f64)> = cfg
            .get("domain_budgets")
            .and_then(Value::as_object)
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_f64().map(|g| (k.clone(), g)))
                    .collect()
            })
            .unwrap_or_default();
        domain_budgets.sort_by(|a, b| a.0.cmp(&b.0));
        Self {
            total_vram_gb: num("total_vram_gb"),
            system_ram_budget_gb: num("system_ram_budget_gb"),
            system_ram_reservation_gb: num("system_ram_reservation_gb"),
            domain_budgets,
        }
    }

    /// The caps of the config file at `path`; `None` when it is missing, unreadable or not a JSON
    /// object. Read on the refusal path only, so an edit since `rv` started is seen.
    pub(crate) fn read(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        let cfg: Value = serde_json::from_str(crate::resolve::strip_bom(&text)).ok()?;
        cfg.is_object().then(|| Self::from_value(&cfg))
    }
}

/// What `rv` can say about a "does not fit" check, beyond its figures.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Refusal {
    /// The refusing gate(s): "GPU budget", "system RAM", "GPU budget and system RAM", or
    /// "GPU budget and/or system RAM" when both are short but an eviction might satisfy either.
    pub gates: &'static str,
    /// Why evicting does not help: "nothing is loaded to evict", "even after evicting all N
    /// evictable models" (the check's sweep list), "no eviction can make room" (a config cap
    /// proves it), or `None` when `rv` cannot tell.
    pub eviction: Option<String>,
    /// Whether a config cap is proven to be the limit (then the generic "pinned, …" causes do not
    /// apply).
    pub cap_proven: bool,
    /// The refusing gates' figures, e.g. `GPU budget: needs 22.0 GB, 18.0 GB free`.
    pub figures: String,
    /// `limit: config.json sets …` when a config cap is proven to be the limit, else the hedged
    /// `the budget comes from config.json (…)` when a cap is set but not proven; `None` otherwise.
    pub config_line: Option<String>,
}

/// `18.0`, `17.5`: a GB figure as config.json would hold it.
fn gb(v: f64) -> String {
    format!("{v:.1}")
}

/// The reservation the RAM gate keeps free after a load: `system_ram_reservation_gb`, else the
/// OS default (`caps` `None`: the config is unreadable, so the default is a guess).
pub(crate) fn ram_floor_gb(caps: Option<&Caps>) -> f64 {
    caps.and_then(|c| c.system_ram_reservation_gb)
        .unwrap_or(DEFAULT_RESERVATION_GB)
}

/// `2.0 GB reserve`, or `4.0 GB default reserve` when the config sets none (or is unreadable):
/// then the figure is `rv`'s copy of the server's OS default, not a value read anywhere.
pub(crate) fn reserve_text(caps: Option<&Caps>) -> String {
    let set = caps.and_then(|c| c.system_ram_reservation_gb).is_some();
    let default = if set { "" } else { "default " };
    format!("{} GB {default}reserve", gb(ram_floor_gb(caps)))
}

/// The config key whose value is the refusing domain's total, proven to be the limit by
/// `needed_gb` above it (evicting everything frees at most what is booked, so no eviction can
/// make room). On the UMA `system` domain (the RAM fields are present) a
/// `domain_budgets["system"]` entry, when set, is that total, else `total_vram_gb` is. Elsewhere
/// `rv` does not know the domain id, so a cap proves nothing unless the config sets exactly one
/// domain total: with two, the need can exceed one domain's cap while another, larger and partly
/// booked domain is the one refusing (e.g. `domain_budgets {"GPU.1": 20}` and `total_vram_gb 18`).
/// A sanity bound guards the guess of which key is the total: `free_gb` can never exceed the
/// domain total, so a `free_gb` above the cap proves the cap is not this domain's.
fn vram_cap_key(caps: &Caps, needed: f64, free: f64, system_domain: bool) -> Option<(String, f64)> {
    let limit = |v: f64| v > 0.0 && needed > v + 1e-9 && free <= v + SAME_GB;
    if system_domain {
        if let Some((_, v)) = caps.domain_budgets.iter().find(|(d, _)| d == "system") {
            return limit(*v).then(|| ("domain_budgets[\"system\"]".to_owned(), *v));
        }
        return caps
            .total_vram_gb
            .filter(|v| limit(*v))
            .map(|v| ("total_vram_gb".to_owned(), v));
    }
    let mut set: Vec<(String, f64)> = caps
        .domain_budgets
        .iter()
        .filter(|(_, v)| *v > 0.0)
        .map(|(d, v)| (format!("domain_budgets[\"{d}\"]"), *v))
        .collect();
    if let Some(v) = caps.total_vram_gb.filter(|v| *v > 0.0) {
        set.push(("total_vram_gb".to_owned(), v));
    }
    (set.len() == 1 && limit(set[0].1)).then(|| set.remove(0))
}

/// The check's sweep list: on a refusal, every evictable resident the server would have evicted
/// and still not fit (empty when nothing evictable is in the way).
pub(crate) fn sweep_len(check: &Value) -> usize {
    check["would_evict"].as_array().map_or(0, Vec::len)
}

/// `even after evicting all 5 evictable models` / `… the only evictable model`.
pub(crate) fn sweep_phrase(n: usize) -> String {
    if n == 1 {
        "even after evicting the only evictable model".to_owned()
    } else {
        format!("even after evicting all {n} evictable models")
    }
}

/// Analyses a check whose verdict is "does not fit" (no headroom refusal, nothing blocked by a busy
/// model — those keep their own messages). `caps`: the config's caps (`None`: unreadable).
/// `nothing_loaded`: `Some(true)` when the server lists no model in any loaded state, `None` when
/// that is unknown. `None` back when neither gate refuses by `rv`'s arithmetic (it cannot tell).
pub(crate) fn analyse(
    check: &Value,
    caps: Option<&Caps>,
    nothing_loaded: Option<bool>,
) -> Option<Refusal> {
    let needed = check["needed_gb"].as_f64().unwrap_or(0.0);
    let free = check["free_gb"].as_f64().unwrap_or(0.0);
    // Older servers omit `vram_gated`: gated.
    let vram_refuses = check["vram_gated"].as_bool().unwrap_or(true) && needed > free;
    let ram = match (
        check["ram_needed_gb"].as_f64(),
        check["ram_avail_gb"].as_f64(),
    ) {
        (Some(n), Some(a)) => Some((n, a)),
        _ => None,
    };
    let floor = ram_floor_gb(caps);
    let ram_refuses = ram.is_some_and(|(n, a)| a - n < floor);
    if !vram_refuses && !ram_refuses {
        return None;
    }

    let mut figures = Vec::new();
    if vram_refuses {
        figures.push(format!(
            "GPU budget: needs {} GB, {} GB free",
            gb(needed),
            gb(free)
        ));
    }
    if let Some((n, a)) = ram.filter(|_| ram_refuses) {
        figures.push(format!(
            "RAM: needs {} GB + {}, {} GB available",
            gb(n),
            reserve_text(caps),
            gb(a)
        ));
    }

    // Proven caps: (key, value) per gate.
    let vram_cap = caps
        .filter(|_| vram_refuses)
        .and_then(|c| vram_cap_key(c, needed, free, ram.is_some()));
    // `a ≤ b` always holds against this server (availability is capped at the budget); it guards
    // a config edited since the server started.
    let ram_cap = caps
        .filter(|_| ram_refuses)
        .and_then(|c| c.system_ram_budget_gb)
        .filter(|b| *b > 0.0 && ram.is_some_and(|(n, a)| n + floor > *b && a <= b + SAME_GB));
    let mut proven: Vec<String> = Vec::new();
    if let Some((k, v)) = &vram_cap {
        proven.push(format!("{k} {}", gb(*v)));
    }
    if let Some(b) = ram_cap {
        proven.push(format!("system_ram_budget_gb {}", gb(b)));
    }
    let config_line = if proven.is_empty() {
        hedge(
            caps,
            vram_refuses.then_some(needed),
            ram.filter(|_| ram_refuses),
            floor,
        )
    } else {
        let line = format!(
            "limit: config.json sets {} — raise to load this model",
            proven.join(" / ")
        );
        // ≤ 100 columns with the two-space indent: two long keys drop the tail's last words.
        Some(if line.chars().count() <= 98 {
            line
        } else {
            format!(
                "limit: config.json sets {} — raise to load",
                proven.join(" / ")
            )
        })
    };

    // The figures are taken before any simulated eviction, and an eviction only frees memory: the
    // gates short now include every gate that still refuses after evicting all it can, and may
    // include one an eviction would satisfy. A short gate surely refuses when nothing is loaded,
    // when its config cap is proven, or when it is the only one short (else the check would fit).
    let nothing = nothing_loaded == Some(true);
    let vram_sure = vram_refuses && (nothing || vram_cap.is_some() || !ram_refuses);
    let ram_sure = ram_refuses && (nothing || ram_cap.is_some() || !vram_refuses);
    let gates = match (vram_sure, ram_sure) {
        (true, true) => "GPU budget and system RAM",
        (true, false) => "GPU budget",
        (false, true) => "system RAM",
        (false, false) => "GPU budget and/or system RAM",
    };

    let cap_proven = vram_cap.is_some() || ram_cap.is_some();
    let sweep = sweep_len(check);
    let eviction = if sweep > 0 {
        Some(sweep_phrase(sweep))
    } else if nothing {
        Some("nothing is loaded to evict".to_owned())
    } else if cap_proven {
        Some("no eviction can make room".to_owned())
    } else {
        None
    };
    Some(Refusal {
        gates,
        eviction,
        cap_proven,
        figures: figures.join(" · "),
        config_line,
    })
}

/// The hedged config line: a refusing gate's need is above a cap config.json sets, but `rv`
/// cannot prove that cap is the domain's limit (something is booked there, or the live figure is
/// below the cap). It names the caps without claiming they are the cause.
fn hedge(
    caps: Option<&Caps>,
    vram_need: Option<f64>,
    ram: Option<(f64, f64)>,
    floor: f64,
) -> Option<String> {
    let c = caps?;
    let mut keys = Vec::new();
    if let Some(n) = vram_need {
        if let Some(v) = c.total_vram_gb.filter(|v| *v > 0.0 && n > *v) {
            keys.push(format!("total_vram_gb {}", gb(v)));
        }
        for (d, v) in c.domain_budgets.iter().filter(|(_, v)| *v > 0.0 && n > *v) {
            keys.push(format!("domain_budgets[\"{d}\"] {}", gb(*v)));
        }
    }
    if let Some((n, _)) = ram
        && let Some(b) = c.system_ram_budget_gb.filter(|b| n + floor > *b)
    {
        keys.push(format!("system_ram_budget_gb {}", gb(b)));
    }
    (!keys.is_empty()).then(|| format!("the budget comes from config.json ({})", keys.join(", ")))
}

/// `Some(true)` when every model `GET /v1/admin/models` lists is `not_loaded` (nothing could be
/// evicted), `Some(false)` when any is in another state, `None` when the list is unavailable.
pub(crate) fn nothing_loaded(models: Option<&Value>) -> Option<bool> {
    let list = models?.get("models")?.as_array()?;
    Some(
        list.iter()
            .all(|m| m.get("state").and_then(Value::as_str) == Some("not_loaded")),
    )
}

/// [`nothing_loaded`] from the live server (one admin GET; on the refusal path only).
pub(crate) fn nothing_loaded_now(ctx: &crate::ctx::Ctx) -> Option<bool> {
    let r = ctx.api.get("/v1/admin/models");
    nothing_loaded(r.json().filter(|_| r.code == Some(200)))
}

#[cfg(test)]
#[allow(clippy::expect_used)] // a refusal the test expects and does not get should fail loudly
mod tests {
    use super::*;
    use serde_json::json;

    /// The live record (2026-11-21, 30 GB UMA laptop): the 35B with nothing loaded, 18/17 caps.
    fn live_35b() -> Value {
        json!({"fits": false, "needed_gb": 22.0, "free_gb": 18.0, "vram_gated": true,
               "would_evict": [], "busy_victims": [], "blocked_by_busy": [],
               "ram_needed_gb": 24.0, "ram_avail_gb": 17.0, "already_loaded": false})
    }

    fn live_caps() -> Caps {
        Caps::from_value(&json!({"total_vram_gb": 18.0, "system_ram_budget_gb": 17.0,
                                 "system_ram_reservation_gb": 2.0}))
    }

    #[test]
    fn live_35b_names_both_gates_both_caps_and_that_nothing_is_loaded() {
        let r = analyse(&live_35b(), Some(&live_caps()), Some(true)).expect("refusal");
        assert_eq!(r.gates, "GPU budget and system RAM");
        assert_eq!(r.eviction.as_deref(), Some("nothing is loaded to evict"));
        assert_eq!(
            r.figures,
            "GPU budget: needs 22.0 GB, 18.0 GB free · RAM: needs 24.0 GB + 2.0 GB reserve, \
             17.0 GB available"
        );
        assert_eq!(
            r.config_line.as_deref(),
            Some(
                "limit: config.json sets total_vram_gb 18.0 / system_ram_budget_gb 17.0 — raise \
                 to load this model"
            )
        );
    }

    #[test]
    fn domain_gate_only_on_a_discrete_gpu() {
        let c = json!({"fits": false, "needed_gb": 9.0, "free_gb": 3.0, "vram_gated": true});
        let r = analyse(&c, None, None).expect("refusal");
        assert_eq!(r.gates, "GPU budget");
        assert_eq!(r.figures, "GPU budget: needs 9.0 GB, 3.0 GB free");
        assert_eq!(
            r.eviction.as_deref(),
            None,
            "unknown evictability stays unknown"
        );
        assert_eq!(r.config_line, None);
    }

    #[test]
    fn ram_gate_only_counts_the_reservation() {
        // 15 needed, 17 available looks like it fits, but 4 GB must stay free (the default floor
        // when the config sets none).
        let c = json!({"fits": false, "needed_gb": 12.0, "free_gb": 20.0,
                       "ram_needed_gb": 15.0, "ram_avail_gb": 17.0});
        let caps = Caps::from_value(&json!({"total_vram_gb": 0.0}));
        let r = analyse(&c, Some(&caps), Some(false)).expect("refusal");
        assert_eq!(r.gates, "system RAM");
        assert_eq!(
            r.figures,
            format!(
                "RAM: needs 15.0 GB + {} GB default reserve, 17.0 GB available",
                gb(DEFAULT_RESERVATION_GB)
            )
        );
        assert_eq!(r.eviction.as_deref(), None);
        assert_eq!(r.config_line, None);
    }

    #[test]
    fn ungated_domain_never_blames_the_gpu_budget() {
        let c = json!({"fits": false, "needed_gb": 22.0, "free_gb": 0.0, "vram_gated": false,
                       "ram_needed_gb": 24.0, "ram_avail_gb": 20.0});
        let r = analyse(&c, None, None).expect("refusal");
        assert_eq!(r.gates, "system RAM");
    }

    #[test]
    fn nothing_refusing_by_rvs_arithmetic_is_left_to_the_generic_text() {
        let c = json!({"fits": false, "needed_gb": 4.0, "free_gb": 9.0});
        assert_eq!(analyse(&c, Some(&live_caps()), Some(true)), None);
    }

    #[test]
    fn live_loaded_35b_proves_both_caps_whatever_is_loaded() {
        // The coordinator's live check (2026-11-21): five models loaded, the sweep is all of
        // them, and still 22 > 18 and 24 + 2 > 17.
        let c = json!({"fits": false, "needed_gb": 22.0, "free_gb": 1.0068, "vram_gated": true,
                       "would_evict": ["multilingual-e5-large-int8", "bge-reranker-v2-m3-int8-ov",
                                       "whisper-large-v3-turbo-int8-ov", "speecht5-tts-ov",
                                       "qwen3.5-9B-int4-ov"],
                       "busy_victims": [], "blocked_by_busy": [],
                       "ram_needed_gb": 24.0, "ram_avail_gb": 10.96, "already_loaded": false});
        let r = analyse(&c, Some(&live_caps()), Some(false)).expect("refusal");
        assert_eq!(r.gates, "GPU budget and system RAM");
        assert!(r.cap_proven);
        assert_eq!(
            r.eviction.as_deref(),
            Some("even after evicting all 5 evictable models")
        );
        assert_eq!(
            r.config_line.as_deref(),
            Some(
                "limit: config.json sets total_vram_gb 18.0 / system_ram_budget_gb 17.0 — raise \
                 to load this model"
            )
        );
    }

    #[test]
    fn both_short_with_residents_and_no_cap_is_and_or() {
        // 12 > 1 free of 18 and 10 + 2 > 11 available of 17: either gate might be satisfied by
        // evicting the two residents, and neither cap explains it.
        let c = json!({"fits": false, "needed_gb": 12.0, "free_gb": 1.0,
                       "would_evict": ["a", "b"], "ram_needed_gb": 10.0, "ram_avail_gb": 11.0});
        let r = analyse(&c, Some(&live_caps()), Some(false)).expect("refusal");
        assert_eq!(r.gates, "GPU budget and/or system RAM");
        assert!(!r.cap_proven);
        assert_eq!(
            r.eviction.as_deref(),
            Some("even after evicting all 2 evictable models")
        );
        assert_eq!(r.config_line, None);
    }

    #[test]
    fn cap_not_matched_or_config_unreadable_names_no_key() {
        // Caps well above the need: no key is named.
        let big = Caps::from_value(&json!({"total_vram_gb": 27.0, "system_ram_budget_gb": 26.0,
                                           "system_ram_reservation_gb": 2.0}));
        let mut c = live_35b();
        c["free_gb"] = json!(10.0);
        c["ram_avail_gb"] = json!(12.0);
        let r = analyse(&c, Some(&big), Some(false)).expect("refusal");
        assert_eq!(r.config_line, None);
        assert_eq!(r.eviction.as_deref(), None);
        // Unreadable config: the gates are still named, no config line.
        let r = analyse(&live_35b(), None, Some(true)).expect("refusal");
        assert_eq!(r.gates, "GPU budget and system RAM");
        assert_eq!(r.config_line, None);
        assert_eq!(r.eviction.as_deref(), Some("nothing is loaded to evict"));
    }

    #[test]
    fn a_proven_cap_says_no_eviction_can_help_when_residents_are_unknown() {
        let r = analyse(&live_35b(), Some(&live_caps()), None).expect("refusal");
        assert_eq!(r.eviction.as_deref(), Some("no eviction can make room"));
    }

    #[test]
    fn domain_budgets_system_overrides_total_vram_gb_on_uma() {
        let caps = Caps::from_value(&json!({"total_vram_gb": 18.0,
                                            "domain_budgets": {"system": 18.0}}));
        let r = analyse(&live_35b(), Some(&caps), None).expect("refusal");
        assert_eq!(
            r.config_line.as_deref(),
            Some(
                "limit: config.json sets domain_budgets[\"system\"] 18.0 — raise to load this model"
            )
        );
    }

    #[test]
    fn two_keys_with_the_free_value_on_a_discrete_gpu_are_ambiguous() {
        let caps = Caps::from_value(&json!({"total_vram_gb": 16.0,
                                            "domain_budgets": {"GPU.1": 16.0}}));
        let c = json!({"fits": false, "needed_gb": 20.0, "free_gb": 16.0});
        let r = analyse(&c, Some(&caps), None).expect("refusal");
        assert!(
            r.config_line
                .as_deref()
                .is_some_and(|l| l.starts_with("the budget comes from config.json")),
            "{r:?}"
        );
        assert_eq!(r.eviction.as_deref(), None);
    }

    #[test]
    fn a_discrete_total_vram_match_beside_a_domain_budget_proves_nothing() {
        // GPU.1's own budget is 20 with 2 booked: free 18 equals total_vram_gb by coincidence.
        let caps = Caps::from_value(&json!({"total_vram_gb": 18.0,
                                            "domain_budgets": {"GPU.1": 20.0}}));
        let c = json!({"fits": false, "needed_gb": 22.0, "free_gb": 18.0});
        let r = analyse(&c, Some(&caps), Some(false)).expect("refusal");
        assert_eq!(r.eviction.as_deref(), None);
        assert_eq!(
            r.config_line.as_deref(),
            Some(
                "the budget comes from config.json (total_vram_gb 18.0, domain_budgets[\"GPU.1\"] \
                 20.0)"
            )
        );
        // The same figures with total_vram_gb the only cap set: proven.
        let only = Caps::from_value(&json!({"total_vram_gb": 18.0}));
        let r = analyse(&c, Some(&only), Some(false)).expect("refusal");
        assert_eq!(r.gates, "GPU budget");
        assert_eq!(r.eviction.as_deref(), Some("no eviction can make room"));
        assert!(
            r.config_line
                .as_deref()
                .is_some_and(|l| l.starts_with("limit: config.json sets total_vram_gb 18.0")),
            "{r:?}"
        );
    }

    #[test]
    fn free_above_the_cap_shows_the_cap_is_not_this_domains() {
        // A CPU model's `system` domain on a dGPU box: its total is RAM − reservation, not
        // total_vram_gb; 40 GB free beside a 16 GB total_vram_gb proves that.
        let caps = Caps::from_value(&json!({"total_vram_gb": 16.0}));
        let c = json!({"fits": false, "needed_gb": 50.0, "free_gb": 40.0,
                       "ram_needed_gb": 5.0, "ram_avail_gb": 50.0});
        let r = analyse(&c, Some(&caps), Some(false)).expect("refusal");
        assert!(!r.cap_proven, "{r:?}");
        assert!(
            r.config_line
                .as_deref()
                .is_some_and(|l| l.starts_with("the budget comes from config.json")),
            "{r:?}"
        );
    }

    #[test]
    fn one_proven_gate_is_named_alone_when_the_other_is_unsure() {
        // RAM budget proven (24 + 2 > 17); the GPU need 15 is under its 18 cap but over the 10
        // free: an eviction might satisfy that gate, so it is not named.
        let mut c = live_35b();
        c["needed_gb"] = json!(15.0);
        c["free_gb"] = json!(10.0);
        let r = analyse(&c, Some(&live_caps()), Some(false)).expect("refusal");
        assert_eq!(r.gates, "system RAM");
        assert_eq!(r.eviction.as_deref(), Some("no eviction can make room"));
        assert_eq!(
            r.config_line.as_deref(),
            Some("limit: config.json sets system_ram_budget_gb 17.0 — raise to load this model")
        );
    }

    #[test]
    fn nothing_loaded_reads_every_state() {
        let none = json!({"models": [{"id": "a", "state": "not_loaded"}]});
        let one = json!({"models": [{"id": "a", "state": "not_loaded"},
                                    {"id": "b", "state": "loading"}]});
        assert_eq!(nothing_loaded(Some(&none)), Some(true));
        assert_eq!(nothing_loaded(Some(&one)), Some(false));
        assert_eq!(nothing_loaded(Some(&json!({"error": "x"}))), None);
        assert_eq!(nothing_loaded(None), None);
    }

    #[test]
    fn caps_skip_null_and_non_numbers() {
        let c = Caps::from_value(&json!({"total_vram_gb": null, "system_ram_budget_gb": "x",
                                         "domain_budgets": {"system": 3.0, "GPU.1": null}}));
        assert_eq!(c.total_vram_gb, None);
        assert_eq!(c.system_ram_budget_gb, None);
        assert_eq!(c.domain_budgets, vec![("system".to_owned(), 3.0)]);
    }
}
