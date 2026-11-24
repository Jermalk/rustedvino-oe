//! `rv metrics [--grep PATTERN] [--json]`: the server's `/metrics`, fetched with rv's admin key,
//! with one speed line per loaded model first (V1-rv minimal P3). No `--watch`.

use std::collections::HashMap;
use std::hash::BuildHasher;

use serde_json::{Map, Value, json};

use crate::commands::print_error;
use crate::ctx::Ctx;
use crate::metrics::{self, Quantile, Sample};

/// The effective prefill histogram (0.9.1): prompt tokens ÷ (first token − engine
/// start), prefix-cache hits included, prompts of 256 tokens or more only.
const PREFILL: &str = "rustedvino_prefill_tokens_per_second";
/// The decode throughput gauge (an EMA per engine).
const DECODE: &str = "rustedvino_tokens_per_second";
/// 1 while a model is loaded.
const LOADED: &str = "rustedvino_model_loaded";

/// One loaded model's speed summary.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    /// Model id.
    pub model: String,
    /// The effective prefill p50 (tok/s), `None` when this model has no observation.
    pub prefill_p50: Option<Quantile>,
    /// The decode throughput (tok/s), `None` when absent.
    pub decode: Option<f64>,
    /// `false` when the server has no prefill series at all (older than 0.9.1).
    pub prefill_known: bool,
    /// The device `prefill_p50` and `decode` are for: the model's current device (the admin
    /// model list, else the device of its only `rustedvino_model_loaded == 1` series). `None`
    /// when it cannot be told; then the two figures are `None` too whenever the model has
    /// series on more than one device, and the lists below carry them all.
    pub device: Option<String>,
    /// Every device's decode figure, sorted by device. The server keeps a series per
    /// `(model, device)` after the model moves, with its last value.
    pub decode_by_device: Vec<(String, f64)>,
    /// Every device's effective prefill p50, sorted by device.
    pub prefill_by_device: Vec<(String, Quantile)>,
}

impl Summary {
    /// `true` when the current device is unknown and the model has figures on several devices:
    /// the summary line then shows each device's figure instead of one.
    fn ambiguous<T>(&self, by_device: &[T]) -> bool {
        self.device.is_none() && by_device.len() > 1
    }
}

/// The summaries of every model with `rustedvino_model_loaded == 1`, sorted by id, the current
/// device taken from the samples alone ([`summarize_with`] with no admin list).
#[must_use]
pub fn summarize(samples: &[Sample]) -> Vec<Summary> {
    summarize_with(samples, &HashMap::new())
}

/// [`summarize`], with `current` (model id → device, from `GET /v1/admin/models`) naming the
/// device each model is on now, so a figure left over from another device is not shown as its.
#[must_use]
pub fn summarize_with<H: BuildHasher>(
    samples: &[Sample],
    current: &HashMap<String, String, H>,
) -> Vec<Summary> {
    let prefill_known = samples.iter().any(|s| s.name.starts_with(PREFILL));
    let mut models: Vec<&str> = samples
        .iter()
        .filter(|s| s.name == LOADED && s.value > 0.5)
        .filter_map(|s| s.labels.get("model").map(String::as_str))
        .collect();
    models.sort_unstable();
    models.dedup();
    models
        .into_iter()
        .map(|m| {
            let device = current
                .get(m)
                .cloned()
                .or_else(|| sole_loaded_device(samples, m));
            let decode_by_device = decode_per_device(samples, m);
            let prefill_devices = bucket_devices(samples, m);
            let prefill_by_device: Vec<(String, Quantile)> = prefill_devices
                .iter()
                .filter_map(|d| {
                    metrics::histogram_p50_on(samples, PREFILL, m, Some(d)).map(|q| (d.clone(), q))
                })
                .collect();
            let decode = if decode_by_device.is_empty() {
                // No series carries a device label (an older server): the one figure there is.
                metrics::by_label(samples, DECODE, &[("model", m)])
            } else {
                match device.as_deref() {
                    Some(d) => decode_by_device
                        .iter()
                        .find(|(x, _)| x == d)
                        .map(|(_, v)| *v),
                    None if decode_by_device.len() == 1 => Some(decode_by_device[0].1),
                    None => None,
                }
            };
            let prefill_p50 = if prefill_devices.is_empty() {
                // No bucket carries a device label (or there is none): as before, over all.
                metrics::histogram_p50(samples, PREFILL, m)
            } else {
                match device.as_deref() {
                    Some(d) => metrics::histogram_p50_on(samples, PREFILL, m, Some(d)),
                    None if prefill_by_device.len() == 1 => Some(prefill_by_device[0].1),
                    None => None,
                }
            };
            Summary {
                model: m.to_owned(),
                prefill_p50,
                decode,
                prefill_known,
                device,
                decode_by_device,
                prefill_by_device,
            }
        })
        .collect()
}

/// The `kind` label of `model`'s first `rustedvino_model_loaded == 1` series.
fn loaded_kind<'a>(samples: &'a [Sample], model: &str) -> Option<&'a str> {
    samples
        .iter()
        .filter(|s| s.name == LOADED && s.value > 0.5)
        .filter(|s| s.labels.get("model").is_some_and(|x| x == model))
        .find_map(|s| s.labels.get("kind").map(String::as_str))
}

/// The device of `model`'s `rustedvino_model_loaded == 1` series, when exactly one device has one.
pub(crate) fn sole_loaded_device(samples: &[Sample], model: &str) -> Option<String> {
    let mut devices: Vec<&str> = samples
        .iter()
        .filter(|s| s.name == LOADED && s.value > 0.5)
        .filter(|s| s.labels.get("model").is_some_and(|x| x == model))
        .filter_map(|s| s.labels.get("device").map(String::as_str))
        .collect();
    devices.sort_unstable();
    devices.dedup();
    match devices.as_slice() {
        [one] => Some((*one).to_owned()),
        _ => None,
    }
}

/// `model`'s decode gauge per device label, sorted by device (series without one left out).
fn decode_per_device(samples: &[Sample], model: &str) -> Vec<(String, f64)> {
    let mut out: Vec<(String, f64)> = Vec::new();
    for s in samples
        .iter()
        .filter(|s| s.name == DECODE && s.labels.get("model").is_some_and(|x| x == model))
    {
        if let Some(d) = s.labels.get("device")
            && !out.iter().any(|(x, _)| x == d)
        {
            out.push((d.clone(), s.value));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// The device labels of `model`'s effective prefill buckets, sorted, unique.
fn bucket_devices(samples: &[Sample], model: &str) -> Vec<String> {
    let bucket = format!("{PREFILL}_bucket");
    let mut out: Vec<String> = samples
        .iter()
        .filter(|s| s.name == bucket && s.labels.get("model").is_some_and(|x| x == model))
        .filter_map(|s| s.labels.get("device").cloned())
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn quantile_text(q: Quantile) -> String {
    match q {
        Quantile::Value(v) => format!("~{v:.0} tok/s (p50)"),
        Quantile::Above(v) => format!("> {v:.0} tok/s (p50)"),
    }
}

/// `<model>  effective prefill ~N tok/s (p50) · decode M tok/s`. When the model's current device
/// is unknown and it has figures on several devices, each device's figure with its name:
/// `decode 26.3 tok/s (CPU), 11.9 tok/s (GPU.0)`.
#[must_use]
pub fn summary_line(s: &Summary) -> String {
    let prefill = if !s.prefill_known {
        "unknown".to_owned()
    } else if s.ambiguous(&s.prefill_by_device) {
        s.prefill_by_device
            .iter()
            .map(|(d, q)| format!("{} ({d})", quantile_text(*q)))
            .collect::<Vec<_>>()
            .join(", ")
    } else {
        s.prefill_p50.map_or_else(|| "-".to_owned(), quantile_text)
    };
    let decode = if s.ambiguous(&s.decode_by_device) {
        s.decode_by_device
            .iter()
            .map(|(d, v)| format!("{v:.1} tok/s ({d})"))
            .collect::<Vec<_>>()
            .join(", ")
    } else {
        s.decode
            .map_or_else(|| "-".to_owned(), |v| format!("{v:.1} tok/s"))
    };
    format!(
        "{}  effective prefill {prefill} \u{b7} decode {decode}",
        s.model
    )
}

fn quantile_json(q: Option<Quantile>) -> (Value, bool) {
    match q {
        Some(Quantile::Value(v)) => (json!(v), false),
        Some(Quantile::Above(v)) => (json!(v), true),
        None => (Value::Null, false),
    }
}

fn summary_json(s: &Summary) -> Value {
    let (p50, above) = quantile_json(s.prefill_p50);
    let decode_by_device: Map<String, Value> = s
        .decode_by_device
        .iter()
        .map(|(d, v)| (d.clone(), json!(v)))
        .collect();
    let prefill_by_device: Map<String, Value> = s
        .prefill_by_device
        .iter()
        .map(|(d, q)| {
            let (v, above) = quantile_json(Some(*q));
            (d.clone(), json!({ "tok_s": v, "above": above }))
        })
        .collect();
    json!({
        "model": s.model,
        "device": s.device,
        "effective_prefill_known": s.prefill_known,
        "effective_prefill_p50_tok_s": p50,
        "effective_prefill_p50_above": above,
        "effective_prefill_p50_by_device": prefill_by_device,
        "decode_tok_s": s.decode,
        "decode_tok_s_by_device": decode_by_device,
    })
}

fn sample_json(s: &Sample) -> Value {
    let mut keys: Vec<&String> = s.labels.keys().collect();
    keys.sort();
    let mut labels = Map::new();
    for k in keys {
        let v = &s.labels[k];
        let shown = if k == "key_suffix" {
            metrics::mask_key_suffix(v)
        } else {
            v.clone()
        };
        labels.insert(k.clone(), Value::String(shown));
    }
    json!({ "name": s.name, "labels": labels, "value": s.value })
}

/// Renders the whole output for a `/metrics` body: the text lines, or the `--json` value.
/// `grep` filters the raw samples (substring of the masked line), never the summary.
///
/// # Errors
/// Only if the `--json` value cannot be serialised.
pub fn render(text: &str, grep: Option<&str>, as_json: bool) -> Result<String, String> {
    render_with(text, &HashMap::new(), grep, as_json)
}

/// [`render`] with each model's current device (`current`: model id → device, from the admin
/// model list) for the summary lines.
///
/// # Errors
/// Only if the `--json` value cannot be serialised.
pub fn render_with<H: BuildHasher>(
    text: &str,
    current: &HashMap<String, String, H>,
    grep: Option<&str>,
    as_json: bool,
) -> Result<String, String> {
    let all = metrics::parse_all(text);
    let samples: Vec<Sample> = all.iter().map(|(s, _)| s.clone()).collect();
    let summaries = summarize_with(&samples, current);
    let shown: Vec<(&Sample, String)> = all
        .iter()
        .map(|(s, line)| (s, metrics::mask_line(line)))
        .filter(|(_, line)| grep.is_none_or(|g| line.contains(g)))
        .collect();
    if as_json {
        let v = json!({
            "models": summaries.iter().map(summary_json).collect::<Vec<_>>(),
            "samples": shown.iter().map(|(s, _)| sample_json(s)).collect::<Vec<_>>(),
        });
        return serde_json::to_string_pretty(&v).map_err(|e| e.to_string());
    }
    let mut out = String::new();
    // The text lines in table order (kind, then id); `--json` above keeps `summarize`'s id order.
    let mut lines: Vec<&Summary> = summaries.iter().collect();
    crate::order::sort_models(&mut lines, |s| (loaded_kind(&samples, &s.model), &s.model));
    for s in lines {
        out.push_str(&summary_line(s));
        out.push('\n');
    }
    if !summaries.is_empty() {
        out.push('\n');
    }
    for (_, line) in &shown {
        out.push_str(line);
        out.push('\n');
    }
    Ok(out)
}

/// Model id → the device it is placed on, from `GET /v1/admin/models`; empty when the list is not
/// available (the summary then falls back to what `/metrics` alone can tell).
fn current_devices(ctx: &Ctx) -> HashMap<String, String> {
    let r = ctx.api.get("/v1/admin/models");
    r.json()
        .filter(|_| r.code == Some(200))
        .and_then(|b| b.get("models"))
        .and_then(Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter_map(|m| {
                    let id = m.get("id")?.as_str()?;
                    let device = m.get("device")?.as_str().filter(|d| !d.is_empty())?;
                    Some((id.to_owned(), device.to_owned()))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Runs `rv metrics`.
#[must_use]
pub fn run(ctx: &Ctx, grep: Option<&str>) -> i32 {
    if !ctx.require_admin_key() {
        return 1;
    }
    let r = ctx.api.get("/metrics");
    let Some(text) = r
        .body
        .as_ref()
        .and_then(crate::api::Payload::text)
        .filter(|_| r.code == Some(200))
    else {
        print_error(ctx, &r);
        return 1;
    };
    match render_with(text, &current_devices(ctx), grep, ctx.json) {
        Ok(out) => {
            print!("{out}");
            0
        }
        Err(e) => {
            eprintln!("{}", ctx.style.red(&format!("error: {e}")));
            1
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // fixtures: a failed render should fail the test loudly
mod tests {
    use super::*;

    const BODY: &str = "# HELP rustedvino_model_loaded x\n\
rustedvino_model_loaded{model=\"qwen\",device=\"GPU.1\",kind=\"text_gen\"} 1\n\
rustedvino_model_loaded{model=\"gone\",device=\"GPU.1\",kind=\"text_gen\"} 0\n\
rustedvino_model_loaded{model=\"emb\",device=\"GPU.0\",kind=\"embedding\"} 1\n\
rustedvino_tokens_per_second{model=\"qwen\",device=\"GPU.1\",kind=\"text_gen\"} 41.25\n\
rustedvino_prefill_tokens_per_second_bucket{model=\"qwen\",device=\"GPU.1\",kind=\"text_gen\",modality=\"text\",le=\"1000\"} 1\n\
rustedvino_prefill_tokens_per_second_bucket{model=\"qwen\",device=\"GPU.1\",kind=\"text_gen\",modality=\"text\",le=\"2000\"} 3\n\
rustedvino_prefill_tokens_per_second_bucket{model=\"qwen\",device=\"GPU.1\",kind=\"text_gen\",modality=\"text\",le=\"+Inf\"} 3\n\
rustedvino_prefill_tokens_per_second_bucket{model=\"qwen\",device=\"GPU.1\",kind=\"text_gen\",modality=\"image\",le=\"1000\"} 0\n\
rustedvino_prefill_tokens_per_second_bucket{model=\"qwen\",device=\"GPU.1\",kind=\"text_gen\",modality=\"image\",le=\"2000\"} 1\n\
rustedvino_prefill_tokens_per_second_bucket{model=\"qwen\",device=\"GPU.1\",kind=\"text_gen\",modality=\"image\",le=\"+Inf\"} 1\n\
rustedvino_key_usage_total{key_suffix=\"b7c4e9\"} 12\n\
rustedvino_gpu_poisoned 0\n";

    #[test]
    fn summary_lines_per_loaded_model() {
        let out = render(BODY, None, false).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        // 4 observations: 1 in (0,1000], 3 in (1000,2000]; rank 2 → 1000 + 1000 * (1/3).
        // Round RVS: by kind (text_gen before embedding), then id.
        assert_eq!(
            lines[0], "qwen  effective prefill ~1333 tok/s (p50) \u{b7} decode 41.2 tok/s",
            "{out}"
        );
        assert_eq!(
            lines[1], "emb  effective prefill - \u{b7} decode -",
            "{out}"
        );
        assert_eq!(lines[2], "");
        assert!(
            out.contains("rustedvino_gpu_poisoned 0"),
            "unlabelled kept: {out}"
        );
        assert!(!out.contains("gone  "), "not loaded: {out}");
        // `--json` keeps the id order it always had.
        let v: Value = serde_json::from_str(&render(BODY, None, true).unwrap()).unwrap();
        assert_eq!(v["models"][0]["model"], "emb");
        assert_eq!(v["models"][1]["model"], "qwen");
    }

    /// A model that ran on the CPU and now runs on `GPU.0`: the server keeps both devices' series
    /// (the CPU ones with their last values, `model_loaded` on the CPU possibly still 1).
    const MOVED: &str = "\
rustedvino_model_loaded{model=\"m\",device=\"CPU\",kind=\"text_gen\"} 1\n\
rustedvino_model_loaded{model=\"m\",device=\"GPU.0\",kind=\"text_gen\"} 1\n\
rustedvino_tokens_per_second{model=\"m\",device=\"CPU\",kind=\"text_gen\"} 26.3\n\
rustedvino_tokens_per_second{model=\"m\",device=\"GPU.0\",kind=\"text_gen\"} 11.9\n\
rustedvino_prefill_tokens_per_second_bucket{model=\"m\",device=\"CPU\",kind=\"text_gen\",modality=\"text\",le=\"100\"} 4\n\
rustedvino_prefill_tokens_per_second_bucket{model=\"m\",device=\"CPU\",kind=\"text_gen\",modality=\"text\",le=\"200\"} 4\n\
rustedvino_prefill_tokens_per_second_bucket{model=\"m\",device=\"CPU\",kind=\"text_gen\",modality=\"text\",le=\"+Inf\"} 4\n\
rustedvino_prefill_tokens_per_second_bucket{model=\"m\",device=\"GPU.0\",kind=\"text_gen\",modality=\"text\",le=\"100\"} 0\n\
rustedvino_prefill_tokens_per_second_bucket{model=\"m\",device=\"GPU.0\",kind=\"text_gen\",modality=\"text\",le=\"200\"} 2\n\
rustedvino_prefill_tokens_per_second_bucket{model=\"m\",device=\"GPU.0\",kind=\"text_gen\",modality=\"text\",le=\"+Inf\"} 2\n";

    fn on(model: &str, device: &str) -> HashMap<String, String> {
        HashMap::from([(model.to_owned(), device.to_owned())])
    }

    #[test]
    fn single_device_line_is_the_same_with_or_without_the_admin_list() {
        let plain = render(BODY, None, false).unwrap();
        let mut current = on("qwen", "GPU.1");
        current.insert("emb".to_owned(), "GPU.0".to_owned());
        assert_eq!(render_with(BODY, &current, None, false).unwrap(), plain);
        let v: Value = serde_json::from_str(&render(BODY, Some("x"), true).unwrap()).unwrap();
        // No admin list, one loaded series: the device comes from it.
        assert_eq!(v["models"][1]["device"], "GPU.1");
        assert_eq!(
            v["models"][1]["decode_tok_s_by_device"],
            json!({"GPU.1": 41.25})
        );
    }

    #[test]
    fn two_devices_the_admin_device_picks_the_series() {
        let out = render_with(MOVED, &on("m", "GPU.0"), None, false).unwrap();
        assert!(
            out.starts_with("m  effective prefill ~150 tok/s (p50) \u{b7} decode 11.9 tok/s\n"),
            "{out}"
        );
        let out = render_with(MOVED, &on("m", "CPU"), None, false).unwrap();
        assert!(
            out.starts_with("m  effective prefill ~50 tok/s (p50) \u{b7} decode 26.3 tok/s\n"),
            "{out}"
        );
    }

    #[test]
    fn current_device_without_its_own_series_reads_dash_not_the_old_device() {
        // Just moved to GPU.0, nothing generated there yet: only the CPU's old figures exist.
        let mut body = String::new();
        for l in MOVED
            .lines()
            .filter(|l| !l.contains("GPU.0") || l.starts_with("rustedvino_model_loaded"))
        {
            body.push_str(l);
            body.push('\n');
        }
        let out = render_with(&body, &on("m", "GPU.0"), None, false).unwrap();
        assert!(
            out.starts_with("m  effective prefill - \u{b7} decode -\n"),
            "{out}"
        );
    }

    #[test]
    fn two_devices_unknown_current_shows_every_device() {
        let out = render(MOVED, None, false).unwrap();
        assert!(
            out.starts_with(
                "m  effective prefill ~50 tok/s (p50) (CPU), ~150 tok/s (p50) (GPU.0) \u{b7} \
                 decode 26.3 tok/s (CPU), 11.9 tok/s (GPU.0)\n"
            ),
            "{out}"
        );
        let v: Value = serde_json::from_str(&render(MOVED, Some("x"), true).unwrap()).unwrap();
        let m = &v["models"][0];
        assert_eq!(m["device"], Value::Null);
        assert_eq!(m["decode_tok_s"], Value::Null);
        assert_eq!(m["effective_prefill_p50_tok_s"], Value::Null);
        assert_eq!(
            m["decode_tok_s_by_device"],
            json!({"CPU": 26.3, "GPU.0": 11.9})
        );
        assert_eq!(
            m["effective_prefill_p50_by_device"],
            json!({"CPU": {"tok_s": 50.0, "above": false},
                   "GPU.0": {"tok_s": 150.0, "above": false}})
        );
    }

    #[test]
    fn without_the_admin_list_the_only_loaded_device_is_the_current_one() {
        let body = MOVED.replacen(
            "device=\"CPU\",kind=\"text_gen\"} 1",
            "device=\"CPU\",kind=\"text_gen\"} 0",
            1,
        );
        let out = render(&body, None, false).unwrap();
        assert!(
            out.starts_with("m  effective prefill ~150 tok/s (p50) \u{b7} decode 11.9 tok/s\n"),
            "{out}"
        );
        let v: Value =
            serde_json::from_str(&render_with(&body, &on("m", "GPU.0"), Some("x"), true).unwrap())
                .unwrap();
        assert_eq!(v["models"][0]["device"], "GPU.0");
        assert_eq!(v["models"][0]["decode_tok_s"], 11.9);
        assert_eq!(v["models"][0]["effective_prefill_p50_tok_s"], 150.0);
    }

    #[test]
    fn older_server_without_the_prefill_family_reads_unknown() {
        let body = "rustedvino_model_loaded{model=\"m\"} 1\n\
rustedvino_tokens_per_second{model=\"m\"} 10\n";
        let out = render(body, None, false).unwrap();
        assert!(
            out.starts_with("m  effective prefill unknown \u{b7} decode 10.0 tok/s\n"),
            "{out}"
        );
    }

    #[test]
    fn grep_filters_samples_not_the_summary() {
        let out = render(BODY, Some("tokens_per_second{"), false).unwrap();
        assert!(out.contains("qwen  effective prefill"), "{out}");
        assert!(
            out.contains("rustedvino_tokens_per_second{model=\"qwen\""),
            "{out}"
        );
        assert!(!out.contains("rustedvino_model_loaded{"), "{out}");
        assert!(!out.contains("_bucket"), "{out}");
    }

    #[test]
    fn key_suffix_is_masked_in_text_and_json() {
        let out = render(BODY, None, false).unwrap();
        assert!(!out.contains("b7c4e9"), "{out}");
        assert!(out.contains("key_suffix=\"\u{2026}e9\""), "{out}");
        // A grep for the real suffix finds nothing: matching runs on the masked line.
        let hidden = render(BODY, Some("b7c4"), false).unwrap();
        assert!(!hidden.contains("key_usage"), "{hidden}");
        let j = render(BODY, None, true).unwrap();
        assert!(!j.contains("b7c4e9"), "{j}");
    }

    #[test]
    fn json_has_models_and_samples() {
        let v: Value =
            serde_json::from_str(&render(BODY, Some("key_usage"), true).unwrap()).unwrap();
        let models = v["models"].as_array().unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[1]["model"], "qwen");
        assert_eq!(models[1]["effective_prefill_known"], true);
        assert!(
            (models[1]["effective_prefill_p50_tok_s"].as_f64().unwrap() - 1333.33).abs() < 0.01
        );
        assert_eq!(models[1]["effective_prefill_p50_above"], false);
        assert_eq!(models[1]["decode_tok_s"], 41.25);
        assert_eq!(models[0]["effective_prefill_p50_tok_s"], Value::Null);
        assert_eq!(
            v["samples"],
            json!([{"name": "rustedvino_key_usage_total",
                    "labels": {"key_suffix": "\u{2026}e9"}, "value": 12.0}])
        );
        let all: Value =
            serde_json::from_str(&render(BODY, Some("gpu_poisoned"), true).unwrap()).unwrap();
        assert_eq!(
            all["samples"],
            json!([{"name": "rustedvino_gpu_poisoned", "labels": {}, "value": 0.0}])
        );
    }
}
