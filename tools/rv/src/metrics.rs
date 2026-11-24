//! Minimal Prometheus text parser. Port of `parse_metrics` / `metric_by_label`
//! in `scripts/rv` — deliberately the same shape, including that [`parse`] skips
//! lines with no `{labels}` block (what `rv ps` and `rv status` read).
//! [`parse_all`] keeps them too (`rv metrics`).

use std::collections::HashMap;

/// One labelled sample.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    /// Metric name.
    pub name: String,
    /// Label set.
    pub labels: HashMap<String, String>,
    /// Sample value.
    pub value: f64,
}

/// Parses the labelled samples out of a Prometheus text exposition.
#[must_use]
pub fn parse(text: &str) -> Vec<Sample> {
    text.lines().filter_map(parse_line).collect()
}

/// Every sample of a Prometheus text exposition, the unlabelled ones (`gpu_poisoned`,
/// `ov_cache_*`, the `realtime_*` totals) included, each with the line it came from (trimmed).
/// Lines [`parse`] would read come out the same.
#[must_use]
pub fn parse_all(text: &str) -> Vec<(Sample, &str)> {
    text.lines()
        .filter_map(|l| {
            parse_line(l)
                .or_else(|| parse_unlabelled(l))
                .map(|s| (s, l.trim()))
        })
        .collect()
}

/// `name value` (no label block), as the exporter writes an unlabelled series.
fn parse_unlabelled(line: &str) -> Option<Sample> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') || line.contains('{') {
        return None;
    }
    let mut parts = line.split_whitespace();
    let name = parts.next()?;
    if !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }
    let value: f64 = parts.next()?.parse().ok().filter(|v: &f64| v.is_finite())?;
    if parts.next().is_some() {
        return None;
    }
    Some(Sample {
        name: name.to_owned(),
        labels: HashMap::new(),
        value,
    })
}

fn parse_line(line: &str) -> Option<Sample> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let open = line.find('{')?;
    let name = &line[..open];
    if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }
    let close = line[open..].find('}')? + open;
    // Python's value pattern `[\d.eE+\-]+` cannot match NaN/Inf, so those
    // samples are skipped there; `f64::parse` would accept them.
    let value: f64 = line[close + 1..]
        .trim()
        .parse()
        .ok()
        .filter(|v: &f64| v.is_finite())?;
    Some(Sample {
        name: name.to_owned(),
        labels: parse_labels(&line[open + 1..close]),
        value,
    })
}

/// Parses `a="x",b="y"` pairs (values contain no `"`, as in the Python regex).
fn parse_labels(s: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let mut rest = s;
    while let Some(eq) = rest.find("=\"") {
        let key = rest[..eq]
            .rsplit(|c: char| !(c.is_alphanumeric() || c == '_'))
            .next()
            .unwrap_or("");
        let after = &rest[eq + 2..];
        let Some(end) = after.find('"') else { break };
        if !key.is_empty() {
            out.insert(key.to_owned(), after[..end].to_owned());
        }
        rest = &after[end + 1..];
    }
    out
}

/// First sample named `name` whose labels include every `(key, value)` in `want`.
#[must_use]
pub fn by_label(samples: &[Sample], name: &str, want: &[(&str, &str)]) -> Option<f64> {
    samples
        .iter()
        .find(|s| {
            s.name == name
                && want
                    .iter()
                    .all(|(k, v)| s.labels.get(*k).is_some_and(|x| x == v))
        })
        .map(|s| s.value)
}

/// `model`'s sample named `name` on its current `device`. The server keeps a series per
/// `(model, device)` after a model moves, with its last value, so the first `model` match can be
/// another device's (round LFO). When no `model` series carries a `device` label (an older
/// server), the first one. When `device` is `None`, the series' only device, else `None`.
#[must_use]
pub fn by_model_on_device(
    samples: &[Sample],
    name: &str,
    model: &str,
    device: Option<&str>,
) -> Option<f64> {
    let series: Vec<&Sample> = samples
        .iter()
        .filter(|s| s.name == name && s.labels.get("model").is_some_and(|m| m == model))
        .collect();
    let mut devices: Vec<&str> = series
        .iter()
        .filter_map(|s| s.labels.get("device").map(String::as_str))
        .collect();
    if devices.is_empty() {
        return series.first().map(|s| s.value);
    }
    devices.sort_unstable();
    devices.dedup();
    let device = device.or(if let [one] = devices.as_slice() {
        Some(*one)
    } else {
        None
    })?;
    series
        .iter()
        .find(|s| s.labels.get("device").is_some_and(|d| d == device))
        .map(|s| s.value)
}

/// The shown form of a `key_suffix` label value: `…` and its last 2 characters (the server keeps
/// 6), so output pasted into a report does not carry key material.
#[must_use]
pub fn mask_key_suffix(value: &str) -> String {
    let tail: String = {
        let chars: Vec<char> = value.chars().collect();
        chars[chars.len().saturating_sub(2)..].iter().collect()
    };
    if value.chars().count() <= 2 {
        "\u{2026}".to_owned()
    } else {
        format!("\u{2026}{tail}")
    }
}

/// `line` with every `key_suffix="..."` value masked ([`mask_key_suffix`]).
#[must_use]
pub fn mask_line(line: &str) -> String {
    const KEY: &str = "key_suffix=\"";
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(at) = rest.find(KEY) {
        // Only a whole label name: the character before must not continue an identifier.
        let whole = rest[..at]
            .chars()
            .next_back()
            .is_none_or(|c| !(c.is_alphanumeric() || c == '_'));
        let value_start = at + KEY.len();
        let Some(len) = rest[value_start..].find('"') else {
            break;
        };
        out.push_str(&rest[..value_start]);
        let value = &rest[value_start..value_start + len];
        if whole {
            out.push_str(&mask_key_suffix(value));
        } else {
            out.push_str(value);
        }
        rest = &rest[value_start + len..];
    }
    out.push_str(rest);
    out
}

/// The p50 of one model's histogram `family` (its `_bucket` series), summed over every other
/// label (`device`, `kind`, `modality`), interpolated linearly inside the bucket the way
/// Prometheus' `histogram_quantile` does. `None` when the model has no bucket sample or no
/// observation; [`Quantile::Above`] when the median falls in the `+Inf` bucket.
#[must_use]
pub fn histogram_p50(samples: &[Sample], family: &str, model: &str) -> Option<Quantile> {
    histogram_p50_on(samples, family, model, None)
}

/// [`histogram_p50`] over the series whose `device` label is `device` only (every device when
/// `None`): a model that ran on two devices keeps a histogram per device.
#[must_use]
pub fn histogram_p50_on(
    samples: &[Sample],
    family: &str,
    model: &str,
    device: Option<&str>,
) -> Option<Quantile> {
    let bucket = format!("{family}_bucket");
    let mut by_le: Vec<(f64, f64)> = Vec::new();
    for s in samples.iter().filter(|s| {
        s.name == bucket
            && s.labels.get("model").is_some_and(|m| m == model)
            && device.is_none_or(|d| s.labels.get("device").is_some_and(|x| x == d))
    }) {
        let Some(le) = s.labels.get("le").and_then(|v| parse_le(v)) else {
            continue;
        };
        if let Some(slot) = by_le.iter_mut().find(|(b, _)| b.total_cmp(&le).is_eq()) {
            slot.1 += s.value;
        } else {
            by_le.push((le, s.value));
        }
    }
    by_le.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total = by_le.last().map(|(_, c)| *c)?;
    if total <= 0.0 {
        return None;
    }
    let rank = total * 0.5;
    let (mut lower, mut below) = (0.0_f64, 0.0_f64);
    for (upper, cum) in &by_le {
        if *cum >= rank {
            if upper.is_infinite() {
                return Some(Quantile::Above(lower));
            }
            let in_bucket = cum - below;
            let frac = if in_bucket > 0.0 {
                (rank - below) / in_bucket
            } else {
                1.0
            };
            return Some(Quantile::Value(lower + (upper - lower) * frac));
        }
        lower = *upper;
        below = *cum;
    }
    None
}

/// A histogram quantile: a value, or "above the last finite bucket bound".
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Quantile {
    /// Interpolated inside a finite bucket.
    Value(f64),
    /// In the `+Inf` bucket: above this (the last finite) bound.
    Above(f64),
}

fn parse_le(v: &str) -> Option<f64> {
    if v == "+Inf" {
        Some(f64::INFINITY)
    } else {
        v.parse().ok().filter(|x: &f64| x.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "# HELP x\n# TYPE x gauge\n\
rustedvino_vram_total_bytes{device=\"GPU\",domain=\"system\"} 27000000000\n\
rustedvino_vram_used_bytes{device=\"GPU\"} -0\n\
rustedvino_requests_running{model=\"a-b\"} 3\n\
unlabelled_metric 5\n\
rustedvino_kv{model=\"m\"} 1.5e1\n";

    #[test]
    fn parses_labelled_samples_and_skips_the_rest() {
        let rows = parse(SAMPLE);
        assert_eq!(rows.len(), 4, "{rows:?}");
        assert_eq!(
            by_label(&rows, "rustedvino_vram_total_bytes", &[("device", "GPU")]),
            Some(27_000_000_000.0)
        );
        assert_eq!(
            by_label(
                &rows,
                "rustedvino_vram_total_bytes",
                &[("domain", "system")]
            ),
            Some(27_000_000_000.0)
        );
        assert_eq!(
            by_label(&rows, "rustedvino_requests_running", &[("model", "a-b")]),
            Some(3.0)
        );
        assert_eq!(
            by_label(&rows, "rustedvino_kv", &[("model", "m")]),
            Some(15.0)
        );
        assert_eq!(
            by_label(&rows, "rustedvino_requests_running", &[("model", "zz")]),
            None
        );
        assert_eq!(by_label(&rows, "unlabelled_metric", &[]), None);
    }

    /// `parse` (rv ps / rv status) still skips unlabelled lines; `parse_all` (rv metrics) keeps
    /// them, and agrees with `parse` on every labelled one.
    #[test]
    fn parse_all_adds_the_unlabelled_samples_and_nothing_else() {
        let all = parse_all(SAMPLE);
        assert_eq!(all.len(), 5, "{all:?}");
        let unlabelled: Vec<_> = all.iter().filter(|(s, _)| s.labels.is_empty()).collect();
        assert_eq!(unlabelled.len(), 1);
        assert_eq!(unlabelled[0].0.name, "unlabelled_metric");
        assert!((unlabelled[0].0.value - 5.0).abs() < f64::EPSILON);
        assert_eq!(unlabelled[0].1, "unlabelled_metric 5");
        let labelled: Vec<Sample> = all
            .into_iter()
            .filter(|(s, _)| !s.labels.is_empty())
            .map(|(s, _)| s)
            .collect();
        assert_eq!(labelled, parse(SAMPLE));
        assert!(parse_all("x NaN\ny 1 2\n# z 3\n").is_empty());
    }

    #[test]
    fn key_suffix_values_are_masked_to_their_last_two_characters() {
        assert_eq!(mask_key_suffix("b7c4e9"), "\u{2026}e9");
        assert_eq!(mask_key_suffix("ab"), "\u{2026}");
        assert_eq!(
            mask_line("rustedvino_key_usage_total{key_suffix=\"b7c4e9\"} 3"),
            "rustedvino_key_usage_total{key_suffix=\"\u{2026}e9\"} 3"
        );
        assert_eq!(
            mask_line("a{x=\"1\",key_suffix=\"aaaaaa\"} 1 b{key_suffix=\"bbbbbb\"}"),
            "a{x=\"1\",key_suffix=\"\u{2026}aa\"} 1 b{key_suffix=\"\u{2026}bb\"}"
        );
        assert_eq!(
            mask_line("a{other_key_suffix=\"keepme\"} 1"),
            "a{other_key_suffix=\"keepme\"} 1"
        );
    }

    #[test]
    fn p50_is_interpolated_from_the_buckets_summed_over_modality() {
        let text = "\
f_bucket{model=\"m\",modality=\"text\",le=\"100\"} 0\n\
f_bucket{model=\"m\",modality=\"text\",le=\"200\"} 2\n\
f_bucket{model=\"m\",modality=\"text\",le=\"+Inf\"} 2\n\
f_bucket{model=\"m\",modality=\"image\",le=\"100\"} 0\n\
f_bucket{model=\"m\",modality=\"image\",le=\"200\"} 2\n\
f_bucket{model=\"m\",modality=\"image\",le=\"+Inf\"} 2\n\
f_bucket{model=\"hi\",le=\"100\"} 0\n\
f_bucket{model=\"hi\",le=\"+Inf\"} 3\n\
f_bucket{model=\"none\",le=\"100\"} 0\n\
f_bucket{model=\"none\",le=\"+Inf\"} 0\n";
        let rows = parse(text);
        // 4 observations, all in (100, 200]: rank 2 of 4 → halfway → 150.
        assert_eq!(histogram_p50(&rows, "f", "m"), Some(Quantile::Value(150.0)));
        assert_eq!(
            histogram_p50(&rows, "f", "hi"),
            Some(Quantile::Above(100.0))
        );
        assert_eq!(histogram_p50(&rows, "f", "none"), None);
        assert_eq!(histogram_p50(&rows, "f", "absent"), None);
    }

    #[test]
    fn p50_on_one_device_reads_only_that_device() {
        let text = "\
f_bucket{model=\"m\",device=\"CPU\",le=\"100\"} 4\n\
f_bucket{model=\"m\",device=\"CPU\",le=\"200\"} 4\n\
f_bucket{model=\"m\",device=\"CPU\",le=\"+Inf\"} 4\n\
f_bucket{model=\"m\",device=\"GPU.0\",le=\"100\"} 0\n\
f_bucket{model=\"m\",device=\"GPU.0\",le=\"200\"} 2\n\
f_bucket{model=\"m\",device=\"GPU.0\",le=\"+Inf\"} 2\n";
        let rows = parse(text);
        assert_eq!(
            histogram_p50_on(&rows, "f", "m", Some("GPU.0")),
            Some(Quantile::Value(150.0))
        );
        assert_eq!(
            histogram_p50_on(&rows, "f", "m", Some("CPU")),
            Some(Quantile::Value(50.0))
        );
        assert_eq!(histogram_p50_on(&rows, "f", "m", Some("NPU")), None);
        // Every device: 6 observations, rank 3 inside (0, 100] (4 there) → 75.
        assert_eq!(histogram_p50(&rows, "f", "m"), Some(Quantile::Value(75.0)));
    }

    #[test]
    fn nan_and_inf_samples_are_skipped() {
        let rows = parse("a{m=\"x\"} NaN\nb{m=\"x\"} +Inf\nc{m=\"x\"} -Inf\nd{m=\"x\"} 2\n");
        assert_eq!(rows.len(), 1, "{rows:?}");
        assert_eq!(rows[0].name, "d");
    }

    #[test]
    fn negative_zero_parses() {
        let rows = parse("m{a=\"b\"} -0\n");
        assert_eq!(rows.len(), 1);
        assert!(rows[0].value.abs() < f64::EPSILON);
    }
}
