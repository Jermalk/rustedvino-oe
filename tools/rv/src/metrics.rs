//! Minimal Prometheus text parser. Port of `parse_metrics` / `metric_by_label`
//! in `scripts/rv` — deliberately the same shape, including that lines with no
//! `{labels}` block are skipped.

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
