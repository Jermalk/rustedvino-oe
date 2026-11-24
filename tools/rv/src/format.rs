//! Formatting helpers. Port of the "formatting helpers" block in `scripts/rv`.

use std::fmt::Write;

use crate::ansi::Style;

/// Bytes in one GiB. The server's memory unit is the GiB (0.9.0): `vram_gb`, `total_vram_gb`
/// and the `rustedvino_vram_*_bytes` metrics (`GiB × 2^30`) all use it (R-12).
pub const GIB: f64 = 1_073_741_824.0;

/// `"12.3 GB"` — bytes in GiB (`n / 2^30`, R-12), one decimal place: the unit of the server's
/// budgets (`vram_gb`, `total_vram_gb`). Labelled "GB" here, as `rv status` and the other
/// memory views show it; `rv add`'s estimate lines say "GiB" (0.9.1 R-12).
#[must_use]
pub fn human_bytes(n: f64) -> String {
    format!("{:.1} GB", n / GIB)
}

/// Python's `round()` on a float: ties go to the even neighbour, unlike Rust's
/// `f64::round`. Only matters for the bar's fill width, but a golden diff
/// against the Python output should not flap on a tie.
#[must_use]
pub fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() - 0.5 == 0.0 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// `[####......]  35%`, coloured by how full it is.
#[must_use]
pub fn bar(style: Style, used: f64, total: f64) -> String {
    const WIDTH: usize = 20;
    if total <= 0.0 {
        return style.dim("n/a");
    }
    let frac = (used / total).clamp(0.0, 1.0);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let filled = round_half_even(frac * WIDTH as f64) as usize;
    let glyph = format!("{}{}", "#".repeat(filled), ".".repeat(WIDTH - filled));
    let pct = frac * 100.0;
    let colored = if pct < 75.0 {
        style.green(&glyph)
    } else if pct < 92.0 {
        style.yellow(&glyph)
    } else {
        style.red(&glyph)
    };
    format!("[{colored}] {pct:4.0}%")
}

/// Python's `str(timedelta(seconds=n))`: `H:MM:SS`, or `N day(s), H:MM:SS`.
#[must_use]
pub fn fmt_duration(secs: u64) -> String {
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    match days {
        0 => format!("{h}:{m:02}:{s:02}"),
        1 => format!("1 day, {h}:{m:02}:{s:02}"),
        d => format!("{d} days, {h}:{m:02}:{s:02}"),
    }
}

/// Colours a model state word.
#[must_use]
pub fn state_glyph(style: Style, state: &str) -> String {
    match state {
        "ready" => style.green(state),
        "loading" | "evicting" => style.yellow(state),
        "not_loaded" => style.dim(state),
        "stalled" => style.red(state),
        s if s.starts_with("unavailable") => style.red(state),
        _ => state.to_owned(),
    }
}

/// A left-aligned table with a dashed rule under the header. `colored_col`
/// names a column holding model states, coloured via [`state_glyph`].
/// Widths are computed from the plain text, so colour never skews alignment.
#[must_use]
pub fn table(
    style: Style,
    headers: &[&str],
    rows: &[Vec<String>],
    colored_col: Option<usize>,
) -> String {
    if rows.is_empty() {
        return format!("{}\n", style.dim("  (none)"));
    }
    let width = |i: usize| {
        rows.iter()
            .map(|r| r[i].chars().count())
            .chain(std::iter::once(headers[i].chars().count()))
            .max()
            .unwrap_or(0)
    };
    let widths: Vec<usize> = (0..headers.len()).map(width).collect();
    let mut out = String::new();
    let head: Vec<String> = headers
        .iter()
        .zip(&widths)
        .map(|(h, w)| format!("{h:<w$}"))
        .collect();
    let _ = writeln!(out, "  {}", head.join("  "));
    let rule: Vec<String> = widths.iter().map(|w| style.dim(&"-".repeat(*w))).collect();
    let _ = writeln!(out, "  {}", rule.join("  "));
    for r in rows {
        let cells: Vec<String> = r
            .iter()
            .zip(&widths)
            .enumerate()
            .map(|(i, (val, w))| {
                if colored_col == Some(i) {
                    let pad = w.saturating_sub(val.chars().count());
                    format!("{}{}", state_glyph(style, val), " ".repeat(pad))
                } else {
                    format!("{val:<w$}")
                }
            })
            .collect();
        let _ = writeln!(out, "  {}", cells.join("  "));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_bytes_is_gib() {
        // R-12: a 24 GiB budget exported as 24 × 2^30 bytes reads 24.0, not 25.8.
        assert_eq!(human_bytes(24.0 * GIB), "24.0 GB");
        assert_eq!(human_bytes(0.0), "0.0 GB");
        assert_eq!(human_bytes(1.55 * GIB), "1.6 GB");
        assert_eq!(human_bytes(27e9), "25.1 GB");
    }

    #[test]
    fn rounding_ties_go_to_even_like_python() {
        assert!((round_half_even(0.5)).abs() < f64::EPSILON);
        assert!((round_half_even(1.5) - 2.0).abs() < f64::EPSILON);
        assert!((round_half_even(2.5) - 2.0).abs() < f64::EPSILON);
        assert!((round_half_even(3.5) - 4.0).abs() < f64::EPSILON);
        assert!((round_half_even(7.2) - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn bar_shapes() {
        let p = Style::plain();
        assert_eq!(bar(p, 0.0, 100.0), "[....................]    0%");
        assert_eq!(bar(p, 50.0, 100.0), "[##########..........]   50%");
        assert_eq!(bar(p, 200.0, 100.0), "[####################]  100%");
        assert_eq!(bar(p, 1.0, 0.0), "n/a");
    }

    #[test]
    fn duration_matches_timedelta_str() {
        assert_eq!(fmt_duration(0), "0:00:00");
        assert_eq!(fmt_duration(3661), "1:01:01");
        assert_eq!(fmt_duration(86_400 + 5), "1 day, 0:00:05");
        assert_eq!(fmt_duration(2 * 86_400 + 3600), "2 days, 1:00:00");
    }

    #[test]
    fn table_aligns_on_plain_width_even_when_coloured() {
        let rows = vec![
            vec!["a".to_owned(), "ready".to_owned()],
            vec!["bbb".to_owned(), "not_loaded".to_owned()],
        ];
        let plain = table(Style::plain(), &["ID", "STATE"], &rows, Some(1));
        let coloured = table(Style::colored(), &["ID", "STATE"], &rows, Some(1));
        let strip = |s: &str| {
            let mut out = String::new();
            let mut in_esc = false;
            for c in s.chars() {
                match (in_esc, c) {
                    (false, '\x1b') => in_esc = true,
                    (true, 'm') => in_esc = false,
                    (false, _) => out.push(c),
                    _ => {}
                }
            }
            out
        };
        assert_eq!(strip(&coloured), plain);
        assert!(plain.starts_with("  ID   STATE"), "{plain:?}");
    }

    #[test]
    fn empty_table_says_none() {
        assert_eq!(table(Style::plain(), &["A"], &[], None), "  (none)\n");
    }
}
