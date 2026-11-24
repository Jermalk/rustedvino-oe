// ============================================================
// src/reasoning_tokens.rs — reasoning-token attribution (0.9.1 V1-usage)
// ============================================================
// `usage.completion_tokens_details.reasoning_tokens` and
// `rustedvino_reasoning_tokens_total` are counted on the engine thread, next
// to the tokens it generates, by one of two rules (P1, decided 2026-11-16):
//
// 1. EXACT per delta (text CB engine). Every CB callback carries the engine's
//    own token count for its text delta, so a delta whose bytes all lie inside
//    the reasoning span (markers included) adds its count to reasoning, one
//    whose bytes all lie outside adds nothing. A multi-token delta (speculative
//    decoding, or a bridge hold-back) that straddles the span edge cannot be
//    split; it goes WHOLLY to reasoning and the request logs one debug line
//    ("reasoning split approximate"). The rule is "a delta counts as reasoning
//    when any of its bytes lies in the span", so a single-token delta is exact
//    by definition (a token that carries a marker plus a newline, e.g. `>\n`
//    on a tokenizer without dedicated `<think>` tokens, is a reasoning token);
//    only a multi-token straddle is approximate. Multi-token deltas are not only
//    speculative decoding: the bridge folds tokens that decode to no text (a
//    skipped special token such as gpt-oss's `<|message|>`, or a held-back
//    partial character) into the next delta that has text (`ov_bridge.cpp`
//    `pending_new_tokens`), so such a token counts where that delta lands
//    (review R-5).
//
// 2. APPROXIMATE recount (VLM, NPU). Their streamer callbacks are text chunks
//    with no per-chunk token count (a chunk can hold several tokens), so after
//    the generation the engine re-encodes the reasoning span's text with the
//    model's own tokenizer and clamps it to the exact generated total.
//    Re-encoding text is not the same as the generated ids, so the figure can
//    differ by a few tokens: it is documented as approximate, never as exact.
//    If the tokenizer fails, the exact total is split by bytes instead (WARN).
//
// Either way the figure is never larger than the completion-token count the
// usage block reports.
// ============================================================

use std::ops::Range;

use crate::prompt_builder::ReasoningSplit;

/// Bytes of `span` (a delta's byte range in the whole output) that lie inside any of
/// `spans[*cursor..]` (sorted, disjoint reasoning ranges). Advances `cursor` past every span that
/// ends at or before `span.start`, so a caller walking deltas in output order makes one pass over
/// both lists: O(deltas + spans) per request, not O(deltas × spans) (review R-1 — a client could
/// otherwise make a long output with many `<think>` blocks stall the CB engine thread at the
/// request's end).
fn overlap(span: &Range<usize>, spans: &[Range<usize>], cursor: &mut usize) -> usize {
    while *cursor < spans.len() && spans[*cursor].end <= span.start {
        *cursor += 1;
    }
    spans[*cursor..]
        .iter()
        .take_while(|r| r.start < span.end)
        .map(|r| {
            let lo = r.start.max(span.start);
            let hi = r.end.min(span.end);
            hi.saturating_sub(lo)
        })
        .sum()
}

/// The exact per-delta reasoning tally of one CB request (rule 1).
///
/// Holds the request's output text and, per non-empty delta, where it ends and
/// how many tokens the engine reported for it. Fed once per delta on the
/// engine thread (a `String` append and a `Vec` push); resolved once, when the
/// request finishes.
#[derive(Debug, Default)]
pub struct DeltaTally {
    text: String,
    /// `(end byte offset in text, tokens)` per non-empty delta, in order.
    deltas: Vec<(usize, usize)>,
}

/// The resolved reasoning count of one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasoningCount {
    /// Reasoning tokens, never above the completion-token total it was resolved against.
    pub tokens: usize,
    /// At least one multi-token delta straddled the reasoning span edge (rule 1) and went wholly
    /// to reasoning, or the figure came from a recount (rule 2): the figure is not exact.
    pub approximate: bool,
}

impl DeltaTally {
    /// Record one delta of `tokens` engine tokens. Empty deltas carry no text the usage block
    /// counts (the handler skips them for `completion_tokens` too), so they are ignored here.
    pub fn push(&mut self, delta: &str, tokens: usize) {
        if delta.is_empty() {
            return;
        }
        self.text.push_str(delta);
        self.deltas.push((self.text.len(), tokens));
    }

    /// Sum of the recorded deltas' tokens: the completion-token figure the handler reports for
    /// these deltas.
    #[must_use]
    pub fn completion_tokens(&self) -> usize {
        self.deltas.iter().map(|&(_, n)| n).sum()
    }

    /// Resolve the reasoning count under `split`.
    #[must_use]
    pub fn resolve(&self, split: ReasoningSplit) -> ReasoningCount {
        let spans = split.reasoning_spans(&self.text);
        if spans.is_empty() {
            return ReasoningCount {
                tokens: 0,
                approximate: false,
            };
        }
        let mut tokens = 0usize;
        let mut straddled = false;
        let mut start = 0usize;
        let mut cursor = 0usize;
        for &(end, n) in &self.deltas {
            let span = start..end;
            let inside = overlap(&span, &spans, &mut cursor);
            if inside == span.len() {
                tokens += n;
            } else if inside > 0 {
                // Partly inside: wholly to reasoning (P1). Exact for a single token (it is a
                // reasoning token by the rule); approximate only when the delta holds several.
                tokens += n;
                straddled = straddled || n > 1;
            }
            start = end;
        }
        ReasoningCount {
            tokens: tokens.min(self.completion_tokens()),
            approximate: straddled,
        }
    }
}

/// Rule 2 (VLM, NPU): recount the reasoning span of `text` (the request's whole output) with
/// `count` (the model's own tokenizer, called on the engine thread) and clamp the result to
/// `completion_total` (the exact generated-token figure the usage block reports).
///
/// Never fails: when `count` errors, the exact total is split by bytes (reasoning bytes ÷ output
/// bytes) and a WARN names the model. No tokenizer call is made when there is no reasoning span.
pub fn recount_reasoning<F>(
    model_id: &str,
    text: &str,
    split: ReasoningSplit,
    completion_total: usize,
    count: F,
) -> ReasoningCount
where
    F: FnOnce(&str) -> anyhow::Result<usize>,
{
    let spans = split.reasoning_spans(text);
    if spans.is_empty() || completion_total == 0 {
        return ReasoningCount {
            tokens: 0,
            approximate: false,
        };
    }
    let reasoning: String = spans.iter().map(|r| &text[r.clone()]).collect();
    let tokens = match count(&reasoning) {
        Ok(n) => n,
        Err(e) => {
            let split_by_bytes = byte_proportional(reasoning.len(), text.len(), completion_total);
            tracing::warn!(
                model_id,
                error = %e,
                reasoning_tokens = split_by_bytes,
                "reasoning-token recount failed — splitting the generated total by bytes instead"
            );
            split_by_bytes
        }
    };
    ReasoningCount {
        tokens: tokens.min(completion_total),
        approximate: true,
    }
}

/// `total × part ÷ whole`, rounded to the nearest token, never above `total`.
fn byte_proportional(part: usize, whole: usize, total: usize) -> usize {
    if whole == 0 {
        return 0;
    }
    let num = (total as u128) * (part as u128) + (whole as u128) / 2;
    let n = num / (whole as u128);
    usize::try_from(n).unwrap_or(total).min(total)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::model_manager::ReasoningParser;

    fn qwen(starts: bool) -> ReasoningSplit {
        ReasoningSplit::new(Some(ReasoningParser::Qwen3), starts)
    }

    fn tally(deltas: &[(&str, usize)]) -> DeltaTally {
        let mut t = DeltaTally::default();
        for (d, n) in deltas {
            t.push(d, *n);
        }
        t
    }

    #[test]
    fn single_token_deltas_count_exactly_markers_included() {
        let t = tally(&[
            ("<think>", 1),
            ("Let", 1),
            (" me", 1),
            ("</think>", 1),
            ("\n\n", 1),
            ("Hi", 1),
        ]);
        let c = t.resolve(qwen(false));
        assert_eq!(c.tokens, 4);
        assert!(!c.approximate);
        assert!(c.tokens <= t.completion_tokens());
    }

    #[test]
    fn prefilled_think_counts_from_the_first_token() {
        let t = tally(&[("a", 1), ("b", 1), ("</think>", 1), ("ok", 1)]);
        assert_eq!(t.resolve(qwen(true)).tokens, 3);
        // The same output without the prefill is all answer.
        assert_eq!(t.resolve(qwen(false)).tokens, 0);
    }

    #[test]
    fn never_closed_block_is_all_reasoning() {
        let t = tally(&[("x", 1), ("y", 1)]);
        let c = t.resolve(qwen(true));
        assert_eq!(c.tokens, 2);
        assert_eq!(c.tokens, t.completion_tokens());
    }

    #[test]
    fn no_parser_counts_a_raw_think_block() {
        let t = tally(&[("<think>", 1), ("r", 1), ("</think>", 1), ("a", 1)]);
        assert_eq!(t.resolve(ReasoningSplit::default()).tokens, 3);
    }

    #[test]
    fn thinking_off_or_no_markup_is_zero() {
        let t = tally(&[("Hello", 1), (" world", 1)]);
        let c = t.resolve(qwen(false));
        assert_eq!(c.tokens, 0);
        assert!(!c.approximate);
    }

    #[test]
    fn speculative_delta_inside_one_region_stays_exact() {
        let t = tally(&[("<think>", 1), ("a b c", 3), ("</think>", 1), ("x y", 2)]);
        let c = t.resolve(qwen(false));
        assert_eq!(c.tokens, 5);
        assert!(!c.approximate);
    }

    #[test]
    fn straddling_multi_token_delta_goes_wholly_to_reasoning_and_is_flagged() {
        let t = tally(&[("<think>", 1), ("r", 1), ("</think>ans", 3), ("!", 1)]);
        let c = t.resolve(qwen(false));
        assert_eq!(c.tokens, 5);
        assert!(c.approximate);
        assert!(c.tokens <= t.completion_tokens());
    }

    #[test]
    fn empty_deltas_are_ignored() {
        let t = tally(&[("<think>", 1), ("", 4), ("</think>", 1)]);
        assert_eq!(t.completion_tokens(), 2);
        assert_eq!(t.resolve(qwen(false)).tokens, 2);
    }

    #[test]
    fn harmony_split_counts_the_analysis_channel() {
        let split = ReasoningSplit::new(Some(ReasoningParser::GptOss), false);
        let t = tally(&[
            ("analysis", 1),
            ("The", 1),
            (" user", 1),
            ("assistant", 1),
            ("final", 1),
            ("4", 1),
        ]);
        assert_eq!(t.resolve(split).tokens, 5);
        let t = tally(&[("final", 1), ("4", 1)]);
        assert_eq!(t.resolve(split).tokens, 0);
    }

    #[test]
    fn recount_uses_the_tokenizer_on_the_reasoning_text_only() {
        let mut seen = String::new();
        let c = recount_reasoning(
            "m",
            "<think>abc</think>answer",
            qwen(false),
            10,
            |s: &str| {
                seen = s.to_owned();
                Ok(3)
            },
        );
        assert_eq!(seen, "<think>abc</think>");
        assert_eq!(c.tokens, 3);
        assert!(c.approximate, "a recount is never exact");
    }

    #[test]
    fn recount_is_clamped_to_the_generated_total() {
        let c = recount_reasoning("m", "abc", qwen(true), 2, |_: &str| Ok(9));
        assert_eq!(c.tokens, 2);
    }

    #[test]
    fn recount_tokenizer_error_falls_back_to_bytes() {
        // 18 reasoning bytes of 24: 10 × 18/24 = 7.5 → 8.
        let c = recount_reasoning(
            "m",
            "<think>abc</think>answer",
            qwen(false),
            10,
            |_: &str| Err(anyhow::anyhow!("tokenizer gone")),
        );
        assert_eq!(c.tokens, 8);
        assert!(c.tokens <= 10);
    }

    #[test]
    fn recount_without_reasoning_never_calls_the_tokenizer() {
        let c = recount_reasoning("m", "plain answer", qwen(false), 5, |_: &str| {
            panic!("no reasoning span: the tokenizer must not be called")
        });
        assert_eq!(c.tokens, 0);
        assert!(!c.approximate);
    }

    #[test]
    fn byte_proportional_bounds() {
        assert_eq!(byte_proportional(0, 10, 7), 0);
        assert_eq!(byte_proportional(10, 10, 7), 7);
        assert_eq!(byte_proportional(5, 0, 7), 0);
        assert_eq!(byte_proportional(usize::MAX, usize::MAX, 7), 7);
    }

    /// On a 32k-token-sized output the fake tokenizer is handed exactly the reasoning span. The
    /// recount's own cost (span scan + string build) is only printed for the PND log
    /// (`--nocapture`, P1 condition 4), not asserted; the tokenizer call itself needs a model.
    /// The fake tokenizer sees exactly the reasoning span (markers included), and its count comes
    /// back unclamped because it is below the total.
    #[test]
    fn recount_of_a_long_output_encodes_exactly_the_reasoning_span() {
        let body = "word ".repeat(32_000);
        let text = format!("<think>{body}</think>answer");
        let mut seen_len = 0;
        let t0 = std::time::Instant::now();
        let c = recount_reasoning("m", &text, qwen(false), 40_000, |s: &str| {
            seen_len = s.len();
            Ok(s.len() / 5)
        });
        let elapsed = t0.elapsed();
        println!(
            "recount overhead without tokenizer: {elapsed:?} for {} bytes",
            text.len()
        );
        assert_eq!(seen_len, text.len() - "answer".len());
        assert_eq!(c.tokens, seen_len / 5);
        assert!(c.approximate);
    }

    /// R-1: an output with many `<think>` blocks and one delta per token resolves in one pass (a
    /// quadratic walk over 20,000 deltas × 5,000 spans is 10^8 range checks); the count stays
    /// exact. The time bound is generous on purpose (a debug build on a loaded box).
    #[test]
    fn many_spans_resolve_in_one_pass_and_stay_exact() {
        let mut t = DeltaTally::default();
        for _ in 0..5_000 {
            t.push("<think>", 1);
            t.push("r", 1);
            t.push("</think>", 1);
            t.push("a", 1);
        }
        let t0 = std::time::Instant::now();
        let c = t.resolve(ReasoningSplit::default());
        let elapsed = t0.elapsed();
        assert_eq!(c.tokens, 15_000);
        assert!(!c.approximate);
        assert_eq!(t.completion_tokens(), 20_000);
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "resolve took {elapsed:?} for 20,000 deltas × 5,000 spans"
        );
    }

    /// The cursor walk agrees with a brute-force overlap, deltas straddling several spans included.
    #[test]
    fn cursor_overlap_matches_brute_force() {
        let spans = vec![2..4, 6..9, 12..13, 15..20];
        let brute = |d: &Range<usize>| -> usize {
            spans
                .iter()
                .map(|r| r.end.min(d.end).saturating_sub(r.start.max(d.start)))
                .sum()
        };
        let mut cursor = 0;
        let mut start = 0;
        for end in [1, 3, 7, 7, 14, 16, 22] {
            let d = start..end;
            assert_eq!(overlap(&d, &spans, &mut cursor), brute(&d), "delta {d:?}");
            start = end;
        }
    }
}
