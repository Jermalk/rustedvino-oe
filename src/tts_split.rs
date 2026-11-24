//! Sentence splitting for TTS engines that cannot read long input in one pass
//! (0.9.1 T1).
//!
//! `SpeechT5` tokenizes per character and has 600 text positions: past about 600
//! characters it drops the rest of the input **without an error**, and quality
//! breaks down from about 250–450 characters. Pieces of at most
//! [`SPEECHT5_SEGMENT_CAP`] characters, cut at sentence ends, were verified safe
//! (8/8 pieces, Whisper word match 0.97–1.00). The handler synthesizes each piece
//! in order and joins the audio.
//!
//! The rules, in order:
//! 1. Text of at most `cap` characters is returned unchanged, as one piece (not
//!    even trimmed): the text handed to the engine is the same as before the
//!    split existed. (The audio is not guaranteed identical: `SpeechT5` output length
//!    varies between server processes for the same text.)
//! 2. Otherwise the text is cut into sentences. A sentence ends after a run of
//!    `.` `!` `?` `…` (plus any closing quotes or brackets) that is followed by
//!    whitespace or the end of the text. After a plain `.` the next word must
//!    not start with a lowercase letter, so "e.g. the" stays together; "Mr. Smith"
//!    does split, which costs a short pause and nothing else (pieces are packed,
//!    rule 4). There is no abbreviation list.
//! 3. A sentence longer than `cap` is cut again: at the last clause break that
//!    leaves a piece of at least [`MIN_CLAUSE_PIECE`] characters; else at the last
//!    line break (`\n`) with as much on its left; else at the last whitespace; else
//!    at any clause break; else hard at `cap` characters. A clause break is `,`
//!    `;` `:` followed by whitespace, or a dash `—` / `–` (em or en) unless it sits
//!    unspaced between two letters or digits (`10–20`, `well—known` are not
//!    breaks). Punctuation stays with the piece on its left.
//! 4. Consecutive sentences are packed into one piece while the piece (with the
//!    whitespace between them) stays within `cap`, so short sentences do not each
//!    become a separately synthesized fragment.
//!
//! Every piece is a slice of the input with surrounding whitespace trimmed;
//! between consecutive pieces there is only whitespace. Lengths are counted in
//! characters (`char`s), never bytes, and cuts land on character boundaries.
//! A hard cut lands on a `char` boundary, not a grapheme-cluster boundary: a
//! combining mark or an emoji sequence (ZWJ, skin tone) can be split from its
//! base at a hard cut. Hard cuts only happen in a run of 200 characters with no
//! whitespace, which speech text does not contain in practice.

/// Largest piece, in characters, handed to `SpeechT5` in one synthesis call.
/// A hard constant (not a config key): the model's limit, not a tuning choice.
pub const SPEECHT5_SEGMENT_CAP: usize = 200;

/// Most pieces one `SpeechT5` `/v1/audio/speech` request may split into (0.9.1
/// T5): about 10,000 characters, ~8 minutes of speech. More is
/// refused with a `400 input_too_long` before any synthesis, so one request
/// (the 4096-digit worst case is ~120 pieces) cannot hold the engine for ~20
/// minutes. A hard constant beside [`SPEECHT5_SEGMENT_CAP`], not a config key.
pub const SPEECHT5_MAX_PIECES: usize = 50;

/// Response header on every successful `SpeechT5` `/v1/audio/speech` response:
/// how many pieces the input was synthesized in (`1` when it fit the cap).
/// Absent for engines that read the whole input in one call, and on errors.
/// Listed in the CORS `Access-Control-Expose-Headers`.
pub const TTS_SEGMENTS_HEADER: &str = "x-rustedvino-tts-segments";

/// Response header next to [`TTS_SEGMENTS_HEADER`]: the per-piece cap in
/// characters ([`SPEECHT5_SEGMENT_CAP`]). Same responses as that header.
pub const TTS_SEGMENT_CAP_HEADER: &str = "x-rustedvino-tts-segment-cap";

/// Shortest piece, in characters, a clause break may leave on its left before
/// the whitespace fallback is preferred (avoids a 5-character fragment when a
/// comma happens to sit near the start of a long sentence).
pub const MIN_CLAUSE_PIECE: usize = 50;

/// Split `text` into pieces of at most `cap` characters (see the module doc).
/// A `cap` of 0 is treated as 1. Whitespace-only text over `cap` comes back as
/// the single unchanged piece (the engine sees what it would have seen before).
#[must_use]
pub fn split_for_speech(text: &str, cap: usize) -> Vec<&str> {
    let cap = cap.max(1);
    if text.chars().count() <= cap {
        return vec![text];
    }
    let mut units: Vec<(usize, usize)> = Vec::new();
    for (start, end) in sentence_ranges(text) {
        split_long(text, start, end, cap, &mut units);
    }
    if units.is_empty() {
        return vec![text];
    }
    pack(text, &units, cap)
        .into_iter()
        .map(|(s, e)| &text[s..e])
        .collect()
}

/// Closing characters that belong to the sentence before a break.
fn is_closer(c: char) -> bool {
    matches!(c, '"' | '\'' | ')' | ']' | '}' | '»' | '”' | '’')
}

fn is_terminator(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '…')
}

/// Byte ranges of the sentences in `text`, trimmed, non-empty, in order.
fn sentence_ranges(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut sent_start = 0usize;
    let mut i = 0usize;
    while i < chars.len() {
        let (_, c) = chars[i];
        if !is_terminator(c) {
            i += 1;
            continue;
        }
        // The run of terminators, then closers.
        let mut j = i;
        let mut only_periods = true;
        while j < chars.len() && is_terminator(chars[j].1) {
            only_periods &= chars[j].1 == '.';
            j += 1;
        }
        while j < chars.len() && is_closer(chars[j].1) {
            j += 1;
        }
        let end_byte = chars.get(j).map_or(text.len(), |&(b, _)| b);
        let at_end = j == chars.len();
        let followed_by_space = at_end || chars[j].1.is_whitespace();
        let mut is_break = followed_by_space;
        if is_break && only_periods && !at_end {
            // Next non-whitespace character: a lowercase letter means "e.g. the".
            let next = chars[j..]
                .iter()
                .map(|&(_, c)| c)
                .find(|c| !c.is_whitespace());
            if next.is_some_and(char::is_lowercase) {
                is_break = false;
            }
        }
        if is_break {
            push_trimmed(text, sent_start, end_byte, &mut out);
            sent_start = end_byte;
        }
        i = j.max(i + 1);
    }
    push_trimmed(text, sent_start, text.len(), &mut out);
    out
}

/// Push `text[start..end]` with whitespace trimmed, if anything is left.
fn push_trimmed(text: &str, start: usize, end: usize, out: &mut Vec<(usize, usize)>) {
    let slice = &text[start..end];
    let lead = slice.len() - slice.trim_start().len();
    let trail = slice.len() - slice.trim_end().len();
    if lead + trail < slice.len() {
        out.push((start + lead, end - trail));
    }
}

/// Cut the trimmed range `text[start..end]` into ranges of at most `cap` chars.
fn split_long(text: &str, start: usize, end: usize, cap: usize, out: &mut Vec<(usize, usize)>) {
    let mut s = start;
    loop {
        let rest = &text[s..end];
        if rest.chars().count() <= cap {
            push_trimmed(text, s, end, out);
            return;
        }
        // Byte offset (within `rest`) just past the first `cap` characters.
        let limit = rest.char_indices().nth(cap).map_or(rest.len(), |(b, _)| b);
        let cut = clause_cut(rest, limit, MIN_CLAUSE_PIECE)
            .or_else(|| newline_cut(rest, limit, MIN_CLAUSE_PIECE))
            .or_else(|| whitespace_cut(rest, limit))
            .or_else(|| clause_cut(rest, limit, 1))
            .unwrap_or(limit);
        push_trimmed(text, s, s + cut, out);
        s += cut;
        // Skip the whitespace the cut left at the front.
        let skipped = text[s..end].len() - text[s..end].trim_start().len();
        s += skipped;
        if s >= end {
            return;
        }
    }
}

/// The last clause break whose left piece (cut just after the punctuation) is
/// within `limit` bytes and holds at least `min_chars` characters.
fn clause_cut(rest: &str, limit: usize, min_chars: usize) -> Option<usize> {
    let mut best = None;
    let mut iter = rest.char_indices().peekable();
    let mut count = 0usize;
    let mut prev: Option<char> = None;
    while let Some((b, c)) = iter.next() {
        count += 1;
        let after = b + c.len_utf8();
        if after > limit {
            break;
        }
        let next = iter.peek().map(|&(_, n)| n);
        let is_break = match c {
            ',' | ';' | ':' => next.is_none_or(char::is_whitespace),
            // `10–20`, `well—known`: an unspaced dash between two letters or
            // digits joins them; it is not a pause.
            '—' | '–' => {
                !(prev.is_some_and(char::is_alphanumeric)
                    && next.is_some_and(char::is_alphanumeric))
            }
            _ => false,
        };
        if is_break && count >= min_chars {
            best = Some(after);
        }
        prev = Some(c);
    }
    best
}

/// The last line break at or before `limit` with at least `min_chars`
/// characters on its left: the cut lands on it (the left piece ends before it).
fn newline_cut(rest: &str, limit: usize, min_chars: usize) -> Option<usize> {
    rest.char_indices()
        .take_while(|&(b, _)| b <= limit)
        .enumerate()
        .filter(|&(i, (_, c))| c == '\n' && i >= min_chars)
        .map(|(_, (b, _))| b)
        .last()
}

/// The last whitespace at or before `limit` (but not at the start): the cut
/// lands on it, so the left piece ends before it.
fn whitespace_cut(rest: &str, limit: usize) -> Option<usize> {
    rest.char_indices()
        .take_while(|&(b, _)| b <= limit)
        .filter(|&(b, c)| b > 0 && c.is_whitespace())
        .map(|(b, _)| b)
        .last()
}

/// Merge consecutive ranges while the merged span stays within `cap` chars.
fn pack(text: &str, units: &[(usize, usize)], cap: usize) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for &(s, e) in units {
        if let Some(last) = out.last_mut()
            && text[last.0..e].chars().count() <= cap
        {
            last.1 = e;
            continue;
        }
        out.push((s, e));
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// Every piece is non-empty, trimmed, within `cap` chars, and the pieces
    /// appear in the input in order with only whitespace between them (and
    /// before the first / after the last): nothing lost, nothing duplicated.
    fn assert_lossless(text: &str, pieces: &[&str], cap: usize) {
        let mut pos = 0usize;
        for p in pieces {
            assert!(!p.is_empty(), "empty piece in {pieces:?}");
            assert_eq!(p.trim(), *p, "untrimmed piece {p:?}");
            assert!(
                p.chars().count() <= cap,
                "piece over cap: {} chars",
                p.chars().count()
            );
            let at = text[pos..].find(p).expect("piece not found in order") + pos;
            assert!(
                text[pos..at].chars().all(char::is_whitespace),
                "non-whitespace text skipped: {:?}",
                &text[pos..at]
            );
            pos = at + p.len();
        }
        assert!(
            text[pos..].chars().all(char::is_whitespace),
            "tail lost: {:?}",
            &text[pos..]
        );
        let squash = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
        assert_eq!(squash(&pieces.concat()), squash(text));
    }

    #[test]
    fn short_input_is_one_unchanged_piece() {
        let t = "  Hello world.  Two sentences here!  ";
        assert_eq!(split_for_speech(t, 200), vec![t]);
        let exactly = "a".repeat(200);
        assert_eq!(split_for_speech(&exactly, 200), vec![exactly.as_str()]);
    }

    #[test]
    fn sentence_ends_cover_period_bang_question_ellipsis() {
        // cap small enough that no two sentences pack together.
        let t =
            "First one is here. Second one now! Third one, why? Fourth one waits… Fifth one ends.";
        let p = split_for_speech(t, 20);
        assert_eq!(
            p,
            vec![
                "First one is here.",
                "Second one now!",
                "Third one, why?",
                "Fourth one waits…",
                "Fifth one ends."
            ]
        );
        assert_lossless(t, &p, 20);
    }

    #[test]
    fn closing_quotes_stay_with_their_sentence() {
        let t = "He said \"stop here.\" Then he left the room?! (It was late.) Done now.";
        let p = split_for_speech(t, 23);
        assert_eq!(
            p,
            vec![
                "He said \"stop here.\"",
                "Then he left the room?!",
                "(It was late.)",
                "Done now."
            ]
        );
        assert_lossless(t, &p, 23);
    }

    #[test]
    fn period_before_lowercase_is_not_a_sentence_end() {
        // "e.g. the" stays together; "Mr. Smith" splits (no abbreviation list).
        let t = "Use a tool, e.g. the hammer. Mr. Smith came.";
        let p = split_for_speech(t, 30);
        assert_eq!(p, vec!["Use a tool, e.g. the hammer.", "Mr. Smith came."]);
        // As sentences, "Mr." is its own (packing then rejoins it with "Smith came.").
        let sentences: Vec<&str> = sentence_ranges(t).iter().map(|&(s, e)| &t[s..e]).collect();
        assert_eq!(
            sentences,
            vec!["Use a tool, e.g. the hammer.", "Mr.", "Smith came."]
        );
    }

    #[test]
    fn decimal_point_is_not_a_sentence_end() {
        let t = "Pi is 3.14 roughly. And e is 2.71 roughly.";
        let p = split_for_speech(t, 25);
        assert_eq!(p, vec!["Pi is 3.14 roughly.", "And e is 2.71 roughly."]);
    }

    #[test]
    fn sentences_are_packed_up_to_the_cap() {
        let s = "This sentence has forty characters in it."; // 41 chars
        let t = [s; 10].join(" ");
        let p = split_for_speech(&t, 200);
        // 4 sentences = 41*4 + 3 = 167 chars; 5 = 209 > 200.
        assert_eq!(p.len(), 3);
        assert_eq!(p[0].chars().count(), 167);
        assert_lossless(&t, &p, 200);
    }

    #[test]
    fn long_sentence_falls_back_to_a_clause_break() {
        let a = "word ".repeat(15); // 75 chars
        let t = format!("{a}first clause, {a}second clause; {a}end.");
        let p = split_for_speech(&t, 120);
        assert!(p[0].ends_with("first clause,"), "{p:?}");
        assert!(p[1].ends_with("second clause;"), "{p:?}");
        assert_lossless(&t, &p, 120);
    }

    #[test]
    fn a_clause_break_too_near_the_start_loses_to_whitespace() {
        let t = format!("Well, {}", "word ".repeat(60).trim_end());
        let p = split_for_speech(&t, 100);
        assert_ne!(p[0], "Well,");
        assert!(p[0].chars().count() > 90, "{p:?}");
        assert_lossless(&t, &p, 100);
    }

    #[test]
    fn a_dash_is_a_clause_break_unless_unspaced_between_letters_or_digits() {
        let left = "x".repeat(60);
        // Spaced, and unspaced after punctuation: breaks.
        for dash in [" — ", " – ", ".—"] {
            let t = format!("{left}{dash}{}", "y".repeat(100));
            let p = split_for_speech(&t, 100);
            assert_eq!(p[0], format!("{left}{}", dash.trim_end()), "{dash:?}");
            assert_lossless(&t, &p, 100);
        }
        // `well—known`, `10–20`: not breaks, so the whitespace fallback decides.
        for joined in ["well—known", "10–20"] {
            let t = format!("{} {joined} {}", "w".repeat(60), "z".repeat(100));
            let p = split_for_speech(&t, 100);
            assert!(!p[0].ends_with('—') && !p[0].ends_with('–'), "{p:?}");
            assert!(p.iter().any(|s| s.contains(joined)), "kept whole: {p:?}");
            assert_lossless(&t, &p, 100);
        }
    }

    #[test]
    fn a_line_break_wins_over_other_whitespace() {
        let line1 = "word ".repeat(12).trim_end().to_owned(); // 59 chars
        let t = format!("{line1}\n{}", "more words here ".repeat(10).trim_end());
        let p = split_for_speech(&t, 120);
        assert_eq!(p[0], line1);
        assert_lossless(&t, &p, 120);
        // Under MIN_CLAUSE_PIECE on its left: plain whitespace decides.
        let t = format!("short\n{}", "word ".repeat(40).trim_end());
        let p = split_for_speech(&t, 120);
        assert_ne!(p[0], "short");
        assert_lossless(&t, &p, 120);
    }

    #[test]
    fn no_clause_break_falls_back_to_whitespace() {
        let t = "abcdefghi ".repeat(50); // 500 chars, words of 9
        let p = split_for_speech(&t, 200);
        assert!(
            p.iter().all(|s| s.split(' ').all(|w| w == "abcdefghi")),
            "{p:?}"
        );
        assert_lossless(&t, &p, 200);
    }

    #[test]
    fn no_whitespace_is_cut_hard_at_the_cap() {
        let t = "z".repeat(450);
        let p = split_for_speech(&t, 200);
        assert_eq!(
            p.iter().map(|s| s.len()).collect::<Vec<_>>(),
            vec![200, 200, 50]
        );
        assert_lossless(&t, &p, 200);
    }

    #[test]
    fn multibyte_text_is_counted_in_chars_and_never_cut_inside_one() {
        // 2-byte, 3-byte and 4-byte characters with no spaces: hard cuts.
        for unit in ["ż", "—x", "😀"] {
            let t = unit.repeat(450 / unit.chars().count());
            let p = split_for_speech(&t, 200);
            for s in &p {
                assert!(s.chars().count() <= 200);
            }
            assert!(p.len() >= 3, "450 chars need 3 pieces");
            assert_lossless(&t, &p, 200);
        }
        // A Polish sentence mix with spaces: 200 chars, not 200 bytes.
        let t = "Zażółć gęślą jaźń. ".repeat(30);
        let p = split_for_speech(&t, 200);
        assert!(
            p.iter().any(|s| s.len() > 200),
            "byte length exceeds 200 while chars do not"
        );
        assert_lossless(&t, &p, 200);
    }

    #[test]
    fn whitespace_only_long_text_comes_back_unchanged() {
        let t = " ".repeat(300);
        assert_eq!(split_for_speech(&t, 200), vec![t.as_str()]);
    }

    #[test]
    fn newlines_and_runs_of_spaces_are_trimmed_at_cuts() {
        let t = format!(
            "{}\n\n   {}",
            "A sentence ends here.".repeat(6),
            "Another one!\n".repeat(20)
        );
        let p = split_for_speech(&t, 200);
        assert_lossless(&t, &p, 200);
    }

    #[test]
    fn a_long_mixed_text_is_lossless_at_several_caps() {
        let t = "The keeper lit the lamp at dusk; ships passed — safely, mostly. \
                 Was it worth it? Yes… it was! He wrote: \"calm sea, light wind.\" \
                 Then, without a word, he climbed the stairs again and again and again \
                 until the morning came over the grey water and the gulls began to cry.";
        let t = t.repeat(5);
        for cap in [1, 7, 30, 64, 120, 200, 500] {
            let p = split_for_speech(&t, cap);
            assert_lossless(&t, &p, cap.max(1));
        }
    }
}
