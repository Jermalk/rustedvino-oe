// ============================================================
// src/systemone/gojson.rs — Go `encoding/json` byte behaviour, where Ollama depends on it
// ============================================================
// Ollama builds the decision prompt with Go's `json.Marshal` and `json.Compact`, so the exact
// bytes the model sees are Go's. `serde_json` makes different choices, and every difference is a
// different token sequence, which means different probabilities:
//   * Go escapes `<`, `>`, `&`, U+2028 and U+2029 (HTML-safe by default); serde_json does not.
//     This is also the prompt-injection defence: `<|im_end|>` in user text becomes
//     `\u003c|im_end|\u003e` and can never form a special token.
//   * `json.Compact` removes whitespace but keeps every other byte verbatim: `"\u00e9"` stays
//     escaped and `2.50` / `1e10` keep their spelling. Re-serialising a `serde_json::Value`
//     would rewrite both.
//
// CRASH COURSE — why `&str` in, `String` out, with no `Result`:
//   Rust strings are guaranteed valid UTF-8, so the one failure Go handles here (invalid bytes)
//   cannot reach these functions. Callers fix invalid bytes once, at the request boundary.
// ============================================================

use std::fmt::Write as _;

/// Marks a byte that was invalid UTF-8 in the request body, inside a compacted object/array
/// value: Go's `json.Marshal` writes such a byte as the escape text `\\ufffd`, while a real
/// U+FFFD character is written raw. A noncharacter, so it cannot collide with real text.
pub const INVALID_BYTE: char = '\u{10ffff}';

/// Append `s` as a JSON string literal exactly as Go's `json.Marshal` writes it
/// (`encodeState.string` with HTML escaping on).
pub fn push_go_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            INVALID_BYTE => out.push_str("\\ufffd"),
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '<' | '>' | '&' | '\u{2028}' | '\u{2029}' => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `s` as a Go JSON string literal.
#[must_use]
pub fn go_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    push_go_string(&mut out, s);
    out
}

/// Go's `json.Compact` on already-valid JSON text: drop insignificant whitespace, keep all
/// other bytes (escapes, number spellings, key order) verbatim.
#[must_use]
pub fn compact(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut in_string = false;
    let mut escaped = false;
    for c in raw.chars() {
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
        } else if c == '"' {
            in_string = true;
            out.push(c);
        } else if !matches!(c, ' ' | '\t' | '\n' | '\r') {
            out.push(c);
        }
    }
    out
}

/// Go's `%q` verb (`strconv.Quote`) for the field names and keys Ollama puts in error messages.
/// Printable characters (Go `strconv.IsPrint`: letters, marks, numbers, punctuation, symbols and
/// the ASCII space) are kept; everything else is escaped the way Go writes it — `\a`…`\v`,
/// `\xNN` for other ASCII, `\uNNNN` / `\UNNNNNNNN` beyond ASCII (NBSP, U+2028, BOM, private use…).
#[must_use]
pub fn go_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if is_go_print(c) => out.push(c),
            '\u{7}' => out.push_str("\\a"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{b}' => out.push_str("\\v"),
            c if c.is_ascii() => {
                let _ = write!(out, "\\x{:02x}", u32::from(c));
            }
            c if u32::from(c) <= 0xffff => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => {
                let _ = write!(out, "\\U{:08x}", u32::from(c));
            }
        }
    }
    out.push('"');
    out
}

/// Go `strconv.IsPrint`: Unicode categories L, M, N, P, S, plus the ASCII space. Uses the
/// `regex` crate's Unicode tables (std has none). If the class cannot be built — it is a
/// constant, so in practice never — fall back to "not an ASCII control".
fn is_go_print(c: char) -> bool {
    static PRINT: std::sync::OnceLock<Option<regex::Regex>> = std::sync::OnceLock::new();
    if c == ' ' {
        return true;
    }
    match PRINT.get_or_init(|| regex::Regex::new(r"^[\p{L}\p{M}\p{N}\p{P}\p{S}]$").ok()) {
        Some(re) => re.is_match(c.encode_utf8(&mut [0u8; 4])),
        None => !c.is_control(),
    }
}

/// Replace every byte that is not part of valid UTF-8 with U+FFFD, one replacement per bad
/// byte, as Go's JSON decoder does inside strings (`utf8.DecodeRune` returns width 1 on error).
/// `String::from_utf8_lossy` differs: it replaces a whole maximal invalid sequence with one U+FFFD.
#[must_use]
pub fn go_lossy_utf8(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    for chunk in bytes.utf8_chunks() {
        out.push_str(chunk.valid());
        for _ in chunk.invalid() {
            out.push('\u{fffd}');
        }
    }
    out
}

/// Decode a JSON string literal (quotes included) as Go does. `serde_json` first; if it refuses,
/// the one case Go accepts and serde does not: a lone UTF-16 surrogate escape (`"\ud800"`),
/// which Go decodes to U+FFFD. `None` when the text is not a string literal at all.
#[must_use]
pub fn go_unquote(raw: &str) -> Option<String> {
    if let Ok(s) = serde_json::from_str::<String>(raw) {
        return Some(s);
    }
    let inner = raw.strip_prefix('"')?.strip_suffix('"')?;
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next()? {
            '"' => out.push('"'),
            '\\' => out.push('\\'),
            '/' => out.push('/'),
            'b' => out.push('\u{8}'),
            'f' => out.push('\u{c}'),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            'u' => {
                let hi = hex4(&mut chars)?;
                if (0xD800..0xDC00).contains(&hi) {
                    // A high surrogate combines only with an immediately following low one.
                    let mut ahead = chars.clone();
                    if ahead.next() == Some('\\')
                        && ahead.next() == Some('u')
                        && let Some(lo) =
                            hex4(&mut ahead).filter(|lo| (0xDC00..0xE000).contains(lo))
                    {
                        out.push(char::from_u32(
                            0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00),
                        )?);
                        chars = ahead;
                        continue;
                    }
                    out.push('\u{fffd}');
                } else {
                    out.push(char::from_u32(hi).unwrap_or('\u{fffd}'));
                }
            }
            _ => return None,
        }
    }
    Some(out)
}

fn hex4(chars: &mut std::str::Chars<'_>) -> Option<u32> {
    let mut v = 0;
    for _ in 0..4 {
        v = v * 16 + chars.next()?.to_digit(16)?;
    }
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn go_string_matches_go_marshal() {
        // Expected strings are Go 1.26.0 `json.Marshal` outputs (the version Ollama v0.35.0 builds with).
        assert_eq!(go_string("a<b>&c"), r#""a\u003cb\u003e\u0026c""#);
        assert_eq!(go_string("\u{2028}\u{2029}"), r#""\u2028\u2029""#);
        assert_eq!(
            go_string("\u{8}\u{c}\n\r\t\u{0}\u{1f}"),
            r#""\b\f\n\r\t\u0000\u001f""#
        );
        assert_eq!(go_string("\u{7f}é🚀\"\\/"), "\"\u{7f}é🚀\\\"\\\\/\"");
    }

    #[test]
    fn compact_keeps_bytes_outside_whitespace() {
        let raw = "{ \"b\" : [1, 2.50, 1e10, -0.0], \"a\": \"\\u00e9 \\\" x\" }";
        assert_eq!(
            compact(raw),
            r#"{"b":[1,2.50,1e10,-0.0],"a":"\u00e9 \" x"}"#
        );
    }

    #[test]
    fn go_quote_escapes_like_strconv() {
        assert_eq!(go_quote("na\"me <&>"), r#""na\"me <&>""#);
        assert_eq!(go_quote(" "), r#"" ""#);
        assert_eq!(go_quote("żółw\n"), "\"żółw\\n\"");
    }

    #[test]
    fn unquote_lone_surrogates_like_go() {
        assert_eq!(go_unquote(r#""a\ud800b""#).as_deref(), Some("a\u{fffd}b"));
        assert_eq!(go_unquote(r#""\udc00""#).as_deref(), Some("\u{fffd}"));
        assert_eq!(
            go_unquote(r#""\ud800\u0041""#).as_deref(),
            Some("\u{fffd}A")
        );
        assert_eq!(go_unquote(r#""\ud83d\ude80 ok""#).as_deref(), Some("🚀 ok"));
        assert_eq!(go_unquote("5"), None);
    }

    #[test]
    fn lossy_replaces_per_byte() {
        assert_eq!(go_lossy_utf8(b"caf\xe9"), "caf\u{fffd}");
        // Truncated 3-byte sequence: Go emits one U+FFFD per byte, Rust's lossy emits one total.
        assert_eq!(go_lossy_utf8(b"a\xe2\x82b"), "a\u{fffd}\u{fffd}b");
    }
}
