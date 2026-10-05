// ============================================================
// src/systemone/wire.rs — decode a /v1/systemone body the way Ollama's Go decoder does
// ============================================================
// Ollama binds the body with Go's `encoding/json` into `decision.Request`. Its rules differ
// from a `#[derive(Deserialize)]` struct, and the golden fixtures
// pin each one:
//   * struct keys match case-insensitively (`"STATE"` fills `state`), and when several keys
//     land on one field the LAST one wins — a derived struct would reject the duplicate;
//   * `null` leaves a field at its zero value (`model: null` → "" → "model is required");
//   * `questions` / `criteria` are ordered maps: a duplicate key keeps its FIRST position
//     and takes its LAST value;
//   * `state`, `instructions` and `criteria` stay raw JSON text (Go `json.RawMessage`) —
//     `compile` needs the verbatim bytes;
//   * unknown fields are ignored (`stream`, `options`, …).
//
// CRASH COURSE — `Box<RawValue>`:
//   serde_json's `RawValue` is an unparsed slice of JSON text that serde has already
//   validated. Keeping it boxed lets us decide later how to read it (a string, an object we
//   compact byte-for-byte, or an ordered map) without re-serialising and losing bytes.
// ============================================================

use serde::Deserialize;
use serde::de::{Deserializer, MapAccess, Visitor};
use serde_json::value::RawValue;

use super::SystemOneError;
use super::gojson::{go_lossy_utf8, go_quote, go_unquote};

/// A JSON object as its raw entries, in order, duplicates included. Struct decoding walks all
/// of them (Go assigns fields in sequence, so the last occurrence wins); ordered maps call
/// [`RawObject::into_ordered_map`].
#[derive(Debug, Default)]
pub struct RawObject(pub Vec<(String, Box<RawValue>)>);

impl RawObject {
    /// Ollama's `orderedmap` semantics (wk8/go-ordered-map `Set`): a duplicate key keeps its
    /// first position and takes the last value.
    #[must_use]
    pub fn into_ordered_map(self) -> Vec<(String, Box<RawValue>)> {
        let mut out: Vec<(String, Box<RawValue>)> = Vec::with_capacity(self.0.len());
        for (k, v) in self.0 {
            if let Some(slot) = out.iter_mut().find(|(ek, _)| *ek == k) {
                slot.1 = v;
            } else {
                out.push((k, v));
            }
        }
        out
    }
}

impl<'de> Deserialize<'de> for RawObject {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = RawObject;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<RawObject, A::Error> {
                let mut out = Vec::new();
                while let Some(entry) = map.next_entry::<String, Box<RawValue>>()? {
                    out.push(entry);
                }
                Ok(RawObject(out))
            }
        }
        de.deserialize_map(V)
    }
}

/// One question as Go decodes `decision.Question`: `type` is a string ("" when absent or
/// null); `instructions` and `criteria` are raw JSON (None when absent).
#[derive(Debug, Default)]
pub struct Question {
    /// `choice`, `noul` or `score` (validated in `compile`).
    pub typ: String,
    /// Raw `instructions` JSON text.
    pub instructions: Option<Box<RawValue>>,
    /// Raw `criteria` JSON text.
    pub criteria: Option<Box<RawValue>>,
}

/// A decoded /v1/systemone request (Ollama `decision.Request`).
#[derive(Debug, Default)]
pub struct Request {
    /// Model name ("" when absent or null).
    pub model: String,
    /// Raw `state` JSON text.
    pub state: Option<Box<RawValue>>,
    /// Questions in request order, duplicate names already merged.
    pub questions: Vec<(String, Question)>,
    /// `keep_alive`, kept raw (Ollama accepts a duration string or seconds); unused for now.
    pub keep_alive: Option<Box<RawValue>>,
    /// The body had invalid UTF-8 that was replaced with U+FFFD. Go keeps those bytes raw
    /// inside an object/array `state` or `instructions` and `json.Marshal` then writes them
    /// as the escape text `\ufffd` — `compile` needs to know to do the same.
    pub repaired_utf8: bool,
}

/// Decode a request body like Ollama (gin `ShouldBindJSON`): the first JSON value only, trailing
/// data ignored, Content-Type not checked. Invalid UTF-8 bytes become U+FFFD first, as Go does
/// inside strings.
///
/// # Errors
/// [`SystemOneError::Bind`] when the body is not JSON of the right shape (Ollama: HTTP 400
/// with Go's decoder message; ours is serde's, only the status and shape match).
pub fn decode_request(body: &[u8]) -> Result<Request, SystemOneError> {
    // Ollama reads the body through `http.MaxBytesReader(64 KiB)` with one `Decode`, so only
    // the first JSON value has to fit: whitespace or garbage after it may run past 64 KiB
    // (200), a value that crosses the limit is 413. Decode the first 64 KiB; an EOF there
    // on a longer body means the value crossed the limit.
    let over_limit = body.len() > MAX_BODY_BYTES;
    let window = &body[..body.len().min(MAX_BODY_BYTES)];
    let (text, repaired_utf8) = match std::str::from_utf8(window) {
        Ok(s) => (std::borrow::Cow::Borrowed(s), false),
        // A multi-byte character cut by the window is not a repair: it lies past the value.
        Err(e) if over_limit && e.error_len().is_none() => (
            std::borrow::Cow::Borrowed(
                std::str::from_utf8(&window[..e.valid_up_to()]).unwrap_or_default(),
            ),
            false,
        ),
        Err(_) => (std::borrow::Cow::Owned(go_lossy_utf8(window)), true),
    };
    // gin's ShouldBindJSON runs one `json.NewDecoder(body).Decode`: it reads the first JSON
    // value and ignores anything after it (`{…}x` is accepted). So: no `Deserializer::end()`.
    // A top-level `null` leaves the request at its zero value (Go), so it fails in compile.
    let mut de = serde_json::Deserializer::from_str(&text);
    let top = match Option::<RawObject>::deserialize(&mut de) {
        Ok(top) => top.unwrap_or_default(),
        Err(e) if over_limit && e.is_eof() => return Err(SystemOneError::TooLarge),
        Err(e) => return Err(SystemOneError::Bind(e.to_string())),
    };
    let mut req = Request {
        repaired_utf8,
        ..Request::default()
    };
    for (key, raw) in top.0 {
        match fold(&key, &["model", "state", "questions", "keep_alive"]) {
            Some("model") => req.model = string_or_null(&raw, "Request.model")?,
            Some("state") => req.state = Some(raw),
            Some("questions") => req.questions = questions(&raw)?,
            Some("keep_alive") => {
                check_keep_alive(&raw)?;
                req.keep_alive = Some(raw);
            }
            _ => {}
        }
    }
    Ok(req)
}

/// Ollama's body cap: 65 536 bytes of the first JSON value (`64<<10`).
pub const MAX_BODY_BYTES: usize = 64 << 10;

/// Ollama's `api.Duration` decode: JSON null (the field is a pointer), any finite number of
/// seconds, or a Go `time.ParseDuration` string; anything else is a bind error. The value
/// itself is unused here (no model unloading on a timer), only its validity matters.
fn check_keep_alive(raw: &RawValue) -> Result<(), SystemOneError> {
    let bad = |m: String| Err(SystemOneError::Bind(m));
    let text = raw.get();
    if text == "null" {
        return Ok(());
    }
    match serde_json::from_str::<serde_json::Value>(text) {
        Ok(serde_json::Value::Number(n)) if n.as_f64().is_some_and(f64::is_finite) => Ok(()),
        Ok(serde_json::Value::String(s)) if go_duration_valid(&s) => Ok(()),
        Ok(serde_json::Value::String(s)) => bad(format!("time: invalid duration {}", go_quote(&s))),
        Ok(other) => bad(format!(
            "Unsupported type: '{}'",
            match other {
                serde_json::Value::Bool(_) => "bool",
                serde_json::Value::Array(_) => "[]interface {}",
                serde_json::Value::Object(_) => "map[string]interface {}",
                _ => "number",
            }
        )),
        // serde_json refuses numbers that overflow f64 (e.g. 1e400), as Go does.
        Err(e) => bad(e.to_string()),
    }
}

/// Go `time.ParseDuration` acceptance: `[-+]` then one or more `<number><unit>` (number may
/// have a fraction, unit one of ns us µs μs ms s m h), or exactly "0"; no overflow past
/// 2^63 ns.
fn go_duration_valid(s: &str) -> bool {
    const UNITS: [(&str, u64); 8] = [
        ("ns", 1),
        ("us", 1_000),
        ("\u{b5}s", 1_000),
        ("\u{3bc}s", 1_000),
        ("ms", 1_000_000),
        ("s", 1_000_000_000),
        ("m", 60_000_000_000),
        ("h", 3_600_000_000_000),
    ];
    const LIMIT: u128 = 1 << 63;
    let s = s.strip_prefix(['-', '+']).unwrap_or(s);
    if s == "0" {
        return true;
    }
    if s.is_empty() {
        return false;
    }
    let (mut rest, mut total) = (s, 0u128);
    while !rest.is_empty() {
        let int_len = rest.bytes().take_while(u8::is_ascii_digit).count();
        let (int, after) = rest.split_at(int_len);
        let (frac, after) = match after.strip_prefix('.') {
            Some(a) => {
                let n = a.bytes().take_while(u8::is_ascii_digit).count();
                a.split_at(n)
            }
            None => ("", after),
        };
        if int.is_empty() && frac.is_empty() {
            return false; // no number before the unit
        }
        let unit_len = after
            .find(|c: char| c == '.' || c.is_ascii_digit())
            .unwrap_or(after.len());
        let (unit, after) = after.split_at(unit_len);
        let Some(&(_, scale)) = UNITS.iter().find(|(u, _)| *u == unit) else {
            return false; // missing or unknown unit
        };
        let Ok(int) = int
            .parse::<u128>()
            .or(if int.is_empty() { Ok(0) } else { Err(()) })
        else {
            return false;
        };
        total += int * u128::from(scale);
        if !frac.is_empty() {
            let digits = frac.len().min(18);
            let f: u128 = frac[..digits].parse().unwrap_or(0);
            total += f * u128::from(scale) / 10u128.pow(u32::try_from(digits).unwrap_or(18));
        }
        if total > LIMIT {
            return false;
        }
        rest = after;
    }
    true
}

fn questions(raw: &RawValue) -> Result<Vec<(String, Question)>, SystemOneError> {
    if raw.get() == "null" {
        return Ok(Vec::new());
    }
    let obj: RawObject =
        serde_json::from_str(raw.get()).map_err(|e| SystemOneError::Bind(e.to_string()))?;
    obj.into_ordered_map()
        .into_iter()
        .map(|(name, v)| Ok((name, question(&v)?)))
        .collect()
}

fn question(raw: &RawValue) -> Result<Question, SystemOneError> {
    let mut q = Question::default();
    if raw.get() == "null" {
        return Ok(q);
    }
    let obj: RawObject = serde_json::from_str(raw.get()).map_err(|_| {
        SystemOneError::Bind(
            "json: cannot unmarshal into Go value of type decision.Question".into(),
        )
    })?;
    for (key, v) in obj.0 {
        match fold(&key, &["type", "instructions", "criteria"]) {
            Some("type") => q.typ = string_or_null(&v, "Question.type")?,
            Some("instructions") => q.instructions = Some(v),
            Some("criteria") => q.criteria = Some(v),
            _ => {}
        }
    }
    Ok(q)
}

/// Go struct-field matching: exact name, else ASCII case-insensitive. (Go also applies
/// Unicode simple folding, e.g. the Kelvin sign; not reproduced — no real client sends it.)
fn fold(key: &str, fields: &[&'static str]) -> Option<&'static str> {
    fields
        .iter()
        .copied()
        .find(|f| key == *f)
        .or_else(|| fields.iter().copied().find(|f| key.eq_ignore_ascii_case(f)))
}

/// A Go `string` field: JSON null leaves it empty; any non-string is a bind error.
fn string_or_null(raw: &RawValue, field: &str) -> Result<String, SystemOneError> {
    if raw.get() == "null" {
        return Ok(String::new());
    }
    go_unquote(raw.get()).ok_or_else(|| {
        SystemOneError::Bind(format!(
            "json: cannot unmarshal into Go struct field {field} of type string"
        ))
    })
}
