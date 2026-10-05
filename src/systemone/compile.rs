// ============================================================
// src/systemone/compile.rs — Ollama `decision.Compile` + the qwen3.5 prompt, byte-for-byte
// ============================================================
// Port of Ollama v0.35.0 `decision/systemone.go` (MIT): validate the questions, give every
// allowed answer a one-letter code (A, B, …), and build one user message per question:
//
//   json.Marshal({"context": <state>, "schema": [<every field>]}) + "\n\nRequested field: " + <name>
//
// then frame it the way Ollama's `qwen3.5` renderer does with think=false
// (`model/renderers/qwen35.go`). Every string here is checked against Ollama's own output in
// `tests/systemone_golden.rs`; error messages are Ollama's verbatim (clients may match on them).
// ============================================================

use serde_json::value::RawValue;

use super::SystemOneError;
use super::gojson::{INVALID_BYTE, compact, go_quote, go_unquote, push_go_string};
use super::wire::{Question, RawObject, Request};

/// Most questions per request (Ollama `Compile`).
pub const MAX_QUESTIONS: usize = 64;
/// Fewest / most answer candidates per question (Ollama `compileField`).
pub const MIN_CANDIDATES: usize = 2;
/// See [`MIN_CANDIDATES`].
pub const MAX_CANDIDATES: usize = 26;

/// The three question types of the wire format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionType {
    /// Pick one key of an object of options.
    Choice,
    /// Yes/no; the answer is P(true).
    Noul,
    /// Ordinal 0..n-1; the answer is the expected index.
    Score,
}

/// A candidate's value as it appears in the schema JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChoiceValue {
    /// `noul`: false / true.
    Bool(bool),
    /// `choice` key, or `score` index as a string ("0", "1", …).
    Str(String),
}

/// One allowed answer: its code letter, value and description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Choice {
    /// "A", "B", … — the single token the model is scored on.
    pub code: String,
    /// Value in the schema (and the response key for choice/score).
    pub value: ChoiceValue,
    /// Human description shown to the model.
    pub description: String,
}

/// A compiled question.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    /// Question name (response key).
    pub name: String,
    /// Question type.
    pub typ: QuestionType,
    /// `instructions`, rendered as text.
    pub description: String,
    /// Candidates in code order.
    pub choices: Vec<Choice>,
}

/// A compiled request: the fields and one user message per field, in request order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compiled {
    /// Compiled questions.
    pub fields: Vec<Field>,
    /// User message per question (before chat framing).
    pub user_content: Vec<String>,
}

impl Compiled {
    /// Candidate codes per question ("A", "B", …), as Ollama's `ScoreRow.Candidates`.
    #[must_use]
    pub fn candidates(&self) -> Vec<Vec<String>> {
        self.fields
            .iter()
            .map(|f| f.choices.iter().map(|c| c.code.clone()).collect())
            .collect()
    }

    /// The full prompt per question: `system` (the model's system prompt; empty = none) and
    /// the user message, framed like Ollama's `qwen3.5` renderer with thinking off.
    #[must_use]
    pub fn prompts(&self, system: &str) -> Vec<String> {
        self.user_content
            .iter()
            .map(|u| render_qwen35(system, u))
            .collect()
    }
}

fn invalid(msg: impl Into<String>) -> SystemOneError {
    SystemOneError::Invalid(msg.into())
}

/// Validate and compile a decoded request (Ollama `decision.Compile`).
///
/// # Errors
/// [`SystemOneError::Invalid`] with Ollama's exact message (HTTP 400).
pub fn compile(req: &Request) -> Result<Compiled, SystemOneError> {
    if req.model.trim().is_empty() {
        return Err(invalid("model is required"));
    }
    if req.questions.is_empty() || req.questions.len() > MAX_QUESTIONS {
        return Err(invalid("questions must contain 1–64 fields"));
    }
    let context = content(req.state.as_deref(), req.repaired_utf8)
        .map_err(|e| invalid(format!("state: {e}")))?;
    if context.trim().is_empty() {
        return Err(invalid("state must not be empty"));
    }
    let fields = req
        .questions
        .iter()
        .map(|(name, q)| {
            compile_field(name, q, req.repaired_utf8)
                .map_err(|e| invalid(format!("question {}: {e}", go_quote(name))))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut data = String::from("{\"context\":");
    push_go_string(&mut data, &context);
    data.push_str(",\"schema\":[");
    for (i, f) in fields.iter().enumerate() {
        if i > 0 {
            data.push(',');
        }
        data.push_str("{\"name\":");
        push_go_string(&mut data, &f.name);
        data.push_str(",\"description\":");
        push_go_string(&mut data, &f.description);
        data.push_str(",\"choices\":[");
        for (j, c) in f.choices.iter().enumerate() {
            if j > 0 {
                data.push(',');
            }
            data.push_str("{\"code\":");
            push_go_string(&mut data, &c.code);
            data.push_str(",\"value\":");
            match &c.value {
                ChoiceValue::Bool(b) => data.push_str(if *b { "true" } else { "false" }),
                ChoiceValue::Str(s) => push_go_string(&mut data, s),
            }
            data.push_str(",\"description\":");
            push_go_string(&mut data, &c.description);
            data.push('}');
        }
        data.push_str("]}");
    }
    data.push_str("]}");

    let user_content = fields
        .iter()
        .map(|f| {
            let mut u = String::with_capacity(data.len() + f.name.len() + 24);
            u.push_str(&data);
            u.push_str("\n\nRequested field: ");
            push_go_string(&mut u, &f.name);
            u
        })
        .collect();
    Ok(Compiled {
        fields,
        user_content,
    })
}

/// Ollama `content()`: a JSON string is used as its text; an object or array as its
/// compacted JSON (verbatim bytes); anything else is refused.
///
/// `repaired`: the body had invalid UTF-8 (now U+FFFD). Go keeps such bytes raw in a
/// compacted object/array, and `json.Marshal` later writes each as the escape text `\\ufffd`;
/// a decoded JSON *string* holds a real U+FFFD instead. So only the object/array path marks
/// them ([`INVALID_BYTE`]) for the escaper. (A literal U+FFFD in such a body is marked too —
/// indistinguishable after repair, and vanishingly rare.)
fn content(raw: Option<&RawValue>, repaired: bool) -> Result<String, &'static str> {
    const WRONG: &str = "must be a string, object, or array";
    let text = raw.map_or("", |r| r.get().trim());
    match text.as_bytes().first() {
        Some(b'"') => go_unquote(text).ok_or(WRONG),
        Some(b'{' | b'[') if repaired => {
            Ok(compact(text).replace('\u{fffd}', &INVALID_BYTE.to_string()))
        }
        Some(b'{' | b'[') => Ok(compact(text)),
        _ => Err(WRONG),
    }
}

fn compile_field(name: &str, q: &Question, repaired: bool) -> Result<Field, String> {
    if name.trim().is_empty() {
        return Err("field name must not be empty".into());
    }
    let description = match content(q.instructions.as_deref(), repaired) {
        Ok(d) if !d.trim().is_empty() => d,
        _ => return Err("instructions must be a nonempty string, object, or array".into()),
    };
    let mut choices: Vec<Choice> = Vec::new();
    let mut add = |value: ChoiceValue, description: String| {
        // Codes exist only for A..Z; past that the 2–26 check below rejects the question, so
        // the placeholder is never shown. (`b'A' + n` as u8 overflowed from n = 191.)
        let code = u8::try_from(choices.len())
            .ok()
            .filter(|&n| usize::from(n) < MAX_CANDIDATES)
            .map_or('?', |n| char::from(b'A' + n));
        choices.push(Choice {
            code: code.to_string(),
            value,
            description,
        });
    };
    let typ = match q.typ.as_str() {
        "noul" => {
            let (mut no, mut yes) = ("No".to_owned(), "Yes".to_owned());
            if let Some(raw) = q.criteria.as_deref() {
                const MSG: &str = "noul criteria must be an object of true/false descriptions";
                if raw.get() == "null" {
                    return Err(MSG.into());
                }
                for (key, value) in string_or_null_map(raw).ok_or(MSG)? {
                    let Some(value) = value else {
                        return Err("noul descriptions must be strings".into());
                    };
                    match key.as_str() {
                        "false" => no = value,
                        "true" => yes = value,
                        _ => return Err(format!("unknown noul criterion {}", go_quote(&key))),
                    }
                }
            }
            add(ChoiceValue::Bool(false), no);
            add(ChoiceValue::Bool(true), yes);
            QuestionType::Noul
        }
        "choice" => {
            const MSG: &str = "choice criteria must map option keys to descriptions or null";
            let raw = q.criteria.as_deref().ok_or(MSG)?;
            // Go: unmarshalling `null` into the ordered map leaves it empty (→ the 2–26 check).
            let entries = if raw.get() == "null" {
                Vec::new()
            } else {
                string_or_null_map(raw).ok_or(MSG)?
            };
            for (key, value) in entries {
                if key.trim().is_empty() {
                    return Err("choice keys must not be empty".into());
                }
                let description = value.unwrap_or_else(|| key.clone());
                add(ChoiceValue::Str(key), description);
            }
            QuestionType::Choice
        }
        "score" => {
            const MSG: &str = "score criteria must be an array of descriptions";
            let raw = q.criteria.as_deref().ok_or(MSG)?;
            // Go: `null` into a slice leaves it nil (→ the 2–26 check), serde refuses it.
            let items: Vec<Option<String>> = if raw.get() == "null" {
                Vec::new()
            } else {
                let raws: Vec<Box<RawValue>> = serde_json::from_str(raw.get()).map_err(|_| MSG)?;
                raws.iter()
                    .map(|r| string_or_null(r).ok())
                    .collect::<Option<_>>()
                    .ok_or(MSG)?
            };
            for (i, description) in items.into_iter().enumerate() {
                let Some(description) = description else {
                    return Err("score descriptions must be strings".into());
                };
                add(ChoiceValue::Str(i.to_string()), description);
            }
            QuestionType::Score
        }
        _ => return Err("type must be choice, noul, or score".into()),
    };
    if !(MIN_CANDIDATES..=MAX_CANDIDATES).contains(&choices.len()) {
        return Err("criteria must contain 2–26 candidates".into());
    }
    Ok(Field {
        name: name.to_owned(),
        typ,
        description,
        choices,
    })
}

/// An ordered `{key: string | null}` object (Ollama `orderedmap[string, *string]`);
/// `None` when it is not one.
fn string_or_null_map(raw: &RawValue) -> Option<Vec<(String, Option<String>)>> {
    let obj: RawObject = serde_json::from_str(raw.get()).ok()?;
    obj.into_ordered_map()
        .into_iter()
        .map(|(k, v)| string_or_null(&v).ok().map(|v| (k, v)))
        .collect()
}

/// A Go `*string`: JSON null → `Ok(None)`, a string → `Ok(Some(s))`, anything else → `Err(())`.
fn string_or_null(raw: &RawValue) -> Result<Option<String>, ()> {
    if raw.get() == "null" {
        Ok(None)
    } else {
        go_unquote(raw.get()).map(Some).ok_or(())
    }
}

/// Ollama's `qwen3.5` renderer for `[system?, user]` with thinking off: both contents are
/// whitespace-trimmed; an empty system prompt means no system block.
#[must_use]
pub fn render_qwen35(system: &str, user: &str) -> String {
    let mut out = String::with_capacity(system.len() + user.len() + 96);
    if !system.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(system.trim());
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>user\n");
    out.push_str(user.trim());
    out.push_str("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::systemone::decode_request;

    /// Review R21: 191+ criteria overflowed the u8 code arithmetic (a panic in debug/test
    /// builds). Any count past 26 must give Ollama's 2–26 error, never a panic.
    #[test]
    fn huge_criteria_lists_are_rejected_not_panicking() {
        for n in [27usize, 190, 191, 255, 256, 300, 1000] {
            let criteria: serde_json::Map<String, serde_json::Value> = (0..n)
                .map(|i| (format!("k{i}"), serde_json::Value::Null))
                .collect();
            let score: Vec<String> = (0..n).map(|i| format!("d{i}")).collect();
            for q in [
                serde_json::json!({"type": "choice", "instructions": "i", "criteria": criteria}),
                serde_json::json!({"type": "score", "instructions": "i", "criteria": score}),
            ] {
                let body = serde_json::json!({"model": "m", "state": "s", "questions": {"a": q}});
                let err =
                    compile(&decode_request(body.to_string().as_bytes()).unwrap()).unwrap_err();
                assert_eq!(
                    err,
                    SystemOneError::Invalid(
                        "question \"a\": criteria must contain 2–26 candidates".into()
                    ),
                    "n = {n}"
                );
            }
        }
    }
}
