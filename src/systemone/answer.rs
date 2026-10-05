// ============================================================
// src/systemone/answer.rs — logits → probabilities → the /v1/systemone response
// ============================================================
// Port of Ollama v0.35.0 `(*Compiled).Answer`. Per question, over the k candidate logits:
//   p        = softmax(logits)                 (computed in f64, shifted by the max logit)
//   choice   = the value with the highest p    (first one on a tie)
//   noul     = p[true]
//   score    = Σ index · p[index]              (expected ordinal)
//   confidence = clamp(1 − H(p) / ln k, 0, 1)  (1 − normalised entropy)
//
// CRASH COURSE — why f64 when the model gives f32:
//   exp() of a logit difference loses precision fast in f32; Go widens each logit to float64
//   before subtracting, so we do the same and get the same numbers to the last bit or two.
// ============================================================

use serde_json::{Map, Value, json};

use super::SystemOneError;
use super::compile::{ChoiceValue, Compiled, QuestionType};

/// Build the response body from one row of candidate logits per question.
///
/// # Errors
/// [`SystemOneError::Scorer`] (HTTP 500) with Ollama's message when a row count or width does
/// not match the request, or a logit is NaN / infinite.
pub fn answer(
    model: &str,
    compiled: &Compiled,
    logits: &[Vec<f32>],
    input_tokens: usize,
    output_tokens: usize,
) -> Result<Value, SystemOneError> {
    if logits.len() != compiled.fields.len() {
        return Err(SystemOneError::Scorer(format!(
            "scorer returned {} rows for {} questions",
            logits.len(),
            compiled.fields.len()
        )));
    }
    let mut answers = Map::new();
    for (field, row) in compiled.fields.iter().zip(logits) {
        let quoted = super::gojson::go_quote(&field.name);
        if row.len() != field.choices.len() {
            return Err(SystemOneError::Scorer(format!(
                "scorer returned the wrong number of candidates for {quoted}"
            )));
        }
        if row.iter().any(|l| !l.is_finite()) {
            return Err(SystemOneError::Scorer(format!(
                "scorer returned a non-finite logit for {quoted}"
            )));
        }
        let peak = f64::from(row.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let mut p: Vec<f64> = row.iter().map(|&l| (f64::from(l) - peak).exp()).collect();
        let sum: f64 = p.iter().sum();
        let (mut entropy, mut score, mut index) = (0.0_f64, 0.0_f64, 0.0_f64);
        for pj in &mut p {
            *pj /= sum;
            score += index * *pj;
            index += 1.0;
            if *pj > 0.0 {
                entropy -= *pj * pj.ln();
            }
        }
        if field.typ == QuestionType::Noul {
            answers.insert(field.name.clone(), json!({"type": "noul", "noul": p[1]}));
            continue;
        }
        let mut probabilities = Map::new();
        let mut legend = Map::new();
        for (c, pj) in field.choices.iter().zip(&p) {
            let key = match &c.value {
                ChoiceValue::Str(s) => s.clone(),
                ChoiceValue::Bool(b) => b.to_string(),
            };
            probabilities.insert(key.clone(), json!(pj));
            legend.insert(key, Value::String(c.description.clone()));
        }
        let confidence = (1.0 - entropy / index.ln()).clamp(0.0, 1.0);
        let body = if field.typ == QuestionType::Choice {
            // First index of the maximum, like Go's `slices.Index(p, slices.Max(p))`.
            let winner = p
                .iter()
                .enumerate()
                .fold(0, |best, (j, pj)| if *pj > p[best] { j } else { best });
            let choice = match &field.choices[winner].value {
                ChoiceValue::Str(s) => s.clone(),
                ChoiceValue::Bool(b) => b.to_string(),
            };
            json!({"type": "choice", "choice": choice, "probabilities": probabilities, "confidence": confidence})
        } else {
            json!({"type": "score", "score": score, "legend": legend, "probabilities": probabilities, "confidence": confidence})
        };
        answers.insert(field.name.clone(), body);
    }
    Ok(json!({
        "model": model,
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }))
}
