// ============================================================
// tests/systemone_golden.rs — /v1/systemone wire parity with Ollama v0.35.0
// ============================================================
// Every case in tests/fixtures/systemone_golden.json was produced by Ollama's own Go code
// (json.Unmarshal → decision.Compile → qwen3.5 renderer → Answer) via
// a small Go harness around Ollama v0.35.0, built with Go 1.26.0 as Ollama's go.mod pins
// (kept with the project's development notes).
// We must reproduce: the exact prompt bytes (they decide the probabilities), Ollama's exact
// validation messages, and Answer()'s numbers. Bind (decode) errors match on status only —
// Go's decoder wording is not reproducible.
// ============================================================

#![allow(clippy::unwrap_used, clippy::expect_used)]

use rustedvino::systemone::{SystemOneError, answer, compile, decode_request};
use serde_json::Value;

/// Cases where we knowingly differ from Ollama, with the reason (kept tiny and explicit).
const KNOWN_DIVERGENCES: &[(&str, &str)] = &[];

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/systemone_golden.json")).unwrap()
}

fn body(case: &Value) -> Vec<u8> {
    if let Some(b64) = case["body_b64"].as_str() {
        return base64_decode(b64);
    }
    case["body"].as_str().unwrap_or("").as_bytes().to_vec() // Go omitempty drops ""
}

fn base64_decode(s: &str) -> Vec<u8> {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::new();
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes().filter(|c| *c != b'=') {
        acc = (acc << 6) | u32::try_from(T.iter().position(|t| *t == c).unwrap()).unwrap();
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(u8::try_from((acc >> bits) & 0xff).unwrap());
        }
    }
    out
}

/// Structural JSON equality: same key order, strings/bools exact, numbers within 1e-12.
fn same_json(a: &Value, b: &Value, path: &str) -> Result<(), String> {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let (kx, ky): (Vec<_>, Vec<_>) = (x.keys().collect(), y.keys().collect());
            if kx != ky {
                return Err(format!("{path}: keys {kx:?} != {ky:?}"));
            }
            x.iter()
                .try_for_each(|(k, v)| same_json(v, &y[k], &format!("{path}.{k}")))
        }
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            if (x - y).abs() <= 1e-12 {
                Ok(())
            } else {
                Err(format!("{path}: {x} != {y}"))
            }
        }
        _ if a == b => Ok(()),
        _ => Err(format!("{path}: {a} != {b}")),
    }
}

fn check_case(case: &Value, system: &str) -> Result<(), String> {
    let decoded = decode_request(&body(case));
    if case["too_large"].as_bool() == Some(true) {
        return match decoded {
            Err(e @ SystemOneError::TooLarge) if e.status() == 413 => Ok(()),
            other => Err(format!("want 413 TooLarge, got {other:?}")),
        };
    }
    if case.get("bind_error").is_some() {
        return match decoded {
            Err(e @ SystemOneError::Bind(_)) if e.status() == 400 => Ok(()),
            other => Err(format!("want a bind error, got {other:?}")),
        };
    }
    let req = decoded.map_err(|e| format!("unexpected decode error: {e}"))?;
    let compiled = compile(&req);
    if let Some(want) = case["compile_error"].as_str() {
        return match compiled {
            Err(SystemOneError::Invalid(got)) if got == want => Ok(()),
            other => Err(format!("want Invalid({want:?}), got {other:?}")),
        };
    }
    let compiled = compiled.map_err(|e| format!("unexpected compile error: {e}"))?;
    let n = case["n_questions"].as_u64().unwrap();
    if compiled.fields.len() as u64 != n {
        return Err(format!("{} questions, want {n}", compiled.fields.len()));
    }
    let indices: Vec<usize> = match case.get("prompt_indices") {
        Some(ix) => ix
            .as_array()
            .unwrap()
            .iter()
            .map(|i| usize::try_from(i.as_u64().unwrap()).unwrap())
            .collect(),
        None => (0..compiled.fields.len()).collect(),
    };
    let prompts = compiled.prompts(system);
    let candidates = compiled.candidates();
    for (k, &i) in indices.iter().enumerate() {
        let want_user = case["user_content"][k].as_str().unwrap();
        if compiled.user_content[i] != want_user {
            return Err(format!(
                "user_content[{i}] differs:\n ours {:?}\n want {want_user:?}",
                compiled.user_content[i]
            ));
        }
        let want_prompt = case["prompts"][k].as_str().unwrap();
        if prompts[i] != want_prompt {
            return Err(format!(
                "prompt[{i}] differs:\n ours {:?}\n want {want_prompt:?}",
                prompts[i]
            ));
        }
        let want_cand: Vec<String> = serde_json::from_value(case["candidates"][k].clone()).unwrap();
        if candidates[i] != want_cand {
            return Err(format!(
                "candidates[{i}] {:?} != {want_cand:?}",
                candidates[i]
            ));
        }
    }
    if let Some(logits) = case.get("logits").filter(|l| !l.is_null()) {
        let logits: Vec<Vec<f32>> = serde_json::from_value(logits.clone()).unwrap();
        let got = answer(&req.model, &compiled, &logits, 7, 1);
        if let Some(want) = case["answer_error"].as_str() {
            return match got {
                Err(SystemOneError::Scorer(m)) if m == want => Ok(()),
                other => Err(format!("want Scorer({want:?}), got {other:?}")),
            };
        }
        let got = got.map_err(|e| format!("unexpected answer error: {e}"))?;
        same_json(&got, &case["answer"], "answer")?;
    }
    Ok(())
}

#[test]
fn systemone_matches_ollama_goldens() {
    let fx = fixture();
    let system = fx["system_prompt"].as_str().unwrap();
    let cases = fx["cases"].as_array().unwrap();
    assert!(cases.len() >= 80, "fixture has {} cases", cases.len());
    let mut failures = Vec::new();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let result = check_case(case, system);
        match (result, KNOWN_DIVERGENCES.iter().find(|(n, _)| *n == name)) {
            (Err(e), None) => failures.push(format!("{name}: {e}")),
            (Ok(()), Some(_)) => failures.push(format!(
                "{name}: listed as a divergence but now matches — remove it"
            )),
            _ => {}
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} cases differ from Ollama:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}
