// ============================================================
// tests/systemone_engine.rs — live check: Rust renderer + ov_decide bridge vs the Python spike
// ============================================================
// Renders the brief's example request with `rustedvino::systemone`, tokenises it with the
// bridge (`ov_decide_encode`), prefills each question (`ov_decide_prefill`) and compares the
// candidate logits with a reference result (Python + OpenVINO, same prompt, DQ off) for the
// same gathered IR and device. Same OpenVINO, same IR, same prompt bytes, same
// compile properties → the logits must agree to float rounding.
//
// Machine-local (needs a gathered decision IR), so `#[ignore]`d. Run:
//   SYSTEMONE_IR=~/.cache/systemone-spike/qwen35-0.8b-text-int8-cw-last \
//   SYSTEMONE_EXPECT=/path/to/reference.json SYSTEMONE_DEVICE=GPU \
//   scripts/rv-cargo.sh test --test systemone_engine -- --ignored --nocapture
// ============================================================

#![allow(clippy::unwrap_used, clippy::expect_used)]
// The Python JSON stores f32 logits widened to f64; narrowing them back is exact.
#![allow(clippy::cast_possible_truncation)]

use rustedvino::ov_decide::OvDecideEngine;
use rustedvino::systemone::{compile, decode_request};
use serde_json::Value;

const BRIEF: &str = r#"{"model":"nimble","state":"Our checkout has returned 500 errors since 9am.","questions":{"label":{"type":"choice","instructions":"Which label fits this ticket?","criteria":{"billing":"Payments and refunds","bug":"Software errors","account":"Login and account access"}},"refund":{"type":"noul","instructions":"Is the customer asking for a refund?"},"urgency":{"type":"score","instructions":"How urgent is this?","criteria":["Routine","Soon","Urgent"]}}}"#;

#[test]
#[ignore = "needs a gathered decision IR: set SYSTEMONE_IR / SYSTEMONE_EXPECT / SYSTEMONE_DEVICE"]
fn bridge_matches_python_spike() {
    let ir = std::env::var("SYSTEMONE_IR").expect("SYSTEMONE_IR");
    let expect: Value = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("SYSTEMONE_EXPECT").expect("SYSTEMONE_EXPECT"))
            .unwrap(),
    )
    .unwrap();
    let device = std::env::var("SYSTEMONE_DEVICE").unwrap_or_else(|_| "GPU".into());
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/systemone_golden.json")).unwrap();
    let system = fixture["system_prompt"].as_str().unwrap();

    let mut engine = OvDecideEngine::new(&ir, &device, "", 8192).unwrap();
    // Candidate codes must each be one ordinary token (the sidecar carries these ids in production).
    let codes: Vec<i64> = ["A", "B", "C"]
        .iter()
        .map(|c| {
            let ids = engine.encode(c).unwrap();
            assert_eq!(ids.len(), 1, "code {c} is not one token: {ids:?}");
            ids[0]
        })
        .collect();
    assert_eq!(codes, [32, 33, 34], "Qwen3.5 code-letter ids");
    // All 26 codes, as a sidecar would list them, pass the load-time check (standalone id and
    // exactly one appended token after a rendered prompt).
    let meta = rustedvino::decision_engine::DecisionMeta {
        prompt_format: "ollama".into(),
        system_prompt: system.to_owned(),
        candidate_ids: (b'A'..=b'Z')
            .map(|c| engine.encode(&char::from(c).to_string()).unwrap()[0])
            .collect(),
        context_length: 8194,
        licence: None,
    };
    meta.verify(&engine).unwrap();

    // Guard rails are enforced in the bridge, before the request is touched.
    assert!(
        engine.prefill(&[], &codes).is_err(),
        "n = 0 must be refused"
    );
    assert!(
        engine
            .prefill(&[1, 2], &[i64::try_from(engine.vocab_size()).unwrap()])
            .is_err(),
        "cand >= V"
    );
    assert!(engine.prefill(&[1, 2], &[-1]).is_err(), "negative cand");

    let compiled = compile(&decode_request(BRIEF.as_bytes()).unwrap()).unwrap();
    let mut worst = 0f32;
    for (field, prompt) in compiled.fields.iter().zip(compiled.prompts(system)) {
        let ids = engine.encode(&prompt).unwrap();
        let want = &expect["answers"][&field.name];
        assert_eq!(
            ids.len() as u64,
            want["prompt_tokens"].as_u64().unwrap(),
            "{}: token count",
            field.name
        );
        let cand = &codes[..field.choices.len()];
        let got = engine.prefill(&ids, cand).unwrap();
        let again = engine.prefill(&ids, cand).unwrap();
        assert_eq!(
            got, again,
            "{}: repeat after reset must be bit-identical",
            field.name
        );
        let want: Vec<f32> = want["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        for (g, w) in got.iter().zip(&want) {
            worst = worst.max((g - w).abs());
        }
        println!(
            "{:8} n={:3} ours {got:?} python {want:?}",
            field.name,
            ids.len()
        );
    }
    println!("max |Δlogit| vs python = {worst:e}");
    assert!(
        worst <= 1e-3,
        "bridge logits differ from the Python spike by {worst}"
    );
}
