// ============================================================
// tests/golden_qwen3_template.rs — prompt-injection fidelity proof
// ============================================================
// Renders the real Qwen3 chat_template.jinja through our minijinja-based
// `build_prompt` and byte-compares against the committed expected output,
// which was produced by the genuine HF tokenizer:
//
//   python3 tests/fixtures/gen_expected.py
// (needs transformers + jinja2; see the script's own header for details)
//
// (= tokenizer.apply_chat_template(messages, tools=..., add_generation_prompt=True)).
//
// A byte match proves minijinja + minijinja-contrib(pycompat) + autoescape-off
// reproduce HF rendering exactly for Qwen3 with tools and tool-call history —
// the foundation OpenAI tool calling stands on for the Default family.
//
// The test is machine-local: `#[ignore]` keeps a default `cargo test` run
// honest — it shows as `ignored` (with the reason) rather than a silent `ok`
// on boxes without the model dir. Default model location is
// /opt/rustedvino/models/<id>; override with the QWEN3_MODEL_DIR env var if
// your model lives elsewhere. Run with `--include-ignored` on a box that has it (the runtime
// existence check in `assert_golden` below is kept as a second layer, in case
// `--include-ignored` runs on a fixture-less box).
// ============================================================

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use rustedvino::prompt_builder::build_prompt;

fn model_dir() -> String {
    std::env::var("QWEN3_MODEL_DIR")
        .unwrap_or_else(|_| "/opt/rustedvino/models/qwen3-8b-int4-ov".to_owned())
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Render `<stem>.json` through `build_prompt` and assert it byte-equals
/// `<stem>.expected.txt`. Skips when the model template is not on this box.
fn assert_golden(stem: &str) {
    let template_path = Path::new(&model_dir()).join("chat_template.jinja");
    if !template_path.exists() {
        eprintln!(
            "SKIP golden[{stem}]: {} not present on this machine",
            template_path.display()
        );
        return;
    }
    let template = std::fs::read_to_string(&template_path).unwrap();

    let dir = fixtures_dir();
    let fixture: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{stem}.json"))).unwrap())
            .unwrap();
    let expected = std::fs::read_to_string(dir.join(format!("{stem}.expected.txt"))).unwrap();

    let messages = fixture["messages"].as_array().unwrap();
    let tools = fixture["tools"].as_array().map(Vec::as_slice);

    let rendered = build_prompt(messages.as_slice(), tools, &template, "", "", None, None).unwrap();

    assert_eq!(
        rendered, expected,
        "minijinja render must byte-match apply_chat_template for fixture '{stem}'"
    );
}

#[test]
#[ignore = "requires the qwen3-8b-int4-ov model fixture (default /opt/rustedvino/models, \
            override with QWEN3_MODEL_DIR) — run with --include-ignored on a box that has it"]
fn golden_tools_simple() {
    assert_golden("tools_simple");
}

#[test]
#[ignore = "requires the qwen3-8b-int4-ov model fixture (default /opt/rustedvino/models, \
            override with QWEN3_MODEL_DIR) — run with --include-ignored on a box that has it"]
fn golden_tool_history() {
    assert_golden("tool_history");
}
