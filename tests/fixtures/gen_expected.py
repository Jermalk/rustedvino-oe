#!/usr/bin/env python3
"""Regenerate golden chat-template fixtures from the real HF tokenizer.

Run with any Python environment that has `transformers` installed:

    python3 tests/fixtures/gen_expected.py

For each `<name>.json` fixture ({messages, tools}), writes `<name>.expected.txt`
containing `tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
add_generation_prompt=True)`. The Rust golden test (tests/golden_qwen3_template.rs)
renders the same fixture through minijinja and byte-compares against this output,
proving prompt-injection fidelity. Re-run only when a fixture or the model's
chat_template changes.

Model dir defaults to /opt/rustedvino/models/qwen3-8b-int4-ov; override with the
QWEN3_MODEL_DIR env var (matches the Rust test's own override).
"""
import json
import os
import pathlib
import sys

from transformers import AutoTokenizer

MODEL_DIR = os.environ.get("QWEN3_MODEL_DIR", "/opt/rustedvino/models/qwen3-8b-int4-ov")
FIXTURES = pathlib.Path(__file__).parent

def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    for src in sorted(FIXTURES.glob("*.json")):
        data = json.loads(src.read_text())
        rendered = tok.apply_chat_template(
            data["messages"],
            tools=data.get("tools"),
            tokenize=False,
            add_generation_prompt=True,
        )
        out = src.with_suffix(".expected.txt")
        out.write_text(rendered)
        print(f"wrote {out.name} ({len(rendered)} chars)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
