#!/usr/bin/env python3
"""cmptable - turn cmpbench result folders into one markdown comparison table.

    python3 cmptable.py RESULTS_DIR PASS_DIR [PASS_DIR ...]      # e.g. cmptable.py results pass1 pass2

Each pass dir (relative to RESULTS_DIR, the --out folder you gave cmpbench) holds <stack>/<model>/results.json. Prints the versions
every stack ran with first, then one table per model; cells are `pass1 / pass2 / ...` when several passes exist. Numbers only: the
table does not interpret them. Read CMPBENCH.md for what they can and cannot support.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

COLS = [("load_cold_s", "load cold s"), ("load_warm_s", "load warm s"), ("en_decode_tok_s", "en decode tok/s"),
        ("pl_decode_tok_s", "pl decode tok/s"), ("fixed256_decode_tok_s", "fixed256 tok/s (usage)"),
        ("fixed256_ms_per_token", "fixed256 ms/tok (chunks)"), ("en_ttft_s", "en TTFT s"), ("prefill_tok_s", "prefill tok/s"),
        ("agg_tok_s_2_streams", "2 streams tok/s"), ("agg_tok_s_4_streams", "4 streams tok/s"),
        ("rss_after_load_mb", "RSS after load MB"), ("rss_peak_workload_mb", "RSS peak, workload only MB"),
        ("vram_after_load_gib", "VRAM after load GiB"), ("vram_peak_workload_gib", "VRAM peak, workload only GiB")]


def genai_release(v: str) -> str:
    """The YYYY.N.N prefix of a version string, so two builds of one release compare equal; '' when unknown."""
    import re
    m = re.match(r"\s*(20\d\d\.\d+\.\d+)", str(v))
    return m.group(1) if m else ""


def esc(x: object) -> str:
    return str(x).replace("|", "/").replace("\n", " ")


def checks_cell(r: dict | None) -> str:
    if not r:
        return ""
    c = r.get("checks")
    if not c:
        return "old result (no checks recorded)" if "error" not in r else "ERROR"
    return ("OK" if not c.get("failed") else "FAILED: " + esc("; ".join(c["failed"]))) + \
        f" (think {c.get('think')}, fixed256 {c.get('fixed256')}, no-usage {c.get('no_usage')}, gpu-before {c.get('gpu_before_gib')} GiB)"


def load(root: Path, pass_dir: str) -> dict:
    out: dict = {}
    for f in sorted((root / pass_dir).glob("*/*/results.json")):
        out[(f.parts[-3], f.parts[-2])] = json.loads(f.read_text())
    return out


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    passes = sys.argv[2:]
    data = [load(root, p) for p in passes]
    if not any(data):
        print(f"cmptable: no <stack>/<model>/results.json under {root} / {passes}", file=sys.stderr)
        return 2
    models = sorted({m for d in data for (_, m) in d})
    print("## Versions each stack ran with\n")
    print("| stack | pass / model | server | OpenVINO | OpenVINO GenAI | device | kernel |")
    print("|---|---|---|---|---|---|---|")
    rows_v: list[tuple] = []
    for stk in ("rv", "ovms", "openarc"):
        for p, d in zip(passes, data):
            for (s_, m_), r in sorted(d.items()):
                if s_ != stk:
                    continue
                v = r.get("versions") or {}
                if isinstance(v, str):  # results written by an older cmpbench
                    v = {"server": v}
                rows_v.append((stk, p, m_, v.get("server", "?"), v.get("openvino", "?"), v.get("genai", "?"), r.get("device", "?"),
                               r.get("kernel", "?")))
    for stk, p, m_, *rest in rows_v:
        print(f"| {stk} | {p} / {m_} | " + " | ".join(esc(x) for x in rest[:3]) + f" | {esc(rest[3])} | {esc(rest[4])} |")
    rel = {genai_release(row[5]) for row in rows_v}
    if "" in rel:
        print("\n**Some results carry no OpenVINO GenAI version: their engine is unknown.**")
        rel.discard("")
    if len(rel) > 1:
        print("\n**The runs used different OpenVINO GenAI releases " + ", ".join(sorted(rel)) +
              " (across stacks, models or passes): every gap below is server + engine together, not the server alone.**")
    for stk in ("rv", "ovms", "openarc"):
        if len({row[3:6] for row in rows_v if row[0] == stk}) > 1:
            print(f"\n**{stk}: the version changed between passes or models - compare the rows above before reading the cells.**")
    for m in models:
        print(f"\n### {m}\n")
        print("| stack | " + " | ".join(h for _, h in COLS) + " |")
        print("|---|" + "---|" * len(COLS))
        for st in ("rv", "ovms", "openarc"):
            rows = [d.get((st, m)) for d in data]
            if not any(rows):
                continue
            cells = []
            for k, _ in COLS:
                vals = []
                for r in rows:
                    if r is None:
                        vals.append("-")  # keeps the pass position when a pass lacks this stack
                        continue
                    v = r.get(k)
                    if v is None and "_peak_workload_" in k:
                        v = r.get(k.replace("_workload", ""))  # results written before the rename
                    vals.append("-" if v is None else (f"{v:.1f}" if isinstance(v, float) and abs(v) >= 10 else str(v)))
                cells.append(" / ".join(vals))
            err = [r.get("error") for r in rows if r and r.get("error")]
            print(f"| {st} | " + " | ".join(cells) + " |" + (f" ERROR {esc(err[0][:80])}" if err else ""))
        for st in ("rv", "ovms", "openarc"):
            for p, d in zip(passes, data):
                r = d.get((st, m))
                if r:
                    v = r.get("versions")
                    v = v if isinstance(v, str) else " / ".join(str(x) for x in (v or {}).values())
                    print(f"\n- {p}/{st}: {esc(v)}; stop {r.get('stop')}; equiv {esc(r.get('equivalence', {}).get('text', ''))!r} "
                          f"(prompt_tokens {r.get('equivalence', {}).get('prompt_tokens')}); checks: {checks_cell(r)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
