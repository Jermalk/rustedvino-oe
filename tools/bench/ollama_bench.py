#!/usr/bin/env python3
"""ollama_bench - the Ollama twin of rvbench: the same workloads, the same prompts, the same metric names,
so an Ollama run and a RustedVINO run on one machine can be put side by side (`rvqa.py compare`).

Stdlib only; needs rvbench.py next to it (it reuses its prompts, image, medians and RSS sampler).

For each model of the suite it starts a *throwaway* `ollama serve` on a spare port with its own model directory
(your normal Ollama install and models are never touched), forces the CPU, pulls the model if it is missing,
runs the workload `--repeat` times, samples the server process tree's memory and clock, then stops it.

    python3 ollama_bench.py --out bench-ollama-cpu --repeat 3
    python3 ollama_bench.py --llm-tag qwen3:1.7b-q8_0 --only qwen3-1.7b-int4-ov      # a different quantisation
    python3 ollama_bench.py --vlm-tag qwen3-vl:4b --only qwen3.5-4b-int4-ov          # if qwen3.5 has no :4b

Results are keyed by the RUSTEDVINO model id the Ollama model stands in for (`qwen3-1.7b-int4-ov` etc.), with
the real Ollama tag, digest and quantisation recorded in each row (`ollama_tag`, `ollama_digest`, `ollama_quant`),
so `rvqa.py compare` lines the two engines up row for row.

What is and is not comparable (read before quoting a number):
  * Same: prompts, `temperature 0`, unique prompt prefix per request (defeats prefix caching in both engines),
    time-to-first-token = first streamed text, decode tok/s = completion tokens / (total - first-token time).
  * Different and unavoidable: Ollama runs GGUF models (llama.cpp, Q4_K_M by default) and RustedVINO runs
    OpenVINO IR models (INT4). Same family and size, similar bits per weight, NOT identical weights or kernels.
  * Ollama runs the context at `--num-ctx` (default 4096) so memory is not inflated by a huge default window.
  * Concurrency: the server is started with OLLAMA_NUM_PARALLEL=4 so the 2- and 4-stream numbers are real.
  * CPU only: CUDA / ROCm / Vulkan are hidden and every request carries num_gpu=0; the harness then CHECKS
    `/api/ps` (size_vram must be 0) and marks a row `gpu_used` if any layer reached a GPU.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rvbench as rb  # noqa: E402  (shares prompts, image, medians, RSS/clock sampling)

PORT = 11597
BASE = f"http://127.0.0.1:{PORT}"
KEEP_ALIVE = "30m"

# RustedVINO id (the comparison key)  ->  the Ollama model that stands in for it.
DEFAULT_SUITE = [
    {"id": "qwen3-1.7b-int4-ov", "kind": "llm", "ollama": "qwen3:1.7b"},
    {"id": "qwen3.5-4b-int4-ov", "kind": "vlm", "ollama": "qwen3.5:4b"},
    {"id": "multilingual-e5-large-int8-ov", "kind": "embed", "ollama": "bge-m3"},
]


# ---------------------------------------------------------------- HTTP
def post(path: str, body: dict, timeout: int = 1800, stream: bool = False):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=timeout)
    if stream:
        return r
    with r:
        return json.loads(r.read())


def get(path: str, timeout: int = 30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        raw = r.read()
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode(errors="replace")


def opts(args: argparse.Namespace, **extra) -> dict:
    o = {"num_gpu": 0, "num_ctx": args.num_ctx, **extra}
    if args.threads:
        o["num_thread"] = args.threads
    return o


def chat_stream(tag: str, prompt: str, max_tokens: int, args: argparse.Namespace, image_b64: str | None = None) -> dict:
    """One streamed /api/chat request, measured the way rvbench measures an OpenAI-style stream."""
    prompt = f"[{uuid.uuid4().hex[:10]}] {prompt}"  # unique prefix: no prefix-cache hits
    msg = {"role": "user", "content": prompt}
    if image_b64:
        msg["images"] = [image_b64]
    body = {"model": tag, "messages": [msg], "stream": True, "think": False, "keep_alive": KEEP_ALIVE,
            "options": opts(args, temperature=0, num_predict=max_tokens)}
    t, first, text, final, leaked = time.time(), None, "", {}, False
    with post("/api/chat", body, stream=True) as r:
        for line in r:
            if not line.strip():
                continue
            j = json.loads(line)
            m = j.get("message") or {}
            if m.get("thinking"):
                leaked = True
            piece = m.get("content") or ""
            if first is None and (piece or m.get("thinking")):
                first = time.time() - t
            text += piece
            if j.get("done"):
                final = j
    total = time.time() - t
    ct = final.get("eval_count", 0)
    dec = total - (first or 0)
    ed, pd = final.get("eval_duration") or 0, final.get("prompt_eval_duration") or 0
    return {"ttft_s": first, "total_s": total, "prompt_tokens": final.get("prompt_eval_count"), "completion_tokens": ct,
            "decode_tok_s": ct / dec if dec > 0 and ct else None, "text": text, "thinking_leaked": leaked,
            "ollama_decode_tok_s": ct / (ed / 1e9) if ed and ct else None,
            "ollama_prefill_tok_s": final.get("prompt_eval_count", 0) / (pd / 1e9) if pd else None}


# ---------------------------------------------------------------- workloads (mirror rvbench)
def w_llm(tag: str, n: int, ctx: dict) -> dict:
    a = ctx["args"]
    chat_stream(tag, "hi", 8, a)  # warm-up, not timed
    out: dict = {}
    for lang, p in rb.LLM_PROMPTS.items():
        rs = [chat_stream(tag, p, 128, a) for _ in range(n)]
        out[f"{lang}_ttft_s"] = rb.med([r["ttft_s"] for r in rs])
        out[f"{lang}_decode_tok_s"] = rb.med([r["decode_tok_s"] for r in rs])
        out[f"ollama_{lang}_decode_tok_s"] = rb.med([r["ollama_decode_tok_s"] for r in rs])
        out["thinking_leaked"] = out.get("thinking_leaked", False) or any(r["thinking_leaked"] for r in rs)
    rs = [chat_stream(tag, rb.LLM_LONG, 8, a) for _ in range(max(1, n - 1))]
    pt = rs[0]["prompt_tokens"] or 0
    out["long_prompt_tokens"] = pt
    out["long_prompt_ttft_s"] = rb.med([r["ttft_s"] for r in rs])
    out["prefill_tok_s"] = round(pt / out["long_prompt_ttft_s"], 1) if out["long_prompt_ttft_s"] else None
    out["ollama_prefill_tok_s"] = rb.med([r["ollama_prefill_tok_s"] for r in rs])
    for k in (2, 4):
        res: list = [None] * k

        def go(i: int) -> None:
            res[i] = chat_stream(tag, rb.LLM_PROMPTS["pl"], 96, a)

        t0 = time.time()
        ths = [threading.Thread(target=go, args=(i,)) for i in range(k)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        out[f"agg_tok_s_{k}_streams"] = round(sum(r["completion_tokens"] for r in res) / (time.time() - t0), 1)
    return out


def w_vlm(tag: str, n: int, ctx: dict) -> dict:
    a = ctx["args"]
    img = base64.b64encode(rb.make_png()).decode()
    ask = "Describe this image in one sentence: shapes and colours."
    chat_stream(tag, ask, 8, a, img)  # warm-up
    rs = [chat_stream(tag, ask, 64, a, img) for _ in range(n)]
    txt = rs[0]["text"].lower()
    t = [chat_stream(tag, rb.LLM_PROMPTS["en"], 96, a) for _ in range(n)]
    return {"image_ttft_s": rb.med([r["ttft_s"] for r in rs]), "image_decode_tok_s": rb.med([r["decode_tok_s"] for r in rs]),
            "image_prompt_tokens": rs[0]["prompt_tokens"], "text_decode_tok_s": rb.med([r["decode_tok_s"] for r in t]),
            "saw_red_and_blue": ("red" in txt and "blue" in txt), "image_answer": rs[0]["text"].strip()[:120],
            "thinking_leaked": any(r["thinking_leaked"] for r in rs + t)}


def w_embed(tag: str, n: int, ctx: dict) -> dict:
    a = ctx["args"]

    def emb(inp) -> tuple:
        t = time.time()
        d = post("/api/embed", {"model": tag, "input": inp, "keep_alive": KEEP_ALIVE, "options": opts(a)})
        return d, time.time() - t

    one, batch = rb.PL_TEXT, [rb.PL_TEXT + f" ({i})" for i in range(16)]  # no "passage: " prefix: bge-m3 needs none
    d, _ = emb(one)
    dims = len(d["embeddings"][0]) if d.get("embeddings") else None
    t1 = [emb(one)[1] for _ in range(n)]
    t16 = [emb(batch)[1] for _ in range(n)]
    return {"latency_1_input_s": rb.med(t1), "latency_16_inputs_s": rb.med(t16),
            "inputs_per_s_batch16": round(16 / rb.med(t16), 1) if rb.med(t16) else None, "embedding_dim": dims}


WORKLOADS = {"llm": w_llm, "vlm": w_vlm, "embed": w_embed}


# ---------------------------------------------------------------- server lifecycle
def server_env(models_dir: Path, args: argparse.Namespace) -> dict:
    env = {**os.environ,
           "OLLAMA_HOST": f"127.0.0.1:{PORT}", "OLLAMA_MODELS": str(models_dir),
           "OLLAMA_NUM_PARALLEL": "4", "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_KEEP_ALIVE": KEEP_ALIVE,
           # CPU only, belt and braces (the per-request num_gpu=0 and the /api/ps check are the real guards)
           "CUDA_VISIBLE_DEVICES": "-1", "HIP_VISIBLE_DEVICES": "-1", "ROCR_VISIBLE_DEVICES": "-1",
           "GGML_VK_VISIBLE_DEVICES": "-1", "OLLAMA_VULKAN": "0", "OLLAMA_LLM_LIBRARY": "cpu"}
    env.pop("OLLAMA_DEBUG", None)
    return env


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass


def run_entry(entry: dict, args: argparse.Namespace, ctx: dict) -> dict:
    tag = entry["ollama"]
    res: dict = {"id": entry["id"], "kind": entry["kind"], "ollama_tag": tag}
    work = Path(tempfile.mkdtemp(prefix="ollama-bench-"))
    log = open(work / "server.log", "w")
    proc = subprocess.Popen([args.ollama, "serve"], env=server_env(Path(args.models_dir).expanduser(), args), stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    sampler = rb.RssSampler(proc.pid)
    sampler.start()
    try:
        t0 = time.time()
        while True:  # wait for the server
            if proc.poll() is not None:
                return {**res, "error": "ollama serve exited: " + "".join(open(work / "server.log").readlines()[-3:])[:300]}
            try:
                get("/")
                break
            except (urllib.error.URLError, OSError, ValueError):
                if time.time() - t0 > 60:
                    return {**res, "error": "ollama serve did not start in 60 s"}
                time.sleep(0.3)
        ctx.setdefault("ollama_version", (get("/api/version") or {}).get("version", "?"))
        have = {m["name"]: m for m in get("/api/tags").get("models", [])}
        if not any(k == tag or k.startswith(tag + ":") or k == tag + ":latest" for k in have):
            if args.no_pull:
                return {**res, "error": f"{tag} is not in {args.models_dir} and --no-pull was given"}
            print(f"    pulling {tag} ...", flush=True)
            tp = time.time()
            post("/api/pull", {"model": tag, "stream": False}, timeout=7200)
            res["pull_s"] = round(time.time() - tp, 1)
            have = {m["name"]: m for m in get("/api/tags").get("models", [])}
        info = next((m for k, m in have.items() if k == tag or k.startswith(tag + ":") or k == tag + ":latest"), {})
        shown = post("/api/show", {"model": tag})
        det = shown.get("details", {})
        res.update({"ollama_digest": (info.get("digest") or "")[:12], "ollama_quant": det.get("quantization_level"),
                    "ollama_params": det.get("parameter_size"), "ollama_family": det.get("family"),
                    "ollama_size_mb": round((info.get("size") or 0) / 1048576), "ollama_capabilities": shown.get("capabilities")})
        if entry["kind"] == "vlm" and "vision" not in (shown.get("capabilities") or ["vision"]):
            return {**res, "error": f"{tag} reports no vision capability: {shown.get('capabilities')}"}
        # cold load, timed (the model is not resident in this fresh server)
        t_load = time.time()
        if entry["kind"] == "embed":
            r = post("/api/embed", {"model": tag, "input": "x", "keep_alive": KEEP_ALIVE, "options": opts(args)})
        else:
            r = post("/api/generate", {"model": tag, "prompt": "", "keep_alive": KEEP_ALIVE, "options": opts(args)})
        res["load_s"] = round(time.time() - t_load, 1)
        res["load_s_engine"] = round(r["load_duration"] / 1e9, 1) if r.get("load_duration") else None  # absent on an empty-prompt load
        res["rss_after_load_mb"] = round(rb.tree_rss_mb(proc.pid) or 0) or None
        ps = (get("/api/ps") or {}).get("models", [])
        vram = sum(m.get("size_vram", 0) for m in ps)
        res["gpu_used"] = vram > 0
        res["ollama_resident_mb"] = round(sum(m.get("size", 0) for m in ps) / 1048576)
        if vram > 0:
            res["warning"] = f"{vram >> 20} MB of the model is on a GPU - this is NOT a CPU-only measurement"
        sampler.peak, sampler.mhz = 0.0, []
        res.update(WORKLOADS[entry["kind"]](tag, args.repeat, ctx))
        res["rss_peak_mb"] = round(max(sampler.peak, rb.tree_rss_mb(proc.pid) or 0)) or None
        res["cpu_mhz_avg"] = round(rb.st.mean(sampler.mhz)) if sampler.mhz else None
    except Exception as e:  # noqa: BLE001 - a failed model must not abort the whole run
        res["error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        sampler.stop_evt.set()
        stop_server(proc)
        log.close()
        shutil.rmtree(work, ignore_errors=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ollama", default=shutil.which("ollama") or "ollama", help="path to the ollama binary")
    ap.add_argument("--models-dir", default="~/ollama-bench-models", help="OLLAMA_MODELS for the throwaway servers (kept between runs)")
    ap.add_argument("--out", default="bench-ollama")
    ap.add_argument("--suite", help="JSON list of {id, kind, ollama}; default = qwen3:1.7b, qwen3.5:4b, bge-m3")
    ap.add_argument("--llm-tag", help="Ollama tag standing in for qwen3-1.7b-int4-ov (e.g. qwen3:1.7b-q8_0)")
    ap.add_argument("--vlm-tag", help="Ollama tag standing in for qwen3.5-4b-int4-ov (e.g. qwen3-vl:4b)")
    ap.add_argument("--embed-tag", help="Ollama tag standing in for multilingual-e5-large-int8-ov (e.g. qwen3-embedding:0.6b)")
    ap.add_argument("--only", action="append", default=[], help="run only this RustedVINO id (repeatable)")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--num-ctx", type=int, default=4096, help="context window for every request (default 4096)")
    ap.add_argument("--threads", type=int, default=0, help="num_thread for every request (default: Ollama's own choice)")
    ap.add_argument("--no-pull", action="store_true", help="fail instead of downloading a missing model")
    ap.add_argument("--allow-busy", action="store_true", help="run even if the machine is not quiet (result is flagged)")
    args = ap.parse_args()

    if not (shutil.which(args.ollama) or os.access(args.ollama, os.X_OK)):
        print(f"ollama binary not found: {args.ollama}", file=sys.stderr)
        return 2
    machine = rb.machine_info("CPU")
    busy = machine["loadavg"] is not None and machine["loadavg"] > 0.35 * (machine["logical_cpus"] or 1)
    if busy and not args.allow_busy:
        print(f"machine not quiet (loadavg {machine['loadavg']:.1f} on {machine['logical_cpus']} threads); "
              "close other apps or pass --allow-busy", file=sys.stderr)
        return 2
    machine["quiet_at_start"] = not busy
    suite = json.load(open(args.suite)) if args.suite else [dict(e) for e in DEFAULT_SUITE]
    for e in suite:
        tag = {"llm": args.llm_tag, "vlm": args.vlm_tag, "embed": args.embed_tag}.get(e["kind"])
        if tag:
            e["ollama"] = tag
    suite = [e for e in suite if not args.only or e["id"] in args.only]
    Path(args.models_dir).expanduser().mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    machine["engine"] = "ollama"
    machine["host"] = machine["hostname"]
    machine["hostname"] = machine["hostname"] + "/ollama"  # a separate column in `rvqa.py compare`
    machine["num_ctx"] = args.num_ctx
    machine["num_thread"] = args.threads or "ollama default"
    ctx: dict = {"args": args}
    results: list[dict] = []
    for e in suite:
        print(f"[{time.strftime('%H:%M:%S')}] {e['id']} <- ollama {e['ollama']} ({e['kind']}) ...", flush=True)
        r = run_entry(e, args, ctx)
        r["loadavg_end"] = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
        results.append(r)
        print("   ", {k: v for k, v in r.items() if k not in ("id", "kind")}, flush=True)
        machine["server_version"] = "ollama " + str(ctx.get("ollama_version", "?"))
        (out / "results.json").write_text(json.dumps({"machine": machine, "results": results}, indent=1))
        (out / "results.md").write_text(rb.to_markdown(machine, results))
    print(f"\nwritten: {out/'results.json'}  {out/'results.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
