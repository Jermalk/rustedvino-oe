#!/usr/bin/env python3
"""rvbench - repeatable speed + RAM benchmark for RustedVINO's starter models, one table per machine.

Stdlib only, so the same file runs unchanged on every machine in a comparison matrix.

For each model in the suite it starts a *throwaway* server on a spare port (so a model's RAM is
measured on its own, and a 16 GB machine never holds two models at once), runs a fixed workload
repeated N times, samples the server process's memory, then stops it. Nothing touches a running
production server or its config.

    python3 tools/bench/rvbench.py --launcher ./rustedvino --models-dir ~/models --out bench-out
    python3 tools/bench/rvbench.py ... --suite my_suite.json --only qwen3-0.6b-int4-ov --repeat 1
    python3 tools/bench/rvbench.py ... --device GPU        # same suite on an iGPU, to compare

Writes <out>/results.json (machine + per-model numbers) and <out>/results.md (tables to paste).
Memory is read from /proc (Linux); on other systems the RAM columns are n/a.

STT needs audio. By default it is synthesised by the suite's TTS model, so no recording ships with
the tool; this measures speed (real-time factor), NOT recognition accuracy. Pass --audio FILE.wav
to time real speech instead.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics as st
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

PORT = 11598

# Fixed inputs: the same on every machine, so results are comparable.
PL_TEXT = ("Zarządzanie to nie są decyzje ekonomiczne, to są decyzje podejmowane w organizacjach. "
           "Teoria organizacji dała początek naukom o zarządzaniu, ponieważ nie ma zarządzania bez organizacji. ")
LLM_PROMPTS = {
    "en": "Explain in two sentences what a vector database is.",
    "pl": "Wyjaśnij w dwóch zdaniach, czym jest baza danych wektorowa.",
}
LLM_LONG = (PL_TEXT * 12) + "Odpowiedz jednym słowem: o czym jest ten tekst?"
TTS_TEXT = {
    "en": ("The lighthouse keeper found a bottle with a letter on the beach yesterday evening. "
           "He read it twice before he believed it. Tomorrow morning he will sail to the city "
           "and look for the person who wrote it. "),
    "vi": "Người giữ đèn biển tìm thấy một chai có lá thư trên bờ biển. Anh ấy đọc hai lần trước khi tin. ",
    "pl": "Latarnik znalazł wczoraj wieczorem na plaży butelkę z listem. Przeczytał go dwa razy, zanim uwierzył. ",
}
RERANK_QUERY = "Czym zajmuje się teoria organizacji?"
RERANK_DOCS = [PL_TEXT, "Warszawa jest stolicą Polski i leży nad Wisłą.",
               "Teoria organizacji bada struktury, procesy i zachowania ludzi w organizacjach.",
               "Przepis na pierogi z kapustą i grzybami wymaga ciasta i długiego gotowania.",
               "Nauki o zarządzaniu obejmują planowanie, organizowanie i kontrolowanie zasobów."] * 4

# kind -> the `kind` field in the generated server config
CFG_KIND = {"llm": "text_gen", "vlm": "vision", "embed": "embedding", "rerank": "reranking", "stt": "stt", "tts": "tts"}

DEFAULT_SUITE = [
    # The six README starter models, one per modality; order matters: the TTS model runs before STT because it
    # synthesises the STT test audio. Keep in step with README.md step 3 (the QA catalogue's BEN-01 uses this).
    {"id": "qwen3-1.7b-int4-ov", "kind": "llm"},
    {"id": "qwen3.5-4b-int4-ov", "kind": "vlm"},
    {"id": "multilingual-e5-large-int8-ov", "kind": "embed"},
    {"id": "bge-reranker-v2-m3-int8-ov", "kind": "rerank"},
    {"id": "speecht5-tts-ov", "kind": "tts", "lang": "en", "make_audio": True},
    {"id": "whisper-large-v3-turbo-int8-ov", "kind": "stt"},
]


# ---------------------------------------------------------------- machine + process info
def _read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def machine_info(device: str) -> dict:
    cpu = ""
    for line in _read("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    mem = {k: int(v.split()[0]) for k, v in (l.split(":", 1) for l in _read("/proc/meminfo").splitlines() if ":" in l)}
    ac = [_read(f"/sys/class/power_supply/{p}/online") for p in os.listdir("/sys/class/power_supply")
          if p.startswith(("AC", "ADP"))] if os.path.isdir("/sys/class/power_supply") else []
    return {
        "hostname": platform.node(), "cpu": cpu or platform.processor(), "logical_cpus": os.cpu_count(),
        "ram_total_gb": round(mem.get("MemTotal", 0) / 1048576, 1),
        "ram_available_gb": round(mem.get("MemAvailable", 0) / 1048576, 1),
        "os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
        "cpu_governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "on_ac_power": (ac[0] == "1") if ac and ac[0] in ("0", "1") else None,
        "loadavg": os.getloadavg()[0] if hasattr(os, "getloadavg") else None, "device": device,
    }


def tree_rss_mb(pid: int) -> float | None:
    """Resident memory (MB) summed over `pid` and its descendants. Linux only."""
    if not os.path.isdir("/proc"):
        return None
    kids: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            m = re.search(r"^PPid:\s*(\d+)", _read(f"/proc/{d}/status"), re.M)
            if m:
                kids.setdefault(int(m.group(1)), []).append(int(d))
    total, stack = 0.0, [pid]
    while stack:
        p = stack.pop()
        m = re.search(r"^VmRSS:\s*(\d+) kB", _read(f"/proc/{p}/status"), re.M)
        total += int(m.group(1)) / 1024 if m else 0
        stack += kids.get(p, [])
    return total


def cpu_mhz_now() -> float | None:
    """Mean current core clock in MHz (Linux cpufreq); None where the kernel does not expose it."""
    vals = []
    base = "/sys/devices/system/cpu"
    if os.path.isdir(base):
        for d in os.listdir(base):
            if re.fullmatch(r"cpu\d+", d):
                v = _read(f"{base}/{d}/cpufreq/scaling_cur_freq")
                if v.isdigit():
                    vals.append(int(v) / 1000)
    return sum(vals) / len(vals) if vals else None


class RssSampler(threading.Thread):
    """Samples the server's RSS and the CPU clock every 500 ms; `peak` is the max RSS seen."""

    def __init__(self, pid: int) -> None:
        super().__init__(daemon=True)
        self.pid, self.peak, self.stop_evt, self.mhz = pid, 0.0, threading.Event(), []

    def run(self) -> None:
        while not self.stop_evt.is_set():
            v = tree_rss_mb(self.pid)
            if v:
                self.peak = max(self.peak, v)
            f = cpu_mhz_now()
            if f:
                self.mhz.append(f)
            self.stop_evt.wait(0.5)


# ---------------------------------------------------------------- HTTP
BASE = f"http://127.0.0.1:{PORT}"


def post_json(path: str, body: dict, timeout: int = 900) -> tuple[dict | bytes, float]:
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    dt = time.time() - t
    try:
        return json.loads(raw), dt
    except ValueError:
        return raw, dt


def post_audio(model: str, wav: bytes, timeout: int = 1800) -> float:
    b = uuid.uuid4().hex
    body = (f'--{b}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{model}\r\n--{b}\r\n'
            f'Content-Disposition: form-data; name="file"; filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()
    body += wav + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(BASE + "/v1/audio/transcriptions", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()
    return time.time() - t


def chat_stream(model: str, prompt: str, max_tokens: int) -> dict:
    # A unique tag up front defeats the server's prefix cache (on by default): without it, repeat runs of
    # the same prompt measure the cache, not prefill, and time-to-first-token collapses to ~0.
    prompt = f"[{uuid.uuid4().hex[:10]}] {prompt}"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "enable_thinking": False}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t, first, usage = time.time(), None, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            j = json.loads(line[5:])
            usage = j.get("usage") or usage
            for c in j.get("choices", []):
                if first is None and (c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content")):
                    first = time.time() - t
    total = time.time() - t
    ct = (usage or {}).get("completion_tokens", 0)
    dec = total - (first or 0)
    return {"ttft_s": first, "total_s": total, "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": ct, "decode_tok_s": ct / dec if dec > 0 and ct else None}


def wav_seconds(data: bytes) -> float:
    import io
    with wave.open(io.BytesIO(data)) as w:
        return w.getnframes() / w.getframerate()


def concat_wavs(parts: list[bytes]) -> bytes:
    import io
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        for i, raw in enumerate(parts):
            with wave.open(io.BytesIO(raw)) as r:
                if i == 0:
                    w.setparams(r.getparams())
                w.writeframes(r.readframes(r.getnframes()))
    return out.getvalue()


def med(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 3) if xs else None


# ---------------------------------------------------------------- workloads
def w_llm(model: str, n: int, ctx: dict) -> dict:
    chat_stream(model, "hi", 8)  # warm-up, not timed
    out: dict = {}
    for lang, p in LLM_PROMPTS.items():
        rs = [chat_stream(model, p, 128) for _ in range(n)]
        out[f"{lang}_ttft_s"] = med([r["ttft_s"] for r in rs])
        out[f"{lang}_decode_tok_s"] = med([r["decode_tok_s"] for r in rs])
    rs = [chat_stream(model, LLM_LONG, 8) for _ in range(max(1, n - 1))]
    pt = rs[0]["prompt_tokens"] or 0
    out["long_prompt_tokens"] = pt
    out["long_prompt_ttft_s"] = med([r["ttft_s"] for r in rs])
    out["prefill_tok_s"] = round(pt / out["long_prompt_ttft_s"], 1) if out["long_prompt_ttft_s"] else None
    for k in (2, 4):
        res: list = [None] * k

        def go(i: int) -> None:
            res[i] = chat_stream(model, LLM_PROMPTS["pl"], 96)

        t0 = time.time()
        ths = [threading.Thread(target=go, args=(i,)) for i in range(k)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        out[f"agg_tok_s_{k}_streams"] = round(sum(r["completion_tokens"] for r in res) / (time.time() - t0), 1)
    return out


def w_embed(model: str, n: int, ctx: dict) -> dict:
    one = {"model": model, "input": "passage: " + PL_TEXT}
    batch = {"model": model, "input": ["passage: " + PL_TEXT + f" ({i})" for i in range(16)]}
    post_json("/v1/embeddings", one)
    t1 = [post_json("/v1/embeddings", one)[1] for _ in range(n)]
    t16 = [post_json("/v1/embeddings", batch)[1] for _ in range(n)]
    return {"latency_1_input_s": med(t1), "latency_16_inputs_s": med(t16),
            "inputs_per_s_batch16": round(16 / med(t16), 1) if med(t16) else None}


def w_rerank(model: str, n: int, ctx: dict) -> dict:
    body = {"model": model, "query": RERANK_QUERY, "documents": RERANK_DOCS}
    post_json("/v1/rerank", body)
    t = [post_json("/v1/rerank", body)[1] for _ in range(n)]
    return {"docs": len(RERANK_DOCS), "latency_s": med(t), "docs_per_s": round(len(RERANK_DOCS) / med(t), 1) if med(t) else None}


def w_tts(model: str, n: int, ctx: dict) -> dict:
    lang = ctx["entry"].get("lang", "en")
    text = TTS_TEXT[lang]  # ~200 chars: stays under every TTS engine's input limit
    body = {"model": model, "input": text, "language": lang}
    post_json("/v1/audio/speech", body, timeout=180)  # a TTS request that takes minutes is a finding, not something to wait out
    rs = []
    for _ in range(n):
        data, dt = post_json("/v1/audio/speech", body, timeout=180)
        rs.append((dt, wav_seconds(data) if isinstance(data, bytes) else 0.0))
    if ctx["entry"].get("make_audio"):  # STT input: several syntheses joined into one ~30 s clip
        parts = [post_json("/v1/audio/speech", body, timeout=180)[0] for _ in range(3)]
        ctx["audio"] = concat_wavs([p for p in parts if isinstance(p, bytes)])
    secs = med([a for _, a in rs])
    return {"language": lang, "chars": len(text), "audio_s": secs, "synth_s": med([d for d, _ in rs]),
            "rtf": med([d / a for d, a in rs if a])}


def make_png(size: int = 448) -> bytes:
    """A blue square with a red disc, built with the stdlib only - the same image on every machine."""
    import struct
    import zlib
    rows = []
    c, r2 = size // 2, (size // 4) ** 2
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            row += bytes((220, 30, 30) if (x - c) ** 2 + (y - c) ** 2 <= r2 else (20, 40, 200))
        rows.append(bytes(row))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


def vlm_stream(model: str, image_b64: str, max_tokens: int) -> dict:
    import base64  # noqa: F401 - image already encoded by the caller
    prompt = f"[{uuid.uuid4().hex[:10]}] Describe this image in one sentence: shapes and colours."
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0, "stream": True, "enable_thinking": False,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                      {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}}]}]}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t, first, usage, text = time.time(), None, None, ""
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            j = json.loads(line[5:])
            usage = j.get("usage") or usage
            for c in j.get("choices", []):
                piece = c.get("delta", {}).get("content") or ""
                if piece and first is None:
                    first = time.time() - t
                text += piece
    total = time.time() - t
    ct = (usage or {}).get("completion_tokens", 0)
    return {"ttft_s": first, "decode_tok_s": ct / (total - first) if first and ct and total > first else None,
            "prompt_tokens": (usage or {}).get("prompt_tokens"), "text": text}


def w_vlm(model: str, n: int, ctx: dict) -> dict:
    import base64
    img = base64.b64encode(make_png()).decode()
    vlm_stream(model, img, 8)  # warm-up
    rs = [vlm_stream(model, img, 64) for _ in range(n)]
    txt = rs[0]["text"].lower()
    t = [chat_stream(model, LLM_PROMPTS["en"], 96) for _ in range(n)]
    return {"image_ttft_s": med([r["ttft_s"] for r in rs]), "image_decode_tok_s": med([r["decode_tok_s"] for r in rs]),
            "image_prompt_tokens": rs[0]["prompt_tokens"], "text_decode_tok_s": med([r["decode_tok_s"] for r in t]),
            "saw_red_and_blue": ("red" in txt and "blue" in txt), "image_answer": rs[0]["text"].strip()[:120]}


def w_stt(model: str, n: int, ctx: dict) -> dict:
    wav = ctx.get("audio")
    if not wav:
        return {"error": "no audio: put a make_audio TTS entry before the STT entries, or pass --audio"}
    dur = wav_seconds(wav)
    post_audio(model, wav)
    ts = [post_audio(model, wav) for _ in range(n)]
    return {"audio_s": round(dur, 1), "proc_s": med(ts), "rtf": med([t / dur for t in ts]),
            "audio_source": ctx.get("audio_source", "synthetic (TTS)")}


WORKLOADS = {"llm": w_llm, "vlm": w_vlm, "embed": w_embed, "rerank": w_rerank, "tts": w_tts, "stt": w_stt}


def device_kinds(launcher: str) -> dict:
    """{OpenVINO device name: kind} from the server's own `--device-info` (kind: CPU, integrated-gpu, discrete-gpu, NPU)."""
    try:
        out = subprocess.run([launcher, "--device-info"], capture_output=True, text=True, timeout=90,
                             env={k: v for k, v in os.environ.items() if k != "RUSTEDVINO_CONFIG"}).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    kinds: dict[str, str] = {}
    cur = None
    for line in out.splitlines():
        m = re.match(r"^\s{2}(\S+) \u2014 ", line)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"^\s+kind\s*:\s*(\S+)", line)
        if m and cur:
            kinds[cur] = m.group(1)
    return kinds


def device_memory_gb(launcher: str, device: str) -> float | None:
    """Total memory of an OpenVINO device in GB, from `--device-info` ('total memory: 22.71 GiB'); None when it reports n/a
    (an iGPU or the CPU shares system RAM)."""
    try:
        out = subprocess.run([launcher, "--device-info"], capture_output=True, text=True, timeout=90,
                             env={k: v for k, v in os.environ.items() if k != "RUSTEDVINO_CONFIG"}).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    cur = None
    for line in out.splitlines():
        m = re.match(r"^\s{2}(\S+) \u2014 ", line)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"^\s+total memory\s*:\s*([\d.]+) GiB", line)
        if m and cur == device:
            return float(m.group(1)) * 1.073741824
    return None


def model_footprint_gb(model_dir: Path) -> float:
    """What `rv add` estimates for the model: every file under the directory, no margin (tools/rv lifecycle.rs, D-03)."""
    return sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / 1e9


def resolve_device(launcher: str, device: str) -> tuple:
    """(device to use, note or None). RustedVINO 0.8.1 rejects the bare name 'GPU' when the host has several GPUs
    (enumerated as GPU.0, GPU.1, ...): 'device GPU is not an available OpenVINO device'. Map it to the first one, loudly."""
    kinds = device_kinds(launcher)
    if device == "GPU" and "GPU" not in kinds:
        first = next((n for n in kinds if n.startswith("GPU.")), None)
        if first:
            return first, (f"this host lists {', '.join(n for n in kinds if n.startswith('GPU'))} and 0.8.1 rejects the bare name "
                           f"'GPU'; benchmarking {first} ({kinds[first]}). Name the device explicitly (--device GPU.n) to choose.")
    return device, None


def device_class(launcher: str, device: str) -> str | None:
    """'cpu' | 'igpu' | 'dgpu' | 'npu' for an OpenVINO device name. The bare name 'GPU' means the first GPU listed."""
    kinds = device_kinds(launcher)
    want = next((n for n in kinds if n.startswith("GPU")), device) if device == "GPU" else device
    return {"CPU": "cpu", "integrated-gpu": "igpu", "discrete-gpu": "dgpu", "NPU": "npu"}.get(kinds.get(want, ""))


TTS_GPU_WARNING = (
    "TTS is NOT benchmarked on a GPU: SpeechT5 misbehaves on every GPU class it was tried on - 3x slower than the CPU on an "
    "Arc 140V iGPU, RTF 16-100 on an Acer UHD iGPU, and a ~10 minute stall ending in HTTP 503 on an Arc Pro B50. It is "
    "measured on the CPU instead, and the row is marked 'ran on CPU'. Pass --tts-on-gpu to measure it on the GPU anyway "
    "(requests then time out after 3 minutes)."
)


# ---------------------------------------------------------------- server lifecycle
def run_entry(entry: dict, args: argparse.Namespace, ctx: dict, prefix_caching: bool = False) -> dict:
    model, kind = entry["id"], entry["kind"]
    res: dict = {"id": model, "kind": kind}
    if not (Path(args.models_dir) / entry.get("dir", model)).is_dir():
        return {**res, "skipped": f"not in models dir: {entry.get('dir', model)}"}
    # F8: this harness starts its own throwaway server with no memory gate, but the shipped server refuses a model that can
    # never fit a discrete GPU ("can never fit"). Say so instead of printing a number the product would not deliver.
    device_gb = device_memory_gb(args.launcher, args.device)
    need_gb = model_footprint_gb(Path(args.models_dir) / entry.get("dir", model)) if device_gb else 0.0
    if device_gb and need_gb > device_gb and not args.no_fit_gate:
        return {**res, "skipped": f"would be refused by the shipped server: needs about {need_gb:.1f} GB, {args.device} has "
                                  f"{device_gb:.1f} GB (pass --no-fit-gate to measure it anyway)"}
    work = Path(tempfile.mkdtemp(prefix="rvbench-"))
    launcher_dir = Path(args.launcher).resolve().parent
    base = next((p for p in (launcher_dir / "config.example.json", launcher_dir.parent / "scripts" / "dist.example.config.json",
                             Path(__file__).resolve().parents[2] / "scripts" / "dist.example.config.json") if p.is_file()), None)
    if base is None:
        return {**res, "error": "no config.example.json next to the launcher (use a release bundle)"}
    cfg = json.loads(base.read_text(encoding="utf-8-sig"))  # the shipped defaults, unchanged except below
    cfg.update({"models_dir": str(Path(args.models_dir).expanduser()), "device": args.device, "port": PORT,
                "bind_addr": "127.0.0.1", "preload": [model],
                "models": {model: {"kind": CFG_KIND[kind], "device": args.device, "vram_gb": entry.get("vram_gb", 1.0)}}})
    if prefix_caching:  # D-03 retry: a hybrid linear-attention model on a GPU only loads with this on
        cfg["models"][model]["enable_prefix_caching"] = True
    (work / "config.json").write_text(json.dumps(cfg))
    (work / "config.keys.json").write_text('{"api_keys": [], "admin_api_keys": []}')
    log = open(work / "server.log", "w")
    env = {**os.environ, "RUSTEDVINO_CONFIG": str(work / "config.json")}
    t0 = time.time()
    proc = subprocess.Popen([args.launcher], env=env, stdout=log, stderr=subprocess.STDOUT, cwd=Path(args.launcher).resolve().parent)
    sampler = RssSampler(proc.pid)
    sampler.start()
    listening_t = None
    try:
        while True:  # wait until the model is listed (= loaded) or the server dies
            if proc.poll() is not None:
                return {**res, "error": "server exited: " + "".join(open(work / 'server.log').readlines()[-3:])[:300]}
            try:
                with urllib.request.urlopen(BASE + "/v1/models", timeout=2) as r:
                    if model in [m["id"] for m in json.load(r)["data"]]:
                        break
            except (urllib.error.URLError, OSError, ValueError):
                pass
            # Preload runs before the port opens ("listening" is logged after it). A model still not listed a
            # few seconds after that has failed to load - the server stays up without it - so stop waiting.
            if listening_t is None and "RustedVINO listening" in open(work / "server.log").read():
                listening_t = time.time()
            if listening_t is not None and time.time() - listening_t > 15:
                errs = [l.strip() for l in open(work / "server.log").read().splitlines()
                        if re.search(r"\b(ERROR|WARN)\b|failed|Hint", l)]
                return {**res, "error": "model failed to load (server is up without it): " + " | ".join(errs[-3:])[:600]}
            if time.time() - t0 > args.load_timeout:
                tail = "".join(open(work / "server.log").readlines()[-3:])
                return {**res, "error": f"model not ready after {args.load_timeout}s: {tail[:300]}"}
            time.sleep(0.5)
        res["load_s"] = round(time.time() - t0, 1)
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
                h = json.load(r)
            ctx["server_version"] = f"{h.get('version', '?')} ({h.get('git_hash', '?')[:7]})"
        except (urllib.error.URLError, OSError, ValueError):
            pass
        m = re.search(r"load_duration_secs=([\d.]+)", open(work / "server.log").read())
        res["load_s_engine"] = round(float(m.group(1)), 1) if m else None
        res["rss_after_load_mb"] = round(tree_rss_mb(proc.pid) or 0) or None
        sampler.peak, sampler.mhz = 0.0, []
        ctx["entry"] = entry
        res.update(WORKLOADS[kind](model, args.repeat, ctx))
        res["rss_peak_mb"] = round(max(sampler.peak, tree_rss_mb(proc.pid) or 0)) or None
        res["cpu_mhz_avg"] = round(st.mean(sampler.mhz)) if sampler.mhz else None
    except Exception as e:  # noqa: BLE001 - a failed model must not abort the whole matrix run
        res["error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        sampler.stop_evt.set()
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    return res


# The shipped server refuses to load a hybrid linear-attention model (Qwen3.5 / 3.6) on a GPU while its prefix-cache guard
# forces caching off: "reserved_la_bytes <= budget_in_bytes". It loads with enable_prefix_caching on (README: Vision on a GPU).
HYBRID_LOAD_FAILURE = re.compile(r"reserved_la_bytes|linear.attention|hybrid", re.I)


def retry_hybrid_on_gpu(r: dict, entry: dict, args: argparse.Namespace, ctx: dict) -> dict:
    """D-03: one retry with enable_prefix_caching on when a model failed to load on a GPU with the hybrid signature.
    The row says so; the first failure is kept. Any other failure is reported unchanged."""
    if "error" not in r or not str(args.device).startswith("GPU") or not HYBRID_LOAD_FAILURE.search(r["error"]):
        return r
    print("   hybrid linear-attention model failed to load on a GPU: retrying once with enable_prefix_caching on", flush=True)
    r2 = run_entry(entry, args, ctx, prefix_caching=True)
    r2["note"] = ("loaded only with enable_prefix_caching=true (hybrid linear-attention model on a GPU); "
                  "the default server config refuses it there")
    r2["first_attempt_error"] = r["error"][:200]
    return r2 if "error" not in r2 else {**r, "retry_error": r2["error"][:200]}


def to_markdown(machine: dict, results: list[dict]) -> str:
    L = [f"## {machine['hostname']} - {machine['cpu']} ({machine['logical_cpus']} threads, {machine['ram_total_gb']} GB RAM, "
         f"device {machine['device']})",
         f"{machine['os']}, governor `{machine['cpu_governor'] or 'n/a'}`, on AC: {machine['on_ac_power']}, "
         f"load at start {machine['loadavg']}, RustedVINO {machine.get('server_version', '?')}", ""]
    by = {k: [r for r in results if r["kind"] == k] for k in WORKLOADS}

    def row(r: dict, cols: list[str]) -> str:
        name = r["id"] + (f" (ran on {r['ran_on']})" if r.get("ran_on") else "") + (" (prefix caching forced on)" if r.get("first_attempt_error") else "")
        if r.get("skipped") or r.get("error"):
            return f"| {name} | " + (r.get("skipped") or "ERROR " + r["error"]).replace("|", "/") + " |" + " |" * (len(cols) - 1)
        return f"| {name} | " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |"

    for kind, cols in (("llm", ["load_s", "cpu_mhz_avg", "rss_after_load_mb", "rss_peak_mb", "en_ttft_s", "en_decode_tok_s", "pl_decode_tok_s",
                                "prefill_tok_s", "agg_tok_s_2_streams", "agg_tok_s_4_streams"]),
                       ("vlm", ["load_s", "cpu_mhz_avg", "rss_after_load_mb", "rss_peak_mb", "image_prompt_tokens", "image_ttft_s",
                                "image_decode_tok_s", "text_decode_tok_s", "saw_red_and_blue"]),
                       ("embed", ["load_s", "cpu_mhz_avg", "rss_after_load_mb", "rss_peak_mb", "latency_1_input_s", "latency_16_inputs_s", "inputs_per_s_batch16"]),
                       ("rerank", ["load_s", "cpu_mhz_avg", "rss_after_load_mb", "rss_peak_mb", "docs", "latency_s", "docs_per_s"]),
                       ("tts", ["load_s", "cpu_mhz_avg", "rss_after_load_mb", "rss_peak_mb", "language", "audio_s", "synth_s", "rtf"]),
                       ("stt", ["load_s", "cpu_mhz_avg", "rss_after_load_mb", "rss_peak_mb", "audio_s", "proc_s", "rtf"])):
        if by[kind]:
            L += [f"### {kind}", "| model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"] + [row(r, cols) for r in by[kind]] + [""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--launcher", required=True, help="path to the rustedvino launcher (the bundle's ./rustedvino)")
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--out", default="bench-out")
    ap.add_argument("--suite", help="JSON list of {id, kind, [dir], [lang], [make_audio], [vram_gb]}; default = starter set")
    ap.add_argument("--only", action="append", default=[], help="run only this model id (repeatable)")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--no-fit-gate", action="store_true",
                    help="measure a model even when the shipped server would refuse it as larger than the device's memory")
    ap.add_argument("--audio", help="real speech WAV for STT instead of synthesised audio")
    ap.add_argument("--load-timeout", type=int, default=600)
    ap.add_argument("--allow-busy", action="store_true", help="run even if the machine is not quiet (result is flagged)")
    ap.add_argument("--tts-on-gpu", "--tts-on-igpu", dest="tts_on_gpu", action="store_true",
                    help="benchmark TTS on a GPU too (by default it is measured on the CPU: SpeechT5 stalls or crawls on every GPU tried)")
    args = ap.parse_args()

    args.device, dev_note = resolve_device(args.launcher, args.device)
    if dev_note:
        print("NOTE: " + dev_note, flush=True)
    machine = machine_info(args.device)
    if dev_note:
        machine["device_note"] = dev_note
    busy = machine["loadavg"] is not None and machine["loadavg"] > 0.35 * (machine["logical_cpus"] or 1)
    if busy and not args.allow_busy:
        print(f"machine not quiet (loadavg {machine['loadavg']:.1f} on {machine['logical_cpus']} threads); "
              "close other apps or pass --allow-busy", file=sys.stderr)
        return 2
    machine["quiet_at_start"] = not busy
    machine["device_class"] = device_class(args.launcher, args.device)
    if machine["device_class"] in ("igpu", "dgpu") and not args.tts_on_gpu:
        print("WARNING: " + TTS_GPU_WARNING, flush=True)
    suite = json.load(open(args.suite)) if args.suite else DEFAULT_SUITE
    suite = [e for e in suite if not args.only or e["id"] in args.only]
    ctx: dict = {}
    if args.audio:
        ctx["audio"], ctx["audio_source"] = Path(args.audio).read_bytes(), f"file {Path(args.audio).name}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for e in suite:
        print(f"[{time.strftime('%H:%M:%S')}] {e['id']} ({e['kind']}) ...", flush=True)
        if e["kind"] == "tts" and machine.get("device_class") in ("igpu", "dgpu") and not args.tts_on_gpu:
            print("   WARNING: " + TTS_GPU_WARNING, flush=True)
            cpu_args = argparse.Namespace(**{**vars(args), "device": "CPU"})
            r = run_entry(e, cpu_args, ctx)
            r.update({"ran_on": "CPU", "warning": TTS_GPU_WARNING})
            if e.get("make_audio"):
                ctx["audio_source"] = "synthetic (TTS, synthesised on the CPU)"
        else:
            r = run_entry(e, args, ctx)
            r = retry_hybrid_on_gpu(r, e, args, ctx)
        r["loadavg_end"] = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
        results.append(r)
        print("   ", {k: v for k, v in r.items() if k not in ("id", "kind")}, flush=True)
        machine["server_version"] = ctx.get("server_version", "?")
        (out / "results.json").write_text(json.dumps({"machine": machine, "results": results}, indent=1))
        (out / "results.md").write_text(to_markdown(machine, results))
    print(f"\nwritten: {out/'results.json'}  {out/'results.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
