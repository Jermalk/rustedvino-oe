#!/usr/bin/env python3
"""cmpbench - run rvbench's LLM workload against RustedVINO, OVMS or OpenArc on one machine.

One stack x one model per call. Lifecycle: cold load (empty compile cache) -> stop -> warm load -> equivalence probe ->
rvbench `w_llm` rows -> fixed-length 256-token decode row -> stop. RSS and VRAM (DRM fdinfo `drm-resident-vram*`) are
sampled over the server's process tree. Stdlib only; reuses rvbench's prompts and workload function unchanged.

    python3 cmpbench.py --config cmp.json --stack rv|ovms|openarc --model qwen3-1.7b-int4-ov --out DIR [--repeat 5]

Nothing about your machine is assumed: the model folder, cache folder, the GPU device name and the location of each stack come from a
config file (`--config`, see cmpbench.example.json) or `CMP_*` environment variables, and the tool stops with a list of what is
missing. Every stack's server, OpenVINO and OpenVINO GenAI versions are printed first and stored in results.json.
Read CMPBENCH.md (next to this file) before running: the method, the fairness rules and, above all, the limits of the numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import platform
import shutil
import signal
import socket
import statistics as st
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rvbench  # noqa: E402

class Cfg:
    """Run configuration, filled by load_config(). No defaults point at any particular machine."""
    models: Path        # one folder holding <model>/ for every model; all stacks read the same directories
    caches: Path        # scratch folder for each stack's compile cache (wiped before the cold load)
    gpu: str            # OpenVINO device name used by every stack, e.g. GPU.1 (see `rustedvino --device-info` or OpenVINO's available_devices)
    rv_dir: Path | None = None       # unpacked RustedVINO release bundle (contains `rustedvino` and `config.example.json`)
    ovms_dir: Path | None = None     # unpacked OVMS binary package (contains bin/ovms and lib/)
    ovms_docker: str = ""            # OR a docker image name; then ovms_render is required
    ovms_render: str = ""            # render node passed into the container, e.g. /dev/dri/renderD128
    ovms_gpu: str = ""               # device name inside the container (default: same as gpu)
    oa_dir: Path | None = None       # OpenArc checkout with its own .venv (uv sync)
    pdev: str = ""                   # PCI address of the GPU under test for the VRAM column; empty = sum over every DRM device the server touches
    rv_vram_gb: float = 1.0
    rv_nocache: bool = False


CFG = Cfg()
KEYS = {  # config key -> (environment variable, who needs it)
    "models": ("CMP_MODELS", "all"), "caches": ("CMP_CACHES", "all"), "gpu": ("CMP_GPU", "all"),
    "rv_dir": ("CMP_RV_DIR", "rv"), "ovms_dir": ("CMP_OVMS_DIR", "ovms (or ovms_docker)"),
    "ovms_docker": ("CMP_OVMS_DOCKER", "ovms (or ovms_dir)"), "ovms_render": ("CMP_OVMS_RENDER", "ovms with ovms_docker"),
    "ovms_gpu": ("CMP_OVMS_GPU", "optional"), "oa_dir": ("CMP_OA_DIR", "openarc"), "pdev": ("CMP_PDEV", "optional"),
    "rv_vram_gb": ("CMP_RV_VRAM", "optional"), "rv_nocache": ("CMP_RV_NOCACHE", "optional"),
}


def load_config(path: str | None, stack: str) -> list[str]:
    """Fill CFG from the JSON file (if any) and the CMP_* environment (which wins). Returns a list of problems; empty = ready."""
    raw: dict = {}
    problems: list[str] = []
    if path:
        try:
            raw = json.loads(Path(path).expanduser().read_text())
        except (OSError, ValueError) as e:
            return [f"cannot read config {path}: {e}"]
        if not isinstance(raw, dict):
            return [f"config {path} must be a JSON object ({{...}}), not {type(raw).__name__}"]
        problems += [f"unknown config key '{k}' (known: {', '.join(KEYS)})" for k in raw if k not in KEYS]
    for key, (env, _) in KEYS.items():
        if os.environ.get(env):
            raw[key] = os.environ[env]
    for key in ("models", "caches", "rv_dir", "ovms_dir", "oa_dir"):
        if raw.get(key):
            setattr(CFG, key, Path(str(raw[key])).expanduser().resolve())  # servers run with other working directories
    for key in ("gpu", "ovms_docker", "ovms_render", "ovms_gpu", "pdev"):
        if raw.get(key):
            setattr(CFG, key, str(raw[key]))
    try:
        CFG.rv_vram_gb = float(raw.get("rv_vram_gb") or 1.0)
    except (TypeError, ValueError):
        problems.append(f"rv_vram_gb must be a number, got {raw.get('rv_vram_gb')!r}")
    CFG.rv_nocache = str(raw.get("rv_nocache", "")).lower() in ("1", "true", "yes")
    CFG.ovms_gpu = CFG.ovms_gpu or ("GPU.0" if CFG.ovms_docker else getattr(CFG, "gpu", ""))  # one mapped render node = GPU.0 inside
    need = [k for k in ("models", "caches", "gpu") if not getattr(CFG, k, None)]
    if stack == "rv" and not CFG.rv_dir:
        need.append("rv_dir")
    if stack == "ovms" and not (CFG.ovms_dir or CFG.ovms_docker):
        need.append("ovms_dir (or ovms_docker)")
    if stack == "ovms" and CFG.ovms_docker and not CFG.ovms_render:
        need.append("ovms_render (required with ovms_docker)")
    if stack == "openarc" and not CFG.oa_dir:
        need.append("oa_dir")
    problems += [f"missing '{k.split(' ')[0]}' (config key, or environment variable {KEYS[k.split(' ')[0]][0]})" for k in need]
    if stack == "ovms" and not (CFG.ovms_dir or CFG.ovms_docker):
        problems.append("OVMS needs either ovms_dir (bare-metal package) or ovms_docker + ovms_render (docker image)")
    if stack == "ovms" and CFG.ovms_dir and CFG.ovms_docker:
        problems.append("both ovms_dir and ovms_docker are set: choose one")
    for key, label in (("rv_dir", "rv"), ("ovms_dir", "ovms"), ("oa_dir", "openarc")):
        d = getattr(CFG, key, None)
        if d and stack == label and not d.is_dir():
            problems.append(f"{key} {d} is not a directory")
    exes = {"rv": [CFG.rv_dir / "rustedvino", CFG.rv_dir / "config.example.json"] if CFG.rv_dir else [],
            "ovms": [CFG.ovms_dir / "bin" / "ovms"] if CFG.ovms_dir else [],
            "openarc": [CFG.oa_dir / ".venv" / "bin" / "openarc", CFG.oa_dir / ".venv" / "bin" / "python"] if CFG.oa_dir else []}
    problems += [f"{f} not found" for f in exes[stack] if not f.exists()]
    if stack == "ovms" and CFG.ovms_docker and not shutil.which("docker"):
        problems.append("ovms_docker is set but `docker` is not on PATH")
    if getattr(CFG, "caches", None):
        try:
            CFG.caches.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            problems.append(f"cannot create caches folder {CFG.caches}: {e}")
    if getattr(CFG, "models", None) and not CFG.models.is_dir():
        problems.append(f"models folder {CFG.models} does not exist")
    return problems


FIXED_PROMPT = "Write a long, detailed story about a lighthouse keeper and the storm that changed her life."
EQUIV_PROMPT = "Name the three primary colours, one per line, nothing else."
THINK_SEEN = 0  # responses that carried thinking (<think> text or reasoning_content): must stay 0
CHUNK_FALLBACK = 0  # streamed responses without a usable usage block (token counts were estimated from chunks)


# ---------------------------------------------------------------- process tree, RSS, VRAM
def tree_pids(pid: int) -> list[int]:
    kids: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                m = re.search(r"^PPid:\s*(\d+)", Path(f"/proc/{d}/status").read_text(), re.M)
            except OSError:
                continue
            if m:
                kids.setdefault(int(m.group(1)), []).append(int(d))
    out, stack = [], [pid]
    while stack:
        p = stack.pop()
        out.append(p)
        stack += kids.get(p, [])
    return out


_UNIT = {"": 1, "KiB": 1024, "MiB": 1 << 20, "GiB": 1 << 30}
UNREADABLE = 0  # processes whose fdinfo could not be read in the last vram_gib() call (other users', usually)


def vram_gib(pids: list[int], pdev: str | None = None) -> float:
    """VRAM resident for the given PIDs: one reading per drm-client-id (duplicated fds share a client).
    With a PCI address (config `pdev`) only that device counts; without one every DRM device the processes touch is summed
    (right when the server tree uses just the GPU under test; set `pdev` if other work shares the GPU)."""
    pdev = CFG.pdev if pdev is None else pdev
    global UNREADABLE
    per_client: dict[tuple[str, str], int] = {}
    UNREADABLE = 0
    for pid in pids:
        try:
            fds = os.listdir(f"/proc/{pid}/fdinfo")
        except OSError:
            UNREADABLE += 1
            continue
        for fd in fds:
            try:
                txt = Path(f"/proc/{pid}/fdinfo/{fd}").read_text()
            except OSError:
                continue
            if "drm-pdev:" not in txt or (pdev and pdev not in txt):
                continue
            cid = re.search(r"^drm-client-id:\s*(\S+)", txt, re.M)
            dev = re.search(r"^drm-pdev:\s*(\S+)", txt, re.M)
            # the kernel prints the largest unit that divides the value evenly: no unit (bytes), KiB or MiB
            nbytes = sum(int(n) * _UNIT.get(u, 1) for n, u in re.findall(r"^drm-resident-vram\d*:\s*(\d+)(?:\s*(KiB|MiB|GiB))?\s*$", txt, re.M))
            key = (dev.group(1) if dev else "", cid.group(1) if cid else f"{pid}:{fd}")  # client ids are unique per system; forked fds share one
            per_client[key] = max(per_client.get(key, 0), nbytes)
    return sum(per_client.values()) / (1 << 30)


class Sampler(threading.Thread):
    def __init__(self, pid: int) -> None:
        super().__init__(daemon=True)
        self.pid, self.stop_evt = pid, threading.Event()
        self.rss_peak, self.vram_peak, self.rss_last, self.vram_last = 0.0, 0.0, 0.0, 0.0
        self.rss_peak_all, self.vram_peak_all = 0.0, 0.0  # since the sampler started (includes the load)

    def run(self) -> None:
        while not self.stop_evt.is_set():
            pids = tree_pids(self.pid)
            rss = rvbench.tree_rss_mb(self.pid) or 0.0
            vr = vram_gib(pids)
            self.rss_last, self.vram_last = rss, vr
            self.rss_peak, self.vram_peak = max(self.rss_peak, rss), max(self.vram_peak, vr)
            self.rss_peak_all, self.vram_peak_all = max(self.rss_peak_all, rss), max(self.vram_peak_all, vr)
            self.stop_evt.wait(0.5)

    def reset_peak(self) -> None:
        self.rss_peak, self.vram_peak = self.rss_last, self.vram_last


# ---------------------------------------------------------------- HTTP with per-stack shaping
class Wire:
    base = "http://127.0.0.1:11598"
    prefix = "/v1"
    thinking = "top"  # top = RustedVINO's enable_thinking field; kwargs = chat_template_kwargs


def _chat_body(model: str, prompt: str, max_tokens: int, stream: bool, tag: bool = True) -> dict:
    if tag:  # unique tag defeats the prefix cache (same as rvbench.chat_stream)
        prompt = f"[{uuid.uuid4().hex[:10]}] {prompt}"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0}
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    if Wire.thinking == "top":
        body["enable_thinking"] = False
    else:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def chat_stream(model: str, prompt: str, max_tokens: int) -> dict:
    """Same contract as rvbench.chat_stream; also returns chunk timing so decode can be computed two ways."""
    global THINK_SEEN, CHUNK_FALLBACK
    body = _chat_body(model, prompt, max_tokens, True)
    req = urllib.request.Request(Wire.base + Wire.prefix + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t, first, last, usage, chunks, text, reasoning = time.time(), None, None, None, 0, [], 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            j = json.loads(line[5:])
            usage = j.get("usage") or usage
            for c in j.get("choices", []):
                d = c.get("delta", {})
                piece = d.get("content") or d.get("reasoning_content")
                reasoning += bool(d.get("reasoning_content"))
                if piece:
                    now = time.time()
                    if first is None:
                        first = now - t
                    last = now - t
                    chunks += 1
                    text.append(piece)
    total = time.time() - t
    if "<think>" in "".join(text) or reasoning:
        THINK_SEEN += 1
    if not (usage and usage.get("completion_tokens")):
        CHUNK_FALLBACK += 1
    ct = (usage or {}).get("completion_tokens") or chunks
    dec = total - (first or 0)
    return {"ttft_s": first, "total_s": total, "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": ct, "decode_tok_s": ct / dec if dec > 0 and ct else None,
            "tok_source": "usage" if usage and usage.get("completion_tokens") else "chunks",
            "chunks": chunks, "t_first": first, "t_last": last}


def http_json(path: str, timeout: float = 5.0):
    with urllib.request.urlopen(Wire.base + path, timeout=timeout) as r:
        return json.load(r)


# ---------------------------------------------------------------- launchers
class Stack:
    name = ""
    log_path: Path
    proc: subprocess.Popen | None = None

    def __init__(self, model: str, out: Path) -> None:
        self.model, self.out = model, out
        self.mdir = CFG.models / model
        self.cache = CFG.caches / f"{self.name}-{model}"
        self.log_path = out / f"{self.name}-server.log"

    def wipe_cache(self) -> None:
        shutil.rmtree(self.cache, ignore_errors=True)
        if self.name == "openarc" and (self.mdir / "model_cache").exists():  # OpenArc's own cache; the other stacks never create it
            print(f"  removing {self.mdir / 'model_cache'} (OpenArc compile cache) for the cold load")
            shutil.rmtree(self.mdir / "model_cache", ignore_errors=True)

    def spawn(self, cmd: list[str], env: dict | None = None, cwd: Path | None = None) -> None:
        self.log = open(self.log_path, "a")
        self.proc = subprocess.Popen(cmd, env={**os.environ, **(env or {})}, stdout=self.log, stderr=subprocess.STDOUT,
                                     cwd=cwd, start_new_session=True)

    def listed(self) -> bool:
        try:
            ids = [m["id"] for m in http_json(Wire.prefix + "/models")["data"]]
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            return False
        return self.model_id in ids

    model_id = ""

    def start(self, timeout: float = 900.0) -> float:
        raise NotImplementedError

    def wait_listed(self, t0: float, timeout: float) -> float:
        while True:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(f"{self.name} exited rc={self.proc.returncode}: {self.tail()}")
            if self.listed():
                return round(time.time() - t0, 1)
            if time.time() - t0 > timeout:
                raise RuntimeError(f"{self.name} model not listed after {timeout}s: {self.tail()}")
            time.sleep(0.05)  # load times are quantised to this poll interval

    def tail(self) -> str:
        try:
            return " | ".join(self.log_path.read_text().splitlines()[-3:])[:400]
        except OSError:
            return ""

    def versions(self) -> dict:
        """Server, OpenVINO and OpenVINO GenAI versions this stack will run with (readable before the server starts)."""
        return {}

    def _group_alive(self, pgid: int) -> bool:
        try:
            os.killpg(pgid, 0)
            return True
        except OSError:
            return False

    # How long stop() waits after SIGTERM, and whether it SIGKILLs what is left after that.
    stop_wait_s = 60
    sigkill_after_wait = True

    def stop(self) -> dict:
        """SIGTERM the whole process group, wait up to `stop_wait_s` for the leader and every member; then SIGKILL
        what is left, unless the stack must never be killed (`sigkill_after_wait = False`: report it instead)."""
        if not self.proc:
            return {}
        t = time.time()
        pgid = self.proc.pid  # start_new_session=True: the leader's pid is the group id
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            pass
        try:
            rc = self.proc.wait(timeout=self.stop_wait_s)
        except subprocess.TimeoutExpired:
            rc = "killed"
        left = False
        while self._group_alive(pgid) and time.time() - t < self.stop_wait_s:
            left = True
            time.sleep(0.2)
        if not self.sigkill_after_wait and (rc == "killed" or self._group_alive(pgid)):
            print(f"WARNING: {self.name} still running {self.stop_wait_s} s after SIGTERM - NOT killed (a kill during a "
                  f"GPU load can wedge the driver); stop it by hand once it exits", file=sys.stderr)
            self.log.close()
            self.proc = None
            return {"exit_code": "still-running", "exit_s": round(time.time() - t, 2), "group_lingered": True}
        if rc == "killed" or self._group_alive(pgid):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                pass
            rc = "killed" if rc == "killed" else f"{rc}+group-killed"
            self.proc.wait()
        self.log.close()
        self.proc = None
        return {"exit_code": rc, "exit_s": round(time.time() - t, 2), "group_lingered": left}


class RV(Stack):
    name = "rv"
    # The server's stop waits for a running model load (up to 600 s) and must never be SIGKILLed inside it: a kill
    # in the middle of a GPU load wedges an Arc Pro B60 (0.9.0 review G). 720 s = the systemd TimeoutStopSec.
    stop_wait_s = 720
    sigkill_after_wait = False

    def __init__(self, model: str, out: Path) -> None:
        super().__init__(model, out)
        self.model_id = model
        Wire.base, Wire.prefix, Wire.thinking = "http://127.0.0.1:11598", "/v1", "top"

    def start(self, timeout: float = 900.0) -> float:
        base = json.loads((CFG.rv_dir / "config.example.json").read_text(encoding="utf-8-sig"))  # shipped defaults
        self.cache.mkdir(parents=True, exist_ok=True)
        base.update({"models_dir": str(CFG.models), "device": CFG.gpu, "port": 11598, "bind_addr": "127.0.0.1",
                     "preload": [self.model], "cache_size_gb": 2.0, "kv_cache_precision": "u8", "max_num_seqs": 16,
                     "enable_prefix_caching": True,
                     "models": {self.model: {"kind": "text_gen", "device": CFG.gpu,
                                             "vram_gb": CFG.rv_vram_gb}}})
        if CFG.rv_nocache:
            base.pop("ov_cache_dir", None)
        else:
            base["ov_cache_dir"] = str(self.cache)
        work = self.out / "rv-work"
        work.mkdir(exist_ok=True)
        (work / "config.json").write_text(json.dumps(base, indent=1))
        (work / "config.keys.json").write_text('{"api_keys": [], "admin_api_keys": []}')
        t0 = time.time()
        self.spawn([str(CFG.rv_dir / "rustedvino")], env={"RUSTEDVINO_CONFIG": str(work / "config.json")}, cwd=CFG.rv_dir)
        return self.wait_listed(t0, timeout)

    def versions(self) -> dict:
        # Builds before 0.8.1 have no --version and would START the server on an unknown flag: run it with a throw-away
        # config path and a timeout, and accept only a version-looking answer.
        server = "unknown (build without --version)"
        try:
            r = subprocess.run([str(CFG.rv_dir / "rustedvino"), "--version"], capture_output=True, text=True, timeout=20,
                               env={**os.environ, "RUSTEDVINO_CONFIG": "/nonexistent/cmpbench.json"}, stdin=subprocess.DEVNULL)
            first = (r.stdout.strip().splitlines() or [""])[0]
            if re.match(r"^rustedvino \d+\.\d+\.\d+", first):
                server = first
        except (subprocess.TimeoutExpired, OSError):
            pass
        out = {"server": server}
        lib = CFG.rv_dir / "lib"
        for key, pat in (("openvino", "libopenvino.so.*"), ("genai", "libopenvino_genai.so.*")):
            found = "not found in the bundle's lib/"
            for f in sorted(lib.glob(pat)):
                if f.is_symlink() or not f.is_file():
                    continue
                m = re.search(rb"20\d\d\.\d+\.\d+(?:\.\d+)?-\d+-[0-9a-f]{6,}[A-Za-z0-9/._-]*", f.read_bytes())
                if m:
                    found = m.group(0).decode()
                    break
            out[key] = found
        return out

    def running(self) -> str:
        try:
            h = http_json("/health")
            return f"rustedvino {h.get('version')} ({str(h.get('git_hash'))[:12]})"
        except (urllib.error.URLError, OSError, ValueError):
            return "unknown"


class OVMS(Stack):
    name = "ovms"

    def __init__(self, model: str, out: Path) -> None:
        super().__init__(model, out)
        self.model_id = "cmp"
        self.cname = f"cmp-ovms-{os.getpid()}"
        Wire.base, Wire.prefix, Wire.thinking = "http://127.0.0.1:11436", "/v3", "kwargs"

    def start(self, timeout: float = 900.0) -> float:
        cfgd = self.out / "ovms-cfg"
        (cfgd / "graph").mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        (cfgd / "ovms_config.json").write_text(json.dumps(
            {"model_config_list": [], "mediapipe_config_list": [{"name": self.model_id, "base_path": str(cfgd / "graph")}]}))
        plugin = json.dumps({"KV_CACHE_PRECISION": "u8", "CACHE_DIR": str(self.cache)})
        (cfgd / "graph" / "graph.pbtxt").write_text(f'''input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"
node: {{
  name: "LLMExecutor"
  calculator: "HttpLLMCalculator"
  input_stream: "LOOPBACK:loopback"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "LLM_NODE_RESOURCES:llm"
  output_stream: "LOOPBACK:loopback"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  input_stream_info: {{ tag_index: 'LOOPBACK:0', back_edge: true }}
  node_options: {{
    [type.googleapis.com/mediapipe.LLMCalculatorOptions]: {{
      models_path: "{self.mdir}"
      max_num_seqs: 16
      device: "{CFG.ovms_gpu if CFG.ovms_docker else CFG.gpu}"
      cache_size: 2
      enable_prefix_caching: true
      plugin_config: '{plugin}'
    }}
  }}
  input_stream_handler {{
    input_stream_handler: "SyncSetInputStreamHandler",
    options {{ [mediapipe.SyncSetInputStreamHandlerOptions.ext] {{ sync_set {{ tag_index: "LOOPBACK:0" }} }} }}
  }}
}}
''')
        t0 = time.time()
        if CFG.ovms_docker:  # same absolute paths inside the container; host network; only the GPU's render node is mapped
            subprocess.run(["docker", "rm", "-f", self.cname], capture_output=True)  # only this run's own name
            gid = os.stat(CFG.ovms_render).st_gid
            self.spawn(["docker", "run", "--rm", "--name", self.cname, "--network", "host", "--device", CFG.ovms_render,
                        "--group-add", str(gid), "--user", f"{os.getuid()}:{os.getgid()}",
                        "-v", f"{CFG.models}:{CFG.models}:ro", "-v", f"{cfgd}:{cfgd}:ro", "-v", f"{CFG.caches}:{CFG.caches}",
                        CFG.ovms_docker, "--config_path", str(cfgd / "ovms_config.json"), "--rest_port", "11436", "--port", "11435",
                        "--rest_bind_address", "127.0.0.1", "--grpc_bind_address", "127.0.0.1"])
        else:
            self.spawn([str(CFG.ovms_dir / "bin" / "ovms"), "--config_path", str(cfgd / "ovms_config.json"), "--rest_port", "11436",
                        "--port", "11435", "--rest_bind_address", "127.0.0.1", "--grpc_bind_address", "127.0.0.1"], env={"LD_LIBRARY_PATH": str(CFG.ovms_dir / "lib")})
        return self.wait_listed(t0, timeout)

    def sample_pid(self) -> int:
        """Host PID of the server: the container's init for docker (the `docker run` client is not where the memory is)."""
        if CFG.ovms_docker:
            out = subprocess.run(["docker", "inspect", "-f", "{{.State.Pid}}", self.cname], capture_output=True, text=True).stdout.strip()
            if out.isdigit() and int(out) > 0:
                return int(out)
        self.sampled = "docker run client (docker inspect failed)" if CFG.ovms_docker else "server"
        return self.proc.pid

    def stop(self) -> dict:
        if CFG.ovms_docker and self.proc:  # killing the `docker run` client would leave the container holding the GPU and the ports
            subprocess.run(["docker", "stop", "-t", "60", self.cname], capture_output=True)
        return super().stop()

    def versions(self) -> dict:
        out = {"server": "unknown", "openvino": "unknown", "genai": "unknown"}
        try:
            if CFG.ovms_docker:  # may pull the image: allow time
                r = subprocess.run(["docker", "run", "--rm", "--entrypoint", "/ovms/bin/ovms", CFG.ovms_docker, "--version"],
                                   capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
            else:
                r = subprocess.run([str(CFG.ovms_dir / "bin" / "ovms"), "--version"], timeout=30, stdin=subprocess.DEVNULL,
                                   env={**os.environ, "LD_LIBRARY_PATH": str(CFG.ovms_dir / "lib")}, capture_output=True, text=True)
        except (subprocess.TimeoutExpired, OSError):
            return out
        for line in (r.stdout + "\n" + r.stderr).splitlines():
            low = line.lower()
            if "model server" in low:
                out["server"] = line.strip() + (f" (docker image {CFG.ovms_docker}, its own GPU user-mode driver)" if CFG.ovms_docker else "")
            elif "genai" in low:
                out["genai"] = line.split("backend", 1)[-1].strip()
            elif "openvino backend" in low:
                out["openvino"] = line.split("backend", 1)[-1].strip()
        return out


class OpenArc(Stack):
    name = "openarc"

    def __init__(self, model: str, out: Path) -> None:
        super().__init__(model, out)
        self.model_id = "cmp-" + model
        Wire.base, Wire.prefix, Wire.thinking = "http://127.0.0.1:8000", "/v1", "kwargs"
        self.bin = str(CFG.oa_dir / ".venv" / "bin" / "openarc")

    def _oa(self, *args: str, timeout: int = 900) -> subprocess.CompletedProcess:
        with open(self.out / "openarc-cli.log", "a") as lg:
            lg.write(f"$ openarc {' '.join(args)}\n")
            lg.flush()
            return subprocess.run([self.bin, *args], cwd=CFG.oa_dir, stdout=lg, stderr=subprocess.STDOUT, timeout=timeout)

    def start(self, timeout: float = 900.0) -> float:
        self._oa("list", "--remove", self.model_id)
        rc_add = self._oa("add", "--model-name", self.model_id, "--model-path", str(self.mdir), "--engine", "ovgenai",
                 "--model-type", "llm", "--device", CFG.gpu, "--cache-size", "2", "--max-num-seqs", "16",
                 "--enable-prefix-caching", "--runtime-config", '{"KV_CACHE_PRECISION": "u8"}',
                 "--cache-dir", str(self.cache)).returncode
        if rc_add != 0:
            raise RuntimeError(f"openarc add rc={rc_add}: see openarc-cli.log")
        t0 = time.time()
        self.spawn([self.bin, "serve", "start", "--host", "127.0.0.1", "--port", "8000"], cwd=CFG.oa_dir)
        while True:  # server up (models list may be empty before load)
            if self.proc.poll() is not None:
                raise RuntimeError(f"openarc serve exited rc={self.proc.returncode}: {self.tail()}")
            try:
                http_json("/v1/models")
                break
            except (urllib.error.URLError, OSError, ValueError):
                if time.time() - t0 > 120:
                    raise RuntimeError("openarc serve not answering: " + self.tail())
                time.sleep(0.5)
        t1 = time.time()
        rc = self._oa("load", self.model_id, timeout=int(timeout)).returncode
        self.load_cmd_s = round(time.time() - t1, 1)
        if rc != 0:
            raise RuntimeError(f"openarc load rc={rc}: see openarc-cli.log")
        return self.wait_listed(t0, 60)

    def stop(self) -> dict:
        res = super().stop()
        if getattr(self, "bin", None):  # leave no cmp-<model> entry in the user's OpenArc config
            try:
                self._oa("list", "--remove", self.model_id, timeout=60)
            except (subprocess.TimeoutExpired, OSError):
                pass
        return res

    def versions(self) -> dict:
        try:
            r = subprocess.run([str(CFG.oa_dir / ".venv" / "bin" / "python"), "-c",
                                "import openvino_genai,openvino;print(openvino.__version__);print(openvino_genai.__version__)"],
                               capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
            g = subprocess.run(["git", "-C", str(CFG.oa_dir), "log", "-1", "--format=%H"], capture_output=True, text=True, timeout=20)
            dirty = subprocess.run(["git", "-C", str(CFG.oa_dir), "status", "--porcelain"], capture_output=True, text=True, timeout=20).stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            return {"server": "openarc (unknown)", "openvino": "unknown", "genai": "unknown"}
        v = r.stdout.split()
        return {"server": f"openarc commit {g.stdout.strip() or 'unknown'}" + (" (working tree modified)" if dirty else ""), "openvino": v[0] if v else "unknown",
                "genai": v[1] if len(v) > 1 else "unknown"}


STACKS = {"rv": RV, "ovms": OVMS, "openarc": OpenArc}


# ---------------------------------------------------------------- measurements
def fixed256(model: str, n: int) -> dict:
    ok, rows = [], []
    for _ in range(n + 2):  # up to 2 spare attempts: samples with completion_tokens != 256 are dropped
        r = chat_stream(model, FIXED_PROMPT, 256)
        rows.append({k: r[k] for k in ("completion_tokens", "tok_source", "ttft_s", "total_s")})
        if r["completion_tokens"] == 256 and r["t_last"] and r["t_first"] and r["t_last"] > r["t_first"]:
            ok.append(r)
        if len(ok) >= n:
            break
    if not ok:
        return {"fixed256_samples": 0, "fixed256_attempts": rows}
    a = [r["decode_tok_s"] for r in ok]
    b = [255 / (r["t_last"] - r["t_first"]) for r in ok]
    return {"fixed256_samples": len(ok), "fixed256_decode_tok_s": rvbench.med(a),
            "fixed256_decode_tok_s_chunks": rvbench.med(b), "fixed256_ms_per_token": rvbench.med([1000 / x for x in b]),
            "fixed256_ttft_s": rvbench.med([r["ttft_s"] for r in ok]), "fixed256_tok_source": ok[0]["tok_source"],
            "fixed256_min_max_tok_s": [round(min(b), 1), round(max(b), 1)]}


def equiv(model: str) -> dict:
    body = _chat_body(model, EQUIV_PROMPT, 64, False, tag=False)
    req = urllib.request.Request(Wire.base + Wire.prefix + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        j = json.load(r)
    msg = j["choices"][0]["message"]
    if "<think>" in (msg.get("content") or "") or msg.get("reasoning_content"):
        global THINK_SEEN
        THINK_SEEN += 1
    return {"prompt_tokens": (j.get("usage") or {}).get("prompt_tokens"),
            "completion_tokens": (j.get("usage") or {}).get("completion_tokens"),
            "text": msg.get("content")}


def to_md(res: dict) -> str:
    skip = {"equivalence", "stop", "versions", "fixed256_attempts", "stop_after_cold"}
    v = res.get("versions") or {}
    rows = ["| key | value |", "|---|---|"]
    rows += [f"| version: {k} | {val} |" for k, val in v.items()]
    rows += [f"| {k} | {str(val).replace('|', '/')} |" for k, val in res.items() if k not in skip]
    return "\n".join(rows) + "\n"


def checks(res: dict, repeat: int, skip_cold: bool) -> dict:
    """Validity signals a reader must see next to the numbers; `failed` lists the ones that make the run unusable."""
    failed = []
    if res.get("think_text_seen"):
        failed.append(f"thinking appeared in {res['think_text_seen']} responses (switch not honoured)")
    if res.get("fixed256_samples", 0) < repeat:
        failed.append(f"fixed256: {res.get('fixed256_samples', 0)}/{repeat} samples with exactly 256 tokens")
    if res.get("responses_without_usage"):
        failed.append(f"{res['responses_without_usage']} responses had no usage block (token counts estimated from chunks)")
    if not skip_cold and res.get("cache_dir_mb_after_cold") == 0 and not res.get("rv_nocache"):
        failed.append("compile-cache folder empty after the cold load (cache setting may have been ignored)")
    if res.get("gpu_resident_gib_before", 0) > 0.2:
        failed.append(f"{res['gpu_resident_gib_before']} GiB of VRAM was resident before the run")
    return {"failed": failed, "think": res.get("think_text_seen"), "fixed256": f"{res.get('fixed256_samples', 0)}/{repeat}",
            "no_usage": res.get("responses_without_usage"), "gpu_before_gib": res.get("gpu_resident_gib_before")}


def print_versions(stack: str, v: dict) -> None:
    print("== versions (read before the server starts) ==")
    print(f"  stack   : {stack}")
    for k in ("server", "openvino", "genai"):
        print(f"  {k:<8}: {v.get(k, 'unknown')}")
    print(f"  device  : {CFG.gpu}   kernel: {platform.release()}   client python: {platform.python_version()}")
    print("  (the same three lines for every stack must be compared before reading any number: a different OpenVINO GenAI "
          "version means the gap is server + engine, not server)\n", flush=True)


def cpu_busy_pct(seconds: float = 5.0) -> float:
    """Share of all CPU time that was not idle or waiting for I/O over `seconds`, from /proc/stat (0-100).
    Unlike the load average it ignores threads blocked in uninterruptible I/O wait, which inflate the load on an idle machine."""
    def snap() -> tuple[int, int]:
        f = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
        return sum(f), f[3] + (f[4] if len(f) > 4 else 0)  # total, idle + iowait
    t0, i0 = snap()
    time.sleep(seconds)
    t1, i1 = snap()
    return 100.0 * (1 - (i1 - i0) / (t1 - t0)) if t1 > t0 else 0.0


def power_state() -> dict:
    """Power facts that change speed by tens of percent on a laptop: AC online?, EPP, platform profile (when readable)."""
    ac = [rvbench._read(str(f)).strip() for f in sorted(Path("/sys/class/power_supply").glob("A*/online"))]
    try:
        prof = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        prof = ""
    return {"ac_online": ac or None, "cpu_epp": rvbench._read("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference").strip() or None,
            "power_profile": prof or None}


def port_busy(port: int) -> bool:
    with socket.socket() as sk:
        sk.settimeout(0.5)
        return sk.connect_ex(("127.0.0.1", port)) == 0


PORTS = {"rv": (11598,), "ovms": (11436, 11435), "openarc": (8000,)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="JSON config (see cmpbench.example.json); CMP_* environment variables override it")
    ap.add_argument("--stack", required=True, choices=sorted(STACKS))
    ap.add_argument("--model", required=True, help="folder name under the models folder, e.g. qwen3-1.7b-int4-ov")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--no-shape-warmup", action="store_true", help="only rvbench's own 'hi' warm-up (skips the per-shape warm-up; under-measures prefill)")
    ap.add_argument("--max-cpu-pct", type=float, default=15.0,
                    help="refuse to start when the machine's CPU is busier than this over 5 s (default 15; the value used is recorded)")
    ap.add_argument("--skip-cold", action="store_true", help="skip the cold-load pass (reuse the existing compile cache)")
    a = ap.parse_args()
    cfg_path = a.config or ("cmp.json" if Path("cmp.json").is_file() else None)
    problems = load_config(cfg_path, a.stack)
    if problems:
        print("cmpbench: configuration incomplete:\n  - " + "\n  - ".join(problems) +
              "\nGive them in a JSON file (--config; see cmpbench.example.json) or as environment variables. Nothing was started.", file=sys.stderr)
        return 2
    if not re.fullmatch(r"[A-Za-z0-9._-]+", a.model) or a.model in (".", ".."):
        print("cmpbench: --model must be a folder name under the models folder (letters, digits, . _ -), not a path", file=sys.stderr)
        return 2
    if a.repeat < 1:
        print("cmpbench: --repeat must be at least 1", file=sys.stderr)
        return 2
    out = Path(a.out).expanduser().resolve() / a.stack / a.model
    if not (CFG.models / a.model).is_dir():
        print(f"cmpbench: model folder {CFG.models / a.model} does not exist", file=sys.stderr)
        return 2
    busy_pct = cpu_busy_pct()
    if busy_pct > a.max_cpu_pct:
        print(f"cmpbench: CPU is {busy_pct:.0f}% busy over 5 s (limit {a.max_cpu_pct:g}%): the machine is not quiet. Stop what is running "
              "(see `top`), or raise --max-cpu-pct knowingly; the value is recorded in the result.", file=sys.stderr)
        return 3
    busy = [p for p in PORTS[a.stack] if port_busy(p)]
    if busy:
        print(f"cmpbench: port(s) {busy} already answer on 127.0.0.1: another server is running there; stop it first", file=sys.stderr)
        return 3
    out.mkdir(parents=True, exist_ok=True)
    s = STACKS[a.stack](a.model, out)
    vers = s.versions()
    print_versions(a.stack, vers)
    idle = vram_gib([int(d) for d in os.listdir("/proc") if d.isdigit()])
    unreadable = UNREADABLE
    print(f"idle-GPU check: {idle:.2f} GiB resident in processes this user can inspect; {unreadable} other processes could not be "
          "inspected (run as root to see every user's GPU memory)\n", flush=True)
    if idle > 0.2:
        print(f"WARNING: {idle:.2f} GiB of VRAM is already resident before the run (another program uses a GPU): "
              "results are not comparable unless it is stopped\n", file=sys.stderr)
    res: dict = {"stack": a.stack, "model": a.model, "repeat": a.repeat, "started": time.strftime("%F %T"), "versions": vers,
                 "device": CFG.gpu, "vram_device_filter": CFG.pdev or "all DRM devices touched by the server",
                 "gpu_resident_gib_before": round(idle, 2), "cpu_busy_pct_before": round(busy_pct, 1), "max_cpu_pct": a.max_cpu_pct,
                 "loadavg_before": os.getloadavg()[0], "idle_check_uninspectable_procs": unreadable,
                 "rv_vram_gb": CFG.rv_vram_gb if a.stack == "rv" else None, "rv_nocache": CFG.rv_nocache if a.stack == "rv" else None, "kernel": platform.release(),
                 "governor": rvbench._read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").strip(),
                 **power_state()}
    rvbench.BASE = Wire.base
    rvbench.chat_stream = chat_stream  # w_llm resolves chat_stream as a module global
    sampler = None
    cache_mb = None
    try:
        if not a.skip_cold:
            s.wipe_cache()
            res["load_cold_s"] = s.start()
            cache_mb = round(sum(f.stat().st_size for f in s.cache.rglob("*") if f.is_file()) / 1e6) if s.cache.exists() else 0
            if hasattr(s, "load_cmd_s"):
                res["load_cold_cmd_s"] = s.load_cmd_s
            res["stop_after_cold"] = s.stop()
            time.sleep(3)
        if res.get("stop_after_cold", {}).get("exit_code") == "still-running":
            raise RuntimeError("the cold-load server did not exit within its stop wait and was left running on "
                               "purpose (see the WARNING above); stop it with `rv stop` and rerun")
        if any(port_busy(p) for p in PORTS[a.stack]):
            raise RuntimeError("a port is still answering after the cold-load server was stopped (orphan process?)")
        t_warm = s.start()
        res["load_warm_s"] = t_warm
        if hasattr(s, "load_cmd_s"):
            res["load_warm_cmd_s"] = s.load_cmd_s
        if hasattr(s, "running"):
            res["versions"]["running_server"] = s.running()
        sampler = Sampler(s.sample_pid() if hasattr(s, "sample_pid") else s.proc.pid)
        sampler.start()
        time.sleep(2)
        res["rss_after_load_mb"], res["vram_after_load_gib"] = round(sampler.rss_last), round(sampler.vram_last, 2)
        res["equivalence"] = equiv(s.model_id)
        if not a.no_shape_warmup:
            chat_stream(s.model_id, "hi", 8)  # shape warm-up (short)
            chat_stream(s.model_id, rvbench.LLM_LONG, 8)  # shape warm-up (long)
            for _ in range(2):
                chat_stream(s.model_id, FIXED_PROMPT, 256)  # shape warm-up (256 out)
            for k in (2, 4):  # one discarded parallel burst of each width, like the measured ones
                ths = [threading.Thread(target=chat_stream, args=(s.model_id, rvbench.LLM_PROMPTS["pl"], 96)) for _ in range(k)]
                [t.start() for t in ths]
                [t.join() for t in ths]
        res["shape_warmup"] = not a.no_shape_warmup
        sampler.reset_peak()
        res.update(rvbench.w_llm(s.model_id, a.repeat, {}))
        if not res.get("long_prompt_tokens"):  # no usage block: rvbench would report 0.0 tok/s; say "missing" instead
            res["prefill_tok_s"] = None
        res.update(fixed256(s.model_id, a.repeat))
        res["rss_peak_workload_mb"], res["vram_peak_workload_gib"] = round(sampler.rss_peak), round(sampler.vram_peak, 2)
        res["rss_peak_incl_warm_load_mb"], res["vram_peak_incl_warm_load_gib"] = round(sampler.rss_peak_all), round(sampler.vram_peak_all, 2)
        res["think_text_seen"] = THINK_SEEN
        res["responses_without_usage"] = CHUNK_FALLBACK
        res["sampled_process"] = getattr(s, "sampled", "server")
        res["cache_dir_mb_after_cold"] = cache_mb
        res["checks"] = checks(res, a.repeat, a.skip_cold)
    except Exception as e:  # noqa: BLE001 - record the failure, keep the evidence
        res["error"] = f"{type(e).__name__}: {e}"[:500]
    finally:
        if sampler:
            sampler.stop_evt.set()
        res["stop"] = s.stop()
    (out / "results.json").write_text(json.dumps(res, indent=1))
    (out / "results.md").write_text(to_md(res))
    print(json.dumps({k: v for k, v in res.items() if k not in ("equivalence",)}, indent=1))
    if "error" in res:
        return 1
    return 4 if res["checks"]["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
