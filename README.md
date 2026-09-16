# RustedVINO 🦀

> **An OpenAI-compatible inference server for Intel hardware, written in Rust —
> GPU, NPU, and CPU, all in play.**  
> Inspired by [stormVINO](https://github.com/Jermalk/stormVINO), my own earlier Python server —
> built to outgrow its asyncio/GIL overhead, not to mirror its API line-for-line.

Runs the full Intel silicon stack, not just the dGPU: Arc dGPUs (B50/B60/B70), Lunar Lake's
Arc iGPU, the NPU ("AI Boost"), and CPU — each model lands on the right device via
capability-based placement (`placement.rs`), never a hardcoded device name. See
[Hardware & NPU notes](#hardware--npu-notes) for what runs where today.

**Don't want to compile it?** Pre-built, self-contained Linux x86_64 bundles — the OpenVINO
runtime is included, so there is nothing else to install — are on the
**[Releases page](https://github.com/Jermalk/rustedvino-oe/releases)**. Unpack, point it at a
model directory, run. Building from source is [Setup](#setup) below.

---

## Why

stormVINO's Python/asyncio/SSE stack adds a fixed ~7 ms overhead per token — the GPU sits idle
while Python schedules, acquires the GIL, and encodes JSON. RustedVINO removes that overhead and
uses OpenVINO's `ContinuousBatchingPipeline` to batch many users' decode steps into one GPU pass:

```
                     stormVINO     RustedVINO (B60)
  1 user              ~45.6         69.0 tok/s   →   1.5×
  8 users (~1k tok)    ~45.6        407 tok/s    →   8.9×
 16 users (128 tok)    ~45.6        814 tok/s    →  17.8×
```

Validated across four Intel GPUs (Arc B50/B60/B70 — all Battlemage Xe2 — and Lunar Lake's
integrated Xe2 GPU). A standardized, reproducible benchmark harness that anyone can run against
their own hardware is planned — see [Benchmark](#benchmark) — until it lands, treat the numbers
above as illustrative rather than independently reproducible.
**Caveat:** the numbers above were measured with `kv_cache_precision: "u8"` explicitly set (now
the server's own default, see [`CONFIG.md`](CONFIG.md)), not the OpenVINO plugin's native f16 —
on the Lunar Lake iGPU the swing between the two was itself +21%, so treat `u8` as part of the
configuration under test, not a free variable.

---

## Current state — full OpenAI-compatible API, plus extensions

RustedVINO serves chat, legacy completions, embeddings, reranking, STT, TTS, image
generation/editing, and a realtime voice pipeline. The cross-pipeline device admission
middleware is implemented and live-verified across every work class, the realtime voice turn
included.

### What works today

**API surface**

| Endpoint | Status |
|---|---|
| `GET /health`, `GET /v1/models` | ✅ |
| `GET /health_generate` — 1-token GPU deep-readiness probe | ✅ |
| `GET /metrics` — Prometheus endpoint (same port); full metric reference in [`METRICS.md`](METRICS.md) | ✅ |
| `POST /v1/chat/completions` (SSE streaming + non-streaming) | ✅ |
| `POST /v1/completions` — legacy raw-prompt completions | ✅ |
| `POST /v1/embeddings` — text embeddings (bge, e5, …) | ✅ |
| `POST /v1/rerank` — cross-encoder document reranking | ✅ |
| `POST /v1/audio/transcriptions` — STT via Whisper OpenVINO | ✅ |
| `POST /v1/audio/speech` — TTS (Kokoro-82M, Coqui VITS, `SpeechT5`) | ✅ |
| `POST /v1/images/generations` — text-to-image (SDXL, FLUX.1-schnell) | ✅ |
| `POST /v1/images/edits` — inpainting + img2img (SDXL) | ✅ |
| `POST /tokenize` / `POST /detokenize` | ✅ |
| `GET /v1/realtime` — WebSocket voice flow (STT → LLM → TTS pipeline, barge-in, session history/retrieval) | ✅ |

**Inference engines**

| Engine | Status |
|---|---|
| `ContinuousBatchingPipeline` — N concurrent streams in one GPU step | ✅ |
| NPU LLM engine — static `LLMPipeline` single-stream path (Lunar Lake NPU) | ✅ |
| VLM chat — vision-language models (`/v1/chat/completions`, image input **or** text-only) | ✅ |

**Chat & generation features**

| Feature | Status |
|---|---|
| Exact `usage.prompt_tokens` via tokenizer bridge | ✅ |
| `stream_options.include_usage` — token counts on streaming final chunk | ✅ |
| Sampling params: `temperature`, `top_p`, `top_k`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, `repetition_penalty` (extension field — server applies a `1.1` safety-net default unless the client sends `1.0` to opt out) | ✅ |
| `max_completion_tokens` alias for `max_tokens` | ✅ |
| Streaming `<think>` strip — `ThinkFilter` removes reasoning tokens from output | ✅ |
| Non-streaming `<think>` strip — reasoning in `reasoning_content`, hidden from `content` | ✅ |
| Per-model `reasoning_parser` flag — `qwen3` / `gpt_oss` / `mistral` / `phi` extractors | ✅ |
| `enable_thinking` field — explicit thinking budget for Qwen3-style models | ✅ |
| OpenAI tool/function calling — `tools` + `tool_choice` (`"none"`/`"auto"`/`"required"`/named function), stream + non-stream; native Qwen3/Phi/Llama3/Gemma dialect, plus Mistral and Qwen3-Coder XML dialects, auto-detected from the model's own chat template | ✅ |
| Structured output — `response_format` (`"text"` / `"json_object"` / `"json_schema"`) via OpenVINO GenAI's `StructuredOutputConfig` (xgrammar); chat only | ✅ |
| `finish_reason` correctly distinguishes `"stop"` (EOS) vs `"length"` (max_tokens) | ✅ |

**Performance & concurrency**

| Feature | Status |
|---|---|
| KV prefix caching (`enable_prefix_caching`, default **on**) — reuses KV blocks across requests sharing a prompt prefix; 49–83% e2e TTFT reduction on deep multi-turn/RAG prompts, measured on Arc Pro B50 and Lunar Lake's integrated GPU | ✅ |
| Speculative decoding — opt-in per-model draft-model-assisted generation (`speculative` config block); load-time greedy-equivalence self-check rejects a bad pairing before it ever serves traffic | ✅ |
| MoE detection + `max_num_seqs` cap — prevents CB throughput collapse on sparse models | ✅ |
| Fixed KV pool + slot-cap (`max_num_seqs`, `cache_size_gb`) → no deadlocks | ✅ |
| Admission wait-queue (`tokio::Semaphore`, `admission_queue_timeout_ms`) — every engine kind (CB, VLM, embed, STT, TTS, NPU), cancellation-safe | ✅ |
| Cross-pipeline device admission (`device_budgets`) — a second gate capping concurrency across *different* engine kinds sharing one physical device (e.g. LLM + VLM + STT + TTS + embedder all on one GPU); ships **disabled by default** (empty config ⇒ pure passthrough, zero overhead) | ✅ (opt-in) |

**Model & resource management**

| Feature | Status |
|---|---|
| Runtime model add — `POST /v1/admin/models/add` (registers + loads, persists to config) | ✅ |
| Runtime model load — `POST /v1/admin/models/{id}/load` | ✅ |
| Runtime model evict — `DELETE /v1/admin/models/{id}` | ✅ |
| Runtime model deregister — `DELETE /v1/admin/models/{id}/register` | ✅ |
| Runtime model update — `PATCH /v1/admin/models/{id}` (partial update of an already-registered model's fields; persisted always, live where safe) | ✅ |
| Runtime KV-pool resize — `POST /v1/admin/models/{id}/resize` (evict + reload a resident model at a new `kv_cache_gb`; other overrides carried forward, cooldown-gated) | ✅ |
| Preload-list management — `POST /v1/admin/config/preload` (explicit list, or `{"from_live": true}` to snapshot the currently-loaded set as the new boot default) | ✅ |
| Config reload — `POST /v1/admin/config/reload` (diffs disk config vs. live registry) | ✅ |
| Config/registry drift audit — `GET /v1/admin/config/audit` | ✅ |
| Voice-flow pin — `GET /v1/admin/voice-pin` (shared STT/LLM/TTS set for realtime sessions) | ✅ |
| Realtime session introspection/kill — `GET`/`DELETE /v1/admin/realtime/sessions[/{id}]` | ✅ |
| On-demand lazy load, opt-in per model (`"load": "on_demand"` in config) — first request kicks off the load and returns `503` + `Retry-After`, the retry serves once ready. Default policy is `eager`, which does **not** auto-load: a chat request to an unloaded eager model is a bare `503` until it's loaded via the admin API or config. | ✅ |
| VRAM budget tracking (internal; OV Cores are blind to each other) | ✅ |
| VRAM estimator: STT/embed/image charge weight-only (no KV pool reservation) | ✅ |
| LRU auto-eviction when a new model doesn't fit | ✅ |
| `domain_budgets` — per-memory-domain VRAM budget overrides, keyed by domain id (a GPU's device name, or `"system"`) | ✅ |
| Size/EU-aware STT placement — heavy Whisper routes to CPU, light to NPU/iGPU | ✅ |
| Capability-based device placement (`placement.rs` tier preference, no hardcoded names) | ✅ |
| Self-managing OpenVINO compile-cache — size-reported always, size-bounded and self-pruning when `ov_cache_max_gb` is set (see [Compile-cache management](#compile-cache-management)) | ✅ |

**Observability & responses**

| Feature | Status |
|---|---|
| Response headers: `x-request-id`, `x-server`, `x-api-version`, `x-server-version` (admin-only), `x-ruvi-host`; CORS config | ✅ |
| OpenAI-compatible error envelopes on all error paths | ✅ |
| Prometheus push metrics: tokens/sec EMA, TTFT histogram, request duration histogram | ✅ |
| Image-gen `generation_metadata` (`/v1/images/generations`/`/edits`) — `model_family`, `device`, `host`, `model_hash`, `sampler`/`scheduler_config`, `openvino_version`, `precision`/`model_source`/`model_revision` | ✅ |
| Chat completions — `model_family` + effective `sampling` (`temperature`/`top_p`/`top_k` actually used) on every response | ✅ |
| `GET /v1/admin/models` — `server{host, openvino_version, engine, engine_version}` block | ✅ |
| OV compile-cache metrics — `rustedvino_ov_cache_bytes` (current on-disk size), `rustedvino_ov_cache_pruned_bytes_total`/`_pruned_files_total` (cumulative, from the `ov_cache_max_gb` sweep) | ✅ |
| KV-cache pressure metrics — `rustedvino_kv_cache_usage_percent` + `rustedvino_kv_cache_usage_supported` (per model; `supported=0` means the value is structurally unqueryable for that engine kind, **not** an empty pool) and `rustedvino_kv_cache_pressure_flagged` (1/0, set by the optional monitor — see [`CONFIG.md`](CONFIG.md)) | ✅ |
| `GET /v1/admin/models` — per-model `kv_cache_usage_pct` + `kv_pressure_flagged`, so the pressure signal is readable without Prometheus | ✅ |
| `GET /v1/admin/health/watchdog` — oldest in-flight generation age; polled by `--supervise` to kill+restart a wedged worker `/health` can't see | ✅ |
| `GET /v1/admin/logs/tail` — tail an in-memory log ring buffer over HTTP | ✅ |

**Reliability, auth & ops**

| Feature | Status |
|---|---|
| Bearer key auth via a separate keys file (never inline in `config.json`); distinct admin-scope keys, hot-reloadable — `POST /v1/admin/keys/reload` | ✅ |
| Crash handler — logs a marker + backtrace on native SIGSEGV/SIGABRT/SIGBUS/SIGILL before the process dies | ✅ |
| `--supervise` — crash auto-restart wrapper with backoff + restart cap (Unix only) | ✅ |
| OpenVINO GenAI C++ bridge (`ov_bridge/`) | ✅ |
| JSON config file (`/opt/rustedvino/config.json`) | ✅ |
| No automatic fallback — wrong model ID → 404, not a guess | ✅ |

**Test coverage**

1089 tests pass (993 lib + 93 integration + 3 supervisor integration). A further 10 are `#[ignore]`d: 6 lib tests needing a real model or real hardware, the 2 golden-template tests, and 2 doc-tests. Run them with `--ignored`.

---

### Known limitations

- **Gemma-family VLMs (`gemma4`) corrupt output on long-context prompts** — duplicated
  fragments, stuttering, eventual collapse. Root-caused via a 6-step elimination (raw curl vs.
  streaming vs. bare `openvino_genai.VLMPipeline`, three Gemma variants) to an upstream
  OpenVINO GenAI defect, not a RustedVINO bug. The fp16 variant additionally hard-crashes the
  whole process on OOM instead of throwing a catchable exception. No workaround yet.
- **Qwen3.5/3.6 (`Qwen3_5ForConditionalGeneration`) KV-admission gate overestimates real
  usable capacity by ~3.3x** — this hybrid linear/full-attention architecture's actual
  per-turn KV cost isn't the uniform per-layer estimate `compute_max_prompt_tokens` uses
  (same root cause as [vLLM issue #37121](https://github.com/vllm-project/vllm/issues/37121)).
  Crossing the real (formula-unknowable) ceiling wedges the pipeline into a silently-"successful"
  zero-token state; mitigated at runtime by a self-check (detect via `generated_tokens==0` →
  honest `503 kv_capacity_wedge` → a learned-ceiling ratchet, scaled by `cache_size_gb` →
  automatic evict+reload), not by a corrected formula. Prefix caching itself *is* wired through
  to this architecture's `VLMPipeline` (`ov_bridge.cpp`'s `OvVlmState` constructor,
  since 2026-08-20) and measurably helps — do not assume the two are the same issue.
- **VLM multi-turn image binding**: images only attach to the *last* message in a
  conversation's history — a hard `ChatHistory` API contract in OpenVINO GenAI, not a bug.
  Chaining multiple images across turns doesn't work the way it might for a text-only
  history.
- **NPU LLM prompt cap and blob-cache absence** — see [Hardware & NPU notes](#hardware--npu-notes) below.

---

## Route table

Every endpoint — OpenAI-compatible surface plus RustedVINO's own extensions (admin/lifecycle
control, rerank, tokenize/detokenize, realtime voice WS) — is listed with method and
description in **[`INTERFACE.md`](INTERFACE.md)**.

---

## Setup

### 1. Prerequisites

```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# OpenVINO (user-level pip; match your Python version)
pip install openvino==2026.2.1 openvino-genai==2026.2.1 openvino-tokenizers==2026.2.1.0

# A C++17 compiler — build.rs compiles the OpenVINO bridge (ov_bridge/ov_bridge.cpp)
# via the `cc` crate. Missing g++/clang++ fails with a cc-crate error that does not
# name the real cause.
sudo apt install build-essential    # or your distro's equivalent
```

**Rust version.** The crate is `edition = "2024"` and declares
`rust-version = "1.91"` in `Cargo.toml`, which is the floor Cargo will enforce.
Development and CI-equivalent testing happen on **1.98.1** — that is the version
this code is actually built and tested against, so it is the safe choice if you
hit anything odd on an older compiler. There is deliberately no
`rust-toolchain.toml` in this repo: pinning an exact toolchain is right for the
upstream development tree but wrong for a public crate, where it would override
whatever you have installed.

**A note on the OpenVINO version.** The pip pins above (2026.2.1) are the runtime
this is developed against. The vendored GenAI headers under `ov_bridge/include/`
are 2026.2.0 — `build.rs` synthesises unversioned symlinks precisely so the exact
patch level of the installed runtime does not have to match.

### 2. Clone and build

```bash
git clone https://github.com/Jermalk/rustedvino-oe
cd rustedvino-oe

# build.rs locates OpenVINO from the env and synthesises unversioned dev symlinks
# into OUT_DIR automatically. scripts/rv-cargo.sh finds your OpenVINO install on
# its own: $RV_OV_VENV if set, else the active virtualenv, else wherever `python3`
# imports openvino_genai from, else a ~/ov*/ or ~/.venv*/ directory. If you ran the
# pip install above, it is found with no further setup.
scripts/rv-cargo.sh build --release

# Only needed to override that detection — point at the site-packages directory
# that contains openvino_genai/:
#   RV_OV_VENV=/path/to/venv/lib/python3.12/site-packages scripts/rv-cargo.sh build --release
```

### 3. Get some models

RustedVINO expects pre-converted OpenVINO IR model directories under `models_dir` — it does not
convert models itself. Two ways to get one:

Models go in whatever directory you set as `models_dir` in step 4 — the two
**must** match, or every request 404s with no hint that the path is why. These
examples use `/opt/rustedvino/models`, matching the config below; substitute
your own path in both places if you prefer.

```bash
export MODELS_DIR=/opt/rustedvino/models
sudo mkdir -p "$MODELS_DIR" && sudo chown "$USER" "$MODELS_DIR"

# (a) pull a pre-converted IR from Intel's own HF org, if one exists there for your model
#     (browse huggingface.co/OpenVINO — not every model is pre-converted)
hf download OpenVINO/<model>-ov --local-dir "$MODELS_DIR/<model>-ov"

# (b) convert any HF model yourself with optimum-cli (needs `optimum[openvino]` installed)
optimum-cli export openvino --model Qwen/Qwen3-8B --weight-format int4 "$MODELS_DIR/qwen3-8b-int4-ov"
```

### 4. Create config file

```bash
sudo mkdir -p /opt/rustedvino
sudo tee /opt/rustedvino/config.json << 'EOF'
{
  "models_dir": "/opt/rustedvino/models",
  "device": "GPU",
  "preload": [],
  "admission_queue_timeout_ms": 30000,
  "models": {
    "qwen3-8b-int4-ov":              { "vram_gb": 5.5 },
    "qwen3-14b-int4-ov":             { "vram_gb": 9.5 },
    "qwen3-coder-30b-a3b-int4-ov":   { "vram_gb": 8.5 },
    "phi-4-int4-ov":                 { "vram_gb": 8.5, "reasoning_parser": "phi" },
    "mistral-small-3.2-24b-int4-ov": { "vram_gb": 13.5 }
  },
  "total_vram_gb": 22.5
}
EOF
```

`device` and `total_vram_gb` above are illustrative — check your own device string first
(command below) and set `total_vram_gb` to what your GPU actually has. Every model lives under
`models` as one stanza combining its VRAM estimate, kind, and per-model policy overrides — there
is no separate `vram_gb`/`model_kinds`/`model_policies` map. A model ID not listed here is
unknown (404 everywhere), even if its directory exists under `models_dir`.

Full field-by-field reference (all `Config` fields, per-model fields, `supervisor` sub-fields):
**[`CONFIG.md`](CONFIG.md)**.

Override the config path at runtime:
```bash
RUSTEDVINO_CONFIG=/path/to/config.json ./target/release/rustedvino
```

Check your GPU device names:
```bash
python3 -c "import openvino as ov; print(ov.Core().available_devices)"
# dGPU boxes:      ['CPU', 'GPU.0', 'GPU.1']  — GPU.1 is the Arc dGPU
# Lunar Lake iGPU: ['CPU', 'GPU', 'NPU']       — GPU = the Arc iGPU (no GPU.0/GPU.1)
```

### 5. Create the keys file (enables auth + the admin API)

Real Bearer keys are never stored in `config.json` — they live in a separate file
(`.gitignore`d), `<config-dir>/<config-stem>.keys.json` by default (e.g.
`/opt/rustedvino/config.json` → `/opt/rustedvino/config.keys.json`; override with `keys_file` in
`config.json`). **This file must exist before you start the server** — if it's missing at boot,
every `/v1/admin/*` route and `/metrics` return a permanent `503 admin_not_configured` for that
process's lifetime (creating it later needs a restart, or `POST /v1/admin/keys/reload` once
admin is unlocked some other way).

```bash
# Minimal — unlocks the admin API with no inference auth required (fine for local/single-user):
sudo tee /opt/rustedvino/config.keys.json << 'EOF'
{}
EOF
sudo chmod 600 /opt/rustedvino/config.keys.json

# Production — real Bearer keys for both scopes:
sudo tee /opt/rustedvino/config.keys.json << 'EOF'
{
  "api_keys": ["sk-..."],
  "admin_api_keys": ["sk-admin-..."]
}
EOF
sudo chmod 600 /opt/rustedvino/config.keys.json
```

An admin key also satisfies the inference gate (superset credential); an inference key presented
to an admin route gets `403 insufficient_scope`, not silent access.

### 6. Build and run

```bash
scripts/rv-cargo.sh build --release
./target/release/rustedvino
# Listens on :11437
```

### 7. Load a model and chat

```bash
# Load a model at runtime (no restart needed)
curl -X POST http://localhost:11437/v1/admin/models/qwen3-8b-int4-ov/load

# Health + model list
curl http://localhost:11437/health
curl http://localhost:11437/health_generate
curl http://localhost:11437/v1/models

# Streaming chat
curl -sN -X POST http://localhost:11437/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-8b-int4-ov","messages":[{"role":"user","content":"hi"}],"stream":true}'

# Non-streaming chat
curl -s -X POST http://localhost:11437/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-8b-int4-ov","messages":[{"role":"user","content":"hi"}],"stream":false}'

# STT transcription
curl -s -X POST http://localhost:11437/v1/audio/transcriptions \
  -F model=whisper-large-v3-int8-ov \
  -F file=@recording.wav

# Text embedding
curl -s -X POST http://localhost:11437/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"multilingual-e5-large-int8","input":"hello world"}'

# Text-to-image
curl -s -X POST http://localhost:11437/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"sdxl-fp16-ov","prompt":"a red fox in the snow","steps":20}'

# Evict when done
curl -X DELETE http://localhost:11437/v1/admin/models/qwen3-8b-int4-ov
```

(Add `-H "Authorization: Bearer sk-..."` to any of the above once the keys file has non-empty
key lists.)

---

## Admin API

Route list and status codes: **[`INTERFACE.md`](INTERFACE.md)**. Auth is gated by the keys
file, not `config.json` — see step 5 above and [`CONFIG.md`](CONFIG.md)'s `keys_file` entry.

### `scripts/rv` — Ollama-shaped admin CLI

An ergonomic wrapper over the admin API above — one command instead of hand-building `curl`
calls for everyday model lifecycle work.

```bash
scripts/rv status                  # process/health/auth/models/VRAM in one report
scripts/rv list                    # registered models, plus on-disk-but-unregistered ones
scripts/rv ps                      # loaded models only, with live KV%/in-flight counts
scripts/rv check qwen3-8b-int4-ov  # dry-run admission check — what would loading it evict?
scripts/rv add MODEL --vram-gb 5.5 -y   # register an on-disk model and load it
scripts/rv load MODEL              # load an already-registered model
scripts/rv unload MODEL            # evict from VRAM, stays registered
scripts/rv deregister MODEL -y     # remove from the registry + config.json
scripts/rv patch MODEL --pinned true --priority 5  # update fields of an already-registered model
scripts/rv preload --from-live -y  # snapshot the currently-loaded set as the new boot default
scripts/rv logs -n 200             # tail recent server log lines
scripts/rv reload-config           # pick up config.json edits — registers changes, loads/evicts nothing
```

**Requirements:** Python 3.8+, stdlib only (`argparse`, `json`, `urllib`, …) — no `pip install`,
no venv. `--json` gives machine-readable output on `status`, `list`, `ps`, `check`,
`reload-config`, `patch` and `preload`; the other commands accept the flag but ignore it, and
`help` does not take it.

**Config resolution:** `RUSTEDVINO_CONFIG`, else a `config.json` next to the script (the
release-bundle layout), else `/opt/rustedvino/config.json`. The
bundle-local step is `rv`'s own convenience — it does **not** resolve identically to the server,
which knows only the env var and the default. Keys come from that config's `keys_file` field, resolved
the same way the server resolves it. Admin key: auto-loaded from the resolved keys file, or set
`RV_ADMIN_KEY` / pass `--admin-key`.

**Port:** taken from the resolved config's own `port`, else `11437`; override with `--port` or
`RV_PORT`. The host is always `localhost` and is not configurable — `rv` reads the PID file, the
log file and `models_dir` directly off the machine it runs on, so it only ever describes a local
server.

**Optional external tools, both degrade gracefully if absent:** `rv status` shells out to `ps`
(process uptime) and `pgrep` (the worker PID under the supervisor process) for two display
fields only — neither is required for `rv` to run or for any other command. Both are standard
on Linux and macOS; on Windows (or a minimal container without them) those two fields just show
"unknown"/blank instead of erroring — `rv` itself is pure Python and has no other OS-specific
calls.

**Not wrapped yet** (use `curl` directly, or ask for it to be added):
`POST /v1/admin/models/{id}/resize`, `GET /v1/admin/models/completeness`,
`POST /v1/admin/keys/reload`, `GET /v1/admin/voice-pin`,
`GET /v1/admin/realtime/sessions`, `GET /v1/admin/realtime/sessions/{id}`,
`DELETE /v1/admin/realtime/sessions/{id}`.
The router in `src/lib.rs` is the source of truth for this list.

---

## Compile-cache management

OpenVINO JIT-compiles a model for the target device on first load and caches the
result, so the second load of the same model is seconds instead of a minute. Point
`ov_cache_dir` at a directory and the server will use it — but a compile cache
grows without bound, and on a box serving many models it can quietly reach tens of
gigabytes. RustedVINO manages it for you.

```jsonc
{
  "ov_cache_dir": "/opt/rustedvino/ov_cache",  // unset = no caching at all
  "ov_cache_max_gb": 20.0,                      // 0.0 = report size, never prune
  "ov_cache_sweep_interval_secs": 21600          // default 6h
}
```

**What the sweep does.** It runs once at startup and then every
`ov_cache_sweep_interval_secs`, on a blocking thread (it stats every file and can
SHA-256 multi-gigabyte weights, so it never touches the async runtime). Every sweep
reports the cache's total on-disk size — *unconditionally*, even with
`ov_cache_max_gb` unset, so you get size visibility without opting into pruning.
Only if the total exceeds the budget does it delete anything.

**What it will not delete.** Blobs belonging to a model that is `Ready` or
`Loading` are protected. A `Loading` model may be reading or rewriting that exact
file at that moment, and deleting it underneath is the failure this guard exists to
prevent. `Evicting` and `NotLoaded` models do no cache I/O, so their blobs stay
eligible.

**Eviction order**, most disposable first, oldest-modified first within each tier:

1. blobs whose owning model is **no longer in your config** at all
2. blobs for a **configured but not currently resident** model
3. **unattributed** blobs — anything the server cannot map to a model, including
   files predating this feature

It stops as soon as the cache is back under budget.

**Metrics** (`/metrics`): `rustedvino_ov_cache_bytes` is the last measured total,
updated per sweep rather than at scrape time; `rustedvino_ov_cache_pruned_bytes_total`
and `rustedvino_ov_cache_pruned_files_total` are cumulative counters since process
start, and only move when a budget is set and was exceeded. Full reference in
[`METRICS.md`](METRICS.md).

---

## Build internals — the four traps

The standard `openvino-genai` Rust crate expects a C API the pip wheel **does not ship**.
`ov_bridge/ov_bridge.cpp` provides it. Four non-obvious issues, all handled by `build.rs`:

| Trap | Symptom | Fix |
|---|---|---|
| ABI mismatch | Silent link failures | Build with `-D_GLIBCXX_USE_CXX11_ABI=1` (pip wheel uses ABI=1) |
| DT_RUNPATH doesn't propagate | `libopenvino.so` not found at runtime | `--disable-new-dtags` → forces `DT_RPATH` (transitive) |
| tokenizers extension path | `[E]` extension load error at startup | Add tokenizers lib dir to `DT_RPATH` |
| versioned soname mismatch | `cannot open shared object` on OV version bump | `build.rs` synthesises unversioned symlinks into `OUT_DIR` — no box-local maintenance |

If you get `cannot open shared object` at runtime:
```bash
ldd ./target/release/rustedvino
readelf -d ./target/release/rustedvino | grep RPATH
```

---

## Hardware & NPU notes

RustedVINO runs across four Intel GPUs (Arc B50/B60/B70 dGPUs, Lunar Lake iGPU+NPU — B50/B60/B70
and Lunar Lake's GPU are all Battlemage/Lunar Lake Xe2 silicon). The facts below are current
operational knowledge, gathered running on that hardware — they affect how you configure a box
today.

**Device strings:** dGPU boxes enumerate `GPU.0`/`GPU.1`; Lunar Lake's UMA iGPU enumerates as
plain `GPU` (no suffix), plus `NPU`.

**NPU (Lunar Lake "AI Boost") constraints:**
- STT (Whisper) works on NPU — a light Whisper variant (e.g. `whisper-tiny-fp16-ov`) routes there
  by default on Lunar Lake; heavier Whisper models route to CPU instead (see size/EU-aware STT
  placement in the feature table above).
- Embeddings (`TextEmbeddingPipeline`) and LLM (`ContinuousBatchingPipeline`) cannot compile on
  NPU at all — both tiers omit NPU; an LLM on a Lunar Lake box lands on the GPU unless explicitly
  overridden to the NPU's single-stream `LLMPipeline` path (see below).
- The NPU LLM path requires channel-wise (`*-cw-ov`) int4 IR — group-wise quantization fails NPU
  compile.
- **Prompt/context cap: 1024 tokens by default, configurable.** The NPU compiles a fixed-shape
  graph ahead of time (unlike GPU/CPU's dynamic-shape kernels), so `MAX_PROMPT_LEN` is baked in
  at compile time. Set a per-model `"max_prompt_len"` (e.g. `2048`) on the model's `models` entry
  to raise it; omit it to keep OpenVINO's default. A templated prompt over the effective cap gets
  a clean `400 context_length_exceeded`, pre-flighted before generation starts — GPU has no such
  ceiling.
- **Never uses the `ov_cache_dir` blob cache** (unlike every other engine kind). Every load —
  including a plain restart with unchanged config — recompiles from scratch (~20–40 s on Lunar
  Lake). One consequence: changing `max_prompt_len` between restarts is always safe (nothing
  persisted could go stale), but there's no fast-restart path for NPU LLMs the way there is for
  everything else.

---

## Benchmark

A standardized, provenance-stamped cross-fleet benchmark tool is planned but not yet part of
this repo — see **[`ROADMAP.md`](ROADMAP.md)**. Until it lands, the headline numbers under
[Why](#why) above are the current reference point.

---

## Project structure

Module-to-purpose map (`src/`, `ov_bridge/`, `scripts/rv-cargo.sh`/`rv`, `tests/`): **[`STRUCTURE.md`](STRUCTURE.md)**.

---

## Roadmap

Phase table: **[`ROADMAP.md`](ROADMAP.md)**. Currently on Phase 5 (media/infrastructure).

---

## Stack

`axum` (incl. `ws`) · `tokio` · `tower`/`tower-http` · `serde`/`serde_json` · `anyhow` · `tracing` ·
`metrics` + `metrics-exporter-prometheus` · `minijinja` · `rubato` · `symphonia` · `image` ·
`kokoro-tts` · `ort` · `uuid` · `sysinfo` · OpenVINO GenAI (C++ via custom bridge)

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). Fully permissive, no copyleft: fork it, run it
commercially, modify it, no obligation to contribute back.

Third-party components bundled in the pre-built releases — notably Symphonia (MPL-2.0) and the
OpenVINO runtime (Apache-2.0) — are listed in [`THIRD-PARTY.md`](THIRD-PARTY.md).

**RustedVINO Bear Edition** is a separate, closed-source commercial tier built as extensions on
top of this Apache-licensed core — never the reverse. The base server in this repo builds and
runs completely on its own; nothing here depends on Bear Edition to function.

---

## About this repository

`rustedvino-oe` is a periodic export of a private development repo, not the primary working
tree. Commits here are squashed on export — the granular commit-by-commit history (and the
private `dev/` engineering log it references) lives upstream and isn't public. If you open a
PR, it'll be reviewed and, once accepted, folded upstream by hand and re-exported here rather
than merged directly — expect that round-trip instead of a same-repo merge.

---

## Related

- [stormVINO](https://github.com/Jermalk/stormVINO) — Python original, behavioural spec
- [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)
- Article series: *"Custom OpenVINO Server"* on Medium (Parts 1–3 published; RustedVINO = Parts 4+)
