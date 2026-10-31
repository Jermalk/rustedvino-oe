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
integrated Xe2 GPU). The tools under [Benchmark](#benchmark) measure comparable workloads on your own
hardware; until you have run them, treat the numbers above as illustrative.
**Caveat:** the numbers above were measured with an 8-bit KV cache, `kv_cache_precision: "u8"` explicitly
set (still the server's default; `"u8"` is the older spelling of `"i8"`, which the server stores and
passes as `"i8"` — see [`CONFIG.md`](CONFIG.md)), not the OpenVINO plugin's native f16 — on the Lunar
Lake iGPU the swing between the two was itself +21%, so treat the KV precision as part of the
configuration under test, not a free variable.

---

## Current state — full OpenAI-compatible API, plus extensions

RustedVINO serves chat, legacy completions, embeddings, reranking, typed decisions (Ollama's
System One API), STT, TTS, image generation/editing, and a realtime voice pipeline. The cross-pipeline device admission
middleware is implemented and live-verified across every work class, the realtime voice turn
included.

### What works today

**API surface**

| Endpoint | Status |
|---|---|
| `GET /health` (liveness; body carries `gpu_poisoned`), `GET /v1/models`, `GET /v1/models/{id}` | ✅ |
| `GET /health_generate` — 1-token GPU deep-readiness probe (503 `gpu_poisoned` before probing once the GPU context is poisoned) | ✅ |
| `GET /metrics` — Prometheus endpoint (same port); full metric reference in [`METRICS.md`](METRICS.md) | ✅ |
| `POST /v1/chat/completions` (SSE streaming + non-streaming) | ✅ |
| `POST /v1/completions` — legacy raw-prompt completions | ✅ |
| `POST /v1/embeddings` — text embeddings (bge, e5, …); per-input length checked against the model's own limit, up to 256 inputs and 32k padded tokens per request (limits in [`INTERFACE.md`](INTERFACE.md)) | ✅ |
| `POST /v1/rerank` — cross-encoder document reranking | ✅ |
| `POST /v1/systemone` — Ollama-compatible typed decisions (choice / yes-no / score) with probabilities from one prefill per question, no text generation; wire format pinned to Ollama v0.35 by fixtures generated from Ollama's own code. Needs a `kind: "decision"` model (Bespoke-Nimble-9B converted to OpenVINO; [`CONFIG.md`](CONFIG.md#decision-models-v1systemone)) | ✅ |
| `POST /v1/audio/transcriptions` — STT via Whisper OpenVINO, including Whisper fine-tunes such as Vietnamese PhoWhisper (see [Vietnamese speech](#vietnamese-speech-phowhisper--piper)) | ✅ |
| `POST /v1/audio/translations` — speech in any Whisper language → English text (needs a multilingual Whisper trained for translation, e.g. `large-v3`; `large-v3-turbo` isn't) | ✅ |
| `POST /v1/audio/speech` — TTS (Kokoro-82M, Coqui VITS, Piper VITS voices such as Vietnamese `vi_VN-vais1000`, `SpeechT5`) | ✅ |
| `POST /v1/images/generations` — text-to-image (SDXL, FLUX.1-schnell) | ✅ |
| `POST /v1/images/edits` — inpainting + img2img (SDXL) | ✅ |
| `POST /tokenize` / `POST /detokenize` (text and vision models; shape in [`INTERFACE.md`](INTERFACE.md)) | ✅ |
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
| Sampling params: `temperature`, `top_p`, `top_k`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, `repetition_penalty` (extension field — server applies a `1.1` safety-net default unless the client sends `1.0` to opt out). Without a `seed`, each request gets a fresh random one; an explicit `seed` is reproducible (with prefix caching on, the first request for a prompt can differ from its repeats — see `enable_prefix_caching` in CONFIG.md) | ✅ |
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
| Config reload — `POST /v1/admin/config/reload` (diffs disk config vs. live registry; a model that isn't loaded picks up any changed field, including next-load settings like `max_prompt_len` or `kv_cache_gb`; loaded models are left alone) | ✅ |
| Config/registry drift audit — `GET /v1/admin/config/audit` | ✅ |
| Voice-flow pin — `GET /v1/admin/voice-pin` (shared STT/LLM/TTS set for realtime sessions) | ✅ |
| Realtime session introspection/kill — `GET`/`DELETE /v1/admin/realtime/sessions[/{id}]` | ✅ |
| On-demand lazy load, opt-in per model (`"load": "on_demand"` in config) — first request kicks off the load and returns `503` + `Retry-After`, the retry serves once ready. Default policy is `eager`, which does **not** auto-load: a chat request to an unloaded eager model is a bare `503` until it's loaded via the admin API or config. | ✅ |
| VRAM budget tracking (internal; OV Cores are blind to each other) | ✅ |
| VRAM estimator: STT/embed/image charge weight-only (no KV pool reservation) | ✅ |
| LRU auto-eviction when a new model doesn't fit | ✅ |
| `domain_budgets` — per-memory-domain VRAM budget overrides, keyed by domain id (a GPU's device name, or `"system"`) | ✅ |
| STT placement under `default_placement: "auto"` — CPU first, then a discrete GPU, then an integrated GPU as a last resort; never the NPU unless you name it | ✅ |
| Capability-based device placement (`placement.rs` tier preference, no hardcoded names) | ✅ |
| Self-managing OpenVINO compile-cache — size-reported always, size-bounded and self-pruning when `ov_cache_max_gb` is set (see [Compile-cache management](#compile-cache-management)) | ✅ |

**Observability & responses**

| Feature | Status |
|---|---|
| Response headers: `x-request-id`, `x-server`, `x-api-version`, `x-server-version` (admin-only), `x-ruvi-host` (the box's hostname, only on responses to a request with a valid inference or admin key; never on `/health`, or on routes left open because no key of that scope is configured); CORS config | ✅ |
| OpenAI-compatible error envelopes on all error paths; an oversized request (over 50 MiB) gets `413 request_too_large` | ✅ |
| Prometheus push metrics: tokens/sec EMA, TTFT histogram, request duration histogram | ✅ |
| Image-gen `generation_metadata` (`/v1/images/generations`/`/edits`) — `model_family`, `device`, `host` (valid-key callers only, like `x-ruvi-host`), `model_hash`, `sampler`/`scheduler_config`, `openvino_version`, `precision`/`model_source`/`model_revision` | ✅ |
| Chat completions — `model_family` + effective `sampling` (`temperature`/`top_p`/`top_k` actually used) on every response | ✅ |
| `GET /v1/admin/models` — `server{host, pid, uptime_secs, openvino_version, engine, engine_version}` block (`host`, `pid` and `uptime_secs` only for a valid-key caller) | ✅ |
| OV compile-cache metrics — `rustedvino_ov_cache_bytes` (current on-disk size), `rustedvino_ov_cache_pruned_bytes_total`/`_pruned_files_total` (cumulative, from the `ov_cache_max_gb` sweep) | ✅ |
| KV-cache pressure metrics — `rustedvino_kv_cache_usage_percent` + `rustedvino_kv_cache_usage_supported` (per model; `supported=0` means the value is structurally unqueryable for that engine kind, **not** an empty pool) and `rustedvino_kv_cache_pressure_flagged` (1/0, set by the optional monitor — see [`CONFIG.md`](CONFIG.md)) | ✅ |
| `GET /v1/admin/models` — per-model `kv_cache_usage_pct` + `kv_pressure_flagged`, so the pressure signal is readable without Prometheus | ✅ |
| Measured device memory (Linux) — `rustedvino_process_gpu_memory_bytes` (per driver/device/region, from the kernel) and a per-model estimate (`rustedvino_model_gpu_memory_estimate_bytes`, admin `gpu_memory_estimate_gb`), with a log warning when a model uses clearly more than its configured `vram_gb` plus its KV pool (above 5 % and 0.75 GB). See [`METRICS.md`](METRICS.md) | ✅ |
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

Unit and integration tests run with `scripts/rv-cargo.sh test`. They need no GPU and no model (the mock engine path covers the HTTP surface), only the OpenVINO runtime from step 1 for the build. Tests that need a real model or real hardware are `#[ignore]`d; run them with `--ignored`.

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
- **NPU LLM prompt and answer caps** — see [Hardware & NPU notes](#hardware--npu-notes) below.
- **Speech translation is into English only** (`/v1/audio/translations`, as in OpenAI's API), and
  it needs a Whisper model trained for it: `whisper-large-v3` translates well, while
  `whisper-large-v3-turbo` returns untranslated text. For other target languages, transcribe and
  translate with a chat model.
- **Kokoro TTS voices are English only** (US `af_*`/`am_*`, UK `bf_*`/`bm_*`). Any other voice name,
  including Kokoro's own non-English voices, silently falls back to the default English voice.

---

## Route table

Every endpoint — OpenAI-compatible surface, the Ollama-compatible `/v1/systemone`, plus
RustedVINO's own extensions (admin/lifecycle control, rerank, tokenize/detokenize, realtime voice
WS) — is listed with method and
description in **[`INTERFACE.md`](INTERFACE.md)**.

---

## Setup

### 1. Prerequisites

```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# A C++17 compiler — build.rs compiles the OpenVINO bridge (ov_bridge/ov_bridge.cpp)
# via the `cc` crate. Missing g++/clang++ fails with a cc-crate error that does not
# name the real cause. python3-venv is for the OpenVINO venv below.
sudo apt install build-essential python3-venv    # or your distro's equivalent

# OpenVINO runtime + GenAI, in a venv (plain `pip install` is blocked on Ubuntu 24.04 /
# Mint 22: "externally-managed-environment"). scripts/rv-cargo.sh finds a ~/ov*/ venv
# on its own; no activation needed.
python3 -m venv ~/ov && ~/ov/bin/pip install openvino==2026.4.1 openvino-genai==2026.4.1 openvino-tokenizers==2026.4.1.0
```

**Rust version.** The crate is `edition = "2024"` and declares
`rust-version = "1.91"` in `Cargo.toml`, which is the floor Cargo will enforce.
Development and CI-equivalent testing happen on **1.98.1** — that is the version
this code is actually built and tested against, so it is the safe choice if you
hit anything odd on an older compiler. There is deliberately no
`rust-toolchain.toml` in this repo: pinning an exact toolchain is right for the
upstream development tree but wrong for a public crate, where it would override
whatever you have installed.

**A note on the OpenVINO version.** The pip pins above (2026.4.1) are the runtime
this is developed against. The vendored GenAI headers under `ov_bridge/include/`
are 2026.4.1.0 — `build.rs` synthesises unversioned symlinks precisely so the exact
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
convert models itself.

Models go in whatever directory you set as `models_dir` in step 4 — the two
**must** match, or every request 404s with no hint that the path is why. These
examples use `/opt/rustedvino/models`, matching the config below; substitute
your own path in both places if you prefer.

#### Starter model set

One small model per kind, all published under [huggingface.co/jermalk](https://huggingface.co/jermalk)
and each tested with this server. The directory name is the model id you use in requests.

| Kind | Model | On disk | RAM in use* | Speed on CPU* (range over five machines) |
|---|---|---|---|---|
| Text | `qwen3-1.7b-int4-ov` | 1.2 GB | 2.9 GB | 13–41 tokens/s |
| Vision | `qwen3.5-4b-int4-ov` | 3.3 GB | 6.8 GB | 6–17 tokens/s; first token after an image ≈ 6–15 s |
| Embedding | `multilingual-e5-large-int8-ov` | 0.6 GB | 1.0 GB | 18–59 texts/s (batches of 16) |
| Reranking | `bge-reranker-v2-m3-int8-ov` | 0.6 GB | 1.0 GB | 17–47 passages/s (≈ 0.02–0.06 s each) |
| Speech-to-text | `whisper-large-v3-turbo-int8-ov` | 0.8 GB | 2.1 GB | 0.17–0.9× real time |
| Speech-to-text, optional | `whisper-large-v3-int8-ov` | 1.5 GB | 4.0 GB | 0.61× real time |
| Text-to-speech, English | `speecht5-tts-ov` | 0.6 GB | 2.5 GB | 0.3–0.6× real time |
| Text-to-speech, Vietnamese | `vi-vais1000-piper` (recipe below) | 61 MB | 0.3 GB | 0.08× real time |

\* Measured with `tools/bench/rvbench.py` on five machines (Core Ultra 7 258V laptop, i7-11800H laptop, i3-14100, i5-13500, Ryzen 7 5800H; **models on the CPU**, three runs per number). The slowest box was the Ryzen (AVX2 only, probably one memory channel); the fastest per-core results came from the i3/i5 desktops. The release bundle's `config.json` loads every model on the CPU
(`"device": "CPU"`, `"default_placement": "device"`) unless you pass `--device`, which is what these figures are; a GPU is something you opt in to. Speeds moved up to ±25 % between runs on that machine, so read them as orders of
magnitude, not promises; more machines will be added. "RAM in use" is the server process's peak while it
worked — much more than it shows right after loading, because weights are mapped lazily. Speech-to-text
speeds were timed on synthesised speech, so they measure speed, not recognition quality. All figures can be
reproduced with `tools/bench/rvbench.py`.

```bash
export MODELS_DIR=/opt/rustedvino/models
sudo mkdir -p "$MODELS_DIR" && sudo chown "$USER" "$MODELS_DIR"
python3 -m venv ~/hf && ~/hf/bin/pip install -U huggingface_hub   # provides ~/hf/bin/hf (no sudo; plain pip is blocked on Ubuntu 24.04 / Mint 22)
export PATH="$HOME/hf/bin:$PATH"

# download only what you need — each is independent
for m in qwen3-1.7b-int4-ov multilingual-e5-large-int8-ov bge-reranker-v2-m3-int8-ov whisper-large-v3-turbo-int8-ov; do
  hf download "jermalk/$m" --local-dir "$MODELS_DIR/$m"
done
hf download jermalk/qwen3.5-4b-int4-ov --local-dir "$MODELS_DIR/qwen3.5-4b-int4-ov"          # vision, optional
hf download jermalk/whisper-large-v3-int8-ov --local-dir "$MODELS_DIR/whisper-large-v3-int8-ov"  # optional, see below
hf download jermalk/speecht5-tts-ov --local-dir "$MODELS_DIR/speecht5-tts-ov"                      # English voice
```

Things worth knowing before you try them (run the commands in the next block once the server is up,
step 6):

- **Qwen3 "thinks" by default.** Send `"enable_thinking": false` unless you want the reasoning; with a small
  `max_tokens` the whole budget can go into the reasoning and the answer comes back empty.
- **Small models are weak in Polish.** The 1.7B writes fluent English, but in Polish it makes factual and
  translation errors (for example, the wrong river for a capital city). Use a larger model when Polish
  facts matter.
- **e5 wants prefixes:** `"query: "` on search questions and `"passage: "` on the texts being searched.
  Without them retrieval quality drops silently.
- **Reranking cost grows with the list** (≈ 0.02–0.06 s per passage on CPU, far less on a discrete GPU): rerank the candidates your retrieval step returns (a few dozen is fine), not thousands.
  `bge-reranker-v2-m3` is multilingual; an English-only reranker such as `ms-marco-MiniLM` was measured at
  less than half the accuracy on Polish and made a good e5 ranking worse, so it is not in the set.
- **Whisper large-v3 vs turbo.** Turbo is smaller and about 1.5× faster on CPU and agreed with large-v3 on
  98 % of the words of a real Polish lecture. Only large-v3 supports `/v1/audio/translations`, which turns
  speech into **English** only (it will not translate English into another language). Don't run turbo on
  the NPU yet: a request hung one machine's NPU until the server was restarted.
- **Vision on a GPU.** Qwen3.5 / Qwen3.6 / Qwen3.8 are hybrid linear-attention models. On OpenVINO GenAI 2026.4.1 they run on a GPU with the normal settings, prefix caching included: `./rv add qwen3.6-27b-int4-ov --device GPU.n`. The KV pool of the 27B with the default 2 GB `cache_size_gb` is small (about 13,900 tokens on the 0.8.4 line), and a longer prompt is refused up front with a clean 400 `context_length_exceeded`. Check the numbers for your card; the 0.8.x measurements were taken on one Arc Pro B60. From 0.9.0 the pool of a hybrid model is sized in whole blocks and counts only its attention layers (a 4 GiB u4 pool of the 27B holds about 203,000 tokens), but with prefix caching on, which halves a hybrid model's capacity, and for vision models on a GPU without a readable free-memory probe, the context limit stays on the conservative estimate; `rv ps` marks an estimate with `~` and a limit that follows the free device memory with `*`.
- **Memory:** a model's "RAM in use" is on top of the OS and your other programs, and the server keeps
  4 GB free for the OS. When RAM runs short, loading a model **unloads the least recently used ones** to make
  room (the server log says `LRU eviction`), and requests to an unloaded model then fail with `model is not
  loaded` until you load it again (`./rv load <model>`; `rv add` is only for a model not yet registered). On a 16 GB machine keep to one or two of these at a time; the vision model
  alone needs ≈ 6.8 GB and is refused outright when less than that is free.

```bash
# register + load (the server must be running). `rv add` asks for confirmation; add -y to skip it.
# With the bundle's config.json every model loads on the CPU; add --device GPU.n (see ./rustedvino --device-info for the names) to use a GPU.
# A bare --device GPU is rejected on a machine with more than one GPU. An integrated GPU is usually slower than the CPU - try it before relying on it.
./rv add qwen3-1.7b-int4-ov
./rv add qwen3.5-4b-int4-ov                   # vision; on a GPU add --device GPU.n (see the vision note above)
./rv add multilingual-e5-large-int8-ov --kind embedding
./rv add bge-reranker-v2-m3-int8-ov --kind reranking
./rv add whisper-large-v3-turbo-int8-ov
./rv add speecht5-tts-ov --kind tts
# `rv` reads the first `admin_api_keys` entry of your keys file (step 5), or `RV_ADMIN_KEY`.
# If the keys file holds no keys at all (for example `{}`), the server is OPEN: `rv` still works but warns
# when it runs list, status, version and the other commands, because anyone who can reach the port can then load, unload and delete models. That is
# acceptable for a single-user machine on 127.0.0.1 and NOT SECURE anywhere else - add keys (step 5).
# If the keys file does hold keys, add -H "Authorization: Bearer <key>" to the curl calls below.

# text
curl -s http://localhost:11437/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3-1.7b-int4-ov","enable_thinking":false,"messages":[{"role":"user","content":"Explain what a vector database is in two sentences."}]}'

# vision — a local picture, base64-encoded
curl -s http://localhost:11437/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model":"qwen3.5-4b-int4-ov","enable_thinking":false,"max_tokens":100,
  "messages":[{"role":"user","content":[
    {"type":"text","text":"Describe this image."},
    {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,'"$(base64 -w0 photo.jpg)"'"}}]}]}'

# embeddings (note the prefix)
curl -s http://localhost:11437/v1/embeddings -H "Content-Type: application/json" \
  -d '{"model":"multilingual-e5-large-int8-ov","input":["passage: Warszawa jest stolicą Polski.","query: stolica Polski"]}'

# reranking
curl -s http://localhost:11437/v1/rerank -H "Content-Type: application/json" \
  -d '{"model":"bge-reranker-v2-m3-int8-ov","query":"stolica Polski","documents":["Warszawa leży nad Wisłą.","Pierogi z kapustą."]}'

# speech-to-text (a WAV file)
curl -s http://localhost:11437/v1/audio/transcriptions -F model=whisper-large-v3-turbo-int8-ov -F file=@speech.wav

# text-to-speech (English; plain 16 kHz voice) - writes hello.wav
curl -s http://localhost:11437/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"model":"speecht5-tts-ov","input":"Hello from RustedVINO.","language":"en"}' -o hello.wav
```

The English voice is deliberately small and plain. Digits and symbols in `input` are spoken as words;
send `"normalize": false` if your text is already speakable. A Polish voice is not in the starter set yet.

Other models: browse [huggingface.co/OpenVINO](https://huggingface.co/OpenVINO) for ready-made IR (not
every model has one), or convert your own with `optimum-cli` (needs `optimum[openvino]`):

```bash
optimum-cli export openvino --model Qwen/Qwen3-8B --weight-format int4 "$MODELS_DIR/qwen3-8b-int4-ov"
```

**Piper voices (TTS)** are plain ONNX, no conversion needed: put the voice's `<name>.onnx` and
`<name>.onnx.json` from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) in
their own model directory and register it with `"kind": "tts"`. Piper voices read *phonemes*,
which RustedVINO gets from **espeak-ng run as a separate program** (`sudo apt install
espeak-ng`); it is never linked into or shipped with RustedVINO. At load, the server checks
that your espeak-ng phonemizes a reference sentence exactly as the voice expects and refuses the
voice if not, so an espeak-ng that would mispronounce fails loudly instead. That check exists
for Vietnamese (`vi_VN-vais1000-medium`, measured to match Piper's own phonemes exactly with
espeak-ng 1.52.0); other languages load with a warning that pronunciation is unverified. Mind
each voice's license: vais1000's training data is CC BY 4.0 (attribution required).

```bash
mkdir -p "$MODELS_DIR/vi-vais1000-piper" && cd "$MODELS_DIR/vi-vais1000-piper"
for f in vi_VN-vais1000-medium.onnx vi_VN-vais1000-medium.onnx.json MODEL_CARD; do
  curl -sLO "https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium/$f"
done
```

#### Vietnamese speech (PhoWhisper + Piper)

**Speech-to-text:** [PhoWhisper](https://huggingface.co/vinai/PhoWhisper-large) (VinAI, BSD-3-Clause)
is a Vietnamese fine-tune of Whisper and runs as an ordinary `"kind": "stt"` model once converted.

**Ready-made OpenVINO int8 builds** (unofficial conversions, with VinAI's licence and a model card),
download one into your models directory:
[small](https://huggingface.co/jermalk/PhoWhisper-small-int8-ov) (243 MB; ~3× faster than real time on
CPU; 2.4% word errors on the clips below),
[medium](https://huggingface.co/jermalk/PhoWhisper-medium-int8-ov) (746 MB; 1.2%),
[large](https://huggingface.co/jermalk/PhoWhisper-large-int8-ov) (1.5 GB; 1.2%).

```bash
hf download jermalk/PhoWhisper-large-int8-ov --local-dir "$MODELS_DIR/phowhisper-large-int8-ov"
```

To convert it yourself, mind the traps in the export:

1. The Hub repo ships only `pytorch_model.bin`, and optimum's export calls the Hub API. Re-save it
   locally with transformers (`from_pretrained(..., use_safetensors=False)` → `save_pretrained`),
   then export offline (`HF_HUB_OFFLINE=1`).
2. Use **Python 3.12**: on Python 3.14, optimum's Whisper export fails with
   `NormalizedConfig.__init__() got multiple values for argument 'allow_new'`. Tested with
   optimum-intel 2.0.0, optimum 2.2.0, transformers 4.57.6, OpenVINO 2026.2.1.
3. Keep the working copy and `TMPDIR` on disk, not a tmpfs `/tmp` (the int8 IR write fails there).
4. The re-save drops tokenizer files: after exporting, copy `added_tokens.json`, `merges.txt`,
   `normalizer.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`,
   `vocab.json` and `preprocessor_config.json` from the original download into the exported model
   directory, then run `convert_tokenizer . --with-detokenizer --skip-special-tokens -o .` there.

```bash
TMPDIR=/path/on/disk HF_HUB_OFFLINE=1 optimum-cli export openvino \
  --model ./phowhisper-large-resaved --task automatic-speech-recognition-with-past \
  --weight-format int8 "$MODELS_DIR/phowhisper-large-int8-ov"
```

It runs on NPU, CPU and GPU. On one Lunar Lake laptop its GPU load crashed inside Intel's OpenCL
compiler once in about ten loads (a driver-stack issue; stock `whisper-large-v3` never did), so
the NPU is the steady choice there.

Measured on that laptop — **5 synthetic clips (Piper vais1000 speech), 18.1 s, 85 words**, so
treat it as a smoke test, not a benchmark:

| Model (device) | Word error rate | Time for the 18.1 s |
|---|---|---|
| PhoWhisper-large (NPU) | 1.2% | 6.3 s |
| PhoWhisper-large (CPU) | 1.2% | 28.0 s |
| whisper-large-v3-turbo (NPU) | 3.5% | 2.5 s |
| whisper-large-v3 (GPU) | 4.7% | 9.4 s |

PhoWhisper writes lowercase with little punctuation; the stock models' misses were tone marks.

**Text-to-speech:** the Piper `vi_VN-vais1000` voice above.

**Voice conversation:** with PhoWhisper (NPU) → `qwen3-8b` (GPU) → vais1000 (CPU) set on a
`/v1/realtime` session, a spoken Vietnamese question gets a spoken Vietnamese answer; on that
laptop, **~3.2 s from the end of speech to the first reply audio** once warm.

### 4. Create config file

```bash
sudo mkdir -p /opt/rustedvino
sudo tee /opt/rustedvino/config.json << 'EOF'
{
  "models_dir": "/opt/rustedvino/models",
  "device": "GPU",
  "preload": [],
  "default_placement": "auto",
  "cache_size_gb": 2.0,
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

Leaving a setting out silently takes its built-in default, and JSON cannot carry comments, so the
shipped example configs write **every** setting out at its default and `CONFIG.md` explains each.
To do the same for your file: `./target/release/rustedvino --print-default-config` prints all
settings with their defaults (`--print-default-model` does it for one model stanza), and
`scripts/fill-config-defaults.py config.json` adds whatever your file is missing without changing
your values (`--check` after an upgrade shows what is new).

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

To run it as a service, start `rustedvino --supervise` under systemd with
`scripts/rustedvino.user.service.example` (the Linux bundle ships it as `rustedvino.user.service.example`).
Keep its `KillSignal=SIGTERM` and `TimeoutStopSec=720`: a stop waits up to 600 s for a model load that is
still running, and systemd's default 90 s would SIGKILL the server in the middle of that load, which can leave
the GPU driver stuck (seen on an Arc Pro B60).

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
  -d '{"model":"multilingual-e5-large-int8-ov","input":"hello world"}'

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

### `rv` — Ollama-shaped admin CLI

An ergonomic wrapper over the admin API above — one command instead of hand-building `curl`
calls for everyday model lifecycle work. The release bundles ship it as `rv` (`rv.exe` on Windows):
a small native binary with no Python and no OpenVINO libraries, so it still answers when the
runtime is broken. From a source checkout, `cargo build --release -p rv` builds it (no OpenVINO
install needed); `scripts/rv` is the original single-file Python version, kept as a reference and an
alternative (Python 3.8+, stdlib only).

```bash
rv start | stop | restart | version   # run the server by hand, and which one answers
rv status                  # process/health/auth/models/VRAM in one report
rv list                    # registered models, plus on-disk-but-unregistered ones
rv ps                      # loaded models only, with live KV%/in-flight counts
rv check qwen3-8b-int4-ov  # dry-run admission check — what would loading it evict?
rv add MODEL --vram-gb 5.5 -y   # register an on-disk model and load it
rv load MODEL              # load an already-registered model
rv unload MODEL            # evict from VRAM, stays registered
rv deregister MODEL -y     # remove from the registry + config.json
rv patch MODEL --pinned true --priority 5  # update fields of an already-registered model
rv preload --from-live -y  # snapshot the currently-loaded set as the new boot default
rv logs -n 200             # tail recent server log lines
rv reload-config           # pick up config.json edits — registers changes, loads/evicts nothing
rv resize MODEL --kv-cache-gb 4 -y   # evict + reload a loaded model with a new KV-cache size
rv completeness            # every registered model's file-completeness report (exit 1 on a gap)
rv reload-keys             # re-read the keys file and hot-swap the API/admin keys (to REPLACE the admin key: RV_ADMIN_KEY=<old key> rv reload-keys)
rv save-defaults -y        # remember the loaded models: they load again at every start
rv voice-pin               # the realtime voice-flow model pin
rv sessions                # live realtime sessions   (also: rv session ID, rv session-kill ID -y)
```

`start`, `stop`, `restart`, `version`, `resize`, `completeness`, `reload-keys`, `voice-pin`, `sessions`,
`session` and `session-kill` exist only in the native `rv`; the Python script covers the other commands.
Commands that evict, deregister, resize or close something ask for confirmation; `-y` skips the prompt, and without a terminal they refuse (exit 2) instead of guessing.
`--json` gives machine-readable output where it makes sense (`status`, `list`, `ps`, `check`,
`reload-config`, `patch`, `preload`, `completeness`, `reload-keys`, `voice-pin`, `sessions`); other
commands accept the flag and ignore it.

**Config resolution:** `RUSTEDVINO_CONFIG`, else a `config.json` next to the executable (the
release-bundle layout), else the platform default: `/opt/rustedvino/config.json` on Linux,
`%LOCALAPPDATA%\RustedVINO\config.json` on Windows. The server's own order puts
the platform default (`/opt/rustedvino/config.json`, or on Windows the per-user file if it exists)
before the executable's directory, so if both exist `rv` can read a
different config than the server it talks to — set `RUSTEDVINO_CONFIG` to remove the doubt. Keys come
from that config's `keys_file` field, resolved the way the server resolves it. Admin key:
auto-loaded from the resolved keys file, or set `RV_ADMIN_KEY` / pass `--admin-key`. A config the
server would reject (unknown fields, bad values) does not stop `rv`: it reads only `port`,
`models_dir`, `keys_file` and the model policies, and tolerates a UTF-8 BOM.

**Port:** taken from the resolved config's own `port`, else `11437`; override with `--port` or
`RV_PORT`. The target is always `127.0.0.1` — `rv` reads the models directory and keys directly off
the machine it runs on, so it only ever describes a local server.

**Process info:** `rv status` takes the server's PID and uptime from the server itself (the admin
`server` block of `GET /v1/admin/models`, 0.8.0+, valid-key callers only). For an older server, or
when none is reported, it falls back to OS probing on Linux (`/proc`, the supervisor PID file,
`ps` for uptime); elsewhere it reports "PID unknown" rather than "not running". Under a supervisor
the reported uptime is the worker's, which the line says.

---

## Client tools

Small, dependency-free Python clients in `tools/`. They use the server's own API, so they work
against any RustedVINO instance (`--target`, default `http://localhost:11437`):

- **[`tools/semsearch`](tools/semsearch/README.md)** — semantic search over a folder of Markdown
  notes, through `/v1/embeddings` and, optionally, `/v1/rerank`.
- **[`tools/livecaptions`](tools/livecaptions/README.md)** — live captions from a microphone, a
  call or a media file, through `/v1/audio/transcriptions`, with optional per-caption
  translation through `/v1/chat/completions` and SRT output. Needs `ffmpeg`.

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
  "ov_cache_sweep_interval_secs": 21600,         // default 6h
  "ov_cache_layout": "keyed"                     // "flat" = the 0.8.x layout
}
```

**One directory per KV precision.** On OpenVINO 2026.4.x a compiled blob does not record
the KV-cache precision, so a blob compiled under one precision is silently reused under
another, or fails at the first request with `Incorrect block size`. With the default
`"keyed"` layout the text models (including a draft model) and the vision models on GPU
and CPU compile into `ov_cache_dir/kv-<tag>/`, one directory per `kv_cache_precision`
string passed to the plugin: `kv-auto` (nothing passed), `kv-i8`, `kv-u4`, `kv-f16`. Vision
models on a GPU pass their resolved precision (a per-model value, else the IR hint, else
`kv-u4` for int4 weights and `kv-i8` for the rest); on the CPU, with `kv_precision_pass:
"plugin"`, or when the server cannot read the model's layout with confidence, they pass none
(`kv-auto`).
Whisper, image, TTS, embedding and NPU models stay in the directory itself. Every
directory holds a full compiled copy of its models, so trying several precisions costs
that much disk per precision. `"flat"` restores the 0.8.x paths **and the precision mixing described above**: it is a
rollback, not a safe setting, and blobs left in the root by an earlier flat run are reused
without a recompile. A keyed directory the server cannot write to (a read-only cache) is not
used; those models compile without a blob cache. Use one `ov_cache_dir` per instance, host and OpenVINO version when
`ov_cache_max_gb` is set: two builds on different OpenVINO versions sharing a directory
rewrite each other's blobs.

**If a model loads but every request fails with `Incorrect block size`.** The compiled blob
in its cache directory was built for another KV precision (a blob copied between `kv-*`
directories, or `ov_cache_layout: "flat"` with a changed `kv_cache_precision`). The server
stays healthy and deletes nothing; it logs one WARN per model that names the directory.
Stop the server, delete the `*.blob` files there (they are read-only files, mode 0440; the
directory is yours to clean), and start it again.

**Two notes.** *Windows:* permissions you set on `ov_cache_dir` with `icacls ... /T` or
with `(OI)(CI)` inheritance flags are inherited by the `kv-*` subdirectories; nothing extra
to configure. *Weights are not hashed:* OpenVINO keys a blob by the model's graph and the
device, not by the weight values, so replacing a model's weights in place (same topology,
same files and layout) can reuse a stale blob. Delete that model's blobs, or the cache
directory, after swapping weights. This predates the keyed layout.

**What the sweep does.** It runs once at startup and then every
`ov_cache_sweep_interval_secs`, on a blocking thread (it stats every file and can
SHA-256 multi-gigabyte weights, so it never touches the async runtime). Every sweep
reports the cache's total on-disk size — *unconditionally*, even with
`ov_cache_max_gb` unset, so you get size visibility without opting into pruning.
Only if the total exceeds the budget does it delete anything.

**What it will not delete.** The blobs a `Ready` or `Loading` model is using: those it
owns on the device it runs on, in the directory it compiles into (the cache root or its
`kv-<tag>/` directory). A `Loading` model may be reading or rewriting that exact file at
that moment, and deleting it underneath is the failure this guard exists to prevent.
Everything else a text or vision model owns is eligible, including 0.8.x blobs left in the
root after the move to `kv-<tag>/` and the directories of other precisions. (Whisper, image,
TTS and NPU models cache in the root itself, so all their blobs there count as in use while
the model is.) An unattributed file
younger than 30 minutes is left alone too (a compile may still be writing it). Kernel
caches (`*.cl_cache`) are never deleted. `Evicting` and `NotLoaded` models do no cache
I/O, so their blobs stay eligible.

**Eviction order**, most disposable first, oldest-modified first within each tier:

1. blobs whose owning model is **no longer in your config** at all
2. blobs for a **configured model that is not using them** (not resident, another
   device, an older layout or precision)
3. **unattributed** blobs — anything the server cannot map to a model, including
   files predating this feature

It stops as soon as the cache is back under budget.

**Metrics** (`/metrics`): `rustedvino_ov_cache_bytes` is the last measured total,
updated per sweep rather than at scrape time; `rustedvino_ov_cache_pruned_bytes_total`
and `rustedvino_ov_cache_pruned_files_total` are cumulative counters since process
start, and only move when a budget is set and was exceeded. The size covers the root and
the `kv-*` directories. Full reference in
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
- STT (Whisper) compiles on the NPU, but `default_placement: "auto"` does **not** send it there: whisper-large-v3-turbo
  hung the NPU on a clip over ~60 s (until the server was restarted), so STT defaults to the CPU, then a discrete GPU (an integrated GPU only as a last resort).
  Put a small Whisper (e.g. `whisper-tiny-fp16-ov`) on the NPU yourself with `"device": "NPU"` if you want it.
- Embeddings (`TextEmbeddingPipeline`) and LLM (`ContinuousBatchingPipeline`) cannot compile on
  NPU at all — both tiers omit NPU; an LLM on a Lunar Lake box lands on the CPU by default (name the Arc iGPU with
  `"device": "GPU"` - it measured 1.6× faster than the CPU on decode and 11× on prefill); the NPU's single-stream
  `LLMPipeline` path needs an explicit override (see below).
- The NPU LLM path requires channel-wise (`*-cw-ov`) int4 IR — group-wise quantization fails NPU
  compile.
- **Prompt/context cap: 1024 tokens by default, configurable.** The NPU compiles a fixed-shape
  graph ahead of time (unlike GPU/CPU's dynamic-shape kernels), so `MAX_PROMPT_LEN` is baked in
  at compile time. Set a per-model `"max_prompt_len"` (e.g. `2048`) on the model's `models` entry
  to raise it; omit it to keep OpenVINO's default. A change takes effect at the model's next load.
  A templated prompt over the cap the loaded model was compiled with gets a clean
  `400 context_length_exceeded`, pre-flighted before generation starts — GPU has no such ceiling.
- **Answer room: 128 tokens by default on top of the prompt cap, configurable.** The NPU's KV
  cache is fixed at `max_prompt_len + min_response_len` tokens, so the answer gets whatever the
  prompt leaves: `max_prompt_len + min_response_len − prompt tokens`, whatever the request's
  `max_tokens`. When prompt + answer fill it, generation ends and the response says
  `finish_reason: "length"`. Set a per-model
  `"min_response_len"` (e.g. `512`) for long answers to long prompts; measured on one Lunar Lake laptop
  (Arc 140V), 512 added no noticeable load time.
- **Memory:** an NPU LLM is charged its weights only (`vram_gb`), not a GPU-style KV pool; its
  fixed KV cache is small and inside the measured footprint.
- **NPU LLMs use the `ov_cache_dir` blob cache.** With `ov_cache_dir` set,
  an NPU LLM stores one weightless compiled `.blob` (~0.66 GB for an 8B int4 model) and later
  loads take ~6 s instead of ~30 s cold (measured on one Lunar Lake laptop, identical output).
  Without it, the NPU driver's own cache (`~/.cache/ze_intel_npu_cache/`) still keeps repeat loads
  at ~8–12 s. Each `max_prompt_len`/`min_response_len` combination compiles its own blob, and
  blobs from earlier combinations stay attributed to the model, so the `ov_cache_max_gb` sweep
  won't prune them while it is loaded (and never prunes anything with the default
  `ov_cache_max_gb: 0`). After experimenting with shapes, stop the server and clear
  `ov_cache_dir`; the next load recompiles the current shape (~30 s).
  NPU Whisper uses `ov_cache_dir` too (cold load ≈ 4 s, warm ≈ 0.3 s on one Lunar Lake laptop). Large Whisper models can
  hang on the NPU on long clips; keep them on the CPU or a GPU.

---

## Benchmark

Two stdlib-only Python tools live in [`tools/bench/`](tools/bench/): `rvbench.py` produces the per-model
figures in the starter-set table (`python3 tools/bench/rvbench.py --help`), and `cmpbench.py` + `cmptable.py`
run the same chat workload against RustedVINO, OVMS and OpenArc on your own card (see
[`CMPBENCH.md`](tools/bench/CMPBENCH.md)). Re-run them on your hardware rather than quoting the headline
numbers under [Why](#why).

---

## Project structure

Module-to-purpose map (`src/`, `ov_bridge/`, `scripts/rv-cargo.sh`/`rv`, `tools/`, `tests/`): **[`STRUCTURE.md`](STRUCTURE.md)**.

---

## Roadmap

Phase table: **[`ROADMAP.md`](ROADMAP.md)**. Currently on Phase 5 (media/infrastructure).
What changed in each release: **[`CHANGELOG.md`](CHANGELOG.md)**.

---

## Stack

`axum` (incl. `ws`) · `tokio` · `tower`/`tower-http` · `serde`/`serde_json` · `anyhow` · `tracing` ·
`metrics` + `metrics-exporter-prometheus` · `minijinja` · `rubato` · `symphonia` · `image` ·
`kokoro-tts` · `ort` · `uuid` · `sysinfo` · OpenVINO GenAI (C++ via custom bridge)

---

## Authors

- **Jerzy Marek Majchrzak** — project lead, architect, maintainer.
- **Claude Code** (Anthropic) — engineering agent.

RustedVINO is a product of agentic engineering. Jerzy set the direction, defined what the
server must do, made every architectural and release decision, and verified the results on
real Intel hardware. Claude Code, working as a coding agent under that direction, was
responsible for the implementation details.

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
private engineering log it references) lives upstream and isn't public. If you open a
PR, it'll be reviewed and, once accepted, folded upstream by hand and re-exported here rather
than merged directly — expect that round-trip instead of a same-repo merge.

---

## Related

- [stormVINO](https://github.com/Jermalk/stormVINO) — Python original, behavioural spec
- [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)
- Article series: *"Custom OpenVINO Server"* on Medium (Parts 1–3 published; RustedVINO = Parts 4+)
