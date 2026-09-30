# RustedVINO — Config Reference

Full field reference for `config.json`. For the setup walkthrough (creating the file, sample
config, running the server) see **[`README.md`](README.md)**.

Every model lives under `models` as one stanza combining its VRAM estimate, kind, and
per-model policy overrides — there is no separate `vram_gb`/`model_kinds`/`model_policies`
map. A model ID not listed here is unknown (404 everywhere), even if its directory exists
under `models_dir`.

**`Config` fields** (see `src/model_manager/config.rs` for full doc comments):

| Field | Type / default | Description |
|---|---|---|
| `models_dir` | `PathBuf`, required | Directory of model subdirectories, one per model ID |
| `device` | `String`, required | OpenVINO device string for inference (`"GPU.1"`, `"GPU"`, `"CPU"`, …) |
| `preload` | `[]` | Model IDs to load at startup; each must have a `models` entry. Manage it at runtime with `POST /v1/admin/config/preload` instead of hand-editing this array — `{"from_live": true}` snapshots whichever models are currently loaded as the new list |
| `models` | `{}` | Map of model ID → `{vram_gb, kind, ...per-model policy}` — see `ModelEntry`/`ModelPolicy` below |
| `total_vram_gb` | required | Usable VRAM (GB) on the inference device; `0.0` disables VRAM gating |
| `max_num_seqs` | `16` | CB scheduler concurrency cap and the HTTP 429 admission gate; `0` = OV default (256) |
| `cache_size_gb` | `0.0` | Cap on the KV cache pool; `0.0` = uncapped (model grabs all remaining VRAM) |
| `default_kv_cache_gb` | `0.0` | Global default bounded KV-pool target (GB) per model; `0.0` = unbounded |
| `vram_safety_margin_gb` | `1.0` | VRAM held back for driver/runtime overhead |
| `min_kv_cache_gb` | `1.0` | Minimum KV pool GB; a load that would leave less is refused |
| `system_ram_reservation_gb` | `null` | System RAM (GB) held back from the shared "system" domain budget; `null` = per-OS default |
| `system_ram_budget_gb` | `null` | **UMA boxes only.** System RAM (GB) the server may treat as its own — the unified-memory twin of `total_vram_gb`. `null` keeps the historical live gate (`MemAvailable` alone). When set, the gate uses `min(MemAvailable + reclaimable GPU page cache, this)`, because `MemAvailable` **omits** the DRM/TTM page pool — freed GPU pages the kernel holds for reuse, counted as `used`. Measured 2026-09-07: that pool held ~8.8 GB and a 12 GB model was refused on a 30 GB box as "can never fit". Opt-in because crediting reclaimable cache is the less conservative direction; `system_ram_reservation_gb` still applies as an OS floor on top |
| `light_model_max_gb` | `2.0` | Size threshold (GB) below which an LLM/VLM is "light" for auto placement |
| `light_stt_max_gb` | `1.0` | Size threshold (GB) below which a Whisper model is "light" for auto placement |
| `dgpu_size_ceiling_fraction` | `0.8` | Default-placement guard: a model whose `vram_gb` exceeds this fraction of a discrete GPU's usable VRAM is skipped for that device in *default* tier preference; explicit `device`/`tier_preference` sidesteps it. `>= 1.0` disables |
| `domain_budgets` | `{}` | Per-memory-domain VRAM budget overrides, keyed by domain id (GPU name or `"system"`) |
| `device_budgets` | `{}` | Cross-pipeline device admission caps, keyed by live OV device string; empty = passthrough |
| `device_admission_queue_timeout_ms` | `5000` | Queue timeout for `device_budgets` (separate from `admission_queue_timeout_ms`) |
| `kv_cache_precision` | `"u8"` | `"u8"` compresses the KV cache 8-bit, roughly doubling context capacity for a given VRAM budget, small quality cost; `""` keeps the plugin's native precision (f16 on Intel GPU); `"f16"`/`"f32"` explicit |
| `enable_prefix_caching` | `true` | Reuse KV blocks across CB requests sharing an identical prompt prefix. Side effect on seeded sampling: a reused-prefix prefill rounds slightly differently from a full one, so the first request for a prompt can differ from its later identical repeats (the repeats match each other). Set `false` when exact reproducibility of the first answer matters |
| `cors_allowed_origins` | `["*"]` | CORS allowed origins |
| `api_keys` | `[]` | **Deprecated, migration tripwire only — never read for auth.** A non-empty value is a hard startup error naming where to move the keys. Real inference-scope keys live in `keys_file` |
| `admin_api_keys` | `[]` | **Deprecated, same tripwire as `api_keys`.** Real admin-scope keys live in `keys_file` |
| `keys_file` | `null` | Path to the real Bearer-key file (`{"api_keys": [...], "admin_api_keys": [...]}`), never `config.json` itself. `null` resolves to `<config-dir>/<config-stem>.keys.json` (e.g. `config.json` → `config.keys.json`); a relative value resolves against the config file's own directory. Never git-tracked (`.gitignore`: `*.keys.json`); should be mode `0600`. **If no keys file exists at boot, every `/v1/admin/*` route and `/metrics` return a permanent `503 admin_not_configured` for that process's lifetime** — creating the file afterward needs a restart (or `POST /v1/admin/keys/reload` once admin is unlocked some other way). An existing-but-empty keys file (`{}`) is valid and unlocks admin with no auth required |
| `admission_queue_timeout_ms` | `5000` | How long to queue a request before returning 429 |
| `eviction_grace_secs` | `300.0` | Window (seconds) after a model finishes serving before it becomes eviction-eligible; `0.0` disables the grace window. Per-model `eviction_grace_secs` overrides this |
| `realtime_defaults` | `null` | Default `{stt_model, llm_model, tts_model, embed_model}` seeded into new realtime voice sessions |
| `realtime_viable_minimum` | `null` | Minimum LLM viability for realtime voice sessions: `{llm: {kinds, min_context_tokens, require_capabilities}}`; `kinds` defaults to `["text_gen","vision"]` |
| `embedding_device` | `null` | OpenVINO device for embedding models; `null` = auto-select |
| `default_embed_model` | `null` | Default embedding model seeded into new realtime sessions |
| `embedding_pooling` | `"mean"` | `"mean"`, `"cls"`, or `"last_token"` |
| `embedding_normalize` | `true` | L2-normalize embedding vectors |
| `max_prompt_array` | `16` | Max prompts accepted in one legacy `/v1/completions` array request (each one is a full generation). Does not apply to `/v1/embeddings` |
| `max_embedding_inputs` | `256` | Max inputs in one `/v1/embeddings` request (`400 too_many_inputs` beyond it); must be `≥ 1` |
| `max_embedding_batch_tokens` | `32768` | Per-request `/v1/embeddings` budget in *padded* tokens — inputs × the longest input's token count, because a batch is padded to its longest input (`400 batch_too_large` beyond it; `0` = no budget). It bounds the batch's GPU working memory, which the server's memory accounting doesn't see and which is kept until the model is unloaded: ≈28 MB per 512-token input measured on multilingual-e5-large. The default is 64 full 512-token chunks; short queries are cheap (256 queries of ~10 tokens ≈ 2.6k) |
| `max_tokens_cap` | `8192` | Server-wide ceiling on `max_tokens`/`max_completion_tokens`; `0` = uncapped |
| `bind_addr` | `"127.0.0.1"` | Bind address. For LAN access set `"0.0.0.0"` **and** populate the keys file (the safe way); `allow_insecure_public_bind: true` is needed only to run a non-loopback bind with no keys at all (an open server) |
| `port` | `11437` | HTTP port |
| `allow_insecure_public_bind` | `false` | Explicit opt-in to a non-loopback bind while no keys file is configured (open server, no credentials at all). Not required for a LAN bind that already has a populated keys file |
| `ov_cache_dir` | `null` | Directory for the OpenVINO compiled-blob cache (GPU kernels, and NPU LLMs' weightless compiled blob: ~6 s cached load vs ~30 s cold, measured on one Lunar Lake laptop). **NPU Whisper never uses it** (it hangs with a cache dir). Each NPU `max_prompt_len`/`min_response_len` combination adds its own blob (~0.66 GB for an 8B int4 model); blobs from earlier combinations stay attributed to the model, so the `ov_cache_max_gb` sweep won't prune them while it is loaded (and never with `ov_cache_max_gb: 0`) — after experimenting with shapes, stop the server and clear the cache dir (the next load recompiles, ~30 s). Once unloaded, a model's blobs become prunable, but the sweep then runs only every `ov_cache_sweep_interval_secs` and deletes oldest-first until under the cap, not stale shapes specifically. RustedVINO also maintains a `cache_manifest/` directory — one JSON file per model, recording OV blob attribution and (independent of `ov_cache_dir`) image-gen `model_hash` memoization across restarts. Placed alongside `ov_cache_dir` as its sibling when `ov_cache_dir` is set, otherwise `$XDG_CACHE_HOME/rustedvino/cache_manifest` or `~/.cache/rustedvino/cache_manifest` — not separately configurable |
| `ov_cache_max_gb` | `0.0` | Ceiling on `ov_cache_dir`'s total on-disk size (GB); `0.0` = unbounded. Enforced by a sweep that runs once at startup and then on `ov_cache_sweep_interval_secs`, pruning once the cap is exceeded — blobs for models no longer in your config go first, then configured-but-not-resident ones, then unattributed files, oldest-modified first within each group. Blobs of `Ready` or `Loading` models are never deleted. `0.0` still reports the cache size every sweep; set a non-zero value only if you want files deleted |
| `ov_cache_sweep_interval_secs` | `21600` (6h) | How often the cache-management sweep re-runs after its startup pass |
| `kv_pressure_monitor_enabled` | `false` | Enables the KV-cache pressure monitor: samples every `Ready` model's live pool occupancy and flags one that stays at/above `kv_pressure_threshold_pct` continuously for `kv_pressure_sustained_secs`. **Detect-and-flag only** — it never evicts, resizes or otherwise acts, and it is deliberately excluded from `/health` (pressure is a live-load signal, not a fault). **Enabling it requires setting both knobs below: the server refuses to boot otherwise** (see the note under this table) |
| `kv_pressure_threshold_pct` | `0.0` = unset | Occupancy percentage (0–100) a model must reach to count as under pressure. **No usable default is provided on purpose** — the right number differs by hardware, since a UMA box's KV pool contends with host RAM while a discrete-VRAM box's does not. `90.0` is a reasonable starting point but is *not* validated against real production traffic |
| `kv_pressure_sustained_secs` | `0` = unset | How long occupancy must stay continuously at/above the threshold before flagging — reacting to *sustained* pressure, not a momentary spike. Note the current rule resets the timer on a **single** sub-threshold sample, so pick this together with `kv_pressure_sweep_interval_secs` rather than in isolation |
| `kv_pressure_sweep_interval_secs` | `15` | How often the pressure monitor samples. Cheap (atomic in-memory reads, no I/O). At the default `15`, a `60`s sustain window is spanned by only 4–5 samples, and because one sub-threshold sample resets the timer, a brief dip can restart the clock — sample faster if you need reliable detection of a condition that fluctuates |
| `kv_resize_cooldown_secs` | `30` | Minimum gap between two `POST /v1/admin/models/{id}/resize` calls for the same model; a call inside the window gets `429`. **Applies unconditionally — independent of `kv_pressure_monitor_enabled`**, because the resize endpoint exists and needs a guardrail whether or not anything is monitoring |
| `supervisor` | see below | Tuning for the `--supervise` crash auto-restart wrapper |

> **Boot refusal — KV-pressure knobs.** `kv_pressure_monitor_enabled: true` combined with an unset
> (`<= 0`) `kv_pressure_threshold_pct` or `kv_pressure_sustained_secs` fails config validation and
> the server **will not start**, naming the missing knob. This is deliberate: a box that turns
> monitoring on must state its own threshold and duration, because one global number would mean
> different things on different hardware. Practical consequence: a partial edit takes the server
> down rather than starting it in a degraded state — set all three together, and confirm the service
> comes back.
>
> Note also that the monitor can only observe models whose KV pool is actually queryable
> (`rustedvino_kv_cache_usage_supported = 1`). Embedding/TTS/STT models have no KV pool, and a model
> loaded on the vision/VLM path has one that cannot be read. If no such model is resident, the
> monitor observes nothing at any threshold — check before relying on it.

**Per-model fields** (`models["id"]`, flattened — no nested `"policy"` wrapper):

| Field | Type / default | Description |
|---|---|---|
| `vram_gb` | required | VRAM estimate (GB); `0.0` skips VRAM gating for this model. On Linux, `GET /v1/admin/models` shows what a loaded model really uses (`gpu_memory_estimate_gb`), and the server logs a warning when that's more than `vram_gb` — use it to size this value. An embedding model's figure includes its batch cache |
| `kind` | `null` | `"text_gen"`, `"vision"`, `"embedding"`, `"stt"`, `"tts"`, `"image_gen"`, `"reranking"` — auto-detected if absent |
| `pinned` | `false` | Never chosen as an eviction victim |
| `priority` | `0` | Soft eviction-order weight; higher = evicted later |
| `kv_cache_gb` | `null` | Per-model bounded KV pool target (GB), overriding `default_kv_cache_gb` |
| `chat_template` | `null` | Path to a chat template that **overrides** the one in the model directory; relative paths resolve against the config file's directory. For converted models shipping a stale or incomplete template — e.g. LFM2's renders *nothing* for an assistant turn carrying `tool_calls`, so a tool-call turn replays as an empty assistant message and the model sees a result it never asked for. A corrected LFM2 template ships at `templates/lfm2.jinja`. A configured-but-unreadable override is a hard error, never a silent fall-back |
| `max_concurrent_streams` | `null` | Per-model concurrency cap, overriding `max_num_seqs` for this model |
| `load` | `"eager"` | What a chat request for this model does while it isn't loaded. `"eager"`: fails with a plain `503` — the model loads only via `preload` or the admin API. `"on_demand"`: starts a background load and answers `503` + `Retry-After` until it's ready (same on every device, NPU included). Neither value loads the model at startup; only `preload` does |
| `evictable` | `true` | When `false`, never chosen as an eviction victim (separate hard-exclude from `pinned`) |
| `device` | `null` | Explicit per-model OpenVINO device override (e.g. `"GPU.0"`) |
| `tier_preference` | `[]` | Ordered capability-tier labels: `"heavy"`, `"strong-igpu"`, `"weak-igpu"`, `"npu"`, `"fallback"` |
| `reasoning_parser` | `null` | `"qwen3"`, `"gpt_oss"`, `"mistral"`, `"phi"`. If absent, only `"qwen3"` is auto-detected (from a chat template that reads `enable_thinking`); `"gpt_oss"`, `"mistral"` and `"phi"` must be set explicitly |
| `max_prompt_len` | `null` | **NPU-only**: compile-time `MAX_PROMPT_LEN` ceiling for the static `LLMPipeline`; must be `≥ 1` when set |
| `min_response_len` | `null` | **NPU-only**: compile-time `MIN_RESPONSE_LEN`, the output room the NPU's fixed KV cache reserves on top of `max_prompt_len` (OpenVINO default 128). Prompt + answer can never exceed `max_prompt_len + min_response_len`; hitting it reports `finish_reason: "length"`. Raise it (e.g. `512`) for long answers to long prompts; must be `≥ 1` when set |
| `speculative` | `null` | Opt-in draft-model speculative decoding — see [Speculative decoding](#speculative-decoding-draft-model-assisted-generation) below |
| `precision` | `null` | Image-gen only: reported verbatim as `generation_metadata.precision` (e.g. `"int8"`); operator-supplied, not inferred |
| `model_source` | `null` | Image-gen only: reported verbatim as `generation_metadata.model_source` (e.g. a HF repo id) |
| `model_revision` | `null` | Image-gen only: reported verbatim as `generation_metadata.model_revision` (e.g. an HF commit hash) |
| `capabilities` | `[]` | Declared model capabilities, e.g. `"tool_calling"`. Validated against a known list — an unrecognized label is a hard registration error, not a silent no-op |
| `eviction_grace_secs` | `null` | Per-model override of the top-level `eviction_grace_secs` |

`"supervisor"` sub-fields (`--supervise` wrapper, Unix only): `backoff_base_secs` (5),
`backoff_max_secs` (60), `max_restarts` (5), `restart_window_secs` (600),
`health_timeout_secs` (90), `hang_kill_grace_secs` (10), `watchdog_poll_secs` (30) — how often
to poll `GET /v1/admin/health/watchdog` once the worker is healthy, `watchdog_hang_ceiling_secs`
(600) — age past which an in-flight generation is treated as hung and killed; `0` disables the
generation watchdog entirely.

## Speculative decoding (draft-model assisted generation)

Opt-in per model: a small "draft" model proposes candidate tokens that the real ("target")
model verifies in batches, amortizing memory-bandwidth-bound decode when the target is slow
enough for the trade to pay off. **Enabling this makes every request to that model
speculative — OpenVINO GenAI rejects any non-assisted request on a draft-attached pipeline
outright, so there is no per-request opt-out once a model has `speculative` configured.**

```json
"qwen3-14b-int8-ov": {
  "vram_gb": 15.0,
  "speculative": {
    "draft_model": "qwen3-0.6b-int8-ov",
    "draft_vram_gb": 0.8,
    "num_assistant_tokens": 5
  }
}
```

`speculative` sub-fields:

| Field | Type / default | Description |
|---|---|---|
| `draft_model` | `String`, required | Directory name under `models_dir` — need NOT be its own `models` entry (and normally shouldn't be: it's never independently loadable/evictable this way) |
| `draft_vram_gb` | `f64`, required | Operator-measured VRAM footprint of the draft (weights + its own KV), folded into the target's admission charge |
| `draft_device` | `String`, `null` | OpenVINO device for the draft. `null` = same device as the target. **v1 refuses any explicit value that differs from the target's resolved device** — cross-device drafting is unverified |
| `num_assistant_tokens` | `usize`, `5` | Candidate tokens proposed per verification step — the only value validated in this project's investigation |
| `verify_on_load` | `bool`, `true` | Run a load-time greedy-equivalence self-check (plain vs. draft-attached decode must agree on a fixed probe) before serving. Set `false` only to skip the ~10–30s extra pipeline construction on a pairing already verified on this exact box/driver/OV version — see "Load-time self-check" below |

**Before opting in, measure TPOT (time-per-output-token) on the target *without* a draft.**
Testing across several same-lineage Qwen3 pairings found a clean, monotonic relationship between
baseline decode latency and speculative-decoding benefit — roughly, **the target needs to
already be ≳19 ms/token on a Battlemage (B-series) card before a draft is worth attaching**; a
fast target (~10 ms/token) measured as a net *loss*. This is a starting heuristic, not a
guarantee — see the three variables below, every one of which can make a TPOT-qualified pairing
a loss anyway. **There is no config-time way to verify a net win; only measuring your own
pairing, on your own box, with your own prompt shapes, does.**

**Three independent variables the validation gates below CANNOT detect — measure your own
pairing before trusting the TPOT heuristic alone:**

1. **Draft/target relatedness**, not just tokenizer match. A same-tokenizer-family,
   cross-lineage pairing sat inside the winning TPOT band and passed every gate, yet lost on an
   open-ended prompt — every observed win was a *same-lineage* pairing (same training family),
   not merely a shared tokenizer.
2. **Prompt shape.** RAG-shaped (low-entropy, quoting/retrieval) and open-ended (high-entropy,
   creative) traffic can diverge — sometimes a win on one and a loss on the other for the exact
   same pairing. Benchmark both shapes you actually expect to serve, not just one.
3. **Deployment shape and warm-up state.** Co-resident models and KV cache pool size measurably
   affect the achieved speedup, and assisted decoding has its own multi-request GPU-kernel
   warm-up curve that plain decoding does not have — the first several requests after loading a
   speculative-decoding-enabled model can run 20–40% below its own steady-state throughput. A
   raw, isolated benchmark script's numbers do not automatically transfer to your production
   config.

**Validation gate refusals** (checked at config-validate time and again at load time; a refusal
means the model's `speculative` block is rejected — the model does not load with a draft
attached, or does not load at all if the base gates already fail elsewhere):

- Target is not a dense text-generation model (vision/embedding/stt/tts/image_gen), is on
  `NPU` (no `ContinuousBatchingPipeline` there), or is a **MoE** model (measured as a net
  throughput loss on the only MoE model tested — blocked until separately verified).
- `draft_model` is not a plain dense LLM directory (missing `openvino_model.xml`), is itself a
  VLM directory, or is a MoE model (banned symmetrically with the target ban).
- Target and draft `vocab_size` don't match, **or vocab_size can't be read on either side** — an
  unverifiable pairing is refused outright, never silently allowed through.
- `draft_device` is set explicitly and differs from the target's resolved device.

**Warned, not refused:** a differing `model_type` between target and draft when `vocab_size`
still matches (cross-family pairings have shown latency alone does not predict a win — variable
1 above); `draft_model` also registered as its own independently-loadable `models` entry (its
weights are then resident twice — once as the draft, once as its own entry — not incorrect, just
worth knowing); no explicit `max_concurrent_streams` on a speculative model (auto-capped to 1 —
multi-stream speculative serving is completely unverified; set the field explicitly to override,
still with a WARN).

**Load-time self-check.** With `verify_on_load: true` (the default), every load builds a plain
engine, runs a fixed low-entropy probe, drops it, builds the draft-attached engine, runs the same
probe, and compares output — **a mismatch fails the load loudly** (the model does not come up in
degraded-silent mode) rather than silently serving diverged output. This catches the class of
failure the MoE finding proved is real (a pairing that decodes differently from plain but never
trips an error otherwise). **A passing self-check proves correctness, never benefit** — it says
nothing about whether the pairing is actually faster (see the three variables above).

**KNOWN BUG (unresolved as of this writing): non-greedy sampling (`temperature > 0` and/or
`top_p`) on a draft-attached model intermittently triggers an OpenVINO GenAI internal assertion
failure — `Check 'content_length <= prompt_ids.size() + m_generated_ids.size()' failed` at
`sequence_group.cpp:32`, upstream vendored code, not RustedVINO's own. Reproduces with no
concurrency needed, fails non-monotonically across a temperature sweep, and **fails every other
request sharing the same batched engine step**, not just the offending one — a single bad request
can collaterally kill unrelated healthy (including greedy) concurrent requests. Not a process
crash; the engine recovers per-request. The load-time self-check does **not** catch this — it only
probes greedy decoding. **Until resolved, do not attach a `speculative` draft to any model a
client might send `temperature > 0` or `top_p` to** (which is most real OpenAI-API clients, whose
defaults are rarely greedy) unless you've independently verified your OV version doesn't hit this.

**`enable_prefix_caching: false` does NOT mitigate this** — it stops the assertion from firing,
but response content is silently corrupted instead; the assertion was only ever a canary for an
underlying token/context desync, not the bug itself. **Draft detachment is the only verified-safe
mitigation.**

Root cause (unconfirmed upstream, not yet filed): a non-greedy rejection landing on exactly the
*last* draft candidate appears to skip a token-count rollback in OpenVINO GenAI's sampler,
leaving the processed-token count one too high for the rest of that sequence — consistent with
the assertion's exact overshoot, why it's non-greedy-only, and why it's non-deterministic (needs
the first rejection to land on the last candidate specifically). Setting `num_assistant_tokens: 1`
(every rejection is then a last-candidate rejection) reproduces the failure far more reliably,
consistent with this mechanism. Intel's own OVMS (same `openvino.genai` backend) restricts EAGLE3
speculative decoding to greedy-only, no prefix caching, no concurrency, and doesn't demonstrate
non-greedy sampling for the plain pairing anywhere in its own docs either.
