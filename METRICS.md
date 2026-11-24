# RustedVINO — Metrics Reference

Every Prometheus metric the server exposes on `GET /metrics`. For the route table and
status codes see **[`INTERFACE.md`](INTERFACE.md)**; for the config fields referenced
below see **[`CONFIG.md`](CONFIG.md)**.

---

## Scraping

| | |
|---|---|
| Endpoint | `GET /metrics` |
| Port | the same port the API is served on (`port`, default `11437`) — there is no separate metrics listener |
| Format | Prometheus text exposition, `Content-Type: text/plain; version=0.0.4` |
| Auth | **required** — gated exactly like an admin route |

```bash
curl -s -H "Authorization: Bearer $ADMIN_KEY" http://localhost:11437/metrics
```

### `/metrics` is an admin route despite the URL

This surprises people, because most servers leave `/metrics` open. RustedVINO does not:
the endpoint discloses the loaded model catalogue, VRAM capacity and usage, per-model
request rates and per-key activity. Combined with the default permissive CORS policy
(`cors_allowed_origins: ["*"]`) an open `/metrics` is readable cross-origin from any web
page a browser on the network visits, so it is folded into the admin gate instead. The
path was kept at `/metrics` so existing scrape configs keep working.

| Code | Cause |
|---|---|
| `200` | Valid admin-scope key (or a keys file that exists with `admin_api_keys` explicitly empty, which delegates to the inference gate; if that file has no `api_keys` either, `/metrics` is served without any key to clients that send no `Origin` header) |
| `401` | No `Authorization` header, or an unrecognized Bearer key |
| `403` | A valid **inference-scope** key presented to a server that also has admin keys — authenticated but not authorized (`insufficient_scope`). Requires both `api_keys` and `admin_api_keys` to be non-empty; with `admin_api_keys` empty the route falls back to the inference gate and an inference key is admitted. Or `browser_origin_refused`: the keys file configures no keys at all and the request carries an `Origin` header (browsers always send one; curl and Prometheus do not) |
| `503` | Two distinct causes — see below |

Two different conditions produce `503`, and they are not interchangeable:

- **`admin_not_configured` — permanent.** No keys file was successfully loaded at boot. In
  this state *no* credential, valid or not, can reach `/metrics` or any `/v1/admin/*` route
  for the lifetime of the process. Creating the keys file afterwards requires a restart.
  This check runs in middleware, before the handler.
- **`metrics not enabled` — configuration.** The process is running without a model manager
  or without an installed Prometheus recorder, so there is nothing to render.

Configure `keys_file` before pointing a scraper at this server. See `keys_file` in
[`CONFIG.md`](CONFIG.md) and "Admin API status codes" in [`INTERFACE.md`](INTERFACE.md).

---

## How to read these metrics without being misled

Four traps, in descending order of how often they bite.

**1. An absent series means "never happened since boot", not zero.** The exporter renders a
series only after its first value. Several metrics below therefore do not appear at all
on a freshly-booted server: the four rejection and clamp counters, the four eviction-policy
counters, both `ov_cache_pruned_*` counters, both per-key metrics (until the first
authenticated request), `rustedvino_model_load_duration_seconds` (until a model's first load
completes) and all five `realtime_*` metrics. For alerting, treat absence with `absent()` — do not
write a rule that assumes `== 0` will match. `rustedvino_ov_cache_bytes` is a special case:
it is absent when `ov_cache_dir` is not configured, which means "no cache directory", not
"empty cache".

**2. A gauge reading `0` frequently means "not measured", not "measured as zero".** This
applies to:

| Gauge | What `0` can mean besides a real zero |
|---|---|
| `rustedvino_kv_cache_usage_percent` | Unmeasurable for this engine kind, or the engine is idle, or the model is unloaded |
| `rustedvino_kv_cache_usage_supported` | The model is unloaded (as well as: this engine kind cannot be queried) |
| `rustedvino_kv_cache_pressure_flagged` | The pressure monitor is disabled (the default) |
| `rustedvino_kv_cache_pool_gb` | The model is unloaded, or this kind has no KV pool at all |
| `rustedvino_requests_max` | The model is unloaded — not "zero capacity configured" |
| `rustedvino_tokens_per_second` | This kind never generates tokens (embedding/stt/tts/image_gen/reranking/decision) |

`rustedvino_model_pinned` and `rustedvino_model_priority` are the honest ones: `0` there
genuinely means unpinned / lowest priority.

**3. A non-zero value can be stale.** `rustedvino_tokens_per_second` is pushed from the
engine thread and nothing resets it when a model is evicted, so an unloaded model keeps
reporting its last throughput figure indefinitely. Always read it against
`rustedvino_model_loaded` for the same model. The same applies to
`rustedvino_model_load_duration_seconds`, which retains the last-known figure by design.
Counters and histograms are cumulative since process start and are likewise never reset by
an eviction or reload.

**4. The VRAM gauges carry two different label sets on one metric name.**
`rustedvino_vram_total_bytes` and `rustedvino_vram_used_bytes` are each emitted twice: once
labelled `device=` (the box-level figure for the primary inference device) and once per
memory domain labelled `domain=`. A bare `sum(rustedvino_vram_used_bytes)` double-counts.
Always filter on one label or the other. On a single-domain (UMA) box the two read
identically; on a multi-GPU box the `domain=` series are the per-GPU breakdown and the
`device=` series is the primary device only.

---

## Labels

| Label | Values | Notes |
|---|---|---|
| `model` | model ID as registered in `config.json` | |
| `device` | resolved OpenVINO device (`CPU`, `GPU`, `GPU.1`, `NPU`, …) | The model's *actual* silicon, not the global `device` config field |
| `kind` | `text_gen`, `vision`, `embedding`, `stt`, `tts`, `image_gen`, `reranking`, `decision` | Property of the loaded model. **`text_gen` covers two different engines** — the continuous-batching pipeline and the NPU static pipeline. The only label on the forced-eviction counters |
| `modality` | `text`, `multimodal` | Property of the *request*, not the model. Only a `kind="vision"` engine registers a `multimodal` series |
| `gate` | `cb`, `vlm`, `npu` | Which prompt-length gate rejected — the only way to tell NPU from CB, which share `kind="text_gen"` |
| `result` | see the realtime section | |
| `key_suffix` | last 6 characters of a Bearer key | Never the full key; keys shorter than 12 characters are not counted at all |
| `domain` | memory-domain id (a GPU name, or `system`) | VRAM gauges only |
| `mode` | `exact`, `legacy` | `rustedvino_kv_gate_mode` only |
| `reason` | `busy`, `incomplete` | Load and eviction refusal counters only |
| `driver`, `pdev`, `region` | DRM driver, PCI device, the driver's memory-region name | `rustedvino_process_gpu_memory_bytes` only |

---

## Model lifecycle and state

Pulled at scrape time — these gauges are recomputed from live server state on every request
to `/metrics`. All carry `model`, `device`, `kind`.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `rustedvino_model_loaded` | gauge | `model`, `device`, `kind` | `1` while the model is resident and `Ready`; `0` otherwise. A model that has been loaded at least once keeps its series after eviction, reading `0` — the series does not disappear |
| `rustedvino_model_load_duration_seconds` | gauge | `model`, `device`, `kind` | Wall-clock time of this model's **most recent successful load** — engine construction and JIT compile only, excluding admission checks and HTTP overhead. Retained after eviction (it describes a past event). Absent until the model's first load has ever completed, rather than reporting a misleading `0` |
| `rustedvino_model_pinned` | gauge | `model`, `device`, `kind` | `1` if the operator pinned the model so it is never chosen as an eviction victim. Static policy — survives eviction |
| `rustedvino_model_priority` | gauge | `model`, `device`, `kind` | Soft eviction-order weight from config; higher is evicted later. Static policy |

## Requests, capacity and throughput

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `rustedvino_requests_running` | gauge | `model`, `device`, `kind` | Requests that hold an engine slot and have not finished. Forced to `0` when the model is unloaded. A split SpeechT5 request holds one slot for all its pieces |
| `rustedvino_requests_max` | gauge | `model`, `device`, `kind` | The model's concurrency cap — the point at which further requests get `429`. Resolved per model (`max_concurrent_streams`, else `max_num_seqs`, else the engine's own default), so different kinds legitimately show different numbers. Reads `0` while unloaded |
| `rustedvino_requests_waiting` | gauge | `model`, `device`, `kind` | Callers parked at the admission gate, not yet running. This is genuine queue pressure: sustained non-zero with `requests_running == requests_max` means the model is the bottleneck. Accounting differs by engine: most maintain a dedicated waiting counter, but the reranking and image-generation engines derive it as in-flight-minus-one, so for those it is not an independent queue measurement |
| `rustedvino_requests_total` | counter | `model`, `device`, `kind`, `modality` | Requests **accepted by an engine**. Excludes anything rejected earlier (`404`, `429`, `503`, and the prompt gates below) — so this is not a request-rate metric for the HTTP surface, it is an engine admission count. A VLM splits into `modality="text"` and `modality="multimodal"` on the same engine. **SpeechT5 counts pieces, not HTTP requests:** since 0.9.1 a long `/v1/audio/speech` input is synthesized in pieces of at most 200 characters (the `X-RustedVINO-TTS-Segments` header gives the count), and each piece is one engine request here |
| `rustedvino_tokens_generated_total` | counter | `model`, `device`, `kind` | Output tokens produced. Incremented by the real per-step token count, so a speculative-decoding step that accepts several draft tokens counts all of them. Only the generation engines (`text_gen`, `vision`) ever increment this; other kinds register the series and leave it at `0` |
| `rustedvino_tokens_per_second` | gauge | `model`, `device`, `kind` | Output throughput; **what it measures depends on the engine.** Text models on a GPU or the CPU (the continuous-batching engine): an exponential moving average of aggregate decode throughput across the live batch, sampled per step — a batch-level figure, not per request, and prompt processing is not in it. Vision (`kind="vision"`) and NPU models: the **last finished request's** output tokens ÷ its end-to-end time, **prompt processing included** — so on a long prompt it reads well below the real generation speed (measured on an i7-11800H CPU: 4.9 tok/s for a 2,029-token prompt against 10.5 tok/s of actual generation). For prompt speed use `rustedvino_prefill_tokens_per_second`. **Not reset on eviction**; see trap 3 above |
| `rustedvino_prompt_tokens_total` | counter | `model`, `device`, `kind`, `modality` | Prompt tokens of finished generation requests (text, vision, NPU), the engine's own count — the same figure as `usage.prompt_tokens`. A request that ends in an error, or that its client stopped (disconnected, or not reading fast enough) before its end, is not counted, on every engine. A request the server stopped at its thinking budget (`finish_reason: "length"`, see INTERFACE.md "Thinking control") **is** a finished request and is counted, here and in the two series below. *(since 0.9.1)* |
| `rustedvino_reasoning_tokens_total` | counter | `model`, `device`, `kind`, `modality` | Reasoning tokens of the same finished requests — the same figure as `usage.completion_tokens_details.reasoning_tokens` (markers included). **Exact on text models, approximate on vision and NPU models** (a tokenizer recount of the reasoning text; see INTERFACE.md "The `usage` object"). Divide by `rustedvino_tokens_generated_total` for the thinking share. *(since 0.9.1)* |

### Rejection and clamp counters

Three of these count *clean, expected* rejections and one (`kv_max_tokens_clamped_total`)
counts a request that was served with a lowered `max_tokens`; none are faults. They are deliberately separate
metrics because each one calls for different operator action, and a single "rejected"
counter would tell you nothing about which. None of them appear until the first event.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `rustedvino_context_length_exceeded_total` | counter | `model`, `gate` | The prompt itself is longer than the model's ceiling — rejected with `400 context_length_exceeded` before reaching the engine. Action: the client must send a shorter prompt. The `gate` label is the only way to distinguish an NPU rejection from a continuous-batching one, since both are `kind="text_gen"` elsewhere |
| `rustedvino_pool_capacity_rejected_total` | counter | `model` | The prompt would fit on its own, but admitting it alongside requests already in flight would overcommit the KV pool. Action: **retry shortly** — this is transient contention, and a rising rate means the model needs a larger KV pool or lower concurrency |
| `rustedvino_kv_max_tokens_clamped_total` | counter | `model` | A vision request's `max_tokens` was lowered so the prompt and the answer fit the KV pool (exact gate only). The reply ends with `finish_reason: length` when the clamp binds. Action: none per request; a rising rate means clients ask for more output than the pool can hold alongside their prompts, so raise `kv_cache_gb` or shorten prompts |
| `rustedvino_own_budget_rejected_total` | counter | `model` | On a single-slot model, this request's own `prompt_tokens + max_tokens` cannot fit the KV pool, with nothing else in flight. Action: **do not retry unchanged** — an identical retry fails identically; the client needs a smaller `max_tokens`, or the model needs a bigger pool |

### Eviction policy

A model with a request running is never evicted by a plain load, unload, deregister or resize;
`force` overrides it. These counters show the policy at work. They do not appear until the
first event.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `rustedvino_load_refusals_total` | counter | `reason` | A load was refused instead of evicting a busy model (`reason="busy"`: the room it needs is held by models with requests running). A sustained rate means clients are keeping other models busy. Action: **retry when the request ends**, or load with `force` |
| `rustedvino_eviction_refusals_total` | counter | `reason` | An eviction did not happen. `reason="busy"`: the target has a request running (admin unload, deregister or resize without `force`). `reason="incomplete"`: a forced eviction's engine thread was still running its request at the 30 s join deadline — its memory is released when the thread exits |
| `rustedvino_forced_evictions_total` | counter | `kind` | An eviction went ahead although requests were running (`force`, or an automatic recovery). The log names the model |
| `rustedvino_eviction_aborted_requests_total` | counter | `kind` | Requests still running when a forced eviction's drain deadline passed, and aborted |

## Latency

Rendered as real Prometheus histograms (bucketed), not the exporter's default summary, so
they aggregate across replicas and work with `histogram_quantile()`. Each exposes the usual
`_bucket`, `_sum` and `_count` series.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `rustedvino_ttft_seconds` | histogram | `model`, `device`, `kind`, `modality` | Time to first token, measured from handler entry (so it includes admission-queue wait, not just prefill). *(since 0.9.1)* A request that waited for an on-demand load (`on_demand_wait_secs`) is measured from the end of that wait, as is its `rustedvino_request_duration_seconds`: both measure serving, not loading; the wait itself is in the INFO log line `on-demand wait: done` (`waited_ms`). Recorded only by the generation engines; other kinds register the series and leave `_count` at `0` |
| `rustedvino_request_duration_seconds` | histogram | `model`, `device`, `kind`, `modality` | End-to-end duration from handler entry to the final token. Recorded by **every** kind, including embeddings, reranking, image generation, STT and TTS — for those it is the only latency signal, since they have no first token. For SpeechT5 each piece of a split `/v1/audio/speech` request is one sample, timed from when that piece is sent to the engine |
| `rustedvino_prefill_tokens_per_second` | histogram | `model`, `device`, `kind`, `modality` | **Effective prefill (prefix-cache hits included)**: prompt tokens ÷ (first generated token − the engine's start of this request). Recorded once per request, only when the prompt has at least **256** tokens (`PREFILL_MIN_PROMPT_TOKENS`, a constant, not a config key); the two token counters above count every finished request. See "Reading the prefill histogram" below. *(since 0.9.1)* |
| `rustedvino_rtf_ratio` | histogram | `model`, `device`, `kind` | Real-time factor for audio: processing seconds ÷ audio seconds. Below `1.0` is faster than real time. Recorded only by STT and TTS engines, covering both the offline endpoints and the realtime voice pipeline (same engine thread either way). For SpeechT5 one sample per piece of a split request |

**Bucket edges (seconds, except RTF which is dimensionless):**

| Metric | Buckets |
|---|---|
| `rustedvino_ttft_seconds` | 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 2, 5, 10 |
| `rustedvino_request_duration_seconds` | 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120 |
| `rustedvino_realtime_stt_duration_seconds` | 0.05, 0.1, 0.2, 0.5, 1, 2, 5 |
| `rustedvino_realtime_llm_duration_seconds` | 0.5, 1, 2, 5, 10, 20, 30 |
| `rustedvino_rtf_ratio` | 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 5 |
| `rustedvino_prefill_tokens_per_second` (tokens/s) | 10, 25, 50, 100, 200, 350, 500, 750, 1000, 1500, 2000, 3000, 5000 |

The three 0.9.1 series carry the `modality` label as `rustedvino_ttft_seconds` does (a vision model splits text-only
and image requests, whose prefill differs a lot), at the cost of one more label value per vision model.

### Reading the prefill histogram

*(since 0.9.1)*

- **Effective, not raw, prefill.** OpenVINO GenAI 2026.4.1 reports no per-request prefix-cache hit count, so a
  follow-up request whose prompt prefix was cached divides all its prompt tokens by a short interval: a cache hit
  shows up as a high rate, not as a separate number.
- **Queue and admission time are left out.** The interval starts when the engine thread takes the request: on the
  text engine just before it is handed to the continuous-batching pipeline, on the vision and NPU engines when the
  engine starts the generation call. The admission-gate wait and the command-queue wait are before that point (they
  stay in `rustedvino_ttft_seconds`, which counts from handler entry).
- **Continuous batching (text engine).** The interval also holds the pipeline scheduler's own wait for room in the
  batch and the steps it spends on other requests (their decode, their prefill chunks) before this request's first
  token. Under load the rate therefore reads lower than the hardware's prefill speed. This is documented, not
  corrected.
- **Vision engine.** The interval holds image preprocessing, the vision encoder and the chat template (all run
  inside the one generation call), and the prompt count is the pipeline's own input-token count. Whether that count
  includes the image's visual tokens has not been verified on hardware. The first point is the first text chunk, which
  can hold more than one token.
- **NPU engine.** The prompt count is the tokenizer's count of the rendered prompt; the first point is the first
  delivered text chunk.
- A request whose first token never came (an error, a disconnect before it), or that its client stopped before its
  end, records no sample.

## VRAM

Read trap 4 above before writing a query against these.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `rustedvino_vram_total_bytes` | gauge | `device` **or** `domain` | Configured usable VRAM. `device=` is the box-level figure for the primary inference device (from `total_vram_gb`); `domain=` is the per-memory-domain budget, one series per domain |
| `rustedvino_vram_used_bytes` | gauge | `device` **or** `domain` | VRAM reserved across all loaded models of every kind — weights plus KV pool. This is RustedVINO's own accounting, not a driver query: separate OpenVINO Core instances cannot see each other's allocations, so the server tracks reservations itself |

Both are derived from the memory accounting's GiB figures (such as `total_vram_gb`, and each model's
`vram_gb` plus its KV pool) multiplied by 2^30 (1,073,741,824), so they are bytes of a GiB
budget. **Changed in 0.9.0:** 0.8.x multiplied by 10^9, so the same configuration now reads
about 7.4 % higher; rescale any dashboard or alert threshold carried over from 0.8.x.

### GPU context

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `rustedvino_gpu_poisoned` | gauge | none | `1` once an out-of-resources failure has poisoned the GPU context: loads and unloads are refused and GPU-placed models are reported `unavailable (gpu_poisoned)` until the server restarts; `0` otherwise. `GET /health` carries the same flag as `gpu_poisoned`. |

### Measured device memory (Linux)

The two gauges above are the server's *accounting* — what each model's configured `vram_gb`
reserves. These two are *measured* from the kernel (`/proc/self/fdinfo`, DRM drivers `xe`,
`i915`, `intel_vpu`). They are absent on other OSes.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `rustedvino_process_gpu_memory_bytes` | gauge | `driver`, `pdev`, `region` | Device memory this server process holds, read at scrape time. `pdev` is the PCI device, so two GPUs on one driver stay separate. `region` is the driver's own name, passed through verbatim: on `xe` for example `gtt` / `system` / `stolen` / `vram0`, on `i915` for example `system0` / `local0`, and `memory` for the NPU (`intel_vpu`). Real bytes, not converted from GiB figures |
| `rustedvino_model_gpu_memory_estimate_bytes` | gauge | `model`, `device`, `kind` | Per loaded model, with its whole KV pool: the measured change across its load, or the measured weights plus the booked pool when the pool was not resident at the end of the load (see below), plus runtime growth for embedding engines (the batch cache OpenVINO keeps until eviction). **NaN** while loaded but unknown; 0 once evicted |

The per-model figure is an **estimate** taken from the process-wide counter, so the server
only trusts a measurement taken while nothing else changed memory. It reports NaN (and
`gpu_memory_estimate_gb` is omitted) for the rest of a model's load when:

- its load overlapped another model's load or eviction;
- its load started within 20 s of an eviction or failed load — the kernel releases an evicted
  model's memory about 11 s *after* the eviction returns (measured on Lunar Lake);
- for an embedding model, any batch hit either of those conditions.

A reload in a quiet moment measures it again. *(Since 0.9.1)* a load that was not measured logs why at
debug level (`RUST_LOG=rustedvino::model_manager=debug`): `load memory not measured` with a `reason`
(`within_quiet_period_after_release` and `since_release_s`, `lifecycle_event_during_window`,
`lease_active_during_window`, `no_reading` or `negative_delta`). One exception logs nothing: a load that
returned only after the stalled-load reaper gave up waiting (log line `stalled load returned after N s — committed`)
is committed unmeasured, with no `not measured` line. **A model on the CPU holds no device memory**, so its
measurement reads about zero: the load logs `load memory measured … path="fallback" … delta_mib=0`, and
`GET /v1/admin/models` shows `gpu_memory_estimate_gb: 0.0` with no `vram_measured_mib`. That `0.0` is not its RAM
use (see the process memory gauges for that). Use the line to check `vram_gb`: when an estimate exceeds its model's `vram_gb` plus its KV pool (`kv_cache_gb`, which the
estimate includes), the accounting under-counts that model. The server also logs this once per load, and
`GET /v1/admin/models` shows it as `gpu_memory_estimate_gb`.

**The KV pool is not always resident when a load ends.** The one-token warm-up after a GPU load
(`kv_warmup`) is meant to allocate it, but when the warm-up is off, fails, or does not allocate it
on that machine, the pool is allocated at the model's first request. That was measured for a vision
pool on a B70: 7.4 GiB appeared in full at the first request. For a text pool it is **not verified
on hardware**; it is what the GenAI 2026.4.1 source does (both engines allocate their KV tensors at
the first scheduler step).

*(since 0.9.1)* So, for a vision model and a text model without a draft on a GPU (with or without `kv_warmup`), the
server also reads device memory after the engine is built and before the warm-up, and splits the
load in two: the weights (the change across construction) and the part of the pool that became
resident (the change across the warm-up, `kv_pool_resident_mib` in `GET /v1/admin/models`). When
less than 90 % of the pool became resident, the estimate is the measured weights plus the booked
pool, so it still includes the whole pool.

The split is not used (the load is measured as before: the estimate is the load's change, and
`kv_pool_resident_mib` is absent) when:

- the reading is missing: a draft-attached text engine, CPU and NPU, other kinds;
- the readings are not monotonic, or the warm-up grew more than twice the pool plus 1 GiB (noise,
  mostly on shared-memory boxes, where the reading is all of the process's GPU-driver memory);
- a text model's warm-up made less than 90 % of the pool resident while its construction took at
  least the pool (the construction figure might hold part of the pool). So a text model whose pool
  the warm-up did not touch (warm-up off or failed) keeps the old, low figure;
- a vision model's construction took at least its weight files on disk (the top-level `*.bin`
  files) plus 90 % of the pool: the pool was probably built at construction. This is a sanity
  bound and can misfire at the margins: runtime overhead on top of a small pool can cause a false
  fallback, and a submodel placed on another device makes the bound loose.

The server logs which way each load was measured: `load memory measured`, with `path` = `direct` or
`fallback` and, for a fallback, a `reason` (`no_mid_reading`, `missing_reading`, `no_pool`,
`not_monotonic`, `absurd_warm_up_growth`, `text_pool_not_seen_resident`,
`vision_above_disk_size`). It warns when a vision model's warm-up succeeded but its pool did not
appear during it (the pool may then have been built at construction, which would make the weights
figure include it; check on hardware).

Limit: when the pool appeared during the warm-up, the warm-up's own scratch memory (activations,
kernels compiled at the first generate) is counted with the pool. `kv_pool_resident_mib` is capped at
the booked pool, but the estimate, which is then the load's change, includes that scratch.

The same measurement gives `vram_measured_mib` in `GET /v1/admin/models`: the weights as measured
at the last load, MiB — the change across engine construction when the load was split, else the
load's device-memory change minus the KV pool it booked (which under-states the weights by any part
of the pool that was not resident yet). It is omitted under
the same conditions and also for every pool sized in whole GiB (`kv_pool_blocks` absent: the legacy
gate, or a model without an exact KV layout), whose real size is not known to the byte. A vision model that has admitted requests with images since its last
load carries `kv_admission_low_confidence: true` and `uncounted_image_admissions: <n>`: image tokens
are not counted by the context limit yet (the limit saw only the text, so an image-heavy prompt can
still overflow the pool); `rv ps` / `rv status` mark its CTX cell with a trailing `!`. Both fields
are absent until the first such request, and a new load clears them. A model unloaded and loaded again within 20 s therefore shows neither field;
wait and reload to get one. `server.headroom` of the same response (every caller of the route, keyless servers included) lists
each discrete GPU's total, what other processes hold, what this process holds, the running
leases and what is left for one prompt, with `probe` `fdinfo` or `none`; `rv status` prints it
under the VRAM bars.

## Process and host memory (Linux)

*(since 0.9.1)* Read from the kernel at scrape time (`/proc/self/status`, `/proc/meminfo`), in bytes (the kernel's
kB figures times 1024). No labels. Each is absent (not `0`) where the OS cannot measure it: on
Windows and other non-Linux hosts none of the four is exported, so alert with `absent()`.
Limit: a gauge keeps its last value if a later read fails (only if `/proc` becomes unreadable
mid-run); the exporter cannot remove a single series.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `rustedvino_process_resident_memory_bytes` | gauge | none | Resident memory of the server process (`VmRSS`): anonymous plus file-backed pages, such as model files the CPU or NPU plugin mapped. Includes pages the kernel can drop and re-read, so it can exceed what the process really costs the host |
| `rustedvino_process_rss_anon_bytes` | gauge | none | Anonymous resident memory of the server process (`RssAnon`): the part that leaves `MemAvailable`. The figure the system-RAM admission measures loads and CPU requests with |
| `rustedvino_host_memory_available_bytes` | gauge | none | Host memory available without swapping (`MemAvailable`: free plus reclaimable). The live figure the system-RAM admission floor checks a load against |
| `rustedvino_host_memory_total_bytes` | gauge | none | Total host memory (`MemTotal`) |

Under `--supervise` the process gauges describe the worker that serves `/metrics`, not the
supervisor. On a UMA box (iGPU, NPU) device memory comes out of the same RAM: compare
`rustedvino_process_gpu_memory_bytes` with these.

## KV cache

`rustedvino_kv_cache_usage_percent`, `_usage_supported` and `_pressure_flagged` only make sense read together. Reading
`rustedvino_kv_cache_usage_percent` alone is the single easiest way to draw a wrong
conclusion from this endpoint.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `rustedvino_kv_cache_pool_gb` | gauge | `model`, `device`, `kind` | KV-cache pool reserved for this model at load time, in GiB (the metric name keeps its historical `_gb` suffix). Follows the live reservation: `0` once the model is evicted, and `0` for kinds that have no pool |
| `rustedvino_kv_cache_usage_percent` | gauge | `model`, `device`, `kind` | Live pool occupancy, 0–100, as of the engine's last step. **Only meaningful when `rustedvino_kv_cache_usage_supported == 1`** |
| `rustedvino_kv_cache_usage_supported` | gauge | `model`, `device`, `kind` | `1` if the percent gauge above is a trustworthy live reading for this model; `0` if it is structurally unqueryable and therefore indistinguishable from a genuinely empty pool |
| `rustedvino_kv_cache_pressure_flagged` | gauge | `model`, `device`, `kind` | `1` while this model's occupancy has stayed continuously at or above `kv_pressure_threshold_pct` for at least `kv_pressure_sustained_secs`. Detect-and-flag only — nothing is evicted or resized as a result |

`rv ps` and `rv status` follow this rule: their `KV%` column shows `n/a` when `_usage_supported` is `0` (the VLM
engine today) and the percent otherwise. A text (CB) model reads `0%` while idle; its gauge moves only during a
generation. Both read the series of the device the model is on now (the server keeps a moved model's old
device's series with their last values). `GET /v1/admin/models` (and so `rv ps --json`) carries the raw percent as `kv_cache_usage_pct` beside the same flag as
`kv_cache_usage_supported` (bool).

### Admission

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `rustedvino_kv_gate_mode` | gauge | `model`, `mode` | `1` for the prompt-length gate mode a loaded GPU/CPU text or vision model uses (NPU models have no KV gate and no series), `0` once it is unloaded. `mode="exact"` = the limit comes from the model's real KV layout and exact pool, `mode="legacy"` = the older estimate (a model reads 1 on exactly one of the two). Labels are `model` and `mode` only (no `device`/`kind`), so join with `on(model)`. |
| `rustedvino_kv_scheduler_drop_total` | counter | `model` | Requests the continuous-batching scheduler ignored because the KV pool had no room for them (the client gets a retryable 503). Text (continuous-batching) and vision models on any device, including a vision model on the NPU (a vision model's KV capacity wedge, `kv_capacity_wedge`, counts too); NPU text models and other kinds have no series. A rising count means the prompt-length gate admits more than the pool holds; see `kv_cache_precision` in `CONFIG.md`. It also counts a request whose answer filled the pool after it started generating (admission reserves the prompt only), which can happen with a correct gate; the server log says which ("KV pool filled from output growth"). |

### `supported = 0` is not "the pool is empty"

There are three distinct reasons the percent gauge reads `0`, and only the first is a real
measurement:

1. **A genuine zero** — a queryable engine that is idle or holding nothing.
2. **No KV pool exists.** Embedding, TTS, STT, image-generation and reranking models have no
   KV cache at all. `0%` here is not a health signal in either direction.
3. **A real pool that cannot be read.** A model on the vision/VLM path allocates a genuine KV
   pool, but OpenVINO exposes no public API to query its occupancy. So does a model routed to
   the **NPU**, which reports `kind="text_gen"` like the continuous-batching engine but runs a
   static pipeline with no batched pool to query — `kind="text_gen"` alone does **not** imply
   the reading is real. A `0%` here is silence, not an empty pool. This is the trap:
   a dashboard showing a flat `0%` on a VLM under heavy load looks healthy and is telling you
   nothing.

`supported` also reads `0` for any model that is not currently loaded, whatever its kind.

### When the percent gauge updates

It is published from inside the engine's step loop, so it moves only while work is in
flight. Two consequences:

- When the engine goes idle it is **explicitly clamped to `0`**, rather than holding the last
  step's value. A brief `0` between requests is expected and is not a measurement of an empty
  pool.
- If the underlying read fails, the **previous value is kept** rather than being zeroed, so
  the gauge does not flap to `0` on a transient failure. A value that stops moving is
  therefore ambiguous between "steady load" and "reads are failing".

It also lags admission: occupancy has been observed to stay at `0.0` for over a second after
a large request starts, because no step has run yet. It is an observability signal, not an
admission-control signal.

### `pressure_flagged` reads 0 for two different reasons

The pressure monitor is **optional and off by default**
(`kv_pressure_monitor_enabled: false`, see [`CONFIG.md`](CONFIG.md)). The gauge reads `0`
both when there is no pressure and when the monitor is disabled, and those two states are
**indistinguishable from this gauge alone**. Before treating a flat `0` as "no pressure",
confirm the monitor is enabled and that at least one resident model has
`rustedvino_kv_cache_usage_supported == 1` — the monitor can only observe models whose pool
is actually queryable, so on a box serving only VLM, embedding and audio models it observes
nothing at any threshold.

Enabling the monitor requires setting `kv_pressure_threshold_pct` and
`kv_pressure_sustained_secs` explicitly; the server refuses to boot otherwise.

## OpenVINO compile cache

Disk usage of `ov_cache_dir`, OpenVINO's compiled-blob and kernel cache. Unlabelled — these
are per-directory, not per-model. Pushed from a background sweep rather than computed at
scrape time (walking the directory on every scrape would be far too expensive), so the value
is static between sweeps and up to `ov_cache_sweep_interval_secs` old.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `rustedvino_ov_cache_bytes` | gauge | — | Total on-disk size of `ov_cache_dir` as of the last sweep, root plus its `kv-*` directories. Reported whenever `ov_cache_dir` is configured, whether or not `ov_cache_max_gb` bounds it. **Absent entirely when `ov_cache_dir` is unset** |
| `rustedvino_ov_cache_pruned_bytes_total` | counter | — | Cumulative bytes reclaimed by pruning since process start. Only moves when `ov_cache_max_gb` is set *and* a sweep found the cache over budget — permanently absent on an unbounded cache |
| `rustedvino_ov_cache_pruned_files_total` | counter | — | Cumulative cache files deleted by pruning. Same gating |

A steadily rising `rustedvino_ov_cache_bytes` with both pruned counters absent means the
cache is unbounded and growing — set `ov_cache_max_gb` if that matters on your box.

## Per-key attribution

Two metrics, because the counter alone answers "which key has the largest cumulative total
since boot", not "which key is active right now". Pair them to tell a busy key from a
formerly-busy one.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `rustedvino_key_usage_total` | counter | `key_suffix` | Requests attributed to one Bearer key. Cumulative since process start, not a rolling window |
| `rustedvino_key_last_seen_timestamp_seconds` | gauge | `key_suffix` | Unix timestamp of the most recent request attributed to that key |

Safety contract, which also bounds what these can tell you:

- The label is only the key's **last 6 characters**, never the key itself. A key shorter than
  12 characters is skipped entirely — its last 6 characters would be most of the secret — so
  short keys produce no series at all rather than a truncated one.
- Only **successfully authenticated** requests are counted. Rejected `401`/`403` attempts are
  deliberately not recorded: counting arbitrary presented tokens would turn a bounded label
  cardinality (one series per configured key) into an attacker-controlled unbounded one.
  These metrics are activity attribution, **not** a failed-auth or intrusion signal.
- When no keys are configured at all the server is open and nothing is recorded.
- Two configured keys sharing the same last 6 characters merge into one series.

## Realtime voice

Emitted by the `GET /v1/realtime` WebSocket pipeline. Unlabelled by model — these describe
the pipeline, not one engine. None of them appear until the first realtime connection.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `rustedvino_realtime_connections_total` | counter | — | WebSocket connections accepted on `/v1/realtime`. Connections, not turns — one connection carries many turns |
| `rustedvino_realtime_turns_total` | counter | `result` | Completed voice turns by outcome. See the `result` values below |
| `rustedvino_realtime_stt_duration_seconds` | histogram | — | Whisper latency per turn. The timer starts when the captured audio is encoded to WAV and stops at the finished transcript, so it includes any wait for an STT engine slot, not just inference |
| `rustedvino_realtime_llm_duration_seconds` | histogram | — | LLM decode latency per turn, from engine admission to the final token. The voice pipeline waits for the complete response, not for stream start, so this spans the whole reply |
| `rustedvino_realtime_tts_sentences_total` | counter | `result` | Per-sentence synthesis attempts — `ok` or `dropped`. TTS is driven sentence-by-sentence so speech can start before the LLM finishes; a rising `dropped` rate means synthesis is not keeping up with generation, or turns are being interrupted |

`rustedvino_realtime_turns_total` `result` values:

| Value | Meaning |
|---|---|
| `ok` | Turn completed normally |
| `stt_error` | Transcription failed |
| `llm_error` | Generation failed |
| `cancelled` | Turn cancelled before completion |
| `barge_in` | The transcript contained the session's barge-in phrase, and the server answered with its stop reply (see [REALTIME.md](REALTIME.md) "Barge-in phrase turn") — normal use, not an error |
| `admission_rejected` | The turn could not get its device admission (the [`device_budgets`](CONFIG.md) caps) |

A skipped turn (empty or too-short transcript) and a turn not started because the client
hung up count under no `result`.

Only `stt_error` and `llm_error` are faults. `barge_in` and `cancelled` are expected in
normal use, and alerting on total turns minus `ok` will fire constantly.
