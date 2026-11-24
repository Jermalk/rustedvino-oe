# RustedVINO — Interface Reference

Full route table for the running server. For prose (why it exists, what's implemented,
setup steps, config field reference) see **[`README.md`](README.md)**.

---

## Route table

### OpenAI-compatible endpoints

Drop-in replacements for the real OpenAI API surface — same request/response shape, same
error envelope. A client written against OpenAI's SDK works against these unmodified as long as it
keeps to the fields this server supports: the parameters listed under
[Not supported, or different from OpenAI](#not-supported-or-different-from-openai) are refused with a
`400` or behave differently.

| Method | Route | Description |
|---|---|---|
| `GET` | `/v1/models` | The models a client can use: those **loaded and ready**, and *(since 0.9.1)* the `load: "on_demand"` models that are not loaded yet. An eager model that is not loaded, and a model being evicted, are not listed (the full registry is `GET /v1/admin/models`). A ready model's entry carries a non-OpenAI field `max_model_len` (the model's prompt-token limit, as vLLM reports it; omitted when not known) and no `state`. An on-demand model that is not loaded carries a non-OpenAI field `state` *(since 0.9.1)*: `"not_loaded"`, or `"loading"` while its load runs, and no `max_model_len` (the limit is known only after the load). A request to it on `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/rerank`, `/v1/systemone`, the image routes or the audio routes starts the load and answers `503 model_unavailable` with `Retry-After` until the model is ready (or waits for it, see "On-demand models: waiting for the load"). `/tokenize` and `/detokenize` never start a load: they answer `503 model_unavailable` until the model is loaded some other way. While the GPU context is poisoned, each GPU-placed model's entry (and `GET /v1/models/{id}`) carries `state: "unavailable (gpu_poisoned)"` instead, on-demand ones included, since no model can load on that GPU until a restart. A client that read every entry as loaded should check that `state` is absent |
| `GET` | `/v1/models/{id}` | One model (OpenAI retrieve-model), the same entry as in the list; 404 `model_not_found` for a model the list does not show (an unknown id, or *(since 0.9.1)* a registered model that is not listed, whose message says it is registered but not loaded) |
| `POST` | `/v1/chat/completions` | Chat completions — streaming (`stream:true`) + non-streaming; VLM image input **or** text-only |
| `POST` | `/v1/completions` | Legacy raw-prompt completions |
| `POST` | `/v1/embeddings` | Text embeddings |
| `POST` | `/v1/audio/transcriptions` | Speech-to-text (Whisper) |
| `POST` | `/v1/audio/translations` | Speech in any language → English text (Whisper) |
| `POST` | `/v1/audio/speech` | Text-to-speech (Coqui VITS, Piper voices, `SpeechT5`; Kokoro-82M loads but is not supported in 0.9.1: its English speech is garbled, see README) |
| `POST` | `/v1/images/generations` | Text-to-image (SDXL, FLUX.1-schnell) |
| `POST` | `/v1/images/edits` | Image inpainting + img2img (SDXL) |

Within these routes, a few request fields are **RustedVINO extensions**, not part of OpenAI's
spec — same convention other OpenAI-compatible servers (vLLM, Ollama, llama.cpp) use: `top_k`,
`repetition_penalty` (when absent the model's sampling profile applies: the operator's per-model
`sampling`, else the model's own `generation_config.json` value above `1.0`, else a `1.1` safety net;
a client opts out with `1.0`),
`enable_thinking` (also read from vLLM's `chat_template_kwargs.enable_thinking` on
`/v1/chat/completions`; the top-level field wins when both are set), and on `/v1/chat/completions`
*(since 0.9.1)* `thinking_budget_tokens` (alias `thinking_token_budget`) and
`chat_template_kwargs.preserve_thinking`; `reasoning_effort` is OpenAI's field, with
RustedVINO's meaning. See [Thinking control](#thinking-control-reasoning_effort-thinking_budget_tokens-since-091).
Other `chat_template_kwargs` keys are ignored. (`max_prompt_len` is a per-model, NPU-only **config** field, not a request
field — see [`CONFIG.md`](CONFIG.md).) Sampling fields a request leaves out (`temperature`, `top_p`,
`top_k`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`) come from the model's
**sampling profile** (CONFIG.md "Sampling profile"); values the client sends always win, and a request
greedy because of tools or its own `temperature: 0` gets no configured `presence_penalty` /
`frequency_penalty` (it keeps the `repetition_penalty` chain).

#### Not supported, or different from OpenAI

- **Refused with `400 unsupported_parameter`** (`param` names the field), rather than silently
  ignored: on `/v1/chat/completions`, `n` above 1, `logprobs: true`, `top_logprobs`, a non-empty
  `logit_bias`; on `/v1/completions`, `n` or `best_of` above 1, `logprobs`, a non-empty
  `logit_bias`; on `/v1/embeddings`, `dimensions` (any value).
- **Images** (`/v1/images/generations`, `/v1/images/edits`): `response_format: "url"` is a
  `400 invalid_response_format` (the server hosts no image URLs; `b64_json` only); `size` must be one
  of ten fixed values; `n` is 1–4 and limited by a pixel budget. Details:
  [below](#post-v1imagesgenerations-and-post-v1imagesedits).
- **Speech** (`/v1/audio/speech`): `response_format` is `wav` (the default, where OpenAI's is
  `mp3`) or `pcm`; `mp3`, `opus`, `aac` and `flac` are a `400 invalid_response_format`.
- **Transcriptions and translations:** `response_format` is `json`, `text`, `srt` or
  `verbose_json` (no `vtt`); `prompt` and `temperature` are accepted and ignored.
- **Chat images:** only `data:` URIs; `http(s)` image URLs are not fetched (see
  [Chat images](#chat-images-vision-models)).
- **`/v1/models`** lists loaded models and *(since 0.9.1)* on-demand models that are not loaded yet,
  with a non-OpenAI `state` field on the latter (see the route table).

### Ollama-compatible endpoints

Same request/response shape and error envelope as Ollama v0.35 (`{"error": "<string>"}`), so
Ollama clients and TypeSafe's System One SDK work unmodified.

| Method | Route | Description |
|---|---|---|
| `POST` | `/v1/systemone` | Typed decisions (choice / yes-no / ordinal score) with probabilities from one prefill per question — no text generation. Needs a `kind: "decision"` model (see [`CONFIG.md`](CONFIG.md)) |

### RustedVINO extensions

Not part of OpenAI's API — server-specific infra, admin/lifecycle control, and endpoints
borrowed from other providers' conventions (rerank: Cohere/Jina-style; tokenize/detokenize:
common in vLLM-style servers). `/v1/realtime` borrows OpenAI's naming for the concept but is a
**bespoke protocol** (barge-in, session history/retrieval, voice-pin), not a re-implementation
of OpenAI's actual Realtime API event schema.

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — returns `{status, version, git_hash, gpu_poisoned, device_stalled, startup_warnings}` (always 200; `startup_warnings` *(since 0.9.1)* = `{count, codes}`: how many startup warnings the server has and their distinct codes, e.g. `{"count": 1, "codes": ["preload_skipped"]}`, never their text, since `/health` is unauthenticated — the messages are in `GET /v1/admin/models` `server.startup_warnings` and `rv status`; `device_stalled: true` = at least one model load is booked as stalled (past `load_timeout_secs`, or its client went away while it ran) and has not returned yet; that device takes no new loads until it does. Details: `GET /v1/admin/health/watchdog`). Auth-exempt (only path that is) |
| `GET` | `/health_generate` | Deep readiness probe — runs a 1-token greedy generation through a Ready text or vision model (on any device; if none is Ready, another Ready model, whose kind may answer 503 "deep-health probe not implemented for … models"). 200 `{status: "ok", model, elapsed_ms}`; 503 `{status: "error", message}` when no model is ready or the probe fails or takes longer than 10 s; 503 with an extra `error: "gpu_poisoned"` field while the GPU context is poisoned (plus `Retry-After: 30` while a supervised restart is pending). Not auth-exempt: needs an API key like the other inference routes |
| `GET` | `/metrics` | Prometheus metrics (tokens/s EMA, TTFT, request duration; device memory; process and host memory on Linux *(since 0.9.1)*: `rustedvino_process_resident_memory_bytes`, `rustedvino_process_rss_anon_bytes`, `rustedvino_host_memory_available_bytes`, `rustedvino_host_memory_total_bytes`, absent where the OS cannot measure them) — full reference in [`METRICS.md`](METRICS.md). Gated exactly like an admin route despite the URL |
| `POST` | `/v1/rerank` | Cross-encoder document reranking. `documents`: 1 to 1000 per request *(since 0.9.1)*; more is a `400 too_many_documents` (`param: "documents"`) naming the limit and the count |
| `POST` | `/tokenize` | Encode text to token IDs via the loaded model's tokenizer (text and vision models); never loads a model, `load: "on_demand"` ones included — see [`POST /tokenize` and `POST /detokenize`](#post-tokenize-and-post-detokenize) |
| `POST` | `/detokenize` | Decode token IDs back to text; never loads a model, like `/tokenize` |
| `GET` | `/v1/realtime` | WebSocket upgrade — realtime STT → LLM → TTS voice flow (bespoke protocol, full reference in [`REALTIME.md`](REALTIME.md)) |
| `GET` | `/v1/admin/models` | List all registered models + state (`ready`/`not_loaded`/`loading`/`stalled`/`evicting`/`unavailable (gpu_poisoned)`; `stalled` = a load past `load_timeout_secs` or whose client went away, still running — see CONFIG.md `load_timeout_secs` — or *(since 0.9.1)* a CPU/NPU model whose engine stalled (no progress within `stream_first_token_timeout_secs` / `stream_token_gap_timeout_secs`), refused until a restart — its stuck call keeps its memory booked, so an unload leaves it `stalled` while that call runs; `unavailable (gpu_poisoned)` = a GPU-placed model while the GPU context is poisoned: listed, and no model loads on the GPU until a restart; after a generation stall the stalled model is refused while other GPU models already loaded keep serving, see CONFIG.md `stream_first_token_timeout_secs`). Each model entry carries `id`, `state`, `kind`, `vram_gb`, `device`, `max_prompt_tokens` (the prompt-token limit, the `max_model_len` of `/v1/models`), `native_context_limit` (the context the model was trained for, from its `config.json` `max_position_embeddings`; omitted when the file does not declare it), `cache_size_gb` (this model's own booked KV pool in GiB, `0.0` when not loaded; not the global `cache_size_gb` cap of the same name), `kv_pressure_flagged` (bool; always `false` while `kv_pressure_monitor_enabled` is off, see CONFIG.md) and `gpu_memory_estimate_gb` (the measured device memory of a loaded model in GiB, see METRICS.md "Measured device memory"; omitted when not loaded or not measurable). Each model entry also reports the resolved KV-cache decision: `kv_cache_precision` (`u4`/`i8`/`f16`, or `auto` when the plugin decided), `kv_precision_source` (who decided it), `kv_bytes_per_token`, `kv_profile_confidence` (`high`/`medium`/`low`), `kv_gate_mode` (`exact`/`legacy`: how `max_prompt_tokens` was computed), `kv_pool_mib` (`0` when not loaded), `kv_pool_blocks`, `kv_requested_gb`, `clamped_by` / `limit_clamped_by` (why the pool, or the context limit, is smaller than asked), `kv_headroom_cap_tokens` (the device-headroom limit of a vision model on a discrete GPU), `vram_measured_mib` (weights as measured at the last load: the change across engine construction, or the load's change minus the booked pool when that split is not available), `kv_pool_resident_mib` *(since 0.9.1)* (how much of the booked pool became resident during the last load's warm-up, MiB; absent when the load was not split that way — see METRICS.md "Measured device memory"), `kv_admission_low_confidence` + `uncounted_image_admissions` (vision requests with images admitted since the last load without their image tokens counted), a `sampling` object *(since 0.9.1)* for every `text_gen` and `vision` model, in every state (the effective sampling profile, see CONFIG.md "Sampling profile": `{"mode": "greedy"|"sample", "temperature": {"value", "source"}, "top_p": {…}, "top_k": {…}, "repetition_penalty": {…}, "presence_penalty": {…}, "frequency_penalty": {…}, "tools": "greedy"}`, `source` one of `config`, `generation_config.json`, `engine default`, `server default`, `none`, `value` `null` when unset; plus `ignored_on_device: ["presence_penalty", "frequency_penalty"]` on an NPU and a `note` when the model's `generation_config.json` is an exporter stub; omitted, not `null`, for other kinds; the older `default_temperature`/`default_top_p`/`default_top_k` keys stay, present only after the first load), `thinking_budget_tokens` *(since 0.9.1)* (the model's configured thinking budget, see CONFIG.md
"Thinking budget"; always present, `null` when unset), and `kv_cache_usage_pct` (live KV-pool
occupancy, 0–100) with `kv_cache_usage_supported` *(since 0.9.1)* (bool, always present: `false` while the model is not loaded and for an engine kind whose occupancy cannot be read, the VLM engine today, where `kv_cache_usage_pct` is a permanent `0` and does not mean an empty pool; same flag as `rustedvino_kv_cache_usage_supported`); optional fields are omitted, not `null`, when unknown or not applicable. The `server` block carries `headroom` for every caller (`{mode, probe, margin_gib, devices: [{device, total_gib, outside_mib, own_gib, leases_mib, free_gib}]}`: the free device memory of each discrete GPU; `devices` is empty with no discrete GPU or when `kv_headroom_mode` is not `dynamic`, `outside_mib` is absent when the probe cannot read it, and `leases_mib` goes only to key-validated callers). It also carries, for key-validated callers, `startup_warnings` (`[{code, message}]`: configuration problems found at boot that did not stop it — e.g. `global_device_unknown`, `ov_cache_dir_unwritable`; *(since 0.9.1)* `models_dir_missing`: `models_dir` does not exist, is not a directory or cannot be read, and `models_dir_empty`: it holds no model directory (no subdirectory whose name does not start with `.`); both name the absolute path checked and, for a relative `models_dir`, the working directory it resolved against (a relative `models_dir` resolves against the server's working directory, not the config file's directory); no model can be served until it is fixed, but the server boots; both are cleared once a model is loaded; *(since 0.9.1)* `restore_failed`: a model loaded before a worker crash that the respawned worker could not restore, cleared once the model is loaded; *(since 0.9.1)* `preload_skipped`: a `preload` model that did not fit beside the preload models loaded before it (a preload never evicts another preload), naming the binding gate (memory budget or system RAM), its need and the shortfall in GB, cleared once the model is loaded; and *(since 0.9.1)* `respawn_preload_skipped`: one warning naming the `preload` models a respawn after a worker crash did not load because they were not loaded at the crash (it stays until a restart, even once those models are loaded); omitted when none) and `platform` (`{drivers: [{device, name, driver_version}], kernel}`: the GPU driver (OpenCL `CL_DRIVER_VERSION`, e.g. the NEO version) and NPU driver each device runs on — RustedVINO uses whatever driver the machine has)|
| `GET` | `/v1/admin/models/completeness` | Report which registered models are missing required directory/tokenizer files |
| `POST` | `/v1/admin/models/add` | Register a new model at runtime and load it immediately; persists to `config.json` — `model_id`/`vram_gb` required, `kind` and every `ModelPolicy` field (device, co-residency knobs, image-gen provenance, …) optional, same shape as a static config stanza. Success is 200 with a plain-text message. When the model loaded but writing `config.json` failed it is still 200, with a JSON body `{"message", "persisted": false, "persist_error"}`: the model serves now and is gone after a restart, so a script must check for `persisted: false` |
| `POST` | `/v1/admin/models/{id}/load` | Load model into VRAM (blocks until ready; evicts idle least-recently-used models if needed). Optional JSON body: `kv_cache_gb` (this load's KV-pool target in GiB, above 0; omit it for the model's configured size), `max_concurrent_streams` (this load's stream cap, at least 1) and `force`; an out-of-range value, an unknown field or malformed JSON is a 400 (plain text). The per-load overrides are ignored when the model is already loaded: use `resize` for that. A model with a request running is never evicted to make room: the load is refused with `503 model_busy` (+ `Retry-After`) unless the body carries `"force": true`, which also bypasses the eviction-grace window (the busy model is drained for up to 15 s, then what is left is aborted) |
| `POST` | `/v1/admin/models/{id}/check` | Dry-run admission check: does it fit, which models would be evicted. `?force=true` previews a load that bypasses the eviction-grace window and the busy rule (never `pinned`, non-`evictable` or a live realtime session's models). Response fields include `would_evict`, `busy_victims` (members of `would_evict` with a request running — non-empty only for a forced check) `blocked_by_busy` (models left alone only because a request is running on them, whose eviction would have made the load fit), and for vision models on a discrete GPU `headroom_refusal` (the free-device-memory check refuses even after every eviction) and `headroom_not_previewed` (the model was not loaded on this server yet, so its own device-memory need is unknown); both omitted when empty. It checks memory and evictions only: not the model's files nor its `chat_template` override, so the load itself can still refuse a model the preview admits |
| `POST` | `/v1/admin/models/{id}/resize` | Resize a **resident** model's KV-cache pool — evicts (draining in-flight requests) then reloads at the new size. Body `{"kv_cache_gb": <f64 > 0>, "force": <bool, default false>}`; a model with a request running is refused with `409 model_busy` unless `force` is `true`, and a resize whose reload could not be admitted is refused before anything is evicted; every other resolved override (`max_concurrent_streams`, …) is carried forward from the model's record, so the caller states only the delta. Deliberately **not** idempotent-when-`Ready` — that is exactly why it exists, since a repeated `/load` silently ignores a new `kv_cache_gb` on an already-loaded model |
| `DELETE` | `/v1/admin/models/{id}` | Evict model from VRAM. A model with a request running is refused with `409 model_busy` (+ `Retry-After`); `?force=true` drains it (up to 2 minutes), aborts what is left, then joins the engine thread. For every engine except the continuous-batching text engine a forced eviction waits up to 30 s for the engine thread to finish its request and otherwise answers `503 eviction_incomplete` (the memory is released when the thread exits). A running split SpeechT5 request does not hold a forced eviction up: it ends after its current piece with a retryable `503 model_unavailable`. With `Accept: application/json` the 200 is a JSON report of what the eviction did *(since 0.9.1)* ([below](#what-an-eviction-did-the-json-answer-accept-applicationjson-since-091)); unload, deregister, resize and load alike |
| `PATCH` | `/v1/admin/models/{id}` | Update fields of an already-registered model — persisted always, live immediately for most fields, next-load for `vram_gb`/`kv_cache_gb`/`kv_cache_precision`/`max_concurrent_streams`/`enable_prefix_caching`/`max_prompt_len`/`min_response_len`/`speculative` (`kv_cache_precision: null` clears the per-model value); `device`/`tier_preference` need the model `NotLoaded` first. Not patchable: `kind` (immutable, a 400), `chat_template` and `load_timeout_secs` (not fields of a PATCH body: an unknown-field 422; edit `config.json`, then unload → `config/reload` → load). `thinking_budget_tokens` *(since 0.9.1)* is live, a `Ready` model included (`null` clears it; a negative or fractional value is a JSON error: `400 invalid_body` *(since 0.9.1; 422 before)*). `sampling` is live, a `Ready` model included (the next request uses it): the object is **replaced whole**, not merged, and `"sampling": null` clears it; a value out of range, or `top_p`/`top_k` without a `temperature` on a model whose `generation_config.json` lacks `"do_sample": true` with a valid temperature, is a 400; an unknown key inside `sampling` (`temprature`) is an unknown field, 422, as for any unknown field of a PATCH body (through `POST /v1/admin/models/add` it is a `400 invalid_body` *(since 0.9.1; 422 before)*, like any malformed body there) |
| `DELETE` | `/v1/admin/models/{id}/register` | Deregister a model entirely: removed from the registry **and from `config.json`** (the whole stanza), so it does not come back after a restart or a config reload; register it again with `POST /v1/admin/models/add`. Evicts it first if loaded: a model with a request running is refused with `409 model_busy` unless `?force=true` |
| `POST` | `/v1/admin/config/reload` | Diff `config.json` against the live registry — add new entries, refresh changed fields, skip disruptive ones. Response: `added`, `updated` (`NotLoaded` entries refreshed), `skipped_missing_files`, `left_untouched` (`Ready`/`Loading`/`Evicting`: never changed, except `sampling` on a `Ready` model, see below), `unchanged` (count), `globals_not_applied` (global keys whose file value differs from the booted one), and `models_not_applied` *(since 0.9.1)*: `[{"model": "<id>", "fields": ["chat_template", ...]}]`, the `left_untouched` models whose stanza changed and which keys. Those edits are **not applied**, not even at the model's next load: unload it, reload the config again (it is then `updated`), then load it (`rv reload-config` prints this). Empty on the common path. One exception: a changed `sampling` or *(since 0.9.1)* `thinking_budget_tokens` on a `Ready` model **is** applied (both are read per request) and listed in `applied_live` *(since 0.9.1)* instead, same shape (`[{"model": "<id>", "fields": ["sampling"]}]`, always present, empty on the common path); on a `Loading`/`Evicting` model it stays in `models_not_applied` |
| `POST` | `/v1/admin/config/preload` | Replace the persisted `preload` list — `{"model_ids": [...]}` explicitly, or `{"from_live": true}` to snapshot the currently-`Ready` set |
| `POST` | `/v1/admin/keys/reload` | Hot-reload the keys file (`keys_file`) — rotate/add/revoke Bearer keys with no restart |
| `GET` | `/v1/admin/config/audit` | Report drift between `config.json` and the live model registry |
| `GET` | `/v1/admin/logs/tail` | Tail the server's in-memory log ring buffer (`?lines=N`, default 200, clamped to the buffer's fixed capacity — not a file). Response is `text/plain`, one log line per line, oldest first (not JSON) |
| `GET` | `/v1/admin/health/watchdog` | `oldest_generation_secs` (worst in-flight generation or OpenVINO call past its allowance; polled by `--supervise` to detect a wedged worker `/health` can't see), `oldest_ffi_overdue_secs` (loads, transcriptions, CB steps only), `in_flight` (OpenVINO calls running), `stalled` (`[{device, model, kind, secs, recovery}]` loads past their deadline; `recovery` = what to do if it never returns, `null` while the load is inside its own deadline: the GPU reset command on a discrete xe GPU with a known PCI address, otherwise advice for that kind of device) |
| `GET` | `/v1/admin/voice-pin` | Current voice-flow pin (shared STT/LLM/TTS model set for realtime sessions) |
| `GET` | `/v1/admin/realtime/sessions` | List all live realtime WebSocket sessions |
| `GET` | `/v1/admin/realtime/sessions/{id}` | Full snapshot of one realtime session |
| `DELETE` | `/v1/admin/realtime/sessions/{id}` | Force-close a live realtime session (WebSocket close code 4000 `closed_by_admin`) |

---

## Request limits and endpoint behaviour

### Request size (every route)

A request body may be at most **50 MiB**. That is a fixed server limit, not a config field. A
larger request gets `413` with error code `request_too_large`, whether it is a JSON body (a chat
request with a large inline image, say) or a multipart upload (audio, images).

### Chat images (vision models)

Only `data:` URIs are accepted (`http(s)` image URLs are not fetched), and only the last message's
images reach the model. Each image's size is read from its header before anything waits *(since 0.9.1)*: over
`max_images_per_request` is `400 too_many_images`, over `max_image_pixels_per_request` (summed over
the request, original sizes) is `400 image_too_large`, and a payload that is not an image is a
`400`. The image is decoded only once the request has its engine slot (so a request that queues —
up to the device admission and engine queue timeouts, then `429 device_at_capacity` or `429
server_overloaded` — holds its compressed
bytes, not decoded pixels); an image whose header is valid but whose body is corrupt is therefore a
`400 failed to decode image` that can come after that wait. An image with more than
`max_image_pixels` pixels (default 4 000 000) is downscaled *(since 0.9.1)*, aspect ratio kept, before the model
sees it: the model works on (and counts tokens for) the smaller image. See CONFIG.md.

### Error responses

Rejections use the OpenAI error envelope, `{"error": {"message", "type", "param", "code"}}`, with a
stable `code`: for example `invalid_body` (malformed or incomplete JSON), `invalid_multipart` (a
multipart route called without a multipart body), `invalid_audio` (an upload that cannot be decoded
as audio), `invalid_image` / `invalid_mask` (400: an `/v1/images/edits` `image` or `mask` part that cannot be decoded), `invalid_query` (a bad query string such as `?force=maybe`), `model_busy` (409 or 503, with `Retry-After`: a model with a request running was not evicted — an unload, deregister or resize refused with 409, or a load refused with 503 because the only room is held by such models; retry when the request ends, or repeat the call with `force`), `eviction_incomplete` (503, with `Retry-After`: a forced eviction's engine is still finishing its request, so its memory is not free yet), `load_timeout` (504: a model load did not return within `load_timeout_secs`; it is booked as stalled and committed or discarded when it returns), `generation_stalled` *(since 0.9.1)* (504 with `Retry-After: 30` on a non-streaming request, or the `code` of the one SSE error frame a streaming request gets before `[DONE]`: the engine the request was queued on or running in made no progress within `stream_first_token_timeout_secs` of a call's start, or within `stream_token_gap_timeout_secs` of its previous progress; every request on that engine ends this way, and the model is then marked dead — a GPU model answers `503 gpu_poisoned` until a restart, a CPU or NPU model answers `503 model_unavailable` ("stopped making progress") until a restart, also to a load or unload while its stuck call runs; see CONFIG.md), `device_stalled` (503, with `Retry-After`: a stalled load on the model's device blocks new loads there until it returns; models already loaded keep serving), `server_stopping` (503, no `Retry-After`: the server is stopping, so a model load is refused: an admin load, add or resize, or the on-demand load of an inference request), `config_file_invalid` (409: `models/add` or a deregister refused before anything changed, because `config.json` is already invalid; not "already registered"), `template_rejected` (400: a vision model's chat template raised an error on these messages; the message carries the template's reason), `invalid_chat_template_kwargs` (400: `chat_template_kwargs.enable_thinking` or *(since 0.9.1)*
`.preserve_thinking` is not a boolean; `param` names the key), *(since 0.9.1)*
`invalid_reasoning_effort` (400: `reasoning_effort`, top level or in `chat_template_kwargs`, is not
one of `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`; `param` names where it was), *(since 0.9.1)*
`invalid_thinking_budget` (400: `thinking_budget_tokens` / `thinking_token_budget` is not a whole
number from 0 to 4294967295), *(since 0.9.1)* `conflicting_thinking_budget` (400: both
`thinking_budget_tokens` and `thinking_token_budget` set, to different values), `size_exceeds_cpu_limit` (400: an image request for more pixels than `cpu_image_gen_max_pixels` allows on the CPU), `frame_serialize_failed` (in an SSE error frame: a stream frame could not be serialised), `method_not_allowed` (405, with the `Allow` header), `context_length_exceeded` (a prompt
that can never fit this model: retrying is pointless) and `pool_capacity_exceeded` (429: the prompt fits
alone but not beside the requests already running: retry shortly), `own_budget_exceeded` (429: on a `max_concurrent_streams: 1` model, the prompt plus `max_tokens` exceeds the KV pool even with nothing else running; reduce `max_tokens`, since retrying unchanged fails the same way), `server_overloaded` (429: every inference slot of the model stayed busy for `admission_queue_timeout_ms`; retry shortly), `device_at_capacity` (429: the device's `device_budgets` gate stayed busy for `device_admission_queue_timeout_ms`; retry shortly), `invalid_value` (a value out of
range, or a chat message whose `role` is not `system`, `developer`, `user`, `assistant`, `tool` or
`function`; `param` names the field, for example `messages[1].role`) and `unknown_url` (404, a path
the server does not have, as OpenAI answers it). Routes added by an extension answer as the extension defines them.

More codes the inference routes return: `model_not_found` (404: the id is not registered, or on `/v1/models/{id}` a registered model that the list does not show; *(since 0.9.1)* the message of an unknown id says it was not found among the registered models, and on `/v1/models/{id}` a registered model's says it is registered but not loaded), `model_unavailable` (503: the model is not loaded — *(since 0.9.1)* the message says it is registered but not loaded and that an administrator can load it or set it to load on demand — or is loading or being evicted — then with `Retry-After` — or *(since 0.9.1)* a CPU/NPU model that stopped making progress), `engine_unavailable` (503: the model's engine thread has exited), `unsupported_parameter` (400: an OpenAI parameter this server refuses rather than ignores, see [Not supported, or different from OpenAI](#not-supported-or-different-from-openai)), `incompatible_parameters` (400: `response_format` together with a forcing `tool_choice`, `"required"` or a named function), `unsupported_content` (400: image input to a text-only model), `invalid_tool_choice` (400), `invalid_max_tokens` (400: `max_tokens` below 1), `empty_messages` (400: chat with no messages), `empty_prompt`, `too_many_prompts` (400: a `/v1/completions` prompt array longer than `max_prompt_array`), `too_many_documents` *(since 0.9.1)* (400: more than 1000 `/v1/rerank` documents), `invalid_language` *(since 0.9.1)* (400: an audio `language` Whisper has no code for) and `multi_prompt_stream` (400: a prompt array with `stream: true`) on `/v1/completions`, `invalid_encoding_format` (400, embeddings), `invalid_response_format` (400: a `response_format` the route does not offer), `invalid_speed` (400, speech), and on the image routes `invalid_size`, `invalid_steps`, `invalid_n`, `invalid_guidance_scale`, `invalid_strength`, `multi_image_unsupported` and `n_exceeds_pixel_budget` (all 400).

### Streaming frames and token counts (`stream: true`)

Count tokens from `usage.completion_tokens` (send `stream_options: {"include_usage": true}`), not
from the number of SSE chunks: the two differ by design. Measured on the text engine (qwen3-1.7b,
CPU, 0.9.0):

- `/v1/chat/completions` sends one role-only frame (`delta.role`, no text) first and one
  finish-only frame (`finish_reason`, empty `delta`) last, before the usage frame and `data: [DONE]`.
  In between, one content chunk per generated token when the model does not think (5 tokens →
  5 chunks). With thinking on, the answer goes to `delta.reasoning_content` and the `<think>`
  marker token produces no chunk (64 tokens → 63 reasoning chunks).
- `/v1/completions` sends one text chunk per token (32 → 32, 400 → 400) and one finish-only frame.
- The end-of-sequence token is not sent as a chunk, and (inferred from the equal counts above) not counted in `completion_tokens`.
- A vision model's streamer can deliver several tokens in one callback, so it can send fewer
  chunks than tokens; `usage` counts the tokens either way.
- A mid-stream failure is an error frame (`{"error": {...}}` with a `code`) followed by a
  finish-only frame with `finish_reason: "error"` and `[DONE]`; the HTTP status stays 200.
  The frame's `type` is always `server_error`, and its `code` is one of `pool_exhausted` (the
  model is momentarily at capacity: retry shortly — except when the request was alone on the model
  and its answer filled the KV cache: then the message names the generated tokens and says that
  retrying the same request fails the same way, so lower `max_tokens` or shorten the prompt; a
  non-streaming request gets the same 503 `pool_exhausted` with that message), `kv_capacity_wedge` (a vision model is
  recovering from a KV-cache capacity failure: retry shortly), `model_unavailable` (the GPU
  context is poisoned; the message says whether a supervised restart is pending — a frame cannot
  carry `Retry-After`), `model_unloading` (the model was unloaded while the request ran: retry
  shortly), `insufficient_memory`, `template_rejected`, `mid_stream_error` (any other engine
  failure) or `frame_serialize_failed`.

### On-demand models: waiting for the load *(since 0.9.1)*

A request to a `load: "on_demand"` model that is not loaded starts its load on every inference
route: `/v1/chat/completions`, `/v1/embeddings`, `/v1/rerank`, `/v1/systemone`,
`/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/audio/translations`,
`/v1/images/generations`, `/v1/images/edits`, and *(since 0.9.1)* `/v1/completions` (before
0.9.1 it answered the plain not-loaded `503` and loaded nothing). `/tokenize` and `/detokenize`
never start a load and never wait, at any `on_demand_wait_secs`: a model that is not loaded gets
the plain not-loaded `503 model_unavailable` (no `Retry-After`), as in 0.9.0, so a token count
can never evict a loaded model. What a request on the other routes gets depends on the server's
`on_demand_wait_secs` (CONFIG.md):

- **`0` (shipped):** `503 model_unavailable` + `Retry-After: 5` at once ("model is currently
  loading"); retry until the model is ready. `/v1/systemone` answers the same in Ollama's shape.
- **Above `0`:** the request waits for the load, at most that many seconds (and at most the
  model's own `load_timeout_secs` minus 2 s), and then:
  - the model is ready: the request is served normally (it queues for the model like any other);
  - the load failed: the load's error, as the route answers it (for example `504 load_timeout`,
    `503 insufficient_memory`); a failure with no error code is `500 server_error` with the
    message "model '…' failed to load; see the server log";
  - the wait ran out: `503 model_unavailable` + `Retry-After: 5`, as with `0`; the load goes on;
  - the model was loaded and unloaded again before the request got in: its load is started
    again and the request gets that `503` + `Retry-After` (no second wait);
  - the model was removed from the registry while the request waited: the route's `404`;
  - the server is stopping: `503 server_stopping`.

  A request whose model's load is already past its `load_timeout_secs` (booked as stalled), or
  whose model would have to load on a device with a stalled load, does not wait: it gets
  `504 load_timeout` or `503 device_stalled` at once, the answer the waiting requests got when
  the load stalled. A waiting request is woken when a stalled load finally returns.
- **Streaming** (`stream: true` on chat and completions): the `200` with `text/event-stream` is
  sent at once, then an SSE comment line (`:`) every 15 s while the request waits. Once the model
  is ready the normal stream follows. An error after the `200` (a failed load, the wait running
  out, a model-specific `400` such as a prompt over the model's limit) comes as one
  `data: {"error": {...}}` frame with the code the non-streaming request would get, then
  `data: [DONE]`; `Retry-After` cannot travel in the stream.
- **Busy server:** at most 64 requests wait at once, across all models. Past that a request gets
  the immediate `503` + `Retry-After`, as with `0` (a streaming one as a plain HTTP `503`).
- A client that disconnects while it waits frees its place; the load is not cancelled (other
  requests may be waiting for it).
- Eager models (`load: "eager"`) never wait and are never loaded by a request: plain `503`, no
  `Retry-After`.
- An on-demand model of a kind the route does not serve (an embedding model on `/tokenize`, a
  text model on `/v1/embeddings`, ...) is refused with the route's wrong-kind answer before any
  load starts, with any `on_demand_wait_secs`, when the model's kind is known: set by `kind` in
  the config, found from the model's files when it was registered, or learned from an earlier
  load. The answer is a `400` (on chat, a `501 not_implemented` for the kinds chat does not serve
  yet). Otherwise (no `kind` and no recognisable files at registration, for example files copied
  in after the server started) the request starts the load as before, and the load decides
  (`/tokenize` and `/detokenize` answer the not-loaded `503` instead). An NPU text model is known
  as such only once loaded, so `/v1/completions` still loads it before refusing it.
- `/v1/realtime` is unchanged: a session loads the model it needs itself and reports the load in
  its own events; `on_demand_wait_secs` does not apply.

### The `usage` object (chat and legacy completions)

Every `usage` object on `/v1/chat/completions` and `/v1/completions` (buffered body, or the streamed
`include_usage` chunk) carries, since 0.9.1:

```json
"usage": {"prompt_tokens": 1312, "completion_tokens": 740, "total_tokens": 2052,
          "completion_tokens_details": {"reasoning_tokens": 512}}
```

- `completion_tokens_details.reasoning_tokens` is always present when `usage` is (0, not absent, when
  thinking is off or the model has no reasoning), and never larger than `completion_tokens`.
- It counts the tokens of the reasoning block, markers included (`<think>` and `</think>`; for gpt-oss the
  leading `analysis`/`commentary` channel word and the `assistantfinal` transition): the tokens the reasoning parser
  routes to `reasoning_content`. For a model with no reasoning parser the same `<think>…</think>` rule counts a raw
  block the model emits (streamed, those tokens go to `reasoning_content`; a buffered reply keeps them in `content`,
  the count is the same).
- **On gpt-oss, a reply that is only a tool call counts entirely as reasoning**: it opens on a channel word and never
  reaches an `assistantfinal` transition, so every token is reasoning up to the end, and `reasoning_tokens` equals
  `completion_tokens` (measured: 35 of 35, 36 of 36). The tool call itself is returned as usual.
- **Exact on text models, per text delta**: the continuous-batching engine reports how many tokens each text delta
  carries, and a delta counts where its text lands. A token that decodes to no text (a skipped special token such as
  gpt-oss's `<|channel|>` or `<|message|>`, or a byte held back until a character is complete) is reported with the
  next delta that has text, and counts where that delta lands. So does every token of a multi-token delta (a
  speculative-decoding step accepting several tokens, or those folded tokens). Such a delta can cross the end of the
  block; it then counts wholly as reasoning, and a debug log line says the split was approximate.
- **Approximate on vision and NPU models**: their streamer delivers text chunks without per-chunk token counts, so
  after the generation the engine re-encodes the reasoning text with the model's own tokenizer and clamps it to the
  exact generated total. Re-encoding text is not the same as the generated ids, so it can differ by a few tokens. If
  the tokenizer fails, the total is split by bytes instead (the server logs a WARN).
- `/v1/completions` runs no reasoning parser (the text comes back raw); it counts the raw `<think>…</think>` rule,
  from the first token when the client's own prompt ends in an open `<think>`. With several prompts, the per-prompt
  counts are summed.
- `prompt_tokens_details.cached_tokens` is not reported: OpenVINO GenAI 2026.4.1 gives no per-request prefix-cache
  hit count, and no estimate is sent. The matching `/metrics` series (`rustedvino_prompt_tokens_total`,
  `rustedvino_reasoning_tokens_total`, `rustedvino_prefill_tokens_per_second`) are described in METRICS.md.

### Thinking control (`reasoning_effort`, `thinking_budget_tokens`, since 0.9.1)

On `/v1/chat/completions`, for models that think (a `<think>` block, or gpt-oss's analysis
channel):

| Request field | Values | Effect |
|---|---|---|
| `reasoning_effort` (also `chat_template_kwargs.reasoning_effort`; the top level wins) | `none`, `minimal` | Thinking off |
| | `low`, `medium`, `high` (`xhigh` and `max` are accepted as `high`) | Thinking on, with a budget of 1,024 / 4,096 / 16,384 tokens |
| `thinking_budget_tokens` (alias `thinking_token_budget`) | whole number ≥ 0 | `0`: thinking off. Above 0: a limit on thinking (turns it on, unless another field turns it off) |
| `enable_thinking` (also `chat_template_kwargs.enable_thinking`) | `true` / `false` | The template's thinking switch, no budget |
| `chat_template_kwargs.preserve_thinking` | `true` / `false` | Passed to templates that can keep earlier turns' reasoning |

Values are exact and lowercase; anything else is a 400 naming the field and listing the accepted
values (`invalid_reasoning_effort`,
`invalid_thinking_budget`, `conflicting_thinking_budget` when both budget names are set to
different values, `invalid_chat_template_kwargs`).

**Which one wins.** An explicit off always wins: `enable_thinking: false`, `reasoning_effort`
`none` or `minimal`, or a budget of `0` turns thinking off whatever else the request sends.
Otherwise any explicit on (`enable_thinking: true`, `low`/`medium`/`high`, a budget above 0) turns
it on; otherwise the model's `thinking_budget_tokens` of `0` turns it off; otherwise the template's
default applies. When thinking is not off, the budget comes from the first of: the request budget,
the `reasoning_effort` preset, the model's `thinking_budget_tokens` (a default, which the request
can raise or lower); otherwise there is none. So `thinking_budget_tokens: 2000` with
`enable_thinking: false` does not think: a budget only limits thinking that is on. The full table
is in CONFIG.md ["Thinking budget"](CONFIG.md#thinking-budget-thinking_budget_tokens).

**The budget is strict.** The server counts the reasoning tokens as they are generated. When the
reasoning reaches the budget before the model has closed it, the request stops there:

- `finish_reason` is `"length"`, streaming and non-streaming;
- `content` is empty (the model had not started its answer) and `reasoning_content` holds the
  reasoning so far; there is no "thinking was cut off" text. On a model with no `reasoning_parser`
  configured the reply is not split: the raw text, `<think>` included, stays in `content`;
- no tool call is parsed from it, even if a call-shaped line appears inside the reasoning;
- `usage.completion_tokens_details.reasoning_tokens` is reported as usual.

A client that gets `length` with an empty `content` should retry with a larger budget, or with
thinking off. The stop happens only at a point where the text is whole (never inside a UTF-8
character); when the text at the budget ends in the start of the closing marker (`</th`), the
stop waits one more piece of text, so a model closing right at the budget is not cut. Only the
first reasoning block counts: once the model closes it, the budget no longer applies and the
answer runs to `max_tokens`. A request that runs out of `max_tokens` while still inside the
reasoning with a budget set gets the same response.

- **gpt-oss: only the analysis channel is budgeted.** Counting ends when the model switches to the
  `commentary` channel (a tool call), and an output that starts in `commentary` is not counted,
  so the budget never cuts a tool call. A tool call that runs out of `max_tokens` is a plain
  `length`, and the call is parsed as before. `usage.completion_tokens_details.reasoning_tokens`
  still counts the commentary channel too, so on gpt-oss it can be larger than the budget.
- **The budget leaves room for an answer.** It is lowered to `max_tokens` minus the answer
  reserve when it is larger; the reserve is 512 tokens, or half of `max_tokens` when that is less
  (`max_tokens: 400` keeps 200 for thinking and 200 for the answer). This is exact on text models;
  on vision and NPU models it is approximate, like the count below.
- **Exact on text models, approximate on vision and NPU models.** The continuous-batching text
  engine counts each token as it comes, so the stop lands on the budget (or a few tokens past it,
  when one step yields several tokens). Vision and NPU models deliver text in chunks without token
  counts, so the server counts chunks and stops after the budget divided by the engine's
  measured tokens per chunk (vision 1.37, NPU 1.04; one measurement each, so approximate until
  measured on more models). Where a model's real ratio differs, the stop moves in proportion: a
  vision model at 1.0 tokens per chunk stops about 27 % early, one at 1.6 about 17 % late. Their
  reported `reasoning_tokens` is the recount described in
  [The `usage` object](#the-usage-object-chat-and-legacy-completions), so it can differ from the
  budget by that much.
- **A literal `<think>` counts as reasoning.** The budget uses the same rule as the reasoning
  split: an answer that writes `<think>` as text (a model explaining its own tags, say) starts a
  reasoning block there, and without a `</think>` after it the request is stopped at the budget.
- **Templates.** `enable_thinking` and `preserve_thinking` reach the chat template as variables of
  those names. `reasoning_effort` reaches it too: the client's word when thinking is on at that
  effort (gpt-oss reads it natively; `xhigh` and `max` arrive as `"high"`), `"none"` when thinking
  is off. A template that has no
  thinking switch ignores them; the budget still applies to any `<think>` block the model emits,
  but "off" (any of the off values above) cannot stop such a model from thinking: it only sets
  the template switch, and no budget is armed.
  These are the only `chat_template_kwargs` keys read, and the template gets the server's own
  values for them, never the request's JSON.
- **Metrics.** A request the budget stopped counts as a finished request (its prompt, reasoning
  and prefill series are recorded), unlike a request whose client went away.
- **Other routes.** `/v1/completions` has no thinking control: its prompt is the client's own text,
  so these fields are not read there and the model's `thinking_budget_tokens` does not apply. The
  realtime voice route (`/v1/realtime`) always turns thinking off, so no budget applies.

### `POST /v1/audio/speech`

- **Fields:** `model` and `input` (required; at most 4096 characters, a `400` above that, but
  see the per-engine limits below), `voice`, `response_format`
  (`wav` default, or `pcm`) and `speed` (0.25-4.0), as in OpenAI's API, plus two extensions:
  - `language` (optional, e.g. `en`, `pl`, `vi`; a region suffix such as `pl-PL` is fine): the
    language of `input`. Omit it and the server uses the language the model speaks (SpeechT5 and
    Kokoro: English, the Polish VITS voice: Polish, a Piper voice: its eSpeak language).
  - `normalize` (optional, default `true`): speak digits and symbols (`%`, `&`, `$`, ...) as
    words before synthesis. Speech models drop or garble raw digits ("Cena wzrosła o 15%" came
    back as "Cena wzrosła o"). Send `false` if the text is already speakable. Either way, Unicode
    space separators (no-break U+00A0, narrow no-break U+202F, thin U+2009, figure U+2007 and the
    other `Zs` spaces) are turned into a plain space before synthesis.
- **What normalisation covers:** English, Polish and Vietnamese number words; decimals, negative
  numbers and thousands separators; English years read in pairs ("nineteen ninety"), except a
  4-digit number before a currency word or a plural noun ("1250 dollars", "1500 people"), which is a
  quantity ("one thousand two hundred fifty") unless a year preposition precedes it ("in 1990 tensions
  rose"). Polish
  additionally gets scale-word agreement ("dwa tysiące", "pięć tysięcy"), `zł` / `kg` / `km`
  agreeing with the number ("trzy złote"), and a four-digit year followed by "roku" or "r."
  read as an ordinal in the right case ("w 1948 roku" → "w tysiąc dziewięćset czterdziestym
  ósmym roku"; "od 1948 roku" → "… czterdziestego ósmego roku"). Polish does not decline
  quantities by case or gender ("dwa", never "dwie"), and a bare "w 1948" with no "roku" stays
  a plain number.
- **An unsupported `language`** (e.g. `de`) leaves the text untouched rather than reading the
  digits as English.
- The same normaliser runs in the realtime flow.
- **Input length per engine.** 4096 characters is the request cap. Some models can't speak
  that much in one pass:
  - **SpeechT5: the server splits long input** *(since 0.9.1)*. Its tokenizer works per character
    and the model has 600 text positions: in one pass, everything past about 600 characters is
    dropped without an error, and quality breaks down from about 250–450 characters. So the
    server cuts the text into pieces of at most 200 characters, synthesizes them in order, and
    returns the joined audio (same format and sample rate; no silence is added, the model's own
    pauses separate the pieces). The cut comes after normalisation (spelled-out digits make the
    text longer):
    - at sentence ends (`.` `!` `?` `…`); after a plain `.` the next word must not start in
      lowercase, so "e.g. the" stays together. Whole sentences are packed into one piece while
      they fit;
    - a sentence over 200 characters is cut at its last clause break that leaves at least 50
      characters on the left: `,` `;` `:` followed by whitespace, or an em or en dash (`—` `–`),
      except an unspaced dash between two letters or digits (`10–20`, `well—known`);
    - else at the last line break with at least 50 characters on its left, else at the last space,
      else at any clause break, and as a last resort hard at 200 characters. A hard cut lands on a
      character (`char`) boundary, not a grapheme boundary, so a combining mark or an emoji sequence
      can be split there; it only happens in 200 characters without a space.

    Text of 200 characters or less is one piece: the text handed to the engine is unchanged (the
    audio can still differ slightly between server processes). Every successful SpeechT5 response
    carries two headers: `X-RustedVINO-TTS-Segments: <n>` (the number of pieces, `1` when the
    text fit) and `X-RustedVINO-TTS-Segment-Cap: 200`. Error responses and other engines don't
    send them. Both are listed in `Access-Control-Expose-Headers`, so a browser client on another
    origin can read them. The request is admitted once and holds one TTS slot until its last piece
    is done; a failing piece ends it with the error and no audio. An eviction of the model (a forced
    unload, a forced load of another model, a resize) ends a running split request after the current
    piece with a retryable `503 model_unavailable` (+ `Retry-After`) and no audio.

    **At most 50 pieces per request** *(since 0.9.1)*. Input that splits into more than 50 pieces
    after normalisation (about 10,000 characters, about 8 minutes of speech) is refused with a
    `400`, before anything is queued or synthesized:
    `{"error": {"type": "invalid_request_error", "code": "input_too_long", "message": "input is too
    long for this model: it splits into 125 pieces after normalisation, at most 50 pieces of 200
    characters are accepted per request"}}` (the message names the actual count). Exactly 50 pieces
    is accepted. The limit is fixed, not a config key. It is what stops the worst case: normalisation
    reads a long run of digits digit by digit, so 4096 digits become about 24,600 characters and
    125 pieces, an estimated 15–20 minutes of synthesis on a laptop CPU, holding one TTS slot the
    whole time. Split such text into several requests. Other engines have no piece limit (only the
    4096-character cap). The realtime flow speaks the reply one sentence at a time and does not
    apply this split or this limit.
  - **Polish VITS (`pl-vits-mai-female`) and Piper voices (`vi-vais1000-piper`):** tested
    complete up to about 3,300 characters (about 170 s of audio), with the duration staying
    proportional to the text. Piper synthesises sentence by sentence; the Polish voice reads the
    whole input in one pass.
  - Kokoro: not measured. **Kokoro-82M is not supported in 0.9.1:** it loads, but its English speech is
    garbled (see README); the Kokoro details below describe how the engine is wired, not a tested path.
- **`voice`:** chooses one of Kokoro's built-in voices (default `af_heart`). An unknown name
  falls back to `af_heart` with a server-side warning, not a `400`. SpeechT5, the Polish VITS
  voice and Piper voices have one voice per model and ignore `voice`, whatever its value.
- **`speed`:** applies to Kokoro, the Polish VITS voice and Piper voices. **SpeechT5 ignores it.**
- **Output sample rate** (mono, 16-bit for `wav`; matters when you join audio from different
  voices): SpeechT5 16 kHz, Kokoro 24 kHz, the Polish VITS voice 22.05 kHz, a Piper voice the
  rate in its `.onnx.json` (22.05 kHz for `vi-vais1000-piper`).

### `POST /v1/audio/transcriptions`

- **Fields:**
  - `file` (required): WAV, MP3, FLAC, Ogg Vorbis, AAC, or M4A/MP4 with AAC audio. Not decoded:
    ALAC *(since 0.9.1)*, Opus (also in an Ogg file), WebM/Matroska. The source sample rate must be
    4–48 kHz *(since 0.9.1)*. The server decodes the file and resamples it to 16 kHz mono itself.
  - `model` (required).
  - `language` (optional, e.g. `en`, `pl`). Recommended: detection from a few seconds of audio
    is unreliable. *(since 0.9.1)* Case and a region suffix do not matter (`PL`, `pl-PL`, `en_US`
    are read as `pl`, `en`); `auto` or an empty value means "detect", like no field. A value that is
    not one of Whisper's language codes (`xx`, a language name such as `polish`) is a `400`
    `invalid_language` with `param: "language"` (0.9.0 answered 500).
  - `response_format`: `json` (default), `text`, `srt` or `verbose_json`.
  - Other OpenAI fields (`prompt`, `temperature`, …) are accepted and ignored.
- **Undecodable audio** (garbage, truncated, an unsupported container) returns `400` with code
  `invalid_audio`.
- **Long audio:** one request transcribes the whole file, including audio longer than Whisper's
  30-second window. For example, an 87.6-second file came back complete, with `verbose_json`
  segments to the end. Two limits apply: the 50 MiB upload cap (`413 request_too_large`; about
  27 minutes of 16 kHz mono 16-bit WAV) and `max_audio_duration_secs` *(since 0.9.1)* (default 7200 s;
  longer audio is a `400 invalid_audio` naming the limit), so a compressed file over 2 hours is
  refused even when it is under 50 MiB.

### `POST /v1/audio/translations`

- **Output is always English**, whatever language is spoken. That is what Whisper's translate
  task does, and what OpenAI's endpoint promises. For any other target language, transcribe and
  then translate with `/v1/chat/completions`.
- **Same request and response format as transcriptions**, and the same limits (50 MiB,
  `max_audio_duration_secs`, the accepted formats). `language`
  isn't in OpenAI's spec for this route; RustedVINO accepts it as a hint for the *spoken*
  language.
- **The model has to be trained for translation.** An English-only Whisper model (`.en`) is
  refused with `400` `model_cannot_translate`. `whisper-large-v3-turbo` was fine-tuned for
  transcription only: it accepts the request but returns text in the spoken language, not
  English, and the server can't detect that from the model's files. Measured on one Polish clip:

  | Model | Result |
  |---|---|
  | `whisper-large-v3` | Correct English translation |
  | `whisper-tiny` | English, but a loose paraphrase |
  | `whisper-large-v3-turbo` | Polish text, untranslated |

### `POST /v1/images/generations` and `POST /v1/images/edits`

- **`/v1/images/generations` fields** (JSON): `model`, `prompt` (required, not blank), `n` (1–4,
  default 1), `size`, `response_format` (`b64_json`, the default; `url` is a
  `400 invalid_response_format`). `quality`, `style` and `user` are accepted and ignored.
  Extensions: `steps` (denoising steps, 1–50, default 20; a distilled model such as FLUX.1-schnell
  wants about 4), `seed` (absent: the pipeline's default),
  `guidance_scale` (0–20; `0` or absent: the pipeline's own default; FLUX ignores it) and
  `negative_prompt` (FLUX ignores it).
- **`size`:** one of `256x256`, `512x512`, `512x768`, `768x512`, `768x1024`, `1024x768`,
  `1024x1024` (the default), `1024x1536`, `1536x1024`, `1536x864`; any other value is a
  `400 invalid_size`.
- **Pixel budget:** `n` × width × height may be at most 2,097,152 pixels (two 1024×1024 images),
  else `400 n_exceeds_pixel_budget`, whose message names the largest `n` that fits. On the CPU,
  `cpu_image_gen_max_pixels` applies as well (`400 size_exceeds_cpu_limit`, see CONFIG.md).
- **`/v1/images/edits`** (multipart): `model`, `prompt`, `image` (required; one image: a second
  `image` / `image[]` part is a `400 multi_image_unsupported`), `mask` (optional; with it the request
  is an inpaint, without it an img2img), `n` (1–4), `strength` (above 0, at most 1; absent: the
  pipeline default; lower keeps more of the source), `seed` (absent: the pipeline's default),
  `response_format` (`b64_json` only). The output has the source image's size; `size`, `quality`,
  `background` and `user` are accepted and ignored. The same pixel budget applies to the source
  size × `n`. The image and the mask are checked against `max_image_pixels_per_request` *(since
  0.9.1)* (`400 image_too_large`) before they are decoded; an undecodable one is a
  `400 invalid_image` / `invalid_mask`.
- **Response:** `{created, elapsed_ms, data: [{b64_json}], generation_metadata}`: the images are
  PNG; `elapsed_ms` and `generation_metadata` (`model_family`, `device`, `host` for a valid-key caller, `model_hash`, the sampler, `openvino_version` and
  the provenance fields `precision` / `model_source` / `model_revision` from the model's config) are
  extensions. `elapsed_ms` is the generation's time; *(since 0.9.1)* it does not include a wait for
  an on-demand load.

### `POST /v1/embeddings`

- **Input length:** each input may be at most the model's own maximum (for example 512 tokens for
  multilingual-e5, including special tokens). A longer input gets `400`
  `context_length_exceeded`; nothing is truncated.
- **Inputs per request:** at most `max_embedding_inputs` (default 256), else `400`
  `too_many_inputs`.
- **Batch size:** at most `max_embedding_batch_tokens` padded tokens per request (default 32768;
  inputs × the longest input's token count), else `400` `batch_too_large`. See
  [`CONFIG.md`](CONFIG.md).

### `POST /tokenize` and `POST /detokenize`

Direct access to a loaded model's own tokenizer. Use it to size a prompt before sending it (the
usable context of a large model can be far below the advertised one) and to turn token ids back
into text without generating. Not part of OpenAI's API; similar to vLLM's routes (the `/detokenize` reply is `{"text": ...}`, vLLM's is `{"prompt": ...}`).

| Route | Request | Response |
|---|---|---|
| `/tokenize` | `{"model": "<id>", "prompt": "<text>"}` | `{"tokens": [...], "count": N}` |
| `/detokenize` | `{"model": "<id>", "tokens": [...]}` | `{"text": "<text>"}` |

```bash
curl -s http://localhost:11437/tokenize -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.5-4b-int4-ov", "prompt": "Hello world"}'
# {"tokens":[9419,1814],"count":2}
```

- The prompt is encoded as given. No chat template is applied, so a chat request costs more
  tokens than its message texts add up to. Build the full prompt yourself if you need that count.
- Works for loaded text-generation and vision models. Other kinds (embeddings, speech, image,
  rerank) and text models placed on the NPU get `400` `unsupported_operation`. An unknown model
  is `404` `model_not_found`; a model that is not ready is `503`. Neither route ever loads a
  model, not even a `load: "on_demand"` one, and neither waits for a load (`on_demand_wait_secs`
  does not apply): a model that is not loaded is a plain `503 model_unavailable` with no
  `Retry-After`; one whose load is running is `503` with `Retry-After`. Load the model first.
- `/detokenize` needs ids inside the model's vocabulary. An id that is negative or at or above
  the vocabulary size gets `400` `invalid_value` with `param: "tokens"`. A model whose detokenizer has no vocabulary table (SentencePiece-style, for example InternVL2.5) can only reject negative ids; an id above its vocabulary is decoded or fails with a `500`. Special tokens (for
  example `<|im_end|>`) decode to empty text.
- Neither route uses a generation slot, so neither is turned away with `429` when generation is
  saturated. Latency: a vision model with its own tokenizer answers in a few milliseconds even
  during a generation; a text model's call is served by the engine thread between steps, so it can
  wait behind a long prefill (up to about a second measured).
- Both follow the same API-key rule as the other inference routes: when API keys are configured, no
  key or a wrong key is a `401`.

### `POST /v1/systemone`

Ollama v0.35's System One wire format. Request: `model`, `state` (a string, or a JSON object/array
used as compact JSON), and `questions`, an object of named questions in the order they should be
answered:

```json
{"model": "nimble", "state": "Our checkout has returned 500 errors since 9am.",
 "questions": {
   "label":   {"type": "choice", "instructions": "Which label fits this ticket?",
               "criteria": {"billing": "Payments and refunds", "bug": "Software errors", "account": null}},
   "refund":  {"type": "noul",   "instructions": "Is the customer asking for a refund?"},
   "urgency": {"type": "score",  "instructions": "How urgent is this?", "criteria": ["Routine", "Soon", "Urgent"]}}}
```

Each answer carries its probabilities: `choice` → `choice` + `probabilities` + `confidence`;
`noul` → `noul` (P(yes)); `score` → `score` (expected level, 0…n−1) + `legend` + `probabilities` +
`confidence`, where confidence is 1 − normalised entropy. `usage.input_tokens` is the sum of the
per-question prompts. The model never generates text, so an answer is always one of the allowed
options.

Limits and checks, in Ollama's order (status, then Ollama's exact message):

| Check | Status |
|---|---|
| The first JSON value runs past 64 KiB (trailing bytes after it are ignored, as in Ollama) | 413 |
| Not decodable as a request (wording differs from Ollama's Go decoder; status matches) | 400 |
| A `keep_alive` that is neither a number of seconds nor a Go duration string, or a duration past 2^63 ns (about 292 years; *(since 0.9.1)* also a digit string too large to represent, which 0.9.0 let through) | 400 |
| Invalid request: 1–64 questions, 2–26 candidates per question, `instructions` required on every type, non-empty `state`, etc. | 400 |
| Unknown model | 404 |
| Model is not a decision model (a non-decision model is never loaded just to be refused) | 400 |
| Model loading / evicting (an on-demand model: after `on_demand_wait_secs`, see [On-demand models](#on-demand-models-waiting-for-the-load-since-091)) | 503 + `Retry-After: 5` |
| Device budget busy, or the model's turn gate stays busy for the whole queue timeout (only when `device_budgets` is configured for the model's device; requests queue at the gate holding no slot) | 429 |
| A prompt longer than the model's context (`prompt %d has %d tokens; expected 1–%d…`, or `…requires %d context tokens for scoring…` for the last two positions) — every prompt is checked before any inference | 400 |
| Inference failed; if the engine also failed to reload, the model is unloaded so it can be loaded again (503 on the next call) | 500 |

Differences from Ollama: Ollama makes the first request wait while a model loads, whereas
RustedVINO answers 503 + `Retry-After` unless `on_demand_wait_secs` is set *(since 0.9.1)*, and
then waits at most that long. `usage.output_tokens` is one per question; Ollama adds 1
when it reuses a shared prompt prefix. Response numbers are equal, but JSON number formatting
differs (`1` vs `1.0`). A client that disconnects stops the request before its next question.

### `GET /v1/realtime` (WebSocket)

The full protocol (every client and server event, limits, how a session ends, examples) is in
**[`REALTIME.md`](REALTIME.md)**. In short:

- **Audio in:** binary frames of 16 kHz mono PCM, 16-bit little-endian.
- **Voice activity detection** is server-side, with fixed values: speech starts after 100 ms
  above an RMS level of 0.015 (about −36 dBFS), and an utterance ends after 500 ms below it.
- **An utterance that never falls silent** is committed anyway after 29.5 s of audio (Whisper's
  window is 30 s); the audio after that point starts the next utterance, so a long monologue is
  answered in 29.5-second turns.
- **Every utterance goes speech-to-text → LLM → text-to-speech.** There is no transcription-only
  mode. For captions only, send each utterance to `/v1/audio/transcriptions` instead, as
  [`tools/livecaptions`](tools/livecaptions/README.md) does.
- **Idle timeout: 5 minutes.** A session that commits no utterance for 5 minutes is closed.
  The timer restarts at the end of every turn and after every `inject_history` (non-empty) or
  `reload_history` batch the server processes. Audio without detected speech and WebSocket
  pings do not count. The timeout is fixed, not a config setting.
- **Every close the server begins sends a close frame** *(since 0.9.1)* with a code and a machine-readable
  reason, then waits up to 2 s for the client's close reply:

  | Cause | Code | Reason |
  |---|---|---|
  | Idle timeout | 1000 | `idle_timeout` |
  | Server shutdown or restart | 1001 | `server_stopping` |
  | Admin force-close | 4000 | `closed_by_admin` |

  1011 `internal_error` is never sent: errors are `error` events, and the session stays open.
  A close the client begins gets only the echo of its own frame. A reset without a close frame
  means the server process died or the network failed. Details:
  [How a session ends](REALTIME.md#how-a-session-ends).

---

## Admin API status codes

Every `/v1/admin/*` route, plus `/metrics`, is gated by the keys file (never `config.json`
itself — see [`CONFIG.md`](CONFIG.md)'s `keys_file` entry). An admin-scope key is required when
`admin_api_keys` is non-empty; a valid inference-scope key presented to an admin route is then
authenticated but unauthorized (`403 insufficient_scope`), not silently admitted. **With
`admin_api_keys` empty, every `api_keys` entry has admin rights**: the admin routes and `/metrics`
fall back to the inference gate, which admits any inference key. Set `admin_api_keys` to separate
inference keys from admin keys. If no keys file was ever
successfully loaded at boot, admin routes hard-503 for the process's lifetime regardless of any
credential presented.

| Code | Meaning |
|---|---|
| 200 | OK |
| 400 | Bad request — invalid input (bad `vram_gb`/`kind`/`device`/`kv_cache_precision`, missing model directory, an empty `PATCH` body, `kind` supplied to `PATCH` (immutable), a duplicate or unknown id in `config/preload`). *(Since 0.9.1)* a request body that is not valid JSON or does not match the route's schema (a wrong type, a missing required field, an unknown field except on `PATCH`, see 422) gets the public JSON envelope `{"error": {"message", "type": "invalid_request_error", "code": "invalid_body"}}` on every admin route with a body (`/load`, `/resize`, `PATCH`, `models/add`, `config/preload`); the message starts `invalid JSON body:` and names the field. Before 0.9.1 these were plain text, and `PATCH` / `models/add` / `config/preload` answered a wrong type with 422. A missing `Content-Type: application/json` on `PATCH`, `models/add` and `config/preload` is still a plain `415`; `/load` and `/resize` accept any. The other `400`s on the admin routes (a value out of range, the cases above) keep their plain-text body. Inference-route codes are listed under [Error responses](#error-responses) |
| 401 | Missing or unrecognized Bearer key (`invalid_api_key`) |
| 403 | Authenticated with a valid inference-scope key, but the route requires admin scope (`insufficient_scope`); only when the keys file has `admin_api_keys` (without them an inference key is admitted, see above) |
| 404 | Model ID not registered; on `/load`, also a registered model whose directory or required files are missing, or whose configured `chat_template` override cannot be read (plain-text body naming the model and the path) |
| 405 | Wrong HTTP method |
| 409 | Conflict — e.g. DELETE on a NotLoaded model, `models/add` on an already-registered ID, any admin operation (`/load`, unload, deregister, `PATCH`, resize) on a model whose load or eviction is in flight (state `loading`/`evicting`; plain-text body; retry when it settles — a second `/load` while the first runs answers this at once), `PATCH`'ing `device`/`tier_preference` while the model is Ready (evict first), or `models/{id}/resize` on a model that is not currently `Ready` (use `/load` for one that isn't resident); `config_file_invalid` when `config.json` is already invalid (`models/add`, deregister) |
| 422 | `POST /v1/admin/config/reload` on a `config.json` that does not parse or validate; nothing was changed. Also, on `PATCH` only, an unknown or misspelled field, at the top level or inside `sampling` / `speculative` (it accepts only its patchable fields, see the route table; plain-text body naming the field). Every other bad JSON body is a `400 invalid_body` *(since 0.9.1)* |
| 429 | `models/{id}/resize` called again within `kv_resize_cooldown_secs` of that model's last resize (plain-text body). The inference routes' 429 codes are listed under [Error responses](#error-responses) |
| 503 | Transient or capacity — a load refused for memory (`insufficient_memory`): the model can never fit in system RAM or VRAM even after evicting everything evictable (nothing was evicted), or GenAI found too little free device memory for its KV cache after admission (transient: retry in a few seconds or unload a model; models the load had already evicted are named in the message), and the same code with `Retry-After: 5` when GenAI refuses a vision model's KV cache at its first request (the model is reloaded); GPU context poisoned (restart required: `GET /v1/admin/models` reports `server.gpu_poisoned: true`, and a server run under `--supervise` restarts its worker by itself when a model that was already serving fails this way — while that restart is pending the 503 says "the worker is restarting — retry shortly" and carries `Retry-After: 30` (a streaming request gets the same text in its SSE error frame), otherwise it says retrying will not help; a poisoning during startup or while loading a model leaves it up and degraded); a load refused while a stalled load on the same device has not returned (`device_stalled`, with `Retry-After`) or while the server is stopping (`server_stopping`); or no keys file was ever configured (`admin_not_configured`) |
| 504 | A model load did not return within `load_timeout_secs` (`load_timeout`); the load keeps running, booked as stalled. (Inference routes also answer 504 `generation_stalled`, see Error responses.) |
| 500 | Internal error (engine factory failure, I/O) |

### `POST /v1/admin/models/{id}/resize` — the failure mode to plan for

A resize is a real **evict-then-reload**, not an in-place operation (OpenVINO sizes a KV pool once,
at pipeline construction, and exposes no way to grow it afterward). Two consequences callers must
handle:

- **It is not atomic.** If the evict succeeds and the reload at the new size is then refused for
  memory (`503 insufficient_memory`), the server tries once to reload the model at its previous KV
  size, without evicting anything for it (models the refused reload already evicted stay
  unloaded). The error message then contains `resize_rolled_back` (the model is loaded again at
  its old size) or `resize_not_restored` (that reload failed too; the model is **unloaded**).
  - The rollback rescues a grow, and a reload refused by the engine (OpenVINO reporting less free
    device memory). It does **not** rescue a shrink refused by the live system-RAM gate on a
    shared-memory (UMA) box: such a refusal means the check credited the unloaded pool with
    memory it did not give back, and the old, larger pool cannot fit either without evicting. The
    model then stays **unloaded** (`resize_not_restored`). This is a known residual risk of 0.9.1.
  - Between the refused reload and the rollback the model is briefly `not_loaded`. A load of the
    same model in that moment (another client's `/load`, an on-demand request) makes the rollback
    report `resize_not_restored` although the model then comes up from that other load: check
    `GET /v1/admin/models` before reloading.
  - The resize cooldown (`kv_resize_cooldown_secs`) stays claimed after a rolled-back resize, so an
    immediate retry of the resize gets `429`.
  - Any other reload failure (`model_busy` because a request reached the model in that window,
    files missing, engine failure: a `500`) leaves the model unloaded, with no retry: treat it as
    down and decide explicitly whether to reload it at the old size or a different one.
- **It takes as long as an evict plus a cold load** (drain in-flight requests, then a fresh JIT
  compile). Use a long HTTP timeout, the same guidance as `/load`.

For a `max_concurrent_streams: 1` model this is a brief outage for that model, since there is no
second slot to serve from while it reloads.

### What an eviction did: the JSON answer (`Accept: application/json`, since 0.9.1)

`DELETE /v1/admin/models/{id}` (unload), `DELETE /v1/admin/models/{id}/register` (deregister),
`POST /v1/admin/models/{id}/resize` and `POST /v1/admin/models/{id}/load` answer a `200` with a
JSON body, which carries that sentence as `message`, **only when the request carries
`Accept: application/json`** (an `Accept` media range equal to `application/json`, any case, in
any of several `Accept` headers; `charset` and other parameters ignored; `q=0` means not wanted).
Without that header, or with `*/*` as `curl` sends, the body is the plain text of earlier versions,
byte for byte. Errors are unchanged either way. `rv` sends the header on its forced operations.
**A client that already sends `application/json` in its `Accept` header to these routes** (axios'
default `application/json, text/plain, */*`, for example) **gets the JSON body from 0.9.1 on**;
its sentence is in `message`.

```json
{"evicted": ["qwen3-14b-int4-ov"], "forced": true, "drained_s": 120.0, "aborted": 1, "teardown_ms": 734,
 "message": "model 'qwen3-14b-int4-ov' evicted"}
```

| Field | Meaning |
|---|---|
| `evicted` | Every model the call evicted, in order: the model itself for an unload; the model, if it was loaded, for a deregister (`[]` when it was not); for a resize the model first, then any model its reload (a forced load when the resize is forced) evicted for room; for a load, the models it evicted for room (`[]` when none) |
| `forced` | The request's `force` |
| `drained_s` | Seconds spent waiting for running requests to finish, summed over `evicted`. Exactly `0` when nothing was running |
| `aborted` | Requests still running when the drain deadline passed (2 minutes for an unload, deregister or resize, 15 s for a model a forced load evicts), on every engine kind: the same number `rustedvino_eviction_aborted_requests_total` gains. A text model aborts them (their clients get `model_unloading`); other engines finish or fail them during the teardown. Summed over `evicted`; see below |
| `teardown_ms` | Engine teardown (channel close, engine exit, pipeline destructor, thread join), summed over `evicted` |
| `message` | The plain-text body the route answers without the header, verbatim (for a load the KV-pool note, for a resize the pool it built) |

The facts come from each eviction itself, not from the server-wide counters. A non-forced call
never evicts a busy model, so its `aborted` is normally `0`. Only the continuous-batching text
engine stops a running generation at the deadline. Every other engine finishes the request it is
running, or fails it, while the forced eviction waits for its thread (up to 30 s, else
`503 eviction_incomplete`); that wait is in `teardown_ms`. An eviction of a stalled engine has no
drain deadline: it reports `aborted: 0` (its requests already ended with `generation_stalled`).
Known gap: a split SpeechT5 request cut between two pieces **before** the deadline (its client gets
`503 model_unavailable`) is not counted in `aborted`, nor in the counter.
