# RustedVINO — Interface Reference

Full route table for the running server. For prose (why it exists, what's implemented,
setup steps, config field reference) see **[`README.md`](README.md)**.

---

## Route table

### OpenAI-compatible endpoints

Drop-in replacements for the real OpenAI API surface — same request/response shape, same
error envelope. A client written against OpenAI's SDK works against these unmodified.

| Method | Route | Description |
|---|---|---|
| `GET` | `/v1/models` | Model list |
| `POST` | `/v1/chat/completions` | Chat completions — streaming (`stream:true`) + non-streaming; VLM image input **or** text-only |
| `POST` | `/v1/completions` | Legacy raw-prompt completions |
| `POST` | `/v1/embeddings` | Text embeddings |
| `POST` | `/v1/audio/transcriptions` | Speech-to-text (Whisper) |
| `POST` | `/v1/audio/speech` | Text-to-speech (Kokoro-82M, Coqui VITS, `SpeechT5`) |
| `POST` | `/v1/images/generations` | Text-to-image (SDXL, FLUX.1-schnell) |
| `POST` | `/v1/images/edits` | Image inpainting + img2img (SDXL) |

Within these routes, a few request fields are **RustedVINO extensions**, not part of OpenAI's
spec — same convention other OpenAI-compatible servers (vLLM, Ollama, llama.cpp) use: `top_k`,
`repetition_penalty` (applies a `1.1` safety-net default unless the client opts out with `1.0`),
`enable_thinking`. (`max_prompt_len` is a per-model, NPU-only **config** field, not a request
field — see [`CONFIG.md`](CONFIG.md).)

### RustedVINO extensions

Not part of OpenAI's API — server-specific infra, admin/lifecycle control, and endpoints
borrowed from other providers' conventions (rerank: Cohere/Jina-style; tokenize/detokenize:
common in vLLM-style servers). `/v1/realtime` borrows OpenAI's naming for the concept but is a
**bespoke protocol** (barge-in, session history/retrieval, voice-pin), not a re-implementation
of OpenAI's actual Realtime API event schema.

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — returns `{status, version, git_hash}`. Auth-exempt (only path that is) |
| `GET` | `/health_generate` | Deep readiness probe — runs 1-token GPU inference pass |
| `GET` | `/metrics` | Prometheus metrics (tokens/s EMA, TTFT, request duration) — full reference in [`METRICS.md`](METRICS.md). Gated exactly like an admin route despite the URL |
| `POST` | `/v1/rerank` | Cross-encoder document reranking |
| `POST` | `/tokenize` | Encode text to token IDs via the loaded model's tokenizer |
| `POST` | `/detokenize` | Decode token IDs back to text |
| `GET` | `/v1/realtime` | WebSocket upgrade — realtime STT → LLM → TTS voice flow (bespoke protocol) |
| `GET` | `/v1/admin/models` | List all registered models + state (`ready`/`not_loaded`/`loading`/`evicting`) |
| `GET` | `/v1/admin/models/completeness` | Report which registered models are missing required directory/tokenizer files |
| `POST` | `/v1/admin/models/add` | Register a new model at runtime and load it immediately; persists to `config.json` — `model_id`/`vram_gb` required, `kind` and every `ModelPolicy` field (device, co-residency knobs, image-gen provenance, …) optional, same shape as a static config stanza |
| `POST` | `/v1/admin/models/{id}/load` | Load model into VRAM (blocks until ready; evicts LRU if needed) |
| `POST` | `/v1/admin/models/{id}/check` | Check model state without loading |
| `POST` | `/v1/admin/models/{id}/resize` | Resize a **resident** model's KV-cache pool — evicts (draining in-flight requests) then reloads at the new size. Body `{"kv_cache_gb": <f64 > 0>}`; every other resolved override (`max_concurrent_streams`, …) is carried forward from the model's record, so the caller states only the delta. Deliberately **not** idempotent-when-`Ready` — that is exactly why it exists, since a repeated `/load` silently ignores a new `kv_cache_gb` on an already-loaded model |
| `DELETE` | `/v1/admin/models/{id}` | Evict model from VRAM (drains in-flight requests, then joins engine thread) |
| `PATCH` | `/v1/admin/models/{id}` | Update fields of an already-registered model — persisted always, live immediately for most fields, next-load for `vram_gb`/`kv_cache_gb`/`max_concurrent_streams`/`max_prompt_len`/`speculative`; `device`/`tier_preference` need the model `NotLoaded` first. `kind` is immutable |
| `DELETE` | `/v1/admin/models/{id}/register` | Deregister a model entirely (removed from `GET /v1/admin/models` until next restart/reload) |
| `POST` | `/v1/admin/config/reload` | Diff `config.json` against the live registry — add new entries, refresh changed fields, skip disruptive ones |
| `POST` | `/v1/admin/config/preload` | Replace the persisted `preload` list — `{"model_ids": [...]}` explicitly, or `{"from_live": true}` to snapshot the currently-`Ready` set |
| `POST` | `/v1/admin/keys/reload` | Hot-reload the keys file (`keys_file`) — rotate/add/revoke Bearer keys with no restart |
| `GET` | `/v1/admin/config/audit` | Report drift between `config.json` and the live model registry |
| `GET` | `/v1/admin/logs/tail` | Tail the server's in-memory log ring buffer (`?lines=N`, default 200, clamped to the buffer's fixed capacity — not a file) |
| `GET` | `/v1/admin/health/watchdog` | Age of the oldest in-flight generation; polled by `--supervise` to detect a wedged worker `/health` can't see |
| `GET` | `/v1/admin/voice-pin` | Current voice-flow pin (shared STT/LLM/TTS model set for realtime sessions) |
| `GET` | `/v1/admin/realtime/sessions` | List all live realtime WebSocket sessions |
| `GET` | `/v1/admin/realtime/sessions/{id}` | Full snapshot of one realtime session |
| `DELETE` | `/v1/admin/realtime/sessions/{id}` | Force-close a live realtime session |

---

## Admin API status codes

Every `/v1/admin/*` route, plus `/metrics`, is gated by the keys file (never `config.json`
itself — see [`CONFIG.md`](CONFIG.md)'s `keys_file` entry). An admin-scope key is required when
`admin_api_keys` is non-empty; a valid inference-scope key presented to an admin route is
authenticated but unauthorized (`403`), not silently admitted. If no keys file was ever
successfully loaded at boot, admin routes hard-503 for the process's lifetime regardless of any
credential presented.

| Code | Meaning |
|---|---|
| 200 | OK |
| 400 | Bad request — invalid input (bad `vram_gb`/`kind`/`device`, missing model directory, an empty `PATCH` body, `kind` supplied to `PATCH` (immutable), a duplicate or unknown id in `config/preload`) |
| 401 | Missing or unrecognized Bearer key (`invalid_api_key`) |
| 403 | Authenticated with a valid inference-scope key, but the route requires admin scope (`insufficient_scope`) |
| 404 | Model ID not registered |
| 405 | Wrong HTTP method |
| 409 | Conflict — e.g. DELETE on a NotLoaded model, `models/add` on an already-registered ID, `PATCH` on a Loading/Evicting model, `PATCH`'ing `device`/`tier_preference` while the model is Ready (evict first), or `models/{id}/resize` on a model that is not currently `Ready` (use `/load` for one that isn't resident) |
| 429 | Too many requests — all slots busy, admission timeout expired, or `models/{id}/resize` called again within `kv_resize_cooldown_secs` of that model's last resize |
| 503 | Transient — model is Loading/Evicting, GPU context poisoned (restart required), or no keys file was ever configured (`admin_not_configured`) |
| 500 | Internal error (engine factory failure, I/O) |

### `POST /v1/admin/models/{id}/resize` — the failure mode to plan for

A resize is a real **evict-then-reload**, not an in-place operation (OpenVINO sizes a KV pool once,
at pipeline construction, and exposes no way to grow it afterward). Two consequences callers must
handle:

- **It is not atomic.** If the evict succeeds and the reload then fails (the new size doesn't fit,
  files missing, engine failure), the model is left **unloaded** — not silently reverted to its old
  size. This is deliberate: there is no automatic rollback-and-retry. On a `500` from this route,
  treat the model as down and decide explicitly whether to reload it at the old size or a different
  one.
- **It takes as long as an evict plus a cold load** (drain in-flight requests, then a fresh JIT
  compile). Use a long HTTP timeout, the same guidance as `/load`.

For a `max_concurrent_streams: 1` model this is a brief outage for that model, since there is no
second slot to serve from while it reloads.
