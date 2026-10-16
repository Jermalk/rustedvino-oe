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
| `GET` | `/v1/models/{id}` | One model (OpenAI retrieve-model); 404 `model_not_found` unless loaded and ready |
| `POST` | `/v1/chat/completions` | Chat completions — streaming (`stream:true`) + non-streaming; VLM image input **or** text-only |
| `POST` | `/v1/completions` | Legacy raw-prompt completions |
| `POST` | `/v1/embeddings` | Text embeddings |
| `POST` | `/v1/audio/transcriptions` | Speech-to-text (Whisper) |
| `POST` | `/v1/audio/translations` | Speech in any language → English text (Whisper) |
| `POST` | `/v1/audio/speech` | Text-to-speech (Kokoro-82M, Coqui VITS, `SpeechT5`) |
| `POST` | `/v1/images/generations` | Text-to-image (SDXL, FLUX.1-schnell) |
| `POST` | `/v1/images/edits` | Image inpainting + img2img (SDXL) |

Within these routes, a few request fields are **RustedVINO extensions**, not part of OpenAI's
spec — same convention other OpenAI-compatible servers (vLLM, Ollama, llama.cpp) use: `top_k`,
`repetition_penalty` (applies a `1.1` safety-net default unless the client opts out with `1.0`),
`enable_thinking`. (`max_prompt_len` is a per-model, NPU-only **config** field, not a request
field — see [`CONFIG.md`](CONFIG.md).)

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
| `GET` | `/health` | Liveness probe — returns `{status, version, git_hash}`. Auth-exempt (only path that is) |
| `GET` | `/health_generate` | Deep readiness probe — runs 1-token GPU inference pass |
| `GET` | `/metrics` | Prometheus metrics (tokens/s EMA, TTFT, request duration) — full reference in [`METRICS.md`](METRICS.md). Gated exactly like an admin route despite the URL |
| `POST` | `/v1/rerank` | Cross-encoder document reranking |
| `POST` | `/tokenize` | Encode text to token IDs via the loaded model's tokenizer (text and vision models) — see [`POST /tokenize` and `POST /detokenize`](#post-tokenize-and-post-detokenize) |
| `POST` | `/detokenize` | Decode token IDs back to text |
| `GET` | `/v1/realtime` | WebSocket upgrade — realtime STT → LLM → TTS voice flow (bespoke protocol) |
| `GET` | `/v1/admin/models` | List all registered models + state (`ready`/`not_loaded`/`loading`/`evicting`) |
| `GET` | `/v1/admin/models/completeness` | Report which registered models are missing required directory/tokenizer files |
| `POST` | `/v1/admin/models/add` | Register a new model at runtime and load it immediately; persists to `config.json` — `model_id`/`vram_gb` required, `kind` and every `ModelPolicy` field (device, co-residency knobs, image-gen provenance, …) optional, same shape as a static config stanza |
| `POST` | `/v1/admin/models/{id}/load` | Load model into VRAM (blocks until ready; evicts idle least-recently-used models if needed). A model with a request running is never evicted to make room: the load is refused with `503 model_busy` (+ `Retry-After`) unless the body carries `"force": true`, which also bypasses the eviction-grace window (the busy model is drained for up to 15 s, then what is left is aborted) |
| `POST` | `/v1/admin/models/{id}/check` | Dry-run admission check: does it fit, which models would be evicted. `?force=true` previews a load that bypasses the eviction-grace window and the busy rule (never `pinned`, non-`evictable` or a live realtime session's models). Response fields include `would_evict`, `busy_victims` (members of `would_evict` with a request running — non-empty only for a forced check) and `blocked_by_busy` (models left alone only because a request is running on them, whose eviction would have made the load fit) |
| `POST` | `/v1/admin/models/{id}/resize` | Resize a **resident** model's KV-cache pool — evicts (draining in-flight requests) then reloads at the new size. Body `{"kv_cache_gb": <f64 > 0>, "force": <bool, default false>}`; a model with a request running is refused with `409 model_busy` unless `force` is `true`, and a resize whose reload could not be admitted is refused before anything is evicted; every other resolved override (`max_concurrent_streams`, …) is carried forward from the model's record, so the caller states only the delta. Deliberately **not** idempotent-when-`Ready` — that is exactly why it exists, since a repeated `/load` silently ignores a new `kv_cache_gb` on an already-loaded model |
| `DELETE` | `/v1/admin/models/{id}` | Evict model from VRAM. A model with a request running is refused with `409 model_busy` (+ `Retry-After`); `?force=true` drains it (up to 2 minutes), aborts what is left, then joins the engine thread. For every engine except the continuous-batching text engine a forced eviction waits up to 30 s for the engine thread to finish its request and otherwise answers `503 eviction_incomplete` (the memory is released when the thread exits) |
| `PATCH` | `/v1/admin/models/{id}` | Update fields of an already-registered model — persisted always, live immediately for most fields, next-load for `vram_gb`/`kv_cache_gb`/`max_concurrent_streams`/`enable_prefix_caching`/`max_prompt_len`/`min_response_len`/`speculative`; `device`/`tier_preference` need the model `NotLoaded` first. `kind` is immutable |
| `DELETE` | `/v1/admin/models/{id}/register` | Deregister a model entirely (removed from `GET /v1/admin/models` until next restart/reload). Evicts it first if loaded: a model with a request running is refused with `409 model_busy` unless `?force=true` |
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

## Request limits and endpoint behaviour

### Request size (every route)

A request body may be at most **50 MiB**. That is a fixed server limit, not a config field. A
larger request gets `413` with error code `request_too_large`, whether it is a JSON body (a chat
request with a large inline image, say) or a multipart upload (audio, images).

### Error responses

Rejections use the OpenAI error envelope, `{"error": {"message", "type", "param", "code"}}`, with a
stable `code`: for example `invalid_body` (malformed or incomplete JSON), `invalid_multipart` (a
multipart route called without a multipart body), `invalid_audio` (an upload that cannot be decoded
as audio), `invalid_query` (a bad query string such as `?force=maybe`), `model_busy` (409 or 503, with `Retry-After`: a model with a request running was not evicted — an unload, deregister or resize refused with 409, or a load refused with 503 because the only room is held by such models; retry when the request ends, or repeat the call with `force`), `eviction_incomplete` (503, with `Retry-After`: a forced eviction's engine is still finishing its request, so its memory is not free yet), `method_not_allowed` (405, with the `Allow` header), `context_length_exceeded` (a prompt
that can never fit this model: retrying is pointless) and `pool_capacity_exceeded` (the prompt fits
alone but not beside the requests already running: retry shortly), `invalid_value` (a value out of
range, or a chat message whose `role` is not `system`, `developer`, `user`, `assistant`, `tool` or
`function`; `param` names the field, for example `messages[1].role`) and `unknown_url` (404, a path
the server does not have, as OpenAI answers it). Routes added by an extension answer as the extension defines them.

### `POST /v1/audio/speech`

- **Fields:** `model` and `input` (required; at most 4096 characters), `voice`, `response_format`
  (`wav` default, or `pcm`) and `speed` (0.25-4.0), as in OpenAI's API, plus two extensions:
  - `language` (optional, e.g. `en`, `pl`, `vi`; a region suffix such as `pl-PL` is fine): the
    language of `input`. Omit it and the server uses the language the model speaks (SpeechT5 and
    Kokoro: English, the Polish VITS voice: Polish, a Piper voice: its eSpeak language).
  - `normalize` (optional, default `true`): speak digits and symbols (`%`, `&`, `$`, ...) as
    words before synthesis. Speech models drop or garble raw digits ("Cena wzrosła o 15%" came
    back as "Cena wzrosła o"). Send `false` if the text is already speakable.
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

### `POST /v1/audio/transcriptions`

- **Fields:**
  - `file` (required): WAV, MP3, OGG, FLAC or M4A. The server decodes it and resamples it to
    16 kHz mono itself.
  - `model` (required).
  - `language` (optional, e.g. `en`, `pl`). Recommended: detection from a few seconds of audio
    is unreliable.
  - `response_format`: `json` (default), `text`, `srt` or `verbose_json`.
  - Other OpenAI fields (`prompt`, `temperature`, …) are accepted and ignored.
- **Undecodable audio** (garbage, truncated, an unsupported container) returns `400` with code
  `invalid_audio`.
- **Long audio:** one request transcribes the whole file, including audio longer than Whisper's
  30-second window. For example, an 87.6-second file came back complete, with `verbose_json`
  segments to the end. In practice the length limit is the 50 MiB upload cap: about 27 minutes
  of 16 kHz mono 16-bit WAV, and much longer as MP3 or OGG.

### `POST /v1/audio/translations`

- **Output is always English**, whatever language is spoken. That is what Whisper's translate
  task does, and what OpenAI's endpoint promises. For any other target language, transcribe and
  then translate with `/v1/chat/completions`.
- **Same request and response format as transcriptions**, and the same 50 MiB cap. `language`
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
  -d '{"model": "qwen3.5-4b-int8-ov", "prompt": "Hello world"}'
# {"tokens":[9419,1814],"count":2}
```

- The prompt is encoded as given. No chat template is applied, so a chat request costs more
  tokens than its message texts add up to. Build the full prompt yourself if you need that count.
- Works for loaded text-generation and vision models. Other kinds (embeddings, speech, image,
  rerank) and text models placed on the NPU get `400` `unsupported_operation`. An unknown model
  is `404` `model_not_found`; a model that is not ready is `503`.
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
| Invalid request: 1–64 questions, 2–26 candidates per question, `instructions` required on every type, non-empty `state`, etc. | 400 |
| Unknown model | 404 |
| Model is not a decision model (a non-decision model is never loaded just to be refused) | 400 |
| Model loading / evicting | 503 + `Retry-After: 5` |
| Device budget busy, or the model's turn gate stays busy for the whole queue timeout (only when `device_budgets` is configured for the model's device; requests queue at the gate holding no slot) | 429 |
| A prompt longer than the model's context (`prompt %d has %d tokens; expected 1–%d…`, or `…requires %d context tokens for scoring…` for the last two positions) — every prompt is checked before any inference | 400 |
| Inference failed; if the engine also failed to reload, the model is unloaded so it can be loaded again (503 on the next call) | 500 |

Differences from Ollama: Ollama makes the first request wait while a model loads, whereas
RustedVINO answers 503 + `Retry-After`. `usage.output_tokens` is one per question; Ollama adds 1
when it reuses a shared prompt prefix. Response numbers are equal, but JSON number formatting
differs (`1` vs `1.0`). A client that disconnects stops the request before its next question.

### `GET /v1/realtime` (WebSocket)

- **Audio in:** binary frames of 16 kHz mono PCM, 16-bit little-endian.
- **Voice activity detection** is server-side, with fixed values: speech starts after 100 ms
  above an RMS level of 0.015 (about −36 dBFS), and an utterance ends after 500 ms below it.
- **Every utterance goes speech-to-text → LLM → text-to-speech.** There is no transcription-only
  mode. For captions only, send each utterance to `/v1/audio/transcriptions` instead, as
  [`tools/livecaptions`](tools/livecaptions/README.md) does.

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
| 503 | Transient or capacity — model is Loading/Evicting; a load refused because the model can never fit in system RAM or VRAM even after evicting everything evictable (`insufficient_memory`, nothing was evicted); GPU context poisoned (restart required: `GET /v1/admin/models` reports `server.gpu_poisoned: true`, and a server run under `--supervise` restarts its worker by itself when a model that was already serving fails this way; a poisoning during startup or while loading a model leaves it up and degraded); or no keys file was ever configured (`admin_not_configured`) |
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
