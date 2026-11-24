# Changelog

All notable changes to RustedVINO are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and from 0.7.0 on the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (0.6.8 was chosen for luck, not as a
semver claim). Full release notes, with measurements and download instructions, are on each
[GitHub release](https://github.com/Jermalk/rustedvino-oe/releases).

## [0.9.1] — 2026-11-25

### Upgrade notes from 0.9.0
- **Set `max_images_per_request` to 4 in an existing config** (the shipped `config.example.json` now does; the
  built-in default stays 16). The vision engine's memory grows with the images in one request, and on an Arc Pro B70
  with 62 GB of RAM 7 photos at the 4 MP `max_image_pixels` got the server killed by the kernel; 4 worked. See Known
  issues below.
- **`reasoning_effort` on `/v1/chat/completions` is now read** (0.9.0 ignored it). A client that already sends it
  gets: `none` / `minimal` = thinking off; `low` / `medium` / `high` (`xhigh` and `max` count as `high`) = thinking
  on with a strict budget of 1,024 / 4,096 / 16,384 tokens, kept below `max_tokens` minus a reserve of 512 tokens
  (half of `max_tokens` when that is less; with no `max_tokens`, the default 8,192 cap makes `high` 7,680). A
  request that thinks past its budget now ends with `finish_reason: "length"` and an empty `content`; on gpt-oss
  only the analysis channel counts, so a tool call is never cut. Any other value (`"ultra"`, `"HIGH"`, a number)
  is a `400 invalid_reasoning_effort`, as is one in
  `chat_template_kwargs`; a non-boolean `chat_template_kwargs.preserve_thinking` is a `400
  invalid_chat_template_kwargs`. Clients that think long should send `none`, a larger `max_tokens`, or no
  `reasoning_effort`.
- **Audio transcription no longer accepts ALAC, or source sample rates outside 4–48 kHz** (`400 invalid_audio`). The ALAC
  decoder allocated a buffer sized by the file before decoding anything, so one crafted file could abort the server.
  Convert ALAC to FLAC or AAC. Whisper hears 16 kHz, so resampling a 96 kHz recording to 48 kHz or less loses nothing.
- **A network `bind_addr` with no `api_keys` in the keys file now stops startup (exit 78), even when the file has admin
  keys.** Admin keys protect only `/v1/admin/*`, so 0.9.0 served inference on such a bind to anyone who could reach the
  port (the keys file `rv start` creates has only an admin key). Add an `api_keys` entry, bind `127.0.0.1`, or set
  `allow_insecure_public_bind: true` to keep inference open on purpose (the server then logs a WARN).
- **A model whose `chat_template` override cannot be read no longer loads.** 0.9.0 logged a WARN and served a generic
  template (tool calling degraded). Check the path each model's `chat_template` names (a relative path resolves against
  the config file's directory); the load error names the model and the path.
- **A boot preload on a discrete GPU can now be refused under the device-headroom rule.** 0.9.0 preloaded without
  reading the device memory, so this rule never ran at boot. Now a preload is refused when it would leave a vision
  model on that GPU (itself, or one preloaded earlier) less room than its `kv_headroom_min_tokens` floor; the refusal is
  the usual `preload_failed` startup warning, the model stays unloaded and the server starts. Nothing is evicted at
  boot (see "A preload never evicts another preload" below). The rule also covers non-vision models
  (embedding, speech, image, text) preloaded after a vision model on that GPU. Load such a model later with `rv load`,
  which may evict to make room, or give the models smaller `kv_cache_gb` values.
- **A model's own `repetition_penalty` from its `generation_config.json` now applies**, with no config change, to
  requests that do not send one (chat, vision, NPU, `/v1/completions` and realtime), when it is above `1.0` and at most
  `2.0`. It replaces the server's `1.1` net for that model: a model that ships `1.05` (a Qwen3-Coder-30B conversion does) now gets
  `1.05`, one that ships `1.2` gets `1.2`. Exactly `1.0` counts as unset (the `1.1` net stays); a value below `1.0` is
  ignored with a warning. To keep `1.1`, set `"sampling": {"repetition_penalty": 1.1}` on the model.
- **Realtime voice turns now use the model's sampling defaults** instead of always decoding greedy. A model whose
  `generation_config.json` has `"do_sample": true` now samples on `/v1/realtime` with its own values (Qwen3: temperature
  `0.6`, `top_p` `0.95`, `top_k` `20`), as it already did on `/v1/chat/completions`, and its file penalty applies (above).
  To keep voice greedy, set `"sampling": {"temperature": 0}` on the model (this applies to its chat requests too).
- **Chat images to a vision model are now downscaled to at most `max_image_pixels` (default 4 000 000 pixels) each.**
  A larger image (a 12 MP phone photo, say) reaches the model at about 2309 × 1732 instead of full size: less fine
  detail, fewer image tokens, much less memory. One such photo could take the server down on a UMA laptop. To send
  larger images on a box with the memory for them, raise `max_image_pixels` (the model's own preprocessor may resize
  again after that). Image edits are not affected.
- **An engine that stops making progress is now ended, and its model is marked dead** (new keys
  `stream_first_token_timeout_secs`, default 600, and `stream_token_gap_timeout_secs`, default 120; `0` turns each off).
  The watch is per engine and needs no client: an engine call with no progress within 600 s of its start, or no next
  progress within 120 s of the last (the text engine: a completed step; vision and NPU engines: a piece of text),
  ends every request on that engine with `504 generation_stalled` (a streaming request: an SSE error frame with that
  code, then `[DONE]`). The model is then refused until the server restarts: a GPU model counts as a poisoned GPU
  context (under `--supervise` the worker restarts itself; without it only that model is refused); a CPU or NPU
  model's stuck call keeps its memory booked, so an unload does not free it — under `--supervise` the generation
  watchdog restarts the worker later for a CPU model; a stuck NPU model needs a manual restart. If your hardware legitimately needs longer for a first progress signal (a very
  long prompt on a slow CPU, a large image on a vision model), raise `stream_first_token_timeout_secs`. Add both
  keys to your `config.json` (`scripts/fill-config-defaults.py` writes them out).
- **A preload never evicts another preload.** 0.9.0 let a later `preload` model LRU-evict an earlier one when the list
  did not fit the budget (logged at INFO only). Now a model that does not fit beside the preload models already loaded
  is skipped: a WARN and a `preload_skipped` startup warning naming the gate that refused it (memory budget or system
  RAM) and the shortfall in GB (text in `rv status` and the admin model list; `/health` shows the count and codes
  only); it stays unloaded, the rest of the list still loads, and the warning clears once the model is loaded. Check
  `preload` against the budget, or `rv load` the model later.
- **After a supervised crash respawn, config `preload` models that were not loaded at the crash are skipped.** The
  respawned worker brings back what the crashed one had loaded (see Fixed), and names the skipped preload models in a
  `respawn_preload_skipped` startup warning (it stays until a restart). A text or vision `preload` model loads last in
  the preload phase, or after listening at the end of the restore when the set has a fixed-size runtime model it
  must leave room for. If every model was unloaded before the crash, the respawn loads nothing; a service restart runs the full
  `preload` as always.

- **An argument the server binary does not know now stops it with exit 64** and the usage on stderr. 0.9.0 ignored
  unknown arguments and started the server for real, so `rustedvino --config my.json` silently served whatever
  config the search found. There is no `--config` option: name the file with the `RUSTEDVINO_CONFIG` environment
  variable (`rv start` sets it). `rustedvino --help` prints every option and the config search order.
- **Some zero values that broke the server are now refused at boot (exit 78)**: `ov_cache_sweep_interval_secs`,
  `kv_pressure_sweep_interval_secs` and `max_prompt_array`, and in `supervisor` `watchdog_poll_secs`, `max_restarts`,
  `health_timeout_secs` and `restart_window_secs`. A `0` there did not mean "off": the sweeps re-ran back to back,
  every `/v1/completions` request was refused, the supervisor crashed (`watchdog_poll_secs`), gave up before the
  first worker (`max_restarts`), killed every new worker as hung (`health_timeout_secs`) or never stopped a crash loop
  (`restart_window_secs`). The nine `supervisor` seconds fields and both sweep intervals are also capped at one year
  (31 536 000). The settings that do mean "off" at `0` are unchanged; CONFIG.md lists what `0` does for each setting.
- **A `speculative.draft_model` that is not a plain folder name is refused** (at boot, `models/add` and PATCH):
  a `/`, `\`, `..` or a drive letter (`C:`) would read the draft from outside `models_dir`. A model id with a drive
  letter is refused the same way on every platform. The ids of the `models` map in `config.json` get the same check:
  such an id now stops startup (exit 78) and fails `POST /v1/admin/config/reload` (422). Refused, though 0.9.0
  accepted them: an id with `..`, an absolute path or a drive letter (`../shared/qwen`, `C:`), which read the model
  from outside `models_dir`; a nested id such as `org/name` (a folder inside `models_dir`, refused because
  a model id must be one folder name, as `POST /v1/admin/models/add` already required); and a one-letter name followed by a colon, such as `m:7b` or
  `q:latest`, which the drive-letter rule refuses on every platform (`qwen:7b` stays allowed). Give the model a
  plain folder name directly inside `models_dir`.
- **`/v1/audio/transcriptions` and `/v1/audio/translations` refuse a `language` Whisper has no code for** with
  `400 invalid_language` (0.9.0 answered 500). `PL`, `pl-PL` and `en_US` are now accepted as `pl` and `en`, and
  `auto` means "detect", like no field.
- **`/v1/rerank` accepts at most 1000 `documents` per request** (`400 too_many_documents`); split a longer list.
- **Admin routes answer a bad JSON body with `400` and the public JSON error** (`"code": "invalid_body"`, message
  `invalid JSON body: …` naming the field), like the inference routes. Changed: `PATCH /v1/admin/models/{id}` and
  `POST /v1/admin/models/add` answered a wrong type or a missing field with `422` (now `400`); `models/add` also an
  unknown key inside `sampling` / `speculative` (now `400`); `POST /v1/admin/config/preload` answered malformed JSON
  with a plain-text `400` and a wrong type with `422` (now both the `400` envelope); `/load` and `/resize` answered
  `400` in plain text (now the envelope, same status). Unchanged: an unknown field on `PATCH` (top level or nested)
  stays `422` with its plain-text body; a missing `Content-Type: application/json` on `PATCH`, `models/add` and
  `config/preload` stays `415`; `/load` and `/resize` still take any `Content-Type`, and `/load` an empty body. A
  script that matched `422` for a wrong type, or read the `/load` 400 as text, should read `error.message`.
- **`GET /v1/models` now also lists `load: "on_demand"` models that are not loaded yet**, with
  `"state": "not_loaded"` (or `"loading"`) and no `max_model_len`. A client that treated every entry as loaded
  should check that `state` is absent; a request to such a model starts its load and answers `503` with
  `Retry-After` until it is ready (or waits for it, see `on_demand_wait_secs` below). Ready entries are
  unchanged; an eager model that is not loaded is still not listed. `GET /v1/models/{id}` follows the same rule.
- **`/v1/completions` now starts the load of a `load: "on_demand"` model** that is not loaded, like chat and the
  other inference routes. 0.9.0 answered it with the plain not-loaded `503` (no `Retry-After`) and loaded nothing;
  now it gets `503` + `Retry-After` while the load runs (or waits for it), so a completion to an unloaded on-demand
  model can now cause a load, and with it an eviction of an idle model to make room. Load the model first, or keep
  it `eager`, if that is not wanted. `/tokenize` and `/detokenize` never load a model or wait, as in 0.9.0 (a model
  that is not loaded gets the plain `503`), so a token count cannot evict a loaded model; their one change is the
  wrong-kind refusal in the next note.
- **An on-demand model of the wrong kind for a route is refused before it is loaded, when the model's kind is
  known.** 0.9.0 started the load of, say, an STT model sent to `/v1/embeddings` (evicting an idle model for room if
  needed) and refused the retry once it was loaded; now the first request gets the route's wrong-kind answer
  (`400`; on chat `501` for the kinds chat does not serve) and nothing loads. The kind is known when the config
  sets `kind`, the model's files showed it at registration, or an earlier load found it; otherwise the request
  loads the model as before and the load decides. An NPU text model sent to `/v1/completions` is still loaded
  before it is refused. `/tokenize` and `/detokenize` give this refusal too (they never load anything).

### Added
- **`on_demand_wait_secs`: a request can wait for an on-demand load** (global, `0` to `300`, shipped `0` = the
  0.9.0 behaviour; it must stay below a non-zero `load_timeout_secs`, or startup stops). Above `0`, a request to a
  `load: "on_demand"` model that is not loaded waits for the load, at most that long (and at most the model's own
  `load_timeout_secs` minus 2 s), and is then served; a failed load answers with its error, an expired wait with
  the `503` + `Retry-After`. Every inference route that loads waits the same way (`/tokenize` and `/detokenize`
  never load or wait). Streaming chat and completions get their
  `200` at once and SSE keep-alive comments every 15 s while waiting; a later error is one SSE error frame, then
  `[DONE]`. At most 64 requests wait at once; past that the immediate `503` + `Retry-After`. A client that
  disconnects frees its place and does not cancel the load. A model whose load is already past its deadline (or
  whose device has a stalled load) is answered at once with that stall's `504 load_timeout` / `503 device_stalled`
  instead of waiting. TTFT and request duration of a waited request are measured from the end of the wait. About
  `120` suits chat UIs such as Open WebUI. Restart-only. See CONFIG.md and INTERFACE.md ("On-demand models: waiting
  for the load").
- **`/v1/models` lists on-demand models before their first load** (`state` `not_loaded` / `loading`), so a chat
  UI's model picker can offer them. See the upgrade note. The repository's `semsearch` and `livecaptions` tools
  probe only entries with no `state`, so the new entries do not make them start an on-demand load.
- **"Connect a client" in the README**: opencode (provider block, limits, thinking setting), Open WebUI
  (connection settings) and the OpenAI Python SDK.
- **`models_dir_missing` and `models_dir_empty` startup warnings**: a `models_dir` that does not exist, is not a
  directory or cannot be read, or that holds no model directory, is a WARN at boot and a startup warning naming
  the absolute path checked (and, for a relative path, the working directory it resolves against: a relative
  `models_dir` resolves against the working directory, not the config file's directory). The server still boots.
  `rv status` prints them; `/health` lists their codes. Both are cleared once a model is loaded.
- **Thinking control on `/v1/chat/completions`**: OpenAI's `reasoning_effort` (`none` and `minimal` turn thinking
  off; `low`, `medium` and `high` turn it on with a budget of 1,024, 4,096 and 16,384 tokens; `xhigh` and `max`
  are accepted as `high`), a request budget `thinking_budget_tokens` (alias `thinking_token_budget`; `0` = off; it
  limits thinking that is on, and an explicit off always wins) and a per-model default
  `thinking_budget_tokens` in `config.json` (`null` in every shipped config; settable with `PATCH`, reload, `rv add` /
  `rv patch --thinking-budget-tokens`, shown in `GET /v1/admin/models`). **The budget is strict** on text, vision and
  NPU models: a request that reaches it while still thinking ends with `finish_reason: "length"`, an empty `content`
  and its reasoning in `reasoning_content`, with no tool call parsed (gpt-oss: only the analysis channel is
  budgeted, so a tool call is never cut). Exact on text models; vision and NPU models count text chunks at a
  measured 1.37 (vision) and 1.04 (NPU) tokens each, so their stop is approximate. The budget is kept below `max_tokens` minus a reserve
  of 512 tokens, or half of `max_tokens` when that is less.
  `chat_template_kwargs` now also reads `reasoning_effort` and `preserve_thinking` (and still `enable_thinking`);
  other keys are ignored, and the template gets server-built values only. A wrong value is a 400 naming the field
  (`invalid_reasoning_effort`, `invalid_thinking_budget`, `conflicting_thinking_budget`,
  `invalid_chat_template_kwargs`). See INTERFACE.md "Thinking control" and CONFIG.md "Thinking budget".
- **`rustedvino --help` / `-h`** prints every option, the config search order (`RUSTEDVINO_CONFIG` first) and the exit
  codes, and exits 0.
- **`usage.completion_tokens_details.reasoning_tokens`** on `/v1/chat/completions` and `/v1/completions`, in the
  buffered body and the streamed `include_usage` chunk, for text, vision and NPU models and the buffered tool path:
  the tokens of the reasoning block (markers included), always present (0 when thinking is off), never above
  `completion_tokens`. Exact on text models; a tokenizer recount, so approximate, on vision and NPU models.
  `prompt_tokens_details.cached_tokens` is not reported (GenAI gives no per-request hit count). See INTERFACE.md
  "The `usage` object".
- **Three per-model series on `/metrics`**: `rustedvino_prompt_tokens_total` and `rustedvino_reasoning_tokens_total`
  (counters, the same figures as `usage`) and `rustedvino_prefill_tokens_per_second` (histogram, "effective prefill
  (prefix-cache hits included)": prompt tokens ÷ the time from the engine's start of the request to its first token,
  queue time excluded; prompts of 256 tokens or more only). Labels `model`, `device`, `kind`, `modality`. See
  METRICS.md "Reading the prefill histogram".
- **Forced operations report what the force did.** Unload, deregister, resize and load answer a `200` with
  `{evicted, forced, drained_s, aborted, teardown_ms, message}` (`message`: the plain-text body) when the request carries `Accept: application/json`; without
  it the plain-text body is unchanged. `drained_s` is `0` exactly when nothing was running, `aborted` counts the
  requests still running when the drain deadline passed (a text model aborts them, other engines finish or fail
  them during the teardown). A client that already sends `application/json` in `Accept` to these routes (axios'
  default, for example) now gets the JSON body. See INTERFACE.md "What an eviction did".
- **`rv unload|deregister|resize|load --force`** print the running requests and the drain deadline before the call,
  then `nothing was running` / `finished during the drain (X s)` / `drained X s; N request(s) still running at the
  drain deadline` (also on stderr);
  `drain/abort unknown` against an older server. **`rv load`** prints its own load time beside the server's
  engine time (`loaded in 13.1 s on GPU.1 (engine 12.8 s)`).
- **`rv metrics [--grep TEXT] [--json]`**: `/metrics` with rv's admin key; one `effective prefill ~N tok/s (p50) ·
  decode M tok/s` line per loaded model, then the raw samples (unlabelled ones included); `key_suffix` values masked.
- **`/health` reports startup warnings as `startup_warnings: {count, codes}`** — the number and the distinct codes
  only (the endpoint is unauthenticated); the messages stay in `GET /v1/admin/models` and `rv status`.
- **`stream_first_token_timeout_secs`** (default 600) and **`stream_token_gap_timeout_secs`** (default 120): the
  engine stall watchdog for every text generation engine (text, vision and NPU; chat, `/v1/completions`, the realtime
  LLM turn), with or without `--supervise`. One watcher per engine, on the engine's own progress, independent of any
  client connection; an idle engine is never stalled. When a deadline passes every request queued on or running in
  that engine ends with `generation_stalled` (`504` + `Retry-After: 30`, or an SSE error frame; a realtime turn gets an
  `llm` error in plain words), and the model is marked dead: a GPU model sets the GPU-poisoned flag (supervised: the
  worker exits 75 and is respawned; unsupervised: only that model answers 503 until a restart), a CPU or NPU model
  shows `stalled` in `rv status` / `GET /v1/admin/models` and answers `503 model_unavailable`; its memory stays
  booked, and a reload is refused, until the stuck call's thread exits. See CONFIG.md, including the known limits
  (vision/NPU progress is visible text only; a vision prefill is one call).
- **Per-model sampling defaults: `"sampling": {temperature, top_p, top_k, repetition_penalty, presence_penalty,
  frequency_penalty}`** in a model's config stanza, `POST /v1/admin/models/add` and `PATCH /v1/admin/models/{id}`. They
  apply to requests that leave those values out, over the model's own `generation_config.json`, so a model whose file
  ships no sampling recommendation (it runs greedy) can get the one on its model card. **Values the client sends
  always win**; a request with tools and no sampling values of its own stays greedy, and a request that is greedy
  because of tools or its own `temperature: 0` gets none of the configured `presence_penalty` / `frequency_penalty`
  (the `repetition_penalty` chain still applies to it). Each model has one profile used by
  chat, vision, NPU, `/v1/completions` and **realtime (which follows the profile)**; on an NPU the configured presence
  and frequency penalties are not sent. Unknown keys inside `sampling` (a `temprature` typo) are refused, values are
  range-checked, and `top_p`/`top_k` without a `temperature` are refused on add and PATCH unless the model's file
  has `"do_sample": true` with a temperature (startup, reload and each load WARN about it instead, since the model's
  file can change on disk). A `PATCH` applies to the next request of a loaded model (`"sampling": null` clears it), and
  `POST /v1/admin/config/reload` applies an edited `sampling` to a loaded model, listed in the response's new
  `applied_live`. `GET /v1/admin/models` shows each text model's effective profile, with the source of every value, in
  a new `sampling` field (also before the first load), and every load logs a `sampling profile` line. Shipped configs
  write `"sampling": null` (no operator layer). See CONFIG.md "Sampling profile".
- **`rv`: sampling flags and the profile line.** `rv add` and `rv patch` take `--temperature`, `--top-p`, `--top-k`,
  `--repetition-penalty`, `--presence-penalty` and `--frequency-penalty`; `rv patch --clear-sampling` drops the
  configured values. They are range-checked before anything is sent, and `rv patch` keeps the configured values you do
  not name (it reads them first and sends the whole object, printing `sampling: <before> → <after>`). `rv check MODEL`,
  the new `rv status MODEL` and `rv add` print the effective profile with the source of every value, a hint for a model
  that runs greedy because its `generation_config.json` gives no sampling settings, and `values the client sends win ·
  realtime: profile`; for a model on disk that is not registered yet, the profile comes from its own
  `generation_config.json` (a local preview, not shown for a directory whose files show a non-text kind). Against a
  server older than 0.9.1, `rv add` and `rv patch` say `server < 0.9.1: sampling settings NOT applied` (`rv patch`
  sends nothing), and `rv check` / `rv status MODEL` say the profile is not reported. `rv reload-config` lists the
  models whose `sampling` was applied live.
- **Realtime: `"tts_model": ""` turns speech off for a session.** Replies then arrive as `text` events only, and
  `model_selection` reports `tts_source: "off"`. Omitting `tts_model` is unchanged: a loaded TTS model is used. A
  session with speech off (or any empty model id) never sets the voice-flow pin, so other clients reading the pin are
  not given an empty TTS model.
- **`REALTIME.md`**: a full reference for the `/v1/realtime` WebSocket protocol — frames, every client and server
  event with its fields, model resolution, history and retrieval, the order of one turn, limits, how a session ends, and
  worked examples. Linked from INTERFACE.md.
- **`max_images_per_request`** (default 16) and **`max_image_pixels_per_request`** (default 64 MP): limits on the images
  one chat request to a vision model, or one `/v1/images/edits` request, may decode. See Fixed.
- **`max_image_pixels`** (default 4 000 000): the per-image downscale cap for chat images to a vision model (C28). See
  Fixed.
- **`max_audio_duration_secs`** (default 7200, 2 h): the longest audio upload `/v1/audio/transcriptions` and
  `/translations` decode. See Fixed.
- **Process and host memory gauges on `/metrics`** (Linux): `rustedvino_process_resident_memory_bytes` (`VmRSS`),
  `rustedvino_process_rss_anon_bytes` (`RssAnon`, the figure the system-RAM admission uses),
  `rustedvino_host_memory_available_bytes` (`MemAvailable`) and `rustedvino_host_memory_total_bytes` (`MemTotal`),
  read at scrape time. Absent, not 0, where the OS cannot measure them (Windows). See METRICS.md.

### Changed
- **Kokoro-82M is temporarily no longer listed as supported** (README). Its English speech is garbled because of the
  server module that turns text into speech sounds for it: the `kokoro-tts` library drops some sounds before
  synthesis (among them the "wh" in *which*, *what*, *when*), so "which" is heard as "ich". The model still loads and
  answers; use SpeechT5, Piper or Coqui VITS meanwhile. A fix that replaces that module is planned for a later
  release, and Kokoro returns to the supported list once it passes.
- **Error messages for a model that cannot be found or is not loaded** (codes and statuses unchanged):
  - a model that is registered but not loaded (`503 model_unavailable`): "the model is registered but not loaded;
    an administrator can load it, or set it to load on demand" (was "model is not loaded");
  - an unknown id (`404 model_not_found`): "not found among the registered models; check the model id" (was
    "check /v1/models for available models");
  - `GET /v1/models/{id}` for a registered model that is not listed: the same `404 model_not_found`, with the
    not-loaded sentence.
- **`rv list`, `rv ps` and `rv completeness` list models by kind** (text, vision, embedding, reranking, speech-to-text,
  text-to-speech, image, decision) and then by name; the server's own order was arbitrary and could change between
  calls. `rv status` (its models table) and the `rv metrics` speed lines use the same order, and `rv list`'s on-disk
  directories are alphabetical. `--json` output is unchanged.

### Fixed
- **`rv load` / `rv check` say which limit refused a load — the GPU budget or system RAM — and name `total_vram_gb`
  / `system_ram_budget_gb` when the config's value is the limit; "nothing more can be evicted" no longer appears
  when nothing is loaded.** A 35B model refused on a 30 GB laptop with nothing loaded read as a hardware limit
  with causes (pinned, grace window) that did not apply; the real limit was the config's 18 / 17 GB caps.
  `rv check` also shows the reserve system RAM must keep free (`system_ram_reservation_gb`), and on a refusal no
  longer lists the loaded models under "would evict:" (evicting all of them would still not be enough; it now says
  so). `--json` and exit codes are unchanged.
- **A load whose memory was not measured said nothing.** `vram_measured_mib`, `kv_pool_resident_mib` and
  `gpu_memory_estimate_gb` are left out after a load that started within 20 s of an eviction (by design: the kernel
  frees an evicted model's memory late), with no log line, even at debug. The server now logs
  `load memory not measured` at debug level with its `reason` (METRICS.md, "Measured device memory").
- **The Linux bundle lacked METRICS.md**, which CONFIG.md and INTERFACE.md link to. It is now in the bundle.
- **METRICS.md described `rustedvino_tokens_per_second` as a decode-speed average on every engine.** That holds for
  text models on a GPU or the CPU only. On vision and NPU models it is the last request's output tokens divided by
  its whole time, prompt processing included, so on a long prompt it reads well below the real generation speed
  (measured: 4.9 tok/s reported against 10.5 tok/s generated, on a 2,029-token prompt). The figure is unchanged; the
  documentation now says which one you get.
- **A chat template saved with a byte-order mark or Windows line endings** (Notepad, PowerShell `Set-Content`) put
  U+FEFF at the start of every prompt and `\r\n` at every line break, unnoticed. The `chat_template` override and a
  model's `chat_template.jinja` are now read with the mark dropped and CRLF turned into LF. A template embedded in
  `tokenizer_config.json` loses a leading mark too, but keeps its line endings (a `\r\n` there is written into the
  JSON on purpose); that file now also loads when it starts with a byte-order mark (before, its template and its
  `eos_token` / `bos_token` were not read at all).
- **One long log event could appear as several lines in `GET /v1/admin/logs/tail`** when stdout took it in parts (a
  console writing in bounded chunks); the tee now writes each event whole.
- **A hung `espeak-ng` (Piper voices) blocked the speech engine for good.** Each run is now stopped after 10 s and the
  request fails with an error naming the limit; the missing-program hint no longer names one Linux package manager.
- **`/v1/systemone` accepted a `keep_alive` duration too large to represent** (a huge digit string wrapped to 0); it is
  now refused like any other bad duration, as Ollama does.
- **`POST /v1/admin/models/add` ignored unknown fields silently**; it still ignores them, and now logs a WARN naming
  them, so a typo (`pinnned`) is visible in the log.
- **The `rustedvino_key_usage_total` HELP text no longer names an internal function** (it said where in the code the
  key suffix is made); it now says only that the full key never appears in a metric.
- **A crash respawn came back with the config preload instead of the models that were loaded** (2026-11-16 live
  crash). The respawned worker loaded every `preload` model first and only then restored the crashed worker's runtime
  models, never evicting: a preload model the operator had swapped out at runtime (qwen3.5-9B unloaded, qwen3.5-4B
  loaded) filled memory again, and the restore of 4B and e5 was skipped. The supervisor now learns every model loaded
  at the crash, `preload` models included, and the respawned worker loads only those: `preload` models in its preload
  phase (one not loaded at the crash is skipped, INFO), the rest restored after listening as before; models whose KV
  pool is sized to the room left load last. A model evicted by a load (or a KV resize / KV-wedge reload) that the crash
  interrupted, or that failed just before the worker left without its teardown (a poisoned GPU), counts as loaded. A crash during a worker's own first preload still respawns with the full `preload`.
  Preload models have no crash-loop limit (as before).
- **A preload list over the budget evicted an earlier preload** (2026-11-16, 16:51 restart: whisper's preload evicted
  bge, the first in the list). Every preload load now never evicts: one that does not fit is skipped with a
  `preload_skipped` startup warning naming the shortfall in GB, the earlier preloads stay loaded.
- **One large image could OOM-kill the server on a UMA box** (C28). A 12 MP photo was passed to the vision encoder at
  full size; its memory grows with the image (qwen3.5-9B on an Arc 140V iGPU: 1 MP ≈ 0.65 GiB, 2 MP ≈ 1.15 GiB,
  4 MP ≈ 3 GiB, 8 MP out of GPU memory). Each decoded chat image over `max_image_pixels` is now resized (an
  integer area-average for a 2× or larger step, then bicubic; aspect ratio kept, never over the cap) before it reaches the engine; one at or under it is passed on untouched, without
  a copy. `max_image_pixels_per_request` still counts the original sizes.
- **A vision request held its decoded images while it waited** (S5c). Images were decoded before device admission and
  the engine-slot wait (up to the queue timeout), so every waiting request pinned up to ~192 MB of RGB, with no bound on
  the number of waiters. The header checks (count, per-request pixels, not an image: the `400`s) still answer before any
  wait; the decode now runs only once the request holds its engine slot. Requests already admitted and queued behind
  the running one (at most the engine's concurrency cap) still hold theirs. A body that is corrupt past a valid header
  is now a `400` after the wait instead of before it. A waiting request holds its images once, compressed (the
  `image_url` strings are freed after the header check): at most about the 50 MiB request-size limit per waiter.
- **A long answer no longer cuts a text model's context limit for good.** When the only request on a model ran out of
  KV-pool room after it had started answering, 0.9.0 read it as a sizing miss: it evicted and reloaded the model and
  saved 0.95 × that prompt as a learned limit that survived restarts. Admission reserves room for the prompt only, so
  the answer is what filled the pool. The client still gets `503 pool_exhausted` (same type and code), now with a
  message that says so: "the answer filled the model's KV cache after N generated tokens; retrying the same request will
  fail the same way — lower max_tokens or shorten the prompt" (JSON body and SSE error frame; every other
  `pool_exhausted` keeps "momentarily at capacity"). The server logs one WARN (prompt and generated tokens, "KV pool
  filled from output growth") and neither reloads nor learns a limit. Running
  out of room before the first token keeps the 0.9.0 recovery. The raw engine text, which the realtime `error` event
  (`stage: "llm"`) and `/v1/admin/health/generate` pass through, now ends in `generated_tokens=N`.
- **The first vision request after an idle gap no longer decides on an old device-memory reading.** On a discrete GPU
  the device-headroom limit of a vision request (chat completions and a realtime turn) used the last reading of what
  other processes hold, however old (876 s seen on a B50), and refreshed it only afterwards. A reading older than 30 s
  is now taken again first, on a worker thread, so that request waits once for the scan (about 0.1–0.2 s); concurrent
  requests share one scan per device. The wait is capped at 2 s: past that the request decides on the cached reading
  (one WARN), and a hung scan is never started twice. A probe that returns nothing is retried at most once per 30 s.
  Within 30 s nothing changes. Integrated GPUs are not scanned, and loads already
  read the device fresh.
- **SpeechT5 no longer drops text past about 600 characters.** 0.9.0 spoke the first ~600 characters and returned
  `200` (inputs of 900 and 1,339 characters both gave the same 33.7 s WAV). The server now splits SpeechT5 input, after
  normalisation, into pieces of at most 200 characters (sentence ends, then clause breaks, then spaces), synthesizes them
  in order and joins the audio. Text of 200 characters or less reaches the engine unchanged. Every successful SpeechT5 response carries
  `X-RustedVINO-TTS-Segments` (the number of pieces) and `X-RustedVINO-TTS-Segment-Cap: 200`, and the server logs one
  INFO line per request. Both headers are in `Access-Control-Expose-Headers`. A split request is admitted once and
  holds one TTS slot until its last piece is done. Kokoro, Coqui VITS and Piper are unchanged. See INTERFACE.md.
- **Realtime: a session no TTS model can serve is silent, not noisy.** When the TTS slot resolves to a model that is
  neither loaded nor loading (no TTS available, a `realtime_defaults.tts_model` that does not fit, a named model without
  room, an unknown id), `model_selection` now reports `tts_model: ""` (still a string), `tts_source: "substituted"` and
  the new `tts_reason: "none_available"`, and replies are text only. 0.9.0 kept the unusable id, so every sentence
  logged a WARN and counted in `rustedvino_realtime_tts_sentences_total{result="dropped"}`. A default that cannot load is
  no longer stored for the session; the next `config` tries it again (a model the client named must be named again).
  A TTS model already being loaded for someone else now counts as loading for the session (`model_loading` `started`,
  watched like its own load). STT, LLM and `request_model_load` are unchanged. See REALTIME.md. (C1)
- **Realtime: a TTS model whose background load fails no longer drops every later sentence.** When the TTS load a
  `config` started fails, the session's TTS id is cleared to `""` as soon as the failure is seen, with one WARN naming
  the model and the session; replies are then text only, without a WARN and a dropped count per sentence. The
  `model_loading` events are unchanged (`failed` still comes when the 90 s watch is up) and no new `model_selection` is
  sent; the next `config` resolves the slot again. (C1b)
- **A SpeechT5 `/v1/audio/speech` request is limited to 50 pieces.** Input that splits into more than 50 pieces after
  normalisation (about 10,000 characters, ~8 minutes of speech) is refused with `400 input_too_long`
  (`invalid_request_error`), the message naming the limit and the piece count, before it is queued or synthesized.
  The limit is a fixed constant, not a config key. Without it, 4096 digits (read digit by digit, ~125 pieces) held the
  engine for an estimated 15–20 minutes. Exactly 50 pieces is accepted; other engines and the realtime flow are
  unchanged. (T5)
- **`vram_measured_mib` and `gpu_memory_estimate_gb` no longer assume the KV pool is resident when a load ends.** When
  the warm-up does not allocate the pool (it is off, it fails, or the machine allocates the pool only at the first
  request), 0.9.0 reported the weights low by the missing part of the pool and the estimate without it. A 27B vision
  model on a B70 showed 7,560 MiB of "weights" against about 15 GiB real, and an estimate of 14.8 GiB against 22.3 GiB
  with its pool. The server now reads device memory after the engine is built and before the warm-up: the weights are
  measured directly, and the estimate (admin, `rustedvino_model_gpu_memory_estimate_bytes`) includes the whole booked
  pool. New admin field `kv_pool_resident_mib`: the part of the pool that became resident during the load. This holds
  only for loads that can be split that way. Measured as before, with figures still low by a pool that is not
  resident yet: a draft-attached text model; a text model whose pool the warm-up did not touch (warm-up off or
  failed); noisy readings; CPU and NPU. Fixed in code, not yet verified on hardware.
- **`/v1/rerank` can no longer abort the server with a double free** (known issue in every 0.8.x and 0.9.0 release).
  An upstream OpenVINO GenAI bug (2026.2.1 and 2026.4.1) corrupts the heap inside `TextRerankPipeline::rerank` when
  the query is longer than the documents and about three or more documents are scored in one call; the worker died
  with `double free or corruption` (SIGABRT) and the supervisor respawned it. The server now scores each document
  with its own pipeline call and sorts the results itself. The response is unchanged: the same results in the same
  order with the same scores (within float rounding); documents with equal scores are listed by their position in
  the request. A request takes longer: on a CPU, 20 short documents went from 10.4 s to 11.8 s. A large rerank is now
  scored document by document and can take long: on a CPU, 20 documents with a ~2,500-word query took about 7 minutes.
- **Models loaded at runtime come back after a worker crash** (`--supervise`). 0.9.0 respawned a crashed worker with
  only its `preload` models; a model loaded later (e.g. with `rv load`) stayed `not_loaded`, and an `eager` model then
  answered `503 "model is not loaded"` until an operator loaded it again. Now the respawned worker reloads the `eager`
  models that were loaded when the worker crashed, once it is listening, one at a time, with their config settings
  (the KV/stream overrides of the earlier load are not kept). Until its restore finishes, such a model still answers
  503. A restore never evicts a model: one that does not fit, or fails, is skipped with a WARN and a `restore_failed`
  startup warning (cleared if the model is loaded later, also when an admin load of the same model ran during the
  restore and won the race); a model that no longer exists is skipped with a WARN. A model
  unloaded before the crash does not come back, nor does anything after a clean stop or restart. To avoid a crash
  loop, a model is not restored again when its restore did not finish, when the worker crashes again while the
  restored model is loaded within `supervisor.restart_window_secs` of the restore, or when it was already restored
  within that window.
- **The Linux bundle ships `templates/`** (0.9.0 known issue): CONFIG.md said the corrected LFM2 chat template ships at
  `templates/lfm2.jinja`, but the bundle had no `templates/`, so a config following CONFIG.md pointed at nothing and LFM2
  ran on a generic template with tool calling degraded. The bundle step now aborts if a template CONFIG.md names is
  missing.
- **An unreadable `chat_template` override refuses the load**, as CONFIG.md always said, instead of falling back to a
  generic template with a WARN. It is checked before anything is reserved or evicted; the error (admin load `404`,
  `rv load`, the preload log and its `preload_failed` startup warning) names the model and the resolved path. A resize
  of a model whose override has become unreadable is refused before the model is unloaded. A model with no override and
  no template of its own still gets the generic template with a WARN.
- **`reload-config` says when a loaded model's edit was not applied.** It never changes a `Ready` model, and an unload
  plus load alone still used the entry the model was loaded with, so an edited `chat_template` silently did nothing. The
  response gains `models_not_applied` (`[{"model", "fields"}]`), the server logs a WARN, and `rv reload-config` prints
  the steps that apply it: `rv unload`, `rv reload-config`, `rv load`.
- **Realtime: every close the server begins sends a WebSocket close frame** (0.9.0 known issue). The idle timeout,
  an admin force-close and a server stop dropped the connection, so the client saw a reset (1006) and could not tell it
  from a crash. Now: idle timeout 1000 `idle_timeout`, server shutdown or restart 1001 `server_stopping`, admin
  force-close 4000 `closed_by_admin`; the server then waits up to 2 s for the client's close reply. A stop cancels the
  turn in progress and waits up to 3 s for the sessions to close before the models are torn down. The server also
  answers a close the client begins (0.9.0 never wrote that reply, so the client saw a reset instead), and a pending
  `model_loading` wait no longer keeps a closed session's connection open for up to 90 s. The 5-minute idle timeout is
  now documented (INTERFACE.md, REALTIME.md).
- **Realtime: a cancelled turn no longer enters the history.** A turn cancelled after some reply text was generated
  (by `cancel`, a hang-up or an admin kill) was still added to the rolling LLM context and the retrieval pool, and sent
  as `turn_history`, so the next turn's LLM saw a reply the user had stopped. Cancelled turns are now recorded nowhere;
  the `text` events already sent stay sent. A cancel that lands while the last sentences are spoken also counts.
- **Realtime: the server no longer gives a session an NPU text model it cannot run.** When a client did not name an
  LLM (or the one it named could not be served), the server could pick a loaded text model on the NPU, and every turn
  then failed with an `llm` error, since realtime turns have no NPU text path. NPU models are now skipped for the LLM
  slot, and a `realtime_defaults.llm_model` on the NPU is ignored with a warning. Naming one explicitly still gets it.
- **Realtime: when the client hangs up, the session ends.** The turn in progress was already cancelled, but up to 4
  utterances still queued behind it each ran as a full turn (STT, LLM, TTS) with nobody listening, and the connection
  stayed open until they finished. Queued utterances are now discarded at the close.
- **Realtime: the barge-in phrase matches whole words only.** It used to match anywhere in the text, so phrase "stop"
  also matched "Stopwatch" or "it stopped", and phrase "no" matched "Now." — Whisper noise could trigger a barge-in once
  short phrases were checked (fix above).
- **Realtime: a barge-in phrase of one or two words, such as "stop", now matches.** Since 0.6.8 the server dropped every transcript
  under 3 words as likely Whisper noise before checking it for the phrase, so a short phrase spoken on its own gave a
  skipped turn. The phrase is now checked first. Still a limit (fix planned for 0.9.2): the phrase is acted on only when
  its own queued turn starts, after the reply it should stop; send `cancel` to stop a reply.
- **Realtime: history entries sent in `config.history` are embedded, and entries from another embedding model are
  re-embedded.** 0.9.0 computed missing embeddings for the pool only once, at connect, before a client's `config` could
  arrive, so `config.history` entries without vectors (and `inject_history`/`reload_history` batches sent before the
  embedding model was in use) were never retrieved; entries embedded by a different model were skipped by retrieval for
  good. The pool is now checked at connect and at every turn start; each entry embedded is echoed as `turn_history`
  (`turn_id: null`). A vector is never applied to an entry that replaced its own while the batch was being embedded.
  The session embeds in batches of 32 texts (it used one batch for the whole pool, outside the `/v1/embeddings`
  limits); a failed batch is logged, and the pool back-fill then pauses for 5 turns.
- **Realtime: a connected session no longer blocks eviction of its embedding model.** The session kept one
  embedding handle for its whole life, which kept the engine thread alive: an eviction then waited out the 8 s join
  timeout and freed the model's memory in the tracker while the thread still held it, and the kept handle failed on
  every later call, so retrieval stayed off for that session even after the model was loaded again. The session now
  keeps only the model id (still fixed for the connection) and fetches a handle for each use.
- **Realtime: the retrieval pool is capped at 200 entries for completed turns too.** `config.history`,
  `inject_history` and `reload_history` already kept the newest 200; each completed turn was appended without the cap,
  so a long session grew the pool (and its copy in the admin snapshot) by one entry per turn.
- **Realtime: a turn whose LLM stream ends without a finish now ends with `error` (`llm`) and `done`.** 0.9.0 sent no
  `done`, did not speak the last buffered sentence, and recorded the cut-off reply as a completed turn (`turn_history`
  and the history). The engine closes a stream that way when it drops a request it cannot deliver to (a full token
  channel), or when its final events are lost to one.
- **Realtime: `request_model_load` no longer answers `ready` for a model that is not loaded.** 0.9.0 did so when the
  model was the slot's default and did not fit, and for an id the server does not know (with no default and nothing
  loaded for that slot); the next turn then failed. The reply is now `failed` with the reason code. An unknown id
  always gets `failed` with `unknown_model` and no longer starts a load of the slot's default; a model already loading
  gets `started`.
- **Audio uploads are bounded in memory** (0.9.0 known issue). The request-size limit bounds only the compressed file;
  a very compressible one decoded without limit. Decoding now stops at `max_audio_duration_secs` of audio with a
  `400 invalid_audio` that names the limit, before decoding anything when the file declares a longer length, and inside
  the decode loop when the declared length is absent or wrong (a test covers an MP3 whose header-based estimate is a
  quarter of its real length). Worst case at the default: about 1.4 GB + 0.5 GB.
- **Image inputs are bounded in memory** (0.9.0 known issue). Each image's size is now read from its header, and a
  request over `max_images_per_request` or `max_image_pixels_per_request` is a `400` (`too_many_images` /
  `image_too_large`) before any pixel is decoded; decoding then runs with explicit limits from that size. Before, a
  request of a few KB per image could declare 65535 × 65535 pixels per image, with no limit on the image count.
  Images are also decoded off the async runtime now, two requests at a time server-wide: a large decode (up to about
  1 s of CPU for 64 MP) used to stall every other request, `/health` included.
- **A crafted M4A/MP4/MOV of under 100 bytes could abort the server**: the demuxer allocates before reading, from sizes
  the file declares (a sample table of 2^32 entries; a metadata box with a 64-bit size of 2^50 bytes; a PCM sample
  entry of 2^32 frames per packet). Such files are now refused before decoding: impossible table counts and box sizes
  are checked before the demuxer runs, and frames per packet before the decoder is built.
- **Realtime: a short gpt-oss reply is sent as `text`, not only spoken** (0.9.0 known issue). With gpt-oss the first
  10 bytes of a reply are held back to strip a leaked channel name; a reply that never grew past them ("Paris.", "8.")
  was flushed at the end of the turn into TTS only, so the client got audio and no `text` event, and the turn log read
  `llm_tokens=0`. Text the reasoning filter still held at the end of a turn had the same gap (any model). Both now go
  through the same path as streamed text. The turn log also counts the pieces held back on longer gpt-oss replies, and
  its `llm_ttft_ms` is the first content piece rather than the moment the held text was released.
- **A vision load on a discrete GPU decides on the device memory of now** (0.9.0 known issue). The memory other
  processes hold was read once and then reused, so a load after an idle gap was admitted on the reading of the previous
  load (876 s old in one log). Loads, their eviction plan and `rv check` now take a fresh reading when the cached one is
  older than 2 s, on the blocking pool; requests keep the cached reading and refresh it in the background, so they
  never wait for the scan (about 0.1-0.2 s). The WARN "device-memory probe returned no reading" now means that the probe
  really returned nothing; an idle server no longer logs "probe keeps failing". Only discrete GPUs are scanned.
- **Admin keys no longer count as authentication for the open-bind check** (0.9.0 known issue). See the upgrade note.
  The refusal now exits 78, so a supervisor does not restart it in a loop. `rv reload-keys` that removes the last
  `api_keys` entry on a network bind still applies (a key being revoked must not stay valid), logs a WARN, and answers
  `inference_open_on_network: true`, which `rv reload-keys` prints as a warning.
- **`/v1/realtime` no longer reads a vision model's reasoning aloud.** A vision model in the LLM slot now renders with
  thinking switched off, as text models already did (0.9.0 known issue). Qwen3.5-9B in a Vietnamese test: the turn took
  2.5–2.8 s instead of about 6.6 s, and the voice gave the answer directly instead of 15–19 s of English reasoning.
- **A vision model preloaded at boot on a discrete GPU gets the same context limit as a later load of it** (seen on a
  B70 with 0.9.0). The device-memory probe was attached only after the preload had run, so every boot-preloaded vision
  model on a discrete GPU took the legacy estimate and logged "the exact limit needs the device-headroom cap, which
  this build does not have": a 27B got 51,526 tokens at boot and 180,781 after an unload and reload. The probe is now
  in place before the preload, so a boot preload is sized and limited exactly as a load at runtime in the same memory
  situation (`kv_gate_mode` `exact`, with the device-headroom limit). New at boot: the device-headroom admission rule
  runs for preloads too, refuse-only (see the upgrade note): it never evicts for that rule, and waits for device memory
  to be released only after the VRAM tracker has evicted for the same load; loads after boot are unchanged. Where the device memory cannot be read,
  the legacy limit stays, and its one-time warning now says why (the probe could not read the device memory, or
  `kv_headroom_mode` is off) instead of claiming the build lacks the cap. The limit still depends on what else is
  resident: it is the free device memory left for one prompt, and it rises again when the other model is unloaded.
- **On an Intel iGPU (shared memory), unloading a model is credited the KV pool it really holds, and `rv resize` no
  longer charges a shrink fresh RAM for its KV cache** (0.9.0 known issue). The live system-RAM gate credited an
  eviction with the model's weights only, as if every KV pool were allocated lazily, although the KV warm-up (on by
  default since 0.9.0) allocates the whole pool at load. On a 30 GB laptop that undercounted lfm2-24b's eviction by
  its 6 GiB pool, and the resize check also charged the new KV as fresh RAM, so even a shrink needed the new pool's
  size free (6 GiB → 3 GiB refused with 2.9 GB available). Now a pool counts as resident when the load measured it
  (`kv_pool_resident_mib`) or else when its warm-up succeeded; a pool whose warm-up was off or failed, or a
  draft-attached model, keeps the old weights-only credit. Loads, the eviction plan (`rv check`) and resizes use the
  same credit; CPU models are unchanged. A resize credits the model's own resident pool up to the new size: a pool
  that is not resident earns nothing, and the safety margin (`vram_safety_margin_gb`) is still charged, so with
  nothing else to evict and less than the OS floor plus the margin available a shrink is still refused. The eviction
  preview of a load with `--kv-cache-gb` (and of a resize's reload) now charges the requested KV, not the model's
  configured size. The resize check caps what unloading can give back at the machine's RAM (`MemTotal`), as a load
  does. When a resize's reload is still refused for memory, the server tries once to reload the model at its
  previous KV size (evicting nothing more); the error then contains `resize_rolled_back`, or `resize_not_restored`
  when that failed too and the model stays unloaded. This rescues a grow and a reload refused by the engine; it does
  not rescue a shrink refused by the live system-RAM gate on a shared-memory box, which means the check credited a
  pool that was not resident: the old, larger pool cannot fit either, and the model stays unloaded. That residual
  risk rests on a warmed-up pool being resident on an iGPU, which was measured on 0.9.1-rc.2 (`kv_pool_resident_mib`
  against the pool booked at load): Intel UHD (Tiger Lake), qwen3-1.7b-int4 2047 of 2047 MiB and qwen3.5-4b-int4
  (vision) 2033 of 2045 MiB; Arc 140V (Lunar Lake), Qwen3.8-27B-int4 (vision) 4075 of 4095 MiB. The model is
  briefly not loaded before the rollback (a concurrent load of it can make the rollback report
  `resize_not_restored` while the model comes up anyway), and the resize cooldown stays claimed after a rolled-back
  resize (an immediate retry gets `429`). See INTERFACE.md. A resize refusal and a load refused for memory now name the gate that refused: the live
  system-RAM gate with the RAM available now, what unloading would give back and the part kept free for the OS, or
  the tracked-VRAM gate with its figures. 0.9.0 printed the tracked-VRAM figures for a load the RAM gate refused
  ("needs 18.0 GB, 23.3 GB free"). RAM refusals say when the KV is charged rounded up to whole GiB (3.5 GiB counts as
  4.0 GiB). The unload credit, the resize check and the rollback themselves are fixed in code but not yet exercised
  on hardware.
- **`rv ps` / `rv status` show `KV%` as `n/a`, not `0%`, for a model whose KV occupancy cannot be measured.** The VLM
  engine has a real pool but no API to read its occupancy, so its percent is always `0` and
  `rustedvino_kv_cache_usage_supported` is `0`; 0.9.0 ignored that flag and printed `0%` with 160k tokens in the pool.
  `GET /v1/admin/models` gains `kv_cache_usage_supported` (bool) beside `kv_cache_usage_pct`, so `rv ps --json` and
  other API clients can tell the two apart too. Models with no KV pool (embedding, speech, image, an NPU text model)
  also report `0` there and now show `n/a` instead of `0%`; so does a model that is not loaded (`rv status`). A text
  (CB) model still shows `0%` while idle: its gauge moves only while a generation is running. The Python
  `scripts/rv` follows the same rule.
- **Text-to-speech reads no-break and other Unicode spaces as spaces.** LLMs write typographic spaces such as
  U+202F (narrow no-break space; gpt-oss writes `328\u202ffeet`) and U+00A0. `/v1/audio/speech` passed them to the TTS engine
  unchanged when `normalize` was `false` or the language has no number words (anything but English, Polish and
  Vietnamese). Every Unicode space separator (U+00A0, U+1680, U+2000–U+200A including U+2007 figure space and U+2009
  thin space, U+202F, U+205F, U+3000) is now a plain space before the text reaches any engine (SpeechT5, Kokoro,
  Piper, Coqui VITS), on `/v1/audio/speech` and in realtime sessions, before the SpeechT5 splitter and the number
  normaliser. Line and paragraph separators and zero-width characters are left alone. With normalisation on, the
  realtime flow and `/v1/audio/speech` already read these as spaces.
- **An eviction right after a chat or completion, speech-to-text, text-to-speech, embedding, reranking, image or
  decision request no longer finds the model busy.** The engine answered the request before it released the request's slot, so an `rv unload` (or a
  load's eviction plan) issued the moment the answer arrived could be refused with "model has 1 request(s) running".
  Every request/response engine (Whisper, SpeechT5, Kokoro, Coqui VITS, Piper, embedding, reranking, image and the
  decision engine) now releases the slot first. The streaming engines (continuous batching, vision and NPU) now
  release it before the final event (the end of the stream, or its error) reaches the client, streaming or not.
- **The generation watchdog no longer waits forever on a worker that accepts its probe and never answers.** Each
  probe of `GET /v1/admin/health/watchdog` (connect, request, response) is now bounded at 2 s, as the startup health
  probe already was; a timeout is logged like any other unreadable probe (one ERROR while the watchdog is blind, one
  INFO when it can read again). Restart policy is unchanged.
- **A vision request with no answer budget now gets the KV pool's room as its budget (vision on discrete GPUs).** With
  `max_tokens_cap: 0` and no `max_tokens`, 0.9.0 passed a budget of 0, which leaves the engine's generation config at
  its own default length rather than a bound the server chose: it skipped the pool clamp, and the watchdog allowed it
  prompt time only, so a legitimate answer longer than about 10 minutes (`watchdog_hang_ceiling_secs`) got the worker
  killed. When the vision gate is exact (a vision model on a discrete GPU whose free memory the server can read, where the pool and prompt sizes are known),
  the budget is now the room the pool has left after the prompt, and the watchdog allowance counts it at 5 tokens/s; a
  prompt that leaves less than 16 tokens of room is refused with the same `400 context_length_exceeded` as a
  `max_tokens` that does not fit. **Not fixed on shared-memory GPUs** (integrated GPUs such as Lunar Lake), on a discrete GPU
  without a readable free-memory probe, or anywhere else the vision gate is legacy (`kv_gate_mode: "legacy"`, a model the server sizes with the legacy estimate): there
  the budget stays 0 and the watchdog allowance prompt-only, as in 0.9.0. A request with `max_tokens`, a non-zero
  `max_tokens_cap`, text (continuous batching) and NPU models are unchanged. Such a request now also gets the KV-wedge
  check every bounded vision request has: when the engine returns without decoding a single token, the client gets the
  KV-wedge error instead of an empty answer, and the model goes through the same wedge recovery as any bounded vision
  request.
- **Docs: the 0.9.0 upgrade note about `POST /v1/admin/models/add` was wrong.** It said the route answers JSON; a
  normal success is plain text, as before, and JSON (`{"message", "persisted": false, "persist_error"}`) comes back
  only when the model loaded but saving `config.json` failed. The code is unchanged; the 0.9.0 note is corrected.
  INTERFACE.md already described it this way.
- **Docs: README, INTERFACE.md, CONFIG.md and METRICS.md corrected where they no longer matched the server.** A
  deregister removes the model from `config.json` too (it was described as lasting until a restart); with
  `admin_api_keys` empty every `api_keys` entry has admin rights (set `admin_api_keys` to separate them); admin routes
  answer `409`, not `503`, for a model that is loading or being evicted; `/v1/models` lists loaded models only, with
  `max_model_len`; the accepted audio formats and both audio length limits; a `PATCH` cannot set `chat_template` or
  `load_timeout_secs`; a config reload does change a loaded model's `sampling`. New: a list of OpenAI parameters the
  server refuses (`400 unsupported_parameter`) or treats differently, a reference for the image routes, the load
  body's per-load overrides, more admin model fields and error codes, and *(since 0.9.1)* marks on what this release
  added. The README walkthrough now uses the starter models it downloads, builds `rv`, and lists devices with
  `rustedvino --list-devices`.
- **`rv --help` text:** `rv add --kind` lists `decision`; `--device` (and `rv patch --clear-device`) says the server
  places a model without one (the global `device` only with `default_placement: "device"`); sizes say GiB.
- **`rv metrics` could show another device's speed.** The server keeps a model's `/metrics` series from every device
  it ran on, with their last values; the summary line took the first one it found, so a model that ran on the CPU
  (decode 26.3 tok/s) and then on the integrated GPU (11.9 measured) showed the CPU's 26.3. The decode and effective
  prefill figures are now those of the device the model is on now, read from `GET /v1/admin/models`; when that
  cannot be told, rv shows each device's figure with its name. `--json` adds `device` and the per-device figures.
  `rv ps` and `rv status` had the same fault in their `KV%` and `RUN/WAIT/MAX` cells; they now read the model's
  current device too (`n/a` until that device has a reading).
- **`rv status` shows memory in GiB.** The server exports its memory metrics in GiB (0.9.0), but `rv status` still
  divided bytes by 10^9: its VRAM lines showed a 24 GiB budget as 25.8 and used memory about 7 % high. They now match
  `total_vram_gb` (24.0). The label stays "GB". The `rv add` `--vram-gb` estimate is unchanged.
- **`rv add` estimates `vram_gb` in GiB, with an explicit 1.07 margin.** It divided the model directory's size by
  10^9, while the server reads `vram_gb` in GiB; that divisor was the estimate's only (hidden) headroom, about 7.4 %.
  It now divides by 2^30 and multiplies by a stated 1.07, so the Rust `rv add` books almost what it did before: never more,
  0.35 % less before rounding (1.07 / 1.0737). Up to about 28 GB on disk that is within 0.1 (a 3.2e9-byte model: 3.2
  before and after; 19.7e9 bytes: 19.7 → 19.6); above it the gap grows (28.75e9 bytes: 28.8 → 28.6; 80e9: 80.0 → 79.7). Its output says GiB: the "auto-detected
  vram_gb" line (now "the size on disk plus a 7 % margin"), the "not counted" line, the confirmation prompt and the
  `--vram-gb` help. The Python `scripts/rv` makes the same change; it used 1.08 on the decimal size, so its estimates
  drop by about 7.7 % to the Rust figures (3.2e9 bytes: 3.5 → 3.2). The margin is revisited after a measured-footprint
  run. (R-12)
- **A GPU out-of-memory that reports only `CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST` (-14) is now a GPU poisoning.**
  Seen on the Arc 140V with an 8 MP image on a vision model: the exception that reached the server said only
  `clWaitForEvents, error code: -14 …` (the explicit `CL_OUT_OF_RESOURCES` went to OpenVINO's own stderr). The client
  got `500 inference failed`, a later unload tore the engine down on the dead GPU context, the destructor threw and
  the process aborted (SIGABRT), and about 13 GiB stayed unavailable for minutes (attributed to the GPU driver). It now
  answers like every other GPU poisoning: `503 model_unavailable` (with `Retry-After: 30` and "retry shortly" when a
  supervised worker is about to restart; the SSE error frame says the same), and under `--supervise` a serving model's
  failure now takes the same exit-75 path (no engine destructors, the supervisor restarts the worker) that released
  the memory within ~3 s for `CL_OUT_OF_RESOURCES`. A -14 during a model load or its KV warm-up fails that load and
  poisons without an exit, like the other markers: GPU models stay unusable until a manual restart.
- **Tool calls from Qwen3.5, Qwen3.6 and Qwen3-Coder returned booleans and null as the strings `"True"`, `"False"`
  and `"None"`**, and numeric-looking text in a string parameter as a number (`"10000"` arrived as `10000`). These
  models write Python literals in their XML tool-call arguments, and a client testing `if args["metric"]` got a true
  value for `"False"`. Each argument is now typed from the tool's schema in the request (`type`, an
  `anyOf`/`oneOf` of types, an all-string `enum`, and `"nullable": true`): a `boolean` gets `true`/`false`; `None`
  becomes `null` unless the parameter's types include `string` without `null`; a `string` keeps the text as written, except
  that one whole JSON string (`"Hanoi"` in quotes) is decoded as before; an `integer`/`number`/`object`/`array` is
  parsed as JSON, and an `object`/`array` also accepts Python syntax (`{'a': True}`) unless its strings use escapes
  other than `\\`, `\'`, `\"`, `\n`, `\r`, `\t` (then the text is kept as written). An argument the schema does not
  type turns a bare `True`/`False`/`None` into `true`/`false`/`null`. Other model families are unchanged.

### Known issues
- The full list, with workarounds, is in the
  [v0.9.1 release notes](https://github.com/Jermalk/rustedvino-oe/releases/tag/v0.9.1): among them, vision memory
  with many images in one request (set `max_images_per_request` to 4), Bielik-11B-v3's thinking showing in the
  answer text (template workaround in the notes), and Kokoro-82M's garbled English speech.

## [0.9.0] — 2026-11-06

Linux bundle only. Windows: not published in this release (the Windows changes are in the source).

### Highlights
- Runs on **OpenVINO GenAI 2026.4.1** (0.8.4 shipped 2026.2.1).
- The **context limit is computed from each model's real KV-cache layout**, and KV pools are sized in whole KV blocks.
- **Per-model KV-cache precision**, an automatic rule (u4 for 4-bit weights, otherwise i8), and a validated `kv_cache_precision`.
- **A device-headroom limit for vision models on a discrete GPU** (Linux): a long prompt no longer exhausts device memory and poisons the GPU.
- **A stop during a model load waits for the load**: no crash at exit, and no Arc Pro B60 left unusable until a GPU reset.
- **Model loads have a deadline** (600 s by default); a late load is booked as stalled and the server names the recovery step.
- **A poisoned GPU context is visible** on `/health`, `rv status`, the model lists and a metric; a supervised worker answers a retryable 503 while it restarts.
- **The supervisor's watchdog fires on servers with admin keys** and no longer kills long legitimate vision requests.
- **Hybrid Qwen3.5 / 3.6 / 3.8 models keep prefix caching on a GPU.**

### Upgrade notes from 0.8.4

**Required actions**
- **Set `TimeoutStopSec=720` in installed systemd units**, then `systemctl daemon-reload`. A stop now waits for a running load (up to 600 s) and then tears down (up to 8 s per engine; 720 s covers 14 models). With 120 s or less, systemd SIGKILLs the server mid-load, which is what leaves a B60 stuck (see Known issues). The bundle ships `rustedvino.user.service.example` with the right settings.
- **Every model recompiles once** (new OpenVINO version). Text and vision models now compile into `ov_cache_dir/kv-<tag>/`; 0.8.x blobs stay in the cache root until pruned. Set `ov_cache_max_gb`, or delete stale `*.blob` files by hand (the root still holds live Whisper, image, TTS and NPU blobs). With `ov_cache_max_gb: 0` the server logs how much is left behind. With pruning on (`ov_cache_max_gb` > 0), use one `ov_cache_dir` per instance, host and OpenVINO version.
- **`kv_cache_precision` is validated.** Accepted: `"auto"` (or `""`), `"i8"`, `"u4"`, `"f16"`; `"u8"` is a deprecated alias of `"i8"` (logged once at INFO). Any other value stops startup with the accepted list. `"f32"`, which 0.8.4 accepted and passed to the plugin, is withdrawn: a config that sets it must change it. Other spellings, which 0.8.4 ignored silently, now stop startup too.
- **If a discrete GPU is your global `device` and `total_vram_gb` is `0.0`, that GPU is now VRAM-gated** at the memory it reports. Models that used to over-commit it now evict each other, or are refused when they can never fit. Check `rv status` after the upgrade; set `total_vram_gb`, or opt out with `domain_budgets` (see Behaviour changes).
- **An unknown global `device` with `default_placement: "device"` stops startup with exit 78** (no model without its own `device` could load). With `"auto"` it stays a startup warning.

**Behaviour changes**
- The global `kv_cache_precision` built-in default stays `"u8"` (= i8; the shipped example config writes it as `"i8"`), so text models keep 0.8.4's 8-bit cache. Set it to `"auto"` to let each model's IR hint or the rule decide (for int4-weight models an explicit i8 is about 1.8× larger per token than u4). The global key never reached vision models (0.8.4 passed them nothing); they now get the per-model value, the IR hint or the rule.
- Vision models on a GPU (and text models under `"auto"`) now pass their resolved precision, so they compile into `kv-u4`, `kv-i8` or `kv-f16`; this is part of the one-time recompile. The precision itself matches what the plugin picked on its own.
- Model loads time out at 600 s (`504 load_timeout`); `load_timeout_secs: 0` restores 0.8.4's wait without limit.
- Context limits move: up for models the old estimate under-promised (hybrid models), down a few percent for dense int8 models it over-promised. Vision prompts now count the chat template and the `tools` schema, and `max_tokens` is lowered to what the pool can still hold.
- The memory accounting unit is the GiB: `rustedvino_vram_total_bytes` / `_used_bytes` read about 7 % higher (× 2^30, was 10^9) and `gpu_memory_estimate_gb` about 7 % lower. `total_vram_gb`, `vram_gb`, `kv_cache_gb`, `cache_size_gb`, `vram_safety_margin_gb` and `min_kv_cache_gb` are GiB; no admission arithmetic changed.
- Pools the server cannot size in KV blocks (CPU, HETERO/AUTO, `kv_precision_pass: "plugin"`, speculative drafts, an unverified version pair, a model it cannot describe) are booked at the whole GiB the engine allocates: up to just under 1 GiB more than a fractional `kv_cache_gb` in 0.8.4, with `max_model_len` rising to match.
- Learned context limits leave the model folder for a per-user directory. `.rustedvino_kv_ratchet.json` files in model folders are ignored (never deleted); each model relearns at its next KV overflow.
- An image-generation model on the CPU is charged its real working set, and a request above `cpu_image_gen_max_pixels` is a 400.
- `POST /v1/admin/models/add` answers a JSON body (`"persisted": false`, `persist_error`) only when the model loaded but `config.json` could not be saved; a normal success is still plain text (corrected in 0.9.1: this note first said every answer was JSON). `POST /v1/admin/config/reload` with an invalid `config.json` answers 422 (was 500).
- `rv start --supervise` with a broken config exits 78 (was 1).
- **`total_vram_gb: 0.0` no longer switches VRAM gating off for a discrete GPU set as the global `device`**: that GPU is budgeted at the memory it reports, as any other discrete GPU already was. As with any configured budget, a load that does not fit evicts idle models first (least recently used), and a model that can never fit is refused by every path (`rv add`, `rv load`, preload: a refused preload is skipped, logged and shown as a startup warning, and startup goes on). To keep the old ungated behaviour on purpose, set `domain_budgets` with the device name `--device-info` prints: `{"GPU": 0.0}` on a one-GPU box, `{"GPU.1": 0.0}` on a two-GPU one. CPU, integrated-GPU, NPU and composite (`HETERO:`…) inference devices are unchanged. An explicit `total_vram_gb` above what the device reports is now capped at the reported GiB (it was converted to decimal GB, about 7 % looser).

**New configuration keys**

| Key | Default | What it does |
|---|---|---|
| `load_timeout_secs` | `600` | Load deadline; `0` = none, otherwise at least `60`. Also per model. |
| `kv_gate_mode` | `"exact"` | `"legacy"` = the 0.8.x context-limit estimate everywhere (pools stay in blocks). |
| `kv_gate_exact_on_unknown_engine` | `false` | Use the exact gate on an unverified OpenVINO/GenAI pair. |
| `kv_precision_pass` | `"explicit"` | `"plugin"` = leave the KV precision to the plugin, as 0.8.x did. |
| `kv_warmup` | `true` | One-token warm-up after every GPU load of a KV engine. |
| `kv_headroom_mode` | `"dynamic"` | `"off"` disables the vision device-headroom limit. |
| `kv_transient_factor` | `2.0` | Multiplier on a prompt token's transient device-memory cost. |
| `kv_headroom_margin_gib` | `0.5` | Device memory kept free for processes the server cannot see. |
| `kv_headroom_min_tokens` | `4096` | Floor of the headroom limit; a load that cannot keep it is shrunk or refused. |
| `kv_headroom_inflight_tokens` | `32768` | Vision requests longer than this hold a device-memory reservation while they run. |
| `ov_cache_layout` | `"keyed"` | `"flat"` = the 0.8.x cache paths (also brings back precision mixing). Restart-only. |
| `cpu_image_gen_max_pixels` | `1048576` | Largest CPU image request (`n × width × height`); sizes the memory charge. |
| `models[id].kv_cache_precision` | `null` | Per-model precision; `"auto"` opts this model out of the global key. |
| `supervisor.gpu_probe_deadline_secs` | `15` | GPU readiness probe after a crash exit; `0` disables it. |
| `supervisor.gpu_first_load_deadline_secs` | `180` | Deadline for the first GPU load after a crash exit; `0` disables it. |

### Added
**KV cache and admission**
- **Exact context limit.** For a text model on a GPU with a known precision, `max_model_len` = min(95 % of the pool's token capacity, native context, a limit learned for exactly this pool), counting only attention layers, hybrid models' linear-attention state, the precision's byte layout and block rounding. On a plain GPU whose layout the server knows, pools are sized in KV blocks (`num_kv_blocks`) under either gate mode and booked as built: a requested 4.2 GiB books about 4.2 GiB.
- The legacy estimate stays for vision models on shared-memory GPUs, CPU and HETERO/AUTO devices, speculative-decoding models, an unverified OpenVINO/GenAI pair, models the server cannot describe with confidence, and `kv_precision_pass: "plugin"`. A pool too small for one block (or a hybrid model's per-stream state) is refused before anything is evicted.
- **Per-model KV precision** (`models[id].kv_cache_precision`: auto / i8 / u4 / f16), also via `models/add` and `PATCH`, from the next load. Resolution under `kv_precision_pass: "explicit"`: per-model > global (text only) > the IR's own hint > u4 for 4-bit weights, otherwise i8.
- **Device-headroom limit for vision models on a discrete GPU (Linux).** The context limit is also capped at `free / (hidden size × 4 × kv_transient_factor)` tokens, where free is the card's memory minus other processes (from `/proc/*/fdinfo`), every booking, running long requests and `kv_headroom_margin_gib`. It follows live free memory, shrinks or refuses a load that would break `kv_headroom_min_tokens`, and its 400 names the free memory.
- Without a readable probe (non-Linux, shared-memory GPU, `kv_headroom_mode: "off"`) vision models keep the legacy limit, with a one-time warning. Without this limit, a 27B vision model with a 6 GiB pool and 3 GiB of free device memory failed at a 164,960-token prompt.
- **Vision requests are admitted on the whole prompt**: about 5 tokens per message for the template, the `tools` schema at 125 % of its JSON tokens; token counting does not wait behind a running generation. `max_tokens` is clamped to the pool, keeping 1 % (at least 32 tokens) spare (`rustedvino_kv_max_tokens_clamped_total`); under 16 tokens of room is a 400.
- Images are not counted yet: a vision model that admitted such requests reports `kv_admission_low_confidence` and `uncounted_image_admissions`, and `rv ps` / `rv status` add `!` to its CTX cell. A too-long-prompt 400 names the limit and the knob that moves it.
- **KV warm-up** (`kv_warmup`): a one-token generation after every GPU load builds the pool at load time, so a broken GPU shows then. A GPU out-of-resources failure there fails the load and marks the context poisoned: the worker stays up degraded (no automatic restart) and the engine is not destroyed on the poisoned context.

**Load lifecycle**
- **Load deadline** (`load_timeout_secs`, default 600). A load past it answers `504 load_timeout` and keeps running booked as stalled: its memory stays reserved, loaded models keep serving, and new loads on that device get `503 device_stalled` until it returns (committed, or discarded on an error). A load whose client disconnects is booked the same way.
- Under `--supervise` a load still running after its deadline + `watchdog_hang_ceiling_secs` restarts the worker. A `models/add` past its deadline keeps the model registered but unsaved: `ready` if the load succeeds (a `PATCH` saves it), else `not_loaded` until deregistered.
- **A stalled load names its recovery step** past its deadline, in `rv status`, the watchdog body (`stalled[].recovery`) and the log: a GPU reset for a discrete Intel GPU on the xe driver, a reboot for other GPUs and the NPU, a server restart for the CPU. `503 device_stalled` points to `rv status`.
- **GPU readiness probe after a crash exit** (exit 75/76, a signal, a hang or deadline kill): the supervisor runs `rustedvino --gpu-probe` (tiny compile + inference per GPU, 15 s, up to 3 tries) before the next worker. That worker's first GPU load gets a deadline (`gpu_first_load_deadline_secs`, 180 s or the load's own if longer; with `load_timeout_secs: 0`, `watchdog_hang_ceiling_secs` if longer); past it the worker is respawned once.
- `models/add` of a model that loaded but could not be saved answers 200 with `"persisted": false` and `persist_error`. `models/add` and deregister refuse with `409 config_file_invalid` when `config.json` is already invalid.

**Observability**
- `GET /v1/admin/models`: the KV decision (`kv_cache_precision`, `kv_precision_source`, `kv_bytes_per_token`, `kv_profile_confidence`, `kv_gate_mode`, `limit_clamped_by`, `kv_pool_mib`, `kv_pool_blocks`, `kv_headroom_cap_tokens`), `kv_requested_gb` / `clamped_by`, `vram_measured_mib`, and `server.headroom` per discrete GPU (lease total for key-validated callers only).
- `rv ps` / `rv status` get `CTX` and `KV PREC` columns; `rv status` adds a headroom line (`off` / `unknown` when it has none), `Loads`, `Stalled` and `In flight` lines, and `Driver` lines.
- `/health` gains `gpu_poisoned` and `device_stalled` (still always 200). `GET /v1/admin/health/watchdog` gains `in_flight`, `stalled[]`, `oldest_ffi_overdue_secs` and the overrun beyond a request's allowance.
- Startup configuration warnings (unknown global device, unwritable cache or config paths, dynamic KV with prefix caching) are kept as `startup_warnings` and printed by `rv start` and `rv status`. GPU/NPU driver versions and the kernel release are logged at boot and shown in the admin server block (key-validated callers only).
- Metrics: `rustedvino_kv_gate_mode{model,mode}`, `rustedvino_kv_scheduler_drop_total{model}` (requests lost to KV-pool admission), `rustedvino_kv_max_tokens_clamped_total`, `rustedvino_gpu_poisoned`. See `METRICS.md`.
- `fill-config-defaults.py --check` reports keys missing inside the `supervisor` block.

### Changed
- **OpenVINO GenAI 2026.4.1.**
- **Hybrid linear-attention models (Qwen3.5 / 3.6 / 3.8) keep prefix caching on a GPU**, and `default_placement: "auto"` may place them on a GPU again. The 2026.2.x wrong-continuation bug is not present on 2026.4.1 (checked on an Arc 140V); the guard, its warning and the GPU ban are gone; per-model `enable_prefix_caching` stays an override. This also lifts 0.8.4's smaller context for hybrid vision models.
- **One compile-cache directory per KV precision** (`kv-auto`, `kv-i8`, `kv-u4`, `kv-f16`) for text, draft and vision models on GPU and CPU, because a 2026.4.x blob does not record the precision. Pruning covers the subdirectories and protects only what a running model uses. An unwritable keyed directory is skipped (no blob cache).
- **Learned context limits are keyed by the exact configuration** (pool, precision, prefix setting, streams, engine path, versions, device) in `.rustedvino_kv_ratchet.v2.json`; the legacy gate keeps the one-number file. Both live beside `ov_cache_dir` as `kv_ratchet/`, else in `~/.cache/rustedvino/kv_ratchet/`, so a model library can be shared and mounted read-only.
- **Measured-footprint learning** also runs on a discrete GPU for an exactly-sized pool; a load that starts while a generation runs is not used to learn a weight.
- A device spec that includes the CPU (`CPU`, `HETERO:GPU.1,CPU`, `AUTO:GPU,CPU`, `MULTI:…`) still gets the 8-bit cache as `u8`, which the CPU plugin requires (elsewhere it is passed as `i8`).
- **Input checks:** `models/add` refuses an unknown per-model `kv_cache_precision` (400); a wrong JSON type for `kv_cache_precision` or `ov_cache_layout` names the key; `rv add` / `rv load` refuse a non-finite or non-positive `--kv-cache-gb`; a pool below 0.01 GiB prints exactly.
- **A stale compiled blob is named in the log.** On the GPU plugin's `Incorrect block size`, one WARN per model names the directory to clean; nothing is deleted.
- The documentation of `kv_cache_precision` matches 2026.4.1: the plugin default is u4 for int4-weight models, and the global key applies to the text engine only.

### Fixed
**Stop and shutdown**
- **A stop while a model was loading crashed the process** (heap corruption, SIGABRT, in most stops during startup). An abandoned load tears its engine down first, and the exit skips C++ destructors when needed; exit code 0. 24 stops into a CPU load after the fix: no crash.
- **A stop during a GPU load could leave an Arc Pro B60 unusable until a GPU reset** (later loads hung 100 s to 3 h 37 min). A stop now waits for the load (up to 600 s, logging `stop requested while a model load is running - waiting for it to finish`), tears down, exits 0, and refuses new loads (`503 server_stopping`).
- The port is reserved before the preload (a taken port fails at once, exit 1) and closed at stop. The supervisor, `rv stop` (waits up to 12 min and prints "do not kill it" while the server waits for a load) and `rv start` wait for the worker instead of killing it. Measured on a B60: 8 stops 1.5-30 s into loads exited 0 in 16-150 s, and the next GPU load worked.

**Supervisor and watchdog**
- **With admin keys, the generation watchdog never fired**: its probe was refused and read as "no hang". It now authenticates with a per-worker token; a probe that cannot read the worker logs an ERROR.
- **A watchdog kill waits at least 48 s** (the worker's drain + teardown + slack) before SIGKILL, not 10 s; repeated watchdog kills stop at `max_restarts`. `rv stop` no longer suggests `kill -9`.
- **The watchdog no longer kills a long legitimate vision request** (a 92k-token prompt with a 6k answer outlasted the flat 600 s). Each request has an allowance (prompt tokens / 200 + `max_tokens` / 5 s) and `watchdog_hang_ceiling_secs` is slack beyond it, counted from when the engine starts it.
- **A supervised preload longer than `health_timeout_secs` (90 s) was killed as hung on every respawn.** The worker reports preload progress; the 90 s limit applies only to a worker that shows none.
- `kill -9` of the supervisor no longer leaves the worker serving orphaned (Linux: it gets SIGTERM and drains).

**GPU poisoning**
- **A poisoned GPU context was reported healthy.** `/health` adds `gpu_poisoned`, `/health_generate` answers 503, GPU models show `unavailable (gpu_poisoned)` (an extra `state` key in `GET /v1/models`), `rv status` prints a red "restart required" line, and `rustedvino_gpu_poisoned` is 1. CPU and NPU models stay `ready`.
- Under `--supervise`, while the worker restarts after a request poisoned the context, requests get a retryable 503 with `Retry-After: 30` (streaming: the text in the error frame). The first poisoning logs one ERROR line with the memory state.
- A GPU out-of-resources failure during a load is recognised when the error carries a context message; the worker stays up degraded and needs a manual restart, also under `--supervise`.

**Memory admission**
- **A model larger than a discrete GPU loaded onto it when that GPU was the global `device`** with the shipped `total_vram_gb: 0.0` (0.0 meant "no gate" for the inference device). On a 16 GB Arc Pro B50 a 26B model loaded through a preload and faulted the GPU (engine reset, a request that hung). The GPU is now budgeted at the memory it reports. Present since 0.6.x.
- **A CPU model made room appear that was not there.** The CPU plugin copies weights into anonymous memory at the first request; the live RAM gate now subtracts what each Ready CPU model is charged but not measured to hold (Linux only).
- **An impossible CPU load evicted every resident model before being refused.** The eviction credit is capped at min(MemTotal, `system_ram_budget_gb`), and eviction stops once the rest cannot make room.
- **A vision load on a discrete GPU could evict models and then be refused.** The eviction plan applies the device-headroom check and waits up to 20 s for the driver's late release (which also fixes "0.00 GiB free" right after an unload). A refusal after evictions names the models evicted. `rv check` previews it (`headroom_refusal`, or `headroom_not_previewed` before a first load).
- **A GenAI device-memory refusal at load is `503 insufficient_memory`**, not a raw 500 with a build path; it names models already evicted. Without a warm-up a vision model refused at its first request gets the same 503 with `Retry-After: 5`.
- **CPU image generation is charged its real working set** (`vram_gb` + the family's working set at `cpu_image_gen_max_pixels`: at 1024×1024 SD 1.5 +8.9 GiB to SD 3.5 +21.3 GiB). Larger requests are `400 size_exceeds_cpu_limit`.
- **Shared-memory GPUs under-counted models that took more than booked.** Such a model is re-booked at its measured footprint for the rest of the run, with a warning when free memory ends below the OS reservation. Set `vram_gb` that high to book it right from the first load; a restart forgets the figure.
- **A requested KV pool that was clamped was reported as success.** Loads, adds and resizes say "KV pool 2.00 GiB (requested 4.00 — clamped by cache_size_gb)" and log a WARN; `clamped_by` is `free_vram`, `cache_size_gb`, `min_kv_cache_gb`, `whole_gib_fallback` or `device_headroom` (a vision model on a discrete GPU).
- **Pools not sized in KV blocks under-booked what the engine allocates** (always whole GiB, about 1 GiB per model unbooked); they are booked at the whole GiB (rounded down when that would not fit, with a WARN above 10 % round-up).
- **Two concurrent admissions could deadlock the model manager** (present since 0.8.x).
- A load blocked only by another model's eviction-grace window is refused as transient ("retry in N s"); this fixes 0.8.4's misleading refusal right after a load.

**API, streaming and audio**
- **Vision `usage.completion_tokens` and token metrics undercounted** (one streamer callback can carry several tokens: 6,425 reported as 4,678); the pipeline's own count feeds `usage`, `rustedvino_tokens_generated_total` and the tok/s gauge.
- **A vision chat template that rejects the messages answers `400 template_rejected`** with the template's reason, not `500`.
- **A per-model `chat_template` set through `models/add` or `PATCH` was ignored until the next restart**; it now applies from the next load.
- **`chat_template_kwargs.enable_thinking` is honoured** when the top-level field is absent; a non-boolean value is `400 invalid_chat_template_kwargs`.
- **Invalid UTF-8 from the text engine** keeps its valid part with U+FFFD and a warning; a stream frame that cannot be serialised becomes a `frame_serialize_failed` error frame.
- **An NPU Whisper request without `language` on audio over 30 s hung until restart.** Language detection uses the first 30 s only.
- **A realtime utterance over 30 s could lose its first 30 s** (whisper-large-v3-turbo on 2026.4.1 returns 1-2 words at exactly 30 s). The VAD auto-commit is at 29.5 s; the rest starts the next utterance.

**CLI and packaging**
- `rv add` without `--vram-gb` skips backups, partial downloads, `.git/`, caches and another server's `model_cache/`, and prints what it skipped. `rv list` shows `/bundle/models`, not `/bundle/./models`.
- **The bundle carries ONNX Runtime's licence.** ONNX Runtime 1.24.2 (MIT; linked for the Kokoro, Piper and VITS voices) shipped without its licence and notices up to 0.8.4; the Linux bundle has them under `licenses/onnxruntime/`, and `THIRD-PARTY.md` lists it.
- Tools: `tools/bench/cmpbench.py` no longer SIGKILLs RustedVINO 60 s after a stop; it waits up to 720 s, then warns.

### Known issues
- **With `bind_addr` set to a network address, a keys file that holds only admin keys leaves inference open.** The startup check that refuses an open public bind counts admin keys too, but inference is protected only by `api_keys`, and the keys file `rv start` creates has none. If the server is reachable from the network, add at least one `api_keys` entry (or keep `bind_addr` at `127.0.0.1`). Fixed in the next release.
- **Audio and image uploads are limited by the 50 MiB request size, not by their decoded size**, so one crafted request can make the server run out of memory and exit, taking every loaded model with it. Give inference keys only to clients you trust and do not expose the port to untrusted networks. Fixed in the next release.
- **A process killed mid-GPU-work can still leave an Arc Pro B60 stuck.** A `kill -9`, OOM kill or crash cannot be caught; the xe driver logs an engine `CAT error` and the next load on that GPU can hang for minutes to hours. The server books it stalled, keeps serving, and `rv status` shows the recovery step. A GPU reset (`sudo cat /sys/kernel/debug/dri/<PCI address>/gt0/force_reset_sync`) fixed it in milliseconds, verified once on kernel 7.0 (older kernels may use `dri/<N>`). Loaded models may need reloading; a restart is not reliable, a reboot is. Under `--supervise` the worker restarts `watchdog_hang_ceiling_secs` after the deadline: reset before that.
- **A supervised worker poisoned by a request exits 75 without tearing its engines down.** On a B60 that exit may hit the same stuck-driver state; the GPU readiness probe runs before the next worker loads anything.
- **After an engine teardown that ran out of time, every stop exits 76** until that thread really exits; a supervised restart then takes the probe path.
- **Windows (built from source): the exit still runs the C++ destructors**, so the stop-during-load crash fix is not active there yet.
- **Memory learning can over-book for the rest of the run.** A generation on another model during a load counts into that load's footprint; a CPU model's first request with a long prompt counts its KV pages into the weight. Later loads may evict more or be refused; a restart forgets it.
- **Two long vision prompts at the same moment can exhaust one GPU** when they go to two vision models on that GPU while the first prompt is still being tokenised. `kv_headroom_mode: "off"` (not `kv_gate_mode: "legacy"`) turns the headroom load gate off.
- **On a discrete GPU, a load admitted after an idle gap uses the outside-memory reading from the previous load**, so memory other processes took since then is covered only by `kv_headroom_margin_gib`. A WARN "device-memory probe keeps failing" can appear although the probe works. Fixed in the next release.
- **A hung vision generation reaches the watchdog late**: allowance + ceiling can exceed half an hour without `max_tokens`. Set `max_tokens` on vision requests.
- **A vision KV wedge repeats when the learned limit cannot be saved** (unwritable `kv_ratchet/` directory): each repeat evicts and reloads the model.
- **Invalid UTF-8 from a vision model is still dropped silently.**
- **Stop edges.** A stop gives up on a load at 600 s; a load that commits later can still be in teardown at exit. A forced eviction with a stuck generation makes a stop wait the full 600 s.
- **Stalled-load edges.** A KV resize on a device with a stalled load cannot reload the model until that load returns. `/health` shows `device_stalled: true` for a client-gone load that is still running normally. `GPU` and `GPU.0` on a one-GPU box do not block each other.
- **`rv` edges.** `rv add` / `load` / `resize` give up after 630 s while a longer load goes on (check `rv ps`). `rv add` exits 0 when the model loaded but was not saved. `rv stop` can report a slow many-model teardown as failed after 90 s when the log level hides the wait line (also stopping `rv restart`); `rv restart` during a preload drops `--supervise`.
- **SpeechT5 speaks at most about 600 characters and still returns `200`.** The rest of `input` is dropped without an error, and quality breaks down from about 250–450 characters (skipped sentences, repeats, loops). Send at most 200 characters per request and join the audio. SpeechT5 ignores `speed`, and every engine accepts an unknown `voice` without a `400`. The Polish VITS and Piper voices were tested complete up to about 3,300 characters. Per-engine limits are in INTERFACE.md.
- **Carried over from 0.8.4, not re-tested on 2026.4.1:** `rv deregister` / `rv save-defaults` without a terminal need `-y`; gemma-4-e4b-it-int8-ov does not load as a vision model; unloading the reranker can crash the process on some CPU setups.

## [0.8.4] — 2026-10-16

The release after 0.8.0. Versions 0.8.1, 0.8.2 and 0.8.3 were internal test builds and were never published; everything they
contain is folded into this release (their entries are kept below for the record). Fixes the defects found by the 0.8.3 test run
and an engine comparison on an Arc Pro B60 (D-01 to D-12, plus D-13 found while fixing D-09), and makes the sampling default
cheaper (K-02a). Tested on an Arc Pro B60 (60 of 60 checks pass, no GPU wedge in a 12-step overlap ladder). The GenAI 2026.3
migration and the hybrid-model capacity rework are 0.9.0. (D-, K- and F- numbers in these entries refer to the project's internal test reports.)

### Security
- A server with no keys at all (empty `api_keys` and `admin_api_keys`) refuses requests with an `Origin` header (a browser) on `/v1/admin/*` and `/metrics`: 403 `browser_origin_refused`. Before, any web page open on the same machine could, with the default permissive CORS policy, send a body-less POST to `127.0.0.1` and add, unload or delete models. curl, `rv` and Prometheus send no `Origin` and are unaffected; with admin keys configured nothing changes.
- `POST /v1/admin/models` refuses a `model_id` that is not a single folder name (`/`, `\`, `..`, absolute paths), so the "directory not found" answer can no longer probe the filesystem.

### Packaging
- The Linux bundle now carries its licences: `LICENSE`, `THIRD-PARTY.md`, `licenses/openvino/` (Intel's licence and third-party notices for the OpenVINO libraries it redistributes, including oneTBB, oneDNN and hwloc) and `licenses/rust-crates-licenses.txt` (the licence files of every Rust crate it is built from). 0.8.0 shipped none of these inside the archive.
- The bundle is built from the exported public source tree (its own `Cargo.lock`, its own commit hash in `rv version` and `/health`), with the pinned Rust toolchain, so what you download can be rebuilt from the tag.

### Added
- `GET /v1/models/{id}` (OpenAI's retrieve-model route); a model that is unknown or not ready answers a JSON 404 `model_not_found` (D-12).

### Changed
- A request that sets only `temperature` (> 0) now takes `top_p`/`top_k` from the model's own `generation_config.json`
  for the fields it left unset, instead of sampling over the whole vocabulary. Faster decode: on an Arc Pro B60 qwen3-1.7b went from 155.9 to 169.7 tok/s (+8.9 %) and qwen3-8b from 66.6 to 69.3 (+4.1 %); at high temperatures the very unlikely tail tokens are no longer drawn.

- **`rv add` estimates a model's memory as its plain on-disk size** (the 1.12 margin of 0.8.3 is gone). The margin fixed Intel's mixed-precision qwen3-14b export but made `rv add` refuse qwen3.6-35b-a3b on a 24 GB card, which runs (needs 21.7 GB, was asked for 24.0); measured footprints range from 0.96 to 1.11 of the disk size, so no single ratio fits. The server still adds its own minimum KV and safety margin. The `rvbench` fit gate uses the same figure (D-03).
- The warning `measured device memory exceeds the configured vram_gb` now fires only for overshoots above 5 % or 0.75 GB (whichever is larger); it used to fire for any overshoot, including a 1.7B model that measured 0.69 GB above its charge (D-01).

### Fixed
- A hybrid linear-attention vision model (Qwen3.5 / Qwen3.6) added on a GPU without prefix caching, which is what the server forces for such models, no longer fails after the compile with `reserved_la_bytes <= budget_in_bytes` (12-61 s of waiting for an error): the vision engine now tells GenAI it holds one sequence instead of the default 256, so the linear-attention reservation is small and the model simply loads (D-02, K-01a without the KV-precision part). Verified on Qwen3.5-4B (prefix caching on and off: two-turn recall and an image answer correct) and on Qwen3-VL-8B and InternVL2.5-8B (both modes).
- `/tokenize` and `/detokenize` on a vision model no longer wait for the generation in progress (they took 21-27 s behind a long answer): the vision engine now has a tokenizer of its own with its own lock, and the calls answer in a few milliseconds while it is generating (D-13).
- A gpt-oss model added without an explicit `reasoning_parser` (for example with `rv add`) leaked its raw harmony channels into the answer: `analysis…assistantfinalPlanet: Mars` in `content` (the QA run saw the `final` glued to the start of the answer). The harmony parser is now chosen automatically from the model's chat template, like the Qwen3 one already was; an explicit setting still wins (D-07). Side effects, both improvements: gpt-oss tool calls are now detected for such a model (they were never parsed without the parser), and the realtime voice path now applies the gpt-oss policy (`<|channel|>final` prefill, the session prompt sent as the model's identity).
- A global `device` in `config.json` that OpenVINO does not enumerate (`"GPU.7"`) is now named at boot together with the devices that do exist, and, when `default_placement` is `"device"`, with the consequence that every model without its own `device` will fail to load. The boot check also understands `HETERO:` lists now (D-08).
- `rv start` no longer reports a deliberate stop during model loading as an error (it said `the server exited during start (exit status: 0)` and dumped the log); it now says the server was stopped before it was ready. The refusal printed by a second `rv start` no longer claims `rv stop` cannot find a loading server (D-04).
- `/detokenize` with a token id outside the model's vocabulary (negative, or at or above its size) is a 400 `invalid_value` naming the index and the vocabulary size, instead of a 200 with silently dropped text (a model whose detokenizer has no vocabulary table can only reject negative ids).
- `/tokenize` and `/detokenize` now work for vision models (the qwen3.6 family and other VLMs), which they used to refuse with a 400; the vision engine runs its own tokenizer, so real ids and a real decode come back (D-09). NPU text models and non-text kinds are still refused.
- The `developer` message role (OpenAI's newer name for `system`) is rendered to the chat template as `system`; before, the template received the unknown role `developer` and the instruction could be dropped.
- The "measured device memory exceeds vram_gb" warning compares like with like: the measurement spans the whole load (weights plus the KV pool), so the charge it is held against is `vram_gb` plus the pool the load allocated. 0.8.4 B60 run: it had fired for every small model that was correctly sized (qwen3-1.7b 2.09 GB vs 1.2 GB of weights; qwen3-8b 5.90 vs 4.9). (D-01)
- An unknown route answers a JSON 404 (`code: unknown_url`, OpenAI's wording) instead of an empty body (D-06). Done as a response layer, not a router fallback, so an extension router that sets its own fallback cannot make the merge panic; a keyless request to an unknown route is still a 401 on the base router (with an extension merged, unknown paths get the JSON 404 but no auth).
- A chat message with an unknown `role` (anything but `system`, `developer`, `user`, `assistant`, `tool`, `function`) is now a 400 naming `messages[i].role`; it used to be answered as if nothing was wrong (D-05).
- `top_k`/`top_p` sent with `temperature: 0` returned a 500 ("inference failed"); the request is now decoded greedily
  and they are ignored (D-11).

## [0.8.3] — internal build, never published

Folds in 0.8.2 (below, never published); fixes everything the Arc Pro B60 torture run of 2026-10-14 found
(57 pass / 2 fail / 1 skipped on an Arc Pro B60) except the load timeout.

### Fixed
- **Stopping a server that is still preloading.** SIGTERM / ctrl-C used to be honoured only when the GPU compile of the preloaded model
  ended (273 s for a 16 GB model on a B60). The startup is now abandoned at once and the process exits cleanly (about 5 s). `rv stop` also
  finds a server that `rv start` launched and that is not listening yet (it checks the pid file and that the process runs this config).
- **`default_placement: "auto"` no longer sends a hybrid linear-attention model (Qwen3.5 / 3.6) to a GPU**, where it cannot load with the
  prefix-cache guard in force. It goes to the CPU; `enable_prefix_caching: true` on the model lifts that, and an explicit `device` still wins.
- **`rv add` memory estimate** uses a 1.12 margin (was 1.08): Intel's mixed INT4/INT8 export of qwen3-14b measured 0.3 GB above the old estimate.

### Changed
- **Schema errors name the field**: ``invalid value for field `max_tokens` `` / ``missing required field `messages[0].role` ``. Values and serde detail are still never returned.
- **`rv add` / `rv load` print which models they evicted.** A failed `rv start` shows the cause first and only that run's log lines.
  `rv check` on an unregistered model points to `rv add`. A CPU placement's memory shows as `x GB (RAM)` in `rv list` / `ps` / `status`.
- `rvbench` retries a hybrid model with prefix caching on after a GPU load failure (and marks the row), and skips a model the shipped server
  would refuse as larger than the GPU (`--no-fit-gate` measures it anyway).
- README: `rv start` is unsupervised (`--supervise` exists); `rv add` needs `-y` without a terminal.

### Not fixed (0.8.3 backlog)
- No load timeout / lock release for a hung GPU compile; no request deadline.

## [0.8.2] — internal build, never published

Folds in the whole of 0.8.1, an internal test build (local tag `v0.8.1`, bundle sha256 `adf7e957...`) that was never
published; the QA reports of 2026-10-12 cite it.

### Changed
- **Models load on the CPU by default; an integrated GPU is a last resort.** New setting `default_placement` (`"auto"` | `"device"`).
  The release bundle's `config.json` ships `"device": "CPU"` with `"default_placement": "device"`: every model without its own `device` loads on
  the global `device`, so `device: CPU` finally means CPU (before, the placement engine ignored it). The capability-aware engine (`"auto"`, stated
  explicitly in every fleet config) no longer puts an iGPU before the CPU for any default order. Chat LLMs, embedders and rerankers take a discrete GPU
  when there is one and the CPU otherwise; STT goes CPU, then a discrete GPU (never the NPU or an iGPU by default: whisper-turbo hung the NPU on a clip over 60 s); a thin vision model takes the CPU; **TTS is CPU-only** (no GPU tier at all);
  the decision model stays dGPU-first; image generation keeps a strong iGPU ahead of the CPU because FLUX needs > 19 GB of host RAM there. Why: on an Arc Pro
  B50 box the 1.7B chat model decoded at 114 tok/s on the B50 against 35 on the CPU and 26 on its UHD 730 iGPU (embeddings 727/s vs 39/s); on an Acer UHD iGPU
  every light model ran 20-45 % slower than the CPU; SpeechT5 stalls or crawls on every GPU class (3x slower on an Arc 140V, RTF 16-100 on the UHD, a ~10 minute
  stall then HTTP 503 on the B50); the old orders were tuned on one unusually good iGPU (Arc 140V).
  **Upgrade note:** a config that relied on automatic iGPU offload for light models now gets the CPU or a dGPU; pin the model with `"device": "GPU"` (or
  `embedding_device`). the example fleet configs pin their embedding/rerank models for this reason.
  `light_stt_max_gb` no longer changes the default order and is kept only so existing configs load.

### Added
- **`rv start`, `rv stop`, `rv restart`, `rv version`.** For a server run by hand from a bundle:
  `rv start` launches it detached (log in `rustedvino.log` beside `config.json`) and waits for
  `/health`; `rv stop` sends SIGTERM to the process holding the configured port and waits for it to
  exit, and refuses one that systemd runs (use `systemctl`, or `--force`). `rv start` refuses a
  `--port` the config does not set and a second start while a server from the same directory is
  still loading; `rv restart` keeps `--supervise`. Errors follow the compiler's shape (`error:`, then
  `= note:` / `= help:`). `rv version` shows the server binary on
  disk and the running server. Linux only for now.
- **`rustedvino --version`** (`-V`) prints the version and commit and exits; it used to fail with
  "no config file found".
- **`rv` works against an open server, and says so.** With a keys file that holds no keys (for
  example `{}`, or the empty-lists template the bundles ship), the server needs no bearer for its
  admin routes, but `rv` used to refuse every admin command with "no admin key available". It now
  proceeds without authentication and prints, on every call, that the server is OPEN and NOT
  SECURE (anyone who can reach the port can load, unload and delete models); `rv resolve` says the
  same. A missing, malformed, or inference-key-only keys file is still refused.
- **Bundles ship `config.keys.example.json`** (empty lists; copy to `config.keys.json`) and a
  download-to-first-request quick start; the README lists a starter set of models with measured
  disk, RAM and CPU speed. `tools/bench/rvbench.py` reproduces those measurements.
- **`/v1/audio/speech` speaks numbers.** Digits and `%` were silently dropped by the Polish and
  SpeechT5 voices and garbled by Kokoro in chat TTS (only the realtime flow normalised text), so
  "W 1948 roku opublikowano 3 artykuły." was heard as "W roku opublikowano artykuły.". The route
  now expands digits and symbols into words before synthesis, in the language of the request's new
  optional `language` field or else of the model. New optional `normalize` (default `true`) turns
  it off. The Polish normaliser also gained scale-word agreement ("dwa tysiące", was "dwa tysiąc"),
  `zł` / `kg` / `km` agreeing with the number, and ordinal years before "roku" / "r." in the right
  case. English no longer reads a quantity like "1250 dollars" or "1500 people" as a year ("twelve
  fifty"; this also affected the realtime flow). See `INTERFACE.md`. Clients that send words already are unaffected.

### Fixed
- **A `0.0` that means "gate off" is no longer used as a number** (bundle default `total_vram_gb: 0.0`; review round 2026-10-13): the `rv add` refusal on a CPU install with plenty of RAM ("needs 5.8 GB (0.0 GB free)", the dry-run eviction plan read 0.0 as no room and could never admit the load; `rv check` now says `no VRAM limit set`); the KV pool on the CPU was always the 1.0 GB floor, now it follows `cache_size_gb`; the discrete-GPU size ceiling collapsed to zero and kept every model off the GPU under `auto` (fixed earlier, now also covered by a test on the shipped config).
- **Kind detection** (`rv add` without `--kind`): SpeechT5 is recognised as text-to-speech (it used to be taken for a Whisper model and "loaded" as one), a one-label sequence classifier as a reranker, a bare BERT / XLM-RoBERTa-family model as an embedder.
- A non-Intel discrete GPU that OpenCL enumerates (an NVIDIA T1200) is no longer treated as the workhorse: OpenVINO cannot build for it, so it takes the last-resort tier.
- **Speech-to-text defaults to the CPU under `auto`**, never the NPU (whisper-large-v3-turbo hung the NPU on a clip over 60 s).
- The hybrid-model load error now says what actually works (`rv add ... --enable-prefix-caching`; `cache_size_gb` caps every KV pool) and no longer quotes a build-farm path.
- **`rv start` creates a missing `config.keys.json`** with ONE random admin key (mode 600, no `api_keys`) and says so, so the quick start no longer dead-ends at `rv add` with "no admin key available" followed by a restart; a key rather than empty lists because an empty file leaves the admin API open to any web page that can reach 127.0.0.1 (permissive CORS). `rv add --enable-prefix-caching` forces prefix caching on for a hybrid Qwen3.5/3.6 model on a GPU (D-03). A relative `models_dir` (`./models`) is resolved against the bundle directory and a relative `RUSTEDVINO_CONFIG` against rv's working directory (the launcher changes directory, so the server used to read another file). `rv list`, `status` and `version` now print the open-server warning too; the "no admin key" error says to restart; `rv status` prints `no limit` instead of `0.0 GB` for a memory domain with the gate off; the no-config error of `rv start` lists where rv looked.
- **Memory refusal names the gate that refused.** When the live system-RAM gate (not the logical tracker) blocks a load, the error now reports the RAM figures (needed / available / OS floor) instead of the tracker's misleading "0.0 GB free".
- **`rv` no longer panics on a closed pipe** (`rv ps | head`): SIGPIPE is restored to its default action.
- Bundle README: step 5 downloads with `download-model.py` (stdlib only) or a virtual environment, which needs `python3-venv` on a stock Ubuntu / Mint (one `sudo apt install`); the bundle ships `download-model.py`, which needs only Python 3 - no sudo, no pip (the venv path fails without python3-venv) and `huggingface_hub[cli]` no longer exists .
- Bundle README: `--kind` for embedding / reranking / tts models, the rotate-the-admin-key recipe, and a pointer to CONFIG.md.
- The bundle now ships `CONFIG.md` (server errors already pointed to it).
- Docs: a larger `kv_cache_gb` does NOT make the hybrid Qwen3.5 / Qwen3.6 models load on a GPU; `enable_prefix_caching: true` per model does. `dgpu_size_ceiling_fraction` behaviour with VRAM gating off documented.
- **Wrong answers from hybrid linear-attention models (Qwen3.5 / Qwen3.6) on a GPU with prefix
  caching on.** OpenVINO GenAI 2026.2.x returns a different continuation when a prompt prefix is
  served from the prefix cache: the same request alternated between a full answer and a truncated
  one (reproduced with plain GenAI, no RustedVINO code; the CPU is unaffected; fixed upstream in
  2026.4.1). The server now turns prefix caching off for such a model when it loads on a GPU and
  logs a warning. Without the cache, the linear-attention state may no longer fit: Qwen3.5-9B loaded
  at `kv_cache_gb` 8, but the 4B and the Qwen3.6 27B / 35B-A3B failed even at 16 on an Arc Pro B70 (use `rv add --enable-prefix-caching`), and repeated
  long prompts lose the prefill speed-up. New per-model `enable_prefix_caching` (`true`/`false`,
  `null` = automatic; also settable with `PATCH /v1/admin/models/{id}`, next load) overrides it —
  see `CONFIG.md`.

### Changed
- **Busy models are not evicted (the D3 fix from the 0.8.0 known issues).** A model with a request
  running is never chosen as a victim by a plain load, and a plain unload, deregister or resize of
  it is refused with `409 model_busy` (a load that could only be satisfied by evicting such models
  is refused with `503 model_busy`, both with `Retry-After`) — the request finishes undisturbed.
  `force` is the override: `"force": true` on `/load` and `/resize`, `?force=true` on
  `DELETE /v1/admin/models/{id}` and `…/register`, and `rv load|unload|deregister|resize --force`
  now bypass the eviction-grace window **and** the busy rule (the model is drained for 15 s on a
  load, up to 2 minutes on an unload, then what is left is aborted). `force` still never overrides
  `pinned`, `evictable: false` or the models of a live realtime session. The automatic recovery
  paths (KV-wedge recovery, a dead decision engine, shutdown) are forced. A hard never-evict-busy
  rule was tried and reverted during 0.8.0 development (it starved other loads for as long as a
  client kept the busy model occupied, and made a wedged generation un-evictable); this one
  refuses immediately instead of waiting, and has the override.
- **A load executes the eviction plan it previewed.** Before it evicts anything the load runs the
  same dry-run as `POST …/check`; if even full eviction would not make room it is refused with
  "nothing was evicted", and it never evicts a model the preview did not name. The upfront RAM
  check credits an evictable model with its weights only (as the plan always did), not its
  reserved KV pool, so on the shared-memory domain a load that was only feasible by counting
  reserved KV as free RAM is refused up front instead of after evicting idle models.
- **A forced eviction of an engine that cannot be aborted waits for it.** Everything except the
  continuous-batching text engine runs its current request to completion; a forced eviction used
  to free the memory tracker after 8 s even if the engine thread was still holding the memory.
  It now waits up to 30 s and otherwise answers `503 eviction_incomplete`; the reservation is
  released when the thread exits.
- **A resize that cannot succeed no longer strands the model or burns the cooldown.** The reload
  at the new size is checked before anything is evicted, and a refused resize (busy model, or a
  reload that would not be admitted) leaves the resize cooldown untouched.
- **Realtime:** a client whose requested model's room is held by busy models gets the new reason
  `would_evict_busy`, and the default-model fallback is no longer started when it would only be
  refused (`loading: false` at once instead of a 90 s wait). `POST …/add` answers a busy load
  refusal with the same `503 model_busy` instead of a 400.

### Added
- **Admin `force` on unload, deregister and resize**, `rv … --force`, and `blocked_by_busy` in
  the `POST …/check` response (the busy models whose eviction would have made the load fit).
- **Metrics:** `rustedvino_load_refusals_total{reason}`, `rustedvino_eviction_refusals_total{reason}`,
  `rustedvino_forced_evictions_total{kind}` and `rustedvino_eviction_aborted_requests_total{kind}`,
  with a WARN naming the blocking models on every refused load.
- Admin eviction refusals use the JSON error envelope (`model_busy`, `eviction_incomplete`).

### Fixed
- **A cancelled eviction completes.** An eviction whose caller went away (an admin client
  disconnected) used to leave the model `Evicting` with its VRAM reserved until a restart; it is
  now finished in the background.
- **A late request during an eviction is refused, not aborted.** A request that slipped in
  through a handle taken before the eviction began gets a retryable 503 instead of being accepted
  into an engine that is being torn down.
- **Image and rerank requests keep their model busy until the engine is done.** A client that
  disconnected mid-request used to make the model look idle while the engine thread still ran.
- Refusal text no longer says "this load can never fit, nothing was evicted" when busy models are
  the only obstacle.
- **Vietnamese speech normalisation:** a session with `language: "vi"` had digits and symbols
  expanded as English words before Piper's own Vietnamese expansion ran ("8 giờ 30" was spoken as
  "eight giờ thirty"). `vi` now reads numbers, decimals, negatives, currency and `%`, `&`, `=`, `@`,
  `+` in Vietnamese. Vietnamese `.`-thousands / `,`-decimal notation is not interpreted (as for
  Polish).

## [0.8.0] — 2026-10-05

### Added
- **Windows default config location:** `%LOCALAPPDATA%\RustedVINO\config.json`, consulted (if the
  file exists) after `RUSTEDVINO_CONFIG` and before the config beside the executable, by both the
  server and `rv`. Per-user, so writing it needs no elevation. Linux is unchanged.
- **`rv` is now a native binary** (`tools/rv`, a separate crate that links no OpenVINO) and the
  release bundles ship it as `rv` / `rv.exe` — no Python needed, which the Windows trip showed is a
  real obstacle (a fresh Windows has only the Microsoft Store stub). It ports every verb of the
  Python script (`status`, `list`, `ps`, `check`, `add`, `load`, `unload`, `deregister`,
  `reload-config`, `patch`, `preload`, `save-defaults`, `logs`) and adds `resize`, `completeness`,
  `reload-keys`, `voice-pin`, `sessions`, `session` and `session-kill`. `scripts/rv` stays in the
  source tree as a reference. `rv status` now takes the server's PID from the server itself, so it no
  longer reports "not running" about a server that is answering (the Python probe was Linux-only).
- `GET /v1/admin/models` `server` block now carries `pid` and `uptime_secs` (valid-key callers only,
  like `host`), so a client can identify the process that is actually answering without OS-specific
  probing — groundwork for a cross-platform `rv`.
- **`POST /v1/systemone`** — Ollama v0.35-compatible typed decisions (choice / yes-no / ordinal
  score) with probabilities, answered by one prefill per question on a `kind: "decision"` model,
  with no text generation. Request validation, check order, status codes and error messages follow
  Ollama; prompts are byte-identical to Ollama's, pinned by 121 fixtures generated from Ollama's own
  code. The model must carry a `decision.json` sidecar and a last-position-gathered IR (see
  [`CONFIG.md`](CONFIG.md#decision-models-v1systemone)). Decision requests take a lease on the
  model's device budget like other endpoints, and a client that disconnects stops its request
  before the next question.
- `kind: "decision"` models: never auto-detected; refuse the NPU; placed on the strongest GPU first.

### Fixed

- A request aborted because its model was evicted under it (the 15 s eviction drain deadline, or shutdown) now gets a retryable `503` with code `model_unloading` and `Retry-After: 5` (streaming: an error frame with the same code) instead of an opaque `500 inference failed`. Found by the GPU-wedge experiment.
- **An unwritable `ov_cache_dir` is now reported at startup.** OpenVINO skips its compile cache without
  a word when it cannot write there, so a server on a read-only or wrongly-labelled folder (the Windows
  trip's `ov_cache\` stayed empty through every load) looked healthy while recompiling each model each
  time. A write probe at boot now logs a WARN naming the path and the likely cause.
- NPU Whisper now gets `CACHE_DIR` like every other pipeline (the old exclusion, a hang seen in
  2026-08, does not reproduce on GenAI 2026.2.1: cold load 3.5-4.1 s, warm 0.33 s). It makes the
  language-drift fix below cheap. Unverified on Windows.
- **`rv check` / the admission dry-run no longer under-predicts evictions under RAM pressure.** It
  credited an evicted model's whole tracked reservation (weights + a KV pool the GPU plugin allocates
  lazily) as freed system RAM; the real load, which re-reads live RAM after each eviction, then
  evicted more than the dry-run said (predicted 1, evicted 2 on the Windows trip). The dry-run now
  credits RAM with the weights only. The eviction log line also names the binding constraint
  (`constraint=vram|system_ram`) instead of always saying "for VRAM".
- **A VLM routed to the NPU no longer fails on a property our code passed.** The NPU's VLM pipeline is
  the static one and rejects `scheduler_config`; it is no longer passed there. (Such a model still
  needs an NPU-compatible export — the failure is now the NPU compiler's, not ours.)
- **Admission budget is capped at what the inference device reports** (`total_vram_gb` larger than the
  device's own figure is clamped, with a startup WARN; an explicit `domain_budgets` entry still
  wins, `0.0` still disables gating). On Windows the Intel driver's reported figure is a WDDM budget:
  over it nothing is refused, the OS keeps the excess non-resident and every model sharing the GPU
  runs 3-7x slower. A config written for Linux (27 GB on a laptop that reports 17.7) used to admit
  everything. No fleet config is affected (all are below their devices' figures).
- **Bad input can no longer take an engine down or crash the worker** (found by a 6.5-hour fuzz/torture run): a WAV declaring sample rate 0 made the audio decoder panic, killing the speech-to-text engine until the model was reloaded - undecodable audio, including a decoder panic, is now a `400 invalid_audio`; a NUL byte, an empty document, or a query/document made only of invisible characters (zero-width space, byte-order mark, bidi overrides) in `/v1/rerank`, and a NUL byte in `/v1/embeddings` or `/v1/audio/speech` input, are `400 invalid_input` instead of `500` (and no longer rebuild the rerank pipeline, which crashed the worker once in 1,143 rebuilds); the reranker now rebuilds its pipeline after a failed call only if a tiny probe call fails too; `/v1/audio/speech` input is capped at 4096 characters (`400 input_too_long`; 70,000 characters made the voice model ask for 78 GB); `top_p: 0` is clamped to the smallest value the engine accepts instead of answering 500.
- **Every setting is written out in the shipped configs.** All seven `scripts/*.config.json` (the fleet configs, the examples and the starter `dist.example.config.json`) now list all 47 global settings and all 20 per-model settings explicitly, at the value already in effect (each parses to the identical configuration as before), so nothing hides behind a built-in default. New: `rustedvino --print-default-config` and `--print-default-model` print every setting with its default; `scripts/fill-config-defaults.py [--check] CONFIG...` adds whatever a config is missing without touching your values. `CONFIG.md` gains an "Every setting is written out" section, and `tests/shipped_configs.rs` / `tests/config_reference.rs` fail if a setting is missing from a shipped config or from the reference. `scripts/coresidency.example.config.json` was rewritten in the current schema (it used the long-removed top-level `vram_gb` / `model_policies` maps and could not load).
- **Boot warning for unbounded prefix caching.** `cache_size_gb: 0.0` together with `enable_prefix_caching: true` lets retained prefix blocks grow outside the memory budget; the server now warns at boot, and the starter `dist.example.config.json` sets `cache_size_gb: 2.0`.
- **Config and keys files saved with a UTF-8 BOM load.** PowerShell 5.1 (`Set-Content -Encoding utf8`) and Notepad add one, and the server exited with "expected value at line 1 column 1".
- **`rv patch` no longer rewrites a model's whole stanza.** Persisting a patch used to write every default (`kind: null`, `pinned: false`, ...) into a hand-written entry; it now keeps the entry as written plus the patched field.
- **Smaller fixes from the Windows pass:** `/metrics` byte gauges are whole numbers (no `-0` or `...000002`); the device-tier cache and model-hash memo now work on Windows (they fell back to nothing without `HOME`/`XDG_CACHE_HOME`; `%LOCALAPPDATA%` is the last resort); `rv status` summarises config-audit problems as sentences instead of raw JSON; the staged Windows keys file has separate inference and admin keys; `CONFIG.md` spells out the default CORS exposure.
- **Rejections on the API routes use the OpenAI-style JSON error.** `/v1/rerank` answered malformed JSON in plain text and a missing `model` with 422, and leaked raw engine text in its 500; a wrong `Content-Type` on the audio/image-edit routes gave axum's plain "Invalid `boundary`"; a wrong HTTP method gave an empty 405; a bad `?force=` on the check route gave a plain-text 400. All now use the shared envelope (`invalid_body`, `invalid_multipart`, `method_not_allowed` with its `Allow` header, `invalid_query`), the 405 does not bypass authentication (keyless callers still get 401), and rerank's unknown-model text matches the other routes. Still plain: unknown paths (404) and routes added by an extension router. Rerank engine failures now go through the same scrub as chat, so an out-of-resources error there marks the GPU poisoned like chat does.
- **Undecodable audio is a 400, not a 500.** Garbage, truncated or empty audio sent to `/v1/audio/transcriptions` or `/translations` returns `400 invalid_audio`; the 500 "inference failed" is left for real engine failures.
- **A prompt that can never fit is a 400, not a retryable 429.** An oversized chat prompt (e.g. 20 MB) used to hit the KV-pool pre-check first and was told to "retry shortly"; it now gets `400 context_length_exceeded` from the byte-length gate.
- **NPU models are charged their real footprint (`npu_footprint_factor`, default 1.5).** The admission arithmetic charged an NPU LLM its configured `vram_gb` (5.5 GB for an 8B int4) although Windows measured 7.6-8.4 GB of NPU adapter memory, so a GPU model beside it could be handed a KV pool the shared budget did not cover. Set `1.0` to charge the configured figure as before (Linux's DRM accounting measures ~5.2 GB for that model).
- **A load waits for the OS to return an evicted model's memory before evicting the next.** After an eviction made for the system-RAM gate the loader waits (up to 3 s) for the freed memory to show up before judging whether the next model has to go, and logs a warning when a load evicts a model `rv check` had not predicted. Windows loads evicted three models where the dry-run said one. A Windows re-test showed this does **not** close the problem: the loader still credits a victim its configured size, not the memory the OS actually gets back (see Known issues).
- **A wedged GPU restarts the worker (under `--supervise`, once serving).** The first out-of-resources failure that poisons the GPU context makes the worker log an ERROR, drain in-flight requests and exit with code 75; the supervisor logs why and respawns it. Engine teardown is skipped (a poisoned context refuses evictions; process exit releases device memory), so exit can take up to about 40 s (30 s drain limit plus teardown) if streams or sessions are open. Only a model that was already serving triggers the restart: a poisoning during startup (a preload OOM) or while LOADING a model leaves the server up and degraded, so a retrying client cannot turn a bad memory estimate into a restart loop; and without a supervisor the server stays up, degraded, and logs that a manual restart is needed.
- **Loading over a busy model is bounded.** An eviction made for another model's load drains the busy victim for at most 15 s (not 2 minutes, which blocked every other load behind the global load lock) and then aborts what is left; `POST /v1/admin/models/{id}/check` reports such victims in `busy_victims` and `rv check` / `rv load` warn before it happens.
- **`eviction_grace_secs` now defaults to 2 s (was 300 s), and `rv load --force` works on its own.** The long default blocked ordinary model switches for five minutes after a model's last use. Every shipped config now spells the setting out (`"eviction_grace_secs": 2.0`) instead of relying on the built-in default. A model with a running request is not protected by the window, but it sorts last in the victim order and evicting it drains it first; `force` bypasses the grace window only (never pins or the realtime set). `POST /v1/admin/models/{id}/check?force=true` previews a forced load, and `rv` uses it.
- **A wedged GPU is now reported as such.** After an out-of-resources failure the GPU context is unusable until the server restarts (a restart is the only proven cure). Inference errors no longer say "the GPU is recovering; retry shortly"; `GET /v1/admin/models` carries `server.gpu_poisoned`, and `rv` prints a banner instead of listing the dead model as healthy.
- **Boot warnings for every runtime-written path.** Besides `ov_cache_dir`, the server now warns at boot when `cache_manifest`, `config.json` or the keys file is not writable (Windows staging ACLs). `stage-windows.sh` ships `fix-permissions.cmd`.
- **Whisper on the NPU no longer misdetects the language after the first request.** On the NPU,
  automatic language detection was right once per pipeline build and German (for English audio)
  from then on — an OpenVINO GenAI 2026.2.1 behaviour reproducible without RustedVINO. An auto-detect
  request on the NPU now detects the language on a small CPU Whisper pipeline built from the same
  model on first use (~0.2 s for tiny) and passes it to the NPU pipeline explicitly; the NPU pipeline
  is built once. The CPU copy of the weights stays in RAM until the model is evicted and is not
  counted by the memory accounting. (A first version rebuilt the NPU pipeline instead and leaked
  3 threads + 5 handles per request — destroying an NPU `WhisperPipeline` leaks, upstream, found on
  Windows. Re-tested on Windows: 500 auto-language requests, threads and handles flat, 500/500 identical transcripts.)
  Explicit `language` and CPU/GPU are untouched.
- **Windows release binaries no longer embed the build box's paths.** The cross-built
  `rustedvino.exe` carried the cargo-registry paths (1,400+) and the OpenVINO SDK header paths of the
  machine it was built on — the build user's name included. Release builds now remap them
  (`--remap-path-prefix`, `-ffile-prefix-map`) and `scripts/make-dist.sh` scans the finished bundles
  and refuses to package one that still carries a private string.
- **`/v1/systemone`: a client that disconnected mid-request could let an eviction race the engine.**
  The in-flight counter was released with the handler's future, so `active()` read 0 while the
  engine thread was still scoring; eviction then dropped the engine and gave up waiting after 8 s,
  freeing the model's memory in the ledger while it was still resident. The counter now stays up
  until the engine has finished (or skipped) the request.
- **`/v1/systemone` could starve chat and embeddings under `device_budgets`.** The decision engine
  scores one request at a time, yet every queued request held a device lease it could not use. With
  a budget of 2, chat and embedding requests were refused with 429 while the GPU ran one thing at
  a time. On a budgeted device a decision request now waits for its model's turn first (bounded by
  `device_admission_queue_timeout_ms`, then the usual 429) and takes a lease only when it can run.
  Devices without a budget behave as before.

### Known issues
- Loading under system-RAM pressure can evict more than `rv check` predicted, including a busy model (its requests are aborted after the 15 s drain with `503 model_unloading`): victims are credited their configured size, but the OS returns less (Windows: ~2.0 GB for a model configured at 5.0). Fix planned for 0.8.1.
- The system-RAM admission check and the `rv` pre-check price a load with the full KV pool (the GPU path would shrink it) and the pre-check ignores `--kv-cache-gb`.
- Refusal text quotes VRAM figures when RAM binds, does not name the protected or busy model, and says "can never fit" for a 2 s grace-window block on the RAM path.
- A GPU context can still wedge (sticky `CL_OUT_OF_RESOURCES`); seen twice in a 6.5-hour fuzz run, not reproduced in ~5,600 churn cycles. Reported via `server.gpu_poisoned`; `--supervise` restarts the worker (Linux only; no Windows supervisor yet).
- Unloading the reranker can segfault in the OpenVINO CPU plugin (upstream), once in 1,143 rebuilds.
- Windows: no prebuilt bundle in this release (re-tested as a staged folder); no installer or service wrapper; closing the console window ends the server without a drain. `rv --version` prints 0.1.0; `rv ps` shows an NPU model's configured size, not the counted footprint.

### Changed
- `device_budgets` guidance: a budget counts every in-flight request and a chat stream holds its slot
  for the whole stream, so it caps continuous batching. Size it at or above the chat model's
  `max_num_seqs` plus headroom (see [`CONFIG.md`](CONFIG.md)); on a B70, a budget of 4 refused 64%
  of a 12-stream chat load and cut throughput by two thirds.
- An admin load refused because the model can never fit in system RAM or VRAM now returns **503**
  (`insufficient_memory`) with the same explanatory message; it was a generic 500.

## [0.7.0] — 2026-09-30

Vietnamese speech end to end — speech-to-text, text-to-speech and voice conversation — and
Lunar Lake's NPU as a fully configurable text-generation target, with several NPU responses that
were quietly wrong now right.

### Added
- **`POST /v1/audio/translations`** (OpenAI-compatible): speech in any Whisper language → English
  text, same request/response formats as transcriptions. Needs a Whisper model trained for
  translation (`whisper-large-v3` works; `large-v3-turbo` returns untranslated text). English-only
  (`.en`) models are refused with `400 model_cannot_translate`.
- **Piper TTS voices** (VITS in ONNX, e.g. Vietnamese `vi_VN-vais1000`) on `/v1/audio/speech` and
  `/v1/realtime`. Phonemes come from `espeak-ng`, run as an external program the operator installs
  (GPL-3.0, never linked or shipped); a load-time self-check refuses the voice if `espeak-ng` is
  missing or produces the wrong phonemes.
- **Vietnamese speech, end to end:** PhoWhisper (VinAI's Vietnamese Whisper fine-tune) runs as a
  Whisper STT model once converted (README has the conversion recipe and measured error rates),
  Piper's `vi_VN-vais1000` voice speaks, and the two combine into a Vietnamese voice conversation
  over `/v1/realtime`.
- **NPU `min_response_len`** (per model): the answer room the NPU's fixed cache reserves on top of
  `max_prompt_len`. Settable in config or via `PATCH /v1/admin/models/{id}`.
- **NPU LLMs use `ov_cache_dir`**: one weightless compiled blob per model (~0.66 GB for an 8B int4
  model) — cached loads in ~6 s instead of ~30 s cold, counted against `ov_cache_max_gb`.
- **Embedding limits of their own**: `max_embedding_inputs` (default 256) and
  `max_embedding_batch_tokens` (default 32768 padded tokens, which bounds GPU memory per batch).
- **Measured device memory**: `/metrics` gauges `rustedvino_process_gpu_memory_bytes` and
  `rustedvino_model_gpu_memory_estimate_bytes`, read from the kernel on Linux;
  `GET /v1/admin/models` shows `gpu_memory_estimate_gb`, and a warning is logged when a model's
  measured memory exceeds its configured `vram_gb`.
- **`tools/livecaptions`**: live captions from a microphone or any PulseAudio/PipeWire source,
  through `/v1/audio/transcriptions`, optionally translated by a chat model.
- **`tools/semsearch`**: semantic search over a folder of Markdown notes using the server's own
  `/v1/embeddings` and optional `/v1/rerank`.
- `413 request_too_large` for bodies over the 50 MiB limit, and a "Request limits and endpoint
  behaviour" section in `INTERFACE.md` (upload cap, long audio, embedding limits, realtime audio
  format).

### Changed
- Requests without a `seed` now really sample: each unseeded request gets a fresh random seed on
  every text engine (GPU, NPU, vision). Previously identical requests at `temperature > 0` returned
  identical text. An explicit `seed` is still reproducible, with one caveat: with prefix caching on
  (the default), the first request for a prompt can differ from its later repeats, which reuse its
  cached prompt blocks. Set `enable_prefix_caching: false` for strict reproducibility.
- NPU LLMs are no longer charged a GPU-style KV cache when the server checks whether a model fits
  (an 8B int4 NPU model was counted at ~12.5 GB against ~5 GB actually used). The admin API reports
  the NPU's real prompt limit as `max_prompt_tokens`.
- `POST /v1/admin/config/reload` picks up file changes to next-load settings (`max_prompt_len`,
  `min_response_len`, `kv_cache_gb`, `max_concurrent_streams`, `speculative`) for models that aren't
  loaded, and no longer reports unchanged Qwen3 models as `updated`.
- `/v1/embeddings` no longer shares the completions array cap (16). Every request valid under
  0.6.8 stays within the new defaults.

### Fixed
- **NPU:** `enable_thinking: false` is honoured (the prompt was chat-templated twice, dropping the
  "thinking off" prefill).
- **NPU:** sampling settings are applied — `temperature`, `top_p`, `top_k`, `seed`, penalties and
  `stop`. Every NPU request used to run greedy while reporting the requested values.
- **NPU:** structured output (`response_format` with a JSON schema) works.
- **NPU:** `usage.completion_tokens` is exact, and streamed text is sent as soon as it is decoded.
- **NPU:** an answer cut short by the fixed cache reports `finish_reason: "length"`, not `"stop"`;
  one ending exactly at the limit reports `"stop"`.
- **NPU:** the prompt-length check follows the length the model was actually compiled with, so an
  over-long prompt after a PATCH gets a clean `400` naming `max_prompt_len`, not OpenVINO's raw
  error.
- Clearing `max_prompt_len`, `min_response_len` or `max_concurrent_streams` via PATCH restores the
  default on the next load instead of keeping the old value until a restart.
- The GPU text path's rare string fallback no longer risks double chat-templating.
- Embedding length limit for XLM-RoBERTa models (multilingual-e5, bge-reranker-v2-m3) is 512 / 8192
  tokens, not 514 / 8194 — the extra tokens were embedded with untrained positions.
- Uploads over the body limit used to fail as `400` "could not read field 'file'" (multipart) or
  "malformed request body" (JSON).

### Security
- The `x-ruvi-host` header (the box's hostname) was sent on every response, including the open
  `/health` and `401`s. It now goes only to requests with a valid inference or admin key, and a
  server with no keys configured never sends it. The same rule applies to `host` in image
  `generation_metadata` and in `GET /v1/admin/models`. **Upgrade note:** tooling that reads the
  header must send a valid key.

## [0.6.8] — 2026-09-20

First public release: an OpenAI-compatible inference server for Intel hardware, written in Rust
as the successor to the Python stormVINO.

- Serves chat completions (streaming and non-streaming), embeddings, reranking, speech-to-text,
  text-to-speech, image generation, vision, a realtime voice WebSocket, an admin API and Prometheus
  metrics.
- Runs on Arc dGPUs (B50 / B60 / B70), Lunar Lake's integrated GPU, the NPU and CPU, with
  capability-based placement instead of hardcoded device names.
- Continuous batching through OpenVINO GenAI's `ContinuousBatchingPipeline`: on an Arc Pro B60,
  69 tok/s for one user and 814 tok/s aggregate for 16 users.
- Self-contained Linux bundle with the OpenVINO 2026.2.1 runtime inside (glibc ≥ 2.39).

[0.7.0]: https://github.com/Jermalk/rustedvino-oe/compare/v0.6.8...v0.7.0
[0.6.8]: https://github.com/Jermalk/rustedvino-oe/releases/tag/v0.6.8
