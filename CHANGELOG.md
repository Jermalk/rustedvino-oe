# Changelog

All notable changes to RustedVINO are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and from 0.7.0 on the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (0.6.8 was chosen for luck, not as a
semver claim). Full release notes, with measurements and download instructions, are on each
[GitHub release](https://github.com/Jermalk/rustedvino-oe/releases).

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
