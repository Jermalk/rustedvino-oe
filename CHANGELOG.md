# Changelog

All notable changes to RustedVINO are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and from 0.7.0 on the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (0.6.8 was chosen for luck, not as a
semver claim). Full release notes, with measurements and download instructions, are on each
[GitHub release](https://github.com/Jermalk/rustedvino-oe/releases).

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
- `POST /v1/admin/models/add` answers a JSON body, not plain text. `POST /v1/admin/config/reload` with an invalid `config.json` answers 422 (was 500).
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
