# RustedVINO — Config Reference

Full field reference for `config.json`. For the setup walkthrough (creating the file, sample
config, running the server) see **[`README.md`](README.md)**.

Every model lives under `models` as one stanza combining its VRAM estimate, kind, and
per-model policy overrides — there is no separate `vram_gb`/`model_kinds`/`model_policies`
map. A model ID not listed here is unknown (404 everywhere), even if its directory exists
under `models_dir`.

## Every setting is written out

`config.json` is plain JSON, so it cannot carry comments: the file holds the **values** and this
document holds the **meaning**. A setting left out of the file silently takes a built-in default,
and a setting you cannot see is one you forget you have, so the starter
`scripts/dist.example.config.json` (shipped in the release bundle as `config.example.json`) lists
**all** global settings, each at the value in effect (its `models` map is empty;
`--print-default-model`, below, prints a stanza with every per-model setting). Writing the default out changes nothing about how the
server behaves (checked when this was done: every shipped config parsed to the identical
configuration afterwards). `null` is accepted only where the default column below shows `null`
(it then means "not set / automatic", as the row says). For every other setting `null` is a
startup error: to use the built-in default, leave the key out or write the default value.

- `rustedvino --print-default-config` prints every global setting with its default (`models_dir`,
  `device` and `total_vram_gb` carry placeholders: they have no default). `rustedvino
  --print-default-model` prints one `models.<id>` stanza the same way. Its `vram_gb: 0.0` is a
  placeholder too, and not a safe one: `0.0` skips VRAM gating for the model, so a stanza pasted
  as printed is silently ungated. Write the model's real footprint (GiB) there; on Linux,
  once the model is loaded, `GET /v1/admin/models` shows what it uses (`gpu_memory_estimate_gb`,
  see `vram_gb` in the per-model table below).
- `scripts/fill-config-defaults.py CONFIG...` adds whatever your file is missing, using exactly
  those defaults. It never changes a value you set, keeps keys it does not know, and rewrites the
  file in a fixed order (globals in the order `--print-default-config` prints them, which is the
  order of the `Config` struct and not the order of the tables below; `models` last; per model
  `vram_gb`, `kind`, then the policy fields). `--check` writes nothing and exits 1 if something is missing - run it after
  an upgrade to see which settings the new version added.
- The placeholders `--print-default-config` prints are not safe defaults to run with: `total_vram_gb: 0.0` (the release bundle ships this) budgets a **discrete-GPU** inference device at the memory it reports, as every other discrete GPU is, and switches the budget OFF for any other inference device: the CPU, an integrated GPU, the NPU, a composite `HETERO:`/`MULTI:`/`AUTO:` spec, or a dGPU that reports no figure. To switch a dGPU's gating off on purpose, set `domain_budgets` with the device name as `--device-info` prints it: `{"GPU": 0.0}` on a one-GPU box, `{"GPU.1": 0.0}` on a two-GPU one (a name the host does not have is a startup warning). System RAM is still gated live (MemAvailable minus the OS floor: 4 GiB on Linux, 6 GiB on Windows; on Linux also minus what loaded CPU models are charged but do not hold yet, since the CPU plugin copies a model's weights into memory only at its first request — so `rv check`'s "system RAM … available" can read lower than `free`). A non-zero value is the budget, capped at what the device reports.
- `api_keys` / `admin_api_keys` are deliberately not in the files: they are deprecated and a
  non-empty value is a startup error (keys live in the keys file, see `keys_file`).
- When a new version changes what a setting does by default, its release notes say so, and the
  starter config names the value explicitly, so an upgrade never changes behaviour behind your
  back for a setting you wrote down. `tests/shipped_configs.rs` and `tests/config_reference.rs`
  fail `cargo test` if a setting is missing from the starter config or from this document.
- **Invalid config = exit 78.** A config file that cannot be read or parsed, a wrong JSON type
  (including a `null` where none is accepted), an unsupported value (for example the former
  `kv_cache_precision: "f32"`, accepted by 0.8.x), a value outside the limits the tables below give for it (*(since 0.9.1)* also a `0` that
  would break the server rather than turn something off, see [Zero values](#zero-values)), a
  `preload` id without a `models` entry, *(since 0.9.1)* a `models` id that is not a plain folder name (a `/`,
  `\`, `..`, `.`, or a drive-letter start such as `C:` or `m:7b`; `qwen:7b` is fine), a security-looking unknown key (next bullet) and an
  unknown global `device` with `default_placement: "device"`, and a non-loopback `bind_addr`
  with no `api_keys` in the keys file and no `allow_insecure_public_bind` (admin keys alone do
  not count *(since 0.9.1)*) all stop the server before it serves, with exit code 78 (`EX_CONFIG`);
  `--supervise` does not restart it. Non-empty inline `api_keys`/`admin_api_keys` and an invalid
  `bind_addr` are refused with exit 1, which the supervisor retries (up to `max_restarts` within
  `restart_window_secs`). `POST /v1/admin/config/reload` with a file that fails to parse or
  validate answers 422 and changes nothing.
- **Unknown keys.** An unknown top-level key is ignored with a WARN naming it; one whose name
  contains `key`, `cors`, `auth`, `bind` or `insecure` stops startup (exit 78), because a typo
  there would silently leave the server open. Inside a `models.<id>` stanza and inside
  `supervisor`, an unknown key (a typo) is ignored **without a warning** and the default applies;
  `scripts/fill-config-defaults.py --check` then reports the correctly spelled key as missing.
  Inside `speculative` an unknown key is a startup error.
- **Reload applies `models` only.** Every global setting in the table below is read at startup:
  `POST /v1/admin/config/reload` re-reads only the `models` registry (see `INTERFACE.md` for what
  it does to loaded models), and lists any global whose value changed in `globals_not_applied`;
  restart to apply those. A loaded model is never changed, except `sampling` *(since 0.9.1)*: an edited
  `sampling` on a `Ready` model applies to its next request and is listed in `applied_live`. Its other
  edited keys are listed in `models_not_applied`, and apply only after unload → reload → load (a plain unload + load keeps the
  entry it was loaded with). Keys have their own reload (`keys_file`).

### Zero values

Every numeric setting either has a documented meaning at `0` (its row says what, often "off",
"unbounded" or "no waiting") or refuses `0` at startup (exit 78). *(since 0.9.1)* `0` is refused for
`max_prompt_array`, `ov_cache_sweep_interval_secs`, `kv_pressure_sweep_interval_secs` and, in
`supervisor`, `watchdog_poll_secs`, `max_restarts`, `health_timeout_secs` and `restart_window_secs`:
none of these meant "off" at `0`; each broke the server instead (a busy loop, every request refused,
a supervisor that crashed, gave up or never stopped restarting). Also refused at `0`:
`max_embedding_inputs`, `max_audio_duration_secs`, `max_images_per_request`, the image pixel limits,
`kv_headroom_min_tokens`, and the factors that must be at least 1 (`kv_transient_factor`,
`npu_footprint_factor`). The two sweep intervals and every `supervisor` seconds field are also at
most 31 536 000 (one year). A test (`every_numeric_knob_at_zero_is_documented_or_refused`) fails
when a new numeric setting is neither.

**`Config` fields** (see `src/model_manager/config.rs` for full doc comments):

| Field | Type / default | Description |
|---|---|---|
| `models_dir` | `PathBuf`, required | Directory of model subdirectories, one per model ID. A relative path resolves against the server's **working directory**, not the config file's directory (unlike `chat_template`), so a server started from another directory looks somewhere else: prefer an absolute path. *(since 0.9.1)* When it does not exist, is not a directory or cannot be read, the server boots with a WARN and a `models_dir_missing` startup warning; when it holds no model directory (no subdirectory whose name does not start with `.`), a `models_dir_empty` one. Both name the absolute path checked (and the working directory, for a relative path); `rv status` prints them, `/health` shows their codes; both are cleared once a model is loaded |
| `device` | `String`, required | OpenVINO device string for inference (`"GPU.1"`, `"GPU"`, `"CPU"`, …). It is the memory-budget anchor; models are placed on their own resolved device. If it is not an enumerated device the server boots with a `global_device_unknown` startup warning (`rv start` / `rv status` print it) — except with `default_placement: "device"`, where no model without its own `device` could load, so the boot is refused (exit 78, config error) |
| `preload` | `[]` | Model IDs to load at startup; each must have a `models` entry. Manage it at runtime with `POST /v1/admin/config/preload` instead of hand-editing this array — `{"from_live": true}` snapshots whichever models are currently loaded as the new list. Loaded in order, and **a preload never evicts another preload** *(since 0.9.1)*: a model that does not fit beside the ones already loaded is skipped (a WARN and a `preload_skipped` startup warning naming the model, the gate that refused it — the memory budget or live system RAM — with that gate's need and the shortfall in GB; cleared once the model is loaded), stays unloaded, and the rest of the list still loads; `rv load` it later (a runtime load may evict). One exception keeps its own wording: a preload refused by the device-headroom rule (a vision model on a discrete GPU, or a model preloaded after one on that GPU; checked at boot *(since 0.9.1)*) is a `preload_failed` warning. A vision or text model whose KV pool is sized to the room left still loads when it fits at its minimum KV, so put it last. `/health` shows only the count and codes of the startup warnings; the text is in `rv status` and `GET /v1/admin/models`. After a supervised crash respawn only the preload models loaded at the crash load (see the restore paragraph under `--supervise`) |
| `models` | `{}` | Map of model ID → `{vram_gb, kind, ...per-model policy}` — see `ModelEntry`/`ModelPolicy` below |
| `total_vram_gb` | required | Usable VRAM on the inference device, in GiB (the unit the memory accounting, the KV pools and the device inventory use; an Arc Pro B60 reports 22.7); `0.0` = the memory a discrete GPU reports, and no VRAM gating on any other device. A value above what the device reports is capped at the reported figure. Memory figures in this file named `*_gb` (this key, `vram_gb`, `kv_cache_gb`, `cache_size_gb`, `vram_safety_margin_gb`, `min_kv_cache_gb`) are GiB, and the `rustedvino_vram_*_bytes` gauges convert them with 2^30 |
| `max_num_seqs` | `16` | CB scheduler concurrency cap and the HTTP 429 admission gate; `0` = OV default (256). For a **vision** model it sets only the admission gate (queue size); the vision engine runs one generation at a time, so its scheduler is always told one sequence (0.8.4) |
| `cache_size_gb` | `0.0` | Cap on the KV cache pool; `0.0` = uncapped (model grabs all remaining VRAM). Do not combine `0.0` with `enable_prefix_caching: true`: retained prefix blocks then grow outside the memory budget (the server warns at boot). The starter config (`config.example.json` in the bundle) sets `2.0`. The cap also limits an explicit `kv_cache_gb` (per model or per load): asking for 4 with a cap of 2 gives 2, and `rv add` / `rv resize` / `GET /v1/admin/models` report it (`kv_requested_gb`, `clamped_by`). Sizes are GiB. A model on a plain GPU whose KV layout the server knows (explicit precision, confident profile, no speculative draft, a verified OpenVINO/GenAI pair) gets its pool sized in whole KV blocks and booked as built, under either `kv_gate_mode`; any other pool (CPU, HETERO/AUTO, `kv_precision_pass: "plugin"`, a model the server cannot describe) is allocated in whole GiB, a fractional pool rounded up (down if the rounded pool would not fit the free memory), and that is what is booked |
| `default_kv_cache_gb` | `0.0` | Global default bounded KV-pool target (GiB) per model; `0.0` = unbounded |
| `vram_safety_margin_gb` | `1.0` | VRAM (GiB) held back for driver/runtime overhead. With `total_vram_gb > 0`, this plus `min_kv_cache_gb` must not exceed `total_vram_gb`, or startup is refused. `0.0` = no margin |
| `npu_footprint_factor` | `1.5` | Multiplier on a model's `vram_gb` when it runs on the NPU, for admission only (the listed figure is unchanged). An NPU model's real footprint exceeds its weights: Windows measured 7.6-8.4 GB of NPU adapter memory for an 8B int4 model reserved at 5.5 GB, which let a GPU model next to it take a KV pool the shared budget could not back. Linux's per-process DRM accounting measures the same model at ~5.2 GB, so a Linux box can set `1.0`. Must be >= 1.0 |
| `cpu_image_gen_max_pixels` | `1048576` | Largest image request (total pixels, `n × width × height`) an image-generation model on the **CPU** serves, and the size its memory charge is sized for. On the CPU a diffusion pipeline keeps a working set that depends on its family and the image size, not on the weights, and does not give it back after the image; admission charges such a model `vram_gb` **plus** that working set at this size, by the family read from `model_index.json` (at 1024×1024: SD 1.5/LCM ≈ 10.2 GiB, SDXL ≈ 12.4, FLUX ≈ 15.1, SD3 ≈ 24.5; an unreadable or unknown family is charged as SD3). A request above it is a 400 `size_exceeds_cpu_limit` (also `/v1/images/edits`, by the uploaded image's size). Measured once per point on one desktop CPU at 512×512 and 1024×1024 (n=1; a second image in one request added about a third); above the default the charge extrapolates. GPU placements are not affected. At least 65536 (256×256). Restart-only (a config reload reports it as not applied) |
| `max_audio_duration_secs` | `7200` | *(since 0.9.1)* Longest audio upload `/v1/audio/transcriptions` and `/v1/audio/translations` decode, in seconds of audio (at least 1). Longer audio is a `400 invalid_audio` that names the limit: refused before decoding when the file declares its length, otherwise as soon as decoding passes it. The request-size limit bounds only the compressed file, and a very compressible one (silence, a very low bitrate) decodes to far more. Worst-case memory while decoding is this × 48 000 × 4 bytes (source rates from 4 kHz to 48 kHz are accepted; outside that range is a `400`) plus this × 16 000 × 4 bytes for the 16 kHz copy: about 1.4 GB + 0.5 GB at the default. Formats: WAV, MP3, FLAC, Ogg Vorbis, AAC and M4A/MP4 with AAC; ALAC is not decoded. `/v1/realtime` is not affected (it commits at most 29.5 s per turn). Restart-only |
| `max_images_per_request` | `16` | *(since 0.9.1)* Most images one request may carry: the `image_url` parts of the last chat message to a vision model, or the `image` and `mask` of `/v1/images/edits`. Over it is a `400 too_many_images`. At least 1. **The shipped `config.example.json` sets `4`** (0.9.1 known issue): the server's own memory per image is bounded, but the vision engine's is not — on an Arc Pro B70 with 62 GB of RAM, 7 images at the 4 MP `max_image_pixels` (qwen3.6-27b) grew the process to 44 GB and the kernel killed it, while 4 such images worked. Since every chat image reaches the model at most `max_image_pixels`, this × `max_image_pixels` bounds what one request feeds the engine; raise it only on a box with RAM to spare. Restart-only |
| `max_image_pixels_per_request` | `67108864` | *(since 0.9.1)* Most pixels (`width × height`, summed over the request's images, the edit mask included) one request may decode. Each image's size is read from its header and the request refused with a `400 image_too_large` before any pixel is decoded: the request-size limit bounds only the compressed bytes, and a few-KB PNG can declare 65535 × 65535. Decoded RGB is 3 bytes per pixel: the default (64 MP: one 50 MP photo, or 16 screenshots at 2560 × 1600) is about 192 MB, up to about 512 MB while a 16-bit PNG decodes. At least 65 536. Restart-only |
| `max_image_pixels` | `4000000` | *(since 0.9.1)* Most pixels (`width × height`) one image may have when it reaches a vision model. A chat image over it is downscaled after decoding (both sides by the same factor, aspect ratio kept; an integer area-average first when a side shrinks 2× or more, then bicubic) to at most this many pixels; an image at or under it is passed through untouched. Per image, unlike `max_image_pixels_per_request`, which still counts every image's original (header) size. The vision encoder's memory grows with the image (qwen3.5-9B on an Arc 140V iGPU, one image: 1 MP ≈ 0.65 GiB / 6 s, 2 MP ≈ 1.15 GiB / 14 s, 4 MP ≈ 3 GiB / 65 s, 8 MP out of GPU memory), so the default 4 MP (decimal) keeps one large photo from taking a UMA box down. The model's own preprocessor (`preprocessor_config.json`) may resize further after this; this cap only makes the image smaller, never larger. A value above `max_image_pixels_per_request` never fires. Image edits are not affected. Server memory per image: the decode (for a request at the 64 MP `max_image_pixels_per_request` default: ~192 MiB for 8-bit RGB, up to ~0.7 GiB for a 16-bit PNG, up to ~1.1 GiB for a progressive or CMYK JPEG) plus the downscale's working set, at most about 32 bytes × `max_image_pixels` (~128 MB at the default; it grows with this cap). At most two vision decodes and two image-edit decodes run at once server-wide, so the worst case in flight is about 4.7 GiB; a typical 12 MP photo needs ~36 MB. At least 65 536. Restart-only |
| `min_kv_cache_gb` | `1.0` | Minimum KV pool (GiB); a load that would leave less is refused. With `total_vram_gb > 0`, this plus `vram_safety_margin_gb` must not exceed `total_vram_gb`, or startup is refused. `0.0` = no floor |
| `system_ram_reservation_gb` | `null` | System RAM (GiB) held back from the shared "system" domain budget; `null` = per-OS default |
| `system_ram_budget_gb` | `null` | **UMA boxes only.** System RAM (GiB) the server may treat as its own — the unified-memory twin of `total_vram_gb`. `null` keeps the historical live gate (`MemAvailable` alone, less the not-yet-resident charge of loaded CPU models on Linux). When set, the gate uses `min(MemAvailable + reclaimable GPU page cache, this)`, because `MemAvailable` **omits** the DRM/TTM page pool — freed GPU pages the kernel holds for reuse, counted as `used`. Measured once: that pool held ~8.8 GB and a 12 GB model was refused on a 30 GB box as "can never fit". Opt-in because crediting reclaimable cache is the less conservative direction; `system_ram_reservation_gb` still applies as an OS floor on top. Must be > 0 when set (leave it `null` to keep the live gate) |
| `default_placement` | `"auto"` | How a model with no `device` and no `tier_preference` of its own is placed. `"auto"`: the capability-aware engine - chat LLMs, embedders and rerankers take a discrete GPU when there is one (an Arc Pro B50 ran a 1.7B model 3x faster than its CPU) and the CPU otherwise, STT goes CPU then a discrete GPU (never the NPU or an iGPU unless you name it: whisper-turbo hung the NPU on a clip over 60 s), TTS is CPU-only, **an integrated GPU only as a last resort** (it measured 20-45 % slower than the CPU on a UHD iGPU; a good iGPU such as an Arc 140V is the exception - name it with `device` or `tier_preference`). `"device"`: the global `device`, nothing else, so `device: "CPU"` really means CPU for every model (the release bundle ships this; the global `device` must then be a device OpenVINO enumerated, and a bare `"GPU"` is rejected on a multi-GPU host - use `GPU.n`). A per-model `device` or `tier_preference` always wins; `embedding_device` still wins for embeddings. Hybrid linear-attention models (Qwen3.5/3.6) are placed like any other chat LLM since 0.9.0 (0.8.x kept them off every GPU under `"auto"`; see the per-model `enable_prefix_caching`) |
| `light_model_max_gb` | `2.0` | Size threshold (GiB, compared with `vram_gb`) below which an LLM/VLM is "light" for auto placement; the engine never puts an iGPU before the CPU, so this only decides light vs heavy. `0.0` = every LLM/VLM is "heavy" |
| `light_stt_max_gb` | `1.0` | Kept for config compatibility: thin and heavy Whisper models now share one default order (CPU, dGPU, iGPU last; never the NPU), so this no longer changes `"auto"` placement |
| `dgpu_size_ceiling_fraction` | `0.8` | Default-placement guard: a model whose `vram_gb` exceeds this fraction of a discrete GPU's usable VRAM is skipped for that device in *default* tier preference; explicit `device`/`tier_preference` sidesteps it. `>= 1.0` disables. "Usable VRAM" is `total_vram_gb`; when that is `0.0` (as in the release bundle) the dGPU's own reported memory (GiB) is used, and with neither the guard is off. `0.0` = default placement skips every discrete GPU |
| `domain_budgets` | `{}` | Per-memory-domain VRAM budget overrides (GiB), keyed by domain id (a discrete GPU's device name as `--device-info` prints it, or `"system"`); each value must be a finite number >= 0; `0.0` switches that domain's gating off; a key the host does not have is a startup warning and does nothing |
| `device_budgets` | `{}` | Cross-pipeline device admission caps, keyed by live OV device string; empty = passthrough. Covers chat, VLM, embeddings, STT, TTS, realtime and `/v1/systemone` (one lease per request) — a budget counts every in-flight request, and a chat stream holds its slot for its whole stream, so it caps continuous batching: size it at or above the chat model's `max_num_seqs` plus headroom (one decision turn, embeddings, rerank) — e.g. `{"GPU.1": 20}` for a 16-sequence chat model. On a B70, 4 refused 64% of a 12-stream chat load and cut throughput by two thirds; a decision model no longer needs a small budget to avoid starving chat, because its requests take one lease at a time. Each value must be at least 1: omit the entry rather than set `0`, which would block the device |
| `device_admission_queue_timeout_ms` | `5000` | Queue timeout for `device_budgets` (separate from `admission_queue_timeout_ms`). `0` = no waiting, as for `admission_queue_timeout_ms` |
| `kv_cache_precision` | `"u8"` | Precision of the CB (text) engine's KV cache. Accepted: `"auto"` (or `""`), `"i8"`, `"u4"`, `"f16"`; any other value (including the former `"f32"`) stops startup. `"u8"` is the pre-0.9.0 spelling of `"i8"`, kept as the built-in default so existing configs parse unchanged: it is stored and passed as `i8` and logged once at INFO as a deprecated alias; the starter config writes `"i8"`. `"auto"` (or `""`): with `kv_precision_pass: "explicit"` on a GPU the model's own IR hint or the rule (u4 for 4-bit weights, otherwise i8) decides, and the result is passed explicitly; the OpenVINO plugin decides only on a CPU or HETERO/AUTO device, with `kv_precision_pass: "plugin"`, or when the server cannot tell the model's weight format with confidence. Any other value (`i8`, so also the default `u8`, `u4`, `f16`) is used for every text model on a GPU in place of the model's IR hint and the 4-bit rule: set `"auto"` to let int4 text models get u4 (smaller per token, so more context in the same pool), or set the per-model `kv_cache_precision`. Measured on GenAI 2026.4.1 (GPU): the plugin's own default is **u4 for models with int4 weights** and i8 otherwise (f16 only when the model's IR asks for it), the same as the rule; an explicit `i8` (and so the default `"u8"`) is about 1.8× **larger** per token than u4 (Qwen3.8-27B: 37,134 vs 20,748 B/token measured), `f16` about 3×. The global key applies to the CB engine only: vision-language (VLM) engines resolve their own precision (see `kv_precision_pass`). With `kv_gate_mode: "exact"` (default) the capacity gate (`max_model_len`) uses the real per-layer layout of the resolved precision. Only the legacy estimate (`kv_gate_mode: "legacy"`, or a model the server cannot describe with confidence) prices `auto` as f16 and `u4` like `i8`: conservative for those and for hybrid models (it counts every layer), but about 14 % optimistic for a dense model on `i8` (it ignores the scale overhead). On a CPU device `i8` is passed as `u8` (the CPU plugin has no i8 KV cache). Read at startup; a reload does not apply it |
| `enable_prefix_caching` | `true` | Reuse KV blocks across CB requests sharing an identical prompt prefix. Side effect on seeded sampling: a reused-prefix prefill rounds slightly differently from a full one, so the first request for a prompt can differ from its later identical repeats (the repeats match each other). Set `false` when exact reproducibility of the first answer matters. **Hybrid linear-attention models (Qwen3.5, Qwen3.6, Qwen3.8) gain much less** (measured on 0.9.1-rc.2): on the CPU a repeated prompt is processed in full every time; on an Intel GPU only an append-only conversation reuses the cache (the next turn of a 2,029-token conversation started in 0.7 s instead of 6–7 s), while sending the same prompt again does not. Plain Qwen3 models reuse in every case (2,029 tokens: 26 s, then 0.3 s on the CPU). For an agent loop on a CPU-only machine, prefer a plain Qwen3 model |
| `cors_allowed_origins` | `["*"]` | CORS allowed origins. The default allows any origin (methods and headers too, no credentials), which suits browser tools but means that on a server with **no API keys**, any web page you open in a browser on the same machine can call its inference routes (the admin routes and `/metrics` of a server with no keys at all refuse any request that carries an `Origin` header, 403 `browser_origin_refused`). List your front-end origins explicitly, or configure keys, anywhere the host also runs a browser |
| `api_keys` | `[]` | **Deprecated, migration tripwire only — never read for auth.** A non-empty value is a hard startup error naming where to move the keys. Real inference-scope keys live in `keys_file` |
| `admin_api_keys` | `[]` | **Deprecated, same tripwire as `api_keys`.** Real admin-scope keys live in `keys_file` |
| `keys_file` | `null` | Path to the real Bearer-key file (`{"api_keys": [...], "admin_api_keys": [...]}`), never `config.json` itself. `null` resolves to `<config-dir>/<config-stem>.keys.json` (e.g. `config.json` → `config.keys.json`); a relative value resolves against the config file's own directory. Never git-tracked (`.gitignore`: `*.keys.json`); should be mode `0600`. **If no keys file exists at boot, every `/v1/admin/*` route and `/metrics` return a permanent `503 admin_not_configured` for that process's lifetime** — creating the file afterward needs a restart (or `POST /v1/admin/keys/reload` once admin is unlocked some other way). An existing-but-empty keys file (`{}`) is valid and unlocks admin with no auth required. The release bundle ships an empty template as `config.keys.example.json`; copy it to the name above (`config.keys.json` next to `config.json`) and `chmod 600` it. With its lists empty it behaves like `{}`: add keys before any non-loopback bind. With `api_keys` set and `admin_api_keys` empty, every inference key is also an admin key (admin routes and `/metrics` fall back to the inference gate); set `admin_api_keys` to separate them |
| `admission_queue_timeout_ms` | `5000` | How long to queue a request before returning 429. `0` = no waiting: a request that finds every slot busy gets the 429 at once |
| `eviction_grace_secs` | `2.0` | Window (seconds) after a model was last used or loaded before it becomes eviction-eligible; `0.0` disables the grace window. A model with a request running is **never evicted** by a plain load, unload, deregister or resize: the request finishes undisturbed and the operation is refused at once with `model_busy` (retryable). `force` (`rv load --force`, `rv unload --force`, `rv deregister --force`, `rv resize --force`, or `"force": true` / `?force=true` on the admin routes) skips this window **and** the busy rule: the model is drained (15 s for a load, up to 2 minutes for an unload) and what is left is aborted. `force` never overrides `pinned`, `evictable: false` or the models of a live realtime session. Was `300.0` before 0.8.0; set it back on a box where voice clients must not displace an operator-loaded model. The starter config states it explicitly, so it is visible and editable in the file rather than a default you have to remember (pinned by `tests/shipped_configs.rs`). Per-model `eviction_grace_secs` overrides this |
| `realtime_defaults` | `null` | Fallback models for realtime voice sessions: `{stt_model, llm_model, tts_model}`, used when a session's `config` is resolved and the requested model (or none) cannot be served; nothing is set at connect (see [REALTIME.md](REALTIME.md) "Model resolution"). An `llm_model` placed on the NPU is ignored. `embed_model` is accepted but not read: the realtime embedding model comes from `default_embed_model` |
| `realtime_viable_minimum` | `null` | Minimum LLM viability for realtime voice sessions: `{llm: {kinds, min_context_tokens, require_capabilities}}`; `kinds` defaults to `["text_gen","vision"]` |
| `embedding_device` | `null` | OpenVINO device for embedding models; `null` = auto-select |
| `default_embed_model` | `null` | Embedding model for realtime memory retrieval: a new session starts with it, and a `config` can change it until the model is fixed for the connection (at connect if it is loaded then, else at the first turn it is loaded; see [REALTIME.md](REALTIME.md) "Embedding model") |
| `embedding_pooling` | `"mean"` | `"mean"`, `"cls"`, or `"last_token"` (`"last"` is accepted as `"last_token"`). An unrecognised value is not refused: it falls back to `"mean"` |
| `embedding_normalize` | `true` | L2-normalize embedding vectors |
| `max_prompt_array` | `16` | Max prompts accepted in one legacy `/v1/completions` array request (each one is a full generation); more is a `400 too_many_prompts`. Does not apply to `/v1/embeddings`. At least 1 *(since 0.9.1)*: `0` would refuse every request, so it stops startup |
| `max_embedding_inputs` | `256` | Max inputs in one `/v1/embeddings` request (`400 too_many_inputs` beyond it); must be `≥ 1` |
| `max_embedding_batch_tokens` | `32768` | Per-request `/v1/embeddings` budget in *padded* tokens — inputs × the longest input's token count, because a batch is padded to its longest input (`400 batch_too_large` beyond it; `0` = no budget). It bounds the batch's GPU working memory, which the server's memory accounting doesn't see and which is kept until the model is unloaded: ≈28 MB per 512-token input measured on multilingual-e5-large. The default is 64 full 512-token chunks; short queries are cheap (256 queries of ~10 tokens ≈ 2.6k) |
| `max_tokens_cap` | `8192` | Server-wide ceiling on `max_tokens`/`max_completion_tokens`; `0` = uncapped. With `0`, a vision request without `max_tokens` on a discrete GPU (exact vision gate) gets the room its model's KV pool has left after the prompt as its budget. On a shared-memory GPU (an integrated GPU such as Lunar Lake), a discrete GPU without a readable free-memory probe, or anywhere else the vision gate is legacy, the budget stays 0 (the engine's own default length) and the generation watchdog allows prompt time only, so a long answer there can be killed after `watchdog_hang_ceiling_secs` |
| `stream_first_token_timeout_secs` | `600` | *(since 0.9.1)* Engine stall watchdog, first deadline: the most seconds one engine call may run before its first progress signal. Watched per engine (the text engine's batch, a vision or NPU engine), not per request, and independent of any client connection: a client that disconnects does not end the watching. Progress is: on the text (continuous-batching) engine, every completed scheduler step — a chunked prefill step (256 tokens for the whole batch), a step in which a preempted or starved sequence gets nothing, a step whose tokens decode to no text; on the vision and NPU engines, every streamer callback, which fires on decoded text only (reasoning and tool-call text count). An idle engine is never stalled; time a request waits for an engine slot is not counted. When the deadline passes the engine is marked dead, once: one ERROR line names the model, its requests, the deadline and the elapsed seconds; every request queued on or running in that engine ends with `generation_stalled` (`504` + `Retry-After: 30`, or one SSE error frame with that code, then `[DONE]`); later requests to the model are refused. OpenVINO cannot abort a call that does not return, so the thread stays stuck and keeps the model's memory booked until it really exits. A GPU model sets the GPU-poisoned flag: under `--supervise` the worker drains and exits 75 and the supervisor respawns it; without `--supervise` the server stays up and only the stalled GPU model is refused (`503 gpu_poisoned`) until a restart, other models keep serving. A CPU or NPU model is booked as `stalled` (`rv status`, `GET /v1/admin/models`) and answers `503 model_unavailable`; an unload of it does not free its memory while the call is stuck (the model stays `stalled` and `evicting`, a reload is refused) — a hung call needs a server restart; under `--supervise` the generation watchdog (below) restarts the worker by itself for a CPU text or vision model once the call passes its allowance plus `watchdog_hang_ceiling_secs`, but it does not watch NPU calls, so a stuck NPU model needs a manual restart (`rv restart`) also under `--supervise`. Only if the stuck call returns late does an unload then a load bring the model back. Known limits: on the vision and NPU engines a long run of tokens that decode to no text (skipped special tokens) reads as silence; a vision model's prefill (images included) is one call with no progress signal inside it, so it must finish within this deadline (a 4 MP image prefill took about 65 s on an Arc 140V); and on a CPU a text-engine prefill step or a long NPU/vision prompt runs much slower than on a GPU. The default, 600 s, covers about 120 000 prompt tokens at 200 tok/s, the slowest prefill the generation watchdog assumes. Raise it if a long prompt on a slow CPU needs longer before its first progress. The deadline is seen at most one sampling tick late (a twentieth of the shorter deadline, at most 5 s), never early. `0` = off. Restart-only |
| `stream_token_gap_timeout_secs` | `120` | *(since 0.9.1)* Engine stall watchdog, second deadline: the longest gap, in seconds, between two progress signals of an engine that has work (see `stream_first_token_timeout_secs` for what counts as progress and what happens). A healthy text engine completes a step every few milliseconds to seconds, and a healthy vision or NPU engine does not go 2 minutes between two pieces of text. `0` = off. Restart-only. Relation to the supervisor's generation watchdog (`--supervise`, below): both stay; this one usually fires first (a stalled engine is ended after at most 10 or 2 minutes, while the supervisor allows the generation's own allowance plus `watchdog_hang_ceiling_secs`). For a GPU model the stall then restarts the worker at once (exit 75). For a CPU model the requests end and the model is refused, but the stuck engine call still counts for the supervisor's watchdog (text-engine steps and vision generations are watched), so under `--supervise` the worker is still restarted when that call passes its allowance plus the ceiling. NPU calls are not watched by the supervisor: a stuck NPU model stays refused until a manual restart |
| `bind_addr` | `"127.0.0.1"` | Bind address. For LAN access set `"0.0.0.0"` **and** put at least one `api_keys` entry in the keys file (the safe way). Admin keys alone do not count *(since 0.9.1)*: they protect only `/v1/admin/*`, so a non-loopback bind with no `api_keys` refuses to start (exit 78) unless `allow_insecure_public_bind: true` |
| `port` | `11437` | HTTP port. `0` lets the operating system pick a free port, which nothing reports back (the log, `rv` and `--supervise` all see 0): not for normal use |
| `allow_insecure_public_bind` | `false` | Explicit opt-in to a non-loopback bind with no `api_keys` in the keys file: inference is open to the network (admin keys, if any, still protect `/v1/admin/*`). The server logs a WARN at startup. Not required when the keys file has `api_keys` |
| `ov_cache_dir` | `null` | Directory for the OpenVINO compiled-blob cache (GPU kernels, and NPU LLMs' weightless compiled blob: ~6 s cached load vs ~30 s cold, measured on one Lunar Lake laptop). NPU Whisper uses it too (a load with a cache dir completes on the current stack; cold ~4 s, warm ~0.3 s). Each NPU `max_prompt_len`/`min_response_len` combination adds its own blob (~0.66 GB for an 8B int4 model); blobs from earlier combinations stay attributed to the model, so the `ov_cache_max_gb` sweep won't prune them while it is loaded (and never with `ov_cache_max_gb: 0`) — after experimenting with shapes, stop the server and clear the cache dir (the next load recompiles, ~30 s). Once unloaded, a model's blobs become prunable, but the sweep then runs only every `ov_cache_sweep_interval_secs` and deletes oldest-first until under the cap, not stale shapes specifically. With `ov_cache_layout: keyed` (default) the text and vision models compile into `<ov_cache_dir>/kv-<tag>/`, one directory per KV precision (see `ov_cache_layout`). RustedVINO also maintains a `cache_manifest/` directory — one JSON file per model, recording OV blob attribution and (independent of `ov_cache_dir`) image-gen `model_hash` memoization across restarts. Placed alongside `ov_cache_dir` as its sibling when `ov_cache_dir` is set, otherwise `$XDG_CACHE_HOME/rustedvino/cache_manifest`, `~/.cache/rustedvino/cache_manifest` or (Windows, where neither is set) `%LOCALAPPDATA%\rustedvino\cache_manifest` — not separately configurable. The learned KV-capacity limits (the ratchet, see `kv_gate_mode`) are stored the same way, in a `kv_ratchet/` directory (one `<model>-<hash>` folder per model) beside `ov_cache_dir` when it is set, otherwise beside `cache_manifest/` in the per-user cache; they are never written into the model library |
| `ov_cache_max_gb` | `0.0` | Ceiling on `ov_cache_dir`'s total on-disk size (GiB); `0.0` = unbounded. Enforced by a sweep that runs once at startup and then on `ov_cache_sweep_interval_secs`, pruning once the cap is exceeded — blobs for models no longer in your config go first, then blobs a configured model is not using (not resident, another device, an older layout or precision), then unattributed files older than 30 minutes, oldest-modified first within each group. What a `Ready` or `Loading` model uses (its blobs on its device, in the directory it compiles into) is never deleted; `*.cl_cache` kernel caches never are. `0.0` still reports the cache size every sweep; set a non-zero value only if you want files deleted |
| `ov_cache_sweep_interval_secs` | `21600` (6h) | How often the cache-management sweep re-runs after its startup pass. 1 to 31 536 000 (one year) *(since 0.9.1)*. `0` is not "off": it re-ran the sweep back to back, so it stops startup; to sweep rarely, set a large value |
| `ov_cache_layout` | `"keyed"` | `"keyed"`: text models (including draft models) and vision models on GPU/CPU compile into `<ov_cache_dir>/kv-<tag>/`, one directory per `kv_cache_precision` string passed to the plugin (`kv-auto` when none is passed, `kv-i8`, `kv-u4`, `kv-f16`); other kinds stay in the root. Needed because on OpenVINO 2026.4.x a blob does not record the KV precision, so one compiled under another precision is silently reused or fails at the first request. Each directory holds a full compiled copy of its models. `"flat"`: the 0.8.x layout - a rollback that also brings the precision mixing back, and reuses same-version blobs already in the root without recompiling. Vision models pass their resolved precision (see `kv_precision_pass`), or none (`kv-auto`) on CPU or with `kv_precision_pass: "plugin"`. A keyed directory that cannot be written to is not used (those models compile without a blob cache). Restart-only; any other value is refused at startup. Use one `ov_cache_dir` per instance, host and OpenVINO version when `ov_cache_max_gb > 0`. Windows: `icacls /T` or `(OI)(CI)` permissions on `ov_cache_dir` are inherited by the `kv-*` directories. Blobs are keyed by graph and device, not weight values: after replacing a model's weights in place, delete that model's blobs (true with either layout) |
| `kv_precision_pass` | `"explicit"` | `"explicit"`: RustedVINO resolves each model's KV-cache precision itself (the model's own `kv_cache_precision` > the global `kv_cache_precision` for text models > the model's own IR hint > a rule: u4 when the model has 4-bit weights, otherwise i8; vision models ignore the global key) and passes the result to the engine, so the context limit the gate computes is the precision the engine really uses. `"plugin"`: the 0.8.x behaviour - vision models pass nothing (plugin default) and text models pass the global `kv_cache_precision` as configured. Restart-only (a config reload reports it in `globals_not_applied`); any other value is refused at startup. The passed string also names the model's `kv-<tag>` blob-cache directory, so switching modes, or the first start of a version that resolves precisions, recompiles a vision model (and a text model under `kv_cache_precision: "auto"`) once into its new directory. CPU and NPU devices, HETERO/AUTO/MULTI device specs and models with a speculative-decoding draft always behave as `"plugin"`. With `"plugin"` the server no longer knows which precision a vision model or an `auto` text model gets, so those models are also sized in whole GiB and keep the legacy context limit (see `kv_gate_mode`) |
| `kv_gate_mode` | `"exact"` | `"exact"`: the context limit (`max_model_len`, the prompt-length gate) is computed from the model's real KV-cache layout and the exact pool the engine was built with (per-layer geometry, linear-attention state, precision, block rounding), as `min(95 % of the pool capacity, the model's native context, a ratchet learned for exactly this pool)`. `"legacy"`: the 0.8.x estimate (every layer at the configured precision times a 0.85 margin), which under-promises hybrid (linear-attention) models because it counts every layer, and over-promises models whose KV precision is f16 by hint (e.g. phi-3.5-vision). These always use the legacy estimate whatever this says: a model the server cannot describe with confidence (unknown layers, low-confidence profile), a CPU/HETERO/AUTO device, a model with a speculative-decoding draft, `kv_precision_pass: "plugin"` (vision models and `auto` text models), speculative decoding, an unverified OpenVINO/GenAI version pair, a model with no `vram_gb`, and a vision model on a shared-memory GPU, or on a discrete GPU where `kv_headroom_mode` is `"off"` or the device memory cannot be read (a vision model on a discrete GPU with the headroom cap active is exact: it takes the lower of the pool limit and the device-headroom limit). `legacy` also turns the device-headroom limit, the transient leases and the `max_tokens` clamp off. `legacy` is a rollback of the limit only: pools are still sized in whole KV blocks (`kv_cache_gb` is booked as built) and the vision template/tool admission still applies. Restart-only (a config reload reports it in `globals_not_applied`); any other value is refused at startup |
| `kv_gate_exact_on_unknown_engine` | `false` | Default `false`. When `true`, models are sized and gated with the exact layout model even on an OpenVINO/GenAI version pair the layouts were not verified on (2026.4.1 is the verified pair); a warning is logged once per pair. Use only to try a newer release before the server learns it. Restart-only. |
| `kv_headroom_mode` | `"dynamic"` | `"dynamic"`: a vision model on a discrete GPU (Linux) gets a second context limit besides the pool: the free device memory left after the weights, every pool and whatever other processes hold, divided by what one prompt token transiently costs (hidden size x 4 bytes x `kv_transient_factor`). It is recomputed after every load, unload and resize. Needed because a prompt that fits the KV pool can still exhaust the device memory (`CL_OUT_OF_RESOURCES`) and poison the GPU context for the whole process; measured on a 27B vision model with a 6 GiB pool at 164,960 tokens. `"off"`: no headroom limit; vision models on a discrete GPU then keep the legacy limit. Where the device memory cannot be read (non-Linux, no probe) the legacy limit applies; the server logs a one-time warning. Any other value is refused at startup. Restart-only. |
| `kv_transient_factor` | `2.0` | Multiplier on the per-token transient cost (hidden size x 4 bytes, the f32 `inputs_embeds` copy) used by the headroom limit; 2.0 covers the staging copy measured on the 27B (about 34 KB per token against 20 KB computed). Must be at least 1. Restart-only. |
| `kv_headroom_margin_gib` | `0.5` | Device memory (GiB) kept free beyond everything the server and other visible processes hold, for the compositor and processes of other users that the server cannot see. Must be >= 0. Restart-only. `0.0` = no margin |
| `kv_headroom_min_tokens` | `4096` | Smallest headroom limit the server will ever report for a vision model (capped at the model's native context). A load whose pool would push the limit below it is made smaller, and refused if it cannot be made small enough; a later load that would push another vision model below it is refused. The limit is never 0. Must be at least 1. Restart-only. |
| `kv_headroom_inflight_tokens` | `32768` | A vision request longer than this many tokens holds a transient reservation (tokens x hidden size x 4 x `kv_transient_factor` bytes) of device memory while it runs, so a load started meanwhile can be refused. Restart-only. `0` = every vision request holds a reservation |
| `kv_warmup` | `true` | `true`: after a load, a one-token generation runs so the KV pool is materialised and a broken GPU shows at load time rather than at the first request, and the memory the weights really took is measured. Adds a few seconds to a load. Only GPU devices are warmed; a draft-attached (speculative) engine is not. `false` keeps the arithmetic path (no warm-up). Restart-only. |
| `kv_pressure_monitor_enabled` | `false` | Enables the KV-cache pressure monitor: samples every `Ready` model's live pool occupancy and flags one that stays at/above `kv_pressure_threshold_pct` continuously for `kv_pressure_sustained_secs`. **Detect-and-flag only** — it never evicts, resizes or otherwise acts, and it is deliberately excluded from `/health` (pressure is a live-load signal, not a fault). **Enabling it requires setting both knobs below: the server refuses to boot otherwise** (see the note under this table) |
| `kv_pressure_threshold_pct` | `0.0` = unset | Occupancy percentage (0–100) a model must reach to count as under pressure. **No usable default is provided on purpose** — the right number differs by hardware, since a UMA box's KV pool contends with host RAM while a discrete-VRAM box's does not. `90.0` is a reasonable starting point but is *not* validated against real production traffic |
| `kv_pressure_sustained_secs` | `0` = unset | How long occupancy must stay continuously at/above the threshold before flagging — reacting to *sustained* pressure, not a momentary spike. Note the current rule resets the timer on a **single** sub-threshold sample, so pick this together with `kv_pressure_sweep_interval_secs` rather than in isolation |
| `kv_pressure_sweep_interval_secs` | `15` | How often the pressure monitor samples. Cheap (atomic in-memory reads, no I/O). At the default `15`, a `60`s sustain window is spanned by only 4–5 samples, and because one sub-threshold sample resets the timer, a brief dip can restart the clock — sample faster if you need reliable detection of a condition that fluctuates. 1 to 31 536 000 (one year) *(since 0.9.1)*; `0` (back-to-back sampling) stops startup, also with the monitor off. To turn the monitor off, set `kv_pressure_monitor_enabled: false` |
| `kv_resize_cooldown_secs` | `30` | Minimum gap between two `POST /v1/admin/models/{id}/resize` calls for the same model; a call inside the window gets `429`. **Applies unconditionally — independent of `kv_pressure_monitor_enabled`**, because the resize endpoint exists and needs a guardrail whether or not anything is monitoring. `0` = no cooldown |
| `load_timeout_secs` | `600` | How long a model load may run before the request gets `504 load_timeout`. `0` = no deadline; otherwise at least `60`. The load is not cancelled (OpenVINO cannot be): it is **booked as stalled** — its memory stays reserved, new loads on its device get `503 device_stalled` (models already loaded there keep serving), `rv status` / `GET /v1/admin/models` show it as `stalled`, and when the load finally returns it is committed (`ready`) or, on an error, discarded and its memory freed. A load whose client disconnects mid-load is booked the same way. Under `--supervise` a running load also counts for the generation watchdog once it is past this deadline: a load still not back after deadline + `watchdog_hang_ceiling_secs` (default 600 + 600 s) restarts the worker. With `0` the watchdog never counts the load. Per-model `load_timeout_secs` overrides this |
| `on_demand_wait_secs` | `0` | *(since 0.9.1)* How long a request to a `load: "on_demand"` model that is not loaded waits for its load. `0` = no wait: the request starts the load and gets `503` + `Retry-After` at once (the behaviour before 0.9.1). Above `0` the request holds until the model is ready and is then served normally, or until the load fails (it gets that load's error: `504 load_timeout`, `503 insufficient_memory`, ...; a failure without a code is a `500 server_error` "failed to load", details in the server log), or until the wait runs out (`503` + `Retry-After`, the load keeps running). A streaming chat or completion gets its `200` at once and SSE keep-alive comments every 15 s while it waits; an error after that comes as one SSE error frame, then `[DONE]`. Applies to every route that loads an on-demand model (chat, completions, embeddings, rerank, System One, speech, transcription, images); eager models never wait, and `/tokenize` and `/detokenize` never load or wait (a model that is not loaded is their plain `503`). 0 to 300; a value at or above a non-zero `load_timeout_secs` stops startup (with `load_timeout_secs: 0` only the 300 cap applies). Per request the wait also stays 2 s below that model's own `load_timeout_secs`, so a model with a shorter deadline gets a shorter wait. At most 64 requests wait at once, across all models; past that a request gets the immediate `503` + `Retry-After`. A waiting request holds its request body in memory (an upload included), so the waits can hold up to 64 × the request-size limit. A request to a model whose load is already past its `load_timeout_secs`, or whose device has a stalled load, does not wait: it gets the stall's error at once (`504 load_timeout`, `503 device_stalled`). A request to an on-demand model of a kind its route does not serve gets that route's wrong-kind answer before any load when the model's kind is known (from `kind`, its files at registration, or an earlier load; a `400`, on chat a `501` for the kinds it does not serve yet). A request that gives up (client disconnect) frees its place and does not cancel the load. About `120` suits chat UIs (Open WebUI) whose first message would otherwise fail while the model loads. Restart-only (a config reload reports it as not applied) |
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
| `vram_gb` | required | VRAM estimate (GiB); `0.0` skips VRAM gating for this model. On Linux, `GET /v1/admin/models` shows what a loaded model uses with its KV pool (`gpu_memory_estimate_gb`; `vram_measured_mib` is the weights alone). A pool that is not resident yet when the load ends is counted only when the load could be split (METRICS.md, "Measured device memory"); a draft-attached text model, and a text model whose pool the warm-up did not touch, still show figures low by that pool. The server logs a warning when the estimate is clearly more than `vram_gb` (above 5 % and above 0.75 GB over the charge; smaller overshoots are scratch space and are not reported) — use it to size this value. An embedding model's figure includes its batch cache |
| `kind` | `null` | `"text_gen"`, `"vision"`, `"embedding"`, `"stt"`, `"tts"`, `"image_gen"`, `"reranking"`, `"decision"` — auto-detected if absent, except `"decision"`, which must be set explicitly (a decision IR looks like a plain LLM on disk; see [Decision models](#decision-models-v1systemone)) |
| `pinned` | `false` | Never chosen as an eviction victim |
| `priority` | `0` | Soft eviction-order weight; higher = evicted later |
| `kv_cache_gb` | `null` | Per-model bounded KV pool target (GiB), overriding `default_kv_cache_gb` |
| `chat_template` | `null` | Path to a chat template that **overrides** the one in the model directory; relative paths resolve against the config file's directory. For converted models shipping a stale or incomplete template — e.g. LFM2's renders *nothing* for an assistant turn carrying `tool_calls`, so a tool-call turn replays as an empty assistant message and the model sees a result it never asked for. A corrected LFM2 template ships at `templates/lfm2.jinja`, in the repository and in the Linux release bundle (from a config at the bundle root: `"chat_template": "templates/lfm2.jinja"`). Text and vision models only; other kinds ignore the key. A configured override that cannot be read **refuses the load** *(since 0.9.1)* (`rv load` / the admin load answer `404` with the model and the resolved path; a preload logs it and adds a `preload_failed` startup warning, the server keeps running) — never a silent fall-back to another template. Without an override, the model's own template is used, and a model that ships none gets a generic template with a WARN (tool calling degraded). The file is read at each load (a fixed file reaches the model at its next load); a changed `chat_template` setting reaches a loaded model only after unload → reload-config → load |
| `max_concurrent_streams` | `null` | Per-model concurrency cap, overriding `max_num_seqs` for this model |
| `load` | `"eager"` | What a request for this model does while it isn't loaded. `"eager"`: fails with a plain `503` — the model loads only via `preload` or the admin API. `"on_demand"`: starts a background load and answers `503` + `Retry-After` until it's ready (same on every device, NPU included), or, with the global `on_demand_wait_secs` above `0` *(since 0.9.1)*, waits for the load and is then served. Every inference route starts the load; *(since 0.9.1)* also `/v1/completions`. `/tokenize` and `/detokenize` never do (plain `503` until the model is loaded). Neither value loads the model at startup; only `preload` does |
| `evictable` | `true` | When `false`, never chosen as an eviction victim (separate hard-exclude from `pinned`) |
| `device` | `null` | Explicit per-model OpenVINO device override (e.g. `"GPU.0"`). A `decision` model refuses any device naming the NPU |
| `tier_preference` | `[]` | Ordered capability-tier labels: `"heavy"`, `"strong-igpu"`, `"weak-igpu"`, `"npu"`, `"fallback"` |
| `reasoning_parser` | `null` | `"qwen3"`, `"gpt_oss"`, `"mistral"`, `"phi"`. If absent, it is auto-detected from the chat template at load, and the log says which (`reasoning_parser auto-detected from the chat template`): a harmony template (gpt-oss, recognised by `<|channel|>`) gets `"gpt_oss"`, checked first; a template that reads `enable_thinking` gets `"qwen3"`; anything else gets none. `"mistral"` and `"phi"` must be set explicitly |
| `max_prompt_len` | `null` | **NPU-only**: compile-time `MAX_PROMPT_LEN` ceiling for the static `LLMPipeline`; must be `≥ 1` when set |
| `min_response_len` | `null` | **NPU-only**: compile-time `MIN_RESPONSE_LEN`, the output room the NPU's fixed KV cache reserves on top of `max_prompt_len` (OpenVINO default 128). Prompt + answer can never exceed `max_prompt_len + min_response_len`; hitting it reports `finish_reason: "length"`. Raise it (e.g. `512`) for long answers to long prompts; must be `≥ 1` when set |
| `speculative` | `null` | Opt-in draft-model speculative decoding — see [Speculative decoding](#speculative-decoding-draft-model-assisted-generation) below |
| `precision` | `null` | Image-gen only: reported verbatim as `generation_metadata.precision` (e.g. `"int8"`); operator-supplied, not inferred |
| `model_source` | `null` | Image-gen only: reported verbatim as `generation_metadata.model_source` (e.g. a HF repo id) |
| `model_revision` | `null` | Image-gen only: reported verbatim as `generation_metadata.model_revision` (e.g. an HF commit hash) |
| `capabilities` | `[]` | Declared model capabilities, e.g. `"tool_calling"`. Validated against a known list — an unrecognized label is a hard registration error, not a silent no-op |
| `eviction_grace_secs` | `null` | Per-model override of the top-level `eviction_grace_secs` |
| `load_timeout_secs` | `null` | Per-model override of the top-level `load_timeout_secs` (same rules: `0` = no deadline, otherwise at least `60`) |
| `enable_prefix_caching` | `null` | Per-model override of the top-level `enable_prefix_caching`. `null` = use the top-level value; `true` / `false` force it on / off for this model. Takes effect on the model's next load. Changed in 0.9.0: hybrid linear-attention models (Qwen3.5/3.6) are no longer forced to prefix caching off on a GPU, and `default_placement: "auto"` no longer keeps them on the CPU; the GPU prefix-cache bug this guarded against is not present on OpenVINO GenAI 2026.4.1, the verified version pair (see `kv_gate_exact_on_unknown_engine`). On an older runtime set `false` for such a model, or give it `device: "CPU"` |
| `kv_cache_precision` | `null` | Per-model KV-cache precision: `"auto"` (or `""`), `"i8"`, `"u4"`, `"f16"` (`"u8"` = deprecated alias of `"i8"`); anything else is refused. `null` = the global `kv_cache_precision` decides for text models and the rule (u4 for 4-bit weights, otherwise i8, or the model's own IR hint) for vision models. `"auto"` (or `""`) is a choice too: it ignores the global key for this one model and lets the model's own IR hint or the rule decide. Any other value beats the global key and the rule. Applies from the next load, on a plain GPU device with `kv_precision_pass: "explicit"` only: on CPU, NPU, HETERO/AUTO devices, for models with a speculative-decoding draft and in plugin mode it is ignored and the load logs a warning. A different precision recompiles the model once into its own `kv-<tag>` blob directory. `PATCH /v1/admin/models/{id}` sets it (`null` clears it) |
| `sampling` | `null` | *(since 0.9.1)* Per-model sampling defaults, a nested object: `{"temperature", "top_p", "top_k", "repetition_penalty", "presence_penalty", "frequency_penalty"}` (each optional; an unknown key such as `temprature` is refused). `null` (or absent) = no operator layer: the model's own `generation_config.json` and the server default apply. Values a client sends always win. Applies to the next request (no reload of the model): `PATCH /v1/admin/models/{id}` replaces the whole object (`null` clears it), and `POST /v1/admin/config/reload` applies an edited `sampling` to a loaded model too. See [Sampling profile](#sampling-profile-sampling) below |
| `thinking_budget_tokens` | `null` | *(since 0.9.1)* The model's default thinking budget in tokens (an integer `≥ 0`): a request that sets no budget and no `reasoning_effort` of its own reasons for at most this many tokens. **Strict:** a request that reaches the budget while still thinking ends with `finish_reason: "length"` and normally **no answer**. `0` = thinking off by default; `null` = no budget. Recommended 2,048–4,096 only where an empty answer at the limit is acceptable. Applies to the next request (no reload of the model): `PATCH /v1/admin/models/{id}` sets it (`null` clears it), `POST /v1/admin/config/reload` applies an edited value to a loaded model too; `rv add` / `rv patch --thinking-budget-tokens N` (`rv patch --clear-thinking-budget-tokens`). See [Thinking budget](#thinking-budget-thinking_budget_tokens) below |

`"supervisor"` sub-fields (`--supervise` wrapper, Unix only): `backoff_base_secs` (5),
`backoff_max_secs` (60), `max_restarts` (5), `restart_window_secs` (600),
`health_timeout_secs` (90), `hang_kill_grace_secs` (10), `watchdog_poll_secs` (30) — how often
to poll `GET /v1/admin/health/watchdog` once the worker is healthy, `watchdog_hang_ceiling_secs`
(600) — slack past a vision generation's own allowance (`prompt_tokens / 200 + max_tokens / 5`
seconds) after which it is treated as hung and killed, so a 92k-token prompt with a 6k-token
answer gets about 28 minutes, plus the 10-minute ceiling, before that; a short chat with
`max_tokens` 512 about 2 + 10 minutes (a chat without `max_tokens` gets the 8,192 `max_tokens_cap`: about 27 + 10 minutes;
with `max_tokens_cap: 0` on a discrete GPU it gets the KV pool's room after the prompt, so a large pool can allow hours:
100,000 tokens of room is about 5.5 hours, which is not a hang; on a shared-memory GPU it gets prompt time only); `0` disables the
generation watchdog entirely. The allowance runs from the moment the engine thread starts the
generation (a request queued behind a long one is not counted while it waits), and a worker that
never picks up what is queued is caught from the last progress, so the worst case before a wedged
worker is restarted is the allowance plus the ceiling plus one poll (`watchdog_poll_secs`) and
`hang_kill_grace_secs`. After a worker left without its teardown (exit 75 or 76, killed by a
signal, or killed by the supervisor as hung or by the first-load deadline below), a GPU can be
left with its copy engine stuck (observed on an Arc Pro B60 with the xe driver) so the next
process's first GPU work never runs; the exit of the process holding that work is what makes the
driver reset it. Two settings handle it:
`gpu_probe_deadline_secs` (15) — before the next worker starts, `rustedvino --gpu-probe` (a tiny
compile + inference on every GPU, one `GPU_PROBE device=… ok=… ms=…` line each) must exit within
this many seconds; past it the probe is killed, the supervisor waits 10 s and probes again, up to
3 times, then starts the worker anyway. The probe's work does not depend on the model, so a long
healthy model load is never mistaken for a stuck card. **15 is provisional**: if a healthy
probe's logged `ms=` comes close to it on your machine, raise it; `0` disables the probe.
`gpu_first_load_deadline_secs` (180) — the backstop: the next worker's first GPU model load must
still finish within this many seconds, or within the model's own `load_timeout_secs` when that
is longer (a load with no deadline, `load_timeout_secs: 0`, gets `watchdog_hang_ceiling_secs`
when that is longer), or the supervisor kills it and respawns. It is
checked until that load returns, loaded or failed, also after `/health` answers. It is a
backstop for a stuck state the probe misses; 180 is above the measured loads of a 27B (48-78 s)
and of a 30B on an Arc Pro B60 (169 s). Armed at most once per episode (a worker killed by it is probed
but not given the deadline again); `0` disables it.

Limits *(since 0.9.1)*: every `supervisor` seconds field is at most 31 536 000 (one year), so the
supervisor's sums of them cannot overflow. `watchdog_poll_secs`, `max_restarts`, `health_timeout_secs` and
`restart_window_secs` must be at least 1: at `0` the supervisor crashed (a zero poll period), gave up before the first
worker ran, treated every new worker as hung at once, or never stopped a crash loop (every restart aged out of a
zero window). `backoff_base_secs` / `backoff_max_secs` `0` = respawn without delay; `hang_kill_grace_secs` `0` = the
1 s floor; `watchdog_hang_ceiling_secs`, `gpu_first_load_deadline_secs` and `gpu_probe_deadline_secs` `0` = off.

A kill by the supervisor sends `SIGTERM` first and `SIGKILL` only once the worker is past its own
stop: a worker that is waiting for a model load to finish (up to 600 s) or tearing its engines
down says so, and the supervisor waits that out, so a kill can take up to about 11 minutes.
`hang_kill_grace_secs` is the wait for a worker that says nothing (never below 1 s); a
generation-watchdog kill of a healthy worker waits at least 48 s (its drain and teardown) whatever
this is set to. A `SIGKILL` during a GPU load can leave the GPU hung until the driver resets it
(observed on an Arc Pro B60).

*(since 0.9.1)* After a worker crash (any respawn above; not a clean stop or a new start), the respawned worker
brings back the set of models the crashed worker had loaded, not the config `preload`: its
preload phase loads only the `preload` models that were loaded at the crash (a `preload` model
that was not, because it was unloaded, did not fit or failed, is skipped with an INFO line, and
the skipped ones are named in one `respawn_preload_skipped` startup warning, which stays until a
restart even after those models are loaded), and then, once
listening, it restores the `eager` models the crashed worker had loaded at runtime (`rv load`),
one at a time, with their config settings (the overrides of the earlier load are not kept). So a
model swapped in at runtime for a preload model comes back instead of that preload model. Models
whose KV pool is sized to the room left (text and vision models) load after the rest of that
set, so they cannot take the room the others held at the crash: a `preload` one loads last in
the preload phase, or — when the set also has a runtime model with a fixed footprint (embedding,
speech, rerank, image) it must leave room for — after listening, at the end of the restore. A
load of such a deferred model (`rv load`, an on-demand request) before the restore reaches it
takes the normal path, which may evict, and the restore then skips it. "Loaded at the crash"
includes a model a load evicted for room (also a KV resize's or a KV-wedge recovery's own unload)
when the worker crashed before that load was ready, or when that load failed and the worker then
left without its teardown (a poisoned GPU, a stalled call, a kill): the respawn does not come back
with less than the crashed worker had. A load that failed while the worker went on serving really
evicted its victims; they are not brought back. Until its restore finishes, such a model answers
503 as before. A restore never evicts: a
model that does not fit, or fails, is skipped with a WARN and a `restore_failed` startup warning
(cleared once the model is loaded); a model that no longer exists is skipped with a WARN. Not
restored: a model unloaded before the crash (except one unloaded while another model's load was
running, when the worker crashed during that load: it is taken for that load's victim and comes
back), an `on_demand` model (it loads on its next request),
a model whose restore did not finish before the next crash, a restored model loaded when the
worker crashes again within `restart_window_secs` of its restore, and a model already restored
once within `restart_window_secs`. These two limits do not apply to `preload` models: one loaded
at a crash loads again after every respawn (as every respawn reloaded `preload` before), so a
`preload` model whose load itself crashes the worker repeats until `max_restarts`.
Edge cases: a worker that crashes during its own first preload, or before any model was loaded,
is respawned as a fresh start (the full `preload`); one that crashes with nothing loaded because
every model was unloaded before the crash is respawned with nothing, **no `preload` either** —
load models with `rv load`, or restart the service (a new start always runs the full `preload`).
A worker respawned after a second crash targets what that worker had loaded, by the same rules
(if it crashed during its own preload, the models it had not loaded yet are carried over).

The watchdog exists only under `--supervise`, which is Unix-only: on Windows nothing cures a
request that hangs on its own. A model stuck busy can be evicted only with `force`. For every
engine except the continuous-batching text engine (NPU, VLM, STT, TTS, embeddings, image,
rerank, decision), a forced eviction cannot abort the running request: it waits up to 30 s for
the engine thread to finish it and, if the thread is still running, answers `503
eviction_incomplete` (retryable). The memory stays reserved until the thread exits, so no load
is admitted into memory that is not free.

## Linux integrated GPUs: memory held by the GPU driver

On a Linux machine whose GPU shares system memory (Intel Lunar Lake and other integrated GPUs, `xe`
or `i915` driver), the GPU driver keeps freed GPU buffers in a page pool (TTM) for reuse. That memory
is counted as **used**, is missing from `MemAvailable`, and belongs to no process: `free`, `top`
and the processes' RSS do not explain it. The pool grows with every model load and unload and can
reach a large share of RAM. On a 30 GB laptop we measured 8.8 GB once and about 13 GB another time,
after repeated loads in RustedVINO and another inference server; the kernel's limit there was
15.2 GiB, half of the RAM.

Effects: the server's live memory gate (`system_ram_budget_gb` unset) sees too little free memory
and refuses loads that would fit; other programs that size themselves from free memory do the same
(another server offloaded fewer layers to the GPU), and a low-memory killer may act early.

- **In RustedVINO:** set `system_ram_budget_gb` (above). The gate then credits the reclaimable GPU
  page cache, up to that budget.
- **See it:** with RustedVINO and other GPU programs stopped, "used" in `free -g` stays well above
  what the processes hold. The pool limit, in 4 KiB pages: `cat /sys/module/ttm/parameters/page_pool_size`.
  With root and debugfs mounted: `sudo cat /sys/kernel/debug/ttm/page_pool`.
- **Free it now (root):** first try `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` (on our
  laptop it took "used" from 12 to 4 GB once). If memory stays used, each read of the pool's shrink
  file frees one batch:
  `sudo sh -c 'for i in $(seq 10000); do cat /sys/kernel/debug/ttm/page_pool_shrink >/dev/null; done'`
  (in one measured case, 10.4 → 16.6 GB available, and 23.4 GB after the server was stopped too).
  The kernel also shrinks the pool on its own under memory pressure, but late.
- **Cap it permanently (not tested by us):** a module option such as
  `options ttm page_pool_size=524288` (2 GiB) in a file under `/etc/modprobe.d/`, then a reboot.
  Its effect on load times is unknown.

## Sampling profile (`sampling`)

*(since 0.9.1)* Each text model (CB, VLM, NPU) has one **sampling profile**: the defaults a request gets for the
values it does not send itself. It is built from three layers, highest first:

1. **config**: the model's `sampling` object in `config.json` (or set with the admin API);
2. **the model's file**: its `generation_config.json`, whose `temperature` / `top_p` / `top_k` count
   only when it says `"do_sample": true` (without it the model runs greedy, as before), and whose
   `repetition_penalty` counts on its own (also for a greedy model) when it is above `1.0` and at most
   `2.0`, replacing the server's `1.1` *(since 0.9.1)* (exactly `1.0` is the usual leftover and means unset; below `1.0` rewards repetition and is
   ignored with a warning; a bad penalty does not discard the sampling values, and the other way round);
3. **the server default**: greedy decoding with `repetition_penalty` `1.1`.

```json
"mistral-small-3.2": { "vram_gb": 14.0, "sampling": { "temperature": 0.15, "top_p": 0.95 } }
```

Validation of the values (a failing value refuses startup, `POST /v1/admin/models/add` / `PATCH`
with 400, and `POST /v1/admin/config/reload` with 422, like every other per-model setting; an unknown
key inside `sampling` is a JSON error, which `PATCH` answers with 422 and `models/add` with `400 invalid_body`
*(since 0.9.1; 422 before)*):

| Key | Accepted |
|---|---|
| `temperature` | `0` to `2`; `0` = greedy |
| `top_p` | above `0`, at most `1` |
| `top_k` | at least `1` (omit it for no top-k cut) |
| `repetition_penalty` | `1.0` to `2.0`; `1.0` turns the server's `1.1` off for this model |
| `presence_penalty`, `frequency_penalty` | `-2` to `2` |

- `top_p` / `top_k` together with `temperature: 0` are refused (greedy decoding uses neither).
- `top_p` / `top_k` **without** a `temperature` need the model's `generation_config.json` to say
  `"do_sample": true` **with** a valid `temperature`; otherwise a request that sends no temperature
  samples at the engine's default temperature 1.0. `POST /v1/admin/models/add` and a `PATCH` that sets
  `sampling` refuse it (400). Because the model's file can change on disk (a re-export) without
  `config.json` changing, startup, `reload-config` and each load only log a WARN naming the model:
  set `temperature`, or remove `top_p` / `top_k`.
- `presence_penalty` and `frequency_penalty` have no file layer. On an NPU model the configured values
  are not sent (the NPU pipeline does not apply them); a client's own values pass through as before.

**How a request is resolved** against the profile:

| The request sends | Result |
|---|---|
| a `temperature` above 0 | that temperature, its own `top_p` / `top_k`, the unset ones from the profile |
| `temperature: 0` | greedy; `top_p` / `top_k` dropped; the config `presence_penalty` / `frequency_penalty` are not applied |
| only `top_p` and/or `top_k` | exactly what it sent; the profile is not consulted |
| none of the three, with tools active | greedy (tool-call format matters more than variety), above the config layer; the config `presence_penalty` / `frequency_penalty` are not applied |
| none of the three, no tools | the profile |
| `repetition_penalty` | the client's value, else the profile's (config, file, `1.1`), on every request, greedy ones included |
| `presence_penalty` / `frequency_penalty` | the client's value, else the config value — except on NPU and on a request that is greedy because of the tools rule or its own `temperature: 0` |

**Values the client sends always win.** A client that sends `temperature: 0` on every request (many
benchmark tools do) gets greedy decoding and none of the profile's temperature, `top_p`, `top_k`,
`presence_penalty` or `frequency_penalty`; it still gets the profile's `repetition_penalty` unless it
sends its own. Leave `temperature` out to get the whole profile.

**Realtime follows the profile** *(since 0.9.1)*: a `/v1/realtime` voice turn has no sampling fields and no tools, so
it uses the profile as is (a model whose file has `do_sample: true` samples there with its own values).

The effective profile is shown with the source of every value (`config`, `generation_config.json`,
`engine default`, `server default`, `none`) in the `sampling` field of `GET /v1/admin/models` (also
before the model's first load) and in the server log at each load (`sampling profile` line, which also
names a `generation_config.json` that is only an exporter stub with no recommendation of its own). A
`PATCH` or reload that changes a loaded model's `sampling` logs the line again. Each request logs the
source it used at debug level (`sampling_source=client|config|model|tools_pin`, and
`repetition_penalty_src=client|config|generation_config.json|server default`).

From the command line: `rv add` / `rv patch` take one flag per key (`--temperature`, `--top-p`,
`--top-k`, `--repetition-penalty`, `--presence-penalty`, `--frequency-penalty`; `rv patch
--clear-sampling` sends `null`), and `rv check MODEL` / `rv status MODEL` print the profile; `rv patch`
keeps the values already configured that it is not told to change (it reads them from the admin
API, where they are shown at single precision, and the read and the write are two requests). See
README.md "`rv`".

## Thinking budget (`thinking_budget_tokens`)

*(since 0.9.1)* A thinking model (Qwen3-style `<think>` blocks, gpt-oss's analysis channel) can be
given a default limit on how many tokens it reasons before it must answer:

```json
"qwen3-8b": { "vram_gb": 6.0, "reasoning_parser": "qwen3", "thinking_budget_tokens": 4096 }
```

> **Warning: the budget is strict.** A request that thinks past the budget ends with no answer:
> the server stops it at the budget with `finish_reason: "length"`, an empty `content` and the
> reasoning so far in `reasoning_content`. Set 2,048–4,096 only where that is acceptable (an agent
> that retries, a chat UI that shows the reasoning). Leave it `null` for no limit.

Every shipped config writes it out as `"thinking_budget_tokens": null`.

Clients choose per request with `reasoning_effort` (`none`/`minimal` = off, `low` = 1,024, `medium` =
4,096, `high` = 16,384 tokens; `xhigh` and `max` = `high`), `thinking_budget_tokens` (alias
`thinking_token_budget`; `0` = off) or `enable_thinking`; see INTERFACE.md "Thinking control".
**Which one wins** — an explicit off always wins; a budget only limits thinking that is on; the
model's value is a default the request can raise or lower:

| Request budget | `enable_thinking` | `reasoning_effort` | Model `thinking_budget_tokens` | Thinking | Budget |
|---|---|---|---|---|---|
| `0` | any | any | any | off | none |
| any | `false` | any | any | off | none |
| any | any | `none`/`minimal` | any | off | none |
| `N > 0` | `true` or — | `low`/`medium`/`high` or — | any | on | `N` |
| — | `true` | `low`/`medium`/`high` | any | on | the effort's |
| — | `true` | — | `M > 0` | on | `M` |
| — | `true` | — | `null` or `0` | on | none |
| — | — | `none`/`minimal` | any | off | none |
| — | — | `low`/`medium`/`high` | any | on | the effort's |
| — | — | — | `M > 0` | the template's default | `M` |
| — | — | — | `0` | off | none |
| — | — | — | `null` | the template's default | none |

(`—` = not sent.) Then the budget is lowered to `max_tokens` minus the answer reserve when
larger: the reserve is 512 tokens, or half of `max_tokens` when that is less (`max_tokens: 400`
leaves a budget of at most 200). So with a model default set, a very short request gets at most
half of its `max_tokens` for thinking. A budget of `0` from any source turns thinking off; it never arms
a stop. "Off" only sets the template's switch: on a template that has no switch, the model may
still think, with no budget.

- **Kinds.** Text (continuous batching), vision and NPU models. The text engine counts each token
  exactly. Vision and NPU models deliver text in chunks with no token count, so they stop after
  the budget divided by the engine's measured tokens per chunk (vision 1.37, NPU 1.04; one
  measurement each). This is approximate until measured on more models: a model whose real ratio
  is x % off stops about x % early or late (a vision model at 1.0 tokens per chunk, about 27 %
  early), and the answer reserve is approximate there too.
- **gpt-oss.** Only the analysis channel is budgeted: the switch to `commentary` (a tool call)
  ends the count, so the budget never cuts a tool call.
- **Templates.** The thinking switch reaches the chat template as `enable_thinking`, and as
  `reasoning_effort` (gpt-oss): `"none"` when off, the client's word when on. A template without
  a thinking switch is unaffected; the budget still stops a `<think>` block the model emits on its
  own.
- **Not applied** on `/v1/completions` (a raw prompt) or on the realtime voice route (thinking is
  always off there).
- **Changing it** applies to the next request, a loaded model included: `PATCH
  /v1/admin/models/{id}` (`null` clears it), or edit `config.json` and `POST
  /v1/admin/config/reload` (listed under `applied_live`); `rv patch MODEL
  --thinking-budget-tokens N` / `--clear-thinking-budget-tokens`, `rv add ... --thinking-budget-tokens
  N`. `GET /v1/admin/models` shows it as `thinking_budget_tokens` (`null` when unset).

## Decision models (`/v1/systemone`)

A decision model answers typed questions by scoring the answer tokens after one prefill — see
[`INTERFACE.md`](INTERFACE.md#post-v1systemone). Register it with an explicit kind:

```json
"models": { "nimble": { "vram_gb": 9.5, "kind": "decision", "device": "GPU.1" } }
```

The model directory must contain:
- a **text-only** OpenVINO IR (`openvino_model.xml/.bin`) whose `logits` output is gathered to the
  last position — shape `[batch, 1, vocab]`, f32. A plain export gives `[batch, seq, vocab]`; it
  is refused at load, because 8K-token prompts would produce ~8 GB of logits per question;
- the OpenVINO tokenizer IRs (`openvino_tokenizer.xml/.bin`);
- a `decision.json` sidecar:

```json
{
  "prompt_format": "ollama",
  "system_prompt": "Classify the context using the supplied schema. The schema defines each field, …",
  "candidate_ids": [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57],
  "context_length": 8194,
  "licence": "Apache-2.0"
}
```

| Field | Meaning |
|---|---|
| `prompt_format` | Prompt builder; only `"ollama"` (Ollama v0.35's System One format) is supported |
| `system_prompt` | The model's system prompt (Ollama: the Modelfile `SYSTEM`) |
| `candidate_ids` | Token id of each answer letter, `A` first, at least 26 (the list above is Qwen3.5's tokenizer). Checked at load: each letter must encode to exactly its id and append exactly one token to a rendered prompt |
| `context_length` | Context the length rule is checked against (Ollama's `num_ctx`); `8194` = 8192 prompt tokens + 2 scoring positions |
| `licence` | Reported in the registry; optional |

Dynamic activation quantisation is always off for decision models (it shifts probabilities), and
any NPU device is refused (the NPU's LLM mode corrupts this model family's recurrent state).
Like any model, a decision model loads at boot only if it is in `preload`; with `load:
"on_demand"` the first request starts the load and gets `503` + `Retry-After` (or waits for it,
with `on_demand_wait_secs` above `0`).

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
| `draft_model` | `String`, required | Directory name under `models_dir`, a plain folder name *(since 0.9.1)* (no `/`, `\`, `..` or drive letter such as `C:`; refused at startup, `models/add` and PATCH, like a model id) — need NOT be its own `models` entry (and normally shouldn't be: it's never independently loadable/evictable this way) |
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
