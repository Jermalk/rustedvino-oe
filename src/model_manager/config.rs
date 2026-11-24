// ============================================================
// src/model_manager/config.rs — server configuration
// ============================================================
// Loaded from a JSON file at startup. serde_json does the
// heavy lifting; this module adds validation and a `load()`
// helper so main.rs is one line.
//
// CRASH COURSE — #[serde(default)]:
//   Fields tagged with this have `Default::default()` as their
//   fallback when the JSON key is absent. For Vec that is [],
//   for HashMap that is {}. Required fields (no default) cause
//   a deserialisation error when missing.
// ============================================================

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Server-wide configuration loaded from `/opt/rustedvino/config.json`.
///
/// **All known models must appear in `models`** — even those with an
/// unknown footprint (set `vram_gb: 0.0` to skip VRAM gating for that model).
/// Model IDs not present in `models` return 404 from every endpoint.
// Independent config toggles for unrelated features (open-bind auth, per-
// feature enable flags) — a state machine/enum would just be these same
// bools with extra ceremony, not a real reduction in what a reader has to
// track. `kv_pressure_monitor_enabled` pushed this over the lint's default
// threshold; not worth restructuring the whole config type for one flag.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    /// Directory containing model subdirectories.
    ///
    /// Each subdirectory is named by its model ID and must contain the
    /// `OpenVINO` IR files (`openvino_model.xml`, `openvino_model.bin`).
    pub models_dir: PathBuf,

    /// `OpenVINO` device string for inference.
    ///
    /// Common values: `"GPU.1"` (B60/B50 dGPU), `"CPU"`.
    pub device: String,

    /// Model IDs to load automatically at startup.
    ///
    /// Every entry must be a key in `vram_gb`. Loading fails at startup if
    /// a preload model is absent from the models directory.
    #[serde(default)]
    pub preload: Vec<String>,

    /// Per-model configuration — VRAM, kind, and policy in one stanza.
    ///
    /// This map defines the set of **known** model IDs. Any model ID not
    /// listed here is unknown — `load_model` and `get_handle` return
    /// `ModelError::NotFound`. Use `vram_gb: 0.0` inside the stanza to register
    /// a model without VRAM gating (useful on an ungated domain: the CPU, an integrated GPU, or a
    /// discrete GPU opted out with `domain_budgets`).
    ///
    /// Each entry replaces what used to be three separate maps (`vram_gb`,
    /// `model_kinds`, `model_policies`): all per-model properties live here.
    #[serde(default)]
    pub models: HashMap<String, ModelEntry>,

    /// Total usable VRAM on the inference device, in GiB (capped at what the device reports).
    ///
    /// `0.0`: a discrete-GPU inference device is budgeted at the memory it reports (like every
    /// other discrete GPU); any other inference device (CPU, integrated GPU, NPU, a composite
    /// `HETERO:`/`MULTI:`/`AUTO:` spec, a dGPU reporting no figure) is not VRAM-gated. With a
    /// budget, `load_model` evicts idle LRU models when a new one would not fit, and refuses a
    /// model that can never fit. `domain_budgets {"<device>": 0.0}` switches gating off on purpose.
    pub total_vram_gb: f64,

    /// Maximum concurrent sequences in the CB scheduler (`SchedulerConfig::max_num_seqs`).
    ///
    /// This is BOTH the KV-pool safety cap and the admission gate: requests that
    /// arrive when this many are already active receive HTTP 429 immediately —
    /// no GPU work started, no stall. `0` keeps the `OpenVINO` default (256),
    /// which is rarely appropriate.
    ///
    /// IMPORTANT — set this ABOVE your expected concurrency, not equal to it.
    /// Pipelined clients (and benchmark tools) transiently keep ~1.5–2× their
    /// nominal concurrency in flight during request refire bursts, so a cap set
    /// equal to the offered load storms spurious 429s. Rule of thumb:
    /// `max_num_seqs ≳ 2 × expected concurrent clients`, bounded by what
    /// `cache_size_gb` can hold at the target context length. See
    /// the project's internal engineering log.
    #[serde(default = "default_max_num_seqs")]
    pub max_num_seqs: usize,

    /// Maximum KV cache pool size in GB (`SchedulerConfig::cache_size`).
    ///
    /// Acts as a **cap** on the dynamically computed KV pool: when non-zero,
    /// the actual pool passed to `ov_cb_create` is `min(computed, cap)`.
    /// `0.0` = uncapped — the model gets all remaining VRAM after its weights
    /// and the safety margin are accounted for.
    ///
    /// The cap also limits an explicit per-model or per-load `kv_cache_gb`: asking
    /// for 4 with a cap of 2 yields 2, and `rv add` / `rv resize` and
    /// `GET /v1/admin/models` (`kv_requested_gb`, `clamped_by`) say so. Sizes are
    /// GiB; decimals are accepted. A pool the exact model can describe (a model on a
    /// plain GPU, known layout; either `kv_gate_mode`) is sized in whole KV blocks and booked
    /// as built; any other pool is allocated in whole GiB, so it is rounded up to the next
    /// whole GiB (rounded down when the rounded pool would not fit the free memory) and that
    /// is what is booked. A model with no known
    /// footprint (`vram_gb: 0`) is not gated: its explicit `kv_cache_gb` is used as given
    /// (rounded up, not capped).
    ///
    /// Existing configs setting `cache_size_gb: 6.0` continue to work
    /// identically to the old fixed-allocation behaviour on a single loaded model.
    #[serde(default)]
    pub cache_size_gb: f64,

    /// Default bounded KV-cache pool target in GB for co-residency (Slice 2).
    ///
    /// The **global** default for every model's KV pool size. `0.0` (the
    /// default) means *unbounded* — a model grabs all VRAM left after its
    /// weights and the safety margin (today's single-model behaviour, so an
    /// absent key is byte-unchanged). When non-zero, `compute_kv_pool_gb`
    /// clamps each model's pool to this target so multiple models can co-reside
    /// instead of the first one claiming the whole card.
    ///
    /// Precedence (lowest to highest): this global default →
    /// [`ModelPolicy::kv_cache_gb`] per-model override → a load-time API
    /// override (Slice 2b). The existing [`cache_size_gb`](Self::cache_size_gb)
    /// cap still applies on top of the resolved target.
    #[serde(default)]
    pub default_kv_cache_gb: f64,

    /// VRAM reserved for driver and runtime overhead, in GB.
    ///
    /// The KV pool computation is: `free_after_weights - vram_safety_margin_gb`,
    /// bounded below by `min_kv_cache_gb`. A 1 GB margin is sufficient for
    /// the `OpenVINO` `OpenCL` runtime on Battlemage (observed headroom from
    /// Session-26 OOM analysis). Increase if you see `CL_OUT_OF_RESOURCES`
    /// errors during multi-model loads.
    #[serde(default = "default_vram_safety_margin")]
    pub vram_safety_margin_gb: f64,

    /// Multiplier applied to a model's `vram_gb` when it is placed on the NPU,
    /// for the memory accounting only (the figure shown for the model is
    /// unchanged). An NPU model's real footprint is well above its weight
    /// size: Windows pass 2 (D8) measured 7.6-8.4 GB of NPU adapter memory
    /// for an 8B int4 model reserved at 5.5 GB, which let the GPU model next to
    /// it take a KV pool the shared budget could not back (and the GPU context
    /// then died, D2). On a shared-memory box that memory comes out of the
    /// same budget as the iGPU's. Default 1.5; `1.0` charges the configured
    /// figure as-is. Must be at least 1.0.
    #[serde(default = "default_npu_footprint_factor")]
    pub npu_footprint_factor: f64,

    /// Largest image request (total pixels: `n x width x height`) an image-generation model
    /// on the CPU serves, and the size its memory charge is sized for (SF-5 D-06). On the CPU a
    /// diffusion pipeline keeps a working set that depends on its family and the image size,
    /// not on the weights (SD 3.5 medium: +21 GiB at 1024x1024 beside 8.3 GB of weights), so
    /// admission charges `vram_gb` plus that working set at this size
    /// ([`super::image_footprint`]), and a larger request is a 400 `size_exceeds_cpu_limit`.
    /// Default 1 048 576 (one 1024x1024 image, the largest measured size; above it the charge
    /// extrapolates). At least 65 536 (256x256). Restart-only.
    #[serde(default = "default_cpu_image_gen_max_pixels")]
    pub cpu_image_gen_max_pixels: u64,

    /// Longest audio upload `/v1/audio/transcriptions` and `/translations` decode, in seconds of
    /// audio (0.9.1 S4, outside review I-01). The request-body limit bounds the *compressed*
    /// file only; a highly compressible one (FLAC of silence, a very low bitrate) decodes to
    /// orders of magnitude more, and one such request could exhaust memory and kill the
    /// process. Decoding stops with a `400 invalid_audio` once the audio passes this length
    /// (refused before decoding when the container declares its length). Default 7200 (2 h).
    /// Worst-case decode memory is this x 48 000 (the highest accepted source rate) x 4 bytes,
    /// plus this x 16 000 x 4 bytes for the 16 kHz copy: ~1.4 GB + ~0.5 GB at the default.
    /// At least 1. Restart-only.
    #[serde(default = "default_max_audio_duration_secs")]
    pub max_audio_duration_secs: u64,

    /// Most images one request may carry: `image_url` parts of the last chat message to a
    /// vision model, or the image and mask of `/v1/images/edits` (0.9.1 S5, outside review
    /// I-02). Over it is a `400 too_many_images`. Default 16. At least 1. Restart-only.
    #[serde(default = "default_max_images_per_request")]
    pub max_images_per_request: usize,

    /// Most pixels (`width x height`, summed over the request's images) one request may decode
    /// (0.9.1 S5). A tiny PNG can declare 16k x 16k; each image's size is read from its header
    /// and the request refused (`400 image_too_large`) before any pixel is allocated. Decoded
    /// RGB costs 3 bytes per pixel: the default 67 108 864 (64 MP — one 50 MP photo, or 16
    /// screenshots at 2560x1600) is ~192 MB, ~512 MB transient for a 16-bit PNG. At least
    /// 65 536. Restart-only.
    #[serde(default = "default_max_image_pixels_per_request")]
    pub max_image_pixels_per_request: u64,

    /// Most pixels (`width x height`) one image may have when it reaches a vision model (0.9.1
    /// C28). A decoded chat image over it is downscaled — both sides by the same factor, aspect
    /// kept — to at most this many pixels; one at or under it is passed through untouched. The
    /// vision encoder's memory grows with the image (qwen3.5-9B on an Arc 140V iGPU: 1 MP
    /// ~0.65 GiB, 2 MP ~1.15 GiB, 4 MP ~3 GiB, 8 MP out of GPU memory), so one large photo
    /// could take a UMA box down. Default 4 000 000 (4 MP). At least 65 536. Per image;
    /// [`Self::max_image_pixels_per_request`] still counts the original (header) pixels.
    /// Image edits (`/v1/images/edits`) are not affected. Restart-only.
    #[serde(default = "default_max_image_pixels")]
    pub max_image_pixels: u64,

    /// Minimum KV cache pool size in GB.
    ///
    /// If the computed KV pool (free VRAM minus safety margin) would be smaller
    /// than this value, the load is refused rather than starting an engine with
    /// a dangerously small pool. Acts as the floor for `ensure_vram_for` eviction
    /// decisions: we evict LRU models until at least this much KV headroom is
    /// available on top of the incoming model's weights.
    #[serde(default = "default_min_kv_cache")]
    pub min_kv_cache_gb: f64,

    /// OS RAM (GB) held back from the shared "system" memory-domain budget
    /// (iGPU + CPU + NPU on a UMA box). The system budget is
    /// `total_system_RAM − this`. `null`/absent uses the per-OS default
    /// (`os_memory::default_reservation_gb` — Linux 4 GB, Windows 6 GB). Only
    /// relevant once a model loads on a system-domain device (Phase C); on a
    /// single-dGPU box the system domain is inert.
    #[serde(default)]
    pub system_ram_reservation_gb: Option<f64>,

    /// System-RAM budget (GB) the server may treat as its own on a **UMA** box —
    /// the unified-memory twin of `total_vram_gb`, and the opt-in that switches
    /// the live M1 admission gate from "guess from `MemAvailable`" to
    /// "measure honestly against a declared budget".
    ///
    /// `null`/absent (the default) keeps the historical behaviour **exactly**:
    /// the gate reads `MemAvailable` alone. Set it (> 0) and the gate instead
    /// uses `min(MemAvailable + reclaimable GPU page cache, this budget)`.
    ///
    /// Why the GPU-cache term matters, and why this is opt-in rather than
    /// always-on: `MemAvailable` **omits** the DRM/TTM page pool — freed GPU
    /// pages the kernel keeps for reuse, counted as `used`. On one measured
    /// UMA laptop, that pool held ~8.8 GB and the gate
    /// refused a 12 GB model on a 30 GB machine claiming "this load can never
    /// fit"; `drop_caches` moved `MemAvailable` 14.76 → 23.39 GB and the same
    /// load then succeeded untouched. The pool grows with every load/evict
    /// cycle, so the defect worsens with use. Full write-up:
    /// the project's internal engineering log.
    ///
    /// Crediting reclaimable cache is the *less* conservative direction, so it
    /// stays behind an explicit operator declaration — a box that never sets
    /// this cannot regress. Discrete-VRAM boxes are unaffected regardless:
    /// their models resolve to a private-VRAM domain, never the system domain,
    /// so this gate never runs there. `system_ram_reservation_gb` still applies
    /// on top as the OS floor.
    #[serde(default)]
    pub system_ram_budget_gb: Option<f64>,

    /// How a model with no `device` / `tier_preference` of its own is placed: `"auto"` (the
    /// capability-aware engine - iGPU / NPU offload for light models, dGPU-first for heavy ones) or
    /// `"device"` (the global `device`, nothing else). The release bundle ships `"device"` with
    /// `device: "CPU"`, so a newcomer's models load on the CPU unless they ask otherwise; every fleet
    /// config states `"auto"`. An explicit per-model `device` or `tier_preference` always wins.
    #[serde(default)]
    pub default_placement: DefaultPlacement,

    /// Phase C2: the size threshold (GB) below which an LLM/VLM is treated as
    /// *light* work by the auto placement engine and gets the offload-first tier
    /// order (`weak-igpu` before `heavy`), parking a thin model off the dGPU. At
    /// or below this it is light; above it, dGPU-first. Compared against the
    /// model's `vram_gb` estimate (the int4 weight-size proxy). Default 2.0 GB —
    /// roughly a sub-2B int4 model, measured interactive (>20 tok/s) on a weak
    /// iGPU. Only consulted when the engine picks the *default* preference;
    /// an explicit `device` or `tier_preference` sidesteps it.
    #[serde(default = "default_light_model_max_gb")]
    pub light_model_max_gb: f64,

    /// Phase C2 (STT refinement): the size threshold (GB) below which a Whisper
    /// STT model is treated as *light* work and keeps the offload-first tier
    /// order (`weak-igpu` before `heavy`). At or below this it is light; above
    /// it, the model is *heavy* (encoder-bound, like `whisper-large`) and a
    /// **weak** iGPU is demoted below the CPU fallback — a live probe
    /// measured the weak iGPU as the *slowest* device for `whisper-large`
    /// (6.87 s vs CPU 4.98 s vs dGPU 0.39 s on 8 s audio), so offload-first is
    /// pessimal on latency there. A *strong* iGPU keeps offload-first regardless.
    /// Compared against the model's `vram_gb` estimate. Default 1.0 GB —
    /// separates `whisper-base`/`small`/`medium` (light) from `whisper-large`
    /// (~1.5 GB, heavy); deliberately distinct from `light_model_max_gb` (2.0),
    /// which would misclassify the 1.5 GB `whisper-large` as light. Only consulted
    /// when the engine picks the *default* preference for an STT model; an explicit
    /// `device` or `tier_preference` sidesteps it.
    #[serde(default = "default_light_stt_max_gb")]
    pub light_stt_max_gb: f64,

    /// A model whose `vram_gb` estimate is at least this fraction of
    /// `total_vram_gb` is assumed unable to safely coexist with anything else
    /// on the dGPU — the size-aware default tier preference excludes the dGPU
    /// (`heavy`) tier entirely for such a model (CPU/iGPU only), instead of
    /// relying on per-registration VRAM budget contention (which depends on
    /// registration order and what else happens to be pinned) to route it
    /// away after the fact. Only consulted when the engine picks the
    /// *default* preference; an explicit `device` or `tier_preference`
    /// sidesteps it, same as the other size thresholds above. Default `0.8`
    /// (80%) — set to `>= 1.0` to disable.
    #[serde(default = "default_dgpu_size_ceiling_fraction")]
    pub dgpu_size_ceiling_fraction: f64,

    /// Explicit per-memory-domain budget overrides (GB), keyed by `domain_id`
    /// (a discrete GPU's device name, or `"system"`). An entry here wins over
    /// every derived budget — `total_vram_gb` for the inference domain, the
    /// device's reported memory for a secondary GPU, or `RAM − reservation` for
    /// the system domain. `0.0` disables gating for that domain. Default empty.
    #[serde(default)]
    pub domain_budgets: HashMap<String, f64>,

    /// How long, in seconds, a model stays hard-protected from eviction after
    /// any channel — admin load, plain API use, or a realtime turn — last used
    /// it. Self-expiring, checked against `last_used` on top of the always-on
    /// `pinned`/`evictable` hard excludes. Uniform across every channel (no
    /// special-cased longer window for admin loads); `force` on a load bypasses
    /// it. Default 2s (was 300s until Windows pass 2, D4). A model with a
    /// running request is NOT protected by this window: it sorts last in the
    /// victim order, and evicting it for another model's load drains it for up
    /// to 15 s and then aborts what is left (`rv check` lists such victims as
    /// busy). `0.0` disables the window globally. A per-model
    /// [`ModelPolicy::eviction_grace_secs`] override takes precedence when set.
    #[serde(default = "default_eviction_grace_secs")]
    pub eviction_grace_secs: f64,

    /// Realtime voice arbitration v2 (D5): the operator-set, deterministic
    /// model set a realtime session gets when it doesn't ask for something
    /// different. Absent (the default) ⇒ today's timing-dependent
    /// voice-pin/discovery behavior, unchanged.
    #[serde(default)]
    pub realtime_defaults: Option<RealtimeDefaults>,

    /// Realtime voice arbitration v2 (D4): the "viable minimum" capability
    /// floor a model must clear to serve the realtime LLM slot. Absent (the
    /// default) ⇒ "any Ready `text_gen`/`vision` model" — today's de-facto
    /// behavior.
    #[serde(default)]
    pub realtime_viable_minimum: Option<RealtimeViableMinimum>,

    /// Cross-pipeline device admission ceiling
    /// (the project's internal engineering log step 2): explicit per-device concurrency caps,
    /// keyed by the live `OpenVINO` device string (e.g. `"GPU.1"`). Sits IN
    /// FRONT OF each engine's own per-engine admission gate — a second,
    /// device-wide layer that also counts different engine kinds sharing one
    /// physical device (an LLM + VLM + STT + TTS + embedder all resident on
    /// one GPU). A device with **no** entry here is pure passthrough: no
    /// `Semaphore` is created for it and `DeviceBudgets::admit` never waits or
    /// holds anything for it. Default empty (disabled) — real numbers are set
    /// only after empirical
    /// per-device testing lands (Migration order #4). A configured cap must be
    /// at least `1` (validated below) — omit the entry to disable gating for a
    /// device rather than setting `0`.
    #[serde(default)]
    pub device_budgets: HashMap<String, u32>,

    /// KV-cache element precision for the inference device (`KV_CACHE_PRECISION`).
    ///
    /// Accepted: `"auto"` / `""` (plugin default), `"i8"`, `"u4"`, `"f16"`;
    /// `"u8"` (the default, kept so every shipped config stays valid) is a
    /// deprecated alias stored as `"i8"`. Any other value (including the old
    /// `"f32"`) fails config validation at startup — see
    /// [`normalize_kv_precision`].
    ///
    /// Measured on `GenAI` 2026.4.1 (GPU): the plugin default (`auto`) is `u4`
    /// for models with int4 weights and `i8` otherwise (`f16` only via an IR
    /// hint), so an explicit `i8` (and the shipped `"u8"`) is about 1.8× larger
    /// per token than the `u4` default for such models (Qwen3.8-27B: 37,134 vs
    /// 20,748 B/token measured) and `f16` about 3×. The global key applies to the
    /// CB (text) engine only; VLM engines resolve their own (see `kv_precision_pass`). The
    /// capacity gate prices `auto` as f16 and `u4` like `i8`, so it is
    /// conservative there and for hybrids (it counts every layer); for a dense
    /// model on `i8` it prices one byte per element and ignores the ~13 % scale
    /// overhead, so it is about 14 % optimistic (unchanged by this release).
    ///
    /// NOTE: Passed as a `compile_model` property to the CB pipeline at load
    /// time. Read once at startup: a config reload never applies it (it is
    /// reported in `globals_not_applied`); a restart does.
    #[serde(
        default = "default_kv_cache_precision",
        deserialize_with = "de_kv_cache_precision"
    )]
    pub kv_cache_precision: String,

    /// Reuse KV blocks across CB requests that share an identical prompt
    /// prefix (repeated system prompts, multi-turn history resent by a
    /// stateless OpenAI-style client).
    ///
    /// Defaults to `true` — an inference *server* is the strongest case for
    /// this, not the weak one: every multi-turn exchange resends full
    /// history, and system prompts recur across independent clients. The
    /// underlying `OpenVINO` `GenAI` library defaults this to `false` for a
    /// directly-constructed `ContinuousBatchingPipeline` (the field's
    /// documented default-on behavior only applies when CB is invoked via
    /// `LLMPipeline`, a different code path this project doesn't use) —
    /// measured on this project to give zero prefill speedup with it off.
    /// Only affects prefill/TTFT, never decode tok/s.
    ///
    /// Retained blocks count against `cache_size_gb`'s pool, not on top of
    /// it — safe to leave on for any config with `cache_size_gb > 0.0`.
    /// Do **not** combine with `cache_size_gb == 0.0` (dynamic allocation):
    /// retained blocks then grow unbounded and untracked by the VRAM budget.
    /// Every fleet config as of 2026-07 sets a fixed `cache_size_gb`, so this
    /// is a safe default; audit before adding a live config with dynamic
    /// allocation.
    #[serde(default = "default_enable_prefix_caching")]
    pub enable_prefix_caching: bool,

    /// CORS allowed origins for the HTTP API.
    ///
    /// Each entry is an origin (e.g. `"https://app.example.com"`) or the
    /// wildcard `"*"`. When any entry is `"*"` the server allows **any**
    /// origin (`Access-Control-Allow-Origin: *`); otherwise only the listed
    /// origins are permitted. Defaults to `["*"]` — permissive, for the
    /// public alpha. Tighten to an explicit allowlist for production.
    #[serde(default = "default_cors_origins")]
    pub cors_allowed_origins: Vec<String>,

    /// **Deprecated as of the keys-file split (the project's internal engineering log) — migration
    /// tripwire only, never read for auth.** Real inference-scope Bearer keys
    /// now live in a separate, `.gitignore`d file (see [`keys_file`](Self::keys_file)),
    /// never inline here. `config.json` is committed to git and routinely
    /// shared/diffed; a real key set in this field would land in git history.
    ///
    /// A non-empty value here is a **hard startup error** (`bootstrap` bails
    /// with a message naming the resolved keys-file path to move it to) —
    /// silently accepting-but-ignoring it would let an operator believe a key
    /// is still active when it no longer authenticates anything (the same
    /// silent-non-revocation failure the keys-file's own privilege-downgrade
    /// guard exists to prevent). Kept as a field, not deleted, purely so that
    /// error can name what it found instead of a generic "unknown key".
    #[serde(default)]
    pub api_keys: Vec<String>,

    /// **Deprecated — see [`api_keys`](Self::api_keys)'s doc comment; the same
    /// tripwire applies.** Real admin-scope keys live in the same keys file as
    /// inference keys (both scopes are reloaded together — see
    /// `ModelManager::reload_keys_file`).
    #[serde(default)]
    pub admin_api_keys: Vec<String>,

    /// Path to the keys file holding the real `api_keys`/`admin_api_keys`
    /// Bearer lists, deserialized directly as [`crate::AuthConfig`].
    ///
    /// `None` (the default) resolves to a path derived from the config file's
    /// own location: strip a trailing `.config.json` (or bare `.json` if the
    /// file wasn't named `*.config.json`) and append `.keys.json` — e.g.
    /// `scripts/myserver.config.json` → `scripts/myserver.keys.json`. A relative
    /// value here resolves against the config file's own directory, never the
    /// process's current working directory — per-box config/keys pairs must
    /// travel together regardless of where the server is launched from. See
    /// [`resolve_keys_file_path`].
    ///
    /// The file itself is never git-tracked (`.gitignore`: `*.keys.json`) and
    /// should be `0600` (checked on every load — a load-bearing warning, not
    /// yet a hard failure; see `startup::load_keys_file`).
    #[serde(default)]
    pub keys_file: Option<PathBuf>,

    /// How long (milliseconds) a request waits for an inference slot before
    /// the server responds `HTTP 429`.
    ///
    /// When all `max_num_seqs` slots are busy, incoming requests join a
    /// short wait-queue instead of being rejected immediately. This absorbs
    /// the refire bursts that pipelined clients (benchmark tools, Go
    /// keep-alive pools) generate while the previous batch completes.
    ///
    /// Setting this too high increases tail latency for clients that are
    /// genuinely over-capacity. `5000` ms (5 s) is a safe default: long
    /// enough to absorb a ~15 ms/token×16-slot batch, short enough that
    /// a truly overloaded server still fails fast relative to an HTTP
    /// client's typical 30 s read timeout.
    #[serde(default = "default_admission_queue_timeout_ms")]
    pub admission_queue_timeout_ms: u64,

    /// How long (milliseconds) a request waits for a [`Self::device_budgets`]
    /// token before returning `HTTP 429` — the cross-pipeline analogue of
    /// [`Self::admission_queue_timeout_ms`]. A **separate** knob because the
    /// device-wide gate and each engine's own gate are two independent
    /// layers a single call waits on in sequence (device gate first, then the
    /// engine's own gate) — sharing one timeout would conflate two different
    /// budgets. Same `5000` ms default/rationale as the per-engine knob.
    /// Irrelevant while `device_budgets` is empty.
    #[serde(default = "default_device_admission_queue_timeout_ms")]
    pub device_admission_queue_timeout_ms: u64,

    /// `OpenVINO` device for embedding models. `null`/absent = auto-select.
    ///
    /// When set (e.g. `"GPU.0"`, `"CPU"`), embedding models load on this device
    /// regardless of `device`. When unset, the server auto-picks: the iGPU
    /// (`GPU.0`) if available and distinct from the inference `device` → else the
    /// inference `device` → else `"CPU"`. The intent is to offload embeddings to
    /// the integrated GPU so the inference dGPU stays free for LLM/VLM work.
    #[serde(default)]
    pub embedding_device: Option<String>,

    /// Default embedding model applied to every new realtime session.
    ///
    /// When set, `realtime_session` seeds each new session's `embed_model`
    /// from this value so history retrieval is active from the first turn —
    /// without waiting for the client to send a `Config` event. The client
    /// can still override it per-session via a `Config` WS event.
    /// `null`/absent = no default (clients that do not send an embed model
    /// get no retrieval, matching the pre-feature behaviour).
    #[serde(default)]
    pub default_embed_model: Option<String>,

    /// Pooling strategy for embedding models: `"mean"` (default), `"cls"`, or
    /// `"last_token"`. MEAN matches the stormVINO reference and the
    /// sentence-transformers `multilingual-e5` family. Unknown values fall back
    /// to MEAN.
    #[serde(default = "default_embedding_pooling")]
    pub embedding_pooling: String,

    /// Whether to L2-normalize embedding vectors (default `true`). Normalized
    /// embeddings make cosine similarity equal to a dot product — the standard
    /// for retrieval/RAG and what `OpenAI`'s embeddings return.
    #[serde(default = "default_embedding_normalize")]
    pub embedding_normalize: bool,

    /// Maximum number of prompts accepted in one legacy `/v1/completions`
    /// array request (the `prompt: [...]` form). Default **16**.
    ///
    /// Each array element occupies an engine slot for its entire generation,
    /// so an unbounded array lets a single connection saturate every slot
    /// (committee finding 3). Requests with more elements are rejected with
    /// HTTP 400 before any engine work starts.
    #[serde(default = "default_max_prompt_array")]
    pub max_prompt_array: usize,

    /// Maximum number of inputs accepted in one `/v1/embeddings` request.
    /// Default **256** (`OpenAI` allows 2048).
    ///
    /// Separate from [`max_prompt_array`](Self::max_prompt_array): an
    /// embeddings array is one short batched pipeline call, not one full
    /// generation per element. The whole array still runs as a single batch,
    /// so memory and latency grow with inputs × input length — lower it on a
    /// small shared-memory iGPU. Must be `>= 1`.
    #[serde(default = "default_max_embedding_inputs")]
    pub max_embedding_inputs: usize,

    /// Per-request budget for `/v1/embeddings` in **padded** tokens: inputs ×
    /// the longest input's token count (a batch is padded to its longest
    /// input). Default **32768** (64 full 512-token chunks); `0` = no budget.
    ///
    /// Bounds the batch's GPU working memory, which the memory accounting
    /// can't see and which `OpenVINO` keeps until the model is evicted —
    /// measured on multilingual-e5-large: ≈28 MB per 512-token input (+1.5 GB
    /// at 64 inputs, +7 GB at 256). Keep the model's `vram_gb` at least its
    /// weights plus this budget's worth. Short inputs are cheap: 256 queries
    /// of ~10 tokens are ~2.5k padded tokens.
    #[serde(default = "default_max_embedding_batch_tokens")]
    pub max_embedding_batch_tokens: usize,

    /// Server-wide ceiling on the per-request generation budget
    /// (`max_tokens` / `max_completion_tokens`), applied to chat AND legacy
    /// completions. Default **8192**; `0` = uncapped (explicit opt-out).
    ///
    /// The admission gate bounds *concurrency*, not *duration*: without this
    /// cap a single request with `max_tokens: 4000000000` — or one that never
    /// hits EOS with no `max_tokens` at all — pins an engine slot for hours
    /// (committee findings 8, 15). Requests above the cap are clamped to it,
    /// not rejected, so drop-in `OpenAI` clients keep working.
    #[serde(default = "default_max_tokens_cap")]
    pub max_tokens_cap: usize,

    /// No-token watchdog, first deadline (0.9.1 S1): the most seconds a text generation (chat,
    /// legacy completions, a realtime LLM turn) may go from the moment its engine starts it (not
    /// while it waits for a slot) to its first generated token. When it passes, the request ends
    /// with `generation_stalled` and the model's engine is marked dead (a GPU model sets the
    /// GPU-poisoned flag; a CPU or NPU model is booked as stalled until it is unloaded and
    /// loaded again). Default 600; `0` = off. Restart-only.
    #[serde(default = "default_stream_first_token_timeout_secs")]
    pub stream_first_token_timeout_secs: u64,

    /// No-token watchdog, second deadline (0.9.1 S1): the longest gap, in seconds, between two
    /// generated tokens after the first. Same action as
    /// [`Self::stream_first_token_timeout_secs`]. Default 120; `0` = off. Restart-only.
    #[serde(default = "default_stream_token_gap_timeout_secs")]
    pub stream_token_gap_timeout_secs: u64,

    /// IP address the HTTP server binds to. Default **`127.0.0.1`**
    /// (loopback — secure by default, T4.2). Set `"0.0.0.0"` for LAN/public
    /// exposure; with no `api_keys` in the keys file (admin keys do not count) that
    /// additionally requires [`allow_insecure_public_bind`](Self::allow_insecure_public_bind).
    #[serde(default = "default_bind_addr")]
    pub bind_addr: String,

    /// TCP port the HTTP server listens on. Default **11437** (dev port;
    /// stormVINO owns 11435 until cutover day).
    #[serde(default = "default_port")]
    pub port: u16,

    /// Explicit opt-in to a non-loopback bind while inference auth is OFF
    /// (no `api_keys` in the keys file; admin keys alone do not close inference,
    /// 0.9.1 S3). Default **false**: a public bind with open inference
    /// refuses to start (committee finding 32 — the shipped defaults must
    /// never boot an open server on the network by accident).
    #[serde(default)]
    pub allow_insecure_public_bind: bool,

    /// Directory for the `OpenVINO` GPU kernel blob cache.
    ///
    /// When set, `OpenVINO` writes the compiled GPU kernel to this directory on
    /// the first model load. The KV engines (text LLMs, vision) use a
    /// `kv-<tag>/` subdirectory per KV precision, the rest the directory itself (see
    /// `ov_cache_layout`). Subsequent server starts load the blob directly — cutting GPU JIT time from
    /// ~1–3 min per model to ~10–30 s. `null`/absent = no cache (default, current
    /// behaviour). Safe to point at a persistent directory across reboots.
    ///
    /// On Windows use a path next to the exe, e.g. `"C:\\RustedVINO\\ov_cache"`.
    ///
    /// **NPU:** NPU LLMs use it too (since 2026-09-28): `ov_pipeline_create`
    /// sets `CACHE_DIR`, storing one weightless compiled `.blob` per model
    /// (~0.66 GB for an 8B int4 model) — measured ~6 s cached load vs 30.7 s
    /// cold. The NPU *Whisper* pipeline uses it as well (since 2026-10-02; the
    /// earlier hang does not reproduce on the current stack; see the `Stt` arm
    /// in `lifecycle.rs`).
    #[serde(default)]
    pub ov_cache_dir: Option<String>,

    /// Ceiling on `ov_cache_dir`'s total on-disk size (GB), enforced by the
    /// background cache sweep — mirrors `total_vram_gb`'s "`0.0` disables
    /// gating" convention, so `0.0` still *reports* the cache size every sweep
    /// but never prunes. Irrelevant when `ov_cache_dir` itself is unset.
    ///
    /// The sweep is real and wired: `prune_ov_cache_if_over_budget`, spawned in
    /// `startup.rs`. This doc said "still-unbuilt" long after it shipped.
    #[serde(default)]
    pub ov_cache_max_gb: f64,

    /// How often the OV-cache management sweep (hash-precompute + prune)
    /// re-runs, on top of the once-at-startup pass. Default 21600s (6h).
    ///
    /// Consumed by the background task spawned in `startup.rs`, which calls
    /// `ModelManager::prune_ov_cache_if_over_budget` via `spawn_blocking` (the
    /// pass does blocking file I/O, including a SHA-256 stream over multi-GB
    /// files). This doc previously said "still-unbuilt ... not yet consumed by
    /// any running code" long after the sweep shipped.
    #[serde(default = "default_ov_cache_sweep_interval_secs")]
    pub ov_cache_sweep_interval_secs: u64,

    /// Layout of `ov_cache_dir` for the KV-cache engines (E-17). `"keyed"` (default): the
    /// continuous-batching engines (text LLMs, including a draft model) and the vision
    /// pipelines on GPU and CPU compile into `<ov_cache_dir>/kv-<tag>/`, one directory per
    /// `KV_CACHE_PRECISION` string passed to the plugin (`kv-auto` when none is passed,
    /// `kv-i8`, `kv-u4`, `kv-f16`; a vision pipeline passes its resolved precision, see
    /// `kv_precision_pass`). `"flat"`:
    /// everything in the root, as 0.8.x did - and with it the precision mixing described below, so
    /// use it only as a rollback.
    ///
    /// Why: on `OpenVINO` 2026.4.x a compiled blob does not record the KV precision, so a blob
    /// compiled under one precision is silently reused under another (or fails at the first
    /// inference with `Incorrect block size`). Whisper, image, TTS, embedding and NPU
    /// engines keep the root. Restart-only. Switching layout recompiles a model only if no usable blob
    /// is in the new location. A keyed directory the server cannot write to (a read-only,
    /// pre-warmed cache) is not used: those models compile without a blob cache.
    #[serde(
        default = "default_ov_cache_layout",
        deserialize_with = "de_ov_cache_layout"
    )]
    pub ov_cache_layout: String,

    /// Who decides the KV-cache precision the engines are built with.
    /// `"explicit"` (default): `RustedVINO` resolves each model's precision (the global
    /// `kv_cache_precision` for text models > the model's own IR hint > a rule: u4
    /// when the model has 4-bit weights, otherwise i8; vision models ignore the global key) and
    /// passes the result to the engine, so the context limit the gate computes is the precision
    /// the engine really uses. `"plugin"`: the 0.8.x behaviour - vision models pass nothing
    /// (plugin default), text models pass the global `kv_cache_precision` as configured.
    /// Restart-only. Changing it changes which `kv-<tag>` blob directory a vision model uses.
    #[serde(
        default = "default_kv_precision_pass",
        deserialize_with = "de_kv_precision_pass"
    )]
    pub kv_precision_pass: String,

    /// How the context limit (`max_model_len`, the prompt-length gate) is computed.
    /// `"exact"` (default): from the model's real KV layout and the exact pool the
    /// engine was built with, as `min(95 % of the pool capacity, native context, a learned
    /// ratchet)`. `"legacy"`: the 0.8.x estimate (every layer at the configured precision
    /// times a 0.85 margin). A model the server cannot describe with confidence, a CPU device,
    /// speculative decoding or an unverified engine version pair always uses the legacy estimate.
    /// Restart-only; the rollback switch if the exact limit proves wrong on a model.
    #[serde(default = "default_kv_gate_mode", deserialize_with = "de_kv_gate_mode")]
    pub kv_gate_mode: String,

    /// Size and gate with the exact layout model even on an OpenVINO/GenAI version pair the
    /// layouts were not verified on (`2026.4.1` is the verified pair). Default `false`.
    #[serde(default)]
    pub kv_gate_exact_on_unknown_engine: bool,

    /// Device-headroom limit for vision models on a discrete GPU.
    /// `"dynamic"` (default): the context limit also respects the free device memory left after
    /// the weights, every pool and whatever other processes hold, divided by what one prompt
    /// token transiently costs (`hidden_size · 4 · kv_transient_factor` bytes); recomputed after
    /// every load, unload and resize. `"off"`: no headroom limit (vision models on a discrete GPU
    /// then keep the legacy limit). Restart-only.
    #[serde(
        default = "default_kv_headroom_mode",
        deserialize_with = "de_kv_headroom_mode"
    )]
    pub kv_headroom_mode: String,
    /// Multiplier on the per-token transient cost of a prompt (default 2.0).
    #[serde(default = "default_kv_transient_factor")]
    pub kv_transient_factor: f64,
    /// Device memory (GiB) kept free beyond everything visible (default 0.5).
    #[serde(default = "default_kv_headroom_margin_gib")]
    pub kv_headroom_margin_gib: f64,
    /// Smallest headroom limit ever reported for a vision model, in tokens (default 4096).
    #[serde(default = "default_kv_headroom_min_tokens")]
    pub kv_headroom_min_tokens: usize,
    /// A vision request longer than this many tokens holds a transient device-memory reservation
    /// while it runs (default 32768).
    #[serde(default = "default_kv_headroom_inflight_tokens")]
    pub kv_headroom_inflight_tokens: usize,
    /// Run a one-token generation after each load so the pool is materialised and the weights'
    /// real memory can be measured (default `true`).
    #[serde(default = "default_true")]
    pub kv_warmup: bool,

    /// Master switch for the KV-cache pressure monitor
    /// (the project's internal engineering log). Default `false` —
    /// deliberately, not incidentally: unlike a numeric ceiling that happens
    /// to be inert at `0.0`, this is an explicit gate an operator must flip,
    /// per the plan's own ops-review finding that "detect and log" is not
    /// automatically harmless (a model legitimately busy near the threshold
    /// would flap warnings indefinitely on a long-uptime box). When `false`,
    /// the sweep task is never even spawned — no periodic cost at all.
    #[serde(default)]
    pub kv_pressure_monitor_enabled: bool,

    /// Sustained KV-pool occupancy (0-100) that counts as "under pressure."
    /// No usable default — `0.0` is a sentinel meaning "not configured," and
    /// `Config::validate` rejects `kv_pressure_monitor_enabled: true` paired
    /// with this left at `0.0`: a box that turns monitoring on must state
    /// its own real number, because what "pressure" means differs by
    /// hardware (a UMA box's KV pressure competes with host RAM already
    /// under its own contention; a discrete-VRAM box's KV pool is isolated
    /// from that entirely) — see the plan's ops-review section.
    #[serde(default)]
    pub kv_pressure_threshold_pct: f64,

    /// How long (seconds) occupancy must stay continuously at or above
    /// `kv_pressure_threshold_pct` before a model is flagged — reacting to a
    /// single sample would flag on any normal traffic burst. Same
    /// no-usable-default convention as `kv_pressure_threshold_pct`: `0`
    /// means "not configured," and enabling the monitor without setting
    /// this is rejected the same way.
    #[serde(default)]
    pub kv_pressure_sustained_secs: u64,

    /// How often the pressure monitor samples every Ready model's live KV
    /// occupancy. Unlike the threshold/duration above, this has a real
    /// default (15s) — it's a cadence, not a safety-relevant threshold that
    /// differs by hardware, and checking a cheap in-memory counter every
    /// 15s costs nothing (contrast the 6h `ov_cache_sweep_interval_secs`,
    /// which walks a real directory). Irrelevant while the monitor is
    /// disabled.
    #[serde(default = "default_kv_pressure_sweep_interval_secs")]
    pub kv_pressure_sweep_interval_secs: u64,

    /// Minimum time (seconds) between two completed resizes of the *same*
    /// model via `POST /v1/admin/models/{id}/resize`. Applies unconditionally
    /// — independent of `kv_pressure_monitor_enabled` — because the resize
    /// endpoint is a manual admin action available regardless of whether the
    /// monitor is on, and nothing else stops a script or a nervous operator
    /// from calling it back-to-back on a single-slot model (each call is a
    /// real evict+reload — a genuine mini-outage for that model, not a free
    /// action). Real default (30s), not a sentinel: this is a safety floor
    /// that should hold even on a box that never touches this config at all.
    #[serde(default = "default_kv_resize_cooldown_secs")]
    pub kv_resize_cooldown_secs: u64,

    /// How long (seconds) a model load may run before the request gets `504 load_timeout` and the
    /// load is booked as stalled (step 6b): its memory stays reserved, new loads on its device get
    /// `503 device_stalled`, and a late result is still committed or discarded. `0` = no deadline.
    /// Otherwise at least 60. Per-model `load_timeout_secs` overrides it.
    #[serde(default = "default_load_timeout_secs")]
    pub load_timeout_secs: u64,

    /// F1b: how long (seconds) a request to a `load: on_demand` model that is
    /// not loaded holds while its load runs, instead of an immediate `503` + `Retry-After`.
    /// `0` (the default) = no wait, the pre-0.9.1 answer. At most [`MAX_ON_DEMAND_WAIT_SECS`],
    /// and below a non-zero global `load_timeout_secs`; per request it is also kept below the
    /// model's own effective `load_timeout_secs`. Global only; a config reload does not apply it
    /// (restart).
    #[serde(default)]
    pub on_demand_wait_secs: u64,

    /// Tuning for the `--supervise` crash auto-restart wrapper (Linux only —
    /// inert on Windows, where supervise mode never runs). Absent block =
    /// every field defaults, so no existing config file needs updating.
    #[serde(default)]
    pub supervisor: SupervisorConfig,

    /// Collector for unknown top-level config keys (T4.1, committee
    /// finding 19). Serde routes any key not matched above into this map so
    /// [`Config::load`] can warn about it — a typo'd `api_key` /
    /// `cors_origins` must never be silently dropped, because the
    /// security-critical defaults (auth OFF, CORS `*`) would silently take
    /// over. Security-adjacent typos hard-fail; everything else logs WARN.
    #[serde(flatten)]
    pub unknown_fields: HashMap<String, serde_json::Value>,
}

fn default_kv_cache_precision() -> String {
    "u8".to_owned()
}

/// Accepted `kv_cache_precision` values, in the order the error message lists them.
const KV_PRECISION_ACCEPTED: &str =
    r#""auto", "i8", "u4", "f16" (or "" = auto; "u8" = deprecated alias of "i8")"#;

/// Normalise a configured `kv_cache_precision` to the string handed to the
/// engine: `""` (plugin default; also what `"auto"` means until Stage 2), `"i8"`,
/// `"u4"` or `"f16"`. `"u8"` becomes `"i8"` with one INFO per process (the GPU
/// plugin stores both as 8-bit integers; `i8` is the honest name).
///
/// # Errors
/// Any other value, including the previously tolerated `"f32"` and `"i4"`
/// (the plugin would rewrite `i4` to `u4` anyway); the message lists the accepted values.
pub fn normalize_kv_precision(raw: &str) -> anyhow::Result<&'static str> {
    match raw {
        "" | "auto" => Ok(""),
        "i8" => Ok("i8"),
        "u4" => Ok("u4"),
        "f16" => Ok("f16"),
        "u8" => {
            static ALIAS_INFO: std::sync::Once = std::sync::Once::new();
            ALIAS_INFO.call_once(|| {
                tracing::info!("kv_cache_precision u8 is a deprecated alias, stored as i8");
            });
            Ok("i8")
        }
        other => anyhow::bail!(
            "kv_cache_precision {other:?} is not supported; accepted values: {KV_PRECISION_ACCEPTED}"
        ),
    }
}

/// Backoff/cap/health-check tuning for the `--supervise` wrapper
/// ([`crate::supervisor`], Unix only). A worker crash restarts it after an
/// increasing backoff; a rolling window bounds total restarts so a
/// deterministic crash cannot loop forever against a slow cold GPU init.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct SupervisorConfig {
    /// Delay before the first respawn attempt, doubling each subsequent
    /// attempt up to [`backoff_max_secs`](Self::backoff_max_secs). Default 5s.
    #[serde(default = "default_backoff_base_secs")]
    pub backoff_base_secs: u64,

    /// Ceiling on the respawn backoff delay. Default 60s.
    #[serde(default = "default_backoff_max_secs")]
    pub backoff_max_secs: u64,

    /// Maximum respawns allowed within [`restart_window_secs`](Self::restart_window_secs)
    /// before the supervisor gives up and exits (no more respawns). Default 5.
    #[serde(default = "default_max_restarts")]
    pub max_restarts: usize,

    /// Rolling window over which [`max_restarts`](Self::max_restarts) is
    /// counted — restarts older than this age out, so occasional unrelated
    /// crashes don't accumulate toward the cap. Default 600s (10 min).
    #[serde(default = "default_restart_window_secs")]
    pub restart_window_secs: u64,

    /// How long to wait for `/health` to come up after a respawn before
    /// treating the worker as hung (cold GPU init is ~47s; this needs
    /// margin above that). Default 90s.
    #[serde(default = "default_health_timeout_secs")]
    pub health_timeout_secs: u64,

    /// Grace period after `SIGTERM`ing a hung worker before escalating to
    /// `SIGKILL` — the one sanctioned `SIGKILL` in this codebase, because a
    /// worker that never became healthy never reached a state worth
    /// preserving. Default 10s. A generation-watchdog kill of a *healthy* worker waits at
    /// least the worker's drain + teardown floor + exit wait (48 s) regardless, so its other
    /// models' streams are not killed with `SIGKILL` mid-drain. A worker still waiting
    /// for a model load or tearing down declares how long it needs, and the supervisor waits
    /// that out too (up to ~11 min); the grace is never below 1 s.
    #[serde(default = "default_hang_kill_grace_secs")]
    pub hang_kill_grace_secs: u64,

    /// How often, once the worker is healthy, to poll
    /// `GET /v1/admin/health/watchdog` for the oldest in-flight generation's
    /// age. `/health` alone can't see a wedged engine thread — see
    /// the project's internal engineering log. Default 30s.
    #[serde(default = "default_watchdog_poll_secs")]
    pub watchdog_poll_secs: u64,

    /// Age past which an in-flight generation is treated as hung rather than
    /// merely slow, triggering the same [`hang_kill_grace_secs`]-graced
    /// `SIGTERM`→`SIGKILL` escalation as a worker that never became healthy.
    /// Default 600s (10 min). This is slack **on top of each request's own allowance**: a
    /// generation is counted from the moment it exceeds `prompt_tokens / 200` + `max_tokens / 5`
    /// seconds (floors well below the measured 375 tok/s prefill and 13-18 tok/s decode of the
    /// 27B vision model on a B60), so a 92k-token prompt with a 6k-token answer is not killed
    /// at 600 s (it was, before the allowance existed). Set to `0` to disable the generation
    /// watchdog entirely.
    #[serde(default = "default_watchdog_hang_ceiling_secs")]
    pub watchdog_hang_ceiling_secs: u64,
    /// After a worker left WITHOUT teardown (exit 75/76, killed by a signal, hang or watchdog
    /// kill) the next worker's first GPU model load must finish within this many seconds, or the
    /// supervisor kills it and respawns. On the B60 such an exit can leave the copy engine stuck
    /// so the next process's first GPU work never runs; the stuck worker's own exit triggers the
    /// driver's GT reset. Armed at most once per episode (a worker killed by this deadline does not arm it again). Since the GPU
    /// readiness probe ([`Self::gpu_probe_deadline_secs`]) this is a backstop, kept until a
    /// destructive test shows the probe catches the stuck state: default 180, above a 27B's
    /// 48-78 s load (the old 30 killed it); `0` disables it.
    #[serde(default = "default_gpu_first_load_deadline_secs")]
    pub gpu_first_load_deadline_secs: u64,
    /// The GPU readiness probe after a no-teardown exit: before the
    /// next worker starts, a child running a tiny compile + inference on every GPU must exit
    /// within this many seconds; past it it is killed (its exit resets a stuck B60) and run
    /// again, up to 3 times. Default 15 — **provisional** until the normal probe time (cold and
    /// warm, desktop busy) is measured on the B60; `0` disables the probe.
    #[serde(default = "default_gpu_probe_deadline_secs")]
    pub gpu_probe_deadline_secs: u64,
}

fn default_gpu_first_load_deadline_secs() -> u64 {
    180
}

fn default_gpu_probe_deadline_secs() -> u64 {
    15
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        Self {
            backoff_base_secs: default_backoff_base_secs(),
            backoff_max_secs: default_backoff_max_secs(),
            max_restarts: default_max_restarts(),
            restart_window_secs: default_restart_window_secs(),
            health_timeout_secs: default_health_timeout_secs(),
            hang_kill_grace_secs: default_hang_kill_grace_secs(),
            watchdog_poll_secs: default_watchdog_poll_secs(),
            watchdog_hang_ceiling_secs: default_watchdog_hang_ceiling_secs(),
            gpu_first_load_deadline_secs: default_gpu_first_load_deadline_secs(),
            gpu_probe_deadline_secs: default_gpu_probe_deadline_secs(),
        }
    }
}

/// A string setting that names itself in the error when the JSON type is wrong (serde's own
/// `invalid type: null, expected a string` does not say which key).
fn de_named_string<'de, D: serde::Deserializer<'de>>(
    d: D,
    key: &str,
    accepted: &str,
) -> Result<String, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    match v {
        serde_json::Value::String(s) => Ok(s),
        other => Err(serde::de::Error::custom(format!(
            "{key} must be a string ({accepted}), got {other}"
        ))),
    }
}

fn de_kv_cache_precision<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    de_named_string(
        d,
        "kv_cache_precision",
        "accepted values: \"auto\", \"i8\", \"u4\", \"f16\" (or \"\" = auto; \"u8\" = deprecated alias of \"i8\")",
    )
}

fn de_ov_cache_layout<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    de_named_string(d, "ov_cache_layout", "\"keyed\" or \"flat\"")
}

fn default_ov_cache_layout() -> String {
    "keyed".to_owned()
}

fn de_kv_precision_pass<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    de_named_string(d, "kv_precision_pass", "\"explicit\" or \"plugin\"")
}

fn default_kv_precision_pass() -> String {
    "explicit".to_owned()
}

fn de_kv_headroom_mode<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    de_named_string(d, "kv_headroom_mode", "\"dynamic\" or \"off\"")
}

fn default_kv_headroom_mode() -> String {
    "dynamic".to_owned()
}

fn default_kv_transient_factor() -> f64 {
    2.0
}

fn default_kv_headroom_margin_gib() -> f64 {
    0.5
}

fn default_kv_headroom_min_tokens() -> usize {
    4096
}

fn default_kv_headroom_inflight_tokens() -> usize {
    32_768
}

fn default_true() -> bool {
    true
}

fn de_kv_gate_mode<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    de_named_string(d, "kv_gate_mode", "\"exact\" or \"legacy\"")
}

fn default_kv_gate_mode() -> String {
    "exact".to_owned()
}

fn default_ov_cache_sweep_interval_secs() -> u64 {
    21_600 // 6h
}

fn default_kv_pressure_sweep_interval_secs() -> u64 {
    15
}

fn default_kv_resize_cooldown_secs() -> u64 {
    30
}

pub(crate) fn default_load_timeout_secs() -> u64 {
    600
}

/// Smallest non-zero `load_timeout_secs`: a cold compile of a large model takes minutes.
pub(crate) const MIN_LOAD_TIMEOUT_SECS: u64 = 60;

/// Largest `on_demand_wait_secs` (F1b): covers the slow fleet loads (a big VLM
/// on a dGPU) and stays well below the 600 s default `load_timeout_secs`.
pub const MAX_ON_DEMAND_WAIT_SECS: u64 = 300;

/// F1b: `on_demand_wait_secs` is at most [`MAX_ON_DEMAND_WAIT_SECS`] and, when the global
/// `load_timeout_secs` is not `0` (no deadline), below it: a request must never be told to
/// wait longer than the load it waits for may run.
fn validate_on_demand_wait(wait_secs: u64, load_timeout_secs: u64) -> anyhow::Result<()> {
    anyhow::ensure!(
        wait_secs <= MAX_ON_DEMAND_WAIT_SECS,
        "on_demand_wait_secs must be at most {MAX_ON_DEMAND_WAIT_SECS} (got {wait_secs})"
    );
    anyhow::ensure!(
        load_timeout_secs == 0 || wait_secs < load_timeout_secs,
        "on_demand_wait_secs ({wait_secs}) must be below load_timeout_secs \
         ({load_timeout_secs}): a request may not wait longer than its load may run"
    );
    Ok(())
}

/// Upper bound on the `supervisor` seconds fields and the two sweep intervals: one year (FM-4).
/// Far above any useful value, and low enough that summing several of them or adding one to an
/// `Instant` cannot overflow.
pub const MAX_INTERVAL_SECS: u64 = 365 * 24 * 60 * 60;

/// Default eviction-grace window: 2s. It only has to cover the race between a
/// load finishing and its first request arriving; a model with a running
/// request is protected separately (`ModelRecord::has_in_flight`) and so is the
/// realtime serving set. Was 300s (decided 2026-08-06, realtime arbitration v2)
/// until Windows pass 2 (D4) showed it blocking ordinary model switches for
/// five minutes; a box that wants the long window sets `eviction_grace_secs`.
pub(crate) fn default_eviction_grace_secs() -> f64 {
    2.0
}

/// Known realtime-capability labels (`ModelPolicy::capabilities`,
/// `ViableMinimumLlm::require_capabilities`) — validated at registration
/// (`validate_model_entry`, `Config::validate`) so a typo'd or
/// forward-looking label fails loud rather than silently never matching.
/// `"tool_calling"` is the only label defined today — the forward-compat
/// seam the project's internal engineering log D7 checks for;
/// add new labels here as new realtime capabilities are actually wired.
pub(crate) const KNOWN_CAPABILITIES: &[&str] = &["tool_calling"];

/// Realtime voice arbitration v2 (D5): operator-set deterministic model
/// defaults for the realtime channel — sibling of [`Config::preload`] in
/// shape, but read by `resolve_realtime_llm` instead of startup loading.
/// Every field independently optional: a node can set only `llm_model` and
/// leave STT/TTS to today's discovery behavior.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct RealtimeDefaults {
    /// Default STT model for a realtime session that doesn't request one.
    #[serde(default)]
    pub stt_model: Option<String>,
    /// Default LLM/VLM model — the `resolve_realtime_llm` substitute of
    /// last resort, and the ID `eviction_protected` treats as the realtime
    /// default while consulted by D2's serving set.
    #[serde(default)]
    pub llm_model: Option<String>,
    /// Default TTS model for a realtime session that doesn't request one.
    #[serde(default)]
    pub tts_model: Option<String>,
    /// Default embedding model for memory retrieval. Not eviction-protected
    /// (D2 deliberately excludes `embed_model` from the serving set —
    /// embeddings are convenience-only and sessions degrade gracefully
    /// without them).
    #[serde(default)]
    pub embed_model: Option<String>,
}

/// Realtime voice arbitration v2 (D4): the "viable minimum" capability
/// floor a model must clear to serve the realtime LLM slot.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
pub struct RealtimeViableMinimum {
    /// The LLM/VLM floor. `None` ⇒ no floor configured for this node (every
    /// `text_gen`/`vision` model is viable, today's de-facto behavior).
    #[serde(default)]
    pub llm: Option<ViableMinimumLlm>,
}

/// The LLM/VLM half of [`RealtimeViableMinimum`]. A model **clears the
/// floor** iff its kind is in `kinds`, its context length clears
/// `min_context_tokens`, and its declared [`ModelPolicy::capabilities`]
/// is a superset of `require_capabilities`.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ViableMinimumLlm {
    /// `ModelKind` labels a model must have one of to be viable. Default:
    /// `["text_gen", "vision"]` — today's de-facto realtime-LLM kind set.
    #[serde(default = "default_viable_minimum_kinds")]
    pub kinds: Vec<String>,
    /// Minimum context length (tokens) a model must support. Checked
    /// against `ModelRecord::max_prompt_tokens` once known (post-load);
    /// `0` (the default) disables this check.
    #[serde(default)]
    pub min_context_tokens: usize,
    /// Capability labels (see [`KNOWN_CAPABILITIES`]) a model's declared
    /// `ModelPolicy::capabilities` must be a superset of. Empty (the
    /// default) requires nothing — this is the tool-calling forward-compat
    /// seam (D7): adding `"tool_calling"` here later is a pure additive
    /// config edit.
    #[serde(default)]
    pub require_capabilities: Vec<String>,
}

fn default_viable_minimum_kinds() -> Vec<String> {
    vec!["text_gen".to_owned(), "vision".to_owned()]
}

fn default_backoff_base_secs() -> u64 {
    5
}

fn default_backoff_max_secs() -> u64 {
    60
}

fn default_max_restarts() -> usize {
    5
}

fn default_restart_window_secs() -> u64 {
    600
}

fn default_health_timeout_secs() -> u64 {
    90
}

fn default_hang_kill_grace_secs() -> u64 {
    10
}

fn default_watchdog_poll_secs() -> u64 {
    30
}

fn default_watchdog_hang_ceiling_secs() -> u64 {
    600
}

/// When a model's weights are loaded into VRAM (co-residency Slice 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadPolicy {
    /// Loaded only on an explicit admin load (or preload); a chat request for a
    /// not-yet-loaded model fails fast with a bare 503. The default — today's
    /// behaviour, unchanged.
    #[default]
    Eager,
    /// Loaded lazily on the first chat request: the chat path kicks off a
    /// background load and returns 503 + `Retry-After` until the model is Ready
    /// (Slice 3c). Lets a rarely-used model (e.g. SDXL on a 16 GB B50) stay
    /// unloaded until first use.
    OnDemand,
}

/// Reasoning-content extraction style for a model that emits `<think>` blocks.
///
/// When set on a [`ModelPolicy`] (or auto-detected from the chat template), the
/// chat handler runs [`extract_thinking`] on the model's buffered response and
/// surfaces the reasoning as `reasoning_content`. When absent (`None`),
/// responses are returned verbatim — correct for models that have no `<think>`
/// blocks, and essential when a model emits `<think>` as literal content rather
/// than a reasoning delimiter (a correctness hazard with an unconditional strip).
///
/// [`extract_thinking`]: crate::prompt_builder::extract_thinking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningParser {
    /// Qwen3 / DeepSeek-style `<think>…</think>` reasoning block.
    Qwen3,
    /// Same `<think>` extraction under the OVMS `gpt-oss` parser name — present
    /// for config parity; the extraction logic is identical to [`Qwen3`](Self::Qwen3).
    /// Also used as a model-family tag: drives channel-prefill and
    /// `model_identity` routing in the realtime voice pipeline.
    GptOss,
    /// Mistral / Mixtral family. No reasoning extraction; used as a model-family
    /// tag so `VoicePolicy` can apply family-specific voice treatment.
    Mistral,
    /// Microsoft Phi family (Phi-3, Phi-3.5, Phi-4, Phi-4-mini, …). No
    /// reasoning extraction for most variants; used as a family tag for voice
    /// policy — add per-variant behaviour here when quirks emerge.
    Phi,
}

/// How a model that has no `device` and no `tier_preference` of its own is placed.
///
/// `Auto` (the default, and what every fleet config states explicitly) runs the capability-aware
/// placement engine: light models are parked on an iGPU or NPU, heavy ones go dGPU-first. `Device`
/// loads such a model on the global [`Config::device`] and nothing else - the bundle default
/// (`device: CPU`), because most machines have no iGPU worth using: on a weak iGPU every light
/// model measured 20-45 % slower than the CPU, `SpeechT5` was unusable (RTF 16-100), and a vision
/// model's GPU kernel build hung. An explicit per-model `device` or `tier_preference` still wins
/// over either value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum DefaultPlacement {
    /// Capability-aware placement by model kind and size (iGPU / NPU offload, dGPU first for heavy).
    #[default]
    Auto,
    /// The global `device`, for every model without its own `device` / `tier_preference`.
    Device,
}

/// Default for [`ModelPolicy::evictable`]: a model with no policy is evictable,
/// i.e. the pre-feature behaviour. (A bare `#[derive(Default)]` would give
/// `false` — wrong — so [`ModelPolicy`] hand-implements `Default`.)
fn default_evictable() -> bool {
    true
}

/// Per-model co-residency policy (Slices 1–3).
///
/// Controls how a model behaves under VRAM pressure when multiple models are
/// resident, and when it is loaded. Every field defaults to the pre-feature
/// behaviour, so a model with no policy block (or an absent map entry) is
/// treated exactly as before.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ModelPolicy {
    /// When `true`, this model is **never** chosen as an eviction victim — the
    /// always-resident primary for a multiuser box. A load that needs VRAM
    /// evicts non-pinned residents instead; if only pinned models are resident
    /// and the load still does not fit, the load fails rather than evicting a
    /// pinned model. Default `false`.
    #[serde(default)]
    pub pinned: bool,

    /// Soft eviction-order weight: among non-pinned, idle candidates, a model
    /// with a **higher** priority is evicted **later** (kept resident longer).
    /// Ordering under pressure is: pinned (never) > in-flight (last resort) >
    /// higher priority > least-recently-used. Default `0`; negative values are
    /// allowed (evicted before the default tier).
    #[serde(default)]
    pub priority: i32,

    /// Path to a chat template that **overrides** the one shipped in the model
    /// directory. Relative paths resolve against the config file's own
    /// directory, so a repo-shipped template can be referenced portably.
    ///
    /// Exists because a converted model can ship a template that is a stale or
    /// incomplete revision of upstream's. Measured case (2026-09-08,
    /// the project's internal engineering log): `lfm2-24b-a2b-int4-ov`'s template
    /// renders **nothing** for an assistant turn carrying `tool_calls`, so a
    /// tool-call turn replays to the model as an empty
    /// `<|im_start|>assistant<|im_end|>` — the model is shown a tool result it
    /// never appears to have asked for, which corrupts every multi-turn tool
    /// conversation.
    ///
    /// Overriding here rather than editing the model directory keeps the
    /// model tree pristine (it is often a read-only or re-downloadable
    /// artifact) and makes the substitution visible in config review.
    #[serde(default)]
    pub chat_template: Option<std::path::PathBuf>,

    /// Co-residency Slice 2: per-model bounded KV-cache pool target in GB,
    /// overriding the global [`Config::default_kv_cache_gb`] for this model.
    /// `None` (the default) falls through to the global default (which is itself
    /// `0.0` = unbounded grab-all unless set). When `Some(g)`, the model's KV
    /// pool is clamped to `g` GB at load. Must be `> 0.0` when present (a `0.0`
    /// partition is meaningless — leave it `None` for unbounded). The global
    /// [`Config::cache_size_gb`] cap still applies on top.
    #[serde(default)]
    pub kv_cache_gb: Option<f64>,

    /// Co-residency Slice 2: per-model cap on concurrent sequences, overriding
    /// the global [`Config::max_num_seqs`] for this model's CB scheduler **and**
    /// its HTTP 429 admission gate. `None` (the default) uses the global value.
    /// Lets a co-resident model take a smaller batch so it does not contend for
    /// the whole card. Must be `>= 1` when present.
    ///
    /// Two-layer resolution (CB and VLM engines alike):
    /// `effective_cap = max_concurrent_streams ?? max_num_seqs (if > 0) ?? engine default`.
    /// This is a default, not a ceiling — an explicit `max_concurrent_streams`
    /// may exceed `max_num_seqs`.
    #[serde(default)]
    pub max_concurrent_streams: Option<usize>,

    /// Co-residency Slice 3: when this model's weights are loaded — `eager`
    /// (default, today's behaviour) or `on_demand` (lazy load on first chat
    /// request, with an honest 503 + `Retry-After` cold start). See
    /// [`LoadPolicy`].
    #[serde(default)]
    pub load: LoadPolicy,

    /// Co-residency Slice 3: when `false`, this model is **never** chosen as an
    /// eviction victim even though it is not [`pinned`](Self::pinned). Both are
    /// hard excludes from eviction; the distinction is intent. Useful combined
    /// with `load: on_demand` — a model that loads lazily yet, once resident,
    /// stays put. Default `true` (today's behaviour: evictable).
    #[serde(default = "default_evictable")]
    pub evictable: bool,

    /// Phase C1: explicit per-model `OpenVINO` device override (e.g. `"GPU.0"`,
    /// `"CPU"`). The operator names the device this model loads on; `None` (the
    /// default) keeps the pre-feature behaviour — the model loads on the global
    /// [`Config::device`]. This generalizes [`Config::embedding_device`] from one
    /// hardcoded kind to every model. The resolved device determines the model's
    /// memory domain (discrete GPU → its own VRAM pool; iGPU/CPU/NPU → the shared
    /// `system` RAM domain), so admission and eviction are charged correctly. A
    /// device the running `OpenVINO` did not enumerate is rejected at startup with
    /// a clear error rather than silently falling back.
    #[serde(default)]
    pub device: Option<String>,

    /// Phase C2: per-model capability-tier preference, overriding the built-in
    /// per-kind default. An ordered list of kebab-case [`DeviceTier`] labels
    /// (`"heavy"`, `"strong-igpu"`, `"weak-igpu"`, `"npu"`, `"fallback"`) — the
    /// placement engine picks the first tier with an available device. Sits
    /// *between* the explicit [`device`](Self::device) override (which wins
    /// outright and skips the engine) and the built-in default. Empty/absent ⇒
    /// the engine uses the kind's size-aware default. An unrecognised tier label,
    /// or a preference no device on this box satisfies, is rejected at startup.
    ///
    /// [`DeviceTier`]: crate::device_inventory::DeviceTier
    #[serde(default)]
    pub tier_preference: Vec<String>,

    /// Per-model reasoning-parser override (gap-2 vs OVMS).
    ///
    /// When set, the chat handler runs [`extract_thinking`] on the buffered
    /// response for this model and surfaces `<think>…</think>` content as
    /// `reasoning_content`. When absent (`None`), the server auto-detects from
    /// the chat template: if it contains `enable_thinking`, the parser is set to
    /// [`ReasoningParser::Qwen3`] automatically. Non-thinking models (Mistral,
    /// FLUX, Whisper, …) are left with `None` — no scan, no false strip.
    ///
    /// Explicit values: `"qwen3"`, `"gpt_oss"`, `"mistral"`, `"phi"`.
    ///
    /// [`extract_thinking`]: crate::prompt_builder::extract_thinking
    #[serde(default)]
    pub reasoning_parser: Option<ReasoningParser>,

    /// NPU-only: the compile-time prompt/context-length ceiling
    /// (`MAX_PROMPT_LEN`) for `OpenVINO` `GenAI`'s static `LLMPipeline`. The NPU
    /// compiles a fixed-shape graph ahead of time (unlike GPU/CPU's dynamic-shape
    /// kernels), so this bound is baked in at compile time, not adjustable per
    /// request. `None` (the default) leaves `OpenVINO`'s own default (1024
    /// tokens) untouched — pure passthrough. Raising it grows the compiled
    /// KV-cache allocation and load/compile time; there is no documented upper
    /// ceiling to validate against, only `>= 1` when present. Ignored for every
    /// non-NPU model. See the project's internal engineering log and `src/npu_engine.rs`'s module
    /// doc comment for the full rationale.
    #[serde(default)]
    pub max_prompt_len: Option<u32>,

    /// NPU-only: compile-time `MIN_RESPONSE_LEN` — the room the static NPU
    /// KV cache reserves for the answer on top of `max_prompt_len`. The NPU
    /// cache holds `max_prompt_len + min_response_len` tokens in total, so a
    /// long prompt leaves only this much for output regardless of the
    /// request's `max_tokens` (the server then reports `finish_reason:
    /// "length"`). `None` keeps `OpenVINO`'s default (128 — measured on Lunar
    /// Lake: 863-token prompt + 290 generated = 1024 + 128 + 1). Raising it
    /// grows the compiled KV cache; 512 cost no measurable extra load time.
    /// Ignored for every non-NPU model.
    #[serde(default)]
    pub min_response_len: Option<u32>,

    /// Draft-model speculative decoding (opt-in). `None` = plain decoding,
    /// today's behaviour. See [`SpeculativeConfig`].
    #[serde(default)]
    pub speculative: Option<SpeculativeConfig>,

    /// Image-gen `generation_metadata.precision` (Tier 3,
    /// the image-metadata plan) — operator-supplied, e.g. `"int8"`,
    /// `"fp16"`. Not derivable from the model directory (only a naming
    /// convention, which the plan explicitly rejects as a source — see
    /// `src/ov_image.rs`'s directory-naming comment). `None` (the default)
    /// omits the field from the response rather than guessing.
    #[serde(default)]
    pub precision: Option<String>,

    /// Image-gen `generation_metadata.model_source` (Tier 3) — the
    /// Hugging Face repo id this model was converted from, e.g.
    /// `"OpenVINO/LCM_Dreamshaper_v7-int8-ov"`. Operator-supplied; `None`
    /// omits the field.
    #[serde(default)]
    pub model_source: Option<String>,

    /// Image-gen `generation_metadata.model_revision` (Tier 3) — the HF commit
    /// sha the model was pulled at. Operator-supplied; `None` omits the field.
    #[serde(default)]
    pub model_revision: Option<String>,

    /// Realtime voice arbitration v2 (D4/D7): operator-declared capability
    /// set (e.g. `"tool_calling"`), checked against
    /// `RealtimeViableMinimum::llm`'s `require_capabilities`. Validated
    /// against [`KNOWN_CAPABILITIES`] at registration (fail-loud, matching
    /// `tier_preference`'s precedent) — an unrecognised capability is a
    /// config error, not a silently-ignored no-op. Default empty.
    #[serde(default)]
    pub capabilities: Vec<String>,

    /// Realtime voice arbitration v2 (D3): per-model override of
    /// [`Config::eviction_grace_secs`]. `None` (the default) uses the
    /// global value. `Some(0.0)` opts this model out of grace protection
    /// entirely (pure LRU for this model, even during its grace window
    /// under the global setting).
    #[serde(default)]
    pub eviction_grace_secs: Option<f64>,

    /// Per-model override of [`Config::load_timeout_secs`] (same rules: `0` = no deadline,
    /// otherwise at least 60). `None` (the default) uses the global value.
    #[serde(default)]
    pub load_timeout_secs: Option<u64>,

    /// Per-model override of [`Config::enable_prefix_caching`] (KV-block reuse
    /// across requests sharing a prompt prefix). `None` (the default) uses the
    /// global flag; `Some(true)` / `Some(false)` force it on / off for this
    /// model. Applies from the model's next load.
    #[serde(default)]
    pub enable_prefix_caching: Option<bool>,

    /// Per-model KV-cache precision (`"auto"`, `"i8"`, `"u4"`, `"f16"`; `"u8"` is the deprecated
    /// alias of `"i8"`), overriding the global `kv_cache_precision` and, for a vision model, the
    /// rule. `None` (the default) lets the global key (text models) and the rule decide.
    /// Applies from the model's next load, on a plain GPU device with
    /// `kv_precision_pass: "explicit"` only; on CPU, NPU or in plugin mode it is ignored with a
    /// one-time warning. Speculative-decoding models ignore it too.
    #[serde(default)]
    pub kv_cache_precision: Option<String>,

    /// The operator's per-model sampling profile layer (0.9.1 V5a): defaults
    /// for a request that does not set its own, over the model's
    /// `generation_config.json`. `None` (the default) = no operator layer.
    /// Hot: a change applies to the next request. See [`SamplingConfig`].
    #[serde(default)]
    pub sampling: Option<SamplingConfig>,

    /// The model's default thinking budget in tokens (0.9.1): a request that sets no
    /// `thinking_budget_tokens` / `reasoning_effort` / `enable_thinking` of its own thinks at
    /// most this long. STRICT: a request that reaches it inside its reasoning stops with
    /// `finish_reason: "length"` and normally no answer. `0` = thinking off by default; `None`
    /// (the default) = no budget. Hot: a change applies to the next request. See
    /// `crate::thinking_budget` for the precedence and the clamp below `max_tokens`.
    #[serde(default)]
    pub thinking_budget_tokens: Option<u32>,
}

impl Default for ModelPolicy {
    fn default() -> Self {
        Self {
            pinned: false,
            priority: 0,
            chat_template: None,
            kv_cache_gb: None,
            max_concurrent_streams: None,
            load: LoadPolicy::Eager,
            evictable: default_evictable(),
            device: None,
            tier_preference: Vec::new(),
            reasoning_parser: None,
            max_prompt_len: None,
            min_response_len: None,
            speculative: None,
            precision: None,
            model_source: None,
            model_revision: None,
            capabilities: Vec::new(),
            eviction_grace_secs: None,
            load_timeout_secs: None,
            enable_prefix_caching: None,
            kv_cache_precision: None,
            sampling: None,
            thinking_budget_tokens: None,
        }
    }
}

/// Opt-in draft-model speculative decoding for a dense `TextGen` model.
///
/// Enabling this makes EVERY request to the target model speculative — the
/// pipeline is built with the draft attached and can no longer serve plain
/// requests at all (empirical: `OpenVINO` `GenAI` rejects non-assisted
/// requests on a draft-attached `ContinuousBatchingPipeline` with a
/// `RuntimeError`). Opt in only after verifying a net win for this specific
/// target — see the project's internal engineering log 2026-07-17: benefit requires target TPOT
/// ≳ 19 ms/token.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeConfig {
    /// Directory name of the draft model under `models_dir` (NOT required to
    /// be — and deliberately should not be — a registered `models` entry).
    /// Must share the target's tokenizer (validated via `vocab_size`).
    pub draft_model: String,
    /// Operator-measured VRAM footprint of the draft (weights + its internal
    /// KV) in GB, folded into the TARGET's admission charge. Required, same
    /// discipline as [`ModelEntry::vram_gb`].
    pub draft_vram_gb: f64,
    /// `OpenVINO` device for the draft. `None` (default) = same device as the
    /// target. v1 REJECTS any explicit value different from the target's
    /// resolved device (cross-device drafting is untested).
    #[serde(default)]
    pub draft_device: Option<String>,
    /// Candidate tokens per verification step. Default 5 — the only value
    /// empirically validated this investigation.
    #[serde(default = "default_num_assistant_tokens")]
    pub num_assistant_tokens: usize,
    /// Run the load-time greedy-equivalence self-check (Part 7 of the plan).
    /// Default true. Set false only for fast restarts of a pairing already
    /// verified on this exact box/driver/OV version.
    #[serde(default = "default_verify_on_load")]
    pub verify_on_load: bool,
}

/// The operator's per-model sampling defaults (0.9.1 V5a), nested so a typo
/// (`temprature`) is an error instead of a silently ignored key. Values a
/// client sends always win; see `model_manager::sampling::SamplingProfile` for
/// the precedence. Validated by [`validate_model_entry`] (ranges) and
/// [`validate_sampling_against_file`] (the `do_sample` rule). Stored as `f64`
/// like the other numeric settings, so a value is written back to `config.json`
/// exactly as the operator typed it (an `f32` widens to `0.15000000596046448`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingConfig {
    /// `[0, 2]`; `0` = greedy (then `top_p`/`top_k` are rejected).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// `(0, 1]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// At least 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    /// `[1.0, 2.0]`; `1.0` is an explicit opt-out of the server's 1.1 net.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    /// `[-2, 2]`. Not forwarded on NPU.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// `[-2, 2]`. Not forwarded on NPU.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
}

/// Range checks of a model's `sampling` object (no disk access).
fn validate_sampling(model_id: &str, s: &SamplingConfig) -> anyhow::Result<()> {
    let at = |f: &str| format!("models['{model_id}'].sampling.{f}");
    if let Some(t) = s.temperature {
        anyhow::ensure!(
            (0.0..=2.0).contains(&t),
            "{} must be between 0 and 2, got {t}",
            at("temperature")
        );
    }
    if let Some(p) = s.top_p {
        anyhow::ensure!(
            p > 0.0 && p <= 1.0,
            "{} must be greater than 0 and at most 1, got {p}",
            at("top_p")
        );
    }
    if let Some(k) = s.top_k {
        anyhow::ensure!(
            k >= 1,
            "{} must be at least 1, got {k} (omit it for no top-k cut)",
            at("top_k")
        );
    }
    if s.temperature.is_some_and(|t| t <= 0.0) {
        anyhow::ensure!(
            s.top_p.is_none() && s.top_k.is_none(),
            "{} is 0 (greedy): top_p/top_k cannot be combined with it — remove them, \
             or set a temperature above 0",
            at("temperature")
        );
    }
    if let Some(r) = s.repetition_penalty {
        anyhow::ensure!(
            (1.0..=2.0).contains(&r),
            "{} must be between 1.0 and 2.0 (1.0 turns the server's 1.1 default off), got {r}",
            at("repetition_penalty")
        );
    }
    for (name, v) in [
        ("presence_penalty", s.presence_penalty),
        ("frequency_penalty", s.frequency_penalty),
    ] {
        if let Some(v) = v {
            anyhow::ensure!(
                (-2.0..=2.0).contains(&v),
                "{} must be between -2 and 2, got {v}",
                at(name)
            );
        }
    }
    Ok(())
}

/// The `sampling` rule that needs the model's own `generation_config.json`
/// (blocking file I/O; call off the async runtime): an operator `top_p` /
/// `top_k` without a `temperature` is rejected unless the file says
/// `do_sample: true` with a valid temperature — otherwise the engine would
/// sample at its own T 1.0 (see `FileSampling::samples_with_temperature`).
/// A missing file counts as no `do_sample`. The file is read only when the
/// rule can fire (see [`check_sampling_do_sample`]).
///
/// # Errors
/// The rule above is violated.
pub(crate) fn validate_sampling_against_file(
    model_id: &str,
    entry: &ModelEntry,
    model_dir: &Path,
) -> anyhow::Result<()> {
    if !sampling_needs_do_sample(entry) {
        return Ok(());
    }
    let file = super::sampling::load_file_sampling(model_dir);
    check_sampling_do_sample(model_id, entry, file.samples_with_temperature())
}

/// `entry`'s `sampling` sets `top_p`/`top_k` without a `temperature`.
fn sampling_needs_do_sample(entry: &ModelEntry) -> bool {
    entry
        .policy
        .sampling
        .is_some_and(|s| s.temperature.is_none() && (s.top_p.is_some() || s.top_k.is_some()))
}

/// [`validate_sampling_against_file`] with the file's answer already known (a
/// registered model's record holds its file layer): `do_sample` is
/// `FileSampling::samples_with_temperature`.
///
/// # Errors
/// `entry`'s `sampling` sets `top_p`/`top_k` without a temperature and
/// `do_sample` is `false`.
pub(crate) fn check_sampling_do_sample(
    model_id: &str,
    entry: &ModelEntry,
    do_sample: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        do_sample || !sampling_needs_do_sample(entry),
        "models['{model_id}'].sampling sets top_p/top_k without a temperature, and the model's \
         generation_config.json has no do_sample: true with a valid temperature — the engine \
         would sample at its own default temperature 1.0. Set sampling.temperature, or remove \
         top_p/top_k"
    );
    Ok(())
}

fn default_num_assistant_tokens() -> usize {
    5
}

fn default_verify_on_load() -> bool {
    true
}

/// All per-model configuration in one stanza — the single source of truth
/// for a model ID.
///
/// Replaces the old three-map layout (`vram_gb`, `model_kinds`,
/// `model_policies`): every property that targets a specific model ID now
/// lives in this struct. [`ModelPolicy`] fields are flattened in, so the
/// JSON stanza is flat:
/// ```json
/// "gpt-oss-20b-int4-ov": { "vram_gb": 12.0, "reasoning_parser": "gpt_oss" }
/// ```
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ModelEntry {
    /// VRAM estimate in GB. Required for every registered model.
    /// Use `0.0` to register a model without VRAM gating (CPU-only or
    /// unmetered).
    pub vram_gb: f64,

    /// Model kind hint. Valid values: `"embedding"`, `"text_gen"`,
    /// `"vision"`, `"stt"`, `"tts"`, `"image_gen"`, `"reranking"`, `"decision"`.
    /// When absent, the manager auto-detects from the model directory layout;
    /// explicit values win. An embedding model MUST declare `"kind":
    /// "embedding"` — it ships the same file layout as a plain LLM.
    #[serde(default)]
    pub kind: Option<String>,

    /// Co-residency and inference policy. Flattened at the JSON level, so
    /// `"pinned": true` sits directly in the model stanza.
    #[serde(flatten)]
    pub policy: ModelPolicy,
}

impl Default for ModelEntry {
    fn default() -> Self {
        Self {
            vram_gb: 0.0,
            kind: None,
            policy: ModelPolicy::default(),
        }
    }
}

/// The JSON keys of a model stanza whose value in `file` differs from `current`, sorted (C8: what
/// a config reload did not apply to a loaded model). Compared as the serialised stanza, so a field
/// added to [`ModelEntry`] or [`ModelPolicy`] is covered without listing it here; a key absent on
/// one side counts as `null`.
pub(crate) fn model_entry_changes(current: &ModelEntry, file: &ModelEntry) -> Vec<String> {
    let as_map = |e: &ModelEntry| match serde_json::to_value(e) {
        Ok(serde_json::Value::Object(m)) => m,
        _ => serde_json::Map::new(),
    };
    let (a, b) = (as_map(current), as_map(file));
    let null = serde_json::Value::Null;
    let mut keys: Vec<String> = a
        .keys()
        .chain(b.keys())
        .filter(|k| a.get(*k).unwrap_or(&null) != b.get(*k).unwrap_or(&null))
        .cloned()
        .collect();
    keys.sort();
    keys.dedup();
    keys
}

/// Deserialize helper for a tri-state optional field: JSON key absent ⇒
/// `#[serde(default)]` gives `None` ("leave unchanged"); key present as
/// `null` ⇒ `Some(None)` ("clear the override"); key present with a value ⇒
/// `Some(Some(v))` ("set the override"). Apply as
/// `#[serde(default, deserialize_with = "deserialize_some")]` on an
/// `Option<Option<T>>` field.
fn deserialize_some<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    T: Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    Deserialize::deserialize(deserializer).map(Some)
}

/// Which of a [`ModelEntryPatch`]'s supplied fields landed where, so the
/// caller can report an auditable summary instead of a silent mutation.
#[derive(Debug, Default)]
pub struct PatchOutcome {
    /// Field names whose new live-record or dynamic-overlay value is already
    /// in effect for the *next* request against this model (or immediately,
    /// for a `NotLoaded` model with no engine to wait for).
    pub applied_live: Vec<&'static str>,
    /// Field names persisted to `config.json` but which only change engine
    /// behaviour at this model's *next* load (`vram_gb`, `kv_cache_gb`,
    /// `max_concurrent_streams`, `max_prompt_len`, `speculative`) — for
    /// `vram_gb` specifically, the admin-listing display value
    /// (`ModelRecord::vram_gb`) updates immediately, but the VRAM tracker's
    /// actual reservation (charged by model id at load time) does not; the
    /// running engine, if any, is unaffected by any of these until its next
    /// load either way.
    pub effective_on_next_load: Vec<&'static str>,
    /// Field names supplied whose value already matched the current
    /// effective entry — accepted, persisted (a no-op write), but nothing
    /// changed.
    pub unchanged: Vec<&'static str>,
}

/// Partial update to an already-registered model's [`ModelEntry`], for
/// `PATCH /v1/admin/models/{id}`. Every field is optional — only supplied
/// fields change; the rest keep their current effective value. No
/// `#[serde(flatten)]` and no reuse of [`ModelPolicy`] here (unlike
/// [`ModelEntry`]) — a flattened partial-update body would silently accept
/// a typo'd field name (the exact `#[serde(deny_unknown_fields)]` gap this
/// struct exists to close), and every nullable field needs
/// [`deserialize_some`]'s three-way "absent / null / value" distinction,
/// which `ModelPolicy` itself does not carry.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelEntryPatch {
    /// Present only so it can be rejected explicitly (400) — `kind` is
    /// immutable via PATCH. Changing a model's kind changes which engine
    /// factory builds it; that needs a fresh registration
    /// (deregister + `POST /v1/admin/models/add`), not a field tweak.
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub vram_gb: Option<f64>,
    #[serde(default)]
    pub pinned: Option<bool>,
    #[serde(default)]
    pub priority: Option<i32>,
    #[serde(default, deserialize_with = "deserialize_some")]
    pub kv_cache_gb: Option<Option<f64>>,
    #[serde(default, deserialize_with = "deserialize_some")]
    pub max_concurrent_streams: Option<Option<usize>>,
    #[serde(default, deserialize_with = "deserialize_some")]
    pub enable_prefix_caching: Option<Option<bool>>,
    #[serde(default, deserialize_with = "deserialize_some")]
    pub kv_cache_precision: Option<Option<String>>,
    #[serde(default)]
    pub load: Option<LoadPolicy>,
    #[serde(default)]
    pub evictable: Option<bool>,
    /// Placement-affecting — rejected (409, by the caller) while the model
    /// is `Ready`; only takes effect while `NotLoaded`.
    #[serde(default, deserialize_with = "deserialize_some")]
    pub device: Option<Option<String>>,
    /// Placement-affecting, same 409-while-`Ready` rule as `device`. Whole-
    /// list replace when supplied; `[]` clears it back to the built-in
    /// per-kind default.
    #[serde(default)]
    pub tier_preference: Option<Vec<String>>,
    #[serde(default, deserialize_with = "deserialize_some")]
    pub reasoning_parser: Option<Option<ReasoningParser>>,
    #[serde(default, deserialize_with = "deserialize_some")]
    pub max_prompt_len: Option<Option<u32>>,
    #[serde(default, deserialize_with = "deserialize_some")]
    pub min_response_len: Option<Option<u32>>,
    #[serde(default, deserialize_with = "deserialize_some")]
    pub speculative: Option<Option<SpeculativeConfig>>,
    #[serde(default, deserialize_with = "deserialize_some")]
    pub precision: Option<Option<String>>,
    #[serde(default, deserialize_with = "deserialize_some")]
    pub model_source: Option<Option<String>>,
    #[serde(default, deserialize_with = "deserialize_some")]
    pub model_revision: Option<Option<String>>,
    /// Whole-list replace when supplied; `[]` clears it.
    #[serde(default)]
    pub capabilities: Option<Vec<String>>,
    #[serde(default, deserialize_with = "deserialize_some")]
    pub eviction_grace_secs: Option<Option<f64>>,
    /// Whole-object replace when supplied; `null` clears the operator layer.
    /// Hot: applies to the next request, a `Ready` model included.
    #[serde(default, deserialize_with = "deserialize_some")]
    pub sampling: Option<Option<SamplingConfig>>,
    /// `null` clears it (no budget). Hot: applies to the next request, a `Ready` model included.
    #[serde(default, deserialize_with = "deserialize_some")]
    pub thinking_budget_tokens: Option<Option<u32>>,
}

impl ModelEntryPatch {
    /// `true` when no field was supplied at all — the caller should reject
    /// this as a 400 ("no fields supplied") rather than silently no-op.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vram_gb.is_none()
            && self.pinned.is_none()
            && self.priority.is_none()
            && self.kv_cache_gb.is_none()
            && self.max_concurrent_streams.is_none()
            && self.enable_prefix_caching.is_none()
            && self.kv_cache_precision.is_none()
            && self.load.is_none()
            && self.evictable.is_none()
            && self.device.is_none()
            && self.tier_preference.is_none()
            && self.reasoning_parser.is_none()
            && self.max_prompt_len.is_none()
            && self.min_response_len.is_none()
            && self.speculative.is_none()
            && self.precision.is_none()
            && self.model_source.is_none()
            && self.model_revision.is_none()
            && self.capabilities.is_none()
            && self.eviction_grace_secs.is_none()
            && self.sampling.is_none()
            && self.thinking_budget_tokens.is_none()
    }

    /// `true` when `device` or `tier_preference` was supplied — the two
    /// placement-affecting fields a `Ready` model must reject (409, evict
    /// first) since they'd otherwise silently disagree with the resident
    /// engine's actual device.
    #[must_use]
    pub fn touches_placement(&self) -> bool {
        self.device.is_some() || self.tier_preference.is_some()
    }

    /// Merge `self` onto `base`, returning the new [`ModelEntry`] plus a
    /// [`PatchOutcome`] classifying every field that was actually supplied.
    /// Fields not supplied in `self` keep `base`'s value untouched.
    #[must_use]
    pub fn apply_to(&self, base: &ModelEntry) -> (ModelEntry, PatchOutcome) {
        let mut entry = base.clone();
        let mut out = PatchOutcome::default();

        macro_rules! hot {
            ($field:ident, $name:literal) => {
                if let Some(v) = self.$field {
                    if entry.policy.$field == v {
                        out.unchanged.push($name);
                    } else {
                        out.applied_live.push($name);
                    }
                    entry.policy.$field = v;
                }
            };
        }
        macro_rules! hot_opt {
            ($field:ident, $name:literal) => {
                if let Some(inner) = self.$field.clone() {
                    if entry.policy.$field == inner {
                        out.unchanged.push($name);
                    } else {
                        out.applied_live.push($name);
                    }
                    entry.policy.$field = inner;
                }
            };
        }
        macro_rules! next_load {
            ($field:ident, $name:literal) => {
                if let Some(inner) = self.$field.clone() {
                    if entry.policy.$field == inner {
                        out.unchanged.push($name);
                    } else {
                        out.effective_on_next_load.push($name);
                    }
                    entry.policy.$field = inner;
                }
            };
        }

        if let Some(v) = self.vram_gb {
            if (entry.vram_gb - v).abs() < f64::EPSILON {
                out.unchanged.push("vram_gb");
            } else {
                out.effective_on_next_load.push("vram_gb");
            }
            entry.vram_gb = v;
        }
        hot!(pinned, "pinned");
        hot!(priority, "priority");
        next_load!(kv_cache_gb, "kv_cache_gb");
        next_load!(max_concurrent_streams, "max_concurrent_streams");
        next_load!(enable_prefix_caching, "enable_prefix_caching");
        next_load!(kv_cache_precision, "kv_cache_precision");
        hot!(load, "load");
        hot!(evictable, "evictable");
        hot_opt!(device, "device");
        if let Some(v) = self.tier_preference.clone() {
            if entry.policy.tier_preference == v {
                out.unchanged.push("tier_preference");
            } else {
                out.applied_live.push("tier_preference");
            }
            entry.policy.tier_preference = v;
        }
        hot_opt!(reasoning_parser, "reasoning_parser");
        next_load!(max_prompt_len, "max_prompt_len");
        next_load!(min_response_len, "min_response_len");
        if let Some(inner) = self.speculative.clone() {
            // SpeculativeConfig has no PartialEq, so a genuine field-by-field
            // change can't be detected — but `None` vs `None` (clearing an
            // already-absent override) is still checkable without one, and
            // is the one case actually worth reporting "unchanged" for
            // (clearing a real config would need a `PartialEq` bound to
            // prove no-op, so that direction still reports "applied").
            if inner.is_none() && entry.policy.speculative.is_none() {
                out.unchanged.push("speculative");
            } else {
                out.effective_on_next_load.push("speculative");
            }
            entry.policy.speculative = inner;
        }
        hot_opt!(precision, "precision");
        hot_opt!(model_source, "model_source");
        hot_opt!(model_revision, "model_revision");
        if let Some(v) = self.capabilities.clone() {
            if entry.policy.capabilities == v {
                out.unchanged.push("capabilities");
            } else {
                out.applied_live.push("capabilities");
            }
            entry.policy.capabilities = v;
        }
        hot_opt!(eviction_grace_secs, "eviction_grace_secs");
        hot_opt!(sampling, "sampling");
        hot_opt!(thinking_budget_tokens, "thinking_budget_tokens");

        (entry, out)
    }
}

/// True when `id` is a single plain path component, so `models_dir.join(id)` stays inside
/// `models_dir`: non-empty, not `.` or `..`, no separators, no NUL, and no Windows drive prefix
/// (`C:`, `c:foo`). The drive check is explicit so it holds on every platform: on Linux `C:` is an
/// ordinary name, but a config written there may be served on Windows, where `join("C:")` replaces
/// the base. A `:` elsewhere (`qwen:7b`) stays allowed. Used for model ids and
/// `speculative.draft_model` (FM-7).
pub(crate) fn is_plain_model_id(id: &str) -> bool {
    let b = id.as_bytes();
    let drive_prefix = b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':';
    !id.is_empty()
        && id != "."
        && id != ".."
        && !drive_prefix
        && !id.contains(['/', '\\', '\0'])
        && matches!(
            std::path::Path::new(id)
                .components()
                .collect::<Vec<_>>()
                .as_slice(),
            [std::path::Component::Normal(_)]
        )
}

/// `0` (no deadline) or at least [`MIN_LOAD_TIMEOUT_SECS`].
fn validate_load_timeout(field: &str, secs: u64) -> anyhow::Result<()> {
    anyhow::ensure!(
        secs == 0 || secs >= MIN_LOAD_TIMEOUT_SECS,
        "{field} must be 0 (no deadline) or at least {MIN_LOAD_TIMEOUT_SECS} (got {secs})"
    );
    Ok(())
}

/// Reads a JSON file as text, dropping a leading UTF-8 byte-order mark.
/// Windows PowerShell 5.1 (`Set-Content -Encoding utf8`) and Notepad write one,
/// and `serde_json` rejects it with a bare "expected value at line 1 column 1"
/// (Windows pass 2, D12). Used for every JSON file the operator edits by hand:
/// the config and the keys file.
///
/// # Errors
///
/// The I/O error from reading the file, or `InvalidData` if it is not UTF-8.
pub fn read_json_text(path: &Path) -> std::io::Result<String> {
    let text = std::fs::read_to_string(path)?;
    Ok(match text.strip_prefix('\u{feff}') {
        Some(rest) => rest.to_owned(),
        None => text,
    })
}

/// The JSON stanza to persist for `merged` under `models.<id>`: its non-default
/// fields plus any field the file's existing stanza (`old`) already spelled
/// out, and always `vram_gb`. Serialising the whole entry wrote every default
/// (`kind: null`, `pinned: false`, ...) into a hand-written stanza on the first
/// `rv patch` (Windows pass 2, D13); a field at its default that the operator
/// never wrote stays unwritten, and one they did write keeps its place.
///
/// # Errors
///
/// A serialisation failure (not expected for a plain data struct).
pub fn minimal_stanza(
    old: Option<&serde_json::Value>,
    merged: &ModelEntry,
) -> serde_json::Result<serde_json::Value> {
    let full = serde_json::to_value(merged)?;
    let defaults = serde_json::to_value(ModelEntry {
        vram_gb: merged.vram_gb,
        ..ModelEntry::default()
    })?;
    let (Some(full_map), Some(default_map)) = (full.as_object(), defaults.as_object()) else {
        return Ok(full);
    };
    // Start from the stanza as written: its key ORDER and any key this build
    // does not model stay exactly where they are; only changed or new fields
    // are touched (`Map::insert` keeps an existing key's position).
    let mut out = old
        .and_then(serde_json::Value::as_object)
        .cloned()
        .unwrap_or_default();
    for (k, v) in full_map {
        if k == "vram_gb" || out.contains_key(k) || default_map.get(k) != Some(v) {
            out.insert(k.clone(), v.clone());
        }
    }
    Ok(serde_json::Value::Object(out))
}

/// Default for [`Config::npu_footprint_factor`]: the measured adapter/reserved
/// ratio on the Windows pass-2 box (8.4 / 5.5 = 1.5).
pub(crate) fn default_npu_footprint_factor() -> f64 {
    1.5
}

/// True for an image-generation model on a CPU device: the case
/// [`Config::cpu_image_gen_max_pixels`] charges and caps (one rule for both).
#[must_use]
pub fn is_cpu_image_gen(device: &str, kind: super::engine::ModelKind) -> bool {
    device.starts_with("CPU") && kind == super::engine::ModelKind::ImageGen
}

/// Default for [`Config::cpu_image_gen_max_pixels`]: one 1024x1024 image.
pub(crate) fn default_cpu_image_gen_max_pixels() -> u64 {
    1024 * 1024
}

/// Default for [`Config::max_audio_duration_secs`]: two hours of audio.
pub(crate) fn default_max_audio_duration_secs() -> u64 {
    2 * 60 * 60
}

/// Default for [`Config::max_images_per_request`].
pub(crate) fn default_max_images_per_request() -> usize {
    16
}

/// Default for [`Config::max_image_pixels_per_request`]: 64 MP.
pub(crate) fn default_max_image_pixels_per_request() -> u64 {
    64 * 1024 * 1024
}

/// Default for [`Config::max_image_pixels`]: 4 MP (decimal — the measured "4 MP" step).
pub(crate) fn default_max_image_pixels() -> u64 {
    4_000_000
}

fn default_vram_safety_margin() -> f64 {
    1.0 // 1 GB — observed headroom for the OV/OpenCL runtime on Battlemage
}

fn default_min_kv_cache() -> f64 {
    1.0 // refuse a load that would leave less than 1 GB for the KV pool
}

fn default_light_model_max_gb() -> f64 {
    2.0 // sub-2B int4: measured interactive (>20 tok/s) on a weak iGPU
}

fn default_enable_prefix_caching() -> bool {
    true // server scenario: repeated system prompts + resent multi-turn history
}

fn default_light_stt_max_gb() -> f64 {
    // whisper-base/small/medium (≤~0.77 GB) light; whisper-large (~1.5 GB) heavy.
    // Distinct from the 2.0 LLM threshold, which would misclassify large STT.
    1.0
}

fn default_dgpu_size_ceiling_fraction() -> f64 {
    0.8
}

fn default_max_num_seqs() -> usize {
    // 16, not 8: leaves ~2× headroom over a typical ~8-concurrent workload so
    // pipelined-client refire bursts do not trip spurious 429s (see the doc
    // comment on `max_num_seqs` and the project's internal engineering log).
    16
}

fn default_cors_origins() -> Vec<String> {
    vec!["*".to_owned()]
}

fn default_admission_queue_timeout_ms() -> u64 {
    5_000 // 5 seconds — absorbs refire bursts, still fails fast vs typical 30 s HTTP timeout
}

fn default_device_admission_queue_timeout_ms() -> u64 {
    5_000 // mirrors admission_queue_timeout_ms's default and rationale
}

fn default_embedding_pooling() -> String {
    "mean".to_owned() // matches stormVINO + the e5 family; the OV header default is CLS
}

fn default_embedding_normalize() -> bool {
    true // cosine-ready vectors, matching OpenAI's embeddings contract
}

/// Default cap on the `/v1/completions` prompt-array fan-out. Also used by
/// the handler when no config is loaded (mock mode).
pub(crate) fn default_max_embedding_inputs() -> usize {
    256
}

pub(crate) fn default_max_embedding_batch_tokens() -> usize {
    32_768
}

pub(crate) fn default_max_prompt_array() -> usize {
    16 // one engine's worth of slots — a sane per-request fan-out ceiling
}

/// Default server-wide `max_tokens` ceiling. Also used by the handlers when
/// no config is loaded (mock mode). ~2.3 minutes of generation at 60 tok/s —
/// generous for real outputs, fatal to slot-pinning abuse.
pub(crate) fn default_max_tokens_cap() -> usize {
    8192
}

/// Default for [`Config::stream_first_token_timeout_secs`]: 600 s (S1): about
/// 120k prompt tokens at the 200 tok/s prefill floor the generation watchdog assumes.
pub(crate) fn default_stream_first_token_timeout_secs() -> u64 {
    600
}

/// Default for [`Config::stream_token_gap_timeout_secs`]: 120 s (S1).
pub(crate) fn default_stream_token_gap_timeout_secs() -> u64 {
    120
}

/// Default HTTP bind address: loopback (secure by default, T4.2).
fn default_bind_addr() -> String {
    "127.0.0.1".to_owned()
}

/// Default HTTP port (dev port; stormVINO owns 11435 until cutover).
fn default_port() -> u16 {
    11_437
}

impl Config {
    /// Setting names deliberately left out of [`default_template`](Self::default_template):
    /// the two deprecated inline key lists (a non-empty value is a startup
    /// error; keys live in the keys file) and the catch-all for unknown keys.
    pub const UNTEMPLATED: [&str; 3] = ["api_keys", "admin_api_keys", "unknown_fields"];

    /// What admission charges for a model's weights before anything was measured: `vram_gb`,
    /// times [`Self::npu_footprint_factor`] on the NPU; for an image-generation model on the
    /// CPU, `vram_gb` plus its family's working set at [`Self::cpu_image_gen_max_pixels`]
    /// (D-06; `family` `None` is charged as the largest family). The one rule shared by
    /// placement at startup and the manager's `budget_weight_gb`.
    #[must_use]
    pub fn weight_charge_gb(
        &self,
        vram_gb: f64,
        device: &str,
        kind: super::engine::ModelKind,
        family: Option<super::image_footprint::ImageFamily>,
    ) -> f64 {
        if device.starts_with("NPU") {
            vram_gb * self.npu_footprint_factor
        } else if is_cpu_image_gen(device, kind) {
            vram_gb
                + family
                    .unwrap_or(super::image_footprint::ImageFamily::Unknown)
                    .cpu_working_set_gb(self.cpu_image_gen_max_pixels)
        } else {
            vram_gb
        }
    }

    /// Every global setting an operator can write in `config.json`, with the
    /// value the server uses when it is absent, in `Config` field order. `models_dir`, `device` and `total_vram_gb` have no default:
    /// they carry placeholders. This is what `--print-default-config` prints
    /// and what the shipped configs and `CONFIG.md` are checked
    /// against, so a setting added to [`Config`] cannot stay out of them.
    ///
    /// # Errors
    ///
    /// A serialisation failure (not expected for these plain data types).
    pub fn default_template() -> anyhow::Result<serde_json::Value> {
        let base: Self = serde_json::from_str(
            r#"{"models_dir":"/path/to/models","device":"GPU","total_vram_gb":0.0}"#,
        )?;
        let mut value = serde_json::to_value(&base)?;
        if let Some(map) = value.as_object_mut() {
            for k in Self::UNTEMPLATED {
                map.shift_remove(k);
            }
        }
        Ok(value)
    }

    /// One model stanza with every per-model setting at its default (the
    /// `models.<id>` object): `vram_gb` and `kind` first, then the policy
    /// fields. What `--print-default-model` prints.
    ///
    /// # Errors
    ///
    /// A serialisation failure (not expected for these plain data types).
    pub fn default_model_template() -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(ModelEntry::default())?)
    }

    /// Load and deserialise configuration from a JSON file on disk.
    ///
    /// # Errors
    /// - File cannot be read (missing, permissions, I/O).
    /// - File is not valid JSON or is missing a required field.
    /// - An unknown key looks like a typo of a security-critical key
    ///   (see [`check_unknown_fields`](Self::check_unknown_fields)).
    /// - A numeric VRAM/cache knob is non-finite, negative, or the margins are
    ///   inconsistent (see [`validate`](Self::validate)).
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = read_json_text(path)
            .map_err(|e| anyhow::anyhow!("cannot read config {}: {e}", path.display()))?;
        let config: Self = serde_json::from_str(&content)
            .map_err(|e| anyhow::anyhow!("invalid config {}: {e}", path.display()))?;
        config.check_unknown_fields()?;
        config.validate()?;
        Ok(config)
    }

    /// T4.4 (committee finding 33): reject semantically-invalid numeric config
    /// at load time, before it can poison VRAM gating.
    ///
    /// serde accepts any well-typed `f64`, including `NaN` (JSON has no NaN
    /// literal, but `1e400` overflows to infinity and a crafted value can be
    /// non-finite) and negatives. Every VRAM comparison is `<`/`>`, and **every
    /// comparison against `NaN` is false** — so a `NaN` `total_vram_gb` makes the
    /// "does it fit?" gate silently pass for everything, and a negative margin
    /// inflates the usable pool. Both defeat the capacity gate. Catch them here.
    ///
    /// # Errors
    /// - Any VRAM/cache knob is non-finite (`NaN`/`±inf`) or negative.
    /// - The reserved margins (`vram_safety_margin_gb + min_kv_cache_gb`) exceed
    ///   `total_vram_gb` when gating is on — no model could ever load.
    #[allow(clippy::too_many_lines)]
    pub fn validate(&self) -> anyhow::Result<()> {
        normalize_kv_precision(&self.kv_cache_precision)?;
        validate_load_timeout("load_timeout_secs", self.load_timeout_secs)?;
        validate_on_demand_wait(self.on_demand_wait_secs, self.load_timeout_secs)?;
        anyhow::ensure!(
            matches!(self.ov_cache_layout.as_str(), "keyed" | "flat"),
            "ov_cache_layout {:?} is not supported; accepted values: \"keyed\", \"flat\"",
            self.ov_cache_layout
        );
        anyhow::ensure!(
            matches!(self.kv_headroom_mode.as_str(), "dynamic" | "off"),
            "kv_headroom_mode {:?} is not supported; accepted values: \"dynamic\", \"off\"",
            self.kv_headroom_mode
        );
        anyhow::ensure!(
            self.kv_transient_factor.is_finite() && self.kv_transient_factor >= 1.0,
            "kv_transient_factor must be a finite number of at least 1 (got {})",
            self.kv_transient_factor
        );
        anyhow::ensure!(
            self.kv_headroom_margin_gib.is_finite() && self.kv_headroom_margin_gib >= 0.0,
            "kv_headroom_margin_gib must be a finite number of at least 0 (got {})",
            self.kv_headroom_margin_gib
        );
        anyhow::ensure!(
            self.kv_headroom_min_tokens >= 1,
            "kv_headroom_min_tokens must be at least 1"
        );
        anyhow::ensure!(
            matches!(self.kv_gate_mode.as_str(), "exact" | "legacy"),
            "kv_gate_mode {:?} is not supported; accepted values: \"exact\", \"legacy\"",
            self.kv_gate_mode
        );
        anyhow::ensure!(
            matches!(self.kv_precision_pass.as_str(), "explicit" | "plugin"),
            "kv_precision_pass {:?} is not supported; accepted values: \"explicit\", \"plugin\"",
            self.kv_precision_pass
        );
        anyhow::ensure!(
            self.max_embedding_inputs >= 1,
            "max_embedding_inputs must be at least 1 (0 would reject every /v1/embeddings request)"
        );
        anyhow::ensure!(
            self.max_prompt_array >= 1,
            "max_prompt_array must be at least 1 (0 would reject every /v1/completions request)"
        );
        self.validate_intervals()?;
        let knobs: [(&str, f64); 7] = [
            ("total_vram_gb", self.total_vram_gb),
            ("cache_size_gb", self.cache_size_gb),
            ("default_kv_cache_gb", self.default_kv_cache_gb),
            ("vram_safety_margin_gb", self.vram_safety_margin_gb),
            ("min_kv_cache_gb", self.min_kv_cache_gb),
            ("ov_cache_max_gb", self.ov_cache_max_gb),
            ("kv_pressure_threshold_pct", self.kv_pressure_threshold_pct),
        ];
        for (name, v) in knobs {
            anyhow::ensure!(
                v.is_finite(),
                "config {name} must be a finite number, got {v}"
            );
            anyhow::ensure!(v >= 0.0, "config {name} must be non-negative, got {v}");
        }
        anyhow::ensure!(
            self.npu_footprint_factor.is_finite() && self.npu_footprint_factor >= 1.0,
            "config npu_footprint_factor must be a finite number >= 1.0, got {}",
            self.npu_footprint_factor
        );
        anyhow::ensure!(
            self.cpu_image_gen_max_pixels >= 256 * 256,
            "config cpu_image_gen_max_pixels must be at least 65536 (256x256), got {}",
            self.cpu_image_gen_max_pixels
        );
        anyhow::ensure!(
            self.max_images_per_request >= 1,
            "config max_images_per_request must be at least 1, got {}",
            self.max_images_per_request
        );
        anyhow::ensure!(
            self.max_image_pixels_per_request >= 256 * 256,
            "config max_image_pixels_per_request must be at least 65536 (256x256), got {}",
            self.max_image_pixels_per_request
        );
        anyhow::ensure!(
            self.max_image_pixels >= 256 * 256,
            "config max_image_pixels must be at least 65536 (256x256), got {}",
            self.max_image_pixels
        );
        anyhow::ensure!(
            self.max_audio_duration_secs >= 1,
            "config max_audio_duration_secs must be at least 1, got {}",
            self.max_audio_duration_secs
        );
        anyhow::ensure!(
            self.kv_pressure_threshold_pct <= 100.0,
            "config kv_pressure_threshold_pct must be a percentage (0-100), got {}",
            self.kv_pressure_threshold_pct
        );
        // kv_pressure_monitor_enabled has no usable numeric default on
        // purpose (the project's internal engineering log's ops-review
        // section) — a box that turns monitoring on must state its own real
        // threshold/duration, not silently inherit whatever this codebase
        // happens to ship as a "default." `0.0`/`0` are the "not configured"
        // sentinels, so catch them here rather than let the monitor start
        // with a threshold of 0% (which would flag every model, instantly).
        if self.kv_pressure_monitor_enabled {
            anyhow::ensure!(
                self.kv_pressure_threshold_pct > 0.0,
                "config kv_pressure_threshold_pct must be set (>0) when \
                 kv_pressure_monitor_enabled is true — no default is provided deliberately: \
                 the right threshold depends on this box's models and traffic"
            );
            anyhow::ensure!(
                self.kv_pressure_sustained_secs > 0,
                "config kv_pressure_sustained_secs must be set (>0) when \
                 kv_pressure_monitor_enabled is true — no default is provided deliberately: \
                 the right threshold depends on this box's models and traffic"
            );
        }
        // Per-model knob validation, shared with `ModelManager::add_model` (the
        // runtime registration path) via `validate_model_entry` so both ways of
        // registering a model enforce identical invariants.
        for (model, entry) in &self.models {
            // P-C (firmness leftover): a `models` id names a folder inside `models_dir`, as
            // `models/add` and `speculative.draft_model` already require; `../x`, `a/b`, an
            // absolute path or a drive letter would read a model from outside it. Refused here
            // (boot: exit 78; config reload: 422), not skipped.
            anyhow::ensure!(
                is_plain_model_id(model),
                "models['{model}']: the model id must be the name of a folder directly inside \
                 models_dir (no '/', '\\', '..' or drive letter)"
            );
            validate_model_entry(model, entry)?;
            // V5a: the `do_sample` rule reads the model's own file, which can change on disk
            // without the operator touching config.json (a re-export), so here (boot, reload,
            // the whole-file check before a config mutation) it only WARNs; it is a hard 400
            // where the operator writes `sampling` (add, a PATCH setting it). A model whose
            // directory is missing is skipped at registration, so it is skipped here too.
            let model_dir = self.models_dir.join(model);
            if model_dir.is_dir()
                && let Err(e) = validate_sampling_against_file(model, entry, &model_dir)
            {
                tracing::warn!(
                    model = %model,
                    error = %e,
                    "sampling: top_p/top_k are set without a temperature and the model's \
                     generation_config.json does not turn sampling on — this model samples at \
                     the engine's default temperature 1.0 when a request sends no temperature"
                );
            }
            // Speculative "resident twice" warning needs the full models map,
            // so it stays here rather than in the shared per-entry validator.
            if let Some(spec) = &entry.policy.speculative
                && self.models.contains_key(&spec.draft_model)
            {
                tracing::warn!(
                    model = %model,
                    draft_model = %spec.draft_model,
                    "speculative.draft_model is also a registered models entry — \
                     its weights will be resident twice (once as the draft, folded \
                     into this target's reservation, once as its own independently \
                     loadable/evictable entry)",
                );
            }
        }
        // Memory-domain knobs: same finite + non-negative discipline as the VRAM
        // knobs above — a NaN reservation or a negative domain budget would
        // silently defeat the per-domain capacity gate (every NaN comparison is
        // false; a negative inflates the pool).
        if let Some(r) = self.system_ram_reservation_gb {
            anyhow::ensure!(
                r.is_finite() && r >= 0.0,
                "config system_ram_reservation_gb must be a finite non-negative number, got {r}"
            );
        }
        if let Some(b) = self.system_ram_budget_gb {
            anyhow::ensure!(
                b.is_finite() && b > 0.0,
                "config system_ram_budget_gb must be a finite positive number when set \
                 (omit it entirely to keep the default MemAvailable-only gate), got {b}"
            );
        }
        for (domain, &v) in &self.domain_budgets {
            anyhow::ensure!(
                v.is_finite() && v >= 0.0,
                "config domain_budgets['{domain}'] must be a finite non-negative number, got {v}"
            );
        }
        for (device, &cap) in &self.device_budgets {
            anyhow::ensure!(
                cap >= 1,
                "config device_budgets['{device}'] must be at least 1 — a 0 cap \
                 blocks the device entirely; omit the entry instead to disable \
                 gating for it"
            );
        }
        // Realtime voice arbitration v2 (D3): same finite/non-negative
        // discipline as every other VRAM/RAM knob above.
        anyhow::ensure!(
            self.eviction_grace_secs.is_finite() && self.eviction_grace_secs >= 0.0,
            "config eviction_grace_secs must be a finite non-negative number, got {}",
            self.eviction_grace_secs
        );
        // Realtime voice arbitration v2 (D4/D7): fail loud on an unrecognised
        // capability label in the viable-minimum floor, same reasoning as the
        // per-model `capabilities` check in `validate_model_entry`.
        if let Some(vm) = &self.realtime_viable_minimum
            && let Some(llm) = &vm.llm
        {
            for cap in &llm.require_capabilities {
                anyhow::ensure!(
                    KNOWN_CAPABILITIES.contains(&cap.as_str()),
                    "realtime_viable_minimum.llm.require_capabilities contains unknown \
                     capability '{cap}' — known: {}",
                    KNOWN_CAPABILITIES.join(", ")
                );
            }
        }
        // T5.5 (#34): every preload model MUST have a `models` entry. Without
        // one the registry used to register it at a silent 0.0 GB — never counted
        // against capacity and never an eviction candidate by weight, so it loaded
        // unconditionally and drove real over-subscription / OOM. Fail loud at
        // startup instead (the config doc already states this invariant).
        for model in &self.preload {
            anyhow::ensure!(
                self.models.contains_key(model),
                "config preload['{model}'] has no models entry — every preload model must \
                 appear in models (a silent 0.0 default loads uncounted and risks OOM)"
            );
        }
        // Margins consistent: when gating is on (total_vram_gb > 0), the reserved
        // safety margin plus the minimum KV floor must leave room for model
        // weights — otherwise the very first load is refused and the server is
        // a no-op. (Both knobs are already proven finite + non-negative above.)
        if self.total_vram_gb > 0.0 {
            let reserved = self.vram_safety_margin_gb + self.min_kv_cache_gb;
            anyhow::ensure!(
                reserved <= self.total_vram_gb,
                "config margins inconsistent: vram_safety_margin_gb ({}) + min_kv_cache_gb ({}) \
                 = {reserved} exceeds total_vram_gb ({}) — no model could ever load",
                self.vram_safety_margin_gb,
                self.min_kv_cache_gb,
                self.total_vram_gb
            );
        }
        Ok(())
    }

    /// FM-4: the period and count knobs whose `0` would panic, spin or give up at once, and an
    /// upper bound on every seconds field of the `supervisor` block and both sweep intervals.
    /// Nothing here is "0 = off": the knobs that do mean off at 0 (`watchdog_hang_ceiling_secs`,
    /// `gpu_first_load_deadline_secs`, `gpu_probe_deadline_secs`) stay allowed at 0.
    ///
    /// - `supervisor.watchdog_poll_secs: 0` — tokio's `interval` panics on a zero period.
    /// - `supervisor.max_restarts: 0` — "gave up after 0 restarts" before the first worker runs.
    /// - `supervisor.health_timeout_secs: 0` — every new worker is treated as hung at once.
    /// - `supervisor.restart_window_secs: 0` — every restart ages out immediately, so the cap
    ///   never trips and a crash loop never ends.
    /// - `ov_cache_sweep_interval_secs: 0` / `kv_pressure_sweep_interval_secs: 0` — the sweep
    ///   re-runs back to back for the life of the process (a directory walk with hashing, in the
    ///   cache sweep's case).
    /// - Upper bound [`MAX_INTERVAL_SECS`] (one year): the supervisor adds several of these
    ///   seconds fields and adds them to `Instant`s, which overflow (wrap in release, panic in a
    ///   debug build) for values near `u64::MAX`.
    fn validate_intervals(&self) -> anyhow::Result<()> {
        let sup = &self.supervisor;
        let at_least_one: [(&str, u64, &str); 6] = [
            (
                "supervisor.watchdog_poll_secs",
                sup.watchdog_poll_secs,
                "a zero poll period crashes the supervisor; to turn the generation watchdog off \
                 set supervisor.watchdog_hang_ceiling_secs to 0",
            ),
            (
                "supervisor.max_restarts",
                u64::try_from(sup.max_restarts).unwrap_or(u64::MAX),
                "0 gives up before the first worker runs",
            ),
            (
                "supervisor.health_timeout_secs",
                sup.health_timeout_secs,
                "0 treats every new worker as hung at once",
            ),
            (
                "supervisor.restart_window_secs",
                sup.restart_window_secs,
                "0 makes the restart cap never trip",
            ),
            (
                "ov_cache_sweep_interval_secs",
                self.ov_cache_sweep_interval_secs,
                "0 re-runs the cache sweep back to back; it is not \"off\"",
            ),
            (
                "kv_pressure_sweep_interval_secs",
                self.kv_pressure_sweep_interval_secs,
                "0 re-runs the pressure sweep back to back; to turn it off set \
                 kv_pressure_monitor_enabled to false",
            ),
        ];
        for (name, v, why) in at_least_one {
            anyhow::ensure!(v >= 1, "config {name} must be at least 1 ({why})");
        }
        let bounded: [(&str, u64); 11] = [
            ("supervisor.backoff_base_secs", sup.backoff_base_secs),
            ("supervisor.backoff_max_secs", sup.backoff_max_secs),
            ("supervisor.restart_window_secs", sup.restart_window_secs),
            ("supervisor.health_timeout_secs", sup.health_timeout_secs),
            ("supervisor.hang_kill_grace_secs", sup.hang_kill_grace_secs),
            ("supervisor.watchdog_poll_secs", sup.watchdog_poll_secs),
            (
                "supervisor.watchdog_hang_ceiling_secs",
                sup.watchdog_hang_ceiling_secs,
            ),
            (
                "supervisor.gpu_first_load_deadline_secs",
                sup.gpu_first_load_deadline_secs,
            ),
            (
                "supervisor.gpu_probe_deadline_secs",
                sup.gpu_probe_deadline_secs,
            ),
            (
                "ov_cache_sweep_interval_secs",
                self.ov_cache_sweep_interval_secs,
            ),
            (
                "kv_pressure_sweep_interval_secs",
                self.kv_pressure_sweep_interval_secs,
            ),
        ];
        for (name, v) in bounded {
            anyhow::ensure!(
                v <= MAX_INTERVAL_SECS,
                "config {name} must be at most {MAX_INTERVAL_SECS} (one year), got {v}"
            );
        }
        Ok(())
    }

    /// T4.1: surface every unknown top-level config key.
    ///
    /// A typo'd key is silently dropped by serde, and for security keys the
    /// permissive default then takes over (`api_keys: []` = auth OFF, CORS
    /// `*`) — the server boots wide open while logging a "normal" policy.
    /// Therefore: a security-adjacent unknown key (anything mentioning
    /// key/cors/auth/bind/insecure) is a **hard error**; every other unknown
    /// key logs a WARN naming it.
    ///
    /// # Errors
    /// When an unknown key looks security-related.
    pub fn check_unknown_fields(&self) -> anyhow::Result<()> {
        const SECURITY_HINTS: [&str; 5] = ["key", "cors", "auth", "bind", "insecure"];
        let mut keys: Vec<&str> = self.unknown_fields.keys().map(String::as_str).collect();
        keys.sort_unstable(); // deterministic order for logs and tests
        for key in keys {
            let lower = key.to_lowercase();
            if SECURITY_HINTS.iter().any(|h| lower.contains(h)) {
                anyhow::bail!(
                    "unknown config key '{key}' looks security-related (did you mean \
                     'api_keys', 'cors_allowed_origins', 'bind_addr', or \
                     'allow_insecure_public_bind'?) — refusing to start with a \
                     possibly-misconfigured security policy"
                );
            }
            tracing::warn!(key, "unknown config key — ignored (check for typos)");
        }
        Ok(())
    }

    /// Parse `bind_addr`/`port` into a [`std::net::SocketAddr`]. Pure address
    /// validation — the T4.2 public-bind-requires-auth gate used to live here
    /// too, back when `api_keys` was read directly off `Config`; now that real
    /// keys live in a separately-loaded keys file (resolved after this runs),
    /// that check is [`enforce_open_bind_gate`](Self::enforce_open_bind_gate),
    /// called once the caller knows whether the keys file actually has any
    /// keys in it.
    ///
    /// # Errors
    /// `bind_addr` is not a valid IP address.
    pub fn validated_bind(&self) -> anyhow::Result<std::net::SocketAddr> {
        let ip: std::net::IpAddr = self.bind_addr.parse().map_err(|e| {
            anyhow::anyhow!(
                "invalid bind_addr '{}' (not an IP address): {e}",
                self.bind_addr
            )
        })?;
        Ok(std::net::SocketAddr::from((ip, self.port)))
    }

    /// T4.2: the public-bind security gate, split out of
    /// [`validated_bind`](Self::validated_bind) — see its doc comment for why.
    ///
    /// A non-loopback `addr` refuses to start unless `allow_insecure_public_bind: true`
    /// explicitly accepts it when the keys file (real [`crate::AuthConfig`], not this
    /// struct's deprecated `api_keys` field) has **no inference keys**
    /// (`has_inference_keys: false`). Admin keys do not count (0.9.1 S3, outside review
    /// I-05): inference is gated by `api_keys` alone, so a keys file with only
    /// `admin_api_keys` — the one `rv start` creates — serves inference to anyone who can
    /// reach the port. `has_admin_keys` only picks the refusal's wording. Call after
    /// resolving the keys file, before any model load.
    ///
    /// # Errors
    /// Non-loopback bind + no inference keys, without the explicit opt-in.
    pub fn enforce_open_bind_gate(
        &self,
        addr: std::net::SocketAddr,
        has_inference_keys: bool,
        has_admin_keys: bool,
    ) -> anyhow::Result<()> {
        if addr.ip().is_loopback() || has_inference_keys || self.allow_insecure_public_bind {
            return Ok(());
        }
        if has_admin_keys {
            anyhow::bail!(
                "refusing to start: bind_addr '{}' is reachable from the network and the keys \
                 file has admin_api_keys but no api_keys. Admin keys protect only /v1/admin/*; \
                 with no api_keys, inference is open to anyone who can reach the port. Add an \
                 api_keys entry to the keys file, bind to 127.0.0.1, or set \
                 allow_insecure_public_bind: true to serve inference without a key on purpose",
                self.bind_addr
            );
        }
        anyhow::bail!(
            "refusing to start: bind_addr '{}' is reachable from the network but no keys are \
             configured (auth OFF). Set keys in the keys file (see \
             keys_file/resolve_keys_file_path), bind to 127.0.0.1, or set \
             allow_insecure_public_bind: true to explicitly run an open server",
            self.bind_addr
        );
    }
}

/// Resolve where a config file's companion keys file lives.
///
/// `explicit` is [`Config::keys_file`] verbatim: a relative path resolves
/// against `config_path`'s parent directory (never the process cwd — per-box
/// config/keys pairs must travel together regardless of launch directory);
/// an absolute path is used as-is. `None` derives the sibling path: strip a
/// trailing `.config.json` (or bare `.json`, for a config file not following
/// the `*.config.json` convention) from `config_path`'s file name and append
/// `.keys.json` — e.g. `scripts/myserver.config.json` → `scripts/myserver.keys.json`.
#[must_use]
pub fn resolve_keys_file_path(config_path: &Path, explicit: Option<&Path>) -> PathBuf {
    if let Some(explicit) = explicit {
        return if explicit.is_absolute() {
            explicit.to_path_buf()
        } else {
            config_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(explicit)
        };
    }
    let dir = config_path.parent().unwrap_or_else(|| Path::new("."));
    let name = config_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("config.json");
    let stem = name
        .strip_suffix(".config.json")
        .or_else(|| name.strip_suffix(".json"))
        .unwrap_or(name);
    dir.join(format!("{stem}.keys.json"))
}

/// Per-model policy invariants — shared between [`Config::validate`] (looping
/// over every entry loaded from the static file) and
/// `ModelManager::add_model` (the runtime registration path,
/// `POST /v1/admin/models/add`), so both ways of registering a model enforce
/// identical invariants rather than the runtime path silently accepting
/// values the static file would reject at startup.
///
/// Does NOT check `speculative.draft_model` against the full model registry
/// (the "resident twice" warning) — that needs the caller's own models map,
/// which differs in shape between a static [`Config`] and `ModelManager`'s
/// runtime maps.
///
/// # Errors
/// See [`Config::validate`]'s per-model knob documentation — the same
/// `kv_cache_gb`/`max_concurrent_streams`/`max_prompt_len`/`vram_gb`/
/// `speculative` checks apply here, scoped to one entry.
pub(crate) fn validate_model_entry(model_id: &str, entry: &ModelEntry) -> anyhow::Result<()> {
    let policy = &entry.policy;
    if let Some(kv) = policy.kv_cache_gb {
        anyhow::ensure!(
            kv.is_finite() && kv > 0.0,
            "models['{model_id}'].kv_cache_gb must be a finite positive number, \
             got {kv} (omit it for unbounded grab-all)"
        );
    }
    if let Some(t) = policy.load_timeout_secs {
        validate_load_timeout(&format!("models['{model_id}'].load_timeout_secs"), t)?;
    }
    if let Some(streams) = policy.max_concurrent_streams {
        anyhow::ensure!(
            streams >= 1,
            "models['{model_id}'].max_concurrent_streams must be at least 1, got {streams}"
        );
    }
    if let Some(p) = policy.kv_cache_precision.as_deref() {
        normalize_kv_precision(p)
            .map_err(|e| anyhow::anyhow!("models['{model_id}'].kv_cache_precision: {e}"))?;
    }
    if let Some(max_prompt_len) = policy.max_prompt_len {
        anyhow::ensure!(
            max_prompt_len >= 1,
            "models['{model_id}'].max_prompt_len must be at least 1, got {max_prompt_len} \
             (omit it to keep OpenVINO's own NPU default)"
        );
    }
    if let Some(min_response_len) = policy.min_response_len {
        anyhow::ensure!(
            min_response_len >= 1,
            "models['{model_id}'].min_response_len must be at least 1, got {min_response_len} \
             (omit it to keep OpenVINO's own NPU default)"
        );
    }
    anyhow::ensure!(
        entry.vram_gb.is_finite() && entry.vram_gb >= 0.0,
        "models['{model_id}'].vram_gb must be a finite non-negative number, got {}",
        entry.vram_gb
    );
    // Realtime voice arbitration v2 (D4/D7): fail loud on an unrecognised
    // capability label rather than let it silently never match any floor.
    for cap in &policy.capabilities {
        anyhow::ensure!(
            KNOWN_CAPABILITIES.contains(&cap.as_str()),
            "models['{model_id}'].capabilities contains unknown capability '{cap}' — known: {}",
            KNOWN_CAPABILITIES.join(", ")
        );
    }
    // Realtime voice arbitration v2 (D3): same finite/non-negative discipline
    // as the global default — a NaN or negative override would defeat the
    // grace check the same way a bad total_vram_gb defeats VRAM gating.
    if let Some(g) = policy.eviction_grace_secs {
        anyhow::ensure!(
            g.is_finite() && g >= 0.0,
            "models['{model_id}'].eviction_grace_secs must be a finite non-negative number, got {g}"
        );
    }
    if let Some(s) = &policy.sampling {
        validate_sampling(model_id, s)?;
    }
    // Speculative decoding (Layer A — pure config, no disk access; see
    // the project's internal engineering log).
    if let Some(spec) = &policy.speculative {
        anyhow::ensure!(
            entry.vram_gb > 0.0,
            "models['{model_id}'].speculative requires vram_gb > 0.0 — the draft's \
             VRAM would never get charged against a 0.0 (unmetered) target"
        );
        anyhow::ensure!(
            spec.draft_vram_gb.is_finite() && spec.draft_vram_gb > 0.0,
            "models['{model_id}'].speculative.draft_vram_gb must be a finite \
             positive number, got {}",
            spec.draft_vram_gb
        );
        anyhow::ensure!(
            spec.num_assistant_tokens >= 1,
            "models['{model_id}'].speculative.num_assistant_tokens must be at \
             least 1, got {}",
            spec.num_assistant_tokens
        );
        anyhow::ensure!(
            !spec.draft_model.is_empty(),
            "models['{model_id}'].speculative.draft_model must not be empty"
        );
        anyhow::ensure!(
            is_plain_model_id(&spec.draft_model),
            "models['{model_id}'].speculative.draft_model '{}' must be the name of a folder \
             directly inside models_dir (no '/', '\\', '..' or drive letter)",
            spec.draft_model
        );
        anyhow::ensure!(
            spec.draft_model != model_id,
            "models['{model_id}'].speculative.draft_model must not name the \
             target model itself"
        );
    }
    Ok(())
}

// ============================================================
// Unit tests
// ============================================================

/// Generates [`globals_not_applied`] from a single field list.
///
/// One list, not two: an earlier version spelled the fields once to destructure
/// and again to compare, which allowed a field to be bound and then silently
/// left out of the diff — the same class of omission the whole function exists
/// to prevent.
macro_rules! define_globals_not_applied {
    (exempt: [$($exempt:ident),* $(,)?], diffed: [$($field:ident),* $(,)?] $(,)?) => {
        /// Names of **global** (non-model) config fields whose value in `file`
        /// differs from the value in `boot` — i.e. edits that a
        /// `POST /v1/admin/config/reload` will *not* apply, because reload
        /// re-reads only the per-model registry.
        ///
        /// # Rot-proofing
        /// The destructure has no `..` rest pattern, so **adding a field to
        /// [`Config`] is a compile error until it is explicitly listed** as
        /// `exempt` or `diffed`. A global that silently escaped this diff would
        /// reintroduce precisely the invisible failure this exists to remove.
        #[allow(clippy::float_cmp)] // exact: "did the operator change this literal?"
        pub(crate) fn globals_not_applied(boot: &Config, file: &Config) -> Vec<String> {
            let Config { $($exempt: _,)* $($field,)* } = file;
            let mut changed = Vec::new();
            $( if boot.$field != *$field { changed.push(stringify!($field).to_owned()); } )*
            changed.sort();
            changed
        }
    };
}

define_globals_not_applied! {
    // Exempt, with reasons:
    //   models            -- reload DOES handle this, via `dynamic_entries`.
    //   api_keys/admin_api_keys/keys_file
    //                     -- own reload (`POST /v1/admin/keys/reload`); a reload
    //                        carrying inline keys is refused before this runs.
    //   unknown_fields    -- parser bookkeeping, not an operator knob.
    exempt: [models, api_keys, admin_api_keys, keys_file, unknown_fields],
    diffed: [
        models_dir, device, preload, total_vram_gb, max_num_seqs, cache_size_gb,
        default_kv_cache_gb, vram_safety_margin_gb, npu_footprint_factor,
        cpu_image_gen_max_pixels, max_audio_duration_secs, max_images_per_request,
        max_image_pixels_per_request, max_image_pixels, min_kv_cache_gb,
        system_ram_reservation_gb, system_ram_budget_gb, default_placement, light_model_max_gb,
        light_stt_max_gb, dgpu_size_ceiling_fraction, domain_budgets,
        eviction_grace_secs, realtime_defaults, realtime_viable_minimum,
        device_budgets, kv_cache_precision, enable_prefix_caching,
        cors_allowed_origins, admission_queue_timeout_ms,
        device_admission_queue_timeout_ms, embedding_device, default_embed_model,
        embedding_pooling, embedding_normalize, max_prompt_array, max_embedding_inputs,
        max_embedding_batch_tokens, max_tokens_cap, stream_first_token_timeout_secs,
        stream_token_gap_timeout_secs, bind_addr, port, allow_insecure_public_bind, ov_cache_dir,
        ov_cache_max_gb, ov_cache_sweep_interval_secs, ov_cache_layout, kv_precision_pass,
        kv_gate_mode, kv_gate_exact_on_unknown_engine, kv_headroom_mode, kv_transient_factor,
        kv_headroom_margin_gib, kv_headroom_min_tokens, kv_headroom_inflight_tokens, kv_warmup,
        kv_pressure_monitor_enabled,
        kv_pressure_threshold_pct, kv_pressure_sustained_secs,
        kv_pressure_sweep_interval_secs, kv_resize_cooldown_secs, load_timeout_secs,
        on_demand_wait_secs, supervisor,
    ],
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    // ---- 0.9.1 V5a: the per-model `sampling` object ----

    #[allow(clippy::needless_pass_by_value)]
    fn entry_with(sampling: serde_json::Value) -> serde_json::Result<ModelEntry> {
        serde_json::from_value(serde_json::json!({"vram_gb": 1.0, "sampling": sampling}))
    }

    /// `deny_unknown_fields`: a typo inside `sampling` is an error, not a silent no-op.
    #[test]
    fn sampling_typo_is_rejected() {
        let err = entry_with(serde_json::json!({"temprature": 0.7}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("temprature"), "{err}");
        let e = entry_with(serde_json::json!({"temperature": 0.7, "top_p": 0.8})).unwrap();
        assert_eq!(e.policy.sampling.unwrap().top_p, Some(0.8));
        // Absent and null are both "no operator layer".
        let e: ModelEntry = serde_json::from_value(serde_json::json!({"vram_gb": 1.0})).unwrap();
        assert_eq!(e.policy.sampling, None);
        assert_eq!(
            entry_with(serde_json::Value::Null).unwrap().policy.sampling,
            None
        );
    }

    #[test]
    fn sampling_ranges_and_greedy_combination() {
        let ok = [
            serde_json::json!({"temperature": 0.0}),
            serde_json::json!({"temperature": 2.0, "top_p": 1.0, "top_k": 1}),
            serde_json::json!({"repetition_penalty": 1.0}),
            serde_json::json!({"repetition_penalty": 2.0}),
            serde_json::json!({"presence_penalty": -2.0, "frequency_penalty": 2.0}),
            serde_json::json!({}),
        ];
        for s in ok {
            let e = entry_with(s.clone()).unwrap();
            assert!(validate_model_entry("m", &e).is_ok(), "{s}");
        }
        let bad = [
            (serde_json::json!({"temperature": 2.1}), "temperature"),
            (serde_json::json!({"temperature": -0.1}), "temperature"),
            (
                serde_json::json!({"temperature": 0.5, "top_p": 0.0}),
                "top_p",
            ),
            (
                serde_json::json!({"temperature": 0.5, "top_p": 1.1}),
                "top_p",
            ),
            (serde_json::json!({"temperature": 0.5, "top_k": 0}), "top_k"),
            (
                serde_json::json!({"temperature": 0.0, "top_p": 0.9}),
                "greedy",
            ),
            (
                serde_json::json!({"temperature": 0.0, "top_k": 20}),
                "greedy",
            ),
            (
                serde_json::json!({"repetition_penalty": 0.9}),
                "repetition_penalty",
            ),
            (
                serde_json::json!({"repetition_penalty": 2.5}),
                "repetition_penalty",
            ),
            (
                serde_json::json!({"presence_penalty": 2.5}),
                "presence_penalty",
            ),
            (
                serde_json::json!({"frequency_penalty": -3.0}),
                "frequency_penalty",
            ),
        ];
        for (s, needle) in bad {
            let e = entry_with(s.clone()).unwrap();
            let err = validate_model_entry("m", &e).unwrap_err().to_string();
            assert!(err.contains(needle), "{s}: {err}");
        }
    }

    /// Operator `top_p`/`top_k` without a temperature: rejected unless the model's own file says
    /// `do_sample: true` (a missing file counts as no `do_sample`); with a temperature, fine.
    #[test]
    fn sampling_top_p_without_temperature_needs_a_do_sample_file() {
        let dir = tempfile::tempdir().unwrap();
        let top_p_only = entry_with(serde_json::json!({"top_p": 0.9})).unwrap();
        let err = validate_sampling_against_file("m", &top_p_only, dir.path())
            .unwrap_err()
            .to_string();
        assert!(err.contains("do_sample"), "{err}");
        std::fs::write(
            dir.path().join("generation_config.json"),
            r#"{"temperature": 0.7, "top_k": 20}"#,
        )
        .unwrap();
        assert!(validate_sampling_against_file("m", &top_p_only, dir.path()).is_err());
        let with_t = entry_with(serde_json::json!({"temperature": 0.2, "top_p": 0.9})).unwrap();
        assert!(validate_sampling_against_file("m", &with_t, dir.path()).is_ok());
        std::fs::write(
            dir.path().join("generation_config.json"),
            r#"{"do_sample": true, "temperature": 0.6, "top_k": 20}"#,
        )
        .unwrap();
        assert!(validate_sampling_against_file("m", &top_p_only, dir.path()).is_ok());
        // F6: `do_sample: true` with no temperature, or a package dropped for a bad value,
        // leaves the engine at T 1.0 — counts as no `do_sample`.
        for body in [
            r#"{"do_sample": true, "top_k": 20}"#,
            r#"{"do_sample": true, "temperature": 0.6, "top_p": 7.0}"#,
        ] {
            std::fs::write(dir.path().join("generation_config.json"), body).unwrap();
            assert!(
                validate_sampling_against_file("m", &top_p_only, dir.path()).is_err(),
                "{body}"
            );
        }
    }

    /// 0.9.1 `thinking_budget_tokens`: default `null` (in the model template too), an integer
    /// >= 0 in a stanza, PATCH set / unchanged / `null` clears, all hot; negatives refused.
    #[test]
    fn thinking_budget_tokens_config_and_patch() {
        let template = Config::default_model_template().unwrap();
        assert_eq!(
            template.get("thinking_budget_tokens"),
            Some(&serde_json::Value::Null)
        );
        let e: ModelEntry = serde_json::from_value(
            serde_json::json!({"vram_gb": 1.0, "thinking_budget_tokens": 0}),
        )
        .unwrap();
        assert_eq!(e.policy.thinking_budget_tokens, Some(0));
        assert!(
            serde_json::from_value::<ModelEntry>(
                serde_json::json!({"vram_gb": 1.0, "thinking_budget_tokens": -5})
            )
            .is_err()
        );
        let base = ModelEntry::default();
        let set: ModelEntryPatch =
            serde_json::from_value(serde_json::json!({"thinking_budget_tokens": 2048})).unwrap();
        assert!(!set.is_empty());
        let (merged, out) = set.apply_to(&base);
        assert_eq!(merged.policy.thinking_budget_tokens, Some(2048));
        assert_eq!(out.applied_live, vec!["thinking_budget_tokens"]);
        let (_, again) = set.apply_to(&merged);
        assert_eq!(again.unchanged, vec!["thinking_budget_tokens"]);
        let clear: ModelEntryPatch =
            serde_json::from_value(serde_json::json!({"thinking_budget_tokens": null})).unwrap();
        assert!(!clear.is_empty(), "a null-only body is a clear, not empty");
        let (cleared, out) = clear.apply_to(&merged);
        assert_eq!(cleared.policy.thinking_budget_tokens, None);
        assert_eq!(out.applied_live, vec!["thinking_budget_tokens"]);
        assert!(
            serde_json::from_value::<ModelEntryPatch>(
                serde_json::json!({"thinking_budget_tokens": -1})
            )
            .is_err()
        );
    }

    /// PATCH `sampling`: whole-object replace, hot; `null` clears; a null-only body is not empty.
    #[test]
    fn sampling_patch_replaces_whole_object_and_null_clears() {
        let base = entry_with(serde_json::json!({"temperature": 0.7, "top_p": 0.8})).unwrap();
        let p: ModelEntryPatch = serde_json::from_value(
            serde_json::json!({"sampling": {"top_k": 5, "temperature": 1.0}}),
        )
        .unwrap();
        let (merged, out) = p.apply_to(&base);
        let s = merged.policy.sampling.unwrap();
        assert_eq!(
            (s.temperature, s.top_p, s.top_k),
            (Some(1.0), None, Some(5))
        );
        assert_eq!(out.applied_live, vec!["sampling"]);
        let clear: ModelEntryPatch =
            serde_json::from_value(serde_json::json!({"sampling": null})).unwrap();
        assert!(!clear.is_empty());
        let (merged, _) = clear.apply_to(&base);
        assert_eq!(merged.policy.sampling, None);
        let typo = serde_json::from_value::<ModelEntryPatch>(
            serde_json::json!({"sampling": {"temprature": 1.0}}),
        );
        assert!(typo.is_err());
    }

    /// The file-dependent `do_sample` rule never fails `Config::load` (boot, reload): a missing
    /// model directory is skipped, a present one only WARNs.
    #[test]
    fn config_validate_skips_the_file_check_for_a_missing_model_dir() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        let cfg = serde_json::json!({
            "models_dir": dir.path().join("models"), "device": "CPU", "total_vram_gb": 0.0,
            "models": {"m": {"vram_gb": 0.0, "sampling": {"top_p": 0.9}}}
        });
        std::fs::write(&path, cfg.to_string()).unwrap();
        assert!(Config::load(&path).is_ok());
        // With the directory present the rule is checked, but at load/boot it only WARNs
        // (the model file can change on disk; coordinator ruling B).
        std::fs::create_dir_all(dir.path().join("models").join("m")).unwrap();
        assert!(Config::load(&path).is_ok());
    }

    /// Step 6b: `load_timeout_secs` is 0 (off) or at least 60, globally and per model (the
    /// per-model check is the one `add_model` shares).
    #[test]
    fn load_timeout_is_zero_or_at_least_a_minute() {
        let base = r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":0.0"#;
        for (extra, ok) in [
            ("", true),
            (r#","load_timeout_secs":0"#, true),
            (r#","load_timeout_secs":60"#, true),
            (r#","load_timeout_secs":59"#, false),
            (
                r#","models":{"m":{"vram_gb":0.0,"load_timeout_secs":5}}"#,
                false,
            ),
            (
                r#","models":{"m":{"vram_gb":0.0,"load_timeout_secs":0}}"#,
                true,
            ),
        ] {
            let cfg: Config = serde_json::from_str(&format!("{base}{extra}}}")).unwrap();
            assert_eq!(cfg.validate().is_ok(), ok, "{extra}");
        }
        assert_eq!(
            serde_json::from_str::<Config>(&format!("{base}}}"))
                .unwrap()
                .load_timeout_secs,
            600
        );
    }

    /// F1b: `on_demand_wait_secs` defaults to 0, is at most 300 and below a
    /// non-zero global `load_timeout_secs`; with `load_timeout_secs: 0` (no deadline) only the
    /// 300 cap applies. The error names the key (boot refusal, exit 78).
    #[test]
    fn on_demand_wait_is_bounded_by_300_and_the_load_timeout() {
        let base = r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":0.0"#;
        for (extra, ok) in [
            ("", true),
            (r#","on_demand_wait_secs":0"#, true),
            (r#","on_demand_wait_secs":120"#, true),
            (r#","on_demand_wait_secs":300"#, true),
            (r#","on_demand_wait_secs":301"#, false),
            (r#","on_demand_wait_secs":18446744073709551615"#, false),
            (r#","load_timeout_secs":60,"on_demand_wait_secs":59"#, true),
            (r#","load_timeout_secs":60,"on_demand_wait_secs":60"#, false),
            (r#","load_timeout_secs":60,"on_demand_wait_secs":61"#, false),
            (r#","load_timeout_secs":0,"on_demand_wait_secs":300"#, true),
            (r#","load_timeout_secs":0,"on_demand_wait_secs":301"#, false),
        ] {
            let cfg: Config = serde_json::from_str(&format!("{base}{extra}}}")).unwrap();
            let r = cfg.validate();
            assert_eq!(r.is_ok(), ok, "{extra}: {r:?}");
            if let Err(e) = r {
                assert!(e.to_string().contains("on_demand_wait_secs"), "{e}");
            }
        }
        assert_eq!(
            serde_json::from_str::<Config>(&format!("{base}}}"))
                .unwrap()
                .on_demand_wait_secs,
            0
        );
    }

    /// F1b: `on_demand_wait_secs` is a global a config reload does not apply: changing it in the
    /// file is reported as needing a restart, like every other global.
    #[test]
    fn on_demand_wait_change_is_reported_as_not_applied_by_reload() {
        let boot = minimal("");
        let file = minimal(r#", "on_demand_wait_secs": 120"#);
        assert_eq!(
            globals_not_applied(&boot, &file),
            vec!["on_demand_wait_secs".to_owned()]
        );
    }

    /// Windows pass 2 D12: a config written with a UTF-8 BOM (PowerShell 5.1
    /// `Set-Content -Encoding utf8`, Notepad) loads exactly like one without.
    #[test]
    fn config_with_a_utf8_bom_loads_like_one_without() {
        let d = tempfile::tempdir().unwrap();
        let body = r#"{"models_dir":"/m","device":"GPU","total_vram_gb":22.5}"#;
        let plain = d.path().join("plain.json");
        let bom = d.path().join("bom.json");
        std::fs::write(&plain, body).unwrap();
        std::fs::write(&bom, format!("\u{feff}{body}")).unwrap();
        let a = Config::load(&plain).unwrap();
        let b = Config::load(&bom).unwrap();
        assert!((a.total_vram_gb - b.total_vram_gb).abs() < f64::EPSILON);
        assert_eq!(a.models_dir, b.models_dir);
        assert_eq!(read_json_text(&bom).unwrap(), body);
    }

    /// Stage 0 (U-07 subset): the accepted `kv_cache_precision` spellings, the
    /// `u8` -> `i8` alias, and the values that now fail validation.
    #[test]
    fn kv_cache_precision_normalises_and_rejects_unknown_values() {
        for (raw, want) in [
            ("", ""),
            ("auto", ""),
            ("i8", "i8"),
            ("u4", "u4"),
            ("f16", "f16"),
            ("u8", "i8"),
        ] {
            assert_eq!(normalize_kv_precision(raw).unwrap(), want, "raw {raw:?}");
        }
        for bad in ["xyz", "f32", "i4", "U8", " i8"] {
            let msg = normalize_kv_precision(bad).unwrap_err().to_string();
            assert!(msg.contains(bad), "message names the value: {msg}");
            assert!(
                msg.contains("\"auto\"") && msg.contains("\"u4\""),
                "lists accepted: {msg}"
            );
        }
    }

    /// The default and every spelling of the key go through `Config::load`;
    /// an unsupported one stops startup instead of silently keeping the plugin default.
    #[test]
    fn config_load_validates_kv_cache_precision() {
        let d = tempfile::tempdir().unwrap();
        let write = |name: &str, extra: &str| {
            let p = d.path().join(name);
            std::fs::write(
                &p,
                format!(r#"{{"models_dir":"/m","device":"GPU","total_vram_gb":22.5{extra}}}"#),
            )
            .unwrap();
            p
        };
        let default = Config::load(&write("default.json", "")).unwrap();
        assert_eq!(default.kv_cache_precision, "u8", "shipped default is kept");
        for ok in ["", "auto", "i8", "u4", "f16", "u8"] {
            let p = write("ok.json", &format!(r#","kv_cache_precision":"{ok}""#));
            Config::load(&p).unwrap_or_else(|e| panic!("{ok:?} must load: {e}"));
        }
        for bad in ["f32", "i4", "xyz"] {
            let p = write("bad.json", &format!(r#","kv_cache_precision":"{bad}""#));
            let err = Config::load(&p).unwrap_err().to_string();
            assert!(
                err.contains(bad) && err.contains("accepted values"),
                "{err}"
            );
        }
    }

    /// Windows pass 2 D13: persisting a patched model keeps the operator's
    /// stanza as written — it gains the patched field, not every default.
    #[test]
    fn kv_cache_precision_policy_parses_validates_and_patches_on_next_load() {
        let entry = |json: &str| -> ModelEntry { serde_json::from_str(json).unwrap() };
        assert_eq!(entry(r#"{"vram_gb":2.5}"#).policy.kv_cache_precision, None);
        for ok in ["auto", "", "i8", "u4", "f16", "u8"] {
            let e = entry(&format!(r#"{{"vram_gb":2.5,"kv_cache_precision":"{ok}"}}"#));
            assert!(validate_model_entry("m", &e).is_ok(), "{ok}");
        }
        for bad in ["f32", "i4", "bogus", "U4"] {
            let e = entry(&format!(
                r#"{{"vram_gb":2.5,"kv_cache_precision":"{bad}"}}"#
            ));
            let err = validate_model_entry("m", &e).unwrap_err().to_string();
            assert!(
                err.contains("models['m'].kv_cache_precision") && err.contains("accepted"),
                "{err}"
            );
        }
        // PATCH: next load; null clears.
        let base = entry(r#"{"vram_gb":2.5}"#);
        let set: ModelEntryPatch = serde_json::from_str(r#"{"kv_cache_precision":"u4"}"#).unwrap();
        let (after, out) = set.apply_to(&base);
        assert_eq!(after.policy.kv_cache_precision.as_deref(), Some("u4"));
        assert_eq!(out.effective_on_next_load, ["kv_cache_precision"]);
        let clear: ModelEntryPatch =
            serde_json::from_str(r#"{"kv_cache_precision":null}"#).unwrap();
        let (cleared, _) = clear.apply_to(&after);
        assert_eq!(cleared.policy.kv_cache_precision, None);
    }

    #[test]
    fn enable_prefix_caching_policy_defaults_to_none_and_round_trips() {
        let unset: ModelEntry = serde_json::from_str(r#"{"vram_gb":2.5}"#).unwrap();
        assert_eq!(
            unset.policy.enable_prefix_caching, None,
            "default: server decides"
        );
        let on: ModelEntry =
            serde_json::from_str(r#"{"vram_gb":2.5,"enable_prefix_caching":true}"#).unwrap();
        assert_eq!(on.policy.enable_prefix_caching, Some(true));
        let off: ModelEntry =
            serde_json::from_str(r#"{"vram_gb":2.5,"enable_prefix_caching":false}"#).unwrap();
        assert_eq!(off.policy.enable_prefix_caching, Some(false));
    }

    /// PATCH sets, changes and clears the override; every change is next-load
    /// (the engine is built once, so a live flip cannot apply), and the
    /// no-op cases report `unchanged`.
    #[test]
    fn enable_prefix_caching_patch_is_next_load_and_null_clears() {
        let base: ModelEntry = serde_json::from_str(r#"{"vram_gb":2.5}"#).unwrap();
        let set: ModelEntryPatch =
            serde_json::from_str(r#"{"enable_prefix_caching":true}"#).unwrap();
        let (after_set, out) = set.apply_to(&base);
        assert_eq!(after_set.policy.enable_prefix_caching, Some(true));
        assert_eq!(out.effective_on_next_load, ["enable_prefix_caching"]);
        assert!(out.applied_live.is_empty());

        let (_, again) = set.apply_to(&after_set);
        assert_eq!(again.unchanged, ["enable_prefix_caching"]);

        let clear: ModelEntryPatch =
            serde_json::from_str(r#"{"enable_prefix_caching":null}"#).unwrap();
        let (after_clear, out) = clear.apply_to(&after_set);
        assert_eq!(after_clear.policy.enable_prefix_caching, None);
        assert_eq!(out.effective_on_next_load, ["enable_prefix_caching"]);

        // An absent field is not a patch at all.
        let empty: ModelEntryPatch = serde_json::from_str("{}").unwrap();
        assert!(empty.is_empty());
        assert!(!set.is_empty());
    }

    #[test]
    fn minimal_stanza_adds_only_the_patched_field() {
        let old: serde_json::Value =
            serde_json::from_str(r#"{"vram_gb":2.5,"pinned":true}"#).unwrap();
        let mut merged: ModelEntry = serde_json::from_value(old.clone()).unwrap();
        merged.policy.priority = 1;
        let out = minimal_stanza(Some(&old), &merged).unwrap();
        let keys: Vec<&str> = out
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, ["vram_gb", "pinned", "priority"], "{out}");
        assert_eq!(out["priority"], 1);
        assert_eq!(out["pinned"], true);
        assert!(out.get("kind").is_none(), "no `kind: null` written: {out}");
    }

    /// A field the operator wrote stays even when patched back to its default;
    /// a brand-new stanza (`add_model`) carries only non-defaults.
    #[test]
    fn minimal_stanza_keeps_written_fields_and_omits_untouched_defaults() {
        let old: serde_json::Value =
            serde_json::from_str(r#"{"vram_gb":2.5,"pinned":true}"#).unwrap();
        let mut merged: ModelEntry = serde_json::from_value(old.clone()).unwrap();
        merged.policy.pinned = false;
        let out = minimal_stanza(Some(&old), &merged).unwrap();
        assert_eq!(out["pinned"], false, "{out}");

        let fresh = minimal_stanza(
            None,
            &ModelEntry {
                vram_gb: 3.0,
                ..ModelEntry::default()
            },
        )
        .unwrap();
        assert_eq!(fresh.as_object().unwrap().len(), 1, "only vram_gb: {fresh}");
    }

    /// The operator's key order and a key this build does not model survive a
    /// patch (a stanza hand-written as `device` before `vram_gb`).
    #[test]
    fn minimal_stanza_keeps_key_order_and_unknown_keys() {
        let old: serde_json::Value =
            serde_json::from_str(r#"{"device":"NPU","note":"keep me","vram_gb":5.5}"#).unwrap();
        let mut merged: ModelEntry = serde_json::from_value(old.clone()).unwrap();
        merged.policy.priority = 2;
        let out = minimal_stanza(Some(&old), &merged).unwrap();
        let keys: Vec<&str> = out
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, ["device", "note", "vram_gb", "priority"], "{out}");
        assert_eq!(out["note"], "keep me");
    }

    /// An unchanged file reports nothing — the common path must stay quiet, so
    /// the new field never adds noise to a routine model-only reload.
    /// C8: the changed keys of a model stanza, by JSON name; an absent key equals `null`.
    #[test]
    fn model_entry_changes_names_the_changed_stanza_keys() {
        let a: ModelEntry = serde_json::from_str(r#"{"vram_gb": 1.0}"#).unwrap();
        let same: ModelEntry =
            serde_json::from_str(r#"{"vram_gb": 1.0, "chat_template": null}"#).unwrap();
        assert!(model_entry_changes(&a, &same).is_empty());
        let b: ModelEntry = serde_json::from_str(
            r#"{"vram_gb": 2.0, "chat_template": "templates/lfm2.jinja", "pinned": true}"#,
        )
        .unwrap();
        assert_eq!(
            model_entry_changes(&a, &b),
            vec!["chat_template", "pinned", "vram_gb"]
        );
    }

    #[test]
    fn globals_not_applied_is_empty_when_nothing_changed() {
        let json = r#"{"models_dir":"/m","device":"GPU","total_vram_gb":22.5}"#;
        let boot: Config = serde_json::from_str(json).unwrap();
        let file: Config = serde_json::from_str(json).unwrap();
        assert!(globals_not_applied(&boot, &file).is_empty());
    }

    /// Editing only `models` is not a global change: that is the one thing
    /// reload genuinely does apply, so flagging it would be a false alarm.
    #[test]
    fn globals_not_applied_ignores_model_registry_edits() {
        let boot: Config = serde_json::from_str(
            r#"{"models_dir":"/m","device":"GPU","total_vram_gb":22.5,
                "models":{"a":{"vram_gb":5.0}}}"#,
        )
        .unwrap();
        let file: Config = serde_json::from_str(
            r#"{"models_dir":"/m","device":"GPU","total_vram_gb":22.5,
                "models":{"a":{"vram_gb":5.0},"b":{"vram_gb":9.0}}}"#,
        )
        .unwrap();
        assert!(globals_not_applied(&boot, &file).is_empty());
    }

    /// The scenario that motivated this: an operator raises the KV knobs,
    /// reloads, gets 200 OK — and the load path keeps using the boot values.
    /// Now the response names them.
    #[test]
    fn globals_not_applied_names_changed_kv_knobs() {
        let boot: Config = serde_json::from_str(
            r#"{"models_dir":"/m","device":"GPU","total_vram_gb":22.5,
                "min_kv_cache_gb":1.0,"cache_size_gb":0.0}"#,
        )
        .unwrap();
        let file: Config = serde_json::from_str(
            r#"{"models_dir":"/m","device":"GPU","total_vram_gb":27.0,
                "min_kv_cache_gb":2.0,"cache_size_gb":8.0}"#,
        )
        .unwrap();
        // Sorted, so the assertion is order-independent.
        assert_eq!(
            globals_not_applied(&boot, &file),
            vec![
                "cache_size_gb".to_owned(),
                "min_kv_cache_gb".to_owned(),
                "total_vram_gb".to_owned(),
            ]
        );
    }

    /// Restart-required knobs are reported the same way as refreshable ones —
    /// the report says "not applied", it does not claim a taxonomy.
    #[test]
    fn globals_not_applied_covers_boot_latched_knobs() {
        let boot: Config = serde_json::from_str(
            r#"{"models_dir":"/m","device":"GPU","total_vram_gb":22.5,"port":11437}"#,
        )
        .unwrap();
        let file: Config = serde_json::from_str(
            r#"{"models_dir":"/m","device":"GPU","total_vram_gb":22.5,"port":8080}"#,
        )
        .unwrap();
        assert_eq!(globals_not_applied(&boot, &file), vec!["port".to_owned()]);
    }

    /// Keys are exempt: they have their own reload endpoint, and a config
    /// carrying inline keys is refused before the diff is ever reached.
    #[test]
    fn globals_not_applied_exempts_keys() {
        let boot: Config = serde_json::from_str(
            r#"{"models_dir":"/m","device":"GPU","total_vram_gb":22.5,"keys_file":"/a.json"}"#,
        )
        .unwrap();
        let file: Config = serde_json::from_str(
            r#"{"models_dir":"/m","device":"GPU","total_vram_gb":22.5,"keys_file":"/b.json"}"#,
        )
        .unwrap();
        assert!(globals_not_applied(&boot, &file).is_empty());
    }

    /// All fields present → deserialises correctly.
    #[test]
    fn test_config_deserializes_valid_json() {
        let json = r#"{
            "models_dir": "/opt/rustedvino/models",
            "device": "GPU.1",
            "preload": ["qwen3-8b-int4-ov"],
            "models": {"qwen3-8b-int4-ov": {"vram_gb": 5.5}},
            "total_vram_gb": 22.5
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.device, "GPU.1");
        assert!((cfg.total_vram_gb - 22.5).abs() < 1e-9);
        assert_eq!(cfg.preload, vec!["qwen3-8b-int4-ov"]);
        assert!((cfg.models["qwen3-8b-int4-ov"].vram_gb - 5.5).abs() < 1e-9);
        assert_eq!(cfg.models_dir, PathBuf::from("/opt/rustedvino/models"));
        // New fields default correctly when absent from JSON.
        assert!((cfg.vram_safety_margin_gb - 1.0).abs() < 1e-9);
        assert!((cfg.min_kv_cache_gb - 1.0).abs() < 1e-9);
    }

    /// Tier 3 (the image-metadata plan): `precision`/`model_source`/
    /// `model_revision` parse from the flattened model stanza when present, and
    /// default to `None` (never fabricated) when absent.
    #[test]
    fn test_model_policy_tier3_provenance_fields() {
        let json = r#"{
            "models_dir": "/opt/rustedvino/models",
            "device": "GPU.1",
            "total_vram_gb": 22.5,
            "models": {
                "lcm-with-provenance": {
                    "vram_gb": 3.5,
                    "precision": "int8",
                    "model_source": "OpenVINO/LCM_Dreamshaper_v7-int8-ov",
                    "model_revision": "a1b2c3d"
                },
                "flux-no-provenance": {"vram_gb": 6.0}
            }
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();

        let with_provenance = &cfg.models["lcm-with-provenance"].policy;
        assert_eq!(with_provenance.precision.as_deref(), Some("int8"));
        assert_eq!(
            with_provenance.model_source.as_deref(),
            Some("OpenVINO/LCM_Dreamshaper_v7-int8-ov")
        );
        assert_eq!(with_provenance.model_revision.as_deref(), Some("a1b2c3d"));

        let without_provenance = &cfg.models["flux-no-provenance"].policy;
        assert_eq!(without_provenance.precision, None);
        assert_eq!(without_provenance.model_source, None);
        assert_eq!(without_provenance.model_revision, None);
    }

    /// Missing required field `models_dir` → error.
    #[test]
    fn test_config_missing_required_field_errors() {
        let json = r#"{"device": "GPU.1", "total_vram_gb": 22.5}"#;
        let result: Result<Config, _> = serde_json::from_str(json);
        assert!(
            result.is_err(),
            "missing models_dir must fail deserialisation"
        );
    }

    // ---- T4.1: unknown-key handling ------------------------------------

    fn minimal(extra: &str) -> Config {
        let json =
            format!(r#"{{"models_dir": "/tmp/m", "device": "CPU", "total_vram_gb": 0.0{extra}}}"#);
        serde_json::from_str(&json).unwrap()
    }

    /// Test gap (firmness review): every numeric setting `--print-default-config` prints, set to
    /// `0`, either validates (and CONFIG.md says what 0 does) or is refused at boot. A new numeric
    /// knob fails this test until it is classified here, so the next `0`-panics knob is caught.
    #[test]
    #[allow(clippy::too_many_lines)] // the two classification tables are the point
    fn every_numeric_knob_at_zero_is_documented_or_refused() {
        // Valid at 0; the comment is what 0 means (CONFIG.md says the same).
        const ALLOWED: &[&str] = &[
            "total_vram_gb",                           // the device's own memory / no gating
            "max_num_seqs",                            // OV default (256)
            "cache_size_gb",                           // uncapped KV pool
            "default_kv_cache_gb",                     // unbounded
            "vram_safety_margin_gb",                   // no margin
            "min_kv_cache_gb",                         // no floor
            "light_model_max_gb",                      // every LLM/VLM is "heavy"
            "light_stt_max_gb",                        // no effect (kept for compatibility)
            "dgpu_size_ceiling_fraction",              // default placement skips every dGPU
            "eviction_grace_secs",                     // no grace window
            "admission_queue_timeout_ms",              // no queueing: 429 at once when full
            "device_admission_queue_timeout_ms",       // the same for device_budgets
            "max_embedding_batch_tokens",              // no budget
            "max_tokens_cap",                          // uncapped
            "stream_first_token_timeout_secs",         // off
            "stream_token_gap_timeout_secs",           // off
            "port",                                    // the OS picks a free port
            "ov_cache_max_gb",                         // unbounded
            "kv_headroom_margin_gib",                  // no margin
            "kv_headroom_inflight_tokens",             // every vision request holds a reservation
            "kv_pressure_threshold_pct",               // unset (refused only with the monitor on)
            "kv_pressure_sustained_secs",              // unset (refused only with the monitor on)
            "kv_resize_cooldown_secs",                 // no cooldown
            "load_timeout_secs",                       // no deadline
            "on_demand_wait_secs",                     // no wait: 503 + Retry-After at once
            "supervisor.backoff_base_secs",            // respawn without delay
            "supervisor.backoff_max_secs",             // respawn without delay
            "supervisor.hang_kill_grace_secs",         // floored at 1 s
            "supervisor.watchdog_hang_ceiling_secs",   // generation watchdog off
            "supervisor.gpu_first_load_deadline_secs", // off
            "supervisor.gpu_probe_deadline_secs",      // probe off
        ];
        // Refused at 0 by `Config::validate` (exit 78 at boot).
        const REFUSED: &[&str] = &[
            "npu_footprint_factor",
            "cpu_image_gen_max_pixels",
            "max_audio_duration_secs",
            "max_images_per_request",
            "max_image_pixels_per_request",
            "max_image_pixels",
            "max_prompt_array",
            "max_embedding_inputs",
            "ov_cache_sweep_interval_secs",
            "kv_transient_factor",
            "kv_headroom_min_tokens",
            "kv_pressure_sweep_interval_secs",
            "supervisor.max_restarts",
            "supervisor.restart_window_secs",
            "supervisor.health_timeout_secs",
            "supervisor.watchdog_poll_secs",
        ];
        fn numeric_paths(v: &serde_json::Value, prefix: &str, out: &mut Vec<String>) {
            if let Some(map) = v.as_object() {
                for (k, child) in map {
                    let path = if prefix.is_empty() {
                        k.clone()
                    } else {
                        format!("{prefix}.{k}")
                    };
                    if child.is_number() {
                        out.push(path);
                    } else {
                        numeric_paths(child, &path, out);
                    }
                }
            }
        }
        let template = Config::default_template().unwrap();
        assert!(
            serde_json::from_value::<Config>(template.clone())
                .unwrap()
                .validate()
                .is_ok(),
            "the template itself must validate"
        );
        let mut paths = Vec::new();
        numeric_paths(&template, "", &mut paths);
        let unclassified: Vec<&String> = paths
            .iter()
            .filter(|p| !ALLOWED.contains(&p.as_str()) && !REFUSED.contains(&p.as_str()))
            .collect();
        assert!(
            unclassified.is_empty(),
            "classify these numeric settings at 0 (ALLOWED with a CONFIG.md line, or refused \
             in Config::validate): {unclassified:?}"
        );
        for name in ALLOWED.iter().chain(REFUSED) {
            assert!(
                paths.iter().any(|p| p == name),
                "{name} is not in the template"
            );
        }
        for path in &paths {
            let mut v = template.clone();
            let mut slot = &mut v;
            for part in path.split('.') {
                slot = slot.get_mut(part).unwrap();
            }
            *slot = serde_json::json!(0);
            let result = serde_json::from_value::<Config>(v)
                .map_err(anyhow::Error::from)
                .and_then(|c| c.validate());
            if REFUSED.contains(&path.as_str()) {
                let err = result.err().map(|e| e.to_string());
                let err = err.unwrap_or_else(|| panic!("{path} at 0 must be refused"));
                let leaf = path.rsplit('.').next().unwrap();
                assert!(
                    err.contains(leaf),
                    "{path}: the error must name the key: {err}"
                );
            } else {
                assert!(result.is_ok(), "{path} at 0 must validate: {result:?}");
            }
        }
    }

    /// FM-4: every config the repo ships (`scripts/*.config.json`: the bundle's example and the
    /// fleet boxes') still loads and validates under the new interval rules.
    #[test]
    fn every_shipped_config_validates() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts");
        let mut checked = 0;
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            if !name.ends_with(".config.json") {
                continue;
            }
            Config::load(&path).unwrap_or_else(|e| panic!("{name}: {e:#}"));
            checked += 1;
        }
        assert!(
            checked >= 7,
            "expected the shipped configs, found {checked}"
        );
    }

    /// FM-4: the supervisor seconds fields and both sweep intervals are bounded at one year, so
    /// the supervisor's sums and `Instant` additions cannot overflow.
    #[test]
    fn interval_knobs_have_an_upper_bound() {
        for key in [
            "\"supervisor\": {\"backoff_max_secs\": X}",
            "\"supervisor\": {\"health_timeout_secs\": X}",
            "\"supervisor\": {\"watchdog_hang_ceiling_secs\": X}",
            "\"supervisor\": {\"gpu_probe_deadline_secs\": X}",
            "\"ov_cache_sweep_interval_secs\": X",
            "\"kv_pressure_sweep_interval_secs\": X",
        ] {
            let ok = minimal(&format!(
                ", {}",
                key.replace('X', &MAX_INTERVAL_SECS.to_string())
            ));
            assert!(ok.validate().is_ok(), "{key} at the bound");
            for over in [MAX_INTERVAL_SECS + 1, u64::MAX] {
                let cfg = minimal(&format!(", {}", key.replace('X', &over.to_string())));
                let err = cfg.validate().unwrap_err().to_string();
                assert!(err.contains("at most"), "{key}={over}: {err}");
            }
        }
    }

    /// A typo of a security-critical key (`api_key`, `cors_origins`, …) is a
    /// hard error — the permissive defaults must never silently take over.
    #[test]
    fn security_adjacent_unknown_key_is_a_hard_error() {
        for typo in [
            r#", "api_key": ["sk-x"]"#,
            r#", "cors_origins": ["*"]"#,
            r#", "bindaddr": "0.0.0.0""#,
        ] {
            let cfg = minimal(typo);
            let err = cfg.check_unknown_fields().unwrap_err().to_string();
            assert!(
                err.contains("security-related"),
                "typo {typo:?} must refuse startup: {err}"
            );
        }
    }

    /// An innocuous unknown key is tolerated (warned, not fatal).
    #[test]
    fn innocuous_unknown_key_is_tolerated() {
        let cfg = minimal(r#", "comment": "my dev box""#);
        assert!(cfg.check_unknown_fields().is_ok());
        assert!(cfg.unknown_fields.contains_key("comment"));
    }

    /// A clean config has no unknown fields.
    #[test]
    fn clean_config_has_no_unknown_fields() {
        let cfg = minimal("");
        assert!(cfg.unknown_fields.is_empty());
        assert!(cfg.check_unknown_fields().is_ok());
    }

    // ---- T4.2: public-bind security gate --------------------------------

    /// Default bind is loopback:11437 — secure by default.
    #[test]
    fn default_bind_is_loopback() {
        let cfg = minimal("");
        let addr = cfg.validated_bind().unwrap();
        assert!(addr.ip().is_loopback());
        assert_eq!(addr.port(), 11_437);
    }

    /// Public bind + no keys refuses to start unless explicitly opted in.
    /// `enforce_open_bind_gate` takes the key presence as explicit bools (the
    /// caller resolves them from the keys file, not from `Config` — see its doc
    /// comment) — no need to construct a `Config` with keys in it to test this.
    #[test]
    fn public_bind_without_auth_refuses_to_start() {
        let cfg = minimal(r#", "bind_addr": "0.0.0.0""#);
        let addr = cfg.validated_bind().unwrap();
        let err = cfg
            .enforce_open_bind_gate(addr, false, false)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("refusing to start") && err.contains("no keys are configured"),
            "open public bind must abort: {err}"
        );
    }

    /// 0.9.1 S3 (I-05): admin keys alone do not protect inference, so a public bind
    /// with an admin-only keys file (what `rv start` creates) is refused, and the
    /// refusal says why.
    #[test]
    fn public_bind_with_only_admin_keys_refuses_to_start() {
        let cfg = minimal(r#", "bind_addr": "0.0.0.0""#);
        let addr = cfg.validated_bind().unwrap();
        let err = cfg
            .enforce_open_bind_gate(addr, false, true)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("refusing to start") && err.contains("no api_keys"),
            "admin-only keys on a public bind must abort: {err}"
        );
        // An IPv6 wildcard is a network bind too.
        let v6 = minimal(r#", "bind_addr": "::""#);
        let addr = v6.validated_bind().unwrap();
        assert!(v6.enforce_open_bind_gate(addr, false, true).is_err());
    }

    /// Public bind is fine with inference keys (with or without admin keys), or with
    /// the explicit insecure opt-in even with no keys or admin keys only. Loopback
    /// never needs keys.
    #[test]
    fn public_bind_allowed_with_keys_or_explicit_opt_in() {
        let cfg = minimal(r#", "bind_addr": "0.0.0.0""#);
        let addr = cfg.validated_bind().unwrap();
        assert!(cfg.enforce_open_bind_gate(addr, true, false).is_ok());
        assert!(cfg.enforce_open_bind_gate(addr, true, true).is_ok());

        let opted_in = minimal(r#", "bind_addr": "0.0.0.0", "allow_insecure_public_bind": true"#);
        let addr = opted_in.validated_bind().unwrap();
        assert!(opted_in.enforce_open_bind_gate(addr, false, false).is_ok());
        assert!(opted_in.enforce_open_bind_gate(addr, false, true).is_ok());

        let loopback = minimal("");
        let addr = loopback.validated_bind().unwrap();
        assert!(loopback.enforce_open_bind_gate(addr, false, true).is_ok());
        assert!(loopback.enforce_open_bind_gate(addr, false, false).is_ok());
    }

    /// A non-empty `api_keys`/`admin_api_keys` in `config.json` is a
    /// migration tripwire, not something `Config` itself enforces — that's
    /// `startup::bootstrap`'s job (it has the resolved keys-file path to name
    /// in the error). `Config` just keeps parsing them structurally.
    #[test]
    fn deprecated_inline_keys_still_parse_but_are_not_the_auth_source() {
        let cfg = minimal(r#", "api_keys": ["sk-old"], "admin_api_keys": ["sk-old-admin"]"#);
        assert_eq!(cfg.api_keys, vec!["sk-old".to_string()]);
        assert_eq!(cfg.admin_api_keys, vec!["sk-old-admin".to_string()]);
    }

    // ---- resolve_keys_file_path -----------------------------------------

    /// Default derivation strips `.config.json` and appends `.keys.json` —
    /// the documented example (`myserver.config.json` → `myserver.keys.json`).
    #[test]
    fn keys_file_path_default_strips_config_json_suffix() {
        let path = resolve_keys_file_path(Path::new("/opt/rv/scripts/myserver.config.json"), None);
        assert_eq!(path, Path::new("/opt/rv/scripts/myserver.keys.json"));
    }

    /// A config file not following the `*.config.json` convention falls back
    /// to stripping bare `.json`.
    #[test]
    fn keys_file_path_default_strips_bare_json_suffix() {
        let path = resolve_keys_file_path(Path::new("/opt/rv/config.json"), None);
        assert_eq!(path, Path::new("/opt/rv/config.keys.json"));
    }

    /// An explicit relative `keys_file` resolves against the config file's
    /// own directory, never the process cwd.
    #[test]
    fn keys_file_path_explicit_relative_resolves_against_config_dir() {
        let path = resolve_keys_file_path(
            Path::new("/opt/rv/scripts/myserver.config.json"),
            Some(Path::new("secrets/keys.json")),
        );
        assert_eq!(path, Path::new("/opt/rv/scripts/secrets/keys.json"));
    }

    /// An explicit absolute `keys_file` is used verbatim.
    #[test]
    fn keys_file_path_explicit_absolute_used_as_is() {
        let path = resolve_keys_file_path(
            Path::new("/opt/rv/scripts/myserver.config.json"),
            Some(Path::new("/etc/rustedvino/keys.json")),
        );
        assert_eq!(path, Path::new("/etc/rustedvino/keys.json"));
    }

    /// A `bind_addr` that is not an IP address is a clear startup error.
    #[test]
    fn invalid_bind_addr_errors() {
        let cfg = minimal(r#", "bind_addr": "not-an-ip""#);
        assert!(cfg.validated_bind().is_err());
    }

    // ---- T4.4: numeric semantic validation -----------------------------

    /// Build a config from a full JSON body (no duplicate-key reliance).
    fn from_json(json: &str) -> Config {
        serde_json::from_str(json).unwrap()
    }

    /// A clean config passes numeric validation.
    #[test]
    fn valid_numeric_config_passes_validation() {
        let cfg = from_json(
            r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0,"models":{"m":{"vram_gb":5.5}}}"#,
        );
        assert!(cfg.validate().is_ok());
    }

    /// 0.9.1 C28: `max_image_pixels` defaults to 4 000 000 (also what `--print-default-config`,
    /// and so `scripts/fill-config-defaults.py`, writes out); 0 and anything under 65 536 are
    /// refused at load; a value above `max_image_pixels_per_request` is allowed (never fires).
    #[test]
    fn max_image_pixels_default_and_validation() {
        let base = r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0"#;
        let cfg = from_json(&format!("{base}}}"));
        assert_eq!(cfg.max_image_pixels, 4_000_000);
        assert_eq!(
            Config::default_template().unwrap()["max_image_pixels"],
            serde_json::json!(4_000_000)
        );
        for (v, ok) in [
            (0_u64, false),
            (65_535, false),
            (65_536, true),
            (4_000_000, true),
            (1 << 40, true),
        ] {
            let cfg = from_json(&format!(r#"{base},"max_image_pixels":{v}}}"#));
            let r = cfg.validate();
            assert_eq!(r.is_ok(), ok, "{v}: {r:?}");
            if let Err(e) = r {
                assert!(e.to_string().contains("max_image_pixels"), "{e}");
            }
        }
    }

    /// 0.9.1 S1: the stall watchdog keys default to 600 / 120 s (also what
    /// `--print-default-config`, and so `scripts/fill-config-defaults.py`, writes out); `0` turns
    /// each one off and is accepted, like any other value; both are restart-only.
    #[test]
    fn stream_watchdog_keys_default_and_zero() {
        let base = r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0"#;
        let cfg = from_json(&format!("{base}}}"));
        assert_eq!(cfg.stream_first_token_timeout_secs, 600);
        assert_eq!(cfg.stream_token_gap_timeout_secs, 120);
        let tpl = Config::default_template().unwrap();
        assert_eq!(
            tpl["stream_first_token_timeout_secs"],
            serde_json::json!(600)
        );
        assert_eq!(tpl["stream_token_gap_timeout_secs"], serde_json::json!(120));
        let cfg = from_json(&format!(
            r#"{base},"stream_first_token_timeout_secs":0,"stream_token_gap_timeout_secs":0}}"#
        ));
        assert!(cfg.validate().is_ok());
        assert_eq!(
            (
                cfg.stream_first_token_timeout_secs,
                cfg.stream_token_gap_timeout_secs
            ),
            (0, 0)
        );
        let mut file = from_json(&format!("{base}}}"));
        let boot = from_json(&format!("{base}}}"));
        file.stream_first_token_timeout_secs = 5;
        file.stream_token_gap_timeout_secs = 2;
        assert_eq!(
            globals_not_applied(&boot, &file),
            vec![
                "stream_first_token_timeout_secs".to_owned(),
                "stream_token_gap_timeout_secs".to_owned()
            ]
        );
    }

    /// Negative `total_vram_gb` is rejected (would inflate the usable pool).
    #[test]
    fn negative_total_vram_is_rejected() {
        let cfg = from_json(r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":-1.0}"#);
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("total_vram_gb"), "must name the knob: {err}");
    }

    /// Negative `ov_cache_max_gb` is rejected — same finite/non-negative
    /// discipline as every other capacity knob.
    #[test]
    fn negative_ov_cache_max_gb_is_rejected() {
        let cfg = from_json(
            r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0,"ov_cache_max_gb":-1.0}"#,
        );
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("ov_cache_max_gb"), "must name the knob: {err}");
    }

    /// `ov_cache_max_gb` defaults to `0.0` (unbounded, mirroring
    /// `total_vram_gb`'s "0 disables gating" convention) and
    /// `ov_cache_sweep_interval_secs` defaults to 6h when the config omits
    /// them — an existing config file with neither key must keep working.
    #[test]
    fn ov_cache_knobs_default_when_omitted() {
        let cfg = from_json(r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0}"#);
        assert!(cfg.ov_cache_max_gb.abs() < 1e-9);
        assert_eq!(cfg.ov_cache_sweep_interval_secs, 21_600);
    }

    /// `ov_cache_layout` (U-C35): `keyed` by default, `flat` accepted, anything else refused with
    /// the accepted values named; a reload that changes it is reported as not applied.
    #[test]
    fn ov_cache_layout_accepts_keyed_and_flat_only() {
        let base = |extra: &str| {
            format!(r#"{{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0{extra}}}"#)
        };
        assert_eq!(from_json(&base("")).ov_cache_layout, "keyed");
        assert_eq!(
            from_json(&base(r#","ov_cache_layout":"flat""#)).ov_cache_layout,
            "flat"
        );
        for bad in ["bogus", "Keyed", ""] {
            let cfg: Config =
                serde_json::from_str(&base(&format!(r#","ov_cache_layout":"{bad}""#))).unwrap();
            let err = cfg.validate().unwrap_err().to_string();
            assert!(
                err.contains("ov_cache_layout") && err.contains("keyed"),
                "{err}"
            );
        }
    }

    /// `kv_precision_pass`: `explicit` by default, `plugin` accepted, anything else refused with the
    /// accepted values named; a wrong JSON type names the key; a reload that changes it is reported
    /// as not applied.
    #[test]
    fn kv_precision_pass_accepts_explicit_and_plugin_only() {
        let base = |extra: &str| {
            format!(r#"{{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0{extra}}}"#)
        };
        assert_eq!(from_json(&base("")).kv_precision_pass, "explicit");
        assert_eq!(
            from_json(&base(r#","kv_precision_pass":"plugin""#)).kv_precision_pass,
            "plugin"
        );
        for bad in ["bogus", "Explicit", ""] {
            let cfg: Config =
                serde_json::from_str(&base(&format!(r#","kv_precision_pass":"{bad}""#))).unwrap();
            let err = cfg.validate().unwrap_err().to_string();
            assert!(
                err.contains("kv_precision_pass") && err.contains("explicit"),
                "{err}"
            );
        }
        let err = serde_json::from_str::<Config>(&base(r#","kv_precision_pass":true"#))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("kv_precision_pass") && err.contains("must be a string"),
            "{err}"
        );
        let boot = from_json(&base(""));
        let file = from_json(&base(r#","kv_precision_pass":"plugin""#));
        assert_eq!(
            globals_not_applied(&boot, &file),
            vec!["kv_precision_pass".to_owned()]
        );
    }

    /// `kv_gate_mode` / `kv_gate_exact_on_unknown_engine`: defaults, accepted values, the refusal naming
    /// the accepted list, and the reload report.
    #[test]
    fn kv_gate_keys_default_validate_and_report_changes() {
        let base = |extra: &str| {
            format!(r#"{{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0{extra}}}"#)
        };
        let d = from_json(&base(""));
        assert_eq!(
            (d.kv_gate_mode.as_str(), d.kv_gate_exact_on_unknown_engine),
            ("exact", false)
        );
        let l = from_json(&base(
            r#","kv_gate_mode":"legacy","kv_gate_exact_on_unknown_engine":true"#,
        ));
        assert_eq!(
            (l.kv_gate_mode.as_str(), l.kv_gate_exact_on_unknown_engine),
            ("legacy", true)
        );
        for bad in ["bogus", "Exact", ""] {
            let cfg: Config =
                serde_json::from_str(&base(&format!(r#","kv_gate_mode":"{bad}""#))).unwrap();
            let err = cfg.validate().unwrap_err().to_string();
            assert!(
                err.contains("kv_gate_mode") && err.contains("legacy"),
                "{err}"
            );
        }
        let err = serde_json::from_str::<Config>(&base(r#","kv_gate_mode":1"#))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("kv_gate_mode") && err.contains("must be a string"),
            "{err}"
        );
        let mut changed = globals_not_applied(&d, &l);
        changed.sort();
        assert_eq!(
            changed,
            vec![
                "kv_gate_exact_on_unknown_engine".to_owned(),
                "kv_gate_mode".to_owned()
            ]
        );
    }

    /// A wrong JSON type for these two keys names the key (family-smoke B1/B2: serde's own
    /// message was only `invalid type: null, expected a string`).
    #[test]
    fn wrong_type_for_precision_or_layout_names_the_key() {
        for (key, json) in [
            ("kv_cache_precision", r#""kv_cache_precision":null"#),
            ("kv_cache_precision", r#""kv_cache_precision":8"#),
            ("kv_cache_precision", r#""kv_cache_precision":["u4"]"#),
            ("ov_cache_layout", r#""ov_cache_layout":null"#),
            ("ov_cache_layout", r#""ov_cache_layout":1"#),
        ] {
            let text =
                format!(r#"{{"models_dir":"/m","device":"CPU","total_vram_gb":1.0,{json}}}"#);
            let err = serde_json::from_str::<Config>(&text)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains(key) && err.contains("must be a string"),
                "{err}"
            );
        }
    }

    /// Unit-level half of U-C35: a changed `ov_cache_layout` is named by `globals_not_applied`.
    #[test]
    fn globals_not_applied_names_a_changed_cache_layout() {
        let boot = from_json(r#"{"models_dir":"/m","device":"GPU","total_vram_gb":22.5}"#);
        let file = from_json(
            r#"{"models_dir":"/m","device":"GPU","total_vram_gb":22.5,"ov_cache_layout":"flat"}"#,
        );
        assert_eq!(
            globals_not_applied(&boot, &file),
            vec!["ov_cache_layout".to_owned()]
        );
    }

    /// `kv_pressure_monitor_enabled` defaults to `false`, and the threshold/
    /// duration default to their "not configured" sentinels (`0.0`/`0`) —
    /// an existing config file with none of these keys must keep working
    /// exactly as before this feature existed
    /// (the project's internal engineering log's ops-review finding:
    /// default off, not an incidentally-safe threshold).
    #[test]
    fn kv_pressure_knobs_default_off_when_omitted() {
        let cfg = from_json(r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0}"#);
        assert!(!cfg.kv_pressure_monitor_enabled);
        assert!(cfg.kv_pressure_threshold_pct.abs() < 1e-9);
        assert_eq!(cfg.kv_pressure_sustained_secs, 0);
        assert_eq!(cfg.kv_pressure_sweep_interval_secs, 15);
        assert_eq!(cfg.kv_resize_cooldown_secs, 30);
    }

    /// Enabling the monitor without setting a real threshold is rejected —
    /// the whole point of the no-usable-default design: a box that turns
    /// this on must state its own number, not silently inherit `0.0`, which
    /// would flag every model instantly.
    #[test]
    fn kv_pressure_monitor_enabled_without_threshold_is_rejected() {
        let cfg = from_json(
            r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0,
                "kv_pressure_monitor_enabled":true,"kv_pressure_sustained_secs":60}"#,
        );
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("kv_pressure_threshold_pct"),
            "must name the knob: {err}"
        );
    }

    /// Same as above, for the sustained-duration half of the pair — both
    /// must be set together, not just one of the two.
    #[test]
    fn kv_pressure_monitor_enabled_without_sustained_secs_is_rejected() {
        let cfg = from_json(
            r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0,
                "kv_pressure_monitor_enabled":true,"kv_pressure_threshold_pct":90.0}"#,
        );
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("kv_pressure_sustained_secs"),
            "must name the knob: {err}"
        );
    }

    /// A fully-configured monitor (both knobs set) validates cleanly.
    #[test]
    fn kv_pressure_monitor_fully_configured_is_accepted() {
        let cfg = from_json(
            r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0,
                "kv_pressure_monitor_enabled":true,"kv_pressure_threshold_pct":90.0,
                "kv_pressure_sustained_secs":60}"#,
        );
        cfg.validate().unwrap();
    }

    /// `system_ram_budget_gb` is absent by default, which must keep the live
    /// admission gate on its historical `MemAvailable`-only reading — the
    /// GPU-page-cache credit is the less conservative direction and is opt-in.
    #[test]
    fn system_ram_budget_is_absent_by_default() {
        let cfg = from_json(r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0}"#);
        assert!(cfg.system_ram_budget_gb.is_none());
        cfg.validate().unwrap();
    }

    /// A declared budget is accepted and survives round-tripping.
    #[test]
    fn system_ram_budget_positive_is_accepted() {
        let cfg = from_json(
            r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0,
                "system_ram_budget_gb":18.0}"#,
        );
        cfg.validate().unwrap();
        assert!(
            cfg.system_ram_budget_gb
                .is_some_and(|b| (b - 18.0).abs() < 1e-9)
        );
    }

    /// `0.0` is rejected rather than silently meaning "unset": a budget of zero
    /// would cap credited availability at zero and refuse every system-domain
    /// load, which is never what an operator means. Omit the field instead.
    #[test]
    fn system_ram_budget_zero_is_rejected() {
        let cfg = from_json(
            r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0,
                "system_ram_budget_gb":0.0}"#,
        );
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("system_ram_budget_gb"),
            "error must name the offending knob, got: {err}"
        );
    }

    /// Negative and non-finite budgets are rejected for the same reason the
    /// sibling RAM knobs are: a NaN defeats every comparison in the gate.
    #[test]
    fn system_ram_budget_negative_is_rejected() {
        let cfg = from_json(
            r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0,
                "system_ram_budget_gb":-4.0}"#,
        );
        assert!(cfg.validate().is_err());
    }

    /// A threshold above 100% is nonsensical for a percentage and rejected
    /// regardless of whether the monitor is enabled.
    #[test]
    fn kv_pressure_threshold_over_100_is_rejected() {
        let cfg = from_json(
            r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0,
                "kv_pressure_threshold_pct":150.0}"#,
        );
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("kv_pressure_threshold_pct"),
            "must name the knob: {err}"
        );
    }

    /// A non-finite VRAM knob (`NaN`) is rejected — every comparison against
    /// `NaN` is false, which silently disables the capacity gate (finding 33).
    #[test]
    fn non_finite_vram_knob_is_rejected() {
        let mut cfg = from_json(r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0}"#);
        cfg.total_vram_gb = f64::NAN;
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("finite"), "must reject non-finite: {err}");
    }

    /// A negative `system_ram_reservation_gb` is rejected (would inflate the
    /// system-domain pool past total RAM).
    #[test]
    fn negative_system_ram_reservation_is_rejected() {
        let cfg = from_json(
            r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":0.0,"system_ram_reservation_gb":-2.0}"#,
        );
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("system_ram_reservation_gb"),
            "must name the knob: {err}"
        );
    }

    /// A negative `domain_budgets` entry is rejected (would defeat that domain's
    /// capacity gate).
    #[test]
    fn negative_domain_budget_is_rejected() {
        let cfg = from_json(
            r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":0.0,"domain_budgets":{"system":-1.0}}"#,
        );
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("domain_budgets['system']"),
            "must name the domain: {err}"
        );
    }

    /// A negative per-model `vram_gb` entry is rejected.
    #[test]
    fn negative_per_model_vram_is_rejected() {
        let cfg = from_json(
            r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0,"models":{"bad":{"vram_gb":-3.0}}}"#,
        );
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("models['bad'].vram_gb"),
            "must name the model: {err}"
        );
    }

    /// Reserved margins that exceed total VRAM are rejected — no model could
    /// ever load, so the server would be a silent no-op.
    #[test]
    fn margins_exceeding_total_vram_are_rejected() {
        let cfg = from_json(
            r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":2.0,"vram_safety_margin_gb":1.5,"min_kv_cache_gb":1.5}"#,
        );
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("margins inconsistent"),
            "must flag margins: {err}"
        );
    }

    /// T5.5/#34: a preload model with no `models` entry is rejected at
    /// validation — it would otherwise register at a silent 0.0 GB and load
    /// uncounted (over-subscription / OOM risk).
    #[test]
    fn preload_without_vram_estimate_is_rejected() {
        let cfg = from_json(
            r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0,"preload":["ghost"],"models":{"other":{"vram_gb":5.5}}}"#,
        );
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("preload['ghost']"),
            "must name the unsized preload model: {err}"
        );
    }

    /// A preload model that DOES have a `models` entry validates cleanly.
    #[test]
    fn preload_with_vram_estimate_passes() {
        let cfg = from_json(
            r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0,"preload":["m"],"models":{"m":{"vram_gb":5.5}}}"#,
        );
        assert!(cfg.validate().is_ok());
    }

    /// Gating OFF (`total_vram_gb` = 0.0) skips the margin-consistency check —
    /// margins are irrelevant when there is no pool to gate.
    #[test]
    fn margins_not_checked_when_gating_disabled() {
        let cfg = from_json(
            r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":0.0,"vram_safety_margin_gb":100.0,"min_kv_cache_gb":100.0}"#,
        );
        assert!(cfg.validate().is_ok(), "no gating → margins irrelevant");
    }

    /// `preload` key absent → defaults to empty Vec.
    #[test]
    fn test_config_defaults_preload_to_empty() {
        let json = r#"{
            "models_dir": "/tmp/models",
            "device": "CPU",
            "total_vram_gb": 0.0
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!(cfg.preload.is_empty(), "preload must default to []");
    }

    /// `models` key absent → defaults to empty `HashMap`.
    #[test]
    fn test_config_defaults_models_to_empty() {
        let json = r#"{
            "models_dir": "/tmp/models",
            "device": "CPU",
            "total_vram_gb": 0.0
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!(cfg.models.is_empty(), "models must default to {{}}");
    }

    /// `cors_allowed_origins` key absent → defaults to `["*"]` (permissive).
    #[test]
    fn test_config_defaults_cors_origins_to_wildcard() {
        let json = r#"{
            "models_dir": "/tmp/models",
            "device": "CPU",
            "total_vram_gb": 0.0
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert_eq!(
            cfg.cors_allowed_origins,
            vec!["*".to_owned()],
            "cors_allowed_origins must default to [\"*\"]"
        );
    }

    /// `cors_allowed_origins` present → parsed as the explicit list.
    #[test]
    fn test_config_parses_explicit_cors_origins() {
        let json = r#"{
            "models_dir": "/tmp/models",
            "device": "CPU",
            "total_vram_gb": 0.0,
            "cors_allowed_origins": ["https://a.example.com", "https://b.example.com"]
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert_eq!(
            cfg.cors_allowed_origins,
            vec![
                "https://a.example.com".to_owned(),
                "https://b.example.com".to_owned()
            ]
        );
    }

    /// `api_keys` key absent → defaults to empty (auth OFF / open).
    #[test]
    fn test_config_defaults_api_keys_to_empty() {
        let json = r#"{
            "models_dir": "/tmp/models",
            "device": "CPU",
            "total_vram_gb": 0.0
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!(
            cfg.api_keys.is_empty(),
            "api_keys must default to [] (auth off)"
        );
    }

    /// `api_keys` present → parsed as the explicit key list.
    #[test]
    fn test_config_parses_explicit_api_keys() {
        let json = r#"{
            "models_dir": "/tmp/models",
            "device": "CPU",
            "total_vram_gb": 0.0,
            "api_keys": ["sk-alpha-1", "sk-alpha-2"]
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert_eq!(
            cfg.api_keys,
            vec!["sk-alpha-1".to_owned(), "sk-alpha-2".to_owned()]
        );
    }

    /// `admin_api_keys` key absent → defaults to empty (no separate admin
    /// scope; admin routes fall back to the inference gate).
    #[test]
    fn test_config_defaults_admin_api_keys_to_empty() {
        let cfg = minimal("");
        assert!(
            cfg.admin_api_keys.is_empty(),
            "admin_api_keys must default to [] (shared scope)"
        );
    }

    /// `admin_api_keys` present → parsed as the explicit admin key list.
    #[test]
    fn test_config_parses_explicit_admin_api_keys() {
        let cfg = minimal(r#", "admin_api_keys": ["sk-admin-1", "sk-admin-2"]"#);
        assert_eq!(
            cfg.admin_api_keys,
            vec!["sk-admin-1".to_owned(), "sk-admin-2".to_owned()]
        );
    }

    /// Embedding knobs default correctly when absent: pooling=mean,
    /// normalize=true, no device override, empty models map.
    #[test]
    fn test_config_embedding_defaults() {
        let json = r#"{
            "models_dir": "/tmp/models",
            "device": "GPU.1",
            "total_vram_gb": 0.0
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.embedding_pooling, "mean");
        assert!(cfg.embedding_normalize);
        assert_eq!(cfg.embedding_device, None);
        assert!(cfg.models.is_empty());
    }

    /// `kind` in models entry + `embedding_device` parse as declared.
    #[test]
    fn test_config_parses_model_kinds_and_embedding_device() {
        let json = r#"{
            "models_dir": "/tmp/models",
            "device": "GPU.1",
            "total_vram_gb": 0.0,
            "models": {"multilingual-e5-large-int8": {"vram_gb": 0.0, "kind": "embedding"}},
            "embedding_device": "GPU.0",
            "embedding_pooling": "cls",
            "embedding_normalize": false
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert_eq!(
            cfg.models
                .get("multilingual-e5-large-int8")
                .and_then(|e| e.kind.as_deref()),
            Some("embedding")
        );
        assert_eq!(cfg.embedding_device.as_deref(), Some("GPU.0"));
        assert_eq!(cfg.embedding_pooling, "cls");
        assert!(!cfg.embedding_normalize);
    }

    /// `models` absent → defaults to empty map.
    #[test]
    fn model_policies_default_empty() {
        let cfg = minimal("");
        assert!(cfg.models.is_empty(), "models must default to {{}}");
    }

    /// A models entry parses with its declared `pinned` / `priority`,
    /// and an entry with only `pinned` defaults `priority` to 0.
    #[test]
    fn model_policies_parse_pinned_and_priority() {
        let cfg = minimal(
            r#", "models": {"chat": {"vram_gb": 0.0, "pinned": true, "priority": 10}, "embed": {"vram_gb": 0.0, "pinned": true}}"#,
        );
        let chat = &cfg.models["chat"].policy;
        assert!(chat.pinned);
        assert_eq!(chat.priority, 10);
        let embed = &cfg.models["embed"].policy;
        assert!(embed.pinned);
        assert_eq!(embed.priority, 0, "priority must default to 0 when absent");
    }

    /// An empty models entry (`{"vram_gb": 0.0}`) yields the pre-feature defaults.
    #[test]
    fn model_policy_empty_block_uses_defaults() {
        let cfg = minimal(r#", "models": {"m": {"vram_gb": 0.0}}"#);
        let p = &cfg.models["m"].policy;
        assert!(!p.pinned, "pinned defaults false");
        assert_eq!(p.priority, 0, "priority defaults 0");
        // Slice 2 knobs default to None (global behaviour).
        assert_eq!(p.kv_cache_gb, None, "kv_cache_gb defaults None");
        assert_eq!(
            p.max_concurrent_streams, None,
            "max_concurrent_streams defaults None"
        );
        assert_eq!(p.max_prompt_len, None, "max_prompt_len defaults None");
        // Slice 3 knobs default to the pre-feature behaviour.
        assert_eq!(p.load, LoadPolicy::Eager, "load defaults Eager");
        assert!(p.evictable, "evictable defaults true");
        assert!(p.speculative.is_none(), "speculative defaults None");
    }

    // ---- Speculative decoding config schema (SpeculativeConfig) -----------

    /// A valid `speculative` block deserialises through the
    /// `#[serde(flatten)]` path (`ModelPolicy` flattened into `ModelEntry`),
    /// its own `#[serde(default)]` fields fill in, and `Config::validate()`
    /// accepts it.
    #[test]
    fn speculative_block_parses_and_validates() {
        let entry: ModelEntry = serde_json::from_str(
            r#"{"vram_gb": 15.0, "speculative": {"draft_model": "d", "draft_vram_gb": 0.8}}"#,
        )
        .unwrap();
        let spec = entry.policy.speculative.as_ref().unwrap();
        assert_eq!(spec.draft_model, "d");
        assert!((spec.draft_vram_gb - 0.8).abs() < 1e-9);
        assert_eq!(spec.draft_device, None, "draft_device defaults None");
        assert_eq!(
            spec.num_assistant_tokens, 5,
            "num_assistant_tokens defaults to 5"
        );
        assert!(spec.verify_on_load, "verify_on_load defaults true");

        let cfg = minimal(
            r#", "models": {"t": {"vram_gb": 15.0, "speculative": {"draft_model": "d", "draft_vram_gb": 0.8}}}"#,
        );
        assert!(cfg.validate().is_ok());
    }

    /// `speculative` on a `vram_gb: 0.0` (unmetered) target is refused —
    /// the draft's VRAM would never get charged (Fable-flagged gap, folded
    /// into Migration step 2).
    #[test]
    fn speculative_on_zero_vram_target_fails_validate() {
        let cfg = minimal(
            r#", "models": {"t": {"vram_gb": 0.0, "speculative": {"draft_model": "d", "draft_vram_gb": 0.8}}}"#,
        );
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("vram_gb > 0.0"),
            "must name the zero-vram gate: {err}"
        );
    }

    /// FM-7: `speculative.draft_model` is joined onto `models_dir` like a model id, so it gets the
    /// same plain-folder-name check (a `../x`, an absolute path or a drive letter would read model
    /// files outside `models_dir`).
    #[test]
    fn speculative_draft_model_must_be_a_plain_folder_name() {
        for bad in ["../x", "/abs/draft", "a/b", "a\\b", "..", "C:", "c:draft"] {
            let mut e: ModelEntry = serde_json::from_str(
                r#"{"vram_gb": 15.0, "speculative": {"draft_model": "d", "draft_vram_gb": 0.8}}"#,
            )
            .unwrap();
            e.policy.speculative.as_mut().unwrap().draft_model = bad.to_owned();
            let err = validate_model_entry("t", &e).unwrap_err().to_string();
            assert!(err.contains("draft_model"), "{bad}: {err}");
        }
    }

    /// P-C (firmness leftover): a config `models` id gets the same plain-folder-name check at
    /// validation (boot: `Config::load` fails, exit 78; config reload: 422) as `models/add`
    /// gives; `qwen:7b` and dotted names stay allowed.
    #[test]
    fn config_model_ids_must_be_plain_folder_names() {
        let base = r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":0.0,"models":"#;
        for (id, ok) in [
            ("qwen3-8b-int4-ov", true),
            ("qwen:7b", true),
            ("v1.2_model", true),
            ("../x", false),
            ("/abs/model", false),
            ("a/b", false),
            ("a\\\\b", false),
            ("..", false),
            (".", false),
            ("", false),
            ("C:", false),
            ("c:model", false),
        ] {
            let text = format!(r#"{base}{{"{id}":{{"vram_gb":0.0}}}}}}"#);
            let cfg: Config = serde_json::from_str(&text).unwrap();
            let r = cfg.validate();
            assert_eq!(r.is_ok(), ok, "{id}: {r:?}");
            if let Err(e) = r {
                assert!(e.to_string().contains("directly inside"), "{id}: {e}");
            }
        }
        // Through the boot path: `Config::load` refuses the file (main maps it to exit 78).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(
            &path,
            format!(r#"{base}{{"../escape":{{"vram_gb":0.0}}}}}}"#),
        )
        .unwrap();
        let err = Config::load(&path).unwrap_err();
        assert!(format!("{err:#}").contains("../escape"), "{err:#}");
    }

    /// An unknown field inside the `speculative` block is a hard deserialize
    /// error — proves `#[serde(deny_unknown_fields)]` still fires through the
    /// `#[serde(flatten)]` path (a documented serde landmine when combined;
    /// this test settles it empirically for this schema).
    #[test]
    fn speculative_block_rejects_unknown_field() {
        let result: Result<ModelEntry, _> = serde_json::from_str(
            r#"{"vram_gb": 15.0, "speculative": {"draft_model": "d", "draft_vram_gb": 0.8, "bogus": 1}}"#,
        );
        assert!(
            result.is_err(),
            "unknown field in speculative block must be rejected, got {result:?}"
        );
    }

    /// A models entry parses `load` and `evictable` (Slice 3), and the
    /// hand-written `Default` matches the serde defaults (evictable = true).
    #[test]
    fn model_policy_parses_slice3_knobs() {
        let cfg = minimal(
            r#", "models": {"sdxl": {"vram_gb": 0.0, "load": "on_demand", "evictable": false}, "chat": {"vram_gb": 0.0, "load": "eager"}}"#,
        );
        let sdxl = &cfg.models["sdxl"].policy;
        assert_eq!(sdxl.load, LoadPolicy::OnDemand);
        assert!(!sdxl.evictable, "evictable parses false when declared");
        let chat = &cfg.models["chat"].policy;
        assert_eq!(chat.load, LoadPolicy::Eager);
        assert!(
            chat.evictable,
            "evictable defaults true when load-only block"
        );
        // Hand-written Default must agree with the serde field defaults.
        let d = ModelPolicy::default();
        assert_eq!(d.load, LoadPolicy::Eager);
        assert!(d.evictable, "ModelPolicy::default() must be evictable");
    }

    /// Phase C1: a models entry parses an explicit `device` override; absent
    /// `device` defaults to `None` (load on the global `config.device`).
    #[test]
    fn model_policy_parses_device_override() {
        let cfg = minimal(
            r#", "models": {"qwen3-0.6b-int4-ov": {"vram_gb": 0.0, "device": "GPU.0"}, "chat": {"vram_gb": 0.0, "pinned": true}}"#,
        );
        let tiny = &cfg.models["qwen3-0.6b-int4-ov"].policy;
        assert_eq!(tiny.device.as_deref(), Some("GPU.0"));
        let chat = &cfg.models["chat"].policy;
        assert_eq!(chat.device, None, "device defaults None when not declared");
        assert_eq!(ModelPolicy::default().device, None);
    }

    // ---- gap-2: per-model reasoning_parser knob ----------------------------

    /// `reasoning_parser` absent → defaults to `None` (no extraction).
    #[test]
    fn reasoning_parser_defaults_none() {
        let cfg = minimal(r#", "models": {"m": {"vram_gb": 0.0}}"#);
        let p = &cfg.models["m"].policy;
        assert_eq!(
            p.reasoning_parser, None,
            "reasoning_parser must default to None"
        );
        assert_eq!(ModelPolicy::default().reasoning_parser, None);
    }

    /// All four named variants deserialise correctly.
    #[test]
    fn reasoning_parser_parses_known_variants() {
        let cfg = minimal(
            r#", "models": {
                "q": {"vram_gb": 0.0, "reasoning_parser": "qwen3"},
                "g": {"vram_gb": 0.0, "reasoning_parser": "gpt_oss"},
                "m": {"vram_gb": 0.0, "reasoning_parser": "mistral"},
                "p": {"vram_gb": 0.0, "reasoning_parser": "phi"}
            }"#,
        );
        assert_eq!(
            cfg.models["q"].policy.reasoning_parser,
            Some(ReasoningParser::Qwen3),
        );
        assert_eq!(
            cfg.models["g"].policy.reasoning_parser,
            Some(ReasoningParser::GptOss),
        );
        assert_eq!(
            cfg.models["m"].policy.reasoning_parser,
            Some(ReasoningParser::Mistral),
        );
        assert_eq!(
            cfg.models["p"].policy.reasoning_parser,
            Some(ReasoningParser::Phi),
        );
    }

    // ---- Co-residency Slice 2: bounded-KV / per-model stream knobs ------

    /// `default_kv_cache_gb` absent → defaults to 0.0 (unbounded grab-all).
    #[test]
    fn default_kv_cache_gb_defaults_zero() {
        let cfg = minimal("");
        assert!(
            (cfg.default_kv_cache_gb - 0.0).abs() < 1e-9,
            "default_kv_cache_gb must default to 0.0 (unbounded)"
        );
    }

    /// A models entry parses `kv_cache_gb` and `max_concurrent_streams`.
    #[test]
    fn model_policy_parses_slice2_knobs() {
        let cfg = minimal(
            r#", "default_kv_cache_gb": 4.0, "models": {"chat": {"vram_gb": 0.0, "kv_cache_gb": 6.0, "max_concurrent_streams": 8}}"#,
        );
        assert!((cfg.default_kv_cache_gb - 4.0).abs() < 1e-9);
        let chat = &cfg.models["chat"].policy;
        assert_eq!(chat.kv_cache_gb, Some(6.0));
        assert_eq!(chat.max_concurrent_streams, Some(8));
    }

    /// A negative `default_kv_cache_gb` is rejected (would inflate the pool).
    #[test]
    fn negative_default_kv_cache_gb_is_rejected() {
        let cfg = from_json(
            r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0,"default_kv_cache_gb":-1.0}"#,
        );
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("default_kv_cache_gb"),
            "must name the knob: {err}"
        );
    }

    /// A non-positive per-model `kv_cache_gb` (0.0) is rejected — a 0-GB
    /// partition is meaningless; absence means unbounded.
    #[test]
    fn zero_per_model_kv_cache_gb_is_rejected() {
        let cfg = from_json(
            r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0,"models":{"m":{"vram_gb":0.0,"kv_cache_gb":0.0}}}"#,
        );
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("models['m'].kv_cache_gb"),
            "must name the model knob: {err}"
        );
    }

    /// A zero `max_concurrent_streams` is rejected — a 0-stream cap admits
    /// nothing.
    #[test]
    fn zero_max_concurrent_streams_is_rejected() {
        let cfg = from_json(
            r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0,"models":{"m":{"vram_gb":0.0,"max_concurrent_streams":0}}}"#,
        );
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("models['m'].max_concurrent_streams"),
            "must name the model knob: {err}"
        );
    }

    /// A models entry parses `max_prompt_len` (NPU `MAX_PROMPT_LEN` override,
    /// the project's internal engineering log).
    #[test]
    fn model_policy_parses_max_prompt_len() {
        let cfg = minimal(r#", "models": {"npu-model": {"vram_gb": 0.0, "max_prompt_len": 2048}}"#);
        assert_eq!(cfg.models["npu-model"].policy.max_prompt_len, Some(2048));
    }

    /// A models entry parses `min_response_len` (NPU `MIN_RESPONSE_LEN`), and 0 is rejected.
    #[test]
    fn model_policy_parses_min_response_len_and_rejects_zero() {
        let cfg =
            minimal(r#", "models": {"npu-model": {"vram_gb": 0.0, "min_response_len": 512}}"#);
        assert_eq!(cfg.models["npu-model"].policy.min_response_len, Some(512));
        let cfg = from_json(
            r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0,"models":{"m":{"vram_gb":0.0,"min_response_len":0}}}"#,
        );
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("models['m'].min_response_len"), "{err}");
    }

    /// A zero `max_prompt_len` is rejected — a 0-token ceiling admits nothing;
    /// omit the field instead to keep `OpenVINO`'s own NPU default.
    #[test]
    fn zero_max_prompt_len_is_rejected() {
        let cfg = from_json(
            r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0,"models":{"m":{"vram_gb":0.0,"max_prompt_len":0}}}"#,
        );
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("models['m'].max_prompt_len"),
            "must name the model knob: {err}"
        );
    }

    /// `Config::load` returns an error for a non-existent file.
    #[test]
    fn test_config_load_missing_file_returns_error() {
        let result = Config::load(Path::new("/tmp/rustedvino_no_such_config_xyzzy.json"));
        assert!(result.is_err(), "loading a missing file must fail");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("cannot read config"),
            "error message must mention 'cannot read config': {msg}"
        );
    }
}
