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
#[derive(Debug, Clone, Deserialize)]
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
    /// a model without VRAM gating (useful when `total_vram_gb` is also 0.0).
    ///
    /// Each entry replaces what used to be three separate maps (`vram_gb`,
    /// `model_kinds`, `model_policies`): all per-model properties live here.
    #[serde(default)]
    pub models: HashMap<String, ModelEntry>,

    /// Total usable VRAM on the inference device, in GB.
    ///
    /// Set to `0.0` to disable VRAM gating entirely — all loads succeed
    /// without checking capacity (useful on CPU or when VRAM is not a
    /// constraint). When non-zero, `load_model` evicts LRU models if a new
    /// model would not fit.
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
    /// `dev/autotest/20260907_uma_ram_admission_ttm_pool.md`.
    ///
    /// Crediting reclaimable cache is the *less* conservative direction, so it
    /// stays behind an explicit operator declaration — a box that never sets
    /// this cannot regress. Discrete-VRAM boxes are unaffected regardless:
    /// their models resolve to a private-VRAM domain, never the system domain,
    /// so this gate never runs there. `system_ram_reservation_gb` still applies
    /// on top as the OS floor.
    #[serde(default)]
    pub system_ram_budget_gb: Option<f64>,

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

    /// Realtime voice arbitration v2 (`dev/plans/realtime-voice-model-
    /// arbitration-v2.md`, D3): how long, in seconds, a model stays
    /// hard-protected from eviction after any channel — admin load, plain
    /// API use, or a realtime turn — last used it. Self-expiring, checked
    /// against `last_used` on top of the always-on `pinned`/`evictable` hard
    /// excludes. Decided 2026-08-06: 300s, uniform across every channel (no
    /// special-cased longer window for admin loads). `0.0` disables grace
    /// globally. A per-model [`ModelPolicy::eviction_grace_secs`] override
    /// takes precedence when set.
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

    /// Cross-pipeline device admission ceiling (`dev/plans/cross-pipeline-
    /// admission-middleware.md` step 2): explicit per-device concurrency caps,
    /// keyed by the live `OpenVINO` device string (e.g. `"GPU.1"`). Sits IN
    /// FRONT OF each engine's own per-engine admission gate — a second,
    /// device-wide layer that also counts different engine kinds sharing one
    /// physical device (an LLM + VLM + STT + TTS + embedder all resident on
    /// one GPU). A device with **no** entry here is pure passthrough: no
    /// `Semaphore` is created for it and `DeviceBudgets::admit` never waits or
    /// holds anything for it. Default empty (disabled) — real numbers are set
    /// only after the project's internal engineering log's empirical
    /// per-device testing lands (Migration order #4). A configured cap must be
    /// at least `1` (validated below) — omit the entry to disable gating for a
    /// device rather than setting `0`.
    #[serde(default)]
    pub device_budgets: HashMap<String, u32>,

    /// KV-cache element precision for the inference device (`KV_CACHE_PRECISION`).
    ///
    /// `"u8"` (default, since 2026-09-02 — every fleet config already set this
    /// explicitly, this just makes an omitted field match established practice) =
    /// compress the KV cache to 8-bit, roughly **doubling** the number of tokens
    /// the `cache_size_gb` pool holds (e.g. ~26K → ~52K for qwen3-14b in a 4 GB
    /// pool) at a small quality cost — the lever for more context WITHOUT more
    /// VRAM. Empty string / `"f16"`/`"f32"` force the plugin default or an
    /// explicit precision. Unknown values are ignored (plugin default kept).
    ///
    /// NOTE: u8 KV compression is well-supported on CPU; GPU-plugin support must be
    /// verified empirically per `OpenVINO` version. Passed as a `compile_model`
    /// property to the CB pipeline at load time.
    #[serde(default = "default_kv_cache_precision")]
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

    /// IP address the HTTP server binds to. Default **`127.0.0.1`**
    /// (loopback — secure by default, T4.2). Set `"0.0.0.0"` for LAN/public
    /// exposure; with `api_keys` empty that additionally requires
    /// [`allow_insecure_public_bind`](Self::allow_insecure_public_bind).
    #[serde(default = "default_bind_addr")]
    pub bind_addr: String,

    /// TCP port the HTTP server listens on. Default **11437** (dev port;
    /// stormVINO owns 11435 until cutover day).
    #[serde(default = "default_port")]
    pub port: u16,

    /// Explicit opt-in to a non-loopback bind while auth is OFF
    /// (`api_keys` empty). Default **false**: a public bind with no auth
    /// refuses to start (committee finding 32 — the shipped defaults must
    /// never boot an open server on the network by accident).
    #[serde(default)]
    pub allow_insecure_public_bind: bool,

    /// Directory for the `OpenVINO` GPU kernel blob cache.
    ///
    /// When set, `OpenVINO` writes the compiled GPU kernel to this directory on
    /// the first model load (`openvino_cache/` subdirectory is created inside).
    /// Subsequent server starts load the blob directly — cutting GPU JIT time from
    /// ~1–3 min per model to ~10–30 s. `null`/absent = no cache (default, current
    /// behaviour). Safe to point at a persistent directory across reboots.
    ///
    /// On Windows use a path next to the exe, e.g. `"C:\\RustedVINO\\ov_cache"`.
    ///
    /// **Does not apply to NPU-routed models.** `ov_pipeline_create`
    /// (`src/npu_engine.rs`'s `OvPipeline::new`) never receives this value and
    /// never sets `ov::cache_dir` — every NPU model load (including a plain
    /// server restart with an unchanged config) always recompiles from
    /// scratch, ~20–40 s. This is also why changing a model's `max_prompt_len`
    /// and restarting is always safe: there is no persisted blob that could go
    /// stale against the new value, because nothing is ever persisted for this
    /// path today.
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

    /// Master switch for the KV-cache pressure monitor
    /// (`dev/plans/kv-cache-pressure-detection.md`). Default `false` —
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

/// Backoff/cap/health-check tuning for the `--supervise` wrapper
/// ([`crate::supervisor`], Unix only). A worker crash restarts it after an
/// increasing backoff; a rolling window bounds total restarts so a
/// deterministic crash cannot loop forever against a slow cold GPU init.
#[derive(Debug, Clone, PartialEq, Deserialize)]
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
    /// preserving. Default 10s.
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
    /// Default 600s (10 min) — deliberately generous: the only measured decode
    /// rate so far is ~60 tok/s (256 tokens in ~4.3s, `dev/autotest/
    /// 20260728_b70_qwen36_35b_bench.md`), so this should comfortably outlast
    /// any legitimate single request, but is **provisional** until a clean
    /// benchmark across realistic `max_tokens`/prompt sizes exists. Set to `0`
    /// to disable the generation watchdog entirely.
    #[serde(default = "default_watchdog_hang_ceiling_secs")]
    pub watchdog_hang_ceiling_secs: u64,
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
        }
    }
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

/// Realtime voice arbitration v2 default eviction-grace window: 300s,
/// matching `voice_pin::VOICE_FLOW_TTL`'s existing scale (decided 2026-08-06).
fn default_eviction_grace_secs() -> f64 {
    300.0
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
#[derive(Debug, Clone, PartialEq, Deserialize)]
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
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Deserialize)]
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
#[derive(Debug, Clone, Deserialize, Serialize)]
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
    /// `dev/plans/lfm2-tool-call-family.md`): `lfm2-24b-a2b-int4-ov`'s template
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
    /// non-NPU model. See the project's internal engineering log #6 and `src/npu_engine.rs`'s module
    /// doc comment for the full rationale.
    #[serde(default)]
    pub max_prompt_len: Option<u32>,

    /// Draft-model speculative decoding (opt-in). `None` = plain decoding,
    /// today's behaviour. See [`SpeculativeConfig`].
    #[serde(default)]
    pub speculative: Option<SpeculativeConfig>,

    /// Image-gen `generation_metadata.precision` (Tier 3,
    /// `PLAN_image_metadata_response.md`) — operator-supplied, e.g. `"int8"`,
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
            speculative: None,
            precision: None,
            model_source: None,
            model_revision: None,
            capabilities: Vec::new(),
            eviction_grace_secs: None,
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
#[derive(Debug, Clone, Deserialize, Serialize)]
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
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelEntry {
    /// VRAM estimate in GB. Required for every registered model.
    /// Use `0.0` to register a model without VRAM gating (CPU-only or
    /// unmetered).
    pub vram_gb: f64,

    /// Model kind hint. Valid values: `"embedding"`, `"text_gen"`,
    /// `"vision"`, `"stt"`, `"tts"`, `"image_gen"`, `"reranking"`.
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
            && self.load.is_none()
            && self.evictable.is_none()
            && self.device.is_none()
            && self.tier_preference.is_none()
            && self.reasoning_parser.is_none()
            && self.max_prompt_len.is_none()
            && self.speculative.is_none()
            && self.precision.is_none()
            && self.model_source.is_none()
            && self.model_revision.is_none()
            && self.capabilities.is_none()
            && self.eviction_grace_secs.is_none()
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

        (entry, out)
    }
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
    // comment on `max_num_seqs` and dev/autotest/20260530_429_saturation_below_cap.md).
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
pub(crate) fn default_max_prompt_array() -> usize {
    16 // one engine's worth of slots — a sane per-request fan-out ceiling
}

/// Default server-wide `max_tokens` ceiling. Also used by the handlers when
/// no config is loaded (mock mode). ~2.3 minutes of generation at 60 tok/s —
/// generous for real outputs, fatal to slot-pinning abuse.
pub(crate) fn default_max_tokens_cap() -> usize {
    8192
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
        let content = std::fs::read_to_string(path)
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
            self.kv_pressure_threshold_pct <= 100.0,
            "config kv_pressure_threshold_pct must be a percentage (0-100), got {}",
            self.kv_pressure_threshold_pct
        );
        // kv_pressure_monitor_enabled has no usable numeric default on
        // purpose (dev/plans/kv-cache-pressure-detection.md's ops-review
        // section) — a box that turns monitoring on must state its own real
        // threshold/duration, not silently inherit whatever this codebase
        // happens to ship as a "default." `0.0`/`0` are the "not configured"
        // sentinels, so catch them here rather than let the monitor start
        // with a threshold of 0% (which would flag every model, instantly).
        if self.kv_pressure_monitor_enabled {
            anyhow::ensure!(
                self.kv_pressure_threshold_pct > 0.0,
                "config kv_pressure_threshold_pct must be set (>0) when \
                 kv_pressure_monitor_enabled is true — no default is provided deliberately, \
                 see dev/plans/kv-cache-pressure-detection.md"
            );
            anyhow::ensure!(
                self.kv_pressure_sustained_secs > 0,
                "config kv_pressure_sustained_secs must be set (>0) when \
                 kv_pressure_monitor_enabled is true — no default is provided deliberately, \
                 see dev/plans/kv-cache-pressure-detection.md"
            );
        }
        // Per-model knob validation, shared with `ModelManager::add_model` (the
        // runtime registration path) via `validate_model_entry` so both ways of
        // registering a model enforce identical invariants.
        for (model, entry) in &self.models {
            validate_model_entry(model, entry)?;
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
    /// A non-loopback `addr` with `has_keys: false` (the keys file — real
    /// [`crate::AuthConfig`], not this struct's deprecated `api_keys` field —
    /// resolved to no keys at all) refuses to start unless
    /// `allow_insecure_public_bind: true` explicitly accepts it. Call after
    /// resolving the keys file, before any model load.
    ///
    /// # Errors
    /// Non-loopback bind + no keys configured, without the explicit opt-in.
    pub fn enforce_open_bind_gate(
        &self,
        addr: std::net::SocketAddr,
        has_keys: bool,
    ) -> anyhow::Result<()> {
        if !addr.ip().is_loopback() && !has_keys && !self.allow_insecure_public_bind {
            anyhow::bail!(
                "refusing to start: bind_addr '{}' is reachable from the network but no \
                 keys are configured (auth OFF). Set keys in the keys file (see \
                 keys_file/resolve_keys_file_path), bind to 127.0.0.1, or set \
                 allow_insecure_public_bind: true to explicitly run an open server",
                self.bind_addr
            );
        }
        Ok(())
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
    if let Some(streams) = policy.max_concurrent_streams {
        anyhow::ensure!(
            streams >= 1,
            "models['{model_id}'].max_concurrent_streams must be at least 1, got {streams}"
        );
    }
    if let Some(max_prompt_len) = policy.max_prompt_len {
        anyhow::ensure!(
            max_prompt_len >= 1,
            "models['{model_id}'].max_prompt_len must be at least 1, got {max_prompt_len} \
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
    // Speculative decoding (Layer A — pure config, no disk access; see
    // dev/plans/speculative-decoding-integration.md Part 4).
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
        default_kv_cache_gb, vram_safety_margin_gb, min_kv_cache_gb,
        system_ram_reservation_gb, system_ram_budget_gb, light_model_max_gb,
        light_stt_max_gb, dgpu_size_ceiling_fraction, domain_budgets,
        eviction_grace_secs, realtime_defaults, realtime_viable_minimum,
        device_budgets, kv_cache_precision, enable_prefix_caching,
        cors_allowed_origins, admission_queue_timeout_ms,
        device_admission_queue_timeout_ms, embedding_device, default_embed_model,
        embedding_pooling, embedding_normalize, max_prompt_array, max_tokens_cap,
        bind_addr, port, allow_insecure_public_bind, ov_cache_dir,
        ov_cache_max_gb, ov_cache_sweep_interval_secs, kv_pressure_monitor_enabled,
        kv_pressure_threshold_pct, kv_pressure_sustained_secs,
        kv_pressure_sweep_interval_secs, kv_resize_cooldown_secs, supervisor,
    ],
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    /// An unchanged file reports nothing — the common path must stay quiet, so
    /// the new field never adds noise to a routine model-only reload.
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

    /// Tier 3 (`PLAN_image_metadata_response.md`): `precision`/`model_source`/
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
    /// `enforce_open_bind_gate` takes `has_keys` as an explicit bool now (the
    /// caller resolves it from the keys file, not from `Config` — see its doc
    /// comment) — no need to construct a `Config` with keys in it to test this.
    #[test]
    fn public_bind_without_auth_refuses_to_start() {
        let cfg = minimal(r#", "bind_addr": "0.0.0.0""#);
        let addr = cfg.validated_bind().unwrap();
        let err = cfg
            .enforce_open_bind_gate(addr, false)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("refusing to start"),
            "open public bind must abort: {err}"
        );
    }

    /// Public bind is fine with `has_keys: true`, or with the explicit
    /// insecure opt-in even with no keys.
    #[test]
    fn public_bind_allowed_with_keys_or_explicit_opt_in() {
        let cfg = minimal(r#", "bind_addr": "0.0.0.0""#);
        let addr = cfg.validated_bind().unwrap();
        assert!(cfg.enforce_open_bind_gate(addr, true).is_ok());

        let opted_in = minimal(r#", "bind_addr": "0.0.0.0", "allow_insecure_public_bind": true"#);
        let addr = opted_in.validated_bind().unwrap();
        assert!(opted_in.enforce_open_bind_gate(addr, false).is_ok());
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

    /// `kv_pressure_monitor_enabled` defaults to `false`, and the threshold/
    /// duration default to their "not configured" sentinels (`0.0`/`0`) —
    /// an existing config file with none of these keys must keep working
    /// exactly as before this feature existed
    /// (`dev/plans/kv-cache-pressure-detection.md`'s ops-review finding:
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
    /// the project's internal engineering log #6).
    #[test]
    fn model_policy_parses_max_prompt_len() {
        let cfg = minimal(r#", "models": {"npu-model": {"vram_gb": 0.0, "max_prompt_len": 2048}}"#);
        assert_eq!(cfg.models["npu-model"].policy.max_prompt_len, Some(2048));
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
