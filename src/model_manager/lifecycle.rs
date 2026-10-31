// ============================================================
// src/model_manager/lifecycle.rs — EngineFactory trait
// ============================================================
// Abstracts "create an engine from a model path" so that:
//   - Production code uses `OvEngineFactory` (real GPU load)
//   - Tests inject `MockEngineFactory` (no GPU, instant, deterministic)
//
// The trait is `pub(crate)` — only `ModelManager` and test code
// inside this crate ever construct an `EngineFactory`. External
// callers only see `ModelManager`, never the factory seam.
//
// CRASH COURSE — why a trait for testing?
//   `OvCbEngine::new()` blocks for ~10–30 s and requires real
//   GPU hardware. Unit tests must be fast and hardware-agnostic.
//   Injecting a mock factory breaks the hard dependency on the GPU
//   without any `unsafe` or conditional compilation in the real
//   code paths.
//
// CRASH COURSE — why `Arc<dyn EngineFactory>` not `impl EngineFactory`?
//   `ModelManager` needs to be `Clone`-able (it's behind `Arc<>` in
//   production, so this isn't strictly required, but storing
//   `Arc<dyn Trait>` avoids lifetime complications and lets the
//   factory be shared cheaply between the manager and internal
//   helpers that call `spawn_blocking` closures, which require
//   `'static` captures).
// ============================================================

use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;

use crate::model_manager::engine::{EngineHandleKind, ModelKind};

/// A validated draft-model attachment for one target model's next load —
/// draft-model speculative decoding
/// (the project's internal engineering log). Built by
/// `ModelManager` from a target's `SpeculativeConfig` after
/// `validate_speculative_pairing` succeeds, and consumed by
/// [`OvEngineFactory::load`]'s `TextGen` arm to attach the draft at
/// `OvCbEngine` construction.
#[derive(Debug, Clone)]
pub(crate) struct DraftHint {
    /// Absolute path to the draft model directory
    /// (`models_dir.join(spec.draft_model)`).
    pub draft_path: String,
    /// Resolved `OpenVINO` device for the draft. Always equals the target's
    /// own resolved device in v1 (cross-device drafting is out of scope —
    /// Part 4 gate 4 already enforced this at validation time).
    pub draft_device: String,
    /// Candidate tokens per verification step, stamped on every request the
    /// resulting engine serves.
    pub num_assistant_tokens: usize,
    /// Run the load-time greedy-equivalence self-check (Part 7) before
    /// serving. `false` skips straight to building the draft-attached engine
    /// — the documented escape hatch for a pairing already verified on this
    /// exact box/driver/OV version.
    pub verify_on_load: bool,
}

/// Contract for constructing a CB engine from a model path and device.
///
/// Implementors must be `Send + Sync + 'static` because the factory
/// is captured by `spawn_blocking` closures that cross thread boundaries.
///
/// The returned pair is:
/// - [`EngineHandleKind`] — the typed, cloneable submit handle (given to the
///   manager; R1 produces `TextGen` or `Vision` per [`detect_kind`](EngineFactory::detect_kind))
/// - [`std::thread::JoinHandle`] — used to join the engine thread during
///   eviction (the quiesce protocol)
pub(crate) trait EngineFactory: Send + Sync + 'static {
    /// Classify the model in `model_dir` so the manager and `load` know which
    /// engine to build. R1 distinguishes [`ModelKind::Vision`] (a VLM, detected
    /// by `openvino_vision_embeddings_model.xml`) from [`ModelKind::TextGen`]
    /// (everything else). This moves the kind decision off `main.rs` and into
    /// the factory — the manager owns "what kind is this model."
    fn detect_kind(&self, model_dir: &Path) -> ModelKind;

    /// The `KV_CACHE_PRECISION` string a load of this model hands the engine (`""` = plugin
    /// default; also the tag of its blob-cache directory), or `None` when the factory does not
    /// resolve one (test factories): callers then use the normalised global string. Must agree
    /// with what [`load`](Self::load) passes, since the prune pass uses it to find the directory
    /// a model's live blobs are in.
    fn kv_passed(&self, _model_path: &Path, _kind: ModelKind, _device: &str) -> Option<String> {
        None
    }

    /// What exact (blocks-first) pool sizing needs for this model: its KV profile and the
    /// precision the engine will be given. `None` = the factory has no such knowledge (test
    /// factories, kinds without a KV cache, no readable `config.json`): the caller sizes the pool
    /// the old way, in whole GiB. Blocking file I/O: not for the async runtime.
    fn kv_pool_inputs(
        &self,
        _model_path: &Path,
        _kind: ModelKind,
        _device: &str,
    ) -> Option<KvPoolInputs> {
        None
    }

    /// The linked `OpenVINO` and `GenAI` build strings, `None` when either is unknown. Gates exact
    /// pool sizing to the engine versions the layouts were verified on. Overridden by test
    /// factories so the result does not depend on the SDK a test machine happens to link.
    fn engine_versions(&self) -> Option<(String, String)> {
        crate::ov_cb::openvino_version().zip(crate::ov_cb::genai_version())
    }

    /// Record the exact KV block count the next load of `model_id` must build its pool with
    /// (`None` = size from the GiB budget as before). Set right before each load.
    fn register_kv_blocks_hint(&self, _model_id: &str, _blocks: Option<u64>) {}

    /// Record an explicit kind hint for `model_id`, so a later [`detect_kind`]
    /// call recognizes it without directory sniffing.
    ///
    /// Only [`OvEngineFactory`] needs this: `add_model` (runtime registration)
    /// learns a model's kind from the request body, but the factory's
    /// `model_kinds` map was previously populated once at construction and
    /// never touched again — so a dynamically-added embedding/tts/reranking
    /// model (file-indistinguishable from a plain LLM) silently defaulted to
    /// `TextGen` and failed to load. Default no-op: test factories classify by
    /// name and have no such map to update.
    fn register_kind_hint(&self, _model_id: &str, _kind: ModelKind) {}

    /// Record a `max_concurrent_streams` cap for `model_id` so the VLM engine
    /// spawned for it uses that channel size rather than [`VLM_CHANNEL_CAP`].
    /// Default no-op: test factories don't spawn real VLM threads.
    /// `None` removes the hint, so clearing the field (PATCH `null`, or a
    /// config reload while the model is not loaded) really restores the
    /// default channel size on the next load.
    fn register_concurrent_streams_hint(&self, _model_id: &str, _cap: Option<usize>) {}

    /// Record a per-model `enable_prefix_caching` override
    /// ([`crate::model_manager::config::ModelPolicy::enable_prefix_caching`])
    /// for `model_id`'s next load. `None` removes the hint, so clearing the
    /// field (PATCH `null`) restores the server's own decision
    /// ([`resolve_prefix_caching`]). Default no-op: test factories build stub
    /// engines that have no prefix cache.
    fn register_prefix_caching_hint(&self, _model_id: &str, _value: Option<bool>) {}

    /// Record the per-model `kv_cache_precision` override (already normalised: `""` = auto, `i8`,
    /// `u4`, `f16`; `None` = no override) the next load of `model_id` resolves with.
    fn register_kv_precision_hint(&self, _model_id: &str, _value: Option<&'static str>) {}

    /// Record a `max_prompt_len` (NPU `MAX_PROMPT_LEN`) hint for `model_id` so
    /// an NPU-routed load compiles its `LLMPipeline` with that ceiling instead
    /// of `OpenVINO`'s own default (1024). Ignored for every other device.
    /// Default no-op: test factories don't spawn real NPU pipelines.
    /// `None` removes the hint, so clearing the field (PATCH `null`, or a
    /// config reload while the model is not loaded) really restores
    /// `OpenVINO`'s default on the next load.
    fn register_max_prompt_len_hint(&self, _model_id: &str, _max_prompt_len: Option<u32>) {}

    /// Record a `min_response_len` (NPU `MIN_RESPONSE_LEN`) hint for
    /// `model_id` — same lifecycle as [`register_max_prompt_len_hint`].
    /// Default no-op: test factories don't spawn real NPU pipelines.
    fn register_min_response_len_hint(&self, _model_id: &str, _min_response_len: Option<u32>) {}

    /// Record (or, with `None`, clear) a draft-model attachment hint for
    /// `model_id`'s next load — draft-model speculative decoding
    /// (the project's internal engineering log). `None` lets
    /// a `reload_config` that removes the `speculative` block clear a stale
    /// hint from a prior load. Default no-op: test factories never attach a
    /// draft (they build a stub engine that ignores it). See [`DraftHint`].
    fn register_draft_hint(&self, _model_id: &str, _hint: Option<DraftHint>) {}

    /// Record operator-supplied model provenance (`precision`/`model_source`/
    /// `model_revision`, Tier 3 of the image-metadata plan) for
    /// `model_id`'s next `ImageGen` load. Default no-op: test factories don't
    /// build real `ImageModelMetadata`.
    fn register_image_provenance_hint(
        &self,
        _model_id: &str,
        _hint: crate::pipelines::image::ImageProvenanceHint,
    ) {
    }

    /// Whether the model's files actually exist at `model_dir`. Checked by
    /// `ModelManager::load_model_with_overrides` **before** any VRAM
    /// reservation or LRU eviction, so a registered-but-absent `model_id`
    /// (stale config, typo, name drift from what was actually downloaded) fails
    /// cheaply instead of evicting a healthy `Ready` model to make room for a
    /// load that was always going to fail once the factory tried to open the IR.
    /// Default `true`: test factories never touch the real filesystem, so only
    /// [`OvEngineFactory`] overrides this with a real check.
    fn model_exists(&self, _model_dir: &Path) -> bool {
        true
    }

    /// Check whether `model_dir` has every file its `kind` needs to actually
    /// run (`crate::model_completeness`), not just exist. Default: complete
    /// — test factories build fixture directories that don't carry real OV
    /// IR files at all, so a real filesystem check would reject every mock
    /// model; only [`OvEngineFactory`] performs the real check.
    fn check_completeness(
        &self,
        _model_dir: &Path,
        _kind: ModelKind,
    ) -> crate::model_completeness::CompletenessReport {
        crate::model_completeness::CompletenessReport::default()
    }

    /// Load the model at `model_path` on `device` and start an engine thread.
    ///
    /// Dispatches on [`detect_kind`](Self::detect_kind): a CB text-gen engine or
    /// a VLM vision engine, returning the matching [`EngineHandleKind`] arm.
    ///
    /// `cache_size_gb` is the KV pool to allocate, computed dynamically by
    /// `ModelManager::ensure_vram_for` just before this call. `0.0` means
    /// "let the plugin decide" (legacy / unknown-VRAM-model path only). The VLM
    /// engine ignores it (it has no CB-style KV pool).
    ///
    /// `max_num_seqs` is the resolved per-model concurrency cap (co-residency
    /// Slice 2): the CB `SchedulerConfig::max_num_seqs` **and** the engine's
    /// HTTP 429 admission gate. `0` keeps the `OpenVINO` default (256). The
    /// text-gen engine uses it directly; the `Vision` arm uses it as its layer-1
    /// default (`max_concurrent_streams` still wins as layer 2) when no
    /// per-model override is registered — see [`VLM_CHANNEL_CAP`]. The
    /// embedding engine remains single-stream and ignores it.
    ///
    /// **Called from `tokio::task::spawn_blocking`** — may block for seconds.
    /// Must not use `await` or touch the tokio runtime directly.
    fn load(
        &self,
        model_path: &Path,
        device: &str,
        cache_size_gb: f64,
        max_num_seqs: usize,
    ) -> anyhow::Result<(EngineHandleKind, std::thread::JoinHandle<()>)>;
}

/// Inputs of exact pool sizing, from [`EngineFactory::kv_pool_inputs`].
#[derive(Clone, Debug)]
pub(crate) struct KvPoolInputs {
    /// The model's KV layout.
    pub profile: std::sync::Arc<super::kv_profile::KvProfile>,
    /// The precision decision (the engine is given `resolution.passed`).
    pub resolution: super::kv_precision::KvResolution,
    /// Which engine builds the pool (decides the linear-attention reservation).
    pub path: super::kv_precision::EnginePath,
}

// ---- Production implementation ----------------------------------------

/// Production engine factory: loads via [`OvCbEngine`] + [`spawn_engine`].
///
/// `admission_queue_timeout_ms` is fixed at construction time. `cache_size_gb`
/// and `max_num_seqs` are passed per-load from `ModelManager` (co-residency
/// Slice 2 resolves them per-model from config policy / overrides).
#[allow(clippy::struct_excessive_bools)] // independent load-time switches from the config
pub(crate) struct OvEngineFactory {
    /// `KV_CACHE_PRECISION` property; `""` = plugin default, `"i8"`/`"u4"`/`"f16"` = explicit (normalised from the config).
    pub kv_cache_precision: String,
    /// Max wait for an inference slot before returning HTTP 429.
    pub admission_queue_timeout_ms: u64,
    /// Explicit per-model kind hints (from `Config::model_kinds` at startup,
    /// plus [`register_kind_hint`](EngineFactory::register_kind_hint) for
    /// models added at runtime), keyed by model ID. Consulted by
    /// [`detect_kind`](Self::detect_kind) before any directory sniffing — the
    /// only way to mark an embedding model, which is otherwise
    /// file-indistinguishable from a plain LLM. `RwLock` because
    /// `add_model` inserts into it after construction, through a shared
    /// `Arc<dyn EngineFactory>`.
    pub model_kinds: std::sync::RwLock<HashMap<String, ModelKind>>,
    /// Per-model VLM channel-capacity hints from `max_concurrent_streams` config.
    /// Keyed by model ID, set via [`register_concurrent_streams_hint`] before
    /// each VLM load so [`spawn_vlm_engine`] uses the configured cap instead of
    /// [`VLM_CHANNEL_CAP`].
    pub concurrent_streams: std::sync::RwLock<HashMap<String, usize>>,
    /// Per-model `enable_prefix_caching` overrides, keyed by model ID, set via
    /// [`EngineFactory::register_prefix_caching_hint`] before each load.
    pub prefix_caching_overrides: std::sync::RwLock<HashMap<String, bool>>,
    /// Per-model NPU `MAX_PROMPT_LEN` hints from `max_prompt_len` config.
    /// Keyed by model ID, set via [`register_max_prompt_len_hint`] before each
    /// NPU load so `OvPipeline::new` compiles with the configured ceiling
    /// instead of `OpenVINO`'s own default (1024).
    pub max_prompt_len_hints: std::sync::RwLock<HashMap<String, u32>>,
    /// Per-model NPU `MIN_RESPONSE_LEN` hints from `min_response_len` config,
    /// registered alongside [`max_prompt_len_hints`](Self::max_prompt_len_hints).
    pub min_response_len_hints: std::sync::RwLock<HashMap<String, u32>>,
    /// Per-model draft-model attachment hints for speculative decoding, keyed
    /// by model ID. Set via [`register_draft_hint`] before each `TextGen`
    /// load; absent (or explicitly cleared via `None`) means plain decoding.
    /// See [`DraftHint`].
    pub draft_hints: std::sync::RwLock<HashMap<String, DraftHint>>,
    /// Per-model operator-supplied provenance (`precision`/`model_source`/
    /// `model_revision`) hints, keyed by model ID. Set via
    /// [`register_image_provenance_hint`] before each `ImageGen` load.
    pub image_provenance_hints:
        std::sync::RwLock<HashMap<String, crate::pipelines::image::ImageProvenanceHint>>,
    /// Pooling strategy for embedding models (from `Config::embedding_pooling`).
    pub embedding_pooling: crate::ov_embed::Pooling,
    /// Whether to L2-normalize embeddings (from `Config::embedding_normalize`).
    pub embedding_normalize: bool,
    /// `OpenVINO` GPU blob cache directory (from `Config::ov_cache_dir`).
    /// `None` = caching disabled (plugin default). Non-empty: compiled blobs are
    /// written on first load and read on subsequent loads, cutting JIT time.
    pub ov_cache_dir: Option<String>,
    /// `Config::ov_cache_layout == "keyed"`: the KV engines compile into `ov_cache_dir/kv-<tag>/`.
    pub ov_cache_keyed: bool,
    /// Reuse KV blocks across CB requests sharing an identical prompt prefix
    /// (from `Config::enable_prefix_caching`). Only affects prefill/TTFT.
    pub enable_prefix_caching: bool,
    /// `Config::kv_warmup`: run a one-token generation after each GPU load of a KV engine.
    pub kv_warmup: bool,
    /// Per-model `kv_cache_precision` overrides (normalised), set before each load.
    pub kv_precision_overrides: std::sync::RwLock<HashMap<String, &'static str>>,
    /// `Config::kv_precision_pass`: who decides the engines' KV precision.
    pub kv_pass_mode: super::kv_precision::PassMode,
    /// The string each model was last resolved to, by (model path, device): read by the prune
    /// pass through [`EngineFactory::kv_passed`] so it need not re-parse the IR.
    pub kv_resolved: std::sync::Mutex<HashMap<(std::path::PathBuf, String), &'static str>>,
    /// Exact KV block count for the next load, by model ID, set through
    /// [`EngineFactory::register_kv_blocks_hint`].
    pub kv_blocks_hints: std::sync::RwLock<HashMap<String, u64>>,
}

impl OvEngineFactory {
    /// The per-model precision override registered for the model in `model_path` (its leaf
    /// directory name is the model id).
    fn per_model_precision(&self, model_path: &Path) -> Option<&'static str> {
        let id = model_path.file_name().and_then(|s| s.to_str())?;
        self.kv_precision_overrides
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(id)
            .copied()
    }

    /// Whether a speculative-decoding draft is registered for the model in `model_path` (its leaf
    /// directory name is the model id). Such a model keeps the 0.8.x pass-through: the draft
    /// engine inherits the main engine's properties and may run on another device.
    fn has_draft_model(&self, model_path: &Path) -> bool {
        model_path
            .file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|id| {
                self.draft_hints
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .contains_key(id)
            })
    }

    /// Resolve (and remember) the KV precision of one model about to load, and log the profile.
    /// Called from `load`, which runs on a blocking thread: it reads `config.json` and the IR.
    fn resolve_kv(
        &self,
        model_path: &Path,
        model_id: &str,
        kind: ModelKind,
        device: &str,
    ) -> super::kv_precision::KvResolution {
        use super::kv_precision::{
            EnginePath, KvResolution, PrecisionSource, resolve_kv_precision,
        };
        let path = match kind {
            ModelKind::TextGen => EnginePath::Cb,
            ModelKind::Vision => EnginePath::Vlm,
            // No KV cache engine: nothing to resolve or pass.
            _ => {
                return KvResolution {
                    precision: None,
                    rule_guess: None,
                    source: PrecisionSource::LegacyPassthrough,
                    passed: "",
                };
            }
        };
        // NPU engines take no KV precision: nothing to read.
        let sd = self.has_draft_model(model_path);
        let profile = (!sd && !device.to_ascii_uppercase().starts_with("NPU"))
            .then(|| super::kv_profile::KvProfile::read(model_path, device))
            .flatten();
        let res = if sd {
            super::kv_precision::legacy_resolution(path, &self.kv_cache_precision)
        } else {
            resolve_kv_precision(
                profile.as_ref(),
                path,
                device,
                self.per_model_precision(model_path),
                &self.kv_cache_precision,
                self.kv_pass_mode,
            )
        };
        self.kv_resolved
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert((model_path.to_path_buf(), device.to_owned()), res.passed);
        tracing::info!(
            model_id,
            kv_precision = ?res.precision,
            kv_rule_guess = ?res.rule_guess,
            kv_precision_source = ?res.source,
            kv_passed = res.passed,
            kv_bytes_per_token = profile
                .as_ref()
                .zip(res.precision)
                .map(|(p, prec)| p.kv_bytes_per_token(prec)),
            kv_profile_confidence = ?profile.as_ref().map(|p| p.confidence),
            engine_path = ?path,
            "kv profile"
        );
        warn_vlm_ignores_global_once(model_id, path, res, &self.kv_cache_precision);
        if let (Some(hint), Some(resolved)) =
            (profile.as_ref().and_then(|p| p.ir_kv_hint), res.precision)
            && hint != resolved
            && matches!(
                res.source,
                super::kv_precision::PrecisionSource::PerModel
                    | super::kv_precision::PrecisionSource::GlobalExplicit
            )
        {
            // E-14: the IR's own runtime option names another precision (f16 for an fp16 model);
            // the operator's explicit setting wins, and this line says so.
            tracing::info!(
                model_id,
                ir_hint = ?hint,
                resolved = ?resolved,
                source = ?res.source,
                "ir_hint overridden by an explicit KV precision setting"
            );
        }
        if let Some(p) = self
            .per_model_precision(model_path)
            .filter(|p| !p.is_empty())
            && res.source != super::kv_precision::PrecisionSource::PerModel
        {
            tracing::warn!(
                model_id,
                requested = p,
                device,
                source = ?res.source,
                "this model's kv_cache_precision override is ignored (it applies on a plain GPU \
                 device with kv_precision_pass \"explicit\", not to CPU, NPU, HETERO/AUTO, \
                 speculative-decoding models or plugin mode)"
            );
        }
        res
    }
}

/// One WARN per process when the global `kv_cache_precision` names a precision a vision model
/// does not get (they resolve their own; the global key applies to text models only).
fn warn_vlm_ignores_global_once(
    model_id: &str,
    path: super::kv_precision::EnginePath,
    res: super::kv_precision::KvResolution,
    global: &str,
) {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if super::kv_precision::vlm_ignores_global(path, res, global)
        && !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed)
    {
        tracing::warn!(
            model_id,
            global,
            resolved = ?res.precision,
            source = ?res.source,
            "the global kv_cache_precision does not apply to vision models; vision models resolve their own (this one: see resolved)"
        );
    }
}

/// Resolve the device an embedding model should load on.
///
/// Precedence: an explicit `embedding_device` (if available) → the iGPU
/// (`GPU.0`) when present and distinct from the inference device → the
/// inference device → `CPU`. The intent is to offload embeddings to the
/// integrated GPU so the inference dGPU stays free, while never hard-failing
/// when the preferred device is absent.
pub(crate) fn resolve_embedding_device(configured: Option<&str>, inference_device: &str) -> String {
    use crate::ov_cb::device_available;

    if let Some(dev) = configured {
        if device_available(dev) {
            return dev.to_owned();
        }
        tracing::warn!(
            device = %dev,
            "configured embedding_device is not available — falling back to auto-select"
        );
    }
    // Prefer a GPU distinct from the inference dGPU (the iGPU on these boxes).
    if inference_device != "GPU.0" && device_available("GPU.0") {
        return "GPU.0".to_owned();
    }
    if device_available(inference_device) {
        return inference_device.to_owned();
    }
    "CPU".to_owned()
}

/// Which TTS backend a `ModelKind::Tts` model directory contains, detected by
/// directory contents (no engine construction — pure, GPU-free, unit-testable).
///
/// Phase 5.2: three backends share `ModelKind::Tts`.
///   Kokoro-82M  — `voices.bin`/`voices-v1.0.bin` in the model dir (pure-Rust ORT;
///                 ONNX model, no OV IR). `voices.bin` = bincode format
///                 (mzdk100/kokoro release) — correct. `voices-v1.0.bin` = numpy
///                 format (thewh1teagle Python pkg) — wrong.
///   Coqui VITS  — `model.onnx` + `tokens.txt` (character-based, no phonemizer —
///                 see the project's internal engineering log for why this replaced an earlier
///                 Piper/espeak-ng prototype, GPL-3.0, ruled out on license
///                 grounds).
///   `SpeechT5`  — fallback; `OpenVINO` `Text2SpeechPipeline` (OV IR).
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum TtsBackend {
    Kokoro {
        onnx_name: &'static str,
        voices_name: &'static str,
    },
    CoquiVits {
        onnx_name: &'static str,
        tokens_name: &'static str,
    },
    /// A Piper voice: `<name>.onnx` + `<name>.onnx.json` (espeak phonemes).
    Piper {
        onnx: std::path::PathBuf,
        config: std::path::PathBuf,
    },
    SpeechT5,
}

pub(crate) fn detect_tts_backend(model_path: &Path) -> anyhow::Result<TtsBackend> {
    let voices_name = if model_path.join("voices.bin").is_file() {
        Some("voices.bin")
    } else if model_path.join("voices-v1.0.bin").is_file() {
        Some("voices-v1.0.bin")
    } else {
        None
    };
    if let Some(voices_name) = voices_name {
        // Kokoro: pick the smallest available ONNX variant.
        let onnx_name = [
            "kokoro-v1.0.int8.onnx",
            "kokoro-v1.0.fp16.onnx",
            "kokoro-v1.0.onnx",
        ]
        .into_iter()
        .find(|n| model_path.join(n).is_file())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Kokoro model dir {} has {voices_name} but no kokoro-v1.0*.onnx \
                     — re-download the model",
                model_path.display()
            )
        })?;
        return Ok(TtsBackend::Kokoro {
            onnx_name,
            voices_name,
        });
    }
    if let Some((onnx, config)) = crate::pipelines::piper::find_piper_voice(model_path) {
        return Ok(TtsBackend::Piper { onnx, config });
    }
    if model_path.join("model.onnx").is_file() && model_path.join("tokens.txt").is_file() {
        return Ok(TtsBackend::CoquiVits {
            onnx_name: "model.onnx",
            tokens_name: "tokens.txt",
        });
    }
    Ok(TtsBackend::SpeechT5)
}

impl EngineFactory for OvEngineFactory {
    fn register_kind_hint(&self, model_id: &str, kind: ModelKind) {
        self.model_kinds
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(model_id.to_owned(), kind);
    }

    fn register_concurrent_streams_hint(&self, model_id: &str, cap: Option<usize>) {
        let mut hints = self
            .concurrent_streams
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match cap {
            Some(v) => hints.insert(model_id.to_owned(), v),
            None => hints.remove(model_id),
        };
    }

    fn register_kv_precision_hint(&self, model_id: &str, value: Option<&'static str>) {
        let mut hints = self
            .kv_precision_overrides
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match value {
            Some(v) => hints.insert(model_id.to_owned(), v),
            None => hints.remove(model_id),
        };
    }

    fn register_prefix_caching_hint(&self, model_id: &str, value: Option<bool>) {
        let mut hints = self
            .prefix_caching_overrides
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match value {
            Some(v) => hints.insert(model_id.to_owned(), v),
            None => hints.remove(model_id),
        };
    }

    fn register_max_prompt_len_hint(&self, model_id: &str, max_prompt_len: Option<u32>) {
        let mut hints = self
            .max_prompt_len_hints
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match max_prompt_len {
            Some(v) => hints.insert(model_id.to_owned(), v),
            None => hints.remove(model_id),
        };
    }

    fn register_min_response_len_hint(&self, model_id: &str, min_response_len: Option<u32>) {
        let mut hints = self
            .min_response_len_hints
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match min_response_len {
            Some(v) => hints.insert(model_id.to_owned(), v),
            None => hints.remove(model_id),
        };
    }

    fn register_draft_hint(&self, model_id: &str, hint: Option<DraftHint>) {
        let mut guard = self
            .draft_hints
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match hint {
            Some(h) => {
                guard.insert(model_id.to_owned(), h);
            }
            None => {
                guard.remove(model_id);
            }
        }
    }

    fn register_image_provenance_hint(
        &self,
        model_id: &str,
        hint: crate::pipelines::image::ImageProvenanceHint,
    ) {
        self.image_provenance_hints
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(model_id.to_owned(), hint);
    }

    fn model_exists(&self, model_dir: &Path) -> bool {
        model_dir.is_dir()
    }

    fn check_completeness(
        &self,
        model_dir: &Path,
        kind: ModelKind,
    ) -> crate::model_completeness::CompletenessReport {
        crate::model_completeness::check(model_dir, kind)
    }

    fn detect_kind(&self, model_dir: &Path) -> ModelKind {
        self.explicit_kind_hint(model_dir)
            .or_else(|| detect_kind_from_markers(model_dir))
            .unwrap_or(ModelKind::TextGen)
    }

    fn register_kv_blocks_hint(&self, model_id: &str, blocks: Option<u64>) {
        let mut hints = self
            .kv_blocks_hints
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match blocks {
            Some(b) => hints.insert(model_id.to_owned(), b),
            None => hints.remove(model_id),
        };
    }

    fn kv_pool_inputs(
        &self,
        model_path: &Path,
        kind: ModelKind,
        device: &str,
    ) -> Option<KvPoolInputs> {
        let path = match kind {
            ModelKind::TextGen => super::kv_precision::EnginePath::Cb,
            ModelKind::Vision => super::kv_precision::EnginePath::Vlm,
            _ => return None,
        };
        if device.to_ascii_uppercase().starts_with("NPU") || self.has_draft_model(model_path) {
            return None;
        }
        let profile = super::kv_profile::KvProfile::read(model_path, device)?;
        let resolution = super::kv_precision::resolve_kv_precision(
            Some(&profile),
            path,
            device,
            self.per_model_precision(model_path),
            &self.kv_cache_precision,
            self.kv_pass_mode,
        );
        // The prune pass asks which directory this model's blobs are in before it ever loads.
        self.kv_resolved
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(
                (model_path.to_path_buf(), device.to_owned()),
                resolution.passed,
            );
        Some(KvPoolInputs {
            profile: std::sync::Arc::new(profile),
            resolution,
            path,
        })
    }

    fn kv_passed(&self, model_path: &Path, kind: ModelKind, device: &str) -> Option<String> {
        if !matches!(kind, ModelKind::TextGen | ModelKind::Vision) {
            return None;
        }
        let cached = self
            .kv_resolved
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&(model_path.to_path_buf(), device.to_owned()))
            .copied();
        if let Some(p) = cached {
            return Some(p.to_owned());
        }
        // Not loaded yet in this process (a candidate model the prune pass asks about):
        // resolve it the way a load would, without logging, and remember the answer so a
        // sweep does not parse the IR again.
        let path = if kind == ModelKind::Vision {
            super::kv_precision::EnginePath::Vlm
        } else {
            super::kv_precision::EnginePath::Cb
        };
        let passed = if self.has_draft_model(model_path) {
            super::kv_precision::legacy_resolution(path, &self.kv_cache_precision).passed
        } else {
            let profile = super::kv_profile::KvProfile::read(model_path, device);
            super::kv_precision::resolve_kv_precision(
                profile.as_ref(),
                path,
                device,
                self.per_model_precision(model_path),
                &self.kv_cache_precision,
                self.kv_pass_mode,
            )
            .passed
        };
        self.kv_resolved
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert((model_path.to_path_buf(), device.to_owned()), passed);
        Some(passed.to_owned())
    }

    // Dispatch on ModelKind — one arm per variant; naturally grows with new kinds.
    #[allow(clippy::too_many_lines)]
    fn load(
        &self,
        model_path: &Path,
        device: &str,
        cache_size_gb: f64,
        max_num_seqs: usize,
    ) -> anyhow::Result<(EngineHandleKind, std::thread::JoinHandle<()>)> {
        let path_str = model_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?;
        // The model id is the leaf directory name (models_dir.join(model_id)).
        // Used only as a metrics label; fall back to the device if absent.
        let model_id = model_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(device);

        // Effective prefix-caching flag for this model: the per-model override,
        // else the global flag.
        let prefix_caching = self
            .prefix_caching_overrides
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(model_id)
            .copied()
            .unwrap_or(self.enable_prefix_caching);

        // OV compile-cache blob attribution (the project's internal engineering log):
        // snapshot `ov_cache_dir` before dispatch, diff after, so any new files
        // this load's `compile_model`/pipeline-construction call created can be
        // attributed to (model_id, device) in the manifest. Safe with no new
        // lock: T5.2's `load_lock` (mod.rs) already serialises this entire
        // function across every model, so no other load can be touching
        // `ov_cache_dir` concurrently. Kinds that don't pass `ov_cache_dir` to
        // their engine constructor (Embedding, Reranking) simply produce no
        // diff — harmless, not special-cased below.
        let kind = self.detect_kind(model_path);
        let kv = self.resolve_kv(model_path, model_id, kind, device);
        // Exact block count from blocks-first sizing (0 = size from the GiB budget).
        let kv_blocks = self
            .kv_blocks_hints
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(model_id)
            .copied()
            .unwrap_or(0);
        let CachePlan {
            engine_dir: kv_cache_dir,
            scope_subdir,
            attribution_root: cache_dir_path,
        } = plan_cache_dir(
            self.ov_cache_dir.as_deref(),
            kind,
            device,
            kv.passed,
            self.ov_cache_keyed,
        );
        // E-17: the KV engines compile into `<root>/kv-<tag>/`, created and proven writable
        // before the engine call (OpenVINO throws on a `CACHE_DIR` it cannot create). When it
        // cannot be made the engine gets no cache at all - never the root, where blobs of
        // another precision would be picked up again.
        let blobs_before = cache_dir_path
            .as_deref()
            .map(|r| crate::cache_manifest::snapshot_cache_scope(r, scope_subdir));

        let result = match kind {
            ModelKind::Vision => {
                // VLMPipeline runs a ContinuousBatchingPipeline internally
                // (VLMContinuousBatchingAdapter — confirmed via a live gdb
                // backtrace during the 2026-08-20 hang investigation) exactly
                // like the plain text-gen path below, so `cache_size_gb` AND
                // `enable_prefix_caching` both apply and must be threaded
                // through — previously dropped entirely, which left OpenVINO's
                // own defaults (dynamic/unbounded cache, prefix caching off)
                // in charge. cache_size_gb alone measured ~2x this project's
                // configured VRAM budget; prefix caching being off separately
                // measured a 30-40x per-turn cost on this server's own
                // resent-full-history chat pattern
                // (the project's internal engineering log —
                // two distinct bugs found in the same investigation).
                let cache_dir = kv_cache_dir.as_str();
                let registered = self
                    .concurrent_streams
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get(model_id)
                    .copied();
                let vlm_cap = resolve_vlm_channel_cap(registered, max_num_seqs);
                if cache_size_gb <= 0.0 && prefix_caching {
                    // Not reachable on this project's own fleet configs (every
                    // Vision entry sets a real kv_cache_gb) but reachable via
                    // documented-legitimate config elsewhere (`vram_gb: 0.0`,
                    // no per-model kv_cache_gb → global default 0.0) — flagged
                    // in the 2026-08-20 adversarial review of this exact code.
                    // cache_size_gb<=0.0 skips scheduler_config entirely below
                    // (ov_bridge.cpp's OvVlmState), so prefix caching silently
                    // stays off regardless of this `true` — the safer of two
                    // silent outcomes (the CB path's equivalent combination
                    // instead enables dynamic-cache-plus-retained-blocks,
                    // itself a real unbounded-VRAM hazard), but still worth a
                    // loud line instead of a silent one.
                    tracing::warn!(
                        model_id,
                        "Vision model has no positive cache_size_gb — prefix caching \
                         will NOT be applied (OpenVINO's un-set default) even though \
                         enable_prefix_caching=true, so every turn will reprocess the \
                         full conversation history from scratch. Set kv_cache_gb for \
                         this model to get real per-turn cost."
                    );
                }
                let engine = crate::ov_vlm::OvVlmEngine::new(
                    path_str,
                    device,
                    cache_dir,
                    cache_size_gb,
                    kv_blocks,
                    prefix_caching,
                    // The vision engine thread runs one generation at a time (`run_generate` in
                    // `vlm_engine::spawn_vlm_engine`), so the scheduler needs room for one
                    // sequence. GenAI's default of 256 reserves linear-attention state for all of
                    // them and made a hybrid model fail to load without prefix caching (D-02).
                    // If the vision engine ever becomes concurrent, raise this with it.
                    1,
                    kv.passed,
                )?;
                if self.kv_warmup
                    && warm_up_applies(device)
                    && let Err(e) = warm_up_vlm(&engine, model_id)
                {
                    return Err(leak_if_poisoned(engine, e, model_id));
                }
                let queue_timeout =
                    std::time::Duration::from_millis(self.admission_queue_timeout_ms);
                // A tokenizer of its own, so /tokenize does not queue behind a generation (D-13).
                // Best effort: without it the engine-thread route still works, only slower.
                let tokenizer = match crate::ov_tokenizer::OvTokenizer::new(std::path::Path::new(
                    path_str,
                )) {
                    Ok(t) => Some(std::sync::Arc::new(t)),
                    Err(e) => {
                        tracing::warn!(model_id, error = %e, "standalone tokenizer not loaded; /tokenize will wait behind generations");
                        None
                    }
                };
                let (handle, thread) = crate::vlm_engine::spawn_vlm_engine(
                    engine,
                    model_id,
                    device,
                    vlm_cap,
                    queue_timeout,
                    tokenizer,
                )?;
                Ok((EngineHandleKind::Vision(handle), thread))
            }
            ModelKind::TextGen => {
                if device.starts_with("NPU") {
                    // NPU LLM: static LLMPipeline, single-stream. CB cannot target
                    // the NPU (proven empirically on a Lunar Lake NPU box). Requires
                    // a channel-wise int4 IR; group-wise fails NPU compile.
                    let max_prompt_len = self
                        .max_prompt_len_hints
                        .read()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .get(model_id)
                        .copied();
                    let min_response_len = self
                        .min_response_len_hints
                        .read()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .get(model_id)
                        .copied();
                    // `ov_cache_dir` → NPU `CACHE_DIR` (weightless blob by
                    // default). Measured 2026-09-28: ~6 s cached load vs 30.7 s
                    // cold, independent of the Level Zero driver cache
                    // (the project's internal engineering log). Unlike the NPU
                    // *Whisper* pipeline (see the Stt arm), the LLM pipeline
                    // caches cleanly.
                    let pipeline = crate::ov_pipeline::OvPipeline::new(
                        path_str,
                        device,
                        max_prompt_len,
                        min_response_len,
                        self.ov_cache_dir.as_deref(),
                    )?;
                    let queue_timeout =
                        std::time::Duration::from_millis(self.admission_queue_timeout_ms);
                    let (handle, thread) = crate::npu_engine::spawn_npu_engine(
                        pipeline,
                        model_id,
                        device,
                        queue_timeout,
                        // Resolved from the same values `OvPipeline::new` just
                        // compiled with — the handle's prompt gate reads this,
                        // so it can't drift from the resident graph.
                        crate::npu_engine::NpuShape::resolve(max_prompt_len, min_response_len),
                    )?;
                    // Early return skips the attribution at the end of this
                    // function, so attribute the NPU blob here.
                    self.attribute_new_blobs(
                        model_id,
                        device,
                        cache_dir_path.as_deref(),
                        scope_subdir,
                        blobs_before.as_ref(),
                    );
                    return Ok((EngineHandleKind::NpuTextGen(handle), thread));
                }
                let cache_dir = kv_cache_dir.as_str();
                if cache_size_gb <= 0.0 && prefix_caching {
                    // The OPPOSITE silent failure mode from the Vision arm's
                    // equivalent warning above, and the more dangerous of the
                    // two: SchedulerConfig with cache_size<=0 is "dynamic
                    // allocation" (scheduler_config.hpp), and combining that
                    // with enable_prefix_caching=true means retained KV blocks
                    // grow unbounded and untracked by this project's own VRAM
                    // budget (see ov_cb_create's doc comment in ov_bridge.cpp)
                    // — the exact unbounded-growth hazard behind the whole
                    // 2026-08-20 hang investigation, just reachable here via a
                    // documented-legitimate config (vram_gb: 0.0, no
                    // per-model kv_cache_gb override) rather than a bug.
                    // Flagged in that day's adversarial review; this project's
                    // own docs say "audit before adding a live config with
                    // dynamic allocation" — this line is that audit made loud
                    // instead of relying on the operator remembering it.
                    tracing::warn!(
                        model_id,
                        "TextGen model has no positive cache_size_gb with \
                         enable_prefix_caching=true — OpenVINO will use dynamic/ \
                         unbounded KV allocation AND retain prefix-cache blocks \
                         unbounded, untracked by this server's VRAM budget. Set \
                         kv_cache_gb for this model or this can grow VRAM without \
                         limit under sustained multi-turn traffic."
                    );
                }
                // Speculative decoding: a registered DraftHint attaches a draft
                // model at construction; absent (the common case) is byte-
                // identical to plain decoding ("", "", 0).
                let draft_hint = self
                    .draft_hints
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get(model_id)
                    .cloned();
                let (draft_path, draft_device, num_assistant_tokens) = match &draft_hint {
                    Some(h) => (
                        h.draft_path.as_str(),
                        h.draft_device.as_str(),
                        h.num_assistant_tokens,
                    ),
                    None => ("", "", 0),
                };
                let mut engine = match draft_hint.as_ref().filter(|h| h.verify_on_load) {
                    Some(hint) => verify_speculative_pairing(
                        path_str,
                        device,
                        max_num_seqs,
                        cache_size_gb,
                        kv.passed,
                        cache_dir,
                        prefix_caching,
                        hint,
                        model_id,
                    )
                    .map_err(|e| hint_if_ambiguous_kind(e, model_path))?,
                    None => crate::ov_cb::OvCbEngine::new(
                        path_str,
                        device,
                        max_num_seqs,
                        cache_size_gb,
                        kv_blocks,
                        kv.passed,
                        cache_dir,
                        prefix_caching,
                        draft_path,
                        draft_device,
                        num_assistant_tokens,
                    )
                    .map_err(|e| hint_if_ambiguous_kind(e, model_path))?,
                };
                // A draft-attached engine already ran probe generations (`verify_on_load`) or is
                // not worth a different warm-up; everything else gets its pool materialised now.
                if self.kv_warmup
                    && warm_up_applies(device)
                    && draft_hint.is_none()
                    && let Err(e) = warm_up_cb(&mut engine, model_id)
                {
                    return Err(leak_if_poisoned(engine, e, model_id));
                }
                let queue_timeout =
                    std::time::Duration::from_millis(self.admission_queue_timeout_ms);
                let (handle, thread) = crate::cb_engine::spawn_engine(
                    engine,
                    max_num_seqs,
                    queue_timeout,
                    model_id,
                    device,
                )?;
                Ok((EngineHandleKind::TextGen(handle), thread))
            }
            ModelKind::Embedding => {
                // R4/G3: the first real "media" flow. Single-stream
                // TextEmbeddingPipeline on a dedicated thread; `cache_size_gb`
                // does not apply (no CB-style KV pool). Phase C2: the embedding
                // device is resolved **once at registration** (per-model `device`
                // / `tier_preference` / `embedding_device` / the engine default)
                // and threaded in as `device` — the factory loads on exactly that,
                // no re-resolution. This keeps the record's device, memory domain,
                // and metrics label in agreement with the real load (closing the
                // The embedding-mislabel edge, where a re-resolve here could move
                // the model off the device the manager charged). A model loaded off
                // the inference device should carry `vram_gb: 0.0` so it is not
                // gated against the dGPU pool.
                let engine = crate::ov_embed::OvEmbedEngine::new(
                    path_str,
                    device,
                    self.embedding_pooling,
                    self.embedding_normalize,
                )?;
                let queue_timeout =
                    std::time::Duration::from_millis(self.admission_queue_timeout_ms);
                // L0 length gate (the project's internal engineering log):
                // resolved once here from config.json, not re-read per request.
                let max_seq_len = crate::ov_embed::resolve_max_seq_len(model_path);
                let (handle, thread) = crate::embed_engine::spawn_embed_engine(
                    engine,
                    model_id,
                    device,
                    queue_timeout,
                    max_seq_len,
                )?;
                Ok((EngineHandleKind::Embedding(handle), thread))
            }
            ModelKind::Stt => {
                // Phase 5.1c: the Whisper speech-to-text flow. Single-stream
                // `WhisperPipeline` on a dedicated thread; `cache_size_gb` does
                // not apply (no CB-style KV pool). Device is resolved-once at
                // registration and threaded in as `device` — same contract as the
                // embedding arm; the factory loads on exactly that device.
                //
                // NPU Whisper gets `CACHE_DIR` like every other pipeline. It did NOT
                // until 2026-10-02: a live test on 2026-08-27 saw an NPU-targeted
                // `WhisperPipeline` with a non-empty `ov_cache_dir` hang the load
                // past the supervisor's 90 s health timeout (upstream
                // openvino.genai#1992). On the current stack (GenAI 2026.2.1, this
                // NPU driver) that no longer reproduces: three cold loads with both
                // caches cleared took 3.5-4.1 s, a warm load 0.33 s, and the blob
                // lands in `ov_cache_dir`. Auto-language requests on the NPU are
                // answered by a lazily built CPU detector plus an explicit language
                // for the NPU pipeline, which is built once and never rebuilt (see
                // `OvWhisperState` in the bridge: rebuilding leaks 3 threads + 5
                // handles per cycle, Windows pass 2 D1). Not yet verified on Windows.
                let cache_dir = self.ov_cache_dir.as_deref().unwrap_or("");
                let engine = crate::ov_whisper::OvWhisperEngine::new(path_str, device, cache_dir)?;
                let queue_timeout =
                    std::time::Duration::from_millis(self.admission_queue_timeout_ms);
                let (handle, thread) = crate::pipelines::stt::spawn_stt_engine(
                    engine,
                    model_id,
                    device,
                    queue_timeout,
                    crate::ov_whisper::resolve_can_translate(model_path),
                )?;
                Ok((EngineHandleKind::Stt(handle), thread))
            }
            ModelKind::ImageGen => {
                // Phase 5.3c: the SDXL text-to-image flow. Single-stream
                // `Text2ImagePipeline` on a dedicated thread; `cache_size_gb`
                // does not apply (no CB-style KV pool). Device is resolved-once at
                // registration and threaded in as `device` — same contract as the
                // STT/embedding arms; the factory loads on exactly that device.
                let cache_dir = self.ov_cache_dir.as_deref().unwrap_or("");
                let engine = crate::ov_image::OvImageEngine::new(path_str, device, cache_dir)?;
                let provenance_hint = self
                    .image_provenance_hints
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get(model_id)
                    .cloned();
                let manifest_root =
                    crate::cache_manifest::manifest_root(self.ov_cache_dir.as_deref());
                let metadata = crate::pipelines::image::ImageModelMetadata::read(
                    model_path,
                    model_id,
                    manifest_root.as_deref(),
                )
                .with_provenance(provenance_hint);
                let (handle, thread) = crate::pipelines::image::spawn_image_engine(
                    engine, model_id, device, metadata,
                )?;
                Ok((EngineHandleKind::ImageGen(handle), thread))
            }
            ModelKind::Reranking => {
                // Cross-encoder reranking (/v1/rerank). Single-stream
                // TextRerankPipeline on a dedicated thread. Device resolved-once at
                // registration — same contract as the embedding arm.
                let engine = crate::ov_rerank::OvRerankEngine::new(path_str, device)?;
                // L0 length gate (the project's internal engineering log):
                // resolved once here from config.json, not re-read per request.
                // Reuses the embedding module's generic BERT-config reader.
                let max_seq_len = crate::ov_embed::resolve_max_seq_len(model_path);
                let (handle, thread) = crate::rerank_engine::spawn_rerank_engine(
                    engine,
                    model_id,
                    device,
                    max_seq_len,
                )?;
                Ok((EngineHandleKind::Reranking(handle), thread))
            }
            ModelKind::Decision => {
                // System One (/v1/systemone): direct InferRequest on a gathered decision
                // IR. The sidecar supplies the system prompt,
                // candidate ids and context length; the ids are verified against the
                // tokenizer before the engine is handed out. Only an explicit
                // `kind: "decision"` gets here — the IR looks like a plain LLM on disk.
                crate::decision_engine::check_device(device)?;
                let meta = crate::decision_engine::DecisionMeta::load(model_path)?;
                let cache_dir = self.ov_cache_dir.as_deref().unwrap_or("");
                let engine = crate::ov_decide::OvDecideEngine::new(
                    path_str,
                    device,
                    cache_dir,
                    meta.context_length,
                )?;
                meta.verify(&engine)?;
                let (handle, thread) =
                    crate::decision_engine::spawn_decision_engine(engine, meta, model_id, device)?;
                Ok((EngineHandleKind::Decision(handle), thread))
            }
            ModelKind::Tts => match detect_tts_backend(model_path)? {
                TtsBackend::Kokoro {
                    onnx_name,
                    voices_name,
                } => {
                    tracing::info!(
                        model_id = %model_id,
                        onnx = %onnx_name,
                        voices = %voices_name,
                        "Kokoro-82M TTS detected"
                    );
                    let onnx_path = model_path.join(onnx_name).to_string_lossy().into_owned();
                    let voices_path = model_path.join(voices_name).to_string_lossy().into_owned();
                    let queue_timeout =
                        std::time::Duration::from_millis(self.admission_queue_timeout_ms);
                    let (handle, thread) = crate::pipelines::tts::spawn_kokoro_engine(
                        onnx_path,
                        voices_path,
                        model_id,
                        device,
                        queue_timeout,
                    )?;
                    Ok((EngineHandleKind::Tts(handle), thread))
                }
                TtsBackend::CoquiVits {
                    onnx_name,
                    tokens_name,
                } => {
                    tracing::info!(
                        model_id = %model_id,
                        "Coqui VITS (Polish) TTS detected"
                    );
                    let onnx_path = model_path.join(onnx_name).to_string_lossy().into_owned();
                    let tokens_path = model_path.join(tokens_name).to_string_lossy().into_owned();
                    let queue_timeout =
                        std::time::Duration::from_millis(self.admission_queue_timeout_ms);
                    let (handle, thread) = crate::pipelines::tts::spawn_coqui_vits_engine(
                        onnx_path,
                        tokens_path,
                        model_id,
                        device,
                        queue_timeout,
                    )?;
                    Ok((EngineHandleKind::Tts(handle), thread))
                }
                TtsBackend::Piper { onnx, config } => {
                    tracing::info!(
                        model_id = %model_id,
                        onnx = %onnx.display(),
                        "Piper TTS voice detected (phonemes via external espeak-ng)"
                    );
                    let queue_timeout =
                        std::time::Duration::from_millis(self.admission_queue_timeout_ms);
                    let (handle, thread) = crate::pipelines::tts::spawn_piper_engine(
                        onnx.to_string_lossy().into_owned(),
                        &config.to_string_lossy(),
                        model_id,
                        device,
                        queue_timeout,
                    )?;
                    Ok((EngineHandleKind::Tts(handle), thread))
                }
                TtsBackend::SpeechT5 => {
                    // Phase 5.2a: OpenVINO Text2SpeechPipeline.
                    let cache_dir = self.ov_cache_dir.as_deref().unwrap_or("");
                    let engine = crate::ov_tts::OvTtsEngine::new(path_str, device, cache_dir)?;
                    let queue_timeout =
                        std::time::Duration::from_millis(self.admission_queue_timeout_ms);
                    let (handle, thread) = crate::pipelines::tts::spawn_tts_engine(
                        engine,
                        model_id,
                        device,
                        queue_timeout,
                    )?;
                    Ok((EngineHandleKind::Tts(handle), thread))
                }
            },
        };

        if result.is_ok() {
            self.attribute_new_blobs(
                model_id,
                device,
                cache_dir_path.as_deref(),
                scope_subdir,
                blobs_before.as_ref(),
            );
        }

        result
    }
}

impl OvEngineFactory {
    /// Record the `ov_cache_dir` files that appeared during a successful load
    /// as this model's blobs in the cache manifest (a before/after directory
    /// diff). No-op without a cache dir or when nothing new was written.
    fn attribute_new_blobs(
        &self,
        model_id: &str,
        device: &str,
        cache_dir_path: Option<&Path>,
        scope_subdir: Option<&str>,
        blobs_before: Option<&std::collections::HashSet<String>>,
    ) {
        let (Some(cache_dir_path), Some(blobs_before)) = (cache_dir_path, blobs_before) else {
            return;
        };
        let blobs_after = crate::cache_manifest::snapshot_cache_scope(cache_dir_path, scope_subdir);
        let new_blobs: Vec<String> = blobs_after.difference(blobs_before).cloned().collect();
        if !new_blobs.is_empty()
            && let Some(root) = crate::cache_manifest::manifest_root(self.ov_cache_dir.as_deref())
        {
            crate::cache_manifest::write_device_blobs_entry(&root, model_id, device, &new_blobs);
        }
    }
}

impl OvEngineFactory {
    /// Whether `model_dir`'s kind is knowable from an explicit hint, i.e.
    /// `detect_kind` would return something other than by falling all the
    /// way through to file-marker sniffing. Only meant for `detect_kind`
    /// itself — **not** a signal of confidence: `build_not_loaded_record`
    /// (mod.rs) unconditionally calls `register_kind_hint` with whatever
    /// `detect_kind` already resolved, including its own ambiguous `TextGen`
    /// fallback, so a hint being present here proves nothing about whether it
    /// was ever actually confident. `hint_if_ambiguous_kind` deliberately
    /// does NOT use this — see `detect_kind_from_markers`.
    fn explicit_kind_hint(&self, model_dir: &Path) -> Option<ModelKind> {
        model_dir.file_name().and_then(|s| s.to_str()).and_then({
            |name| {
                self.model_kinds
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get(name)
                    .copied()
            }
        })
    }
}

/// The blob-cache subdirectory (relative to `ov_cache_dir`) a model of `kind` on `device`
/// compiles into: `Ok(None)` for the root (E-17).
///
/// Keyed only for the engines whose KV precision the plugin hashes out of the blob id: the
/// continuous-batching text engine and the vision pipeline (tag = the precision string the
/// loader passes, `passed_kv`; `kv-auto` when it passes none), on GPU and CPU. The NPU engines and every
/// kind without a KV cache keep the root, as does `flat` layout. Used by the loader and by the
/// prune pass, so both agree on where a model's active blobs live.
///
/// # Errors
/// A precision string with no tag. The caller must then give the engine no cache at all: the
/// root is where blobs of every precision mix.
pub(crate) fn kv_cache_subdir_for(
    kind: ModelKind,
    device: &str,
    passed_kv: &str,
    keyed: bool,
) -> anyhow::Result<Option<&'static str>> {
    if !keyed || device.starts_with("NPU") {
        return Ok(None);
    }
    if !matches!(kind, ModelKind::TextGen | ModelKind::Vision) {
        return Ok(None);
    }
    crate::cache_manifest::kv_cache_subdir(passed_kv).map(Some)
}

/// What a load hands the engine and what it attributes afterwards.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct CachePlan {
    /// The `CACHE_DIR` string for the engine; `""` = no cache.
    pub engine_dir: String,
    /// The keyed subdirectory the engine compiles into, if any.
    pub scope_subdir: Option<&'static str>,
    /// The cache root the load's before/after diff runs against; `None` = attribute nothing.
    pub attribution_root: Option<std::path::PathBuf>,
}

/// Chooses (and creates) the cache directory for one load. A keyed directory that cannot be
/// created or written, or a precision without a tag, yields no cache at all - never the root.
pub(crate) fn plan_cache_dir(
    ov_cache_dir: Option<&str>,
    kind: ModelKind,
    device: &str,
    passed_kv: &str,
    keyed: bool,
) -> CachePlan {
    let none = CachePlan {
        engine_dir: String::new(),
        scope_subdir: None,
        attribution_root: None,
    };
    let Some(root) = ov_cache_dir.filter(|d| !d.is_empty()).map(Path::new) else {
        return none;
    };
    let sub = match kv_cache_subdir_for(kind, device, passed_kv, keyed) {
        Ok(sub) => sub,
        Err(e) => {
            tracing::warn!(error = %e, "OV blob cache disabled for this model");
            return none;
        }
    };
    let Some(sub) = sub else {
        return CachePlan {
            engine_dir: root.to_string_lossy().into_owned(),
            scope_subdir: None,
            attribution_root: Some(root.to_path_buf()),
        };
    };
    match crate::cache_manifest::ensure_writable_subdir(root, sub) {
        Ok(dir) => CachePlan {
            engine_dir: dir.to_string_lossy().into_owned(),
            scope_subdir: Some(sub),
            attribution_root: Some(root.to_path_buf()),
        },
        Err(e) => {
            warn_unusable_cache_dir_once(&root.join(sub), &e);
            none
        }
    }
}

/// One WARN per (directory, process) when a keyed cache directory cannot be made.
fn warn_unusable_cache_dir_once(dir: &Path, err: &std::io::Error) {
    static SEEN: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashSet<std::path::PathBuf>>,
    > = std::sync::OnceLock::new();
    let seen = SEEN.get_or_init(Default::default);
    let first = seen
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(dir.to_path_buf());
    if first {
        tracing::warn!(
            dir = %dir.display(),
            error = %err,
            "OV blob cache directory is unusable - this model compiles without a blob cache"
        );
    }
}

/// What `config.json` says an encoder-only or speech model is: a `*ForTextToSpeech` architecture is TTS,
/// a one-label `*ForSequenceClassification` is a reranker, and a bare `*Model` of an encoder family
/// (BERT, XLM-RoBERTa, ...) is an embedder. `None` for everything else - decoders (`*ForCausalLM`),
/// multi-label classifiers and unknown families keep the plain-LLM default and its hint.
fn kind_from_hf_config(model_dir: &Path) -> Option<ModelKind> {
    const ENCODERS: [&str; 8] = [
        "bert",
        "xlm-roberta",
        "roberta",
        "distilbert",
        "electra",
        "modernbert",
        "deberta-v2",
        "nomic_bert",
    ];
    let text = std::fs::read_to_string(model_dir.join("config.json")).ok()?;
    let cfg: serde_json::Value = serde_json::from_str(&text).ok()?;
    let arch = cfg.get("architectures")?.as_array()?.first()?.as_str()?;
    let family = cfg.get("model_type").and_then(serde_json::Value::as_str)?;
    if arch.ends_with("ForTextToSpeech") {
        return Some(ModelKind::Tts);
    }
    if !ENCODERS.contains(&family) {
        return None;
    }
    if arch.ends_with("ForSequenceClassification") {
        let labels = cfg
            .get("id2label")
            .and_then(serde_json::Value::as_object)
            .map(serde_json::Map::len)
            .or_else(|| {
                cfg.get("num_labels")
                    .and_then(serde_json::Value::as_u64)
                    .map(|n| usize::try_from(n).unwrap_or(usize::MAX))
            });
        return (labels.unwrap_or(1) == 1).then_some(ModelKind::Reranking);
    }
    arch.ends_with("Model").then_some(ModelKind::Embedding)
}

/// The kind `model_dir`'s own files confidently prove, independent of any
/// hint cache: `Some` for `Stt`'s encoder/decoder split, `ImageGen`'s
/// `model_index.json` + submodel dir, or `Vision`'s vision-embeddings IR;
/// `None` when nothing matched — the one case `detect_kind` cannot actually
/// tell apart from a plain LLM (embedding/reranking/tts models ship the
/// identical single `openvino_model.xml`, no marker file to sniff).
/// Deliberately excludes `OvEngineFactory::explicit_kind_hint`'s cache: that
/// cache stores `detect_kind`'s own ambiguous `TextGen` fallback just as
/// readily as a real signal (`build_not_loaded_record`'s unconditional
/// `register_kind_hint` call, mod.rs) — treating "cached" as "confident"
/// would make a merely-remembered guess look like proof the moment
/// `add_model`/`reload_config` has registered a model once, defeating the
/// whole point of `hint_if_ambiguous_kind`'s check (confirmed live: a first
/// `add_model` attempt that failed on an unrelated admission check still
/// poisoned the hint cache with the ambiguous `TextGen` guess before ever
/// reaching this function, silencing the hint on retry). A free function,
/// not a method: it touches only the filesystem, not factory state, and
/// this keeps `OvEngineFactory::hint_if_ambiguous_kind` from having to
/// (mis)read the hint cache to answer a question the cache cannot answer.
fn detect_kind_from_markers(model_dir: &Path) -> Option<ModelKind> {
    // SpeechT5 ships an encoder + decoder IR too (so the Whisper rule below would claim it), but
    // also a vocoder and a postnet that no speech-to-text model has: those, or a text-to-speech
    // architecture in config.json, make it a TTS model. Checked FIRST.
    if model_dir.join("openvino_vocoder.xml").is_file()
        || model_dir.join("openvino_postnet.xml").is_file()
    {
        return Some(ModelKind::Tts);
    }
    if let Some(kind) = kind_from_hf_config(model_dir) {
        return Some(kind);
    }
    // A Whisper OV GenAI dir ships a SEPARATE encoder + decoder IR
    // (`openvino_encoder_model.xml` + `openvino_decoder_model.xml`) — the
    // encoder/decoder split is unique to the speech models; a plain LLM ships
    // a single `openvino_model.xml` and a VLM its vision-embeddings IR. Both
    // files must be present so a model that happens to ship only one does not
    // misclassify. (Phase 5.1c.)
    if model_dir.join("openvino_encoder_model.xml").is_file()
        && model_dir.join("openvino_decoder_model.xml").is_file()
    {
        return Some(ModelKind::Stt);
    }
    // A Text2Image (SDXL/SD/SD3/FLUX) OV GenAI dir ships a `model_index.json`
    // plus a `unet/` (SD/SDXL) or `transformer/` (SD3/FLUX) submodel subdir —
    // the multi-submodel diffusion layout, never present in an LLM/VLM/Whisper
    // dir. Both must be present so a stray `model_index.json` does not
    // misclassify. (Phase 5.3c.)
    if model_dir.join("model_index.json").is_file()
        && (model_dir.join("unet").is_dir() || model_dir.join("transformer").is_dir())
    {
        return Some(ModelKind::ImageGen);
    }
    // A VLM ships `openvino_vision_embeddings_model.xml` — present in all
    // three supported layouts (Qwen2.5-VL, Qwen3-VL, InternVL2.5) and never
    // in a plain LLM directory. This is the check that used to live in
    // `main.rs::is_vlm_model`.
    if model_dir
        .join("openvino_vision_embeddings_model.xml")
        .is_file()
    {
        return Some(ModelKind::Vision);
    }
    None
}

/// If `err` is `OpenVINO`'s specific "this graph has no KV-cache state
/// variables, cannot convert to paged attention" rejection — the exact
/// failure a plain BERT-style encoder model (embedding/reranking) hits when
/// it gets built as a `TextGen` CB pipeline — AND `model_dir`'s files carry
/// no confident kind marker (`detect_kind_from_markers`, deliberately
/// ignoring the hint cache — see its own doc comment for why), append an
/// actionable hint pointing at `kind` instead of leaving the caller with
/// only `OpenVINO`'s own C++ transform-pass error text. A no-op for every
/// other failure (OOM, missing files, wrong device, ...) and for a model
/// whose kind a file marker actually confirms — `SDPAToPagedAttention` is
/// specific enough that a real, correctly-classified model hitting it would
/// indicate an unrelated conversion bug, not a kind mismatch, and this hint
/// would only be noise there.
fn hint_if_ambiguous_kind(err: anyhow::Error, model_dir: &Path) -> anyhow::Error {
    let msg = err.to_string();
    let is_stateless_rejection =
        msg.contains("SDPAToPagedAttention") || msg.contains("supposed to be stateful");
    if is_stateless_rejection && detect_kind_from_markers(model_dir).is_none() {
        return anyhow::anyhow!(
            "{err}\n\nHint: this model's kind defaulted to \"text_gen\" because its directory \
             has no distinguishing marker files — embedding, reranking, and TTS models are \
             indistinguishable from a plain LLM by file-sniffing alone. If this is actually one \
             of those, retry with an explicit kind (\"embedding\" / \"reranking\" / \"tts\")."
        );
    }
    err
}

/// Two-layer VLM concurrency resolution: layer 2 (`registered`, a per-model
/// `max_concurrent_streams` hint) wins if set, else layer 1 (`max_num_seqs`,
/// the pipeline default — `0` means "not set"), else `None` (caller falls
/// back to [`crate::vlm_engine::VLM_CHANNEL_CAP`]).
///
/// Kept a free function (not a method) so the precedence is unit-testable
/// without spinning up a real `OvEngineFactory` or GPU.
fn resolve_vlm_channel_cap(registered: Option<usize>, max_num_seqs: usize) -> Option<usize> {
    registered.or((max_num_seqs > 0).then_some(max_num_seqs))
}

/// Low-entropy, RAG-shaped probe prompt for the speculative-decoding
/// load-time self-check (Part 7). Deliberately NOT open-ended: fact #8
/// (the project's internal engineering log) showed
/// open-ended prompts are exactly where a *working* draft pairing is
/// expected to diverge from plain decoding, which would false-positive this
/// gate on the feature's best pairings. Every pairing tested during the
/// investigation (dense and `MoE`, win and loss) held byte-identical on this
/// prompt shape.
const SPECULATIVE_PROBE_PROMPT: &str = "Context: The Eiffel Tower was completed in 1889 for the \
World's Fair in Paris. It stands 330 meters tall and was designed by Gustave Eiffel's engineering \
company.\n\nQuestion: In what year was the Eiffel Tower completed, and how tall is it?\n\nAnswer \
in one sentence.";

const SPECULATIVE_PROBE_MAX_NEW_TOKENS: usize = 64;

/// Run a fixed, deterministic (greedy — `GenParams::default()` is greedy)
/// generation through `engine` and return the concatenated decoded text.
/// Load-time self-check only (Part 7) — never part of the request-serving
/// hot path.
fn run_probe(
    engine: &mut crate::ov_cb::OvCbEngine,
    prompt: &str,
    max_new_tokens: usize,
) -> anyhow::Result<String> {
    // A sentinel far outside the range `AppState::next_request_id` (which
    // starts at 0) will ever hand out, so the probe's request id can never
    // collide with a real client request on the serving engine this probe's
    // draft-attached build gets reused as.
    const PROBE_REQUEST_ID: u64 = u64::MAX;
    let params = crate::ov_cb::GenParams {
        max_new_tokens,
        ..Default::default()
    };
    engine.add_request(PROBE_REQUEST_ID, prompt, &params)?;
    let mut output = String::new();
    while engine.has_unfinished() {
        engine.step(|_id, delta, _new_tokens, _finish, _pool_exhausted| output.push_str(delta))?;
    }
    Ok(output)
}

/// Whether a load on `device` gets the KV warm-up: GPU-class devices only (the pool of a CPU or
/// NPU engine is host memory or a fixed compiled shape, and nothing there can poison a GPU).
fn warm_up_applies(device: &str) -> bool {
    device.to_ascii_uppercase().starts_with("GPU")
}

/// A GPU context poisoned by an out-of-resources failure must show at load, not at the first
/// request; any other warm-up failure is logged and the load goes on (the first real request
/// will report a real problem with a real prompt).
fn warm_up_outcome(
    model_id: &str,
    started: std::time::Instant,
    r: anyhow::Result<()>,
) -> anyhow::Result<()> {
    match r {
        Ok(()) => {
            tracing::info!(
                model_id,
                warm_up_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
                "KV warm-up generation complete (pool materialised)"
            );
            Ok(())
        }
        Err(e) if crate::model_manager::is_gpu_poison_error(&format!("{e:#}")) => {
            Err(e.context("KV warm-up generation hit an out-of-resources failure"))
        }
        // MR-2: a vision pool is built at its first generate (this one); GenAI refusing it for
        // device memory must fail the load, not leave a Ready model whose first request 500s.
        Err(e) if crate::model_manager::is_genai_memory_budget_error(&format!("{e:#}")) => {
            Err(e.context("KV warm-up generation: GenAI refused the KV cache"))
        }
        Err(e) => {
            tracing::warn!(model_id, error = %format!("{e:#}"), "KV warm-up generation failed; continuing");
            Ok(())
        }
    }
}

/// A failed load-time probe's engine (warm-up, speculative self-check): on a GPU-poison failure
/// it is leaked, not dropped. Destroying a pipeline on a poisoned context is the destructor path the poison exit avoids (it can hang or
/// abort); the process gives the memory back when it exits, and every GPU model is unavailable
/// until then anyway (0.9.0 review F). Any other failure drops it normally.
fn leak_if_poisoned<E>(engine: E, e: anyhow::Error, model_id: &str) -> anyhow::Error {
    if crate::model_manager::is_gpu_poison_error(&format!("{e:#}")) {
        tracing::warn!(
            model_id,
            "GPU context poisoned during a load-time probe (KV warm-up or speculative \
             self-check) — the engine is left alive on purpose \
             (not destroyed on the poisoned context); its memory is released when the process \
             exits"
        );
        std::mem::forget(engine);
    } else {
        drop(engine);
    }
    e
}

/// One-token generation on a freshly built text engine: materialises the pool and shows a broken
/// GPU at load time.
fn warm_up_cb(engine: &mut crate::ov_cb::OvCbEngine, model_id: &str) -> anyhow::Result<()> {
    let started = std::time::Instant::now();
    let r = run_probe(engine, "Hello", 1).map(|_| ());
    warm_up_outcome(model_id, started, r)
}

/// One-token text-only generation on a freshly built vision engine (the pool of a vision model is
/// allocated at its first generate, not at construction).
fn warm_up_vlm(engine: &crate::ov_vlm::OvVlmEngine, model_id: &str) -> anyhow::Result<()> {
    let started = std::time::Instant::now();
    let params = crate::ov_cb::GenParams {
        max_new_tokens: 1,
        ..Default::default()
    };
    let messages = [serde_json::json!({"role": "user", "content": "Hello"})];
    let r = engine
        .generate(&messages, &[], None, None, &params, |_| false)
        .map(|_| ());
    warm_up_outcome(model_id, started, r)
}

/// Speculative decoding load-time greedy-equivalence self-check (Part 7 of
/// the project's internal engineering log, design question 5's
/// safety net). Builds a plain engine, probes it, drops it (synchronous VRAM
/// release, per the hard rule) so the draft-attached build below is the only
/// pipeline resident at once, then builds the draft-attached engine and
/// probes that too.
///
/// A mismatch is a silent-breakage class this API can genuinely produce (the
/// `MoE` finding, the project's internal engineering log 2026-07-17) — the model must never come up
/// in degraded-silent mode, so a divergence fails the load loudly rather than
/// serving wrong output. Note (Fable review, 2026-07-19): a PASS proves
/// correctness, never benefit — `verify_on_load` is a safety net, not a
/// benefit oracle.
///
/// On success, returns the already-built draft-attached engine so the caller
/// reuses it as the serving engine instead of paying for a third pipeline
/// construction — the probe's request has already fully finished (and been
/// evicted from the KV pool) by the time this returns.
///
/// # Errors
/// Returns an error if either engine fails to build or generate, or if the
/// two probes disagree.
#[allow(clippy::too_many_arguments)]
fn verify_speculative_pairing(
    path_str: &str,
    device: &str,
    max_num_seqs: usize,
    cache_size_gb: f64,
    kv_cache_precision: &str,
    cache_dir: &str,
    enable_prefix_caching: bool,
    hint: &DraftHint,
    model_id: &str,
) -> anyhow::Result<crate::ov_cb::OvCbEngine> {
    let mut plain_engine = crate::ov_cb::OvCbEngine::new(
        path_str,
        device,
        max_num_seqs,
        cache_size_gb,
        0,
        kv_cache_precision,
        cache_dir,
        enable_prefix_caching,
        "",
        "",
        0,
    )
    .context("speculative self-check: failed to build the plain probe engine")?;
    // A poison found here must not run the engine's destructor on the poisoned context.
    let plain_output = match run_probe(
        &mut plain_engine,
        SPECULATIVE_PROBE_PROMPT,
        SPECULATIVE_PROBE_MAX_NEW_TOKENS,
    )
    .context("speculative self-check: plain probe generation failed")
    {
        Ok(out) => out,
        Err(e) => return Err(leak_if_poisoned(plain_engine, e, model_id)),
    };
    drop(plain_engine);

    let mut draft_engine = crate::ov_cb::OvCbEngine::new(
        path_str,
        device,
        max_num_seqs,
        cache_size_gb,
        0,
        kv_cache_precision,
        cache_dir,
        enable_prefix_caching,
        &hint.draft_path,
        &hint.draft_device,
        hint.num_assistant_tokens,
    )
    .context("speculative self-check: failed to build the draft-attached engine")?;
    // A poison found here must not run the engine's destructor on the poisoned context.
    let assisted_output = match run_probe(
        &mut draft_engine,
        SPECULATIVE_PROBE_PROMPT,
        SPECULATIVE_PROBE_MAX_NEW_TOKENS,
    )
    .context("speculative self-check: assisted probe generation failed")
    {
        Ok(out) => out,
        Err(e) => return Err(leak_if_poisoned(draft_engine, e, model_id)),
    };

    anyhow::ensure!(
        plain_output == assisted_output,
        "speculative decoding self-check FAILED for model {model_id:?}: plain and \
         draft-attached decode diverge on the load-time probe — this is the same \
         silent-breakage class as the MoE target whose speculative output diverged \
         from plain greedy decoding without any error, \
         not expected divergence (the probe is deliberately RAG-shaped, not \
         open-ended). The model will NOT be loaded. Plain: {plain_output:?} \
         Assisted: {assisted_output:?}. Remove the `speculative` block, or set \
         `verify_on_load: false` only after confirming this divergence is benign \
         on this exact box/driver/OV version.",
    );

    tracing::info!(
        model_id,
        "speculative decoding self-check passed: plain and draft-attached decode agree on probe"
    );
    Ok(draft_engine)
}

// ---- Mock implementation (tests only) --------------------------------

/// Mock engine factory for unit tests — never touches the GPU.
///
/// Creates a real `tokio::sync::mpsc` channel pair so the returned
/// `EngineHandle` is fully functional. The "engine thread" simply drains
/// all incoming commands and exits cleanly when the handle is dropped.
///
/// Set `should_fail = true` to simulate a generic load failure (bad path, etc.).
/// Set `fail_with_oom = true` to simulate a `CL_OUT_OF_RESOURCES` GPU OOM — the
/// kind that poisons the `OpenCL` context (L3 trigger). Both imply `should_fail`.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct MockEngineFactory {
    /// When `true`, `load()` returns a generic error instead of a handle.
    pub should_fail: bool,
    /// When `true`, `load()` returns a `CL_OUT_OF_RESOURCES` error, simulating
    /// the GPU context-poisoning OOM that triggers the L3 gate.
    pub fail_with_oom: bool,
    /// `(path substring, message)`: `load()` of a model whose path contains the substring fails
    /// with that message (a real C++ error text), every other load succeeds.
    pub fail_message_for: Option<(String, String)>,
    /// What [`EngineFactory::kv_passed`] answers (`None` = the trait default: the manager falls
    /// back to the global string). Lets a test give a model its own blob-cache directory.
    pub kv_passed: Option<String>,
    /// What [`EngineFactory::kv_pool_inputs`] answers (`None` = no exact sizing).
    pub pool_inputs: Option<KvPoolInputs>,
    /// The block counts passed to [`EngineFactory::register_kv_blocks_hint`], in order.
    pub blocks_hints: std::sync::Mutex<Vec<Option<u64>>>,
}

#[cfg(test)]
impl EngineFactory for MockEngineFactory {
    fn kv_passed(&self, _model_path: &Path, _kind: ModelKind, _device: &str) -> Option<String> {
        self.kv_passed.clone()
    }

    fn kv_pool_inputs(
        &self,
        _model_path: &Path,
        _kind: ModelKind,
        _device: &str,
    ) -> Option<KvPoolInputs> {
        self.pool_inputs.clone()
    }

    fn engine_versions(&self) -> Option<(String, String)> {
        Some((
            "2026.4.1-22982-e213a147257".to_owned(),
            "2026.4.1.0-3408-ab01d1771e4".to_owned(),
        ))
    }

    fn register_kv_blocks_hint(&self, _model_id: &str, blocks: Option<u64>) {
        self.blocks_hints
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(blocks);
    }

    /// Name-based kind detection (no GPU): the model id's leaf name selects the
    /// kind by a `-`/`_`-delimited **segment** prefix — a segment beginning with
    /// `vlm`→`Vision`, `embed`→`Embedding`, `stt`→`Stt`, `tts`→`Tts`,
    /// `imagegen`→`ImageGen`; a name with no kind segment→`TextGen`. Lets a test
    /// `ModelManager` hold any mix of kinds (mirroring how the production factory
    /// inspects the model directory) without touching disk — the keystone that
    /// makes every kind's lifecycle/routing/metrics unit-testable.
    ///
    /// T3/F8b: matching whole segments (not raw substrings) and **panicking on
    /// an ambiguous name** replaces the old branch-order substring chain, where
    /// e.g. `embedding-stt-test` silently resolved `Embedding` over `Stt` purely
    /// because the `embed` branch came first. STT/TTS are the new media kinds, so
    /// that ambiguity would have bitten media tests; now it is loud at the call
    /// site and order no longer decides.
    fn detect_kind(&self, model_dir: &Path) -> ModelKind {
        // (keyword, kind) table — checked against each delimited segment. Listed
        // for a deterministic ambiguity message, not for precedence (a name that
        // matches two kinds panics rather than silently preferring an earlier row).
        const KINDS: &[(&str, ModelKind)] = &[
            ("embed", ModelKind::Embedding),
            ("rerank", ModelKind::Reranking),
            ("decide", ModelKind::Decision),
            ("stt", ModelKind::Stt),
            ("tts", ModelKind::Tts),
            ("imagegen", ModelKind::ImageGen),
            ("vlm", ModelKind::Vision),
        ];
        let name = model_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let mut matched: Option<ModelKind> = None;
        for segment in name.split(['-', '_']) {
            for &(keyword, kind) in KINDS {
                if segment.starts_with(keyword) {
                    assert!(
                        matched.is_none_or(|m| m == kind),
                        "ambiguous mock model name {name:?}: matches two kinds \
                         ({:?} and {kind:?}) — give the test model exactly one \
                         kind segment",
                        matched.unwrap_or(kind),
                    );
                    matched = Some(kind);
                }
            }
        }
        matched.unwrap_or(ModelKind::TextGen)
    }

    fn load(
        &self,
        model_path: &Path,
        _device: &str,
        _cache_size_gb: f64,
        _max_num_seqs: usize,
    ) -> anyhow::Result<(EngineHandleKind, std::thread::JoinHandle<()>)> {
        if self.fail_with_oom {
            return Err(anyhow::anyhow!(
                "ov_cb_create failed: CL_OUT_OF_RESOURCES (-5) from ProgramBuilder build failed"
            ));
        }
        if self.should_fail {
            return Err(anyhow::anyhow!("mock load failure (should_fail = true)"));
        }
        if let Some((needle, msg)) = &self.fail_message_for
            && model_path.to_string_lossy().contains(needle.as_str())
        {
            return Err(anyhow::anyhow!("{msg}"));
        }

        match self.detect_kind(model_path) {
            ModelKind::Vision => {
                // GPU-free Vision engine: drains Generate, answers with a token
                // + Done, and shares the handle's in-flight counter.
                let leaf = model_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("mock-vlm");
                let (handle, thread) = crate::vlm_engine::spawn_mock_vlm(leaf);
                Ok((EngineHandleKind::Vision(handle), thread))
            }
            ModelKind::TextGen => {
                // Build a real channel so EngineHandle works correctly.
                let (tx, mut rx) =
                    tokio::sync::mpsc::channel::<crate::cb_engine::EngineCommand>(32);

                // Drain commands until the sender is dropped, then exit.
                // Tokenize/Detokenize are answered with empty results so callers
                // that use them (L0 gate, /tokenize endpoint) get a valid response
                // rather than a channel-dropped error.
                let thread = std::thread::spawn(move || {
                    while let Some(cmd) = rx.blocking_recv() {
                        match cmd {
                            crate::cb_engine::EngineCommand::Tokenize { reply, .. } => {
                                let _ = reply.send(Ok(vec![]));
                            }
                            crate::cb_engine::EngineCommand::Detokenize { reply, .. } => {
                                let _ = reply.send(Ok(String::new()));
                            }
                            crate::cb_engine::EngineCommand::AddRequest { .. } => {} // drain without processing
                        }
                    }
                });

                Ok((
                    EngineHandleKind::TextGen(crate::cb_engine::EngineHandle::from_sender(tx)),
                    thread,
                ))
            }
            ModelKind::Embedding => {
                // GPU-free embedding engine: shares the handle's in-flight
                // counter and answers each batch with a fixed small vector per
                // input, so the full embeddings flow is unit-testable.
                let leaf = model_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("mock-embed");
                let (handle, thread) = crate::embed_engine::spawn_mock_embed(leaf);
                Ok((EngineHandleKind::Embedding(handle), thread))
            }
            ModelKind::Stt => {
                // GPU-free STT engine: answers each request with a canned
                // transcription (no audio decode, no GPU) and shares the handle's
                // in-flight counter, so the full transcription flow is testable.
                let leaf = model_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("mock-stt");
                let (handle, thread) = crate::pipelines::stt::spawn_mock_stt(leaf);
                Ok((EngineHandleKind::Stt(handle), thread))
            }
            ModelKind::ImageGen => {
                // GPU-free image engine: answers each request with `n` canned
                // PNGs (no GPU) and shares the handle's in-flight counter, so the
                // full generation flow is testable.
                let leaf = model_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("mock-image");
                let (handle, thread) = crate::pipelines::image::spawn_mock_image(leaf);
                Ok((EngineHandleKind::ImageGen(handle), thread))
            }
            ModelKind::Tts => {
                // GPU-free mock TTS engine: answers each request with silence so
                // the full TTS flow (handler → channel → WAV encode) is testable.
                let leaf = model_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("mock-tts");
                let (handle, thread) = crate::pipelines::tts::spawn_mock_tts(leaf);
                Ok((EngineHandleKind::Tts(handle), thread))
            }
            ModelKind::Reranking => {
                // GPU-free mock reranking engine: returns documents in reverse
                // order with synthetic scores so the full reranking flow is testable.
                let leaf = model_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("mock-rerank");
                let (handle, thread) = crate::rerank_engine::spawn_mock_rerank(leaf);
                Ok((EngineHandleKind::Reranking(handle), thread))
            }
            ModelKind::Decision => {
                // GPU-free mock decision engine: one token per prompt character (so tests
                // hit the length boundaries exactly) and logit j for candidate j.
                let leaf = model_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("mock-decide");
                let (handle, thread) = crate::decision_engine::spawn_mock_decision(leaf);
                Ok((EngineHandleKind::Decision(handle), thread))
            }
        }
    }
}

// ============================================================
// Unit tests — mock kind detection (T3/F8b)
// ============================================================
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// 0.9.0 review F: a poison-class warm-up failure leaks the engine (no destructor on the
    /// poisoned context); any other failure drops it.
    #[test]
    fn leak_if_poisoned_skips_the_destructor_only_on_poison() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        struct Counted;
        impl Drop for Counted {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        let poison = anyhow::anyhow!("[GPU] clEnqueue failed: CL_OUT_OF_RESOURCES");
        let e = leak_if_poisoned(Counted, poison, "m");
        assert!(format!("{e:#}").contains("CL_OUT_OF_RESOURCES"));
        assert_eq!(DROPS.load(Ordering::SeqCst), 0, "poisoned: not dropped");
        let _ = leak_if_poisoned(Counted, anyhow::anyhow!("tokenizer said no"), "m");
        assert_eq!(DROPS.load(Ordering::SeqCst), 1, "other failure: dropped");
    }

    fn model_dir_with_config(json: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), json).unwrap();
        dir
    }

    /// A model dir with `config.json` and the named (empty) marker files.
    fn model_dir_with(config: &str, files: &[&str]) -> tempfile::TempDir {
        let dir = model_dir_with_config(config);
        for f in files {
            std::fs::write(dir.path().join(f), b"").unwrap();
        }
        dir
    }

    /// The layouts seen on the QA machines (2026-10-13): `SpeechT5` was claimed by the Whisper rule
    /// (encoder + decoder IR), and the embedder / reranker had no marker at all.
    #[test]
    fn kind_is_sniffed_for_tts_reranker_and_embedder_but_not_for_decoders() {
        let speecht5 = model_dir_with(
            r#"{"architectures":["SpeechT5ForTextToSpeech"],"model_type":"speecht5"}"#,
            &[
                "openvino_encoder_model.xml",
                "openvino_decoder_model.xml",
                "openvino_postnet.xml",
                "openvino_vocoder.xml",
            ],
        );
        assert_eq!(
            detect_kind_from_markers(speecht5.path()),
            Some(ModelKind::Tts)
        );
        let reranker = model_dir_with(
            r#"{"architectures":["XLMRobertaForSequenceClassification"],"model_type":"xlm-roberta","id2label":{"0":"LABEL_0"}}"#,
            &["openvino_model.xml"],
        );
        assert_eq!(
            detect_kind_from_markers(reranker.path()),
            Some(ModelKind::Reranking)
        );
        let e5 = model_dir_with(
            r#"{"architectures":["XLMRobertaModel"],"model_type":"xlm-roberta"}"#,
            &["openvino_model.xml"],
        );
        assert_eq!(
            detect_kind_from_markers(e5.path()),
            Some(ModelKind::Embedding)
        );
        // A multi-label classifier is neither a reranker nor an embedder; decoders stay LLMs.
        let classifier = model_dir_with(
            r#"{"architectures":["BertForSequenceClassification"],"model_type":"bert","id2label":{"0":"a","1":"b","2":"c"}}"#,
            &[],
        );
        assert_eq!(detect_kind_from_markers(classifier.path()), None);
        let llm = model_dir_with(
            r#"{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}"#,
            &["openvino_model.xml"],
        );
        assert_eq!(detect_kind_from_markers(llm.path()), None);
        // GPT-2's `GPT2LMHeadModel` also ends in "Model" but is not an encoder family.
        let gpt2 = model_dir_with(
            r#"{"architectures":["GPT2LMHeadModel"],"model_type":"gpt2"}"#,
            &[],
        );
        assert_eq!(detect_kind_from_markers(gpt2.path()), None);
        // Whisper keeps its own rule.
        let whisper = model_dir_with(
            r#"{"architectures":["WhisperForConditionalGeneration"],"model_type":"whisper"}"#,
            &["openvino_encoder_model.xml", "openvino_decoder_model.xml"],
        );
        assert_eq!(
            detect_kind_from_markers(whisper.path()),
            Some(ModelKind::Stt)
        );
    }

    fn kind_of(name: &str) -> ModelKind {
        MockEngineFactory::default().detect_kind(Path::new(name))
    }

    /// Every kind segment resolves, including the real fleet test names, and a
    /// name with no kind segment falls through to `TextGen`.
    #[test]
    fn detect_kind_resolves_each_segment() {
        assert_eq!(kind_of("bge-embed-ov"), ModelKind::Embedding);
        assert_eq!(kind_of("whisper-stt-ov"), ModelKind::Stt);
        assert_eq!(kind_of("kokoro-tts"), ModelKind::Tts);
        assert_eq!(kind_of("sdxl-imagegen"), ModelKind::ImageGen);
        assert_eq!(kind_of("qwen-vlm-8b"), ModelKind::Vision);
        assert_eq!(kind_of("qwen3-8b-int4-ov"), ModelKind::TextGen);
    }

    /// The keyword must be a delimited *segment*, not a raw substring lurking in
    /// another word — `tts` inside `attts` would be a segment of its own only if
    /// delimited, so an embedded run does not false-trigger.
    #[test]
    fn detect_kind_matches_segments_not_substrings() {
        // "latte" contains "tts"? no — but "latte-art" has no kind segment.
        assert_eq!(kind_of("latte-art-model"), ModelKind::TextGen);
        // "embedding" is a segment beginning with "embed" → Embedding.
        assert_eq!(kind_of("e5-embedding-large"), ModelKind::Embedding);
        // "attts" DOES contain "tts" as a raw substring (positions 2-4), but
        // does not START a delimited segment with it — segment matching must
        // NOT false-trigger here, unlike naive substring matching, which
        // would misclassify this as Tts.
        assert_eq!(kind_of("attts-model"), ModelKind::TextGen);
    }

    /// Layer 1 (`max_num_seqs`) is used as the VLM channel cap when layer 2
    /// (`max_concurrent_streams`) is not registered for the model.
    #[test]
    fn vlm_uses_max_num_seqs_as_layer1_default() {
        assert_eq!(resolve_vlm_channel_cap(None, 16), Some(16));
    }

    /// A registered per-model override (layer 2) wins over the pipeline
    /// default (layer 1) even when both are set.
    #[test]
    fn vlm_layer2_override_wins_over_layer1() {
        assert_eq!(resolve_vlm_channel_cap(Some(2), 16), Some(2));
    }

    /// `max_num_seqs == 0` means "not set" (the CB convention for "let OV
    /// decide") — it must not resolve to `Some(0)`, which would zero out the
    /// VLM channel and deadlock every request.
    #[test]
    fn vlm_zero_max_num_seqs_is_not_set() {
        assert_eq!(resolve_vlm_channel_cap(None, 0), None);
    }

    #[test]
    fn detect_tts_backend_finds_kokoro_by_voices_bin() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("voices.bin"), b"").expect("write voices.bin");
        std::fs::write(dir.path().join("kokoro-v1.0.onnx"), b"").expect("write onnx");
        assert_eq!(
            detect_tts_backend(dir.path()).expect("detect"),
            TtsBackend::Kokoro {
                onnx_name: "kokoro-v1.0.onnx",
                voices_name: "voices.bin",
            }
        );
    }

    #[test]
    fn detect_tts_backend_errors_when_kokoro_voices_present_but_onnx_missing() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("voices.bin"), b"").expect("write voices.bin");
        assert!(detect_tts_backend(dir.path()).is_err());
    }

    #[test]
    fn detect_tts_backend_finds_coqui_vits_by_model_onnx_and_tokens_txt() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("model.onnx"), b"").expect("write model.onnx");
        std::fs::write(dir.path().join("tokens.txt"), b"").expect("write tokens.txt");
        assert_eq!(
            detect_tts_backend(dir.path()).expect("detect"),
            TtsBackend::CoquiVits {
                onnx_name: "model.onnx",
                tokens_name: "tokens.txt",
            }
        );
    }

    /// A Piper voice (`<name>.onnx` + an espeak `<name>.onnx.json`) is
    /// detected by its config, ahead of the Coqui `model.onnx` check.
    #[test]
    fn detect_tts_backend_finds_piper_voice() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("vi_VN-vais1000-medium.onnx"), b"").expect("write onnx");
        std::fs::write(
            dir.path().join("vi_VN-vais1000-medium.onnx.json"),
            r#"{"audio":{"sample_rate":22050},"espeak":{"voice":"vi"},
                "inference":{"noise_scale":0.667,"length_scale":1,"noise_w":0.8},
                "num_speakers":1,"phoneme_type":"espeak","phoneme_id_map":{"_":[0]}}"#,
        )
        .expect("write config");
        assert_eq!(
            detect_tts_backend(dir.path()).expect("detect"),
            TtsBackend::Piper {
                onnx: dir.path().join("vi_VN-vais1000-medium.onnx"),
                config: dir.path().join("vi_VN-vais1000-medium.onnx.json"),
            }
        );
    }

    #[test]
    fn detect_tts_backend_falls_back_to_speecht5_when_dir_is_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert_eq!(
            detect_tts_backend(dir.path()).expect("detect"),
            TtsBackend::SpeechT5
        );
    }

    /// The F8b regression: a name carrying two distinct kind segments no longer
    /// silently resolves the earliest branch — it panics so the test author
    /// renames, rather than the order of the match arms deciding the kind.
    #[test]
    #[should_panic(expected = "ambiguous mock model name")]
    fn detect_kind_panics_on_ambiguous_name() {
        let _ = kind_of("embedding-stt-test");
    }

    /// The warm-up fails a load for a GPU out-of-resources failure (and, since MR-2, for `GenAI`'s
    /// device-memory refusal: see the test below); anything else is logged and the load goes on.
    /// Only GPU devices are warmed.
    #[test]
    fn warm_up_fails_the_load_for_a_poisoned_gpu_not_for_other_errors() {
        let t = std::time::Instant::now();
        assert!(warm_up_outcome("m", t, Ok(())).is_ok());
        assert!(warm_up_outcome("m", t, Err(anyhow::anyhow!("template exploded"))).is_ok());
        let err = warm_up_outcome(
            "m",
            t,
            Err(anyhow::anyhow!("clEnqueueWriteBuffer: CL_OUT_OF_RESOURCES")),
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("out-of-resources"), "{err:#}");
        assert!(warm_up_applies("GPU.1") && warm_up_applies("gpu"));
        assert!(!warm_up_applies("CPU") && !warm_up_applies("NPU"));
    }

    /// A per-model precision override beats the global key and the rule on a plain GPU device,
    /// and is ignored (global string, pass-through) on CPU; clearing it restores the default.
    #[test]
    fn a_per_model_precision_override_wins_on_gpu_and_is_ignored_on_cpu() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("m-8b");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(
            model.join("config.json"),
            r#"{"num_hidden_layers":4,"num_attention_heads":8,"num_key_value_heads":8,
                "head_dim":128,"hidden_size":1024}"#,
        )
        .unwrap();
        let mut f = test_factory();
        f.kv_cache_precision = "i8".to_owned();
        f.register_kv_precision_hint("m-8b", Some("u4"));
        let gpu_inputs = f
            .kv_pool_inputs(&model, ModelKind::TextGen, "GPU.1")
            .unwrap();
        assert_eq!(
            (gpu_inputs.resolution.passed, gpu_inputs.resolution.source),
            (
                "u4",
                crate::model_manager::kv_precision::PrecisionSource::PerModel
            )
        );
        assert_eq!(
            f.kv_passed(&model, ModelKind::TextGen, "GPU.1").as_deref(),
            Some("u4")
        );
        let cpu_inputs = f.kv_pool_inputs(&model, ModelKind::TextGen, "CPU").unwrap();
        assert_eq!(
            cpu_inputs.resolution.passed, "i8",
            "the CPU keeps the global string"
        );
        // An explicit per-model "auto" opts this model out of the global key (Block B2b): the
        // IR hint or the rule decides, whatever the global says.
        f.register_kv_precision_hint("m-8b", Some(""));
        let auto = f
            .kv_pool_inputs(&model, ModelKind::TextGen, "GPU.1")
            .unwrap()
            .resolution;
        assert_eq!(
            auto.source,
            crate::model_manager::kv_precision::PrecisionSource::AutoRule,
            "not the global key"
        );
        // No per-model value inherits the global key again.
        f.register_kv_precision_hint("m-8b", None);
        assert_eq!(
            f.kv_pool_inputs(&model, ModelKind::TextGen, "GPU.1")
                .unwrap()
                .resolution
                .passed,
            "i8"
        );
    }

    /// Fable review 2: a PATCH of a Ready model's precision registers a new hint but changes
    /// nothing the running engine was built with; the directory the prune pass protects (what
    /// `kv_passed` answers) stays the one the engine compiled into until the next load resolves it
    /// afresh.
    #[test]
    fn a_precision_patch_keeps_the_old_blob_directory_protected_until_reload() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("m-8b");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(
            model.join("config.json"),
            r#"{"num_hidden_layers":4,"num_attention_heads":8,"num_key_value_heads":8,
                "head_dim":128,"hidden_size":1024}"#,
        )
        .unwrap();
        let f = test_factory();
        f.register_kv_precision_hint("m-8b", Some("u4"));
        // The load resolves and remembers what it passed.
        let loaded = f.resolve_kv(&model, "m-8b", ModelKind::TextGen, "GPU.1");
        assert_eq!(loaded.passed, "u4");
        // PATCH: the hint changes; the engine does not.
        f.register_kv_precision_hint("m-8b", Some("i8"));
        assert_eq!(
            f.kv_passed(&model, ModelKind::TextGen, "GPU.1").as_deref(),
            Some("u4"),
            "the running engine still compiles into kv-u4: that directory stays protected"
        );
        // The next load resolves afresh and the answer moves.
        let reloaded = f.resolve_kv(&model, "m-8b", ModelKind::TextGen, "GPU.1");
        assert_eq!(reloaded.passed, "i8");
        assert_eq!(
            f.kv_passed(&model, ModelKind::TextGen, "GPU.1").as_deref(),
            Some("i8")
        );
    }

    /// A model with a speculative-decoding draft keeps the 0.8.x pass-through: the draft engine
    /// inherits the main engine's properties, so nothing about the precision is decided for it
    /// (owner decision P6), it is never sized exactly, and the prune pass is told the same string.
    #[test]
    fn a_model_with_a_draft_keeps_the_legacy_pass_through() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("main-8b");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(
            model.join("config.json"),
            r#"{"num_hidden_layers":4,"num_attention_heads":8,"num_key_value_heads":8,
                "head_dim":128,"hidden_size":1024}"#,
        )
        .unwrap();
        for (global, want) in [("", ""), ("i8", "i8")] {
            let mut f = test_factory();
            f.kv_cache_precision = global.to_owned();
            f.draft_hints.write().unwrap().insert(
                "main-8b".to_owned(),
                DraftHint {
                    draft_path: "/x/draft".to_owned(),
                    draft_device: "GPU.1".to_owned(),
                    num_assistant_tokens: 4,
                    verify_on_load: false,
                },
            );
            assert_eq!(
                f.kv_passed(&model, ModelKind::TextGen, "GPU.1").as_deref(),
                Some(want)
            );
            assert!(
                f.kv_pool_inputs(&model, ModelKind::TextGen, "GPU.1")
                    .is_none()
            );
            let r = f.resolve_kv(&model, "main-8b", ModelKind::TextGen, "GPU.1");
            assert_eq!(r.passed, want);
            assert_eq!(
                r.source,
                crate::model_manager::kv_precision::PrecisionSource::LegacyPassthrough
            );
            // Without the draft the same model is resolved by the rule (no IR here: i8 passes
            // only when the profile is trusted, which an IR-less profile is not).
            f.draft_hints.write().unwrap().clear();
            assert!(
                f.kv_pool_inputs(&model, ModelKind::TextGen, "GPU.1")
                    .is_some()
            );
        }
    }

    fn test_factory() -> OvEngineFactory {
        OvEngineFactory {
            kv_cache_precision: String::new(),
            kv_pass_mode: crate::model_manager::kv_precision::PassMode::Explicit,
            kv_resolved: std::sync::Mutex::new(HashMap::new()),
            kv_blocks_hints: std::sync::RwLock::new(HashMap::new()),
            kv_warmup: false,
            kv_precision_overrides: std::sync::RwLock::new(HashMap::new()),
            admission_queue_timeout_ms: 0,
            model_kinds: std::sync::RwLock::new(HashMap::new()),
            concurrent_streams: std::sync::RwLock::new(HashMap::new()),
            prefix_caching_overrides: std::sync::RwLock::new(HashMap::new()),
            max_prompt_len_hints: std::sync::RwLock::new(HashMap::new()),
            min_response_len_hints: std::sync::RwLock::new(HashMap::new()),
            draft_hints: std::sync::RwLock::new(HashMap::new()),
            image_provenance_hints: std::sync::RwLock::new(HashMap::new()),
            embedding_pooling: crate::ov_embed::Pooling::Mean,
            embedding_normalize: false,
            ov_cache_dir: None,
            ov_cache_keyed: false,
            enable_prefix_caching: true,
        }
    }

    /// `add_model` registers a kind hint for a runtime-added model before its
    /// first `load()` — this is the fix for the bug where a dynamically-added
    /// embedding/tts/reranking model (file-indistinguishable from a plain LLM)
    /// silently defaulted to `TextGen` and failed deep inside `OpenVINO`.
    #[test]
    fn register_kind_hint_makes_detect_kind_recognize_a_runtime_added_model() {
        let factory = test_factory();
        let dir = Path::new("/nonexistent/embed-model-added-at-runtime");

        // Before the hint: no marker files on disk, so it falls through to the
        // TextGen default — exactly the bug (a silent misclassification).
        assert_eq!(factory.detect_kind(dir), ModelKind::TextGen);

        factory.register_kind_hint("embed-model-added-at-runtime", ModelKind::Embedding);

        assert_eq!(factory.detect_kind(dir), ModelKind::Embedding);
    }

    #[test]
    fn detect_kind_from_markers_is_none_for_a_marker_less_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("openvino_model.xml"), b"").expect("write");
        // A plain LLM and an embedding/reranking model both ship exactly this
        // one file — genuinely ambiguous, which is the whole point of this test.
        assert_eq!(detect_kind_from_markers(dir.path()), None);
        assert_eq!(test_factory().detect_kind(dir.path()), ModelKind::TextGen);
    }

    #[test]
    fn detect_kind_from_markers_is_some_for_a_marker_that_exists() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("openvino_vision_embeddings_model.xml"), b"")
            .expect("write");
        assert_eq!(
            detect_kind_from_markers(dir.path()),
            Some(ModelKind::Vision)
        );
    }

    /// The regression this whole split exists for: `register_kind_hint` gets
    /// called unconditionally by `build_not_loaded_record` (mod.rs) with
    /// whatever `detect_kind` resolved, including its own ambiguous `TextGen`
    /// fallback — so a model whose first registration attempt already ran
    /// (e.g. failed on an unrelated admission check before ever reaching a
    /// real load) has a `TextGen` hint cached despite never being confidently
    /// classified. `hint_if_ambiguous_kind` must still fire in exactly this
    /// case — checking the hint cache instead of `detect_kind_from_markers`
    /// would silently suppress the hint the moment a caller has retried
    /// (confirmed live against a real server before this test existed).
    #[test]
    fn hint_if_ambiguous_kind_still_fires_after_the_hint_cache_was_poisoned_by_a_prior_attempt() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("openvino_model.xml"), b"").expect("write");
        let factory = test_factory();
        let model_id = dir
            .path()
            .file_name()
            .and_then(|s| s.to_str())
            .expect("tempdir has a name")
            .to_owned();

        // Simulates build_not_loaded_record's unconditional register_kind_hint
        // call on a prior attempt that never even reached the real load.
        factory.register_kind_hint(&model_id, ModelKind::TextGen);
        assert_eq!(
            factory.explicit_kind_hint(dir.path()),
            Some(ModelKind::TextGen)
        );

        let err = anyhow::anyhow!("... SDPAToPagedAttention ...");
        let hinted = hint_if_ambiguous_kind(err, dir.path());
        assert!(hinted.to_string().contains("Hint:"));
    }

    /// The exact failure a plain BERT-style encoder model (embedding/
    /// reranking) hits when the caller omitted `kind` and it silently
    /// defaulted to `TextGen`: `OpenVINO` rejects building a paged-attention
    /// pipeline around a stateless graph. With no marker files present (the
    /// genuinely ambiguous case), the hint gets appended.
    #[test]
    fn hint_if_ambiguous_kind_appends_a_hint_when_evidence_is_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = anyhow::anyhow!(
            "ov_cb_create failed: ... Model is supposed to be stateful, cannot perform \
             the SDPAToPagedAttention transformation."
        );
        let hinted = hint_if_ambiguous_kind(err, dir.path());
        let msg = hinted.to_string();
        assert!(
            msg.contains("SDPAToPagedAttention"),
            "original error text must survive"
        );
        assert!(msg.contains("Hint:"), "must append the actionable hint");
        assert!(msg.contains("kind"));
    }

    /// A model whose kind IS confirmed by a marker file hitting this same
    /// `OpenVINO` error would mean a real, unrelated conversion bug in that
    /// model — not a kind mismatch — so the hint must not fire and add noise.
    #[test]
    fn hint_if_ambiguous_kind_is_a_noop_when_evidence_exists() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("openvino_vision_embeddings_model.xml"), b"")
            .expect("write");
        let err = anyhow::anyhow!("... SDPAToPagedAttention ...");
        let hinted = hint_if_ambiguous_kind(err, dir.path());
        assert!(!hinted.to_string().contains("Hint:"));
    }

    /// An unrelated failure (OOM, bad device, missing files, ...) must never
    /// get this hint tacked on, even for a kind-ambiguous directory.
    #[test]
    fn hint_if_ambiguous_kind_is_a_noop_for_an_unrelated_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = anyhow::anyhow!("CL_OUT_OF_RESOURCES (-5) from ProgramBuilder build failed");
        let hinted = hint_if_ambiguous_kind(err, dir.path());
        assert!(!hinted.to_string().contains("Hint:"));
    }

    /// MR-2: a vision pool is built at the warm-up generate; `GenAI` refusing it for device memory
    /// fails the load (not a Ready model whose first request 500s). Other failures still pass.
    #[test]
    fn a_genai_memory_budget_refusal_at_warm_up_fails_the_load() {
        let raw = "Check 'budget_in_bytes <= total_available_memory' failed at \
                   /home/jenkins/agent/src/cache_orchestrator.hpp:759";
        let r = warm_up_outcome("vlm", std::time::Instant::now(), Err(anyhow::anyhow!(raw)));
        let e = r.expect_err("must fail the load");
        assert!(crate::model_manager::is_genai_memory_budget_error(
            &format!("{e:#}")
        ));
        let other = anyhow::anyhow!("some unrelated generate failure");
        assert!(warm_up_outcome("vlm", std::time::Instant::now(), Err(other)).is_ok());
    }

    /// Which engines compile into a `kv-<tag>` directory (U-20): the KV engines on GPU and
    /// CPU, by the passed string (the vision pipeline too: `kv-auto` when it passes none, as on
    /// CPU or with `kv_precision_pass: plugin`); NPU, kinds without a KV cache
    /// and the flat layout stay in the root; an untagged precision is an error, not the root.
    #[test]
    fn kv_engines_get_a_directory_per_passed_precision() {
        use ModelKind::*;
        let f = |k, d, p, keyed| kv_cache_subdir_for(k, d, p, keyed).unwrap();
        assert_eq!(f(TextGen, "GPU.1", "", true), Some("kv-auto"));
        assert_eq!(f(TextGen, "GPU.1", "i8", true), Some("kv-i8"));
        assert_eq!(f(TextGen, "GPU.1", "u4", true), Some("kv-u4"));
        assert_eq!(f(TextGen, "GPU.1", "f16", true), Some("kv-f16"));
        assert_eq!(f(TextGen, "CPU", "i8", true), Some("kv-i8"));
        assert_eq!(f(TextGen, "HETERO:GPU.0,GPU.1", "u4", true), Some("kv-u4"));
        assert_eq!(f(Vision, "GPU.1", "u4", true), Some("kv-u4"));
        assert_eq!(f(Vision, "GPU.1", "", true), Some("kv-auto"));
        assert_eq!(f(Vision, "CPU", "", true), Some("kv-auto"));
        assert_eq!(f(TextGen, "NPU", "i8", true), None);
        assert_eq!(f(TextGen, "NPU.0", "i8", true), None);
        assert_eq!(f(Vision, "NPU", "", true), None);
        for kind in [Embedding, Stt, Tts, Decision] {
            assert_eq!(f(kind, "GPU.1", "i8", true), None, "{kind:?}");
        }
        assert_eq!(f(TextGen, "GPU.1", "i8", false), None);
        assert_eq!(f(Vision, "GPU.1", "", false), None);
        assert!(kv_cache_subdir_for(TextGen, "GPU.1", "bf16", true).is_err());
    }

    /// What the loader actually hands the engine (U-C26, U-C27, U-21): per kind and layout, the
    /// directory and the attribution scope; a blocked directory means no cache and no
    /// attribution, never the root.
    #[test]
    fn loader_cache_plan_follows_kind_device_and_layout() {
        use ModelKind::*;
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_str().unwrap();
        let plan = |k, d, p, keyed| plan_cache_dir(Some(root), k, d, p, keyed);

        let cb = plan(TextGen, "GPU.1", "u4", true);
        assert_eq!(cb.engine_dir, format!("{root}/kv-u4"));
        assert_eq!(cb.scope_subdir, Some("kv-u4"));
        assert_eq!(cb.attribution_root.as_deref(), Some(tmp.path()));
        assert!(tmp.path().join("kv-u4").is_dir(), "created up front");

        let vlm = plan(Vision, "GPU.1", "u4", true);
        assert_eq!(vlm.engine_dir, format!("{root}/kv-u4"));
        assert_eq!(vlm.scope_subdir, Some("kv-u4"));
        let vlm_plugin = plan(Vision, "GPU.1", "", true);
        assert_eq!(vlm_plugin.engine_dir, format!("{root}/kv-auto"));

        for (kind, dev) in [
            (TextGen, "NPU"),
            (Vision, "NPU"),
            (Stt, "GPU.1"),
            (Tts, "CPU"),
            (Embedding, "GPU.0"),
        ] {
            let p = plan(kind, dev, "i8", true);
            assert_eq!(
                (p.engine_dir.as_str(), p.scope_subdir),
                (root, None),
                "{kind:?} {dev}"
            );
        }
        let flat = plan(TextGen, "GPU.1", "i8", false);
        assert_eq!((flat.engine_dir.as_str(), flat.scope_subdir), (root, None));

        // A regular file in the way: no cache, nothing to attribute, root untouched.
        std::fs::write(tmp.path().join("kv-i8"), b"x").unwrap();
        let blocked = plan(TextGen, "GPU.1", "i8", true);
        assert_eq!(
            blocked,
            CachePlan {
                engine_dir: String::new(),
                scope_subdir: None,
                attribution_root: None
            }
        );
        // A precision without a tag: also no cache.
        assert_eq!(plan(TextGen, "GPU.1", "bf16", true).engine_dir, "");
        // No cache directory configured.
        assert_eq!(
            plan_cache_dir(None, TextGen, "GPU.1", "i8", true).engine_dir,
            ""
        );
        assert_eq!(
            plan_cache_dir(Some(""), TextGen, "GPU.1", "i8", true).engine_dir,
            ""
        );
    }
}
