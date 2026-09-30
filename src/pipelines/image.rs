// ============================================================
// src/pipelines/image.rs — text-to-image engine thread + handle (Phase 5.3c)
// ============================================================
// The convergence step for Image: wires the Text2Image FFI (`crate::ov_image`,
// 5.3b) to the HTTP handler through a real, channel-backed engine handle — the
// image analogue of `crate::pipelines::stt`. Request/response (one prompt in →
// one or more images out, no token stream), so the command carries a `oneshot`
// reply channel and the handle holds a cancellation-safe `InFlightGuard`.
//
//   - One dedicated OS thread owns the `OvImageEngine` (`Text2ImagePipeline`).
//   - The thread GENERATES the images and PNG-ENCODES them (`image_util`) before
//     replying — all CPU-heavy pixel work stays off the async runtime, on the
//     same dedicated thread the denoising loop runs on.
//   - Commands arrive via an mpsc channel (capacity `IMAGE_CHANNEL_CAP`); one job
//     runs at a time (the pipeline is single-stream and blocking).
//   - `in_flight` keeps the engine honest to `/metrics`, the same counter the
//     embedding/STT engines use (shared `InFlightGuard`).
// ============================================================

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use tokio::sync::{mpsc::Sender, oneshot};

use crate::in_flight::InFlightGuard;
use crate::metrics::{HotMetrics, Modality};
use crate::model_manager::{ManagedEngine, ModelKind};
use crate::ov_image::{ImageEditOptions, ImageGenOptions, OvImageEngine, RgbImage};

/// Image command-channel capacity. The pipeline runs one generation at a time,
/// so this is also the reported concurrency cap (`max_concurrency`): up to this
/// many requests may be queued/in-flight before backpressure applies. Small — a
/// single SDXL generation holds the engine for seconds.
pub(crate) const IMAGE_CHANNEL_CAP: usize = 2;

// ── Commands ────────────────────────────────────────────────────────────────

pub(crate) enum ImageCommand {
    /// Generate (and PNG-encode) image(s) for one prompt.
    Generate {
        /// Positive prompt.
        prompt: String,
        /// Generation parameters (size, steps, n, seed, guidance, negative).
        opts: ImageGenOptions,
        /// One-shot reply channel — carries the PNG-encoded images back. Each
        /// `Vec<u8>` is a complete PNG file.
        reply: oneshot::Sender<anyhow::Result<Vec<Vec<u8>>>>,
        /// Wall-clock start (handler entry) for the duration metric.
        started_at: Instant,
    },
    /// Edit an existing image: inpaint (`mask` present) or img2img (`mask` absent).
    Edit {
        /// Positive prompt describing the desired result.
        prompt: String,
        /// Source image (NHWC RGB u8). Output size matches it.
        init: RgbImage,
        /// OV inpaint mask (white = regenerate). `None` → img2img over the frame.
        mask: Option<RgbImage>,
        /// Edit parameters (steps, n, seed, guidance, strength, negative).
        opts: ImageEditOptions,
        /// One-shot reply channel — the PNG-encoded images, like `Generate`.
        reply: oneshot::Sender<anyhow::Result<Vec<Vec<u8>>>>,
        /// Wall-clock start (handler entry) for the duration metric.
        started_at: Instant,
    },
}

// ── Handle ──────────────────────────────────────────────────────────────────

/// Cloneable submit handle for the image engine thread.
///
/// `generate` is request/response: it sends the prompt and awaits the reply. The
/// engine thread processes one job at a time — requests serialise in the channel.
#[derive(Clone, Debug)]
pub struct ImageHandle {
    tx: Sender<ImageCommand>,
    /// The model ID of the loaded image model (the response `model` field).
    model_id: Arc<str>,
    /// Model-directory-derived identity, read once at load time — never
    /// per request (the image-metadata plan).
    metadata: Arc<ImageModelMetadata>,
    /// Accepted-but-unfinished requests (queued + the one generating). Each
    /// request holds an [`InFlightGuard`] for its whole lifetime so the counter
    /// is the honest [`ManagedEngine::active`] value even when the awaiting task
    /// is cancelled (HTTP client disconnects mid-generation).
    in_flight: Arc<AtomicUsize>,
}

impl ImageHandle {
    /// The model ID of the loaded image model.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// The diffusers pipeline class read from `model_index.json` at load time
    /// (e.g. `"FluxPipeline"`, `"StableDiffusionXLPipeline"`), if available.
    #[must_use]
    pub fn model_family(&self) -> Option<&str> {
        self.metadata.model_family.as_deref()
    }

    /// `sha256:<hex>`-prefixed digest of the primary backbone weight file
    /// (`unet/` or `transformer/openvino_model.bin`), if one was found.
    #[must_use]
    pub fn model_hash(&self) -> Option<&str> {
        self.metadata.model_hash.as_deref()
    }

    /// The scheduler class name (`scheduler/scheduler_config.json`'s
    /// `_class_name`, e.g. `"LCMScheduler"`), if the file was readable.
    #[must_use]
    pub fn sampler(&self) -> Option<&str> {
        self.metadata
            .scheduler_config
            .as_deref()?
            .get("_class_name")?
            .as_str()
    }

    /// The raw `scheduler/scheduler_config.json` contents, passed through
    /// verbatim — no per-scheduler-algorithm parsing (the image-metadata plan
    /// §1a Finding 1).
    #[must_use]
    pub fn scheduler_config(&self) -> Option<&serde_json::Value> {
        self.metadata.scheduler_config.as_deref()
    }

    /// Operator-supplied precision label (Tier 3), if set in config.
    #[must_use]
    pub fn precision(&self) -> Option<&str> {
        self.metadata.precision.as_deref()
    }

    /// Operator-supplied HF source repo id (Tier 3), if set in config.
    #[must_use]
    pub fn model_source(&self) -> Option<&str> {
        self.metadata.model_source.as_deref()
    }

    /// Operator-supplied HF revision/commit sha (Tier 3), if set in config.
    #[must_use]
    pub fn model_revision(&self) -> Option<&str> {
        self.metadata.model_revision.as_deref()
    }

    /// Build a handle around an existing command channel — test seam only.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn from_sender(tx: Sender<ImageCommand>, model_id: &str) -> Self {
        Self {
            tx,
            model_id: Arc::from(model_id),
            metadata: Arc::new(ImageModelMetadata::default()),
            in_flight: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Generate image(s) for `prompt`, returning each as PNG-encoded bytes.
    ///
    /// # Errors
    /// Returns an error if the engine thread has exited (channel closed or the
    /// reply was dropped), the pipeline call failed, or PNG encoding failed.
    pub async fn generate(
        &self,
        prompt: String,
        opts: ImageGenOptions,
    ) -> anyhow::Result<Vec<Vec<u8>>> {
        let started_at = Instant::now();
        let (reply_tx, reply_rx) = oneshot::channel();

        // Increment now, decrement on Drop — covers every exit including task
        // cancellation across the awaits below.
        let _guard = InFlightGuard::new(&self.in_flight);

        self.tx
            .send(ImageCommand::Generate {
                prompt,
                opts,
                reply: reply_tx,
                started_at,
            })
            .await
            .map_err(|_| anyhow::anyhow!("image engine is not available"))?;

        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("image engine dropped the request"))?
    }

    /// Edit `init` toward `prompt`, returning each result as PNG bytes. Inpaint
    /// when `mask` is `Some` (white = regenerate), img2img when `None`.
    ///
    /// # Errors
    /// Returns an error if the engine thread has exited (channel closed or the
    /// reply was dropped), the model has no `vae_encoder`, the pipeline call
    /// failed, or PNG encoding failed.
    pub async fn edit(
        &self,
        prompt: String,
        init: RgbImage,
        mask: Option<RgbImage>,
        opts: ImageEditOptions,
    ) -> anyhow::Result<Vec<Vec<u8>>> {
        let started_at = Instant::now();
        let (reply_tx, reply_rx) = oneshot::channel();

        // Same cancellation-safe in-flight accounting as `generate`.
        let _guard = InFlightGuard::new(&self.in_flight);

        self.tx
            .send(ImageCommand::Edit {
                prompt,
                init,
                mask,
                opts,
                reply: reply_tx,
                started_at,
            })
            .await
            .map_err(|_| anyhow::anyhow!("image engine is not available"))?;

        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("image engine dropped the request"))?
    }
}

/// The image engine is a [`ModelKind::ImageGen`] engine. Uniform with the other
/// `ManagedEngine` impls so `ModelManager` reads metrics/lifecycle identically.
impl ManagedEngine for ImageHandle {
    fn kind(&self) -> ModelKind {
        ModelKind::ImageGen
    }

    fn active(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }

    fn max_concurrency(&self) -> usize {
        IMAGE_CHANNEL_CAP
    }

    /// Single-stream: whenever any request is in flight exactly one is running
    /// and the rest are queued. Honest waiting = in-flight minus the running one.
    fn waiting(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed).saturating_sub(1)
    }
}

// ── Engine thread ─────────────────────────────────────────────────────────────

/// Spawn a dedicated OS thread owning `engine` and return an [`ImageHandle`].
///
/// The thread exits cleanly when all handle clones are dropped (channel closes).
///
/// `metadata`: the model-directory-derived identity, already resolved by the
/// caller (`ImageModelMetadata::read`) — passed in rather than re-read here so
/// this function stays free of filesystem access.
///
/// # Errors
/// Only propagates thread-spawn errors (rare OS resource exhaustion).
pub fn spawn_image_engine(
    engine: OvImageEngine,
    model_id: &str,
    device: &str,
    metadata: ImageModelMetadata,
) -> anyhow::Result<(ImageHandle, std::thread::JoinHandle<()>)> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<ImageCommand>(IMAGE_CHANNEL_CAP);
    let model_id_arc: Arc<str> = Arc::from(model_id);
    let model_id_owned = model_id.to_owned();
    let device = device.to_owned();

    // Image joins the `kind="image_gen"` metric family. There is no token stream,
    // so only request-count and end-to-end duration are emitted. Image has no
    // dedicated `Modality` yet → `Modality::Text` (same placeholder as STT).
    let metrics = HotMetrics::new(ModelKind::ImageGen, model_id, &device);

    let thread = std::thread::Builder::new()
        .name(format!("image-engine-{model_id_owned}"))
        .spawn(move || {
            tracing::info!(
                model_id = model_id_owned,
                kind = ModelKind::ImageGen.label(),
                device,
                "image engine thread started"
            );
            while let Some(cmd) = rx.blocking_recv() {
                match cmd {
                    ImageCommand::Generate {
                        prompt,
                        opts,
                        reply,
                        started_at,
                    } => {
                        metrics.request_accepted(Modality::Text);
                        let result = run_generate(&engine, &model_id_owned, &prompt, &opts);
                        metrics.record_duration(Modality::Text, started_at.elapsed().as_secs_f64());
                        // Receiver may be gone if the client disconnected — ignore.
                        let _ = reply.send(result);
                    }
                    ImageCommand::Edit {
                        prompt,
                        init,
                        mask,
                        opts,
                        reply,
                        started_at,
                    } => {
                        metrics.request_accepted(Modality::Text);
                        let result = run_edit(
                            &engine,
                            &model_id_owned,
                            &prompt,
                            &init,
                            mask.as_ref(),
                            &opts,
                        );
                        metrics.record_duration(Modality::Text, started_at.elapsed().as_secs_f64());
                        let _ = reply.send(result);
                    }
                }
            }
            tracing::info!(
                model_id = model_id_owned,
                kind = ModelKind::ImageGen.label(),
                "image engine thread exiting — all handles dropped"
            );
        })?;

    Ok((
        ImageHandle {
            tx,
            model_id: model_id_arc,
            metadata: Arc::new(metadata),
            in_flight: Arc::new(AtomicUsize::new(0)),
        },
        thread,
    ))
}

/// Model-directory-derived identity for `generation_metadata`
/// (the image-metadata plan), read once at load time and cached on
/// [`ImageHandle`] — never per request. Each field independently omits
/// (`None`) rather than fabricates when its source file is missing or
/// unparseable.
#[derive(Debug, Default)]
pub struct ImageModelMetadata {
    /// `model_index.json`'s `_class_name` (Tier 1, e.g. `"FluxPipeline"`).
    model_family: Option<Arc<str>>,
    /// `sha256:<hex>` of the primary backbone weight file (Tier 2).
    model_hash: Option<Arc<str>>,
    /// Raw `scheduler/scheduler_config.json` contents (Tier 2) — `_class_name`
    /// within it is the `sampler` field ([`ImageHandle::sampler`]).
    scheduler_config: Option<Arc<serde_json::Value>>,
    /// Operator-supplied precision label (Tier 3), from `ModelPolicy::precision`.
    precision: Option<Arc<str>>,
    /// Operator-supplied HF source repo id (Tier 3), from
    /// `ModelPolicy::model_source`.
    model_source: Option<Arc<str>>,
    /// Operator-supplied HF revision/commit sha (Tier 3), from
    /// `ModelPolicy::model_revision`.
    model_revision: Option<Arc<str>>,
}

/// Operator-supplied model provenance (Tier 3,
/// the image-metadata plan) — precision, HF source repo, HF
/// revision. Unlike Tiers 1-2, these are never derivable from the model
/// directory, so they come from config (`ModelPolicy`) via the same
/// hint-registry pattern `max_prompt_len`/draft-model hints already use
/// (`EngineFactory::register_image_provenance_hint`).
#[derive(Debug, Clone, Default)]
pub(crate) struct ImageProvenanceHint {
    pub(crate) precision: Option<String>,
    pub(crate) model_source: Option<String>,
    pub(crate) model_revision: Option<String>,
}

impl ImageModelMetadata {
    /// Reads the Tier 1/2 fields from `model_dir` — one call at load time,
    /// before the `OpenVINO` pipeline construction. Tier 3 fields start empty;
    /// attach them with [`with_provenance`](Self::with_provenance).
    ///
    /// `model_id` and `manifest_root` feed `model_hash` memoization
    /// (`crate::cache_manifest`): a `None` `manifest_root` (no `ov_cache_dir`
    /// configured) disables caching and every call recomputes, matching the
    /// pre-memoization behavior exactly.
    #[must_use]
    pub fn read(
        model_dir: &std::path::Path,
        model_id: &str,
        manifest_root: Option<&std::path::Path>,
    ) -> Self {
        Self {
            model_family: read_pipeline_class(model_dir).map(Arc::from),
            model_hash: resolve_model_hash(model_dir, model_id, manifest_root).map(Arc::from),
            scheduler_config: read_scheduler_config(model_dir).map(Arc::new),
            precision: None,
            model_source: None,
            model_revision: None,
        }
    }

    /// Attaches the Tier 3 operator-supplied provenance fields, if a hint was
    /// registered for this model. A `None` hint (no config fields set) leaves
    /// all three `None` — omitted from the response, not fabricated.
    #[must_use]
    pub(crate) fn with_provenance(mut self, hint: Option<ImageProvenanceHint>) -> Self {
        if let Some(hint) = hint {
            self.precision = hint.precision.map(Arc::from);
            self.model_source = hint.model_source.map(Arc::from);
            self.model_revision = hint.model_revision.map(Arc::from);
        }
        self
    }
}

/// Reads the diffusers pipeline class from `<model_dir>/model_index.json`'s
/// `_class_name` field (e.g. `"FluxPipeline"`, `"StableDiffusionXLPipeline"`).
///
/// Mirrors `OvImageState::pipeline_class` in `ov_bridge.cpp` (same file, same
/// key) — done independently in Rust rather than added to the C ABI, since the
/// model directory is already available here before the `OpenVINO` load call.
/// Read-only, side-channel: does not affect what `GenAI` itself loads.
///
/// Returns `None` if the file is missing, unreadable, or has no `_class_name`
/// key — `generation_metadata.model_family` is simply omitted in that case,
/// never fabricated.
fn read_pipeline_class(model_dir: &std::path::Path) -> Option<String> {
    let raw = std::fs::read_to_string(model_dir.join("model_index.json")).ok()?;
    let parsed: serde_json::Value = serde_json::from_str(&raw).ok()?;
    parsed
        .get("_class_name")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}

/// Computes a `sha256:`-prefixed hex digest of the model's primary denoising
/// backbone weight file — `unet/openvino_model.bin` (SD/SDXL/LCM-family
/// pipelines) or `transformer/openvino_model.bin` (FLUX/SD3-family — `DiT`
/// backbones use this directory name instead) — the two backbone directory
/// names the diffusers `OpenVINO` export convention uses across every pipeline
/// family surveyed (the image-metadata plan §1a).
///
/// Matches the pyramu-panel spec's convention (the Pyramu image-metadata spec
/// §5.1: "Model hash = ... the model file's SHA-256") — a single representative
/// file, the one that changes when an operator swaps checkpoints, not the
/// VAE/text-encoder submodels often shared across them.
///
/// Streams the file through the hasher (`std::io::copy`) rather than reading it
/// into memory — backbone files run into the low gigabytes. Still a multi-second
/// synchronous read; acceptable because it runs once at load time on the same
/// call path as the `OpenVINO` JIT compile, which already takes far longer.
///
/// Returns `None` if neither backbone file exists or is readable — omitted
/// from the response, never fabricated.
/// Unmemoized hash — kept only as the tests' ground truth against
/// [`resolve_model_hash`], which callers use in production now.
#[cfg(test)]
fn compute_model_hash(model_dir: &std::path::Path) -> Option<String> {
    let backbone = find_backbone(model_dir)?;
    hash_file(&backbone)
}

/// Finds the model's denoising backbone weight file under either known
/// directory convention (`unet` for SD/SDXL/LCM-family, `transformer` for
/// FLUX/SD3-family), same search `compute_model_hash` has always done —
/// extracted so [`resolve_model_hash`] can stat the file before deciding
/// whether to hash it.
fn find_backbone(model_dir: &std::path::Path) -> Option<std::path::PathBuf> {
    ["unet", "transformer"]
        .into_iter()
        .map(|dir| model_dir.join(dir).join("openvino_model.bin"))
        .find(|path| path.is_file())
}

/// Streams `path` through SHA-256 and returns a `sha256:`-prefixed hex
/// digest. `None` on any read failure — never a fabricated or partial hash.
fn hash_file(path: &std::path::Path) -> Option<String> {
    use sha2::{Digest, Sha256};
    use std::fmt::Write as _;

    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).ok()?;

    let mut hex = String::with_capacity(Sha256::output_size() * 2);
    for byte in hasher.finalize() {
        let _ = write!(hex, "{byte:02x}");
    }
    Some(format!("sha256:{hex}"))
}

/// Memoized wrapper around [`compute_model_hash`] (the image-metadata plan
/// Tier 2, the project's internal engineering log's 2026-08-01 update): checks
/// `crate::cache_manifest` for a cached digest keyed by
/// `(backbone_relpath, size_bytes, mtime_unix)` before streaming the backbone
/// file through SHA-256 again. A `None` `manifest_root` (no `ov_cache_dir`
/// configured) always recomputes — same as before memoization existed.
///
/// Any manifest read/write failure (corrupt file, unwritable directory) is a
/// silent cache miss, never a load failure — hashing falls back to
/// `compute_model_hash`'s own file-not-found handling either way.
pub(crate) fn resolve_model_hash(
    model_dir: &std::path::Path,
    model_id: &str,
    manifest_root: Option<&std::path::Path>,
) -> Option<String> {
    let backbone = find_backbone(model_dir)?;
    let relpath = backbone
        .strip_prefix(model_dir)
        .ok()?
        .to_string_lossy()
        .into_owned();
    let meta = std::fs::metadata(&backbone).ok()?;
    let size_bytes = meta.len();
    let mtime_unix = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .and_then(|d| i64::try_from(d.as_secs()).ok());

    if let (Some(root), Some(mtime_unix)) = (manifest_root, mtime_unix)
        && let Some(cached) = crate::cache_manifest::read(root, model_id).and_then(|f| f.model_hash)
        && cached.backbone_relpath == relpath
        && cached.size_bytes == size_bytes
        && cached.mtime_unix == mtime_unix
    {
        return Some(cached.sha256);
    }

    let hash = hash_file(&backbone)?;

    if let (Some(root), Some(mtime_unix)) = (manifest_root, mtime_unix) {
        crate::cache_manifest::write_model_hash_entry(
            root,
            model_id,
            crate::cache_manifest::ModelHashEntry {
                backbone_relpath: relpath,
                size_bytes,
                mtime_unix,
                sha256: hash.clone(),
            },
        );
    }

    Some(hash)
}

/// Reads `<model_dir>/scheduler/scheduler_config.json` and returns it as a raw
/// JSON value — no per-scheduler-algorithm parsing. The path is a subdirectory
/// (not the model root, unlike `model_index.json`) and generalizes across every
/// pipeline family surveyed; deliberately does NOT consult `model_index.json`'s
/// own `scheduler` pointer, which can be stale relative to this file
/// (the image-metadata plan §1a Findings 1 and 2).
///
/// Returns `None` if the file is missing, unparseable, or not a JSON object
/// (a pathological `scheduler_config.json` whose whole content is e.g. `null`
/// or a bare array would otherwise parse successfully and later serialize as
/// `"scheduler_config": null` in the response instead of being omitted) —
/// `sampler` and `scheduler_config` are both omitted in every such case.
fn read_scheduler_config(model_dir: &std::path::Path) -> Option<serde_json::Value> {
    let raw =
        std::fs::read_to_string(model_dir.join("scheduler").join("scheduler_config.json")).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    value.is_object().then_some(value)
}

/// Run one generation on the engine thread: generate the raw pixels, then
/// PNG-encode each image. Both steps are CPU/GPU-heavy and stay on this thread.
fn run_generate(
    engine: &OvImageEngine,
    model_id: &str,
    prompt: &str,
    opts: &ImageGenOptions,
) -> anyhow::Result<Vec<Vec<u8>>> {
    let images = engine.generate(prompt, opts)?;
    let pngs = images
        .iter()
        .map(|img| crate::image_util::rgb_to_png(img.width, img.height, &img.data))
        .collect::<anyhow::Result<Vec<_>>>()?;
    tracing::debug!(model_id, images = pngs.len(), "image generation complete");
    Ok(pngs)
}

/// Run one edit on the engine thread: inpaint/img2img the raw pixels, then
/// PNG-encode each image. CPU/GPU-heavy work stays on this thread.
fn run_edit(
    engine: &OvImageEngine,
    model_id: &str,
    prompt: &str,
    init: &RgbImage,
    mask: Option<&RgbImage>,
    opts: &ImageEditOptions,
) -> anyhow::Result<Vec<Vec<u8>>> {
    let images = engine.edit(prompt, init, mask, opts)?;
    let pngs = images
        .iter()
        .map(|img| crate::image_util::rgb_to_png(img.width, img.height, &img.data))
        .collect::<anyhow::Result<Vec<_>>>()?;
    tracing::debug!(
        model_id,
        images = pngs.len(),
        inpaint = mask.is_some(),
        "image edit complete"
    );
    Ok(pngs)
}

// ── Mock engine (tests only) ──────────────────────────────────────────────────

/// Spawn a GPU-free mock image engine — the `ImageGen` analogue of
/// `spawn_mock_stt`, used by `MockEngineFactory` and routing tests.
///
/// Answers each request with `n` canned 1×1 PNGs (a solid black pixel) without
/// touching a GPU, so a handler test yields a valid response shape.
#[cfg(test)]
pub(crate) fn spawn_mock_image(model_id: &str) -> (ImageHandle, std::thread::JoinHandle<()>) {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<ImageCommand>(IMAGE_CHANNEL_CAP);

    let thread = std::thread::spawn(move || {
        // Both ops answer with `n` canned 1×1 black PNGs — no GPU.
        let canned = |n: u32| {
            (0..n.max(1))
                .map(|_| crate::image_util::rgb_to_png(1, 1, &[0, 0, 0]))
                .collect::<anyhow::Result<Vec<_>>>()
        };
        while let Some(cmd) = rx.blocking_recv() {
            match cmd {
                ImageCommand::Generate { opts, reply, .. } => {
                    let _ = reply.send(canned(opts.num_images));
                }
                ImageCommand::Edit { opts, reply, .. } => {
                    let _ = reply.send(canned(opts.num_images));
                }
            }
        }
    });

    (
        ImageHandle {
            tx,
            model_id: Arc::from(model_id),
            metadata: Arc::new(ImageModelMetadata::default()),
            in_flight: Arc::new(AtomicUsize::new(0)),
        },
        thread,
    )
}

// ============================================================
// Unit tests (no GPU)
// ============================================================
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn opts(n: u32) -> ImageGenOptions {
        ImageGenOptions {
            negative_prompt: None,
            width: 512,
            height: 512,
            num_inference_steps: 20,
            num_images: n,
            seed: None,
            guidance_scale: 0.0,
        }
    }

    /// The single-stream engine reports `waiting = in_flight − 1`, zero when idle.
    #[test]
    fn waiting_is_in_flight_minus_the_running_one() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<ImageCommand>(IMAGE_CHANNEL_CAP);
        let h = ImageHandle::from_sender(tx, "test-image");

        assert_eq!(h.active(), 0);
        assert_eq!(h.waiting(), 0, "idle engine has nothing queued");

        h.in_flight.store(2, Ordering::Relaxed);
        assert_eq!(h.active(), 2);
        assert_eq!(h.waiting(), 1, "2 in flight = 1 running + 1 queued");
    }

    /// The mock engine round-trips a request: `n` valid PNGs, and the in-flight
    /// counter settles back to zero.
    #[tokio::test]
    async fn mock_image_round_trips() {
        let (handle, _thread) = spawn_mock_image("mock-image");
        let pngs = handle
            .generate("a red circle".to_owned(), opts(3))
            .await
            .expect("mock generate");
        assert_eq!(pngs.len(), 3);
        for png in &pngs {
            assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n", "each blob is a PNG");
        }
        assert_eq!(handle.active(), 0, "counter settled back to zero");
    }

    fn edit_opts(n: u32) -> ImageEditOptions {
        ImageEditOptions {
            negative_prompt: None,
            num_inference_steps: 20,
            num_images: n,
            seed: None,
            guidance_scale: 0.0,
            strength: 0.0,
        }
    }

    /// The mock engine round-trips an edit (img2img: no mask) into `n` PNGs.
    #[tokio::test]
    async fn mock_image_edit_round_trips() {
        let (handle, _thread) = spawn_mock_image("mock-image");
        let init = RgbImage {
            width: 2,
            height: 2,
            data: vec![0u8; 2 * 2 * 3],
        };
        let pngs = handle
            .edit("make it blue".to_owned(), init, None, edit_opts(2))
            .await
            .expect("mock edit");
        assert_eq!(pngs.len(), 2);
        for png in &pngs {
            assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n", "each blob is a PNG");
        }
        assert_eq!(handle.active(), 0, "counter settled back to zero");
    }

    /// A dropped engine (closed channel) surfaces as an error, not a hang.
    #[tokio::test]
    async fn generate_errors_when_engine_gone() {
        let (tx, rx) = tokio::sync::mpsc::channel::<ImageCommand>(IMAGE_CHANNEL_CAP);
        let handle = ImageHandle::from_sender(tx, "dead-image");
        drop(rx);
        let err = handle.generate("x".to_owned(), opts(1)).await;
        assert!(err.is_err(), "generate on a dead engine must error");
        assert_eq!(handle.active(), 0, "counter rolled back after send failure");
    }

    /// `read_pipeline_class` extracts `_class_name` from `model_index.json`,
    /// generically across pipeline families (the image-metadata plan
    /// §1a Finding 1) — no per-family parsing branch.
    #[test]
    fn read_pipeline_class_extracts_class_name_across_families() {
        for (class_name, extra) in [
            (
                "StableDiffusionXLPipeline",
                r#""force_zeros_for_empty_prompt": true"#,
            ),
            (
                "LatentConsistencyModelPipeline",
                r#""requires_safety_checker": true"#,
            ),
            ("FluxPipeline", r#""_diffusers_version": "0.32.1""#),
        ] {
            let dir = tempfile::tempdir().expect("tempdir");
            std::fs::write(
                dir.path().join("model_index.json"),
                format!(r#"{{"_class_name": "{class_name}", {extra}}}"#),
            )
            .expect("write model_index.json");
            assert_eq!(read_pipeline_class(dir.path()).as_deref(), Some(class_name));
        }
    }

    /// A missing/unparseable `model_index.json` omits the field rather than
    /// fabricating one — no crash, no default guess.
    #[test]
    fn read_pipeline_class_returns_none_when_unreadable() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert_eq!(read_pipeline_class(dir.path()), None, "no file at all");

        std::fs::write(dir.path().join("model_index.json"), "not json").expect("write garbage");
        assert_eq!(read_pipeline_class(dir.path()), None, "unparseable JSON");
    }

    /// `compute_model_hash` finds the backbone under either directory name
    /// (`unet` for SD-family, `transformer` for FLUX/SD3-family) and produces a
    /// stable `sha256:`-prefixed digest of that file's exact bytes.
    #[test]
    fn compute_model_hash_finds_unet_or_transformer_backbone() {
        for backbone_dir in ["unet", "transformer"] {
            let dir = tempfile::tempdir().expect("tempdir");
            std::fs::create_dir(dir.path().join(backbone_dir)).expect("mkdir backbone");
            std::fs::write(
                dir.path().join(backbone_dir).join("openvino_model.bin"),
                b"pretend-weights",
            )
            .expect("write backbone");

            let hash = compute_model_hash(dir.path()).expect("hash computed");
            assert!(hash.starts_with("sha256:"), "hash: {hash}");
            assert_eq!(hash.len(), "sha256:".len() + 64, "full 64 hex chars");

            // Same bytes → same hash, deterministic across the two directory names.
            assert_eq!(compute_model_hash(dir.path()), Some(hash));
        }
    }

    /// No backbone file under either known directory name omits the hash
    /// entirely — never a fabricated or zero digest.
    #[test]
    fn compute_model_hash_returns_none_without_a_backbone_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert_eq!(compute_model_hash(dir.path()), None, "empty model dir");

        // A vae_decoder alone (no unet/transformer) still doesn't count — only
        // the two known backbone directory names are checked.
        std::fs::create_dir(dir.path().join("vae_decoder")).expect("mkdir vae_decoder");
        std::fs::write(
            dir.path().join("vae_decoder").join("openvino_model.bin"),
            b"not the backbone",
        )
        .expect("write vae file");
        assert_eq!(
            compute_model_hash(dir.path()),
            None,
            "non-backbone file present"
        );
    }

    /// `resolve_model_hash` with `manifest_root: None` always recomputes —
    /// same output as `compute_model_hash`, no caching attempted.
    #[test]
    fn resolve_model_hash_recomputes_when_manifest_root_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(dir.path().join("unet")).expect("mkdir unet");
        std::fs::write(
            dir.path().join("unet").join("openvino_model.bin"),
            b"weights",
        )
        .expect("write backbone");

        let hash = resolve_model_hash(dir.path(), "some-model", None).expect("hash computed");
        assert_eq!(hash, compute_model_hash(dir.path()).expect("direct hash"));
    }

    /// A second call with an unchanged backbone file hits the manifest cache
    /// instead of rehashing. `first == second` alone doesn't prove a hit — a
    /// silent rehash of unchanged bytes returns the same digest too. Proof:
    /// overwrite the backbone with *different* same-length content but
    /// restore the original mtime, then confirm the second call still
    /// returns the *first* (now-stale) digest — only possible if it read the
    /// cache instead of rehashing the new bytes.
    #[test]
    fn resolve_model_hash_hits_cache_on_unchanged_backbone() {
        let model_dir = tempfile::tempdir().expect("model tempdir");
        let manifest_dir = tempfile::tempdir().expect("manifest tempdir");
        std::fs::create_dir(model_dir.path().join("transformer")).expect("mkdir transformer");
        let backbone = model_dir
            .path()
            .join("transformer")
            .join("openvino_model.bin");
        std::fs::write(&backbone, b"backbone-bytes").expect("write backbone");
        let original_mtime = std::fs::metadata(&backbone)
            .expect("stat backbone")
            .modified()
            .expect("mtime");

        let first = resolve_model_hash(
            model_dir.path(),
            "flux-schnell-int4-ov",
            Some(manifest_dir.path()),
        )
        .expect("first hash");
        assert!(
            crate::cache_manifest::read(manifest_dir.path(), "flux-schnell-int4-ov").is_some(),
            "manifest entry written after first resolve"
        );

        // Same length (14 bytes), different content, mtime restored to the
        // original value — a real rehash would notice the changed bytes.
        std::fs::write(&backbone, b"replaced-bytes").expect("overwrite backbone");
        std::fs::OpenOptions::new()
            .write(true)
            .open(&backbone)
            .expect("reopen backbone")
            .set_modified(original_mtime)
            .expect("restore mtime");

        let second = resolve_model_hash(
            model_dir.path(),
            "flux-schnell-int4-ov",
            Some(manifest_dir.path()),
        )
        .expect("second hash");
        assert_eq!(
            second, first,
            "stale digest served from cache despite changed bytes — proves a real hit, not a lucky rehash"
        );
    }

    /// A backbone that grows/shrinks (same mtime granularity, different size
    /// — e.g. a re-export at a different precision) invalidates the cache
    /// even if mtime alone didn't change enough to be distinguishable.
    #[test]
    fn resolve_model_hash_misses_cache_when_size_changes() {
        let model_dir = tempfile::tempdir().expect("model tempdir");
        let manifest_dir = tempfile::tempdir().expect("manifest tempdir");
        std::fs::create_dir(model_dir.path().join("unet")).expect("mkdir unet");
        let backbone = model_dir.path().join("unet").join("openvino_model.bin");
        std::fs::write(&backbone, b"short").expect("write short backbone");

        let first = resolve_model_hash(model_dir.path(), "some-model", Some(manifest_dir.path()))
            .expect("short hash");

        std::fs::write(&backbone, b"a much longer backbone payload").expect("write long backbone");

        let second = resolve_model_hash(model_dir.path(), "some-model", Some(manifest_dir.path()))
            .expect("long hash");
        assert_ne!(
            first, second,
            "size change must invalidate the cached digest"
        );
    }

    /// A model family swap that changes which backbone directory wins
    /// (`unet` → `transformer`, e.g. re-exported as a different architecture)
    /// invalidates the cache — the manifest key includes the relative path,
    /// not just size/mtime, so a same-size coincidence under a different
    /// directory can't produce a false hit.
    #[test]
    fn resolve_model_hash_misses_cache_when_backbone_relpath_changes() {
        let model_dir = tempfile::tempdir().expect("model tempdir");
        let manifest_dir = tempfile::tempdir().expect("manifest tempdir");
        std::fs::create_dir(model_dir.path().join("unet")).expect("mkdir unet");
        std::fs::write(
            model_dir.path().join("unet").join("openvino_model.bin"),
            b"unet-weights",
        )
        .expect("write unet backbone");

        resolve_model_hash(model_dir.path(), "some-model", Some(manifest_dir.path()))
            .expect("unet hash");

        // Same bytes/size as before, deliberately — if the cache-hit check
        // only compared size+mtime and ignored relpath, this rewrite could
        // produce a false hit. Using identical content makes that failure
        // mode indistinguishable by return value alone, so the real proof
        // below is the manifest's recorded relpath, not the returned digest.
        std::fs::remove_dir_all(model_dir.path().join("unet")).expect("remove unet dir");
        std::fs::create_dir(model_dir.path().join("transformer")).expect("mkdir transformer");
        std::fs::write(
            model_dir
                .path()
                .join("transformer")
                .join("openvino_model.bin"),
            b"unet-weights",
        )
        .expect("write transformer backbone with the same bytes/size as before");

        resolve_model_hash(model_dir.path(), "some-model", Some(manifest_dir.path()))
            .expect("transformer hash");

        let manifest = crate::cache_manifest::read(manifest_dir.path(), "some-model")
            .expect("manifest present")
            .model_hash
            .expect("model_hash entry present");
        assert_eq!(
            manifest.backbone_relpath,
            std::path::Path::new("transformer")
                .join("openvino_model.bin")
                .to_string_lossy(),
            "manifest must record the new relpath, not the stale unet one"
        );
    }

    /// Changing the backbone's mtime (a checkpoint swap that rewrites the
    /// same bytes) invalidates the cache entry rather than trusting stale
    /// content — the manifest key is `(relpath, size, mtime)`, not content.
    #[test]
    fn resolve_model_hash_misses_cache_when_mtime_changes() {
        let model_dir = tempfile::tempdir().expect("model tempdir");
        let manifest_dir = tempfile::tempdir().expect("manifest tempdir");
        std::fs::create_dir(model_dir.path().join("unet")).expect("mkdir unet");
        let backbone = model_dir.path().join("unet").join("openvino_model.bin");
        std::fs::write(&backbone, b"v1").expect("write v1");

        let first = resolve_model_hash(model_dir.path(), "some-model", Some(manifest_dir.path()))
            .expect("v1 hash");

        // Simulate a checkpoint swap: new content, new mtime, same size.
        std::fs::write(&backbone, b"v2").expect("write v2");
        let future = std::time::SystemTime::now() + std::time::Duration::from_mins(2);
        std::fs::OpenOptions::new()
            .write(true)
            .open(&backbone)
            .expect("reopen backbone")
            .set_modified(future)
            .expect("bump mtime");

        let second = resolve_model_hash(model_dir.path(), "some-model", Some(manifest_dir.path()))
            .expect("v2 hash");
        assert_ne!(
            first, second,
            "mtime bump must invalidate the cached digest"
        );
    }

    /// A corrupt manifest file degrades to recompute, never a load failure.
    #[test]
    fn resolve_model_hash_recomputes_when_manifest_corrupt() {
        let model_dir = tempfile::tempdir().expect("model tempdir");
        let manifest_dir = tempfile::tempdir().expect("manifest tempdir");
        std::fs::create_dir(model_dir.path().join("unet")).expect("mkdir unet");
        std::fs::write(
            model_dir.path().join("unet").join("openvino_model.bin"),
            b"weights",
        )
        .expect("write backbone");
        std::fs::write(manifest_dir.path().join("some-model.json"), b"not json")
            .expect("write corrupt manifest");

        let hash = resolve_model_hash(model_dir.path(), "some-model", Some(manifest_dir.path()))
            .expect("hash computed despite corrupt manifest");
        assert_eq!(
            hash,
            compute_model_hash(model_dir.path()).expect("direct hash")
        );
    }

    /// `read_scheduler_config` reads the `scheduler/` subdirectory (not the
    /// model root) and passes the JSON through verbatim.
    #[test]
    fn read_scheduler_config_reads_the_subdirectory_verbatim() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(dir.path().join("scheduler")).expect("mkdir scheduler");
        std::fs::write(
            dir.path().join("scheduler").join("scheduler_config.json"),
            r#"{"_class_name": "LCMScheduler", "beta_schedule": "scaled_linear"}"#,
        )
        .expect("write scheduler_config.json");

        let config = read_scheduler_config(dir.path()).expect("config parsed");
        assert_eq!(config["_class_name"], "LCMScheduler");
        assert_eq!(config["beta_schedule"], "scaled_linear");

        // A copy at the model root (wrong location) must NOT be picked up.
        assert_eq!(
            read_scheduler_config(&dir.path().join("scheduler")),
            None,
            "must not treat the scheduler dir itself as the model dir"
        );
    }

    /// A missing/unparseable `scheduler_config.json` omits the field, matching
    /// `read_pipeline_class`'s contract.
    #[test]
    fn read_scheduler_config_returns_none_when_unreadable() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert_eq!(read_scheduler_config(dir.path()), None, "no scheduler dir");
    }

    /// `with_provenance` attaches all three Tier 3 fields from a `Some` hint,
    /// readable back through the same accessors `ImageHandle` exposes.
    #[test]
    fn with_provenance_attaches_all_three_fields() {
        let dir = tempfile::tempdir().expect("tempdir");
        let metadata = ImageModelMetadata::read(dir.path(), "test-model", None).with_provenance(
            Some(ImageProvenanceHint {
                precision: Some("int8".to_owned()),
                model_source: Some("OpenVINO/LCM_Dreamshaper_v7-int8-ov".to_owned()),
                model_revision: Some("a1b2c3d".to_owned()),
            }),
        );
        assert_eq!(metadata.precision.as_deref(), Some("int8"));
        assert_eq!(
            metadata.model_source.as_deref(),
            Some("OpenVINO/LCM_Dreamshaper_v7-int8-ov")
        );
        assert_eq!(metadata.model_revision.as_deref(), Some("a1b2c3d"));
    }

    /// A `None` hint (no config fields set for this model) leaves every Tier 3
    /// field `None` — omitted from the response, never fabricated.
    #[test]
    fn with_provenance_none_hint_leaves_fields_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let metadata =
            ImageModelMetadata::read(dir.path(), "test-model", None).with_provenance(None);
        assert_eq!(metadata.precision, None);
        assert_eq!(metadata.model_source, None);
        assert_eq!(metadata.model_revision, None);
    }

    /// `ImageHandle::sampler()` — the accessor the response actually calls,
    /// not just the underlying file reader — regression test for §1a
    /// Finding 2: `model_index.json`'s own `scheduler` pointer can be stale
    /// (empirically, on `Juggernaut-XL-v9-fp16-ov`: it named `DDPMScheduler`
    /// while the real `scheduler/scheduler_config.json` was
    /// `EulerDiscreteScheduler`). `sampler()` must reflect the latter — the
    /// file `Scheduler::from_config` actually loads — never the former.
    #[test]
    fn sampler_accessor_reflects_scheduler_config_not_model_index_pointer() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("model_index.json"),
            r#"{"_class_name": "StableDiffusionXLPipeline", "scheduler": ["diffusers", "DDPMScheduler"]}"#,
        )
        .expect("write model_index.json");
        std::fs::create_dir(dir.path().join("scheduler")).expect("mkdir scheduler");
        std::fs::write(
            dir.path().join("scheduler").join("scheduler_config.json"),
            r#"{"_class_name": "EulerDiscreteScheduler", "beta_schedule": "scaled_linear"}"#,
        )
        .expect("write scheduler_config.json");

        let metadata = ImageModelMetadata::read(dir.path(), "test-model", None);
        let (tx, _rx) = tokio::sync::mpsc::channel::<ImageCommand>(IMAGE_CHANNEL_CAP);
        let handle = ImageHandle {
            tx,
            model_id: Arc::from("test-model"),
            metadata: Arc::new(metadata),
            in_flight: Arc::new(AtomicUsize::new(0)),
        };

        assert_eq!(handle.model_family(), Some("StableDiffusionXLPipeline"));
        assert_eq!(
            handle.sampler(),
            Some("EulerDiscreteScheduler"),
            "must come from scheduler/scheduler_config.json, not model_index.json's stale pointer"
        );
    }
}
