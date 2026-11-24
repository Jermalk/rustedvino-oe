// ============================================================
// src/handlers/admin.rs — health, model list, model admin
// ============================================================
// Phase 0: GET /health (liveness) + GET /v1/models (stub).
// Phase 2: /v1/models reads from ModelManager; admin routes:
//   POST   /v1/admin/models/{model_id}/load   → load a model
//   DELETE /v1/admin/models/{model_id}         → evict a model
// Phase 3.3: GET /metrics (Prometheus, same port).
// Phase 3.8: GET /health_generate (deep GPU readiness probe).
// ============================================================

use std::sync::{OnceLock, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::{
    Extension, Json,
    body::Bytes,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use crate::app_state::AppState;
use crate::model_manager::{EngineHandleKind, EvictionReport, LoadOverrides, ModelError};
use crate::voice_pin::VoiceFlowPin;

/// Approximate server-start Unix timestamp (seconds), captured once on first
/// use. The `OpenAI` `/v1/models` schema requires a `created` field, but we do
/// not record a per-model load time — every model reports this same
/// server-start value. Good enough for client compatibility (strict parsers
/// reject a model object with no `created`); not a meaningful per-model time.
fn server_created_ts() -> i64 {
    static TS: OnceLock<i64> = OnceLock::new();
    *TS.get_or_init(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(0))
    })
}

/// Minimal message shape for [`health_generate`]'s probe inference.
/// Serialised to apply the model's own chat template.
#[derive(serde::Serialize)]
struct ProbeMessage {
    role: &'static str,
    content: &'static str,
}

/// Edge length of the synthetic image used by the Vision deep-health probe.
/// 64 px clears every supported VLM's minimum-pixel / patch-size floor (a
/// 16 px image is below Qwen-VL's patch grid), while staying trivially cheap to
/// encode. Matches the 64×64 image the R1 live verification proved end-to-end.
const PROBE_IMAGE_EDGE: u32 = 64;

/// Build the tiny solid-colour image the Vision `/health_generate` probe feeds
/// through the VLM so the deep-health check actually exercises the vision
/// encoder (not just the text decoder). Flat NHWC RGB, `edge × edge × 3` bytes —
/// the layout [`crate::image_util::DecodedImage`] documents.
fn probe_image() -> crate::image_util::DecodedImage {
    let edge = PROBE_IMAGE_EDGE;
    // Solid blue (R=0, G=0, B=255), one RGB triple per pixel.
    let data = std::iter::repeat_n([0u8, 0, 255], (edge * edge) as usize)
        .flatten()
        .collect();
    crate::image_util::DecodedImage {
        data,
        height: edge,
        width: edge,
    }
}

// ---- /health -------------------------------------------------------

/// Response body for GET /health.
#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
    /// Short (12-char) commit hash the running binary was built from, or
    /// `"unknown"` for a build without a `.git` dir (e.g. a source tarball).
    /// Set at compile time by `build.rs` — see `emit_git_hash`.
    pub git_hash: &'static str,
    /// `true` once the GPU context is poisoned (E-12): the process is alive but every
    /// GPU model is unusable until a restart. `/health` keeps its 200 (it is a liveness
    /// probe, and a supervisor must not kill a worker that is about to exit by itself);
    /// this field is how a monitor or `rv status` learns the server cannot serve.
    pub gpu_poisoned: bool,
    /// `true` while a model load has not returned past its deadline (step 6b/6e): new loads on
    /// that device are refused (`503 device_stalled`) until it does; loaded models keep serving.
    /// Details (which device, model, how long): `GET /v1/admin/health/watchdog` or `rv status`.
    pub device_stalled: bool,
    /// m1: how many startup warnings this process has and their codes (distinct),
    /// never their text — `/health` is unauthenticated. The full messages: `GET
    /// /v1/admin/models` (`server.startup_warnings`) or `rv status`.
    pub startup_warnings: HealthStartupWarnings,
}

/// The `/health` summary of the startup warnings: `{"count": 0, "codes": []}` when none.
#[derive(Serialize)]
pub struct HealthStartupWarnings {
    /// Number of startup warnings.
    pub count: usize,
    /// Their distinct codes (e.g. `preload_skipped`), in first-seen order.
    pub codes: Vec<&'static str>,
}

/// GET /health — liveness probe.
///
/// Returns `{"status":"ok","version":"<semver>","git_hash":"<12-char hex or \"unknown\">","gpu_poisoned":false,"device_stalled":false,"startup_warnings":{"count":0,"codes":[]}}`.
/// Never returns an error — if the server is alive enough to route here, it's healthy;
/// `gpu_poisoned: true` is the one signal that it is alive but cannot serve GPU models.
pub async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        git_hash: env!("RV_GIT_HASH"),
        gpu_poisoned: state
            .model_manager
            .as_ref()
            .is_some_and(|mm| mm.is_gpu_poisoned()),
        device_stalled: state
            .model_manager
            .as_ref()
            .is_some_and(|mm| !mm.stalled_loads().is_empty()),
        startup_warnings: {
            let (count, codes) = crate::startup::startup_warning_codes();
            HealthStartupWarnings { count, codes }
        },
    })
}

// ---- /health_generate ----------------------------------------------

/// Success body for `GET /health_generate`.
#[derive(Serialize)]
pub struct HealthGenerateResponse {
    /// Always `"ok"` on a 200 response.
    pub status: &'static str,
    /// The model that served the probe inference.
    pub model: String,
    /// Wall time from handler entry to inference completion, milliseconds.
    pub elapsed_ms: u64,
}

/// Error body for `GET /health_generate` (503 responses).
#[derive(Serialize)]
struct HealthGenerateError {
    status: &'static str,
    /// Machine-readable reason, present only for `"gpu_poisoned"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'static str>,
    message: String,
}

/// Build a 503 error response for [`health_generate`].
fn hg_err(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(HealthGenerateError {
            status: "error",
            error: None,
            message: message.into(),
        }),
    )
        .into_response()
}

/// Submit the deep-health probe through the model's engine kind.
///
/// `TextGen` builds a prompt from the model's chat template and submits to the
/// CB engine; `Vision` passes a raw message plus a tiny synthetic image straight
/// to the VLM (which templates internally). Returns `Ok(())` once the request is
/// accepted, or `Err(Response)` carrying the 503 the caller should return on any
/// submission failure (bad template, at-capacity, dead engine).
#[allow(clippy::too_many_lines)]
// Response is the idiomatic "early-return a ready-built HTTP error" type used
// throughout this handler module; called once per health probe, not a hot loop.
#[allow(clippy::result_large_err)]
async fn submit_health_probe(
    ctx: &crate::model_manager::ChatContext,
    mm: &crate::model_manager::ModelManager,
    gen_params: crate::ov_cb::GenParams,
    tx: crate::streaming::TokenSender,
    start: Instant,
) -> Result<(), Response> {
    match &ctx.handle {
        EngineHandleKind::TextGen(handle) => {
            let msgs = [ProbeMessage {
                role: "user",
                content: "hi",
            }];
            let prompt = crate::prompt_builder::build_prompt(
                &msgs,
                None,
                &ctx.template,
                &ctx.eos_token,
                &ctx.bos_token,
                None,
                None,
            )
            .map_err(|e| {
                hg_err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("prompt build: {e}"),
                )
            })?;
            match handle
                .add_request(mm.next_id(), prompt, None, gen_params, tx, start)
                .await
            {
                crate::cb_engine::SubmitResult::Submitted => Ok(()),
                crate::cb_engine::SubmitResult::AtCapacity => Err(hg_err(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "engine at capacity",
                )),
                crate::cb_engine::SubmitResult::Closing => Err(hg_err(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "model is being evicted",
                )),
                crate::cb_engine::SubmitResult::EngineDead => {
                    Err(hg_err(StatusCode::SERVICE_UNAVAILABLE, "engine dead"))
                }
            }
        }
        EngineHandleKind::Vision(vlm) => {
            // The VLM applies its own chat template internally, so pass the raw
            // message rather than a pre-built prompt string. R2: feed a tiny
            // synthetic image so the probe exercises the *vision* encoder path,
            // not just the text decoder — a VLM that JIT-fails only on image
            // tensors would otherwise pass a text-only probe.
            let messages = vec![serde_json::json!({
                "role": "user",
                "content": "What colour is this image?"
            })];
            match vlm
                .generate(
                    messages,
                    vec![probe_image()],
                    None,
                    None,
                    gen_params,
                    tx,
                    start,
                )
                .await
            {
                crate::cb_engine::SubmitResult::Submitted => Ok(()),
                crate::cb_engine::SubmitResult::AtCapacity => Err(hg_err(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "vlm engine at capacity",
                )),
                crate::cb_engine::SubmitResult::Closing => Err(hg_err(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "model is being evicted",
                )),
                crate::cb_engine::SubmitResult::EngineDead => {
                    Err(hg_err(StatusCode::SERVICE_UNAVAILABLE, "vlm engine dead"))
                }
            }
        }
        EngineHandleKind::NpuTextGen(npu) => {
            let msgs = [ProbeMessage {
                role: "user",
                content: "hi",
            }];
            let prompt = crate::prompt_builder::build_prompt(
                &msgs,
                None,
                &ctx.template,
                &ctx.eos_token,
                &ctx.bos_token,
                None,
                None,
            )
            .map_err(|e| {
                hg_err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("prompt build: {e}"),
                )
            })?;
            match npu.generate(prompt, gen_params, tx, start).await {
                crate::cb_engine::SubmitResult::Submitted => Ok(()),
                crate::cb_engine::SubmitResult::AtCapacity => Err(hg_err(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "npu engine at capacity",
                )),
                crate::cb_engine::SubmitResult::Closing => Err(hg_err(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "model is being evicted",
                )),
                crate::cb_engine::SubmitResult::EngineDead => {
                    Err(hg_err(StatusCode::SERVICE_UNAVAILABLE, "npu engine dead"))
                }
            }
        }
        // R3 seams: no pipeline to probe yet. Report unavailable rather than
        // pretend the deep-health check ran for a kind we cannot generate from.
        EngineHandleKind::Embedding(_)
        | EngineHandleKind::Stt(_)
        | EngineHandleKind::Tts(_)
        | EngineHandleKind::ImageGen(_)
        | EngineHandleKind::Reranking(_)
        | EngineHandleKind::Decision(_) => Err(hg_err(
            StatusCode::SERVICE_UNAVAILABLE,
            format!(
                "deep-health probe not implemented for {} models",
                ctx.handle.kind().label()
            ),
        )),
    }
}

/// Pick which Ready model [`health_generate`] should probe.
///
/// Prefers a `TextGen`/`Vision` Ready model — the only kinds
/// [`submit_health_probe`] actually knows how to drive (an NPU-placed
/// `TextGen` model still reports `ModelKind::TextGen` here; device placement
/// doesn't change `configured_kind`) — falling back to any Ready model of
/// another kind only if none of those are available, so the 503 in that case
/// still honestly reports "not implemented for X models" rather than the
/// more misleading "no ready models". `None` iff no model is Ready at all.
fn select_probe_target(models: &[crate::model_manager::ModelInfo]) -> Option<String> {
    use crate::model_manager::{ModelKind, ModelState};
    models
        .iter()
        .find(|m| {
            m.state == ModelState::Ready && matches!(m.kind, ModelKind::TextGen | ModelKind::Vision)
        })
        .or_else(|| models.iter().find(|m| m.state == ModelState::Ready))
        .map(|m| m.id.clone())
}

/// GET `/health_generate` — deep readiness probe.
///
/// Fires a 1-token inference through a Ready model (picked by
/// [`select_probe_target`]) to verify that the GPU pipeline is actually
/// working end-to-end. Returns 200 on success or 503 when no model is ready,
/// the engine is dead, or inference times out. Complements `GET /health`,
/// which is a pure liveness probe (no GPU).
///
/// A `health_generate_during_eviction_pressure` stress-test scenario caught the
/// bug [`select_probe_target`] fixes: the old code picked
/// `list_ready_models().into_iter().next()`, the first entry in unspecified
/// `HashMap` iteration order, and reported a blanket 503 "not implemented
/// for embedding models" whenever that happened to land on an Embedding/
/// STT/TTS/ImageGen/Reranking model — even while a perfectly healthy
/// TextGen/Vision model sat Ready right next to it. Every box in the fleet
/// preloads at least one embedding model alongside a chat/vision model, so
/// this endpoint was effectively coin-flip flaky in production: a load
/// balancer or k8s readiness probe polling it could see spurious
/// "unhealthy" and fail the box over for no real reason.
pub async fn health_generate(State(state): State<AppState>) -> Response {
    let start = Instant::now();

    let Some(mm) = state.model_manager.as_ref() else {
        return hg_err(StatusCode::SERVICE_UNAVAILABLE, "no model manager");
    };

    // E-12: a poisoned GPU context can hang or lie to a probe inference; say so
    // up front instead of probing.
    if mm.is_gpu_poisoned() {
        // G1 (step 7): a supervised worker on its way to a restart says so (and Retry-After), so a
        // probe does not restart it a second time.
        let (message, retry) = super::error::gpu_poisoned_reply_now();
        let mut resp = (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(HealthGenerateError {
                status: "error",
                error: Some("gpu_poisoned"),
                message: message.to_owned(),
            }),
        )
            .into_response();
        if let Some(v) = retry {
            resp.headers_mut()
                .insert(axum::http::header::RETRY_AFTER, v);
        }
        return resp;
    }

    let Some(model_id) = select_probe_target(&mm.list_models()) else {
        return hg_err(StatusCode::SERVICE_UNAVAILABLE, "no ready models");
    };

    let ctx = match mm.get_chat_context(&model_id) {
        Ok(c) => c,
        Err(e) => {
            return hg_err(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("model unavailable: {e}"),
            );
        }
    };

    // R1: probe per kind. A 1-token greedy generation exercises the real
    // pipeline. A VLM serves text-only requests too, so its probe sends a plain
    // text message with no images.
    let gen_params = crate::ov_cb::GenParams {
        max_new_tokens: 1,
        temperature: Some(0.0),
        ..Default::default()
    };
    let (tx, rx) = crate::streaming::stream_channel();

    // Submit the 1-token probe through whichever engine kind backs this model.
    // On any submission failure the helper returns the 503 response to send.
    if let Err(resp) = submit_health_probe(&ctx, mm, gen_params, tx, start).await {
        return resp;
    }

    let drain = async move {
        let mut rx = rx;
        while let Some(event) = rx.recv().await {
            if let crate::streaming::StreamEvent::Error(e) = event {
                return Err(e);
            }
        }
        Ok::<(), String>(())
    };

    match tokio::time::timeout(Duration::from_secs(10), drain).await {
        Ok(Ok(())) => {
            let elapsed_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
            (
                StatusCode::OK,
                Json(HealthGenerateResponse {
                    status: "ok",
                    model: model_id,
                    elapsed_ms,
                }),
            )
                .into_response()
        }
        Ok(Err(e)) => hg_err(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("inference error: {e}"),
        ),
        Err(_) => hg_err(
            StatusCode::SERVICE_UNAVAILABLE,
            "inference timed out after 10s",
        ),
    }
}

// ---- /v1/admin/models (registry listing) ---------------------------

/// E-09: the sentence that tells an operator the pool they got is not the pool they asked for,
/// empty when it is. Read from the model's record after a load or resize.
fn kv_pool_note(mm: &crate::model_manager::ModelManager, model_id: &str) -> String {
    mm.list_models()
        .into_iter()
        .find(|m| m.id == model_id)
        .and_then(|m| {
            let (asked, by) = (m.kv_requested_gb?, m.kv_clamped_by?);
            Some(format!(
                "KV pool {} GiB (requested {} \u{2014} {})",
                fmt_gib(m.kv_cache_gb),
                fmt_gib(asked),
                kv_clamp_phrase(by)
            ))
        })
        .unwrap_or_default()
}

/// A GiB figure for the pool sentences: two decimals, except a value below 0.01 that would
/// print as `0.00` (a request of 0.001 must not read as "requested 0.00").
fn fmt_gib(v: f64) -> String {
    if v > 0.0 && v < 0.01 {
        format!("{v}")
    } else {
        format!("{v:.2}")
    }
}

/// How a [`KvClamp`](crate::model_manager::KvClamp) reads in the sentences `rv` prints.
fn kv_clamp_phrase(by: crate::model_manager::KvClamp) -> String {
    match by {
        crate::model_manager::KvClamp::WholeGibFallback => "rounded up to whole GiB".to_owned(),
        crate::model_manager::KvClamp::DeviceHeadroom => {
            "a larger pool would leave too little free device memory for long prompts".to_owned()
        }
        other => format!("clamped by {}", other.label()),
    }
}

/// State reported for a GPU-placed model while the GPU context is poisoned (E-12).
const STATE_GPU_POISONED: &str = "unavailable (gpu_poisoned)";

/// One entry in the `GET /v1/admin/models` registry response.
#[derive(Serialize)]
struct AdminModelEntry {
    id: String,
    /// `"not_loaded"` | `"loading"` | `"ready"` | `"evicting"`, or
    /// `"unavailable (gpu_poisoned)"` for a GPU-placed model while the GPU context is poisoned, or
    /// `"stalled"` for a load past its deadline (step 6b) or a CPU/NPU model whose engine stalled
    /// (0.9.1 S1: refused until a restart; also while an unload waits for its stuck thread).
    state: &'static str,
    /// From `model_kinds` config; `"text_gen"` when not specified.
    kind: &'static str,
    /// VRAM estimate from config (GB). `0.0` = unknown / CPU-only model.
    vram_gb: f64,
    /// `OpenVINO` device this model is placed on — e.g. `"GPU"`, `"CPU"`,
    /// `"NPU"`. Resolved at registration, so it's accurate even before the
    /// model's first load.
    device: String,
    /// This model's own resolved sampling default (`generation_config.json`,
    /// post sanity-check), applied to a request that sets none of
    /// `temperature`/`top_p`/`top_k` itself, and (`top_p`/`top_k` only) to one that sets
    /// only a positive `temperature` — see
    /// `crate::handlers::chat::resolve_sampling_defaults`. Key omitted
    /// entirely (not `null`) before the model's first successful load, for
    /// media/embedding models, or when the file didn't pass the loader's gate
    /// — a TextGen/Vision model that hasn't loaded yet, and every STT/TTS/
    /// embedding/reranking/image-gen model, simply never carries these three
    /// keys at all.
    #[serde(skip_serializing_if = "Option::is_none")]
    default_temperature: Option<f32>,
    /// See `default_temperature`.
    #[serde(skip_serializing_if = "Option::is_none")]
    default_top_p: Option<f32>,
    /// See `default_temperature`.
    #[serde(skip_serializing_if = "Option::is_none")]
    default_top_k: Option<usize>,
    /// The model's effective sampling profile (0.9.1 V5c), with a source per value:
    /// `{mode, temperature: {value, source}, top_p, top_k, repetition_penalty,
    /// presence_penalty, frequency_penalty, tools: "greedy"}` (+ `ignored_on_device` on
    /// NPU, `note` for an exporter-stub `generation_config.json`). Present for every
    /// text-generating model in every state; omitted for other kinds.
    #[serde(skip_serializing_if = "Option::is_none")]
    sampling: Option<serde_json::Value>,
    /// The configured default thinking budget in tokens (`thinking_budget_tokens` in
    /// config.json, 0.9.1); `null` when unset. Always present, every kind.
    thinking_budget_tokens: Option<u32>,
    /// The L0 prompt-length gate ceiling in tokens — the number a `400
    /// context_length_exceeded` on this model would cite. `0` before the
    /// model's first successful load, once evicted, for media/embedding
    /// models, or when the gate is disabled. Lets an external client/operator
    /// discover the real usable context window proactively instead of only
    /// reactively via that error.
    max_prompt_tokens: usize,
    /// KV cache pool actually reserved for this model at load time (GB).
    /// `0.0` before first load or once evicted.
    cache_size_gb: f64,
    /// The explicit KV target the last load was given (GB); omitted when the load
    /// asked for "all that fits" or the model is not loaded.
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_requested_gb: Option<f64>,
    /// Why the pool differs from `kv_requested_gb` (E-09): `free_vram`,
    /// `cache_size_gb`, `min_kv_cache_gb` or `whole_gib_fallback`. Omitted when it does not.
    #[serde(skip_serializing_if = "Option::is_none")]
    clamped_by: Option<&'static str>,
    /// The model's own native (trained/converted) context ceiling —
    /// `max_position_embeddings`, `rope_scaling`-adjusted. A hardware-
    /// independent model property: contrast with `max_prompt_tokens`, which
    /// is what this box's KV pool can currently *serve*. The two can differ
    /// wildly (confirmed live 2026-08-23,
    /// the project's internal engineering log: a VRAM-derived ceiling
    /// of 142k-306k tokens against a real trained ceiling of 40,960) —
    /// when `max_prompt_tokens < native_context_limit`, VRAM is the binding
    /// constraint; when the two are equal, the model itself is. Omitted
    /// (not `null`) when `config.json` doesn't declare
    /// `max_position_embeddings` — an unknown ceiling, not a guessed one.
    #[serde(skip_serializing_if = "Option::is_none")]
    native_context_limit: Option<usize>,
    /// Live KV-cache occupancy percentage (0-100). `0.0` before first load,
    /// once evicted, or for an engine kind whose live occupancy can't be
    /// read (see `kv_pressure_flagged`'s doc comment — same caveat applies).
    kv_cache_usage_pct: f64,
    /// Whether `kv_cache_usage_pct` is a live reading for this model (`rustedvino_kv_cache_usage_
    /// supported`). `false` while not loaded and for an engine kind whose occupancy cannot be read
    /// (the VLM engine): its `0.0` is then not "empty" (C21/P1).
    kv_cache_usage_supported: bool,
    /// Whether this model's KV occupancy has been continuously at/above the
    /// configured pressure threshold for at least the configured sustained
    /// duration (the project's internal engineering log). Always `false`
    /// while the monitor is disabled (the fleet-wide default) — surfaced
    /// here, not only in `/metrics`, so an operator not watching Prometheus
    /// can still see it via this endpoint.
    kv_pressure_flagged: bool,
    /// Measured device memory for a loaded model, in GB: the change across its
    /// load plus runtime growth (the embedding batch cache). Compare with
    /// `vram_gb` — a larger figure means the memory accounting under-counts
    /// this model. Omitted when not loaded or on an OS without DRM fdinfo.
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_memory_estimate_gb: Option<f64>,
    /// The KV precision the engine was built with: `u4`, `i8`, `f16`, or `auto` when the plugin
    /// decided and the server cannot say. Omitted until the first load and for models without a
    /// KV engine.
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_cache_precision: Option<&'static str>,
    /// Who decided it: `per_model`, `global`, `ir_hint`, `auto_rule` or `passthrough`.
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_precision_source: Option<&'static str>,
    /// KV bytes one token costs at that precision (from the model's layout).
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_bytes_per_token: Option<u64>,
    /// How far the KV profile can be trusted: `high`, `medium` or `low` (low = legacy limit).
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_profile_confidence: Option<&'static str>,
    /// How `max_prompt_tokens` was computed: `exact` or `legacy`; omitted when there is no gate.
    #[serde(skip_serializing_if = "str::is_empty")]
    kv_gate_mode: &'static str,
    /// What limits an exact `max_prompt_tokens`: `pool`, `native`, `ratchet` or
    /// `device_headroom`. Omitted under the legacy estimate.
    #[serde(skip_serializing_if = "Option::is_none")]
    limit_clamped_by: Option<&'static str>,
    /// The KV pool in MiB as booked (the engine allocates exactly this for an exactly-sized
    /// pool); `0` when not loaded.
    kv_pool_mib: u64,
    /// KV blocks of an exactly-sized pool.
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_pool_blocks: Option<u64>,
    /// The device-headroom limit right now in tokens (vision model on a discrete GPU).
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_headroom_cap_tokens: Option<usize>,
    /// Weights as measured at the last load, MiB (device-memory delta minus the KV pool); absent
    /// when that load's measurement window was not clean, and for a whole-GiB pool (F3-N-3).
    #[serde(skip_serializing_if = "Option::is_none")]
    vram_measured_mib: Option<u64>,
    /// C25: the part of the booked KV pool that became resident during the last load, MiB; absent
    /// when not measured that way (see `ModelInfo::kv_pool_resident_mib`).
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_pool_resident_mib: Option<u64>,
    /// F2-S-5 (E-16): `true` once a vision request with images was admitted since the last load
    /// without its image tokens counted — the context limit then saw only the text.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    kv_admission_low_confidence: bool,
    /// How many such requests (absent when none).
    #[serde(skip_serializing_if = "is_zero_u64")]
    uncounted_image_admissions: u64,
}

/// Serde helper: omit a zero counter.
#[allow(clippy::trivially_copy_pass_by_ref)] // serde passes `&T`
fn is_zero_u64(n: &u64) -> bool {
    *n == 0
}

/// Response envelope for `GET /v1/admin/models`.
#[derive(Serialize)]
struct AdminModelsResponse {
    models: Vec<AdminModelEntry>,
    server: ServerInfo,
}

/// Fleet-debugging technical info: which box this is, which `OpenVINO`
/// runtime it's linked against, which `RustedVINO` build. One consistent JSON
/// object rather than scattering these across response headers — headers are
/// awkward to query/compare across a fleet and easy to overlook; a JSON field
/// on an already-authenticated admin endpoint is not. Lives here (not on every
/// pipeline's per-request response) because these facts are static for the
/// process lifetime, not per-request — an admin poll is the right cadence.
/// `host`/`openvino_version` are individually omitted (never `null`) when
/// unavailable, matching the same convention `generation_metadata` uses.
///
/// `pid`/`uptime_secs` let `rv status` identify the process that is *actually
/// answering* without OS-specific probing (`/proc`, `pgrep`, PID files) — the
/// probes it used to rely on do not exist on Windows. Like `host`, they go
/// only to key-validated callers.
#[derive(Serialize)]
struct ServerInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    host: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pid: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    uptime_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    openvino_version: Option<String>,
    engine: &'static str,
    engine_version: &'static str,
    /// The GPU `OpenCL` context is wedged (an earlier `CL_OUT_OF_RESOURCES`);
    /// every GPU model is unusable and only a process restart clears it.
    /// Always present, to every caller: it is state an operator must not miss.
    gpu_poisoned: bool,
    /// The device-headroom state of each discrete GPU, to every caller (state, not identity).
    #[serde(skip_serializing_if = "Option::is_none")]
    headroom: Option<crate::model_manager::headroom::HeadroomReport>,
    /// Configuration problems found at boot that did not stop it (step 8). Key-validated callers
    /// only (messages name paths); omitted when there are none.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    startup_warnings: Vec<crate::startup::StartupWarning>,
    /// GPU/NPU driver versions and the kernel release (step 8). Key-validated callers only.
    #[serde(skip_serializing_if = "Option::is_none")]
    platform: Option<crate::startup::PlatformInfo>,
}

impl ServerInfo {
    /// `with_host`: whether the caller was key-validated — the hostname, PID and
    /// uptime go only to those (see [`crate::Authenticated`]), never on a
    /// keyless server.
    fn current(with_host: bool) -> Self {
        Self {
            host: if with_host {
                crate::os_memory::host_name()
            } else {
                None
            },
            pid: with_host.then(std::process::id),
            uptime_secs: if with_host {
                crate::startup::process_uptime_secs()
            } else {
                None
            },
            openvino_version: crate::ov_cb::openvino_version(),
            engine: "RustedVINO",
            engine_version: env!("CARGO_PKG_VERSION"),
            gpu_poisoned: false,
            headroom: None,
            startup_warnings: if with_host {
                crate::startup::startup_warnings()
            } else {
                Vec::new()
            },
            platform: with_host.then(crate::startup::platform_info),
        }
    }
}

/// GET /v1/admin/models — list every registered model with its current state.
///
/// Unlike `GET /v1/models` (OpenAI-compatible: Ready models plus `on_demand` ones not loaded
/// yet), this returns the
/// full registry: models that are `not_loaded`, `loading`, `ready`, or
/// `evicting`. Useful for clients that want to know what can be loaded.
///
/// Each entry also carries `max_prompt_tokens`/`cache_size_gb` — the L0
/// prompt-length gate ceiling and the KV pool it was computed against, both
/// `0`/`0.0` before first load or once evicted. Added so an external client
/// or operator can discover a model's real usable context window
/// proactively, instead of only reactively via a `400
/// context_length_exceeded` after crossing it
/// (the project's internal engineering log).
///
/// Also carries `native_context_limit` (omitted when unknown) — the model's
/// own trained/converted context ceiling, independent of `max_prompt_tokens`'
/// VRAM-derived number, so an operator can tell "VRAM-limited" from
/// "model-limited" at a glance
/// (the project's internal engineering log).
///
/// - 200 OK — registry snapshot returned
/// - 503 Service Unavailable — model manager not initialised
pub async fn admin_list_models(
    State(state): State<AppState>,
    authenticated: Option<Extension<crate::Authenticated>>,
) -> Response {
    let Some(mm) = state.model_manager.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model manager is not initialised",
        )
            .into_response();
    };

    mm.reap_stalled_loads();
    let models = mm
        .list_models()
        .into_iter()
        .map(|info| AdminModelEntry {
            id: info.id,
            state: if mm.gpu_unavailable(&info.device) {
                STATE_GPU_POISONED
            } else if info.stalled_secs.is_some() || info.generation_stalled {
                "stalled"
            } else {
                match info.state {
                    crate::model_manager::ModelState::NotLoaded => "not_loaded",
                    crate::model_manager::ModelState::Loading => "loading",
                    crate::model_manager::ModelState::Ready => "ready",
                    crate::model_manager::ModelState::Evicting => "evicting",
                }
            },
            kind: info.kind.label(),
            vram_gb: info.vram_gb,
            device: info.device,
            default_temperature: info.generation_defaults.temperature,
            default_top_p: info.generation_defaults.top_p,
            default_top_k: info.generation_defaults.top_k,
            sampling: info.sampling.map(|(p, npu)| p.admin_json(npu)),
            thinking_budget_tokens: info.thinking_budget_tokens,
            max_prompt_tokens: info.max_prompt_tokens,
            cache_size_gb: info.kv_cache_gb,
            kv_requested_gb: info.kv_requested_gb,
            clamped_by: info.kv_clamped_by.map(crate::model_manager::KvClamp::label),
            native_context_limit: info.native_context_limit,
            kv_cache_usage_pct: info.kv_cache_usage_pct,
            kv_cache_usage_supported: info.kv_cache_usage_supported,
            kv_pressure_flagged: info.kv_pressure_flagged,
            gpu_memory_estimate_gb: info.gpu_memory_estimate_gb,
            kv_cache_precision: info
                .kv_resolution
                .as_ref()
                .map(|r| r.precision.unwrap_or("auto")),
            kv_precision_source: info.kv_resolution.as_ref().map(|r| r.source),
            kv_bytes_per_token: info.kv_resolution.as_ref().and_then(|r| r.bytes_per_token),
            kv_profile_confidence: info
                .kv_resolution
                .as_ref()
                .and_then(|r| r.profile_confidence),
            kv_gate_mode: info.kv_gate_mode,
            limit_clamped_by: info
                .kv_gate_clamped_by
                .map(crate::model_manager::GateClamp::label),
            kv_pool_mib: mib_of_gib(info.kv_cache_gb),
            kv_pool_blocks: info.kv_pool_blocks,
            kv_headroom_cap_tokens: info.kv_headroom_cap_tokens,
            vram_measured_mib: info.vram_measured_mib,
            kv_pool_resident_mib: info.kv_pool_resident_mib,
            kv_admission_low_confidence: info.uncounted_image_admissions > 0,
            uncounted_image_admissions: info.uncounted_image_admissions,
        })
        .collect();

    let mut server = ServerInfo::current(authenticated.is_some());
    server.gpu_poisoned = mm.is_gpu_poisoned();
    // Shown to every caller of this route, like `gpu_poisoned`: it is operational state (what a
    // long prompt can still be given), the VRAM totals are on `/metrics` anyway, and a keyless
    // server would otherwise make `rv status` go quiet. Host, PID and uptime stay key-gated.
    // The first reading scans /proc: keep it off the async threads.
    let mm = std::sync::Arc::clone(mm);
    server.headroom = tokio::task::spawn_blocking(move || mm.headroom_report())
        .await
        .ok()
        .map(|mut report| {
            if authenticated.is_none() {
                // Whether a long request is running right now is activity, not state.
                for d in &mut report.devices {
                    d.leases_mib = None;
                }
            }
            report
        });
    Json(AdminModelsResponse { models, server }).into_response()
}

/// A pool in GiB as whole MiB (the unit the admin API and `rv` show).
fn mib_of_gib(gib: f64) -> u64 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        (gib.max(0.0) * 1024.0).round() as u64
    }
}

/// One model's entry in the `GET /v1/admin/models/completeness` audit.
#[derive(Serialize)]
struct ModelCompletenessEntry {
    id: String,
    kind: &'static str,
    /// `true` when nothing blocking is missing — `advisory_missing` alone
    /// does not make this `false`.
    complete: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    blocking_missing: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    advisory_missing: Vec<String>,
}

/// Response envelope for `GET /v1/admin/models/completeness`.
#[derive(Serialize)]
struct ModelCompletenessResponse {
    models: Vec<ModelCompletenessEntry>,
    /// Count of models with at least one blocking gap. `0` means every
    /// registered model is currently safe to load, file-wise — a real load
    /// can of course still fail for other reasons (VRAM, a corrupt IR file
    /// that parses as present but isn't valid, GPU state).
    incomplete_count: usize,
}

/// `GET /v1/admin/models/completeness` — audit every *registered* model's
/// file completeness (`crate::model_completeness`) without loading any of
/// them.
///
/// Pure filesystem inspection: same check `load_model` runs before any real
/// load (`ModelError::FilesIncomplete`, checked right after `model_exists`,
/// before VRAM reservation or eviction) — surfaced here proactively for
/// every registered model in one call, rather than discovering a gap only
/// when that specific model is finally loaded. Built after a live incident
/// (2026-08-03) where `ms-marco-MiniLM-L6-v2-int8-ov` was missing its
/// converted `openvino_tokenizer.*` pair: the model loaded successfully and
/// only failed on the first real `/v1/rerank` call — this endpoint is the
/// fleet-wide version of the by-hand investigation that incident required.
///
/// - 200 OK — every registered model's report, plus a summary count
/// - 503 Service Unavailable — model manager not initialised
pub async fn admin_models_completeness(State(state): State<AppState>) -> Response {
    let Some(mm) = state.model_manager.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model manager is not initialised",
        )
            .into_response();
    };

    let models: Vec<ModelCompletenessEntry> = mm
        .list_models()
        .into_iter()
        .map(|info| {
            let report = mm.check_model_completeness(&info.id);
            ModelCompletenessEntry {
                id: info.id,
                kind: info.kind.label(),
                complete: report.is_complete(),
                blocking_missing: report.blocking_missing,
                advisory_missing: report.advisory_missing,
            }
        })
        .collect();
    let incomplete_count = models.iter().filter(|m| !m.complete).count();

    Json(ModelCompletenessResponse {
        models,
        incomplete_count,
    })
    .into_response()
}

// ---- /v1/admin/health/watchdog --------------------------------------

/// Response for `GET /v1/admin/health/watchdog`.
#[derive(Serialize)]
struct AdminWatchdogResponse {
    /// How many seconds the worst generation in flight is past its own time allowance (prompt
    /// tokens at a floor prefill speed plus the answer budget at a floor decode speed), across
    /// every Ready model: `0` while every one is inside its allowance, `null` if nothing is
    /// generating right now. The supervisor kills the worker when this passes
    /// `watchdog_hang_ceiling_secs`.
    oldest_generation_secs: Option<f64>,
    /// The `ffi_watch` part of `oldest_generation_secs` alone (loads, transcriptions, CB steps):
    /// seconds the worst of them is past its allowance, `null` when none runs (step 6e).
    oldest_ffi_overdue_secs: Option<f64>,
    /// `OpenVINO` calls in flight right now (`ffi_watch`: loads, transcriptions, CB steps, VLM
    /// generations) — what a process exit would wait for, then report as exit 76.
    in_flight: usize,
    /// Loads past their deadline that have not returned (step 6b).
    stalled: Vec<WatchdogStalled>,
}

/// One entry of [`AdminWatchdogResponse::stalled`].
#[derive(Serialize)]
struct WatchdogStalled {
    device: String,
    model: String,
    /// Always `"load"` today (the only call that is booked as stalled).
    kind: &'static str,
    secs: u64,
    /// What the operator can do if it never returns (D-04 S9): the GT reset command on a
    /// discrete xe GPU with a known PCI address, otherwise advice for that kind of device.
    /// `null` while the load is inside its deadline (stalled only because its client left).
    recovery: Option<String>,
}

/// `GET /v1/admin/health/watchdog` — the supervisor's hang-detection signal.
///
/// Deliberately distinct from `GET /health`: that endpoint only reports
/// whether the HTTP listener is up, which stays `200 ok` even while a single
/// model's dedicated engine thread is permanently wedged inside an `OpenVINO`
/// FFI call (reproduced 2026-07-28,
/// the project's internal engineering log — a GPU driver
/// engine-reset that `openvino_genai` never surfaced as an error). This
/// endpoint reports the one signal that DOES see that: how long the oldest
/// in-flight generation has been running, aggregated in-process via
/// [`crate::model_manager::ModelManager::watchdog_age`].
///
/// Always 200 — including when the model manager isn't initialised yet — so
/// the supervisor's minimal hand-rolled HTTP/1.0 client (`supervisor.rs`,
/// mirrors `probe_health_once`) never has to distinguish a real hang from a
/// transient 5xx.
pub async fn admin_watchdog(State(state): State<AppState>) -> Response {
    let oldest_generation_secs = state
        .model_manager
        .as_ref()
        .and_then(|mm| mm.watchdog_age())
        .map(|d| d.as_secs_f64());
    let watch = crate::ffi_watch::global();
    let stalled = state
        .model_manager
        .as_ref()
        .map(|mm| {
            mm.stalled_loads()
                .into_iter()
                .map(|s| WatchdogStalled {
                    recovery: s.overdue.then(|| mm.stall_recovery(&s.device)),
                    device: s.device,
                    model: s.model,
                    kind: "load",
                    secs: s.secs,
                })
                .collect()
        })
        .unwrap_or_default();
    Json(AdminWatchdogResponse {
        oldest_generation_secs,
        oldest_ffi_overdue_secs: watch.overdue().map(|d| d.as_secs_f64()),
        in_flight: watch.in_flight_count(),
        stalled,
    })
    .into_response()
}

// ---- /v1/admin/voice-pin -------------------------------------------

/// Response for `GET /v1/admin/voice-pin`.
#[derive(Serialize)]
struct VoicePinResponse {
    /// The currently pinned model set, or `null` if no pin is active.
    pin: Option<VoiceFlowPin>,
    /// Number of WebSocket realtime sessions currently open.
    active_sessions: u32,
    /// TTL in seconds — how long after the last session closes before the pin
    /// is cleared.
    ttl_secs: u64,
}

/// GET /v1/admin/voice-pin — return the current voice flow pin.
///
/// Pyramu clients should call this before doing model discovery. If a pin
/// is present, use those model IDs directly in the `ConfigEvent` rather than
/// running `/v1/admin/models` discovery, so all clients converge on the same
/// model set.
///
/// - 200 OK — always (even when `pin` is null — null means no pin yet)
pub async fn admin_voice_pin(State(state): State<AppState>) -> impl IntoResponse {
    let vp = &state.voice_pin;
    Json(VoicePinResponse {
        pin: vp.get(),
        active_sessions: vp.active_sessions(),
        ttl_secs: vp.ttl().as_secs(),
    })
}

// ---- /v1/models ----------------------------------------------------

/// One entry in the `OpenAI` /v1/models list.
#[derive(Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    /// Unix timestamp (seconds). Required by the `OpenAI` schema; reports the
    /// server-start time since no per-model load time is recorded.
    pub created: i64,
    pub owned_by: &'static str,
    /// The effective serving context ceiling in tokens (`max_prompt_tokens`
    /// — the L0 gate's number, VRAM-derived and possibly clamped to the
    /// model's native context limit). Not part of the `OpenAI` schema, but a
    /// de-facto extension several inference servers (e.g. vLLM) and agent
    /// frameworks already read; omitted (not `0`) when unknown, and extra
    /// JSON keys are ignored by strict `OpenAI` clients, so this is
    /// additive/zero-risk. Gives non-admin clients (no admin key) the number
    /// they need to budget context without requiring `GET
    /// /v1/admin/models`
    /// (the project's internal engineering log).
    /// Also omitted for a model that is not loaded yet (`state` `"not_loaded"` / `"loading"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<usize>,
    /// Omitted for a loaded, healthy model, so a Ready model's entry is unchanged. Otherwise:
    /// - `"not_loaded"` / `"loading"` *(since 0.9.1, F1a)*: an `on_demand` model that is not
    ///   loaded yet; a request to it starts (or waits on) its load;
    /// - `"unavailable (gpu_poisoned)"`: a GPU-placed model that cannot serve, or load, until
    ///   the server restarts (E-12).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<&'static str>,
}

/// `state` of a `/v1/models` entry for an `on_demand` model that is not loaded.
const STATE_NOT_LOADED: &str = "not_loaded";
/// `state` of a `/v1/models` entry for an `on_demand` model whose load is running.
const STATE_LOADING: &str = "loading";

/// The public `GET /v1/models` entry for one model, or `None` when the model is not listed
/// (F1a). Shared by the list and the retrieve route so the two cannot drift apart.
///
/// Listed: a `Ready` model (no `state`, or the poisoned one), and an `on_demand` model that is
/// `NotLoaded` or `Loading` (`"not_loaded"` / `"loading"`, or the poisoned state when its GPU
/// context is poisoned, since it could not load). An eager model that is not Ready, and any
/// `Evicting` model, stay hidden: listed = usable. A stalled load reads `"loading"` here (the
/// admin list says `stalled`). The policy is read through [`ModelManager::is_on_demand`], the
/// same predicate that decides whether a request starts a load. `max_model_len` only for a Ready
/// model: before the first load the ceiling is unknown.
///
/// [`ModelManager::is_on_demand`]: crate::model_manager::ModelManager::is_on_demand
fn v1_entry(
    mm: &crate::model_manager::ModelManager,
    info: crate::model_manager::ModelInfo,
) -> Option<ModelObject> {
    use crate::model_manager::ModelState;
    let poisoned = mm.gpu_unavailable(&info.device);
    let state = match info.state {
        ModelState::Ready => poisoned.then_some(STATE_GPU_POISONED),
        ModelState::NotLoaded | ModelState::Loading if mm.is_on_demand(&info.id) => {
            Some(if poisoned {
                STATE_GPU_POISONED
            } else if info.state == ModelState::NotLoaded {
                STATE_NOT_LOADED
            } else {
                STATE_LOADING
            })
        }
        ModelState::NotLoaded | ModelState::Loading | ModelState::Evicting => return None,
    };
    let ready = info.state == ModelState::Ready;
    Some(ModelObject {
        state,
        id: info.id,
        object: "model",
        created: server_created_ts(),
        owned_by: "rustedvino",
        max_model_len: (ready && info.max_prompt_tokens > 0).then_some(info.max_prompt_tokens),
    })
}

/// Top-level /v1/models response envelope.
#[derive(Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

/// GET /v1/models — OpenAI-compatible model discovery.
///
/// Lists the models a client can use: every `Ready` model, and *(since 0.9.1, F1a)* every
/// `on_demand` model that is not loaded or loading, with a `state` field and no
/// `max_model_len` (see [`v1_entry`]). An eager model that is not loaded stays invisible to
/// `OpenAI` clients; the admin routes control loading.
///
/// When `model_manager` is absent (mock mode), returns an empty list.
pub async fn models_list(State(state): State<AppState>) -> Json<ModelsResponse> {
    let data = state
        .model_manager
        .as_ref()
        .map(|mm| {
            // A stalled load that has returned shows up here as Ready (step 6b).
            mm.reap_stalled_loads();
            // `list_models` returns a snapshot (the models lock is released) before
            // `v1_entry` reads the policy overlay.
            mm.list_models()
                .into_iter()
                .filter_map(|info| v1_entry(mm, info))
                .collect()
        })
        .unwrap_or_default();

    Json(ModelsResponse {
        object: "list",
        data,
    })
}

/// GET /v1/models/{id} — `OpenAI`'s retrieve-model route (D-12).
///
/// Same visibility rule as the list ([`v1_entry`]): a `Ready` model, or an `on_demand` model
/// that is not loaded or loading, answers its entry. Any other registered model (an eager one
/// that is not loaded, or one being evicted) answers 404 `model_not_found` like an unknown id,
/// with the not-loaded sentence as its message (F1a). Without a model manager (mock mode) every
/// id is unknown.
pub async fn model_retrieve(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    let Some(mm) = state.model_manager.as_ref() else {
        return super::error::model_error_response(crate::model_manager::ModelError::NotFound(id));
    };
    mm.reap_stalled_loads();
    let Some(info) = mm.list_models().into_iter().find(|info| info.id == id) else {
        return super::error::model_error_response(crate::model_manager::ModelError::NotFound(id));
    };
    match v1_entry(mm, info) {
        Some(obj) => Json(obj).into_response(),
        None => super::error::openai_error(
            StatusCode::NOT_FOUND,
            super::error::not_loaded_message(Some(&id)),
            "invalid_request_error",
            Some("model_not_found"),
        ),
    }
}

// ---- /metrics ------------------------------------------------------

/// GET /metrics — Prometheus text exposition (Phase 3.3).
///
/// Sets the state gauges from a live `ModelManager` snapshot, then renders the
/// recorder. Served on the same port as the API (`:11437`). Returns 503 when
/// metrics are not installed or no model manager is attached (mock/test mode).
pub async fn metrics_handler(State(state): State<AppState>) -> Response {
    let (Some(mm), Some(handle)) = (state.model_manager.as_ref(), state.metrics_handle.as_ref())
    else {
        return (StatusCode::SERVICE_UNAVAILABLE, "metrics not enabled").into_response();
    };

    crate::metrics::record_state(&mm.metrics_snapshot());

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        handle.render(),
    )
        .into_response()
}

// ---- /v1/admin/models/* --------------------------------------------

/// Optional JSON body for `POST /v1/admin/models/{id}/load` — per-load runtime
/// overrides (co-residency Slice 2b).
///
/// An empty body (or no body) means "no overrides — use the per-model policy
/// and global config", which is the pre-Slice-2b behaviour. Both fields are
/// optional and independent: send only the one you want to override. Unknown
/// fields are rejected (a typo like `kv_cache` instead of `kv_cache_gb` would
/// otherwise be silently ignored).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LoadRequest {
    /// Override the KV-cache pool target in GB for this load. Must be `> 0`;
    /// to request an unbounded (grab-all-remaining) pool, omit the field.
    #[serde(default)]
    kv_cache_gb: Option<f64>,
    /// Override the max concurrent streams (`max_num_seqs`) for this load.
    /// Must be `>= 1`.
    #[serde(default)]
    max_concurrent_streams: Option<usize>,
    /// Bypass the eviction-grace window AND the busy rule for this load (realtime voice
    /// arbitration v2 D3, then the busy-eviction policy): a model with a request running is
    /// otherwise never evicted to make room — the load is refused with `503 model_busy` — and
    /// with `force` it is drained (up to 15 s) and what is left is aborted. Never the realtime
    /// serving set, `pinned`, or non-`evictable`, all of which remain unconditionally
    /// protected. Default `false`.
    #[serde(default)]
    force: bool,
}

impl LoadRequest {
    /// Validate the request and convert it into [`LoadOverrides`].
    ///
    /// Returns `Err(message)` (mapped to HTTP 400 by the caller) when a present
    /// field is out of range. `kv_cache_gb` must be a finite value `> 0`;
    /// `max_concurrent_streams` must be `>= 1`.
    fn into_overrides(self) -> Result<LoadOverrides, String> {
        if let Some(kv) = self.kv_cache_gb
            && (!kv.is_finite() || kv <= 0.0)
        {
            return Err(format!(
                "kv_cache_gb must be a finite value greater than 0 (got {kv}); omit the field for an unbounded pool"
            ));
        }
        if let Some(streams) = self.max_concurrent_streams
            && streams < 1
        {
            return Err("max_concurrent_streams must be at least 1".to_owned());
        }
        Ok(LoadOverrides {
            kv_cache_gb: self.kv_cache_gb,
            max_concurrent_streams: self.max_concurrent_streams,
            force: self.force,
            // Internal only (C27 respawn restore); never from a request body.
            no_evict: false,
            resize_rollback: false,
        })
    }
}

#[allow(clippy::doc_markdown)]
/// POST /v1/admin/models/{model_id}/load — load a model into VRAM.
///
/// Accepts an optional [`LoadRequest`] JSON body of per-load runtime overrides
/// (Slice 2b). An empty body loads with the model's configured policy.
///
/// - 200 OK       — model is now Ready
/// - 400 Bad Request — body is malformed JSON or has an out-of-range override
/// - 404 Not Found — model ID is not in the registry
/// - 409 Conflict  — model is already loading or in a non-loadable state
/// - 503 Service Unavailable — model manager not initialised, or (JSON, code `model_busy`,
///   `Retry-After`) the room the load needs is held by models with a request running: nothing
///   was evicted — retry, or load with `"force": true`; or (`eviction_incomplete`) a forced
///   eviction's engine is still finishing its request
/// - 500 Internal Server Error — engine / factory failure
///
/// Loading may take 10–60 seconds depending on model size and GPU JIT.
/// The request blocks until the model is ready (or fails). Callers should
/// use a long HTTP timeout.
pub async fn admin_load_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Parse + validate the optional overrides body first, so a malformed
    // request is a 400 regardless of server state. An empty body is the
    // pre-Slice-2b path: no overrides.
    let overrides = if body.is_empty() {
        LoadOverrides::default()
    } else {
        // Bad JSON (an unknown field included) is the 400 `invalid_body` envelope (P-C).
        match super::error::admin_json_from_bytes::<LoadRequest>(&body) {
            Ok(req) => match req.into_overrides() {
                Ok(ov) => ov,
                Err(msg) => return (StatusCode::BAD_REQUEST, msg).into_response(),
            },
            Err(resp) => return *resp,
        }
    };

    let Some(mm) = state.model_manager.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model manager is not initialised",
        )
            .into_response();
    };

    let forced = overrides.force;
    match mm.load_model_reported(&model_id, overrides).await {
        Ok(victims) => {
            let note = kv_pool_note(mm, &model_id);
            let sep = if note.is_empty() { "" } else { ". " };
            eviction_success(
                &headers,
                &victims,
                forced,
                format!("model '{model_id}' loaded{sep}{note}"),
            )
        }
        Err(e) => {
            // Downcast to ModelError for fine-grained HTTP status mapping.
            if let Some(model_err) = e.downcast_ref::<ModelError>() {
                match model_err {
                    ModelError::ConfigFileInvalid(_) => {
                        return super::error::config_file_invalid_response(model_err);
                    }
                    // D3: busy-eviction refusals and an incomplete forced eviction carry the JSON
                    // envelope.
                    ModelError::TargetBusy(_)
                    | ModelError::BlockedByBusy(_)
                    | ModelError::EvictionIncomplete(_) => {
                        return eviction_refusal(model_err);
                    }
                    ModelError::LoadTimeout { .. } | ModelError::DeviceStalled { .. } => {
                        return super::error::load_stall_response(model_err);
                    }
                    ModelError::NotFound(_)
                    | ModelError::FilesMissing(_)
                    | ModelError::FilesIncomplete(..)
                    | ModelError::TemplateUnreadable { .. } => {
                        return (StatusCode::NOT_FOUND, format!("{e:#}")).into_response();
                    }
                    ModelError::Loading | ModelError::Evicting => {
                        return (StatusCode::CONFLICT, format!("{e:#}")).into_response();
                    }
                    ModelError::InsufficientMemory(_) => {
                        // A capacity refusal, not a fault: the model can never fit, or GenAI found too little
                        // free device memory (MR-2, retry later); models already evicted are named.
                        return (StatusCode::SERVICE_UNAVAILABLE, format!("{e:#}")).into_response();
                    }
                    // D-04 S2: the same 503 `server_stopping` envelope as the inference routes.
                    ModelError::ShuttingDown => {
                        return super::error::model_error_response(ModelError::ShuttingDown);
                    }
                    ModelError::GpuPoisoned => {
                        // L3: OpenCL context poisoned by a prior OOM. G1 (step 7): a supervised
                        // worker already on its way to a restart says so; otherwise the operator
                        // must restart (gracefully: SIGTERM drains and frees VRAM, T6.1).
                        return gpu_poisoned_admin(&e);
                    }
                    // 0.9.1 S1: a stalled engine's model — the inference routes' 503 `model_unavailable`.
                    ModelError::GenerationStalled(id) => {
                        return super::error::model_error_response(ModelError::GenerationStalled(
                            id.clone(),
                        ));
                    }
                    // WrongKind never originates from load_model; fall through to 500.
                    ModelError::NotLoaded | ModelError::WrongKind(_) => {} // fall through to 500
                }
            }
            tracing::error!(model_id, error = %e, "admin_load_model failed");
            (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response()
        }
    }
}

/// Body for `POST /v1/admin/models/{model_id}/resize`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResizeRequest {
    /// The new KV-cache pool target in GB. Required, must be `> 0` — unlike
    /// `/load`'s optional override, a resize with nothing to change is a
    /// no-op the caller should simply not make.
    kv_cache_gb: f64,
    /// Bypass the busy rule and the eviction-grace window (D3): a model with requests running
    /// is refused (409 `model_busy`) unless this is `true`, in which case it is drained
    /// (bounded) and what is left is aborted, and the reload is a forced load. Default `false`.
    #[serde(default)]
    force: bool,
}

/// The sentence a successful resize answers with. A pool sized exactly is whole KV blocks, so it
/// is always a hair under the request: say what was built (and the request only when it visibly
/// differs); a pool clamped by a setting names that setting.
fn resize_message(
    model_id: &str,
    achieved: f64,
    requested: f64,
    blocks: Option<u64>,
    clamped_by: Option<crate::model_manager::KvClamp>,
) -> String {
    if let (Some(blocks), None) = (blocks, clamped_by) {
        if (achieved - requested).abs() < 0.005 {
            format!(
                "model '{model_id}' resized to {} GiB ({blocks} KV blocks)",
                fmt_gib(achieved)
            )
        } else {
            format!(
                "model '{model_id}' resized to {} GiB ({blocks} KV blocks; requested {})",
                fmt_gib(achieved),
                fmt_gib(requested)
            )
        }
    } else if (achieved - requested).abs() < 1e-9 {
        format!("model '{model_id}' resized to {} GiB", fmt_gib(achieved))
    } else {
        let why = clamped_by.map_or_else(
            || "clamped by free VRAM, cache_size_gb or the min_kv_cache_gb floor".to_owned(),
            kv_clamp_phrase,
        );
        format!(
            "model '{model_id}' resized to {} GiB (requested {} \u{2014} {why})",
            fmt_gib(achieved),
            fmt_gib(requested)
        )
    }
}

#[allow(clippy::doc_markdown)]
/// POST /v1/admin/models/{model_id}/resize — evict then reload a currently-
/// resident model with a new KV-cache size
/// (the project's internal engineering log).
///
/// Unlike `/load`, this is NOT idempotent-when-Ready by design: `/load`'s
/// idempotency is exactly why this endpoint exists (a repeated `/load` call
/// silently ignores a new `kv_cache_gb` on an already-Ready model). This
/// call always evicts-then-reloads when it succeeds.
///
/// - 200 OK       — model resized (evicted and reloaded at the new size)
/// - 400 Bad Request — body is malformed JSON or `kv_cache_gb` is not finite/positive
/// - 404 Not Found — model ID is not in the registry
/// - 409 Conflict  — model is not currently `Ready` (not loaded, loading, or
///   already evicting) — resize only applies to a resident model; use
///   `/load` for one that isn't loaded yet
/// - 429 Too Many Requests — called again within `kv_resize_cooldown_secs`
///   of this model's last resize
/// - 503 Service Unavailable — model manager not initialised, or the GPU
///   context is poisoned (same as `/load`)
/// - 500 Internal Server Error — engine/factory failure during the reload;
///   the model may be left unloaded (evicted but not successfully reloaded
///   at the new size) — deliberately no automatic rollback, see the plan.
///
/// Resizing may take as long as an evict+load cycle (drain the old engine,
/// then a fresh JIT compile) — callers should use a long HTTP timeout, same
/// guidance as `/load`.
pub async fn admin_resize_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Bad JSON (an unknown field included) is the 400 `invalid_body` envelope (P-C).
    let request: ResizeRequest = match super::error::admin_json_from_bytes(&body) {
        Ok(req) => req,
        Err(resp) => return *resp,
    };
    if !request.kv_cache_gb.is_finite() || request.kv_cache_gb <= 0.0 {
        return (
            StatusCode::BAD_REQUEST,
            format!(
                "kv_cache_gb must be a finite positive number, got {}",
                request.kv_cache_gb
            ),
        )
            .into_response();
    }

    let Some(mm) = state.model_manager.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model manager is not initialised",
        )
            .into_response();
    };

    match mm
        .resize_model_kv_cache_reported(&model_id, request.kv_cache_gb, request.force)
        .await
    {
        // `achieved` is the pool the allocator actually handed out, which can be
        // short of the request: the KV target is clamped by free VRAM after
        // weights, by the global `cache_size_gb` cap, and floored at
        // `min_kv_cache_gb`. Report the achieved figure, and say so explicitly
        // when it differs — echoing the request turned a clamped pool into a
        // silent full-size "success".
        Ok((achieved, reports)) => {
            let info = mm.list_models().into_iter().find(|m| m.id == model_id);
            let (clamped_by, blocks) = info
                .as_ref()
                .map_or((None, None), |m| (m.kv_clamped_by, m.kv_pool_blocks));
            let msg = resize_message(&model_id, achieved, request.kv_cache_gb, blocks, clamped_by);
            eviction_success(&headers, &reports, request.force, msg)
        }
        Err(e) => {
            // Cooldown rejection is a plain anyhow error, not a ModelError
            // variant (same string-matching convention admin_reload_config
            // already uses for its own narrow "config_path not set" case) —
            // a new enum variant would ripple through every exhaustive
            // ModelError match in this file for one rare, purely-informational
            // rejection.
            if format!("{e:#}").contains("cooldown is") {
                return (StatusCode::TOO_MANY_REQUESTS, format!("{e:#}")).into_response();
            }
            if let Some(model_err) = e.downcast_ref::<ModelError>() {
                match model_err {
                    ModelError::ConfigFileInvalid(_) => {
                        return super::error::config_file_invalid_response(model_err);
                    }
                    // D3: busy-eviction refusals and an incomplete forced eviction carry the JSON
                    // envelope.
                    ModelError::TargetBusy(_)
                    | ModelError::BlockedByBusy(_)
                    | ModelError::EvictionIncomplete(_) => {
                        return eviction_refusal(model_err);
                    }
                    ModelError::LoadTimeout { .. } | ModelError::DeviceStalled { .. } => {
                        return super::error::load_stall_response(model_err);
                    }
                    ModelError::NotFound(_) => {
                        return (StatusCode::NOT_FOUND, format!("{e:#}")).into_response();
                    }
                    ModelError::NotLoaded | ModelError::Loading | ModelError::Evicting => {
                        return (StatusCode::CONFLICT, format!("{e:#}")).into_response();
                    }
                    ModelError::InsufficientMemory(_) => {
                        // A capacity refusal, not a fault: the model can never fit, or GenAI found too little
                        // free device memory (MR-2, retry later); models already evicted are named.
                        return (StatusCode::SERVICE_UNAVAILABLE, format!("{e:#}")).into_response();
                    }
                    // D-04 S2: the same 503 `server_stopping` envelope as the inference routes.
                    ModelError::ShuttingDown => {
                        return super::error::model_error_response(ModelError::ShuttingDown);
                    }
                    ModelError::GpuPoisoned => {
                        return gpu_poisoned_admin(&e);
                    }
                    // 0.9.1 S1: a stalled engine's model — the inference routes' 503 `model_unavailable`.
                    ModelError::GenerationStalled(id) => {
                        return super::error::model_error_response(ModelError::GenerationStalled(
                            id.clone(),
                        ));
                    }
                    // FilesMissing/FilesIncomplete/WrongKind never originate
                    // from resize_model_kv_cache — completeness is only
                    // checked before a load, and resize never changes kind.
                    // TemplateUnreadable does (C8: checked before evicting) and
                    // keeps the 500 below, as before the variant existed.
                    ModelError::FilesMissing(_)
                    | ModelError::FilesIncomplete(..)
                    | ModelError::TemplateUnreadable { .. }
                    | ModelError::WrongKind(_) => {}
                }
            }
            tracing::error!(model_id, error = %e, "admin_resize_model failed");
            (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response()
        }
    }
}

#[allow(clippy::doc_markdown)]
/// DELETE /v1/admin/models/{model_id} — evict a model from VRAM.
///
/// A model with a request running is not evicted: the call is refused with the JSON 409
/// `model_busy` and the request is untouched. `?force=true` drains it (up to 2 minutes) and
/// aborts what is left.
///
/// - 200 OK       — model evicted; VRAM freed
/// - 404 Not Found — model ID is not in the registry
/// - 409 Conflict  — model is not in a state that can be evicted (not loaded, already
///   evicting), or (JSON, code `model_busy`) it has a request running and `force` was not given
/// - 503 Service Unavailable — model manager not initialised, or the GPU context is poisoned
///   (eviction refused — dropping a resident pipeline on a poisoned context can crash the
///   process; see the project's internal engineering log)
/// - 500 Internal Server Error — engine thread panicked during eviction
///
/// Eviction blocks until the engine thread has drained all in-flight requests
/// and exited. In-flight requests receive their remaining tokens before the
/// model is freed. For every engine except the continuous-batching text engine a FORCED
/// eviction waits up to 30 s for the engine thread to finish its request and otherwise
/// answers 503 `eviction_incomplete` (retryable; the memory is released when it exits).
pub async fn admin_unload_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    headers: HeaderMap,
    query: Result<axum::extract::Query<ForceQuery>, axum::extract::rejection::QueryRejection>,
) -> Response {
    let force = match force_from_query(query) {
        Ok(force) => force,
        Err(resp) => return resp,
    };
    let Some(mm) = state.model_manager.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model manager is not initialised",
        )
            .into_response();
    };

    match mm.evict_model_reported(&model_id, force).await {
        Ok(report) => eviction_success(
            &headers,
            &[report],
            force,
            format!("model '{model_id}' evicted"),
        ),
        Err(e) => {
            if let Some(model_err) = e.downcast_ref::<ModelError>() {
                match model_err {
                    ModelError::ConfigFileInvalid(_) => {
                        return super::error::config_file_invalid_response(model_err);
                    }
                    // D3: busy-eviction refusals and an incomplete forced eviction carry the JSON
                    // envelope.
                    ModelError::TargetBusy(_)
                    | ModelError::BlockedByBusy(_)
                    | ModelError::EvictionIncomplete(_) => {
                        return eviction_refusal(model_err);
                    }
                    ModelError::LoadTimeout { .. } | ModelError::DeviceStalled { .. } => {
                        return super::error::load_stall_response(model_err);
                    }
                    ModelError::NotFound(_) => {
                        return (StatusCode::NOT_FOUND, format!("{e:#}")).into_response();
                    }
                    ModelError::NotLoaded | ModelError::Evicting | ModelError::Loading => {
                        return (StatusCode::CONFLICT, format!("{e:#}")).into_response();
                    }
                    // L3 gate, eviction side (2026-08-04,
                    // the project's internal engineering log):
                    // begin_evict now refuses eviction while the GPU context is
                    // poisoned, so this arm is reachable — mapped the same way
                    // model_error_response maps it for the load path.
                    // InsufficientMemory: a predictable capacity refusal, not a fault.
                    // D-04 S2: the same 503 `server_stopping` envelope as the inference routes.
                    ModelError::ShuttingDown => {
                        return super::error::model_error_response(ModelError::ShuttingDown);
                    }
                    ModelError::GpuPoisoned => return gpu_poisoned_admin(&e),
                    ModelError::InsufficientMemory(_) => {
                        return (StatusCode::SERVICE_UNAVAILABLE, format!("{e:#}")).into_response();
                    }
                    // 0.9.1 S1: a stalled engine's model — the inference routes' 503 `model_unavailable`.
                    ModelError::GenerationStalled(id) => {
                        return super::error::model_error_response(ModelError::GenerationStalled(
                            id.clone(),
                        ));
                    }
                    // WrongKind/FilesMissing/FilesIncomplete never originate
                    // from evict_model — completeness is only checked before a load.
                    ModelError::WrongKind(_)
                    | ModelError::FilesMissing(_)
                    | ModelError::FilesIncomplete(..)
                    | ModelError::TemplateUnreadable { .. } => {}
                }
            }
            tracing::error!(model_id, error = %e, "admin_unload_model failed");
            (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response()
        }
    }
}

#[allow(clippy::doc_markdown)]
/// DELETE /v1/admin/models/{model_id}/register — deregister a model entirely.
///
/// Unlike `DELETE /v1/admin/models/{model_id}` (evict from VRAM, stay
/// registered), this erases the model from the live registry and, if it was
/// durably persisted, from `config.json` — it will not reappear in
/// `GET /v1/admin/models`, on the next restart, or after a config reload.
/// Evicts from VRAM first if the model is currently `Ready`: a model with a request running is
/// refused with the JSON 409 `model_busy` unless `?force=true` (drain, then abort).
///
/// - 200 OK        — model deregistered (and evicted first, if it was loaded)
/// - 404 Not Found — model ID is not in the registry
/// - 409 Conflict  — a load or eviction is already in flight for this model, or (JSON, code
///   `model_busy`) it has a request running and `force` was not given
/// - 503 Service Unavailable — model manager not initialised
/// - 500 Internal Server Error — eviction failed unexpectedly
pub async fn admin_deregister_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    headers: HeaderMap,
    query: Result<axum::extract::Query<ForceQuery>, axum::extract::rejection::QueryRejection>,
) -> Response {
    let force = match force_from_query(query) {
        Ok(force) => force,
        Err(resp) => return resp,
    };
    let Some(mm) = state.model_manager.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model manager is not initialised",
        )
            .into_response();
    };

    match mm.deregister_model_reported(&model_id, force).await {
        Ok(report) => eviction_success(
            &headers,
            report.as_slice(),
            force,
            format!("model '{model_id}' deregistered"),
        ),
        Err(e) => {
            if let Some(model_err) = e.downcast_ref::<ModelError>() {
                match model_err {
                    ModelError::ConfigFileInvalid(_) => {
                        return super::error::config_file_invalid_response(model_err);
                    }
                    // D3: busy-eviction refusals and an incomplete forced eviction carry the JSON
                    // envelope.
                    ModelError::TargetBusy(_)
                    | ModelError::BlockedByBusy(_)
                    | ModelError::EvictionIncomplete(_) => {
                        return eviction_refusal(model_err);
                    }
                    ModelError::LoadTimeout { .. } | ModelError::DeviceStalled { .. } => {
                        return super::error::load_stall_response(model_err);
                    }
                    ModelError::NotFound(_) => {
                        return (StatusCode::NOT_FOUND, format!("{e:#}")).into_response();
                    }
                    ModelError::Loading | ModelError::Evicting => {
                        return (StatusCode::CONFLICT, format!("{e:#}")).into_response();
                    }
                    ModelError::InsufficientMemory(_) => {
                        // A capacity refusal, not a fault: the model can never fit, or GenAI found too little
                        // free device memory (MR-2, retry later); models already evicted are named.
                        return (StatusCode::SERVICE_UNAVAILABLE, format!("{e:#}")).into_response();
                    }
                    // D-04 S2: the same 503 `server_stopping` envelope as the inference routes.
                    ModelError::ShuttingDown => {
                        return super::error::model_error_response(ModelError::ShuttingDown);
                    }
                    ModelError::GpuPoisoned => {
                        return gpu_poisoned_admin(&e);
                    }
                    // 0.9.1 S1: a stalled engine's model — the inference routes' 503 `model_unavailable`.
                    ModelError::GenerationStalled(id) => {
                        return super::error::model_error_response(ModelError::GenerationStalled(
                            id.clone(),
                        ));
                    }
                    // deregister_model only ever evicts a Ready model — NotLoaded,
                    // WrongKind, FilesMissing, and FilesIncomplete never originate
                    // from that path.
                    ModelError::NotLoaded
                    | ModelError::WrongKind(_)
                    | ModelError::FilesMissing(_)
                    | ModelError::FilesIncomplete(..)
                    | ModelError::TemplateUnreadable { .. } => {}
                }
            }
            tracing::error!(model_id, error = %e, "admin_deregister_model failed");
            (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response()
        }
    }
}

// ---- PATCH /v1/admin/models/{id} ------------------------------------

/// Response body for `PATCH /v1/admin/models/{model_id}`.
#[derive(Serialize)]
struct PatchModelResponse {
    model_id: String,
    /// Field names already in effect — immediately, for a `Ready` model too.
    applied_live: Vec<&'static str>,
    /// Field names persisted but only changing engine behaviour at this
    /// model's next load (`vram_gb`, `kv_cache_gb`, `max_concurrent_streams`,
    /// `max_prompt_len`, `min_response_len`, `speculative`).
    effective_on_next_load: Vec<&'static str>,
    /// Field names supplied whose value already matched — accepted, a no-op.
    unchanged: Vec<&'static str>,
    /// The full entry as it now reads in `config.json` — every `ModelPolicy`
    /// field, not just the ones this call touched. The only place a caller
    /// can currently read a model's complete policy back (`GET
    /// /v1/admin/models` doesn't carry it).
    #[serde(flatten)]
    entry: crate::model_manager::config::ModelEntry,
}

#[allow(clippy::doc_markdown)]
/// PATCH /v1/admin/models/{model_id} — update fields of an already-registered
/// model, live where safe, persisted to `config.json` always.
///
/// The missing "update" alongside `POST .../add` (create) and
/// `DELETE .../register` (destroy) — before this, changing a live model's
/// `vram_gb`/`load` policy/etc. needed evict + hand-edit `config.json` +
/// `POST .../config/reload` + reload. Every field is optional; only supplied
/// fields change. `kind` and `model_id` are immutable (400) — deregister and
/// re-add to change either.
///
/// `device`/`tier_preference` are placement-affecting: rejected (409) while
/// the model is `Ready` (they'd disagree with the resident engine's actual
/// placement — evict first), applied immediately while `NotLoaded`.
/// `vram_gb`/`kv_cache_gb`/`max_concurrent_streams`/`max_prompt_len`/`min_response_len`/
/// `speculative` persist immediately but only take effect at this model's
/// *next* load. Everything else (`pinned`/`priority`/`evictable`/`load`/
/// `reasoning_parser`/`capabilities`/`eviction_grace_secs`/image-provenance
/// fields) is live the moment this call returns.
///
/// - 200 OK — JSON [`PatchModelResponse`]
/// - 400 Bad Request — `kind` supplied (immutable), empty body, invalid
///   `vram_gb`/`device`/policy field, or a policy field fails validation
/// - 400 `invalid_body` (envelope, P-C) — malformed JSON, a wrong type
/// - 404 Not Found — model ID is not registered
/// - 409 Conflict — model is `Loading`/`Evicting`, or `device`/
///   `tier_preference` supplied while `Ready`
/// - 415 — no `Content-Type: application/json` (axum's answer)
/// - 422 — an unknown field, at any depth (axum's plain-text answer, kept)
/// - 503 Service Unavailable — model manager not initialised, or the server
///   was started without a durable config file
pub async fn admin_patch_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    body: Result<
        Json<crate::model_manager::config::ModelEntryPatch>,
        axum::extract::rejection::JsonRejection,
    >,
) -> Response {
    // P-C: malformed JSON, a wrong type or a missing field is the 400 `invalid_body` envelope;
    // an unknown field (at any depth) keeps its 422 exactly as before, and a missing
    // `Content-Type` its 415.
    let patch = match body {
        Ok(Json(patch)) => patch,
        Err(r) => {
            return super::error::admin_json_rejection(r, super::error::UnknownField::Keep422);
        }
    };
    let Some(mm) = state.model_manager.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model manager is not initialised",
        )
            .into_response();
    };

    // `patch_model` is synchronous and does blocking file I/O (config.json, and a placement
    // PATCH re-reads the model directory) under a std mutex: run it on the blocking pool.
    let result = {
        let (mm, id) = (std::sync::Arc::clone(mm), model_id.clone());
        match tokio::task::spawn_blocking(move || mm.patch_model(&id, &patch)).await {
            Ok(result) => result,
            // A panicked or cancelled task is a server fault, not the caller's bad input: 500,
            // as the reload handler does.
            Err(e) => {
                tracing::error!(model_id, error = %e, "admin_patch_model task failed");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("PATCH task failed: {e}"),
                )
                    .into_response();
            }
        }
    };
    match result {
        Ok(report) => (
            StatusCode::OK,
            Json(PatchModelResponse {
                model_id,
                applied_live: report.applied_live,
                effective_on_next_load: report.effective_on_next_load,
                unchanged: report.unchanged,
                entry: report.entry,
            }),
        )
            .into_response(),
        Err(e) => {
            if let Some(model_err) = e.downcast_ref::<ModelError>() {
                match model_err {
                    ModelError::ConfigFileInvalid(_) => {
                        return super::error::config_file_invalid_response(model_err);
                    }
                    // D3: busy-eviction refusals and an incomplete forced eviction carry the JSON
                    // envelope.
                    ModelError::TargetBusy(_)
                    | ModelError::BlockedByBusy(_)
                    | ModelError::EvictionIncomplete(_) => {
                        return eviction_refusal(model_err);
                    }
                    ModelError::LoadTimeout { .. } | ModelError::DeviceStalled { .. } => {
                        return super::error::load_stall_response(model_err);
                    }
                    ModelError::NotFound(_) => {
                        return (StatusCode::NOT_FOUND, format!("{e:#}")).into_response();
                    }
                    ModelError::Loading | ModelError::Evicting => {
                        return (StatusCode::CONFLICT, format!("{e:#}")).into_response();
                    }
                    ModelError::InsufficientMemory(_) => {
                        // A capacity refusal, not a fault: the model can never fit, or GenAI found too little
                        // free device memory (MR-2, retry later); models already evicted are named.
                        return (StatusCode::SERVICE_UNAVAILABLE, format!("{e:#}")).into_response();
                    }
                    // D-04 S2: the same 503 `server_stopping` envelope as the inference routes.
                    ModelError::ShuttingDown => {
                        return super::error::model_error_response(ModelError::ShuttingDown);
                    }
                    ModelError::GpuPoisoned
                    | ModelError::NotLoaded
                    | ModelError::WrongKind(_)
                    | ModelError::FilesMissing(_)
                    | ModelError::FilesIncomplete(..)
                    | ModelError::GenerationStalled(_)
                    | ModelError::TemplateUnreadable { .. } => {} // never originate from patch_model
                }
            }
            let msg = format!("{e:#}");
            if msg.contains("config_path not set") {
                return (StatusCode::SERVICE_UNAVAILABLE, msg).into_response();
            }
            if msg.contains("would disagree with the resident engine")
                || msg.contains("stopped being NotLoaded while this PATCH was in flight")
            {
                return (StatusCode::CONFLICT, msg).into_response();
            }
            // Disk I/O / a config.json that no longer parses are server-side
            // failures, not the caller's bad input — string-matching every
            // validation message would be endless and fragile, so this
            // checks for the two concrete non-anyhow-`bail!` error types
            // that can reach here (`?`-propagated from `std::fs`/
            // `serde_json`) instead, same principle `admin_add_model`
            // doesn't need since its errors are all `anyhow::ensure!`s.
            if e.downcast_ref::<std::io::Error>().is_some()
                || e.downcast_ref::<serde_json::Error>().is_some()
            {
                tracing::error!(model_id, error = %e, "admin_patch_model I/O failure");
                return (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response();
            }
            tracing::error!(model_id, error = %e, "admin_patch_model failed");
            (StatusCode::BAD_REQUEST, msg).into_response()
        }
    }
}

// ---- POST /v1/admin/models/{id}/check ------------------------------

/// Response body for `POST /v1/admin/models/{model_id}/check` — the dry-run
/// "what fits" admission check (co-residency Slice 1).
#[derive(Serialize)]
struct AdmissionCheckResponse {
    /// The model the check was run for.
    model: String,
    /// `true` if the model can be admitted now — either it fits in free VRAM
    /// or it fits after evicting `would_evict`. On the UMA `system` domain this
    /// also accounts for the live system-RAM gate (`ram_needed_gb`), which is
    /// the gate that actually refuses loads there.
    fits: bool,
    /// VRAM the load needs (`weights + min KV + safety margin`), GB.
    needed_gb: f64,
    /// Unreserved VRAM in the model's inference domain before any eviction, GB.
    /// `0.0` when VRAM gating is disabled for the domain (see `vram_gated`).
    free_gb: f64,
    /// `false` when the model's domain has no VRAM budget (`total_vram_gb: 0.0`): `free_gb` is then
    /// meaningless and only the live system-RAM gate applies.
    vram_gated: bool,
    /// Residents that would be evicted (in eviction order) to admit the model.
    /// Empty when it already fits or cannot be admitted at all.
    would_evict: Vec<String>,
    /// Members of `would_evict` with a request running now. Non-empty only for a **forced**
    /// check (`?force=true`): a plain load never evicts a busy model (see `blocked_by_busy`),
    /// and a forced one drains each for at most 15 s and then aborts what is left.
    busy_victims: Vec<String>,
    /// Residents that were NOT counted as victims only because they have a request running
    /// (busy models are not evicted without `force`) and whose eviction would have made the
    /// load fit. Empty unless the load cannot proceed for that reason.
    blocked_by_busy: Vec<String>,
    /// Live system-RAM the load needs (`weights + conservative KV + margin`), GB.
    /// Present **only on the shared UMA `system` domain**, where a live-RAM gate
    /// applies on top of the VRAM arithmetic; omitted entirely on discrete-GPU
    /// domains, where no such gate exists. Larger than `needed_gb`: it counts
    /// the model's resolved KV target, not the `min_kv_cache_gb` floor.
    #[serde(skip_serializing_if = "Option::is_none")]
    ram_needed_gb: Option<f64>,
    /// Live available system RAM before any eviction, GB. Present exactly when
    /// `ram_needed_gb` is.
    #[serde(skip_serializing_if = "Option::is_none")]
    ram_avail_gb: Option<f64>,
    /// `true` if the model is already loaded — the check is moot.
    already_loaded: bool,
    /// Why the device-headroom gate (vision models on a discrete GPU) refuses the load even
    /// after every eviction, when that gate is the one refusing. Omitted otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    headroom_refusal: Option<String>,
    /// Why the check could not apply the model's own part of the device-headroom gate (it has
    /// not been loaded on this server yet). Omitted when the gate was fully previewed.
    #[serde(skip_serializing_if = "Option::is_none")]
    headroom_not_previewed: Option<String>,
}

/// Whether the client asked for the JSON outcome of an eviction route (V1-rv P1): an `Accept`
/// media range equal to `application/json` (any case, in any of several `Accept` headers; other
/// parameters such as `charset` ignored) whose `q` is not `0` (RFC 9110: `q=0` means "not
/// acceptable"). `*/*` and no header keep the plain-text body.
fn wants_json(headers: &HeaderMap) -> bool {
    headers
        .get_all(axum::http::header::ACCEPT)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .any(|range| {
            let mut parts = range.split(';');
            let media = parts.next().unwrap_or("").trim();
            let refused = parts.any(|p| {
                p.trim()
                    .split_once('=')
                    .filter(|(k, _)| k.trim().eq_ignore_ascii_case("q"))
                    .and_then(|(_, v)| v.trim().parse::<f64>().ok())
                    .is_some_and(|q| q <= 0.0)
            });
            media.eq_ignore_ascii_case("application/json") && !refused
        })
}

/// The 200 of an unload, deregister, resize or load: the plain-text `message` as before, or, when
/// the client asked for JSON ([`wants_json`]), [`eviction_outcome`] carrying it.
fn eviction_success(
    headers: &HeaderMap,
    reports: &[EvictionReport],
    forced: bool,
    message: String,
) -> Response {
    if wants_json(headers) {
        eviction_outcome(reports, forced, &message)
    } else {
        (StatusCode::OK, message).into_response()
    }
}

/// The JSON outcome of an unload, deregister, resize or load (V1-rv P1; `message` 0.9.1):
/// `{evicted, forced, drained_s, aborted, teardown_ms, message}`. `evicted` lists every model the
/// call evicted, in order (`[]` when it evicted nothing); the three figures are summed over them
/// (the evictions run one after another). `drained_s` is `0` exactly when nothing was running.
/// `message` is the plain-text body the route answers without the header, verbatim.
fn eviction_outcome(reports: &[EvictionReport], forced: bool, message: &str) -> Response {
    let evicted: Vec<&str> = reports.iter().map(|r| r.model_id.as_str()).collect();
    let drained: Duration = reports.iter().map(|r| r.drained).sum();
    let aborted: usize = reports.iter().map(|r| r.aborted).sum();
    let teardown_ms = reports
        .iter()
        .fold(0u64, |acc, r| acc.saturating_add(r.teardown_ms));
    // Milliseconds are enough: the drain polls every 50 ms.
    let drained_s = (drained.as_secs_f64() * 1000.0).round() / 1000.0;
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "evicted": evicted,
            "forced": forced,
            "drained_s": drained_s,
            "aborted": aborted,
            "teardown_ms": teardown_ms,
            "message": message,
        })),
    )
        .into_response()
}

/// Query string of `POST /v1/admin/models/{id}/check`, `DELETE /v1/admin/models/{id}` and
/// `DELETE /v1/admin/models/{id}/register`.
#[derive(serde::Deserialize, Default)]
pub struct ForceQuery {
    /// Bypass the eviction-grace window **and** the busy rule (D3): a model with requests
    /// running is not evicted unless this is set — then it is drained (bounded) and the rest
    /// aborted. Never bypasses `pinned`, non-`evictable` or a live realtime session's models.
    /// For `/check`, previews a load run that way. Default `false`.
    #[serde(default)]
    force: bool,
}

/// Parse the optional `?force=` query of an eviction route, answering a malformed one with the
/// shared JSON envelope (not axum's plain-text rejection).
#[allow(clippy::result_large_err)]
fn force_from_query(
    query: Result<axum::extract::Query<ForceQuery>, axum::extract::rejection::QueryRejection>,
) -> Result<bool, Response> {
    match query {
        Ok(axum::extract::Query(q)) => Ok(q.force),
        Err(rejection) => {
            tracing::debug!(detail = %rejection.body_text(), "rejected malformed query string");
            Err(super::error::openai_error(
                StatusCode::BAD_REQUEST,
                "malformed query string: `force` must be true or false",
                "invalid_request_error",
                Some("invalid_query"),
            ))
        }
    }
}

#[allow(clippy::doc_markdown)]
/// POST /v1/admin/models/{model_id}/check — dry-run admission check.
///
/// Runs the VRAM admission arithmetic **without loading anything**: reports
/// whether the model fits, how much VRAM it needs, what is free now, and which
/// residents would be evicted to make room. No GPU work, no state change — the
/// "tell it what to pin; it tells you what fits" answer. Admin-scoped (the
/// `/v1/admin/` prefix gates it under `admin_api_keys`).
///
/// - 200 OK        — JSON [`AdmissionCheckResponse`]
/// - 404 Not Found  — model ID is not in the registry
/// - 503 Service Unavailable — model manager not initialised
///
/// `?force=true` previews the load as `force: true` would run it (grace window and
/// busy rule bypassed, nothing else).
pub async fn admin_check_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    query: Result<axum::extract::Query<ForceQuery>, axum::extract::rejection::QueryRejection>,
) -> Response {
    // A bad query string (`?force=maybe`) gets the shared JSON envelope, not
    // axum's plain-text rejection.
    let force = match force_from_query(query) {
        Ok(force) => force,
        Err(resp) => return resp,
    };
    let Some(mm) = state.model_manager.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model manager is not initialised",
        )
            .into_response();
    };

    // The dry-run must see the device memory of now, as the load it previews will (S2).
    mm.refresh_all_device_probes().await;
    match mm.check_admission_with(&model_id, force) {
        Ok(check) => (
            StatusCode::OK,
            Json(AdmissionCheckResponse {
                model: model_id,
                fits: check.fits,
                needed_gb: check.needed_gb,
                free_gb: check.free_gb,
                vram_gated: check.vram_gated,
                would_evict: check.would_evict,
                busy_victims: check.busy_victims,
                blocked_by_busy: check.blocked_by_busy,
                ram_needed_gb: check.ram_needed_gb,
                ram_avail_gb: check.ram_avail_gb,
                already_loaded: check.already_loaded,
                headroom_refusal: check.headroom_refusal,
                headroom_not_previewed: check.headroom_not_previewed,
            }),
        )
            .into_response(),
        Err(e) => {
            if let Some(ModelError::NotFound(_)) = e.downcast_ref::<ModelError>() {
                return (StatusCode::NOT_FOUND, format!("{e:#}")).into_response();
            }
            tracing::error!(model_id, error = %e, "admin_check_model failed");
            (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response()
        }
    }
}

// ── Realtime session admin ────────────────────────────────────────────────────

/// `GET /v1/admin/realtime/sessions` — list all live WebSocket sessions.
///
/// Returns a summary for each session (no history). Useful for at-a-glance
/// monitoring of concurrent voice connections.
pub async fn admin_realtime_sessions(State(state): State<AppState>) -> impl IntoResponse {
    let sessions: Vec<serde_json::Value> = {
        let registry = state
            .realtime_sessions
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        registry
            .values()
            .map(|entry| {
                let snap = entry
                    .snapshot
                    .read()
                    .unwrap_or_else(PoisonError::into_inner);
                serde_json::json!({
                    "id": snap.id,
                    "conn_id": snap.conn_id,
                    "connected_at": snap.connected_at,
                    "stt_model": snap.stt_model,
                    "llm_model": snap.llm_model,
                    "tts_model": snap.tts_model,
                    "turn_count": snap.turn_count,
                })
            })
            .collect()
    };
    Json(serde_json::json!({ "sessions": sessions }))
}

/// `GET /v1/admin/realtime/sessions/{id}` — full snapshot for one session.
///
/// Includes conversation history with embeddings, current config, and turn count.
/// Returns 404 when no session with that UUID is found.
pub async fn admin_realtime_session_by_id(
    Path(id): Path<uuid::Uuid>,
    State(state): State<AppState>,
) -> Response {
    let snapshot = {
        let registry = state
            .realtime_sessions
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        registry.get(&id).map(|entry| {
            entry
                .snapshot
                .read()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
        })
    };
    match snapshot {
        Some(snap) => Json(snap).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "session not found" })),
        )
            .into_response(),
    }
}

/// `DELETE /v1/admin/realtime/sessions/{id}` — force-close a live session.
///
/// Fires the session's `kill` token, which cancels the turn in progress and
/// makes the pipeline loop exit. The session then sends the WebSocket close
/// frame 4000 `closed_by_admin`, waits up to 2 s for the client's close reply,
/// and drops the connection (0.9.1 C5).
pub async fn admin_kill_realtime_session(
    Path(id): Path<uuid::Uuid>,
    State(state): State<AppState>,
) -> Response {
    let token = {
        let registry = state
            .realtime_sessions
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        registry.get(&id).map(|entry| entry.kill.clone())
    };
    match token {
        Some(t) => {
            t.cancel();
            tracing::info!(session_id = %id, "admin: realtime session kill requested");
            StatusCode::NO_CONTENT.into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "session not found" })),
        )
            .into_response(),
    }
}

// ---- POST /v1/admin/models/add -------------------------------------

/// Request body for `POST /v1/admin/models/add`.
///
/// `model_id`/`vram_gb` are the only required fields — everything else is a
/// [`ModelPolicy`](crate::model_manager::config::ModelPolicy) field, flattened
/// at the JSON level exactly like a static `config.json` model stanza (same
/// shape, same optionality): `pinned`, `priority`, `kv_cache_gb`,
/// `max_concurrent_streams`, `load`, `evictable`, `device`, `tier_preference`,
/// `reasoning_parser`, `max_prompt_len`, `min_response_len`, `speculative`, and the image-gen Tier
/// 3 provenance fields `precision`/`model_source`/`model_revision`. None of
/// these are obligatory — an operator adding a model at runtime may not know
/// all of it yet; every field defaults exactly as it would in `config.json`.
#[derive(Deserialize)]
pub struct AddModelRequest {
    /// Model ID — must match a subdirectory under `models_dir`.
    pub model_id: String,
    /// VRAM estimate in GB. Use `0.0` to skip VRAM gating (CPU-only).
    pub vram_gb: f64,
    /// Kind hint: `"text_gen"`, `"vision"`, `"embedding"`, `"stt"`,
    /// `"tts"`, `"image_gen"`, `"reranking"`, `"decision"` (never auto-detected). Omit for auto-detection.
    #[serde(default)]
    pub kind: Option<String>,
    /// The full per-model policy (device override, co-residency knobs,
    /// image-gen provenance, …) — see the struct doc comment above.
    #[serde(flatten)]
    pub policy: crate::model_manager::config::ModelPolicy,
    /// Top-level fields no other member claimed (docs-drift L2): still ignored, as before, but
    /// logged as a WARN naming them, so a typo (`pinnned`) does not vanish silently.
    #[serde(flatten)]
    pub unknown: std::collections::BTreeMap<String, serde_json::Value>,
}

/// `POST /v1/admin/models/add` — register a new model at runtime and load it immediately.
///
/// - 200 OK        — model added to config and loaded into VRAM
/// - 400 Bad Request — invalid input (bad `vram_gb`/`kind`/`device`, directory missing,
///   or a policy field fails validation, e.g. `max_concurrent_streams: 0`)
/// - 409 Conflict  — model ID is already registered
/// - 503 Service Unavailable — GPU context poisoned or model manager not ready
/// - 500 Internal Server Error — unexpected error during load
///
/// The model entry — `vram_gb`, `kind`, and every `ModelPolicy` field the
/// caller set — is persisted to `config.json` (it survives restart but is
/// **not** added to `preload` — use `POST /v1/admin/config/preload` if you
/// want it on every boot). The call blocks until the model is loaded; use a
/// long HTTP timeout.
pub async fn admin_add_model(
    State(state): State<AppState>,
    body: Result<Json<AddModelRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    // P-C: bad JSON (a nested unknown key in `sampling`/`speculative` included) is the 400
    // `invalid_body` envelope; a missing `Content-Type` keeps axum's 415.
    let req = match body {
        Ok(Json(req)) => req,
        Err(r) => {
            return super::error::admin_json_rejection(r, super::error::UnknownField::InvalidBody);
        }
    };
    let Some(mm) = state.model_manager.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model manager is not initialised",
        )
            .into_response();
    };

    if !req.unknown.is_empty() {
        let keys: Vec<&str> = req.unknown.keys().map(String::as_str).collect();
        tracing::warn!(
            model = %req.model_id,
            keys = %keys.join(", "),
            "models/add: unknown field(s) ignored — check for typos"
        );
    }
    match mm
        .add_model(req.model_id.clone(), req.vram_gb, req.kind, req.policy)
        .await
    {
        Ok(outcome) => {
            let note = kv_pool_note(mm, &req.model_id);
            let sep = if note.is_empty() { "" } else { ". " };
            let Some(cause) = outcome.persist_error else {
                return (
                    StatusCode::OK,
                    format!(
                        "model '{}' added to config and loaded{sep}{note}",
                        req.model_id
                    ),
                )
                    .into_response();
            };
            // Loaded and serving, but not durable: say so instead of an error that `rv ps`
            // would contradict.
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "message": format!(
                        "model '{}' loaded, but NOT persisted to config.json — it will be gone \
                         after a restart: {cause}{sep}{note}",
                        req.model_id
                    ),
                    "persisted": false,
                    "persist_error": cause,
                })),
            )
                .into_response()
        }
        Err(e) => {
            if let Some(model_err @ crate::model_manager::ModelError::ConfigFileInvalid(_)) =
                e.downcast_ref::<crate::model_manager::ModelError>()
            {
                return super::error::config_file_invalid_response(model_err);
            }
            if let Some(model_err) = e.downcast_ref::<crate::model_manager::ModelError>()
                && matches!(model_err, crate::model_manager::ModelError::GpuPoisoned)
            {
                return gpu_poisoned_admin(&e);
            }
            // D-04 S2: the load that follows the registration was refused because the server is
            // stopping (registration rolled back) — the same 503 `server_stopping` as `/load`.
            if matches!(
                e.downcast_ref::<crate::model_manager::ModelError>(),
                Some(crate::model_manager::ModelError::ShuttingDown)
            ) {
                return super::error::model_error_response(
                    crate::model_manager::ModelError::ShuttingDown,
                );
            }
            // D3: the load that follows the registration was refused because the room is held
            // by busy models (or a forced eviction is still finishing). The registration was
            // rolled back; answer with the same JSON busy codes as `/load` instead of a
            // misleading 400.
            if let Some(
                model_err @ (crate::model_manager::ModelError::BlockedByBusy(_)
                | crate::model_manager::ModelError::TargetBusy(_)
                | crate::model_manager::ModelError::EvictionIncomplete(_)),
            ) = e.downcast_ref::<crate::model_manager::ModelError>()
            {
                return eviction_refusal(model_err);
            }
            let msg = format!("{e:#}");
            if msg.contains("already registered") {
                return (StatusCode::CONFLICT, msg).into_response();
            }
            tracing::error!(
                model_id = %req.model_id,
                error = %e,
                "admin_add_model failed"
            );
            (StatusCode::BAD_REQUEST, msg).into_response()
        }
    }
}

// ---- /v1/admin/config/reload and /audit -----------------------------

/// Response body for `POST /v1/admin/config/reload`.
#[derive(Serialize)]
struct ConfigReloadResponse {
    /// Model IDs newly present in `config.json` and now registered
    /// `NotLoaded`.
    added: Vec<String>,
    /// Model IDs already registered `NotLoaded` whose config fields changed
    /// on disk and were refreshed in place.
    updated: Vec<String>,
    /// Model IDs the file declares but whose directory doesn't exist under
    /// `models_dir` — skipped, same guard as startup registration.
    skipped_missing_files: Vec<String>,
    /// Model IDs the file declares that reload deliberately left alone
    /// because they are `Ready`/`Loading`/`Evicting` — never disrupted.
    left_untouched: Vec<String>,
    /// Count of entries present in both the file and the live registry with
    /// no field differences.
    unchanged: usize,
    /// Global (non-model) config fields whose file value differs from the one
    /// this process booted with — **edits this reload did not apply**. Reload
    /// re-reads only the per-model registry; the globals are a boot snapshot.
    /// Empty on the common path, so the response shape is unchanged for anyone
    /// who never edits a global.
    globals_not_applied: Vec<String>,
    /// Loaded models (`left_untouched`) whose stanza in the file changed, with the changed keys:
    /// `[{"model": "...", "fields": ["chat_template"]}]` (C8). **Not applied** — to apply: unload
    /// the model, reload the config again, then load it. Empty on the common path.
    models_not_applied: Vec<crate::model_manager::ModelChangesNotApplied>,
    /// Loaded (`Ready`) models whose `sampling` or `thinking_budget_tokens` changed in the file:
    /// **applied** to the next request (both are hot), `[{"model": "...", "fields":
    /// ["sampling", "thinking_budget_tokens"]}]` (the ones that changed), and not repeated in
    /// `models_not_applied`. Always present; empty on the common path.
    applied_live: Vec<crate::model_manager::ModelChangesNotApplied>,
}

#[allow(clippy::doc_markdown)]
/// POST /v1/admin/config/reload — pick up `config.json` edits without a
/// restart.
///
/// Registers any model newly declared in the file (validated the same way
/// as startup — a missing directory is skipped, not registered) and
/// refreshes any `NotLoaded` entry whose fields changed. Deliberately
/// zero-blast-radius for anything already live: a `Ready`/`Loading`/`Evicting`
/// model is never touched even if its file entry changed, an entry the file
/// dropped is never removed (use `DELETE .../{id}/register` for that), and no
/// load is ever triggered — not even for a model newly added to `preload`.
/// Global knobs (`total_vram_gb`, `bind_addr`, …) are not re-read — any that
/// differ from the booted values are listed in the response's
/// `globals_not_applied` so the caller can see what was ignored rather than
/// reading a `200 OK` as "everything landed". **Never**
/// touches API keys — those have their own independent reload,
/// `POST /v1/admin/keys/reload` / [`admin_reload_keys`] — see
/// `ModelManager::reload_config`'s doc comment for why the two are split.
///
/// - 200 OK — JSON [`ConfigReloadResponse`] summarising what changed
/// - 422 Unprocessable Entity — `config.json` cannot be read, parsed or validated (e.g. an unsupported
///   `kv_cache_precision`); nothing was changed
/// - 500 Internal Server Error — file unreadable, has deprecated
///   inline `api_keys`/`admin_api_keys` (migration tripwire — move them to
///   the keys file), or placement failed for a newly-added entry (e.g. an
///   explicit `device` this `OpenVINO` build doesn't enumerate)
/// - 503 Service Unavailable — model manager not initialised, or the server
///   was started without a durable config file (`config_path` unset)
pub async fn admin_reload_config(State(state): State<AppState>) -> Response {
    let Some(mm) = state.model_manager.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model manager is not initialised",
        )
            .into_response();
    };

    // Synchronous, with blocking file I/O (config.json and, for the V5 `sampling` rule and new
    // entries, model directories) under a std mutex: run it on the blocking pool.
    let result = {
        let mm = std::sync::Arc::clone(mm);
        tokio::task::spawn_blocking(move || mm.reload_config())
            .await
            .unwrap_or_else(|e| Err(anyhow::anyhow!("config reload task failed: {e}")))
    };
    match result {
        Ok(report) => (
            StatusCode::OK,
            Json(ConfigReloadResponse {
                added: report.added,
                updated: report.updated,
                skipped_missing_files: report.skipped_missing_files,
                left_untouched: report.left_untouched,
                unchanged: report.unchanged,
                globals_not_applied: report.globals_not_applied,
                models_not_applied: report.models_not_applied,
                applied_live: report.applied_live,
            }),
        )
            .into_response(),
        Err(e) => {
            let msg = format!("{e:#}");
            if msg.contains("config_path not set") {
                return (StatusCode::SERVICE_UNAVAILABLE, msg).into_response();
            }
            if msg.starts_with(crate::model_manager::CONFIG_INVALID_PREFIX) {
                // The edited file is wrong, the server is fine: a 4xx, not a server fault.
                tracing::warn!(error = %e, "admin_reload_config: config.json is invalid");
                return (StatusCode::UNPROCESSABLE_ENTITY, msg).into_response();
            }
            tracing::error!(error = %e, "admin_reload_config failed");
            (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response()
        }
    }
}

// ---- POST /v1/admin/config/preload -----------------------------------

/// Request body for `POST /v1/admin/config/preload`. Exactly one of
/// `model_ids`/`from_live` must be supplied — both or neither is a 400.
#[derive(Deserialize)]
pub struct SetPreloadRequest {
    /// Replace `preload` with exactly this list (order preserved). Every id
    /// must already have a `models` entry.
    #[serde(default)]
    model_ids: Option<Vec<String>>,
    /// Convenience mode: snapshot whichever registered models are currently
    /// `Ready` as the new list (alphabetically sorted). "This is my current
    /// working set, remember it across restarts" in one call.
    #[serde(default)]
    from_live: Option<bool>,
}

/// Response body for `POST /v1/admin/config/preload`.
#[derive(Serialize)]
struct SetPreloadResponse {
    preload: Vec<String>,
    previous: Vec<String>,
    added: Vec<String>,
    removed: Vec<String>,
    source: &'static str,
    /// `from_live` only: registered-but-not-`Ready` model IDs excluded from
    /// the snapshot. Not an error — a preload list only ever describes
    /// future boot state.
    skipped_not_ready: Vec<String>,
}

#[allow(clippy::doc_markdown)]
/// POST /v1/admin/config/preload — replace the persisted `preload` list.
///
/// Never touches the `models` map itself, and — like every config mutation
/// in this file — never triggers a load: it changes what auto-loads on the
/// *next* restart, not what's resident now. Closes the gap where the only
/// way to remember "this loaded set is my known-good default" was hand-
/// editing `config.json`'s `preload` array directly.
///
/// Two mutually exclusive modes: `{"model_ids": [...]}` sets the list
/// explicitly; `{"from_live": true}` snapshots whichever models are
/// currently `Ready`. Either way, every resulting id must already have a
/// `models` entry in the file — an id that doesn't (explicit typo, or a
/// live-`Ready` model whose `add_model` persist previously failed) is
/// rejected rather than written, since it would fail validation on the very
/// next boot.
///
/// - 200 OK — JSON [`SetPreloadResponse`]
/// - 400 Bad Request — both or neither of `model_ids`/`from_live` supplied,
///   a duplicate id in `model_ids`, or any resulting id lacks a `models`
///   entry in the file
/// - 503 Service Unavailable — model manager not initialised, or the server
///   was started without a durable config file
pub async fn admin_set_preload(
    State(state): State<AppState>,
    body: Result<Json<SetPreloadRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    // P-C: bad JSON is the 400 `invalid_body` envelope; a missing `Content-Type` keeps 415.
    let req = match body {
        Ok(Json(req)) => req,
        Err(r) => {
            return super::error::admin_json_rejection(r, super::error::UnknownField::InvalidBody);
        }
    };
    let Some(mm) = state.model_manager.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model manager is not initialised",
        )
            .into_response();
    };

    let source = match (req.model_ids, req.from_live) {
        (Some(ids), None | Some(false)) => crate::model_manager::PreloadSource::Explicit(ids),
        (None, Some(true)) => crate::model_manager::PreloadSource::FromLive,
        (None, None | Some(false)) => {
            return (
                StatusCode::BAD_REQUEST,
                "exactly one of `model_ids` or `from_live: true` must be supplied",
            )
                .into_response();
        }
        (Some(_), Some(true)) => {
            return (
                StatusCode::BAD_REQUEST,
                "`model_ids` and `from_live: true` are mutually exclusive",
            )
                .into_response();
        }
    };
    let source_label = if matches!(source, crate::model_manager::PreloadSource::FromLive) {
        "live"
    } else {
        "explicit"
    };

    match mm.set_preload(source) {
        Ok(report) => (
            StatusCode::OK,
            Json(SetPreloadResponse {
                preload: report.preload,
                previous: report.previous,
                added: report.added,
                removed: report.removed,
                source: source_label,
                skipped_not_ready: report.skipped_not_ready,
            }),
        )
            .into_response(),
        Err(e) => {
            let msg = format!("{e:#}");
            if msg.contains("config_path not set") {
                return (StatusCode::SERVICE_UNAVAILABLE, msg).into_response();
            }
            // See admin_patch_model's identical check: disk I/O / an
            // unparseable config.json are server-side, not the caller's
            // fault.
            if e.downcast_ref::<std::io::Error>().is_some()
                || e.downcast_ref::<serde_json::Error>().is_some()
            {
                tracing::error!(error = %e, "admin_set_preload I/O failure");
                return (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response();
            }
            tracing::error!(error = %e, "admin_set_preload failed");
            (StatusCode::BAD_REQUEST, msg).into_response()
        }
    }
}

/// Response body for `POST /v1/admin/keys/reload`. Counts only — never key
/// values.
#[derive(Serialize)]
#[allow(clippy::struct_excessive_bools)] // independent report flags, one per wire field
struct KeysReloadResponse {
    /// `true` when `api_keys` differed from the file and was hot-swapped in
    /// — no server restart.
    keys_updated: bool,
    /// `true` when `admin_api_keys` differed from the file and was
    /// hot-swapped in.
    admin_keys_updated: bool,
    /// `true` when the file's new `admin_api_keys` was empty while a
    /// non-empty admin scope was live — that specific downgrade was refused,
    /// not applied (`admin_keys_updated` is `false` in this case).
    admin_downgrade_refused: bool,
    /// `true` when no `api_keys` remain and the server binds a network address: inference is
    /// open to anyone who can reach the port (0.9.1 S3).
    inference_open_on_network: bool,
    /// Live inference-key count after this reload.
    api_key_count: usize,
    /// Live admin-key count after this reload.
    admin_key_count: usize,
}

#[allow(clippy::doc_markdown)]
/// POST /v1/admin/keys/reload — pick up keys-file edits without a restart.
///
/// Re-reads the keys file (`ModelManager::reload_keys_file` — resolved once
/// at boot via `config::resolve_keys_file_path`, never repointed by a
/// reload) and hot-swaps `api_keys`/`admin_api_keys` — the same
/// `Arc<ArcSwap<AuthConfig>>` the auth middleware reads on every request, so
/// a rotation is visible to the very next request. All-or-nothing: a
/// missing or malformed file aborts with the live keys completely
/// untouched. Carries no key material in its own request body, and never
/// will — see `ModelManager::reload_keys_file`'s doc comment for the
/// invariant this rests on.
///
/// - 200 OK — JSON [`KeysReloadResponse`] summarising what changed (counts
///   only)
/// - 500 Internal Server Error — keys file missing, unreadable, or fails to
///   parse (`AuthConfig`'s `deny_unknown_fields` included)
/// - 503 Service Unavailable — model manager not initialised, or no
///   keys-file path wired (server started without a durable config file)
pub async fn admin_reload_keys(State(state): State<AppState>) -> Response {
    let Some(mm) = state.model_manager.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model manager is not initialised",
        )
            .into_response();
    };

    match mm.reload_keys_file() {
        Ok(report) => (
            StatusCode::OK,
            Json(KeysReloadResponse {
                keys_updated: report.keys_updated,
                admin_keys_updated: report.admin_keys_updated,
                admin_downgrade_refused: report.admin_downgrade_refused,
                inference_open_on_network: report.inference_open_on_network,
                api_key_count: report.api_key_count,
                admin_key_count: report.admin_key_count,
            }),
        )
            .into_response(),
        Err(e) => {
            let msg = format!("{e:#}");
            if msg.contains("keys_file_path") || msg.contains("no auth handle wired") {
                return (StatusCode::SERVICE_UNAVAILABLE, msg).into_response();
            }
            tracing::error!(error = %e, "admin_reload_keys failed");
            (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response()
        }
    }
}

/// One problem entry in [`ConfigAuditResponse`].
#[derive(Serialize)]
struct ConfigAuditEntryResponse {
    /// The model ID from `config.json`'s `models` map.
    model_id: String,
    /// The directory that should exist under `models_dir` for this entry.
    expected_path: String,
    /// `true` if `expected_path` does not exist — this entry can never load.
    files_missing: bool,
    /// `true` if this `model_id` also appears in `config.preload` — a missing
    /// directory here is fatal at next startup, not just unloadable.
    in_preload: bool,
    /// `true` if the entry exists on disk but isn't in the live registry yet
    /// — resolved by calling `POST /v1/admin/config/reload`.
    pending_reload: bool,
}

/// Response body for `GET /v1/admin/config/audit`.
#[derive(Serialize)]
struct ConfigAuditResponse {
    /// Total `model_id` entries declared in `config.json`, healthy or not.
    total_declared: usize,
    /// Entries with at least one problem — a healthy entry is omitted.
    problems: Vec<ConfigAuditEntryResponse>,
}

#[allow(clippy::doc_markdown)]
/// GET /v1/admin/config/audit — report `config.json` drift without changing
/// anything.
///
/// Re-reads the file fresh on every call and cross-checks it against the
/// live registry and the real filesystem: which declared models have no
/// files under `models_dir` (the stale/typo'd-`model_id` class of bug this
/// project has repeatedly hit by hand — see the project's internal engineering log), which of
/// those are also in `preload` (fatal at next restart, not just unloadable),
/// and which exist on disk but haven't been picked up by a reload yet.
/// Purely read-only — never mutates config or the registry.
///
/// - 200 OK — JSON [`ConfigAuditResponse`]
/// - 500 Internal Server Error — file unreadable or invalid
/// - 503 Service Unavailable — model manager not initialised, or the server
///   was started without a durable config file (`config_path` unset)
pub async fn admin_audit_config(State(state): State<AppState>) -> Response {
    let Some(mm) = state.model_manager.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model manager is not initialised",
        )
            .into_response();
    };

    match mm.audit_config() {
        Ok(report) => (
            StatusCode::OK,
            Json(ConfigAuditResponse {
                total_declared: report.total_declared,
                problems: report
                    .problems
                    .into_iter()
                    .map(|p| ConfigAuditEntryResponse {
                        model_id: p.model_id,
                        expected_path: p.expected_path,
                        files_missing: p.files_missing,
                        in_preload: p.in_preload,
                        pending_reload: p.pending_reload,
                    })
                    .collect(),
            }),
        )
            .into_response(),
        Err(e) => {
            let msg = format!("{e:#}");
            if msg.contains("config_path not set") {
                return (StatusCode::SERVICE_UNAVAILABLE, msg).into_response();
            }
            tracing::error!(error = %e, "admin_audit_config failed");
            (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response()
        }
    }
}

/// Query params for `GET /v1/admin/logs/tail`.
#[derive(Deserialize)]
pub struct LogsTailQuery {
    /// Number of recent lines to return, oldest first. Default 200; silently
    /// clamped to however many lines the buffer actually holds (it never
    /// holds more than [`crate::log_ring`]'s fixed capacity).
    lines: Option<usize>,
}

/// `GET /v1/admin/logs/tail?lines=N` — the last `N` recent log lines (default
/// 200), oldest first, one per line as `text/plain`. Lets an operator (or a
/// UI, e.g. a Pyramu panel) view recent server activity over HTTP without
/// shelling in and finding wherever stdout happened to be redirected.
pub async fn admin_logs_tail(
    State(state): State<AppState>,
    Query(params): Query<LogsTailQuery>,
) -> Response {
    let n = params.lines.unwrap_or(200);
    let body = state.log_buffer.tail(n).join("\n");
    (
        StatusCode::OK,
        [("content-type", "text/plain; charset=utf-8")],
        body,
    )
        .into_response()
}

// ============================================================
// Unit tests
// ============================================================
/// An admin route refused on a poisoned GPU context (plain-text body, as `rv` prints it). G1
/// (step 7): while a supervised worker is on its way to a restart, "retry shortly" +
/// `Retry-After: 30`; otherwise the operator has to restart it.
fn gpu_poisoned_admin(e: &impl std::fmt::Display) -> Response {
    if crate::startup::gpu_poison_exit_requested() {
        let mut resp = (
            StatusCode::SERVICE_UNAVAILABLE,
            format!("{e} — the worker is restarting (supervised) — retry shortly"),
        )
            .into_response();
        resp.headers_mut().insert(
            axum::http::header::RETRY_AFTER,
            super::error::RETRY_AFTER_WORKER_RESTART,
        );
        return resp;
    }
    (
        StatusCode::SERVICE_UNAVAILABLE,
        format!(
            "{e} — restart rustedvino to recover (a graceful stop via SIGTERM/systemctl drains \
             in-flight requests and frees VRAM first)"
        ),
    )
        .into_response()
}

/// The response for a refusal caused by the eviction policy (D3): `409` when an operator's
/// eviction of a busy model was refused ([`ModelError::TargetBusy`]); `503` when a load was
/// refused because the only room is held by busy models ([`ModelError::BlockedByBusy`], code
/// `model_busy`) or because a forced eviction's engine is still finishing its request and its
/// memory is not free yet ([`ModelError::EvictionIncomplete`], code `eviction_incomplete`).
/// All use the same JSON error envelope as the inference routes and a `Retry-After`: the cause
/// resolves when the in-flight work ends.
fn eviction_refusal(e: &ModelError) -> Response {
    let (status, err_type, code) = match e {
        ModelError::TargetBusy(_) => (StatusCode::CONFLICT, "invalid_request_error", "model_busy"),
        ModelError::EvictionIncomplete(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "eviction_incomplete",
        ),
        _ => (
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "model_busy",
        ),
    };
    let mut resp = super::error::openai_error(status, format!("{e:#}"), err_type, Some(code));
    resp.headers_mut().insert(
        axum::http::header::RETRY_AFTER,
        super::error::RETRY_AFTER_TRANSIENT,
    );
    resp
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::model_manager::KvClamp;

    /// docs-drift L2: the catch-all collects exactly the fields nothing else claims. Every policy
    /// field (the whole `--print-default-model` stanza, so a new field is covered too) still
    /// reaches `policy`, none of them lands in `unknown`, and a typo does.
    #[test]
    fn add_request_collects_only_unknown_fields() {
        let mut body = crate::model_manager::Config::default_model_template().unwrap();
        let map = body.as_object_mut().unwrap();
        map.insert("model_id".to_owned(), serde_json::json!("m"));
        map.insert("vram_gb".to_owned(), serde_json::json!(1.5));
        map.insert("pinned".to_owned(), serde_json::json!(true));
        map.insert("pinnned".to_owned(), serde_json::json!(true));
        map.insert("bogus".to_owned(), serde_json::json!({"x": 1}));
        let req: AddModelRequest = serde_json::from_value(body).unwrap();
        assert_eq!(
            req.unknown.keys().map(String::as_str).collect::<Vec<_>>(),
            ["bogus", "pinnned"]
        );
        assert!(req.policy.pinned, "a known field still reaches the policy");
        assert!((req.vram_gb - 1.5).abs() < 1e-9);

        let plain: AddModelRequest =
            serde_json::from_str(r#"{"model_id": "m", "vram_gb": 0.0, "kind": "embedding"}"#)
                .unwrap();
        assert!(plain.unknown.is_empty());
    }

    /// D-04 S9 review #13b: the watchdog's stalled entry carries `recovery`, `null` while the
    /// load is inside its deadline; `rv status` reads exactly these keys.
    #[test]
    fn watchdog_stalled_entry_shape() {
        let entry = |recovery: Option<&str>| {
            serde_json::to_value(WatchdogStalled {
                device: "GPU.1".to_owned(),
                model: "m".to_owned(),
                kind: "load",
                secs: 7,
                recovery: recovery.map(str::to_owned),
            })
            .unwrap()
        };
        assert_eq!(
            entry(None),
            serde_json::json!({"device": "GPU.1", "model": "m", "kind": "load", "secs": 7,
                               "recovery": null})
        );
        assert_eq!(entry(Some("reboot"))["recovery"], "reboot");
    }

    /// The resize sentence: an exact pool says what was built (not a false clamp), a clamped one
    /// names the setting, an untouched whole-GiB pool stays terse.
    #[test]
    fn resize_message_covers_exact_clamped_and_plain_pools() {
        // Exact pool, a hair under the request: no clamp talk.
        let m = resize_message("m", 3.999_98, 4.0, Some(3_192), None);
        assert!(
            m.contains("(3192 KV blocks)") && !m.contains("clamped"),
            "{m}"
        );
        // Exact pool, visibly different (the budget was smaller than asked) and not clamped by
        // a named setting: the request is shown next to what was built.
        let m = resize_message("m", 2.5, 4.0, Some(2_000), None);
        assert!(m.contains("2000 KV blocks; requested 4"), "{m}");
        // Clamped exact pool: the setting is named.
        let m = resize_message("m", 2.0, 4.0, Some(1_600), Some(KvClamp::CacheSizeGb));
        assert!(m.contains("clamped by"), "{m}");
        // Whole-GiB pool, as asked.
        assert_eq!(
            resize_message("m", 4.0, 4.0, None, None),
            "model 'm' resized to 4.00 GiB"
        );
    }

    /// D3-S1.B4/E5: eviction refusals use the JSON envelope with a `Retry-After`: 409
    /// `model_busy` for an eviction of a busy model, 503 `model_busy` for a load blocked by busy
    /// models, 503 `eviction_incomplete` for a forced eviction whose engine is still running.
    #[tokio::test]
    async fn eviction_refusals_use_the_json_envelope() {
        for (e, status, err_type, code) in [
            (
                ModelError::TargetBusy(3),
                StatusCode::CONFLICT,
                "invalid_request_error",
                "model_busy",
            ),
            (
                ModelError::BlockedByBusy(vec!["bielik".to_owned()]),
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "model_busy",
            ),
            (
                ModelError::EvictionIncomplete("whisper".to_owned()),
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "eviction_incomplete",
            ),
        ] {
            let text = format!("{e:#}");
            let resp = eviction_refusal(&e);
            assert_eq!(resp.status(), status);
            assert_eq!(
                resp.headers().get(axum::http::header::RETRY_AFTER).unwrap(),
                "5"
            );
            let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(json["error"]["code"], code);
            assert_eq!(json["error"]["type"], err_type);
            assert_eq!(json["error"]["message"], text.as_str());
        }
    }

    /// S281 defect (a): an already-invalid config.json answers 409 with its own code, so `rv`
    /// does not read it as a bare 409 ("already registered — use rv load").
    #[tokio::test]
    async fn config_file_invalid_is_a_409_with_its_own_code() {
        let e = ModelError::ConfigFileInvalid("config.json (/x) is already invalid: boom".into());
        let resp = super::super::error::config_file_invalid_response(&e);
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "config_file_invalid");
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("not caused by this request")
        );
    }

    /// G1 (step 7): an unsupervised (or not-restarting) worker keeps telling the operator to restart;
    /// no `Retry-After`. (The restarting branch reads a process-global flag set only under
    /// `--supervise`; it is covered live, Test 10.)
    #[test]
    fn gpu_poisoned_admin_without_a_pending_restart_says_restart() {
        let resp = gpu_poisoned_admin(&"GPU context poisoned (out of resources)");
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(
            resp.headers()
                .get(axum::http::header::RETRY_AFTER)
                .is_none()
        );
    }

    /// `HealthResponse` serialises to the exact JSON shape clients expect.
    #[test]
    fn health_response_serialises_correctly() {
        let response = HealthResponse {
            status: "ok",
            version: "1.2.3",
            git_hash: "abc123def456",
            gpu_poisoned: false,
            device_stalled: false,
            startup_warnings: HealthStartupWarnings {
                count: 2,
                codes: vec!["preload_skipped"],
            },
        };
        let json = serde_json::to_value(response).unwrap();
        assert_eq!(json["status"], "ok");
        assert_eq!(json["version"], "1.2.3");
        assert_eq!(json["git_hash"], "abc123def456");
        assert_eq!(json["gpu_poisoned"], false);
        assert_eq!(json["device_stalled"], false);
        // m1: count + codes only, no message text (`/health` is unauthenticated).
        assert_eq!(
            json["startup_warnings"],
            serde_json::json!({"count": 2, "codes": ["preload_skipped"]})
        );
        assert_eq!(
            json.as_object().unwrap().len(),
            6,
            "HealthResponse must have exactly 6 fields"
        );
    }

    /// `AdminModelEntry` exposes `max_prompt_tokens`/`cache_size_gb` so an
    /// external client/operator can discover a model's real usable context
    /// window proactively, instead of only reactively via a `400
    /// context_length_exceeded`
    /// (the project's internal engineering log).
    #[test]
    fn admin_model_entry_exposes_context_ceiling_fields() {
        let entry = AdminModelEntry {
            id: "qwen3-4b-int4-ov".to_owned(),
            state: "ready",
            kind: "text_gen",
            vram_gb: 2.5,
            device: "GPU.1".to_owned(),
            default_temperature: None,
            default_top_p: None,
            default_top_k: None,
            sampling: None,
            thinking_budget_tokens: None,
            max_prompt_tokens: 37_137,
            cache_size_gb: 3.0,
            kv_requested_gb: None,
            clamped_by: None,
            native_context_limit: Some(40_960),
            kv_cache_usage_pct: 0.0,
            kv_cache_usage_supported: false,
            kv_pressure_flagged: false,
            gpu_memory_estimate_gb: None,
            kv_cache_precision: None,
            kv_precision_source: None,
            kv_bytes_per_token: None,
            kv_profile_confidence: None,
            kv_gate_mode: "",
            limit_clamped_by: None,
            kv_pool_mib: 0,
            kv_pool_blocks: None,
            kv_headroom_cap_tokens: None,
            vram_measured_mib: None,
            kv_pool_resident_mib: None,
            kv_admission_low_confidence: false,
            uncounted_image_admissions: 0,
        };
        let json = serde_json::to_value(entry).unwrap();
        assert_eq!(json["max_prompt_tokens"], 37_137);
        assert!((json["cache_size_gb"].as_f64().unwrap() - 3.0).abs() < f64::EPSILON);
        assert_eq!(json["native_context_limit"], 40_960);
        // C21/P1: the flag that says whether `kv_cache_usage_pct` is a live reading, always sent.
        assert_eq!(json["kv_cache_usage_supported"], false);
    }

    /// The KV decision fields: present when known, omitted (never `null`) when not, and the pool
    /// is always shown in MiB.
    #[test]
    fn admin_model_entry_shows_the_kv_decision_and_omits_unknowns() {
        let mut entry = AdminModelEntry {
            id: "qwen3-8b".to_owned(),
            state: "ready",
            kind: "text_gen",
            vram_gb: 5.5,
            device: "GPU.1".to_owned(),
            default_temperature: None,
            default_top_p: None,
            default_top_k: None,
            sampling: None,
            thinking_budget_tokens: None,
            max_prompt_tokens: 86_381,
            cache_size_gb: 3.9999,
            kv_requested_gb: None,
            clamped_by: None,
            native_context_limit: None,
            kv_cache_usage_pct: 0.0,
            kv_cache_usage_supported: false,
            kv_pressure_flagged: false,
            gpu_memory_estimate_gb: None,
            kv_cache_precision: Some("u4"),
            kv_precision_source: Some("auto_rule"),
            kv_bytes_per_token: Some(47_232),
            kv_profile_confidence: Some("high"),
            kv_gate_mode: "exact",
            limit_clamped_by: Some("pool"),
            kv_pool_mib: mib_of_gib(3.9999),
            kv_pool_blocks: Some(5_683),
            kv_headroom_cap_tokens: None,
            vram_measured_mib: None,
            kv_pool_resident_mib: None,
            kv_admission_low_confidence: false,
            uncounted_image_admissions: 0,
        };
        let json = serde_json::to_value(&entry).unwrap();
        assert_eq!(json["kv_cache_precision"], "u4");
        assert_eq!(json["kv_precision_source"], "auto_rule");
        assert_eq!(json["kv_bytes_per_token"], 47_232);
        assert_eq!(json["kv_profile_confidence"], "high");
        assert_eq!(json["kv_gate_mode"], "exact");
        assert_eq!(json["limit_clamped_by"], "pool");
        assert_eq!(json["kv_pool_mib"], 4096);
        assert_eq!(json["kv_pool_blocks"], 5_683);
        assert!(json.get("kv_headroom_cap_tokens").is_none());
        // Not loaded: everything optional disappears, the pool reads 0.
        entry.kv_cache_precision = None;
        entry.kv_precision_source = None;
        entry.kv_bytes_per_token = None;
        entry.kv_profile_confidence = None;
        entry.kv_gate_mode = "";
        entry.limit_clamped_by = None;
        entry.kv_pool_mib = 0;
        entry.kv_pool_blocks = None;
        let json = serde_json::to_value(&entry).unwrap();
        let obj = json.as_object().unwrap();
        for k in [
            "kv_cache_precision",
            "kv_precision_source",
            "kv_bytes_per_token",
            "kv_profile_confidence",
            "kv_gate_mode",
            "limit_clamped_by",
            "kv_pool_blocks",
        ] {
            assert!(!obj.contains_key(k), "{k} must be omitted: {json}");
        }
        assert_eq!(json["kv_pool_mib"], 0);
    }

    /// `native_context_limit` is omitted entirely (not `null`) when the
    /// model's `config.json` doesn't declare `max_position_embeddings` — an
    /// unknown ceiling, not a guessed one, same convention as
    /// `default_temperature`/`default_top_p`/`default_top_k`.
    #[test]
    fn admin_model_entry_omits_native_context_limit_when_unknown() {
        let entry = AdminModelEntry {
            id: "some-model".to_owned(),
            state: "not_loaded",
            kind: "text_gen",
            vram_gb: 2.5,
            device: "GPU.1".to_owned(),
            default_temperature: None,
            default_top_p: None,
            default_top_k: None,
            sampling: None,
            thinking_budget_tokens: None,
            max_prompt_tokens: 0,
            cache_size_gb: 0.0,
            kv_requested_gb: None,
            clamped_by: None,
            native_context_limit: None,
            kv_cache_usage_pct: 0.0,
            kv_cache_usage_supported: false,
            kv_pressure_flagged: false,
            gpu_memory_estimate_gb: None,
            kv_cache_precision: None,
            kv_precision_source: None,
            kv_bytes_per_token: None,
            kv_profile_confidence: None,
            kv_gate_mode: "",
            limit_clamped_by: None,
            kv_pool_mib: 0,
            kv_pool_blocks: None,
            kv_headroom_cap_tokens: None,
            vram_measured_mib: None,
            kv_pool_resident_mib: None,
            kv_admission_low_confidence: false,
            uncounted_image_admissions: 0,
        };
        let json = serde_json::to_value(entry).unwrap();
        assert!(
            json.as_object()
                .unwrap()
                .get("native_context_limit")
                .is_none(),
            "native_context_limit must be omitted, not null, when unknown: {json:?}"
        );
    }

    /// `ServerInfo` always carries `engine`/`engine_version`; `host`/
    /// `openvino_version` are individually omitted (never `null`) when
    /// unavailable — same "omit, don't fabricate" contract
    /// `generation_metadata` uses.
    /// An open (keyless) server's `server` block still carries the headroom report next to
    /// `gpu_poisoned`, while host, pid and uptime stay withheld (`ServerInfo::current(false)`).
    #[test]
    fn keyless_server_block_has_headroom_but_no_identity() {
        let mut info = ServerInfo::current(false);
        info.headroom = Some(crate::model_manager::headroom::HeadroomReport {
            mode: "dynamic".to_owned(),
            probe: "fdinfo",
            margin_gib: 0.5,
            devices: Vec::new(),
        });
        let v = serde_json::to_value(info).unwrap();
        assert_eq!(v["headroom"]["mode"], "dynamic");
        assert_eq!(v["headroom"]["probe"], "fdinfo");
        assert!(v.get("host").is_none() && v.get("pid").is_none());
        assert!(v.get("uptime_secs").is_none());
    }

    #[test]
    fn server_info_omits_unavailable_optional_fields() {
        let info = ServerInfo {
            host: None,
            pid: None,
            uptime_secs: None,
            openvino_version: None,
            engine: "RustedVINO",
            engine_version: "1.2.3",
            gpu_poisoned: false,
            headroom: None,
            startup_warnings: Vec::new(),
            platform: None,
        };
        let value = serde_json::to_value(info).unwrap();
        let obj = value.as_object().unwrap();
        let keys: std::collections::BTreeSet<&str> =
            obj.keys().map(std::string::String::as_str).collect();
        assert_eq!(
            keys,
            std::collections::BTreeSet::from(["engine", "engine_version", "gpu_poisoned"]),
            "host/openvino_version must be absent when None, never null; gpu_poisoned is always present: {obj:?}"
        );
    }

    /// `ServerInfo::current()` on the real running test box: non-empty engine
    /// version always present; `host`/`openvino_version` are whatever this
    /// environment actually reports (both `Some` on the fleet, both plausibly
    /// `None` in a minimal sandbox) — the point is it never panics and always
    /// produces valid JSON either way.
    #[test]
    fn server_info_current_serializes_without_panicking() {
        let value = serde_json::to_value(ServerInfo::current(true)).unwrap();
        assert_eq!(value["engine"], "RustedVINO");
        assert_eq!(value["engine_version"], env!("CARGO_PKG_VERSION"));
    }

    /// Without a key-validated caller the admin `server` block omits `host`.
    #[test]
    fn server_info_without_auth_omits_host() {
        let value = serde_json::to_value(ServerInfo::current(false)).unwrap();
        assert!(value.get("host").is_none(), "{value}");
        assert!(value.get("pid").is_none(), "{value}");
        assert!(value.get("uptime_secs").is_none(), "{value}");
        assert!(value.get("startup_warnings").is_none(), "{value}");
        assert!(value.get("platform").is_none(), "{value}");
        assert_eq!(value["engine"], "RustedVINO");
    }

    /// A key-validated caller gets the process's own PID, and `uptime_secs`
    /// once startup has marked the process start.
    #[test]
    fn server_info_with_auth_reports_pid_and_uptime() {
        crate::startup::mark_process_start();
        let value = serde_json::to_value(ServerInfo::current(true)).unwrap();
        assert_eq!(value["pid"], std::process::id());
        assert!(value["uptime_secs"].is_u64(), "{value}");
    }

    /// `ModelsResponse` with one entry serialises to the `OpenAI` envelope shape.
    #[test]
    fn models_response_serialises_correctly() {
        let response = ModelsResponse {
            object: "list",
            data: vec![ModelObject {
                id: "qwen3-8b-int4-ov".to_owned(),
                object: "model",
                created: 1_700_000_000,
                owned_by: "rustedvino",
                max_model_len: Some(37_137),
                state: None,
            }],
        };
        let json = serde_json::to_value(response).unwrap();
        assert_eq!(json["object"], "list");
        let data = json["data"].as_array().unwrap();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0]["id"], "qwen3-8b-int4-ov");
        assert_eq!(data[0]["object"], "model");
        assert_eq!(
            data[0]["created"], 1_700_000_000,
            "OpenAI model object must carry a numeric `created` timestamp"
        );
        assert_eq!(data[0]["owned_by"], "rustedvino");
        assert_eq!(data[0]["max_model_len"], 37_137);
    }

    /// `max_model_len` is omitted entirely (not `0`) when the ceiling is
    /// unknown — same "omit, don't fabricate" convention as
    /// `native_context_limit`/`default_temperature`.
    #[test]
    fn model_object_omits_max_model_len_when_unknown() {
        let obj = ModelObject {
            id: "some-model".to_owned(),
            object: "model",
            created: 1_700_000_000,
            owned_by: "rustedvino",
            max_model_len: None,
            state: None,
        };
        let json = serde_json::to_value(obj).unwrap();
        assert!(
            json.as_object().unwrap().get("max_model_len").is_none(),
            "max_model_len must be omitted, not 0, when unknown: {json:?}"
        );
    }

    /// `HealthGenerateResponse` serialises to the expected JSON shape.
    #[test]
    fn health_generate_response_serialises() {
        let r = HealthGenerateResponse {
            status: "ok",
            model: "qwen3-8b".to_owned(),
            elapsed_ms: 42,
        };
        let v = serde_json::to_value(r).unwrap();
        assert_eq!(v["status"], "ok");
        assert_eq!(v["model"], "qwen3-8b");
        assert_eq!(v["elapsed_ms"], 42);
    }

    /// Synthetic `ModelInfo` for `select_probe_target` tests — only `id`/
    /// `state`/`kind` matter for the selection logic under test.
    fn mi(
        id: &str,
        state: crate::model_manager::ModelState,
        kind: crate::model_manager::ModelKind,
    ) -> crate::model_manager::ModelInfo {
        crate::model_manager::ModelInfo {
            id: id.to_owned(),
            state,
            stalled_secs: None,
            generation_stalled: false,
            kind,
            vram_gb: 0.0,
            device: "CPU".to_owned(),
            generation_defaults: crate::model_manager::template::GenerationDefaults::default(),
            sampling: None,
            thinking_budget_tokens: None,
            max_prompt_tokens: 0,
            kv_cache_gb: 0.0,
            kv_requested_gb: None,
            kv_clamped_by: None,
            kv_pool_blocks: None,
            kv_resolution: None,
            kv_headroom_cap_tokens: None,
            vram_measured_mib: None,
            kv_pool_resident_mib: None,
            kv_gate_mode: "",
            kv_gate_clamped_by: None,
            native_context_limit: None,
            kv_cache_usage_pct: 0.0,
            kv_cache_usage_supported: false,
            kv_pressure_flagged: false,
            gpu_memory_estimate_gb: None,
            uncounted_image_admissions: 0,
        }
    }

    /// A Ready `TextGen`/`Vision` model is preferred over a Ready model of a
    /// kind the probe can't drive, regardless of registration order —
    /// closes the bug a stress-test scenario caught live: picking whichever model
    /// happened to be first in unspecified `HashMap` order reported a
    /// blanket 503 whenever that was an Embedding/STT/TTS/ImageGen/
    /// Reranking model, even with a perfectly healthy chat model sitting
    /// right next to it.
    #[test]
    fn select_probe_target_prefers_a_probeable_kind_over_first_in_list() {
        use crate::model_manager::{ModelKind, ModelState};
        let models = vec![
            mi("embed-1", ModelState::Ready, ModelKind::Embedding),
            mi("stt-1", ModelState::Ready, ModelKind::Stt),
            mi("chat-1", ModelState::Ready, ModelKind::TextGen),
            mi("tts-1", ModelState::Ready, ModelKind::Tts),
        ];
        assert_eq!(
            select_probe_target(&models),
            Some("chat-1".to_owned()),
            "a Ready TextGen model must win even though it's third in the list"
        );
    }

    /// A `Vision` model is an equally valid probe target as `TextGen` — the
    /// first Ready one of either kind wins, in list order between them.
    #[test]
    fn select_probe_target_accepts_vision_too() {
        use crate::model_manager::{ModelKind, ModelState};
        let models = vec![
            mi("embed-1", ModelState::Ready, ModelKind::Embedding),
            mi("vlm-1", ModelState::Ready, ModelKind::Vision),
        ];
        assert_eq!(select_probe_target(&models), Some("vlm-1".to_owned()));
    }

    /// A `NotLoaded`/`Loading`/`Evicting` `TextGen` model is not a candidate
    /// — only `Ready` counts, exactly like the pre-fix behavior.
    #[test]
    fn select_probe_target_ignores_non_ready_states() {
        use crate::model_manager::{ModelKind, ModelState};
        let models = vec![
            mi("chat-loading", ModelState::Loading, ModelKind::TextGen),
            mi("embed-ready", ModelState::Ready, ModelKind::Embedding),
        ];
        assert_eq!(
            select_probe_target(&models),
            Some("embed-ready".to_owned()),
            "a non-Ready probeable model must not be picked over a Ready unprobeable one"
        );
    }

    /// When no `TextGen`/`Vision` model is Ready at all, falls back to any
    /// Ready model — preserving the honest "not implemented for X models"
    /// 503 (rather than a misleading "no ready models") for a box that
    /// genuinely has nothing the deep probe can drive.
    #[test]
    fn select_probe_target_falls_back_when_nothing_probeable_is_ready() {
        use crate::model_manager::{ModelKind, ModelState};
        let models = vec![
            mi("embed-1", ModelState::Ready, ModelKind::Embedding),
            mi("chat-notloaded", ModelState::NotLoaded, ModelKind::TextGen),
        ];
        assert_eq!(select_probe_target(&models), Some("embed-1".to_owned()));
    }

    /// No Ready model at all → `None`, the "no ready models" 503 case.
    #[test]
    fn select_probe_target_none_when_nothing_ready() {
        use crate::model_manager::{ModelKind, ModelState};
        let models = vec![mi("chat-1", ModelState::NotLoaded, ModelKind::TextGen)];
        assert_eq!(select_probe_target(&models), None);
    }

    /// The Vision health-probe image is a well-formed NHWC RGB buffer of the
    /// expected dimensions — `edge × edge × 3` bytes, so the VLM bridge can wrap
    /// it as `ov::Tensor({1, edge, edge, 3})` without a size mismatch.
    #[test]
    fn probe_image_is_well_formed_nhwc_rgb() {
        let img = probe_image();
        assert_eq!(img.width, PROBE_IMAGE_EDGE);
        assert_eq!(img.height, PROBE_IMAGE_EDGE);
        assert_eq!(
            img.data.len(),
            (PROBE_IMAGE_EDGE * PROBE_IMAGE_EDGE * 3) as usize,
            "NHWC RGB buffer must be height × width × 3 bytes"
        );
        // Solid blue: every pixel is (0, 0, 255).
        assert!(
            img.data
                .as_chunks::<3>()
                .0
                .iter()
                .all(|px| *px == [0, 0, 255]),
            "probe image must be uniform blue"
        );
    }

    /// `AdmissionCheckResponse` serialises to the documented "what fits" shape.
    #[test]
    fn admission_check_response_serialises() {
        let r = AdmissionCheckResponse {
            model: "qwen3-8b".to_owned(),
            fits: true,
            needed_gb: 6.5,
            free_gb: 3.0,
            vram_gated: true,
            would_evict: vec!["embed".to_owned()],
            busy_victims: Vec::new(),
            blocked_by_busy: Vec::new(),
            ram_needed_gb: None,
            ram_avail_gb: None,
            already_loaded: false,
            headroom_refusal: None,
            headroom_not_previewed: None,
        };
        let v = serde_json::to_value(r).unwrap();
        assert_eq!(v["model"], "qwen3-8b");
        assert_eq!(v["fits"], true);
        assert_eq!(v["needed_gb"], 6.5);
        assert_eq!(v["free_gb"], 3.0);
        assert_eq!(v["would_evict"], serde_json::json!(["embed"]));
        assert_eq!(v["already_loaded"], false);
        // Discrete-GPU shape: the RAM-gate keys are omitted entirely rather
        // than serialised as null, so an older `rv` sees the exact bytes it
        // saw before this field pair existed.
        assert!(v.get("ram_needed_gb").is_none());
        assert!(v.get("ram_avail_gb").is_none());
        // F1: the headroom keys appear only when they say something.
        assert!(v.get("headroom_refusal").is_none());
        assert!(v.get("headroom_not_previewed").is_none());
    }

    /// F1/A2: a headroom refusal and an unpreviewed headroom gate are both reported.
    #[test]
    fn admission_check_response_reports_the_headroom_gate() {
        let r = AdmissionCheckResponse {
            model: "qwen3.6-27b-int4-ov".to_owned(),
            fits: false,
            needed_gb: 19.7,
            free_gb: 2.5,
            vram_gated: true,
            would_evict: Vec::new(),
            busy_victims: Vec::new(),
            blocked_by_busy: Vec::new(),
            ram_needed_gb: None,
            ram_avail_gb: None,
            already_loaded: false,
            headroom_refusal: Some("the device headroom leaves room for 0.91 GiB".to_owned()),
            headroom_not_previewed: Some("not loaded here yet".to_owned()),
        };
        let v = serde_json::to_value(r).unwrap();
        assert_eq!(
            v["headroom_refusal"],
            "the device headroom leaves room for 0.91 GiB"
        );
        assert_eq!(v["headroom_not_previewed"], "not loaded here yet");
    }

    /// UMA shape: when the RAM gate applies, both keys are present.
    #[test]
    fn admission_check_response_includes_ram_gate_on_uma() {
        let r = AdmissionCheckResponse {
            model: "qwen3.6-35b-a3b-int4-ov".to_owned(),
            fits: false,
            needed_gb: 22.0,
            free_gb: 27.0,
            vram_gated: true,
            would_evict: Vec::new(),
            busy_victims: Vec::new(),
            blocked_by_busy: Vec::new(),
            ram_needed_gb: Some(24.0),
            ram_avail_gb: Some(23.0),
            already_loaded: false,
            headroom_refusal: None,
            headroom_not_previewed: None,
        };
        let v = serde_json::to_value(r).unwrap();
        assert_eq!(v["ram_needed_gb"], 24.0);
        assert_eq!(v["ram_avail_gb"], 23.0);
        // The shape measured: logical VRAM says yes, the RAM gate says no.
        assert_eq!(v["fits"], false);
    }

    /// `server_created_ts()` is stable across calls and a plausible recent time.
    #[test]
    fn server_created_ts_is_stable_and_positive() {
        let a = server_created_ts();
        let b = server_created_ts();
        assert_eq!(a, b, "server_created_ts must be captured once and stable");
        assert!(a > 1_600_000_000, "timestamp must be a recent Unix time");
    }

    // ---- LoadRequest validation (Slice 2b) -----------------------------

    /// A full, valid body maps straight through to `LoadOverrides`.
    #[test]
    fn load_request_valid_both_fields() {
        let req: LoadRequest =
            serde_json::from_str(r#"{"kv_cache_gb": 4.0, "max_concurrent_streams": 8}"#).unwrap();
        let ov = req.into_overrides().unwrap();
        assert_eq!(ov.kv_cache_gb, Some(4.0));
        assert_eq!(ov.max_concurrent_streams, Some(8));
    }

    /// An empty JSON object is valid and yields all-`None` (use config).
    #[test]
    fn load_request_empty_object_is_no_overrides() {
        let req: LoadRequest = serde_json::from_str("{}").unwrap();
        let ov = req.into_overrides().unwrap();
        assert_eq!(ov.kv_cache_gb, None);
        assert_eq!(ov.max_concurrent_streams, None);
    }

    /// Each field can be overridden independently.
    #[test]
    fn load_request_partial_fields() {
        let kv: LoadRequest = serde_json::from_str(r#"{"kv_cache_gb": 2.5}"#).unwrap();
        let ov = kv.into_overrides().unwrap();
        assert_eq!(ov.kv_cache_gb, Some(2.5));
        assert_eq!(ov.max_concurrent_streams, None);

        let streams: LoadRequest =
            serde_json::from_str(r#"{"max_concurrent_streams": 4}"#).unwrap();
        let ov = streams.into_overrides().unwrap();
        assert_eq!(ov.kv_cache_gb, None);
        assert_eq!(ov.max_concurrent_streams, Some(4));
    }

    /// `kv_cache_gb <= 0` is rejected (omit the field for an unbounded pool).
    #[test]
    fn load_request_rejects_nonpositive_kv() {
        for body in [r#"{"kv_cache_gb": 0.0}"#, r#"{"kv_cache_gb": -1.0}"#] {
            let req: LoadRequest = serde_json::from_str(body).unwrap();
            assert!(
                req.into_overrides().is_err(),
                "kv_cache_gb must be > 0 ({body})"
            );
        }
    }

    /// `max_concurrent_streams: 0` is rejected (must be >= 1).
    #[test]
    fn load_request_rejects_zero_streams() {
        let req: LoadRequest = serde_json::from_str(r#"{"max_concurrent_streams": 0}"#).unwrap();
        assert!(req.into_overrides().is_err());
    }

    /// A negative stream count fails at deserialisation (usize), not validation.
    #[test]
    fn load_request_negative_streams_fails_to_parse() {
        let parsed: Result<LoadRequest, _> =
            serde_json::from_str(r#"{"max_concurrent_streams": -2}"#);
        assert!(parsed.is_err(), "negative usize must fail to deserialise");
    }

    /// An unknown field is rejected so typos surface as 400 rather than being
    /// silently dropped.
    #[test]
    fn load_request_rejects_unknown_field() {
        let parsed: Result<LoadRequest, _> = serde_json::from_str(r#"{"kv_cache": 4.0}"#);
        assert!(parsed.is_err(), "unknown field must be rejected");
    }

    // ---- D3-S1.F6: the `force` parameter of the eviction routes (HTTP level) ---------------

    /// A router over a mock manager with one loaded, BUSY text model (`busy-a`): the returned
    /// permit stands for a request that is running on it.
    async fn busy_router() -> (
        axum::Router,
        tokio::sync::OwnedSemaphorePermit,
        std::sync::Arc<crate::model_manager::ModelManager>,
    ) {
        use crate::model_manager::lifecycle::MockEngineFactory;
        let cfg: crate::model_manager::Config = serde_json::from_value(serde_json::json!({
            "models_dir": "/tmp/test-admin-force-models", "device": "CPU", "total_vram_gb": 22.5,
            "eviction_grace_secs": 0.0,
            "models": { "busy-a": {"vram_gb": 2.0} }
        }))
        .unwrap();
        let mm = std::sync::Arc::new(
            crate::model_manager::ModelManager::new(
                cfg,
                std::sync::Arc::new(MockEngineFactory::default()),
            )
            .await
            .unwrap(),
        );
        mm.load_model("busy-a").await.unwrap();
        let permit = match mm.get_chat_context("busy-a").unwrap().handle {
            EngineHandleKind::TextGen(h) => h.occupy_slot_for_test().unwrap(),
            other => panic!("busy-a must be a TextGen mock, got {other:?}"),
        };
        let app = crate::create_router(
            crate::app_state::AppState::new().with_model_manager(std::sync::Arc::clone(&mm)),
        );
        (app, permit, mm)
    }

    /// E-12 / U-27: a router over a mock manager with one GPU-placed and one CPU-placed model,
    /// both loaded; the returned manager lets the test poison the GPU context.
    async fn gpu_and_cpu_router() -> (
        axum::Router,
        std::sync::Arc<crate::model_manager::ModelManager>,
    ) {
        use crate::model_manager::lifecycle::MockEngineFactory;
        let cfg: crate::model_manager::Config = serde_json::from_value(serde_json::json!({
            "models_dir": "/tmp/test-admin-poison-models", "device": "GPU.1", "total_vram_gb": 22.5,
            "models": { "gpu-m": {"vram_gb": 2.0}, "cpu-m": {"vram_gb": 1.0, "device": "CPU"} }
        }))
        .unwrap();
        let mm = std::sync::Arc::new(
            crate::model_manager::ModelManager::new(
                cfg,
                std::sync::Arc::new(MockEngineFactory::default()),
            )
            .await
            .unwrap(),
        );
        mm.load_model("gpu-m").await.unwrap();
        mm.load_model("cpu-m").await.unwrap();
        let app = crate::create_router(
            crate::app_state::AppState::new().with_model_manager(std::sync::Arc::clone(&mm)),
        );
        (app, mm)
    }

    /// E-12: before poisoning nothing changes; after, `/health` keeps 200 and says so,
    /// `/health_generate` refuses before probing, GPU models read `unavailable
    /// (gpu_poisoned)` in the admin list and `/v1/models`, CPU models stay `ready`.
    #[tokio::test]
    async fn gpu_poison_is_visible_on_health_probe_and_model_lists() {
        let (app, mm) = gpu_and_cpu_router().await;

        let (status, health) = send(&app, "GET", "/health", "").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(health["gpu_poisoned"], false);
        let (_, admin) = send(&app, "GET", "/v1/admin/models", "").await;
        let state_of = |v: &serde_json::Value, id: &str| {
            v["models"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| m["id"] == id)
                .map(|m| m["state"].clone())
        };
        assert_eq!(state_of(&admin, "gpu-m"), Some("ready".into()));
        let (_, v1) = send(&app, "GET", "/v1/models", "").await;
        assert!(
            v1["data"]
                .as_array()
                .unwrap()
                .iter()
                .all(|m| m.get("state").is_none()),
            "no state key on a healthy server: {v1}"
        );

        mm.poison_for_test();

        let (status, health) = send(&app, "GET", "/health", "").await;
        assert_eq!(
            status,
            StatusCode::OK,
            "liveness probe keeps its status code"
        );
        assert_eq!(health["gpu_poisoned"], true);
        let (status, hg) = send(&app, "GET", "/health_generate", "").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(hg["error"], "gpu_poisoned", "{hg}");
        let (_, admin) = send(&app, "GET", "/v1/admin/models", "").await;
        assert_eq!(
            state_of(&admin, "gpu-m"),
            Some("unavailable (gpu_poisoned)".into())
        );
        assert_eq!(state_of(&admin, "cpu-m"), Some("ready".into()));
        let (_, v1) = send(&app, "GET", "/v1/models", "").await;
        let v1_state = |id: &str| {
            v1["data"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| m["id"] == id)
                .map(|m| m.get("state").cloned())
        };
        assert_eq!(
            v1_state("gpu-m"),
            Some(Some("unavailable (gpu_poisoned)".into()))
        );
        assert_eq!(
            v1_state("cpu-m"),
            Some(None),
            "CPU model carries no state key"
        );
        assert!(mm.metrics_snapshot().gpu_poisoned);
    }

    /// F1a: a router over a mock manager with a loaded eager model (`ready-m`), two `on_demand`
    /// models (`od-cold` not loaded, `od-warm` forced to `Loading`) and an eager model that is
    /// not loaded (`eager-cold`), all GPU-placed.
    async fn on_demand_listing_router() -> (
        axum::Router,
        std::sync::Arc<crate::model_manager::ModelManager>,
    ) {
        use crate::model_manager::lifecycle::MockEngineFactory;
        let cfg: crate::model_manager::Config = serde_json::from_value(serde_json::json!({
            "models_dir": "/tmp/test-admin-f1a-listing", "device": "GPU.1", "total_vram_gb": 22.5,
            "models": {
                "ready-m": {"vram_gb": 2.0},
                "od-cold": {"vram_gb": 1.0, "load": "on_demand"},
                "od-warm": {"vram_gb": 1.0, "load": "on_demand"},
                "eager-cold": {"vram_gb": 1.0}
            }
        }))
        .unwrap();
        let mm = std::sync::Arc::new(
            crate::model_manager::ModelManager::new(
                cfg,
                std::sync::Arc::new(MockEngineFactory::default()),
            )
            .await
            .unwrap(),
        );
        mm.load_model("ready-m").await.unwrap();
        mm.set_max_prompt_tokens_for_test("ready-m", 4096);
        mm.set_state_for_test("od-warm", crate::model_manager::ModelState::Loading);
        // A non-zero ceiling on a model that is not Ready must still not be shown.
        mm.set_max_prompt_tokens_for_test("od-warm", 4096);
        let app = crate::create_router(
            crate::app_state::AppState::new().with_model_manager(std::sync::Arc::clone(&mm)),
        );
        (app, mm)
    }

    /// F1a: `/v1/models` lists Ready models unchanged (no `state`, with
    /// `max_model_len`), `on_demand` models that are not loaded / loading with `state`
    /// `"not_loaded"` / `"loading"` and no `max_model_len`, and hides an eager model that is not
    /// loaded. `GET /v1/models/{id}` answers the same entry for each listed id and 404
    /// `model_not_found` for the hidden one.
    #[tokio::test]
    async fn v1_models_lists_on_demand_models_with_their_state() {
        let (app, _mm) = on_demand_listing_router().await;
        let (status, v1) = send(&app, "GET", "/v1/models", "").await;
        assert_eq!(status, StatusCode::OK);
        let data = v1["data"].as_array().unwrap();
        let entry = |id: &str| data.iter().find(|m| m["id"] == id).cloned();

        let ready = entry("ready-m").unwrap();
        assert!(ready.get("state").is_none(), "{ready}");
        assert_eq!(ready["max_model_len"], 4096);
        let cold = entry("od-cold").unwrap();
        assert_eq!(cold["state"], "not_loaded");
        assert!(cold.get("max_model_len").is_none(), "{cold}");
        assert_eq!(cold["object"], "model");
        let warm = entry("od-warm").unwrap();
        assert_eq!(warm["state"], "loading");
        assert!(warm.get("max_model_len").is_none(), "{warm}");
        assert!(
            entry("eager-cold").is_none(),
            "eager not-loaded stays hidden: {v1}"
        );
        assert_eq!(data.len(), 3, "{v1}");

        for id in ["ready-m", "od-cold", "od-warm"] {
            let (status, one) = send(&app, "GET", &format!("/v1/models/{id}"), "").await;
            assert_eq!(status, StatusCode::OK, "{id}: {one}");
            assert_eq!(Some(one), entry(id), "retrieve == list entry for {id}");
        }
        let (status, hidden) = send(&app, "GET", "/v1/models/eager-cold", "").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(hidden["error"]["code"], "model_not_found");
        assert_eq!(
            hidden["error"]["message"],
            "model 'eager-cold' is registered but not loaded; an administrator can load it, or \
             set it to load on demand"
        );
        let (status, unknown) = send(&app, "GET", "/v1/models/nope", "").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(unknown["error"]["code"], "model_not_found");
        assert!(
            unknown["error"]["message"]
                .as_str()
                .unwrap()
                .contains("not found among the registered models"),
            "{unknown}"
        );
    }

    /// F1a: an `Evicting` model is hidden even when `on_demand` (only not-loaded and loading
    /// are listed), and once the GPU context is poisoned an `on_demand` GPU model that is not
    /// loaded reads `"unavailable (gpu_poisoned)"` like a Ready one: it could not load.
    #[tokio::test]
    async fn v1_models_on_demand_evicting_hidden_and_poison_takes_precedence() {
        let (app, mm) = on_demand_listing_router().await;
        mm.set_state_for_test("od-warm", crate::model_manager::ModelState::Evicting);
        let (_, v1) = send(&app, "GET", "/v1/models", "").await;
        let ids: Vec<&str> = v1["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert!(!ids.contains(&"od-warm"), "evicting hidden: {v1}");
        let (status, _) = send(&app, "GET", "/v1/models/od-warm", "").await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        mm.poison_for_test();
        let (_, v1) = send(&app, "GET", "/v1/models", "").await;
        let state = |id: &str| {
            v1["data"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| m["id"] == id)
                .map(|m| m["state"].clone())
        };
        assert_eq!(state("ready-m"), Some("unavailable (gpu_poisoned)".into()));
        assert_eq!(state("od-cold"), Some("unavailable (gpu_poisoned)".into()));
        let (_, one) = send(&app, "GET", "/v1/models/od-cold", "").await;
        assert_eq!(one["state"], "unavailable (gpu_poisoned)");
        assert!(one.get("max_model_len").is_none(), "{one}");
    }

    /// F1a: the listing reads the policy overlay-first, like `request_on_demand_load`: an eager
    /// model `PATCH`ed to `load: on_demand` appears with `state: "not_loaded"`.
    #[tokio::test]
    async fn v1_models_follows_a_patched_load_policy() {
        use crate::model_manager::lifecycle::MockEngineFactory;
        // PATCH persists, so this manager needs a durable config file.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(
            &path,
            serde_json::json!({
                "models_dir": dir.path().join("models"), "device": "CPU", "total_vram_gb": 0.0,
                "models": { "eager-cold": {"vram_gb": 0.0} }
            })
            .to_string(),
        )
        .unwrap();
        let mm = std::sync::Arc::new(
            crate::model_manager::ModelManager::new(
                crate::model_manager::Config::load(&path).unwrap(),
                std::sync::Arc::new(MockEngineFactory::default()),
            )
            .await
            .unwrap()
            .with_config_path(path.clone()),
        );
        let app = crate::create_router(
            crate::app_state::AppState::new().with_model_manager(std::sync::Arc::clone(&mm)),
        );
        let (status, _) = send(&app, "GET", "/v1/models/eager-cold", "").await;
        assert_eq!(
            status,
            StatusCode::NOT_FOUND,
            "eager: hidden before the patch"
        );
        assert!(!mm.is_on_demand("eager-cold"));
        let (status, body) = send(
            &app,
            "PATCH",
            "/v1/admin/models/eager-cold",
            r#"{"load": "on_demand"}"#,
        )
        .await;
        assert!(status.is_success(), "{status}: {body}");
        assert!(mm.is_on_demand("eager-cold"));
        let (status, one) = send(&app, "GET", "/v1/models/eager-cold", "").await;
        assert_eq!(status, StatusCode::OK, "{one}");
        assert_eq!(one["state"], "not_loaded");
    }

    /// `add` validates a per-model `kv_cache_precision` (it used to refuse the key), and a tiny
    /// pool request is not printed as 0.00.
    #[tokio::test]
    async fn add_takes_a_per_model_kv_cache_precision_and_small_pools_print_exactly() {
        let models = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(models.path().join("x")).unwrap();
        std::fs::create_dir_all(models.path().join("y")).unwrap();
        let app = crate::create_router(
            crate::app_state::AppState::new().with_model_manager(std::sync::Arc::new(
                crate::model_manager::ModelManager::new(
                    serde_json::from_value(serde_json::json!({
                        "models_dir": models.path().to_str().unwrap(), "device": "CPU",
                        "total_vram_gb": 0.0
                    }))
                    .unwrap(),
                    std::sync::Arc::new(
                        crate::model_manager::lifecycle::MockEngineFactory::default(),
                    ),
                )
                .await
                .unwrap(),
            )),
        );
        let (status, body) = send(
            &app,
            "POST",
            "/v1/admin/models/add",
            r#"{"model_id":"x","vram_gb":0.0,"kv_cache_precision":"u4"}"#,
        )
        .await;
        // A per-model precision is a policy field now: an unknown value is refused with the
        // accepted list (the model directory does not exist either way), a valid one is read.
        assert_eq!(status, StatusCode::OK, "{body}");
        let (status, body) = send(
            &app,
            "POST",
            "/v1/admin/models/add",
            r#"{"model_id":"y","vram_gb":0.0,"kv_cache_precision":"bogus"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(
            body.as_str().unwrap().contains("kv_cache_precision"),
            "{body}"
        );
        assert_eq!(fmt_gib(0.001), "0.001");
        assert_eq!(fmt_gib(2.0), "2.00");
        assert_eq!(fmt_gib(4.004), "4.00");
        assert_eq!(fmt_gib(0.0), "0.00");
    }

    /// Family-smoke B1x / mismatch-cell M3b: an invalid `config.json` makes the reload answer 422
    /// (not 500) with the accepted-values text, and a valid one 200.
    #[tokio::test]
    async fn reload_answers_422_for_an_invalid_config_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        let write = |precision: &str| {
            std::fs::write(
                &path,
                format!(
                    r#"{{"models_dir":"/tmp/test-admin-reload-422","device":"CPU","total_vram_gb":0.0,"kv_cache_precision":"{precision}"}}"#
                ),
            )
            .unwrap();
        };
        write("u8");
        let mm = crate::model_manager::ModelManager::new(
            crate::model_manager::Config::load(&path).unwrap(),
            std::sync::Arc::new(crate::model_manager::lifecycle::MockEngineFactory::default()),
        )
        .await
        .unwrap()
        .with_config_path(path.clone());
        let app = crate::create_router(
            crate::app_state::AppState::new().with_model_manager(std::sync::Arc::new(mm)),
        );
        write("f32");
        let (status, body) = send(&app, "POST", "/v1/admin/config/reload", "").await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
        assert!(body.as_str().unwrap().contains("accepted values"), "{body}");
        write("u8");
        let (status, body) = send(&app, "POST", "/v1/admin/config/reload", "").await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }

    /// E-09: a request the server could not honour says so — in the load/add sentence and in
    /// `/v1/admin/models` (`kv_requested_gb`, `clamped_by`) — instead of reporting plain success.
    #[tokio::test]
    async fn clamped_kv_pool_is_reported_on_load_and_in_the_model_list() {
        use crate::model_manager::lifecycle::MockEngineFactory;
        let cfg: crate::model_manager::Config = serde_json::from_value(serde_json::json!({
            "models_dir": "/tmp/test-admin-clamp-models", "device": "GPU.1", "total_vram_gb": 22.5,
            "cache_size_gb": 2.0,
            "models": { "big-ask": {"vram_gb": 2.0}, "no-ask": {"vram_gb": 2.0}, "frac-ask": {"vram_gb": 2.0} }
        }))
        .unwrap();
        let mm = std::sync::Arc::new(
            crate::model_manager::ModelManager::new(
                cfg,
                std::sync::Arc::new(MockEngineFactory::default()),
            )
            .await
            .unwrap(),
        );
        let app = crate::create_router(
            crate::app_state::AppState::new().with_model_manager(std::sync::Arc::clone(&mm)),
        );

        let (status, body) = send(
            &app,
            "POST",
            "/v1/admin/models/big-ask/load",
            r#"{"kv_cache_gb": 4.0}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let text = body.as_str().unwrap();
        assert!(
            text.contains("KV pool 2.00 GiB (requested 4.00")
                && text.contains("clamped by cache_size_gb"),
            "{text}"
        );
        // A fractional request is rounded up, and the sentence says that (not "clamped").
        let (_, body) = send(
            &app,
            "POST",
            "/v1/admin/models/frac-ask/load",
            r#"{"kv_cache_gb": 1.21}"#,
        )
        .await;
        let text = body.as_str().unwrap();
        assert!(
            text.contains("KV pool 2.00 GiB (requested 1.21")
                && text.contains("rounded up to whole GiB"),
            "{text}"
        );
        let (status, body) = send(&app, "POST", "/v1/admin/models/no-ask/load", "").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body, "model 'no-ask' loaded",
            "no request, no clamp sentence"
        );

        let (_, admin) = send(&app, "GET", "/v1/admin/models", "").await;
        let entry = |id: &str| {
            admin["models"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| m["id"] == id)
                .unwrap()
                .clone()
        };
        let big = entry("big-ask");
        assert_eq!(big["cache_size_gb"], 2.0);
        assert_eq!(big["kv_requested_gb"], 4.0);
        assert_eq!(big["clamped_by"], "cache_size_gb");
        let none = entry("no-ask");
        assert!(none.get("clamped_by").is_none() && none.get("kv_requested_gb").is_none());
    }

    async fn send(
        app: &axum::Router,
        method: &str,
        uri: &str,
        body: &str,
    ) -> (StatusCode, serde_json::Value) {
        use tower::ServiceExt;
        let req = axum::http::Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_owned()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json = serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| serde_json::Value::String(String::from_utf8_lossy(&bytes).into()));
        (status, json)
    }

    /// 0.9.1 V5c: every text model's admin entry carries the `sampling` profile with a source
    /// per value — before its first load too (the file layer is read at registration) — while
    /// `default_*` keeps its old "omitted before the first load" meaning; a non-text kind has
    /// no `sampling`.
    #[tokio::test]
    async fn admin_models_carry_the_sampling_profile() {
        use crate::model_manager::lifecycle::MockEngineFactory;
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path().join("models");
        std::fs::create_dir_all(models.join("m-text")).unwrap();
        std::fs::write(
            models.join("m-text").join("generation_config.json"),
            r#"{"do_sample": true, "temperature": 0.6, "top_k": 20, "repetition_penalty": 1.05}"#,
        )
        .unwrap();
        let cfg: crate::model_manager::Config = serde_json::from_value(serde_json::json!({
            "models_dir": models, "device": "CPU", "total_vram_gb": 0.0,
            "models": {
                "m-text": {"vram_gb": 0.0, "sampling": {"top_p": 0.8, "presence_penalty": 1.5}},
                "embed-x": {"vram_gb": 0.0}
            }
        }))
        .unwrap();
        let mm = std::sync::Arc::new(
            crate::model_manager::ModelManager::new(
                cfg,
                std::sync::Arc::new(MockEngineFactory::default()),
            )
            .await
            .unwrap(),
        );
        let app = crate::create_router(
            crate::app_state::AppState::new().with_model_manager(std::sync::Arc::clone(&mm)),
        );
        let entry = |v: &serde_json::Value, id: &str| {
            v["models"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| m["id"] == id)
                .cloned()
                .unwrap()
        };
        let (_, admin) = send(&app, "GET", "/v1/admin/models", "").await;
        let text = entry(&admin, "m-text");
        assert_eq!(text["state"], "not_loaded");
        assert!(text.get("default_temperature").is_none(), "{text}");
        let s = &text["sampling"];
        assert_eq!(s["mode"], "sample");
        assert_eq!(s["temperature"]["source"], "generation_config.json");
        assert_eq!(s["top_p"]["source"], "config");
        assert_eq!(s["top_k"]["value"], 20);
        assert_eq!(s["repetition_penalty"]["source"], "generation_config.json");
        assert_eq!(s["presence_penalty"]["source"], "config");
        assert_eq!(
            s["frequency_penalty"],
            serde_json::json!({"value": null, "source": "none"})
        );
        assert_eq!(s["tools"], "greedy");
        assert!(entry(&admin, "embed-x").get("sampling").is_none());

        mm.load_model("m-text").await.unwrap();
        let (_, admin) = send(&app, "GET", "/v1/admin/models", "").await;
        let text = entry(&admin, "m-text");
        assert!((text["default_temperature"].as_f64().unwrap() - 0.6).abs() < 1e-6);
        assert_eq!(text["sampling"]["top_p"]["source"], "config");
    }

    /// 0.9.1: every admin model entry carries `thinking_budget_tokens` (`null` when unset); a
    /// PATCH sets it hot on a `Ready` model (the next chat context reads it), `null` clears it,
    /// and the value is persisted; a negative or non-integer value is refused.
    #[tokio::test]
    async fn thinking_budget_tokens_admin_field_and_patch() {
        use crate::model_manager::lifecycle::MockEngineFactory;
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path().join("models");
        std::fs::create_dir_all(models.join("m-text")).unwrap();
        std::fs::create_dir_all(models.join("m-set")).unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(
            &path,
            serde_json::json!({
                "models_dir": models, "device": "CPU", "total_vram_gb": 0.0,
                "models": {
                    "m-text": {"vram_gb": 0.0, "thinking_budget_tokens": null},
                    "m-set": {"vram_gb": 0.0, "thinking_budget_tokens": 3000}
                }
            })
            .to_string(),
        )
        .unwrap();
        let cfg = crate::model_manager::Config::load(&path).unwrap();
        let mm = std::sync::Arc::new(
            crate::model_manager::ModelManager::new(
                cfg,
                std::sync::Arc::new(MockEngineFactory::default()),
            )
            .await
            .unwrap()
            .with_config_path(path.clone()),
        );
        let app = crate::create_router(
            crate::app_state::AppState::new().with_model_manager(std::sync::Arc::clone(&mm)),
        );
        let field = |v: &serde_json::Value, id: &str| {
            v["models"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| m["id"] == id)
                .unwrap()
                .as_object()
                .unwrap()
                .get("thinking_budget_tokens")
                .cloned()
        };
        let (_, admin) = send(&app, "GET", "/v1/admin/models", "").await;
        assert_eq!(field(&admin, "m-text"), Some(serde_json::Value::Null));
        assert_eq!(field(&admin, "m-set"), Some(serde_json::json!(3000)));

        mm.load_model("m-text").await.unwrap();
        let (status, body) = send(
            &app,
            "PATCH",
            "/v1/admin/models/m-text",
            r#"{"thinking_budget_tokens": 2048}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(
            body.to_string().contains("thinking_budget_tokens"),
            "{body}"
        );
        let (_, admin) = send(&app, "GET", "/v1/admin/models", "").await;
        assert_eq!(field(&admin, "m-text"), Some(serde_json::json!(2048)));
        let ctx = mm.get_chat_context_for_request("m-text").await.unwrap();
        assert_eq!(
            ctx.thinking_budget_tokens,
            Some(2048),
            "hot on the Ready model"
        );
        drop(ctx);
        let on_disk: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(
            on_disk["models"]["m-text"]["thinking_budget_tokens"],
            serde_json::json!(2048)
        );

        let (status, body) = send(
            &app,
            "PATCH",
            "/v1/admin/models/m-text",
            r#"{"thinking_budget_tokens": null}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let (_, admin) = send(&app, "GET", "/v1/admin/models", "").await;
        assert_eq!(field(&admin, "m-text"), Some(serde_json::Value::Null));
        let ctx = mm.get_chat_context_for_request("m-text").await.unwrap();
        assert_eq!(ctx.thinking_budget_tokens, None);
        drop(ctx);

        for bad in [
            r#"{"thinking_budget_tokens": -1}"#,
            r#"{"thinking_budget_tokens": 1.5}"#,
        ] {
            // A wrong type: the 400 `invalid_body` envelope since P-C (422 before).
            let (status, body) = send(&app, "PATCH", "/v1/admin/models/m-text", bad).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}: {body}");
            assert_eq!(body["error"]["code"], "invalid_body", "{bad}: {body}");
        }
    }

    /// Review F3: the status a typo'd key inside `sampling` really gets over HTTP, through PATCH
    /// and through the add body's flattened policy (axum's `Json` extractor answers a serde data
    /// error itself), and that nothing is changed.
    #[tokio::test]
    async fn sampling_typo_status_over_http() {
        use crate::model_manager::lifecycle::MockEngineFactory;
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path().join("models");
        std::fs::create_dir_all(models.join("m-text")).unwrap();
        std::fs::create_dir_all(models.join("m-new")).unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(
            &path,
            serde_json::json!({
                "models_dir": models, "device": "CPU", "total_vram_gb": 0.0,
                "models": { "m-text": {"vram_gb": 0.0} }
            })
            .to_string(),
        )
        .unwrap();
        let cfg = crate::model_manager::Config::load(&path).unwrap();
        let mm = std::sync::Arc::new(
            crate::model_manager::ModelManager::new(
                cfg,
                std::sync::Arc::new(MockEngineFactory::default()),
            )
            .await
            .unwrap()
            .with_config_path(path.clone()),
        );
        let app = crate::create_router(
            crate::app_state::AppState::new().with_model_manager(std::sync::Arc::clone(&mm)),
        );
        let before = std::fs::read_to_string(&path).unwrap();
        let (status, body) = send(
            &app,
            "PATCH",
            "/v1/admin/models/m-text",
            r#"{"sampling": {"temprature": 1}}"#,
        )
        .await;
        // PATCH: an unknown field at any depth keeps its 422 (plain text) after P-C.
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
        assert!(body.is_string(), "plain text as before: {body}");
        assert!(body.to_string().contains("temprature"), "{body}");
        let (status, body) = send(
            &app,
            "POST",
            "/v1/admin/models/add",
            r#"{"model_id": "m-new", "vram_gb": 0.0, "sampling": {"temprature": 1}}"#,
        )
        .await;
        // models/add: the 400 `invalid_body` envelope since P-C (422 before), naming the typo.
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["code"], "invalid_body", "{body}");
        assert!(body.to_string().contains("temprature"), "{body}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
        assert!(mm.model_state("m-new").is_none());
        // A value out of range is the handler's own 400.
        let (status, _) = send(
            &app,
            "PATCH",
            "/v1/admin/models/m-text",
            r#"{"sampling": {"temperature": 3}}"#,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    /// Release the simulated request after a short delay, so a forced eviction has something to
    /// drain.
    fn release_later(permit: tokio::sync::OwnedSemaphorePermit) {
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
            drop(permit);
        });
    }

    /// C8: the admin load of a model whose `chat_template` override cannot be read answers 404
    /// with a body naming the model and the path; after a reload that edits a loaded model's
    /// `chat_template`, the reload response lists it in `models_not_applied`.
    #[tokio::test]
    async fn chat_template_load_error_and_reload_notice_reach_the_admin_responses() {
        use crate::model_manager::lifecycle::MockEngineFactory;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        let write = |tpl: &str| {
            std::fs::write(
                &path,
                serde_json::json!({
                    "models_dir": "/tmp/test-admin-tpl-models", "device": "CPU",
                    "total_vram_gb": 22.5,
                    "models": {
                        "tpl-bad": {"vram_gb": 1.0, "chat_template": "/nonexistent/c8/admin.jinja"},
                        "tpl-ok": {"vram_gb": 1.0, "chat_template": tpl}
                    }
                })
                .to_string(),
            )
            .unwrap();
        };
        std::fs::write(dir.path().join("a.jinja"), "A").unwrap();
        std::fs::write(dir.path().join("b.jinja"), "B").unwrap();
        write("a.jinja");
        let cfg = crate::model_manager::Config::load(&path).unwrap();
        let mm = std::sync::Arc::new(
            crate::model_manager::ModelManager::new(
                cfg,
                std::sync::Arc::new(MockEngineFactory::default()),
            )
            .await
            .unwrap()
            .with_config_path(path.clone()),
        );
        let app = crate::create_router(
            crate::app_state::AppState::new().with_model_manager(std::sync::Arc::clone(&mm)),
        );
        let (status, body) = send(&app, "POST", "/v1/admin/models/tpl-bad/load", "").await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
        let text = body.as_str().unwrap_or_default();
        assert!(text.contains("'tpl-bad'"), "{text}");
        assert!(text.contains("/nonexistent/c8/admin.jinja"), "{text}");

        let (status, body) = send(&app, "POST", "/v1/admin/models/tpl-ok/load", "").await;
        assert_eq!(status, StatusCode::OK, "{body}");
        write("b.jinja");
        let (status, json) = send(&app, "POST", "/v1/admin/config/reload", "").await;
        assert_eq!(status, StatusCode::OK, "{json}");
        assert_eq!(
            json["models_not_applied"],
            serde_json::json!([{"model": "tpl-ok", "fields": ["chat_template"]}]),
            "{json}"
        );
    }

    /// Unload: a busy model is refused with the JSON 409 `model_busy` unless `?force=true`,
    /// which drains it and evicts it.
    #[tokio::test]
    async fn unload_refuses_a_busy_model_unless_forced() {
        let (app, permit, mm) = busy_router().await;
        let (status, json) = send(&app, "DELETE", "/v1/admin/models/busy-a", "").await;
        assert_eq!(status, StatusCode::CONFLICT, "{json}");
        assert_eq!(json["error"]["code"], "model_busy");
        assert!(
            mm.list_models()
                .iter()
                .any(|m| m.id == "busy-a" && m.state == crate::model_manager::ModelState::Ready),
            "a refused unload leaves the model loaded"
        );

        release_later(permit);
        let (status, body) = send(&app, "DELETE", "/v1/admin/models/busy-a?force=true", "").await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(mm.list_models().iter().any(|m| m.id == "busy-a"
            && m.state == crate::model_manager::ModelState::NotLoaded));
    }

    /// `send` with an optional `Accept` header, returning the raw body and its content type.
    async fn send_accept(
        app: &axum::Router,
        method: &str,
        uri: &str,
        body: &str,
        accept: Option<&str>,
    ) -> (StatusCode, String, Vec<u8>) {
        use tower::ServiceExt;
        let mut req = axum::http::Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json");
        if let Some(a) = accept {
            req = req.header("accept", a);
        }
        let req = req.body(axum::body::Body::from(body.to_owned())).unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let ctype = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_owned();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, ctype, bytes.to_vec())
    }

    fn outcome(bytes: &[u8]) -> serde_json::Value {
        serde_json::from_slice(bytes)
            .unwrap_or_else(|e| panic!("not JSON ({e}): {}", String::from_utf8_lossy(bytes)))
    }

    #[test]
    fn json_is_wanted_only_for_an_explicit_application_json() {
        let h = |v: &str| {
            let mut m = HeaderMap::new();
            m.insert(axum::http::header::ACCEPT, v.parse().unwrap());
            m
        };
        assert!(!wants_json(&HeaderMap::new()));
        assert!(!wants_json(&h("*/*")));
        assert!(!wants_json(&h("text/plain")));
        assert!(!wants_json(&h("application/jsonx")));
        assert!(wants_json(&h("application/json")));
        assert!(wants_json(&h("text/plain, Application/JSON;q=0.9")));
        assert!(wants_json(&h("application/json; charset=utf-8")));
        // axios' default: names application/json, so it gets the JSON (INTERFACE.md says so).
        assert!(wants_json(&h("application/json, text/plain, */*")));
        assert!(!wants_json(&h("application/json;q=0")));
        assert!(!wants_json(&h("application/json; q=0.0, text/plain")));
        // Two Accept headers: either may name it.
        let mut two = HeaderMap::new();
        two.append(axum::http::header::ACCEPT, "text/plain".parse().unwrap());
        two.append(
            axum::http::header::ACCEPT,
            "application/json".parse().unwrap(),
        );
        assert!(wants_json(&two));
        let mut neither = HeaderMap::new();
        neither.append(axum::http::header::ACCEPT, "text/plain".parse().unwrap());
        neither.append(axum::http::header::ACCEPT, "*/*".parse().unwrap());
        assert!(!wants_json(&neither));
    }

    /// V1-rv P1: without `Accept: application/json` every route answers the plain text byte for
    /// byte as before (no header, or `*/*` as curl sends); with it, the JSON outcome.
    #[tokio::test]
    async fn eviction_routes_answer_plain_text_unless_json_is_asked_for() {
        let (app, permit, _mm) = busy_router().await;
        drop(permit);
        for accept in [None, Some("*/*")] {
            let (status, ctype, body) = send_accept(
                &app,
                "DELETE",
                "/v1/admin/models/busy-a?force=true",
                "",
                accept,
            )
            .await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(body, b"model 'busy-a' evicted", "{accept:?}");
            assert!(ctype.starts_with("text/plain"), "{ctype}");
            let (status, _, body) =
                send_accept(&app, "POST", "/v1/admin/models/busy-a/load", "", accept).await;
            assert_eq!(status, StatusCode::OK);
            assert!(
                body.starts_with(b"model 'busy-a' loaded"),
                "{}",
                String::from_utf8_lossy(&body)
            );
        }
        let (status, _, body) = send_accept(
            &app,
            "POST",
            "/v1/admin/models/busy-a/resize",
            r#"{"kv_cache_gb": 2.0}"#,
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            body.starts_with(b"model 'busy-a' resized to "),
            "{}",
            String::from_utf8_lossy(&body)
        );
        let (status, _, body) =
            send_accept(&app, "DELETE", "/v1/admin/models/busy-a/register", "", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, b"model 'busy-a' deregistered");
    }

    /// V1-rv P1: the JSON outcome of an unload — nothing running, a request that finished
    /// during the drain, and one aborted at the (shortened) drain deadline.
    #[tokio::test]
    async fn unload_json_reports_nothing_running_drained_and_aborted() {
        let json = Some("application/json");
        let (app, permit, mm) = busy_router().await;
        drop(permit);
        let (status, ctype, body) =
            send_accept(&app, "DELETE", "/v1/admin/models/busy-a", "", json).await;
        assert_eq!(status, StatusCode::OK);
        assert!(ctype.starts_with("application/json"), "{ctype}");
        let v = outcome(&body);
        assert_eq!(v["evicted"], serde_json::json!(["busy-a"]));
        assert_eq!(v["forced"], false);
        assert_eq!(v["drained_s"], 0.0);
        assert_eq!(v["aborted"], 0);
        assert!(v["teardown_ms"].is_u64(), "{v}");
        let keys: Vec<&String> = v.as_object().unwrap().keys().collect();
        assert_eq!(
            keys,
            [
                "evicted",
                "forced",
                "drained_s",
                "aborted",
                "teardown_ms",
                "message"
            ],
            "the field set is fixed (`message` since 0.9.1)"
        );
        assert_eq!(v["message"], "model 'busy-a' evicted");

        // Drained: the request finishes ~150 ms into the drain.
        mm.load_model("busy-a").await.unwrap();
        let permit = match mm.get_chat_context("busy-a").unwrap().handle {
            EngineHandleKind::TextGen(h) => h.occupy_slot_for_test().unwrap(),
            other => panic!("busy-a must be a TextGen mock, got {other:?}"),
        };
        release_later(permit);
        let (status, _, body) = send_accept(
            &app,
            "DELETE",
            "/v1/admin/models/busy-a?force=true",
            "",
            json,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let v = outcome(&body);
        assert_eq!(v["forced"], true);
        assert!(v["drained_s"].as_f64().unwrap() > 0.0, "{v}");
        assert_eq!(v["aborted"], 0, "{v}");

        // Aborted: still running at the drain deadline.
        mm.load_model("busy-a").await.unwrap();
        mm.set_evict_drain_deadline_for_test(Duration::from_millis(200));
        let permit = match mm.get_chat_context("busy-a").unwrap().handle {
            EngineHandleKind::TextGen(h) => h.occupy_slot_for_test().unwrap(),
            other => panic!("busy-a must be a TextGen mock, got {other:?}"),
        };
        let (status, _, body) = send_accept(
            &app,
            "DELETE",
            "/v1/admin/models/busy-a?force=true",
            "",
            json,
        )
        .await;
        drop(permit);
        assert_eq!(status, StatusCode::OK);
        let v = outcome(&body);
        assert_eq!(v["evicted"], serde_json::json!(["busy-a"]));
        assert_eq!(v["aborted"], 1, "{v}");
        assert!(v["drained_s"].as_f64().unwrap() >= 0.15, "{v}");
    }

    /// V1-rv P1: deregister and resize answer the same JSON; a deregistration of a model that
    /// is not loaded evicted nothing.
    #[tokio::test]
    async fn deregister_and_resize_answer_the_json_outcome() {
        let json = Some("application/json");
        let (app, permit, _mm) = busy_router().await;
        release_later(permit);
        let (status, _, body) = send_accept(
            &app,
            "POST",
            "/v1/admin/models/busy-a/resize",
            r#"{"kv_cache_gb": 2.0, "force": true}"#,
            json,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
        let v = outcome(&body);
        assert_eq!(v["evicted"], serde_json::json!(["busy-a"]));
        assert_eq!(v["forced"], true);
        assert!(v["drained_s"].as_f64().unwrap() > 0.0, "{v}");
        assert!(
            v["message"]
                .as_str()
                .unwrap()
                .starts_with("model 'busy-a' resized to "),
            "the plain-text sentence, achieved pool included: {v}"
        );

        let (status, _, body) =
            send_accept(&app, "DELETE", "/v1/admin/models/busy-a", "", json).await;
        assert_eq!(status, StatusCode::OK);
        let (status, _, body2) =
            send_accept(&app, "DELETE", "/v1/admin/models/busy-a/register", "", json).await;
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&body2)
        );
        assert_eq!(outcome(&body)["evicted"], serde_json::json!(["busy-a"]));
        let v = outcome(&body2);
        assert_eq!(v["evicted"], serde_json::json!([]));
        assert_eq!(v["drained_s"], 0.0);
        assert_eq!(v["aborted"], 0);
        assert_eq!(v["teardown_ms"], 0);
        assert_eq!(v["message"], "model 'busy-a' deregistered");
    }

    /// V1-rv P1: a forced load's JSON names the victim it evicted for room.
    #[tokio::test]
    async fn forced_load_json_names_its_victims() {
        use crate::model_manager::lifecycle::MockEngineFactory;
        let cfg: crate::model_manager::Config = serde_json::from_value(serde_json::json!({
            "models_dir": "/tmp/test-admin-force-models", "device": "CPU", "total_vram_gb": 17.0,
            "cache_size_gb": 1.0, "eviction_grace_secs": 0.0, "vram_safety_margin_gb": 0.0,
            "min_kv_cache_gb": 0.0,
            "models": { "a": {"vram_gb": 5.0}, "b": {"vram_gb": 5.0}, "incoming": {"vram_gb": 8.0} }
        }))
        .unwrap();
        let mm = std::sync::Arc::new(
            crate::model_manager::ModelManager::new(
                cfg,
                std::sync::Arc::new(MockEngineFactory::default()),
            )
            .await
            .unwrap(),
        );
        mm.load_model("a").await.unwrap();
        mm.load_model("b").await.unwrap();
        let hold = |id: &str| match mm.get_chat_context(id).unwrap().handle {
            EngineHandleKind::TextGen(h) => h.occupy_slot_for_test().unwrap(),
            other => panic!("{id} must be a TextGen mock, got {other:?}"),
        };
        let (pa, pb) = (hold("a"), hold("b"));
        let app = crate::create_router(
            crate::app_state::AppState::new().with_model_manager(std::sync::Arc::clone(&mm)),
        );
        release_later(pa);
        release_later(pb);
        let (status, _, body) = send_accept(
            &app,
            "POST",
            "/v1/admin/models/incoming/load",
            r#"{"force": true}"#,
            Some("application/json"),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
        let v = outcome(&body);
        assert_eq!(v["forced"], true);
        let evicted = v["evicted"].as_array().unwrap();
        assert_eq!(evicted.len(), 1, "{v}");
        assert!(["a", "b"].contains(&evicted[0].as_str().unwrap()), "{v}");
        assert!(v["drained_s"].as_f64().unwrap() > 0.0, "{v}");
        assert!(
            v["message"]
                .as_str()
                .unwrap()
                .starts_with("model 'incoming' loaded"),
            "{v}"
        );
        assert_eq!(v["aborted"], 0, "{v}");
    }

    /// Deregister: same contract as unload.
    #[tokio::test]
    async fn deregister_refuses_a_busy_model_unless_forced() {
        let (app, permit, mm) = busy_router().await;
        let (status, json) = send(&app, "DELETE", "/v1/admin/models/busy-a/register", "").await;
        assert_eq!(status, StatusCode::CONFLICT, "{json}");
        assert_eq!(json["error"]["code"], "model_busy");

        release_later(permit);
        let (status, body) = send(
            &app,
            "DELETE",
            "/v1/admin/models/busy-a/register?force=true",
            "",
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(
            mm.list_models().iter().all(|m| m.id != "busy-a"),
            "deregistered"
        );
    }

    /// Resize: a busy model is refused with the JSON 409 unless the body says `"force": true`,
    /// and a refused resize does NOT burn the resize cooldown (D3-S1.F2): the forced retry right
    /// after it succeeds instead of being answered 429.
    #[tokio::test]
    async fn resize_refuses_a_busy_model_unless_forced() {
        let (app, permit, _mm) = busy_router().await;
        let (status, json) = send(
            &app,
            "POST",
            "/v1/admin/models/busy-a/resize",
            r#"{"kv_cache_gb": 2.0}"#,
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT, "{json}");
        assert_eq!(json["error"]["code"], "model_busy");

        release_later(permit);
        let (status, body) = send(
            &app,
            "POST",
            "/v1/admin/models/busy-a/resize",
            r#"{"kv_cache_gb": 2.0, "force": true}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }

    /// D3-S1.F3: `POST /v1/admin/models/add` whose load can only be satisfied by evicting a busy
    /// model is answered with the JSON 503 `model_busy` (not a 400), and the registration is
    /// rolled back as for any failed load; once the resident is idle the same add succeeds.
    #[tokio::test]
    async fn add_whose_load_is_blocked_by_a_busy_model_answers_model_busy() {
        let (app, permit, mm) = busy_router().await;
        // `add` checks the model's directory exists under `models_dir` before registering.
        std::fs::create_dir_all("/tmp/test-admin-force-models/newcomer").unwrap();
        let body = r#"{"model_id": "newcomer", "vram_gb": 15.0}"#;
        let (status, json) = send(&app, "POST", "/v1/admin/models/add", body).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{json}");
        assert_eq!(json["error"]["code"], "model_busy", "{json}");
        assert!(
            mm.list_models().iter().all(|m| m.id != "newcomer"),
            "a refused add leaves nothing registered"
        );
        let busy_state = mm
            .list_models()
            .into_iter()
            .find(|m| m.id == "busy-a")
            .unwrap()
            .state;
        assert_eq!(
            busy_state,
            crate::model_manager::ModelState::Ready,
            "nothing was evicted"
        );

        drop(permit);
        let (status, ok) = send(&app, "POST", "/v1/admin/models/add", body).await;
        assert_eq!(status, StatusCode::OK, "{ok}");
    }
}
