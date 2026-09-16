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
    Json,
    body::Bytes,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use crate::app_state::AppState;
use crate::model_manager::{EngineHandleKind, LoadOverrides, ModelError};
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
}

/// GET /health — liveness probe.
///
/// Returns `{"status":"ok","version":"<semver>","git_hash":"<12-char hex or \"unknown\">"}`.
/// Never returns an error — if the server is alive enough to route here, it's healthy.
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        git_hash: env!("RV_GIT_HASH"),
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
    message: String,
}

/// Build a 503 error response for [`health_generate`].
fn hg_err(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(HealthGenerateError {
            status: "error",
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
            match npu
                .generate(prompt, gen_params.max_new_tokens, tx, start)
                .await
            {
                crate::cb_engine::SubmitResult::Submitted => Ok(()),
                crate::cb_engine::SubmitResult::AtCapacity => Err(hg_err(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "npu engine at capacity",
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
        | EngineHandleKind::Reranking(_) => Err(hg_err(
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
/// RTH's own `health_generate_during_eviction_pressure` scenario caught the
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

/// One entry in the `GET /v1/admin/models` registry response.
#[derive(Serialize)]
struct AdminModelEntry {
    id: String,
    /// `"not_loaded"` | `"loading"` | `"ready"` | `"evicting"`
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
    /// `temperature`/`top_p`/`top_k` itself — see
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
    /// The model's own native (trained/converted) context ceiling —
    /// `max_position_embeddings`, `rope_scaling`-adjusted. A hardware-
    /// independent model property: contrast with `max_prompt_tokens`, which
    /// is what this box's KV pool can currently *serve*. The two can differ
    /// wildly (confirmed live 2026-08-23, `dev/autotest/
    /// 20260823_max_position_embeddings_gate_gap.md`: a VRAM-derived ceiling
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
    /// Whether this model's KV occupancy has been continuously at/above the
    /// configured pressure threshold for at least the configured sustained
    /// duration (`dev/plans/kv-cache-pressure-detection.md`). Always `false`
    /// while the monitor is disabled (the fleet-wide default) — surfaced
    /// here, not only in `/metrics`, so an operator not watching Prometheus
    /// can still see it via this endpoint.
    kv_pressure_flagged: bool,
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
#[derive(Serialize)]
struct ServerInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    host: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    openvino_version: Option<String>,
    engine: &'static str,
    engine_version: &'static str,
}

impl ServerInfo {
    fn current() -> Self {
        Self {
            host: crate::os_memory::host_name(),
            openvino_version: crate::ov_cb::openvino_version(),
            engine: "RustedVINO",
            engine_version: env!("CARGO_PKG_VERSION"),
        }
    }
}

/// GET /v1/admin/models — list every registered model with its current state.
///
/// Unlike `GET /v1/models` (OpenAI-compatible, Ready-only), this returns the
/// full registry: models that are `not_loaded`, `loading`, `ready`, or
/// `evicting`. Useful for clients that want to know what can be loaded.
///
/// Each entry also carries `max_prompt_tokens`/`cache_size_gb` — the L0
/// prompt-length gate ceiling and the KV pool it was computed against, both
/// `0`/`0.0` before first load or once evicted. Added so an external client
/// or operator can discover a model's real usable context window
/// proactively, instead of only reactively via a `400
/// context_length_exceeded` after crossing it (`dev/autotest/
/// 20260823_external_metrics_interface_audit.md`).
///
/// Also carries `native_context_limit` (omitted when unknown) — the model's
/// own trained/converted context ceiling, independent of `max_prompt_tokens`'
/// VRAM-derived number, so an operator can tell "VRAM-limited" from
/// "model-limited" at a glance (`dev/autotest/
/// 20260823_max_position_embeddings_gate_gap.md`).
///
/// - 200 OK — registry snapshot returned
/// - 503 Service Unavailable — model manager not initialised
pub async fn admin_list_models(State(state): State<AppState>) -> Response {
    let Some(mm) = state.model_manager.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model manager is not initialised",
        )
            .into_response();
    };

    let models = mm
        .list_models()
        .into_iter()
        .map(|info| AdminModelEntry {
            id: info.id,
            state: match info.state {
                crate::model_manager::ModelState::NotLoaded => "not_loaded",
                crate::model_manager::ModelState::Loading => "loading",
                crate::model_manager::ModelState::Ready => "ready",
                crate::model_manager::ModelState::Evicting => "evicting",
            },
            kind: info.kind.label(),
            vram_gb: info.vram_gb,
            device: info.device,
            default_temperature: info.generation_defaults.temperature,
            default_top_p: info.generation_defaults.top_p,
            default_top_k: info.generation_defaults.top_k,
            max_prompt_tokens: info.max_prompt_tokens,
            cache_size_gb: info.kv_cache_gb,
            native_context_limit: info.native_context_limit,
            kv_cache_usage_pct: info.kv_cache_usage_pct,
            kv_pressure_flagged: info.kv_pressure_flagged,
        })
        .collect();

    Json(AdminModelsResponse {
        models,
        server: ServerInfo::current(),
    })
    .into_response()
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
    /// Age in seconds of the longest generation currently in flight across
    /// every Ready model, or `null` if nothing is generating right now.
    oldest_generation_secs: Option<f64>,
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
/// [`crate::model_manager::ModelManager::oldest_generation_age`].
///
/// Always 200 — including when the model manager isn't initialised yet — so
/// the supervisor's minimal hand-rolled HTTP/1.0 client (`supervisor.rs`,
/// mirrors `probe_health_once`) never has to distinguish a real hang from a
/// transient 5xx.
pub async fn admin_watchdog(State(state): State<AppState>) -> Response {
    let oldest_generation_secs = state
        .model_manager
        .as_ref()
        .and_then(|mm| mm.oldest_generation_age())
        .map(|d| d.as_secs_f64());
    Json(AdminWatchdogResponse {
        oldest_generation_secs,
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
/// ruvi-voice clients should call this before doing model discovery. If a pin
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
    /// /v1/admin/models` (`dev/autotest/
    /// 20260823_max_position_embeddings_gate_gap.md`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<usize>,
}

/// Top-level /v1/models response envelope.
#[derive(Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

/// GET /v1/models — OpenAI-compatible model discovery.
///
/// Returns only models in the `Ready` state (i.e. loaded and accepting requests).
/// Models that are registered but not loaded are invisible to `OpenAI` clients;
/// the admin routes control loading.
///
/// When `model_manager` is absent (mock mode), returns an empty list.
pub async fn models_list(State(state): State<AppState>) -> Json<ModelsResponse> {
    let data = state
        .model_manager
        .as_ref()
        .map(|mm| {
            mm.list_models()
                .into_iter()
                .filter(|info| info.state == crate::model_manager::ModelState::Ready)
                .map(|info| ModelObject {
                    id: info.id,
                    object: "model",
                    created: server_created_ts(),
                    owned_by: "rustedvino",
                    max_model_len: (info.max_prompt_tokens > 0).then_some(info.max_prompt_tokens),
                })
                .collect()
        })
        .unwrap_or_default();

    Json(ModelsResponse {
        object: "list",
        data,
    })
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
    /// Realtime voice arbitration v2 (`dev/plans/realtime-voice-model-
    /// arbitration-v2.md`, D3): bypass the eviction-grace window for this
    /// load. Only the grace window — never the realtime serving set,
    /// `pinned`, or non-`evictable`, all of which remain unconditionally
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
/// - 503 Service Unavailable — model manager not initialised
/// - 500 Internal Server Error — engine / factory failure
///
/// Loading may take 10–60 seconds depending on model size and GPU JIT.
/// The request blocks until the model is ready (or fails). Callers should
/// use a long HTTP timeout.
pub async fn admin_load_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    body: Bytes,
) -> Response {
    // Parse + validate the optional overrides body first, so a malformed
    // request is a 400 regardless of server state. An empty body is the
    // pre-Slice-2b path: no overrides.
    let overrides = if body.is_empty() {
        LoadOverrides::default()
    } else {
        match serde_json::from_slice::<LoadRequest>(&body) {
            Ok(req) => match req.into_overrides() {
                Ok(ov) => ov,
                Err(msg) => return (StatusCode::BAD_REQUEST, msg).into_response(),
            },
            Err(e) => {
                return (StatusCode::BAD_REQUEST, format!("invalid JSON body: {e}"))
                    .into_response();
            }
        }
    };

    let Some(mm) = state.model_manager.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model manager is not initialised",
        )
            .into_response();
    };

    match mm.load_model_with_overrides(&model_id, overrides).await {
        Ok(()) => (StatusCode::OK, format!("model '{model_id}' loaded")).into_response(),
        Err(e) => {
            // Downcast to ModelError for fine-grained HTTP status mapping.
            if let Some(model_err) = e.downcast_ref::<ModelError>() {
                match model_err {
                    ModelError::NotFound(_)
                    | ModelError::FilesMissing(_)
                    | ModelError::FilesIncomplete(..) => {
                        return (StatusCode::NOT_FOUND, e.to_string()).into_response();
                    }
                    ModelError::Loading | ModelError::Evicting => {
                        return (StatusCode::CONFLICT, e.to_string()).into_response();
                    }
                    ModelError::GpuPoisoned => {
                        // L3: OpenCL context poisoned by a prior OOM — operator must
                        // restart the process; no retry will succeed. The restart
                        // should be graceful: SIGTERM/`systemctl stop` now drains
                        // in-flight requests and frees VRAM (T6.1) before exit.
                        return (
                            StatusCode::SERVICE_UNAVAILABLE,
                            format!("{e} — restart rustedvino to recover (a graceful stop via SIGTERM/systemctl drains in-flight requests and frees VRAM first)"),
                        )
                            .into_response();
                    }
                    // WrongKind never originates from load_model; fall through to 500.
                    ModelError::NotLoaded | ModelError::WrongKind(_) => {} // fall through to 500
                }
            }
            tracing::error!(model_id, error = %e, "admin_load_model failed");
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
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
}

#[allow(clippy::doc_markdown)]
/// POST /v1/admin/models/{model_id}/resize — evict then reload a currently-
/// resident model with a new KV-cache size
/// (`dev/plans/kv-cache-pressure-detection.md`).
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
    body: Bytes,
) -> Response {
    let request: ResizeRequest = match serde_json::from_slice(&body) {
        Ok(req) => req,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, format!("invalid JSON body: {e}")).into_response();
        }
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
        .resize_model_kv_cache(&model_id, request.kv_cache_gb)
        .await
    {
        // `achieved` is the pool the allocator actually handed out, which can be
        // short of the request: the KV target is clamped by free VRAM after
        // weights, by the global `cache_size_gb` cap, and floored at
        // `min_kv_cache_gb`. Report the achieved figure, and say so explicitly
        // when it differs — echoing the request turned a clamped pool into a
        // silent full-size "success".
        Ok(achieved) => {
            let msg = if (achieved - request.kv_cache_gb).abs() < 1e-9 {
                format!("model '{model_id}' resized to {achieved} GB")
            } else {
                format!(
                    "model '{model_id}' resized to {achieved} GB \
                     (requested {} GB — clamped by free VRAM, cache_size_gb, \
                     or the min_kv_cache_gb floor)",
                    request.kv_cache_gb
                )
            };
            (StatusCode::OK, msg).into_response()
        }
        Err(e) => {
            // Cooldown rejection is a plain anyhow error, not a ModelError
            // variant (same string-matching convention admin_reload_config
            // already uses for its own narrow "config_path not set" case) —
            // a new enum variant would ripple through every exhaustive
            // ModelError match in this file for one rare, purely-informational
            // rejection.
            if e.to_string().contains("cooldown is") {
                return (StatusCode::TOO_MANY_REQUESTS, e.to_string()).into_response();
            }
            if let Some(model_err) = e.downcast_ref::<ModelError>() {
                match model_err {
                    ModelError::NotFound(_) => {
                        return (StatusCode::NOT_FOUND, e.to_string()).into_response();
                    }
                    ModelError::NotLoaded | ModelError::Loading | ModelError::Evicting => {
                        return (StatusCode::CONFLICT, e.to_string()).into_response();
                    }
                    ModelError::GpuPoisoned => {
                        return (
                            StatusCode::SERVICE_UNAVAILABLE,
                            format!("{e} — restart rustedvino to recover (a graceful stop via SIGTERM/systemctl drains in-flight requests and frees VRAM first)"),
                        )
                            .into_response();
                    }
                    // FilesMissing/FilesIncomplete/WrongKind never originate
                    // from resize_model_kv_cache — completeness is only
                    // checked before a load, and resize never changes kind.
                    ModelError::FilesMissing(_)
                    | ModelError::FilesIncomplete(..)
                    | ModelError::WrongKind(_) => {}
                }
            }
            tracing::error!(model_id, error = %e, "admin_resize_model failed");
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
        }
    }
}

#[allow(clippy::doc_markdown)]
/// DELETE /v1/admin/models/{model_id} — evict a model from VRAM.
///
/// - 200 OK       — model evicted; VRAM freed
/// - 404 Not Found — model ID is not in the registry
/// - 409 Conflict  — model is not in a state that can be evicted (not loaded, already evicting)
/// - 503 Service Unavailable — model manager not initialised, or the GPU context is poisoned
///   (eviction refused — dropping a resident pipeline on a poisoned context can crash the
///   process; see the project's internal engineering log)
/// - 500 Internal Server Error — engine thread panicked during eviction
///
/// Eviction blocks until the engine thread has drained all in-flight requests
/// and exited. In-flight requests receive their remaining tokens before the
/// model is freed.
pub async fn admin_unload_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Response {
    let Some(mm) = state.model_manager.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model manager is not initialised",
        )
            .into_response();
    };

    match mm.evict_model(&model_id).await {
        Ok(()) => (StatusCode::OK, format!("model '{model_id}' evicted")).into_response(),
        Err(e) => {
            if let Some(model_err) = e.downcast_ref::<ModelError>() {
                match model_err {
                    ModelError::NotFound(_) => {
                        return (StatusCode::NOT_FOUND, e.to_string()).into_response();
                    }
                    ModelError::NotLoaded | ModelError::Evicting | ModelError::Loading => {
                        return (StatusCode::CONFLICT, e.to_string()).into_response();
                    }
                    // L3 gate, eviction side (2026-08-04,
                    // dev/autotest/20260804_gpu_poisoned_eviction_crash.md):
                    // begin_evict now refuses eviction while the GPU context is
                    // poisoned, so this arm is reachable — mapped the same way
                    // model_error_response maps it for the load path.
                    ModelError::GpuPoisoned => {
                        return (StatusCode::SERVICE_UNAVAILABLE, e.to_string()).into_response();
                    }
                    // WrongKind/FilesMissing/FilesIncomplete never originate
                    // from evict_model — completeness is only checked before a load.
                    ModelError::WrongKind(_)
                    | ModelError::FilesMissing(_)
                    | ModelError::FilesIncomplete(..) => {}
                }
            }
            tracing::error!(model_id, error = %e, "admin_unload_model failed");
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
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
/// Evicts from VRAM first if the model is currently `Ready`.
///
/// - 200 OK        — model deregistered (and evicted first, if it was loaded)
/// - 404 Not Found — model ID is not in the registry
/// - 409 Conflict  — a load or eviction is already in flight for this model
/// - 503 Service Unavailable — model manager not initialised
/// - 500 Internal Server Error — eviction failed unexpectedly
pub async fn admin_deregister_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Response {
    let Some(mm) = state.model_manager.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model manager is not initialised",
        )
            .into_response();
    };

    match mm.deregister_model(&model_id).await {
        Ok(()) => (StatusCode::OK, format!("model '{model_id}' deregistered")).into_response(),
        Err(e) => {
            if let Some(model_err) = e.downcast_ref::<ModelError>() {
                match model_err {
                    ModelError::NotFound(_) => {
                        return (StatusCode::NOT_FOUND, e.to_string()).into_response();
                    }
                    ModelError::Loading | ModelError::Evicting => {
                        return (StatusCode::CONFLICT, e.to_string()).into_response();
                    }
                    ModelError::GpuPoisoned => {
                        return (
                            StatusCode::SERVICE_UNAVAILABLE,
                            format!(
                                "{e} — restart rustedvino to recover (a graceful stop via \
                                 SIGTERM/systemctl drains in-flight requests and frees VRAM first)"
                            ),
                        )
                            .into_response();
                    }
                    // deregister_model only ever evicts a Ready model — NotLoaded,
                    // WrongKind, FilesMissing, and FilesIncomplete never originate
                    // from that path.
                    ModelError::NotLoaded
                    | ModelError::WrongKind(_)
                    | ModelError::FilesMissing(_)
                    | ModelError::FilesIncomplete(..) => {}
                }
            }
            tracing::error!(model_id, error = %e, "admin_deregister_model failed");
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
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
    /// `max_prompt_len`, `speculative`).
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
/// `vram_gb`/`kv_cache_gb`/`max_concurrent_streams`/`max_prompt_len`/
/// `speculative` persist immediately but only take effect at this model's
/// *next* load. Everything else (`pinned`/`priority`/`evictable`/`load`/
/// `reasoning_parser`/`capabilities`/`eviction_grace_secs`/image-provenance
/// fields) is live the moment this call returns.
///
/// - 200 OK — JSON [`PatchModelResponse`]
/// - 400 Bad Request — `kind` supplied (immutable), empty body, invalid
///   `vram_gb`/`device`/policy field, or a policy field fails validation
/// - 404 Not Found — model ID is not registered
/// - 409 Conflict — model is `Loading`/`Evicting`, or `device`/
///   `tier_preference` supplied while `Ready`
/// - 503 Service Unavailable — model manager not initialised, or the server
///   was started without a durable config file
pub async fn admin_patch_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    Json(patch): Json<crate::model_manager::config::ModelEntryPatch>,
) -> Response {
    let Some(mm) = state.model_manager.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model manager is not initialised",
        )
            .into_response();
    };

    match mm.patch_model(&model_id, &patch) {
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
                    ModelError::NotFound(_) => {
                        return (StatusCode::NOT_FOUND, e.to_string()).into_response();
                    }
                    ModelError::Loading | ModelError::Evicting => {
                        return (StatusCode::CONFLICT, e.to_string()).into_response();
                    }
                    ModelError::GpuPoisoned
                    | ModelError::NotLoaded
                    | ModelError::WrongKind(_)
                    | ModelError::FilesMissing(_)
                    | ModelError::FilesIncomplete(..) => {} // never originate from patch_model
                }
            }
            let msg = e.to_string();
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
    /// `0.0` when VRAM gating is disabled for the domain.
    free_gb: f64,
    /// Residents that would be evicted (in eviction order) to admit the model.
    /// Empty when it already fits or cannot be admitted at all.
    would_evict: Vec<String>,
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
pub async fn admin_check_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Response {
    let Some(mm) = state.model_manager.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model manager is not initialised",
        )
            .into_response();
    };

    match mm.check_admission(&model_id) {
        Ok(check) => (
            StatusCode::OK,
            Json(AdmissionCheckResponse {
                model: model_id,
                fits: check.fits,
                needed_gb: check.needed_gb,
                free_gb: check.free_gb,
                would_evict: check.would_evict,
                ram_needed_gb: check.ram_needed_gb,
                ram_avail_gb: check.ram_avail_gb,
                already_loaded: check.already_loaded,
            }),
        )
            .into_response(),
        Err(e) => {
            if let Some(ModelError::NotFound(_)) = e.downcast_ref::<ModelError>() {
                return (StatusCode::NOT_FOUND, e.to_string()).into_response();
            }
            tracing::error!(model_id, error = %e, "admin_check_model failed");
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
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
/// Fires the session's `CancellationToken`, which causes the pipeline loop to
/// exit on its next iteration. The WebSocket closes cleanly (no SIGKILL).
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
/// `reasoning_parser`, `max_prompt_len`, `speculative`, and the image-gen Tier
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
    /// `"tts"`, `"image_gen"`, `"reranking"`. Omit for auto-detection.
    #[serde(default)]
    pub kind: Option<String>,
    /// The full per-model policy (device override, co-residency knobs,
    /// image-gen provenance, …) — see the struct doc comment above.
    #[serde(flatten)]
    pub policy: crate::model_manager::config::ModelPolicy,
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
    Json(req): Json<AddModelRequest>,
) -> Response {
    let Some(mm) = state.model_manager.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model manager is not initialised",
        )
            .into_response();
    };

    match mm
        .add_model(req.model_id.clone(), req.vram_gb, req.kind, req.policy)
        .await
    {
        Ok(()) => (
            StatusCode::OK,
            format!("model '{}' added to config and loaded", req.model_id),
        )
            .into_response(),
        Err(e) => {
            if let Some(model_err) = e.downcast_ref::<crate::model_manager::ModelError>()
                && matches!(model_err, crate::model_manager::ModelError::GpuPoisoned)
            {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!(
                        "{e} — restart rustedvino to recover \
                         (a graceful stop via SIGTERM/systemctl drains in-flight \
                         requests and frees VRAM first)"
                    ),
                )
                    .into_response();
            }
            let msg = e.to_string();
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
/// - 500 Internal Server Error — file unreadable/invalid, has deprecated
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

    match mm.reload_config() {
        Ok(report) => (
            StatusCode::OK,
            Json(ConfigReloadResponse {
                added: report.added,
                updated: report.updated,
                skipped_missing_files: report.skipped_missing_files,
                left_untouched: report.left_untouched,
                unchanged: report.unchanged,
                globals_not_applied: report.globals_not_applied,
            }),
        )
            .into_response(),
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("config_path not set") {
                return (StatusCode::SERVICE_UNAVAILABLE, msg).into_response();
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
    Json(req): Json<SetPreloadRequest>,
) -> Response {
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
            let msg = e.to_string();
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
                api_key_count: report.api_key_count,
                admin_key_count: report.admin_key_count,
            }),
        )
            .into_response(),
        Err(e) => {
            let msg = e.to_string();
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
            let msg = e.to_string();
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
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    /// `HealthResponse` serialises to the exact JSON shape clients expect.
    #[test]
    fn health_response_serialises_correctly() {
        let response = HealthResponse {
            status: "ok",
            version: "1.2.3",
            git_hash: "abc123def456",
        };
        let json = serde_json::to_value(response).unwrap();
        assert_eq!(json["status"], "ok");
        assert_eq!(json["version"], "1.2.3");
        assert_eq!(json["git_hash"], "abc123def456");
        assert_eq!(
            json.as_object().unwrap().len(),
            3,
            "HealthResponse must have exactly 3 fields"
        );
    }

    /// `AdminModelEntry` exposes `max_prompt_tokens`/`cache_size_gb` so an
    /// external client/operator can discover a model's real usable context
    /// window proactively, instead of only reactively via a `400
    /// context_length_exceeded` (`dev/autotest/
    /// 20260823_external_metrics_interface_audit.md`).
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
            max_prompt_tokens: 37_137,
            cache_size_gb: 3.0,
            native_context_limit: Some(40_960),
            kv_cache_usage_pct: 0.0,
            kv_pressure_flagged: false,
        };
        let json = serde_json::to_value(entry).unwrap();
        assert_eq!(json["max_prompt_tokens"], 37_137);
        assert!((json["cache_size_gb"].as_f64().unwrap() - 3.0).abs() < f64::EPSILON);
        assert_eq!(json["native_context_limit"], 40_960);
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
            max_prompt_tokens: 0,
            cache_size_gb: 0.0,
            native_context_limit: None,
            kv_cache_usage_pct: 0.0,
            kv_pressure_flagged: false,
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
    #[test]
    fn server_info_omits_unavailable_optional_fields() {
        let info = ServerInfo {
            host: None,
            openvino_version: None,
            engine: "RustedVINO",
            engine_version: "1.2.3",
        };
        let value = serde_json::to_value(info).unwrap();
        let obj = value.as_object().unwrap();
        let keys: std::collections::BTreeSet<&str> =
            obj.keys().map(std::string::String::as_str).collect();
        assert_eq!(
            keys,
            std::collections::BTreeSet::from(["engine", "engine_version"]),
            "host/openvino_version must be absent when None, never null: {obj:?}"
        );
    }

    /// `ServerInfo::current()` on the real running test box: non-empty engine
    /// version always present; `host`/`openvino_version` are whatever this
    /// environment actually reports (both `Some` on the fleet, both plausibly
    /// `None` in a minimal sandbox) — the point is it never panics and always
    /// produces valid JSON either way.
    #[test]
    fn server_info_current_serializes_without_panicking() {
        let value = serde_json::to_value(ServerInfo::current()).unwrap();
        assert_eq!(value["engine"], "RustedVINO");
        assert_eq!(value["engine_version"], env!("CARGO_PKG_VERSION"));
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
            kind,
            vram_gb: 0.0,
            device: "CPU".to_owned(),
            generation_defaults: crate::model_manager::template::GenerationDefaults::default(),
            max_prompt_tokens: 0,
            kv_cache_gb: 0.0,
            native_context_limit: None,
            kv_cache_usage_pct: 0.0,
            kv_pressure_flagged: false,
        }
    }

    /// A Ready `TextGen`/`Vision` model is preferred over a Ready model of a
    /// kind the probe can't drive, regardless of registration order —
    /// closes the bug an RTH scenario caught live: picking whichever model
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
            would_evict: vec!["embed".to_owned()],
            ram_needed_gb: None,
            ram_avail_gb: None,
            already_loaded: false,
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
    }

    /// UMA shape: when the RAM gate applies, both keys are present.
    #[test]
    fn admission_check_response_includes_ram_gate_on_uma() {
        let r = AdmissionCheckResponse {
            model: "qwen3.6-35b-a3b-int4-ov".to_owned(),
            fits: false,
            needed_gb: 22.0,
            free_gb: 27.0,
            would_evict: Vec::new(),
            ram_needed_gb: Some(24.0),
            ram_avail_gb: Some(23.0),
            already_loaded: false,
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
}
