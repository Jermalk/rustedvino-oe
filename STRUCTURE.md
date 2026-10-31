# RustedVINO — Project Structure

Module-to-purpose map (`src/`, `ov_bridge/`, `scripts/rv-cargo.sh`/`rv`, `tools/`, `tests/`). For prose (why,
what's implemented, setup) see **[`README.md`](README.md)**; for the HTTP surface see
**[`INTERFACE.md`](INTERFACE.md)**.

```
src/
  main.rs                    — entry point; parses CLI flags (--supervise, --list-devices, --device-info, --print-default-config, --print-default-model, --version/-V, --gpu-probe) and dispatches
  startup.rs                 — boot sequence: resolve/load config, load keys file, bind gate, construct ModelManager, spawn the OV-cache sweep task
  lib.rs                     — router, module declarations, auth + header middleware
  app_state.rs               — Arc<AppState> (holds ModelManager)
  device_inventory.rs        — DeviceInventory: probe + tier-based placement
  cache_manifest.rs          — per-model cache manifest (JSON, ~/.cache/rustedvino/cache_manifest/): model_hash memoization and OV blob attribution
  log_ring.rs                — in-memory log ring buffer backing GET /v1/admin/logs/tail
  model_completeness.rs      — backs GET /v1/admin/models/completeness (missing directory/tokenizer file detection)
  tts_normalize.rs           — text normalization before TTS synthesis (markdown stripping, language-specific rules)
  supervisor.rs              — `--supervise` crash auto-restart wrapper (backoff + cap, Unix only)
  gpu_probe.rs               — `--gpu-probe`: the small GPU readiness check the supervisor runs before restarting a worker that exited without teardown
  gpu_driver.rs              — identifies the GPU driver (OpenCL runtime, kernel driver) in use, for the startup log; names the recovery step for a stalled GPU load
  shutdown_signals.rs        — SIGTERM/SIGINT taken on one dedicated thread (GPU driver libraries install their own signal handlers)
  ffi_watch.rs               — registry of long OpenVINO calls in flight (loads, CB steps, VLM generations, transcriptions), read at process exit (exit 76 instead of running C++ destructors under a live call) and by the generation watchdog
  fault_injection.rs         — test-only GPU-poison injection (cargo feature `fault-injection`; never in release bundles)
  crash_handler.rs           — installs a SIGSEGV/SIGABRT/SIGBUS/SIGILL handler that logs a backtrace (Unix only)
  voice_pin.rs               — realtime voice-flow pin: shared STT/LLM/TTS model set across sessions
  realtime_types.rs          — WS message/event types shared by the realtime handler
  handlers/
    mod.rs                   — handler module declarations
    admin.rs                 — /health, /health_generate, /v1/models, /v1/admin/* (load/check/evict/add/register/reload/keys-reload/audit/completeness/logs-tail/watchdog/voice-pin/realtime-sessions)
    chat.rs                  — /v1/chat/completions (streaming + non-streaming; VLM image input or text-only)
    completions.rs           — /v1/completions (legacy raw-prompt)
    embeddings.rs            — /v1/embeddings
    reranking.rs             — /v1/rerank (cross-encoder document reranking)
    media.rs                 — /v1/audio/transcriptions, /v1/audio/translations, /v1/audio/speech, /v1/images/*
    tokenize.rs              — /tokenize, /detokenize
    systemone.rs             — POST /v1/systemone (Ollama-compatible typed decisions)
    realtime.rs              — /v1/realtime WebSocket voice-flow handler (STT → LLM → TTS)
    error.rs                 — OpenAI-compatible error envelope helpers
  streaming.rs               — mpsc → SSE encoder; StreamEvent types
  admission/
    mod.rs                   — DeviceBudgets/WorkLease: cross-pipeline device admission gate (disabled by default)
  model_manager/
    mod.rs                   — ModelManager: state machine, LRU, VRAM gating
    config.rs                — Config struct, ModelPolicy, ReasoningParser, SupervisorConfig, JSON load
    vram.rs                  — VramTracker (internal VRAM accounting, domain budgets)
    lifecycle.rs             — EngineFactory trait, OvEngineFactory, MockEngineFactory
    engine.rs                — ModelKind enum, EngineHandleKind
    stub_engines.rs          — stub engine factories for testing (non-CB model kinds)
    placement.rs             — device tier resolution (NPU / iGPU / dGPU / CPU preference lists)
    template.rs              — chat_template.jinja loader; ModelFamily detection
    kv_profile.rs            — per-model KV-cache layout (attention dimensions, linear-attention state, 4-bit weights)
    kv_precision.rs          — resolves each model's KV-cache precision and the string passed to the plugin (also names the kv-<tag> cache directory)
    kv_capacity.rs           — KV pool capacity in whole blocks, mirroring GenAI's block math; the exact context limit
    kv_ratchet.rs            — learned context-limit ceiling, stored with everything that determined the capacity it was observed under
    kv_headroom.rs           — device-headroom limit of a prompt (free GPU memory after weights and pools)
    headroom.rs              — runtime side of the headroom limit: per-model facts, cached memory probe, transient leases
    image_footprint.rs       — system-memory charge of CPU image generation beyond the weights
    poison_diag.rs           — memory-state log line written when a GPU context is poisoned
  ov_cb.rs                   — OvCbEngine: safe Rust wrapper over CB C bridge
  cb_engine.rs               — CB engine thread + EngineHandle (mpsc, Semaphore admission queue)
  npu_engine.rs              — NPU LLM engine: static LLMPipeline single-stream path (Lunar Lake)
  vlm_engine.rs              — VLM engine: VLMPipeline single-stream (qwen3-vl, qwen2.5-vl, gemma4, internvl)
  embed_engine.rs            — Embedding engine: TextEmbeddingPipeline
  rerank_engine.rs           — Reranking engine: cross-encoder pipeline
  decision_engine.rs         — Decision engine thread for /v1/systemone (one prefill per question)
  ov_embed.rs                — Safe Rust wrapper for OV text embedding C bridge
  ov_rerank.rs               — Safe Rust wrapper for OV reranking C bridge
  ov_whisper.rs              — Safe Rust wrapper for OV Whisper C bridge
  ov_image.rs                — Safe Rust wrapper for OV image pipeline C bridge
  ov_vlm.rs                  — Safe Rust wrapper for OV VLM C bridge
  ov_decide.rs               — Safe Rust wrapper for the decision (/v1/systemone) C bridge
  ov_tokenizer.rs            — standalone thread-safe tokenizer, so /tokenize and /detokenize need not wait for the engine thread
  ov_tts.rs                  — Safe Rust wrapper for OV TTS C bridge (SpeechT5 only; Kokoro and VITS are pure-Rust ORT paths in pipelines/tts.rs)
  ov_pipeline.rs             — Shared OV pipeline helpers
  pipelines/
    mod.rs                   — Pipeline trait definitions
    stt.rs                   — STT engine thread + SttHandle
    tts.rs                   — TTS engine thread + TtsHandle (Kokoro-82M, Coqui VITS, Piper, SpeechT5)
    piper.rs                 — Piper VITS voices: voice config, espeak-ng phonemizer (run as a separate program), phoneme ids
    image.rs                 — Image engine thread + ImageHandle (SDXL, FLUX)
  systemone/
    mod.rs                   — /v1/systemone wire format (Ollama v0.35.0-compatible), the model-free half
    wire.rs                  — decodes a request body the way Ollama's Go decoder does
    compile.rs               — question validation, answer codes and the per-question prompt
    gojson.rs                — Go `encoding/json` byte behaviour where the prompt depends on it
    answer.rs                — candidate logits → probabilities → response
  image_util.rs              — Base64 encode/decode, image format helpers
  in_flight.rs               — In-flight request drain tracker
  os_memory.rs               — System RAM probing (for UMA budget)
  gpu_memory.rs              — Measured device memory from /proc/self/fdinfo (Linux DRM: xe/i915/intel_vpu) + measurement windows for per-model estimates
  metrics.rs                 — Prometheus registry, counters, histograms
  prompt_builder.rs          — Chat template rendering (minijinja), tool-call parsing, ThinkFilter
  bin/
    fake_worker.rs           — deterministic-crash test binary for supervisor integration tests
    supervise_harness.rs     — test harness driving `--supervise` end-to-end

ov_bridge/
  ov_bridge.cpp              — C wrapper over C++ ContinuousBatchingPipeline + LLMPipeline + …
  include/                   — OpenVINO GenAI C++ headers (vendored @ 2026.4.1.0)

scripts/
  rv-cargo.sh                — run any cargo command with the OpenVINO build/runtime env wired in (auto-detects the OV install, or set RV_OV_VENV)
  rv                         — the original Ollama-shaped admin CLI (stdlib-only Python); kept as a reference and alternative — the native port is tools/rv

tools/
  rv/                        — the admin CLI (Rust port of scripts/rv, plus start/stop/restart/version/resize/completeness/reload-keys/voice-pin/sessions); a separate crate with no OpenVINO dependency, shipped in the release bundles
  bench/                     — rvbench.py (per-model figures), cmpbench.py + cmptable.py (RustedVINO vs OVMS vs OpenArc on your card; see CMPBENCH.md), stdlib-only Python
  semsearch/                 — semantic search over Markdown notes (/v1/embeddings, /v1/rerank)
  livecaptions/              — live captions + optional translation (/v1/audio/transcriptions, /v1/chat/completions)

tests/
  api.rs                     — integration tests, mock path, no GPU needed
  supervisor.rs              — `--supervise` crash-and-restart integration tests
  golden_qwen3_template.rs   — golden chat-template rendering tests (tool calls, history)
  config_reference.rs        — CONFIG.md must document every setting the server accepts
  shipped_configs.rs         — every shipped config writes out every setting (no hidden defaults)
  gpu_poison_exit.rs         — first GPU-context poisoning under `--supervise` requests the worker exit (exit code 75)
  gpu_poison_exit_supervised.rs — companion of gpu_poison_exit.rs (separate binary: process-global state)
  rv_keys_path_parity.rs     — tools/rv's copy of the keys-file path resolution matches the server's
  rv_stop_contract.rs        — rv's copies of the server's stop log texts and timeouts, and the example unit's TimeoutStopSec, stay in step
  systemone_golden.rs        — /v1/systemone wire parity with Ollama v0.35.0 (fixtures generated by Ollama's own code)
  systemone_engine.rs        — `#[ignore]`d live check of the decision bridge against reference logits (needs a decision model)
  fixtures/                  — shared test fixtures
```
