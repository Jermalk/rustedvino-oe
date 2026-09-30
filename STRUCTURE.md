# RustedVINO — Project Structure

Module-to-purpose map (`src/`, `ov_bridge/`, `scripts/rv-cargo.sh`/`rv`, `tools/`, `tests/`). For prose (why,
what's implemented, setup) see **[`README.md`](README.md)**; for the HTTP surface see
**[`INTERFACE.md`](INTERFACE.md)**.

```
src/
  main.rs                    — entry point; parses CLI flags (--supervise, --list-devices, --device-info) and dispatches
  startup.rs                 — boot sequence: resolve/load config, load keys file, bind gate, construct ModelManager, spawn the OV-cache sweep task
  lib.rs                     — router, module declarations, auth + header middleware
  app_state.rs               — Arc<AppState> (holds ModelManager)
  device_inventory.rs        — DeviceInventory: probe + tier-based placement
  cache_manifest.rs          — per-model cache manifest (JSON, ~/.cache/rustedvino/cache_manifest/): model_hash memoization and OV blob attribution
  log_ring.rs                — in-memory log ring buffer backing GET /v1/admin/logs/tail
  model_completeness.rs      — backs GET /v1/admin/models/completeness (missing directory/tokenizer file detection)
  tts_normalize.rs           — text normalization before TTS synthesis (markdown stripping, language-specific rules)
  supervisor.rs              — `--supervise` crash auto-restart wrapper (backoff + cap, Unix only)
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
  ov_cb.rs                   — OvCbEngine: safe Rust wrapper over CB C bridge
  cb_engine.rs               — CB engine thread + EngineHandle (mpsc, Semaphore admission queue)
  npu_engine.rs              — NPU LLM engine: static LLMPipeline single-stream path (Lunar Lake)
  vlm_engine.rs              — VLM engine: VLMPipeline single-stream (qwen3-vl, qwen2.5-vl, gemma4, internvl)
  embed_engine.rs            — Embedding engine: TextEmbeddingPipeline
  rerank_engine.rs           — Reranking engine: cross-encoder pipeline
  ov_embed.rs                — Safe Rust wrapper for OV text embedding C bridge
  ov_rerank.rs               — Safe Rust wrapper for OV reranking C bridge
  ov_whisper.rs              — Safe Rust wrapper for OV Whisper C bridge
  ov_image.rs                — Safe Rust wrapper for OV image pipeline C bridge
  ov_vlm.rs                  — Safe Rust wrapper for OV VLM C bridge
  ov_tts.rs                  — Safe Rust wrapper for OV TTS C bridge (SpeechT5 only; Kokoro and VITS are pure-Rust ORT paths in pipelines/tts.rs)
  ov_pipeline.rs             — Shared OV pipeline helpers
  pipelines/
    mod.rs                   — Pipeline trait definitions
    stt.rs                   — STT engine thread + SttHandle
    tts.rs                   — TTS engine thread + TtsHandle (Kokoro-82M, Coqui VITS, SpeechT5)
    image.rs                 — Image engine thread + ImageHandle (SDXL, FLUX)
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
  include/                   — OpenVINO GenAI C++ headers (vendored @ 2026.2.0)

scripts/
  rv-cargo.sh                — run any cargo command with the OpenVINO build/runtime env wired in (auto-detects the OV install, or set RV_OV_VENV)
  rv                         — Ollama-shaped admin CLI (status/list/ps/check/add/load/unload/deregister/logs); stdlib-only Python, no venv needed

tools/                       — stdlib-only Python clients of the server's API
  semsearch/                 — semantic search over Markdown notes (/v1/embeddings, /v1/rerank)
  livecaptions/              — live captions + optional translation (/v1/audio/transcriptions, /v1/chat/completions)

tests/
  api.rs                     — integration tests, mock path, no GPU needed
  supervisor.rs              — `--supervise` crash-and-restart integration tests
  golden_qwen3_template.rs   — golden chat-template rendering tests (tool calls, history)
  fixtures/                  — shared test fixtures
```
