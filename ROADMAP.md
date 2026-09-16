# RustedVINO — Roadmap

| Phase | Status | What |
|---|---|---|
| 0 | ✅ | Scaffold, `/health`, `/v1/models` |
| 1 | ✅ | SSE streaming, C++ OV bridge, ContinuousBatching engine, benchmark |
| 2 | ✅ | ModelManager, CB engine, fixed KV pool + HTTP 429 backpressure, LRU eviction, STOP/LENGTH finish_reason |
| 3 | ✅ | Full chat — non-streaming, sampling params, `<think>` strip, tool/function calling, Prometheus metrics, tokenizer bridge, admission wait-queue, `/health_generate` |
| 4 | ~~retired~~ | ~~PostgreSQL observability~~ — Prometheus endpoint is sufficient; DB coupling rejected |
| 5 | 🚧 | Media modalities (STT/TTS/embeddings/rerank/VLM/image gen/edit) ✅ · NPU LLM engine + configurable `MAX_PROMPT_LEN` ✅ · Structured output + KV prefix caching ✅ · Realtime voice WS (`/v1/realtime`) ✅ · Crash handler + `--supervise` auto-restart ✅ · Runtime model lifecycle (add/deregister/config reload/audit) ✅ · Cross-pipeline device admission (`device_budgets`) ✅ for chat/completions/embeddings/STT/TTS and the realtime voice turn ✅ · server-managed, size-bounded self-pruning OV compile cache ✅ (`ov_cache_max_gb`, swept every `ov_cache_sweep_interval_secs`) |
| 6 | — | Mesh-routing library — planned, part of the closed-source Bear Edition tier (see [`README.md`](README.md)'s License section), not this repo |
