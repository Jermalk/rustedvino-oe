# Changelog

All notable changes to RustedVINO are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and from 0.7.0 on the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (0.6.8 was chosen for luck, not as a
semver claim). Full release notes, with measurements and download instructions, are on each
[GitHub release](https://github.com/Jermalk/rustedvino-oe/releases).

## [0.7.0] — 2026-09-30

The NPU release: Lunar Lake's NPU becomes a fully configurable text-generation target, and several
NPU responses that were quietly wrong are now right.

### Added
- **`POST /v1/audio/translations`** (OpenAI-compatible): speech in any Whisper language → English
  text, same request/response formats as transcriptions. Needs a Whisper model trained for
  translation (`whisper-large-v3` works; `large-v3-turbo` returns untranslated text). English-only
  (`.en`) models are refused with `400 model_cannot_translate`.
- **Piper TTS voices** (VITS in ONNX, e.g. Vietnamese `vi_VN-vais1000`) on `/v1/audio/speech` and
  `/v1/realtime`. Phonemes come from `espeak-ng`, run as an external program the operator installs
  (GPL-3.0, never linked or shipped); a load-time self-check refuses the voice if `espeak-ng` is
  missing or produces the wrong phonemes.
- **NPU `min_response_len`** (per model): the answer room the NPU's fixed cache reserves on top of
  `max_prompt_len`. Settable in config or via `PATCH /v1/admin/models/{id}`.
- **NPU LLMs use `ov_cache_dir`**: one weightless compiled blob per model (~0.66 GB for an 8B int4
  model) — cached loads in ~6 s instead of ~30 s cold, counted against `ov_cache_max_gb`.
- **Embedding limits of their own**: `max_embedding_inputs` (default 256) and
  `max_embedding_batch_tokens` (default 32768 padded tokens, which bounds GPU memory per batch).
- **Measured device memory**: `/metrics` gauges `rustedvino_process_gpu_memory_bytes` and
  `rustedvino_model_gpu_memory_estimate_bytes`, read from the kernel on Linux;
  `GET /v1/admin/models` shows `gpu_memory_estimate_gb`, and a warning is logged when a model's
  measured memory exceeds its configured `vram_gb`.
- **`tools/livecaptions`**: live captions from a microphone or any PulseAudio/PipeWire source,
  through `/v1/audio/transcriptions`, optionally translated by a chat model.
- **`tools/semsearch`**: semantic search over a folder of Markdown notes using the server's own
  `/v1/embeddings` and optional `/v1/rerank`.
- `413 request_too_large` for bodies over the 50 MiB limit, and a "Request limits and endpoint
  behaviour" section in `INTERFACE.md` (upload cap, long audio, embedding limits, realtime audio
  format).

### Changed
- Requests without a `seed` now really sample: each unseeded request gets a fresh random seed on
  every text engine (GPU, NPU, vision). Previously identical requests at `temperature > 0` returned
  identical text. An explicit `seed` is still reproducible, with one caveat: with prefix caching on
  (the default), the first request for a prompt can differ from its later repeats, which reuse its
  cached prompt blocks. Set `enable_prefix_caching: false` for strict reproducibility.
- NPU LLMs are no longer charged a GPU-style KV cache when the server checks whether a model fits
  (an 8B int4 NPU model was counted at ~12.5 GB against ~5 GB actually used). The admin API reports
  the NPU's real prompt limit as `max_prompt_tokens`.
- `POST /v1/admin/config/reload` picks up file changes to next-load settings (`max_prompt_len`,
  `min_response_len`, `kv_cache_gb`, `max_concurrent_streams`, `speculative`) for models that aren't
  loaded, and no longer reports unchanged Qwen3 models as `updated`.
- `/v1/embeddings` no longer shares the completions array cap (16). Every request valid under
  0.6.8 stays within the new defaults.

### Fixed
- **NPU:** `enable_thinking: false` is honoured (the prompt was chat-templated twice, dropping the
  "thinking off" prefill).
- **NPU:** sampling settings are applied — `temperature`, `top_p`, `top_k`, `seed`, penalties and
  `stop`. Every NPU request used to run greedy while reporting the requested values.
- **NPU:** structured output (`response_format` with a JSON schema) works.
- **NPU:** `usage.completion_tokens` is exact, and streamed text is sent as soon as it is decoded.
- **NPU:** an answer cut short by the fixed cache reports `finish_reason: "length"`, not `"stop"`;
  one ending exactly at the limit reports `"stop"`.
- **NPU:** the prompt-length check follows the length the model was actually compiled with, so an
  over-long prompt after a PATCH gets a clean `400` naming `max_prompt_len`, not OpenVINO's raw
  error.
- Clearing `max_prompt_len`, `min_response_len` or `max_concurrent_streams` via PATCH restores the
  default on the next load instead of keeping the old value until a restart.
- The GPU text path's rare string fallback no longer risks double chat-templating.
- Embedding length limit for XLM-RoBERTa models (multilingual-e5, bge-reranker-v2-m3) is 512 / 8192
  tokens, not 514 / 8194 — the extra tokens were embedded with untrained positions.
- Uploads over the body limit used to fail as `400` "could not read field 'file'" (multipart) or
  "malformed request body" (JSON).

### Security
- The `x-ruvi-host` header (the box's hostname) was sent on every response, including the open
  `/health` and `401`s. It now goes only to requests with a valid inference or admin key, and a
  server with no keys configured never sends it. The same rule applies to `host` in image
  `generation_metadata` and in `GET /v1/admin/models`. **Upgrade note:** tooling that reads the
  header must send a valid key.

## [0.6.8] — 2026-09-20

First public release: an OpenAI-compatible inference server for Intel hardware, written in Rust
as the successor to the Python stormVINO.

- Serves chat completions (streaming and non-streaming), embeddings, reranking, speech-to-text,
  text-to-speech, image generation, vision, a realtime voice WebSocket, an admin API and Prometheus
  metrics.
- Runs on Arc dGPUs (B50 / B60 / B70), Lunar Lake's integrated GPU, the NPU and CPU, with
  capability-based placement instead of hardcoded device names.
- Continuous batching through OpenVINO GenAI's `ContinuousBatchingPipeline`: on an Arc Pro B60,
  69 tok/s for one user and 814 tok/s aggregate for 16 users.
- Self-contained Linux bundle with the OpenVINO 2026.2.1 runtime inside (glibc ≥ 2.39).

[0.7.0]: https://github.com/Jermalk/rustedvino-oe/compare/v0.6.8...v0.7.0
[0.6.8]: https://github.com/Jermalk/rustedvino-oe/releases/tag/v0.6.8
