// CRASH COURSE — Rust modules:
//
//   In Python, a directory with __init__.py is a package.
//   In Rust, a directory of source files is a *module tree*.
//   This file (mod.rs) is the "index" for the `handlers` module —
//   it declares which submodules exist, making them visible to the rest
//   of the crate.
//
//   `pub mod admin` = "there is a file src/handlers/admin.rs and it is
//   publicly accessible as `crate::handlers::admin`."
//
//   Private by default: if you write `mod admin` (no `pub`), only code
//   inside `handlers/` can see it. `pub` opens it upward.

/// Admin, health, and metrics route handlers.
pub mod admin;

/// Chat completions handler (`POST /v1/chat/completions`).
pub mod chat;

/// Legacy text-completions handler (`POST /v1/completions`).
pub mod completions;

/// Embeddings handler (`POST /v1/embeddings`).
pub mod embeddings;

/// Shared `OpenAI`-compatible error envelope (`openai_error`).
pub mod error;

/// Phase 5 media handlers (`/v1/audio/transcriptions`, `/v1/audio/speech`,
/// `/v1/images/generations`). 5.0 scaffold: all three return 501.
pub mod media;

/// Tokenizer helper handlers (`POST /tokenize`, `POST /detokenize`).
pub mod tokenize;

/// WebSocket realtime voice pipeline (`GET /v1/realtime`).
pub mod realtime;

/// Reranking handler (`POST /v1/rerank`).
pub mod reranking;
