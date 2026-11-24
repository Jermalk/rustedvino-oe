//! `rv` — `RustedVINO` admin CLI, Rust port of `scripts/rv`.
//!
//! A client for the admin API: it ports `scripts/rv`, verb for verb. This
//! crate must stay free of any dependency on the `rustedvino` library: that crate's `build.rs` links
//! `OpenVINO` into every target, and `rv status` has to work on a box whose
//! `OpenVINO` runtime is broken or absent.

pub mod ansi;
pub mod api;
pub mod commands;
pub mod ctx;
pub mod format;
pub mod kind;
pub mod metrics;
pub mod order;
pub mod proc;
pub mod resolve;
pub mod sampling;
