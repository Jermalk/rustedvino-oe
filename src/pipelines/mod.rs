// ============================================================
// src/pipelines/ — Phase 5 media pipelines (STT / TTS / Image)
// ============================================================
// Each media modality whose request shape is request/response (one job in →
// one result out, no token stream) gets a dedicated-thread engine here, mirroring
// `crate::embed_engine`. STT (Whisper) and Image (SDXL Text2Image) are live; TTS
// lands with its own sub-plan (the project's internal engineering log).
// ============================================================

pub mod image;
pub mod piper;
pub mod stt;
pub mod tts;
