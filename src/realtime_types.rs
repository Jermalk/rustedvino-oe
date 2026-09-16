// ============================================================
// src/realtime_types.rs — shared realtime API types
// ============================================================
// Defined here (not in handlers/realtime.rs) to let admin.rs
// and app_state.rs reference them without circular imports.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// One conversation turn — persisted by the client, replayed at connect time,
/// and returned by the server via `turn_history` events.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClientHistoryEntry {
    pub user: String,
    pub assistant: String,
    /// Pre-computed embedding for this turn (user + "\n" + assistant).
    /// Absent when the client did not embed; the server fills it in when an
    /// `embed_model` is configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    /// Unix timestamp in milliseconds — stable key for client-side upsert.
    pub timestamp: u64,
    /// Model ID that produced `embedding` — set whenever the server fills the
    /// vector in, or round-tripped from a client-supplied one. Absent for
    /// entries with no `embedding`, or from a pre-provenance client/entry.
    /// `retrieve_memories` gates similarity scoring on this (exact match, not
    /// just matching vector length): two different embedding models can
    /// coincidentally produce same-length vectors that live in unrelated
    /// vector spaces, so a length-only check lets that pair silently score a
    /// meaningless cosine similarity instead of being recognised as
    /// incompatible.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
    /// STT model that transcribed `user` for this turn. Provenance only, for
    /// long-term diagnostics (e.g. correlating a transcription artifact with
    /// the model that produced it) — never consulted by retrieval.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stt_model: Option<String>,
    /// LLM model that generated `assistant` for this turn. Provenance only,
    /// same rationale as `stt_model`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_model: Option<String>,
}

/// Serialisable view of a live realtime session, exposed by the admin API.
#[derive(Clone, Serialize)]
pub struct SessionSnapshot {
    pub id: Uuid,
    pub conn_id: u64,
    /// Unix milliseconds at WebSocket connection open.
    pub connected_at: u64,
    pub stt_model: Option<String>,
    pub llm_model: Option<String>,
    pub tts_model: Option<String>,
    pub tts_voice: Option<String>,
    pub language: Option<String>,
    pub system_prompt: Option<String>,
    pub barge_in_phrase: Option<String>,
    pub embed_model: Option<String>,
    pub turn_count: usize,
    pub history_entries: Vec<ClientHistoryEntry>,
}

/// Entry in the realtime session registry: snapshot + kill token.
pub struct RealtimeSessionEntry {
    pub snapshot: Arc<RwLock<SessionSnapshot>>,
    /// Cancel to force-close the WebSocket from outside the handler.
    pub kill: CancellationToken,
}

/// Global registry of live realtime sessions, keyed by session UUID.
///
/// Inserted on WebSocket connect, removed on close.
/// Uses `std::sync::RwLock` because no guard is held across `.await` points —
/// all critical sections are brief in-memory operations.
pub type RealtimeSessionRegistry = Arc<RwLock<HashMap<Uuid, RealtimeSessionEntry>>>;
