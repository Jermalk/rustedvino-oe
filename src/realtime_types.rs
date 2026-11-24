// ============================================================
// src/realtime_types.rs — shared realtime API types
// ============================================================
// Defined here (not in handlers/realtime.rs) to let admin.rs
// and app_state.rs reference them without circular imports.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

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

/// Entry in the realtime session registry: snapshot + kill and stopping tokens.
pub struct RealtimeSessionEntry {
    /// Admin-visible state of the session, updated as it progresses.
    pub snapshot: Arc<RwLock<SessionSnapshot>>,
    /// Cancel to force-close the WebSocket from outside the handler (admin
    /// `DELETE`): close frame 4000 `closed_by_admin`.
    pub kill: CancellationToken,
    /// Cancelled when the server begins a graceful shutdown (0.9.1 C5): close
    /// frame 1001 `server_stopping`. The HTTP drain does not wait for upgraded
    /// WebSocket connections, so without this signal they were simply dropped.
    pub stopping: CancellationToken,
}

/// Default idle timeout of a realtime session: with no committed utterance for
/// this long (an `inject_history`/`reload_history` batch also restarts it), the
/// server closes it with 1000 `idle_timeout`. Not configurable; tests
/// shorten it through `AppState::realtime_idle_timeout`.
pub const REALTIME_IDLE_TIMEOUT: Duration = Duration::from_mins(5);

/// How long a server-initiated close waits for the client's close reply
/// before the connection is dropped (0.9.1 C5).
pub const CLOSE_REPLY_WAIT: Duration = Duration::from_secs(2);

/// Why the server ends a realtime WebSocket session itself (0.9.1 C5).
///
/// Every server-initiated close sends an RFC 6455 close frame with
/// [`code`](Self::code) and [`reason`](Self::reason), then waits a bounded time
/// for the client's close reply. A close begun by the client, or a dropped
/// connection, has no cause and gets no frame from the server.
///
/// There is deliberately no `internal_error` (1011) variant: no server path
/// ends a session because of an internal error. Errors are reported as `error`
/// events and the session stays open; a panic ends the task, which drops the
/// socket without a close frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CloseCause {
    /// No utterance was committed for the idle timeout (5 minutes): 1000
    /// `idle_timeout`.
    IdleTimeout,
    /// The server is shutting down or restarting: 1001 `server_stopping`.
    ServerStopping,
    /// An admin force-closed the session: 4000 `closed_by_admin`.
    ClosedByAdmin,
}

impl CloseCause {
    /// The WebSocket close code sent for this cause.
    #[must_use]
    pub const fn code(self) -> u16 {
        match self {
            Self::IdleTimeout => 1000,
            Self::ServerStopping => 1001,
            Self::ClosedByAdmin => 4000,
        }
    }

    /// The machine-readable close reason sent for this cause.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::IdleTimeout => "idle_timeout",
            Self::ServerStopping => "server_stopping",
            Self::ClosedByAdmin => "closed_by_admin",
        }
    }

    /// The cause of a session end, from what ended the session loop. An admin
    /// kill wins over a server stop, which wins over the idle timeout; with none
    /// of them the client closed or the connection dropped (`None`, no frame).
    #[must_use]
    pub const fn from_signals(killed: bool, stopping: bool, idle: bool) -> Option<Self> {
        if killed {
            Some(Self::ClosedByAdmin)
        } else if stopping {
            Some(Self::ServerStopping)
        } else if idle {
            Some(Self::IdleTimeout)
        } else {
            None
        }
    }
}

/// Cancel the `stopping` token of every live realtime session, so each one
/// sends close 1001 `server_stopping` and ends, and mark the registry stopping
/// so a session that registers later starts stopped. Called from the graceful
/// shutdown path when the stop signal arrives (0.9.1 C5). Idempotent.
pub fn stop_realtime_sessions(registry: &RealtimeSessionRegistry) {
    let mut registry = registry
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    registry.stopping = true;
    for entry in registry.values() {
        entry.stopping.cancel();
    }
}

/// The live realtime sessions, keyed by session UUID, plus the server-wide
/// "stopping" flag (0.9.1 C5).
///
/// Derefs (read-only) to the `HashMap` for lookups; the only mutations are
/// [`Self::register`] and [`Self::remove`].
#[derive(Default)]
pub struct RealtimeSessions {
    entries: HashMap<Uuid, RealtimeSessionEntry>,
    /// Set once by [`stop_realtime_sessions`]; never cleared. A session that
    /// registers after it is set starts stopped (see [`Self::register`]).
    stopping: bool,
}

impl RealtimeSessions {
    /// Insert a session entry. If the server is already stopping, its
    /// `stopping` token is cancelled at once, so a session that upgraded after
    /// the stop signal still closes with 1001 instead of being dropped at exit.
    pub fn register(&mut self, id: Uuid, entry: RealtimeSessionEntry) {
        if self.stopping {
            entry.stopping.cancel();
        }
        self.entries.insert(id, entry);
    }

    /// Remove a session entry (on close). With [`Self::register`] the only
    /// mutations: there is no `DerefMut`, so nothing can insert past the
    /// stopping check.
    pub fn remove(&mut self, id: &Uuid) -> Option<RealtimeSessionEntry> {
        self.entries.remove(id)
    }

    /// Whether [`stop_realtime_sessions`] has run.
    #[must_use]
    pub const fn is_stopping(&self) -> bool {
        self.stopping
    }
}

impl std::ops::Deref for RealtimeSessions {
    type Target = HashMap<Uuid, RealtimeSessionEntry>;
    fn deref(&self) -> &Self::Target {
        &self.entries
    }
}

/// Global registry of live realtime sessions, keyed by session UUID.
///
/// Inserted on WebSocket connect ([`RealtimeSessions::register`]), removed on close.
/// Uses `std::sync::RwLock` because no guard is held across `.await` points —
/// all critical sections are brief in-memory operations.
pub type RealtimeSessionRegistry = Arc<RwLock<RealtimeSessions>>;

/// Upper bound on how long a graceful shutdown waits for the realtime sessions
/// to finish their close handshake: an in-flight turn's abort plus
/// [`CLOSE_REPLY_WAIT`], with margin.
pub const REALTIME_STOP_WAIT: Duration = Duration::from_secs(3);

/// Wait until `registry` is empty (every session finished its close and removed
/// its entry), at most `within`. Returns `false` if sessions were still open.
///
/// The HTTP drain does not track upgraded connections, so without this wait the
/// process can exit before a session's 1001 close frame is written (0.9.1 C5).
pub async fn wait_realtime_sessions_closed(
    registry: &RealtimeSessionRegistry,
    within: Duration,
) -> bool {
    const POLL: Duration = Duration::from_millis(20);
    let deadline = tokio::time::Instant::now() + within;
    loop {
        let open = registry
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len();
        if open == 0 {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            tracing::warn!(open, "realtime sessions still open after the stop wait");
            return false;
        }
        tokio::time::sleep(POLL).await;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn close_cause_codes_and_reasons_match_the_c5_table() {
        assert_eq!(CloseCause::IdleTimeout.code(), 1000);
        assert_eq!(CloseCause::IdleTimeout.reason(), "idle_timeout");
        assert_eq!(CloseCause::ServerStopping.code(), 1001);
        assert_eq!(CloseCause::ServerStopping.reason(), "server_stopping");
        assert_eq!(CloseCause::ClosedByAdmin.code(), 4000);
        assert_eq!(CloseCause::ClosedByAdmin.reason(), "closed_by_admin");
    }

    #[test]
    fn close_cause_priority_admin_then_stop_then_idle() {
        use CloseCause::{ClosedByAdmin, IdleTimeout, ServerStopping};
        assert_eq!(CloseCause::from_signals(false, false, false), None);
        assert_eq!(
            CloseCause::from_signals(false, false, true),
            Some(IdleTimeout)
        );
        assert_eq!(
            CloseCause::from_signals(false, true, true),
            Some(ServerStopping)
        );
        assert_eq!(
            CloseCause::from_signals(true, true, true),
            Some(ClosedByAdmin)
        );
        assert_eq!(
            CloseCause::from_signals(true, false, false),
            Some(ClosedByAdmin)
        );
    }

    #[tokio::test]
    async fn stop_realtime_sessions_cancels_every_stopping_token() {
        let registry: RealtimeSessionRegistry = Arc::default();
        let stops: Vec<CancellationToken> = (0..2).map(|_| CancellationToken::new()).collect();
        for stop in &stops {
            let snapshot = SessionSnapshot {
                id: Uuid::new_v4(),
                conn_id: 0,
                connected_at: 0,
                stt_model: None,
                llm_model: None,
                tts_model: None,
                tts_voice: None,
                language: None,
                system_prompt: None,
                barge_in_phrase: None,
                embed_model: None,
                turn_count: 0,
                history_entries: Vec::new(),
            };
            let entry = RealtimeSessionEntry {
                snapshot: Arc::new(RwLock::new(snapshot)),
                kill: CancellationToken::new(),
                stopping: stop.clone(),
            };
            registry.write().unwrap().register(Uuid::new_v4(), entry);
        }
        stop_realtime_sessions(&registry);
        assert!(stops.iter().all(CancellationToken::is_cancelled));

        // The stop wait returns once the sessions are gone, and is bounded.
        let short = Duration::from_millis(100);
        assert!(!wait_realtime_sessions_closed(&registry, short).await);
        let remover = {
            let registry = Arc::clone(&registry);
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(50)).await;
                let mut reg = registry.write().unwrap();
                let ids: Vec<Uuid> = reg.keys().copied().collect();
                for id in &ids {
                    reg.remove(id);
                }
            })
        };
        assert!(wait_realtime_sessions_closed(&registry, Duration::from_secs(2)).await);
        remover.await.unwrap();
    }

    #[test]
    fn a_session_registered_after_the_stop_starts_stopped() {
        let registry: RealtimeSessionRegistry = Arc::default();
        let entry = |stopping: &CancellationToken| RealtimeSessionEntry {
            snapshot: Arc::new(RwLock::new(SessionSnapshot {
                id: Uuid::new_v4(),
                conn_id: 0,
                connected_at: 0,
                stt_model: None,
                llm_model: None,
                tts_model: None,
                tts_voice: None,
                language: None,
                system_prompt: None,
                barge_in_phrase: None,
                embed_model: None,
                turn_count: 0,
                history_entries: Vec::new(),
            })),
            kill: CancellationToken::new(),
            stopping: stopping.clone(),
        };
        let early = CancellationToken::new();
        registry
            .write()
            .unwrap()
            .register(Uuid::new_v4(), entry(&early));
        assert!(!early.is_cancelled(), "not stopping yet");
        assert!(!registry.read().unwrap().is_stopping());

        stop_realtime_sessions(&registry);
        assert!(early.is_cancelled());
        assert!(registry.read().unwrap().is_stopping());

        // The race the flag closes: an upgrade that registers after the stop.
        let late = CancellationToken::new();
        registry
            .write()
            .unwrap()
            .register(Uuid::new_v4(), entry(&late));
        assert!(late.is_cancelled(), "a late session must start stopped");
        assert_eq!(registry.read().unwrap().len(), 2);
    }
}
