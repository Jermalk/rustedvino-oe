// ============================================================
// src/voice_pin.rs — voice flow pin: shared model-set for realtime sessions
// ============================================================
// The first realtime session that completes a successful turn writes the pin
// (stt + llm + tts model IDs). Subsequent blank-config sessions read the pin
// via GET /v1/admin/voice-pin before connecting, rather than doing full
// discovery.
//
// Lifecycle:
//   session opens  → session_opened()  — increments active count, cancels TTL
//   session closes → session_closed()  — decrements count; starts TTL when → 0
//   TTL fires      → pin cleared; next client rediscovers
//
// First-writer-wins: try_set() is a no-op if a pin already exists. The pin is
// only set after a successful turn so the models are confirmed working.
//
// Swing mitigation: place try_set() AFTER the first successful turn, not on
// ConfigEvent receipt. This prevents an explicit-override session (loading a
// model) from pinning an unconfirmed model set before the baseline session
// completes its first turn.
// ============================================================

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use serde::Serialize;
use tokio::sync::oneshot;

/// TTL after the last session closes before the pin is cleared.
pub const VOICE_FLOW_TTL: Duration = Duration::from_mins(5);

/// A pinned set of model IDs for a realtime voice flow.
#[derive(Clone, Debug, Serialize)]
pub struct VoiceFlowPin {
    pub stt_model: String,
    pub llm_model: String,
    pub tts_model: String,
}

struct PinInner {
    pin: Option<VoiceFlowPin>,
    active_sessions: u32,
    /// Dropping this cancels the TTL timer: the spawned task's `oneshot::Receiver`
    /// resolves with `Err` and the task exits without clearing the pin.
    ttl_cancel: Option<oneshot::Sender<()>>,
    /// Realtime voice arbitration v2 (`dev/plans/realtime-voice-model-
    /// arbitration-v2.md`, D2): `model_id -> count of active sessions whose
    /// resolved {stt, llm, tts} set includes it`. `ModelManager::
    /// eviction_protected` consults `count > 0` to hard-exclude a model any
    /// live session depends on — independent of the first-writer-wins
    /// `pin` above (that's a cold-start *suggestion*; this is a hard
    /// eviction guard covering every resolved session, not just the first).
    serving_counts: HashMap<String, u32>,
}

/// Shared voice flow pin manager. Held in `AppState` behind an `Arc`.
///
/// All public methods are synchronous — the internal [`std::sync::Mutex`] is
/// never held across an `.await` point, so there is no risk of blocking the
/// async runtime.
pub struct VoicePinManager {
    inner: Arc<Mutex<PinInner>>,
    ttl: Duration,
}

impl Default for VoicePinManager {
    fn default() -> Self {
        Self::with_ttl(VOICE_FLOW_TTL)
    }
}

impl VoicePinManager {
    /// Create a manager with a custom TTL (useful in tests).
    #[must_use]
    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            inner: Arc::new(Mutex::new(PinInner {
                pin: None,
                active_sessions: 0,
                ttl_cancel: None,
                serving_counts: HashMap::new(),
            })),
            ttl,
        }
    }

    /// Called when a WebSocket session opens.
    ///
    /// Increments the active-session count and cancels any pending TTL expiry.
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned (another thread panicked while
    /// holding it — the process is in an unrecoverable state regardless).
    pub fn session_opened(&self) {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.active_sessions += 1;
        // Drop the old Sender — its paired Receiver resolves with Err, and the
        // TTL task's select! arm fires without clearing the pin.
        inner.ttl_cancel = None;
        tracing::debug!(
            active_sessions = inner.active_sessions,
            "voice pin: session opened"
        );
    }

    /// Called when a WebSocket session closes.
    ///
    /// Decrements the active-session count. When it reaches zero and a pin is
    /// set, spawns a TTL task that clears the pin after [`VOICE_FLOW_TTL`] unless
    /// a new session opens first.
    ///
    /// Must be called from within a Tokio async context so that `tokio::spawn`
    /// is available.
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned.
    pub fn session_closed(&self) {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.active_sessions = inner.active_sessions.saturating_sub(1);
        tracing::debug!(
            active_sessions = inner.active_sessions,
            "voice pin: session closed"
        );

        if inner.active_sessions == 0 && inner.pin.is_some() {
            let (ttl_tx, ttl_rx) = oneshot::channel::<()>();
            inner.ttl_cancel = Some(ttl_tx);
            let inner_arc = Arc::clone(&self.inner);
            let ttl = self.ttl;
            tokio::spawn(async move {
                tokio::select! {
                    () = tokio::time::sleep(ttl) => {
                        let mut inner = inner_arc.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                        // Guard against a session opening between sleep end and
                        // lock acquisition.
                        if inner.active_sessions == 0 {
                            tracing::info!("voice pin: TTL expired — pin cleared");
                            inner.pin = None;
                            inner.ttl_cancel = None;
                        }
                    }
                    // Sender dropped by session_opened → cancel without clearing.
                    _ = ttl_rx => {
                        tracing::debug!("voice pin: TTL cancelled by new session");
                    }
                }
            });
        }
    }

    /// Attempt to write the pin. Succeeds only when no pin is currently set
    /// (first-writer-wins). Returns `true` if the pin was newly written.
    ///
    /// Callers should invoke this only after a confirmed successful turn so that
    /// the model IDs are known to be working before they are pinned.
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned.
    pub fn try_set(&self, stt_model: String, llm_model: String, tts_model: String) -> bool {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if inner.pin.is_none() {
            tracing::info!(
                stt_model = %stt_model,
                llm_model = %llm_model,
                tts_model = %tts_model,
                "voice pin: pinned"
            );
            inner.pin = Some(VoiceFlowPin {
                stt_model,
                llm_model,
                tts_model,
            });
            true
        } else {
            false
        }
    }

    /// Clear the pin if its `llm_model` matches `model_id`.
    ///
    /// Called from `ModelManager` eviction (explicit unload, LRU, or shutdown)
    /// so a pin never outlives the model it points at — a stale pin would
    /// otherwise both mislead new blank-config clients into requesting an
    /// unloaded model, and (since `try_set` is first-writer-wins) permanently
    /// block a legitimately different model set from ever being pinned until
    /// the TTL happens to expire.
    ///
    /// A no-op when no pin is set or the evicted model isn't the pinned LLM
    /// (STT/TTS eviction does not clear the pin — only the LLM identifies the
    /// flow for this check).
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned.
    pub fn clear_if_llm(&self, model_id: &str) {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if inner.pin.as_ref().is_some_and(|p| p.llm_model == model_id) {
            tracing::info!(model_id, "voice pin: cleared — pinned LLM was evicted");
            inner.pin = None;
        }
    }

    /// Returns a snapshot of the current pin, or `None` if no pin is set.
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn get(&self) -> Option<VoiceFlowPin> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pin
            .clone()
    }

    /// Returns the current active-session count.
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn active_sessions(&self) -> u32 {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .active_sessions
    }

    /// Returns the configured TTL duration.
    #[must_use]
    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    /// D2: register `ids` as served by one more active realtime session —
    /// increments each distinct id's count by exactly one, regardless of how
    /// many times it appears in `ids` (a session's `{stt, llm, tts}` set is
    /// deduplicated first, so a session that happens to use the same model
    /// for two roles registers it once, and `deregister_serving` reverses it
    /// with one matching decrement, not two).
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned.
    pub fn register_serving<'a>(&self, ids: impl IntoIterator<Item = &'a str>) {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut seen = std::collections::HashSet::new();
        for id in ids {
            if seen.insert(id) {
                *inner.serving_counts.entry(id.to_owned()).or_insert(0) += 1;
            }
        }
    }

    /// D2: the inverse of [`register_serving`](Self::register_serving) —
    /// decrements each distinct id's count by one (floor zero via
    /// `saturating_sub`), removing the entry once it reaches zero so the map
    /// doesn't grow unbounded with stale zero-count ids across a long-running
    /// server's model churn.
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned.
    pub fn deregister_serving<'a>(&self, ids: impl IntoIterator<Item = &'a str>) {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut seen = std::collections::HashSet::new();
        for id in ids {
            if !seen.insert(id) {
                continue;
            }
            if let Some(count) = inner.serving_counts.get_mut(id) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    inner.serving_counts.remove(id);
                }
            }
        }
    }

    /// D2: how many active realtime sessions currently have `id` in their
    /// resolved `{stt, llm, tts}` set. `0` means no live session depends on
    /// it — `ModelManager::eviction_protected` hard-excludes any id with a
    /// count `> 0` from eviction.
    ///
    /// # Panics
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn serving_count(&self, id: &str) -> u32 {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .serving_counts
            .get(id)
            .copied()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn pin(llm: &str) -> (String, String, String) {
        ("stt".to_owned(), llm.to_owned(), "tts".to_owned())
    }

    #[test]
    fn try_set_is_first_writer_wins() {
        let vp = VoicePinManager::default();
        let (stt, llm, tts) = pin("llm-a");
        assert!(vp.try_set(stt, llm, tts), "first write succeeds");

        let (stt, llm, tts) = pin("llm-b");
        assert!(
            !vp.try_set(stt, llm, tts),
            "second write is a no-op while a pin is set"
        );
        assert_eq!(vp.get().unwrap().llm_model, "llm-a");
    }

    #[test]
    fn clear_if_llm_clears_on_exact_match() {
        let vp = VoicePinManager::default();
        let (stt, llm, tts) = pin("llm-a");
        vp.try_set(stt, llm, tts);

        vp.clear_if_llm("llm-a");

        assert!(vp.get().is_none(), "pin cleared — its LLM was evicted");
    }

    #[test]
    fn clear_if_llm_ignores_a_different_model() {
        let vp = VoicePinManager::default();
        let (stt, llm, tts) = pin("llm-a");
        vp.try_set(stt, llm, tts);

        // Evicting an unrelated model (or the STT/TTS side of this same pin)
        // must not touch the pin — only its LLM identifies the flow.
        vp.clear_if_llm("some-other-model");

        assert_eq!(
            vp.get().unwrap().llm_model,
            "llm-a",
            "pin survives eviction of a model it doesn't reference as LLM"
        );
    }

    #[test]
    fn clear_if_llm_is_a_no_op_without_a_pin() {
        let vp = VoicePinManager::default();
        vp.clear_if_llm("llm-a"); // must not panic
        assert!(vp.get().is_none());
    }

    #[test]
    fn clear_if_llm_unblocks_a_new_pin() {
        // The whole point of clearing (vs. just masking on read): try_set's
        // first-writer-wins gate must open back up once the stale pin is gone,
        // so a genuinely different model set can be pinned without waiting out
        // the TTL.
        let vp = VoicePinManager::default();
        let (stt, llm, tts) = pin("llm-a");
        vp.try_set(stt, llm, tts);
        vp.clear_if_llm("llm-a");

        let (stt, llm, tts) = pin("llm-b");
        assert!(
            vp.try_set(stt, llm, tts),
            "try_set succeeds again once the stale pin is cleared"
        );
        assert_eq!(vp.get().unwrap().llm_model, "llm-b");
    }

    // ---- D2: realtime serving set (dev/plans/realtime-voice-model-arbitration-v2.md) ----

    #[test]
    fn serving_count_is_zero_for_an_unregistered_model() {
        let vp = VoicePinManager::default();
        assert_eq!(vp.serving_count("model-a"), 0);
    }

    #[test]
    fn register_serving_increments_and_deregister_decrements() {
        let vp = VoicePinManager::default();
        vp.register_serving(["model-a", "model-b"]);
        assert_eq!(vp.serving_count("model-a"), 1);
        assert_eq!(vp.serving_count("model-b"), 1);

        // A second session also depends on model-a.
        vp.register_serving(["model-a"]);
        assert_eq!(vp.serving_count("model-a"), 2);

        vp.deregister_serving(["model-a"]);
        assert_eq!(
            vp.serving_count("model-a"),
            1,
            "one session's worth removed, the other still holds it"
        );
        assert_eq!(
            vp.serving_count("model-b"),
            1,
            "unaffected by model-a's count"
        );

        vp.deregister_serving(["model-a", "model-b"]);
        assert_eq!(vp.serving_count("model-a"), 0);
        assert_eq!(vp.serving_count("model-b"), 0);
    }

    #[test]
    fn register_and_deregister_serving_dedup_within_one_call() {
        // A session whose stt/llm/tts happen to share one model id must only
        // count once per (de)registration — otherwise register/deregister
        // drift out of symmetry after repeated Config resolutions.
        let vp = VoicePinManager::default();
        vp.register_serving(["same-model", "same-model", "same-model"]);
        assert_eq!(vp.serving_count("same-model"), 1);

        vp.deregister_serving(["same-model", "same-model", "same-model"]);
        assert_eq!(vp.serving_count("same-model"), 0);
    }

    #[test]
    fn deregister_serving_floors_at_zero_and_does_not_panic() {
        let vp = VoicePinManager::default();
        vp.deregister_serving(["never-registered"]); // must not panic/underflow
        assert_eq!(vp.serving_count("never-registered"), 0);
    }
}
