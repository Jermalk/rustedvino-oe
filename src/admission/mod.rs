// ============================================================
// src/admission/mod.rs — cross-pipeline device admission middleware
// ============================================================
// Step 2 of the cross-pipeline admission middleware migration (rationale:
// the project's internal engineering log, 2026-07-14 "Cross-pipeline admission middleware design").
//
// A SECOND admission gate, sitting IN FRONT OF each engine's own per-engine
// Semaphore gate (cb_engine/vlm_engine/npu_engine/embed_engine/pipelines::
// {stt,tts}) — it does not replace or merge with them. Its job: cap
// concurrency across DIFFERENT engine kinds sharing one physical device
// (e.g. an LLM + VLM + STT + TTS + embedder all resident on one GPU), which
// no per-engine gate can see.
//
// PASSTHROUGH IS THE CORE SAFETY CONTRACT: a device with no entry in
// `device_budgets` config gets no `Semaphore` at all. `admit()` never waits,
// never allocates, never holds anything for an unconfigured device — see
// `admit_is_pure_passthrough_for_unconfigured_device` below. Every device
// budget ships empty (disabled) until empirical testing produces real
// numbers.
//
// Metrics are explicitly OUT OF SCOPE this step (Migration order #2 defers
// Prometheus to step 4's calibration work) — `WorkClass` exists now only so
// call sites don't need touching again when the metrics facade lands; today
// it flows only into a `tracing::debug!` on rejection.
// ============================================================

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::model_manager::config::Config;

/// Coarse classification of the caller's work, for logging today and as the
/// future `class` label on `rustedvino_lease_wait_seconds` (step 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkClass {
    /// Plain `/v1/chat/completions` and `/v1/completions` (CB or NPU backend).
    Chat,
    /// `/v1/chat/completions` routed to a VLM (`EngineHandleKind::Vision`).
    Vision,
    /// `/v1/embeddings`.
    Embed,
    /// Standalone `/v1/audio/transcriptions` (outside a realtime turn).
    Stt,
    /// Standalone `/v1/audio/speech` (outside a realtime turn).
    Tts,
    /// The realtime voice turn's atomic up-front multi-device lease (step 3;
    /// wired in `run_response`, `src/handlers/realtime.rs`).
    RealtimeTurn,
}

impl WorkClass {
    /// Stable label for `tracing` today; will become the metrics label.
    fn label(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Vision => "vision",
            Self::Embed => "embed",
            Self::Stt => "stt",
            Self::Tts => "tts",
            Self::RealtimeTurn => "realtime_turn",
        }
    }
}

/// A device's admission token could not be acquired within
/// `device_admission_queue_timeout_ms` — caller should return `HTTP 429`.
///
/// Deliberately **not** named/shaped like [`crate::cb_engine::AdmitError`] (a
/// different, per-engine gate) so the two never collide in scope: this type
/// has no `EngineDead`/`Failed` variants (there is no engine thread here,
/// only a `Semaphore`), so it is a struct, not an enum. Handler files never
/// need to name this type directly — every call site maps it to a `Response`
/// via `crate::handlers::error::device_admission_error_response`, so it never
/// shares an import list with `cb_engine::AdmitError`.
#[derive(Debug)]
pub struct AdmitError {
    /// The device whose budget timed out.
    pub device: String,
}

impl std::fmt::Display for AdmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "device '{}' at capacity (admission timeout)",
            self.device
        )
    }
}

impl std::error::Error for AdmitError {}

/// A held set of per-device admission tokens, RAII: every permit releases on
/// `Drop` (including on early return, task cancellation, or a client
/// disconnect that drops the future holding this lease) — no separate guard
/// type needed, unlike `in_flight::InFlightGuard`, because
/// `OwnedSemaphorePermit` already carries real Drop-release semantics;
/// `WorkLease` is just a `HashMap` of them.
#[derive(Debug, Default)]
pub struct WorkLease {
    permits: HashMap<String, Vec<OwnedSemaphorePermit>>,
}

impl WorkLease {
    fn push(&mut self, device: String, mut new_permits: Vec<OwnedSemaphorePermit>) {
        self.permits
            .entry(device)
            .or_default()
            .append(&mut new_permits);
    }

    /// Give back up to `n` held permits on `device` early — e.g. after
    /// `run_stt()` returns in the step-3 realtime turn, before the LLM stage
    /// starts. Release-while-holding cannot deadlock (only acquire-while-
    /// holding can), so this is the only mutation `WorkLease` exposes
    /// post-acquire. A no-op if `device` isn't held or `n` is 0 — never
    /// panics on an unconfigured/untracked device.
    pub fn release(&mut self, device: &str, n: usize) {
        if let Some(held) = self.permits.get_mut(device) {
            let keep = held.len().saturating_sub(n);
            held.truncate(keep); // drops the tail permits, returning them
        }
    }
}

/// Cross-pipeline device-wide admission ceiling. One instance lives in
/// `AppState`, shared by every handler.
pub struct DeviceBudgets {
    /// One `Semaphore` per *configured* device. A device string absent here
    /// has no entry — `admit` treats that as unconditional passthrough, not
    /// a very-large cap. Immutable after construction: step 2 never adds a
    /// device at runtime (no admin endpoint for it yet); revisit if that
    /// need appears (the design doc's Part 3 sketch used `RwLock<HashMap<..>>`
    /// anticipating that — deferred here for simplicity since nothing
    /// mutates it this step).
    devices: HashMap<String, Arc<Semaphore>>,
    /// Shared deadline budget for one `admit()` call, apportioned across
    /// however many devices it requests (sorted order, single deadline).
    timeout: Duration,
}

impl Default for DeviceBudgets {
    /// Every device passthrough (no entry) — used by `AppState::default()`
    /// and mock/test state. The timeout value is irrelevant with an empty
    /// device map (nothing is ever waited on).
    fn default() -> Self {
        Self::new(HashMap::new(), Duration::from_secs(5))
    }
}

impl DeviceBudgets {
    fn new(caps: HashMap<String, u32>, timeout: Duration) -> Self {
        let devices = caps
            .into_iter()
            .map(|(device, cap)| (device, Arc::new(Semaphore::new(cap as usize))))
            .collect();
        Self { devices, timeout }
    }

    /// Builds device semaphores from `config.device_budgets`
    /// (`#[serde(default)]`; absent/empty ⇒ every device passes through —
    /// the core passthrough contract). `config.validate()` already rejects a
    /// configured cap of `0` at load time, so every entry here is `>= 1`.
    #[must_use]
    pub fn from_config(config: &Config) -> Self {
        Self::new(
            config.device_budgets.clone(),
            Duration::from_millis(config.device_admission_queue_timeout_ms),
        )
    }

    /// Acquire `cost` tokens on each `(device, cost)` pair, in sorted-device
    /// order, under one shared deadline — forward-compatible with the future
    /// multi-device realtime lease (step 3) even though every call this step
    /// passes exactly one pair.
    ///
    /// A device absent from the configured budgets is a pure no-op: no wait,
    /// no permit held, `cost` ignored for it entirely. A `cost` of `0` for a
    /// configured device is also a no-op (nothing to acquire).
    ///
    /// On timeout for any device, every permit already acquired for
    /// earlier (sorted-earlier) devices in this same call is released
    /// automatically — the partially-built lease is dropped when this
    /// function returns `Err`, so nothing leaks.
    ///
    /// # Errors
    /// [`AdmitError`] naming the device that timed out — caller should
    /// return `HTTP 429`.
    pub async fn admit(
        &self,
        wants: &[(String, u32)],
        class: WorkClass,
    ) -> Result<WorkLease, AdmitError> {
        if wants.is_empty() {
            return Ok(WorkLease::default());
        }
        let mut sorted: Vec<&(String, u32)> = wants.iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));

        let start = Instant::now();
        let mut lease = WorkLease::default();

        for (device, cost) in sorted {
            if *cost == 0 {
                continue;
            }
            let Some(sem) = self.devices.get(device) else {
                continue; // unconfigured device: pure passthrough
            };
            let remaining = self.timeout.saturating_sub(start.elapsed());
            match tokio::time::timeout(remaining, acquire_n(sem, *cost)).await {
                Ok(Ok(permits)) => lease.push(device.clone(), permits),
                Ok(Err(_)) => {
                    // The semaphore is never closed anywhere in this module
                    // (no `close()` call exists) — this should be
                    // unreachable, but fail closed rather than unwrap.
                    tracing::error!(device, "device semaphore unexpectedly closed");
                    return Err(AdmitError {
                        device: device.clone(),
                    });
                }
                Err(_elapsed) => {
                    tracing::debug!(
                        device,
                        cost,
                        class = class.label(),
                        "device admission timed out"
                    );
                    return Err(AdmitError {
                        device: device.clone(),
                    });
                }
            }
        }
        Ok(lease)
    }
}

/// Acquire `n` individual 1-count owned permits on `sem`, so a caller can
/// later give back a subset via [`WorkLease::release`] (impossible once
/// permits are combined via `acquire_many_owned`). Not atomic across `n`
/// against other concurrent callers — acceptable because every call site
/// this step passes `cost == 1`; a future multi-cost caller (step 3's
/// realtime turn) may want `acquire_many_owned` for true atomicity and
/// release it via a distinct combined-permit path if that's ever needed.
async fn acquire_n(
    sem: &Arc<Semaphore>,
    n: u32,
) -> Result<Vec<OwnedSemaphorePermit>, tokio::sync::AcquireError> {
    let mut permits = Vec::with_capacity(n as usize);
    for _ in 0..n {
        permits.push(Arc::clone(sem).acquire_owned().await?);
    }
    Ok(permits)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{DeviceBudgets, HashMap, WorkClass};
    use std::time::Duration;

    /// THE passthrough contract: an unconfigured device never waits, never
    /// holds anything. Proven by running far more concurrent admits than any
    /// real cap would allow, all resolving immediately.
    #[tokio::test]
    async fn admit_is_pure_passthrough_for_unconfigured_device() {
        let budgets = DeviceBudgets::default(); // empty config ⇒ every device passthrough
        let mut leases = Vec::new();
        for _ in 0..1000 {
            let lease = budgets
                .admit(&[("GPU.0".to_owned(), 1)], WorkClass::Chat)
                .await
                .expect("unconfigured device must never reject");
            leases.push(lease); // held concurrently — still no contention
        }
        assert_eq!(leases.len(), 1000);
    }

    /// Empty `wants` is always a no-op, regardless of config.
    #[tokio::test]
    async fn admit_with_no_wants_is_a_no_op() {
        let budgets = DeviceBudgets::new(
            HashMap::from([("GPU.0".to_owned(), 1)]),
            Duration::from_millis(50),
        );
        assert!(budgets.admit(&[], WorkClass::Chat).await.is_ok());
    }

    /// A configured device enforces its cap and returns `AdmitError` on
    /// timeout once exhausted — the real gate, once enabled.
    #[tokio::test]
    async fn admit_rejects_after_cap_exhausted() {
        let budgets = DeviceBudgets::new(
            HashMap::from([("GPU.0".to_owned(), 1)]),
            Duration::from_millis(50),
        );
        let _held = budgets
            .admit(&[("GPU.0".to_owned(), 1)], WorkClass::Chat)
            .await
            .unwrap();

        let err = budgets
            .admit(&[("GPU.0".to_owned(), 1)], WorkClass::Chat)
            .await
            .expect_err("cap is exhausted — must reject, not hang");
        assert_eq!(err.device, "GPU.0");
    }

    /// Dropping a lease returns its permits — a subsequent admit succeeds.
    #[tokio::test]
    async fn dropping_lease_frees_the_slot() {
        let budgets = DeviceBudgets::new(
            HashMap::from([("GPU.0".to_owned(), 1)]),
            Duration::from_millis(200),
        );
        let held = budgets
            .admit(&[("GPU.0".to_owned(), 1)], WorkClass::Chat)
            .await
            .unwrap();
        drop(held);
        assert!(
            budgets
                .admit(&[("GPU.0".to_owned(), 1)], WorkClass::Chat)
                .await
                .is_ok()
        );
    }

    /// `release(device, n)` gives back exactly `n` permits early, letting a
    /// later admit for the freed count succeed while the rest stays held —
    /// the incremental-release contract step 3's realtime turn needs.
    #[tokio::test]
    async fn release_returns_only_the_requested_count() {
        let budgets = DeviceBudgets::new(
            HashMap::from([("GPU.0".to_owned(), 2)]),
            Duration::from_millis(200),
        );
        let mut held = budgets
            .admit(&[("GPU.0".to_owned(), 2)], WorkClass::Chat)
            .await
            .unwrap();
        // Both permits held: a further 1-cost admit must time out.
        assert!(
            budgets
                .admit(&[("GPU.0".to_owned(), 1)], WorkClass::Chat)
                .await
                .is_err()
        );
        held.release("GPU.0", 1);
        // One freed: a 1-cost admit now succeeds.
        assert!(
            budgets
                .admit(&[("GPU.0".to_owned(), 1)], WorkClass::Chat)
                .await
                .is_ok()
        );
    }

    /// `release` on a device this lease never touched is a safe no-op.
    #[test]
    fn release_on_untracked_device_does_not_panic() {
        let mut lease = super::WorkLease::default();
        lease.release("GPU.9", 5);
    }

    /// A `wants` list naming the same device more than once (the realtime
    /// turn's STT/LLM/TTS devices commonly collapse to one physical device)
    /// accumulates: each entry acquires its own permit rather than being
    /// deduplicated, and `release(device, n)` gives back exactly `n` of the
    /// pooled permits regardless of which `wants` entry originally acquired
    /// them (permits on one semaphore are fungible).
    #[tokio::test]
    async fn admit_accumulates_repeated_wants_entries_on_the_same_device() {
        let budgets = DeviceBudgets::new(
            HashMap::from([("GPU.0".to_owned(), 3)]),
            Duration::from_millis(200),
        );
        let mut held = budgets
            .admit(
                &[
                    ("GPU.0".to_owned(), 1),
                    ("GPU.0".to_owned(), 1),
                    ("GPU.0".to_owned(), 1),
                ],
                WorkClass::RealtimeTurn,
            )
            .await
            .unwrap();
        // All 3 permits held: even a 1-cost admit must time out.
        assert!(
            budgets
                .admit(&[("GPU.0".to_owned(), 1)], WorkClass::Chat)
                .await
                .is_err()
        );
        // Mirrors the realtime turn releasing its STT token after run_stt.
        held.release("GPU.0", 1);
        // One freed: a 2-cost admit still doesn't fit, a 1-cost one does.
        assert!(
            budgets
                .admit(&[("GPU.0".to_owned(), 2)], WorkClass::Chat)
                .await
                .is_err()
        );
        assert!(
            budgets
                .admit(&[("GPU.0".to_owned(), 1)], WorkClass::Chat)
                .await
                .is_ok()
        );
    }
}
