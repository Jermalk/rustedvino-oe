// ============================================================
// src/in_flight.rs — cancellation-safe in-flight counter guard
// ============================================================
// THE request/response in-flight discipline. A request/response engine
// (embeddings today; STT / TTS / image in Phase 5) accepts a job, sends it to
// its dedicated engine thread, and `await`s the reply *in the caller's task*.
// If that future is dropped mid-`await` — the HTTP client disconnected, a
// timeout fired, the handler was cancelled — any plain `fetch_add` on submit
// paired with a `fetch_sub` after the await NEVER RUNS, and the in-flight
// counter leaks upward, permanently corrupting `/metrics` (`active`/`waiting`).
//
// `InFlightGuard` ties the decrement to `Drop`, so it settles on every exit
// path including cancellation. Hold one for the lifetime of each request:
//
//     let _guard = InFlightGuard::new(&self.in_flight);
//     self.tx.send(cmd).await?;       // cancelled here? guard still drops.
//     reply_rx.await?                 // or here? still drops.
//
// CRASH COURSE — why request/response needs this but token streams don't:
//   The continuous-batching (CB) and VLM engines decrement on their *engine
//   thread* as each request finishes, off the caller's future entirely — a
//   disconnected token-stream client is reclaimed by the engine's own
//   step loop (see `cb_engine::step_and_route`), not by a caller-side guard.
//   Request/response engines have no such per-step loop touching the caller's
//   await, so the guard is their cancellation-safe equivalent.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// RAII guard for an in-flight request counter: increments on construction,
/// decrements on `Drop`. Because the decrement runs in `Drop`, it survives task
/// cancellation — if the awaiting future is dropped (client disconnect,
/// timeout), the count settles back rather than leaking upward.
///
/// Shared by every request/response engine so the discipline is defined once
/// (T2/F3). The counter itself lives in the engine handle (an
/// `Arc<AtomicUsize>` also read by `ManagedEngine::active`); the guard only
/// borrows it for the request's lifetime.
pub(crate) struct InFlightGuard(Arc<AtomicUsize>);

impl InFlightGuard {
    /// Increment `counter` now and hold a clone; the matching decrement runs
    /// when the returned guard drops. Bind it to a `_guard` local that lives for
    /// the whole request (so it is also dropped if the future is cancelled).
    pub(crate) fn new(counter: &Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::Relaxed);
        Self(Arc::clone(counter))
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// RAII guard reserving `tokens` against a shared admitted-prompt-tokens
/// counter, for the concurrent-KV-admission check (root cause: the L0 gate
/// sized a request's ceiling off a model's *entire* pool with no accounting
/// for what else was already admitted). Unlike [`InFlightGuard`], construction is fallible:
/// [`try_new`](Self::try_new) atomically reserves first, then checks whether
/// that reservation fits under `capacity`, rolling back immediately if not
/// — never a plain check-then-increment, which two requests arriving
/// together could both pass (confirmed as a real risk, not a theoretical
/// one, empirically measuring `rustedvino_kv_cache_usage_percent` staying
/// at a stale `0.0` for ~1.5s after a large request is admitted, before the
/// engine's first step publishes).
///
/// Bind the returned guard to a `_guard` local held for the WHOLE request
/// lifetime (through prefill and decode, not just until submission) — same
/// discipline as [`InFlightGuard`], same reason: only `Drop` survives every
/// exit path, including a client disconnecting mid-stream.
pub(crate) struct AdmittedTokensGuard {
    counter: Arc<AtomicUsize>,
    tokens: usize,
}

impl AdmittedTokensGuard {
    /// Attempts to reserve `tokens` against `counter`, capped at `capacity`.
    /// Returns `None` (reservation already rolled back) if admitting `tokens`
    /// would push the counter's new total over `capacity` — the caller
    /// should reject the request instead of submitting it.
    ///
    /// Deliberately makes no exception for "nothing else is in flight right
    /// now": it doesn't need to. `capacity` here is always the model's raw,
    /// unclamped pool-capacity figure (`ModelRecord::pool_capacity_tokens`),
    /// which by construction is always `>=` the L0 gate's own per-request
    /// ceiling (`max_prompt_tokens` — clamps only ever tighten the formula
    /// estimate, never loosen it). So a solo request that already passed the
    /// existing L0 gate is mathematically guaranteed `0 + tokens <=
    /// max_prompt_tokens <= capacity` here too — this call adds no extra
    /// margin on top, so it can never be the thing that rejects a request
    /// the static gate would have allowed alone. See
    /// `solo_request_within_gate_never_rejected_by_pool_capacity_check` below.
    pub(crate) fn try_new(
        counter: &Arc<AtomicUsize>,
        tokens: usize,
        capacity: usize,
    ) -> Option<Self> {
        let prev = counter.fetch_add(tokens, Ordering::Relaxed);
        if prev.saturating_add(tokens) > capacity {
            counter.fetch_sub(tokens, Ordering::Relaxed);
            return None;
        }
        Some(Self {
            counter: Arc::clone(counter),
            tokens,
        })
    }
}

impl Drop for AdmittedTokensGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(self.tokens, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    /// The guard increments on construction and decrements on drop — the
    /// happy-path round trip.
    #[test]
    fn guard_increments_then_decrements() {
        let counter = Arc::new(AtomicUsize::new(0));
        {
            let _g = InFlightGuard::new(&counter);
            assert_eq!(counter.load(Ordering::Relaxed), 1, "incremented on new");
        }
        assert_eq!(counter.load(Ordering::Relaxed), 0, "decremented on drop");
    }

    /// Dropping the guard early (the cancellation analogue: the future holding
    /// it is dropped mid-await) still settles the counter back to zero.
    #[test]
    fn dropping_guard_early_does_not_leak() {
        let counter = Arc::new(AtomicUsize::new(0));
        let g = InFlightGuard::new(&counter);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
        drop(g); // simulate the awaiting future being cancelled
        assert_eq!(counter.load(Ordering::Relaxed), 0, "no leak on early drop");
    }

    /// Nested guards stack and unwind independently — two concurrent requests
    /// against the same counter.
    #[test]
    fn nested_guards_stack() {
        let counter = Arc::new(AtomicUsize::new(0));
        let a = InFlightGuard::new(&counter);
        let b = InFlightGuard::new(&counter);
        assert_eq!(counter.load(Ordering::Relaxed), 2);
        drop(b);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
        drop(a);
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    /// A request that fits reserves successfully and the counter reflects it.
    #[test]
    fn admitted_tokens_guard_reserves_when_it_fits() {
        let counter = Arc::new(AtomicUsize::new(0));
        let g = AdmittedTokensGuard::try_new(&counter, 100, 1000);
        assert!(g.is_some());
        assert_eq!(counter.load(Ordering::Relaxed), 100);
    }

    /// The core regression this guard exists for: two requests that
    /// individually fit but together overcommit the pool — the second must
    /// be rejected, and rolled back to zero net effect, not merely warned
    /// about (the project's internal engineering log's 2x120k-token pile-up).
    #[test]
    fn admitted_tokens_guard_rejects_and_rolls_back_when_it_would_overcommit() {
        let counter = Arc::new(AtomicUsize::new(0));
        let capacity = 150;
        let first = AdmittedTokensGuard::try_new(&counter, 100, capacity);
        assert!(first.is_some(), "first request alone fits");
        assert_eq!(counter.load(Ordering::Relaxed), 100);

        let second = AdmittedTokensGuard::try_new(&counter, 100, capacity);
        assert!(
            second.is_none(),
            "100 + 100 = 200 > capacity 150 — must be rejected"
        );
        assert_eq!(
            counter.load(Ordering::Relaxed),
            100,
            "rejected reservation must roll back to exactly the first request's share, no leak"
        );
    }

    /// A request landing exactly on the capacity boundary is admitted, not
    /// rejected — the check is `> capacity`, not `>= capacity`.
    #[test]
    fn admitted_tokens_guard_admits_exact_capacity_boundary() {
        let counter = Arc::new(AtomicUsize::new(0));
        let g = AdmittedTokensGuard::try_new(&counter, 150, 150);
        assert!(
            g.is_some(),
            "exactly-at-capacity must be admitted, not rejected"
        );
    }

    /// Dropping a successful reservation releases it — the next request can
    /// then use the freed room, matching the request lifecycle (admit,
    /// generate, finish, release).
    #[test]
    fn admitted_tokens_guard_releases_on_drop_for_the_next_request() {
        let counter = Arc::new(AtomicUsize::new(0));
        let capacity = 100;
        let first = AdmittedTokensGuard::try_new(&counter, 100, capacity).unwrap();
        assert!(AdmittedTokensGuard::try_new(&counter, 1, capacity).is_none());
        drop(first);
        assert_eq!(counter.load(Ordering::Relaxed), 0);
        assert!(AdmittedTokensGuard::try_new(&counter, 100, capacity).is_some());
    }

    /// The invariant the whole design leans on (see `try_new`'s doc comment):
    /// a solo request whose own token count is within the model's L0 gate
    /// ceiling (`max_prompt_tokens`) must NEVER be rejected by this check,
    /// because `pool_capacity_tokens` (this check's `capacity`) is always
    /// `>=` `max_prompt_tokens` by construction (clamps only tighten).
    /// Exercise the boundary directly: reserve exactly `max_prompt_tokens`
    /// (the largest a gate-passing solo request could ever be) against a
    /// `capacity` equal to it — must succeed, not just when capacity is
    /// generously larger.
    #[test]
    fn solo_request_within_gate_never_rejected_by_pool_capacity_check() {
        let counter = Arc::new(AtomicUsize::new(0));
        let max_prompt_tokens = 12_379; // a real observed ceiling this session (qwen3-*-int4-ov, 1GB pool)
        let pool_capacity_tokens = max_prompt_tokens; // worst case: clamp didn't loosen anything
        let g = AdmittedTokensGuard::try_new(&counter, max_prompt_tokens, pool_capacity_tokens);
        assert!(
            g.is_some(),
            "a solo request at exactly the L0 gate's own ceiling must be admitted"
        );
    }
}
