// ============================================================
// src/ffi_watch.rs — registry of OpenVINO calls currently in flight (step 6)
// ============================================================
// Every long OpenVINO call that can wedge (a model load, a transcription, a
// CB step, a VLM generation) holds an `FfiToken` for its duration. The token
// is armed on the thread that makes the FFI call and removed by its `Drop`,
// so it stays in the table when the awaiting future is cancelled or the
// model's record leaves `Ready` — exactly the cases where a thread is still
// inside the plugin and nothing else can see it.
//
// Two readers:
// - process exit (`main`): with a call in flight, returning from `main` runs
//   the C++ static destructors under a live pipeline (D-01 heap corruption),
//   so the process leaves with `_exit` and exit code 76 instead;
// - the supervisor's generation watchdog, through
//   `ModelManager::watchdog_age`, via [`FfiWatch::overdue`].
//
// CRASH COURSE — why RAII and not "insert before, remove after": a call that
// panics or a thread that unwinds still runs `Drop`, and a remove statement
// after the call never runs if the call never returns — which is the state
// we must *keep* visible. `Drop` removes the entry exactly when the call is
// over, and never earlier.
// ============================================================

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// What kind of `OpenVINO` call a token covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FfiKind {
    /// A pipeline construction / compile (`factory.load`).
    Load,
    /// A Whisper transcription.
    Transcribe,
    /// One continuous-batching step (decode iteration or prefill chunk).
    CbStep,
    /// One VLM generation. Counted for the exit decision only: the VLM's own
    /// `GenerationClock` already feeds the watchdog, and counting it in
    /// [`FfiWatch::overdue`] too would make a generation orphaned by a failed
    /// runtime eviction a new watchdog kill trigger.
    VlmGenerate,
    /// An engine thread's exit after its eviction: it closes its pipeline, whose destructor
    /// frees the device memory (D-04 S6). Armed for the whole join — also one an eviction gave
    /// up on, which stays here until the thread really exits — so the exit decision sees a
    /// thread still inside the plugin. Not counted for the watchdog: a slow teardown is not a
    /// stuck request.
    Teardown,
}

impl FfiKind {
    /// Short label for logs.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Load => "load",
            Self::Transcribe => "transcribe",
            Self::CbStep => "cb_step",
            Self::VlmGenerate => "vlm_generate",
            Self::Teardown => "teardown",
        }
    }

    /// Whether [`FfiWatch::overdue`] measures this kind (see [`Self::VlmGenerate`],
    /// [`Self::Teardown`]).
    fn counts_for_watchdog(self) -> bool {
        !matches!(self, Self::VlmGenerate | Self::Teardown)
    }
}

/// One call in the table.
#[derive(Debug)]
struct Entry {
    kind: FfiKind,
    model_id: Arc<str>,
    device: Option<String>,
    started_at: Instant,
    allowance: Duration,
}

/// One engine's step slot (see [`StepSlot`]): registered once, updated per step without a lock.
#[derive(Debug)]
struct SlotState {
    /// Step start in nanoseconds since [`epoch`], `0` while no step runs.
    started_ns: AtomicU64,
    model_id: Arc<str>,
    device: Option<String>,
    allowance: Duration,
}

/// The registry. One process-wide instance ([`global`]); tests build their own.
#[derive(Debug, Default)]
pub struct FfiWatch {
    entries: Mutex<HashMap<u64, Entry>>,
    /// Per-engine CB step slots (step 6d), keyed like `entries`.
    slots: Mutex<HashMap<u64, Arc<SlotState>>>,
    next_id: AtomicU64,
}

/// The process-wide registry every production call site arms into.
pub fn global() -> &'static FfiWatch {
    static WATCH: OnceLock<FfiWatch> = OnceLock::new();
    WATCH.get_or_init(FfiWatch::default)
}

/// The reference instant step slots count from.
fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

/// `at` as nanoseconds since [`epoch`], never `0` (`0` means idle).
fn ns_since_epoch(at: Instant) -> u64 {
    u64::try_from(at.saturating_duration_since(epoch()).as_nanos())
        .unwrap_or(u64::MAX)
        .max(1)
}

/// One in-flight call: removed from its [`FfiWatch`] when dropped. Hold it on
/// the thread making the call, for exactly the call's duration.
#[derive(Debug)]
#[must_use = "the call is only tracked while the token is held"]
pub struct FfiToken<'w> {
    watch: &'w FfiWatch,
    id: u64,
}

impl Drop for FfiToken<'_> {
    fn drop(&mut self) {
        if let Ok(mut entries) = self.watch.entries.lock() {
            entries.remove(&self.id);
        }
    }
}

/// A continuous-batching engine's step slot (step 6d): registered once when the engine thread
/// starts and removed when dropped. The per-step cost is one atomic store at each end
/// ([`Self::running`]) — no lock, no allocation on the token hot path.
#[derive(Debug)]
#[must_use = "the engine's steps are only tracked while the slot is held"]
pub struct StepSlot<'w> {
    watch: &'w FfiWatch,
    id: u64,
    state: Arc<SlotState>,
}

impl StepSlot<'_> {
    /// Mark a step as running from `started` until the returned guard drops (also on unwind).
    #[inline]
    pub fn running(&self, started: Instant) -> StepRun<'_> {
        self.state
            .started_ns
            .store(ns_since_epoch(started), Ordering::Relaxed);
        StepRun { state: &self.state }
    }
}

impl Drop for StepSlot<'_> {
    fn drop(&mut self) {
        if let Ok(mut slots) = self.watch.slots.lock() {
            slots.remove(&self.id);
        }
    }
}

/// One running step of a [`StepSlot`]; marks the slot idle when dropped.
#[derive(Debug)]
#[must_use = "the step is only tracked while the guard is held"]
pub struct StepRun<'s> {
    state: &'s SlotState,
}

impl Drop for StepRun<'_> {
    #[inline]
    fn drop(&mut self) {
        self.state.started_ns.store(0, Ordering::Relaxed);
    }
}

/// A snapshot of one in-flight call, from either table.
struct Call {
    kind: FfiKind,
    model_id: Arc<str>,
    device: Option<String>,
    started_at: Instant,
    allowance: Duration,
}

impl FfiWatch {
    /// Start tracking a call. `allowance` is how long it may legitimately run;
    /// [`Self::overdue`] counts only the time beyond it.
    pub fn arm(
        &self,
        kind: FfiKind,
        model_id: &str,
        device: Option<&str>,
        allowance: Duration,
    ) -> FfiToken<'_> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut entries) = self.entries.lock() {
            entries.insert(
                id,
                Entry {
                    kind,
                    model_id: Arc::from(model_id),
                    device: device.map(str::to_owned),
                    started_at: Instant::now(),
                    allowance,
                },
            );
        }
        FfiToken { watch: self, id }
    }

    /// Register a CB engine's step slot ([`FfiKind::CbStep`]); `allowance` applies per step.
    pub fn register_step_slot(
        &self,
        model_id: &str,
        device: Option<&str>,
        allowance: Duration,
    ) -> StepSlot<'_> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let state = Arc::new(SlotState {
            started_ns: AtomicU64::new(0),
            model_id: Arc::from(model_id),
            device: device.map(str::to_owned),
            allowance,
        });
        if let Ok(mut slots) = self.slots.lock() {
            slots.insert(id, Arc::clone(&state));
        }
        StepSlot {
            watch: self,
            id,
            state,
        }
    }

    /// Every call in flight right now: the token table plus the running step slots.
    fn calls(&self) -> Vec<Call> {
        let mut out: Vec<Call> = self
            .entries
            .lock()
            .map(|entries| {
                entries
                    .values()
                    .map(|e| Call {
                        kind: e.kind,
                        model_id: Arc::clone(&e.model_id),
                        device: e.device.clone(),
                        started_at: e.started_at,
                        allowance: e.allowance,
                    })
                    .collect()
            })
            .unwrap_or_default();
        if let Ok(slots) = self.slots.lock() {
            for s in slots.values() {
                let ns = s.started_ns.load(Ordering::Relaxed);
                if ns != 0 {
                    out.push(Call {
                        kind: FfiKind::CbStep,
                        model_id: Arc::clone(&s.model_id),
                        device: s.device.clone(),
                        started_at: epoch() + Duration::from_nanos(ns),
                        allowance: s.allowance,
                    });
                }
            }
        }
        out
    }

    /// Number of calls in flight, of every kind.
    #[must_use]
    pub fn in_flight_count(&self) -> usize {
        self.calls().len()
    }

    /// Calls in flight per kind, sorted by kind — for the shutdown log.
    #[must_use]
    pub fn in_flight_by_kind(&self) -> Vec<(FfiKind, usize)> {
        let mut counts: HashMap<FfiKind, usize> = HashMap::new();
        for c in self.calls() {
            *counts.entry(c.kind).or_insert(0) += 1;
        }
        let mut out: Vec<_> = counts.into_iter().collect();
        out.sort_unstable();
        out
    }

    /// The models (and devices, when known) with a call in flight, sorted — for the shutdown log.
    #[must_use]
    pub fn in_flight_models(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .calls()
            .iter()
            .map(|c| match &c.device {
                Some(d) => format!("{} {} on {d}", c.kind.label(), c.model_id),
                None => format!("{} {}", c.kind.label(), c.model_id),
            })
            .collect();
        out.sort_unstable();
        out
    }

    /// How far past its allowance the worst watchdog-counted call is: zero
    /// while every one is inside it, `None` when none is in flight.
    #[must_use]
    pub fn overdue(&self) -> Option<Duration> {
        let now = Instant::now();
        self.calls()
            .iter()
            .filter(|c| c.kind.counts_for_watchdog())
            .map(|c| {
                now.saturating_duration_since(c.started_at)
                    .saturating_sub(c.allowance)
            })
            .max()
    }

    /// The stop-path wait's view of [`Self::overdue`] (D-04, Fable S8 review #10): while any
    /// model load is still inside its allowance, zero — the stop waits for it, though a load that
    /// stalled earlier (its thread still in `OpenVINO`) is overdue; once every load in flight is
    /// past its allowance, how far the least late one is; with no load in flight, [`Self::overdue`].
    #[must_use]
    pub fn stop_wait_overdue(&self) -> Option<Duration> {
        let now = Instant::now();
        let loads = self.calls();
        let least_late = loads
            .iter()
            .filter(|c| c.kind == FfiKind::Load)
            .map(|c| {
                now.saturating_duration_since(c.started_at)
                    .saturating_sub(c.allowance)
            })
            .min();
        least_late.or_else(|| self.overdue())
    }

    /// Wait up to `limit` for every call to return, polling every `poll`.
    /// Returns the count still in flight at the end (0 at once when idle).
    /// Blocking on purpose: `main` calls it after the runtime has stopped.
    #[must_use]
    pub fn wait_idle(&self, limit: Duration, poll: Duration) -> usize {
        let deadline = Instant::now() + limit;
        loop {
            let n = self.in_flight_count();
            if n == 0 || Instant::now() >= deadline {
                return n;
            }
            std::thread::sleep(poll.min(deadline.saturating_duration_since(Instant::now())));
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn token_is_counted_while_held_and_removed_on_drop() {
        let w = FfiWatch::default();
        assert_eq!(w.in_flight_count(), 0);
        assert_eq!(w.overdue(), None);
        let t = w.arm(FfiKind::Load, "m", Some("GPU.1"), Duration::from_secs(600));
        assert_eq!(w.in_flight_count(), 1);
        assert_eq!(w.overdue(), Some(Duration::ZERO), "inside its allowance");
        assert_eq!(w.in_flight_models(), vec!["load m on GPU.1".to_string()]);
        drop(t);
        assert_eq!(w.in_flight_count(), 0);
        assert_eq!(w.overdue(), None);
    }

    #[test]
    fn overdue_counts_only_time_past_the_allowance() {
        let w = FfiWatch::default();
        let _t = w.arm(FfiKind::Transcribe, "stt", None, Duration::ZERO);
        std::thread::sleep(Duration::from_millis(20));
        assert!(w.overdue().unwrap() >= Duration::from_millis(20));
    }

    /// Fable S8 #10: a load that stalled earlier does not end the stop wait while a later load is
    /// still inside its allowance; once only overdue loads are left, it does.
    #[test]
    fn stop_wait_overdue_waits_while_any_load_is_inside_its_allowance() {
        let w = FfiWatch::default();
        let stalled = w.arm(FfiKind::Load, "old", Some("GPU.1"), Duration::ZERO);
        std::thread::sleep(Duration::from_millis(20));
        assert!(w.stop_wait_overdue().unwrap() >= Duration::from_millis(20));
        let healthy = w.arm(
            FfiKind::Load,
            "new",
            Some("GPU.1"),
            Duration::from_secs(600),
        );
        assert!(
            w.overdue().unwrap() >= Duration::from_millis(20),
            "the watchdog view is unchanged"
        );
        assert_eq!(w.stop_wait_overdue(), Some(Duration::ZERO));
        drop(healthy);
        assert!(w.stop_wait_overdue().unwrap() >= Duration::from_millis(20));
        drop(stalled);
        // No load in flight: the other calls decide, as before.
        let _g = w.arm(FfiKind::Transcribe, "stt", None, Duration::ZERO);
        std::thread::sleep(Duration::from_millis(5));
        assert!(w.stop_wait_overdue().unwrap() >= Duration::from_millis(5));
    }

    /// A load with no deadline arms an unbounded allowance: never overdue, still in flight.
    #[test]
    fn unbounded_allowance_is_never_overdue() {
        let w = FfiWatch::default();
        let _t = w.arm(FfiKind::Load, "m", None, Duration::MAX);
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(w.overdue(), Some(Duration::ZERO));
        assert_eq!(w.in_flight_count(), 1);
    }

    /// 6d: a step slot counts only while a step runs, is measured from the step start, and
    /// leaves the table when the slot drops.
    #[test]
    fn step_slot_counts_only_while_a_step_runs() {
        let w = FfiWatch::default();
        let slot = w.register_step_slot("cb", Some("GPU.1"), Duration::ZERO);
        assert_eq!(w.in_flight_count(), 0, "idle engine: nothing in flight");
        assert_eq!(w.overdue(), None);
        {
            let _run = slot.running(Instant::now());
            std::thread::sleep(Duration::from_millis(20));
            assert_eq!(w.in_flight_count(), 1);
            assert_eq!(
                w.in_flight_models(),
                vec!["cb_step cb on GPU.1".to_string()]
            );
            assert!(w.overdue().unwrap() >= Duration::from_millis(20));
            assert_eq!(w.in_flight_by_kind(), vec![(FfiKind::CbStep, 1)]);
        }
        assert_eq!(w.in_flight_count(), 0, "step over: idle again");
        drop(slot);
        assert!(
            w.slots.lock().unwrap().is_empty(),
            "a dropped slot leaves the table"
        );
    }

    #[test]
    fn vlm_tokens_count_for_exit_but_not_for_the_watchdog() {
        let w = FfiWatch::default();
        let _t = w.arm(FfiKind::VlmGenerate, "vlm", None, Duration::ZERO);
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(w.in_flight_count(), 1);
        assert_eq!(w.overdue(), None);
    }

    /// D-04 S6: a teardown keeps the exit decision honest (in flight) but never trips the
    /// watchdog, however long the pipeline destructor takes.
    #[test]
    fn teardown_tokens_count_for_exit_but_not_for_the_watchdog() {
        let w = FfiWatch::default();
        let _t = w.arm(FfiKind::Teardown, "m", None, Duration::ZERO);
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(w.in_flight_count(), 1);
        assert_eq!(w.overdue(), None);
        assert_eq!(w.in_flight_models(), vec!["teardown m".to_owned()]);
    }

    #[test]
    fn by_kind_groups_and_sorts() {
        let w = FfiWatch::default();
        let _a = w.arm(FfiKind::VlmGenerate, "v", None, Duration::ZERO);
        let _b = w.arm(FfiKind::Load, "x", None, Duration::ZERO);
        let _c = w.arm(FfiKind::Load, "y", None, Duration::ZERO);
        assert_eq!(
            w.in_flight_by_kind(),
            vec![(FfiKind::Load, 2), (FfiKind::VlmGenerate, 1)]
        );
    }

    #[test]
    fn token_survives_its_arming_thread_and_drops_where_it_is_dropped() {
        // Armed on one thread, the token is visible from another until dropped there.
        let w = std::sync::Arc::new(FfiWatch::default());
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let w2 = std::sync::Arc::clone(&w);
        let h = std::thread::spawn(move || {
            let _t = w2.arm(FfiKind::CbStep, "cb", None, Duration::ZERO);
            rx.recv().unwrap();
        });
        while w.in_flight_count() == 0 {
            std::thread::yield_now();
        }
        tx.send(()).unwrap();
        h.join().unwrap();
        assert_eq!(w.in_flight_count(), 0);
    }

    #[test]
    fn wait_idle_returns_at_once_when_idle_and_reports_stragglers() {
        let w = FfiWatch::default();
        let t0 = Instant::now();
        assert_eq!(
            w.wait_idle(Duration::from_secs(3), Duration::from_millis(10)),
            0
        );
        assert!(
            t0.elapsed() < Duration::from_millis(100),
            "no wait when idle"
        );
        let _t = w.arm(FfiKind::Load, "m", None, Duration::ZERO);
        assert_eq!(
            w.wait_idle(Duration::from_millis(30), Duration::from_millis(10)),
            1
        );
    }
}
