//! F1b: the bounded wait on an on-demand load — how long a request may wait, how many may wait
//! at once, and what a stalled load answers.
//!
//! A request to a `load: on_demand` model that is not loaded used to start the load and answer
//! `503` + `Retry-After` at once. With `on_demand_wait_secs > 0` it starts the load (or joins the
//! one running) and holds until the model is `Ready`, the load fails, or the wait expires — then
//! falls back to the old `503`. [`ModelManager::wait_on_demand`] is the one wait every route uses
//! (through `handlers::on_demand`).
//!
//! **Waking.** No polling: the waiters share one [`tokio::sync::Notify`]. Every load end
//! ([`ModelManager::note_load_end`]: `load_model_reported` on success or failure — a failure is
//! recorded first, so a waiter that wakes always finds it — the load-cancel guard and the
//! stalled-load reaper), the end of an engine construction (so a stalled load's late result is
//! reaped), the end of an eviction, a model's removal and `begin_shutdown`
//! ([`ModelManager::note_load_event`]) wake them. A waiter registers its `Notified` before it
//! reads the model's state, so an event between the read and the await is not lost. Each turn
//! reaps the stalled loads first (cheap when there are none).
//!
//! **Past the deadline.** A load past its `load_timeout_secs` (or a device with a stalled load,
//! for a model that would have to load there) is answered at once with the error the waiters got
//! at the stall (`load_timeout`, `device_stalled`): no wait, no waiter place.
//!
//! **Cancel safety.** The wait holds only a [`WaiterSlot`] (released in `Drop`), a `Notified` and
//! timers, all released when the future is dropped (the client went away). The load itself runs
//! on a detached task (`request_on_demand_load`), so dropping a waiter never cancels it: other
//! requests may be waiting on the same load. The wait holds no admission permit.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::{ModelError, ModelManager, ModelState, OnDemandLoad};

/// Most requests that may wait for on-demand loads at once, across all models.
/// Past it a request gets the immediate `503` + `Retry-After`, as with no wait configured.
pub const MAX_ON_DEMAND_WAITERS: usize = 64;

/// How far below the model's own `load_timeout_secs` a wait ends, so a request never outlives
/// the deadline of the load it waits for: a request that starts the load expires (503 +
/// `Retry-After`) before the load could be booked as stalled (504). Covers the gap between the
/// request starting its wait and the load task reaching `begin_load`, and writing the answer.
pub const ON_DEMAND_WAIT_MARGIN: Duration = Duration::from_secs(2);

/// The shared state of the on-demand waits, one per [`ModelManager`].
pub(crate) struct OnDemandWaits {
    /// Requests waiting now. An `Arc` so a [`WaiterSlot`] is `'static` (it moves into a
    /// streaming response body).
    waiters: Arc<AtomicUsize>,
    /// Woken (`notify_waiters`) by every load end and the other events in the module docs. An
    /// `Arc` so an engine-construction thread and an eviction tail can wake it.
    events: Arc<tokio::sync::Notify>,
    /// Per model: how its loads ended (see [`LoadEnds`]).
    ends: Arc<std::sync::Mutex<HashMap<String, LoadEnds>>>,
    /// Tests only: the wait instead of `on_demand_wait_secs`.
    #[cfg(test)]
    budget_override: std::sync::Mutex<Option<Duration>>,
    /// Tests only: the SSE keep-alive interval instead of the 15 s default.
    #[cfg(test)]
    keep_alive_override: std::sync::Mutex<Option<Duration>>,
}

impl Default for OnDemandWaits {
    fn default() -> Self {
        Self {
            waiters: Arc::new(AtomicUsize::new(0)),
            events: Arc::new(tokio::sync::Notify::new()),
            ends: Arc::new(std::sync::Mutex::new(HashMap::new())),
            #[cfg(test)]
            budget_override: std::sync::Mutex::new(None),
            #[cfg(test)]
            keep_alive_override: std::sync::Mutex::new(None),
        }
    }
}

/// How one model's loads ended, for the requests waiting on it.
#[derive(Debug, Default, Clone)]
struct LoadEnds {
    /// Loads of this model that ended (Ready, failed, or abandoned) since the server started.
    count: u64,
    /// `count` at the last failed load (0 = none failed yet).
    failed_at: u64,
    /// That failure's error, when it was a [`ModelError`].
    error: Option<ModelError>,
    /// Load calls of this model under way ([`LoadInFlight`]): from an on-demand start (before
    /// its task runs) or a `load_model_reported` call, until the call has recorded its end. A
    /// waiter that finds the model `NotLoaded` with none under way and no end since it started
    /// knows nothing is loading it, and starts the load again.
    in_flight: u32,
}

/// One load call of a model counted in [`LoadEnds::in_flight`]; uncounted when dropped.
pub(crate) struct LoadInFlight {
    ends: Arc<std::sync::Mutex<HashMap<String, LoadEnds>>>,
    model_id: String,
}

impl Drop for LoadInFlight {
    fn drop(&mut self) {
        if let Some(e) = self
            .ends
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_mut(&self.model_id)
        {
            e.in_flight = e.in_flight.saturating_sub(1);
        }
    }
}

/// One of the [`MAX_ON_DEMAND_WAITERS`] waiter places. Released when dropped — on every exit of
/// the wait, a dropped request future (client gone) included.
#[derive(Debug)]
pub struct WaiterSlot {
    count: Arc<AtomicUsize>,
}

impl Drop for WaiterSlot {
    fn drop(&mut self) {
        self.count.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Where [`ModelManager::wait_on_demand`] gets its waiter place.
#[derive(Debug)]
pub enum WaitPlace {
    /// Take one now; at the cap, no wait (the immediate `503` + `Retry-After`).
    Take,
    /// One taken earlier (a streaming response that started before the wait).
    Held(WaiterSlot),
    /// Do not wait: a streaming request that found no free place before its response started
    /// (it answers as at the cap, never waits without its keep-alives).
    NoWait,
}

/// How [`ModelManager::wait_on_demand`] ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OnDemandWait {
    /// The model's load ended (it is `Ready`, or it left `Loading` some other way): resolve the
    /// model again, once. If it is not usable then, start its load again (no second wait) and
    /// answer as without a wait.
    Settled,
    /// No wait happened, or it ended without a usable model: answer as
    /// [`ModelManager::request_on_demand_load`] says (`Loading` also on expiry and when the
    /// waiter cap is reached; `ShuttingDown` once a stop was requested).
    Refused(OnDemandLoad),
    /// The load failed: its error when it was a [`ModelError`], else `None`. Also the answer
    /// for a load past its deadline (`LoadTimeout`), a device with a stalled load
    /// (`DeviceStalled`) and a model removed while the request waited (`NotFound`).
    LoadFailed(Option<ModelError>),
}

/// The wait for one request: `wait` clamped to `load_timeout − ON_DEMAND_WAIT_MARGIN` when the
/// model's load has a deadline (`None` = no deadline).
fn clamp_to_load_timeout(wait: Duration, load_timeout: Option<Duration>) -> Duration {
    load_timeout.map_or(wait, |t| wait.min(t.saturating_sub(ON_DEMAND_WAIT_MARGIN)))
}

impl ModelManager {
    /// How long a request to `model_id` may wait for its on-demand load: `on_demand_wait_secs`,
    /// clamped below the model's effective `load_timeout_secs` (per-model override, else
    /// global) by [`ON_DEMAND_WAIT_MARGIN`]. Zero = no wait.
    #[must_use]
    pub fn on_demand_wait_budget(&self, model_id: &str) -> Duration {
        #[cfg(test)]
        let wait = self
            .on_demand
            .budget_override
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .unwrap_or(Duration::from_secs(self.config.on_demand_wait_secs));
        #[cfg(not(test))]
        let wait = Duration::from_secs(self.config.on_demand_wait_secs);
        if wait.is_zero() {
            return wait;
        }
        clamp_to_load_timeout(wait, self.load_timeout(model_id))
    }

    /// The SSE keep-alive interval of a streaming response that waits for an on-demand load
    /// (`handlers::on_demand::SSE_KEEP_ALIVE`; tests shorten it per manager).
    #[must_use]
    pub fn on_demand_keep_alive(&self, default: Duration) -> Duration {
        #[cfg(test)]
        return self
            .on_demand
            .keep_alive_override
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .unwrap_or(default);
        #[cfg(not(test))]
        default
    }

    /// Whether a request to `model_id` that finds it not loaded would wait for its load: a wait
    /// is configured (non-zero for this model), the model is `load: on_demand`, and no stop was
    /// requested. Does not look at the waiter cap.
    #[must_use]
    pub fn on_demand_wait_applies(&self, model_id: &str) -> bool {
        !self.on_demand_wait_budget(model_id).is_zero()
            && self.is_on_demand(model_id)
            && !self.is_shutting_down()
    }

    /// Take one of the [`MAX_ON_DEMAND_WAITERS`] waiter places, or `None` when all are taken.
    /// Atomic (one `fetch_update`), so racing requests never go past the cap.
    #[must_use]
    pub fn try_on_demand_waiter_slot(&self) -> Option<WaiterSlot> {
        let count = &self.on_demand.waiters;
        count
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < MAX_ON_DEMAND_WAITERS).then_some(n + 1)
            })
            .ok()
            .map(|_| WaiterSlot {
                count: Arc::clone(count),
            })
    }

    /// Requests waiting for an on-demand load right now.
    #[must_use]
    pub fn on_demand_waiters(&self) -> usize {
        self.on_demand.waiters.load(Ordering::Acquire)
    }

    /// Wake every request waiting for an on-demand load, to look at the models again (a stop,
    /// an eviction's end, a removal; load ends go through [`note_load_end`](Self::note_load_end)).
    pub(crate) fn note_load_event(&self) {
        self.on_demand.events.notify_waiters();
    }

    /// The waiters' wake-up, for a thread that has no `ModelManager` (an engine construction, an
    /// eviction tail): `notify_waiters` on it is [`note_load_event`](Self::note_load_event).
    pub(crate) fn on_demand_waker(&self) -> Arc<tokio::sync::Notify> {
        Arc::clone(&self.on_demand.events)
    }

    /// Count one load call of `model_id` as under way until the returned guard is dropped (see
    /// [`LoadEnds::in_flight`]).
    pub(crate) fn load_in_flight(&self, model_id: &str) -> LoadInFlight {
        self.on_demand
            .ends
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(model_id.to_owned())
            .or_default()
            .in_flight += 1;
        LoadInFlight {
            ends: Arc::clone(&self.on_demand.ends),
            model_id: model_id.to_owned(),
        }
    }

    /// Record that a load of `model_id` ended — failed with `failure`, or (`None`) reached Ready
    /// or was abandoned — then wake the waiters. Recorded before the wake-up, so a waiter that
    /// wakes finds it. A refusal because the model is already loading (`ModelError::Loading`,
    /// the loser of two racing starts) is not the end of that load and is not recorded (other
    /// refusals of a start are filtered by `load_model_reported`).
    pub(crate) fn note_load_end(&self, model_id: &str, failure: Option<&anyhow::Error>) {
        let error = failure.map(|e| e.downcast_ref::<ModelError>().cloned());
        if error == Some(Some(ModelError::Loading)) {
            return;
        }
        {
            let mut ends = self
                .on_demand
                .ends
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let e = ends.entry(model_id.to_owned()).or_default();
            e.count += 1;
            if let Some(error) = error {
                e.failed_at = e.count;
                e.error = error;
            }
        }
        self.note_load_event();
    }

    /// How `model_id`'s loads have ended so far.
    fn load_ends(&self, model_id: &str) -> LoadEnds {
        self.on_demand
            .ends
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(model_id)
            .cloned()
            .unwrap_or_default()
    }

    /// The state of `model_id`'s record (`None` for an unknown id).
    fn on_demand_state(&self, model_id: &str) -> Option<ModelState> {
        self.read_models("on_demand_wait:state")
            .get(model_id)
            .map(|r| r.state)
    }

    /// When `model_id`'s running load passes its `load_timeout_secs` (`None`: no load running,
    /// or no deadline).
    fn pending_load_deadline(&self, model_id: &str) -> Option<Instant> {
        let timeout = self.load_timeout(model_id)?;
        self.read_models("on_demand_wait:deadline")
            .get(model_id)
            .and_then(|r| r.pending_load.as_ref())
            .map(|slot| slot.started + timeout)
    }

    /// `LoadTimeout` when `model_id`'s running load is past its `load_timeout_secs` — what the
    /// requests waiting for it were answered when it stalled.
    fn overdue_load(&self, model_id: &str) -> Option<ModelError> {
        let timeout = self.load_timeout(model_id)?;
        let deadline = self.pending_load_deadline(model_id)?;
        (Instant::now() >= deadline).then(|| ModelError::LoadTimeout {
            model: model_id.to_owned(),
            secs: timeout.as_secs(),
        })
    }

    /// The immediate answer for a request that would wait for `model_id` while a stall stands in
    /// the way: its own load is past its deadline (`LoadTimeout`), or it is not
    /// loaded and its device has a stalled load (`DeviceStalled`, what its load would be refused
    /// with). `None` otherwise. Reaps the stalled loads first, so a late result counts.
    #[must_use]
    pub fn on_demand_stall_answer(&self, model_id: &str) -> Option<ModelError> {
        self.reap_stalled_loads();
        if let Some(e) = self.overdue_load(model_id) {
            return Some(e);
        }
        let guard = self.read_models("on_demand_wait:stall");
        let r = guard.get(model_id)?;
        if r.state != ModelState::NotLoaded {
            return None;
        }
        Self::stalled_load_on(&guard, &r.device).map(|(model, secs)| ModelError::DeviceStalled {
            device: r.device.clone(),
            model,
            secs,
        })
    }

    /// The waiter place for a wait on `model_id`: the one held, a free one, or `None` (no wait:
    /// [`WaitPlace::NoWait`], or the cap is reached — logged).
    fn waiter_place(&self, model_id: &str, place: WaitPlace) -> Option<WaiterSlot> {
        match place {
            WaitPlace::Held(slot) => Some(slot),
            WaitPlace::NoWait => None,
            WaitPlace::Take => {
                let slot = self.try_on_demand_waiter_slot();
                if slot.is_none() {
                    tracing::info!(
                        model = %model_id,
                        cap = MAX_ON_DEMAND_WAITERS,
                        "on-demand wait: waiter cap reached - answering 503 + Retry-After at once"
                    );
                }
                slot
            }
        }
    }

    /// Start `model_id`'s on-demand load (or join the one running) and wait, at most
    /// [`on_demand_wait_budget`](Self::on_demand_wait_budget), for it to end.
    ///
    /// No wait (the answer of [`request_on_demand_load`](Self::request_on_demand_load), as
    /// before F1b) when the budget is zero, the model is not `on_demand`, a stop was requested,
    /// `place` is [`WaitPlace::NoWait`], or no waiter place is free. A stall in the way
    /// ([`on_demand_stall_answer`](Self::on_demand_stall_answer)) is answered at once. The wait
    /// holds no admission permit and never cancels the load; see the module docs for the
    /// wake-up and cancel-safety rules.
    pub async fn wait_on_demand(
        self: &Arc<Self>,
        model_id: &str,
        place: WaitPlace,
    ) -> OnDemandWait {
        let budget = self.on_demand_wait_budget(model_id);
        if budget.is_zero() || !self.is_on_demand(model_id) || self.is_shutting_down() {
            return OnDemandWait::Refused(self.request_on_demand_load(model_id));
        }
        if let Some(e) = self.on_demand_stall_answer(model_id) {
            tracing::info!(
                model = %model_id,
                error = %e,
                "on-demand wait: a stalled load is in the way - answered at once"
            );
            return OnDemandWait::LoadFailed(Some(e));
        }
        let Some(slot) = self.waiter_place(model_id, place) else {
            return OnDemandWait::Refused(self.request_on_demand_load(model_id));
        };
        // Before the load can start (and end): an end after this point is one this wait must see.
        let baseline = self.load_ends(model_id).count;
        let started = self.request_on_demand_load(model_id);
        if started != OnDemandLoad::Loading {
            return OnDemandWait::Refused(started);
        }
        let t0 = Instant::now();
        let deadline = tokio::time::Instant::now() + budget;
        tracing::debug!(
            model = %model_id,
            wait_ms = u64::try_from(budget.as_millis()).unwrap_or(u64::MAX),
            waiters = self.on_demand_waiters(),
            "on-demand wait: waiting for the model's load"
        );
        let (outcome, why) = loop {
            let notified = self.on_demand.events.notified();
            tokio::pin!(notified);
            // Registered before the reads below: a load ending between them still wakes us.
            notified.as_mut().enable();
            // A stalled load whose result arrived is committed (or discarded) now. Safe here on a
            // poisoned GPU context too (F1b N-3: bookkeeping only; see `reap_stalled_loads`).
            self.reap_stalled_loads();
            if self.is_shutting_down() {
                break (
                    OnDemandWait::Refused(OnDemandLoad::ShuttingDown),
                    "server stopping",
                );
            }
            let ends = self.load_ends(model_id);
            if ends.failed_at > baseline {
                break (OnDemandWait::LoadFailed(ends.error), "load failed");
            }
            if let Some(e) = self.overdue_load(model_id) {
                break (OnDemandWait::LoadFailed(Some(e)), "load past its deadline");
            }
            match self.on_demand_state(model_id) {
                Some(ModelState::Ready) => break (OnDemandWait::Settled, "ready"),
                // A load of this model ended since the wait started and the model is not Ready:
                // the load was abandoned, or the model was loaded and already evicted. The
                // caller re-checks once.
                Some(ModelState::NotLoaded | ModelState::Evicting) if ends.count > baseline => {
                    break (OnDemandWait::Settled, "load ended, model not ready");
                }
                // Nothing is loading it (an eviction ended, the load that was running ended
                // before this wait counted it): start it again, within the same wait.
                Some(ModelState::NotLoaded) if ends.in_flight == 0 => {
                    let again = self.request_on_demand_load(model_id);
                    if again != OnDemandLoad::Loading {
                        break (OnDemandWait::Refused(again), "load not restartable");
                    }
                }
                // `NotLoaded` with a load call under way is that call before `begin_load`.
                Some(ModelState::Loading | ModelState::NotLoaded | ModelState::Evicting) => {}
                None => {
                    break (
                        OnDemandWait::LoadFailed(Some(ModelError::NotFound(model_id.to_owned()))),
                        "model removed",
                    );
                }
            }
            // Wake at the load's own deadline too: the next turn answers it as overdue.
            let load_deadline = self.pending_load_deadline(model_id).map_or(deadline, |at| {
                deadline.min(tokio::time::Instant::from_std(at))
            });
            tokio::select! {
                () = &mut notified => {}
                () = tokio::time::sleep_until(deadline) => {
                    break (OnDemandWait::Refused(OnDemandLoad::Loading), "wait expired");
                }
                () = tokio::time::sleep_until(load_deadline), if load_deadline < deadline => {}
                // The next turn answers `ShuttingDown`.
                () = crate::startup::stop_requested() => {}
            }
        };
        drop(slot);
        tracing::info!(
            model = %model_id,
            waited_ms = u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX),
            outcome = why,
            "on-demand wait: done"
        );
        outcome
    }

    /// Tests: make the wait `wait` (milliseconds precision) instead of `on_demand_wait_secs`.
    #[cfg(test)]
    pub(crate) fn set_on_demand_wait_for_test(&self, wait: Duration) {
        *self
            .on_demand
            .budget_override
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(wait);
    }

    /// Tests: the SSE keep-alive interval of this manager's waited streams.
    #[cfg(test)]
    pub(crate) fn set_keep_alive_for_test(&self, every: Duration) {
        *self
            .on_demand
            .keep_alive_override
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(every);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]

    use std::path::Path;

    use super::*;
    use crate::model_manager::config;
    use crate::model_manager::engine::ModelKind;
    use crate::model_manager::lifecycle::{EngineFactory, LoadGate, MockEngineFactory};
    use crate::model_manager::{EngineHandleKind, ModelState};

    /// A mock factory that counts `load` calls (the gate and failure knobs are the mock's).
    struct CountingFactory {
        inner: MockEngineFactory,
        loads: Arc<AtomicUsize>,
    }

    impl EngineFactory for CountingFactory {
        fn detect_kind(&self, model_dir: &Path) -> ModelKind {
            self.inner.detect_kind(model_dir)
        }
        fn load(
            &self,
            model_path: &Path,
            device: &str,
            cache_size_gb: f64,
            max_num_seqs: usize,
        ) -> anyhow::Result<(EngineHandleKind, std::thread::JoinHandle<()>)> {
            self.loads.fetch_add(1, Ordering::SeqCst);
            self.inner
                .load(model_path, device, cache_size_gb, max_num_seqs)
        }
    }

    struct Rig {
        mm: Arc<ModelManager>,
        gate: Arc<LoadGate>,
        loads: Arc<AtomicUsize>,
    }

    /// One `on_demand` model "m" (1 GB of 10), loads held at a gate the test releases, the wait
    /// set to `wait`. `fail`: the load fails (a non-`ModelError`) once the gate opens.
    async fn rig(wait: Duration, fail: bool) -> Rig {
        let mut cfg = crate::model_manager::tests::make_config(&[("m", 1.0)], 10.0);
        cfg.models.get_mut("m").unwrap().policy.load = config::LoadPolicy::OnDemand;
        let gate = Arc::new(LoadGate::default());
        let loads = Arc::new(AtomicUsize::new(0));
        let factory = CountingFactory {
            inner: MockEngineFactory {
                load_gate: Some(Arc::clone(&gate)),
                should_fail: fail,
                ..Default::default()
            },
            loads: Arc::clone(&loads),
        };
        let mm = Arc::new(ModelManager::new(cfg, Arc::new(factory)).await.unwrap());
        mm.set_on_demand_wait_for_test(wait);
        Rig { mm, gate, loads }
    }

    fn state(mm: &ModelManager, id: &str) -> ModelState {
        mm.on_demand_state(id).unwrap()
    }

    /// Poll (5 ms) until `cond` holds; panics after 5 s.
    async fn until(what: &str, mut cond: impl FnMut() -> bool) {
        for _ in 0..1000 {
            if cond() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("timed out waiting for: {what}");
    }

    const LONG: Duration = Duration::from_secs(30);

    #[test]
    fn the_wait_stays_below_the_load_timeout_by_the_margin() {
        let s = Duration::from_secs;
        assert_eq!(clamp_to_load_timeout(s(120), Some(s(600))), s(120));
        assert_eq!(clamp_to_load_timeout(s(300), Some(s(60))), s(58));
        assert_eq!(clamp_to_load_timeout(s(300), None), s(300));
        assert_eq!(clamp_to_load_timeout(s(10), Some(s(1))), Duration::ZERO);
    }

    /// The budget: `on_demand_wait_secs`, clamped by a model's own shorter `load_timeout_secs`
    /// (per-model override) minus the margin; a model with no deadline keeps the full wait.
    #[tokio::test]
    async fn a_shorter_per_model_load_timeout_clamps_the_wait() {
        let mut cfg = crate::model_manager::tests::make_config(
            &[("short", 1.0), ("plain", 1.0), ("nodeadline", 1.0)],
            10.0,
        );
        cfg.on_demand_wait_secs = 300;
        cfg.load_timeout_secs = 600;
        cfg.models
            .get_mut("short")
            .unwrap()
            .policy
            .load_timeout_secs = Some(60);
        cfg.models
            .get_mut("nodeadline")
            .unwrap()
            .policy
            .load_timeout_secs = Some(0);
        let mm = ModelManager::new(cfg, Arc::new(MockEngineFactory::default()))
            .await
            .unwrap();
        assert_eq!(mm.on_demand_wait_budget("short"), Duration::from_secs(58));
        assert_eq!(mm.on_demand_wait_budget("plain"), Duration::from_secs(300));
        assert_eq!(
            mm.on_demand_wait_budget("nodeadline"),
            Duration::from_secs(300)
        );
    }

    /// Wait 0: no wait — the load starts and the answer is today's `Loading` at once; no waiter
    /// place is taken.
    #[tokio::test]
    async fn wait_zero_starts_the_load_and_answers_at_once() {
        let r = rig(Duration::ZERO, false).await;
        assert_eq!(
            r.mm.wait_on_demand("m", WaitPlace::Take).await,
            OnDemandWait::Refused(OnDemandLoad::Loading)
        );
        assert_eq!(r.mm.on_demand_waiters(), 0);
        until("the load started", || {
            state(&r.mm, "m") == ModelState::Loading
        })
        .await;
        r.gate.release();
        until("Ready", || state(&r.mm, "m") == ModelState::Ready).await;
    }

    /// An eager model never waits and never loads (today's `NotApplicable`).
    #[tokio::test]
    async fn an_eager_model_does_not_wait() {
        let r = rig(LONG, false).await;
        // A second, eager model on the same manager.
        let entry = config::ModelEntry {
            vram_gb: 1.0,
            kind: None,
            policy: config::ModelPolicy::default(),
        };
        r.mm.dynamic_entries
            .write()
            .unwrap()
            .insert("m".to_owned(), entry);
        assert_eq!(
            r.mm.wait_on_demand("m", WaitPlace::Take).await,
            OnDemandWait::Refused(OnDemandLoad::NotApplicable)
        );
        assert_eq!(state(&r.mm, "m"), ModelState::NotLoaded);
        assert_eq!(r.mm.on_demand_waiters(), 0);
    }

    /// Wait > 0: the request holds while the load runs and is told to proceed once Ready; the
    /// waiter place is released.
    #[tokio::test]
    async fn the_wait_ends_when_the_model_is_ready() {
        let r = rig(LONG, false).await;
        let mm = Arc::clone(&r.mm);
        let waiter = tokio::spawn(async move { mm.wait_on_demand("m", WaitPlace::Take).await });
        until("one waiter", || r.mm.on_demand_waiters() == 1).await;
        assert!(!waiter.is_finished(), "holds while the load runs");
        r.gate.release();
        assert_eq!(waiter.await.unwrap(), OnDemandWait::Settled);
        assert_eq!(state(&r.mm, "m"), ModelState::Ready);
        assert_eq!(r.mm.on_demand_waiters(), 0);
    }

    /// A failed load ends the wait with that load's error (`None`: not a `ModelError`).
    #[tokio::test]
    async fn a_failed_load_ends_the_wait_with_its_error() {
        let r = rig(LONG, true).await;
        let mm = Arc::clone(&r.mm);
        let waiter = tokio::spawn(async move { mm.wait_on_demand("m", WaitPlace::Take).await });
        until("one waiter", || r.mm.on_demand_waiters() == 1).await;
        r.gate.release();
        assert_eq!(waiter.await.unwrap(), OnDemandWait::LoadFailed(None));
        assert_eq!(state(&r.mm, "m"), ModelState::NotLoaded);
        assert_eq!(r.mm.on_demand_waiters(), 0);
    }

    /// A load refused with a `ModelError` (no room: 20 GB on a 10 GB device) hands the waiter
    /// that error, for the route's usual mapping (503 `insufficient_memory`).
    #[tokio::test]
    async fn a_model_error_from_the_load_reaches_the_waiter() {
        let mut cfg = crate::model_manager::tests::make_config(&[("big", 20.0)], 10.0);
        cfg.models.get_mut("big").unwrap().policy.load = config::LoadPolicy::OnDemand;
        let mm = Arc::new(
            ModelManager::new(cfg, Arc::new(MockEngineFactory::default()))
                .await
                .unwrap(),
        );
        mm.set_on_demand_wait_for_test(LONG);
        let out = tokio::time::timeout(
            Duration::from_secs(5),
            mm.wait_on_demand("big", WaitPlace::Take),
        )
        .await
        .unwrap();
        assert!(
            matches!(
                out,
                OnDemandWait::LoadFailed(Some(ModelError::InsufficientMemory(_)))
            ),
            "{out:?}"
        );
        assert_eq!(mm.on_demand_waiters(), 0);
    }

    /// Expiry: the wait ends with today's `Loading` answer; the load keeps running.
    #[tokio::test]
    async fn an_expired_wait_answers_loading_and_leaves_the_load_running() {
        let r = rig(Duration::from_millis(150), false).await;
        let t0 = Instant::now();
        assert_eq!(
            r.mm.wait_on_demand("m", WaitPlace::Take).await,
            OnDemandWait::Refused(OnDemandLoad::Loading)
        );
        assert!(t0.elapsed() >= Duration::from_millis(150));
        assert_eq!(r.mm.on_demand_waiters(), 0);
        assert_eq!(
            state(&r.mm, "m"),
            ModelState::Loading,
            "the load still runs"
        );
        r.gate.release();
        until("Ready", || state(&r.mm, "m") == ModelState::Ready).await;
    }

    /// The cap: with every place taken a request is answered at once (the load still starts),
    /// and places come back as they are dropped.
    #[tokio::test]
    async fn the_waiter_cap_answers_at_once() {
        let r = rig(LONG, false).await;
        let slots: Vec<WaiterSlot> = (0..MAX_ON_DEMAND_WAITERS)
            .map(|_| r.mm.try_on_demand_waiter_slot().unwrap())
            .collect();
        assert!(r.mm.try_on_demand_waiter_slot().is_none(), "the 65th place");
        let out = tokio::time::timeout(
            Duration::from_secs(2),
            r.mm.wait_on_demand("m", WaitPlace::Take),
        )
        .await
        .expect("no wait at the cap");
        assert_eq!(out, OnDemandWait::Refused(OnDemandLoad::Loading));
        assert_eq!(r.mm.on_demand_waiters(), MAX_ON_DEMAND_WAITERS);
        until("the load started", || {
            state(&r.mm, "m") == ModelState::Loading
        })
        .await;
        drop(slots);
        assert_eq!(r.mm.on_demand_waiters(), 0);
        r.gate.release();
    }

    /// Cancel safety: a waiter whose request is dropped (client gone) gives its place back at
    /// once, and the load it started is not cancelled — another waiter still gets the model.
    #[tokio::test]
    async fn a_dropped_waiter_releases_its_place_and_not_the_load() {
        let r = rig(LONG, false).await;
        let mm = Arc::clone(&r.mm);
        let gone = tokio::spawn(async move { mm.wait_on_demand("m", WaitPlace::Take).await });
        let mm = Arc::clone(&r.mm);
        let stays = tokio::spawn(async move { mm.wait_on_demand("m", WaitPlace::Take).await });
        until("two waiters", || r.mm.on_demand_waiters() == 2).await;
        gone.abort();
        assert!(gone.await.unwrap_err().is_cancelled());
        until("the place came back", || r.mm.on_demand_waiters() == 1).await;
        assert_eq!(state(&r.mm, "m"), ModelState::Loading, "the load runs on");
        r.gate.release();
        assert_eq!(stays.await.unwrap(), OnDemandWait::Settled);
        assert_eq!(r.mm.on_demand_waiters(), 0);
        assert_eq!(r.loads.load(Ordering::SeqCst), 1);
    }

    /// Many waiters on one load: one engine is built and every waiter wakes.
    #[tokio::test]
    async fn many_waiters_on_one_load_all_wake() {
        let r = rig(LONG, false).await;
        let waiters: Vec<_> = (0..8)
            .map(|_| {
                let mm = Arc::clone(&r.mm);
                tokio::spawn(async move { mm.wait_on_demand("m", WaitPlace::Take).await })
            })
            .collect();
        until("eight waiters", || r.mm.on_demand_waiters() == 8).await;
        r.gate.release();
        for w in waiters {
            assert_eq!(w.await.unwrap(), OnDemandWait::Settled);
        }
        assert_eq!(r.loads.load(Ordering::SeqCst), 1, "one load for all");
        assert_eq!(r.mm.on_demand_waiters(), 0);
    }

    /// A load that ends without Ready and without an error (abandoned: its future was dropped
    /// before the engine build) ends the wait as `Settled`, so the caller re-checks once.
    #[tokio::test]
    async fn an_abandoned_load_ends_the_wait_for_a_recheck() {
        let r = rig(LONG, false).await;
        // A load already running (not one this wait starts).
        r.mm.write_models("test").get_mut("m").unwrap().state = ModelState::Loading;
        let mm = Arc::clone(&r.mm);
        let waiter = tokio::spawn(async move { mm.wait_on_demand("m", WaitPlace::Take).await });
        until("one waiter", || r.mm.on_demand_waiters() == 1).await;
        r.mm.write_models("test").get_mut("m").unwrap().state = ModelState::NotLoaded;
        r.mm.note_load_end("m", None);
        assert_eq!(waiter.await.unwrap(), OnDemandWait::Settled);
        assert_eq!(r.mm.on_demand_waiters(), 0);
        assert_eq!(
            r.loads.load(Ordering::SeqCst),
            0,
            "the wait itself started none"
        );
        r.gate.release();
    }

    /// A stop ends every wait with `ShuttingDown` at once.
    #[tokio::test]
    async fn a_stop_ends_the_wait() {
        let r = rig(LONG, false).await;
        let mm = Arc::clone(&r.mm);
        let waiter = tokio::spawn(async move { mm.wait_on_demand("m", WaitPlace::Take).await });
        until("one waiter", || r.mm.on_demand_waiters() == 1).await;
        r.mm.begin_shutdown();
        let out = tokio::time::timeout(Duration::from_secs(2), waiter)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(out, OnDemandWait::Refused(OnDemandLoad::ShuttingDown));
        assert_eq!(r.mm.on_demand_waiters(), 0);
        r.gate.release();
    }

    /// A duplicate start refused with `Loading` is not the end of the running load.
    #[test]
    fn a_duplicate_start_is_not_a_load_end() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let r = rig(LONG, false).await;
            r.mm.note_load_end("m", Some(&anyhow::Error::new(ModelError::Loading)));
            assert_eq!(r.mm.load_ends("m").count, 0);
            r.mm.note_load_end("m", Some(&anyhow::anyhow!("boom")));
            let ends = r.mm.load_ends("m");
            assert_eq!((ends.count, ends.failed_at, ends.error), (1, 1, None));
        });
    }

    fn record_of<T>(
        mm: &ModelManager,
        f: impl FnOnce(&crate::model_manager::ModelRecord) -> T,
    ) -> T {
        f(mm.read_models("test").get("m").unwrap())
    }

    /// Review R-1: a load past its deadline answers the request that waits
    /// for it with `load_timeout` (woken at the deadline), and every later request at once —
    /// no wait, no waiter place, however many retry — until the stalled load returns.
    #[tokio::test]
    async fn a_request_after_the_stall_is_answered_at_once_with_the_stall() {
        // `m`'s own deadline: 3 s, so the wait is 3 - 2 = 1 s.
        let mut cfg = crate::model_manager::tests::make_config(&[("m", 1.0)], 10.0);
        let policy = &mut cfg.models.get_mut("m").unwrap().policy;
        policy.load = config::LoadPolicy::OnDemand;
        policy.load_timeout_secs = Some(3);
        let gate = Arc::new(LoadGate::default());
        let factory = MockEngineFactory {
            load_gate: Some(Arc::clone(&gate)),
            ..Default::default()
        };
        let mm = Arc::new(ModelManager::new(cfg, Arc::new(factory)).await.unwrap());
        mm.set_on_demand_wait_for_test(LONG);
        assert_eq!(mm.on_demand_wait_budget("m"), Duration::from_secs(1));

        assert_eq!(mm.request_on_demand_load("m"), OnDemandLoad::Loading);
        until("constructing", || {
            record_of(&mm, |x| x.pending_load.is_some())
        })
        .await;
        // A request that arrives late crosses the load's deadline while it waits.
        tokio::time::sleep(Duration::from_millis(2500)).await;
        let late = tokio::spawn({
            let mm = Arc::clone(&mm);
            async move { mm.wait_on_demand("m", WaitPlace::Take).await }
        });
        let out = tokio::time::timeout(Duration::from_secs(3), late)
            .await
            .expect("woken at the load's deadline")
            .unwrap();
        assert!(
            matches!(
                out,
                OnDemandWait::LoadFailed(Some(ModelError::LoadTimeout { .. }))
            ),
            "{out:?}"
        );
        until("booked as stalled", || record_of(&mm, |x| x.load_stalled)).await;

        // Retries after the stall: answered at once, no place taken (the cap cannot be pinned).
        for _ in 0..(2 * MAX_ON_DEMAND_WAITERS) {
            let t = Instant::now();
            let out = mm.wait_on_demand("m", WaitPlace::Take).await;
            assert!(
                matches!(
                    out,
                    OnDemandWait::LoadFailed(Some(ModelError::LoadTimeout { .. }))
                ),
                "{out:?}"
            );
            assert!(t.elapsed() < Duration::from_millis(200));
            assert_eq!(mm.on_demand_waiters(), 0);
        }
        assert!(
            mm.try_on_demand_waiter_slot().is_some(),
            "every place is free"
        );

        // The stalled load returns: the next request reaps it and is served.
        gate.release();
        until("construction ended", || {
            record_of(&mm, |x| {
                x.pending_load.as_ref().is_none_or(|s| s.is_finished())
            })
        })
        .await;
        let out = tokio::time::timeout(
            Duration::from_secs(5),
            mm.wait_on_demand("m", WaitPlace::Take),
        )
        .await
        .unwrap();
        assert_eq!(out, OnDemandWait::Settled);
        assert_eq!(state(&mm, "m"), ModelState::Ready);
    }

    /// Review R-1: a load booked as stalled because its caller went away (healthy, inside its
    /// deadline) is waited for; when its construction ends the waiter wakes, the late result
    /// is committed, and the request is served.
    #[tokio::test]
    async fn a_stalled_load_that_returns_wakes_the_waiting_request() {
        let r = rig(LONG, false).await;
        let admin = tokio::spawn({
            let mm = Arc::clone(&r.mm);
            async move { mm.load_model("m").await }
        });
        until("constructing", || {
            record_of(&r.mm, |x| x.pending_load.is_some())
        })
        .await;
        admin.abort();
        let _ = admin.await;
        until("booked as stalled", || record_of(&r.mm, |x| x.load_stalled)).await;
        assert_eq!(
            r.mm.on_demand_stall_answer("m"),
            None,
            "inside its deadline"
        );

        let waiter = tokio::spawn({
            let mm = Arc::clone(&r.mm);
            async move { mm.wait_on_demand("m", WaitPlace::Take).await }
        });
        until("waiting", || r.mm.on_demand_waiters() == 1).await;
        r.gate.release();
        let out = tokio::time::timeout(Duration::from_secs(5), waiter)
            .await
            .expect("woken when the construction ended")
            .unwrap();
        assert_eq!(out, OnDemandWait::Settled);
        assert_eq!(state(&r.mm, "m"), ModelState::Ready);
        assert_eq!(r.loads.load(Ordering::SeqCst), 1);
    }

    /// F1b N-3: the wait loop reaps stalled loads on a request task, and the GPU context can be
    /// poisoned by then (`teardown_engines` skips its reap in that case). The reap is safe there:
    /// committing a late result is bookkeeping only (no engine call, no `OpenVINO` call; the KV
    /// warm-up ran inside the load), and it stores the engine instead of dropping it, so no
    /// pipeline destructor runs on the broken context. The stop that follows skips the teardown
    /// (`shutdown` on a poisoned context), so the engine is still not destroyed.
    #[tokio::test]
    async fn reaping_a_late_load_on_a_poisoned_context_commits_and_never_drops_the_engine() {
        let r = rig(LONG, false).await;
        let admin = tokio::spawn({
            let mm = Arc::clone(&r.mm);
            async move { mm.load_model("m").await }
        });
        until("constructing", || {
            record_of(&r.mm, |x| x.pending_load.is_some())
        })
        .await;
        admin.abort();
        let _ = admin.await;
        until("booked as stalled", || record_of(&r.mm, |x| x.load_stalled)).await;

        let waiter = tokio::spawn({
            let mm = Arc::clone(&r.mm);
            async move { mm.wait_on_demand("m", WaitPlace::Take).await }
        });
        until("waiting", || r.mm.on_demand_waiters() == 1).await;
        r.mm.gpu_poisoned.store(true, Ordering::SeqCst);
        r.gate.release();
        // The waiter's own loop reaps the late result (nothing else touches the manager here).
        let out = tokio::time::timeout(Duration::from_secs(5), waiter)
            .await
            .expect("woken when the construction ended")
            .unwrap();
        assert_eq!(out, OnDemandWait::Settled);
        assert_eq!(state(&r.mm, "m"), ModelState::Ready);
        // Kept, not dropped: the mock engine thread runs until its handle is dropped.
        assert!(record_of(&r.mm, |x| x.handle.is_some()));
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            record_of(&r.mm, |x| x
                .thread
                .as_ref()
                .is_some_and(|t| !t.is_finished())),
            "the engine was not torn down by the reap"
        );
        // A stop on the poisoned context tears nothing down either.
        let report = r.mm.shutdown().await;
        assert!(report.unclean.is_empty());
        assert_eq!(state(&r.mm, "m"), ModelState::Ready);
        assert!(record_of(&r.mm, |x| x
            .thread
            .as_ref()
            .is_some_and(|t| !t.is_finished())));
        r.mm.gpu_poisoned.store(false, Ordering::SeqCst);
    }

    /// Review R-2: a second start of a model that is loading, refused by `begin_load` for
    /// another reason (here a poisoned GPU), is not the end of the running load.
    #[tokio::test]
    async fn a_refused_second_start_is_not_the_running_loads_end() {
        let r = rig(LONG, false).await;
        let waiter = tokio::spawn({
            let mm = Arc::clone(&r.mm);
            async move { mm.wait_on_demand("m", WaitPlace::Take).await }
        });
        until("loading", || {
            r.mm.on_demand_waiters() == 1 && state(&r.mm, "m") == ModelState::Loading
        })
        .await;
        r.mm.gpu_poisoned.store(true, Ordering::SeqCst);
        let refused = r.mm.load_model("m").await.unwrap_err();
        assert_eq!(
            refused.downcast_ref::<ModelError>(),
            Some(&ModelError::GpuPoisoned)
        );
        assert_eq!(r.mm.load_ends("m").count, 0, "not recorded");
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!waiter.is_finished(), "still waiting for the running load");
        r.mm.gpu_poisoned.store(false, Ordering::SeqCst);
        r.gate.release();
        let out = tokio::time::timeout(Duration::from_secs(5), waiter)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(out, OnDemandWait::Settled);
    }

    /// F1b N-2: whether a refused start ends a load is decided from the state read under the
    /// refusal's own lock, not afterwards. The test opens the old window on purpose: a start is
    /// refused while another load owns the model, then that load reaches Ready before the
    /// refusal is looked at. A read afterwards would now say "not loading" (and the old code
    /// recorded the refusal as the load's end, handing a waiter the error for a loaded model);
    /// the captured flag still says another load owned it. A start refused when the owner has
    /// already reached Ready (the poison check runs before the state is read; a stalled load on
    /// the device is checked before the Ready answer) ends no load either: the waiters the
    /// Ready woke must not find a failure recorded after it. From `NotLoaded` the refusal is
    /// this start's own end and is recorded.
    #[tokio::test]
    async fn a_refusals_load_end_is_decided_under_the_refusals_lock() {
        let r = rig(LONG, false).await;
        let owner = tokio::spawn({
            let mm = Arc::clone(&r.mm);
            async move { mm.load_model("m").await }
        });
        until("loading", || state(&r.mm, "m") == ModelState::Loading).await;
        r.mm.gpu_poisoned.store(true, Ordering::SeqCst);
        let refusal = r.mm.begin_load("m").unwrap_err();
        r.mm.gpu_poisoned.store(false, Ordering::SeqCst);
        // Under the refusal's lock: the owner's load.
        assert!(refusal.ends_no_load);
        r.gate.release();
        owner.await.unwrap().unwrap();
        assert_eq!(state(&r.mm, "m"), ModelState::Ready);
        assert_ne!(
            r.mm.model_state("m"),
            Some(ModelState::Loading),
            "a Loading-only read now would take the refusal for this start's own end"
        );
        // `load_model_reported` records the refusal only when this flag is false; the owner's
        // load ended as a success, with no failure on record.
        assert_eq!(r.mm.load_ends("m").failed_at, 0);

        // Refused from Ready (the owner finished first): not recorded, through the wrapper.
        let before = r.mm.load_ends("m");
        r.mm.gpu_poisoned.store(true, Ordering::SeqCst);
        let err = r.mm.load_model("m").await.unwrap_err();
        r.mm.gpu_poisoned.store(false, Ordering::SeqCst);
        assert_eq!(
            err.downcast_ref::<ModelError>(),
            Some(&ModelError::GpuPoisoned)
        );
        let after = r.mm.load_ends("m");
        assert_eq!(
            (after.count, after.failed_at),
            (before.count, 0),
            "a refusal at Ready ends no load"
        );
        assert_eq!(state(&r.mm, "m"), ModelState::Ready);

        // Refused from NotLoaded (nothing owns the model): recorded as this start's end.
        let r = rig(LONG, false).await;
        r.mm.gpu_poisoned.store(true, Ordering::SeqCst);
        let refusal = r.mm.begin_load("m").unwrap_err();
        assert!(!refusal.ends_no_load);
        let err = r.mm.load_model("m").await.unwrap_err();
        assert_eq!(
            err.downcast_ref::<ModelError>(),
            Some(&ModelError::GpuPoisoned)
        );
        let ends = r.mm.load_ends("m");
        assert_eq!((ends.count, ends.failed_at), (1, 1), "recorded once");
        r.mm.gpu_poisoned.store(false, Ordering::SeqCst);
    }

    /// Review R-3: a request waiting on a model that is being evicted is woken when the eviction
    /// ends, finds nothing loading it, and starts the load (within the same wait).
    #[tokio::test]
    async fn an_eviction_that_ends_while_a_request_waits_restarts_the_load() {
        let r = rig(LONG, false).await;
        r.mm.write_models("test").get_mut("m").unwrap().state = ModelState::Evicting;
        let waiter = tokio::spawn({
            let mm = Arc::clone(&r.mm);
            async move { mm.wait_on_demand("m", WaitPlace::Take).await }
        });
        until("waiting", || r.mm.on_demand_waiters() == 1).await;
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(
            r.loads.load(Ordering::SeqCst),
            0,
            "nothing loads an evicting model"
        );
        ModelManager::reconcile_evicted(&r.mm.models, &r.mm.vram, "m", &r.mm.on_demand_waker());
        until("load restarted", || r.loads.load(Ordering::SeqCst) == 1).await;
        r.gate.release();
        let out = tokio::time::timeout(Duration::from_secs(5), waiter)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(out, OnDemandWait::Settled);
        assert_eq!(state(&r.mm, "m"), ModelState::Ready);
    }

    /// P-C review R-2: an admin `/load` refused because the model is being evicted is not a load
    /// end. The request waiting through the eviction is not failed by it (0.9.1 before this:
    /// `LoadFailed(None)`, a 500 "failed to load"): it restarts the load when the eviction ends
    /// and is served.
    #[tokio::test]
    async fn a_load_refused_while_evicting_does_not_fail_the_waiting_request() {
        let r = rig(LONG, false).await;
        r.mm.write_models("test").get_mut("m").unwrap().state = ModelState::Evicting;
        let waiter = tokio::spawn({
            let mm = Arc::clone(&r.mm);
            async move { mm.wait_on_demand("m", WaitPlace::Take).await }
        });
        until("waiting", || r.mm.on_demand_waiters() == 1).await;
        let refused = r.mm.load_model("m").await.unwrap_err();
        assert!(refused.to_string().contains("being evicted"), "{refused:#}");
        assert_eq!(r.mm.load_ends("m").failed_at, 0, "not recorded");
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(!waiter.is_finished(), "still waiting through the eviction");
        ModelManager::reconcile_evicted(&r.mm.models, &r.mm.vram, "m", &r.mm.on_demand_waker());
        until("load restarted", || r.loads.load(Ordering::SeqCst) == 1).await;
        r.gate.release();
        let out = tokio::time::timeout(Duration::from_secs(5), waiter)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(out, OnDemandWait::Settled);
        assert_eq!(state(&r.mm, "m"), ModelState::Ready);
    }

    /// Review R-3: a model removed while a request waits for it: the request is woken and gets
    /// `NotFound` (the route's 404).
    #[tokio::test]
    async fn a_model_removed_while_a_request_waits_answers_not_found() {
        let r = rig(LONG, false).await;
        r.mm.write_models("test").get_mut("m").unwrap().state = ModelState::Evicting;
        let waiter = tokio::spawn({
            let mm = Arc::clone(&r.mm);
            async move { mm.wait_on_demand("m", WaitPlace::Take).await }
        });
        until("waiting", || r.mm.on_demand_waiters() == 1).await;
        {
            // The eviction ends unseen (no wake), then the model is removed.
            let _no_restart = r.mm.load_in_flight("m");
            r.mm.write_models("test").get_mut("m").unwrap().state = ModelState::NotLoaded;
            r.mm.deregister_model("m").await.unwrap();
        }
        let out = tokio::time::timeout(Duration::from_secs(5), waiter)
            .await
            .expect("woken by the removal")
            .unwrap();
        assert_eq!(
            out,
            OnDemandWait::LoadFailed(Some(ModelError::NotFound("m".to_owned())))
        );
        assert_eq!(r.mm.on_demand_waiters(), 0);
        assert_eq!(r.loads.load(Ordering::SeqCst), 0);
    }

    /// The cap under contention: many threads taking places at once never get more than
    /// [`MAX_ON_DEMAND_WAITERS`], and every place comes back when dropped.
    #[test]
    fn racing_requests_never_pass_the_cap() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let mm = rt.block_on(rig(LONG, false)).mm;
        let threads = 16;
        let barrier = Arc::new(std::sync::Barrier::new(threads));
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let (mm, barrier) = (Arc::clone(&mm), Arc::clone(&barrier));
                std::thread::spawn(move || {
                    barrier.wait();
                    (0..20)
                        .filter_map(|_| mm.try_on_demand_waiter_slot())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let slots: Vec<WaiterSlot> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();
        assert_eq!(slots.len(), MAX_ON_DEMAND_WAITERS);
        assert_eq!(mm.on_demand_waiters(), MAX_ON_DEMAND_WAITERS);
        drop(slots);
        assert_eq!(mm.on_demand_waiters(), 0);
    }
}
