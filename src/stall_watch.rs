// ============================================================
// src/stall_watch.rs — the engine liveness watchdog (0.9.1 S1)
// ============================================================
// A GPU fault on the B50 once left one request sending SSE keep-alives for two hours: the engine
// thread sat inside an OpenVINO call that never returned, and nothing in the request path bounds a
// generation that makes no progress. `--supervise` catches that eventually, but `rv start` runs
// unsupervised by default and Windows has no supervisor at all.
//
// The verdict belongs to the ENGINE, not to a request (S1 review). Each engine
// instance (CB text, VLM, NPU) owns one `EngineLiveness`, shared by its handle clones and its
// engine thread. The thread reports, with a few relaxed atomics and no lock:
//   * `begin`    — an OpenVINO call starts (VLM / NPU: one generate call; CB: a busy period —
//                  the first AddRequest or step after the engine was idle);
//   * `progress` — the engine moved (CB: every completed step, whether or not any sequence got
//                  visible text — chunked prefill and a starved sequence in a stepping batch are
//                  progress; VLM / NPU: every streamer callback, i.e. visible text only);
//   * `end`      — the engine is idle again.
// One watcher task per engine reads those counters on its own (tokio) clock:
//   * first-token deadline (`stream_first_token_timeout_secs`): the longest a call may run before
//     its first progress signal;
//   * gap deadline (`stream_token_gap_timeout_secs`): the longest between two progress signals
//     while the engine has work.
// An idle engine is never stalled. Nothing here looks at a client connection: a client that
// leaves does not end the watching (review F1).
//
// When a deadline passes the engine is marked dead, once: one ERROR line, the model-level action
// (`ModelManager::on_generation_stalled`), and every request registered on the engine gets a
// final `Error(generation_stalled…)` + `Done` on its own channel. OpenVINO cannot abort a hung
// call, so the thread stays stuck; the request path refuses the model from then on.
//
// CRASH COURSE — why counters and not timestamps: the engine threads are plain OS threads, the
// watcher is a tokio task. If the threads wrote `Instant`s, the watcher would compare its clock
// with theirs — and in a test with paused tokio time those clocks differ. Counters carry no time
// at all: the watcher notes WHEN it saw a counter change on its own clock. A stall is therefore
// never declared early (the change happened no later than it was seen) and at most one tick late.
//
// CRASH COURSE — `WeakSender`: the registry keeps each request's sender as a `WeakSender`, which
// does not keep the channel open. When the engine finishes a request and drops its sender, the
// consumer still sees the channel close as before; the registry just fails to upgrade that entry.
// ============================================================

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use tokio::sync::{Notify, watch};
use tokio::time::Instant;

use crate::model_manager::Config;
use crate::model_manager::ModelManager;
use crate::ov_cb::FinishReason;
use crate::streaming::{StreamEvent, TokenSender};

/// Prefix of the `StreamEvent::Error` text a stall verdict sends. The inference error mapping
/// recognises it first (`handlers::error`) and answers `generation_stalled`.
pub(crate) const GENERATION_STALLED_MARKER: &str = "generation_stalled:";

/// Whether an engine error text is a stall verdict.
pub(crate) fn is_generation_stalled_error(raw: &str) -> bool {
    raw.starts_with(GENERATION_STALLED_MARKER)
}

/// The error text every request on a stalled engine receives.
pub(crate) fn stalled_error_text() -> String {
    format!(
        "{GENERATION_STALLED_MARKER} the model's engine stopped making progress and is marked dead"
    )
}

/// Which of the two deadlines passed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StallDeadline {
    /// A call made no progress within `stream_first_token_timeout_secs` of its start.
    FirstToken,
    /// No progress within `stream_token_gap_timeout_secs` of the previous progress signal.
    TokenGap,
}

impl StallDeadline {
    /// The config key that sets this deadline (log field).
    #[must_use]
    pub fn config_key(self) -> &'static str {
        match self {
            Self::FirstToken => "stream_first_token_timeout_secs",
            Self::TokenGap => "stream_token_gap_timeout_secs",
        }
    }
}

/// The two deadlines; `None` = that deadline is off (`0` in the config).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WatchSpec {
    pub(crate) first_token: Option<Duration>,
    pub(crate) token_gap: Option<Duration>,
}

/// Longest sampling interval of a busy engine.
const MAX_TICK: Duration = Duration::from_secs(5);
/// Shortest sampling interval (tests run sub-second deadlines).
const MIN_TICK: Duration = Duration::from_millis(5);

impl WatchSpec {
    /// The deadlines the config sets.
    pub(crate) fn from_config(cfg: &Config) -> Self {
        let secs = |s: u64| (s > 0).then(|| Duration::from_secs(s));
        Self {
            first_token: secs(cfg.stream_first_token_timeout_secs),
            token_gap: secs(cfg.stream_token_gap_timeout_secs),
        }
    }

    fn is_off(self) -> bool {
        self.first_token.is_none() && self.token_gap.is_none()
    }

    /// How often a busy engine's counters are sampled: a twentieth of the shorter deadline,
    /// between [`MIN_TICK`] and [`MAX_TICK`]. A stall is declared at most one tick late.
    fn tick(self) -> Duration {
        let shortest = match (self.first_token, self.token_gap) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => MAX_TICK,
        };
        (shortest / 20).clamp(MIN_TICK, MAX_TICK)
    }
}

/// One request registered on an engine: it receives the verdict if the engine stalls.
#[derive(Debug)]
struct Registered {
    tx: tokio::sync::mpsc::WeakSender<StreamEvent>,
    request_id: String,
}

/// The state one engine instance shares between its handle clones, its engine thread and its
/// watcher (the watcher holds it weakly).
#[derive(Debug)]
struct Shared {
    /// True while the engine has work (between `begin` and `end`).
    busy: AtomicBool,
    /// Calls (VLM/NPU) or busy periods (CB) begun.
    calls: AtomicU64,
    /// Progress signals seen.
    progress: AtomicU64,
    /// The verdict. A `watch` so a waiter (an admission wait) can await it.
    dead: watch::Sender<bool>,
    /// Wakes the watcher: on `begin`, and when this state is dropped (the engine is gone).
    wake: Arc<Notify>,
    /// The requests in flight or queued on this engine. Its lock also orders a registration
    /// against the verdict.
    requests: Mutex<Vec<Registered>>,
    /// Set once the watcher task is spawned (lazily, at the first registration).
    watcher_started: AtomicBool,
}

impl Drop for Shared {
    fn drop(&mut self) {
        // `notify_one` stores a permit when nobody waits, so the watcher wakes either way and
        // finds the engine gone.
        self.wake.notify_one();
    }
}

/// See [`EngineLiveness::idle_on_drop`].
pub(crate) struct IdleOnDrop(EngineLiveness);

impl Drop for IdleOnDrop {
    fn drop(&mut self) {
        self.0.end();
    }
}

/// One engine instance's liveness (0.9.1 S1). Created with the engine handle at load and cloned
/// into its engine thread; an unload + load builds a new one, clear.
#[derive(Clone, Debug)]
pub struct EngineLiveness(Arc<Shared>);

impl Default for EngineLiveness {
    fn default() -> Self {
        Self(Arc::new(Shared {
            busy: AtomicBool::new(false),
            calls: AtomicU64::new(0),
            progress: AtomicU64::new(0),
            dead: watch::Sender::new(false),
            wake: Arc::new(Notify::new()),
            requests: Mutex::new(Vec::new()),
            watcher_started: AtomicBool::new(false),
        }))
    }
}

/// The counters at one instant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Snapshot {
    busy: bool,
    calls: u64,
    progress: u64,
}

impl EngineLiveness {
    /// A live, idle engine.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // ---- engine-thread side (no lock, no allocation) ------------------------------------------

    /// A call starts (VLM / NPU generate). Call it BEFORE the `OpenVINO` call, so a call that
    /// hangs at once is still seen.
    pub(crate) fn begin(&self) {
        self.0.calls.fetch_add(1, Ordering::Relaxed);
        self.0.busy.store(true, Ordering::Release);
        self.0.wake.notify_one();
    }

    /// A busy period starts, if the engine was idle (CB: every `AddRequest` and every step calls
    /// it; only the first after an idle spell counts).
    #[inline]
    pub(crate) fn begin_if_idle(&self) {
        if !self.0.busy.load(Ordering::Relaxed) {
            self.begin();
        }
    }

    /// The engine moved (CB: a step returned; VLM / NPU: a streamer callback).
    #[inline]
    pub(crate) fn progress(&self) {
        self.0.progress.fetch_add(1, Ordering::Relaxed);
    }

    /// The engine is idle (its call returned / its batch is empty).
    #[inline]
    pub(crate) fn end(&self) {
        self.0.busy.store(false, Ordering::Release);
    }

    /// A guard that marks the engine idle when dropped. An engine thread takes one first, so it
    /// is never left `busy` when it exits — by shutdown, a broken loop or a panic — which would
    /// otherwise read as a stuck call to the watcher while a handle still holds the state.
    #[must_use]
    pub(crate) fn idle_on_drop(&self) -> IdleOnDrop {
        IdleOnDrop(self.clone())
    }

    /// Mark the engine dead now, as the watcher's verdict would (tests elsewhere in the crate).
    #[cfg(test)]
    pub(crate) fn declare_dead_for_test(&self) {
        self.declare_stalled("test", StallDeadline::FirstToken, Duration::ZERO, None);
    }

    /// Progress signals so far, and whether the engine is busy (tests of the engine threads).
    #[cfg(test)]
    pub(crate) fn signals_for_test(&self) -> (u64, bool) {
        let s = self.snapshot();
        (s.progress, s.busy)
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            busy: self.0.busy.load(Ordering::Acquire),
            calls: self.0.calls.load(Ordering::Relaxed),
            progress: self.0.progress.load(Ordering::Relaxed),
        }
    }

    // ---- request side --------------------------------------------------------------------------

    /// True once the engine stalled: its thread is presumed stuck inside a call that will not
    /// return, and the model must not be sent more work.
    #[must_use]
    pub fn is_dead(&self) -> bool {
        *self.0.dead.borrow()
    }

    /// Resolves once the engine is dead (at once if it already is).
    pub(crate) async fn dead(&self) {
        let mut rx = self.0.dead.subscribe();
        // `wait_for` errs only when the sender is gone; `self` holds it, so it cannot be.
        let _ = rx.wait_for(|dead| *dead).await;
    }

    /// Await `fut` (a request's wait on the engine thread: a tokenizer reply, a command send)
    /// unless the engine is declared dead first — then an error carrying the stall marker, so a
    /// caller parked behind a stuck call is released (callers fail open or map the marker).
    ///
    /// # Errors
    /// The stall error, when the verdict comes first.
    pub(crate) async fn unless_dead<F: std::future::Future>(
        &self,
        fut: F,
    ) -> anyhow::Result<F::Output> {
        tokio::select! {
            biased;
            out = fut => Ok(out),
            () = self.dead() => Err(anyhow::anyhow!(stalled_error_text())),
        }
    }

    /// Both handles share one state (the same engine instance).
    #[must_use]
    pub fn same_engine(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Register a request on this engine, before it is submitted: if the engine stalls while the
    /// request is queued or running, its channel gets `Error(generation_stalled…)` + `Done`. On
    /// an engine already dead the pair is sent at once. Starts the engine's watcher on first use,
    /// with the deadlines `mm`'s config sets (both off: nothing to watch, nothing registered).
    pub(crate) fn register(
        &self,
        tx: &TokenSender,
        request_id: &str,
        model_id: &str,
        mm: &Arc<ModelManager>,
    ) {
        let spec = mm.stream_watch_spec();
        if spec.is_off() {
            return;
        }
        self.register_with(tx, request_id);
        self.ensure_watcher(spec, model_id, Some(Arc::downgrade(mm)));
    }

    fn register_with(&self, tx: &TokenSender, request_id: &str) {
        {
            let mut requests = self
                .0
                .requests
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if !self.is_dead() {
                requests.retain(|r| r.tx.upgrade().is_some_and(|t| !t.is_closed()));
                requests.push(Registered {
                    tx: tx.downgrade(),
                    request_id: request_id.to_owned(),
                });
                return;
            }
        }
        send_stalled_pair_now(tx);
    }

    /// Spawn this engine's watcher unless it runs already.
    fn ensure_watcher(&self, spec: WatchSpec, model_id: &str, mm: Option<Weak<ModelManager>>) {
        if self.0.watcher_started.swap(true, Ordering::AcqRel) {
            return;
        }
        let shared = Arc::downgrade(&self.0);
        let wake = Arc::clone(&self.0.wake);
        let model_id = model_id.to_owned();
        tokio::spawn(run_watcher(shared, wake, spec, model_id, mm));
    }

    /// The verdict, once: mark dead (under the registry lock, so a registration either is in the
    /// list taken here or sees the flag), log the one ERROR line, act on the model, and end every
    /// registered request. `false` if the engine was already dead.
    fn declare_stalled(
        &self,
        model_id: &str,
        deadline: StallDeadline,
        elapsed: Duration,
        mm: Option<&Arc<ModelManager>>,
    ) -> bool {
        let taken = {
            let mut requests = self
                .0
                .requests
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let changed = self.0.dead.send_if_modified(|dead| {
                if *dead {
                    false
                } else {
                    *dead = true;
                    true
                }
            });
            if !changed {
                return false;
            }
            std::mem::take(&mut *requests)
        };
        let live: Vec<(TokenSender, String)> = taken
            .into_iter()
            .filter_map(|r| {
                r.tx.upgrade()
                    .filter(|t| !t.is_closed())
                    .map(|t| (t, r.request_id))
            })
            .collect();
        let request_ids: Vec<&str> = live.iter().map(|(_, id)| id.as_str()).collect();
        tracing::error!(
            model_id,
            request_ids = %request_ids.join(","),
            deadline = deadline.config_key(),
            elapsed_secs = elapsed.as_secs(),
            "generation stalled — the model's engine made no progress within the deadline; it is \
             marked dead and its requests end with generation_stalled"
        );
        if let Some(mm) = mm {
            mm.on_generation_stalled(model_id, self, deadline, elapsed);
        }
        for (tx, _) in live {
            // `send().await`, not `try_send`: the consumer may be behind. Its receiver closing
            // ends the wait. Error then Done, the order every engine uses for a failure.
            tokio::spawn(async move {
                if tx
                    .send(StreamEvent::Error(stalled_error_text()))
                    .await
                    .is_ok()
                {
                    let _ = tx.send(StreamEvent::Done(FinishReason::Stop)).await;
                }
            });
        }
        true
    }
}

/// The verdict pair on a channel the caller still owns and nobody has written to yet (an
/// admission wait that ended on the verdict, or a registration after it). `try_send`: the channel
/// has room for two events; a duplicate of the watcher's pair is harmless (consumers stop at the
/// first terminal event).
pub(crate) fn send_stalled_pair_now(tx: &TokenSender) {
    let _ = tx.try_send(StreamEvent::Error(stalled_error_text()));
    let _ = tx.try_send(StreamEvent::Done(FinishReason::Stop));
}

/// What the watcher concluded from one sample.
#[derive(Debug, PartialEq, Eq)]
enum Watch {
    /// No work: wait for the next `begin`.
    Idle,
    /// Work, within its deadline: sample again after a tick.
    Busy,
    /// Work, past a deadline.
    Stalled {
        deadline: StallDeadline,
        elapsed: Duration,
    },
}

/// The watcher's own clock over the engine's counters (pure: the instants are passed in).
#[derive(Debug, Default)]
struct Observer {
    /// The counters at the previous sample.
    seen: Option<Snapshot>,
    /// When the current call / busy period was first seen.
    call_since: Option<Instant>,
    /// When progress in the current call was last seen.
    last_progress: Option<Instant>,
}

impl Observer {
    fn observe(&mut self, snap: Snapshot, now: Instant, spec: WatchSpec) -> Watch {
        let prev = self.seen.replace(snap);
        if !snap.busy {
            self.call_since = None;
            self.last_progress = None;
            return Watch::Idle;
        }
        let progressed = prev.is_some_and(|p| p.progress != snap.progress);
        let same_call = prev.is_some_and(|p| p.calls == snap.calls) && self.call_since.is_some();
        if same_call {
            if progressed {
                self.last_progress = Some(now);
            }
        } else {
            // A new call (or the first sample of this one): its clock starts now, on the
            // first-token deadline. Progress seen in the same sample may belong to the previous
            // call, so it is not counted for this one (the lenient choice: a VLM call queued
            // behind another gets its whole first-token allowance for its prefill).
            self.call_since = Some(now);
            self.last_progress = None;
        }
        let (since, limit, which) = match (self.last_progress, self.call_since) {
            (Some(t), _) => (t, spec.token_gap, StallDeadline::TokenGap),
            (None, Some(t)) => (t, spec.first_token, StallDeadline::FirstToken),
            (None, None) => return Watch::Busy,
        };
        let elapsed = now.saturating_duration_since(since);
        match limit {
            Some(limit) if elapsed >= limit => Watch::Stalled {
                deadline: which,
                elapsed,
            },
            _ => Watch::Busy,
        }
    }
}

/// One engine's watcher: runs until the engine is gone or has been declared dead.
async fn run_watcher(
    shared: Weak<Shared>,
    wake: Arc<Notify>,
    spec: WatchSpec,
    model_id: String,
    mm: Option<Weak<ModelManager>>,
) {
    let tick = spec.tick();
    let mut observer = Observer::default();
    loop {
        let Some(state) = shared.upgrade() else {
            return;
        };
        let liveness = EngineLiveness(state);
        if liveness.is_dead() {
            return;
        }
        match observer.observe(liveness.snapshot(), Instant::now(), spec) {
            Watch::Idle => {
                // A `begin` between the sample and this wait stored a permit: no lost wakeup.
                drop(liveness);
                wake.notified().await;
            }
            Watch::Busy => {
                drop(liveness);
                tokio::time::sleep(tick).await;
            }
            Watch::Stalled { deadline, elapsed } => {
                let mm = mm.as_ref().and_then(Weak::upgrade);
                liveness.declare_stalled(&model_id, deadline, elapsed, mm.as_ref());
                return;
            }
        }
    }
}

/// Test-only engine behaviours for the mock engines (CB, VLM, NPU), chosen by a marker in the
/// prompt so every other mock request behaves as before. Each reports to the engine's liveness
/// like a real engine thread.
#[cfg(test)]
pub(crate) mod mock {
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    use super::EngineLiveness;
    use crate::ov_cb::FinishReason;
    use crate::streaming::{StreamEvent, TokenSender};

    /// A call that makes no progress; it returns late, once nobody reads the channel.
    pub(crate) const STALL_FIRST: &str = "__mock_stall_first__";
    /// Three tokens, then the same late-returning hang.
    pub(crate) const STALL_AFTER_3: &str = "__mock_stall_after_3__";
    /// Two tokens and `Done`, at once: a healthy answer (reported to no liveness).
    pub(crate) const OK: &str = "__mock_ok__";
    /// Prefix of a call that STAYS stuck — a closed channel does not end it — until the test
    /// calls [`release`] with the key that follows the prefix (`__mock_stuck_<key>__`).
    pub(crate) const STUCK_PREFIX: &str = "__mock_stuck_";

    /// A REAL-time bound on one test case on paused tokio time: there a `tokio::time::timeout`
    /// is fast-forwarded while a mock engine's OS thread still works, so a case that hangs (its
    /// bug back) aborts the test process after 60 s instead of hanging the run.
    pub(crate) struct RealTimeBound(Option<std::sync::mpsc::Sender<()>>);

    impl RealTimeBound {
        pub(crate) fn start(what: &'static str) -> Self {
            let (tx, rx) = std::sync::mpsc::channel::<()>();
            std::thread::spawn(move || {
                if let Err(std::sync::mpsc::RecvTimeoutError::Timeout) =
                    rx.recv_timeout(Duration::from_secs(60))
                {
                    eprintln!("test case hung for 60 s: {what}");
                    std::process::abort();
                }
            });
            Self(Some(tx))
        }
    }

    impl Drop for RealTimeBound {
        fn drop(&mut self) {
            if let Some(tx) = self.0.take() {
                let _ = tx.send(());
            }
        }
    }

    /// Keys of the stuck calls the tests have released.
    static RELEASED: Mutex<Vec<String>> = Mutex::new(Vec::new());

    /// The marker for a stuck call with `key`.
    pub(crate) fn stuck(key: &str) -> String {
        format!("{STUCK_PREFIX}{key}__")
    }

    /// Let the stuck call with `key` return.
    pub(crate) fn release(key: &str) {
        RELEASED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(key.to_owned());
    }

    fn released(key: &str) -> bool {
        RELEASED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .any(|k| k == key)
    }

    fn stuck_key(text: &str) -> Option<String> {
        let rest = &text[text.find(STUCK_PREFIX)? + STUCK_PREFIX.len()..];
        Some(rest[..rest.find("__")?].to_owned())
    }

    /// Run the marked behaviour on the calling (mock engine) thread; `false` = no marker, the
    /// mock answers as usual. Every hang is capped at 120 s so a failing test cannot leak the
    /// thread forever.
    pub(crate) fn run(text: &str, token_tx: &TokenSender, liveness: &EngineLiveness) -> bool {
        enum Hang {
            No,
            UntilClosed,
            UntilReleased(String),
        }
        let (tokens, hang) = if let Some(key) = stuck_key(text) {
            (0, Hang::UntilReleased(key))
        } else if text.contains(STALL_FIRST) {
            (0, Hang::UntilClosed)
        } else if text.contains(STALL_AFTER_3) {
            (3, Hang::UntilClosed)
        } else if text.contains(OK) {
            (2, Hang::No)
        } else {
            return false;
        };
        // A healthy answer is over before any clock could matter: it reports nothing, so a test
        // on paused time cannot see it "stall" while this OS thread is between two calls.
        if matches!(hang, Hang::No) {
            for i in 0..tokens {
                let _ = token_tx.try_send(StreamEvent::Token(format!("tok{i} "), 1));
            }
            let _ = token_tx.try_send(StreamEvent::Done(FinishReason::Stop));
            return true;
        }
        liveness.begin();
        for i in 0..tokens {
            let _ = token_tx.try_send(StreamEvent::Token(format!("tok{i} "), 1));
            liveness.progress();
        }
        let until = Instant::now() + Duration::from_secs(120);
        let wait = |done: &dyn Fn() -> bool| {
            while !done() && Instant::now() < until {
                std::thread::sleep(Duration::from_millis(5));
            }
        };
        match hang {
            Hang::No => {}
            Hang::UntilClosed => {
                wait(&|| token_tx.is_closed());
                liveness.end();
            }
            Hang::UntilReleased(key) => {
                wait(&|| released(&key));
                liveness.end();
            }
        }
        true
    }
}

// ============================================================
// Unit tests — on paused tokio time, the engine side driven from the test task
// ============================================================

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::streaming::stream_channel;

    fn secs(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    fn spec(first: Option<Duration>, gap: Option<Duration>) -> WatchSpec {
        WatchSpec {
            first_token: first,
            token_gap: gap,
        }
    }

    /// 600 s / 120 s, the defaults.
    fn defaults() -> WatchSpec {
        spec(Some(secs(600)), Some(secs(120)))
    }

    /// Start the watcher with no model manager (the flag and the requests only).
    fn watched(spec: WatchSpec) -> EngineLiveness {
        let l = EngineLiveness::new();
        l.ensure_watcher(spec, "m", None);
        l
    }

    /// Let the watcher task run after a change made from the test task.
    async fn settle() {
        for _ in 0..4 {
            tokio::task::yield_now().await;
        }
    }

    /// Advance paused time in small steps, so the watcher samples on its tick as it would live.
    async fn run_for(d: Duration) {
        let step = Duration::from_millis(500);
        let mut left = d;
        while left > Duration::ZERO {
            let s = step.min(left);
            tokio::time::advance(s).await;
            settle().await;
            left -= s;
        }
    }

    /// An idle engine is never stalled, however long it stays idle; nor is one whose calls end.
    #[tokio::test(start_paused = true)]
    async fn an_idle_engine_is_never_stalled() {
        let l = watched(defaults());
        run_for(secs(3600)).await;
        assert!(!l.is_dead());
        l.begin();
        l.progress();
        l.end();
        run_for(secs(3600)).await;
        assert!(!l.is_dead(), "a finished call leaves the engine idle");
    }

    /// The first-token deadline: a call with no progress stalls at 600 s (not before, and at
    /// most one tick after), and every registered request gets the verdict pair.
    #[tokio::test(start_paused = true)]
    async fn a_call_without_progress_stalls_at_the_first_token_deadline() {
        let l = watched(defaults());
        let (tx, mut rx) = stream_channel();
        l.register_with(&tx, "r1");
        l.begin();
        settle().await;
        run_for(secs(599)).await;
        assert!(!l.is_dead(), "not before the deadline");
        run_for(secs(7)).await;
        assert!(l.is_dead(), "at the deadline, within one tick (5 s)");
        settle().await;
        assert!(
            matches!(rx.recv().await, Some(StreamEvent::Error(e)) if is_generation_stalled_error(&e))
        );
        assert!(matches!(rx.recv().await, Some(StreamEvent::Done(_))));
    }

    /// The gap deadline: steady progress keeps a long call alive; silence after it stalls at the
    /// gap (120 s), not the first-token deadline (600 s).
    #[tokio::test(start_paused = true)]
    async fn steady_progress_keeps_a_long_call_alive_and_silence_trips_the_gap() {
        let l = watched(defaults());
        l.begin();
        settle().await;
        for _ in 0..60 {
            run_for(secs(60)).await;
            l.progress();
        }
        assert!(!l.is_dead(), "an hour of progress every 60 s");
        run_for(secs(119)).await;
        assert!(!l.is_dead());
        run_for(secs(7)).await;
        assert!(l.is_dead(), "120 s after the last progress");
    }

    /// An engine thread that exits while busy (shutdown mid-batch, a panic) leaves the engine
    /// idle through its `idle_on_drop` guard: no verdict later, although a handle keeps the state.
    #[tokio::test(start_paused = true)]
    async fn a_thread_that_exits_while_busy_is_not_stalled() {
        let l = watched(defaults());
        let (tx, mut rx) = stream_channel();
        l.register_with(&tx, "r1");
        let engine = l.clone();
        let joined = std::thread::spawn(move || {
            let _idle = engine.idle_on_drop();
            engine.begin();
            panic!("engine thread dies mid-call");
        })
        .join();
        assert!(joined.is_err(), "the thread panicked");
        settle().await;
        run_for(secs(3600)).await;
        assert!(!l.is_dead(), "an exited thread is not a stuck call");
        assert!(rx.try_recv().is_err(), "no verdict pair");
    }

    /// The verdict does not depend on any client: with no request registered (every client
    /// left), a stuck engine is still declared dead, and a request registered after that gets
    /// the verdict at once (review F1).
    #[tokio::test(start_paused = true)]
    async fn the_verdict_needs_no_client_and_reaches_a_later_request() {
        let l = watched(defaults());
        let (tx, rx) = stream_channel();
        l.register_with(&tx, "gone");
        drop(rx); // the client left
        l.begin();
        settle().await;
        run_for(secs(610)).await;
        assert!(l.is_dead());
        let (tx2, mut rx2) = stream_channel();
        l.register_with(&tx2, "retry");
        assert!(
            matches!(rx2.recv().await, Some(StreamEvent::Error(e)) if is_generation_stalled_error(&e))
        );
    }

    /// A CB batch whose steps keep completing is alive even when no sequence gets visible text
    /// for longer than the gap (chunked prefill, a preempted or starved sequence): step
    /// completions are the progress signal (review F2).
    #[tokio::test(start_paused = true)]
    async fn steps_without_text_are_progress() {
        let l = watched(defaults());
        l.begin_if_idle();
        settle().await;
        for _ in 0..(20 * 60) {
            run_for(Duration::from_millis(500)).await;
            l.begin_if_idle(); // every step start
            l.progress(); // every step end — no token anywhere
        }
        assert!(!l.is_dead(), "10 minutes of steps");
    }

    /// A slow reader is not a stalled engine: nobody reading the channel changes nothing while
    /// the engine progresses.
    #[tokio::test(start_paused = true)]
    async fn a_slow_client_is_not_a_stalled_engine() {
        let l = watched(defaults());
        let (tx, _rx_never_read) = stream_channel();
        l.register_with(&tx, "slow");
        l.begin();
        settle().await;
        for i in 0..20 {
            run_for(secs(100)).await;
            let _ = tx.try_send(StreamEvent::Token(format!("t{i}"), 1));
            l.progress();
        }
        assert!(!l.is_dead());
    }

    /// `0` turns each deadline off on its own.
    #[tokio::test(start_paused = true)]
    async fn zero_disables_each_deadline() {
        let no_first = watched(spec(None, Some(secs(120))));
        no_first.begin();
        settle().await;
        run_for(secs(3600)).await;
        assert!(
            !no_first.is_dead(),
            "first-token off: no progress yet is fine"
        );
        no_first.progress();
        settle().await;
        run_for(secs(130)).await;
        assert!(no_first.is_dead(), "the gap still applies");

        let no_gap = watched(spec(Some(secs(600)), None));
        no_gap.begin();
        settle().await;
        run_for(secs(10)).await;
        no_gap.progress();
        settle().await;
        run_for(secs(3600)).await;
        assert!(!no_gap.is_dead(), "gap off");
    }

    /// A new call starts a new first-token clock: a VLM request queued behind a long one gets
    /// its own 600 s once it starts, not the gap from the previous call's last token.
    #[tokio::test(start_paused = true)]
    async fn each_call_gets_its_own_first_token_clock() {
        let l = watched(defaults());
        l.begin();
        settle().await;
        run_for(secs(10)).await;
        l.progress();
        l.end();
        l.begin(); // the next queued call, at once
        settle().await;
        run_for(secs(300)).await;
        assert!(
            !l.is_dead(),
            "300 s of prefill in the second call is within 600 s"
        );
    }

    /// The verdict is taken once; a second declaration changes nothing.
    #[tokio::test(start_paused = true)]
    async fn the_verdict_counts_once() {
        let l = EngineLiveness::new();
        assert!(l.declare_stalled("m", StallDeadline::TokenGap, secs(1), None));
        assert!(!l.declare_stalled("m", StallDeadline::TokenGap, secs(1), None));
        assert!(l.is_dead());
    }

    /// The watcher ends with its engine.
    #[tokio::test(start_paused = true)]
    async fn the_watcher_ends_with_its_engine() {
        let l = EngineLiveness::new();
        let weak = Arc::downgrade(&l.0);
        l.ensure_watcher(defaults(), "m", None);
        settle().await;
        drop(l);
        settle().await;
        assert!(weak.upgrade().is_none());
    }

    /// The tick is a twentieth of the shorter deadline, within its bounds.
    #[test]
    fn tick_follows_the_shorter_deadline() {
        assert_eq!(defaults().tick(), MAX_TICK);
        assert_eq!(spec(Some(secs(20)), None).tick(), secs(1));
        assert_eq!(spec(Some(Duration::from_millis(1)), None).tick(), MIN_TICK);
    }

    /// The config maps `0` to off and anything else to that many seconds.
    #[test]
    fn spec_from_config() {
        let mut cfg: Config =
            serde_json::from_str(r#"{"models_dir":"/tmp/m","device":"CPU","total_vram_gb":16.0}"#)
                .unwrap();
        assert_eq!(WatchSpec::from_config(&cfg), defaults());
        cfg.stream_first_token_timeout_secs = 0;
        cfg.stream_token_gap_timeout_secs = 0;
        assert!(WatchSpec::from_config(&cfg).is_off());
    }

    /// The marker cannot be mistaken for the other classified engine errors, and the text names
    /// no config key (it reaches realtime clients).
    #[test]
    fn the_marker_is_distinct() {
        let msg = stalled_error_text();
        assert!(is_generation_stalled_error(&msg));
        assert!(!msg.contains("_secs"));
        assert!(!crate::model_manager::is_gpu_poison_error(&msg));
        assert!(!msg.contains(crate::handlers::error::ENGINE_SHUTDOWN_MARKER));
        assert!(!is_generation_stalled_error(
            "inference failed: generation_stalled:"
        ));
    }
}
