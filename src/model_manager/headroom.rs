// ============================================================
// src/model_manager/headroom.rs — runtime side of the device-headroom limit
// ============================================================
// `kv_headroom.rs` is the arithmetic. This module holds what the manager needs around it: the
// facts of one vision model (`HeadroomSpec`, `HeadroomRecord`), the physical probe with a short
// cache (a request must not scan /proc each time), and the transient leases of long requests.
// ============================================================

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::kv_headroom::{HeadroomInputs, transient_bytes_per_token};
use crate::gpu_memory::VramProbe;

/// How long a probe reading is reused. Other processes' memory moves slowly relative to a request;
/// the cache keeps a burst of requests from scanning `/proc` each.
const PROBE_TTL: Duration = Duration::from_secs(2);

/// The oldest outside-memory reading a request-side decision (the vision device-headroom cap) may
/// use. An older one is refreshed first, synchronously, with one probe read on the blocking pool
/// ([`HeadroomRuntime::refresh_if_older_than`]); within this age the cached reading and its
/// background refresh serve as before. E3 (R-01): before this bound the cached reading served at
/// any age, and the first vision request after an idle gap was capped on a reading up to hours
/// old. A hard limit on purpose, not a config key.
pub(crate) const REQUEST_READING_MAX_AGE: Duration = Duration::from_secs(30);

/// The longest a request waits for that synchronous refresh (its turn on the per-device lock and
/// the scan together). A scan takes about 0.1-0.2 s; past this bound the request logs one WARN and
/// decides on the cached reading, and the scan finishes (or hangs) on its blocking thread without
/// a second one being started for the device. Same E3 rule; a hard limit too.
pub(crate) const REQUEST_REFRESH_TIMEOUT: Duration = Duration::from_secs(2);

/// What the limit of one vision model on a discrete GPU is computed from.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HeadroomSpec {
    /// The model's hidden size.
    pub hidden_size: usize,
    /// PCI address of the device (the probe's key).
    pub pdev: String,
    /// Total device memory the server may use, in bytes.
    pub total_bytes: u64,
    /// `kv_transient_factor`.
    pub factor: f64,
    /// `kv_headroom_margin_gib` in bytes.
    pub margin_bytes: u64,
    /// The smallest limit ever reported (`kv_headroom_min_tokens`, capped at the native context).
    pub floor_tokens: usize,
}

impl HeadroomSpec {
    /// The inputs of the limit for the given measurements.
    pub(crate) fn inputs(&self, outside: u64, own_booked: u64, leases: u64) -> HeadroomInputs {
        HeadroomInputs {
            total_bytes: self.total_bytes,
            outside_bytes: outside,
            own_booked_bytes: own_booked,
            lease_bytes: leases,
            margin_bytes: self.margin_bytes,
            hidden_size: self.hidden_size,
            factor: self.factor,
        }
    }

    /// The limit with the never-0 floor applied.
    pub(crate) fn limit_tokens(&self, outside: u64, own_booked: u64, leases: u64) -> usize {
        self.inputs(outside, own_booked, leases)
            .cap_tokens()
            .max(self.floor_tokens)
    }
}

/// A Ready vision model's headroom facts, kept on its record.
#[derive(Clone, Debug)]
pub(crate) struct HeadroomRecord {
    /// The model this record belongs to (its own leases do not lower its own limit).
    pub model_id: String,
    /// What the limit is computed from.
    pub spec: HeadroomSpec,
    /// The memory domain the model is charged to.
    pub domain: String,
    /// The limit the pool, native context and ratchet allow, before the headroom.
    pub pool_limit_tokens: usize,
    /// What limits the pool-side limit (reported when the device headroom is not the tighter one).
    pub pool_clamp: super::GateClamp,
}

/// One discrete GPU in the headroom report (`GET /v1/admin/models` `server.headroom`).
#[derive(Clone, Debug, serde::Serialize)]
pub struct HeadroomDevice {
    /// The device name (`GPU.1`).
    pub device: String,
    /// Total device memory the server may use, GiB.
    pub total_gib: f64,
    /// Held by other processes (physical probe), MiB; absent when the probe cannot read it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outside_mib: Option<u64>,
    /// This process's commitments: the larger of the tracker's bookings and the driver's figure,
    /// GiB.
    pub own_gib: f64,
    /// Transient reservations of running long requests, MiB (withheld from callers without a
    /// key: it says whether a long request is running right now).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub leases_mib: Option<u64>,
    /// Left for one prompt's transient write after everything above and the margin, GiB.
    pub free_gib: f64,
}

/// The device-headroom state for the admin API and `rv status`.
#[derive(Clone, Debug, serde::Serialize)]
pub struct HeadroomReport {
    /// `kv_headroom_mode` (`dynamic` or `off`).
    pub mode: String,
    /// `fdinfo` when the physical probe exists, `none` otherwise.
    pub probe: &'static str,
    /// The safety margin kept free, GiB.
    pub margin_gib: f64,
    /// One entry per discrete GPU with a PCI address.
    pub devices: Vec<HeadroomDevice>,
}

/// The probe, its cache, and the leases.
#[derive(Clone, Debug)]
pub(crate) struct HeadroomRuntime {
    probe: Option<Arc<dyn VramProbe>>,
    /// Last reading per `pdev`; shared with the background refresh threads.
    cache: Arc<Mutex<HashMap<String, (Instant, u64)>>>,
    /// Last reading of this process's own device memory per `pdev` (a few `/proc/self` reads).
    own_cache: Arc<Mutex<HashMap<String, (Instant, u64)>>>,
    /// `pdev`s with a background refresh in flight (at most one each).
    refreshing: Arc<Mutex<HashSet<String>>>,
    /// `pdev`s whose probe failure was already logged (F2-N-5): set when a reading returns
    /// nothing, cleared by the next good one. The age of a reading alone never sets it: an idle
    /// server is not a failing probe (S2).
    stale_warned: Arc<Mutex<HashSet<String>>>,
    ttl: Duration,
    /// One async lock per `pdev` for the request-side synchronous refresh, so concurrent requests
    /// after an idle gap run one probe per device: the others wait for it and use its reading.
    request_refresh: Arc<Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
    /// When the last request-side synchronous scan of each `pdev` started. Within
    /// [`REQUEST_READING_MAX_AGE`] of it no other one starts, so a probe that returns nothing
    /// (whose reading therefore stays old) costs one scan per 30 s, not one per request.
    request_attempts: Arc<Mutex<HashMap<String, Instant>>>,
    /// `pdev`s with a request-side scan running on the blocking pool. Cleared by the scan's own
    /// closure (also on a panic), so a waiter that timed out or was cancelled never leads to a
    /// second scan of a device whose first one is still running.
    request_scanning: Arc<Mutex<HashSet<String>>>,
    /// [`REQUEST_REFRESH_TIMEOUT`]; tests shorten it.
    request_timeout: Duration,
    /// Transient reservations of running long requests: domain -> model -> bytes.
    leases: Arc<Mutex<HashMap<String, HashMap<String, u64>>>>,
}

/// Store one probe reading of `pdev` and return what to use: the reading, or the last known value
/// when the probe returned nothing. A probe that returns nothing is logged once per device until a
/// reading succeeds again (F2-N-5): that, and not the age of a reading, is a failing probe.
fn record_outside(
    cache: &Mutex<HashMap<String, (Instant, u64)>>,
    warned: &Mutex<HashSet<String>>,
    pdev: &str,
    fresh: Option<u64>,
) -> Option<u64> {
    let mut warned = warned
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut cache = cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(bytes) = fresh {
        cache.insert(pdev.to_owned(), (Instant::now(), bytes));
        warned.remove(pdev);
        return Some(bytes);
    }
    let last = cache.get(pdev).copied();
    if warned.insert(pdev.to_owned()) {
        tracing::warn!(
            pdev,
            last_reading_age_secs = last.map(|(at, _)| at.elapsed().as_secs()),
            "device-memory probe returned no reading: using the last one; memory other processes \
             took since then is covered only by kv_headroom_margin_gib"
        );
    }
    last.map(|(_, b)| b)
}

impl HeadroomRuntime {
    /// A runtime over `probe` (`None` where the device memory cannot be read).
    pub(crate) fn new(probe: Option<Arc<dyn VramProbe>>) -> Self {
        Self {
            probe,
            cache: Arc::new(Mutex::new(HashMap::new())),
            own_cache: Arc::new(Mutex::new(HashMap::new())),
            refreshing: Arc::new(Mutex::new(HashSet::new())),
            stale_warned: Arc::new(Mutex::new(HashSet::new())),
            ttl: PROBE_TTL,
            request_refresh: Arc::new(Mutex::new(HashMap::new())),
            request_attempts: Arc::new(Mutex::new(HashMap::new())),
            request_scanning: Arc::new(Mutex::new(HashSet::new())),
            request_timeout: REQUEST_REFRESH_TIMEOUT,
            leases: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Tests: a zero TTL makes every reading fresh and synchronous.
    #[cfg(test)]
    pub(crate) fn set_ttl(&mut self, ttl: Duration) {
        self.ttl = ttl;
    }

    /// Tests: a shorter bound on the request-side synchronous refresh.
    #[cfg(test)]
    pub(crate) fn set_request_timeout(&mut self, timeout: Duration) {
        self.request_timeout = timeout;
    }

    /// Tests: make every cached outside-memory reading, and every request-side scan attempt, `by`
    /// older (an idle gap).
    #[cfg(test)]
    pub(crate) fn age_readings(&self, by: Duration) {
        let older = |at: &mut Instant| {
            if let Some(o) = at.checked_sub(by) {
                *at = o;
            }
        };
        for (at, _) in self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values_mut()
        {
            older(at);
        }
        for at in self
            .request_attempts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values_mut()
        {
            older(at);
        }
    }

    /// Whether a probe exists at all (`rv status` reports `headroom: unknown` without one).
    pub(crate) fn has_probe(&self) -> bool {
        self.probe.is_some()
    }

    /// Bytes other processes hold on `pdev`. The first reading is taken synchronously; after that
    /// a reading older than the TTL is served as it is while one background thread refreshes it,
    /// so a request never waits for a `/proc` scan (about 0.1-0.2 s on a desktop). When the probe
    /// fails the last known value stays; `None` only if it never worked.
    ///
    /// The served value can be as old as the last call: after an idle gap it is the reading of the
    /// previous request or load. Paths that decide a load call [`Self::outside_bytes_fresh`] on
    /// the blocking pool first (`ModelManager::refresh_device_probes`), so they find a fresh one;
    /// request paths call [`Self::refresh_if_older_than`] with [`REQUEST_READING_MAX_AGE`] first
    /// (`ModelManager::get_chat_context_for_request`), so they decide on one at most 30 s old —
    /// unless the probe returned nothing (the last good reading stays, one retry per 30 s) or the
    /// scan took longer than [`REQUEST_REFRESH_TIMEOUT`] (the cached reading, logged).
    pub(crate) fn outside_bytes(&self, pdev: &str) -> Option<u64> {
        let probe = self.probe.as_ref()?;
        let cached = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(pdev)
            .copied();
        match cached {
            Some((at, bytes)) if at.elapsed() < self.ttl => Some(bytes),
            Some((_, bytes)) if !self.ttl.is_zero() => {
                self.refresh_in_background(Arc::clone(probe), pdev);
                Some(bytes)
            }
            // First reading (or a zero TTL): read now.
            _ => self.read_outside_now(probe.as_ref(), pdev),
        }
    }

    /// [`Self::outside_bytes`] read from `/proc` now, ignoring the TTL; the last known value when
    /// the probe returns nothing, `None` if it never worked. Blocking (a scan of every readable
    /// process): call it from `spawn_blocking`.
    pub(crate) fn outside_bytes_fresh(&self, pdev: &str) -> Option<u64> {
        let probe = self.probe.as_ref()?;
        self.read_outside_now(probe.as_ref(), pdev)
    }

    /// Whether the reading of `pdev` is missing or older than the TTL, so a caller about to decide
    /// a load should take a fresh one first. `false` without a probe (there is nothing to read).
    pub(crate) fn needs_refresh(&self, pdev: &str) -> bool {
        self.probe.is_some()
            && self
                .cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(pdev)
                .is_none_or(|(at, _)| at.elapsed() >= self.ttl)
    }

    /// Whether the reading of `pdev` is missing or at least `max_age` old. `false` without a probe.
    fn reading_older_than(&self, pdev: &str, max_age: Duration) -> bool {
        self.probe.is_some()
            && self
                .cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(pdev)
                .is_none_or(|(at, _)| reached(at.elapsed(), max_age))
    }

    /// Whether a request-side scan of `pdev` started less than `max_age` ago (so a probe that
    /// returned nothing is not retried on every request).
    fn attempted_within(&self, pdev: &str, max_age: Duration) -> bool {
        self.request_attempts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(pdev)
            .is_some_and(|at| !reached(at.elapsed(), max_age))
    }

    /// Request side (E3): when the reading of `pdev` is missing or at least `max_age` old, take a
    /// fresh one before returning — one probe read on the blocking pool (`spawn_blocking`), never
    /// on the async runtime. Concurrent callers for the same device wait on one per-device lock
    /// and, finding the reading fresh once they hold it, do not probe again. Within `max_age`
    /// nothing happens (the synchronous readers serve the cache and refresh it in the
    /// background). A failed probe keeps the last reading, as [`Self::outside_bytes_fresh`] does,
    /// and is not retried here for `max_age`. The wait is bounded by [`REQUEST_REFRESH_TIMEOUT`]:
    /// past it one WARN, and the caller decides on the cached reading.
    pub(crate) async fn refresh_if_older_than(&self, pdev: &str, max_age: Duration) {
        if !self.reading_older_than(pdev, max_age) {
            return;
        }
        let started = Instant::now();
        if tokio::time::timeout(
            self.request_timeout,
            self.refresh_single_flight(pdev, max_age),
        )
        .await
        .is_err()
        {
            tracing::warn!(
                pdev,
                waited_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
                "device-memory probe did not answer in time: this request decides on the cached \
                 reading"
            );
        }
    }

    /// The part of [`Self::refresh_if_older_than`] under its time bound: the per-device lock (a
    /// waiter behind a live scan waits for it and uses its reading), then at most one scan of
    /// `pdev` running at a time; none when one started within `max_age`.
    async fn refresh_single_flight(&self, pdev: &str, max_age: Duration) {
        let lock = Arc::clone(
            self.request_refresh
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .entry(pdev.to_owned())
                .or_default(),
        );
        let _held = lock.lock().await;
        // Another request may have refreshed it while this one waited (the common case after an
        // idle gap), or tried within `max_age` and got nothing back (a failing probe).
        if !self.reading_older_than(pdev, max_age) || self.attempted_within(pdev, max_age) {
            return;
        }
        if !self
            .request_scanning
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(pdev.to_owned())
        {
            // A scan whose waiter gave up is still running: use the cache, start no second one.
            return;
        }
        self.request_attempts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(pdev.to_owned(), Instant::now());
        let flag = ScanFlag {
            set: Arc::clone(&self.request_scanning),
            pdev: pdev.to_owned(),
        };
        let rt = self.clone();
        if let Err(e) = tokio::task::spawn_blocking(move || {
            rt.outside_bytes_fresh(&flag.pdev);
            drop(flag);
        })
        .await
        {
            // The cached reading stays; the request decides on it, as before E3.
            tracing::debug!(pdev, error = %e, "request-side device-memory probe did not finish");
        }
    }

    /// One probe reading of `pdev`, stored with [`record_outside`]; no lock is held during the scan.
    fn read_outside_now(&self, probe: &dyn VramProbe, pdev: &str) -> Option<u64> {
        let fresh = probe.outside_used_bytes(pdev);
        record_outside(&self.cache, &self.stale_warned, pdev, fresh)
    }

    /// What this process holds on `pdev` according to the driver, cached for the TTL; `None`
    /// when unknown. The headroom limit counts the larger of this and the tracker's bookings, so
    /// memory the tracker never booked (a model charged 0, `OpenCL` scratch) still lowers the limit.
    pub(crate) fn own_bytes(&self, pdev: &str) -> Option<u64> {
        let probe = self.probe.as_ref()?;
        let cached = self
            .own_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(pdev)
            .copied();
        if let Some((at, bytes)) = cached
            && at.elapsed() < self.ttl
        {
            return Some(bytes);
        }
        self.probe_own_bytes(probe.as_ref(), pdev)
    }

    /// [`Self::own_bytes`] read from the driver now, ignoring the TTL. Load admission uses this:
    /// a figure cached while a just-unloaded model was still resident would count that model's
    /// memory as this process's own unbooked memory and refuse the next load ("0.00 GiB free").
    pub(crate) fn own_bytes_fresh(&self, pdev: &str) -> Option<u64> {
        let probe = self.probe.as_ref()?;
        self.probe_own_bytes(probe.as_ref(), pdev)
    }

    /// One driver reading of this process's memory on `pdev`, stored in the cache; the last cached
    /// figure when the driver gives none.
    fn probe_own_bytes(&self, probe: &dyn VramProbe, pdev: &str) -> Option<u64> {
        let fresh = probe.own_used_bytes(pdev);
        let mut cache = self
            .own_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match fresh {
            Some(bytes) => {
                cache.insert(pdev.to_owned(), (Instant::now(), bytes));
                Some(bytes)
            }
            None => cache.get(pdev).map(|(_, b)| *b),
        }
    }

    fn refresh_in_background(&self, probe: Arc<dyn VramProbe>, pdev: &str) {
        {
            let mut busy = self
                .refreshing
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if !busy.insert(pdev.to_owned()) {
                return;
            }
        }
        let cache = Arc::clone(&self.cache);
        let refreshing = Arc::clone(&self.refreshing);
        let stale_warned = Arc::clone(&self.stale_warned);
        let pdev = pdev.to_owned();
        let spawned = std::thread::Builder::new()
            .name("vram-probe".to_owned())
            .spawn({
                let pdev = pdev.clone();
                move || {
                    let fresh = probe.outside_used_bytes(&pdev);
                    record_outside(&cache, &stale_warned, &pdev, fresh);
                    refreshing
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .remove(&pdev);
                }
            });
        if spawned.is_err() {
            self.refreshing
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&pdev);
        }
    }

    /// Device memory currently reserved by running long requests on `domain`, not counting the
    /// ones of `except_model` (a vision engine runs one generation at a time, so a model's own
    /// running request does not compete with the next request on the same model).
    pub(crate) fn lease_bytes(&self, domain: &str, except_model: Option<&str>) -> u64 {
        self.leases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(domain)
            .map_or(0, |m| {
                m.iter()
                    .filter(|(model, _)| Some(model.as_str()) != except_model)
                    .map(|(_, b)| *b)
                    .sum()
            })
    }

    /// Reserve `tokens x hidden x 4 x factor` bytes of `domain` for `model_id` for the life of the
    /// returned guard.
    pub(crate) fn acquire(
        &self,
        domain: &str,
        model_id: &str,
        spec: &HeadroomSpec,
        tokens: usize,
    ) -> TransientLease {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let bytes =
            (transient_bytes_per_token(spec.hidden_size, spec.factor) * tokens as f64) as u64;
        *self
            .leases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(domain.to_owned())
            .or_default()
            .entry(model_id.to_owned())
            .or_default() += bytes;
        TransientLease {
            leases: Arc::clone(&self.leases),
            domain: domain.to_owned(),
            model_id: model_id.to_owned(),
            bytes,
        }
    }
}

/// Whether something `elapsed` old has reached `max_age` (inclusive: exactly 30 s is too old).
fn reached(elapsed: Duration, max_age: Duration) -> bool {
    elapsed >= max_age
}

/// Marks a request-side scan of `pdev` as running; cleared on drop, inside the blocking closure
/// that runs the scan (also when the scan panics).
struct ScanFlag {
    set: Arc<Mutex<HashSet<String>>>,
    pdev: String,
}

impl Drop for ScanFlag {
    fn drop(&mut self) {
        self.set
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.pdev);
    }
}

/// The transient device-memory reservation of one running long request; released on drop.
#[derive(Debug)]
pub struct TransientLease {
    leases: Arc<Mutex<HashMap<String, HashMap<String, u64>>>>,
    domain: String,
    model_id: String,
    bytes: u64,
}

impl Drop for TransientLease {
    fn drop(&mut self) {
        let mut map = self
            .leases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(models) = map.get_mut(&self.domain) {
            if let Some(v) = models.get_mut(&self.model_id) {
                *v = v.saturating_sub(self.bytes);
                if *v == 0 {
                    models.remove(&self.model_id);
                }
            }
            if models.is_empty() {
                map.remove(&self.domain);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[derive(Debug, Default)]
    struct CountingProbe {
        reads: AtomicU64,
        value: AtomicU64,
        fail: std::sync::atomic::AtomicBool,
    }

    impl VramProbe for CountingProbe {
        fn outside_used_bytes(&self, _pdev: &str) -> Option<u64> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            (!self.fail.load(Ordering::Relaxed)).then(|| self.value.load(Ordering::Relaxed))
        }
    }

    fn wait_for_refresh(rt: &HeadroomRuntime, pdev: &str) {
        while rt.refreshing.lock().unwrap().contains(pdev) {
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// E3: the request-side refresh probes only a reading at least `max_age` old (or missing),
    /// on a blocking-pool thread, and stores what it read.
    #[tokio::test]
    async fn the_request_side_refresh_reads_only_a_reading_past_its_age() {
        let probe = Arc::new(CountingProbe::default());
        probe.value.store(9, Ordering::Relaxed);
        let rt = HeadroomRuntime::new(Some(probe.clone() as Arc<dyn VramProbe>));
        let pdev = "0000:03:00.0";
        // Missing: read.
        rt.refresh_if_older_than(pdev, REQUEST_READING_MAX_AGE)
            .await;
        assert_eq!(probe.reads.load(Ordering::Relaxed), 1);
        // Fresh: nothing.
        rt.refresh_if_older_than(pdev, REQUEST_READING_MAX_AGE)
            .await;
        assert_eq!(probe.reads.load(Ordering::Relaxed), 1);
        // 29 s: nothing; 31 s: read, and the new value is stored.
        rt.age_readings(Duration::from_secs(29));
        rt.refresh_if_older_than(pdev, REQUEST_READING_MAX_AGE)
            .await;
        assert_eq!(probe.reads.load(Ordering::Relaxed), 1);
        rt.age_readings(Duration::from_secs(2));
        probe.value.store(11, Ordering::Relaxed);
        rt.refresh_if_older_than(pdev, REQUEST_READING_MAX_AGE)
            .await;
        assert_eq!(probe.reads.load(Ordering::Relaxed), 2);
        assert_eq!(
            rt.cache.lock().unwrap().get(pdev).map(|(_, b)| *b),
            Some(11)
        );
        // No probe: nothing to read, and no panic.
        HeadroomRuntime::new(None)
            .refresh_if_older_than(pdev, REQUEST_READING_MAX_AGE)
            .await;
        assert_eq!(REQUEST_READING_MAX_AGE, Duration::from_secs(30));
    }

    /// F7: the age limit is inclusive — a reading exactly 30 s old is refreshed.
    #[test]
    fn exactly_the_age_limit_is_too_old() {
        assert!(reached(REQUEST_READING_MAX_AGE, REQUEST_READING_MAX_AGE));
        assert!(!reached(
            Duration::from_millis(29_999),
            REQUEST_READING_MAX_AGE
        ));
        assert_eq!(REQUEST_REFRESH_TIMEOUT, Duration::from_secs(2));
    }

    /// A probe whose scan blocks until the test lets it go (a hung `/proc` fdinfo read).
    #[derive(Debug)]
    struct StuckProbe {
        calls: AtomicU64,
        release: Mutex<std::sync::mpsc::Receiver<()>>,
    }

    impl VramProbe for StuckProbe {
        fn outside_used_bytes(&self, _pdev: &str) -> Option<u64> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let _ = self.release.lock().unwrap().recv();
            Some(99)
        }
    }

    /// F1: a scan that hangs does not hold the request past the bound — it returns with the
    /// cached reading — and neither a later request nor one after a cancelled waiter starts a
    /// second scan of the device while the first is stuck. The stuck scan, once released, stores
    /// its reading and clears its flag itself.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_hung_scan_is_bounded_and_never_doubled() {
        let (tx, rx) = std::sync::mpsc::channel();
        let probe = Arc::new(StuckProbe {
            calls: AtomicU64::new(0),
            release: Mutex::new(rx),
        });
        let mut rt = HeadroomRuntime::new(Some(probe.clone() as Arc<dyn VramProbe>));
        rt.set_request_timeout(Duration::from_millis(200));
        let pdev = "0000:03:00.0";
        rt.cache
            .lock()
            .unwrap()
            .insert(pdev.to_owned(), (Instant::now(), 5));
        rt.age_readings(Duration::from_secs(31));

        let t0 = Instant::now();
        rt.refresh_if_older_than(pdev, REQUEST_READING_MAX_AGE)
            .await;
        assert!(t0.elapsed() < Duration::from_secs(1), "{:?}", t0.elapsed());
        assert_eq!(probe.calls.load(Ordering::Relaxed), 1);
        assert_eq!(rt.cache.lock().unwrap().get(pdev).map(|(_, b)| *b), Some(5));
        assert!(rt.request_scanning.lock().unwrap().contains(pdev));

        // A later request (even past the attempt's own 30 s): the scan is still running, so no
        // second one, and it returns at once.
        rt.age_readings(Duration::from_secs(31));
        let t1 = Instant::now();
        rt.refresh_if_older_than(pdev, REQUEST_READING_MAX_AGE)
            .await;
        assert!(
            t1.elapsed() < Duration::from_millis(150),
            "{:?}",
            t1.elapsed()
        );
        // A waiter cancelled mid-wait leaves no second scan behind either.
        let waiter = {
            let rt = rt.clone();
            tokio::spawn(async move {
                rt.refresh_if_older_than(pdev, REQUEST_READING_MAX_AGE)
                    .await;
            })
        };
        waiter.abort();
        let _ = waiter.await;
        rt.refresh_if_older_than(pdev, REQUEST_READING_MAX_AGE)
            .await;
        assert_eq!(
            probe.calls.load(Ordering::Relaxed),
            1,
            "never a second scan"
        );

        // Released: the scan stores its reading and clears its own flag.
        tx.send(()).unwrap();
        let until = Instant::now() + Duration::from_secs(5);
        while rt.request_scanning.lock().unwrap().contains(pdev) {
            assert!(Instant::now() < until, "the scan never cleared its flag");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            rt.cache.lock().unwrap().get(pdev).map(|(_, b)| *b),
            Some(99)
        );
    }

    /// F2: a probe that returns nothing leaves the reading old, but requests retry it once per
    /// 30 s, not each time: five requests, one synchronous attempt.
    #[tokio::test]
    async fn a_failing_probe_is_retried_once_per_age_limit() {
        let probe = Arc::new(CountingProbe::default());
        probe.fail.store(true, Ordering::Relaxed);
        let rt = HeadroomRuntime::new(Some(probe.clone() as Arc<dyn VramProbe>));
        let pdev = "0000:03:00.0";
        rt.cache
            .lock()
            .unwrap()
            .insert(pdev.to_owned(), (Instant::now(), 5));
        rt.age_readings(Duration::from_secs(31));
        for _ in 0..5 {
            rt.refresh_if_older_than(pdev, REQUEST_READING_MAX_AGE)
                .await;
        }
        assert_eq!(probe.reads.load(Ordering::Relaxed), 1);
        assert_eq!(rt.cache.lock().unwrap().get(pdev).map(|(_, b)| *b), Some(5));
        // 30 s later it is tried again.
        rt.age_readings(Duration::from_secs(31));
        rt.refresh_if_older_than(pdev, REQUEST_READING_MAX_AGE)
            .await;
        assert_eq!(probe.reads.load(Ordering::Relaxed), 2);
    }

    /// S2: an old reading from a working probe (an idle server) is not a failure: served while a
    /// thread refreshes it, and nothing is logged.
    #[test]
    fn an_old_reading_from_a_working_probe_is_not_logged() {
        let probe = Arc::new(CountingProbe::default());
        probe.value.store(7, Ordering::Relaxed);
        let rt = HeadroomRuntime::new(Some(probe.clone() as Arc<dyn VramProbe>));
        let pdev = "0000:03:00.0";
        let old = Instant::now()
            .checked_sub(Duration::from_secs(876))
            .unwrap();
        rt.cache.lock().unwrap().insert(pdev.to_owned(), (old, 5));
        assert_eq!(rt.outside_bytes(pdev), Some(5), "served as it is");
        assert!(
            !rt.stale_warned.lock().unwrap().contains(pdev),
            "age alone is not logged"
        );
        wait_for_refresh(&rt, pdev);
        assert!(!rt.stale_warned.lock().unwrap().contains(pdev));
        assert_eq!(rt.outside_bytes(pdev), Some(7), "the refresh landed");
    }

    /// F2-N-5 / S2: a probe that returns nothing is logged once (the last reading is served), and
    /// a good reading re-arms the warning; the fresh read behaves the same.
    #[test]
    fn a_failing_probe_marks_the_device_warned_until_a_reading_succeeds() {
        let probe = Arc::new(CountingProbe::default());
        probe.value.store(7, Ordering::Relaxed);
        let rt = HeadroomRuntime::new(Some(probe.clone() as Arc<dyn VramProbe>));
        let pdev = "0000:03:00.0";
        let old = Instant::now().checked_sub(Duration::from_secs(61)).unwrap();
        rt.cache.lock().unwrap().insert(pdev.to_owned(), (old, 5));
        probe.fail.store(true, Ordering::Relaxed);
        assert_eq!(
            rt.outside_bytes(pdev),
            Some(5),
            "the last reading is served"
        );
        wait_for_refresh(&rt, pdev);
        assert!(
            rt.stale_warned.lock().unwrap().contains(pdev),
            "the failed refresh is logged"
        );
        assert_eq!(
            rt.outside_bytes_fresh(pdev),
            Some(5),
            "fresh read: last known"
        );
        assert!(rt.stale_warned.lock().unwrap().contains(pdev));
        probe.fail.store(false, Ordering::Relaxed);
        assert_eq!(rt.outside_bytes_fresh(pdev), Some(7));
        assert!(
            !rt.stale_warned.lock().unwrap().contains(pdev),
            "a good reading re-arms"
        );
        assert_eq!(rt.cache.lock().unwrap().get(pdev).map(|(_, b)| *b), Some(7));
    }

    /// S2: a load path finds out it must re-read (`needs_refresh`), and the fresh read returns the
    /// value now, not the one cached before the gap.
    #[test]
    fn a_fresh_read_replaces_a_reading_taken_before_an_idle_gap() {
        let probe = Arc::new(CountingProbe::default());
        probe.value.store(100, Ordering::Relaxed);
        let rt = HeadroomRuntime::new(Some(probe.clone() as Arc<dyn VramProbe>));
        let pdev = "p";
        assert!(rt.needs_refresh(pdev), "never read");
        assert_eq!(rt.outside_bytes(pdev), Some(100));
        assert!(!rt.needs_refresh(pdev), "just read");
        let old = Instant::now()
            .checked_sub(Duration::from_secs(876))
            .unwrap();
        rt.cache.lock().unwrap().insert(pdev.to_owned(), (old, 100));
        probe.value.store(6 << 30, Ordering::Relaxed);
        assert!(rt.needs_refresh(pdev), "older than the TTL");
        let reads = probe.reads.load(Ordering::Relaxed);
        assert_eq!(rt.outside_bytes_fresh(pdev), Some(6 << 30));
        assert_eq!(
            probe.reads.load(Ordering::Relaxed),
            reads + 1,
            "read now, synchronously"
        );
        assert!(!rt.needs_refresh(pdev));
        assert_eq!(
            rt.outside_bytes(pdev),
            Some(6 << 30),
            "and served from the cache"
        );
        assert!(
            !HeadroomRuntime::new(None).needs_refresh(pdev),
            "no probe: nothing to read"
        );
    }

    fn spec() -> HeadroomSpec {
        HeadroomSpec {
            hidden_size: 5120,
            pdev: "0000:03:00.0".to_owned(),
            total_bytes: 24 << 30,
            factor: 2.0,
            margin_bytes: 1 << 29,
            floor_tokens: 4096,
        }
    }

    #[test]
    fn the_floor_applies_and_the_limit_never_reads_zero() {
        let s = spec();
        // plenty of room: the plain limit
        assert!(s.limit_tokens(0, 10 << 30, 0) > 100_000);
        // device full: the floor, not 0
        assert_eq!(s.limit_tokens(0, 30 << 30, 0), 4096);
    }

    #[test]
    fn probe_readings_are_cached_and_survive_a_failing_probe() {
        let probe = Arc::new(CountingProbe::default());
        probe.value.store(500, Ordering::Relaxed);
        let mut rt = HeadroomRuntime::new(Some(Arc::clone(&probe) as Arc<dyn VramProbe>));
        assert_eq!(rt.outside_bytes("p"), Some(500));
        assert_eq!(rt.outside_bytes("p"), Some(500));
        assert_eq!(
            probe.reads.load(Ordering::Relaxed),
            1,
            "second call is cached"
        );
        // expire the cache; a failing probe keeps the last known value
        rt.ttl = Duration::ZERO;
        probe.fail.store(true, Ordering::Relaxed);
        assert_eq!(rt.outside_bytes("p"), Some(500), "last known");
        assert_eq!(rt.outside_bytes("other"), None, "never read");
        assert!(HeadroomRuntime::new(None).outside_bytes("p").is_none());
        assert!(!HeadroomRuntime::new(None).has_probe());
    }

    /// A stale reading is served immediately while one background thread refreshes it: a request
    /// never waits for a `/proc` scan.
    #[test]
    fn a_stale_reading_is_served_while_a_thread_refreshes_it() {
        let probe = Arc::new(CountingProbe::default());
        probe.value.store(100, Ordering::Relaxed);
        let mut rt = HeadroomRuntime::new(Some(Arc::clone(&probe) as Arc<dyn VramProbe>));
        rt.set_ttl(Duration::from_millis(20));
        assert_eq!(rt.outside_bytes("p"), Some(100));
        probe.value.store(900, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(40));
        assert_eq!(
            rt.outside_bytes("p"),
            Some(100),
            "stale value, refresh started"
        );
        let mut seen = None;
        for _ in 0..100 {
            std::thread::sleep(Duration::from_millis(10));
            seen = rt.outside_bytes("p");
            if seen == Some(900) {
                break;
            }
        }
        assert_eq!(seen, Some(900), "the refreshed reading arrives");
    }

    #[test]
    fn leases_add_up_per_domain_and_release_on_drop() {
        let rt = HeadroomRuntime::new(None);
        let s = spec();
        let a = rt.acquire("GPU.1", "m1", &s, 100_000);
        let b = rt.acquire("GPU.1", "m2", &s, 50_000);
        let per_token = 5120 * 4 * 2;
        assert_eq!(rt.lease_bytes("GPU.1", None), 150_000 * per_token);
        assert_eq!(rt.lease_bytes("system", None), 0);
        // a model's own lease does not count against its own limit
        assert_eq!(rt.lease_bytes("GPU.1", Some("m1")), 50_000 * per_token);
        drop(a);
        assert_eq!(rt.lease_bytes("GPU.1", None), 50_000 * per_token);
        drop(b);
        assert_eq!(rt.lease_bytes("GPU.1", None), 0);
    }
}
