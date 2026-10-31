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

/// F2-N-5: a reading served this long after it was taken means the background refresh keeps
/// failing; logged once per device until a refresh succeeds again.
const STALE_WARN_AFTER: Duration = Duration::from_secs(60);

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
    /// `pdev`s whose stale reading was already logged (F2-N-5); cleared by a good refresh.
    stale_warned: Arc<Mutex<HashSet<String>>>,
    ttl: Duration,
    /// Transient reservations of running long requests: domain -> model -> bytes.
    leases: Arc<Mutex<HashMap<String, HashMap<String, u64>>>>,
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
            leases: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Tests: a zero TTL makes every reading fresh and synchronous.
    #[cfg(test)]
    pub(crate) fn set_ttl(&mut self, ttl: Duration) {
        self.ttl = ttl;
    }

    /// Whether a probe exists at all (`rv status` reports `headroom: unknown` without one).
    pub(crate) fn has_probe(&self) -> bool {
        self.probe.is_some()
    }

    /// Bytes other processes hold on `pdev`. The first reading is taken synchronously; after that
    /// a reading older than the TTL is served as it is while one background thread refreshes it,
    /// so a request never waits for a `/proc` scan. When the probe fails the last known value
    /// stays; `None` only if it never worked.
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
            Some((at, bytes)) if !self.ttl.is_zero() => {
                if at.elapsed() >= STALE_WARN_AFTER
                    && self
                        .stale_warned
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .insert(pdev.to_owned())
                {
                    tracing::warn!(
                        pdev,
                        age_secs = at.elapsed().as_secs(),
                        "device-memory probe keeps failing: serving its last reading; memory other \
                         processes took since then is covered only by the headroom margin"
                    );
                }
                self.refresh_in_background(Arc::clone(probe), pdev);
                Some(bytes)
            }
            _ => {
                // First reading (or a zero TTL): read now, without holding any lock.
                let fresh = probe.outside_used_bytes(pdev);
                let mut cache = self
                    .cache
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
        }
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
                    if let Some(bytes) = probe.outside_used_bytes(&pdev) {
                        cache
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .insert(pdev.clone(), (Instant::now(), bytes));
                        stale_warned
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .remove(&pdev);
                    }
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

    /// F2-N-5: a reading older than a minute (the refresh keeps failing) is still served and
    /// marked as logged once; a good refresh re-arms the warning.
    #[test]
    fn a_stale_reading_is_served_and_logged_once_until_a_refresh_succeeds() {
        let probe = Arc::new(CountingProbe::default());
        probe.value.store(7, Ordering::Relaxed);
        let rt = HeadroomRuntime::new(Some(probe.clone() as Arc<dyn VramProbe>));
        let pdev = "0000:03:00.0";
        let old = Instant::now()
            .checked_sub(STALE_WARN_AFTER + Duration::from_secs(1))
            .unwrap();
        rt.cache.lock().unwrap().insert(pdev.to_owned(), (old, 5));
        probe.fail.store(true, Ordering::Relaxed);
        assert_eq!(
            rt.outside_bytes(pdev),
            Some(5),
            "the last reading is served"
        );
        assert!(rt.stale_warned.lock().unwrap().contains(pdev), "logged");
        // Wait for the (failing) background refresh to finish, then let one succeed.
        while rt.refreshing.lock().unwrap().contains(pdev) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            rt.stale_warned.lock().unwrap().contains(pdev),
            "a failed refresh keeps it"
        );
        probe.fail.store(false, Ordering::Relaxed);
        rt.cache.lock().unwrap().insert(pdev.to_owned(), (old, 5));
        let _ = rt.outside_bytes(pdev);
        while rt.refreshing.lock().unwrap().contains(pdev) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            !rt.stale_warned.lock().unwrap().contains(pdev),
            "a good refresh re-arms"
        );
        assert_eq!(rt.cache.lock().unwrap().get(pdev).map(|(_, b)| *b), Some(7));
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
