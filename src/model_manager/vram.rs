// ============================================================
// src/model_manager/vram.rs — internal memory accounting, per domain
// ============================================================
// OpenVINO Core instances cannot see each other's allocations —
// querying `GPU_MEMORY_STATISTICS` from one Core always returns
// zero for memory held by another Core in the same process.
// This module maintains the ground-truth accounting table that
// `ModelManager` updates on every load and eviction.
//
// CRASH COURSE — why not query OpenVINO?
//   In Python stormVINO this was discovered the hard way: after
//   loading two models, each Core reported ~0 GB for the other
//   model's allocation. In Rust the situation is identical —
//   the only reliable VRAM source is the table we keep ourselves.
//
// CRASH COURSE — why "memory domains", not "VRAM"?
//   On the discrete Arc fleet each GPU has a private VRAM pool, so a
//   single global budget was enough. On a Lunar Lake laptop the CPU +
//   iGPU + NPU all allocate from one shared LPDDR5X system-RAM pool. A
//   per-*device* budget there would let every device believe it owns the
//   whole system RAM → free memory multiply-counted → OOM. So the tracker
//   keys budgets by **memory domain**: a discrete GPU is its own private
//   domain; every integrated GPU / CPU / NPU joins the shared "system"
//   domain (see `device_inventory::DOMAIN_SYSTEM`). Admission and eviction
//   are per-domain. A single-dGPU box has exactly one domain and behaves
//   exactly as the old single-budget tracker did.
//
// CRASH COURSE — why the 0.0 "disabled" mode?
//   CPU inference has no VRAM budget; cloud runners may have more
//   memory than any single model needs. A domain with `total_gb = 0.0`
//   skips all gating — `fits()` always returns true for it. An unknown
//   domain (one never registered) is treated the same way.
// ============================================================

use std::collections::HashMap;

/// One memory domain's budget and its current per-model allocations.
///
/// All values are in gigabytes (GB). A `total_gb` of `0.0` disables gating for
/// this domain (see module docs).
struct DomainBudget {
    /// Total usable memory in this domain. `0.0` = gating disabled.
    total_gb: f64,
    /// Currently allocated memory in this domain, keyed by model ID.
    allocated: HashMap<String, f64>,
}

impl DomainBudget {
    fn new(total_gb: f64) -> Self {
        Self {
            total_gb,
            allocated: HashMap::new(),
        }
    }

    /// Memory reserved across all models in this domain.
    fn used_gb(&self) -> f64 {
        self.allocated.values().copied().sum()
    }

    /// Memory not yet claimed in this domain. Saturates at zero.
    fn free_gb(&self) -> f64 {
        (self.total_gb - self.used_gb()).max(0.0)
    }
}

/// Tracks memory allocated to loaded models, partitioned by memory domain.
///
/// Each model's allocation is charged to the domain of the device it loaded on.
/// Thread-safety is the caller's responsibility — `ModelManager` wraps this in
/// `Arc<RwLock<MemoryTracker>>`.
pub(super) struct MemoryTracker {
    /// One budget per memory domain, keyed by `domain_id`.
    domains: HashMap<String, DomainBudget>,
}

impl MemoryTracker {
    /// Creates a tracker with one budget per `(domain_id, total_gb)` pair.
    ///
    /// Pass `total_gb = 0.0` for a domain to disable its gating — [`fits`](Self::fits)
    /// will always return `true` for it. A single-dGPU box passes exactly one
    /// pair (`{device → total_vram_gb}`) and the tracker behaves as the old
    /// single-budget `VramTracker` did.
    pub(super) fn new(budgets: impl IntoIterator<Item = (String, f64)>) -> Self {
        let domains = budgets
            .into_iter()
            .map(|(id, total_gb)| (id, DomainBudget::new(total_gb)))
            .collect();
        Self { domains }
    }

    /// Record `gb` gigabytes allocated to `model_id` in `domain`.
    ///
    /// Overwrites any previous entry for the same model ID. Callers pre-allocate
    /// before the load attempt and call [`free`](Self::free) on failure. An
    /// allocation against an unregistered domain creates it as gating-disabled
    /// (`total_gb = 0.0`) — the accounting is still tracked, just never gated.
    pub(super) fn allocate(&mut self, domain: &str, model_id: &str, gb: f64) {
        self.domains
            .entry(domain.to_owned())
            .or_insert_with(|| DomainBudget::new(0.0))
            .allocated
            .insert(model_id.to_owned(), gb);
    }

    /// Release the allocation for `model_id` from whichever domain holds it.
    ///
    /// Model IDs are globally unique, so a model is charged to at most one
    /// domain; this removes it wherever it lives. No-op if the model was never
    /// allocated (e.g. a `0.0`-budget domain, or an already-freed model). The
    /// domain need not be known by the caller — eviction and load-rollback paths
    /// free by model ID alone.
    pub(super) fn free(&mut self, model_id: &str) {
        for budget in self.domains.values_mut() {
            budget.allocated.remove(model_id);
        }
    }

    /// Returns `true` if `domain` has room for an additional `gb` gigabytes.
    ///
    /// A domain with `total_gb = 0.0` (gating disabled), or one that was never
    /// registered, always returns `true`.
    pub(super) fn fits(&self, domain: &str, gb: f64) -> bool {
        match self.domains.get(domain) {
            Some(b) if b.total_gb > 0.0 => b.free_gb() >= gb,
            _ => true, // disabled domain or unknown domain → no gating
        }
    }

    /// Gigabytes not yet claimed by any loaded model in `domain`.
    ///
    /// Saturates at zero — will not return a negative value even if recorded
    /// allocations exceed the domain budget (a misconfigured budget could cause
    /// this). An unknown domain reports `0.0` free.
    pub(super) fn free_gb(&self, domain: &str) -> f64 {
        self.domains.get(domain).map_or(0.0, DomainBudget::free_gb)
    }

    /// Gigabytes currently reserved by `model_id`, across whichever domain holds
    /// it (model IDs are globally unique → at most one domain). `0.0` if the
    /// model holds no reservation. Used by the dry-run admission check
    /// ([`ModelManager::check_admission`](super::ModelManager::check_admission))
    /// to compute how much VRAM evicting a given resident would free.
    pub(super) fn allocated_gb(&self, model_id: &str) -> f64 {
        self.domains
            .values()
            .find_map(|b| b.allocated.get(model_id).copied())
            .unwrap_or(0.0)
    }

    /// Per-domain snapshot for Prometheus gauges: one `(domain_id, total_gb, used_gb)`
    /// triple per tracked domain. Gating-disabled domains (`total_gb == 0.0`) are
    /// included — their `used_gb` can be non-zero (allocations are still tracked).
    pub(super) fn domain_snapshot(&self) -> Vec<(String, f64, f64)> {
        self.domains
            .iter()
            .map(|(id, b)| (id.clone(), b.total_gb, b.used_gb()))
            .collect()
    }

    /// Total and used gigabytes for exactly one `domain` — `(0.0, 0.0)` if it
    /// isn't tracked. The legacy `device`-labelled `/metrics` gauges
    /// (`rustedvino_vram_total_bytes`/`_used_bytes`) need exactly this: a prior
    /// version summed budgets across *every* domain instead, which happened to
    /// equal a single device's own numbers only on a single-domain box — and
    /// silently double-counted the "system" domain (CPU/iGPU, for
    /// TTS/STT/embedding) into the GPU's reported total the moment co-residency
    /// introduced a second domain.
    pub(super) fn domain_gb(&self, domain: &str) -> (f64, f64) {
        self.domains
            .get(domain)
            .map_or((0.0, 0.0), |b| (b.total_gb, b.used_gb()))
    }
}

// ============================================================
// Unit tests
// ============================================================
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    /// The discrete-GPU domain id used throughout these single-domain tests —
    /// stands in for `config.device` (a discrete GPU's domain is its own name).
    const DGPU: &str = "GPU.1";

    /// Convenience: a single-domain tracker, the back-compat shape.
    fn single(total_gb: f64) -> MemoryTracker {
        MemoryTracker::new([(DGPU.to_owned(), total_gb)])
    }

    /// A new tracker reports all memory in the domain as free.
    #[test]
    fn test_tracker_starts_fully_free() {
        let t = single(22.5);
        assert!(
            (t.free_gb(DGPU) - 22.5).abs() < 1e-9,
            "new tracker must be fully free"
        );
        assert!(t.fits(DGPU, 22.5));
    }

    /// Allocating a model reduces free GB by the allocated amount.
    #[test]
    fn test_allocate_reduces_free_gb() {
        let mut t = single(22.5);
        t.allocate(DGPU, "model-a", 5.5);
        assert!((t.free_gb(DGPU) - 17.0).abs() < 0.001);
    }

    /// Freeing a model restores its GB to the free pool.
    #[test]
    fn test_free_restores_capacity() {
        let mut t = single(22.5);
        t.allocate(DGPU, "model-a", 5.5);
        t.free("model-a");
        assert!((t.free_gb(DGPU) - 22.5).abs() < 0.001);
    }

    /// `fits` returns `true` when there is enough headroom.
    #[test]
    fn test_fits_true_when_enough_space() {
        let mut t = single(22.5);
        t.allocate(DGPU, "model-a", 10.0);
        assert!(t.fits(DGPU, 10.0), "10 GB should fit in 12.5 GB free");
    }

    /// `fits` returns `false` when the request exceeds free GB.
    #[test]
    fn test_fits_false_when_too_full() {
        let mut t = single(22.5);
        t.allocate(DGPU, "model-a", 20.0);
        assert!(!t.fits(DGPU, 5.0), "5 GB should NOT fit in 2.5 GB free");
    }

    /// `fits` always returns `true` when the domain budget is `0.0` (disabled).
    #[test]
    fn test_fits_always_when_gating_disabled() {
        let t = single(0.0);
        assert!(
            t.fits(DGPU, 9999.0),
            "gating disabled must always report fit"
        );
    }

    /// `fits` against a never-registered domain is unconstrained (no gating).
    #[test]
    fn test_fits_true_for_unknown_domain() {
        let t = single(10.0);
        assert!(
            t.fits("system", 9999.0),
            "an unknown domain must not gate (treated as disabled)"
        );
        assert!(
            (t.free_gb("system")).abs() < 1e-9,
            "unknown domain reports 0.0 free"
        );
    }

    /// Freeing a model ID that was never allocated is a safe no-op.
    #[test]
    fn test_free_unknown_model_is_noop() {
        let mut t = single(10.0);
        t.free("ghost-model"); // must not panic
        assert!((t.free_gb(DGPU) - 10.0).abs() < 0.001);
    }

    /// Allocating a second model accumulates correctly within the domain.
    #[test]
    fn test_accumulates_multiple_models() {
        let mut t = single(22.5);
        t.allocate(DGPU, "model-a", 5.5);
        t.allocate(DGPU, "model-b", 9.0);
        assert!((t.free_gb(DGPU) - 8.0).abs() < 0.001);
        assert!(!t.fits(DGPU, 9.0), "9 GB should NOT fit in 8 GB free");
        assert!(t.fits(DGPU, 8.0), "8 GB should fit exactly");
    }

    /// `domain_gb`'s used half is the complement of `free_gb` and the sum of
    /// every allocation in that domain, returning to zero once all models are
    /// freed (the evict invariant); the total half never changes.
    #[test]
    fn test_domain_gb_tracks_allocations_and_resets() {
        let mut t = single(22.5);
        let (total, used) = t.domain_gb(DGPU);
        assert!((total - 22.5).abs() < 1e-9 && used.abs() < 1e-9);
        t.allocate(DGPU, "model-a", 5.5);
        t.allocate(DGPU, "model-b", 9.0);
        let (total, used) = t.domain_gb(DGPU);
        assert!(
            (total - 22.5).abs() < 1e-9 && (used - 14.5).abs() < 1e-9,
            "used = sum of allocations; total unchanged"
        );
        assert!(
            (used + t.free_gb(DGPU) - 22.5).abs() < 1e-9,
            "used + free = total"
        );
        t.free("model-a");
        t.free("model-b");
        let (_, used) = t.domain_gb(DGPU);
        assert!(
            used.abs() < 1e-9,
            "used returns to zero after freeing all models"
        );
    }

    /// An untracked domain reports `(0.0, 0.0)` rather than panicking — the
    /// legacy `/metrics` gauge must degrade gracefully if `inference_domain`
    /// somehow isn't in the tracker (e.g. a poisoned lock upstream substitutes
    /// an empty tracker).
    #[test]
    fn test_domain_gb_unknown_domain_reports_zero() {
        let t = single(22.5);
        assert_eq!(t.domain_gb("no-such-domain"), (0.0, 0.0));
    }

    /// `allocated_gb` reports a model's reservation and `0.0` for an unknown
    /// model — the per-victim freed-VRAM input the dry-run admission check uses.
    #[test]
    fn test_allocated_gb_reports_per_model_reservation() {
        let mut t = single(22.5);
        t.allocate(DGPU, "model-a", 5.5);
        assert!((t.allocated_gb("model-a") - 5.5).abs() < 1e-9);
        assert!(
            t.allocated_gb("ghost").abs() < 1e-9,
            "unknown model holds nothing"
        );
        t.free("model-a");
        assert!(
            t.allocated_gb("model-a").abs() < 1e-9,
            "freed model holds nothing"
        );
    }

    // ── Multi-domain behaviour (the Phase-B generalization) ──────────────────

    /// Two domains track budgets independently: allocating in one does not
    /// reduce free GB in the other, and admission is per-domain.
    #[test]
    fn test_domains_are_independent() {
        let mut t = MemoryTracker::new([("GPU.1".to_owned(), 20.0), ("system".to_owned(), 8.0)]);
        t.allocate("GPU.1", "llm", 15.0);
        // The dGPU domain shrank; the system domain is untouched.
        assert!((t.free_gb("GPU.1") - 5.0).abs() < 1e-9);
        assert!((t.free_gb("system") - 8.0).abs() < 1e-9);
        // A 6 GB request fits in system (8 free) but not in GPU.1 (5 free).
        assert!(t.fits("system", 6.0));
        assert!(!t.fits("GPU.1", 6.0));
        // Each domain's own `domain_gb` pair is independent of the other's
        // allocations — this is the invariant the legacy `/metrics` gauge fix
        // depends on: GPU.1's reported total/used must never include system's.
        t.allocate("system", "embed", 3.0);
        assert_eq!(t.domain_gb("GPU.1"), (20.0, 15.0));
        assert_eq!(t.domain_gb("system"), (8.0, 3.0));
    }

    /// `free` finds the model regardless of which domain it was charged to —
    /// callers free by model ID without knowing the domain.
    #[test]
    fn test_free_is_domain_agnostic() {
        let mut t = MemoryTracker::new([("GPU.1".to_owned(), 20.0), ("system".to_owned(), 8.0)]);
        t.allocate("system", "embed", 3.0);
        t.free("embed"); // caller does not name the domain
        assert!(
            (t.free_gb("system") - 8.0).abs() < 1e-9,
            "system domain restored"
        );
    }
}
