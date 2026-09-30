// ============================================================
// src/gpu_memory.rs — measured device memory of this process
// ============================================================
// The memory accounting (`MemoryTracker`) is config-declared: each model
// charges its `vram_gb`. Nothing checks that against what the driver actually
// holds, and some usage is invisible to it by construction — e.g. an embedding
// batch's working memory, which OpenVINO keeps until the model is evicted
// (measured 2026-09-29 on multilingual-e5-large: ≈28 MB per 512-token input,
// plateauing at ≈2.5 GB under the default batch budget).
//
// Linux exposes each DRM client's memory in `/proc/<pid>/fdinfo/<fd>`
// (`drm-total-<region>: <n> KiB`, the kernel's drm-usage-stats format), for
// the GPU (`xe`, `i915`) and the NPU (`intel_vpu`) alike. Several fds can share
// one DRM client, so entries are de-duplicated by `(driver, client-id)`. This
// is process-wide; a per-model figure is estimated from deltas around that
// model's own work (see `ModelManager`'s GPU memory estimate).
//
// Other OSes return `None` — the gauges simply don't appear there.
// ============================================================

use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Allocated bytes per `(driver, pdev, region)` for this process, e.g.
/// `("xe", "0000:00:02.0", "gtt") → 4.8 GB`. `pdev` (the PCI device) keeps two
/// GPUs on the same driver apart; it is `""` if the driver doesn't report it.
pub type DrmMemory = BTreeMap<(String, String, String), u64>;

/// One fdinfo file's DRM identity and `drm-total-<region>` sizes, or `None`
/// when the fd is not a DRM client.
#[derive(Debug, PartialEq, Eq)]
struct FdinfoEntry {
    driver: String,
    pdev: String,
    client_id: String,
    regions: Vec<(String, u64)>,
}

/// Parse a `drm-total-*` value: `"<n>"` (bytes) or `"<n> KiB|MiB|GiB"`.
fn parse_size(value: &str) -> Option<u64> {
    let mut parts = value.split_whitespace();
    let n: u64 = parts.next()?.parse().ok()?;
    let mult = match parts.next() {
        None => 1,
        Some("KiB") => 1 << 10,
        Some("MiB") => 1 << 20,
        Some("GiB") => 1 << 30,
        Some(_) => return None,
    };
    n.checked_mul(mult)
}

/// Parse one `/proc/<pid>/fdinfo/<fd>` file.
fn parse_fdinfo(contents: &str) -> Option<FdinfoEntry> {
    let mut driver = None;
    let mut pdev = String::new();
    let mut client_id = None;
    let mut regions = Vec::new();
    for line in contents.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match key {
            "drm-driver" => driver = Some(value.to_owned()),
            "drm-client-id" => client_id = Some(value.to_owned()),
            "drm-pdev" => value.clone_into(&mut pdev),
            _ => {
                // `drm-total-cycles-<engine>` are busy counters, not memory.
                if let Some(region) = key.strip_prefix("drm-total-")
                    && !region.starts_with("cycles-")
                    && let Some(bytes) = parse_size(value)
                {
                    regions.push((region.to_owned(), bytes));
                }
            }
        }
    }
    Some(FdinfoEntry {
        driver: driver?,
        pdev,
        client_id: client_id?,
        regions,
    })
}

/// Sum parsed entries into [`DrmMemory`], counting each
/// `(driver, pdev, client-id)` once however many fds refer to it.
fn aggregate(entries: impl IntoIterator<Item = FdinfoEntry>) -> DrmMemory {
    let mut seen = std::collections::HashSet::new();
    let mut out = DrmMemory::new();
    for e in entries {
        if !seen.insert((e.driver.clone(), e.pdev.clone(), e.client_id.clone())) {
            continue;
        }
        for (region, bytes) in e.regions {
            *out.entry((e.driver.clone(), e.pdev.clone(), region))
                .or_default() += bytes;
        }
    }
    out
}

/// This process's DRM memory per `(driver, region)`, or `None` where the OS
/// doesn't expose it (non-Linux) or `/proc/self/fdinfo` is unreadable.
/// Cheap (a few dozen small file reads) — fine per scrape or per batch.
#[must_use]
pub fn read_process() -> Option<DrmMemory> {
    #[cfg(target_os = "linux")]
    {
        let dir = std::fs::read_dir("/proc/self/fdinfo").ok()?;
        let entries = dir
            .filter_map(Result::ok)
            .filter_map(|e| std::fs::read_to_string(e.path()).ok())
            .filter_map(|c| parse_fdinfo(&c));
        Some(aggregate(entries))
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Total DRM bytes across every driver and region — the scalar the per-model
/// delta estimates are taken on (a load or batch may land on the GPU or the
/// NPU driver). `None` where [`read_process`] is.
#[must_use]
pub fn process_total_bytes() -> Option<u64> {
    read_process().map(|m| m.values().sum())
}

// ── Measurement windows ────────────────────────────────────────────────────
//
// A per-model estimate is a delta of this process-wide total, so anything else
// that changes the total inside the window lands in it. Two such changes are
// known: another model's load (allocations), and an eviction or failed load,
// whose memory the kernel releases *later* — measured ≈11.4 s after the evict
// call returned (xe, Lunar Lake, 2026-09-29). A window is therefore clean only
// when no lifecycle event happened during it and it started at least
// [`QUIET_AFTER_RELEASE`] after the last release.

/// How long after an eviction / failed load a measurement window is presumed
/// polluted by the kernel's delayed release (measured 11.4 s, plus margin).
pub const QUIET_AFTER_RELEASE: Duration = Duration::from_secs(20);

static EPOCH: OnceLock<Instant> = OnceLock::new();
/// Bumped on every lifecycle event (load start/end, eviction, failed load).
static GENERATION: AtomicU64 = AtomicU64::new(0);
/// Milliseconds since `EPOCH` (+1, so 0 = never) of the last release event.
static LAST_RELEASE_MS: AtomicU64 = AtomicU64::new(0);

fn now_ms() -> u64 {
    u64::try_from(EPOCH.get_or_init(Instant::now).elapsed().as_millis())
        .unwrap_or(u64::MAX)
        .saturating_add(1)
}

/// Record a load starting or finishing — it allocates, so any window it
/// overlaps is polluted.
pub fn note_load_event() {
    GENERATION.fetch_add(1, Ordering::Relaxed);
}

/// Record an eviction or failed load — its memory is released later, so
/// windows starting within [`QUIET_AFTER_RELEASE`] are polluted too.
pub fn note_release_event() {
    LAST_RELEASE_MS.store(now_ms(), Ordering::Relaxed);
    GENERATION.fetch_add(1, Ordering::Relaxed);
}

/// A measurement window: open with [`MeasureWindow::open`] before the work,
/// check [`MeasureWindow::is_clean`] after it.
#[derive(Debug, Clone, Copy)]
pub struct MeasureWindow {
    generation: u64,
    quiet_at_start: bool,
}

impl MeasureWindow {
    /// Open a window now.
    #[must_use]
    pub fn open() -> Self {
        let last = LAST_RELEASE_MS.load(Ordering::Relaxed);
        let quiet_ms = u64::try_from(QUIET_AFTER_RELEASE.as_millis()).unwrap_or(u64::MAX);
        Self {
            generation: GENERATION.load(Ordering::Relaxed),
            quiet_at_start: last == 0 || now_ms().saturating_sub(last) > quiet_ms,
        }
    }

    /// Whether nothing else could have changed the process total since
    /// [`open`](Self::open): no lifecycle event since, and no release shortly
    /// before.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.quiet_at_start && GENERATION.load(Ordering::Relaxed) == self.generation
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    const XE: &str = "pos:\t0\nflags:\t02100002\ndrm-driver:\txe\ndrm-pdev:\t0000:00:02.0\n\
        drm-client-id:\t1108\n\
        drm-total-system:\t136352 KiB\ndrm-resident-system:\t0\ndrm-total-gtt:\t4743324 KiB\n\
        drm-total-stolen:\t0\ndrm-total-cycles-rcs:\t6909556267484\ndrm-cycles-rcs:\t12\n";
    const VPU: &str = "drm-driver:\tintel_vpu\ndrm-client-id:\t1119\n\
        drm-total-memory:\t2170248 KiB\ndrm-resident-memory:\t88524 KiB\n";

    /// Real fdinfo lines from a Lunar Lake box: memory regions parsed with
    /// units, busy-cycle counters and non-total keys ignored.
    #[test]
    fn parses_regions_and_skips_cycle_counters() {
        let e = parse_fdinfo(XE).unwrap();
        assert_eq!(e.driver, "xe");
        assert_eq!(e.pdev, "0000:00:02.0");
        assert_eq!(e.client_id, "1108");
        assert_eq!(
            e.regions,
            vec![
                ("system".to_owned(), 136_352 * 1024),
                ("gtt".to_owned(), 4_743_324 * 1024),
                ("stolen".to_owned(), 0),
            ]
        );
        assert!(
            parse_fdinfo("pos:\t0\nflags:\t0\n").is_none(),
            "not a DRM fd"
        );
    }

    fn key(driver: &str, pdev: &str, region: &str) -> (String, String, String) {
        (driver.to_owned(), pdev.to_owned(), region.to_owned())
    }

    /// Several fds on one client count once; different clients add up; a second
    /// GPU on the same driver (another `drm-pdev`) stays a separate series —
    /// the multi-GPU case (a dual-GPU box's `GPU.0` + `GPU.1`).
    #[test]
    fn aggregates_once_per_client_and_per_device() {
        let second_gpu = XE.replace("0000:00:02.0", "0000:03:00.0");
        let files = [XE, XE, VPU, second_gpu.as_str()];
        let m = aggregate(files.iter().map(|c| parse_fdinfo(c).unwrap()));
        assert_eq!(m[&key("xe", "0000:00:02.0", "gtt")], 4_743_324 * 1024);
        assert_eq!(m[&key("xe", "0000:03:00.0", "gtt")], 4_743_324 * 1024);
        assert_eq!(m[&key("intel_vpu", "", "memory")], 2_170_248 * 1024);
        assert_eq!(m.len(), 7);
    }

    /// A load or release after opening dirties the window; a window opened
    /// right after a release is dirty from the start (the kernel frees later).
    /// Only the dirty direction is asserted: other tests running in parallel
    /// bump the same process-wide counters, so "clean" can't be guaranteed.
    #[test]
    fn windows_are_dirtied_by_lifecycle_events() {
        let w = MeasureWindow::open();
        note_load_event();
        assert!(!w.is_clean());
        note_release_event();
        assert!(!MeasureWindow::open().is_clean(), "within the quiet period");
    }

    #[test]
    fn parses_size_units() {
        assert_eq!(parse_size("0"), Some(0));
        assert_eq!(parse_size("3 MiB"), Some(3 << 20));
        assert_eq!(parse_size("1 GiB"), Some(1 << 30));
        assert_eq!(parse_size("5 TiB"), None);
        assert_eq!(parse_size("x KiB"), None);
    }
}
