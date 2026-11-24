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

/// Dedicated device memory this process holds on one PCI device (`pdev`), summed over its VRAM
/// regions only: what a model load on that card really took, not system-memory regions or
/// another device's. `None` where [`read_process`] is.
#[must_use]
pub fn process_vram_bytes(pdev: &str) -> Option<u64> {
    read_process().map(|m| {
        m.iter()
            .filter(|((_, p, region), _)| p == pdev && is_device_region(region))
            .map(|(_, b)| *b)
            .sum()
    })
}

// ── Physical probe of the memory other processes hold ────────────────────

/// Reads what the rest of the machine holds of a device, so the server's own bookkeeping can be
/// completed with what it cannot know (the desktop compositor, other users' processes). The
/// manager's tracker supplies this process's own commitments; the probe never reports them.
pub trait VramProbe: Send + Sync + std::fmt::Debug {
    /// Bytes of dedicated device memory held by DRM clients other than this process on the PCI
    /// device `pdev` (`"0000:03:00.0"`), or `None` when they cannot be read.
    fn outside_used_bytes(&self, pdev: &str) -> Option<u64>;

    /// Bytes of dedicated device memory this process itself holds on `pdev`, as the driver
    /// reports it (what the tracker may not have booked: scratch buffers, weights booked at 0).
    /// `None` when unknown, which is the default.
    fn own_used_bytes(&self, _pdev: &str) -> Option<u64> {
        None
    }
}

/// [`VramProbe`] over `/proc/<pid>/fdinfo/<fd>` of every process this user may read (the
/// compositor of the logged-in session is among them; other users' processes and root's are not
/// visible, which is what `kv_headroom_margin_gib` is for).
#[derive(Debug)]
pub struct LinuxFdinfoProbe {
    proc_root: std::path::PathBuf,
    own_pid: u32,
}

impl LinuxFdinfoProbe {
    /// The probe for this process, or `None` off Linux or when `/proc` is not readable.
    #[must_use]
    pub fn new() -> Option<Self> {
        if !cfg!(target_os = "linux") || std::fs::read_dir("/proc/self/fdinfo").is_err() {
            return None;
        }
        Some(Self {
            proc_root: std::path::PathBuf::from("/proc"),
            own_pid: std::process::id(),
        })
    }

    /// A probe over another root (tests build a fake `/proc`).
    #[cfg(test)]
    fn with_root(root: &std::path::Path, own_pid: u32) -> Self {
        Self {
            proc_root: root.to_path_buf(),
            own_pid,
        }
    }
}

/// Dedicated-memory regions: `vram0` on `xe`; `local0` on `i915`. System-memory regions
/// (`gtt`, `system0`) do not count against a discrete GPU's own memory.
fn is_device_region(region: &str) -> bool {
    region.starts_with("vram") || region.starts_with("local")
}

impl VramProbe for LinuxFdinfoProbe {
    fn own_used_bytes(&self, pdev: &str) -> Option<u64> {
        process_vram_bytes(pdev)
    }

    fn outside_used_bytes(&self, pdev: &str) -> Option<u64> {
        let procs = std::fs::read_dir(&self.proc_root).ok()?;
        let mut seen = std::collections::HashSet::new();
        let mut total = 0_u64;
        for entry in procs.filter_map(Result::ok) {
            let Some(pid) = entry
                .file_name()
                .to_str()
                .and_then(|n| n.parse::<u32>().ok())
            else {
                continue;
            };
            if pid == self.own_pid {
                continue;
            }
            // Not readable (another user's process): invisible, by design.
            let Ok(fds) = std::fs::read_dir(entry.path().join("fdinfo")) else {
                continue;
            };
            for fd in fds.filter_map(Result::ok) {
                let Some(e) = std::fs::read_to_string(fd.path())
                    .ok()
                    .and_then(|c| parse_fdinfo(&c))
                else {
                    continue;
                };
                if e.pdev != pdev || !seen.insert((e.driver.clone(), e.pdev.clone(), e.client_id)) {
                    continue;
                }
                total = total.saturating_add(
                    e.regions
                        .iter()
                        .filter(|(r, _)| is_device_region(r))
                        .map(|(_, b)| *b)
                        .sum::<u64>(),
                );
            }
        }
        Some(total)
    }
}

/// `"{domain: 0 bus: 3 device: 0x0 function: 0}"` (`OpenVINO`'s `DEVICE_PCI_INFO`) as the
/// `"0000:03:00.0"` the DRM fdinfo reports, or `None` when it does not parse.
#[must_use]
pub fn pdev_from_pci_info(info: &str) -> Option<String> {
    let field = |name: &str| -> Option<u32> {
        let rest = info
            .split(name)
            .nth(1)?
            .trim_start_matches(':')
            .trim_start();
        let tok = rest.split(|c: char| c.is_whitespace() || c == '}').next()?;
        match tok.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => tok.parse().ok(),
        }
    };
    Some(format!(
        "{:04x}:{:02x}:{:02x}.{:x}",
        field("domain")?,
        field("bus")?,
        field("device")?,
        field("function")?
    ))
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
    /// Time from the last release to [`open`](Self::open); `None` when nothing was released yet.
    since_release: Option<Duration>,
}

/// Why a [`MeasureWindow`] is not clean ([`MeasureWindow::taint`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowTaint {
    /// The window opened within [`QUIET_AFTER_RELEASE`] of an eviction or failed load.
    QuietPeriod {
        /// Time from that release to the window's opening.
        since_release: Duration,
    },
    /// Another load, eviction or failed load happened while the window was open.
    LifecycleEvent,
}

impl WindowTaint {
    /// A stable short code for logs.
    #[must_use]
    pub fn reason(self) -> &'static str {
        match self {
            Self::QuietPeriod { .. } => "within_quiet_period_after_release",
            Self::LifecycleEvent => "lifecycle_event_during_window",
        }
    }
}

impl MeasureWindow {
    /// Open a window now.
    #[must_use]
    pub fn open() -> Self {
        let last = LAST_RELEASE_MS.load(Ordering::Relaxed);
        Self {
            generation: GENERATION.load(Ordering::Relaxed),
            since_release: (last != 0)
                .then(|| Duration::from_millis(now_ms().saturating_sub(last))),
        }
    }

    /// Why the window is not clean, or `None` when nothing else could have changed the process
    /// total since [`open`](Self::open): no release shortly before, and no lifecycle event since.
    #[must_use]
    pub fn taint(&self) -> Option<WindowTaint> {
        match self.since_release {
            Some(since_release) if since_release <= QUIET_AFTER_RELEASE => {
                Some(WindowTaint::QuietPeriod { since_release })
            }
            _ if GENERATION.load(Ordering::Relaxed) != self.generation => {
                Some(WindowTaint::LifecycleEvent)
            }
            _ => None,
        }
    }

    /// Whether nothing else could have changed the process total since
    /// [`open`](Self::open): no lifecycle event since, and no release shortly
    /// before.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.taint().is_none()
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

    /// The taint names its cause: a window opened right after a release reports the quiet
    /// period with the time since that release (the reason the load log gives).
    #[test]
    fn taint_reports_the_quiet_period() {
        note_release_event();
        match MeasureWindow::open().taint() {
            Some(WindowTaint::QuietPeriod { since_release }) => {
                assert!(since_release <= QUIET_AFTER_RELEASE);
            }
            other => panic!("expected the quiet period, got {other:?}"),
        }
        assert_eq!(
            WindowTaint::LifecycleEvent.reason(),
            "lifecycle_event_during_window"
        );
    }

    #[test]
    fn parses_size_units() {
        assert_eq!(parse_size("0"), Some(0));
        assert_eq!(parse_size("3 MiB"), Some(3 << 20));
        assert_eq!(parse_size("1 GiB"), Some(1 << 30));
        assert_eq!(parse_size("5 TiB"), None);
        assert_eq!(parse_size("x KiB"), None);
    }

    #[test]
    fn pdev_from_the_openvino_pci_info() {
        assert_eq!(
            pdev_from_pci_info("{domain: 0 bus: 3 device: 0x0 function: 0}").as_deref(),
            Some("0000:03:00.0")
        );
        assert_eq!(
            pdev_from_pci_info("{domain: 0 bus: 0 device: 0x2 function: 0}").as_deref(),
            Some("0000:00:02.0")
        );
        assert_eq!(
            pdev_from_pci_info("{domain: 1 bus: 0xaf device: 0x1f function: 7}").as_deref(),
            Some("0001:af:1f.7")
        );
        assert_eq!(pdev_from_pci_info("garbage"), None);
    }

    fn write_fdinfo(root: &std::path::Path, pid: u32, fd: u32, body: &str) {
        let dir = root.join(pid.to_string()).join("fdinfo");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(fd.to_string()), body).unwrap();
    }

    fn client(pdev: &str, id: u32, vram_kib: u64, gtt_kib: u64) -> String {
        format!(
            "pos:\t0\ndrm-driver:\txe\ndrm-client-id:\t{id}\ndrm-pdev:\t{pdev}\n\
             drm-total-vram0:\t{vram_kib} KiB\ndrm-total-gtt:\t{gtt_kib} KiB\n"
        )
    }

    /// The probe counts other processes' dedicated memory on one device: not its own process, not
    /// system-memory regions, not other devices, each client once, and ignores non-DRM fds.
    #[test]
    fn linux_probe_sums_other_processes_vram_on_one_device() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        // own process (pid 10): excluded
        write_fdinfo(root, 10, 3, &client("0000:03:00.0", 1, 9_000_000, 1));
        // compositor (pid 20): 500 MiB on the B60, with a duplicate fd of the same client
        write_fdinfo(root, 20, 5, &client("0000:03:00.0", 7, 512_000, 100_000));
        write_fdinfo(root, 20, 6, &client("0000:03:00.0", 7, 512_000, 100_000));
        // another process: 100 MiB on the B60, and some on the iGPU (other device)
        write_fdinfo(root, 30, 4, &client("0000:03:00.0", 8, 102_400, 5));
        write_fdinfo(root, 30, 9, &client("0000:00:02.0", 9, 0, 4_000_000));
        // a non-DRM fd and a non-process entry
        write_fdinfo(root, 30, 11, "pos:\t0\nflags:\t02\n");
        std::fs::create_dir_all(root.join("self")).unwrap();

        let probe = LinuxFdinfoProbe::with_root(root, 10);
        assert_eq!(
            probe.outside_used_bytes("0000:03:00.0"),
            Some((512_000 + 102_400) * 1024)
        );
        assert_eq!(
            probe.outside_used_bytes("0000:00:02.0"),
            Some(0),
            "vram regions only"
        );
        assert_eq!(probe.outside_used_bytes("0000:99:00.0"), Some(0));
        let missing = LinuxFdinfoProbe::with_root(&root.join("nope"), 10);
        assert_eq!(missing.outside_used_bytes("0000:03:00.0"), None);
    }
}
