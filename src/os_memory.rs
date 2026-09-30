// ============================================================
// src/os_memory.rs — OS-specific system-RAM facts
// ============================================================
// The UMA "system" memory domain (iGPU + CPU + NPU, all drawing from one
// shared DRAM pool) is budgeted as `total system RAM − OS reservation`. Both
// figures are inherently per-OS:
//
//   * Reading total RAM is a different syscall on each OS (`/proc/meminfo` on
//     Linux, `GlobalMemoryStatusEx` on Windows).
//   * The right OS *reservation* differs too: a headless-ish Linux box needs
//     less held back than Windows 11 (idle ~3–4 GB) + Arc drivers + the
//     desktop/compositor + Defender. So the default is decided per OS, not
//     shared (2026-06-15): Linux 4 GB, Windows 6 GB.
//
// KYE (the reason Phase A gated this): an integrated GPU's
// `GPU_DEVICE_TOTAL_MEM_SIZE` reports ~the whole shared pool (observed on one
// integrated-GPU box: reported 14.02 of 15.38 GiB total), NOT a private
// carve-out. So the system
// budget must come from OS RAM here, never the iGPU's figure, and never the sum
// of per-device figures (that would multiply-count the one shared pool → OOM).
//
// Because we read total RAM (a static ceiling) and do our own allocation
// accounting in `MemoryTracker`, total — not "available" — RAM is the right
// input: it is deterministic and not perturbed by transient usage.
// ============================================================

/// Default OS reservation (GB) held back from total RAM for the system domain,
/// chosen per OS. Operator-overridable via `config.system_ram_reservation_gb`.
#[must_use]
pub fn default_reservation_gb() -> f64 {
    #[cfg(target_os = "windows")]
    {
        6.0 // Windows 11 + Arc drivers + desktop/compositor + Defender
    }
    #[cfg(not(target_os = "windows"))]
    {
        4.0 // Linux: server-ish use, but a dev box may also run a desktop
    }
}

/// Total physical system RAM in GiB, or `None` when this OS's reader is not
/// implemented or the query fails (the caller then leaves the system domain
/// gating-disabled rather than guessing a budget).
///
/// Linux reads `/proc/meminfo` (`MemTotal`); Windows reads `ullTotalPhys` from
/// `GlobalMemoryStatusEx`. Other OSes return `None`.
#[must_use]
pub fn total_ram_gb() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kb = parse_meminfo_total_kb(&contents)?;
        #[allow(clippy::cast_precision_loss)]
        Some(kb as f64 / (1024.0 * 1024.0))
    }
    #[cfg(target_os = "windows")]
    {
        win_mem::query().map(|(total_phys, _avail)| bytes_to_gb(total_phys))
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

/// The local hostname — for tying a generation record back to the per-box
/// benchmark results. Linux reads `/proc/sys/kernel/hostname` directly (same dependency-free
/// `/proc` convention as [`total_ram_gb`] — the whole fleet is Linux); Windows
/// uses the already-Windows-gated `sysinfo` crate this module's memory query
/// also uses. `None` if unreadable/empty on Linux, or unreported on Windows —
/// omitted from callers' output, never fabricated.
#[must_use]
pub fn host_name() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let raw = std::fs::read_to_string("/proc/sys/kernel/hostname").ok()?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_owned())
        }
    }
    #[cfg(target_os = "windows")]
    {
        sysinfo::System::host_name()
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

/// Parses the `MemTotal:` line (kB) out of `/proc/meminfo` contents.
///
/// The line looks like `MemTotal:       16129844 kB`. Returns `None` if the key
/// is absent or the value does not parse — the caller degrades to "no budget".
#[cfg(any(target_os = "linux", test))]
#[must_use]
pub fn parse_meminfo_total_kb(contents: &str) -> Option<u64> {
    parse_meminfo_field_kb(contents, "MemTotal:")
}

/// Live free system RAM in GiB — the kernel's `MemAvailable` estimate (free +
/// reclaimable), or `None` when this OS's reader is not implemented or the query
/// fails.
///
/// Unlike [`total_ram_gb`] (a deterministic static ceiling for the *accounting*
/// budget), this is the **live** figure used as an OOM safety floor at model
/// load time: the logical VRAM tracker cannot see real RSS, `OpenVINO` runtime
/// overhead, or memory held by *other* processes, so on a UMA box a load the
/// accounting "fits" can still OOM the machine. A live read catches that.
///
/// Linux reads `/proc/meminfo` (`MemAvailable`, which includes reclaimable
/// cache); Windows reads `ullAvailPhys` from `GlobalMemoryStatusEx` (free
/// physical only — slightly more conservative, the safe direction for an OOM
/// floor). Other OSes return `None`.
#[must_use]
pub fn available_ram_gb() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kb = parse_meminfo_available_kb(&contents)?;
        #[allow(clippy::cast_precision_loss)]
        Some(kb as f64 / (1024.0 * 1024.0))
    }
    #[cfg(target_os = "windows")]
    {
        win_mem::query().map(|(_total, avail_phys)| bytes_to_gb(avail_phys))
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

/// Parses the `MemAvailable:` line (kB) out of `/proc/meminfo` contents.
#[cfg(any(target_os = "linux", test))]
#[must_use]
pub fn parse_meminfo_available_kb(contents: &str) -> Option<u64> {
    parse_meminfo_field_kb(contents, "MemAvailable:")
}

/// Returns `true` when a load needing `need_gb` may proceed without pushing live
/// free RAM below `floor_gb` — i.e. `avail_gb - need_gb >= floor_gb`. The pure
/// arithmetic core of the M1 system-RAM admission gate, split out for testing.
#[must_use]
pub fn ram_floor_admits(avail_gb: f64, need_gb: f64, floor_gb: f64) -> bool {
    avail_gb - need_gb >= floor_gb
}

/// Reclaimable GPU page-cache in GiB — memory the kernel counts as *used* but
/// will hand back on demand, and which `MemAvailable` therefore **omits**.
///
/// This exists because of a real, measured defect (2026-09-07,
/// the project's internal engineering log): on a UMA box the
/// DRM/TTM subsystem keeps freed GPU pages in a pool for fast reuse. Those
/// pages are `used` and absent from `MemAvailable`, so the M1 gate refused a
/// 12 GB model on a 30 GB machine with "this load can never fit" — while
/// ~8.8 GB sat in that pool, reclaimable, waiting to back exactly such a load.
/// Worse, the pool grows with every load/evict cycle, so the server got
/// *worse* at loading models the more it had been used.
///
/// Returns `None` (credit nothing — the conservative direction) when the TTM
/// pool ceiling can't be read, so a box without the driver, or a non-Linux
/// host, keeps today's behaviour exactly.
#[must_use]
pub fn reclaimable_gpu_cache_gb() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let pool_cap_pages =
            std::fs::read_to_string("/sys/module/ttm/parameters/page_pool_size").ok()?;
        let pool_cap_pages: u64 = pool_cap_pages.trim().parse().ok()?;
        Some(reclaimable_gpu_cache_gb_from(
            &meminfo,
            zram_used_bytes(),
            pool_cap_pages.saturating_mul(4096),
        ))
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Sum of every zram device's `mem_used_total` (3rd field of `mm_stat`) in
/// bytes. zram's compressed pool is real RAM that is *not* reclaimable by
/// dropping caches, so it must be excluded from the GPU-cache credit or the
/// gate would over-credit by however much has been swapped out.
#[cfg(target_os = "linux")]
#[must_use]
fn zram_used_bytes() -> u64 {
    let Ok(entries) = std::fs::read_dir("/sys/block") else {
        return 0;
    };
    entries
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("zram"))
        .filter_map(|e| std::fs::read_to_string(e.path().join("mm_stat")).ok())
        .filter_map(|s| {
            s.split_whitespace()
                .nth(2)
                .and_then(|v| v.parse::<u64>().ok())
        })
        .sum()
}

/// Pure core of [`reclaimable_gpu_cache_gb`], split out so the arithmetic is
/// testable on any platform against captured `/proc/meminfo` samples.
///
/// Computes the memory that `free` calls *used* but which no standard counter
/// accounts for — `MemTotal` minus free, page cache, anonymous pages and the
/// kernel's own structures — then removes zram's compressed pool (real RAM,
/// but not reclaimable) and clamps the remainder into `[0, pool_cap_bytes]`.
/// The clamp is what keeps this honest: whatever the unaccounted memory
/// actually is, the credit can never exceed the TTM pool's own ceiling.
///
/// The result is an **upper bound** on what is genuinely reclaimable, and is
/// known to overshoot: on the 2026-09-07 sample this computes 10.16 GB while
/// `drop_caches` actually returned 8.63 GB. The ~1.5 GB residue is *not*
/// explained by the kernel fields subtracted above (`VmallocUsed` + `Percpu`
/// are only ~0.11 GB here) — it is either pool memory `drop_caches` declines
/// to drain, or driver memory that is genuinely not reclaimable. Treat this as
/// an optimistic estimate, which is precisely why it is opt-in and why two
/// independent bounds sit between it and an OOM: `system_ram_budget_gb` caps
/// the resulting figure, and `system_ram_reservation_gb` is still subtracted
/// on top as the OS floor.
#[cfg(any(target_os = "linux", test))]
#[must_use]
pub fn reclaimable_gpu_cache_gb_from(meminfo: &str, zram_bytes: u64, pool_cap_bytes: u64) -> f64 {
    let field = |k: &str| parse_meminfo_field_kb(meminfo, k).unwrap_or(0);
    let Some(total) = parse_meminfo_field_kb(meminfo, "MemTotal:") else {
        return 0.0;
    };
    let accounted = field("MemFree:")
        .saturating_add(field("Buffers:"))
        .saturating_add(field("Cached:"))
        .saturating_add(field("AnonPages:"))
        .saturating_add(field("Slab:"))
        .saturating_add(field("PageTables:"))
        .saturating_add(field("KernelStack:"))
        // Kernel allocations that are neither cache nor anonymous and are not
        // reclaimable by dropping caches. Small in practice (~0.11 GB measured
        // on this box) but correct to exclude.
        .saturating_add(field("VmallocUsed:"))
        .saturating_add(field("Percpu:"));
    let unaccounted_bytes = total
        .saturating_sub(accounted)
        .saturating_mul(1024)
        .saturating_sub(zram_bytes);
    #[allow(clippy::cast_precision_loss)]
    {
        unaccounted_bytes.min(pool_cap_bytes) as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Parses a `<Key>: <value> kB` line out of `/proc/meminfo` contents.
#[cfg(any(target_os = "linux", test))]
#[must_use]
fn parse_meminfo_field_kb(contents: &str, key: &str) -> Option<u64> {
    contents
        .lines()
        .find_map(|line| line.strip_prefix(key))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|n| n.parse::<u64>().ok())
}

/// Bytes → GiB. The conversion the Windows readers apply to
/// `GlobalMemoryStatusEx` figures; split out so the arithmetic is unit-testable
/// on any platform without a live memory query.
#[cfg(any(target_os = "windows", test))]
#[must_use]
pub fn bytes_to_gb(bytes: u64) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    {
        bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Windows physical-memory query via the cross-platform, well-tested `sysinfo`
/// crate — no hand-rolled `unsafe` FFI. `sysinfo` reads `ullTotalPhys` /
/// `ullAvailPhys` from `GlobalMemoryStatusEx` internally. Returns
/// `(total_phys_bytes, avail_phys_bytes)`, or `None` when no total is reported.
#[cfg(target_os = "windows")]
mod win_mem {
    pub(super) fn query() -> Option<(u64, u64)> {
        use sysinfo::System;
        let mut sys = System::new();
        sys.refresh_memory();
        let total = sys.total_memory();
        let avail = sys.available_memory();
        if total == 0 {
            None
        } else {
            Some((total, avail))
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    /// `host_name()` on the real running box (this test binary's own host):
    /// non-empty, stable across calls, and — the specific thing left
    /// untested before — no trailing newline/whitespace left over from the
    /// raw `/proc/sys/kernel/hostname` read.
    #[cfg(target_os = "linux")]
    #[test]
    fn host_name_is_trimmed_nonempty_and_stable() {
        let a = host_name().expect("this host reports a hostname");
        let b = host_name();
        assert_eq!(Some(a.clone()), b, "must be stable across calls");
        assert!(!a.is_empty());
        assert_eq!(
            a,
            a.trim(),
            "must not carry the /proc read's trailing newline"
        );
    }

    #[test]
    fn parses_real_meminfo_line() {
        let sample = "MemFree:         1234 kB\nMemTotal:       16129844 kB\nBuffers: 5 kB\n";
        assert_eq!(parse_meminfo_total_kb(sample), Some(16_129_844));
    }

    #[test]
    fn missing_memtotal_is_none() {
        assert_eq!(parse_meminfo_total_kb("MemFree: 100 kB\n"), None);
    }

    #[test]
    fn garbled_value_is_none() {
        assert_eq!(
            parse_meminfo_total_kb("MemTotal:   not-a-number kB\n"),
            None
        );
    }

    #[test]
    fn parses_real_memavailable_line() {
        let sample = "MemTotal: 16129844 kB\nMemFree: 100 kB\nMemAvailable:  9876543 kB\n";
        assert_eq!(parse_meminfo_available_kb(sample), Some(9_876_543));
    }

    #[test]
    fn missing_memavailable_is_none() {
        assert_eq!(parse_meminfo_available_kb("MemTotal: 100 kB\n"), None);
    }

    #[test]
    fn bytes_to_gb_converts() {
        // 32 GiB exactly, and the zero edge.
        assert!((bytes_to_gb(34_359_738_368) - 32.0).abs() < 1e-9);
        assert!(bytes_to_gb(0).abs() < 1e-9);
    }

    #[test]
    fn ram_floor_admits_basic() {
        // 20 avail, need 10, keep 4 floor → 20-10=10 >= 4 → admit.
        assert!(ram_floor_admits(20.0, 10.0, 4.0));
        // 20 avail, need 17, keep 4 floor → 20-17=3 < 4 → refuse.
        assert!(!ram_floor_admits(20.0, 17.0, 4.0));
        // Exactly on the floor admits (>=).
        assert!(ram_floor_admits(20.0, 16.0, 4.0));
    }

    #[test]
    fn default_reservation_is_positive() {
        // Per-OS, but always a sane positive figure.
        assert!(default_reservation_gb() > 0.0);
    }

    /// On Linux the live `/proc/meminfo` read returns a plausible figure (or
    /// `None` in an exotic sandbox — tolerated so the test never flakes).
    #[cfg(target_os = "linux")]
    #[test]
    fn linux_total_ram_is_plausible_when_readable() {
        if let Some(gb) = total_ram_gb() {
            assert!(gb > 0.1 && gb < 100_000.0, "implausible RAM figure: {gb}");
        }
    }

    /// A real sample from a shared-memory (UMA) laptop in the broken state:
    /// `free` said 15.6 GB used, but only ~3 GB was accounted for by any
    /// standard counter. The credit must recover the ~8.8 GB the TTM pool was
    /// holding — the exact memory `MemAvailable` omitted and the gate needed.
    #[test]
    fn gpu_cache_credit_recovers_the_measured_ttm_pool() {
        // MemTotal 30.47 GB; unaccounted 12.66 GB of which zram held 2.51 GB.
        let meminfo = "\
MemTotal:       31954944 kB
MemFree:        11764736 kB
Buffers:           62914 kB
Cached:          3743416 kB
AnonPages:       2055208 kB
Slab:             975175 kB
PageTables:        52428 kB
KernelStack:       20971 kB
";
        let zram = 2_694_915_000_u64; // ~2.51 GiB, real RAM but NOT reclaimable
        let pool_cap = 3_993_416_u64 * 4096; // 15.23 GiB, this box's page_pool_size
        let credit = reclaimable_gpu_cache_gb_from(meminfo, zram, pool_cap);
        // 10.16 GB, against the 8.63 GB `drop_caches` actually recovered on the
        // live box. The credit is deliberately an upper bound (see the function
        // doc); what matters for the gate is that it recovers the bulk of the
        // pool the old MemAvailable-only reading missed entirely.
        assert!(
            (10.0..10.3).contains(&credit),
            "expected ~10.16 GB from this sample, got {credit}"
        );
        assert!(
            credit > 8.63,
            "credit must at least cover the memory drop_caches provably freed"
        );
    }

    /// The clamp is the safety property: whatever the unaccounted memory
    /// actually is, the credit can never exceed the TTM pool's own ceiling.
    #[test]
    fn gpu_cache_credit_is_clamped_to_the_pool_ceiling() {
        let meminfo = "MemTotal: 31954944 kB\nMemFree: 1000 kB\n";
        let credit = reclaimable_gpu_cache_gb_from(meminfo, 0, 1024 * 1024 * 1024);
        assert!(
            (credit - 1.0).abs() < 1e-9,
            "credit must clamp to the 1 GiB pool cap, got {credit}"
        );
    }

    /// zram's compressed pool is real RAM that dropping caches will not free,
    /// so it must reduce the credit — otherwise a heavily-swapped box would be
    /// over-credited by exactly the amount it can least afford.
    #[test]
    fn gpu_cache_credit_excludes_zram() {
        let meminfo = "MemTotal: 8388608 kB\nMemFree: 4194304 kB\n";
        let cap = 64 * 1024 * 1024 * 1024;
        let without = reclaimable_gpu_cache_gb_from(meminfo, 0, cap);
        let with_zram = reclaimable_gpu_cache_gb_from(meminfo, 1024 * 1024 * 1024, cap);
        assert!(
            (without - with_zram - 1.0).abs() < 1e-6,
            "1 GiB of zram must cost exactly 1 GiB of credit: {without} vs {with_zram}"
        );
    }

    /// Unparseable/foreign `/proc/meminfo` must credit nothing rather than
    /// guess — the conservative direction for an OOM gate.
    #[test]
    fn gpu_cache_credit_is_zero_without_memtotal() {
        let credit = reclaimable_gpu_cache_gb_from("Nonsense: 1 kB\n", 0, u64::MAX);
        assert!(credit.abs() < f64::EPSILON, "expected 0.0, got {credit}");
    }

    /// Guards the arithmetic an earlier investigation got wrong by hand: the OS
    /// floor is subtracted **on top of** the need, not folded into it.
    #[test]
    fn ram_floor_is_additive_on_top_of_the_need() {
        assert!(
            !ram_floor_admits(14.8, 14.5, 2.0),
            "14.8 - 14.5 = 0.3 < 2.0 floor"
        );
        assert!(
            ram_floor_admits(23.4, 14.5, 2.0),
            "23.4 - 14.5 = 8.9 >= 2.0 floor"
        );
    }
}
