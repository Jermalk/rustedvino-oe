//! Peak heap allocation of the C28 downscale (0.9.1 review F2), measured with a counting global
//! allocator — this test binary's own, so nothing else runs in it.
//!
//! The ignored test is the measurement on a 64 MP image (run by hand:
//! `scripts/rv-cargo.sh test --test image_downscale_peak -- --ignored --nocapture`); the other
//! one keeps the bound in the gate on a smaller image.
#![allow(clippy::unwrap_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use rustedvino::image_util::{ImageOwner, downscale_rgb, downscaled_dims};

struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards every call to `System` unchanged; the counters are only bookkeeping.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract; passed through as is.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let now = CURRENT.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            PEAK.fetch_max(now, Ordering::SeqCst);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` above with this `layout` (caller's contract).
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Bytes allocated by `f` at its peak, above what was live when it started.
fn peak_above_start<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let start = CURRENT.load(Ordering::SeqCst);
    PEAK.store(start, Ordering::SeqCst);
    let out = f();
    (out, PEAK.load(Ordering::SeqCst) - start)
}

fn source(w: u32, h: u32) -> image::RgbImage {
    #[allow(clippy::cast_possible_truncation)]
    image::RgbImage::from_fn(w, h, |x, y| image::Rgb([x as u8, y as u8, (x ^ y) as u8]))
}

const MIB: f64 = 1024.0 * 1024.0;

/// One measurement at a time: the counters are process-wide, and `--include-ignored` would run
/// the two tests in parallel (review N4).
static MEASURE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// One measurement: the source counts as live at the start (it was decoded before), so the
/// figure is what the downscale adds on top of it.
#[allow(clippy::cast_precision_loss)]
fn measure(w: u32, h: u32, cap: u64) -> (usize, usize) {
    let _one_at_a_time = MEASURE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (nw, nh) = downscaled_dims(w, h, cap).unwrap();
    let src = source(w, h);
    let t = std::time::Instant::now();
    let (old, old_peak) = peak_above_start(|| {
        image::imageops::resize(&src, nw, nh, image::imageops::FilterType::CatmullRom)
    });
    let old_time = t.elapsed();
    assert_eq!(old.dimensions(), (nw, nh));
    drop(old);
    let t = std::time::Instant::now();
    let (new, new_peak) = peak_above_start(|| downscale_rgb(src, cap, ImageOwner::default()));
    let new_time = t.elapsed();
    assert_eq!(new.dimensions(), (nw, nh));
    println!(
        "{w}x{h} -> {nw}x{nh} (cap {cap}): source {:.1} MiB; resize alone +{:.1} MiB in {old_time:?}; \
         two-pass +{:.1} MiB in {new_time:?}",
        f64::from(w) * f64::from(h) * 3.0 / MIB,
        old_peak as f64 / MIB,
        new_peak as f64 / MIB,
    );
    (old_peak, new_peak)
}

/// Gate: 2048 x 2048 under a 250 000 cap (4x per side). The two-pass downscale adds at most
/// ~32 B x cap (its bicubic pass) plus the pre-reduced image; `resize` alone adds
/// `width x new_height x 16 B` (~16 MiB here).
#[test]
fn two_pass_downscale_peak_is_bounded_by_the_cap() {
    let cap = 250_000_u64;
    let (old_peak, new_peak) = measure(2048, 2048, cap);
    let bound = usize::try_from(40 * cap).unwrap();
    assert!(
        new_peak <= bound,
        "two-pass peak {new_peak} B over {bound} B"
    );
    assert!(
        new_peak < old_peak / 2,
        "two-pass {new_peak} B vs resize {old_peak} B"
    );
}

/// The review's case: a 64 MP source (8192 x 8192) under the 4 MP default.
#[test]
#[ignore = "measurement: 64 MP, slow in a debug build; run by hand with --ignored --nocapture"]
fn measure_64_mp_under_the_default_cap() {
    measure(8192, 8192, 4_000_000);
    measure(10_000, 6400, 4_000_000);
    measure(2800, 2800, 4_000_000);
}
