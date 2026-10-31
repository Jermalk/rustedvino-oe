//! `rustedvino --gpu-probe`: the GPU readiness probe the supervisor runs after a worker left
//! without teardown. On the B60 such an exit can leave the next
//! process's first GPU work never scheduled; a fixed deadline on the first *model* load could
//! not tell that from a long healthy load (a 27B takes 48-78 s). The probe's work is tiny and
//! model-size independent, so a short deadline on this process separates the two.
//!
//! Prints one line per GPU device, `GPU_PROBE device=<name> ok=<bool> ms=<n>[ error=<text>]`,
//! for the supervisor's log and for measuring the normal latency (RECONCILED §2). Exit codes:
//! `0` every GPU passed (or there is none), `2` a GPU failed fast (an error, not a hang).

use std::time::Instant;

/// Exit code when a GPU device answered with an error (not a hang).
pub const EXIT_PROBE_FAILED: u8 = 2;

/// Run the probe on every enumerated GPU device and return the exit code. The bridge releases
/// its objects before each call returns; the process then waits the same short quiesce as a
/// worker's exit before leaving, so it exits with no GPU work in flight.
#[must_use]
pub fn run() -> u8 {
    let gpus: Vec<String> = crate::ov_cb::list_devices()
        .into_iter()
        .filter(|d| d.starts_with("GPU"))
        .collect();
    if gpus.is_empty() {
        println!("GPU_PROBE none");
        return 0;
    }
    let mut failed = false;
    for device in &gpus {
        let started = Instant::now();
        let result = crate::ov_cb::gpu_probe(device);
        let ms = started.elapsed().as_millis();
        match result {
            Ok(()) => println!("GPU_PROBE device={device} ok=true ms={ms}"),
            Err(e) => {
                failed = true;
                println!(
                    "GPU_PROBE device={device} ok=false ms={ms} error={}",
                    e.replace('\n', " ")
                );
            }
        }
    }
    std::thread::sleep(crate::startup::EXIT_QUIESCE);
    if failed { EXIT_PROBE_FAILED } else { 0 }
}
