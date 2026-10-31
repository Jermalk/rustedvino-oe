// ============================================================
// src/fault_injection.rs — test-only GPU-poison injection (cargo feature `fault-injection`)
// ============================================================
// Compiled only with `--features fault-injection`; release bundles never contain it. With
// `RUSTEDVINO_FAULT_POISON_AT_STEP=N` set, the Nth continuous-batching step (counted across every
// engine in the process) fails ONCE with the GPU plugin's real `CL_OUT_OF_RESOURCES` text. That
// drives the real poison path — `classify_inference_error` → `mark_gpu_poisoned` → G2 diagnostics
// → `on_gpu_poisoned` (supervised: drain + exit 75 + respawn) — and the G1 client reply, with no
// driver fault. The device itself stays healthy, so it is safe on the B60.
// ============================================================

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// The environment variable that arms the injection (a step number, 1-based).
pub const POISON_AT_STEP_ENV: &str = "RUSTEDVINO_FAULT_POISON_AT_STEP";

/// The error text the GPU plugin produced in the 2026-10-27 B60 VLM c=4 poison, verbatim except
/// for the trailing marker.
pub const INJECTED_POISON_TEXT: &str = "Exception from src/inference/src/cpp/infer_request.cpp:224:\n\
     Exception from src/plugins/intel_gpu/src/runtime/ocl/ocl_stream.cpp:395:\n\
     [GPU] clFinish, error code: -5 CL_OUT_OF_RESOURCES (fault injection)";

static STEPS: AtomicU64 = AtomicU64::new(0);

fn armed_step() -> Option<u64> {
    static AT: OnceLock<Option<u64>> = OnceLock::new();
    *AT.get_or_init(|| {
        let at = std::env::var(POISON_AT_STEP_ENV)
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n| *n > 0);
        if let Some(n) = at {
            tracing::warn!(
                step = n,
                "FAULT INJECTION armed: CB step {n} will fail with a fake GPU poison"
            );
        }
        at
    })
}

/// Log whether the injection is armed; called once at boot so a mistyped variable is visible
/// before the first request (Fable review).
pub fn log_arming() {
    if armed_step().is_none() {
        tracing::warn!(
            "FAULT INJECTION build: {POISON_AT_STEP_ENV} not set (or not a positive number) — nothing will be injected"
        );
    }
}

/// Count one CB step; `true` exactly once, at the armed step.
#[must_use]
pub fn poison_this_step() -> bool {
    poison_at(armed_step(), STEPS.fetch_add(1, Ordering::Relaxed) + 1)
}

/// Pure core of [`poison_this_step`]: fire when step number `n` equals the armed step.
fn poison_at(armed: Option<u64>, n: u64) -> bool {
    armed == Some(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fires_once_at_the_armed_step_only() {
        assert!(!poison_at(None, 1));
        assert!(!poison_at(Some(3), 2));
        assert!(poison_at(Some(3), 3));
        assert!(!poison_at(Some(3), 4));
    }

    #[test]
    fn injected_text_is_classified_as_poison() {
        assert!(crate::model_manager::is_gpu_poison_error(
            INJECTED_POISON_TEXT
        ));
    }
}
