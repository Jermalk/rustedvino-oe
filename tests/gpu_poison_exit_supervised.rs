//! Companion to `gpu_poison_exit.rs` (separate binary: the state is process-global).
//! Supervised: a poisoning during startup keeps the process up; once serving it
//! requests the exit.

use rustedvino::startup::{
    SUPERVISED_ENV, gpu_poison_exit_requested, mark_serving, on_gpu_poisoned,
};

#[test]
fn supervised_exit_waits_for_serving() {
    // SAFETY: this binary has a single test; no other thread reads the environment.
    unsafe { std::env::set_var(SUPERVISED_ENV, "1") };

    // A preload OOM (before `serve` binds): exiting would loop on the same failure.
    on_gpu_poisoned("m", false);
    assert!(
        !gpu_poison_exit_requested(),
        "a startup poisoning must not exit the worker"
    );

    mark_serving();
    // A model LOAD that wedges the context (a retrying client would repeat it):
    // stay up degraded, do not restart.
    on_gpu_poisoned("m", false);
    assert!(
        !gpu_poison_exit_requested(),
        "a load-time poisoning must not restart the worker"
    );

    // A model that was already serving fails: restart.
    on_gpu_poisoned("m", true);
    assert!(
        gpu_poison_exit_requested(),
        "serving + supervised + inference: exit"
    );
}
