//! First GPU-context poisoning under `--supervise` asks the worker to exit with
//! `EXIT_GPU_POISONED` - but only once the server is serving, and never without
//! a supervisor. One test per process: the request is process-global state, so
//! the cases are ordered inside a single test.

use rustedvino::startup::{
    EXIT_GPU_POISONED, SUPERVISED_ENV, gpu_poison_exit_requested, mark_serving, on_gpu_poisoned,
};

#[test]
fn poison_requests_exit_only_when_supervised_and_serving() {
    assert_eq!(EXIT_GPU_POISONED, 75);
    assert!(!gpu_poison_exit_requested());

    // No supervisor parent: stay up (degraded), do not request an exit.
    // SAFETY: this binary has a single test; no other thread reads the environment.
    unsafe { std::env::remove_var(SUPERVISED_ENV) };
    mark_serving();
    on_gpu_poisoned("m", true);
    assert!(
        !gpu_poison_exit_requested(),
        "an unsupervised server must not exit: nothing would restart it"
    );
}
