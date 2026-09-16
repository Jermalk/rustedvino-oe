//! Last-resort logging for native (non-Rust) crashes.
//!
//! `OpenVINO`'s C++ runtime can segfault inside a model's dedicated OS thread
//! (2026-07-06 — `internvl2.5-8b-int4-ov` VLM SIGSEGV, see `dev/PROGRESS.md`
//! history). Rust's panic machinery never runs for these: the whole process
//! just vanishes, leaving a bare `segfault at ...` line in `dmesg` and
//! nothing at all in the application log. This installs a handler for the
//! fatal signals that writes a marker plus a raw backtrace to stderr —
//! captured by every launch script's `... > "$LOG_FILE" 2>&1` redirect —
//! before letting the signal finish the crash normally (core dump included),
//! so a crash is diagnosable from the log alone.

use std::os::raw::c_int;

/// Signals that mean the process is already corrupted. Logged, then the
/// default disposition finishes the job — no attempt to continue running;
/// resuming a thread after SIGSEGV/SIGBUS/SIGILL is undefined behaviour.
const FATAL_SIGNALS: [c_int; 4] = [libc::SIGSEGV, libc::SIGABRT, libc::SIGBUS, libc::SIGILL];

/// Bytes reserved for the alternate signal stack the handler runs on. A
/// SIGSEGV from stack overflow means the normal stack is exhausted; without
/// an altstack the handler itself would fault on entry and nothing would be
/// logged. Sized well above the handler's small fixed-size locals (a 64-slot
/// pointer array). Deliberately not `libc::SIGSTKSZ` — recent glibc made that
/// a runtime (non-const) value the compiler can't use as an array length.
const ALT_STACK_SIZE: usize = 64 * 1024;

/// Max stack frames captured for the crash backtrace — a generous fixed
/// bound so the handler never needs to allocate.
const MAX_FRAMES: usize = 64;

/// Installs the crash handler. Call once at startup, before any `OpenVINO`
/// engine thread spawns, so a crash on any of them is caught.
///
/// # Safety rationale
///
/// `sigaltstack`/`sigaction` are raw FFI: the compiler can't verify the
/// handler's signature matches what the kernel will call, or that installing
/// it can't race a signal. Soundness relies on:
///
/// 1. [`handle_fatal_signal`] only calls functions documented as
///    async-signal-safe (`write`, `backtrace`, `backtrace_symbols_fd`) and
///    never allocates, locks, or panics.
/// 2. The one-time `libc::backtrace` warmup call below runs the unwinder's
///    lazy `dlopen("libgcc_s.so.1")` here, outside signal context. glibc does
///    that dlopen — which mallocs and takes locks — on a backtrace's *first*
///    call ever, from any thread. Without this warmup, a crash on a thread
///    that happened to be holding the malloc lock would deadlock inside the
///    handler instead of logging anything.
/// 3. `SA_RESETHAND` resets the disposition to `SIG_DFL` before the handler
///    runs, so the handler returning normally lets the same fault re-execute
///    under the default disposition — producing the normal core-dump /
///    process-death behaviour instead of us reimplementing it, and guarding
///    against a fault inside the handler itself recursing forever.
pub fn install() {
    // Force the unwinder's one-time dlopen now, at a safe point — see safety
    // rationale point 2. The result is discarded; only the side effect
    // (unwinder loaded and ready) matters.
    let mut warmup: [*mut libc::c_void; 1] = [std::ptr::null_mut()];
    unsafe {
        libc::backtrace(warmup.as_mut_ptr(), 1);
    }

    unsafe {
        static mut ALT_STACK: [u8; ALT_STACK_SIZE] = [0; ALT_STACK_SIZE];
        let stack = libc::stack_t {
            ss_sp: std::ptr::addr_of_mut!(ALT_STACK).cast(),
            ss_flags: 0,
            ss_size: ALT_STACK_SIZE,
        };
        if libc::sigaltstack(&raw const stack, std::ptr::null_mut()) != 0 {
            tracing::warn!("sigaltstack failed — crash handler skips the stack-overflow case");
        }

        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = handle_fatal_signal as *const () as usize;
        action.sa_flags = libc::SA_ONSTACK | libc::SA_RESETHAND;
        libc::sigemptyset(&raw mut action.sa_mask);
        for &sig in &FATAL_SIGNALS {
            if libc::sigaction(sig, &raw const action, std::ptr::null_mut()) != 0 {
                tracing::warn!(signal = sig, "failed to install crash handler for signal");
            }
        }
    }
}

/// Signal-safe crash handler: writes the signal name and a raw backtrace to
/// stderr, then returns so the reset-to-default disposition (`SA_RESETHAND`)
/// finishes the crash normally. Must not allocate, lock, or panic — see
/// [`install`]'s safety rationale.
extern "C" fn handle_fatal_signal(sig: c_int) {
    const HEADER: &[u8] = b"\n=== RustedVINO: fatal signal received ===\n";
    unsafe {
        libc::write(libc::STDERR_FILENO, HEADER.as_ptr().cast(), HEADER.len());
        let name = signal_name(sig);
        libc::write(libc::STDERR_FILENO, name.as_ptr().cast(), name.len());

        let mut frames: [*mut libc::c_void; MAX_FRAMES] = [std::ptr::null_mut(); MAX_FRAMES];
        // MAX_FRAMES is a small fixed compile-time constant, always in c_int
        // range — no truncation or sign-wrap is actually possible here.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let count = libc::backtrace(frames.as_mut_ptr(), MAX_FRAMES as c_int);
        libc::backtrace_symbols_fd(frames.as_ptr(), count, libc::STDERR_FILENO);
    }
}

/// Human-readable name for the signals in [`FATAL_SIGNALS`], newline-
/// terminated for a direct `write(2)`.
fn signal_name(sig: c_int) -> &'static str {
    match sig {
        libc::SIGSEGV => "signal 11 (SIGSEGV — segmentation fault)\n",
        libc::SIGABRT => "signal 6 (SIGABRT — abort)\n",
        libc::SIGBUS => "signal 7 (SIGBUS — bus error)\n",
        libc::SIGILL => "signal 4 (SIGILL — illegal instruction)\n",
        _ => "signal (unrecognised)\n",
    }
}
