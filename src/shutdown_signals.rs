//! Shutdown signals (SIGTERM / SIGINT) taken on one dedicated thread.
//!
//! Why: `libigc.so.2` / `libigdrcl.so` install their own process-wide signal
//! handlers, at least transiently. With tokio's ordinary handler, a SIGTERM
//! that lands while one of those handlers is installed runs *their* handler
//! on whatever thread the kernel picked (the main thread, parked in tokio's
//! `block_on`), which `longjmp`s on a stale `jmp_buf` and aborts the process
//! (`*** longjmp causes uninitialized stack frame ***: terminated`; reproduced
//! under gdb).
//!
//! A signal that is blocked on every thread and consumed by `sigwait` never
//! runs a handler at all, so nobody's handler can be tricked into running on
//! the wrong thread. [`install`] blocks the signals on the calling thread
//! (and so on every thread it spawns later) and starts the waiter.
//!
//! Call order matters: [`install`] must run on the main thread BEFORE the
//! tokio runtime or any other thread exists, because only threads created
//! afterwards inherit the mask. Do not call it in `--supervise` mode - the
//! supervisor is a plain tokio process and relies on tokio's own handler.
//! Child processes are not affected: `std::process::Command` resets the
//! signal mask in the child (checked by a test below).

use std::io;
use std::os::raw::c_int;

/// The signals that start a graceful shutdown.
const SHUTDOWN_SIGNALS: [c_int; 2] = [libc::SIGTERM, libc::SIGINT];

fn sigset_of(signals: &[c_int]) -> libc::sigset_t {
    // Safety: `sigset_t` is plain data; `sigemptyset` initialises it fully
    // before any use, and `sigaddset` only fails for an invalid signal number
    // (every number passed here is a constant valid signal).
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&raw mut set);
        for &sig in signals {
            libc::sigaddset(&raw mut set, sig);
        }
        set
    }
}

/// Blocks (`block == true`) or unblocks `signals` on the calling thread.
fn set_blocked(signals: &[c_int], block: bool) -> io::Result<()> {
    let set = sigset_of(signals);
    let how = if block {
        libc::SIG_BLOCK
    } else {
        libc::SIG_UNBLOCK
    };
    // Safety: `set` is a valid, initialised sigset and the old-mask pointer
    // may be null. `pthread_sigmask` only changes the calling thread's mask.
    let rc = unsafe { libc::pthread_sigmask(how, &raw const set, std::ptr::null_mut()) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::from_raw_os_error(rc))
    }
}

/// Waits for any of `signals` (which MUST be blocked on this thread) and calls
/// `on_signal` for each one received, until it returns `false` or `sigwait`
/// fails. Returns the `sigwait` error code, if that is what ended the loop.
fn wait_loop(signals: &[c_int], mut on_signal: impl FnMut(c_int) -> bool) -> Option<c_int> {
    let set = sigset_of(signals);
    loop {
        let mut sig: c_int = 0;
        // Safety: `set` is valid and initialised, `sig` is a valid out
        // pointer. `sigwait` returns the error number directly (not via
        // `errno`) and has no other side effects.
        let rc = unsafe { libc::sigwait(&raw const set, &raw mut sig) };
        if rc != 0 {
            return Some(rc);
        }
        if !on_signal(sig) {
            return None;
        }
    }
}

/// Blocks SIGTERM/SIGINT on the calling thread and starts the thread that
/// turns them into a shutdown request ([`crate::startup::request_shutdown`]).
///
/// On failure the calling thread's mask is restored, so the caller can carry
/// on with tokio's normal signal handling.
///
/// # Errors
///
/// Returns an error if the mask cannot be changed or the thread cannot be
/// spawned.
pub fn install() -> io::Result<()> {
    set_blocked(&SHUTDOWN_SIGNALS, true)?;
    let spawned = std::thread::Builder::new()
        .name("shutdown-sigwait".into())
        .spawn(|| {
            // The mask was inherited from the spawning thread, which is what
            // `sigwait` requires.
            let mut seen = false;
            let failure = wait_loop(&SHUTDOWN_SIGNALS, |sig| {
                if seen {
                    tracing::info!(signal = sig, "shutdown already in progress - ignoring");
                } else {
                    seen = true;
                    tracing::info!(signal = sig, "shutdown signal received");
                    crate::startup::request_shutdown();
                }
                true
            });
            if let Some(code) = failure {
                // Nobody else could take the signals, so give them back to
                // the process: with them unblocked here, tokio's handler (or
                // the default action) applies again instead of the process
                // becoming unstoppable by SIGTERM.
                eprintln!("shutdown-sigwait: sigwait failed ({code}) - unblocking signals");
                let _ = set_blocked(&SHUTDOWN_SIGNALS, false);
            }
        });
    if let Err(e) = spawned {
        let _ = set_blocked(&SHUTDOWN_SIGNALS, false);
        return Err(e);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    /// A signal sent to a thread that blocks it is consumed by `sigwait` and
    /// never runs a handler. SIGUSR1 stands in for SIGTERM so the test cannot
    /// disturb the harness; `pthread_kill` targets only the waiter thread,
    /// which blocks it, so no other thread can take the default action.
    #[test]
    fn blocked_signal_is_consumed_by_sigwait() {
        let (tx, rx) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            set_blocked(&[libc::SIGUSR1], true).unwrap();
            // Safety: `pthread_self` has no preconditions.
            tx.send(unsafe { libc::pthread_self() }).unwrap();
            let mut got = Vec::new();
            let failure = wait_loop(&[libc::SIGUSR1], |sig| {
                got.push(sig);
                false
            });
            (failure, got)
        });
        let tid = rx.recv().unwrap();
        // Safety: `tid` is the live waiter thread (it is blocked in sigwait
        // until this signal arrives), and SIGUSR1 is blocked on it.
        assert_eq!(unsafe { libc::pthread_kill(tid, libc::SIGUSR1) }, 0);
        let (failure, got) = waiter.join().unwrap();
        assert_eq!(failure, None);
        assert_eq!(got, vec![libc::SIGUSR1]);
    }

    /// Children must not inherit the blocked mask, or a spawned helper
    /// (espeak-ng) could not be stopped with SIGTERM.
    #[test]
    fn child_processes_do_not_inherit_the_mask() {
        let out = std::thread::spawn(|| {
            set_blocked(&SHUTDOWN_SIGNALS, true).unwrap();
            std::process::Command::new("sh")
                .args(["-c", "grep '^SigBlk' /proc/self/status"])
                .output()
                .unwrap()
        })
        .join()
        .unwrap();
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        assert!(
            text.trim().ends_with("0000000000000000"),
            "child inherited a blocked-signal mask: {text}"
        );
    }
}
