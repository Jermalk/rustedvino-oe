//! D-04 S8: `rv start`/`rv stop` follow the server's log and wait as long as its stop may take.
//! rv cannot depend on this crate (it would link `OpenVINO`), so it carries copies of the log
//! texts and timeouts; this test fails if a copy drifts, and pins the shipped stopper timeouts
//! (`TimeoutStopSec`) to the same bound.
#![allow(clippy::unwrap_used)]

use std::time::Duration;

use rustedvino::startup;

/// Engines the stopper timeouts are sized for (systemd example comment: "with six engines").
const ENGINES: usize = 6;

#[test]
fn rv_follows_the_servers_log_lines() {
    assert!(
        startup::STOP_WAIT_LOG.contains(rv::commands::service::STOP_WAIT_LINE),
        "rv stop would no longer see the stop wait"
    );
    assert_eq!(startup::PRELOAD_LOG, rv::commands::service::PRELOAD_LINE);
}

#[test]
fn rv_waits_as_long_as_the_server_stop() {
    assert_eq!(
        rv::commands::service::UNCAPPED_PRELOAD,
        startup::LOAD_SETTLE_CAP
    );
    assert!(rv::commands::service::STOP_LOAD_TIMEOUT >= startup::serving_stop_bound(ENGINES));
    // No load wait: drain, then the teardown and the exit (Fable S8 review, 2).
    assert!(
        rv::commands::service::STOP_TIMEOUT
            >= startup::stop_remaining(startup::DRAIN_DEADLINE, ENGINES)
    );
}

/// Every `TimeoutStopSec=` value (in seconds) set outside a comment in the files under
/// `scripts/`, by file (the public export ships the example user unit). An unreadable
/// `scripts/` is a test failure, not an empty list (R-15).
fn stop_timeouts_in_scripts() -> Vec<(String, Duration)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts");
    let entries =
        std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        for line in text.lines().filter(|l| !l.trim_start().starts_with('#')) {
            if let Some(v) = line.split("TimeoutStopSec=").nth(1) {
                let secs: u64 = v
                    .chars()
                    .take_while(char::is_ascii_digit)
                    .collect::<String>()
                    .parse()
                    .unwrap();
                out.push((
                    entry.file_name().to_string_lossy().into_owned(),
                    Duration::from_secs(secs),
                ));
            }
        }
    }
    out
}

#[test]
fn shipped_stop_timeouts_cover_the_server_stop() {
    let found = stop_timeouts_in_scripts();
    // R-15: with nothing found the loop below checks nothing and the test would pass vacuously.
    assert!(
        found
            .iter()
            .any(|(file, _)| file == "rustedvino.user.service.example"),
        "no TimeoutStopSec= found in scripts/rustedvino.user.service.example (found: {found:?})"
    );
    for (file, t) in found {
        assert!(
            t >= startup::serving_stop_bound(ENGINES),
            "scripts/{file}: TimeoutStopSec {t:?} < the server's stop bound"
        );
        assert_eq!(
            t,
            rv::commands::service::STOP_LOAD_TIMEOUT,
            "scripts/{file}"
        );
    }
}
