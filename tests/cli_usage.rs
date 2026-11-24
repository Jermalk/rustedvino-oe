//! FM-1: the real binary prints usage for `--help` / `-h` (exit 0) and refuses an unknown
//! argument with the usage on stderr (exit 64). Both exit before any config, runtime or device
//! work, so no server starts; a deadline guards against one starting anyway.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

fn run(args: &[&str]) -> Output {
    let tmp = tempfile::tempdir().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_rustedvino"))
        .args(args)
        // A config that does not exist: had the server started, it would fail at once.
        .env("RUSTEDVINO_CONFIG", tmp.path().join("missing.json"))
        .env("HOME", tmp.path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("rustedvino {args:?} did not exit within 30 s");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    child.wait_with_output().unwrap()
}

#[test]
fn help_prints_usage_and_exits_0() {
    for flag in ["--help", "-h"] {
        let out = run(&[flag]);
        assert_eq!(out.status.code(), Some(0), "{flag}");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("Usage: rustedvino"), "{flag}: {stdout}");
        assert!(stdout.contains("RUSTEDVINO_CONFIG"), "{flag}: {stdout}");
    }
}

#[test]
fn unknown_argument_prints_usage_to_stderr_and_exits_64() {
    for args in [&["--bogus"][..], &["--config", "x.json"], &["/?"]] {
        let out = run(args);
        assert_eq!(out.status.code(), Some(64), "{args:?}");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("unknown argument"), "{args:?}: {stderr}");
        assert!(stderr.contains("Usage: rustedvino"), "{args:?}: {stderr}");
        assert!(out.stdout.is_empty(), "{args:?}: usage goes to stderr");
    }
}
