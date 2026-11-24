//! 0.9.1 S3 (outside review I-05), end to end: the real server binary, a network bind and a
//! keys file with only an admin key. 0.9.0 started and served inference with no key (the gate
//! counted admin keys as auth); now it must refuse before binding the port or loading anything,
//! with `EX_CONFIG` (78) so a supervisor does not respawn it, and say why.
#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn write_pair(dir: &Path, extra: &str, keys: &str) -> std::path::PathBuf {
    let models = dir.join("models");
    let cache = dir.join("cache");
    std::fs::create_dir_all(&models).unwrap();
    std::fs::create_dir_all(&cache).unwrap();
    let config = dir.join("gate.config.json");
    std::fs::write(
        &config,
        format!(
            r#"{{"models_dir": "{}", "ov_cache_dir": "{}", "device": "CPU", "total_vram_gb": 0.0,
               "bind_addr": "0.0.0.0", "port": {}, "models": {{}}{extra}}}"#,
            models.display(),
            cache.display(),
            free_port()
        ),
    )
    .unwrap();
    let keys_path = dir.join("gate.keys.json");
    std::fs::write(&keys_path, keys).unwrap();
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&keys_path, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    config
}

/// Runs the server; returns (exit code, combined output). A server that does not exit by
/// itself within the deadline gets a SIGTERM (never SIGKILL) and the test fails.
fn run_server(config: &Path) -> (Option<i32>, String) {
    let out_path = config.with_extension("out");
    let out = std::fs::File::create(&out_path).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_rustedvino"))
        .env("RUSTEDVINO_CONFIG", config)
        .stdin(Stdio::null())
        .stdout(out.try_clone().unwrap())
        .stderr(out)
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return (status.code(), std::fs::read_to_string(&out_path).unwrap());
        }
        if Instant::now() > deadline {
            let _ = Command::new("kill")
                .args(["-TERM", &child.id().to_string()])
                .status();
            let _ = child.wait();
            panic!(
                "the server did not refuse to start (still running after 60 s):\n{}",
                std::fs::read_to_string(&out_path).unwrap()
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[test]
fn admin_only_keys_on_a_network_bind_exit_78_with_the_reason() {
    let dir = tempfile::tempdir().unwrap();
    let config = write_pair(
        dir.path(),
        "",
        r#"{"api_keys": [], "admin_api_keys": ["sk-test-admin-only"]}"#,
    );
    let (code, out) = run_server(&config);
    assert_eq!(code, Some(78), "expected EX_CONFIG; output:\n{out}");
    assert!(
        out.contains("refusing to start") && out.contains("no api_keys"),
        "the refusal must name the admin-only case; output:\n{out}"
    );
}
