//! Verb-level tests: the real `rv` binary against a path-routed stub of the
//! admin API. They pin exit codes (0/1/2), the `--json` shapes, request routing
//! and the server-reported-PID path — the behaviour the unit tests cannot see.

#![allow(clippy::expect_used)] // test harness: a failed setup should fail loudly

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

/// Routes are `("METHOD /path", status, body)`; anything else is a 404.
type Routes = Vec<(&'static str, u16, &'static str)>;

struct Stub {
    port: u16,
    hits: Arc<Mutex<Vec<String>>>,
}

fn serve(routes: Routes) -> Stub {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let hits = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&hits);
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(sock) = conn else { continue };
            let routes = routes.clone();
            let log = Arc::clone(&log);
            std::thread::spawn(move || {
                let mut rd = BufReader::new(sock.try_clone().expect("clone"));
                let mut first = String::new();
                if rd.read_line(&mut first).is_err() {
                    return;
                }
                let mut len = 0usize;
                loop {
                    let mut h = String::new();
                    if rd.read_line(&mut h).unwrap_or(0) == 0 || h == "\r\n" {
                        break;
                    }
                    if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
                        len = v.trim().parse().unwrap_or(0);
                    }
                }
                let mut body = vec![0u8; len];
                let _ = rd.read_exact(&mut body);
                let mut parts = first.split_whitespace();
                let key = format!(
                    "{} {}",
                    parts.next().unwrap_or(""),
                    parts.next().unwrap_or("").split('?').next().unwrap_or("")
                );
                log.lock().expect("lock").push(key.clone());
                let (status, text) = routes
                    .iter()
                    .find(|(k, _, _)| *k == key)
                    .map_or((404, "not routed"), |(_, s, b)| (*s, *b));
                let mut out = sock;
                let _ = write!(
                    out,
                    "HTTP/1.1 {status} X\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\
                     Connection: close\r\n\r\n{text}",
                    text.len()
                );
            });
        }
    });
    Stub { port, hits }
}

struct Run {
    code: i32,
    out: String,
    err: String,
}

/// Runs `rv` with a config pointing at `port`; `admin_key` fills the keys file.
fn rv(port: u16, admin_key: Option<&str>, args: &[&str]) -> Run {
    let d = tempfile::tempdir().expect("tempdir");
    let cfg = d.path().join("c.config.json");
    std::fs::write(&cfg, format!(r#"{{"port": {port}}}"#)).expect("cfg");
    let keys = match admin_key {
        Some(k) => format!(r#"{{"api_keys": [], "admin_api_keys": ["{k}"]}}"#),
        None => r#"{"api_keys": [], "admin_api_keys": []}"#.to_owned(),
    };
    std::fs::write(d.path().join("c.keys.json"), keys).expect("keys");
    let o = Command::new(env!("CARGO_BIN_EXE_rv"))
        .args(args)
        .env("RUSTEDVINO_CONFIG", &cfg)
        .env_remove("RV_ADMIN_KEY")
        .env_remove("RV_PORT")
        .env_remove("NO_COLOR")
        .stdin(Stdio::null())
        .output()
        .expect("run rv");
    Run {
        code: o.status.code().unwrap_or(-1),
        out: String::from_utf8_lossy(&o.stdout).into_owned(),
        err: String::from_utf8_lossy(&o.stderr).into_owned(),
    }
}

const FITS: &str =
    r#"{"fits":true,"needed_gb":4.5,"free_gb":27.0,"would_evict":[],"already_loaded":false}"#;
const NO_FIT: &str =
    r#"{"fits":false,"needed_gb":40.0,"free_gb":3.0,"would_evict":[],"already_loaded":false}"#;
const EVICTS: &str =
    r#"{"fits":true,"needed_gb":9.0,"free_gb":2.0,"would_evict":["other"],"already_loaded":false}"#;
const LOADED: &str =
    r#"{"fits":true,"needed_gb":1.0,"free_gb":9.0,"would_evict":[],"already_loaded":true}"#;

#[test]
fn check_exit_codes_follow_whether_the_model_fits() {
    let s = serve(vec![
        ("POST /v1/admin/models/a/check", 200, FITS),
        ("POST /v1/admin/models/b/check", 200, NO_FIT),
        ("POST /v1/admin/models/c/check", 200, LOADED),
    ]);
    let a = rv(s.port, Some("adm"), &["check", "a"]);
    assert_eq!(a.code, 0, "{}", a.err);
    assert!(a.out.contains("needs 4.5 GB, 27.0 GB free"), "{}", a.out);
    let b = rv(s.port, Some("adm"), &["check", "b"]);
    assert_eq!(b.code, 1);
    assert!(
        b.out.contains("does not fit even after evicting"),
        "{}",
        b.out
    );
    assert_eq!(rv(s.port, Some("adm"), &["check", "c"]).code, 0);
    let missing = rv(s.port, Some("adm"), &["check", "nope"]);
    assert_eq!(missing.code, 1);
    assert!(missing.err.contains("error: not routed"), "{}", missing.err);
}

#[test]
fn check_json_is_the_servers_object_unchanged() {
    let s = serve(vec![("POST /v1/admin/models/a/check", 200, FITS)]);
    let r = rv(s.port, Some("adm"), &["check", "a", "--json"]);
    assert_eq!(r.code, 0);
    let got: serde_json::Value = serde_json::from_str(&r.out).expect("json");
    assert_eq!(
        got,
        serde_json::from_str::<serde_json::Value>(FITS).expect("json")
    );
}

#[test]
fn admin_verbs_without_a_key_refuse_with_exit_1_and_send_nothing() {
    let s = serve(vec![]);
    let r = rv(s.port, None, &["ps"]);
    assert_eq!(r.code, 1);
    assert!(r.err.contains("no admin key available"), "{}", r.err);
    assert!(s.hits.lock().expect("lock").is_empty());
}

#[test]
fn destructive_verbs_refuse_without_yes_on_a_non_tty_with_exit_2() {
    let s = serve(vec![]);
    for args in [
        &["deregister", "m"][..],
        &["resize", "m", "--kv-cache-gb", "2"],
        &["preload", "m"],
        &["session-kill", "123e4567-e89b-12d3-a456-426614174000"],
    ] {
        let r = rv(s.port, Some("adm"), args);
        assert_eq!(r.code, 2, "{args:?}: {}", r.err);
        assert!(
            r.err.contains("refusing without --yes"),
            "{args:?}: {}",
            r.err
        );
    }
    assert!(
        s.hits.lock().expect("lock").is_empty(),
        "nothing may be sent before consent"
    );
}

#[test]
fn load_checks_first_and_only_loads_when_it_fits() {
    let s = serve(vec![
        ("POST /v1/admin/models/a/check", 200, FITS),
        ("POST /v1/admin/models/a/load", 200, "model 'a' loaded"),
        ("POST /v1/admin/models/big/check", 200, NO_FIT),
        ("POST /v1/admin/models/ev/check", 200, EVICTS),
        ("POST /v1/admin/models/ev/load", 200, "ok"),
        ("POST /v1/admin/models/up/check", 200, LOADED),
    ]);
    let ok = rv(s.port, Some("adm"), &["load", "a"]);
    assert_eq!(ok.code, 0, "{}", ok.err);
    assert!(ok.out.contains("model 'a' loaded"), "{}", ok.out);

    let big = rv(s.port, Some("adm"), &["load", "big"]);
    assert_eq!(big.code, 1);
    assert!(big.out.contains("does not fit"), "{}", big.out);

    // An evicting load needs consent: refused (2) without -y, proceeds with it.
    assert_eq!(rv(s.port, Some("adm"), &["load", "ev"]).code, 2);
    assert_eq!(rv(s.port, Some("adm"), &["load", "ev", "-y"]).code, 0);

    let up = rv(s.port, Some("adm"), &["load", "up"]);
    assert_eq!(up.code, 0);
    assert!(up.out.contains("already loaded"), "{}", up.out);

    let hits = s.hits.lock().expect("lock").clone();
    assert!(
        !hits.contains(&"POST /v1/admin/models/big/load".to_owned()),
        "{hits:?}"
    );
    assert!(
        !hits.contains(&"POST /v1/admin/models/up/load".to_owned()),
        "{hits:?}"
    );
    let nope = rv(s.port, Some("adm"), &["load", "ghost"]);
    assert_eq!(nope.code, 1);
    assert!(nope.err.contains("is not registered"), "{}", nope.err);
}

#[test]
fn patch_flag_validation_happens_before_any_request() {
    let s = serve(vec![]);
    let none = rv(s.port, Some("adm"), &["patch", "m"]);
    assert_eq!(none.code, 1);
    assert!(none.err.contains("no fields supplied"), "{}", none.err);
    let both = rv(
        s.port,
        Some("adm"),
        &["patch", "m", "--device", "GPU", "--clear-device"],
    );
    assert_eq!(both.code, 1);
    assert!(both.err.contains("mutually exclusive"), "{}", both.err);
    assert!(s.hits.lock().expect("lock").is_empty());
}

const ADMIN_MODELS: &str = r#"{"models":[
  {"id":"m1","state":"ready","device":"GPU","kind":"text_gen","vram_gb":5.0,"cache_size_gb":2.0},
  {"id":"m2","state":"not_loaded","device":"CPU","kind":"embedding","vram_gb":1.0,"cache_size_gb":0.0}],
  "server":{"pid":4242424,"uptime_secs":3700,"engine":"RustedVINO"}}"#;
const METRICS: &str = "rustedvino_kv_cache_usage_percent{model=\"m1\"} 12.5\n\
rustedvino_requests_running{model=\"m1\"} 2\nrustedvino_requests_waiting{model=\"m1\"} 0\n\
rustedvino_requests_max{model=\"m1\"} 16\n";

#[test]
fn ps_lists_only_ready_models_with_live_load() {
    let s = serve(vec![
        ("GET /v1/admin/models", 200, ADMIN_MODELS),
        ("GET /metrics", 200, METRICS),
    ]);
    let r = rv(s.port, Some("adm"), &["ps"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(r.out.contains("1 model(s) loaded"), "{}", r.out);
    assert!(r.out.contains("m1") && !r.out.contains("m2"), "{}", r.out);
    assert!(r.out.contains("2/0/16"), "{}", r.out);
    let j = rv(s.port, Some("adm"), &["ps", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&j.out).expect("json");
    assert_eq!(v.as_array().map(Vec::len), Some(1));
}

fn status_routes() -> Routes {
    vec![
        ("GET /health", 200, r#"{"status":"ok","version":"0.8.0"}"#),
        ("GET /v1/models", 200, r#"{"data":[{"id":"m1"}]}"#),
        ("GET /v1/admin/models", 200, ADMIN_MODELS),
        (
            "GET /v1/admin/config/audit",
            200,
            r#"{"problems":[],"total_declared":2}"#,
        ),
        (
            "GET /v1/admin/health/watchdog",
            200,
            r#"{"oldest_generation_secs":null}"#,
        ),
        ("GET /metrics", 200, METRICS),
    ]
}

#[test]
fn status_takes_the_pid_from_the_server_and_says_so() {
    let s = serve(status_routes());
    let r = rv(s.port, Some("adm"), &["status"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(r.out.contains("pid 4242424"), "{}", r.out);
    assert!(r.out.contains("reported by the server itself"), "{}", r.out);
    assert!(r.out.contains("1:01:40"), "uptime 3700s: {}", r.out);
    let j = rv(s.port, Some("adm"), &["status", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&j.out).expect("json");
    assert_eq!(v["process"]["alive"], true);
    assert_eq!(v["process"]["supervisor_pid"], 4_242_424);
    assert_eq!(v["health"]["version"], "0.8.0");
}

#[test]
fn status_exit_code_follows_health() {
    let mut routes = status_routes();
    routes[0] = ("GET /health", 500, "boom");
    let s = serve(routes);
    assert_eq!(rv(s.port, Some("adm"), &["status"]).code, 1);
}

#[test]
fn completeness_exits_1_on_a_blocking_gap() {
    let ok = serve(vec![(
        "GET /v1/admin/models/completeness",
        200,
        r#"{"models":[{"id":"a","kind":"llm","complete":true}],"incomplete_count":0}"#,
    )]);
    assert_eq!(rv(ok.port, Some("adm"), &["completeness"]).code, 0);
    let bad = serve(vec![(
        "GET /v1/admin/models/completeness",
        200,
        r#"{"models":[{"id":"a","kind":"llm","complete":false,"blocking_missing":["x.bin"]}],"incomplete_count":1}"#,
    )]);
    let r = rv(bad.port, Some("adm"), &["completeness"]);
    assert_eq!(r.code, 1);
    assert!(
        r.out.contains("INCOMPLETE") && r.out.contains("x.bin"),
        "{}",
        r.out
    );
}

#[test]
fn bearer_key_is_sent_to_admin_routes_and_never_printed() {
    // The stub cannot see headers in this harness, so check the other half:
    // the key must not appear in anything rv prints, success or failure.
    let s = serve(status_routes());
    for args in [&["status"][..], &["ps"], &["list"], &["resolve"]] {
        let r = rv(s.port, Some("sekrit-admin-key"), args);
        assert!(!r.out.contains("sekrit-admin-key"), "{args:?}: {}", r.out);
        assert!(!r.err.contains("sekrit-admin-key"), "{args:?}: {}", r.err);
    }
}
