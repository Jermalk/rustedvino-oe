//! Verb-level tests: the real `rv` binary against a path-routed stub of the
//! admin API. They pin exit codes (0/1/2), the `--json` shapes, request routing
//! and the server-reported-PID path — the behaviour the unit tests cannot see.

#![allow(clippy::expect_used)] // test harness: a failed setup should fail loudly
#![allow(clippy::many_single_char_names)] // one short-lived binding per rv run

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

/// A request log shared with the stub's threads.
type Log<T> = Arc<Mutex<Vec<T>>>;

/// Routes are `("METHOD /path", status, body)`; anything else is a 404.
type Routes = Vec<(&'static str, u16, &'static str)>;

struct Stub {
    port: u16,
    hits: Arc<Mutex<Vec<String>>>,
    /// Every request as `(METHOD /path?query, body)`: what `hits` leaves out.
    requests: Arc<Mutex<Vec<(String, String)>>>,
    /// Whether each request (arrival order) carried an `Authorization` header.
    authed: Arc<Mutex<Vec<bool>>>,
    /// Every request as `(METHOD /path?query, its Accept header)` (V1-rv P1).
    accepts: Log<(String, Option<String>)>,
}

fn serve(routes: Routes) -> Stub {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let hits = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&hits);
    let requests = Arc::new(Mutex::new(Vec::new()));
    let full_log = Arc::clone(&requests);
    let authed = Arc::new(Mutex::new(Vec::new()));
    let auth_log = Arc::clone(&authed);
    let accepts = Arc::new(Mutex::new(Vec::new()));
    let accept_log = Arc::clone(&accepts);
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(sock) = conn else { continue };
            let routes = routes.clone();
            let log = Arc::clone(&log);
            let full_log = Arc::clone(&full_log);
            let auth_log = Arc::clone(&auth_log);
            let accept_log = Arc::clone(&accept_log);
            std::thread::spawn(move || {
                let mut rd = BufReader::new(sock.try_clone().expect("clone"));
                let mut first = String::new();
                if rd.read_line(&mut first).is_err() {
                    return;
                }
                let mut len = 0usize;
                let mut has_auth = false;
                let mut accept = None;
                loop {
                    let mut h = String::new();
                    if rd.read_line(&mut h).unwrap_or(0) == 0 || h == "\r\n" {
                        break;
                    }
                    if h.to_ascii_lowercase().starts_with("authorization:") {
                        has_auth = true;
                    }
                    if h.to_ascii_lowercase().starts_with("accept:") {
                        accept = Some(h["accept:".len()..].trim().to_owned());
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
                auth_log.lock().expect("lock").push(has_auth);
                accept_log.lock().expect("lock").push((
                    first.split_whitespace().nth(1).unwrap_or("").to_owned(),
                    accept,
                ));
                log.lock().expect("lock").push(key.clone());
                full_log.lock().expect("lock").push((
                    format!(
                        "{} {}",
                        first.split_whitespace().next().unwrap_or(""),
                        first.split_whitespace().nth(1).unwrap_or("")
                    ),
                    String::from_utf8_lossy(&body).into_owned(),
                ));
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
    Stub {
        port,
        hits,
        requests,
        authed,
        accepts,
    }
}

struct Run {
    code: i32,
    out: String,
    err: String,
}

/// Runs `rv` with a config pointing at `port`; `admin_key` fills the keys file
/// (`None` = a keys file with empty lists, i.e. an OPEN server).
fn rv(port: u16, admin_key: Option<&str>, args: &[&str]) -> Run {
    let keys = match admin_key {
        Some(k) => format!(r#"{{"api_keys": [], "admin_api_keys": ["{k}"]}}"#),
        None => r#"{"api_keys": [], "admin_api_keys": []}"#.to_owned(),
    };
    rv_keys_file(port, Some(&keys), args)
}

/// Like [`rv`] but with the keys file's exact text (`None` = no keys file at all).
fn rv_keys_file(port: u16, keys: Option<&str>, args: &[&str]) -> Run {
    let d = tempfile::tempdir().expect("tempdir");
    let cfg = d.path().join("c.config.json");
    std::fs::write(&cfg, format!(r#"{{"port": {port}}}"#)).expect("cfg");
    if let Some(keys) = keys {
        std::fs::write(d.path().join("c.keys.json"), keys).expect("keys");
    }
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

// ---- RVR: the does-not-fit verdict names the refusing gate and a proven config cap ----

/// The live record (2026-11-21, 30 GB UMA laptop): the 35B's check with nothing loaded.
const LIVE_35B_CHECK: &str = r#"{"fits":false,"needed_gb":22.0,"free_gb":18.0,"vram_gated":true,"would_evict":[],"busy_victims":[],"blocked_by_busy":[],"ram_needed_gb":24.0,"ram_avail_gb":17.0,"already_loaded":false,"headroom_refusal":null,"headroom_not_previewed":null}"#;
const LIVE_35B_MODELS: &str = r#"{"models":[{"id":"qwen3.6-35b-a3b-int4-ov","state":"not_loaded"},{"id":"bielik","state":"not_loaded"}]}"#;

/// Like [`rv`] with an admin key, but the config also holds `extra`'s keys (the memory caps).
fn rv_config(port: u16, extra: &serde_json::Value, args: &[&str]) -> Run {
    let d = tempfile::tempdir().expect("tempdir");
    let cfg = d.path().join("c.config.json");
    let mut c = extra.clone();
    c["port"] = serde_json::json!(port);
    std::fs::write(&cfg, c.to_string()).expect("cfg");
    std::fs::write(
        d.path().join("c.keys.json"),
        r#"{"api_keys": [], "admin_api_keys": ["adm"]}"#,
    )
    .expect("keys");
    let o = Command::new(env!("CARGO_BIN_EXE_rv"))
        .args(args)
        .env("RUSTEDVINO_CONFIG", &cfg)
        .env_remove("RV_ADMIN_KEY")
        .env_remove("RV_PORT")
        .env("NO_COLOR", "1")
        .stdin(Stdio::null())
        .output()
        .expect("run rv");
    Run {
        code: o.status.code().unwrap_or(-1),
        out: String::from_utf8_lossy(&o.stdout).into_owned(),
        err: String::from_utf8_lossy(&o.stderr).into_owned(),
    }
}

#[test]
fn live_35b_refusal_names_both_gates_and_the_config_caps() {
    let id = "qwen3.6-35b-a3b-int4-ov";
    let s = serve(vec![
        (
            "POST /v1/admin/models/qwen3.6-35b-a3b-int4-ov/check",
            200,
            LIVE_35B_CHECK,
        ),
        ("GET /v1/admin/models", 200, LIVE_35B_MODELS),
    ]);
    let caps = serde_json::json!({"total_vram_gb": 18.0, "system_ram_budget_gb": 17.0,
                                  "system_ram_reservation_gb": 2.0});

    let load = rv_config(s.port, &caps, &["load", id]);
    assert_eq!(load.code, 1, "{}{}", load.out, load.err);
    // `rv load`'s own blank spacer lines are not part of the verdict.
    let lines: Vec<&str> = load.out.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(
        lines,
        vec![
            "● 'qwen3.6-35b-a3b-int4-ov' does not fit (GPU budget and system RAM) — nothing is \
             loaded to evict",
            "  GPU budget: needs 22.0 GB, 18.0 GB free · RAM: needs 24.0 GB + 2.0 GB reserve, \
             17.0 GB available",
            "  limit: config.json sets total_vram_gb 18.0 / system_ram_budget_gb 17.0 — raise to \
             load this model",
        ],
        "{}",
        load.out
    );
    assert!(!load.out.contains("pinned"), "{}", load.out);
    // The verdict stops before any load is sent.
    assert!(
        !s.hits
            .lock()
            .expect("lock")
            .iter()
            .any(|h| h.ends_with("/load")),
        "no load may be sent"
    );

    let check = rv_config(s.port, &caps, &["check", id]);
    assert_eq!(check.code, 1, "{}{}", check.out, check.err);
    for want in [
        "  system RAM: needs 24.0 GB + 2.0 GB reserve, 17.0 GB available",
        "  refused by: GPU budget and system RAM — nothing is loaded to evict",
        "  limit: config.json sets total_vram_gb 18.0 / system_ram_budget_gb 17.0 — raise to \
         load this model",
    ] {
        assert!(
            check.out.contains(want),
            "missing {want:?} in:\n{}",
            check.out
        );
    }
    assert!(!check.out.contains("even after evicting"), "{}", check.out);

    // `--json` is the server's object, unchanged, and asks nothing more.
    let before = s.hits.lock().expect("lock").len();
    let j = rv_config(s.port, &caps, &["check", id, "--json"]);
    assert_eq!(j.code, 0);
    let got: serde_json::Value = serde_json::from_str(&j.out).expect("json");
    assert_eq!(
        got,
        serde_json::from_str::<serde_json::Value>(LIVE_35B_CHECK).expect("json")
    );
    assert_eq!(
        s.hits.lock().expect("lock").len(),
        before + 1,
        "one POST only"
    );
}

/// The coordinator's live check (2026-11-21): five models loaded; on a refusal `would_evict` is
/// the server's whole sweep, which still does not make room.
const LIVE_LOADED_35B_CHECK: &str = r#"{"fits":false,"needed_gb":22.0,"free_gb":1.0068,"vram_gated":true,"would_evict":["multilingual-e5-large-int8","bge-reranker-v2-m3-int8-ov","whisper-large-v3-turbo-int8-ov","speecht5-tts-ov","qwen3.5-9B-int4-ov"],"busy_victims":[],"blocked_by_busy":[],"ram_needed_gb":24.0,"ram_avail_gb":10.96,"already_loaded":false}"#;
const LIVE_LOADED_MODELS: &str = r#"{"models":[{"id":"qwen3.6-35b-a3b-int4-ov","state":"not_loaded"},{"id":"multilingual-e5-large-int8","state":"ready"},{"id":"bge-reranker-v2-m3-int8-ov","state":"ready"},{"id":"whisper-large-v3-turbo-int8-ov","state":"ready"},{"id":"speecht5-tts-ov","state":"ready"},{"id":"qwen3.5-9B-int4-ov","state":"ready"}]}"#;

#[test]
fn live_loaded_35b_refusal_names_the_sweep_and_the_proven_caps() {
    let id = "qwen3.6-35b-a3b-int4-ov";
    let s = serve(vec![
        (
            "POST /v1/admin/models/qwen3.6-35b-a3b-int4-ov/check",
            200,
            LIVE_LOADED_35B_CHECK,
        ),
        ("GET /v1/admin/models", 200, LIVE_LOADED_MODELS),
    ]);
    let caps = serde_json::json!({"total_vram_gb": 18.0, "system_ram_budget_gb": 17.0,
                                  "system_ram_reservation_gb": 2.0, "domain_budgets": {}});

    let load = rv_config(s.port, &caps, &["load", id]);
    assert_eq!(load.code, 1, "{}{}", load.out, load.err);
    let lines: Vec<&str> = load.out.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(
        lines,
        vec![
            "● 'qwen3.6-35b-a3b-int4-ov' does not fit (GPU budget and system RAM) — even after \
             evicting all 5 evictable models",
            "  GPU budget: needs 22.0 GB, 1.0 GB free · RAM: needs 24.0 GB + 2.0 GB reserve, \
             11.0 GB available",
            "  limit: config.json sets total_vram_gb 18.0 / system_ram_budget_gb 17.0 — raise to \
             load this model",
        ],
        "{}",
        load.out
    );
    assert!(
        !s.hits
            .lock()
            .expect("lock")
            .iter()
            .any(|h| h.ends_with("/load")),
        "no load may be sent"
    );

    let check = rv_config(s.port, &caps, &["check", id]);
    assert_eq!(check.code, 1, "{}{}", check.out, check.err);
    assert!(!check.out.contains("would evict:"), "{}", check.out);
    for want in [
        "  system RAM: needs 24.0 GB + 2.0 GB reserve, 11.0 GB available",
        "  refused by: GPU budget and system RAM — even after evicting all 5 evictable models",
        "  limit: config.json sets total_vram_gb 18.0 / system_ram_budget_gb 17.0 — raise to \
         load this model",
    ] {
        assert!(
            check.out.contains(want),
            "missing {want:?} in:\n{}",
            check.out
        );
    }
    assert!(
        !check.out.contains("the budget comes from"),
        "{}",
        check.out
    );
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
fn admin_verbs_without_a_usable_key_refuse_with_exit_1_and_send_nothing() {
    // Not open, and no admin key: no keys file at all, an unreadable one, or one that holds only an
    // inference key. rv must refuse each and send nothing.
    for (case, keys) in [
        ("no keys file", None),
        ("malformed keys file", Some("{ not json")),
        (
            "inference key only",
            Some(r#"{"api_keys": ["k"], "admin_api_keys": []}"#),
        ),
    ] {
        let s = serve(vec![]);
        let r = rv_keys_file(s.port, keys, &["ps"]);
        assert_eq!(r.code, 1, "{case}");
        assert!(
            r.err.contains("no admin key available"),
            "{case}: {}",
            r.err
        );
        assert!(s.hits.lock().expect("lock").is_empty(), "{case}");
    }
}

#[test]
fn empty_keys_file_means_an_open_server_so_rv_proceeds_unauthenticated_and_says_so() {
    let s = serve(vec![
        ("GET /v1/admin/models", 200, ADMIN_MODELS),
        ("GET /metrics", 200, METRICS),
    ]);
    // `rv(.., None, ..)` writes {"api_keys": [], "admin_api_keys": []}.
    let r = rv(s.port, None, &["ps"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(r.out.contains("1 model(s) loaded"), "{}", r.out);
    assert!(
        r.err.contains("OPEN") && r.err.contains("NOT SECURE"),
        "the open state must be stated plainly: {}",
        r.err
    );
    assert!(
        s.authed.lock().expect("lock").iter().all(|a| !a),
        "an open server must get no Authorization header"
    );
    // `{}` is the README's "minimal" keys file: same state.
    let r = rv_keys_file(s.port, Some("{}"), &["ps"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(r.err.contains("NOT SECURE"), "{}", r.err);
}

#[test]
fn a_real_admin_key_is_sent_and_no_open_server_warning_appears() {
    let s = serve(vec![
        ("GET /v1/admin/models", 200, ADMIN_MODELS),
        ("GET /metrics", 200, METRICS),
    ]);
    let r = rv(s.port, Some("adm"), &["ps"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(!r.err.contains("NOT SECURE"), "{}", r.err);
    assert!(
        s.authed.lock().expect("lock").iter().any(|a| *a),
        "the admin key must be sent as a bearer"
    );
}

#[test]
fn version_reports_the_answering_server_from_health() {
    let s = serve(vec![(
        "GET /health",
        200,
        r#"{"status":"ok","version":"9.9.9","git_hash":"abc123"}"#,
    )]);
    let r = rv(s.port, None, &["version"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(r.out.contains("rustedvino 9.9.9 (abc123)"), "{}", r.out);
    assert!(r.out.contains(&s.port.to_string()), "{}", r.out);
}

#[test]
fn start_refuses_a_port_the_server_would_not_listen_on() {
    // rv_keys_file writes a config with "port": 1; --port 9000 steers rv only.
    let r = rv(1, None, &["start", "--port", "9000"]);
    assert_eq!(r.code, 1, "{}", r.err);
    assert!(
        r.err.starts_with("error: port 9000 was requested"),
        "{}",
        r.err
    );
    assert!(
        r.err.contains("  = help:"),
        "rustc-style help line: {}",
        r.err
    );
}

#[test]
fn stop_with_nothing_on_the_port_says_so_and_touches_no_other_server() {
    // Port 1 is never a rustedvino. Even if another server runs on this
    // machine, `stop` must not pick it up through a command-line scan.
    let r = rv(1, None, &["stop"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(r.out.contains("not running"), "{}", r.out);
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

const BUSY_409: &str = r#"{"error":{"message":"model has 1 request(s) running","type":"invalid_request_error","param":null,"code":"model_busy"}}"#;
const BUSY_503: &str =
    r#"{"error":{"message":"model busy","type":"server_error","param":null,"code":"model_busy"}}"#;

/// D3-S1.F6: `rv unload` / `rv deregister` send `?force=true` only with `--force`, and a refusal
/// because a request is running (`model_busy`) explains itself and names the override.
#[test]
fn unload_and_deregister_forward_force_and_explain_a_busy_refusal() {
    let s = serve(vec![
        ("DELETE /v1/admin/models/a", 409, BUSY_409),
        ("DELETE /v1/admin/models/a/register", 409, BUSY_409),
        ("DELETE /v1/admin/models/b", 200, "model 'b' evicted"),
        (
            "DELETE /v1/admin/models/b/register",
            200,
            "model 'b' deregistered",
        ),
    ]);
    let u = rv(s.port, Some("adm"), &["unload", "a"]);
    assert_eq!(u.code, 1);
    assert!(
        u.err.contains("model_busy") || u.err.contains("request(s) running"),
        "{}",
        u.err
    );
    assert!(u.err.contains("rv unload a --force"), "{}", u.err);
    let d = rv(s.port, Some("adm"), &["deregister", "a", "-y"]);
    assert_eq!(d.code, 1);
    assert!(d.err.contains("rv deregister a --force"), "{}", d.err);

    assert_eq!(rv(s.port, Some("adm"), &["unload", "b", "--force"]).code, 0);
    assert_eq!(
        rv(s.port, Some("adm"), &["deregister", "b", "-y", "--force"]).code,
        0
    );
    assert_eq!(rv(s.port, Some("adm"), &["unload", "b"]).code, 0);

    let reqs = s.requests.lock().expect("lock").clone();
    let lines: Vec<&str> = reqs.iter().map(|(l, _)| l.as_str()).collect();
    assert!(lines.contains(&"DELETE /v1/admin/models/a"), "{lines:?}");
    assert!(
        lines.contains(&"DELETE /v1/admin/models/b?force=true"),
        "{lines:?}"
    );
    assert!(
        lines.contains(&"DELETE /v1/admin/models/b/register?force=true"),
        "{lines:?}"
    );
    assert!(
        lines.contains(&"DELETE /v1/admin/models/b"),
        "a plain unload sends no force: {lines:?}"
    );
}

/// D3-S1.F6: `rv resize` puts `"force": true` in the body only with `--force`.
#[test]
fn resize_forwards_force_in_the_body() {
    let s = serve(vec![
        ("POST /v1/admin/models/a/resize", 409, BUSY_409),
        ("POST /v1/admin/models/b/resize", 200, "resized"),
    ]);
    let busy = rv(
        s.port,
        Some("adm"),
        &["resize", "a", "--kv-cache-gb", "2", "-y"],
    );
    assert_eq!(busy.code, 1);
    assert!(busy.err.contains("--force"), "{}", busy.err);
    assert!(
        !busy.err.contains("resize only applies to a loaded model"),
        "a busy refusal is not the not-loaded 409: {}",
        busy.err
    );
    assert_eq!(
        rv(
            s.port,
            Some("adm"),
            &["resize", "b", "--kv-cache-gb", "2", "-y", "--force"]
        )
        .code,
        0
    );
    assert_eq!(
        rv(
            s.port,
            Some("adm"),
            &["resize", "b", "--kv-cache-gb", "2", "-y"]
        )
        .code,
        0
    );

    let reqs = s.requests.lock().expect("lock").clone();
    let bodies: Vec<&str> = reqs
        .iter()
        .filter(|(l, _)| l.starts_with("POST /v1/admin/models/b/resize"))
        .map(|(_, b)| b.as_str())
        .collect();
    assert_eq!(bodies.len(), 2, "{reqs:?}");
    assert!(bodies[0].contains(r#""force":true"#), "{bodies:?}");
    assert!(!bodies[1].contains("force"), "{bodies:?}");
}

/// D3-S1.F6: `rv add` reads a 409 as "already registered" only when the server did not say it
/// is busy; and `rv load`'s pre-check names the busy models that hold the room.
#[test]
fn add_409_and_load_precheck_are_busy_aware() {
    let s = serve(vec![
        (
            "POST /v1/admin/models/add",
            409,
            "model 'x' already registered",
        ),
        ("POST /v1/admin/models/big/check", 200, BLOCKED_BY_BUSY),
        ("POST /v1/admin/models/viaload/check", 200, FITS),
        ("POST /v1/admin/models/viaload/load", 503, BUSY_503),
    ]);
    let dup = rv(s.port, Some("adm"), &["add", "x", "--vram-gb", "1", "-y"]);
    assert_eq!(dup.code, 1);
    assert!(dup.err.contains("is already registered"), "{}", dup.err);

    let big = rv(s.port, Some("adm"), &["load", "big"]);
    assert_eq!(big.code, 1);
    assert!(big.out.contains("has a request running"), "{}", big.out);
    assert!(big.out.contains("bielik"), "{}", big.out);
    assert!(big.out.contains("rv load big --force"), "{}", big.out);

    let refused = rv(s.port, Some("adm"), &["load", "viaload"]);
    assert_eq!(refused.code, 1);
    assert!(
        refused.err.contains("rv load viaload --force"),
        "{}",
        refused.err
    );
}

const BLOCKED_BY_BUSY: &str = r#"{"fits":false,"needed_gb":9.0,"free_gb":2.0,"would_evict":[],"busy_victims":[],"blocked_by_busy":["bielik"],"already_loaded":false}"#;

// ---- 0.9.1 V5c: sampling flags, patch merge, the profile line ----

/// Like [`rv`] with an admin key, but the config also names `models_dir`.
fn rv_models_dir(port: u16, models_dir: &std::path::Path, args: &[&str]) -> Run {
    let d = tempfile::tempdir().expect("tempdir");
    let cfg = d.path().join("c.config.json");
    std::fs::write(
        &cfg,
        serde_json::json!({"port": port, "models_dir": models_dir}).to_string(),
    )
    .expect("cfg");
    std::fs::write(
        d.path().join("c.keys.json"),
        r#"{"api_keys": [], "admin_api_keys": ["adm"]}"#,
    )
    .expect("keys");
    let o = Command::new(env!("CARGO_BIN_EXE_rv"))
        .args(args)
        .env("RUSTEDVINO_CONFIG", &cfg)
        .env_remove("RV_ADMIN_KEY")
        .env_remove("RV_PORT")
        .env("NO_COLOR", "1")
        .stdin(Stdio::null())
        .output()
        .expect("run rv");
    Run {
        code: o.status.code().unwrap_or(-1),
        out: String::from_utf8_lossy(&o.stdout).into_owned(),
        err: String::from_utf8_lossy(&o.stderr).into_owned(),
    }
}

/// A 0.9.1 registry: `mistral` (operator `top_p` 0.8 and presence 1.5 over a file T 0.6 and
/// `top_k` 20), `old` (`text_gen` without the field: a 0.9.0 server's entry), `embed` (no field, as
/// P9 says for non-text kinds).
const V5_MODELS: &str = r#"{"models":[
  {"id":"mistral","state":"not_loaded","device":"GPU","kind":"text_gen","vram_gb":14.0,
   "sampling":{"mode":"sample",
     "temperature":{"value":0.6,"source":"generation_config.json"},
     "top_p":{"value":0.8,"source":"config"},
     "top_k":{"value":20,"source":"generation_config.json"},
     "repetition_penalty":{"value":1.1,"source":"server default"},
     "presence_penalty":{"value":1.5,"source":"config"},
     "frequency_penalty":{"value":null,"source":"none"},
     "tools":"greedy"}},
  {"id":"old","state":"ready","device":"GPU","kind":"text_gen","vram_gb":5.0},
  {"id":"embed","state":"ready","device":"GPU","kind":"embedding","vram_gb":1.0}],
  "server":{"pid":1}}"#;

fn bodies(s: &Stub, prefix: &str) -> Vec<String> {
    s.requests
        .lock()
        .expect("lock")
        .iter()
        .filter(|(l, _)| l.starts_with(prefix))
        .map(|(_, b)| b.clone())
        .collect()
}

#[test]
fn patch_sampling_flags_are_checked_before_any_request() {
    let s = serve(vec![]);
    for args in [
        &["patch", "m", "--temperature", "3"][..],
        &["patch", "m", "--top-p", "0"],
        &["patch", "m", "--top-k", "0"],
        &["patch", "m", "--repetition-penalty", "0.9"],
        &["patch", "m", "--presence-penalty", "-2.5"],
        &["patch", "m", "--frequency-penalty", "nan"],
        &["patch", "m", "--temperature", "0", "--top-k", "20"],
        &["patch", "m", "--temperature", "0.2", "--clear-sampling"],
    ] {
        let r = rv(s.port, Some("adm"), args);
        assert_eq!(r.code, 1, "{args:?}: {}", r.err);
        assert!(r.err.contains("error:"), "{args:?}: {}", r.err);
    }
    assert!(s.hits.lock().expect("lock").is_empty());
}

#[test]
fn patch_merges_the_flags_over_the_operator_values() {
    let s = serve(vec![
        ("GET /v1/admin/models", 200, V5_MODELS),
        (
            "PATCH /v1/admin/models/mistral",
            200,
            r#"{"applied_live":["sampling"],"effective_on_next_load":[],"unchanged":[]}"#,
        ),
    ]);
    let r = rv(
        s.port,
        Some("adm"),
        &[
            "patch",
            "mistral",
            "--temperature",
            "0.15",
            "--presence-penalty",
            "-0.5",
        ],
    );
    assert_eq!(r.code, 0, "{}", r.err);
    let sent = bodies(&s, "PATCH /v1/admin/models/mistral");
    assert_eq!(sent.len(), 1);
    let v: serde_json::Value = serde_json::from_str(&sent[0]).expect("json");
    assert_eq!(
        v,
        serde_json::json!({"sampling": {"temperature": 0.15, "top_p": 0.8, "presence_penalty": -0.5}}),
        "operator top_p kept, the flags win"
    );
    assert!(
        r.out.contains("sampling: top_p 0.8 · presence_penalty 1.5 → T 0.15 · top_p 0.8 · presence_penalty -0.5"),
        "{}",
        r.out
    );

    let c = rv(
        s.port,
        Some("adm"),
        &["patch", "mistral", "--clear-sampling"],
    );
    assert_eq!(c.code, 0, "{}", c.err);
    let sent = bodies(&s, "PATCH /v1/admin/models/mistral");
    assert_eq!(sent[1], r#"{"sampling":null}"#);
    assert!(c.out.contains("→ none"), "{}", c.out);
}

#[test]
fn patch_refuses_sampling_on_a_non_text_kind_and_an_old_server() {
    let s = serve(vec![("GET /v1/admin/models", 200, V5_MODELS)]);
    let e = rv(
        s.port,
        Some("adm"),
        &["patch", "embed", "--temperature", "0.2"],
    );
    assert_eq!(e.code, 1);
    assert!(e.err.contains("of kind embedding"), "{}", e.err);
    assert!(
        !e.err.contains("0.9.1"),
        "the kind check comes first: {}",
        e.err
    );
    let o = rv(s.port, Some("adm"), &["patch", "old", "--clear-sampling"]);
    assert_eq!(o.code, 1);
    assert!(
        o.err
            .contains("server < 0.9.1: sampling settings NOT applied"),
        "{}",
        o.err
    );
    let u = rv(s.port, Some("adm"), &["patch", "ghost", "--top-k", "5"]);
    assert_eq!(u.code, 1);
    assert!(u.err.contains("not registered"), "{}", u.err);
    assert!(bodies(&s, "PATCH").is_empty(), "nothing was sent");
}

/// Review F1: `--clear-sampling` goes through on a non-text model (an auto-detected add may have
/// stored sampling there); value flags stay refused.
#[test]
fn clear_sampling_is_allowed_on_a_non_text_model() {
    let s = serve(vec![
        ("GET /v1/admin/models", 200, V5_MODELS),
        (
            "PATCH /v1/admin/models/embed",
            200,
            r#"{"applied_live":["sampling"],"effective_on_next_load":[],"unchanged":[]}"#,
        ),
    ]);
    let r = rv(s.port, Some("adm"), &["patch", "embed", "--clear-sampling"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert_eq!(
        bodies(&s, "PATCH /v1/admin/models/embed"),
        vec![r#"{"sampling":null}"#]
    );
    assert!(
        r.out
            .contains("sampling: (not reported for this kind) → none"),
        "{}",
        r.out
    );
}

#[test]
fn reload_config_prints_applied_live() {
    let s = serve(vec![(
        "POST /v1/admin/config/reload",
        200,
        r#"{"added":[],"updated":[],"left_untouched":["mistral"],"skipped_missing_files":[],
            "unchanged":3,"models_not_applied":[],
            "applied_live":[{"model":"mistral","fields":["sampling"]}]}"#,
    )]);
    let r = rv(s.port, Some("adm"), &["reload-config"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(
        r.out.contains("mistral: sampling applied live"),
        "{}",
        r.out
    );
}

#[test]
fn status_with_an_id_shows_the_model_and_its_profile() {
    let s = serve(vec![("GET /v1/admin/models", 200, V5_MODELS)]);
    let d = tempfile::tempdir().expect("tempdir");
    let r = rv_models_dir(s.port, d.path(), &["status", "mistral"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(
        r.out
            .contains("mistral · not_loaded · GPU · 14.0 GB · text_gen"),
        "{}",
        r.out
    );
    assert!(
        r.out.contains(
            "sampling: T 0.6 (generation_config.json) · top_p 0.8 (config) · top_k 20 \
             (generation_config.json) · repetition_penalty 1.1 (server default) · \
             presence_penalty 1.5 (config) · tools → greedy"
        ),
        "{}",
        r.out
    );
    assert!(
        r.out
            .contains("values the client sends win · realtime: profile"),
        "{}",
        r.out
    );
    // No full report: no health/metrics requests.
    assert_eq!(*s.hits.lock().expect("lock"), vec!["GET /v1/admin/models"]);

    let e = rv_models_dir(s.port, d.path(), &["status", "embed"]);
    assert_eq!(e.code, 0);
    assert!(
        !e.out.contains("sampling") && !e.err.contains("0.9.1"),
        "{}{}",
        e.out,
        e.err
    );
    // Review F3: check/status on an old server say it neutrally (nothing was sent).
    let o = rv_models_dir(s.port, d.path(), &["status", "old"]);
    assert_eq!(o.code, 0);
    assert!(
        o.out
            .contains("sampling profile: not reported by this server (< 0.9.1)"),
        "{}",
        o.out
    );
    assert!(!o.err.contains("NOT applied"), "{}", o.err);
    let g = rv_models_dir(s.port, d.path(), &["status", "ghost"]);
    assert_eq!(g.code, 1);
    assert!(g.err.contains("rv check ghost"), "{}", g.err);
    let j = rv_models_dir(s.port, d.path(), &["status", "mistral", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&j.out).expect("json");
    assert_eq!(v["id"], "mistral");
}

/// F1a: since 0.9.1 `/v1/models` also lists `on_demand` models that are not loaded, with a
/// `state`. Without the admin list, `rv status` shows that `state` instead of `?`.
#[test]
fn status_without_admin_list_shows_the_v1_models_state() {
    let s = serve(vec![
        ("GET /health", 200, r#"{"status":"ok","version":"0.9.1"}"#),
        (
            "GET /v1/models",
            200,
            r#"{"data":[{"id":"m1"},{"id":"od-model","state":"not_loaded"}]}"#,
        ),
    ]);
    let r = rv(s.port, None, &["status"]);
    let row = r
        .out
        .lines()
        .find(|l| l.contains("od-model"))
        .unwrap_or_default();
    assert!(row.contains("not_loaded"), "{}", r.out);
}

#[test]
fn status_without_an_id_is_the_full_report() {
    let s = serve(status_routes());
    let r = rv(s.port, Some("adm"), &["status"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(r.out.contains("pid 4242424"), "{}", r.out);
    assert!(!r.out.contains("sampling:"), "{}", r.out);
}

#[test]
fn check_shows_the_server_profile_or_a_local_preview() {
    let s = serve(vec![
        ("GET /v1/admin/models", 200, V5_MODELS),
        ("POST /v1/admin/models/mistral/check", 200, FITS),
    ]);
    let d = tempfile::tempdir().expect("tempdir");
    let r = rv_models_dir(s.port, d.path(), &["check", "mistral"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(
        r.out.contains("sampling: T 0.6 (generation_config.json)"),
        "{}",
        r.out
    );
    let j = rv_models_dir(s.port, d.path(), &["check", "mistral", "--json"]);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&j.out).expect("json"),
        serde_json::from_str::<serde_json::Value>(FITS).expect("json")
    );

    // Not registered (404), on disk with a greedy file: the local preview + its hint.
    let m = d.path().join("ministral");
    std::fs::create_dir(&m).expect("dir");
    std::fs::write(m.join("generation_config.json"), r#"{"bos_token_id": 1}"#).expect("file");
    let p = rv_models_dir(s.port, d.path(), &["check", "ministral"]);
    assert_eq!(p.code, 1, "a 404 still exits 1");
    assert!(
        p.out.contains(
            "sampling (local preview, text_gen/vision): greedy · repetition_penalty 1.1 (server \
             default)"
        ),
        "{}",
        p.out
    );
    assert!(
        p.out
            .contains("rv add ministral --temperature 0.15 --top-p 0.95"),
        "{}",
        p.out
    );
    // Not on disk: no preview.
    let n = rv_models_dir(s.port, d.path(), &["check", "nowhere"]);
    assert_eq!(n.code, 1);
    assert!(!n.out.contains("sampling"), "{}", n.out);
}

#[test]
fn add_previews_the_profile_and_prints_the_servers() {
    let added = r#"{"models":[{"id":"ministral","state":"ready","device":"GPU","kind":"text_gen",
      "vram_gb":1.0,"sampling":{"mode":"sample",
        "temperature":{"value":0.15,"source":"config"},
        "top_p":{"value":0.95,"source":"config"},
        "top_k":{"value":null,"source":"none"},
        "repetition_penalty":{"value":1.1,"source":"server default"},
        "presence_penalty":{"value":null,"source":"none"},
        "frequency_penalty":{"value":null,"source":"none"},"tools":"greedy"}}]}"#;
    let s = serve(vec![
        ("GET /v1/admin/models", 200, added),
        (
            "POST /v1/admin/models/add",
            200,
            r#"{"message":"added and loaded"}"#,
        ),
    ]);
    let d = tempfile::tempdir().expect("tempdir");
    let m = d.path().join("ministral");
    std::fs::create_dir(&m).expect("dir");
    std::fs::write(m.join("generation_config.json"), r#"{"bos_token_id": 1}"#).expect("file");
    let r = rv_models_dir(
        s.port,
        d.path(),
        &[
            "add",
            "ministral",
            "--vram-gb",
            "1",
            "-y",
            "--temperature",
            "0.15",
            "--top-p",
            "0.95",
        ],
    );
    assert_eq!(r.code, 0, "{}", r.err);
    let preview = "sampling (local preview, text_gen/vision): T 0.15 (config) · top_p 0.95 \
                   (config) · top_k — none · repetition_penalty 1.1 (server default) · tools → greedy";
    let server = "sampling: T 0.15 (config) · top_p 0.95 (config) · top_k — none · \
                  repetition_penalty 1.1 (server default) · tools → greedy";
    assert!(r.out.contains(preview), "{}", r.out);
    assert!(r.out.contains(server), "{}", r.out);
    let sent = bodies(&s, "POST /v1/admin/models/add");
    let v: serde_json::Value = serde_json::from_str(&sent[0]).expect("json");
    assert_eq!(
        v["sampling"],
        serde_json::json!({"temperature": 0.15, "top_p": 0.95})
    );

    // top_p without T on a greedy local file: refused before the request (the server would 400).
    let bad = rv_models_dir(
        s.port,
        d.path(),
        &["add", "ministral", "--vram-gb", "1", "-y", "--top-p", "0.9"],
    );
    assert_eq!(bad.code, 1);
    assert!(bad.err.contains("do_sample"), "{}", bad.err);
    // A sampling flag on a non-text --kind: refused.
    let k = rv_models_dir(
        s.port,
        d.path(),
        &[
            "add",
            "ministral",
            "--vram-gb",
            "1",
            "-y",
            "--kind",
            "stt",
            "--temperature",
            "0.2",
        ],
    );
    assert_eq!(k.code, 1);
    assert!(k.err.contains("text_gen and vision"), "{}", k.err);
    assert_eq!(
        bodies(&s, "POST /v1/admin/models/add").len(),
        1,
        "only the first add was sent"
    );
}

#[test]
fn add_against_an_old_server_says_sampling_was_not_applied() {
    let old = r#"{"models":[{"id":"ministral","state":"ready","device":"GPU","kind":"text_gen",
      "vram_gb":1.0}]}"#;
    let s = serve(vec![
        ("GET /v1/admin/models", 200, old),
        (
            "POST /v1/admin/models/add",
            200,
            r#"{"message":"added and loaded"}"#,
        ),
    ]);
    let d = tempfile::tempdir().expect("tempdir");
    let with = rv_models_dir(
        s.port,
        d.path(),
        &[
            "add",
            "ministral",
            "--vram-gb",
            "1",
            "-y",
            "--temperature",
            "0.15",
        ],
    );
    assert_eq!(with.code, 1, "flags sent to a server that dropped them");
    assert!(
        with.err
            .contains("server < 0.9.1: sampling settings NOT applied"),
        "{}",
        with.err
    );
    let without = rv_models_dir(
        s.port,
        d.path(),
        &["add", "ministral", "--vram-gb", "1", "-y"],
    );
    assert_eq!(without.code, 0, "{}", without.err);
    assert!(without.err.contains("server < 0.9.1"), "{}", without.err);
}

/// Review F10: the "previews a model on disk" help only follows a not-registered model, not a
/// failed list request.
#[test]
fn status_with_an_id_gives_the_help_only_for_an_unregistered_model() {
    let s = serve(vec![(
        "GET /v1/admin/models",
        401,
        r#"{"error":"bad key"}"#,
    )]);
    let r = rv(s.port, Some("adm"), &["status", "m"]);
    assert_eq!(r.code, 1);
    assert!(r.err.contains("401"), "{}", r.err);
    assert!(!r.err.contains("rv check"), "{}", r.err);
}

/// Review F3: `rv check` on an old server prints the neutral line, exit code unchanged.
#[test]
fn check_on_an_old_server_is_neutral() {
    let s = serve(vec![
        ("GET /v1/admin/models", 200, V5_MODELS),
        ("POST /v1/admin/models/old/check", 200, LOADED),
    ]);
    let d = tempfile::tempdir().expect("tempdir");
    let r = rv_models_dir(s.port, d.path(), &["check", "old"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(
        r.out
            .contains("sampling profile: not reported by this server (< 0.9.1)"),
        "{}",
        r.out
    );
    assert!(!r.err.contains("NOT applied"), "{}", r.err);
}

/// Review F2: a non-text directory on disk (here Whisper's encoder + decoder) gets no preview,
/// and `rv add` without --kind refuses sampling flags for it.
#[test]
fn non_text_dirs_get_no_preview_and_refuse_sampling_flags() {
    let s = serve(vec![(
        "POST /v1/admin/models/add",
        200,
        r#"{"message":"added and loaded"}"#,
    )]);
    let d = tempfile::tempdir().expect("tempdir");
    let w = d.path().join("whisper");
    std::fs::create_dir(&w).expect("dir");
    for f in ["openvino_encoder_model.xml", "openvino_decoder_model.xml"] {
        std::fs::write(w.join(f), b"").expect("file");
    }
    std::fs::write(w.join("generation_config.json"), r#"{"do_sample": false}"#).expect("file");
    let c = rv_models_dir(s.port, d.path(), &["check", "whisper"]);
    assert_eq!(c.code, 1, "a 404 still exits 1");
    assert!(!c.out.contains("sampling"), "{}", c.out);
    let a = rv_models_dir(
        s.port,
        d.path(),
        &[
            "add",
            "whisper",
            "--vram-gb",
            "1",
            "-y",
            "--temperature",
            "0.2",
        ],
    );
    assert_eq!(a.code, 1);
    assert!(a.err.contains("of kind stt"), "{}", a.err);
    assert!(bodies(&s, "POST /v1/admin/models/add").is_empty());
    let plain = rv_models_dir(
        s.port,
        d.path(),
        &["add", "whisper", "--vram-gb", "1", "-y"],
    );
    assert_eq!(plain.code, 0, "{}", plain.err);
    assert!(!plain.out.contains("local preview"), "{}", plain.out);
}

/// Review F1: a sampling flag stored on a model the server detected as non-text: the warning
/// names the way back.
#[test]
fn add_that_lands_on_a_non_text_kind_names_clear_sampling() {
    let s = serve(vec![
        (
            "GET /v1/admin/models",
            200,
            r#"{"models":[{"id":"bge","state":"ready","device":"GPU","kind":"embedding","vram_gb":1.0}]}"#,
        ),
        (
            "POST /v1/admin/models/add",
            200,
            r#"{"message":"added and loaded"}"#,
        ),
    ]);
    let d = tempfile::tempdir().expect("tempdir");
    let r = rv_models_dir(
        s.port,
        d.path(),
        &["add", "bge", "--vram-gb", "1", "-y", "--temperature", "0.2"],
    );
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(r.err.contains("rv patch bge --clear-sampling"), "{}", r.err);
    assert!(r.err.contains("of kind embedding"), "{}", r.err);
}

/// `/metrics` of a 0.9.1 server with one text model loaded: decode gauge, prefill buckets over
/// two modalities, a key-usage series and two unlabelled samples.
const METRICS_091: &str = "# HELP rustedvino_model_loaded x\n\
# TYPE rustedvino_model_loaded gauge\n\
rustedvino_model_loaded{model=\"m1\",device=\"GPU.1\",kind=\"text_gen\"} 1\n\
rustedvino_tokens_per_second{model=\"m1\",device=\"GPU.1\",kind=\"text_gen\"} 38.4\n\
rustedvino_prefill_tokens_per_second_bucket{model=\"m1\",device=\"GPU.1\",kind=\"text_gen\",modality=\"text\",le=\"500\"} 0\n\
rustedvino_prefill_tokens_per_second_bucket{model=\"m1\",device=\"GPU.1\",kind=\"text_gen\",modality=\"text\",le=\"1000\"} 2\n\
rustedvino_prefill_tokens_per_second_bucket{model=\"m1\",device=\"GPU.1\",kind=\"text_gen\",modality=\"text\",le=\"+Inf\"} 2\n\
rustedvino_prefill_tokens_per_second_bucket{model=\"m1\",device=\"GPU.1\",kind=\"text_gen\",modality=\"image\",le=\"500\"} 0\n\
rustedvino_prefill_tokens_per_second_bucket{model=\"m1\",device=\"GPU.1\",kind=\"text_gen\",modality=\"image\",le=\"1000\"} 2\n\
rustedvino_prefill_tokens_per_second_bucket{model=\"m1\",device=\"GPU.1\",kind=\"text_gen\",modality=\"image\",le=\"+Inf\"} 2\n\
rustedvino_key_usage_total{key_suffix=\"9f3a7c\"} 4\n\
rustedvino_gpu_poisoned 0\n\
rustedvino_ov_cache_bytes 123456\n";

/// V1-rv P3: `rv metrics` fetches /metrics with the admin key and prints the per-model line
/// (p50 from the buckets summed over modality), then the raw samples, unlabelled ones included,
/// with `key_suffix` masked.
#[test]
fn metrics_prints_the_model_line_then_the_masked_samples() {
    let s = serve(vec![("GET /metrics", 200, METRICS_091)]);
    let r = rv(s.port, Some("adm"), &["metrics"]);
    assert_eq!(r.code, 0, "{}", r.err);
    // 4 observations in (500, 1000]: rank 2 → halfway → 750.
    assert!(
        r.out
            .contains("m1  effective prefill ~750 tok/s (p50) \u{b7} decode 38.4 tok/s"),
        "{}",
        r.out
    );
    assert!(r.out.contains("rustedvino_gpu_poisoned 0"), "{}", r.out);
    assert!(
        r.out.contains("rustedvino_ov_cache_bytes 123456"),
        "{}",
        r.out
    );
    assert!(!r.out.contains("9f3a7c"), "key_suffix leaked: {}", r.out);
    assert!(r.out.contains("key_suffix=\"\u{2026}7c\""), "{}", r.out);
    assert!(!r.out.contains("# HELP"), "{}", r.out);
    assert!(
        s.authed.lock().expect("lock").iter().all(|a| *a),
        "sent with the admin key"
    );
}

/// A model that ran on the CPU and now runs on `GPU.0`: both devices' series are still exposed.
const METRICS_MOVED: &str = "\
rustedvino_model_loaded{model=\"m1\",device=\"CPU\",kind=\"text_gen\"} 1\n\
rustedvino_model_loaded{model=\"m1\",device=\"GPU.0\",kind=\"text_gen\"} 1\n\
rustedvino_tokens_per_second{model=\"m1\",device=\"CPU\",kind=\"text_gen\"} 26.3\n\
rustedvino_tokens_per_second{model=\"m1\",device=\"GPU.0\",kind=\"text_gen\"} 11.9\n";

/// `rv metrics` reads the admin model list for the device each model is on now and shows that
/// device's decode figure; without the list it shows every device's figure, never one at random.
#[test]
fn metrics_summary_uses_the_current_device() {
    let s = serve(vec![
        ("GET /metrics", 200, METRICS_MOVED),
        (
            "GET /v1/admin/models",
            200,
            r#"{"models": [{"id": "m1", "state": "ready", "device": "GPU.0"}]}"#,
        ),
    ]);
    let r = rv(s.port, Some("adm"), &["metrics"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(
        r.out
            .contains("m1  effective prefill unknown \u{b7} decode 11.9 tok/s\n"),
        "{}",
        r.out
    );
    assert!(
        s.hits
            .lock()
            .expect("lock")
            .contains(&"GET /v1/admin/models".to_owned()),
        "admin list read"
    );

    let s = serve(vec![("GET /metrics", 200, METRICS_MOVED)]);
    let r = rv(s.port, Some("adm"), &["metrics"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(
        r.out.contains(
            "m1  effective prefill unknown \u{b7} decode 26.3 tok/s (CPU), 11.9 tok/s (GPU.0)\n"
        ),
        "{}",
        r.out
    );
}

#[test]
fn metrics_grep_and_json() {
    let s = serve(vec![("GET /metrics", 200, METRICS_091)]);
    let g = rv(s.port, Some("adm"), &["metrics", "--grep", "ov_cache"]);
    assert_eq!(g.code, 0, "{}", g.err);
    assert!(g.out.contains("m1  effective prefill"), "{}", g.out);
    assert!(
        g.out.contains("rustedvino_ov_cache_bytes 123456"),
        "{}",
        g.out
    );
    assert!(!g.out.contains("_bucket"), "{}", g.out);
    let j = rv(
        s.port,
        Some("adm"),
        &["metrics", "--json", "--grep", "key_usage"],
    );
    assert_eq!(j.code, 0, "{}", j.err);
    let v: serde_json::Value = serde_json::from_str(&j.out).expect("json");
    assert_eq!(v["models"][0]["model"], "m1");
    assert_eq!(v["models"][0]["effective_prefill_p50_tok_s"], 750.0);
    assert_eq!(v["models"][0]["decode_tok_s"], 38.4);
    assert_eq!(
        v["samples"],
        serde_json::json!([{"name": "rustedvino_key_usage_total",
                            "labels": {"key_suffix": "\u{2026}7c"}, "value": 4.0}])
    );
}

#[test]
fn metrics_against_an_older_server_says_unknown_and_without_a_key_refuses() {
    let old = "rustedvino_model_loaded{model=\"m1\",device=\"GPU.1\",kind=\"text_gen\"} 1\n";
    let s = serve(vec![("GET /metrics", 200, old)]);
    let r = rv(s.port, Some("adm"), &["metrics"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(
        r.out
            .contains("m1  effective prefill unknown \u{b7} decode -"),
        "{}",
        r.out
    );
    let refused = rv_keys_file(s.port, None, &["metrics"]);
    assert_eq!(refused.code, 1);
    assert!(
        refused.err.contains("no admin key available"),
        "{}",
        refused.err
    );
    let s = serve(vec![(
        "GET /metrics",
        503,
        r#"{"error":{"message":"admin keys are not configured","code":"admin_not_configured"}}"#,
    )]);
    let r = rv(s.port, Some("adm"), &["metrics"]);
    assert_eq!(r.code, 1);
    assert!(r.err.contains("error (503)"), "{}", r.err);
}

/// V1-rv P3: `rv ps` and `rv status` keep reading only the labelled samples — unlabelled lines
/// in the scrape change nothing in their output.
#[test]
fn ps_and_status_output_is_unchanged_by_unlabelled_samples() {
    let with_unlabelled: &'static str = Box::leak(
        format!("{METRICS}rustedvino_gpu_poisoned 1\nrustedvino_ov_cache_bytes 99\n")
            .into_boxed_str(),
    );
    for verb in [
        &["ps"][..],
        &["ps", "--json"][..],
        &["status"][..],
        &["status", "--json"][..],
    ] {
        let plain = status_routes();
        let mut extra = status_routes();
        let at = extra
            .iter()
            .position(|(k, _, _)| *k == "GET /metrics")
            .expect("metrics route");
        extra[at] = ("GET /metrics", 200, with_unlabelled);
        let a = rv(serve(plain).port, Some("adm"), verb);
        let b = rv(serve(extra).port, Some("adm"), verb);
        assert_eq!(a.code, b.code, "{verb:?}");
        assert_eq!(
            without_temp_paths(&a.out),
            without_temp_paths(&b.out),
            "{verb:?}"
        );
    }
}

/// `out` with every token naming the per-run temporary config or keys file replaced.
fn without_temp_paths(out: &str) -> String {
    out.lines()
        .map(|l| {
            l.split(' ')
                .map(|t| {
                    if t.contains("c.keys.json") || t.contains("c.config.json") {
                        "<tmp>"
                    } else {
                        t
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

const RUNNING_1: &str = "rustedvino_requests_running{model=\"d\"} 1\n\
rustedvino_requests_running{model=\"a\"} 1\nrustedvino_requests_running{model=\"n\"} 0\n\
rustedvino_requests_running{model=\"o\"} 1\n";

/// V1-rv P1: `rv unload --force` says what it will do, asks for the JSON outcome and prints
/// one of the three outcomes; the abort is also a warning on stderr; an older server's text
/// answer reads "unknown". Exit 0 in every case.
#[test]
fn forced_unload_prints_the_precall_line_and_the_outcome() {
    let s = serve(vec![
        ("GET /metrics", 200, RUNNING_1),
        (
            "DELETE /v1/admin/models/n",
            200,
            r#"{"evicted":["n"],"forced":true,"drained_s":0.0,"aborted":0,"teardown_ms":5}"#,
        ),
        (
            "DELETE /v1/admin/models/d",
            200,
            r#"{"evicted":["d"],"forced":true,"drained_s":81.05,"aborted":0,"teardown_ms":700}"#,
        ),
        (
            "DELETE /v1/admin/models/a",
            200,
            r#"{"evicted":["a"],"forced":true,"drained_s":120.0,"aborted":1,"teardown_ms":900,"message":"model 'a' evicted"}"#,
        ),
        ("DELETE /v1/admin/models/o", 200, "model 'o' evicted"),
    ]);
    let n = rv(s.port, Some("adm"), &["unload", "n", "--force"]);
    assert_eq!(n.code, 0, "{}", n.err);
    assert!(
        n.out
            .contains("0 request(s) running \u{2014} will drain up to 120 s then abort"),
        "{}",
        n.out
    );
    assert!(
        n.out
            .contains("model 'n' evicted \u{2014} nothing was running"),
        "{}",
        n.out
    );
    let d = rv(s.port, Some("adm"), &["unload", "d", "--force"]);
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(d.out.contains("1 request(s) running"), "{}", d.out);
    assert!(
        d.out.contains(
            "model 'd' evicted \u{2014} its 1 running request(s) finished during the drain (81.0 s)"
        ),
        "{}",
        d.out
    );
    assert!(
        !d.err.contains("ABORTED") && !d.err.contains("aborted"),
        "{}",
        d.err
    );
    let a = rv(s.port, Some("adm"), &["unload", "a", "--force"]);
    assert_eq!(a.code, 0, "{}", a.err);
    assert!(
        a.out.contains(
            "model 'a' evicted \u{2014} drained 120.0 s; 1 request(s) still running at the drain \
             deadline: a text model aborts them (client sees model_unloading); other engines \
             finish or fail them during teardown"
        ),
        "{}",
        a.out
    );
    assert!(
        a.err
            .contains("warning: unload a: 1 request(s) still running at the drain deadline"),
        "{}",
        a.err
    );
    let o = rv(s.port, Some("adm"), &["unload", "o", "--force"]);
    assert_eq!(o.code, 0, "{}", o.err);
    assert!(
        o.out
            .contains("model 'o' evicted \u{2014} drain/abort unknown"),
        "{}",
        o.out
    );

    let accepts = s.accepts.lock().expect("lock").clone();
    for m in ["n", "d", "a", "o"] {
        let want = format!("/v1/admin/models/{m}?force=true");
        assert!(
            accepts
                .iter()
                .any(|(p, a)| *p == want && a.as_deref() == Some("application/json")),
            "{m}: {accepts:?}"
        );
    }
    // The scrape itself and a plain unload send no JSON Accept.
    assert!(
        accepts
            .iter()
            .filter(|(p, _)| p == "/metrics")
            .all(|(_, a)| a.as_deref() != Some("application/json")),
        "{accepts:?}"
    );
}

/// V1-rv P1: a plain unload is unchanged — no pre-call line, no JSON Accept, the server's text.
#[test]
fn plain_unload_sends_no_accept_and_prints_the_servers_text() {
    let s = serve(vec![(
        "DELETE /v1/admin/models/b",
        200,
        "model 'b' evicted",
    )]);
    let r = rv(s.port, Some("adm"), &["unload", "b"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(r.out.contains("model 'b' evicted\n"), "{}", r.out);
    assert!(!r.out.contains("request(s) running"), "{}", r.out);
    let accepts = s.accepts.lock().expect("lock").clone();
    assert_eq!(accepts.len(), 1, "no /metrics scrape: {accepts:?}");
    assert_ne!(accepts[0].1.as_deref(), Some("application/json"));
}

/// V1-rv P1: `rv deregister --force` the same way; a model that was not loaded evicted nothing.
#[test]
fn forced_deregister_prints_the_outcome() {
    let s = serve(vec![
        ("GET /metrics", 200, RUNNING_1),
        (
            "DELETE /v1/admin/models/a/register",
            200,
            r#"{"evicted":["a"],"forced":true,"drained_s":120.0,"aborted":2,"teardown_ms":9}"#,
        ),
        (
            "DELETE /v1/admin/models/z/register",
            200,
            r#"{"evicted":[],"forced":true,"drained_s":0.0,"aborted":0,"teardown_ms":0}"#,
        ),
    ]);
    let a = rv(s.port, Some("adm"), &["deregister", "a", "-y", "--force"]);
    assert_eq!(a.code, 0, "{}", a.err);
    assert!(
        a.out.contains(
            "model 'a' deregistered \u{2014} drained 120.0 s; 2 request(s) still running"
        ),
        "{}",
        a.out
    );
    assert!(
        a.err.contains("deregister a: 2 request(s) still running"),
        "{}",
        a.err
    );
    let z = rv(s.port, Some("adm"), &["deregister", "z", "-y", "--force"]);
    assert_eq!(z.code, 0, "{}", z.err);
    assert!(
        z.out.contains("unknown number of request(s) running"),
        "no series for z: {}",
        z.out
    );
    assert!(
        z.out
            .contains("model 'z' deregistered \u{2014} it was not loaded, nothing was evicted"),
        "{}",
        z.out
    );
}

/// V1-rv P5: `rv load` prints the time it measured beside the server's engine load time and
/// device; "engine unknown" (device from the registry) when the metric is not there.
#[test]
fn load_prints_its_time_beside_the_engine_time() {
    let s = serve(vec![
        ("POST /v1/admin/models/m1/check", 200, FITS),
        ("POST /v1/admin/models/m1/load", 200, "model 'm1' loaded"),
        (
            "GET /metrics",
            200,
            "rustedvino_model_load_duration_seconds{model=\"m1\",device=\"GPU.1\",kind=\"text_gen\"} 12.84\n",
        ),
        ("GET /v1/admin/models", 200, NOT_LOADED_GPU1),
    ]);
    let r = rv(s.port, Some("adm"), &["load", "m1"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(r.out.contains("model 'm1' loaded"), "{}", r.out);
    let line = r
        .out
        .lines()
        .find(|l| l.contains("loaded in "))
        .unwrap_or_else(|| panic!("no timing line: {}", r.out));
    assert!(line.contains(" s on GPU.1 (engine 12.8 s)"), "{line}");
    assert!(
        !r.out.contains("request(s) running"),
        "a plain load: {}",
        r.out
    );

    let old = serve(vec![
        ("POST /v1/admin/models/m1/load", 200, "model 'm1' loaded"),
        ("GET /v1/admin/models", 200, NOT_LOADED_GPU),
    ]);
    let r = rv(old.port, Some("adm"), &["load", "m1", "--no-check"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(r.out.contains(" s on GPU (engine unknown)"), "{}", r.out);
}

const EVICTS_M1: &str = r#"{"fits":true,"needed_gb":9.0,"free_gb":2.0,"would_evict":["m1"],"busy_victims":["m1"],"already_loaded":false}"#;

/// V1-rv P1: a forced load says how many requests its victims run and the 15 s drain, asks for
/// the outcome JSON and names what each eviction did; an older server reads "unknown".
#[test]
fn forced_load_prints_the_precall_line_and_its_victims_outcome() {
    let s = serve(vec![
        ("POST /v1/admin/models/big/check", 200, EVICTS_M1),
        (
            "POST /v1/admin/models/big/load",
            200,
            r#"{"evicted":["m1"],"forced":true,"drained_s":15.0,"aborted":2,"teardown_ms":800,"message":"model 'big' loaded. KV pool 1.25 GiB (requested 4.00 — clamped)"}"#,
        ),
        ("POST /v1/admin/models/old/check", 200, EVICTS_M1),
        ("POST /v1/admin/models/old/load", 200, "model 'old' loaded"),
        ("GET /metrics", 200, METRICS),
        ("GET /v1/admin/models", 200, POOL_MODELS),
    ]);
    let r = rv(s.port, Some("adm"), &["load", "big", "--force", "-y"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(
        r.out
            .contains("2 request(s) running \u{2014} will drain up to 15 s then abort"),
        "{}",
        r.out
    );
    assert!(r.out.contains("model 'big' loaded"), "{}", r.out);
    assert!(
        r.out.contains(
            "evicted to make room: m1 \u{2014} drained 15.0 s; 2 request(s) still running at the \
             drain deadline"
        ),
        "{}",
        r.out
    );
    assert!(
        r.err
            .contains("the forced load: 2 request(s) still running"),
        "{}",
        r.err
    );
    assert!(
        r.out
            .contains("model 'big' loaded. KV pool 1.25 GiB (requested 4.00 \u{2014} clamped)"),
        "the server's message, verbatim: {}",
        r.out
    );
    assert!(r.out.contains("loaded in "), "{}", r.out);
    let accepts = s.accepts.lock().expect("lock").clone();
    assert!(
        accepts
            .iter()
            .any(|(p, a)| p == "/v1/admin/models/big/load"
                && a.as_deref() == Some("application/json")),
        "{accepts:?}"
    );

    let o = rv(s.port, Some("adm"), &["load", "old", "--force", "-y"]);
    assert_eq!(o.code, 0, "{}", o.err);
    assert!(o.out.contains("model 'old' loaded"), "{}", o.out);
    assert!(o.out.contains("drain/abort unknown"), "{}", o.out);
}

/// V1-rv P1: `rv resize --force` says what it will do, asks for the outcome JSON and prints what
/// the force did, the reload's own victims included; an older server reads "unknown".
#[test]
fn forced_resize_prints_the_precall_line_and_the_outcome() {
    let s = serve(vec![
        (
            "POST /v1/admin/models/m1/resize",
            200,
            r#"{"evicted":["m1","m2"],"forced":true,"drained_s":4.25,"aborted":0,"teardown_ms":800,"message":"model 'm1' resized to 1.00 GiB (64 KV blocks; requested 2.00)"}"#,
        ),
        (
            "POST /v1/admin/models/old/resize",
            200,
            "model 'old' resized to 2 GiB",
        ),
        ("GET /metrics", 200, METRICS),
        ("GET /v1/admin/models", 200, POOL_MODELS),
    ]);
    let r = rv(
        s.port,
        Some("adm"),
        &["resize", "m1", "--kv-cache-gb", "2", "-y", "--force"],
    );
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(
        r.out
            .contains("2 request(s) running \u{2014} will drain up to 120 s then abort"),
        "{}",
        r.out
    );
    assert!(
        r.out.contains(
            "model 'm1' resized to 1.00 GiB (64 KV blocks; requested 2.00) \u{2014} its running \
             request(s) finished during the drain (4.2 s) (the drain figure includes m2, which the \
             reload evicted to make room)"
        ),
        "{}",
        r.out
    );

    let o = rv(
        s.port,
        Some("adm"),
        &["resize", "old", "--kv-cache-gb", "2", "-y", "--force"],
    );
    assert_eq!(o.code, 0, "{}", o.err);
    assert!(
        o.out
            .contains("model 'old' resized to 2 GiB \u{2014} drain/abort unknown"),
        "{}",
        o.out
    );
    let accepts = s.accepts.lock().expect("lock").clone();
    assert!(
        accepts
            .iter()
            .any(|(p, a)| p == "/v1/admin/models/m1/resize"
                && a.as_deref() == Some("application/json")),
        "{accepts:?}"
    );
}

/// A registry for the forced load/resize tests (the load-time line and the evicted diff read it).
const POOL_MODELS: &str = r#"{"models":[
  {"id":"big","state":"not_loaded","device":"GPU.1","kind":"text_gen","vram_gb":9.0},
  {"id":"m1","state":"ready","device":"GPU.1","kind":"text_gen","vram_gb":5.0,"kv_pool_mib":1024,
   "kv_requested_gb":2.0,"clamped_by":"cache_size_gb"}],
  "server":{"pid":1,"uptime_secs":1,"engine":"RustedVINO"}}"#;

/// V1-rv R-5 / R-9: the engine time is the series of the device the model is on now (the gauge
/// keeps one per device it was ever loaded on); a forced load whose check names no victim says
/// it evicts nothing; an already-Ready model (the idempotent 200 under --no-check) prints
/// "already loaded" and no time.
#[test]
fn load_time_follows_the_current_device_and_already_loaded_has_none() {
    let s = serve(vec![
        ("POST /v1/admin/models/m1/check", 200, FITS),
        ("POST /v1/admin/models/m1/load", 200, "model 'm1' loaded"),
        (
            "GET /metrics",
            200,
            "rustedvino_model_load_duration_seconds{model=\"m1\",device=\"CPU\",kind=\"text_gen\"} 99.0\n\
rustedvino_model_load_duration_seconds{model=\"m1\",device=\"GPU\",kind=\"text_gen\"} 7.25\n",
        ),
        (
            "GET /v1/admin/models",
            200,
            r#"{"models":[{"id":"m1","state":"not_loaded","device":"GPU","kind":"text_gen"}]}"#,
        ),
    ]);
    let r = rv(s.port, Some("adm"), &["load", "m1", "--force"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(
        r.out
            .contains("the pre-load check says this load evicts nothing"),
        "{}",
        r.out
    );
    assert!(!r.out.contains("request(s) running"), "{}", r.out);
    assert!(r.out.contains(" s on GPU (engine 7.2 s)"), "{}", r.out);

    let up = serve(vec![
        ("POST /v1/admin/models/m1/load", 200, "model 'm1' loaded"),
        ("GET /v1/admin/models", 200, ADMIN_MODELS),
    ]);
    let r = rv(up.port, Some("adm"), &["load", "m1", "--no-check"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(r.out.contains("'m1' is already loaded"), "{}", r.out);
    assert!(!r.out.contains("loaded in "), "{}", r.out);
}

/// `m1` registered but not loaded, on `GPU.1` / `GPU` (the `rv load` time tests).
const NOT_LOADED_GPU1: &str =
    r#"{"models":[{"id":"m1","state":"not_loaded","device":"GPU.1","kind":"text_gen"}]}"#;
const NOT_LOADED_GPU: &str =
    r#"{"models":[{"id":"m1","state":"not_loaded","device":"GPU","kind":"text_gen"}]}"#;

// ---- round RVS: model tables by kind, then id; `--json` keeps the server's order ----

/// Every kind, an unknown (`custom`) and a missing one, in a shuffled (server hash-map) order;
/// `Alpaca` / `alpaca` tie ignoring case. The ids are not kind names, so no cell is mistaken
/// for an id.
const SHUFFLED: &str = r#"{"models":[
  {"id":"pick","state":"ready","device":"GPU","kind":"decision","vram_gb":1.0},
  {"id":"vec","state":"ready","device":"GPU","kind":"embedding","vram_gb":1.0},
  {"id":"bare","state":"ready","device":"GPU","vram_gb":1.0},
  {"id":"voice","state":"ready","device":"GPU","kind":"tts","vram_gb":1.0},
  {"id":"Zephyr","state":"ready","device":"GPU","kind":"text_gen","vram_gb":1.0},
  {"id":"ear","state":"ready","device":"GPU","kind":"stt","vram_gb":1.0},
  {"id":"odd","state":"ready","device":"GPU","kind":"custom","vram_gb":1.0},
  {"id":"alpaca","state":"ready","device":"GPU","kind":"text_gen","vram_gb":1.0},
  {"id":"paint","state":"ready","device":"GPU","kind":"image_gen","vram_gb":1.0},
  {"id":"lens","state":"ready","device":"GPU","kind":"vision","vram_gb":1.0},
  {"id":"judge","state":"ready","device":"GPU","kind":"reranking","vram_gb":1.0},
  {"id":"Alpaca","state":"ready","device":"GPU","kind":"text_gen","vram_gb":1.0}],
  "server":{"pid":4242424,"uptime_secs":10}}"#;
const SHUFFLED_ORDER: [&str; 12] = [
    "pick", "vec", "bare", "voice", "Zephyr", "ear", "odd", "alpaca", "paint", "lens", "judge",
    "Alpaca",
];
const SORTED_ORDER: [&str; 12] = [
    "Alpaca", "alpaca", "Zephyr", "lens", "vec", "judge", "ear", "voice", "paint", "pick", "odd",
    "bare",
];

/// The lines' first tokens that are one of `ids`, in output order.
fn first_tokens<'a>(out: &str, ids: &[&'a str]) -> Vec<&'a str> {
    out.lines()
        .filter_map(|l| l.split_whitespace().next())
        .filter_map(|t| ids.iter().find(|i| **i == t).copied())
        .collect()
}

/// The `id` of each object in `v`, in order.
fn json_ids(v: &serde_json::Value) -> Vec<String> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|m| m["id"].as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn list_sorts_registered_by_kind_then_id_and_on_disk_alphabetically() {
    let s = serve(vec![("GET /v1/admin/models", 200, SHUFFLED)]);
    let dir = tempfile::tempdir().expect("tempdir");
    for d in ["zulu", "Bravo", "alpha", "bravo", "pick"] {
        std::fs::create_dir(dir.path().join(d)).expect("mkdir");
    }
    let r = rv_models_dir(s.port, dir.path(), &["list"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert_eq!(
        first_tokens(&r.out, &SHUFFLED_ORDER),
        SORTED_ORDER,
        "{}",
        r.out
    );
    let disk = ["zulu", "Bravo", "alpha", "bravo"];
    assert_eq!(
        first_tokens(&r.out, &disk),
        ["alpha", "Bravo", "bravo", "zulu"],
        "{}",
        r.out
    );
    let j = rv_models_dir(s.port, dir.path(), &["list", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&j.out).expect("json");
    assert_eq!(json_ids(&v["registered"]), SHUFFLED_ORDER, "server order");
    // `on_disk` is unchanged too: the byte order `list_model_dirs` has always given.
    assert_eq!(
        v["on_disk"],
        serde_json::json!(["Bravo", "alpha", "bravo", "pick", "zulu"])
    );
}

#[test]
fn ps_sorts_by_kind_then_id_and_json_keeps_the_server_order() {
    let s = serve(vec![
        ("GET /v1/admin/models", 200, SHUFFLED),
        ("GET /metrics", 200, ""),
    ]);
    let r = rv(s.port, Some("adm"), &["ps"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert_eq!(
        first_tokens(&r.out, &SHUFFLED_ORDER),
        SORTED_ORDER,
        "{}",
        r.out
    );
    let j = rv(s.port, Some("adm"), &["ps", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&j.out).expect("json");
    assert_eq!(json_ids(&v), SHUFFLED_ORDER, "server order");
}

#[test]
fn completeness_sorts_by_kind_then_id_and_json_keeps_the_server_order() {
    let body: &'static str = Box::leak(
        format!(
            r#"{{"models":[{}],"incomplete_count":0}}"#,
            serde_json::from_str::<serde_json::Value>(SHUFFLED).expect("fixture")["models"]
                .as_array()
                .expect("models")
                .iter()
                .map(|m| {
                    let mut e = serde_json::json!({"id": m["id"], "complete": true});
                    if let Some(k) = m.get("kind") {
                        e["kind"] = k.clone();
                    }
                    e.to_string()
                })
                .collect::<Vec<_>>()
                .join(",")
        )
        .into_boxed_str(),
    );
    let s = serve(vec![("GET /v1/admin/models/completeness", 200, body)]);
    let r = rv(s.port, Some("adm"), &["completeness"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert_eq!(
        first_tokens(&r.out, &SHUFFLED_ORDER),
        SORTED_ORDER,
        "{}",
        r.out
    );
    assert_eq!(
        s.hits.lock().expect("lock").as_slice(),
        ["GET /v1/admin/models/completeness"],
        "the entries carry their kind: no admin-list GET"
    );
    let j = rv(s.port, Some("adm"), &["completeness", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&j.out).expect("json");
    assert_eq!(json_ids(&v["models"]), SHUFFLED_ORDER, "server order");
}

/// `/v1/models` in the shuffled order, as the server lists it.
fn shuffled_v1_models() -> &'static str {
    let data: Vec<serde_json::Value> = SHUFFLED_ORDER
        .iter()
        .map(|id| serde_json::json!({"id": id}))
        .collect();
    Box::leak(
        serde_json::json!({ "data": data })
            .to_string()
            .into_boxed_str(),
    )
}

#[test]
fn status_models_sort_by_kind_from_the_admin_list_else_by_id() {
    let mut routes = status_routes();
    for (k, _, b) in &mut routes {
        match *k {
            "GET /v1/models" => *b = shuffled_v1_models(),
            "GET /v1/admin/models" => *b = SHUFFLED,
            "GET /metrics" => *b = "",
            _ => {}
        }
    }
    let s = serve(routes.clone());
    let r = rv(s.port, Some("adm"), &["status"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert_eq!(
        first_tokens(&r.out, &SHUFFLED_ORDER),
        SORTED_ORDER,
        "{}",
        r.out
    );
    let j = rv(s.port, Some("adm"), &["status", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&j.out).expect("json");
    assert_eq!(
        json_ids(&v["models"]["data"]),
        SHUFFLED_ORDER,
        "server order"
    );
    assert_eq!(
        json_ids(&v["admin_models"]["models"]),
        SHUFFLED_ORDER,
        "server order"
    );

    // No admin list: no kinds, so alphabetically by id.
    for (k, st, _) in &mut routes {
        if *k == "GET /v1/admin/models" {
            *st = 500;
        }
    }
    let r = rv(serve(routes).port, Some("adm"), &["status"]);
    let mut alpha = SHUFFLED_ORDER;
    alpha.sort_by(|a, b| {
        a.to_lowercase()
            .cmp(&b.to_lowercase())
            .then_with(|| a.cmp(b))
    });
    assert_eq!(first_tokens(&r.out, &SHUFFLED_ORDER), alpha, "{}", r.out);
}

/// Round LFO P4 folded into RVS: `m1` moved CPU → `GPU.0`; the CPU series come first.
const METRICS_MOVED_LOAD: &str = "\
rustedvino_model_loaded{model=\"m1\",device=\"CPU\",kind=\"text_gen\"} 0\n\
rustedvino_model_loaded{model=\"m1\",device=\"GPU.0\",kind=\"text_gen\"} 1\n\
rustedvino_kv_cache_usage_percent{model=\"m1\",device=\"CPU\",kind=\"text_gen\"} 81\n\
rustedvino_kv_cache_usage_percent{model=\"m1\",device=\"GPU.0\",kind=\"text_gen\"} 12\n\
rustedvino_requests_running{model=\"m1\",device=\"CPU\",kind=\"text_gen\"} 0\n\
rustedvino_requests_waiting{model=\"m1\",device=\"CPU\",kind=\"text_gen\"} 0\n\
rustedvino_requests_max{model=\"m1\",device=\"CPU\",kind=\"text_gen\"} 4\n\
rustedvino_requests_running{model=\"m1\",device=\"GPU.0\",kind=\"text_gen\"} 3\n\
rustedvino_requests_waiting{model=\"m1\",device=\"GPU.0\",kind=\"text_gen\"} 1\n\
rustedvino_requests_max{model=\"m1\",device=\"GPU.0\",kind=\"text_gen\"} 16\n";

#[test]
fn ps_and_status_load_cells_read_the_current_device() {
    let admin = r#"{"models":[{"id":"m1","state":"ready","device":"GPU.0","kind":"text_gen","vram_gb":5.0}],
      "server":{"pid":4242424,"uptime_secs":10}}"#;
    let s = serve(vec![
        ("GET /v1/admin/models", 200, admin),
        ("GET /metrics", 200, METRICS_MOVED_LOAD),
    ]);
    let r = rv(s.port, Some("adm"), &["ps"]);
    assert_eq!(r.code, 0, "{}", r.err);
    assert!(
        r.out.contains("12%") && r.out.contains("3/1/16"),
        "{}",
        r.out
    );
    assert!(
        !r.out.contains("81%") && !r.out.contains("0/0/4"),
        "{}",
        r.out
    );

    let mut routes = status_routes();
    for (k, _, b) in &mut routes {
        match *k {
            "GET /v1/admin/models" => *b = admin,
            "GET /metrics" => *b = METRICS_MOVED_LOAD,
            _ => {}
        }
    }
    let r = rv(serve(routes).port, Some("adm"), &["status"]);
    assert_eq!(r.code, 0, "{}", r.err);
    let row = r
        .out
        .lines()
        .find(|l| l.split_whitespace().next() == Some("m1"))
        .unwrap_or_default();
    assert!(row.contains("12%") && row.contains("3/1/16"), "{}", r.out);
}
