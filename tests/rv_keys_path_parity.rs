//! `tools/rv` cannot depend on this crate (it would link `OpenVINO`), so it carries
//! a copy of `resolve_keys_file_path`. This test fails if the two ever disagree.

use std::path::Path;

use rustedvino::model_manager::config::resolve_keys_file_path as server;

#[test]
fn rv_and_server_resolve_the_same_keys_file() {
    let configs = [
        "/opt/rustedvino/config.json",
        "scripts/myserver.config.json",
        "scripts/plain.json",
        "/etc/rv/no-extension",
        "config.json",
        "C:\\RustedVINO\\check07\\config.json",
        "/a/b/c.config.json",
        "/a/b/.json",
    ];
    let explicits = [
        None,
        Some("keys.json"),
        Some("sub/keys.json"),
        Some("/abs/keys.json"),
    ];
    for cfg in configs {
        for ex in explicits {
            let ex = ex.map(Path::new);
            assert_eq!(
                server(Path::new(cfg), ex),
                rv::resolve::resolve_keys_file_path(Path::new(cfg), ex),
                "disagreement for config {cfg:?}, explicit {ex:?}"
            );
        }
    }
}

#[test]
fn rv_and_server_agree_on_the_per_user_config_path() {
    use std::ffi::OsStr;
    for base in [
        Some(OsStr::new("C:\\Users\\someone\\AppData\\Local")),
        Some(OsStr::new("/home/x/.local")),
        Some(OsStr::new("")),
        None,
    ] {
        assert_eq!(
            rustedvino::startup::per_user_config_path(base),
            rv::resolve::per_user_config_path(base),
            "disagreement for {base:?}"
        );
    }
}
