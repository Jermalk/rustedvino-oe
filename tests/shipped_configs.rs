//! Every shipped `scripts/*.config.json` writes out EVERY setting - each global
//! one and each one of every model - instead of leaning on a default the
//! operator cannot see (a hidden setting is one people forget they have). The
//! lists come from the server itself (`Config::default_template` /
//! `default_model_template`), so a setting added to `Config` fails this test
//! until `scripts/fill-config-defaults.py` has been run over the configs.

#![allow(clippy::unwrap_used)]

use rustedvino::model_manager::Config;

/// How many `*.config.json` the private repo ships (the fleet configs plus the examples); the public
/// export ships only the starter. Counted rather than named, so no fleet hostname appears here.
const PRIVATE_REPO_CONFIGS: usize = 7;

#[test]
fn shipped_configs_write_out_every_setting_and_still_load() {
    let globals = Config::default_template().unwrap();
    let model_keys = Config::default_model_template().unwrap();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts");
    let mut checked = 0;
    let mut models_seen = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let is_config = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(".config.json"));
        if !is_config {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let missing: Vec<&String> = globals
            .as_object()
            .unwrap()
            .keys()
            .filter(|k| v.get(k.as_str()).is_none())
            .collect();
        assert!(
            missing.is_empty(),
            "{} does not write out {missing:?}; run scripts/fill-config-defaults.py",
            path.display()
        );
        for (id, stanza) in v["models"].as_object().unwrap() {
            models_seen += 1;
            let absent: Vec<&String> = model_keys
                .as_object()
                .unwrap()
                .keys()
                .filter(|k| stanza.get(k.as_str()).is_none())
                .collect();
            assert!(
                absent.is_empty(),
                "{}: model {id} does not write out {absent:?}; run scripts/fill-config-defaults.py",
                path.display()
            );
        }
        Config::load(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        checked += 1;
    }
    assert!(checked >= 1, "no shipped config found under scripts/");
    // The export script is itself never exported, so only the private repo has it and must show the full set.
    if dir.join("export-public.sh").exists() {
        assert_eq!(
            checked, PRIVATE_REPO_CONFIGS,
            "unexpected set of shipped configs"
        );
        assert!(
            models_seen > 100,
            "the per-model check ran on only {models_seen} stanzas"
        );
    }
}

/// The template must list the settings the docs promise: guards against the
/// generator being pointed at a stale binary or the untemplated list growing
/// silently.
#[test]
fn template_covers_the_expected_settings() {
    let globals = Config::default_template().unwrap();
    let keys = globals.as_object().unwrap();
    for must in [
        "eviction_grace_secs",
        "npu_footprint_factor",
        "supervisor",
        "models",
    ] {
        assert!(keys.contains_key(must), "template lacks {must}");
    }
    for hidden in Config::UNTEMPLATED {
        assert!(
            !keys.contains_key(hidden),
            "{hidden} must stay out of the template"
        );
    }
}

/// The keys template the bundles ship as `config.keys.example.json` must load through the
/// server's own loader and carry no live credential - a placeholder token copied unedited
/// would become a real, publicly known key.
#[test]
fn dist_keys_template_loads_and_holds_no_keys() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("dist.example-keys.json");
    let auth = rustedvino::AuthConfig::load_from_file(&path).unwrap();
    assert!(auth.api_keys.is_empty() && auth.admin_api_keys.is_empty());
}
