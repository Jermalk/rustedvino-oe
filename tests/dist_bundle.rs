//! C8 (0.9.1): the Linux release bundle carries `templates/` (the corrected LFM2 chat template
//! CONFIG.md documents at `templates/lfm2.jinja`). 0.9.0 shipped the doc without the file, so an
//! operator following CONFIG.md pointed `chat_template` at nothing. Both bundle builders
//! (`scripts/make-dist.sh`, `scripts/dist-linux-container-build.sh`) copy the templates through
//! `scripts/dist-templates.sh`; this test runs that step into a scratch bundle and fails if a
//! builder stops calling it or the step stops shipping a documented template.
//!
//! The dist scripts are never exported to the public tree, so these tests only run where they are
//! present (the same guard as `shipped_configs.rs`).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// `None` in a tree without the dist scripts (the public export).
fn dist_helper() -> Option<PathBuf> {
    let helper = repo().join("scripts").join("dist-templates.sh");
    (helper.is_file() && repo().join("scripts").join("make-dist.sh").is_file()).then_some(helper)
}

/// A scratch directory under the build's own target dir, never the system temp dir.
fn scratch() -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix("dist-bundle-")
        .tempdir_in(env!("CARGO_TARGET_TMPDIR"))
        .unwrap()
}

fn run_helper(helper: &Path, bundle: &Path, src: &Path) -> std::process::Output {
    Command::new("bash")
        .arg(helper)
        .arg(bundle)
        .arg(src)
        .output()
        .unwrap()
}

#[test]
fn both_bundle_builders_ship_the_templates() {
    if dist_helper().is_none() {
        return;
    }
    for script in ["make-dist.sh", "dist-linux-container-build.sh"] {
        let text = std::fs::read_to_string(repo().join("scripts").join(script)).unwrap();
        let calls = text
            .lines()
            .filter(|l| !l.trim_start().starts_with('#'))
            .any(|l| l.contains("scripts/dist-templates.sh"));
        assert!(calls, "{script} no longer runs scripts/dist-templates.sh");
    }
}

#[test]
fn the_bundle_step_ships_every_template_config_md_names() {
    let Some(helper) = dist_helper() else {
        return;
    };
    let out_dir = scratch();
    let bundle = out_dir.path().join("rustedvino-linux-x86_64-test");
    std::fs::create_dir_all(&bundle).unwrap();
    let out = run_helper(&helper, &bundle, &repo());
    assert!(
        out.status.success(),
        "dist-templates.sh failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let shipped = bundle.join("templates").join("lfm2.jinja");
    assert_eq!(
        std::fs::read(&shipped).unwrap(),
        std::fs::read(repo().join("templates").join("lfm2.jinja")).unwrap(),
        "the bundle's templates/lfm2.jinja differs from the repo's"
    );
    // Every `templates/<file>.jinja` the shipped CONFIG.md names is in the bundle.
    let config_md = std::fs::read_to_string(repo().join("CONFIG.md")).unwrap();
    let mut named = 0;
    for (i, _) in config_md.match_indices("templates/") {
        let name: String = config_md[i..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || "/_.-".contains(*c))
            .collect();
        let name = name.trim_end_matches('.');
        if Path::new(name)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("jinja"))
        {
            named += 1;
            assert!(bundle.join(name).is_file(), "CONFIG.md names {name}");
        }
    }
    assert!(named >= 1, "CONFIG.md no longer names templates/lfm2.jinja");
}

/// The step aborts (non-zero) when CONFIG.md documents a template the source tree does not have,
/// rather than packaging a bundle whose doc points at nothing (the 0.9.0 defect).
#[test]
fn the_bundle_step_refuses_a_documented_template_that_is_missing() {
    let Some(helper) = dist_helper() else {
        return;
    };
    let dir = scratch();
    let src = dir.path().join("src");
    std::fs::create_dir_all(src.join("templates")).unwrap();
    std::fs::write(src.join("templates").join("other.jinja"), "{{ x }}").unwrap();
    std::fs::write(
        src.join("CONFIG.md"),
        "A corrected template ships at `templates/lfm2.jinja`.\n",
    )
    .unwrap();
    let bundle = dir.path().join("bundle");
    std::fs::create_dir_all(&bundle).unwrap();
    let out = run_helper(&helper, &bundle, &src);
    assert!(
        !out.status.success(),
        "a missing documented template passed"
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("templates/lfm2.jinja"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // And a source tree with no templates/ at all is refused too.
    std::fs::remove_dir_all(src.join("templates")).unwrap();
    let out = run_helper(&helper, &bundle, &src);
    assert!(!out.status.success(), "a tree without templates/ passed");
}
