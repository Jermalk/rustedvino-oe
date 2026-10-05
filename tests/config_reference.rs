//! `CONFIG.md` is the one place a setting's meaning is written down (the JSON
//! files cannot carry comments), so it must cover every setting the server
//! accepts. The lists come from the server itself; a setting added to `Config`
//! fails this test until the reference has a row for it. Global and per-model
//! settings are matched in their OWN tables: a `device` row in the global table
//! does not satisfy the per-model `device`.

#![allow(clippy::unwrap_used)]

use rustedvino::model_manager::Config;

const GLOBAL_TABLE: &str = "**`Config` fields**";
const MODEL_TABLE: &str = "**Per-model fields**";

fn reference() -> String {
    std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("CONFIG.md"))
        .unwrap()
}

/// The text between two markers (start inclusive, end exclusive).
fn section<'a>(doc: &'a str, start: &str, end: &str) -> &'a str {
    let a = doc
        .find(start)
        .unwrap_or_else(|| panic!("CONFIG.md lost its {start:?} heading"));
    let b = doc[a..].find(end).map_or(doc.len(), |i| a + i);
    &doc[a..b]
}

/// Whether `table` has a row whose first cell is `` `name` ``.
fn has_row(table: &str, name: &str) -> bool {
    table
        .lines()
        .any(|l| l.starts_with(&format!("| `{name}` |")))
}

#[test]
fn every_global_and_per_model_setting_has_a_row_in_its_own_table() {
    let doc = reference();
    let globals = section(&doc, GLOBAL_TABLE, MODEL_TABLE);
    let per_model = section(&doc, MODEL_TABLE, "\n## ");
    let template = Config::default_template().unwrap();
    assert!(
        template.as_object().unwrap().len() >= 40,
        "template suspiciously small"
    );
    for key in template.as_object().unwrap().keys() {
        assert!(
            has_row(globals, key),
            "the global table has no row for `{key}`"
        );
    }
    let model = Config::default_model_template().unwrap();
    assert!(
        model.as_object().unwrap().len() >= 15,
        "model template suspiciously small"
    );
    for key in model.as_object().unwrap().keys() {
        assert!(
            has_row(per_model, key),
            "the per-model table has no row for `{key}`"
        );
    }
}

#[test]
fn nested_settings_are_named_where_they_belong() {
    let doc = reference();
    let globals = section(&doc, GLOBAL_TABLE, MODEL_TABLE);
    // The supervisor sub-fields are described in the paragraph that follows the per-model table.
    let supervisor = section(&doc, "`\"supervisor\"` sub-fields", "\n\n");
    for key in Config::default_template().unwrap()["supervisor"]
        .as_object()
        .unwrap()
        .keys()
    {
        assert!(
            supervisor.contains(&format!("`{key}`")),
            "the supervisor paragraph never names `{key}`"
        );
    }
    // `realtime_defaults` / `realtime_viable_minimum` default to null, so the template cannot list
    // their fields; each must appear in its own row of the global table.
    let row = |name: &str| {
        globals
            .lines()
            .find(|l| l.starts_with(&format!("| `{name}` |")))
            .unwrap()
            .to_owned()
    };
    let defaults = row("realtime_defaults");
    for key in ["stt_model", "llm_model", "tts_model", "embed_model"] {
        assert!(
            defaults.contains(key),
            "the realtime_defaults row never names `{key}`"
        );
    }
    let viable = row("realtime_viable_minimum");
    for key in ["kinds", "min_context_tokens", "require_capabilities"] {
        assert!(
            viable.contains(key),
            "the realtime_viable_minimum row never names `{key}`"
        );
    }
}
