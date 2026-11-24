//! Tests of the `rv` sampling module. The resolver tests port the server's
//! (`src/model_manager/sampling.rs` tests, `src/model_manager/template.rs`
//! `generation_defaults_*` tests) row by row; each names its server original.

use super::*;

fn file(t: Option<f64>, p: Option<f64>, k: Option<u64>) -> FileLayer {
    FileLayer {
        temperature: t,
        top_p: p,
        top_k: k,
        do_sample: t.is_some() || p.is_some() || k.is_some(),
        ..FileLayer::default()
    }
}

fn val(j: &Value, key: &str) -> (Option<f64>, String) {
    (
        j[key]["value"].as_f64(),
        j[key]["source"].as_str().unwrap_or("").to_owned(),
    )
}

fn layer(body: &str) -> FileLayer {
    file_layer_from(&serde_json::from_str(body).expect("json"))
}

// ---- resolver: port of sampling.rs tests ----

/// sampling.rs `profile_t_operator_wins_over_file`.
#[test]
fn profile_t_operator_wins_over_file() {
    let op = Operator {
        temperature: Some(0.15),
        ..Operator::default()
    };
    let j = preview_json(&op, &file(Some(0.6), None, None), false);
    assert_eq!(val(&j, "temperature"), (Some(0.15), "config".into()));
}

/// sampling.rs `profile_t_file_when_no_operator`.
#[test]
fn profile_t_file_when_no_operator() {
    let j = preview_json(
        &Operator::default(),
        &file(Some(0.6), Some(0.95), Some(20)),
        false,
    );
    assert_eq!(
        val(&j, "temperature"),
        (Some(0.6), "generation_config.json".into())
    );
    assert_eq!(j["mode"], "sample");
}

/// sampling.rs `profile_t_none_is_greedy_server_default`.
#[test]
fn profile_t_none_is_greedy_server_default() {
    let j = preview_json(&Operator::default(), &FileLayer::default(), false);
    assert_eq!(j["mode"], "greedy");
    assert_eq!(val(&j, "temperature"), (None, "none".into()));
    assert_eq!(
        val(&j, "repetition_penalty"),
        (Some(1.1), "server default".into())
    );
}

/// sampling.rs `profile_top_p_top_k_operator_then_file`.
#[test]
fn profile_top_p_top_k_operator_then_file() {
    let op = Operator {
        temperature: Some(0.7),
        top_p: Some(0.8),
        ..Operator::default()
    };
    let j = preview_json(&op, &file(Some(1.0), Some(0.95), Some(64)), false);
    assert_eq!(val(&j, "top_p"), (Some(0.8), "config".into()));
    assert_eq!(
        j["top_k"],
        json!({"value": 64, "source": "generation_config.json"})
    );
}

/// sampling.rs `profile_t0_blanks_the_file_top_k`.
#[test]
fn profile_t0_blanks_the_file_top_k() {
    let op = Operator {
        temperature: Some(0.0),
        ..Operator::default()
    };
    let j = preview_json(&op, &file(Some(1.0), Some(0.95), Some(64)), false);
    assert_eq!(val(&j, "temperature"), (Some(0.0), "config".into()));
    assert_eq!(val(&j, "top_p"), (None, "none".into()));
    assert_eq!(j["top_k"], json!({"value": null, "source": "none"}));
    assert_eq!(j["mode"], "greedy");
}

/// sampling.rs `profile_file_top_k_only_samples_at_engine_t`.
#[test]
fn profile_file_top_k_only_samples_at_engine_t() {
    let j = preview_json(&Operator::default(), &file(None, None, Some(40)), false);
    assert_eq!(j["mode"], "sample");
    assert_eq!(val(&j, "temperature"), (Some(1.0), "engine default".into()));
    assert_eq!(val(&j, "top_p"), (Some(1.0), "engine default".into()));
}

/// sampling.rs `profile_operator_t_only_on_greedy_file`.
#[test]
fn profile_operator_t_only_on_greedy_file() {
    let op = Operator {
        temperature: Some(0.15),
        ..Operator::default()
    };
    let j = preview_json(&op, &FileLayer::default(), false);
    assert_eq!(j["mode"], "sample");
    assert_eq!(val(&j, "top_p"), (Some(1.0), "engine default".into()));
    assert_eq!(j["top_k"]["value"], Value::Null);
}

/// sampling.rs `penalty_operator_then_file_then_server_default`.
#[test]
fn penalty_operator_then_file_then_server_default() {
    let f = FileLayer {
        repetition_penalty: Some(1.05),
        ..FileLayer::default()
    };
    let j = preview_json(&Operator::default(), &f, false);
    assert_eq!(
        val(&j, "repetition_penalty"),
        (Some(1.05), "generation_config.json".into())
    );
    let op = Operator {
        repetition_penalty: Some(1.0),
        ..Operator::default()
    };
    let j = preview_json(&op, &f, false);
    assert_eq!(
        val(&j, "repetition_penalty"),
        (Some(1.0), "config".into()),
        "an operator 1.0 is an explicit opt-out"
    );
}

/// sampling.rs `presence_frequency_operator_only_and_not_on_npu` + `admin_json_shape`.
#[test]
fn presence_frequency_and_admin_shape() {
    let op = Operator {
        temperature: Some(0.15),
        top_p: Some(0.95),
        presence_penalty: Some(1.5),
        ..Operator::default()
    };
    let j = preview_json(&op, &FileLayer::default(), true);
    assert_eq!(j["mode"], "sample");
    assert_eq!(val(&j, "temperature"), (Some(0.15), "config".into()));
    assert_eq!(j["top_k"], json!({"value": null, "source": "none"}));
    assert_eq!(j["repetition_penalty"]["source"], "server default");
    assert_eq!(val(&j, "presence_penalty"), (Some(1.5), "config".into()));
    assert_eq!(val(&j, "frequency_penalty"), (None, "none".into()));
    assert_eq!(j["tools"], "greedy");
    assert_eq!(
        j["ignored_on_device"],
        json!(["presence_penalty", "frequency_penalty"])
    );
    assert!(j.get("note").is_none());
    assert!(
        preview_json(&op, &FileLayer::default(), false)
            .get("ignored_on_device")
            .is_none()
    );
}

// ---- file layer: port of sampling.rs + template.rs tests ----

/// sampling.rs `file_penalty_gate`.
#[test]
fn file_penalty_gate() {
    for (body, want) in [
        (r#"{"repetition_penalty": 1.0}"#, None),
        (r#"{"repetition_penalty": 0.9}"#, None),
        (r#"{"repetition_penalty": 1.05}"#, Some(1.05)),
        (r#"{"repetition_penalty": 2.5}"#, None),
        (r#"{"repetition_penalty": "1.2"}"#, None),
        (r#"{"repetition_penalty": "x"}"#, None),
    ] {
        assert_eq!(layer(body).repetition_penalty, want, "{body}");
    }
}

/// sampling.rs `file_penalty_is_read_without_do_sample`.
#[test]
fn file_penalty_is_read_without_do_sample() {
    let f = layer(r#"{"temperature": 0.7, "repetition_penalty": 1.05}"#);
    assert_eq!(f.repetition_penalty, Some(1.05));
    assert!(!f.do_sample);
    assert_eq!((f.temperature, f.top_p, f.top_k), (None, None, None));
}

/// sampling.rs `bad_penalty_keeps_the_package_and_bad_package_keeps_the_penalty`.
#[test]
fn bad_penalty_keeps_the_package_and_bad_package_keeps_the_penalty() {
    let f = layer(r#"{"do_sample": true, "temperature": 0.6, "repetition_penalty": 0.8}"#);
    assert_eq!(f.temperature, Some(0.6));
    assert_eq!(f.repetition_penalty, None);
    let f = layer(r#"{"do_sample": true, "top_p": 7.0, "repetition_penalty": 1.05}"#);
    assert_eq!((f.temperature, f.top_p, f.top_k), (None, None, None));
    assert!(f.do_sample);
    assert_eq!(f.repetition_penalty, Some(1.05));
}

/// sampling.rs `exporter_stub_is_named`.
#[test]
fn exporter_stub_is_named() {
    let f = layer(r#"{"_from_model_config": true, "bos_token_id": 1}"#);
    assert!(f.exporter_stub);
    assert!(preview_json(&Operator::default(), &f, false)["note"].is_string());
}

/// sampling.rs `missing_file_is_an_empty_layer`; plus: a missing directory is no preview.
#[test]
fn missing_file_and_missing_dir() {
    let d = tempfile::tempdir().expect("tempdir");
    assert_eq!(read_file_layer(d.path()), Some(FileLayer::default()));
    std::fs::write(d.path().join("generation_config.json"), "{not json").expect("write");
    assert_eq!(
        read_file_layer(d.path()),
        Some(FileLayer::default()),
        "unparseable"
    );
    std::fs::write(
        d.path().join("generation_config.json"),
        r#"{"do_sample": true, "temperature": 0.6, "top_k": 20}"#,
    )
    .expect("write");
    let f = read_file_layer(d.path()).expect("layer");
    assert_eq!((f.temperature, f.top_k), (Some(0.6), Some(20)));
    assert_eq!(read_file_layer(&d.path().join("nope")), None);
}

/// template.rs `generation_defaults_*`, one row each.
#[test]
fn do_sample_package_gate() {
    type Pkg = (Option<f64>, Option<f64>, Option<u64>);
    let rows: [(&str, &str, Pkg); 12] = [
        (
            "do_sample_absent_is_all_none",
            r#"{"temperature": 0.7, "top_p": 0.8, "top_k": 20}"#,
            (None, None, None),
        ),
        (
            "do_sample_false_with_stale_fields_is_all_none",
            r#"{"do_sample": false, "temperature": 0.7, "top_p": 0.8, "top_k": 20}"#,
            (None, None, None),
        ),
        (
            "do_sample_true_full_package_is_parsed",
            r#"{"do_sample": true, "temperature": 0.7, "top_p": 0.8, "top_k": 20}"#,
            (Some(0.7), Some(0.8), Some(20)),
        ),
        (
            "do_sample_true_no_numeric_fields_is_all_none",
            r#"{"do_sample": true}"#,
            (None, None, None),
        ),
        (
            "top_k_zero_is_treated_as_unset",
            r#"{"do_sample": true, "temperature": 0.7, "top_k": 0}"#,
            (Some(0.7), None, None),
        ),
        (
            "temperature_zero_contradicts_do_sample_drops_whole_package",
            r#"{"do_sample": true, "temperature": 0.0, "top_p": 0.8}"#,
            (None, None, None),
        ),
        (
            "out_of_range_top_p_drops_whole_package",
            r#"{"do_sample": true, "temperature": 0.7, "top_p": 1.5}"#,
            (None, None, None),
        ),
        (
            "null_temperature_drops_whole_package",
            r#"{"do_sample": true, "temperature": null, "top_p": 0.8, "top_k": 20}"#,
            (None, None, None),
        ),
        (
            "string_typed_top_p_drops_whole_package",
            r#"{"do_sample": true, "temperature": 0.7, "top_p": "0.8"}"#,
            (None, None, None),
        ),
        (
            "top_k_float_literal_whole_number_is_parsed",
            r#"{"do_sample": true, "temperature": 0.7, "top_k": 20.0}"#,
            (Some(0.7), None, Some(20)),
        ),
        (
            "negative_top_k_drops_whole_package",
            r#"{"do_sample": true, "temperature": 0.7, "top_k": -5}"#,
            (None, None, None),
        ),
        (
            "fractional_top_k_drops_whole_package",
            r#"{"do_sample": true, "temperature": 0.7, "top_k": 20.5}"#,
            (None, None, None),
        ),
    ];
    for (name, body, want) in rows {
        let f = layer(body);
        assert_eq!((f.temperature, f.top_p, f.top_k), want, "{name}");
    }
}

/// F6: a `do_sample` file without a package temperature counts as no `do_sample`.
#[test]
fn do_sample_without_temperature_does_not_sample_with_temperature() {
    assert!(!layer(r#"{"do_sample": true, "top_k": 40}"#).samples_with_temperature());
    assert!(!layer(r#"{"do_sample": true, "temperature": 9.0}"#).samples_with_temperature());
    assert!(layer(r#"{"do_sample": true, "temperature": 0.6}"#).samples_with_temperature());
    let op = Operator {
        top_p: Some(0.9),
        ..Operator::default()
    };
    assert!(op.check_do_sample(false).is_err());
    assert!(op.check_do_sample(true).is_ok());
}

// ---- flags: ranges, conflicts, merge ----

#[test]
fn flag_ranges_mirror_the_server() {
    let bad = [
        Operator {
            temperature: Some(2.5),
            ..Operator::default()
        },
        Operator {
            temperature: Some(-0.1),
            ..Operator::default()
        },
        Operator {
            top_p: Some(0.0),
            ..Operator::default()
        },
        Operator {
            top_p: Some(1.01),
            ..Operator::default()
        },
        Operator {
            top_k: Some(0),
            ..Operator::default()
        },
        Operator {
            repetition_penalty: Some(0.9),
            ..Operator::default()
        },
        Operator {
            repetition_penalty: Some(2.1),
            ..Operator::default()
        },
        Operator {
            presence_penalty: Some(-2.5),
            ..Operator::default()
        },
        Operator {
            frequency_penalty: Some(2.5),
            ..Operator::default()
        },
        Operator {
            temperature: Some(f64::NAN),
            ..Operator::default()
        },
        Operator {
            top_p: Some(f64::INFINITY),
            ..Operator::default()
        },
        Operator {
            temperature: Some(0.0),
            top_k: Some(20),
            ..Operator::default()
        },
        Operator {
            temperature: Some(0.0),
            top_p: Some(0.9),
            ..Operator::default()
        },
    ];
    for op in bad {
        assert!(op.validate().is_err(), "{op:?}");
    }
    let good = [
        Operator {
            temperature: Some(0.0),
            ..Operator::default()
        },
        Operator {
            temperature: Some(2.0),
            top_p: Some(1.0),
            top_k: Some(1),
            ..Operator::default()
        },
        Operator {
            repetition_penalty: Some(1.0),
            ..Operator::default()
        },
        Operator {
            repetition_penalty: Some(2.0),
            ..Operator::default()
        },
        Operator {
            presence_penalty: Some(-2.0),
            frequency_penalty: Some(2.0),
            ..Operator::default()
        },
    ];
    for op in good {
        assert!(op.validate().is_ok(), "{op:?}");
    }
    let e = Operator {
        presence_penalty: Some(3.0),
        ..Operator::default()
    }
    .validate()
    .unwrap_err();
    assert!(e.contains("--presence-penalty"), "{e}");
}

/// An admin `sampling` field with an operator `top_p` 0.8 and `presence` 1.5 over a file T 0.6.
fn admin_with_operator() -> Value {
    json!({
        "mode": "sample",
        "temperature": {"value": 0.6, "source": "generation_config.json"},
        "top_p": {"value": 0.8, "source": "config"},
        "top_k": {"value": 20, "source": "generation_config.json"},
        "repetition_penalty": {"value": 1.1, "source": "server default"},
        "presence_penalty": {"value": 1.5, "source": "config"},
        "frequency_penalty": {"value": null, "source": "none"},
        "tools": "greedy"
    })
}

#[test]
fn operator_values_are_the_config_sourced_ones() {
    let op = Operator::from_admin(&admin_with_operator());
    assert_eq!(
        op,
        Operator {
            top_p: Some(0.8),
            presence_penalty: Some(1.5),
            ..Operator::default()
        }
    );
    assert_eq!(op.summary(), "top_p 0.8 · presence_penalty 1.5");
    assert_eq!(Operator::default().summary(), "none");
}

#[test]
fn merge_keeps_operator_values_and_the_flag_wins() {
    let flags = SamplingFlags {
        temperature: Some(0.7),
        presence_penalty: Some(0.5),
        ..SamplingFlags::default()
    };
    let (before, after, body) = merge_for_patch(&admin_with_operator(), &flags, false).expect("ok");
    assert_eq!(before.summary(), "top_p 0.8 · presence_penalty 1.5");
    assert_eq!(after.summary(), "T 0.7 · top_p 0.8 · presence_penalty 0.5");
    assert_eq!(
        body,
        json!({"temperature": 0.7, "top_p": 0.8, "presence_penalty": 0.5})
    );
    // Key order is the config's: temperature first.
    assert_eq!(
        serde_json::to_string(&body).expect("text"),
        r#"{"temperature":0.7,"top_p":0.8,"presence_penalty":0.5}"#
    );
}

#[test]
fn merge_clear_sends_null() {
    let (before, after, body) =
        merge_for_patch(&admin_with_operator(), &SamplingFlags::default(), true).expect("ok");
    assert_eq!(body, Value::Null);
    assert!(after.is_empty());
    assert!(!before.is_empty());
}

#[test]
fn merge_checks_the_merged_object() {
    // Operator top_p 0.8 + --temperature 0: the server would 400.
    let flags = SamplingFlags {
        temperature: Some(0.0),
        ..SamplingFlags::default()
    };
    let e = merge_for_patch(&admin_with_operator(), &flags, false).unwrap_err();
    assert!(e.contains("temperature 0"), "{e}");
    assert!(e.contains("--clear-sampling"), "{e}");
}

#[test]
fn merge_infers_the_do_sample_rule_from_the_profile() {
    // File T present (source generation_config.json): top_p alone is fine.
    let flags = SamplingFlags {
        top_k: Some(40),
        ..SamplingFlags::default()
    };
    assert!(merge_for_patch(&admin_with_operator(), &flags, false).is_ok());
    // Greedy file: top_p/top_k without a temperature are refused.
    let greedy = json!({
        "mode": "greedy",
        "temperature": {"value": null, "source": "none"},
        "top_p": {"value": null, "source": "none"},
        "top_k": {"value": null, "source": "none"},
        "repetition_penalty": {"value": 1.1, "source": "server default"},
        "presence_penalty": {"value": null, "source": "none"},
        "frequency_penalty": {"value": null, "source": "none"},
        "tools": "greedy"
    });
    let e = merge_for_patch(&greedy, &flags, false).unwrap_err();
    assert!(e.contains("do_sample"), "{e}");
}

#[test]
fn kind_check_comes_before_the_old_server_warning() {
    let embed = json!({"id": "e", "kind": "embedding"});
    assert_eq!(entry_profile(&embed), EntryProfile::NotApplicable);
    let old = json!({"id": "m", "kind": "text_gen"});
    assert_eq!(entry_profile(&old), EntryProfile::OldServer);
    let old_vlm = json!({"id": "v", "kind": "vision"});
    assert_eq!(entry_profile(&old_vlm), EntryProfile::OldServer);
    let new = json!({"id": "m", "kind": "text_gen", "sampling": admin_with_operator()});
    assert!(matches!(entry_profile(&new), EntryProfile::Profile(_)));
}

// ---- formatter goldens: one per plan example line, hint and footer ----

const FOOTER: &str = "values the client sends win · realtime: profile";

/// Plan example 1: a greedy model (no `do_sample` in the file), with its hint (wording F9).
#[test]
fn golden_greedy_with_hint() {
    let j = preview_json(&Operator::default(), &FileLayer::default(), false);
    let p = render("mistral", &j, Origin::Server);
    assert_eq!(
        p.line,
        "sampling: greedy · repetition_penalty 1.1 (server default)"
    );
    assert_eq!(
        p.hint.as_deref(),
        Some(
            "hint: generation_config.json gives no sampling settings, so this model runs greedy; \
             the model card may recommend sampling: rv patch mistral --temperature 0.15 --top-p 0.95"
        )
    );
    assert_eq!(p.footer, FOOTER);
    assert_eq!(p.note, None);
}

/// Plan example 2: the file's whole package, one shared label.
#[test]
fn golden_file_package() {
    let j = preview_json(
        &Operator::default(),
        &file(Some(1.0), Some(0.95), Some(64)),
        false,
    );
    let p = render("gemma", &j, Origin::Server);
    assert_eq!(
        p.line,
        "sampling: T 1.0 · top_p 0.95 · top_k 64 (generation_config.json) · repetition_penalty 1.1 \
         (server default) · tools → greedy"
    );
    assert_eq!(p.hint, None);
}

/// Plan example 3: engine defaults around a file `top_k` (fixed order T, `top_p`, `top_k`).
#[test]
fn golden_engine_defaults() {
    let j = preview_json(&Operator::default(), &file(None, None, Some(40)), false);
    let p = render("m", &j, Origin::Server);
    assert_eq!(
        p.line,
        "sampling: T 1.0 (engine default) · top_p 1.0 (engine default) · top_k 40 \
         (generation_config.json) · repetition_penalty 1.1 (server default) · tools → greedy"
    );
    assert_eq!(p.hint, None);
}

/// Plan example 4: operator T + `top_p`, no `top_k`.
#[test]
fn golden_operator_values() {
    let op = Operator {
        temperature: Some(0.15),
        top_p: Some(0.95),
        ..Operator::default()
    };
    let p = render(
        "m",
        &preview_json(&op, &FileLayer::default(), false),
        Origin::Server,
    );
    assert_eq!(
        p.line,
        "sampling: T 0.15 (config) · top_p 0.95 (config) · top_k — none · repetition_penalty 1.1 \
         (server default) · tools → greedy"
    );
    assert_eq!(p.hint, None);
}

/// Operator T without `top_p`/`top_k`: the full-vocabulary warning.
#[test]
fn golden_full_vocabulary_warning() {
    let op = Operator {
        temperature: Some(0.15),
        ..Operator::default()
    };
    let p = render(
        "m",
        &preview_json(&op, &FileLayer::default(), false),
        Origin::Server,
    );
    assert_eq!(
        p.line,
        "sampling: T 0.15 (config) · top_p 1.0 (engine default) · top_k — none · \
         repetition_penalty 1.1 (server default) · tools → greedy"
    );
    assert_eq!(
        p.hint.as_deref(),
        Some("warn: samples over the full vocabulary; consider --top-p")
    );
    assert!(p.hint_is_warning());
}

/// Exporter stub note, a file penalty (qwen3-coder 1.05), the local label.
#[test]
fn golden_stub_note_file_penalty_and_local_label() {
    let f = layer(r#"{"_from_model_config": true, "repetition_penalty": 1.05}"#);
    let p = render(
        "qwen",
        &preview_json(&Operator::default(), &f, false),
        Origin::Local { file_found: true },
    );
    assert_eq!(
        p.line,
        "sampling (local preview, text_gen/vision): greedy · repetition_penalty 1.05 \
         (generation_config.json)"
    );
    assert_eq!(
        p.note.as_deref(),
        Some("note: generation_config.json is an exporter stub (no sampling recommendation)")
    );
    assert!(
        p.hint.as_deref().is_some_and(|h| h.starts_with("hint:")
            && h.ends_with("rv add qwen --temperature 0.15 --top-p 0.95")),
        "a local preview's hint names rv add: {:?}",
        p.hint
    );
    assert_eq!(
        p.lines(),
        vec![
            p.line.clone(),
            "  note: generation_config.json is an exporter stub (no sampling recommendation)"
                .to_owned(),
            format!("  {}", p.hint.clone().unwrap_or_default()),
            format!("  {FOOTER}"),
        ]
    );
}

/// Operator penalties on NPU, and the footer's greedy rule.
#[test]
fn golden_penalties_npu_and_footer() {
    let op = Operator {
        temperature: Some(0.7),
        top_p: Some(0.8),
        top_k: Some(20),
        repetition_penalty: Some(1.0),
        presence_penalty: Some(1.5),
        ..Operator::default()
    };
    let p = render(
        "qwen",
        &preview_json(&op, &FileLayer::default(), true),
        Origin::Server,
    );
    assert_eq!(
        p.line,
        "sampling: T 0.7 (config) · top_p 0.8 (config) · top_k 20 (config) · repetition_penalty \
         1.0 (config) · presence_penalty 1.5 (config, ignored on this device) · tools → greedy"
    );
    assert_eq!(
        p.footer,
        "values the client sends win · realtime: profile · presence/frequency are not sent on a \
         greedy request (tools, or a client T 0)"
    );
}

/// Operator T 0: greedy, the T shown with its source, no hint.
#[test]
fn golden_operator_greedy() {
    let op = Operator {
        temperature: Some(0.0),
        ..Operator::default()
    };
    let p = render(
        "m",
        &preview_json(&op, &FileLayer::default(), false),
        Origin::Server,
    );
    assert_eq!(
        p.line,
        "sampling: greedy · T 0.0 (config) · repetition_penalty 1.1 (server default)"
    );
    assert_eq!(p.hint, None);
}

/// The formatter reads a server-made admin field the same way (shape from INTERFACE.md).
#[test]
fn golden_server_field() {
    let p = render("qwen", &admin_with_operator(), Origin::Server);
    assert_eq!(
        p.line,
        "sampling: T 0.6 (generation_config.json) · top_p 0.8 (config) · top_k 20 \
         (generation_config.json) · repetition_penalty 1.1 (server default) · presence_penalty \
         1.5 (config) · tools → greedy"
    );
}

/// Review F9: a local preview of a directory without a readable file says so in the hint.
#[test]
fn golden_hint_names_a_missing_file_in_a_local_preview() {
    let j = preview_json(&Operator::default(), &FileLayer::default(), false);
    let p = render("ministral", &j, Origin::Local { file_found: false });
    assert_eq!(
        p.hint.as_deref(),
        Some(
            "hint: no readable generation_config.json, so this model runs greedy; the model card \
             may recommend sampling: rv add ministral --temperature 0.15 --top-p 0.95"
        )
    );
}

/// Review F6: a merged `do_sample` failure names the values the user did not pass.
#[test]
fn merge_do_sample_error_names_the_current_operator_values() {
    let current = json!({
        "mode": "sample",
        "temperature": {"value": 1.0, "source": "engine default"},
        "top_p": {"value": 0.8, "source": "config"},
        "top_k": {"value": null, "source": "none"},
        "repetition_penalty": {"value": 1.1, "source": "server default"},
        "presence_penalty": {"value": null, "source": "none"},
        "frequency_penalty": {"value": null, "source": "none"},
        "tools": "greedy"
    });
    let flags = SamplingFlags {
        presence_penalty: Some(0.5),
        ..SamplingFlags::default()
    };
    let e = merge_for_patch(&current, &flags, false).unwrap_err();
    assert!(e.contains("current operator values: top_p 0.8"), "{e}");
    assert!(e.contains("--clear-sampling"), "{e}");
    assert!(!e.contains("drop --top-p"), "{e}");
}

/// Review F11: the shared fixture `tests/fixtures/sampling_profiles.json` ({file, operator, npu}
/// → the admin `sampling` JSON) holds for the mirror. The server asserts the same file against
/// its real resolver (`src/model_manager/sampling.rs` `shared_fixture_matches_the_real_resolver`),
/// so the mirror and the server cannot drift apart on these cases unnoticed.
#[test]
fn shared_fixture_matches_the_mirror() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("../../tests/fixtures/sampling_profiles.json"))
            .expect("fixture");
    assert!(cases.len() >= 10);
    for c in &cases {
        let file = c["generation_config"]
            .as_object()
            .map_or_else(FileLayer::default, |_| {
                file_layer_from(&c["generation_config"])
            });
        let op = fixture_operator(&c["operator"]);
        assert_eq!(
            preview_json(&op, &file, c["npu"].as_bool().expect("npu")),
            c["expected"],
            "{}",
            c["name"]
        );
    }
}

/// An operator `sampling` object of the fixture as [`Operator`].
fn fixture_operator(o: &Value) -> Operator {
    Operator {
        temperature: o["temperature"].as_f64(),
        top_p: o["top_p"].as_f64(),
        top_k: o["top_k"].as_u64(),
        repetition_penalty: o["repetition_penalty"].as_f64(),
        presence_penalty: o["presence_penalty"].as_f64(),
        frequency_penalty: o["frequency_penalty"].as_f64(),
    }
}
