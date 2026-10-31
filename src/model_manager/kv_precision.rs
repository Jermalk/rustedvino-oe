// ============================================================
// src/model_manager/kv_precision.rs — which KV-cache precision an engine gets
// ============================================================
// RustedVINO resolves the precision itself and passes it to the engine,
// so the capacity gate prices what the engine really builds. The result is both the
// precision (for accounting) and the exact string handed to the plugin (for the engine and
// for the blob-cache directory tag): the two differ when the plugin is left to decide.
//
// Pure functions of the profile and the configuration; the loader owns the logging.
// ============================================================

use super::kv_profile::{Confidence, KvPrecision, KvProfile};

/// Where the resolved precision came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrecisionSource {
    /// The model's own `kv_cache_precision` override.
    PerModel,
    /// The global `kv_cache_precision` (text models only).
    GlobalExplicit,
    /// The IR's `rt_info/runtime_options/KV_CACHE_PRECISION` hint.
    IrRuntimeOptions,
    /// The profile rule: u4 when the weights are 4-bit, otherwise i8.
    AutoRule,
    /// Nothing is decided here: the configured string goes through unchanged (CPU, NPU,
    /// `kv_precision_pass: plugin`, or no usable profile).
    LegacyPassthrough,
}

impl PrecisionSource {
    /// Stable `snake_case` name for the admin API and logs.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::PerModel => "per_model",
            Self::GlobalExplicit => "global",
            Self::IrRuntimeOptions => "ir_hint",
            Self::AutoRule => "auto_rule",
            Self::LegacyPassthrough => "passthrough",
        }
    }
}

/// Which engine builds the KV cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EnginePath {
    /// Continuous-batching text engine.
    Cb,
    /// `VLMPipeline` (vision models).
    Vlm,
}

/// `kv_precision_pass`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PassMode {
    /// `RustedVINO` resolves and passes the precision.
    Explicit,
    /// In 0.8.x: vision models pass nothing, text models pass the global string.
    Plugin,
}

impl PassMode {
    /// Parse the validated config string; anything but `"plugin"` is `Explicit`.
    pub(crate) fn from_config(s: &str) -> Self {
        if s == "plugin" {
            Self::Plugin
        } else {
            Self::Explicit
        }
    }
}

/// Outcome of [`resolve_kv_precision`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KvResolution {
    /// The precision the engine will build, when `RustedVINO` knows it. `None` = unknown or not
    /// applicable (the plugin decides and the profile cannot say what it will pick; NPU; a CPU
    /// or HETERO spec without an explicit string): the gate must stay legacy. Never a guess.
    pub precision: Option<KvPrecision>,
    /// What the rule would pick even when it is not trusted enough to pass (display only; the
    /// plugin may build something else).
    pub rule_guess: Option<KvPrecision>,
    /// Where the precision came from.
    pub source: PrecisionSource,
    /// The exact `KV_CACHE_PRECISION` string handed to the engine: `""` (plugin default),
    /// `"i8"`, `"u4"` or `"f16"`. Also the tag of the blob-cache directory.
    pub passed: &'static str,
}

/// The string the plugin takes for a precision.
pub(crate) fn precision_str(p: KvPrecision) -> &'static str {
    match p {
        KvPrecision::U4 => "u4",
        KvPrecision::I8 => "i8",
        KvPrecision::F16 => "f16",
    }
}

/// Parse a normalised precision string (`""`, `i8`, `u4`, `f16`; `u8` and `i4` accepted as
/// their aliases). `""`, `auto` and anything else give `None`.
pub(crate) fn parse_precision(s: &str) -> Option<KvPrecision> {
    match s {
        "i8" | "u8" => Some(KvPrecision::I8),
        "u4" | "i4" => Some(KvPrecision::U4),
        "f16" => Some(KvPrecision::F16),
        _ => None,
    }
}

/// Resolve the KV-cache precision of one model.
///
/// - `profile`: `None` when `config.json` could not be read.
/// - `per_model`: the model's own override (`ModelPolicy`), normalised like `global`. `Some("")` is
///   an explicit per-model `"auto"`: the global key is ignored for this model and the IR hint or the
///   rule decides; `None` (no override) inherits the global key.
/// - `global`: the normalised global `kv_cache_precision` (`""`, `i8`, `u4`, `f16`).
///
/// Precedence on a plain GPU device: text models: per-model > global (non-empty) > IR hint >
/// rule; vision models: per-model > IR hint > rule (the global key does not apply to them).
/// The rule is u4 for 4-bit weights, i8 otherwise, and is only trusted (passed, and reported
/// as the engine's precision) when the 4-bit scan ran and the profile confidence is at least
/// Medium; otherwise the plugin keeps the decision (`passed == ""`, `precision == None`).
///
/// NPU engines take no KV precision. Every other device (CPU, HETERO/AUTO/MULTI specs) keeps
/// today's behaviour exactly: the global string for text, nothing for vision.
/// `PassMode::Plugin` restores that pass-through on GPU as well, while the resolved
/// precision stays available for the gate.
pub(crate) fn resolve_kv_precision(
    profile: Option<&KvProfile>,
    path: EnginePath,
    device: &str,
    per_model: Option<&str>,
    global: &str,
    mode: PassMode,
) -> KvResolution {
    let dev = device.to_ascii_uppercase();
    let legacy_pass = legacy_pass(path, global);
    if dev.starts_with("NPU") {
        return KvResolution {
            precision: None,
            rule_guess: None,
            source: PrecisionSource::LegacyPassthrough,
            passed: "",
        };
    }
    if !dev.starts_with("GPU") {
        return passthrough(legacy_pass);
    }
    let resolved = resolve_gpu(profile, path, per_model, global);
    if mode == PassMode::Plugin {
        return KvResolution {
            passed: legacy_pass,
            ..resolved
        };
    }
    resolved
}

/// The resolution of a model whose precision `RustedVINO` deliberately does not decide (a text
/// model with a speculative-decoding draft: the draft inherits the main engine's properties and may
/// run on another device): exactly what 0.8.x handed the engine.
pub(crate) fn legacy_resolution(path: EnginePath, global: &str) -> KvResolution {
    passthrough(legacy_pass(path, global))
}

/// What the 0.8.x code handed the engine: the global string for text, nothing for vision.
fn legacy_pass(path: EnginePath, global: &str) -> &'static str {
    match path {
        EnginePath::Cb => parse_precision(global).map_or("", precision_str),
        EnginePath::Vlm => "",
    }
}

fn passthrough(passed: &'static str) -> KvResolution {
    KvResolution {
        precision: parse_precision(passed),
        rule_guess: None,
        source: PrecisionSource::LegacyPassthrough,
        passed,
    }
}

/// Explicit-mode resolution on a plain GPU device.
fn resolve_gpu(
    profile: Option<&KvProfile>,
    path: EnginePath,
    per_model: Option<&str>,
    global: &str,
) -> KvResolution {
    if let Some(p) = per_model.and_then(parse_precision) {
        return KvResolution {
            precision: Some(p),
            rule_guess: None,
            source: PrecisionSource::PerModel,
            passed: precision_str(p),
        };
    }
    // A per-model `"auto"` (`Some("")`) opts this model out of the global setting: the IR hint or
    // the rule decides. No per-model value (`None`) inherits the global key.
    let opted_out_of_global = per_model == Some("");
    if path == EnginePath::Cb
        && !opted_out_of_global
        && let Some(p) = parse_precision(global)
    {
        return KvResolution {
            precision: Some(p),
            rule_guess: None,
            source: PrecisionSource::GlobalExplicit,
            passed: precision_str(p),
        };
    }
    let Some(profile) = profile else {
        // No profile to apply the rule to: the plugin decides. A per-model `"auto"` still keeps
        // the global key out (0.9.0 review, docs-upgrade-9).
        return passthrough(if opted_out_of_global {
            ""
        } else {
            legacy_pass(path, global)
        });
    };
    if let Some(p) = profile.ir_kv_hint {
        return KvResolution {
            precision: Some(p),
            rule_guess: None,
            source: PrecisionSource::IrRuntimeOptions,
            passed: precision_str(p),
        };
    }
    let p = if profile.weights_4bit == Some(true) {
        KvPrecision::U4
    } else {
        KvPrecision::I8
    };
    let trusted = profile.weights_4bit.is_some() && profile.confidence >= Confidence::Medium;
    // An untrusted rule passes nothing and claims nothing: the plugin may pick something else
    // (an IR hint the scan missed, a GPU without XMX), and a guess would over-promise.
    KvResolution {
        precision: trusted.then_some(p),
        rule_guess: Some(p),
        source: PrecisionSource::AutoRule,
        passed: if trusted { precision_str(p) } else { "" },
    }
}

/// Whether the one-per-process "the global key does not apply to vision models" warning is due:
/// a vision model resolved on its own to something other than what the global key names.
pub(crate) fn vlm_ignores_global(path: EnginePath, res: KvResolution, global: &str) -> bool {
    if path != EnginePath::Vlm || res.source == PrecisionSource::LegacyPassthrough {
        return false;
    }
    parse_precision(global).is_some_and(|g| Some(g) != res.precision)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::super::kv_profile::PaLayer;
    use super::*;

    fn profile(w4: Option<bool>, hint: Option<KvPrecision>, conf: Confidence) -> KvProfile {
        KvProfile {
            pa_layers: vec![
                PaLayer {
                    kv_heads: 8,
                    k_head_dim: 128,
                    v_head_dim: 128,
                };
                4
            ],
            block_tokens: 16,
            weights_4bit: w4,
            ir_kv_hint: hint,
            linear_state_bytes_per_seq: 0,
            hidden_size: 4096,
            confidence: conf,
        }
    }

    fn r(
        p: Option<&KvProfile>,
        path: EnginePath,
        dev: &str,
        per_model: Option<&str>,
        global: &str,
    ) -> KvResolution {
        resolve_kv_precision(p, path, dev, per_model, global, PassMode::Explicit)
    }

    #[test]
    fn u03_cb_gpu_precedence() {
        let hint = profile(Some(false), Some(KvPrecision::F16), Confidence::High);
        // global explicit beats the IR hint
        let a = r(Some(&hint), EnginePath::Cb, "GPU.1", None, "i8");
        assert_eq!(
            (a.precision, a.source, a.passed),
            (Some(KvPrecision::I8), PrecisionSource::GlobalExplicit, "i8")
        );
        // global auto -> IR hint
        let a = r(Some(&hint), EnginePath::Cb, "GPU.1", None, "");
        assert_eq!(
            (a.precision, a.source, a.passed),
            (
                Some(KvPrecision::F16),
                PrecisionSource::IrRuntimeOptions,
                "f16"
            )
        );
        // per-model beats everything
        let a = r(Some(&hint), EnginePath::Cb, "GPU.1", Some("u4"), "i8");
        assert_eq!(
            (a.precision, a.source, a.passed),
            (Some(KvPrecision::U4), PrecisionSource::PerModel, "u4")
        );
        // no hint: the rule
        let w4 = profile(Some(true), None, Confidence::High);
        let a = r(Some(&w4), EnginePath::Cb, "GPU.1", None, "");
        assert_eq!(
            (a.precision, a.source, a.passed),
            (Some(KvPrecision::U4), PrecisionSource::AutoRule, "u4")
        );
        let w8 = profile(Some(false), None, Confidence::High);
        let a = r(Some(&w8), EnginePath::Cb, "GPU.1", None, "");
        assert_eq!(
            (a.precision, a.passed),
            (Some(KvPrecision::I8), "i8"),
            "no 4-bit weights: i8"
        );
    }

    /// Block B2b: a per-model `"auto"` (`Some("")`) opts a text model out of the global key; no
    /// per-model value (`None`) inherits it.
    #[test]
    fn a_per_model_auto_opts_out_of_the_global_key() {
        let w4 = profile(Some(true), None, Confidence::High);
        let inherit = r(Some(&w4), EnginePath::Cb, "GPU.1", None, "i8");
        assert_eq!(
            (inherit.source, inherit.passed),
            (PrecisionSource::GlobalExplicit, "i8")
        );
        let auto = r(Some(&w4), EnginePath::Cb, "GPU.1", Some(""), "i8");
        assert_eq!(
            (auto.precision, auto.source, auto.passed),
            (Some(KvPrecision::U4), PrecisionSource::AutoRule, "u4"),
            "the rule decides, not the global i8"
        );
        let hint = profile(Some(false), Some(KvPrecision::F16), Confidence::High);
        let a = r(Some(&hint), EnginePath::Cb, "GPU.1", Some(""), "i8");
        assert_eq!(
            (a.source, a.passed),
            (PrecisionSource::IrRuntimeOptions, "f16")
        );
        // An explicit per-model value still wins over everything.
        let a = r(Some(&w4), EnginePath::Cb, "GPU.1", Some("f16"), "i8");
        assert_eq!((a.source, a.passed), (PrecisionSource::PerModel, "f16"));
        // No readable profile: the plugin decides, and the global key stays out.
        let a = r(None, EnginePath::Cb, "GPU.1", Some(""), "i8");
        assert_eq!(
            (a.source, a.passed),
            (PrecisionSource::LegacyPassthrough, "")
        );
        let a = r(None, EnginePath::Cb, "GPU.1", None, "i8");
        assert_eq!(
            (a.source, a.passed),
            (PrecisionSource::GlobalExplicit, "i8")
        );
    }

    #[test]
    fn u03_vlm_ignores_the_global_key() {
        let hint = profile(Some(true), Some(KvPrecision::F16), Confidence::High);
        let a = r(Some(&hint), EnginePath::Vlm, "GPU.1", None, "i8");
        assert_eq!(
            (a.precision, a.source, a.passed),
            (
                Some(KvPrecision::F16),
                PrecisionSource::IrRuntimeOptions,
                "f16"
            )
        );
        let w4 = profile(Some(true), None, Confidence::High);
        let a = r(Some(&w4), EnginePath::Vlm, "GPU.1", None, "i8");
        assert_eq!(
            (a.precision, a.source, a.passed),
            (Some(KvPrecision::U4), PrecisionSource::AutoRule, "u4")
        );
        let a = r(Some(&w4), EnginePath::Vlm, "GPU.1", Some("i8"), "");
        assert_eq!(
            (a.precision, a.source, a.passed),
            (Some(KvPrecision::I8), PrecisionSource::PerModel, "i8")
        );
    }

    #[test]
    fn u03_cpu_and_npu_keep_today_s_behaviour() {
        let w4 = profile(Some(true), Some(KvPrecision::F16), Confidence::High);
        // CPU text: the normalised global string, whatever the profile says
        let a = r(Some(&w4), EnginePath::Cb, "CPU", Some("u4"), "i8");
        assert_eq!(
            (a.source, a.passed),
            (PrecisionSource::LegacyPassthrough, "i8")
        );
        let a = r(Some(&w4), EnginePath::Cb, "CPU", None, "");
        assert_eq!(a.passed, "");
        // CPU vision: nothing
        let a = r(Some(&w4), EnginePath::Vlm, "CPU", None, "i8");
        assert_eq!(
            (a.source, a.passed),
            (PrecisionSource::LegacyPassthrough, "")
        );
        // NPU: no KV precision at all
        let a = r(Some(&w4), EnginePath::Cb, "NPU", None, "i8");
        assert_eq!((a.precision, a.passed), (None, ""));
    }

    #[test]
    fn u03_plugin_mode_restores_the_old_pass_through() {
        let w4 = profile(Some(true), None, Confidence::High);
        let cb = resolve_kv_precision(
            Some(&w4),
            EnginePath::Cb,
            "GPU.1",
            Some("u4"),
            "i8",
            PassMode::Plugin,
        );
        // The engine gets today's string; the gate keeps the resolved value.
        assert_eq!(cb.passed, "i8");
        assert_eq!(
            (cb.precision, cb.source),
            (Some(KvPrecision::U4), PrecisionSource::PerModel)
        );
        let vlm = resolve_kv_precision(
            Some(&w4),
            EnginePath::Vlm,
            "GPU.1",
            None,
            "i8",
            PassMode::Plugin,
        );
        assert_eq!(vlm.passed, "");
        assert_eq!(
            (vlm.precision, vlm.source),
            (Some(KvPrecision::U4), PrecisionSource::AutoRule)
        );
        // Text models under global auto: nothing passed, the rule's value still priced.
        let auto = resolve_kv_precision(
            Some(&w4),
            EnginePath::Cb,
            "GPU.1",
            None,
            "",
            PassMode::Plugin,
        );
        assert_eq!((auto.passed, auto.precision), ("", Some(KvPrecision::U4)));
    }

    #[test]
    fn untrusted_profile_leaves_the_decision_to_the_plugin() {
        // The rule passes nothing and claims nothing; the guess is for display.
        let low = profile(Some(true), None, Confidence::Low);
        let a = r(Some(&low), EnginePath::Vlm, "GPU.1", None, "");
        assert_eq!(
            (a.precision, a.rule_guess, a.source, a.passed),
            (None, Some(KvPrecision::U4), PrecisionSource::AutoRule, "")
        );
        // IR unreadable: weights unknown
        let unknown = profile(None, None, Confidence::Medium);
        let a = r(Some(&unknown), EnginePath::Cb, "GPU.1", None, "");
        assert_eq!((a.passed, a.precision), ("", None));
        // no profile at all
        let a = r(None, EnginePath::Cb, "GPU.1", None, "u4");
        assert_eq!(a.passed, "u4", "an explicit global still passes");
        let a = r(None, EnginePath::Vlm, "GPU.1", None, "u4");
        assert_eq!((a.precision, a.passed), (None, ""));
        // Medium is trusted when the 4-bit scan ran
        let med = profile(Some(false), None, Confidence::Medium);
        let a = r(Some(&med), EnginePath::Cb, "GPU.1", None, "");
        assert_eq!((a.passed, a.precision), ("i8", Some(KvPrecision::I8)));
    }

    #[test]
    fn hetero_auto_and_multi_specs_keep_the_pass_through() {
        let w4 = profile(Some(true), None, Confidence::High);
        for dev in ["HETERO:GPU.0,GPU.1", "AUTO", "MULTI:GPU.0,CPU"] {
            let a = r(Some(&w4), EnginePath::Cb, dev, Some("u4"), "i8");
            assert_eq!(
                (a.source, a.passed),
                (PrecisionSource::LegacyPassthrough, "i8"),
                "{dev}"
            );
            let v = r(Some(&w4), EnginePath::Vlm, dev, None, "i8");
            assert_eq!(v.passed, "", "{dev}");
        }
    }

    #[test]
    fn vision_warning_is_due_only_when_the_global_key_is_overridden() {
        let w4 = profile(Some(true), None, Confidence::High);
        let vlm = r(Some(&w4), EnginePath::Vlm, "GPU.1", None, "i8");
        assert!(
            vlm_ignores_global(EnginePath::Vlm, vlm, "i8"),
            "u4 vs global i8"
        );
        assert!(
            !vlm_ignores_global(EnginePath::Vlm, vlm, "u4"),
            "same value"
        );
        assert!(!vlm_ignores_global(EnginePath::Vlm, vlm, ""), "global auto");
        let cb = r(Some(&w4), EnginePath::Cb, "GPU.1", None, "i8");
        assert!(!vlm_ignores_global(EnginePath::Cb, cb, "i8"), "text models");
        let cpu = r(Some(&w4), EnginePath::Vlm, "CPU", None, "i8");
        assert!(
            !vlm_ignores_global(EnginePath::Vlm, cpu, "i8"),
            "pass-through"
        );
        let low = profile(Some(true), None, Confidence::Low);
        let untrusted = r(Some(&low), EnginePath::Vlm, "GPU.1", None, "i8");
        assert!(
            vlm_ignores_global(EnginePath::Vlm, untrusted, "i8"),
            "nothing known about what the plugin builds: still not the global key"
        );
    }

    #[test]
    fn precision_strings_round_trip() {
        for p in [KvPrecision::U4, KvPrecision::I8, KvPrecision::F16] {
            assert_eq!(parse_precision(precision_str(p)), Some(p));
        }
        assert_eq!(parse_precision("u8"), Some(KvPrecision::I8));
        assert_eq!(parse_precision(""), None);
        assert_eq!(parse_precision("auto"), None);
        assert_eq!(PassMode::from_config("plugin"), PassMode::Plugin);
        assert_eq!(PassMode::from_config("explicit"), PassMode::Explicit);
    }

    /// Every string `passed` can take has a blob-cache directory tag.
    #[test]
    fn every_passed_string_has_a_cache_tag() {
        for s in ["", "i8", "u4", "f16"] {
            assert!(crate::cache_manifest::kv_cache_subdir(s).is_ok(), "{s}");
        }
    }
}
