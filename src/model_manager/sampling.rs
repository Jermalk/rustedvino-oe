// ============================================================
// src/model_manager/sampling.rs — the per-model sampling profile (0.9.1 V5)
// ============================================================
// One profile per model: the operator's `sampling` object (config) over the
// model's own `generation_config.json` (file layer) over the server default
// (greedy + repetition penalty 1.1). Every generation path (CB chat, VLM, NPU,
// legacy completions, realtime) reads this one profile; the per-request client
// rules (`handlers::chat::resolve_sampling_defaults`) then run against it
// unchanged.
// ============================================================

use std::path::Path;

use serde::Serialize;

use super::config::SamplingConfig;
use super::template::{self, GenerationDefaults};

/// Server-side repetition-penalty safety net when neither the client, the
/// operator nor the model file sets one — see
/// `handlers::chat::resolve_repetition_penalty` for why it exists (the
/// 2026-07-14 VLM loop; `OpenVINO`'s unset default is `1.0`, i.e. off).
///
/// `1.1` matches the long-standing `llama.cpp`/Ollama community default: mild
/// enough not to visibly change well-behaved output, but enough to break the
/// degenerate "same token/word forever" loop.
pub const DEFAULT_REPETITION_PENALTY: f32 = 1.1;

/// `GenerationConfig`'s own temperature when sampling is on and none is set.
const ENGINE_DEFAULT_TEMPERATURE: f32 = 1.0;
/// `GenerationConfig`'s own `top_p` when none is set (no nucleus cut).
const ENGINE_DEFAULT_TOP_P: f32 = 1.0;

/// An `f32` as the `f64` with the same shortest decimal form (0.15, not the
/// widened 0.15000000596046448), for logs and JSON.
#[must_use]
pub fn dec(v: f32) -> f64 {
    v.to_string().parse().unwrap_or_else(|_| f64::from(v))
}

/// Where one profile value comes from. Serialised as the label the admin API
/// and the load-log line use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SamplingSource {
    /// The operator's per-model `sampling` object.
    #[serde(rename = "config")]
    Config,
    /// The model's own `generation_config.json`.
    #[serde(rename = "generation_config.json")]
    GenerationConfig,
    /// Not set anywhere; the engine's own default applies (sampling is on).
    #[serde(rename = "engine default")]
    EngineDefault,
    /// The server's own fallback (the repetition-penalty net).
    #[serde(rename = "server default")]
    ServerDefault,
    /// Not set and not applicable (e.g. `top_p` on a greedy model).
    #[serde(rename = "none")]
    None,
}

impl SamplingSource {
    /// The label used in logs and the admin API.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Config => "config",
            Self::GenerationConfig => "generation_config.json",
            Self::EngineDefault => "engine default",
            Self::ServerDefault => "server default",
            Self::None => "none",
        }
    }
}

/// The file layer: what a model's `generation_config.json` says about
/// sampling, read independently of the operator layer.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FileSampling {
    /// The `do_sample` package (`temperature`/`top_p`/`top_k`), exactly as
    /// [`template::load_generation_defaults`] returns it (all `None` unless
    /// `do_sample: true` and every field passes its sanity gate).
    pub package: GenerationDefaults,
    /// The file says `do_sample: true` (even if the package was then dropped
    /// for a bad value). Gates the operator's `top_p`/`top_k` without a
    /// temperature (V5a).
    pub do_sample: bool,
    /// The file's `repetition_penalty`, kept only when in `(1.0, 2.0]`. Read
    /// outside the `do_sample` gate: a penalty applies to greedy decoding too.
    pub repetition_penalty: Option<f32>,
    /// The file carries `_from_model_config: true`: a transformers exporter
    /// stub, which holds no sampling recommendation of the model's authors.
    pub exporter_stub: bool,
}

impl FileSampling {
    /// The file really turns sampling on for the V5a rule (operator `top_p`/`top_k`
    /// without a temperature): `do_sample: true` **and** a temperature that passed the
    /// sanity gate. A dropped package or a `do_sample` file without a temperature would
    /// leave the engine sampling at its own T 1.0, so it counts as no `do_sample`.
    #[must_use]
    pub fn samples_with_temperature(&self) -> bool {
        self.do_sample && self.package.temperature.is_some()
    }
}

/// Read a model's file layer (blocking file I/O). A missing or unparseable
/// file is an empty layer. The penalty and the `do_sample` package are judged
/// independently: a bad penalty does not drop the package and vice versa.
#[must_use]
pub fn load_file_sampling(model_dir: &Path) -> FileSampling {
    let Some(cfg) = template::read_generation_config(model_dir) else {
        return FileSampling::default();
    };
    let package = template::generation_defaults_from(&cfg, model_dir);
    let do_sample = cfg.get("do_sample").and_then(serde_json::Value::as_bool) == Some(true);
    let exporter_stub = cfg
        .get("_from_model_config")
        .and_then(serde_json::Value::as_bool)
        == Some(true);
    let repetition_penalty = match cfg.get("repetition_penalty") {
        None => None,
        Some(v) => match v.as_f64() {
            // Exactly 1.0 is the transformers default, often a leftover: unset.
            Some(p) if (p - 1.0).abs() < f64::EPSILON => None,
            #[allow(clippy::cast_possible_truncation)]
            Some(p) if p > 1.0 && p <= 2.0 => Some(p as f32),
            other => {
                tracing::warn!(
                    model_dir = %model_dir.display(),
                    repetition_penalty = ?other,
                    "generation_config.json: repetition_penalty is not in (1.0, 2.0] — ignored \
                     (below 1.0 rewards repetition); the server default applies"
                );
                None
            }
        },
    };
    FileSampling {
        package,
        do_sample,
        repetition_penalty,
        exporter_stub,
    }
}

/// One model's effective sampling profile with a source per value. Resolved
/// from the operator layer and the file layer by [`SamplingProfile::resolve`];
/// client values are applied on top per request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingProfile {
    /// Profile temperature (`None` = not set; greedy unless `top_p`/`top_k`
    /// turn sampling on at the engine's T 1.0).
    pub temperature: Option<f32>,
    /// Source of [`temperature`](Self::temperature).
    pub temperature_src: SamplingSource,
    /// Profile `top_p` (blanked when the profile temperature is 0).
    pub top_p: Option<f32>,
    /// Source of [`top_p`](Self::top_p).
    pub top_p_src: SamplingSource,
    /// Profile `top_k` (blanked when the profile temperature is 0).
    pub top_k: Option<usize>,
    /// Source of [`top_k`](Self::top_k).
    pub top_k_src: SamplingSource,
    /// The repetition penalty a request without its own gets.
    pub repetition_penalty: f32,
    /// Source of [`repetition_penalty`](Self::repetition_penalty).
    pub repetition_penalty_src: SamplingSource,
    /// The operator's `presence_penalty` (no file layer).
    pub presence_penalty: Option<f32>,
    /// The operator's `frequency_penalty` (no file layer).
    pub frequency_penalty: Option<f32>,
    /// The model's `generation_config.json` is a transformers exporter stub.
    pub exporter_stub: bool,
}

impl Default for SamplingProfile {
    /// The server default: greedy, repetition penalty 1.1.
    fn default() -> Self {
        Self::resolve(None, &FileSampling::default())
    }
}

impl SamplingProfile {
    /// Merge the operator layer over the file layer (V5b):
    /// - T: operator → file (only with `do_sample`) → none;
    /// - `top_p`/`top_k`: operator → file (only with `do_sample`); blanked at T 0;
    /// - `repetition_penalty`: operator `[1.0, 2.0]` → file `(1.0, 2.0]` → 1.1;
    /// - presence/frequency: operator → none.
    #[must_use]
    #[allow(clippy::similar_names)] // top_p / top_k are the API's own names
    pub fn resolve(operator: Option<&SamplingConfig>, file: &FileSampling) -> Self {
        let op = operator.copied().unwrap_or_default();
        // The config layer is f64 (written back as typed); the engine takes f32.
        #[allow(clippy::cast_possible_truncation)]
        let narrow = |v: Option<f64>| v.map(|x| x as f32);
        let pick = |o: Option<f32>, f: Option<f32>| match (o, f) {
            (Some(v), _) => (Some(v), SamplingSource::Config),
            (None, Some(v)) => (Some(v), SamplingSource::GenerationConfig),
            (None, None) => (None, SamplingSource::None),
        };
        let (temperature, temperature_src) = pick(narrow(op.temperature), file.package.temperature);
        let (mut top_p, mut top_p_src) = pick(narrow(op.top_p), file.package.top_p);
        let (mut top_k, mut top_k_src) = match (op.top_k, file.package.top_k) {
            (Some(v), _) => (Some(v), SamplingSource::Config),
            (None, Some(v)) => (Some(v), SamplingSource::GenerationConfig),
            (None, None) => (None, SamplingSource::None),
        };
        if temperature.is_some_and(|t| t <= 0.0) {
            // Greedy: GenAI rejects top_p/top_k with T 0 (D-11), and they would
            // never apply, so the profile must not show them either.
            (top_p, top_p_src, top_k, top_k_src) =
                (None, SamplingSource::None, None, SamplingSource::None);
        }
        let (repetition_penalty, repetition_penalty_src) =
            match (narrow(op.repetition_penalty), file.repetition_penalty) {
                (Some(v), _) => (v, SamplingSource::Config),
                (None, Some(v)) => (v, SamplingSource::GenerationConfig),
                (None, None) => (DEFAULT_REPETITION_PENALTY, SamplingSource::ServerDefault),
            };
        Self {
            temperature,
            temperature_src,
            top_p,
            top_p_src,
            top_k,
            top_k_src,
            repetition_penalty,
            repetition_penalty_src,
            presence_penalty: narrow(op.presence_penalty),
            frequency_penalty: narrow(op.frequency_penalty),
            exporter_stub: file.exporter_stub,
        }
    }

    /// The `(temperature, top_p, top_k)` package the client rules compose
    /// against (`handlers::chat::resolve_sampling_defaults`).
    #[must_use]
    pub fn package(&self) -> GenerationDefaults {
        GenerationDefaults {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
        }
    }

    /// Whether a request that sets nothing samples: the bridge's own rule
    /// (`build_gen_config`) — a positive temperature, or no temperature and a
    /// `top_p < 1` / `top_k > 0`.
    #[must_use]
    pub fn samples(&self) -> bool {
        match self.temperature {
            Some(t) => t > 0.0,
            None => self.top_p.is_some_and(|p| p < 1.0) || self.top_k.is_some_and(|k| k > 0),
        }
    }

    /// `"sample"` or `"greedy"`.
    #[must_use]
    pub fn mode(&self) -> &'static str {
        if self.samples() { "sample" } else { "greedy" }
    }

    /// Any of T/`top_p`/`top_k` comes from the operator.
    #[must_use]
    pub fn package_from_config(&self) -> bool {
        [self.temperature_src, self.top_p_src, self.top_k_src].contains(&SamplingSource::Config)
    }

    /// The displayed temperature: the engine's 1.0 when sampling with none set.
    #[must_use]
    pub fn shown_temperature(&self) -> (Option<f32>, SamplingSource) {
        match self.temperature {
            Some(t) => (Some(t), self.temperature_src),
            None if self.samples() => (
                Some(ENGINE_DEFAULT_TEMPERATURE),
                SamplingSource::EngineDefault,
            ),
            None => (None, SamplingSource::None),
        }
    }

    /// The displayed `top_p`: the engine's 1.0 when sampling with none set.
    #[must_use]
    pub fn shown_top_p(&self) -> (Option<f32>, SamplingSource) {
        match self.top_p {
            Some(p) => (Some(p), self.top_p_src),
            None if self.samples() => (Some(ENGINE_DEFAULT_TOP_P), SamplingSource::EngineDefault),
            None => (None, SamplingSource::None),
        }
    }

    /// The operator's `presence_penalty` for a request that sends none, or `None`
    /// when it is withheld: on NPU (no presence/frequency penalty there) and on a
    /// request made greedy by the tools pin or by its own `temperature: 0` (no
    /// model-card presence penalty there) — the caller passes `withhold` for both.
    #[must_use]
    pub fn presence_for(&self, withhold: bool) -> Option<f32> {
        if withhold {
            None
        } else {
            self.presence_penalty
        }
    }

    /// See [`presence_for`](Self::presence_for).
    #[must_use]
    pub fn frequency_for(&self, withhold: bool) -> Option<f32> {
        if withhold {
            None
        } else {
            self.frequency_penalty
        }
    }

    /// The `"sampling profile"` load-log line (V5c): at load, and when a
    /// PATCH or a config reload changes the operator layer of a loaded model.
    /// Values are plain numbers; an unset value and an empty note are omitted.
    pub fn log(&self, model_id: &str, npu: bool, reason: &'static str) {
        let (temperature, temperature_src) = self.shown_temperature();
        let (top_p, top_p_src) = self.shown_top_p();
        let note = self
            .exporter_stub
            .then_some("generation_config.json is an exporter stub (no sampling recommendation)");
        let npu_note = (npu
            && (self.presence_penalty.is_some() || self.frequency_penalty.is_some()))
        .then_some("presence/frequency penalties are ignored on NPU");
        tracing::info!(
            model = model_id,
            reason,
            mode = self.mode(),
            temperature = temperature.map(dec),
            temperature_src = temperature_src.label(),
            top_p = top_p.map(dec),
            top_p_src = top_p_src.label(),
            top_k = self.top_k.map(|k| k as u64),
            top_k_src = self.top_k_src.label(),
            rep_pen = dec(self.repetition_penalty),
            rep_pen_src = self.repetition_penalty_src.label(),
            presence = self.presence_penalty.map(dec),
            frequency = self.frequency_penalty.map(dec),
            note,
            npu_note,
            "sampling profile"
        );
    }

    /// The admin API's `sampling` object (V5c; on every `text_gen`/`vision` entry, in every state).
    #[must_use]
    pub fn admin_json(&self, npu: bool) -> serde_json::Value {
        // `json!` widens an f32 to f64 (0.15 → 0.15000000596046448); go through the f32's
        // shortest decimal form so the API shows the value as configured.
        let v = |value: Option<f32>, source: SamplingSource| {
            let value = value.map(dec);
            serde_json::json!({ "value": value, "source": source })
        };
        let (temperature, temperature_src) = self.shown_temperature();
        let (top_p, top_p_src) = self.shown_top_p();
        let pen_src = |p: Option<f32>| {
            if p.is_some() {
                SamplingSource::Config
            } else {
                SamplingSource::None
            }
        };
        let mut out = serde_json::json!({
            "mode": self.mode(),
            "temperature": v(temperature, temperature_src),
            "top_p": v(top_p, top_p_src),
            "top_k": { "value": self.top_k, "source": self.top_k_src },
            "repetition_penalty": v(Some(self.repetition_penalty), self.repetition_penalty_src),
            "presence_penalty": v(self.presence_penalty, pen_src(self.presence_penalty)),
            "frequency_penalty": v(self.frequency_penalty, pen_src(self.frequency_penalty)),
            "tools": "greedy",
        });
        if let Some(map) = out.as_object_mut() {
            if npu {
                map.insert(
                    "ignored_on_device".to_owned(),
                    serde_json::json!(["presence_penalty", "frequency_penalty"]),
                );
            }
            if self.exporter_stub {
                map.insert(
                    "note".to_owned(),
                    serde_json::json!(
                        "generation_config.json is an exporter stub (no sampling recommendation)"
                    ),
                );
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]

    use super::*;

    fn file(t: Option<f32>, p: Option<f32>, k: Option<usize>) -> FileSampling {
        FileSampling {
            package: GenerationDefaults {
                temperature: t,
                top_p: p,
                top_k: k,
            },
            do_sample: t.is_some() || p.is_some() || k.is_some(),
            ..FileSampling::default()
        }
    }

    fn op() -> SamplingConfig {
        SamplingConfig::default()
    }

    // ---- profile T row ----

    #[test]
    fn profile_t_operator_wins_over_file() {
        let s = SamplingConfig {
            temperature: Some(0.15),
            ..op()
        };
        let p = SamplingProfile::resolve(Some(&s), &file(Some(0.6), None, None));
        assert_eq!(p.temperature, Some(0.15));
        assert_eq!(p.temperature_src, SamplingSource::Config);
    }

    #[test]
    fn profile_t_file_when_no_operator() {
        let p = SamplingProfile::resolve(None, &file(Some(0.6), Some(0.95), Some(20)));
        assert_eq!(p.temperature, Some(0.6));
        assert_eq!(p.temperature_src, SamplingSource::GenerationConfig);
        assert_eq!(p.mode(), "sample");
    }

    #[test]
    fn profile_t_none_is_greedy_server_default() {
        let p = SamplingProfile::resolve(None, &FileSampling::default());
        assert_eq!(p.temperature, None);
        assert_eq!(p.mode(), "greedy");
        assert_eq!(p.shown_temperature(), (None, SamplingSource::None));
        assert_eq!(p.repetition_penalty, DEFAULT_REPETITION_PENALTY);
        assert_eq!(p.repetition_penalty_src, SamplingSource::ServerDefault);
        assert_eq!(p, SamplingProfile::default());
    }

    // ---- profile top_p / top_k row ----

    #[test]
    fn profile_top_p_top_k_operator_then_file() {
        let s = SamplingConfig {
            temperature: Some(0.7),
            top_p: Some(0.8),
            ..op()
        };
        let p = SamplingProfile::resolve(Some(&s), &file(Some(1.0), Some(0.95), Some(64)));
        assert_eq!((p.top_p, p.top_p_src), (Some(0.8), SamplingSource::Config));
        assert_eq!(
            (p.top_k, p.top_k_src),
            (Some(64), SamplingSource::GenerationConfig)
        );
    }

    #[test]
    fn profile_t0_blanks_the_file_top_k() {
        let s = SamplingConfig {
            temperature: Some(0.0),
            ..op()
        };
        let p = SamplingProfile::resolve(Some(&s), &file(Some(1.0), Some(0.95), Some(64)));
        assert_eq!(p.temperature, Some(0.0));
        assert_eq!((p.top_p, p.top_k), (None, None));
        assert_eq!(p.top_k_src, SamplingSource::None);
        assert_eq!(p.mode(), "greedy");
    }

    #[test]
    fn profile_file_top_k_only_samples_at_engine_t() {
        // do_sample with only a top_k: the bridge samples at GenAI's T 1.0.
        let p = SamplingProfile::resolve(None, &file(None, None, Some(40)));
        assert_eq!(p.mode(), "sample");
        assert_eq!(
            p.shown_temperature(),
            (Some(1.0), SamplingSource::EngineDefault)
        );
        assert_eq!(p.shown_top_p(), (Some(1.0), SamplingSource::EngineDefault));
    }

    #[test]
    fn profile_operator_t_only_on_greedy_file() {
        let s = SamplingConfig {
            temperature: Some(0.15),
            ..op()
        };
        let p = SamplingProfile::resolve(Some(&s), &FileSampling::default());
        assert_eq!(p.mode(), "sample");
        assert_eq!((p.top_p, p.top_k), (None, None));
        assert!(p.package_from_config());
    }

    // ---- repetition_penalty row ----

    #[test]
    fn penalty_operator_then_file_then_server_default() {
        let f = FileSampling {
            repetition_penalty: Some(1.05),
            ..FileSampling::default()
        };
        let p = SamplingProfile::resolve(None, &f);
        assert_eq!(
            (p.repetition_penalty, p.repetition_penalty_src),
            (1.05, SamplingSource::GenerationConfig)
        );
        let s = SamplingConfig {
            repetition_penalty: Some(1.0),
            ..op()
        };
        let p = SamplingProfile::resolve(Some(&s), &f);
        assert_eq!(
            (p.repetition_penalty, p.repetition_penalty_src),
            (1.0, SamplingSource::Config),
            "an operator 1.0 is an explicit opt-out"
        );
    }

    #[test]
    fn presence_frequency_operator_only_and_not_on_npu() {
        let s = SamplingConfig {
            presence_penalty: Some(1.5),
            frequency_penalty: Some(-0.5),
            ..op()
        };
        let p = SamplingProfile::resolve(Some(&s), &FileSampling::default());
        assert_eq!(p.presence_for(false), Some(1.5));
        assert_eq!(p.frequency_for(false), Some(-0.5));
        assert_eq!(p.presence_for(true), None);
        assert_eq!(p.frequency_for(true), None);
        let none = SamplingProfile::default();
        assert_eq!(none.presence_for(false), None);
    }

    // ---- file layer ----

    fn dir(suffix: &str, body: &str) -> std::path::PathBuf {
        let d =
            std::env::temp_dir().join(format!("rv-filesampling-{}-{suffix}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("generation_config.json"), body).unwrap();
        d
    }

    #[test]
    fn file_penalty_gate() {
        let cases = [
            ("one", r#"{"repetition_penalty": 1.0}"#, None),
            ("below", r#"{"repetition_penalty": 0.9}"#, None),
            ("ok", r#"{"repetition_penalty": 1.05}"#, Some(1.05_f32)),
            ("above", r#"{"repetition_penalty": 2.5}"#, None),
            ("string", r#"{"repetition_penalty": "1.2"}"#, None),
        ];
        for (suffix, body, want) in cases {
            let d = dir(suffix, body);
            assert_eq!(load_file_sampling(&d).repetition_penalty, want, "{body}");
            let _ = std::fs::remove_dir_all(&d);
        }
    }

    #[test]
    fn file_penalty_is_read_without_do_sample() {
        let d = dir(
            "nodosample",
            r#"{"temperature": 0.7, "repetition_penalty": 1.05}"#,
        );
        let f = load_file_sampling(&d);
        assert_eq!(f.repetition_penalty, Some(1.05));
        assert!(!f.do_sample);
        assert_eq!(f.package, GenerationDefaults::default());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn bad_penalty_keeps_the_package_and_bad_package_keeps_the_penalty() {
        let d = dir(
            "badpen",
            r#"{"do_sample": true, "temperature": 0.6, "repetition_penalty": 0.8}"#,
        );
        let f = load_file_sampling(&d);
        assert_eq!(f.package.temperature, Some(0.6));
        assert_eq!(f.repetition_penalty, None);
        let _ = std::fs::remove_dir_all(&d);

        let d = dir(
            "badpkg",
            r#"{"do_sample": true, "top_p": 7.0, "repetition_penalty": 1.05}"#,
        );
        let f = load_file_sampling(&d);
        assert_eq!(f.package, GenerationDefaults::default());
        assert!(f.do_sample);
        assert_eq!(f.repetition_penalty, Some(1.05));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn exporter_stub_is_named() {
        let d = dir("stub", r#"{"_from_model_config": true, "bos_token_id": 1}"#);
        let f = load_file_sampling(&d);
        assert!(f.exporter_stub);
        let p = SamplingProfile::resolve(None, &f);
        assert!(p.admin_json(false)["note"].is_string());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn missing_file_is_an_empty_layer() {
        let d =
            std::env::temp_dir().join(format!("rv-filesampling-{}-missing", std::process::id()));
        assert_eq!(load_file_sampling(&d), FileSampling::default());
    }

    /// f32 values print as written (0.15, not 0.15000000596046448) in the admin JSON.
    #[test]
    fn admin_json_prints_f32_values_as_written() {
        let s = SamplingConfig {
            temperature: Some(0.15),
            ..op()
        };
        let text = serde_json::to_string(
            &SamplingProfile::resolve(Some(&s), &FileSampling::default()).admin_json(false),
        )
        .unwrap();
        assert!(text.contains(r#""value":0.15,"#), "{text}");
        assert!(text.contains(r#""value":1.1,"#), "{text}");
    }

    /// Review F11 (round V5-rv): the fixture `rv`'s mirror is tested against
    /// (`tools/rv/tests/fixtures/sampling_profiles.json`, {file, operator, npu} → admin
    /// `sampling` JSON) also holds for the real file layer + resolver + `admin_json`.
    #[test]
    fn shared_fixture_matches_the_real_resolver() {
        let cases: Vec<serde_json::Value> = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tools/rv/tests/fixtures/sampling_profiles.json"
        )))
        .unwrap();
        assert!(cases.len() >= 10);
        for c in &cases {
            let d = tempfile::tempdir().unwrap();
            if c["generation_config"].is_object() {
                std::fs::write(
                    d.path().join("generation_config.json"),
                    c["generation_config"].to_string(),
                )
                .unwrap();
            }
            let op: Option<SamplingConfig> = serde_json::from_value(c["operator"].clone()).unwrap();
            let profile = SamplingProfile::resolve(op.as_ref(), &load_file_sampling(d.path()));
            assert_eq!(
                profile.admin_json(c["npu"].as_bool().unwrap()),
                c["expected"],
                "{}",
                c["name"]
            );
        }
    }

    #[test]
    fn admin_json_shape() {
        let s = SamplingConfig {
            temperature: Some(0.15),
            top_p: Some(0.95),
            presence_penalty: Some(1.5),
            ..op()
        };
        let j = SamplingProfile::resolve(Some(&s), &FileSampling::default()).admin_json(true);
        assert_eq!(j["mode"], "sample");
        assert_eq!(j["temperature"]["source"], "config");
        assert!((j["temperature"]["value"].as_f64().unwrap() - 0.15).abs() < 1e-6);
        assert_eq!(
            j["top_k"],
            serde_json::json!({"value": null, "source": "none"})
        );
        assert_eq!(j["repetition_penalty"]["source"], "server default");
        assert_eq!(j["presence_penalty"]["source"], "config");
        assert_eq!(j["frequency_penalty"]["source"], "none");
        assert_eq!(j["tools"], "greedy");
        assert_eq!(
            j["ignored_on_device"],
            serde_json::json!(["presence_penalty", "frequency_penalty"])
        );
        assert!(j.get("note").is_none());
    }
}
