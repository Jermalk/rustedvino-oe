//! The per-model sampling profile on the `rv` side (0.9.1 V5c): the operator's
//! sampling flags of `rv add` / `rv patch`, their client-side checks, the
//! client-side merge of a patch over the operator values the server reports,
//! a local preview of the profile from a model's own `generation_config.json`,
//! and the one formatter every verb uses to print a profile.
//!
//! The server is the authority: these checks only give the error before the
//! request. `rv` deliberately does not depend on the server crate (it links
//! `OpenVINO`), so the local preview is a small **mirror** of
//! `src/model_manager/sampling.rs` (`load_file_sampling`, `SamplingProfile::resolve`,
//! `admin_json`) and of `src/model_manager/template.rs` (`generation_defaults_from`);
//! the tests below port the server's resolver tests row by row. Keep them in step.

use std::path::Path;

use serde_json::{Map, Value, json};

/// The server's repetition-penalty net when nobody sets one.
pub const DEFAULT_REPETITION_PENALTY: f64 = 1.1;
/// `GenerationConfig`'s own temperature / `top_p` when sampling with none set.
const ENGINE_DEFAULT: f64 = 1.0;

/// The model kinds that carry a sampling profile (the admin `sampling` field
/// exists for these only, P9/F2).
#[must_use]
pub fn kind_has_sampling(kind: &str) -> bool {
    matches!(kind, "text_gen" | "vision")
}

/// The warning for a text model whose admin entry carries no `sampling`.
pub const OLD_SERVER_WARNING: &str = "server < 0.9.1: sampling settings NOT applied";

// ---------------------------------------------------------------------------
// Operator values and the flags
// ---------------------------------------------------------------------------

/// The operator's per-model `sampling` object (the config keys).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Operator {
    /// `[0, 2]`.
    pub temperature: Option<f64>,
    /// `(0, 1]`.
    pub top_p: Option<f64>,
    /// At least 1.
    pub top_k: Option<u64>,
    /// `[1.0, 2.0]`.
    pub repetition_penalty: Option<f64>,
    /// `[-2, 2]`.
    pub presence_penalty: Option<f64>,
    /// `[-2, 2]`.
    pub frequency_penalty: Option<f64>,
}

impl Operator {
    /// No value set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// `other`'s values over `self`'s (a flag wins over the operator value).
    #[must_use]
    pub fn overlay(self, other: &Self) -> Self {
        Self {
            temperature: other.temperature.or(self.temperature),
            top_p: other.top_p.or(self.top_p),
            top_k: other.top_k.or(self.top_k),
            repetition_penalty: other.repetition_penalty.or(self.repetition_penalty),
            presence_penalty: other.presence_penalty.or(self.presence_penalty),
            frequency_penalty: other.frequency_penalty.or(self.frequency_penalty),
        }
    }

    /// The `sampling` object as the server's config takes it (unset keys left out).
    #[must_use]
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        for (key, v) in self.floats() {
            if let Some(v) = v {
                m.insert(key.to_owned(), json!(v));
            }
        }
        if let Some(k) = self.top_k {
            m.insert("top_k".to_owned(), json!(k));
        }
        // Keep the config key order: temperature, top_p, top_k, the penalties.
        let order = [
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
        ];
        Value::Object(
            order
                .iter()
                .filter_map(|k| m.get(*k).map(|v| ((*k).to_owned(), v.clone())))
                .collect(),
        )
    }

    /// The operator values the server reports in an admin `sampling` field:
    /// every value whose `source` is `config`.
    #[must_use]
    pub fn from_admin(sampling: &Value) -> Self {
        let cfg = |key: &str| {
            let e = sampling.get(key)?;
            (e.get("source").and_then(Value::as_str) == Some("config"))
                .then(|| e.get("value"))
                .flatten()
                .cloned()
        };
        let f = |key: &str| cfg(key).as_ref().and_then(Value::as_f64);
        Self {
            temperature: f("temperature"),
            top_p: f("top_p"),
            top_k: cfg("top_k").as_ref().and_then(Value::as_u64),
            repetition_penalty: f("repetition_penalty"),
            presence_penalty: f("presence_penalty"),
            frequency_penalty: f("frequency_penalty"),
        }
    }

    /// `T 0.15 · top_p 0.95`, or `none` — the `before → after` text of `rv patch`.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if let Some(t) = self.temperature {
            parts.push(format!("T {}", num(t)));
        }
        if let Some(p) = self.top_p {
            parts.push(format!("top_p {}", num(p)));
        }
        if let Some(k) = self.top_k {
            parts.push(format!("top_k {k}"));
        }
        for (key, v) in [
            ("repetition_penalty", self.repetition_penalty),
            ("presence_penalty", self.presence_penalty),
            ("frequency_penalty", self.frequency_penalty),
        ] {
            if let Some(v) = v {
                parts.push(format!("{key} {}", num(v)));
            }
        }
        if parts.is_empty() {
            "none".to_owned()
        } else {
            parts.join(" · ")
        }
    }

    fn floats(&self) -> [(&'static str, Option<f64>); 5] {
        [
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("repetition_penalty", self.repetition_penalty),
            ("presence_penalty", self.presence_penalty),
            ("frequency_penalty", self.frequency_penalty),
        ]
    }

    /// The server's range checks (`config.rs` `validate_sampling`), worded for
    /// the flags. Run on the flags and again on the merged object.
    ///
    /// # Errors
    /// A value is not finite or out of range, or `top_p`/`top_k` meet T 0.
    pub fn validate(&self) -> Result<(), String> {
        for (key, v) in self.floats() {
            if let Some(v) = v
                && !v.is_finite()
            {
                return Err(format!(
                    "error: --{} must be a finite number, got {v}",
                    flag(key)
                ));
            }
        }
        if let Some(t) = self.temperature
            && !(0.0..=2.0).contains(&t)
        {
            return Err(format!(
                "error: --temperature must be between 0 and 2, got {t}"
            ));
        }
        if let Some(p) = self.top_p
            && !(p > 0.0 && p <= 1.0)
        {
            return Err(format!(
                "error: --top-p must be greater than 0 and at most 1, got {p}"
            ));
        }
        if self.top_k == Some(0) {
            return Err(
                "error: --top-k must be at least 1, got 0 (omit it for no top-k cut)".to_owned(),
            );
        }
        if let Some(r) = self.repetition_penalty
            && !(1.0..=2.0).contains(&r)
        {
            return Err(format!(
                "error: --repetition-penalty must be between 1.0 and 2.0 (1.0 turns the server's \
                 1.1 default off), got {r}"
            ));
        }
        for (key, v) in [
            ("presence_penalty", self.presence_penalty),
            ("frequency_penalty", self.frequency_penalty),
        ] {
            if let Some(v) = v
                && !(-2.0..=2.0).contains(&v)
            {
                return Err(format!(
                    "error: --{} must be between -2 and 2, got {v}",
                    flag(key)
                ));
            }
        }
        if self.temperature.is_some_and(|t| t <= 0.0)
            && (self.top_p.is_some() || self.top_k.is_some())
        {
            return Err(
                "error: temperature 0 (greedy) cannot be combined with top_p/top_k — drop them, \
                 or set a temperature above 0"
                    .to_owned(),
            );
        }
        Ok(())
    }

    /// The server's `do_sample` rule (`config.rs` `check_sampling_do_sample`):
    /// `top_p`/`top_k` without a temperature need a file that samples with a
    /// temperature of its own, else the engine samples at its T 1.0. The error
    /// carries no advice; the caller adds what fits its flags.
    ///
    /// # Errors
    /// The rule is violated.
    pub fn check_do_sample(&self, file_samples_with_temperature: bool) -> Result<(), String> {
        if self.temperature.is_none()
            && (self.top_p.is_some() || self.top_k.is_some())
            && !file_samples_with_temperature
        {
            return Err(
                "error: top_p/top_k without a temperature, and the model's generation_config.json \
                 has no do_sample: true with a valid temperature — the engine would sample at its \
                 own default temperature 1.0"
                    .to_owned(),
            );
        }
        Ok(())
    }
}

/// `presence_penalty` → `presence-penalty`.
fn flag(key: &str) -> String {
    key.replace('_', "-")
}

/// The sampling flags of `rv add` and `rv patch`. Doc comments
/// are `--help` text.
#[allow(clippy::doc_markdown)]
#[derive(clap::Args, Debug, Clone, Default)]
pub struct SamplingFlags {
    /// sampling temperature default for this model, 0-2 (0 = greedy; a client's value wins)
    #[arg(long, allow_hyphen_values = true)]
    pub temperature: Option<f64>,
    /// nucleus cut default, (0, 1]
    #[arg(long, allow_hyphen_values = true)]
    pub top_p: Option<f64>,
    /// top-k cut default, at least 1
    #[arg(long)]
    pub top_k: Option<u64>,
    /// repetition penalty default, 1.0-2.0 (1.0 turns the server's 1.1 net off)
    #[arg(long, allow_hyphen_values = true)]
    pub repetition_penalty: Option<f64>,
    /// presence penalty default, -2 to 2 (not sent on greedy requests or on NPU)
    #[arg(long, allow_hyphen_values = true)]
    pub presence_penalty: Option<f64>,
    /// frequency penalty default, -2 to 2 (not sent on greedy requests or on NPU)
    #[arg(long, allow_hyphen_values = true)]
    pub frequency_penalty: Option<f64>,
}

impl SamplingFlags {
    /// The flags as operator values.
    #[must_use]
    pub fn operator(&self) -> Operator {
        Operator {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            repetition_penalty: self.repetition_penalty,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
        }
    }

    /// Any sampling value flag was given.
    #[must_use]
    pub fn any(&self) -> bool {
        !self.operator().is_empty()
    }
}

/// The `sampling` value `rv patch` sends: `null` for `--clear-sampling`, else
/// the operator values the server reports (`current`, the admin `sampling`
/// field) with the flags over them (the server replaces the
/// whole object, P3, so values the user did not mention must be resent).
/// Returns `(before, after, body value)`.
///
/// # Errors
/// The merged object breaks a range rule or the `do_sample` rule (inferred
/// from the server's profile: the file samples with a temperature exactly when
/// a temperature without an operator value comes from `generation_config.json`).
pub fn merge_for_patch(
    current: &Value,
    flags: &SamplingFlags,
    clear: bool,
) -> Result<(Operator, Operator, Value), String> {
    let before = Operator::from_admin(current);
    if clear {
        return Ok((before, Operator::default(), Value::Null));
    }
    let after = before.overlay(&flags.operator());
    // The merged object can fail on values the user did not pass: name them (review F6).
    let with_current = |e: String| {
        format!(
            "{e} (the model's current operator values: {}; --clear-sampling drops them)",
            before.summary()
        )
    };
    after.validate().map_err(with_current)?;
    let file_samples = current["temperature"]["source"] == "generation_config.json";
    after
        .check_do_sample(file_samples)
        .map_err(|e| with_current(format!("{e}. Add --temperature")))?;
    Ok((before, after, after.to_json()))
}

// ---------------------------------------------------------------------------
// Local preview: the file layer + resolver mirror
// ---------------------------------------------------------------------------

/// What a model's `generation_config.json` says about sampling (mirror of the
/// server's `FileSampling`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FileLayer {
    /// The `do_sample` package: all `None` unless `do_sample: true` and every
    /// field passes its gate.
    pub temperature: Option<f64>,
    /// See [`temperature`](Self::temperature).
    pub top_p: Option<f64>,
    /// See [`temperature`](Self::temperature); `0` counts as unset.
    pub top_k: Option<u64>,
    /// The file says `do_sample: true`.
    pub do_sample: bool,
    /// Kept only in `(1.0, 2.0]`, read outside the `do_sample` gate.
    pub repetition_penalty: Option<f64>,
    /// `_from_model_config: true` (a transformers exporter stub).
    pub exporter_stub: bool,
    /// The file was there and parsed (rv only: the hint names a missing file).
    pub found: bool,
}

impl FileLayer {
    /// `do_sample: true` with a temperature that passed the gate (F6).
    #[must_use]
    pub fn samples_with_temperature(&self) -> bool {
        self.do_sample && self.temperature.is_some()
    }
}

/// An `f32` as the `f64` with its shortest decimal form — the server narrows
/// file and config values to `f32` and prints them this way.
fn dec(v: f32) -> f64 {
    v.to_string().parse().unwrap_or_else(|_| f64::from(v))
}

/// The file layer of `model_dir`: `None` when the directory does not exist
/// (no preview); an empty layer when the file is missing or unparseable.
#[must_use]
pub fn read_file_layer(model_dir: &Path) -> Option<FileLayer> {
    if !model_dir.is_dir() {
        return None;
    }
    let cfg = std::fs::read_to_string(model_dir.join("generation_config.json"))
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok());
    Some(
        cfg.as_ref()
            .map_or_else(FileLayer::default, file_layer_from),
    )
}

/// The file layer of a parsed `generation_config.json` (mirror of
/// `load_file_sampling` + `generation_defaults_from`).
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn file_layer_from(cfg: &Value) -> FileLayer {
    let do_sample = cfg.get("do_sample").and_then(Value::as_bool) == Some(true);
    let exporter_stub = cfg.get("_from_model_config").and_then(Value::as_bool) == Some(true);
    let repetition_penalty = match cfg.get("repetition_penalty").map(Value::as_f64) {
        Some(Some(p)) if (p - 1.0).abs() < f64::EPSILON => None,
        Some(Some(p)) if p > 1.0 && p <= 2.0 => Some(dec(p as f32)),
        _ => None,
    };
    let package = || -> Option<(Option<f64>, Option<f64>, Option<u64>)> {
        if !do_sample {
            return None;
        }
        // Absent = not set; present but not a number = the whole package drops.
        let read = |key: &str| match cfg.get(key) {
            None => Ok(None),
            Some(v) => v.as_f64().map(|x| Some(x as f32)).ok_or(()),
        };
        let t = read("temperature").ok()?;
        let p = read("top_p").ok()?;
        let k = match cfg.get("top_k") {
            None => None,
            Some(v) => match v.as_f64() {
                Some(f) if f >= 0.0 && f.fract() == 0.0 && f <= f64::from(u32::MAX) => {
                    Some(f as u64)
                }
                _ => return None,
            },
        }
        .filter(|&k| k > 0);
        if t.is_some_and(|t| !(t > 0.0 && t <= 2.0)) || p.is_some_and(|p| !(p > 0.0 && p <= 1.0)) {
            return None;
        }
        Some((t.map(dec), p.map(dec), k))
    };
    let (temperature, top_p, top_k) = package().unwrap_or_default();
    FileLayer {
        temperature,
        top_p,
        top_k,
        do_sample,
        repetition_penalty,
        exporter_stub,
        found: true,
    }
}

/// The profile the server would resolve from `op` over `file`, in the admin
/// `sampling` field's shape (mirror of `SamplingProfile::resolve` +
/// `admin_json`), so the one formatter prints both. `npu` adds
/// `ignored_on_device` as the server does on an NPU.
#[must_use]
#[allow(clippy::similar_names)] // top_p / top_k are the API's own names
pub fn preview_json(op: &Operator, file: &FileLayer, npu: bool) -> Value {
    let pick = |o: Option<f64>, f: Option<f64>| match (o, f) {
        (Some(v), _) => (Some(v), "config"),
        (None, Some(v)) => (Some(v), "generation_config.json"),
        (None, None) => (None, "none"),
    };
    let (temperature, t_src) = pick(op.temperature, file.temperature);
    let (mut top_p, mut p_src) = pick(op.top_p, file.top_p);
    let (mut top_k, mut k_src) = match (op.top_k, file.top_k) {
        (Some(v), _) => (Some(v), "config"),
        (None, Some(v)) => (Some(v), "generation_config.json"),
        (None, None) => (None, "none"),
    };
    if temperature.is_some_and(|t| t <= 0.0) {
        (top_p, p_src, top_k, k_src) = (None, "none", None, "none");
    }
    let (rep, rep_src) = match (op.repetition_penalty, file.repetition_penalty) {
        (Some(v), _) => (v, "config"),
        (None, Some(v)) => (v, "generation_config.json"),
        (None, None) => (DEFAULT_REPETITION_PENALTY, "server default"),
    };
    let samples = match temperature {
        Some(t) => t > 0.0,
        None => top_p.is_some_and(|p| p < 1.0) || top_k.is_some_and(|k| k > 0),
    };
    let shown = |v: Option<f64>, src: &'static str| match v {
        Some(v) => (Some(v), src),
        None if samples => (Some(ENGINE_DEFAULT), "engine default"),
        None => (None, "none"),
    };
    let (shown_t, shown_t_src) = shown(temperature, t_src);
    let (shown_p, shown_p_src) = shown(top_p, p_src);
    let pen =
        |v: Option<f64>| json!({"value": v, "source": if v.is_some() { "config" } else { "none" }});
    let mut out = json!({
        "mode": if samples { "sample" } else { "greedy" },
        "temperature": {"value": shown_t, "source": shown_t_src},
        "top_p": {"value": shown_p, "source": shown_p_src},
        "top_k": {"value": top_k, "source": k_src},
        "repetition_penalty": {"value": rep, "source": rep_src},
        "presence_penalty": pen(op.presence_penalty),
        "frequency_penalty": pen(op.frequency_penalty),
        "tools": "greedy",
    });
    if let Some(map) = out.as_object_mut() {
        if npu {
            map.insert(
                "ignored_on_device".to_owned(),
                json!(["presence_penalty", "frequency_penalty"]),
            );
        }
        if file.exporter_stub {
            map.insert(
                "note".to_owned(),
                json!("generation_config.json is an exporter stub (no sampling recommendation)"),
            );
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The formatter
// ---------------------------------------------------------------------------

/// Where a rendered profile comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// The server's admin `sampling` field.
    Server,
    /// [`preview_json`] from the local `generation_config.json`.
    Local {
        /// The file was there and parsed.
        file_found: bool,
    },
}

/// A number as the plan writes it: `1.0`, `0.15`, `1.1` (never `1`).
fn num(v: f64) -> String {
    format!("{v:?}")
}

/// One rendered profile: the line, an optional note, at most one hint (or
/// warning), and the footer. Plain text; the caller colours it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileText {
    /// `sampling: ...`.
    pub line: String,
    /// `note: ...` (the exporter stub), when the server or the file says so.
    pub note: Option<String>,
    /// `hint: ...` or `warn: ...`; at most one.
    pub hint: Option<String>,
    /// `values the client sends win · realtime: profile` (+ the greedy penalty rule).
    pub footer: String,
}

impl ProfileText {
    /// The lines in print order (note, hint and footer indented).
    #[must_use]
    pub fn lines(&self) -> Vec<String> {
        let mut out = vec![self.line.clone()];
        out.extend(self.note.iter().map(|n| format!("  {n}")));
        out.extend(self.hint.iter().map(|h| format!("  {h}")));
        out.push(format!("  {}", self.footer));
        out
    }

    /// The hint is a warning (`warn: ...`).
    #[must_use]
    pub fn hint_is_warning(&self) -> bool {
        self.hint.as_deref().is_some_and(|h| h.starts_with("warn:"))
    }
}

/// Renders an admin `sampling` object (or a [`preview_json`] one). Rules:
/// - fixed order T, `top_p`, `top_k`, `repetition_penalty`, then the
///   presence/frequency penalties when set;
/// - every value carries its source, except that a run of consecutive
///   `generation_config.json` values shares one label at its end;
/// - greedy shows `greedy` and leaves out T/`top_p`/`top_k` unless the
///   operator set T 0; an unset `top_k` while sampling shows `top_k — none`;
/// - `tools → greedy` when sampling and the server pins tools to greedy.
#[must_use]
pub fn render(model: &str, s: &Value, origin: Origin) -> ProfileText {
    let local = origin != Origin::Server;
    let src = |key: &str| s[key]["source"].as_str().unwrap_or("none").to_owned();
    let fval = |key: &str| s[key]["value"].as_f64();
    let sample = s["mode"] == "sample";
    let ignored: Vec<&str> = s["ignored_on_device"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();

    // (text, source label or None) per item.
    let mut items: Vec<(String, Option<String>)> = Vec::new();
    if !sample {
        items.push(("greedy".to_owned(), None));
    }
    if (sample || src("temperature") == "config")
        && let Some(t) = fval("temperature")
    {
        items.push((format!("T {}", num(t)), Some(src("temperature"))));
    }
    if sample {
        match fval("top_p") {
            Some(p) => items.push((format!("top_p {}", num(p)), Some(src("top_p")))),
            None => items.push(("top_p — none".to_owned(), None)),
        }
        match s["top_k"]["value"].as_u64() {
            Some(k) => items.push((format!("top_k {k}"), Some(src("top_k")))),
            None => items.push(("top_k — none".to_owned(), None)),
        }
    }
    if let Some(r) = fval("repetition_penalty") {
        items.push((
            format!("repetition_penalty {}", num(r)),
            Some(src("repetition_penalty")),
        ));
    }
    for key in ["presence_penalty", "frequency_penalty"] {
        if let Some(v) = fval(key) {
            let label = if ignored.contains(&key) {
                format!("{}, ignored on this device", src(key))
            } else {
                src(key)
            };
            items.push((format!("{key} {}", num(v)), Some(label)));
        }
    }
    let mut parts = Vec::new();
    for (i, (text, label)) in items.iter().enumerate() {
        let next = items.get(i + 1).and_then(|(_, l)| l.as_deref());
        match label.as_deref() {
            Some("generation_config.json") if next == Some("generation_config.json") => {
                parts.push(text.clone());
            }
            Some(l) => parts.push(format!("{text} ({l})")),
            None => parts.push(text.clone()),
        }
    }
    if sample && s["tools"] == "greedy" {
        parts.push("tools → greedy".to_owned());
    }
    let head = if local {
        "sampling (local preview, text_gen/vision): "
    } else {
        "sampling: "
    };
    let line = format!("{head}{}", parts.join(" · "));

    let note = s["note"].as_str().map(|n| format!("note: {n}"));
    let hint = if !sample && src("temperature") == "none" {
        // A local preview is of a model not registered yet: the flags go on `rv add` there.
        let verb = if local { "add" } else { "patch" };
        // Cause-neutral (review F9): no do_sample, a dropped package and a missing file all run
        // greedy; only a local preview knows the file is missing.
        let why = if origin == (Origin::Local { file_found: false }) {
            "no readable generation_config.json"
        } else {
            "generation_config.json gives no sampling settings"
        };
        Some(format!(
            "hint: {why}, so this model runs greedy; the model card may recommend sampling: \
             rv {verb} {model} --temperature 0.15 --top-p 0.95"
        ))
    } else if sample
        && src("temperature") == "config"
        && matches!(src("top_p").as_str(), "engine default" | "none")
        && src("top_k") == "none"
    {
        Some("warn: samples over the full vocabulary; consider --top-p".to_owned())
    } else {
        None
    };
    let operator_pf = ["presence_penalty", "frequency_penalty"]
        .iter()
        .any(|k| fval(k).is_some() && src(k) == "config");
    let mut footer = "values the client sends win · realtime: profile".to_owned();
    if operator_pf {
        footer.push_str(
            " · presence/frequency are not sent on a greedy request (tools, or a client T 0)",
        );
    }
    ProfileText {
        line,
        note,
        hint,
        footer,
    }
}

/// Prints a rendered profile with the verb's colours (the warning yellow,
/// note and footer dim).
pub fn print(style: crate::ansi::Style, p: &ProfileText) {
    println!("{}", p.line);
    if let Some(n) = &p.note {
        println!("  {}", style.dim(n));
    }
    if let Some(h) = &p.hint {
        let text = if p.hint_is_warning() {
            style.yellow(h)
        } else {
            h.clone()
        };
        println!("  {text}");
    }
    println!("  {}", style.dim(&p.footer));
}

/// What an admin model entry says about sampling.
#[derive(Debug, Clone, PartialEq)]
pub enum EntryProfile<'a> {
    /// Not a text model: no profile, no warning.
    NotApplicable,
    /// A text model on a server without the field (< 0.9.1).
    OldServer,
    /// The server's profile.
    Profile(&'a Value),
}

/// Classify an admin model entry: the kind check first, then the field (P9/F2).
#[must_use]
pub fn entry_profile(entry: &Value) -> EntryProfile<'_> {
    if !kind_has_sampling(entry["kind"].as_str().unwrap_or("")) {
        return EntryProfile::NotApplicable;
    }
    match entry.get("sampling").filter(|s| s.is_object()) {
        Some(s) => EntryProfile::Profile(s),
        None => EntryProfile::OldServer,
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::float_cmp)]
mod tests;
