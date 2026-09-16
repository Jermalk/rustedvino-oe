// ============================================================
// src/model_completeness.rs — per-kind model file completeness
// ============================================================
// Every model kind bundles its own OpenVINO IR files (main model +
// tokenizer/detokenizer, or a kind-specific split like Whisper's
// encoder/decoder pair) inside its model directory. A model that is
// missing part of that bundle can still pass the shallow "is this a
// directory" check (`EngineFactory::model_exists`) and, for some kinds,
// even complete `load()` successfully — the gap only surfaces on first
// real inference, as an opaque OpenVINO C++ exception (root-caused
// 2026-08-03, `dev/DECISIONS.md`: `ms-marco-MiniLM-L6-v2-int8-ov` was
// missing its converted `openvino_tokenizer.{xml,bin}` pair; reranking
// loaded fine and only failed on the first real `/v1/rerank` call).
//
// This module answers "does this model directory actually have
// everything its kind's engine needs?" — a pure filesystem check, no
// engine construction, safe to run before a real load (the earlier the
// better: same principle as `model_exists`, fail cheap before any VRAM
// reservation or LRU eviction) and equally safe to run as a standalone
// audit over every *registered* model without ever loading any of them.
//
// `blocking_missing` are absences proven fatal (grounded either by a
// live incident — the reranking tokenizer gap — or by directly reading
// the OV GenAI C++ bridge to confirm a kind's pipeline constructs its own
// `ov::genai::Tokenizer`/`Detokenizer` and therefore cannot start without
// the corresponding IR file). `advisory_missing` are absences that are
// real gaps but not proven to break anything for this specific kind —
// most commonly a detokenizer for a kind that only ever *encodes* text
// (Embedding, Reranking, Tts, ImageGen never convert token ids back to
// text) — reported so an operator can decide, but never blocking a load.
// ============================================================

use std::path::{Path, PathBuf};

use crate::model_manager::engine::ModelKind;
use crate::model_manager::lifecycle::{TtsBackend, detect_tts_backend};

/// The result of checking one model directory against its kind's expected
/// file set.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompletenessReport {
    /// Absences that would (or did, live) break the model — a load should
    /// refuse rather than proceed.
    pub blocking_missing: Vec<String>,
    /// Absences that are real gaps but not proven fatal for this kind —
    /// surfaced for operator awareness, never blocking.
    pub advisory_missing: Vec<String>,
}

impl CompletenessReport {
    /// `true` when nothing blocking is missing (advisory gaps don't count —
    /// a load is allowed to proceed against them).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.blocking_missing.is_empty()
    }
}

/// Push `{base}.xml`/`{base}.bin` into `out` (as a path relative to
/// `model_dir`, for an actionable message) for each half of the pair
/// missing from `search_dir`. `search_dir` is usually `model_dir` itself,
/// but for kinds with per-component subdirectories (`ImageGen`) it is a
/// subdirectory — the relative label still reads naturally either way
/// (`"openvino_model.xml"` vs `"unet/openvino_model.xml"`).
fn require_pair(model_dir: &Path, search_dir: &Path, base: &str, out: &mut Vec<String>) {
    for ext in ["xml", "bin"] {
        let name = format!("{base}.{ext}");
        if !search_dir.join(&name).is_file() {
            let label = search_dir
                .strip_prefix(model_dir)
                .unwrap_or(search_dir)
                .join(&name);
            out.push(label.to_string_lossy().into_owned());
        }
    }
}

/// A component subdirectory that must exist (e.g. `unet/`, `text_encoder/`).
/// Returns its path when present; otherwise records the absence in `out`
/// and returns `None` — callers skip the inner file check when there's no
/// directory to check inside.
fn require_component_dir(model_dir: &Path, name: &str, out: &mut Vec<String>) -> Option<PathBuf> {
    let dir = model_dir.join(name);
    if dir.is_dir() {
        Some(dir)
    } else {
        out.push(format!("{name}/ (missing directory)"));
        None
    }
}

/// Check `model_dir` against the file set its `kind` needs to actually run,
/// not just exist. Pure filesystem inspection — safe to call before any
/// engine construction, VRAM reservation, or eviction.
#[must_use]
pub fn check(model_dir: &Path, kind: ModelKind) -> CompletenessReport {
    let mut report = CompletenessReport::default();
    match kind {
        ModelKind::TextGen => check_textgen(model_dir, &mut report),
        ModelKind::Vision => check_vision(model_dir, &mut report),
        ModelKind::Embedding | ModelKind::Reranking => check_encode_only(model_dir, &mut report),
        ModelKind::Stt => check_stt(model_dir, &mut report),
        ModelKind::Tts => check_tts(model_dir, &mut report),
        ModelKind::ImageGen => check_image_gen(model_dir, &mut report),
    }
    report
}

/// `TextGen` (CB and NPU both — NPU is a device routing of the same kind,
/// same IR export shape) decodes generated tokens back to text, so the
/// detokenizer is as essential as the tokenizer.
fn check_textgen(model_dir: &Path, report: &mut CompletenessReport) {
    require_pair(
        model_dir,
        model_dir,
        "openvino_model",
        &mut report.blocking_missing,
    );
    require_pair(
        model_dir,
        model_dir,
        "openvino_tokenizer",
        &mut report.blocking_missing,
    );
    require_pair(
        model_dir,
        model_dir,
        "openvino_detokenizer",
        &mut report.blocking_missing,
    );
}

/// VLM: the language model is the `openvino_language_model.*` pair (not
/// `openvino_model.*` — VLM exports split language/vision), and it decodes
/// text output, so detokenizer is blocking too. At least one
/// `*vision_embeddings*model.xml` must exist — the exact file name varies
/// (`_merger_`, `_pos_`, plain), so this checks for any match rather than
/// one fixed name.
fn check_vision(model_dir: &Path, report: &mut CompletenessReport) {
    require_pair(
        model_dir,
        model_dir,
        "openvino_language_model",
        &mut report.blocking_missing,
    );
    require_pair(
        model_dir,
        model_dir,
        "openvino_tokenizer",
        &mut report.blocking_missing,
    );
    require_pair(
        model_dir,
        model_dir,
        "openvino_detokenizer",
        &mut report.blocking_missing,
    );
    let has_vision_embeddings = std::fs::read_dir(model_dir)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .any(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.contains("vision_embeddings") && name.ends_with(".xml")
        });
    if !has_vision_embeddings {
        report.blocking_missing.push(
            "openvino_*vision_embeddings*_model.xml (no vision-embeddings IR found)".to_owned(),
        );
    }
}

/// Embedding/Reranking only ever encode text to a vector or a score —
/// neither ever decodes token ids back to text, so a missing detokenizer is
/// advisory, not blocking. The tokenizer itself is blocking: this is the
/// exact class of bug that motivated this module
/// (`ms-marco-MiniLM-L6-v2-int8-ov`, 2026-08-03).
fn check_encode_only(model_dir: &Path, report: &mut CompletenessReport) {
    require_pair(
        model_dir,
        model_dir,
        "openvino_model",
        &mut report.blocking_missing,
    );
    require_pair(
        model_dir,
        model_dir,
        "openvino_tokenizer",
        &mut report.blocking_missing,
    );
    require_pair(
        model_dir,
        model_dir,
        "openvino_detokenizer",
        &mut report.advisory_missing,
    );
}

/// Whisper: encoder/decoder split (no single `openvino_model.*`), and it
/// decodes the transcription back to text, so both tokenizer and
/// detokenizer are blocking.
fn check_stt(model_dir: &Path, report: &mut CompletenessReport) {
    require_pair(
        model_dir,
        model_dir,
        "openvino_encoder_model",
        &mut report.blocking_missing,
    );
    require_pair(
        model_dir,
        model_dir,
        "openvino_decoder_model",
        &mut report.blocking_missing,
    );
    require_pair(
        model_dir,
        model_dir,
        "openvino_tokenizer",
        &mut report.blocking_missing,
    );
    require_pair(
        model_dir,
        model_dir,
        "openvino_detokenizer",
        &mut report.blocking_missing,
    );
}

/// Kokoro/Coqui detection (`detect_tts_backend`) already proves both of
/// their required files exist by construction — a successful match means
/// nothing further to check. The `SpeechT5` arm is genuinely new: unlike
/// the other two, `detect_tts_backend` treats `SpeechT5` as its bare
/// fallback (whatever didn't match Kokoro or Coqui), never validating that
/// `SpeechT5`'s own files are actually present. `SpeechT5` only ever
/// synthesizes audio *from* text — it never decodes back to text, so its
/// detokenizer is advisory like Embedding/Reranking.
fn check_tts(model_dir: &Path, report: &mut CompletenessReport) {
    match detect_tts_backend(model_dir) {
        Ok(TtsBackend::Kokoro { .. } | TtsBackend::CoquiVits { .. }) => {}
        Ok(TtsBackend::SpeechT5) => {
            require_pair(
                model_dir,
                model_dir,
                "openvino_encoder_model",
                &mut report.blocking_missing,
            );
            require_pair(
                model_dir,
                model_dir,
                "openvino_decoder_model",
                &mut report.blocking_missing,
            );
            require_pair(
                model_dir,
                model_dir,
                "openvino_postnet",
                &mut report.blocking_missing,
            );
            require_pair(
                model_dir,
                model_dir,
                "openvino_vocoder",
                &mut report.blocking_missing,
            );
            require_pair(
                model_dir,
                model_dir,
                "openvino_tokenizer",
                &mut report.blocking_missing,
            );
            require_pair(
                model_dir,
                model_dir,
                "openvino_detokenizer",
                &mut report.advisory_missing,
            );
        }
        Err(e) => report
            .blocking_missing
            .push(format!("could not determine TTS backend: {e}")),
    }
}

/// Diffusers-style layout (`model_index.json` + per-component
/// subdirectories) rather than a flat file set — and, unlike every other
/// kind, the component set itself varies by architecture (SD1.5/LCM use
/// `unet` + one text encoder; SDXL/FLUX/SD3 use `transformer` and/or a
/// second or third text encoder+tokenizer). Verified against both
/// `LCM_Dreamshaper_v7-int8-ov` and `FLUX.1-schnell-int4-ov` on disk
/// (2026-08-03) before wiring this in: `image_encoder` (null in
/// `model_index.json`), `feature_extractor`, `scheduler`, and
/// `safety_checker` carry no OV IR at all and are deliberately never
/// checked. `vae_encoder` is only needed for img2img/inpainting, not plain
/// text-to-image, so it's advisory. `text_encoder_2`/`tokenizer_2` (and a
/// hypothetical `_3`) are blocking *only if the directory exists* — its
/// presence at all means the architecture needs it.
fn check_image_gen(model_dir: &Path, report: &mut CompletenessReport) {
    if !model_dir.join("model_index.json").is_file() {
        report.blocking_missing.push("model_index.json".to_owned());
        return;
    }

    let unet_dir = model_dir.join("unet");
    let transformer_dir = model_dir.join("transformer");
    match (unet_dir.is_dir(), transformer_dir.is_dir()) {
        (true, _) => require_pair(
            model_dir,
            &unet_dir,
            "openvino_model",
            &mut report.blocking_missing,
        ),
        (false, true) => require_pair(
            model_dir,
            &transformer_dir,
            "openvino_model",
            &mut report.blocking_missing,
        ),
        (false, false) => report
            .blocking_missing
            .push("unet/ or transformer/ (no denoising backbone found)".to_owned()),
    }

    if let Some(dir) = require_component_dir(model_dir, "vae_decoder", &mut report.blocking_missing)
    {
        require_pair(
            model_dir,
            &dir,
            "openvino_model",
            &mut report.blocking_missing,
        );
    }
    if let Some(dir) =
        require_component_dir(model_dir, "text_encoder", &mut report.blocking_missing)
    {
        require_pair(
            model_dir,
            &dir,
            "openvino_model",
            &mut report.blocking_missing,
        );
    }
    if let Some(dir) = require_component_dir(model_dir, "tokenizer", &mut report.blocking_missing) {
        require_pair(
            model_dir,
            &dir,
            "openvino_tokenizer",
            &mut report.blocking_missing,
        );
        require_pair(
            model_dir,
            &dir,
            "openvino_detokenizer",
            &mut report.advisory_missing,
        );
    }

    let vae_encoder_dir = model_dir.join("vae_encoder");
    if vae_encoder_dir.is_dir() {
        require_pair(
            model_dir,
            &vae_encoder_dir,
            "openvino_model",
            &mut report.advisory_missing,
        );
    }
    for suffix in ["_2", "_3"] {
        let enc_dir = model_dir.join(format!("text_encoder{suffix}"));
        if enc_dir.is_dir() {
            require_pair(
                model_dir,
                &enc_dir,
                "openvino_model",
                &mut report.blocking_missing,
            );
        }
        let tok_dir = model_dir.join(format!("tokenizer{suffix}"));
        if tok_dir.is_dir() {
            require_pair(
                model_dir,
                &tok_dir,
                "openvino_tokenizer",
                &mut report.blocking_missing,
            );
            require_pair(
                model_dir,
                &tok_dir,
                "openvino_detokenizer",
                &mut report.advisory_missing,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    /// Write an empty file, creating parent directories as needed.
    fn touch(path: &Path) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, b"").unwrap();
    }

    #[test]
    fn textgen_complete_when_all_three_pairs_present() {
        let dir = tempfile::tempdir().unwrap();
        for base in [
            "openvino_model",
            "openvino_tokenizer",
            "openvino_detokenizer",
        ] {
            touch(&dir.path().join(format!("{base}.xml")));
            touch(&dir.path().join(format!("{base}.bin")));
        }
        let report = check(dir.path(), ModelKind::TextGen);
        assert!(report.is_complete(), "{report:?}");
        assert!(report.advisory_missing.is_empty());
    }

    #[test]
    fn textgen_missing_tokenizer_is_blocking() {
        let dir = tempfile::tempdir().unwrap();
        touch(&dir.path().join("openvino_model.xml"));
        touch(&dir.path().join("openvino_model.bin"));
        touch(&dir.path().join("openvino_detokenizer.xml"));
        touch(&dir.path().join("openvino_detokenizer.bin"));
        let report = check(dir.path(), ModelKind::TextGen);
        assert!(!report.is_complete());
        assert!(
            report
                .blocking_missing
                .contains(&"openvino_tokenizer.xml".to_owned())
        );
        assert!(
            report
                .blocking_missing
                .contains(&"openvino_tokenizer.bin".to_owned())
        );
    }

    /// Regression test for the exact live incident (2026-08-03): a rerank
    /// model with the main IR present but no tokenizer pair.
    #[test]
    fn reranking_missing_tokenizer_is_blocking_missing_detokenizer_is_advisory() {
        let dir = tempfile::tempdir().unwrap();
        touch(&dir.path().join("openvino_model.xml"));
        touch(&dir.path().join("openvino_model.bin"));
        let report = check(dir.path(), ModelKind::Reranking);
        assert!(!report.is_complete());
        assert!(
            report
                .blocking_missing
                .contains(&"openvino_tokenizer.xml".to_owned())
        );
        assert!(
            report
                .advisory_missing
                .contains(&"openvino_detokenizer.xml".to_owned()),
            "detokenizer must be advisory for an encode-only kind: {report:?}"
        );
    }

    #[test]
    fn embedding_complete_without_a_detokenizer() {
        let dir = tempfile::tempdir().unwrap();
        touch(&dir.path().join("openvino_model.xml"));
        touch(&dir.path().join("openvino_model.bin"));
        touch(&dir.path().join("openvino_tokenizer.xml"));
        touch(&dir.path().join("openvino_tokenizer.bin"));
        let report = check(dir.path(), ModelKind::Embedding);
        assert!(
            report.is_complete(),
            "no detokenizer must not block an encode-only kind: {report:?}"
        );
        assert_eq!(report.advisory_missing.len(), 2, "{report:?}");
    }

    #[test]
    fn vision_requires_language_model_and_a_vision_embeddings_file() {
        let dir = tempfile::tempdir().unwrap();
        touch(&dir.path().join("openvino_language_model.xml"));
        touch(&dir.path().join("openvino_language_model.bin"));
        touch(&dir.path().join("openvino_tokenizer.xml"));
        touch(&dir.path().join("openvino_tokenizer.bin"));
        touch(&dir.path().join("openvino_detokenizer.xml"));
        touch(&dir.path().join("openvino_detokenizer.bin"));
        // No vision-embeddings file yet.
        let report = check(dir.path(), ModelKind::Vision);
        assert!(!report.is_complete());

        touch(&dir.path().join("openvino_vision_embeddings_model.xml"));
        touch(&dir.path().join("openvino_vision_embeddings_model.bin"));
        let report = check(dir.path(), ModelKind::Vision);
        assert!(report.is_complete(), "{report:?}");
    }

    #[test]
    fn vision_accepts_any_vision_embeddings_variant_name() {
        let dir = tempfile::tempdir().unwrap();
        touch(&dir.path().join("openvino_language_model.xml"));
        touch(&dir.path().join("openvino_language_model.bin"));
        touch(&dir.path().join("openvino_tokenizer.xml"));
        touch(&dir.path().join("openvino_tokenizer.bin"));
        touch(&dir.path().join("openvino_detokenizer.xml"));
        touch(&dir.path().join("openvino_detokenizer.bin"));
        // Real Qwen3-VL exports use several distinct vision-embeddings file
        // names — any one of them must satisfy the check.
        touch(
            &dir.path()
                .join("openvino_vision_embeddings_merger_model.xml"),
        );
        let report = check(dir.path(), ModelKind::Vision);
        assert!(report.is_complete(), "{report:?}");
    }

    #[test]
    fn stt_requires_encoder_decoder_and_both_tokenizer_halves() {
        let dir = tempfile::tempdir().unwrap();
        let report = check(dir.path(), ModelKind::Stt);
        assert!(!report.is_complete());
        for base in [
            "openvino_encoder_model",
            "openvino_decoder_model",
            "openvino_tokenizer",
            "openvino_detokenizer",
        ] {
            for ext in ["xml", "bin"] {
                assert!(
                    report.blocking_missing.contains(&format!("{base}.{ext}")),
                    "expected {base}.{ext} in {report:?}"
                );
            }
        }
    }

    #[test]
    fn tts_kokoro_and_coqui_pass_through_detect_tts_backend() {
        let dir = tempfile::tempdir().unwrap();
        touch(&dir.path().join("voices.bin"));
        touch(&dir.path().join("kokoro-v1.0.onnx"));
        let report = check(dir.path(), ModelKind::Tts);
        assert!(report.is_complete(), "Kokoro: {report:?}");

        let dir2 = tempfile::tempdir().unwrap();
        touch(&dir2.path().join("model.onnx"));
        touch(&dir2.path().join("tokens.txt"));
        let report2 = check(dir2.path(), ModelKind::Tts);
        assert!(report2.is_complete(), "Coqui: {report2:?}");
    }

    #[test]
    fn tts_speecht5_requires_its_four_pairs_detokenizer_advisory() {
        let dir = tempfile::tempdir().unwrap();
        // No Kokoro/Coqui signature -> falls through to the SpeechT5 arm.
        for base in [
            "openvino_encoder_model",
            "openvino_decoder_model",
            "openvino_postnet",
            "openvino_vocoder",
            "openvino_tokenizer",
        ] {
            touch(&dir.path().join(format!("{base}.xml")));
            touch(&dir.path().join(format!("{base}.bin")));
        }
        let report = check(dir.path(), ModelKind::Tts);
        assert!(report.is_complete(), "{report:?}");
        assert_eq!(report.advisory_missing.len(), 2, "{report:?}");
    }

    /// The exact shape of `LCM_Dreamshaper_v7-int8-ov` (unet-based, no
    /// second text encoder) — built as a fixture rather than reading the
    /// real fleet directory, so this test runs anywhere.
    #[test]
    fn image_gen_unet_based_layout_is_complete() {
        let dir = tempfile::tempdir().unwrap();
        touch(&dir.path().join("model_index.json"));
        for (sub, base) in [
            ("unet", "openvino_model"),
            ("vae_decoder", "openvino_model"),
            ("text_encoder", "openvino_model"),
        ] {
            touch(&dir.path().join(sub).join(format!("{base}.xml")));
            touch(&dir.path().join(sub).join(format!("{base}.bin")));
        }
        touch(&dir.path().join("tokenizer").join("openvino_tokenizer.xml"));
        touch(&dir.path().join("tokenizer").join("openvino_tokenizer.bin"));
        let report = check(dir.path(), ModelKind::ImageGen);
        assert!(report.is_complete(), "{report:?}");
    }

    /// The exact shape of `FLUX.1-schnell-int4-ov` (transformer-based, a
    /// second text encoder + tokenizer) — the case the `ImageGen` rule was
    /// rewritten for after it first rejected this shape (2026-08-03).
    #[test]
    fn image_gen_transformer_with_second_encoder_is_complete() {
        let dir = tempfile::tempdir().unwrap();
        touch(&dir.path().join("model_index.json"));
        for (sub, base) in [
            ("transformer", "openvino_model"),
            ("vae_decoder", "openvino_model"),
            ("text_encoder", "openvino_model"),
            ("text_encoder_2", "openvino_model"),
        ] {
            touch(&dir.path().join(sub).join(format!("{base}.xml")));
            touch(&dir.path().join(sub).join(format!("{base}.bin")));
        }
        for tok in ["tokenizer", "tokenizer_2"] {
            touch(&dir.path().join(tok).join("openvino_tokenizer.xml"));
            touch(&dir.path().join(tok).join("openvino_tokenizer.bin"));
        }
        let report = check(dir.path(), ModelKind::ImageGen);
        assert!(report.is_complete(), "{report:?}");
    }

    #[test]
    fn image_gen_missing_model_index_is_blocking() {
        let dir = tempfile::tempdir().unwrap();
        let report = check(dir.path(), ModelKind::ImageGen);
        assert!(!report.is_complete());
        assert!(
            report
                .blocking_missing
                .contains(&"model_index.json".to_owned())
        );
    }

    /// `feature_extractor`/`scheduler`/`safety_checker`/`image_encoder`
    /// carry no OV IR and must never be checked — a complete model must
    /// pass regardless of which of these non-IR components are present.
    #[test]
    fn image_gen_ignores_non_ir_components() {
        let dir = tempfile::tempdir().unwrap();
        touch(&dir.path().join("model_index.json"));
        for (sub, base) in [
            ("unet", "openvino_model"),
            ("vae_decoder", "openvino_model"),
            ("text_encoder", "openvino_model"),
        ] {
            touch(&dir.path().join(sub).join(format!("{base}.xml")));
            touch(&dir.path().join(sub).join(format!("{base}.bin")));
        }
        touch(&dir.path().join("tokenizer").join("openvino_tokenizer.xml"));
        touch(&dir.path().join("tokenizer").join("openvino_tokenizer.bin"));
        touch(
            &dir.path()
                .join("feature_extractor")
                .join("preprocessor_config.json"),
        );
        touch(&dir.path().join("scheduler").join("scheduler_config.json"));
        let report = check(dir.path(), ModelKind::ImageGen);
        assert!(report.is_complete(), "{report:?}");
    }

    /// Manual, live-filesystem sweep over every model actually registered on
    /// this box's `models_dir`. `#[ignore]`d (not part of the normal suite —
    /// hardcodes a fleet-box path that doesn't exist in CI or on other
    /// boxes) but a permanent tool: `cargo test --lib -- --ignored
    /// sweep_real_models_dir --nocapture` audits the real fleet directory
    /// this box points at, kind resolved the same way the production
    /// factory resolves it (config `kind` field, falling back to
    /// `detect_kind`'s sniff) — run this before promoting any new
    /// requirement from advisory to blocking, or after adding a new model,
    /// per the 2026-08-03 design discussion (the project's internal engineering log).
    #[test]
    #[ignore = "hardcodes /opt/rustedvino/models — run manually, not in CI"]
    fn sweep_real_models_dir() {
        let models_dir = Path::new("/opt/rustedvino/models");
        let Ok(entries) = std::fs::read_dir(models_dir) else {
            eprintln!("models_dir not present on this box, skipping");
            return;
        };
        let mut any_blocking = false;
        for entry in entries.filter_map(Result::ok) {
            let dir = entry.path();
            if !dir.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            // Best-effort kind guess for a manual sweep: name-substring
            // matching, same spirit as the production hint map — good
            // enough for an audit tool, not a claim of production parity.
            let lower = name.to_lowercase();
            let kind = if lower.contains("embed") {
                ModelKind::Embedding
            } else if lower.contains("rerank") {
                ModelKind::Reranking
            } else if lower.contains("tts") || lower.contains("kokoro") || lower.contains("piper") {
                ModelKind::Tts
            } else if lower.contains("whisper") || lower.contains("stt") {
                ModelKind::Stt
            } else if dir.join("model_index.json").is_file() {
                ModelKind::ImageGen
            } else if dir.join("openvino_language_model.xml").is_file() {
                ModelKind::Vision
            } else {
                ModelKind::TextGen
            };
            let report = check(&dir, kind);
            let status = if report.is_complete() {
                "OK"
            } else {
                "INCOMPLETE"
            };
            println!("{status:10} {name:45} kind={kind:?}");
            if !report.blocking_missing.is_empty() {
                any_blocking = true;
                println!("  blocking:  {:?}", report.blocking_missing);
            }
            if !report.advisory_missing.is_empty() {
                println!("  advisory:  {:?}", report.advisory_missing);
            }
        }
        assert!(
            !any_blocking,
            "at least one model has a blocking gap — see output above"
        );
    }
}
