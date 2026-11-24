//! The kind a model directory's own files prove — a mirror of the server's
//! `detect_kind_from_markers` + `kind_from_hf_config`
//! (`src/model_manager/lifecycle.rs`), used to keep the sampling preview off a
//! non-text model on disk that is not registered yet (review F2). `None` = no
//! marker: the server then registers the model as `text_gen` (unless config.json
//! names its kind). The tests port the server's detection tests by name; keep
//! them in step.

use std::path::Path;

use serde_json::Value;

/// The `config.json` rule: `*ForTextToSpeech` is `tts`; of the encoder families, a
/// one-label `*ForSequenceClassification` is `reranking` and a bare `*Model` is `embedding`.
fn kind_from_hf_config(model_dir: &Path) -> Option<&'static str> {
    const ENCODERS: [&str; 8] = [
        "bert",
        "xlm-roberta",
        "roberta",
        "distilbert",
        "electra",
        "modernbert",
        "deberta-v2",
        "nomic_bert",
    ];
    let text = std::fs::read_to_string(model_dir.join("config.json")).ok()?;
    let cfg: Value = serde_json::from_str(&text).ok()?;
    let arch = cfg.get("architectures")?.as_array()?.first()?.as_str()?;
    let family = cfg.get("model_type").and_then(Value::as_str)?;
    if arch.ends_with("ForTextToSpeech") {
        return Some("tts");
    }
    if !ENCODERS.contains(&family) {
        return None;
    }
    if arch.ends_with("ForSequenceClassification") {
        let labels = cfg
            .get("id2label")
            .and_then(Value::as_object)
            .map(serde_json::Map::len)
            .or_else(|| {
                cfg.get("num_labels")
                    .and_then(Value::as_u64)
                    .map(|n| usize::try_from(n).unwrap_or(usize::MAX))
            });
        return (labels.unwrap_or(1) == 1).then_some("reranking");
    }
    arch.ends_with("Model").then_some("embedding")
}

/// The kind `model_dir`'s files confidently prove, in the server's order (TTS markers,
/// `config.json`, Whisper's encoder + decoder, the diffusion layout, the VLM's vision IR).
#[must_use]
pub fn detect_kind_from_markers(model_dir: &Path) -> Option<&'static str> {
    let file = |f: &str| model_dir.join(f).is_file();
    if file("openvino_vocoder.xml") || file("openvino_postnet.xml") {
        return Some("tts");
    }
    if let Some(kind) = kind_from_hf_config(model_dir) {
        return Some(kind);
    }
    if file("openvino_encoder_model.xml") && file("openvino_decoder_model.xml") {
        return Some("stt");
    }
    if file("model_index.json")
        && (model_dir.join("unet").is_dir() || model_dir.join("transformer").is_dir())
    {
        return Some("image_gen");
    }
    if file("openvino_vision_embeddings_model.xml") {
        return Some("vision");
    }
    None
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn model_dir_with(config: &str, files: &[&str]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.json"), config).expect("write");
        for f in files {
            std::fs::write(dir.path().join(f), b"").expect("write");
        }
        dir
    }

    /// lifecycle.rs `kind_is_sniffed_for_tts_reranker_and_embedder_but_not_for_decoders`.
    #[test]
    fn kind_is_sniffed_for_tts_reranker_and_embedder_but_not_for_decoders() {
        let speecht5 = model_dir_with(
            r#"{"architectures":["SpeechT5ForTextToSpeech"],"model_type":"speecht5"}"#,
            &[
                "openvino_encoder_model.xml",
                "openvino_decoder_model.xml",
                "openvino_postnet.xml",
                "openvino_vocoder.xml",
            ],
        );
        assert_eq!(detect_kind_from_markers(speecht5.path()), Some("tts"));
        let reranker = model_dir_with(
            r#"{"architectures":["XLMRobertaForSequenceClassification"],"model_type":"xlm-roberta","id2label":{"0":"LABEL_0"}}"#,
            &["openvino_model.xml"],
        );
        assert_eq!(detect_kind_from_markers(reranker.path()), Some("reranking"));
        let e5 = model_dir_with(
            r#"{"architectures":["XLMRobertaModel"],"model_type":"xlm-roberta"}"#,
            &["openvino_model.xml"],
        );
        assert_eq!(detect_kind_from_markers(e5.path()), Some("embedding"));
        let classifier = model_dir_with(
            r#"{"architectures":["BertForSequenceClassification"],"model_type":"bert","id2label":{"0":"a","1":"b","2":"c"}}"#,
            &[],
        );
        assert_eq!(detect_kind_from_markers(classifier.path()), None);
        let llm = model_dir_with(
            r#"{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}"#,
            &["openvino_model.xml"],
        );
        assert_eq!(detect_kind_from_markers(llm.path()), None);
        let gpt2 = model_dir_with(
            r#"{"architectures":["GPT2LMHeadModel"],"model_type":"gpt2"}"#,
            &[],
        );
        assert_eq!(detect_kind_from_markers(gpt2.path()), None);
        let whisper = model_dir_with(
            r#"{"architectures":["WhisperForConditionalGeneration"],"model_type":"whisper"}"#,
            &["openvino_encoder_model.xml", "openvino_decoder_model.xml"],
        );
        assert_eq!(detect_kind_from_markers(whisper.path()), Some("stt"));
    }

    /// lifecycle.rs `detect_kind_from_markers_is_none_for_a_marker_less_directory`.
    #[test]
    fn detect_kind_from_markers_is_none_for_a_marker_less_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("openvino_model.xml"), b"").expect("write");
        assert_eq!(detect_kind_from_markers(dir.path()), None);
    }

    /// lifecycle.rs `detect_kind_from_markers_is_some_for_a_marker_that_exists`.
    #[test]
    fn detect_kind_from_markers_is_some_for_a_marker_that_exists() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("openvino_vision_embeddings_model.xml"), b"")
            .expect("write");
        assert_eq!(detect_kind_from_markers(dir.path()), Some("vision"));
    }

    /// The diffusion layout (no server test of its own; the rule at lifecycle.rs's
    /// `detect_kind_from_markers`): `model_index.json` plus `unet/` or `transformer/`.
    #[test]
    fn diffusion_layout_is_image_gen_only_with_a_submodel_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("model_index.json"), b"{}").expect("write");
        assert_eq!(detect_kind_from_markers(dir.path()), None);
        std::fs::create_dir(dir.path().join("unet")).expect("mkdir");
        assert_eq!(detect_kind_from_markers(dir.path()), Some("image_gen"));
    }
}
