// ============================================================
// src/model_manager/template.rs — chat-template loading
// ============================================================
// Loads a model's Jinja chat template from disk at load time. The template is
// stored on the model's record and handed to `prompt_builder::build_prompt`
// for every chat request (tool-schema injection + multi-turn formatting).
// ============================================================

use std::path::Path;

/// Read `config.json` from `model_dir` and compute the KV-cache bytes consumed
/// per token at **f16** precision.
///
/// Formula: `num_hidden_layers × num_key_value_heads × head_dim × 2 (K+V) × 2 (f16 bytes)`.
///
/// Deliberately uniform across all layers, including hybrid attention
/// architectures (Qwen3.5/3.6's `GatedDeltaNet`-mixed layers) where the
/// resulting `max_prompt_tokens` **overestimates real usable capacity** by
/// ~3.2-3.3x (the project's internal engineering log —
/// same root cause as `vllm-project/vllm#37121`): real per-token KV cost on
/// these architectures is *higher* than this formula predicts, not lower.
/// The overestimate isn't unique to hybrid architectures either — a plain
/// dense GQA model measured ~1.9x too optimistic against the same formula
/// (the project's internal engineering log),
/// smaller but still large enough to matter; root cause unconfirmed there
/// too (likely CB-scheduler block-allocation overhead this raw K+V byte
/// count has no visibility into). A
/// `layer_types`-aware correction was attempted and reverted — it went the
/// wrong direction (counting fewer layers *lowers* bytes-per-token, which
/// *raises* the token ceiling, worsening the overestimate) and no principled
/// derivation from `config.json` was found. Getting the static formula
/// exactly right for every architecture is not required once
/// `ModelManager::spawn_kv_wedge_recovery`'s runtime ratchet exists — the
/// formula only needs to be roughly right on first contact, since the ratchet
/// corrects it from a real observed ceiling after at most one wedge per
/// model. Do not attempt another static per-architecture correction here
/// without new data — see the ratchet's own doc comment.
///
/// The caller divides by the actual element size to get the precision-adjusted
/// per-token cost: `bytes_per_token = result * precision_bytes / 2`.
///
/// Required fields are read from the top level first, then from a nested
/// `text_config` / `llm_config` / `language_config` block — newer VLM configs
/// (e.g. Qwen3-VL) carry the language-model params only in the nested block.
///
/// Returns `0` when `config.json` is absent or any required field cannot be
/// parsed — the caller should disable the L0 gate for this model and log a
/// warning rather than treating it as a fatal error.
pub(crate) fn read_kv_bytes_per_token_f16(model_dir: &Path) -> usize {
    kv_bytes_impl(model_dir).unwrap_or(0)
}

fn kv_bytes_impl(model_dir: &Path) -> Option<usize> {
    let raw = std::fs::read_to_string(model_dir.join("config.json")).ok()?;
    let cfg = serde_json::from_str::<serde_json::Value>(&raw).ok()?;

    // Newer VLM configs nest the language-model params under a sub-object with
    // NO top-level copy: Qwen3-VL uses `text_config`, InternVL-style configs use
    // `llm_config`. Older VLMs (Qwen2.5-VL) duplicate them at the top level.
    // Resolve each key from the top level first, then fall back to those blocks —
    // per-key, so a partially-duplicated config still resolves cleanly.
    let get = |key: &str| -> Option<usize> {
        [
            Some(&cfg),
            cfg.get("text_config"),
            cfg.get("llm_config"),
            cfg.get("language_config"),
        ]
        .into_iter()
        .flatten()
        .find_map(|c| c.get(key).and_then(serde_json::Value::as_u64))
        .and_then(|v| usize::try_from(v).ok())
    };

    let layers = get("num_hidden_layers")?;
    let num_heads = get("num_attention_heads")?;
    // GQA models set this to fewer heads than num_attention_heads.
    let kv_heads = get("num_key_value_heads").unwrap_or(num_heads);
    let hidden_size = get("hidden_size")?;
    // Some models (e.g. Gemma) expose head_dim directly; others compute it.
    let head_dim = get("head_dim").unwrap_or_else(|| hidden_size / num_heads.max(1));

    // × 2  for K + V tensors
    // × 2  for f16 (2 bytes per element) — caller adjusts for u8/f32
    Some(layers * kv_heads * head_dim * 2 * 2)
}

/// Read `config.json` from `model_dir` and return the total number of experts
/// if this is a Mixture-of-Experts model, or `None` for dense models.
///
/// `MoE` is detected by the presence of `num_experts` (or `moe_num_experts`) with
/// a value > 1.  The same nested-block fallback as [`read_kv_bytes_per_token_f16`]
/// is applied — some VLM configs nest these fields under `text_config` /
/// `llm_config` / `language_config`.
///
/// Returns `None` when `config.json` is absent, unparseable, or the model is
/// not a sparse-expert architecture.
pub(crate) fn read_moe_num_experts(model_dir: &Path) -> Option<u64> {
    let raw = std::fs::read_to_string(model_dir.join("config.json")).ok()?;
    let cfg = serde_json::from_str::<serde_json::Value>(&raw).ok()?;

    let blocks = [
        Some(&cfg),
        cfg.get("text_config"),
        cfg.get("llm_config"),
        cfg.get("language_config"),
    ];

    for block in blocks.into_iter().flatten() {
        for key in &["num_experts", "moe_num_experts"] {
            if let Some(n) = block.get(key).and_then(serde_json::Value::as_u64)
                && n > 1
            {
                return Some(n);
            }
        }
    }
    None
}

/// Read `config.json` from `model_dir` and return `vocab_size`, if present.
///
/// Same nested-block fallback as [`read_moe_num_experts`] — some VLM configs
/// nest the language-model params under `text_config` / `llm_config` /
/// `language_config`.
///
/// Returns `None` when `config.json` is absent, unparseable, or `vocab_size`
/// is not present in any of the checked blocks. Used to gate speculative
/// decoding pairings on tokenizer identity (the project's internal engineering log
/// Part 4 gate 3) — an unreadable/absent `vocab_size` is a hard refusal there,
/// not a pass.
pub(crate) fn read_vocab_size(model_dir: &Path) -> Option<u64> {
    let raw = std::fs::read_to_string(model_dir.join("config.json")).ok()?;
    let cfg = serde_json::from_str::<serde_json::Value>(&raw).ok()?;

    let blocks = [
        Some(&cfg),
        cfg.get("text_config"),
        cfg.get("llm_config"),
        cfg.get("language_config"),
    ];

    blocks
        .into_iter()
        .flatten()
        .find_map(|block| block.get("vocab_size").and_then(serde_json::Value::as_u64))
}

/// Read `config.json` from `model_dir` and return the model's **native**
/// (trained/converted) context ceiling — `max_position_embeddings`, adjusted
/// for `rope_scaling`/`YaRN` extension when present. This is a hardware-
/// independent model property, unrelated to [`read_kv_bytes_per_token_f16`]'s
/// VRAM-derived `max_prompt_tokens` formula: a model's KV pool can easily
/// afford more tokens than the model was ever trained on (confirmed live
/// 2026-08-23, the project's internal engineering log —
/// `qwen3-4b-int4-ov`'s formula promised 142k-306k tokens depending on pool
/// size, against a real trained ceiling of 40,960).
///
/// Same nested-block fallback as [`read_moe_num_experts`] — some VLM configs
/// nest `max_position_embeddings` under `text_config` / `llm_config` /
/// `language_config` with no top-level copy.
///
/// `rope_scaling` handling: HF configs are inconsistent about whether
/// `max_position_embeddings` is already bumped to the post-extension value.
/// When `rope_scaling.factor` and `rope_scaling.original_max_position_embeddings`
/// are both present and `factor > 1.0`, this returns
/// `max(max_position_embeddings, original_max_position_embeddings * factor)` —
/// the `max()` handles both an un-bumped config (mpe stays at the pre-YaRN
/// value, e.g. 32768) and an already-bumped one (mpe already reads 131072)
/// without double-counting the extension. A `rope_scaling` block that lacks
/// `original_max_position_embeddings`, or whose `factor` isn't a finite value
/// `> 1.0`, is treated as unusable and ignored — `max_position_embeddings` is
/// returned unchanged; under-reporting here only makes the gate conservative,
/// never admits beyond the model's real trained range.
///
/// `rope_parameters` fallback: newer HF configs (confirmed live 2026-08-23 on
/// `ministral-3-3b-int8-ov`, the project's internal engineering log)
/// carry the same `factor`/`original_max_position_embeddings`/`rope_type`
/// shape under `rope_parameters` instead of `rope_scaling`, with
/// `rope_scaling` present but explicitly `null`. Only consulted when
/// `rope_scaling` is absent or `null` — a config with a *usable*
/// `rope_scaling` block never falls through to `rope_parameters`, so the two
/// can never double-apply.
///
/// Returns `None` when `config.json` is absent, unparseable, or
/// `max_position_embeddings` is not present in any of the checked blocks —
/// fail-open, same "unknown ceiling means no gate, not a guessed one" policy
/// as [`crate::ov_embed::resolve_max_seq_len`]'s identical BERT-family read.
/// Unlike that helper, this one also handles `RoPE` extension, since text-gen
/// models (unlike the BERT-family embedding/reranking models) commonly ship
/// with one.
#[must_use]
pub(crate) fn read_native_context_limit(model_dir: &Path) -> Option<usize> {
    let raw = std::fs::read_to_string(model_dir.join("config.json")).ok()?;
    let cfg = serde_json::from_str::<serde_json::Value>(&raw).ok()?;

    let blocks = [
        Some(&cfg),
        cfg.get("text_config"),
        cfg.get("llm_config"),
        cfg.get("language_config"),
    ];

    let (mpe, rope_scaling, rope_parameters) = blocks.into_iter().flatten().find_map(|block| {
        let mpe = block
            .get("max_position_embeddings")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())?;
        Some((mpe, block.get("rope_scaling"), block.get("rope_parameters")))
    })?;

    let scaling = rope_scaling
        .filter(|v| !v.is_null())
        .or_else(|| rope_parameters.filter(|v| !v.is_null()));
    let Some(scaling) = scaling else {
        return Some(mpe);
    };
    let factor = scaling.get("factor").and_then(serde_json::Value::as_f64);
    let original = scaling
        .get("original_max_position_embeddings")
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| usize::try_from(v).ok());
    match (factor, original) {
        (Some(factor), Some(original)) if factor.is_finite() && factor > 1.0 => {
            // `original` is a context-length field (well under 2^52) — the
            // f64 round-trip through a small `factor` multiplier is exact
            // enough for this purpose (a token-count ceiling, not a byte
            // count needing bit-for-bit precision).
            #[allow(
                clippy::cast_sign_loss,
                clippy::cast_possible_truncation,
                clippy::cast_precision_loss
            )]
            let extended = (original as f64 * factor) as usize;
            Some(mpe.max(extended))
        }
        _ => Some(mpe),
    }
}

/// Filename of the learned KV-capacity ratchet sidecar, written alongside
/// `config.json` in a model's own directory.
const KV_RATCHET_FILENAME: &str = ".rustedvino_kv_ratchet.json";

#[derive(serde::Serialize, serde::Deserialize)]
struct KvCapacityRatchet {
    /// Prompt tokens actually observed live at the moment the wedge tripped
    /// (already discounted by a safety margin — see
    /// `ModelManager::spawn_kv_wedge_recovery`).
    observed_prompt_tokens: usize,
    /// The KV pool size (GB) that was active when `observed_prompt_tokens`
    /// was measured — needed to scale the ratchet if `cache_size_gb` changes
    /// later (see this struct's doc comment on [`read_kv_capacity_ratchet`]).
    kv_cache_gb_at_learn: f64,
}

/// Read a previously-learned KV-capacity ceiling for `model_dir`, if this
/// model has ever tripped the VLM-wedge detector (`dev/autotest/
/// 20260821_omnicoder9b_qwen35_hybrid_stall.md`) and had one written, scaled
/// to `current_kv_gb` — the KV pool size (GB) about to be used for the load
/// this ceiling will gate.
///
/// This exists because the static `compute_max_prompt_tokens` formula cannot
/// be made exactly right for every architecture against a closed `.so` — for
/// hybrid attention models in particular, it's calibrated at runtime instead:
/// `ModelManager::spawn_kv_wedge_recovery` writes this file the first
/// time a real wedge is observed. The stored value is **not** an absolute
/// token count — the characterization doc found real usable capacity scales
/// linearly with pool size (30.0% vs 30.8% of the formula ceiling across a
/// 7.5x pool-size difference), so a ratchet learned at a small
/// `cache_size_gb` must not permanently cap the model after an operator
/// raises the pool. Instead this scales the learned observation by
/// `current_kv_gb / kv_cache_gb_at_learn`. `compute_max_prompt_tokens`'s
/// caller takes `min(formula_result, this_value)` — the ratchet can only
/// ever tighten the gate relative to the formula, never loosen it beyond what
/// the formula itself would allow.
///
/// Returns `None` when no ratchet has been recorded yet, or the file is
/// unreadable/corrupt/degenerate (zero tokens or non-positive pool size —
/// treated the same as "none": the formula-only value still applies, this is
/// a refinement, not a required input, and a degenerate stored value must
/// never silently disable the gate by scaling to 0).
pub(crate) fn read_kv_capacity_ratchet(model_dir: &Path, current_kv_gb: f64) -> Option<usize> {
    let raw = std::fs::read_to_string(model_dir.join(KV_RATCHET_FILENAME)).ok()?;
    let entry = serde_json::from_str::<KvCapacityRatchet>(&raw).ok()?;
    if entry.observed_prompt_tokens == 0
        || entry.kv_cache_gb_at_learn <= 0.0
        || current_kv_gb <= 0.0
    {
        return None;
    }
    let scale = current_kv_gb / entry.kv_cache_gb_at_learn;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let scaled = (entry.observed_prompt_tokens as f64 * scale) as usize;
    (scaled > 0).then_some(scaled)
}

/// Persist a learned KV-capacity ceiling for `model_dir` — see
/// [`read_kv_capacity_ratchet`]. Writes via temp file + rename (same idiom as
/// `cache_manifest::commit`) so a crash mid-write can never leave a truncated
/// file that poisons the next read. Best-effort: logs and returns on any I/O
/// error rather than failing the caller (the in-memory ratchet, applied
/// immediately by the caller, still protects this process — only a future
/// restart would miss out on the learned value).
pub(crate) fn write_kv_capacity_ratchet(
    model_dir: &Path,
    observed_prompt_tokens: usize,
    kv_cache_gb_at_learn: f64,
) {
    let entry = KvCapacityRatchet {
        observed_prompt_tokens,
        kv_cache_gb_at_learn,
    };
    let Ok(serialized) = serde_json::to_string_pretty(&entry) else {
        tracing::warn!(
            model_dir = %model_dir.display(),
            "kv capacity ratchet: could not serialize, skipping write"
        );
        return;
    };
    let final_path = model_dir.join(KV_RATCHET_FILENAME);
    let tmp_path = model_dir.join(format!("{KV_RATCHET_FILENAME}.tmp"));
    if let Err(err) = std::fs::write(&tmp_path, &serialized) {
        tracing::warn!(
            model_dir = %model_dir.display(),
            error = %err,
            "kv capacity ratchet: could not write temp file, skipping"
        );
        return;
    }
    if let Err(err) = std::fs::rename(&tmp_path, &final_path) {
        tracing::warn!(
            model_dir = %model_dir.display(),
            error = %err,
            "kv capacity ratchet: could not commit write"
        );
    }
}

/// Read the top-level `model_type` string from `config.json` in `model_dir`.
///
/// Unlike [`read_vocab_size`]/[`read_moe_num_experts`], this checks only the
/// top level — `model_type` describes the wrapper architecture, and a nested
/// block's `model_type` (e.g. a VLM's `text_config.model_type`) describes a
/// different thing. Returns `None` when `config.json` is absent, unparseable,
/// or the field is missing/not a string.
pub(crate) fn read_model_type(model_dir: &Path) -> Option<String> {
    let raw = std::fs::read_to_string(model_dir.join("config.json")).ok()?;
    let cfg = serde_json::from_str::<serde_json::Value>(&raw).ok()?;
    cfg.get("model_type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}

/// Returns `true` when the chat template hints that this model emits
/// `<think>…</think>` reasoning blocks.
///
/// The marker checked is `enable_thinking` — present in Qwen3 and compatible
/// templates. Used by the model manager to auto-detect a reasoning parser at
/// load time when none is configured explicitly in the model policy.
pub(crate) fn template_hints_thinking(template: &str) -> bool {
    template.contains("enable_thinking")
}

/// Minimal generic chat template used only as a last resort when a model
/// directory ships **no** template at all.
///
/// Real `OpenVINO` models always ship `chat_template.jinja`, so in practice
/// this is hit only by tests pointing at a non-existent model dir. It renders
/// a plain `role: content` transcript so the engine still receives *something*
/// coherent rather than failing the load.
pub(crate) const FALLBACK_TEMPLATE: &str = concat!(
    "{%- for m in messages -%}",
    "{{ m.role }}: {{ m.content }}\n",
    "{%- endfor -%}",
    "{%- if add_generation_prompt %}assistant:{% endif -%}",
);

/// Load `eos_token` and `bos_token` from a model's `tokenizer_config.json`.
///
/// Returns `(eos_token, bos_token)` — both default to empty string when
/// absent, so the template receives a defined value rather than `undefined`.
/// Handles both the plain-string form (`"eos_token": "</s>"`) and the
/// object form (`"eos_token": {"content": "</s>", ...}`).
pub(crate) fn load_special_tokens(model_dir: &Path) -> (String, String) {
    let path = model_dir.join("tokenizer_config.json");
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return (String::new(), String::new());
    };
    let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return (String::new(), String::new());
    };

    let extract = |key: &str| -> String {
        match cfg.get(key) {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Object(o)) => o
                .get("content")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_owned(),
            _ => String::new(),
        }
    };

    (extract("eos_token"), extract("bos_token"))
}

/// A model's own shipped sampling defaults, read from its
/// `generation_config.json` at load time. `None` on any field means "the
/// caller keeps whatever it would otherwise use" — see
/// [`load_generation_defaults`]'s doc comment for the full gate.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GenerationDefaults {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
}

/// Load a model's own sampling defaults from `generation_config.json`.
///
/// Returns [`GenerationDefaults::default`] (all `None` — today's pre-existing
/// behavior, unchanged) when:
/// - the file is missing or unparseable;
/// - `do_sample` is absent or `false` — `transformers`' own `GenerationConfig`
///   class defaults `do_sample` to `false`, and a config that doesn't
///   explicitly ask for sampling gets no defaults here, even when stale
///   `temperature`/`top_p`/`top_k` fields are still present in the file (a
///   common leftover on configs that don't sample);
/// - a field is present but not the JSON type it should be (e.g. `null`, a
///   string, a negative `top_k`) — treated the same as an out-of-range value,
///   *not* the same as the field being absent;
/// - any field present fails a basic sanity range check — `temperature` must
///   be in `(0.0, 2.0]` (`0.0` contradicts an explicit `do_sample: true`: the
///   model's publisher asked for sampling but also for zero-temperature
///   greedy, a self-contradictory config) and `top_p` must be in `(0.0,
///   1.0]`. **The whole package is dropped** on any one field failing, not
///   just the offending field — a config that fails a sanity check shouldn't
///   be half-trusted for its other fields either. This applies uniformly to
///   a bad *type* (above) and a bad *value* (here): a field the key is simply
///   missing from the file is the only case treated as "not set" rather than
///   "malformed".
///
/// `top_k: 0` in the file is treated as "not set" (`None`), matching the
/// FFI's own `0 = unset` convention (see `ov_cb::GenParams::top_k`) — a
/// present, non-negative, whole-number `top_k` (including a float literal
/// like `20.0`) is otherwise accepted without a range check, since there is
/// no invalid `top_k` value shy of `0` or non-whole-number.
pub(crate) fn load_generation_defaults(model_dir: &Path) -> GenerationDefaults {
    let path = model_dir.join("generation_config.json");
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return GenerationDefaults::default();
    };
    let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return GenerationDefaults::default();
    };
    if cfg.get("do_sample").and_then(serde_json::Value::as_bool) != Some(true) {
        return GenerationDefaults::default();
    }

    // `Ok(None)` = key absent from the file (legitimate — this model doesn't
    // ship the field). `Err(())` = key present but not a JSON number
    // (`null`, a string, a bool, ...) — a malformed config, given the same
    // "drop the whole package" fate as an out-of-range value below.
    let read_f64 = |key: &str| -> Result<Option<f64>, ()> {
        match cfg.get(key) {
            None => Ok(None),
            Some(v) => v.as_f64().map(Some).ok_or(()),
        }
    };

    let Ok(raw_temperature) = read_f64("temperature") else {
        tracing::warn!(
            model_dir = %model_dir.display(),
            "generation_config.json: do_sample=true but temperature is present \
             and not a JSON number — ignoring this model's sampling defaults entirely"
        );
        return GenerationDefaults::default();
    };
    let Ok(raw_top_p) = read_f64("top_p") else {
        tracing::warn!(
            model_dir = %model_dir.display(),
            "generation_config.json: do_sample=true but top_p is present and \
             not a JSON number — ignoring this model's sampling defaults entirely"
        );
        return GenerationDefaults::default();
    };
    #[allow(clippy::cast_possible_truncation)]
    let temperature = raw_temperature.map(|t| t as f32);
    #[allow(clippy::cast_possible_truncation)]
    let top_p = raw_top_p.map(|p| p as f32);

    let top_k = match cfg.get("top_k") {
        None => None,
        Some(v) => match v.as_f64() {
            Some(f) if f >= 0.0 && f.fract() == 0.0 && f <= f64::from(u32::MAX) =>
            {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                Some(f as usize)
            }
            _ => {
                tracing::warn!(
                    model_dir = %model_dir.display(),
                    "generation_config.json: do_sample=true but top_k is present \
                     and not a non-negative whole number — ignoring this model's \
                     sampling defaults entirely"
                );
                return GenerationDefaults::default();
            }
        },
    }
    .filter(|&k| k > 0);

    if temperature.is_some_and(|t| !(t > 0.0 && t <= 2.0)) {
        tracing::warn!(
            model_dir = %model_dir.display(),
            ?temperature,
            "generation_config.json: do_sample=true but temperature is out of \
             sane range (0.0, 2.0] — ignoring this model's sampling defaults entirely"
        );
        return GenerationDefaults::default();
    }
    if top_p.is_some_and(|p| !(p > 0.0 && p <= 1.0)) {
        tracing::warn!(
            model_dir = %model_dir.display(),
            ?top_p,
            "generation_config.json: do_sample=true but top_p is out of sane \
             range (0.0, 1.0] — ignoring this model's sampling defaults entirely"
        );
        return GenerationDefaults::default();
    }

    GenerationDefaults {
        temperature,
        top_p,
        top_k,
    }
}

/// Load a model's Jinja chat template.
///
/// Resolution order:
/// 1. `chat_template.jinja` — the standalone file HF writes (preferred).
/// 2. The `chat_template` field inside `tokenizer_config.json` — either a
///    plain string, or a list of `{name, template}` objects (some models use
///    the multi-template form; the first `template` entry is taken).
///
/// # Errors
/// Returns an error when neither source exists or is readable/parseable.
/// [`load_chat_template`], but consulting an operator-supplied override first.
///
/// `override_path` is `ModelPolicy::chat_template`, already resolved to an
/// absolute path by the caller. When set it wins outright over anything in the
/// model directory — a converted model can ship a stale or incomplete template,
/// and the operator's declared substitute is the whole point of the field.
///
/// An override that is set but unreadable is a hard error rather than a silent
/// fall-back to the model's own template: the operator asked for a specific
/// template, and quietly serving a different one would produce exactly the kind
/// of wrong-but-plausible prompting this field exists to fix.
pub(crate) fn load_chat_template_with_override(
    model_dir: &Path,
    override_path: Option<&Path>,
) -> anyhow::Result<String> {
    if let Some(path) = override_path {
        return std::fs::read_to_string(path).map_err(|e| {
            anyhow::anyhow!(
                "reading chat_template override {}: {e} — the model's own template was NOT \
                 used as a fallback, because an explicitly configured override that cannot be \
                 read is a configuration error, not a hint",
                path.display()
            )
        });
    }
    let jinja = model_dir.join("chat_template.jinja");
    if jinja.is_file() {
        return std::fs::read_to_string(&jinja)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", jinja.display()));
    }

    let cfg_path = model_dir.join("tokenizer_config.json");
    if cfg_path.is_file() {
        let raw = std::fs::read_to_string(&cfg_path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", cfg_path.display()))?;
        let cfg: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("parsing {}: {e}", cfg_path.display()))?;

        if let Some(t) = cfg.get("chat_template").and_then(serde_json::Value::as_str) {
            return Ok(t.to_owned());
        }
        if let Some(t) = cfg
            .get("chat_template")
            .and_then(serde_json::Value::as_array)
            .and_then(|arr| {
                arr.iter()
                    .find_map(|o| o.get("template").and_then(serde_json::Value::as_str))
            })
        {
            return Ok(t.to_owned());
        }
    }

    Err(anyhow::anyhow!(
        "no chat template found in {}",
        model_dir.display()
    ))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    // ---- template_hints_thinking ----------------------------------------

    #[test]
    fn hints_thinking_true_for_enable_thinking_marker() {
        assert!(template_hints_thinking("{% if enable_thinking %}...",));
    }

    #[test]
    fn hints_thinking_false_for_plain_template() {
        assert!(!template_hints_thinking("{{ role }}: {{ content }}"));
    }

    #[test]
    fn hints_thinking_false_for_mistral_template() {
        // Mistral templates use strftime_now but no enable_thinking.
        assert!(!template_hints_thinking(
            "{{ strftime_now('%Y-%m-%d') }} {{ content }}"
        ));
    }

    /// An operator override wins outright over the model's own template —
    /// the whole point of `ModelPolicy.chat_template`.
    #[test]
    fn chat_template_override_wins_over_the_model_directory() {
        let dir = tempfile::tempdir().expect("tempdir").keep();
        std::fs::write(dir.join("chat_template.jinja"), "FROM MODEL DIR").expect("write");
        let over = dir.join("override.jinja");
        std::fs::write(&over, "FROM OVERRIDE").expect("write");
        let t = load_chat_template_with_override(&dir, Some(&over)).expect("must load");
        assert_eq!(t, "FROM OVERRIDE");
    }

    /// An override that is configured but unreadable is a hard error, NOT a
    /// silent fall-back to the model's own template: the operator asked for a
    /// specific template, and quietly serving a different one would reproduce
    /// exactly the wrong-but-plausible prompting the field exists to fix.
    #[test]
    fn unreadable_chat_template_override_is_an_error_not_a_fallback() {
        let dir = tempfile::tempdir().expect("tempdir").keep();
        std::fs::write(dir.join("chat_template.jinja"), "FROM MODEL DIR").expect("write");
        let missing = dir.join("does-not-exist.jinja");
        let err = load_chat_template_with_override(&dir, Some(&missing))
            .expect_err("a missing override must fail loudly");
        assert!(
            err.to_string().contains("chat_template override"),
            "error must name the override, got: {err}"
        );
    }

    /// With no override configured, behaviour is exactly as before.
    #[test]
    fn no_override_falls_through_to_the_model_directory() {
        let dir = tempfile::tempdir().expect("tempdir").keep();
        std::fs::write(dir.join("chat_template.jinja"), "FROM MODEL DIR").expect("write");
        let t = load_chat_template_with_override(&dir, None).expect("must load");
        assert_eq!(t, "FROM MODEL DIR");
    }

    #[test]
    fn loads_chat_template_jinja_file() {
        let dir = std::env::temp_dir().join(format!("rv-tmpl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("chat_template.jinja"), "HELLO {{ x }}").unwrap();
        let t = load_chat_template_with_override(&dir, None).unwrap();
        assert_eq!(t, "HELLO {{ x }}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn falls_back_to_tokenizer_config_field() {
        let dir = std::env::temp_dir().join(format!("rv-tmpl-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template": "FROM CONFIG {{ x }}"}"#,
        )
        .unwrap();
        let t = load_chat_template_with_override(&dir, None).unwrap();
        assert_eq!(t, "FROM CONFIG {{ x }}");
        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- load_generation_defaults ----------------------------------------

    fn gendefaults_dir(suffix: &str) -> std::path::PathBuf {
        let d =
            std::env::temp_dir().join(format!("rv-gendefaults-{}-{suffix}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn generation_defaults_missing_file_is_all_none() {
        let dir = gendefaults_dir("missing");
        assert_eq!(
            load_generation_defaults(&dir),
            GenerationDefaults::default()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn generation_defaults_unparseable_json_is_all_none() {
        let dir = gendefaults_dir("unparseable");
        std::fs::write(dir.join("generation_config.json"), "{ not json").unwrap();
        assert_eq!(
            load_generation_defaults(&dir),
            GenerationDefaults::default()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn generation_defaults_do_sample_absent_is_all_none() {
        let dir = gendefaults_dir("absent");
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"temperature": 0.7, "top_p": 0.8, "top_k": 20}"#,
        )
        .unwrap();
        assert_eq!(
            load_generation_defaults(&dir),
            GenerationDefaults::default()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn generation_defaults_do_sample_false_with_stale_fields_is_all_none() {
        let dir = gendefaults_dir("false");
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"do_sample": false, "temperature": 0.7, "top_p": 0.8, "top_k": 20}"#,
        )
        .unwrap();
        assert_eq!(
            load_generation_defaults(&dir),
            GenerationDefaults::default()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The real `qwen3-vl-8b-int8-ov` shape that motivated this feature.
    #[test]
    fn generation_defaults_do_sample_true_full_package_is_parsed() {
        let dir = gendefaults_dir("full");
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"do_sample": true, "temperature": 0.7, "top_p": 0.8, "top_k": 20}"#,
        )
        .unwrap();
        let d = load_generation_defaults(&dir);
        assert_eq!(d.temperature, Some(0.7));
        assert_eq!(d.top_p, Some(0.8));
        assert_eq!(d.top_k, Some(20));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn generation_defaults_do_sample_true_no_numeric_fields_is_all_none() {
        let dir = gendefaults_dir("no-numerics");
        std::fs::write(dir.join("generation_config.json"), r#"{"do_sample": true}"#).unwrap();
        assert_eq!(
            load_generation_defaults(&dir),
            GenerationDefaults::default()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn generation_defaults_top_k_zero_is_treated_as_unset() {
        let dir = gendefaults_dir("top-k-zero");
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"do_sample": true, "temperature": 0.7, "top_k": 0}"#,
        )
        .unwrap();
        let d = load_generation_defaults(&dir);
        assert_eq!(d.temperature, Some(0.7));
        assert_eq!(d.top_k, None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn generation_defaults_temperature_zero_contradicts_do_sample_drops_whole_package() {
        let dir = gendefaults_dir("temp-zero");
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"do_sample": true, "temperature": 0.0, "top_p": 0.8}"#,
        )
        .unwrap();
        // Whole package dropped, not just temperature — top_p must also be None.
        assert_eq!(
            load_generation_defaults(&dir),
            GenerationDefaults::default()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn generation_defaults_out_of_range_top_p_drops_whole_package() {
        let dir = gendefaults_dir("top-p-oor");
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"do_sample": true, "temperature": 0.7, "top_p": 1.5}"#,
        )
        .unwrap();
        // Whole package dropped — temperature must also be None despite being in range.
        assert_eq!(
            load_generation_defaults(&dir),
            GenerationDefaults::default()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A malformed (non-numeric) field must be treated the same as an
    /// out-of-range value — dropping the whole package — not the same as the
    /// field being absent. Regression test for the gate-bypass finding: the
    /// original `.and_then(as_f64)` silently mapped `null` to `None` and let
    /// `top_p`/`top_k` sail through unchecked, reproducing the exact
    /// top_p-vs-temperature gating collision this feature exists to prevent.
    #[test]
    fn generation_defaults_null_temperature_drops_whole_package() {
        let dir = gendefaults_dir("temp-null");
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"do_sample": true, "temperature": null, "top_p": 0.8, "top_k": 20}"#,
        )
        .unwrap();
        // Whole package dropped — top_p/top_k must also be None despite being
        // individually valid.
        assert_eq!(
            load_generation_defaults(&dir),
            GenerationDefaults::default()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn generation_defaults_string_typed_top_p_drops_whole_package() {
        let dir = gendefaults_dir("top-p-string");
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"do_sample": true, "temperature": 0.7, "top_p": "0.8"}"#,
        )
        .unwrap();
        assert_eq!(
            load_generation_defaults(&dir),
            GenerationDefaults::default()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `top_k` as a float literal representing a whole number (a real shape
    /// some conversion scripts emit) must parse the same as an integer
    /// literal — `as_u64()` alone rejects any JSON number written with a
    /// decimal point, whole-valued or not.
    #[test]
    fn generation_defaults_top_k_float_literal_whole_number_is_parsed() {
        let dir = gendefaults_dir("top-k-float");
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"do_sample": true, "temperature": 0.7, "top_k": 20.0}"#,
        )
        .unwrap();
        let d = load_generation_defaults(&dir);
        assert_eq!(d.top_k, Some(20));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn generation_defaults_negative_top_k_drops_whole_package() {
        let dir = gendefaults_dir("top-k-negative");
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"do_sample": true, "temperature": 0.7, "top_k": -5}"#,
        )
        .unwrap();
        assert_eq!(
            load_generation_defaults(&dir),
            GenerationDefaults::default()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn generation_defaults_fractional_top_k_drops_whole_package() {
        let dir = gendefaults_dir("top-k-fractional");
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"do_sample": true, "temperature": 0.7, "top_k": 20.5}"#,
        )
        .unwrap();
        assert_eq!(
            load_generation_defaults(&dir),
            GenerationDefaults::default()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- read_kv_bytes_per_token_f16 ------------------------------------

    fn tmp_dir(suffix: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("rv-kv-{}-{}", std::process::id(), suffix));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Standard GQA model (`Qwen3-8B` style): derived `head_dim`, `kv_heads` < `num_heads`.
    /// `layers=28`, `kv_heads=8`, `hidden=4096`, `heads=32` → `head_dim=128`
    /// Expected: `28 × 8 × 128 × 2 × 2 = 114_688`
    #[test]
    fn kv_bytes_gqa_derived_head_dim() {
        let dir = tmp_dir("gqa");
        std::fs::write(
            dir.join("config.json"),
            r#"{"num_hidden_layers":28,"num_attention_heads":32,"num_key_value_heads":8,"hidden_size":4096}"#,
        )
        .unwrap();
        assert_eq!(read_kv_bytes_per_token_f16(&dir), 28 * 8 * 128 * 4);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Explicit `head_dim` field overrides the derived value.
    #[test]
    fn kv_bytes_explicit_head_dim() {
        let dir = tmp_dir("explicit-hd");
        std::fs::write(
            dir.join("config.json"),
            r#"{"num_hidden_layers":40,"num_attention_heads":32,"num_key_value_heads":8,"hidden_size":4096,"head_dim":64}"#,
        )
        .unwrap();
        // head_dim=64 (explicit), not 128 (derived)
        assert_eq!(read_kv_bytes_per_token_f16(&dir), 40 * 8 * 64 * 4);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// No GQA: `num_key_value_heads` absent → falls back to `num_attention_heads`.
    #[test]
    fn kv_bytes_no_gqa_fallback() {
        let dir = tmp_dir("no-gqa");
        std::fs::write(
            dir.join("config.json"),
            r#"{"num_hidden_layers":32,"num_attention_heads":16,"hidden_size":2048}"#,
        )
        .unwrap();
        // head_dim = 2048/16 = 128; kv_heads = 16 (= num_heads)
        assert_eq!(read_kv_bytes_per_token_f16(&dir), 32 * 16 * 128 * 4);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Newer VLM config (`Qwen3-VL` style): LLM params live ONLY under
    /// `text_config`, nothing at the top level. The per-key fallback resolves
    /// them. `layers=36`, `kv_heads=8`, `head_dim=128` → `36 × 8 × 128 × 4`.
    #[test]
    fn kv_bytes_nested_text_config() {
        let dir = tmp_dir("nested-text-config");
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen3_vl","vision_config":{"depth":27},"text_config":{"num_hidden_layers":36,"num_attention_heads":32,"num_key_value_heads":8,"hidden_size":4096,"head_dim":128}}"#,
        )
        .unwrap();
        assert_eq!(read_kv_bytes_per_token_f16(&dir), 36 * 8 * 128 * 4);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `llm_config` (InternVL-style) is also searched.
    #[test]
    fn kv_bytes_nested_llm_config() {
        let dir = tmp_dir("nested-llm-config");
        std::fs::write(
            dir.join("config.json"),
            r#"{"llm_config":{"num_hidden_layers":32,"num_attention_heads":16,"hidden_size":2048}}"#,
        )
        .unwrap();
        // head_dim = 2048/16 = 128; kv_heads = 16 (= num_heads)
        assert_eq!(read_kv_bytes_per_token_f16(&dir), 32 * 16 * 128 * 4);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Top-level params win over a nested block when both exist (Qwen2.5-VL
    /// duplicates them at the top level; that copy must still resolve).
    #[test]
    fn kv_bytes_top_level_takes_precedence() {
        let dir = tmp_dir("top-level-wins");
        std::fs::write(
            dir.join("config.json"),
            r#"{"num_hidden_layers":28,"num_attention_heads":28,"num_key_value_heads":4,"hidden_size":3584,"text_config":{"num_hidden_layers":99,"num_attention_heads":28,"num_key_value_heads":4,"hidden_size":3584}}"#,
        )
        .unwrap();
        // Uses top-level layers=28, not text_config's 99. head_dim = 3584/28 = 128.
        assert_eq!(read_kv_bytes_per_token_f16(&dir), 28 * 4 * 128 * 4);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Missing required field returns 0 (gate disabled).
    #[test]
    fn kv_bytes_missing_field_returns_zero() {
        let dir = tmp_dir("missing");
        std::fs::write(
            dir.join("config.json"),
            r#"{"num_attention_heads":32,"hidden_size":4096}"#, // num_hidden_layers absent
        )
        .unwrap();
        assert_eq!(read_kv_bytes_per_token_f16(&dir), 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Missing `config.json` returns 0 (gate disabled, not an error).
    #[test]
    fn kv_bytes_missing_file_returns_zero() {
        let dir = tmp_dir("no-file");
        // Don't create config.json
        assert_eq!(read_kv_bytes_per_token_f16(&dir), 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- KV-capacity ratchet ---------------------------------------------

    /// No sidecar written yet → `None`, not a panic or a spurious 0.
    #[test]
    fn ratchet_absent_returns_none() {
        let dir = tmp_dir("ratchet-absent");
        assert_eq!(read_kv_capacity_ratchet(&dir, 7.5), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Round-trip at the SAME pool size the ratchet was learned at: the
    /// stored value comes back unchanged (scale factor of 1.0).
    #[test]
    fn ratchet_round_trip_same_pool_size() {
        let dir = tmp_dir("ratchet-round-trip");
        write_kv_capacity_ratchet(&dir, 4665, 1.0);
        assert_eq!(read_kv_capacity_ratchet(&dir, 1.0), Some(4665));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The core fix this test locks in: a ratchet learned at a SMALL
    /// `cache_size_gb` must scale UP proportionally when the operator later
    /// raises the pool size, not stay pinned at the small absolute value
    /// forever (the project's internal engineering log's
    /// 30.0%-vs-30.8%-of-ceiling finding: real capacity is proportional to
    /// pool size, not a fixed token count).
    #[test]
    fn ratchet_scales_with_pool_size_increase() {
        let dir = tmp_dir("ratchet-scale-up");
        write_kv_capacity_ratchet(&dir, 4665, 1.0);
        // Pool raised 7.5x → learned ceiling should scale 7.5x too.
        assert_eq!(read_kv_capacity_ratchet(&dir, 7.5), Some(34_987));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Symmetric case: pool shrunk after the ratchet was learned at a larger
    /// size — the ceiling scales down, not just up.
    #[test]
    fn ratchet_scales_with_pool_size_decrease() {
        let dir = tmp_dir("ratchet-scale-down");
        write_kv_capacity_ratchet(&dir, 34_987, 7.5);
        assert_eq!(read_kv_capacity_ratchet(&dir, 1.0), Some(4_664));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A corrupt/degenerate sidecar (zero tokens) must never scale down to a
    /// value that disables the L0 gate — `None`, so the formula-only result
    /// applies instead of silently opening the gate wide.
    #[test]
    fn ratchet_zero_tokens_is_treated_as_absent() {
        let dir = tmp_dir("ratchet-zero-tokens");
        write_kv_capacity_ratchet(&dir, 0, 1.0);
        assert_eq!(read_kv_capacity_ratchet(&dir, 7.5), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A degenerate stored pool size (0.0, e.g. from a media-model
    /// misconfiguration) must not divide-by-zero or scale to `usize::MAX` —
    /// treated as absent.
    #[test]
    fn ratchet_degenerate_learn_pool_size_is_treated_as_absent() {
        let dir = tmp_dir("ratchet-degenerate-pool");
        write_kv_capacity_ratchet(&dir, 4665, 0.0);
        assert_eq!(read_kv_capacity_ratchet(&dir, 7.5), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn errors_when_no_template_present() {
        let dir = std::env::temp_dir().join(format!("rv-tmpl-none-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(load_chat_template_with_override(&dir, None).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- read_moe_num_experts -------------------------------------------

    /// Dense model (no `num_experts` field) → `None`.
    #[test]
    fn moe_dense_returns_none() {
        let dir = tmp_dir("moe-dense");
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen3","num_hidden_layers":36,"num_attention_heads":16,"hidden_size":2048}"#,
        )
        .unwrap();
        assert_eq!(read_moe_num_experts(&dir), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `MoE` model with top-level `num_experts` → `Some(n)`.
    #[test]
    fn moe_top_level_num_experts() {
        let dir = tmp_dir("moe-top");
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen3_moe","num_experts":128,"num_experts_per_tok":8,"num_hidden_layers":94}"#,
        )
        .unwrap();
        assert_eq!(read_moe_num_experts(&dir), Some(128));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `MoE` config nested under `text_config` (some VLM wrappers) → `Some(n)`.
    #[test]
    fn moe_nested_text_config() {
        let dir = tmp_dir("moe-nested");
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"vlm_wrapper","text_config":{"num_experts":64,"num_experts_per_tok":4}}"#,
        )
        .unwrap();
        assert_eq!(read_moe_num_experts(&dir), Some(64));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `num_experts: 1` is a dense model masquerading as `MoE` config → `None`.
    #[test]
    fn moe_num_experts_one_is_dense() {
        let dir = tmp_dir("moe-one");
        std::fs::write(dir.join("config.json"), r#"{"num_experts":1}"#).unwrap();
        assert_eq!(read_moe_num_experts(&dir), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Missing `config.json` → `None` (not an error).
    #[test]
    fn moe_missing_config_returns_none() {
        let dir = tmp_dir("moe-no-file");
        assert_eq!(read_moe_num_experts(&dir), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- read_vocab_size -------------------------------------------------

    /// Top-level `vocab_size` → `Some(n)`.
    #[test]
    fn vocab_size_top_level() {
        let dir = tmp_dir("vocab-top");
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen3","vocab_size":151936}"#,
        )
        .unwrap();
        assert_eq!(read_vocab_size(&dir), Some(151_936));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `vocab_size` nested under `text_config` (VLM-style wrapper) → `Some(n)`.
    #[test]
    fn vocab_size_nested_text_config() {
        let dir = tmp_dir("vocab-nested");
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"vlm_wrapper","text_config":{"vocab_size":151936}}"#,
        )
        .unwrap();
        assert_eq!(read_vocab_size(&dir), Some(151_936));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Missing `vocab_size` field → `None`.
    #[test]
    fn vocab_size_missing_field_returns_none() {
        let dir = tmp_dir("vocab-missing");
        std::fs::write(dir.join("config.json"), r#"{"model_type":"qwen3"}"#).unwrap();
        assert_eq!(read_vocab_size(&dir), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Missing `config.json` → `None` (not an error).
    #[test]
    fn vocab_size_missing_config_returns_none() {
        let dir = tmp_dir("vocab-no-file");
        assert_eq!(read_vocab_size(&dir), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- read_native_context_limit ------------------------------------------

    /// Plain `max_position_embeddings`, no `rope_scaling` — the real
    /// `qwen3-4b-int4-ov` shape that started this investigation.
    #[test]
    fn native_context_limit_plain_max_position_embeddings() {
        let dir = tmp_dir("native-ctx-plain");
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen3","max_position_embeddings":40960,"rope_scaling":null}"#,
        )
        .unwrap();
        assert_eq!(read_native_context_limit(&dir), Some(40_960));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `rope_scaling` present with `factor`/`original_max_position_embeddings`,
    /// `max_position_embeddings` still at its pre-extension value — the
    /// extension must be applied.
    #[test]
    fn native_context_limit_applies_yarn_factor() {
        let dir = tmp_dir("native-ctx-yarn-unbumped");
        std::fs::write(
            dir.join("config.json"),
            r#"{"max_position_embeddings":32768,
                "rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}"#,
        )
        .unwrap();
        assert_eq!(read_native_context_limit(&dir), Some(131_072));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `rope_scaling` present but `max_position_embeddings` already reflects
    /// the extended value — must not double-count it via `factor`.
    #[test]
    fn native_context_limit_does_not_double_count_prebumped_mpe() {
        let dir = tmp_dir("native-ctx-yarn-bumped");
        std::fs::write(
            dir.join("config.json"),
            r#"{"max_position_embeddings":131072,
                "rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}"#,
        )
        .unwrap();
        assert_eq!(read_native_context_limit(&dir), Some(131_072));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `rope_scaling` present but missing `original_max_position_embeddings`
    /// (old-style linear-scaling config) — unusable, `max_position_embeddings`
    /// returned unchanged rather than guessed at.
    #[test]
    fn native_context_limit_ignores_scaling_without_original() {
        let dir = tmp_dir("native-ctx-scaling-incomplete");
        std::fs::write(
            dir.join("config.json"),
            r#"{"max_position_embeddings":4096,"rope_scaling":{"type":"linear","factor":2.0}}"#,
        )
        .unwrap();
        assert_eq!(read_native_context_limit(&dir), Some(4096));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `rope_parameters` (newer HF key) with `rope_scaling` explicitly `null`
    /// — the real `ministral-3-3b-int8-ov` shape
    /// (the project's internal engineering log).
    /// `max_position_embeddings` is already the post-extension value here, so
    /// this only confirms the fallback lookup itself doesn't error or return
    /// `None` — `native_context_limit_applies_yarn_factor_from_rope_parameters`
    /// below confirms the factor math applies through this path too.
    #[test]
    fn native_context_limit_reads_rope_parameters_when_rope_scaling_null() {
        let dir = tmp_dir("native-ctx-rope-parameters-prebumped");
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"ministral3","max_position_embeddings":262144,
                "rope_scaling":null,
                "rope_parameters":{"rope_type":"yarn","factor":16.0,
                "original_max_position_embeddings":16384}}"#,
        )
        .unwrap();
        assert_eq!(read_native_context_limit(&dir), Some(262_144));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Same `rope_parameters` fallback, but `max_position_embeddings` is left
    /// at its pre-extension value — the factor must still apply through the
    /// `rope_parameters` path, same as it does for `rope_scaling`.
    #[test]
    fn native_context_limit_applies_yarn_factor_from_rope_parameters() {
        let dir = tmp_dir("native-ctx-rope-parameters-unbumped");
        std::fs::write(
            dir.join("config.json"),
            r#"{"max_position_embeddings":16384,
                "rope_parameters":{"rope_type":"yarn","factor":16.0,
                "original_max_position_embeddings":16384}}"#,
        )
        .unwrap();
        assert_eq!(read_native_context_limit(&dir), Some(262_144));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A *usable* `rope_scaling` block must win over `rope_parameters` —
    /// never consult the fallback, never double-apply both.
    #[test]
    fn native_context_limit_prefers_rope_scaling_over_rope_parameters() {
        let dir = tmp_dir("native-ctx-rope-scaling-wins");
        std::fs::write(
            dir.join("config.json"),
            r#"{"max_position_embeddings":32768,
                "rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768},
                "rope_parameters":{"rope_type":"yarn","factor":16.0,"original_max_position_embeddings":32768}}"#,
        )
        .unwrap();
        assert_eq!(read_native_context_limit(&dir), Some(131_072));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `max_position_embeddings` nested under `text_config` (VLM-style wrapper).
    #[test]
    fn native_context_limit_falls_back_to_text_config() {
        let dir = tmp_dir("native-ctx-nested");
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"vlm_wrapper","text_config":{"max_position_embeddings":32768}}"#,
        )
        .unwrap();
        assert_eq!(read_native_context_limit(&dir), Some(32_768));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Missing `max_position_embeddings` field → `None` (fail-open, not a
    /// guessed default).
    #[test]
    fn native_context_limit_missing_field_returns_none() {
        let dir = tmp_dir("native-ctx-missing-field");
        std::fs::write(dir.join("config.json"), r#"{"model_type":"qwen3"}"#).unwrap();
        assert_eq!(read_native_context_limit(&dir), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Missing `config.json` → `None` (not an error).
    #[test]
    fn native_context_limit_missing_config_returns_none() {
        let dir = tmp_dir("native-ctx-no-file");
        assert_eq!(read_native_context_limit(&dir), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- read_model_type ---------------------------------------------------

    /// Top-level `model_type` → `Some(String)`.
    #[test]
    fn model_type_top_level() {
        let dir = tmp_dir("model-type-top");
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen3","vocab_size":151936}"#,
        )
        .unwrap();
        assert_eq!(read_model_type(&dir), Some("qwen3".to_owned()));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A nested `text_config.model_type` is NOT read — only the top level
    /// counts (it describes a different thing than the wrapper's own type).
    #[test]
    fn model_type_ignores_nested_block() {
        let dir = tmp_dir("model-type-nested-ignored");
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"vlm_wrapper","text_config":{"model_type":"qwen3"}}"#,
        )
        .unwrap();
        assert_eq!(read_model_type(&dir), Some("vlm_wrapper".to_owned()));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Missing `config.json` → `None` (not an error).
    #[test]
    fn model_type_missing_config_returns_none() {
        let dir = tmp_dir("model-type-no-file");
        assert_eq!(read_model_type(&dir), None);
        std::fs::remove_dir_all(&dir).ok();
    }
}
