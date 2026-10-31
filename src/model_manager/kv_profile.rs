// ============================================================
// src/model_manager/kv_profile.rs — KV-cache layout profile of a model
// ============================================================
// Describes what a model's KV cache costs on the GPU plugin (2026.4.1 layouts):
// the per-layer attention dimensions, the linear-attention (LA) state a hybrid
// model keeps per sequence, whether the IR carries 4-bit weights (which decides
// the plugin's default KV precision), and an IR-level precision hint.
//
// Everything here is a pure function of the model directory: `config.json` for
// the layer list and `openvino_language_model.xml` / `openvino_model.xml` for the
// IR facts. Nothing is cached on disk. The capacity arithmetic that consumes the
// profile lives in `kv_capacity.rs`.
// ============================================================

use std::collections::{HashMap, HashSet};
use std::path::Path;

/// KV-cache element layout that the engine can be asked for.
///
/// `f32` is deliberately absent: it is no longer an accepted configuration value.
/// `u8` is not a layout either; it is an alias of [`KvPrecision::I8`] (the plugin
/// rewrites an explicit `u8` to `i8`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum KvPrecision {
    /// 4-bit keys and values (the plugin default when any `MatMul` weight is 4-bit).
    U4,
    /// 8-bit keys and values (the plugin default otherwise).
    I8,
    /// Plain f16 (only through an IR hint or an explicit request).
    F16,
}

impl KvPrecision {
    /// Parse the value of an IR `runtime_options/KV_CACHE_PRECISION` hint.
    /// Anything that is not one of the three layouts (including `f32`) is ignored.
    fn from_ir_hint(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "u4" | "i4" => Some(Self::U4),
            "i8" | "u8" => Some(Self::I8),
            "f16" => Some(Self::F16),
            _ => None,
        }
    }
}

/// How far the profile can be trusted. Ordered: `Low < Medium < High`.
/// `Low` sends the gate back to the legacy formula.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Confidence {
    /// Something does not add up (SDPA count, unknown layer type, `mtp_mode`, f32 inference).
    Low,
    /// Derived from `config.json` only: no IR, or LA state computed rather than read.
    Medium,
    /// Config and IR agree.
    High,
}

impl Confidence {
    /// Stable lower-case name for the admin API.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// One paged-attention layer: its KV heads and the key/value head dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PaLayer {
    /// Number of KV heads (`h` in the layout table).
    pub kv_heads: usize,
    /// Key head dimension.
    pub k_head_dim: usize,
    /// Value head dimension.
    pub v_head_dim: usize,
}

/// Tokens per GPU KV block (the layout table below is defined for this size).
pub(crate) const GPU_BLOCK_TOKENS: usize = 16;
/// Tokens per CPU KV block (informational: CPU stays on the legacy gate).
const CPU_BLOCK_TOKENS: usize = 32;

/// Round `n` up to a multiple of 16 (the plugin aligns the by-token value row).
fn align16(n: u64) -> u64 {
    n.div_ceil(16) * 16
}

impl PaLayer {
    /// Bytes this layer stores for one 16-token block at precision `p`.
    ///
    /// GPU plugin 2026.4.1 layout (verified against `padump` and the measured
    /// pools, plan 06 §1.1): keys are stored by channel, values by token.
    /// With `h` heads, key dim `dk`, value dim `dv`:
    /// - u4: key `0.75·h·dk` B/token, value `h·(align16(dv/2)+4)` B/token
    /// - i8: key `1.25·h·dk`, value `h·(dv+4)`
    /// - f16: `2·h·dk` and `2·h·dv`
    ///
    /// The key part is computed per 16-token block so it stays an integer.
    fn block16_bytes(&self, p: KvPrecision) -> u64 {
        let h = self.kv_heads as u64;
        let dk = self.k_head_dim as u64;
        let dv = self.v_head_dim as u64;
        let (key_per_block, value_per_token) = match p {
            KvPrecision::U4 => (12 * h * dk, h * (align16(dv / 2) + 4)),
            KvPrecision::I8 => (20 * h * dk, h * (dv + 4)),
            KvPrecision::F16 => (32 * h * dk, 2 * h * dv),
        };
        key_per_block + 16 * value_per_token
    }
}

/// KV-cache profile of one model.
#[derive(Clone, Debug)]
pub(crate) struct KvProfile {
    /// One entry per paged-attention layer (`full_attention` and
    /// `sliding_attention`; sliding layers are charged in full, nothing is evicted).
    pub pa_layers: Vec<PaLayer>,
    /// Tokens per KV block: 16 on GPU, 32 on CPU.
    pub block_tokens: usize,
    /// `Some(true)` when a `MatMul` weight chain ends in an `i4`/`u4` constant (the
    /// plugin then defaults to u4); `None` when the IR could not be read.
    pub weights_4bit: Option<bool>,
    /// `rt_info/runtime_options/KV_CACHE_PRECISION` of the IR, if present.
    pub ir_kv_hint: Option<KvPrecision>,
    /// Linear-attention / convolution state per sequence in bytes (f16), 0 for dense models.
    pub linear_state_bytes_per_seq: u64,
    /// Hidden size of the language model (headroom cap input).
    pub hidden_size: usize,
    /// How far the numbers can be trusted.
    pub confidence: Confidence,
}

impl KvProfile {
    /// Read the profile of the model in `model_dir` for `device`.
    ///
    /// Returns `None` when `config.json` is missing or lacks the attention
    /// geometry. Blocking file I/O plus an XML parse (tens of ms): call from
    /// `spawn_blocking` on a request path.
    pub(crate) fn read(model_dir: &Path, device: &str) -> Option<Self> {
        Self::read_with(model_dir, device, false)
    }

    /// Like [`KvProfile::read`]; `mtp_configured` marks a model whose engine is
    /// configured with `mtp_mode` (the extra draft head changes the KV layout the
    /// formulas assume), which forces `Confidence::Low`.
    pub(crate) fn read_with(model_dir: &Path, device: &str, mtp_configured: bool) -> Option<Self> {
        let raw = std::fs::read_to_string(model_dir.join("config.json")).ok()?;
        let cfg = serde_json::from_str::<serde_json::Value>(&raw).ok()?;
        let ir = ["openvino_language_model.xml", "openvino_model.xml"]
            .iter()
            .map(|n| model_dir.join(n))
            .find(|p| p.is_file())
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|xml| scan_ir(&xml));
        Self::from_parts(&cfg, ir.as_ref(), device, mtp_configured)
    }

    /// Assemble a profile from a parsed `config.json` and an optional IR scan.
    fn from_parts(
        cfg: &serde_json::Value,
        ir: Option<&IrScan>,
        device: &str,
        mtp_configured: bool,
    ) -> Option<Self> {
        let layers = config_layers(cfg)?;
        let hidden_size = cfg_usize(cfg, "hidden_size")?;
        let mut confidence = Confidence::High;

        if layers.unknown_layer_type || mtp_configured {
            confidence = Confidence::Low;
        }
        // The IR holds one SDPA op per paged-attention layer. A mismatch means the
        // config's layer list does not describe this graph (gemma-4-e4b shares KV
        // across layers and has no SDPA ops): do not guess, fall back to legacy.
        if layers.shared_kv_layers {
            // KV shared across layers (gemma-4-e4b): the layer list over-counts.
            confidence = Confidence::Low;
        }
        if let Some(ir) = ir {
            if ir.sdpa_count != layers.pa.len() || ir.f32_inference || ir.ir_hint_unparsed {
                confidence = Confidence::Low;
            }
            // The KV variables carry each layer's real (heads, dim); they must be exactly
            // the multiset the config implies (keys and values), else the formulas would
            // price the wrong geometry (MLA, a separate v_head_dim, a wrong hidden/heads guess).
            if !ir.kv_variables.is_empty() && !kv_variables_match(&layers.pa, &ir.kv_variables) {
                confidence = Confidence::Low;
            }
        } else {
            confidence = confidence.min(Confidence::Medium);
        }

        let ir_state = ir.map(|i| i.state_bytes).filter(|b| *b > 0);
        let linear_state_bytes_per_seq = match ir_state {
            Some(b) => {
                // Both sources exist: the IR is authoritative but must agree with the config.
                if b != layers.config_state_bytes {
                    confidence = Confidence::Low;
                }
                b
            }
            None if layers.config_state_bytes > 0 => {
                // LA layers exist but the IR did not show their state variables:
                // computed from the config instead.
                confidence = confidence.min(Confidence::Medium);
                layers.config_state_bytes
            }
            None => 0,
        };

        Some(Self {
            pa_layers: layers.pa,
            block_tokens: if device.to_ascii_uppercase().starts_with("CPU") {
                CPU_BLOCK_TOKENS
            } else {
                GPU_BLOCK_TOKENS
            },
            weights_4bit: ir.map(|i| i.weights_4bit),
            ir_kv_hint: ir.and_then(|i| i.ir_kv_hint),
            linear_state_bytes_per_seq,
            hidden_size,
            confidence,
        })
    }

    /// Bytes of one KV block at precision `p`, summed over all paged-attention layers.
    pub(crate) fn kv_block_bytes(&self, p: KvPrecision) -> u64 {
        let block16: u64 = self.pa_layers.iter().map(|l| l.block16_bytes(p)).sum();
        block16 * self.block_tokens as u64 / GPU_BLOCK_TOKENS as u64
    }

    /// KV bytes one token costs at precision `p` (sum of the layout table over the
    /// paged-attention layers; excludes the per-sequence LA state).
    pub(crate) fn kv_bytes_per_token(&self, p: KvPrecision) -> u64 {
        self.kv_block_bytes(p) / self.block_tokens as u64
    }
}

// ---------------------------------------------------------------- config.json

/// The `(heads, dim)` multiset of the IR's KV variables equals the config's, each layer
/// contributing its key and its value.
fn kv_variables_match(pa: &[PaLayer], ir: &[(usize, usize)]) -> bool {
    let mut want: Vec<(usize, usize)> = pa
        .iter()
        .flat_map(|l| [(l.kv_heads, l.k_head_dim), (l.kv_heads, l.v_head_dim)])
        .collect();
    let mut got = ir.to_vec();
    want.sort_unstable();
    got.sort_unstable();
    want == got
}

/// Look `key` up at the top level of `config.json`, then in the nested
/// language-model blocks that VLM configs use (`text_config`, `llm_config`,
/// `language_config`), per key so a partially duplicated config still resolves.
fn cfg_get<'a>(cfg: &'a serde_json::Value, key: &str) -> Option<&'a serde_json::Value> {
    [
        Some(cfg),
        cfg.get("text_config"),
        cfg.get("llm_config"),
        cfg.get("language_config"),
    ]
    .into_iter()
    .flatten()
    .find_map(|c| c.get(key).filter(|v| !v.is_null()))
}

fn cfg_usize(cfg: &serde_json::Value, key: &str) -> Option<usize> {
    cfg_get(cfg, key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
}

/// Upper bound on `num_hidden_layers` accepted from a config (a corrupt file must not
/// allocate without limit).
const MAX_LAYERS: usize = 4096;

/// Layer list derived from `config.json`.
struct ConfigLayers {
    pa: Vec<PaLayer>,
    /// LA/conv state per sequence computed from the config (used only when the IR does not show it).
    config_state_bytes: u64,
    /// A `layer_types` entry that is neither paged attention nor a known LA kind, or the
    /// list does not cover `num_hidden_layers`.
    unknown_layer_type: bool,
    /// `num_kv_shared_layers > 0`: later layers reuse earlier layers' KV.
    shared_kv_layers: bool,
}

fn config_layers(cfg: &serde_json::Value) -> Option<ConfigLayers> {
    let num_layers = cfg_usize(cfg, "num_hidden_layers").filter(|n| *n <= MAX_LAYERS)?;
    let shared_kv_layers = cfg_usize(cfg, "num_kv_shared_layers").is_some_and(|n| n > 0);
    let heads = cfg_usize(cfg, "num_attention_heads").filter(|h| *h > 0)?;
    let kv_heads = cfg_usize(cfg, "num_key_value_heads").unwrap_or(heads);
    let hidden = cfg_usize(cfg, "hidden_size")?;
    let head_dim = cfg_usize(cfg, "head_dim").unwrap_or(hidden / heads);
    // Gemma-4 full-attention layers use their own (fewer, wider) KV heads.
    let global = cfg_usize(cfg, "global_head_dim").map(|d| {
        (
            cfg_usize(cfg, "num_global_key_value_heads").unwrap_or(kv_heads),
            d,
        )
    });
    let sliding = PaLayer {
        kv_heads,
        k_head_dim: head_dim,
        v_head_dim: head_dim,
    };
    let full = global.map_or(sliding, |(h, d)| PaLayer {
        kv_heads: h,
        k_head_dim: d,
        v_head_dim: d,
    });

    let Some(types) = cfg_get(cfg, "layer_types").and_then(serde_json::Value::as_array) else {
        // No layer list: every layer is a dense attention layer.
        return Some(ConfigLayers {
            pa: vec![sliding; num_layers],
            config_state_bytes: 0,
            unknown_layer_type: false,
            shared_kv_layers,
        });
    };

    let mut pa = Vec::new();
    let mut state = 0u64;
    let mut unknown = types.len() != num_layers;
    for t in types {
        match t.as_str() {
            Some("full_attention") => pa.push(full),
            Some("sliding_attention") => pa.push(sliding),
            Some("linear_attention") => match linear_attention_state_bytes(cfg) {
                Some(b) => state += b,
                None => unknown = true,
            },
            Some("conv") => match cfg_usize(cfg, "conv_L_cache") {
                Some(k) => state += (hidden * k * 2) as u64,
                None => unknown = true,
            },
            _ => unknown = true,
        }
    }
    Some(ConfigLayers {
        pa,
        config_state_bytes: state,
        unknown_layer_type: unknown,
        shared_kv_layers,
    })
}

/// f16 state of one gated-delta-net layer: the recurrent state
/// `[value_heads, k_dim, v_dim]` plus the convolution state `[conv_dim, kernel]`
/// with `conv_dim = 2·key_heads·k_dim + value_heads·v_dim`.
fn linear_attention_state_bytes(cfg: &serde_json::Value) -> Option<u64> {
    let value_heads = cfg_usize(cfg, "linear_num_value_heads")? as u64;
    let key_heads = cfg_usize(cfg, "linear_num_key_heads")? as u64;
    let k_dim = cfg_usize(cfg, "linear_key_head_dim")? as u64;
    let v_dim = cfg_usize(cfg, "linear_value_head_dim")? as u64;
    let kernel = cfg_usize(cfg, "linear_conv_kernel_dim")? as u64;
    let conv_dim = 2 * key_heads * k_dim + value_heads * v_dim;
    Some((value_heads * k_dim * v_dim + conv_dim * kernel) * 2)
}

// ------------------------------------------------------------------------ IR

/// Facts read from the IR `.xml`.
#[derive(Debug, Default, PartialEq)]
struct IrScan {
    /// A `MatMul` weight chain ends in an `i4`/`u4` constant.
    weights_4bit: bool,
    /// `rt_info/runtime_options/KV_CACHE_PRECISION`.
    ir_kv_hint: Option<KvPrecision>,
    /// Number of `ScaledDotProductAttention` ops (one per paged-attention layer).
    sdpa_count: usize,
    /// Σ over the static `ReadValue` states (LA/conv) of elements × 2 B (f16).
    state_bytes: u64,
    /// The IR asks for f32 inference precision (scale/zero-point bytes then differ).
    f32_inference: bool,
    /// A `KV_CACHE_PRECISION` hint is present but is not u4/i4/i8/u8/f16 (e.g. `f32`): the
    /// plugin would honour it and the layout table does not cover it.
    ir_hint_unparsed: bool,
    /// `(heads, dim)` of every KV-cache `ReadValue` (`?,h,?,d`), keys and values both.
    kv_variables: Vec<(usize, usize)>,
}

/// An IR layer as far as the scans need it.
struct Layer<'a> {
    ty: &'a str,
    const_type: Option<&'a str>,
}

/// How many nodes up the port-0 chain from a `MatMul` weight input are searched for a constant.
const WEIGHT_CHAIN_HOPS: usize = 8;

/// Parse an IR `.xml` and extract the facts above. `None` when it is not parseable IR.
///
/// The 4-bit scan mirrors the GPU plugin's rule (`execution_config.cpp`): for every
/// top-level `MatMul`, take input port 1 (the weights), then follow input port 0 of each
/// node for at most [`WEIGHT_CHAIN_HOPS`] hops until a `Const`; one `i4`/`u4` constant
/// anywhere is enough. Only direct children of `<layers>` are inspected, so ops inside
/// nested bodies (loops) do not count.
fn scan_ir(xml: &str) -> Option<IrScan> {
    let doc = roxmltree::Document::parse(xml).ok()?;
    let net = doc.root_element();
    let layers = net.children().find(|n| n.has_tag_name("layers"))?;

    let mut by_id: HashMap<&str, Layer> = HashMap::new();
    let mut matmuls: Vec<&str> = Vec::new();
    let mut scan = IrScan::default();
    let mut seen_states: HashSet<&str> = HashSet::new();

    for l in layers.children().filter(|n| n.has_tag_name("layer")) {
        let (Some(id), Some(ty)) = (l.attribute("id"), l.attribute("type")) else {
            continue;
        };
        let data = l.children().find(|n| n.has_tag_name("data"));
        let const_type = if ty == "Const" {
            data.and_then(|d| d.attribute("element_type"))
        } else {
            None
        };
        match ty {
            "MatMul" => matmuls.push(id),
            "ScaledDotProductAttention" => scan.sdpa_count += 1,
            "ReadValue" => {
                if let Some(d) = data {
                    let var = d.attribute("variable_id").unwrap_or(id);
                    if seen_states.insert(var) {
                        let shape = d.attribute("variable_shape");
                        scan.state_bytes += static_state_bytes(shape);
                        if let Some(hd) = kv_variable_dims(shape) {
                            scan.kv_variables.push(hd);
                        }
                    }
                }
            }
            _ => {}
        }
        by_id.insert(id, Layer { ty, const_type });
    }

    // (to-layer, to-port) -> from-layer
    let mut producer: HashMap<(&str, &str), &str> = HashMap::new();
    if let Some(edges) = net.children().find(|n| n.has_tag_name("edges")) {
        for e in edges.children().filter(|n| n.has_tag_name("edge")) {
            if let (Some(from), Some(to), Some(port)) = (
                e.attribute("from-layer"),
                e.attribute("to-layer"),
                e.attribute("to-port"),
            ) {
                producer.insert((to, port), from);
            }
        }
    }

    for mm in matmuls {
        let mut cur = producer.get(&(mm, "1")).copied();
        for _ in 0..=WEIGHT_CHAIN_HOPS {
            let Some(id) = cur else { break };
            let Some(layer) = by_id.get(id) else { break };
            if layer.ty == "Const" {
                if matches!(layer.const_type, Some("i4" | "u4")) {
                    scan.weights_4bit = true;
                }
                break;
            }
            cur = producer.get(&(id, "0")).copied();
        }
        if scan.weights_4bit {
            break;
        }
    }

    if let Some(opts) = net
        .children()
        .find(|n| n.has_tag_name("rt_info"))
        .and_then(|rt| rt.children().find(|n| n.has_tag_name("runtime_options")))
    {
        for o in opts.children().filter(roxmltree::Node::is_element) {
            let value = o.attribute("value").unwrap_or_default();
            match o.tag_name().name() {
                "KV_CACHE_PRECISION" => {
                    scan.ir_kv_hint = KvPrecision::from_ir_hint(value);
                    scan.ir_hint_unparsed = scan.ir_kv_hint.is_none();
                }
                "INFERENCE_PRECISION_HINT" => {
                    scan.f32_inference = value.trim().eq_ignore_ascii_case("f32");
                }
                _ => {}
            }
        }
    }
    Some(scan)
}

/// `(heads, dim)` of a KV-cache variable shaped `?,h,?,d`.
fn kv_variable_dims(shape: Option<&str>) -> Option<(usize, usize)> {
    let dims: Vec<&str> = shape?.split(',').map(str::trim).collect();
    match dims.as_slice() {
        [b, h, s, d] if *b == "?" && *s == "?" => Some((h.parse().ok()?, d.parse().ok()?)),
        _ => None,
    }
}

/// Bytes (f16) of a `ReadValue` variable whose shape is `?,d1,d2,…` with every dim after the
/// batch static. KV-cache variables (`?,h,?,d`) have a second dynamic dim and count as 0.
fn static_state_bytes(shape: Option<&str>) -> u64 {
    let Some(shape) = shape else { return 0 };
    let mut dims = shape.split(',').map(str::trim);
    if dims.next() != Some("?") {
        return 0;
    }
    let mut elems = 1u64;
    let mut any = false;
    for d in dims {
        match d.parse::<u64>() {
            Ok(n) => {
                elems = elems.saturating_mul(n);
                any = true;
            }
            Err(_) => return 0,
        }
    }
    if any { elems.saturating_mul(2) } else { 0 }
}

// ------------------------------------------------------------- engine version

/// Normalise an `OpenVINO` / `GenAI` version string pair for comparison with
/// `KNOWN_ENGINE_PAIRS`, the ratchet key and log lines: drop the build suffix and keep
/// at most the first three numeric components (`"2026.4.1-22982-e213a147257"` and
/// `"2026.4.1.0-…"` both become `"2026.4.1"`).
pub(crate) fn engine_pair(ov: &str, genai: &str) -> (String, String) {
    fn norm(v: &str) -> String {
        let core = v.trim().split('-').next().unwrap_or_default();
        core.split('.').take(3).collect::<Vec<_>>().join(".")
    }
    (norm(ov), norm(genai))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use serde_json::json;
    use std::fmt::Write as _;

    // ---- layout arithmetic (plan 06 §1.1 table) ----------------------------

    fn dense(layers: usize, heads: usize, dim: usize) -> KvProfile {
        KvProfile {
            pa_layers: vec![
                PaLayer {
                    kv_heads: heads,
                    k_head_dim: dim,
                    v_head_dim: dim,
                };
                layers
            ],
            block_tokens: GPU_BLOCK_TOKENS,
            weights_4bit: Some(true),
            ir_kv_hint: None,
            linear_state_bytes_per_seq: 0,
            hidden_size: 4096,
            confidence: Confidence::High,
        }
    }

    #[test]
    fn layout_table_per_layer_token() {
        // (h, d) -> (u4, i8, f16) bytes per layer-token, straight from the plan table.
        let rows = [
            ((4, 256), (1296, 2320, 4096)),
            ((8, 128), (1312, 2336, 4096)),
            ((8, 64), (672, 1184, 2048)),
            ((32, 96), (3968, 7040, 12_288)),
            ((2, 512), (1288, 2312, 4096)),
        ];
        for ((h, d), (u4, i8, f16)) in rows {
            let p = dense(1, h, d);
            assert_eq!(p.kv_bytes_per_token(KvPrecision::U4), u4, "u4 h={h} d={d}");
            assert_eq!(p.kv_bytes_per_token(KvPrecision::I8), i8, "i8 h={h} d={d}");
            assert_eq!(
                p.kv_bytes_per_token(KvPrecision::F16),
                f16,
                "f16 h={h} d={d}"
            );
        }
    }

    #[test]
    fn block_bytes_scale_with_block_tokens() {
        let mut p = dense(36, 8, 128);
        assert_eq!(p.kv_block_bytes(KvPrecision::U4), 755_712);
        assert_eq!(p.kv_block_bytes(KvPrecision::I8), 1_345_536);
        p.block_tokens = CPU_BLOCK_TOKENS;
        assert_eq!(p.kv_block_bytes(KvPrecision::U4), 2 * 755_712);
        assert_eq!(p.kv_bytes_per_token(KvPrecision::U4), 47_232);
    }

    #[test]
    fn heterogeneous_layers_gemma4() {
        // U-04: 5 global layers (2 heads x 512) + 25 sliding layers (8 x 256).
        let cfg = json!({
            "num_hidden_layers": 30, "num_attention_heads": 16, "num_key_value_heads": 8,
            "head_dim": 256, "hidden_size": 2816, "global_head_dim": 512,
            "num_global_key_value_heads": 2,
            "layer_types": (0..30).map(|i| if i % 6 == 5 { "full_attention" } else { "sliding_attention" }).collect::<Vec<_>>(),
        });
        let p = KvProfile::from_parts(&cfg, None, "GPU.1", false).expect("profile");
        assert_eq!(p.pa_layers.len(), 30);
        let u4 = 5 * (2 * (384 + 256 + 4)) + 25 * (8 * (192 + 128 + 4));
        let i8 = 5 * (2 * (640 + 512 + 4)) + 25 * (8 * (320 + 256 + 4));
        assert_eq!(p.kv_bytes_per_token(KvPrecision::U4), 71_240);
        assert_eq!(p.kv_bytes_per_token(KvPrecision::I8), 127_560);
        assert_eq!(u4, 71_240);
        assert_eq!(i8, 127_560);
    }

    // ---- config.json -------------------------------------------------------

    #[test]
    fn dense_config_without_layer_types() {
        let cfg = json!({"num_hidden_layers": 36, "num_attention_heads": 32,
            "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 4096});
        let p = KvProfile::from_parts(&cfg, None, "GPU.1", false).expect("profile");
        assert_eq!(p.pa_layers.len(), 36);
        assert_eq!(p.kv_bytes_per_token(KvPrecision::U4), 47_232);
        assert_eq!(p.kv_bytes_per_token(KvPrecision::I8), 84_096);
        assert_eq!(p.linear_state_bytes_per_seq, 0);
        // No IR: the facts that need it are unknown and the profile is only Medium.
        assert_eq!(p.weights_4bit, None);
        assert_eq!(p.confidence, Confidence::Medium);
    }

    #[test]
    fn head_dim_falls_back_to_hidden_over_heads() {
        let cfg = json!({"num_hidden_layers": 32, "num_attention_heads": 32,
            "num_key_value_heads": 32, "hidden_size": 3072});
        let p = KvProfile::from_parts(&cfg, None, "GPU.1", false).expect("profile");
        assert_eq!(p.kv_bytes_per_token(KvPrecision::U4), 32 * 3968);
        assert_eq!(p.kv_bytes_per_token(KvPrecision::F16), 32 * 12_288);
    }

    #[test]
    fn nested_text_config_is_resolved_per_key() {
        let cfg = json!({"text_config": {"num_hidden_layers": 36, "num_attention_heads": 32,
            "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 4096}});
        let p = KvProfile::from_parts(&cfg, None, "GPU.1", false).expect("profile");
        assert_eq!(p.pa_layers.len(), 36);
        assert_eq!(p.hidden_size, 4096);
    }

    #[test]
    fn hybrid_qwen35_la_state_from_config() {
        let mut types = Vec::new();
        for i in 0..64 {
            types.push(if i % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            });
        }
        let cfg = json!({"text_config": {"num_hidden_layers": 64, "num_attention_heads": 24,
            "num_key_value_heads": 4, "head_dim": 256, "hidden_size": 5120,
            "layer_types": types, "linear_num_value_heads": 48, "linear_key_head_dim": 128,
            "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "linear_num_key_heads": 16}});
        let p = KvProfile::from_parts(&cfg, None, "GPU.1", false).expect("profile");
        assert_eq!(p.pa_layers.len(), 16);
        assert_eq!(p.kv_bytes_per_token(KvPrecision::U4), 20_736);
        assert_eq!(p.kv_bytes_per_token(KvPrecision::I8), 37_120);
        assert_eq!(p.linear_state_bytes_per_seq, 79_429_632);
        // LA state computed rather than read from the IR: Medium.
        assert_eq!(p.confidence, Confidence::Medium);
    }

    #[test]
    fn lfm2_conv_state_from_config() {
        let mut types = Vec::new();
        for i in 0..40 {
            types.push(if i % 4 == 2 { "full_attention" } else { "conv" });
        }
        let cfg = json!({"num_hidden_layers": 40, "num_attention_heads": 32,
            "num_key_value_heads": 8, "hidden_size": 2048, "layer_types": types,
            "conv_L_cache": 3});
        let p = KvProfile::from_parts(&cfg, None, "GPU.1", false).expect("profile");
        assert_eq!(p.pa_layers.len(), 10);
        assert_eq!(p.kv_bytes_per_token(KvPrecision::U4), 6720);
        assert_eq!(p.kv_bytes_per_token(KvPrecision::I8), 11_840);
        assert_eq!(p.linear_state_bytes_per_seq, 368_640);
    }

    #[test]
    fn unknown_layer_type_is_low_confidence() {
        let cfg = json!({"num_hidden_layers": 2, "num_attention_heads": 8, "hidden_size": 512,
            "layer_types": ["full_attention", "mamba2"]});
        let p = KvProfile::from_parts(&cfg, None, "GPU.1", false).expect("profile");
        assert_eq!(p.confidence, Confidence::Low);
    }

    #[test]
    fn missing_geometry_gives_no_profile() {
        let cfg = json!({"num_hidden_layers": 2});
        assert!(KvProfile::from_parts(&cfg, None, "GPU.1", false).is_none());
        let zero_heads =
            json!({"num_hidden_layers": 2, "num_attention_heads": 0, "hidden_size": 8});
        assert!(KvProfile::from_parts(&zero_heads, None, "GPU.1", false).is_none());
    }

    #[test]
    fn cpu_device_uses_32_token_blocks() {
        let cfg = json!({"num_hidden_layers": 1, "num_attention_heads": 8, "hidden_size": 1024,
            "head_dim": 128});
        let p = KvProfile::from_parts(&cfg, None, "CPU", false).expect("profile");
        assert_eq!(p.block_tokens, 32);
        let gpu = KvProfile::from_parts(&cfg, None, "GPU.1", false).expect("profile");
        assert_eq!(gpu.block_tokens, 16);
    }

    #[test]
    fn mtp_mode_forces_low_confidence() {
        // U-28
        let cfg = json!({"num_hidden_layers": 2, "num_attention_heads": 8, "hidden_size": 512});
        let ir = IrScan {
            sdpa_count: 2,
            ..IrScan::default()
        };
        let ok = KvProfile::from_parts(&cfg, Some(&ir), "GPU.1", false).expect("profile");
        assert_eq!(ok.confidence, Confidence::High);
        let mtp = KvProfile::from_parts(&cfg, Some(&ir), "GPU.1", true).expect("profile");
        assert_eq!(mtp.confidence, Confidence::Low);
    }

    #[test]
    fn sdpa_mismatch_or_f32_inference_is_low() {
        let cfg = json!({"num_hidden_layers": 4, "num_attention_heads": 8, "hidden_size": 512});
        // gemma-4-e4b shape: layers in the config, no SDPA ops in the graph.
        let none = IrScan::default();
        let p = KvProfile::from_parts(&cfg, Some(&none), "GPU.1", false).expect("profile");
        assert_eq!(p.confidence, Confidence::Low);
        let f32_ir = IrScan {
            sdpa_count: 4,
            f32_inference: true,
            ..IrScan::default()
        };
        let p = KvProfile::from_parts(&cfg, Some(&f32_ir), "GPU.1", false).expect("profile");
        assert_eq!(p.confidence, Confidence::Low);
    }

    fn la_cfg() -> serde_json::Value {
        // One LA layer: (2·4·4 + (2·1·4 + 2·4)·4) · 2 = 192 B of state, one full-attention layer.
        json!({"num_hidden_layers": 2, "num_attention_heads": 8, "num_key_value_heads": 2,
            "head_dim": 16, "hidden_size": 512,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_value_heads": 2, "linear_num_key_heads": 1, "linear_key_head_dim": 4,
            "linear_value_head_dim": 4, "linear_conv_kernel_dim": 4})
    }

    #[test]
    fn ir_state_agreeing_with_config_is_high() {
        let ir = IrScan {
            sdpa_count: 1,
            state_bytes: 192,
            kv_variables: vec![(2, 16), (2, 16)],
            ..IrScan::default()
        };
        let p = KvProfile::from_parts(&la_cfg(), Some(&ir), "GPU.1", false).expect("profile");
        assert_eq!(p.linear_state_bytes_per_seq, 192);
        assert_eq!(p.confidence, Confidence::High);
    }

    #[test]
    fn ir_state_disagreeing_with_config_is_low() {
        let ir = IrScan {
            sdpa_count: 1,
            state_bytes: 12_345,
            kv_variables: vec![(2, 16), (2, 16)],
            ..IrScan::default()
        };
        let p = KvProfile::from_parts(&la_cfg(), Some(&ir), "GPU.1", false).expect("profile");
        // The IR figure is kept (it is what the graph holds) but the profile is not trusted.
        assert_eq!(p.linear_state_bytes_per_seq, 12_345);
        assert_eq!(p.confidence, Confidence::Low);
        // IR state on a model whose config has no LA layers is a mismatch too.
        let dense = json!({"num_hidden_layers": 1, "num_attention_heads": 8, "hidden_size": 512});
        let ir = IrScan {
            sdpa_count: 1,
            state_bytes: 64,
            ..IrScan::default()
        };
        let p = KvProfile::from_parts(&dense, Some(&ir), "GPU.1", false).expect("profile");
        assert_eq!(p.confidence, Confidence::Low);
    }

    #[test]
    fn ir_kv_variable_dims_must_match_the_config() {
        let cfg = json!({"num_hidden_layers": 2, "num_attention_heads": 8,
            "num_key_value_heads": 4, "head_dim": 64, "hidden_size": 512});
        let good = IrScan {
            sdpa_count: 2,
            kv_variables: vec![(4, 64); 4],
            ..IrScan::default()
        };
        let p = KvProfile::from_parts(&cfg, Some(&good), "GPU.1", false).expect("profile");
        assert_eq!(p.confidence, Confidence::High);
        // A value head dim that differs from the key head dim (MLA-style): not priced.
        let mla = IrScan {
            sdpa_count: 2,
            kv_variables: vec![(4, 64), (4, 32), (4, 64), (4, 32)],
            ..IrScan::default()
        };
        let p = KvProfile::from_parts(&cfg, Some(&mla), "GPU.1", false).expect("profile");
        assert_eq!(p.confidence, Confidence::Low);
        // Wrong head count (a bad hidden/heads guess would look like this).
        let wrong = IrScan {
            sdpa_count: 2,
            kv_variables: vec![(8, 64); 4],
            ..IrScan::default()
        };
        let p = KvProfile::from_parts(&cfg, Some(&wrong), "GPU.1", false).expect("profile");
        assert_eq!(p.confidence, Confidence::Low);
    }

    #[test]
    fn kv_variables_are_parsed_from_the_ir() {
        let layers = r#"
            <layer id="60" type="ReadValue"><data variable_id="k0" variable_shape="?,4,?,256"/></layer>
            <layer id="61" type="ReadValue"><data variable_id="v0" variable_shape="?,4,?,256"/></layer>
            <layer id="62" type="ReadValue"><data variable_id="s0" variable_shape="?,32,128,128"/></layer>"#;
        let s = scan_ir(&ir_with_chain(&[("Const", "f16")], layers, "", "")).expect("ir");
        assert_eq!(s.kv_variables, vec![(4, 256), (4, 256)]);
    }

    #[test]
    fn shared_kv_layers_or_short_layer_list_is_low() {
        let shared = json!({"num_hidden_layers": 4, "num_attention_heads": 8, "hidden_size": 512,
            "num_kv_shared_layers": 2});
        let p = KvProfile::from_parts(&shared, None, "GPU.1", false).expect("profile");
        assert_eq!(p.confidence, Confidence::Low);
        let short = json!({"num_hidden_layers": 4, "num_attention_heads": 8, "hidden_size": 512,
            "layer_types": ["full_attention", "full_attention"]});
        let p = KvProfile::from_parts(&short, None, "GPU.1", false).expect("profile");
        assert_eq!(p.confidence, Confidence::Low);
    }

    #[test]
    fn ir_hint_i4_is_u4_and_an_unparsed_hint_is_low() {
        assert_eq!(KvPrecision::from_ir_hint("i4"), Some(KvPrecision::U4));
        let cfg = json!({"num_hidden_layers": 1, "num_attention_heads": 8, "hidden_size": 512});
        let rt = r#"<rt_info><runtime_options><KV_CACHE_PRECISION value="f32"/></runtime_options></rt_info>"#;
        let s = scan_ir(&ir_with_chain(&[("Const", "f16")], "", "", rt)).expect("ir");
        assert!(s.ir_hint_unparsed);
        let ir = IrScan { sdpa_count: 1, ..s };
        let p = KvProfile::from_parts(&cfg, Some(&ir), "GPU.1", false).expect("profile");
        assert_eq!(p.confidence, Confidence::Low);
    }

    #[test]
    fn null_nested_value_does_not_shadow_and_layer_count_is_capped() {
        let cfg = json!({"head_dim": null, "text_config": {"head_dim": 128}, "num_hidden_layers": 1,
            "num_attention_heads": 8, "hidden_size": 1024});
        assert_eq!(cfg_usize(&cfg, "head_dim"), Some(128));
        let huge = json!({"num_hidden_layers": 4_000_000_000_u64, "num_attention_heads": 8,
            "hidden_size": 512});
        assert!(KvProfile::from_parts(&huge, None, "GPU.1", false).is_none());
    }

    // ---- IR scan (U-02) ----------------------------------------------------

    /// Build a tiny IR: a chain of (type, `element_type`) nodes feeding `MatMul` port 1.
    /// `chain[0]` is the node adjacent to the `MatMul`; each next node feeds the previous one's port 0.
    fn ir_with_chain(
        chain: &[(&str, &str)],
        extra_layers: &str,
        extra_edges: &str,
        rt: &str,
    ) -> String {
        let mut layers = String::new();
        let mut edges = String::new();
        for (i, (ty, et)) in chain.iter().enumerate() {
            let data = if *ty == "Const" {
                format!(r#"<data element_type="{et}" shape="1" offset="0" size="1"/>"#)
            } else {
                String::new()
            };
            write!(
                layers,
                r#"<layer id="{}" type="{ty}" name="n{i}">{data}</layer>"#,
                10 + i
            )
            .unwrap();
            if i == 0 {
                edges.push_str(r#"<edge from-layer="10" from-port="0" to-layer="1" to-port="1"/>"#);
            } else {
                write!(
                    edges,
                    r#"<edge from-layer="{}" from-port="0" to-layer="{}" to-port="0"/>"#,
                    10 + i,
                    9 + i
                )
                .unwrap();
            }
        }
        format!(
            r#"<net><layers><layer id="1" type="MatMul" name="mm"/>{layers}{extra_layers}</layers><edges>{edges}{extra_edges}</edges>{rt}</net>"#
        )
    }

    fn fires(chain: &[(&str, &str)]) -> bool {
        scan_ir(&ir_with_chain(chain, "", "", ""))
            .expect("ir")
            .weights_4bit
    }

    #[test]
    fn u02_standard_decompression_chain_fires() {
        // Multiply <- Subtract <- Convert <- Const(u4)
        assert!(fires(&[
            ("Multiply", ""),
            ("Subtract", ""),
            ("Convert", ""),
            ("Const", "u4"),
        ]));
    }

    #[test]
    fn u02_const_on_port_one_of_subtract_does_not_fire() {
        // The u4 constant is the zero point (Subtract port 1), not on the port-0 chain.
        let xml = r#"<net><layers>
            <layer id="1" type="MatMul"/>
            <layer id="2" type="Multiply"/>
            <layer id="3" type="Subtract"/>
            <layer id="4" type="Const"><data element_type="f16"/></layer>
            <layer id="5" type="Const"><data element_type="u4"/></layer>
            </layers><edges>
            <edge from-layer="2" from-port="0" to-layer="1" to-port="1"/>
            <edge from-layer="3" from-port="0" to-layer="2" to-port="0"/>
            <edge from-layer="4" from-port="0" to-layer="3" to-port="0"/>
            <edge from-layer="5" from-port="0" to-layer="3" to-port="1"/>
            </edges></net>"#;
        assert!(!scan_ir(xml).expect("ir").weights_4bit);
    }

    #[test]
    fn u02_i4_fires_u8_does_not() {
        assert!(fires(&[("Convert", ""), ("Const", "i4")]));
        assert!(!fires(&[("Convert", ""), ("Const", "u8")]));
    }

    #[test]
    fn u02_hop_limit() {
        // 8 hops from the weight producer to the constant fires, 9 does not.
        let mut within: Vec<(&str, &str)> = vec![("Convert", ""); WEIGHT_CHAIN_HOPS];
        within.push(("Const", "u4"));
        assert!(fires(&within));
        let mut beyond: Vec<(&str, &str)> = vec![("Convert", ""); WEIGHT_CHAIN_HOPS + 1];
        beyond.push(("Const", "u4"));
        assert!(!fires(&beyond));
    }

    #[test]
    fn u02_moe_expert_i4_with_u8_attention_is_true() {
        let xml = r#"<net><layers>
            <layer id="1" type="MatMul"/><layer id="2" type="Const"><data element_type="u8"/></layer>
            <layer id="3" type="MatMul"/><layer id="4" type="Convert"/>
            <layer id="5" type="Const"><data element_type="i4"/></layer>
            </layers><edges>
            <edge from-layer="2" from-port="0" to-layer="1" to-port="1"/>
            <edge from-layer="4" from-port="0" to-layer="3" to-port="1"/>
            <edge from-layer="5" from-port="0" to-layer="4" to-port="0"/>
            </edges></net>"#;
        assert!(scan_ir(xml).expect("ir").weights_4bit);
    }

    #[test]
    fn u02_chain_without_const_is_ignored() {
        // A Gather from an embedding table: Parameter -> Gather -> MatMul[1]; no Const in reach.
        assert!(!fires(&[("Gather", ""), ("Parameter", "")]));
    }

    #[test]
    fn ir_hint_and_inference_precision_are_read() {
        let rt = r#"<rt_info><runtime_options><KV_CACHE_PRECISION value="f16"/></runtime_options></rt_info>"#;
        let s = scan_ir(&ir_with_chain(&[("Const", "f16")], "", "", rt)).expect("ir");
        assert_eq!(s.ir_kv_hint, Some(KvPrecision::F16));
        assert!(!s.f32_inference);
        let rt = r#"<rt_info><runtime_options><KV_CACHE_PRECISION value="u8"/><INFERENCE_PRECISION_HINT value="f32"/></runtime_options></rt_info>"#;
        let s = scan_ir(&ir_with_chain(&[("Const", "f16")], "", "", rt)).expect("ir");
        assert_eq!(s.ir_kv_hint, Some(KvPrecision::I8));
        assert!(s.f32_inference);
        let rt = r#"<rt_info><runtime_options><KV_CACHE_PRECISION value="f32"/></runtime_options></rt_info>"#;
        let s = scan_ir(&ir_with_chain(&[("Const", "f16")], "", "", rt)).expect("ir");
        assert_eq!(s.ir_kv_hint, None);
    }

    #[test]
    fn sdpa_and_state_variables_are_counted() {
        let layers = r#"
            <layer id="50" type="ScaledDotProductAttention"/>
            <layer id="51" type="ScaledDotProductAttention"/>
            <layer id="60" type="ReadValue"><data variable_id="s0" variable_shape="?,32,128,128"/></layer>
            <layer id="61" type="ReadValue"><data variable_id="s1" variable_shape="?,8192,4"/></layer>
            <layer id="62" type="ReadValue"><data variable_id="kv" variable_shape="?,4,?,256"/></layer>
            <layer id="63" type="ReadValue"><data variable_id="s0" variable_shape="?,32,128,128"/></layer>"#;
        let s = scan_ir(&ir_with_chain(&[("Const", "f16")], layers, "", "")).expect("ir");
        assert_eq!(s.sdpa_count, 2);
        // s0 and s1 once each (the duplicate id is ignored), the KV variable not at all.
        assert_eq!(s.state_bytes, (32 * 128 * 128 + 8192 * 4) * 2);
    }

    #[test]
    fn unparseable_ir_is_none() {
        assert!(scan_ir("not xml <<<").is_none());
        assert!(scan_ir("<net></net>").is_none());
    }

    // ---- engine_pair (U-08, U-28) -----------------------------------------

    #[test]
    fn engine_pair_strips_build_suffix_and_extra_components() {
        assert_eq!(
            engine_pair("2026.4.1-22982-e213a147257", "2026.4.1.0-22982-e213a147257"),
            ("2026.4.1".to_string(), "2026.4.1".to_string())
        );
        assert_eq!(
            engine_pair("2026.5.0", "2026.5.0"),
            ("2026.5.0".to_string(), "2026.5.0".to_string())
        );
        assert_eq!(engine_pair("", ""), (String::new(), String::new()));
        assert_eq!(
            engine_pair("2026.4", "unknown"),
            ("2026.4".to_string(), "unknown".to_string())
        );
    }

    // ---- fleet golden table (U-01), needs the model drive -------------------

    const FLEET: &str = "/media/jerzy/F87E-8DF3/models";

    /// `(dir, pa_layers, w4, hint, la_bytes, u4 B/token, i8 B/token)`
    #[allow(clippy::type_complexity)]
    const GOLDEN: &[(&str, usize, bool, Option<KvPrecision>, u64, u64, u64)] = &[
        (
            "Qwen3.8-27B-int4-ov",
            16,
            true,
            None,
            79_429_632,
            20_736,
            37_120,
        ),
        (
            "qwen3.6-27b-int4-ov",
            16,
            true,
            None,
            79_429_632,
            20_736,
            37_120,
        ),
        (
            "qwen3.6-35b-a3b-int4-ov",
            10,
            true,
            None,
            33_423_360,
            6_480,
            11_600,
        ),
        (
            "qwen3.5-9B-int4-ov",
            8,
            true,
            None,
            26_738_688,
            10_368,
            18_560,
        ),
        (
            "qwen3.5-4b-int4-ov",
            8,
            true,
            None,
            26_738_688,
            10_368,
            18_560,
        ),
        (
            "OmniCoder-9B-int4_sym-ov",
            8,
            true,
            None,
            26_738_688,
            10_368,
            18_560,
        ),
        (
            "lfm2-24b-a2b-int4-ov",
            10,
            true,
            None,
            368_640,
            6_720,
            11_840,
        ),
        ("qwen3-8b-int4-ov", 36, true, None, 0, 47_232, 84_096),
        ("qwen3-vl-8b-int4-ov", 36, true, None, 0, 47_232, 84_096),
        ("qwen3-14b-int4-ov", 40, true, None, 0, 52_480, 93_440),
        (
            "qwen3-coder-30b-a3b-int4-ov",
            48,
            true,
            None,
            0,
            31_488,
            56_064,
        ),
        ("gpt-oss-20b-int4-ov", 24, true, None, 0, 16_128, 28_416),
        (
            "gemma-4-26b-a4b-it-int4-ov",
            30,
            true,
            None,
            0,
            71_240,
            127_560,
        ),
        ("phi-4-int4-ov", 40, true, None, 0, 65_600, 116_800),
        (
            "qwen3-8B-fp16-ov",
            36,
            false,
            Some(KvPrecision::F16),
            0,
            47_232,
            84_096,
        ),
        (
            "phi-3.5-vision-int4-ov",
            32,
            true,
            Some(KvPrecision::F16),
            0,
            126_976,
            225_280,
        ),
        ("qwen3-8b-int8-ov", 36, false, None, 0, 47_232, 84_096),
        ("bielik-11b-v3-int4-ov", 50, true, None, 0, 65_600, 116_800),
        ("bielik-11b-v3-int8-ov", 50, false, None, 0, 65_600, 116_800),
        (
            "mistral-small-3.2-24b-int4-ov",
            40,
            true,
            None,
            0,
            52_480,
            93_440,
        ),
        (
            "nanbeige4.1-3b-int4-awq-ov",
            32,
            true,
            None,
            0,
            20_992,
            37_376,
        ),
        ("qwen2.5-vl-7b-int4-ov", 28, true, None, 0, 18_368, 32_704),
        ("qwen2.5-coder-14b-int4", 48, true, None, 0, 62_976, 112_128),
        ("internvl2.5-8b-int4-ov", 32, true, None, 0, 41_984, 74_752),
        ("ministral-3-3b-int8-ov", 26, false, None, 0, 34_112, 60_736),
        ("qwen3-0.6b-int4-ov", 28, true, None, 0, 36_736, 65_408),
        ("qwen3-0.6b-int8-ov", 28, false, None, 0, 36_736, 65_408),
        ("qwen3-1.7b-int4-ov", 28, true, None, 0, 36_736, 65_408),
        ("qwen3-4b-int4-ov", 36, true, None, 0, 47_232, 84_096),
        ("qwen3-4b-int8-ov", 36, false, None, 0, 47_232, 84_096),
        ("qwen3-14b-int8-ov", 40, false, None, 0, 52_480, 93_440),
        ("qwen3-vl-8b-int8-ov", 36, false, None, 0, 47_232, 84_096),
        ("gpt-oss-20b-int8-ov", 24, false, None, 0, 16_128, 28_416),
        ("qwen3-8b-int4-cw-ov", 36, true, None, 0, 47_232, 84_096),
    ];

    #[test]
    #[ignore = "needs the fleet model drive"]
    fn u01_fleet_golden_table() {
        let root = Path::new(FLEET);
        assert!(root.is_dir(), "fleet model drive not mounted: {FLEET}");
        let mut failures = Vec::new();
        for (dir, layers, w4, hint, la, u4, i8) in GOLDEN {
            let Some(p) = KvProfile::read(&root.join(dir), "GPU.1") else {
                failures.push(format!("{dir}: no profile"));
                continue;
            };
            let got = (
                p.pa_layers.len(),
                p.weights_4bit,
                p.ir_kv_hint,
                p.linear_state_bytes_per_seq,
                p.kv_bytes_per_token(KvPrecision::U4),
                p.kv_bytes_per_token(KvPrecision::I8),
            );
            let want = (*layers, Some(*w4), *hint, *la, *u4, *i8);
            if got != want {
                failures.push(format!("{dir}: got {got:?}, want {want:?}"));
            }
            if p.confidence != Confidence::High {
                failures.push(format!("{dir}: confidence {:?}", p.confidence));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    #[ignore = "needs the fleet model drive"]
    fn u01_ungradable_models_are_low_confidence() {
        // gemma-4-e4b shares KV across layers; the pruned qwen3 keeps the 28-layer config
        // of its parent while the graph has 6.
        for name in [
            "gemma-4-e4b-it-int8-ov",
            "qwen3-pruned-6l-from-0.6b-int8-ov",
        ] {
            let dir = Path::new(FLEET).join(name);
            assert!(dir.is_dir(), "model missing: {}", dir.display());
            let p = KvProfile::read(&dir, "GPU.1").expect("profile");
            assert_eq!(p.confidence, Confidence::Low, "{name}");
        }
    }
}
