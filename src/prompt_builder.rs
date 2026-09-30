// ============================================================
// src/prompt_builder.rs — chat-template injection + tool-call parsing
// ============================================================
// Behavioural spec ported from stormVINO `prompt_builder.py`.
//
// Two responsibilities, mirroring the Python original:
//   1. INJECTION  — render the model's own Jinja chat template with the
//                   conversation + tool schemas (`build_prompt`, added in a
//                   later step). The template tells the model what tools exist
//                   and how to emit calls.
//   2. PARSING    — pull the model's tool calls back out of its raw output
//                   (`parse_tool_calls`), and strip Qwen-style `<think>` blocks
//                   (`extract_thinking`).
//
// Family handling: the *parser* differs per family (Qwen/Phi emit
// `<tool_call>{json}</tool_call>`; Mistral emits `name{json}`), so callers
// pass a [`ModelFamily`] detected once at model-load time.
//
// CRASH COURSE — why the regexes are `OnceLock<Option<Regex>>`:
//   These patterns are compile-time-constant literals that cannot realistically
//   fail to compile. But this crate denies `unwrap`/`expect` outside tests, so
//   we cannot `Regex::new(..).unwrap()`. Instead we cache `Regex::new(..).ok()`
//   once and, in the impossible event compilation failed, degrade gracefully to
//   "no matches" — never panicking, never violating the lint.
// ============================================================

use std::sync::OnceLock;

use chrono::Local;
use minijinja::{AutoEscape, Environment, Value as JinjaValue, context};
use regex::Regex;
use serde::Serialize;

// ---- Model family ---------------------------------------------------

/// Which prompt/tool-call dialect a model speaks.
///
/// Detected from the model's chat template at load time (see [`ModelFamily::detect`]).
/// Drives [`parse_tool_calls`] — injection is universal (the model's own template).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    /// Native `apply_chat_template` tool support: Qwen3, Phi, Llama 3, Gemma.
    /// Output format: `<tool_call>{"name":…, "arguments":…}</tool_call>`.
    Default,
    /// Mistral anthracite-core `[SYSTEM_PROMPT][INST]` tokenizer.
    /// Output format: `function_name{"key": "value"}`.
    Mistral,
    /// Qwen3-Coder XML dialect.
    /// Injection: XML `<tools><function>…</function></tools>` schema block.
    /// Output format: `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`.
    QwenCoder,
    /// gpt-oss's harmony format tool calls (the `commentary` channel).
    /// Output format (after `HarmonyFilter`/`extract_reasoning_gpt_oss` has
    /// already stripped every true special token and channel-name word):
    /// `to=functions.name json{"key": "value"}` — the function name lives in
    /// the `to=` header, *outside* the JSON, which has no `"name"` key at
    /// all, unlike every other dialect above.
    GptOss,
    /// `LiquidAI` LFM2, which carries a dedicated tool protocol in its own
    /// vocabulary rather than in plain text: `<|tool_list_start|>`,
    /// `<|tool_call_start|>`, `<|tool_response_start|>` (and matching `_end`
    /// tokens).
    ///
    /// Output format: `<|tool_call_start|>[name(key="value"), …]<|tool_call_end|>`
    /// — a **Python list of keyword calls**, not JSON, optionally followed by
    /// prose in the same turn.
    ///
    /// **Caveat measured 2026-09-08** (the project's internal engineering log):
    /// those delimiters are *special* tokens (ids 8-13) and the model's
    /// detokenizer strips them — verified live via `/detokenize`, where each
    /// decodes to the empty string. [`parse_lfm2`] therefore bounds the call
    /// list by **shape** rather than by markers, and strips the markers when
    /// they are present so it upgrades cleanly if the bridge ever preserves
    /// them.
    Lfm2,
}

impl ModelFamily {
    /// Detect the family from the model's chat-template string.
    ///
    /// Mirrors stormVINO `get_adapter`: a `[SYSTEM_PROMPT]` marker means the
    /// Mistral template; `<function=` in the template instruction block means the
    /// Qwen3-Coder XML dialect; `<|channel|>` (present only in gpt-oss's harmony
    /// template on this box — checked against every other served model's
    /// template) means gpt-oss; everything else uses the Default (native-Jinja)
    /// dialect. (The `InternVL` `<IMG_CONTEXT>` branch is out of scope for text
    /// tool calling.)
    #[must_use]
    pub fn detect(template: &str) -> Self {
        if template.contains("[SYSTEM_PROMPT]") {
            Self::Mistral
        } else if template.contains("<function=") {
            Self::QwenCoder
        } else if template.contains("<|channel|>") {
            Self::GptOss
        } else if template.contains("<|tool_call_start|>")
            || template.contains("keep_past_thinking")
        {
            // Two markers because LFM2 ships two template generations. A
            // corrected template renders the model's own tool protocol and so
            // contains `<|tool_call_start|>` directly. The template actually
            // shipped with `lfm2-24b-a2b-int4-ov` does *not* — it renders tools
            // as plain `List of tools: [...]` text — so it is recognised by
            // `keep_past_thinking`, a variable unique to LFM2 across every
            // model served on this fleet (checked against every
            // `chat_template.jinja` present). Matching either keeps detection
            // stable across a template correction.
            Self::Lfm2
        } else {
            Self::Default
        }
    }

    /// Snake-case label for API surfaces (`generation_metadata`-style
    /// debugging fields) — same convention as [`ModelKind::label`]
    /// (`crate::model_manager::engine::ModelKind::label`).
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Mistral => "mistral",
            Self::QwenCoder => "qwen_coder",
            Self::GptOss => "gpt_oss",
            Self::Lfm2 => "lfm2",
        }
    }
}

// ---- OpenAI tool-call shapes ----------------------------------------

/// The `function` payload inside an `OpenAI` tool call.
///
/// `arguments` is a JSON **string** (not an object) — this is the `OpenAI`
/// wire format: the model's argument object re-serialised to a compact string.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct FunctionCall {
    /// The function/tool name the model chose to call.
    pub name: String,
    /// The call arguments, encoded as a compact JSON string.
    pub arguments: String,
}

/// One `OpenAI`-format tool call, as it appears in `message.tool_calls`.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ToolCall {
    /// Unique-within-response id (`"call_" + 8 hex`); the client echoes it back
    /// in the follow-up `tool` message so server and client can correlate.
    pub id: String,
    /// Always `"function"` — the only tool type `OpenAI` defines today.
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// The chosen function and its arguments.
    pub function: FunctionCall,
}

// ---- Compiled patterns (cached once) --------------------------------

fn tool_call_re() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    // (?s) = DOTALL: `.` matches newlines, so multi-line JSON inside the block
    // is captured. Non-greedy `.*?` stops at the first closing tag.
    RE.get_or_init(|| Regex::new(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>").ok())
        .as_ref()
}

fn mistral_call_re() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?s)^([A-Za-z_][A-Za-z0-9_]*)(\{.*\})\s*$").ok())
        .as_ref()
}

fn qwen_coder_fn_re() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"<function=([^>]+)>").ok())
        .as_ref()
}

fn qwen_coder_param_re() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?s)<parameter=([^>]+)>\s*(.*?)\s*</parameter>").ok())
        .as_ref()
}

fn fenced_block_re() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    // (?s) = DOTALL. Non-greedy body: fences don't nest, so the first closing
    // ``` after the opening one is always the right one. The language tag is
    // any run of word/`+`/`-` chars (not just "json") — a model wrapping a
    // call in ` ```bash ` or ` ```python ` is common (live-observed:
    // `qwen3-vl-8b-int8-ov` used ` ```bash ` for a `write_file` pythonic
    // call, 2026-07-22) and previously leaked straight into the captured
    // body, breaking anything downstream that expects the body to start
    // clean (e.g. the pythonic-call identifier check).
    RE.get_or_init(|| Regex::new(r"(?s)```[A-Za-z0-9_+-]*\s*\n?(.*?)```").ok())
        .as_ref()
}

fn think_closed_re() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?s)<think>(.*?)</think>").ok())
        .as_ref()
}

fn think_unclosed_re() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?s)<think>(.*)").ok())
        .as_ref()
}

// ---- Helpers --------------------------------------------------------

/// Char-boundary-safe truncation for log messages (never splits a UTF-8 char).
fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Byte offset in `s` for a `serde_json::Error`'s 1-based `(line, column)`.
/// `column` counts characters, not bytes, so within the target line we walk
/// `char_indices` rather than slicing by column directly.
fn line_col_to_byte_offset(s: &str, line: usize, column: usize) -> usize {
    let mut offset = 0;
    for (i, l) in s.split_inclusive('\n').enumerate() {
        if i + 1 == line {
            return offset
                + l.char_indices()
                    .nth(column.saturating_sub(1))
                    .map_or(l.len(), |(b, _)| b);
        }
        offset += l.len();
    }
    s.len()
}

/// A char-boundary-safe window of `radius` chars either side of `offset` in
/// `s`, marked with `…` where clipped. Centers a diagnostic log snippet on a
/// parse-error location instead of blindly truncating from the start, which
/// misses failures buried deep in a multi-KB tool-call argument blob.
fn context_window(s: &str, offset: usize, radius: usize) -> String {
    let offset = offset.min(s.len());
    let mut start = offset.saturating_sub(radius);
    while !s.is_char_boundary(start) {
        start -= 1;
    }
    let mut end = (offset + radius).min(s.len());
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    let prefix = if start > 0 { "…" } else { "" };
    let suffix = if end < s.len() { "…" } else { "" };
    format!("{prefix}{}{suffix}", &s[start..end])
}

/// Double any `\` in `raw` that isn't followed by one of JSON's own escape
/// characters (`"\/bfnrtu`), turning an invalid escape sequence into a
/// literal backslash. Never touches an already-valid escape, so it's a
/// no-op on well-formed JSON.
///
/// Fixes a real, repeatedly-observed failure mode (2026-07-22, live
/// incident — the project's internal engineering log): a model embedding generated code
/// (JS/CSS with a regex or template-literal backslash) inside a JSON
/// string `arguments` value without doubling the backslash for the JSON
/// wrapper. Without this, `serde_json::from_str` rejects the whole tool
/// call outright — the client (Hermes) got nothing usable back and kept
/// retrying the identical failing generation, pegging the GPU on an
/// ever-growing context each retry.
fn repair_invalid_backslash_escapes(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.peek() {
                Some('"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u') => out.push('\\'),
                _ => out.push_str(r"\\"),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Parse `raw` as JSON (object, array, whatever `T` expects), falling back
/// to a repaired copy ([`repair_invalid_backslash_escapes`]) if the first
/// attempt fails. Used by every tool-call parser that hard-fails the whole
/// call on invalid JSON — the repair pass is what stands between one
/// malformed backslash and a client stuck retrying a doomed generation
/// forever.
fn parse_json_lenient<T: serde::de::DeserializeOwned>(raw: &str) -> Result<T, serde_json::Error> {
    match serde_json::from_str::<T>(raw) {
        Ok(v) => Ok(v),
        Err(first_err) => match serde_json::from_str::<T>(&repair_invalid_backslash_escapes(raw)) {
            Ok(v) => {
                tracing::debug!(
                    "tool_call JSON had an invalid backslash escape — repaired and parsed \
                     successfully"
                );
                Ok(v)
            }
            Err(_) => Err(first_err),
        },
    }
}

/// Generate an `OpenAI`-style tool-call id: `"call_"` + 8 lowercase hex.
///
/// Deterministic in its inputs (uses the fixed-key `DefaultHasher`, not the
/// randomised `RandomState`), so the same call content yields the same id —
/// convenient for tests. `index` is mixed in so two *identical* calls in one
/// response still get distinct ids. Uniqueness only needs to hold within a
/// single response (the client correlates `tool_calls[i].id` with its `tool`
/// reply), which this satisfies.
fn gen_call_id(index: usize, name: &str, arguments: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    index.hash(&mut hasher);
    name.hash(&mut hasher);
    arguments.hash(&mut hasher);
    // Low 32 bits → 8 hex chars, matching stormVINO's uuid4().hex[:8] width.
    format!("call_{:08x}", hasher.finish() & 0xFFFF_FFFF)
}

// ---- Prompt injection (chat template) -------------------------------

/// A `serde_json` formatter that mirrors Python `json.dumps` default separators
/// — `", "` between elements and `": "` after keys, no newlines.
///
/// HF chat templates call a `tojson` that is `json.dumps(x, ensure_ascii=False)`,
/// so it emits spaced separators. minijinja's built-in `tojson` is compact
/// (`,`/`:`) and would not byte-match the prompt the model was trained on, so
/// [`build_prompt`] registers a `tojson` override built on this formatter.
struct PyJsonFormatter;

impl serde_json::ser::Formatter for PyJsonFormatter {
    fn begin_array_value<W>(&mut self, w: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first { Ok(()) } else { w.write_all(b", ") }
    }

    fn begin_object_key<W>(&mut self, w: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first { Ok(()) } else { w.write_all(b", ") }
    }

    fn begin_object_value<W>(&mut self, w: &mut W) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        w.write_all(b": ")
    }
}

/// `tojson` filter override producing Python-`json.dumps`-compatible output
/// (spaced separators, insertion-order keys, no Unicode/HTML escaping) so the
/// rendered prompt byte-matches HF `apply_chat_template`.
fn tojson_py(value: &JinjaValue) -> Result<JinjaValue, minijinja::Error> {
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, PyJsonFormatter);
    value.serialize(&mut ser).map_err(|e| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            "tojson serialization failed",
        )
        .with_source(e)
    })?;
    let s = String::from_utf8(buf).map_err(|e| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            "tojson produced invalid UTF-8",
        )
        .with_source(e)
    })?;
    Ok(JinjaValue::from(s))
}

/// Render the model's own Jinja chat template with the conversation and any
/// tool schemas — the prompt actually sent to the engine.
///
/// `messages` are passed to the template **verbatim** (OpenAI-standard: the
/// client's messages, including any `system` turn, govern; the server injects
/// nothing). `tools` (when `Some`) populate the template's `tools` variable,
/// which triggers its native tool-injection branch (e.g. Qwen3's `<tools>`
/// block). `add_generation_prompt` is always true (we want the assistant turn).
///
/// The template is compiled per call: compilation is microseconds against tens
/// of milliseconds of generation, so it is not worth caching yet.
///
/// CRASH COURSE — two non-obvious minijinja settings, both load-bearing:
///   * **autoescape OFF** — HF chat templates emit raw text; HTML-escaping would
///     turn `<tool_call>` into `&lt;tool_call&gt;` and corrupt every JSON arg.
///   * **pycompat method callback** — HF templates call Python string methods
///     (`.split`, `.strip`, `.startswith`, …) that minijinja core lacks;
///     `minijinja-contrib`'s `pycompat` supplies them.
///
/// # Errors
/// Returns an error if the template fails to compile or to render (e.g. a
/// message shape the template cannot handle) — the handler maps this to HTTP 400.
pub fn build_prompt<M: Serialize>(
    messages: &[M],
    tools: Option<&[serde_json::Value]>,
    template: &str,
    eos_token: &str,
    bos_token: &str,
    enable_thinking: Option<bool>,
    // Override for the model's built-in identity string. gpt-oss templates
    // expose this as `model_identity` and use it in the hardcoded system block
    // (replacing "You are ChatGPT…"). Other templates ignore it.
    model_identity: Option<&str>,
) -> anyhow::Result<String> {
    let mut env = Environment::new();
    env.set_auto_escape_callback(|_| AutoEscape::None);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    // Override `tojson` to match HF's Python-style output (see tojson_py).
    env.add_filter("tojson", tojson_py);
    // `from_json` filter: parse a JSON string back to a structured value.
    // Used by Qwen3-Coder template to unpack tool_call.arguments on round-trips.
    env.add_filter(
        "from_json",
        |value: &JinjaValue| -> Result<JinjaValue, minijinja::Error> {
            let s = value.as_str().ok_or_else(|| {
                minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    "from_json requires a string value",
                )
            })?;
            let parsed: serde_json::Value = serde_json::from_str(s).map_err(|e| {
                minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    format!("from_json: invalid JSON: {e}"),
                )
            })?;
            Ok(JinjaValue::from_serialize(parsed))
        },
    );
    // Mistral templates call strftime_now(fmt) to embed today's date in the
    // default system prompt. Standard Jinja2 doesn't have this — it's a
    // Mistral-specific extension. We supply it via chrono::Local::now().
    env.add_function("strftime_now", |fmt: String| {
        Local::now().format(&fmt).to_string()
    });
    // Some templates call raise_exception(msg) in error branches (e.g. when an
    // unsupported content block type is encountered). We wire it to a minijinja
    // error so the template render fails with a clear message rather than
    // "unknown function".
    env.add_function(
        "raise_exception",
        |msg: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    env.add_template("chat", template)
        .map_err(|e| anyhow::anyhow!("invalid chat template: {e}"))?;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| anyhow::anyhow!("chat template unavailable: {e}"))?;

    // When tools is None we pass UNDEFINED so templates that guard with
    // `{%- if not tools is defined %}` (e.g. Qwen3-Coder) correctly fall
    // into their "no tools" branch.  Passing serialised None would give a
    // defined-but-null value that breaks `tools | length` in those templates.
    let tools_val = tools.map_or(JinjaValue::UNDEFINED, JinjaValue::from_serialize);
    // `enable_thinking`: pass UNDEFINED when not set (template defaults to thinking
    // enabled). Pass `false` explicitly to trigger the Qwen3 template's
    // `<think>\n\n</think>` closed-block prefill, which suppresses thinking.
    // `true` is passed as defined-true (model decides, same as default for most templates).
    let thinking_val = enable_thinking.map_or(JinjaValue::UNDEFINED, JinjaValue::from);
    // `reasoning_effort`: gpt-oss uses this instead of `enable_thinking`. When
    // thinking is suppressed (`enable_thinking=false`), pass `"none"` so the
    // template pre-fills an empty analysis channel and the model skips reasoning.
    // For all other cases pass UNDEFINED so the template uses its own default.
    let reasoning_effort_val = match enable_thinking {
        Some(false) => JinjaValue::from("none"),
        _ => JinjaValue::UNDEFINED,
    };
    let model_identity_val = model_identity.map_or(JinjaValue::UNDEFINED, JinjaValue::from);
    let ctx = context! {
        messages => JinjaValue::from_serialize(messages),
        tools => tools_val,
        add_generation_prompt => true,
        eos_token => eos_token,
        bos_token => bos_token,
        enable_thinking => thinking_val,
        reasoning_effort => reasoning_effort_val,
        model_identity => model_identity_val,
    };
    tmpl.render(ctx)
        .map_err(|e| anyhow::anyhow!("chat template render failed: {e}"))
}

// ---- Tool-call parsing ----------------------------------------------

/// Extract tool calls from a model's raw output.
///
/// Returns `(Some(calls), remaining_text)` when one or more calls are found
/// (with the call markup stripped from `remaining_text`), or `(None, text)`
/// when there are none. Malformed individual calls are logged and skipped
/// rather than failing the whole parse — matching stormVINO.
///
/// Call [`extract_thinking`] first to remove `<think>` blocks; this function
/// does not look inside them.
#[must_use]
pub fn parse_tool_calls(text: &str, family: ModelFamily) -> (Option<Vec<ToolCall>>, String) {
    match family {
        ModelFamily::Default => parse_default(text),
        ModelFamily::Lfm2 => parse_lfm2(text),
        ModelFamily::Mistral => parse_mistral(text),
        ModelFamily::QwenCoder => parse_qwen_coder_xml(text),
        ModelFamily::GptOss => parse_gpt_oss_tool_calls(text),
    }
}

// ---- LFM2 (LiquidAI) ------------------------------------------------

/// LFM2 emits a **Python list of keyword calls**, normally delimited by
/// `<|tool_call_start|>` / `<|tool_call_end|>`:
///
/// ```text
/// <|tool_call_start|>[get_status(candidate_id="12345")]<|tool_call_end|>Checking now.
/// ```
///
/// Those delimiters are *special tokens* (ids 10/11) and `OpenVINO`'s detokenizer
/// strips them — verified live via `/detokenize`, where each decodes to the
/// empty string — so in practice only the bare `[...]` list survives into the
/// text this sees. The list is therefore bounded by **shape**, not by markers.
/// Markers are still stripped first when present, so this becomes a pure
/// upgrade if the bridge is ever changed to preserve them.
///
/// Bounding walks the text quote-aware (both `'` and `"`, with backslash
/// escapes) and bracket-depth-aware, which is precisely where a regex fails:
/// an argument value may legally contain `)`, `]` or `,`.
///
/// Trailing prose after the call is normal for this model and is preserved in
/// `remaining`, so it lands in `content` beside `tool_calls` — a legal `OpenAI`
/// shape.
fn parse_lfm2(text: &str) -> (Option<Vec<ToolCall>>, String) {
    // Markers are normally absent (stripped by the detokenizer) but are
    // authoritative bounds when present.
    let cleaned = text
        .replace("<|tool_call_start|>", "")
        .replace("<|tool_call_end|>", "");
    let scan: &str = if cleaned == text { text } else { &cleaned };

    let bytes = scan.as_bytes();
    let mut calls = Vec::new();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'[' {
            i += 1;
            continue;
        }
        let Some(end) = match_bracket(scan, i) else {
            i += 1;
            continue;
        };
        let inner = &scan[i + 1..end];
        if let Some(parsed) = parse_lfm2_call_list(inner, calls.len()) {
            calls.extend(parsed);
            spans.push((i, end + 1));
            i = end + 1;
        } else {
            i += 1;
        }
    }

    if calls.is_empty() {
        // The model card allows a JSON dialect when the system prompt asks for
        // one, so fall through to the shared fallbacks rather than giving up.
        let (json_calls, rest) = parse_fenced_json_fallback(scan);
        if json_calls.is_some() {
            return (json_calls, rest);
        }
        return parse_pythonic_call_fallback(scan);
    }
    (Some(calls), remove_spans(scan, &spans))
}

/// Index of the `]`/`)`/`}` closing the bracket that opens at `open`, skipping
/// over quoted strings so a bracket inside a string literal never counts.
fn match_bracket(s: &str, open: usize) -> Option<usize> {
    let b = s.as_bytes();
    let mut depth = 0i32;
    let mut quote: Option<u8> = None;
    let mut escaped = false;
    for (idx, &c) in b.iter().enumerate().skip(open) {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == q {
                quote = None;
            }
            continue;
        }
        match c {
            b'\'' | b'"' => quote = Some(c),
            b'[' | b'(' | b'{' => depth += 1,
            b']' | b')' | b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(idx);
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse the inside of a `[...]` as a comma-separated list of `name(k=v, …)`
/// calls. Returns `None` unless **every** element is a well-formed keyword
/// call — that all-or-nothing rule is what keeps ordinary prose containing
/// brackets (markdown links, citations, `[Note(s)]`) from being mistaken for a
/// tool call.
fn parse_lfm2_call_list(inner: &str, id_offset: usize) -> Option<Vec<ToolCall>> {
    let trimmed = inner.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut calls = Vec::new();
    for element in split_top_level(trimmed, b',') {
        let element = element.trim();
        let open = element.find('(')?;
        if !element.ends_with(')') {
            return None;
        }
        let name = element[..open].trim();
        if name.is_empty()
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
            || name.starts_with(|c: char| c.is_ascii_digit())
        {
            return None;
        }
        let args_src = &element[open + 1..element.len() - 1];
        let mut args = serde_json::Map::new();
        for arg in split_top_level(args_src.trim(), b',') {
            let arg = arg.trim();
            if arg.is_empty() {
                continue;
            }
            let Some(eq) = find_top_level_eq(arg) else {
                // Positional arguments cannot be mapped to parameter names
                // without the tool schema, which is not available here — skip
                // the whole call rather than guess at a binding.
                tracing::warn!(
                    call = %truncate(element, 100),
                    "LFM2 tool call has a positional argument — skipped (no schema to bind it to)"
                );
                return None;
            };
            let key = arg[..eq].trim();
            if key.is_empty() {
                return None;
            }
            args.insert(key.to_owned(), parse_py_literal(arg[eq + 1..].trim())?);
        }
        let arguments = serde_json::to_string(&serde_json::Value::Object(args)).ok()?;
        let idx = id_offset + calls.len();
        calls.push(ToolCall {
            id: gen_call_id(idx, name, &arguments),
            kind: "function",
            function: FunctionCall {
                name: name.to_owned(),
                arguments,
            },
        });
    }
    (!calls.is_empty()).then_some(calls)
}

/// Split on `sep` at bracket depth 0, outside string literals.
fn split_top_level(s: &str, sep: u8) -> Vec<&str> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut quote: Option<u8> = None;
    let mut escaped = false;
    let mut start = 0;
    for (i, &c) in b.iter().enumerate() {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == q {
                quote = None;
            }
            continue;
        }
        match c {
            b'\'' | b'"' => quote = Some(c),
            b'[' | b'(' | b'{' => depth += 1,
            b']' | b')' | b'}' => depth -= 1,
            _ if c == sep && depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    out
}

/// Offset of the `=` separating a keyword from its value, at depth 0 and
/// outside strings. Rejects `==`/`!=`/`<=`/`>=` so a comparison inside a
/// positional expression is not mistaken for a binding.
fn find_top_level_eq(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let mut depth = 0i32;
    let mut quote: Option<u8> = None;
    let mut escaped = false;
    for (i, &c) in b.iter().enumerate() {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == q {
                quote = None;
            }
            continue;
        }
        match c {
            b'\'' | b'"' => quote = Some(c),
            b'[' | b'(' | b'{' => depth += 1,
            b']' | b')' | b'}' => depth -= 1,
            b'=' if depth == 0 => {
                let prev = i.checked_sub(1).map(|p| b[p]);
                let next = b.get(i + 1).copied();
                if matches!(prev, Some(b'!' | b'<' | b'>' | b'=')) || next == Some(b'=') {
                    continue;
                }
                return Some(i);
            }
            _ => {}
        }
    }
    None
}

/// Read one Python/JSON literal into a [`serde_json::Value`].
///
/// Accepts both quote styles, Python's `True`/`False`/`None` **and** JSON's
/// `true`/`false`/`null` (the model's own template renders nested values with
/// `tojson`, so both dialects genuinely occur in one call), numbers, and
/// nested lists/dicts. Returns `None` for anything unrecognised, which fails
/// the whole call — deliberately conservative.
fn parse_py_literal(raw: &str) -> Option<serde_json::Value> {
    let s = raw.trim();
    match s {
        "True" | "true" => return Some(serde_json::Value::Bool(true)),
        "False" | "false" => return Some(serde_json::Value::Bool(false)),
        "None" | "null" => return Some(serde_json::Value::Null),
        "" => return None,
        _ => {}
    }
    let b = s.as_bytes();
    if (b[0] == b'\'' || b[0] == b'"') && b.len() >= 2 && b[b.len() - 1] == b[0] {
        return Some(serde_json::Value::String(unescape_py(&s[1..s.len() - 1])));
    }
    if b[0] == b'[' || b[0] == b'{' {
        // Normalise Python literals to JSON, then let serde do the parsing.
        // Single-quoted strings are re-quoted by the same walker used above,
        // so a `"` inside a `'...'` string survives.
        let json = pythonish_to_json(s)?;
        return serde_json::from_str(&json).ok();
    }
    s.parse::<i64>()
        .ok()
        .map(|n| serde_json::Value::Number(n.into()))
        .or_else(|| {
            s.parse::<f64>()
                .ok()
                .and_then(serde_json::Number::from_f64)
                .map(serde_json::Value::Number)
        })
}

/// Python string-escape handling for the subset the model's own template can
/// emit (`\\`, `\'`, `\"`, `\n`, `\r`, `\t`).
fn unescape_py(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some(other) => out.push(other),
            None => out.push('\\'),
        }
    }
    out
}

/// Rewrite a Python-ish container literal as JSON: single-quoted strings become
/// double-quoted, and the bare words `True`/`False`/`None` become their JSON
/// spellings. Content inside strings is left alone.
fn pythonish_to_json(src: &str) -> Option<String> {
    let mut out = String::with_capacity(src.len() + 8);
    let bytes = src.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\'' | b'"' => {
                let quote = bytes[i];
                let mut j = i + 1;
                let mut escaped = false;
                while j < bytes.len() {
                    if escaped {
                        escaped = false;
                    } else if bytes[j] == b'\\' {
                        escaped = true;
                    } else if bytes[j] == quote {
                        break;
                    }
                    j += 1;
                }
                if j >= bytes.len() {
                    return None; // unterminated string
                }
                let body = unescape_py(src.get(i + 1..j)?);
                out.push_str(&serde_json::to_string(&body).ok()?);
                i = j + 1;
            }
            _ => {
                let mut replaced = false;
                for (word, json) in [("True", "true"), ("False", "false"), ("None", "null")] {
                    // Only a whole word, so `Nonetheless` is not rewritten.
                    let boundary_ok = src.get(i + word.len()..).is_none_or(|rest| {
                        rest.chars()
                            .next()
                            .is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_')
                    });
                    if src[i..].starts_with(word) && boundary_ok {
                        out.push_str(json);
                        i += word.len();
                        replaced = true;
                        break;
                    }
                }
                if !replaced {
                    let ch = src.get(i..)?.chars().next()?;
                    out.push(ch);
                    i += ch.len_utf8();
                }
            }
        }
    }
    Some(out)
}

/// Qwen/Phi/Llama/Gemma: one or more `<tool_call>{json}</tool_call>` blocks.
fn parse_default(text: &str) -> (Option<Vec<ToolCall>>, String) {
    let Some(re) = tool_call_re() else {
        return (None, text.trim().to_owned());
    };

    let mut calls = Vec::new();
    for (idx, cap) in re.captures_iter(text).enumerate() {
        let raw = &cap[1];
        match parse_json_lenient::<serde_json::Value>(raw) {
            Ok(value) => {
                let Some(name) = value.get("name").and_then(serde_json::Value::as_str) else {
                    tracing::warn!(raw = %truncate(raw, 100), "tool_call JSON missing string 'name'");
                    continue;
                };
                // `arguments` defaults to {} when absent; re-serialise to a
                // compact JSON string (the OpenAI wire format for arguments).
                let args = value
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                let arguments = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_owned());
                calls.push(ToolCall {
                    id: gen_call_id(idx, name, &arguments),
                    kind: "function",
                    function: FunctionCall {
                        name: name.to_owned(),
                        arguments,
                    },
                });
            }
            Err(e) => {
                let offset = line_col_to_byte_offset(raw, e.line(), e.column());
                tracing::warn!(
                    error = %e,
                    len = raw.len(),
                    context = %context_window(raw, offset, 160),
                    "failed to parse tool_call JSON (repair attempt also failed)"
                );
                tracing::debug!(raw = %raw, "full unparseable tool_call blob");
            }
        }
    }

    if calls.is_empty() {
        let (json_calls, json_remaining) = parse_fenced_json_fallback(text);
        if json_calls.is_some() {
            return (json_calls, json_remaining);
        }
        return parse_pythonic_call_fallback(text);
    }
    let remaining = re.replace_all(text, "").trim().to_owned();
    (Some(calls), remaining)
}

/// Lenient fallback for `Default`-family models that ignore the `<tool_call>`
/// delimiter instruction and instead emit the same `{"name":…, "arguments":…}`
/// shape wrapped in a markdown code fence, or bare with no wrapper at all.
/// Only reached when no `<tool_call>` tags matched.
///
/// Observed with `qwen2.5-coder-14b-int4`: the rendered prompt correctly
/// instructs `<tool_call>` delimiters (verified by direct template render),
/// but the model's code-completion training biases it toward wrapping JSON
/// in a ` ```json ` fence instead — the call content is right, the wrapper
/// isn't. Deliberately conservative: a candidate object must carry a string
/// `name` plus an object `arguments`/`parameters` key before it's accepted as
/// a call, so ordinary JSON-in-prose (a common thing to see in real answers)
/// doesn't get misread as a tool call.
fn parse_fenced_json_fallback(text: &str) -> (Option<Vec<ToolCall>>, String) {
    let trimmed = text.trim();

    if let Some(re) = fenced_block_re() {
        let mut calls = Vec::new();
        let mut spans = Vec::new();
        for (idx, cap) in re.captures_iter(trimmed).enumerate() {
            let whole = cap.get(0).map_or((0, 0), |m| (m.start(), m.end()));
            let Some(obj) = cap.get(1).and_then(|b| balanced_json_object(b.as_str())) else {
                continue;
            };
            let Some(call) = try_tool_call_from_json(obj, idx) else {
                continue;
            };
            calls.push(call);
            spans.push(whole);
        }
        if !calls.is_empty() {
            return (Some(calls), remove_spans(trimmed, &spans));
        }
    }

    // No fence, or none of the fenced bodies were call-shaped: maybe the
    // entire response IS the bare object.
    if let Some(call) = try_tool_call_from_json(trimmed, 0) {
        return (Some(vec![call]), String::new());
    }

    (None, trimmed.to_owned())
}

/// Locate the first balanced `{...}` span in `text`. Mirrors
/// `parse_mistral_v3`'s brace-depth walk — doesn't account for `{`/`}` inside
/// string literals, which tool-call argument values essentially never carry.
fn balanced_json_object(text: &str) -> Option<&str> {
    let open = text.find('{')?;
    let mut depth = 0i32;
    for (i, ch) in text[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[open..open + i + ch.len_utf8()]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Remove the given byte spans (in `re.captures_iter` order — ascending,
/// non-overlapping since matches can't overlap) from `text`.
fn remove_spans(text: &str, spans: &[(usize, usize)]) -> String {
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0;
    for &(start, end) in spans {
        out.push_str(&text[cursor..start]);
        cursor = end;
    }
    out.push_str(&text[cursor..]);
    out.trim().to_owned()
}

/// Parse `raw` as JSON and, if shaped like a tool call (string `name`, object
/// `arguments`/`parameters`), build a `ToolCall`. Returns `None` for anything
/// else — most non-matches here are ordinary JSON or prose, not malformed
/// calls, so unlike the primary `<tool_call>` parser this doesn't log.
fn try_tool_call_from_json(raw: &str, idx: usize) -> Option<ToolCall> {
    let value: serde_json::Value = parse_json_lenient(raw).ok()?;
    let name = value.get("name")?.as_str()?;
    let args = value
        .get("arguments")
        .or_else(|| value.get("parameters"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    if !args.is_object() {
        return None;
    }
    let arguments = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_owned());
    Some(ToolCall {
        id: gen_call_id(idx, name, &arguments),
        kind: "function",
        function: FunctionCall {
            name: name.to_owned(),
            arguments,
        },
    })
}

/// Bash/pythonic fallback: `function_name(key="value", key2="value2")`,
/// optionally inside a fence. Reached only when neither the primary
/// `<tool_call>` parser nor the fenced/bare-JSON fallback matched.
///
/// Observed live (2026-07-22, real Hermes traffic — the project's internal engineering log):
/// `qwen3-vl-8b-int8-ov` emitted `write_file(path="...", content="...")` — a
/// keyword-call expression, neither `<tool_call>` XML nor a JSON object —
/// for a large-content call. Deliberately conservative like the JSON
/// fallback: only tried against a whole fenced body or the whole trimmed
/// response, never a substring found mid-prose (a bare `name(...)` inside an
/// explanatory sentence is rejected by the identifier check in
/// [`try_parse_pythonic_call`] — a name containing spaces or punctuation
/// from surrounding prose fails the alphanumeric/underscore-only test).
fn parse_pythonic_call_fallback(text: &str) -> (Option<Vec<ToolCall>>, String) {
    let trimmed = text.trim();

    if let Some(cap) = fenced_block_re().and_then(|re| re.captures(trimmed)) {
        let body = cap[1].trim();
        if let Some(call) = try_parse_pythonic_call(body) {
            let whole = cap.get(0).map_or((0, 0), |m| (m.start(), m.end()));
            let remaining = remove_spans(trimmed, &[whole]);
            return (Some(vec![call]), remaining);
        }
    }

    if let Some(call) = try_parse_pythonic_call(trimmed) {
        return (Some(vec![call]), String::new());
    }

    (None, trimmed.to_owned())
}

/// Parse a single `identifier(key="value", key2=bareToken, ...)` expression.
///
/// Python/JS string escaping (`\"`, `\\`, `\n`, `\t`, `\r`, ...) is a strict
/// subset of JSON's, so once a quoted value's boundaries are found — walked
/// char-by-char, honouring `\` only so an escaped quote doesn't end the
/// string early — the raw captured text (escapes untouched) drops straight
/// into a hand-built JSON string and parses correctly with no
/// re-interpretation needed. A bare (unquoted) value is passed through
/// as-is, letting [`parse_json_lenient`] accept plain JSON literals
/// (`true`, `false`, `123`) same as a real JSON object would.
fn try_parse_pythonic_call(text: &str) -> Option<ToolCall> {
    use std::fmt::Write as _;
    let text = text.trim();
    let open_byte = text.find('(')?;
    let name = text[..open_byte].trim();
    let first = name.chars().next()?;
    if !(first.is_ascii_alphabetic() || first == '_')
        || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return None;
    }

    let after_open = &text[open_byte + 1..];
    let close_byte = after_open.rfind(')')?;
    let args_src = &after_open[..close_byte];

    let cs: Vec<char> = args_src.chars().collect();
    let n = cs.len();
    let mut i = 0usize;
    let mut fields: Vec<(String, String)> = Vec::new();

    while i < n {
        while i < n && (cs[i].is_whitespace() || cs[i] == ',') {
            i += 1;
        }
        if i >= n {
            break;
        }
        let key_start = i;
        while i < n && (cs[i].is_ascii_alphanumeric() || cs[i] == '_') {
            i += 1;
        }
        if i == key_start {
            return None;
        }
        let key: String = cs[key_start..i].iter().collect();
        while i < n && cs[i].is_whitespace() {
            i += 1;
        }
        if i >= n || cs[i] != '=' {
            return None;
        }
        i += 1;
        while i < n && cs[i].is_whitespace() {
            i += 1;
        }
        if i >= n {
            return None;
        }
        let value_json = if cs[i] == '"' {
            i += 1;
            let val_start = i;
            let mut closed = false;
            while i < n {
                if cs[i] == '\\' && i + 1 < n {
                    i += 2;
                    continue;
                }
                if cs[i] == '"' {
                    closed = true;
                    break;
                }
                i += 1;
            }
            if !closed {
                return None;
            }
            let raw: String = cs[val_start..i].iter().collect();
            i += 1;
            format!("\"{raw}\"")
        } else {
            let bare_start = i;
            while i < n && cs[i] != ',' {
                i += 1;
            }
            let bare: String = cs[bare_start..i].iter().collect();
            bare.trim().to_owned()
        };
        fields.push((key, value_json));
    }

    if fields.is_empty() {
        return None;
    }

    let mut json = String::from("{");
    for (idx, (k, v)) in fields.iter().enumerate() {
        if idx > 0 {
            json.push(',');
        }
        let _ = write!(json, "\"{k}\":{v}");
    }
    json.push('}');

    let args_value: serde_json::Value = parse_json_lenient(&json).ok()?;
    let arguments = serde_json::to_string(&args_value).unwrap_or_else(|_| "{}".to_owned());
    Some(ToolCall {
        id: gen_call_id(0, name, &arguments),
        kind: "function",
        function: FunctionCall {
            name: name.to_owned(),
            arguments,
        },
    })
}

/// Mistral v3+: `[TOOL_CALLS] [{"name":…, "arguments":{…}, "id":"…"}, …]`
///
/// The model emits `arguments` as a JSON **object** (not a string), and
/// includes a native `id`. If the id is absent we fall back to `gen_call_id`.
fn parse_mistral_v3(text: &str) -> Option<Vec<ToolCall>> {
    let marker = "[TOOL_CALLS]";
    let pos = text.find(marker)?;
    let after = text[pos + marker.len()..].trim();
    if !after.starts_with('[') {
        return None;
    }
    // Find the end of the JSON array. We walk past the opening '['.
    let arr_json = {
        let mut depth = 0i32;
        let mut end = 0usize;
        for (i, ch) in after.char_indices() {
            match ch {
                '[' | '{' => depth += 1,
                ']' | '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        if end == 0 {
            return None;
        }
        &after[..end]
    };

    let entries = parse_json_lenient::<Vec<serde_json::Value>>(arr_json).ok()?;
    let mut calls = Vec::with_capacity(entries.len());
    for (idx, entry) in entries.iter().enumerate() {
        let name = entry.get("name").and_then(serde_json::Value::as_str)?;
        let args_val = entry
            .get("arguments")
            .cloned()
            .unwrap_or(serde_json::json!({}));
        let arguments = serde_json::to_string(&args_val).unwrap_or_else(|_| "{}".to_owned());
        let id = entry
            .get("id")
            .and_then(serde_json::Value::as_str)
            .map_or_else(|| gen_call_id(idx, name, &arguments), str::to_owned);
        calls.push(ToolCall {
            id,
            kind: "function",
            function: FunctionCall {
                name: name.to_owned(),
                arguments,
            },
        });
    }
    if calls.is_empty() { None } else { Some(calls) }
}

/// Mistral: try v3 `[TOOL_CALLS]` format first, fall back to bare `name{json}`.
fn parse_mistral(text: &str) -> (Option<Vec<ToolCall>>, String) {
    if let Some(calls) = parse_mistral_v3(text) {
        return (Some(calls), String::new());
    }

    let trimmed = text.trim();
    let Some(re) = mistral_call_re() else {
        return (None, trimmed.to_owned());
    };
    let Some(cap) = re.captures(trimmed) else {
        return (None, trimmed.to_owned());
    };

    let name = &cap[1];
    let args_str = &cap[2];
    match parse_json_lenient::<serde_json::Value>(args_str) {
        Ok(value) => {
            let arguments = serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_owned());
            let call = ToolCall {
                id: gen_call_id(0, name, &arguments),
                kind: "function",
                function: FunctionCall {
                    name: name.to_owned(),
                    arguments,
                },
            };
            (Some(vec![call]), String::new())
        }
        Err(e) => {
            tracing::warn!(error = %e, "failed to parse Mistral tool_call args");
            (None, trimmed.to_owned())
        }
    }
}

/// Qwen3-Coder XML dialect: `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`
///
/// Arguments are extracted as key/value pairs; each value is attempted as JSON
/// first (numbers, booleans, nested objects) and falls back to a plain string.
/// Multiple sequential `<tool_call>` blocks are each parsed as a separate call.
fn parse_qwen_coder_xml(text: &str) -> (Option<Vec<ToolCall>>, String) {
    let (Some(outer_re), Some(fn_re), Some(param_re)) =
        (tool_call_re(), qwen_coder_fn_re(), qwen_coder_param_re())
    else {
        return (None, text.trim().to_owned());
    };

    let mut calls = Vec::new();
    for (idx, cap) in outer_re.captures_iter(text).enumerate() {
        let block = &cap[1];
        let Some(fn_cap) = fn_re.captures(block) else {
            tracing::warn!(
                block = %truncate(block, 100),
                "QwenCoder: no <function=...> in tool_call block"
            );
            continue;
        };
        let name = &fn_cap[1];
        let mut args = serde_json::Map::new();
        for param_cap in param_re.captures_iter(block) {
            let key = param_cap[1].trim().to_owned();
            let raw_val = param_cap[2].trim();
            // Try to parse as JSON scalar/object; fall back to bare string.
            let json_val = serde_json::from_str::<serde_json::Value>(raw_val)
                .unwrap_or_else(|_| serde_json::Value::String(raw_val.to_owned()));
            args.insert(key, json_val);
        }
        let arguments = serde_json::to_string(&serde_json::Value::Object(args))
            .unwrap_or_else(|_| "{}".to_owned());
        calls.push(ToolCall {
            id: gen_call_id(idx, name, &arguments),
            kind: "function",
            function: FunctionCall {
                name: name.to_owned(),
                arguments,
            },
        });
    }

    if calls.is_empty() {
        return (None, text.trim().to_owned());
    }
    let remaining = outer_re.replace_all(text, "").trim().to_owned();
    (Some(calls), remaining)
}

fn gpt_oss_tool_call_re() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    // gpt-oss's harmony tool-call header, as it survives `HarmonyFilter`
    // (every true special token already stripped): "to=functions.NAME" then
    // a content-type word (almost always "json") glued directly onto the
    // JSON arguments object with no separator — the residue of the harmony
    // spec's `<|message|>` token. The JSON body itself is matched separately
    // via `balanced_json_object` (nested objects need brace-depth tracking,
    // not a regex).
    RE.get_or_init(|| Regex::new(r"to=functions\.([A-Za-z_][A-Za-z0-9_.]*)\s+[A-Za-z]*").ok())
        .as_ref()
}

/// gpt-oss's harmony tool-call dialect: zero or more
/// `to=functions.NAME <content_type>{ARGS}` segments. Unlike every other
/// family this server parses, the function name lives in the `to=` header,
/// *outside* the JSON — the JSON body holds only the arguments, no `"name"`
/// key — so `try_tool_call_from_json`'s "candidate object must carry a
/// string `name`" heuristic (used by the other families' lenient fallbacks)
/// cannot recognize this shape at all; it needs its own parser.
///
/// Meant to run on already-`HarmonyFilter`-processed text (the `commentary`
/// channel content that ends up in `reasoning_content`, not `content` — see
/// `extract_reasoning_gpt_oss`), not raw engine output.
fn parse_gpt_oss_tool_calls(text: &str) -> (Option<Vec<ToolCall>>, String) {
    let Some(re) = gpt_oss_tool_call_re() else {
        return (None, text.trim().to_owned());
    };

    let mut calls = Vec::new();
    let mut spans = Vec::new();
    for (idx, cap) in re.captures_iter(text).enumerate() {
        let Some(whole) = cap.get(0) else { continue };
        let name = &cap[1];
        let Some(obj) = balanced_json_object(&text[whole.end()..]) else {
            tracing::warn!(
                name,
                "gpt-oss tool call header with no following JSON object"
            );
            continue;
        };
        match parse_json_lenient::<serde_json::Value>(obj) {
            Ok(value) => {
                let arguments = serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_owned());
                calls.push(ToolCall {
                    id: gen_call_id(idx, name, &arguments),
                    kind: "function",
                    function: FunctionCall {
                        name: name.to_owned(),
                        arguments,
                    },
                });
                spans.push((whole.start(), whole.end() + obj.len()));
            }
            Err(e) => {
                tracing::warn!(
                    name,
                    error = %e,
                    raw = %truncate(obj, 160),
                    "failed to parse gpt-oss tool call arguments"
                );
            }
        }
    }

    if calls.is_empty() {
        return (None, text.trim().to_owned());
    }
    (Some(calls), remove_spans(text, &spans))
}

// ---- Thinking extraction --------------------------------------------

/// Strip Qwen-style `<think>…</think>` reasoning from raw output.
///
/// Returns `(thinking, answer)`. A closed block yields its inner text as
/// `thinking` and the block removed from `answer`. An *unclosed* `<think>`
/// (the model hit `max_tokens` mid-thought) returns the partial thinking and
/// the text *before* the tag as the answer — with a placeholder if that is
/// empty. No `<think>` at all → `(None, trimmed_text)`.
///
/// A *dangling close* — a `</think>` with no matching opening `<think>` — is
/// the Qwen3 thinking-mode shape: the chat template prefills the opening
/// `<think>` into the **prompt**, so the model's output begins mid-thought and
/// emits only the closing tag. Everything up to the first `</think>` is the
/// reasoning; the remainder is the answer.
#[must_use]
pub fn extract_thinking(text: &str, starts_in_thinking: bool) -> (Option<String>, String) {
    // Shape (d): the prompt prefilled an open `<think>` and the
    // model never emitted the closing tag (token budget cut it mid-thought).
    // Without this branch the text has no tags at all and would fall through
    // to the no-op arm below, leaking the entire reasoning into `content`.
    // Checked FIRST: when the output genuinely starts inside thinking, any
    // `<think>` the model re-opens later is still part of the reasoning, so
    // the pair/unclosed branches below must not carve it up.
    if starts_in_thinking && !text.contains(CLOSE_TAG) {
        tracing::warn!(
            chars = text.len(),
            "prefilled <think> never closed — model likely hit max_tokens mid-thought"
        );
        return (
            Some(text.trim().to_owned()),
            "*(thinking was cut off by max_tokens limit)*".to_owned(),
        );
    }

    if let Some(re) = think_closed_re()
        && let Some(cap) = re.captures(text)
    {
        let thinking = cap[1].trim().to_owned();
        let answer = re.replace_all(text, "").trim().to_owned();
        return (Some(thinking), answer);
    }

    // Dangling close: the matched-pair regex failed, so any `</think>` here has
    // no opening tag in the output (it was prefilled into the prompt). Treat the
    // text before it as reasoning. Checked BEFORE the unclosed-`<think>` branch,
    // which requires an opening tag and so cannot match this shape anyway.
    if let Some(idx) = text.find(CLOSE_TAG) {
        let thinking = text[..idx].trim().to_owned();
        let answer = text[idx + CLOSE_TAG.len()..].trim().to_owned();
        return (Some(thinking), answer);
    }

    if let Some(re) = think_unclosed_re()
        && let Some(cap) = re.captures(text)
    {
        let start = cap.get(0).map_or(0, |m| m.start());
        let thinking = cap[1].trim().to_owned();
        let prefix = text[..start].trim();
        let answer = if prefix.is_empty() {
            "*(thinking was cut off by max_tokens limit)*".to_owned()
        } else {
            prefix.to_owned()
        };
        tracing::warn!(
            chars = thinking.len(),
            "unclosed <think> block — model likely hit max_tokens mid-thought"
        );
        return (Some(thinking), answer);
    }

    (None, text.trim().to_owned())
}

/// Does this chat template **prefill an open `<think>`** into the assistant
/// generation prompt? (Qwen3 thinking mode.)
///
/// Such a template puts the opening `<think>` in the prompt, so the model's
/// output starts mid-thought and emits only the closing `</think>`. The
/// streaming [`ThinkFilter`] must therefore *start* in thinking mode — by the
/// time the close arrives, the reasoning tokens have already streamed.
///
/// Detection: remove every matched `<think>…</think>` pair (the explicit
/// *non*-thinking prefill `<think>\n\n</think>`). If a bare `<think>` survives,
/// the default (thinking-enabled) branch prefills an unclosed tag → `true`.
/// Templates with no think tags, or only closed ones, → `false`.
#[must_use]
pub fn template_prefills_thinking(template: &str) -> bool {
    match think_closed_re() {
        Some(re) => re.replace_all(template, "").contains(OPEN_TAG),
        None => template.contains(OPEN_TAG),
    }
}

/// Does this **rendered prompt** end with an open `<think>` — i.e. will the
/// model's output start inside a thinking block?
///
/// Per-request ground truth, unlike [`template_prefills_thinking`]'s static
/// template scan: a request with `enable_thinking: false` renders a
/// *closed* `<think>\n\n</think>` prefill, so the output starts OUTSIDE
/// thinking even though the template source contains a bare `<think>` in its
/// other branch. Use this wherever the rendered prompt exists (CB/NPU paths);
/// the VLM path renders inside `OpenVINO` `GenAI` and must keep the static check.
#[must_use]
pub fn prompt_ends_in_thinking(prompt: &str) -> bool {
    prompt.trim_end().ends_with(OPEN_TAG)
}

// ---- Streaming think filter -----------------------------------------

const OPEN_TAG: &str = "<think>";
const CLOSE_TAG: &str = "</think>";

/// One filtered output piece from the streaming [`ThinkFilter`].
#[derive(Debug, PartialEq, Eq)]
pub enum ThinkPiece {
    /// Visible answer text — goes to `delta.content` in SSE.
    Content(String),
    /// Reasoning text from inside `<think>…</think>` — goes to
    /// `delta.reasoning_content` in SSE.
    Reasoning(String),
}

#[derive(Debug, Clone, Copy)]
enum ThinkMode {
    Normal,
    Thinking,
}

/// Stateful streaming filter that strips `<think>…</think>` blocks from a
/// token-by-token feed, handling tags that split across token boundaries.
///
/// Feed each engine token via [`process`](Self::process); call
/// [`flush`](Self::flush) once the token channel closes.
pub struct ThinkFilter {
    mode: ThinkMode,
    /// Bytes held back because they might be the start of a tag.
    pending: String,
}

impl Default for ThinkFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl ThinkFilter {
    /// Create a fresh filter in `Normal` (content-passthrough) mode.
    #[must_use]
    pub fn new() -> Self {
        Self::starting_in_thinking(false)
    }

    /// Create a filter whose initial mode is chosen by `yes`.
    ///
    /// Pass `true` when the model's chat template prefills an open `<think>`
    /// (see [`template_prefills_thinking`]): the output begins mid-thought with
    /// no opening tag, so the filter must start in `Thinking` mode to route the
    /// leading reasoning to `reasoning_content` and drop the trailing
    /// `</think>` rather than leaking it into `content`.
    #[must_use]
    pub fn starting_in_thinking(yes: bool) -> Self {
        Self {
            mode: if yes {
                ThinkMode::Thinking
            } else {
                ThinkMode::Normal
            },
            pending: String::new(),
        }
    }

    /// Feed one raw token. Returns the content/reasoning pieces safe to emit.
    /// Empty strings are never included in the output.
    pub fn process(&mut self, tok: &str) -> Vec<ThinkPiece> {
        self.pending.push_str(tok);
        let mut out = Vec::new();
        loop {
            match self.mode {
                ThinkMode::Normal => {
                    if let Some(idx) = self.pending.find(OPEN_TAG) {
                        let before = self.pending[..idx].to_owned();
                        let rest = self.pending[idx + OPEN_TAG.len()..].to_owned();
                        if !before.is_empty() {
                            out.push(ThinkPiece::Content(before));
                        }
                        self.pending = rest;
                        self.mode = ThinkMode::Thinking;
                        // Continue loop: process `rest` in Thinking mode.
                    } else {
                        let hold = max_tag_overlap(&self.pending, OPEN_TAG);
                        let safe_end = self.pending.len() - hold;
                        if safe_end > 0 {
                            out.push(ThinkPiece::Content(self.pending[..safe_end].to_owned()));
                        }
                        self.pending = self.pending[safe_end..].to_owned();
                        break;
                    }
                }
                ThinkMode::Thinking => {
                    if let Some(idx) = self.pending.find(CLOSE_TAG) {
                        let before = self.pending[..idx].to_owned();
                        let rest = self.pending[idx + CLOSE_TAG.len()..].to_owned();
                        if !before.is_empty() {
                            out.push(ThinkPiece::Reasoning(before));
                        }
                        self.pending = rest;
                        self.mode = ThinkMode::Normal;
                        // Continue loop: process `rest` in Normal mode.
                    } else {
                        let hold = max_tag_overlap(&self.pending, CLOSE_TAG);
                        let safe_end = self.pending.len() - hold;
                        if safe_end > 0 {
                            out.push(ThinkPiece::Reasoning(self.pending[..safe_end].to_owned()));
                        }
                        self.pending = self.pending[safe_end..].to_owned();
                        break;
                    }
                }
            }
        }
        out
    }

    /// Flush any remaining buffered bytes when the token stream ends.
    ///
    /// The pending buffer holds at most `tag.len() - 1` bytes — a partial tag
    /// prefix that was never completed. Emit it in the current mode.
    pub fn flush(&mut self) -> Vec<ThinkPiece> {
        if self.pending.is_empty() {
            return vec![];
        }
        let s = std::mem::take(&mut self.pending);
        match self.mode {
            ThinkMode::Normal => vec![ThinkPiece::Content(s)],
            ThinkMode::Thinking => vec![ThinkPiece::Reasoning(s)],
        }
    }
}

/// Returns the length of the longest suffix of `s` that is also a prefix of
/// `tag`. At most `tag.len() - 1` (a full match would have been found by
/// `str::find` already). Used to decide how many bytes to hold back.
fn max_tag_overlap(s: &str, tag: &str) -> usize {
    // tag.len().saturating_sub(1) avoids underflow on a zero-length tag
    // (impossible with our constants, but safe).
    let max_len = s.len().min(tag.len().saturating_sub(1));
    for i in (1..=max_len).rev() {
        if s.ends_with(&tag[..i]) {
            return i;
        }
    }
    0
}

// ---- gpt-oss harmony channel handling --------------------------------
//
// gpt-oss doesn't use `<think>…</think>` — its reasoning is structured into
// "channels" (`analysis`, `final`, `commentary`) delimited by true special
// tokens (`<|channel|>`, `<|message|>`, `<|start|>`, `<|end|>`). OpenVINO's
// tokenizer strips those special tokens by default, so only the channel-name
// *word* (an ordinary vocabulary token) and, at a mid-stream transition, the
// "assistant" role word survive as plain text. Confirmed live against
// `gpt-oss-20b-int4-ov` (2026-07-27, non-streaming): a thinking-enabled
// response decodes verbatim as
// `"analysisThe user asks...Ensure minimal output.\n\nassistantfinal4"` —
// bare "analysis" at the very start, then the glued "assistantfinal" at the
// analysis→final boundary, zero separator either time. Streaming confirmed
// the same content splits across ordinary token-sized SSE deltas
// (`"assistant"` and `"final"` as two separate events), not one atomic chunk.

/// gpt-oss channel names that can appear as literal plain text at a channel
/// boundary.
const GPT_OSS_CHANNEL_NAMES: &[&str] = &["final", "analysis", "commentary"];

/// Longest channel name ("commentary") — the leading-word detector buffers
/// this many bytes before concluding no channel word is present.
const MAX_CHANNEL_NAME_LEN: usize = 10;

/// If `text` starts with a gpt-oss channel name, word-boundary-guarded,
/// return the matched name.
///
/// Case-insensitive; matches only when the character after the name is
/// non-alphabetic or uppercase, so "finally" is not mistaken for a leaked
/// "final" but "finalI'm" and "final answer" both match. Same rule (and the
/// same accepted trade-off — a genuine answer opening with "Final ...",
/// "Analysis ...", or "Commentary ..." as its very first word is misread as
/// a channel leak) as `realtime.rs::strip_gpt_oss_channel_prefix`, which
/// proved it out in production first; not shared code (different call
/// shape), but deliberately the same rule.
fn match_leading_channel_word(text: &str) -> Option<&'static str> {
    let lower = text.to_ascii_lowercase();
    GPT_OSS_CHANNEL_NAMES.iter().copied().find(|&name| {
        lower.starts_with(name)
            && text[name.len()..]
                .chars()
                .next()
                .is_none_or(|c| !c.is_alphabetic() || c.is_uppercase())
    })
}

/// Which visible destination a gpt-oss channel's content routes to.
/// `analysis` and `commentary` (gpt-oss's tool-call channel — its
/// `to=functions...` syntax isn't parsed into a structured tool call yet,
/// tracked separately) both stay out of `content` for now by routing to
/// `Reasoning`.
fn harmony_mode_for(channel_name: &str) -> HarmonyMode {
    if channel_name == "final" {
        HarmonyMode::Content
    } else {
        HarmonyMode::Reasoning
    }
}

/// Find the first mid-stream harmony channel transition in `text`: the
/// literal, zero-separator sequence "assistant" + a channel name — the
/// residue of `<|end|><|start|>assistant<|channel|>NAME<|message|>` once
/// `OpenVINO` strips every true special token.
///
/// No trailing word-boundary check is needed here (unlike the leading-word
/// case): "assistant" is never organically followed with zero separator by
/// "final"/"analysis"/"commentary" in real English, so the compound glue
/// alone is the safety margin — reasoning prose that legitimately contains
/// those words (e.g. "in the final analysis") scans clean because they
/// aren't glued to a preceding "assistant".
///
/// Returns `(byte_index_of_match_start, marker_byte_len, mode_to_enter)`.
fn find_harmony_transition(text: &str) -> Option<(usize, usize, HarmonyMode)> {
    let lower = text.to_ascii_lowercase();
    GPT_OSS_CHANNEL_NAMES
        .iter()
        .filter_map(|&name| {
            let marker = format!("assistant{name}");
            lower
                .find(&marker)
                .map(|idx| (idx, marker.len(), harmony_mode_for(name)))
        })
        .min_by_key(|&(idx, _, _)| idx)
}

/// Longest possible partial-match holdback across all three transition
/// markers ("assistant" + channel name), so the streaming filter never emits
/// text that might still turn out to be the start of a transition once more
/// tokens arrive.
fn max_harmony_overlap(s: &str) -> usize {
    GPT_OSS_CHANNEL_NAMES
        .iter()
        .map(|name| max_tag_overlap(s, &format!("assistant{name}")))
        .max()
        .unwrap_or(0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HarmonyMode {
    /// Buffering the first bytes of the turn to detect a leading bare
    /// channel-name word (gpt-oss sometimes re-declares its own starting
    /// channel as plain text despite a prompt-side channel hint).
    AwaitingLeadChannel,
    /// Emitting reasoning text (`analysis`/`commentary` channel content) —
    /// routed to `reasoning_content`.
    Reasoning,
    /// Emitting the visible answer (`final` channel) — routed to `content`.
    Content,
}

/// Stateful streaming filter for gpt-oss's harmony channel format — the
/// counterpart to [`ThinkFilter`] for models whose reasoning isn't wrapped in
/// `<think>…</think>` tags. See [`extract_reasoning_gpt_oss`] for the
/// buffered/non-streaming equivalent, built on this same filter so the two
/// paths cannot drift apart.
pub struct HarmonyFilter {
    mode: HarmonyMode,
    pending: String,
}

impl Default for HarmonyFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl HarmonyFilter {
    #[must_use]
    pub fn new() -> Self {
        Self {
            mode: HarmonyMode::AwaitingLeadChannel,
            pending: String::new(),
        }
    }

    /// Feed one raw token. Returns the content/reasoning pieces safe to emit.
    pub fn process(&mut self, tok: &str) -> Vec<ThinkPiece> {
        self.pending.push_str(tok);
        let mut out = Vec::new();
        loop {
            match self.mode {
                HarmonyMode::AwaitingLeadChannel => {
                    if self.pending.len() <= MAX_CHANNEL_NAME_LEN {
                        break; // wait for more (or `flush()` at stream end)
                    }
                    self.mode = match match_leading_channel_word(&self.pending) {
                        Some(name) => {
                            self.pending = self.pending[name.len()..].to_owned();
                            harmony_mode_for(name)
                        }
                        // gpt-oss should always open with a channel word;
                        // degrade safely to Content rather than lose the
                        // response.
                        None => HarmonyMode::Content,
                    };
                    // loop continues, reprocessing `pending` under the new mode
                }
                HarmonyMode::Reasoning => {
                    if let Some((idx, marker_len, next_mode)) =
                        find_harmony_transition(&self.pending)
                    {
                        let before = self.pending[..idx].to_owned();
                        self.pending = self.pending[idx + marker_len..].to_owned();
                        if !before.is_empty() {
                            out.push(ThinkPiece::Reasoning(before));
                        }
                        self.mode = next_mode;
                        // loop continues
                    } else {
                        let hold = max_harmony_overlap(&self.pending);
                        let safe_end = self.pending.len() - hold;
                        if safe_end > 0 {
                            out.push(ThinkPiece::Reasoning(self.pending[..safe_end].to_owned()));
                        }
                        self.pending = self.pending[safe_end..].to_owned();
                        break;
                    }
                }
                HarmonyMode::Content => {
                    // Harmony emits at most one `final` segment per assistant
                    // turn — no further transition is expected once here, so
                    // pass everything through without holding anything back.
                    if !self.pending.is_empty() {
                        out.push(ThinkPiece::Content(std::mem::take(&mut self.pending)));
                    }
                    break;
                }
            }
        }
        out
    }

    /// Flush any remaining buffered bytes when the token stream ends.
    pub fn flush(&mut self) -> Vec<ThinkPiece> {
        if self.pending.is_empty() {
            return vec![];
        }
        match self.mode {
            HarmonyMode::AwaitingLeadChannel => {
                // Short response (<= MAX_CHANNEL_NAME_LEN bytes) never
                // crossed the buffering threshold in `process` — resolve now.
                match match_leading_channel_word(&self.pending) {
                    Some(name) => {
                        let rest = self.pending[name.len()..].to_owned();
                        self.pending.clear();
                        if rest.is_empty() {
                            vec![]
                        } else {
                            vec![match harmony_mode_for(name) {
                                HarmonyMode::Content => ThinkPiece::Content(rest),
                                HarmonyMode::Reasoning | HarmonyMode::AwaitingLeadChannel => {
                                    ThinkPiece::Reasoning(rest)
                                }
                            }]
                        }
                    }
                    None => vec![ThinkPiece::Content(std::mem::take(&mut self.pending))],
                }
            }
            HarmonyMode::Reasoning => {
                vec![ThinkPiece::Reasoning(std::mem::take(&mut self.pending))]
            }
            HarmonyMode::Content => vec![ThinkPiece::Content(std::mem::take(&mut self.pending))],
        }
    }
}

/// Buffered (non-streaming) counterpart to [`HarmonyFilter`] — splits a
/// complete gpt-oss response into `(reasoning, answer)`, mirroring
/// [`extract_thinking`]'s signature for `<think>`-tag models. Built on
/// `HarmonyFilter` itself (process the whole text, then flush) rather than a
/// second hand-written scanner, so streaming and buffered behavior cannot
/// drift apart.
#[must_use]
pub fn extract_reasoning_gpt_oss(text: &str) -> (Option<String>, String) {
    let mut filter = HarmonyFilter::new();
    let mut reasoning = String::new();
    let mut answer = String::new();
    for piece in filter.process(text).into_iter().chain(filter.flush()) {
        match piece {
            ThinkPiece::Reasoning(s) => reasoning.push_str(&s),
            ThinkPiece::Content(s) => answer.push_str(&s),
        }
    }
    let reasoning = if reasoning.trim().is_empty() {
        None
    } else {
        Some(reasoning.trim().to_owned())
    };
    (reasoning, answer.trim().to_owned())
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    // ---- ModelFamily::detect ----

    #[test]
    fn detect_mistral_from_system_prompt_marker() {
        assert_eq!(
            ModelFamily::detect("...[SYSTEM_PROMPT]{tools}..."),
            ModelFamily::Mistral
        );
    }

    #[test]
    fn detect_default_for_qwen_style_template() {
        assert_eq!(
            ModelFamily::detect("{%- if tools %}<tools>...</tools>"),
            ModelFamily::Default
        );
    }

    // ---- parse_tool_calls: Default (<tool_call>) ----

    #[test]
    fn default_single_tool_call() {
        let text =
            r#"<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>"#;
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::Default);
        let calls = calls.expect("one call expected");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        // arguments is a compact JSON *string*
        assert_eq!(calls[0].function.arguments, r#"{"city":"Paris"}"#);
        assert_eq!(calls[0].kind, "function");
        assert!(calls[0].id.starts_with("call_"));
        assert!(
            remaining.is_empty(),
            "tags should be stripped: {remaining:?}"
        );
    }

    #[test]
    fn default_multiple_tool_calls() {
        let text = concat!(
            r#"<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>"#,
            "\n",
            r#"<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>"#,
        );
        let (calls, _) = parse_tool_calls(text, ModelFamily::Default);
        let calls = calls.expect("two calls expected");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(calls[1].function.name, "b");
        // ids must be distinct within a response
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn default_mixed_prose_and_call_strips_to_prose() {
        let text = concat!(
            "Let me check that for you.\n",
            r#"<tool_call>{"name": "lookup", "arguments": {}}</tool_call>"#,
        );
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::Default);
        assert!(calls.is_some());
        assert_eq!(remaining, "Let me check that for you.");
    }

    #[test]
    fn default_arguments_absent_defaults_to_empty_object() {
        let text = r#"<tool_call>{"name": "ping"}</tool_call>"#;
        let (calls, _) = parse_tool_calls(text, ModelFamily::Default);
        let calls = calls.unwrap();
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn default_malformed_json_is_skipped() {
        let text = "<tool_call>{not valid json}</tool_call>";
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::Default);
        assert!(
            calls.is_none(),
            "malformed call must not produce a ToolCall"
        );
        // No valid call → original text returned (trimmed).
        assert_eq!(remaining, text);
    }

    #[test]
    fn default_invalid_backslash_escape_is_repaired() {
        // Reproduces the 2026-07-22 live incident (the project's internal engineering log): a model
        // embeds generated JS with an unescaped regex backslash inside the
        // JSON `content` argument. Un-repaired, this fails outright and the
        // client is left retrying a doomed generation forever.
        let text = r#"<tool_call>{"name": "write_file", "arguments": {"path": "x.js", "content": "const re = /\d+/;"}}</tool_call>"#;
        let (calls, _) = parse_tool_calls(text, ModelFamily::Default);
        let calls = calls.expect("invalid \\d escape should be repaired, not dropped");
        assert_eq!(calls[0].function.name, "write_file");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["content"], r"const re = /\d+/;");
    }

    #[test]
    fn default_repair_does_not_corrupt_valid_escapes() {
        let text = r#"<tool_call>{"name": "write_file", "arguments": {"content": "line1\nline2\t\"quoted\"\\backslash"}}</tool_call>"#;
        let (calls, _) = parse_tool_calls(text, ModelFamily::Default);
        let calls = calls.expect("well-formed escapes must parse without repair");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["content"], "line1\nline2\t\"quoted\"\\backslash");
    }

    #[test]
    fn default_one_valid_one_malformed_keeps_the_valid_one() {
        let text = concat!(
            "<tool_call>{bad}</tool_call>",
            r#"<tool_call>{"name": "ok", "arguments": {}}</tool_call>"#,
        );
        let (calls, _) = parse_tool_calls(text, ModelFamily::Default);
        let calls = calls.expect("the valid call survives");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "ok");
    }

    #[test]
    fn default_no_call_returns_text_unchanged() {
        let text = "Just a normal answer with no tools.";
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::Default);
        assert!(calls.is_none());
        assert_eq!(remaining, text);
    }

    #[test]
    fn context_window_centers_on_offset_with_markers() {
        let s = "0123456789abcdefghij";
        assert_eq!(context_window(s, 10, 3), "…789abc…");
    }

    #[test]
    fn context_window_no_markers_when_offset_covers_whole_string() {
        let s = "short";
        assert_eq!(context_window(s, 2, 10), "short");
    }

    #[test]
    fn line_col_to_byte_offset_finds_second_line() {
        let s = "abc\ndef";
        // serde_json errors are 1-based; column 2 on line 2 is the 'e'.
        assert_eq!(line_col_to_byte_offset(s, 2, 2), 5);
    }

    #[test]
    fn default_malformed_json_log_context_centers_on_error_location() {
        // A long valid-looking prefix followed by a broken tail: the old
        // 200-char-prefix truncation would have shown only the harmless
        // start. Confirms line_col_to_byte_offset/context_window land on the
        // actual failure instead.
        let padding = "x".repeat(300);
        let raw = format!(r#"{{"name": "{padding}", "arguments": {{}} BROKEN"#);
        let err = serde_json::from_str::<serde_json::Value>(&raw).expect_err("must fail to parse");
        let offset = line_col_to_byte_offset(&raw, err.line(), err.column());
        let window = context_window(&raw, offset, 20);
        assert!(
            window.contains("BROKEN"),
            "expected the error-centered window to contain the broken tail, got: {window}"
        );
    }

    // ---- parse_tool_calls: Default fenced-json fallback (qwen2.5-coder) ----

    #[test]
    fn default_fenced_json_call_is_parsed() {
        // Reproduces the qwen2.5-coder-14b-int4 / Hermes Agent bug: the model
        // ignores the <tool_call> delimiter instruction and wraps the same
        // shape in a markdown fence instead.
        let text = "```json\n{\"name\": \"search_files\", \"arguments\": {\"pattern\": \"*\", \"path\": \"/home/user\"}}\n```";
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::Default);
        let calls = calls.expect("fenced call should be recognised");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search_files");
        assert_eq!(
            calls[0].function.arguments,
            r#"{"pattern":"*","path":"/home/user"}"#
        );
        assert!(
            remaining.is_empty(),
            "fence should be stripped: {remaining:?}"
        );
    }

    #[test]
    fn default_fenced_json_nested_arguments_not_truncated() {
        // Guards the balanced-brace scan: a naive non-greedy regex over the
        // JSON body would stop at the FIRST '}' (closing the inner object),
        // producing invalid/truncated JSON.
        let text = r#"```json
{"name": "search_files", "arguments": {"filter": {"path": "/home", "depth": 2}}}
```"#;
        let (calls, _) = parse_tool_calls(text, ModelFamily::Default);
        let calls = calls.expect("nested-argument call should still parse");
        assert_eq!(
            calls[0].function.arguments,
            r#"{"filter":{"path":"/home","depth":2}}"#
        );
    }

    #[test]
    fn default_bare_json_call_no_fence_is_parsed() {
        let text = r#"{"name": "ping", "arguments": {}}"#;
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::Default);
        let calls = calls.expect("bare call should be recognised");
        assert_eq!(calls[0].function.name, "ping");
        assert!(remaining.is_empty());
    }

    #[test]
    fn default_fenced_json_parameters_key_is_honored() {
        let text = "```json\n{\"name\": \"lookup\", \"parameters\": {\"q\": \"rust\"}}\n```";
        let (calls, _) = parse_tool_calls(text, ModelFamily::Default);
        let calls = calls.expect("parameters key should be accepted like arguments");
        assert_eq!(calls[0].function.arguments, r#"{"q":"rust"}"#);
    }

    #[test]
    fn default_fenced_non_call_json_is_left_as_content() {
        // A fenced block that IS valid JSON but isn't shaped like a call
        // (no "name") must not be misread as one.
        let text = "```json\n{\"city\": \"Paris\", \"temp_c\": 18}\n```";
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::Default);
        assert!(calls.is_none());
        assert_eq!(remaining, text.trim());
    }

    #[test]
    fn default_fenced_code_block_is_left_as_content() {
        // An ordinary (non-JSON) fenced code example must not be consumed.
        let text = "```python\ndef f(x):\n    return x + 1\n```";
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::Default);
        assert!(calls.is_none());
        assert_eq!(remaining, text.trim());
    }

    #[test]
    fn default_incidental_json_in_prose_is_not_a_call() {
        // JSON mentioned mid-sentence, not fenced and not the whole response
        // — must not trigger the bare-object fallback.
        let text = r#"Sure, something like {"name": "x", "arguments": {}} would work."#;
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::Default);
        assert!(calls.is_none());
        assert_eq!(remaining, text);
    }

    // ---- parse_tool_calls: Default pythonic-call fallback (2026-07-22 live incident) ----

    #[test]
    fn default_fenced_pythonic_call_is_parsed() {
        // Reproduces the real Hermes/qwen3-vl-8b-int8-ov incident: a bash
        // fence wrapping a keyword-call expression instead of any JSON shape.
        let text = "```bash\nwrite_file(path=\"/home/user/test.html\", content=\"<html>\\n<body>ok</body>\\n</html>\")\n```";
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::Default);
        let calls = calls.expect("pythonic call should be recognised");
        assert_eq!(calls[0].function.name, "write_file");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["path"], "/home/user/test.html");
        assert_eq!(args["content"], "<html>\n<body>ok</body>\n</html>");
        assert!(
            remaining.is_empty(),
            "fence should be stripped: {remaining:?}"
        );
    }

    #[test]
    fn default_bare_pythonic_call_no_fence_is_parsed() {
        let text = r#"read_file(path="test.html")"#;
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::Default);
        let calls = calls.expect("bare pythonic call should be recognised");
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[0].function.arguments, r#"{"path":"test.html"}"#);
        assert!(remaining.is_empty());
    }

    #[test]
    fn default_pythonic_call_escaped_quote_in_value_does_not_terminate_early() {
        let text = r#"write_file(path="x.html", content="<div class=\"a\">hi</div>")"#;
        let (calls, _) = parse_tool_calls(text, ModelFamily::Default);
        let calls = calls.expect("escaped quote inside value must not truncate the string");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["content"], r#"<div class="a">hi</div>"#);
    }

    #[test]
    fn default_pythonic_call_bare_value_is_parsed_as_json_literal() {
        let text = r"write_todos(merge=true, count=3)";
        let (calls, _) = parse_tool_calls(text, ModelFamily::Default);
        let calls = calls.expect("bare true/number values should parse");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["merge"], true);
        assert_eq!(args["count"], 3);
    }

    #[test]
    fn default_fenced_pythonic_call_with_leading_narration_is_parsed() {
        // Reproduces the OTHER 2026-07-22 incident (the project's internal engineering log,
        // "Real-Hermes tool-call flakiness"): qwen3-vl-8b-int8-ov narrated a fake
        // ```bash\nread_file(path="./test.html")\n``` block instead of a
        // structured <tool_call>, prefixed by conversational narration —
        // unlike the bare mid-prose case above, `fenced_block_re` isn't
        // anchored to the whole text, so a fence anywhere in the response
        // (with prose before/after it) is still found and its body parsed.
        // Verifies the pythonic fallback (built later that session for the
        // sibling write_file incident) already catches this exact shape,
        // independent of any temperature/sampling change.
        let text = "Let me check that file for you:\n```bash\nread_file(path=\"./test.html\")\n```\nOne moment.";
        let (calls, _remaining) = parse_tool_calls(text, ModelFamily::Default);
        let calls =
            calls.expect("fenced pythonic call must be found despite surrounding narration");
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[0].function.arguments, r#"{"path":"./test.html"}"#);
    }

    #[test]
    fn default_pythonic_call_embedded_in_prose_is_rejected() {
        // A call-shaped fragment mid-sentence must not be mistaken for a
        // real call — the "name" would contain spaces/punctuation from the
        // surrounding prose and fail the identifier check.
        let text = r#"I will call search(query="x") once I have more info."#;
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::Default);
        assert!(calls.is_none());
        assert_eq!(remaining, text);
    }

    // ---- parse_tool_calls: Mistral (name{json}) ----

    #[test]
    fn mistral_single_call() {
        let text = r#"get_weather{"city": "Paris"}"#;
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::Mistral);
        let calls = calls.expect("one call expected");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Paris"}"#);
        assert!(remaining.is_empty());
    }

    #[test]
    fn mistral_non_matching_returns_text() {
        let text = "This is just prose, not a call.";
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::Mistral);
        assert!(calls.is_none());
        assert_eq!(remaining, text);
    }

    #[test]
    fn mistral_malformed_args_returns_text() {
        let text = "do_thing{not json}";
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::Mistral);
        assert!(calls.is_none());
        assert_eq!(remaining, text);
    }

    // ---- parse_tool_calls: Mistral v3 ([TOOL_CALLS]) ----

    #[test]
    fn mistral_v3_single_call_with_id() {
        let text = r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}, "id": "abc123xyz"}]"#;
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::Mistral);
        let calls = calls.expect("one call expected");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        // arguments serialised to compact JSON string
        assert_eq!(calls[0].function.arguments, r#"{"city":"Paris"}"#);
        assert_eq!(calls[0].id, "abc123xyz");
        assert!(remaining.is_empty());
    }

    #[test]
    fn mistral_v3_multiple_calls() {
        let text = concat!(
            r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}, "id": "id000001a"}, "#,
            r#"{"name": "get_time", "arguments": {"tz": "UTC"}, "id": "id000002b"}]"#,
        );
        let (calls, _) = parse_tool_calls(text, ModelFamily::Mistral);
        let calls = calls.expect("two calls expected");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
        assert_eq!(calls[0].id, "id000001a");
        assert_eq!(calls[1].id, "id000002b");
    }

    #[test]
    fn mistral_v3_missing_id_gets_generated() {
        let text = r#"[TOOL_CALLS] [{"name": "ping", "arguments": {}}]"#;
        let (calls, _) = parse_tool_calls(text, ModelFamily::Mistral);
        let calls = calls.expect("one call expected");
        assert!(
            calls[0].id.starts_with("call_"),
            "generated id: {}",
            calls[0].id
        );
    }

    #[test]
    fn mistral_v3_takes_priority_over_v1() {
        // If both patterns appear somehow, v3 wins (checked first).
        let text = r#"[TOOL_CALLS] [{"name": "real_fn", "arguments": {}, "id": "123456789"}] fake_fn{"x":1}"#;
        let (calls, _) = parse_tool_calls(text, ModelFamily::Mistral);
        let calls = calls.expect("v3 parse expected");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "real_fn");
    }

    // ---- ModelFamily::detect: QwenCoder ----

    #[test]
    fn detect_qwen_coder_from_function_tag() {
        assert_eq!(
            ModelFamily::detect("... <function=example_fn> ... <tool_call> ..."),
            ModelFamily::QwenCoder
        );
    }

    #[test]
    fn detect_mistral_beats_qwen_coder() {
        // [SYSTEM_PROMPT] is checked first; should never appear in QwenCoder
        // templates, but if it did, Mistral wins.
        assert_eq!(
            ModelFamily::detect("[SYSTEM_PROMPT] <function=x>"),
            ModelFamily::Mistral
        );
    }

    // ---- parse_tool_calls: QwenCoder XML ----

    #[test]
    fn qwen_coder_single_call_string_param() {
        let text = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::QwenCoder);
        let calls = calls.expect("one call expected");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Paris"}"#);
        assert!(calls[0].id.starts_with("call_"));
        assert!(remaining.is_empty());
    }

    #[test]
    fn qwen_coder_multiple_calls() {
        let text = concat!(
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n",
            "<tool_call>\n<function=get_current_time>\n<parameter=timezone>\nEurope/Paris\n</parameter>\n</function>\n</tool_call>",
        );
        let (calls, _) = parse_tool_calls(text, ModelFamily::QwenCoder);
        let calls = calls.expect("two calls expected");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_current_time");
        // IDs must be distinct
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn qwen_coder_no_call_returns_text() {
        let text = "Just a normal answer.";
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::QwenCoder);
        assert!(calls.is_none());
        assert_eq!(remaining, text);
    }

    #[test]
    fn qwen_coder_prose_before_call_stripped() {
        let text = "Let me check.\n<tool_call>\n<function=lookup>\n<parameter=q>\nrust\n</parameter>\n</function>\n</tool_call>";
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::QwenCoder);
        assert!(calls.is_some());
        assert_eq!(remaining, "Let me check.");
    }

    // ---- templates/lfm2.jinja (the shipped override) ----

    /// The repo-shipped LFM2 template must actually parse under minijinja.
    /// Upstream's copy does not: it uses `{% generation %}`, a transformers-only
    /// tag minijinja has no statement for. This is the regression guard for
    /// edit 1 in that file's header.
    #[test]
    fn shipped_lfm2_template_parses_and_renders() {
        let template = include_str!("../templates/lfm2.jinja");
        let messages = vec![serde_json::json!({"role": "user", "content": "hi"})];
        let out = build_prompt(
            &messages,
            None,
            template,
            "<|im_end|>",
            "<|startoftext|>",
            None,
            None,
        )
        .expect("shipped LFM2 template must render");
        assert!(out.contains("<|im_start|>user"), "got: {out}");
        assert!(out.contains("hi"), "got: {out}");
    }

    /// The bug this override exists to fix: the model-shipped template renders
    /// **nothing** for an assistant turn carrying `tool_calls`, so the turn
    /// replays as an empty `<|im_start|>assistant<|im_end|>` and the model sees
    /// a tool result it never appears to have asked for. The corrected template
    /// must render the call back in the model's own protocol.
    #[test]
    fn shipped_lfm2_template_renders_an_assistant_tool_call_turn() {
        let template = include_str!("../templates/lfm2.jinja");
        let messages = vec![
            serde_json::json!({"role": "user", "content": "weather in Gdansk?"}),
            serde_json::json!({
                "role": "assistant",
                "content": null,
                // OpenAI wire format: `arguments` is a JSON *string*, which
                // upstream's macro rejects outright (edit 2 in the header).
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\": \"Gdansk\"}"}
                }]
            }),
            serde_json::json!({"role": "tool", "content": "12C"}),
        ];
        let out = build_prompt(
            &messages,
            None,
            template,
            "<|im_end|>",
            "<|startoftext|>",
            None,
            None,
        )
        .expect("tool-call turn must render");
        assert!(
            out.contains("<|tool_call_start|>"),
            "the assistant turn must replay the call in LFM2's own protocol, got: {out}"
        );
        assert!(
            out.contains("get_weather("),
            "call must be rendered Pythonic, got: {out}"
        );
        assert!(
            out.contains("Gdansk"),
            "the JSON-string arguments must be parsed, got: {out}"
        );
        assert!(
            out.contains("12C"),
            "the tool result must still render, got: {out}"
        );
    }

    /// A round trip: what `parse_lfm2` extracts must be renderable back into a
    /// prompt, which is what makes multi-turn tool conversations work at all.
    #[test]
    fn lfm2_parse_then_render_round_trips() {
        let (calls, _) = parse_tool_calls("[get_weather(city='Gdansk')]", ModelFamily::Lfm2);
        let calls = calls.expect("must parse");
        let messages = vec![serde_json::json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": calls[0].id,
                "type": "function",
                "function": {
                    "name": calls[0].function.name,
                    "arguments": calls[0].function.arguments,
                }
            }]
        })];
        let template = include_str!("../templates/lfm2.jinja");
        let out = build_prompt(
            &messages,
            None,
            template,
            "<|im_end|>",
            "<|startoftext|>",
            None,
            None,
        )
        .expect("round trip must render");
        assert!(out.contains("get_weather(city="), "got: {out}");
        assert!(out.contains("Gdansk"), "got: {out}");
    }

    // ---- enable_thinking on the VLM path (Option D) ----

    /// The regression that landing `extra_context` WITHOUT the request-aware
    /// `vlm_starts_thinking` guard would have caused, pinned at the layer where
    /// the damage happens.
    ///
    /// With `enable_thinking: false` the template emits a CLOSED prefill, so
    /// generation starts OUTSIDE thinking. If the caller still passes
    /// `starts_in_thinking = true`, `extract_thinking` takes the shape-(d)
    /// cut-off branch: the real answer is swallowed into reasoning and the
    /// caller is handed a fabricated placeholder instead.
    #[test]
    fn thinking_misroute_swallows_the_answer_when_starts_in_thinking_is_wrong() {
        let model_output = "The answer is 4.";
        let (thinking, answer) = extract_thinking(model_output, true);
        assert_eq!(
            answer, "*(thinking was cut off by max_tokens limit)*",
            "wrong starts_in_thinking must produce the fabricated placeholder \
             this test exists to document"
        );
        assert_eq!(thinking.as_deref(), Some(model_output));
    }

    /// The same output with the CORRECT flag: answer preserved, no reasoning.
    #[test]
    fn closed_prefill_output_is_preserved_when_starts_in_thinking_is_false() {
        let (thinking, answer) = extract_thinking("The answer is 4.", false);
        assert_eq!(answer, "The answer is 4.");
        assert!(thinking.is_none());
    }

    /// The consequence that actually matters for an agent client: tool calls
    /// are parsed out of `answer`, so a misrouted `starts_in_thinking` loses
    /// every one of them. This is the concrete "3/3 clean calls become 0/3"
    /// regression.
    #[test]
    fn misrouted_thinking_loses_tool_calls_entirely() {
        let with_call =
            "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Gdansk\"}}</tool_call>";

        // Correct flag: the call survives and parses.
        let (_, answer_ok) = extract_thinking(with_call, false);
        let (calls_ok, _) = parse_tool_calls(&answer_ok, ModelFamily::Default);
        assert_eq!(calls_ok.expect("call must parse").len(), 1);

        // Misrouted: `answer` is the placeholder, so there is nothing to parse.
        let (_, answer_bad) = extract_thinking(with_call, true);
        let (calls_bad, _) = parse_tool_calls(&answer_bad, ModelFamily::Default);
        assert!(
            calls_bad.is_none(),
            "documents the regression: a misrouted flag destroys tool calls"
        );
    }

    /// `template_prefills_thinking` is a static scan and always reports `true`
    /// for a Qwen-style template, because the source contains BOTH branches.
    /// That is exactly why the runtime guard has to consider the request — the
    /// template alone cannot tell you which branch was taken.
    #[test]
    fn qwen_style_template_always_scans_as_prefilling_thinking() {
        let tmpl = "{%- if enable_thinking is defined and enable_thinking is false %}\
                    {{- '<think>\\n\\n</think>\\n\\n' }}{%- else %}{{- '<think>\\n' }}{%- endif %}";
        assert!(
            template_prefills_thinking(tmpl),
            "static scan sees the open-branch tag regardless of the flag"
        );
    }

    // ---- parse_lfm2 ----

    /// The model card's own example, verbatim — including the markers, which
    /// the detokenizer normally strips, and the prose that follows the call in
    /// the same turn.
    #[test]
    fn lfm2_parses_the_model_card_example() {
        let text = "<|tool_call_start|>[get_candidate_status(candidate_id=\"12345\")]\
                    <|tool_call_end|>Checking the current status of candidate ID 12345.";
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::Lfm2);
        let calls = calls.expect("model-card example must parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_candidate_status");
        assert_eq!(calls[0].function.arguments, r#"{"candidate_id":"12345"}"#);
        assert_eq!(
            remaining,
            "Checking the current status of candidate ID 12345."
        );
    }

    /// The realistic case: markers stripped by the detokenizer, so only the
    /// bare list survives. This is what the parser actually sees in production.
    #[test]
    fn lfm2_parses_a_bare_list_without_markers() {
        let (calls, _) = parse_tool_calls("[get_weather(city='Gdansk')]", ModelFamily::Lfm2);
        let calls = calls.expect("bare list must parse");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Gdansk"}"#);
    }

    /// Both quote styles occur: the card uses `"`, the upstream template's own
    /// `format_arg_value` emits `'`.
    #[test]
    fn lfm2_accepts_both_quote_styles() {
        let (a, _) = parse_tool_calls("[f(x=\"v\")]", ModelFamily::Lfm2);
        let (b, _) = parse_tool_calls("[f(x='v')]", ModelFamily::Lfm2);
        assert_eq!(
            a.expect("double")
                .first()
                .map(|c| c.function.arguments.clone()),
            b.expect("single")
                .first()
                .map(|c| c.function.arguments.clone())
        );
    }

    /// Several calls arrive comma-joined inside one list.
    #[test]
    fn lfm2_parses_multiple_calls_in_one_list() {
        let (calls, remaining) = parse_tool_calls("[a(x=1), b(y='two')] done", ModelFamily::Lfm2);
        let calls = calls.expect("two calls");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(calls[1].function.arguments, r#"{"y":"two"}"#);
        assert_eq!(remaining, "done");
        assert_ne!(calls[0].id, calls[1].id, "ids must be distinct");
    }

    /// The case a regex cannot handle: brackets, commas and quotes *inside* a
    /// string argument must not terminate the call.
    #[test]
    fn lfm2_handles_delimiters_inside_string_arguments() {
        let (calls, _) = parse_tool_calls(r#"[search(q="a) , ] b", limit=2)]"#, ModelFamily::Lfm2);
        let calls = calls.expect("must survive delimiters inside the string");
        assert_eq!(calls.len(), 1);
        // Compare parsed values, not the serialised string — key order is an
        // implementation detail of the JSON map, and what matters here is that
        // the `)`, `,` and `]` inside the quoted argument did not terminate it.
        let v: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).expect("valid JSON");
        assert_eq!(v["q"], serde_json::json!("a) , ] b"));
        assert_eq!(v["limit"], serde_json::json!(2));
    }

    /// Python and JSON spellings both occur — the template renders nested
    /// values with `tojson` while scalars go through `| string`.
    #[test]
    fn lfm2_accepts_python_and_json_scalars() {
        let (calls, _) = parse_tool_calls(
            "[f(a=True, b=None, c=false, d=1.5, e=[1, 2], g={'k': 'v'})]",
            ModelFamily::Lfm2,
        );
        let calls = calls.expect("must parse");
        let args = &calls.first().expect("one call").function.arguments;
        let v: serde_json::Value = serde_json::from_str(args).expect("valid JSON");
        assert_eq!(v["a"], serde_json::json!(true));
        assert_eq!(v["b"], serde_json::Value::Null);
        assert_eq!(v["c"], serde_json::json!(false));
        assert_eq!(v["d"], serde_json::json!(1.5));
        assert_eq!(v["e"], serde_json::json!([1, 2]));
        assert_eq!(v["g"], serde_json::json!({"k": "v"}));
    }

    /// Ordinary prose containing brackets must not be mistaken for a call —
    /// this is the whole risk of shape-based bounding, so pin it hard.
    #[test]
    fn lfm2_does_not_false_positive_on_prose() {
        for text in [
            "See [the docs](https://example.com) for details.",
            "As shown in [1] and [2].",
            "Use the list [1, 2, 3] here.",
            "Call [Note(s)] carefully.",
            "Nothing bracketed at all.",
        ] {
            let (calls, remaining) = parse_tool_calls(text, ModelFamily::Lfm2);
            assert!(calls.is_none(), "must not parse a call from: {text}");
            assert_eq!(remaining, text, "prose must survive untouched: {text}");
        }
    }

    /// A truncated list (hit `max_tokens` mid-call) must stay prose rather than
    /// be half-parsed into a bogus call.
    #[test]
    fn lfm2_unterminated_list_is_left_as_text() {
        let text = "[get_weather(city=\"Gda";
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::Lfm2);
        assert!(calls.is_none());
        assert_eq!(remaining, text);
    }

    /// Positional arguments cannot be bound to parameter names without the
    /// tool schema, so the call is skipped rather than guessed at.
    #[test]
    fn lfm2_positional_arguments_are_not_guessed() {
        let (calls, _) = parse_tool_calls("[get_weather('Gdansk')]", ModelFamily::Lfm2);
        assert!(
            calls.is_none(),
            "positional args must not be invented into names"
        );
    }

    /// The card allows a JSON dialect when the system prompt asks for one, so
    /// the shared fallbacks must still be reachable.
    #[test]
    fn lfm2_falls_back_to_json_dialect() {
        let text =
            "```json\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Gdansk\"}}\n```";
        let (calls, _) = parse_tool_calls(text, ModelFamily::Lfm2);
        assert!(calls.is_some(), "JSON dialect must still be parsed");
    }

    // ---- ModelFamily::detect: Lfm2 ----

    /// A corrected LFM2 template renders the model's own tool protocol, so it
    /// contains the marker directly.
    #[test]
    fn detects_lfm2_from_tool_call_token_in_template() {
        assert_eq!(
            ModelFamily::detect(
                "{%- if tools %}<|tool_list_start|>...<|tool_call_start|>{% endif %}"
            ),
            ModelFamily::Lfm2
        );
    }

    /// The template actually shipped with `lfm2-24b-a2b-int4-ov` contains none
    /// of the tool tokens — it renders tools as plain `List of tools: [...]`
    /// text — so detection falls back to `keep_past_thinking`, which is unique
    /// to LFM2 across every model served on this fleet.
    #[test]
    fn detects_lfm2_from_shipped_template_without_tool_tokens() {
        let shipped = "{{- bos_token -}}{%- set keep_past_thinking = keep_past_thinking \
                       | default(false) -%}{%- if tools -%}List of tools: [{%- endif -%}";
        assert!(
            !shipped.contains("<|tool_call_start|>"),
            "fixture must lack the marker"
        );
        assert_eq!(ModelFamily::detect(shipped), ModelFamily::Lfm2);
    }

    /// Ordinary Qwen-style templates must not be captured by either LFM2
    /// marker — this is the regression that would silently re-dialect every
    /// other served model.
    #[test]
    fn qwen_style_template_is_not_lfm2() {
        assert_eq!(
            ModelFamily::detect("{%- if tools %}<tools>...</tools>{% endif %}<tool_call>"),
            ModelFamily::Default
        );
    }

    /// The earlier families win over LFM2 when both could match — order in
    /// `detect` is load-bearing, so pin it.
    #[test]
    fn earlier_families_take_precedence_over_lfm2() {
        assert_eq!(
            ModelFamily::detect("[SYSTEM_PROMPT] keep_past_thinking"),
            ModelFamily::Mistral
        );
        assert_eq!(
            ModelFamily::detect("<|channel|>analysis keep_past_thinking"),
            ModelFamily::GptOss
        );
    }

    /// The label is part of the response surface (`model_family`), so an
    /// operator can see which dialect was applied.
    #[test]
    fn lfm2_label_is_snake_case() {
        assert_eq!(ModelFamily::Lfm2.label(), "lfm2");
    }

    // ---- ModelFamily::detect: GptOss ----

    #[test]
    fn detect_gpt_oss_from_channel_tag() {
        assert_eq!(
            ModelFamily::detect("... <|channel|>analysis<|message|> ..."),
            ModelFamily::GptOss
        );
    }

    #[test]
    fn detect_mistral_beats_gpt_oss() {
        // Same precedence rule as detect_mistral_beats_qwen_coder — checked
        // first, wins if a template somehow carried both markers.
        assert_eq!(
            ModelFamily::detect("[SYSTEM_PROMPT] <|channel|>x"),
            ModelFamily::Mistral
        );
    }

    // ---- ModelFamily::label ----

    /// Every variant has a distinct, stable `snake_case` label — the
    /// `generation_metadata`/chat-response `model_family` wire value.
    #[test]
    fn label_covers_every_variant_distinctly() {
        assert_eq!(ModelFamily::Default.label(), "default");
        assert_eq!(ModelFamily::Mistral.label(), "mistral");
        assert_eq!(ModelFamily::QwenCoder.label(), "qwen_coder");
        assert_eq!(ModelFamily::GptOss.label(), "gpt_oss");

        let labels = [
            ModelFamily::Default.label(),
            ModelFamily::Mistral.label(),
            ModelFamily::QwenCoder.label(),
            ModelFamily::GptOss.label(),
        ];
        let unique: std::collections::BTreeSet<&str> = labels.iter().copied().collect();
        assert_eq!(unique.len(), labels.len(), "labels must all be distinct");
    }

    // ---- parse_tool_calls: gpt-oss harmony ----
    //
    // Live-verified against `gpt-oss-20b-int4-ov` (2026-07-27) via a real
    // /v1/chat/completions tool-calling request. Meant to run on text
    // `HarmonyFilter` has already processed (see extract_reasoning_gpt_oss) —
    // these samples are exactly what showed up in `reasoning_content`.

    #[test]
    fn gpt_oss_single_call_with_leading_reasoning() {
        let text = "We need to call function get_weather with location \"San Francisco\". \
            to=functions.get_weather json{\"location\":\"San Francisco\"}";
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::GptOss);
        let calls = calls.expect("one call expected");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            calls[0].function.arguments,
            r#"{"location":"San Francisco"}"#
        );
        assert!(calls[0].id.starts_with("call_"));
        assert_eq!(
            remaining,
            "We need to call function get_weather with location \"San Francisco\"."
        );
    }

    #[test]
    fn gpt_oss_bare_call_no_leading_reasoning() {
        // The enable_thinking:false live sample: no analysis text at all.
        let text = "to=functions.get_weather json{\"location\":\"San Francisco\"}";
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::GptOss);
        let calls = calls.expect("one call expected");
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(remaining.is_empty());
    }

    #[test]
    fn gpt_oss_call_with_nested_json_arguments() {
        // balanced_json_object must track brace depth, not stop at the first
        // '}' (which would truncate a nested object).
        let text = "to=functions.book_flight json\
            {\"origin\":\"SFO\",\"passenger\":{\"name\":\"A\",\"seat\":{\"row\":12}}}";
        let (calls, _) = parse_tool_calls(text, ModelFamily::GptOss);
        let calls = calls.expect("one call expected");
        assert_eq!(calls[0].function.name, "book_flight");
        assert_eq!(
            calls[0].function.arguments,
            r#"{"origin":"SFO","passenger":{"name":"A","seat":{"row":12}}}"#
        );
    }

    #[test]
    fn gpt_oss_multiple_calls() {
        let text = "to=functions.get_weather json{\"location\":\"SF\"} \
            to=functions.get_time json{\"tz\":\"PST\"}";
        let (calls, _) = parse_tool_calls(text, ModelFamily::GptOss);
        let calls = calls.expect("two calls expected");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn gpt_oss_no_call_returns_text() {
        let text = "Just ordinary analysis-channel reasoning, no tool call here.";
        let (calls, remaining) = parse_tool_calls(text, ModelFamily::GptOss);
        assert!(calls.is_none());
        assert_eq!(remaining, text);
    }

    #[test]
    fn gpt_oss_malformed_arguments_are_skipped_not_fatal() {
        let text = "to=functions.get_weather json{not valid json}";
        let (calls, _) = parse_tool_calls(text, ModelFamily::GptOss);
        assert!(
            calls.is_none(),
            "malformed call must not be reported as a call"
        );
    }

    // ---- extract_thinking ----

    #[test]
    fn thinking_closed_block_is_stripped() {
        let text = "<think>reasoning here</think>The answer.";
        let (thinking, answer) = extract_thinking(text, false);
        assert_eq!(thinking.as_deref(), Some("reasoning here"));
        assert_eq!(answer, "The answer.");
    }

    #[test]
    fn thinking_unclosed_block_keeps_prefix_as_answer() {
        let text = "Partial answer.<think>cut off mid thought";
        let (thinking, answer) = extract_thinking(text, false);
        assert_eq!(thinking.as_deref(), Some("cut off mid thought"));
        assert_eq!(answer, "Partial answer.");
    }

    #[test]
    fn thinking_unclosed_with_no_prefix_uses_placeholder() {
        let text = "<think>only thinking, no answer yet";
        let (thinking, answer) = extract_thinking(text, false);
        assert!(thinking.is_some());
        assert_eq!(answer, "*(thinking was cut off by max_tokens limit)*");
    }

    #[test]
    fn thinking_absent_returns_none_and_trimmed_text() {
        let text = "  plain answer  ";
        let (thinking, answer) = extract_thinking(text, false);
        assert!(thinking.is_none());
        assert_eq!(answer, "plain answer");
    }

    #[test]
    fn thinking_dangling_close_is_stripped() {
        // Qwen3.6 shape: template prefilled the opening <think> into the prompt,
        // so the output has reasoning + a closing </think> + answer, with NO
        // opening tag. Regression for the B70 Qwen3.6-35B-A3B think leak.
        let text = "Here's a thinking process:\n1. capital of France is Paris.\n</think>\n\nParis";
        let (thinking, answer) = extract_thinking(text, false);
        assert_eq!(
            thinking.as_deref(),
            Some("Here's a thinking process:\n1. capital of France is Paris.")
        );
        assert_eq!(answer, "Paris");
    }

    #[test]
    fn prefilled_thinking_never_closed_is_all_reasoning() {
        // Shape (d): prompt prefilled <think>, model never emitted the
        // closing tag before max_tokens — NO tags in the output at all. With
        // starts_in_thinking the whole text is reasoning + placeholder answer;
        // without it (pre-fix behavior) it would all leak into content.
        let text = "Here's a thinking process:\n1. Analyze the request.";
        let (thinking, answer) = extract_thinking(text, true);
        assert_eq!(thinking.as_deref(), Some(text));
        assert_eq!(answer, "*(thinking was cut off by max_tokens limit)*");
    }

    #[test]
    fn prefilled_thinking_with_close_still_splits_normally() {
        // starts_in_thinking + a closing tag present → the dangling-close
        // branch handles it; the new shape-(d) branch must not swallow it.
        let text = "reasoning prose\n</think>\n\nWarsaw";
        let (thinking, answer) = extract_thinking(text, true);
        assert_eq!(thinking.as_deref(), Some("reasoning prose"));
        assert_eq!(answer, "Warsaw");
    }

    #[test]
    fn plain_answer_without_prefill_flag_is_untouched() {
        // A non-prefill request (starts_in_thinking = false) with tagless
        // output stays entirely in content — no false reasoning extraction.
        let (thinking, answer) = extract_thinking("Warsaw", false);
        assert!(thinking.is_none());
        assert_eq!(answer, "Warsaw");
    }

    // ---- prompt_ends_in_thinking ----

    #[test]
    fn rendered_prompt_open_think_tail_detected() {
        // Thinking-enabled render: assistant turn ends with an open <think>.
        assert!(prompt_ends_in_thinking("<|im_start|>assistant\n<think>\n"));
        // enable_thinking:false render: the prefill is CLOSED — output starts
        // outside thinking. The static template scan gets this wrong;
        // the rendered-prompt check must not.
        assert!(!prompt_ends_in_thinking(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ));
        // No think tags at all (non-reasoning model).
        assert!(!prompt_ends_in_thinking("<|im_start|>assistant\n"));
    }

    // ---- template_prefills_thinking ----

    #[test]
    fn prefill_detected_when_default_branch_leaves_open_think() {
        // Qwen3-style: explicit-disable branch emits a closed pair, default
        // branch prefills an unclosed <think>. The bare tag survives stripping.
        let tmpl = "...{{- '<think>\\n\\n</think>\\n\\n' }}...{{- '<think>\\n' }}...";
        assert!(template_prefills_thinking(tmpl));
    }

    #[test]
    fn prefill_absent_for_closed_only_or_no_think_templates() {
        assert!(!template_prefills_thinking(
            "...{{- '<think>\\n\\n</think>\\n\\n' }}..."
        ));
        assert!(!template_prefills_thinking(
            "{{- '<|im_start|>assistant\\n' }}"
        ));
    }

    #[test]
    fn think_filter_starting_in_thinking_strips_prefilled_reasoning() {
        // Streaming counterpart: no opening tag in the feed; the filter starts
        // in Thinking mode, routes leading text to Reasoning, drops the close.
        let mut f = ThinkFilter::starting_in_thinking(true);
        let pieces = f.process("reasoning text</think>Paris");
        assert_eq!(
            pieces,
            vec![
                ThinkPiece::Reasoning("reasoning text".into()),
                ThinkPiece::Content("Paris".into()),
            ]
        );
    }

    // ---- build_prompt (minijinja) ----

    /// A tiny stand-in template exercising the same features the real Qwen3
    /// template uses: a `tools` branch, `tojson`, a loop over messages, and a
    /// generation-prompt tail.
    const TEST_TEMPLATE: &str = concat!(
        "{%- if tools %}<tools>{{ tools | tojson }}</tools>\n{% endif -%}",
        "{%- for m in messages -%}{{ m.role }}: {{ m.content }}\n{% endfor -%}",
        "{%- if add_generation_prompt %}assistant:{% endif -%}",
    );

    #[test]
    fn build_prompt_renders_messages_and_tools() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "You are helpful."}),
            serde_json::json!({"role": "user", "content": "Hi"}),
        ];
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {"name": "get_weather"}
        })];
        let out = build_prompt(&messages, Some(&tools), TEST_TEMPLATE, "", "", None, None).unwrap();
        assert!(out.contains("<tools>"), "tools branch must render: {out}");
        assert!(out.contains("get_weather"), "tool name must appear: {out}");
        assert!(out.contains("system: You are helpful."));
        assert!(out.contains("user: Hi"));
        assert!(out.ends_with("assistant:"), "gen prompt tail: {out}");
    }

    #[test]
    fn build_prompt_without_tools_skips_tools_branch() {
        let messages = vec![serde_json::json!({"role": "user", "content": "Hi"})];
        let out = build_prompt(&messages, None, TEST_TEMPLATE, "", "", None, None).unwrap();
        assert!(!out.contains("<tools>"), "no tools → no tools block: {out}");
    }

    /// Qwen3-Coder style: guards with `{%- if not tools is defined %}` and then
    /// calls `tools | length`.  When tools=None is serialised as null the guard
    /// is skipped (null IS defined) and |length on null crashes.  UNDEFINED fixes it.
    #[test]
    fn build_prompt_undefined_tools_passes_is_defined_guard() {
        let messages = vec![serde_json::json!({"role": "user", "content": "hi"})];
        let tmpl = concat!(
            "{%- if not tools is defined %}{%- set tools = [] %}{%- endif %}",
            "{%- if tools | length > 0 %}HAS_TOOLS{% else %}NO_TOOLS{% endif %}"
        );
        let out = build_prompt(&messages, None, tmpl, "", "", None, None).unwrap();
        assert_eq!(out, "NO_TOOLS");
    }

    /// Autoescape MUST be off — angle brackets, ampersands, and quotes pass
    /// through verbatim. If this regresses, every `<tool_call>` and JSON arg
    /// silently corrupts into HTML entities.
    #[test]
    fn build_prompt_does_not_html_escape() {
        let messages = vec![serde_json::json!({
            "role": "user",
            "content": r#"compare <a> & <b> say "hi""#
        })];
        let out = build_prompt(&messages, None, TEST_TEMPLATE, "", "", None, None).unwrap();
        assert!(
            out.contains(r#"<a> & <b> say "hi""#),
            "must not escape: {out}"
        );
        assert!(!out.contains("&lt;"), "no &lt; entity: {out}");
        assert!(!out.contains("&amp;"), "no &amp; entity: {out}");
        assert!(!out.contains("&quot;"), "no &quot; entity: {out}");
    }

    /// Proves the pycompat callback is wired: a Python string method the HF
    /// templates rely on (`.startswith`) resolves instead of erroring.
    #[test]
    fn build_prompt_supports_python_string_methods() {
        let messages = vec![serde_json::json!({"role": "system", "content": "x"})];
        let tmpl = "{%- if messages[0].role.startswith('sys') %}YES{% else %}NO{% endif -%}";
        let out = build_prompt(&messages, None, tmpl, "", "", None, None).unwrap();
        assert_eq!(out, "YES");
    }

    #[test]
    fn build_prompt_render_error_is_propagated() {
        // Calling an undefined function should surface as Err, not a panic.
        let messages = vec![serde_json::json!({"role": "user", "content": "hi"})];
        let tmpl = "{{ this_function_does_not_exist() }}";
        assert!(build_prompt(&messages, None, tmpl, "", "", None, None).is_err());
    }

    #[test]
    fn build_prompt_strftime_now_renders_current_date() {
        let messages = vec![serde_json::json!({"role": "user", "content": "hi"})];
        // Minimal Mistral-style template that calls strftime_now.
        let tmpl = r#"{%- set today = strftime_now("%Y-%m-%d") %}{{ today }}"#;
        let out = build_prompt(&messages, None, tmpl, "", "", None, None).unwrap();
        // Must look like a date: YYYY-MM-DD, starting with "20".
        assert!(out.starts_with("20"), "expected date, got: {out}");
        assert_eq!(out.len(), 10, "expected YYYY-MM-DD, got: {out}");
        assert_eq!(out.as_bytes()[4], b'-');
        assert_eq!(out.as_bytes()[7], b'-');
    }

    #[test]
    fn build_prompt_raise_exception_propagates_as_error() {
        let messages = vec![serde_json::json!({"role": "user", "content": "hi"})];
        let tmpl = r#"{{ raise_exception("unsupported role") }}"#;
        let err = build_prompt(&messages, None, tmpl, "", "", None, None).unwrap_err();
        assert!(err.to_string().contains("unsupported role"), "got: {err}");
    }

    // ---- gen_call_id ----

    #[test]
    fn call_id_format_is_call_plus_8_hex() {
        let id = gen_call_id(0, "f", "{}");
        assert!(id.starts_with("call_"));
        let hex = &id["call_".len()..];
        assert_eq!(hex.len(), 8);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn call_id_distinct_for_distinct_index() {
        assert_ne!(gen_call_id(0, "f", "{}"), gen_call_id(1, "f", "{}"));
    }

    #[test]
    fn call_id_deterministic_for_same_inputs() {
        assert_eq!(
            gen_call_id(3, "f", r#"{"a":1}"#),
            gen_call_id(3, "f", r#"{"a":1}"#)
        );
    }

    // ---- ThinkFilter ----

    #[test]
    fn think_filter_normal_token_passes_through() {
        let mut f = ThinkFilter::new();
        let pieces = f.process("hello");
        assert_eq!(pieces, vec![ThinkPiece::Content("hello".into())]);
    }

    #[test]
    fn think_filter_complete_block_in_one_token() {
        let mut f = ThinkFilter::new();
        let pieces = f.process("<think>reasoning</think>answer");
        assert_eq!(
            pieces,
            vec![
                ThinkPiece::Reasoning("reasoning".into()),
                ThinkPiece::Content("answer".into()),
            ]
        );
    }

    #[test]
    fn think_filter_open_tag_split_across_boundaries() {
        let mut f = ThinkFilter::new();
        let p1 = f.process("hello <thi");
        let p2 = f.process("nk>thinking</think>answer");
        let all: Vec<_> = p1.into_iter().chain(p2).collect();
        assert_eq!(
            all,
            vec![
                ThinkPiece::Content("hello ".into()),
                ThinkPiece::Reasoning("thinking".into()),
                ThinkPiece::Content("answer".into()),
            ]
        );
    }

    #[test]
    fn think_filter_close_tag_split_across_boundaries() {
        let mut f = ThinkFilter::new();
        let p1 = f.process("<think>thinking</");
        let p2 = f.process("think>answer");
        let all: Vec<_> = p1.into_iter().chain(p2).collect();
        assert_eq!(
            all,
            vec![
                ThinkPiece::Reasoning("thinking".into()),
                ThinkPiece::Content("answer".into()),
            ]
        );
    }

    /// `flush` returns the partial-close-tag buffer as Reasoning when the
    /// stream ends while the filter is still inside a `<think>` block.
    #[test]
    fn think_filter_flush_partial_close_tag_is_reasoning() {
        // "reasoning" is emitted immediately; "</thi" is held as a potential
        // close-tag prefix and is only released by flush().
        let mut f = ThinkFilter::new();
        let during = f.process("<think>reasoning</thi");
        let flushed = f.flush();
        assert_eq!(during, vec![ThinkPiece::Reasoning("reasoning".into())]);
        assert_eq!(flushed, vec![ThinkPiece::Reasoning("</thi".into())]);
    }

    #[test]
    fn think_filter_flush_partial_open_tag_is_content() {
        // "<thi" was held as a potential open tag; flush emits it as Content.
        let mut f = ThinkFilter::new();
        let during = f.process("partial <thi");
        let flushed = f.flush();
        assert_eq!(during, vec![ThinkPiece::Content("partial ".into())]);
        assert_eq!(flushed, vec![ThinkPiece::Content("<thi".into())]);
    }

    #[test]
    fn think_filter_empty_token_produces_nothing() {
        let mut f = ThinkFilter::new();
        assert!(f.process("").is_empty());
    }

    #[test]
    fn think_filter_flush_empty_normal_is_empty() {
        let mut f = ThinkFilter::new();
        assert!(f.flush().is_empty());
    }

    // ---- gpt-oss harmony channel handling ----

    /// Live-verified byte-for-byte against `gpt-oss-20b-int4-ov`
    /// (2026-07-27, non-streaming, thinking enabled): the full raw response
    /// for "What is 2+2? Answer briefly."
    const HARMONY_LIVE_SAMPLE: &str = "analysisThe user asks \"What is 2+2?\" \
        They want a brief answer. Simple.\n\nWe can respond with \"4\".\n\n\
        Thus reply. Ensure minimal output.\n\nassistantfinal4";

    #[test]
    fn extract_reasoning_gpt_oss_splits_live_sample() {
        let (reasoning, answer) = extract_reasoning_gpt_oss(HARMONY_LIVE_SAMPLE);
        assert_eq!(answer, "4");
        let reasoning = reasoning.expect("analysis channel must surface as reasoning");
        assert!(
            reasoning.starts_with("The user asks"),
            "leading bare 'analysis' word must be stripped: {reasoning:?}"
        );
        assert!(
            reasoning.ends_with("Ensure minimal output."),
            "trailing 'assistantfinal' marker must be stripped: {reasoning:?}"
        );
        assert!(
            !reasoning.contains("assistantfinal"),
            "transition marker must not leak into reasoning_content: {reasoning:?}"
        );
    }

    /// Live-verified: `enable_thinking:false` response for the same prompt
    /// decodes as exactly "final4" — short enough to never cross
    /// `HarmonyFilter`'s streaming buffering threshold, so this also proves
    /// `flush()` resolves a still-buffering `AwaitingLeadChannel` state.
    #[test]
    fn extract_reasoning_gpt_oss_strips_short_leading_final_no_analysis() {
        let (reasoning, answer) = extract_reasoning_gpt_oss("final4");
        assert_eq!(reasoning, None);
        assert_eq!(answer, "4");
    }

    #[test]
    fn extract_reasoning_gpt_oss_no_channel_word_passes_through() {
        // Degrade-safely case: no leading channel word at all (should not
        // happen for a well-behaved gpt-oss response, but must not eat text).
        let (reasoning, answer) = extract_reasoning_gpt_oss("just a plain answer, no channels");
        assert_eq!(reasoning, None);
        assert_eq!(answer, "just a plain answer, no channels");
    }

    #[test]
    fn extract_reasoning_gpt_oss_commentary_channel_hidden_from_answer() {
        // No `final` segment at all -- a tool-call-only turn. commentary is
        // hidden from `content` (routed to reasoning) rather than leaking the
        // raw `to=functions...` syntax into the visible answer.
        let (reasoning, answer) =
            extract_reasoning_gpt_oss("commentaryI should call the weather tool.");
        assert_eq!(answer, "");
        assert_eq!(
            reasoning,
            Some("I should call the weather tool.".to_owned())
        );
    }

    #[test]
    fn harmony_filter_streaming_matches_live_token_boundaries() {
        // Live-verified SSE delta shape for the same prompt: "analysis" and
        // the transition's "assistant"/"final" each arrive as their own
        // whole-word token event, not glued or sub-word-split. Concatenated,
        // these tokens reproduce HARMONY_LIVE_SAMPLE byte-for-byte (asserted
        // below) — this is the streaming-granularity view of that same
        // captured response, not an independent live run.
        let tokens = [
            "analysis",
            "The",
            " user",
            " asks",
            " \"",
            "What",
            " is",
            " ",
            "2",
            "+",
            "2",
            "?\"",
            " They",
            " want",
            " a",
            " brief",
            " answer",
            ".",
            " Simple",
            ".\n\n",
            "We",
            " can",
            " respond",
            " with",
            " \"",
            "4",
            "\"",
            ".\n\n",
            "Thus",
            " reply",
            ".",
            " Ensure",
            " minimal",
            " output",
            ".\n\n",
            "assistant",
            "final",
            "4",
        ];
        assert_eq!(
            tokens.concat(),
            HARMONY_LIVE_SAMPLE,
            "token boundaries must reassemble to the exact live-captured sample"
        );
        let mut f = HarmonyFilter::new();
        let mut reasoning = String::new();
        let mut content = String::new();
        for tok in tokens {
            for piece in f.process(tok) {
                match piece {
                    ThinkPiece::Reasoning(s) => reasoning.push_str(&s),
                    ThinkPiece::Content(s) => content.push_str(&s),
                }
            }
        }
        for piece in f.flush() {
            match piece {
                ThinkPiece::Reasoning(s) => reasoning.push_str(&s),
                ThinkPiece::Content(s) => content.push_str(&s),
            }
        }
        assert_eq!(content, "4");
        assert!(reasoning.starts_with("The user asks"));
        assert!(reasoning.trim_end().ends_with("minimal output."));
        assert!(!reasoning.contains("assistantfinal"));
        assert!(!reasoning.contains("assistant"));
    }

    #[test]
    fn harmony_filter_transition_marker_split_across_many_tokens() {
        // Worst-case token granularity: the compound marker itself split
        // into single-character pieces across the boundary. "Thinking"
        // (capitalized, like real model output starting a new sentence) —
        // lowercase would fail the leading-word boundary guard by design
        // (see `match_leading_channel_word`'s doc comment).
        let mut f = HarmonyFilter::new();
        let mut pieces = Vec::new();
        for tok in [
            "analysis", "Thinking", "a", "s", "s", "i", "s", "t", "a", "n", "t", "f", "i", "n",
            "a", "l", "answer",
        ] {
            pieces.extend(f.process(tok));
        }
        pieces.extend(f.flush());
        let reasoning: String = pieces
            .iter()
            .filter_map(|p| match p {
                ThinkPiece::Reasoning(s) => Some(s.as_str()),
                ThinkPiece::Content(_) => None,
            })
            .collect();
        let content: String = pieces
            .iter()
            .filter_map(|p| match p {
                ThinkPiece::Content(s) => Some(s.as_str()),
                ThinkPiece::Reasoning(_) => None,
            })
            .collect();
        assert_eq!(reasoning, "Thinking");
        assert_eq!(content, "answer");
    }

    #[test]
    fn harmony_filter_reasoning_prose_containing_channel_words_not_misdetected() {
        // "final analysis" and "commentary" as ordinary reasoning prose, with
        // no preceding "assistant" glue, must not trigger a false transition.
        let (reasoning, answer) = extract_reasoning_gpt_oss(
            "analysisIn the final analysis, the commentary track was unhelpful.\
            assistantfinalDone.",
        );
        assert_eq!(answer, "Done.");
        assert_eq!(
            reasoning,
            Some("In the final analysis, the commentary track was unhelpful.".to_owned())
        );
    }

    #[test]
    fn harmony_filter_empty_token_produces_nothing() {
        let mut f = HarmonyFilter::new();
        assert!(f.process("").is_empty());
    }

    #[test]
    fn harmony_filter_flush_empty_is_empty() {
        let mut f = HarmonyFilter::new();
        assert!(f.flush().is_empty());
    }
}
