// ============================================================
// src/thinking_budget.rs — per-request thinking control (0.9.1)
// ============================================================
// Three request controls decide whether a reasoning model thinks and for how
// long, plus one per-model default:
//
//   * `thinking_budget_tokens` (alias `thinking_token_budget`, vLLM's name):
//     an integer >= 0. 0 = thinking off.
//   * `enable_thinking` (top level, or in `chat_template_kwargs`): the
//     template switch.
//   * `reasoning_effort` (top level, or in `chat_template_kwargs`):
//     `none` / `minimal` = off, `low` = 1,024, `medium` = 4,096,
//     `high` (aliases `xhigh`, `max`) = 16,384 thinking tokens.
//   * the model's `thinking_budget_tokens` from config.json (null = none).
//
// The switch: an explicit off (budget 0, `enable_thinking: false`, effort
// `none`/`minimal`) always wins; else an explicit on; else the model default's
// 0; else the template's own default. The budget, only when thinking is not
// off: request budget > effort preset > model default (see [`resolve`]). A
// budget limits thinking that is on; it never switches it on against an
// explicit `false`.
//
// The budget is STRICT: when a request has spent it inside its reasoning
// block, the engine stops the request (its usual early-stop path) and the
// response ends with `finish_reason: "length"` — normally with no answer. It
// is clamped below `max_tokens` by [`answer_reserve`] (min(512, max_tokens / 2)),
// so a budget can never be the whole generation (exact on text models; VLM /
// NPU count chunks, see [`chunk_budget`]).
//
// [`BudgetTracker`] is the running detector the engines feed, delta by delta,
// on the engine thread. It follows the same reasoning split as the usage count
// (`crate::prompt_builder::ReasoningSplit::reasoning_spans`, markers included)
// for the FIRST reasoning block: a natural close (`</think>`, or gpt-oss's
// `assistantfinal`) before the budget ends counting for the request. On gpt-oss
// only the analysis channel is budgeted: counting also ends at the first
// switch to `commentary` (the tool-call channel), and an output that opens on
// `commentary` is never counted, so a tool call is never cut by the budget.
// ============================================================

use std::collections::VecDeque;

use crate::model_manager::ReasoningParser;
use crate::prompt_builder::{
    HARMONY_FINAL_MARKER, HARMONY_LEAD_DECISION_BYTES, ReasoningSplit, THINK_CLOSE_TAG,
    THINK_OPEN_TAG, TemplateKwargs, harmony_lead_still_open, harmony_reasoning_lead_len,
};

/// The most tokens of `max_tokens` kept free for the answer (see [`answer_reserve`]). 512 tokens
/// holds a short answer; a request that wants a longer one after a long think raises `max_tokens`.
pub const ANSWER_RESERVE_TOKENS: usize = 512;

/// The tokens of `max_new_tokens` kept free for the answer: `min(512, max_new_tokens / 2)`, so a
/// short request keeps half for thinking instead of losing thinking altogether. The effective
/// budget is `min(budget, max_new_tokens - answer_reserve(max_new_tokens))`.
#[must_use]
pub fn answer_reserve(max_new_tokens: usize) -> usize {
    ANSWER_RESERVE_TOKENS.min(max_new_tokens / 2)
}

/// Measured tokens per streamer chunk on the VLM engine, in hundredths (1.37): one VLM
/// generation delivered 6,425 tokens in 4,678 streamer callbacks (the VLM tail's note on the
/// pipeline count, `vlm_engine.rs` "Block B2c"). One measurement: approximate until measured on
/// more models.
pub const VLM_TOKENS_PER_CHUNK_PERCENT: u64 = 137;

/// Measured tokens per streamer chunk on the NPU engine, in hundredths (1.04): 200 tokens in 192
/// chunks (`npu_engine.rs`, the `NpuStream::acct` note). One measurement: approximate until
/// measured on more models.
pub const NPU_TOKENS_PER_CHUNK_PERCENT: u64 = 104;

/// The chunk budget of a VLM / NPU request with a `tokens` budget, for an engine that averages
/// `tokens_per_chunk_percent` / 100 tokens a chunk: `ceil(tokens * 100 / percent)`, at least 1.
#[must_use]
pub fn chunk_budget(tokens: u32, tokens_per_chunk_percent: u64) -> usize {
    let chunks = (u64::from(tokens) * 100).div_ceil(tokens_per_chunk_percent.max(1));
    usize::try_from(chunks).unwrap_or(usize::MAX).max(1)
}

/// A request's `reasoning_effort` (`OpenAI` values).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    /// Thinking off.
    None,
    /// Thinking off (no level below "low" on a binary switch).
    Minimal,
    /// Thinking on, 1,024-token budget.
    Low,
    /// Thinking on, 4,096-token budget.
    Medium,
    /// Thinking on, 16,384-token budget (also `xhigh` and `max`, which some clients send).
    High,
}

impl ReasoningEffort {
    /// The accepted values, as the 400 lists them.
    pub const ACCEPTED: &'static str = "none, minimal, low, medium, high, xhigh, max";

    /// The effort named `s` (exact, lower case), or `None` for any other word.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "none" => Some(Self::None),
            "minimal" => Some(Self::Minimal),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" | "xhigh" | "max" => Some(Self::High),
            _ => None,
        }
    }

    /// The `OpenAI` word for this effort.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    /// `false` for `none` / `minimal`.
    #[must_use]
    pub fn thinking_on(self) -> bool {
        !matches!(self, Self::None | Self::Minimal)
    }

    /// The preset thinking budget; `None` for the two "off" values.
    #[must_use]
    pub fn preset_budget(self) -> Option<u32> {
        match self {
            Self::None | Self::Minimal => None,
            Self::Low => Some(1_024),
            Self::Medium => Some(4_096),
            Self::High => Some(16_384),
        }
    }
}

/// A request's thinking controls, parsed and type-checked; no model knowledge yet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ThinkingRequest {
    /// `thinking_budget_tokens` / `thinking_token_budget`.
    pub budget: Option<u32>,
    /// `enable_thinking` (top level wins over `chat_template_kwargs`).
    pub enable_thinking: Option<bool>,
    /// `reasoning_effort` (top level wins over `chat_template_kwargs`).
    pub effort: Option<ReasoningEffort>,
    /// `chat_template_kwargs.preserve_thinking`.
    pub preserve_thinking: Option<bool>,
}

/// A request field that failed its check: the 400 the handler sends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThinkingParamError {
    /// Client-facing message.
    pub message: String,
    /// `error.code`.
    pub code: &'static str,
    /// `error.param` (the field's path).
    pub param: &'static str,
}

/// The raw request fields [`parse_request`] reads.
#[derive(Debug, Clone, Copy, Default)]
pub struct RawThinkingFields<'a> {
    /// Top-level `enable_thinking` (already typed by serde).
    pub enable_thinking: Option<bool>,
    /// Top-level `reasoning_effort`.
    pub reasoning_effort: Option<&'a serde_json::Value>,
    /// Top-level `thinking_budget_tokens`.
    pub thinking_budget_tokens: Option<&'a serde_json::Value>,
    /// Top-level `thinking_token_budget` (vLLM alias).
    pub thinking_token_budget: Option<&'a serde_json::Value>,
    /// `chat_template_kwargs`.
    pub kwargs: Option<&'a serde_json::Map<String, serde_json::Value>>,
}

/// The `chat_template_kwargs` keys that reach the template; every other key is ignored.
pub const ALLOWED_TEMPLATE_KWARGS: [&str; 3] =
    ["enable_thinking", "reasoning_effort", "preserve_thinking"];

/// Error code: a `reasoning_effort` that is not one of [`ReasoningEffort::ACCEPTED`].
pub const CODE_INVALID_EFFORT: &str = "invalid_reasoning_effort";
/// Error code: a thinking budget that is not an integer in `0..=4294967295`.
pub const CODE_INVALID_BUDGET: &str = "invalid_thinking_budget";
/// Error code: `thinking_budget_tokens` and `thinking_token_budget` both set, to different values.
pub const CODE_CONFLICTING_BUDGET: &str = "conflicting_thinking_budget";
/// Error code (existing): a `chat_template_kwargs` switch that is not a boolean.
pub const CODE_INVALID_KWARGS: &str = "invalid_chat_template_kwargs";

fn effort_from(
    v: Option<&serde_json::Value>,
    param: &'static str,
) -> Result<Option<ReasoningEffort>, ThinkingParamError> {
    match v {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(s)) => {
            ReasoningEffort::parse(s)
                .map(Some)
                .ok_or_else(|| ThinkingParamError {
                    message: format!("{param} must be one of: {}", ReasoningEffort::ACCEPTED),
                    code: CODE_INVALID_EFFORT,
                    param,
                })
        }
        Some(_) => Err(ThinkingParamError {
            message: format!(
                "{param} must be a string, one of: {}",
                ReasoningEffort::ACCEPTED
            ),
            code: CODE_INVALID_EFFORT,
            param,
        }),
    }
}

fn budget_from(
    v: Option<&serde_json::Value>,
    param: &'static str,
) -> Result<Option<u32>, ThinkingParamError> {
    match v {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Number(n)) => n
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .map(Some)
            .ok_or_else(|| budget_error(param)),
        Some(_) => Err(budget_error(param)),
    }
}

fn budget_error(param: &'static str) -> ThinkingParamError {
    ThinkingParamError {
        message: format!("{param} must be an integer between 0 and 4294967295"),
        code: CODE_INVALID_BUDGET,
        param,
    }
}

fn kwarg_bool(
    kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
    key: &str,
    param: &'static str,
) -> Result<Option<bool>, ThinkingParamError> {
    match kwargs.and_then(|k| k.get(key)) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(ThinkingParamError {
            message: format!("{param} must be a boolean"),
            code: CODE_INVALID_KWARGS,
            param,
        }),
    }
}

/// Type-check and combine a request's thinking fields. `null` is the same as absent everywhere.
///
/// # Errors
/// The first field that fails: a `reasoning_effort` outside the accepted values or not a string
/// (`invalid_reasoning_effort`); a budget that is not an integer in `0..=u32::MAX`
/// (`invalid_thinking_budget`); both budget names set to different values
/// (`conflicting_thinking_budget`); a kwargs `enable_thinking` / `preserve_thinking` that is not
/// a boolean (`invalid_chat_template_kwargs`).
pub fn parse_request(raw: RawThinkingFields<'_>) -> Result<ThinkingRequest, ThinkingParamError> {
    let kw_enable = kwarg_bool(
        raw.kwargs,
        "enable_thinking",
        "chat_template_kwargs.enable_thinking",
    )?;
    let kw_preserve = kwarg_bool(
        raw.kwargs,
        "preserve_thinking",
        "chat_template_kwargs.preserve_thinking",
    )?;
    let kw_effort = effort_from(
        raw.kwargs.and_then(|k| k.get("reasoning_effort")),
        "chat_template_kwargs.reasoning_effort",
    )?;
    let effort = effort_from(raw.reasoning_effort, "reasoning_effort")?;
    let budget = budget_from(raw.thinking_budget_tokens, "thinking_budget_tokens")?;
    let alias = budget_from(raw.thinking_token_budget, "thinking_token_budget")?;
    let budget = match (budget, alias) {
        (Some(a), Some(b)) if a != b => {
            return Err(ThinkingParamError {
                message: format!(
                    "thinking_budget_tokens ({a}) and thinking_token_budget ({b}) are two names \
                     for one setting: send one, or the same value in both"
                ),
                code: CODE_CONFLICTING_BUDGET,
                param: "thinking_token_budget",
            });
        }
        (a, b) => a.or(b),
    };
    Ok(ThinkingRequest {
        budget,
        enable_thinking: raw.enable_thinking.or(kw_enable),
        effort: effort.or(kw_effort),
        preserve_thinking: kw_preserve,
    })
}

/// The `chat_template_kwargs` keys this server does not read (everything outside
/// [`ALLOWED_TEMPLATE_KWARGS`]), for the ignored-keys log.
#[must_use]
pub fn ignored_kwargs(kwargs: &serde_json::Map<String, serde_json::Value>) -> Vec<&str> {
    kwargs
        .keys()
        .map(String::as_str)
        .filter(|k| !ALLOWED_TEMPLATE_KWARGS.contains(k))
        .collect()
}

/// What one request does about thinking on one model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ThinkingPlan {
    /// The template variables (server-built).
    pub template: TemplateKwargs,
    /// The strict budget to arm on the engine, already clamped below `max_tokens`; `None` = no
    /// budget (thinking off, or unlimited).
    pub budget: Option<u32>,
}

/// Resolve `req` against the model's configured default and the request's final
/// `max_new_tokens` (`0` = unbounded: no clamp).
///
/// - **Switch** (template `enable_thinking`): an explicit off — request budget `0`,
///   `enable_thinking: false`, `reasoning_effort` `none`/`minimal` — always wins; else an
///   explicit on (budget `> 0`, `enable_thinking: true`, an "on" effort); else the model
///   default's `0` (off; a positive default does not touch the switch); else left undefined (the
///   template's own default). A budget only limits thinking that is on.
/// - **Budget**, only when the switch is not off, first source that states one: request budget,
///   the `reasoning_effort` preset, the model default; else none (unlimited).
/// - **Clamp**: `min(budget, max_new_tokens - answer_reserve(max_new_tokens))`; `<= 0` turns
///   thinking off (only reachable with no room at all).
/// - **`reasoning_effort` variable** (gpt-oss): `"none"` when thinking is off, the request's
///   effort word when thinking is on and that word is an "on" level, else undefined.
#[must_use]
pub fn resolve(
    req: &ThinkingRequest,
    model_default: Option<u32>,
    max_new_tokens: usize,
) -> ThinkingPlan {
    let explicit_off = req.budget == Some(0)
        || req.enable_thinking == Some(false)
        || req.effort.is_some_and(|e| !e.thinking_on());
    let explicit_on = req.budget.is_some_and(|b| b > 0)
        || req.enable_thinking == Some(true)
        || req.effort.is_some_and(ReasoningEffort::thinking_on);
    let switch = if explicit_off {
        Some(false)
    } else if explicit_on {
        Some(true)
    } else {
        (model_default == Some(0)).then_some(false)
    };
    let budget = if switch == Some(false) {
        None
    } else {
        req.budget
            .filter(|b| *b > 0)
            .or_else(|| req.effort.and_then(ReasoningEffort::preset_budget))
            .or_else(|| model_default.filter(|b| *b > 0))
    };
    let mut enable_thinking = switch;
    let mut armed = None;
    if let Some(b) = budget {
        let b = usize::try_from(b).unwrap_or(usize::MAX);
        let effective = if max_new_tokens == 0 {
            b
        } else {
            b.min(max_new_tokens.saturating_sub(answer_reserve(max_new_tokens)))
        };
        // Defensive: `b > 0` here and `max_new_tokens - answer_reserve(max_new_tokens)` is at
        // least half of a non-zero `max_new_tokens`, so `effective` cannot be 0 today; kept so a
        // future reserve rule that leaves no room turns thinking off instead of arming a 0.
        if effective == 0 {
            enable_thinking = Some(false);
        } else {
            armed = Some(u32::try_from(effective).unwrap_or(u32::MAX));
        }
    }
    let reasoning_effort = match enable_thinking {
        Some(false) => Some("none"),
        Some(true) => req
            .effort
            .filter(|e| e.thinking_on())
            .map(ReasoningEffort::as_str),
        None => None,
    };
    ThinkingPlan {
        template: TemplateKwargs {
            enable_thinking,
            reasoning_effort,
            preserve_thinking: req.preserve_thinking,
        },
        budget: armed,
    }
}

// ---- Running detector -------------------------------------------------

/// What the engine does after a delta.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetStep {
    /// Keep generating.
    Continue,
    /// The budget is spent inside the reasoning block: stop this request now.
    Stop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Not in reasoning yet (waiting for `<think>`, or for gpt-oss's lead channel word).
    Before,
    /// Inside the first reasoning block: counting.
    Inside,
    /// The first block closed (or the output has none): the budget no longer applies.
    Done,
    /// The budget was spent: the request is being stopped; every later delta answers `Stop`.
    Stopped,
}

/// The largest char boundary of `s` at or below `at`.
fn floor_boundary(s: &str, mut at: usize) -> usize {
    at = at.min(s.len());
    while !s.is_char_boundary(at) {
        at -= 1;
    }
    at
}

/// The running thinking-budget detector of one request, fed every non-empty delta in output
/// order with the engine's token count for it (CB: exact; VLM / NPU: `1` per text chunk, a lower
/// bound — see the module docs and INTERFACE.md).
///
/// A delta counts as reasoning when any of its bytes lies in the first reasoning span, markers
/// included — the usage count's rule — so on CB the stop lands at the same figure
/// `usage.completion_tokens_details.reasoning_tokens` reports. Memory is bounded: only a marker's
/// length of text is kept (gpt-oss: the first few bytes until the lead word is decided).
#[derive(Debug)]
pub struct BudgetTracker {
    budget: usize,
    harmony: bool,
    phase: Phase,
    count: usize,
    /// Text not yet ruled out as part of a marker (see [`Self::push`]).
    window: String,
    /// `(end offset in window, tokens)` of the deltas that make up `window`, in order (`Before`).
    segments: VecDeque<(usize, usize)>,
    /// The stop was put off once because the text ended in a partial close marker.
    deferred: bool,
}

/// gpt-oss's switch to the tool-call channel: the analysis budget ends here.
const HARMONY_COMMENTARY_MARKER: &str = "assistantcommentary";

impl BudgetTracker {
    /// A tracker for a request whose output splits as `split`, with `budget` (>= 1) thinking
    /// tokens.
    #[must_use]
    pub fn new(split: ReasoningSplit, budget: usize) -> Self {
        let harmony = split.parser == Some(ReasoningParser::GptOss);
        let phase = if !harmony && split.starts_in_thinking {
            Phase::Inside
        } else {
            Phase::Before
        };
        Self {
            budget,
            harmony,
            phase,
            count: 0,
            window: String::new(),
            segments: VecDeque::new(),
            deferred: false,
        }
    }

    /// `true` while the output is inside its first reasoning block: an engine `length` stop now
    /// is a budget stop too (the generation ran out inside reasoning, so no answer came).
    #[must_use]
    pub fn inside(&self) -> bool {
        matches!(self.phase, Phase::Inside | Phase::Stopped)
    }

    /// `true` once [`Self::push`] has answered [`BudgetStep::Stop`].
    #[must_use]
    pub fn stopped(&self) -> bool {
        self.phase == Phase::Stopped
    }

    /// Reasoning tokens counted so far (first block only).
    #[must_use]
    pub fn counted(&self) -> usize {
        self.count
    }

    /// Feed one delta of `tokens` engine tokens.
    pub fn push(&mut self, delta: &str, tokens: usize) -> BudgetStep {
        if self.phase == Phase::Stopped {
            return BudgetStep::Stop;
        }
        if delta.is_empty() || self.phase == Phase::Done {
            return BudgetStep::Continue;
        }
        if self.harmony {
            self.push_harmony(delta, tokens)
        } else {
            self.push_think(delta, tokens)
        }
    }

    fn verdict(&mut self) -> BudgetStep {
        if self.phase == Phase::Inside && self.count >= self.budget {
            // The text may be in the middle of its close marker (`</th` …): wait one delta,
            // once, so a model that is closing on the edge is not cut.
            if !self.deferred && self.ends_in_partial_close() {
                self.deferred = true;
                return BudgetStep::Continue;
            }
            self.phase = Phase::Stopped;
            self.window.clear();
            BudgetStep::Stop
        } else {
            BudgetStep::Continue
        }
    }

    /// `true` when the window ends in a proper prefix of a close marker (`</think>`; gpt-oss: two
    /// bytes or more of `assistantfinal` / `assistantcommentary`, ASCII case ignored).
    fn ends_in_partial_close(&self) -> bool {
        let w = self.window.as_bytes();
        let (markers, min): (&[&str], usize) = if self.harmony {
            (&[HARMONY_FINAL_MARKER, HARMONY_COMMENTARY_MARKER], 2)
        } else {
            (&[THINK_CLOSE_TAG], 1)
        };
        markers.iter().any(|m| {
            let m = m.as_bytes();
            (min..m.len()).any(|k| k <= w.len() && w[w.len() - k..].eq_ignore_ascii_case(&m[..k]))
        })
    }

    /// Keep the last `keep` bytes of `text[from..]` (char-safe) as the window.
    fn keep_tail(&mut self, text: &str, from: usize, keep: usize) {
        let start = floor_boundary(text, text.len().saturating_sub(keep).max(from));
        text[start.max(from)..].clone_into(&mut self.window);
    }

    fn push_think(&mut self, delta: &str, tokens: usize) -> BudgetStep {
        let mut text = std::mem::take(&mut self.window);
        text.push_str(delta);
        let mut close_from = 0;
        if self.phase == Phase::Before {
            self.segments.push_back((text.len(), tokens));
            if let Some(i) = text.find(THINK_OPEN_TAG) {
                // Every delta with a byte at or after the tag's start is in the span.
                self.count += self
                    .segments
                    .iter()
                    .filter(|(end, _)| *end > i)
                    .map(|(_, n)| n)
                    .sum::<usize>();
                self.segments.clear();
                self.phase = Phase::Inside;
                close_from = i + THINK_OPEN_TAG.len();
            } else {
                let keep = THINK_OPEN_TAG.len() - 1;
                let cut = floor_boundary(&text, text.len().saturating_sub(keep));
                while self.segments.front().is_some_and(|(end, _)| *end <= cut) {
                    self.segments.pop_front();
                }
                for seg in &mut self.segments {
                    seg.0 -= cut;
                }
                text[cut..].clone_into(&mut self.window);
                return BudgetStep::Continue;
            }
        } else {
            self.count += tokens;
        }
        if text[close_from..].contains(THINK_CLOSE_TAG) {
            self.phase = Phase::Done;
            self.window.clear();
            return BudgetStep::Continue;
        }
        self.keep_tail(&text, close_from, THINK_CLOSE_TAG.len() - 1);
        self.verdict()
    }

    fn push_harmony(&mut self, delta: &str, tokens: usize) -> BudgetStep {
        let mut text = std::mem::take(&mut self.window);
        text.push_str(delta);
        let mut marker_from = 0;
        if self.phase == Phase::Before {
            self.segments.push_back((text.len(), tokens));
            if text.len() < HARMONY_LEAD_DECISION_BYTES && harmony_lead_still_open(&text) {
                self.window = text;
                return BudgetStep::Continue;
            }
            // No reasoning lead (`final`, plain text), or a `commentary` lead (a tool call): not
            // budgeted.
            let Some(lead) = harmony_reasoning_lead_len(&text)
                .filter(|&n| !text[..n].eq_ignore_ascii_case("commentary"))
            else {
                self.phase = Phase::Done;
                self.segments.clear();
                return BudgetStep::Continue;
            };
            // The span starts at byte 0: every delta so far is in it.
            self.count += self.segments.iter().map(|(_, n)| n).sum::<usize>();
            self.segments.clear();
            self.phase = Phase::Inside;
            marker_from = lead;
        } else {
            self.count += tokens;
        }
        // ASCII lowercasing keeps byte offsets. `assistantfinal` closes the analysis; the switch
        // to `commentary` (a tool call) ends the budgeted part too.
        let lower = text[marker_from..].to_ascii_lowercase();
        if lower.contains(HARMONY_FINAL_MARKER) || lower.contains(HARMONY_COMMENTARY_MARKER) {
            self.phase = Phase::Done;
            self.window.clear();
            return BudgetStep::Continue;
        }
        self.keep_tail(&text, marker_from, HARMONY_COMMENTARY_MARKER.len() - 1);
        self.verdict()
    }
}

/// Where a request's armed budget ended, for the engines' finish tails (VLM, NPU).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BudgetOutcome {
    /// The request had a budget.
    pub armed: bool,
    /// The engine stopped it at the budget.
    pub stopped: bool,
    /// The output ended inside its first reasoning block.
    pub inside: bool,
}

impl BudgetOutcome {
    /// The outcome of `tracker` (`None`: no budget). A request its client stopped
    /// (`client_stopped`) is reported as unarmed: its finish is nobody's and stays as it is.
    #[must_use]
    pub fn of(tracker: Option<&BudgetTracker>, client_stopped: bool) -> Self {
        tracker
            .filter(|_| !client_stopped)
            .map_or_else(Self::default, |t| Self {
                armed: true,
                stopped: t.stopped(),
                inside: t.inside(),
            })
    }

    /// The finish to report for a request that ended with `finish`:
    /// [`crate::ov_cb::FinishReason::with_budget`] when a budget was armed, else `finish` unchanged.
    #[must_use]
    pub fn finish(self, finish: crate::ov_cb::FinishReason) -> crate::ov_cb::FinishReason {
        if self.armed {
            finish.with_budget(self.stopped, self.inside)
        } else {
            finish
        }
    }
}

/// The tracker for a CB request (exact token counts): armed only when the plan carries a budget.
#[must_use]
pub fn tracker_for(split: ReasoningSplit, budget: Option<u32>) -> Option<BudgetTracker> {
    budget.map(|b| BudgetTracker::new(split, usize::try_from(b).unwrap_or(usize::MAX)))
}

/// The tracker for a VLM / NPU request, fed one per streamer chunk: the token budget scaled to
/// chunks by [`chunk_budget`] with the engine's `tokens_per_chunk_percent`
/// ([`VLM_TOKENS_PER_CHUNK_PERCENT`] / [`NPU_TOKENS_PER_CHUNK_PERCENT`]).
#[must_use]
pub fn chunk_tracker_for(
    split: ReasoningSplit,
    budget: Option<u32>,
    tokens_per_chunk_percent: u64,
) -> Option<BudgetTracker> {
    budget.map(|b| BudgetTracker::new(split, chunk_budget(b, tokens_per_chunk_percent)))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use serde_json::json;

    fn parse(v: &serde_json::Value) -> Result<ThinkingRequest, ThinkingParamError> {
        parse_request(RawThinkingFields {
            enable_thinking: v
                .get("enable_thinking")
                .and_then(serde_json::Value::as_bool),
            reasoning_effort: v.get("reasoning_effort"),
            thinking_budget_tokens: v.get("thinking_budget_tokens"),
            thinking_token_budget: v.get("thinking_token_budget"),
            kwargs: v
                .get("chat_template_kwargs")
                .and_then(serde_json::Value::as_object),
        })
    }

    fn plan(v: &serde_json::Value, model: Option<u32>, max: usize) -> ThinkingPlan {
        resolve(&parse(v).unwrap(), model, max)
    }

    // ---- effort mapping ----

    #[test]
    fn effort_values_map_to_switch_and_presets() {
        for (word, on, budget) in [
            ("none", false, None),
            ("minimal", false, None),
            ("low", true, Some(1_024)),
            ("medium", true, Some(4_096)),
            ("high", true, Some(16_384)),
            ("xhigh", true, Some(16_384)),
            ("max", true, Some(16_384)),
        ] {
            let p = plan(&json!({"reasoning_effort": word}), None, 0);
            assert_eq!(p.template.enable_thinking, Some(on), "{word}");
            assert_eq!(p.budget, budget, "{word}");
            // The aliases reach the template as `high` (gpt-oss knows low / medium / high).
            let word_var = if matches!(word, "xhigh" | "max") {
                "high"
            } else {
                word
            };
            let effort_var = if on { Some(word_var) } else { Some("none") };
            assert_eq!(p.template.reasoning_effort, effort_var, "{word}");
            // The kwargs form is the same control.
            let k = plan(
                &json!({"chat_template_kwargs": {"reasoning_effort": word}}),
                None,
                0,
            );
            assert_eq!(k, p, "{word} via kwargs");
        }
    }

    #[test]
    fn unknown_or_mistyped_effort_is_a_400_listing_the_values() {
        for v in [
            json!("ultra"),
            json!("MAX"),
            json!("High"),
            json!(""),
            json!(3),
            json!(true),
        ] {
            let e = parse(&json!({"reasoning_effort": v})).unwrap_err();
            assert_eq!(e.code, CODE_INVALID_EFFORT);
            assert_eq!(e.param, "reasoning_effort");
            assert!(
                e.message
                    .contains("none, minimal, low, medium, high, xhigh, max"),
                "{e:?}"
            );
        }
        let e = parse(&json!({"chat_template_kwargs": {"reasoning_effort": "ultra"}})).unwrap_err();
        assert_eq!(e.param, "chat_template_kwargs.reasoning_effort");
        // null = absent.
        assert_eq!(
            parse(&json!({"reasoning_effort": null})).unwrap().effort,
            None
        );
    }

    #[test]
    fn kwargs_switches_are_type_checked() {
        let e = parse(&json!({"chat_template_kwargs": {"enable_thinking": "false"}})).unwrap_err();
        assert_eq!(
            (e.code, e.param),
            (CODE_INVALID_KWARGS, "chat_template_kwargs.enable_thinking")
        );
        let e = parse(&json!({"chat_template_kwargs": {"preserve_thinking": 1}})).unwrap_err();
        assert_eq!(
            (e.code, e.param),
            (
                CODE_INVALID_KWARGS,
                "chat_template_kwargs.preserve_thinking"
            )
        );
        let r = parse(&json!({"chat_template_kwargs": {"preserve_thinking": true}})).unwrap();
        assert_eq!(r.preserve_thinking, Some(true));
        assert_eq!(
            plan(
                &json!({"chat_template_kwargs": {"preserve_thinking": false}}),
                None,
                0
            )
            .template
            .preserve_thinking,
            Some(false)
        );
    }

    #[test]
    fn top_level_wins_over_kwargs() {
        let r = parse(&json!({
            "enable_thinking": true, "reasoning_effort": "low",
            "chat_template_kwargs": {"enable_thinking": false, "reasoning_effort": "high"}
        }))
        .unwrap();
        assert_eq!(r.enable_thinking, Some(true));
        assert_eq!(r.effort, Some(ReasoningEffort::Low));
    }

    #[test]
    fn only_the_allowlisted_kwargs_are_read() {
        let k = json!({"enable_thinking": true, "reasoning_effort": "low",
            "preserve_thinking": true, "thinking": true, "bos_token": "X",
            "thinking_budget_tokens": 5});
        let k = k.as_object().unwrap();
        let mut ignored = ignored_kwargs(k);
        ignored.sort_unstable();
        assert_eq!(
            ignored,
            vec!["bos_token", "thinking", "thinking_budget_tokens"]
        );
        // A budget inside kwargs is not a budget.
        assert_eq!(
            parse(&json!({"chat_template_kwargs": {"thinking_budget_tokens": 5}}))
                .unwrap()
                .budget,
            None
        );
    }

    // ---- budget fields and alias ----

    #[test]
    fn budget_fields_type_and_range() {
        for v in [
            json!(-1),
            json!(1.5),
            json!("4096"),
            json!(4_294_967_296_u64),
            json!([1]),
        ] {
            for name in ["thinking_budget_tokens", "thinking_token_budget"] {
                let e = parse(&json!({ name: v })).unwrap_err();
                assert_eq!((e.code, e.param), (CODE_INVALID_BUDGET, name), "{v}");
            }
        }
        assert_eq!(
            parse(&json!({"thinking_budget_tokens": 4_294_967_295_u64}))
                .unwrap()
                .budget,
            Some(u32::MAX)
        );
        assert_eq!(
            parse(&json!({"thinking_budget_tokens": null}))
                .unwrap()
                .budget,
            None
        );
    }

    #[test]
    fn alias_rules() {
        assert_eq!(
            parse(&json!({"thinking_token_budget": 300}))
                .unwrap()
                .budget,
            Some(300)
        );
        assert_eq!(
            parse(&json!({"thinking_budget_tokens": 300, "thinking_token_budget": 300}))
                .unwrap()
                .budget,
            Some(300)
        );
        let e = parse(&json!({"thinking_budget_tokens": 300, "thinking_token_budget": 200}))
            .unwrap_err();
        assert_eq!(
            (e.code, e.param),
            (CODE_CONFLICTING_BUDGET, "thinking_token_budget")
        );
        // null on one side is "not set", not a conflict.
        assert_eq!(
            parse(&json!({"thinking_budget_tokens": null, "thinking_token_budget": 200}))
                .unwrap()
                .budget,
            Some(200)
        );
    }

    // ---- precedence table (every row of CONFIG.md's table) ----

    /// (request, model default, -> `enable_thinking`, effort variable, budget).
    type Row = (
        serde_json::Value,
        Option<u32>,
        Option<bool>,
        Option<&'static str>,
        Option<u32>,
    );

    #[test]
    #[allow(clippy::too_many_lines)]
    fn precedence_table() {
        // max_tokens 0 (unbounded) so the clamp does not interfere.
        let rows: Vec<Row> = vec![
            // 1. nothing set: template default, no budget.
            (json!({}), None, None, None, None),
            // 2. model default only: budget armed, switch untouched.
            (json!({}), Some(2048), None, None, Some(2048)),
            // 3. model default 0: off.
            (json!({}), Some(0), Some(false), Some("none"), None),
            // 4. effort beats the model default (value and switch).
            (
                json!({"reasoning_effort": "low"}),
                Some(8000),
                Some(true),
                Some("low"),
                Some(1024),
            ),
            (
                json!({"reasoning_effort": "none"}),
                Some(8000),
                Some(false),
                Some("none"),
                None,
            ),
            (
                json!({"reasoning_effort": "high"}),
                Some(0),
                Some(true),
                Some("high"),
                Some(16384),
            ),
            // 5. an explicit off always wins the switch (owner ruling); with thinking on, the
            //    budget comes from the next source that has one.
            (
                json!({"enable_thinking": false, "reasoning_effort": "high"}),
                None,
                Some(false),
                Some("none"),
                None,
            ),
            (
                json!({"enable_thinking": true, "reasoning_effort": "none"}),
                Some(3000),
                Some(false),
                Some("none"),
                None,
            ),
            (
                json!({"enable_thinking": true, "reasoning_effort": "minimal"}),
                None,
                Some(false),
                Some("none"),
                None,
            ),
            (
                json!({"enable_thinking": true, "reasoning_effort": "medium"}),
                Some(100),
                Some(true),
                Some("medium"),
                Some(4096),
            ),
            (
                json!({"enable_thinking": true}),
                Some(0),
                Some(true),
                None,
                None,
            ),
            (
                json!({"enable_thinking": false}),
                Some(2048),
                Some(false),
                Some("none"),
                None,
            ),
            // 6. the request budget beats every other budget, but never an explicit off.
            (
                json!({"thinking_budget_tokens": 500, "enable_thinking": false}),
                None,
                Some(false),
                Some("none"),
                None,
            ),
            (
                json!({"thinking_budget_tokens": 500, "reasoning_effort": "high"}),
                Some(9),
                Some(true),
                Some("high"),
                Some(500),
            ),
            (
                json!({"thinking_budget_tokens": 500, "reasoning_effort": "none"}),
                None,
                Some(false),
                Some("none"),
                None,
            ),
            (
                json!({"thinking_budget_tokens": 0, "enable_thinking": true}),
                Some(2048),
                Some(false),
                Some("none"),
                None,
            ),
            (
                json!({"thinking_budget_tokens": 0, "reasoning_effort": "high"}),
                None,
                Some(false),
                Some("none"),
                None,
            ),
            (
                json!({"thinking_token_budget": 700}),
                Some(0),
                Some(true),
                None,
                Some(700),
            ),
        ];
        for (i, (req, model, enable, effort, budget)) in rows.into_iter().enumerate() {
            let p = plan(&req, model, 0);
            assert_eq!(
                p.template.enable_thinking, enable,
                "row {i}: {req} model {model:?}"
            );
            assert_eq!(
                p.template.reasoning_effort, effort,
                "row {i}: {req} model {model:?}"
            );
            assert_eq!(p.budget, budget, "row {i}: {req} model {model:?}");
        }
    }

    // ---- clamp below max_tokens ----

    #[test]
    fn budget_is_clamped_below_max_tokens_and_off_at_or_below_zero() {
        let low = json!({"reasoning_effort": "low"});
        // Room enough: the preset stands.
        assert_eq!(plan(&low, None, 4096).budget, Some(1024));
        // max_tokens 1024: 1024 - 512.
        assert_eq!(plan(&low, None, 1024).budget, Some(512));
        assert_eq!(plan(&low, None, 1024).template.enable_thinking, Some(true));
        // Below 1024 the reserve is half of max_tokens: thinking stays on, with half.
        for (max, budget) in [
            (1000, 500),
            (512, 256),
            (300, 150),
            (513, 257),
            (3, 2),
            (1, 1),
        ] {
            let p = plan(&low, None, max);
            assert_eq!(p.budget, Some(budget), "max {max}");
            assert_eq!(p.template.enable_thinking, Some(true), "max {max}");
        }
        assert_eq!(answer_reserve(0), 0);
        assert_eq!(answer_reserve(100), 50);
        assert_eq!(answer_reserve(5_000), 512);
        // A budget that fits is kept as asked.
        assert_eq!(
            plan(&json!({"thinking_budget_tokens": 100}), None, 400).budget,
            Some(100)
        );
        // The model default is clamped the same way.
        assert_eq!(plan(&json!({}), Some(8192), 8192).budget, Some(8192 - 512));
        // No budget: no clamp, the switch is left alone.
        assert_eq!(plan(&json!({}), None, 100), ThinkingPlan::default());
        // 0 = unbounded generation: no clamp.
        assert_eq!(
            plan(&json!({"thinking_budget_tokens": 99_999}), None, 0).budget,
            Some(99_999)
        );
    }

    // ---- running detector ----

    fn qwen(starts: bool) -> ReasoningSplit {
        ReasoningSplit::new(Some(ReasoningParser::Qwen3), starts)
    }
    fn oss() -> ReasoningSplit {
        ReasoningSplit::new(Some(ReasoningParser::GptOss), false)
    }

    /// Feed `deltas` (each `tokens` 1) to a tracker; the index of the delta that stopped it.
    fn stop_at(split: ReasoningSplit, budget: usize, deltas: &[&str]) -> Option<usize> {
        let mut t = BudgetTracker::new(split, budget);
        deltas.iter().position(|d| t.push(d, 1) == BudgetStep::Stop)
    }

    #[test]
    fn stops_at_the_budget_inside_the_block() {
        let d = ["<think>", "a", "b", "c", "d", "</think>", "ok"];
        // Markers count: <think>, a, b reach 3.
        assert_eq!(stop_at(qwen(false), 3, &d), Some(2));
        assert_eq!(stop_at(qwen(false), 1, &d), Some(0));
    }

    #[test]
    fn a_natural_close_before_the_budget_ends_counting() {
        let d = [
            "<think>", "a", "</think>", "x", "y", "z", "<think>", "q", "r", "s",
        ];
        // 3 tokens in the first block; the budget is 3, so the close lands exactly on it — the
        // close wins, and the later block is not counted.
        assert_eq!(stop_at(qwen(false), 3, &d), None);
        assert_eq!(stop_at(qwen(false), 4, &d), None);
        // An answer with no reasoning at all never stops.
        assert_eq!(stop_at(qwen(false), 1, &["Hello", " world"]), None);
    }

    #[test]
    fn starts_in_thinking_counts_from_the_first_token() {
        let d = ["a", "b", "c", "</think>", "ok"];
        assert_eq!(stop_at(qwen(true), 2, &d), Some(1));
        // The same output on a prompt that did not open a block has no reasoning.
        assert_eq!(stop_at(qwen(false), 2, &d), None);
    }

    #[test]
    fn harmony_lead_channel_and_final_transition() {
        let d = [
            "analysis",
            "Let",
            " me",
            " think",
            "assistant",
            "final",
            "4",
        ];
        assert_eq!(stop_at(oss(), 3, &d), Some(2));
        assert_eq!(
            stop_at(oss(), 6, &d),
            None,
            "assistantfinal closes at the 6th token"
        );
        // Opens on `final`: no reasoning.
        assert_eq!(stop_at(oss(), 1, &["final", "The", " answer"]), None);
        // Mixed case and a split lead word: decided once 11 bytes are in.
        assert_eq!(
            stop_at(oss(), 2, &["Ana", "lysis", " x", "yz and more"]),
            Some(3)
        );
        // `analysisx…` is a word, not the channel: no reasoning.
        assert_eq!(stop_at(oss(), 1, &["analysisx", "yz and more"]), None);
    }

    /// Owner ruling: gpt-oss budgets the analysis channel only. The switch to `commentary` (a tool
    /// call) ends counting, a `commentary` lead is never counted, and the tracker is not `inside`
    /// during the call, so an engine `length` there stays a plain `length`.
    #[test]
    fn harmony_budgets_the_analysis_channel_only() {
        let call = [
            "analysis",
            "Need",
            " the",
            " weather",
            "assistant",
            "commentary",
            " to=functions.get_weather",
            " json",
            "{\"location\":",
            "\"SF\"}",
        ];
        // `analysis` .. `assistant` are 5 counted tokens; the 6th delta (`commentary`) reaches a
        // budget of 6 but completes the switch, which ends the count first (no deferral involved).
        let mut t = BudgetTracker::new(oss(), 6);
        for d in call {
            assert_eq!(t.push(d, 1), BudgetStep::Continue, "{d}");
        }
        assert!(
            !t.inside(),
            "in the tool call: not inside the budgeted block"
        );
        assert_eq!(t.counted(), 6, "analysis..assistantcommentary");
        // An analysis that runs past the budget still stops, before any call.
        assert_eq!(stop_at(oss(), 3, &call), Some(2));
        // A `commentary` lead (the call comes first) is never counted.
        let lead = ["commentary", " to=functions.f", " json{}", " more", " more"];
        let mut t = BudgetTracker::new(oss(), 1);
        for d in lead {
            assert_eq!(t.push(d, 1), BudgetStep::Continue, "{d}");
        }
        assert_eq!(t.counted(), 0);
        assert!(!t.inside());
        assert_eq!(
            stop_at(oss(), 1, &["Commentary", " to=functions.f json{}"]),
            None
        );
    }

    /// The budget edge falls inside a close marker split over two deltas: the stop waits one
    /// delta (once), so the close wins; a text that does not close is stopped one delta later.
    #[test]
    fn a_partial_close_marker_at_the_budget_edge_waits_one_delta() {
        let d = ["<think>", "a", "</th", "ink>", "answer"];
        assert_eq!(stop_at(qwen(false), 3, &d), None, "the close wins");
        let d = ["<think>", "a", "</th", "ought", "more", "more"];
        assert_eq!(stop_at(qwen(false), 3, &d), Some(3), "one delta later");
        // gpt-oss: `assistant` + `final` at the edge.
        let d = ["analysis", " a", "assistant", "final", "4"];
        assert_eq!(stop_at(oss(), 3, &d), None);
        // Deferred only once.
        let d = ["<think>", "a", "<", "<", "<", "x"];
        assert_eq!(stop_at(qwen(false), 3, &d), Some(3));
    }

    #[test]
    fn the_chunk_budget_scales_tokens_by_the_engine_ratio() {
        let vlm = VLM_TOKENS_PER_CHUNK_PERCENT;
        let npu = NPU_TOKENS_PER_CHUNK_PERCENT;
        assert_eq!(chunk_budget(137, vlm), 100);
        assert_eq!(chunk_budget(1_024, vlm), 748);
        assert_eq!(chunk_budget(4_096, vlm), 2_990);
        assert_eq!(chunk_budget(16_384, vlm), 11_960);
        assert_eq!(chunk_budget(1, vlm), 1);
        assert_eq!(chunk_budget(4, npu), 4);
        assert_eq!(chunk_budget(1_024, npu), 985);
        assert_eq!(chunk_budget(4_096, npu), 3_939);
        let t = chunk_tracker_for(qwen(true), Some(137), vlm).unwrap();
        assert_eq!(t.budget, 100);
        assert!(chunk_tracker_for(qwen(true), None, npu).is_none());
    }

    #[test]
    fn after_the_stop_every_delta_answers_stop() {
        let mut t = BudgetTracker::new(qwen(true), 2);
        assert_eq!(t.push("a", 1), BudgetStep::Continue);
        assert!(!t.stopped());
        assert_eq!(t.push("b", 1), BudgetStep::Stop);
        assert!(t.stopped() && t.inside());
        assert_eq!(t.push("</think>", 1), BudgetStep::Stop);
        assert_eq!(t.push("", 0), BudgetStep::Stop);
        assert_eq!(t.counted(), 2);
        // A multi-token delta overshoots, never undershoots.
        let mut t = BudgetTracker::new(qwen(true), 2);
        assert_eq!(t.push("abc", 3), BudgetStep::Stop);
        assert_eq!(t.counted(), 3);
        assert_eq!(
            BudgetOutcome::of(Some(&t), false),
            BudgetOutcome {
                armed: true,
                stopped: true,
                inside: true
            }
        );
        assert_eq!(BudgetOutcome::of(Some(&t), true), BudgetOutcome::default());
        assert!(tracker_for(qwen(true), None).is_none());
    }

    #[test]
    fn length_inside_the_block_is_visible() {
        let mut t = BudgetTracker::new(qwen(false), 100);
        assert!(!t.inside());
        t.push("<think>", 1);
        t.push("long", 1);
        assert!(t.inside());
        t.push("</think>", 1);
        assert!(!t.inside());
    }

    #[test]
    fn multi_byte_text_never_panics() {
        let mut t = BudgetTracker::new(qwen(false), 1_000);
        for d in ["żółw", "<thi", "nk>", "ąę", "€€€", "</th", "ink>", "日本"] {
            t.push(d, 1);
        }
        assert!(!t.inside());
        let mut h = BudgetTracker::new(oss(), 1_000);
        for d in ["anal", "ysisż", "日本語", "assistantf", "inal", "x"] {
            h.push(d, 1);
        }
        assert!(!h.inside());
    }

    /// The tokens the usage count's rule gives the FIRST reasoning span of `text`, for deltas
    /// cut at `cuts`.
    fn first_span_tokens(split: ReasoningSplit, deltas: &[&str]) -> usize {
        let text: String = deltas.concat();
        let Some(span) = split.reasoning_spans(&text).into_iter().next() else {
            return 0;
        };
        let mut start = 0;
        let mut n = 0;
        for d in deltas {
            let end = start + d.len();
            if !d.is_empty() && start < span.end && end > span.start {
                n += 1;
            }
            start = end;
        }
        n
    }

    /// Every way of cutting a text into three deltas: the tracker's count equals the usage
    /// rule's count for the first reasoning block (budget never reached).
    #[test]
    fn tracker_matches_the_usage_split_at_every_cut() {
        let cases: Vec<(&str, ReasoningSplit)> = vec![
            (
                "pre<think>reason here</think>answer<think>again</think>",
                qwen(false),
            ),
            ("<think>r</think>a", qwen(false)),
            ("reason</think>answer", qwen(true)),
            ("no reasoning at all", qwen(false)),
            ("<think>never closed", qwen(false)),
            ("analysisThe user asks.assistantfinal4", oss()),
            ("analysis Unclosed reasoning that goes on", oss()),
            ("finalJust the answer", oss()),
            ("plain answer", oss()),
        ];
        for (text, split) in cases {
            let b: Vec<usize> = (0..=text.len())
                .filter(|i| text.is_char_boundary(*i))
                .collect();
            for (ai, &a) in b.iter().enumerate() {
                for &c in &b[ai..] {
                    let deltas = [&text[..a], &text[a..c], &text[c..]];
                    let mut t = BudgetTracker::new(split, usize::MAX);
                    for d in deltas {
                        assert_eq!(t.push(d, 1), BudgetStep::Continue);
                    }
                    assert_eq!(
                        t.counted(),
                        first_span_tokens(split, &deltas),
                        "{text:?} cut at {a},{c}"
                    );
                }
            }
        }
        // gpt-oss, analysis only: the count runs to the end of the `assistantcommentary` switch
        // (deltas overlapping it), and a `commentary` lead counts nothing.
        let text = "analysisThink.assistantcommentary to=functions.f json{}";
        let end = text.find("commentary").unwrap() + "commentary".len();
        for a in 0..=text.len() {
            for c in a..=text.len() {
                let deltas = [&text[..a], &text[a..c], &text[c..]];
                let mut t = BudgetTracker::new(oss(), usize::MAX);
                let mut start = 0;
                let mut expect = 0;
                for d in deltas {
                    t.push(d, 1);
                    if !d.is_empty() && start < end {
                        expect += 1;
                    }
                    start += d.len();
                }
                assert_eq!(t.counted(), expect, "cut at {a},{c}");
                let lead = "COMMENTARYCall a toolAssistantFinaldone";
                let (x, y) = (a.min(lead.len()), c.min(lead.len()));
                let mut t = BudgetTracker::new(oss(), usize::MAX);
                for d in [&lead[..x], &lead[x..y], &lead[y..]] {
                    t.push(d, 1);
                }
                assert_eq!(t.counted(), 0, "commentary lead cut at {x},{y}");
            }
        }
    }
}
