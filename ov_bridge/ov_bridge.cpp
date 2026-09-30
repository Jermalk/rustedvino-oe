// ov_bridge/ov_bridge.cpp
// Phase 5.2: VLMPipeline C-API added at the bottom of this file.
// ─────────────────────────────────────────────────────────────────────────────
// Thin extern "C" wrapper around the OpenVINO GenAI C++ LLMPipeline API.
//
// WHY THIS EXISTS
// ───────────────
// The `openvino-genai` Rust crate depends on `libopenvino_genai_c.so` (a C
// binding shipped with the archive distribution of OpenVINO). The pip-wheel
// installation — which is what this system has — ships only the C++ API
// (`libopenvino_genai.so.2610`). There is no C layer.
//
// We therefore write our own C layer: three functions that create, use and
// destroy an LLMPipeline. Rust calls these directly via `extern "C"` FFI,
// with zero overhead on the token-streaming hot path.
//
// ABI NOTE
// ────────
// The pip wheel (openvino-genai 2026.1) was compiled with
// _GLIBCXX_USE_CXX11_ABI=1 (the "new" __cxx11 ABI, GCC default since GCC 5).
// Confirmed by: nm -D libopenvino_genai.so.2610 | grep __cxx11 (symbols exist).
// This file MUST be compiled with the same flag or the linker will fail with
// "undefined symbol" errors for every std::string-based function.
// build.rs passes -D_GLIBCXX_USE_CXX11_ABI=1.
//
// THREAD SAFETY
// ─────────────
// LLMPipeline is NOT thread-safe. Rust wraps the opaque pointer in
// Arc<Mutex<OvPipeline>> so only one thread calls ov_pipeline_generate at a
// time. The generate call blocks the OS thread for its full duration; Rust
// runs it inside tokio::task::spawn_blocking so the async runtime is never
// stalled.
// ─────────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <fstream>

#include "openvino/genai/llm_pipeline.hpp"
// Pulls in continuous_batching_pipeline + generation_handle + tokenizer +
// scheduler_config (and, transitively, the visual_language tree the CB header
// hard-includes). Vendored at tag 2026.1.0.0 to match the wheel ABI.
#include "openvino/genai/continuous_batching_pipeline.hpp"
// Required for ov_device_available(): ov::Core::get_available_devices().
#include "openvino/runtime/core.hpp"
// ov::hint::kv_cache_precision + ov::element types for the KV_CACHE_PRECISION
// property passed at CB pipeline construction.
#include "openvino/runtime/properties.hpp"
// ov::intel_gpu::device_total_mem_size for ov_device_property() — the GPU
// VRAM/UMA-budget probe (Phase A device inventory). GPU-only property.
#include "openvino/runtime/intel_gpu/properties.hpp"
// ov::get_openvino_version() for ov_get_openvino_version() — Tier 3 of
// the image-metadata plan. Pulled in transitively by core.hpp above,
// included explicitly here for clarity.
#include "openvino/core/version.hpp"
// VLMPipeline (Phase 5.2): ChatHistory, JsonContainer, VLMPipeline.
// visual_language/pipeline.hpp transitively includes chat_history.hpp.
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/genai/chat_history.hpp"
// TextEmbeddingPipeline (R4/G3): text-embedding model → float vectors.
#include "openvino/genai/rag/text_embedding_pipeline.hpp"
// TextRerankPipeline: reranking — query × documents → scored pairs (/v1/rerank).
#include "openvino/genai/rag/text_rerank_pipeline.hpp"
// WhisperPipeline (Phase 5.1b): speech-to-text. Pulls in whisper_generation_config.hpp.
#include "openvino/genai/whisper_pipeline.hpp"
// Text2SpeechPipeline (Phase 5.2a): text-to-speech (SpeechT5). 16 kHz mono f32 output.
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
// Text2ImagePipeline (Phase 5.3b): text-to-image (SDXL). Pulls in image_generation/*.
#include "openvino/genai/image_generation/text2image_pipeline.hpp"
#include "openvino/genai/image_generation/image2image_pipeline.hpp"
#include "openvino/genai/image_generation/inpainting_pipeline.hpp"
#include <variant>

// Per-thread error buffer — avoids heap allocation in the error path.
// thread_local is safe because each spawn_blocking call runs on its own
// thread.
static thread_local char s_last_error[4096] = {};

static void set_error(const char* msg) {
    std::strncpy(s_last_error, msg, sizeof(s_last_error) - 1);
    s_last_error[sizeof(s_last_error) - 1] = '\0';
}

extern "C" {

// ─── Types ───────────────────────────────────────────────────────────────────

// Opaque handle. Rust sees *mut std::ffi::c_void.
typedef void* OvPipelineHandle;

// Token callback invoked once per token during generation.
//   token     – UTF-8 bytes of the generated token (NOT null-terminated)
//   len       – byte length of token
//   user_data – arbitrary pointer passed through unchanged from Rust
// Return 0 to continue; return 1 to stop generation early.
typedef int (*OvTokenCallback)(const char* token, size_t len, void* user_data);

// ─── Lifecycle ───────────────────────────────────────────────────────────────

/// Create an LLMPipeline and return an opaque handle.
/// Returns NULL on failure; call ov_last_error() for the message.
///
/// model_path      – path to the directory containing model .xml/.bin and
///                   tokenizer files (e.g. "/opt/rustedvino/models/qwen3-8b-int4-ov")
/// device          – OpenVINO device string, e.g. "GPU.1" or "CPU"
/// max_prompt_len  – NPU-only compile-time MAX_PROMPT_LEN ceiling; 0 = unset,
///                   keep OpenVINO's own default (1024 as of this writing).
///                   Ignored (harmlessly) for non-NPU devices.
/// min_response_len – NPU-only compile-time MIN_RESPONSE_LEN: output room the
///                   static KV cache reserves on top of MAX_PROMPT_LEN; 0 =
///                   unset (OpenVINO default 128). Verified live on Lunar Lake:
///                   512 let an 863-token prompt generate its full 400 tokens
///                   where the default stopped at 290.
/// ov_cache_dir    – OpenVINO CACHE_DIR, or NULL/empty for no OpenVINO-level
///                   cache. On NPU this stores a weightless compiled blob (default
///                   CACHE_MODE): measured ~6 s cached load vs 30.7 s cold for an
///                   8B int4 model on Lunar Lake, identical output.
OvPipelineHandle ov_pipeline_create(const char* model_path, const char* device,
                                     uint32_t max_prompt_len, uint32_t min_response_len,
                                     const char* ov_cache_dir) {
    s_last_error[0] = '\0';
    try {
        const bool has_cache_dir = ov_cache_dir && ov_cache_dir[0] != '\0';
        if (max_prompt_len > 0 || min_response_len > 0 || has_cache_dir) {
            ov::AnyMap props;
            if (has_cache_dir) {
                props.emplace(ov::cache_dir(std::string(ov_cache_dir)));
            }
            // Raw NPU-plugin config key — not a typed ov::genai::*_property
            // helper (none exists in the vendored headers for this). The NPU
            // plugin's property parser requires exactly int/int64_t — a plain
            // uint32_t fails with "Failed to extract MAX_PROMPT_LEN. Type
            // mismatch: expected types: int or int64_t" (confirmed live on
            // Lunar Lake/NPU). Widen explicitly rather than relying on
            // implicit conversion into ov::Any's type-erased storage.
            if (max_prompt_len > 0) {
                props.emplace("MAX_PROMPT_LEN", static_cast<int64_t>(max_prompt_len));
            }
            // Same int/int64_t requirement as MAX_PROMPT_LEN.
            if (min_response_len > 0) {
                props.emplace("MIN_RESPONSE_LEN", static_cast<int64_t>(min_response_len));
            }
            auto* pipe = new ov::genai::LLMPipeline(
                std::filesystem::path(model_path),
                std::string(device),
                props
            );
            return static_cast<OvPipelineHandle>(pipe);
        }
        auto* pipe = new ov::genai::LLMPipeline(
            std::filesystem::path(model_path),
            std::string(device)
        );
        return static_cast<OvPipelineHandle>(pipe);
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("unknown exception in ov_pipeline_create");
        return nullptr;
    }
}

/// Destroy an LLMPipeline and free its GPU/CPU memory.
/// Passing NULL is a no-op.
/// After this call the handle is invalid — do not use it.
void ov_pipeline_free(OvPipelineHandle handle) {
    if (handle) {
        delete static_cast<ov::genai::LLMPipeline*>(handle);
    }
}

// ─── Generation ──────────────────────────────────────────────────────────────

// ─── Tokenization ────────────────────────────────────────────────────────────

/// Encode `text` (`len` bytes; need not be null-terminated) with the pipeline's
/// tokenizer — independent of generate(), so it can pre-flight a prompt's
/// token count before committing to a (possibly NPU-limited) generation call.
///
/// Same two-pass contract as ov_cb_encode: always writes the true token count
/// to `*out_count`; if `out_ids` is non-null and `cap > 0`, writes up to `cap`
/// ids. Pass `out_ids = NULL, cap = 0` for a count-only pass. Returns 0 on
/// success, -1 on error (call ov_last_error()).
int ov_pipeline_encode(OvPipelineHandle handle,
                       const char*      text,
                       size_t           len,
                       int64_t*         out_ids,
                       size_t           cap,
                       size_t*          out_count) {
    s_last_error[0] = '\0';
    if (!handle)      { set_error("null pipeline handle"); return -1; }
    if (!text && len) { set_error("null text with non-zero len"); return -1; }
    if (!out_count)   { set_error("null out_count"); return -1; }
    auto* pipe = static_cast<ov::genai::LLMPipeline*>(handle);
    try {
        ov::genai::TokenizedInputs enc = pipe->get_tokenizer().encode(std::string(text, len));
        const ov::Tensor& ids = enc.input_ids;
        *out_count = ids.get_size();

        if (out_ids && cap > 0) {
            const size_t to_copy = (*out_count < cap) ? *out_count : cap;
            const auto   et      = ids.get_element_type();
            if (et == ov::element::i64) {
                std::memcpy(out_ids, ids.data<int64_t>(), to_copy * sizeof(int64_t));
            } else if (et == ov::element::i32) {
                const int32_t* src = ids.data<int32_t>();
                for (size_t i = 0; i < to_copy; ++i) {
                    out_ids[i] = static_cast<int64_t>(src[i]);
                }
            } else {
                set_error("unexpected input_ids element type (not i32/i64)");
                return -1;
            }
        }
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception in ov_pipeline_encode");
        return -1;
    }
}

// ─── Diagnostics ─────────────────────────────────────────────────────────────

/// Returns the last error message set by this thread, or an empty string.
/// The pointer is valid until the next call to any ov_* function on this thread.
const char* ov_last_error(void) {
    return s_last_error;
}

} // extern "C"

// ═════════════════════════════════════════════════════════════════════════════
// ContinuousBatchingPipeline C-API
// ─────────────────────────────────────────────────────────────────────────────
// The Phase-2 engine. Unlike the LLMPipeline wrapper above (one blocking
// generate() per request), this exposes the GenAI *granular* API:
//   add_request()        — enqueue a prompt, get a per-request handle
//   step()               — advance ALL active requests by one model iteration
//   has_non_finished()   — loop condition for the engine thread
//   read()               — pull the DELTA token ids a handle produced this step
// One ContinuousBatchingPipeline serves N concurrent requests through a single
// shared KV pool. The Rust engine thread owns the OvCbEngine and drives the
// step loop; this layer holds NO locks (single-threaded by contract).
//
// DETOKENIZATION IS MANUAL. The handle yields token *ids*, not text. We keep a
// per-request id accumulator, re-decode the full list each step (cheap next to
// a model step), and emit only the new suffix — holding back a trailing partial
// UTF-8 character until it is whole (see utf8_safe_len).
// ═════════════════════════════════════════════════════════════════════════════

namespace {

/// Largest prefix length of `s` ending on a complete UTF-8 character boundary.
/// A token may decode to only the first byte(s) of a multibyte char this step,
/// the rest arriving next step. Emitting those partial bytes would corrupt the
/// SSE stream, so we hold them back until the character is complete.
size_t utf8_safe_len(const std::string& s) {
    const size_t n = s.size();
    if (n == 0) {
        return 0;
    }
    // Walk back over continuation bytes (10xxxxxx) to the lead byte of the final
    // character. A UTF-8 char is at most 4 bytes → at most 3 continuation bytes.
    size_t i = n;
    while (i > 0 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80 && (n - i) < 3) {
        --i;
    }
    if (i == 0) {
        return n;  // no lead byte found (malformed) — don't stall, emit all
    }
    const unsigned char lead = static_cast<unsigned char>(s[i - 1]);
    size_t expected;
    if ((lead & 0x80) == 0x00) {
        expected = 1;  // 0xxxxxxx  ASCII
    } else if ((lead & 0xE0) == 0xC0) {
        expected = 2;  // 110xxxxx
    } else if ((lead & 0xF0) == 0xE0) {
        expected = 3;  // 1110xxxx
    } else if ((lead & 0xF8) == 0xF0) {
        expected = 4;  // 11110xxx
    } else {
        return n;  // malformed lead byte — emit all, don't stall
    }
    const size_t have = n - (i - 1);          // bytes available for final char
    return (have >= expected) ? n : (i - 1);  // whole → emit all; partial → hold
}

/// Back `len` off any trailing U+FFFD replacement-character sequences
/// (EF BF BD), returning the reduced length.
///
/// `utf8_safe_len` only checks byte *structure* — a lead byte plus the right
/// count of continuation bytes — and U+FFFD is itself a structurally
/// complete 3-byte sequence, so it passes that check. But the tokenizer's
/// decode() returns a `std::string` that must always be valid UTF-8 (it is
/// backed by a Rust `String` internally), so a still-incomplete trailing
/// multibyte character — the true partial-byte case `utf8_safe_len` exists to
/// catch — gets provisionally rendered as U+FFFD rather than left as
/// dangling raw bytes. Without this extra check that placeholder is emitted
/// to the client immediately; the *next* step's full redecode (now with the
/// completing token) resolves it to the real character at a DIFFERENT byte
/// length, permanently desyncing every byte offset emitted after it — not a
/// cosmetic flicker but silent, irrecoverable data loss on the SSE wire
/// (reproduced live: Polish "źdźbło" streamed as "źbło", losing "dź").
/// Only called for non-final steps — at genuine finish there is nothing left
/// to wait for, so the caller emits whatever bytes remain, U+FFFD or not.
size_t strip_trailing_replacement_char(const std::string& s, size_t len) {
    static constexpr char REPL[3] = {'\xEF', '\xBF', '\xBD'};
    while (len >= 3 && std::memcmp(s.data() + len - 3, REPL, 3) == 0) {
        len -= 3;
    }
    return len;
}

/// Per-request decode state. `all_ids` accumulates every token id the handle has
/// yielded; `emitted` is how many bytes of decode(all_ids) we have already sent.
struct OvCbReqState {
    ov::genai::GenerationHandle handle;
    std::vector<int64_t>        all_ids;
    size_t                      emitted = 0;
    // Real token ids generated but not yet reported via the callback's
    // new_tokens parameter. Accumulates across steps where a token was
    // generated but its decoded text is held back (a multi-byte UTF-8 char
    // split across generation steps — see strip_trailing_replacement_char);
    // reset to 0 once actually handed to the callback. Must NOT be a
    // per-step-local counter: that would silently drop a token's count on
    // any step where its text isn't emittable yet.
    size_t                      pending_new_tokens = 0;
};

/// Owns the ContinuousBatchingPipeline, its tokenizer, and the live request map.
/// One instance lives on the dedicated Rust engine thread — no internal locking.
struct OvCbEngine {
    ov::genai::ContinuousBatchingPipeline      pipe;
    ov::genai::Tokenizer                       tok;
    std::unordered_map<uint64_t, OvCbReqState> requests;

    OvCbEngine(const std::string& model_path,
               const ov::genai::SchedulerConfig& sched,
               const std::string& device,
               const ov::AnyMap& properties)
        : pipe(std::filesystem::path(model_path), sched, device, properties),
          tok(pipe.get_tokenizer()) {}
};

} // anonymous namespace

// ─── Generation parameters ────────────────────────────────────────────────────
// Sampling params passed from Rust through the FFI boundary.
// Float fields use NaN as "not set — keep the engine default".
// stop_strings/stop_count are NULL/0 when no stop strings are requested.
// This layout must EXACTLY match OvGenParamsC in src/ov_cb.rs.
struct OvGenParams {
    size_t      max_new_tokens;         // 0  = use model default
    float       temperature;            // NaN = keep engine default
    float       top_p;                  // NaN = keep engine default
    float       presence_penalty;       // NaN = keep engine default
    float       frequency_penalty;      // NaN = keep engine default
    float       repetition_penalty;     // NaN = keep engine default
    int         use_rng_seed;           // 1 = apply rng_seed
    uint64_t    rng_seed;
    const char* const* stop_strings;   // NULL = no stop strings
    size_t      stop_count;
    const char* json_schema;           // NULL = free-form; else G1 structured output
    size_t      top_k;                 // 0 = not set (engine default). APPENDED
                                       // Never insert mid-struct:
                                       // the size-only drift guard cannot catch
                                       // same-size field reordering.
    size_t      num_assistant_tokens;  // 0 = plain decoding. >0 = assisted request
                                       // (speculative decoding; requires a pipeline
                                       // built with draft_model attached). APPENDED
                                       // (speculative-decoding plan) — never insert
                                       // mid-struct: the size-only drift guard cannot
                                       // catch same-size field reordering.
};

// Apply the G1 structured-output (JSON Schema) constraint to a GenerationConfig
// when params->json_schema is set. Shared by the CB and VLM config builders so
// both chat paths behave identically. The schema string comes straight from the
// OpenAI response_format mapping on the Rust side; xgrammar (bundled in
// libopenvino_genai) enforces it during decoding.
static void apply_structured_output(ov::genai::GenerationConfig& cfg,
                                    const OvGenParams* params) {
    if (params->json_schema && params->json_schema[0] != '\0') {
        // Build the AnyMap explicitly (emplace, like the kv_cache_precision
        // property above) to avoid the brace-init ambiguity with the copy ctor.
        ov::AnyMap props;
        props.emplace(ov::genai::json_schema(std::string(params->json_schema)));
        cfg.structured_output_config = ov::genai::StructuredOutputConfig(props);
    }
}

// Populate a GenerationConfig from the flat OvGenParams sentinel layout. Shared
// by both add_request entry points (string prompt and pre-tokenized ids) so the
// two paths apply IDENTICAL sampling/stop/structured-output semantics — the only
// difference between them is whether the prompt arrives as text or token ids.
static void build_gen_config(ov::genai::GenerationConfig& cfg,
                             const OvGenParams* params) {
    if (params->max_new_tokens > 0) {
        cfg.max_new_tokens = params->max_new_tokens;
    }
    // Temperature: NaN = keep engine default. 0 = greedy; > 0 enables sampling.
    if (!std::isnan(params->temperature)) {
        cfg.temperature = params->temperature;
        cfg.do_sample = (params->temperature > 0.0f);
    }
    // top_p < 1 overrides greedy only when temperature is also non-zero.
    if (!std::isnan(params->top_p)) {
        cfg.top_p = params->top_p;
        if (params->top_p < 1.0f) cfg.do_sample = true;
    }
    // top_k: 0 = not set (GenerationConfig's own default is SIZE_MAX =
    // unlimited). Mirrors top_p's do_sample semantics, quirks included.
    if (params->top_k > 0) {
        cfg.top_k = params->top_k;
        cfg.do_sample = true;
    }
    if (!std::isnan(params->presence_penalty)) {
        cfg.presence_penalty = params->presence_penalty;
    }
    if (!std::isnan(params->frequency_penalty)) {
        cfg.frequency_penalty = params->frequency_penalty;
    }
    if (!std::isnan(params->repetition_penalty)) {
        cfg.repetition_penalty = params->repetition_penalty;
    }
    if (params->use_rng_seed) {
        cfg.rng_seed = params->rng_seed;
    }
    for (size_t i = 0; i < params->stop_count; ++i) {
        if (params->stop_strings && params->stop_strings[i]) {
            cfg.stop_strings.insert(std::string(params->stop_strings[i]));
        }
    }
    apply_structured_output(cfg, params);
    // Speculative decoding: >0 marks this request "assisted" — required (and
    // only valid) when the pipeline was built with a draft model attached.
    if (params->num_assistant_tokens > 0) {
        cfg.num_assistant_tokens = params->num_assistant_tokens;
    }
}

extern "C" {

/// Run one NPU (static LLMPipeline) generation on `prompt` and stream decoded
/// text chunks to `callback`.
///
/// This call BLOCKS until generation is complete or stopped.
/// Run it on the NPU engine thread (never on the async runtime).
///
/// handle               – pipeline handle from ov_pipeline_create
/// prompt               – null-terminated UTF-8 prompt, ALREADY rendered through
///                        the model's chat template on the Rust side. The
///                        pipeline must not template it again, so this sets
///                        GenerationConfig::apply_chat_template = false
///                        (its default is true: generation_config.hpp,
///                        llm_pipeline.hpp). Re-templating wrapped the whole
///                        rendered prompt as one user turn and dropped the
///                        closed-think prefill, so enable_thinking:false was
///                        silently ignored (the project's internal engineering log).
/// params               – sampling/stop/structured-output settings, the same
///                        OvGenParams layout and build_gen_config the CB and VLM
///                        paths use. Must be non-null. num_assistant_tokens is
///                        ignored here (NPU speculative decoding is a
///                        construction-time draft model, not a request field).
/// callback             – called once per decoded text CHUNK (not per token: the
///                        streamer can hold back and flush several tokens at
///                        once); return non-zero to stop early
/// user_data            – passed through to callback unchanged
/// finish_code_out      – optional: 1 = STOP (EOS), 2 = LENGTH (budget reached)
/// generated_tokens_out – optional: the pipeline's own generated-token count
///                        (perf_metrics) — the authoritative count for usage
/// input_tokens_out     – optional: the pipeline's own prompt-token count
///
/// Returns 0 on success, -1 on error (call ov_last_error() for details).
int ov_pipeline_generate(
    OvPipelineHandle   handle,
    const char*        prompt,
    const OvGenParams* params,
    OvTokenCallback    callback,
    void*              user_data,
    int*               finish_code_out,
    size_t*            generated_tokens_out,
    size_t*            input_tokens_out
) {
    s_last_error[0] = '\0';

    if (!handle) { set_error("null pipeline handle"); return -1; }
    if (!prompt) { set_error("null prompt");          return -1; }
    if (!params) { set_error("null params");          return -1; }

    auto* pipe = static_cast<ov::genai::LLMPipeline*>(handle);

    // Stream decoded chunks to Rust. StreamingStatus::RUNNING = 0, STOP = 1.
    auto streamer_fn = [callback, user_data](const std::string& chunk)
        -> ov::genai::StreamingStatus
    {
        if (callback) {
            int ret = callback(chunk.c_str(), chunk.size(), user_data);
            if (ret != 0) {
                return ov::genai::StreamingStatus::STOP;
            }
        }
        return ov::genai::StreamingStatus::RUNNING;
    };

    try {
        ov::genai::GenerationConfig cfg;  // greedy by default, CB parity
        cfg.apply_chat_template = false;  // prompt is pre-rendered in Rust
        build_gen_config(cfg, params);
        cfg.num_assistant_tokens = 0;     // see `params` above
        // Keep the model's own extra stop ids (e.g. Qwen3's <|endoftext|>
        // alongside <|im_end|>) — a fresh GenerationConfig would drop them.
        const ov::genai::GenerationConfig model_cfg = pipe->get_generation_config();
        cfg.stop_token_ids.insert(model_cfg.stop_token_ids.begin(),
                                  model_cfg.stop_token_ids.end());
        if (cfg.eos_token_id < 0) {
            cfg.eos_token_id = model_cfg.eos_token_id;
        }

        auto result = pipe->generate(std::string(prompt), cfg, streamer_fn);

        // Finish reason from the pipeline's own token count, same rule as
        // ov_vlm_generate: generated >= budget → LENGTH, else STOP.
        const size_t generated = result.perf_metrics.get_num_generated_tokens();
        int finish_code = 1; // STOP (EOS) default
        if (params->max_new_tokens > 0 && generated >= params->max_new_tokens) {
            finish_code = 2; // LENGTH
        }
        if (finish_code_out)      *finish_code_out      = finish_code;
        if (generated_tokens_out) *generated_tokens_out = generated;
        if (input_tokens_out)     *input_tokens_out     = result.perf_metrics.num_input_tokens;
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception in ov_pipeline_generate");
        return -1;
    }
}

} // extern "C" (NPU generate)

extern "C" {

// ─── Types ───────────────────────────────────────────────────────────────────

// Opaque handle to an OvCbEngine. Rust sees *mut std::ffi::c_void.
typedef void* OvCbHandle;

// Invoked by ov_cb_step once per request that produced output (or finished)
// this step. Fans tokens out from one FFI call to all active requests; Rust
// routes request_id → that request's mpsc Sender.
//   user_data     – passthrough pointer from the ov_cb_step call
//   request_id    – the id given to ov_cb_add_request
//   delta         – new UTF-8 text bytes (NOT null-terminated); may be empty
//   len           – byte length of delta
//   finish_reason – 0 = still running, 1 = STOP (eos), 2 = LENGTH (max tokens),
//                   3 = IGNORED (RustedVINO-internal sentinel: the CB scheduler
//                   hit GenerationStatus::IGNORED — KV-pool exhaustion, this
//                   request never actually ran to completion — NOT a value
//                   OpenVINO GenAI itself defines; do not confuse with its
//                   GenerationFinishReason enum). Nonzero means this request is
//                   done; no more callbacks for it.
//   new_tokens    – real token ids reported as of THIS callback (from
//                   GenerationOutput::generated_ids, before decoding to
//                   text), accumulated since the last time this request's
//                   callback fired — so a token generated while its decoded
//                   text was held back (a multi-byte UTF-8 char split across
//                   steps) is still counted exactly once, on whichever later
//                   callback finally emits it. 0 on the terminal-with-
//                   nothing-left branch. Plain decoding: normally 1 per
//                   callback. Speculative decoding (draft attached,
//                   num_assistant_tokens > 0): can be >1 — one verification
//                   step can accept several draft tokens at once, which land
//                   in the SAME callback/delta.
//                   Callers must count real tokens from this field, not by
//                   assuming one callback == one token (that assumption
//                   silently undercounts under speculative decoding).
typedef void (*OvCbTokenCallback)(void*       user_data,
                                  uint64_t    request_id,
                                  const char* delta,
                                  size_t      len,
                                  int         finish_reason,
                                  size_t      new_tokens);

// ─── Lifecycle ───────────────────────────────────────────────────────────────

/// Create a ContinuousBatchingPipeline on `device` for the model at
/// `model_path`.
///
/// `max_num_seqs`  — max concurrent sequences the CB scheduler will accept.
///                   0 = keep OV default (256). Set to a small value (e.g. 8)
///                   to prevent KV-pool exhaustion stalls under heavy load.
///
/// `cache_size_gb` — KV cache size in whole GB (size_t in SchedulerConfig).
///                   0.0 = keep OV default (dynamic allocation).
///                   Dynamic allocation can deadlock when VRAM fills; setting
///                   a fixed value (e.g. 8.0) bounds the pool and allows the
///                   scheduler to preempt sequences instead of stalling.
///
/// `kv_cache_precision` — value for the KV_CACHE_PRECISION compile property.
///                   NULL or "" keeps the plugin default (f16 on GPU). "u8"
///                   compresses the KV cache to 8-bit (~2× tokens per GB at a
///                   small quality cost); "f16"/"f32" force that precision.
///                   Unknown values are ignored (plugin default kept).
///
/// `enable_prefix_caching` — reuse KV blocks across requests that share an
///                   identical prompt prefix (e.g. repeated system prompts,
///                   multi-turn history resent by a stateless OpenAI-style
///                   client). SchedulerConfig's own library default is
///                   `false` for a directly-constructed
///                   ContinuousBatchingPipeline (the default-on behavior
///                   documented on the field only applies when CB is
///                   invoked via LLMPipeline, which this bridge does not
///                   do) — measured on this project to give zero prefill
///                   speedup with it off. Only affects prefill/TTFT, not
///                   decode tok/s. Retained blocks count against
///                   `cache_size_gb`'s pool, not on top of it, EXCEPT when
///                   `cache_size_gb` is 0.0 (dynamic allocation): retained
///                   blocks are then unbounded and untracked by the VRAM
///                   budget — do not combine dynamic allocation with this.
///
/// `draft_model_path` — directory of a draft model for speculative decoding.
///                   NULL or "" = no draft (plain decoding, unchanged behavior).
///                   Non-empty attaches the draft via ov::genai::draft_model();
///                   every request to the resulting pipeline must then set
///                   OvGenParams::num_assistant_tokens > 0 or the pipeline
///                   rejects it.
///
/// `draft_device` — OpenVINO device for the draft. NULL or "" = same device
///                   as `device`. Ignored when draft_model_path is empty.
///
/// Returns NULL on failure; call ov_last_error() for the reason.
OvCbHandle ov_cb_create(const char* model_path,
                         const char* device,
                         size_t      max_num_seqs,
                         double      cache_size_gb,
                         const char* kv_cache_precision,
                         const char* ov_cache_dir,
                         bool        enable_prefix_caching,
                         const char* draft_model_path,   // NULL/"" = no draft
                         const char* draft_device) {      // NULL/"" = same as `device`
    s_last_error[0] = '\0';
    if (!model_path) { set_error("null model_path"); return nullptr; }
    if (!device)     { set_error("null device");     return nullptr; }
    try {
        ov::genai::SchedulerConfig sched;
        if (max_num_seqs > 0)    sched.max_num_seqs = max_num_seqs;
        // round UP, never truncate: static_cast<size_t>(0.5) silently produces
        // 0, which SchedulerConfig treats as "dynamic/unbounded allocation"
        // (scheduler_config.hpp: cache_size==0 && num_kv_blocks==0 => dynamic)
        // — a caller who set a real, if fractional, budget below 1.0 would
        // silently get the exact unbounded-growth hazard this project's own
        // docs warn against, with zero error or log line (found in adversarial
        // review, 2026-08-20, of the VLM scheduler_config fix below).
        if (cache_size_gb > 0.0) sched.cache_size = static_cast<size_t>(std::ceil(cache_size_gb));
        sched.enable_prefix_caching = enable_prefix_caching;

        // Map the requested KV-cache precision to the compile property. Only set
        // it when explicitly requested so the empty/NULL case is byte-identical to
        // the previous behaviour (plugin default).
        ov::AnyMap props;
        if (kv_cache_precision && kv_cache_precision[0] != '\0') {
            const std::string p(kv_cache_precision);
            if      (p == "u8")  props.emplace(ov::hint::kv_cache_precision(ov::element::u8));
            else if (p == "f16") props.emplace(ov::hint::kv_cache_precision(ov::element::f16));
            else if (p == "f32") props.emplace(ov::hint::kv_cache_precision(ov::element::f32));
            // unknown → leave props empty, keep plugin default
        }
        // GPU blob cache: first load compiles + writes; subsequent loads read
        // the blob (~10–30 s) instead of recompiling from IR (~1–3 min).
        if (ov_cache_dir && ov_cache_dir[0] != '\0') {
            props.emplace(ov::cache_dir(std::string(ov_cache_dir)));
        }
        if (draft_model_path && draft_model_path[0] != '\0') {
            const std::string ddev =
                (draft_device && draft_device[0] != '\0') ? draft_device : device;
            props.emplace(ov::genai::draft_model(
                std::filesystem::path(draft_model_path), ddev));
        }

        auto* eng = new OvCbEngine(std::string(model_path), sched, std::string(device), props);
        return static_cast<OvCbHandle>(eng);
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("unknown exception in ov_cb_create");
        return nullptr;
    }
}

/// Destroy the engine and free its KV pool / GPU memory. NULL is a no-op.
void ov_cb_free(OvCbHandle handle) {
    if (!handle) { return; }
    // A C++ destructor must never unwind across the C ABI boundary — that calls
    // std::terminate (SIGABRT) and takes the whole server down mid-shutdown.
    // Swallow any teardown exception so freeing an engine is always survivable;
    // record it best-effort for a caller that checks ov_last_error().
    try {
        delete static_cast<OvCbEngine*>(handle);
    } catch (const std::exception& e) {
        set_error(e.what());
    } catch (...) {
        set_error("unknown exception in ov_cb_free");
    }
}

// ─── Requests ────────────────────────────────────────────────────────────────

/// Enqueue a generation request. `request_id` must be unique among live
/// requests. `prompt` is sent raw — apply the chat template on the Rust side.
/// Sampling params in `*params`; see `OvGenParams` for the sentinel conventions.
/// Returns 0 on success, -1 on error (call ov_last_error()).
int ov_cb_add_request(OvCbHandle         handle,
                      uint64_t           request_id,
                      const char*        prompt,
                      const OvGenParams* params) {
    s_last_error[0] = '\0';
    if (!handle) { set_error("null engine handle"); return -1; }
    if (!prompt) { set_error("null prompt");        return -1; }
    if (!params) { set_error("null params");        return -1; }
    auto* eng = static_cast<OvCbEngine*>(handle);
    try {
        ov::genai::GenerationConfig cfg;  // greedy by default (do_sample = false)
        // The prompt arrives already chat-templated from Rust (this is the
        // fail-open fallback when pre-tokenizing failed); never template it
        // twice. Harmless if this pipeline version does not template strings.
        cfg.apply_chat_template = false;
        build_gen_config(cfg, params);

        ov::genai::GenerationHandle h =
            eng->pipe.add_request(request_id, std::string(prompt), cfg);
        eng->requests.emplace(request_id, OvCbReqState{std::move(h), {}, 0});
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception in ov_cb_add_request");
        return -1;
    }
}

/// Enqueue a generation request from PRE-TOKENIZED input ids (T7.3).
///
/// The chat/completions L0 prompt-length gate already tokenizes the prompt to
/// check it against the KV-pool capacity. Rather than discard those ids and let
/// the string `add_request` re-tokenize (a second full pass on the engine
/// thread → inflated TTFT on long prompts), the Rust side feeds the gate's ids
/// straight here. `ids`/`count` are copied into an owned i64 tensor — the
/// pipeline holds the request past this call, so the buffer must not alias
/// Rust-owned memory. Sampling/stop/structured-output are built identically to
/// the string path via the shared `build_gen_config`.
int ov_cb_add_request_ids(OvCbHandle         handle,
                          uint64_t           request_id,
                          const int64_t*     ids,
                          size_t             count,
                          const OvGenParams* params) {
    s_last_error[0] = '\0';
    if (!handle) { set_error("null engine handle"); return -1; }
    if (!ids)    { set_error("null ids");           return -1; }
    if (!params) { set_error("null params");        return -1; }
    if (count == 0) { set_error("empty ids");       return -1; }
    auto* eng = static_cast<OvCbEngine*>(handle);
    try {
        ov::genai::GenerationConfig cfg;  // greedy by default (do_sample = false)
        build_gen_config(cfg, params);

        // Owned tensor: the pipeline retains the request beyond this call, so a
        // view over the transient Rust buffer would dangle. Shape {1, count}
        // matches the tokenizer's batch-of-one input_ids layout.
        ov::Tensor input_ids(ov::element::i64, ov::Shape{1, count});
        std::memcpy(input_ids.data<int64_t>(), ids, count * sizeof(int64_t));

        ov::genai::GenerationHandle h =
            eng->pipe.add_request(request_id, input_ids, cfg);
        eng->requests.emplace(request_id, OvCbReqState{std::move(h), {}, 0});
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception in ov_cb_add_request_ids");
        return -1;
    }
}

/// Cancel a live request (e.g. the client disconnected). Stops generation and
/// removes our tracking; the pipeline frees its KV blocks on the next step().
/// Unknown id or NULL handle is a no-op.
void ov_cb_drop_request(OvCbHandle handle, uint64_t request_id) {
    if (!handle) {
        return;
    }
    auto* eng = static_cast<OvCbEngine*>(handle);
    auto it = eng->requests.find(request_id);
    if (it != eng->requests.end()) {
        try {
            it->second.handle->stop();
        } catch (...) {
            // best-effort cancel; still drop our tracking below
        }
        eng->requests.erase(it);
    }
}

// ─── Stepping ────────────────────────────────────────────────────────────────

/// 1 if the pipeline still has unfinished requests, 0 otherwise (or on error).
/// The engine thread uses this as its step-loop condition.
int ov_cb_has_unfinished(OvCbHandle handle) {
    if (!handle) {
        return 0;
    }
    auto* eng = static_cast<OvCbEngine*>(handle);
    try {
        return eng->pipe.has_non_finished_requests() ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

/// Read the live KV-cache occupancy from the pipeline's last scheduler step.
/// Writes the percentage (0–100) to *out_cache_usage_pct on success. Returns 0
/// on success, -1 on error (sets ov_last_error; *out left untouched).
///
/// get_metrics() returns a struct the pipeline caches during each step, so this
/// is a cheap field read — safe to call once per step from the engine thread
/// (co-residency Slice 3a live KV-occupancy gauge).
int ov_cb_get_metrics(OvCbHandle handle, double* out_cache_usage_pct) {
    s_last_error[0] = '\0';
    if (!handle)              { set_error("null engine handle"); return -1; }
    if (!out_cache_usage_pct) { set_error("null out pointer");  return -1; }
    auto* eng = static_cast<OvCbEngine*>(handle);
    try {
        const ov::genai::PipelineMetrics m = eng->pipe.get_metrics();
        *out_cache_usage_pct = static_cast<double>(m.cache_usage);
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception in ov_cb_get_metrics");
        return -1;
    }
}

/// Advance every active request by one model iteration, then drain newly
/// produced tokens, invoking `callback` per request that produced output or
/// finished this step. Finished requests are detokenized fully and removed.
/// Returns 0 on success, -1 on error (call ov_last_error()).
int ov_cb_step(OvCbHandle handle, OvCbTokenCallback callback, void* user_data) {
    s_last_error[0] = '\0';
    if (!handle) { set_error("null engine handle"); return -1; }
    auto* eng = static_cast<OvCbEngine*>(handle);

    try {
        eng->pipe.step();
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception in ov_cb_step (step)");
        return -1;
    }

    // Collect finished ids and erase after the loop — erasing mid-iteration
    // would invalidate the iterator.
    std::vector<uint64_t> to_erase;
    try {
        for (auto& [rid, st] : eng->requests) {
            int  fr       = 0;     // GenerationFinishReason::NONE
            bool produced = false; // did this request produce output / finish?

            if (st.handle->can_read()) {
                ov::genai::GenerationOutputs outs = st.handle->read();
                for (auto& kv : outs) {
                    for (int64_t id : kv.second.generated_ids) {
                        st.all_ids.push_back(id);
                    }
                    // Accumulate on the PERSISTENT per-request counter, not a
                    // step-local one: a token generated this step may have its
                    // decoded text held back (below) until a later step, and
                    // must still be reported once it finally is.
                    st.pending_new_tokens += kv.second.generated_ids.size();
                    if (kv.second.finish_reason != ov::genai::GenerationFinishReason::NONE) {
                        fr = static_cast<int>(kv.second.finish_reason);
                    }
                }
                produced = true;
                // A readable output is not proof the generation is still live:
                // `finish_reason` can stay NONE on every output even though the
                // handle's own status has already gone terminal (observed:
                // KV-pool exhaustion -> GenerationStatus::IGNORED, whose last
                // readable output reports finish_reason NONE). Missing this
                // left the request un-finalized on both sides of the FFI
                // boundary forever: has_non_finished_requests() eventually
                // goes false (nothing left to step), so no later drain ever
                // gets a chance to notice either. Force finalization from the
                // handle's status when the per-output reason didn't already
                // give us one (the project's internal engineering log 2026-08-19, nanbeige hang).
                if (fr == 0 && st.handle->get_status() != ov::genai::GenerationStatus::RUNNING) {
                    // 2026-08-21 (found live, VLM instant-EOS investigation):
                    // confirmed by direct instrumentation that this branch is hit
                    // with GenerationStatus::IGNORED (KV-pool exhaustion, request
                    // never actually ran), not genuine EOS. Report that distinctly
                    // (fr=3) so the Rust side can surface a real error instead of
                    // fabricating a successful empty completion — any OTHER
                    // terminal status here (unexpected; CANCEL is handled via the
                    // client-disconnect path, not this one) still falls back to
                    // the safe STOP default rather than hanging the stream.
                    fr = (st.handle->get_status() == ov::genai::GenerationStatus::IGNORED)
                             ? 3 // RustedVINO-internal sentinel: pool-exhausted, not a real STOP
                             : static_cast<int>(ov::genai::GenerationFinishReason::STOP);
                }
            } else if (st.handle->get_status() != ov::genai::GenerationStatus::RUNNING) {
                // Terminal with nothing left to read (cancelled / OOM-ignored).
                fr = (st.handle->get_status() == ov::genai::GenerationStatus::IGNORED)
                         ? 3
                         : static_cast<int>(ov::genai::GenerationFinishReason::STOP);
                produced = true;
            }

            if (!produced) {
                continue;
            }

            const bool is_final = (fr != 0);
            // Re-decode the full id list; emit only the new suffix. At finish,
            // emit everything (sequence is whole); otherwise hold a partial char
            // — and hold a still-provisional U+FFFD placeholder too (see
            // strip_trailing_replacement_char).
            const std::string text = eng->tok.decode(st.all_ids);
            const size_t      target =
                is_final ? text.size()
                         : strip_trailing_replacement_char(text, utf8_safe_len(text));
            // Tokens are "reported" (and the accumulator cleared) exactly when
            // the callback fires below — whether or not a callback is actually
            // registered, so pending_new_tokens never grows unbounded.
            const bool reporting_tokens = (target > st.emitted) || is_final;

            if (callback) {
                if (target > st.emitted) {
                    callback(user_data, rid, text.data() + st.emitted,
                             target - st.emitted, fr, st.pending_new_tokens);
                } else if (is_final) {
                    // Signal completion even when no new bytes landed this step.
                    callback(user_data, rid, "", 0, fr, st.pending_new_tokens);
                }
            }
            if (reporting_tokens) {
                st.pending_new_tokens = 0;
            }
            if (target > st.emitted) {
                st.emitted = target;
            }
            if (is_final) {
                to_erase.push_back(rid);
            }
        }
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception in ov_cb_step (drain)");
        return -1;
    }

    for (uint64_t rid : to_erase) {
        eng->requests.erase(rid);
    }
    return 0;
}

// ─── Tokenizer ─────────────────────────────────────────────────────────────────
// The CB engine already owns the model's tokenizer (eng->tok, copied from
// pipe.get_tokenizer() at construction). These expose it so the Rust side can
// count prompt tokens (exact usage), tokenize (/tokenize) and detokenize
// (/detokenize). The Tokenizer wraps a non-thread-safe InferRequest, so — like
// step() — these must only be called from the single engine thread.

// One-shot text sink for ov_cb_decode: invoked exactly once with the full
// decoded UTF-8 string (NOT null-terminated). Mirrors the streaming token
// callback shape so Rust can reuse its trampoline pattern.
typedef void (*OvTextCallback)(void* user_data, const char* text, size_t len);

/// Encode `text` (`len` bytes; need not be null-terminated) with the model's
/// tokenizer.
///
/// Always writes the true token count to `*out_count`. If `out_ids` is non-null
/// and `cap > 0`, writes up to `cap` token ids (as int64) into it. To obtain
/// only the count (the exact-`usage` path), pass `out_ids = NULL, cap = 0`. If
/// `*out_count > cap`, re-call with a buffer of at least `*out_count` for the
/// full id list. Returns 0 on success, -1 on error (call ov_last_error()).
int ov_cb_encode(OvCbHandle handle,
                 const char* text,
                 size_t      len,
                 int64_t*    out_ids,
                 size_t      cap,
                 size_t*     out_count) {
    s_last_error[0] = '\0';
    if (!handle)      { set_error("null engine handle"); return -1; }
    if (!text && len) { set_error("null text with non-zero len"); return -1; }
    if (!out_count)   { set_error("null out_count"); return -1; }
    auto* eng = static_cast<OvCbEngine*>(handle);
    try {
        ov::genai::TokenizedInputs enc = eng->tok.encode(std::string(text, len));
        const ov::Tensor& ids = enc.input_ids;
        *out_count = ids.get_size();

        if (out_ids && cap > 0) {
            const size_t to_copy = (*out_count < cap) ? *out_count : cap;
            const auto   et      = ids.get_element_type();
            if (et == ov::element::i64) {
                std::memcpy(out_ids, ids.data<int64_t>(), to_copy * sizeof(int64_t));
            } else if (et == ov::element::i32) {
                // Some tokenizers emit int32 ids; widen to the int64 contract.
                const int32_t* src = ids.data<int32_t>();
                for (size_t i = 0; i < to_copy; ++i) {
                    out_ids[i] = static_cast<int64_t>(src[i]);
                }
            } else {
                set_error("unexpected input_ids element type (not i32/i64)");
                return -1;
            }
        }
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception in ov_cb_encode");
        return -1;
    }
}

/// Decode `count` token ids (int64) to UTF-8 text via the model's detokenizer,
/// delivering the whole string to `callback` exactly once. `ids` may be NULL
/// only when `count == 0` (decodes to the empty string). Returns 0 on success,
/// -1 on error (call ov_last_error()).
int ov_cb_decode(OvCbHandle     handle,
                 const int64_t* ids,
                 size_t         count,
                 OvTextCallback callback,
                 void*          user_data) {
    s_last_error[0] = '\0';
    if (!handle)       { set_error("null engine handle"); return -1; }
    if (!ids && count) { set_error("null ids with non-zero count"); return -1; }
    auto* eng = static_cast<OvCbEngine*>(handle);
    try {
        std::vector<int64_t> v(ids, ids + count);
        std::string text = eng->tok.decode(v);
        if (callback) {
            callback(user_data, text.data(), text.size());
        }
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception in ov_cb_decode");
        return -1;
    }
}

/// Returns 1 if `device_name` is listed in `ov::Core::get_available_devices()`,
/// 0 otherwise (including on any exception). Used by the Rust startup assertion
/// to give a clear early error when the required GPU is absent.
int ov_device_available(const char* device_name) {
    if (!device_name) return 0;
    try {
        ov::Core core;
        for (const auto& dev : core.get_available_devices()) {
            if (dev == std::string(device_name)) return 1;
        }
        return 0;
    } catch (...) {
        return 0;
    }
}

/// Writes all available device names into `out` as newline-separated strings.
///
/// Returns the number of devices written, or -1 on any error.
/// `out` is always NUL-terminated if `out_size > 0`. Used by `--list-devices`
/// in `main.rs` to enumerate compute devices before a config file is loaded.
int ov_list_devices(char* out, size_t out_size) {
    if (!out || out_size == 0) return -1;
    out[0] = '\0';
    try {
        ov::Core core;
        auto devices = core.get_available_devices();
        size_t pos = 0;
        for (size_t i = 0; i < devices.size(); ++i) {
            const auto& dev = devices[i];
            bool add_newline = (i + 1 < devices.size());
            size_t needed = dev.size() + (add_newline ? 1 : 0);
            if (pos + needed + 1 > out_size) break; // +1 for NUL
            std::memcpy(out + pos, dev.c_str(), dev.size());
            pos += dev.size();
            if (add_newline) out[pos++] = '\n';
        }
        out[pos] = '\0';
        return static_cast<int>(devices.size());
    } catch (...) {
        return -1;
    }
}

/// Writes a single device property (as a string) into `out` for the Phase-A
/// device inventory. `key` selects the property:
///   "FULL_DEVICE_NAME"          → ov::device::full_name        (string)
///   "DEVICE_TYPE"               → ov::device::type             → "DISCRETE" |
///                                 "INTEGRATED" | "OTHER"
///   "DEVICE_ARCHITECTURE"       → ov::device::architecture     (string)
///   "GPU_DEVICE_TOTAL_MEM_SIZE" → ov::intel_gpu::device_total_mem_size
///                                 (uint64 bytes, formatted as decimal; GPU only)
/// Any other key is attempted generically via get_property(...).as<string>().
///
/// Returns the string length written (>= 0) on success, or -1 on any error
/// (unsupported property for the device, unknown device, exception). `out` is
/// always NUL-terminated when out_size > 0. On -1, call ov_last_error().
int ov_device_property(const char* device, const char* key,
                       char* out, size_t out_size) {
    if (!device || !key || !out || out_size == 0) {
        if (out && out_size > 0) out[0] = '\0';
        set_error("ov_device_property: null/zero argument");
        return -1;
    }
    out[0] = '\0';
    try {
        ov::Core core;
        const std::string k(key);
        std::string val;
        if (k == "FULL_DEVICE_NAME") {
            val = core.get_property(device, ov::device::full_name);
        } else if (k == "DEVICE_TYPE") {
            const auto t = core.get_property(device, ov::device::type);
            val = (t == ov::device::Type::DISCRETE)   ? "DISCRETE"
                : (t == ov::device::Type::INTEGRATED) ? "INTEGRATED"
                                                      : "OTHER";
        } else if (k == "DEVICE_ARCHITECTURE") {
            val = core.get_property(device, ov::device::architecture);
        } else if (k == "GPU_DEVICE_TOTAL_MEM_SIZE") {
            const uint64_t bytes =
                core.get_property(device, ov::intel_gpu::device_total_mem_size);
            val = std::to_string(bytes);
        } else {
            val = core.get_property(device, k).as<std::string>();
        }
        const size_t copy_len = std::min(val.size(), out_size - 1);
        std::memcpy(out, val.c_str(), copy_len);
        out[copy_len] = '\0';
        return static_cast<int>(copy_len);
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception in ov_device_property");
        return -1;
    }
}

/// Writes the running `OpenVINO` runtime's build-number string (e.g.
/// `"2026.2.1-19140-c01cd93e24d"`) into `out` — `ov::get_openvino_version()`'s
/// `buildNumber` field. Static for the process lifetime (no `Core` instance
/// needed, no device argument); the Rust side caches it after the first call
/// (the image-metadata plan Tier 3 — `generation_metadata.openvino_version`).
///
/// Returns the string length written (>= 0) on success, or -1 on a null/zero
/// argument. `get_openvino_version()` is `noexcept`, so there is no exception
/// path here. `out` is always NUL-terminated when `out_size > 0`.
int ov_get_openvino_version(char* out, size_t out_size) {
    if (!out || out_size == 0) {
        set_error("ov_get_openvino_version: null/zero argument");
        return -1;
    }
    out[0] = '\0';
    const ov::Version version = ov::get_openvino_version();
    const std::string val = version.buildNumber ? version.buildNumber : "";
    const size_t copy_len = std::min(val.size(), out_size - 1);
    std::memcpy(out, val.c_str(), copy_len);
    out[copy_len] = '\0';
    return static_cast<int>(copy_len);
}

/// `sizeof(OvGenParams)` — lets Rust assert its `#[repr(C)]` mirror(s) stay
/// layout-identical to this struct. There are TWO independent Rust copies
/// (`ov_cb.rs` and `ov_vlm.rs`, each private to its module — see their doc
/// comments) and nothing previously caught them drifting apart: adding a
/// field here without updating both silently misaligns every field after it
/// in the VLM (or CB) path specifically, producing garbage parameter values
/// rather than a compile error (found live, 2026-07-14, adding
/// `repetition_penalty` — updated `ov_cb.rs` first, missed `ov_vlm.rs`,
/// which then read next-to-garbage into `use_rng_seed` and everything after).
size_t ov_gen_params_size(void) {
    return sizeof(OvGenParams);
}

} // extern "C" (CB section)

// ═════════════════════════════════════════════════════════════════════════════
// VLMPipeline C-API (Phase 5.2)
// ─────────────────────────────────────────────────────────────────────────────
// Thin extern "C" wrapper around VLMPipeline::generate() for the VLM path.
// VLMPipeline takes a ChatHistory (full message array) + ov::Tensor images
// (NHWC uint8). Generation blocks the calling thread; Rust runs it on the
// dedicated VLM engine thread (same pattern as the LLMPipeline wrapper above).
//
// The OvGenParams struct and OvTokenCallback typedef defined earlier in this
// file are reused here — same C ABI, same Rust mirror struct.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// OvVlmState bundles a heap-allocated VLMPipeline.
// Rust sees the whole thing as *mut c_void (OvVlmHandle).
struct OvVlmState {
    ov::genai::VLMPipeline* pipe = nullptr;
    // Standalone tokenizer for the L0 prompt-length gate (ov_vlm_count_tokens)
    // — built directly from the model dir, same pattern as OvEmbedState::tok
    // below, rather than pipe->get_tokenizer() (avoids depending on pipe's
    // construction order and keeps this member independently testable).
    ov::genai::Tokenizer    tok;

    OvVlmState(const char* model_path, const char* device, const char* ov_cache_dir,
               double cache_size_gb, bool enable_prefix_caching)
        : tok(std::filesystem::path(model_path))
    {
        ov::AnyMap props;
        if (ov_cache_dir && ov_cache_dir[0] != '\0') {
            props.emplace(ov::cache_dir(std::string(ov_cache_dir)));
        }
        // VLMPipeline runs on a ContinuousBatchingPipeline internally
        // (VLMContinuousBatchingAdapter) exactly like the plain text-gen path,
        // but unlike ov_cb_create it has no constructor overload that takes a
        // SchedulerConfig directly — the *only* way in is the
        // ov::genai::scheduler_config AnyMap property (see llm_pipeline.hpp).
        // Omitting it (the previous behavior) left OpenVINO's own internal
        // default (dynamic/unbounded cache_size) in charge, which measured
        // ~2x this project's configured `kv_cache_gb` budget for
        // qwen3.5-4b-int8-ov at 75-100K context depth
        // (the project's internal engineering log). This
        // model's hybrid linear/full-attention architecture also reserves a
        // large *fixed* floor (~6-7GB observed) for the linear-attention
        // layers' state regardless of context length — too small a
        // cache_size_gb fails pipeline construction outright with "Requested
        // linear attention cache allocation exceeds the configured cache
        // size," so a caller passing a too-small budget gets a loud error
        // instead of silent unbounded growth.
        //
        // `SchedulerConfig`'s own constructor defaults `enable_prefix_caching`
        // to false — measured live (isolated repro) at a
        // 30-40x cost per turn on the exact resent-full-history-every-turn
        // pattern this server's own OpenAI-compatible chat handler produces:
        // ~16s/turn with it off vs ~0.4s/turn from the 2nd turn onward with
        // it on, at ~50K token depth. The CB/text-gen path
        // (ov_cb_create, above) always threads this through from
        // Config::enable_prefix_caching explicitly for exactly this reason —
        // the VLM path simply never did.
        if (cache_size_gb > 0.0) {
            ov::genai::SchedulerConfig sched;
            // round UP, never truncate — see the matching comment in
            // ov_cb_create above; same landmine, same fix.
            sched.cache_size = static_cast<size_t>(std::ceil(cache_size_gb));
            sched.enable_prefix_caching = enable_prefix_caching;
            props.emplace(ov::genai::scheduler_config(sched));
        }
        // cache_size_gb <= 0.0 here means no scheduler_config is built at all
        // — prefix caching silently stays off (OpenVINO's own un-set default),
        // never the CB path's "dynamic cache + prefix caching on" combination
        // (which retains blocks unbounded and untracked — see ov_cb_create's
        // doc). Deliberately the safer of the two silent outcomes, but still
        // silent; the Rust caller (lifecycle.rs) logs a warning before this
        // constructor ever runs when it detects this case.
        pipe = new ov::genai::VLMPipeline(
            std::filesystem::path(model_path),
            std::string(device),
            props
        );
    }

    ~OvVlmState() {
        delete pipe;
        pipe = nullptr;
    }
};

} // anonymous namespace

extern "C" {

typedef void* OvVlmHandle;

/// Create a VLMPipeline and return an opaque handle.
///
/// `cache_size_gb` — KV cache size in whole GB, same semantics as
///                   `ov_cb_create`'s parameter of the same name. 0.0 = keep
///                   OpenVINO's own default (dynamic/unbounded — not
///                   recommended, see `OvVlmState`'s constructor doc).
///
/// `enable_prefix_caching` — same semantics as `ov_cb_create`'s parameter of
///                   the same name; only applied when `cache_size_gb > 0.0`.
///                   Measured critical for this server's resent-full-history
///                   chat pattern — see `OvVlmState`'s constructor doc.
///
/// Returns NULL on failure; call ov_last_error() for the message.
OvVlmHandle ov_vlm_create(const char* model_path, const char* device,
                           const char* ov_cache_dir, double cache_size_gb,
                           bool enable_prefix_caching) {
    s_last_error[0] = '\0';
    if (!model_path || !device) {
        set_error("ov_vlm_create: null model_path or device");
        return nullptr;
    }
    try {
        return new OvVlmState(model_path, device, ov_cache_dir, cache_size_gb,
                               enable_prefix_caching);
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("unknown exception in ov_vlm_create");
        return nullptr;
    }
}

/// Destroy a VLMPipeline and free its GPU memory. Passing NULL is a no-op.
void ov_vlm_free(OvVlmHandle handle) {
    if (!handle) { return; }
    // Never let a destructor unwind across the C ABI (→ std::terminate). See
    // ov_cb_free for the rationale.
    try {
        delete static_cast<OvVlmState*>(handle);
    } catch (const std::exception& e) {
        set_error(e.what());
    } catch (...) {
        set_error("unknown exception in ov_vlm_free");
    }
}

/// Generate text from a chat history + images, streaming tokens to callback.
///
/// messages_json  – UTF-8 JSON array of {role, content} objects. On the Rust
///                  side each image-URL part is replaced in the content text by
///                  a `<ov_genai_image_N>` tag (N = zero-based index into
///                  `images`); the decoded pixel data arrives in `images`.
///                  VLMPipeline resolves each tag to images[N] via the template.
/// tools_json     – UTF-8 JSON array of OpenAI-shaped tool definitions, or
///                  NULL/empty when no tools are active for this request.
///                  Set on `ChatHistory` via `set_tools()` before generate() —
///                  VLMPipeline renders its own chat template internally
///                  (unlike the text path's `build_prompt`), and `ChatHistory`
///                  is the only way to hand it a `tools` binding for that
///                  render.
/// images         – array of `num_images` pointers to NHWC uint8 pixel data.
///                  Buffers are owned by Rust and outlive this blocking call.
/// heights/widths – per-image dimensions (height × width × 3 bytes = pixels).
/// num_images     – element count of the three image arrays.
/// params         – sampling parameters (same OvGenParams as the CB bridge).
/// callback       – called once per decoded token text fragment; same
///                  OvTokenCallback type as ov_pipeline_generate. Return 0 to
///                  continue, 1 to abort early (client disconnected).
/// user_data      – passed through to callback unchanged.
/// finish_code_out    – out: 1=STOP (EOS), 2=LENGTH (max_new_tokens hit).
/// input_tokens_out   – out: number of input tokens counted by the model's
///                      tokenizer (from PerfMetrics::num_input_tokens). Written
///                      only on success; may be 0 if metrics are unavailable.
///
/// Returns 0 on success, -1 on error (call ov_last_error()).
int ov_vlm_generate(
    OvVlmHandle                handle,
    const char*                messages_json,
    const char*                tools_json,
    // Extra chat-template variables as a JSON object, or NULL/"" to pass none.
    // Mirrors `tools_json` deliberately: both are per-request additions to the
    // same ChatHistory, and keeping this OUT of OvGenParams avoids that
    // struct's documented size-check/drift landmine.
    //
    // This is how `enable_thinking` reaches a VLM's chat template. Without it
    // the flag is silently dropped, because VLMPipeline applies the template
    // internally (unlike the CB path, which renders via build_prompt).
    // Verified upstream (releases/2026/2): ChatHistoryInternalState::
    // build_normalized_history copies extra_context onto the normalized
    // history, and TokenizerImpl::apply_chat_template merges it into minja's
    // template variables.
    //
    // NOTE: the merge happens AFTER bos_token/eos_token/pad_token are set, so
    // a key named like those would override them. The caller must build this
    // object server-side and never forward client-supplied JSON verbatim.
    const char*                extra_context_json,
    const uint8_t* const*      images,
    const uint32_t*            heights,
    const uint32_t*            widths,
    uint32_t                   num_images,
    const OvGenParams*         params,
    OvTokenCallback            callback,
    void*                      user_data,
    int*                       finish_code_out,
    size_t*                    input_tokens_out,
    // 2026-08-21 (VLM instant-EOS investigation, Part 5):
    // diagnostic-only additions for the VLM-path instant-EOS investigation —
    // the CB path's GenerationStatus::IGNORED fix (commit 123a576) has no
    // direct equivalent here (VLMPipeline::generate is a single blocking
    // call, no handle/status to inspect), so these surface the pipeline's
    // OWN internal token/timing accounting (`VLMDecodedResults::perf_metrics`)
    // instead, letting the Rust side compare it against the streamer
    // callback's own count (0 callbacks but perf_metrics reporting 1
    // generated token distinguishes "sampled EOS immediately" from
    // "never actually ran" — the streamer never receives the EOS token
    // itself, so a real single-token decode still shows 0 callbacks).
    size_t*                    generated_tokens_out,
    double*                    inference_ms_out
) {
    s_last_error[0] = '\0';
    if (!handle)        { set_error("ov_vlm_generate: null handle");        return -1; }
    if (!messages_json) { set_error("ov_vlm_generate: null messages_json"); return -1; }
    if (!params)        { set_error("ov_vlm_generate: null params");        return -1; }

    auto* state = static_cast<OvVlmState*>(handle);

    // ── Build GenerationConfig from OvGenParams ───────────────────────────
    // Shared with the CB path (dedup): this block was a verbatim copy
    // of build_gen_config that had already drifted once (top_k landed only in
    // the shared builder). One builder, identical semantics on both paths.
    ov::genai::GenerationConfig cfg;
    build_gen_config(cfg, params);

    // ── Build image tensors (zero-copy — Rust keeps pixel buffers alive) ──
    std::vector<ov::Tensor> image_tensors;
    image_tensors.reserve(num_images);
    for (uint32_t i = 0; i < num_images; ++i) {
        if (!images || !images[i]) continue;
        ov::Shape shape{1, static_cast<size_t>(heights[i]), static_cast<size_t>(widths[i]), 3};
        // const_cast: VLMPipeline reads pixels, never writes them.
        image_tensors.emplace_back(ov::element::u8, shape, const_cast<uint8_t*>(images[i]));
    }

    // ── Streamer: forward chunks; early stop via callback return value ───
    // NOTE: the callback receives detokenized CHUNKS, not tokens (UTF-8
    // hold-back merges tokens), so a callback counter CANNOT detect the
    // max_new_tokens cut — that misreported `length` as `stop`.
    // The finish reason is derived after generate() from real token counts.
    auto streamer_fn = [callback, user_data]
        (const std::string& token) -> ov::genai::StreamingStatus
    {
        if (callback) {
            int ret = callback(token.c_str(), token.size(), user_data);
            if (ret != 0) return ov::genai::StreamingStatus::STOP;
        }
        return ov::genai::StreamingStatus::RUNNING;
    };

    // ── Parse messages JSON → ChatHistory (+ tools) → generate ────────────
    try {
        auto json_msgs = ov::genai::JsonContainer::from_json_string(
            std::string(messages_json)
        );
        ov::genai::ChatHistory history(json_msgs);
        if (tools_json && tools_json[0] != '\0') {
            history.set_tools(
                ov::genai::JsonContainer::from_json_string(std::string(tools_json))
            );
        }
        if (extra_context_json && extra_context_json[0] != '\0') {
            // Throws on malformed JSON, and asserts the container is
            // object-like — both land in this function's existing catch.
            history.set_extra_context(
                ov::genai::JsonContainer::from_json_string(std::string(extra_context_json))
            );
        }
        auto result = state->pipe->generate(history, image_tensors, cfg, streamer_fn);
        // Finish reason from the pipeline's own token count: generated >= cap
        // means the budget cut the generation → LENGTH. An EOS landing exactly
        // on the final budgeted token also reports LENGTH — the same `>=`
        // ambiguity every server has; accepted. max_tok == 0 (cap opted out,
        // no client budget) keeps the STOP default, as before.
        const size_t max_tok = params->max_new_tokens;
        int finish_code = 1; // 1=STOP (EOS) default
        if (max_tok > 0 &&
            result.perf_metrics.get_num_generated_tokens() >= max_tok) {
            finish_code = 2; // LENGTH
        }
        if (finish_code_out)  *finish_code_out  = finish_code;
        // num_input_tokens is populated by VLMPipeline after generation completes.
        if (input_tokens_out) *input_tokens_out  = result.perf_metrics.num_input_tokens;
        // Diagnostic-only (see param doc above): the pipeline's own count,
        // independent of the streamer callback.
        if (generated_tokens_out) {
            *generated_tokens_out = result.perf_metrics.get_num_generated_tokens();
        }
        if (inference_ms_out) {
            *inference_ms_out =
                static_cast<double>(result.perf_metrics.get_inference_duration().mean);
        }
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception in ov_vlm_generate");
        return -1;
    }
}

/// Count the tokens `text` (`len` bytes; need not be null-terminated) encodes
/// to with the VLM's tokenizer. Count-only mirror of ov_cb_encode's
/// out_ids=NULL mode — the L0 prompt-length gate (chat.rs) only ever needs the
/// length, never the ids themselves, since VLMPipeline::generate takes raw
/// messages/images, not pre-tokenized ids the way the CB engine does.
///
/// This is an approximation of what VLMPipeline will actually submit (it
/// counts the gate's own concatenated message text, not the fully
/// template-rendered prompt VLMPipeline builds internally) — close enough to
/// catch a grossly oversized prompt before it reaches the GPU, which is the
/// failure mode this exists to prevent (a CL_OUT_OF_RESOURCES that poisons
/// the OpenCL context for the whole process, not just this request).
///
/// Returns 0 on success (writes *out_count), -1 on error (call ov_last_error()).
int ov_vlm_count_tokens(OvVlmHandle handle,
                        const char* text,
                        size_t      len,
                        size_t*     out_count) {
    s_last_error[0] = '\0';
    if (!handle)      { set_error("ov_vlm_count_tokens: null handle"); return -1; }
    if (!text && len) { set_error("ov_vlm_count_tokens: null text with non-zero len"); return -1; }
    if (!out_count)   { set_error("ov_vlm_count_tokens: null out_count"); return -1; }
    auto* state = static_cast<OvVlmState*>(handle);
    try {
        ov::genai::TokenizedInputs enc = state->tok.encode(std::string(text, len));
        *out_count = enc.input_ids.get_size();
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception in ov_vlm_count_tokens");
        return -1;
    }
}

} // extern "C" (VLM section)

// ═════════════════════════════════════════════════════════════════════════════
// TextEmbeddingPipeline C-API (R4/G3 — the first real "media" registry flow)
// ─────────────────────────────────────────────────────────────────────────────
// Thin extern "C" wrapper around ov::genai::TextEmbeddingPipeline. The pipeline
// is single-stream and blocking; Rust runs it on a dedicated embed-engine
// thread (same pattern as the VLM wrapper above). It internally tokenizes,
// runs the encoder, pools (CLS/MEAN/LAST_TOKEN) and optionally L2-normalizes,
// returning one float vector per input document.
//
// We also bundle a standalone ov::genai::Tokenizer loaded from the same model
// directory so the embeddings endpoint can report honest usage.prompt_tokens —
// TextEmbeddingPipeline does not expose its own tokenizer. (stormVINO counts
// tokens the same way, separately from the embedding call.)
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Map the int pooling code from Rust to the pipeline enum. Matches
// TextEmbeddingPipeline::PoolingType: 0=CLS, 1=MEAN, 2=LAST_TOKEN. Unknown
// codes fall back to MEAN (the e5 / stormVINO default), never silently CLS.
ov::genai::TextEmbeddingPipeline::PoolingType pooling_from_code(int code) {
    using PT = ov::genai::TextEmbeddingPipeline::PoolingType;
    switch (code) {
        case 0:  return PT::CLS;
        case 2:  return PT::LAST_TOKEN;
        case 1:  // fallthrough — MEAN
        default: return PT::MEAN;
    }
}

// OvEmbedState bundles a heap-allocated TextEmbeddingPipeline + a tokenizer for
// token counting. Rust sees the whole thing as *mut c_void (OvEmbedHandle).
struct OvEmbedState {
    ov::genai::TextEmbeddingPipeline* pipe = nullptr;
    ov::genai::Tokenizer              tok;

    OvEmbedState(const char* model_path, const char* device,
                 int pooling_type, bool normalize)
        : tok(std::filesystem::path(model_path))
    {
        ov::genai::TextEmbeddingPipeline::Config cfg;
        cfg.pooling_type = pooling_from_code(pooling_type);
        cfg.normalize    = normalize;
        pipe = new ov::genai::TextEmbeddingPipeline(
            std::filesystem::path(model_path),
            std::string(device),
            cfg
        );
    }

    ~OvEmbedState() {
        delete pipe;
        pipe = nullptr;
    }
};

} // anonymous namespace

extern "C" {

typedef void* OvEmbedHandle;

/// Called once per embedded document with its index and float vector.
/// `vec` points to `dim` floats, valid only for the duration of the call.
typedef void (*OvEmbedCallback)(void* user_data, size_t index,
                                const float* vec, size_t dim);

/// Create a TextEmbeddingPipeline (+ tokenizer) and return an opaque handle.
/// `pooling_type`: 0=CLS, 1=MEAN, 2=LAST_TOKEN. `normalize`: non-zero = L2.
/// Returns NULL on failure; call ov_last_error() for the message.
OvEmbedHandle ov_embed_create(const char* model_path, const char* device,
                              int pooling_type, int normalize) {
    s_last_error[0] = '\0';
    if (!model_path || !device) {
        set_error("ov_embed_create: null model_path or device");
        return nullptr;
    }
    try {
        return new OvEmbedState(model_path, device, pooling_type, normalize != 0);
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("unknown exception in ov_embed_create");
        return nullptr;
    }
}

/// Destroy a TextEmbeddingPipeline and free its GPU memory. NULL is a no-op.
void ov_embed_free(OvEmbedHandle handle) {
    if (!handle) { return; }
    // Never let a destructor unwind across the C ABI (→ std::terminate). See
    // ov_cb_free for the rationale.
    try {
        delete static_cast<OvEmbedState*>(handle);
    } catch (const std::exception& e) {
        set_error(e.what());
    } catch (...) {
        set_error("unknown exception in ov_embed_free");
    }
}

/// Embed `n` documents. `texts` is an array of `n` null-terminated UTF-8
/// strings. For each document, `callback` is invoked once with its index and
/// float vector (in input order). Returns 0 on success, -1 on error
/// (call ov_last_error()). The embeddings are pooled + normalized per the
/// pipeline config; with the default float output the result is the
/// std::vector<std::vector<float>> variant of EmbeddingResults.
int ov_embed_documents(OvEmbedHandle    handle,
                       const char* const* texts,
                       size_t           n,
                       OvEmbedCallback  callback,
                       void*            user_data) {
    s_last_error[0] = '\0';
    if (!handle)        { set_error("ov_embed_documents: null handle"); return -1; }
    if (!texts && n)    { set_error("ov_embed_documents: null texts with non-zero n"); return -1; }
    if (!callback)      { set_error("ov_embed_documents: null callback"); return -1; }
    auto* state = static_cast<OvEmbedState*>(handle);
    try {
        std::vector<std::string> docs;
        docs.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            docs.emplace_back(texts[i] ? texts[i] : "");
        }

        ov::genai::EmbeddingResults results = state->pipe->embed_documents(docs);

        // Default (non-quantized) output is the float variant. If a future
        // config requests int8/uint8 embeddings this would need handling; for
        // now treat anything else as an error rather than silently mis-reading.
        const auto* floats =
            std::get_if<std::vector<std::vector<float>>>(&results);
        if (!floats) {
            set_error("ov_embed_documents: non-float embedding output not supported");
            return -1;
        }
        for (size_t i = 0; i < floats->size(); ++i) {
            const std::vector<float>& v = (*floats)[i];
            callback(user_data, i, v.data(), v.size());
        }
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception in ov_embed_documents");
        return -1;
    }
}

/// Count the tokens `text` (null-terminated UTF-8) encodes to, using the
/// embedding model's tokenizer. For usage.prompt_tokens. Returns SIZE_MAX on
/// error (call ov_last_error()); a valid count otherwise (0 for empty text).
size_t ov_embed_count_tokens(OvEmbedHandle handle, const char* text) {
    s_last_error[0] = '\0';
    if (!handle) { set_error("ov_embed_count_tokens: null handle"); return SIZE_MAX; }
    auto* state = static_cast<OvEmbedState*>(handle);
    try {
        ov::genai::TokenizedInputs enc = state->tok.encode(std::string(text ? text : ""));
        return enc.input_ids.get_size();
    } catch (const std::exception& e) {
        set_error(e.what());
        return SIZE_MAX;
    } catch (...) {
        set_error("unknown exception in ov_embed_count_tokens");
        return SIZE_MAX;
    }
}

} // extern "C" (embedding section)

// ═════════════════════════════════════════════════════════════════════════════
// TextRerankPipeline C-API (/v1/rerank)
// ─────────────────────────────────────────────────────────────────────────────
// Thin extern "C" wrapper around ov::genai::TextRerankPipeline.
// Takes a query string + a batch of documents, returns (original_index, score)
// pairs sorted by relevance score descending via a per-result callback.
// The pipeline is created once per model load; top_n is set to a large value
// so all documents are scored and Rust applies the per-request top_n trim.
//
// Like TextEmbeddingPipeline, TextRerankPipeline does not expose its own
// tokenizer (no get_tokenizer()), so we bundle a standalone ov::genai::Tokenizer
// loaded from the same model directory — used only for the pre-inference length
// gate (the project's internal engineering log), not for reranking
// itself.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

struct OvRerankState {
    ov::genai::TextRerankPipeline* pipe = nullptr;
    ov::genai::Tokenizer           tok;

    OvRerankState(const char* model_path, const char* device, size_t top_n)
        : tok(std::filesystem::path(model_path))
    {
        ov::genai::TextRerankPipeline::Config cfg;
        cfg.top_n = top_n;
        pipe = new ov::genai::TextRerankPipeline(
            std::filesystem::path(model_path), std::string(device), cfg);
    }

    ~OvRerankState() {
        delete pipe;
        pipe = nullptr;
    }
};

} // anonymous namespace

extern "C" {

typedef void* OvRerankHandle;
typedef void (*OvRerankCallback)(void* user_data, size_t index, float score);

/// Create a TextRerankPipeline and return an opaque handle.
/// top_n: maximum documents to return (pass a large value to get all scored).
OvRerankHandle ov_rerank_create(const char* model_path, const char* device, size_t top_n) {
    s_last_error[0] = '\0';
    if (!model_path || !device) {
        set_error("ov_rerank_create: null model_path or device");
        return nullptr;
    }
    try {
        return new OvRerankState(model_path, device, top_n);
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("unknown exception in ov_rerank_create");
        return nullptr;
    }
}

/// Destroy a TextRerankPipeline. NULL is a no-op.
void ov_rerank_free(OvRerankHandle handle) {
    s_last_error[0] = '\0';
    if (!handle) return;
    try {
        delete static_cast<OvRerankState*>(handle);
    } catch (...) {
        set_error("unknown exception in ov_rerank_free");
    }
}

/// Rerank documents against query. Calls callback once per result (sorted by
/// score descending, up to top_n). Returns 0 on success, -1 on error.
int ov_rerank_documents(OvRerankHandle     handle,
                        const char*        query,
                        const char* const* texts,
                        size_t             n,
                        OvRerankCallback   callback,
                        void*              user_data) {
    s_last_error[0] = '\0';
    if (!handle)     { set_error("ov_rerank_documents: null handle");   return -1; }
    if (!query)      { set_error("ov_rerank_documents: null query");    return -1; }
    if (!texts && n) { set_error("ov_rerank_documents: null texts");    return -1; }
    if (!callback)   { set_error("ov_rerank_documents: null callback"); return -1; }
    auto* state = static_cast<OvRerankState*>(handle);
    try {
        std::vector<std::string> docs;
        docs.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            docs.emplace_back(texts[i] ? texts[i] : "");
        }
        auto results = state->pipe->rerank(std::string(query), docs);
        for (const auto& [idx, score] : results) {
            callback(user_data, idx, score);
        }
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception in ov_rerank_documents");
        return -1;
    }
}

/// Count the tokens `(query, document)` need combined, for the pre-inference
/// length gate (the project's internal engineering log).
///
/// Encodes `query` and `document` as two independent single-text sequences
/// and sums them, rather than using the tokenizer's paired-prompt encoding
/// overload — live-verified that overload throws on this fleet's actual
/// converted `openvino_tokenizer.xml` ("Input port for index 1 was not
/// found! The model has only 1 inputs."): not every tokenizer export
/// compiles a genuine two-input pair-encoding graph, so relying on it isn't
/// portable across reranking models. Two independent counts slightly
/// overcounts versus the real `[CLS] query [SEP] document [SEP]` sequence
/// (no shared truncation, and BOS/EOS may be counted on both sides) — a
/// conservative direction for a length *ceiling* check, not an exact
/// tokenizer-parity requirement, so overcounting only ever makes the gate
/// fire slightly earlier than a truly precise count would, never later.
/// Returns SIZE_MAX on error (call ov_last_error()); a valid count otherwise.
size_t ov_rerank_count_tokens(OvRerankHandle handle, const char* query, const char* document) {
    s_last_error[0] = '\0';
    if (!handle) { set_error("ov_rerank_count_tokens: null handle"); return SIZE_MAX; }
    auto* state = static_cast<OvRerankState*>(handle);
    try {
        ov::genai::TokenizedInputs query_enc =
            state->tok.encode(std::string(query ? query : ""));
        ov::genai::TokenizedInputs doc_enc =
            state->tok.encode(std::string(document ? document : ""));
        return query_enc.input_ids.get_size() + doc_enc.input_ids.get_size();
    } catch (const std::exception& e) {
        set_error(e.what());
        return SIZE_MAX;
    } catch (...) {
        set_error("unknown exception in ov_rerank_count_tokens");
        return SIZE_MAX;
    }
}

} // extern "C" (reranking section)

// ═════════════════════════════════════════════════════════════════════════════
// WhisperPipeline C-API (Phase 5.1b — speech-to-text)
// ─────────────────────────────────────────────────────────────────────────────
// Thin extern "C" wrapper around ov::genai::WhisperPipeline. The pipeline takes
// a RawSpeechInput (std::vector<float>, normalized ~[-1,1] @ 16 kHz — the Rust
// side decodes + resamples) and returns WhisperDecodedResults (full text +
// detected language + optional per-segment chunks). Single-stream and blocking;
// Rust runs it on a dedicated STT engine thread (same pattern as VLM/embed).
//
// Results are delivered via callbacks (same approach as ov_embed_documents) to
// avoid output-buffer sizing: text_cb fires once with the transcript; chunk_cb
// fires once per timestamped segment (only when return_timestamps != 0). The
// detected language is written into the caller-provided language_out buffer.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// OvWhisperState bundles a heap-allocated WhisperPipeline.
struct OvWhisperState {
    ov::genai::WhisperPipeline* pipe = nullptr;

    OvWhisperState(const char* model_path, const char* device, const char* ov_cache_dir) {
        ov::AnyMap props;
        if (ov_cache_dir && ov_cache_dir[0] != '\0') {
            props.emplace(ov::cache_dir(std::string(ov_cache_dir)));
        }
        pipe = new ov::genai::WhisperPipeline(
            std::filesystem::path(model_path),
            std::string(device),
            props
        );
    }

    ~OvWhisperState() {
        delete pipe;
        pipe = nullptr;
    }
};

} // anonymous namespace

extern "C" {

typedef void* OvWhisperHandle;

// Delivers the full transcript once. text is NOT null-terminated; use len.
typedef void (*OvWhisperTextCallback)(void* user_data, const char* text, size_t len);
// Delivers one timestamped segment. start_ts/end_ts in seconds; text via len.
typedef void (*OvWhisperChunkCallback)(void* user_data, float start_ts, float end_ts,
                                       const char* text, size_t len);

/// Create a WhisperPipeline and return an opaque handle.
/// Returns NULL on failure; call ov_last_error() for the message.
OvWhisperHandle ov_whisper_create(const char* model_path, const char* device,
                                  const char* ov_cache_dir) {
    s_last_error[0] = '\0';
    if (!model_path || !device) {
        set_error("ov_whisper_create: null model_path or device");
        return nullptr;
    }
    try {
        return new OvWhisperState(model_path, device, ov_cache_dir);
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("unknown exception in ov_whisper_create");
        return nullptr;
    }
}

/// Destroy a WhisperPipeline and free its GPU memory. Passing NULL is a no-op.
void ov_whisper_free(OvWhisperHandle handle) {
    if (!handle) { return; }
    // Never let a destructor unwind across the C ABI (→ std::terminate).
    try {
        delete static_cast<OvWhisperState*>(handle);
    } catch (const std::exception& e) {
        set_error(e.what());
    } catch (...) {
        set_error("unknown exception in ov_whisper_free");
    }
}

/// Transcribe `num_samples` floats of 16 kHz mono PCM.
///
/// samples           – pointer to num_samples f32 PCM samples (Rust-owned,
///                      alive for this blocking call). normalized ~[-1, 1].
/// language          – source-language code ("en", "pl", …) or NULL to
///                     autodetect. Wrapped into the "<|xx|>" whisper token here.
/// return_timestamps – 0/1; when 1, chunk_cb fires per timestamped segment.
/// translate         – 0/1; when 1, Whisper's "translate" task: the output is
///                     English whatever the spoken language (multilingual
///                     models only — the caller checks). 0 keeps the model's
///                     default task, transcription.
/// text_cb           – called once with the full transcript.
/// chunk_cb          – called per segment when return_timestamps != 0.
/// user_data         – passed through to both callbacks unchanged.
/// language_out      – buffer for the detected language; NUL-terminated, may be
///                     NULL to skip. language_cap = its capacity in bytes.
///
/// Returns 0 on success, -1 on error (call ov_last_error()).
int ov_whisper_generate(
    OvWhisperHandle        handle,
    const float*           samples,
    size_t                 num_samples,
    const char*            language,
    int                    return_timestamps,
    int                    translate,
    OvWhisperTextCallback  text_cb,
    OvWhisperChunkCallback chunk_cb,
    void*                  user_data,
    char*                  language_out,
    size_t                 language_cap
) {
    s_last_error[0] = '\0';
    if (!handle)                  { set_error("ov_whisper_generate: null handle");  return -1; }
    if (!samples && num_samples)  { set_error("ov_whisper_generate: null samples"); return -1; }

    auto* state = static_cast<OvWhisperState*>(handle);

    // Copy PCM into the RawSpeechInput vector<float> the pipeline expects.
    ov::genai::RawSpeechInput raw(samples, samples + num_samples);

    // Start from the model's own generation config (decoder-start tokens, etc.).
    ov::genai::WhisperGenerationConfig cfg = state->pipe->get_generation_config();
    if (language && language[0] != '\0') {
        cfg.language = std::string("<|") + language + "|>";
    }
    cfg.return_timestamps = (return_timestamps != 0);
    if (translate != 0) {
        cfg.task = std::string("translate");
    }

    try {
        ov::genai::WhisperDecodedResults result = state->pipe->generate(raw, cfg);

        if (text_cb && !result.texts.empty()) {
            const std::string& t = result.texts[0];
            text_cb(user_data, t.c_str(), t.size());
        }
        if (chunk_cb && result.chunks.has_value()) {
            for (const auto& ch : *result.chunks) {
                chunk_cb(user_data, ch.start_ts, ch.end_ts, ch.text.c_str(), ch.text.size());
            }
        }
        if (language_out && language_cap > 0) {
            std::strncpy(language_out, result.language.c_str(), language_cap - 1);
            language_out[language_cap - 1] = '\0';
        }
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception in ov_whisper_generate");
        return -1;
    }
}

} // extern "C" (whisper section)

// ═════════════════════════════════════════════════════════════════════════════
// Text2ImagePipeline C-API (Phase 5.3b — text-to-image / SDXL)
// ─────────────────────────────────────────────────────────────────────────────
// Thin extern "C" wrapper around ov::genai::Text2ImagePipeline. The single-device
// constructor auto-detects the model family (SDXL, SD, SD3, FLUX) from the model
// dir and compiles every submodel on one device — the first-pass placement.
// Per-submodel device split (compile(text, denoise, vae)) is a later refinement
// for heterogeneous GPU.0/GPU.1 boxes (see the project's internal engineering log).
//
// generate() returns an ov::Tensor shaped [N, H, W, 3], NHWC u8 (0–255). Each
// image is delivered via image_cb (same per-index callback pattern as
// ov_embed_documents): the callback receives a pointer to that image's
// H*W*3 contiguous bytes, valid only for the duration of the call. Rust copies
// the bytes out and does PNG encoding — image codecs stay out of C++.
// ═════════════════════════════════════════════════════════════════════════════

namespace {

// OvImageState bundles the image pipelines for one model load.
//
// LOAD ARCHITECTURE (1× VRAM — see the project's internal engineering log 2026-06-16):
//   OV GenAI's pipeline conversion graph is directional — Image2Image ⇄ Inpainting
//   are mutually constructible, and Text2Image derives from either, but NOT the
//   reverse. So the only single-load design that backs all three ops is to load
//   an InpaintingPipeline as the BASE and derive the other two from it; the derived
//   pipelines share the base's already-compiled submodels (no extra VRAM).
//
//   The inpaint base needs a vae_encoder. If the model lacks one (or inpaint
//   construction fails for any reason), we fall back to a direct Text2ImagePipeline:
//   generations still work; edits report `edits_available == false`.
struct OvImageState {
    ov::genai::Text2ImagePipeline*  t2i     = nullptr;  // generations (always present)
    ov::genai::Image2ImagePipeline* img2img = nullptr;  // /edits, no mask
    ov::genai::InpaintingPipeline*  inpaint = nullptr;  // /edits, with mask (base load)
    bool edits_available = false;

    // FLUX's and SD3's inpaint→t2i derivation is BROKEN in OV GenAI 2026.2:
    // the InpaintingPipeline loads, but a Text2ImagePipeline derived from it
    // crashes on generate() with "longjmp causes uninitialized stack frame"
    // (FLUX: segfault verified; SD3.5 Medium: same fatal crash verified).
    // Both load fine as plain Text2ImagePipeline — generations work; edits report
    // unavailable until OV GenAI fixes the derive path for these families.
    static std::string pipeline_class(const std::filesystem::path& model_dir) {
        std::ifstream f(model_dir / "model_index.json");
        if (!f) {
            return {};
        }
        const std::string s((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
        // Extract "_class_name" value — simple string search, no JSON parser needed.
        const std::string key = "\"_class_name\"";
        auto pos = s.find(key);
        if (pos == std::string::npos) return {};
        auto q1 = s.find('"', pos + key.size() + 1);
        if (q1 == std::string::npos) return {};
        auto q2 = s.find('"', q1 + 1);
        if (q2 == std::string::npos) return {};
        return s.substr(q1 + 1, q2 - q1 - 1);
    }

    static bool generations_only(const std::filesystem::path& model_dir) {
        const std::string cls = pipeline_class(model_dir);
        return cls == "FluxPipeline" || cls == "StableDiffusion3Pipeline";
    }

    OvImageState(const char* model_path, const char* device, const char* ov_cache_dir) {
        ov::AnyMap props;
        if (ov_cache_dir && ov_cache_dir[0] != '\0') {
            props.emplace(ov::cache_dir(std::string(ov_cache_dir)));
        }
        const std::filesystem::path path(model_path);
        const std::string dev(device);

        // FLUX + SD3: skip the inpaint-base architecture — it crashes on generate().
        // A crash can't be caught, so this MUST be decided by family up front.
        if (generations_only(path)) {
            t2i = new ov::genai::Text2ImagePipeline(path, dev, props);
            edits_available = false;
            return;
        }

        try {
            // Inpaint-base: one compile, all three ops share the submodels.
            inpaint = new ov::genai::InpaintingPipeline(path, dev, props);
            img2img = new ov::genai::Image2ImagePipeline(*inpaint);
            t2i     = new ov::genai::Text2ImagePipeline(*inpaint);
            edits_available = true;
        } catch (const std::exception& e) {
            // No vae_encoder (or inpaint unsupported) → generations-only fallback.
            delete t2i;     t2i     = nullptr;
            delete img2img; img2img = nullptr;
            delete inpaint; inpaint = nullptr;
            t2i = new ov::genai::Text2ImagePipeline(path, dev, props);
            edits_available = false;
        }
    }

    ~OvImageState() {
        delete t2i;     t2i     = nullptr;
        delete img2img; img2img = nullptr;
        delete inpaint; inpaint = nullptr;
    }
};

} // anonymous namespace

extern "C" {

typedef void* OvImageHandle;

// Delivers one generated image. `data` points to `height * width * 3` contiguous
// NHWC RGB u8 bytes, valid only for the duration of the call.
typedef void (*OvImageCallback)(void* user_data, size_t index,
                                const uint8_t* data, uint32_t height, uint32_t width);

/// Validate an [N,H,W,3] u8 result tensor and deliver each image via image_cb.
/// `who` names the calling function for error messages. Returns 0 / -1.
static int deliver_image_tensor(const ov::Tensor& result, OvImageCallback image_cb,
                                void* user_data, const char* who) {
    if (result.get_element_type() != ov::element::u8) {
        set_error((std::string(who) + ": unexpected output element type (not u8)").c_str());
        return -1;
    }
    const ov::Shape shape = result.get_shape();
    if (shape.size() != 4 || shape[3] != 3) {
        set_error((std::string(who) + ": unexpected output shape (not [N,H,W,3])").c_str());
        return -1;
    }
    const size_t n = shape[0];
    const uint32_t h = static_cast<uint32_t>(shape[1]);
    const uint32_t w = static_cast<uint32_t>(shape[2]);
    const size_t per_image = static_cast<size_t>(h) * w * 3;
    const uint8_t* base = static_cast<const uint8_t*>(result.data());
    for (size_t i = 0; i < n; ++i) {
        image_cb(user_data, i, base + i * per_image, h, w);
    }
    return 0;
}

/// Load the image model and return an opaque handle. Loads an InpaintingPipeline
/// base (deriving Text2Image + Image2Image from it) when the model has a
/// vae_encoder, else falls back to a generations-only Text2ImagePipeline.
/// Returns NULL on failure; call ov_last_error() for the message.
OvImageHandle ov_image_create(const char* model_path, const char* device,
                              const char* ov_cache_dir) {
    s_last_error[0] = '\0';
    if (!model_path || !device) {
        set_error("ov_image_create: null model_path or device");
        return nullptr;
    }
    try {
        return new OvImageState(model_path, device, ov_cache_dir);
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("unknown exception in ov_image_create");
        return nullptr;
    }
}

/// Destroy a Text2ImagePipeline and free its GPU memory. Passing NULL is a no-op.
void ov_image_free(OvImageHandle handle) {
    if (!handle) { return; }
    // Never let a destructor unwind across the C ABI (→ std::terminate).
    try {
        delete static_cast<OvImageState*>(handle);
    } catch (const std::exception& e) {
        set_error(e.what());
    } catch (...) {
        set_error("unknown exception in ov_image_free");
    }
}

/// Generate one or more images from `prompt`.
///
/// prompt              – positive prompt (null-terminated UTF-8).
/// negative_prompt     – negative prompt, or NULL/empty to omit.
/// width, height       – output size in pixels (must be divisible by 8).
/// num_inference_steps – denoising steps (0 → leave the pipeline default).
/// num_images          – images to generate this call (maps to
///                       num_images_per_prompt). The output tensor is batched.
/// use_seed            – non-zero pins the RNG seed for reproducibility.
/// seed                – the seed, used only when use_seed != 0.
/// guidance_scale      – CFG scale; <= 0 leaves the pipeline default.
/// image_cb            – called once per generated image (in batch order).
/// user_data           – passed through to image_cb unchanged.
///
/// Returns 0 on success, -1 on error (call ov_last_error()).
int ov_image_generate(
    OvImageHandle   handle,
    const char*     prompt,
    const char*     negative_prompt,
    uint32_t        width,
    uint32_t        height,
    uint32_t        num_inference_steps,
    uint32_t        num_images,
    int             use_seed,
    uint64_t        seed,
    float           guidance_scale,
    OvImageCallback image_cb,
    void*           user_data
) {
    s_last_error[0] = '\0';
    if (!handle)   { set_error("ov_image_generate: null handle");   return -1; }
    if (!prompt)   { set_error("ov_image_generate: null prompt");   return -1; }
    if (!image_cb) { set_error("ov_image_generate: null image_cb"); return -1; }

    auto* state = static_cast<OvImageState*>(handle);

    try {
        ov::AnyMap props;
        if (width > 0)  { props.emplace(ov::genai::width(static_cast<int64_t>(width))); }
        if (height > 0) { props.emplace(ov::genai::height(static_cast<int64_t>(height))); }
        if (num_inference_steps > 0) {
            props.emplace(ov::genai::num_inference_steps(static_cast<size_t>(num_inference_steps)));
        }
        props.emplace(ov::genai::num_images_per_prompt(
            static_cast<size_t>(num_images == 0 ? 1 : num_images)));
        if (negative_prompt && negative_prompt[0] != '\0') {
            props.emplace(ov::genai::negative_prompt(std::string(negative_prompt)));
        }
        if (guidance_scale > 0.0f) {
            props.emplace(ov::genai::guidance_scale(guidance_scale));
        }
        if (use_seed) {
            props.emplace(ov::genai::rng_seed(static_cast<size_t>(seed)));
        }

        ov::Tensor result = state->t2i->generate(std::string(prompt), props);
        return deliver_image_tensor(result, image_cb, user_data,
                                    "ov_image_generate");
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception in ov_image_generate");
        return -1;
    }
}

/// Edit an existing image: inpaint (mask present) or img2img (mask absent).
///
/// prompt              – positive prompt (null-terminated UTF-8).
/// negative_prompt     – negative prompt, or NULL/empty to omit.
/// init_data           – source image, `init_h * init_w * 3` NHWC RGB u8.
/// init_h, init_w      – source image dims. Output size matches (rounded down to
///                       the VAE scale factor by the pipeline).
/// mask_data           – mask image `mask_h * mask_w * 3` NHWC RGB u8, or NULL for
///                       img2img. White (255) marks the region to regenerate.
/// mask_h, mask_w      – mask dims (caller ensures they match the init image).
/// num_inference_steps – denoising steps (0 → pipeline default).
/// num_images          – images to generate (num_images_per_prompt).
/// use_seed / seed     – pin the RNG seed when use_seed != 0.
/// guidance_scale      – CFG scale; <= 0 leaves the pipeline default.
/// strength            – denoise strength (0,1]; <= 0 leaves the pipeline default.
/// image_cb / user_data– called once per output image (batch order).
///
/// Returns 0 on success, -1 on error (call ov_last_error()).
int ov_image_edit(
    OvImageHandle   handle,
    const char*     prompt,
    const char*     negative_prompt,
    const uint8_t*  init_data,
    uint32_t        init_h,
    uint32_t        init_w,
    const uint8_t*  mask_data,
    uint32_t        mask_h,
    uint32_t        mask_w,
    uint32_t        num_inference_steps,
    uint32_t        num_images,
    int             use_seed,
    uint64_t        seed,
    float           guidance_scale,
    float           strength,
    OvImageCallback image_cb,
    void*           user_data
) {
    s_last_error[0] = '\0';
    if (!handle)    { set_error("ov_image_edit: null handle");    return -1; }
    if (!prompt)    { set_error("ov_image_edit: null prompt");    return -1; }
    if (!init_data) { set_error("ov_image_edit: null init_data"); return -1; }
    if (!image_cb)  { set_error("ov_image_edit: null image_cb");  return -1; }

    auto* state = static_cast<OvImageState*>(handle);
    if (!state->edits_available) {
        set_error("ov_image_edit: this model has no vae_encoder — edits unsupported");
        return -1;
    }

    try {
        ov::AnyMap props;
        if (num_inference_steps > 0) {
            props.emplace(ov::genai::num_inference_steps(static_cast<size_t>(num_inference_steps)));
        }
        props.emplace(ov::genai::num_images_per_prompt(
            static_cast<size_t>(num_images == 0 ? 1 : num_images)));
        if (negative_prompt && negative_prompt[0] != '\0') {
            props.emplace(ov::genai::negative_prompt(std::string(negative_prompt)));
        }
        if (guidance_scale > 0.0f) {
            props.emplace(ov::genai::guidance_scale(guidance_scale));
        }
        if (strength > 0.0f) {
            props.emplace(ov::genai::strength(strength));
        }
        if (use_seed) {
            props.emplace(ov::genai::rng_seed(static_cast<size_t>(seed)));
        }

        // Wrap the caller's pixels in [1,H,W,3] u8 tensors. The pipeline reads
        // them during generate(); the Rust side keeps the buffers alive across
        // the call, so a non-owning tensor view is safe.
        ov::Tensor init_tensor(ov::element::u8,
                               ov::Shape{1, init_h, init_w, 3},
                               const_cast<uint8_t*>(init_data));

        ov::Tensor result;
        if (mask_data) {
            // InpaintingPipeline defaults to the model's native size (1024² for
            // SDXL) rather than deriving from the init image the way img2img does.
            // Pin the output to the init dims (rounded down to the VAE scale
            // factor of 8) so an edit returns an image the size of its input —
            // the OpenAI /edits contract. img2img needs no such pin.
            props.emplace(ov::genai::width(static_cast<int64_t>((init_w / 8) * 8)));
            props.emplace(ov::genai::height(static_cast<int64_t>((init_h / 8) * 8)));
            ov::Tensor mask_tensor(ov::element::u8,
                                   ov::Shape{1, mask_h, mask_w, 3},
                                   const_cast<uint8_t*>(mask_data));
            result = state->inpaint->generate(std::string(prompt), init_tensor,
                                              mask_tensor, props);
        } else {
            result = state->img2img->generate(std::string(prompt), init_tensor, props);
        }
        return deliver_image_tensor(result, image_cb, user_data, "ov_image_edit");
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception in ov_image_edit");
        return -1;
    }
}

} // extern "C" (image section)

// ═════════════════════════════════════════════════════════════════════════════
// Text2SpeechPipeline C-API (Phase 5.2a — text-to-speech / SpeechT5)
// ─────────────────────────────────────────────────────────────────────────────
// Thin extern "C" wrapper around ov::genai::Text2SpeechPipeline. The pipeline
// takes a text string and returns Text2SpeechDecodedResults with
// speeches: std::vector<ov::Tensor> — each tensor is a 1D f32 waveform at
// 16 kHz. Single-stream and blocking; Rust runs it on a dedicated TTS engine
// thread (same pattern as Whisper/embed).
//
// Results are delivered via a samples callback (one call per waveform tensor)
// to avoid output-buffer sizing. For a single input string there is exactly
// one entry in speeches.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// OvTtsState bundles a heap-allocated Text2SpeechPipeline.
struct OvTtsState {
    ov::genai::Text2SpeechPipeline* pipe = nullptr;

    OvTtsState(const char* model_path, const char* device, const char* ov_cache_dir) {
        ov::AnyMap props;
        if (ov_cache_dir && ov_cache_dir[0] != '\0') {
            props.emplace(ov::cache_dir(std::string(ov_cache_dir)));
        }
        pipe = new ov::genai::Text2SpeechPipeline(
            std::filesystem::path(model_path),
            std::string(device),
            props
        );
    }

    ~OvTtsState() {
        delete pipe;
        pipe = nullptr;
    }
};

} // anonymous namespace

extern "C" {

typedef void* OvTtsHandle;

// Delivers one waveform: data points to num_samples f32 at 16 kHz. index is
// the position in Text2SpeechDecodedResults.speeches (always 0 for a single
// input string).
typedef void (*OvTtsSamplesCallback)(void* user_data, size_t index,
                                     const float* data, size_t num_samples);

/// Create a Text2SpeechPipeline and return an opaque handle.
/// Returns NULL on failure; call ov_last_error() for the message.
OvTtsHandle ov_tts_create(const char* model_path, const char* device,
                           const char* ov_cache_dir) {
    s_last_error[0] = '\0';
    if (!model_path || !device) {
        set_error("ov_tts_create: null model_path or device");
        return nullptr;
    }
    try {
        return new OvTtsState(model_path, device, ov_cache_dir);
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("unknown exception in ov_tts_create");
        return nullptr;
    }
}

/// Destroy a Text2SpeechPipeline and free its device memory. NULL is a no-op.
void ov_tts_free(OvTtsHandle handle) {
    if (!handle) { return; }
    try {
        delete static_cast<OvTtsState*>(handle);
    } catch (const std::exception& e) {
        set_error(e.what());
    } catch (...) {
        set_error("unknown exception in ov_tts_free");
    }
}

/// Synthesise speech for text.
///
/// text       – NUL-terminated input text.
/// samples_cb – called once per waveform in Text2SpeechDecodedResults.speeches.
/// user_data  – passed through to samples_cb unchanged.
///
/// Returns 0 on success, -1 on error (call ov_last_error()).
int ov_tts_generate(OvTtsHandle handle, const char* text,
                    OvTtsSamplesCallback samples_cb, void* user_data) {
    s_last_error[0] = '\0';
    if (!handle) { set_error("ov_tts_generate: null handle"); return -1; }
    if (!text)   { set_error("ov_tts_generate: null text");   return -1; }

    auto* state = static_cast<OvTtsState*>(handle);
    try {
        ov::genai::Text2SpeechDecodedResults result =
            state->pipe->generate(std::string(text));

        if (samples_cb) {
            for (size_t i = 0; i < result.speeches.size(); ++i) {
                const ov::Tensor& t = result.speeches[i];
                // The pipeline may return a GPU-side (remote) tensor when the
                // device is a dGPU/iGPU. Copy to a host tensor before reading
                // the raw float pointer — copy_to is a no-op when already on host.
                ov::Tensor host(t.get_element_type(), t.get_shape());
                t.copy_to(host);
                const float* data = host.data<float>();
                size_t n = host.get_size();
                samples_cb(user_data, i, data, n);
            }
        }
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown exception in ov_tts_generate");
        return -1;
    }
}

} // extern "C" (TTS section)
