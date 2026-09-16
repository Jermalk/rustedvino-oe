// ============================================================
// src/model_manager/stub_engines.rs — R3 future-flow seams
// ============================================================
// Placeholder engine handles for the model kinds the registry is
// *designed* to host but does not yet *execute*: text embeddings,
// speech-to-text, text-to-speech, and text-to-image. R3 adds them as
// seams only — no pipelines, no GPU work — to prove the central claim
// of the engine-registry refactor:
//
//   A new model kind = a new handle type + a factory arm + routing arms.
//   The manager CORE (load/evict/VRAM/LRU/metrics) needs ZERO edits,
//   because it only ever reads engines through `ManagedEngine`.
//
// Each stub is a unit struct that implements `ManagedEngine` with a
// fixed kind and zero capacity (there is nothing to run). The
// `MockEngineFactory` builds them so a dummy kind can flow all the way
// through register → load → list → metrics → evict in a unit test with
// no GPU. The real pipelines (and a non-stub handle carrying a command
// channel + in-flight counter, like `cb_engine`/`vlm_engine`) land in R4
// when each flow is actually implemented; until then every HTTP route
// for these kinds returns 501 Not Implemented.
// ============================================================

// NOTE: `Embedding` (R4/G3), `Stt` (Phase 5.1c), `ImageGen` (Phase 5.3c), and
// `Tts` (Phase 5.2a) were all promoted out of this stub module — they now have
// real pipeline-backed handles in their respective pipelines modules. This file
// is kept as the home for any future R3-style seams.

// ============================================================
// Unit tests
// ============================================================
#[cfg(test)]
mod tests {
    // No stubs remain — tests migrate with the promoted handles.
}
