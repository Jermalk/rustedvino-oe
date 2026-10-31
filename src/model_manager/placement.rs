//! Phase C2 — the auto tier-preference placement engine.
//!
//! Given a model's [`ModelKind`] and size, [`place`] walks a **tier preference**
//! (an ordered list of [`DeviceTier`]) and returns the `OpenVINO` name of the
//! first device whose tier appears earliest in the preference *and* is present in
//! the startup [`DeviceInventory`]. Placement matches a model's demand class to a
//! capability *tier*, never a device *name* — because roles invert across boxes
//! (on one dual-GPU box, `GPU.0` is the weak offload target; on a Lunar Lake
//! laptop the iGPU is the workhorse).
//!
//! Fit-in-domain is **deferred** to the `MemoryTracker` at admission time (Slice
//! 2/3): the engine answers "which device *kind* should this model prefer", not
//! "does it fit right now". Compatibility constraints beyond availability (e.g.
//! NPU static-shape) are a future refinement — the current fleet has no NPU.
//!
//! Precedence (resolved by the caller in `new_with_inventory`):
//! 1. per-model explicit `device` (C1) — always wins, the engine is skipped;
//! 2. per-model `tier_preference: [..]` override → [`place`] with that order;
//! 3. embedding only: explicit `config.embedding_device` if set;
//! 4. built-in [`default_tier_preference`] for the kind → [`place`].

use crate::device_inventory::{DeviceInventory, DeviceTier};
use crate::model_manager::engine::ModelKind;

/// Walk `preference` in order and return the name of the first inventory device
/// whose [`DeviceTier`] matches the earliest-listed tier with any available
/// device. Returns `Err` when **no** device satisfies any tier in the preference
/// (validate-and-reject — the operator named a topology this box can't serve).
///
/// Iteration is preference-major: tier `preference[0]` is tried across all
/// devices before `preference[1]`, so the *most preferred satisfiable* tier wins.
/// Within a tier, the first device the inventory enumerated wins (deterministic;
/// the inventory preserves OV's enumeration order).
pub(crate) fn place(
    model_id: &str,
    preference: &[DeviceTier],
    inventory: &DeviceInventory,
) -> anyhow::Result<String> {
    for &tier in preference {
        if let Some(d) = inventory.devices().iter().find(|d| d.tier == tier) {
            return Ok(d.name.clone());
        }
    }
    let available: Vec<String> = inventory
        .devices()
        .iter()
        .map(|d| format!("{} ({})", d.name, d.tier))
        .collect();
    anyhow::bail!(
        "no available OpenVINO device satisfies the tier preference {preference:?} for model \
         '{model_id}' (devices present: {available:?})"
    )
}

/// Like [`place`], but **fit-aware** (T5/F4, D-F4 option A): among the devices
/// the preference admits, return the most-preferred whose memory *domain* still
/// has room for `needed_gb` per the `fits` oracle. When no admitted device fits
/// as-is, fall back to [`place`]'s most-preferred satisfiable device unchanged —
/// so load-time eviction still gets its chance and a placement that previously
/// succeeded never becomes a hard rejection.
///
/// `fits(domain_id, gb)` reports whether `domain_id` can accept another `gb`
/// gigabytes; a gating-disabled or unknown domain reports `true` (unbounded).
/// This is the cross-domain feedback C2 lacked: a domain fully reserved by
/// *immovable* (pinned / non-evictable) models is skipped, so the model routes
/// to a tier that can actually hold it instead of resolving permanently onto a
/// full domain and failing at load. Eviction handles *evictable* contention
/// within a domain at load time, so callers reserve only immovable footprints.
///
/// With an always-`true` `fits` (nothing reserved, or every domain gating-
/// disabled) this is identical to [`place`] — the first device of the
/// most-preferred satisfiable tier.
pub(crate) fn place_with_fit(
    model_id: &str,
    preference: &[DeviceTier],
    needed_gb: f64,
    inventory: &DeviceInventory,
    fits: impl Fn(&str, f64) -> bool,
) -> anyhow::Result<String> {
    for &tier in preference {
        if let Some(d) = inventory
            .devices()
            .iter()
            .find(|d| d.tier == tier && fits(&d.domain_id, needed_gb))
        {
            return Ok(d.name.clone());
        }
    }
    // Nothing fits as-is — preserve the C2 placement (most-preferred satisfiable
    // tier) and let the MemoryTracker + eviction reclaim space at load time.
    place(model_id, preference, inventory)
}

/// The built-in per-kind tier preference (override-able via `tier_preference`).
///
/// Light, frequent, latency-tolerant work (embeddings, STT) prefers the
/// *weakest adequate* tier first — offloading frees the strong device for the
/// money workload. Heavy interactive decode (LLM/VLM, image) prefers the
/// *strongest* tier first. Both orderings serve one goal: keep the strong device
/// on the workload that needs it.
///
/// **Size-aware LLM/VLM tiering:** a *thin* LLM (int4 weights ≲
/// `light_max_gb`) is light work — measured 0.6B at 33 tok/s on a UHD 770 iGPU,
/// genuinely interactive — so a small LLM gets the *light* (offload-first) order.
/// A heavy LLM stays dGPU-first. `size_gb` is the model's `vram_gb` estimate (the
/// closest available proxy for int4 weight size); `light_max_gb` is config-tuned.
///
/// **Size/EU-aware STT tiering (measured):** STT is also size-aware, but the
/// caller passes the *STT* threshold (`light_stt_max_gb`, smaller than the LLM
/// one). A heavy STT model (whisper-large) is encoder-bound and **pessimal on a
/// weak iGPU** — so it demotes a `weak-igpu` below the CPU `fallback` while a
/// `strong-igpu` stays offload-first. Light STT (whisper-base/small) keeps the
/// embedding-style offload-first order.
pub(crate) fn default_tier_preference(
    kind: ModelKind,
    size_gb: f64,
    light_max_gb: f64,
) -> Vec<DeviceTier> {
    use DeviceTier::{Fallback, Heavy, StrongIgpu, WeakIgpu};
    // **An integrated GPU is a last resort, after the CPU, in every default order below.** The earlier
    // orders put it first for all light work; that came from one machine (a Lunar Lake Arc 140V, an
    // unusually capable iGPU). Measured on an Acer ConceptD's UHD iGPU (i7-11800H): every light model
    // ran 20-45 % SLOWER than on the CPU, SpeechT5 ran at RTF 16-100 (unusable), and the vision model's
    // GPU kernel build never finished. Most machines have an iGPU like that one, so the CPU wins by
    // default; an operator who owns a good iGPU names it (`device`) or lists the tier (`tier_preference`).
    // A later tier is only reached when an earlier one is absent or has no room for the model.
    match kind {
        // Light + frequent + latency-tolerant: a discrete GPU first when there is one (an Arc Pro B50 embedded 727
        // inputs/s against 39 on its CPU and 20 on its UHD iGPU), else the CPU - and **never the NPU**. The embedding pipeline
        // (`ov::genai::TextEmbeddingPipeline`) does not static-shape for the NPU, and the BERT encoder's
        // attention nodes trip the NPU plugin's `check_sdpa_nodes` gate -> the load *throws* (proven on a
        // Lunar Lake NPU box: NPU fails `check_sdpa_nodes`, GPU/CPU give 768-dim). `place()` returns the
        // first *present* tier and does not fall through on a load-time failure, so `Npu` here would
        // make embeddings unserveable on any NPU box.
        // Decision (/v1/systemone): a full LLM-sized prefill per question, so the strongest device first
        // (a dGPU) - and never the NPU: NPUW's LLM mode corrupts the hybrid model's conv/SSM states
        // (measured), and a static prefill-only NPU graph is still unproven.
        ModelKind::Embedding | ModelKind::Reranking | ModelKind::Decision => {
            vec![Heavy, Fallback, StrongIgpu, WeakIgpu]
        }
        // STT: the CPU first, then the dGPU (`Heavy` sits after the CPU so STT leaves the dGPU to the LLM -
        // Whisper next to an LLM on one Arc driver is the pattern that wedges it), an iGPU only as a last
        // resort. **Never the NPU by default** (owner decision, 2026-10-13): whisper-large-v3-turbo hung the
        // NPU on a clip over ~60 s, forever, and nothing in the server bounds a request - a small Whisper
        // on the NPU is something the operator asks for with `device`. A heavy encoder-bound model was
        // measured pessimal on a weak iGPU (6.87 s vs CPU 4.98 s vs dGPU 0.39 s on 8 s audio), and
        // whisper-turbo ran at RTF 1.39 on the Acer's UHD against 0.93 on its CPU.
        ModelKind::Stt => vec![Fallback, Heavy, StrongIgpu, WeakIgpu],
        // Chat LLMs go to a discrete GPU when there is one - even thin ones: on an Arc Pro B50 a 1.7B int4 model
        // decoded at 114 tok/s against 35 on its CPU and 26 on its UHD iGPU. Without a dGPU the CPU wins and an iGPU is
        // the last resort. (A GPU that runs out of room falls through to the CPU via `place_with_fit`.)
        ModelKind::TextGen => vec![Heavy, Fallback, StrongIgpu, WeakIgpu],
        // Vision keeps the size split: a *thin* VLM is light work and parks on the CPU; a heavy one goes dGPU-first.
        ModelKind::Vision => {
            if size_gb <= light_max_gb {
                vec![Fallback, Heavy, StrongIgpu, WeakIgpu]
            } else {
                vec![Heavy, Fallback, StrongIgpu]
            }
        }
        // Bursty heavy: the strongest device. Unlike the other kinds this keeps a capable iGPU BEFORE
        // the CPU: FLUX.1-schnell int4 needs > 19 GB of host RAM on the CPU and never finished a
        // 512x512 image, but took 9.5 GB and 11.6 s on a strong iGPU. Revisit with `forbidden_devices`
        // (0.8.2 plan), which can ban the CPU for exactly that model instead of ordering by kind.
        ModelKind::ImageGen => vec![Heavy, StrongIgpu, Fallback],
        // SpeechT5: CPU only. It misbehaves on every GPU class it was tried on - 3x slower than the CPU on an Arc 140V
        // iGPU, RTF 16-100 on an Acer UHD iGPU, and a ~10 minute stall ending in HTTP 503 on an Arc Pro B50 - so no GPU
        // tier is listed: if the CPU domain has no room the load fails loudly instead of landing on a GPU that stalls.
        ModelKind::Tts => vec![Fallback],
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::device_inventory::{DeviceInfo, DeviceKind};

    /// Build a synthetic device with an explicit tier (placement only reads
    /// `name` + `tier`; domain/mem are irrelevant to the engine).
    fn dev(name: &str, kind: DeviceKind, tier: DeviceTier) -> DeviceInfo {
        DeviceInfo {
            name: name.to_owned(),
            full_name: name.to_owned(),
            architecture: None,
            kind,
            tier,
            domain_id: name.to_owned(),
            total_mem_gb: None,
            pdev: None,
        }
    }

    /// A full Arc dGPU box: dGPU (heavy) + iGPU (weak) + CPU (fallback).
    fn arc_box() -> DeviceInventory {
        DeviceInventory::from_devices(vec![
            dev("GPU.1", DeviceKind::DiscreteGpu, DeviceTier::Heavy),
            dev("GPU.0", DeviceKind::IntegratedGpu, DeviceTier::WeakIgpu),
            dev("CPU", DeviceKind::Cpu, DeviceTier::Fallback),
        ])
    }

    #[test]
    fn heavy_llm_prefers_dgpu() {
        let pref = default_tier_preference(ModelKind::TextGen, 8.0, 2.0);
        assert_eq!(place("qwen3-8b", &pref, &arc_box()).unwrap(), "GPU.1");
    }

    #[test]
    fn thin_llm_uses_the_dgpu_when_there_is_one_else_the_cpu() {
        // A 0.6B int4 model decodes 3x faster on an Arc Pro B50 than on the CPU of the same box, and an
        // iGPU measured slower than that CPU: dGPU first, never the iGPU while a CPU exists.
        let pref = default_tier_preference(ModelKind::TextGen, 1.0, 2.0);
        assert_eq!(place("qwen3-0.6b", &pref, &arc_box()).unwrap(), "GPU.1");
        let igpu_box = DeviceInventory::from_devices(vec![
            dev("GPU.0", DeviceKind::IntegratedGpu, DeviceTier::WeakIgpu),
            dev("CPU", DeviceKind::Cpu, DeviceTier::Fallback),
        ]);
        assert_eq!(place("qwen3-0.6b", &pref, &igpu_box).unwrap(), "CPU");
    }

    /// The new rule in one place: for EVERY kind except image generation, no iGPU tier comes before
    /// the CPU fallback in the default order.
    #[test]
    fn igpu_is_a_last_resort_in_every_default_order_but_image_gen() {
        for kind in [
            ModelKind::TextGen,
            ModelKind::Vision,
            ModelKind::Embedding,
            ModelKind::Reranking,
            ModelKind::Decision,
            ModelKind::Stt,
            ModelKind::Tts,
        ] {
            for size in [0.25_f64, 1.0, 5.0] {
                let pref = default_tier_preference(kind, size, 2.0);
                let cpu = pref.iter().position(|&t| t == DeviceTier::Fallback);
                for igpu in [DeviceTier::StrongIgpu, DeviceTier::WeakIgpu] {
                    if let Some(i) = pref.iter().position(|&t| t == igpu) {
                        assert!(
                            cpu.is_some_and(|c| c < i),
                            "{kind:?} ({size} GB): {igpu:?} must come after the CPU in {pref:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn size_threshold_boundary_is_inclusive_light() {
        // Exactly at the threshold counts as light (<=).
        let pref = default_tier_preference(ModelKind::Vision, 2.0, 2.0);
        assert_eq!(pref.first(), Some(&DeviceTier::Fallback));
        // ...and just above it a vision model is heavy: dGPU first.
        let pref = default_tier_preference(ModelKind::Vision, 2.1, 2.0);
        assert_eq!(pref.first(), Some(&DeviceTier::Heavy));
    }

    #[test]
    fn embedding_prefers_the_dgpu_then_the_cpu_never_the_igpu() {
        let pref = default_tier_preference(ModelKind::Embedding, 0.5, 2.0);
        assert_eq!(place("e5", &pref, &arc_box()).unwrap(), "GPU.1");
    }

    #[test]
    fn embedding_uses_the_cpu_when_there_is_no_dgpu() {
        let inv = DeviceInventory::from_devices(vec![
            dev("GPU.0", DeviceKind::IntegratedGpu, DeviceTier::WeakIgpu),
            dev("CPU", DeviceKind::Cpu, DeviceTier::Fallback),
        ]);
        let pref = default_tier_preference(ModelKind::Embedding, 0.5, 2.0);
        assert_eq!(place("e5", &pref, &inv).unwrap(), "CPU");
    }

    /// A box with an NPU + CPU and no GPU (e.g. a GPU-less Lunar Lake part). An
    /// embedding model must **never** route to the NPU — `TextEmbeddingPipeline`
    /// fails the NPU plugin's `check_sdpa_nodes` gate at load and `place()` would
    /// not fall through. It must land on the CPU fallback instead.
    /// (Regression: the project's internal engineering log.)
    #[test]
    fn embedding_never_routes_to_npu() {
        let pref = default_tier_preference(ModelKind::Embedding, 0.5, 2.0);
        assert!(
            !pref.contains(&DeviceTier::Npu),
            "NPU must not appear in the embedding preference: {pref:?}"
        );
        let npu_cpu_box = DeviceInventory::from_devices(vec![
            dev("NPU", DeviceKind::Npu, DeviceTier::Npu),
            dev("CPU", DeviceKind::Cpu, DeviceTier::Fallback),
        ]);
        assert_eq!(place("e5", &pref, &npu_cpu_box).unwrap(), "CPU");
    }

    /// STT never lands on the NPU by default (owner decision 2026-10-13: whisper-turbo hung the NPU on
    /// a clip over 60 s and nothing bounds a request): even on an NPU + CPU box, the CPU wins.
    #[test]
    fn stt_never_defaults_to_the_npu() {
        let npu_cpu_box = DeviceInventory::from_devices(vec![
            dev("NPU", DeviceKind::Npu, DeviceTier::Npu),
            dev("CPU", DeviceKind::Cpu, DeviceTier::Fallback),
        ]);
        for size in [0.25, 0.8, 1.5] {
            let pref = default_tier_preference(ModelKind::Stt, size, 1.0);
            assert!(!pref.contains(&DeviceTier::Npu), "{pref:?}");
            assert_eq!(place("whisper", &pref, &npu_cpu_box).unwrap(), "CPU");
        }
    }

    /// Light STT (whisper-base/small) lands on the CPU, not on the weak iGPU.
    #[test]
    fn light_stt_lands_on_the_cpu() {
        let pref = default_tier_preference(ModelKind::Stt, 0.25, 1.0);
        assert_eq!(pref.first(), Some(&DeviceTier::Fallback));
        assert_eq!(place("whisper-small", &pref, &arc_box()).unwrap(), "CPU");
    }

    /// Heavy STT (whisper-large, > `light_stt_max_gb`) on a box whose only iGPU
    /// is *weak* demotes the iGPU below the CPU fallback (measured) → CPU wins.
    #[test]
    fn heavy_stt_prefers_cpu_over_weak_igpu() {
        let pref = default_tier_preference(ModelKind::Stt, 1.5, 1.0);
        // CPU (fallback) must outrank the weak iGPU in the order.
        let cpu = pref.iter().position(|&t| t == DeviceTier::Fallback);
        let weak = pref.iter().position(|&t| t == DeviceTier::WeakIgpu);
        assert!(cpu < weak, "weak iGPU must be demoted below CPU: {pref:?}");
        assert_eq!(place("whisper-large", &pref, &arc_box()).unwrap(), "CPU");
    }

    /// Even a *strong* iGPU (an Arc 140V) is no longer chosen by default for heavy STT: the CPU
    /// outranks it. An operator who owns one names it with `device` / `tier_preference`.
    #[test]
    fn heavy_stt_prefers_cpu_even_over_a_strong_igpu() {
        let strong_box = DeviceInventory::from_devices(vec![
            dev("GPU.1", DeviceKind::DiscreteGpu, DeviceTier::Heavy),
            dev("GPU.0", DeviceKind::IntegratedGpu, DeviceTier::StrongIgpu),
            dev("CPU", DeviceKind::Cpu, DeviceTier::Fallback),
        ]);
        let pref = default_tier_preference(ModelKind::Stt, 1.5, 1.0);
        assert_eq!(place("whisper-large", &pref, &strong_box).unwrap(), "CPU");
        // ...but it is still reachable on purpose.
        let on_purpose = [DeviceTier::StrongIgpu];
        assert_eq!(
            place("whisper-large", &on_purpose, &strong_box).unwrap(),
            "GPU.0"
        );
    }

    #[test]
    fn tts_is_cpu_only_by_default() {
        // SpeechT5 stalls or crawls on every GPU class it was tried on (Arc 140V iGPU 3x slower than the CPU,
        // UHD iGPU RTF 16-100, Arc Pro B50 a 10 minute stall then 503), so no GPU tier is listed at all.
        let pref = default_tier_preference(ModelKind::Tts, 0.25, 2.0);
        assert_eq!(pref, vec![DeviceTier::Fallback]);
        assert_eq!(place("speecht5-tts", &pref, &arc_box()).unwrap(), "CPU");
    }

    #[test]
    fn image_prefers_dgpu() {
        let pref = default_tier_preference(ModelKind::ImageGen, 6.0, 2.0);
        assert_eq!(place("sdxl", &pref, &arc_box()).unwrap(), "GPU.1");
    }

    #[test]
    fn preference_order_is_honoured_first_match_wins() {
        // Explicit preference: prefer CPU over the dGPU.
        let pref = vec![DeviceTier::Fallback, DeviceTier::Heavy];
        assert_eq!(place("m", &pref, &arc_box()).unwrap(), "CPU");
    }

    // ── Fit-aware placement (T5/F4) ──────────────────────────────────────────

    /// With an always-`true` `fits` oracle, `place_with_fit` is identical to
    /// `place` — the most-preferred satisfiable tier.
    #[test]
    fn place_with_fit_matches_place_when_everything_fits() {
        let pref = default_tier_preference(ModelKind::TextGen, 8.0, 2.0);
        let got = place_with_fit("qwen3-8b", &pref, 8.0, &arc_box(), |_, _| true).unwrap();
        assert_eq!(got, "GPU.1");
    }

    /// The preferred tier's domain is full → fall through to the next tier whose
    /// domain has room, instead of resolving onto the full domain.
    #[test]
    fn place_with_fit_falls_through_a_full_domain() {
        // Heavy-first preference; the dGPU domain ("GPU.1") cannot fit, the CPU
        // fallback domain can. (arc_box has no strong-igpu tier.)
        let pref = vec![
            DeviceTier::Heavy,
            DeviceTier::StrongIgpu,
            DeviceTier::Fallback,
        ];
        let got =
            place_with_fit("img", &pref, 6.0, &arc_box(), |domain, _| domain != "GPU.1").unwrap();
        assert_eq!(
            got, "CPU",
            "routed around the full dGPU domain to the fallback"
        );
    }

    /// When NOTHING fits, fall back to `place` (most-preferred satisfiable tier)
    /// rather than rejecting — load-time eviction still gets its chance, so a
    /// placement that C2 would have made never becomes a hard failure.
    #[test]
    fn place_with_fit_falls_back_to_place_when_nothing_fits() {
        let pref = default_tier_preference(ModelKind::TextGen, 8.0, 2.0);
        let got = place_with_fit("qwen3-8b", &pref, 8.0, &arc_box(), |_, _| false).unwrap();
        assert_eq!(
            got, "GPU.1",
            "fell back to the most-preferred satisfiable tier"
        );
    }

    #[test]
    fn no_satisfiable_tier_is_rejected() {
        // CPU-only box, but the preference asks only for an NPU.
        let inv =
            DeviceInventory::from_devices(vec![dev("CPU", DeviceKind::Cpu, DeviceTier::Fallback)]);
        let err = place("m", &[DeviceTier::Npu], &inv).unwrap_err();
        assert!(err.to_string().contains("no available OpenVINO device"));
    }

    #[test]
    fn tier_from_label_roundtrips_display() {
        for t in [
            DeviceTier::Heavy,
            DeviceTier::StrongIgpu,
            DeviceTier::WeakIgpu,
            DeviceTier::Npu,
            DeviceTier::Fallback,
        ] {
            assert_eq!(DeviceTier::from_label(&t.to_string()), Some(t));
        }
        assert_eq!(DeviceTier::from_label("nonsense"), None);
    }
}
