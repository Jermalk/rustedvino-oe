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
    use DeviceTier::{Fallback, Heavy, Npu, StrongIgpu, WeakIgpu};
    match kind {
        // Light + frequent + latency-tolerant: offload first — but **NOT to the
        // NPU**. The embedding pipeline (`ov::genai::TextEmbeddingPipeline`) does
        // not static-shape for the NPU, and the BERT encoder's attention nodes
        // trip the NPU plugin's `check_sdpa_nodes` gate → the load *throws*
        // (proven on a Lunar Lake NPU box: NPU ❌ `check_sdpa_nodes failed`,
        // GPU/CPU ✅ 768-dim).
        // `place()` returns the first *present* tier and does not fall through on
        // a load-time failure, so leaving `Npu` here would make embeddings
        // unserveable on any NPU box. Whisper (STT, below) keeps `Npu` because
        // `WhisperPipeline` *does* static-shape and compiles clean on the NPU.
        ModelKind::Embedding | ModelKind::Reranking => vec![WeakIgpu, StrongIgpu, Heavy, Fallback],
        // STT is size-aware like LLMs, but with a *separate* threshold
        // (`light_stt_max_gb`, smaller because Whisper models are smaller).
        // A *thin* STT model (whisper-base/small) is bandwidth-bound → offload
        // first (NPU-first here is fine — Whisper compiles on the NPU, unlike the
        // embedding encoder above). A *heavy* one (whisper-large, encoder-bound)
        // is pessimal on a **weak** iGPU (measured: 6.87 s vs CPU 4.98 s vs dGPU
        // 0.39 s on 8 s audio), so it demotes a weak iGPU *below* the CPU fallback
        // while keeping a **strong** iGPU offload-first. One static order encodes
        // both: `StrongIgpu` above `Fallback` (strong iGPU stays the offload
        // target) and `WeakIgpu` last (CPU wins on a weak-iGPU box). `Heavy`
        // (dGPU) sits after the CPU so heavy STT still frees the dGPU for the LLM.
        ModelKind::Stt => {
            if size_gb <= light_max_gb {
                vec![Npu, WeakIgpu, StrongIgpu, Heavy, Fallback]
            } else {
                vec![Npu, StrongIgpu, Fallback, Heavy, WeakIgpu]
            }
        }
        // Heavy interactive decode — but a *thin* LLM/VLM is light work and gets
        // the offload-first order so it parks off the dGPU.
        ModelKind::TextGen | ModelKind::Vision => {
            if size_gb <= light_max_gb {
                vec![WeakIgpu, StrongIgpu, Heavy, Fallback]
            } else {
                vec![Heavy, StrongIgpu, Fallback]
            }
        }
        // Bursty heavy: strongest device (load/evict handled elsewhere).
        ModelKind::ImageGen => vec![Heavy, StrongIgpu, Fallback],
        // SpeechT5 is a real OV GenAI device model (small, ~0.25 GB) — offload-first
        // like STT. No NPU (Text2SpeechPipeline unverified on NPU); no Heavy dGPU
        // (too small to benefit; keep the dGPU free for LLMs).
        ModelKind::Tts => vec![StrongIgpu, WeakIgpu, Fallback],
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
    fn thin_llm_offloads_to_weak_igpu() {
        // 0.6B int4 (~0.34 GB weights, vram_gb estimate well under 2 GB) is light
        // work → offload-first order picks the iGPU off the dGPU.
        let pref = default_tier_preference(ModelKind::TextGen, 1.0, 2.0);
        assert_eq!(place("qwen3-0.6b", &pref, &arc_box()).unwrap(), "GPU.0");
    }

    #[test]
    fn size_threshold_boundary_is_inclusive_light() {
        // Exactly at the threshold counts as light (<=).
        let pref = default_tier_preference(ModelKind::TextGen, 2.0, 2.0);
        assert_eq!(pref.first(), Some(&DeviceTier::WeakIgpu));
    }

    #[test]
    fn embedding_prefers_weak_igpu_over_dgpu() {
        let pref = default_tier_preference(ModelKind::Embedding, 0.5, 2.0);
        assert_eq!(place("e5", &pref, &arc_box()).unwrap(), "GPU.0");
    }

    #[test]
    fn embedding_falls_to_cpu_when_no_igpu() {
        let inv = DeviceInventory::from_devices(vec![
            dev("GPU.1", DeviceKind::DiscreteGpu, DeviceTier::Heavy),
            dev("CPU", DeviceKind::Cpu, DeviceTier::Fallback),
        ]);
        // No weak/strong iGPU, no NPU → next satisfiable tier is heavy (dGPU).
        let pref = default_tier_preference(ModelKind::Embedding, 0.5, 2.0);
        assert_eq!(place("e5", &pref, &inv).unwrap(), "GPU.1");
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

    /// The asymmetric counterpart: STT *keeps* the NPU (Whisper compiles there).
    /// On the same NPU + CPU box a light STT model lands on the NPU.
    #[test]
    fn light_stt_keeps_npu_when_present() {
        let npu_cpu_box = DeviceInventory::from_devices(vec![
            dev("NPU", DeviceKind::Npu, DeviceTier::Npu),
            dev("CPU", DeviceKind::Cpu, DeviceTier::Fallback),
        ]);
        let pref = default_tier_preference(ModelKind::Stt, 0.25, 1.0);
        assert_eq!(place("whisper-small", &pref, &npu_cpu_box).unwrap(), "NPU");
    }

    /// Light STT (whisper-base/small, ≤ `light_stt_max_gb`) keeps the
    /// embedding-style offload-first order → lands on the weak iGPU.
    #[test]
    fn light_stt_offloads_to_weak_igpu() {
        let pref = default_tier_preference(ModelKind::Stt, 0.25, 1.0);
        assert_eq!(pref.first(), Some(&DeviceTier::Npu));
        assert_eq!(place("whisper-small", &pref, &arc_box()).unwrap(), "GPU.0");
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

    /// Heavy STT on a *strong*-iGPU box keeps offload-first — the strong iGPU
    /// still outranks the CPU. (Same heavy preference list; first-present wins.)
    #[test]
    fn heavy_stt_keeps_strong_igpu_offload_first() {
        let strong_box = DeviceInventory::from_devices(vec![
            dev("GPU.1", DeviceKind::DiscreteGpu, DeviceTier::Heavy),
            dev("GPU.0", DeviceKind::IntegratedGpu, DeviceTier::StrongIgpu),
            dev("CPU", DeviceKind::Cpu, DeviceTier::Fallback),
        ]);
        let pref = default_tier_preference(ModelKind::Stt, 1.5, 1.0);
        assert_eq!(place("whisper-large", &pref, &strong_box).unwrap(), "GPU.0");
    }

    #[test]
    fn tts_offloads_to_igpu_first() {
        // SpeechT5 is a real OV GenAI device model — offload-first like STT.
        // On an Arc dGPU box (arc_box: GPU.1=Heavy, GPU.0=WeakIgpu, CPU=Fallback)
        // a small TTS model should land on GPU.0 (WeakIgpu), not the dGPU or CPU.
        let pref = default_tier_preference(ModelKind::Tts, 0.25, 2.0);
        assert_eq!(
            pref,
            vec![
                DeviceTier::StrongIgpu,
                DeviceTier::WeakIgpu,
                DeviceTier::Fallback
            ]
        );
        assert_eq!(place("speecht5-tts", &pref, &arc_box()).unwrap(), "GPU.0");
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
