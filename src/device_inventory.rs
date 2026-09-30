// ============================================================
// src/device_inventory.rs — Phase-A device inventory + property probing
// ============================================================
//
// Builds, at startup, a snapshot of every OpenVINO compute device on the box:
// its name ("GPU.1"/"CPU"/"NPU"), kind (discrete/integrated GPU, CPU, NPU),
// capability *tier*, memory *domain*, and total memory.
//
// WHY tiers, not names (see the project's internal engineering log):
//   Device roles invert across boxes. On one dual-GPU box, GPU.0 (an
//   integrated Intel UHD) is the weak offload target and GPU.1 (the Arc dGPU)
//   is the workhorse; on a Lunar Lake laptop
//   the iGPU *is* the workhorse and the NPU is the offload target. "iGPU" is a
//   name, not a capability — so placement (Phase C) reasons over tiers, and this
//   module maps each discovered physical device → a tier.
//
// WHY domains, not per-device budgets (the UMA crux):
//   A discrete GPU has a private VRAM pool. On Lunar Lake the CPU + iGPU + NPU
//   all allocate from one shared LPDDR5X system-RAM pool. If the memory tracker
//   kept a per-*device* budget on UMA, each device would believe it owns the
//   whole system RAM → free memory multiply-counted → OOM. So every device is
//   tagged with a `domain_id`: a discrete GPU gets its own private domain; every
//   integrated GPU / CPU / NPU joins the shared `DOMAIN_SYSTEM` domain. The
//   Phase-B MemoryTracker keys budgets by domain. (Phase A only records it.)
//
// Tier resolution is three layers, first hit wins (settled):
//   1. Operator override (per-device config) — wired in Phase C; empty here.
//   2. Static heuristic table — keyed on the device fingerprint. The entire
//      current fleet hits here. Zero cost, deterministic.
//   3. Property-only ranking — fires only for a device in neither 1 nor 2. Its
//      derived tier is persisted to a per-box cache (device_tiers.json) and a
//      warning logged, so next boot it resolves instantly (≈ promoted to L2).
// ============================================================

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

use crate::ov_cb;

/// Domain id shared by every integrated GPU, CPU, and NPU — they all draw from
/// system RAM. A discrete GPU instead uses its own device name as its domain id
/// (a private VRAM pool). The Phase-B `MemoryTracker` budgets per domain id.
pub const DOMAIN_SYSTEM: &str = "system";

/// Physical class of a compute device, from `DEVICE_TYPE` + the name prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    /// The host CPU (`"CPU"`).
    Cpu,
    /// An integrated GPU sharing system RAM (`DEVICE_TYPE = INTEGRATED`).
    IntegratedGpu,
    /// A discrete GPU with its own VRAM (`DEVICE_TYPE = DISCRETE`).
    DiscreteGpu,
    /// A neural processing unit (`"NPU"`).
    Npu,
    /// A device whose class could not be determined from its properties.
    Other,
}

impl fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DeviceKind::Cpu => "CPU",
            DeviceKind::IntegratedGpu => "integrated-gpu",
            DeviceKind::DiscreteGpu => "discrete-gpu",
            DeviceKind::Npu => "NPU",
            DeviceKind::Other => "other",
        };
        f.write_str(s)
    }
}

/// Capability tier — the coarse placement bucket a device falls into. Placement
/// (Phase C) matches a model's demand class to a tier preference, never a name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DeviceTier {
    /// Discrete GPU — the workhorse for heavy interactive decode (LLM/VLM/SDXL).
    Heavy,
    /// High-EU / XMX integrated GPU (e.g. Lunar Lake Xe2) — strong, shared RAM.
    StrongIgpu,
    /// Low-EU integrated GPU (e.g. UHD 730/770, Gen12) — the worthy *offload*
    /// target: slow, but frees the strong device for the money workload.
    WeakIgpu,
    /// Neural processing unit — low-power, static-shape; good for STT/embeddings.
    Npu,
    /// CPU — the universal fallback.
    Fallback,
}

impl DeviceTier {
    /// Parse a kebab-case tier label — the inverse of [`Display`](fmt::Display) —
    /// back into a tier. Used to interpret operator config (`tier_preference`,
    /// per-device tier overrides), mirroring [`ModelKind::from_label`]. Returns
    /// `None` for an unrecognised value so the caller can reject it with context.
    ///
    /// Note this is distinct from the derived `serde` representation (`PascalCase`
    /// variant names), which the on-disk learned-tier cache uses — config speaks
    /// the human kebab labels, the cache speaks serde.
    ///
    /// [`ModelKind::from_label`]: crate::model_manager::engine::ModelKind::from_label
    #[must_use]
    pub fn from_label(s: &str) -> Option<Self> {
        match s {
            "heavy" => Some(DeviceTier::Heavy),
            "strong-igpu" => Some(DeviceTier::StrongIgpu),
            "weak-igpu" => Some(DeviceTier::WeakIgpu),
            "npu" => Some(DeviceTier::Npu),
            "fallback" => Some(DeviceTier::Fallback),
            _ => None,
        }
    }
}

impl fmt::Display for DeviceTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DeviceTier::Heavy => "heavy",
            DeviceTier::StrongIgpu => "strong-igpu",
            DeviceTier::WeakIgpu => "weak-igpu",
            DeviceTier::Npu => "npu",
            DeviceTier::Fallback => "fallback",
        };
        f.write_str(s)
    }
}

/// Everything Phase A knows about one compute device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// `OpenVINO` device name — `"GPU.1"`, `"CPU"`, `"NPU"`. The placement target.
    pub name: String,
    /// `FULL_DEVICE_NAME` — the human-readable silicon name, e.g.
    /// `"Intel(R) Arc(TM) Pro B50 Graphics"`. Also the fingerprint cache key.
    pub full_name: String,
    /// `DEVICE_ARCHITECTURE`, when the device exposes it (`None` otherwise).
    pub architecture: Option<String>,
    /// Physical class (discrete/integrated GPU, CPU, NPU).
    pub kind: DeviceKind,
    /// Capability tier (resolved via the three-layer scheme).
    pub tier: DeviceTier,
    /// Memory-domain id. A discrete GPU's own name (private pool) or
    /// [`DOMAIN_SYSTEM`] (shared system RAM for integrated GPU / CPU / NPU).
    pub domain_id: String,
    /// Total device memory in GiB, when reported (`GPU_DEVICE_TOTAL_MEM_SIZE`,
    /// GPUs only). `None` for devices that do not expose a memory figure.
    pub total_mem_gb: Option<f64>,
}

/// The startup snapshot of all compute devices.
#[derive(Debug, Clone, Default)]
pub struct DeviceInventory {
    devices: Vec<DeviceInfo>,
}

impl DeviceInventory {
    /// Probes every `OpenVINO` device and resolves its kind, tier, and domain.
    ///
    /// Infallible by design: a device that fails a property query degrades
    /// gracefully (unknown architecture → conservative tier) rather than
    /// aborting startup. An empty result means `OpenVINO` enumerated no devices.
    #[must_use]
    pub fn probe() -> Self {
        Self::probe_with_overrides(&HashMap::new())
    }

    /// As [`probe`](Self::probe), but with operator tier overrides (Phase C):
    /// `device name → tier`. An override is layer 1 of resolution — it always
    /// wins. Phase A callers pass an empty map.
    #[must_use]
    pub fn probe_with_overrides(overrides: &HashMap<String, DeviceTier>) -> Self {
        let names = ov_cb::list_devices();
        let mut cache = TierCache::load();
        let mut cache_dirty = false;

        let devices = names
            .into_iter()
            .map(|name| {
                let full_name = ov_cb::device_property(&name, "FULL_DEVICE_NAME")
                    .unwrap_or_else(|| name.clone());
                let architecture = ov_cb::device_property(&name, "DEVICE_ARCHITECTURE");
                let kind = classify_kind(
                    &name,
                    ov_cb::device_property(&name, "DEVICE_TYPE").as_deref(),
                );
                let total_mem_gb = query_total_mem_gb(&name, kind);

                let (tier, learned) = resolve_tier(&name, &full_name, kind, overrides, &cache);
                if let Some(t) = learned {
                    cache.insert(full_name.clone(), t);
                    cache_dirty = true;
                }

                let domain_id = if kind == DeviceKind::DiscreteGpu {
                    name.clone()
                } else {
                    DOMAIN_SYSTEM.to_owned()
                };

                DeviceInfo {
                    name,
                    full_name,
                    architecture,
                    kind,
                    tier,
                    domain_id,
                    total_mem_gb,
                }
            })
            .collect();

        if cache_dirty {
            cache.save();
        }

        Self { devices }
    }

    /// All discovered devices.
    #[must_use]
    pub fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    /// The device with this `OpenVINO` name, if present.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&DeviceInfo> {
        self.devices.iter().find(|d| d.name == name)
    }

    /// Phase D — capability-based startup gate. `true` when `OpenVINO`
    /// enumerated at least one inference device.
    ///
    /// This is the *only* device check boot performs up front: it never names a
    /// specific device (no hardcoded `GPU.1`), so a box without a discrete GPU
    /// — e.g. Lunar Lake (CPU + iGPU + NPU) — still passes. Per-model
    /// serveability is enforced later, at model registration, where each
    /// preloaded model's resolved device is validated against this inventory and
    /// an unplaceable model is rejected with a clear error. `false` means the
    /// GPU driver or `OpenVINO` runtime is not accessible — boot must fail fast,
    /// before any model-load attempt.
    #[must_use]
    pub fn has_serveable_device(&self) -> bool {
        !self.devices.is_empty()
    }

    /// Test-only constructor from a fixed device list — lets unit tests in other
    /// modules (memory-domain budgets, placement) build a synthetic topology
    /// without a real `OpenVINO` probe.
    #[cfg(test)]
    pub(crate) fn from_devices(devices: Vec<DeviceInfo>) -> Self {
        Self { devices }
    }

    /// The distinct memory-domain ids across all devices.
    #[must_use]
    pub fn domains(&self) -> Vec<String> {
        let mut seen: Vec<String> = Vec::new();
        for d in &self.devices {
            if !seen.contains(&d.domain_id) {
                seen.push(d.domain_id.clone());
            }
        }
        seen
    }

    /// Emits the inventory to the `tracing` log at INFO — one structured line
    /// per device. Called at startup so every boot records the device topology.
    pub fn log(&self) {
        if self.devices.is_empty() {
            tracing::warn!("device inventory is empty — OpenVINO enumerated no compute devices");
            return;
        }
        for d in &self.devices {
            tracing::info!(
                device = %d.name,
                full_name = %d.full_name,
                kind = %d.kind,
                tier = %d.tier,
                domain = %d.domain_id,
                mem_gb = d.total_mem_gb.unwrap_or(-1.0),
                "device inventory",
            );
        }
    }

    /// Human-readable report for the `--device-info` CLI flag. Prints the raw
    /// probe view (including memory figures) so the integrated-GPU memory KYE
    /// can be read off a real box.
    pub fn print_report(&self) {
        if self.devices.is_empty() {
            println!("No OpenVINO devices enumerated.");
            return;
        }
        println!(
            "OpenVINO device inventory ({} device(s)):\n",
            self.devices.len()
        );
        for d in &self.devices {
            let mem = d
                .total_mem_gb
                .map_or_else(|| "n/a".to_owned(), |g| format!("{g:.2} GiB"));
            println!("  {} — {}", d.name, d.full_name);
            println!("      kind        : {}", d.kind);
            println!("      tier        : {}", d.tier);
            println!("      domain      : {}", d.domain_id);
            println!("      total memory: {mem}");
            println!(
                "      architecture: {}",
                d.architecture.as_deref().unwrap_or("n/a")
            );
        }
        println!("\nMemory domains: {}", self.domains().join(", "));
    }
}

/// Classifies a device into a [`DeviceKind`] from its name and `DEVICE_TYPE`.
///
/// The name prefix decides CPU vs NPU vs GPU; `DEVICE_TYPE` (`"DISCRETE"` /
/// `"INTEGRATED"`) then splits the GPU case. A GPU that does not report a type
/// degrades to [`DeviceKind::Other`] rather than guessing.
fn classify_kind(name: &str, device_type: Option<&str>) -> DeviceKind {
    if name == "CPU" {
        return DeviceKind::Cpu;
    }
    if name.starts_with("NPU") {
        return DeviceKind::Npu;
    }
    if name.starts_with("GPU") {
        return match device_type {
            Some("DISCRETE") => DeviceKind::DiscreteGpu,
            Some("INTEGRATED") => DeviceKind::IntegratedGpu,
            _ => DeviceKind::Other,
        };
    }
    DeviceKind::Other
}

/// Reads `GPU_DEVICE_TOTAL_MEM_SIZE` (bytes) for GPU devices and converts to
/// GiB. Non-GPU devices (CPU/NPU) do not expose this; returns `None`.
fn query_total_mem_gb(name: &str, kind: DeviceKind) -> Option<f64> {
    if !matches!(kind, DeviceKind::DiscreteGpu | DeviceKind::IntegratedGpu) {
        return None;
    }
    let bytes: f64 = ov_cb::device_property(name, "GPU_DEVICE_TOTAL_MEM_SIZE")?
        .parse()
        .ok()?;
    Some(bytes / (1024.0 * 1024.0 * 1024.0))
}

/// Resolves a device's tier via the three-layer scheme. Returns the tier and,
/// when it came from the layer-3 property algorithm, `Some(tier)` to be cached
/// (so the next boot resolves it from the learned cache instead).
fn resolve_tier(
    name: &str,
    full_name: &str,
    kind: DeviceKind,
    overrides: &HashMap<String, DeviceTier>,
    cache: &TierCache,
) -> (DeviceTier, Option<DeviceTier>) {
    // Layer 1 — operator override (per-device config). Always wins.
    if let Some(&t) = overrides.get(name) {
        return (t, None);
    }
    // Layer 2 — static heuristic table. The whole known fleet hits here.
    if let Some(t) = static_tier(full_name, kind) {
        return (t, None);
    }
    // Layer 2.5 — learned cache (a previously-derived layer-3 result).
    if let Some(&t) = cache.get(full_name) {
        return (t, None);
    }
    // Layer 3 — property-only ranking. Persist + warn so it graduates.
    let t = property_tier(kind);
    tracing::warn!(
        device = %name,
        full_name = %full_name,
        tier = %t,
        "device not in static tier table — derived tier from properties and cached it; \
         consider adding it to the built-in table",
    );
    (t, Some(t))
}

/// Layer 2: the static heuristic table, keyed on the device fingerprint
/// (`FULL_DEVICE_NAME`) + kind. Covers the whole known fleet. Returns `None`
/// for a device the table does not recognise (→ falls through to layer 3).
fn static_tier(full_name: &str, kind: DeviceKind) -> Option<DeviceTier> {
    match kind {
        DeviceKind::Cpu => Some(DeviceTier::Fallback),
        DeviceKind::Npu => Some(DeviceTier::Npu),
        // Every discrete GPU in the fleet (Arc B50/B60/B70) is the workhorse.
        DeviceKind::DiscreteGpu => Some(DeviceTier::Heavy),
        DeviceKind::IntegratedGpu => {
            let n = full_name.to_lowercase();
            // Low-EU Gen12 integrated parts — the weak offload target.
            if n.contains("uhd") || n.contains("hd graphics") {
                Some(DeviceTier::WeakIgpu)
            // Lunar Lake / Arc-class Xe2 integrated — the strong worker.
            } else if n.contains("arc")
                || n.contains("xe2")
                || n.contains("lunar")
                || n.contains("iris")
            {
                Some(DeviceTier::StrongIgpu)
            } else {
                None
            }
        }
        DeviceKind::Other => None,
    }
}

/// Layer 3: property-only tier ranking for a device in neither the override map
/// nor the static table. Coarse and conservative — no execution.
fn property_tier(kind: DeviceKind) -> DeviceTier {
    match kind {
        DeviceKind::DiscreteGpu => DeviceTier::Heavy,
        // EU count is not exposed by our probe set, so an unrecognised
        // integrated GPU degrades to weak-igpu (conservative).
        DeviceKind::IntegratedGpu => DeviceTier::WeakIgpu,
        DeviceKind::Npu => DeviceTier::Npu,
        DeviceKind::Cpu | DeviceKind::Other => DeviceTier::Fallback,
    }
}

// ── Learned-tier cache ───────────────────────────────────────────────────────
//
// Per-box, NOT repo-committed: each box has fixed silicon, so a local cache
// avoids the multi-machine-sync drift a committed table would cause. Keyed by
// FULL_DEVICE_NAME → tier. Best-effort: any read/write failure degrades to
// "no cache" and never aborts the probe.

/// On-disk learned-tier cache (`device_tiers.json`).
struct TierCache {
    map: HashMap<String, DeviceTier>,
}

impl TierCache {
    /// Loads the cache, or an empty one if absent/unreadable/corrupt.
    fn load() -> Self {
        let Some(path) = cache_path() else {
            return Self {
                map: HashMap::new(),
            };
        };
        let map = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        Self { map }
    }

    fn get(&self, fingerprint: &str) -> Option<&DeviceTier> {
        self.map.get(fingerprint)
    }

    fn insert(&mut self, fingerprint: String, tier: DeviceTier) {
        self.map.insert(fingerprint, tier);
    }

    /// Best-effort persist. Logs a warning on failure but never errors.
    fn save(&self) {
        let Some(path) = cache_path() else {
            return;
        };
        if let Some(parent) = path.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            tracing::warn!(error = %e, path = %parent.display(), "could not create device-tier cache dir");
            return;
        }
        match serde_json::to_string_pretty(&self.map) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&path, json) {
                    tracing::warn!(error = %e, path = %path.display(), "could not write device-tier cache");
                }
            }
            Err(e) => tracing::warn!(error = %e, "could not serialise device-tier cache"),
        }
    }
}

/// `$XDG_CACHE_HOME/rustedvino/device_tiers.json`, or `~/.cache/...` as the
/// fallback. `None` if neither environment variable is set.
fn cache_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("rustedvino").join("device_tiers.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal device for capability-gate tests (only presence matters).
    #[cfg(test)]
    fn min_dev(name: &str, kind: DeviceKind) -> DeviceInfo {
        DeviceInfo {
            name: name.to_owned(),
            full_name: name.to_owned(),
            architecture: None,
            kind,
            tier: DeviceTier::Fallback,
            domain_id: name.to_owned(),
            total_mem_gb: None,
        }
    }

    #[test]
    fn capability_gate_passes_with_any_device_fails_when_empty() {
        // Phase D: boot fails only when OpenVINO enumerated *nothing* (broken
        // driver/runtime) — never on the absence of a specific device.
        assert!(!DeviceInventory::from_devices(vec![]).has_serveable_device());
        // A no-dGPU box (Lunar Lake shape: CPU + iGPU + NPU) still passes — the
        // gate names no device.
        let uma = DeviceInventory::from_devices(vec![
            min_dev("CPU", DeviceKind::Cpu),
            min_dev("GPU", DeviceKind::IntegratedGpu),
            min_dev("NPU", DeviceKind::Npu),
        ]);
        assert!(uma.has_serveable_device());
        assert!(uma.get("GPU.1").is_none());
    }

    #[test]
    fn classify_kind_by_name_and_type() {
        assert_eq!(classify_kind("CPU", None), DeviceKind::Cpu);
        assert_eq!(classify_kind("NPU", None), DeviceKind::Npu);
        assert_eq!(
            classify_kind("GPU.1", Some("DISCRETE")),
            DeviceKind::DiscreteGpu
        );
        assert_eq!(
            classify_kind("GPU.0", Some("INTEGRATED")),
            DeviceKind::IntegratedGpu
        );
        // A GPU with no reported type degrades to Other, not a guess.
        assert_eq!(classify_kind("GPU.0", None), DeviceKind::Other);
    }

    #[test]
    fn static_table_covers_the_fleet() {
        assert_eq!(
            static_tier("any", DeviceKind::Cpu),
            Some(DeviceTier::Fallback)
        );
        assert_eq!(static_tier("any", DeviceKind::Npu), Some(DeviceTier::Npu));
        assert_eq!(
            static_tier("Intel(R) Arc(TM) Pro B50 Graphics", DeviceKind::DiscreteGpu),
            Some(DeviceTier::Heavy)
        );
        assert_eq!(
            static_tier("Intel(R) UHD Graphics 730", DeviceKind::IntegratedGpu),
            Some(DeviceTier::WeakIgpu)
        );
        assert_eq!(
            static_tier("Intel(R) Arc(TM) 140V GPU", DeviceKind::IntegratedGpu),
            Some(DeviceTier::StrongIgpu)
        );
    }

    #[test]
    fn unknown_integrated_gpu_falls_through_to_property_layer() {
        // An integrated GPU the static table doesn't recognise → layer 3.
        assert_eq!(
            static_tier("Some Future iGPU", DeviceKind::IntegratedGpu),
            None
        );
        assert_eq!(
            property_tier(DeviceKind::IntegratedGpu),
            DeviceTier::WeakIgpu
        );
        assert_eq!(property_tier(DeviceKind::DiscreteGpu), DeviceTier::Heavy);
    }

    #[test]
    fn override_wins_over_static_table() {
        let mut overrides = HashMap::new();
        overrides.insert("GPU.1".to_owned(), DeviceTier::WeakIgpu);
        let cache = TierCache {
            map: HashMap::new(),
        };
        let (tier, learned) = resolve_tier(
            "GPU.1",
            "Intel(R) Arc(TM) Pro B50 Graphics",
            DeviceKind::DiscreteGpu,
            &overrides,
            &cache,
        );
        assert_eq!(tier, DeviceTier::WeakIgpu);
        assert_eq!(learned, None);
    }

    #[test]
    fn property_layer_result_is_marked_for_caching() {
        let cache = TierCache {
            map: HashMap::new(),
        };
        let (tier, learned) = resolve_tier(
            "GPU.2",
            "Some Future iGPU",
            DeviceKind::IntegratedGpu,
            &HashMap::new(),
            &cache,
        );
        assert_eq!(tier, DeviceTier::WeakIgpu);
        assert_eq!(learned, Some(DeviceTier::WeakIgpu));
    }

    #[test]
    fn cached_tier_resolves_without_relearning() {
        let mut map = HashMap::new();
        map.insert("Some Future iGPU".to_owned(), DeviceTier::StrongIgpu);
        let cache = TierCache { map };
        let (tier, learned) = resolve_tier(
            "GPU.2",
            "Some Future iGPU",
            DeviceKind::IntegratedGpu,
            &HashMap::new(),
            &cache,
        );
        assert_eq!(tier, DeviceTier::StrongIgpu);
        assert_eq!(learned, None, "a cache hit must not re-mark for caching");
    }
}
