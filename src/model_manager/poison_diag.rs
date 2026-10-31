// ============================================================
// src/model_manager/poison_diag.rs — memory state at the moment a GPU context is poisoned (G2)
// ============================================================
// A poisoned context (`CL_OUT_OF_RESOURCES`, …) used to leave one error line and nothing else, so a
// cause (host memory pressure on an iGPU after idle, device memory on a discrete card) could not be
// told apart afterwards. `ModelManager::mark_gpu_poisoned_inner` logs one line built here on the
// FIRST poisoning. Collection lives in `ModelManager::poison_diagnostics`; this file is the pure
// formatter so the line's shape is unit-tested.
// ============================================================

use crate::os_memory::MemoryPressure;

/// Everything the poison-time line reports. `None` = could not be read on this host.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PoisonDiag {
    pub model_id: String,
    pub device: String,
    pub domain: String,
    /// Seconds since the failing model's last request started (or its load): `0`-ish when it
    /// failed mid-request, large when it failed on the first request after a long idle (H2).
    pub idle_secs: Option<u64>,
    pub mem_available_gb: Option<f64>,
    pub swap_used_gb: Option<f64>,
    pub zram_used_gb: f64,
    pub pressure: Option<MemoryPressure>,
    /// This process's DRM memory: (driver, pci device, region, bytes).
    pub drm: Vec<(String, String, String, u64)>,
    /// The tracker's booking per model, GiB (only models with a booking).
    pub booked: Vec<(String, f64)>,
}

#[allow(clippy::cast_precision_loss)] // bytes → GiB for a log line
fn gib(bytes: u64) -> f64 {
    bytes as f64 / f64::from(1u32 << 30)
}

fn opt_gb(v: Option<f64>) -> String {
    v.map_or_else(|| "?".to_owned(), |g| format!("{g:.2} GiB"))
}

/// The single diagnostics line (stable `key=value` order; lists in brackets).
pub(crate) fn format_poison_diag(d: &PoisonDiag) -> String {
    let psi = d.pressure.map_or_else(
        || "psi=?".to_owned(),
        |p| {
            format!(
                "psi_some10={:.2} psi_some60={:.2} psi_full10={:.2} psi_full60={:.2}",
                p.some_avg10, p.some_avg60, p.full_avg10, p.full_avg60
            )
        },
    );
    let drm = d
        .drm
        .iter()
        .filter(|(_, _, _, b)| *b > 0)
        .map(|(drv, pdev, region, b)| format!("{drv} {pdev} {region} {:.2} GiB", gib(*b)))
        .collect::<Vec<_>>()
        .join("; ");
    let booked = d
        .booked
        .iter()
        .map(|(m, g)| format!("{m} {g:.2} GiB"))
        .collect::<Vec<_>>()
        .join("; ");
    format!(
        "model={} device={} domain={} since_last_request={} mem_available={} swap_used={} zram_used={:.2} GiB {psi} \
         drm=[{drm}] booked=[{booked}]",
        d.model_id,
        d.device,
        d.domain,
        d.idle_secs
            .map_or_else(|| "?".to_owned(), |s| format!("{s}s")),
        opt_gb(d.mem_available_gb),
        opt_gb(d.swap_used_gb),
        d.zram_used_gb,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_has_every_field_and_skips_empty_drm_regions() {
        let d = PoisonDiag {
            model_id: "qwen3.5-4b-int4-ov".to_owned(),
            device: "GPU.1".to_owned(),
            domain: "gpu:0000:03:00.0".to_owned(),
            idle_secs: Some(12),
            mem_available_gb: Some(40.5),
            swap_used_gb: Some(0.0),
            zram_used_gb: 0.25,
            pressure: Some(MemoryPressure {
                some_avg10: 1.5,
                some_avg60: 0.75,
                full_avg10: 0.25,
                full_avg60: 0.05,
            }),
            drm: vec![
                ("xe".into(), "0000:03:00.0".into(), "vram0".into(), 3 << 30),
                ("xe".into(), "0000:03:00.0".into(), "gtt".into(), 0),
            ],
            booked: vec![
                ("qwen3-8b-int4-ov".into(), 6.9),
                ("qwen3.5-4b-int4-ov".into(), 4.3),
            ],
        };
        assert_eq!(
            format_poison_diag(&d),
            "model=qwen3.5-4b-int4-ov device=GPU.1 domain=gpu:0000:03:00.0 since_last_request=12s \
             mem_available=40.50 GiB swap_used=0.00 GiB zram_used=0.25 GiB psi_some10=1.50 \
             psi_some60=0.75 psi_full10=0.25 psi_full60=0.05 drm=[xe 0000:03:00.0 vram0 3.00 GiB] \
             booked=[qwen3-8b-int4-ov 6.90 GiB; qwen3.5-4b-int4-ov 4.30 GiB]"
        );
    }

    #[test]
    fn unreadable_values_print_as_question_marks() {
        let d = PoisonDiag {
            model_id: "m".to_owned(),
            device: "GPU".to_owned(),
            domain: "system".to_owned(),
            idle_secs: None,
            mem_available_gb: None,
            swap_used_gb: None,
            zram_used_gb: 0.0,
            pressure: None,
            drm: Vec::new(),
            booked: Vec::new(),
        };
        let line = format_poison_diag(&d);
        assert!(
            line.contains(
                "since_last_request=? mem_available=? swap_used=? zram_used=0.00 GiB psi=? drm=[] booked=[]"
            ),
            "{line}"
        );
    }
}
