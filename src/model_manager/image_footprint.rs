//! What an image-generation model needs in system memory on the CPU, beyond its weights
//! (SF-5 D-06).
//!
//! On the CPU a diffusion pipeline keeps a generation working set (latents, attention,
//! the VAE decode) that depends on the pipeline family and the image size, not on the
//! weight size, and is not returned after the image. Measured 2026-10-28 on a desktop CPU
//! (n=1 per point, GiB of anonymous
//! memory grown over the load):
//!
//! | family | 512x512 | 1024x1024 |
//! |---|---|---|
//! | SD 1.5 / LCM (LCM Dreamshaper v7 int8) | 4.0 | 8.85 |
//! | SDXL (Juggernaut-XL v9 int8) | 5.63 | 10.77 |
//! | SD3 (SD 3.5 medium int8) | 10.89 | 21.29 |
//! | FLUX (FLUX.1-schnell int4) | 7.1 | 13.1 |
//!
//! Each family is fitted as `a + b * megapixels` (2^20 pixels; 512x512 = 0.25) through its two points and charged with a
//! 15 % margin. `n` images in one request ran nearly one after another (LCM: n=2 grew 1.33x
//! of n=1), so a cap on the request's total pixels keeps the per-request working set within
//! what one image of that many pixels is charged.

use std::path::Path;

/// Margin on the fitted working set (measurement spread: cold vs warm compile cache moved the
/// LCM 512 px peak by 2.5 GiB).
const MARGIN: f64 = 1.15;

/// The diffusion pipeline family, from `model_index.json`'s `_class_name`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageFamily {
    /// Stable Diffusion 1.x/2.x and Latent Consistency Models (a `UNet`, 512 px native).
    Sd15,
    /// Stable Diffusion XL (a larger `UNet`, 1024 px native).
    Sdxl,
    /// Stable Diffusion 3 / 3.5 (an `MM-DiT` transformer, three text encoders).
    Sd3,
    /// FLUX.1 (a `DiT` transformer, T5 text encoder).
    Flux,
    /// No readable class, or a class none of the above covers: charged as the largest.
    Unknown,
}

impl ImageFamily {
    /// The family of a diffusers pipeline class name. Img2img / inpaint variants map to their
    /// base family; order matters (`StableDiffusion3…` and `StableDiffusionXL…` before the
    /// plain `StableDiffusion…` prefix).
    #[must_use]
    pub fn from_class_name(class: &str) -> Self {
        if class.starts_with("StableDiffusion3") {
            Self::Sd3
        } else if class.starts_with("Flux") {
            Self::Flux
        } else if class.starts_with("StableDiffusionXL") {
            Self::Sdxl
        } else if class.starts_with("StableDiffusion") || class.starts_with("LatentConsistency") {
            Self::Sd15
        } else {
            Self::Unknown
        }
    }

    /// The family of the model in `model_dir` (`model_index.json`), [`Self::Unknown`] when it
    /// cannot be read.
    #[must_use]
    pub fn detect(model_dir: &Path) -> Self {
        crate::pipelines::image::read_pipeline_class(model_dir)
            .map_or(Self::Unknown, |c| Self::from_class_name(&c))
    }

    /// `(a, b)` of the fitted working set `a + b * megapixels`, GiB.
    fn fit(self) -> (f64, f64) {
        match self {
            Self::Sd15 => (2.38, 6.47),
            Self::Sdxl => (3.92, 6.85),
            // Unknown: the largest family measured.
            Self::Sd3 | Self::Unknown => (7.42, 13.87),
            Self::Flux => (5.10, 8.00),
        }
    }

    /// GiB a generation of `pixels` (all images of one request) keeps on the CPU, with the
    /// margin. Beyond 1024x1024 this extrapolates the measured line.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn cpu_working_set_gb(self, pixels: u64) -> f64 {
        let (a, b) = self.fit();
        (a + b * pixels as f64 / 1_048_576.0) * MARGIN
    }

    /// Short name for logs and messages.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Sd15 => "SD 1.5 / LCM",
            Self::Sdxl => "SDXL",
            Self::Sd3 => "SD3",
            Self::Flux => "FLUX",
            Self::Unknown => "unknown (charged as SD3)",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_names_map_to_families_in_order() {
        use ImageFamily::*;
        let f = ImageFamily::from_class_name;
        assert_eq!(f("StableDiffusion3Pipeline"), Sd3);
        assert_eq!(f("StableDiffusion3Img2ImgPipeline"), Sd3);
        assert_eq!(f("StableDiffusionXLPipeline"), Sdxl);
        assert_eq!(f("StableDiffusionXLInpaintPipeline"), Sdxl);
        assert_eq!(f("StableDiffusionPipeline"), Sd15);
        assert_eq!(f("LatentConsistencyModelPipeline"), Sd15);
        assert_eq!(f("FluxPipeline"), Flux);
        assert_eq!(f("KandinskyPipeline"), Unknown);
        assert_eq!(ImageFamily::detect(Path::new("/nonexistent")), Unknown);
    }

    /// Regression guard: weights (`vram_gb` = disk) + the working set at 1024x1024 covers
    /// what each measured model really held: anonymous peak plus the weights that were not
    /// already in anonymous memory after the load (file-backed mmap).
    #[test]
    fn the_charge_at_1024_covers_every_measured_model() {
        // (family, vram_gb charged, anon after load, anon peak after 1024x1024), GiB.
        let measured: [(ImageFamily, f64, f64, f64); 4] = [
            (ImageFamily::Flux, 9.2, 10.2, 23.3),
            (ImageFamily::Sd15, 2.3, 0.33, 9.18),
            (ImageFamily::Sdxl, 3.3, 2.09, 12.86),
            (ImageFamily::Sd3, 8.3, 0.61, 21.90),
        ];
        for (family, vram, anon_load, anon_peak) in measured {
            let needed = anon_peak + (vram - anon_load).max(0.0);
            let charged = vram + family.cpu_working_set_gb(1024 * 1024);
            assert!(
                charged >= needed,
                "{family:?}: charged {charged:.2} < needed {needed:.2}"
            );
        }
    }

    #[test]
    fn the_working_set_grows_with_pixels_and_unknown_is_the_largest() {
        for family in [ImageFamily::Sd15, ImageFamily::Sdxl, ImageFamily::Flux] {
            assert!(
                family.cpu_working_set_gb(1 << 20)
                    < ImageFamily::Unknown.cpu_working_set_gb(1 << 20)
            );
            assert!(family.cpu_working_set_gb(512 * 512) < family.cpu_working_set_gb(1 << 20));
        }
    }
}
