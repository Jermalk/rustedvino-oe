// ============================================================
// src/model_manager/kv_headroom.rs — device-headroom limit of a prompt
// ============================================================
// A prompt that fits the KV pool can still exhaust the device: its embeddings are written to the
// GPU (f32, `hidden_size · 4` bytes per token, about twice that with the staging copy), and when
// the free device memory after the weights, every pool and whatever other processes hold cannot
// take that write, the driver fails with `CL_OUT_OF_RESOURCES` 3-5 s into the request and the
// OpenCL context of the whole process is poisoned (measured on the 27B vision model: a 6 GiB pool
// failed at 164,960 and 179,976 tokens with 3.04 GiB free, a 4 GiB pool passed both).
//
// The limit is `free / (hidden_size · 4 · factor)` tokens. This module is the arithmetic only;
// the manager supplies the numbers (tracker commitments, the physical probe, leases).
// ============================================================

/// What the headroom limit is computed from, in bytes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HeadroomInputs {
    /// Total device memory the server may use.
    pub total_bytes: u64,
    /// Held by clients other than this process (physical probe).
    pub outside_bytes: u64,
    /// This process's commitments on the device: weights and pools of every model, as booked.
    pub own_booked_bytes: u64,
    /// Transient reservations of requests currently running.
    pub lease_bytes: u64,
    /// Kept free beyond everything visible.
    pub margin_bytes: u64,
    /// The model's hidden size.
    pub hidden_size: usize,
    /// `kv_transient_factor`.
    pub factor: f64,
}

/// Bytes one prompt token transiently costs on the device.
#[allow(clippy::cast_precision_loss)] // hidden sizes are far below 2^52
pub(crate) fn transient_bytes_per_token(hidden_size: usize, factor: f64) -> f64 {
    hidden_size as f64 * 4.0 * factor
}

impl HeadroomInputs {
    /// Device memory left for a prompt's transient write.
    pub(crate) fn free_bytes(&self) -> u64 {
        self.total_bytes
            .saturating_sub(self.outside_bytes)
            .saturating_sub(self.own_booked_bytes)
            .saturating_sub(self.lease_bytes)
            .saturating_sub(self.margin_bytes)
    }

    /// The headroom limit in tokens (0 when nothing is free; callers apply the never-0 floor).
    pub(crate) fn cap_tokens(&self) -> usize {
        let per_token = transient_bytes_per_token(self.hidden_size, self.factor);
        if per_token <= 0.0 {
            return usize::MAX;
        }
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        {
            (self.free_bytes() as f64 / per_token) as usize
        }
    }

    /// The largest pool (bytes) this model could be given while its own limit stays at or above
    /// `floor_tokens`, with everything else unchanged: `own_booked_bytes` counts the pool at
    /// `current_pool_bytes`. `None` when even a zero pool leaves less than the floor.
    #[allow(clippy::cast_precision_loss)] // token counts are far below 2^52
    pub(crate) fn max_pool_bytes_for_floor(
        &self,
        current_pool_bytes: u64,
        floor_tokens: usize,
    ) -> Option<u64> {
        let need_free =
            transient_bytes_per_token(self.hidden_size, self.factor) * floor_tokens as f64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let need_free = need_free.ceil() as u64;
        // free(pool) = free(current) + current - pool >= need_free  =>  pool <= free(current) + current - need
        let room = self
            .free_bytes()
            .saturating_add(current_pool_bytes)
            .checked_sub(need_free)?;
        Some(room)
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    use super::*;

    const GIB: u64 = 1 << 30;

    /// Plan 06 section 1.5 (27B, hidden 5120, factor 2, margin 0.5 GiB): the free memory after the
    /// weights and the pool (5.036 GiB for a 4 GiB pool, 1 GiB less for each GiB more) gives these
    /// limits once the margin is taken off.
    #[test]
    fn u12_headroom_numbers_of_the_27b() {
        let cap = |free_after_pool_gib: f64| {
            HeadroomInputs {
                total_bytes: (free_after_pool_gib * GIB as f64) as u64,
                outside_bytes: 0,
                own_booked_bytes: 0,
                lease_bytes: 0,
                margin_bytes: GIB / 2,
                hidden_size: 5120,
                factor: 2.0,
            }
            .cap_tokens()
        };
        assert_eq!(cap(5.036), 118_908, "4 GiB pool");
        assert_eq!(cap(4.036), 92_694, "5 GiB pool");
        assert_eq!(cap(3.036), 66_479, "6 GiB pool");
        assert_eq!(cap(7.036), 171_337, "2 GiB pool");
        assert_eq!(cap(8.036), 197_551, "1 GiB pool");
    }

    #[test]
    fn everything_subtracts_and_saturates() {
        let base = HeadroomInputs {
            total_bytes: 20 * GIB,
            outside_bytes: GIB,
            own_booked_bytes: 15 * GIB,
            lease_bytes: GIB,
            margin_bytes: GIB / 2,
            hidden_size: 4096,
            factor: 2.0,
        };
        assert_eq!(base.free_bytes(), 20 * GIB - GIB - 15 * GIB - GIB - GIB / 2);
        let over = HeadroomInputs {
            own_booked_bytes: 30 * GIB,
            ..base
        };
        assert_eq!(over.free_bytes(), 0);
        assert_eq!(
            over.cap_tokens(),
            0,
            "never negative; the caller applies the floor"
        );
    }

    #[test]
    fn pool_that_keeps_the_floor() {
        // 24 GiB device, 15 GiB weights + 4 GiB pool booked, nothing outside, margin 0.5 GiB.
        let i = HeadroomInputs {
            total_bytes: 24 * GIB,
            outside_bytes: 0,
            own_booked_bytes: 19 * GIB,
            lease_bytes: 0,
            margin_bytes: GIB / 2,
            hidden_size: 5120,
            factor: 2.0,
        };
        let floor = 4096;
        let max = i.max_pool_bytes_for_floor(4 * GIB, floor).unwrap();
        // Rebuilding the inputs with that pool gives a limit of exactly the floor (to a token).
        let rebuilt = HeadroomInputs {
            own_booked_bytes: 15 * GIB + max,
            ..i
        };
        let cap = rebuilt.cap_tokens();
        assert!((floor..=floor + 1).contains(&cap), "{cap}");
        // A device so full that even a zero pool leaves less than the floor.
        let full = HeadroomInputs {
            own_booked_bytes: 25_600_000_000,
            ..i
        };
        assert_eq!(full.max_pool_bytes_for_floor(0, floor), None);
    }
}
