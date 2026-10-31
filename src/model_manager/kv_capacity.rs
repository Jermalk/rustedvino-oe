// ============================================================
// src/model_manager/kv_capacity.rs — KV pool capacity model (blocks-first)
// ============================================================
// Mirrors how `GenAI` turns a byte budget into KV blocks, so the server can state
// the capacity of a pool exactly instead of estimating it:
//   `GenAI` `normalize_block_counts`, `cache_orchestrator.hpp:834-920`,
//   releases/2026/4 @ `ab01d1771e4`.
// The inputs are a [`KvProfile`] (layer geometry, LA state per sequence), the KV
// precision, and whether prefix caching is on. Pure arithmetic: no I/O, no GPU.
//
// Unverified against the 2026.4.1 source (only a 2026.2 copy and the 2026.4 header
// were available): the default prefix-cache interval multiplier `clamp(ceil(la/kv),
// 8, 256)` (verified on 2026.4.1 by hardware test G-12: adaptive, not a fixed 8).
// ============================================================

use std::fmt;

use super::kv_profile::{GPU_BLOCK_TOKENS, KvPrecision, KvProfile};

/// `OpenVINO` / `GenAI` version pairs (normalised, see `kv_profile::engine_pair`) whose
/// cache layout the formulas were verified against. Any other pair falls back to
/// the legacy gate unless the operator opts in.
pub(crate) const KNOWN_ENGINE_PAIRS: &[(&str, &str)] = &[("2026.4.1", "2026.4.1")];

/// How many linear-attention state blocks the engine reserves when prefix caching is off.
/// (Upstream picks `max_num_seqs` blocks unless `max_num_batched_tokens` is unbounded;
/// the VLM path configures `max_num_seqs = 1`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LaBlocksRule {
    /// One LA block per concurrent sequence (`max_num_seqs`).
    MaxNumSeqs(usize),
    /// A single LA block.
    One,
}

impl LaBlocksRule {
    fn blocks(self) -> u64 {
        match self {
            Self::MaxNumSeqs(n) => n as u64,
            Self::One => 1,
        }
    }
}

/// The pool `GenAI` will build for a byte budget or a block count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PoolPlan {
    /// Paged-attention KV blocks (`num_kv_blocks`); never 0 (0 means "size from `cache_size`").
    pub kv_blocks: u64,
    /// Linear-attention state blocks (0 for dense models).
    pub la_blocks: u64,
    /// `kv_blocks · kv_block_bytes + la_blocks · la_bytes`: what the engine allocates
    /// (equals the engine's `kv_cache_size_in_bytes`) and what the tracker books.
    pub bytes_allocated: u64,
    /// Tokens the pool can hold for one sequence.
    pub capacity_tokens: u64,
    /// Prefix-cache checkpoint interval in tokens (hybrid models with prefix caching on).
    pub interval_tokens: Option<u64>,
}

/// Why a pool cannot be planned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CapacityError {
    /// The linear-attention reservation alone exceeds the budget.
    LaExceedsBudget {
        /// Bytes the LA reservation needs.
        needed: u64,
        /// Bytes available.
        budget: u64,
    },
    /// The budget does not buy a single KV block (`num_kv_blocks = 0` would make the
    /// engine size the pool itself, losing the blocks-first guarantee).
    BudgetTooSmall {
        /// Bytes needed for one block plus any LA reservation.
        needed: u64,
        /// Bytes available.
        budget: u64,
    },
    /// The model has no paged-attention layers.
    EmptyProfile,
    /// The formulas are the GPU layout (16-token blocks); a CPU profile must not be planned.
    UnsupportedBlockSize,
}

impl fmt::Display for CapacityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LaExceedsBudget { needed, budget } => write!(
                f,
                "the linear-attention state reservation ({needed} bytes) exceeds the KV budget ({budget} bytes)"
            ),
            Self::BudgetTooSmall { needed, budget } => write!(
                f,
                "the KV budget ({budget} bytes) does not hold one block ({needed} bytes needed)"
            ),
            Self::EmptyProfile => write!(f, "the model has no paged-attention layers"),
            Self::UnsupportedBlockSize => {
                write!(
                    f,
                    "the capacity model covers the GPU layout (16-token blocks) only"
                )
            }
        }
    }
}

impl std::error::Error for CapacityError {}

/// Bounds of the prefix-cache interval multiplier (`GenAI` `cache_orchestrator.hpp`).
const INTERVAL_MULTIPLIER_MIN: u64 = 8;
const INTERVAL_MULTIPLIER_MAX: u64 = 256;

impl KvProfile {
    /// Checks shared by both planners; returns `(block_bytes, block_tokens, la_bytes)`.
    fn plan_inputs(&self, precision: KvPrecision) -> Result<(u64, u64, u64), CapacityError> {
        if self.block_tokens != GPU_BLOCK_TOKENS {
            return Err(CapacityError::UnsupportedBlockSize);
        }
        let block_bytes = self.kv_block_bytes(precision);
        if block_bytes == 0 {
            return Err(CapacityError::EmptyProfile);
        }
        Ok((
            block_bytes,
            self.block_tokens as u64,
            self.linear_state_bytes_per_seq,
        ))
    }

    /// Prefix-cache interval in tokens for a hybrid model: `block·clamp(ceil(la/kv), 8, 256)`,
    /// or `block·multiplier` when the operator gives one (unclamped; 0 is treated as 1).
    fn prefix_interval(
        block_tokens: u64,
        block_bytes: u64,
        la_bytes: u64,
        multiplier: Option<u64>,
    ) -> u64 {
        let m = multiplier
            .unwrap_or_else(|| {
                la_bytes
                    .div_ceil(block_bytes)
                    .clamp(INTERVAL_MULTIPLIER_MIN, INTERVAL_MULTIPLIER_MAX)
            })
            .max(1);
        block_tokens * m
    }

    /// Plan the pool `GenAI` builds for `budget_bytes` (the configured `GiB`, decimals
    /// included, times 2^30).
    ///
    /// - prefix caching off: `kv_blocks = floor((budget − la_blocks·la_bytes) / kv_block_bytes)`;
    ///   [`CapacityError::LaExceedsBudget`] when the LA reservation alone does not fit.
    /// - prefix caching on, hybrid model: the largest token count `T` with
    ///   `ceil(T/block)·kv_block_bytes + ceil(T/interval)·la_bytes ≤ budget`.
    /// - dense model: `floor(budget / kv_block_bytes)`, prefix caching or not.
    ///
    /// A budget that does not buy one block is [`CapacityError::BudgetTooSmall`] in every
    /// branch.
    pub(crate) fn plan_pool(
        &self,
        budget_bytes: u64,
        precision: KvPrecision,
        prefix_caching: bool,
        la_rule: LaBlocksRule,
        cache_interval_multiplier: Option<u64>,
    ) -> Result<PoolPlan, CapacityError> {
        let (block_bytes, block_tokens, la_bytes) = self.plan_inputs(precision)?;
        let too_small = |needed: u64| CapacityError::BudgetTooSmall {
            needed,
            budget: budget_bytes,
        };

        if la_bytes == 0 {
            let kv_blocks = budget_bytes / block_bytes;
            if kv_blocks == 0 {
                return Err(too_small(block_bytes));
            }
            return Ok(PoolPlan {
                kv_blocks,
                la_blocks: 0,
                bytes_allocated: kv_blocks * block_bytes,
                capacity_tokens: kv_blocks * block_tokens,
                interval_tokens: None,
            });
        }

        if !prefix_caching {
            let la_blocks = la_rule.blocks();
            let la_total = la_blocks.saturating_mul(la_bytes);
            if la_total > budget_bytes {
                return Err(CapacityError::LaExceedsBudget {
                    needed: la_total,
                    budget: budget_bytes,
                });
            }
            let kv_blocks = (budget_bytes - la_total) / block_bytes;
            if kv_blocks == 0 {
                return Err(too_small(la_total.saturating_add(block_bytes)));
            }
            return Ok(PoolPlan {
                kv_blocks,
                la_blocks,
                bytes_allocated: kv_blocks * block_bytes + la_total,
                capacity_tokens: kv_blocks * block_tokens,
                interval_tokens: None,
            });
        }

        let interval = Self::prefix_interval(
            block_tokens,
            block_bytes,
            la_bytes,
            cache_interval_multiplier,
        );
        let cost = |t: u64| {
            t.div_ceil(block_tokens)
                .saturating_mul(block_bytes)
                .saturating_add(t.div_ceil(interval).saturating_mul(la_bytes))
        };
        if cost(1) > budget_bytes {
            return Err(too_small(cost(1)));
        }
        // `cost` is non-decreasing in T: binary search the largest T that fits.
        let (mut lo, mut hi) = (
            1u64,
            budget_bytes / block_bytes * block_tokens + block_tokens,
        );
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if cost(mid) <= budget_bytes {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let t = lo;
        let kv_blocks = t.div_ceil(block_tokens);
        let la_blocks = t.div_ceil(interval);
        Ok(PoolPlan {
            kv_blocks,
            la_blocks,
            bytes_allocated: kv_blocks * block_bytes + la_blocks * la_bytes,
            capacity_tokens: t,
            interval_tokens: Some(interval),
        })
    }

    /// The pool `GenAI` builds when `num_kv_blocks` is fixed (the engine takes the block
    /// count as given; there is no budget to split).
    ///
    /// Capacity is `kv_blocks · block_tokens`. What differs is the bytes: prefix caching off
    /// reserves `la_rule` LA blocks, prefix caching on a hybrid model reserves
    /// `ceil(capacity/interval)` of them. [`PoolPlan::bytes_allocated`] is what to book.
    pub(crate) fn plan_for_blocks(
        &self,
        kv_blocks: u64,
        precision: KvPrecision,
        prefix_caching: bool,
        la_rule: LaBlocksRule,
        cache_interval_multiplier: Option<u64>,
    ) -> Result<PoolPlan, CapacityError> {
        let (block_bytes, block_tokens, la_bytes) = self.plan_inputs(precision)?;
        if kv_blocks == 0 {
            return Err(CapacityError::BudgetTooSmall {
                needed: block_bytes,
                budget: 0,
            });
        }
        let capacity_tokens = kv_blocks * block_tokens;
        let (la_blocks, interval_tokens) = if la_bytes == 0 {
            (0, None)
        } else if prefix_caching {
            let interval = Self::prefix_interval(
                block_tokens,
                block_bytes,
                la_bytes,
                cache_interval_multiplier,
            );
            (capacity_tokens.div_ceil(interval), Some(interval))
        } else {
            (la_rule.blocks(), None)
        };
        Ok(PoolPlan {
            kv_blocks,
            la_blocks,
            bytes_allocated: kv_blocks * block_bytes + la_blocks * la_bytes,
            capacity_tokens,
            interval_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    use super::super::kv_profile::{Confidence, GPU_BLOCK_TOKENS, PaLayer};
    use super::*;

    const GIB: u64 = 1 << 30;

    fn profile(layers: usize, heads: usize, dim: usize, la_bytes: u64) -> KvProfile {
        KvProfile {
            pa_layers: vec![
                PaLayer {
                    kv_heads: heads,
                    k_head_dim: dim,
                    v_head_dim: dim,
                };
                layers
            ],
            block_tokens: GPU_BLOCK_TOKENS,
            weights_4bit: Some(true),
            ir_kv_hint: None,
            linear_state_bytes_per_seq: la_bytes,
            hidden_size: 4096,
            confidence: Confidence::High,
        }
    }

    fn q27b() -> KvProfile {
        profile(16, 4, 256, 79_429_632)
    }
    fn q8b() -> KvProfile {
        profile(36, 8, 128, 0)
    }

    fn off(p: &KvProfile, gib: f64, prec: KvPrecision, rule: LaBlocksRule) -> PoolPlan {
        p.plan_pool((gib * GIB as f64) as u64, prec, false, rule, None)
            .expect("plan")
    }

    #[test]
    fn u05_27b_prefix_off() {
        let p = q27b();
        let a = off(&p, 4.0, KvPrecision::U4, LaBlocksRule::One);
        assert_eq!((a.kv_blocks, a.capacity_tokens), (12_705, 203_280));
        assert_eq!(a.bytes_allocated, 4_294_643_712);
        let a = off(&p, 4.0, KvPrecision::I8, LaBlocksRule::One);
        assert_eq!((a.kv_blocks, a.capacity_tokens), (7_097, 113_552));
        let a = off(&p, 4.0, KvPrecision::F16, LaBlocksRule::One);
        assert_eq!((a.kv_blocks, a.capacity_tokens), (4_020, 64_320));
        let a = off(&p, 1.0, KvPrecision::U4, LaBlocksRule::One);
        assert_eq!((a.kv_blocks, a.capacity_tokens), (2_996, 47_936));
        let a = off(&p, 1.0, KvPrecision::I8, LaBlocksRule::One);
        assert_eq!((a.kv_blocks, a.capacity_tokens), (1_674, 26_784));
        assert_eq!(
            off(&p, 2.0, KvPrecision::U4, LaBlocksRule::One).capacity_tokens,
            99_728
        );
    }

    /// G-12 (B60, `GenAI` 2026.4.1, qwen3.5-4b, 8 PA layers + 26,738,688 B of LA state per sequence):
    /// the default multiplier is adaptive, `ceil(la/kv) = 162`, so the interval is 2,592 tokens.
    /// Measured: 13 LA blocks at 2,000 KV blocks (a fixed 8 would give 250) and
    /// 1,072,250,880 B for a 1 GiB budget (3,240 KV blocks + 20 LA blocks).
    #[test]
    fn g12_adaptive_interval_on_the_4b() {
        let p = profile(8, 4, 256, 26_738_688);
        assert_eq!(p.kv_block_bytes(KvPrecision::U4), 165_888);
        let a = p
            .plan_for_blocks(2_000, KvPrecision::U4, true, LaBlocksRule::One, None)
            .expect("plan");
        assert_eq!(a.interval_tokens, Some(2_592));
        assert_eq!(a.la_blocks, 13);
        assert_eq!(a.bytes_allocated, 679_378_944, "measured on-2000");
        let b = p
            .plan_pool(GIB, KvPrecision::U4, true, LaBlocksRule::One, None)
            .expect("plan");
        assert_eq!((b.kv_blocks, b.la_blocks), (3_240, 20));
        assert_eq!(b.bytes_allocated, 1_072_250_880, "measured budget-1g");
        // The operator's fixed multiplier still wins (control run: x8 at 500 blocks).
        let c = p
            .plan_for_blocks(500, KvPrecision::U4, true, LaBlocksRule::One, Some(8))
            .expect("plan");
        assert_eq!(c.bytes_allocated, 1_767_481_344, "measured ctrl-x8-500");
    }

    /// Block B1/B2a, hardware: the 27B (hybrid, 16 PA layers of (4, 256), 79,429,632 B of LA state
    /// per sequence) with prefix caching on gets 1,560 / 3,120 / 6,480 blocks at 1 / 2 / 4 GiB.
    /// The steps are not linear because LA state is reserved in whole blocks of `ceil(T/3840)`:
    /// at 2 GiB a 3,121st KV block would need a 14th LA block and miss the budget by 4,096 B.
    #[test]
    fn b1_27b_prefix_on_block_counts_and_the_la_stair_step() {
        let p = q27b();
        let at = |gib: u64| {
            p.plan_pool(gib * GIB, KvPrecision::U4, true, LaBlocksRule::One, None)
                .expect("plan")
        };
        assert_eq!(at(1).kv_blocks, 1_560);
        let two = at(2);
        assert_eq!((two.kv_blocks, two.la_blocks), (3_120, 13));
        assert_eq!(two.bytes_allocated, 2_067_726_336);
        assert_eq!(two.capacity_tokens, 49_920);
        assert_eq!(at(4).kv_blocks, 6_480);
        // One more block misses the budget by 4,096 B because it needs a 14th LA block.
        let more = p
            .plan_for_blocks(3_121, KvPrecision::U4, true, LaBlocksRule::One, None)
            .expect("plan");
        assert_eq!(more.la_blocks, 14);
        assert_eq!(more.bytes_allocated - 2 * GIB, 4_096);
    }

    #[test]
    fn u05_27b_prefix_on_interval_3840() {
        let p = q27b();
        let a = p
            .plan_pool(4 * GIB, KvPrecision::U4, true, LaBlocksRule::One, None)
            .expect("plan");
        assert_eq!(a.interval_tokens, Some(3_840));
        assert_eq!(a.capacity_tokens, 103_680, "{a:?}");
        assert!(a.bytes_allocated <= 4 * GIB);
    }

    #[test]
    fn u05_dense_8b() {
        let p = q8b();
        let a = off(&p, 4.0, KvPrecision::I8, LaBlocksRule::MaxNumSeqs(4));
        assert_eq!(
            (a.kv_blocks, a.capacity_tokens, a.la_blocks),
            (3_192, 51_072, 0)
        );
        let a = off(&p, 4.0, KvPrecision::U4, LaBlocksRule::One);
        assert_eq!((a.kv_blocks, a.capacity_tokens), (5_683, 90_928));
        let a = off(&p, 1.0, KvPrecision::I8, LaBlocksRule::One);
        assert_eq!((a.kv_blocks, a.capacity_tokens), (798, 12_768));
        // Dense: prefix caching changes nothing.
        let on = p
            .plan_pool(4 * GIB, KvPrecision::U4, true, LaBlocksRule::One, None)
            .expect("plan");
        assert_eq!(on.capacity_tokens, 90_928);
        assert_eq!(on.interval_tokens, None);
    }

    #[test]
    fn u05_phi_vision_f16() {
        let p = profile(32, 32, 96, 0);
        let a = off(&p, 2.0, KvPrecision::F16, LaBlocksRule::One);
        assert_eq!((a.kv_blocks, a.capacity_tokens), (341, 5_456));
        let a = off(&p, 2.0, KvPrecision::U4, LaBlocksRule::One);
        assert_eq!(a.capacity_tokens, 16_912);
    }

    #[test]
    fn u05_lfm2_prefix_on() {
        let p = profile(10, 8, 64, 368_640);
        let a = p
            .plan_pool(GIB, KvPrecision::U4, true, LaBlocksRule::One, None)
            .expect("plan");
        assert_eq!(a.interval_tokens, Some(128));
        assert_eq!(p.kv_bytes_per_token(KvPrecision::U4), 6_720);
        assert_eq!(p.kv_bytes_per_token(KvPrecision::I8), 11_840);
        // 2,880 B/token of LA state at that interval (368,640 / 128).
        assert_eq!(368_640 / 128, 2_880);
        assert!(a.bytes_allocated <= GIB);
    }

    #[test]
    fn u05_cb_path_reserves_one_la_block_per_sequence() {
        // qwen3.5-4b: 8 layers (4 x 256), LA 26,738,688, max_num_seqs 4, 2,000 blocks.
        let p = profile(8, 4, 256, 26_738_688);
        assert_eq!(p.kv_block_bytes(KvPrecision::U4), 165_888);
        // Budget that yields exactly 2,000 blocks: 2,000 x 165,888 + 4 x 26,738,688.
        let budget = 2_000 * 165_888 + 4 * 26_738_688;
        let a = p
            .plan_pool(
                budget,
                KvPrecision::U4,
                false,
                LaBlocksRule::MaxNumSeqs(4),
                None,
            )
            .expect("plan");
        assert_eq!((a.kv_blocks, a.la_blocks), (2_000, 4));
        assert_eq!(a.bytes_allocated, 438_730_752);
    }

    #[test]
    fn u05_omnicoder_cb_8_seqs() {
        let p = profile(8, 4, 256, 26_738_688);
        let a = off(&p, 4.0, KvPrecision::U4, LaBlocksRule::MaxNumSeqs(8));
        let want = (4 * GIB - 8 * 26_738_688) / 165_888;
        assert_eq!(a.kv_blocks, want);
        assert_eq!(a.la_blocks, 8);
    }

    #[test]
    fn u05_la_exceeds_budget() {
        // 27B, max_num_seqs 256: 256 x 79 MB = 20 GB > 4 GiB.
        let err = q27b()
            .plan_pool(
                4 * GIB,
                KvPrecision::U4,
                false,
                LaBlocksRule::MaxNumSeqs(256),
                None,
            )
            .expect_err("must not fit");
        assert!(matches!(err, CapacityError::LaExceedsBudget { .. }));
        assert!(err.to_string().contains("exceeds the KV budget"));
    }

    #[test]
    fn u05_fractional_pools() {
        // 27B 4.2 GiB u4 -> 13,353 blocks, 4,509,634,560 B (4.1999 GiB).
        let a = off(&q27b(), 4.2, KvPrecision::U4, LaBlocksRule::One);
        assert_eq!(a.kv_blocks, 13_353);
        assert_eq!(a.bytes_allocated, 4_509_634_560);
        // qwen3-14b i8 4.2 GiB -> 3,016 blocks (4.1994 GiB).
        let p = profile(40, 8, 128, 0);
        let a = off(&p, 4.2, KvPrecision::I8, LaBlocksRule::One);
        assert_eq!(a.kv_blocks, 3_016);
        let gib = a.bytes_allocated as f64 / GIB as f64;
        assert!((gib - 4.1994).abs() < 1e-3, "{gib}");
    }

    #[test]
    fn explicit_interval_multiplier_is_unclamped() {
        let p = q27b();
        let a = p
            .plan_pool(4 * GIB, KvPrecision::U4, true, LaBlocksRule::One, Some(4))
            .expect("plan");
        assert_eq!(a.interval_tokens, Some(64));
        assert!(a.bytes_allocated <= 4 * GIB);
    }

    #[test]
    fn prefix_on_search_is_maximal() {
        // One more token must not fit.
        let p = q27b();
        let a = p
            .plan_pool(4 * GIB, KvPrecision::U4, true, LaBlocksRule::One, None)
            .expect("plan");
        let t = a.capacity_tokens + 1;
        let block = p.kv_block_bytes(KvPrecision::U4);
        let interval = a.interval_tokens.expect("interval");
        let cost = t.div_ceil(16) * block + t.div_ceil(interval) * p.linear_state_bytes_per_seq;
        assert!(cost > 4 * GIB);
    }

    #[test]
    fn empty_profile_is_an_error() {
        let p = profile(0, 8, 128, 0);
        assert_eq!(
            p.plan_pool(GIB, KvPrecision::U4, false, LaBlocksRule::One, None),
            Err(CapacityError::EmptyProfile)
        );
    }

    #[test]
    fn plan_for_blocks_books_the_la_reservation() {
        // Dense: bytes are the blocks only.
        let a = q8b()
            .plan_for_blocks(3_192, KvPrecision::I8, false, LaBlocksRule::One, None)
            .expect("plan");
        assert_eq!((a.capacity_tokens, a.la_blocks), (51_072, 0));
        assert_eq!(a.bytes_allocated, 3_192 * 1_345_536);
        // Hybrid, prefix off: 7,097 blocks + one LA block = the planned pool.
        let h = q27b()
            .plan_for_blocks(7_097, KvPrecision::I8, false, LaBlocksRule::One, None)
            .expect("plan");
        assert_eq!(h.capacity_tokens, 113_552);
        assert_eq!(h.la_blocks, 1);
        let planned = off(&q27b(), 4.0, KvPrecision::I8, LaBlocksRule::One);
        assert_eq!(h.bytes_allocated, planned.bytes_allocated);
        // Hybrid, prefix on: capacity is blocks x 16; LA blocks follow the interval.
        let on = q27b()
            .plan_for_blocks(6_480, KvPrecision::U4, true, LaBlocksRule::One, None)
            .expect("plan");
        assert_eq!(on.capacity_tokens, 103_680);
        assert_eq!((on.la_blocks, on.interval_tokens), (27, Some(3_840)));
        let planned = q27b()
            .plan_pool(4 * GIB, KvPrecision::U4, true, LaBlocksRule::One, None)
            .expect("plan");
        assert_eq!(on.bytes_allocated, planned.bytes_allocated);
        // CB path: qwen3.5-4b, 2,000 blocks, 4 sequences = the measured pool.
        let cb = profile(8, 4, 256, 26_738_688)
            .plan_for_blocks(
                2_000,
                KvPrecision::U4,
                false,
                LaBlocksRule::MaxNumSeqs(4),
                None,
            )
            .expect("plan");
        assert_eq!(cb.bytes_allocated, 438_730_752);
    }

    #[test]
    fn zero_blocks_are_refused_everywhere() {
        // A budget that buys no block would become num_kv_blocks = 0, which the engine
        // reads as "size it yourself": refuse in all three branches and in plan_for_blocks.
        let dense = q8b();
        assert!(matches!(
            dense.plan_pool(1_000, KvPrecision::U4, false, LaBlocksRule::One, None),
            Err(CapacityError::BudgetTooSmall { .. })
        ));
        let hybrid = q27b();
        let la = 79_429_632;
        assert!(matches!(
            hybrid.plan_pool(la + 100, KvPrecision::U4, false, LaBlocksRule::One, None),
            Err(CapacityError::BudgetTooSmall { .. })
        ));
        assert!(matches!(
            hybrid.plan_pool(la + 100, KvPrecision::U4, true, LaBlocksRule::One, None),
            Err(CapacityError::BudgetTooSmall { .. })
        ));
        assert!(matches!(
            dense.plan_for_blocks(0, KvPrecision::U4, false, LaBlocksRule::One, None),
            Err(CapacityError::BudgetTooSmall { .. })
        ));
    }

    #[test]
    fn cpu_block_size_is_not_planned() {
        let mut p = q8b();
        p.block_tokens = 32;
        assert_eq!(
            p.plan_pool(GIB, KvPrecision::U4, false, LaBlocksRule::One, None),
            Err(CapacityError::UnsupportedBlockSize)
        );
    }

    #[test]
    fn known_engine_pairs_hold_the_verified_release() {
        assert!(KNOWN_ENGINE_PAIRS.contains(&("2026.4.1", "2026.4.1")));
    }
}
