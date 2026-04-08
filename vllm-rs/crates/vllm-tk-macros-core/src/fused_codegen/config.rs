// SPDX-License-Identifier: Apache-2.0
//! Configuration for fused prefill kernel variants.

use super::units::{Count, Dim, Iters, Tiles};

/// How the GEMM inner loop distributes work across warps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmMode {
    /// All warps redundantly compute the same tile; warp 0 stores.
    Redundant,
    /// Each warp owns `gemm_warp_m` rows of the CTA tile; B is shared across warps.
    /// Requires `cta_rows == num_warps * gemm_warp_m`.
    Cooperative,
    /// Warp 0 is producer (loads only), warps 1..N are consumers (MMA only).
    WarpSpecialized,
}

/// Configuration for one fused prefill kernel variant.
#[derive(Clone, Debug)]
pub struct FusedPrefillConfig {
    /// Rows per CTA (must be multiple of 16).
    pub cta_rows: Dim,
    /// GEMM distribution strategy.
    pub gemm_mode: GemmMode,
    /// K-dimension tile size (64 or 128).
    pub k_dim: Dim,
    /// Output block size (columns per tile).
    pub out_block: Dim,
    /// Number of warps per CTA.
    pub num_warps: Count,
    /// KV page size for attention.
    pub kv_page_size: Dim,
    /// Number of output col tiles computed in parallel per work-unit (redundant mode).
    pub col_batch: Tiles,
    /// Number of pipeline stages (2 or 3).
    pub num_stages: Count,
    /// If true, each warp gets its own B tile — eliminates group::sync from K-loop.
    pub per_warp_b: bool,
    /// If true, the fused gate+up GEMM phase loads A once per K-iter and
    /// accumulates into two separate accumulators (gate_acc, up_acc). This
    /// halves A traffic at the cost of ~33% more gate_up shmem (2 B tiles
    /// per stage) and ~2x registers (two fp32 accumulators live at once).
    /// Only meaningful when gemm_warp_m × out_block × 2 ≤ 4096 register cap.
    pub dual_accum_gate_up: bool,
    /// If true, the fused gate+up phase uses column-fixed CTA scheduling:
    /// each CTA owns a contiguous subset of col tiles and iterates over
    /// row_tiles within each col. Improves L2 reuse on the gate/up B-weight
    /// tiles (they stay hot across row_tile iterations) at the cost of some
    /// load imbalance when col_tiles doesn't evenly divide num_ctas.
    pub col_fixed_schedule: bool,
    /// If true, GEMM phases use a CUTLASS-style K-pipelined inner loop:
    /// instead of loading the full per-warp A tile (`rt_bf<gemm_warp_m,k_dim>`)
    /// once and holding it across the n-loop, the inner loop iterates K-stripes,
    /// loading only `rt_bf<gemm_warp_m,16>` (one k-strip) at a time. Drops the
    /// live A register footprint from `gemm_warp_m * k_dim / 16` registers to
    /// `gemm_warp_m / 16` per K-stripe. This unlocks `gemm_warp_m=64` single-acc
    /// (which otherwise spills) without sacrificing compute density.
    pub kstripe_inner: bool,
    /// Per-warp GEMM accumulator M dimension (must be multiple of 16).
    ///
    /// Each warp accumulates a logical tile of size `gemm_warp_m × out_block`,
    /// issuing `(gemm_warp_m/16) × (out_block/16)` mma.m16n8k16 instructions
    /// per K-iteration. Larger values increase compute density per shmem load
    /// (CUTLASS uses 64×64 or 128×64 warp tiles) at the cost of more registers.
    ///
    /// Cooperative-mode invariant: `cta_rows == num_warps × gemm_warp_m`.
    /// Register budget on sm89 (255/thread): `gemm_warp_m × out_block / 32`
    /// floats for the accumulator, plus a_reg and loop state. Safe envelope:
    /// `gemm_warp_m × out_block ≤ 4096` (e.g. 64×64, 32×128, 16×256).
    ///
    /// This is independent of `PFL_Q_ROWS` (which stays at 16 for attention).
    pub gemm_warp_m: Dim,
}

impl FusedPrefillConfig {
    /// 16-row CTA, redundant GEMM, single col tile.
    pub fn rows16() -> Self {
        Self {
            cta_rows: Dim(16),
            gemm_mode: GemmMode::Redundant,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(8),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(2),
            per_warp_b: false,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(16),
        }
    }

    /// 16-row CTA, redundant GEMM, 4 col tiles in parallel.
    pub fn rows16_col4() -> Self {
        Self {
            col_batch: Tiles(4),
            ..Self::rows16()
        }
    }

    /// 16-row CTA, redundant GEMM, 8 col tiles in parallel.
    pub fn rows16_col8() -> Self {
        Self {
            col_batch: Tiles(8),
            ..Self::rows16()
        }
    }

    /// 32-row CTA, cooperative GEMM (2 warps × 16 rows).
    pub fn rows32() -> Self {
        Self {
            cta_rows: Dim(32),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(8),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(2),
            per_warp_b: false,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(16),
        }
    }

    /// 64-row CTA, cooperative GEMM (4 warps × 16 rows).
    pub fn rows64() -> Self {
        Self {
            cta_rows: Dim(64),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(4),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(2),
            per_warp_b: false,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(16),
        }
    }

    /// 128-row CTA, cooperative GEMM (8 warps × 16 rows).
    pub fn rows128() -> Self {
        Self {
            cta_rows: Dim(128),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(8),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(2),
            per_warp_b: false,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(16),
        }
    }

    /// 128-row CTA, cooperative, wider output block (128 cols).
    pub fn rows128_wide() -> Self {
        Self {
            out_block: Dim(128),
            ..Self::rows128()
        }
    }

    /// 64-row CTA, cooperative, wider K-dim (128).
    pub fn rows64_k128() -> Self {
        Self {
            k_dim: Dim(128),
            ..Self::rows64()
        }
    }

    /// 64-row CTA, cooperative, 3-stage pipeline.
    pub fn rows64_3stage() -> Self {
        Self {
            num_stages: Count(3),
            ..Self::rows64()
        }
    }

    /// Number of 16-row attention passes per CTA.
    pub fn attn_passes(&self) -> Iters {
        Iters(self.cta_rows.0 / 16)
    }

    /// 128-row CTA, warp-specialized producer/consumer.
    /// Warp 0 loads, warps 1-7 compute (112 effective rows). 3-stage pipeline.
    /// cta_rows=128 so non-GEMM phases (rmsnorm, attention) have 16 rows/warp.
    pub fn rows112_warpspec() -> Self {
        Self {
            cta_rows: Dim(128), // 8 warps × 16 rows (producer warp's rows unused in GEMM)
            gemm_mode: GemmMode::WarpSpecialized,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(8),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(2), // 3 stages would need 108KB, L40S limit is 99KB
            per_warp_b: false,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(16),
        }
    }

    /// 64-row CTA, warp-specialized, 4 warps total.
    /// Warp 0 loads, warps 1-3 compute (48 effective rows). 3-stage pipeline.
    /// cta_rows=64 so non-GEMM phases have 16 rows/warp.
    pub fn rows48_warpspec() -> Self {
        Self {
            cta_rows: Dim(64), // 4 warps × 16 rows (producer warp's rows unused in GEMM)
            gemm_mode: GemmMode::WarpSpecialized,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(4),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(3),
            per_warp_b: false,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(16),
        }
    }

    /// 64-row CTA, cooperative, per-warp B tiles (no group::sync in K-loop).
    pub fn rows64_nosync() -> Self {
        Self {
            per_warp_b: true,
            ..Self::rows64()
        }
    }

    /// 32-row CTA, 2 warps, k128, per-warp B (no sync, no bank conflicts).
    /// Shmem: 2 × (8192 + 16384) = 49152/stage × 2 stages = 96KB.
    pub fn rows32_nosync_k128() -> Self {
        Self {
            cta_rows: Dim(32),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(128),
            out_block: Dim(64),
            num_warps: Count(2),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(2),
            per_warp_b: true,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(16),
        }
    }

    /// 64-row CTA, k64, out_block=128, 3-stage pipeline.
    /// Doubles MMA per K-iter (8 N-tiles vs 4), better amortizes sync overhead.
    /// Shmem: 3 × (4×4096 + 16384) = 3×32768 = 96KB.
    pub fn rows64_wide_3stage() -> Self {
        Self {
            cta_rows: Dim(64),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(128),
            num_warps: Count(4),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(3),
            per_warp_b: false,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(16),
        }
    }

    /// 32-row CTA, 2 warps, k64, out_block=128, per-warp B, 2-stage.
    /// Shmem: 2 × (2 × (4096 + 16384)) = 2 × 40960 = 80KB. Fits L40S.
    pub fn rows32_wide_nosync() -> Self {
        Self {
            cta_rows: Dim(32),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(128),
            num_warps: Count(2),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(2),
            per_warp_b: true,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(16),
        }
    }

    // ── gemm_warp_m variants: larger per-warp GEMM accumulators ──
    //
    // These variants exploit the PFL_GEMM_M knob to increase per-warp compute
    // density. Each warp issues (gemm_warp_m/16) × (out_block/16) mma.m16n8k16
    // instructions per K-iter per B-slice load, vs. 1 × (out_block/16) for the
    // baseline gemm_warp_m=16. This is the CUTLASS-style warp-tile win:
    // MMA pipeline saturation via more compute per shmem load.
    //
    // Register budget (sm89 hard cap 255/thread): `gemm_warp_m × out_block / 32`
    // floats for the accumulator alone. All variants below are ≤ 128/thread.
    //
    // Cooperative invariant: cta_rows == num_warps × gemm_warp_m.

    /// 64-row CTA, 2 warps × 32, cooperative. Smallest gemm_warp_m=32 variant.
    /// Reg acc: 2048 floats / 32 = 64 per thread. Shmem: 32KB.
    pub fn rows64_gemm32() -> Self {
        Self {
            cta_rows: Dim(64),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(2),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(2),
            per_warp_b: false,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(32),
        }
    }

    /// 128-row CTA, 4 warps × 32, cooperative. Parallelism + compute density.
    /// Reg acc: 2048 / 32 = 64/thread. Shmem: 48KB.
    pub fn rows128_gemm32() -> Self {
        Self {
            cta_rows: Dim(128),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(4),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(2),
            per_warp_b: false,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(32),
        }
    }

    /// 128-row CTA, 2 warps × 64, cooperative. Max per-warp M density.
    /// Reg acc: 4096 / 32 = 128/thread (at register cap). Shmem: 48KB.
    /// This is the CUTLASS-shaped warp tile (64×64 logical).
    pub fn rows128_gemm64() -> Self {
        Self {
            cta_rows: Dim(128),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(2),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(2),
            per_warp_b: false,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(64),
        }
    }

    /// 128-row CTA, 4 warps × 32, out_block=128. Wide output tile.
    /// Reg acc: 4096 / 32 = 128/thread (at cap). Shmem: 64KB.
    /// More N-direction parallelism; good for large-N GEMMs (gate/up).
    pub fn rows128_gemm32_wide() -> Self {
        Self {
            cta_rows: Dim(128),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(128),
            num_warps: Count(4),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(2),
            per_warp_b: false,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(32),
        }
    }

    /// 64-row CTA, 2 warps × 32, k_dim=128. Bigger K-tile → more work/B-load.
    /// Reg acc: 2048 / 32 = 64/thread. Shmem: 64KB.
    pub fn rows64_gemm32_k128() -> Self {
        Self {
            cta_rows: Dim(64),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(128),
            out_block: Dim(64),
            num_warps: Count(2),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(2),
            per_warp_b: false,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(32),
        }
    }

    // ── Round 2: bigger-CTA / less-barrier variants ──
    //
    // Round 1 showed rows128_gemm32 beats baseline by 7-24%, with the speedup
    // partly from 2x occupancy (shmem-small enough to fit 2 CTAs/SM). Round 2
    // pushes the opposite direction: FEWER bigger CTAs with MORE per-CTA work,
    // so cross-CTA barriers fire half as often. Shmem is sized to force
    // 1 CTA/SM (~80KB/stage × 2 stages), giving grid=sm_count.

    /// 256-row CTA, 8 warps × 32, cooperative. 1 CTA/SM on L4 (82KB).
    /// 2x per-CTA work vs rows128 → 50% fewer mcta_barrier trips.
    /// Reg acc: 2048 / 32 = 64/thread (plenty of headroom).
    /// Shmem: 8 × 4096 + 8192 = 40960 per stage × 2 = 81920 bytes.
    pub fn rows256_gemm32() -> Self {
        Self {
            cta_rows: Dim(256),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(8),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(2),
            per_warp_b: false,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(32),
        }
    }

    /// 256-row CTA, 4 warps × 64, cooperative. CUTLASS-style 64x64 warp tile.
    /// 1 CTA/SM (82KB shmem). Reg acc: 4096 / 32 = 128/thread (at cap).
    /// Shmem: 4 × 8192 + 8192 = 40960 per stage × 2 = 81920 bytes.
    pub fn rows256_gemm64() -> Self {
        Self {
            cta_rows: Dim(256),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(4),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(2),
            per_warp_b: false,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(64),
        }
    }

    /// 128-row CTA, 4 warps × 32, 3-stage pipeline. Deeper pipelining on the
    /// round-1 winner. Shmem: 4 × 4096 + 8192 = 24576 per stage × 3 = 73728.
    /// Still fits 2 CTAs/SM? 73728 × 2 = 147456 > 82K → only 1 CTA/SM.
    pub fn rows128_gemm32_3stage() -> Self {
        Self {
            cta_rows: Dim(128),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(4),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(3),
            per_warp_b: false,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(32),
        }
    }

    // ── Round 3: occupancy vs per-CTA-work tradeoff probes ──

    /// 192-row CTA, 6 warps × 32. Middle ground between 128 (grid=116) and 256 (grid=58).
    /// Shmem: 6 × 4096 + 8192 = 32768 per stage × 2 = 65536. 1 CTA/SM.
    /// Tests whether the sweet spot is between the two round-2 extremes.
    pub fn rows192_gemm32() -> Self {
        Self {
            cta_rows: Dim(192),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(6),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(2),
            per_warp_b: false,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(32),
        }
    }

    /// 64-row CTA, 2 warps × 32, per_warp_b. Eliminates K-loop group::sync by
    /// giving each warp its own B tile copy. Shmem: 2 × (4096 + 8192) × 2 = 49152.
    /// Tests whether K-loop synchronization is a bottleneck on the round-1 winner
    /// shape scaled down to fit per-warp-B shmem.
    pub fn rows64_gemm32_nosync() -> Self {
        Self {
            cta_rows: Dim(64),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(2),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(2),
            per_warp_b: true,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(32),
        }
    }

    /// 128-row CTA, 4 warps × 32, out_block=32. Narrower output tile for
    /// higher occupancy via smaller B shmem. Shmem: 4×4096 + 2048 = 18432 per
    /// stage × 2 = 36864. Targets 2 CTAs/SM with less per-CTA B shmem contention.
    pub fn rows128_gemm32_narrow() -> Self {
        Self {
            cta_rows: Dim(128),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(32),
            num_warps: Count(4),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(2),
            per_warp_b: false,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(32),
        }
    }

    // ── Round 4: dual-accumulator fused gate+up (A reuse) ──

    /// 128-row CTA, 8 warps × 16, cooperative, dual-accum gate+up.
    /// A loaded once per K-iter, applied to both gate_acc and up_acc.
    /// Reg acc: 2 × 16 × 64 / 32 = 64 floats/thread (comfortable).
    /// Shmem/stage: 8 × 2048 + 2 × 8192 = 32768 × 2-stage = 65536 bytes.
    /// Occupancy: 1 CTA/SM (shmem-bound).
    pub fn rows128_gemm16_dual() -> Self {
        Self {
            cta_rows: Dim(128),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(8),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(2),
            per_warp_b: false,
            dual_accum_gate_up: true,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(16),
        }
    }

    /// 128-row CTA, 4 warps × 32, cooperative, dual-accum gate+up.
    /// Combines the gemm_warp_m=32 density win with A reuse.
    /// Reg acc: 2 × 32 × 64 / 32 = 128 floats/thread (at sm89 cap — risky).
    /// Shmem/stage: 4 × 4096 + 2 × 8192 = 32768 × 2-stage = 65536 bytes.
    /// Occupancy: 1 CTA/SM.
    pub fn rows128_gemm32_dual() -> Self {
        Self {
            cta_rows: Dim(128),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(4),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(2),
            per_warp_b: false,
            dual_accum_gate_up: true,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(32),
        }
    }

    // ── Round 5: 1-stage dual_accum variants (sacrifice pipeline → 2 CTAs/SM) ──

    /// 128-row CTA, 8 warps × 16, dual-accum, 1-stage. Shmem: 32768 bytes
    /// total (single stage, no pipelining). Fits 2 CTAs/SM on L4, giving
    /// occupancy parity with rows128_gemm32 while keeping the A-reuse win.
    /// Trade-off: no prefetch pipelining — relies on warp-level latency hiding
    /// across 16 warps/SM.
    pub fn rows128_gemm16_dual_1stage() -> Self {
        Self {
            cta_rows: Dim(128),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(8),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(1),
            per_warp_b: false,
            dual_accum_gate_up: true,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(16),
        }
    }

    /// 128-row CTA, 4 warps × 32, dual-accum, 1-stage. Same shmem as
    /// gemm16_dual_1stage (32768) but compute density from gemm_warp_m=32.
    pub fn rows128_gemm32_dual_1stage() -> Self {
        Self {
            cta_rows: Dim(128),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(4),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(1),
            per_warp_b: false,
            dual_accum_gate_up: true,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(32),
        }
    }

    /// 256-row CTA, 8 warps × 32, dual-accum, 1-stage. Shmem: 8×4096 + 2×8192 = 49152.
    /// Fits 1 CTA/SM (same as rows256_gemm32 but with A reuse).
    pub fn rows256_gemm32_dual_1stage() -> Self {
        Self {
            cta_rows: Dim(256),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(8),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(1),
            per_warp_b: false,
            dual_accum_gate_up: true,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(32),
        }
    }

    // ── Round 6: k_dim=128 variants on top of round 5 winners ──
    //
    // Doubles K-tile size → halves num_k_iters → halves per-iter loop overhead
    // (cp.async.wait, group::sync, prefetch bookkeeping). At k_dim=64 the
    // gate_up phase does 32 iters; at k_dim=128 only 16 iters. Same total
    // compute, same A/B traffic, fewer barriers in the inner loop.

    /// 128-row CTA, 4 warps × 32, dual-accum, 1-stage, k_dim=128.
    /// Shmem: 4×8192 + 2×16384 = 65536. 1 CTA/SM.
    pub fn rows128_gemm32_dual_1stage_k128() -> Self {
        Self {
            cta_rows: Dim(128),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(128),
            out_block: Dim(64),
            num_warps: Count(4),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(1),
            per_warp_b: false,
            dual_accum_gate_up: true,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(32),
        }
    }

    /// 256-row CTA, 8 warps × 32, dual-accum, 1-stage, k_dim=128.
    /// Shmem: 8×8192 + 2×16384 = 98304. 1 CTA/SM (at the 99KB opt-in ceiling).
    pub fn rows256_gemm32_dual_1stage_k128() -> Self {
        Self {
            cta_rows: Dim(256),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(128),
            out_block: Dim(64),
            num_warps: Count(8),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(1),
            per_warp_b: false,
            dual_accum_gate_up: true,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(32),
        }
    }

    /// 256-row CTA, 8 warps × 32, dual-accum, 1-stage, out_block=32.
    /// Shmem: 8×4096 + 2×4096 = 40960. 2 CTAs/SM! (same per-CTA work density
    /// as rows256_gemm32_dual_1stage but with 2x the occupancy.)
    /// Note: out_block=32 means id_col_tiles = 256 (vs 128 for out_block=64),
    /// doubling the work-unit count. L4 scheduler has more parallelism to
    /// work with even at half the per-tile arithmetic intensity.
    pub fn rows256_gemm32_dual_1stage_narrow() -> Self {
        Self {
            cta_rows: Dim(256),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(32),
            num_warps: Count(8),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(1),
            per_warp_b: false,
            dual_accum_gate_up: true,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(32),
        }
    }

    // ── Round 7: col-fixed scheduling for L2 reuse on gate_up B tiles ──

    /// 256-row CTA, 8 warps × 32, dual-accum, 1-stage, col-fixed scheduling.
    /// Each CTA owns a contiguous set of col tiles and iterates over row_tiles
    /// within each col. The gate/up B weight for a given col stays hot in L2
    /// across the CTA's row_tile iterations. Imbalance: at col_tiles=128 and
    /// num_ctas=58, 12 CTAs get 3 cols × 4 rows = 12 wu each, 46 CTAs get 8 wu.
    /// Max/avg ratio ≈ 1.36. If the L2-reuse win exceeds 36%, this wins.
    pub fn rows256_gemm32_dual_1stage_colfix() -> Self {
        Self {
            cta_rows: Dim(256),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(8),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(1),
            per_warp_b: false,
            dual_accum_gate_up: true,
            col_fixed_schedule: true,
            kstripe_inner: false,
            gemm_warp_m: Dim(32),
        }
    }

    // ── Round 8: K-pipelined inner loop (CUTLASS-style register lifetime) ──
    //
    // Instead of `warp::load(a_reg, a_smem)` loading the full per-warp A
    // (`rt_bf<gemm_warp_m, k_dim>`) up front, the inner loop iterates K-stripes
    // and loads only `rt_bf<gemm_warp_m, 16>` (1 K-strip) at a time. The live
    // A register footprint drops from `gemm_warp_m × k_dim / 16` to
    // `gemm_warp_m / 16` per stripe. This unlocks `gemm_warp_m=64` single-acc.

    /// 256-row CTA, 4 warps × 64, single-accum, 1-stage, K-stripe inner loop.
    /// Reg budget (with K-pipelining): acc 128 + a_strip 16 + b_strip 4 ≈ 168/lane.
    /// Shmem: 4×8192 + 8192 = 40960 → fits 2 CTAs/SM on L4.
    /// Combines CUTLASS-shape 64×64 warp tile + high occupancy + small registers.
    pub fn rows256_gemm64_kstripe_1stage() -> Self {
        Self {
            cta_rows: Dim(256),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(4),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(1),
            per_warp_b: false,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: true,
            gemm_warp_m: Dim(64),
        }
    }

    /// 256-row CTA, 8 warps × 32, dual-accum, 2-stage.
    /// Same as `_large` polyalgo winner (1-stage) but with shmem prefetch
    /// pipeline. Shmem: 49152 × 2 = 98304 bytes (just under 99KB opt-in cap).
    /// 1 CTA/SM (shmem-bound). Tests if cp.async pipelining helps when
    /// we have the shmem budget for it.
    pub fn rows256_gemm32_dual_2stage() -> Self {
        Self {
            cta_rows: Dim(256),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(8),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(2),
            per_warp_b: false,
            dual_accum_gate_up: true,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(32),
        }
    }

    /// 256-row CTA, 8 warps × 32, single-accum, 1-stage.
    /// Drops dual_accum vs the current `_large` winner; gate_up reverts to
    /// back-to-back gate then up. The win: shmem drops from 49152 → 40960
    /// (single B tile per stage instead of two), unlocking 2 CTAs/SM!
    /// Loses A-reuse in gate_up but gains 2× occupancy across all phases.
    pub fn rows256_gemm32_1stage() -> Self {
        Self {
            cta_rows: Dim(256),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(8),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(1),
            per_warp_b: false,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: false,
            gemm_warp_m: Dim(32),
        }
    }

    /// 256-row CTA, 8 warps × 32, dual-accum + K-stripe, 1-stage.
    /// Same shape as the current `_large` polyalgo winner, but with K-stripe
    /// inner loop. Tests whether reducing register pressure on the dual-accum
    /// hot path improves compiler scheduling. Shmem unchanged: 49152 bytes,
    /// 1 CTA/SM. Live regs: 2×acc(64) + a_strip(16) + 2×b_strip(2) ≈ 146/lane
    /// vs ~210/lane without kstripe (32 reg savings).
    pub fn rows256_gemm32_dual_1stage_kstripe() -> Self {
        Self {
            cta_rows: Dim(256),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(8),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(1),
            per_warp_b: false,
            dual_accum_gate_up: true,
            col_fixed_schedule: false,
            kstripe_inner: true,
            gemm_warp_m: Dim(32),
        }
    }

    /// 256-row CTA, 8 warps × 32, single-accum, 1-stage, K-stripe, out_block=128.
    /// Wider N tile (closer to CUTLASS 128 threadblock N). Shmem: 8×4096 + 16384 = 49152.
    /// Reg with kstripe: rt_fl<32,128>=128/lane + a_strip 8 + b_strip 8 ≈ 150/lane.
    pub fn rows256_gemm32_kstripe_wide_1stage() -> Self {
        Self {
            cta_rows: Dim(256),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(128),
            num_warps: Count(8),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(1),
            per_warp_b: false,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: true,
            gemm_warp_m: Dim(32),
        }
    }

    /// 128-row CTA, 2 warps × 64, single-accum, 1-stage, K-stripe inner loop.
    /// Smaller variant for medium seq. Shmem: 2×8192 + 8192 = 24576 → 2 CTAs/SM.
    pub fn rows128_gemm64_kstripe_1stage() -> Self {
        Self {
            cta_rows: Dim(128),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(2),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(1),
            per_warp_b: false,
            dual_accum_gate_up: false,
            col_fixed_schedule: false,
            kstripe_inner: true,
            gemm_warp_m: Dim(64),
        }
    }

    /// 128-row CTA, 8 warps × 16, dual-accum, 1-stage, col-fixed scheduling.
    /// Same as rows128_gemm16_dual_1stage but with col-fixed CTA scheduling.
    /// At rows128 with 2 CTAs/SM → grid=116, L2 reuse in a different regime.
    pub fn rows128_gemm16_dual_1stage_colfix() -> Self {
        Self {
            cta_rows: Dim(128),
            gemm_mode: GemmMode::Cooperative,
            k_dim: Dim(64),
            out_block: Dim(64),
            num_warps: Count(8),
            kv_page_size: Dim(64),
            col_batch: Tiles(1),
            num_stages: Count(1),
            per_warp_b: false,
            dual_accum_gate_up: true,
            col_fixed_schedule: true,
            kstripe_inner: false,
            gemm_warp_m: Dim(16),
        }
    }
}
