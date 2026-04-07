// SPDX-License-Identifier: Apache-2.0
//! Configuration for fused prefill kernel variants.

use super::units::{Count, Dim, Iters, Tiles};

/// How the GEMM inner loop distributes work across warps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmMode {
    /// All warps redundantly compute the same tile; warp 0 stores.
    Redundant,
    /// Each warp owns 16 rows of the CTA tile; B is shared across warps.
    Cooperative,
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
        }
    }
}
