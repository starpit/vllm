// SPDX-License-Identifier: Apache-2.0
//! Configuration for the fused prefill kernel.

/// GEMM parallelism strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmMode {
    /// All warps load same A, compute same result, warp 0 stores.
    /// Each CTA owns 16 rows total.
    Redundant,
    /// Each warp owns its own 16-row A slice. 8 warps × 16 rows = 128 rows per CTA.
    /// All warps store their own results.
    Cooperative,
}

/// Configuration for a fused prefill kernel variant.
#[derive(Clone, Debug)]
pub struct FusedPrefillConfig {
    /// Total rows owned by each CTA (16 for Redundant, 128 for Cooperative).
    pub cta_rows: usize,
    /// GEMM parallelism strategy.
    pub gemm_mode: GemmMode,
    /// GEMM K-dimension tile size.
    pub k_dim: usize,
    /// GEMM output column tile size.
    pub out_block: usize,
    /// Number of cooperative warps per CTA.
    pub num_warps: usize,
    /// KV cache page size (tokens per page).
    pub kv_page_size: usize,
    /// Number of output column tiles computed in parallel per K-loop pass.
    /// 1 = all warps compute same col (redundant). N = N warps compute different cols.
    /// Must divide num_warps. Shmem = 2 * (a_size + col_batch * b_size).
    pub col_batch: usize,
}

impl FusedPrefillConfig {
    /// Original 16-row redundant config (matches existing v1 monolith).
    pub fn rows16() -> Self {
        Self {
            cta_rows: 16,
            gemm_mode: GemmMode::Redundant,
            k_dim: 64,
            out_block: 64,
            num_warps: 8,
            kv_page_size: 64,
            col_batch: 1,
        }
    }

    /// 16-row config with 4-way col-distributed GEMMs.
    /// 4 warps compute different cols; 4 warps help with loads.
    /// Shmem: 2 × (A + 4×B) = 2 × (4096 + 32768) = 73728 = 72KB. Fits 99KB.
    pub fn rows16_col4() -> Self {
        Self {
            cta_rows: 16,
            gemm_mode: GemmMode::Redundant,
            k_dim: 64,
            out_block: 64,
            num_warps: 8,
            kv_page_size: 64,
            col_batch: 4,
        }
    }

    /// 16-row config with 8-way col-distributed GEMMs (no B double-buffer).
    /// All 8 warps compute different cols.
    /// Shmem: 2 × A + 8 × B = 2×4096 + 8×8192 = 73728 = 72KB. Fits 99KB.
    /// (A is double-buffered, B is single-staged per col_base iteration.)
    pub fn rows16_col8() -> Self {
        Self {
            cta_rows: 16,
            gemm_mode: GemmMode::Redundant,
            k_dim: 64,
            out_block: 64,
            num_warps: 8,
            kv_page_size: 64,
            col_batch: 8,
        }
    }

    /// 32-row cooperative GEMM config.
    pub fn rows32() -> Self {
        Self {
            cta_rows: 32,
            gemm_mode: GemmMode::Cooperative,
            k_dim: 64,
            out_block: 64,
            num_warps: 8,
            kv_page_size: 64,
            col_batch: 1,
        }
    }

    /// 64-row cooperative GEMM config.
    pub fn rows64() -> Self {
        Self {
            cta_rows: 64,
            gemm_mode: GemmMode::Cooperative,
            k_dim: 64,
            out_block: 64,
            num_warps: 8,
            kv_page_size: 64,
            col_batch: 1,
        }
    }

    /// 128-row cooperative GEMM config.
    pub fn rows128() -> Self {
        Self {
            cta_rows: 128,
            gemm_mode: GemmMode::Cooperative,
            k_dim: 64,
            out_block: 64,
            num_warps: 8,
            kv_page_size: 64,
            col_batch: 1,
        }
    }

    pub fn rows_per_warp(&self) -> usize {
        self.cta_rows / self.num_warps
    }

    pub fn attn_passes(&self) -> usize {
        self.cta_rows / 16
    }
}
