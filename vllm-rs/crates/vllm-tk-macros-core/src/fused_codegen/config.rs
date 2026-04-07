// SPDX-License-Identifier: Apache-2.0
//! Configuration for fused prefill and decode kernels.

// ── Newtypes for type-safe shmem/GEMM arithmetic ──────────────────────

/// Byte offset into shared memory. `ShmemOffset + ByteSize = ShmemOffset`, but
/// `ShmemOffset + ShmemOffset` is a compile error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ShmemOffset(pub usize);

/// Size in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ByteSize(pub usize);

/// Number of output column tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileCount(pub usize);

/// Number of K-loop iterations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KIters(pub usize);

// Arithmetic: ShmemOffset + ByteSize = ShmemOffset
impl std::ops::Add<ByteSize> for ShmemOffset {
    type Output = ShmemOffset;
    fn add(self, rhs: ByteSize) -> ShmemOffset {
        ShmemOffset(self.0 + rhs.0)
    }
}

// ByteSize + ByteSize = ByteSize
impl std::ops::Add for ByteSize {
    type Output = ByteSize;
    fn add(self, rhs: ByteSize) -> ByteSize {
        ByteSize(self.0 + rhs.0)
    }
}

// ByteSize * usize = ByteSize (for `num_stages * b_size`)
impl std::ops::Mul<usize> for ByteSize {
    type Output = ByteSize;
    fn mul(self, rhs: usize) -> ByteSize {
        ByteSize(self.0 * rhs)
    }
}

impl std::ops::Mul<ByteSize> for usize {
    type Output = ByteSize;
    fn mul(self, rhs: ByteSize) -> ByteSize {
        ByteSize(self * rhs.0)
    }
}

// Comparison helpers for shmem budget checks
impl ByteSize {
    pub fn max(self, other: ByteSize) -> ByteSize {
        ByteSize(self.0.max(other.0))
    }
}

// ── Storage strategy (polyalgorithmic choice per DAG buffer) ──────────

/// How an activation buffer is stored between producer and consumer phases.
///
/// The DAG emitter assigns a `StorageStrategy` to each inter-op buffer based on
/// the shmem budget, buffer size, and consumer count. "Fusion" is an emergent
/// property: adjacent ops sharing shmem or registers get optimized by nvcc+ptxas.
#[derive(Clone, Debug, PartialEq)]
pub enum StorageStrategy {
    /// Lives in shared memory at a fixed offset. Persists until overwritten.
    Shmem { offset: ShmemOffset, size: ByteSize },
    /// Lives in registers (only valid for single-consumer, small buffers).
    Register,
    /// Lives in global memory (unavoidable for KV cache, large intermediates).
    Global { accessor: String },
    /// Output is computed on-the-fly by the consumer (e.g., norm fused into GEMM A-load).
    /// Only valid when there's exactly one consumer and the transform is element-wise.
    FusedIntoConsumer,
}

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
    /// Number of pipeline stages for GEMM K-loop (2 = double-buffer, 3 = triple-buffer).
    /// More stages hide memory latency but cost more shmem.
    pub num_stages: usize,
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
            num_stages: 2,
        }
    }

    /// 16-row config with 4-way col-distributed GEMMs.
    pub fn rows16_col4() -> Self {
        Self {
            cta_rows: 16,
            gemm_mode: GemmMode::Redundant,
            k_dim: 64,
            out_block: 64,
            num_warps: 8,
            kv_page_size: 64,
            col_batch: 4,
            num_stages: 2,
        }
    }

    /// 16-row config with 8-way col-distributed GEMMs.
    pub fn rows16_col8() -> Self {
        Self {
            cta_rows: 16,
            gemm_mode: GemmMode::Redundant,
            k_dim: 64,
            out_block: 64,
            num_warps: 8,
            kv_page_size: 64,
            col_batch: 8,
            num_stages: 2,
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
            num_stages: 2,
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
            num_stages: 2,
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
            num_stages: 2,
        }
    }

    /// 128-row cooperative GEMM with wider output tiles (out_block=128).
    /// Halves col_tiles (64→32 for ID GEMMs), doubles output per K-loop pass.
    /// stage = 8×4096 + 16384 = 49152, gemm = 2×49152 = 98304 = 96KB. Fits!
    pub fn rows128_wide() -> Self {
        Self {
            cta_rows: 128,
            gemm_mode: GemmMode::Cooperative,
            k_dim: 64,
            out_block: 128,
            num_warps: 8,
            kv_page_size: 64,
            col_batch: 1,
            num_stages: 2,
        }
    }

    /// 64-row cooperative GEMM with k_dim=128.
    /// Halves K-loop iterations (32→16 for HD GEMMs, 64→32 for ID).
    /// 4 warps × 16 rows = 64 rows.
    /// stage = 4×8192 + 16384 = 49152, gemm = 2×49152 = 98304 = 96KB. Fits!
    pub fn rows64_k128() -> Self {
        Self {
            cta_rows: 64,
            gemm_mode: GemmMode::Cooperative,
            k_dim: 128,
            out_block: 64,
            num_warps: 4,
            kv_page_size: 64,
            col_batch: 1,
            num_stages: 2,
        }
    }

    /// 64-row cooperative GEMM with 3-stage pipeline.
    /// 4 warps × 16 rows = 64 rows. 3 stages hide L2 latency (~200 cycles).
    /// stage = 4×4096 + 8192 = 24576, gemm = 3×24576 = 73728 = 72KB. Fits!
    pub fn rows64_3stage() -> Self {
        Self {
            cta_rows: 64,
            gemm_mode: GemmMode::Cooperative,
            k_dim: 64,
            out_block: 64,
            num_warps: 4,
            kv_page_size: 64,
            col_batch: 1,
            num_stages: 3,
        }
    }

    pub fn rows_per_warp(&self) -> usize {
        self.cta_rows / self.num_warps
    }

    pub fn attn_passes(&self) -> usize {
        self.cta_rows / 16
    }
}

// ── Decode configuration ──────────────────────────────────────────────

/// Configuration for a fused decode kernel variant.
///
/// Row-fused architecture: each CTA owns `cta_rows` sequences and runs them
/// through ALL phases of ALL layers with no cross-CTA barriers.  Activations
/// stay in shmem between phases; only weights and KV cache touch global memory.
#[derive(Clone, Debug)]
pub struct FusedDecodeConfig {
    /// Sequences (rows) owned by each CTA.
    pub cta_rows: usize,
    /// GEMM K-dimension tile size.
    pub k_dim: usize,
    /// GEMM output column tile size.
    pub out_block: usize,
    /// Number of warps per CTA.
    pub num_warps: usize,
    /// KV cache page size (tokens per page, must match cache allocation).
    pub kv_page_size: usize,
    /// Number of pipeline stages for weight loads (2 = double-buffer).
    pub num_stages: usize,
}

/// Maximum shared memory per CTA on sm89 (in bytes).
pub const SM89_MAX_SHMEM: usize = 101_376; // 99 KB

impl FusedDecodeConfig {
    /// Primary config: 16 rows per CTA — optimal for BS >= 16.
    ///
    /// 8 warps, each owns 2 rows in MMA tiles.  Cooperative GEMM with shared B.
    /// Shmem peak ~82KB (attention phase: Q[16,HD] + K_page + V_page).
    pub fn decode_rows16() -> Self {
        Self {
            cta_rows: 16,
            k_dim: 64,
            out_block: 64,
            num_warps: 8,
            kv_page_size: 64,
            num_stages: 2,
        }
    }

    /// Small-batch config: 4 rows per CTA — for BS in [2, 15].
    pub fn decode_rows4() -> Self {
        Self {
            cta_rows: 4,
            k_dim: 64,
            out_block: 64,
            num_warps: 8,
            kv_page_size: 64,
            num_stages: 2,
        }
    }

    /// Single-row config: BS = 1.
    pub fn decode_rows1() -> Self {
        Self {
            cta_rows: 1,
            k_dim: 64,
            out_block: 64,
            num_warps: 8,
            kv_page_size: 64,
            num_stages: 2,
        }
    }

    /// Padded CTA rows (rounded up to 16 for MMA tile alignment).
    pub fn padded_cta_rows(&self) -> usize {
        (self.cta_rows + 15) & !15
    }
}
