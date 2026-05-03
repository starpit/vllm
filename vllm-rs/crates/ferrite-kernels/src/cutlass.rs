// SPDX-License-Identifier: Apache-2.0
//! CUTLASS standalone GEMM kernel FFI.
//!
//! Each `cutlass_gemm_<TB_M>x<TB_N>_s<STAGES>_launch` is a specialised
//! CUTLASS GEMM kernel compiled from
//! `vllm-cuda/csrc/cutlass_standalone_gemm.cu`. They all share the
//! same signature:
//!
//! ```text
//! int cutlass_gemm_..._launch(
//!     uint16* C, const uint16* A, const uint16* B,
//!     int M, int N, int K,
//!     float alpha, float beta, uint64_t stream,
//! );
//! ```
//!
//! where `A` is `[M, K]` row-major, `B` is `[N, K]` row-major
//! (cuBLAS-style weight layout — B^T is the mathematical operand),
//! `C` is `[M, N]`, and the computation is `C = alpha * A @ B^T +
//! beta * C` in BF16 / FP16.
//!
//! `cutlass_gemv_launch` is the M=1 specialisation (SIMT GEMV) with
//! the same prototype.
//!
//! These symbols are defined by `vllm-cuda`'s C++ build; the final
//! binary (`vllm-cli`) links both crates so the ferrite-forward-
//! emitted code in `ferrite-models` can reach them through here.
//!
//! ferrite-forward's solver picks a tile variant per Gemm tile ×
//! workload bucket from the calibrated `target_profiles/cost_*.csv`
//! tables; the macro-emitted forward calls the corresponding launch
//! fn directly — no runtime tile-variant dispatch.

/// The cutlass tile size + pipeline stages — the `(tile_m, tile_n,
/// stages)` triple uniquely identifies one kernel variant in the
/// library. Use [`CutlassTile::launch`] to invoke.
///
/// `variant` distinguishes shape-equivalent kernel families that
/// share `(tile_m, tile_n, stages)` but differ in threadblock
/// swizzle, TB_K, or warp count. `Basic` is the default
/// `GemmIdentityThreadblockSwizzle<>` + TB_K=32 + 4-warp config the
/// cuBLAS-replacement zoo started with; `Sw` swaps in
/// `GemmIdentityThreadblockSwizzle<8>` for L2-locality at large
/// grids; future variants (`K64`, `W8`, `K64W8`) plug in here when
/// their bindings land.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CutlassTile {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
    pub variant: GemmVariant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GemmVariant {
    Basic,
    /// `GemmIdentityThreadblockSwizzle<8>` — boosts L2 locality at
    /// large-M shapes where the default identity swizzle re-fetches
    /// activation rows past L2 capacity.
    Sw,
    /// `OpClassWmmaTensorOp` + `nvcuda::wmma::mma_sync` 16×16×16 inst
    /// shape — matches cuBLAS's small-M `cutlass_80_wmma_tensorop_*`
    /// kernels that beat plain `mma 16×8×16` at decode shapes
    /// (TB 16×16, 32×32 with K=64/128). Requires the bf16 Wmma<>
    /// spec from `cutlass_wmma_bf16.h` (vendored CUTLASS lacks it).
    Wmma,
}

impl CutlassTile {
    pub const fn new(tile_m: u32, tile_n: u32, stages: u32) -> Self {
        Self {
            tile_m,
            tile_n,
            stages,
            variant: GemmVariant::Basic,
        }
    }

    pub const fn with_variant(tile_m: u32, tile_n: u32, stages: u32, variant: GemmVariant) -> Self {
        Self {
            tile_m,
            tile_n,
            stages,
            variant,
        }
    }

    /// The kernel name used in `target_profiles/cost_*.csv`.
    /// Matches the CSV column value byte-for-byte.
    pub fn csv_name(self) -> String {
        match self.variant {
            GemmVariant::Basic => {
                format!("cutlass_{}x{}_s{}", self.tile_m, self.tile_n, self.stages)
            }
            GemmVariant::Sw => format!(
                "cutlass_{}x{}_sw_s{}",
                self.tile_m, self.tile_n, self.stages
            ),
            GemmVariant::Wmma => format!(
                "cutlass_{}x{}_wmma_s{}",
                self.tile_m, self.tile_n, self.stages
            ),
        }
    }
}

/// Default tile picked when a caller doesn't have shape-specific
/// cost data (the post-cuBLAS replacement for `cublas.gemm`'s
/// shape-agnostic dispatch). 16×128 stage 3 supports M down to 1
/// without wasting too much on padding; AlignmentB = 8 still
/// applies, so callers must ensure N % 8 == 0 and K % 8 == 0
/// (loader-side weight padding for arches whose vocab is not a
/// multiple of 8 — e.g. granite-3.3-2B vocab=49159).
pub const DEFAULT_GEMM_TILE: CutlassTile = CutlassTile {
    tile_m: 16,
    tile_n: 128,
    stages: 3,
    variant: GemmVariant::Basic,
};

// Every tile variant compiled by `cutlass_standalone_gemm.cu` that
// is covered by the calibrated CSV tables. When the .cu file gains
// a variant, regenerate the CSV via `gpu_cost_sweep` and add the
// entry here.
#[cfg(feature = "cuda")]
unsafe extern "C" {
    pub fn cutlass_gemm_16x64_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_16x64_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_16x128_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_16x128_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_32x64_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_32x64_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_32x128_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_32x128_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_32x256_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x64_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x64_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x128_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x128_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x256_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_256x64_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_256x64_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;

    // ── Deep-stage / stages-2 variants of the basic tile zoo ──
    //
    // Same kernel template family as the basic tiles above, just
    // different `stages` parameter — cuBLAS routinely picks
    // 5..8-stage variants on sm89 (per nsys NVTX traces against
    // cuBLAS-on at qwen2.5-3b prefill). All `_make_op` / `_run_op` /
    // `_drop_op` shims come from the same `CUTLASS_GEMM_LAUNCH`
    // macro so the `cutlass_gemm_cached` 2-phase path Just Works
    // for these tuples. `cutlass_gemm_256x128_s2` is the only
    // 256x128 variant that fits sm89's 99KB SMEM at the
    // (warp_M=64, warp_N=32) shape used in csrc.
    pub fn cutlass_gemm_64x64_s2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x64_s5_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x64_s6_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x64_s8_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x64_s10_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s5_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s6_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s7_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s8_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s5_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s6_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s7_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s8_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x128_s2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x128_s5_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x128_s6_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x256_s2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x256_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x256_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x256_s5_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x256_s2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x256_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_256x64_s2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_256x64_s5_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_256x64_s6_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_256x128_s2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;

    // ── Swizzle (`_sw_`) variants ──
    //
    // `cutlass_gemm_<WxH>_sw_s<STAGES>_launch`: same kernel template
    // as the unsuffixed tile, but uses `GemmIdentityThreadblockSwizzle<8>`
    // instead of the default `GemmIdentityThreadblockSwizzle<>`. The 8-wide
    // swizzle reorders threadblock launches in the N-tile dimension to
    // boost L2 locality at large grids — cuBLAS's documented choice for
    // `128×128 at M=1024+` (csrc note flags this as the "worst remaining
    // gap" against cuBLAS). Same launch ABI as the unsuffixed family.
    pub fn cutlass_gemm_64x128_sw_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_sw_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x256_sw_s2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x256_sw_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x128_sw_s2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x128_sw_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x128_sw_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x256_sw_s2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x256_sw_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_256x64_sw_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_256x64_sw_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;

    // ── Wmma bf16 variants (cuBLAS regime-map matches) ──
    //
    // `cutlass_gemm_wmma_<TBxTB>_k<TBK>_s<S>_launch` — uses
    // `OpClassWmmaTensorOp` + `nvcuda::wmma 16×16×16` instruction. Adds
    // bf16 wmma support via `csrc/cutlass_wmma_bf16.h` (vendored CUTLASS
    // ships only f16 + int4 wmma). Threadblock shapes match cuBLAS's
    // observed picks: 16×16, 32×32 with K∈{64,128} stages=2.
    pub fn cutlass_gemm_wmma_16x16_k128_s2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_wmma_32x32_k128_s2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_wmma_32x32_k64_s2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;

    // ── SplitK parallel variants ──
    //
    // `cutlass_gemm_<WxH>_s<STAGES>_sk<SLICES>_launch`: same signature
    // as the standard GEMM launchers plus a caller-owned `workspace`
    // pointer. The kernel splits the K dim across `SLICES` CTAs and
    // reduces via a separate reduction kernel.
    //
    // Workspace contract (GemmSplitKParallel): f32 scratch sized
    // `SLICES × M × N × 4` bytes. The Rust safe wrapper allocates this
    // via the ferrite CachingAllocator.
    pub fn cutlass_gemm_64x64_s4_sk2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x64_s4_sk4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x64_s4_sk8_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s4_sk2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s4_sk4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s4_sk8_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s4_sk2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s4_sk4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s4_sk8_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x128_s4_sk2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x128_s4_sk4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x128_s4_sk8_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;

    // tile_m=16 splitK — small-M long-K regime (M=8 + K≥8192).
    pub fn cutlass_gemm_16x64_s4_sk2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_16x64_s4_sk4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_16x64_s4_sk8_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_16x128_s4_sk2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_16x128_s4_sk4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_16x128_s4_sk8_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;

    pub fn cutlass_gemv_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;

    // 2-phase GEMV API. `cutlass_gemv` only has one variant (no tile
    // zoo) so dispatch is trivial.
    pub fn cutlass_gemv_make_op(n: i32, k: i32, alpha: f32, beta: f32) -> *mut std::ffi::c_void;
    pub fn cutlass_gemv_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemv_drop_op(op: *mut std::ffi::c_void);

    /// CUTLASS GEMV with row-broadcast bias. Output: `D = W @ x + bias`.
    /// One launch — replaces `cublasGemvParamsEx`'s bias-fused path at
    /// qwen2's biased QKV (M=1, N=q+2*kv, K=hidden).
    pub fn cutlass_gemv_bias_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;

    /// Fused Gate GEMM + SiLU + Mul via CUTLASS 2.x EVT.
    /// Computes `D[M,N] = silu(A[M,K] @ B_gate[N,K]^T) * C_up[M,N]`,
    /// where `C_up` is the separately-computed up-projection output
    /// (aux-loaded from GMEM). Internal 3-way tile dispatch on `m`.
    pub fn cutlass_gemm_silu_mul_launch(
        d: *mut u16,
        a: *const u16,
        b_gate: *const u16,
        c_up: *mut u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;

    // ── Bias-add GEMM zoo ──
    //
    // `cutlass_gemm_bias_<TB_M>x<TB_N>_s<STAGES>_launch`. Computes
    // `D[M,N] = A[M,K] @ B[N,K]^T + bias[N]` in one launch, where
    // `bias` is `[N]` bf16 broadcast across rows.
    //
    // Backed by `cutlass::gemm::device::Gemm` (same template family
    // as the standalone tile zoo) with `LinearCombination` epilogue +
    // ldc=0 broadcast — the bias rides as the C operand at stride 0.
    // Tile zoo mirrors `CUTLASS_TILE_ZOO` in
    // ferrite-forward-macro/src/impl_lib.rs.
    pub fn cutlass_gemm_bias_16x64_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_16x64_s4_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_16x128_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_16x128_s4_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_32x64_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_32x64_s4_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_32x128_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_32x128_s4_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_32x256_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_64x64_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_64x64_s4_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_64x128_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_64x128_s4_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_128x64_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_128x64_s4_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_128x128_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_128x128_s4_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_128x256_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_256x64_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_256x64_s4_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;

    // 2-phase API for `cutlass_gemm_bias` family. Same shape as the
    // `cutlass_gemm_*_make_op/_run_op/_drop_op` trio above. See the
    // `cutlass_standalone_gemm.cu` design comment.
    pub fn cutlass_gemm_bias_16x64_s3_make_op(M: i32, N: i32, K: i32) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_bias_16x64_s3_run_op(
        op: *mut std::ffi::c_void,
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_16x64_s3_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_bias_16x64_s4_make_op(M: i32, N: i32, K: i32) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_bias_16x64_s4_run_op(
        op: *mut std::ffi::c_void,
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_16x64_s4_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_bias_16x128_s3_make_op(M: i32, N: i32, K: i32) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_bias_16x128_s3_run_op(
        op: *mut std::ffi::c_void,
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_16x128_s3_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_bias_16x128_s4_make_op(M: i32, N: i32, K: i32) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_bias_16x128_s4_run_op(
        op: *mut std::ffi::c_void,
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_16x128_s4_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_bias_32x64_s3_make_op(M: i32, N: i32, K: i32) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_bias_32x64_s3_run_op(
        op: *mut std::ffi::c_void,
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_32x64_s3_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_bias_32x64_s4_make_op(M: i32, N: i32, K: i32) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_bias_32x64_s4_run_op(
        op: *mut std::ffi::c_void,
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_32x64_s4_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_bias_32x128_s3_make_op(M: i32, N: i32, K: i32) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_bias_32x128_s3_run_op(
        op: *mut std::ffi::c_void,
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_32x128_s3_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_bias_32x128_s4_make_op(M: i32, N: i32, K: i32) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_bias_32x128_s4_run_op(
        op: *mut std::ffi::c_void,
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_32x128_s4_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_bias_32x256_s3_make_op(M: i32, N: i32, K: i32) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_bias_32x256_s3_run_op(
        op: *mut std::ffi::c_void,
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_32x256_s3_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_bias_64x64_s3_make_op(M: i32, N: i32, K: i32) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_bias_64x64_s3_run_op(
        op: *mut std::ffi::c_void,
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_64x64_s3_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_bias_64x64_s4_make_op(M: i32, N: i32, K: i32) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_bias_64x64_s4_run_op(
        op: *mut std::ffi::c_void,
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_64x64_s4_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_bias_64x128_s3_make_op(M: i32, N: i32, K: i32) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_bias_64x128_s3_run_op(
        op: *mut std::ffi::c_void,
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_64x128_s3_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_bias_64x128_s4_make_op(M: i32, N: i32, K: i32) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_bias_64x128_s4_run_op(
        op: *mut std::ffi::c_void,
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_64x128_s4_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_bias_128x64_s3_make_op(M: i32, N: i32, K: i32) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_bias_128x64_s3_run_op(
        op: *mut std::ffi::c_void,
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_128x64_s3_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_bias_128x64_s4_make_op(M: i32, N: i32, K: i32) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_bias_128x64_s4_run_op(
        op: *mut std::ffi::c_void,
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_128x64_s4_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_bias_128x128_s3_make_op(M: i32, N: i32, K: i32) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_bias_128x128_s3_run_op(
        op: *mut std::ffi::c_void,
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_128x128_s3_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_bias_128x128_s4_make_op(M: i32, N: i32, K: i32) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_bias_128x128_s4_run_op(
        op: *mut std::ffi::c_void,
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_128x128_s4_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_bias_128x256_s3_make_op(M: i32, N: i32, K: i32) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_bias_128x256_s3_run_op(
        op: *mut std::ffi::c_void,
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_128x256_s3_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_bias_256x64_s3_make_op(M: i32, N: i32, K: i32) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_bias_256x64_s3_run_op(
        op: *mut std::ffi::c_void,
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_256x64_s3_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_bias_256x64_s4_make_op(M: i32, N: i32, K: i32) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_bias_256x64_s4_run_op(
        op: *mut std::ffi::c_void,
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_256x64_s4_drop_op(op: *mut std::ffi::c_void);
}

type CutlassLaunchFn =
    unsafe extern "C" fn(*mut u16, *const u16, *const u16, i32, i32, i32, f32, f32, u64) -> i32;

/// Resolve the `(tile_m, tile_n, stages, variant)` tuple to its
/// specialized launch fn. Panics if the tuple isn't in the zoo.
#[cfg(feature = "cuda")]
fn launch_fn_for(tile: CutlassTile) -> CutlassLaunchFn {
    match (tile.variant, tile.tile_m, tile.tile_n, tile.stages) {
        (GemmVariant::Basic, 16, 64, 3) => cutlass_gemm_16x64_s3_launch,
        (GemmVariant::Basic, 16, 64, 4) => cutlass_gemm_16x64_s4_launch,
        (GemmVariant::Basic, 16, 128, 3) => cutlass_gemm_16x128_s3_launch,
        (GemmVariant::Basic, 16, 128, 4) => cutlass_gemm_16x128_s4_launch,
        (GemmVariant::Basic, 32, 64, 3) => cutlass_gemm_32x64_s3_launch,
        (GemmVariant::Basic, 32, 64, 4) => cutlass_gemm_32x64_s4_launch,
        (GemmVariant::Basic, 32, 128, 3) => cutlass_gemm_32x128_s3_launch,
        (GemmVariant::Basic, 32, 128, 4) => cutlass_gemm_32x128_s4_launch,
        (GemmVariant::Basic, 32, 256, 3) => cutlass_gemm_32x256_s3_launch,
        (GemmVariant::Basic, 64, 64, 3) => cutlass_gemm_64x64_s3_launch,
        (GemmVariant::Basic, 64, 64, 4) => cutlass_gemm_64x64_s4_launch,
        (GemmVariant::Basic, 64, 128, 3) => cutlass_gemm_64x128_s3_launch,
        (GemmVariant::Basic, 64, 128, 4) => cutlass_gemm_64x128_s4_launch,
        (GemmVariant::Basic, 128, 64, 3) => cutlass_gemm_128x64_s3_launch,
        (GemmVariant::Basic, 128, 64, 4) => cutlass_gemm_128x64_s4_launch,
        (GemmVariant::Basic, 128, 128, 3) => cutlass_gemm_128x128_s3_launch,
        (GemmVariant::Basic, 128, 128, 4) => cutlass_gemm_128x128_s4_launch,
        (GemmVariant::Basic, 128, 256, 3) => cutlass_gemm_128x256_s3_launch,
        (GemmVariant::Basic, 256, 64, 3) => cutlass_gemm_256x64_s3_launch,
        (GemmVariant::Basic, 256, 64, 4) => cutlass_gemm_256x64_s4_launch,
        (GemmVariant::Sw, 64, 128, 3) => cutlass_gemm_64x128_sw_s3_launch,
        (GemmVariant::Sw, 64, 128, 4) => cutlass_gemm_64x128_sw_s4_launch,
        (GemmVariant::Sw, 64, 256, 2) => cutlass_gemm_64x256_sw_s2_launch,
        (GemmVariant::Sw, 64, 256, 3) => cutlass_gemm_64x256_sw_s3_launch,
        (GemmVariant::Sw, 128, 128, 2) => cutlass_gemm_128x128_sw_s2_launch,
        (GemmVariant::Sw, 128, 128, 3) => cutlass_gemm_128x128_sw_s3_launch,
        (GemmVariant::Sw, 128, 128, 4) => cutlass_gemm_128x128_sw_s4_launch,
        (GemmVariant::Sw, 128, 256, 2) => cutlass_gemm_128x256_sw_s2_launch,
        (GemmVariant::Sw, 128, 256, 3) => cutlass_gemm_128x256_sw_s3_launch,
        (GemmVariant::Sw, 256, 64, 3) => cutlass_gemm_256x64_sw_s3_launch,
        (GemmVariant::Sw, 256, 64, 4) => cutlass_gemm_256x64_sw_s4_launch,
        // Deep-stage / stages-2 basic-tile variants — match cuBLAS's
        // mid-M picks (`stages_64x3`, `_32x6`, `_64x5`, etc.).
        (GemmVariant::Basic, 64, 64, 2) => cutlass_gemm_64x64_s2_launch,
        (GemmVariant::Basic, 64, 64, 5) => cutlass_gemm_64x64_s5_launch,
        (GemmVariant::Basic, 64, 64, 6) => cutlass_gemm_64x64_s6_launch,
        (GemmVariant::Basic, 64, 64, 8) => cutlass_gemm_64x64_s8_launch,
        (GemmVariant::Basic, 64, 64, 10) => cutlass_gemm_64x64_s10_launch,
        (GemmVariant::Basic, 64, 128, 2) => cutlass_gemm_64x128_s2_launch,
        (GemmVariant::Basic, 64, 128, 5) => cutlass_gemm_64x128_s5_launch,
        (GemmVariant::Basic, 64, 128, 6) => cutlass_gemm_64x128_s6_launch,
        (GemmVariant::Basic, 64, 128, 7) => cutlass_gemm_64x128_s7_launch,
        (GemmVariant::Basic, 64, 128, 8) => cutlass_gemm_64x128_s8_launch,
        (GemmVariant::Basic, 128, 64, 2) => cutlass_gemm_128x64_s2_launch,
        (GemmVariant::Basic, 128, 64, 5) => cutlass_gemm_128x64_s5_launch,
        (GemmVariant::Basic, 128, 64, 6) => cutlass_gemm_128x64_s6_launch,
        (GemmVariant::Basic, 128, 64, 7) => cutlass_gemm_128x64_s7_launch,
        (GemmVariant::Basic, 128, 64, 8) => cutlass_gemm_128x64_s8_launch,
        (GemmVariant::Basic, 128, 128, 2) => cutlass_gemm_128x128_s2_launch,
        (GemmVariant::Basic, 128, 128, 5) => cutlass_gemm_128x128_s5_launch,
        (GemmVariant::Basic, 128, 128, 6) => cutlass_gemm_128x128_s6_launch,
        (GemmVariant::Basic, 64, 256, 2) => cutlass_gemm_64x256_s2_launch,
        (GemmVariant::Basic, 64, 256, 3) => cutlass_gemm_64x256_s3_launch,
        (GemmVariant::Basic, 64, 256, 4) => cutlass_gemm_64x256_s4_launch,
        (GemmVariant::Basic, 64, 256, 5) => cutlass_gemm_64x256_s5_launch,
        (GemmVariant::Basic, 128, 256, 2) => cutlass_gemm_128x256_s2_launch,
        (GemmVariant::Basic, 128, 256, 4) => cutlass_gemm_128x256_s4_launch,
        (GemmVariant::Basic, 256, 64, 2) => cutlass_gemm_256x64_s2_launch,
        (GemmVariant::Basic, 256, 64, 5) => cutlass_gemm_256x64_s5_launch,
        (GemmVariant::Basic, 256, 64, 6) => cutlass_gemm_256x64_s6_launch,
        (GemmVariant::Basic, 256, 128, 2) => cutlass_gemm_256x128_s2_launch,
        // Wmma bf16 variants — match cuBLAS's small-M wmma picks. The
        // `cutlass_<TM>x<TN>_wmma_s<S>` CSV name maps to one of the
        // `cutlass_gemm_wmma_<TM>x<TN>_k<TBK>_s<S>_launch` symbols.
        // Only stages=2 is exposed today; cuBLAS doesn't pick higher
        // stages for these tiles in practice.
        (GemmVariant::Wmma, 16, 16, 2) => cutlass_gemm_wmma_16x16_k128_s2_launch,
        (GemmVariant::Wmma, 32, 32, 2) => cutlass_gemm_wmma_32x32_k128_s2_launch,
        other => panic!(
            "cutlass: unsupported tile {:?} — add its extern declaration + csv entry",
            other,
        ),
    }
}

// ── 2-phase API for the cutlass_gemm tile zoo ──────────────────────
//
// Each `cutlass_gemm_*_launch` extern above is the legacy monolithic
// path: every call rebuilds Arguments, runs `can_implement`,
// `initialize` (which derives Params: grid swizzle, problem-size
// tables, etc.), then launches the kernel. With ~300 launches per
// decode forward, that's ~1.5 ms of pure host overhead per token in
// eager mode (CUDA graphs amortize it; eager pays per call).
//
// The 2-phase API splits that into:
//   1. `_make_op(M, N, K, α, β) -> *mut Op` — done once per shape.
//      Heap-allocates an Op handle with `initialize()` already done.
//   2. `_run_op(op, C, A, B, M, N, K, α, β, stream)` — done per call.
//      Uses CUTLASS's `update()` to patch operand pointers into the
//      cached Params, then `run()` launches. No grid recomputation.
//   3. `_drop_op(op)` — frees the cached handle.
//
// `cutlass_gemm_cached` below uses a global `Mutex<HashMap>` keyed on
// (tile, M, N, K) to memoize the make_op step. A future pass moves
// the cache into the codegen-emitted dispatch table (per-Instruction
// `params_idx` → `OnceCell<Op*>`) so per-call lookup is branch-free.
#[cfg(feature = "cuda")]
unsafe extern "C" {
    pub fn cutlass_gemm_16x64_s3_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_16x64_s3_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_16x64_s3_drop_op(op: *mut std::ffi::c_void);

    pub fn cutlass_gemm_16x64_s4_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_16x64_s4_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_16x64_s4_drop_op(op: *mut std::ffi::c_void);

    pub fn cutlass_gemm_16x128_s3_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_16x128_s3_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_16x128_s3_drop_op(op: *mut std::ffi::c_void);

    pub fn cutlass_gemm_16x128_s4_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_16x128_s4_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_16x128_s4_drop_op(op: *mut std::ffi::c_void);

    pub fn cutlass_gemm_32x64_s3_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_32x64_s3_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_32x64_s3_drop_op(op: *mut std::ffi::c_void);

    pub fn cutlass_gemm_32x64_s4_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_32x64_s4_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_32x64_s4_drop_op(op: *mut std::ffi::c_void);

    pub fn cutlass_gemm_32x128_s3_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_32x128_s3_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_32x128_s3_drop_op(op: *mut std::ffi::c_void);

    pub fn cutlass_gemm_32x128_s4_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_32x128_s4_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_32x128_s4_drop_op(op: *mut std::ffi::c_void);

    pub fn cutlass_gemm_32x256_s3_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_32x256_s3_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_32x256_s3_drop_op(op: *mut std::ffi::c_void);

    pub fn cutlass_gemm_64x64_s3_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_64x64_s3_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x64_s3_drop_op(op: *mut std::ffi::c_void);

    pub fn cutlass_gemm_64x64_s4_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_64x64_s4_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x64_s4_drop_op(op: *mut std::ffi::c_void);

    pub fn cutlass_gemm_64x128_s3_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_64x128_s3_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s3_drop_op(op: *mut std::ffi::c_void);

    pub fn cutlass_gemm_64x128_s4_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_64x128_s4_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s4_drop_op(op: *mut std::ffi::c_void);

    pub fn cutlass_gemm_128x64_s3_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_128x64_s3_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s3_drop_op(op: *mut std::ffi::c_void);

    pub fn cutlass_gemm_128x64_s4_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_128x64_s4_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s4_drop_op(op: *mut std::ffi::c_void);

    pub fn cutlass_gemm_128x128_s3_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_128x128_s3_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x128_s3_drop_op(op: *mut std::ffi::c_void);

    pub fn cutlass_gemm_128x128_s4_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_128x128_s4_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x128_s4_drop_op(op: *mut std::ffi::c_void);

    pub fn cutlass_gemm_128x256_s3_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_128x256_s3_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x256_s3_drop_op(op: *mut std::ffi::c_void);

    pub fn cutlass_gemm_256x64_s3_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_256x64_s3_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_256x64_s3_drop_op(op: *mut std::ffi::c_void);

    pub fn cutlass_gemm_256x64_s4_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_256x64_s4_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_256x64_s4_drop_op(op: *mut std::ffi::c_void);

    // 2-phase API for the deep-stage / stages-2 basic-tile variants
    // declared above. Same triple of `_make_op` / `_run_op` /
    // `_drop_op` shims the standard CUTLASS_GEMM_LAUNCH macro emits;
    // `cutlass_gemm_cached` uses these for the same plan-caching
    // benefit it gives the original 20-tile zoo.
    pub fn cutlass_gemm_64x64_s2_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_64x64_s2_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x64_s2_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_64x64_s5_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_64x64_s5_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x64_s5_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_64x64_s6_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_64x64_s6_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x64_s6_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_64x64_s8_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_64x64_s8_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x64_s8_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_64x64_s10_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_64x64_s10_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x64_s10_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_64x128_s2_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_64x128_s2_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s2_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_64x128_s5_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_64x128_s5_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s5_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_64x128_s6_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_64x128_s6_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s6_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_64x128_s7_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_64x128_s7_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s7_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_64x128_s8_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_64x128_s8_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s8_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_128x64_s2_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_128x64_s2_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s2_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_128x64_s5_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_128x64_s5_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s5_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_128x64_s6_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_128x64_s6_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s6_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_128x64_s7_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_128x64_s7_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s7_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_128x64_s8_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_128x64_s8_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s8_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_128x128_s2_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_128x128_s2_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x128_s2_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_128x128_s5_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_128x128_s5_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x128_s5_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_128x128_s6_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_128x128_s6_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x128_s6_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_64x256_s2_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_64x256_s2_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x256_s2_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_64x256_s3_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_64x256_s3_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x256_s3_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_64x256_s4_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_64x256_s4_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x256_s4_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_64x256_s5_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_64x256_s5_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x256_s5_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_128x256_s2_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_128x256_s2_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x256_s2_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_128x256_s4_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_128x256_s4_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x256_s4_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_256x64_s2_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_256x64_s2_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_256x64_s2_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_256x64_s5_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_256x64_s5_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_256x64_s5_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_256x64_s6_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_256x64_s6_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_256x64_s6_drop_op(op: *mut std::ffi::c_void);
    pub fn cutlass_gemm_256x128_s2_make_op(
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        beta: f32,
    ) -> *mut std::ffi::c_void;
    pub fn cutlass_gemm_256x128_s2_run_op(
        op: *mut std::ffi::c_void,
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_256x128_s2_drop_op(op: *mut std::ffi::c_void);
}

#[cfg(feature = "cuda")]
type CutlassMakeOpFn = unsafe extern "C" fn(i32, i32, i32, f32, f32) -> *mut std::ffi::c_void;

#[cfg(feature = "cuda")]
type CutlassRunOpFn = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    *mut u16,
    *const u16,
    *const u16,
    i32,
    i32,
    i32,
    f32,
    f32,
    u64,
) -> i32;

#[cfg(feature = "cuda")]
type CutlassDropOpFn = unsafe extern "C" fn(*mut std::ffi::c_void);

#[cfg(feature = "cuda")]
fn make_op_fn_for(tile: CutlassTile) -> CutlassMakeOpFn {
    match (tile.tile_m, tile.tile_n, tile.stages) {
        (16, 64, 3) => cutlass_gemm_16x64_s3_make_op,
        (16, 64, 4) => cutlass_gemm_16x64_s4_make_op,
        (16, 128, 3) => cutlass_gemm_16x128_s3_make_op,
        (16, 128, 4) => cutlass_gemm_16x128_s4_make_op,
        (32, 64, 3) => cutlass_gemm_32x64_s3_make_op,
        (32, 64, 4) => cutlass_gemm_32x64_s4_make_op,
        (32, 128, 3) => cutlass_gemm_32x128_s3_make_op,
        (32, 128, 4) => cutlass_gemm_32x128_s4_make_op,
        (32, 256, 3) => cutlass_gemm_32x256_s3_make_op,
        (64, 64, 3) => cutlass_gemm_64x64_s3_make_op,
        (64, 64, 4) => cutlass_gemm_64x64_s4_make_op,
        (64, 128, 3) => cutlass_gemm_64x128_s3_make_op,
        (64, 128, 4) => cutlass_gemm_64x128_s4_make_op,
        (128, 64, 3) => cutlass_gemm_128x64_s3_make_op,
        (128, 64, 4) => cutlass_gemm_128x64_s4_make_op,
        (128, 128, 3) => cutlass_gemm_128x128_s3_make_op,
        (128, 128, 4) => cutlass_gemm_128x128_s4_make_op,
        (128, 256, 3) => cutlass_gemm_128x256_s3_make_op,
        (256, 64, 3) => cutlass_gemm_256x64_s3_make_op,
        (256, 64, 4) => cutlass_gemm_256x64_s4_make_op,
        // Deep-stage / stages-2 basic-tile variants.
        (64, 64, 2) => cutlass_gemm_64x64_s2_make_op,
        (64, 64, 5) => cutlass_gemm_64x64_s5_make_op,
        (64, 64, 6) => cutlass_gemm_64x64_s6_make_op,
        (64, 64, 8) => cutlass_gemm_64x64_s8_make_op,
        (64, 64, 10) => cutlass_gemm_64x64_s10_make_op,
        (64, 128, 2) => cutlass_gemm_64x128_s2_make_op,
        (64, 128, 5) => cutlass_gemm_64x128_s5_make_op,
        (64, 128, 6) => cutlass_gemm_64x128_s6_make_op,
        (64, 128, 7) => cutlass_gemm_64x128_s7_make_op,
        (64, 128, 8) => cutlass_gemm_64x128_s8_make_op,
        (128, 64, 2) => cutlass_gemm_128x64_s2_make_op,
        (128, 64, 5) => cutlass_gemm_128x64_s5_make_op,
        (128, 64, 6) => cutlass_gemm_128x64_s6_make_op,
        (128, 64, 7) => cutlass_gemm_128x64_s7_make_op,
        (128, 64, 8) => cutlass_gemm_128x64_s8_make_op,
        (128, 128, 2) => cutlass_gemm_128x128_s2_make_op,
        (128, 128, 5) => cutlass_gemm_128x128_s5_make_op,
        (128, 128, 6) => cutlass_gemm_128x128_s6_make_op,
        (64, 256, 2) => cutlass_gemm_64x256_s2_make_op,
        (64, 256, 3) => cutlass_gemm_64x256_s3_make_op,
        (64, 256, 4) => cutlass_gemm_64x256_s4_make_op,
        (64, 256, 5) => cutlass_gemm_64x256_s5_make_op,
        (128, 256, 2) => cutlass_gemm_128x256_s2_make_op,
        (128, 256, 4) => cutlass_gemm_128x256_s4_make_op,
        (256, 64, 2) => cutlass_gemm_256x64_s2_make_op,
        (256, 64, 5) => cutlass_gemm_256x64_s5_make_op,
        (256, 64, 6) => cutlass_gemm_256x64_s6_make_op,
        (256, 128, 2) => cutlass_gemm_256x128_s2_make_op,
        other => panic!("cutlass: unsupported tile {:?}", other),
    }
}

#[cfg(feature = "cuda")]
fn run_op_fn_for(tile: CutlassTile) -> CutlassRunOpFn {
    match (tile.tile_m, tile.tile_n, tile.stages) {
        (16, 64, 3) => cutlass_gemm_16x64_s3_run_op,
        (16, 64, 4) => cutlass_gemm_16x64_s4_run_op,
        (16, 128, 3) => cutlass_gemm_16x128_s3_run_op,
        (16, 128, 4) => cutlass_gemm_16x128_s4_run_op,
        (32, 64, 3) => cutlass_gemm_32x64_s3_run_op,
        (32, 64, 4) => cutlass_gemm_32x64_s4_run_op,
        (32, 128, 3) => cutlass_gemm_32x128_s3_run_op,
        (32, 128, 4) => cutlass_gemm_32x128_s4_run_op,
        (32, 256, 3) => cutlass_gemm_32x256_s3_run_op,
        (64, 64, 3) => cutlass_gemm_64x64_s3_run_op,
        (64, 64, 4) => cutlass_gemm_64x64_s4_run_op,
        (64, 128, 3) => cutlass_gemm_64x128_s3_run_op,
        (64, 128, 4) => cutlass_gemm_64x128_s4_run_op,
        (128, 64, 3) => cutlass_gemm_128x64_s3_run_op,
        (128, 64, 4) => cutlass_gemm_128x64_s4_run_op,
        (128, 128, 3) => cutlass_gemm_128x128_s3_run_op,
        (128, 128, 4) => cutlass_gemm_128x128_s4_run_op,
        (128, 256, 3) => cutlass_gemm_128x256_s3_run_op,
        (256, 64, 3) => cutlass_gemm_256x64_s3_run_op,
        (256, 64, 4) => cutlass_gemm_256x64_s4_run_op,
        (64, 64, 2) => cutlass_gemm_64x64_s2_run_op,
        (64, 64, 5) => cutlass_gemm_64x64_s5_run_op,
        (64, 64, 6) => cutlass_gemm_64x64_s6_run_op,
        (64, 64, 8) => cutlass_gemm_64x64_s8_run_op,
        (64, 64, 10) => cutlass_gemm_64x64_s10_run_op,
        (64, 128, 2) => cutlass_gemm_64x128_s2_run_op,
        (64, 128, 5) => cutlass_gemm_64x128_s5_run_op,
        (64, 128, 6) => cutlass_gemm_64x128_s6_run_op,
        (64, 128, 7) => cutlass_gemm_64x128_s7_run_op,
        (64, 128, 8) => cutlass_gemm_64x128_s8_run_op,
        (128, 64, 2) => cutlass_gemm_128x64_s2_run_op,
        (128, 64, 5) => cutlass_gemm_128x64_s5_run_op,
        (128, 64, 6) => cutlass_gemm_128x64_s6_run_op,
        (128, 64, 7) => cutlass_gemm_128x64_s7_run_op,
        (128, 64, 8) => cutlass_gemm_128x64_s8_run_op,
        (128, 128, 2) => cutlass_gemm_128x128_s2_run_op,
        (128, 128, 5) => cutlass_gemm_128x128_s5_run_op,
        (128, 128, 6) => cutlass_gemm_128x128_s6_run_op,
        (64, 256, 2) => cutlass_gemm_64x256_s2_run_op,
        (64, 256, 3) => cutlass_gemm_64x256_s3_run_op,
        (64, 256, 4) => cutlass_gemm_64x256_s4_run_op,
        (64, 256, 5) => cutlass_gemm_64x256_s5_run_op,
        (128, 256, 2) => cutlass_gemm_128x256_s2_run_op,
        (128, 256, 4) => cutlass_gemm_128x256_s4_run_op,
        (256, 64, 2) => cutlass_gemm_256x64_s2_run_op,
        (256, 64, 5) => cutlass_gemm_256x64_s5_run_op,
        (256, 64, 6) => cutlass_gemm_256x64_s6_run_op,
        (256, 128, 2) => cutlass_gemm_256x128_s2_run_op,
        other => panic!("cutlass: unsupported tile {:?}", other),
    }
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn drop_op_fn_for(tile: CutlassTile) -> CutlassDropOpFn {
    match (tile.tile_m, tile.tile_n, tile.stages) {
        (16, 64, 3) => cutlass_gemm_16x64_s3_drop_op,
        (16, 64, 4) => cutlass_gemm_16x64_s4_drop_op,
        (16, 128, 3) => cutlass_gemm_16x128_s3_drop_op,
        (16, 128, 4) => cutlass_gemm_16x128_s4_drop_op,
        (32, 64, 3) => cutlass_gemm_32x64_s3_drop_op,
        (32, 64, 4) => cutlass_gemm_32x64_s4_drop_op,
        (32, 128, 3) => cutlass_gemm_32x128_s3_drop_op,
        (32, 128, 4) => cutlass_gemm_32x128_s4_drop_op,
        (32, 256, 3) => cutlass_gemm_32x256_s3_drop_op,
        (64, 64, 3) => cutlass_gemm_64x64_s3_drop_op,
        (64, 64, 4) => cutlass_gemm_64x64_s4_drop_op,
        (64, 128, 3) => cutlass_gemm_64x128_s3_drop_op,
        (64, 128, 4) => cutlass_gemm_64x128_s4_drop_op,
        (128, 64, 3) => cutlass_gemm_128x64_s3_drop_op,
        (128, 64, 4) => cutlass_gemm_128x64_s4_drop_op,
        (128, 128, 3) => cutlass_gemm_128x128_s3_drop_op,
        (128, 128, 4) => cutlass_gemm_128x128_s4_drop_op,
        (128, 256, 3) => cutlass_gemm_128x256_s3_drop_op,
        (256, 64, 3) => cutlass_gemm_256x64_s3_drop_op,
        (256, 64, 4) => cutlass_gemm_256x64_s4_drop_op,
        (64, 64, 2) => cutlass_gemm_64x64_s2_drop_op,
        (64, 64, 5) => cutlass_gemm_64x64_s5_drop_op,
        (64, 64, 6) => cutlass_gemm_64x64_s6_drop_op,
        (64, 64, 8) => cutlass_gemm_64x64_s8_drop_op,
        (64, 64, 10) => cutlass_gemm_64x64_s10_drop_op,
        (64, 128, 2) => cutlass_gemm_64x128_s2_drop_op,
        (64, 128, 5) => cutlass_gemm_64x128_s5_drop_op,
        (64, 128, 6) => cutlass_gemm_64x128_s6_drop_op,
        (64, 128, 7) => cutlass_gemm_64x128_s7_drop_op,
        (64, 128, 8) => cutlass_gemm_64x128_s8_drop_op,
        (128, 64, 2) => cutlass_gemm_128x64_s2_drop_op,
        (128, 64, 5) => cutlass_gemm_128x64_s5_drop_op,
        (128, 64, 6) => cutlass_gemm_128x64_s6_drop_op,
        (128, 64, 7) => cutlass_gemm_128x64_s7_drop_op,
        (128, 64, 8) => cutlass_gemm_128x64_s8_drop_op,
        (128, 128, 2) => cutlass_gemm_128x128_s2_drop_op,
        (128, 128, 5) => cutlass_gemm_128x128_s5_drop_op,
        (128, 128, 6) => cutlass_gemm_128x128_s6_drop_op,
        (64, 256, 2) => cutlass_gemm_64x256_s2_drop_op,
        (64, 256, 3) => cutlass_gemm_64x256_s3_drop_op,
        (64, 256, 4) => cutlass_gemm_64x256_s4_drop_op,
        (64, 256, 5) => cutlass_gemm_64x256_s5_drop_op,
        (128, 256, 2) => cutlass_gemm_128x256_s2_drop_op,
        (128, 256, 4) => cutlass_gemm_128x256_s4_drop_op,
        (256, 64, 2) => cutlass_gemm_256x64_s2_drop_op,
        (256, 64, 5) => cutlass_gemm_256x64_s5_drop_op,
        (256, 64, 6) => cutlass_gemm_256x64_s6_drop_op,
        (256, 128, 2) => cutlass_gemm_256x128_s2_drop_op,
        other => panic!("cutlass: unsupported tile {:?}", other),
    }
}

// ── Cached cutlass_gemm: 2-phase API with global memo ──────────────
//
// Wraps `_make_op` + `_run_op` behind a `Mutex<HashMap<key, *mut Op>>`
// keyed on (tile, M, N, K). First call at a new key: invokes
// `_make_op` (paying full CUTLASS setup once) and stores the handle.
// Subsequent calls: skip directly to `_run_op`, which only patches
// pointers via CUTLASS's `update()` and launches the kernel.
//
// Op handles are leaked at process exit — that's fine; the CUTLASS
// device::Gemm Op is a tiny struct (Params + a few stride/grid
// metadata), and there are at most ~30 tile variants × dozens of
// shapes per model = O(few hundred) handles in the worst case.
//
// `*mut c_void` isn't `Send`/`Sync` by default. The `OpHandle`
// newtype below asserts both — safe because: the Op handle is read
// from one thread at a time during a forward (the executor's compute
// thread), CUTLASS's `update()`/`run()` themselves are thread-safe
// w.r.t. their own state (they only touch params_ on the calling
// thread + submit a kernel to a stream).
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct OpHandle(*mut std::ffi::c_void);
#[cfg(feature = "cuda")]
unsafe impl Send for OpHandle {}
#[cfg(feature = "cuda")]
unsafe impl Sync for OpHandle {}

#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
static GEMM_OP_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<(u32, u32, u32, i32, i32, i32), OpHandle>>,
> = std::sync::OnceLock::new();

#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
static GEMV_OP_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<(i32, i32), OpHandle>>,
> = std::sync::OnceLock::new();

#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
static BIAS_OP_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<(u32, u32, u32, i32, i32, i32), OpHandle>>,
> = std::sync::OnceLock::new();

#[cfg(feature = "cuda")]
fn get_or_make_gemv_op(n: i32, k: i32) -> *mut std::ffi::c_void {
    let cache =
        GEMV_OP_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let key = (n, k);
    let mut guard = cache.lock().expect("GEMV_OP_CACHE poisoned");
    if let Some(handle) = guard.get(&key) {
        return handle.0;
    }
    let handle = unsafe { cutlass_gemv_make_op(n, k, 1.0, 0.0) };
    if handle.is_null() {
        panic!("cutlass_gemv_make_op(N={n}, K={k}) returned null");
    }
    guard.insert(key, OpHandle(handle));
    handle
}

#[cfg(feature = "cuda")]
fn get_or_make_gemm_op(tile: CutlassTile, m: i32, n: i32, k: i32) -> *mut std::ffi::c_void {
    let cache =
        GEMM_OP_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let key = (tile.tile_m, tile.tile_n, tile.stages, m, n, k);
    let mut guard = cache.lock().expect("GEMM_OP_CACHE poisoned");
    if let Some(handle) = guard.get(&key) {
        return handle.0;
    }
    let make = make_op_fn_for(tile);
    let handle = unsafe { make(m, n, k, 1.0, 0.0) };
    if handle.is_null() {
        panic!(
            "cutlass_gemm_make_op({:?}, M={m}, N={n}, K={k}) returned null — \
             can_implement or initialize failed",
            (tile.tile_m, tile.tile_n, tile.stages)
        );
    }
    guard.insert(key, OpHandle(handle));
    handle
}

/// 2-phase variant of [`cutlass_gemm`]. First call at a (tile, M, N, K):
/// pays full CUTLASS setup. Subsequent calls: just `update()` + `run()`.
/// Same alignment fallback as `cutlass_gemm` — non-aligned shapes go
/// through `any_align_bf16_gemm`.
///
/// # Safety
/// Same as [`cutlass_gemm`].
#[cfg(feature = "cuda")]
pub unsafe fn cutlass_gemm_cached(
    a: ferrite_cuda_core::tensor::GpuTensor,
    b: ferrite_cuda_core::tensor::GpuTensor,
    tile: CutlassTile,
    alloc: &mut ferrite_cuda_core::alloc::CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> ferrite_cuda_core::alloc::OwnedTensor {
    debug_assert_eq!(a.ndim(), 2);
    debug_assert_eq!(b.ndim(), 2);
    debug_assert_eq!(a.dim(1), b.dim(1), "GEMM K mismatch");
    let m = a.dim(0);
    let n = b.dim(0);
    let k = a.dim(1);
    if !n.is_multiple_of(8) || !k.is_multiple_of(8) {
        return unsafe { any_align_bf16_gemm(a, b, alloc, stream) };
    }
    let out = alloc.alloc_tensor(&[m, n], a.dtype());
    let op = get_or_make_gemm_op(tile, m as i32, n as i32, k as i32);
    let run = run_op_fn_for(tile);
    let rc = unsafe {
        run(
            op,
            out.as_mut_ptr::<u16>(),
            a.as_ptr::<u16>(),
            b.as_ptr::<u16>(),
            m as i32,
            n as i32,
            k as i32,
            1.0,
            0.0,
            stream as u64,
        )
    };
    debug_assert_eq!(
        rc, 0,
        "cutlass_gemm_cached {:?} M={m} N={n} K={k} returned {}",
        tile, rc
    );
    out
}

/// Any-alignment bf16 GEMM. Backed by a custom SMEM-tiled kernel
/// (`any_align_bf16_gemm_kernel` in `cutlass_standalone_gemm.cu`)
/// that handles any (M, N, K) ≥ (1, 1, 1) — required for shapes
/// the standalone CUTLASS tile zoo rejects (granite-3.3-2B's
/// `vocab_size = 49159` lm_head, AlignmentB=8 fail). ~30× slower
/// than the tensor-core path; only chosen when no aligned tile
/// works.
///
/// # Safety
/// All inputs must be valid GPU bf16 memory with the shapes
/// claimed by their `GpuTensor`. `stream` must be the live compute
/// stream.
#[cfg(feature = "cuda")]
pub unsafe fn any_align_bf16_gemm(
    a: ferrite_cuda_core::tensor::GpuTensor,
    b: ferrite_cuda_core::tensor::GpuTensor,
    alloc: &mut ferrite_cuda_core::alloc::CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> ferrite_cuda_core::alloc::OwnedTensor {
    debug_assert_eq!(a.ndim(), 2);
    debug_assert_eq!(b.ndim(), 2);
    debug_assert_eq!(a.dim(1), b.dim(1), "GEMM K mismatch");
    let m = a.dim(0);
    let n = b.dim(0);
    let k = a.dim(1);
    let out = alloc.alloc_tensor(&[m, n], a.dtype());

    unsafe extern "C" {
        fn any_align_bf16_gemm_launch(
            c: *mut std::ffi::c_void,
            a: *const std::ffi::c_void,
            b: *const std::ffi::c_void,
            m: i32,
            n: i32,
            k: i32,
            stream: u64,
        ) -> i32;
    }
    let rc = unsafe {
        any_align_bf16_gemm_launch(
            out.as_mut_ptr::<u16>() as *mut _,
            a.as_ptr::<u16>() as *const _,
            b.as_ptr::<u16>() as *const _,
            m as i32,
            n as i32,
            k as i32,
            stream as u64,
        )
    };
    assert_eq!(rc, 0, "any_align_bf16_gemm M={m} N={n} K={k} returned {rc}");
    out
}

/// Any-alignment bf16 GEMM with bias broadcast — `D[M,N] = A @ Bᵀ + bias[N]`.
///
/// # Safety
/// As `any_align_bf16_gemm`; `bias` must be a 1-D bf16 tensor of
/// length `N`.
#[cfg(feature = "cuda")]
pub unsafe fn any_align_bf16_gemm_bias(
    a: ferrite_cuda_core::tensor::GpuTensor,
    b: ferrite_cuda_core::tensor::GpuTensor,
    bias: ferrite_cuda_core::tensor::GpuTensor,
    alloc: &mut ferrite_cuda_core::alloc::CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> ferrite_cuda_core::alloc::OwnedTensor {
    debug_assert_eq!(a.ndim(), 2);
    debug_assert_eq!(b.ndim(), 2);
    debug_assert_eq!(a.dim(1), b.dim(1), "GEMM K mismatch");
    debug_assert_eq!(bias.ndim(), 1);
    debug_assert_eq!(b.dim(0), bias.dim(0), "GEMM bias N mismatch");
    let m = a.dim(0);
    let n = b.dim(0);
    let k = a.dim(1);
    let out = alloc.alloc_tensor(&[m, n], a.dtype());

    unsafe extern "C" {
        fn any_align_bf16_gemm_bias_launch(
            c: *mut std::ffi::c_void,
            a: *const std::ffi::c_void,
            b: *const std::ffi::c_void,
            bias: *const std::ffi::c_void,
            m: i32,
            n: i32,
            k: i32,
            stream: u64,
        ) -> i32;
    }
    let rc = unsafe {
        any_align_bf16_gemm_bias_launch(
            out.as_mut_ptr::<u16>() as *mut _,
            a.as_ptr::<u16>() as *const _,
            b.as_ptr::<u16>() as *const _,
            bias.as_ptr::<u16>() as *const _,
            m as i32,
            n as i32,
            k as i32,
            stream as u64,
        )
    };
    assert_eq!(
        rc, 0,
        "any_align_bf16_gemm_bias M={m} N={n} K={k} returned {rc}"
    );
    out
}

/// Safe wrapper: allocate output `[M, N]` and invoke the tile
/// variant's launch fn on the given cuBLAS-convention pointers.
/// Falls back to `any_align_bf16_gemm` when the shape isn't
/// 8-element aligned on N/K (granite-3.3-2B's vocab=49159 hits
/// this on lm_head); the tensor-core tiles return -1 for those
/// shapes and the fallback is the only correctness path.
///
/// # Safety
/// All inputs must be valid GPU BF16/FP16 memory with the shapes
/// claimed by their `GpuTensor`. `stream` must be the live compute
/// stream.
#[cfg(feature = "cuda")]
pub unsafe fn cutlass_gemm(
    a: ferrite_cuda_core::tensor::GpuTensor,
    b: ferrite_cuda_core::tensor::GpuTensor,
    tile: CutlassTile,
    alloc: &mut ferrite_cuda_core::alloc::CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> ferrite_cuda_core::alloc::OwnedTensor {
    // Basic variants route through `cutlass_gemm_cached` (2-phase API:
    // full CUTLASS setup once per (tile, M, N, K), subsequent calls
    // skip to `update()` + `run()`). Non-basic variants (Sw and the
    // future K64 / W8 / K64W8 families) only have the monolithic
    // `_launch` symbol in `cutlass_standalone_gemm.cu` today; they go
    // through `cutlass_gemm_legacy`. Per-call CUTLASS setup adds a few
    // microseconds vs. the cached path, but at large-M shapes (where
    // the swizzle helps) the GEMM itself is hundreds of microseconds —
    // overhead is well-amortized. Add `_make_op` / `_run_op` shims in
    // the .cu when the legacy overhead becomes the dominant cost.
    match tile.variant {
        GemmVariant::Basic => unsafe { cutlass_gemm_cached(a, b, tile, alloc, stream) },
        GemmVariant::Sw => unsafe { cutlass_gemm_legacy(a, b, tile, alloc, stream) },
        // Wmma kernels only have the legacy `_launch` symbol — the
        // 2-phase `_make_op` / `_run_op` shims aren't compiled in
        // `cutlass_standalone_gemm.cu` for the Wmma family today.
        GemmVariant::Wmma => unsafe { cutlass_gemm_legacy(a, b, tile, alloc, stream) },
    }
}

/// Monolithic launch path (full setup per call). Used both as a
/// regression-bisecting reference for the cached path AND as the
/// production entrypoint for tile variants whose `_make_op`/`_run_op`
/// shims aren't compiled in `cutlass_standalone_gemm.cu` (currently
/// the `Sw` swizzle family — see `cutlass_gemm`'s variant dispatch).
///
/// # Safety
/// Same as [`cutlass_gemm`].
#[cfg(feature = "cuda")]
pub unsafe fn cutlass_gemm_legacy(
    a: ferrite_cuda_core::tensor::GpuTensor,
    b: ferrite_cuda_core::tensor::GpuTensor,
    tile: CutlassTile,
    alloc: &mut ferrite_cuda_core::alloc::CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> ferrite_cuda_core::alloc::OwnedTensor {
    debug_assert_eq!(a.ndim(), 2);
    debug_assert_eq!(b.ndim(), 2);
    debug_assert_eq!(a.dim(1), b.dim(1), "GEMM K mismatch");
    let m = a.dim(0);
    let n = b.dim(0);
    let k = a.dim(1);
    if !n.is_multiple_of(8) || !k.is_multiple_of(8) {
        return unsafe { any_align_bf16_gemm(a, b, alloc, stream) };
    }
    let out = alloc.alloc_tensor(&[m, n], a.dtype());
    let launch = launch_fn_for(tile);
    let rc = unsafe {
        launch(
            out.as_mut_ptr::<u16>(),
            a.as_ptr::<u16>(),
            b.as_ptr::<u16>(),
            m as i32,
            n as i32,
            k as i32,
            1.0,
            0.0,
            stream as u64,
        )
    };
    assert_eq!(
        rc, 0,
        "cutlass_gemm {:?} M={m} N={n} K={k} returned {rc}",
        tile
    );
    out
}

/// SplitK tile variant — `(tile_m, tile_n, stages, split_k_slices)`.
/// Backed by `cutlass::gemm::device::GemmSplitKParallel` in the .cu;
/// the kernel launches a partial-GEMM grid followed by a reduction
/// kernel. Workspace is managed statically inside the .cu.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CutlassSplitKTile {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
    pub split_k: u32,
}

impl CutlassSplitKTile {
    pub const fn new(tile_m: u32, tile_n: u32, stages: u32, split_k: u32) -> Self {
        Self {
            tile_m,
            tile_n,
            stages,
            split_k,
        }
    }

    /// CSV column name — `cutlass_WxH_sS_splitN`. Must match the
    /// sweep's emitted row name and `CutlassGemmSplitKImpl::csv_name`.
    pub fn csv_name(self) -> String {
        format!(
            "cutlass_{}x{}_s{}_split{}",
            self.tile_m, self.tile_n, self.stages, self.split_k
        )
    }
}

/// SplitK launch fn type — adds a workspace pointer before stream.
#[cfg(feature = "cuda")]
type CutlassSplitKLaunchFn = unsafe extern "C" fn(
    *mut u16,
    *const u16,
    *const u16,
    i32,
    i32,
    i32,
    f32,
    f32,
    *mut u8,
    u64,
) -> i32;

#[cfg(feature = "cuda")]
fn launch_fn_for_splitk(tile: CutlassSplitKTile) -> CutlassSplitKLaunchFn {
    match (tile.tile_m, tile.tile_n, tile.stages, tile.split_k) {
        (64, 64, 4, 2) => cutlass_gemm_64x64_s4_sk2_launch,
        (64, 64, 4, 4) => cutlass_gemm_64x64_s4_sk4_launch,
        (64, 64, 4, 8) => cutlass_gemm_64x64_s4_sk8_launch,
        (64, 128, 4, 2) => cutlass_gemm_64x128_s4_sk2_launch,
        (64, 128, 4, 4) => cutlass_gemm_64x128_s4_sk4_launch,
        (64, 128, 4, 8) => cutlass_gemm_64x128_s4_sk8_launch,
        (128, 64, 4, 2) => cutlass_gemm_128x64_s4_sk2_launch,
        (128, 64, 4, 4) => cutlass_gemm_128x64_s4_sk4_launch,
        (128, 64, 4, 8) => cutlass_gemm_128x64_s4_sk8_launch,
        (128, 128, 4, 2) => cutlass_gemm_128x128_s4_sk2_launch,
        (128, 128, 4, 4) => cutlass_gemm_128x128_s4_sk4_launch,
        (128, 128, 4, 8) => cutlass_gemm_128x128_s4_sk8_launch,
        (16, 64, 4, 2) => cutlass_gemm_16x64_s4_sk2_launch,
        (16, 64, 4, 4) => cutlass_gemm_16x64_s4_sk4_launch,
        (16, 64, 4, 8) => cutlass_gemm_16x64_s4_sk8_launch,
        (16, 128, 4, 2) => cutlass_gemm_16x128_s4_sk2_launch,
        (16, 128, 4, 4) => cutlass_gemm_16x128_s4_sk4_launch,
        (16, 128, 4, 8) => cutlass_gemm_16x128_s4_sk8_launch,
        other => panic!(
            "cutlass splitk: unsupported tile {:?} — add its extern + csv entry",
            other,
        ),
    }
}

/// SplitK GEMM: `C[M, N] = A[M, K] @ B[N, K]^T` with the K dim split
/// across `split_k` CTAs, reduced in a second kernel. Same calling
/// convention as [`cutlass_gemm`].
///
/// # Safety
/// All inputs must be valid GPU BF16/FP16 memory with the shapes
/// claimed by their `GpuTensor`. `stream` must be the live compute
/// stream.
#[cfg(feature = "cuda")]
pub unsafe fn cutlass_gemm_splitk(
    a: ferrite_cuda_core::tensor::GpuTensor,
    b: ferrite_cuda_core::tensor::GpuTensor,
    tile: CutlassSplitKTile,
    alloc: &mut ferrite_cuda_core::alloc::CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> ferrite_cuda_core::alloc::OwnedTensor {
    debug_assert_eq!(a.ndim(), 2);
    debug_assert_eq!(b.ndim(), 2);
    debug_assert_eq!(a.dim(1), b.dim(1), "splitK GEMM K mismatch");
    let m = a.dim(0);
    let n = b.dim(0);
    let k = a.dim(1);
    // CUTLASS GemmSplitKParallel needs N % 8 == K % 8 == 0
    // (AlignmentB=8). Same fallback pattern as `cutlass_gemm` /
    // `cutlass_gemv` — granite-3.3-2b's lm_head at N=49159
    // (vocab not a multiple of 8) takes this path at M=1024 prefill
    // and faults `CUDA_ERROR_INVALID_VALUE` (status=-3 from launcher)
    // without the any-align fallback. Predictor commit 2693a5c96
    // surfaced the bug by picking SplitK at unaligned shapes where
    // roofline previously tied tiles and the safer fused peer won.
    if !n.is_multiple_of(8) || !k.is_multiple_of(8) {
        return unsafe { any_align_bf16_gemm(a, b, alloc, stream) };
    }
    let out = alloc.alloc_tensor(&[m, n], a.dtype());

    // GemmSplitKParallel workspace: partial-sum buffer of shape
    // [split_k, M, N] in f32 (accumulator type). Allocated through
    // the same caching allocator that owns every other scratch
    // tensor in the forward — never cudaMalloc/Free per call.
    let ws_elems = (tile.split_k as usize) * m * n;
    let workspace = alloc.alloc_tensor(&[ws_elems], ferrite_cuda_core::DType::F32);

    let launch = launch_fn_for_splitk(tile);
    let rc = unsafe {
        launch(
            out.as_mut_ptr::<u16>(),
            a.as_ptr::<u16>(),
            b.as_ptr::<u16>(),
            m as i32,
            n as i32,
            k as i32,
            1.0,
            0.0,
            workspace.as_mut_ptr::<u8>(),
            stream as u64,
        )
    };
    assert_eq!(
        rc, 0,
        "cutlass_gemm_splitk {:?} M={m} N={n} K={k} returned {rc}",
        tile
    );
    // workspace dropped here → returned to caching pool. Safe because
    // the forward runs on a single compute stream: any later alloc
    // that recycles these bytes enqueues its kernel behind the splitK
    // GEMM+reduce on the same stream (FIFO), so the bytes are read to
    // completion before they're rewritten. Same discipline as
    // `cutlass_gemm_silu_mul` dropping `up_out` post-launch.
    drop(workspace);
    out
}

/// Fused GEMM + residual-add via CUTLASS `LinearCombination`
/// epilogue: mutates `residual` in-place with `residual += A @ B^T`
/// (alpha=1.0, beta=1.0). Same buffer-aliasing convention as
/// `add_inplace` — caller binds the downstream tile to a TensorView
/// of `residual` via `as_view()`.
///
/// # Safety
/// Same as [`cutlass_gemm`]. `residual` must point to writable GPU
/// memory of shape `[M, N]` (the output shape of `A @ B^T`).
#[cfg(feature = "cuda")]
pub unsafe fn cutlass_gemm_add(
    a: ferrite_cuda_core::tensor::GpuTensor,
    b: ferrite_cuda_core::tensor::GpuTensor,
    residual: ferrite_cuda_core::tensor::GpuTensor,
    tile: CutlassTile,
    stream: cudarc::driver::sys::CUstream,
) {
    debug_assert_eq!(a.ndim(), 2);
    debug_assert_eq!(b.ndim(), 2);
    debug_assert_eq!(residual.ndim(), 2);
    debug_assert_eq!(a.dim(1), b.dim(1), "cutlass_gemm_add K mismatch");
    debug_assert_eq!(a.dim(0), residual.dim(0), "cutlass_gemm_add M mismatch");
    debug_assert_eq!(b.dim(0), residual.dim(1), "cutlass_gemm_add N mismatch");
    let m = a.dim(0);
    let n = b.dim(0);
    let k = a.dim(1);
    let launch = launch_fn_for(tile);
    let rc = unsafe {
        launch(
            residual.as_mut_ptr::<u16>(),
            a.as_ptr::<u16>(),
            b.as_ptr::<u16>(),
            m as i32,
            n as i32,
            k as i32,
            1.0,
            1.0,
            stream as u64,
        )
    };
    debug_assert_eq!(rc, 0, "cutlass_gemm_add {:?} returned {}", tile, rc);
}

/// M=1 specialisation — SIMT GEMV.
///
/// # Safety
/// Same as [`cutlass_gemm`].
#[cfg(feature = "cuda")]
pub unsafe fn cutlass_gemv(
    a: ferrite_cuda_core::tensor::GpuTensor,
    b: ferrite_cuda_core::tensor::GpuTensor,
    alloc: &mut ferrite_cuda_core::alloc::CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> ferrite_cuda_core::alloc::OwnedTensor {
    debug_assert_eq!(a.ndim(), 2);
    debug_assert_eq!(b.ndim(), 2);
    debug_assert_eq!(a.dim(1), b.dim(1), "GEMV K mismatch");
    let m = a.dim(0);
    let n = b.dim(0);
    let k = a.dim(1);
    // CUTLASS gemv kernel requires N % 8 == 0 (AlignmentB=8) — same
    // restriction `cutlass_gemm` falls back through. Without this
    // gate, granite-3.3-2b's lm_head at N=49159 (vocab not a
    // multiple of 8) faults with `CUDA_ERROR_MISALIGNED_ADDRESS`.
    // The any-align fallback works at any (M, N, K) ≥ (1,1,1) at
    // ~30× the per-FLOP cost of the optimized tile, but for granite's
    // vocab N at decode (M=1) the FLOP delta is small.
    if !n.is_multiple_of(8) || !k.is_multiple_of(8) {
        return unsafe { any_align_bf16_gemm(a, b, alloc, stream) };
    }
    let out = alloc.alloc_tensor(&[m, n], a.dtype());
    // Note: CUTLASS device::Gemv has a bug in
    // `cutlass/gemm/kernel/gemv.h` where `Params::update()` does
    // `ref_A = ref_A;` (self-assignment, no-op) instead of copying
    // `args.ref_A`. So `update()` never patches the weight pointer
    // — caching an Op + calling update + run reads stale (nullptr)
    // ref_A and faults `CUDA_ERROR_ILLEGAL_ADDRESS`. Stay on the
    // monolithic launch path until either CUTLASS upstream fixes
    // the bug or we hard-fork a corrected `gemv.h` locally.
    let _ = get_or_make_gemv_op; // keep the cache helper compiled
    let rc = unsafe {
        cutlass_gemv_launch(
            out.as_mut_ptr::<u16>(),
            a.as_ptr::<u16>(),
            b.as_ptr::<u16>(),
            m as i32,
            n as i32,
            k as i32,
            1.0,
            0.0,
            stream as u64,
        )
    };
    debug_assert_eq!(rc, 0, "cutlass_gemv returned {}", rc);
    out
}

/// CUTLASS GEMV with row-broadcast bias. Output: `out = b @ a + bias`.
/// One launch — `cublasGemvParamsEx`'s biased-gemv equivalent.
///
/// Requires `n % 8 == 0` and `k % 8 == 0`. For misaligned shapes the
/// caller must fall back to a separate gemm + add_bias chain
/// (matches `cutlass_gemv`'s alignment gate exactly).
///
/// # Safety
/// Same as [`cutlass_gemv`].
#[cfg(feature = "cuda")]
pub unsafe fn cutlass_gemv_bias(
    a: ferrite_cuda_core::tensor::GpuTensor,
    b: ferrite_cuda_core::tensor::GpuTensor,
    bias: ferrite_cuda_core::tensor::GpuTensor,
    alloc: &mut ferrite_cuda_core::alloc::CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> ferrite_cuda_core::alloc::OwnedTensor {
    debug_assert_eq!(a.ndim(), 2);
    debug_assert_eq!(b.ndim(), 2);
    debug_assert_eq!(a.dim(1), b.dim(1), "GEMV K mismatch");
    let m = a.dim(0);
    let n = b.dim(0);
    let k = a.dim(1);
    debug_assert_eq!(m, 1, "cutlass_gemv_bias: M must be 1");
    debug_assert_eq!(bias.numel(), n, "bias size must equal N");
    debug_assert!(
        n.is_multiple_of(8) && k.is_multiple_of(8),
        "cutlass_gemv_bias requires (N, K) % 8 == 0",
    );
    let out = alloc.alloc_tensor(&[m, n], a.dtype());
    let rc = unsafe {
        cutlass_gemv_bias_launch(
            out.as_mut_ptr::<u16>(),
            a.as_ptr::<u16>(),
            b.as_ptr::<u16>(),
            bias.as_ptr::<u16>(),
            m as i32,
            n as i32,
            k as i32,
            stream as u64,
        )
    };
    debug_assert_eq!(rc, 0, "cutlass_gemv_bias returned {}", rc);
    out
}

#[cfg(feature = "cuda")]
type CutlassBiasMakeOpFn = unsafe extern "C" fn(i32, i32, i32) -> *mut std::ffi::c_void;
#[cfg(feature = "cuda")]
type CutlassBiasRunOpFn = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    *mut u16,
    *const u16,
    *const u16,
    *const u16,
    i32,
    i32,
    i32,
    u64,
) -> i32;

#[cfg(feature = "cuda")]
fn make_op_fn_for_bias(tile: CutlassTile) -> CutlassBiasMakeOpFn {
    match (tile.tile_m, tile.tile_n, tile.stages) {
        (16, 64, 3) => cutlass_gemm_bias_16x64_s3_make_op,
        (16, 64, 4) => cutlass_gemm_bias_16x64_s4_make_op,
        (16, 128, 3) => cutlass_gemm_bias_16x128_s3_make_op,
        (16, 128, 4) => cutlass_gemm_bias_16x128_s4_make_op,
        (32, 64, 3) => cutlass_gemm_bias_32x64_s3_make_op,
        (32, 64, 4) => cutlass_gemm_bias_32x64_s4_make_op,
        (32, 128, 3) => cutlass_gemm_bias_32x128_s3_make_op,
        (32, 128, 4) => cutlass_gemm_bias_32x128_s4_make_op,
        (32, 256, 3) => cutlass_gemm_bias_32x256_s3_make_op,
        (64, 64, 3) => cutlass_gemm_bias_64x64_s3_make_op,
        (64, 64, 4) => cutlass_gemm_bias_64x64_s4_make_op,
        (64, 128, 3) => cutlass_gemm_bias_64x128_s3_make_op,
        (64, 128, 4) => cutlass_gemm_bias_64x128_s4_make_op,
        (128, 64, 3) => cutlass_gemm_bias_128x64_s3_make_op,
        (128, 64, 4) => cutlass_gemm_bias_128x64_s4_make_op,
        (128, 128, 3) => cutlass_gemm_bias_128x128_s3_make_op,
        (128, 128, 4) => cutlass_gemm_bias_128x128_s4_make_op,
        (128, 256, 3) => cutlass_gemm_bias_128x256_s3_make_op,
        (256, 64, 3) => cutlass_gemm_bias_256x64_s3_make_op,
        (256, 64, 4) => cutlass_gemm_bias_256x64_s4_make_op,
        other => panic!("cutlass_gemm_bias: unsupported tile {:?}", other),
    }
}

#[cfg(feature = "cuda")]
fn run_op_fn_for_bias(tile: CutlassTile) -> CutlassBiasRunOpFn {
    match (tile.tile_m, tile.tile_n, tile.stages) {
        (16, 64, 3) => cutlass_gemm_bias_16x64_s3_run_op,
        (16, 64, 4) => cutlass_gemm_bias_16x64_s4_run_op,
        (16, 128, 3) => cutlass_gemm_bias_16x128_s3_run_op,
        (16, 128, 4) => cutlass_gemm_bias_16x128_s4_run_op,
        (32, 64, 3) => cutlass_gemm_bias_32x64_s3_run_op,
        (32, 64, 4) => cutlass_gemm_bias_32x64_s4_run_op,
        (32, 128, 3) => cutlass_gemm_bias_32x128_s3_run_op,
        (32, 128, 4) => cutlass_gemm_bias_32x128_s4_run_op,
        (32, 256, 3) => cutlass_gemm_bias_32x256_s3_run_op,
        (64, 64, 3) => cutlass_gemm_bias_64x64_s3_run_op,
        (64, 64, 4) => cutlass_gemm_bias_64x64_s4_run_op,
        (64, 128, 3) => cutlass_gemm_bias_64x128_s3_run_op,
        (64, 128, 4) => cutlass_gemm_bias_64x128_s4_run_op,
        (128, 64, 3) => cutlass_gemm_bias_128x64_s3_run_op,
        (128, 64, 4) => cutlass_gemm_bias_128x64_s4_run_op,
        (128, 128, 3) => cutlass_gemm_bias_128x128_s3_run_op,
        (128, 128, 4) => cutlass_gemm_bias_128x128_s4_run_op,
        (128, 256, 3) => cutlass_gemm_bias_128x256_s3_run_op,
        (256, 64, 3) => cutlass_gemm_bias_256x64_s3_run_op,
        (256, 64, 4) => cutlass_gemm_bias_256x64_s4_run_op,
        other => panic!("cutlass_gemm_bias: unsupported tile {:?}", other),
    }
}

#[cfg(feature = "cuda")]
fn get_or_make_bias_op(tile: CutlassTile, m: i32, n: i32, k: i32) -> *mut std::ffi::c_void {
    let cache =
        BIAS_OP_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let key = (tile.tile_m, tile.tile_n, tile.stages, m, n, k);
    let mut guard = cache.lock().expect("BIAS_OP_CACHE poisoned");
    if let Some(handle) = guard.get(&key) {
        return handle.0;
    }
    let make = make_op_fn_for_bias(tile);
    let handle = unsafe { make(m, n, k) };
    if handle.is_null() {
        panic!(
            "cutlass_gemm_bias_make_op({:?}, M={m}, N={n}, K={k}) returned null",
            (tile.tile_m, tile.tile_n, tile.stages)
        );
    }
    guard.insert(key, OpHandle(handle));
    handle
}

/// Fused GEMM + bias broadcast.
///
/// Computes `D[M, N] = A @ W^T + bias` where:
/// - `a` is `[M, K]` bf16 activation,
/// - `weight` is `[N, K]` bf16 weight (row-major, cuBLAS-convention),
/// - `bias` is `[N]` bf16 per-column bias,
/// - output `D` is `[M, N]` bf16 allocated fresh.
///
/// `tile` selects the threadblock variant; the DP solver picks per
/// (M, N, K) workload from calibrated CSV rows.
///
/// # Safety
/// All inputs must be valid GPU bf16 memory with the shapes claimed
/// by their `GpuTensor`. `stream` must be the live compute stream.
#[cfg(feature = "cuda")]
pub unsafe fn cutlass_gemm_bias(
    a: ferrite_cuda_core::tensor::GpuTensor,
    weight: ferrite_cuda_core::tensor::GpuTensor,
    bias: ferrite_cuda_core::tensor::GpuTensor,
    tile: CutlassTile,
    alloc: &mut ferrite_cuda_core::alloc::CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> ferrite_cuda_core::alloc::OwnedTensor {
    debug_assert_eq!(a.ndim(), 2);
    debug_assert_eq!(weight.ndim(), 2);
    debug_assert_eq!(bias.ndim(), 1);
    debug_assert_eq!(a.dim(1), weight.dim(1), "cutlass_gemm_bias K mismatch");
    debug_assert_eq!(weight.dim(0), bias.dim(0), "cutlass_gemm_bias N mismatch");
    let m = a.dim(0);
    let n = weight.dim(0);
    let k = a.dim(1);
    if !n.is_multiple_of(8) || !k.is_multiple_of(8) {
        return unsafe { any_align_bf16_gemm_bias(a, weight, bias, alloc, stream) };
    }
    let out = alloc.alloc_tensor(&[m, n], a.dtype());
    // 2-phase API: per-(tile, M, N, K) Op cached, per-call just
    // patches pointers via CUTLASS's `update()` and runs the kernel.
    let op = get_or_make_bias_op(tile, m as i32, n as i32, k as i32);
    let run = run_op_fn_for_bias(tile);
    let rc = unsafe {
        run(
            op,
            out.as_mut_ptr::<u16>(),
            a.as_ptr::<u16>(),
            weight.as_ptr::<u16>(),
            bias.as_ptr::<u16>(),
            m as i32,
            n as i32,
            k as i32,
            stream as u64,
        )
    };
    debug_assert_eq!(rc, 0, "cutlass_gemm_bias {:?} returned {}", tile, rc);
    out
}

/// Fused Gate GEMM + SiLU + Mul in a single CUTLASS EVT kernel.
///
/// Computes `D[M, I] = silu(A @ W_gate^T) * C_up` where:
/// - `a` is `[M, K]` bf16 activation (row-major),
/// - `gate_weight` is `[I, K]` bf16 gate-projection weight (row-major,
///   cuBLAS-convention — the kernel treats it as `W^T`),
/// - `up_out` is a pre-computed `[M, I]` bf16 up-projection output
///   that the epilogue aux-loads from GMEM,
/// - output `D` is `[M, I]` bf16 allocated fresh.
///
/// # Safety
/// All inputs must be valid GPU bf16 memory with the shapes claimed
/// by their `GpuTensor`. `stream` must be the live compute stream.
#[cfg(feature = "cuda")]
pub unsafe fn cutlass_gemm_silu_mul(
    a: ferrite_cuda_core::tensor::GpuTensor,
    gate_weight: ferrite_cuda_core::tensor::GpuTensor,
    up_out: ferrite_cuda_core::alloc::OwnedTensor,
    alloc: &mut ferrite_cuda_core::alloc::CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> ferrite_cuda_core::alloc::OwnedTensor {
    debug_assert_eq!(a.ndim(), 2);
    debug_assert_eq!(gate_weight.ndim(), 2);
    debug_assert_eq!(up_out.ndim(), 2);
    debug_assert_eq!(
        a.dim(1),
        gate_weight.dim(1),
        "cutlass_gemm_silu_mul K mismatch"
    );
    debug_assert_eq!(a.dim(0), up_out.dim(0), "cutlass_gemm_silu_mul M mismatch");
    debug_assert_eq!(
        gate_weight.dim(0),
        up_out.dim(1),
        "cutlass_gemm_silu_mul I mismatch"
    );
    let m = a.dim(0);
    let n = gate_weight.dim(0);
    let k = a.dim(1);
    let out = alloc.alloc_tensor(&[m, n], a.dtype());
    let rc = unsafe {
        cutlass_gemm_silu_mul_launch(
            out.as_mut_ptr::<u16>(),
            a.as_ptr::<u16>(),
            gate_weight.as_ptr::<u16>(),
            up_out.as_mut_ptr::<u16>(),
            m as i32,
            n as i32,
            k as i32,
            stream as u64,
        )
    };
    debug_assert_eq!(rc, 0, "cutlass_gemm_silu_mul returned {}", rc);
    // up_out is consumed — its buffer is no longer needed after the
    // epilogue aux-load completes.
    drop(up_out);
    out
}
