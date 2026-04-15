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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CutlassTile {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
}

impl CutlassTile {
    pub const fn new(tile_m: u32, tile_n: u32, stages: u32) -> Self {
        Self {
            tile_m,
            tile_n,
            stages,
        }
    }

    /// The kernel name used in `target_profiles/cost_*.csv`.
    /// Matches the CSV column value byte-for-byte.
    pub fn csv_name(self) -> String {
        format!("cutlass_{}x{}_s{}", self.tile_m, self.tile_n, self.stages)
    }
}

// Every tile variant compiled by `cutlass_standalone_gemm.cu` that
// is covered by the calibrated CSV tables. When the .cu file gains
// a variant, regenerate the CSV via `gpu_cost_sweep` and add the
// entry here.
#[cfg(feature = "cuda")]
unsafe extern "C" {
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
}

/// Safe wrapper: allocate output `[M, N]` and invoke the tile
/// variant's launch fn on the given cuBLAS-convention pointers.
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
    debug_assert_eq!(a.ndim(), 2);
    debug_assert_eq!(b.ndim(), 2);
    debug_assert_eq!(a.dim(1), b.dim(1), "GEMM K mismatch");
    let m = a.dim(0);
    let n = b.dim(0);
    let k = a.dim(1);
    let out = alloc.alloc_tensor(&[m, n], a.dtype());

    let launch: unsafe extern "C" fn(
        *mut u16,
        *const u16,
        *const u16,
        i32,
        i32,
        i32,
        f32,
        f32,
        u64,
    ) -> i32 = match (tile.tile_m, tile.tile_n, tile.stages) {
        (32, 64, 3) => cutlass_gemm_32x64_s3_launch,
        (32, 64, 4) => cutlass_gemm_32x64_s4_launch,
        (32, 128, 3) => cutlass_gemm_32x128_s3_launch,
        (32, 128, 4) => cutlass_gemm_32x128_s4_launch,
        (32, 256, 3) => cutlass_gemm_32x256_s3_launch,
        (64, 64, 3) => cutlass_gemm_64x64_s3_launch,
        (64, 64, 4) => cutlass_gemm_64x64_s4_launch,
        (64, 128, 3) => cutlass_gemm_64x128_s3_launch,
        (64, 128, 4) => cutlass_gemm_64x128_s4_launch,
        (128, 64, 3) => cutlass_gemm_128x64_s3_launch,
        (128, 64, 4) => cutlass_gemm_128x64_s4_launch,
        (128, 128, 3) => cutlass_gemm_128x128_s3_launch,
        (128, 128, 4) => cutlass_gemm_128x128_s4_launch,
        (128, 256, 3) => cutlass_gemm_128x256_s3_launch,
        (256, 64, 3) => cutlass_gemm_256x64_s3_launch,
        (256, 64, 4) => cutlass_gemm_256x64_s4_launch,
        other => panic!(
            "cutlass_gemm: unsupported tile {:?} — add its extern declaration + csv entry",
            other,
        ),
    };

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
    debug_assert_eq!(rc, 0, "cutlass_gemm {:?} returned {}", tile, rc);
    out
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
    let out = alloc.alloc_tensor(&[m, n], a.dtype());
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
