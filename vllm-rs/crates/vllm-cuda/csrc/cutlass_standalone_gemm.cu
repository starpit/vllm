// SPDX-License-Identifier: Apache-2.0
// Standalone CUTLASS 2.x GEMM launchers for the solver, templatized
// over the activation/output element type T (bf16 or f16). Each macro
// invocation generates two kernel + launch wrappers per (TB_M, TB_N,
// WARP_M, WARP_N, STAGES) config — one bf16, one f16. Both share the
// HMMA tensor-core path on sm80/sm89/sm90 (same micro-arch shapes,
// same alignment-8 LinearCombination), so cost calibration is shared.
//
// C[M,N] = alpha * A[M,K] @ B[K,N]^T + beta * C[M,N]
// A: RowMajor T, B: ColumnMajor T, C: RowMajor T

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>
#include <cutlass/gemm/device/gemv.h>
#include <cutlass/gemm/kernel/gemv.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_silu.h>
#include <cutlass/reduction/device/reduce_split_k.h>
#include <cuda_runtime.h>

// ── Generic launch ──
//
// `T` is deduced from `GemmOp::ElementA` so callers stay unchanged.

template <typename GemmOp>
static int run_gemm(
    void* C, const void* A, const void* B,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream
) {
    using T = typename GemmOp::ElementA;
    typename GemmOp::Arguments args(
        {M, N, K},
        {(T const*)A, K},   // A [M,K] row-major, lda=K
        {(T const*)B, K},   // B [N,K] col-major, ldb=K
        {(T*)C, N},          // C [M,N] row-major, ldc=N
        {(T*)C, N},          // D = C (in-place for beta!=0)
        {alpha, beta}
    );

    GemmOp op;
    auto status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;

    status = op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) return -2;

    status = op(stream);
    return (status == cutlass::Status::kSuccess) ? 0 : -3;
}

// ── Macro: define type alias + extern "C" launch wrapper ──
//
// `DTYPE_TAG` is one of `bf16` / `f16`. `T` is the matching CUTLASS
// element type. Each `CUTLASS_GEMM_TYPED(...)` invocation registers a
// dtype-specific symbol (`cutlass_gemm_<tile>_<dtype>_launch`); the
// public `CUTLASS_GEMM(...)` calls the typed variant twice — once for
// each dtype.

#define CUTLASS_GEMM_CONFIG(DTYPE_TAG, T, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES) \
    using Gemm_##DTYPE_TAG##_##TB_M##x##TB_N##x##TB_K##_s##STAGES =                          \
        cutlass::gemm::device::Gemm<                                                          \
            T, cutlass::layout::RowMajor,                                                     \
            T, cutlass::layout::ColumnMajor,                                                  \
            T, cutlass::layout::RowMajor,                                                     \
            float,                                                                            \
            cutlass::arch::OpClassTensorOp,                                                   \
            cutlass::arch::Sm80,                                                              \
            cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>,                                      \
            cutlass::gemm::GemmShape<WARP_M, WARP_N, WARP_K>,                                \
            cutlass::gemm::GemmShape<16, 8, 16>,                                             \
            cutlass::epilogue::thread::LinearCombination<                                     \
                T, 8, float, float>,                                                          \
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,                     \
            STAGES                                                                            \
        >;

#define CUTLASS_GEMM_LAUNCH(DTYPE_TAG, TB_M, TB_N, TB_K, STAGES)                              \
    extern "C" int cutlass_gemm_##TB_M##x##TB_N##_s##STAGES##_##DTYPE_TAG##_launch(           \
        void* C, const void* A, const void* B,                                                \
        int M, int N, int K,                                                                  \
        float alpha, float beta,                                                              \
        uint64_t stream                                                                       \
    ) {                                                                                       \
        return run_gemm<Gemm_##DTYPE_TAG##_##TB_M##x##TB_N##x##TB_K##_s##STAGES>(            \
            C, A, B, M, N, K, alpha, beta, (cudaStream_t)stream);                             \
    }

#define CUTLASS_GEMM_TYPED(DTYPE_TAG, T, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES)   \
    CUTLASS_GEMM_CONFIG(DTYPE_TAG, T, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES)      \
    CUTLASS_GEMM_LAUNCH(DTYPE_TAG, TB_M, TB_N, TB_K, STAGES)

#define CUTLASS_GEMM(TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES)                       \
    CUTLASS_GEMM_TYPED(bf16, cutlass::bfloat16_t, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES) \
    CUTLASS_GEMM_TYPED(f16,  cutlass::half_t,    TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES)

// ── Instantiate all configs ──
//
// Covers the same threadblock shapes cuBLAS uses on sm80/sm89:
//   64×64, 64×128, 128×64, 128×128, 128×256, 256×64, 256×128
// Each at 3 and 4 pipeline stages (trades SMEM for latency hiding).
// The solver's cost sweep measures all of them; it picks per-workload.

// TB_M  TB_N  TB_K  WARP_M  WARP_N  WARP_K  STAGES
CUTLASS_GEMM( 64,   64,  32,    32,     32,     32,    4)
CUTLASS_GEMM( 64,  128,  32,    32,     64,     32,    4)
CUTLASS_GEMM(128,   64,  32,    64,     32,     32,    4)
CUTLASS_GEMM(128,  128,  32,    64,     32,     32,    4)
CUTLASS_GEMM(128,  256,  32,    64,     64,     32,    3)
CUTLASS_GEMM(256,   64,  32,    64,     32,     32,    4)
// 256×128 s3 exceeds sm89 SMEM (99KB); can_implement fails silently.
// CUTLASS_GEMM(256,  128,  32,    64,     32,     32,    3)

CUTLASS_GEMM( 64,   64,  32,    32,     32,     32,    3)
CUTLASS_GEMM( 64,  128,  32,    32,     64,     32,    3)
CUTLASS_GEMM(128,   64,  32,    64,     32,     32,    3)
CUTLASS_GEMM(128,  128,  32,    64,     32,     32,    3)
CUTLASS_GEMM(256,   64,  32,    64,     32,     32,    3)

// TB_K=64 variants — fewer main-loop iterations, better memory coalescing
// at larger M. MMA shape stays 16×8×16 but TB_K doubles.
// These need separate launch symbol names to avoid colliding with TB_K=32.
#define CUTLASS_GEMM_K64_LAUNCH(DTYPE_TAG, TB_M, TB_N, STAGES)                                \
    extern "C" int cutlass_gemm_##TB_M##x##TB_N##_k64_s##STAGES##_##DTYPE_TAG##_launch(       \
        void* C, const void* A, const void* B,                                                \
        int M, int N, int K,                                                                  \
        float alpha, float beta,                                                              \
        uint64_t stream                                                                       \
    ) {                                                                                       \
        return run_gemm<Gemm_##DTYPE_TAG##_##TB_M##x##TB_N##x64_s##STAGES>(                  \
            C, A, B, M, N, K, alpha, beta, (cudaStream_t)stream);                             \
    }
#define CUTLASS_GEMM_K64_TYPED(DTYPE_TAG, T, TB_M, TB_N, WARP_M, WARP_N, WARP_K, STAGES)     \
    CUTLASS_GEMM_CONFIG(DTYPE_TAG, T, TB_M, TB_N, 64, WARP_M, WARP_N, WARP_K, STAGES)         \
    CUTLASS_GEMM_K64_LAUNCH(DTYPE_TAG, TB_M, TB_N, STAGES)
#define CUTLASS_GEMM_K64(TB_M, TB_N, WARP_M, WARP_N, WARP_K, STAGES)                          \
    CUTLASS_GEMM_K64_TYPED(bf16, cutlass::bfloat16_t, TB_M, TB_N, WARP_M, WARP_N, WARP_K, STAGES) \
    CUTLASS_GEMM_K64_TYPED(f16,  cutlass::half_t,    TB_M, TB_N, WARP_M, WARP_N, WARP_K, STAGES)

CUTLASS_GEMM_K64( 64,   64,    32,     32,     64,    4)
CUTLASS_GEMM_K64( 64,   64,    32,     32,     64,    3)
CUTLASS_GEMM_K64( 64,  128,    32,     64,     64,    4)
CUTLASS_GEMM_K64( 64,  128,    32,     64,     64,    3)
CUTLASS_GEMM_K64(128,   64,    64,     32,     64,    4)
CUTLASS_GEMM_K64(128,   64,    64,     32,     64,    3)
CUTLASS_GEMM_K64(128,  128,    64,     32,     64,    4)
CUTLASS_GEMM_K64(128,  128,    64,     32,     64,    3)
CUTLASS_GEMM_K64(256,   64,    64,     32,     64,    4)
CUTLASS_GEMM_K64(256,   64,    64,     32,     64,    3)
// 128×256_k64 and 256×128_k64 likely exceed SMEM on sm89
CUTLASS_GEMM_K64(128,  256,    64,     64,     64,    3)
CUTLASS_GEMM_K64( 32,   64,    32,     32,     64,    4)
CUTLASS_GEMM_K64( 32,  128,    32,     64,     64,    4)
// 64x128_k64 with 8 warps (256 threads) — matches cuBLAS thread count
// Warp shape 32x32x64: 64/32 x 128/32 = 2x4 = 8 warps
#define CUTLASS_GEMM_K64_W8_CONFIG(DTYPE_TAG, T, TB_M, TB_N, STAGES)                          \
    using Gemm_w8_##DTYPE_TAG##_##TB_M##x##TB_N##x64_s##STAGES =                              \
        cutlass::gemm::device::Gemm<                                                          \
            T, cutlass::layout::RowMajor,                                                     \
            T, cutlass::layout::ColumnMajor,                                                  \
            T, cutlass::layout::RowMajor,                                                     \
            float,                                                                            \
            cutlass::arch::OpClassTensorOp,                                                   \
            cutlass::arch::Sm80,                                                               \
            cutlass::gemm::GemmShape<TB_M, TB_N, 64>,                                        \
            cutlass::gemm::GemmShape<32, 32, 64>,                                            \
            cutlass::gemm::GemmShape<16, 8, 16>,                                             \
            cutlass::epilogue::thread::LinearCombination<                                     \
                T, 8, float, float>,                                                          \
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,                     \
            STAGES                                                                            \
        >;
#define CUTLASS_GEMM_K64_W8_LAUNCH(DTYPE_TAG, TB_M, TB_N, STAGES)                             \
    extern "C" int cutlass_gemm_##TB_M##x##TB_N##_k64w8_s##STAGES##_##DTYPE_TAG##_launch(     \
        void* C, const void* A, const void* B,                                                \
        int M, int N, int K,                                                                  \
        float alpha, float beta,                                                              \
        uint64_t stream                                                                       \
    ) {                                                                                       \
        return run_gemm<Gemm_w8_##DTYPE_TAG##_##TB_M##x##TB_N##x64_s##STAGES>(               \
            C, A, B, M, N, K, alpha, beta, (cudaStream_t)stream);                             \
    }
#define CUTLASS_GEMM_K64_W8_TYPED(DTYPE_TAG, T, TB_M, TB_N, STAGES)                           \
    CUTLASS_GEMM_K64_W8_CONFIG(DTYPE_TAG, T, TB_M, TB_N, STAGES)                              \
    CUTLASS_GEMM_K64_W8_LAUNCH(DTYPE_TAG, TB_M, TB_N, STAGES)
#define CUTLASS_GEMM_K64_W8(TB_M, TB_N, STAGES)                                               \
    CUTLASS_GEMM_K64_W8_TYPED(bf16, cutlass::bfloat16_t, TB_M, TB_N, STAGES)                  \
    CUTLASS_GEMM_K64_W8_TYPED(f16,  cutlass::half_t,    TB_M, TB_N, STAGES)

CUTLASS_GEMM_K64_W8(64, 128, 3)
CUTLASS_GEMM_K64_W8(64, 128, 4)
CUTLASS_GEMM_K64_W8(64, 128, 2)
// Also try TB_K=32 with 8-warp config (warp 32x32x32)
// For TB 64x128x32: 2x4=8 warps, 8KB+16KB=24KB/stage
//
// Note: this shares the `Gemm_w8_<dtype>_<tile>x32_s<stages>` namespace
// prefix with `CUTLASS_GEMM_K64_W8_CONFIG` (TB_K=64). The full typedef
// name disambiguates them via the `x32` vs `x64` infix.
#define CUTLASS_GEMM_W8_CONFIG(DTYPE_TAG, T, TB_M, TB_N, STAGES)                              \
    using Gemm_w8_##DTYPE_TAG##_##TB_M##x##TB_N##x32_s##STAGES =                              \
        cutlass::gemm::device::Gemm<                                                          \
            T, cutlass::layout::RowMajor,                                                     \
            T, cutlass::layout::ColumnMajor,                                                  \
            T, cutlass::layout::RowMajor,                                                     \
            float,                                                                            \
            cutlass::arch::OpClassTensorOp,                                                   \
            cutlass::arch::Sm80,                                                              \
            cutlass::gemm::GemmShape<TB_M, TB_N, 32>,                                        \
            cutlass::gemm::GemmShape<32, 32, 32>,                                            \
            cutlass::gemm::GemmShape<16, 8, 16>,                                             \
            cutlass::epilogue::thread::LinearCombination<                                     \
                T, 8, float, float>,                                                          \
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,                     \
            STAGES                                                                            \
        >;
#define CUTLASS_GEMM_W8_LAUNCH(DTYPE_TAG, TB_M, TB_N, STAGES)                                 \
    extern "C" int cutlass_gemm_##TB_M##x##TB_N##_w8_s##STAGES##_##DTYPE_TAG##_launch(        \
        void* C, const void* A, const void* B,                                                \
        int M, int N, int K,                                                                  \
        float alpha, float beta,                                                              \
        uint64_t stream                                                                       \
    ) {                                                                                       \
        return run_gemm<Gemm_w8_##DTYPE_TAG##_##TB_M##x##TB_N##x32_s##STAGES>(               \
            C, A, B, M, N, K, alpha, beta, (cudaStream_t)stream);                             \
    }
#define CUTLASS_GEMM_W8_TYPED(DTYPE_TAG, T, TB_M, TB_N, STAGES)                               \
    CUTLASS_GEMM_W8_CONFIG(DTYPE_TAG, T, TB_M, TB_N, STAGES)                                  \
    CUTLASS_GEMM_W8_LAUNCH(DTYPE_TAG, TB_M, TB_N, STAGES)
#define CUTLASS_GEMM_W8(TB_M, TB_N, STAGES)                                                   \
    CUTLASS_GEMM_W8_TYPED(bf16, cutlass::bfloat16_t, TB_M, TB_N, STAGES)                      \
    CUTLASS_GEMM_W8_TYPED(f16,  cutlass::half_t,    TB_M, TB_N, STAGES)

CUTLASS_GEMM_W8(64, 128, 3)
CUTLASS_GEMM_W8(64, 128, 4)
CUTLASS_GEMM_W8(64, 128, 5)
CUTLASS_GEMM_W8(64, 128, 6)
CUTLASS_GEMM_W8(128, 128, 3)
CUTLASS_GEMM_W8(128, 128, 4)
CUTLASS_GEMM_W8(128, 128, 5)
// k64 stages=2 — more occupancy on sm89
CUTLASS_GEMM_K64(128,  128,    64,     32,     64,    2)
CUTLASS_GEMM_K64( 64,   64,    32,     32,     64,    2)
CUTLASS_GEMM_K64( 64,  128,    32,     64,     64,    2)
CUTLASS_GEMM_K64(128,   64,    64,     32,     64,    2)
CUTLASS_GEMM_K64(256,   64,    64,     32,     64,    2)
// 256×128 at stages=2 — attempt to fit in sm89 SMEM
CUTLASS_GEMM(256,  128,  32,    64,     32,     32,    2)

// ── Deep pipeline variants (stages=5,6,7,8) ──
// cuBLAS uses 7-17 stages on sm89. CUTLASS 2.x has no stage cap — only SMEM.
// 64×128 at TB_K=32: ~12KB/stage → 8 stages = 96KB (fits in 100KB sm89 SMEM)
CUTLASS_GEMM( 64,  128,  32,    32,     64,     32,    5)
CUTLASS_GEMM( 64,  128,  32,    32,     64,     32,    6)
CUTLASS_GEMM( 64,  128,  32,    32,     64,     32,    7)
CUTLASS_GEMM( 64,  128,  32,    32,     64,     32,    8)
// 128×128: ~16KB/stage → 6 stages = 96KB
CUTLASS_GEMM(128,  128,  32,    64,     32,     32,    5)
CUTLASS_GEMM(128,  128,  32,    64,     32,     32,    6)
// 64×64: ~8KB/stage → stages=8 = 64KB, 10 = 80KB, 12 = 96KB
CUTLASS_GEMM( 64,   64,  32,    32,     32,     32,    5)
CUTLASS_GEMM( 64,   64,  32,    32,     32,     32,    6)
CUTLASS_GEMM( 64,   64,  32,    32,     32,     32,    8)
CUTLASS_GEMM( 64,   64,  32,    32,     32,     32,   10)
// 128×64: ~12KB/stage → 8 stages = 96KB
CUTLASS_GEMM(128,   64,  32,    64,     32,     32,    5)
CUTLASS_GEMM(128,   64,  32,    64,     32,     32,    6)
CUTLASS_GEMM(128,   64,  32,    64,     32,     32,    7)
CUTLASS_GEMM(128,   64,  32,    64,     32,     32,    8)
// 64×256: ~20KB/stage → 5 stages = 100KB (barely fits)
CUTLASS_GEMM( 64,  256,  32,    32,     64,     32,    4)
CUTLASS_GEMM( 64,  256,  32,    32,     64,     32,    5)
// 128×256: ~24KB/stage → 4 stages = 96KB
CUTLASS_GEMM(128,  256,  32,    64,     64,     32,    4)
// 256×64: ~12KB/stage
CUTLASS_GEMM(256,   64,  32,    64,     32,     32,    5)
CUTLASS_GEMM(256,   64,  32,    64,     32,     32,    6)

// ── CTA-swizzled variants ──
// cuBLAS uses swizzle=1 for large-M shapes. This improves L2 locality
// by reordering threadblock launch order.
// GemmIdentityThreadblockSwizzle<N> swizzles N-wide in the N-tile dimension.
// N=8 gives strong L2 locality for large grids.
#define CUTLASS_GEMM_SWIZZLE_CONFIG(DTYPE_TAG, T, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES) \
    using Gemm_sw_##DTYPE_TAG##_##TB_M##x##TB_N##x##TB_K##_s##STAGES =                              \
        cutlass::gemm::device::Gemm<                                                                \
            T, cutlass::layout::RowMajor,                                                           \
            T, cutlass::layout::ColumnMajor,                                                        \
            T, cutlass::layout::RowMajor,                                                           \
            float,                                                                                  \
            cutlass::arch::OpClassTensorOp,                                                         \
            cutlass::arch::Sm80,                                                                    \
            cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>,                                            \
            cutlass::gemm::GemmShape<WARP_M, WARP_N, WARP_K>,                                      \
            cutlass::gemm::GemmShape<16, 8, 16>,                                                   \
            cutlass::epilogue::thread::LinearCombination<                                           \
                T, 8, float, float>,                                                                \
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,                          \
            STAGES                                                                                  \
        >;

#define CUTLASS_GEMM_SWIZZLE_LAUNCH(DTYPE_TAG, TB_M, TB_N, TB_K, STAGES)                            \
    extern "C" int cutlass_gemm_##TB_M##x##TB_N##_sw_s##STAGES##_##DTYPE_TAG##_launch(              \
        void* C, const void* A, const void* B,                                                      \
        int M, int N, int K,                                                                        \
        float alpha, float beta,                                                                    \
        uint64_t stream                                                                             \
    ) {                                                                                             \
        return run_gemm<Gemm_sw_##DTYPE_TAG##_##TB_M##x##TB_N##x##TB_K##_s##STAGES>(                \
            C, A, B, M, N, K, alpha, beta, (cudaStream_t)stream);                                   \
    }

#define CUTLASS_GEMM_SWIZZLE_TYPED(DTYPE_TAG, T, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES)  \
    CUTLASS_GEMM_SWIZZLE_CONFIG(DTYPE_TAG, T, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES)     \
    CUTLASS_GEMM_SWIZZLE_LAUNCH(DTYPE_TAG, TB_M, TB_N, TB_K, STAGES)

#define CUTLASS_GEMM_SWIZZLE(TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES)                     \
    CUTLASS_GEMM_SWIZZLE_TYPED(bf16, cutlass::bfloat16_t, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES) \
    CUTLASS_GEMM_SWIZZLE_TYPED(f16,  cutlass::half_t,    TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES)

// cuBLAS uses swizzle=1 for 128×128 at M=1024 — our worst remaining gap
CUTLASS_GEMM_SWIZZLE(128,  128,  32,    64,     32,     32,    4)
CUTLASS_GEMM_SWIZZLE(128,  128,  32,    64,     32,     32,    3)
CUTLASS_GEMM_SWIZZLE(128,  128,  32,    64,     32,     32,    2)
CUTLASS_GEMM_SWIZZLE(128,  256,  32,    64,     64,     32,    3)
CUTLASS_GEMM_SWIZZLE(128,  256,  32,    64,     64,     32,    2)
CUTLASS_GEMM_SWIZZLE(256,   64,  32,    64,     32,     32,    4)
CUTLASS_GEMM_SWIZZLE(256,   64,  32,    64,     32,     32,    3)
// Also swizzle on 64×128 (cuBLAS's top pick for M=128 K=8192)
CUTLASS_GEMM_SWIZZLE( 64,  128,  32,    32,     64,     32,    4)
CUTLASS_GEMM_SWIZZLE( 64,  128,  32,    32,     64,     32,    3)

// ── 64×256 tile — cuBLAS heuristic picks this for M=128 K=8192 ──
CUTLASS_GEMM( 64,  256,  32,    32,     64,     32,    3)
CUTLASS_GEMM( 64,  256,  32,    32,     64,     32,    2)
CUTLASS_GEMM_SWIZZLE( 64,  256,  32,    32,     64,     32,    3)
CUTLASS_GEMM_SWIZZLE( 64,  256,  32,    32,     64,     32,    2)

// TB_K=64 splitK — combine wider K-tile with K-splitting
#define CUTLASS_SPLITK_K64_CONFIG(DTYPE_TAG, T, TB_M, TB_N, WARP_M, WARP_N, WARP_K, STAGES) \
    using GemmSplitK_##DTYPE_TAG##_##TB_M##x##TB_N##_k64_s##STAGES =                        \
        cutlass::gemm::device::GemmSplitKParallel<                                          \
            T, cutlass::layout::RowMajor,                                                   \
            T, cutlass::layout::ColumnMajor,                                                \
            T, cutlass::layout::RowMajor,                                                   \
            float,                                                                          \
            cutlass::arch::OpClassTensorOp,                                                 \
            cutlass::arch::Sm80,                                                            \
            cutlass::gemm::GemmShape<TB_M, TB_N, 64>,                                      \
            cutlass::gemm::GemmShape<WARP_M, WARP_N, WARP_K>,                              \
            cutlass::gemm::GemmShape<16, 8, 16>,                                           \
            cutlass::epilogue::thread::LinearCombination<                                   \
                T, 8, float, float>,                                                        \
            cutlass::epilogue::thread::Convert<float, 8, float>,                            \
            cutlass::reduction::thread::ReduceAdd<                                          \
                float, float, 8>,                                                           \
            cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle,             \
            STAGES                                                                          \
        >;

#define CUTLASS_SPLITK_K64_LAUNCH(DTYPE_TAG, TB_M, TB_N, STAGES, SLICES)                                \
    extern "C" int cutlass_gemm_##TB_M##x##TB_N##_k64_s##STAGES##_sk##SLICES##_##DTYPE_TAG##_launch(    \
        void* C, const void* A, const void* B,                                                          \
        int M, int N, int K,                                                                            \
        float alpha, float beta,                                                                        \
        void* workspace,                                                                                \
        uint64_t stream                                                                                 \
    ) {                                                                                                 \
        return run_gemm_splitk<GemmSplitK_##DTYPE_TAG##_##TB_M##x##TB_N##_k64_s##STAGES>(               \
            C, A, B, M, N, K, alpha, beta, SLICES, workspace, (cudaStream_t)stream);                    \
    }

#define CUTLASS_SPLITK_K64_TYPED(DTYPE_TAG, T, TB_M, TB_N, WARP_M, WARP_N, WARP_K, STAGES, SLICES) \
    CUTLASS_SPLITK_K64_CONFIG(DTYPE_TAG, T, TB_M, TB_N, WARP_M, WARP_N, WARP_K, STAGES)            \
    CUTLASS_SPLITK_K64_LAUNCH(DTYPE_TAG, TB_M, TB_N, STAGES, SLICES)

#define CUTLASS_SPLITK_K64(TB_M, TB_N, WARP_M, WARP_N, WARP_K, STAGES, SLICES)                     \
    CUTLASS_SPLITK_K64_TYPED(bf16, cutlass::bfloat16_t, TB_M, TB_N, WARP_M, WARP_N, WARP_K, STAGES, SLICES) \
    CUTLASS_SPLITK_K64_TYPED(f16,  cutlass::half_t,    TB_M, TB_N, WARP_M, WARP_N, WARP_K, STAGES, SLICES)

#define CUTLASS_SPLITK_K64_LAUNCH_ONLY(TB_M, TB_N, STAGES, SLICES)                                 \
    CUTLASS_SPLITK_K64_LAUNCH(bf16, TB_M, TB_N, STAGES, SLICES)                                    \
    CUTLASS_SPLITK_K64_LAUNCH(f16,  TB_M, TB_N, STAGES, SLICES)

// stages=2 variants — less SMEM, more occupancy on sm89
CUTLASS_GEMM( 64,   64,  32,    32,     32,     32,    2)
CUTLASS_GEMM( 64,  128,  32,    32,     64,     32,    2)
CUTLASS_GEMM(128,   64,  32,    64,     32,     32,    2)
CUTLASS_GEMM(128,  128,  32,    64,     32,     32,    2)
CUTLASS_GEMM(128,  256,  32,    64,     64,     32,    2)
CUTLASS_GEMM(256,   64,  32,    64,     32,     32,    2)

// Smaller M tiles for small-M regime (M=2-16). These waste less
// of the threadblock's M dimension when the batch is tiny.
CUTLASS_GEMM( 32,   64,  32,    32,     32,     32,    4)
CUTLASS_GEMM( 32,  128,  32,    32,     64,     32,    4)
CUTLASS_GEMM( 32,  256,  32,    32,     64,     32,    3)
CUTLASS_GEMM( 32,   64,  32,    32,     32,     32,    3)
CUTLASS_GEMM( 32,  128,  32,    32,     64,     32,    3)

// tile_m=16 — minimum threadblock-M with sm80 tensor cores
// (MMA m16n8k16 has m=16 floor; warp_m must be a multiple). Targets
// the prefill lm_head + small-batch QKV regime where M ∈ [8, 16] and
// the audit measured cuBLAS 5-7% ahead of the smallest existing
// tile_m=32 zoo. Each TB has 1 warp on M × 2 on N → 2 warps.
//
// `16×256` is excluded: CUTLASS's PitchLinearWarpRakedThreadMap<32,16>
// hits a div-by-zero at instantiation when ThreadblockShape::M=16 +
// ThreadblockShape::N=256 — the aspect is too extreme for the default
// shared-memory tile partitioning. `16×{64,128}` with warp `16×{32,64}`
// stay in the supported regime.
CUTLASS_GEMM( 16,   64,  32,    16,     32,     32,    4)
CUTLASS_GEMM( 16,  128,  32,    16,     64,     32,    4)
CUTLASS_GEMM( 16,   64,  32,    16,     32,     32,    3)
CUTLASS_GEMM( 16,  128,  32,    16,     64,     32,    3)

// ── SplitK GEMM — splits K-reduction across CTAs ──
//
// Critical for shapes where K >> M*N (e.g. down_proj at small batch).
// Uses GemmSplitKParallel which launches a GEMM grid + a reduction kernel.
// The split_k_slices parameter controls how many CTAs share the K dim.

// Workspace is supplied by the caller (ferrite's CachingAllocator).
// Contract for GemmSplitKParallel: workspace size is
//   split_k_slices * M * N * sizeof(ElementAccumulator)
// Here ElementAccumulator = float → 4 bytes/elem. The Rust wrapper
// computes the same formula when sizing its f32 scratch tensor.

template <typename GemmSplitK>
static int run_gemm_splitk(
    void* C, const void* A, const void* B,
    int M, int N, int K,
    float alpha, float beta,
    int split_k_slices,
    void* workspace,
    cudaStream_t stream
) {
    using T = typename GemmSplitK::ElementA;
    typename GemmSplitK::Arguments args(
        {M, N, K},
        {(T const*)A, K},
        {(T const*)B, K},
        {(T*)C, N},
        {(T*)C, N},
        {alpha, beta},
        split_k_slices
    );

    GemmSplitK op;
    auto status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;

    // Caller sized workspace from the (split_k, M, N) contract above;
    // CUTLASS's get_workspace_size matches this exactly for
    // GemmSplitKParallel. A short-K or tiny-M/N edge case could return
    // zero — the workspace pointer is then unused and ignored by
    // initialize.
    status = op.initialize(args, workspace);
    if (status != cutlass::Status::kSuccess) return -2;

    status = op(stream);
    return (status == cutlass::Status::kSuccess) ? 0 : -3;
}

// GemmSplitKParallel template order:
//   ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
//   Accumulator, OpClass, Arch, TBShape, WarpShape, InsnShape,
//   EpilogueOp, ConvertScaledOp, ReductionOp, Swizzle, Stages
#define CUTLASS_SPLITK_CONFIG(DTYPE_TAG, T, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES) \
    using GemmSplitK_##DTYPE_TAG##_##TB_M##x##TB_N##_s##STAGES =                              \
        cutlass::gemm::device::GemmSplitKParallel<                                            \
            T, cutlass::layout::RowMajor,                                                     \
            T, cutlass::layout::ColumnMajor,                                                  \
            T, cutlass::layout::RowMajor,                                                     \
            float,                                                                            \
            cutlass::arch::OpClassTensorOp,                                                   \
            cutlass::arch::Sm80,                                                              \
            cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>,                                      \
            cutlass::gemm::GemmShape<WARP_M, WARP_N, WARP_K>,                                \
            cutlass::gemm::GemmShape<16, 8, 16>,                                             \
            cutlass::epilogue::thread::LinearCombination<                                     \
                T, 8, float, float>,                                                          \
            cutlass::epilogue::thread::Convert<float, 8, float>,                              \
            cutlass::reduction::thread::ReduceAdd<                                            \
                float, float, 8>,                                                             \
            cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle,               \
            STAGES                                                                            \
        >;

#define CUTLASS_SPLITK_LAUNCH(DTYPE_TAG, TB_M, TB_N, STAGES, SLICES)                              \
    extern "C" int cutlass_gemm_##TB_M##x##TB_N##_s##STAGES##_sk##SLICES##_##DTYPE_TAG##_launch(  \
        void* C, const void* A, const void* B,                                                    \
        int M, int N, int K,                                                                      \
        float alpha, float beta,                                                                  \
        void* workspace,                                                                          \
        uint64_t stream                                                                           \
    ) {                                                                                           \
        return run_gemm_splitk<GemmSplitK_##DTYPE_TAG##_##TB_M##x##TB_N##_s##STAGES>(            \
            C, A, B, M, N, K, alpha, beta, SLICES, workspace, (cudaStream_t)stream);              \
    }

#define CUTLASS_SPLITK_TYPED(DTYPE_TAG, T, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES, SLICES) \
    CUTLASS_SPLITK_CONFIG(DTYPE_TAG, T, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES)            \
    CUTLASS_SPLITK_LAUNCH(DTYPE_TAG, TB_M, TB_N, STAGES, SLICES)

#define CUTLASS_SPLITK(TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES, SLICES)                     \
    CUTLASS_SPLITK_TYPED(bf16, cutlass::bfloat16_t, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES, SLICES) \
    CUTLASS_SPLITK_TYPED(f16,  cutlass::half_t,    TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES, SLICES)

#define CUTLASS_SPLITK_LAUNCH_ONLY(TB_M, TB_N, STAGES, SLICES)                                       \
    CUTLASS_SPLITK_LAUNCH(bf16, TB_M, TB_N, STAGES, SLICES)                                          \
    CUTLASS_SPLITK_LAUNCH(f16,  TB_M, TB_N, STAGES, SLICES)

// SplitK configs — 64×64 and 128×128 with 2, 4, 8 slices.
// These cover the K=8192 shapes where standard GEMM loses to cuBLAS.
CUTLASS_SPLITK( 64,  64, 32, 32, 32, 32, 4, 2)
CUTLASS_SPLITK_LAUNCH_ONLY(64, 64, 4, 4)
CUTLASS_SPLITK_LAUNCH_ONLY(64, 64, 4, 8)

CUTLASS_SPLITK(128, 128, 32, 64, 32, 32, 3, 2)
CUTLASS_SPLITK_LAUNCH_ONLY(128, 128, 3, 4)
CUTLASS_SPLITK_LAUNCH_ONLY(128, 128, 3, 8)

CUTLASS_SPLITK( 64, 128, 32, 32, 64, 32, 4, 2)
CUTLASS_SPLITK_LAUNCH_ONLY(64, 128, 4, 4)
CUTLASS_SPLITK_LAUNCH_ONLY(64, 128, 4, 8)

// 64×256 splitK — cuBLAS's #2 pick for M=128 K=8192 is 64×256 with sk8
CUTLASS_SPLITK( 64, 256, 32, 32, 64, 32, 3, 2)
CUTLASS_SPLITK_LAUNCH_ONLY(64, 256, 3, 4)
CUTLASS_SPLITK_LAUNCH_ONLY(64, 256, 3, 8)

CUTLASS_SPLITK( 32,  64, 32, 32, 32, 32, 4, 4)
CUTLASS_SPLITK_LAUNCH_ONLY(32, 64, 4, 8)
CUTLASS_SPLITK_LAUNCH_ONLY(32, 64, 4, 16)

// tile_m=16 splitK — small-M long-K regime (M=8 + K≥8192).
// Diagnostic on f44eb134a flagged 347 borderline picks losing to
// cuBLAS by 1-8% in this regime; existing splitK starts at tile_m=64
// (12.5% utilization at M=8) so these tiles fill the gap. Same warp
// shape as standalone CUTLASS_GEMM(16, 64/128, _, 16, _, _, 4) above.
CUTLASS_SPLITK( 16,  64, 32, 16, 32, 32, 4, 2)
CUTLASS_SPLITK_LAUNCH_ONLY(16, 64, 4, 4)
CUTLASS_SPLITK_LAUNCH_ONLY(16, 64, 4, 8)

CUTLASS_SPLITK( 16, 128, 32, 16, 64, 32, 4, 2)
CUTLASS_SPLITK_LAUNCH_ONLY(16, 128, 4, 4)
CUTLASS_SPLITK_LAUNCH_ONLY(16, 128, 4, 8)

// Additional splitK configs to close cuBLAS gaps at M=128/1024
CUTLASS_SPLITK(128, 128, 32, 64, 32, 32, 4, 2)
CUTLASS_SPLITK_LAUNCH_ONLY(128, 128, 4, 4)
CUTLASS_SPLITK_LAUNCH_ONLY(128, 128, 4, 8)

CUTLASS_SPLITK(128,  64, 32, 64, 32, 32, 4, 2)
CUTLASS_SPLITK_LAUNCH_ONLY(128, 64, 4, 4)
CUTLASS_SPLITK_LAUNCH_ONLY(128, 64, 4, 8)

CUTLASS_SPLITK(256,  64, 32, 64, 32, 32, 4, 2)
CUTLASS_SPLITK_LAUNCH_ONLY(256, 64, 4, 4)

CUTLASS_SPLITK( 64,  64, 32, 32, 32, 32, 4, 16)

CUTLASS_SPLITK( 64, 128, 32, 32, 64, 32, 4, 16)

// Deep pipeline splitK — cuBLAS uses 15 stages + sk4 for M=128 K=8192
// 64×128 splitK at higher stages: ~12KB/stage base + workspace
CUTLASS_SPLITK( 64, 128, 32, 32, 64, 32, 5, 4)
CUTLASS_SPLITK( 64, 128, 32, 32, 64, 32, 6, 4)
CUTLASS_SPLITK_LAUNCH_ONLY(64, 128, 6, 2)
CUTLASS_SPLITK_LAUNCH_ONLY(64, 128, 6, 8)
CUTLASS_SPLITK( 64, 128, 32, 32, 64, 32, 7, 4)
CUTLASS_SPLITK( 64, 128, 32, 32, 64, 32, 8, 4)
// 128×128 splitK at higher stages
CUTLASS_SPLITK(128, 128, 32, 64, 32, 32, 5, 4)
CUTLASS_SPLITK_LAUNCH_ONLY(128, 128, 5, 8)
// 128×64 splitK at higher stages
CUTLASS_SPLITK(128,  64, 32, 64, 32, 32, 5, 4)
CUTLASS_SPLITK(128,  64, 32, 64, 32, 32, 6, 4)

// k64 splitK: target M=128 K=8192 where 64×64_k64_s4 already did 28.6µs
CUTLASS_SPLITK_K64( 64,  64, 32, 32, 64, 4, 2)
CUTLASS_SPLITK_K64_LAUNCH_ONLY(64, 64, 4, 4)
CUTLASS_SPLITK_K64_LAUNCH_ONLY(64, 64, 4, 8)
CUTLASS_SPLITK_K64(128, 128, 64, 32, 64, 3, 2)
CUTLASS_SPLITK_K64_LAUNCH_ONLY(128, 128, 3, 4)

// ── CUTLASS GEMV (M=1 specialization) ──
//
// For M=1 decode, we don't have tensor-core coverage (MMA needs M>=16).
// CUTLASS's SIMT GEMV handles y[N,1] = A[N,K] @ x[K,1] efficiently.
//
// Our GEMM convention: C[M,N] = A[M,K] @ B[K,N]^T where B is column-major [K,N]
// (equivalently row-major [N,K]). At M=1, C is [1,N] — a row vector.
//
// We remap to GEMV as: y[N] = W[N,K] @ x[K] where W is the "A" of GEMV and
// our GEMM's B (row-major [N,K]) becomes GEMV's A (row-major [N,K]).
// Our GEMM's A[1,K] becomes GEMV's B (vector [K]).

// GEMV templated over T — bf16 and f16 both share the SIMT vector
// path on sm80+ (alignment 8, scalar accumulator float).
template <typename T>
using GemvKernel_8_T = cutlass::gemm::kernel::Gemv<
    T,                                 // ElementA (the weight matrix)
    cutlass::layout::RowMajor,          // LayoutA
    T,                                 // ElementB (the input vector)
    T,                                 // ElementC (the output vector)
    float,                              // ElementAccumulator
    cutlass::epilogue::thread::LinearCombination<
        T, 1, float, float>,
    8                                   // kElementsPerAccess (max for 16-bit vectorized load)
>;

#define CUTLASS_GEMV_LAUNCH(DTYPE_TAG, T)                                                         \
    extern "C" int cutlass_gemv_##DTYPE_TAG##_launch(                                             \
        void* C, const void* A, const void* B,                                                    \
        int M, int N, int K,                                                                      \
        float alpha, float beta,                                                                  \
        uint64_t stream                                                                           \
    ) {                                                                                           \
        /* Our GEMM: C[M,N] = A[M,K] @ B[K,N]^T, A row-major, B col-major (= row-major [N,K]).  */\
        /* We ignore M (assume M=1) and map to GEMV: y[N] = W[N,K] @ x[K] where W = our B.      */\
        if (M != 1) return -10;                                                                   \
        using Gemv = cutlass::gemm::device::Gemv<GemvKernel_8_T<T>>;                              \
        using TensorRefA = typename Gemv::GemvKernel::TensorRefA;                                 \
        TensorRefA ref_A{(T*)B, cutlass::layout::RowMajor(K)};                                    \
        typename Gemv::Arguments args(                                                            \
            cutlass::MatrixCoord{N, K},                                                           \
            1,                                                                                    \
            {alpha, beta},                                                                        \
            ref_A,                                                                                \
            A,                                                                                    \
            C,                                                                                    \
            C,                                                                                    \
            (int64_t)N * K,                                                                       \
            (int64_t)K,                                                                           \
            (int64_t)N,                                                                           \
            (int64_t)N                                                                            \
        );                                                                                        \
        Gemv op;                                                                                  \
        auto status = op.can_implement(args);                                                     \
        if (status != cutlass::Status::kSuccess) return -1;                                       \
        status = op.initialize(args, nullptr, (cudaStream_t)stream);                              \
        if (status != cutlass::Status::kSuccess) return -2;                                       \
        status = op((cudaStream_t)stream);                                                        \
        return (status == cutlass::Status::kSuccess) ? 0 : -3;                                    \
    }

CUTLASS_GEMV_LAUNCH(bf16, cutlass::bfloat16_t)
CUTLASS_GEMV_LAUNCH(f16,  cutlass::half_t)

// ── CUTLASS 3.x sm90 (Hopper) configs ──
//
// Uses wgmma + TMA via the CollectiveBuilder API.
// Only compiled when targeting sm_90+.
// CUTLASS 3.x sm90 requires CUDA >= 12.0 and C++20.
// The arch guard is handled by -arch=sm_90 in the build command;
// we only need the CUDA version check here.
#if (__CUDACC_VER_MAJOR__ >= 12)

#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp>
#include <cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp>
#include <cute/tensor.hpp>
#include <cutlass/util/packed_stride.hpp>

namespace sm90 {

using namespace cute;

using ElementAccumulator = float;

// A: RowMajor [M,K], B: ColumnMajor [K,N] (= RowMajor [N,K])
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

// Alignment — 8 elements = 16 bytes for both bf16 and f16.
static constexpr int AlignmentA = 8;
static constexpr int AlignmentB = 8;

// sm90 warp-specialized cooperative schedule — best for large tiles
using KernelScheduleCooperative = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueScheduleCooperative = cutlass::epilogue::TmaWarpSpecializedCooperative;

// sm90 warp-specialized schedule — better for smaller tiles
using KernelScheduleWS = cutlass::gemm::KernelTmaWarpSpecialized;
using EpilogueScheduleWS = cutlass::epilogue::TmaWarpSpecialized;

// sm90 pingpong schedule — overlaps compute and memory for better throughput
// Epilogue uses TmaWarpSpecialized (no separate pingpong epilogue exists).
using KernelSchedulePingpong = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using EpilogueSchedulePingpong = cutlass::epilogue::TmaWarpSpecialized;

// 2×1 cluster cooperative — same schedule, different cluster shape
using KernelScheduleCoop2x1 = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueScheduleCoop2x1 = cutlass::epilogue::TmaWarpSpecializedCooperative;


// ── Macro for sm90 configs ──
// Uses CollectiveBuilder to auto-configure TMA + wgmma mainloop.
// `ELEMENT` is the activation/output element type (`cutlass::bfloat16_t`
// or `cutlass::half_t`). One typedef family + one launch symbol per
// (DTYPE_TAG, tile, schedule) combo.

#define SM90_GEMM_CONFIG(DTYPE_TAG, ELEMENT, TILE_M, TILE_N, TILE_K, CLUSTER_M, CLUSTER_N, STAGES, SCHED_SUFFIX) \
    using TileShape_##DTYPE_TAG##_##TILE_M##x##TILE_N##_##SCHED_SUFFIX = Shape<_##TILE_M, _##TILE_N, _##TILE_K>; \
    using ClusterShape_##DTYPE_TAG##_##TILE_M##x##TILE_N##_##SCHED_SUFFIX = Shape<_##CLUSTER_M, _##CLUSTER_N, _1>; \
    \
    using CollectiveMainloop_##DTYPE_TAG##_##TILE_M##x##TILE_N##_##SCHED_SUFFIX = \
        typename cutlass::gemm::collective::CollectiveBuilder< \
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
            ELEMENT, LayoutA, AlignmentA, \
            ELEMENT, LayoutB, AlignmentB, \
            ElementAccumulator, \
            TileShape_##DTYPE_TAG##_##TILE_M##x##TILE_N##_##SCHED_SUFFIX, \
            ClusterShape_##DTYPE_TAG##_##TILE_M##x##TILE_N##_##SCHED_SUFFIX, \
            cutlass::gemm::collective::StageCountAutoCarveout< \
                static_cast<int>(sizeof(typename cutlass::epilogue::collective::CollectiveBuilder< \
                    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
                    TileShape_##DTYPE_TAG##_##TILE_M##x##TILE_N##_##SCHED_SUFFIX, \
                    ClusterShape_##DTYPE_TAG##_##TILE_M##x##TILE_N##_##SCHED_SUFFIX, \
                    cutlass::epilogue::collective::EpilogueTileAuto, \
                    ElementAccumulator, ElementAccumulator, \
                    ELEMENT, LayoutC, AlignmentA, \
                    ELEMENT, LayoutC, AlignmentA, \
                    EpilogueSchedule##SCHED_SUFFIX \
                >::CollectiveOp::SharedStorage))>, \
            KernelSchedule##SCHED_SUFFIX \
        >::CollectiveOp; \
    \
    using CollectiveEpilogue_##DTYPE_TAG##_##TILE_M##x##TILE_N##_##SCHED_SUFFIX = \
        typename cutlass::epilogue::collective::CollectiveBuilder< \
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, \
            TileShape_##DTYPE_TAG##_##TILE_M##x##TILE_N##_##SCHED_SUFFIX, \
            ClusterShape_##DTYPE_TAG##_##TILE_M##x##TILE_N##_##SCHED_SUFFIX, \
            cutlass::epilogue::collective::EpilogueTileAuto, \
            ElementAccumulator, ElementAccumulator, \
            ELEMENT, LayoutC, AlignmentA, \
            ELEMENT, LayoutC, AlignmentA, \
            EpilogueSchedule##SCHED_SUFFIX \
        >::CollectiveOp; \
    \
    using GemmKernel_sm90_##DTYPE_TAG##_##TILE_M##x##TILE_N##_##SCHED_SUFFIX = \
        cutlass::gemm::kernel::GemmUniversal< \
            Shape<int, int, int, int>, \
            CollectiveMainloop_##DTYPE_TAG##_##TILE_M##x##TILE_N##_##SCHED_SUFFIX, \
            CollectiveEpilogue_##DTYPE_TAG##_##TILE_M##x##TILE_N##_##SCHED_SUFFIX \
        >; \
    \
    using Gemm_sm90_##DTYPE_TAG##_##TILE_M##x##TILE_N##_##SCHED_SUFFIX = \
        cutlass::gemm::device::GemmUniversalAdapter< \
            GemmKernel_sm90_##DTYPE_TAG##_##TILE_M##x##TILE_N##_##SCHED_SUFFIX>;

#define SM90_GEMM_LAUNCH(DTYPE_TAG, ELEMENT, TILE_M, TILE_N, SCHED_SUFFIX, EXPORT_BASENAME) \
    extern "C" int EXPORT_BASENAME##_##DTYPE_TAG##_launch( \
        void* C, const void* A, const void* B, \
        int M, int N, int K, \
        float alpha, float beta, \
        uint64_t stream \
    ) { \
        using Gemm = Gemm_sm90_##DTYPE_TAG##_##TILE_M##x##TILE_N##_##SCHED_SUFFIX; \
        using StrideA = typename Gemm::GemmKernel::StrideA; \
        using StrideB = typename Gemm::GemmKernel::StrideB; \
        using StrideC = typename Gemm::GemmKernel::StrideC; \
        using StrideD = typename Gemm::GemmKernel::StrideD; \
        StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1)); \
        StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1)); \
        StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1)); \
        StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1)); \
        typename Gemm::Arguments args{ \
            cutlass::gemm::GemmUniversalMode::kGemm, \
            {M, N, K, 1}, \
            {(ELEMENT const*)A, stride_a, (ELEMENT const*)B, stride_b}, \
            {{alpha, beta}, (ELEMENT*)C, stride_c, (ELEMENT*)C, stride_d} \
        }; \
        Gemm op; \
        auto status = op.can_implement(args); \
        if (status != cutlass::Status::kSuccess) return -1; \
        size_t workspace_size = Gemm::get_workspace_size(args); \
        void* workspace = nullptr; \
        if (workspace_size > 0) { \
            cudaMalloc(&workspace, workspace_size); \
        } \
        status = op.initialize(args, workspace, (cudaStream_t)stream); \
        if (status != cutlass::Status::kSuccess) { \
            if (workspace) cudaFree(workspace); \
            return -2; \
        } \
        status = op((cudaStream_t)stream); \
        if (workspace) cudaFree(workspace); \
        return (status == cutlass::Status::kSuccess) ? 0 : -3; \
    }

#define SM90_GEMM_TYPED(DTYPE_TAG, ELEMENT, TILE_M, TILE_N, TILE_K, CLUSTER_M, CLUSTER_N, STAGES, SCHED_SUFFIX, EXPORT_BASENAME) \
    SM90_GEMM_CONFIG(DTYPE_TAG, ELEMENT, TILE_M, TILE_N, TILE_K, CLUSTER_M, CLUSTER_N, STAGES, SCHED_SUFFIX) \
    SM90_GEMM_LAUNCH(DTYPE_TAG, ELEMENT, TILE_M, TILE_N, SCHED_SUFFIX, EXPORT_BASENAME)

// `EXPORT_BASENAME` is the symbol stem (no `_launch` suffix); the
// macro appends `_<dtype>_launch` per instantiation.
#define SM90_GEMM(TILE_M, TILE_N, TILE_K, CLUSTER_M, CLUSTER_N, STAGES, SCHED_SUFFIX, EXPORT_BASENAME) \
    SM90_GEMM_TYPED(bf16, cutlass::bfloat16_t, TILE_M, TILE_N, TILE_K, CLUSTER_M, CLUSTER_N, STAGES, SCHED_SUFFIX, EXPORT_BASENAME) \
    SM90_GEMM_TYPED(f16,  cutlass::half_t,    TILE_M, TILE_N, TILE_K, CLUSTER_M, CLUSTER_N, STAGES, SCHED_SUFFIX, EXPORT_BASENAME)

// ── Instantiate sm90 configs ──
//
// Cooperative schedule — best for large tiles with cluster support.
// Warp-specialized — better for smaller tiles.
// Pingpong — overlaps compute and memory for better throughput.
// Stream-K — splits K across CTAs, crucial for small-M (decode).
//
// These use wgmma + TMA, the native H100 datapath.

// Large tiles with 1×1 cluster (safe default, always works)
SM90_GEMM(128, 128, 64, 1, 1, 0, Cooperative, cutlass_sm90_gemm_128x128_coop)
SM90_GEMM(128, 256, 64, 1, 1, 0, Cooperative, cutlass_sm90_gemm_128x256_coop)
SM90_GEMM(256, 128, 64, 1, 1, 0, Cooperative, cutlass_sm90_gemm_256x128_coop)

// Smaller tiles with warp-specialized schedule
SM90_GEMM( 64, 128, 64, 1, 1, 0, WS, cutlass_sm90_gemm_64x128_ws)
SM90_GEMM(128,  64, 64, 1, 1, 0, WS, cutlass_sm90_gemm_128x64_ws)
SM90_GEMM( 64,  64, 64, 1, 1, 0, WS, cutlass_sm90_gemm_64x64_ws)

// Pingpong schedule — overlaps compute and memory
SM90_GEMM(128, 128, 64, 1, 1, 0, Pingpong, cutlass_sm90_gemm_128x128_pp)
SM90_GEMM( 64, 128, 64, 1, 1, 0, Pingpong, cutlass_sm90_gemm_64x128_pp)
SM90_GEMM(128,  64, 64, 1, 1, 0, Pingpong, cutlass_sm90_gemm_128x64_pp)

// 2×1 cluster — doubles occupancy via distributed shared memory
SM90_GEMM(128, 128, 64, 2, 1, 0, Coop2x1, cutlass_sm90_gemm_128x128_c2x1)
SM90_GEMM(128, 256, 64, 2, 1, 0, Coop2x1, cutlass_sm90_gemm_128x256_c2x1)

} // namespace sm90

#endif // __CUDACC_VER_MAJOR__ >= 12

// Default-stages convenience aliases — shapes the solver uses by name
// (solver_dispatch.rs references these directly). Both bf16 and f16
// variants alias the matching `_s4` launcher.
extern "C" int cutlass_gemm_128x128_bf16_launch(
    void* C, const void* A, const void* B,
    int M, int N, int K, float alpha, float beta, uint64_t stream
) {
    return cutlass_gemm_128x128_s4_bf16_launch(C, A, B, M, N, K, alpha, beta, stream);
}
extern "C" int cutlass_gemm_128x128_f16_launch(
    void* C, const void* A, const void* B,
    int M, int N, int K, float alpha, float beta, uint64_t stream
) {
    return cutlass_gemm_128x128_s4_f16_launch(C, A, B, M, N, K, alpha, beta, stream);
}

extern "C" int cutlass_gemm_64x64_bf16_launch(
    void* C, const void* A, const void* B,
    int M, int N, int K, float alpha, float beta, uint64_t stream
) {
    return cutlass_gemm_64x64_s4_bf16_launch(C, A, B, M, N, K, alpha, beta, stream);
}
extern "C" int cutlass_gemm_64x64_f16_launch(
    void* C, const void* A, const void* B,
    int M, int N, int K, float alpha, float beta, uint64_t stream
) {
    return cutlass_gemm_64x64_s4_f16_launch(C, A, B, M, N, K, alpha, beta, stream);
}

// ── CUTLASS GEMM with SiLU epilogue ──
//
// D = silu(alpha * A[M,K] @ B[K,N]^T + beta * C[M,N])
//
// Drop-in replacement for the standard GEMM with LinearCombinationSiLU
// epilogue. Used for the Gate GEMM phase where the activation function
// is fused into the GEMM epilogue, saving one kernel launch + one
// full GMEM round-trip of the gate output.

#define CUTLASS_GEMM_SILU_CONFIG(DTYPE_TAG, T, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES) \
    using GemmSilu_##DTYPE_TAG##_##TB_M##x##TB_N##x##TB_K##_s##STAGES =                          \
        cutlass::gemm::device::Gemm<                                                              \
            T, cutlass::layout::RowMajor,                                                         \
            T, cutlass::layout::ColumnMajor,                                                      \
            T, cutlass::layout::RowMajor,                                                         \
            float,                                                                                \
            cutlass::arch::OpClassTensorOp,                                                       \
            cutlass::arch::Sm80,                                                                  \
            cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>,                                          \
            cutlass::gemm::GemmShape<WARP_M, WARP_N, WARP_K>,                                    \
            cutlass::gemm::GemmShape<16, 8, 16>,                                                  \
            cutlass::epilogue::thread::LinearCombinationSilu<                                     \
                T, 8, float, float>,                                                              \
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,                         \
            STAGES                                                                                \
        >;

#define CUTLASS_GEMM_SILU_LAUNCH(DTYPE_TAG, TB_M, TB_N, TB_K, STAGES)                             \
    extern "C" int cutlass_gemm_silu_##TB_M##x##TB_N##_s##STAGES##_##DTYPE_TAG##_launch(          \
        void* C, const void* A, const void* B,                                                    \
        int M, int N, int K,                                                                      \
        float alpha, float beta,                                                                  \
        uint64_t stream                                                                           \
    ) {                                                                                           \
        return run_gemm<GemmSilu_##DTYPE_TAG##_##TB_M##x##TB_N##x##TB_K##_s##STAGES>(             \
            C, A, B, M, N, K, alpha, beta, (cudaStream_t)stream);                                 \
    }

#define CUTLASS_GEMM_SILU_TYPED(DTYPE_TAG, T, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES)   \
    CUTLASS_GEMM_SILU_CONFIG(DTYPE_TAG, T, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES)      \
    CUTLASS_GEMM_SILU_LAUNCH(DTYPE_TAG, TB_M, TB_N, TB_K, STAGES)

#define CUTLASS_GEMM_SILU(TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES)                       \
    CUTLASS_GEMM_SILU_TYPED(bf16, cutlass::bfloat16_t, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES) \
    CUTLASS_GEMM_SILU_TYPED(f16,  cutlass::half_t,    TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES)

// Same tile configs as the standard GEMM — solver picks per workload.
CUTLASS_GEMM_SILU( 64,   64,  32,    32,     32,     32,    4)
CUTLASS_GEMM_SILU( 64,  128,  32,    32,     64,     32,    4)
CUTLASS_GEMM_SILU(128,   64,  32,    64,     32,     32,    4)
CUTLASS_GEMM_SILU(128,  128,  32,    64,     32,     32,    4)
CUTLASS_GEMM_SILU(128,  256,  32,    64,     64,     32,    3)
CUTLASS_GEMM_SILU(256,   64,  32,    64,     32,     32,    4)

CUTLASS_GEMM_SILU( 64,   64,  32,    32,     32,     32,    3)
CUTLASS_GEMM_SILU( 64,  128,  32,    32,     64,     32,    3)
CUTLASS_GEMM_SILU(128,   64,  32,    64,     32,     32,    3)
CUTLASS_GEMM_SILU(128,  128,  32,    64,     32,     32,    3)
CUTLASS_GEMM_SILU(256,   64,  32,    64,     32,     32,    3)

// Small M tiles for decode regime.
CUTLASS_GEMM_SILU( 32,   64,  32,    32,     32,     32,    4)
CUTLASS_GEMM_SILU( 32,  128,  32,    32,     64,     32,    4)
CUTLASS_GEMM_SILU( 32,  256,  32,    32,     64,     32,    3)
CUTLASS_GEMM_SILU( 32,   64,  32,    32,     32,     32,    3)
CUTLASS_GEMM_SILU( 32,  128,  32,    32,     64,     32,    3)

// ── Null kernel for measuring launch overhead ──
__global__ void null_kernel() {}

extern "C" void null_kernel_launch(uint64_t stream) {
    null_kernel<<<1, 1, 0, (cudaStream_t)stream>>>();
}
