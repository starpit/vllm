// SPDX-License-Identifier: Apache-2.0
// Standalone CUTLASS 2.x bf16 GEMM launchers for the solver.
// Generates one kernel + launch wrapper per (TB_M, TB_N, WARP_M, WARP_N, STAGES) config.
//
// C[M,N] = alpha * A[M,K] @ B[K,N]^T + beta * C[M,N]
// A: RowMajor bf16, B: ColumnMajor bf16, C: RowMajor bf16

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemv.h>
#include <cutlass/gemm/kernel/gemv.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cuda_runtime.h>

// ── Generic launch ──

template <typename GemmOp>
static int run_gemm(
    void* C, const void* A, const void* B,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream
) {
    typename GemmOp::Arguments args(
        {M, N, K},
        {(cutlass::bfloat16_t const*)A, K},   // A [M,K] row-major, lda=K
        {(cutlass::bfloat16_t const*)B, K},   // B [N,K] col-major, ldb=K
        {(cutlass::bfloat16_t*)C, N},          // C [M,N] row-major, ldc=N
        {(cutlass::bfloat16_t*)C, N},          // D = C (in-place for beta!=0)
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

#define CUTLASS_GEMM_CONFIG(TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES)     \
    using Gemm_##TB_M##x##TB_N##x##TB_K##_s##STAGES = cutlass::gemm::device::Gemm< \
        cutlass::bfloat16_t, cutlass::layout::RowMajor,                             \
        cutlass::bfloat16_t, cutlass::layout::ColumnMajor,                          \
        cutlass::bfloat16_t, cutlass::layout::RowMajor,                             \
        float,                                                                      \
        cutlass::arch::OpClassTensorOp,                                             \
        cutlass::arch::Sm80,                                                        \
        cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>,                                \
        cutlass::gemm::GemmShape<WARP_M, WARP_N, WARP_K>,                          \
        cutlass::gemm::GemmShape<16, 8, 16>,                                       \
        cutlass::epilogue::thread::LinearCombination<                               \
            cutlass::bfloat16_t, 8, float, float>,                                  \
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,               \
        STAGES                                                                      \
    >;

#define CUTLASS_GEMM_LAUNCH(TB_M, TB_N, TB_K, STAGES)                              \
    extern "C" int cutlass_gemm_##TB_M##x##TB_N##_s##STAGES##_launch(               \
        void* C, const void* A, const void* B,                                      \
        int M, int N, int K,                                                        \
        float alpha, float beta,                                                    \
        uint64_t stream                                                             \
    ) {                                                                             \
        return run_gemm<Gemm_##TB_M##x##TB_N##x##TB_K##_s##STAGES>(                \
            C, A, B, M, N, K, alpha, beta, (cudaStream_t)stream);                   \
    }

#define CUTLASS_GEMM(TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES)            \
    CUTLASS_GEMM_CONFIG(TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES)         \
    CUTLASS_GEMM_LAUNCH(TB_M, TB_N, TB_K, STAGES)

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

// Smaller M tiles for small-M regime (M=2-16). These waste less
// of the threadblock's M dimension when the batch is tiny.
CUTLASS_GEMM( 32,   64,  32,    32,     32,     32,    4)
CUTLASS_GEMM( 32,  128,  32,    32,     64,     32,    4)
CUTLASS_GEMM( 32,  256,  32,    32,     64,     32,    3)
CUTLASS_GEMM( 32,   64,  32,    32,     32,     32,    3)
CUTLASS_GEMM( 32,  128,  32,    32,     64,     32,    3)

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

using GemvKernel_8 = cutlass::gemm::kernel::Gemv<
    cutlass::bfloat16_t,              // ElementA (the weight matrix)
    cutlass::layout::RowMajor,         // LayoutA
    cutlass::bfloat16_t,              // ElementB (the input vector)
    cutlass::bfloat16_t,              // ElementC (the output vector)
    float,                             // ElementAccumulator
    cutlass::epilogue::thread::LinearCombination<
        cutlass::bfloat16_t, 1, float, float>,
    8                                  // kElementsPerAccess (max for bf16 vectorized load)
>;
using GemvOp_8 = cutlass::gemm::device::Gemv<GemvKernel_8>;

extern "C" int cutlass_gemv_launch(
    void* C, const void* A, const void* B,
    int M, int N, int K,
    float alpha, float beta,
    uint64_t stream
) {
    // Our GEMM: C[M,N] = A[M,K] @ B[K,N]^T, A row-major, B col-major (= row-major [N,K])
    // We ignore M (assume M=1) and map to GEMV:
    //   y[N] = W[N,K] @ x[K]
    //   W = our B (row-major [N,K]) → GEMV's ref_A
    //   x = our A (the M=1 input row) → GEMV's ptr_B
    //   y = our C (output row) → GEMV's ptr_D
    if (M != 1) return -10;  // GEMV only valid at M=1

    using Gemv = GemvOp_8;
    using TensorRefA = typename Gemv::GemvKernel::TensorRefA;

    TensorRefA ref_A{(cutlass::bfloat16_t*)B, cutlass::layout::RowMajor(K)};
    typename Gemv::Arguments args(
        cutlass::MatrixCoord{N, K},                                // problem_size {rows, cols}
        1,                                                          // batch_count
        {alpha, beta},                                              // epilogue params
        ref_A,                                                      // ref_A (weight)
        A,                                                          // ptr_B (input vector)
        C,                                                          // ptr_C
        C,                                                          // ptr_D
        (int64_t)N * K,                                             // batch_stride_A
        (int64_t)K,                                                 // batch_stride_B
        (int64_t)N,                                                 // batch_stride_C
        (int64_t)N                                                  // batch_stride_D
    );

    Gemv op;
    auto status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;

    status = op.initialize(args, nullptr, (cudaStream_t)stream);
    if (status != cutlass::Status::kSuccess) return -2;

    status = op((cudaStream_t)stream);
    return (status == cutlass::Status::kSuccess) ? 0 : -3;
}

// Keep the old symbol names as aliases for backward compatibility
// (solver_dispatch.rs references these directly).
extern "C" int cutlass_gemm_128x128_launch(
    void* C, const void* A, const void* B,
    int M, int N, int K, float alpha, float beta, uint64_t stream
) {
    return cutlass_gemm_128x128_s4_launch(C, A, B, M, N, K, alpha, beta, stream);
}

extern "C" int cutlass_gemm_64x64_launch(
    void* C, const void* A, const void* B,
    int M, int N, int K, float alpha, float beta, uint64_t stream
) {
    return cutlass_gemm_64x64_s4_launch(C, A, B, M, N, K, alpha, beta, stream);
}
