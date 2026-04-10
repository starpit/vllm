// SPDX-License-Identifier: Apache-2.0
// Standalone CUTLASS 2.x bf16 GEMM launcher for the CP5 interpreter.
// Two tile configs: 128×128×32 and 64×64×32.
//
// C[M,N] = alpha * A[M,K] @ B[K,N]^T + beta * C[M,N]
// A: RowMajor bf16, B: ColumnMajor bf16, C: RowMajor bf16

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cuda_runtime.h>

// ── 128×128×32 tile (matches pfl_cutlass_small) ──
using Gemm128 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,     // A
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,   // B
    cutlass::bfloat16_t, cutlass::layout::RowMajor,      // C
    float,                                                // accumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,               // threadblock
    cutlass::gemm::GemmShape<64, 32, 32>,                  // warp
    cutlass::gemm::GemmShape<16, 8, 16>,                   // instruction
    cutlass::epilogue::thread::LinearCombination<
        cutlass::bfloat16_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4  // stages
>;

// ── 64×64×32 tile ──
using Gemm64 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        cutlass::bfloat16_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
>;

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

extern "C" {

int cutlass_gemm_128x128_launch(
    void* C, const void* A, const void* B,
    int M, int N, int K,
    float alpha, float beta,
    uint64_t stream
) {
    return run_gemm<Gemm128>(C, A, B, M, N, K, alpha, beta, (cudaStream_t)stream);
}

int cutlass_gemm_64x64_launch(
    void* C, const void* A, const void* B,
    int M, int N, int K,
    float alpha, float beta,
    uint64_t stream
) {
    return run_gemm<Gemm64>(C, A, B, M, N, K, alpha, beta, (cudaStream_t)stream);
}

}  // extern "C"
