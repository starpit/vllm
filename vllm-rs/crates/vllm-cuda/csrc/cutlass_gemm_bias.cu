// SPDX-License-Identifier: Apache-2.0
// CUTLASS 2.x BF16 GEMM with fused bias-broadcast epilogue.
//
// Computes: D[M, N] = A[M, K] @ B[N, K]^T + bias[N]
//
// where `bias` is a `[N]` vector broadcast across rows. Uses the
// CUTLASS 2.x EVT (Epilogue Visitor Tree) pattern — single kernel
// launch, bias folded into the epilogue register pass.
//
// Internal 3-way tile dispatch on `m` mirrors `cutlass_gemm_silu_mul.cu`.

#include "cutlass_scaled_mm/scaled_mm_c2x.cuh"
#include "cutlass_gemm_bias_epilogue.hpp"

using namespace vllm;

// ── Tile dispatch: pick the best tile for the problem size ──

static void dispatch_gemm_bias(
    void* d, const void* a, const void* b, const void* bias,
    int32_t m, int32_t n, int32_t k,
    cudaStream_t stream)
{
    using BF16 = cutlass::bfloat16_t;

    if (m <= 16) {
        using Gemm = cutlass_2x_gemm<
            cutlass::arch::Sm89, enable_sm89_to_sm100,
            BF16, BF16,
            c2x::BiasAddEpilogue,
            cutlass::gemm::GemmShape<32, 128, 32>,
            cutlass::gemm::GemmShape<32, 64, 32>,
            cutlass::gemm::GemmShape<16, 8, 16>,
            4>;
        cutlass_gemm_caller<Gemm>(
            d, a, b, m, n, k, k, k, n, stream,
            static_cast<const BF16*>(bias));
    } else if (m <= 64) {
        using Gemm = cutlass_2x_gemm<
            cutlass::arch::Sm89, enable_sm89_to_sm100,
            BF16, BF16,
            c2x::BiasAddEpilogue,
            cutlass::gemm::GemmShape<64, 128, 32>,
            cutlass::gemm::GemmShape<32, 64, 32>,
            cutlass::gemm::GemmShape<16, 8, 16>,
            4>;
        cutlass_gemm_caller<Gemm>(
            d, a, b, m, n, k, k, k, n, stream,
            static_cast<const BF16*>(bias));
    } else {
        using Gemm = cutlass_2x_gemm<
            cutlass::arch::Sm89, enable_sm89_to_sm100,
            BF16, BF16,
            c2x::BiasAddEpilogue,
            cutlass::gemm::GemmShape<128, 128, 32>,
            cutlass::gemm::GemmShape<64, 32, 32>,
            cutlass::gemm::GemmShape<16, 8, 16>,
            4>;
        cutlass_gemm_caller<Gemm>(
            d, a, b, m, n, k, k, k, n, stream,
            static_cast<const BF16*>(bias));
    }
}

// ── Extern "C" launch wrapper ──

extern "C" int cutlass_gemm_bias_launch(
    void* d,            // [M, N] bf16 output: A @ B^T + bias
    const void* a,      // [M, K] bf16 activation
    const void* b,      // [N, K] bf16 weight (row-major)
    const void* bias,   // [N] bf16 bias vector
    int M, int N, int K,
    uint64_t stream
) {
    dispatch_gemm_bias(
        d, a, b, bias,
        M, N, K,
        (cudaStream_t)stream);
    return 0;
}
