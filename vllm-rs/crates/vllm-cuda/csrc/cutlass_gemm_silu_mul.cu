// SPDX-License-Identifier: Apache-2.0
// CUTLASS 2.x GEMM with fused SiLU + element-wise multiply epilogue —
// templatized over T (bf16 / f16). Both share the EVT visitor tree.
//
// Computes: D[M,N] = silu(A[M,K] @ B_gate[N,K]^T) * C_up[M,N]
//
// Uses the CUTLASS 2.x EVT (Epilogue Visitor Tree) pattern.
// Claims [GemmGate, GateUpConcat, SiluMul] in the solver.

#include "cutlass_scaled_mm/scaled_mm_c2x.cuh"
#include "cutlass_silu_mul_epilogue.hpp"

using namespace vllm;

// ── Tile dispatch: pick the best tile for the problem size ──

template <typename T>
static void dispatch_silu_mul_gemm_t(
    void* d, const void* a, const void* b, void* up,
    int32_t m, int32_t n, int32_t k,
    cudaStream_t stream)
{
    if (m <= 16) {
        using Gemm = cutlass_2x_gemm<
            cutlass::arch::Sm89, enable_sm89_to_sm100,
            T, T,
            c2x::SiluMulEpilogue,
            cutlass::gemm::GemmShape<32, 128, 32>,
            cutlass::gemm::GemmShape<32, 64, 32>,
            cutlass::gemm::GemmShape<16, 8, 16>,
            4>;
        cutlass_gemm_caller<Gemm>(
            d, a, b, m, n, k, k, k, n, stream,
            static_cast<T*>(up), n);
    } else if (m <= 64) {
        using Gemm = cutlass_2x_gemm<
            cutlass::arch::Sm89, enable_sm89_to_sm100,
            T, T,
            c2x::SiluMulEpilogue,
            cutlass::gemm::GemmShape<64, 128, 32>,
            cutlass::gemm::GemmShape<32, 64, 32>,
            cutlass::gemm::GemmShape<16, 8, 16>,
            4>;
        cutlass_gemm_caller<Gemm>(
            d, a, b, m, n, k, k, k, n, stream,
            static_cast<T*>(up), n);
    } else {
        using Gemm = cutlass_2x_gemm<
            cutlass::arch::Sm89, enable_sm89_to_sm100,
            T, T,
            c2x::SiluMulEpilogue,
            cutlass::gemm::GemmShape<128, 128, 32>,
            cutlass::gemm::GemmShape<64, 32, 32>,
            cutlass::gemm::GemmShape<16, 8, 16>,
            4>;
        cutlass_gemm_caller<Gemm>(
            d, a, b, m, n, k, k, k, n, stream,
            static_cast<T*>(up), n);
    }
}

// ── Extern "C" launch wrappers ──

#define CUTLASS_SILU_MUL_LAUNCH(DTYPE_TAG, T)                                                  \
    extern "C" int cutlass_gemm_silu_mul_##DTYPE_TAG##_launch(                                 \
        void* d,            /* [M, N] T output: silu(gate) * up */                             \
        const void* a,      /* [M, K] T normed hidden states */                                \
        const void* b_gate, /* [N, K] T gate weight (row-major) */                             \
        void* c_up,         /* [M, N] T up-projection output (aux load) */                     \
        int M, int N, int K,                                                                   \
        uint64_t stream                                                                        \
    ) {                                                                                        \
        dispatch_silu_mul_gemm_t<T>(                                                           \
            d, a, b_gate, c_up,                                                                \
            M, N, K,                                                                           \
            (cudaStream_t)stream);                                                             \
        return 0;                                                                              \
    }

CUTLASS_SILU_MUL_LAUNCH(bf16, cutlass::bfloat16_t)
CUTLASS_SILU_MUL_LAUNCH(f16,  cutlass::half_t)
