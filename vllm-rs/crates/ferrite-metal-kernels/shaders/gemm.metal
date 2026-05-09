// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ---------------------------------------------------------------------------
// gemm_bf16_specialized
// ---------------------------------------------------------------------------
//
// Generic dense GEMM for bf16 inputs:
//
//     C = A @ B^T,   A: [M, K], B: [N, K] (transposed-right), C: [M, N]
//
// Hardcodes the "Linear layer" convention used by every Llama / Qwen /
// Mistral GEMM in the metal backend: `transpose_a = false`,
// `transpose_b = true`, `alpha = 1.0`, `beta = 0.0`. The runtime never
// invokes the more general MPS surface, so baking these in lets the
// shader stay short and the function-constant bag stay at three
// dimensions.
//
// Why a custom kernel: `MPSMatrixMultiplication` only accepts
// `MPSDataTypeFloat32`, `MPSDataTypeFloat16`, `MPSDataTypeInt8`,
// `MPSDataTypeInt16` (asserted at runtime — see
// `MPSMatrixMultiplication.mm:3260`). Even though the hardware on
// Apple Silicon M3+ has native bf16 MMA and `MPSDataTypeBFloat16`
// is a valid MPSCore type, the matmul kernel itself doesn't take
// it. We use `simdgroup_bfloat8x8` (typedef of
// `simdgroup_matrix<bfloat, 8, 8>`, available since Metal 3.1) which
// drives the same hardware MMA from a compute kernel.
//
// Bindings (must match `interpreter::metal::lowering::lower_one`'s
// `Instruction::Gemm` arm — same shape as MPS path: out, in, weight):
//   buffer(0) = output  [M, N]
//   buffer(1) = input   [M, K]
//   buffer(2) = weight  [N, K]
//
// Function constants:
//   0 = M
//   1 = N
//   2 = K
//
// Dispatch: threadgroups (ceil(N/8), ceil(M/8), 1), threads (32, 1, 1).
// One simdgroup per threadgroup; each simdgroup computes an 8×8
// output tile.
// ---------------------------------------------------------------------------

constant uint GEMM_M [[function_constant(0)]];
constant uint GEMM_N [[function_constant(1)]];
constant uint GEMM_K [[function_constant(2)]];

kernel void gemm_bf16_specialized(
    device       bfloat* output [[buffer(0)]],
    device const bfloat* input  [[buffer(1)]],
    device const bfloat* weight [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]])
{
    const uint tid = tid3.x;
    constexpr uint TILE = 8u;

    const uint m_base = tgid.y * TILE;
    const uint n_base = tgid.x * TILE;
    if (m_base >= GEMM_M || n_base >= GEMM_N) return;

    const uint M = GEMM_M;
    const uint N = GEMM_N;
    const uint K = GEMM_K;

    const bool m_full = (m_base + TILE <= M);
    const bool n_full = (n_base + TILE <= N);

    simdgroup_float8x8 acc = simdgroup_float8x8(0.0f);

    threadgroup bfloat a_pad[TILE * TILE];
    threadgroup bfloat b_pad[TILE * TILE];

    for (uint k_base = 0u; k_base < K; k_base += TILE) {
        const bool k_full = (k_base + TILE <= K);

        simdgroup_bfloat8x8 A;
        simdgroup_bfloat8x8 B;

        if (m_full && k_full) {
            simdgroup_load(A, input + m_base * K + k_base, K);
        } else {
            for (uint t = tid; t < TILE * TILE; t += 32u) {
                uint r = t / TILE;
                uint c = t % TILE;
                uint mr = m_base + r;
                uint kc = k_base + c;
                a_pad[r * TILE + c] = (mr < M && kc < K)
                    ? input[mr * K + kc]
                    : bfloat(0);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(A, a_pad, TILE);
        }

        // weight is [N, K]; the kernel computes C = A @ W^T, so we
        // load each B fragment with `transpose=true` to fetch the
        // [TILE_K, TILE_N] view of the [TILE_N, TILE_K] memory slice.
        if (n_full && k_full) {
            simdgroup_load(B, weight + n_base * K + k_base, K, ulong2(0, 0), true);
        } else {
            for (uint t = tid; t < TILE * TILE; t += 32u) {
                uint r = t / TILE;
                uint c = t % TILE;
                uint nr = n_base + r;
                uint kc = k_base + c;
                b_pad[r * TILE + c] = (nr < N && kc < K)
                    ? weight[nr * K + kc]
                    : bfloat(0);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(B, b_pad, TILE, ulong2(0, 0), true);
        }
        simdgroup_multiply_accumulate(acc, A, B, acc);
    }

    threadgroup float c_scratch[TILE * TILE];
    simdgroup_store(acc, c_scratch, TILE);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    if (m_full && n_full) {
        for (uint t = tid; t < TILE * TILE; t += 32u) {
            uint r = t / TILE;
            uint c = t % TILE;
            output[(m_base + r) * N + (n_base + c)] = bfloat(c_scratch[r * TILE + c]);
        }
    } else {
        for (uint t = tid; t < TILE * TILE; t += 32u) {
            uint r = t / TILE;
            uint c = t % TILE;
            uint mr = m_base + r;
            uint nc = n_base + c;
            if (mr < M && nc < N) {
                output[mr * N + nc] = bfloat(c_scratch[r * TILE + c]);
            }
        }
    }
}
