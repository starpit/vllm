// SPDX-License-Identifier: Apache-2.0
//
// Stencil CUDA prelude, SM89 target.
//
// The `ferrite-stencil` emitter's per-op expansions reference helper
// functions like `cp_async_128`, `mma_sync_accumulate`, `row_max`,
// `stg_128`, and `exp2f_frag`. This header resolves those names into
// concrete `__device__` inlines so nvcc can compile what the emitter
// writes. First-cut implementations are deliberately simple — no
// pipelining, no mma.sync yet — so the plumbing lands before we start
// tuning.
//
// Scope at 3a: enough helpers that the smoke kernel (stencil_smoke_sm89.cu)
// compiles. Every helper has a docstring pointing to the op that
// references it; subsequent commits (3c, 3d) will replace these
// implementations with cp.async + mma.sync versions as needed.

#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace stencil_sm89 {

using bf16 = __nv_bfloat16;

// ── Load helpers ────────────────────────────────────────────────────
//
// Referenced by load_q_tile / load_kv_tile (SM89 branch). First
// version is a scalar gmem→smem copy with a __syncthreads fence;
// 3d upgrades this to cp.async.ca + cp_async_commit_group.

template <int ROWS, int COLS>
__device__ inline void cp_async_128(bf16 (*smem)[COLS], const bf16* gmem,
                                    uint32_t row_base, uint32_t col_base,
                                    uint32_t gmem_row_stride) {
    const uint32_t tid = threadIdx.x;
    const uint32_t nt  = blockDim.x;
    for (uint32_t idx = tid; idx < ROWS * COLS; idx += nt) {
        uint32_t r = idx / COLS;
        uint32_t c = idx % COLS;
        smem[r][c] = gmem[(row_base + r) * gmem_row_stride + (col_base + c)];
    }
}

__device__ inline void cp_async_commit_group() {
    // Placeholder: real impl issues `cp.async.commit_group;`. Without
    // cp.async the scalar copy above is already synchronous, so this
    // is a no-op — the companion `cp_async_wait_group` below pairs.
}

__device__ inline void cp_async_wait_group(int /*keep*/) {
    __syncthreads();
}

// ── Compute helpers ────────────────────────────────────────────────
//
// Referenced by qk_matmul / softmax_update / pv_matmul. 3a keeps
// these as scalar reference implementations per-row; 3d cuts over to
// mma.sync.m16n8k16 tiles.

template <int M, int N, int K>
__device__ inline void mma_sync_accumulate(float (*acc)[N], const bf16 (*a)[K],
                                           const bf16 (*b)[K]) {
    const uint32_t tid = threadIdx.x;
    for (uint32_t idx = tid; idx < (uint32_t)(M * N); idx += blockDim.x) {
        uint32_t m = idx / N;
        uint32_t n = idx % N;
        float s = 0.0f;
        for (int k = 0; k < K; ++k) {
            s += __bfloat162float(a[m][k]) * __bfloat162float(b[n][k]);
        }
        acc[m][n] += s;
    }
    __syncthreads();
}

template <int N>
__device__ inline float row_max(const float (*frag)[N], float prev_m) {
    float m = prev_m;
    #pragma unroll
    for (int j = 0; j < N; ++j) {
        float v = frag[0][j];
        m = v > m ? v : m;
    }
    return m;
}

template <int N>
__device__ inline float row_sum(const float (*frag)[N]) {
    float s = 0.0f;
    #pragma unroll
    for (int j = 0; j < N; ++j) s += frag[0][j];
    return s;
}

template <int N>
__device__ inline void exp2f_frag(float (*dst)[N], const float (*src)[N],
                                  float m) {
    #pragma unroll
    for (int j = 0; j < N; ++j) {
        dst[0][j] = exp2f(src[0][j] - m);
    }
}

__device__ inline float rcp(float x) { return __frcp_rn(x); }

// ── Store helpers ──────────────────────────────────────────────────
//
// Referenced by store_o_tile (SM89 branch). 3a is a straight gmem
// store; no TMA on this arch.

template <int ROWS, int COLS>
__device__ inline void stg_128(bf16* gmem, const float (*frag)[COLS],
                               uint32_t row_base, uint32_t col_base,
                               uint32_t gmem_row_stride) {
    const uint32_t tid = threadIdx.x;
    const uint32_t nt  = blockDim.x;
    for (uint32_t idx = tid; idx < ROWS * COLS; idx += nt) {
        uint32_t r = idx / COLS;
        uint32_t c = idx % COLS;
        gmem[(row_base + r) * gmem_row_stride + (col_base + c)] =
            __float2bfloat16_rn(frag[r][c]);
    }
}

} // namespace stencil_sm89
