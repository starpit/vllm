// SPDX-License-Identifier: Apache-2.0
//
// Smoke kernel for ferrite-stencil CUDA build plumbing.
//
// Purpose at 3a: prove that nvcc can compile the stencil prelude +
// an emitter-shaped kernel end-to-end into a static library, and
// that the resulting symbol is visible to Rust. The body here is
// deliberately trivial — it references each prelude helper once so
// an accidental regression in the header is caught at build time,
// but does no real attention math. 3b replaces this with the
// hand-picked FA2 reference (or the emitter's output).

#include "stencil_prelude_sm89.cuh"

using stencil_sm89::bf16;

// Block shape matches the smallest profile we'll generate: tile_q=64,
// head_dim=128 / 2 per warpgroup (placeholder); scaled back to a
// trivial launch (8 threads) so the smoke test runs anywhere.
static constexpr int SMOKE_TQ = 8;
static constexpr int SMOKE_TK = 8;
static constexpr int SMOKE_HD = 8;

extern "C" __global__ void ferrite_stencil_smoke_sm89(
    const bf16* __restrict__ q_gmem,
    const bf16* __restrict__ k_gmem,
    const bf16* __restrict__ v_gmem,
    bf16* __restrict__ o_gmem,
    uint32_t row_stride
) {
    __shared__ bf16 smem_q[SMOKE_TQ][SMOKE_HD];
    __shared__ bf16 smem_k[SMOKE_TK][SMOKE_HD];
    __shared__ bf16 smem_v[SMOKE_TK][SMOKE_HD];
    __shared__ float s_frag[SMOKE_TQ][SMOKE_TK];
    __shared__ float p_frag[SMOKE_TQ][SMOKE_TK];
    __shared__ float o_frag[SMOKE_TQ][SMOKE_HD];

    if (threadIdx.x == 0) {
        for (int i = 0; i < SMOKE_TQ; ++i)
            for (int j = 0; j < SMOKE_TK; ++j) { s_frag[i][j] = 0.0f; p_frag[i][j] = 0.0f; }
        for (int i = 0; i < SMOKE_TQ; ++i)
            for (int j = 0; j < SMOKE_HD; ++j) o_frag[i][j] = 0.0f;
    }
    __syncthreads();

    stencil_sm89::cp_async_128<SMOKE_TQ, SMOKE_HD>(smem_q, q_gmem, 0, 0, row_stride);
    stencil_sm89::cp_async_128<SMOKE_TK, SMOKE_HD>(smem_k, k_gmem, 0, 0, row_stride);
    stencil_sm89::cp_async_128<SMOKE_TK, SMOKE_HD>(smem_v, v_gmem, 0, 0, row_stride);
    stencil_sm89::cp_async_commit_group();
    stencil_sm89::cp_async_wait_group(0);

    stencil_sm89::mma_sync_accumulate<SMOKE_TQ, SMOKE_TK, SMOKE_HD>(
        s_frag, smem_q, smem_k);

    if (threadIdx.x == 0) {
        float m = -INFINITY;
        m = stencil_sm89::row_max<SMOKE_TK>(s_frag, m);
        stencil_sm89::exp2f_frag<SMOKE_TK>(p_frag, s_frag, m);
        float l = stencil_sm89::row_sum<SMOKE_TK>(p_frag);
        float r = stencil_sm89::rcp(l + 1e-6f);
        for (int j = 0; j < SMOKE_HD; ++j) o_frag[0][j] = r;
    }
    __syncthreads();

    stencil_sm89::stg_128<SMOKE_TQ, SMOKE_HD>(o_gmem, o_frag, 0, 0, row_stride);
}
