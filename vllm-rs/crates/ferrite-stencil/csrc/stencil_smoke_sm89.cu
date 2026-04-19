// SPDX-License-Identifier: Apache-2.0
//
// Smoke kernel for ferrite-stencil, step 3b: a tiny but numerically
// correct FA2 prefill kernel for a fixed SEQ_Q × SEQ_K × HD shape,
// built from the SM89 prelude helpers the emitter will eventually
// reference. Single-CTA, scalar per-thread — the goal is "does a
// kernel built from prelude helpers produce the right answer?", not
// performance. Performance work lands at step 3d (cp.async) and
// beyond (mma.sync, multi-CTA, etc.).

#include "stencil_prelude_sm89.cuh"

using stencil_sm89::bf16;

// Fixed reference shape. Keeping it square + small makes the CPU
// reference in the accompanying Rust test trivially easy to verify.
// The shape is large enough to exercise the kv-tile loop (SEQ_K /
// TILE_K = 2) but small enough to run in a single CTA with a single
// thread carrying the softmax + accumulator state.
static constexpr int SEQ_Q  = 8;
static constexpr int SEQ_K  = 16;
static constexpr int HD     = 8;
static constexpr int TILE_Q = 8;
static constexpr int TILE_K = 8;

static __global__ void stencil_smoke_sm89_kernel(
    const bf16* __restrict__ q_gmem,   // [SEQ_Q,  HD]
    const bf16* __restrict__ k_gmem,   // [SEQ_K,  HD]
    const bf16* __restrict__ v_gmem,   // [SEQ_K,  HD]
    bf16* __restrict__ o_gmem          // [SEQ_Q,  HD]
) {
    __shared__ bf16  smem_q[TILE_Q][HD];
    __shared__ bf16  smem_k[TILE_K][HD];
    __shared__ bf16  smem_v[TILE_K][HD];
    __shared__ float s_frag[TILE_Q][TILE_K];
    __shared__ float p_frag[TILE_Q][TILE_K];
    __shared__ float o_frag[TILE_Q][HD];
    // Per-row softmax state, kept in smem so every thread sees the
    // same values when the softmax update runs on thread 0.
    __shared__ float m_state[TILE_Q];
    __shared__ float l_state[TILE_Q];

    if (threadIdx.x == 0) {
        for (int r = 0; r < TILE_Q; ++r) {
            m_state[r] = -INFINITY;
            l_state[r] = 0.0f;
            for (int d = 0; d < HD; ++d) o_frag[r][d] = 0.0f;
        }
    }
    __syncthreads();

    // ── preamble: load Q once ─────────────────────────────────────
    stencil_sm89::cp_async_128<TILE_Q, HD>(smem_q, q_gmem, 0, 0, HD);
    stencil_sm89::cp_async_commit_group();
    stencil_sm89::cp_async_wait_group(0);

    // ── body: iterate over kv tiles ───────────────────────────────
    const int num_kv_tiles = SEQ_K / TILE_K;
    for (int kv = 0; kv < num_kv_tiles; ++kv) {
        const uint32_t row = kv * TILE_K;

        stencil_sm89::cp_async_128<TILE_K, HD>(smem_k, k_gmem, row, 0, HD);
        stencil_sm89::cp_async_128<TILE_K, HD>(smem_v, v_gmem, row, 0, HD);
        stencil_sm89::cp_async_commit_group();
        stencil_sm89::cp_async_wait_group(0);

        // Zero S_frag, then S = Q @ K^T (TILE_Q × TILE_K).
        if (threadIdx.x == 0) {
            for (int r = 0; r < TILE_Q; ++r)
                for (int c = 0; c < TILE_K; ++c) s_frag[r][c] = 0.0f;
        }
        __syncthreads();
        stencil_sm89::mma_sync_accumulate<TILE_Q, TILE_K, HD>(
            s_frag, smem_q, smem_k);

        // Online softmax + accumulator update, per Q-row. Scalar on
        // thread 0 — 3d splits this across the warp.
        if (threadIdx.x == 0) {
            for (int r = 0; r < TILE_Q; ++r) {
                float prev_m = m_state[r];
                // row_max over this row of S_frag, seeded with prev_m.
                float m_new = prev_m;
                for (int c = 0; c < TILE_K; ++c) {
                    float v = s_frag[r][c];
                    m_new = v > m_new ? v : m_new;
                }
                float scale = prev_m == -INFINITY ? 0.0f
                                                  : expf(prev_m - m_new);
                // P_frag = expf(S - m_new)
                float sum_p = 0.0f;
                for (int c = 0; c < TILE_K; ++c) {
                    float p = expf(s_frag[r][c] - m_new);
                    p_frag[r][c] = p;
                    sum_p += p;
                }
                // Rescale O accumulator by `scale`, accumulate P @ V.
                for (int d = 0; d < HD; ++d) {
                    float acc = scale * o_frag[r][d];
                    for (int c = 0; c < TILE_K; ++c) {
                        acc += p_frag[r][c] * __bfloat162float(smem_v[c][d]);
                    }
                    o_frag[r][d] = acc;
                }
                l_state[r] = scale * l_state[r] + sum_p;
                m_state[r] = m_new;
            }
        }
        __syncthreads();
    }

    // ── epilogue: normalize by l and store ────────────────────────
    if (threadIdx.x == 0) {
        for (int r = 0; r < TILE_Q; ++r) {
            float inv_l = stencil_sm89::rcp(l_state[r] + 1e-9f);
            for (int d = 0; d < HD; ++d) o_frag[r][d] *= inv_l;
        }
    }
    __syncthreads();
    stencil_sm89::stg_128<TILE_Q, HD>(o_gmem, o_frag, 0, 0, HD);
}

// Host-side launch wrapper. Rust calls this; nvcc turns the <<<>>>
// syntax into the right cudaLaunchKernel sequence. Grid/block are
// hard-coded to match the single-CTA scalar layout the kernel body
// assumes; a configurable launch arrives with 3c.
extern "C" void ferrite_stencil_smoke_sm89_launch(
    const bf16* q,
    const bf16* k,
    const bf16* v,
    bf16* o,
    uint64_t stream_handle
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    stencil_smoke_sm89_kernel<<<1, 32, 0, stream>>>(q, k, v, o);
}
