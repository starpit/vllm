// SPDX-License-Identifier: Apache-2.0
// Ferrite-owned TK 2.0 Embed op — 4 warp-role functions.
//
// Math: `out[row, :] = embed_tokens[input_ids[row], :]` — a pure
// row-wise gather from the embedding table into the first
// activation slot. bf16 in/out, no accumulation.
//
// Parallelism (Phase 3 step 7): persistent-thread grid. Each
// role loops `for (int row = blockIdx.x; row < NUM_TOKENS; row +=
// gridDim.x)`, semaphore waits use `iter & 1`, __syncthreads at
// end of each iter fences page reuse. See rms_norm.cuh for the
// full pattern.
//
// Page layout (per embed call, caller gives `base_stage`):
//   pages[base_stage + 0] — output activation row
//                           (sv_bf<HIDDEN_DIM>), filled directly by
//                           the loader's cp.async.bulk.
//
// Semaphore handoff (init'd once in init_shared_state):
//   page_ready[base_stage + 0] — loader → consumer (output row
//                                landed).
//   page_done [base_stage + 0] — consumer → storer (handoff; the
//                                consumer does no mutation, just
//                                forwards the semaphore).

#pragma once

#include "kittens.cuh"
#include "ferrite_globals.cuh"
#include "ferrite_substrate.cuh"

namespace ferrite {
namespace ops {
namespace embed {

constexpr int kOutputPageOff = 0;

// ---------- Loader --------------------------------------------------

template <typename Config, int HIDDEN_DIM, int NUM_TOKENS>
__device__ __forceinline__ void loader(
    ferrite::u32_cptr input_ids,     // [NUM_TOKENS]
    ferrite::bf16_cptr embed_tokens, // [VOCAB_SIZE, HIDDEN_DIM]
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    using warp = kittens::group<1>;
    constexpr uint32_t row_bytes = HIDDEN_DIM * sizeof(__nv_bfloat16);

    void* out_page =
        reinterpret_cast<void*>(ss.pages[base_stage + kOutputPageOff]);

    for (int row = blockIdx.x; row < NUM_TOKENS; row += gridDim.x) {
        const uint32_t token_id = input_ids[row];

        if (kittens::laneid() == 0) {
            kittens::tma::expect_bytes(
                ss.page_ready[base_stage + kOutputPageOff], row_bytes);
        }
        warp::tma::load_async(
            out_page,
            const_cast<__nv_bfloat16*>(
                embed_tokens + static_cast<size_t>(token_id) * HIDDEN_DIM),
            row_bytes,
            ss.page_ready[base_stage + kOutputPageOff]);

        __syncthreads();
    }
}

// ---------- Consumer ------------------------------------------------

// Pure gather — no math. Warp 0 lane 0 waits then arrives page_done
// per iter. Other consumer warps still hit the per-iter
// __syncthreads so page reuse stays balanced.
template <typename Config, int HIDDEN_DIM, int NUM_TOKENS>
__device__ __forceinline__ void consumer(
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int warp_in_role
) {
    (void)ss; (void)base_stage;

    int iter = 0;
    for (int row = blockIdx.x; row < NUM_TOKENS; row += gridDim.x, ++iter) {
        (void)row;
        const int phase = iter & 1;

        if (warp_in_role == 0) {
            kittens::wait(ss.page_ready[base_stage + kOutputPageOff], phase);
            if (kittens::laneid() == 0) {
                kittens::arrive(ss.page_done[base_stage + kOutputPageOff]);
            }
        }
        __syncthreads();
    }
}

// ---------- Launcher ------------------------------------------------

template <typename Config, int HIDDEN_DIM, int NUM_TOKENS>
__device__ __forceinline__ void launcher(
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    (void)ss; (void)base_stage;
    for (int row = blockIdx.x; row < NUM_TOKENS; row += gridDim.x) {
        (void)row;
        __syncthreads();
    }
}

// ---------- Storer --------------------------------------------------

template <typename Config, int HIDDEN_DIM, int NUM_TOKENS>
__device__ __forceinline__ void storer(
    ferrite::bf16_ptr hidden_states_out,  // [NUM_TOKENS, HIDDEN_DIM]
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    using warp = kittens::group<1>;
    constexpr uint32_t row_bytes = HIDDEN_DIM * sizeof(__nv_bfloat16);

    int iter = 0;
    for (int row = blockIdx.x; row < NUM_TOKENS; row += gridDim.x, ++iter) {
        const int phase = iter & 1;
        kittens::wait(ss.page_done[base_stage + kOutputPageOff], phase);

        void* src_page = reinterpret_cast<void*>(
            ss.pages[base_stage + kOutputPageOff]);
        warp::tma::store_async(
            static_cast<void*>(
                hidden_states_out + static_cast<size_t>(row) * HIDDEN_DIM),
            src_page,
            row_bytes);
        kittens::tma::store_async_wait<0>();

        __syncthreads();
    }
}

}  // namespace embed
}  // namespace ops
}  // namespace ferrite
