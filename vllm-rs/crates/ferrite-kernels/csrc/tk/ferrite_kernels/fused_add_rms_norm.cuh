// SPDX-License-Identifier: Apache-2.0
// Ferrite-owned TK 2.0 FusedAddRmsNorm op — 4 warp-role functions.
//
// Math (per token row):
//   sum     = delta_in + residual_in            (fp32)
//   residual_out = bf16(sum)                    (in-place on residual)
//   delta_out    = bf16(rms_norm(sum) * weight) (in-place on delta)
// where `rms_norm(x)[i] = x[i] * rsqrt(mean(x^2) + eps)`.
//
// Parallelism (Phase 3 step 7): persistent-thread grid. Each
// role loops `for (int row = blockIdx.x; row < NUM_TOKENS; row +=
// gridDim.x)`, semaphore waits use `iter & 1` as the phase bit,
// `__syncthreads` at end of each iter fences page reuse. See
// rms_norm.cuh for the full pattern.
//
// Page layout (per call, caller gives `base_stage`):
//   pages[base_stage + 0] — delta tile (loader input, consumer
//                           overwrites with rms_norm output).
//   pages[base_stage + 1] — residual tile (loader input, consumer
//                           overwrites with delta + residual sum).
//   pages[base_stage + 2] — weight tile (loader input only).
//
// Semaphores:
//   page_ready[base_stage + 0..2] — loader → consumer (each input).
//   page_done [base_stage + 0]    — consumer → storer (delta out).
//   page_done [base_stage + 1]    — consumer → storer (residual out).
//
// bar IDs for consumer-scoped syncs: 5, 6 (1..4 are claimed by
// rms_norm + gemv/gemm).

#pragma once

#include "kittens.cuh"
#include "ferrite_globals.cuh"
#include "ferrite_substrate.cuh"
#include "ferrite_tk_helpers.cuh"

namespace ferrite {
namespace ops {
namespace fused_add_rms_norm {

constexpr int kDeltaPageOff    = 0;
constexpr int kResidualPageOff = 1;
constexpr int kWeightPageOff   = 2;

// Consumer-scoped bar.sync IDs. Picked to avoid collision with CTA-wide
// bar 0 (__syncthreads) and with rms_norm's bar 1/2 and gemv's bar 3/4.
//   kConsumerBarReduce  — cross-warp sum-of-squares reduction inside
//                         ferrite::tk::rms_norm_scale_from_rv.
//   kConsumerBarPublish — all consumer warps must have warp::store'd
//                         their slices (sum → residual page, normalised
//                         → delta page) before warp 0 arrives on
//                         page_done for the storer.
constexpr int kConsumerBarReduce  = 5;
constexpr int kConsumerBarPublish = 6;

// ---------- Loader --------------------------------------------------

template <typename Config, int HIDDEN_DIM, int NUM_TOKENS>
__device__ __forceinline__ void loader(
    ferrite::bf16_cptr delta_in,     // [NUM_TOKENS, HIDDEN_DIM]
    ferrite::bf16_cptr residual_in,  // [NUM_TOKENS, HIDDEN_DIM]
    ferrite::bf16_cptr rms_weight,   // [HIDDEN_DIM]
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    using warp = kittens::group<1>;
    constexpr uint32_t row_bytes = HIDDEN_DIM * sizeof(__nv_bfloat16);

    void* delta_page =
        reinterpret_cast<void*>(ss.pages[base_stage + kDeltaPageOff]);
    void* residual_page =
        reinterpret_cast<void*>(ss.pages[base_stage + kResidualPageOff]);
    void* weight_page =
        reinterpret_cast<void*>(ss.pages[base_stage + kWeightPageOff]);

    for (int row = blockIdx.x; row < NUM_TOKENS; row += gridDim.x) {
        if (kittens::laneid() == 0) {
            kittens::tma::expect_bytes(ss.page_ready[base_stage + kDeltaPageOff],    row_bytes);
            kittens::tma::expect_bytes(ss.page_ready[base_stage + kResidualPageOff], row_bytes);
            kittens::tma::expect_bytes(ss.page_ready[base_stage + kWeightPageOff],   row_bytes);
        }

        warp::tma::load_async(
            delta_page,
            const_cast<__nv_bfloat16*>(delta_in + static_cast<size_t>(row) * HIDDEN_DIM),
            row_bytes,
            ss.page_ready[base_stage + kDeltaPageOff]);

        warp::tma::load_async(
            residual_page,
            const_cast<__nv_bfloat16*>(residual_in + static_cast<size_t>(row) * HIDDEN_DIM),
            row_bytes,
            ss.page_ready[base_stage + kResidualPageOff]);

        warp::tma::load_async(
            weight_page,
            const_cast<__nv_bfloat16*>(rms_weight),
            row_bytes,
            ss.page_ready[base_stage + kWeightPageOff]);

        __syncthreads();
    }
}

// ---------- Consumer ------------------------------------------------

// Per-row fused residual-add + RMS norm on TK register tiles. Each of
// `NUM_CONSUMER_WARPS` consumer warps owns a contiguous
// `HIDDEN_DIM / NUM_CONSUMER_WARPS` slice of every row.
//
// Flow per iter:
//   1. wait for loader to fill delta + residual pages;
//   2. warp::load both slices into rv_fl, warp::add → sum rv;
//   3. warp::store the sum back to the residual page (residual_out in
//      bf16 — in-place);
//   4. wait for weight page;
//   5. call ferrite::tk::rms_norm_scale_from_rv on the sum rv (this
//      squares, sums, and does the cross-warp reduction via
//      BAR_ID==kConsumerBarReduce), returning rsqrt(mean + eps);
//   6. warp::mul sum by the scale and by the weight slice (loaded as
//      rv_fl from the weight page);
//   7. warp::store the result to the delta page (delta_out in bf16 —
//      in-place);
//   8. cross-warp BAR_ID==kConsumerBarPublish, then warp 0 arrives on
//      page_done for both delta and residual outputs.
//
// No __shfl_xor butterflies, no per-thread bf16 pointer walks.
template <typename Config, int HIDDEN_DIM, int NUM_TOKENS>
__device__ __forceinline__ void consumer(
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int warp_in_role,
    float eps
) {
    constexpr int NUM_CONSUMER_WARPS = Config::NUM_CONSUMER_WARPS;
    constexpr int ELEMS_PER_WARP     = HIDDEN_DIM / NUM_CONSUMER_WARPS;
    static_assert(HIDDEN_DIM % NUM_CONSUMER_WARPS == 0,
                  "fused_add_rms_norm: HIDDEN_DIM must be divisible by "
                  "NUM_CONSUMER_WARPS");
    static_assert(ELEMS_PER_WARP % 16 == 0,
                  "fused_add_rms_norm: ELEMS_PER_WARP must be a multiple of 16 "
                  "(TK sv_bf register/shared alignment)");

    using slice_sv = kittens::sv_bf<ELEMS_PER_WARP>;
    using rv_t     = kittens::rv_fl<ELEMS_PER_WARP>;

    auto *delta_slices    = reinterpret_cast<slice_sv *>(
        ss.pages[base_stage + kDeltaPageOff]);
    auto *residual_slices = reinterpret_cast<slice_sv *>(
        ss.pages[base_stage + kResidualPageOff]);
    auto *weight_slices   = reinterpret_cast<slice_sv *>(
        ss.pages[base_stage + kWeightPageOff]);
    slice_sv &delta_slice    = delta_slices[warp_in_role];
    slice_sv &residual_slice = residual_slices[warp_in_role];
    slice_sv &weight_slice   = weight_slices[warp_in_role];

    float *partial_sums_scratch = reinterpret_cast<float *>(ss.scratch);

    int iter = 0;
    for (int row = blockIdx.x; row < NUM_TOKENS; row += gridDim.x, ++iter) {
        (void)row;
        const int phase = iter & 1;

        kittens::wait(ss.page_ready[base_stage + kDeltaPageOff],    phase);
        kittens::wait(ss.page_ready[base_stage + kResidualPageOff], phase);

        // Step 2: delta + residual in registers.
        rv_t delta_vec, residual_vec, sum_vec;
        kittens::warp::load(delta_vec,    delta_slice);
        kittens::warp::load(residual_vec, residual_slice);
        kittens::warp::add(sum_vec, delta_vec, residual_vec);

        // Step 3: residual_out = bf16(sum), in-place on the residual page.
        kittens::warp::store(residual_slice, sum_vec);

        // Step 4: weight page must be resident before we load it.
        kittens::wait(ss.page_ready[base_stage + kWeightPageOff], phase);

        // Step 5: cross-warp RMS scale from the sum rv.
        const float rms_scale = ferrite::tk::rms_norm_scale_from_rv<
            NUM_CONSUMER_WARPS, HIDDEN_DIM, kConsumerBarReduce>(
            sum_vec, eps, partial_sums_scratch);

        // Step 6: delta_out_rv = sum * rms_scale * weight.
        kittens::warp::mul(sum_vec, sum_vec, rms_scale);
        rv_t weight_vec;
        kittens::warp::load(weight_vec, weight_slice);
        kittens::warp::mul(sum_vec, sum_vec, weight_vec);

        // Step 7: delta_out written to the delta page.
        kittens::warp::store(delta_slice, sum_vec);

        kittens::group<NUM_CONSUMER_WARPS>::sync(kConsumerBarPublish);
        if (warp_in_role == 0 && kittens::laneid() == 0) {
            kittens::arrive(ss.page_done[base_stage + kDeltaPageOff]);
            kittens::arrive(ss.page_done[base_stage + kResidualPageOff]);
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
    ferrite::bf16_ptr delta_out,     // [NUM_TOKENS, HIDDEN_DIM]
    ferrite::bf16_ptr residual_out,  // [NUM_TOKENS, HIDDEN_DIM]
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    using warp = kittens::group<1>;
    constexpr uint32_t row_bytes = HIDDEN_DIM * sizeof(__nv_bfloat16);

    int iter = 0;
    for (int row = blockIdx.x; row < NUM_TOKENS; row += gridDim.x, ++iter) {
        const int phase = iter & 1;
        kittens::wait(ss.page_done[base_stage + kDeltaPageOff], phase);
        kittens::wait(ss.page_done[base_stage + kResidualPageOff], phase);

        void* delta_src = reinterpret_cast<void*>(
            ss.pages[base_stage + kDeltaPageOff]);
        void* residual_src = reinterpret_cast<void*>(
            ss.pages[base_stage + kResidualPageOff]);

        warp::tma::store_async(
            static_cast<void*>(delta_out + static_cast<size_t>(row) * HIDDEN_DIM),
            delta_src,
            row_bytes);
        warp::tma::store_async(
            static_cast<void*>(residual_out + static_cast<size_t>(row) * HIDDEN_DIM),
            residual_src,
            row_bytes);

        kittens::tma::store_async_wait<0>();

        __syncthreads();
    }
}

}  // namespace fused_add_rms_norm
}  // namespace ops
}  // namespace ferrite
