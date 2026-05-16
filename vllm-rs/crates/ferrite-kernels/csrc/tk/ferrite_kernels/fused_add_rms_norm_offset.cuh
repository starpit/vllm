// SPDX-License-Identifier: Apache-2.0
// Ferrite-owned TK 2.0 FusedAddRmsNormWithOffset op — 4 warp-role functions.
//
// Math (per token row):
//   sum         = delta_in + residual_in          (fp32)
//   residual_out = bf16(sum)                       (in-place on residual)
//   delta_out    = bf16(rms_norm(sum) * (weight + offset))   (in-place on delta)
//
// Gemma2 mid-layer pattern: the post-block residual-add feeds directly
// into the next pre-norm, which uses (1+w) weight convention.
// offset = 1.0 for Gemma2.
//
// Identical to `fused_add_rms_norm.cuh` except the consumer adds
// `offset` to each weight element after loading.
// Page layout, semaphores, bar IDs: identical to fused_add_rms_norm.cuh.

#pragma once

#include "kittens.cuh"
#include "ferrite_globals.cuh"
#include "ferrite_substrate.cuh"
#include "ferrite_tk_helpers.cuh"

namespace ferrite {
namespace ops {
namespace fused_add_rms_norm_offset {

constexpr int kDeltaPageOff    = 0;
constexpr int kResidualPageOff = 1;
constexpr int kWeightPageOff   = 2;

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

template <typename Config, int HIDDEN_DIM, int NUM_TOKENS>
__device__ __forceinline__ void consumer(
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int warp_in_role,
    float eps,
    float offset
) {
    constexpr int NUM_CONSUMER_WARPS = Config::NUM_CONSUMER_WARPS;
    constexpr int ELEMS_PER_WARP     = HIDDEN_DIM / NUM_CONSUMER_WARPS;
    static_assert(HIDDEN_DIM % NUM_CONSUMER_WARPS == 0,
                  "fused_add_rms_norm_offset: HIDDEN_DIM must be divisible by "
                  "NUM_CONSUMER_WARPS");
    static_assert(ELEMS_PER_WARP % 16 == 0,
                  "fused_add_rms_norm_offset: ELEMS_PER_WARP must be a multiple of 16");

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

        rv_t delta_vec, residual_vec, sum_vec;
        kittens::warp::load(delta_vec,    delta_slice);
        kittens::warp::load(residual_vec, residual_slice);
        kittens::warp::add(sum_vec, delta_vec, residual_vec);

        // residual_out = bf16(sum), in-place on residual page.
        kittens::warp::store(residual_slice, sum_vec);

        kittens::wait(ss.page_ready[base_stage + kWeightPageOff], phase);

        const float rms_scale = ferrite::tk::rms_norm_scale_from_rv<
            NUM_CONSUMER_WARPS, HIDDEN_DIM, kConsumerBarReduce>(
            sum_vec, eps, partial_sums_scratch);

        kittens::warp::mul(sum_vec, sum_vec, rms_scale);

        // Load weight and apply (weight + offset).
        rv_t weight_vec;
        kittens::warp::load(weight_vec, weight_slice);
        #pragma unroll
        for (int _o = 0; _o < weight_vec.outer_dim; ++_o) {
            #pragma unroll
            for (int _i = 0; _i < weight_vec.inner_dim; ++_i) {
                weight_vec.data[_o][_i] += offset;
            }
        }
        kittens::warp::mul(sum_vec, sum_vec, weight_vec);

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

}  // namespace fused_add_rms_norm_offset
}  // namespace ops
}  // namespace ferrite
