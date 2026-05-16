// SPDX-License-Identifier: Apache-2.0
// Ferrite-owned TK 2.0 ScalarOffsetRmsNorm op — 4 warp-role functions.
//
// Math: `out[row, :] = (weight + offset) * input[row, :] * rsqrt(mean(input^2) + eps)`
// Gemma2 uses offset=1.0 (their weight convention is 1+w). bf16 in/out, fp32 accum.
//
// Identical to `rms_norm.cuh` except the consumer adds `offset` to each
// weight element after loading it into a register vector, implementing
// the `(1 + w)` normalizer pattern without a separate pre-add kernel.
//
// Page layout, semaphores, bar IDs: identical to rms_norm.cuh.

#pragma once

#include "kittens.cuh"
#include "ferrite_globals.cuh"
#include "ferrite_substrate.cuh"
#include "ferrite_tk_helpers.cuh"

namespace ferrite {
namespace ops {
namespace rms_norm_offset {

constexpr int kInputPageOff  = 0;
constexpr int kWeightPageOff = 1;

constexpr int kConsumerBarReduce  = 1;
constexpr int kConsumerBarPublish = 2;

// ---------- Loader --------------------------------------------------

template <typename Config, int HIDDEN_DIM, int NUM_TOKENS>
__device__ __forceinline__ void loader(
    ferrite::bf16_cptr rms_input,   // [NUM_TOKENS, HIDDEN_DIM]
    ferrite::bf16_cptr rms_weight,  // [HIDDEN_DIM]
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    using warp = kittens::group<1>;
    constexpr uint32_t row_bytes = HIDDEN_DIM * sizeof(__nv_bfloat16);

    void* input_page =
        reinterpret_cast<void*>(ss.pages[base_stage + kInputPageOff]);
    void* weight_page =
        reinterpret_cast<void*>(ss.pages[base_stage + kWeightPageOff]);

    int iter = 0;
    for (int row = blockIdx.x; row < NUM_TOKENS; row += gridDim.x, ++iter) {
        (void)iter;
        if (kittens::laneid() == 0) {
            kittens::tma::expect_bytes(ss.page_ready[base_stage + kInputPageOff], row_bytes);
        }
        warp::tma::load_async(
            input_page,
            const_cast<__nv_bfloat16*>(rms_input + static_cast<size_t>(row) * HIDDEN_DIM),
            row_bytes,
            ss.page_ready[base_stage + kInputPageOff]);

        if (kittens::laneid() == 0) {
            kittens::tma::expect_bytes(ss.page_ready[base_stage + kWeightPageOff], row_bytes);
        }
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
                  "rms_norm_offset: HIDDEN_DIM must be divisible by NUM_CONSUMER_WARPS");
    static_assert(ELEMS_PER_WARP % 16 == 0,
                  "rms_norm_offset: ELEMS_PER_WARP must be a multiple of 16");

    using slice_sv = kittens::sv_bf<ELEMS_PER_WARP>;
    using rv_t     = kittens::rv_fl<ELEMS_PER_WARP>;

    auto *input_slices  = reinterpret_cast<slice_sv *>(
        ss.pages[base_stage + kInputPageOff]);
    auto *weight_slices = reinterpret_cast<slice_sv *>(
        ss.pages[base_stage + kWeightPageOff]);
    slice_sv &input_slice  = input_slices[warp_in_role];
    slice_sv &weight_slice = weight_slices[warp_in_role];

    float *partial_sums_scratch = reinterpret_cast<float *>(ss.scratch);

    int iter = 0;
    for (int row = blockIdx.x; row < NUM_TOKENS; row += gridDim.x, ++iter) {
        (void)row;
        const int phase = iter & 1;

        kittens::wait(ss.page_ready[base_stage + kInputPageOff],  phase);
        kittens::wait(ss.page_ready[base_stage + kWeightPageOff], phase);

        // Load activation, compute RMS scale, apply to activation.
        rv_t act_rv;
        kittens::warp::load(act_rv, input_slice);
        const float rms_scale =
            ferrite::tk::rms_norm_scale_from_rv<
                NUM_CONSUMER_WARPS, HIDDEN_DIM, kConsumerBarReduce>(
            act_rv, eps, partial_sums_scratch);
        kittens::warp::mul(act_rv, act_rv, rms_scale);

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
        kittens::warp::mul(act_rv, act_rv, weight_vec);

        // In-place publish: overwrite input slice with bf16-downcast output.
        kittens::warp::store(input_slice, act_rv);

        kittens::group<NUM_CONSUMER_WARPS>::sync(kConsumerBarPublish);
        if (warp_in_role == 0 && kittens::laneid() == 0) {
            kittens::arrive(ss.page_done[base_stage + kInputPageOff]);
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
    ferrite::bf16_ptr rms_output,   // [NUM_TOKENS, HIDDEN_DIM]
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    using warp = kittens::group<1>;
    constexpr uint32_t row_bytes = HIDDEN_DIM * sizeof(__nv_bfloat16);

    int iter = 0;
    for (int row = blockIdx.x; row < NUM_TOKENS; row += gridDim.x, ++iter) {
        const int phase = iter & 1;
        kittens::wait(ss.page_done[base_stage + kInputPageOff], phase);

        void* src_page = reinterpret_cast<void*>(
            ss.pages[base_stage + kInputPageOff]);
        warp::tma::store_async(
            static_cast<void*>(rms_output + static_cast<size_t>(row) * HIDDEN_DIM),
            src_page,
            row_bytes);
        kittens::tma::store_async_wait<0>();

        __syncthreads();
    }
}

}  // namespace rms_norm_offset
}  // namespace ops
}  // namespace ferrite
