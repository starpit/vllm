// SPDX-License-Identifier: Apache-2.0
// Ferrite-owned TK 2.0 RmsNorm op — 4 warp-role functions.
//
// Math: `out[row, :] = weight * input[row, :] * rsqrt(mean(input^2) + eps)`
// bf16 in/out, fp32 accumulator. Mirrors the host-interpreter
// kernel in `vllm-cuda/csrc/layernorm_kernels.cu` (weight_offset=0
// Llama case).
//
// Parallelism layout (Phase 3 step 7 — persistent-thread grid):
//   - Grid: the mega launcher sizes the grid to `NUM_SMS *
//     ctas_per_sm` CTAs, one resident wave. Each role loops
//     `for (int row = blockIdx.x; row < NUM_TOKENS; row += gridDim.x)`
//     — CTAs with blockIdx.x >= NUM_TOKENS run zero iterations
//     (harmless); CTAs with native work iterate until NUM_TOKENS
//     is covered.
//   - Within CTA: four warp roles from ferrite_warp_roles.cuh.
//   - Loader warp: per-iter cp.async.bulk loads for input row +
//     weight, arrives on page_ready.
//   - Consumer warps: per-iter RMS math; scratch reductions bounded
//     by consumer-scoped bar.sync 1 / 2.
//   - Storer warp: per-iter TMA-bulk-store of the normalized row.
//
// Per-iter semaphore phase: `wait(sem, iter & 1)`. Each
// load_async / arrive flips the mbarrier parity, so iteration T
// waits on parity `T & 1`. `__syncthreads()` at the end of each
// iteration serializes page reuse so the next iter's TMA doesn't
// race the prior iter's consumer reads.
//
// Page layout (per rms_norm call, caller gives `base_stage`):
//   pages[base_stage + 0] — input activation row (sv_bf<HIDDEN_DIM>),
//                           reused in-place for output.
//   pages[base_stage + 1] — weight row (sv_bf<HIDDEN_DIM>).
//
// Semaphore handoff (init'd once in init_shared_state):
//   page_ready[base_stage + 0] — loader → consumer (input).
//   page_ready[base_stage + 1] — loader → consumer (weight).
//   page_done [base_stage + 0] — consumer → storer (output ready).
//
// bar IDs used for consumer-scoped syncs: 1 and 2 (bar 0 is
// __syncthreads; non-consumer warps must not touch 1/2).

#pragma once

#include "kittens.cuh"
#include "ferrite_globals.cuh"
#include "ferrite_substrate.cuh"
#include "ferrite_tk_helpers.cuh"

namespace ferrite {
namespace ops {
namespace rms_norm {

// Page-slot offsets relative to `base_stage`. Stable across callers
// so variant_cpp.rs can emit `base_stage` constants known to the
// page-liveness analysis.
constexpr int kInputPageOff  = 0;
constexpr int kWeightPageOff = 1;

// Consumer-scoped bar.sync IDs. Picked to avoid collision with
// CTA-wide bar 0 (== __syncthreads) and with each other.
//   kConsumerBarReduce  — cross-warp sum-of-squares reduction inside
//                         ferrite::tk::rms_norm_vec.
//   kConsumerBarPublish — all consumer warps must have warp::store'd
//                         their slice back to the input page before
//                         warp 0 arrives on page_done for the storer.
constexpr int kConsumerBarReduce  = 1;
constexpr int kConsumerBarPublish = 2;

// ---------- Loader --------------------------------------------------

// Issues cp.async.bulk loads for each owned row's input activation
// and rms weight, looping over the persistent-thread tile space.
// Called from the single loader warp (warp id == NUM_CONSUMER_WARPS).
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
        // Input row.
        if (kittens::laneid() == 0) {
            kittens::tma::expect_bytes(ss.page_ready[base_stage + kInputPageOff], row_bytes);
        }
        warp::tma::load_async(
            input_page,
            const_cast<__nv_bfloat16*>(rms_input + static_cast<size_t>(row) * HIDDEN_DIM),
            row_bytes,
            ss.page_ready[base_stage + kInputPageOff]);

        // Weight (shared across rows; still loaded per CTA in Phase 3
        // for simplicity — a cross-op pipelining pass can hoist it).
        if (kittens::laneid() == 0) {
            kittens::tma::expect_bytes(ss.page_ready[base_stage + kWeightPageOff], row_bytes);
        }
        warp::tma::load_async(
            weight_page,
            const_cast<__nv_bfloat16*>(rms_weight),
            row_bytes,
            ss.page_ready[base_stage + kWeightPageOff]);

        __syncthreads();  // end of per-iter body — page reuse fence
    }
}

// ---------- Consumer ------------------------------------------------

// Per-row RMS norm on TK register tiles. Each of `NUM_CONSUMER_WARPS`
// consumer warps (called with `warp_in_role` ∈ [0, NUM_CONSUMER_WARPS))
// owns a contiguous `HIDDEN_DIM / NUM_CONSUMER_WARPS` slice of every row.
//
// Flow per iter:
//   1. wait for loader to fill input + weight pages;
//   2. call ferrite::tk::rms_norm_vec, which (a) loads this warp's slice
//      from the input sv, (b) squares + sums + does cross-warp reduction
//      via BAR_ID==kConsumerBarReduce, (c) multiplies by rsqrt(mean + eps)
//      and by the rms weight slice — all in registers;
//   3. warp::store the resulting rv_fl back to this warp's slice of the
//      input page (in-place output for the storer);
//   4. cross-warp BAR_ID==kConsumerBarPublish so warp 0 sees all slices
//      written before arriving on page_done.
//
// No __shfl_xor butterflies, no per-thread bf16 pointer walks, no `asm
// bar.sync` — everything routes through TK primitives and the register-
// vector / shared-vector machinery.
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
                  "rms_norm: HIDDEN_DIM must be divisible by NUM_CONSUMER_WARPS");
    static_assert(ELEMS_PER_WARP % 16 == 0,
                  "rms_norm: ELEMS_PER_WARP must be a multiple of 16 "
                  "(TK sv_bf register/shared alignment)");

    using slice_sv = kittens::sv_bf<ELEMS_PER_WARP>;

    // Warp-sliced views into the full-row pages. The page is a raw byte
    // buffer; we reinterpret it as a C array of sv_bf<ELEMS_PER_WARP>,
    // indexed by `warp_in_role`. Each slice is contiguous bf16 of the
    // warp's owned range within the row.
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

        auto normalised = ferrite::tk::rms_norm_vec<
            NUM_CONSUMER_WARPS, HIDDEN_DIM, kConsumerBarReduce>(
            weight_slice, input_slice, eps, partial_sums_scratch);

        // In-place publish: overwrite the warp's input slice with the
        // bf16-downcast normalised output. warp::store emits the
        // register→shared conversion (rv_fl → sv_bf) that sets us up
        // for the storer's tma::store_async of the whole page.
        kittens::warp::store(input_slice, normalised);

        kittens::group<NUM_CONSUMER_WARPS>::sync(kConsumerBarPublish);
        if (warp_in_role == 0 && kittens::laneid() == 0) {
            kittens::arrive(ss.page_done[base_stage + kInputPageOff]);
        }
        __syncthreads();  // end of per-iter body — page reuse fence
    }
}

// ---------- Launcher ------------------------------------------------

// Hopper first-cut: no wgmma/tcgen05 to launch, so this role is a
// no-op per iter. The CTA-wide __syncthreads at the end of each
// iter keeps the launcher warp in lock-step with the other roles
// so page reuse is sound.
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

// Waits for the consumer to publish the output (page reuse: output
// lives in pages[base_stage + 0]), then TMA-bulk-stores the row to
// gmem. `store_async` emits the required fence.proxy.async before
// the cp.async.bulk, so no extra barrier is needed.
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

        __syncthreads();  // end of per-iter body — page reuse fence
    }
}

}  // namespace rms_norm
}  // namespace ops
}  // namespace ferrite
