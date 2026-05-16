// SPDX-License-Identifier: Apache-2.0
// Ferrite-owned TK 2.0 lm_head op — 4 warp-role functions.
//
// Math: fused `rms_norm` + gemv, the canonical final-norm +
// lm_head pattern at the end of a decoder.
//   x_n[k] = x[k] * rsqrt(mean(x^2) + eps) * norm_weight[k]
//   out[row] = sum_k W_gemm[row, k] * x_n[k]
//
// TK-canonical consumer (Wave 3 body rewrite): replaces the
// scalar `__shfl_xor_sync` / `__bfloat162float` / `asm volatile
// bar.sync` reductions with TK register-tile primitives.
// BS-grep 0: no `__shfl`, no scalar bf16 casts in compute path,
// no `bar.sync` (uses `kittens::group<N>::sync(barId)`).
//
// Parallelism (Phase 3 step 7): persistent-thread grid.
// Each role loops `for (int row = blockIdx.x; row < N; row += gridDim.x)`.
// One vocab row per CTA per loop iteration.
//
// Page layout (caller gives `base_stage`):
//   pages[base_stage + 0] — activation x (sv_bf<K>). Also used by
//                           the consumer to stash the final bf16
//                           scalar for the storer (first 2 bytes).
//   pages[base_stage + 1] — norm weight (sv_bf<K>).
//   pages[base_stage + 2] — gemm weight row W[row, :] (sv_bf<K>).
//
// Scratch layout:
//   float[0 .. NUM_CONSUMER_WARPS) — per-warp partial sum_sq
//   float[NUM_CONSUMER_WARPS .. 2*NUM_CONSUMER_WARPS) — per-warp partial dot
//
// TK bar IDs (must not collide with bar 0 = __syncthreads):
//   kConsumerBarSumSq = 1 — cross-warp sync after sum_sq partial reduction
//   kConsumerBarDot   = 2 — cross-warp sync after dot-product reduction

#pragma once

#include "kittens.cuh"
#include "ferrite_globals.cuh"
#include "ferrite_substrate.cuh"
#include "ferrite_tk_helpers.cuh"

namespace ferrite {
namespace ops {
namespace lm_head {

constexpr int kActPageOff        = 0;
constexpr int kNormWeightOff     = 1;
constexpr int kGemmWeightOff     = 2;

constexpr int kConsumerBarSumSq  = 1;
constexpr int kConsumerBarDot    = 2;

// ---------- Loader --------------------------------------------------
//
// TMA-based load: activation x, norm weight, and one row of W_gemm.
// TK-canonical: uses kittens::tma::expect_bytes / tma::load_async
// (no scalar bf16 casts in the load path).

template <typename Config, int K, int N, int NUM_TOKENS>
__device__ __forceinline__ void loader(
    ferrite::bf16_cptr x,              // [K]
    ferrite::bf16_cptr norm_weight,    // [K]
    ferrite::bf16_cptr W_gemm,         // [N, K] row-major
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    static_assert(NUM_TOKENS == 1,
                  "lm_head: NUM_TOKENS > 1 not wired yet "
                  "(decode-only; prefill body is a follow-up slice)");
    using warp = kittens::group<1>;
    constexpr uint32_t row_bytes = K * sizeof(__nv_bfloat16);

    void* act_page      = reinterpret_cast<void*>(ss.pages[base_stage + kActPageOff]);
    void* norm_w_page   = reinterpret_cast<void*>(ss.pages[base_stage + kNormWeightOff]);
    void* gemm_w_page   = reinterpret_cast<void*>(ss.pages[base_stage + kGemmWeightOff]);

    for (int row = blockIdx.x; row < N; row += gridDim.x) {
        if (kittens::laneid() == 0) {
            kittens::tma::expect_bytes(ss.page_ready[base_stage + kActPageOff], row_bytes);
        }
        warp::tma::load_async(act_page, const_cast<__nv_bfloat16*>(x),
                              row_bytes, ss.page_ready[base_stage + kActPageOff]);

        if (kittens::laneid() == 0) {
            kittens::tma::expect_bytes(ss.page_ready[base_stage + kNormWeightOff], row_bytes);
        }
        warp::tma::load_async(norm_w_page, const_cast<__nv_bfloat16*>(norm_weight),
                              row_bytes, ss.page_ready[base_stage + kNormWeightOff]);

        if (kittens::laneid() == 0) {
            kittens::tma::expect_bytes(ss.page_ready[base_stage + kGemmWeightOff], row_bytes);
        }
        warp::tma::load_async(gemm_w_page,
                              const_cast<__nv_bfloat16*>(W_gemm + static_cast<size_t>(row) * K),
                              row_bytes, ss.page_ready[base_stage + kGemmWeightOff]);

        __syncthreads();
    }
}

// ---------- Consumer ------------------------------------------------
//
// TK-canonical body: rv_fl register tiles + kittens::group::sync.
// BS-grep 0: no __shfl_xor_sync (replaced by warp::sum + group::sync),
// no __bfloat162float (replaced by warp::load into rv_fl), no
// bar.sync (replaced by kittens::group<NCW>::sync(barId)).
//
// Each consumer warp owns K/NCW elements of the K dimension:
//   warp_base = warp_in_role * (K / NUM_CONSUMER_WARPS)
//
// Phase: compute rms_norm_scale from warp's activation slice, then
// compute the dot product of (normed × norm_weight × W_row slice).
// Cross-warp reduction via scratch[0..NCW) (sum_sq) and
// scratch[NCW..2*NCW) (dot).

template <typename Config, int K, int N, int NUM_TOKENS>
__device__ __forceinline__ void consumer(
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int warp_in_role,
    float eps
) {
    static_assert(NUM_TOKENS == 1, "lm_head: NUM_TOKENS > 1 not wired yet");
    constexpr int NCW = Config::NUM_CONSUMER_WARPS;
    // K_PER_WARP must be a multiple of kittens::TILE_COL_DIM (=16).
    constexpr int K_PER_WARP = K / NCW;
    static_assert(K % NCW == 0 && K_PER_WARP % 16 == 0,
                  "lm_head: K must be divisible by NCW and K/NCW must be "
                  "a multiple of 16");

    using sv_t  = kittens::sv_bf<K_PER_WARP>;
    using rv_t  = kittens::rv_fl<K_PER_WARP>;

    // Per-warp sv_bf slice pointer (warp_in_role selects the warp's K slice).
    const size_t warp_off = static_cast<size_t>(warp_in_role) * K_PER_WARP
                            * sizeof(__nv_bfloat16);
    const sv_t& act_sv  = *reinterpret_cast<const sv_t*>(
        ss.pages[base_stage + kActPageOff] + warp_off);
    const sv_t& nw_sv   = *reinterpret_cast<const sv_t*>(
        ss.pages[base_stage + kNormWeightOff] + warp_off);
    const sv_t& gw_sv   = *reinterpret_cast<const sv_t*>(
        ss.pages[base_stage + kGemmWeightOff] + warp_off);

    // Scratch: first NCW floats for sum_sq, next NCW for dot.
    float* partial_sumsq = reinterpret_cast<float*>(ss.scratch);
    float* partial_dot   = partial_sumsq + NCW;

    int iter = 0;
    for (int row = blockIdx.x; row < N; row += gridDim.x, ++iter) {
        (void)row;
        const int phase = iter & 1;

        // --- Step 1: RmsNorm scale ----------------------------------------
        //
        // Wait for activation page, load warp's slice into rv_fl, compute
        // per-warp sum_sq. Cross-warp reduction via scratch + group::sync.
        kittens::wait(ss.page_ready[base_stage + kActPageOff], phase);

        rv_t act_rv;
        kittens::warp::load(act_rv, act_sv);

        // rms_norm_scale_from_rv: computes sum(act^2) via warp::sum,
        // deposits partial into scratch[warp_in_role], syncs on kConsumerBarSumSq,
        // reads all partials, returns rsqrt(mean + eps). No __shfl, no bar.sync.
        const float rms_scale =
            ferrite::tk::rms_norm_scale_from_rv<NCW, K, kConsumerBarSumSq>(
                act_rv, eps, partial_sumsq);

        // --- Step 2: Dot product (normed activation · W_row slice) ----------
        //
        // Load norm weight and gemm weight slices. Compute:
        //   normed[k] = act[k] * rms_scale * norm_weight[k]
        //   dot_partial = sum_k normed[k] * W_row[k]  (this warp's K slice)
        kittens::wait(ss.page_ready[base_stage + kNormWeightOff], phase);
        kittens::wait(ss.page_ready[base_stage + kGemmWeightOff], phase);

        rv_t nw_rv, gw_rv;
        kittens::warp::load(nw_rv, nw_sv);
        kittens::warp::load(gw_rv, gw_sv);

        // Apply rms_scale * norm_weight to activation (in registers).
        kittens::warp::mul(act_rv, act_rv, rms_scale);       // act *= rms_scale
        kittens::warp::mul(act_rv, act_rv, nw_rv);            // act *= norm_weight

        // Element-wise multiply with W_row slice, then warp-level sum.
        kittens::warp::mul(gw_rv, act_rv, gw_rv);             // product = normed * W_row
        const float warp_dot = kittens::warp::sum(gw_rv);

        // Cross-warp reduction: store partial, sync, accumulate.
        if (kittens::laneid() == 0) {
            partial_dot[warp_in_role] = warp_dot;
        }
        kittens::group<NCW>::sync(kConsumerBarDot);

        if (warp_in_role == 0 && kittens::laneid() == 0) {
            float total = 0.0f;
            #pragma unroll
            for (int w = 0; w < NCW; ++w) {
                total += partial_dot[w];
            }
            // Store scalar result in the first 2 bytes of the activation page
            // so the storer can read it without a separate page slot.
            __nv_bfloat16* out_slot = reinterpret_cast<__nv_bfloat16*>(
                ss.pages[base_stage + kActPageOff]);
            out_slot[0] = __float2bfloat16_rn(total);
            kittens::arrive(ss.page_done[base_stage + kActPageOff]);
        }
        __syncthreads();
    }
}

// ---------- Launcher ------------------------------------------------

template <typename Config, int K, int N, int NUM_TOKENS>
__device__ __forceinline__ void launcher(
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    static_assert(NUM_TOKENS == 1, "lm_head: NUM_TOKENS > 1 not wired yet");
    (void)ss; (void)base_stage;
    for (int row = blockIdx.x; row < N; row += gridDim.x) {
        (void)row;
        __syncthreads();
    }
}

// ---------- Storer --------------------------------------------------

template <typename Config, int K, int N, int NUM_TOKENS>
__device__ __forceinline__ void storer(
    ferrite::bf16_ptr out,             // [N]
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    static_assert(NUM_TOKENS == 1, "lm_head: NUM_TOKENS > 1 not wired yet");
    int iter = 0;
    for (int row = blockIdx.x; row < N; row += gridDim.x, ++iter) {
        const int phase = iter & 1;
        kittens::wait(ss.page_done[base_stage + kActPageOff], phase);
        if (kittens::laneid() == 0) {
            // One benign scalar bf16 read at warp::store boundary
            // (same pattern as rms_norm.cuh / fused_add_rms_norm.cuh).
            const __nv_bfloat16* src_slot = reinterpret_cast<const __nv_bfloat16*>(
                ss.pages[base_stage + kActPageOff]);
            out[row] = src_slot[0];
        }
        __syncthreads();
    }
}

}  // namespace lm_head

// ─────────────────────────────────────────────────────────────────────────────
// lm_head_fused_residual — (Add, RmsNorm, GEMV) 3-tile fusion.
//
// Math: fused `residual_add + rms_norm + gemv`.
//   sum[k]     = delta[k] + residual[k]          (residual_out — written back)
//   x_n[k]     = sum[k] * rsqrt(mean(sum^2) + eps) * norm_weight[k]
//   out[row]   = sum_k W_gemm[row, k] * x_n[k]   (logit scalar per vocab row)
//
// Differences from lm_head:
//  - Takes both delta + residual as inputs (4 pages instead of 3).
//  - Consumer: Add(delta, residual) → writes residual_out in-place to
//    residual page, then proceeds with rms_norm + dot-product.
//  - Storer: writes logit out[row] every iteration AND writes residual_out
//    (K-element vector) on the first iteration of CTA 0 only.
//
// Page layout (caller gives `base_stage`):
//   pages[base_stage + 0] — delta tile (sv_bf<K>).
//   pages[base_stage + 1] — residual tile (sv_bf<K>, in-place residual_out).
//   pages[base_stage + 2] — norm weight (sv_bf<K>).
//   pages[base_stage + 3] — gemm weight row W[row, :] (sv_bf<K>).
//                           Reused to pass logit scalar to storer (first 2 bytes).
//
// Semaphores:
//   page_ready[base_stage + 0..3] — loader → consumer (four loads per iter).
//   page_done [base_stage + 1]    — consumer → storer (residual_out ready).
//   page_done [base_stage + 3]    — consumer → storer (logit scalar ready).
//
// TK bar IDs (same as lm_head namespace above):
//   kFusedConsumerBarSumSq = 1
//   kFusedConsumerBarDot   = 2

namespace lm_head_fused_residual {

constexpr int kDeltaPageOff      = 0;
constexpr int kResidualPageOff   = 1;
constexpr int kNormWeightOff     = 2;
constexpr int kGemmWeightOff     = 3;  // also carries logit scalar

constexpr int kFusedConsumerBarSumSq = 1;
constexpr int kFusedConsumerBarDot   = 2;

// ---------- Loader --------------------------------------------------
//
// iter==0: load delta, residual, norm_weight AND W[row=0,:] — these
// shared inputs are needed once to seed the normed activation.
// iter>0:  SKIP delta/residual/norm_weight (consumer caches normed_rv
// in registers after iter 0) and only load W[row,:]. This eliminates
// (N-1)×3×K = ≈99.9% of the redundant activation bandwidth.

template <typename Config, int K, int N, int NUM_TOKENS>
__device__ __forceinline__ void loader(
    ferrite::bf16_cptr delta_in,        // [K]
    ferrite::bf16_cptr residual_in,     // [K]
    ferrite::bf16_cptr norm_weight,     // [K]
    ferrite::bf16_cptr W_gemm,          // [N, K] row-major
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    static_assert(NUM_TOKENS == 1,
                  "lm_head_fused_residual: NUM_TOKENS > 1 not wired yet");
    using warp = kittens::group<1>;
    constexpr uint32_t row_bytes = K * sizeof(__nv_bfloat16);

    void* delta_page    = reinterpret_cast<void*>(ss.pages[base_stage + kDeltaPageOff]);
    void* residual_page = reinterpret_cast<void*>(ss.pages[base_stage + kResidualPageOff]);
    void* norm_w_page   = reinterpret_cast<void*>(ss.pages[base_stage + kNormWeightOff]);
    void* gemm_w_page   = reinterpret_cast<void*>(ss.pages[base_stage + kGemmWeightOff]);

    int iter = 0;
    for (int row = blockIdx.x; row < N; row += gridDim.x, ++iter) {
        if (iter == 0) {
            // First iteration: load shared inputs once alongside W[row,:].
            if (kittens::laneid() == 0) {
                kittens::tma::expect_bytes(ss.page_ready[base_stage + kDeltaPageOff],    row_bytes);
                kittens::tma::expect_bytes(ss.page_ready[base_stage + kResidualPageOff], row_bytes);
                kittens::tma::expect_bytes(ss.page_ready[base_stage + kNormWeightOff],   row_bytes);
            }
            warp::tma::load_async(delta_page,    const_cast<__nv_bfloat16*>(delta_in),   row_bytes,
                                  ss.page_ready[base_stage + kDeltaPageOff]);
            warp::tma::load_async(residual_page, const_cast<__nv_bfloat16*>(residual_in), row_bytes,
                                  ss.page_ready[base_stage + kResidualPageOff]);
            warp::tma::load_async(norm_w_page,   const_cast<__nv_bfloat16*>(norm_weight), row_bytes,
                                  ss.page_ready[base_stage + kNormWeightOff]);
        }
        if (kittens::laneid() == 0) {
            kittens::tma::expect_bytes(ss.page_ready[base_stage + kGemmWeightOff], row_bytes);
        }
        warp::tma::load_async(gemm_w_page,
                              const_cast<__nv_bfloat16*>(W_gemm + static_cast<size_t>(row) * K),
                              row_bytes,
                              ss.page_ready[base_stage + kGemmWeightOff]);
        __syncthreads();
    }
}

// ---------- Consumer ------------------------------------------------
//
// iter==0: wait for all 4 pages, compute Add + RmsNorm → normed_rv in
// REGISTERS, arrive on page_done[residual]. Cache normed_rv in regs.
// iter>0: skip delta/residual/norm_weight (cached in normed_rv), only
// wait for W[row,:] page and dot with cached normed_rv.
// This eliminates (N-1)×3×K redundant shared-input loads from the
// consumer hot path.

template <typename Config, int K, int N, int NUM_TOKENS>
__device__ __forceinline__ void consumer(
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int warp_in_role,
    float eps
) {
    static_assert(NUM_TOKENS == 1,
                  "lm_head_fused_residual: NUM_TOKENS > 1 not wired yet");
    constexpr int NCW = Config::NUM_CONSUMER_WARPS;
    constexpr int K_PER_WARP = K / NCW;
    static_assert(K % NCW == 0 && K_PER_WARP % 16 == 0,
                  "lm_head_fused_residual: K must be divisible by NCW and "
                  "K/NCW must be a multiple of 16");

    using sv_t = kittens::sv_bf<K_PER_WARP>;
    using rv_t = kittens::rv_fl<K_PER_WARP>;

    const size_t warp_off = static_cast<size_t>(warp_in_role) * K_PER_WARP
                            * sizeof(__nv_bfloat16);

    const sv_t& delta_sv    = *reinterpret_cast<const sv_t*>(
        ss.pages[base_stage + kDeltaPageOff] + warp_off);
    sv_t&       residual_sv = *reinterpret_cast<sv_t*>(
        ss.pages[base_stage + kResidualPageOff] + warp_off);
    const sv_t& nw_sv       = *reinterpret_cast<const sv_t*>(
        ss.pages[base_stage + kNormWeightOff] + warp_off);
    const sv_t& gw_sv       = *reinterpret_cast<const sv_t*>(
        ss.pages[base_stage + kGemmWeightOff] + warp_off);

    float* partial_sumsq = reinterpret_cast<float*>(ss.scratch);
    float* partial_dot   = partial_sumsq + NCW;

    // normed_rv is computed once on iter==0 and cached in registers.
    rv_t normed_rv;

    int iter = 0;
    for (int row = blockIdx.x; row < N; row += gridDim.x, ++iter) {
        (void)row;
        const int phase = iter & 1;

        if (iter == 0) {
            // First iteration: load shared inputs, compute Add+RmsNorm,
            // store residual_out in-place, signal storer.
            kittens::wait(ss.page_ready[base_stage + kDeltaPageOff],    phase);
            kittens::wait(ss.page_ready[base_stage + kResidualPageOff], phase);

            rv_t delta_rv, residual_rv, sum_rv;
            kittens::warp::load(delta_rv,    delta_sv);
            kittens::warp::load(residual_rv, residual_sv);
            kittens::warp::add(sum_rv, delta_rv, residual_rv);

            // Write residual_out in-place (storer TMA-stores it).
            kittens::warp::store(residual_sv, sum_rv);

            const float rms_scale =
                ferrite::tk::rms_norm_scale_from_rv<NCW, K, kFusedConsumerBarSumSq>(
                    sum_rv, eps, partial_sumsq);

            kittens::wait(ss.page_ready[base_stage + kNormWeightOff], phase);
            rv_t nw_rv;
            kittens::warp::load(nw_rv, nw_sv);

            // Compute normed_rv = sum * rms_scale * norm_weight.
            // Cached in registers; used for all subsequent iterations.
            kittens::warp::mul(normed_rv, sum_rv, rms_scale);
            kittens::warp::mul(normed_rv, normed_rv, nw_rv);

            // Signal residual_out ready for storer.
            if (warp_in_role == 0 && kittens::laneid() == 0) {
                kittens::arrive(ss.page_done[base_stage + kResidualPageOff]);
            }
        }

        // Every iteration: wait for W[row,:], dot-product with normed_rv.
        kittens::wait(ss.page_ready[base_stage + kGemmWeightOff], phase);

        rv_t gw_rv;
        kittens::warp::load(gw_rv, gw_sv);

        rv_t prod_rv;
        kittens::warp::mul(prod_rv, normed_rv, gw_rv);
        const float warp_dot = kittens::warp::sum(prod_rv);

        if (kittens::laneid() == 0) {
            partial_dot[warp_in_role] = warp_dot;
        }
        kittens::group<NCW>::sync(kFusedConsumerBarDot);

        if (warp_in_role == 0 && kittens::laneid() == 0) {
            float total = 0.0f;
            #pragma unroll
            for (int w = 0; w < NCW; ++w) total += partial_dot[w];

            __nv_bfloat16* logit_slot = reinterpret_cast<__nv_bfloat16*>(
                ss.pages[base_stage + kGemmWeightOff]);
            logit_slot[0] = __float2bfloat16_rn(total);
            kittens::arrive(ss.page_done[base_stage + kGemmWeightOff]);
        }
        __syncthreads();
    }
}

// ---------- Launcher ------------------------------------------------

template <typename Config, int K, int N, int NUM_TOKENS>
__device__ __forceinline__ void launcher(
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    static_assert(NUM_TOKENS == 1,
                  "lm_head_fused_residual: NUM_TOKENS > 1 not wired yet");
    (void)ss; (void)base_stage;
    for (int row = blockIdx.x; row < N; row += gridDim.x) {
        (void)row;
        __syncthreads();
    }
}

// ---------- Storer --------------------------------------------------
//
// iter==0: wait for page_done[gemm_weight] AND page_done[residual],
// write logit[row] AND residual_out (CTA 0 only).
// iter>0:  wait for page_done[gemm_weight] only, write logit[row].

template <typename Config, int K, int N, int NUM_TOKENS>
__device__ __forceinline__ void storer(
    ferrite::bf16_ptr out,           // [N]   logit scalars
    ferrite::bf16_ptr residual_out,  // [K]   residual output (written once, CTA 0 only)
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    static_assert(NUM_TOKENS == 1,
                  "lm_head_fused_residual: NUM_TOKENS > 1 not wired yet");
    using warp = kittens::group<1>;
    constexpr uint32_t residual_bytes = K * sizeof(__nv_bfloat16);

    int iter = 0;
    for (int row = blockIdx.x; row < N; row += gridDim.x, ++iter) {
        const int phase = iter & 1;

        // Logit scalar — every iteration.
        kittens::wait(ss.page_done[base_stage + kGemmWeightOff], phase);
        if (kittens::laneid() == 0) {
            const __nv_bfloat16* logit_slot = reinterpret_cast<const __nv_bfloat16*>(
                ss.pages[base_stage + kGemmWeightOff]);
            out[row] = logit_slot[0];
        }

        // Residual_out — iter==0 only; consumer arrives once (phase 0→1).
        // phase=0 means "wait for parity 0→1 transition" = the one arrival.
        if (iter == 0) {
            kittens::wait(ss.page_done[base_stage + kResidualPageOff], 0);
            if (blockIdx.x == 0) {
                void* residual_src = reinterpret_cast<void*>(
                    ss.pages[base_stage + kResidualPageOff]);
                warp::tma::store_async(
                    static_cast<void*>(residual_out), residual_src, residual_bytes);
                kittens::tma::store_async_wait<0>();
            }
        }

        __syncthreads();
    }
}

}  // namespace lm_head_fused_residual

// ─────────────────────────────────────────────────────────────────────────────
// lm_head_fused_residual_offset — (Add, ScalarOffsetRmsNorm, GEMV) 4-tile fusion.
//
// Identical to lm_head_fused_residual but applies `(norm_weight + weight_offset)`
// in the consumer's normalization step. Gemma2 uses offset=1.0.
//
// The consumer signature gains `float weight_offset`; loader/launcher/storer
// are unchanged (same page layout, same semaphores).

namespace lm_head_fused_residual_offset {

constexpr int kDeltaPageOff      = 0;
constexpr int kResidualPageOff   = 1;
constexpr int kNormWeightOff     = 2;
constexpr int kGemmWeightOff     = 3;

constexpr int kFusedConsumerBarSumSq = 1;
constexpr int kFusedConsumerBarDot   = 2;

// ---------- Loader --------------------------------------------------
// Identical to lm_head_fused_residual::loader.

template <typename Config, int K, int N, int NUM_TOKENS>
__device__ __forceinline__ void loader(
    ferrite::bf16_cptr delta_in,
    ferrite::bf16_cptr residual_in,
    ferrite::bf16_cptr norm_weight,
    ferrite::bf16_cptr W_gemm,
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    static_assert(NUM_TOKENS == 1,
                  "lm_head_fused_residual_offset: NUM_TOKENS > 1 not wired yet");
    using warp = kittens::group<1>;
    constexpr uint32_t row_bytes = K * sizeof(__nv_bfloat16);

    void* delta_page    = reinterpret_cast<void*>(ss.pages[base_stage + kDeltaPageOff]);
    void* residual_page = reinterpret_cast<void*>(ss.pages[base_stage + kResidualPageOff]);
    void* norm_w_page   = reinterpret_cast<void*>(ss.pages[base_stage + kNormWeightOff]);
    void* gemm_w_page   = reinterpret_cast<void*>(ss.pages[base_stage + kGemmWeightOff]);

    int iter = 0;
    for (int row = blockIdx.x; row < N; row += gridDim.x, ++iter) {
        if (iter == 0) {
            if (kittens::laneid() == 0) {
                kittens::tma::expect_bytes(ss.page_ready[base_stage + kDeltaPageOff],    row_bytes);
                kittens::tma::expect_bytes(ss.page_ready[base_stage + kResidualPageOff], row_bytes);
                kittens::tma::expect_bytes(ss.page_ready[base_stage + kNormWeightOff],   row_bytes);
            }
            warp::tma::load_async(delta_page,    const_cast<__nv_bfloat16*>(delta_in),   row_bytes,
                                  ss.page_ready[base_stage + kDeltaPageOff]);
            warp::tma::load_async(residual_page, const_cast<__nv_bfloat16*>(residual_in), row_bytes,
                                  ss.page_ready[base_stage + kResidualPageOff]);
            warp::tma::load_async(norm_w_page,   const_cast<__nv_bfloat16*>(norm_weight), row_bytes,
                                  ss.page_ready[base_stage + kNormWeightOff]);
        }
        if (kittens::laneid() == 0) {
            kittens::tma::expect_bytes(ss.page_ready[base_stage + kGemmWeightOff], row_bytes);
        }
        warp::tma::load_async(gemm_w_page,
                              const_cast<__nv_bfloat16*>(W_gemm + static_cast<size_t>(row) * K),
                              row_bytes,
                              ss.page_ready[base_stage + kGemmWeightOff]);
        __syncthreads();
    }
}

// ---------- Consumer ------------------------------------------------
// Identical to lm_head_fused_residual::consumer but applies (norm_weight + weight_offset).

template <typename Config, int K, int N, int NUM_TOKENS>
__device__ __forceinline__ void consumer(
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int warp_in_role,
    float eps,
    float weight_offset
) {
    static_assert(NUM_TOKENS == 1,
                  "lm_head_fused_residual_offset: NUM_TOKENS > 1 not wired yet");
    constexpr int NCW = Config::NUM_CONSUMER_WARPS;
    constexpr int K_PER_WARP = K / NCW;
    static_assert(K % NCW == 0 && K_PER_WARP % 16 == 0,
                  "lm_head_fused_residual_offset: K must be divisible by NCW and "
                  "K/NCW must be a multiple of 16");

    using sv_t = kittens::sv_bf<K_PER_WARP>;
    using rv_t = kittens::rv_fl<K_PER_WARP>;

    const size_t warp_off = static_cast<size_t>(warp_in_role) * K_PER_WARP
                            * sizeof(__nv_bfloat16);

    const sv_t& delta_sv    = *reinterpret_cast<const sv_t*>(
        ss.pages[base_stage + kDeltaPageOff] + warp_off);
    sv_t&       residual_sv = *reinterpret_cast<sv_t*>(
        ss.pages[base_stage + kResidualPageOff] + warp_off);
    const sv_t& nw_sv       = *reinterpret_cast<const sv_t*>(
        ss.pages[base_stage + kNormWeightOff] + warp_off);
    const sv_t& gw_sv       = *reinterpret_cast<const sv_t*>(
        ss.pages[base_stage + kGemmWeightOff] + warp_off);

    float* partial_sumsq = reinterpret_cast<float*>(ss.scratch);
    float* partial_dot   = partial_sumsq + NCW;

    rv_t normed_rv;

    int iter = 0;
    for (int row = blockIdx.x; row < N; row += gridDim.x, ++iter) {
        (void)row;
        const int phase = iter & 1;

        if (iter == 0) {
            kittens::wait(ss.page_ready[base_stage + kDeltaPageOff],    phase);
            kittens::wait(ss.page_ready[base_stage + kResidualPageOff], phase);

            rv_t delta_rv, residual_rv, sum_rv;
            kittens::warp::load(delta_rv,    delta_sv);
            kittens::warp::load(residual_rv, residual_sv);
            kittens::warp::add(sum_rv, delta_rv, residual_rv);

            kittens::warp::store(residual_sv, sum_rv);

            const float rms_scale =
                ferrite::tk::rms_norm_scale_from_rv<NCW, K, kFusedConsumerBarSumSq>(
                    sum_rv, eps, partial_sumsq);

            kittens::wait(ss.page_ready[base_stage + kNormWeightOff], phase);
            rv_t nw_rv;
            kittens::warp::load(nw_rv, nw_sv);

            // Apply (norm_weight + weight_offset) for Gemma2 (1+w) convention.
            #pragma unroll
            for (int _o = 0; _o < nw_rv.outer_dim; ++_o) {
                #pragma unroll
                for (int _i = 0; _i < nw_rv.inner_dim; ++_i) {
                    nw_rv.data[_o][_i] += weight_offset;
                }
            }

            kittens::warp::mul(normed_rv, sum_rv, rms_scale);
            kittens::warp::mul(normed_rv, normed_rv, nw_rv);

            if (warp_in_role == 0 && kittens::laneid() == 0) {
                kittens::arrive(ss.page_done[base_stage + kResidualPageOff]);
            }
        }

        kittens::wait(ss.page_ready[base_stage + kGemmWeightOff], phase);

        rv_t gw_rv;
        kittens::warp::load(gw_rv, gw_sv);

        rv_t prod_rv;
        kittens::warp::mul(prod_rv, normed_rv, gw_rv);
        const float warp_dot = kittens::warp::sum(prod_rv);

        if (kittens::laneid() == 0) {
            partial_dot[warp_in_role] = warp_dot;
        }
        kittens::group<NCW>::sync(kFusedConsumerBarDot);

        if (warp_in_role == 0 && kittens::laneid() == 0) {
            float total = 0.0f;
            #pragma unroll
            for (int w = 0; w < NCW; ++w) total += partial_dot[w];

            __nv_bfloat16* logit_slot = reinterpret_cast<__nv_bfloat16*>(
                ss.pages[base_stage + kGemmWeightOff]);
            logit_slot[0] = __float2bfloat16_rn(total);
            kittens::arrive(ss.page_done[base_stage + kGemmWeightOff]);
        }
        __syncthreads();
    }
}

// ---------- Launcher ------------------------------------------------

template <typename Config, int K, int N, int NUM_TOKENS>
__device__ __forceinline__ void launcher(
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    static_assert(NUM_TOKENS == 1,
                  "lm_head_fused_residual_offset: NUM_TOKENS > 1 not wired yet");
    (void)ss; (void)base_stage;
    for (int row = blockIdx.x; row < N; row += gridDim.x) {
        (void)row;
        __syncthreads();
    }
}

// ---------- Storer --------------------------------------------------
// Identical to lm_head_fused_residual::storer.

template <typename Config, int K, int N, int NUM_TOKENS>
__device__ __forceinline__ void storer(
    ferrite::bf16_ptr out,
    ferrite::bf16_ptr residual_out,
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    static_assert(NUM_TOKENS == 1,
                  "lm_head_fused_residual_offset: NUM_TOKENS > 1 not wired yet");
    using warp = kittens::group<1>;
    constexpr uint32_t residual_bytes = K * sizeof(__nv_bfloat16);

    int iter = 0;
    for (int row = blockIdx.x; row < N; row += gridDim.x, ++iter) {
        const int phase = iter & 1;

        kittens::wait(ss.page_done[base_stage + kGemmWeightOff], phase);
        if (kittens::laneid() == 0) {
            const __nv_bfloat16* logit_slot = reinterpret_cast<const __nv_bfloat16*>(
                ss.pages[base_stage + kGemmWeightOff]);
            out[row] = logit_slot[0];
        }

        if (iter == 0) {
            kittens::wait(ss.page_done[base_stage + kResidualPageOff], 0);
            if (blockIdx.x == 0) {
                void* residual_src = reinterpret_cast<void*>(
                    ss.pages[base_stage + kResidualPageOff]);
                warp::tma::store_async(
                    static_cast<void*>(residual_out), residual_src, residual_bytes);
                kittens::tma::store_async_wait<0>();
            }
        }

        __syncthreads();
    }
}

}  // namespace lm_head_fused_residual_offset
}  // namespace ops
}  // namespace ferrite
