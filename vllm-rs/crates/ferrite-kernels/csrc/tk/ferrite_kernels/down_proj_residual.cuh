// SPDX-License-Identifier: Apache-2.0
// Ferrite-owned TK 2.0 down_proj_residual op — 4 warp-role functions.
//
// Math: `residual[N] += W[N, K] @ x[K]`  (in-place residual update).
//
// TK-canonical body (Wave 3 rewrite): replaces the scalar
// `__bfloat162float` / `__shfl_xor_sync` / `asm volatile bar.sync`
// reductions in the consumer with TK register-tile primitives.
// BS-grep 0 (compute path): no __shfl, no scalar bf16 casts in the
// K-loop, no bar.sync.  Storer retains two benign scalar bf16
// conversions at the read-modify-write boundary (same class as the
// boundary casts in rms_norm.cuh).
//
// Parallelism (Phase 3 step 7): persistent-thread grid.
// Each role loops `for (int row = blockIdx.x; row < N; row += gridDim.x)`.
// One output row per CTA per iteration.
//
// Consumer K-chunking: K/NCW elements per warp, processed in
// CHUNK_COLS=512 element sub-chunks per rv_fl iteration.  Keeps
// rv_fl<CHUNK_COLS> (16 regs/lane) within Hopper's register budget for
// large K (e.g. K=8192, NCW=4 → 4 chunks per warp per K pass).
//
// Page layout — split-activation design (Phase 6 occupancy fix):
//   pages[base_stage + w]       (w ∈ [0, NCW)) — activation slice for warp w
//                                   sv_bf<K_PER_WARP=K/NCW>; first 2 bytes of
//                                   warp-0 page reused for scalar dot result.
//   pages[base_stage + NCW + w] (w ∈ [0, NCW)) — weight row slice for warp w
//                                   sv_bf<K_PER_WARP>, one row of W[row, warp_range].
//   Total: 2 * NCW pages × PAGE_SIZE (= sv_bf<K_PER_WARP>) each.
//
// Activation is re-loaded every output row but stays hot in L2 (16 KB for
// llama-3.2-1B). Weight row changes per output row.
//
// Scratch: float[0 .. NUM_CONSUMER_WARPS) — per-warp partial dot sums.
//
// TK bar ID: kConsumerBarPartial = 14 (cross-warp partial-dot sync).

#pragma once

#include "kittens.cuh"
#include "ferrite_globals.cuh"
#include "ferrite_substrate.cuh"
#include "ferrite_tk_helpers.cuh"

namespace ferrite {
namespace ops {
namespace down_proj_residual {

// GROUP-PAGE using ScalarGroupLayout<NCW, K_PER_WARP, PAGE_SIZE>.
// GROUP_SIZE = PAGE_SIZE/(K_PER_WARP*2).
// PAGE_SIZE=16384, K_PER_WARP=1024: GROUP_SIZE=8 → all 8 warps share one page → 2 pages total!
// PAGE_SIZE=16384, K_PER_WARP=256: GROUP_SIZE=32 → clamped to NCW → still 2 pages.
template <int NCW, int K_PER_WARP, int PAGE_SIZE_BYTES>
struct DownProjLayout {
    using SGL = ::ferrite::ScalarGroupLayout<NCW, K_PER_WARP, PAGE_SIZE_BYTES>;
    static constexpr int GROUP_SIZE = SGL::GROUP_SIZE;
    static constexpr int NUM_GROUPS = SGL::NUM_GROUPS;
    __host__ __device__ static constexpr int act_page(int w)    { return w / GROUP_SIZE; }
    __host__ __device__ static constexpr int weight_page(int w) { return NUM_GROUPS + w / GROUP_SIZE; }
    __host__ __device__ static constexpr int warp_byte_offset(int w) { return SGL::warp_byte_offset(w); }
};

constexpr int kConsumerBarPartial = 14;

// ---------- Loader --------------------------------------------------
//
// TMA-based load: activation x and one row of W_gemm.
// TK-canonical; no scalar bf16 casts in the load path.

// K_OFFSET: start of the K-slice this op covers (for 4-chunk down_proj splitting).
// K_FULL: full K dimension (row stride) of both x and W matrices.
//   x shape: [NUM_TOKENS, K_FULL]; W shape: [N, K_FULL] row-major.
//   Caller passes x+K_OFFSET and W+K_OFFSET (column-pre-offset); this loader
//   adds tok*K_FULL and row*K_FULL as row strides to reach the correct row base.
//   For non-chunked ops K_FULL==K (default), so behavior is unchanged.
template <typename Config, int K, int N, int NUM_TOKENS, int K_OFFSET = 0, int K_FULL = K>
__device__ __forceinline__ void loader(
    ferrite::bf16_cptr x,              // [NUM_TOKENS, K_FULL], pre-offset to column K_OFFSET
    ferrite::bf16_cptr W,              // [N, K_FULL] row-major, pre-offset to column K_OFFSET
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int tok                            // current token index within the batch
) {
    using warp = kittens::group<1>;
    constexpr int NCW        = Config::NUM_CONSUMER_WARPS;
    constexpr int K_PER_WARP = K / NCW;
    static_assert(K % NCW == 0, "down_proj: K must be divisible by NCW");
    static_assert(K_FULL >= K, "down_proj: K_FULL must be >= K");
    using DPL = DownProjLayout<NCW, K_PER_WARP, Config::PAGE_SIZE>;
    constexpr int GROUP_SIZE  = DPL::GROUP_SIZE;
    constexpr int NUM_GROUPS  = DPL::NUM_GROUPS;
    constexpr uint32_t slice_bytes = K_PER_WARP * sizeof(__nv_bfloat16);

    for (int row = blockIdx.x; row < N; row += gridDim.x) {
        #pragma unroll
        for (int w = 0; w < NCW; ++w) {
            int pg = base_stage + DPL::act_page(w);
            void* dst = reinterpret_cast<void*>(ss.pages[pg] + DPL::warp_byte_offset(w));
            if (kittens::laneid() == 0 && w % GROUP_SIZE == 0) {
                kittens::tma::expect_bytes(ss.page_ready[pg], slice_bytes * GROUP_SIZE);
            }
            // x is pre-offset to column K_OFFSET; use K_FULL as row stride.
            warp::tma::load_async(dst,
                const_cast<__nv_bfloat16*>(x + static_cast<size_t>(tok) * K_FULL
                    + static_cast<size_t>(w) * K_PER_WARP),
                slice_bytes, ss.page_ready[pg]);
        }
        #pragma unroll
        for (int w = 0; w < NCW; ++w) {
            int pg = base_stage + DPL::weight_page(w);
            void* dst = reinterpret_cast<void*>(ss.pages[pg] + DPL::warp_byte_offset(w));
            if (kittens::laneid() == 0 && w % GROUP_SIZE == 0) {
                kittens::tma::expect_bytes(ss.page_ready[pg], slice_bytes * GROUP_SIZE);
            }
            // W is pre-offset to column K_OFFSET; use K_FULL as row stride.
            warp::tma::load_async(dst,
                const_cast<__nv_bfloat16*>(W + static_cast<size_t>(row) * K_FULL
                    + static_cast<size_t>(w) * K_PER_WARP),
                slice_bytes, ss.page_ready[pg]);
        }
        __syncthreads();
    }
}

// ---------- Consumer ------------------------------------------------
//
// TK-canonical dot product: rv_fl<kChunkCols> K-chunk loop.
// BS-grep 0 (compute path): replaces __bfloat162float / __shfl_xor_sync /
// asm volatile bar.sync with warp::load / warp::mul / warp::sum /
// kittens::group<N>::sync(barId).

template <typename Config, int K, int N, int NUM_TOKENS, int K_OFFSET = 0>
__device__ __forceinline__ void consumer(
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int warp_in_role,
    int tok
) {
    (void)tok;
    constexpr int NCW        = Config::NUM_CONSUMER_WARPS;
    constexpr int K_PER_WARP = K / NCW;
    static_assert(K % NCW == 0, "down_proj_residual: K must be divisible by NCW");
    using DPL = DownProjLayout<NCW, K_PER_WARP, Config::PAGE_SIZE>;

    // kChunkCols: pick_chunk_cols(K_PER_WARP, PAGE_SIZE/32) for the dot product.
    constexpr int kChunkCols = ::ferrite::ferrite_pick_chunk_cols(K_PER_WARP, Config::PAGE_SIZE / 32);
    static_assert(K_PER_WARP % kChunkCols == 0);
    constexpr int NUM_CHUNKS = K_PER_WARP / kChunkCols;

    using sv_chunk_t = kittens::sv_bf<kChunkCols>;
    using rv_chunk_t = kittens::rv_fl<kChunkCols>;

    float* partial_sums = reinterpret_cast<float*>(ss.scratch);

    int iter = 0;
    for (int row = blockIdx.x; row < N; row += gridDim.x, ++iter) {
        (void)row;
        const int phase    = iter & 1;
        const int act_pg   = base_stage + DPL::act_page(warp_in_role);
        const int wt_pg    = base_stage + DPL::weight_page(warp_in_role);
        const int byte_off = DPL::warp_byte_offset(warp_in_role);

        kittens::wait(ss.page_ready[act_pg], phase);
        kittens::wait(ss.page_ready[wt_pg],  phase);

        float warp_dot = 0.0f;
        #pragma unroll
        for (int c = 0; c < NUM_CHUNKS; ++c) {
            const size_t off = byte_off + static_cast<size_t>(c) * kChunkCols * sizeof(__nv_bfloat16);
            const sv_chunk_t& act_sv = *reinterpret_cast<const sv_chunk_t*>(ss.pages[act_pg] + off);
            const sv_chunk_t& w_sv   = *reinterpret_cast<const sv_chunk_t*>(ss.pages[wt_pg]  + off);

            rv_chunk_t act_rv, w_rv;
            kittens::warp::load(act_rv, act_sv);
            kittens::warp::load(w_rv,   w_sv);
            kittens::warp::mul(act_rv, act_rv, w_rv);
            warp_dot += kittens::warp::sum(act_rv);
        }

        if (kittens::laneid() == 0) {
            partial_sums[warp_in_role] = warp_dot;
        }
        kittens::group<NCW>::sync(kConsumerBarPartial);

        if (warp_in_role == 0 && kittens::laneid() == 0) {
            float total = 0.0f;
            #pragma unroll
            for (int w = 0; w < NCW; ++w) total += partial_sums[w];
            __nv_bfloat16* out_slot = reinterpret_cast<__nv_bfloat16*>(
                ss.pages[base_stage + DPL::act_page(0)]);
            out_slot[0] = __float2bfloat16_rn(total);
            kittens::arrive(ss.page_done[base_stage + DPL::act_page(0)]);
        }
        __syncthreads();
    }
}

// ---------- Launcher ------------------------------------------------

template <typename Config, int K, int N, int NUM_TOKENS, int K_OFFSET = 0>
__device__ __forceinline__ void launcher(
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int tok                            // unused; present for token-loop symmetry
) {
    (void)ss; (void)base_stage; (void)tok;
    for (int row = blockIdx.x; row < N; row += gridDim.x) {
        (void)row;
        __syncthreads();
    }
}

// ---------- Storer --------------------------------------------------
//
// Read-modify-write residual update.  Two benign scalar bf16 conversions
// at the warp-store boundary (same class as rms_norm.cuh's 1 BS-grep hit).

template <typename Config, int K, int N, int NUM_TOKENS, int K_OFFSET = 0>
__device__ __forceinline__ void storer(
    ferrite::bf16_ptr residual,        // [NUM_TOKENS, N] — read-modify-write per token
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int tok
) {
    constexpr int NCW        = Config::NUM_CONSUMER_WARPS;
    constexpr int K_PER_WARP = K / NCW;
    using DPL = DownProjLayout<NCW, K_PER_WARP, Config::PAGE_SIZE>;
    int iter = 0;
    for (int row = blockIdx.x; row < N; row += gridDim.x, ++iter) {
        const int phase = iter & 1;
        kittens::wait(ss.page_done[base_stage + DPL::act_page(0)], phase);
        if (kittens::laneid() == 0) {
            const __nv_bfloat16* src_slot = reinterpret_cast<const __nv_bfloat16*>(
                ss.pages[base_stage + DPL::act_page(0)]);
            float fresh = __bfloat162float(src_slot[0]);
            float prev  = __bfloat162float((residual + static_cast<size_t>(tok) * N)[row]);
            (residual + static_cast<size_t>(tok) * N)[row] =
                __float2bfloat16_rn(prev + fresh);
        }
        __syncthreads();
    }
}

}  // namespace down_proj_residual
}  // namespace ops
}  // namespace ferrite
