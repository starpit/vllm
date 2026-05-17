// SPDX-License-Identifier: Apache-2.0
// Ferrite-TK shared compute helpers.
//
// TK-canonical register-tile primitives used across ferrite op bodies.
// Ported from TK's in-tree llama reference:
//   third_party/thunderkittens/tests/vm/llama_official/utils.cuh
//   third_party/thunderkittens/tests/vm/llama_official/attention_partial.cu
//
// These helpers replace the hand-rolled `ss.scratch` + `__shfl_xor_sync`
// reductions that used to live in every ferrite op body. Every compute op
// (`rms_norm`, `gemv_bf16`, `lm_head`, `fused_qkv_rope_cache`,
// `silu_upgate`, `down_proj_residual`, `attention_partial`) either calls
// one of these helpers directly or uses the same TK primitives
// (`warp::row_sum` / `warp::row_max` / `warp::mma_*` / `warp::broadcast_col`)
// in-line. The shape of ferrite's 4-warp-role split is unchanged; these
// helpers are the inner compute, not the role-level scaffolding.
//
// Three primitives:
//
//   rms_norm_vec<NUM_CONSUMER_WARPS, HIDDEN_DIM, BAR_ID>(rms_scale_smem,
//                                                        act_smem,
//                                                        eps,
//                                                        partial_sums_scratch)
//       -> rv_fl<HIDDEN_DIM / NUM_CONSUMER_WARPS>
//     Returns the warp-sliced register-vector of RMS-normalised activations.
//     Mirrors `utils.cuh::rms_norm`, generalised on HIDDEN_DIM (TK upstream
//     hard-codes 2048) and parameterised on BAR_ID (TK upstream uses bar 0,
//     which collides with `__syncthreads()` — ferrite mega composes
//     multiple ops with active loader/storer warps, so the caller must
//     pick a named-barrier ID in [1, 15] that no other op in the variant
//     claims). `partial_sums_scratch` is a caller-owned fp32 buffer of at
//     least NUM_CONSUMER_WARPS floats inside `ss.scratch[]`.
//
//   matvec(out_smem, weights_smem, activations_vec)
//     Single-warp matvec into `sv_fl<16>` output via `warp::broadcast_col`
//     + `warp::mul` + `warp::row_sum` + `atomicAdd`. Ported verbatim from
//     `utils.cuh::matvec` Hopper variant. Multiple consumer warps call
//     with disjoint K-slices to produce the full output chunk; atomics
//     serialise the partial-sum accumulation.
//
//   store_n_rows<N>(dst_sv[N], src_rt, row_base)
//     Unpack N contiguous rows of a 16-row rt_fl register tile into N
//     sv_bf shared vectors. Generalises TK's `store_4_rows`
//     (`llama_official/attention_partial.cu`) for arbitrary GQA_RATIO.
//     Batched lane-parallel implementation: every lane whose groupId
//     holds a target row participates simultaneously in a single pass
//     over `src.tiles[0][j]`. For N==4, row_base==0/4/8/12 this reduces
//     to TK's 16-active-lanes single-branch pattern bit-for-bit; for
//     N==8/16 at aligned row_base it uses all 32 lanes at once; for
//     unaligned N the active-lane count is the number of distinct
//     groupIds covered by the target-row range (always the hardware
//     maximum for that row set). No per-row loop around the mma-store,
//     no throughput cliff vs TK upstream.

#pragma once

#include "kittens.cuh"

namespace ferrite {
namespace tk {

// --------------------------------------------------------------------
// rms_norm_vec — returns rv_fl of normalised activations (warp slice)
// --------------------------------------------------------------------

// Port of `utils.cuh::rms_norm` (TK llama_official), generalised on
// HIDDEN_DIM and NUM_CONSUMER_WARPS (TK hard-codes both for Llama-1B).
//
// Each consumer warp owns a HIDDEN_DIM / NUM_CONSUMER_WARPS slice of the
// activation row. Per-warp sum-of-squares is deposited into
// `partial_sums_scratch[warpid()]`, then all warps read the full array
// to compute the global variance. Final normalised activations stay in
// the warp's register vector `activations_vec` for downstream ops (matvec
// chain, silu-upgate, etc.) to consume directly without a round-trip
// through shared memory.
//
// Caller must provide `partial_sums_scratch` pointing at
// `NUM_CONSUMER_WARPS` fp32 slots inside `ss.scratch[]`, and a `BAR_ID`
// that does not collide with any other op or with bar 0 (__syncthreads).

// Compute the RMS scaling factor `rsqrt(mean(x^2) + eps)` across all
// consumer warps' register-vector slices, given a pre-computed rv_fl
// holding this warp's slice of the input row. Does the warp-local
// sum-of-squares reduction, deposits per-warp partial into scratch,
// syncs on BAR_ID, and computes the global mean. Returned as a scalar
// replicated across all lanes of the warp (ready to use in a
// `warp::mul(rv, rv, scale)` call).
//
// Used directly by `fused_add_rms_norm` (where the rv comes from a
// residual-add in registers, not from a shared load) and internally by
// `rms_norm_vec` (which prepends a `warp::load` from the activation sv).
template <int NUM_CONSUMER_WARPS, int HIDDEN_DIM, int BAR_ID, typename RV>
__device__ static inline float rms_norm_scale_from_rv(
    const RV &activations_vec,
    float rms_norm_eps,
    float *partial_sums_scratch
) {
    static_assert(BAR_ID >= 1 && BAR_ID <= 15,
                  "rms_norm_scale_from_rv: BAR_ID must be in [1, 15] "
                  "(bar 0 is __syncthreads)");
    RV sq_activations_vec;
    kittens::warp::copy(sq_activations_vec, activations_vec);
    kittens::warp::mul(sq_activations_vec, sq_activations_vec, sq_activations_vec);
    const float partial_sum = kittens::warp::sum(sq_activations_vec);

    if (kittens::laneid() == 0) {
        partial_sums_scratch[kittens::warpid()] = partial_sum;
    }
    kittens::group<NUM_CONSUMER_WARPS>::sync(BAR_ID);

    float full_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < NUM_CONSUMER_WARPS; ++i) {
        full_sum += partial_sums_scratch[i];
    }
    const float variance = full_sum / static_cast<float>(HIDDEN_DIM);
    return rsqrtf(variance + rms_norm_eps);
}

// Load activations from shared, compute the RMS scale, apply it in
// registers, multiply by the rms weight row, return the normalised
// register vector. Port of `utils.cuh::rms_norm` (TK llama_official).
template <int NUM_CONSUMER_WARPS, int HIDDEN_DIM, int BAR_ID, typename SV>
__device__ static inline auto rms_norm_vec(
    const SV &rms_scale_smem,
    const SV &activations_smem,
    float rms_norm_eps,
    float *partial_sums_scratch
) {
    using rv_t = kittens::rv_fl<SV::length>;
    rv_t activations_vec;
    kittens::warp::load(activations_vec, activations_smem);

    const float rms_scale =
        rms_norm_scale_from_rv<NUM_CONSUMER_WARPS, HIDDEN_DIM, BAR_ID>(
            activations_vec, rms_norm_eps, partial_sums_scratch);

    kittens::warp::mul(activations_vec, activations_vec, rms_scale);
    rv_t rms_scale_vec;
    kittens::warp::load(rms_scale_vec, rms_scale_smem);
    kittens::warp::mul(activations_vec, activations_vec, rms_scale_vec);
    return activations_vec;
}

// --------------------------------------------------------------------
// matvec — one-warp partial matvec, PLAIN STORE into exclusive per-warp sv_fl<16>.
// Matches TK utils.cuh::matvec (non-Blackwell/Hopper CUDA-core path) exactly.
//
// Each consumer warp writes to its OWN exclusive out_smem slot (no atomic contention).
// After all k-chunks, storer calls matvec_reduce to sum across all NCW per-warp slots.
template <typename ST>
__device__ static inline void matvec(
    kittens::sv_fl<ST::rows> &out_smem,      // PER-WARP exclusive slot in scratch
    const ST &weights_smem,
    const kittens::rv_fl<ST::cols> &activations_vec
) {
    using rt_t  = kittens::rt_fl<ST::rows, ST::cols>;
    using rrv_t = typename rt_t::row_vec;
    using rcv_t = typename rt_t::col_vec;
    using rv_t  = kittens::rv_fl<ST::rows>;

    rrv_t row_activations;
    kittens::warp::copy(row_activations, activations_vec);

    rt_t broadcast_activations, weights;
    kittens::warp::broadcast_col(broadcast_activations, row_activations);
    kittens::warp::load(weights, weights_smem);
    kittens::warp::mul(broadcast_activations, broadcast_activations, weights);

    rcv_t sum_col_vec;
    kittens::warp::row_sum(sum_col_vec, broadcast_activations);

    rv_t sum_vec;
    kittens::warp::copy(sum_vec, sum_col_vec);

    if (kittens::laneid() < ST::rows) {
        out_smem[kittens::laneid()] = sum_vec[0][0]; // plain store, no atomic
    }
    kittens::warp::sync();
}

// --------------------------------------------------------------------
// tanh_softcap_vec — apply `x = tanhf(x / cap) * cap` to every lane
// slot of a register vector, in-place. Used by Gemma2's final-logit
// softcap (and only Gemma2 — every other arch passes cap == 0 which
// short-circuits at codegen time to identity).
//
// Per-lane scalar math; no cross-warp coordination, no shared
// memory. Iteration shape `outer_dim x inner_dim` comes from TK's
// rv layout (`include/types/register/rv.cuh`).
// --------------------------------------------------------------------

template <typename RV>
__device__ static inline void tanh_softcap_vec(RV &vec, float cap) {
    const float inv_cap = 1.0f / cap;
    #pragma unroll
    for (int i = 0; i < RV::outer_dim; ++i) {
        #pragma unroll
        for (int j = 0; j < RV::inner_dim; ++j) {
            const float x = static_cast<float>(vec.data[i][j]) * inv_cap;
            vec.data[i][j] = tanhf(x) * cap;
        }
    }
}

// matvec_reduce — sum NCW per-warp partial results from scratch into rv_fl<16>.
// Matches TK utils.cuh::matvec_reduce.
template <int NCW>
__device__ static inline void matvec_reduce(
    uint8_t *scratch,
    kittens::rv_fl<16> &sum_vec
) {
    using sv_t = kittens::sv_fl<16>;
    constexpr int SCRATCH_BYTES_PER_WARP = 16 * sizeof(float);
    kittens::rv_fl<16> part_vec;
    kittens::warp::zero(sum_vec);
    #pragma unroll
    for (int i = 0; i < NCW; i++) {
        sv_t &part = *reinterpret_cast<sv_t *>(scratch + i * SCRATCH_BYTES_PER_WARP);
        kittens::warp::load(part_vec, part);
        kittens::warp::add(sum_vec, sum_vec, part_vec);
    }
}

// output_pipe_arrive/wait — OUTPUT_PIPELINE_STAGES=3 semaphore pattern matching TK.
// outputs_arrived: NCW consumer warps each arrive → storer can read (count=NCW)
// outputs_finished: storer arrives once → consumer can reuse slot (count=1)
static constexpr int OUTPUT_PIPELINE_STAGES = 3;

struct OutputPipeSems {
    kittens::semaphore outputs_arrived[OUTPUT_PIPELINE_STAGES];
    kittens::semaphore outputs_finished[OUTPUT_PIPELINE_STAGES];
    // init must be called by lane 0 of warp 0, followed by a CTA sync
    __device__ void init(int NCW) {
        for (int i = 0; i < OUTPUT_PIPELINE_STAGES; i++) {
            kittens::init_semaphore(outputs_arrived[i],  NCW);
            kittens::init_semaphore(outputs_finished[i], 1);
        }
    }
};

// --------------------------------------------------------------------
// store_n_rows — unpack N rows of rt_fl<16, K> into N sv_bf<K>
// --------------------------------------------------------------------

// Register-tile layout for rt_fl<16, cols> under the m16nk mma
// accumulator convention (per lane, per 16x16 base tile `tiles[0][j]`):
//
//   groupId = lane / 4                  ∈ [0, 8)
//   lig     = lane % 4                  ∈ [0, 4)
//
//   data[0] : row = groupId,     cols { lig*2,     lig*2+1 }   (tile half 0)
//   data[1] : row = groupId + 8, cols { lig*2,     lig*2+1 }   (tile half 0)
//   data[2] : row = groupId,     cols { lig*2 + 8, lig*2 + 9 } (tile half 1)
//   data[3] : row = groupId + 8, cols { lig*2 + 8, lig*2 + 9 } (tile half 1)
//
// Each lane owns fp32 values for exactly TWO rows of the 16-row tile:
// row groupId (via data[0]+data[2]) and row groupId+8 (via data[1]+data[3]).
// To write K target rows of the tile into K shared sv_bf destinations we
// split the work into two parallel passes:
//
//   Upper pass — target rows in [row_base, min(8, row_base+N)) are held
//     by the lanes with groupId in that exact range. data[0]+data[2]
//     source. Each participating lane writes one target row (its groupId
//     row) across all `src.width` col-tiles.
//
//   Lower pass — target rows in [max(8, row_base), row_base+N) are held
//     by lanes with groupId = target_row - 8, i.e. groupId in
//     [max(0, row_base-8), max(0, row_base+N-8)). data[1]+data[3] source.
//
// For N==4 aligned this reduces to exactly TK's `store_4_rows` pattern:
// 16 lanes active, a single branch, all src.width col-tiles in one pass.
// For N==8 aligned (row_base==0 or 8) ALL 32 lanes are active for one
// pass (full-warp throughput). For N==16 both passes run, each with 32
// lanes. Unaligned N (e.g. Qwen2's GQA_RATIO==5) uses the hardware
// maximum for the covered groupId range — no per-row loop, no throughput
// cliff.
//
// Generalisation of TK's `store_4_rows`
// (third_party/thunderkittens/tests/vm/llama_official/attention_partial.cu).
// N must be in [1, 16] and row_base + N <= 16.
template <int N, typename SV, typename RT>
__device__ static inline void store_n_rows(
    SV (&dst)[N],
    const RT &src,
    int row_base
) {
    static_assert(N > 0 && N <= 16, "store_n_rows: N must be in [1, 16]");
    static_assert(RT::rows == 16, "store_n_rows: src must be a 16-row rt tile");
    static_assert(SV::length == RT::cols,
                  "store_n_rows: dst length must match src cols");

    using T2 = typename RT::dtype;
    using U  = typename SV::dtype;
    using U2 = typename kittens::base_types::packing<U>::packed_type;

    const int lane    = kittens::laneid();
    const int groupId = lane / 4;
    const int lig     = lane % 4;

    // ---- Upper pass: rows in [row_base, upper_hi) held via data[0]+data[2].
    // `row_base` is a runtime parameter (GQA lowering picks it per CTA);
    // branches below are uniform within a groupId (4 lanes per groupId all
    // take the same path), so no warp divergence beyond the inherent
    // lane-gating that TK's original single-branch N==4 pattern also has.
    const int upper_hi = (row_base + N < 8) ? (row_base + N) : 8;
    if (row_base < upper_hi && groupId >= row_base && groupId < upper_hi) {
        const int dst_index = groupId - row_base;
        const uint32_t dst_ptr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&dst[dst_index].data[0]));
        #pragma unroll
        for (int j = 0; j < src.width; ++j) {
            U2 tmp0 = kittens::base_types::convertor<U2, T2>::convert(
                src.tiles[0][j].data[0]);
            U2 tmp1 = kittens::base_types::convertor<U2, T2>::convert(
                src.tiles[0][j].data[2]);
            const int col_idx = lig * 2 + j * 16;
            kittens::move<U2>::sts(dst_ptr + sizeof(U) * col_idx,       tmp0);
            kittens::move<U2>::sts(dst_ptr + sizeof(U) * (col_idx + 8), tmp1);
        }
    }

    // ---- Lower pass: rows in [max(8, row_base), row_base+N)
    //                  held via data[1]+data[3]; groupId = target_row - 8.
    const int lower_lo_g = (row_base > 8) ? (row_base - 8) : 0;
    const int lower_hi_g = (row_base + N > 8) ? (row_base + N - 8) : 0;
    if (lower_lo_g < lower_hi_g &&
        groupId >= lower_lo_g && groupId < lower_hi_g) {
        const int dst_index = (groupId + 8) - row_base;
        const uint32_t dst_ptr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&dst[dst_index].data[0]));
        #pragma unroll
        for (int j = 0; j < src.width; ++j) {
            U2 tmp0 = kittens::base_types::convertor<U2, T2>::convert(
                src.tiles[0][j].data[1]);
            U2 tmp1 = kittens::base_types::convertor<U2, T2>::convert(
                src.tiles[0][j].data[3]);
            const int col_idx = lig * 2 + j * 16;
            kittens::move<U2>::sts(dst_ptr + sizeof(U) * col_idx,       tmp0);
            kittens::move<U2>::sts(dst_ptr + sizeof(U) * (col_idx + 8), tmp1);
        }
    }
    kittens::warp::sync();
}

} // namespace tk

// ---------- Shared GROUP-PAGE layout helpers ----------------------------
//
// GROUP-PAGE: GROUP_SIZE warps share one page. Matches TK batch-vm design:
// PAGE_SIZE=16KB, kChunkCols=pick_chunk_cols(K_PER_WARP, PAGE_SIZE/32).
// With PAGE_SIZE=16384:
//   K_PER_WARP=256 (hidden, NCW=8): kChunkCols=256 → 1 chunk, GROUP_SIZE=2
//   K_PER_WARP=1024 (intermediate, NCW=8): kChunkCols=512 → 2 chunks, GROUP_SIZE=1
// vs old kGroupChunkCols=32: 16 chunks per warp → 16x more sync overhead.

__host__ __device__ constexpr int ferrite_pick_chunk_cols(int k_per_warp, int max_cols) {
    const int start = (k_per_warp < max_cols) ? k_per_warp : max_cols;
    for (int c = start; c >= 16; c -= 16) {
        if (k_per_warp % c == 0) return c;
    }
    return 16;
}

// Tile-based: kChunkCols = PAGE_SIZE/128 = WARPS_PER_PAGE × 16 × 2 inverse.
// Matches TK matvec_pipeline.cuh: STAGE_PAGES=4, WARPS_PER_PAGE=4, kChunkCols=128.
// For ANY K_PER_WARP: kChunkCols=128, GROUP_SIZE=4, NUM_GROUPS=NCW/4.
// K_PER_WARP/kChunkCols = NUM_K_CHUNKS (1 for K=2048/NCW=128, 4 for K=8192/NCW=512).
// Pages = 1 + NUM_GROUPS × STAGES = 1 + 4×3 = 13 always for NCW=16, PAGE_SIZE=16KB ✓
template <int NCW, int K_PER_WARP, int PAGE_SIZE_BYTES>
struct TileGroupLayout {
    // kChunkCols = PAGE_SIZE/128 gives GROUP_SIZE=4 (= WARPS_PER_PAGE) always.
    // Must also divide K_PER_WARP: K_PER_WARP = hidden_dim/NCW is always divisible by 128
    // (hidden_dim ∈ {2048, 3072, ...} all divisible by NCW × 128 = 16 × 128 = 2048).
    static constexpr int kChunkCols = PAGE_SIZE_BYTES / 128;
    static constexpr int GROUP_SIZE_RAW = PAGE_SIZE_BYTES / (16 * kChunkCols * 2);
    static constexpr int GROUP_SIZE = (GROUP_SIZE_RAW <= NCW) ? GROUP_SIZE_RAW : NCW;
    static constexpr int NUM_GROUPS = (NCW + GROUP_SIZE - 1) / GROUP_SIZE;

    __host__ __device__ static constexpr int warp_byte_offset(int warp_id) {
        return (warp_id % GROUP_SIZE) * 16 * kChunkCols * 2;
    }
};

// Scalar group layout for dot-product ops (down_proj_residual):
// packs as many warps' K_PER_WARP-element slices as fit per page.
template <int NCW, int K_PER_WARP, int PAGE_SIZE_BYTES>
struct ScalarGroupLayout {
    static constexpr int SLICE_BYTES    = K_PER_WARP * 2;
    static constexpr int GROUP_SIZE_RAW = PAGE_SIZE_BYTES / SLICE_BYTES;
    static constexpr int GROUP_SIZE = (GROUP_SIZE_RAW >= 1)
        ? ((GROUP_SIZE_RAW <= NCW) ? GROUP_SIZE_RAW : NCW) : 1;
    static constexpr int NUM_GROUPS = (NCW + GROUP_SIZE - 1) / GROUP_SIZE;

    __host__ __device__ static constexpr int warp_byte_offset(int warp_id) {
        return (warp_id % GROUP_SIZE) * SLICE_BYTES;
    }
};

} // namespace ferrite
