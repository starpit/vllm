// SPDX-License-Identifier: Apache-2.0
// Ferrite-owned TK 2.0 gemv_bf16 op — 4 warp-role functions.
//
// Math: `out[N] = W[N, K] @ x[K]`.
// bf16 in/out, fp32 accumulator. Decode path (bs=1) matmul.
//
// GROUP-PAGE design (TK-style shared weight tiles):
//   GROUP_SIZE = PAGE_SIZE / (16 * kGroupChunkCols * 2)
//             = 8192 / (16 * 32 * 2) = 8 warps share one weight page.
//   NUM_GROUPS = NCW / GROUP_SIZE  (e.g. NCW=8 → 1 group, NCW=4 → 1 group)
//   kGroupChunkCols = 32 elements per warp per K-chunk.
//
// Within each group weight page (PAGE_SIZE bytes), warp `w`'s
// data is at byte offset `(w % GROUP_SIZE) * 16 * kGroupChunkCols * 2`,
// forming a contiguous `st_bf<16, kGroupChunkCols>` tile. ✓
//
// Page layout:
//   pages[base_stage + 0]
//       — sv_bf<K> activation, loaded once per CTA, shared across all
//         block-iters AND all K-chunks.
//   pages[base_stage + 1 + stage * NUM_GROUPS + group_id]
//       — GROUP_SIZE-warp shared weight tile for ring stage `stage`
//         and group `group_id`. Holds GROUP_SIZE × 16 × kGroupChunkCols
//         bf16 elements = exactly PAGE_SIZE bytes.
// Total pages: 1 + NUM_GROUPS * STAGES
//   NCW=4  → 1 + 1*2 = 3  pages (PAGE_SIZE=8KB each)
//   NCW=16 → 1 + 4*2 = 9  pages
//
// K-inner loop: each warp processes K_PER_WARP = K/NCW elements in
// NUM_K_CHUNKS_PER_WARP = K_PER_WARP / kGroupChunkCols chunks.
// For NCW=16, K=2048: K_PER_WARP=128, NUM_K_CHUNKS=2 per warp.
// For NCW=4,  K=2048: K_PER_WARP=512, NUM_K_CHUNKS=8 per warp.
//
// page_done count = GROUP_SIZE (all warps in group must arrive before
// loader can refill the shared page).

#pragma once

#include "kittens.cuh"
#include "ferrite_globals.cuh"
#include "ferrite_substrate.cuh"
#include "ferrite_tk_helpers.cuh"

namespace ferrite {
namespace ops {
namespace gemv_bf16 {

constexpr int kActPageOff         = 0;
constexpr int kConsumerBarInit    = 3;
constexpr int kConsumerBarPublish = 4;

// GROUP-PAGE layout using TileGroupLayout<NCW, K_PER_WARP, PAGE_SIZE>.
// kChunkCols = ferrite_pick_chunk_cols(K_PER_WARP, PAGE_SIZE/32).
// With PAGE_SIZE=16384, K_PER_WARP=256 (NCW=8): kChunkCols=256 → 1 chunk, GROUP_SIZE=2.
// With PAGE_SIZE=16384, K_PER_WARP=512 (NCW=4): kChunkCols=512 → 1 chunk, GROUP_SIZE=1.
template <int NCW, int K_PER_WARP, int PAGE_SIZE_BYTES>
struct GroupLayout {
    using GL = ::ferrite::TileGroupLayout<NCW, K_PER_WARP, PAGE_SIZE_BYTES>;
    static constexpr int kChunkCols = GL::kChunkCols;
    static constexpr int GROUP_SIZE = GL::GROUP_SIZE;
    static constexpr int NUM_GROUPS = GL::NUM_GROUPS;
    __host__ __device__ static constexpr int weight_page(int stage, int warp_id) {
        return 1 + stage * NUM_GROUPS + warp_id / GROUP_SIZE;
    }
    __host__ __device__ static constexpr int warp_byte_offset(int warp_id) {
        return GL::warp_byte_offset(warp_id);
    }
};

// ---------- Loader --------------------------------------------------

template <typename Config, int K, int N>
__device__ __forceinline__ void loader(
    ferrite::bf16_cptr x,           // [NUM_TOKENS, K] — tok selects row
    ferrite::bf16_cptr W,           // [N, K] row-major
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int tok                         // current token index within the batch
) {
    constexpr int NCW                    = Config::NUM_CONSUMER_WARPS;
    constexpr int STAGES                 = Config::INSTRUCTION_PIPE_STAGES;
    constexpr int K_PER_WARP             = K / NCW;
    using GL = GroupLayout<NCW, K_PER_WARP, Config::PAGE_SIZE>;
    constexpr int kChunkCols             = GL::kChunkCols;
    constexpr int NUM_K_CHUNKS_PER_WARP  = K_PER_WARP / kChunkCols;
    constexpr int GROUP_SIZE             = GL::GROUP_SIZE;
    constexpr int NUM_GROUPS             = GL::NUM_GROUPS;

    static_assert(N % 16 == 0,  "gemv_bf16: N must be a multiple of 16");
    static_assert(K % NCW == 0, "gemv_bf16: K must be divisible by NCW");
    static_assert(NCW % GROUP_SIZE == 0,
                  "gemv_bf16: NCW must be divisible by GROUP_SIZE");
    static_assert(K_PER_WARP % kChunkCols == 0,
                  "gemv_bf16: K_PER_WARP must be divisible by kChunkCols");

    using warp = kittens::group<1>;
    constexpr uint32_t act_bytes         = K * sizeof(__nv_bfloat16);
    constexpr uint32_t group_tile_bytes  = GROUP_SIZE * 16 * kChunkCols
                                           * sizeof(__nv_bfloat16);  // = PAGE_SIZE
    constexpr uint32_t warp_row_bytes    = kChunkCols * sizeof(__nv_bfloat16);

    const int num_blocks = N / 16;

    // One-shot activation load — shared across all block-iters and K-chunks.
    void* act_page = reinterpret_cast<void*>(ss.pages[base_stage + kActPageOff]);
    if (kittens::laneid() == 0) {
        kittens::tma::expect_bytes(
            ss.page_ready[base_stage + kActPageOff], act_bytes);
    }
    warp::tma::load_async(
        act_page,
        const_cast<__nv_bfloat16*>(x + static_cast<size_t>(tok) * K),
        act_bytes,
        ss.page_ready[base_stage + kActPageOff]);

    // num_positions = (num_blocks × NUM_K_CHUNKS_PER_WARP) per CTA.
    const int num_positions = num_blocks * NUM_K_CHUNKS_PER_WARP;

    for (int pos = 0; pos < num_positions; ++pos) {
        const int block   = blockIdx.x + (pos / NUM_K_CHUNKS_PER_WARP) * gridDim.x;
        if (block >= num_blocks) break;
        const int k_chunk = pos % NUM_K_CHUNKS_PER_WARP;
        const int stage   = pos % STAGES;
        const int prev_phase = ((pos - STAGES) / STAGES) & 1;

        // Wait for each group page to be released by its group leader.
        // (page_done count=1, only group leader arrives — see consumer.)
        if (pos >= STAGES) {
            #pragma unroll
            for (int g = 0; g < NUM_GROUPS; ++g) {
                kittens::wait(
                    ss.page_done[base_stage + GL::weight_page(stage, g * GROUP_SIZE)],
                    prev_phase);
            }
        }
        // Note: loader accesses all pages for all groups in this stage,
        // but waits once per group (indexed by group leader warp = g*GROUP_SIZE).

        // Signal how many bytes each group page will receive.
        if (kittens::laneid() == 0) {
            #pragma unroll
            for (int g = 0; g < NUM_GROUPS; ++g) {
                kittens::tma::expect_bytes(
                    ss.page_ready[base_stage + GL::weight_page(stage, g * GROUP_SIZE)],
                    group_tile_bytes);
            }
        }

        // Load GROUP_SIZE warps' weight data per group page.
        // Within each group page the layout is:
        //   [warp_in_group=0: rows 0..15 of K-slice][warp_in_group=1: ...][...]
        const size_t block_row_base = static_cast<size_t>(block) * 16;
        #pragma unroll
        for (int w = 0; w < NCW; ++w) {
            int pg = base_stage + GL::weight_page(stage, w);
            __nv_bfloat16* dst_page = reinterpret_cast<__nv_bfloat16*>(ss.pages[pg]);
            int warp_in_group       = w % GROUP_SIZE;
            const size_t col_base   =
                static_cast<size_t>(w) * K_PER_WARP
                + static_cast<size_t>(k_chunk) * kChunkCols;
            #pragma unroll
            for (int r = 0; r < 16; ++r) {
                const __nv_bfloat16* src_row =
                    W + (block_row_base + r) * K + col_base;
                // Destination: warp_in_group's row r within the group page.
                __nv_bfloat16* dst =
                    dst_page + (warp_in_group * 16 + r) * kChunkCols;
                warp::tma::load_async(
                    dst,
                    const_cast<__nv_bfloat16*>(src_row),
                    warp_row_bytes,
                    ss.page_ready[pg]);
            }
        }
    }
}

// ---------- Consumer ------------------------------------------------
//
// Per-warp exclusive partial-sum slots; warp 0 reduces and signals the
// storer via page_done[kActPageOff] (one arrive per output block).
// No OutputPipeSems — proven compatible with cooperative grid.sync().

template <typename Config, int K, int N>
__device__ __forceinline__ void consumer(
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int warp_in_role,
    int tok
) {
    constexpr int NCW                   = Config::NUM_CONSUMER_WARPS;
    constexpr int STAGES                = Config::INSTRUCTION_PIPE_STAGES;
    constexpr int K_PER_WARP            = K / NCW;
    using GL = GroupLayout<NCW, K_PER_WARP, Config::PAGE_SIZE>;
    constexpr int kChunkCols            = GL::kChunkCols;
    constexpr int NUM_K_CHUNKS_PER_WARP = K_PER_WARP / kChunkCols;
    constexpr int GROUP_SIZE            = GL::GROUP_SIZE;
    static_assert(N % 16 == 0);
    static_assert(K % NCW == 0);

    using warp             = kittens::group<1>;
    using chunk_weight_st  = kittens::st_bf<16, kChunkCols, /*_swizzle=*/false>;
    using chunk_act_sv     = kittens::sv_bf<kChunkCols>;
    using out_sv           = kittens::sv_fl<16>;

    // Scratch layout:
    //   [0..NCW*64)         NCW × sv_fl<16> per-warp partial sum slots (fp32)
    //   [NCW*64..NCW*64+32) sv_bf<16> staging for storer's TMA
    uint8_t* scratch_base = ss.scratch;
    out_sv& my_out_smem = *reinterpret_cast<out_sv*>(
        scratch_base + warp_in_role * 16 * sizeof(float));

    // Activation: K/NCW elements per warp, each kChunkCols-wide.
    auto* act_chunks = reinterpret_cast<chunk_act_sv*>(
        ss.pages[base_stage + kActPageOff]);

    kittens::wait(ss.page_ready[base_stage + kActPageOff], tok & 1);
    warp::sync();

    const int num_blocks = N / 16;
    int block_iter = 0;
    for (int block = blockIdx.x; block < num_blocks; block += gridDim.x, ++block_iter) {
        // Zero this warp's exclusive partial-sum slot.
        if (kittens::laneid() < 16) {
            my_out_smem[kittens::laneid()] = 0.0f;
        }
        kittens::group<NCW>::sync(kConsumerBarInit);

        #pragma unroll
        for (int k_chunk = 0; k_chunk < NUM_K_CHUNKS_PER_WARP; ++k_chunk) {
            const int pos         = block_iter * NUM_K_CHUNKS_PER_WARP + k_chunk;
            const int stage       = pos % STAGES;
            const int ready_phase = (pos / STAGES) & 1;
            const int pg          = base_stage + GL::weight_page(stage, warp_in_role);

            kittens::wait(ss.page_ready[pg], ready_phase);

            chunk_weight_st& weight_chunk = *reinterpret_cast<chunk_weight_st*>(
                ss.pages[pg] + GL::warp_byte_offset(warp_in_role));

            chunk_act_sv& act_chunk_slice =
                act_chunks[warp_in_role * NUM_K_CHUNKS_PER_WARP + k_chunk];
            kittens::rv_fl<kChunkCols> act_rv;
            warp::load(act_rv, act_chunk_slice);

            ferrite::tk::matvec(my_out_smem, weight_chunk, act_rv);

            if (kittens::laneid() == 0 && (warp_in_role % GROUP_SIZE == 0)) {
                kittens::arrive(ss.page_done[pg]);
            }
        }

        // All NCW consumer warps sync: partial sums in scratch ready.
        kittens::group<NCW>::sync(kConsumerBarPublish);

        // Warp 0: reduce NCW partial sums → bf16 staging, signal storer.
        if (warp_in_role == 0) {
            constexpr size_t STAGING_OFF = NCW * 16 * sizeof(float);
            kittens::rv_fl<16> out_rv;
            ferrite::tk::matvec_reduce<NCW>(scratch_base, out_rv);
            auto& out_bf = *reinterpret_cast<kittens::sv_bf<16>*>(
                scratch_base + STAGING_OFF);
            if (kittens::laneid() < 16) {
                out_bf[kittens::laneid()] = __float2bfloat16_rn(out_rv[0][0]);
            }
            warp::sync();
            if (kittens::laneid() == 0) {
                kittens::arrive(ss.page_done[base_stage + kActPageOff]);
            }
        }
    }
}

// ---------- Launcher ------------------------------------------------
template <typename Config, int K, int N>
__device__ __forceinline__ void launcher(
    ferrite::SharedState<Config>& ss, int base_stage, int tok
) { (void)ss; (void)base_stage; (void)tok; }

// ---------- Storer --------------------------------------------------
//
// Wait for consumer warp 0's page_done[kActPageOff] arrive (one per block,
// phase = block_iter & 1), then TMA store the pre-reduced bf16 staging area.

template <typename Config, int K, int N>
__device__ __forceinline__ void storer(
    ferrite::bf16_ptr out,
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int tok
) {
    constexpr int NCW = Config::NUM_CONSUMER_WARPS;
    static_assert(N % 16 == 0);

    using warp = kittens::group<1>;
    constexpr uint32_t out_block_bytes = 16 * sizeof(__nv_bfloat16);
    constexpr size_t STAGING_OFF = NCW * 16 * sizeof(float);

    uint8_t* scratch_base = ss.scratch;

    const int num_blocks = N / 16;
    int block_iter = 0;
    for (int block = blockIdx.x; block < num_blocks; block += gridDim.x, ++block_iter) {
        // Wait for consumer warp 0 to reduce partial sums and fill staging area.
        // Phase alternates: block 0→phase=0, block 1→phase=1, block 2→phase=0, ...
        const int phase = block_iter & 1;
        kittens::wait(ss.page_done[base_stage + kActPageOff], phase);

        // TMA store the pre-reduced bf16 staging area to global memory.
        auto& out_bf = *reinterpret_cast<kittens::sv_bf<16>*>(scratch_base + STAGING_OFF);
        warp::tma::store_async(
            static_cast<void*>(
                out + static_cast<size_t>(tok) * N
                    + static_cast<size_t>(block) * 16),
            &out_bf, out_block_bytes);
        kittens::tma::store_async_wait<0>();
    }
}

// ---------- Semaphore initialisation (called by controller) ---------
template <typename Config, int K, int N>
__host__ __device__ constexpr int num_semaphores() {
    // page_ready/page_done are handled by ferrite-substrate SharedState.
    return 0;
}

// Convenience: total op pages consumed (matches op_emit.rs op_page_count).
template <int NCW, int K_PER_WARP, int PAGE_SIZE_BYTES, int STAGES>
__host__ __device__ constexpr int total_pages() {
    return 1 + GroupLayout<NCW, K_PER_WARP, PAGE_SIZE_BYTES>::NUM_GROUPS * STAGES;
}

}  // namespace gemv_bf16
}  // namespace ops
}  // namespace ferrite
