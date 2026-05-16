// SPDX-License-Identifier: Apache-2.0
// Ferrite-owned TK 2.0 FusedQkvRopeCache op — 4 warp-role functions.
//
// One kernel fuses:
//   1. Packed QKV GEMM    qkv = W_packed @ x
//      (`x` is rms-normalised upstream by `fused_add_rms_norm`; no
//      rms here — TK's `rms_matvec_rope_append.cu` fuses rms
//      because ITS tape doesn't have a preceding fused-add-rms, but
//      ferrite's tape does.)
//   2. NEOX RoPE          Q' = rope(Q), K' = rope(K)
//   3. KV cache write     K', V → key_cache, value_cache at
//                           slot_mapping[0].
//   4. Q output           Q' → q_out.
//
// NEOX rope pattern (matches vLLM / ferrite host path so the
// prefill→decode KV cache stays consistent): within each head,
// dim d in [0, HEAD_DIM/2) pairs with dim d + HEAD_DIM/2. Rotation:
//   out[d]           = x[d] * cos[d]            - x[d + half] * sin[d]
//   out[d + half]    = x[d + half] * cos[d]     + x[d] * sin[d]
// (cos/sin reuse the same freq index d across the pair.)
//
// Parallelism: each CTA produces 32 consecutive output rows per
// iter — a rope-partnered PAIR of two 16-row blocks (block_a and
// block_b = block_a + HEAD_DIM/32 within the same head). Two
// sequential register-tile matvecs per warp per iter share the
// pre-loaded activation rv; the per-warp atomicAdd lands each
// warp's partial into separate sv_fl<16> accumulators (one per
// block). After every chunk across every warp has landed, warp 0
// applies NEOX rope between the two 16-row blocks' fp32 accumulators
// in registers (no __shfl — NEOX partner is in the OTHER rv, same
// lane index), converts to bf16, hands off to the storer.
//
// Grid: one CTA per block-pair. Native pair count is
// `NUM_HEADS_TOTAL * (HEAD_DIM / 32)` where
// `NUM_HEADS_TOTAL = NUM_Q_HEADS + 2 * NUM_KV_HEADS`. Persistent-
// thread loop handles any grid ≤ that count.
//
// Pair enumeration:
//   pair_idx         ∈ [0, NUM_HEADS_TOTAL * (HEAD_DIM/32))
//   head_linear      = pair_idx / (HEAD_DIM/32)
//   within_head      = pair_idx % (HEAD_DIM/32)
//   block_a          = head_linear * (HEAD_DIM/16) + within_head
//   block_b          = block_a + (HEAD_DIM/32)
// Family of the pair = family of `block_a` (both blocks live in the
// same head by construction, so same family):
//   head_linear < NUM_Q_HEADS                        → Q
//   head_linear < NUM_Q_HEADS + NUM_KV_HEADS         → K
//   otherwise                                        → V (no rope)
//
// Page layout (caller gives `base_stage`):
//   pages[base_stage + 0]                         — sv_bf<HIDDEN_DIM>
//                                                   activation x
//                                                   (pre-normalised).
//                                                   Loaded once per
//                                                   CTA; the K-chunk
//                                                   loop reads warp-
//                                                   sliced sub-chunks.
//   pages[base_stage + 1]                         — sv_bf<HEAD_DIM>
//                                                   cos/sin pair for
//                                                   the decode
//                                                   position (NEOX
//                                                   layout: first
//                                                   HEAD_DIM/2 are
//                                                   cos by freq,
//                                                   rest are sin).
//   pages[base_stage + 2 + 2 * (s * NCW + w)]     — st_bf<16,
//                                                   CHUNK_COLS>
//                                                   weight tile for
//                                                   BLOCK_A of pair
//                                                   (stage s, warp w).
//   pages[base_stage + 2 + 2 * (s * NCW + w) + 1] — weight tile for
//                                                   BLOCK_B.
// Total pages: `2 + 2 * NCW * STAGES`.
//
// Scratch layout (STAGES × (sv_fl<16> a-accum + sv_fl<16> b-accum)):
//   float[pair_stage * 32 .. + 16]   — a_accum (sv_fl<16>).
//   float[pair_stage * 32 + 16 .. + 32] — b_accum (sv_fl<16>).
// `STAGES * 32` floats total for the block-pair-iter ring.
//
// bar IDs: 7 (init), 8 (publish).

#pragma once

#include "kittens.cuh"
#include "ferrite_globals.cuh"
#include "ferrite_substrate.cuh"
#include "ferrite_tk_helpers.cuh"

namespace ferrite {
namespace ops {
namespace fused_qkv_rope_cache {

constexpr int kActPageOff    = 0;
constexpr int kCosSinPageOff = 1;

// GROUP-PAGE layout for QKV weight tiles (block_a and block_b).
// kChunkCols = ferrite_pick_chunk_cols(K_PER_WARP, PAGE_SIZE/32).
// STAGES=1: 2 fixed (act+cos_sin) + NUM_GROUPS_A + NUM_GROUPS_B = 2+2*NUM_GROUPS.
template <int NCW, int K_PER_WARP, int PAGE_SIZE_BYTES>
struct QkvGroupLayout {
    using GL = ::ferrite::TileGroupLayout<NCW, K_PER_WARP, PAGE_SIZE_BYTES>;
    static constexpr int kChunkCols = GL::kChunkCols;
    static constexpr int GROUP_SIZE = GL::GROUP_SIZE;
    static constexpr int NUM_GROUPS = GL::NUM_GROUPS;
    __host__ __device__ static constexpr int weight_page_a(int stage, int warp_id) {
        return 2 + stage * NUM_GROUPS * 2 + warp_id / GROUP_SIZE;
    }
    __host__ __device__ static constexpr int weight_page_b(int stage, int warp_id) {
        return 2 + stage * NUM_GROUPS * 2 + NUM_GROUPS + warp_id / GROUP_SIZE;
    }
    __host__ __device__ static constexpr int warp_byte_offset(int warp_id) {
        return GL::warp_byte_offset(warp_id);
    }
};

constexpr int kConsumerBarInit    = 7;
constexpr int kConsumerBarPublish = 8;

// ---------- Loader --------------------------------------------------

template <
    typename Config,
    int HIDDEN_DIM, int HEAD_DIM,
    int NUM_Q_HEADS, int NUM_KV_HEADS,
    bool BIASED, bool INTERLEAVED
>
__device__ __forceinline__ void loader(
    ferrite::bf16_cptr   x,               // [NUM_TOKENS, HIDDEN_DIM]
    ferrite::bf16_cptr   w_packed,        // [(Q+2KV)*HEAD_DIM, HIDDEN_DIM]
    ferrite::bf16_cptr   cos_sin_cache,   // [max_pos, HEAD_DIM]
    const uint32_t*      positions,       // [NUM_TOKENS]
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int tok                               // current token index within the batch
) {
    static_assert(!BIASED,
                  "fused_qkv_rope_cache: BIASED path not yet implemented");
    static_assert(!INTERLEAVED,
                  "fused_qkv_rope_cache: INTERLEAVED (GPT-J) path not yet "
                  "implemented; ferrite uses NEOX rope (pairs (d, d+head_dim/2)) "
                  "to stay compatible with Python vLLM's KV-cache write order");
    static_assert(HEAD_DIM % 32 == 0,
                  "HEAD_DIM must be a multiple of 32 (rope-pair = 2 × 16-row block)");

    constexpr int NCW                   = Config::NUM_CONSUMER_WARPS;
    constexpr int STAGES                = 1;
    constexpr int K_PER_WARP            = HIDDEN_DIM / NCW;
    using QGL = QkvGroupLayout<NCW, K_PER_WARP, Config::PAGE_SIZE>;
    constexpr int kChunkCols            = QGL::kChunkCols;
    constexpr int NUM_K_CHUNKS_PER_WARP = K_PER_WARP / kChunkCols;
    constexpr int GROUP_SIZE            = QGL::GROUP_SIZE;
    constexpr int NUM_GROUPS            = QGL::NUM_GROUPS;
    constexpr int BLOCKS_PER_HEAD       = HEAD_DIM / 16;
    constexpr int PAIRS_PER_HEAD        = HEAD_DIM / 32;
    constexpr int NUM_HEADS_TOTAL       = NUM_Q_HEADS + 2 * NUM_KV_HEADS;
    constexpr int TOTAL_PAIRS           = NUM_HEADS_TOTAL * PAIRS_PER_HEAD;
    static_assert(HIDDEN_DIM % NCW == 0);

    using warp = kittens::group<1>;
    constexpr uint32_t act_bytes        = HIDDEN_DIM * sizeof(__nv_bfloat16);
    constexpr uint32_t cos_sin_bytes    = HEAD_DIM   * sizeof(__nv_bfloat16);
    constexpr uint32_t group_tile_bytes = GROUP_SIZE * 16 * kChunkCols * sizeof(__nv_bfloat16);
    constexpr uint32_t warp_row_bytes   = kChunkCols * sizeof(__nv_bfloat16);

    // One-shot activation + cos/sin.
    void* act_page     = reinterpret_cast<void*>(ss.pages[base_stage + kActPageOff]);
    void* cos_sin_page = reinterpret_cast<void*>(ss.pages[base_stage + kCosSinPageOff]);
    if (kittens::laneid() == 0) {
        kittens::tma::expect_bytes(ss.page_ready[base_stage + kActPageOff],    act_bytes);
        kittens::tma::expect_bytes(ss.page_ready[base_stage + kCosSinPageOff], cos_sin_bytes);
    }
    warp::tma::load_async(
        act_page, const_cast<__nv_bfloat16*>(x + static_cast<size_t>(tok) * HIDDEN_DIM),
        act_bytes, ss.page_ready[base_stage + kActPageOff]);
    const uint32_t pos = positions[tok];
    const __nv_bfloat16* cos_sin_src =
        cos_sin_cache + static_cast<size_t>(pos) * HEAD_DIM;
    warp::tma::load_async(
        cos_sin_page, const_cast<__nv_bfloat16*>(cos_sin_src),
        cos_sin_bytes,
        ss.page_ready[base_stage + kCosSinPageOff]);

    // Pair loop.
    for (int pair_idx = blockIdx.x, pair_iter = 0;
         pair_idx < TOTAL_PAIRS;
         pair_idx += gridDim.x, ++pair_iter) {
        const int head_linear    = pair_idx / PAIRS_PER_HEAD;
        const int within_head    = pair_idx % PAIRS_PER_HEAD;
        const int block_a        = head_linear * BLOCKS_PER_HEAD + within_head;
        const int block_b        = block_a + PAIRS_PER_HEAD;
        const size_t row_a_base  = static_cast<size_t>(block_a) * 16;
        const size_t row_b_base  = static_cast<size_t>(block_b) * 16;

        #pragma unroll
        for (int k_chunk = 0; k_chunk < NUM_K_CHUNKS_PER_WARP; ++k_chunk) {
            const int pos_i      = pair_iter * NUM_K_CHUNKS_PER_WARP + k_chunk;
            const int stage      = pos_i % STAGES;
            const int prev_phase = ((pos_i - STAGES) / STAGES) & 1;

            if (pos_i >= STAGES) {
                #pragma unroll
                for (int g = 0; g < NUM_GROUPS; ++g) {
                    kittens::wait(ss.page_done[base_stage + QGL::weight_page_a(stage, g*GROUP_SIZE)], prev_phase);
                    kittens::wait(ss.page_done[base_stage + QGL::weight_page_b(stage, g*GROUP_SIZE)], prev_phase);
                }
            }

            if (kittens::laneid() == 0) {
                #pragma unroll
                for (int g = 0; g < NUM_GROUPS; ++g) {
                    kittens::tma::expect_bytes(
                        ss.page_ready[base_stage + QGL::weight_page_a(stage, g*GROUP_SIZE)], group_tile_bytes);
                    kittens::tma::expect_bytes(
                        ss.page_ready[base_stage + QGL::weight_page_b(stage, g*GROUP_SIZE)], group_tile_bytes);
                }
            }

            #pragma unroll
            for (int w = 0; w < NCW; ++w) {
                int pg_a = base_stage + QGL::weight_page_a(stage, w);
                int pg_b = base_stage + QGL::weight_page_b(stage, w);
                __nv_bfloat16* dst_a = reinterpret_cast<__nv_bfloat16*>(ss.pages[pg_a])
                    + QGL::warp_byte_offset(w) / sizeof(__nv_bfloat16);
                __nv_bfloat16* dst_b = reinterpret_cast<__nv_bfloat16*>(ss.pages[pg_b])
                    + QGL::warp_byte_offset(w) / sizeof(__nv_bfloat16);
                const size_t col_base = static_cast<size_t>(w) * K_PER_WARP
                    + static_cast<size_t>(k_chunk) * kChunkCols;
                #pragma unroll
                for (int r = 0; r < 16; ++r) {
                    warp::tma::load_async(dst_a + static_cast<size_t>(r)*kChunkCols,
                        const_cast<__nv_bfloat16*>(w_packed + (row_a_base+r)*HIDDEN_DIM + col_base),
                        warp_row_bytes, ss.page_ready[pg_a]);
                    warp::tma::load_async(dst_b + static_cast<size_t>(r)*kChunkCols,
                        const_cast<__nv_bfloat16*>(w_packed + (row_b_base+r)*HIDDEN_DIM + col_base),
                        warp_row_bytes, ss.page_ready[pg_b]);
                }
            }
        }
    }
}

// ---------- Consumer ------------------------------------------------

template <
    typename Config,
    int HIDDEN_DIM, int HEAD_DIM,
    int NUM_Q_HEADS, int NUM_KV_HEADS,
    bool BIASED, bool INTERLEAVED
>
__device__ __forceinline__ void consumer(
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int warp_in_role,
    int tok                               // current token index within the batch
) {
    static_assert(!BIASED);
    static_assert(!INTERLEAVED);
    static_assert(HEAD_DIM % 32 == 0);

    constexpr int NCW                   = Config::NUM_CONSUMER_WARPS;
    constexpr int STAGES                = 1;
    constexpr int K_PER_WARP            = HIDDEN_DIM / NCW;
    using QGL = QkvGroupLayout<NCW, K_PER_WARP, Config::PAGE_SIZE>;
    constexpr int kChunkCols            = QGL::kChunkCols;
    constexpr int NUM_K_CHUNKS_PER_WARP = K_PER_WARP / kChunkCols;
    constexpr int HALF_DIM              = HEAD_DIM / 2;
    constexpr int BLOCKS_PER_HEAD       = HEAD_DIM / 16;
    constexpr int PAIRS_PER_HEAD        = HEAD_DIM / 32;
    constexpr int NUM_HEADS_TOTAL       = NUM_Q_HEADS + 2 * NUM_KV_HEADS;
    constexpr int TOTAL_PAIRS           = NUM_HEADS_TOTAL * PAIRS_PER_HEAD;
    static_assert(HIDDEN_DIM % NCW == 0);

    using warp            = kittens::group<1>;
    using chunk_weight_st = kittens::st_bf<16, kChunkCols, /*_swizzle=*/false>;
    using chunk_act_sv    = kittens::sv_bf<kChunkCols>;
    using out_sv          = kittens::sv_fl<16>;

    auto* act_chunks = reinterpret_cast<chunk_act_sv*>(
        ss.pages[base_stage + kActPageOff]);

    const __nv_bfloat16* cos_sin_sv = reinterpret_cast<const __nv_bfloat16*>(
        ss.pages[base_stage + kCosSinPageOff]);

    // Scratch: [2*NCW*sv_fl<16>][2*sv_bf<16> staging (a+b)]
    constexpr size_t QKV_BF16_OFF = 2u * NCW * 16 * sizeof(float);
    uint8_t* scratch_base = ss.scratch;

    // Phase alternates with token index: tok=0 → phase 0, tok=1 → phase 1, etc.
    const int tok_phase = tok & 1;
    kittens::wait(ss.page_ready[base_stage + kActPageOff],    tok_phase);
    kittens::wait(ss.page_ready[base_stage + kCosSinPageOff], tok_phase);
    warp::sync();

    for (int pair_idx = blockIdx.x, pair_iter = 0;
         pair_idx < TOTAL_PAIRS;
         pair_idx += gridDim.x, ++pair_iter) {
        const int head_linear  = pair_idx / PAIRS_PER_HEAD;
        const int within_head  = pair_idx % PAIRS_PER_HEAD;
        const int block_a      = head_linear * BLOCKS_PER_HEAD + within_head;
        const int dim_a_in_head = (block_a * 16) % HEAD_DIM;
        const bool is_q        = head_linear < NUM_Q_HEADS;
        const bool is_k        = (!is_q) && (head_linear < NUM_Q_HEADS + NUM_KV_HEADS);
        const bool needs_rope  = is_q || is_k;

        // Per-warp exclusive slots for a and b partial sums.
        out_sv& my_out_a = *reinterpret_cast<out_sv*>(
            scratch_base + warp_in_role * 16 * sizeof(float));
        out_sv& my_out_b = *reinterpret_cast<out_sv*>(
            scratch_base + (NCW + warp_in_role) * 16 * sizeof(float));

        // Zero this warp's slots.
        if (kittens::laneid() < 16) {
            my_out_a[kittens::laneid()] = 0.0f;
            my_out_b[kittens::laneid()] = 0.0f;
        }
        kittens::group<NCW>::sync(kConsumerBarInit);

        #pragma unroll
        for (int k_chunk = 0; k_chunk < NUM_K_CHUNKS_PER_WARP; ++k_chunk) {
            const int pos_i       = pair_iter * NUM_K_CHUNKS_PER_WARP + k_chunk;
            const int stage       = pos_i % STAGES;
            const int ready_phase = (pos_i / STAGES) & 1;

            const int pg_a = base_stage + QGL::weight_page_a(stage, warp_in_role);
            const int pg_b = base_stage + QGL::weight_page_b(stage, warp_in_role);
            kittens::wait(ss.page_ready[pg_a], ready_phase);
            kittens::wait(ss.page_ready[pg_b], ready_phase);

            chunk_act_sv& act_chunk_slice =
                act_chunks[warp_in_role * NUM_K_CHUNKS_PER_WARP + k_chunk];
            kittens::rv_fl<kChunkCols> act_rv;
            warp::load(act_rv, act_chunk_slice);

            chunk_weight_st& w_a = *reinterpret_cast<chunk_weight_st*>(
                ss.pages[pg_a] + QGL::warp_byte_offset(warp_in_role));
            chunk_weight_st& w_b = *reinterpret_cast<chunk_weight_st*>(
                ss.pages[pg_b] + QGL::warp_byte_offset(warp_in_role));

            ferrite::tk::matvec(my_out_a, w_a, act_rv);
            ferrite::tk::matvec(my_out_b, w_b, act_rv);

            // Group leader signals page_done.
            if (kittens::laneid() == 0 && (warp_in_role % QGL::GROUP_SIZE == 0)) {
                kittens::arrive(ss.page_done[pg_a]);
                kittens::arrive(ss.page_done[pg_b]);
            }
        }

        kittens::group<NCW>::sync(kConsumerBarPublish);

        // Warp 0: reduce, NEOX rope, write bf16 staging, signal storer via page_done.
        if (warp_in_role == 0) {
            kittens::rv_fl<16> rv_a, rv_b;
            ferrite::tk::matvec_reduce<NCW>(scratch_base, rv_a);
            ferrite::tk::matvec_reduce<NCW>(scratch_base + NCW * 16 * sizeof(float), rv_b);
            warp::sync();

            if (needs_rope) {
                const int lane = kittens::laneid();
                if (lane < 16) {
                    const int freq    = dim_a_in_head + lane;
                    const float cos_v = __bfloat162float(cos_sin_sv[freq]);
                    const float sin_v = __bfloat162float(cos_sin_sv[HALF_DIM + freq]);
                    const float a     = rv_a[0][0];
                    const float b     = rv_b[0][0];
                    rv_a[0][0] = a * cos_v - b * sin_v;
                    rv_b[0][0] = b * cos_v + a * sin_v;
                }
            }

            auto& out_bf_a = *reinterpret_cast<kittens::sv_bf<16>*>(scratch_base + QKV_BF16_OFF);
            auto& out_bf_b = *reinterpret_cast<kittens::sv_bf<16>*>(scratch_base + QKV_BF16_OFF + 16);
            if (kittens::laneid() < 16) {
                out_bf_a[kittens::laneid()] = __float2bfloat16_rn(rv_a[0][0]);
                out_bf_b[kittens::laneid()] = __float2bfloat16_rn(rv_b[0][0]);
            }
            warp::sync();
            if (kittens::laneid() == 0) {
                kittens::arrive(ss.page_done[base_stage + kActPageOff]);
            }
        }
    }
}

// ---------- Launcher ------------------------------------------------

template <
    typename Config,
    int HIDDEN_DIM, int HEAD_DIM,
    int NUM_Q_HEADS, int NUM_KV_HEADS,
    bool BIASED, bool INTERLEAVED
>
__device__ __forceinline__ void launcher(
    ferrite::SharedState<Config>& ss,
    int base_stage
) {
    (void)ss; (void)base_stage;
}

// ---------- Storer --------------------------------------------------

template <
    typename Config,
    int HIDDEN_DIM, int HEAD_DIM,
    int NUM_Q_HEADS, int NUM_KV_HEADS, int BLOCK_SIZE,
    bool BIASED, bool INTERLEAVED
>
__device__ __forceinline__ void storer(
    ferrite::bf16_ptr      q_out,         // [NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM]
    ferrite::bf16_ptr      key_cache,     // [num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]
    ferrite::bf16_ptr      value_cache,
    const int64_t*         slot_mapping,  // [NUM_TOKENS]
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int tok                               // current token index within the batch
) {
    static_assert(!BIASED);
    static_assert(!INTERLEAVED);
    static_assert(HEAD_DIM % 32 == 0);

    // STAGES=1 for fused_qkv — see loader for the shmem-cap rationale.
    constexpr int STAGES           = 1;
    constexpr int BLOCKS_PER_HEAD  = HEAD_DIM / 16;
    constexpr int PAIRS_PER_HEAD   = HEAD_DIM / 32;
    constexpr int NUM_HEADS_TOTAL  = NUM_Q_HEADS + 2 * NUM_KV_HEADS;
    constexpr int TOTAL_PAIRS      = NUM_HEADS_TOTAL * PAIRS_PER_HEAD;

    using warp = kittens::group<1>;
    constexpr uint32_t out_block_bytes = 16 * sizeof(__nv_bfloat16);

    constexpr int NCW_S = Config::NUM_CONSUMER_WARPS;
    constexpr size_t QKV_BF16_OFF_S = 2u * NCW_S * 16 * sizeof(float);
    uint8_t* scratch_base = ss.scratch;

    for (int pair_idx = blockIdx.x, pair_iter = 0;
         pair_idx < TOTAL_PAIRS;
         pair_idx += gridDim.x, ++pair_iter) {
        // Wait for consumer warp 0 to reduce, rope-rotate, and fill staging.
        kittens::wait(ss.page_done[base_stage + kActPageOff], pair_iter & 1);

        auto& out_bf_a = *reinterpret_cast<kittens::sv_bf<16>*>(scratch_base + QKV_BF16_OFF_S);
        auto& out_bf_b = *reinterpret_cast<kittens::sv_bf<16>*>(scratch_base + QKV_BF16_OFF_S + 16);

        const int head_linear   = pair_idx / PAIRS_PER_HEAD;
        const int within_head   = pair_idx % PAIRS_PER_HEAD;
        const int dim_a_in_head = within_head * 16;
        const int dim_b_in_head = dim_a_in_head + HEAD_DIM / 2;
        const bool is_q         = head_linear < NUM_Q_HEADS;
        const bool is_k         = (!is_q) && (head_linear < NUM_Q_HEADS + NUM_KV_HEADS);

        if (is_q) {
            const int q_head = head_linear;
            // q_out layout: [NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM]
            __nv_bfloat16* q_tok = q_out
                + static_cast<size_t>(tok) * NUM_Q_HEADS * HEAD_DIM;
            __nv_bfloat16* dst_a = q_tok
                + static_cast<size_t>(q_head) * HEAD_DIM
                + static_cast<size_t>(dim_a_in_head);
            __nv_bfloat16* dst_b = q_tok
                + static_cast<size_t>(q_head) * HEAD_DIM
                + static_cast<size_t>(dim_b_in_head);
            warp::tma::store_async(static_cast<void*>(dst_a), &out_bf_a, out_block_bytes);
            warp::tma::store_async(static_cast<void*>(dst_b), &out_bf_b, out_block_bytes);
            kittens::tma::store_async_wait<0>();
        } else {
            const int64_t slot = slot_mapping[tok];
            if (slot >= 0) {
                const int kv_head = is_k
                    ? (head_linear - NUM_Q_HEADS)
                    : (head_linear - NUM_Q_HEADS - NUM_KV_HEADS);
                const int64_t block_idx  = slot / BLOCK_SIZE;
                const int64_t block_off  = slot % BLOCK_SIZE;
                const size_t block_stride = static_cast<size_t>(BLOCK_SIZE)
                                              * NUM_KV_HEADS * HEAD_DIM;
                const size_t slot_stride  = static_cast<size_t>(NUM_KV_HEADS) * HEAD_DIM;
                const size_t head_stride  = HEAD_DIM;
                __nv_bfloat16* base = is_k ? key_cache : value_cache;
                __nv_bfloat16* head_base = base
                    + static_cast<size_t>(block_idx) * block_stride
                    + static_cast<size_t>(block_off) * slot_stride
                    + static_cast<size_t>(kv_head)   * head_stride;
                __nv_bfloat16* dst_a = head_base + static_cast<size_t>(dim_a_in_head);
                __nv_bfloat16* dst_b = head_base + static_cast<size_t>(dim_b_in_head);
                warp::tma::store_async(static_cast<void*>(dst_a), &out_bf_a, out_block_bytes);
                warp::tma::store_async(static_cast<void*>(dst_b), &out_bf_b, out_block_bytes);
                kittens::tma::store_async_wait<0>();
            }
        }
    }
}

}  // namespace fused_qkv_rope_cache
}  // namespace ops
}  // namespace ferrite
