// SPDX-License-Identifier: Apache-2.0
// Ferrite-owned TK 2.0 gelu_upgate op — 4 warp-role functions.
//
// Math (the fused MLP "gate / up / gelu / mul" block):
//   gate[row] = sum_k W_gate[row, k] * x[k]
//   up  [row] = sum_k W_up  [row, k] * x[k]
//   out [row] = gelu(gate[row]) * up[row]
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
//
// Gemma2/3 uses `gelu_pytorch_tanh` activation (the tanh approximation,
// also called `gelu_new`). This op is the Gemma2 peer of `silu_upgate.cuh`.
//
// Structure is identical to `silu_upgate.cuh`: same page layout, same
// TMA patterns, same warp roles. Only the consumer's activation computation
// differs (GELU via tanhf instead of SiLU via exp).
//
// Page layout (STAGES=1 cross-block — same as silu_upgate):
//   pages[base_stage + 0]
//       — sv_bf<HIDDEN_DIM> activation, loaded once per CTA.
//   pages[base_stage + 1 + w]              (w ∈ [0, NCW))
//       — st_bf<16, CHUNK_COLS, false> gate weight tile for warp w.
//   pages[base_stage + 1 + NCW + w]
//       — st_bf<16, CHUNK_COLS, false> up weight tile for warp w.
// Total pages: 1 + 2 * NCW.
//
// Scratch layout (same as silu_upgate):
//   float[0 .. 16)    — gate_accum (sv_fl<16>).
//   float[16 .. 32)   — up_accum   (sv_fl<16>).
//   float[32 .. 40)   — bf16 output staging (sv_bf<16> = 32 bytes).
//
// bar IDs: 13 (zero-accumulators), 14 (publish).

#pragma once

#include "kittens.cuh"
#include "ferrite_globals.cuh"
#include "ferrite_substrate.cuh"
#include "ferrite_tk_helpers.cuh"

namespace ferrite {
namespace ops {
namespace gelu_upgate {

constexpr int kActPageOff = 0;

// Same GROUP-PAGE layout as silu_upgate.
template <int NCW, int K_PER_WARP, int PAGE_SIZE_BYTES>
struct UpgateGroupLayout {
    using GL = ::ferrite::TileGroupLayout<NCW, K_PER_WARP, PAGE_SIZE_BYTES>;
    static constexpr int kChunkCols = GL::kChunkCols;
    static constexpr int GROUP_SIZE = GL::GROUP_SIZE;
    static constexpr int NUM_GROUPS = GL::NUM_GROUPS;
    __host__ __device__ static constexpr int gate_page(int warp_id) {
        return 1 + warp_id / GROUP_SIZE;
    }
    __host__ __device__ static constexpr int up_page(int warp_id) {
        return 1 + NUM_GROUPS + warp_id / GROUP_SIZE;
    }
    __host__ __device__ static constexpr int warp_byte_offset(int warp_id) {
        return GL::warp_byte_offset(warp_id);
    }
};

constexpr int kConsumerBarInit    = 13;
constexpr int kConsumerBarPublish = 14;

// ---------- Loader --------------------------------------------------
// Identical to silu_upgate::loader.

template <typename Config, int HIDDEN_DIM, int INTERMEDIATE_DIM, int NUM_TOKENS>
__device__ __forceinline__ void loader(
    ferrite::bf16_cptr x,              // [NUM_TOKENS, HIDDEN_DIM]
    ferrite::bf16_cptr W_gate_up,      // [2 * INTERMEDIATE_DIM, HIDDEN_DIM]
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int tok
) {
    constexpr int NCW                    = Config::NUM_CONSUMER_WARPS;
    constexpr int K_PER_WARP             = HIDDEN_DIM / NCW;
    using UGL = UpgateGroupLayout<NCW, K_PER_WARP, Config::PAGE_SIZE>;
    constexpr int kChunkCols             = UGL::kChunkCols;
    constexpr int NUM_K_CHUNKS_PER_WARP  = K_PER_WARP / kChunkCols;
    constexpr int GROUP_SIZE             = UGL::GROUP_SIZE;
    constexpr int NUM_GROUPS             = UGL::NUM_GROUPS;
    static_assert(INTERMEDIATE_DIM % 16 == 0);
    static_assert(HIDDEN_DIM % NCW == 0);

    using warp = kittens::group<1>;
    constexpr uint32_t act_bytes        = HIDDEN_DIM * sizeof(__nv_bfloat16);
    constexpr uint32_t group_tile_bytes = GROUP_SIZE * 16 * kChunkCols * sizeof(__nv_bfloat16);
    constexpr uint32_t warp_row_bytes   = kChunkCols * sizeof(__nv_bfloat16);

    const int num_blocks = INTERMEDIATE_DIM / 16;

    void* act_page = reinterpret_cast<void*>(ss.pages[base_stage + kActPageOff]);
    if (kittens::laneid() == 0) {
        kittens::tma::expect_bytes(ss.page_ready[base_stage + kActPageOff], act_bytes);
    }
    warp::tma::load_async(act_page,
        const_cast<__nv_bfloat16*>(x + static_cast<size_t>(tok) * HIDDEN_DIM),
        act_bytes, ss.page_ready[base_stage + kActPageOff]);

    int iter = 0;
    for (int block = blockIdx.x; block < num_blocks; block += gridDim.x) {
        const size_t block_row_base = static_cast<size_t>(block) * 16;
        #pragma unroll
        for (int k_chunk = 0; k_chunk < NUM_K_CHUNKS_PER_WARP; ++k_chunk, ++iter) {
            const int prev_phase = (iter - 1) & 1;
            if (iter >= 1) {
                #pragma unroll
                for (int g = 0; g < NUM_GROUPS; ++g) {
                    kittens::wait(ss.page_done[base_stage + UGL::gate_page(g*GROUP_SIZE)], prev_phase);
                    kittens::wait(ss.page_done[base_stage + UGL::up_page(g*GROUP_SIZE)],   prev_phase);
                }
            }
            if (kittens::laneid() == 0) {
                #pragma unroll
                for (int g = 0; g < NUM_GROUPS; ++g) {
                    kittens::tma::expect_bytes(
                        ss.page_ready[base_stage + UGL::gate_page(g*GROUP_SIZE)], group_tile_bytes);
                    kittens::tma::expect_bytes(
                        ss.page_ready[base_stage + UGL::up_page(g*GROUP_SIZE)], group_tile_bytes);
                }
            }
            #pragma unroll
            for (int w = 0; w < NCW; ++w) {
                int gate_pg = base_stage + UGL::gate_page(w);
                int up_pg   = base_stage + UGL::up_page(w);
                __nv_bfloat16* gate_dst = reinterpret_cast<__nv_bfloat16*>(ss.pages[gate_pg])
                    + UGL::warp_byte_offset(w) / sizeof(__nv_bfloat16);
                __nv_bfloat16* up_dst = reinterpret_cast<__nv_bfloat16*>(ss.pages[up_pg])
                    + UGL::warp_byte_offset(w) / sizeof(__nv_bfloat16);
                const size_t col_base = static_cast<size_t>(w) * K_PER_WARP
                    + static_cast<size_t>(k_chunk) * kChunkCols;
                #pragma unroll
                for (int r = 0; r < 16; ++r) {
                    warp::tma::load_async(gate_dst + static_cast<size_t>(r)*kChunkCols,
                        const_cast<__nv_bfloat16*>(W_gate_up+(block_row_base+r)*HIDDEN_DIM+col_base),
                        warp_row_bytes, ss.page_ready[gate_pg]);
                    warp::tma::load_async(up_dst + static_cast<size_t>(r)*kChunkCols,
                        const_cast<__nv_bfloat16*>(W_gate_up+(static_cast<size_t>(INTERMEDIATE_DIM)
                            +block_row_base+r)*HIDDEN_DIM+col_base),
                        warp_row_bytes, ss.page_ready[up_pg]);
                }
            }
        }
    }
}

// ---------- Consumer ------------------------------------------------
// Identical to silu_upgate::consumer except the activation function:
// GELU (tanh approx) replaces SiLU.

template <typename Config, int HIDDEN_DIM, int INTERMEDIATE_DIM, int NUM_TOKENS>
__device__ __forceinline__ void consumer(
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int warp_in_role,
    int tok
) {
    constexpr int NCW                    = Config::NUM_CONSUMER_WARPS;
    constexpr int K_PER_WARP             = HIDDEN_DIM / NCW;
    using UGL = UpgateGroupLayout<NCW, K_PER_WARP, Config::PAGE_SIZE>;
    constexpr int kChunkCols             = UGL::kChunkCols;
    constexpr int NUM_K_CHUNKS_PER_WARP  = K_PER_WARP / kChunkCols;
    static_assert(INTERMEDIATE_DIM % 16 == 0);
    static_assert(HIDDEN_DIM % NCW == 0);

    using warp             = kittens::group<1>;
    using chunk_weight_st  = kittens::st_bf<16, kChunkCols, /*_swizzle=*/false>;
    using chunk_act_sv     = kittens::sv_bf<kChunkCols>;
    using out_sv           = kittens::sv_fl<16>;

    auto* act_chunks = reinterpret_cast<chunk_act_sv*>(
        ss.pages[base_stage + kActPageOff]);

    constexpr size_t SU_BF16_OFF = 2u * NCW * 16 * sizeof(float);
    uint8_t* scratch_base = ss.scratch;
    out_sv& my_gate_accum = *reinterpret_cast<out_sv*>(scratch_base + warp_in_role * 16 * sizeof(float));
    out_sv& my_up_accum = *reinterpret_cast<out_sv*>(scratch_base + (NCW + warp_in_role) * 16 * sizeof(float));

    kittens::wait(ss.page_ready[base_stage + kActPageOff], tok & 1);
    warp::sync();

    const int num_blocks = INTERMEDIATE_DIM / 16;
    int iter = 0;
    int block_counter = 0;
    for (int block = blockIdx.x; block < num_blocks; block += gridDim.x, ++block_counter) {
        if (kittens::laneid() < 16) {
            my_gate_accum[kittens::laneid()] = 0.0f;
            my_up_accum  [kittens::laneid()] = 0.0f;
        }
        kittens::group<NCW>::sync(kConsumerBarInit);

        #pragma unroll
        for (int k_chunk = 0; k_chunk < NUM_K_CHUNKS_PER_WARP; ++k_chunk, ++iter) {
            const int phase = iter & 1;

            const int gate_pg = base_stage + UGL::gate_page(warp_in_role);
            const int up_pg   = base_stage + UGL::up_page(warp_in_role);
            kittens::wait(ss.page_ready[gate_pg], phase);
            kittens::wait(ss.page_ready[up_pg],   phase);

            chunk_act_sv& act_chunk_slice =
                act_chunks[warp_in_role * NUM_K_CHUNKS_PER_WARP + k_chunk];
            kittens::rv_fl<kChunkCols> act_rv;
            warp::load(act_rv, act_chunk_slice);

            chunk_weight_st& gate_tile = *reinterpret_cast<chunk_weight_st*>(
                ss.pages[gate_pg] + UGL::warp_byte_offset(warp_in_role));
            chunk_weight_st& up_tile = *reinterpret_cast<chunk_weight_st*>(
                ss.pages[up_pg]   + UGL::warp_byte_offset(warp_in_role));

            ferrite::tk::matvec(my_gate_accum, gate_tile, act_rv);
            ferrite::tk::matvec(my_up_accum,   up_tile,   act_rv);

            if (kittens::laneid() == 0 && (warp_in_role % UGL::GROUP_SIZE == 0)) {
                kittens::arrive(ss.page_done[gate_pg]);
                kittens::arrive(ss.page_done[up_pg]);
            }
        }

        kittens::group<NCW>::sync(kConsumerBarPublish);

        // Warp 0: reduce gate+up, compute gelu(gate)*up, write staging, signal storer.
        // GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x^3)))
        if (warp_in_role == 0) {
            kittens::rv_fl<16> gate_rv, up_rv;
            ferrite::tk::matvec_reduce<NCW>(scratch_base, gate_rv);
            ferrite::tk::matvec_reduce<NCW>(scratch_base + NCW * 16 * sizeof(float), up_rv);

            static constexpr float kGeluKappa = 0.7978845608f;
            static constexpr float kGeluCoeff = 0.044715f;
            #pragma unroll
            for (int outer = 0; outer < gate_rv.outer_dim; ++outer) {
                #pragma unroll
                for (int inner = 0; inner < gate_rv.inner_dim; ++inner) {
                    float x = gate_rv.data[outer][inner];
                    float xc = kGeluKappa * fmaf(kGeluCoeff * x * x, x, x);
                    gate_rv.data[outer][inner] = 0.5f * x * (1.0f + tanhf(xc));
                }
            }
            kittens::warp::mul(gate_rv, gate_rv, up_rv);

            auto& out_bf = *reinterpret_cast<kittens::sv_bf<16>*>(scratch_base + SU_BF16_OFF);
            if (kittens::laneid() < 16) {
                out_bf[kittens::laneid()] = __float2bfloat16_rn(gate_rv[0][0]);
            }
            warp::sync();
            if (kittens::laneid() == 0) {
                kittens::arrive(ss.page_done[base_stage + kActPageOff]);
            }
        }
    }
}

// ---------- Launcher ------------------------------------------------

template <typename Config, int HIDDEN_DIM, int INTERMEDIATE_DIM, int NUM_TOKENS>
__device__ __forceinline__ void launcher(
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int tok
) {
    (void)ss; (void)base_stage; (void)tok;
}

// ---------- Storer --------------------------------------------------
// Identical to silu_upgate::storer.

template <typename Config, int HIDDEN_DIM, int INTERMEDIATE_DIM, int NUM_TOKENS>
__device__ __forceinline__ void storer(
    ferrite::bf16_ptr out,             // [NUM_TOKENS, INTERMEDIATE_DIM]
    ferrite::SharedState<Config>& ss,
    int base_stage,
    int tok
) {
    static_assert(INTERMEDIATE_DIM % 16 == 0);

    using warp = kittens::group<1>;
    constexpr uint32_t out_block_bytes = 16 * sizeof(__nv_bfloat16);

    constexpr int NCW_S = Config::NUM_CONSUMER_WARPS;
    constexpr size_t GS_BF16_OFF = 2u * NCW_S * 16 * sizeof(float);
    uint8_t* scratch_base = ss.scratch;

    const int num_blocks = INTERMEDIATE_DIM / 16;
    int block_iter = 0;
    for (int block = blockIdx.x; block < num_blocks; block += gridDim.x, ++block_iter) {
        // Wait for consumer warp 0 to reduce, compute gelu, and fill staging area.
        kittens::wait(ss.page_done[base_stage + kActPageOff], block_iter & 1);

        auto& out_bf = *reinterpret_cast<kittens::sv_bf<16>*>(scratch_base + GS_BF16_OFF);
        warp::tma::store_async(
            static_cast<void*>(
                out + static_cast<size_t>(tok) * INTERMEDIATE_DIM
                    + static_cast<size_t>(block) * 16),
            &out_bf,
            out_block_bytes);
        kittens::tma::store_async_wait<0>();
    }
}

}  // namespace gelu_upgate
}  // namespace ops
}  // namespace ferrite
