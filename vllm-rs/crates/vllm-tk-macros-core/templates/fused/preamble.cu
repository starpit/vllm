// GENERATED: Fused prefill layer kernel ({{ mode_label }}, {{ cta_rows }}-row CTA)
// Grid: ceil(num_prefill_tokens/{{ cta_rows }}) CTAs

#define SM89_NUM_LAYERS             {{ nl }}
#define SM89_HIDDEN_DIM             {{ hd }}
#define SM89_INTERMEDIATE_DIM       {{ id }}
#define SM89_HEAD_DIM               {{ hdm }}
#define SM89_NUM_ATTENTION_HEADS    {{ nah }}
#define SM89_NUM_KV_HEADS           {{ nkh }}

#include "llama_sm89.cuh"

using namespace kittens;
using namespace kittens::prototype::vm;
using globals = llama_sm89_globals;

constexpr int PFL_NUM_WARPS = {{ num_warps }};
constexpr int PFL_GQA_RATIO = {{ gqa_ratio }};
constexpr int PFL_KV_PAGE_SIZE = {{ kv_page_size }};
constexpr int PFL_ITERS_PER_PAGE = {{ iters_per_page }};
constexpr int PFL_HEAD_DIM = {{ hdm }};
constexpr int PFL_SHMEM = {{ total_shmem }};
constexpr int PFL_KV_TILE_BYTES = {{ kv_tile_bytes }};
constexpr int PFL_Q_ROWS = 16;
constexpr int PFL_CTA_ROWS = {{ cta_rows }};
constexpr int PFL_K_DIM = {{ k_dim }};
constexpr int PFL_OUT_BLOCK = {{ out_block }};
constexpr int PFL_RDPW = {{ rdpw }};
constexpr int PFL_N_TILES = PFL_OUT_BLOCK / 16;

using pfl_a_st = st_bf<PFL_Q_ROWS, PFL_K_DIM>;
using pfl_b_st = st_bf<PFL_OUT_BLOCK, PFL_K_DIM>;
using pfl_acc_rt = rt_fl<16, PFL_OUT_BLOCK>;
using pfl_b_slice_st = st_bf<16, PFL_K_DIM>;

using pfl_q_st  = st_bf<PFL_Q_ROWS, PFL_HEAD_DIM>;
using pfl_kv_st = st_bf<PFL_KV_PAGE_SIZE, PFL_HEAD_DIM>;
using pfl_q_rt  = rt_bf<PFL_Q_ROWS, PFL_HEAD_DIM>;
using pfl_k_rt  = rt_bf<PFL_KV_PAGE_SIZE, PFL_HEAD_DIM>;
using pfl_v_rt  = rt_bf<PFL_KV_PAGE_SIZE, PFL_HEAD_DIM, col_l>;
using pfl_score_fl = rt_fl<PFL_Q_ROWS, PFL_KV_PAGE_SIZE>;
using pfl_score_bf = rt_bf<PFL_Q_ROWS, PFL_KV_PAGE_SIZE>;
using pfl_o_rt  = rt_fl<PFL_Q_ROWS, PFL_HEAD_DIM>;
using pfl_o_bf  = rt_bf<PFL_Q_ROWS, PFL_HEAD_DIM>;
using pfl_max_rv = col_vec<rt_fl<PFL_Q_ROWS, PFL_HEAD_DIM>>;
using pfl_norm_rv = col_vec<rt_fl<PFL_Q_ROWS, PFL_HEAD_DIM>>;
using pfl_o_sv  = sv_bf<PFL_HEAD_DIM>;

__device__ static inline void pfl_cp_async_wait_all() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_all;\n"     ::: "memory");
}

__device__ static inline void pfl_load_b_slice(
    rt_bf<16, PFL_K_DIM> &dst, const st_bf<16, PFL_K_DIM> &src) {
    uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    int lane = kittens::laneid();
    int row = lane % 16;
    bf16_2 tmp[4];
    #pragma unroll
    for (int j = 0; j < PFL_K_DIM / 16; j++) {
        int col = j * 16 + (lane / 16) * 8;
        move<bf16_2>::ldsm4(tmp[0], tmp[1], tmp[2], tmp[3], src.idx(saddr, {row, col}));
        dst.tiles[0][j].data[0] = tmp[0];
        dst.tiles[0][j].data[1] = tmp[1];
        dst.tiles[0][j].data[2] = tmp[2];
        dst.tiles[0][j].data[3] = tmp[3];
    }
}
