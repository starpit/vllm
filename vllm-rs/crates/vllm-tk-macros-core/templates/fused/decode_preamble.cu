{# Preamble for fused decode kernel.
   Variables: model dims + config constants from FusedDecodeDerived.
#}
// GENERATED: Fused decode layer kernel (row-fused, {{ cta_rows }}-row CTA)
// Grid: ceil(batch_size / {{ cta_rows }}) CTAs — each CTA fully independent
// No cross-CTA barriers — activations stay in shmem across all phases and layers.

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

// ── Decode kernel constants ───────────────────────────────────────────
constexpr int DEC_CTA_ROWS      = {{ cta_rows }};
constexpr int DEC_PADDED_ROWS   = {{ padded_cta_rows }};
constexpr int DEC_NUM_WARPS     = {{ num_warps }};
constexpr int DEC_GQA_RATIO     = {{ gqa_ratio }};
constexpr int DEC_KV_PAGE_SIZE  = {{ kv_page_size }};
constexpr int DEC_ITERS_PER_PAGE = {{ iters_per_page }};
constexpr int DEC_HEAD_DIM      = {{ hdm }};
constexpr int DEC_K_DIM         = {{ k_dim }};
constexpr int DEC_OUT_BLOCK     = {{ out_block }};
constexpr int DEC_PEAK_SHMEM    = {{ peak_shmem }};
constexpr int DEC_KV_TILE_BYTES = {{ kv_tile_bytes }};
constexpr int DEC_HIDDEN_SHMEM  = {{ hidden_shmem }};
constexpr int DEC_META_SHMEM    = {{ meta_shmem }};
constexpr int DEC_NUM_STAGES    = {{ num_stages }};

// Tile types
using dec_a_st   = st_bf<16, DEC_K_DIM>;       // padded to 32 rows for cp.async
using dec_b_st   = st_bf<DEC_OUT_BLOCK, DEC_K_DIM>;
using dec_acc_rt = rt_fl<16, DEC_OUT_BLOCK>;

// Attention types
using dec_q_st   = st_bf<16, DEC_HEAD_DIM>;
using dec_kv_st  = st_bf<DEC_KV_PAGE_SIZE, DEC_HEAD_DIM>;
using dec_q_rt   = rt_bf<16, DEC_HEAD_DIM>;
using dec_k_rt   = rt_bf<DEC_KV_PAGE_SIZE, DEC_HEAD_DIM>;
using dec_v_rt   = rt_bf<DEC_KV_PAGE_SIZE, DEC_HEAD_DIM, col_l>;
using dec_score_fl = rt_fl<16, DEC_KV_PAGE_SIZE>;
using dec_score_bf = rt_bf<16, DEC_KV_PAGE_SIZE>;
using dec_o_rt   = rt_fl<16, DEC_HEAD_DIM>;
using dec_o_bf   = rt_bf<16, DEC_HEAD_DIM>;
using dec_max_rv  = col_vec<rt_fl<16, DEC_HEAD_DIM>>;
using dec_norm_rv = col_vec<rt_fl<16, DEC_HEAD_DIM>>;
using dec_o_sv   = sv_bf<DEC_HEAD_DIM>;

__device__ static inline void dec_cp_async_wait_all() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_all;\n"     ::: "memory");
}

// Per-row metadata: loaded once, used by all layers
struct DecRowMeta {
    int position_id;
    int kv_indptr_start;
    int kv_indptr_end;
    int kv_last_page_len;
    int kv_append_slot;
};
