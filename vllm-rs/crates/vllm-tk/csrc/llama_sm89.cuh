#pragma once

// sm89 (L40S/RTX 4090) port of the KVM Llama megakernel.
// Key differences from llama_official (Blackwell/Hopper):
//   - No TMA: all global->shared loads use cp.async (group::load_async)
//   - No WGMMA / tensor allocator: GEMM uses warp-scope mma.sync via warp::mma
//   - Smaller shmem budget (~91KB dynamic vs 228KB on H100)
//   - PAGE_SIZE = 8192 (8KB), NUM_PAGES = 10
//   - PIPELINE_K_DIM = 32 (vs 64), matmul_out_block_size = 64 (vs 256)
//   - No cluster launches (sm89 has no cluster support)

#include "kittens.cuh"
#include "vm/vm.cuh"
#include <iostream>

#define OPCODE_AttnNorm              1
#define OPCODE_QKV_RopeAppend        2
#define OPCODE_GQA_AttentionPrefill  3
#define OPCODE_GQA_AttentionDecode   4
#define OPCODE_O_ProjResidual        5
#define OPCODE_MlpNorm               6
#define OPCODE_GateSiLU              7
#define OPCODE_UpMatmul              8
#define OPCODE_DownProjResidual      9
#define OPCODE_LM_HeadNorm          10
#define OPCODE_LM_Head              11

// Model size selection — default to 1B for easier sm89 testing.
// Override with -DUSE_LLAMA_SM89_8B.
#if defined(USE_LLAMA_SM89_8B)
#  define SM89_NUM_LAYERS              32
#  define SM89_HIDDEN_DIM            4096
#  define SM89_INTERMEDIATE_DIM     14336
#  define SM89_HEAD_DIM               128
#  define SM89_NUM_ATTENTION_HEADS     32
#  define SM89_NUM_KV_HEADS             8
#else
// LLaMA 1B defaults
#  define SM89_NUM_LAYERS              16
#  define SM89_HIDDEN_DIM            2048
#  define SM89_INTERMEDIATE_DIM      8192
#  define SM89_HEAD_DIM                64
#  define SM89_NUM_ATTENTION_HEADS     32
#  define SM89_NUM_KV_HEADS             8
#endif

#define SM89_KV_BLOCK_SIZE          16
#define SM89_KV_PAGE_SIZE           64    // tokens per KV page (must be multiple of KV_BLOCK_SIZE)
#define SM89_MATMUL_BATCH_BLOCK_SIZE 128
#define SM89_SM_COUNT               142   // L40S SM count

namespace kittens::prototype::vm {

// Compute the optimal GEMM output tile width for a given output dimension and SM count.
// Picks the largest power-of-2 in [min_block, 128] such that output_dim / tile >= target,
// where target = sm_count / 2 (aiming for ~50% SM utilization per batch block).
// Using sm_count/2 instead of sm_count balances SM fill against per-instruction VM overhead.
// Falls back to the smallest valid tile if no size reaches the target.
// min_block: minimum tile width (e.g. head_dim for QKV where HEADS_PER_BLOCK >= 1).
constexpr int optimal_out_block(int output_dim, int sm_count, int min_block = 32) {
    int target = sm_count / 2;  // 50% fill balances utilization vs VM overhead
    for (int ob = 128; ob >= min_block; ob /= 2) {
        if (output_dim % ob == 0 && output_dim / ob >= target)
            return ob;
    }
    for (int ob = min_block; ob <= 128; ob *= 2) {
        if (output_dim % ob == 0)
            return ob;
    }
    return min_block;
}

// ─────────────────────────────────────────────────────────────────────────────
// sm89 KVM config
// ─────────────────────────────────────────────────────────────────────────────
// Shmem budget (L40S, sm89):
//   cudaFuncAttributeMaxDynamicSharedMemorySize → 99KB opt-in
//   SCRATCH_BYTES = 8192  (attention Q/O overlay needs up to ~7KB for head_dim=128)
//   STATIC_SHARED_MEMORY = 512 + 2*(8192 + (32+128)*4 + 32*8) = 18688 bytes
//   DYNAMIC_SHARED_MEMORY = 99*1024 - 18688 = 82688 bytes
//   PAGE_SIZE = 8192  → NUM_PAGES = 82688 / 8192 = 10
//
// GEMM shmem usage per stage (K_DIM=64, OUT_BLOCK=128, BATCH_BLOCK=128):
//   a_smem = st_bf<128, 64> = 16 KB = 2 pages
//   b_smem = st_bf<128, 64> = 16 KB = 2 pages
//   per stage = 4 pages; 2 stages = 8 pages
//   + 1 rope page for QKV = 9 pages total for matmul ops
//   attention: 4 pages/stage × 2 stages = 8 pages (head_dim=128)
struct llama_sm89_config {
    static constexpr int INSTRUCTION_PIPELINE_STAGES       = 2;
    static constexpr int INSTRUCTION_PIPELINE_STAGES_BITS  = 1;
    static constexpr int INSTRUCTION_WIDTH                  = 32;
    using instruction_t = int[INSTRUCTION_WIDTH];
    static constexpr int TIMING_WIDTH                       = 128;
    using timing_t = int[TIMING_WIDTH];
    static constexpr int DYNAMIC_SEMAPHORES                 = 32;

    static constexpr int NUM_CONSUMER_WARPS  = 8;
    static constexpr int NUM_WARPS           = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS         = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int NUM_BLOCKS          = 1;
    static constexpr int CLUSTER_BLOCKS      = 1;  // no cluster on sm89

    static constexpr int SCRATCH_BYTES = 8192;
    static constexpr int MAX_SHARED_MEMORY = 99 * 1024;  // opt-in to 99 KB
    static constexpr int STATIC_SHARED_MEMORY =
        512 + INSTRUCTION_PIPELINE_STAGES * (SCRATCH_BYTES + (INSTRUCTION_WIDTH + TIMING_WIDTH) * 4 + DYNAMIC_SEMAPHORES * 8);
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int PAGE_SIZE = 8192;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
    static_assert(NUM_PAGES >= 10, "Need at least 10 pages for sm89 KVM");

    static constexpr bool TIMING_RECORD_ENABLED        = false;
    static constexpr bool GMEM_SPIN_LOOP_SLEEP_NANOS   = 20;

    // Register budgets: no setmaxnreg on sm89, so these are advisory only
    // (the guards in vm.cuh make them no-ops).
    static constexpr int CONSUMER_REGISTERS     = 128;
    static constexpr int NON_CONSUMER_REGISTERS = 64;
};

// ─────────────────────────────────────────────────────────────────────────────
// Globals struct
// ─────────────────────────────────────────────────────────────────────────────
template <int _num_hidden_layers, int _hidden_dim, int _intermediate_dim,
          int _head_dim, int _num_attention_heads, int _num_kv_heads,
          int _kv_block_size, int _kv_page_size,
          int _matmul_batch_block_size, int _sm_count>
struct sm89_globals_t {
    constexpr static int num_hidden_layers       = _num_hidden_layers;
    constexpr static int hidden_dim              = _hidden_dim;
    constexpr static int intermediate_dim        = _intermediate_dim;
    constexpr static int head_dim                = _head_dim;
    constexpr static int num_attention_heads     = _num_attention_heads;
    constexpr static int num_kv_heads            = _num_kv_heads;
    constexpr static int kv_block_size           = _kv_block_size;
    constexpr static int kv_page_size            = _kv_page_size;
    constexpr static int matmul_batch_block_size = _matmul_batch_block_size;
    constexpr static int sm_count                = _sm_count;
    constexpr static int iters_per_page          = kv_page_size / kv_block_size;
    static_assert(kv_page_size % kv_block_size == 0, "kv_page_size must be multiple of kv_block_size");

    // Per-op output tile sizes, auto-computed from model dimensions and SM count.
    // Each op gets the largest power-of-2 tile that still fills all SMs.
    constexpr static int nbh             = num_attention_heads + 2 * num_kv_heads;
    constexpr static int qkv_out_block   = optimal_out_block(nbh * head_dim, sm_count, head_dim);
    constexpr static int o_proj_out_block    = optimal_out_block(hidden_dim, sm_count);
    constexpr static int gate_out_block  = optimal_out_block(intermediate_dim, sm_count);
    constexpr static int up_out_block    = optimal_out_block(intermediate_dim, sm_count);
    constexpr static int down_out_block  = optimal_out_block(hidden_dim, sm_count);
    constexpr static int lm_head_out_block = 128;  // vocab_size is runtime, use max tile

    // Divisibility constraints for GEMM tiling and attention
    static_assert(head_dim % 16 == 0, "head_dim must be a multiple of 16 (warp tile size)");
    static_assert(num_attention_heads % num_kv_heads == 0, "GQA ratio must be integer");

    using config = llama_sm89_config;

    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout      = ::kittens::prototype::vm::timing_layout<config>;

    // Weight layouts: [1, num_layers, output_dim / out_block, input_dim] packed tiles.
    // Multiple tile types (all power-of-2 sizes 16..128) so a single GL can be used
    // by ops with different OUT_BLOCK values.
    using weights_t         = gl<bf16, 1, -1, -1, hidden_dim,
                                 st_bf<16, 64>, st_bf<32, 64>, st_bf<64, 64>, st_bf<128, 64>>;
    using weights_big_t     = gl<bf16, 1, -1, -1, intermediate_dim,
                                 st_bf<16, 64>, st_bf<32, 64>, st_bf<64, 64>, st_bf<128, 64>>;

    // Activation layouts: [1, 1, batch, hidden_dim]
    // sv_bf for per-head / per-hidden vector loads; st_bf for GEMM and store tiles.
    using activations_t     = gl<bf16, 1, 1, -1, hidden_dim,
                                 sv_bf<head_dim>, sv_bf<hidden_dim>,
                                 st_bf<matmul_batch_block_size, 64>,  // GEMM A tiles
                                 st_bf<16, 16>, st_bf<16, 32>, st_bf<16, 64>, st_bf<16, 128>>;
    using activations_big_t = gl<bf16, 1, 1, -1, intermediate_dim,
                                 st_bf<matmul_batch_block_size, 64>,
                                 st_bf<16, 16>, st_bf<16, 32>, st_bf<16, 64>, st_bf<16, 128>>;
    using logits_t          = gl<bf16, 1, 1, -1, -1,
                                 st_bf<16, 16>, st_bf<16, 32>, st_bf<16, 64>, st_bf<16, 128>>;

    using norm_weights_t    = gl<bf16, 1, 1, -1, hidden_dim, sv_bf<hidden_dim>>;
    using rope_table_t      = gl<float, 1, 1, -1, head_dim,  sv_fl<head_dim>>;

    // KV cache: paged layout [num_layers * num_pages, page_size / kv_block_size, num_kv_heads, head_dim]
    // Indexed as: cache[{num_pages * layer + page_idx, iter_in_page, kv_head, 0}]
    using kv_cache_t        = gl<bf16, -1, -1, num_kv_heads, head_dim,
                                 st_bf<kv_block_size, head_dim>>;

    // Paged KV metadata (CSR format, same as vLLM / Megakernels throughput)
    using int32_vector_t    = gl<int, 1, 1, 1, -1>;

    using barriers          = gl<uint, -1, -1, -1, -1>;

    // vm stuff
    barriers         Bar;
    instruction_layout instructions;
    timing_layout    timings;

    // model weights
    weights_t        qkv_weights;
    norm_weights_t   attn_norm_weights;
    weights_t        o_weights;
    norm_weights_t   mlp_norm_weights;
    weights_t        up_weights;
    weights_t        gate_weights;
    weights_big_t    down_weights;
    norm_weights_t   lm_head_norm_weights;
    weights_t        lm_head_weights;

    // kv cache
    kv_cache_t       k_cache;
    kv_cache_t       v_cache;

    // rope tables
    rope_table_t     rope_cos;
    rope_table_t     rope_sin;

    // activation buffers
    activations_t    hidden_states;
    activations_t    rms_rope_intermediates;
    activations_t    rms_gate_intermediates;
    activations_t    q_post_rope;
    activations_t    attn_out;
    activations_big_t silu_out;
    activations_t    rms_lm_head_intermediates;
    logits_t         logits;

    // Per-token position IDs (for RoPE lookup).
    // [batch_size] int32 — position_ids[batch_idx] gives the RoPE position for that token.
    int32_vector_t   position_ids;

    // Paged KV cache index arrays (CSR format, following vLLM / Megakernels throughput).
    // decode_kv_indptr:        [batch_size + 1]  — CSR row pointers into decode_kv_indices
    // decode_kv_indices:       [total_pages]     — physical page IDs for decode sequences
    // decode_kv_last_page_len: [batch_size]      — number of valid tokens in last page per seq
    // kv_append_indices:       [batch_size]      — flat slot index for writing new KV entries
    int32_vector_t   decode_kv_indptr;
    int32_vector_t   decode_kv_indices;
    int32_vector_t   decode_kv_last_page_len;
    int32_vector_t   kv_append_indices;

    // Prefill KV metadata (CSR format, same structure as decode).
    // prefill_qo_indptr:        [num_prefill_seqs + 1]  — CSR row pointers into q_post_rope
    // prefill_kv_indptr:        [num_prefill_seqs + 1]  — CSR row pointers into prefill_kv_indices
    // prefill_kv_indices:       [total_prefill_pages]   — physical page IDs for prefill sequences
    // prefill_kv_last_page_len: [num_prefill_seqs]      — valid tokens in last page per prefill seq
    int32_vector_t   prefill_qo_indptr;
    int32_vector_t   prefill_kv_indptr;
    int32_vector_t   prefill_kv_indices;
    int32_vector_t   prefill_kv_last_page_len;
    int              num_prefill_tokens;

    float            attn_scale;
    float            rms_norm_eps;
    int              num_pages;
    int              batch_size;

    dim3 grid()                  { return dim3(sm_count); }
    dim3 block()                 { return dim3(config::NUM_THREADS); }
    int  dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

typedef sm89_globals_t<
    SM89_NUM_LAYERS,
    SM89_HIDDEN_DIM,
    SM89_INTERMEDIATE_DIM,
    SM89_HEAD_DIM,
    SM89_NUM_ATTENTION_HEADS,
    SM89_NUM_KV_HEADS,
    SM89_KV_BLOCK_SIZE,
    SM89_KV_PAGE_SIZE,
    SM89_MATMUL_BATCH_BLOCK_SIZE,
    SM89_SM_COUNT>
    llama_sm89_globals;

// Forward declarations (no default args — definitions provide them)
template <typename Config, typename Globals> struct attn_norm;
template <typename Config, typename Globals> struct qkv_rope_append;
template <typename Config, typename Globals> struct attention_prefill;
template <typename Config, typename Globals> struct attention_decode;
template <typename Config, typename Globals> struct o_proj;
template <typename Config, typename Globals> struct mlp_norm;
template <typename Config, typename Globals> struct gate_silu;
template <typename Config, typename Globals> struct up_matmul;
template <typename Config, typename Globals> struct downproj;
template <typename Config, typename Globals> struct lm_head_norm;
template <typename Config, typename Globals> struct lm_head;

} // namespace kittens::prototype::vm
