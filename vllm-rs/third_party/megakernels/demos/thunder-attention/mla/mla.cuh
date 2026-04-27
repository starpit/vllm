#pragma once

#include "kittens.cuh"
#include "../../../include/megakernel.cuh"

#define MLA_GLOBAL_WORK_QUEUE

// MLA-specific opcodes
#define OPCODE_MLA_Partial 1
#define OPCODE_MLA_Reduction 2

// MLA model dimensions (from original ThunderKittens implementation)
#define MLA_QKRot_D 64
#define MLA_QVO_D 512
#define MLA_QVO_Dd2 (MLA_QVO_D/2)
#define MLA_NUM_ROWS 32
#define MLA_PAGE_SIZE 256
#define MLA_Q_HEADS 16

#ifdef KITTENS_BLACKWELL
#define MLA_MATMUL_BATCH_BLOCK_SIZE 256
#else
#define MLA_MATMUL_BATCH_BLOCK_SIZE 128
#endif

#ifdef KITTENS_BLACKWELL
#define SM_COUNT 148
#else
#define SM_COUNT 132
#endif

struct base_mla_config {
    // Instruction pipeline
    static constexpr int INSTRUCTION_PIPELINE_STAGES = 2;  // More stages for MLA pipelining

    // num bits required to represent num pipeline stages
    static constexpr int INSTRUCTION_PIPELINE_STAGES_BITS = 1;

    static constexpr int INSTRUCTION_WIDTH = 32;  // 128 bytes per instruction.
    using instruction_t = int[INSTRUCTION_WIDTH];

    // Timing info
    static constexpr int TIMING_WIDTH = 128;
    using timing_t = int[TIMING_WIDTH];

#ifdef MLA_GLOBAL_WORK_QUEUE
    static constexpr bool ENABLE_GLOBAL_WORK_QUEUE = true;
#else
    static constexpr bool ENABLE_GLOBAL_WORK_QUEUE = false;
#endif

    // How many semaphores are available for dynamic use?
    static constexpr int DYNAMIC_SEMAPHORES = 64;

    // Warp configuration for MLA workload
#ifdef KITTENS_BLACKWELL
    static constexpr int NUM_CONSUMER_WARPS = 16;
#else
    static constexpr int NUM_CONSUMER_WARPS = 8;
#endif
    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int CLUSTER_BLOCKS = 1;
    static constexpr int MAX_SHARED_MEMORY = kittens::MAX_SHARED_MEMORY;

    // Shared memory declared statically
    static constexpr int SCRATCH_BYTES = 1024;
    static constexpr int STATIC_SHARED_MEMORY =
        256 + INSTRUCTION_PIPELINE_STAGES * (SCRATCH_BYTES + (INSTRUCTION_WIDTH + TIMING_WIDTH) * 4 + DYNAMIC_SEMAPHORES * 8);
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    // Shared memory declared dynamically - use larger pages for MLA
    static constexpr int PAGE_SIZE = 73728;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;

#ifdef KITTENS_BLACKWELL
    static constexpr int CONSUMER_REGISTERS = 104;
#else
    static constexpr int CONSUMER_REGISTERS = 208;
#endif
    static constexpr int NON_CONSUMER_REGISTERS = 64;
};

struct mla_config_timer : public base_mla_config {
    static constexpr bool TIMING_RECORD_ENABLED = true;
};

struct mla_config : public base_mla_config {
    static constexpr bool TIMING_RECORD_ENABLED = true;
};

// MLA-specific type definitions (from original implementation)
using qrot_tile           = kittens::st_bf<64, MLA_QKRot_D>;
using qv_tile             = kittens::st_bf<64, MLA_QVO_D>;
using q_tile              = kittens::st_bf<64, MLA_QKRot_D + MLA_QVO_D>; // used for mma
using qrot_global         = kittens::gl<kittens::bf16, -1, -1, -1, MLA_QKRot_D, qrot_tile>;
using qv_global           = kittens::gl<kittens::bf16, -1, -1, -1, MLA_QVO_D, qv_tile>;
using krot_tile           = kittens::st_bf<MLA_NUM_ROWS, MLA_QKRot_D>;
using v_tile              = kittens::st_bf<MLA_NUM_ROWS, MLA_QVO_D>;
using k_tile              = kittens::st_bf<MLA_NUM_ROWS, MLA_QKRot_D + MLA_QVO_D>;
using vd2_tile            = kittens::st_bf<MLA_NUM_ROWS, MLA_QVO_Dd2>;
using krot_global         = kittens::gl<kittens::bf16, 1, -1, MLA_PAGE_SIZE, MLA_QKRot_D, krot_tile>;
using v_global            = kittens::gl<kittens::bf16, 1, -1, MLA_PAGE_SIZE, MLA_QVO_D, v_tile>;
using instructions_global = kittens::gl<int, 1, -1, -1, 32>;
using table_global        = kittens::gl<int, 1, 1, -1, -1>;
using out_tile            = kittens::st_bf<16, MLA_QVO_D>;
using outd2_tile          = kittens::st_bf<16, MLA_QVO_Dd2>;
using outd8_tile          = kittens::st_bf<16, MLA_QVO_D/8>;
using o_tile              = kittens::st_fl<16, MLA_QVO_D>;
using od2_tile            = kittens::st_fl<16, MLA_QVO_Dd2>;
using od8_tile            = kittens::st_fl<16, MLA_QVO_D/8>;
using o_global            = kittens::gl<kittens::bf16, -1, -1, -1, MLA_QVO_D, outd2_tile, out_tile>;

template<int Q_HEADS=16>
using o_scratch_global    = kittens::gl<float, -1, -1, Q_HEADS, MLA_QVO_D, od2_tile, od8_tile>;

template<int Q_HEADS=16>
using lvec_scratch_global       = kittens::gl<float,  1, -1, -1, Q_HEADS, kittens::sv_fl<16>>;
using completion_flag_global    = kittens::gl<int,    1,  1,  -1, -1>;

template<int Q_HEADS=16>
struct mla_globals {
    using instruction_layout = megakernel::instruction_layout<mla_config>;
    using timing_layout = megakernel::timing_layout<mla_config>;

    // VM stuff
    instruction_layout instructions;
    timing_layout timings;
    kittens::gl<int, 1, 1, 1, 1> global_instruction_index;

    // MLA-specific buffers
    qrot_global Qrot;
    qv_global Qv;
    krot_global Krot;
    v_global V;
    table_global Table;
    o_global O;
    o_scratch_global<Q_HEADS> O_scratch;
    lvec_scratch_global<Q_HEADS> Lvec_scratch;
    completion_flag_global completion_flag;
    const float Softmax_scale;
    int tic;

    dim3 grid()  { return dim3(SM_COUNT); }
    dim3 block() { return dim3(mla_config::NUM_THREADS); }
    int dynamic_shared_memory() { return mla_config::DYNAMIC_SHARED_MEMORY; }
};

using mla_16_globals = mla_globals<16>;
using mla_8_globals = mla_globals<8>;

// Location structure from original
struct location {
    int batch_idx; // batch_idx >=0, otherwise it's the negative index, minus one, into scratch
    int seq_idx;
};