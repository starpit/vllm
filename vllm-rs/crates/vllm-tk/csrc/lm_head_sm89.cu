// sm89 port of lm_head.cu
// LM head: rms_lm_head_intermediates @ lm_head_weights^T → logits

#include "llama_sm89.cuh"
#include "matmul_pipeline_sm89.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm {

using globals = llama_sm89_globals;
using config  = llama_sm89_config;

struct lm_head_gmem_waiter;

template <typename Config = config, typename Globals = globals>
struct lm_head {
    static constexpr int opcode    = OPCODE_LM_Head;
    static constexpr int BATCH_BLOCK = Globals::matmul_batch_block_size;
    static constexpr int OUT_BLOCK   = Globals::lm_head_out_block;
    static constexpr int NUM_ITERS   = Globals::hidden_dim / SM89_PIPELINE_K_DIM;

    struct parsed_instruction {
        int row, col;
        __device__ inline parsed_instruction(typename Config::instruction_t &i) {
            row = i[1]; col = i[2];
        }
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    using pipeline = matmul_pipeline_sm89<Config, Globals, parsed_instruction, lm_head_gmem_waiter,
                                          &Globals::rms_lm_head_intermediates, &Globals::lm_head_weights,
                                          NUM_ITERS, OUT_BLOCK>;
    using acc_rt = typename pipeline::acc_rt;

    struct controller {
        static __device__ int release_lid(const Globals &g, typename Config::instruction_t &ins, int &q) {
            return pipeline::release_lid(g, ins, q);
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            return pipeline::init_semaphores(s);
        }
    };
    struct loader  { static __device__ void run(const Globals &g, state<Config> &s) { pipeline::loader_loop(s, g); pipeline::release_unused_pages(s); } };
    struct launcher { static __device__ void run(const Globals &g, state<Config> &s) { pipeline::launcher_loop(s, g); } };
    struct storer  { static __device__ void run(const Globals &g, state<Config> &s) {} };

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            // sm89: no pre-wait on outputs_arrived (would deadlock; see qkv_rope_append_sm89.cu)
            acc_rt acc;
            pipeline::consumer_loop(s, g, acc);

            // Store bf16 logits.
            rt_bf<16, OUT_BLOCK> out_bf;
            warp::copy(out_bf, acc);
            // group<1>::warpid()=0 always; use kittens::warpid() to place each warp's rows.
            warp::store(g.logits, out_bf, {inst.row * (BATCH_BLOCK / 16) + ::kittens::warpid(), inst.col});
        }
    };
};

struct lm_head_gmem_waiter {
    template <typename Cfg, typename G, typename Inst>
    static __device__ inline void gmem_wait(const G &g, state<Cfg> &s, Inst &inst) {
        while (*(volatile int *)&g.Bar[{0, OPCODE_LM_HeadNorm - 1, inst.row, 0}]
               < (int)G::matmul_batch_block_size)
            __nanosleep(20);
    }
};

} // namespace kittens::prototype::vm
