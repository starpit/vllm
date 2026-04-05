// sm89 port of matmul_adds.cu (O projection and down projection with residual add).
// Replaces:
//   - s.tensor_alloc + tt<> + kittens::mm  →  warp::mma on rt_fl accumulator
//   - warp::tma::store_add_async           →  load existing + add + store (plain)
//   - tma::store_async_read_wait           →  __threadfence_block()

#include "llama_sm89.cuh"
#include "matmul_pipeline_sm89.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm {

using globals = llama_sm89_globals;
using config  = llama_sm89_config;

template <
    auto InputActivationsPtr,
    auto WeightsPtr,
    auto OutputActivationsPtr,
    int  iters,
    int  _opcode,
    typename gmem_waiter,
    int  _out_block,
    typename Config  = config,
    typename Globals = globals>
struct MatMulAddOp_sm89 {
    static constexpr int opcode    = _opcode;
    static constexpr int BATCH_BLOCK = Globals::matmul_batch_block_size;
    static constexpr int OUT_BLOCK   = _out_block;

    struct parsed_instruction {
        int layer;
        int row;   // batch block index
        int col;   // output block index
        __device__ inline parsed_instruction(typename Config::instruction_t &instruction) {
            layer = instruction[1];
            row   = instruction[2];
            col   = instruction[3];
        }
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    using pipeline = matmul_pipeline_sm89<Config, Globals, parsed_instruction, gmem_waiter,
                                          InputActivationsPtr, WeightsPtr, iters, _out_block>;
    using acc_rt   = typename pipeline::acc_rt;

    struct controller {
        static __device__ int release_lid(const Globals &g,
                                          typename Config::instruction_t &instruction,
                                          int &query) {
            return pipeline::release_lid(g, instruction, query);
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            return pipeline::init_semaphores(s);
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            pipeline::loader_loop(s, g, inst.layer);
            pipeline::release_unused_pages(s);
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, state<Config> &s) {
            pipeline::launcher_loop(s, g);
        }
    };

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};

            // On sm89, consumers do warp::mma inside consumer_loop — do NOT pre-wait on
            // outputs_arrived (loader signals it only after consumer_loop, causing deadlock).
            // Run warp-scope GEMM accumulation.
            acc_rt acc;
            pipeline::consumer_loop(s, g, acc, inst.layer);

            // Convert fp32 accumulator → bf16 and store to output (with residual add).
            auto &OutputActivations = g.*OutputActivationsPtr;
            const int warp_row = inst.row * (BATCH_BLOCK / 16) + ::kittens::warpid();

            // Cast accumulator to bf16, load residual, add, store.
            rt_bf<16, OUT_BLOCK> acc_bf;
            warp::copy(acc_bf, acc);
            rt_bf<16, OUT_BLOCK> existing;
            warp::load(existing, OutputActivations, {warp_row, inst.col});
            warp::add(acc_bf, acc_bf, existing);
            warp::store(OutputActivations, acc_bf, {warp_row, inst.col});
            __threadfence();   // flush to L2 so dependent warps/SMs see the update

            // Sync all consumer warps so all writes are globally visible,
            // then only warp 0 signals — otherwise 16 warps × N columns
            // overshoots the threshold and dependent ops start too early.
            group<Config::NUM_CONSUMER_WARPS>::sync(0);
            if (laneid() == 0 && ::kittens::warpid() == 0) {
                atomicAdd(&g.Bar[{inst.layer, opcode - 1, inst.row, 0}], 1);
            }
        }
    };

    // No separate storer needed: consumer writes directly via warp::store.
    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            // Nothing: consumer handled the output write + barrier signal.
        }
    };
};

// ─── concrete ops ────────────────────────────────────────────────────────────

struct o_proj_gmem_waiter {
    template <typename Cfg, typename G, typename Inst>
    static __device__ inline void gmem_wait(const G &g, state<Cfg> &s, Inst &inst) {
        while (*(volatile int *)&g.Bar[{inst.layer, OPCODE_GQA_AttentionDecode - 1, inst.row, 0}]
               < (int)(G::matmul_batch_block_size * G::num_kv_heads))
            __nanosleep(20);
    }
};
template <typename Config, typename Globals>
struct o_proj : MatMulAddOp_sm89<
    &Globals::attn_out,
    &Globals::o_weights,
    &Globals::hidden_states,
    Globals::hidden_dim / SM89_PIPELINE_K_DIM,
    OPCODE_O_ProjResidual,
    o_proj_gmem_waiter,
    Globals::o_proj_out_block,
    Config, Globals> {};

struct downproj_gmem_waiter {
    template <typename Cfg, typename G, typename Inst>
    static __device__ inline void gmem_wait(const G &g, state<Cfg> &s, Inst &inst) {
        while (*(volatile int *)&g.Bar[{inst.layer, OPCODE_UpMatmul - 1, inst.row, 0}]
               < (int)(G::intermediate_dim / G::up_out_block))
            __nanosleep(20);
    }
};
template <typename Config, typename Globals>
struct downproj : MatMulAddOp_sm89<
    &Globals::silu_out,
    &Globals::down_weights,
    &Globals::hidden_states,
    Globals::intermediate_dim / SM89_PIPELINE_K_DIM,
    OPCODE_DownProjResidual,
    downproj_gmem_waiter,
    Globals::down_out_block,
    Config, Globals> {};

} // namespace kittens::prototype::vm
