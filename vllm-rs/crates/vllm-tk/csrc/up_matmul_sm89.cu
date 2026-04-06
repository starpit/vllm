// sm89 port of up_matmul.cu
// UpMatmul: X @ up_weights^T, element-wise multiply with silu_out (Hadamard product).

#include "llama_sm89.cuh"
#include "matmul_pipeline_sm89.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm {

using globals = llama_sm89_globals;
using config  = llama_sm89_config;

struct up_matmul_gmem_waiter {
    template <typename Cfg, typename G, typename Inst>
    static __device__ inline void gmem_wait(const G &g, state<Cfg> &s, Inst &inst) {
        // Wait for GateSiLU to write silu_out before we read it.
        while (*(volatile int *)&g.Bar[{inst.layer, OPCODE_GateSiLU - 1, inst.row, 0}]
               < (int)(G::intermediate_dim / G::gate_out_block))
            __nanosleep(20);
    }
};

template <typename Config = config, typename Globals = globals>
struct up_matmul {
    static constexpr int opcode    = OPCODE_UpMatmul;
    static constexpr int BATCH_BLOCK = Globals::matmul_batch_block_size;
    static constexpr int OUT_BLOCK   = Globals::up_out_block;
    static constexpr int NUM_ITERS   = Globals::hidden_dim / SM89_PIPELINE_K_DIM;

    struct parsed_instruction {
        int layer, row, col;
        __device__ inline parsed_instruction(typename Config::instruction_t &i) {
            layer = i[1]; row = i[2]; col = i[3];
        }
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    using pipeline = matmul_pipeline_sm89<Config, Globals, parsed_instruction, up_matmul_gmem_waiter,
                                          &Globals::rms_gate_intermediates, &Globals::up_weights,
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
    struct loader  { static __device__ void run(const Globals &g, state<Config> &s) { parsed_instruction i{s}; pipeline::loader_loop(s, g, i.layer); pipeline::release_unused_pages(s); } };
    struct launcher { static __device__ void run(const Globals &g, state<Config> &s) { pipeline::launcher_loop(s, g); } };
    struct storer  { static __device__ void run(const Globals &g, state<Config> &s) {} };

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            // sm89: no pre-wait on outputs_arrived (would deadlock; see qkv_rope_append_sm89.cu)
            acc_rt acc;
            pipeline::consumer_loop(s, g, acc, inst.layer);

            // Cast matmul result to bf16 first (frees fp32 acc registers).
            rt_bf<16, OUT_BLOCK> acc_bf;
            warp::copy(acc_bf, acc);

            // Load gate values from silu_out.
            rt_bf<16, OUT_BLOCK> gate_bf;
            warp::load(gate_bf, g.silu_out, {inst.row * (BATCH_BLOCK / 16) + ::kittens::warpid(), inst.col});

            // Element-wise multiply in fp32 to match PyTorch/HF precision.
            // PyTorch's element-wise * on bf16 tensors computes in fp32 internally.
            // We convert one packed pair at a time to avoid needing two full fp32
            // register tiles simultaneously (which exceeds register budget).
            #pragma unroll
            for (int r = 0; r < acc_bf.height; r++) {
                #pragma unroll
                for (int c = 0; c < acc_bf.width; c++) {
                    #pragma unroll
                    for (int k = 0; k < acc_bf.tiles[0][0].packed_per_thread; k++) {
                        bf16_2 &a = acc_bf.tiles[r][c].data[k];
                        bf16_2 &g_ = gate_bf.tiles[r][c].data[k];
                        float a_lo = __bfloat162float(__low2bfloat16(a));
                        float a_hi = __bfloat162float(__high2bfloat16(a));
                        float g_lo = __bfloat162float(__low2bfloat16(g_));
                        float g_hi = __bfloat162float(__high2bfloat16(g_));
                        a = __floats2bfloat162_rn(a_lo * g_lo, a_hi * g_hi);
                    }
                }
            }

            // Write result to silu_out (reuse buffer for down_proj input).
            warp::store(g.silu_out, acc_bf, {inst.row * (BATCH_BLOCK / 16) + ::kittens::warpid(), inst.col});
            __threadfence();

            group<Config::NUM_CONSUMER_WARPS>::sync(0);
            if (laneid() == 0 && ::kittens::warpid() == 0)
                atomicAdd(&g.Bar[{inst.layer, opcode - 1, inst.row, 0}], 1);
        }
    };
};

} // namespace kittens::prototype::vm
