// sm89 port of gate_silu.cu
// GateSiLU: X @ gate_weights^T, apply SiLU element-wise.

#include "llama_sm89.cuh"
#include "matmul_pipeline_sm89.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm {

using globals = llama_sm89_globals;
using config  = llama_sm89_config;

struct gate_silu_gmem_waiter {
    template <typename Cfg, typename G, typename Inst>
    static __device__ inline void gmem_wait(const G &g, state<Cfg> &s, Inst &inst) {
        int _w = 0;
        while (*(volatile int *)&g.Bar[{inst.layer, OPCODE_MlpNorm - 1, inst.row, 0}]
               < G::matmul_batch_block_size) {
            __nanosleep(20);
            if (++_w > 50000000 && warp::laneid() == 0) {
                printf("GATE_SILU HANG: layer=%d row=%d col=%d val=%d need=%d\n",
                    inst.layer, inst.row, inst.col,
                    *(volatile int *)&g.Bar[{inst.layer, OPCODE_MlpNorm - 1, inst.row, 0}],
                    (int)G::matmul_batch_block_size);
                break;
            }
        }
    }
};

template <typename Config = config, typename Globals = globals>
struct gate_silu {
    static constexpr int opcode    = OPCODE_GateSiLU;
    static constexpr int BATCH_BLOCK = Globals::matmul_batch_block_size;
    static constexpr int OUT_BLOCK   = Globals::gate_out_block;
    static constexpr int NUM_ITERS   = Globals::hidden_dim / SM89_PIPELINE_K_DIM;

    struct parsed_instruction {
        int layer, row, col;
        __device__ inline parsed_instruction(typename Config::instruction_t &i) {
            layer = i[1]; row = i[2]; col = i[3];
        }
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    using pipeline = matmul_pipeline_sm89<Config, Globals, parsed_instruction, gate_silu_gmem_waiter,
                                          &Globals::rms_gate_intermediates, &Globals::gate_weights,
                                          NUM_ITERS, OUT_BLOCK>;
    using acc_rt   = typename pipeline::acc_rt;

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
            // sm89: do NOT pre-wait on outputs_arrived — loader signals it only after all
            // iters complete, but the consumer must drain inputs_finished first. Pre-waiting
            // here deadlocks (same pattern as qkv_rope_append_sm89.cu).
            acc_rt acc;
            pipeline::consumer_loop(s, g, acc, inst.layer);

            // Apply SiLU in-register: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
            // TK stores fp32 rt tiles; operate element-by-element.
            // SiLU is applied to each accumulator element.
            #pragma unroll
            for (int r = 0; r < acc_rt::height; r++) {
                #pragma unroll
                for (int c = 0; c < acc_rt::width; c++) {
                    #pragma unroll
                    for (int rr = 0; rr < acc_rt::tile_size_row; rr++) {
                        #pragma unroll
                        for (int cc = 0; cc < acc_rt::tile_size_col / 2; cc++) {
                            float2 &v = acc.tiles[r][c].data[rr * (acc_rt::tile_size_col / 2) + cc];
                            v.x = v.x * (1.f / (1.f + expf(-v.x)));
                            v.y = v.y * (1.f / (1.f + expf(-v.y)));
                        }
                    }
                }
            }

            // Store bf16 result to silu_out.
            rt_bf<16, OUT_BLOCK> out_bf;
            warp::copy(out_bf, acc);
            warp::store(g.silu_out, out_bf, {inst.row * (BATCH_BLOCK / 16) + ::kittens::warpid(), inst.col});
            __threadfence();

            group<Config::NUM_CONSUMER_WARPS>::sync(0);
            if (laneid() == 0 && ::kittens::warpid() == 0)
                atomicAdd(&g.Bar[{inst.layer, opcode - 1, inst.row, 0}], 1);
        }
    };
};

} // namespace kittens::prototype::vm
