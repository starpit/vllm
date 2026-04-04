// sm89 port of rms_norm.cu
// Changes from the Hopper/Blackwell version:
//   - tma::expect / tma::load_async  →  group::load_async + cp.async.wait + arrive()
//   - tma::store_async               →  group::store (synchronous)
//   - No warpgroup register ops (guarded in vm.cuh)

#include "llama_sm89.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm {

using globals = llama_sm89_globals;
using config  = llama_sm89_config;

// Helper: cp.async commit + wait-all (used after group::load_async calls)
__device__ static inline void cp_async_wait_all() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_all;\n"     ::: "memory");
}

template <
    auto weights_ptr,
    auto outputs_ptr,
    int  _opcode,
    typename gmem_waiter,
    typename Config  = config,
    typename Globals = globals>
struct rms_op_sm89 {
    static constexpr int opcode = _opcode;
    static constexpr int REDUCTION_DIM_PER_WARP = Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

    using activations_vec = sv_bf<Globals::hidden_dim>;
    using weights_vec     = sv_bf<Globals::hidden_dim>;

    struct parsed_instruction {
        int layer_idx;
        int batch_idx;
        __device__ inline parsed_instruction(typename Config::instruction_t &instruction) {
            layer_idx = instruction[1];
            batch_idx = instruction[2];
        }
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
    };

    // Semaphores
    __device__ static inline semaphore &activations_arrived(state<Config> &s) { return s.semaphores()[0]; }
    __device__ static inline semaphore &weights_arrived(state<Config> &s)     { return s.semaphores()[1]; }
    __device__ static inline semaphore &outputs_arrived(state<Config> &s)     { return s.semaphores()[2]; }

    // Use page 0 for both activations and weights (they fit in one 8KB page together
    // for hidden_dim=2048: sv_bf<2048>=4KB activations + sv_bf<2048>=4KB weights = 8KB exactly).
    // For hidden_dim=4096 (LLaMA 8B): sv_bf<4096>=8KB each → need 2 pages.
    static constexpr int ACT_PAGE = 0;
    static constexpr int WGT_PAGE = (Globals::hidden_dim * 2 > Config::PAGE_SIZE) ? 1 : 0;

    __device__ static inline activations_vec &get_activations_vec(state<Config> &s) {
        return *reinterpret_cast<activations_vec *>(s.pages[s.pid(ACT_PAGE)].data);
    }
    __device__ static inline weights_vec &get_weights_vec(state<Config> &s) {
        // Weights follow activations in the same or next page.
        char *base = reinterpret_cast<char *>(s.pages[s.pid(ACT_PAGE)].data);
        if constexpr (WGT_PAGE == ACT_PAGE) {
            return *reinterpret_cast<weights_vec *>(base + sizeof(activations_vec));
        } else {
            return *reinterpret_cast<weights_vec *>(s.pages[s.pid(WGT_PAGE)].data);
        }
    }

    struct controller {
        static __device__ int release_lid(const Globals &g,
                                          typename Config::instruction_t &instruction,
                                          int &query) {
            return query;
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            init_semaphore(activations_arrived(s), 1);
            init_semaphore(weights_arrived(s),     1);
            init_semaphore(outputs_arrived(s),     Config::NUM_CONSUMER_WARPS);
            return 3;
        }
    };

    struct loader {
        static __device__ inline void gmem_wait(const Globals &g, state<Config> &s) {}

        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            // Clear scratch partial-sum slots.
            ((uint64_t *)s.scratch())[laneid()] = 0;
            warp::sync();

            // Wait for pages to be ready (only lane 0 checks; others sync).
            if (laneid() == 0) {
                s.wait_page_ready(s.pid(ACT_PAGE));
                if constexpr (WGT_PAGE != ACT_PAGE)
                    s.wait_page_ready(s.pid(WGT_PAGE));
            }
            warp::sync(); // all lanes wait for page_ready check

            // ── cp.async: load RMS scale weights (ALL lanes participate) ────
            {
                weights_vec &rms_scale = get_weights_vec(s);
                auto &weights_global   = g.*weights_ptr;
                warp::load_async(rms_scale, weights_global, {inst.layer_idx, 0});
                cp_async_wait_all();
            }
            if (laneid() == 0) arrive(weights_arrived(s));

            // ── gmem barrier: wait for previous op (lane 0 spins) ────────────
            if (laneid() == 0) gmem_waiter::gmem_wait(g, s, inst);
            warp::sync(); // all lanes wait for gmem_wait to complete

            // ── cp.async: load activations (ALL lanes participate) ───────────
            {
                activations_vec &act_vec = get_activations_vec(s);
                warp::load_async(act_vec, g.hidden_states, {inst.batch_idx, 0});
                cp_async_wait_all();
            }
            if (laneid() == 0) arrive(activations_arrived(s));

            // Release unused pages (the pages we didn't load into).
            if (laneid() >= 1 && laneid() < Config::NUM_PAGES) {
                int extra_page = (WGT_PAGE != ACT_PAGE) ? laneid() + WGT_PAGE : laneid();
                if (extra_page < Config::NUM_PAGES) {
                    s.wait_page_ready(s.pid(extra_page));
                    s.finish_page(s.pid(extra_page), Config::NUM_CONSUMER_WARPS);
                }
            }
        }
    };

    struct launcher {
        // No MMA work for rms_norm; just pass through.
        static __device__ void run(const Globals &g, state<Config> &s) {
#ifdef KITTENS_BLACKWELL
            if (warp::laneid() == 0) {
                s.wait_tensor_ready();
                arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
            }
#endif
        }
    };

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            rv_fl<REDUCTION_DIM_PER_WARP> act_vec, copy_vec, scale_vec;

            sv_bf<REDUCTION_DIM_PER_WARP> *act_smem =
                reinterpret_cast<sv_bf<REDUCTION_DIM_PER_WARP> *>(
                    s.pages[s.pid(ACT_PAGE)].ptr());
            sv_bf<REDUCTION_DIM_PER_WARP> *wgt_smem =
                reinterpret_cast<sv_bf<REDUCTION_DIM_PER_WARP> *>(
                    s.pages[s.pid(ACT_PAGE)].ptr(sizeof(activations_vec)));
            if constexpr (WGT_PAGE != ACT_PAGE) {
                wgt_smem = reinterpret_cast<sv_bf<REDUCTION_DIM_PER_WARP> *>(
                    s.pages[s.pid(WGT_PAGE)].ptr());
            }

            // Wait for activations in shmem.
            wait(activations_arrived(s), 0);
            warp::load(act_vec, act_smem[warpid()]);
            warp::sync();

            // RMS norm: compute partial sum of squares.
            warp::copy(copy_vec, act_vec);
            warp::mul(copy_vec, copy_vec, copy_vec);
            float partial_sum = warp::sum(copy_vec);

            auto *smem_sums = (float *)s.scratch();
            if (laneid() == 0) smem_sums[warpid()] = partial_sum;
            group<Config::NUM_CONSUMER_WARPS>::sync(0);

            float full_sum = 0.f;
            for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++)
                full_sum += smem_sums[i];

            float rms = rsqrtf(full_sum / (float)Globals::hidden_dim + g.rms_norm_eps);
            warp::copy(copy_vec, act_vec);
            warp::mul(copy_vec, copy_vec, rms);
            warp::copy(act_vec, copy_vec);

            // Multiply by learned scale.
            wait(weights_arrived(s), 0);
            warp::load(scale_vec, wgt_smem[warpid()]);
            warp::sync();
            warp::mul(act_vec, act_vec, scale_vec);

            // Write result back to shmem for the storer.
            warp::store(act_smem[warpid()], act_vec);
            warp::sync();
            warp::arrive(outputs_arrived(s));
        }
    };

    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};

            // Wait for all consumers to finish writing to shmem (lane 0 waits).
            if (warp::laneid() == 0) wait(outputs_arrived(s), 0);
            warp::sync(); // all lanes wait for the semaphore

            // ── plain store: ALL lanes participate in warp::store ────────────
            {
                activations_vec &act_vec = get_activations_vec(s);
                auto &outputs_global     = g.*outputs_ptr;
                warp::store(outputs_global, act_vec, {inst.batch_idx, 0});
            }

            if (warp::laneid() == 0) {
                s.finish_page(s.pid(ACT_PAGE), Config::NUM_CONSUMER_WARPS);
                if constexpr (WGT_PAGE != ACT_PAGE)
                    s.finish_page(s.pid(WGT_PAGE), Config::NUM_CONSUMER_WARPS);
            }

            warp::sync();
            asm volatile("fence.acq_rel.gpu;\n");

            if (warp::laneid() == 0) {
                int batch_block_idx = inst.batch_idx / Globals::matmul_batch_block_size;
                // Increment by matmul_batch_block_size to match the convention used by waiters
                // (e.g. qkv_gmem_waiter waits for >= matmul_batch_block_size).
                // The Hopper design fired one instruction per token (256 × +1 = 256 total);
                // the sm89 design fires one instruction per batch block (+256 at once).
                atomicAdd(&g.Bar[{inst.layer_idx, opcode - 1, batch_block_idx, 0}], 1u);
            }
        }
    };
};

// ─── concrete op types ───────────────────────────────────────────────────────

struct attn_norm_gmem_waiter {
    template <typename Cfg, typename G, typename Inst>
    static __device__ inline void gmem_wait(const G &g, state<Cfg> &s, Inst &inst) {
        int bb = inst.batch_idx / G::matmul_batch_block_size;
        if (inst.layer_idx > 0) {
            int _w = 0;
            while (*(volatile int *)&g.Bar[{inst.layer_idx - 1, OPCODE_DownProjResidual - 1, bb, 0}]
                   < (int)(G::hidden_dim / G::down_out_block)) {
                __nanosleep(20);
                if (++_w > 50000000 && inst.batch_idx == 0 && warp::laneid() == 0) {
                    printf("ATTN_NORM HANG: waiting DownProj barrier, layer=%d bb=%d val=%d need=%d\n",
                        inst.layer_idx - 1, bb,
                        *(volatile int *)&g.Bar[{inst.layer_idx - 1, OPCODE_DownProjResidual - 1, bb, 0}],
                        (int)(G::hidden_dim / G::down_out_block));
                    break;
                }
            }
        }
    }
};
template <typename Config, typename Globals>
struct attn_norm : rms_op_sm89<&Globals::attn_norm_weights, &Globals::rms_rope_intermediates,
                               OPCODE_AttnNorm, attn_norm_gmem_waiter, Config, Globals> {};

struct mlp_norm_gmem_waiter {
    template <typename Cfg, typename G, typename Inst>
    static __device__ inline void gmem_wait(const G &g, state<Cfg> &s, Inst &inst) {
        int bb = inst.batch_idx / G::matmul_batch_block_size;
        while (*(volatile int *)&g.Bar[{inst.layer_idx, OPCODE_O_ProjResidual - 1, bb, 0}]
               < (int)(G::hidden_dim / G::o_proj_out_block))
            __nanosleep(20);
    }
};
template <typename Config, typename Globals>
struct mlp_norm : rms_op_sm89<&Globals::mlp_norm_weights, &Globals::rms_gate_intermediates,
                              OPCODE_MlpNorm, mlp_norm_gmem_waiter, Config, Globals> {};

struct lm_head_norm_gmem_waiter {
    template <typename Cfg, typename G, typename Inst>
    static __device__ inline void gmem_wait(const G &g, state<Cfg> &s, Inst &inst) {
        int bb = inst.batch_idx / G::matmul_batch_block_size;
        while (*(volatile int *)&g.Bar[{G::num_hidden_layers - 1, OPCODE_DownProjResidual - 1, bb, 0}]
               < (int)(G::hidden_dim / G::down_out_block))
            __nanosleep(20);
    }
};
template <typename Config, typename Globals>
struct lm_head_norm : rms_op_sm89<&Globals::lm_head_norm_weights, &Globals::rms_lm_head_intermediates,
                                  OPCODE_LM_HeadNorm, lm_head_norm_gmem_waiter, Config, Globals> {};

} // namespace kittens::prototype::vm
