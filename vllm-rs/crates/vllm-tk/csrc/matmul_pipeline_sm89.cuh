// sm89 GEMM pipeline for the KVM megakernel.
//
// Consumer-self-loading design: all 8 consumer warps cooperatively issue
// cp.async loads via group<8>::load_async (256 threads), eliminating the
// loader warp as data-movement bottleneck.  K_DIM=64 halves iteration count.

#pragma once

#include "llama_sm89.cuh"

namespace kittens::prototype::vm {

static constexpr int SM89_PIPELINE_K_DIM = 64;

template <typename Config, typename Globals,
          typename parsed_instruction,
          typename gmem_waiter,
          auto A_Ptr,
          auto B_Ptr,
          int  _num_iters,
          int  _out_block>
struct matmul_pipeline_sm89 {
    static_assert(Config::NUM_CONSUMER_WARPS * 16 == Globals::matmul_batch_block_size,
                  "Each consumer warp handles 16 rows; NUM_CONSUMER_WARPS * 16 must equal BATCH_BLOCK");

    static constexpr int BATCH_BLOCK = Globals::matmul_batch_block_size;
    static constexpr int OUT_BLOCK   = _out_block;
    static constexpr int K_DIM       = SM89_PIPELINE_K_DIM;

    using a_st = st_bf<BATCH_BLOCK, K_DIM>;
    using b_st = st_bf<OUT_BLOCK,   K_DIM>;

    using acc_rt = rt_fl<16, OUT_BLOCK>;

    static constexpr int INPUT_PIPELINE_STAGES = 2;
    static constexpr int A_PAGES_PER_STAGE = sizeof(a_st) / Config::PAGE_SIZE
                                           + ((sizeof(a_st) % Config::PAGE_SIZE) ? 1 : 0);
    static constexpr int B_PAGES_PER_STAGE = sizeof(b_st) / Config::PAGE_SIZE
                                           + ((sizeof(b_st) % Config::PAGE_SIZE) ? 1 : 0);
    static constexpr int PAGES_PER_STAGE   = A_PAGES_PER_STAGE + B_PAGES_PER_STAGE;

    static constexpr int SEM_COUNT = 0;

    __device__ static inline semaphore &inputs_arrived(state<Config> &s, int stage) {
        return *reinterpret_cast<semaphore*>((char*)s.scratch() + stage * 8);
    }
    __device__ static inline semaphore &inputs_finished(state<Config> &s, int stage) {
        return *reinterpret_cast<semaphore*>((char*)s.scratch() + (INPUT_PIPELINE_STAGES + stage) * 8);
    }
    __device__ static inline semaphore &outputs_arrived(state<Config> &s) {
        return *reinterpret_cast<semaphore*>((char*)s.scratch() + 2 * INPUT_PIPELINE_STAGES * 8);
    }

    __device__ static inline int a_page_base(state<Config> &s, int stage) {
        return s.pid(stage * PAGES_PER_STAGE);
    }
    __device__ static inline int b_page(state<Config> &s, int stage) {
        return s.pid(stage * PAGES_PER_STAGE + A_PAGES_PER_STAGE);
    }

    __device__ static inline a_st &get_a_smem(state<Config> &s, int stage) {
        return *reinterpret_cast<a_st *>(s.pages[a_page_base(s, stage)].data);
    }
    __device__ static inline b_st &get_b_smem(state<Config> &s, int stage) {
        return *reinterpret_cast<b_st *>(s.pages[b_page(s, stage)].data);
    }

    __device__ static inline int init_semaphores(state<Config> &s) {
        for (int i = 0; i < INPUT_PIPELINE_STAGES; i++) {
            init_semaphore(inputs_arrived(s, i),  1);
            init_semaphore(inputs_finished(s, i), Config::NUM_CONSUMER_WARPS);
        }
        init_semaphore(outputs_arrived(s), 1);
        return 0;
    }

    __device__ static inline int release_lid(const Globals &g,
                                             typename Config::instruction_t &instruction,
                                             int &query) {
        constexpr int PIPELINE_PAGES = INPUT_PIPELINE_STAGES * PAGES_PER_STAGE;
        if (query >= PIPELINE_PAGES) return query;
        auto remainder = _num_iters % INPUT_PIPELINE_STAGES;
        if (remainder == 0) {
            return query;
        } else {
            return (query + PAGES_PER_STAGE) % PIPELINE_PAGES;
        }
    }

    // ── loader: manages pages and gmem readiness only ────────────────────────
    template <int _stages_to_not_release = 0>
    __device__ static inline void loader_loop(state<Config> &s, const Globals &g,
                                              int layer_idx = 0) {
        parsed_instruction inst{s};
        int stage = 0;

        for (int iter = 0; iter < _num_iters; iter++) {
            if (laneid() == 0) {
                if (iter < INPUT_PIPELINE_STAGES) {
                    for (int p = 0; p < PAGES_PER_STAGE; p++)
                        s.wait_page_ready(s.pid(stage * PAGES_PER_STAGE + p));
                }

                wait(inputs_finished(s, stage),
                     (iter % (2 * INPUT_PIPELINE_STAGES)) < INPUT_PIPELINE_STAGES);

                gmem_waiter::gmem_wait(g, s, inst);

                arrive(inputs_arrived(s, stage));
            }

            stage = (stage + 1) % INPUT_PIPELINE_STAGES;
        }

        for (int i = 0; i < INPUT_PIPELINE_STAGES; i++) {
            if (laneid() == 0) {
                wait(inputs_finished(s, stage),
                     ((_num_iters + i) % (2 * INPUT_PIPELINE_STAGES)) < INPUT_PIPELINE_STAGES);

                if (i == INPUT_PIPELINE_STAGES - 1)
                    arrive(outputs_arrived(s));

                if (i < INPUT_PIPELINE_STAGES - _stages_to_not_release) {
                    for (int p = 0; p < PAGES_PER_STAGE; p++)
                        s.finish_page(s.pid(stage * PAGES_PER_STAGE + p), Config::NUM_CONSUMER_WARPS);
                }
            }
            stage = (stage + 1) % INPUT_PIPELINE_STAGES;
        }
    }

    __device__ static inline void release_unused_pages(state<Config> &s, int first_reserved = Config::NUM_PAGES) {
        if (warp::laneid() == 0) {
            int first_unused = INPUT_PIPELINE_STAGES * PAGES_PER_STAGE;
            for (int p = first_unused; p < first_reserved; p++) {
                int pid = s.pid(p);
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }
        }
    }

    __device__ static inline void launcher_loop(state<Config> &s, const Globals &g) {
#ifdef KITTENS_BLACKWELL
        if (warp::laneid() == 0) {
            s.wait_tensor_ready();
            arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
        }
#endif
    }

    // Load a 16-row B slice from shared memory with zero warpid row offset.
    __device__ static inline void load_b_slice(rt_bf<16, K_DIM> &dst, const st_bf<16, K_DIM> &src) {
        uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
        int lane = ::kittens::laneid();
        int row  = lane % 16;
        bf16_2 tmp[4];
        #pragma unroll
        for (int j = 0; j < K_DIM / 16; j++) {
            int col = j * 16 + (lane / 16) * 8;
            move<bf16_2>::ldsm4(tmp[0], tmp[1], tmp[2], tmp[3], src.idx(saddr, {row, col}));
            dst.tiles[0][j].data[0] = tmp[0];
            dst.tiles[0][j].data[1] = tmp[1];
            dst.tiles[0][j].data[2] = tmp[2];
            dst.tiles[0][j].data[3] = tmp[3];
        }
    }

    // ── consumer: self-loading + compute ─────────────────────────────────────
    static constexpr int CONSUMER_BAR = 14;

    __device__ static inline void consumer_loop(state<Config> &s, const Globals &g,
                                                acc_rt &acc, int layer_idx = 0) {
        parsed_instruction inst{s};

        warp::zero(acc);

        using b_slice_st = st_bf<16, K_DIM>;
        static constexpr int N_TILES = OUT_BLOCK / 16;

        int stage = 0;

        for (int iter = 0; iter < _num_iters; iter++) {
            // Wait for loader to signal pages are ready.
            wait(inputs_arrived(s, stage),
                 (iter % (2 * INPUT_PIPELINE_STAGES)) >= INPUT_PIPELINE_STAGES);

            a_st &a_smem = get_a_smem(s, stage);
            b_st &b_smem = get_b_smem(s, stage);

            // All 8 consumer warps cooperatively load A and B.
            group<Config::NUM_CONSUMER_WARPS>::load_async(a_smem, g.*A_Ptr, {inst.row, iter});
            group<Config::NUM_CONSUMER_WARPS>::load_async(b_smem, g.*B_Ptr, {layer_idx, inst.col, iter});

            asm volatile("cp.async.wait_all;\n" ::: "memory");
            group<Config::NUM_CONSUMER_WARPS>::sync(CONSUMER_BAR);

            // Per-warp compute.
            rt_bf<16, K_DIM> a_reg;
            {
                using a_slice_st = st_bf<16, K_DIM>;
                const a_slice_st &a_warp = reinterpret_cast<const a_slice_st *>(&a_smem)[::kittens::warpid()];
                warp::load(a_reg, a_warp);
            }

            b_slice_st *b_slices = reinterpret_cast<b_slice_st *>(&b_smem);
            #pragma unroll
            for (int n = 0; n < N_TILES; n++) {
                rt_bf<16, K_DIM> b_n; load_b_slice(b_n, b_slices[n]);
                warp::mma_ABt_base(acc.tiles[0][n], a_reg.tiles[0][0], b_n.tiles[0][0], acc.tiles[0][n]);
                #pragma unroll
                for (int k = 1; k < a_reg.width; k++)
                    warp::mma_ABt_base(acc.tiles[0][n], a_reg.tiles[0][k], b_n.tiles[0][k], acc.tiles[0][n]);
            }

            // Signal loader: done with this stage. With 2-stage double-buffering,
            // the next iteration uses the other stage so no shmem conflict.
            warp::arrive(inputs_finished(s, stage));

            stage = (stage + 1) % INPUT_PIPELINE_STAGES;
        }
    }
};

} // namespace kittens::prototype::vm
