#include "../../../include/megakernel.cuh"
#include "mla.cuh"

using namespace kittens;
using namespace megakernel;

template<int Q_HEADS=16>
struct mla_partial {
    static constexpr int opcode = OPCODE_MLA_Partial;
    using config = mla_config;
    using globals = mla_globals<Q_HEADS>;

    static constexpr int NUM_STAGES = 2;

    // Preserve original instruction format
    __device__ static inline int get_uid(state<config> &s) { return s.instruction()[1]; }
    __device__ static inline location get_dst(state<config> &s) { 
        return {s.instruction()[2], s.instruction()[3]}; 
    }
    __device__ static inline int get_q_batch_idx(state<config> &s) { return s.instruction()[4]; }
    __device__ static inline int get_q_seq_idx(state<config> &s) { return s.instruction()[5]; }
    __device__ static inline int get_start_pos(state<config> &s) { return s.instruction()[6]; }
    __device__ static inline int get_end_pos(state<config> &s) { return s.instruction()[7]; }
    __device__ static inline int get_length(state<config> &s, const globals &g) { return s.instruction()[8] - (g.Qv.depth() - (s.instruction()[5] + warpgroup::warpid()) - 1); }

    // Memory layout mapping to pages
    __device__ static inline int q_page(state<config> &s) { return s.pid(0); }
    __device__ static inline int kv_page(state<config> &s) { return s.pid(1); }
    __device__ static inline int output_page(state<config> &s) { return s.pid(2); }
    
    // Helper functions for page-based memory access
    __device__ static inline qrot_tile &get_qrot(state<config> &s) {
        static constexpr int offset_bytes = 8 * sizeof(qrot_tile); // last columns
        return *reinterpret_cast<qrot_tile*>(reinterpret_cast<uint8_t*>(s.pages[q_page(s)].data) + offset_bytes);
    }
    __device__ static inline qv_tile &get_qv(state<config> &s) {
        static constexpr int offset_bytes = 0; // start of tile.
        return *reinterpret_cast<qv_tile*>(reinterpret_cast<uint8_t*>(s.pages[q_page(s)].data) + offset_bytes);
    }
    __device__ static inline q_tile &get_q(state<config> &s) {
        return *reinterpret_cast<q_tile*>(s.pages[q_page(s)].data);
    }
    __device__ static inline krot_tile &get_krot(state<config> &s, int stage) {
        static constexpr int stage_bytes = sizeof(k_tile);
        static constexpr int offset_bytes = 8 * sizeof(krot_tile); // last columns
        return *reinterpret_cast<krot_tile*>(reinterpret_cast<uint8_t*>(s.pages[kv_page(s)].data) + stage_bytes * stage + offset_bytes);
    }
    __device__ static inline v_tile &get_v(state<config> &s, int stage) {
        static constexpr int stage_bytes = sizeof(k_tile);
        static constexpr int offset_bytes = 0; // start of tile.
        return *reinterpret_cast<v_tile*>(reinterpret_cast<uint8_t*>(s.pages[kv_page(s)].data) + stage_bytes * stage + offset_bytes);
    }
    __device__ static inline k_tile &get_k(state<config> &s, int stage) {
        static constexpr int stage_bytes = sizeof(k_tile);
        static constexpr int offset_bytes = 0;
        return *reinterpret_cast<k_tile*>(reinterpret_cast<uint8_t*>(s.pages[kv_page(s)].data) + stage_bytes * stage + offset_bytes);
    }
    __device__ static inline sv_fl<64> &get_max_vec(state<config> &s) {
        return reinterpret_cast<sv_fl<64>*>(s.scratch())[0];
    }
    __device__ static inline sv_fl<64> &get_norm_vec(state<config> &s) {
        return reinterpret_cast<sv_fl<64>*>(s.scratch())[1];
    }
    __device__ static inline st_bf<64, krot_tile::rows> &get_att_block(state<config> &s, int stage) {
        return reinterpret_cast<st_bf<64, krot_tile::rows>*>(s.pages[output_page(s)].data)[stage];
    }
    __device__ static inline sv_fl<16> &get_l_vec(state<config> &s, int w) {
        return reinterpret_cast<sv_fl<16>*>(s.scratch())[2*w]; // This forces 128-byte alignments as needed by TMA.
    }
    template<typename T> __device__ static inline st<T, 16, vd2_tile::cols> &get_o(state<config> &s, int wg, int w) {
        if constexpr (std::is_same_v<T, float>) {
            if(wg == 0) {
                return reinterpret_cast<st<T, 16, vd2_tile::cols>*>(s.pages[output_page(s)].data)[w];
            }
            else {
                return reinterpret_cast<st<T, 16, vd2_tile::cols>*>(s.pages[q_page(s)].data)[w];
            }
        }
        else {
            return reinterpret_cast<st<T, 16, vd2_tile::cols>*>(s.pages[output_page(s)].data)[wg*4 + w];
        }
    }

    // Semaphores
    __device__ static inline semaphore &kv_arrived(state<config> &s, int stage) {
        return s.semaphores()[stage];
    }
    __device__ static inline semaphore &kv_finished(state<config> &s, int stage) {
        return s.semaphores()[NUM_STAGES + stage];
    }
    __device__ static inline semaphore &output_arrived(state<config> &s) {
        return s.semaphores()[NUM_STAGES + NUM_STAGES];
    }

    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            return query;
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            init_semaphore(output_arrived(s), config::NUM_CONSUMER_WARPS);
            for(int i = 0; i < NUM_STAGES; i++) {
                init_semaphore(kv_arrived(s, i), 1);
                init_semaphore(kv_finished(s, i), config::NUM_CONSUMER_WARPS);
            }
            return 2 * NUM_STAGES + 1;
        }
    };

    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            int num_iters = (get_end_pos(s) - get_start_pos(s) + MLA_NUM_ROWS - 1) / MLA_NUM_ROWS;

            s.wait_page_ready(kv_page(s));
            
            for(int iter = 0; iter < num_iters; iter++) {
                if(iter == num_iters-2 || num_iters == 1) {
                    warp::arrive(s.instruction_fetch_ready, config::NUM_CONSUMER_WARPS);
                }
                int stage = iter % NUM_STAGES;
                // Wait for pages to be ready
                
                int pos = get_start_pos(s) + MLA_NUM_ROWS * iter;
                int within_page_idx = (pos % MLA_PAGE_SIZE) / MLA_NUM_ROWS;
                int next_page_id = g.Table[coord<>{get_q_batch_idx(s), pos/MLA_PAGE_SIZE}];
                
                // Load K and V cache using TMA
                auto &krot = get_krot(s, stage);
                auto &v = get_v(s, stage);

                wait(kv_finished(s, stage), 1 ^ ((iter/NUM_STAGES) % 2));
                s.loader_record(LOAD_EVENT);

                warp::tma::expect(kv_arrived(s, stage), krot, v);
                warp::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                    krot, g.Krot, {0, next_page_id, within_page_idx, 0}, kv_arrived(s, stage));
                warp::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                    v, g.V, {0, next_page_id, within_page_idx, 0}, kv_arrived(s, stage));
            }
        }
    };

    struct launcher {
        static __device__ void run(const globals &g, state<config> &s) {
#ifdef KITTENS_BLACKWELL
            if (warp::laneid() == 0) {
                s.wait_tensor_ready();
                arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
            }
#endif
        }
    };

    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
            // Preserve original consumer logic structure

            s.wait_page_ready(q_page(s));
            s.consumer_record(LOAD_EVENT);
            
            // Setup - load Q data (similar to original setup)
            auto qrot_st = get_qrot(s).template subtile<16, MLA_QKRot_D/2>({warpgroup::warpid(), warpgroup::groupid()});
            warp::load_async(qrot_st, g.Qrot, {get_q_batch_idx(s), get_q_seq_idx(s) + warpgroup::warpid(), 0, warpgroup::groupid()});
            
            auto qv_st = get_qv(s).template subtile<16, MLA_QVO_Dd2>({warpgroup::warpid(), warpgroup::groupid()});
            warp::load_async(qv_st, g.Qv, {get_q_batch_idx(s), get_q_seq_idx(s) + warpgroup::warpid(), 0, warpgroup::groupid()});
            
            // Initialize accumulator variables
            col_vec<rt_fl<16, krot_tile::rows>> max_vec, norm_vec;
            rt_fl<16, MLA_QVO_Dd2> o_acc;
            
            int num_iters = (get_end_pos(s) - get_start_pos(s) + MLA_NUM_ROWS - 1) / MLA_NUM_ROWS;
            warp::zero(norm_vec);
            if(num_iters > 0) warp::neg_infty(max_vec);
            else { warp::one(max_vec); warp::mul(max_vec, max_vec, -999999.f); }
            warp::zero(o_acc);
            warp::load_async_wait();
            s.consumer_record(READY_EVENT);

            int seq_length = get_length(s, g);

            s.wait_page_ready(output_page(s));

            auto &q = get_q(s);
            
            // Main compute loop - preserve original attention computation
            for(int iter = 0; iter < num_iters; iter++) {
                const float SOFTMAX_TEMPERATURE = g.Softmax_scale * 1.44269504089f;
                int stage = iter % NUM_STAGES;
                
                col_vec<rt_fl<16, krot_tile::rows>> max_vec_last_scaled, max_vec_scaled;
                
                if(warpgroup::groupid() == 0) {
                    // Wait for input data
                    wait(kv_arrived(s, stage), (iter/NUM_STAGES) % 2); // kv arrived
                    s.consumer_record(COMPUTE_EVENT);
                    
                    // A = Q @ K.T computation (preserve original logic)
                    rt_fl<16, krot_tile::rows> att_block_fp32;
                    auto &k = get_k(s, stage);
                    
                    warpgroup::mm_ABt(att_block_fp32, q, k);
                    
                    warp::mul(max_vec_last_scaled, max_vec, SOFTMAX_TEMPERATURE);
                    
                    warpgroup::mma_async_wait();
                    
                    // Softmax computation (preserve original)
                    bool do_right_fill = (iter >= num_iters-2);
                    if (do_right_fill) {
                        const int length = seq_length - get_start_pos(s) - iter*MLA_NUM_ROWS;
                        warp::apply(att_block_fp32, att_block_fp32, [length]__device__(int row, int col, const float &val) {
                            return col < length ? val : -9999999999.f;
                        });
                    }
                    
                    warp::row_max(max_vec, att_block_fp32, max_vec);
                    warp::mul(max_vec_scaled, max_vec, SOFTMAX_TEMPERATURE);
                    warp::mul(att_block_fp32, att_block_fp32, SOFTMAX_TEMPERATURE);
                    warp::sub_row(att_block_fp32, att_block_fp32, max_vec_scaled);
                    warp::exp2(att_block_fp32, att_block_fp32);
                    
                    warp::sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
                    warp::exp2(max_vec_last_scaled, max_vec_last_scaled);
                    
                    warp::mul(norm_vec, norm_vec, max_vec_last_scaled);
                    warp::row_sum(norm_vec, att_block_fp32, norm_vec);
                    
                    // Store intermediate results to shared memory
                    warpgroup::store(get_max_vec(s), max_vec_last_scaled);
                    warpgroup::store(get_att_block(s, stage), att_block_fp32);
                    group<8>::sync(10);
                }
                else {
                    group<8>::sync(10);
                    warpgroup::load(max_vec_last_scaled, get_max_vec(s));
                }
                
                warp::mul_row(o_acc, o_acc, max_vec_last_scaled);
                
                // O += A @ V
                auto (&v_d2)[2] = reinterpret_cast<vd2_tile(&)[2]>(get_v(s, stage));
                warpgroup::mma_AB(o_acc, get_att_block(s, stage), v_d2[warpgroup::groupid()]);               
                warpgroup::mma_async_wait();
                warp::arrive(kv_finished(s, stage), 1); // we can now release the kv page in any case
            }
            s.warp_finish_page(kv_page(s), 1); // we can now release the kv page in any case
            
            auto dst = get_dst(s);
            if(dst.batch_idx >= 0) s.warp_finish_page(q_page(s), 1); // if we don't need for store, release early!

            if (warpgroup::groupid() == 0) warpgroup::store(get_norm_vec(s), norm_vec);
            group<8>::sync(10);
            if (warpgroup::groupid() == 1) warpgroup::load(norm_vec, get_norm_vec(s));
            warp::div_row(o_acc, o_acc, norm_vec);

            s.consumer_record(STORE_EVENT);
            if(dst.batch_idx >= 0) {
                auto &o_smem = get_o<bf16>(s, warpgroup::groupid(), warpgroup::warpid());
                warp::store(o_smem, o_acc);
            }
            else {
                if(warpgroup::groupid() == 0) {
                    warp::mul(max_vec, max_vec, g.Softmax_scale * 1.44269504089f);
                    warp::log2(norm_vec, norm_vec);
                    warp::add(norm_vec, norm_vec, max_vec); // l_vec = log2(norm_vec) + max_vec
                    warp::store(get_l_vec(s, warpgroup::warpid()), norm_vec);
                }
                auto &o_smem = get_o<float>(s, warpgroup::groupid(), warpgroup::warpid());
                warp::store(o_smem, o_acc);
            }
            warp::sync();
            s.consumer_record(STORE2_EVENT);
            warp::arrive(output_arrived(s), 1); // inputs_finished
        }
    };

    struct storer {
        static __device__ void run(const globals &g, state<config> &s) {
            // Wait for computation to complete
            wait(output_arrived(s), 0); // inputs_finished
            
            // Normalize and store output (preserve original finish logic)
            location dst = get_dst(s);
            
            // Store to appropriate destination based on dst.batch_idx
            s.storer_record(STORE_EVENT);
            if(dst.batch_idx >= 0) {
                int w = warp::laneid();
                if(w < 4) {
                    #pragma unroll
                    for(int wg = 0; wg < 2; wg++) {
                        tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
                            g.O, get_o<bf16>(s, wg, w), {dst.batch_idx, dst.seq_idx + w, 0, wg});
                    }
                }
            }
            else {
                int w = warp::laneid();
                if(w < 4) {
                    // Store Lvec first.
                    tma::store_async<cache_policy::EVICT_LAST>(g.Lvec_scratch, get_l_vec(s, w), {-dst.batch_idx-1, dst.seq_idx+w, 0});
                    // Store O to scratch for reduction
                    #pragma unroll
                    for(int wg = 0; wg < 2; wg++) {
                        tma::store_async<dim::ROW, cache_policy::EVICT_LAST>(
                            g.O_scratch, get_o<float>(s, wg, w), {-dst.batch_idx-1, dst.seq_idx+w, 0, wg});
                    }
                }
            }
            tma::store_async_read_wait();
            s.storer_record(READY_EVENT);
            if(warp::laneid() < 4) {
                if(dst.batch_idx < 0) s.finish_page(q_page(s), config::NUM_CONSUMER_WARPS / 4);
                s.finish_page(output_page(s), config::NUM_CONSUMER_WARPS / 4);
            }
            
            // Signal completion
            if(dst.batch_idx < 0) {
                int w = warp::laneid();
                if(w < 4) {
                    tma::store_async_wait();
                    s.storer_record(STORE2_EVENT);
                    asm volatile("fence.release.gpu;\n");
                    g.completion_flag[{-dst.batch_idx-1, dst.seq_idx+w}] = g.tic;
                }
            }
        }
    };
};