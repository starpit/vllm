#include "../../../include/megakernel.cuh"
#include "mla.cuh"

using namespace kittens;
using namespace megakernel;

template<int Q_HEADS=16>
struct mla_reduction {
    static constexpr int opcode = OPCODE_MLA_Reduction;
    using config = mla_config;
    using globals = mla_globals<Q_HEADS>;

    static constexpr int NUM_STAGES = 6;

    // Preserve original instruction format
    __device__ static inline int get_uid(state<config> &s) { return s.instruction()[1]; }
    __device__ static inline int get_num_iters(state<config> &s) { return s.instruction()[2]; }
    __device__ static inline location get_dst(state<config> &s) { 
        return {s.instruction()[3], s.instruction()[4]}; 
    }
    __device__ static inline int get_load_uid(state<config> &s, int iter) { return s.instruction()[5 + iter]; }

    // Memory layout mapping to pages
    static __device__ inline od8_tile &get_od8_tile(state<config> &s, int stage, int w) {
        return reinterpret_cast<od8_tile*>(s.pages[s.pid(stage/2)].data)[(stage%2)*8 + w];
    }
    static __device__ inline o_tile &get_o_tile(state<config> &s, int stage) {
        return reinterpret_cast<o_tile*>(s.pages[s.pid(stage/2)].data)[stage%2];
    }
    static __device__ inline outd8_tile &get_outd8_tile(state<config> &s, int stage, int w) {
        return reinterpret_cast<outd8_tile*>(&reinterpret_cast<o_tile*>(s.pages[s.pid(stage/2)].data)[stage%2])[w];
    }
    static __device__ inline out_tile &get_out_tile(state<config> &s, int stage) {
        return reinterpret_cast<out_tile*>(&reinterpret_cast<o_tile*>(s.pages[s.pid(stage/2)].data)[stage%2])[0];
    }
    __device__ static inline auto &get_l_vec(state<config> &s, int stage) {
        return *reinterpret_cast<sv_fl<16>*>(reinterpret_cast<uint8_t*>(s.scratch()) + 128*stage);
    }
    __device__ static inline int* get_signal_vec(state<config> &s) {
        return reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(s.scratch()) + 128*NUM_STAGES);
    }

    __device__ static inline semaphore &inputs_arrived(state<config> &s, int stage) {
        return s.semaphores()[stage];
    }
    __device__ static inline semaphore &inputs_finished(state<config> &s, int stage) {
        return s.semaphores()[stage + NUM_STAGES];
    }
    __device__ static inline semaphore &outputs_arrived(state<config> &s) {
        return s.semaphores()[NUM_STAGES * 2];
    }

    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            static constexpr int release_order[3] = {2, 0, 1};
            return release_order[query];
        }
        
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            for(int stage = 0; stage < NUM_STAGES; stage++) {
                init_semaphore(inputs_arrived(s, stage), 1);
                init_semaphore(inputs_finished(s, stage), config::NUM_CONSUMER_WARPS);
            }
            init_semaphore(outputs_arrived(s), config::NUM_CONSUMER_WARPS + NUM_STAGES);
            return 2*NUM_STAGES + 1;
        }
    };

    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            int num_iters = get_num_iters(s);
            location dst = get_dst(s);

            get_signal_vec(s)[laneid()] = 0; // reset signal vector
            group<2>::sync(9);

            // s.wait_page_ready(s.pid(0));
            uint32_t signal_vec_base = static_cast<uint32_t>(__cvta_generic_to_shared(get_signal_vec(s)));
            constexpr int LOADER_TIMING_OFFSET = megakernel::detail::TIMING_EVENT_LOADER_REGION_START;
            if(laneid() < NUM_STAGES) {
                s.wait_page_ready(s.pid(laneid() / 2));
                for(int iter = laneid(); iter < num_iters+1; iter+=NUM_STAGES) {
                    if(iter == num_iters-1) {
                        arrive(s.instruction_fetch_ready, config::NUM_CONSUMER_WARPS);
                    }
                    int stage = iter % NUM_STAGES;
                    int load_uid = get_load_uid(s, iter);
                    if(iter <= 10) s.internal_record(LOADER_TIMING_OFFSET + iter*3, WAIT_EVENT);
                    int signal;
                    while(true) {
                        move<int>::lds(signal, signal_vec_base + iter*4);
                        if(signal) break;
                        __nanosleep(20);
                    }
                    asm volatile("fence.acquire.cta;\n");
                    if(iter <= 10) s.internal_record(LOADER_TIMING_OFFSET + iter*3 + 1, READY_EVENT);
                    wait(inputs_finished(s, stage), 1^((iter/NUM_STAGES)%2));
                    if(iter <= 9) s.internal_record(LOADER_TIMING_OFFSET + iter*3 + 2, LOAD_EVENT);
                    tma::expect_bytes(inputs_arrived(s, stage), (int)(sizeof(od8_tile)*8 + 16*sizeof(float)));
                    tma::load_async(get_l_vec(s, stage), g.Lvec_scratch, {load_uid, dst.seq_idx, 0}, inputs_arrived(s, stage));
                    // Load all 8 output tiles
                    #pragma unroll
                    for(int i = 0; i < 8; i++) {
                        tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                            get_od8_tile(s, stage, i), g.O_scratch, {load_uid, dst.seq_idx, 0, i}, inputs_arrived(s, stage));
                    }
                }
                arrive(outputs_arrived(s));
            }
        }
    };

    struct launcher {
        static __device__ void run(const globals &g, state<config> &s) {
            group<2>::sync(9);
            int num_iters = get_num_iters(s);
            location dst = get_dst(s);
            if(laneid() < num_iters+1) {
                int load_uid = get_load_uid(s, laneid());
                while(*(volatile int*)&g.completion_flag[{load_uid, dst.seq_idx}] != g.tic) { __nanosleep(50); }
                asm volatile("fence.acquire.gpu;\n");
                get_signal_vec(s)[laneid()] = 1;
            }
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
            int num_iters = get_num_iters(s);
            
            int iter = 0;        
            // Load initial accumulator state
            rt_fl<16, MLA_QVO_D / 8> o_acc;
            col_vec<rt_fl<16, krot_tile::rows>> lvec_acc;
            auto &input_o = get_od8_tile(s, iter, warpid());
            auto &input_lvec = get_l_vec(s, iter);
            
            wait(inputs_arrived(s, iter), iter); // load_arrived[src_uid]
            s.consumer_record(LOAD_EVENT);
            warp::load(o_acc, input_o);
            warp::load(lvec_acc, input_lvec);
            warp::sync();
            warp::arrive(inputs_finished(s, iter)); // inputs_finished
            
            // Reduction loop - preserve original reduction logic
            for(iter = 1; iter < num_iters+1; iter++) {
                // Wait for next partial result to arrive
                int stage = iter%NUM_STAGES;
                wait(inputs_arrived(s, stage), (iter/NUM_STAGES)%2); // load_arrived[iter]
                s.consumer_record(COMPUTE_EVENT);
                col_vec<rt_fl<16, krot_tile::rows>> lvec, max_lvec, sum_lvec;
                rt_fl<16, MLA_QVO_D / 8> o;
                
                auto &input_o = get_od8_tile(s, stage, warpid());
                auto &input_lvec = get_l_vec(s, stage);
                
                // reduction computation (preserve original)
                warp::load(lvec, input_lvec);
                warp::max(max_lvec, lvec_acc, lvec);
                warp::sub(lvec_acc, lvec_acc, max_lvec);
                warp::sub(lvec, lvec, max_lvec);
                warp::exp2(lvec_acc, lvec_acc);
                warp::exp2(lvec, lvec);
                warp::add(sum_lvec, lvec_acc, lvec);
                warp::div(lvec_acc, lvec_acc, sum_lvec);
                warp::div(lvec, lvec, sum_lvec);

                warp::load(o, input_o);
                warp::mul_row(o_acc, o_acc, lvec_acc);
                warp::mul_row(o, o, lvec);
                warp::add(o_acc, o_acc, o);
                warp::log2(sum_lvec, sum_lvec);
                warp::add(lvec_acc, sum_lvec, max_lvec);

                warp::sync();
                warp::arrive(inputs_finished(s, stage)); // inputs_finished
            }
            
            // Store final accumulated results back to scratch
            int final_stage = (get_num_iters(s)+1)%NUM_STAGES;
            location dst = get_dst(s);
            group<8>::sync(11);
            s.consumer_record(STORE_EVENT);
            if(dst.batch_idx >= 0) {
                auto &output_o = get_outd8_tile(s, final_stage, warpid());
                warp::store(output_o, o_acc);
            }
            else {
                auto &output_o = get_od8_tile(s, final_stage, warpid());
                warp::store(output_o, o_acc);
                if(warpid() == 0) warp::store(get_l_vec(s, final_stage), lvec_acc);
            }
            warp::sync();
            s.consumer_record(STORE2_EVENT);
            warp::arrive(outputs_arrived(s));
        }
    };

    struct storer {
        static __device__ void run(const globals &g, state<config> &s) {
            
            // Wait for reduction to complete
            wait(outputs_arrived(s), 0); // finish_finished
            warp::sync();
            
            location dst = get_dst(s);
            int final_stage = (get_num_iters(s)+1)%NUM_STAGES;
            auto &o = get_o_tile(s, final_stage);
            auto &lvec = get_l_vec(s, final_stage);
            
            if(dst.batch_idx >= 0) {
                s.storer_record(STORE_EVENT);
                warp::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
                    g.O, get_out_tile(s, final_stage), {dst.batch_idx, dst.seq_idx, 0, 0});
            }
            else {
                s.storer_record(STORE_EVENT);
                warp::tma::store_async<cache_policy::EVICT_LAST>(
                    g.Lvec_scratch, get_l_vec(s, final_stage), {-dst.batch_idx-1, dst.seq_idx, 0});
                #pragma unroll
                for(int i = 0; i < 8; i++) {
                    warp::tma::store_async<dim::ROW, cache_policy::EVICT_LAST>(
                        g.O_scratch, get_od8_tile(s, final_stage, i), 
                        {-dst.batch_idx-1, dst.seq_idx, 0, i});
                }
            }
            tma::store_async_read_wait();
            s.storer_record(READY_EVENT);
            // Release the two pages used.
            s.warp_finish_page(s.pid(0), config::NUM_CONSUMER_WARPS);
            s.warp_finish_page(s.pid(1), config::NUM_CONSUMER_WARPS);
            s.warp_finish_page(s.pid(2), config::NUM_CONSUMER_WARPS);
            
            
            // Signal completion for dependent operations
            if(dst.batch_idx < 0) {
                if(laneid() == 0) {
                    tma::store_async_wait();
                    asm volatile("fence.release.gpu;\n");
                    s.storer_record(STORE2_EVENT);
                    g.completion_flag[{-dst.batch_idx-1, dst.seq_idx}] = g.tic;
                }
            }
        }
    };
};