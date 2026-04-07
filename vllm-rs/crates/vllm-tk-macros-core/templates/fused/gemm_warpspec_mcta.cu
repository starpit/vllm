{# Multi-CTA GEMM with warp-specialized producer/consumer pipeline.
   Warp 0: producer — issues cp.async loads for A (all consumers) + B.
   Warps 1..NUM_WARPS-1: consumers — read shmem, do MMA.

   Synchronization via shmem flags (one per stage):
   - Producer: cp.async → wait_group 0 → threadfence_block → flag[stage] = epoch
   - Consumers: spin on flag[stage] >= epoch → threadfence_block → read shmem
   - Consumers signal done via consumer_done[stage] (producer waits before reuse)

   3-stage pipeline, CTA rows = (NUM_WARPS - 1) * 16.
#}
    // ════ {{ phase_comment }} ════
    {
    const int col_tiles = {{ num_col_tiles }};
    constexpr int NUM_CONSUMERS = PFL_NUM_WARPS - 1;
    constexpr int STAGES = {{ num_stages }};
    const int rows_per_cta = NUM_CONSUMERS * PFL_Q_ROWS;
    const int row_tiles_coop = (q_size + rows_per_cta - 1) / rows_per_cta;
    const int total_work = row_tiles_coop * col_tiles;

    // Shmem layout: [STAGES × stage_size] [STAGES × 2 ints (producer_ready, consumer_done)]
    pfl_b_st *b_stages[STAGES];
    #pragma unroll
    for (int s = 0; s < STAGES; s++) {
        b_stages[s] = reinterpret_cast<pfl_b_st*>(__shm + s * {{ stage_size }} + {{ b_offset }});
    }

    // Flags at end of GEMM region: [producer_ready_0, consumer_done_0, pr_1, cd_1, pr_2, cd_2]
    volatile int *flags = reinterpret_cast<volatile int*>(__shm + STAGES * {{ stage_size }});
    // Initialize flags
    if (threadIdx.x < STAGES * 2) {
        *const_cast<int*>(&flags[threadIdx.x]) = 0;
    }
    __syncthreads();

    const bool is_producer = (wid == 0);
    const int cid = wid - 1;

    pfl_a_st *my_a_stages[STAGES];
    if (!is_producer) {
        #pragma unroll
        for (int s = 0; s < STAGES; s++) {
            my_a_stages[s] = reinterpret_cast<pfl_a_st*>(__shm + s * {{ stage_size }} + cid * {{ a_size }});
        }
    }

    for (int wu = bid; wu < total_work; wu += num_ctas) {
        const int coop_row_tile = wu / col_tiles;
        const int col = wu % col_tiles;

        // Reset flags for this work-unit
        if (threadIdx.x < STAGES * 2) {
            *const_cast<int*>(&flags[threadIdx.x]) = 0;
        }
        __syncthreads();

        if (is_producer) {
            // ── Producer warp ──
            for (int iter = 0; iter < {{ num_k_iters }}; iter++) {
                const int stg = iter % STAGES;

                // Wait for consumers to finish reading this stage (if reusing)
                if (iter >= STAGES) {
                    while (flags[stg * 2 + 1] < (iter - STAGES + 1) * NUM_CONSUMERS) {}
                    __threadfence_block();
                }

                // Load A tiles for each consumer
                for (int c = 0; c < NUM_CONSUMERS; c++) {
                    int row = coop_row_tile * (rows_per_cta / PFL_Q_ROWS) + c;
                    int valid = (coop_row_tile * rows_per_cta + c * PFL_Q_ROWS) < q_size;
                    int safe_row = valid ? row : 0;
                    pfl_a_st *a_dst = reinterpret_cast<pfl_a_st*>(
                        __shm + stg * {{ stage_size }} + c * {{ a_size }});
                    warp::load_async(*a_dst, {{ input_global }}, {safe_row, iter});
                }

                // Load shared B tile
                warp::load_async(*b_stages[stg], {{ weight_global }}, {layer, col, iter});

                asm volatile("cp.async.commit_group;\n" ::: "memory");
                asm volatile("cp.async.wait_group 0;\n" ::: "memory");
                __threadfence_block();

                // Signal: stage is ready (monotonically increasing epoch)
                *const_cast<int*>(&flags[stg * 2]) = iter + 1;
            }

        } else {
            // ── Consumer warp ──
            const int my_row_tile = coop_row_tile * (rows_per_cta / PFL_Q_ROWS) + cid;
            const int my_row_valid = (coop_row_tile * rows_per_cta + cid * PFL_Q_ROWS) < q_size;

            pfl_acc_rt acc;
            warp::zero(acc);

            for (int iter = 0; iter < {{ num_k_iters }}; iter++) {
                const int stg = iter % STAGES;

                // Wait for producer to fill this stage
                while (flags[stg * 2] < iter + 1) {}
                __threadfence_block();

                pfl_a_st &a_smem = *my_a_stages[stg];
                pfl_b_st &b_smem = *b_stages[stg];
                rt_bf<16, PFL_K_DIM> a_reg;
                warp::load(a_reg, a_smem);
                pfl_b_slice_st *b_slices = reinterpret_cast<pfl_b_slice_st*>(&b_smem);
                #pragma unroll
                for (int n = 0; n < PFL_N_TILES; n++) {
                    rt_bf<16, PFL_K_DIM> b_n; pfl_load_b_slice(b_n, b_slices[n]);
                    warp::mma_ABt_base(acc.tiles[0][n], a_reg.tiles[0][0], b_n.tiles[0][0], acc.tiles[0][n]);
                    #pragma unroll
                    for (int k = 1; k < a_reg.width; k++)
                        warp::mma_ABt_base(acc.tiles[0][n], a_reg.tiles[0][k], b_n.tiles[0][k], acc.tiles[0][n]);
                }

                // Signal: done reading this stage (one atomicAdd per consumer)
                if (lid == 0) {
                    atomicAdd(const_cast<int*>(&flags[stg * 2 + 1]), 1);
                }
            }

            if (my_row_valid) {
                const int row_tile = my_row_tile;
{{ epilogue }}
            }
        }
        __syncthreads();  // barrier between work-units
    }
    }
