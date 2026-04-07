{# GEMM phase template — handles Redundant (with optional col_batch), and Cooperative modes.
   Variables:
     phase_comment: header comment string
     input_global: e.g. "g.rms_rope_intermediates"
     weight_global: e.g. "g.qkv_weights"
     num_k_iters: K-dimension iteration count
     num_col_tiles: output column tile count
     a_size: A tile bytes (padded)
     b_size: B tile bytes
     stage_size: per-stage shmem bytes
     b_offset: B tile offset within a stage (cooperative only)
     epilogue: rendered epilogue code
     cooperative: bool — true for 128-row mode
     col_batch: number of B tiles loaded per K-loop pass (1 = redundant, >1 = col-distributed)
#}
    // ════ {{ phase_comment }} ════
    {
{%- if cooperative %}
    const int my_row_bid = bid * (PFL_CTA_ROWS / PFL_Q_ROWS) + wid;
    const bool my_row_valid = (abs_q_row + wid * PFL_Q_ROWS) <= (q_start + rel_q_row_last);
    pfl_a_st &my_a_s0 = *reinterpret_cast<pfl_a_st*>(__shm + wid * {{ a_size }});
    pfl_b_st &b_s0 = *reinterpret_cast<pfl_b_st*>(__shm + {{ b_offset }});
    pfl_a_st &my_a_s1 = *reinterpret_cast<pfl_a_st*>(__shm + {{ stage_size }} + wid * {{ a_size }});
    pfl_b_st &b_s1 = *reinterpret_cast<pfl_b_st*>(__shm + {{ stage_size }} + {{ b_offset }});
    pfl_a_st *my_a_stages[2] = {&my_a_s0, &my_a_s1};
    pfl_b_st *b_stages[2] = {&b_s0, &b_s1};
    const int safe_row_bid = my_row_valid ? my_row_bid : 0;
    for (int col = 0; col < {{ num_col_tiles }}; col++) {
        pfl_acc_rt acc;
        warp::zero(acc);
        // Pre-load iter 0
        warp::load_async(*my_a_stages[0], {{ input_global }}, {safe_row_bid, 0});
        group<PFL_NUM_WARPS>::load_async(*b_stages[0], {{ weight_global }}, {layer, col, 0});
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        for (int iter = 0; iter < {{ num_k_iters }}; iter++) {
            int cur = iter % 2;
            if (iter + 1 < {{ num_k_iters }}) {
                int nxt = (iter + 1) % 2;
                warp::load_async(*my_a_stages[nxt], {{ input_global }}, {safe_row_bid, iter + 1});
                group<PFL_NUM_WARPS>::load_async(*b_stages[nxt], {{ weight_global }}, {layer, col, iter + 1});
                asm volatile("cp.async.commit_group;\n" ::: "memory");
                asm volatile("cp.async.wait_group 1;\n" ::: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            }
            pfl_a_st &a_smem = *my_a_stages[cur];
            pfl_b_st &b_smem = *b_stages[cur];
            group<PFL_NUM_WARPS>::sync(1);
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
        }
        if (my_row_valid) {
{{ epilogue }}
        }
    }
{%- elif col_batch > 1 %}
    // ── Col-distributed mode: {{ col_batch }} warps compute different output col tiles ──
    // Shmem: 1 shared A + {{ col_batch }} B tiles per stage, double-buffered.
    constexpr int COL_BATCH = {{ col_batch }};
    pfl_a_st *a_stages[2] = {
        reinterpret_cast<pfl_a_st*>(__shm),
        reinterpret_cast<pfl_a_st*>(__shm + {{ stage_size }})
    };
    // B tiles: b_stages[stage][batch_idx]
    pfl_b_st *b_tiles_s0[COL_BATCH], *b_tiles_s1[COL_BATCH];
    #pragma unroll
    for (int i = 0; i < COL_BATCH; i++) {
        b_tiles_s0[i] = reinterpret_cast<pfl_b_st*>(__shm + {{ a_size }} + i * {{ b_size }});
        b_tiles_s1[i] = reinterpret_cast<pfl_b_st*>(__shm + {{ stage_size }} + {{ a_size }} + i * {{ b_size }});
    }
    // Which B tile does this warp compute? (wraps if num_warps > col_batch)
    const int my_b_idx = wid % COL_BATCH;
    for (int col_base = 0; col_base < {{ num_col_tiles }}; col_base += COL_BATCH) {
        const int cols_this_batch = min(COL_BATCH, {{ num_col_tiles }} - col_base);
        pfl_acc_rt acc;
        warp::zero(acc);
        // Pre-load iter 0 into stage 0
        group<PFL_NUM_WARPS>::load_async(*a_stages[0], {{ input_global }}, {bid, 0});
        #pragma unroll
        for (int i = 0; i < COL_BATCH; i++) {
            int c = col_base + i;
            if (c < {{ num_col_tiles }})
                group<PFL_NUM_WARPS>::load_async(*b_tiles_s0[i], {{ weight_global }}, {layer, c, 0});
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        for (int iter = 0; iter < {{ num_k_iters }}; iter++) {
            int cur = iter % 2;
            // Pre-load next K-iter
            if (iter + 1 < {{ num_k_iters }}) {
                int nxt = (iter + 1) % 2;
                pfl_a_st *a_nxt = a_stages[nxt];
                group<PFL_NUM_WARPS>::load_async(*a_nxt, {{ input_global }}, {bid, iter + 1});
                #pragma unroll
                for (int i = 0; i < COL_BATCH; i++) {
                    int c = col_base + i;
                    pfl_b_st *b_nxt = (cur == 0) ? b_tiles_s1[i] : b_tiles_s0[i];
                    if (c < {{ num_col_tiles }})
                        group<PFL_NUM_WARPS>::load_async(*b_nxt, {{ weight_global }}, {layer, c, iter + 1});
                }
                asm volatile("cp.async.commit_group;\n" ::: "memory");
                asm volatile("cp.async.wait_group 1;\n" ::: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            }
            group<PFL_NUM_WARPS>::sync(1);
            // All warps load shared A
            rt_bf<16, PFL_K_DIM> a_reg;
            warp::load(a_reg, *a_stages[cur]);
            // Each warp reads its own B tile
            pfl_b_st &my_b = *(cur == 0 ? b_tiles_s0[my_b_idx] : b_tiles_s1[my_b_idx]);
            pfl_b_slice_st *b_slices = reinterpret_cast<pfl_b_slice_st*>(&my_b);
            #pragma unroll
            for (int n = 0; n < PFL_N_TILES; n++) {
                rt_bf<16, PFL_K_DIM> b_n; pfl_load_b_slice(b_n, b_slices[n]);
                warp::mma_ABt_base(acc.tiles[0][n], a_reg.tiles[0][0], b_n.tiles[0][0], acc.tiles[0][n]);
                #pragma unroll
                for (int k = 1; k < a_reg.width; k++)
                    warp::mma_ABt_base(acc.tiles[0][n], a_reg.tiles[0][k], b_n.tiles[0][k], acc.tiles[0][n]);
            }
        }
        // Only warps with valid col store. col variable used by epilogue.
        {
            const int col = col_base + my_b_idx;
            if (my_b_idx < cols_this_batch) {
{{ epilogue }}
            }
        }
    }
{%- else %}
    // ── Original redundant mode (col_batch=1) ──
    pfl_a_st &a_s0 = *reinterpret_cast<pfl_a_st*>(__shm);
    pfl_b_st &b_s0 = *reinterpret_cast<pfl_b_st*>(__shm + {{ a_size }});
    pfl_a_st &a_s1 = *reinterpret_cast<pfl_a_st*>(__shm + {{ stage_size }});
    pfl_b_st &b_s1 = *reinterpret_cast<pfl_b_st*>(__shm + {{ stage_size }} + {{ a_size }});
    pfl_a_st *a_stages[2] = {&a_s0, &a_s1};
    pfl_b_st *b_stages[2] = {&b_s0, &b_s1};
    for (int col = 0; col < {{ num_col_tiles }}; col++) {
        pfl_acc_rt acc;
        warp::zero(acc);
        group<PFL_NUM_WARPS>::load_async(*a_stages[0], {{ input_global }}, {bid, 0});
        group<PFL_NUM_WARPS>::load_async(*b_stages[0], {{ weight_global }}, {layer, col, 0});
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        for (int iter = 0; iter < {{ num_k_iters }}; iter++) {
            int cur = iter % 2;
            if (iter + 1 < {{ num_k_iters }}) {
                int nxt = (iter + 1) % 2;
                group<PFL_NUM_WARPS>::load_async(*a_stages[nxt], {{ input_global }}, {bid, iter + 1});
                group<PFL_NUM_WARPS>::load_async(*b_stages[nxt], {{ weight_global }}, {layer, col, iter + 1});
                asm volatile("cp.async.commit_group;\n" ::: "memory");
                asm volatile("cp.async.wait_group 1;\n" ::: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            }
            pfl_a_st &a_smem = *a_stages[cur];
            pfl_b_st &b_smem = *b_stages[cur];
            group<PFL_NUM_WARPS>::sync(1);
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
        }
        if (wid == 0) {
{{ epilogue }}
        }
    }
{%- endif %}
    }
    __syncthreads();
