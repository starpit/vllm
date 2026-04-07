{# Multi-CTA GEMM phase: tiles distributed across all CTAs.
   Variables: same as gemm.cu, but uses row_tiles (from kernel scope) and num_ctas.
   Row variable for epilogues: row_tile (not bid).
#}
    // ════ {{ phase_comment }} ════
    {
    const int col_tiles = {{ num_col_tiles }};
{%- if col_batch > 1 %}
    // ── Multi-CTA col-distributed: {{ col_batch }} warps compute different output cols ──
    constexpr int COL_BATCH = {{ col_batch }};
    const int col_groups = (col_tiles + COL_BATCH - 1) / COL_BATCH;
    const int total_work = row_tiles * col_groups;
    const int my_b_idx = wid % COL_BATCH;

    pfl_a_st *a_stages[2] = {
        reinterpret_cast<pfl_a_st*>(__shm),
        reinterpret_cast<pfl_a_st*>(__shm + {{ stage_size }})
    };
    pfl_b_st *b_tiles_s0[COL_BATCH], *b_tiles_s1[COL_BATCH];
    #pragma unroll
    for (int i = 0; i < COL_BATCH; i++) {
        b_tiles_s0[i] = reinterpret_cast<pfl_b_st*>(__shm + {{ a_size }} + i * {{ b_size }});
        b_tiles_s1[i] = reinterpret_cast<pfl_b_st*>(__shm + {{ stage_size }} + {{ a_size }} + i * {{ b_size }});
    }

    for (int wu = bid; wu < total_work; wu += num_ctas) {
        const int row_tile = wu / col_groups;
        const int col_base = (wu % col_groups) * COL_BATCH;
        const int cols_this_batch = min(COL_BATCH, col_tiles - col_base);

        pfl_acc_rt acc;
        warp::zero(acc);

        // Pre-load iter 0
        group<PFL_NUM_WARPS>::load_async(*a_stages[0], {{ input_global }}, {row_tile, 0});
        #pragma unroll
        for (int i = 0; i < COL_BATCH; i++) {
            if (col_base + i < col_tiles)
                group<PFL_NUM_WARPS>::load_async(*b_tiles_s0[i], {{ weight_global }}, {layer, col_base + i, 0});
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");

        for (int iter = 0; iter < {{ num_k_iters }}; iter++) {
            int cur = iter % 2;
            if (iter + 1 < {{ num_k_iters }}) {
                int nxt = (iter + 1) % 2;
                pfl_a_st &a_nxt = *a_stages[nxt];
                group<PFL_NUM_WARPS>::load_async(a_nxt, {{ input_global }}, {row_tile, iter + 1});
                #pragma unroll
                for (int i = 0; i < COL_BATCH; i++) {
                    pfl_b_st &b_nxt = *(nxt == 0 ? b_tiles_s0[i] : b_tiles_s1[i]);
                    if (col_base + i < col_tiles)
                        group<PFL_NUM_WARPS>::load_async(b_nxt, {{ weight_global }}, {layer, col_base + i, iter + 1});
                }
                asm volatile("cp.async.commit_group;\n" ::: "memory");
                asm volatile("cp.async.wait_group 1;\n" ::: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            }
            group<PFL_NUM_WARPS>::sync(1);

            rt_bf<16, PFL_K_DIM> a_reg;
            warp::load(a_reg, *a_stages[cur]);
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

        if (my_b_idx < cols_this_batch) {
            const int col = col_base + my_b_idx;
{{ epilogue }}
        }
    }
{%- else %}
    // ── Multi-CTA redundant: all warps compute same tile, warp 0 stores ──
    const int total_work = row_tiles * col_tiles;
    pfl_a_st &a_s0 = *reinterpret_cast<pfl_a_st*>(__shm);
    pfl_b_st &b_s0 = *reinterpret_cast<pfl_b_st*>(__shm + {{ a_size }});
    pfl_a_st &a_s1 = *reinterpret_cast<pfl_a_st*>(__shm + {{ stage_size }});
    pfl_b_st &b_s1 = *reinterpret_cast<pfl_b_st*>(__shm + {{ stage_size }} + {{ a_size }});
    pfl_a_st *a_stages[2] = {&a_s0, &a_s1};
    pfl_b_st *b_stages[2] = {&b_s0, &b_s1};

    for (int wu = bid; wu < total_work; wu += num_ctas) {
        const int row_tile = wu / col_tiles;
        const int col = wu % col_tiles;

        pfl_acc_rt acc;
        warp::zero(acc);
        group<PFL_NUM_WARPS>::load_async(*a_stages[0], {{ input_global }}, {row_tile, 0});
        group<PFL_NUM_WARPS>::load_async(*b_stages[0], {{ weight_global }}, {layer, col, 0});
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        for (int iter = 0; iter < {{ num_k_iters }}; iter++) {
            int cur = iter % 2;
            if (iter + 1 < {{ num_k_iters }}) {
                int nxt = (iter + 1) % 2;
                group<PFL_NUM_WARPS>::load_async(*a_stages[nxt], {{ input_global }}, {row_tile, iter + 1});
                group<PFL_NUM_WARPS>::load_async(*b_stages[nxt], {{ weight_global }}, {layer, col, iter + 1});
                asm volatile("cp.async.commit_group;\n" ::: "memory");
                asm volatile("cp.async.wait_group 1;\n" ::: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            }
            group<PFL_NUM_WARPS>::sync(1);
            rt_bf<16, PFL_K_DIM> a_reg;
            warp::load(a_reg, *a_stages[cur]);
            pfl_b_slice_st *b_slices = reinterpret_cast<pfl_b_slice_st*>(b_stages[cur]);
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
