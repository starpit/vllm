{# Multi-CTA GEMM phase: tiles distributed across all CTAs.
   Variables: same as gemm.cu, but uses row_tiles (from kernel scope) and num_ctas.
   Row variable for epilogues: row_tile (not bid).
   Modes:
     cooperative=true: 128-row CTA, each warp owns 16 rows, B shared. Best compute intensity.
     col_batch>1: col-distributed, each warp computes a different output col tile.
     else: redundant mode, all warps compute same tile, warp 0 stores.
#}
    // ════ {{ phase_comment }} ════
    {
    const int col_tiles = {{ num_col_tiles }};
{%- if per_warp_b %}
    // ── Multi-CTA cooperative, per-warp B (no sync): {{ num_stages }}-stage pipeline ──
    const int rows_per_cta = PFL_CTA_ROWS;
    const int row_tiles_coop = (q_size + rows_per_cta - 1) / rows_per_cta;
    const int total_work = row_tiles_coop * col_tiles;

    constexpr int STAGES = {{ num_stages }};
    constexpr int PER_WARP = {{ a_size }} + {{ b_size }};
    pfl_a_st *my_a_stages[STAGES];
    pfl_b_st *my_b_stages[STAGES];
    #pragma unroll
    for (int s = 0; s < STAGES; s++) {
        int base = s * {{ stage_size }} + wid * PER_WARP;
        my_a_stages[s] = reinterpret_cast<pfl_a_st*>(__shm + base);
        my_b_stages[s] = reinterpret_cast<pfl_b_st*>(__shm + base + {{ b_offset }});
    }

    for (int wu = bid; wu < total_work; wu += num_ctas) {
        const int coop_row_tile = wu / col_tiles;
        const int col = wu % col_tiles;
        const int my_row_tile = coop_row_tile * (rows_per_cta / PFL_GEMM_M) + wid;
        const int my_row_valid = (coop_row_tile * rows_per_cta + wid * PFL_GEMM_M) < q_size;
        const int safe_row_tile = my_row_valid ? my_row_tile : 0;

        pfl_acc_rt acc;
        warp::zero(acc);

        // Prologue: fill stages 0..STAGES-2
        #pragma unroll
        for (int s = 0; s < STAGES - 1 && s < {{ num_k_iters }}; s++) {
            warp::load_async(*my_a_stages[s], {{ input_global }}, {safe_row_tile, s});
            warp::load_async(*my_b_stages[s], {{ weight_global }}, {layer, col, s});
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }

        for (int iter = 0; iter < {{ num_k_iters }}; iter++) {
            int cur = iter % STAGES;
            int prefetch_iter = iter + STAGES - 1;
            if (prefetch_iter < {{ num_k_iters }}) {
                int nxt = prefetch_iter % STAGES;
                warp::load_async(*my_a_stages[nxt], {{ input_global }}, {safe_row_tile, prefetch_iter});
                warp::load_async(*my_b_stages[nxt], {{ weight_global }}, {layer, col, prefetch_iter});
                asm volatile("cp.async.commit_group;\n" ::: "memory");
            }
{%- if num_stages == 3 %}
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");
{%- elif num_stages == 2 %}
            if (prefetch_iter < {{ num_k_iters }}) {
                asm volatile("cp.async.wait_group 1;\n" ::: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            }
{%- else %}
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
{%- endif %}
            // No group::sync needed — each warp owns its own A and B tiles
            pfl_a_st &a_smem = *my_a_stages[cur];
            pfl_b_st &b_smem = *my_b_stages[cur];
            pfl_a_rt a_reg;
            warp::load(a_reg, a_smem);
            pfl_b_slice_st *b_slices = reinterpret_cast<pfl_b_slice_st*>(&b_smem);
            #pragma unroll
            for (int n = 0; n < PFL_N_TILES; n++) {
                rt_bf<16, PFL_K_DIM> b_n; pfl_load_b_slice(b_n, b_slices[n]);
                // Reuse each loaded b_n across all M sub-tiles — this is the
                // compute-density win: PFL_GEMM_M_SUBS MMA chains per shmem B-load.
                #pragma unroll
                for (int m_sub = 0; m_sub < PFL_GEMM_M_SUBS; m_sub++) {
                    warp::mma_ABt_base(acc.tiles[m_sub][n], a_reg.tiles[m_sub][0], b_n.tiles[0][0], acc.tiles[m_sub][n]);
                    #pragma unroll
                    for (int k = 1; k < a_reg.width; k++)
                        warp::mma_ABt_base(acc.tiles[m_sub][n], a_reg.tiles[m_sub][k], b_n.tiles[0][k], acc.tiles[m_sub][n]);
                }
            }
        }
        if (my_row_valid) {
            const int row_tile = my_row_tile;
{{ epilogue }}
        }
    }
{%- elif cooperative %}
    // ── Multi-CTA cooperative: {{ num_stages }}-stage pipeline ──
    const int rows_per_cta = PFL_CTA_ROWS;
    const int row_tiles_coop = (q_size + rows_per_cta - 1) / rows_per_cta;
    const int total_work = row_tiles_coop * col_tiles;

    constexpr int STAGES = {{ num_stages }};
    pfl_a_st *my_a_stages[STAGES];
    pfl_b_st *b_stages[STAGES];
    #pragma unroll
    for (int s = 0; s < STAGES; s++) {
        my_a_stages[s] = reinterpret_cast<pfl_a_st*>(__shm + s * {{ stage_size }} + wid * {{ a_size }});
        b_stages[s] = reinterpret_cast<pfl_b_st*>(__shm + s * {{ stage_size }} + {{ b_offset }});
    }

    for (int wu = bid; wu < total_work; wu += num_ctas) {
        const int coop_row_tile = wu / col_tiles;
        const int col = wu % col_tiles;
        const int my_row_tile = coop_row_tile * (rows_per_cta / PFL_GEMM_M) + wid;
        const int my_row_valid = (coop_row_tile * rows_per_cta + wid * PFL_GEMM_M) < q_size;
        const int safe_row_tile = my_row_valid ? my_row_tile : 0;

        pfl_acc_rt acc;
        warp::zero(acc);

        // Prologue: fill stages 0..STAGES-2
        #pragma unroll
        for (int s = 0; s < STAGES - 1 && s < {{ num_k_iters }}; s++) {
            warp::load_async(*my_a_stages[s], {{ input_global }}, {safe_row_tile, s});
            group<PFL_NUM_WARPS>::load_async(*b_stages[s], {{ weight_global }}, {layer, col, s});
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }

        for (int iter = 0; iter < {{ num_k_iters }}; iter++) {
            int cur = iter % STAGES;
            int prefetch_iter = iter + STAGES - 1;
            if (prefetch_iter < {{ num_k_iters }}) {
                int nxt = prefetch_iter % STAGES;
                warp::load_async(*my_a_stages[nxt], {{ input_global }}, {safe_row_tile, prefetch_iter});
                group<PFL_NUM_WARPS>::load_async(*b_stages[nxt], {{ weight_global }}, {layer, col, prefetch_iter});
                asm volatile("cp.async.commit_group;\n" ::: "memory");
            }
            // Wait until current stage is ready: at most STAGES-2 groups in-flight
{%- if num_stages == 3 %}
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");
{%- elif num_stages == 2 %}
            if (prefetch_iter < {{ num_k_iters }}) {
                asm volatile("cp.async.wait_group 1;\n" ::: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            }
{%- else %}
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
{%- endif %}
            pfl_a_st &a_smem = *my_a_stages[cur];
            pfl_b_st &b_smem = *b_stages[cur];
            group<PFL_NUM_WARPS>::sync(1);
{%- if kstripe %}
            // ── CUTLASS-style K-stripe inner loop with double-buffering ──
            // Two register fragments per warp held simultaneously: one being
            // computed against (cur), one being prefetched from shmem (nxt).
            // The compiler can overlap `ldsm` for the prefetch with `mma` on
            // the current fragment, exposing instruction-level parallelism
            // that single-buffered loops miss. This is the trick CUTLASS uses
            // in mma_multistage.h to hit ~70% of peak.
            constexpr int K_STRIPES = PFL_K_DIM / 16;
            rt_bf<PFL_GEMM_M, 16> a_buf[2];
            rt_bf<16, 16> b_buf[PFL_N_TILES][2];

            // Prologue: prime stripe 0 (a + all n tiles of b).
            {
                auto a_view = a_smem.template subtile<PFL_GEMM_M, 16>(int2{0, 0});
                warp::load(a_buf[0], a_view);
                #pragma unroll
                for (int n = 0; n < PFL_N_TILES; n++) {
                    auto b_view = b_smem.template subtile<16, 16>(int2{n, 0});
                    warp::load(b_buf[n][0], b_view);
                }
            }

            #pragma unroll
            for (int kt = 0; kt < K_STRIPES; kt++) {
                const int cur = kt & 1;
                const int nxt = (kt + 1) & 1;
                // Prefetch next K-stripe (A + all N B-tiles) while we still
                // have the current stripe in regs. Compiler should issue these
                // ldsm in parallel with the upcoming mma instructions.
                if (kt + 1 < K_STRIPES) {
                    auto a_view = a_smem.template subtile<PFL_GEMM_M, 16>(int2{0, kt + 1});
                    warp::load(a_buf[nxt], a_view);
                    #pragma unroll
                    for (int n = 0; n < PFL_N_TILES; n++) {
                        auto b_view = b_smem.template subtile<16, 16>(int2{n, kt + 1});
                        warp::load(b_buf[n][nxt], b_view);
                    }
                }
                // Compute on current stripe.
                #pragma unroll
                for (int n = 0; n < PFL_N_TILES; n++) {
                    #pragma unroll
                    for (int m_sub = 0; m_sub < PFL_GEMM_M_SUBS; m_sub++) {
                        warp::mma_ABt_base(acc.tiles[m_sub][n], a_buf[cur].tiles[m_sub][0], b_buf[n][cur].tiles[0][0], acc.tiles[m_sub][n]);
                    }
                }
            }
{%- else %}
            pfl_a_rt a_reg;
            warp::load(a_reg, a_smem);
            pfl_b_slice_st *b_slices = reinterpret_cast<pfl_b_slice_st*>(&b_smem);
            #pragma unroll
            for (int n = 0; n < PFL_N_TILES; n++) {
                rt_bf<16, PFL_K_DIM> b_n; pfl_load_b_slice(b_n, b_slices[n]);
                // Reuse each loaded b_n across all M sub-tiles — this is the
                // compute-density win: PFL_GEMM_M_SUBS MMA chains per shmem B-load.
                #pragma unroll
                for (int m_sub = 0; m_sub < PFL_GEMM_M_SUBS; m_sub++) {
                    warp::mma_ABt_base(acc.tiles[m_sub][n], a_reg.tiles[m_sub][0], b_n.tiles[0][0], acc.tiles[m_sub][n]);
                    #pragma unroll
                    for (int k = 1; k < a_reg.width; k++)
                        warp::mma_ABt_base(acc.tiles[m_sub][n], a_reg.tiles[m_sub][k], b_n.tiles[0][k], acc.tiles[m_sub][n]);
                }
            }
{%- endif %}
        }
        if (my_row_valid) {
            const int row_tile = my_row_tile;
{{ epilogue }}
        }
    }
{%- elif col_batch > 1 %}
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

            pfl_a_rt a_reg;
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
            pfl_a_rt a_reg;
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
