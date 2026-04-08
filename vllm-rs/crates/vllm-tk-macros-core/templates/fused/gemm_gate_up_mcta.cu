{# Fused gate+up GEMM: two GEMMs sharing the same A input, no barrier between.
   Gate: input × gate_weights → SiLU → silu_out
   Up:   input × up_weights → mulgate(silu_out) → silu_out
   Same work distribution ensures each CTA reads its own gate output.

   Variables:
     phase_comment: header comment
     input_global: shared A input (e.g. "g.rms_gate_intermediates")
     gate_weight_global, up_weight_global: B weights
     output_global: silu_out for both
     num_k_iters: K-dimension iterations
     num_col_tiles: output col tiles (same for both GEMMs)
     a_size, b_size, stage_size, b_offset: shmem layout
     cooperative: bool
     num_stages: pipeline stages (2 or 3)
#}
    // ════ {{ phase_comment }} ════
    // Fused gate+up: two GEMMs, same A input, no barrier between.
    {
    const int col_tiles = {{ num_col_tiles }};
{%- if per_warp_b %}
    // ── Cooperative fused gate+up, per-warp B (no sync), {{ num_stages }}-stage pipeline ──
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
        const int row_tile = my_row_tile;

        // ── Gate GEMM: input × gate_weights → SiLU → silu_out ──
        {
            pfl_acc_rt acc;
            warp::zero(acc);
            #pragma unroll
            for (int s = 0; s < STAGES - 1 && s < {{ num_k_iters }}; s++) {
                warp::load_async(*my_a_stages[s], {{ input_global }}, {safe_row_tile, s});
                warp::load_async(*my_b_stages[s], {{ gate_weight_global }}, {layer, col, s});
                asm volatile("cp.async.commit_group;\n" ::: "memory");
            }
            for (int iter = 0; iter < {{ num_k_iters }}; iter++) {
                int cur = iter % STAGES;
                int prefetch_iter = iter + STAGES - 1;
                if (prefetch_iter < {{ num_k_iters }}) {
                    int nxt = prefetch_iter % STAGES;
                    warp::load_async(*my_a_stages[nxt], {{ input_global }}, {safe_row_tile, prefetch_iter});
                    warp::load_async(*my_b_stages[nxt], {{ gate_weight_global }}, {layer, col, prefetch_iter});
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
                pfl_a_st &a_smem = *my_a_stages[cur];
                pfl_b_st &b_smem = *my_b_stages[cur];
                pfl_a_rt a_reg;
                warp::load(a_reg, a_smem);
                pfl_b_slice_st *b_slices = reinterpret_cast<pfl_b_slice_st*>(&b_smem);
                #pragma unroll
                for (int n = 0; n < PFL_N_TILES; n++) {
                    rt_bf<16, PFL_K_DIM> b_n; pfl_load_b_slice(b_n, b_slices[n]);
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
                rt_bf<PFL_GEMM_M, PFL_OUT_BLOCK> out_bf;
                #pragma unroll
                for (int i = 0; i < acc.height; i++)
                    #pragma unroll
                    for (int j = 0; j < acc.width; j++)
                        #pragma unroll
                        // NOTE: per-thread data[] has packed_per_thread (=4) entries,
                        // NOT num_elements (=256 = whole-subtile count). See the
                        // dual_accum epilogue comment above for the full story.
                        for (int d = 0; d < acc.tiles[0][0].packed_per_thread; d++) {
                            float2 &v = acc.tiles[i][j].data[d];
                            v.x = v.x / (1.f + expf(-v.x));
                            v.y = v.y / (1.f + expf(-v.y));
                        }
                warp::copy(out_bf, acc);
                warp::store({{ output_global }}, out_bf, {row_tile, col});
            }
        }

        // ── Up GEMM: input × up_weights → mulgate(silu_out) → silu_out ──
        {
            pfl_acc_rt acc;
            warp::zero(acc);
            #pragma unroll
            for (int s = 0; s < STAGES - 1 && s < {{ num_k_iters }}; s++) {
                warp::load_async(*my_a_stages[s], {{ input_global }}, {safe_row_tile, s});
                warp::load_async(*my_b_stages[s], {{ up_weight_global }}, {layer, col, s});
                asm volatile("cp.async.commit_group;\n" ::: "memory");
            }
            for (int iter = 0; iter < {{ num_k_iters }}; iter++) {
                int cur = iter % STAGES;
                int prefetch_iter = iter + STAGES - 1;
                if (prefetch_iter < {{ num_k_iters }}) {
                    int nxt = prefetch_iter % STAGES;
                    warp::load_async(*my_a_stages[nxt], {{ input_global }}, {safe_row_tile, prefetch_iter});
                    warp::load_async(*my_b_stages[nxt], {{ up_weight_global }}, {layer, col, prefetch_iter});
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
                pfl_a_st &a_smem = *my_a_stages[cur];
                pfl_b_st &b_smem = *my_b_stages[cur];
                pfl_a_rt a_reg;
                warp::load(a_reg, a_smem);
                pfl_b_slice_st *b_slices = reinterpret_cast<pfl_b_slice_st*>(&b_smem);
                #pragma unroll
                for (int n = 0; n < PFL_N_TILES; n++) {
                    rt_bf<16, PFL_K_DIM> b_n; pfl_load_b_slice(b_n, b_slices[n]);
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
                rt_bf<PFL_GEMM_M, PFL_OUT_BLOCK> acc_bf;
                warp::copy(acc_bf, acc);
                rt_bf<PFL_GEMM_M, PFL_OUT_BLOCK> gate_bf;
                warp::load(gate_bf, {{ output_global }}, {row_tile, col});
                #pragma unroll
                for (int r = 0; r < acc_bf.height; r++)
                    #pragma unroll
                    for (int c = 0; c < acc_bf.width; c++)
                        #pragma unroll
                        for (int k = 0; k < acc_bf.tiles[0][0].packed_per_thread; k++) {
                            bf16_2 &a = acc_bf.tiles[r][c].data[k];
                            bf16_2 &gv = gate_bf.tiles[r][c].data[k];
                            float a_lo = __bfloat162float(__low2bfloat16(a));
                            float a_hi = __bfloat162float(__high2bfloat16(a));
                            float g_lo = __bfloat162float(__low2bfloat16(gv));
                            float g_hi = __bfloat162float(__high2bfloat16(gv));
                            a = __floats2bfloat162_rn(a_lo * g_lo, a_hi * g_hi);
                        }
                warp::store({{ output_global }}, acc_bf, {row_tile, col});
            }
        }
    }
{%- elif col_fixed %}
    // ── Dual-accumulator fused gate+up + col-fixed scheduling, {{ num_stages }}-stage ──
    // Each CTA owns a contiguous block of col tiles and iterates over row_tiles
    // WITHIN each col, so the gate/up B weight for a col stays L2-resident across
    // the CTA's row_tile iterations. Reduces B GMEM traffic by up to row_tiles×
    // at the cost of some load imbalance when col_tiles % num_ctas != 0.
    const int rows_per_cta = PFL_CTA_ROWS;
    const int row_tiles_coop = (q_size + rows_per_cta - 1) / rows_per_cta;

    constexpr int STAGES = {{ num_stages }};
    pfl_a_st *my_a_stages[STAGES];
    pfl_b_st *bg_stages[STAGES];
    pfl_b_st *bu_stages[STAGES];
    #pragma unroll
    for (int s = 0; s < STAGES; s++) {
        my_a_stages[s] = reinterpret_cast<pfl_a_st*>(__shm + s * {{ stage_size }} + wid * {{ a_size }});
        bg_stages[s] = reinterpret_cast<pfl_b_st*>(__shm + s * {{ stage_size }} + {{ b_offset }});
        bu_stages[s] = reinterpret_cast<pfl_b_st*>(__shm + s * {{ stage_size }} + {{ up_b_offset }});
    }

    // Compute this CTA's [col_begin, col_end) range. CTAs 0..(col_tiles%num_ctas)-1
    // get one extra col each; the rest get the floor. Max imbalance = 1 col.
    const int cols_base = col_tiles / num_ctas;
    const int cols_extra = col_tiles - cols_base * num_ctas;
    int col_begin, col_end;
    if (bid < cols_extra) {
        col_begin = bid * (cols_base + 1);
        col_end = col_begin + cols_base + 1;
    } else {
        col_begin = cols_extra * (cols_base + 1) + (bid - cols_extra) * cols_base;
        col_end = col_begin + cols_base;
    }

    for (int col = col_begin; col < col_end; col++) {
        for (int coop_row_tile = 0; coop_row_tile < row_tiles_coop; coop_row_tile++) {
            const int my_row_tile = coop_row_tile * (rows_per_cta / PFL_GEMM_M) + wid;
            const int my_row_valid = (coop_row_tile * rows_per_cta + wid * PFL_GEMM_M) < q_size;
            const int safe_row_tile = my_row_valid ? my_row_tile : 0;
            const int row_tile = my_row_tile;

            pfl_acc_rt gate_acc;
            pfl_acc_rt up_acc;
            warp::zero(gate_acc);
            warp::zero(up_acc);

            #pragma unroll
            for (int s = 0; s < STAGES - 1 && s < {{ num_k_iters }}; s++) {
                warp::load_async(*my_a_stages[s], {{ input_global }}, {safe_row_tile, s});
                group<PFL_NUM_WARPS>::load_async(*bg_stages[s], {{ gate_weight_global }}, {layer, col, s});
                group<PFL_NUM_WARPS>::load_async(*bu_stages[s], {{ up_weight_global }}, {layer, col, s});
                asm volatile("cp.async.commit_group;\n" ::: "memory");
            }

            for (int iter = 0; iter < {{ num_k_iters }}; iter++) {
                int cur = iter % STAGES;
                int prefetch_iter = iter + STAGES - 1;
                if (prefetch_iter < {{ num_k_iters }}) {
                    int nxt = prefetch_iter % STAGES;
                    warp::load_async(*my_a_stages[nxt], {{ input_global }}, {safe_row_tile, prefetch_iter});
                    group<PFL_NUM_WARPS>::load_async(*bg_stages[nxt], {{ gate_weight_global }}, {layer, col, prefetch_iter});
                    group<PFL_NUM_WARPS>::load_async(*bu_stages[nxt], {{ up_weight_global }}, {layer, col, prefetch_iter});
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
                pfl_a_st &a_smem = *my_a_stages[cur];
                pfl_b_st &bg_smem = *bg_stages[cur];
                pfl_b_st &bu_smem = *bu_stages[cur];
                group<PFL_NUM_WARPS>::sync(1);
                pfl_a_rt a_reg;
                warp::load(a_reg, a_smem);
                pfl_b_slice_st *bg_slices = reinterpret_cast<pfl_b_slice_st*>(&bg_smem);
                pfl_b_slice_st *bu_slices = reinterpret_cast<pfl_b_slice_st*>(&bu_smem);
                #pragma unroll
                for (int n = 0; n < PFL_N_TILES; n++) {
                    rt_bf<16, PFL_K_DIM> bg_n; pfl_load_b_slice(bg_n, bg_slices[n]);
                    rt_bf<16, PFL_K_DIM> bu_n; pfl_load_b_slice(bu_n, bu_slices[n]);
                    #pragma unroll
                    for (int m_sub = 0; m_sub < PFL_GEMM_M_SUBS; m_sub++) {
                        #pragma unroll
                        for (int k = 0; k < a_reg.width; k++) {
                            warp::mma_ABt_base(gate_acc.tiles[m_sub][n], a_reg.tiles[m_sub][k], bg_n.tiles[0][k], gate_acc.tiles[m_sub][n]);
                            warp::mma_ABt_base(up_acc.tiles[m_sub][n],   a_reg.tiles[m_sub][k], bu_n.tiles[0][k], up_acc.tiles[m_sub][n]);
                        }
                    }
                }
            }

            if (my_row_valid) {
                rt_bf<PFL_GEMM_M, PFL_OUT_BLOCK> out_bf;
                #pragma unroll
                for (int i = 0; i < gate_acc.height; i++)
                    #pragma unroll
                    for (int j = 0; j < gate_acc.width; j++)
                        #pragma unroll
                        for (int d = 0; d < gate_acc.tiles[0][0].packed_per_thread; d++) {
                            float2 &v = gate_acc.tiles[i][j].data[d];
                            v.x = v.x / (1.f + expf(-v.x));
                            v.y = v.y / (1.f + expf(-v.y));
                        }
                #pragma unroll
                for (int i = 0; i < gate_acc.height; i++)
                    #pragma unroll
                    for (int j = 0; j < gate_acc.width; j++)
                        #pragma unroll
                        for (int d = 0; d < gate_acc.tiles[0][0].packed_per_thread; d++) {
                            float2 &gv = gate_acc.tiles[i][j].data[d];
                            float2 &uv = up_acc.tiles[i][j].data[d];
                            gv.x *= uv.x;
                            gv.y *= uv.y;
                        }
                warp::copy(out_bf, gate_acc);
                warp::store({{ output_global }}, out_bf, {row_tile, col});
            }
        }
    }
{%- elif dual_accum %}
    // ── Dual-accumulator fused gate+up (A reuse), {{ num_stages }}-stage ──
    // Single K-loop. Each iteration loads A once and issues MMAs into BOTH
    // gate_acc and up_acc. A shmem tile is reused across the two GEMMs;
    // only B_gate and B_up are reloaded per stage. Halves A traffic vs. the
    // back-to-back implementation at the cost of 2 live accumulators (~2x
    // register pressure) and 2x gate_up B shmem footprint.
    const int rows_per_cta = PFL_CTA_ROWS;
    const int row_tiles_coop = (q_size + rows_per_cta - 1) / rows_per_cta;
    const int total_work = row_tiles_coop * col_tiles;

    constexpr int STAGES = {{ num_stages }};
    pfl_a_st *my_a_stages[STAGES];
    pfl_b_st *bg_stages[STAGES];
    pfl_b_st *bu_stages[STAGES];
    #pragma unroll
    for (int s = 0; s < STAGES; s++) {
        my_a_stages[s] = reinterpret_cast<pfl_a_st*>(__shm + s * {{ stage_size }} + wid * {{ a_size }});
        bg_stages[s] = reinterpret_cast<pfl_b_st*>(__shm + s * {{ stage_size }} + {{ b_offset }});
        bu_stages[s] = reinterpret_cast<pfl_b_st*>(__shm + s * {{ stage_size }} + {{ up_b_offset }});
    }

    for (int wu = bid; wu < total_work; wu += num_ctas) {
        const int coop_row_tile = wu / col_tiles;
        const int col = wu % col_tiles;
        const int my_row_tile = coop_row_tile * (rows_per_cta / PFL_GEMM_M) + wid;
        const int my_row_valid = (coop_row_tile * rows_per_cta + wid * PFL_GEMM_M) < q_size;
        const int safe_row_tile = my_row_valid ? my_row_tile : 0;
        const int row_tile = my_row_tile;

        pfl_acc_rt gate_acc;
        pfl_acc_rt up_acc;
        warp::zero(gate_acc);
        warp::zero(up_acc);

        // Prologue: fill stages 0..STAGES-2 with A, B_gate, B_up.
        #pragma unroll
        for (int s = 0; s < STAGES - 1 && s < {{ num_k_iters }}; s++) {
            warp::load_async(*my_a_stages[s], {{ input_global }}, {safe_row_tile, s});
            group<PFL_NUM_WARPS>::load_async(*bg_stages[s], {{ gate_weight_global }}, {layer, col, s});
            group<PFL_NUM_WARPS>::load_async(*bu_stages[s], {{ up_weight_global }}, {layer, col, s});
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }

        // Single K-loop: load once, compute gate + up per iteration.
        for (int iter = 0; iter < {{ num_k_iters }}; iter++) {
            int cur = iter % STAGES;
            int prefetch_iter = iter + STAGES - 1;
            if (prefetch_iter < {{ num_k_iters }}) {
                int nxt = prefetch_iter % STAGES;
                warp::load_async(*my_a_stages[nxt], {{ input_global }}, {safe_row_tile, prefetch_iter});
                group<PFL_NUM_WARPS>::load_async(*bg_stages[nxt], {{ gate_weight_global }}, {layer, col, prefetch_iter});
                group<PFL_NUM_WARPS>::load_async(*bu_stages[nxt], {{ up_weight_global }}, {layer, col, prefetch_iter});
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
            pfl_a_st &a_smem = *my_a_stages[cur];
            pfl_b_st &bg_smem = *bg_stages[cur];
            pfl_b_st &bu_smem = *bu_stages[cur];
            group<PFL_NUM_WARPS>::sync(1);
{%- if kstripe %}
            // Dual-accum + double-buffered K-stripe inner loop. Two register
            // ring buffers per warp: one being computed against (cur), one
            // being prefetched (nxt). Compiler can interleave ldsm with mma
            // for instruction-level parallelism.
            constexpr int K_STRIPES_DA = PFL_K_DIM / 16;
            rt_bf<PFL_GEMM_M, 16> a_buf[2];
            rt_bf<16, 16> bg_buf[PFL_N_TILES][2];
            rt_bf<16, 16> bu_buf[PFL_N_TILES][2];

            {
                auto a_view = a_smem.template subtile<PFL_GEMM_M, 16>(int2{0, 0});
                warp::load(a_buf[0], a_view);
                #pragma unroll
                for (int n = 0; n < PFL_N_TILES; n++) {
                    auto bg_view = bg_smem.template subtile<16, 16>(int2{n, 0});
                    auto bu_view = bu_smem.template subtile<16, 16>(int2{n, 0});
                    warp::load(bg_buf[n][0], bg_view);
                    warp::load(bu_buf[n][0], bu_view);
                }
            }

            #pragma unroll
            for (int kt = 0; kt < K_STRIPES_DA; kt++) {
                const int cur = kt & 1;
                const int nxt = (kt + 1) & 1;
                if (kt + 1 < K_STRIPES_DA) {
                    auto a_view = a_smem.template subtile<PFL_GEMM_M, 16>(int2{0, kt + 1});
                    warp::load(a_buf[nxt], a_view);
                    #pragma unroll
                    for (int n = 0; n < PFL_N_TILES; n++) {
                        auto bg_view = bg_smem.template subtile<16, 16>(int2{n, kt + 1});
                        auto bu_view = bu_smem.template subtile<16, 16>(int2{n, kt + 1});
                        warp::load(bg_buf[n][nxt], bg_view);
                        warp::load(bu_buf[n][nxt], bu_view);
                    }
                }
                #pragma unroll
                for (int n = 0; n < PFL_N_TILES; n++) {
                    #pragma unroll
                    for (int m_sub = 0; m_sub < PFL_GEMM_M_SUBS; m_sub++) {
                        warp::mma_ABt_base(gate_acc.tiles[m_sub][n], a_buf[cur].tiles[m_sub][0], bg_buf[n][cur].tiles[0][0], gate_acc.tiles[m_sub][n]);
                        warp::mma_ABt_base(up_acc.tiles[m_sub][n],   a_buf[cur].tiles[m_sub][0], bu_buf[n][cur].tiles[0][0], up_acc.tiles[m_sub][n]);
                    }
                }
            }
{%- else %}
            pfl_a_rt a_reg;
            warp::load(a_reg, a_smem);
            pfl_b_slice_st *bg_slices = reinterpret_cast<pfl_b_slice_st*>(&bg_smem);
            pfl_b_slice_st *bu_slices = reinterpret_cast<pfl_b_slice_st*>(&bu_smem);
            // Dual-accumulator inner loop: each b_n load is reused across
            // both gate and up accumulators for this (m_sub, n) position.
            // A register is loaded ONCE per K-iter and applied to both.
            #pragma unroll
            for (int n = 0; n < PFL_N_TILES; n++) {
                rt_bf<16, PFL_K_DIM> bg_n; pfl_load_b_slice(bg_n, bg_slices[n]);
                rt_bf<16, PFL_K_DIM> bu_n; pfl_load_b_slice(bu_n, bu_slices[n]);
                #pragma unroll
                for (int m_sub = 0; m_sub < PFL_GEMM_M_SUBS; m_sub++) {
                    #pragma unroll
                    for (int k = 0; k < a_reg.width; k++) {
                        warp::mma_ABt_base(gate_acc.tiles[m_sub][n], a_reg.tiles[m_sub][k], bg_n.tiles[0][k], gate_acc.tiles[m_sub][n]);
                        warp::mma_ABt_base(up_acc.tiles[m_sub][n],   a_reg.tiles[m_sub][k], bu_n.tiles[0][k], up_acc.tiles[m_sub][n]);
                    }
                }
            }
{%- endif %}
        }

        // Epilogue: silu(gate_acc) * up_acc → silu_out.
        //
        // IMPORTANT: iterate `packed_per_thread` (= 4 for rt_fl sub-tiles),
        // NOT `num_elements` (= 256 = rows*cols per sub-tile). Each per-thread
        // data[] array only has packed_per_thread entries; iterating up to
        // num_elements reads/writes 252 out-of-bounds slots that happen to be
        // other registers in the warp. In the single-accumulator silu loop
        // this is silently tolerated (the compiler folds OOB writes to
        // unused regs), but with two live accumulators the OOB writes on
        // gate_acc cross-contaminate up_acc's valid registers and produce
        // garbage output.
        if (my_row_valid) {
            rt_bf<PFL_GEMM_M, PFL_OUT_BLOCK> out_bf;
            // SiLU on gate_acc in-place (fp32 accumulator, pre-silu).
            #pragma unroll
            for (int i = 0; i < gate_acc.height; i++)
                #pragma unroll
                for (int j = 0; j < gate_acc.width; j++)
                    #pragma unroll
                    for (int d = 0; d < gate_acc.tiles[0][0].packed_per_thread; d++) {
                        float2 &v = gate_acc.tiles[i][j].data[d];
                        v.x = v.x / (1.f + expf(-v.x));
                        v.y = v.y / (1.f + expf(-v.y));
                    }
            // Elementwise multiply: gate_acc (now silu-gate) *= up_acc.
            // Local names carefully avoid shadowing the outer `g` (globals).
            #pragma unroll
            for (int i = 0; i < gate_acc.height; i++)
                #pragma unroll
                for (int j = 0; j < gate_acc.width; j++)
                    #pragma unroll
                    for (int d = 0; d < gate_acc.tiles[0][0].packed_per_thread; d++) {
                        float2 &gv = gate_acc.tiles[i][j].data[d];
                        float2 &uv = up_acc.tiles[i][j].data[d];
                        gv.x *= uv.x;
                        gv.y *= uv.y;
                    }
            warp::copy(out_bf, gate_acc);
            warp::store({{ output_global }}, out_bf, {row_tile, col});
        }
    }
{%- elif cooperative %}
    // ── Cooperative fused gate+up, {{ num_stages }}-stage pipeline ──
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
        const int row_tile = my_row_tile;

        // ── Gate GEMM: input × gate_weights → SiLU → silu_out ──
        {
            pfl_acc_rt acc;
            warp::zero(acc);
            // Prologue: fill stages 0..STAGES-2
            #pragma unroll
            for (int s = 0; s < STAGES - 1 && s < {{ num_k_iters }}; s++) {
                warp::load_async(*my_a_stages[s], {{ input_global }}, {safe_row_tile, s});
                group<PFL_NUM_WARPS>::load_async(*b_stages[s], {{ gate_weight_global }}, {layer, col, s});
                asm volatile("cp.async.commit_group;\n" ::: "memory");
            }
            for (int iter = 0; iter < {{ num_k_iters }}; iter++) {
                int cur = iter % STAGES;
                int prefetch_iter = iter + STAGES - 1;
                if (prefetch_iter < {{ num_k_iters }}) {
                    int nxt = prefetch_iter % STAGES;
                    warp::load_async(*my_a_stages[nxt], {{ input_global }}, {safe_row_tile, prefetch_iter});
                    group<PFL_NUM_WARPS>::load_async(*b_stages[nxt], {{ gate_weight_global }}, {layer, col, prefetch_iter});
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
                pfl_a_st &a_smem = *my_a_stages[cur];
                pfl_b_st &b_smem = *b_stages[cur];
                group<PFL_NUM_WARPS>::sync(1);
{%- if kstripe %}
                // K-stripe inner loop (CUTLASS-style register lifetime)
                constexpr int K_STRIPES_GU = PFL_K_DIM / 16;
                #pragma unroll
                for (int kt = 0; kt < K_STRIPES_GU; kt++) {
                    rt_bf<PFL_GEMM_M, 16> a_strip;
                    auto a_view = a_smem.template subtile<PFL_GEMM_M, 16>(int2{0, kt});
                    warp::load(a_strip, a_view);
                    #pragma unroll
                    for (int n = 0; n < PFL_N_TILES; n++) {
                        rt_bf<16, 16> b_strip;
                        auto b_view = b_smem.template subtile<16, 16>(int2{n, kt});
                        warp::load(b_strip, b_view);
                        #pragma unroll
                        for (int m_sub = 0; m_sub < PFL_GEMM_M_SUBS; m_sub++) {
                            warp::mma_ABt_base(acc.tiles[m_sub][n], a_strip.tiles[m_sub][0], b_strip.tiles[0][0], acc.tiles[m_sub][n]);
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
            // SiLU epilogue + store
            if (my_row_valid) {
                rt_bf<PFL_GEMM_M, PFL_OUT_BLOCK> out_bf;
                #pragma unroll
                for (int i = 0; i < acc.height; i++)
                    #pragma unroll
                    for (int j = 0; j < acc.width; j++)
                        #pragma unroll
                        // NOTE: per-thread data[] has packed_per_thread (=4) entries,
                        // NOT num_elements (=256 = whole-subtile count). See the
                        // dual_accum epilogue comment above for the full story.
                        for (int d = 0; d < acc.tiles[0][0].packed_per_thread; d++) {
                            float2 &v = acc.tiles[i][j].data[d];
                            v.x = v.x / (1.f + expf(-v.x));
                            v.y = v.y / (1.f + expf(-v.y));
                        }
                warp::copy(out_bf, acc);
                warp::store({{ output_global }}, out_bf, {row_tile, col});
            }
        }

        // No barrier needed — same CTA wrote silu_out[row_tile, col] above.

        // ── Up GEMM: input × up_weights → mulgate(silu_out) → silu_out ──
        {
            pfl_acc_rt acc;
            warp::zero(acc);
            // Prologue: fill stages 0..STAGES-2
            #pragma unroll
            for (int s = 0; s < STAGES - 1 && s < {{ num_k_iters }}; s++) {
                warp::load_async(*my_a_stages[s], {{ input_global }}, {safe_row_tile, s});
                group<PFL_NUM_WARPS>::load_async(*b_stages[s], {{ up_weight_global }}, {layer, col, s});
                asm volatile("cp.async.commit_group;\n" ::: "memory");
            }
            for (int iter = 0; iter < {{ num_k_iters }}; iter++) {
                int cur = iter % STAGES;
                int prefetch_iter = iter + STAGES - 1;
                if (prefetch_iter < {{ num_k_iters }}) {
                    int nxt = prefetch_iter % STAGES;
                    warp::load_async(*my_a_stages[nxt], {{ input_global }}, {safe_row_tile, prefetch_iter});
                    group<PFL_NUM_WARPS>::load_async(*b_stages[nxt], {{ up_weight_global }}, {layer, col, prefetch_iter});
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
                pfl_a_st &a_smem = *my_a_stages[cur];
                pfl_b_st &b_smem = *b_stages[cur];
                group<PFL_NUM_WARPS>::sync(1);
{%- if kstripe %}
                // K-stripe inner loop (CUTLASS-style register lifetime)
                constexpr int K_STRIPES_GU = PFL_K_DIM / 16;
                #pragma unroll
                for (int kt = 0; kt < K_STRIPES_GU; kt++) {
                    rt_bf<PFL_GEMM_M, 16> a_strip;
                    auto a_view = a_smem.template subtile<PFL_GEMM_M, 16>(int2{0, kt});
                    warp::load(a_strip, a_view);
                    #pragma unroll
                    for (int n = 0; n < PFL_N_TILES; n++) {
                        rt_bf<16, 16> b_strip;
                        auto b_view = b_smem.template subtile<16, 16>(int2{n, kt});
                        warp::load(b_strip, b_view);
                        #pragma unroll
                        for (int m_sub = 0; m_sub < PFL_GEMM_M_SUBS; m_sub++) {
                            warp::mma_ABt_base(acc.tiles[m_sub][n], a_strip.tiles[m_sub][0], b_strip.tiles[0][0], acc.tiles[m_sub][n]);
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
            // Mulgate epilogue: acc * silu_out → silu_out
            if (my_row_valid) {
                rt_bf<PFL_GEMM_M, PFL_OUT_BLOCK> acc_bf;
                warp::copy(acc_bf, acc);
                rt_bf<PFL_GEMM_M, PFL_OUT_BLOCK> gate_bf;
                warp::load(gate_bf, {{ output_global }}, {row_tile, col});
                #pragma unroll
                for (int r = 0; r < acc_bf.height; r++)
                    #pragma unroll
                    for (int c = 0; c < acc_bf.width; c++)
                        #pragma unroll
                        for (int k = 0; k < acc_bf.tiles[0][0].packed_per_thread; k++) {
                            bf16_2 &a = acc_bf.tiles[r][c].data[k];
                            bf16_2 &gv = gate_bf.tiles[r][c].data[k];
                            float a_lo = __bfloat162float(__low2bfloat16(a));
                            float a_hi = __bfloat162float(__high2bfloat16(a));
                            float g_lo = __bfloat162float(__low2bfloat16(gv));
                            float g_hi = __bfloat162float(__high2bfloat16(gv));
                            a = __floats2bfloat162_rn(a_lo * g_lo, a_hi * g_hi);
                        }
                warp::store({{ output_global }}, acc_bf, {row_tile, col});
            }
        }
    }
{%- else %}
    // ── 16-row fused gate+up (col_batch path) ──
    if (bid == 0 && wid == 0 && lid == 0 && layer == 0)
        printf("WARNING: fused gate+up not implemented for 16-row mode\n");
{%- endif %}
    }
    __syncthreads();
