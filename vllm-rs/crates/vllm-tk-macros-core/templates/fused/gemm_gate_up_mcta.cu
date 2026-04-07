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
#}
    // ════ {{ phase_comment }} ════
    // Fused gate+up: two GEMMs, same A input, no barrier between.
    {
    const int col_tiles = {{ num_col_tiles }};
{%- if cooperative %}
    // ── Cooperative 128-row fused gate+up ──
    const int rows_per_cta = PFL_CTA_ROWS;
    const int row_tiles_coop = (q_size + rows_per_cta - 1) / rows_per_cta;
    const int total_work = row_tiles_coop * col_tiles;

    pfl_a_st &my_a_s0 = *reinterpret_cast<pfl_a_st*>(__shm + wid * {{ a_size }});
    pfl_b_st &b_s0 = *reinterpret_cast<pfl_b_st*>(__shm + {{ b_offset }});
    pfl_a_st &my_a_s1 = *reinterpret_cast<pfl_a_st*>(__shm + {{ stage_size }} + wid * {{ a_size }});
    pfl_b_st &b_s1 = *reinterpret_cast<pfl_b_st*>(__shm + {{ stage_size }} + {{ b_offset }});
    pfl_a_st *my_a_stages[2] = {&my_a_s0, &my_a_s1};
    pfl_b_st *b_stages[2] = {&b_s0, &b_s1};

    for (int wu = bid; wu < total_work; wu += num_ctas) {
        const int coop_row_tile = wu / col_tiles;
        const int col = wu % col_tiles;
        const int my_row_tile = coop_row_tile * (rows_per_cta / PFL_Q_ROWS) + wid;
        const int my_row_valid = (coop_row_tile * rows_per_cta + wid * PFL_Q_ROWS) < q_size;
        const int safe_row_tile = my_row_valid ? my_row_tile : 0;
        const int row_tile = my_row_tile;

        // ── Gate GEMM: input × gate_weights → SiLU → silu_out ──
        {
            pfl_acc_rt acc;
            warp::zero(acc);
            warp::load_async(*my_a_stages[0], {{ input_global }}, {safe_row_tile, 0});
            group<PFL_NUM_WARPS>::load_async(*b_stages[0], {{ gate_weight_global }}, {layer, col, 0});
            asm volatile("cp.async.commit_group;\n" ::: "memory");
            for (int iter = 0; iter < {{ num_k_iters }}; iter++) {
                int cur = iter % 2;
                if (iter + 1 < {{ num_k_iters }}) {
                    int nxt = (iter + 1) % 2;
                    warp::load_async(*my_a_stages[nxt], {{ input_global }}, {safe_row_tile, iter + 1});
                    group<PFL_NUM_WARPS>::load_async(*b_stages[nxt], {{ gate_weight_global }}, {layer, col, iter + 1});
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
            // SiLU epilogue + store
            if (my_row_valid) {
                rt_bf<16, PFL_OUT_BLOCK> out_bf;
                #pragma unroll
                for (int i = 0; i < acc.height; i++)
                    #pragma unroll
                    for (int j = 0; j < acc.width; j++)
                        #pragma unroll
                        for (int d = 0; d < acc.tiles[i][j].num_elements; d++) {
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
            warp::load_async(*my_a_stages[0], {{ input_global }}, {safe_row_tile, 0});
            group<PFL_NUM_WARPS>::load_async(*b_stages[0], {{ up_weight_global }}, {layer, col, 0});
            asm volatile("cp.async.commit_group;\n" ::: "memory");
            for (int iter = 0; iter < {{ num_k_iters }}; iter++) {
                int cur = iter % 2;
                if (iter + 1 < {{ num_k_iters }}) {
                    int nxt = (iter + 1) % 2;
                    warp::load_async(*my_a_stages[nxt], {{ input_global }}, {safe_row_tile, iter + 1});
                    group<PFL_NUM_WARPS>::load_async(*b_stages[nxt], {{ up_weight_global }}, {layer, col, iter + 1});
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
            // Mulgate epilogue: acc * silu_out → silu_out
            if (my_row_valid) {
                rt_bf<16, PFL_OUT_BLOCK> acc_bf;
                warp::copy(acc_bf, acc);
                rt_bf<16, PFL_OUT_BLOCK> gate_bf;
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
    // Falls back to sequential: gate GEMM then up GEMM, same work distribution.
    // TODO: implement col_batch variant if needed.
    // For now, use the cooperative path via v2-mcta-128row.
    if (bid == 0 && wid == 0 && lid == 0 && layer == 0)
        printf("WARNING: fused gate+up not implemented for 16-row mode\n");
{%- endif %}
    }
    __syncthreads();
