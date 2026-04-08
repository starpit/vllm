{# Decode GEMM with A loaded from GLOBAL memory (not shmem).
   Used for down_proj in MLP where the input (silu_out) is in global.
   Same K-loop structure as decode_gemm.cu but A comes from a global accessor.

   Variables:
     phase_comment: description string
     a_global: global accessor for A input (e.g., "g.silu_out")
     a_stride: row stride in elements for A (e.g., intermediate_dim)
     weight_global: global accessor for B weights
     num_k_iters: K-loop iteration count
     num_col_tiles: number of output column tiles
     a_size, b_size, stage_size, b_offset: GEMM tile sizes
     num_stages: pipeline depth
     epilogue: rendered epilogue string
#}
    // ════ {{ phase_comment }} ════
    {
    // A source: global memory ({{ a_global }})

    // GEMM shmem region: B tiles only
    constexpr int DEC_GEMM_STAGES = {{ num_stages }};
    dec_b_st *b_stages[DEC_GEMM_STAGES];
    #pragma unroll
    for (int s = 0; s < DEC_GEMM_STAGES; s++) {
        b_stages[s] = reinterpret_cast<dec_b_st*>(__shm + DEC_META_SHMEM + DEC_HIDDEN_SHMEM + s * {{ b_size }});
    }

    const int col_tiles = {{ num_col_tiles }};
    for (int col = 0; col < col_tiles; col++) {
        dec_acc_rt acc;
        warp::zero(acc);

        // Prologue: fill stages 0..STAGES-2
        #pragma unroll
        for (int s = 0; s < DEC_GEMM_STAGES - 1 && s < {{ num_k_iters }}; s++) {
            group<DEC_NUM_WARPS>::load_async(*b_stages[s], {{ weight_global }}, {layer, col, s});
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }

        for (int iter = 0; iter < {{ num_k_iters }}; iter++) {
            int cur = iter % DEC_GEMM_STAGES;
            int prefetch_iter = iter + DEC_GEMM_STAGES - 1;
            if (prefetch_iter < {{ num_k_iters }}) {
                int nxt = prefetch_iter % DEC_GEMM_STAGES;
                group<DEC_NUM_WARPS>::load_async(*b_stages[nxt], {{ weight_global }}, {layer, col, prefetch_iter});
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
            __syncthreads();

            // Load A tile from GLOBAL into registers via shmem scratch
            rt_bf<16, DEC_K_DIM> a_reg;
            {
                dec_a_st &a_smem = *reinterpret_cast<dec_a_st*>(__shm + DEC_META_SHMEM + DEC_HIDDEN_SHMEM + DEC_GEMM_STAGES * {{ b_size }});
                // Copy from global: A[row, iter*k_dim .. (iter+1)*k_dim]
                if (wid == 0) {
                    for (int r = 0; r < my_rows && r < 16; r++) {
                        for (int j = lid; j < DEC_K_DIM; j += 32) {
                            int global_col = iter * DEC_K_DIM + j;
                            a_smem[{r, j}] = {{ a_global }}[{row_start + r, global_col}];
                        }
                    }
                }
                __syncthreads();
                warp::load(a_reg, a_smem);
            }

            dec_b_st &b_smem = *b_stages[cur];
            st_bf<16, DEC_K_DIM> *b_slices = reinterpret_cast<st_bf<16, DEC_K_DIM>*>(&b_smem);
            constexpr int N_TILES = DEC_OUT_BLOCK / 16;
            #pragma unroll
            for (int n = 0; n < N_TILES; n++) {
                rt_bf<16, DEC_K_DIM> b_n;
                uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(&b_slices[n].data[0]));
                int lane = kittens::laneid();
                int row = lane % 16;
                bf16_2 tmp[4];
                #pragma unroll
                for (int j = 0; j < DEC_K_DIM / 16; j++) {
                    int bcol = j * 16 + (lane / 16) * 8;
                    move<bf16_2>::ldsm4(tmp[0], tmp[1], tmp[2], tmp[3], b_slices[n].idx(saddr, {row, bcol}));
                    b_n.tiles[0][j].data[0] = tmp[0];
                    b_n.tiles[0][j].data[1] = tmp[1];
                    b_n.tiles[0][j].data[2] = tmp[2];
                    b_n.tiles[0][j].data[3] = tmp[3];
                }
                warp::mma_ABt_base(acc.tiles[0][n], a_reg.tiles[0][0], b_n.tiles[0][0], acc.tiles[0][n]);
                #pragma unroll
                for (int k = 1; k < a_reg.width; k++)
                    warp::mma_ABt_base(acc.tiles[0][n], a_reg.tiles[0][k], b_n.tiles[0][k], acc.tiles[0][n]);
            }
        }
        // Epilogue: store result
        if (wid == 0) {
{{ epilogue }}
        }
        __syncthreads();
    }
    }
