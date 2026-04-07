{# Decode GEMM: A from shmem (activations), B from global (weights), output via epilogue.
   Always cooperative: each warp owns its own 16-row A slice.
   Double-buffered K-loop.

   For decode, we process rows sequentially (small batch). Each warp handles
   one 16-row MMA tile. With padded_cta_rows=16 and 8 warps, all warps
   redundantly compute the same tile (warp 0 stores). With padded_cta_rows=16,
   only wid=0 is active; others skip.

   Variables:
     phase_comment: description string
     input_shmem_offset: byte offset into phase_shm for A activation slab
     weight_global: global accessor for B weights
     num_k_iters: K-loop iteration count
     num_col_tiles: number of output column tiles
     a_size, b_size, stage_size, b_offset: GEMM tile sizes
     num_stages: pipeline depth
     epilogue: rendered epilogue string
     epilogue_target: "shmem" or "global" — where epilogue writes
     output_shmem_offset: byte offset for output slab (when epilogue_target == "shmem")
#}
    // ════ {{ phase_comment }} ════
    {
    // A source: shmem activation slab at phase_shm + {{ input_shmem_offset }}
    bf16 *gemm_a_src = reinterpret_cast<bf16*>(phase_shm + {{ input_shmem_offset }});

    // GEMM shmem region: reuse the phase_shm area (time-shared with RMSNorm/attention)
    // We need space for B tiles only — A is loaded from shmem directly into registers.
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

            // Load A tile from shmem into registers (row 0, k_iter = iter)
            // For decode: each row is [HD] BF16 in shmem. We need a [16, k_dim] slice.
            // Only wid==0 has valid data for the single MMA row tile.
            // A[row, k_iter*k_dim .. (k_iter+1)*k_dim]
            rt_bf<16, DEC_K_DIM> a_reg;
            // Load A directly from activation shmem — this is the key decode optimization.
            // For now, wid==0 loads the data; all warps compute redundantly.
            {
                // A layout in shmem: [padded_rows, HD] BF16, row-major
                // For the current row being processed (in the outer row loop):
                // We need shmem[row * HD + iter * k_dim] as a [16, k_dim] tile.
                // But we only have 1 actual row per "MMA tile" in decode.
                // Broadcast: load the row into all 16 MMA rows of a_reg.
                dec_a_st &a_smem = *reinterpret_cast<dec_a_st*>(__shm + DEC_META_SHMEM + DEC_HIDDEN_SHMEM + DEC_GEMM_STAGES * {{ b_size }});
                // Copy row slice from activation shmem to A tile shmem
                if (wid == 0) {
                    for (int r = 0; r < my_rows && r < 16; r++) {
                        bf16 *src = gemm_a_src + r * globals::hidden_dim + iter * DEC_K_DIM;
                        bf16 *dst = reinterpret_cast<bf16*>(&a_smem) + r * DEC_K_DIM;
                        for (int j = lid; j < DEC_K_DIM; j += 32) {
                            dst[j] = src[j];
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
                // Load B slice using same pattern as prefill
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
