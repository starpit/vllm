{# Decode epilogue: GEMM output + residual from GLOBAL → write back to GLOBAL.
   Used for o_proj when input and output share the same shmem region.
   Writing to global avoids corrupting the A input shmem during the col loop.
   A subsequent global→shmem copy loads the result back.
   Variables:
     col_var: current column tile index
     a_size, b_size: tile sizes
#}
            // Residual add: out = gemm_result + g.hidden_states[global], write to GLOBAL
            {
                st_bf<16, DEC_OUT_BLOCK> *acc_st = reinterpret_cast<st_bf<16, DEC_OUT_BLOCK>*>(__shm + DEC_META_SHMEM + DEC_HIDDEN_SHMEM + DEC_GEMM_STAGES * {{ b_size }} + {{ a_size }});
                dec_o_bf acc_bf;
                warp::copy(acc_bf, acc);
                warp::store(*acc_st, acc_bf);
                warp::sync();
                bf16 *h_rw = reinterpret_cast<bf16*>(g.hidden_states.raw_ptr);
                for (int r = 0; r < my_rows && r < 16; r++) {
                    for (int j = lid; j < DEC_OUT_BLOCK; j += 32) {
                        int col_idx = {{ col_var }} * DEC_OUT_BLOCK + j;
                        int gidx = (row_start + r) * globals::hidden_dim + col_idx;
                        float g_val = __bfloat162float((*acc_st)[{r, j}]);
                        float res = __bfloat162float(h_rw[gidx]);
                        h_rw[gidx] = __float2bfloat16(g_val + res);
                    }
                }
            }
