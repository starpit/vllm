{# Decode epilogue: SiLU(GEMM output) → global memory.
   Used for gate GEMM in MLP: silu(normed × gate_proj) → g.silu_out.
   Variables:
     output_global: global accessor for output (e.g., "g.silu_out")
     col_var: variable name holding the current column tile index
#}
            // SiLU epilogue → global: out[row, col * out_block .. ] = SiLU(acc)
            {
                st_bf<16, DEC_OUT_BLOCK> *acc_st = reinterpret_cast<st_bf<16, DEC_OUT_BLOCK>*>(__shm + DEC_META_SHMEM + DEC_HIDDEN_SHMEM + DEC_GEMM_STAGES * {{ b_size }} + {{ a_size }});
                dec_o_bf acc_bf;
                warp::copy(acc_bf, acc);
                warp::store(*acc_st, acc_bf);
                warp::sync();
                for (int r = 0; r < my_rows && r < 16; r++) {
                    for (int j = lid; j < DEC_OUT_BLOCK; j += 32) {
                        int col_idx = {{ col_var }} * DEC_OUT_BLOCK + j;
                        float val = __bfloat162float((*acc_st)[{r, j}]);
                        float silu_val = val / (1.f + expf(-val));
                        {{ output_global }}[{row_start + r, col_idx}] = __float2bfloat16(silu_val);
                    }
                }
            }
