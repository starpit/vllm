{# Decode epilogue: GEMM output × existing gate values → global memory.
   Used for up GEMM in MLP: (normed × up_proj) * silu_out → silu_out (in-place).
   Variables:
     gate_global: global accessor for gate values to multiply with (e.g., "g.silu_out")
     col_var: variable name holding the current column tile index
#}
            // MulGate epilogue → global: out[row, col] = acc * gate[row, col]
            {
                st_bf<16, DEC_OUT_BLOCK> *acc_st = reinterpret_cast<st_bf<16, DEC_OUT_BLOCK>*>(__shm + DEC_META_SHMEM + DEC_HIDDEN_SHMEM + DEC_GEMM_STAGES * {{ b_size }} + {{ a_size }});
                dec_o_bf acc_bf;
                warp::copy(acc_bf, acc);
                warp::store(*acc_st, acc_bf);
                warp::sync();
                for (int r = 0; r < my_rows && r < 16; r++) {
                    for (int j = lid; j < DEC_OUT_BLOCK; j += 32) {
                        int col_idx = {{ col_var }} * DEC_OUT_BLOCK + j;
                        float up_val = __bfloat162float((*acc_st)[{r, j}]);
                        float gate_val = __bfloat162float({{ gate_global }}[{row_start + r, col_idx}]);
                        {{ gate_global }}[{row_start + r, col_idx}] = __float2bfloat16(up_val * gate_val);
                    }
                }
            }
