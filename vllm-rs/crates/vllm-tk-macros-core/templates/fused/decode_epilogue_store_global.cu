{# Decode epilogue: store GEMM output to global memory.
   Used for QKV output (K/V go to paged cache) and lm_head.
   Variables:
     output_global: global accessor for output
     col_var: variable name holding the current column tile index
#}
            // Store to global: out[row_start + r, col * out_block .. ]
            {
                st_bf<16, DEC_OUT_BLOCK> *acc_st = reinterpret_cast<st_bf<16, DEC_OUT_BLOCK>*>(__shm + DEC_META_SHMEM + DEC_HIDDEN_SHMEM + DEC_GEMM_STAGES * {{ b_size }} + {{ a_size }});
                dec_o_bf acc_bf;
                warp::copy(acc_bf, acc);
                warp::store(*acc_st, acc_bf);
                warp::sync();
                for (int r = 0; r < my_rows && r < 16; r++) {
                    // Write row to global
                    sv_bf<DEC_OUT_BLOCK> &row_sv = *reinterpret_cast<sv_bf<DEC_OUT_BLOCK>*>(
                        reinterpret_cast<bf16*>(acc_st) + r * DEC_OUT_BLOCK);
                    warp::store({{ output_global }}, row_sv, {row_start + r, {{ col_var }} * DEC_OUT_BLOCK / 16});
                }
            }
