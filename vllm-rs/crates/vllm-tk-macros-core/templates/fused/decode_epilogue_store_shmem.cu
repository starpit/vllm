{# Decode epilogue: store GEMM output to shmem slab.
   Variables:
     output_shmem_offset: byte offset into phase_shm for output
     col_var: variable name holding the current column tile index
#}
            // Store to shmem: out[row, col * out_block .. (col+1) * out_block]
            {
                bf16 *out_smem = reinterpret_cast<bf16*>(phase_shm + {{ output_shmem_offset }});
                st_bf<16, DEC_OUT_BLOCK> *acc_st = reinterpret_cast<st_bf<16, DEC_OUT_BLOCK>*>(__shm + DEC_META_SHMEM + DEC_HIDDEN_SHMEM + DEC_GEMM_STAGES * {{ b_size }} + {{ a_size }});
                dec_o_bf acc_bf;
                warp::copy(acc_bf, acc);
                warp::store(*acc_st, acc_bf);
                warp::sync();
                for (int r = 0; r < my_rows && r < 16; r++) {
                    bf16 *dst = out_smem + r * {{ output_stride }} + {{ col_var }} * DEC_OUT_BLOCK;
                    for (int j = lid; j < DEC_OUT_BLOCK; j += 32) {
                        dst[j] = (*acc_st)[{r, j}];
                    }
                }
            }
