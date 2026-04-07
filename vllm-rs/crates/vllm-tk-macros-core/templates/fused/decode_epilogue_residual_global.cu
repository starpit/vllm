{# Decode epilogue: GEMM output + residual from GLOBAL hidden_states → shmem.
   Used for o_proj when hidden_states in shmem have been overwritten by
   intermediate phases (QKV, RoPE, attention).
   Variables:
     output_shmem_offset: byte offset for output slab in phase_shm
     col_var: current column tile index
#}
            // Residual add: out = gemm_result + g.hidden_states[global], write to shmem
            {
                bf16 *out_smem = reinterpret_cast<bf16*>(phase_shm + {{ output_shmem_offset }});
                st_bf<16, DEC_OUT_BLOCK> *acc_st = reinterpret_cast<st_bf<16, DEC_OUT_BLOCK>*>(__shm + DEC_META_SHMEM + DEC_HIDDEN_SHMEM + DEC_GEMM_STAGES * {{ b_size }} + {{ a_size }});
                dec_o_bf acc_bf;
                warp::copy(acc_bf, acc);
                warp::store(*acc_st, acc_bf);
                warp::sync();
                for (int r = 0; r < my_rows && r < 16; r++) {
                    bf16 *out_row = out_smem + r * globals::hidden_dim + {{ col_var }} * DEC_OUT_BLOCK;
                    const bf16 *h_base = reinterpret_cast<const bf16*>(g.hidden_states.raw_ptr);
                    for (int j = lid; j < DEC_OUT_BLOCK; j += 32) {
                        float g_val = __bfloat162float((*acc_st)[{r, j}]);
                        int col_idx = {{ col_var }} * DEC_OUT_BLOCK + j;
                        float res = __bfloat162float(
                            h_base[(row_start + r) * globals::hidden_dim + col_idx]);
                        out_row[j] = __float2bfloat16(g_val + res);
                    }
                }
            }
