{# Decode epilogue: GEMM output + residual from shmem → shmem.
   Used for o_proj + residual and down_proj + residual.
   Variables:
     residual_shmem_offset: byte offset for residual (hidden_states) slab in phase_shm
     output_shmem_offset: byte offset for output slab
     col_var: current column tile index
#}
            // Residual add: out = gemm_result + residual, both in shmem
            {
                bf16 *res_smem = reinterpret_cast<bf16*>(phase_shm + {{ residual_shmem_offset }});
                bf16 *out_smem = reinterpret_cast<bf16*>(phase_shm + {{ output_shmem_offset }});
                st_bf<16, DEC_OUT_BLOCK> *acc_st = reinterpret_cast<st_bf<16, DEC_OUT_BLOCK>*>(__shm + DEC_META_SHMEM + DEC_HIDDEN_SHMEM + DEC_GEMM_STAGES * {{ b_size }} + {{ a_size }});
                dec_o_bf acc_bf;
                warp::copy(acc_bf, acc);
                warp::store(*acc_st, acc_bf);
                warp::sync();
                for (int r = 0; r < my_rows && r < 16; r++) {
                    bf16 *gemm_row = reinterpret_cast<bf16*>(acc_st) + r * DEC_OUT_BLOCK;
                    bf16 *res_row = res_smem + r * globals::hidden_dim + {{ col_var }} * DEC_OUT_BLOCK;
                    bf16 *out_row = out_smem + r * globals::hidden_dim + {{ col_var }} * DEC_OUT_BLOCK;
                    for (int j = lid; j < DEC_OUT_BLOCK; j += 32) {
                        float g = __bfloat162float(gemm_row[j]);
                        float re = __bfloat162float(res_row[j]);
                        out_row[j] = __float2bfloat16(g + re);
                    }
                }
            }
