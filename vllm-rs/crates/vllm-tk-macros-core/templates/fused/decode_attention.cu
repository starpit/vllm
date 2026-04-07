{# Decode attention: per-row paged flash-decode.
   Q from shmem (written by RoPE phase). KV from paged cache (global).
   Output to shmem (for o_proj GEMM).

   Each warp handles one KV head for all rows sequentially.
   With NKH=8 and 8 warps, each warp = one KV head.
   GQA: each warp computes GQA_RATIO query heads against its KV head.

   Variables:
     nkh: number of KV heads
     nah: number of attention heads
     q_shmem_offset: byte offset for Q slab in phase_shm
     q_stride: stride for Q (nah * hdm)
     output_shmem_offset: byte offset for attention output slab
     output_stride: stride for output (nah * hdm or hd)
#}
    // ════ Decode attention (per-row paged flash-decode) ════
    {
    const int kv_head = wid % {{ nkh }};

    for (int r = 0; r < my_rows; r++) {
        const int abs_row = row_start + r;
        const int kv_start = row_meta[r].kv_indptr_start;
        const int kv_end = row_meta[r].kv_indptr_end;
        const int num_kv_pages = kv_end - kv_start;
        const int last_page_len = row_meta[r].kv_last_page_len;

        // Process GQA_RATIO query heads for this KV head
        for (int gqa = 0; gqa < DEC_GQA_RATIO; gqa++) {
            const int q_head = kv_head * DEC_GQA_RATIO + gqa;

            // Load Q from shmem into registers
            dec_q_st &Q_smem = *reinterpret_cast<dec_q_st*>(
                phase_shm + {{ q_shmem_offset }} + r * {{ q_stride }} * 2 + q_head * DEC_HEAD_DIM * 2);
            // Q is actually a single row, but we use a 16-row tile type.
            // Only row 0 has valid data. We broadcast it.

            // Online softmax state
            float running_max = -999999999999.f;
            float running_sum = 0.f;
            float o_acc[DEC_HEAD_DIM];
            for (int d = 0; d < DEC_HEAD_DIM; d++) o_acc[d] = 0.f;

            // Load Q values for this head into registers
            float q_vals[DEC_HEAD_DIM];
            {
                bf16 *q_ptr = reinterpret_cast<bf16*>(
                    phase_shm + {{ q_shmem_offset }}) + r * {{ q_stride }} + q_head * DEC_HEAD_DIM;
                for (int d = lid; d < DEC_HEAD_DIM; d += 32) {
                    q_vals[d] = __bfloat162float(q_ptr[d]);
                }
            }

            for (int page = 0; page < num_kv_pages; page++) {
                int kv_page_index = g.decode_kv_indices[{kv_start + page}];
                int page_batch = (int)g.num_pages * layer + kv_page_index;
                // +1 on last page: RoPE phase just appended the new token at slot last_page_len
                int valid_tokens = (page == num_kv_pages - 1) ? (last_page_len + 1) : DEC_KV_PAGE_SIZE;

                // Compute QK^T for each token in this page
                for (int tok = 0; tok < valid_tokens; tok++) {
                    // Dot product Q[1, HDM] × K[tok, HDM]^T
                    float dot = 0.f;
                    for (int d = lid; d < DEC_HEAD_DIM; d += 32) {
                        float k_val = __bfloat162float(
                            g.k_cache[coord<>{page_batch, tok, kv_head, d}]);
                        dot += q_vals[d] * k_val;
                    }
                    // Warp reduction
                    for (int mask = 16; mask > 0; mask >>= 1)
                        dot += __shfl_xor_sync(0xFFFFFFFF, dot, mask);

                    float score = dot * g.attn_scale;

                    // Online softmax update
                    float old_max = running_max;
                    running_max = fmaxf(running_max, score);
                    float exp_old = expf(old_max - running_max);
                    float exp_new = expf(score - running_max);
                    running_sum = running_sum * exp_old + exp_new;

                    // Update O accumulator
                    for (int d = lid; d < DEC_HEAD_DIM; d += 32) {
                        float v_val = __bfloat162float(
                            g.v_cache[coord<>{page_batch, tok, kv_head, d}]);
                        o_acc[d] = o_acc[d] * exp_old + exp_new * v_val;
                    }
                }
            }

            // Normalize and write to shmem
            float inv_sum = (running_sum > 0.f) ? (1.f / running_sum) : 0.f;
            bf16 *out_ptr = reinterpret_cast<bf16*>(
                phase_shm + {{ output_shmem_offset }}) + r * {{ output_stride }} + q_head * DEC_HEAD_DIM;
            for (int d = lid; d < DEC_HEAD_DIM; d += 32) {
                out_ptr[d] = __float2bfloat16(o_acc[d] * inv_sum);
            }
        }  // gqa loop
    }  // row loop
    __syncthreads();
    }
