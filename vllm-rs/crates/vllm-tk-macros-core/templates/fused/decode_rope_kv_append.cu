{# Decode RoPE + KV cache append.
   Reads QKV from global (silu_out), applies per-row RoPE using position_id
   from row_meta. Writes K/V to paged cache. Writes Q to shmem for attention.

   Variables:
     hdm: head dimension
     q_end: Q elements per row (nah * hdm)
     k_start: K offset in QKV
     v_start: V offset in QKV
     kv_elems: total K elements per row (nkh * hdm)
     q_output_shmem_offset: byte offset for Q output slab in phase_shm
     q_output_stride: stride for Q output (nah * hdm)
#}
    // ════ Decode RoPE + KV cache append ════
    {
    for (int r = 0; r < my_rows; r++) {
        const int abs_row = row_start + r;
        const int token_pos = row_meta[r].position_id;
        // KV cache page for this row's new token
        const int kv_slot = row_meta[r].kv_indptr_end - 1;  // last page
        const int page_idx = g.decode_kv_indices[{kv_slot}] + layer * (int)g.num_pages;
        const int last_page_len = row_meta[r].kv_last_page_len;
        const int slot_in_page = last_page_len;  // append at end of last page

        // RoPE on Q: read from global silu_out, write to shmem
        bf16 *q_shmem = reinterpret_cast<bf16*>(phase_shm + {{ q_output_shmem_offset }});
        const int elems_per_warp_q = ({{ q_end }} + DEC_NUM_WARPS - 1) / DEC_NUM_WARPS;
        const int q_start_elem = wid * elems_per_warp_q;
        const int q_end_elem = min(q_start_elem + elems_per_warp_q, {{ q_end }});
        for (int tid = q_start_elem + lid; tid < q_end_elem; tid += 32) {
            const int head = tid / {{ hdm }};
            const int d = tid % {{ hdm }};
            const int half = {{ hdm }} / 2;
            float val = __bfloat162float(g.silu_out[coord<>{abs_row, tid}]);
            float cos_val = g.rope_cos[coord<>{token_pos, d}];
            float sin_val = g.rope_sin[coord<>{token_pos, d}];
            int pair_d = (d < half) ? (d + half) : (d - half);
            int pair_idx = head * {{ hdm }} + pair_d;
            float pair_val = __bfloat162float(g.silu_out[coord<>{abs_row, pair_idx}]);
            float rotated;
            if (d < half) rotated = val * cos_val - pair_val * sin_val;
            else          rotated = val * cos_val + pair_val * sin_val;
            // Write Q to shmem for attention phase
            q_shmem[r * {{ q_output_stride }} + tid] = __float2bfloat16(rotated);
        }

        // RoPE on K + write to cache
        const int elems_per_warp_kv = ({{ kv_elems }} + DEC_NUM_WARPS - 1) / DEC_NUM_WARPS;
        const int kv_start_elem = wid * elems_per_warp_kv;
        const int kv_end_elem = min(kv_start_elem + elems_per_warp_kv, {{ kv_elems }});
        for (int tid = kv_start_elem + lid; tid < kv_end_elem; tid += 32) {
            const int kv_head = tid / {{ hdm }};
            const int d = tid % {{ hdm }};
            const int half = {{ hdm }} / 2;
            float val = __bfloat162float(g.silu_out[coord<>{abs_row, {{ k_start }} + tid}]);
            float cos_val = g.rope_cos[coord<>{token_pos, d}];
            float sin_val = g.rope_sin[coord<>{token_pos, d}];
            int pair_d = (d < half) ? (d + half) : (d - half);
            float pair_val = __bfloat162float(g.silu_out[coord<>{abs_row, {{ k_start }} + kv_head * {{ hdm }} + pair_d}]);
            float rotated;
            if (d < half) rotated = val * cos_val - pair_val * sin_val;
            else          rotated = val * cos_val + pair_val * sin_val;
            g.k_cache[coord<>{page_idx, slot_in_page, kv_head, d}] = __float2bfloat16(rotated);
        }

        // V: copy to cache (no RoPE)
        for (int tid = kv_start_elem + lid; tid < kv_end_elem; tid += 32) {
            const int kv_head = tid / {{ hdm }};
            const int d = tid % {{ hdm }};
            bf16 val = g.silu_out[coord<>{abs_row, {{ v_start }} + tid}];
            g.v_cache[coord<>{page_idx, slot_in_page, kv_head, d}] = val;
        }
        __syncthreads();
    }
    }
