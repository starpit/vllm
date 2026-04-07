    // ════ RoPE + KV cache append ════
    for (int qr = 0; qr < PFL_CTA_ROWS && (abs_q_row + qr) <= (q_start + rel_q_row_last); qr++) {
    {
    const int token_pos = abs_q_row + qr;
    const int page_idx = g.prefill_kv_indices[{token_pos / PFL_KV_PAGE_SIZE}] + layer * (int)g.num_pages;
    const int slot_in_page = token_pos % PFL_KV_PAGE_SIZE;

    const int elems_per_warp_q = ({{ q_end }} + PFL_NUM_WARPS - 1) / PFL_NUM_WARPS;
    const int q_start_elem = wid * elems_per_warp_q;
    const int q_end_elem = min(q_start_elem + elems_per_warp_q, {{ q_end }});
    for (int tid = q_start_elem + lid; tid < q_end_elem; tid += 32) {
        const int head = tid / {{ hdm }};
        const int d = tid % {{ hdm }};
        const int half = {{ hdm }} / 2;
        float val = __bfloat162float(g.silu_out[coord<>{token_pos, tid}]);
        float cos_val = g.rope_cos[coord<>{token_pos, d}];
        float sin_val = g.rope_sin[coord<>{token_pos, d}];
        int pair_d = (d < half) ? (d + half) : (d - half);
        int pair_idx = head * {{ hdm }} + pair_d;
        float pair_val = __bfloat162float(g.silu_out[coord<>{token_pos, pair_idx}]);
        float rotated;
        if (d < half) rotated = val * cos_val - pair_val * sin_val;
        else          rotated = val * cos_val + pair_val * sin_val;
        g.q_post_rope[coord<>{token_pos, tid}] = __float2bfloat16(rotated);
    }
    const int elems_per_warp_kv = ({{ kv_elems }} + PFL_NUM_WARPS - 1) / PFL_NUM_WARPS;
    const int kv_start_elem = wid * elems_per_warp_kv;
    const int kv_end_elem = min(kv_start_elem + elems_per_warp_kv, {{ kv_elems }});
    for (int tid = kv_start_elem + lid; tid < kv_end_elem; tid += 32) {
        const int kv_head = tid / {{ hdm }};
        const int d = tid % {{ hdm }};
        const int half = {{ hdm }} / 2;
        float val = __bfloat162float(g.silu_out[coord<>{token_pos, {{ k_start }} + tid}]);
        float cos_val = g.rope_cos[coord<>{token_pos, d}];
        float sin_val = g.rope_sin[coord<>{token_pos, d}];
        int pair_d = (d < half) ? (d + half) : (d - half);
        float pair_val = __bfloat162float(g.silu_out[coord<>{token_pos, {{ k_start }} + kv_head * {{ hdm }} + pair_d}]);
        float rotated;
        if (d < half) rotated = val * cos_val - pair_val * sin_val;
        else          rotated = val * cos_val + pair_val * sin_val;
        g.k_cache[coord<>{page_idx, slot_in_page, kv_head, d}] = __float2bfloat16(rotated);
    }
    for (int tid = kv_start_elem + lid; tid < kv_end_elem; tid += 32) {
        const int kv_head = tid / {{ hdm }};
        const int d = tid % {{ hdm }};
        bf16 val = g.silu_out[coord<>{token_pos, {{ v_start }} + tid}];
        g.v_cache[coord<>{page_idx, slot_in_page, kv_head, d}] = val;
    }
    }
    }
    __threadfence(); __syncthreads();

