{# Multi-CTA FlashAttention-2 prefill: distribute (q_block, kv_head) pairs across CTAs.
   Variables: same as attention.cu
   Uses row_tiles, q_start, q_end, q_size from kernel scope.
#}
    // ════ Attention prefill (multi-CTA) ════
    {
    const int kv_indptr_start = g.prefill_kv_indptr[{seq_idx}];
    const int attn_work = row_tiles * {{ nkh }};

    for (int wu = bid; wu < attn_work; wu += num_ctas) {
        const int q_block = wu / {{ nkh }};
        const int kv_head = wu % {{ nkh }};
        const int attn_abs_q_row = q_start + q_block * PFL_Q_ROWS;
        const int attn_rel_q_row = q_block * PFL_Q_ROWS;
        if (attn_abs_q_row >= q_end) continue;

        const int sequence_length = attn_rel_q_row + min(PFL_Q_ROWS, q_size - attn_rel_q_row);
        const int attn_pages = (sequence_length + PFL_KV_PAGE_SIZE - 1) / PFL_KV_PAGE_SIZE;

    if (wid < PFL_GQA_RATIO) {
    const int q_head = kv_head * PFL_GQA_RATIO + wid;

    pfl_q_st &Q_smem = *reinterpret_cast<pfl_q_st*>(__shm);
    {
        using T = bf16;
        constexpr int elem_per_cp = sizeof(float4) / sizeof(T);
        constexpr int lanes_per_row = PFL_HEAD_DIM / elem_per_cp;
        constexpr int rows_per_iter = 32 / lanes_per_row;
        auto *src_ptr = (T*)&g.q_post_rope[coord<>{attn_abs_q_row, q_head * PFL_HEAD_DIM}];
        uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&Q_smem.data[0]));
        for (int ri = 0; ri < (PFL_Q_ROWS + rows_per_iter - 1) / rows_per_iter; ri++) {
            int row = ri * rows_per_iter + lid / lanes_per_row;
            int col = (lid % lanes_per_row) * elem_per_cp;
            if (row < PFL_Q_ROWS && (attn_abs_q_row + row) < q_end) {
                asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::
                    "r"(Q_smem.idx(dst_ptr, {row, col})),
                    "l"(&src_ptr[row * {{ nah }} * PFL_HEAD_DIM + col]) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_all;\n" ::: "memory");
    }
    __syncwarp();
    pfl_q_rt Q_reg;
    warp::load(Q_reg, Q_smem);

    pfl_o_rt O_reg;
    pfl_max_rv max_vec, scaled_max, last_scaled_max, diff_scaled_max;
    pfl_norm_rv norm_vec;
    warp::neg_infty(max_vec);
    warp::zero(last_scaled_max);
    warp::zero(norm_vec);
    warp::zero(O_reg);
    float softmax_temp = g.attn_scale * 1.44269504089f;

    for (int page = 0; page < attn_pages; page++) {
        int stage = page % 2;
        pfl_kv_st &K_smem = *reinterpret_cast<pfl_kv_st*>(__shm + stage * {{ stage_sz }});
        pfl_kv_st &V_smem = *reinterpret_cast<pfl_kv_st*>(__shm + stage * {{ stage_sz }} + PFL_KV_TILE_BYTES);
        int kv_page_index = g.prefill_kv_indices[{kv_indptr_start + page}];
        int page_batch = (int)g.num_pages * layer + kv_page_index;
        {
            using T = bf16;
            constexpr int nkh = {{ nkh }};
            constexpr int hd = PFL_HEAD_DIM;
            constexpr int ipp = PFL_ITERS_PER_PAGE;
            constexpr int elem_per_cp = sizeof(float4) / sizeof(T);
            constexpr int lanes_per_row = hd / elem_per_cp;
            constexpr int rows_per_iter = 32 / lanes_per_row;
            T *k_base = (T*)g.k_cache.raw_ptr;
            T *v_base = (T*)g.v_cache.raw_ptr;
            uint32_t k_smem = static_cast<uint32_t>(__cvta_generic_to_shared(&K_smem.data[0]));
            uint32_t v_smem = static_cast<uint32_t>(__cvta_generic_to_shared(&V_smem.data[0]));
            for (int ri = 0; ri < (PFL_KV_PAGE_SIZE + rows_per_iter - 1) / rows_per_iter; ri++) {
                int row = ri * rows_per_iter + lid / lanes_per_row;
                int col = (lid % lanes_per_row) * elem_per_cp;
                if (row < PFL_KV_PAGE_SIZE) {
                    long src_off = ((long)page_batch * ipp + row) * nkh * hd + (long)kv_head * hd + col;
                    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::
                        "r"(K_smem.idx(k_smem, {row, col})),
                        "l"(&k_base[src_off]) : "memory");
                    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::
                        "r"(V_smem.idx(v_smem, {row, col})),
                        "l"(&v_base[src_off]) : "memory");
                }
            }
        }
        pfl_cp_async_wait_all();
        __syncwarp();

        pfl_k_rt K_reg;
        warp::load(K_reg, K_smem);
        pfl_score_fl attn_fl;
        warp::zero(attn_fl);
        warp::mma_ABt(attn_fl, Q_reg, K_reg, attn_fl);
        int kv_pos_start = page * PFL_KV_PAGE_SIZE;
        warp::apply(attn_fl, attn_fl,
            [kv_pos_start, attn_rel_q_row] __device__(int row, int col, float val) {
                return (kv_pos_start + col > attn_rel_q_row + row) ? -999999999999.f : val;
            });
        if (page == attn_pages - 1) {
            int valid_kv = sequence_length - page * PFL_KV_PAGE_SIZE;
            if (valid_kv < PFL_KV_PAGE_SIZE)
                warp::apply(attn_fl, attn_fl, [valid_kv] __device__(int row, int col, float val) {
                    return (col >= valid_kv) ? -999999999999.f : val; });
        }
        warp::row_max(max_vec, attn_fl, max_vec);
        warp::mul(attn_fl, attn_fl, softmax_temp);
        warp::mul(scaled_max, max_vec, softmax_temp);
        warp::sub_row(attn_fl, attn_fl, scaled_max);
        warp::exp2(attn_fl, attn_fl);
        warp::sub(diff_scaled_max, last_scaled_max, scaled_max);
        warp::exp2(diff_scaled_max, diff_scaled_max);
        warp::mul_row(O_reg, O_reg, diff_scaled_max);
        pfl_v_rt V_reg;
        warp::load(V_reg, V_smem);
        pfl_score_bf attn_bf;
        warp::copy(attn_bf, attn_fl);
        warp::mma_AB(O_reg, attn_bf, V_reg, O_reg);
        warp::mul(norm_vec, norm_vec, diff_scaled_max);
        warp::row_sum(norm_vec, attn_fl, norm_vec);
        warp::copy(last_scaled_max, scaled_max);
    }

    warp::add(norm_vec, norm_vec, 1e-16f);
    warp::div_row(O_reg, O_reg, norm_vec);
    pfl_o_bf O_bf;
    warp::copy(O_bf, O_reg);
    pfl_q_st &O_st = *reinterpret_cast<pfl_q_st*>(__shm);
    warp::store(O_st, O_bf);
    warp::sync();
    {
        uint32_t src_base = static_cast<uint32_t>(__cvta_generic_to_shared(&O_st.data[0]));
        for (int row = 0; row < PFL_Q_ROWS; row++) {
            if (attn_abs_q_row + row >= q_end) break;
            auto *dst = (bf16*)&g.attn_out[coord<>{attn_abs_q_row + row, q_head * PFL_HEAD_DIM}];
            for (int i = lid; i < PFL_HEAD_DIM; i += 32) {
                bf16 val; move<bf16>::lds(val, O_st.idx(src_base, {row, i}));
                dst[i] = val;
            }
        }
    }

    }  // wid < GQA_RATIO
    }  // work unit loop
    }
    __threadfence(); __syncthreads();
