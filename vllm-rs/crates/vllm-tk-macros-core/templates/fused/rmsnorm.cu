    // ════ RMSNorm ({{ input_global }} → {{ output_global }}) ════
    for (int qr = 0; qr < PFL_CTA_ROWS && (abs_q_row + qr) <= (q_start + rel_q_row_last); qr++) {
    {
    bf16 *act_smem = reinterpret_cast<bf16*>(__shm);
    bf16 *wgt_smem = reinterpret_cast<bf16*>(__shm + {{ wgt_offset }});
    float *scratch = reinterpret_cast<float*>(__shm + {{ scratch_offset }});
    sv_bf<PFL_RDPW> *act_tiles = reinterpret_cast<sv_bf<PFL_RDPW>*>(act_smem);
    sv_bf<PFL_RDPW> *wgt_tiles = reinterpret_cast<sv_bf<PFL_RDPW>*>(wgt_smem);
    if (qr == 0) {
    { sv_bf<globals::hidden_dim> &w = *reinterpret_cast<sv_bf<globals::hidden_dim>*>(wgt_smem);
       warp::load_async(w, {{ weight_global }}, {layer, 0}); }
    }
    { sv_bf<globals::hidden_dim> &a = *reinterpret_cast<sv_bf<globals::hidden_dim>*>(act_smem);
       warp::load_async(a, {{ input_global }}, {abs_q_row + qr, 0}); }
    pfl_cp_async_wait_all();
    group<PFL_NUM_WARPS>::sync(0);
    rv_fl<PFL_RDPW> act_vec, copy_vec, scale_vec;
    warp::load(act_vec, act_tiles[wid]); warp::sync();
    warp::copy(copy_vec, act_vec); warp::mul(copy_vec, copy_vec, copy_vec);
    float ps = warp::sum(copy_vec);
    if (lid == 0) scratch[wid] = ps;
    group<PFL_NUM_WARPS>::sync(0);
    float fs = 0.f; for (int i = 0; i < PFL_NUM_WARPS; i++) fs += scratch[i];
    float rms = rsqrtf(fs / (float)globals::hidden_dim + g.rms_norm_eps);
    warp::copy(copy_vec, act_vec); warp::mul(copy_vec, copy_vec, rms);
    warp::copy(act_vec, copy_vec);
    warp::load(scale_vec, wgt_tiles[wid]); warp::sync();
    warp::mul(act_vec, act_vec, scale_vec);
    warp::store(act_tiles[wid], act_vec); warp::sync();
    group<PFL_NUM_WARPS>::sync(0);
    if (wid == 0) {
        sv_bf<globals::hidden_dim> &r = *reinterpret_cast<sv_bf<globals::hidden_dim>*>(act_smem);
        warp::store({{ output_global }}, r, {abs_q_row + qr, 0});
    }
    __threadfence(); group<PFL_NUM_WARPS>::sync(0);
    }
    }
