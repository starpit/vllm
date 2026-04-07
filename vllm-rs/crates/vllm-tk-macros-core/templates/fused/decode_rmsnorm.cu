{# Decode RMSNorm: reads activations from shmem, writes normalized output to shmem.
   Weight vector loaded from global once per layer (small: HD * 2 bytes).

   Input:  phase_shm + input_offset  (BF16, [padded_rows, HD])
   Weight: {{ weight_global }}[layer]  (BF16, [HD])
   Output: phase_shm + output_offset (BF16, [padded_rows, HD])

   Uses the hidden_shmem region for both input and output (can be same or different).
   Weight is loaded into a temporary shmem region past the hidden data.

   Variables:
     weight_global: global accessor for weight (e.g. "g.attn_norm_weights")
     input_offset: byte offset into phase_shm for input activation slab
     output_offset: byte offset into phase_shm for output (often same as input)
     wgt_shmem_offset: byte offset for weight vector in shmem (after activations)
     scratch_offset: byte offset for warp reduction scratch
     rdpw: elements per warp (HD / num_warps)
#}
    // ════ Decode RMSNorm (shmem → shmem, weight from {{ weight_global }}) ════
    {
    bf16 *act_smem = reinterpret_cast<bf16*>(phase_shm + {{ input_offset }});
    bf16 *out_smem = reinterpret_cast<bf16*>(phase_shm + {{ output_offset }});
    bf16 *wgt_smem = reinterpret_cast<bf16*>(phase_shm + {{ wgt_shmem_offset }});
    float *scratch = reinterpret_cast<float*>(phase_shm + {{ scratch_offset }});

    // Load weight once per layer (all warps participate, warp 0 loads)
    if (wid == 0) {
        sv_bf<globals::hidden_dim> &w = *reinterpret_cast<sv_bf<globals::hidden_dim>*>(wgt_smem);
        warp::load_async(w, {{ weight_global }}, {layer, 0});
    }
    dec_cp_async_wait_all();
    __syncthreads();

    // Process each row
    sv_bf<{{ rdpw }}> *wgt_tiles = reinterpret_cast<sv_bf<{{ rdpw }}>*>(wgt_smem);
    for (int r = 0; r < my_rows; r++) {
        sv_bf<{{ rdpw }}> *row_tiles = reinterpret_cast<sv_bf<{{ rdpw }}>*>(act_smem + r * globals::hidden_dim);
        sv_bf<{{ rdpw }}> *out_tiles = reinterpret_cast<sv_bf<{{ rdpw }}>*>(out_smem + r * globals::hidden_dim);

        rv_fl<{{ rdpw }}> act_vec, copy_vec, scale_vec;
        warp::load(act_vec, row_tiles[wid]);
        warp::sync();

        // Compute sum of squares
        warp::copy(copy_vec, act_vec);
        warp::mul(copy_vec, copy_vec, copy_vec);
        float ps = warp::sum(copy_vec);
        if (lid == 0) scratch[wid] = ps;
        __syncthreads();

        // Cross-warp reduction
        float fs = 0.f;
        for (int i = 0; i < DEC_NUM_WARPS; i++) fs += scratch[i];
        float rms = rsqrtf(fs / (float)globals::hidden_dim + g.rms_norm_eps);

        // Scale and apply weight
        warp::copy(copy_vec, act_vec);
        warp::mul(copy_vec, copy_vec, rms);
        warp::load(scale_vec, wgt_tiles[wid]);
        warp::sync();
        warp::mul(copy_vec, copy_vec, scale_vec);
        warp::store(out_tiles[wid], copy_vec);
        warp::sync();
        __syncthreads();
    }
    }
