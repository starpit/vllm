{# Decode fused MLP: streams gate+up+SiLU+down over ID tiles.
   No [R, ID] intermediate materialization.

   Input:  shmem activation slab (after mlp_norm)
   Output: shmem hidden_states slab (with residual add)
   Weights: gate_weights, up_weights, down_weights from global

   Algorithm:
     down_acc[R, HD] = 0  (in registers, accumulated across ID tiles)
     for id_tile in 0..ID/out_block:
       gate_acc = act_shmem × gate_w[layer, id_tile]   // [R, out_block] registers
       up_acc   = act_shmem × up_w[layer, id_tile]     // [R, out_block] registers
       silu_up  = SiLU(gate_acc) * up_acc               // register × register
       down_acc += silu_up × down_w[layer, id_tile]^T   // accumulate [R, HD] partial

   Variables:
     input_shmem_offset: byte offset for normalized activations
     hidden_shmem_offset: byte offset for hidden_states (residual)
     gate_weight_global: global accessor for gate weights
     up_weight_global: global accessor for up weights
     down_weight_global: global accessor for down weights
     hd_k_iters: K-loop iterations for HD dimension
     id_col_tiles: number of ID column tiles
     a_size, b_size: tile sizes
     num_stages: pipeline depth
     hd: hidden dimension
     id: intermediate dimension
#}
    // ════ Decode fused MLP (streaming gate+up+SiLU+down, no [R,ID] materialization) ════
    {
    bf16 *mlp_act_smem = reinterpret_cast<bf16*>(phase_shm + {{ input_shmem_offset }});
    // Residual for MLP comes from global g.hidden_states (written back after o_proj).
    // Cannot use shmem because mlp_norm overwrote it in-place.

    // Raw weight pointers — avoids tiled GL coord access issues.
    // gate_weights: [NL, ID, HD] (output=ID, input=HD)
    // up_weights:   [NL, ID, HD]
    // down_weights: [NL, HD, ID] (output=HD, input=ID)
    const bf16 *gate_w_base = reinterpret_cast<const bf16*>({{ gate_weight_global }}.raw_ptr)
        + (long long)layer * {{ id }} * {{ hd }};
    const bf16 *up_w_base = reinterpret_cast<const bf16*>({{ up_weight_global }}.raw_ptr)
        + (long long)layer * {{ id }} * {{ hd }};
    const bf16 *down_w_base = reinterpret_cast<const bf16*>({{ down_weight_global }}.raw_ptr)
        + (long long)layer * {{ hd }} * {{ id }};

    for (int r = 0; r < my_rows; r++) {
        // Each warp handles a subset of HD output elements
        const int hd_per_warp = {{ hd }} / DEC_NUM_WARPS;
        const int hd_start = wid * hd_per_warp;

        for (int hd_out = hd_start; hd_out < hd_start + hd_per_warp; hd_out += DEC_OUT_BLOCK) {
            float tile_acc[DEC_OUT_BLOCK];
            for (int j = 0; j < DEC_OUT_BLOCK; j++) tile_acc[j] = 0.f;

            for (int id_tile = 0; id_tile < {{ id_col_tiles }}; id_tile++) {
                // Compute gate[out_block] and up[out_block] via dot products
                float gate_vals[DEC_OUT_BLOCK];
                float up_vals[DEC_OUT_BLOCK];

                for (int j = 0; j < DEC_OUT_BLOCK; j++) {
                    float g_dot = 0.f, u_dot = 0.f;
                    int id_elem = id_tile * DEC_OUT_BLOCK + j;
                    // gate_w[id_elem, k] for k in 0..HD, dot with act_row[k]
                    for (int kk = lid; kk < {{ hd }}; kk += 32) {
                        float a_val = __bfloat162float(mlp_act_smem[r * {{ hd }} + kk]);
                        float gw = __bfloat162float(gate_w_base[id_elem * {{ hd }} + kk]);
                        float uw = __bfloat162float(up_w_base[id_elem * {{ hd }} + kk]);
                        g_dot += a_val * gw;
                        u_dot += a_val * uw;
                    }
                    // Warp reduce
                    for (int mask = 16; mask > 0; mask >>= 1) {
                        g_dot += __shfl_xor_sync(0xFFFFFFFF, g_dot, mask);
                        u_dot += __shfl_xor_sync(0xFFFFFFFF, u_dot, mask);
                    }
                    // SiLU(gate) * up
                    float silu_gate = g_dot / (1.f + expf(-g_dot));
                    gate_vals[j] = silu_gate * u_dot;
                }

                // Accumulate into down_acc tile: [out_block] × down_w row slice
                // down_w[hd_out+h, id_tile*out_block+j]
                for (int h = 0; h < DEC_OUT_BLOCK; h++) {
                    float dot = 0.f;
                    int hd_row = hd_out + h;
                    for (int j = lid; j < DEC_OUT_BLOCK; j += 32) {
                        int id_elem = id_tile * DEC_OUT_BLOCK + j;
                        float dw = __bfloat162float(down_w_base[hd_row * {{ id }} + id_elem]);
                        dot += gate_vals[j] * dw;
                    }
                    for (int mask = 16; mask > 0; mask >>= 1)
                        dot += __shfl_xor_sync(0xFFFFFFFF, dot, mask);
                    tile_acc[h] += dot;
                }
            }  // id_tile loop

            // Write tile_acc to GLOBAL with residual add.
            // Cannot write to shmem because the input (mlp_act_smem) shares the same
            // region — writing partial tiles would corrupt the input for subsequent tiles.
            {
                bf16 *h_rw = reinterpret_cast<bf16*>(g.hidden_states.raw_ptr);
                for (int h = lid; h < DEC_OUT_BLOCK; h += 32) {
                    int gidx = (row_start + r) * {{ hd }} + hd_out + h;
                    float res = __bfloat162float(h_rw[gidx]);
                    h_rw[gidx] = __float2bfloat16(tile_acc[h] + res);
                }
            }
        }  // hd_out loop
        __syncthreads();
    }  // row loop
    }
