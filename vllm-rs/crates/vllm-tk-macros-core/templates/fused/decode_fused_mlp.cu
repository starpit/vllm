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
#}
    // ════ Decode fused MLP (streaming gate+up+SiLU+down, no [R,ID] materialization) ════
    {
    bf16 *mlp_act_smem = reinterpret_cast<bf16*>(phase_shm + {{ input_shmem_offset }});
    bf16 *hidden_smem = reinterpret_cast<bf16*>(phase_shm + {{ hidden_shmem_offset }});

    // Per-row accumulators for down_proj output [HD]. Each warp processes one row.
    // We accumulate across all ID tiles in registers.
    // Since HD can be large (2048), we tile the down accumulation.
    // Strategy: for each ID col tile, compute gate+up+SiLU, then immediately
    // multiply by the corresponding down_w row slice and accumulate.

    // We'll use a simpler row-sequential approach:
    // For each row, accumulate down_acc[HD] across all ID tiles.
    for (int r = 0; r < my_rows; r++) {
        // down_acc accumulates [1, HD] — but HD=2048 is too many registers.
        // Instead, we loop over HD output tiles and ID input tiles.
        // For each hd_out_tile in 0..hd_col_tiles:
        //   acc = 0
        //   for each id_tile in 0..id_col_tiles:
        //     gate_val = dot(act_row, gate_w[id_tile][hd_out_tile's k-slice])  -- wrong, gate has [HD, ID] shape
        //
        // Actually: gate_w is [ID, HD], up_w is [ID, HD], down_w is [HD, ID]
        // gate_out[id] = sum_k(act[k] * gate_w[id, k])
        // up_out[id]   = sum_k(act[k] * up_w[id, k])
        // mlp_out[hd]  = sum_id(SiLU(gate_out[id]) * up_out[id] * down_w[hd, id])
        //
        // We stream over groups of out_block ID elements:
        //   For id_tile in 0..id_col_tiles:
        //     gate_acc[out_block] = GEMM(act_row[HD], gate_w[layer, id_tile]) -- [1, out_block]
        //     up_acc[out_block]   = GEMM(act_row[HD], up_w[layer, id_tile])   -- [1, out_block]
        //     fused[out_block]    = SiLU(gate_acc) * up_acc
        //     -- Now fused is [out_block] and we need: down_acc[hd] += sum(fused[id] * down_w[hd, id])
        //     -- But down_w is [HD, ID] tiled as [hd_col_tiles, id_col_tiles, k_dim, out_block]
        //     -- Actually down_w[layer, hd_tile, id_k_iter] if viewed as GEMM(fused, down_w^T)
        //
        // Simpler: compute full gate_out and up_out for this id_tile block into shmem/registers,
        // then do a small GEMM: [1, out_block] × down_w_tile[out_block, k_dim] for each HD k-iter.
        //
        // This is getting complex. For now, use a scalar approach for correctness:
        // Compute gate and up via warp-parallel dot products, accumulate down in registers.

        // Scalar approach: each warp handles a subset of output HD elements
        const int hd_per_warp = globals::hidden_dim / DEC_NUM_WARPS;
        const int hd_start = wid * hd_per_warp;

        // Per-warp accumulator for down_proj partial output
        float down_acc[64]; // max hd_per_warp = 2048/8 = 256... too big for registers
        // Use a tiled approach instead: accumulate out_block at a time
        // Actually with 256 floats = 1KB per warp, that's 8KB total — fits.
        // But it's better to go tile by tile.

        for (int hd_out = hd_start; hd_out < hd_start + hd_per_warp; hd_out += DEC_OUT_BLOCK) {
            float tile_acc[DEC_OUT_BLOCK];
            for (int j = 0; j < DEC_OUT_BLOCK; j++) tile_acc[j] = 0.f;

            for (int id_tile = 0; id_tile < {{ id_col_tiles }}; id_tile++) {
                // Compute gate[out_block] and up[out_block] via dot products
                float gate_vals[DEC_OUT_BLOCK];
                float up_vals[DEC_OUT_BLOCK];

                for (int j = 0; j < DEC_OUT_BLOCK; j++) {
                    float g_dot = 0.f, u_dot = 0.f;
                    // id_elem = id_tile * out_block + j
                    // gate_w[layer, id_tile * out_block + j, :] dot act_row[:]
                    // These are accessed as g.gate_weights[{layer, col_tile, k_iter}]
                    // which is a [out_block, k_dim] tile at [layer][col][k]
                    // We need to sum over all k_iters for this (id_tile, j) pair.
                    for (int k_iter = 0; k_iter < {{ hd_k_iters }}; k_iter++) {
                        for (int kk = lid; kk < DEC_K_DIM; kk += 32) {
                            int hd_idx = k_iter * DEC_K_DIM + kk;
                            float a_val = __bfloat162float(mlp_act_smem[r * globals::hidden_dim + hd_idx]);
                            float gw = __bfloat162float({{ gate_weight_global }}[coord<>{layer, id_tile, k_iter, j * DEC_K_DIM + kk}]);
                            float uw = __bfloat162float({{ up_weight_global }}[coord<>{layer, id_tile, k_iter, j * DEC_K_DIM + kk}]);
                            g_dot += a_val * gw;
                            u_dot += a_val * uw;
                        }
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

                // Accumulate into down_acc tile: [out_block] × down_w[hd_out_tile, id_tile]
                // down_w is [HD, ID], so down_w[hd_out+h, id_tile*out_block+j]
                // Accessed as down_weights[{layer, hd_col, id_k_iter}] — [out_block, k_dim] tiles
                // We need: for each h in 0..out_block:
                //   tile_acc[h] += sum_j(gate_vals[j] * down_w[hd_out+h, id_tile*out_block+j])
                int hd_col = hd_out / DEC_OUT_BLOCK;
                for (int h = 0; h < DEC_OUT_BLOCK; h++) {
                    float dot = 0.f;
                    for (int j = lid; j < DEC_OUT_BLOCK; j += 32) {
                        // down_w[hd_out+h, id_tile*out_block+j]
                        // This is in down_weights[{layer, hd_col, id_tile}] tile at [h, j]
                        // But the tile layout is [out_block, k_dim] = [64, 64]
                        // where the k_dim iterates over the ID dimension
                        // Actually, for down_proj: input is [ID], output is [HD]
                        // So: out[hd] = sum_id(in[id] * down_w[hd, id])
                        // Tiled: down_w[{layer, hd_col, id_k_iter}] where id_k_iter = id_tile
                        float dw = __bfloat162float({{ down_weight_global }}[coord<>{layer, hd_col, id_tile, h * DEC_K_DIM + j}]);
                        dot += gate_vals[j] * dw;
                    }
                    for (int mask = 16; mask > 0; mask >>= 1)
                        dot += __shfl_xor_sync(0xFFFFFFFF, dot, mask);
                    tile_acc[h] += dot;
                }
            }  // id_tile loop

            // Write tile_acc to hidden_shmem with residual add
            for (int h = lid; h < DEC_OUT_BLOCK; h += 32) {
                float res = __bfloat162float(hidden_smem[r * globals::hidden_dim + hd_out + h]);
                hidden_smem[r * globals::hidden_dim + hd_out + h] =
                    __float2bfloat16(tile_acc[h] + res);
            }
        }  // hd_out loop
        __syncthreads();
    }  // row loop
    }
