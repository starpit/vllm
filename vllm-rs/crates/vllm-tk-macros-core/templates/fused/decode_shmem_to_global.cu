{# Write shmem hidden_states back to global g.hidden_states.
   Used mid-layer when a later phase reads residual from global.
   Variables:
     shmem_offset: byte offset of the hidden slab in phase_shm
#}
    // ════ Writeback: shmem hidden → g.hidden_states ════
    {
        bf16 *hidden = reinterpret_cast<bf16*>(phase_shm + {{ shmem_offset }});
        for (int r = wid; r < my_rows; r += DEC_NUM_WARPS) {
            sv_bf<globals::hidden_dim> &row_sv =
                *reinterpret_cast<sv_bf<globals::hidden_dim>*>(hidden + r * globals::hidden_dim);
            warp::store(g.hidden_states, row_sv, {row_start + r, 0});
        }
    }
    __syncthreads();
