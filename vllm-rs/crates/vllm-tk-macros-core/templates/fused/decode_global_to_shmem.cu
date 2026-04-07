{# Load g.hidden_states back into shmem hidden region.
   Used after o_proj writes results to global (to avoid input/output shmem overlap).
   Variables:
     shmem_offset: byte offset of the hidden slab in phase_shm
#}
    // ════ Load: g.hidden_states → shmem hidden ════
    {
        bf16 *hidden = reinterpret_cast<bf16*>(phase_shm + {{ shmem_offset }});
        for (int r = wid; r < my_rows; r += DEC_NUM_WARPS) {
            sv_bf<globals::hidden_dim> &row_sv =
                *reinterpret_cast<sv_bf<globals::hidden_dim>*>(hidden + r * globals::hidden_dim);
            warp::load_async(row_sv, g.hidden_states, {row_start + r, 0});
        }
        dec_cp_async_wait_all();
    }
    __syncthreads();
