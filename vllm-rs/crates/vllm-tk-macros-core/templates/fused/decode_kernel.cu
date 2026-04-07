{# Top-level decode kernel: row-fused architecture.
   Each CTA owns cta_rows sequences through ALL phases of ALL layers.
   No cross-CTA barriers. Activations stay in shmem.

   Variables:
     preamble: rendered decode_preamble.cu
     phases: list of rendered phase strings
     launch_wrapper: rendered decode_launch_wrapper.cu
     num_threads: kernel thread count
     num_phases: phase count per layer
     phase_names_str: C string array for timing
     cta_rows: sequences per CTA
     padded_cta_rows: MMA-aligned rows
     hidden_shmem: bytes for hidden_states slab in shmem
     meta_shmem: bytes for per-row metadata
#}
{{ preamble }}

__global__ void __launch_bounds__({{ num_threads }}, 1)
fused_decode_layer(const globals g, int batch_size, int num_layers) {
    const int wid = kittens::warpid();
    const int lid = kittens::laneid();
    const int bid = blockIdx.x;
    extern __shared__ char __shm[];

    // ── Shmem layout ──
    // [0 .. meta_shmem): per-row metadata (persistent across all layers)
    // [meta_shmem .. meta_shmem + hidden_shmem): hidden_states slab (reused per phase)
    // The hidden region is time-shared: RMSNorm, GEMM, and attention all use it.
    DecRowMeta *row_meta = reinterpret_cast<DecRowMeta*>(__shm);
    char *phase_shm = __shm + DEC_META_SHMEM;

    // ── Range check ──
    const int row_start = bid * DEC_CTA_ROWS;
    if (row_start >= batch_size) return;
    const int my_rows = min(DEC_CTA_ROWS, batch_size - row_start);

    // ── Load per-row metadata into shmem (once, before layer loop) ──
    if (wid == 0) {
        for (int r = lid; r < my_rows; r += 32) {
            row_meta[r].position_id    = g.decode_positions[{row_start + r}];
            row_meta[r].kv_indptr_start = g.decode_kv_indptr[{row_start + r}];
            row_meta[r].kv_indptr_end   = g.decode_kv_indptr[{row_start + r + 1}];
            row_meta[r].kv_last_page_len = g.decode_kv_last_page_len[{row_start + r}];
            row_meta[r].kv_append_slot  = g.decode_kv_indptr[{row_start + r + 1}] - 1;
        }
    }
    __syncthreads();

    // ── Load initial hidden_states into shmem ──
    // hidden_shmem region: [padded_rows, HD] BF16
    {
        bf16 *hidden = reinterpret_cast<bf16*>(phase_shm);
        // Each warp loads one row at a time
        for (int r = wid; r < my_rows; r += DEC_NUM_WARPS) {
            sv_bf<globals::hidden_dim> &row_sv =
                *reinterpret_cast<sv_bf<globals::hidden_dim>*>(hidden + r * globals::hidden_dim);
            warp::load_async(row_sv, g.hidden_states, {row_start + r, 0});
        }
        dec_cp_async_wait_all();
    }
    __syncthreads();

    // Per-phase timing (CTA 0, warp 0, lane 0 only)
    constexpr int NUM_PHASES = {{ num_phases }};
    long long phase_clocks[NUM_PHASES + 1];

    for (int layer = 0; layer < num_layers; layer++) {

    if (bid == 0 && wid == 0 && lid == 0 && layer == 0)
        phase_clocks[0] = clock64();

{% for phase in phases -%}
{{ phase }}
    if (bid == 0 && wid == 0 && lid == 0 && layer == 0)
        phase_clocks[{{ loop.index }}] = clock64();

{% endfor %}
    }  // end layer loop

    // ── Write final hidden_states back to global ──
    {
        bf16 *hidden = reinterpret_cast<bf16*>(phase_shm);
        for (int r = wid; r < my_rows; r += DEC_NUM_WARPS) {
            sv_bf<globals::hidden_dim> &row_sv =
                *reinterpret_cast<sv_bf<globals::hidden_dim>*>(hidden + r * globals::hidden_dim);
            warp::store(g.hidden_states, row_sv, {row_start + r, 0});
        }
    }
    __threadfence();

    // Print phase timing for layer 0
    if (bid == 0 && wid == 0 && lid == 0) {
        long long total = phase_clocks[NUM_PHASES] - phase_clocks[0];
        printf("DECODE PHASE TIMING (layer 0, %d phases, total=%lld clocks):\n", NUM_PHASES, total);
        const char *phase_names[] = { {{ phase_names_str }} };
        for (int i = 0; i < NUM_PHASES; i++) {
            long long dt = phase_clocks[i+1] - phase_clocks[i];
            printf("  phase %d (%s): %lld clocks (%.1f%%)\n", i, phase_names[i], dt, 100.0 * (double)dt / (double)total);
        }
    }
}  // end fused_decode_layer

{{ launch_wrapper }}
