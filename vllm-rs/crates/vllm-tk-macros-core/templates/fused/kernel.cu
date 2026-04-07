{# Top-level kernel template. Composes preamble + kernel body + launch wrapper.
   Variables:
     preamble: rendered preamble.cu
     phases: list of rendered phase strings (rmsnorm/gemm/rope/attention)
     launch_wrapper: rendered launch_wrapper.cu
     num_threads: kernel thread count
#}
{{ preamble }}

__global__ void __launch_bounds__({{ num_threads }}, 1)
fused_prefill_layer(const globals g, int batch_size, int num_layers) {
    const int wid = kittens::warpid();
    const int lid = kittens::laneid();
    const int bid = blockIdx.x;
    extern __shared__ char __shm[];

    // Per-phase timing (CTA 0, warp 0, lane 0 only)
    constexpr int NUM_PHASES = {{ num_phases }};
    long long phase_clocks[NUM_PHASES + 1];

    for (int layer = 0; layer < num_layers; layer++) {

    const int seq_idx = 0;
    const int q_start = g.prefill_qo_indptr[{seq_idx}];
    const int q_end = g.prefill_qo_indptr[{seq_idx + 1}];
    const int q_size = q_end - q_start;
    const int rel_q_row = PFL_CTA_ROWS * bid;
    const int rel_q_row_last = min(rel_q_row + PFL_CTA_ROWS - 1, q_size - 1);
    if (rel_q_row >= q_size) return;
    const int abs_q_row = rel_q_row + q_start;

    if (bid == 0 && wid == 0 && lid == 0 && layer == 0)
        phase_clocks[0] = clock64();

{% for phase in phases -%}
{{ phase }}
    if (bid == 0 && wid == 0 && lid == 0 && layer == 0)
        phase_clocks[{{ loop.index }}] = clock64();

{% endfor %}
    }  // end layer loop

    // Print phase timing for layer 0
    if (bid == 0 && wid == 0 && lid == 0) {
        long long total = phase_clocks[NUM_PHASES] - phase_clocks[0];
        printf("PREFILL PHASE TIMING (layer 0, %d phases, total=%lld clocks):\n", NUM_PHASES, total);
        const char *phase_names[] = { {{ phase_names_str }} };
        for (int i = 0; i < NUM_PHASES; i++) {
            long long dt = phase_clocks[i+1] - phase_clocks[i];
            printf("  phase %d (%s): %lld clocks (%.1f%%)\n", i, phase_names[i], dt, 100.0 * (double)dt / (double)total);
        }
    }
}  // end fused_prefill_layer

{{ launch_wrapper }}
