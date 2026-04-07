{# Multi-CTA prefill kernel: all CTAs collaborate on each phase.
   Variables:
     preamble: rendered preamble
     num_threads: threads per CTA
     num_phases: number of phases per layer
     phase_names_str: C string array for timing
     grid_size: max grid size (launch wrapper picks actual grid at runtime)
     phases: list of rendered phase strings
     launch_wrapper: rendered launch wrapper
#}
{{ preamble }}

// ── Cross-CTA barrier helpers ──────────────────────────────────────────
// barrier array: [num_layers * num_phases] ints, zero-initialized before launch.
// Each (layer, phase) pair uses a unique slot — no reset needed.
// Grid size is dynamic (gridDim.x), not constexpr.
constexpr int MCTA_NUM_PHASES = {{ num_phases }};
constexpr int MCTA_MAX_GRID = {{ grid_size }};

__device__ static inline void mcta_barrier(int *bar, int layer, int phase, int grid) {
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        int idx = layer * MCTA_NUM_PHASES + phase;
        atomicAdd(&bar[idx], 1);
        volatile int *vbar = (volatile int*)&bar[idx];
        while (*vbar < grid) {}
    }
    __syncthreads();
}

__global__ void __launch_bounds__({{ num_threads }}, 1)
fused_prefill_layer(const globals g, int batch_size, int num_layers, int *mcta_bar) {
    const int wid = kittens::warpid();
    const int lid = kittens::laneid();
    const int bid = blockIdx.x;
    const int num_ctas = gridDim.x;
    extern __shared__ char __shm[];

    // Per-phase timing (CTA 0, warp 0, lane 0 only)
    constexpr int NUM_PHASES = {{ num_phases }};
    long long phase_clocks[NUM_PHASES + 1];

    for (int layer = 0; layer < num_layers; layer++) {

    const int seq_idx = 0;
    const int q_start = g.prefill_qo_indptr[{seq_idx}];
    const int q_end = g.prefill_qo_indptr[{seq_idx + 1}];
    const int q_size = q_end - q_start;
    const int row_tiles = (q_size + PFL_Q_ROWS - 1) / PFL_Q_ROWS;

    if (bid == 0 && wid == 0 && lid == 0 && layer == 0)
        phase_clocks[0] = clock64();

{% for phase in phases -%}
{{ phase }}
    mcta_barrier(mcta_bar, layer, {{ loop.index0 }}, num_ctas);
    if (bid == 0 && wid == 0 && lid == 0 && layer == 0)
        phase_clocks[{{ loop.index }}] = clock64();

{% endfor %}
    }  // end layer loop

    // Print phase timing for layer 0
    if (bid == 0 && wid == 0 && lid == 0) {
        long long total = phase_clocks[NUM_PHASES] - phase_clocks[0];
        printf("MCTA PHASE TIMING (layer 0, %d phases, grid=%d, total=%lld clocks):\n",
               NUM_PHASES, num_ctas, total);
        const char *phase_names[] = { {{ phase_names_str }} };
        for (int i = 0; i < NUM_PHASES; i++) {
            long long dt = phase_clocks[i+1] - phase_clocks[i];
            printf("  phase %d (%s): %lld clocks (%.1f%%)\n", i, phase_names[i], dt, 100.0 * (double)dt / (double)total);
        }
    }
}  // end fused_prefill_layer

{{ launch_wrapper }}
