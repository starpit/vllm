static inline int fused_prefill_layer{{ kernel_suffix }}_launch_inner(
    const globals &g, int batch_size, int num_layers, int num_prefill_tokens, cudaStream_t stream) {

    int shmem = PFL_SHMEM;
    auto err = cudaFuncSetAttribute(fused_prefill_layer{{ kernel_suffix }},
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    if (err != cudaSuccess) return (int)err;

    // Cap grid at the number of CTAs that can be concurrently resident.
    // The kernel uses a spin-wait cross-CTA barrier — if we launch more CTAs
    // than fit resident at once, the late CTAs never enter execution while
    // the resident CTAs spin forever → deadlock. This hits L4 (58 SMs) when
    // MCTA_MAX_GRID was tuned for L40S (142 SMs).
    int max_resident_per_sm = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_resident_per_sm,
        fused_prefill_layer{{ kernel_suffix }},
        {{ num_threads }}, shmem);
    if (err != cudaSuccess) return (int)err;
    int device = 0;
    cudaGetDevice(&device);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    const int occ_cap = max_resident_per_sm * sm_count;
    const int max_grid = min((int)MCTA_MAX_GRID, occ_cap > 0 ? occ_cap : (int)MCTA_MAX_GRID);

    const int row_tiles = (num_prefill_tokens + PFL_CTA_ROWS - 1) / PFL_CTA_ROWS;
    constexpr int id_col_tiles = {{ id_col_tiles }};
    const int max_work = row_tiles * id_col_tiles;
    int grid = min(max_grid, max(1, (max_work + 1) / 2));
    grid = max(grid, min(max_grid, id_col_tiles));
    grid = max(grid, min(max_grid, row_tiles));

    constexpr int num_phases = MCTA_NUM_PHASES;
    int *mcta_bar = nullptr;
    size_t bar_bytes = sizeof(int) * num_layers * num_phases;
    err = cudaMallocAsync(&mcta_bar, bar_bytes, stream);
    if (err != cudaSuccess) return (int)err;
    err = cudaMemsetAsync(mcta_bar, 0, bar_bytes, stream);
    if (err != cudaSuccess) { cudaFreeAsync(mcta_bar, stream); return (int)err; }

    fused_prefill_layer{{ kernel_suffix }}<<<grid, {{ num_threads }}, shmem, stream>>>(
        g, batch_size, num_layers, mcta_bar);
    err = cudaGetLastError();
    cudaFreeAsync(mcta_bar, stream);
    return (int)err;
}
