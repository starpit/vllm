static inline int fused_prefill_layer{{ kernel_suffix }}_launch_inner(
    const globals &g, int batch_size, int num_layers, int num_prefill_tokens, cudaStream_t stream) {

    int shmem = PFL_SHMEM;
    auto err = cudaFuncSetAttribute(fused_prefill_layer{{ kernel_suffix }},
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    if (err != cudaSuccess) return (int)err;

    constexpr int max_grid = MCTA_MAX_GRID;
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
