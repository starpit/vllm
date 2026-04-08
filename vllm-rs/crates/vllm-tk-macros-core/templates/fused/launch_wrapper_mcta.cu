{{ tensor_arg_helper }}

extern "C" int fused_prefill_layer{{ kernel_suffix }}_launch(
{{ launch_params }}
) {
  try {
{{ globals_construction }}

    int shmem = PFL_SHMEM;
    auto err = cudaFuncSetAttribute(fused_prefill_layer{{ kernel_suffix }},
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    if (err != cudaSuccess) return (int)err;

    // ── Polyalgorithm grid sizing ──
    // Pick grid to match workload: enough CTAs to cover the largest GEMM phase.
    // The largest GEMM is gate/up/down with {{ id_col_tiles }} col tiles.
    //
    // Cap at the number of CTAs that can actually be resident simultaneously.
    // The cross-CTA spin-wait barrier deadlocks if we launch more CTAs than fit
    // (e.g. MCTA_MAX_GRID=128 on L4's 58 SMs with 1 CTA/SM occupancy).
    int max_resident_per_sm = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_resident_per_sm,
        fused_prefill_layer{{ kernel_suffix }},
        {{ num_threads }}, shmem);
    if (err != cudaSuccess) return (int)err;
    int device_id = 0;
    cudaGetDevice(&device_id);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
    const int occ_cap = max_resident_per_sm * sm_count;
    const int max_grid = min((int)MCTA_MAX_GRID, occ_cap > 0 ? occ_cap : (int)MCTA_MAX_GRID);
    const int row_tiles = (num_prefill_tokens + PFL_CTA_ROWS - 1) / PFL_CTA_ROWS;
    // Largest GEMM work = row_tiles × id_col_tiles (e.g. 8 × 64 = 512 for seq=1024)
    constexpr int id_col_tiles = {{ id_col_tiles }};
    const int max_work = row_tiles * id_col_tiles;
    // Use enough CTAs so each does ≤2 work units for the biggest GEMM
    int grid = min(max_grid, max(1, (max_work + 1) / 2));
    // At minimum, cover id_col_tiles (even with 1 row_tile, we want 1 CTA per col)
    grid = max(grid, min(max_grid, id_col_tiles));
    // Also cover row_tiles for rmsnorm/rope distribution
    grid = max(grid, min(max_grid, row_tiles));

    constexpr int num_phases = MCTA_NUM_PHASES;
    int *mcta_bar = nullptr;
    size_t bar_bytes = sizeof(int) * num_layers * num_phases;
    err = cudaMallocAsync(&mcta_bar, bar_bytes, (cudaStream_t)stream);
    if (err != cudaSuccess) return (int)err;
    err = cudaMemsetAsync(mcta_bar, 0, bar_bytes, (cudaStream_t)stream);
    if (err != cudaSuccess) { cudaFreeAsync(mcta_bar, (cudaStream_t)stream); return (int)err; }

    fused_prefill_layer{{ kernel_suffix }}<<<grid, {{ num_threads }}, shmem, (cudaStream_t)stream>>>(
        g, batch_size, num_layers, mcta_bar);
    err = cudaGetLastError();
    cudaFreeAsync(mcta_bar, (cudaStream_t)stream);
    return (int)err;
  } catch (...) { return -2; }
}
