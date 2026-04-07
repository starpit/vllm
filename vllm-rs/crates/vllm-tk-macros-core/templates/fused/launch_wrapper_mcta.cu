{{ tensor_arg_helper }}

extern "C" int fused_prefill_layer_launch(
{{ launch_params }}
) {
  try {
{{ globals_construction }}

    int shmem = PFL_SHMEM;
    auto err = cudaFuncSetAttribute(fused_prefill_layer,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    if (err != cudaSuccess) return (int)err;

    constexpr int grid = {{ grid_size }};
    constexpr int num_phases = MCTA_NUM_PHASES;

    // Allocate cross-CTA barrier array (one int per (layer, phase))
    int *mcta_bar = nullptr;
    size_t bar_bytes = sizeof(int) * num_layers * num_phases;
    err = cudaMallocAsync(&mcta_bar, bar_bytes, (cudaStream_t)stream);
    if (err != cudaSuccess) return (int)err;
    err = cudaMemsetAsync(mcta_bar, 0, bar_bytes, (cudaStream_t)stream);
    if (err != cudaSuccess) { cudaFreeAsync(mcta_bar, (cudaStream_t)stream); return (int)err; }

    fused_prefill_layer<<<grid, {{ num_threads }}, shmem, (cudaStream_t)stream>>>(
        g, batch_size, num_layers, mcta_bar);
    err = cudaGetLastError();
    cudaFreeAsync(mcta_bar, (cudaStream_t)stream);
    return (int)err;
  } catch (...) { return -2; }
}
