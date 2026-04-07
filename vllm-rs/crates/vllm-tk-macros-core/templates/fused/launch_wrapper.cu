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
    int grid = (num_prefill_tokens + PFL_CTA_ROWS - 1) / PFL_CTA_ROWS;
    fused_prefill_layer<<<grid, {{ num_threads }}, shmem, (cudaStream_t)stream>>>(
        g, batch_size, num_layers);
    err = cudaGetLastError();
    return (int)err;
  } catch (...) { return -2; }
}
