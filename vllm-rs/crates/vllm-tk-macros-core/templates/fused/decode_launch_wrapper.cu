{{ tensor_arg_helper }}

extern "C" int fused_decode_layer_launch(
{{ launch_params }}
) {
  try {
{{ globals_construction }}

    int shmem = DEC_PEAK_SHMEM;
    auto err = cudaFuncSetAttribute(fused_decode_layer,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    if (err != cudaSuccess) return (int)err;
    int grid = (batch_size + DEC_CTA_ROWS - 1) / DEC_CTA_ROWS;
    fused_decode_layer<<<grid, {{ num_threads }}, shmem, (cudaStream_t)stream>>>(
        g, batch_size, num_layers);
    err = cudaGetLastError();
    return (int)err;
  } catch (...) { return -2; }
}
