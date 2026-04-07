// Bring globals type into file scope for the dispatch wrapper
using namespace kittens;
using namespace kittens::prototype::vm;
using globals = llama_sm89_globals;

{{ tensor_arg_helper }}

extern "C" int fused_prefill_layer_launch(
{{ launch_params }}
) {
  try {
{{ globals_construction }}

    // ── Polyalgorithm dispatch ──
    // Pick kernel variant based on sequence length.
    // Thresholds determined by L40S benchmarks (LLaMA 1B, 16 layers):
    //   seq ≤ 128:  64row-k128  (5.49ms @ seq48, 6.29ms @ seq128)
    //   128 < seq < 1024: 128row-fused (7.97ms @ seq256, 10.94ms @ seq512)
    //   seq ≥ 1024: 128row-wide (16.82ms @ seq1024)

    if (num_prefill_tokens <= 128) {
        return pfl_small::fused_prefill_layer_small_launch_inner(
            g, batch_size, num_layers, num_prefill_tokens, (cudaStream_t)stream);
    } else if (num_prefill_tokens < 1024) {
        return pfl_medium::fused_prefill_layer_medium_launch_inner(
            g, batch_size, num_layers, num_prefill_tokens, (cudaStream_t)stream);
    } else {
        return pfl_large::fused_prefill_layer_large_launch_inner(
            g, batch_size, num_layers, num_prefill_tokens, (cudaStream_t)stream);
    }
  } catch (...) { return -2; }
}
