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
    // Thresholds from L4 benchmarks (LLaMA 1B, 16 layers, round-5 dual_accum sweep):
    //   seq ≤ 64:   rows64_k128                 (10.86 ms @ seq48)
    //                 Classic 64-row CTA with k_dim=128 (half the K iterations,
    //                 half the per-iter barrier overhead). Optimal for tiny work
    //                 where setup cost dominates over compute.
    //   64 < seq < 256: rows128_gemm16_dual_1stage (12.97/~13 ms @ seq=128/192, 1.40×)
    //                 Dual-accumulator gate+up, 1-stage pipeline, 2 CTAs/SM,
    //                 8 warps × 16 rows.
    //   seq ≥ 256: rows256_gemm32_dual_1stage    (16.94/30.56/55.63 ms @ 256/512/1024,
    //                                             1.34-1.42×)
    //                 8 warps × 32 rows cooperative, dual-acc, 1-stage, 1 CTA/SM;
    //                 half the mcta_barrier trips of rows128 + A reuse in gate+up.

    if (num_prefill_tokens <= 64) {
        return pfl_small::fused_prefill_layer_small_launch_inner(
            g, batch_size, num_layers, num_prefill_tokens, (cudaStream_t)stream);
    } else if (num_prefill_tokens < 256) {
        return pfl_medium::fused_prefill_layer_medium_launch_inner(
            g, batch_size, num_layers, num_prefill_tokens, (cudaStream_t)stream);
    } else {
        return pfl_large::fused_prefill_layer_large_launch_inner(
            g, batch_size, num_layers, num_prefill_tokens, (cudaStream_t)stream);
    }
  } catch (...) { return -2; }
}
