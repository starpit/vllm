# Decode Baseline: Llama-3.2-1B on L40S (sm89)

Collected 2026-04-07 on nick3 pod (2x L40S, single GPU used).
Model: unsloth/Llama-3.2-1B (16 layers, hidden=2048, 8 heads, head_dim=64).

## Per-token decode latency (BS=1)

| Backend                    | 64 tok | 128 tok | 256 tok |
|----------------------------|--------|---------|---------|
| Rust feat/rust (eager)     | 4.16   | 4.09    | 4.06    |
| Rust feat/rust (graphs)    | 4.02   | 3.92    | 3.90    |
| Python vLLM (eager)        | 6.09   | 6.13    | 6.17    |
| Python vLLM (graphs)       | 3.92   | 3.90    | 3.90    |

Units: ms/tok. Lower is better.

## Batched decode throughput (128 tok/seq)

| Backend                    | BS=1     | BS=8      | BS=32     |
|----------------------------|----------|-----------|-----------|
| Rust feat/rust (eager)     | 243 t/s  | 1,793 t/s | 6,639 t/s |
| Rust feat/rust (graphs)    | 254 t/s  | 1,876 t/s | 6,759 t/s |
| Python vLLM (eager)        | 162 t/s  | 1,196 t/s | 4,496 t/s |
| Python vLLM (graphs)       | 256 t/s  | 1,912 t/s | 7,158 t/s |

Units: tokens/sec. Higher is better.
Note: Python eager numbers from offline LLM.generate() API (inflated overhead).
Server-path eager is ~4.0 ms/tok, not 6.1 — multiprocess arch hides Python overhead.

## Server-side timing (Rust feat/rust, graphs, BS=32)

- TTFT: 6.2ms
- Avg ITL: 4.6ms/tok
- Total latency: 594.8ms for 128 decode steps

## nsys kernel breakdown (Rust, eager, BS=32, 32 output tokens, 5 iters)

Profiled with: `nsys profile --trace=cuda vllm bench latency ... --enforce-eager`

### GPU kernel time by category

| Category              | Time (ms) | % of GPU | Key kernels                                      |
|-----------------------|-----------|----------|--------------------------------------------------|
| GEMM (cuBLAS)         | 2,011     | 88.0%    | ampere_bf16_s16816gemm (various tile configs)    |
| FlashAttention        | 137       | 6.0%     | flash_fwd_splitkv_kernel, flash_fwd_kernel       |
| Fused norms           | 35        | 1.5%     | fused_add_rms_norm_kernel, rms_norm_kernel       |
| SiLU+mul              | 25        | 1.1%     | act_and_mul_fused_kernel                         |
| Sampling              | 18        | 0.8%     | sample_gumbel_phase1/2                           |
| RoPE + QKV + KV cache | 22        | 1.0%     | fused_qkv_rope_cache, reshape_and_cache, etc.   |
| Other (transpose etc) | 20        | 0.9%     | ngroups_transpose/untranspose, split_qkv, embed  |

Total GPU kernel time: ~2,290ms across 5 iterations.

### CPU-side CUDA API overhead

| API call              | Total (ms) | Calls  | Avg (μs) |
|-----------------------|------------|--------|----------|
| cuMemcpyDtoHAsync     | 1,710      | 480    | 3,563    |
| cudaLaunchKernel_ptsz | 196        | 55,809 | 3.5      |
| cuStreamSynchronize   | 114        | 4      | 28,415   |
| cuMemFree             | 109        | 161    | 675      |
| cudaLaunchKernel      | 87         | 23,504 | 3.7      |
| cuLaunchKernel        | 75         | 22,865 | 3.3      |

Total kernel launch CPU overhead: ~358ms out of ~790ms wall clock = ~45% in eager mode.

### Top GEMM kernels

| Kernel                                    | Time (ms) | Instances | Avg (μs) | Likely role          |
|-------------------------------------------|-----------|-----------|----------|----------------------|
| ampere_bf16_s16816gemm_64x128_ldg8_3stg   | 782       | 7,440     | 105      | down_proj (large)    |
| cutlass_wmma_bf16_32x32_128x2             | 672       | 22,320    | 30       | Small decode GEMMs   |
| ampere_bf16_s1688gemm_128x128_ldg8_1stg   | 405       | 704       | 575      | Prefill GEMMs        |
| cutlass_relu_bf16_256x128_32x3            | 105       | 289       | 363      | Prefill GEMMs        |

## Key findings

1. **CUDA graphs give Rust only ~4% on L40S** (4.06 → 3.90 ms/tok) because Rust
   launch overhead is already low. Python gets ~36% from graphs (6.1 → 3.9 in
   offline API) but only ~3-4% in actual server path (multiprocess hides overhead).

2. **GEMM is 88% of GPU kernel time** at BS=32 decode. These are memory-bound
   (loading full weight matrices for [32, 2048] activations). Neither megakernel
   nor CUDA graphs can make these faster — HBM bandwidth is the floor.

3. **Inter-op fusion opportunity is ~3.6% of GPU time** (norms + SiLU + RoPE/cache).
   This is what a megakernel saves beyond what CUDA graphs provide.

4. **Kernel launch overhead is ~45% of wall time in eager mode.** CUDA graphs
   eliminate this. A megakernel also eliminates this, plus the 3.6% fusion bonus.

5. **The megakernel value proposition for sm89 decode:** replace CUDA graphs
   (eliminating graph capture overhead, memory overhead, static-shape constraints)
   while also fusing inter-op traffic for a few % bonus. On sm90+ with cluster
   shared memory, the fusion bonus grows substantially.
