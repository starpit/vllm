  Gap Analysis: CudaWorker vs CandleWorker

  1. Model Architectures

  CudaWorker has 7 (LLaMA, Mistral, Qwen2, Qwen3, Phi-3, Gemma2, Gemma3) vs CandleWorker has 15+:

  ┌──────────────────────────┬──────────────┬────────────┬────────────────────────────────────┐
  │       Architecture       │ CandleWorker │ CudaWorker │               Notes                │
  ├──────────────────────────┼──────────────┼────────────┼────────────────────────────────────┤
  │ LLaMA/LLaMA2/LLaMA3      │ yes          │ yes        │                                    │
  ├──────────────────────────┼──────────────┼────────────┼────────────────────────────────────┤
  │ Mistral                  │ yes          │ yes        │ LLaMA alias in CudaWorker           │
  ├──────────────────────────┼──────────────┼────────────┼────────────────────────────────────┤
  │ Qwen2/Qwen2.5            │ yes          │ yes        │                                    │
  ├──────────────────────────┼──────────────┼────────────┼────────────────────────────────────┤
  │ Qwen3                    │ yes          │ yes        │ LLaMA alias, E2E verified           │
  ├──────────────────────────┼──────────────┼────────────┼────────────────────────────────────┤
  │ Phi-3/Phi-4              │ yes          │ yes        │ LLaMA alias in CudaWorker           │
  ├──────────────────────────┼──────────────┼────────────┼────────────────────────────────────┤
  │ Gemma2                   │ yes          │ yes        │                                    │
  ├──────────────────────────┼──────────────┼────────────┼────────────────────────────────────┤
  │ Gemma3 (text)            │ yes          │ yes        │ CUDA graphs supported               │
  ├──────────────────────────┼──────────────┼────────────┼────────────────────────────────────┤
  │ DeepSeek V2/V3 (MLA+MoE) │ yes          │ no         │ Complex: MLA attention, MoE gating │
  ├──────────────────────────┼──────────────┼────────────┼────────────────────────────────────┤
  │ Command R                │ yes          │ no         │                                    │
  ├──────────────────────────┼──────────────┼────────────┼────────────────────────────────────┤
  │ Qwen2 MoE                │ yes          │ no         │ MoE routing                        │
  ├──────────────────────────┼──────────────┼────────────┼────────────────────────────────────┤
  │ Qwen3 MoE                │ yes          │ no         │ MoE routing                        │
  ├──────────────────────────┼──────────────┼────────────┼────────────────────────────────────┤
  │ Mixtral (MoE)            │ yes          │ no         │ MoE routing                        │
  ├──────────────────────────┼──────────────┼────────────┼────────────────────────────────────┤
  │ Granite                  │ yes          │ yes        │ LLaMA + 4 scalar multipliers       │
  ├──────────────────────────┼──────────────┼────────────┼────────────────────────────────────┤
  │ Kimi K2.5                │ yes          │ no         │ DeepSeek V2 backbone               │
  ├──────────────────────────┼──────────────┼────────────┼────────────────────────────────────┤
  │ Qwen3-Next (GDN+MoE)     │ yes          │ no         │ Hybrid linear+full attention       │
  └──────────────────────────┴──────────────┴────────────┴────────────────────────────────────┘

  Effort: ~~Mistral/Qwen3/Phi-3 are trivial (alias LLaMA)~~ — DONE. Granite is similar. MoE models (DeepSeek, Mixtral, Qwen MoE) need a fused MoE GEMM
  kernel or scatter-gather approach. DeepSeek MLA is the hardest.

  2. Quantization Support

  CudaWorker has zero quantization vs CandleWorker:

  ┌──────────────────┬──────────────┬────────────┬────────────────────────────────────────────────┐
  │      Format      │ CandleWorker │ CudaWorker │                 Kernel needed                  │
  ├──────────────────┼──────────────┼────────────┼────────────────────────────────────────────────┤
  │ GGUF (k-quants)  │ yes          │ no         │ Dequant kernels or candle's QCudaStorage       │
  ├──────────────────┼──────────────┼────────────┼────────────────────────────────────────────────┤
  │ GPTQ (W4A16)     │ yes          │ no         │ Marlin kernel (already exists in vllm-kernels) │
  ├──────────────────┼──────────────┼────────────┼────────────────────────────────────────────────┤
  │ AWQ (W4A16)      │ yes          │ no         │ Marlin kernel (already exists in vllm-kernels) │
  ├──────────────────┼──────────────┼────────────┼────────────────────────────────────────────────┤
  │ BitsAndBytes NF4 │ yes          │ no         │ NF4 dequant kernel                             │
  └──────────────────┴──────────────┴────────────┴────────────────────────────────────────────────┘

  Effort: Marlin is already compiled in vllm-kernels/csrc/ — need FFI bindings from GpuTensor and quantized weight loading in GpuWeights.
  GGUF requires either porting candle's QCudaStorage approach or writing dequant-on-the-fly kernels. BnB is lower priority.

  3. Tensor Parallelism (TP)

  CudaWorker has none vs CandleWorker which has full NCCL-based TP:

  - No ColumnParallelLinear / RowParallelLinear / VocabParallelEmbedding equivalents
  - No NCCL process group integration
  - No rank/world_size-aware weight sharding at load time
  - No MultiprocExecutor wiring

  Effort: Medium-large. Need parallel layer types in vllm-cuda, NCCL FFI from GpuTensor (not candle Tensor), and weight-sharding logic in
  GpuWeights.

  4. Missing Worker Features

  ┌───────────────────────────┬───────────────┬───────────────────────────┐
  │          Feature          │ CandleWorker  │        CudaWorker         │
  ├───────────────────────────┼───────────────┼───────────────────────────┤
  │ GGUF model loading        │ yes           │ no                        │
  ├───────────────────────────┼───────────────┼───────────────────────────┤
  │ Embeddings / pooling mode │ yes (embed()) │ yes (embed() + pooling)   │
  ├───────────────────────────┼───────────────┼───────────────────────────┤
  │ LoRA adapter loading      │ yes           │ no                        │
  ├───────────────────────────┼───────────────┼───────────────────────────┤
  │ Speculative decoding      │ yes (n-gram)  │ no                        │
  ├───────────────────────────┼───────────────┼───────────────────────────┤
  │ Grammar vocabulary init   │ yes           │ yes                       │
  ├───────────────────────────┼───────────────┼───────────────────────────┤
  │ Device auto-detection     │ yes           │ no (hardcoded CUDA)       │
  ├───────────────────────────┼───────────────┼───────────────────────────┤
  │ CPU fallback              │ yes           │ no (by design)            │
  └───────────────────────────┴───────────────┴───────────────────────────┘

  5. Sampling — FULL GPU PARITY

  CudaWorker now has full GPU-native sampling matching Python vLLM's Sampler.forward() pipeline.
  No CPU fallback ever — all logit modifications and sampling happen on GPU.

  Pipeline (matches Python vLLM v1/sample/sampler.py):
  1. Cast logits to f32 (if any modification needed)
  2. Save raw logits copy for logprobs (GPU D2D, before modifications)
  3. Apply grammar mask on GPU (CSR-packed allow-list → set disallowed to -inf)
  4. Apply logit bias on GPU (CSR-packed scatter-add)
  5. Apply penalties on GPU (fused rep/freq/pres kernel, one block per request)
  6. Sample on GPU (argmax, Gumbel-max, or fused top-k/top-p/min-p)
  7. Gather logprobs on GPU (fused log-softmax + top-K from raw logits)
  8. D2H only sampled token IDs + small logprobs tensors
  9. Grammar FSM advance on CPU (same as Python)

  ┌────────────────────────────┬────────────────┬──────────────────┬─────────────────────────────────────────────┐
  │          Feature           │ CandleWorker   │ CudaWorker (GPU) │                   Notes                     │
  ├────────────────────────────┼────────────────┼──────────────────┼─────────────────────────────────────────────┤
  │ Greedy (argmax)            │ yes            │ yes (graphed)    │ In-graph argmax, zero-copy D2D scatter      │
  ├────────────────────────────┼────────────────┼──────────────────┼─────────────────────────────────────────────┤
  │ Temperature                │ yes            │ yes              │ Gumbel-max (fast) or fused kernel           │
  ├────────────────────────────┼────────────────┼──────────────────┼─────────────────────────────────────────────┤
  │ Top-k                      │ yes            │ yes              │ Fused radix-select kernel                   │
  ├────────────────────────────┼────────────────┼──────────────────┼─────────────────────────────────────────────┤
  │ Top-p                      │ yes            │ yes              │ Fused kernel                                │
  ├────────────────────────────┼────────────────┼──────────────────┼─────────────────────────────────────────────┤
  │ Min-p                      │ yes            │ yes              │ Fused kernel                                │
  ├────────────────────────────┼────────────────┼──────────────────┼─────────────────────────────────────────────┤
  │ Repetition penalty         │ yes            │ yes (GPU)        │ Fused kernel, one block per request         │
  ├────────────────────────────┼────────────────┼──────────────────┼─────────────────────────────────────────────┤
  │ Frequency penalty          │ yes            │ yes (GPU)        │ Fused with rep/pres in single kernel        │
  ├────────────────────────────┼────────────────┼──────────────────┼─────────────────────────────────────────────┤
  │ Presence penalty           │ yes            │ yes (GPU)        │ Fused with rep/freq in single kernel        │
  ├────────────────────────────┼────────────────┼──────────────────┼─────────────────────────────────────────────┤
  │ Logit bias                 │ yes            │ yes (GPU)        │ CSR-packed scatter-add kernel               │
  ├────────────────────────────┼────────────────┼──────────────────┼─────────────────────────────────────────────┤
  │ Grammar / constrained dec  │ yes            │ yes (GPU)        │ CSR-packed allow-list mask kernel           │
  ├────────────────────────────┼────────────────┼──────────────────┼─────────────────────────────────────────────┤
  │ Logprobs                   │ yes            │ yes (GPU)        │ Fused log-softmax + top-K kernel            │
  ├────────────────────────────┼────────────────┼──────────────────┼─────────────────────────────────────────────┤
  │ Seed-based RNG             │ yes (per-req)  │ partial          │ murmurhash from uniform; not per-request    │
  ├────────────────────────────┼────────────────┼──────────────────┼─────────────────────────────────────────────┤
  │ n>1 completions            │ yes            │ yes              │ Engine-level, not worker-level              │
  └────────────────────────────┴────────────────┴──────────────────┴─────────────────────────────────────────────┘

  No CPU fallback paths remain. Mixed batches (some requests with penalties, some without)
  are handled entirely on GPU — no batch poisoning.

  Minor gaps vs Python (not in CandleWorker either) — see section 5a.

  5a. Sampling TODOs (minor gaps vs Python vLLM)

  These features exist in Python vLLM's Sampler but are NOT implemented in CudaWorker (or CandleWorker).
  They are rarely used in practice but listed here for completeness.

  ┌──────────────────────────────┬─────────────────────────────────────────────────────────────────────────┐
  │           Feature            │                                 Notes                                   │
  ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ allowed_token_ids whitelist  │ Per-request token whitelist bitmask (different from grammar). Python    │
  │                              │ uses masked_fill_ on a pre-built bool mask. Need a GPU kernel or       │
  │                              │ reuse the grammar mask kernel with a different allow-list source.       │
  ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ bad_words exclusion          │ Per-request list of banned token sequences. Python uses apply_bad_words │
  │                              │ which checks output_token_ids suffix matches and sets logit to -inf.    │
  │                              │ Needs CPU-side suffix matching + GPU scatter to -inf.                   │
  ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ min_tokens logits processor  │ Forces EOS token logit to -inf until min_tokens generated. Simple      │
  │                              │ per-request check: if len(output) < min_tokens, set logits[eos] = -inf │
  ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ Per-request seed-based RNG   │ Python creates per-request torch.Generator from user-provided seed.    │
  │                              │ CudaWorker currently uses shared thread_rng with murmurhash mixing.    │
  │                              │ Need per-request Philox state on GPU keyed by user seed.               │
  └──────────────────────────────┴─────────────────────────────────────────────────────────────────────────┘

  6. Missing Kernels

  CudaWorker already has: fused_add_rms_norm, silu_and_mul, gelu_and_mul, rotary, reshape_and_cache, flash_attn_paged, embedding_gather,
  split_qkv, fused_qkv_rope, gpu_sampling (argmax + Gumbel-max + fused top-k/top-p/min-p), apply_penalties (fused rep/freq/pres),
  apply_logit_bias (CSR scatter-add), apply_grammar_mask (CSR allow-list), log_softmax_topk (fused logprobs), cast_to_f32.

  Still needed:
  - Fused MoE GEMM — required for DeepSeek, Mixtral, Qwen MoE, Qwen3 MoE
  - Marlin INT4 GEMM FFI — exists in vllm-kernels but no GpuTensor bindings
  - GGUF dequant kernels — if not using candle's QCudaStorage
  - MoE top-k gating — exists in vllm-kernels, needs GpuTensor FFI

  7. Multimodal

  CandleWorker supports Gemma3-MM and Qwen2-VL/Qwen2.5-VL (vision encoder + projector). CudaWorker has none. This is likely out of scope for
  initial parity but worth noting.

  ---
  Priority Order (to retire CandleWorker CUDA)

  1. ~~Low-hanging fruit: Alias Mistral/Qwen3/Phi-3 to LLaMA in CudaWorker~~ — **DONE** (covers ~50% of real usage)
  2. Marlin FFI for GPTQ/AWQ: Wire existing Marlin kernel to GpuTensor — unlocks quantized serving for LLaMA-family — **IN PROGRESS**
  3. ~~Sampling correctness: GPU-native penalties, logit bias, grammar, logprobs~~ — **DONE** (full GPU parity, no CPU fallback, 18/18 E2E tests)
  4. GGUF support: Either port candle's QCudaStorage approach or add dequant kernels
  5. MoE kernel + models: Fused MoE GEMM, then port Mixtral/Qwen MoE/Qwen3 MoE
  6. DeepSeek V2/V3 (MLA): Most complex arch — absorbed-MLA attention, MoE
  7. Tensor parallelism: Parallel layers, NCCL, multi-GPU init
  8. Remaining dense archs: Command R, Qwen3-Next
  9. ~~Sampling perf: Fused penalty kernel on GPU, no CPU fallback~~ — **DONE**
  10. LoRA, speculative decoding: Feature parity on worker traits (embeddings done)
  11. Multimodal: Vision encoders (Gemma3-MM, Qwen2-VL)

  The critical path is items 1-5. That covers the vast majority of real-world CUDA usage (dense LLaMA-family models in
  FP16/BF16/GPTQ/AWQ/GGUF with correct sampling).
