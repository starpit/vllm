  Gap Analysis: CudaWorker vs CandleWorker

  1. Model Architectures

  CudaWorker has 6 (LLaMA, Mistral, Qwen2, Qwen3, Phi-3, Gemma2) vs CandleWorker has 15+:

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
  │ Gemma3 (text)            │ yes          │ no         │                                    │
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
  │ Grammar vocabulary init   │ yes           │ no (may work via sampler) │
  ├───────────────────────────┼───────────────┼───────────────────────────┤
  │ Device auto-detection     │ yes           │ no (hardcoded CUDA)       │
  ├───────────────────────────┼───────────────┼───────────────────────────┤
  │ CPU fallback              │ yes           │ no (by design)            │
  └───────────────────────────┴───────────────┴───────────────────────────┘

  5. Missing Kernels

  CudaWorker already has: fused_add_rms_norm, silu_and_mul, gelu_and_mul, rotary, reshape_and_cache, flash_attn_paged, embedding_gather,
  split_qkv, gpu_sampling.

  Still needed:
  - Fused MoE GEMM — required for DeepSeek, Mixtral, Qwen MoE, Qwen3 MoE
  - Marlin INT4 GEMM FFI — exists in vllm-kernels but no GpuTensor bindings
  - GGUF dequant kernels — if not using candle's QCudaStorage
  - MoE top-k gating — exists in vllm-kernels, needs GpuTensor FFI

  6. Multimodal

  CandleWorker supports Gemma3-MM and Qwen2-VL/Qwen2.5-VL (vision encoder + projector). CudaWorker has none. This is likely out of scope for
  initial parity but worth noting.

  ---
  Priority Order (to retire CandleWorker CUDA)

  1. ~~Low-hanging fruit: Alias Mistral/Qwen3/Phi-3 to LLaMA in CudaWorker~~ — **DONE** (covers ~50% of real usage)
  2. Marlin FFI for GPTQ/AWQ: Wire existing Marlin kernel to GpuTensor — unlocks quantized serving for LLaMA-family
  3. GGUF support: Either port candle's QCudaStorage approach or add dequant kernels
  4. MoE kernel + models: Fused MoE GEMM, then port Mixtral/Qwen MoE/Qwen3 MoE
  5. DeepSeek V2/V3 (MLA): Most complex arch — absorbed-MLA attention, MoE
  6. Tensor parallelism: Parallel layers, NCCL, multi-GPU init
  7. Remaining dense archs: Gemma3, Command R, Granite, Qwen3-Next
  8. LoRA, speculative decoding: Feature parity on worker traits (embeddings done)
  9. Multimodal: Vision encoders (Gemma3-MM, Qwen2-VL)

  The critical path is items 1-4. That covers the vast majority of real-world CUDA usage (dense LLaMA-family models in
  FP16/BF16/GPTQ/AWQ/GGUF). TP and MoE models are important but affect fewer users.
