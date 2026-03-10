  Gap Analysis: CudaWorker vs CandleWorker

  1. Model Architectures

  CudaWorker has 10 (LLaMA, Mistral, Qwen2, Qwen3, Phi-3, Gemma2, Gemma3, Mixtral, Qwen2 MoE, Qwen3 MoE) vs CandleWorker has 15+:

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
  │ Qwen2 MoE                │ yes          │ yes        │ MoE + shared expert (gated)        │
  ├──────────────────────────┼──────────────┼────────────┼────────────────────────────────────┤
  │ Qwen3 MoE                │ yes          │ yes        │ MoE + shared expert + QK-norm      │
  ├──────────────────────────┼──────────────┼────────────┼────────────────────────────────────┤
  │ Mixtral (MoE)            │ yes          │ yes        │ Pure MoE (8 experts, top-2)        │
  ├──────────────────────────┼──────────────┼────────────┼────────────────────────────────────┤
  │ Granite                  │ yes          │ yes        │ LLaMA + 4 scalar multipliers       │
  ├──────────────────────────┼──────────────┼────────────┼────────────────────────────────────┤
  │ Kimi K2.5                │ yes          │ no         │ DeepSeek V2 backbone               │
  ├──────────────────────────┼──────────────┼────────────┼────────────────────────────────────┤
  │ Qwen3-Next (GDN+MoE)     │ yes          │ no         │ Hybrid linear+full attention       │
  └──────────────────────────┴──────────────┴────────────┴────────────────────────────────────┘

  Effort: ~~Mistral/Qwen3/Phi-3 are trivial (alias LLaMA)~~ — DONE. Granite is similar. ~~MoE models (Mixtral, Qwen MoE) need a fused MoE GEMM
  kernel~~ — DONE (custom WMMA tensor-core kernel, see MoE section below). DeepSeek MLA is the hardest.

  2. Quantization Support

  ┌──────────────────┬──────────────┬─────────────┬────────────────────────────────────────────────┐
  │      Format      │ CandleWorker │ CudaWorker  │                     Notes                      │
  ├──────────────────┼──────────────┼─────────────┼────────────────────────────────────────────────┤
  │ GGUF (k-quants)  │ yes          │ no          │ Dequant kernels or candle's QCudaStorage       │
  ├──────────────────┼──────────────┼─────────────┼────────────────────────────────────────────────┤
  │ GPTQ (W4A16)     │ yes          │ YES         │ Marlin kernel, E2E verified (Qwen2.5-0.5B)    │
  ├──────────────────┼──────────────┼─────────────┼────────────────────────────────────────────────┤
  │ AWQ (W4A16)      │ yes          │ YES         │ Marlin kernel, E2E verified (Qwen2.5-0.5B)     │
  ├──────────────────┼──────────────┼─────────────┼────────────────────────────────────────────────┤
  │ BitsAndBytes NF4 │ yes          │ no          │ NF4 dequant kernel                             │
  └──────────────────┴──────────────┴─────────────┴────────────────────────────────────────────────┘

  ### GPTQ/AWQ Parity vs Python vLLM (detailed)

  **Functional parity:**

  ┌─────────────────────────────┬────────────┬──────────┬──────────────────────────────────────────┐
  │          Feature            │ Python vLLM│ Rust vLLM│                  Notes                   │
  ├─────────────────────────────┼────────────┼──────────┼──────────────────────────────────────────┤
  │ GPTQ 4-bit symmetric        │ yes        │ yes      │ b_type_id=0 (kU4B8), E2E verified       │
  ├─────────────────────────────┼────────────┼──────────┼──────────────────────────────────────────┤
  │ GPTQ 4-bit asymmetric       │ yes        │ no       │ Needs zero-point loading                 │
  ├─────────────────────────────┼────────────┼──────────┼──────────────────────────────────────────┤
  │ AWQ 4-bit                   │ yes        │ yes      │ Marlin kernel, E2E verified (Qwen2.5-0.5B) │
  ├─────────────────────────────┼────────────┼──────────┼──────────────────────────────────────────┤
  │ Linear bias (QKV)           │ yes        │ yes      │ Post-GEMM bias_add_inplace               │
  ├─────────────────────────────┼────────────┼──────────┼──────────────────────────────────────────┤
  │ desc_act (act ordering)     │ yes        │ no       │ g_idx sort + perm needed                 │
  ├─────────────────────────────┼────────────┼──────────┼──────────────────────────────────────────┤
  │ Fused QKV GEMM (quant)      │ yes        │ YES      │ Concat on CPU, single repack + GEMM      │
  ├─────────────────────────────┼────────────┼──────────┼──────────────────────────────────────────┤
  │ Fused gate+up GEMM (quant)  │ yes        │ YES      │ Concat on CPU, single repack + GEMM      │
  ├─────────────────────────────┼────────────┼──────────┼──────────────────────────────────────────┤
  │ use_fp32_reduce              │ yes (dflt) │ no       │ Python defaults true; Rust passes false   │
  ├─────────────────────────────┼────────────┼──────────┼──────────────────────────────────────────┤
  │ In-kernel bias (permuted)   │ yes        │ no       │ Python uses marlin_permute_bias()         │
  ├─────────────────────────────┼────────────┼──────────┼──────────────────────────────────────────┤
  │ CUDA graphs (quant)         │ yes        │ yes      │ Works with decode graphs                  │
  ├─────────────────────────────┼────────────┼──────────┼──────────────────────────────────────────┤
  │ Architectures (quant)       │ all        │ LLaMA, Qwen2, Gemma2, Granite │ Gemma2 GPTQ E2E verified; Granite/Gemma3/MoE E2E tests needed │
  └─────────────────────────────┴────────────┴──────────┴──────────────────────────────────────────┘

  **Performance (Qwen2.5-0.5B-Instruct-GPTQ-Int4, nick5 L40S, 128 output tokens):**

  ┌───────────────────┬──────────┬──────────┐
  │     Metric        │ Rust GPTQ│ Rust Dense│
  ├───────────────────┼──────────┼──────────┤
  │ avg ITL (decode)  │ 3.0 ms   │ 2.2 ms   │
  ├───────────────────┼──────────┼──────────┤
  │ TTFT (prefill)    │ 8.7 ms   │ 2.6 ms   │
  └───────────────────┴──────────┴──────────┘

  GPTQ is 1.36x slower on decode, 3.3x slower on prefill. Expected for a tiny 0.5B model where
  Marlin overhead dominates. On larger models (7B+) the memory bandwidth savings should make GPTQ
  faster than dense.

  **Key gaps to close (priority order):**
  1. ~~Fused QKV/gate_up at load time~~ — **DONE** (5→2 GEMMs per layer, CPU concat + single repack)
  2. use_fp32_reduce=true — match Python default for numerical accuracy
  3. ~~AWQ E2E testing~~ — **DONE** (Qwen2.5-0.5B-Instruct-AWQ, nick4 L40S)
  4. ~~Wire LLaMA/Gemma2/Granite for quantized loading~~ — **DONE** (LLaMA, Qwen2, Gemma2, Granite all wired)
  5. desc_act support — needed for some GPTQ models

  Effort: GGUF requires either porting candle's QCudaStorage approach or writing dequant-on-the-fly kernels. BnB is lower priority.

  3. ~~Tensor Parallelism (TP)~~ — **DONE**

  CudaWorker now has full NCCL-based TP, matching CandleWorker:

  - ColumnParallelLinear / RowParallelLinear / VocabParallelEmbedding in `vllm-cuda/src/layers.rs`
  - NcclGroup wrapping cudarc NCCL FFI for all-reduce/all-gather on GpuTensor (`vllm-cuda/src/nccl.rs`)
  - CPU-side weight sharding via `take_shard()`/`take_shard_into()` in GpuWeights
  - `load_fused_tp()` on LLaMA/Qwen2/Gemma2 attention + MLP
  - NCCL all-reduce after row-parallel layers (o_proj, down_proj), all-gather after lm_head
  - `initialize_stack_tp()` with concurrent NCCL init + profiling + warmup on scoped threads
  - ThreadPoolExecutor dispatches execute_model concurrently across ranks
  - E2E verified: TP=2 Qwen2.5-0.5B on 2x L40S (nick3)

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
  ├───────────────────────────┼───────────────┼───────────────────────────┤
  │ Tensor parallelism (NCCL) │ yes           │ yes (TP=2 E2E verified)   │
  └───────────────────────────┴───────────────┴───────────────────────────┘

  5. Sampling — FULL GPU PARITY

  CudaWorker now has full GPU-native sampling matching Python vLLM's Sampler.forward() pipeline.
  No CPU fallback ever — all logit modifications and sampling happen on GPU.

  Pipeline (matches Python vLLM v1/sample/sampler.py):
  1. Cast logits to f32 (if any modification needed)
  2. Save raw logits copy for logprobs (GPU D2D, before modifications)
  3. Apply grammar mask on GPU (CSR-packed allow-list → set disallowed to -inf)
  4. Apply min_tokens on GPU (suppress EOS/stop tokens until min_tokens reached)
  5. Apply logit bias on GPU (CSR-packed scatter-add)
  6. Apply penalties on GPU (fused rep/freq/pres kernel, one block per request)
  7. Sample on GPU (argmax, Gumbel-max, or fused top-k/top-p/min-p)
  8. Gather logprobs on GPU (fused log-softmax + top-K from raw logits)
  9. D2H only sampled token IDs + small logprobs tensors
  10. Grammar FSM advance on CPU (same as Python)

  Architecture: Formal LogitsProcessor framework (mirrors Python vLLM v1/sample/logits_processor/):
  - LogitsProcessor trait: update_state(), apply(), is_argmax_invariant(), is_active()
  - LogitsProcessorPipeline: container splitting processors by argmax-invariance
  - BatchUpdate: notification struct for batch composition changes
  - Persistent GPU state: tensors rebuilt only on batch changes (not every step)
  - Built-in processors: GrammarMaskProcessor, MinTokensProcessor, LogitBiasProcessor, PenaltiesProcessor

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
  │ ~~min_tokens logits processor~~│ ~~DONE — MinTokensProcessor suppresses EOS/stop tokens on GPU~~       │
  │                              │ ~~until min_tokens generated. CUDA kernel + E2E test verified.~~       │
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
  - ~~Marlin INT4 GEMM FFI~~ — **DONE** (marlin_gemm, repack, permute_scales all wired)
  - GGUF dequant kernels — if not using candle's QCudaStorage

  Already implemented:
  - ~~Fused MoE GEMM~~ — custom WMMA kernel (BLOCK_M=128, BLOCK_N=128, BLOCK_K=32)
  - ~~MoE top-k gating~~ — TRT-LLM topk_softmax kernel with GpuTensor FFI
  - ~~moe_align_block_size~~ — ported from Python vLLM (small + large batch paths)
  - ~~moe_sum~~ — reduction kernel with GpuTensor FFI
  - ~~sigmoid_mul_add~~ — shared expert gating kernel (vectorized 128-bit loads)
  - ~~qk_norm_rope~~ — fused per-head RMS norm + NeoX RoPE (Qwen3 MoE, Gemma3)
  - ~~Weight dtype casting~~ — GpuWeights casts F32→BF16/F16 at load time via pinned host memory (matches Python's torch_dtype auto-cast)

  6a. MoE TODOs

  Remaining work items for full MoE parity with Python vLLM:

  - Qwen2 MoE E2E test: Qwen/Qwen1.5-MoE-A2.7B-Chat is ~30GB, times out downloading on pod.
    Need a smaller Qwen2 MoE test model or pre-cache the model.
  - Qwen3 MoE E2E test: No small Qwen3 MoE test model identified yet.
  - CUDA graphs for MoE: Decode CUDA graphs are disabled for MoE models because the
    fused MoE GEMM kernel uses dynamic shared memory and variable grid sizes based on
    num_tokens_post_padded (output of moe_align_block_size). Need to either pad to
    fixed sizes or capture multiple graph variants.
  - MoE + TP: Expert parallelism (expert_map) needed for multi-GPU MoE serving.
    Currently only single-GPU MoE works.
  - Profile vs Python: No nsys data yet comparing our WMMA MoE kernel against
    Python's Triton fused_moe_kernel on real workloads (Mixtral-8x7B decode/prefill).

  6b. MoE Performance Gaps

  The fused MoE GEMM kernel matches Python vLLM's Triton `fused_moe_kernel` functionally but has known performance gaps:

  ┌──────────────────────────────────────┬───────────────┬─────────────────────────────────────────────────────┐
  │                 Gap                  │ Est. Impact   │                        Notes                        │
  ├──────────────────────────────────────┼───────────────┼─────────────────────────────────────────────────────┤
  │ Fixed tile sizes (128/128/32) vs     │ 2-3x slower   │ Triton autotuning picks optimal tile per shape.     │
  │ Triton autotuning                    │ some shapes   │ Profile to find which shapes regress most.          │
  ├──────────────────────────────────────┼───────────────┼─────────────────────────────────────────────────────┤
  │ WMMA 16x16x16 vs native mma PTX     │ 10-30% slower │ WMMA is portable but generates suboptimal PTX.      │
  │                                      │               │ Triton emits mma.m16n8k16 directly.                 │
  ├──────────────────────────────────────┼───────────────┼─────────────────────────────────────────────────────┤
  │ No GROUP_SIZE_M L2 cache grouping    │ Varies        │ Triton reorders threadblocks for L2 reuse.          │
  │                                      │               │ Matters most at large token counts (prefill).       │
  ├──────────────────────────────────────┼───────────────┼─────────────────────────────────────────────────────┤
  │ No chunked processing                │ OOM risk      │ Python splits large batches via                     │
  │                                      │               │ VLLM_FUSED_MOE_CHUNK_SIZE to limit scratch memory.  │
  └──────────────────────────────────────┴───────────────┴─────────────────────────────────────────────────────┘

  Mitigation plan: Profile on real MoE models (Mixtral-8x7B, Qwen2-57B-MoE) first. For decode (BS=1-32),
  the kernel launch overhead dominates and our kernel should be close. For prefill, if the gap is > 2x,
  upgrade WMMA → inline PTX mma and add tile-size selection based on problem dimensions.

  6c. MoE Feature Gaps (not needed for initial launch)

  ┌──────────────────────────────────────┬──────────────────────────────────────────────────────────────┐
  │              Feature                 │                            Notes                             │
  ├──────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
  │ Expert parallelism (expert_map)      │ Needed for TP of MoE models where experts are split across  │
  │                                      │ GPUs. Not needed for single-GPU or TP on dense layers only. │
  ├──────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
  │ Quantized MoE (FP8/INT8/INT4)       │ Python supports FP8 W8A8, INT8 W8A8, INT4 W4A16 for MoE    │
  │                                      │ expert weights. Our kernel only supports BF16/F16.          │
  ├──────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
  │ token_mask in align_block_size       │ Used for masked expert routing in some configurations.      │
  │                                      │ Not used by Mixtral/Qwen MoE.                               │
  ├──────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
  │ DeepSeek MoE (shared + routed)       │ Different MoE pattern: shared expert runs unconditionally,  │
  │                                      │ routed experts have fine-grained routing. Needs MLA too.    │
  └──────────────────────────────────────┴──────────────────────────────────────────────────────────────┘

  7. Multimodal

  CandleWorker supports Gemma3-MM and Qwen2-VL/Qwen2.5-VL (vision encoder + projector). CudaWorker has none. This is likely out of scope for
  initial parity but worth noting.

  ---
  Priority Order (to retire CandleWorker CUDA)

  1. ~~Low-hanging fruit: Alias Mistral/Qwen3/Phi-3 to LLaMA in CudaWorker~~ — **DONE**
  2. ~~Marlin FFI for GPTQ/AWQ~~ — **DONE** (both GPTQ + AWQ E2E verified, Qwen2.5-0.5B)
  3. ~~Sampling correctness: GPU-native penalties, logit bias, grammar, logprobs~~ — **DONE** (full GPU parity, no CPU fallback, 18/18 E2E tests)
  4. GGUF support: Either port candle's QCudaStorage approach or add dequant kernels
  5. ~~MoE kernel + models: Fused MoE GEMM, then port Mixtral/Qwen MoE/Qwen3 MoE~~ — **DONE** (WMMA tensor-core kernel, 3 models)
  6. DeepSeek V2/V3 (MLA): Most complex arch — absorbed-MLA attention, MoE
  7. ~~Tensor parallelism: Parallel layers, NCCL, multi-GPU init~~ — **DONE** (TP=2 Qwen2.5-0.5B E2E verified)
  8. Remaining dense archs: Command R, Qwen3-Next
  9. ~~Sampling perf: Fused penalty kernel on GPU, no CPU fallback~~ — **DONE**
  10. LoRA, speculative decoding: Feature parity on worker traits (embeddings done)
  11. Multimodal: Vision encoders (Gemma3-MM, Qwen2-VL)
  12. MoE perf tuning: Inline PTX mma, tile autoselection, L2 grouping (see section 6)
  13. Quantized MoE: FP8/INT8/INT4 expert weights

  The critical path is item 4 (GGUF). Items 1, 2, 3, 5, 7, 9 are done. That covers dense + MoE LLaMA-family models
  in FP16/BF16 with correct sampling, GPTQ/AWQ quantization, and multi-GPU tensor parallelism.
