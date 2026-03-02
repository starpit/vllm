# CUDA Parity Plan: Rust vLLM ↔ Python vLLM

> Generated 2026-03-01 | Baseline: `feat/rust` branch (889 tests, 0 clippy errors)
> Goal: feature-for-feature CUDA parity with Python vLLM's V1 engine on NVIDIA GPUs
>
> **Progress (2026-03-02)**: Phases 0, 1, 2.1, 3 DONE on `worktree-cuda` branch.
> E2E verified on L40S (48GB Ada): Qwen2.5-0.5B BF16.
> 6 fused CUDA kernels, all with vectorized 128-bit loads (vec_utils.cuh).
> FlashAttention v2 integrated via `candle-flash-attn` crate — auto-dispatches on CUDA F16/BF16.
> 33 CUDA kernel unit tests + 9 FlashAttention tests + 5 CUDA E2E tests.
> Next: CUDA pod verification, then Phase 4 (CUDA Graphs) or Phase 2b (batched FA2).

---

## Executive Summary

The Rust port has **working CUDA inference** with custom fused kernels (Phases 0, 1, 3 complete):

- **Candle CUDA baseline** — `candle-core` 0.9 runs matmul, softmax, element-wise ops on GPU automatically via cuBLAS
- **GPU memory management** — VRAM detection via `cudarc::driver::result::mem_get_info()`, KV cache blocks allocated on GPU, GPU↔CPU block swapping
- **6 fused CUDA kernels** in `vllm-kernels/csrc/` — RMSNorm, fused-add-RMSNorm, SiLU+mul / GELU+mul, RoPE, reshape_and_cache, QK-norm+RoPE — all with vectorized 128-bit loads via `vec_utils.cuh`
- **Runtime kernel dispatch** — `KernelSet` trait + `ops.rs` auto-routes to fused CUDA kernels on GPU, CPU fallbacks otherwise
- **`fused_add_rms_norm`** wired into all model decoder layers — saves 1 kernel launch + 1 tensor alloc per layer per forward pass
- **E2E verified** — Qwen2.5-0.5B BF16 on L40S (48GB Ada), correct completions + chat. TTFT ~650ms, ITL ~142ms.

- **FlashAttention v2** — via `candle-flash-attn` crate, auto-dispatches on CUDA F16/BF16. Replaces naive O(n²) SDPA with tiled IO-aware algorithm — O(1) extra memory, ~10x faster on long sequences.

**What's NOT yet done** (biggest remaining gaps vs Python vLLM):

- **Batched FlashAttention** (Phase 2b) — FA2 dispatches per-request; batched `flash_attn_varlen()` across all requests in a batch would improve throughput further.
- **CUDA Graphs** (Phase 4) — no graph capture for decode phase. Kernel launch overhead dominates single-token decode steps.
- **Multi-GPU / NCCL** (Phase 5) — single-GPU only.
- **Quantization kernels** (Phase 6) — no GPTQ/AWQ/FP8 CUDA compute. GGUF models fall back to CPU via candle's QMatMul.
- **Fused MoE** (Phase 7) — MoE models (DeepSeek, Qwen3-MoE) use per-expert loops on GPU, no fused top-k routing + expert matmul.

Python vLLM has **~183 CUDA/C++ source files** in `csrc/`, plus **72+ Triton kernels**. This plan targets feature-for-feature parity organized into **7 phases**, roughly ordered by impact and dependency.

---

## What's Next

The recommended priority order for remaining phases:

### 1. Phase 2 CUDA pod verification
FlashAttention v2 is integrated (Phase 2.1 ✅) but needs pod testing. Build + clippy + unit tests + E2E on L40S to verify correctness and measure ITL improvement over naive SDPA.

### 2. Phase 4: CUDA Graphs (decode acceleration)
With FlashAttention done, decode-phase kernel launch overhead becomes the next bottleneck. CUDA graphs capture the entire decode forward pass as a single GPU-side graph, eliminating ~100 individual kernel launches per step. Python vLLM gets ~30-50% decode ITL improvement from this.

### 3. Phase 7.1: Fused MoE Kernels (MoE model perf)
Critical for DeepSeek V2/V3, Qwen3-MoE, and Mixtral performance on CUDA. Current per-expert loop is extremely slow on GPU. Fused top-k gating + expert GEMM is a well-known optimization. Can be done independently of Phases 2/4.

### 4. Phase 5: Multi-GPU / NCCL (model size scaling)
Enables models >13B that don't fit on a single GPU. Important for production use but lower priority than single-GPU performance. NCCL tensor parallelism is the standard approach.

### 5. Phase 6: Quantization Kernels (GPTQ/AWQ/FP8)
Enables quantized model inference on CUDA. Most production deployments use 4-bit or 8-bit models. Currently GGUF falls back to CPU; GPTQ/AWQ don't work at all. Large effort but high production value.

---

## Phase 0: Candle CUDA End-to-End Validation ✅ DONE

**Goal**: Verify that candle-core's built-in CUDA support works end-to-end with the existing Rust codebase — model loading, forward pass, generation — without any custom kernels.

**Why first**: candle-core 0.9 already compiles CUDA matmul/softmax/element-wise ops. If we can run the existing `CandleWorker` on `cuda:0` and get correct output, we have a working CUDA baseline before writing any custom code. This is the cheapest possible win.

### Tasks

| # | Task | Details | Status |
|---|------|---------|--------|
| 0.1 | **Enable `candle-core/cuda` feature flag** | `candle-core = { version = "0.9", default-features = false }` + cudarc 0.19 workspace dep. | ✅ |
| 0.2 | **Workspace `cuda` feature cascade** | cuda feature threads: vllm-cli → vllm-serve → vllm-executor → vllm-models → vllm-kernels → candle-core/cuda + cudarc | ✅ |
| 0.3 | **GPU VRAM detection** | `cudarc::driver::result::mem_get_info()` in `determine_available_memory()` when device is CUDA. Fallback to sysinfo. | ✅ |
| 0.4 | **Fix `.contiguous()` calls** | Added in siglip.rs (k transpose), quantized_llama.rs (tied embed w.t()), Linear::forward (input x). | ✅ |
| 0.5 | **E2E smoke test on CUDA** | Qwen2.5-0.5B BF16 on L40S: correct completions + chat output. TTFT ~650ms, ITL ~142ms. | ✅ |
| 0.6 | **BF16 on CUDA** | Auto-detected from config.json torch_dtype, works on L40S (Ada). | ✅ |
| 0.7 | **CI: CUDA build check** | Not yet done. | |
| 0.8 | **Dockerfile.cuda update** | Updated to CUDA 12.9, `--features cuda`, ubuntu 24.04 base. | ✅ |

**Exit criteria**: ✅ MET — `vllm serve Qwen/Qwen2.5-0.5B --device cuda` produces correct output on L40S. VRAM read from GPU (41.9 GB).

---

## Phase 1: GPU Memory Management & KV Cache on GPU ✅ DONE

**Goal**: Proper CUDA memory lifecycle — allocate KV cache blocks on GPU, profile memory, compute block counts from actual VRAM.

### Tasks

| # | Task | Details | Status |
|---|------|---------|--------|
| 1.1 | **cudarc dependency** | Added to vllm-kernels + vllm-executor behind `cuda` feature. | ✅ (Phase 0) |
| 1.2 | **CudaDevice memory query** | `cudarc::driver::result::mem_get_info()` in determine_available_memory(). | ✅ (Phase 0) |
| 1.3 | **KvBlockPool on GPU** | Already works — pool takes &Device, candle allocates on GPU. Verified by E2E test (206K blocks on L40S). | ✅ |
| 1.4 | **GPU ↔ CPU block swapping** | `swap_out()`/`swap_in()` on KvBlockPool via candle `to_device()`. | ✅ |
| 1.5 | **reshape_and_cache CUDA kernel** | Deferred to Phase 3 (fused kernels). | |
| 1.6 | **Memory profiling** | Already correct — VRAM queried after model load, model footprint naturally excluded. | ✅ |

**Exit criteria**: ✅ MET — KV blocks on GPU, VRAM-based block counts, GPU↔CPU swap works.

---

## Phase 2: FlashAttention Integration (Highest-Impact Kernel) — PHASE 2.1 DONE

**Goal**: FFI bindings to FlashAttention v2 for batched variable-length attention on CUDA. This is the single largest performance win — it's what makes Python vLLM fast.

**Why next**: Attention is the bottleneck for all LLM inference. FlashAttention is ~10x faster than naive SDPA on long sequences, uses O(1) extra memory (no materialized attention matrix), and supports paged KV cache natively.

### Approach: `candle-flash-attn` crate (0.9.2)

Instead of vendoring FA2 source or writing custom FFI, we use the `candle-flash-attn` crate which wraps FlashAttention v2 CUDA kernels with a native candle `Tensor` API. It matches our `candle-core` 0.9 version exactly.

**Integration point**: `attention_with_cache()` in `vllm-models/src/attention.rs` — the single function through which ALL model attention flows. On CUDA with F16/BF16, it auto-dispatches to FlashAttention via `flash_attn()` / `flash_attn_windowed()`. No model architecture changes needed — all models get FA2 automatically.

### Tasks

| # | Task | Details | Status |
|---|------|---------|--------|
| 2.1 | **FA2 single-sequence integration** | `candle-flash-attn` dep + `flash_attention_single_seq()` helper + dispatch in `attention_with_cache()`. Auto-routes CUDA F16/BF16 to FA2, CPU/F32 to SDPA. On CUDA, paged decode path is skipped (gather+FA2 is faster than per-block Rust loop). 9 unit tests. | ✅ |
| 2.2 | **CUDA pod verification** | Build + clippy + kernel tests + E2E on L40S pod. Verify FA2 correctness and measure ITL improvement. | |
| 2.3 | **Batched attention (Phase 2b)** | Replace per-request `attention_with_cache()` loop with single batched `flash_attn_varlen()` call across all requests in `forward_batch()`. This is the continuous batching optimization — all Q/K/V concatenated, FA2 handles ragged seqlens via cu_seqlens. | |
| 2.4 | **Paged decode with FA2 (Phase 2c)** | Use FA2's `flash_attn_with_kvcache` for paged KV cache decode — reads directly from block table without gather. Requires FlashInfer or custom paged wrapper. | |

**Phase 2.1 exit criteria**: ✅ MET — `attention_with_cache()` dispatches to FA2 on CUDA F16/BF16. All model architectures use it automatically. 9 CUDA unit tests pass locally (awaiting pod verification). Local clippy + all non-CUDA tests pass.

---

## Phase 3: Fused CUDA Kernels (Norm, Activation, RoPE, Cache) — IN PROGRESS

**Goal**: Port the critical fused CUDA kernels from `csrc/` that eliminate intermediate tensor allocations and kernel launch overhead.

**Why next**: After FlashAttention handles attention, the remaining per-token overhead is dominated by RMSNorm (every layer, 2x), activation (every layer), and RoPE (every layer). Fusing these gives ~30-50% speedup on non-attention ops.

### Tasks

| # | Task | Details | Status |
|---|------|---------|--------|
| 3.1 | **Build infrastructure** | `vllm-kernels/build.rs` compiles `csrc/*.cu` via `cc` crate + nvcc. SM80/86/89/90. `-O3 --use_fast_math`. | ✅ |
| 3.2 | **Fused RMSNorm kernel** | `csrc/layernorm_kernels.cu` — vectorized loads, warp shuffle, 2D. `CudaNormKernels` FFI for f32/f16/bf16. `fused_add_rms_norm` wired. | ✅ |
| 3.3 | **Fused SiLU+mul kernel** | Port `csrc/activation_kernels.cu` → `silu_and_mul()`, `gelu_and_mul()`, `gelu_new_and_mul()`. Vectorized loads. | ✅ |
| 3.4 | **Fused RoPE kernel** | Port `csrc/pos_encoding_kernels.cu` → `rotary_embedding()`. NeoX-style, per-head rotation. Vectorized loads. | ✅ |
| 3.5 | **reshape_and_cache fused kernel** | Port `csrc/cache_kernels.cu` for paged KV cache scatter. NHD layout. Vectorized loads. | ✅ |
| 3.6 | **CudaKernelSet struct** | `KernelSet` trait + `CpuKernelSet`/`CudaKernelSet` + `create_kernel_set(device)` factory. | ✅ |
| 3.7 | **Fused QK-norm+RoPE** | Per-head RMS norm + NeoX RoPE in single kernel. Used by Gemma3. `qk_norm_rope_kernels.cu`, wired via `ops::qk_norm_and_rope()`. 4 GPU unit tests. | ✅ |
| 3.8 | **Kernel dispatch in model layers** | Wire kernel traits into model forward() methods. `ops.rs` dispatch for all model architectures. | ✅ |

**Remaining simplifications vs Python vLLM:**
- Warp shuffle reduction instead of CUB BlockReduce
- 2D only (no 3D/4D per-head QK-norm)
- Python `.cu` files can't be used directly — they depend on PyTorch C++ API (torch/cuda.h, ATen dispatch)

**Exit criteria**: All fused kernels match `csrc/` behavior. Per-token latency on CUDA is within 20% of Python vLLM (without CUDA graphs or TP).

---

## Phase 4: CUDA Graphs

**Goal**: Capture and replay static computation graphs for the decode phase, eliminating kernel launch overhead.

**Why**: The decode phase processes a single token per request per step. The compute per token is small, so kernel launch overhead dominates. CUDA graphs capture the entire decode forward pass as a single GPU-side graph, reducing ~100 kernel launches to 1 graph launch. Python vLLM V1 gets ~30-50% decode speedup from this.

### Tasks

| # | Task | Details | Est. |
|---|------|---------|------|
| 4.1 | **cudarc graph capture API** | Use cudarc's CUDA graph APIs (`cuGraphCreate`, `cuGraphLaunch`). Prototype capturing a simple forward pass. | M |
| 4.2 | **Static decode batch** | Create a "padded batch" abstraction for decode: fixed-size input tensors (max_num_seqs × 1 token) with actual data in the first N slots. This is necessary because CUDA graphs require fixed tensor shapes. | M |
| 4.3 | **Graph capture for decode** | Capture the decode forward pass (embedding → layers → lm_head → sampling) as a CUDA graph. Use dummy input for capture, then memcpy real data into the captured tensors before replay. | L |
| 4.4 | **Graph pool / multi-batch-size** | Capture graphs for multiple batch sizes (1, 2, 4, 8, 16, ..., max_num_seqs). Select the smallest graph that fits the current batch. | M |
| 4.5 | **Prefill bypass** | Prefill (variable-length input) skips CUDA graphs and runs eagerly. Only decode steps use captured graphs. | S |
| 4.6 | **`--cuda-graph-mode` CLI flag** | Expose Python vLLM's `--cuda-graph-mode` flag: `full` (capture all decode), `piecewise` (capture per-layer), or `none` (disable). Default: `full`. | S |
| 4.7 | **Integration tests** | Verify output correctness with and without CUDA graphs (should be bitwise identical). Benchmark ITL improvement. | M |

**Exit criteria**: Decode ITL (inter-token latency) within 10% of Python vLLM for same model/batch/hardware.

---

## Phase 5: Multi-GPU (NCCL + Tensor Parallelism)

**Goal**: Enable tensor parallelism across multiple GPUs via NCCL, allowing large models that don't fit on a single GPU.

**Why**: Models >13B typically need multi-GPU. Python vLLM supports TP (tensor parallelism) via NCCL all-reduce/all-gather. Without this, the Rust port is limited to models that fit on one GPU.

### Tasks

| # | Task | Details | Est. |
|---|------|---------|------|
| 5.1 | **NCCL Rust bindings** | Add `nccl-rs` crate (or FFI bindings to libnccl). Expose `ncclAllReduce`, `ncclAllGather`, `ncclReduceScatter`, communicator init. | M |
| 5.2 | **ProcessGroup abstraction** | Create a `ProcessGroup` trait with NCCL backend. Methods: `all_reduce(tensor)`, `all_gather(tensor)`, `reduce_scatter(tensor)`, `broadcast(tensor)`. | M |
| 5.3 | **Multi-process worker launch** | Extend `MultiprocExecutor` to spawn N worker processes (one per GPU). Each worker initializes its own CUDA device and NCCL communicator. Use shared memory or sockets for control plane. | L |
| 5.4 | **Weight sharding** | Implement actual tensor parallel weight loading: `ColumnParallelLinear` shards weights across GPUs (each gets `[hidden, hidden/tp]`), `RowParallelLinear` shards the other dimension. `VocabParallelEmbedding` partitions the vocab. The layer abstractions already exist in `vllm-model` — they need to actually shard. | L |
| 5.5 | **All-reduce in model layers** | Insert NCCL all-reduce after `RowParallelLinear` and at attention output. This is where tensor-parallel outputs are summed across GPUs. | M |
| 5.6 | **Custom all-reduce (optional)** | Port `csrc/custom_all_reduce.cu` — optimized IPC-based all-reduce for same-node multi-GPU that bypasses NCCL overhead for small tensors. Python vLLM uses this for <2MB transfers. | L |
| 5.7 | **`--tensor-parallel-size` CLI flag** | Wire TP size through config → executor → worker launch. Validate against available GPU count. | S |
| 5.8 | **TP correctness tests** | Compare single-GPU vs 2-GPU TP output for a model (should be numerically close). Test various TP sizes. | M |
| 5.9 | **Pipeline parallelism (Phase 5b)** | Split model layers across GPUs (PP). Requires point-to-point NCCL sends between stages. Lower priority than TP. | XL |

**Exit criteria**: `vllm serve <70B-model> --tensor-parallel-size 4 --device cuda` works correctly across 4 GPUs with NCCL all-reduce.

---

## Phase 6: Quantization Kernels (GPTQ, AWQ, FP8)

**Goal**: Enable quantized model inference on CUDA with dedicated compute kernels.

**Why**: Most production deployments use quantized models (4-bit GPTQ/AWQ, 8-bit FP8) to fit larger models in less VRAM and increase throughput. Without quantization kernels, CUDA inference is limited to FP16/BF16 full-precision models.

### Tasks

| # | Task | Details | Est. |
|---|------|---------|------|
| 6.1 | **GPTQ dequantize + matmul** | Port `csrc/quantization/gptq/q_gemm.cu`. GPTQ stores weights as packed 4-bit integers + scales + zero points. The kernel dequantizes on-the-fly during matmul. Implement `GptqLinear` layer in `vllm-model`. | L |
| 6.2 | **AWQ gemm kernel** | Port `csrc/quantization/awq/gemm_kernels.cu`. AWQ uses per-channel scaling — similar to GPTQ but different packing. Implement `AwqLinear` layer. | L |
| 6.3 | **Marlin kernel (fast GPTQ/AWQ)** | Port `csrc/quantization/marlin/marlin.cu`. Marlin is an optimized 4-bit GEMM kernel that's ~2-4x faster than naive GPTQ dequant+matmul. Used by Python vLLM for GPTQ-Marlin and AWQ-Marlin formats. | XL |
| 6.4 | **FP8 matmul (CUTLASS)** | Port `csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu`. FP8 (E4M3/E5M2) matmul via CUTLASS on Hopper/Ada GPUs. Implement `Fp8Linear` layer with per-tensor or per-token scaling. | L |
| 6.5 | **INT8 quantization** | Port `csrc/quantization/w8a8/int8/scaled_quant.cu`. Dynamic per-token INT8 quantization for activations. | M |
| 6.6 | **GGUF on CUDA** | Port `csrc/quantization/gguf/gguf_kernel.cu` — GPU-accelerated GGUF dequantize + matmul. Currently GGUF models run on CPU via candle's QMatMul; this would enable GGUF on GPU. | L |
| 6.7 | **Quantization auto-detection** | Parse `quantization_config` from config.json to auto-select GPTQ/AWQ/FP8 linear layers. Mirror Python vLLM's `QuantizationConfig` pattern. | M |
| 6.8 | **HuggingFace quantized weight loading** | Load GPTQ/AWQ weight files (packed integers + scales). Handle the various packing formats (row-wise, column-wise, different group sizes). | M |

**Exit criteria**: `vllm serve TheBloke/Llama-2-13B-GPTQ --device cuda` works at near-Python-vLLM throughput.

---

## Phase 7: Advanced Performance & Polish

**Goal**: Close remaining performance gaps and enable advanced features.

### Tasks

| # | Task | Details | Est. |
|---|------|---------|------|
| 7.1 | **Fused MoE routing + expert matmul** | Port `csrc/moe/` kernels. Critical for DeepSeek V2/V3, Mixtral, Qwen3-MoE performance on CUDA. Current code does per-expert loops which is very slow on GPU. | XL |
| 7.2 | **GPU-side sampling** | Port `csrc/sampler.cu` — top-k/top-p on GPU. Currently sampling runs on CPU after pulling logits back. For batched inference with large vocab (128K+ tokens), GPU sampling avoids the ~1ms logit transfer. | M |
| 7.3 | **Persistent InputBatch** | Python vLLM V1's `InputBatch` persists across scheduler steps — only delta-updates changed request slots. Eliminates per-step tensor allocation for input_ids, positions, slot_mapping. | L |
| 7.4 | **Mixed prefill+decode batch** | Run prefill and decode requests in the same batch (Python vLLM V1 does this). Requires careful attention masking: prefill tokens see full causal context, decode tokens see only their cache. FlashAttention v2 supports this natively. | L |
| 7.5 | **KV cache compression (MLA latent)** | For DeepSeek V2/V3 MLA: cache the compressed latent `c_kv` instead of full K/V, saving ~4x memory. | M |
| 7.6 | **FP8 KV cache** | Store KV cache in FP8 (E4M3) instead of FP16, doubling effective cache capacity. Requires scale factors and dequant during attention. FlashInfer supports this natively. | M |
| 7.7 | **Speculative decoding on CUDA** | N-gram proposer already exists. Add draft model speculator: small model proposes K tokens, target model verifies in one forward pass. Need multi-token forward + rejection sampling. | L |
| 7.8 | **LoRA serving on CUDA** | Load LoRA adapter weights, apply low-rank updates during forward pass. Requires batched LoRA matmul kernels for multi-LoRA serving. | L |
| 7.9 | **Benchmark suite** | Automated benchmarks comparing Rust CUDA vs Python vLLM on standard models (LLaMA-7B, LLaMA-70B TP4, Mixtral-8x7B) across metrics: TTFT, ITL, throughput (tok/s), memory usage. | M |

---

## Dependency Graph

```
Phase 0 (Candle CUDA works)
    │
    ├── Phase 1 (GPU memory + KV cache on GPU)
    │       │
    │       ├── Phase 2 (FlashAttention)
    │       │       │
    │       │       ├── Phase 4 (CUDA Graphs)
    │       │       │
    │       │       └── Phase 7.4 (Mixed prefill+decode)
    │       │
    │       ├── Phase 3 (Fused kernels)
    │       │
    │       └── Phase 6 (Quantization)
    │
    └── Phase 5 (NCCL + Tensor Parallelism)
            │
            └── Phase 5.9 (Pipeline Parallelism)
```

Phases 2, 3, 5, and 6 can be worked on **in parallel** after Phase 1 is complete.

---

## Estimated Effort by Phase

| Phase | Description | T-shirt | Key Dependency |
|-------|-------------|---------|----------------|
| **0** | Candle CUDA E2E | **S** (1-2 weeks) | CUDA-capable machine |
| **1** | GPU memory + KV cache | **M** (2-3 weeks) | Phase 0 |
| **2** | FlashAttention FFI | **XL** (4-6 weeks) | Phase 1, FA2 source |
| **3** | Fused CUDA kernels | **L** (3-4 weeks) | Phase 1 |
| **4** | CUDA Graphs | **L** (3-4 weeks) | Phase 2 |
| **5** | NCCL + TP | **XL** (4-6 weeks) | Phase 0 |
| **6** | Quantization kernels | **XL** (6-8 weeks) | Phase 1 |
| **7** | Advanced perf | **XL** (ongoing) | Phases 2-6 |

**Total to "production CUDA parity"** (Phases 0-4): ~14-19 weeks
**Total to "full CUDA parity"** (all phases): ~30-40 weeks

---

## What We Get for Free (Already Works)

These features are already implemented and will work on CUDA once Phase 0 is complete:

- All 11 candle model architectures (LLaMA, Mistral, Qwen2/3, Phi-3, Gemma2, DeepSeek V2/V3, Command R, Qwen3 MoE, Mixtral)
- All sampling methods (greedy, temperature, top-k/p, min-p, penalties, logprobs)
- Structured output / guided decoding (outlines-core runs on CPU logits)
- Tool calling + streaming
- Chat templates
- GGUF quantized models (via candle QMatMul — runs on CUDA)
- OpenAI-compatible API (chat, completions, streaming, n parameter)
- Scheduler, engine core, async engine
- Paged KV cache (block allocation, eviction, prefix caching)
- Speculative decoding (n-gram proposer)

---

## Key Technical Decisions

### 1. Build System for CUDA Kernels

**Decision**: Use `cc` crate in `vllm-kernels/build.rs` with nvcc compiler.

```rust
// vllm-kernels/build.rs (sketch)
#[cfg(feature = "cuda")]
fn main() {
    let cuda_files = glob("csrc/*.cu");
    cc::Build::new()
        .cuda(true)
        .flag("-gencode=arch=compute_80,code=sm_80")  // Ampere
        .flag("-gencode=arch=compute_89,code=sm_89")  // Ada
        .flag("-gencode=arch=compute_90,code=sm_90")  // Hopper
        .files(cuda_files)
        .compile("vllm_kernels");
    // bindgen for C header → Rust FFI
}
```

### 2. FlashAttention Integration Strategy

**Decision**: Vendor FA2 source (MIT/BSD license), compile via build.rs, expose through the existing `AttentionKernels` trait.

Alternative considered: `flash-attention-rs` crate — but it may not track upstream FA2 closely enough. Vendoring gives us control over version and patches.

### 3. Candle vs. cudarc for Tensor Management

**Decision**: Keep candle as the primary tensor library (model code, weight loading, basic ops). Use cudarc directly only for:
- Custom CUDA kernel launches
- Memory management (VRAM queries, memory pools)
- NCCL communicator init
- CUDA graph capture/replay

This avoids rewriting the entire tensor stack while getting the performance benefits of custom kernels.

### 4. Kernel Dispatch Pattern

**Decision**: Runtime dispatch via trait objects. CandleWorker holds a `Box<dyn KernelSet>` that is either `CpuKernelSet` or `CudaKernelSet`. Model layers call kernel methods through this trait.

Alternative considered: Compile-time dispatch via generics. Rejected because it would require generic parameters throughout the model stack, making the code significantly more complex.

---

## Risk Factors

| Risk | Mitigation |
|------|------------|
| FlashAttention build complexity (CUDA version deps, arch flags) | Start with a minimal build (SM80+ only), expand arch support iteratively |
| candle-core CUDA bugs (contiguity, dtype casting) | Phase 0 validates baseline; file upstream issues early |
| Performance regression vs Python (Torch is highly optimized) | Benchmark early and often; focus on the top-3 hotspots first |
| NCCL version compatibility | Pin to NCCL 2.18+ (matches CUDA 12.x), test on specific cloud GPU instances |
| cudarc API stability | Pin version, review changelogs before upgrading |

---

## Python vLLM Files → Rust Mapping

| Python / C++ Source | Rust Target | Phase |
|---------------------|-------------|-------|
| `csrc/attention/paged_attention_v1.cu` | `vllm-kernels/src/attention.rs` (CudaAttentionKernels) | 2 |
| `csrc/attention/paged_attention_v2.cu` | `vllm-kernels/src/attention.rs` (CudaAttentionKernels) | 2 |
| `csrc/attention/merge_attn_states.cu` | `vllm-kernels/src/attention.rs` | 2 |
| `csrc/layernorm_kernels.cu` | `vllm-kernels/src/norm.rs` (CudaNormKernels) | 3 |
| `csrc/activation_kernels.cu` | `vllm-kernels/src/activation.rs` (CudaActivationKernels) | 3 |
| `csrc/pos_encoding_kernels.cu` | `vllm-kernels/src/rotary.rs` (CudaRotaryKernels) | 3 |
| `csrc/cache_kernels.cu` | `vllm-kernels/src/cache.rs` (CudaCacheKernels) | 1/3 |
| `csrc/sampler.cu` | `vllm-kernels/src/sampler.rs` (new) | 7 |
| `csrc/custom_all_reduce.cu` | `vllm-kernels/src/allreduce.rs` (new) | 5 |
| `csrc/moe/*.cu` | `vllm-kernels/src/moe.rs` (new) | 7 |
| `csrc/quantization/gptq/*.cu` | `vllm-kernels/src/quantization/gptq.rs` (new) | 6 |
| `csrc/quantization/awq/*.cu` | `vllm-kernels/src/quantization/awq.rs` (new) | 6 |
| `csrc/quantization/marlin/*.cu` | `vllm-kernels/src/quantization/marlin.rs` (new) | 6 |
| `csrc/quantization/w8a8/*.cu` | `vllm-kernels/src/quantization/fp8.rs` (new) | 6 |
| `vllm/v1/attention/backends/flash_attn.py` | `vllm-models/src/attention.rs` (FA2 dispatch) | 2 |
| `vllm/v1/worker/gpu_worker.py` | `vllm-executor/src/candle_worker.rs` (CUDA path) | 0-1 |
| `vllm/v1/worker/gpu/input_batch.py` | `vllm-executor/src/input_batch.rs` (new) | 7 |
| `vllm/compilation/cuda_graph.py` | `vllm-executor/src/cuda_graph.rs` (new) | 4 |

---

## Success Metrics

| Metric | Target | Python vLLM Baseline |
|--------|--------|---------------------|
| TTFT (time to first token) | Within 1.2x | ~100ms (LLaMA-7B, 512 prompt) |
| ITL (inter-token latency) | Within 1.1x | ~10ms (LLaMA-7B, batch=1) |
| Throughput (tok/s, batch=32) | Within 1.3x | ~2000 tok/s (LLaMA-7B, A100) |
| Max model size (single GPU) | Same | 70B FP8 on A100-80GB |
| Max model size (4-GPU TP) | Same | 405B FP8 on 4×A100 |
| Memory efficiency | Within 1.1x | ~95% VRAM utilization |
| GGUF on CUDA | Functional | N/A (Python uses different path) |
| GPTQ/AWQ on CUDA | Within 1.3x | ~3000 tok/s (13B-GPTQ, A100) |
