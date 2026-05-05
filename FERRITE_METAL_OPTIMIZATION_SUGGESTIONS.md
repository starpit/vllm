# Ferrite Metal — Fusion & Optimization Suggestions

**Audience**: the agent building out `ferrite-metal-impl-lib`, `ferrite-metal-kernels`, and the MSL codegen path in `ferrite-forward-macro`.

**Source**: derived from a read of `mlx-source-for-bob/mlx/` (compile.cpp, backend/metal/compiled.cpp, fast.cpp, kernels/, steel/) plus Apple Silicon hardware characteristics (M1 → M4).

**Goal**: tell you which fusions to prioritize, which to skip, what MLX already does well that we should match (not re-derive), and where we can credibly beat MLX on Metal.

---

## TL;DR priority list

Implement in this order. Each item is a `Metal*Impl` satisfying the existing `Implementation` trait used by the solver, following the `FusedQkvRopeCache` / `FusedGateUpSiluMul` / `CutlassFusedRmsNormGemm` naming pattern already in `ferrite-forward/src/info.rs`.

| # | Impl | Why here | Rough LoC |
|---|------|----------|-----------|
| 1 | `MetalRmsNormImpl` (plus bf16/f16 variants) | Already stubbed; unblocks everything else | ~200 |
| 2 | `MetalSdpaDecodeImpl` (vector, Q_seq≤8) | Decode is 90%+ of user-visible latency | ~400 |
| 3 | `MetalFusedQkvRopeCacheImpl` | Single biggest bandwidth win per decoder layer | ~400 |
| 4 | `MetalFusedAddRmsNormImpl` | Fuses residual + next norm (MLX can't) | ~150 |
| 5 | `MetalFusedGateUpSiluMulImpl` | SwiGLU inner; wins on memory-bound MLP | ~350 |
| 6 | `MetalQuantizedMatmulImpl` (q4_0 / q4_K / q8_0) | 4× bandwidth cut = 4× decode speedup | ~500 |
| 7 | `MetalSdpaPrefillImpl` (tiled flash) | Prefill throughput; can use MLX steel as reference | ~600 |
| 8 | `MetalFusedRmsNormQkvImpl` | Norm-epilogue-into-GEMM, beyond MLX tier 3 | ~400 |
| 9 | `MetalDecoderBlockICB` (session-level) | Dispatch amortization; tier-zero for latency | ~300 |

Items 1–6 should land before any model end-to-end; 7–9 are performance phase.

---

## 1. Memory-bandwidth fusions (the ones that actually matter)

Apple Silicon's arith/bandwidth ratio is brutal (100–550 GB/s DRAM, 2.6–4.5 TF16 TFLOPS). Decode is memory-bound for almost everything. **Every materialized intermediate round-trips through DRAM and costs measurable ns.**

### 1.1 `MetalFusedAddRmsNormImpl` — residual + next norm
**Pattern**: `x' = rmsnorm(x + residual, w, eps)`

Pre-norm and post-norm residual patterns both hit this. Running them as two kernels reads `x` twice. One kernel, reading `x` and `residual` once each.

**MSL skeleton** (follow `mlx/backend/metal/kernels/rms_norm.metal` `rms_single_row`):
```metal
// Pass 1: sum of squares across x+residual, reduce within threadgroup
// Pass 2: write out w * (x+residual) * rsqrt(mean+eps)
//         AND write out (x+residual) if residual slot is also consumed downstream
```

**Output slots**: keep the add-result live too — the next layer's residual needs it. Emit two output buffers.

**Ferrite angle MLX can't match**: MLX's `fast::rms_norm` doesn't take a residual. MLX users either do `rmsnorm(a+b)` inside `mx.compile(...)` and hope the tier-1 fuser handles the add (it does, as a separate kernel pre-norm, because norm is not fusable), or eat the extra pass. We can emit a single kernel.

### 1.2 `MetalFusedQkvRopeCacheImpl` — paged KV write in same kernel as RoPE
**Pattern**: `q, k, v = split(qkv); q, k = rope(q, k, pos); kv_cache[block_table[...]] = (k, v)`

MLX issues these as three separate kernels (split, rope, cache-write). Each one streams the K/V tensors through DRAM. On a decode step with bs=1, seq=1, h_kv=8, d=128 that's ~8 KB/layer of pointless traffic per step, plus kernel-launch overhead.

**One kernel**:
- thread grid: `(num_heads, batch, 1)`
- each thread loads its Q/K/V row from the fused `qkv` buffer
- computes rotation pairs in registers (no freq table materialization)
- for K/V, computes `block_idx = block_table[batch * max_blocks + (pos / block_size)]` and writes to paged cache at `k_cache[block_idx, pos % block_size, head, d]`
- Q output written to a contiguous output buffer

**Copy from**: `mlx/backend/metal/kernels/rope.metal` for the rotation math; there's no MLX equivalent of the paged-write fusion to copy.

### 1.3 `MetalFusedGateUpSiluMulImpl` — SwiGLU
**Pattern**: `silu(gate_proj(x)) * up_proj(x)`

If `gate_proj` and `up_proj` are separate GEMMs emitting to DRAM, you read each back for the elementwise. If you've fused them into a single `gate_up` projection (common — concatenate weights in column dim), the output is one tensor `[B, 2*I]` that gets read once and written once as `[B, I]`.

**Two variants to emit**:
- `MetalFusedGateUpSiluMulImpl_SeparateGemms`: kernel takes `gate_out` and `up_out`, writes `silu(g)*u`. Trivial — a single elementwise kernel. This is all tier-1 MLX can do.
- `MetalFusedGateUpSiluMulImpl_Epilogue`: **fused into the steel-equivalent GEMM epilogue** so the intermediate never reaches DRAM. Requires writing your own GEMM with a parameterized epilogue, or finding the right template seam in the MLX steel headers. This is tier-3 territory and where MLX stops — they don't have SiLU/GeLU as epilogues.

Ship the Epilogue variant; the SeparateGemms one is a baseline for the solver to compare against.

**GELU variant** (`MetalFusedGateUpGeluMulImpl`) for Gemma2/3, same shape.

### 1.4 `MetalFusedAddRmsNormQkvImpl` — norm output feeds QKV projection
**Pattern**: `x' = rmsnorm(x + residual); qkv = x' @ W_qkv`

This is what MLX's tier-3 `TransformAdd`/`TransformAxpby` gets you a *subset* of (only adds a constant bias). A full fused norm-output-as-A-load means the normalized activation never leaves L1 for the GEMM A-load.

Implementation sketch: adapt `steel/gemm/kernels/steel_gemm_fused.h`. The A-prologue is normally `load(A[m,k])`; replace with `load_and_normalize(x[m,k], x_residual[m,k], w[k], running_rsqrt)` where the threadgroup has pre-computed the rsqrt. This is the op that actually uses Apple's shared threadgroup memory to earn its keep.

**If it's too risky for phase 1**, ship as two kernels (FusedAddRmsNorm → Gemm) and revisit later. But the analytical model should flag this pattern as high-value so the solver knows the ceiling.

---

## 2. Attention (copy MLX, don't re-derive)

### 2.1 `MetalSdpaDecodeImpl` (Q_seq ≤ 8)
**Port**: `mlx/backend/metal/kernels/sdpa_vector.h` (the `sdpa_vector` kernel, ~170 lines). Near-verbatim is fine.

Key features to preserve:
- Online softmax (running max + sum, Flash-style) — **non-negotiable on bandwidth-bound hardware**.
- GQA via `gqa_factor` parameter; don't materialize expanded K/V.
- Scale baked into Q-load (`q[i] = scale * queries[i]`).
- Mask via function constants (`bool_mask`, `float_mask`, `do_causal` all compiled out when not used).
- Attention sinks support.

**Ferrite-specific**: it should accept a paged-KV block table and compute `key_ptr = k_cache[block_table[b,p/bs]] + (p%bs)*kv_stride`. MLX's version assumes contiguous KV; paged is our addition.

### 2.2 `MetalSdpaDecodeSplitKVImpl` (long context)
**Port**: `sdpa_vector_2pass_1` + `sdpa_vector_2pass_2` from the same file. Threshold for use: context ≥ ~1024 tokens (tune per-chip in the cost model).

### 2.3 `MetalSdpaPrefillImpl` (Q_seq > 8, tiled flash)
**Port**: `mlx/backend/metal/kernels/steel/attn/` + `scaled_dot_product_attention.metal`. This is the biggest undertaking — steel's attention is template-heavy and depends on its whole simdgroup-matrix tiling infrastructure. Options:
- **Short-term**: use MLX's kernel binary directly if license allows (MIT). Build-time dependency on MLX's kernel source.
- **Medium-term**: port the kernel, keep the template structure.
- **Long-term**: replace with Metal 4 tensor API (see §5).

### 2.4 Sliding-window attention (Gemma2, Mistral)
Add as a function-constant branch inside the decode/prefill kernels — `do_sliding_window` bool plus a `window_size` constant. Cheap to add once the base kernels exist.

---

## 3. Quantized matmul (`MetalQuantizedMatmulImpl`)

GGUF / MLX quants all share one property on Apple: **weights stream from DRAM, dequant happens in registers, GEMM proceeds**. Never materialize a dequantized weight tensor.

**Priority quant formats** (in order of usefulness):
1. **q4_0 / q4_K_M** (GGUF) — most common; 4-bit weights, 16 or 32 weights per block with scale (+ min for K-quants).
2. **mlx-4bit** — MLX's native quant; Apple already has tuned kernels.
3. **q8_0** (GGUF) — 8-bit; higher quality / smaller speedup.
4. **q5_K_M, q6_K** (GGUF) — common but less critical.

**Reference kernels**:
- MLX: `mlx/backend/metal/kernels/quantized.metal` + `quantized.h` (for mlx-4bit); `qmm_t`, `qmm_n`, `qmm_splitk`, `gather_qmm` for MoE.
- llama.cpp: `ggml-metal.metal` for q4_K/q6_K/q8_0 — battle-tested, ~8 years of tuning.

**Solver hint**: emit both a `gemv` path (bs=1, no tiling) and a `gemm` path (bs>1, tile in M). Crossover around batch 4–8 depending on chip.

**MoE**: a `MetalFusedGatedQuantMoEImpl` with gather-qmm-scatter pattern. This is big — Qwen MoE, Mixtral, DeepSeek all need it. MLX has `gather_qmm_nax` as reference.

---

## 4. Dispatch-level optimization (`MetalDecoderBlockICB`)

**The thing MLX doesn't do that matters most.**

MLX issues each op as a separate `computeEncoder.dispatchThreadgroups`. On M-series, per-dispatch overhead is ~0.5–2 μs. A 28-layer Qwen2.5-7B decode step issues ~250 dispatches, adding ~250–500 μs of pure overhead per token. That's ~5–15% of total decode time for small models, and gets worse as models shrink or chips improve.

**Solution — Indirect Command Buffer (ICB) batching**:
- Build one ICB per decoder block (or per full model, if shapes are static).
- Encoder records all dispatches once at init time with `[[buffer(N)]]` bindings that point to per-request scratch.
- Decode step = `executeCommandsInBuffer(icb, range)`. GPU-side, zero CPU involvement per dispatch.

**Constraints**:
- Requires static-ish kernel selection. If your solver picks `sdpa_vector` vs `sdpa_2pass` based on runtime seq length, you need both recorded and branch at the ICB level (or re-record on seq-length phase change).
- Shape-dependent kernels (prefill, bs>1 decode) usually rebuild ICB per call. That's fine — cost is amortized per step, not per dispatch.

**Apple docs**: `MTLIndirectCommandBuffer`, `MTLComputeCommandEncoder.executeCommandsInBuffer`. Requires `GPUFamilyMetal3` (M1+, so fine).

**Phase this in last.** Land correctness with plain dispatch first; ICB is a perf gate, not a feature gate.

---

## 5. M3+ specific

### 5.1 Dynamic Caching — don't spill what you don't have to
M1/M2 statically partition register files per-kernel. A fused kernel with high register pressure reduces the **whole GPU's** occupancy, which is why MLX's fused kernels are often narrower than they could be. M3+ allocates registers per-thread dynamically.

**What changes in codegen**:
- Your solver cost model should have two register-pressure coefficients: a steep cliff on M1/M2 (each register over ~32 costs real occupancy), a gentler curve on M3+ (spilling only matters past ~64).
- On M3+, **widen the fused region**. Example: on M1 you might cap fusion at "rmsnorm + gate_up+silu+mul + down_proj" as three kernels because the middle one's register footprint is already high. On M3+, try merging the silu+mul into down_proj's A-prologue.
- Emit kernel variants with a `#define MAX_LIVE_REGISTERS` guard that the solver picks based on target profile.

**Cost-sweep implication**: `ferrite-metal-cost-sweep` should separately measure M1/M2 vs M3+ and the solver should consult the right table. Don't assume cross-chip generalization for fusion width.

### 5.2 Native bf16 — make it the default
M1/M2 do bf16 *storage* natively but lower arithmetic to FP32 in many paths. M3+ has native bf16 arithmetic.

- Emit bf16 kernel variants with bf16 accumulators where it doesn't hurt accuracy (norm stats, attention scores still accumulate in fp32).
- For attention softmax, keep fp32 accumulators on all chips — the numerical range matters more than the perf.
- For GEMM: bf16 input, bf16 output, **fp32 accum on all chips** (matches what NVIDIA tensor cores do).

### 5.3 ResidencySet — pin weights at load time
`MTL::ResidencySet` (macOS 15+, GPUFamilyMetal3) lets you mark buffers as wired-in-memory so the OS doesn't page them out under pressure. For a 14 GB model on a 16 GB M3 Pro, this is the difference between 50 t/s and 5 t/s when the OS starts compacting memory.

**Implementation**: `ferrite-metal-kernels::MetalBuffer::pin()` method that calls `requestResidency()` on the buffer's residency set. Call after weight load, before first forward.

MLX already does this — copy their pattern from `mlx/backend/metal/resident.cpp`.

### 5.4 Fast fences — sub-μs CPU↔GPU sync
`MTL::SharedEvent` is ~10 μs per signal/wait. MLX's "fast fence" path (polled uint32 buffer, `GPUFamilyMetal3` + macOS 15+) is ~1 μs. For decode with one CPU step per token, this is a real fraction of end-to-end latency.

Reference: `mlx/backend/metal/fence.cpp` (also has the fallback for older OS).

---

## 6. M4+ specific

### 6.1 Metal 4 tensor primitives — the future SDPA/GEMM
WWDC 2024 introduced `MTLTensor`, `metal::tensor<>`, and hardware tensor ops on M4. This is Apple's analog to Hopper's wgmma / Blackwell's tensor memory.

**Status**: documented publicly, SDK available. MLX does **not** yet use these in its production path — their steel kernels are still simdgroup_matrix-based.

**Opportunity**: a `MetalTensorOpGemmImpl` that uses `metal::tensor<>` for the MMA could beat MLX on M4 Max. This is real differentiation, not just catch-up.

**Risk**: API is new, docs are thin, compatibility matrix is fuzzy. Ship as an opt-in `Impl` the solver prefers only on M4+ with recent-OS gate, with `MetalSteelGemmImpl` as the always-works fallback.

### 6.2 Larger L2 / better cache policy
M4 Max has substantially more L2 than M3 Max, and the cache policy is smarter about KV-cache patterns. Tuning implication: **larger K/V block sizes for paged attention on M4**. Where M1/M2 want block_size=16 to fit hot blocks in cache, M4 can profitably use block_size=32 or 64. Make it a profile parameter, not a constant.

### 6.3 SME on CPU (M4)
Not a GPU feature but worth mentioning: M4's CPU has Scalable Matrix Extension. If any part of the pipeline falls back to CPU (sampling, logit processing, embedding lookup for rare tokens), SME-aware kernels can be 5–10× faster than NEON. Probably out of scope for phase 1 but keep in mind for the sampler hot path.

---

## 7. Things NOT to bother with (compared to the CUDA path)

CUDA-ferrite has optimizations that do not translate:

- **Async memcpy / cp.async overlap** — no TMA analog. `MTL::Buffer` copies are what they are. Don't model async overlap in the solver for Metal.
- **Warp specialization / producer-consumer pipelines** — no wgmma.commit-style fence; threadgroup barriers are cheap and synchronous. Skip.
- **Occupancy tuning for 132 SMs** — M-series has 10–40 "GPU cores". The optimization targets are different; don't port the CUDA occupancy heuristics.
- **NVSHMEM / multi-GPU collectives** — no multi-GPU Apple silicon exists. TP/PP is out of scope for Metal.
- **FP8** — not supported in hardware on any current Apple Silicon. If it's ever added, it'll be M5+.
- **Graph capture / CUDA Graphs** — **replaced** by ICB (§4). Same motivation, different API.

---

## 8. MLX kernels to copy as-is vs re-derive

### Copy verbatim (license permitting; MLX is MIT)
- `rms_norm.metal` → `MetalRmsNormImpl`
- `layer_norm.metal` → `MetalLayerNormImpl` (for BERT-family models)
- `rope.metal` → use as starting point for `MetalFusedQkvRopeCacheImpl`; add the paged-write
- `sdpa_vector.h` → `MetalSdpaDecodeImpl`
- `softmax.metal`, `logsumexp.metal` → general utilities
- `quantized.metal` (mlx-4bit) → `MetalQuantizedMatmulImpl` (mlx-4bit variant)

### Re-derive (we can beat MLX)
- **Residual+norm fusion** — MLX can't express this; emit our own.
- **GEMM epilogue fusion beyond bias/axpby** — our solver can pick which elementwise chain goes in the epilogue.
- **Paged KV + RoPE + cache-write single kernel** — MLX has no paged path.
- **ICB-batched decoder block** — MLX doesn't do this.

### Port but re-architect
- **Steel GEMM** — start from MLX's steel as a known-good reference, then parameterize the epilogue more aggressively. Long-term, replace with Metal 4 tensor API on M4+.
- **Steel attention (prefill)** — same story.

---

## 9. Suggested phase ordering (refines `FERRITE_METAL_ARCHITECTURE.md`)

Current plan has Phases 1–6 over 8–12 weeks. Suggest resequencing for end-to-end as early as possible:

**Phase 1 (weeks 1–2) — Foundation + one model**: land `MetalRmsNormImpl` + basic Linear + unquantized `MetalSdpaDecodeImpl` + `MetalFusedQkvRopeCacheImpl`. Goal: a BF16 Llama-style forward pass ends-to-end, even if slow. Everything below is a perf iteration from this baseline.

**Phase 2 (weeks 3–4) — Solver + cost sweep**: wire cost model, get real numbers per chip. Measure the gap vs MLX.

**Phase 3 (weeks 5–6) — The big wins**: `MetalFusedAddRmsNormImpl`, `MetalFusedGateUpSiluMulImpl` (both variants), `MetalQuantizedMatmulImpl` (q4_K_M + mlx-4bit). These should close most of the MLX gap.

**Phase 4 (weeks 7–8) — Epilogue fusion (beat MLX)**: `MetalFusedAddRmsNormQkvImpl`, GEMM-with-SiLU-epilogue. Actually pull ahead.

**Phase 5 (weeks 9–10) — Dispatch amortization (beat MLX more)**: ICB batching. M3+ fast fences. Residency pinning.

**Phase 6 (weeks 11–12) — M4 tensor APIs**: `MetalTensorOpGemmImpl`, opt-in. New ceiling on M4 Max.

Prefill (`MetalSdpaPrefillImpl`) slots in somewhere around phase 3–4; it's substantial work but not on the latency critical path for user-facing decode.

---

## 10. Testing & verification

Borrow ferrite's existing golden-trace infrastructure:
- Every `Metal*Impl` should have a `vs MLX` parity test: run the same op via `mlx_core::fast::*` and compare tensors. Atol ~1e-3 for bf16, 1e-4 for f16.
- End-to-end: run ferrite-metal and MLX on the same prompt with a fixed-seed model, compare top-1 tokens across 256 steps. Any divergence is a bug.
- Performance: `ferrite-metal-cost-sweep` should record (chip, op, shape, measured_us) and the solver consumes this. Don't hand-code costs.

---

## Open questions for the maintainer to decide

1. **Minimum silicon target**: M1 (broadest compat, constrains codegen), or M3+ (cleaner Dynamic Caching story, loses ~40% of installed base)? Recommendation: M3+ baseline, M1/M2 compat path for inference correctness but don't optimize.
2. **Prefill strategy**: port steel, depend on MLX binaries, or build from scratch on Metal 4 tensor API? Recommendation: port steel for phase 3, revisit Metal 4 in phase 6.
3. **ICB granularity**: per-block or per-model? Recommendation: per-block for flexibility, full-model ICB for static-shape decode as a perf escape hatch.
4. **Quant format priority**: GGUF-first or MLX-quant-first? Depends on target user — GGUF is the broader ecosystem, MLX quant integrates cleaner. Recommendation: GGUF q4_K_M first, mlx-4bit second.
