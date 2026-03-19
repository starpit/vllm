# Ferrite: A Rust-Native Megakernel Compiler for LLM Inference

> Research document — March 19, 2026
>
> A comprehensive feasibility study for building 100% Rust CUDA kernels
> using proc macros, LLVM, and megakernel fusion for transformer inference.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Motivation](#motivation)
3. [Background Research](#background-research)
   - [LLVM NVPTX Backend](#llvm-nvptx-backend)
   - [Rust's nvptx Target](#rusts-nvptx-target)
   - [Existing Rust GPU Projects](#existing-rust-gpu-projects)
   - [Triton's Architecture](#tritons-architecture)
   - [Megakernel Research](#megakernel-research)
4. [Architecture](#architecture)
5. [The Codegen Path](#the-codegen-path)
6. [Integration with vllm-rs](#integration-with-vllm-rs)
7. [Competitive Analysis](#competitive-analysis)
8. [Risk Assessment](#risk-assessment)
9. [Team and Timeline](#team-and-timeline)
10. [Phase 0: Proof of Concept](#phase-0-proof-of-concept)
11. [Sources](#sources)

---

## Executive Summary

**Ferrite** is a proposed Rust-native GPU compiler for LLM inference. Rather than
replacing CUTLASS kernel-for-kernel, Ferrite targets the frontier of GPU
optimization: **megakernels** that fuse entire transformer layers (or full forward
passes) into single persistent GPU kernels, eliminating inter-kernel HBM
round-trips and launch overhead.

The core thesis: individual GEMM kernels are a solved problem (CUTLASS, cuBLAS).
The remaining performance gains are at the **boundaries between operations** — the
30-40% of inference time spent writing intermediates to HBM and re-reading them.
Three independent research groups (Mirage MPK, FlashFormer, Hazy "No Bubbles")
have demonstrated 1.5-6.7x speedups with megakernel approaches, validating this
thesis.

Rust is uniquely suited because:
- **Proc macros** can analyze model dataflow and generate fused kernels at compile
  time (zero warmup, unlike torch.compile's JIT)
- **The type system** can encode hardware constraints (shared memory limits,
  register budgets, warp roles) as compile-time errors
- **Const generics** enable tile-level algebra without C++ template complexity
- **LLVM integration** (via inkwell) provides the same NVPTX backend that Triton
  uses, with identical codegen quality

Estimated effort: 6-8 engineers, 18-24 months to production.

---

## Motivation

### The inter-kernel bottleneck

A typical transformer attention layer in vLLM executes as separate kernel launches:

```
RMS Norm  → [write HBM] → launch → QKV Linear  → [write HBM] → launch →
Rotary    → [write HBM] → launch → Attention    → [write HBM] → launch →
Out Linear → [write HBM] → launch → Add Residual → [write HBM]
```

Each arrow is:
- A round-trip through HBM (~3.35 TB/s on H100, but still microseconds per tensor)
- A kernel launch overhead (3-10μs)
- A pipeline drain (the GPU finishes one kernel before starting the next)

For large batch, compute-bound workloads, this overhead is amortized. But for the
latency-sensitive cases that matter most in LLM serving (small batch, long context,
time-to-first-token), inter-kernel costs can dominate.

### Why not torch.compile?

torch.compile (via Inductor) does fuse operations, but:
- It fuses **pointwise chains** and **epilogues around GEMMs** — it does NOT fuse
  across GEMM boundaries
- It has multi-second warmup (JIT compilation on first request)
- It recompiles when shapes change
- It requires Python in the serving path
- Its fusion heuristics are conservative (runtime shape uncertainty)

A compile-time system with full model knowledge can be more aggressive.

### Why not just use CUTLASS via FFI?

CUTLASS optimizes individual kernels to near-peak performance. But it cannot:
- Fuse across operation boundaries (norm → linear → activation)
- Keep intermediates in registers/SRAM across operations
- Pipeline weight loading across operations
- Generate persistent megakernels

Ferrite uses CUTLASS-equivalent techniques for individual operations but composes
them into fused megakernels — a strictly more powerful optimization scope.

---

## Background Research

### LLVM NVPTX Backend

LLVM has an actively maintained NVPTX backend that generates PTX from LLVM IR. This
is the same backend Triton uses. NVIDIA is investing in it as a first-class target
(upstreaming Blackwell support, converging NVVM IR with upstream LLVM).

#### Intrinsic coverage (as of March 2026)

| Feature | LLVM Intrinsic? | Notes |
|---------|----------------|-------|
| TMA (cp.async.bulk.tensor) | Yes, 1D-5D, Tile+Im2Col | Well covered |
| mbarrier (arrive/wait/init) | Yes, comprehensive | CTA + cluster scope |
| setmaxnreg | Yes | Dynamic register adjustment |
| elect.sync | Yes | Leader election |
| cp.async.bulk (non-tensor) | Yes | Bulk copy |
| **wgmma.mma_async (core MMA)** | **Fence/commit only** | **Core dispatch requires inline PTX asm** |
| **tcgen05.mma (Blackwell)** | **Partial, broken on sm103** | **Triton reverted to inline asm** |

**Critical finding**: The surrounding infrastructure (TMA, barriers, memory
management) is well-covered by LLVM intrinsics, but the core tensor core compute
instructions (wgmma, tcgen05) require inline PTX assembly. This is true for every
project using LLVM (Triton, MLIR, Mojo). It is not a Rust-specific limitation.

#### NVVM IR convergence

NVIDIA's NVVM IR specification jumped from LLVM 7 to **LLVM 21** for Blackwell+
(sm100). This signals convergence between NVIDIA's internal compiler and upstream
LLVM, meaning the inline-asm escape hatch may become less necessary over time. But
"over time" likely means years.

### Rust's nvptx Target

**Status: Tier 2 with `core` only (no `std`), nightly required.**

| Feature | Status |
|---------|--------|
| Basic compilation to PTX | Works |
| `extern "ptx-kernel"` ABI | Unstable (`#![feature(abi_ptx)]`) |
| Inline PTX asm | Unstable (`#![feature(asm_experimental_arch)]`), 3 register classes |
| `core::arch::nvptx` | Bare minimum: `_syncthreads`, thread indexing, packed f16 |
| Warp shuffles | Not available |
| Tensor cores | Not available |
| TMA / cp.async | Not available |

**Active improvements:**
- Compiler team proposal #965 (Feb 2026): narrowing to SM 7.0+, PTX ISA 7.0 default
- Compiler team proposal #927: `llvm-bitcode-linker` as default (landed)
- Tracking issues for NVPTX arch intrinsics (#111199), shared memory (#135516)

**Assessment**: rustc's nvptx target is too limited for production GPU kernels.
Our recommended approach bypasses it entirely by using inkwell to emit LLVM IR
directly, targeting the NVPTX backend without going through rustc's GPU codegen.

### Existing Rust GPU Projects

| Project | Approach | Tensor Cores | Maturity | Relevance |
|---------|----------|-------------|----------|-----------|
| **rust-cuda** | rustc backend → NVVM IR → PTX | No | Early dev, ~7 contributors | Proves Rust→GPU compilation works |
| **CubeCL** | Proc macro JIT → CUDA C → NVRTC | Yes (basic) | Alpha, used by Burn | Closest to our approach |
| **rust-gpu** | rustc backend → SPIR-V | No | Active, Vulkan only | Different target |
| **cudarc** | Bindings only (CUDA driver API) | Via cuBLAS | Active, used by candle | Useful for kernel launching |

**CubeCL/Burn** is the most relevant prior art. Their matmul kernels are
competitive with cuBLAS. Their approach: `#[cube]` proc macro → custom IR → CUDA C
strings → NVRTC runtime compilation. This validates the "Rust metaprogramming for
GPU kernels" concept but uses runtime JIT rather than compile-time codegen.

**No existing Rust project has demonstrated CUTLASS-class kernels** (high-performance
GEMM with warp specialization, software pipelining, tensor cores).

### Triton's Architecture

Triton is the best proof that LLVM NVPTX can produce high-performance GPU code
without nvcc. Understanding its architecture informs Ferrite's design.

#### Compilation pipeline

```
Python (@triton.jit)
  → Triton IR (TTIR) — tile-level, hardware-agnostic
  → TritonGPU IR (TTGIR) — adds layout encodings (how tiles map to threads)
  → TritonNvidiaGPU IR — NVIDIA-specific ops (wgmma, TMA)
  → LLVM IR — with NVVM intrinsics + inline PTX asm
  → PTX — via LLVM NVPTX backend
  → SASS — via ptxas (NVIDIA proprietary, required)
```

#### Key abstractions

- **Layout encodings**: Attributes on tensor types describing data distribution
  across threads/warps/CTAs (Blocked, Shared, MMA, DotOperand). This is the single
  most complex piece of Triton — it determines memory access patterns and bank
  conflict avoidance.
- **`tt.dot`**: Tile-level matrix multiply that maps to tensor core instructions.
  The compiler selects wgmma vs mma.sync based on architecture and operand layout.
- **Software pipelining**: Transforms loops into prologue/steady-state/epilogue
  with `num_stages` concurrent iterations of async loads in flight.

#### Codebase size

~100,000 lines of C++ (MLIR dialects, passes, lowering) plus ~50,000 lines of
Python (frontend, autotuner, tests). Maintained since 2019 by Meta, NVIDIA, AMD,
Microsoft, and community.

#### Performance vs CUTLASS

| Workload | Triton vs CUTLASS | Root Cause |
|----------|------------------|------------|
| Hopper FP16 GEMM | 70-90% of CUTLASS | Lacks warp specialization on Hopper |
| Hopper FP8 GEMM | Significantly behind | Missing Ping-Pong scheduling |
| Blackwell Flash Attention | Near parity with cuDNN | autoWS + NVIDIA collaboration |
| Memory-bound ops | ~50% peak DRAM BW | Automatic access patterns less optimal |

**Key insight**: Triton's gap vs CUTLASS comes from scheduling decisions (warp
specialization, pipeline staging), NOT from LLVM codegen quality. The LLVM path
produces good PTX. The optimization passes above LLVM are what matter.

#### Twill: Optimal scheduling via SAT solving

A December 2025 paper from Triton research treats software pipelining and warp
specialization as a unified **constraint satisfaction problem**:
- Extracts dependence graphs from TTGIR
- Uses CBC (modulo scheduling), Yices2 (SMT), SCIP (cost normalization)
- Guarantees provably optimal schedules
- Evaluated on FMHA kernels: matches or exceeds hand-tuned schedules

This approach is directly applicable to Ferrite's fusion engine.

### Megakernel Research

The most exciting development in GPU optimization for LLM inference: fusing entire
transformer layers (or full models) into single persistent kernels.

#### Mirage Persistent Kernel (MPK) — December 2025

- Automatically transforms LLM inference into a **single megakernel**
- SM-level graph representation capturing data dependencies per-SM
- Cross-operator software pipelining and fine-grained kernel overlap
- **1.2-6.7x latency reduction** vs standard approaches
- Has an RFC for vLLM integration (issue #22201)

#### FlashFormer — May 2025

- Fuses **entire forward pass** into a single kernel
- Specialized for specific model config + hardware at compile time
- Uses shared pipelined buffer and fast synchronization
- Targets low-batch inference where launch overhead dominates

#### "Look Ma, No Bubbles!" — Hazy Research, May 2025

- Merges entire **Llama-1B forward pass** into one megakernel
- Achieves **78% of peak memory bandwidth** on H100 (vs ~50% for vLLM/SGLang)
- **1.5x+ faster** than existing inference systems
- Pipelines weight loads across "instructions" — weights for the next operation
  load while the current operation executes, eliminating bubbles

#### Deep Kernel Fusion — February 2026

- Fuses SwiGLU FFN blocks (separate GEMMs + nonlinearities) into one kernel
- **9.7% throughput improvement on A100, 13.2% on H100**
- Integrated with SGLang

#### Fusion landscape summary

| Approach | Fusion Scope | Speedup | Maturity |
|----------|-------------|---------|----------|
| torch.compile/Inductor | Pointwise chains, epilogues | 1.3-3.2x | Production |
| FlashAttention-3 | Full attention block | 2-7.6x over naive | Production |
| FlashInfer JIT | Attention + RoPE + sampling | 28-30% latency | Production |
| TensorRT-LLM | GEMM+SwiGLU, norm+residual | Varies | Production |
| Deep Kernel Fusion | SwiGLU FFN (both GEMMs) | 9.7-13.2% | Research/SGLang |
| Mirage MPK | Entire model, one megakernel | 1.2-6.7x | Research |
| FlashFormer | Entire forward pass | Meaningful at batch=1 | Research |
| No Bubbles | Entire Llama-1B forward | 1.5x+, 78% peak BW | Research |

**The trend is clear**: the field is moving from fusing 2-3 adjacent ops toward
fusing entire transformer layers or models into single persistent megakernels.

---

## Architecture

Ferrite is a layered system. Each layer builds on the one below and can be used
independently.

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Model Compiler (ferrite-compile)                  │
│  HuggingFace config → specialized inference binary          │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Fusion Engine (libfuse)                           │
│  Proc macro + compiler: dataflow graph → megakernel         │
│  SAT-solver-based warp scheduling (Twill-inspired)          │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Operation Library (libops)                        │
│  GEMM, FlashAttention, RMSNorm, Rotary, SwiGLU, etc.       │
│  Each op: Rust trait with tile-level implementation          │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Tile Engine (libtile)                             │
│  CuTe-equivalent: layout algebra, copy/MMA atoms,           │
│  software pipelining, shared memory with swizzle             │
├─────────────────────────────────────────────────────────────┤
│  Layer 0: PTX Intrinsic Library (libptx)                    │
│  Safe Rust wrappers around inline PTX asm                    │
│  ~50 intrinsics: wgmma, TMA, mbarrier, elect, setmaxnreg   │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  LLVM IR (via inkwell) → NVPTX backend → PTX → ptxas → SASS│
└─────────────────────────────────────────────────────────────┘
```

### Layer 0: PTX Intrinsic Library (`libptx`)

Safe Rust wrappers around inline PTX assembly for all modern GPU intrinsics.
Const-generic on data type, tile dimensions, and architecture.

```rust
// Example: wgmma wrapper (conceptual)
pub unsafe fn wgmma_mma_async<
    const M: u32,      // 64
    const N: u32,      // 24..256
    const K: u32,      // 16
    DTypeA: PtxDtype,  // f16, bf16, fp8, etc.
    DTypeB: PtxDtype,
    DTypeC: PtxDtype,
>(
    desc_a: &TmaDescriptor,
    desc_b: &TmaDescriptor,
    accum: &mut [DTypeC; (M * N / WARP_SIZE) as usize],
) {
    // Expands to inline PTX asm for the specific type/size combination
    // The proc macro generates all valid combinations at compile time
}
```

**Scope**: ~50 intrinsics, ~2-3 months of work. Well-bounded.

### Layer 1: Tile Engine (`libtile`)

The CuTe equivalent in Rust. This is the most architecturally critical layer.

**Layout algebra**: Compile-time layout composition via const generics.

```rust
// Layout = (Shape, Stride) with compile-time algebra
struct Layout<Shape: TileShape, Stride: TileStride> { ... }

// Composition, complement, inverse — all at compile time
type Composed = Compose<LayoutA, LayoutB>;
type Swizzled = Swizzle<Layout, B, M, S>;
```

**Atoms**: Traits for copy and MMA operations.

```rust
trait CopyAtom {
    type SrcLayout;
    type DstLayout;
    fn copy(src: &TileRef<Self::SrcLayout>, dst: &mut TileRef<Self::DstLayout>);
}

trait MmaAtom {
    type LayoutA;
    type LayoutB;
    type LayoutC;
    const M: u32;
    const N: u32;
    const K: u32;
    fn mma(a: &TileRef<Self::LayoutA>, b: &TileRef<Self::LayoutB>,
           c: &mut TileRef<Self::LayoutC>);
}
```

**Software pipelining**: Async pipeline with configurable stages.

```rust
struct AsyncPipeline<const STAGES: usize> {
    barriers: [MBarrier; STAGES],
    buffers: [SharedMemBuffer; STAGES],
}

impl<const STAGES: usize> AsyncPipeline<STAGES> {
    fn producer_acquire(&self, stage: usize);
    fn producer_commit(&self, stage: usize);
    fn consumer_wait(&self, stage: usize);
    fn consumer_release(&self, stage: usize);
}
```

**Scope**: 6-12 months. This is the hardest layer — requires deep GPU architecture
knowledge and careful design of the const-generic type-level algebra.

### Layer 2: Operation Library (`libops`)

Individual operations built on Layer 1, each as a Rust trait.

```rust
trait GemmOp {
    type DTypeA: Numeric;
    type DTypeB: Numeric;
    type DTypeC: Numeric;
    type Epilogue: EpilogueFn;
    const TILE_M: u32;
    const TILE_N: u32;
    const TILE_K: u32;
    const STAGES: u32;

    /// Returns the tile of C in registers (not written to global memory).
    /// This is critical for fusion — the caller decides where the output goes.
    fn compute_tile(
        a: &GmemTileIter<Self::DTypeA>,
        b: &GmemTileIter<Self::DTypeB>,
        smem: &mut SharedMemAlloc,
    ) -> RegisterTile<Self::DTypeC>;
}

trait RmsNormOp {
    type DType: Numeric;
    const HIDDEN_DIM: u32;

    /// Returns normalized values in registers.
    fn normalize(
        input: &GmemTileIter<Self::DType>,
        weight: &GmemTileIter<Self::DType>,
    ) -> RegisterTile<Self::DType>;
}
```

**Key design decision**: Operations return values in **registers**, not global
memory. This enables the fusion engine (Layer 3) to compose operations without
HBM round-trips.

Operations needed for transformer inference:
- GEMM (f16, bf16, fp8, int8, with all epilogue variants)
- FlashAttention (forward only, GQA/MQA/MHA)
- RMSNorm / LayerNorm
- Rotary position embeddings
- SwiGLU / GELU activations
- Quantize / dequantize
- Residual add
- Softmax

**Scope**: 6-12 months, parallelizable across team members.

### Layer 3: Fusion Engine (`libfuse`)

The novel contribution. A proc macro that analyzes a sequence of Layer 2 operations
and generates a single fused megakernel.

```rust
#[ferrite::fused_kernel(
    arch = sm90,
    max_batch = 64,
    model = "llama-70b",
)]
fn attention_block(
    x: &Tensor<bf16>,           // [batch, seq, hidden]
    wq: &Tensor<fp8>,           // [hidden, hidden]
    wk: &Tensor<fp8>,           // [hidden, kv_hidden]
    wv: &Tensor<fp8>,           // [hidden, kv_hidden]
    wo: &Tensor<fp8>,           // [hidden, hidden]
    norm_weight: &Tensor<bf16>, // [hidden]
    k_cache: &Tensor<bf16>,     // [layers, max_seq, kv_heads, head_dim]
    v_cache: &Tensor<bf16>,
) -> Tensor<bf16> {
    let normed = rms_norm(x, norm_weight);
    let q = linear(normed, wq);
    let k = linear(normed, wk);
    let v = linear(normed, wv);
    let q = rotary_embed(q);
    let k = rotary_embed(k);
    let attn = flash_attention(q, k, v, k_cache, v_cache);
    let out = linear(attn, wo);
    residual_add(x, out)
}
```

The proc macro:
1. **Parses** the function body into a dataflow graph of operations
2. **Analyzes** data dependencies and lifetimes
3. **Plans shared memory**: which intermediates live in SRAM vs registers
4. **Plans register budget**: ensures the fused kernel doesn't exceed per-SM limits
5. **Schedules warp roles**: assigns warps to producer (TMA load), consumer (MMA),
   or helper (norm, activation) roles
6. **Solves for optimal schedule**: Twill-inspired SAT solver finds the optimal
   initiation interval for the pipelined megakernel
7. **Generates LLVM IR**: via inkwell, with inline PTX asm for MMA/TMA instructions
8. **Compiles to cubin**: LLVM NVPTX → PTX → ptxas → embedded in binary

**When fusion hurts** (and the engine must detect this):
- Register pressure exceeds SM limits → split into 2-3 kernels
- Shared memory exceeds 228KB (H100) → spill to global with explicit prefetch
- Occupancy drops below threshold → the fusion is net negative
- Operations have incompatible access patterns (row-major output → column-major input)

**Scope**: 12-18 months. This is the most innovative and highest-risk layer.

### Layer 4: Model Compiler (`ferrite-compile`)

Takes a HuggingFace model config and weight files, generates a complete specialized
inference binary.

```bash
$ ferrite-compile \
    --model meta-llama/Llama-3.1-70B \
    --arch sm90 \
    --dtype fp8 \
    --max-batch 64 \
    --max-seq 8192 \
    --output llama-70b-h100.so
```

This generates:
- Fused megakernels for each transformer layer (or the full model)
- Shape-specialized variants for common batch sizes
- A C ABI shared library loadable by vllm-rs
- Weight loading and KV cache management code

**Scope**: 3-6 months on top of Layer 3.

---

## The Codegen Path

Based on extensive research, we bypass rustc's nvptx target entirely and emit LLVM
IR directly via inkwell.

```
Ferrite proc macro (compile time, pure Rust)
    │
    ├─ Dataflow graph analysis
    ├─ Fusion decisions
    ├─ Shared memory planning
    ├─ Warp scheduling (SAT solver)
    │
    ▼
inkwell (Rust LLVM bindings, mature, LLVM 8-21)
    │
    ├─ LLVM IR with NVVM intrinsics (TMA, barriers, etc.)
    ├─ Inline PTX asm blocks (wgmma, tcgen05)
    ├─ LLVM optimization passes (inlining, DCE, loop unrolling)
    │
    ▼
LLVM NVPTX backend
    │
    ├─ Register allocation
    ├─ Instruction scheduling
    ├─ PTX emission
    │
    ▼
ptxas (NVIDIA proprietary, required)
    │
    ├─ Final SASS optimization
    ├─ Architecture-specific tuning
    │
    ▼
CUBIN (embedded in Rust binary via include_bytes!())
```

### Why inkwell, not rustc nvptx?

| | rustc nvptx | inkwell → LLVM NVPTX |
|---|---|---|
| Intrinsic access | `core::arch::nvptx` (bare minimum) | Full LLVM IR + inline asm |
| Stability | Nightly-only, tier 2 | Stable crate, LLVM 8-21 |
| Control | Limited (compiler decides codegen) | Full control over IR |
| Layout | Rust's memory layout rules apply | We define the layout |
| Dependencies | Requires rustc nightly | Requires LLVM + ptxas |

### Why not CubeCL's approach (emit CUDA C → NVRTC)?

CubeCL emits CUDA C source strings and compiles via NVRTC at runtime. This works
for individual kernels but is problematic for megakernels:

- NVRTC adds runtime compilation latency
- Less control over barrier placement and warp scheduling
- Can't embed compiled cubins at build time
- CUDA C has its own abstraction overhead for low-level intrinsics

By going through LLVM IR directly, we get compile-time codegen with precise control.

---

## Integration with vllm-rs

Ferrite is designed as a **drop-in acceleration layer** inside the existing vllm-rs
architecture. It does not replace `vllm-cuda` — it lives inside it as an alternative
codegen backend behind a feature flag.

### Current vllm-rs GPU architecture

```
vllm-serve (axum HTTP + tonic gRPC)
  → vllm-executor (CudaWorker — model dispatch, 261KB)
    → vllm-cuda
        ├── model/llama.rs       10 kernel launches per layer, HBM round-trip each
        ├── model/qwen2.rs       Same pattern
        ├── model/deepseek.rs    Same pattern (+ MLA kernels)
        ├── layers.rs            Linear, MarlinLinear, RmsNorm, Embedding
        ├── kernels.rs           300+ unsafe extern "C" FFI declarations
        ├── csrc/                39 .cu files (5 kernel libraries)
        │   ├── vllm_kernels     norm, activation, rotary, cache, MoE, sampling
        │   ├── marlin_kernels   INT4×FP16 GEMM (Marlin)
        │   ├── cutlass_mm       FP8 GEMM (CUTLASS v4.2.1)
        │   └── flash_attn       FlashAttention-2 paged kernels
        ├── tensor.rs            GpuTensor (32B descriptor, Copy, no Drop)
        ├── alloc.rs             OwnedTensor (RAII), CachingAllocator (PyTorch-style)
        ├── device.rs            GpuDevice (streams, cuBLAS, allocator)
        └── driver.rs            Thin cudarc FFI wrappers
```

### The problem Ferrite solves

A single Llama transformer layer currently executes as:

```rust
// model/llama.rs — current forward pass (simplified)
for layer in &self.model.layers {
    h = rms_norm(h, layer.input_norm);           // kernel 1  → write HBM
    qkv = cublas.gemm(h, layer.qkv_proj);        // kernel 2  → write HBM
    q, k = rotary_embedding(qkv, positions);      // kernel 3  → write HBM
    h = flash_attn(q, k, v, kv_cache);            // kernel 4  → write HBM
    h = cublas.gemm(h, layer.o_proj);             // kernel 5  → write HBM
    h = residual_add(h, residual);                // kernel 6  → write HBM
    h = rms_norm(h, layer.post_norm);             // kernel 7  → write HBM
    gate_up = cublas.gemm(h, layer.gate_up_proj); // kernel 8  → write HBM
    h = silu_and_mul(gate_up);                    // kernel 9  → write HBM
    h = cublas.gemm(h, layer.down_proj);          // kernel 10 → write HBM
    h = residual_add(h, residual);                // kernel 11 → write HBM
}
```

**11 kernel launches and 11 HBM writes per layer.** For Llama-70B (80 layers),
that's 880 launches and 880 HBM round-trips per forward pass.

### Where Ferrite lives in the crate structure

```
vllm-rs/crates/
├── vllm-cuda/                   (EXISTING — unchanged public API)
│   ├── src/
│   │   ├── tensor.rs            UNCHANGED — GpuTensor, TensorView<'a>
│   │   ├── alloc.rs             UNCHANGED — OwnedTensor, CachingAllocator
│   │   ├── device.rs            UNCHANGED — GpuDevice
│   │   ├── driver.rs            UNCHANGED — cudarc FFI
│   │   ├── kernels.rs           UNCHANGED — kept as fallback
│   │   ├── layers.rs            UNCHANGED — kept as fallback
│   │   ├── kv_cache.rs          UNCHANGED — paged KV cache
│   │   └── model/
│   │       ├── llama.rs         MODIFIED — #[cfg(feature = "ferrite")] alternate path
│   │       ├── qwen2.rs         MODIFIED — same pattern
│   │       └── ...
│   ├── csrc/                    UNCHANGED — .cu files kept as fallback
│   └── Cargo.toml               MODIFIED — add ferrite feature flag
│
├── ferrite/                     (NEW — the compiler workspace)
│   ├── Cargo.toml               Workspace root
│   ├── ferrite-ptx/             Layer 0: PTX intrinsic wrappers
│   ├── ferrite-tile/            Layer 1: tile engine, layout algebra
│   ├── ferrite-ops/             Layer 2: GEMM, attention, norm, etc.
│   ├── ferrite-fuse/            Layer 3: fusion engine proc macro
│   └── ferrite-compile/         Layer 4: model → megakernel binary
```

### The integration point

```rust
// model/llama.rs — with Ferrite
impl LlamaForCausalLM {
    unsafe fn forward(
        &self,
        input_ids: TensorView<'_>,
        positions: TensorView<'_>,
        slot_mapping: TensorView<'_>,
        cu_seqlens_q: TensorView<'_>,
        seqused_k: TensorView<'_>,
        block_table: TensorView<'_>,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let h = kernels::embedding_gather(input_ids, self.embed_tokens.weight);

        // ── Ferrite megakernel: single launch replaces 11 × N_layers ──
        #[cfg(feature = "ferrite")]
        let h = self.ferrite_layers.launch(
            h.view(),
            positions.view(),
            slot_mapping.view(),
            cu_seqlens_q.view(),
            seqused_k.view(),
            block_table.view(),
            max_seqlen_q,
            max_seqlen_k,
            kv_cache,
            device,
        );

        // ── Fallback: original kernel-per-op path ──
        #[cfg(not(feature = "ferrite"))]
        let h = self.forward_layers_legacy(
            h, positions, slot_mapping,
            cu_seqlens_q, seqused_k, block_table,
            max_seqlen_q, max_seqlen_k,
            kv_cache, device,
        );

        let h = kernels::rms_norm(h.view(), self.model.norm.weight, device);
        device.cublas.gemm(h.view(), self.lm_head.weight)
    }
}
```

### What the megakernel launcher looks like

```rust
/// Generated at compile time by ferrite-fuse, loaded at model init
pub struct FerriteLayers {
    cubin: &'static [u8],            // Embedded at compile time
    module: CUmodule,                // Loaded once at init
    function: CUfunction,            // Kernel entry point
    smem_size: usize,                // Dynamic shared memory requirement
    grid_config: GridConfig,         // Block/grid dims per batch size
}

impl FerriteLayers {
    /// Load the compiled megakernel (called once during model init)
    pub fn load(device: &GpuDevice) -> Self {
        let cubin = include_bytes!(concat!(env!("OUT_DIR"), "/llama_layers.cubin"));
        let module = cudarc::driver::module_load(cubin);
        let function = cudarc::driver::module_get_function(module, "llama_fused_layer");
        // ...
    }

    /// Single kernel launch replaces the entire layer loop
    pub unsafe fn launch(
        &self,
        hidden_states: TensorView<'_>,    // Existing tensor type
        positions: TensorView<'_>,         // Existing tensor type
        slot_mapping: TensorView<'_>,      // Existing tensor type
        cu_seqlens_q: TensorView<'_>,
        seqused_k: TensorView<'_>,
        block_table: TensorView<'_>,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,            // Existing KV cache
        device: &mut GpuDevice,            // Existing device
    ) -> OwnedTensor {
        // Allocate output via existing CachingAllocator
        let output = device.caching.alloc(
            hidden_states.shape(),
            hidden_states.dtype(),
        );

        // One launch. One kernel. All layers.
        cuLaunchKernel(
            self.function,
            self.grid_config.grid_x, self.grid_config.grid_y, 1,
            self.grid_config.block_x, self.grid_config.block_y, 1,
            self.smem_size as u32,
            device.compute_stream,       // Uses existing stream
            &[
                hidden_states.as_ptr() as *mut c_void,
                self.weights_ptr as *mut c_void,  // All layer weights contiguous
                positions.as_ptr() as *mut c_void,
                kv_cache.k_ptr() as *mut c_void,
                kv_cache.v_ptr() as *mut c_void,
                slot_mapping.as_ptr() as *mut c_void,
                block_table.as_ptr() as *mut c_void,
                output.as_mut_ptr() as *mut c_void,
                &max_seqlen_q as *const _ as *mut c_void,
                &max_seqlen_k as *const _ as *mut c_void,
            ],
        );

        output
    }
}
```

### What stays unchanged

| Component | Changes? | Reason |
|-----------|----------|--------|
| `GpuTensor` / `TensorView<'a>` | No | Ferrite inputs/outputs are the same tensors |
| `OwnedTensor` / `CachingAllocator` | No | Ferrite allocates output via existing allocator |
| `GpuDevice` / streams / cuBLAS | No | Ferrite launches on existing compute stream |
| `KvCachePool` / paged KV | No | Megakernel reads/writes KV cache in-place |
| `CudaWorker` / model dispatch | No | Calls same `model.forward()` method |
| Scheduler / `vllm-core` | No | Doesn't know about kernel internals |
| Serving layer / `vllm-serve` | No | Doesn't know about kernel internals |
| Weight loading / `vllm-model` | No | Same weights, same format |
| CUDA Graphs / `graph.rs` | No | Megakernel is itself a single launch — graphs less needed |

### What changes

| Component | Change | Effort |
|-----------|--------|--------|
| `model/llama.rs` | `#[cfg(feature = "ferrite")]` alternate forward | Small |
| `model/qwen2.rs`, etc. | Same pattern per model | Small per model |
| `build.rs` | Conditionally compile ferrite cubins | Medium |
| `Cargo.toml` | Add `ferrite` feature flag + dep | Small |
| New: `ferrite/` workspace | The compiler itself | **The whole project** |

### Incremental migration path

Ferrite is **not all-or-nothing**. Each fusion level is independently shippable:

```
Level 0: No Ferrite (current state)
  11 kernels per layer, 880 launches for Llama-70B

Level 1: Fuse pointwise ops (easiest)
  Replace silu_and_mul+residual_add with fused kernels
  ~9 kernels per layer, ~720 launches
  Estimated win: 5-10% (eliminates small-tensor HBM trips)

Level 2: Fuse norm+GEMM (medium)
  RMSNorm output stays in registers, feeds directly into GEMM
  ~7 kernels per layer, ~560 launches
  Estimated win: 10-15% (eliminates hidden_dim-sized HBM trips)

Level 3: Fuse MLP block (significant)
  norm → gate_up GEMM → SiLU → down GEMM → residual as one kernel
  ~4 kernels per layer (attention is still separate)
  Estimated win: 15-25% (eliminates all MLP intermediate HBM)

Level 4: Fuse attention block (hard)
  norm → QKV GEMM → rotary → flash_attn → output GEMM → residual
  ~2 kernels per layer (attention megakernel + MLP megakernel)
  Estimated win: 25-40%

Level 5: Full layer megakernel
  Entire transformer layer in one persistent kernel
  1 kernel per layer
  Estimated win: 30-50% (matches "No Bubbles" results)

Level 6: Full model megakernel
  All layers pipelined in one persistent kernel
  1 kernel for entire forward pass
  Estimated win: 50%+ (matches Mirage MPK results)
```

Each level can be benchmarked against the previous, merged independently, and
rolled back if issues arise. The `#[cfg(feature = "ferrite")]` flag makes this
safe — the existing `.cu` kernel path is always available as fallback.

### Build system integration

```rust
// vllm-cuda/build.rs — extended for Ferrite
fn main() {
    // Existing: compile .cu files via cudaforge
    cudaforge::build("vllm_kernels", &cu_files);
    cudaforge::build("marlin_kernels", &marlin_files);
    cudaforge::build("cutlass_scaled_mm", &cutlass_files);
    cudaforge::build("vllm_flash_attn", &flash_attn_files);

    // NEW: compile Ferrite megakernels (only if feature enabled)
    #[cfg(feature = "ferrite")]
    {
        // ferrite-compile generates cubins at build time
        let cubins = ferrite_compile::generate_cubins(&FerriteBuildConfig {
            models: &["llama", "qwen2", "deepseek"],
            arch: detect_gpu_arch(),      // sm90, sm100
            dtypes: &[DType::BF16, DType::FP8],
            max_batch: 64,
        });

        for (name, cubin) in cubins {
            let out_path = PathBuf::from(env::var("OUT_DIR").unwrap())
                .join(format!("{name}.cubin"));
            std::fs::write(&out_path, cubin).unwrap();
        }
    }
}
```

### Compatibility with existing features

**Tensor parallelism (TP):** The megakernel's GEMM operations produce partial
results per GPU. The AllReduce points remain at the same logical positions (after
attention output projection, after MLP down projection). Ferrite emits the
AllReduce as a "split point" — the megakernel writes to global memory at these
points, NCCL AllReduce runs, then the next megakernel segment reads the result.

**Pipeline parallelism (PP):** Each PP stage runs its own megakernel covering its
assigned layers. The `ForwardOutput::Intermediate` path works unchanged — it's
just a tensor handoff between stages.

**CUDA Graphs:** Less needed — the megakernel is already a single launch, so graph
capture overhead is already eliminated. But graphs can still wrap the embedding +
megakernel + lm_head sequence if desired.

**Quantization:** The megakernel supports the same quantization schemes as existing
kernels — Marlin (INT4), FP8, BNB 4-bit. Ferrite's Layer 2 ops include quantized
GEMM variants that match the existing `MarlinLinear` and `cutlass_scaled_mm`
performance.

**KV Cache:** The megakernel reads/writes the paged KV cache in exactly the same
format as FlashAttention-2. The `block_table`, `slot_mapping`, `cu_seqlens_q/k`
inputs are passed through unchanged. No migration of existing KV cache state.

---

## Competitive Analysis

### vs CUTLASS (individual kernel performance)

For a single GEMM: **parity at best**. The MMA instruction is fixed-function
hardware. CUTLASS has it fully optimized. We match, not beat.

For fused operations: **Ferrite wins**. CUTLASS cannot fuse across operation
boundaries. A fused RMSNorm+GEMM+SwiGLU eliminates 2 HBM round-trips that CUTLASS
cannot avoid.

### vs Triton

| | Triton | Ferrite |
|---|---|---|
| Frontend | Python | Rust proc macros |
| Compilation | JIT (runtime) | AOT (compile time) |
| Warmup | Seconds | Zero |
| Fusion scope | Single kernel | Megakernel (full layer) |
| Warp scheduling | Heuristic (autoWS) | SAT solver (optimal) |
| Target | General GPU compute | Transformer inference |
| Ecosystem | Massive (PyTorch) | vllm-rs |

Triton's individual kernel performance on Hopper is 70-90% of CUTLASS. On
Blackwell with autoWS, it approaches parity. Ferrite's individual kernels would
be similar (same LLVM backend). The advantage is in **fusion scope** and **zero
warmup**.

### vs torch.compile

torch.compile = dynamo (graph capture) + inductor (fusion) + Triton (codegen).

Ferrite replaces all three with compile-time equivalents:
- No graph capture needed (model structure is explicit Rust code)
- Fusion is a compile-time proc macro (not runtime heuristics)
- Codegen produces cubins (not Triton Python requiring JIT)

torch.compile fuses pointwise chains and GEMM epilogues. It does NOT fuse across
GEMM boundaries. Ferrite fuses entire transformer blocks.

### vs Megakernel research (Mirage MPK, FlashFormer, No Bubbles)

| | Mirage MPK | FlashFormer | No Bubbles | Ferrite |
|---|---|---|---|---|
| Implementation | Python/C++ | C++ | C++ | Rust |
| Compilation | Runtime | Manual | Manual | Compile time |
| Model support | Generic | Transformer | Llama only | Transformer family |
| Reusable infra | No | No | No | Yes (layered) |
| Scheduling | Automatic | Manual | Manual | SAT solver |
| Open source | Yes | Paper only | Blog post | Planned |

Ferrite's advantage: a **reusable, layered compiler** vs one-off research
prototypes. The same infrastructure that fuses a Llama attention block also fuses
Mistral, DeepSeek, Qwen, etc.

### vs TensorRT-LLM

TensorRT-LLM does GEMM+SwiGLU fusion, norm+residual+AllReduce fusion, and other
targeted fusions. It does NOT do full-layer megakernels. It's proprietary (NVIDIA
only, closed-source core).

Ferrite targets the same fusion scope as the megakernel research papers, which is
strictly broader than TensorRT-LLM's current fusions.

---

## Risk Assessment

### Technical risks

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| LLVM NVPTX codegen quality insufficient | High | Low | Triton proves it works. Phase 0 validates. |
| inkwell doesn't expose needed LLVM features | Medium | Low | Mature crate; fallback to llvm-sys (raw FFI). |
| Register pressure in megakernels | High | Medium | SAT solver models register budget. Split if exceeded. |
| Shared memory limits prevent full-layer fusion | Medium | Medium | Graceful degradation to partial fusion. |
| ptxas dependency | Low | Certain | Universal — every GPU toolchain needs it. |
| New GPU arch requires significant rework | Medium | Certain | Layered design isolates arch code to Layer 0-1. |
| Inline PTX asm from Rust is fragile | Medium | Medium | Comprehensive test suite against reference PTX. |

### Organizational risks

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Hiring Rust+GPU+compiler engineers | **Very high** | **High** | ~50-100 people worldwide have this intersection. Consider training GPU engineers in Rust or Rust engineers in GPU. |
| Scope creep (supporting every model/arch) | High | High | Strict scoping: Llama-class transformers on Hopper/Blackwell first. |
| NVIDIA changes PTX ISA significantly | Medium | Low | PTX is backwards-compatible by design. |
| Megakernel research proves impractical at scale | High | Low | Multiple independent groups show it works. |
| CubeCL/Burn reaches our goals first | Medium | Low | Different approach (JIT vs AOT) with different tradeoffs. |

### The "honest assessment" risk

Building a GPU compiler is a multi-year, multi-person effort. Triton took 6+ years
with backing from OpenAI, Meta, NVIDIA, AMD, and Microsoft. We are scoping to ~20%
of Triton's problem space (inference only, transformers only, NVIDIA only), but even
that 20% is a serious engineering undertaking.

The most likely failure mode is not "it doesn't work" but "it takes 3x longer than
estimated and we ship something useful but less ambitious than the full vision."

---

## Team and Timeline

### No-expenses-spared version

| Phase | Duration | People | Deliverable |
|-------|----------|--------|-------------|
| **Phase 0** | 3 months | 2 senior GPU engineers | Proof of concept: inkwell → GEMM → benchmark vs cuBLAS |
| **Phase 1** | 6 months | 3-4 engineers | Layer 0 + Layer 1. One GEMM matching CUTLASS. |
| **Phase 2** | 6 months | 3-4 engineers (parallel) | Layer 2. FlashAttention, all dtypes, quantized GEMM. |
| **Phase 3** | 12 months | 2-3 compiler engineers | Layer 3. Fusion engine + SAT scheduler. First megakernel. |
| **Phase 4** | 6 months | 2 engineers | Layer 4. HuggingFace → binary. Llama, Mistral, DeepSeek. |
| **Phase 5** | Ongoing | Full team | Blackwell, new models, optimization, production hardening. |

**Total to first megakernel**: ~18 months
**Total to production model compiler**: ~24 months
**Team size**: 6-8 engineers

### Pragmatic version

If the full vision is too ambitious, a reduced scope that still delivers value:

| Phase | Duration | People | Deliverable |
|-------|----------|--------|-------------|
| **Phase 0** | 3 months | 2 engineers | Proof of concept (same as above) |
| **Phase 1** | 6 months | 2-3 engineers | Layer 0 + individual kernels (GEMM, attention) via inkwell |
| **Phase 2** | 6 months | 2-3 engineers | Targeted fusions: norm+linear, GEMM+activation, attention block |

This delivers torch.compile-level fusion (but at compile time) without attempting
full megakernels. Still valuable, still novel for Rust.

---

## Phase 0: Proof of Concept

The critical experiment that validates (or kills) the entire plan.

### Goal

Emit a single GEMM kernel (f16, 128x256x64 tiles, wgmma, 3-stage pipeline) via
inkwell → LLVM IR → PTX → ptxas → cubin, launch from Rust, benchmark vs cuBLAS.

**Success criterion**: ≥85% of cuBLAS throughput on H100 for M=N=K=4096 f16 GEMM.

### Steps

1. **Set up inkwell with NVPTX target**
   - Build LLVM 21 with NVPTX backend enabled
   - Create Rust project with inkwell dependency
   - Verify we can emit LLVM IR and compile to PTX

2. **Emit a trivial kernel**
   - Vector add: load two arrays, add, store result
   - Compile to PTX, load via cudarc, verify correctness
   - This validates the full pipeline without GPU complexity

3. **Emit a naive GEMM**
   - Simple tiled GEMM without tensor cores
   - Shared memory tiling, no software pipelining
   - Benchmark: expect ~20-30% of cuBLAS

4. **Add wgmma via inline PTX asm in LLVM IR**
   - Emit `InlineAsm` nodes in LLVM IR for wgmma instructions
   - Add TMA loads for A and B operands
   - Add mbarrier synchronization
   - Benchmark: expect ~60-80% of cuBLAS

5. **Add software pipelining**
   - 3-stage async pipeline with double-buffered shared memory
   - TMA prefetch for next iteration
   - Benchmark: target ≥85% of cuBLAS

6. **Evaluate**
   - If ≥85%: proceed to Phase 1
   - If 70-85%: investigate gaps (register spills? instruction scheduling?)
   - If <70%: the LLVM NVPTX codegen path may not be viable for peak perf

### What Phase 0 proves

- inkwell can target NVPTX and produce correct PTX
- Inline PTX asm for wgmma works from LLVM IR emitted by Rust
- LLVM's optimization passes handle the non-asm code well
- The cudarc launch path works for custom cubins
- We have a realistic performance ceiling for the LLVM path

---

## Sources

### LLVM and NVPTX
- [LLVM NVPTX Backend User Guide](https://llvm.org/docs/NVPTXUsage.html)
- [NVVM Dialect - MLIR](https://mlir.llvm.org/docs/Dialects/NVVMDialect/)
- [NVVM IR Specification 13.2](https://docs.nvidia.com/cuda/nvvm-ir-spec/)
- [PR #120523 - wgmma.fence intrinsics](https://github.com/llvm/llvm-project/pull/120523)
- [PR #122344 - TMA Bulk Copy intrinsics](https://github.com/llvm/llvm-project/pull/122344)
- [PR #116854 - TMA bulk tensor reduction](https://github.com/llvm/llvm-project/pull/116854)
- [PR #123398 - PTX 8.6 / sm100a support](https://github.com/llvm/llvm-project/pull/123398)
- [Bringing Blackwell GPU support to LLVM/MLIR](https://llvm.org/devmtg/2025-04/slides/technical_talk/ozen_blackwell.pdf)

### Rust GPU ecosystem
- [rustc nvptx64-nvidia-cuda docs](https://doc.rust-lang.org/rustc/platform-support/nvptx64-nvidia-cuda.html)
- [Compiler team issue #965](https://github.com/rust-lang/compiler-team/issues/965)
- [core::arch::nvptx tracking issue #111199](https://github.com/rust-lang/rust/issues/111199)
- [Rust-GPU/Rust-CUDA](https://github.com/Rust-GPU/Rust-CUDA)
- [Rust-GPU/rust-gpu](https://github.com/Rust-GPU/rust-gpu)
- [CubeCL](https://github.com/tracel-ai/cubecl)
- [Burn SOTA matmul](https://burn.dev/blog/sota-multiplatform-matmul/)
- [cudarc](https://github.com/coreylowman/cudarc)
- [inkwell](https://github.com/TheDan64/inkwell)
- [melior (MLIR bindings)](https://github.com/mlir-rs/melior)
- [pliron (native Rust IR)](https://github.com/vaivaswatha/pliron)

### Triton
- [Triton GitHub](https://github.com/triton-lang/triton)
- [Triton Kernel Compilation Stages](https://pytorch.org/blog/triton-kernel-compilation-stages/)
- [Warp Specialization in Triton](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/)
- [OpenAI Triton on Blackwell](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/)
- [Twill: Optimal Software Pipelining](https://arxiv.org/abs/2512.18134)

### Megakernels and fusion
- [Mirage Persistent Kernel](https://arxiv.org/abs/2512.22219)
- [Mirage GitHub](https://github.com/mirage-project/mirage)
- [vLLM RFC: Integrate MPK](https://github.com/vllm-project/vllm/issues/22201)
- [FlashFormer](https://arxiv.org/html/2505.22758v1)
- [Look Ma, No Bubbles!](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- [Deep Kernel Fusion](https://arxiv.org/html/2602.11808v1)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)
- [FlashInfer](https://arxiv.org/abs/2501.01005)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)
- [CUTLASS in your CUDA/Triton kernels](https://maknee.github.io/blog/2025/Maybe-Consider-Putting-Cutlass-In-Your-CUDA-Kernels/)
