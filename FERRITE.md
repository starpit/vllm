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
│  Proc macro: dataflow graph → fused Rust GPU kernel source  │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Operation Library (libops)                        │
│  GEMM, FlashAttention, RMSNorm, Rotary, SwiGLU, etc.       │
│  Each op: Rust GPU function using Layer 0/1 primitives      │
│  GEMM patterns ported from CubeK (tile configs, scheduling) │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Tile Engine (libtile)                             │
│  Shared memory management, swizzle, tiling abstractions     │
│  Software pipelining, double buffering helpers              │
├─────────────────────────────────────────────────────────────┤
│  Layer 0: PTX Intrinsic Library (libptx)                    │
│  Safe Rust wrappers around asm!() inline PTX                │
│  mma.sync, ldmatrix, cp.async, mbarrier, etc.              │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  rust-cuda (rustc_codegen_nvvm) → libnvvm → PTX → ptxas    │
│  NVIDIA's own optimizer — nvcc-quality register allocation  │
└─────────────────────────────────────────────────────────────┘
```

### Layer 0: PTX Builder (`ferrite-ptx`) — IMPLEMENTED

**Status: Working. Validated at Triton-matching performance.**

Not inline asm wrappers — a PTX string builder. Each method emits one PTX
instruction as a formatted string. Register allocation tracks usage per class
(pred, b32, b64, f32). `finalize()` wraps with header, register declarations,
and kernel entry point.

```rust
// Actual working API (not conceptual)
let mut ptx = PtxBuilder::new(GemmConfig::default_64x64());
let d = ptx.regs.alloc_b32();
let a = ptx.regs.alloc_b32();
ptx.ldmatrix_x4_trans([d0, d1, d2, d3], addr, None);
ptx.mma_m16n8k16(acc, a_frag, b_frag, acc);
ptx.cp_async_cg(dst, 0, src, 0, size_pred);
let kernel_ptx: String = ptx.finalize("my_kernel", &params);
```

**What exists** (in `vllm-rs/crates/ferrite-poc/src/ptx_builder/`):
- `mod.rs` — PtxBuilder struct, RegAllocator, 50+ instruction emitters
- `config.rs` — GemmConfig with derived constants
- `smem.rs` — Shared memory layout, swizzle computation
- `gemm.rs` — GEMM kernel builder (55 TFLOPS, matches Triton)
- `silu.rs` — SiLU phase emitter + standalone kernel (238 GB/s)
- `rmsnorm.rs` — RMSNorm phase emitter + standalone kernel
- `fused.rs` — First fusion attempt (correct but slow — needs Layer 1)

### Layer 1: Tile Engine — THE CRITICAL LAYER

This is the CuTe equivalent. **This is what makes fusion automatic.** Without it,
every fusion is a hand-written kernel. With it, fusions are atom configurations.

The tile engine defines a **generic K-loop pipeline** parameterized by pluggable
atoms. A GEMM kernel is not hand-written code — it is a configuration of the
pipeline. A fused RmsNorm→GEMM→SiLU kernel is a DIFFERENT configuration of the
SAME pipeline. The pipeline handles scheduling, double-buffering, prefetch, and
interleaving automatically.

#### The Pipeline

```rust
/// The mainloop: a software-pipelined K-loop with configurable stages.
/// This is the ONLY K-loop in the entire system. All GEMM-based operations
/// are expressed as configurations of this pipeline.
struct MainloopPipeline<const STAGES: u32> {
    /// Emit the complete K-loop: prologue → loop body → epilogue.
    /// The loop body interleaves:
    ///   1. Prefetch next iteration's fragments from smem (ldmatrix)
    ///   2. Transform previous iteration's A fragments (TransformAtom)
    ///   3. Execute MMA on current iteration (MMAAtom)
    ///   4. Issue async copies for next stage (CopyAtom)
    ///   5. Pipeline synchronization (wait_group, barrier)
    fn emit_mainloop(
        &self,
        ptx: &mut PtxBuilder,
        copy_a: &dyn CopyAtom,
        copy_b: &dyn CopyAtom,
        transform_a: &dyn TransformAtom,
        mma: &dyn MMAAtom,
    ) -> AccumulatorMap;
}
```

#### The Atoms

```rust
/// How tiles move from global memory to shared memory.
/// SM89: cp.async.cg (hardware DMA, 16 bytes/thread)
/// SM90+: TMA descriptors (hardware tensor memory accelerator)
trait CopyAtom {
    fn emit_async_copy(&self, ptx: &mut PtxBuilder, smem_dst: Reg, glob_src: Reg, pred: Reg);
    fn emit_commit(&self, ptx: &mut PtxBuilder);
    fn async_groups_per_tile(&self) -> u32;
}

/// How A fragments are transformed between ldmatrix and MMA.
/// This is WHERE FUSION HAPPENS for pre-GEMM operations.
///
/// Identity: no transform (standalone GEMM)
/// RmsNorm: 2x mul.rn.f16x2 (fused normalization)
/// Dequantize: scale packed int4/int8 to f16 (quantized inference)
///
/// CRITICAL: The transform operates on PACKED f16x2 registers.
/// It must use f16x2 arithmetic (mul.rn.f16x2, fma.rn.f16x2).
/// NEVER unpack to f32 and repack — that's 12 instructions instead of 2.
/// CUTLASS proved this: their layernorm transform is 2x fma.rn.f16x2.
trait TransformAtom {
    /// One-time setup before the K-loop (e.g., load norm factors into registers).
    fn emit_prologue(&self, ptx: &mut PtxBuilder);
    /// Per-K-iteration setup (e.g., load gamma from smem for this K chunk).
    fn emit_k_setup(&self, ptx: &mut PtxBuilder, ki: u32);
    /// Transform 4 b32 registers (8 packed f16) in-place. Called per ldmatrix.
    fn emit_transform(&self, ptx: &mut PtxBuilder, frag: &mut [Reg; 4], ki: u32, rm: u32);
}

/// How tensor cores consume fragments.
/// SM89: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
/// SM90+: wgmma.mma_async (warp group MMA)
trait MMAAtom {
    fn emit_mma(&self, ptx: &mut PtxBuilder, a: [Reg; 4], b: [Reg; 2], acc: &mut [Reg; 4]);
}

/// How accumulators are post-processed before store.
/// This is WHERE FUSION HAPPENS for post-GEMM operations.
///
/// Identity: store raw f32 accumulators
/// SiLu: x * sigmoid(x) on accumulators (6 ALU ops per element, zero memory)
/// Gelu: approximate GELU on accumulators
/// ResidualAdd: load residual from global, add to accumulators
/// Quantize: convert f32 accumulators to int8/fp8 before store
trait EpilogueAtom {
    fn emit_epilogue(&self, ptx: &mut PtxBuilder, acc: &mut AccumulatorMap);
}
```

#### What a GEMM Looks Like

A standalone GEMM is NOT hand-written. It is:

```rust
let pipeline = MainloopPipeline::<2> { config: GemmConfig::default_64x64() };
let acc = pipeline.emit_mainloop(
    &mut ptx,
    &CpAsyncCopy,       // A tiles: hardware DMA
    &CpAsyncCopy,       // B tiles: hardware DMA
    &IdentityTransform, // no transform
    &MMA_m16n8k16,      // tensor core op
);
emit_store_f32(&mut ptx, &acc);  // store accumulators
```

#### What a Fused Kernel Looks Like

A fused RmsNorm→GEMM→SiLU is the SAME pipeline with DIFFERENT atoms:

```rust
let pipeline = MainloopPipeline::<2> { config: GemmConfig::default_64x64() };

// Phase 1: compute norm factors (separate from pipeline)
let norm_ctx = emit_rmsnorm_reduction(&mut ptx, input_ptr, hidden_size);

// Phase 2: GEMM with transform
let mut acc = pipeline.emit_mainloop(
    &mut ptx,
    &CpAsyncCopy,                        // A tiles: SAME cp.async, loads RAW input
    &CpAsyncCopy,                        // B tiles: SAME
    &RmsNormTransform::new(norm_ctx),    // 2x mul.rn.f16x2 between ldmatrix and MMA
    &MMA_m16n8k16,                       // SAME
);

// Phase 3: SiLU epilogue (register ALU, zero memory)
SiLuEpilogue.emit_epilogue(&mut ptx, &mut acc);
emit_store_f32(&mut ptx, &acc);
```

**The pipeline is identical. Only the atoms change.** If the pipeline is correct
and fast for standalone GEMM (55 TFLOPS), swapping IdentityTransform for
RmsNormTransform adds only the cost of the transform itself (2 instructions per
f16x2 register) — NOT the cost of restructuring the K-loop, adding barriers,
changing the smem layout, or any other architectural change.

#### Reference: CUTLASS CollectiveMainloop

This is exactly what CUTLASS does. Their `MmaLayernormMainloopFusionMultistage`
(SM80) is a mainloop pipeline where:
- `TransformA` = `LayernormScaleBiasTransform` (2x `fma.rn.f16x2` per element)
- Gamma/beta loaded via `WarpIteratorGammaBeta` (from smem, prefetched per K-iter)
- Var/mean loaded ONCE in prologue, kept in REGISTERS (not smem) for entire K-loop
- The transform is interleaved with prefetch: while loading iteration N+1's
  fragments, transform iteration N's fragments, MMA iteration N-1's fragments

We replicate this architecture exactly in PTX.

#### Everything is AOT. Everything is proc macro.

**There is no runtime code generation.** The pipeline, atoms, and their composition
all execute at **compile time** inside a proc macro. The output is a PTX string
constant embedded in the binary. At runtime, the only cost is one `cuModuleLoadData`
call (JIT by the CUDA driver) on first use.

The proc macro IS the compiler:

```rust
// User writes this:
#[ferrite::fuse(arch = "sm_89")]
fn mlp_block(x: &Tensor, w_gate: &Tensor, w_down: &Tensor, norm_w: &Tensor) {
    let n = rmsnorm(x, norm_w);
    let g = gemm(n, w_gate);
    let h = silu(g);
    gemm(h, w_down)
}

// At COMPILE TIME, the proc macro:
// 1. Parses the function body into an op graph
// 2. Recognizes: rmsnorm → gemm = TransformAtom::RmsNorm on A input
// 3. Recognizes: gemm → silu = EpilogueAtom::SiLu
// 4. Selects atoms: CpAsync, CpAsync, RmsNorm, MMA16816, SiLu
// 5. Calls MainloopPipeline::emit_mainloop() with those atoms
// 6. PtxBuilder generates the PTX string
// 7. Embeds it as: const PTX: &str = "...";
// 8. Generates a wrapper function that JITs and launches

// At RUNTIME, the generated function:
// 1. OnceLock: first call loads PTX via cuModuleLoadData (one-time cost)
// 2. Every call: cuLaunchKernel with the pre-compiled module
// Zero warmup beyond the first call. No Python. No JIT compilation in the hot path.
```

This is the key advantage over torch.compile (JIT, seconds of warmup, recompiles
on shape change) and Triton (JIT, requires Python). Ferrite kernels are compiled
into the Rust binary at build time. The serving process starts with kernels ready.

**The proc macro depends on `ferrite-ptx` (Layer 0) at build time.** Since
`ferrite-ptx` is a pure Rust library with no native dependencies, it runs inside
`rustc`'s process during compilation. No CUDA toolkit needed at build time — only
at runtime (for `cuModuleLoadData` and `ptxas` JIT).

---

### Phase 0 Lessons Learned: What NOT to Do

These mistakes were made during Phase 0 and must never be repeated:

**Mistake 1: Trying to make fusion work by hacking the K-loop**

Five different approaches were tried to fuse RmsNorm into the GEMM:
1. `NormalizedLoader`: replaced cp.async with ld.global+normalize+st.shared (0.48x)
2. Split-phase loads: issue ld.global early, process after MMA (0.53x)
3. cp.async-then-transform: cp.async to temp smem, transform smem→smem (0.47x)
4. Register transform with f32 unpack/repack: 12 ALU per register (0.65x)
5. Register transform with mul.rn.f16x2: 2 ALU per register (0.65x)

All were one-off hacks that tried to "make this specific fusion work" without
building the underlying pipeline abstraction. Each one required restructuring the
K-loop, adding barriers, changing smem layouts — exactly the work that the pipeline
abstraction is supposed to handle automatically.

**The root cause:** We built Layer 0 (PtxBuilder) and Layer 3 (proc macro) but
SKIPPED Layer 1 (tile engine / pipeline). Without the pipeline abstraction, every
fusion required hand-writing a new K-loop variant. With the pipeline, fusions are
just atom swaps.

**Mistake 2: Using LLVM IR for kernel codegen**

The LLVM IR path (inkwell + individual inline asm blocks) produced PTX with
identical instructions, registers, and occupancy as hand-written PTX — but ran
1.77x slower. The root cause: each inline asm block creates a scheduling barrier
that prevents ptxas from interleaving loads with compute. This is unfixable without
rewriting LLVM's NVPTX backend.

**Lesson:** Drop LLVM for kernel codegen entirely. Emit PTX strings directly.
ptxas is the real backend.

**Mistake 3: Proposing "keep operations as separate kernels" as a fallback**

When the fused kernel was slow, the instinct was to fall back to emitting 3
separate kernel launches. This defeats the entire purpose of Ferrite. The answer
is never "give up on fusion" — it's "build the right abstraction so fusion works."

**Mistake 4: Building standalone op kernels before the pipeline**

Building standalone RMSNorm and SiLU kernels was useful for benchmarking but
created the illusion that fusion = "concatenate standalone kernels." It led to
the `fused.rs` approach of bolting phases together without a unifying pipeline.
The standalone kernels are BENCHMARKS, not BUILDING BLOCKS. The building blocks
are ATOMS that plug into the pipeline.

**Mistake 5: Using f32 unpack/repack instead of f16x2 packed arithmetic**

The first register transform attempt unpacked each f16 pair to two f32 values,
multiplied, and repacked — 12 instructions per b32 register. CUTLASS uses
`fma.rn.f16x2` — 2 instructions per b32 register. This was discovered by
reading the CUTLASS source code, which should have been done FIRST. Always
read the reference implementation before designing.

**Mistake 6: Not reading CUTLASS and Megakernels early enough**

Both CUTLASS (layernorm mainloop fusion) and Megakernels (Hazy "No Bubbles")
have solved the fusion problem. Their solutions are public. We should have
studied them in detail BEFORE attempting our own fusion, not AFTER failing
five times. The pipeline-with-atoms architecture was right there in CUTLASS
the whole time.

**Mistake 7: Using 64×64 tiles when CUTLASS uses 128×128**

Five iterations of fusion at 64×64 tiles (0.48x–0.69x of unfused) before
discovering CUTLASS uses 128×128 tiles with 162 registers. The larger tile
amortizes the transform cost: 64 MMA per K-iter (128×128) vs 16 MMA (64×64)
means the 2× `mul.rn.f16x2` per fragment register is <3% of compute instead
of ~10%. Once we matched CUTLASS's tile size, the fusion worked (1.1x faster
than unfused at batch=1024).

**Lesson:** Always check the reference implementation's EXACT configuration
before designing. Tile size, warp layout, register budget, pipeline stages —
copy all of it, not just the algorithm.

**Mistake 8: Comparing against own unfused baseline instead of production**

We spent iterations trying to beat our own 3-kernel unfused baseline. The real
comparison is against PyTorch (the production baseline). Even at 0.88× our
unfused, the fused kernel was already 1.4-2.1× faster than PyTorch. The
obsession with beating our own optimized unfused baseline obscured the real win.

---

### Phase 0 Results Summary (March 21, 2026)

**Validated against real-world baselines (L4 GPU, SM89):**

| MLP (batch=1024, hidden=inter=out=4096) | Time | TFLOPS | vs torch.compile |
|------------------------------------------|------|--------|-----------------|
| PyTorch eager (norm+GEMM+SiLU+GEMM) | 1729 μs | 40 | — |
| PyTorch torch.compile | 1679 μs | 41 | 1.0× |
| Ferrite MLP (proc macro, DAG-composed) | 1229 μs | 50 | **1.37× faster** |
| Ferrite MLP (hand-written) | 1370 μs | 50 | **1.23× faster** |

| MLP (batch=4096) | Time | TFLOPS | vs torch.compile |
|-------------------|------|--------|-----------------|
| PyTorch torch.compile | 6324 μs | 44 | 1.0× |
| Ferrite MLP (proc macro) | 5221 μs | 53 | **1.21× faster** |

| Fused RmsNorm→GEMM→SiLU (batch=1024) | Time | TFLOPS | vs eager |
|---------------------------------------|------|--------|---------|
| PyTorch eager | 1151 μs | 30 | 1.0× |
| Ferrite fused (hand-written 128×128) | 703 μs | 49 | **1.6× faster** |

| Kernel (batch=4096) | Time | TFLOPS | vs PyTorch |
|---------------------|------|--------|-----------|
| PyTorch unfused (RMSNorm+GEMM+SiLU) | 7015 μs | 19.6 | 1.0× |
| Ferrite fused RmsNorm→GEMM→SiLU | 2926 μs | 47.0 | **2.4× faster** |
| Ferrite MLP block (norm→GEMM→SiLU→GEMM) | 5221 μs | 52.6 | **~2× faster** |

**Flash Attention (L4 GPU, d=64):**

| Config | Ferrite | PyTorch FA2 | Ratio |
|--------|---------|------------|-------|
| B=4 H=32 seq=512 | **71.0 TFLOPS** | 68.2 TFLOPS | **1.04× faster** |
| B=1 H=32 seq=1024 | **71.5 TFLOPS** | 68.4 TFLOPS | **1.05× faster** |
| B=1 H=1 seq=512 | 4.3 TFLOPS | 2.1 TFLOPS | **2.05× faster** |

**Standalone building blocks:**

| Kernel | Performance | vs Reference |
|--------|------------|-------------|
| GEMM 64×64 (hand-written PTX) | 55 TFLOPS | matches Triton |
| GEMM 128×128 (hand-written PTX) | 48 TFLOPS | matches Triton |
| Flash Attention (hand-written PTX) | 71 TFLOPS | **exceeds FA2 (68 TF)** |
| SiLU standalone | 238 GB/s | matches PyTorch (232 GB/s) |
| GELU standalone | 237 GB/s | matches PyTorch (232 GB/s) |
| RMSNorm standalone | 152 GB/s (batch=32) | 5.9× faster than Triton at batch=1 |
| CUTLASS fused GEMM→LayerNorm→GEMM | 74.6 TFLOPS | reference implementation |

**Architecture validated:**
- Direct PTX generation beats LLVM IR path by 1.77×
- Pipeline + atoms abstraction: zero overhead for standalone GEMM (55 TFLOPS)
- 128×128 tiles + CUTLASS composition pattern: fusion overhead <3% at scale
- `fma.rn.f16x2` / `mul.rn.f16x2` in-place transforms: zero extra registers
- MLP block (GEMM→GEMM chain) via intermediate in global/L2: ~50 TFLOPS
- Flash attention: 4 warps, BLOCK_M=128, double-buffered K/V, B128 swizzle
- Proc macro generates PTX at compile time, embeds as const string
- DAG-based composition: adding GELU/ResidualAdd required zero strategy changes
- 155 unit tests (19 macros + 111 ptx + 25 flash attn)

**Supported operations (all composable via DAG edge classification):**

| Op | OpClass | As Prologue (TransformAtom) | As Epilogue (EpilogueAtom) | Standalone |
|----|---------|---------------------------|--------------------------|-----------|
| RmsNorm | Elementwise | Yes (2× mul.rn.f16x2) | — | Yes |
| Gemm | Matmul | — | — | Yes (128×128) |
| FlashAttention | Attention | — | — | Yes (71 TFLOPS) |
| SiLU | Elementwise | — | Yes (6 ALU/elem) | Yes (238 GB/s) |
| GELU | Elementwise | — | Yes (7 ALU/elem) | Yes (237 GB/s) |
| ResidualAdd | Elementwise | — | Future | Yes |

---

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

### Phase 0 Results (March 20, 2026)

Phase 0 tested three codegen paths. The results fundamentally changed the
architecture.

#### Path 1: inkwell/LLVM IR (tested, insufficient)

Emit LLVM IR via inkwell → NVPTX backend → PTX. Validated on L40S and L4:

- ✅ Vector add, tiled GEMM, tensor core MMA via inline PTX asm — all work
- ✅ B128 shared memory swizzle, cp.async, ldmatrix, ldmatrix.trans — all work
- ✅ `-nvptx-short-ptr` makes addrspace(3) pointers 32-bit (fixes register bloat)
- ❌ **Peak GEMM: 31.6 TFLOPS on L4 (57% of Triton's 55.2 TFLOPS)**

**Root cause**: Not register allocation. Not occupancy. Not instruction count.
The LLVM IR path produces **identical PTX instructions** (same ldmatrix, mma,
cp.async counts, same 72 registers, 0 spills) but 1.77x slower. The bottleneck
is **SASS instruction scheduling**: each inline asm block in LLVM IR creates a
scheduling barrier that prevents ptxas from interleaving loads with compute.

#### Path 2: Hand-written PTX (tested, matches Triton)

Emit PTX strings directly from Rust, bypassing LLVM entirely:

- ✅ **55.2 TFLOPS on L4 — identical to Triton** (80 regs, 0 spills)
- ✅ Standalone SiLU: 238 GB/s (matches Triton's 239 GB/s)
- ✅ Standalone RMSNorm: 5.9x faster than Triton at batch=1

When all instructions are in one PTX scheduling region, ptxas generates
optimal SASS. The problem was never LLVM's register allocation or instruction
selection — it was the scheduling barriers between inline asm blocks.

#### Path 3: PtxBuilder (tested, matches hand-written)

Rust code that programmatically generates PTX strings:

- ✅ **55.0 TFLOPS — identical to hand-written PTX**
- ✅ Parameterizable by tile size, warp layout, data types
- ✅ Composable phases (GEMM K-loop, SiLU, RMSNorm as separate emitters)
- ❌ Naive phase composition (inlining RMSNorm into GEMM K-loop) is 3x slower
  than unfused — the composition mechanism needs the tile engine (Layer 1)

#### Conclusions

1. **Drop LLVM/libnvvm/rust-cuda for kernel codegen.** Any path that emits
   individual inline asm blocks (whether via LLVM IR, NVVM IR, or rustc) will
   hit the same scheduling barrier. The performance comes from ptxas, and ptxas
   needs to see all instructions in one scheduling region.

2. **Direct PTX generation is the right codegen path.** Rust code generates PTX
   strings. ptxas compiles to SASS. This is simpler than any LLVM-based path
   and produces identical performance to Triton.

3. **Phase composition requires the tile engine.** Concatenating phase code
   doesn't work — the tile engine must analyze data flow between operations
   and choose the right tiling/buffering strategy.

### Revised Codegen Path: PtxBuilder

```
Ferrite proc macro (compile time, pure Rust)
    |
    +-- Dataflow graph analysis (Layer 3)
    +-- Tile planning, smem layout, register budget (Layer 1)
    +-- Warp role assignment
    |
    v
PtxBuilder (Rust library, Layer 0)
    |
    +-- Emits PTX instructions as formatted strings
    +-- Register allocator tracks usage per class
    +-- Instruction emitters: mma, ldmatrix, cp.async, ALU, etc.
    +-- Phase emitters: GEMM K-loop, RMSNorm, SiLU, attention
    |
    v
PTX string (one kernel, one scheduling region)
    |
    v
ptxas (NVIDIA proprietary) --> SASS --> CUBIN
    |
    +-- Optimal instruction scheduling
    +-- Register allocation within ptxas's budget
    +-- Bank conflict resolution
```

This is architecturally simpler than the rust-cuda path and empirically
produces identical performance. The complexity lives in the tile engine
(Layer 1) and fusion engine (Layer 3), not in the codegen backend.

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

## Risk Assessment (Updated post-Phase 0)

### Retired risks

| Risk | Status | Notes |
|------|--------|-------|
| LLVM NVPTX codegen quality insufficient | **CONFIRMED** | 41.5% of peak. Pivoted to libnvvm via rust-cuda. |
| inkwell doesn't expose needed features | **Moot** | Worked fine, but the backend itself was the bottleneck. |

### Active technical risks

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| rust-cuda maturity (nightly, early dev) | High | Medium | Active project, recent reboot. Contribute fixes upstream. |
| libnvvm version lag vs CUDA toolkit | Medium | Medium | rust-cuda tracks CUDA 13.0+. |
| Register pressure in fused megakernels | High | Medium | libnvvm handles this much better than upstream LLVM. CubeK proves 128-reg kernels work through libnvvm. |
| Shared memory limits prevent full-layer fusion | Medium | Medium | Graceful degradation to partial fusion. |
| ptxas dependency | Low | Certain | Universal. |
| New GPU arch requires rework | Medium | Certain | Layer 0 isolates arch-specific asm!(). |
| rust-cuda `asm!()` for MMA is unproven | Medium | Medium | Phase 1 step 1 validates this immediately. |

### Organizational risks

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Hiring Rust+GPU engineers | **Very high** | **High** | rust-cuda lowers the GPU side — kernel code is Rust, not LLVM IR. |
| Scope creep | High | High | Strict: Llama-class on Ada/Hopper first. |
| rust-cuda project dies or diverges | Medium | Low | Fork if necessary; libnvvm is the actual dependency. |

### Lessons from Phase 0

1. **Don't fight the toolchain.** LLVM's NVPTX backend is not nvcc. Years of
   custom passes (Triton) or using nvcc directly (CubeK) are the proven paths.
2. **Study existing implementations first.** CubeK's matmul architecture (tile
   configs, swizzle, partition scheduling) is the reference. Port it, don't reinvent.
3. **The GEMM is solved.** CubeK/Burn achieve cuBLAS parity. Our value is fusion.
4. **Register pressure is the #1 GPU performance constraint.** Every design
   decision must account for register budget.

---

## Team and Timeline (Revised)

### Updated plan

| Phase | Duration | People | Deliverable |
|-------|----------|--------|-------------|
| **Phase 0** | ~~3 months~~ 1 day | 1 | ✅ DONE. inkwell path validated, <70% threshold triggered. |
| **Phase 1** | 2-4 weeks | 1-2 | rust-cuda GEMM matching cuBLAS + first fused kernel |
| **Phase 2** | 2-3 months | 2-3 | Layer 0-2: MMA wrappers, tile engine, operation library |
| **Phase 3** | 3-6 months | 2-3 | Layer 3: Fusion proc macro. First auto-fused megakernel. |
| **Phase 4** | 3-6 months | 2 | Layer 4: Model compiler. Llama, Mistral, DeepSeek. |
| **Phase 5** | Ongoing | Full team | Hopper/Blackwell, new models, production hardening. |

**Total to first fused kernel**: ~1 month (Phase 1)
**Total to auto-fused megakernels**: ~6-9 months (Phase 3)
**Total to production model compiler**: ~12-18 months (Phase 4)

Timeline is significantly shorter than the original plan because:
1. Phase 0 is done (1 day instead of 3 months)
2. rust-cuda eliminates the need to build a codegen backend
3. CubeK provides the GEMM reference implementation to port
4. The GEMM is a port, not original research

---

## Phase 0: Proof of Concept — COMPLETE

### Results (2026-03-19, NVIDIA L40S sm_89)

| Step | Result | TFLOPS | % peak |
|------|--------|--------|--------|
| Vector add (inkwell → PTX) | ✅ Correct, 3069 GB/s | — | — |
| Tiled GEMM (shared memory, f32) | ✅ Correct | 4.41 | 20% of f32 |
| MMA GEMM (inline PTX asm, tensor cores) | ✅ Correct | 5.35 | 3.0% |
| Multi-warp + register tiling (64×64) | ✅ Correct | 52.72 | 29.1% |
| + cp.async both tiles | ✅ | 60.12 | 33.2% |
| + B128 swizzle | ✅ | **75.06** | **41.5%** |
| 128×128 CubeK-style | ✅ Correct but 255 regs | 45.99 | 25.4% |

**Success criterion was ≥85%. Result: 41.5%. Outcome: <70% threshold triggered.**

### What Phase 0 proved

- ✅ inkwell → LLVM IR → NVPTX → PTX pipeline works end-to-end
- ✅ Inline PTX asm for mma.sync works from Rust-generated LLVM IR
- ✅ B128 swizzle, cp.async, shared memory all work correctly
- ✅ Tensor cores can be driven from Rust through LLVM
- ❌ LLVM NVPTX codegen quality is insufficient for peak GEMM performance
- ❌ 64-bit shared memory pointers cause register bloat at larger tiles
- ❌ `-nvptx-short-ptr` (Triton's fix) helps addressing but doesn't solve instruction count

### Key discovery: the LLVM NVPTX limitation

The fundamental issue: LLVM's NVPTX backend uses 64-bit pointer arithmetic for
shared memory and generates excessive instructions for fragment loading. At 64×64
tiles (80 registers), occupancy is good. At 128×128 tiles (255 registers, needed
for adequate compute-to-memory ratio), occupancy drops to 1 block/SM.

CubeK achieves cuBLAS parity through NVRTC (which uses nvcc's backend). Triton
achieves ~70-90% of CUTLASS through years of custom LLVM optimization passes.
Neither path was available to us in the Phase 0 timeframe.

### Decision: pivot to rust-cuda backend

The plan's risk assessment correctly predicted this outcome. The revised path uses
rust-cuda (`rustc_codegen_nvvm`) which compiles Rust through libnvvm — NVIDIA's
own optimizer, the same backend as nvcc. This gives nvcc-quality register allocation
and instruction scheduling while keeping everything in Rust.

## Phase 1: rust-cuda Megakernel Prototype

### Goal

Write a fused RMSNorm → GEMM → SiLU kernel in Rust using rust-cuda, compiled
through libnvvm. Benchmark the GEMM alone for parity with cuBLAS, then benchmark
the fused kernel against unfused cuBLAS + separate norm/activation.

### Steps

1. **Set up rust-cuda compilation for vllm-rs**
   - Add rust-cuda as a build dependency
   - Write a standalone GEMM kernel in Rust GPU code with `asm!()` for mma.sync
   - Port CubeK's tile config (128×128, partition 4×4×2, B128 swizzle)
   - Benchmark: target ≥85% of cuBLAS (libnvvm should handle register allocation)

2. **Add MMA wrapper library**
   - Safe Rust wrappers around `asm!()` for mma.sync, ldmatrix, cp.async
   - Swizzle helpers matching CubeK's patterns
   - Shared memory tile abstractions

3. **Write the first fused kernel**
   - RMSNorm → GEMM → SiLU in one `#[kernel]` function
   - Norm output stays in shared memory, feeds directly into GEMM tiles
   - SiLU applied in registers before writing to global memory
   - Benchmark against: cuBLAS GEMM + separate norm kernel + separate SiLU kernel

4. **Wire into vllm-rs**
   - `#[cfg(feature = "ferrite")]` in model/llama.rs
   - Replace one MLP block with the fused kernel
   - End-to-end inference benchmark

5. **Build the proc macro**
   - Analyze Rust function body → dataflow graph
   - Identify fusible operation sequences
   - Generate fused `#[kernel]` Rust GPU code
   - cuda_builder compiles at build time, cubin embedded in binary

### What Phase 1 proves

- rust-cuda/libnvvm achieves cuBLAS-parity GEMM from Rust code
- Cross-operation fusion (norm → GEMM → activation) works in one kernel
- The fused kernel beats unfused cuBLAS + separate kernels
- The proc macro can automatically generate fused kernels

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
- [Rust-CUDA (rustc_codegen_nvvm)](https://github.com/Rust-GPU/Rust-CUDA) — compiles Rust to GPU via libnvvm
- [Rust-CUDA reboot announcement](https://rust-gpu.github.io/blog/2025/01/27/rust-cuda-reboot/)
- [CubeK (cubek-matmul)](https://github.com/tracel-ai/cubek) — CubeCL's matmul kernel library (reference for tile configs)
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
