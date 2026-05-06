# Ferrite Metal Architecture

**Status**: Phase 5 — design finalized (worker pool + specialized pipelines), implementation pending.

> **Supersedes** `FERRITE_METAL_ICB_INHERIT_BUFFERS.md` (proposed `inheritBuffers=true` slot↔encoder-binding mapping; doesn't scale past Metal's 31-binding limit) and the "Phase 5.1–5.5 complete" claim at the top of `FERRITE_METAL_PHASE5_PLAN.md` (which describes work that hasn't actually landed). The Phase 5.6+ steps in PHASE5_PLAN.md are still valid.

## Overview

This document describes the architecture for porting ferrite's compile-time DSL → kernel compilation approach from CUDA to Metal for Apple Silicon GPUs.

The shared frontend (parse → classify → shape-infer → CFG → FUF → solver → scheduler → coloring → static `Instruction<W>[]` tape) is **unchanged**. ferrite-metal contributes exactly two backend-specific pieces:

1. Metal kernel `Implementation`s that participate in the shared claim system, registered alongside CUDA impls in `starter_library()`.
2. A Metal **worker pool** that lowers the `Instruction<W>` tape into ICBs and executes them per forward.

## Key Design Decisions

### 1. Backend-Agnostic Solver + Explicit Lowering to a Metal Tape

**Decision**: The solver emits backend-agnostic `Instruction<W>` lists. ferrite-metal **lowers** that tape — explicitly, via a `From<&[Instruction<W>]> for LoweredMetalTape` translation — into a buffer-pointer-free Metal tape that the worker pool consumes.

**Architecture**:
```
DSL → Solver (backend-agnostic) → static Instruction<W>[]
                                         ↓
              ┌──────────────────────────┴──────────────────────────┐
              ↓                                                     ↓
         CUDA: tile_table runtime interpreter             Metal: lower → LoweredMetalTape
              walks tape, eval() per op                            (one per (model, bucket))
              (per-forward kernel launches)                        ↓
                                                              MetalWorkerPool
                                                              ├── checkout() → MetalWorker
                                                              │     ├── arena (Vec<Buffer>)
                                                              │     └── ICBs (one per bucket)
                                                              ├── per forward:
                                                              │     bind inputs → executeCommandsInBuffer
                                                              └── checkin()
```

**Key points**:
- `LoweredMetalTape` is **buffer-pointer-free** — it carries pipeline ids, dispatch shape, scalar constants, and slot ids only. Computed once per (model, bucket); shared across all workers via `Arc`.
- `MetalWorker` instantiates the lowered tape against its own arena: walks the lowered tape, looks up `arena[slot_id]` per command, records the resulting buffer pointers into a fresh `IndirectCommandBuffer`. Done once at worker creation.
- Per-forward path is just `bind_inputs → encoder.executeCommandsInBuffer(icb)` — no allocator churn, no argbuf rewriting, no per-forward CPU work beyond encoder commit.
- Solver picks `Implementation`s based on `target_compatible()`. Metal impls register in the shared `starter_library()` and emit the same `Instruction<W>` opcodes the CUDA impls do — no Metal-private opcodes, no Metal-specific solver/scheduler/codegen.

**Why explicit lowering instead of `Instruction::record_to_icb()` methods**: keeps Metal-specific concerns (pipeline ids, dispatch shapes, slot resolution) out of the shared `Instruction<W>` enum. Lowering is a normal Rust `From` impl, testable in isolation, and one mental hop away from the CUDA `Instruction::eval()` analog.

### 2. Worker Pool: Growable, Capped, Each Worker Fully Baked

**Decision**: Metal forwards run on a `MetalWorkerPool`. The pool starts at size 1 and grows on demand up to a known maximum (`max_workers = floor((device_total - weights - misc) / per_worker_arena)`). Each worker holds a private arena and a fully-recorded ICB per bucket.

**Why this shape:**
- **Realistic concurrency cost.** Single-tenant single-batch hits size 1 and pays for one arena. Concurrency demand (spec-decode draft+verify, prefill/decode overlap, multi-stream serving) grows the pool only when actually needed.
- **Predictable memory ceiling.** ferrite is a compiler for model instances — the macro emits `NUM_TILES` post-coloring, so per-worker arena size is known at compile time per (arch, bucket). Combined with the model's weight footprint and total device memory, `max_workers` is computable, not probabilistic.
- **No kernel rewrite.** Phase 4 kernels stay as `[[buffer(0)]] input, [[buffer(1)]] output` style. Slot↔buffer mapping is resolved at worker init by walking the lowered tape against the worker's arena.
- **Per-forward path is hot.** No allocator interaction, no argument buffer mutation, no ICB re-recording. Encoder commit + GPU execute. Recording cost (~250 commands × tens of µs) amortizes over thousands of forwards.

**What's already bought from the macro:** `colored_slot_map()` (in `ferrite-forward-macro/src/interpreter_codegen.rs`) does linear-scan register allocation on the FUF — non-overlapping live ranges share slot ids, partitioned by shape. The arena size is the colored count, not the raw tile count. Worker memory is already at the floor.

**Pool sketch:**
```rust
struct MetalWorkerPool {
    model_meta:  Arc<MetalModelMeta>,            // weights + pipelines, shared
    lowered:     Arc<[LoweredBucketTape]>,       // one per bucket, computed once
    workers:     Mutex<Vec<MetalWorker>>,
    max_workers: usize,
    semaphore:   Semaphore,                       // bounds in-flight forwards
}

struct MetalWorker {
    arena: Vec<metal::Buffer>,                    // sized for max bucket
    icbs:  [IndirectCommandBuffer; NUM_BUCKETS],  // one per bucket, baked
}
```

`pool.checkout()` blocks if all workers are in use and `len() == max_workers`; otherwise it grows the pool by one (allocates arena, records ICBs against it). `forward()` picks a bucket, checks out a worker, binds inputs, executes the bucket's ICB, checks in.

### 3. Specialized Pipelines via Function Constants

**Decision**: Each `(model variant, bucket, kernel)` gets its own `MTLComputePipelineState` with bucket and model constants baked in via `MTLFunctionConstantValues`.

**What gets baked:**
- **Bucket-derived**: `M = num_tokens` (the workload point).
- **Model-derived (layer-independent CanonicalParams)**: `hidden_size` / `Q_SIZE`, `INTERMEDIATE_SIZE`, `NUM_Q_HEADS`, `NUM_KV_HEADS`, `HEAD_DIM`, `VOCAB_SIZE`, `MAX_SEQ_LEN`, `ROPE_THETA`, RMSNorm `eps` values, GQA group ratio, etc.
- **Derived**: anything compile-time-foldable from the above (`hidden_size / num_heads`, `2 * intermediate_size`, group masks, strides).

**What stays runtime (still bound as buffers):**
- Per-layer weight tensor pointers (different memory per layer).
- KV cache pointers, position arrays, RoPE cos/sin tables, embeddings.
- Anything tensor-shaped — function constants are scalar-only.

**What this buys:**
- Loop unrolling on `for (uint m = 0; m < M; m++)`-style reductions when M is small.
- Bounds-check elimination on `if (tid < num_tokens)`.
- Strength reduction on stride math and divisor folding.
- One fewer buffer binding — kill the `constants` buffer entirely. Frees a binding index (real budget under Metal's 31-bind ICB limit) and saves a load per dispatch.
- Tile-shape specialization for hand-rolled kernels (RMSNorm, fused, attention). MPS-backed GEMM is opaque and gets none of this — fine.

**Realistic gain shape:** ~10–30% on hand-rolled kernels at small buckets (`bucket = 1`, `8`) where instruction count and register pressure dominate. Sub-5% on large buckets where bandwidth is the limiter. Decode-path (`bucket = 1`) gets the biggest relative win — exactly where latency matters most.

**Cost:** ~14 hand-rolled kernels × 5 buckets × (per model variant) specialized pipelines. Single ms each to compile, all cacheable across runs via Metal's pipeline cache. One-time at first model load.

**Implementation note:** kernels that consume runtime constants today (e.g. `rmsnorm.metal` reading `eps` and `hidden_size` from a `[[buffer(3)]]` constants buffer) get rewritten to declare those as `[[function_constant(N)]]`. Pipeline state object construction switches to the `MTLFunctionConstantValues`-bearing variant. The recording loop in `MetalWorker` looks up the right specialized pipeline per (bucket, kernel) instead of a generic one.

### 4. Reuse 100% of Frontend Pipeline

**Backend-Agnostic Components** (unchanged for Metal):
- Parse → Classify → Shape Inference → CFG → FUF (tile graph)
- Solver: DP-based implementation selection
- Scheduler: Wave-based scheduling
- **Coloring**: `colored_slot_map()` already does linear-scan register allocation
- Codegen: Static `Instruction<W>[]` slice emission

**Backend-Specific Components** (Metal-only):
- `Implementation` trait impls in `ferrite-metal-impl-lib` (kernels participate in claims for `Backend::Metal`)
- Lowering pass: `Instruction<W>` → `LoweredMetalTape` (computed once per (model, bucket))
- `MetalWorkerPool` + `MetalWorker` in `ferrite-forward/src/interpreter/metal.rs`
- ICB recording helpers in `ferrite-metal-kernels/src/instruction_executor/`
- Metal kernel wrappers + specialized-pipeline construction (RMSNorm, GEMM, Attention, etc.)
- MSL shaders in `ferrite-metal-kernels/shaders/`, declaring function constants for layer-independent params

### 5. Cost Model Strategy

**Approach**: Hybrid analytical + measured costs
- **Analytical models**: Memory bandwidth-based estimates for memory-bound ops (RMSNorm, elementwise)
- **Measured costs**: Microbenchmark sweeps for compute-bound ops (GEMM, attention)
- **Fallback**: Conservative estimates when no data available

**Cost Table Format**:
```rust
BTreeMap<String, Vec<CostEntry>>
// kernel_name -> [(M, N, K, cost_us), ...]
```

### 6. Fusion Strategy

**Leverage Existing Ferrite Fusion**:
- Ferrite already has extensive fusion via `Implementation` trait
- Examples: `FusedQkvRopeCacheImpl`, `FusedGateUpSiluImpl`
- MLX's fusion is simpler (generic elementwise chains + GEMM epilogues)
- Port existing fusion patterns to Metal rather than adopting MLX's approach

**Metal-Specific Optimizations**:
- Threadgroup memory for reductions
- Simdgroup operations for warp-level primitives
- Metal Performance Shaders (MPS) integration for standard ops

## Crate Structure

```
vllm-rs/crates/
├── ferrite-forward/             # Frontend + Metal worker pool
│   └── src/
│       └── interpreter/
│           ├── metal/
│           │   ├── mod.rs          # Re-exports
│           │   ├── lowering.rs     # Instruction<W> → LoweredMetalTape
│           │   ├── pool.rs         # MetalWorkerPool (growable, capped)
│           │   ├── worker.rs       # MetalWorker (arena + per-bucket ICBs)
│           │   ├── pipelines.rs    # SpecializedPipelineCache (function constants)
│           │   └── lowered.rs      # LoweredMetalTape, LoweredCommand types
│           └── mod.rs              # cfg(feature = "metal") gate
├── ferrite-metal-targets/       # Device profiles (M1/M2/M3/M4)
│   ├── src/lib.rs              # MetalTargetProfile, cost tables
│   └── profiles/cost_*.csv     # Measured kernel costs
├── ferrite-metal-kernels/       # Metal runtime & kernel execution
│   ├── src/
│   │   ├── lib.rs              # MetalDevice, MetalStream, MetalAllocator
│   │   ├── device.rs           # Device detection & capabilities
│   │   ├── stream.rs           # Command buffer management
│   │   ├── allocator.rs        # Buffer pooling
│   │   ├── instruction_executor/  # ICB recording primitives (Phase 4.6)
│   │   │   ├── mod.rs          # RecordingContext, dispatch helpers
│   │   │   ├── rmsnorm.rs      # record_rmsnorm()
│   │   │   ├── gemm.rs         # record_gemm()
│   │   │   ├── attention.rs    # record_attention()
│   │   │   └── fused.rs        # record_fused_*()
│   │   ├── rmsnorm.rs          # RMSNorm kernel wrapper (function-constant aware)
│   │   ├── gemm.rs             # MPS GEMM wrapper
│   │   ├── attention.rs        # Attention kernel wrapper
│   │   └── fused_kernels.rs    # Fused kernel wrappers
│   └── shaders/
│       ├── rmsnorm.metal       # RMSNorm MSL — function constants for hidden_size, eps
│       ├── attention.metal     # Attention MSL — function constants for head_dim, num_heads
│       └── fused_*.metal       # Fused kernel MSL — bucket M + model constants baked
├── ferrite-metal-impl-lib/      # Metal Implementation trait impls
│   └── src/
│       └── metal/
│           ├── rmsnorm.rs      # MetalRmsNormImpl
│           ├── gemm.rs         # MetalGemmImpl
│           ├── attention.rs    # MetalAttentionImpl (4 variants)
│           ├── activation.rs   # MetalActivationImpl (5 variants)
│           ├── awq.rs          # MetalAwqImpl (2 variants)
│           └── fused_kernels.rs # MetalFused*Impl
└── ferrite-forward-macro/       # Proc-macro (unchanged for Metal)
    └── src/
        ├── solver.rs           # Backend-agnostic solver
        ├── schedule.rs         # Backend-agnostic scheduler
        ├── interpreter_codegen.rs # colored_slot_map() — already minimizes arena
        └── impl_lib.rs         # Implementation trait + starter_library()
```

**Key architectural points**:
1. **No Metal-specific solver/scheduler/codegen.** Frontend pipeline (including post-coloring slot allocation) is shared.
2. **No `record_to_icb()` on `Instruction<W>`.** Metal-specific concerns live in `interpreter/metal/lowering.rs` as a normal `From` impl.
3. **One worker = one arena + one ICB per bucket.** All ICBs reference that worker's arena buffers; never shared across workers.
4. **Specialized pipelines per `(model variant, bucket, kernel)`.** Function constants bake `M`, `hidden_size`, `num_heads`, `head_dim`, `eps`, `rope_theta`, etc., into the SSA — no `constants` buffer at runtime.
5. **Solver integration unchanged**: `starter_library()` registers Metal impls alongside CUDA impls, gated by `Backend::Metal` in `target_compatible()`.

## Implementation Phases

### Phase 1: Foundation ✅ COMPLETE
- [x] Create `ferrite-metal-targets` with M1/M2/M3/M4 profiles
- [x] Create `ferrite-metal-kernels` with Metal-rs bindings
- [x] Create `ferrite-metal-impl-lib` with `MetalImplementation` trait
- [x] Implement `MetalRmsNormF16Impl` as first example
- [x] Add Metal shaders directory with `rmsnorm.metal`
- [x] Verify basic Metal device detection and buffer allocation

### Phase 2: Solver Integration ✅ COMPLETE
- [x] Extend `TargetProfile` to support Metal backend via `Backend` enum
- [x] Metal implementations registered in `starter_library()` (MetalRmsNormImpl)
- [x] Cost lookup works for Metal targets via `MetalTargetProfile::cost_us_for()`
- [x] Microbenchmark harness created (`ferrite-metal-cost-sweep`)
- [x] Initial cost tables populated for M1 Max from real hardware measurements

### Phase 3: Runtime & ICB Infrastructure ✅ COMPLETE
- [x] MetalStream for command buffer management
- [x] MetalAllocator for buffer pooling
- [x] Integration tests passing (21 tests total)
- [x] ICB architecture decided (multi-launch, not single persistent kernel)
- [x] Basic ICB recording/execution infrastructure in place

**Note**: No Metal-specific codegen needed - solver already emits backend-agnostic `Instruction<W>` lists

### Phase 4: Kernel Library & Implementation Registration ✅ COMPLETE
**Status**: All 18 critical path operations implemented

**Completed**:
- [x] 4.1: Attention kernels (basic, paged, multi-head, GQA, optimized)
- [x] 4.2: Fused kernels (Add+RMSNorm, Gate-Up-SiLU-Mul, GEMM via MPS)
- [x] 4.3: Activation functions (SiLU, GELU, FatReLU)
- [x] 4.4: Quantization (AWQ dequantization + GEMM integration)
- [x] 4.5: Implementation trait impls (18 critical path operations)
  - Modular structure in `src/metal/` directory
  - All registered in `starter_library()`
  - 78 tests passing
- [x] 4.6: ICB recording infrastructure
  - `RecordingContext` with ICB management
  - `record_compute_dispatch()` for recording commands
  - `inheritPipelineState=true` for Apple Silicon compatibility
  - Module structure: `rmsnorm`, `gemm`, `attention`, `fused`

### Phase 5: Worker Pool + Lowering + Specialized Pipelines 🔄 PLANNED

**Goal**: End-to-end Metal forward via the worker-pool architecture above. Existing `metal.rs` skeleton is replaced (it bakes the wrong assumptions — `record_to_icb()` methods on `Instruction<W>`, no lowering step, no pool).

**Sub-phases:**

- **5.A: Lowering pass** — `From<&[Instruction<W>]> for LoweredMetalTape`. Buffer-pointer-free; carries pipeline ids (specialized-pipeline keys), dispatch shape, scalar constants, slot ids. Pure CPU code, unit-testable without a Metal device. Covers TinyLlama-1.1B critical path first (~13 `Instruction<W>` variants); extends as more models come online.

- **5.B: Function-constant pipeline specialization.** Rewrite hand-rolled MSL shaders (`rmsnorm.metal`, `attention.metal`, `fused_*.metal`) to declare layer-independent params as `[[function_constant(N)]]`. Add `SpecializedPipelineCache` keyed on `(model_variant_id, bucket_id, kernel_name)`. Construct `MTLComputePipelineState`s with `MTLFunctionConstantValues` populated from `CanonicalParams` + bucket M.

- **5.C: `MetalWorker` build.** Allocates arena per colored slot, walks the lowered tape, records one ICB per bucket against the arena using the specialized pipelines. ICB has actual buffer pointers baked in (Option B); reused across all forwards on this worker.

- **5.D: `MetalWorkerPool`.** Growable, capped, semaphore-bounded checkout/checkin. RAII guard. `max_workers` derived from device memory minus weights at construction time.

- **5.E: `forward()`.** Pick bucket from `num_tokens`, checkout worker, bind input/position buffers, encoder.executeCommandsInBuffer(this bucket's ICB), checkin.

- **5.F: Macro emission.** `#[forward]` macro emits Metal codegen alongside CUDA: `MetalWorkerPool::for_<model>()` constructor, model meta with weight load helpers. Lowering happens at constructor time.

- **5.G: Correctness wiring.** Hook into the existing `cpu_golden::*` per-op references and the `vllm-e2e` golden framework — same path CUDA uses. No bespoke Metal-only test scaffolding.

### Phase 5.6: Validate against TinyLlama-1.1B golden 🔜 NEXT
- [ ] Add `metal` feature to `ferrite-forward` Cargo.toml (currently absent — interpreter module is dead code without it).
- [ ] Generate / reuse TinyLlama-1.1B golden via existing `vllm-e2e` framework.
- [ ] Pass golden under `--features metal` on M1+ hardware.
- [ ] Profile latency per token; surface any function-constant wins or losses vs. a non-specialized control build.

### Phase 6: Production Readiness 🔜 PLANNED
- [ ] Add comprehensive error handling
- [ ] Document all public APIs
- [ ] Create migration guide for CUDA users
- [ ] Benchmark full model inference (Llama, Qwen, etc.)
- [ ] Performance comparison vs MLX baseline

## Metal-Specific Considerations

### Threadgroup Memory
- 32KB per threadgroup on all Apple Silicon
- Use for reductions (RMSNorm, softmax)
- Careful bank conflict avoidance

### Simdgroup Operations
- 32-wide SIMD on M1/M2/M3/M4
- Use `simd_sum()`, `simd_max()` for warp-level reductions
- Faster than threadgroup barriers for small reductions

### Memory Bandwidth
- M1: 68.25 GB/s
- M2: 100 GB/s
- M3: 100 GB/s
- M4: 120 GB/s
- Memory-bound ops (RMSNorm, layernorm) scale linearly with bandwidth

### Compute Throughput
- M1: 2.6 TFLOPS FP16
- M2: 3.6 TFLOPS FP16
- M3: 4.0 TFLOPS FP16
- M4: 4.5 TFLOPS FP16
- Compute-bound ops (GEMM, attention) scale with TFLOPS

## Comparison: Ferrite vs MLX

| Aspect | Ferrite | MLX |
|--------|---------|-----|
| **Fusion** | Extensive via `Implementation` trait | Generic elementwise + GEMM epilogues |
| **Scheduling** | Wave-based with ICB | Sequential dispatch |
| **Cost Model** | DP solver with measured costs | Heuristic-based |
| **Kernel Selection** | Compile-time specialization | Runtime dispatch |
| **Advantage** | Global optimization, zero runtime overhead | Simpler implementation |

**Strategy**: Leverage ferrite's superior fusion and cost modeling, adopt MLX's tuned Metal kernel implementations.

## Future Architectural Improvements

### Metal Tensor Abstraction (RAII Pattern)

**Current State**: Raw `metal::Buffer` pointers with manual lifecycle management
- Direct buffer creation and passing to kernels
- No automatic cleanup or ownership tracking
- Prone to memory management bugs (e.g., double-free in MPS integration)

**Proposed**: Adopt CUDA backend's tensor abstraction pattern
```rust
// Lightweight view (no ownership)
pub struct MetalTensor {
    buffer: Arc<Buffer>,      // RAII via Arc
    shape: [u32; MAX_DIMS],
    dtype: DType,
}

pub struct MetalTensorView<'a> {
    buffer: &'a Buffer,
    shape: [u32; MAX_DIMS],
    dtype: DType,
}

// Owned tensor with arena/pool management
pub struct OwnedMetalTensor {
    tensor: MetalTensor,
    arena: Arc<MetalArena>,  // Ties lifetime to arena
}
```

**Benefits**:
- Automatic buffer lifecycle management via `Arc<Buffer>`
- Type-safe shape/dtype tracking at compile time
- Consistent API with CUDA backend (`ferrite-cuda-core/src/tensor.rs`)
- Prevents manual `release` calls and double-free bugs
- Enables zero-copy views and slicing operations

**Implementation Priority**: Post-Phase 5 (after runtime interpreter complete)

**Related Issues**:
- Fixed in Phase 4.4: Manual MPS object release causing SIGSEGV
- Fixed in Phase 4.4: Wrong MPSDataType enum values (268435472 vs 16)

## Next Steps

1. **Phase 5.A — Lowering pass.** `LoweredMetalTape` types + `From<&[Instruction<W>]>` impl, covering TinyLlama-1.1B critical path. Pure Rust; testable without a Metal device.
2. **Phase 5.B — Function-constant specialization.** Rewrite hand-rolled MSL shaders to declare layer-independent params as `function_constant`s; build `SpecializedPipelineCache`.
3. **Phase 5.C/D/E — Worker + Pool + forward.** End-to-end Metal forward via the pool.
4. **Phase 5.F — Macro emission.** `#[forward]` macro emits Metal `MetalWorkerPool::for_<model>()` constructors alongside CUDA `try_load()`.
5. **Phase 5.G/5.6 — Validate.** Hook into existing `cpu_golden::*` per-op checks and the `vllm-e2e` golden framework. Pass TinyLlama-1.1B golden under `--features metal`.
6. **Profile.** Quantify the function-constant specialization win at small buckets vs. an unspecialized control.

## References

- [Ferrite CUDA Implementation](../vllm-rs/crates/ferrite-forward-macro/src/codegen.rs)
- [MLX Fusion Logic](../mlx-source-for-bob/mlx/compile.cpp)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Apple Silicon GPU Architecture](https://developer.apple.com/documentation/metal/gpu_features)