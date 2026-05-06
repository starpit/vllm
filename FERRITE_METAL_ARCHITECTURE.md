# Ferrite Metal Architecture

**Status**: Phase 5 implementation (runtime interpreter with ICB recording)

## Overview

This document describes the architecture for porting ferrite's compile-time DSL → kernel compilation approach from CUDA to Metal for Apple Silicon GPUs.

## Key Design Decisions

### 1. Backend-Agnostic Solver + Backend-Specific Execution

**Decision**: Solver emits backend-agnostic `Instruction<W>` lists; Metal execution records these to ICB at init time.

**Architecture**:
```
DSL → Solver (backend-agnostic) → Instruction<W> list
                                         ↓
                    ┌────────────────────┴────────────────────┐
                    ↓                                         ↓
            CUDA: eval() each instruction          Metal: MetalExecutor walks tape at init
                  (runtime kernel launch)                calls record_*() methods
                                                          ↓
                                                    ICB pre-recorded at init
                                                          ↓
                                                    execute_icb() at runtime
                                                    (single GPU call per forward)
```

**Key Points**:
- Solver picks `Implementation`s based on `target_compatible()` check
- Metal impls implement `Implementation` trait directly (no wrapper)
- Same `Instruction<W>` enum for both backends
- CUDA: `Instruction::eval()` launches kernels at runtime
- Metal: `MetalExecutor` walks tape at init, calls `record_*()` to populate ICB

### 2. Reuse 100% of Frontend Pipeline

**Backend-Agnostic Components** (unchanged for Metal):
- Parse → Classify → Shape Inference → CFG → FUF (tile graph)
- Solver: DP-based implementation selection
- Scheduler: Wave-based scheduling
- Codegen: Static `Instruction<W>` slice emission

**Backend-Specific Components** (Metal-only):
- `Implementation` trait impls in `ferrite-metal-impl-lib`
- ICB recording infrastructure in `ferrite-metal-kernels/src/instruction_executor/`
- `MetalExecutor` runtime interpreter in `ferrite-forward/src/interpreter/metal.rs`
- Metal kernel wrappers (RMSNorm, GEMM, Attention, etc.)
- MSL shaders in `ferrite-metal-kernels/shaders/`

### 3. Cost Model Strategy

**Approach**: Hybrid analytical + measured costs
- **Analytical models**: Memory bandwidth-based estimates for memory-bound ops (RMSNorm, elementwise)
- **Measured costs**: Microbenchmark sweeps for compute-bound ops (GEMM, attention)
- **Fallback**: Conservative estimates when no data available

**Cost Table Format**:
```rust
BTreeMap<String, Vec<CostEntry>>
// kernel_name -> [(M, N, K, cost_us), ...]
```

### 4. Fusion Strategy

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
├── ferrite-forward/             # Frontend + runtime interpreter
│   └── src/
│       └── interpreter/
│           ├── metal.rs        # MetalExecutor (walks tape, calls record_*())
│           └── metal_tests.rs  # MetalExecutor tests
├── ferrite-metal-targets/       # Device profiles (M1/M2/M3/M4)
│   ├── src/lib.rs              # MetalTargetProfile, cost tables
│   └── profiles/cost_*.csv     # Measured kernel costs
├── ferrite-metal-kernels/       # Metal runtime & kernel execution
│   ├── src/
│   │   ├── lib.rs              # MetalDevice, MetalStream, MetalAllocator
│   │   ├── device.rs           # Device detection & capabilities
│   │   ├── stream.rs           # Command buffer management
│   │   ├── allocator.rs        # Buffer pooling
│   │   ├── instruction_executor/  # ICB recording infrastructure
│   │   │   ├── mod.rs          # RecordingContext, dispatch helpers
│   │   │   ├── rmsnorm.rs      # record_rmsnorm()
│   │   │   ├── gemm.rs         # record_gemm()
│   │   │   ├── attention.rs    # record_attention()
│   │   │   └── fused.rs        # record_fused_*()
│   │   ├── rmsnorm.rs          # RMSNorm kernel wrapper
│   │   ├── gemm.rs             # MPS GEMM wrapper
│   │   ├── attention.rs        # Attention kernel wrapper
│   │   └── fused_kernels.rs    # Fused kernel wrappers
│   └── shaders/
│       ├── rmsnorm.metal       # RMSNorm MSL shader
│       ├── attention.metal     # Attention MSL shader
│       └── fused_*.metal       # Fused kernel MSL shaders
├── ferrite-metal-impl-lib/      # Metal Implementation trait impls
│   └── src/
│       ├── lib.rs              # Re-exports
│       ├── metal/              # Modular implementation structure
│       │   ├── mod.rs          # Module organization
│       │   ├── rmsnorm.rs      # MetalRmsNormImpl
│       │   ├── gemm.rs         # MetalGemmImpl
│       │   ├── attention.rs    # MetalAttentionImpl (4 variants)
│       │   ├── activation.rs   # MetalActivationImpl (5 variants)
│       │   ├── awq.rs          # MetalAwqImpl (2 variants)
│       │   └── fused_kernels.rs # MetalFused*Impl
│       └── metal_bridge.rs     # Deprecated compatibility shim
└── ferrite-forward-macro/       # Proc-macro (unchanged for Metal)
    └── src/
        ├── solver.rs           # Backend-agnostic solver
        ├── schedule.rs         # Backend-agnostic scheduler
        └── impl_lib.rs         # Implementation trait + starter_library()
```

**Key Architectural Points**:
1. **No Metal-specific codegen**: Solver already emits `Instruction<W>` lists
2. **No wrapper traits**: Metal impls implement `Implementation` directly
3. **ICB recording**: `instruction_executor/` provides `record_*()` methods
4. **Runtime interpreter**: `MetalExecutor` walks tape at init, calls `record_*()`
5. **Solver integration**: `starter_library()` registers Metal impls alongside CUDA impls

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

### Phase 5: Runtime Interpreter 🔄 IN PROGRESS
**Goal**: Runtime interpreter that walks instruction tape and records to ICB  
**Status**: Skeleton created, recording loop TODO

**Architecture**:
- `MetalExecutor` walks instruction tape at init time
- Calls Phase 4.6's `record_*()` methods to populate ICB
- Executes ICB on each forward pass

**Implementation**:
- [x] 5.1: Create `MetalExecutor` struct (skeleton with instruction matching)
- [x] 5.2: Create `RecordingContext` (from Phase 4.6)
- [ ] 5.3: Implement instruction recording loop
- [ ] 5.4: Implement `forward()` execution (tile table + ICB execution)
- [ ] 5.5: Add MetalExecutor tests

### Phase 5.6: Test with Real Model 🔜 NEXT
- [ ] Complete Phase 5.3-5.5 (recording loop + forward execution)
- [ ] Wire `#[forward]` macro to emit Metal code
- [ ] Implement `MetalWeights::load_safetensors()`
- [ ] Add Metal backend to `vllm-e2e` tests
- [ ] Debug and validate with TinyLlama-1.1B

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

1. **Complete Phase 5.3-5.5**: Implement instruction recording loop in `MetalExecutor`
2. **Test with real model**: Wire into `vllm-e2e` golden test framework
3. **Profile and optimize**: Identify hot paths and optimize Metal kernels
4. **Production readiness**: Error handling, documentation, CI/CD

## References

- [Ferrite CUDA Implementation](../vllm-rs/crates/ferrite-forward-macro/src/codegen.rs)
- [MLX Fusion Logic](../mlx-source-for-bob/mlx/compile.cpp)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Apple Silicon GPU Architecture](https://developer.apple.com/documentation/metal/gpu_features)