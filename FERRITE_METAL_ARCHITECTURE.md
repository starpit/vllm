# Ferrite Metal Architecture

**Status**: Initial skeleton implementation (Week 1 of 8-12 week roadmap)

## Overview

This document describes the architecture for porting ferrite's compile-time DSL → kernel compilation approach from CUDA to Metal for Apple Silicon GPUs.

## Key Design Decisions

### 1. Backend-Agnostic Solver + Backend-Specific Execution

**Decision**: Solver emits backend-agnostic `Instruction<W>` lists; Metal execution records these to ICB.

**Architecture**:
```
DSL → Solver (backend-agnostic) → Instruction<W> list
                                         ↓
                    ┌────────────────────┴────────────────────┐
                    ↓                                         ↓
            CUDA: eval() each instruction          Metal: record_to_icb() each instruction
                  (direct kernel launch)                 (ICB pre-recording at init)
                                                          ↓
                                                    execute_icb() at runtime
                                                    (single GPU call per forward)
```

**Key Points**:
- Solver picks `Implementation`s based on `target_compatible()` check
- Metal impls implement `Implementation` trait directly (no wrapper)
- Same `Instruction<W>` enum for both backends
- CUDA: `Instruction::eval()` launches kernels directly
- Metal: `Instruction::record_to_icb()` pre-records to ICB at init time

### 2. Reuse 100% of Frontend Pipeline

**Backend-Agnostic Components** (unchanged for Metal):
- Parse → Classify → Shape Inference → CFG → FUF (tile graph)
- Solver: DP-based implementation selection
- Scheduler: Wave-based scheduling
- Codegen: Static `Instruction<W>` slice emission

**Backend-Specific Components** (Metal-only):
- `Implementation` trait impls in `ferrite-metal-impl-lib`
- ICB recording in `ferrite-metal-kernels/src/instruction_executor.rs`
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
├── ferrite-metal-targets/       # Device profiles (M1/M2/M3/M4)
│   ├── src/lib.rs              # MetalTargetProfile, cost tables
│   └── profiles/cost_*.csv     # Measured kernel costs
├── ferrite-metal-kernels/       # Metal runtime & kernel execution
│   ├── src/
│   │   ├── lib.rs              # MetalDevice, MetalStream, MetalAllocator
│   │   ├── device.rs           # Device detection & capabilities
│   │   ├── stream.rs           # Command buffer management
│   │   ├── allocator.rs        # Buffer pooling
│   │   ├── instruction_executor.rs  # ICB recording for Instruction<W>
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
│       ├── rmsnorm.rs          # MetalRmsNormF16Impl: Implementation
│       ├── gemm.rs             # MetalGemmImpl: Implementation
│       ├── attention.rs        # MetalAttentionImpl: Implementation
│       └── fused_kernels.rs    # MetalFused*Impl: Implementation
└── ferrite-forward-macro/       # Proc-macro (unchanged for Metal)
    └── src/
        ├── solver.rs           # Backend-agnostic solver
        ├── schedule.rs         # Backend-agnostic scheduler
        └── impl_lib.rs         # Implementation trait + starter_library()
```

**Key Architectural Points**:
1. **No Metal-specific codegen**: Solver already emits `Instruction<W>` lists
2. **No wrapper traits**: Metal impls implement `Implementation` directly
3. **ICB execution**: `instruction_executor.rs` records each `Instruction` variant to ICB
4. **Solver integration**: `starter_library()` registers Metal impls alongside CUDA impls

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2) ✅ CURRENT
- [x] Create `ferrite-metal-targets` with M1/M2/M3/M4 profiles
- [x] Create `ferrite-metal-kernels` with Metal-rs bindings
- [x] Create `ferrite-metal-impl-lib` with `MetalImplementation` trait
- [x] Implement `MetalRmsNormF16Impl` as first example
- [x] Add Metal shaders directory with `rmsnorm.metal`
- [ ] Verify basic Metal device detection and buffer allocation

### Phase 2: Solver Integration (Weeks 3-4) ✅ COMPLETE
- [x] Extend `TargetProfile` to support Metal backend via `Backend` enum
- [x] Metal implementations registered in `starter_library()` (MetalRmsNormImpl)
- [x] Cost lookup works for Metal targets via `MetalTargetProfile::cost_us_for()`
- [x] Microbenchmark harness created (`ferrite-metal-cost-sweep`)
- [x] Initial cost tables populated for M1 Max from real hardware measurements

### Phase 3: Runtime & ICB Execution (Weeks 5-6) ✅ COMPLETE
- [x] MetalStream for command buffer management
- [x] MetalAllocator for buffer pooling
- [x] Integration tests passing (21 tests total)
- [x] ICB architecture decided (multi-launch, not single persistent kernel)
- [x] Basic ICB recording/execution infrastructure in place

**Note**: No Metal-specific codegen needed - solver already emits backend-agnostic `Instruction<W>` lists

### Phase 4: Kernel Library & Implementation Registration (Weeks 7-10) 🔄 IN PROGRESS
**Current Status**: ~15% complete (1 of ~50 instruction types)

**Completed**:
- [x] 4.1: Attention kernels (basic, paged, multi-head, GQA, optimized)
- [x] 4.2: Fused kernels (Add+RMSNorm, Gate-Up-SiLU-Mul, GEMM via MPS)
- [x] 4.3: Activation functions (SiLU, GELU, FatReLU)
- [x] 4.4: Quantization (AWQ dequantization + GEMM integration)

**In Progress**:
- [ ] 4.5: Create `Implementation` trait impls for each Metal kernel
  - [x] MetalRmsNormImpl (fp16, bf16) - registered in `starter_library()`
  - [ ] MetalGemmImpl (wraps Phase 4.2.5 MPS GEMM)
  - [ ] MetalFusedAddRmsNormImpl (wraps Phase 4.2.1)
  - [ ] MetalFusedGateUpSiluMulImpl (wraps Phase 4.2.2)
  - [ ] MetalAttentionImpl (wraps Phase 4.1 attention kernels)
  - [ ] ~45 more instruction variants...
- [ ] 4.6: Implement `Instruction<W>::record_to_icb()` for Metal
  - [ ] Create `ferrite-metal-kernels/src/instruction_executor.rs`
  - [ ] One match arm per `Instruction` variant (~50 total)
  - [ ] Record pipeline state, buffer bindings, dispatch size to ICB

**Estimated Remaining**: 2-3 weeks

### Phase 5: End-to-End Integration (Weeks 11-12) 🔜 PLANNED
- [ ] Wire up Metal execution path in vllm-executor
- [ ] Test full forward pass: DSL → Solver → ICB → Metal kernels
- [ ] Verify numerical correctness against CUDA reference
- [ ] Profile and optimize hot paths

### Phase 6: Production Readiness (Weeks 13-14) 🔜 PLANNED
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

**Implementation Priority**: Post-Phase 4 (after core kernel functionality complete)

**Related Issues**:
- Fixed in Phase 4.4: Manual MPS object release causing SIGSEGV
- Fixed in Phase 4.4: Wrong MPSDataType enum values (268435472 vs 16)


## Next Steps

1. **Verify skeleton compiles**: `cd vllm-rs && cargo check -p ferrite-metal-targets -p ferrite-metal-kernels -p ferrite-metal-impl-lib`
2. **Run basic tests**: `cargo test -p ferrite-metal-targets -p ferrite-metal-impl-lib`
3. **Start Phase 2**: Integrate Metal backend into solver
4. **Microbenchmark harness**: Port `ferrite-cost-sweep` to Metal

## References

- [Ferrite CUDA Implementation](../vllm-rs/crates/ferrite-forward-macro/src/codegen.rs)
- [MLX Fusion Logic](../mlx-source-for-bob/mlx/compile.cpp)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Apple Silicon GPU Architecture](https://developer.apple.com/documentation/metal/gpu_features)
