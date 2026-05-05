# Metal Implementation Tracker

**Purpose:** Single source of truth for Metal `Implementation` trait adapter progress.  
**Location:** All implementations in `vllm-rs/crates/ferrite-forward-macro/src/metal/`  
**Registration:** All registered in `impl_lib.rs::starter_library()`

## Quick Status Summary

- **Total OpKinds:** 28 (from `classified.rs::OpKind`)
- **Implemented:** 18 (64%)
- **Critical Path Remaining:** 0 ✅ ALL COMPLETE
- **Advanced Features:** 10 (MoE, MLA, TP ops, multimodal)

## Implementation Status by OpKind

### ✅ COMPLETE (14 implementations)

| OpKind | Variants | File | Notes |
|--------|----------|------|-------|
| RmsNorm | fp16, bf16 | `rmsnorm.rs` | Singleton tile claim |
| Gemm | fp16, fp32 | `gemm.rs` | Uses MPS (Metal Performance Shaders) |
| Attention | 4 variants | `attention.rs` | Basic, paged, multi-head, optimized |
| Silu | fp16 | `activation.rs` | Part of activation suite |
| Gelu | fp16 (3 variants) | `activation.rs` | Tanh, Quick, standard |
| Add | fp16, bf16 | `add.rs` | Elementwise addition |
| FusedAddRmsNorm | fp16, bf16 | `fused_kernels.rs` | Multi-tile fusion |
| FusedGateUpSiluMul | fp16, bf16, gelu | `fused_kernels.rs` | SwiGLU + GELU-MLP |
| Awq (dequant) | fp16, bf16 | `awq.rs` | 4-bit dequantization |
| Reshape | metadata-only | `reshape.rs` | View operation (~0.1µs) |
| ScalarMul | fp16, bf16 | `scalar_mul.rs` | Broadcast scalar multiply |
| Embed | fp16, bf16 | `embed.rs` | Lookup table operation |
| RopeAppend | fp16, bf16 | `rope.rs` | ✅ NeoX-style rotary encoding - REGISTERED |
| RopeAppendInterleaved | fp16, bf16 | `rope.rs` | ✅ GPT-J/CommandR style - REGISTERED |
| Mul | fp16, bf16 | `mul.rs` | ✅ Elementwise multiply (gate * up) - REGISTERED |
| BiasAdd | fp16, bf16 | `bias_add.rs` | ✅ Broadcast addition - REGISTERED |
| TanhSoftCap | fp16, bf16 | `softcap.rs` | ✅ Logit capping (Gemma2) - REGISTERED |
| Sub | fp16, bf16 | `sub.rs` | ✅ Elementwise subtraction - REGISTERED |

### ✅ CRITICAL PATH - ALL COMPLETE (2026-05-05)

All critical path operations for basic LLaMA inference are now implemented:
- ✅ Mul - Elementwise multiply (gate * up)
- ✅ BiasAdd - Broadcast addition
- ✅ TanhSoftCap - Logit capping (Gemma2)
- ✅ Sub - Elementwise subtraction (LayerNorm fusion)

**Phase 4.5 COMPLETE** - Ready for Phase 4.6 (ICB Recording)

### 🟡 EXTENDED FEATURES (5 ops)

| OpKind | Priority | Complexity | Notes |
|--------|----------|------------|-------|
| **RopeAppendInterleaved** | P2 | Medium | Cohere CommandR variant |
| **SlidingAttention** | P2 | Medium | Window-masked attention (Gemma2) |
| **Mean** | P2 | Low | For LayerNorm fusion pattern |
| **FatReLU** | P3 | Low | Activation variant (implemented but not registered) |

### 🟣 ADVANCED ARCHITECTURES (11 ops)

| OpKind | Architecture | Priority | Complexity |
|--------|--------------|----------|------------|
| **Moe** | Mixtral, Qwen2-MoE | P3 | High - routing + top-K |
| **MlaSplit** | DeepSeek V2/V3 | P3 | Medium - tuple return |
| **MlaAttention** | DeepSeek V2/V3 | P3 | High - compressed KV |
| **AllReduce** | Tensor Parallel | P3 | N/A - NCCL/Metal collective |
| **AllGather** | Tensor Parallel | P3 | N/A - NCCL/Metal collective |
| **MmEmbedSplice** | Multimodal | P3 | Medium - vision embed splice |

## CUTLASS vs Metal Performance Shaders

**CUTLASS is CUDA-only and NOT applicable to Metal.**

### What CUTLASS Provides (CUDA)
- Optimized GEMM kernels with tile-level parallelism
- Mixed-precision support (FP16, INT8, INT4)
- Epilogue fusion (bias, activation, residual)
- Grouped GEMM for MoE

### Metal Equivalent: MPS (Metal Performance Shaders)
- **Already implemented** in `gemm.rs` via `MPSMatrixMultiplication`
- Apple's optimized BLAS library for Metal
- Supports FP16, FP32, BF16
- Hardware-accelerated on all Apple Silicon
- **No manual kernel tuning needed** - MPS handles optimization

### What We DON'T Need from CUTLASS
- ❌ CUTLASS kernel sources (CUDA-specific)
- ❌ CUTLASS templates (C++ metaprogramming)
- ❌ CUTLASS profiler (CUDA profiling)
- ❌ Manual tile size tuning (MPS auto-tunes)

### What We DO Need (Metal-specific)
- ✅ MPS GEMM wrapper (done in `gemm.rs`)
- ✅ Epilogue fusion via custom kernels (done in `fused_kernels.rs`)
- ✅ Quantized GEMM via AWQ dequant + MPS (done in `awq.rs`)
- 🔲 Grouped GEMM for MoE (future work, can use MPS batched API)

## Implementation Priority Order

### Phase 4.5 Completion ✅ COMPLETE (2026-05-05)
1. ✅ Embed - P0 (lookup table) - DONE
2. ✅ RopeAppend - P0 (rotary encoding) - DONE
3. ✅ RopeAppendInterleaved - P0 (interleaved variant) - DONE
4. ✅ Mul - P0 (elementwise multiply) - DONE
5. ✅ BiasAdd - P1 (broadcast add) - DONE
6. ✅ TanhSoftCap - P1 (logit capping) - DONE
7. ✅ Sub - P2 (for LayerNorm fusion) - DONE

### Phase 4.6: ICB Recording
- Implement `Instruction<W>::record_to_icb()` for all implemented ops
- Test ICB execution with simple instruction sequences

### Phase 4.7: Extended Features
1. RopeAppendInterleaved (Cohere)
2. SlidingAttention (Gemma2)
3. Mean (LayerNorm fusion)

### Phase 5+: Advanced Architectures
- MoE (Mixtral, Qwen2-MoE)
- MLA (DeepSeek V2/V3)
- Tensor Parallel ops (AllReduce, AllGather)
- Multimodal (MmEmbedSplice)

## File Organization

```
vllm-rs/crates/ferrite-forward-macro/src/metal/
├── mod.rs              # Module exports
├── activation.rs       # Silu, Gelu variants, FatReLU
├── add.rs             # Add (elementwise)
├── attention.rs       # Attention variants (4)
├── awq.rs            # AWQ quantization
├── fused_kernels.rs  # Multi-tile fusions
├── gemm.rs           # MPS GEMM wrapper
├── reshape.rs        # Reshape (metadata-only)
├── rmsnorm.rs        # RmsNorm
├── scalar_mul.rs     # ScalarMul (broadcast)
├── embed.rs          # ✅ Embed lookup (fp16, bf16)
├── rope.rs           # ✅ RopeAppend variants (NeoX, Interleaved)
├── mul.rs            # ✅ Elementwise multiply
├── bias_add.rs       # ✅ BiasAdd
├── softcap.rs        # ✅ TanhSoftCap (Gemma2)
└── sub.rs            # ✅ Sub (elementwise subtraction)
```

## CRITICAL: Implementation Trait API Reference

**THERE IS ONLY ONE Implementation API - DO NOT HALLUCINATE AN "OLD" API**

The correct API signature (from `impl_lib.rs` lines 458-550):

```rust
pub trait Implementation: fmt::Debug + Send + Sync {
    fn name(&self) -> &'static str;  // ← RETURNS &'static str, NOT String
    fn target_compatible(&self, profile: &TargetProfile) -> bool;
    fn workload_constraint(&self) -> WorkloadConstraint { WorkloadConstraint::Any }
    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo>;
    fn applies_to(&self, _ctx: &MatchContext) -> bool { true }
    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64;
    fn resources(&self, m: &MatchInfo) -> Resources;
    fn launch_kind(&self) -> LaunchKind;
    fn supported_input_handoffs(&self) -> &[Handoff];
    fn supported_output_handoffs(&self) -> &[Handoff];
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout>;
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout>;
    fn is_compute_bound(&self) -> bool { false }
    fn can_share_kernel_with(&self, _other: &dyn Implementation) -> bool { false }
    fn required_weights(&self, claimed_tiles: &[TileId], fuf: &Fuf, program: &Program) -> Vec<WeightAccessor>;
}
```

**Key imports needed:**
```rust
use crate::classified::{OpKind, Program};
use crate::fuf::{Fuf, TileId};
use crate::impl_lib::{
    CostCtx, Handoff, Implementation, LaunchKind, Layout, MatchInfo, Resources,
    WeightAccessor, WorkloadConstraint, default_required_weights,
};
use crate::target::{Backend, TargetProfile};
```

**Reference implementation:** See `add.rs` for a complete working example.

**DO NOT USE:**
- ❌ `use crate::tile_table::TileId` (wrong module)
- ❌ `use ferrite_metal_targets::TargetProfile` (wrong crate)
- ❌ `use crate::impl_lib::TileClaim` (doesn't exist)
- ❌ `use crate::shape::Shape` (wrong API)
- ❌ `fn name(&self) -> String` (wrong return type)

## Testing Strategy

Each implementation file should have:
1. Unit tests with reference implementations
2. Numerical accuracy tests (tolerance checks)
3. Performance benchmarks (optional but recommended)

Current test coverage:
- ✅ RmsNorm: 2 tests
- ✅ Gemm: 2 tests
- ✅ Attention: 8 tests (basic, paged, multi-head, optimized)
- ✅ Fused kernels: 6 tests
- ✅ Activation: 3 tests
- ✅ AWQ: 7 tests
- ✅ Add: 2 tests
- ✅ Reshape: 1 test
- ✅ ScalarMul: 2 tests
- ✅ Embed: 4 tests
- ✅ Rope: 8 tests (4 per variant)
- ✅ Mul: 4 tests
- ✅ BiasAdd: 4 tests
- ✅ TanhSoftCap: 4 tests
- ✅ Sub: 4 tests

**Total: 61 tests passing**

## Cost Model Strategy

All implementations use **analytical cost models** based on:
- Memory bandwidth (for memory-bound ops)
- Compute throughput (for compute-bound ops)
- Hardware characteristics from `TargetProfile`

No empirical benchmarking required for Phase 4.5 - analytical models are sufficient for solver to make reasonable choices.

## Next Session Quick Start

1. Check this file for current status
2. Pick next P0/P1 op from "CRITICAL PATH" section
3. Create implementation file in `src/metal/`
4. Add to `mod.rs` exports
5. Register in `impl_lib.rs::starter_library()`
6. Write tests
7. Update this tracker

## Notes

- **No CUTLASS dependency** - Metal uses MPS for GEMM
- **Modular structure** - Each kernel category in separate file
- **Analytical costs** - No need for empirical benchmarking yet
- **Test-driven** - Every implementation needs tests
- **Incremental progress** - Can test each op independently
