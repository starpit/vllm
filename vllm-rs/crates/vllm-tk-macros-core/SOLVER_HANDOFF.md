# CP5 Constraint Solver — Context Handoff

## What exists

A constraint-solving compiler that discovers optimal kernel execution plans
for LLM inference. Given a model DAG, a library of kernel implementations
(cuBLAS, CUTLASS, ThunderKittens, FlashInfer, vllm-rs), and a target GPU
profile, the solver produces a per-layer execution plan that minimizes
predicted wall-clock time subject to hardware and correctness constraints.

### Key files

| File | What it does |
|------|-------------|
| `lowering/tile_graph.rs` | Normalized DAG (15 tiles/layer for LLaMA) |
| `lowering/library.rs` | Implementation entries + GpuCostTable + CostCurve |
| `lowering/solver/backtrack_cp.rs` | B&B CP solver + PlanFamily + ASCII viz |
| `lowering/constraint.rs` | 8 declarative constraint types |
| `lowering/problem.rs` | Problem + PrecisionMode |
| `lowering/implementation.rs` | Implementation trait + Handoff + Layout |
| `target_profile.rs` | TargetProfile with seq_len + batch_size |
| `tests/scheduled_megakernel_test.rs` | Golden tests + bench + microbench sweeps |
| `csrc/cutlass_standalone_gemm.cu` | Standalone CUTLASS 128×128 + 64×64 launcher |

### Implementation library (l4_sm89_starter)

| Impl | Tiles claimed | Library |
|------|--------------|---------|
| TkFusedMlpBlockImpl | [norm+gate+up+cat+silu+down+res] (7) | TK |
| CublasGemmExWithResidualImpl | [oproj+res] or [down+res] (2) | cuBLAS |
| CutlassGemmWithResidualImpl | [oproj+res] or [down+res] (2) | CUTLASS |
| CutlassNormGemmImpl | [norm+qkv] or [norm+gate/up] (2) | CUTLASS |
| CutlassGemmImpl | single GEMM (1) | CUTLASS 128/64 |
| CublasGemmExImpl | single GEMM (1) | cuBLAS |
| VllmRsFusedQkvRopeCacheImpl | [split+rope+kv_w] (3) | vllm-rs |
| VllmRsSiluAndMulFusedImpl | [cat+silu] (2) | vllm-rs |
| VllmRsRmsNormImpl | [norm] (1) | vllm-rs |
| FlashInferStandaloneImpl | [attn] (1) | FlashInfer |

### Constraint system

1. **CoverComplete** — every tile claimed exactly once
2. **DependencyOrder** — producer step < consumer step
3. **IntermediateMaterialized** — fused tile with external consumers must write output
4. **PrecisionBounded** — handoff truncation vs fusion precision (Serving/BitExact modes)
5. **CooperativeExclusive** — at most one coop launch per step
6. **CompilationUnitRegBudget** / **ShmemBudget** — hardware resource limits
7. **HandoffCompatible** — handoff mechanism supported by both impls

### Measured data (L4 sm_89)

cuBLAS GEMM sweep (gate shape N=8192 K=2048):
```
M=1: 27µs  M=32: 39µs  M=64: 78µs  M=128: 71µs  M=1024: 393µs
```

CUTLASS 64×64 sweep (same shape):
```
M=1: 30µs  M=32: 31µs  M=64: 38µs  M=128: 75µs  M=1024: 621µs
```

CUTLASS 64×64 sweet spot: **M=32-64 (21-51% faster than cuBLAS)**.

### Solver-discovered plan family

```
M=1-16:   all cuBLAS (GEMV)                              10 launches/layer
M=32-64:  CUTLASS 64×64 for all GEMMs                    10 launches/layer
M=128:    cuBLAS attn + TK fused MLP                      6 launches/layer
M=256+:   cuBLAS/CUTLASS individual GEMMs                10 launches/layer
```

### Golden validation

- `cp5_solver_driven_matches_committed_golden`: passes at seq=64 with
  CUTLASS 64×64 (max_abs=8192, max_rel=0.94%)
- Graph replay also passes (bit-identical)

## What's next (priority order)

1. **Runtime integration** — PlanFamily drives vllm-rs forward pass
   (currently plans are test-only; the runtime uses hardcoded eager dispatch)
2. **Model generalization** — TileGraph for Mistral, Qwen, etc.
3. **Multi-GPU calibration** — GpuCostTable for A100, H100
4. **CUTLASS norm+GEMM prologue kernel** — impl exists in solver but
   no backing CUDA kernel (eliminates 2 launches/layer)
5. **TK sweep** — measured TK fused MLP costs across M values
6. **ILP backend** — the Solver trait is ready; needs MILP encoding

## Known issues

- `renders_polyalgorithm_kernel` test fails (pre-existing, unrelated)
- CutlassNormGemmImpl: solver doesn't pick it within 10K step budget
  (saving is ~1% of forward pass; B&B bound is too loose)
- `lowering/tests.rs` hand-built tests replaced with solver-driven
- TK fused MLP costs are placeholder (120µs at M=1, not measured)
- Elementwise costs are rough estimates, not measured

## Environment notes

- Must use `CUDA_PATH=/usr/local/cuda-12.9` (system nvcc 12.0 breaks TK)
- After clearing cudaforge cache: `touch crates/vllm-cuda/csrc/*.cu`
  then `cargo build -p vllm-kernels-cuda --features cuda` to regenerate
- CUDA kernel compilation takes minutes (cicc); don't kill cargo during build
