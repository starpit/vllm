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
| `lowering/backend/dispatch.rs` | DispatchSequence: plan → step-ordered typed entries |
| `lowering/backend/compile_dsl.rs` | `compile!` DSL parser (binding modes) |
| `lowering/backend/codegen.rs` | TokenStream codegen (fully specialized / GPU / dynamic) |
| `vllm-tk-macros/src/lib.rs` | `compile!` proc macro entry point |
| `vllm-cuda/src/model/solver_dispatch.rs` | FFI shim + `compile!` invocation site |
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

1. **TileGraph from DSL** — replace hardcoded `build_llama_forward`
   with DAG parsed from model description. Enables Mistral, Qwen, etc.
2. **`model: runtime` binding** — solver runs at model load, reads
   dims from loaded weights. No model catalog needed.
3. **CUTLASS norm+GEMM prologue kernel** — impl exists in solver but
   no backing CUDA kernel (eliminates 2 launches/layer)
4. **TK fused MLP wiring** — codegen placeholder exists, needs the
   TK launcher FFI in vllm-cuda
5. **Multi-GPU calibration** — run `gpu_cost_sweep` on A100, H100,
   check in CSV files
6. **ILP backend** — the Solver trait is ready; needs MILP encoding

## Architecture: compile-time vs runtime binding

The solver, constraints, and plan representation support three binding
modes from the same codebase. The `compile!` macro controls how much
is resolved at compile time vs deferred to runtime:

```
Fully specialized          GPU-specialized           Fully dynamic
(all compile-time)         (solver at startup)       (everything at startup)
─────────────────          ─────────────────         ─────────────────────
compile! {                 compile! {                compile! {
  model: llama_3_2_1b,      model: llama_3_2_1b,      model: runtime,
  target: l4_sm89,          target: runtime,           target: runtime,
  workloads: [1..1024],   }                          }
}
```

**Compile time**: TileGraph, CUDA kernel instantiations (standalone
launchers for HostCallback impls, megakernel bodies for DeviceCallable
impls), Rust dispatch functions, Implementation library entries.

**Runtime (model load, ~1s)**: GPU detection → GpuCostTable (cached
or microbench), PlanFamily::solve_grid(), store alongside weights.

**Per-forward (hot path)**: `plan_family.lookup(num_tokens)` → one
table lookup → walk pre-solved schedule → dispatch.

### Megakernel (1-launch) vs multi-launch

The solver's CompilationUnit assignment determines this:

- **sm_89 (L4)**: most impls are HostCallback (separate launches),
  because FlashInfer + CUTLASS can't share 99KB shmem. Result: 6-10
  launches/layer.

- **sm_90+ (H100)**: impls can be DeviceCallable (compiled into one
  `__global__`), sharing 228KB shmem and using mbarrier sync between
  steps. Result: 1 cooperative launch for the entire forward pass.

The solver is identical for both cases. The library entries differ
(DeviceCallable vs HostCallback), and the proc macro codegen branches
(standalone launcher vs megakernel body). The constraints
(CompilationUnitRegBudget, ShmemBudget) automatically determine how
many impls fit in one compilation unit.

### Fully specialized mode

When all variables are bound at compile time, the proc macro runs the
solver and emits monomorphized dispatch — no match on impl names, no
plan lookup, just a flat sequence of FFI calls per workload bucket:

```rust
pub fn forward(tokens: &Tensor, num_tokens: u32) {
    match num_tokens {
        0..=16   => plan_m1_dispatch(tokens),    // cuBLAS GEMV
        17..=48  => plan_m32_dispatch(tokens),   // CUTLASS 64×64
        49..=192 => plan_m128_dispatch(tokens),  // TK fused MLP
        _        => plan_m1024_dispatch(tokens),  // cuBLAS
    }
}
```

## How to run

All tests require `CUDA_PATH=/usr/local/cuda-12.9`:

```bash
# Solver tests (no GPU needed, fast):
cargo test -p vllm-tk-macros-core -- --nocapture

# Plan family visualization:
cargo test -p vllm-tk-macros-core print_plan_family_compact -- --nocapture

# Golden test (needs GPU, ~80s):
CUDA_PATH=/usr/local/cuda-12.9 cargo test -p vllm-tk-test-harness \
  --features cuda --test scheduled_megakernel_test \
  cp5_solver_driven_matches_committed_golden -- --ignored --nocapture

# Bench (needs GPU, ~45s):
CUDA_PATH=/usr/local/cuda-12.9 cargo test -p vllm-tk-test-harness \
  --features cuda --test scheduled_megakernel_test \
  cp5_solver_driven_natural_forward_bench -- --ignored --nocapture

# cuBLAS sweep:
CUDA_PATH=/usr/local/cuda-12.9 cargo test -p vllm-tk-test-harness \
  --features cuda --test scheduled_megakernel_test \
  cublas_gemm_sweep_microbench -- --ignored --nocapture

# CUTLASS vs cuBLAS comparison:
CUDA_PATH=/usr/local/cuda-12.9 cargo test -p vllm-tk-test-harness \
  --features cuda --test scheduled_megakernel_test \
  cutlass_vs_cublas_sweep -- --ignored --nocapture
```

## Interpreter dispatch pattern

The CP5 bench/golden tests contain a "runtime interpreter" — a
`dispatch_one` closure that matches on impl name and calls the
corresponding FFI:

```rust
let mut dispatch_one = |sg: SubgraphId| {
    let imp_name = library.get(plan.assignment.impls[&sg]).name();
    match imp_name {
        "cublas_gemm_ex_qkv" => gemm(seq, qkv_dim, hd, w, act, out, 0.0),
        "cutlass_qkv_64x64"  => ffi::cutlass_gemm_64x64_launch(...),
        "vllm_rs_rms_norm"    => ffi::rms_norm_bf16(...),
        "flashinfer_standalone_fa2" => ffi::cp5_run_flashinfer_attention_for_layer(...),
        "vllm_rs_fused_qkv_rope_cache" => ffi::fused_qkv_rope_cache_bf16(...),
        "tk_fused_mlp_block"  => ffi::cp5_fused_mlp_launch(...),
        name if name.starts_with("cutlass_") => { /* generic CUTLASS dispatch */ },
        ...
    }
};
for (_, sg) in &scheduled { dispatch_one(*sg); }
```

This is the pattern the runtime integration should follow — the proc
macro generates the dispatch function at compile time.

## Relationship to the existing megakernel

The existing megakernel (`templates/scheduled/megakernel.cu`) is a
hand-tuned cooperative-grid kernel that runs the entire LLaMA forward
pass in one launch. It uses:
- Barrier-based phase synchronization (gmem flags)
- CUTLASS GEMM phases with custom prologues
- FlashInfer attention inlined via `BlockBatchPagedAttentionPersistent`
- TK RMSNorm + RoPE + SiLU tile bodies

Performance: 55ms at seq=1024 on L4 (vs 40ms for the CP5 solver's
multi-launch cuBLAS plan). The megakernel is slower because:
- Single cooperative grid limits parallelism (1 CTA/SM on L4)
- All dispatch arms compiled into one kernel → register spill (255 regs)
- Barrier overhead between phases (~100µs per grid sync)

The CP5 solver replaces the megakernel with a **multi-launch plan**
that lets each kernel use its optimal register/shmem budget. On sm_90+
with more shmem and mbarrier, the solver could discover that a
megakernel IS optimal — but it would be a solver-discovered megakernel,
not a hand-tuned one.

## Why not torch.compile / Triton / TensorRT

torch.compile: fuses pointwise ops via Triton but doesn't jointly
optimize GEMM kernel selection × fusion × workload shape. One plan
for all batch sizes. Can't pick CUTLASS 64×64 at M=32 and cuBLAS at
M=1.

TensorRT: profiles multiple backends (closest to us) but uses
pattern-matching for fusion, not constraint solving. Can't express
"this fusion is invalid because tile X has an external consumer."
Opaque plan — can't inspect why it chose what.

Our approach: the solver discovers fusion from the constraint system.
Adding a new kernel is one Implementation struct. The constraints
validate it automatically. Plans are inspectable (`tag:l([a+b])`
visualization). Per-GPU calibrated via microbench sweep.

## forward! macro — runtime integration (CP5-C)

The `forward!` proc macro generates `solver_forward_layer()` — a
drop-in replacement for `LlamaDecoderLayer::forward()` that uses
the constraint solver's optimal kernel mix per workload bucket.

```rust
// In vllm-cuda/src/model/solver_dispatch.rs:
vllm_tk_macros::forward! {
    model: llama_3_2_1b,
    target: l4_sm89,
    workloads: [1..4096],
}
```

### What it emits

The macro runs the solver at compile time and emits:
- `solver_forward_layer(layer, num_tokens, ...)` — matches on
  `num_tokens` and dispatches to per-bucket functions.
- `solver_layer_bucket_N(layer, ...)` — one per workload bucket,
  each a complete layer body using solver-selected kernels.

The generated code uses the same types as the existing forward
pass: `OwnedTensor`, `GpuTensor`, `CublasHandle`, `kernels::*`.
No FFI shim, no raw pointer ctx struct.

### Integration

`LlamaModel` has a `use_solver_dispatch: bool` field. When true,
`forward()` calls `solver_forward_layer()` instead of
`layer.forward()`. Currently defaults to false.

### What works now

- Solver runs at proc-macro time (fully specialized path)
- Per-bucket layer bodies are generated with correct types
- cuBLAS GEMM dispatch (via `LinearLayer::forward()`)
- Norm, attention, MLP delegate to existing kernels

### What needs wiring

- CUTLASS standalone launchers (codegen falls back to cuBLAS)
- TK fused MLP launch (codegen falls back to eager MLP)
- GPU-specialized / fully dynamic paths (emit empty placeholders)
- TileGraph from DSL instead of hardcoded `build_llama_forward()`

### Tests (22 passing)

```bash
cargo test -p vllm-tk-macros-core -- lowering
```

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
