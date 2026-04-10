# CP5 Constraint Solver — Complete Handoff

## What this is

A **compiler** that discovers optimal kernel execution plans for LLM
inference. Given a model DAG (parsed from a DSL), a library of kernel
implementations (cuBLAS, CUTLASS, FlashInfer, vllm-rs), and a target
GPU cost table (measured via sweep), the solver produces per-workload
execution plans that minimize predicted wall-clock time.

The `forward!` proc macro is the entry point. It runs the solver at
compile time and emits `solver_forward_layer()` — a drop-in replacement
for `LlamaDecoderLayer::forward()` that dispatches to solver-selected
kernels based on `num_tokens`.

**Result**: 5% throughput improvement on Llama 3.2 3B (L4 GPU),
verified correct output on both decode and prefill.

## Rules of the road

**These are critical. Violating them produces garbage or regressions.**

1. **Everything is DAG-driven.** The codegen walks the solver's
   `DispatchSequence` entry by entry and emits one kernel call per
   entry. It does NOT interpret, merge, shortcut, or second-guess
   the plan. The dispatch sequence is the IR. The codegen is the
   backend. A compiler doesn't make stuff up.

2. **No hardcoded numbers.** GEMM costs come from the CSV cost table
   (`data/cost_l4_sm89.csv`), measured by `gpu_cost_sweep`. Model
   dimensions come from the `forward!` DSL (`models: [{ ... }]`).
   The tile graph is built from the DSL body via `from_model_dag`.
   Nothing is hardcoded to a specific model or shape.

3. **Every DispatchEntry field matters.** `fused_residual`, `gemm_phase`,
   `kind`, `is_attn_norm` — the codegen must respect all of them. Ignoring
   `fused_residual` broke the residual stream and produced garbage.

4. **Decode vs prefill are separate implementations.** The solver
   produces different plans for `seq_len=1` (decode) and `seq_len>1`
   (prefill). The implementation library has separate entries:
   `VllmRsFusedQkvRopeCacheImpl` (decode only) vs
   `VllmRsPrefillRopeCacheImpl` (prefill only), and
   `FlashInferStandaloneImpl` (decode) vs `FlashInferStandardImpl`
   (prefill). The codegen emits different kernel calls for each.
   No runtime branching on `max_seqlen_q` in the generated code.

5. **CUTLASS GEMMs always beta=0.** The output buffer is freshly
   allocated. Residual accumulation happens in
   `fused_add_rms_norm_inplace`, not in the GEMM epilogue.

6. **Test-driven.** Write tests that would catch the bug BEFORE
   running on GPU. Structural codegen tests (does the output contain
   `reshape`? does gate_up_proj.forward appear only once per bucket?)
   catch dataflow bugs at compile time.

## The `forward!` DSL

```rust
vllm_tk_macros::forward! {
    // Body: model structure (required, always first).
    // Same syntax as megakernel! DSL.
    for layer in 0..NL {
        let normed = rmsnorm(hidden_states, attn_norm[layer]);
        let qkv = gemm(normed, qkv_weights[layer]);
        let (q, k, v) = rope_append(qkv, positions, kv_cache[layer]);
        let attn = attention_decode(q, k, v, kv_cache[layer], block_table);
        hidden_states = gemm_add(attn, o_proj[layer], hidden_states);

        let normed2 = rmsnorm(hidden_states, mlp_norm[layer]);
        let gate = silu(gemm(normed2, gate_weights[layer]));
        let up = gemm(normed2, up_weights[layer]);
        hidden_states = gemm_add(gate * up, down_proj[layer], hidden_states);
    }

    // Optional fields after the body:
    models: [
        { layers: 28, hidden: 3072, intermediate: 8192,
          heads: 24, kv_heads: 8, head_dim: 128 },
    ],
    target: l4_sm89,
    workloads: [1..4096],
}
```

- `models:` — omit for runtime (solver at startup). List multiple
  for multi-model binary.
- `target:` — omit for runtime (GPU detection at startup).
- `workloads:` — omit for default grid (1,2,4,...,4096).

## Pipeline

```
forward! DSL
    │
    ▼
MegakernelDef parser (parse.rs)
    │
    ▼
ModelDag (dag.rs) — typed DAG with buffer shapes
    │
    ▼
TileGraph::from_model_dag (tile_graph.rs)
    │  Maps OpKind → TileKind(s):
    │  RmsNorm → RmsNorm
    │  Gemm → GemmQkv/Gate/Up (by weight name)
    │  GemmAdd → GemmOProj/Down + ResidualAdd
    │  RopeAppend → QkvSplit + Rope + KvCacheWrite
    │  AttentionDecode → Attention
    │  Silu + Mul → GateUpConcat + SiluMul
    │
    ▼
Problem::build (problem.rs)
    │  Tile graph + Implementation library + TargetProfile
    │  → auto-generates constraints
    │
    ▼
PlanFamily::solve_grid (solver/mod.rs)
    │  For each seq_len in grid:
    │    BacktrackCpSolver → ExecutionPlan
    │
    ▼
DispatchSequence::from_plan (backend/dispatch.rs)
    │  Flattens plan into step-ordered typed entries
    │  Each entry: ImplDispatchKind + GemmPhase + layer + flags
    │
    ▼
codegen::generate (backend/codegen.rs)
    │  Walks entries, emits one Rust call per entry
    │
    ▼
solver_forward_layer() — the generated function
```

## Key files

| File | Purpose |
|------|---------|
| **Solver core** | |
| `lowering/tile_graph.rs` | TileGraph, TileKind, ModelDims, `from_model_dag()` |
| `lowering/library.rs` | Implementation entries, GpuCostTable, l4_cost_model |
| `lowering/cost_table.rs` | GpuCostGrid: CSV-based (M,N,K) cost lookup |
| `lowering/solver/backtrack_cp.rs` | B&B CP solver |
| `lowering/solver/mod.rs` | PlanFamily, ExecutionPlan, Solver trait |
| `lowering/constraint.rs` | 8 declarative constraint types |
| `lowering/problem.rs` | Problem + PrecisionMode |
| `lowering/implementation.rs` | Implementation trait, Handoff, Layout, LaunchKind |
| **Codegen backend** | |
| `lowering/backend/compile_dsl.rs` | ForwardDef parser (body + models + target + workloads) |
| `lowering/backend/dispatch.rs` | DispatchSequence, ImplDispatchKind, GemmPhase |
| `lowering/backend/codegen.rs` | TokenStream emission — one call per dispatch entry |
| `lowering/backend/codegen_test.rs` | Structural codegen tests |
| **Proc macro** | |
| `vllm-tk-macros/src/lib.rs` | `forward!` proc macro entry point |
| **Runtime integration** | |
| `vllm-cuda/src/model/solver_dispatch.rs` | forward! invocation + CUTLASS FFI declarations |
| `vllm-cuda/src/model/llama.rs` | LlamaModel::forward() calls solver_forward_layer() |
| `vllm-cuda/src/layers.rs` | LinearLayer::dense_weight() for CUTLASS pointer extraction |
| `vllm-cuda/csrc/cutlass_standalone_gemm.cu` | CUTLASS 64×64 + 128×128 standalone launchers |
| `vllm-cuda/build.rs` | Links libcutlass_standalone_gemm.a |
| `vllm-kernels-cuda/build.rs` | Compiles cutlass_standalone_gemm.cu |
| **Cost data** | |
| `data/cost_l4_sm89.csv` | 702 measured cost points (18 shapes × 13 M × 3 kernels) |
| **Sweep test** | |
| `tests/scheduled_megakernel_test.rs` | `gpu_cost_sweep` — THE entrypoint for new GPUs |

## Adding a new GPU

Run the sweep test on the target GPU:

```bash
CUDA_PATH=/usr/local/cuda-12.9 cargo test -p vllm-tk-test-harness \
  --features cuda --test scheduled_megakernel_test \
  gpu_cost_sweep -- --ignored --nocapture \
  2>/dev/null > crates/vllm-tk-macros-core/data/cost_YOUR_GPU.csv
```

Clean the test runner lines from the CSV:
```bash
grep -E "^#|^kernel|^cublas|^cutlass" cost_YOUR_GPU.csv > clean.csv
mv clean.csv cost_YOUR_GPU.csv
```

Update `cost_table.rs` to load the new CSV, add a new `TargetId` in
`compile_dsl.rs`, and a new `TargetProfile` constructor. The solver
automatically produces optimal plans for the new GPU.

## Adding a new model

Change the `models:` field in `solver_dispatch.rs`:

```rust
models: [
    { layers: 32, hidden: 4096, intermediate: 14336,
      heads: 32, kv_heads: 8, head_dim: 128 },
],
```

If the model has the same structure as LLaMA (norm → QKV → rope →
attention → oproj → residual → norm → gate → up → silu → down →
residual), it just works. Different dims, same body.

For non-LLaMA architectures (MoE, sliding window, etc.), add new
`OpKind` variants to `dag.rs`, new `TileKind` variants to
`tile_graph.rs`, and new `Implementation` entries to `library.rs`.

## Implementation library (current)

| Impl | Tiles claimed | Decode/Prefill | Status |
|------|--------------|----------------|--------|
| CublasGemmExImpl | single GEMM (1) | both | ✅ wired |
| CublasGemmExWithResidualImpl | GEMM+res (2) | both | ✅ wired |
| CutlassGemmImpl (64/128) | single GEMM (1) | both | ✅ wired |
| CutlassGemmWithResidualImpl | GEMM+res (2) | both | ✅ wired |
| VllmRsRmsNormImpl | norm (1) | both | ✅ wired |
| VllmRsFusedQkvRopeCacheImpl | split+rope+kv (3) | decode only | ✅ wired |
| VllmRsPrefillRopeCacheImpl | split+rope+kv (3) | prefill only | ✅ wired |
| VllmRsSiluAndMulFusedImpl | cat+silu (2) | both | ✅ wired |
| FlashInferStandaloneImpl | attn (1) | decode only | ✅ wired |
| FlashInferStandardImpl | attn (1) | prefill only | ✅ wired |
| CutlassNormGemmImpl | norm+GEMM (2) | both | ❌ no backing kernel |
| TkFusedMlpBlockImpl | 7-tile MLP | both | ❌ disabled (codegen not wired) |
| KvCacheWriteImpl | kv_write (1) | both | ✅ noop |
| ResidualAddImpl | res_add (1) | both | ✅ noop |

## Codegen dataflow variables

The generated per-bucket function uses these `Option<OwnedTensor>`
variables to pass data between dispatch entries:

| Variable | Written by | Read by |
|----------|-----------|---------|
| `hidden_states` | input, OProj, Down | attn norm, MLP norm, output |
| `residual` | attn norm | MLP norm, output |
| `normed` | RmsNorm | QKV GEMM, Gate GEMM, Up GEMM |
| `qkv_out` | QKV GEMM, FusedQkvRopeCache, PrefillRopeCache | RopeCache, Attention |
| `k_out` / `v_out` | PrefillRopeCache | FlashInferStandard |
| `attn_out` | Attention | OProj |
| `gate_up` | Gate GEMM, Up concat | SiluAndMul |
| `silu_out` | SiluAndMul | Down GEMM |

## What's next (priority order)

1. **`models: runtime` / `target: runtime`** — solver runs at startup
   with dims from loaded weights. Any model without recompiling.
   The ForwardDef parser already accepts `models: runtime`.

2. **TK fused MLP wiring** — `TkFusedMlpBlockImpl` is disabled
   (`target_compatible = false`). Wire the TK launcher FFI into
   vllm-cuda so the solver can pick it at M=128. The codegen has
   a `compile_error!` placeholder for `ImplDispatchKind::TkFusedMlpBlock`.

3. **CUTLASS norm+GEMM prologue** — `CutlassNormGemmImpl` exists in
   the library but has no backing CUDA kernel. Eliminates 2 norm
   launches/layer by folding RMSNorm into the CUTLASS prologue.

4. **TP all-reduce tiles** — add `TileKind::AllReduce` after OProj
   and Down in the tile graph. Required for multi-GPU correctness.
   With TP, GEMM shapes change (sharded dims) — needs `models: runtime`.

5. **Multi-GPU cost CSVs** — run `gpu_cost_sweep` on A100, H100.
   Check in the CSVs. Add `TargetId` variants.

6. **Elementwise + attention in CSV** — norm, SiLU, FlashInfer costs
   are still hardcoded `CostCurve`. Move them to the sweep.

7. **Remove `megakernel!`** — dead code, replaced by `forward!`.

8. **ILP backend** — the `Solver` trait is ready; `solver::ilp` is a
   stub. Encode constraints as MILP when the CP solver proves
   intractable on richer libraries.

## How to run

```bash
# Solver + codegen tests (no GPU needed, ~30s):
cargo test -p vllm-tk-macros-core -- lowering

# Build with CUDA (compiles CUTLASS standalone GEMM):
cargo build -p vllm-cuda --features cuda --release

# GPU cost sweep (needs GPU, ~75s, generates CSV):
CUDA_PATH=/usr/local/cuda-12.9 cargo test -p vllm-tk-test-harness \
  --features cuda --test scheduled_megakernel_test \
  gpu_cost_sweep -- --ignored --nocapture

# Benchmark:
./target/release/vllm bench throughput MODEL --input-len 64
./target/release/vllm bench latency MODEL
```

## Known issues

- `renders_polyalgorithm_kernel` test fails (pre-existing, unrelated)
- CutlassNormGemmImpl: solver doesn't pick it within 10K step budget
- TkFusedMlpBlockImpl: disabled until codegen is wired
- Elementwise/attention costs are rough estimates, not in CSV yet
- Multi-model (`models: [...]` with >1 entry) not yet implemented
  (uses first model only)

## Environment notes

- `CUDA_PATH=/usr/local/cuda-12.9` required (system nvcc 12.0 breaks TK)
- After clearing cudaforge cache: `touch crates/vllm-cuda/csrc/*.cu`
  then `cargo build -p vllm-kernels-cuda --features cuda` to regenerate
- CUDA kernel compilation takes minutes (cicc); don't kill cargo
- CSV file is gitignored by default — use `git add -f` to check it in
