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

7. **Don't hack around solver picks with `target_compatible = false`.**
   If the solver picks a wrong impl, the cost or constraint model is
   missing information — add it. `target_compatible` is about GPU
   *capability* (sm_89 vs sm_90), NOT about workload shape or kernel
   correctness. Use `workload_constraint()` for "this kernel only
   handles M=1" or "this fused kernel requires a specific batch
   range". Use `cost_us` for soft signals. If neither expresses what
   you need, **add a new constraint kind** to `Constraint` and the
   `workload_constraint()`-equivalent trait method — that's what
   happened for the GEMV/TK workload-range issue. The constraint
   system is the IR; extend it, don't lie to it.

8. **Benchmark ≠ correctness.** `vllm bench latency` measures wall
   clock, not output quality. It won't catch GEMMs producing garbage —
   you'll just see a stable number with nonsense logits. Always verify
   correctness with `vllm chat` or a golden test after any kernel
   dispatch change. The cuBLAS-free regression (commit `c01fa1e57`)
   was masked for hours because the latency benchmark was happy.

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
    // Post-loop: runs once per forward pass after all decoder layers.
    // Currently we model only the lm_head GEMM; the final rmsnorm
    // is still emitted by the caller via fused_add_rms_norm_inplace.
    logits = gemm(hidden_states, lm_head);

    // Optional fields after the body:
    models: [
        { layers: 28, hidden: 3072, intermediate: 8192,
          heads: 24, kv_heads: 8, head_dim: 128, vocab: 128256 },
    ],
    target: l4_sm89,
    workloads: [1..4096],
}
```

- `models:` — omit for runtime (solver at startup). List multiple
  for multi-model binary. `vocab` defaults to 128256 (Llama 3) if
  omitted.
- `target:` — omit for runtime (GPU detection at startup).
- `workloads:` — omit for default grid (1,2,4,...,4096).

The codegen emits two entry points: `solver_forward_layer()` (one
call per layer from the model's Rust code) and `solver_forward_lm_head()`
(one call at the end of the forward pass, replacing the pre-Ferrite
`self.lm_head.forward(...)`).

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

The codegen emits two top-level dispatchers per `forward!` invocation:
`solver_forward_layer()` for the decoder layer body (called once per
layer by the model's `forward` method) and `solver_forward_lm_head()`
for the post-loop `logits = gemm(hidden_states, lm_head)` phase.

## Constraint kinds

Declarative data, not closures, so the solver dispatches on variants
and the ILP backend can linearize each kind. The `Constraint` enum
currently has these variants (see `lowering/constraint.rs`):

1. **CoverComplete** — every tile claimed exactly once
2. **SubgraphMatches** — chosen impl's matcher accepts the claimed tiles
3. **TargetCompatible** — chosen impl reports `target_compatible(profile)`
   (GPU capability, e.g. sm_89 vs sm_90)
4. **WorkloadCompatible** — chosen impl's `workload_constraint()` accepts
   `profile.num_tokens()` (correctness requirements on workload shape,
   e.g. "GEMV only handles M=1")
5. **DependencyOrder** — producer's step < consumer's step (or same
   subgraph with internal handoff)
6. **CooperativeExclusive** — ≤1 `CooperativeLaunch` per step
7. **CompilationUnitRegBudget** — per-unit union regs ≤ cap
8. **CompilationUnitShmemBudget** — per-unit union shmem ≤ cap
9. **HandoffCompatible** — per producer→consumer edge, handoff is in
   both impls' supported sets and available on target
10. **IntermediateMaterialized** — claimed tile with external consumers
    must expose its output to GMEM
11. **PrecisionBounded** — optionally force truncation at a dep edge

The solver's backtracking search also filters candidates early by
`target_compatible` and `workload_constraint` (the two correctness
gates) before computing cost.

**If a kernel has a correctness requirement the solver doesn't know
about, add a constraint kind**. Do not hack around it via cost, and
do not lie via `target_compatible`.

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
| `vllm-cuda/src/model/solver_dispatch.rs` | forward! invocation + CUTLASS/GEMV FFI via `cutlass_gemm_ffi!` macro |
| `vllm-cuda/src/model/llama.rs` | LlamaModel::forward() calls solver_forward_layer() |
| `vllm-cuda/src/layers.rs` | LinearLayer::dense_weight() for CUTLASS pointer extraction |
| `vllm-cuda/csrc/cutlass_standalone_gemm.cu` | **canonical** CUTLASS GEMM grid (16 configs + GEMV) via `CUTLASS_GEMM` macro |
| `vllm-cuda/build.rs` | Links libcutlass_standalone_gemm.a |
| `vllm-kernels-cuda/build.rs` | Compiles cutlass_standalone_gemm.cu + cp5_fused_mlp (TK) |
| `vllm-tk-test-harness/build.rs` | Builds test harness; references **canonical** `cutlass_standalone_gemm.cu` (no duplicate) |
| **Cost data** | |
| `data/cost_l4_sm89.csv` | 3999 measured cost points (18 shapes × 13 M × 17 kernels) |
| **Sweep test** | |
| `vllm-tk-test-harness/tests/scheduled_megakernel_test.rs` | `gpu_cost_sweep` — THE entrypoint for new GPUs. Benchmarks via the `bench_cutlass!` data-driven macro so new configs are picked up automatically. |

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

| Impl | Tiles claimed | Workload | Status |
|------|--------------|----------|--------|
| CutlassGemmImpl (16 configs × 6 phases = 96) | single GEMM | any M | ✅ wired, covers QKV/OProj/Gate/Up/Down/**LmHead** |
| CutlassGemmWithResidualImpl (16 × 2 = 32) | GEMM+res (2) | any M | ✅ wired (OProj, Down) |
| CutlassGemvImpl (4 phases) | single GEMM | **M=1 only** (WorkloadConstraint) | ✅ wired (QKV/Gate/Up/LmHead) |
| CublasGemmExImpl (per-phase) | single GEMM | any M | ✅ wired |
| CublasGemmExWithResidualImpl (2) | GEMM+res (2) | any M | ✅ wired (OProj, Down) |
| VllmRsRmsNormImpl | norm (1) | any M | ✅ wired |
| VllmRsFusedQkvRopeCacheImpl | split+rope+kv (3) | decode only | ✅ wired |
| VllmRsPrefillRopeCacheImpl | split+rope+kv (3) | prefill only | ✅ wired |
| VllmRsSiluAndMulFusedImpl | cat+silu (2) | any M | ✅ wired |
| FlashInferStandaloneImpl | attn (1) | decode | ✅ wired |
| FlashInferStandardImpl | attn (1) | prefill | ✅ wired |
| CutlassNormGemmImpl | norm+GEMM (2) | any M | ❌ no backing kernel |
| TkFusedMlpBlockImpl | 7-tile MLP | **M=1 only** (WorkloadConstraint) | ⚠️ kernel is single-row; constraint prevents picks at M>1 |
| KvCacheWriteImpl | kv_write (1) | any M | ✅ noop |
| ResidualAddImpl | res_add (1) | any M | ✅ noop |

### CUTLASS tile grid

Defined via C macro `CUTLASS_GEMM(M, N, K, WM, WN, WK, STAGES)` in
`vllm-cuda/csrc/cutlass_standalone_gemm.cu`. Each entry stamps out a
kernel + `extern "C"` launch wrapper. Current grid:

| TB shape | Stages | Use case |
|----------|--------|----------|
| 32×64    | 3, 4   | M=2–16 small-batch decode |
| 32×128   | 3, 4   | M=2–16 gate/up where N>>K |
| 32×256   | 3      | M=2–16 large output dim |
| 64×64    | 3, 4   | General small-M |
| 64×128   | 3, 4   | M≈16-64, larger N |
| 128×64   | 3, 4   | M≈64-128 |
| 128×128  | 3, 4   | General large-M |
| 128×256  | 3      | Large prefill, wide N |
| 256×64   | 3, 4   | Large prefill, narrow N |
| ~~256×128 s3~~ | — | excluded — exceeds 99 KB sm89 SMEM |

Plus `cutlass_gemv_launch` (SIMT GEMV for M=1, wraps `cutlass::gemm::device::Gemv`).

Adding a new tile config is **three edits**: the `CUTLASS_GEMM(...)`
line in the `.cu`, the `(M, N, K)` tuple in `CutlassGemmImpl::all_configs`
and `CutlassGemmWithResidualImpl::all_configs` in `library.rs`, and the
symbol name in `cutlass_gemm_ffi!` in both `solver_dispatch.rs` and
`vllm-tk-test-harness/src/lib.rs`. Then re-run `gpu_cost_sweep`.

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

## Done recently (2026-04-11)

- ✅ **Expanded CUTLASS grid**: 2 tile configs → 16 configs via a C macro
  (`CUTLASS_GEMM(TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, STAGES)`).
  Covers the same threadblock shapes cuBLAS picks on sm80/sm89 plus
  both 3 and 4 pipeline stages. Adding a new config is 3 edits + re-sweep.
- ✅ **CUTLASS GEMV at M=1**: `CutlassGemvImpl` wraps
  `cutlass::gemm::device::Gemv`. 1.5–1.9× faster than cuBLAS at M=1 on
  the shapes that matter for small LLMs.
- ✅ **Dims-aware cost model**: plumbed `ModelDims` through
  `l4_sm89_starter` → `CutlassGemmImpl::cost_us()`. Previously used
  hardcoded 1B shapes which caused the solver to pick wrong kernels on
  3B models.
- ✅ **String-keyed cost table**: `GpuCostGrid::lookup` takes `&str`
  instead of an enum. CSV kernel names (`cutlass_{M}x{N}_s{stages}`,
  `cutlass_gemv`) map directly — no code change needed when adding
  configs.
- ✅ **Single-source CUTLASS GEMM**: test harness now builds the
  canonical `vllm-cuda/csrc/cutlass_standalone_gemm.cu` directly instead
  of a drifting duplicate.
- ✅ **Workload compatibility constraint** (commit `700682422`): new
  `WorkloadConstraint` enum + `Implementation::workload_constraint()`
  trait method. Distinct from `target_compatible` (GPU capability):
  expresses correctness requirements on the workload shape — e.g.
  "GEMV only handles M=1" or "this fused kernel requires M in a
  specific range". The solver enforces it via an early filter in all
  three candidate-enumeration sites alongside `target_compatible`,
  plus a new `Constraint::WorkloadCompatible` variant for ILP later.
  `CutlassGemvImpl` and `TkFusedMlpBlockImpl` both declare
  `NumTokensRange { min: 1, max: 1 }` — the first because GEMV is
  mathematically a matrix-vector product, the second because the
  kernel is single-row by design (norm phase hardcodes row=0).
- ✅ **`forward!` DSL now covers lm_head** (commit `700682422`):
  `from_model_dag` processes post-loop ops (tagged with layer=num_layers).
  New `TileKind::GemmLmHead` + `GemmPhase::LmHead`. `ModelDims` gains
  `vocab_size`. `CutlassGemmImpl`/`CutlassGemvImpl` both extend
  `all_configs()` to cover the LmHead phase. Codegen emits a new
  `solver_forward_lm_head()` dispatcher alongside `solver_forward_layer()`.
  `LlamaForCausalLM::forward` now calls it instead of
  `self.lm_head.forward()`. **This means Ferrite manages the whole
  forward pass, not just one decoder layer body.**
- ✅ **cuBLAS-free solver path is correct** (commit `c01fa1e57`):
  shipped a latent bug in `cutlass_input_expr(OProj)` — flash attention
  output is 3D `[num_tokens, num_q_heads, head_dim]`, but the CUTLASS
  OProj path passed the raw tensor without reshaping, so the kernel
  saw `K = num_q_heads = 24` instead of `K = q_size = 3072`. The cuBLAS
  OProj path reshapes via `gemm_operands`, masking the bug until
  cuBLAS was removed from the library. Verified cuBLAS-free chat
  produces coherent output ("why is the sky blue?" → correct Rayleigh
  scattering explanation).
- ✅ **TK fused MLP wiring investigation**: fully wired the solver
  codegen + FFI + build, then discovered the underlying kernel is
  **single-row by design** (norm phase hardcoded to row=0, GEMM tiles
  are internal tiling not batch parallelism). Now expressed as a
  `WorkloadConstraint::NumTokensRange { min: 1, max: 1 }` — correct
  even though the kernel is still bound to M=1.
- ✅ **Dev-mode solver perf investigation**: thought the solver was
  slow (16s for lowering tests), actually it's ~70ms per `forward!`
  expansion in release mode — the 16s was 35 debug-mode tests × ~500ms
  each. ILP backend is unnecessary at current library size.

## What's next (priority order)

1. **Port other model architectures to `forward!`** — currently only
   Llama is Ferrite-managed. Every non-Llama model (qwen3_moe, gemma2,
   gemma3, deepseek_v2, commandr, qwen3_next, plus non-solver Llama
   fallback paths like PP boundaries) still calls `LinearLayer::Dense::forward()`
   which goes to `cublas.gemm()`. **This is the gate on binary-level
   cuBLAS elimination** (TODO #2). The Llama `forward!` DSL is the
   template — each arch needs its own DSL describing the forward pass,
   plus any new op/tile kinds its architecture introduces (MoE routing,
   sliding window attention, multi-token prediction, ...).

2. **Binary-level cuBLAS elimination** — blocked by TODO #1. To drop
   `libcublas.so.12` (816 MB) from the container image, every path that
   reaches `LinearLayer::Dense::forward()` must be replaced with a
   Ferrite-managed `solver_forward_*` call OR `LinearLayer::Dense` must
   get a CUTLASS backend. The latter is a smaller change and would work
   for every model at once, but bypasses Ferrite's kernel selection.
   The former is the principled fix. Either way, once no code reaches
   `cublas.gemm()`, remove `cublas`/`cublaslt` features from `cudarc`
   in `vllm-cuda/Cargo.toml`. Ferrite's Llama path is already
   cuBLAS-free-correct (commit `c01fa1e57`); what's missing is arch
   coverage.

3. **`models: runtime` / `target: runtime`** — solver runs at startup
   with dims from loaded weights. Any model without recompiling.
   The `ForwardDef` parser already accepts `models: runtime`. Blocks
   multi-model deployments and simplifies the "different dims = different
   binary" workflow. Related: enables TP where GEMM shapes change at
   load time based on sharding.

4. **CUTLASS norm+GEMM prologue** — `CutlassNormGemmImpl` exists in
   the library but has no backing CUDA kernel. Eliminates 2 norm
   launches/layer by folding RMSNorm into the CUTLASS prologue. Profile
   shows RMSNorm is ~2.5% of GPU time — real gain is ~1-2% after
   accounting for launch overhead savings. Lower priority than it looks.

5. **Batch-aware TK fused MLP** — the current kernel is single-row by
   design (norm processes exactly 1 row, GEMM's 128-row tile is K-dim
   tiling not batch). To make fusion worthwhile for M > 1, either:
   (a) rewrite the norm phase to cooperatively norm all M rows, or
   (b) launch grid=M with per-row addressing throughout. Theoretical
   win: ~5-15% at BS=1-4 from eliminating GMEM round-trips between
   norm/gate/up/silu/down. Real kernel work, not a config change.
   Once the kernel is fixed, widen its `WorkloadConstraint` past the
   current `NumTokensRange { min: 1, max: 1 }`.

6. **TP all-reduce tiles** — add `TileKind::AllReduce` after OProj
   and Down in the tile graph. Required for multi-GPU correctness.
   With TP, GEMM shapes change (sharded dims) — needs `models: runtime`.

7. **Multi-GPU cost CSVs** — run `gpu_cost_sweep` on A100, H100.
   Check in the CSVs. Add `TargetId` variants. The sweep is now
   data-driven (all 16 configs enumerated in a `bench_cutlass!` macro
   loop), so this is a matter of running it on new hardware.

8. **Elementwise + attention in CSV** — norm, SiLU, FlashInfer costs
   are still hardcoded `CostCurve`. Move them to the sweep.

9. **Split-K CUTLASS variants** — at some shapes (e.g. Down GEMM with
   K=8192 at M=64), cuBLAS's split-K strategy can outperform our serial-K
   `device::Gemm`. `cutlass::gemm::device::GemmSplitKParallel` exists
   and would slot into the existing `CUTLASS_GEMM` macro pattern.
   Currently CUTLASS wins everywhere anyway, so this is optional polish.

10. **Remove `megakernel!`** — dead code, replaced by `forward!`.

11. **ILP backend** — the `Solver` trait is ready; `solver::ilp` is a
    stub. Encode constraints as MILP when the CP solver proves
    intractable on richer libraries. **Not currently needed**: release-
    mode solve time is ~70ms per `forward!` expansion.

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
  (also has no backing kernel, so picks would fail anyway)
- TkFusedMlpBlockImpl: kernel is single-row;
  `workload_constraint() = NumTokensRange { min: 1, max: 1 }` prevents
  the solver from picking it at M > 1. Won't fire at M=1 either
  because individual CUTLASS + GEMV ops win. Kernel needs a real
  batch-aware rewrite to be useful.
- Elementwise/attention costs are rough estimates, not in CSV yet
- Multi-model (`models: [...]` with >1 entry) not yet implemented
  (uses first model only)
- `cutlass_256x128_s3` config exceeds sm89 99 KB SMEM limit and fails
  `can_implement` silently; excluded from the library (see
  `cutlass_standalone_gemm.cu` comment)
- Cost sweep runs kernels in tight warm-cache loops; real production
  cost may differ ~3-5% due to cold-cache effects. In practice the
  solver's picks still track real performance closely, but individual
  microbenchmark numbers should be treated as relative, not absolute.
- Binary still dynamically links `libcublas.so.12` + `libcublasLt.so.12`
  (816 MB combined). The Llama `forward!` path no longer calls cuBLAS
  (when the solver picks CUTLASS for every tile), but all non-Llama
  models and non-Ferrite Llama paths still route through
  `LinearLayer::Dense::forward()` → cuBLAS. See TODO #1.
- **Only Llama has a `forward!` DSL.** Every other architecture in
  `vllm-cuda/src/model/` uses the pre-Ferrite `LinearLayer`-based
  forward pass. Porting them is TODO #1 and gates binary-level
  cuBLAS removal.

## Environment notes

- `CUDA_PATH=/usr/local/cuda-12.9` required (system nvcc 12.0 breaks TK)
- After clearing cudaforge cache: `touch crates/vllm-cuda/csrc/*.cu`
  then `cargo build -p vllm-kernels-cuda --features cuda` to regenerate
- CUDA kernel compilation takes minutes (cicc); don't kill cargo
- CSV file is gitignored by default — use `git add -f` to check it in
