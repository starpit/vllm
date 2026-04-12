# CP5 Constraint Solver — Complete Handoff

## What this is

A **compiler** that discovers optimal kernel execution plans for LLM
inference. Given a model DAG (parsed from a DSL), a library of kernel
implementations (cuBLAS, CUTLASS, FlashInfer, vllm-rs), and a target
GPU cost table (measured via sweep), the solver produces per-workload
execution plans that minimize predicted wall-clock time.

The `forward!` proc macro is the entry point. It runs the solver at
compile time and emits everything needed for a model forward pass:

- `struct Layer` / `struct RuntimeDims` / `struct Model` — derived from
  the DAG + solver fusion decisions (not hand-written)
- `Model::load()` — loads weights from safetensors by HF path names,
  with fusion (QKV concat, gate+up concat) generated from the plan
- `Model::forward()` — full `input_ids → logits` pipeline with
  `last_token_indices` gather and solver-dispatched lm_head
- Per-bucket dispatch functions with solver-selected kernels

**Result**: The entire dense Llama forward pass — struct, load, forward —
is generated from `forward!()`. cuda_worker calls `m.forward(...)` in
one line.

## Rules of the road

**These are critical. Violating them produces garbage or regressions.**

1. **Everything is DAG-driven.** The DSL describes the math. The solver
   optimizes. The codegen walks the solver's `DispatchSequence` entry by
   entry. Struct fields come from the DAG (adjusted by solver fusion
   decisions). Load code comes from the plan + DAG buffer names. Nothing
   is hardcoded to a specific model or shape.

2. **DSL idents = HF weight paths.** The DSL uses dotted paths that match
   the actual safetensors weight names: `self_attn.q_proj`, `mlp.gate_proj`,
   `input_layernorm`, etc. The loader constructs full paths from these
   names. No mapping tables.

3. **Fusion is a solver decision, not a DSL decision.** The DSL has
   separate `gemm(normed, self_attn.q_proj[layer])` / `gemm(normed,
   self_attn.k_proj[layer])` / `gemm(normed, self_attn.v_proj[layer])`.
   The solver sees three GEMMs from the same input and elects to fuse
   them into one (`FusedQkvGemm`). Same for gate+up. The struct reflects
   the solver's decision: fused → one field, unfused → separate fields.

4. **Every DispatchEntry field matters.** `fused_residual`, `gemm_phase`,
   `kind`, `is_attn_norm` — the codegen must respect all of them.

5. **TensorView for async GPU reads.** Functions that take GPU buffers
   read asynchronously by CUTLASS kernels must take `TensorView<'_>`
   (borrow), not `OwnedTensor` (move). This prevents the caching
   allocator from freeing the buffer while the kernel is still reading.
   `solver_forward_lm_head` takes `TensorView` for this reason.

6. **Decode vs prefill are separate implementations.** The solver
   produces different plans for `seq_len=1` (decode) and `seq_len>1`
   (prefill). No runtime branching in the generated code.

7. **Test-driven.** Golden tests (`e_correctness`) verify numerical
   correctness against HF Transformers references. Run them after
   EVERY codegen change. Structural codegen tests verify field names,
   bucket phases, solver fusion decisions.

8. **Benchmark ≠ correctness.** `vllm bench latency` won't catch wrong
   logits. Always verify with golden tests or `vllm chat`.

## The `forward!` DSL

```rust
vllm_tk_macros::forward! {
    hidden_states = embed(input_ids, embed_tokens);
    for layer in 0..NL {
        let normed = rmsnorm(hidden_states, input_layernorm[layer]);
        let q = gemm(normed, self_attn.q_proj[layer]);
        let k = gemm(normed, self_attn.k_proj[layer]);
        let v = gemm(normed, self_attn.v_proj[layer]);
        let (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
        let attn = attention_decode(q, k, v, kv_cache[layer], block_table);
        hidden_states = gemm_add(attn, self_attn.o_proj[layer], hidden_states);

        let normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
        let gate = silu(gemm(normed2, mlp.gate_proj[layer]));
        let up = gemm(normed2, mlp.up_proj[layer]);
        hidden_states = gemm_add(gate * up, mlp.down_proj[layer], hidden_states);
    }
    hidden_states = rmsnorm(hidden_states, norm);
    logits = gemm(hidden_states, lm_head);

    models: [
        { layers: 28, hidden: 3072, intermediate: 8192,
          heads: 24, kv_heads: 8, head_dim: 128, vocab: 128256 },
    ],
    target: l4_sm89,
    workloads: [1..4096],
}
```

Key design points:
- **DSL idents are HF weight paths**: `self_attn.q_proj` maps to
  `model.layers.{i}.self_attn.q_proj.weight` in safetensors.
- **Separate Q/K/V GEMMs**: the solver fuses them (one GEMM with
  concatenated weight) — this is an optimization, not a DSL concern.
- **`qkv_bias: true`** on model entries (Qwen2) inserts BiasAdd tiles
  after Q/K/V GEMMs. Same DSL body, different plan.
- **`embed` + post-loop `rmsnorm` + `logits = gemm`**: the DSL covers
  the full pipeline. `Model::forward` handles everything.

## Generated output

The `forward!` macro generates (all in the calling module's scope):

| Generated item | Purpose |
|----------------|---------|
| `struct Layer { ... }` | Per-layer weights. Fields from DAG + solver fusion. |
| `struct RuntimeDims { ... }` | Per-forward dims (q_size, kv_size, etc.) from config. |
| `struct Model { layers, dims, embed_tokens, norm, rotary, lm_head }` | Top-level model. |
| `impl Model { fn load() }` | Loads weights by HF path. Fused loads generated from plan. |
| `impl Model { fn forward() }` | Full `input_ids → logits` with `last_token_indices`. |
| `solver_hidden_states()` | Backbone: embed → layers → final norm. |
| `solver_forward_layer()` | Per-layer dispatch (num_tokens → bucket → kernels). |
| `solver_forward_lm_head()` | lm_head dispatch (takes TensorView, not OwnedTensor). |
| `solver_layer_bucket_N()` | Per-bucket kernel sequence. |
| `solver_lm_head_bucket_N()` | Per-bucket lm_head kernel. |

## Pipeline

```
forward! DSL body (HF weight paths, separate Q/K/V)
    │
    ▼
parse.rs → ModelDag (typed DAG, dotted idents)
    │
    ▼
tile_graph.rs → TileGraph (GemmQ, GemmK, GemmV, BiasAdd, Embed, ...)
    │
    ▼
solver → PlanFamily (per-workload plans, fusion decisions)
    │
    ├─→ extract_weight_fields (DAG + fusion → struct fields)
    │       └─→ emit_structs (Layer, RuntimeDims, Model)
    │
    ├─→ emit_model_load (plan + DAG paths → load code)
    │       └─→ fused: take_into at offsets
    │       └─→ unfused: Linear::load / RmsNorm::load
    │
    ├─→ emit per-bucket functions (solver_layer_bucket_N)
    │
    ├─→ emit lm_head dispatch (solver_forward_lm_head, TensorView)
    │
    └─→ emit Model::forward (hidden_states + gather + lm_head)
```

## cuda_worker integration

```rust
enum CudaModel {
    Llama(LlamaForCausalLM),           // legacy: quant/TP/PP
    LlamaSolver(llama::Model),          // generated: dense safetensors
    Qwen2(Qwen2ForCausalLM),           // legacy (Qwen2 forward!() coming)
    // ... other archs
}

// Dense Llama load:
let (model, _lm_head) = llama::Model::load(&mut weights, &config, dtype, device)?;
CudaModel::LlamaSolver(model)

// Forward:
Self::LlamaSolver(m) => m.forward(input_ids, ..., last_token_indices, device),
```

The legacy `CudaModel::Llama(LlamaForCausalLM)` path remains for
quant/TP/PP until those are ported. It uses the hand-written
`LlamaDecoderLayer::forward` (cuBLAS). The two paths are independent.

## Solver fusion

The solver recognizes fusable GEMM patterns and claims multi-tile
subgraphs:

| Pattern | Impl | Tiles claimed | Result |
|---------|------|---------------|--------|
| Q+K+V from same input | `CublasFusedQkvGemmImpl` | {GemmQ, GemmK, GemmV} | One `self_attn_qkv_proj` field |
| gate+up from same input | `CublasFusedGateUpGemmImpl` | {GemmGate, GemmUp} | One `mlp_gate_up_proj` field |
| QKV + bias (Qwen2) | `CublasGemmExWithBiasImpl` | {GemmQ, BiasAdd} | Fused GEMM+bias |

Struct generation runs AFTER the solver. If the solver fuses Q+K+V,
the Layer struct has `self_attn_qkv_proj: LinearLayer` instead of three
separate fields. The loader generates concatenation code for fused
fields.

## Adding a new model architecture

1. Write the `forward!()` DSL body using the model's HF weight paths
2. Add `models: [{ ... }]` entries with the architecture's dims
3. If the architecture has new ops (MoE routing, sliding window),
   add `OpKind` variants to `dag.rs`, `TileKind` variants to
   `tile_graph.rs`, and `Implementation` entries to `library.rs`
4. Add a `CudaModel::NewArchSolver(new_arch::Model)` variant
5. Run golden tests

For Llama-family architectures with `qkv_bias` differences (Qwen2),
just change the `models:` entries — the bias handling is automatic.

## Key files

| File | Purpose |
|------|---------|
| **Solver core** | |
| `lowering/tile_graph.rs` | TileGraph, TileKind (GemmQ/K/V, BiasAdd, Embed), ModelDims |
| `lowering/library.rs` | Implementation entries + fused impls (QKV, gate+up) |
| `lowering/solver/backtrack_cp.rs` | B&B CP solver |
| `lowering/implementation.rs` | Implementation trait, MatchInfo |
| **Codegen backend** | |
| `lowering/backend/compile_dsl.rs` | ForwardDef parser (qkv_bias support) |
| `lowering/backend/dispatch.rs` | DispatchSequence, FusedQkvGemm, FusedGateUpGemm |
| `lowering/backend/codegen.rs` | Struct gen, Model::load gen, Model::forward gen, bucket fns |
| `lowering/backend/codegen_test.rs` | Structural + fusion + golden tests |
| **Runtime integration** | |
| `vllm-cuda/src/model/llama.rs` | `forward!()` invocation + CUTLASS FFI |
| `vllm-executor/src/cuda_worker.rs` | `CudaModel::LlamaSolver(Model)` |
| `vllm-cuda/src/layers.rs` | LinearLayer, Linear, shallow_clone |
| `vllm-cuda/csrc/cutlass_standalone_gemm.cu` | CUTLASS GEMM grid + GEMV |
| **Tests** | |
| `vllm-e2e/tests/e_correctness.rs` | Golden logprob tests (SmolLM, Qwen2) |
| `vllm-e2e/testdata/golden/` | HF Transformers reference logprobs |

## Known issues

- `Model::load` returns `(Model, LinearLayer)` — the separate lm_head
  is redundant since Model already has it as a field. Cleanup TODO.
- Mixed gate+up fusion across buckets: some buckets may not fuse at
  certain M values due to cost model. Struct uses fused field if ANY
  bucket fuses. Cost model calibration needed.
- `clippy::possible_missing_comma` lint triggered by generated code.
  Suppressed via `-A` flag.
- CutlassNormGemmImpl: no backing kernel yet.
- Binary still links libcublas.so.12 for non-Llama models.
- Only Llama has a `forward!()` DSL. Qwen2 is next.

## What's next (priority order)

1. **Qwen2 `forward!()`** — same DSL body as Llama, `qkv_bias: true`
   in models entries. BiasAdd infrastructure already exists. Should be
   minimal work.

2. **Other architectures** — each needs its own DSL body + any new
   op/tile kinds. MoE, sliding window, etc.

3. **`models: runtime`** — solver at startup, any model dims without
   recompiling.

4. **TP tiles** — `TileKind::AllReduce` after OProj and Down.

5. **Multi-GPU cost CSVs** — run `gpu_cost_sweep` on A100, H100.

6. **Cleanup** — remove `Model::load` returning separate lm_head,
   remove legacy `from_llama` bridge if still present, remove
   `_num_tokens` in legacy `LlamaModel::forward`.

## How to run

```bash
# Solver + codegen tests (no GPU needed):
cargo test -p vllm-tk-macros-core

# Golden correctness tests (needs GPU):
cargo test -p vllm-e2e --features e2e,cuda --release --test e_correctness \
  -- --ignored --test-threads=1

# Build + chat test:
cargo build -p vllm-cli --features cuda --release
timeout 60 target/release/vllm chat -m unsloth/Llama-3.2-1B-Instruct \
  --prompt "The capital of France is" --max-tokens 20
```
