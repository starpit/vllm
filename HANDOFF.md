# ferrite-forward handoff

> **Read this first if you are picking this up cold.** Paired with
> `PLAN.md`. PLAN describes what to build; this describes what's
> built, what decisions were made along the way, the landmines a
> fresh context needs to know, and — most importantly — **the
> failure patterns that have repeatedly burned the user's time.**
>
> **Fast start**: most of the "Re-scoped path" + "Gap table"
> sections below are **historical** — they describe the state
> when the new compiler was scaffolding without kernels. Several
> of the "steps" are DONE. For current open work, jump to
> `## Next session starts here` (≈line 594). The historical
> sections are kept because their framing of the invariants and
> failure patterns is still load-bearing; file paths like
> `ferrite-solver/src/lowering/...` are **deleted** — recover via
> `git show 5b9fd90ef^:<path>` if you need them.

## TL;DR

- **Goal**: a forward-pass compiler. DSL body → solver picks
  kernels per tile → scheduler orders into wavefront LOOP → codegen
  emits Rust + CUDA. Replaces `ferrite-macros` + `ferrite-solver`
  (both now deleted; see commit `5b9fd90ef`).
- **End-state proof**: `timeout 60 vllm chat --model=<small-llama>`
  produces coherent output through the new compiler at perf parity
  with the existing path. Correctness is **proven** for SmolLM-135M,
  Qwen2.5-0.5B, Gemma2-2B, and Granite-3.3-2B
  (`test_cuda_correctness_smollm_135m`,
  `test_cuda_correctness_qwen2_0_5b`, `test_cuda_correctness_gemma2_2b`,
  `test_cuda_correctness_granite_3_3_2b` — all four pass via the
  `*Ferrite` paths, run via `cargo test -p vllm-e2e --features
  e2e,cuda --release --test e_correctness -- --ignored
  --test-threads=1`). Perf: cutlass tile zoo + CSV-driven selection
  landed in commit `b43bb85b9`.
- **Method**: port the working old ferrite verbatim, detoxifying
  as you port. Do not rewrite mechanisms that already work.
- **Current state, honest version**: scaffolding AND correctness
  both green. Dense-bf16 Llama, Qwen2, Gemma2, and Granite all route
  through ferrite via `CudaModel::{Llama,Qwen2,Gemma2,Granite}Ferrite`
  in `cuda_worker.rs`. Decode attention uses the
  `attention_decode_from_cache` wrapper (so span rotation + fp8 KV
  light up automatically); fused-QKV emit has a runtime
  `is_fp8()` branch. Backbone-only forward is emitted alongside
  the full forward for PP intermediate ranks. Quant / TP / PP /
  MoE / MLA models still route to the hand-written path for the
  reasons enumerated in the "Remaining gaps" section.
  The **library is still a tiny fraction of the old one**: 10
  Impls (after `FusedGemmBiasImpl` + `OpKind::BiasAdd` landed this
  session) vs. the old library's ~1,400 kernel variants. Cost
  model is analytical estimates vs. the old CSV lookup against
  55k rows of measured sweep data. See "Real gap inventory" below —
  those gaps are perf work, not correctness work.

## Real gap inventory (what this compiler is missing vs old ferrite)

Supersedes any earlier "library complete" language. Every gap
below blocks perf parity on `vllm bench`; starred ones also block
correctness or functionality.

| # | Gap | Old | New | Impact |
|---|---|---|---|---|
| 1 | **Cutlass tile zoo** | 424 `CutlassGemmImpl` + 424 w/Residual + 424 w/SiluMul + 128 `CutlassGemvImpl` ≈ 1,400 variants | 0 | Perf parity — DP has nothing to optimize over |
| 2 | **CSV cost tables** | `cost_l4_sm89.csv` (4k rows), `cost_l40s_sm89.csv` (16k), `cost_h100_sm90.csv` (32k); `GpuCostGrid` with `(M,N,K)` lookup + launch-overhead accounting | Analytical flops/bandwidth proxy | Solver picks blind |
| 3 | **GEMV specialization for M=1 decode**\* | 128 `CutlassGemvImpl` variants | 0 | Decode latency regresses |
| 4 | **SplitK / TB_K=64 variants** | 22 SplitK + 10 TB_K=64 per phase | 0 | L4 memory-bound shapes regress |
| 5 | **Qwen2 fused-QKV-with-bias** (LANDED) | `CublasFusedQkvGemmWithBiasImpl` | `FusedQkvRopeCacheImpl` + `FusedQkvRopePrefillImpl` matchers now walk through an optional `BiasAdd` wrapper on each rope input; claim grows 4→7 tiles when biased. `OpKind::BiasAdd` + `FusedGemmBiasImpl` for stray pairs. | Qwen2 DSL now says `bias_add` explicitly; solver absorbs it into the fused QKV kernel via cuBLAS `gemm_bias` epilog |
| 6 | **Cublas / cutlass gemm+residual fusion** | `CublasGemmExWithResidualImpl`, `CutlassGemmWithResidualImpl` (424 variants) | 0 | Extra launches vs. fused |
| 7 | **Cutlass gemm+silu+mul fusion** | `CutlassGemmSiluMulImpl` (424 variants) | `FusedGateUpSiluMulImpl` emits cuBLAS | Perf delta on SwiGLU MLP |
| 8 | **Attention variant split** (LANDED) | `TkAttentionDecodeImpl`, `TkAttentionPrefillImpl`, `FlashInferStandaloneImpl`, `FlashInferStandardImpl` | `AttentionViaCacheImpl` + `FusedQkvRopeCacheImpl` for decode (M=1); `AttentionPrefillContiguousImpl` + `FusedQkvRopePrefillImpl` for prefill (M≥2); solver picks per `WorkloadConstraint`. Done in Step D commit `16ef52925`. | No longer a split issue — but did not fix `vllm chat` garbage output on its own |
| 9 | **Prefill rope variant** | `VllmRsPrefillRopeCacheImpl` | 0 | Prefill KV-write path missing |
| 10 | **Scheduled megakernel**\* (**HIGH PRIORITY**) | Full BSP persistent cooperative-launch kernel system: `kernel_library.rs`, `schedule.rs`, `templates/scheduled/megakernel.cu`, `SCHEDULED_MEGAKERNEL_HANDOFF.md` (~41 KB) | Nothing | Entire megakernel codegen path absent |
| 11 | **Codegen: stream assignment, wave merging, cooperative-launch emission, graph capture** | All present in `ferrite-solver/src/lowering/backend/cuda_codegen.rs` | Sequential one-launch-per-subgraph only | DeviceCallable waves can't form megakernels |
| 12 | **Solver constraints** | `CoopExclusivity`, `RegBudget`, `ShmemBudget`, `LayoutCompatibility`, `HandoffCompatibility` (10 classes) | matches() + cost only | Solver will pick infeasible plans when megakernel lands |
| 13 | **Solver backends** | CP (backtrack) + DP + ILP stub | DP only | Large search spaces may need CP |
| 14 | **Concurrency model** | Ported but dormant — no DeviceCallable Impls exist, no CoopExclusivity constraint wired | — | Machinery present, unused |
| 15 | **Layout adapters** | Insert layout-conversion impls between mismatched producers/consumers | `Layout` enum exists; no adapter insertion | Mismatched layouts produce `UnclaimedTile` |
| 16 | **Test depth** | Structural golden codegen tests + real inference | 85 unit + 6 integration type-check tests | **No end-to-end correctness or perf test exists** |

Additional items not in the table (lower-risk):
- `OpKind` enum missing `BiasAdd` (blocks Qwen2 DSL), `Gelu`,
  `SoftCap`, `SlidingAttention` (block Gemma2).
- `ferrite-solver/data/*.csv` still present in the repo. Source of
  truth for gap #2. Do not regenerate — measured.

## Re-scoped path to vllm chat + parity

Priority-ordered. Each step is a port of a real thing the old code
does. See the table above for which gaps each step closes.

### Step A — Cutlass tile zoo + CSV cost backend

Closes gaps #1, #2, #3, #4. Port `ferrite-solver/src/lowering/cost_table.rs`
(the `GpuCostGrid`) into `ferrite-forward-macro/src/cost_table.rs`,
pointing at the existing CSVs. Add cutlass tile Impls starting
with `CutlassGemmImpl` (single Gemm), then `CutlassGemvImpl`. FFI
block lives in a new `ferrite-kernels::cutlass` module (today
duplicated in `vllm-cuda/src/model/llama.rs:3645`,
`vllm-cuda/src/model/qwen2.rs:254`, `ferrite-test-harness/src/lib.rs:79`).
Done when: `vllm bench` on Llama-3.2-1B hits within 5% of the old
path on L4.

### Step B — DeviceCallable Impls + fused-launch codegen (HIGH PRIORITY per user)

Today every Impl in the library is `LaunchKind::HostCallback`.
`LaunchKind::DeviceCallable` exists in the enum but is unused, and
`codegen.rs` emits one kernel launch per subgraph with no
awareness of the tag. This step makes DeviceCallable real: a wave
of DeviceCallable subgraphs becomes one fused GPU launch instead
of N sequential ones.

Closes gaps #10, #11, #12, #14. Concretely:

1. **Add one DeviceCallable Impl** (e.g. `DeviceCallableRmsNormImpl`).
   A wave containing DeviceCallable subgraphs is the minimal case
   that forces every other piece below to get real.
2. **Scheduler: compilation units within a wave.** Today `Wave =
   Vec<(SubgraphId, ImplId)>`; in the new shape, a wave is a list
   of compilation units, where a unit is one HostCallback subgraph
   *or* a contiguous run of DeviceCallable subgraphs bundled into
   one fused launch.
3. **Solver: wire `Resources` budget checks.** When the DP bundles
   DeviceCallables into a unit, the union of `Resources
   { registers_per_thread, shared_memory_bytes }` must fit the
   target SM's per-block limits. Today `Resources::ZERO` is
   returned by every Impl and nothing checks. This becomes a real
   constraint once DeviceCallables exist.
4. **Codegen: emit fused kernel launches.** Sub-decision to make
   with the first DeviceCallable Impl: (a) emit `.cu` at
   macro-expansion time, (b) NVRTC at runtime, or (c) a pre-
   compiled template kernel with a runtime dispatch table.
5. **Handoff:** `Handoff::Shmem` for intra-unit (DeviceCallable →
   DeviceCallable inside the same fused launch);
   `Handoff::StreamOrder` across unit boundaries.

Done when: a wave with two adjacent DeviceCallable Impls emits one
fused kernel launch, and `vllm bench` shows measurably fewer launch
overheads on decode.

### Step C — Fusion parity

Closes gaps #5, #6, #7, #9. Port `CublasFusedQkvGemmWithBiasImpl`
(unblocks Qwen2), `CublasGemmExWithResidualImpl` +
`CutlassGemmWithResidualImpl`, `CutlassGemmSiluMulImpl`,
`VllmRsPrefillRopeCacheImpl`.

### Step D — Attention variant split (DONE, commit `16ef52925`)

Decode vs prefill impls now split via `WorkloadConstraint`:
`FusedQkvRopeCacheImpl` + `AttentionViaCacheImpl` gated to M=1;
`FusedQkvRopePrefillImpl` (uses `fused_qkv_rope` + `write_kv_cache`)
+ `AttentionPrefillContiguousImpl` (uses `flash_attn_contiguous`
on contiguous K/V) for M≥2. Solver's DP picks per bucket.
Necessary but **not sufficient** for correctness — the SmolLM
golden still fails after Step D landed. Kept here for reference.

### Step E — Wiring to vllm-cuda + vllm chat (PARTIALLY DONE)

Dispatch wired up: vllm-executor routes dense-bf16 Llama to
`CudaModel::LlamaFerrite` which calls
`ferrite_models::llama::forward(&weights, &ctx, device,
num_tokens)`. The old `LlamaSolver`/`Qwen2Solver` variants are
deleted; old `ferrite_macros::forward!{}` in
`vllm-cuda/src/model/{llama,qwen2}.rs` removed. Qwen2 stays on
hand-written `Qwen2ForCausalLM` until Step C lands its bias-fused
QKV impl.

**Correctness is NOT yet achieved.** `vllm chat --model=<llama>`
runs through the ferrite path but produces garbage tokens (e.g.
`,,,,,,,,,`). `test_cuda_correctness_smollm_135m` fails.
`test_cuda_correctness_qwen2_0_5b` passes (Qwen2 still on
hand-written path). The failure is specific to the
ferrite-forward-emitted forward.

Diagnostic proof that scaffolding is fine: the uncommitted
`FERRITE_DISABLE=1` env-var gate (routes dense through the
hand-written `LlamaForCausalLM` path instead) makes the SmolLM
golden pass. So the bug is entirely in the ferrite-emitted
forward's semantics, not in the dispatch / KV cache / weight
loading / ForwardCtx wiring. Static analysis has been exhausted.

Next-session move: add per-layer goldens so the bug is
locatable. See "Next session starts here" below.

### Step F — Qwen2 + Gemma2 + Mixtral + DeepSeek-V2

Qwen2 unblocks after C (gap #5). Gemma2 needs `Gelu`, `SoftCap`,
`SlidingAttention` ops. Mixtral needs DSL extension for top-k
expert dispatch. DeepSeek-V2 needs `attention_mla`.

### Step G — Delete legacy

Once no `forward!{}` site remains: delete `ferrite-solver`,
`ferrite-macros`. **PRESERVE `ferrite-solver/data/*.csv`** — move
into `ferrite-forward-macro/data/` first. Measured, not cheaply
regenerable.

## Verify current state

```bash
cd /home/moosevan/vllm/.claude/worktrees/ferrite-forward/vllm-rs
cargo test    -p ferrite-forward -p ferrite-forward-macro
cargo fmt     -p ferrite-forward -p ferrite-forward-macro --check
cargo clippy  -p ferrite-forward -p ferrite-forward-macro --lib --tests -- -D warnings
cargo build   -p ferrite-forward --features cuda --tests
```

Expected state:
- **84 unit + 6 integration tests pass.**
- fmt + clippy clean (non-cuda).
- Cuda test build compiles the emitted Llama forward fn for every
  (model × workload bucket) pair — no more fake-symbol errors.

## What's actually done

### Foundation (in place, tested)

- **Parser** (`parse.rs`, `ast.rs`): syn-based parse of the DSL
  body. Rejects `let`. Recognizes assignments, tuple destructure,
  for-loops with literal or symbolic bounds, dotted weight paths
  with optional indexing, the `*` operator, op calls.
- **Classifier** (`classify.rs`, `classified.rs`): walks AST,
  classifies free vars as Extern (fixed enum: `InputIds, Positions,
  Rotary, BlockTable, KvCache`), Weight (interned `WeightId` per
  dotted path), or Local (SSA `LocalId` per assignment). `OpKind`
  is uniform — no `GemmQ/K/V` sub-variants.
- **Shape inference** (`shape.rs`): `Dim = Lit | Bound(name) | Mul
  | Var`; union-find over Vars; per-op shape signatures pin what
  each op's I/O shapes must be.
- **HF conventions** (`weight_conventions.rs`): standard HF
  weight-name → shape lookup; `derive_implicit_bounds` fills config
  defaults like `head_dim = hidden_size / num_attention_heads`.
- **Config loader** (`config.rs`): reads
  `model_architectures/<arch>/*.json`; captures every top-level
  integer field as a bound.
- **CFG + unroll** (`cfg.rs`, `fuf.rs`): per-model CFG with
  concrete `u64` trip counts, then unrolled to a flat
  `Fuf { nodes: Vec<FufNode> }`. Loop induction vars are resolved
  to concrete integers during unroll — they show up as the `index`
  field on `FufInput::Weight` and `FufInput::Extern`. No separate
  `layer_idx: u64` field on `FufNode` needed; impls read the layer
  off the `Extern { kind: KvCache, index: Some(L) }` input when
  they need it.
- **Solver** (`solver.rs`): DP ported from
  `ferrite-solver/src/lowering/solver/dp.rs`. Topo-forward greedy
  with `claimed[]` bitmap and multi-tile claim support. Candidate
  matches at each seed are sorted **claim-size DESC, cost ASC** —
  fusion is preferred whenever available because leaving a
  multi-tile claim's downstream tiles uncovered by picking a
  cheaper singleton at the seed is a correctness failure (no
  singleton Silu/Mul/Add-without-RmsNorm/RopeAppend impls exist),
  not a cost tradeoff. Unmatched tile → hard `SolveError::UnclaimedTile`;
  no auto-claim fallback. The DP's per-bucket `predicted_us` is
  overwritten post-scheduler by the contention-aware aggregator in
  `cost.rs`.
- **Concurrency model** (`concurrency.rs`): ported verbatim from
  old ferrite. Four rules: CooperativeLaunch exclusivity (coresident
  = `INFINITY`), compute+compute = 1.0 (serialize on tensor cores),
  memory shadows compute = 0.5, DeviceCallable megakernel stub =
  1.0. Consumes `launch_kind()` and `is_compute_bound()` off each
  Impl.
- **Cost aggregator** (`cost.rs`): `loop_cost_us` walks LOOP waves,
  applies `contention_factor(self, others_in_wave)` per Impl, sums.
  `refresh_predicted_us` overwrites each bucket's
  `Assignment.predicted_us` in place. On the current serial-chain
  library (every wave = 1 subgraph) contention is always 1.0 —
  machinery stays dormant until DeviceCallable + megakernel waves
  land.
- **Scheduler** (`schedule.rs`): topological wavefront over
  subgraphs. `Loop { waves: Vec<Wave> }` per workload point;
  `WorkloadLoops` keyed by `num_tokens`. Post-fusion the real
  Llama body is a serial chain (each subgraph depends on the
  previous), so `waves == subgraphs` — no wave merging. The old
  parallelism (Q/K/V gemms) now lives *inside* the fused subgraphs.
- **Implementation trait** (`impl_lib.rs`): name, target_compatible,
  workload_constraint, matches, cost_us, resources, launch_kind,
  supported_input/output_handoffs, input/output_layouts,
  is_compute_bound, can_share_kernel_with, **emit_call**,
  **required_weights**. Supporting types: `Resources, LaunchKind,
  Handoff, Layout (RowMajorBf16, ColMajorBf16, Any), MatchInfo,
  WorkloadConstraint, WeightAccessor`.
- **Impl-declares-weights mechanism** (`impl_lib.rs`): each Impl
  declares its required `WeightAccessor`s (`{ name, rust_type,
  source_weights }`). Default impl (`default_required_weights`)
  derives one accessor per weight input of each claimed tile, typed
  by the consuming op (Embed → `Embedding`, RmsNorm → `RmsNorm`,
  Gemm → `LinearLayer`). Fusion impls override to declare packed
  accessors (e.g. one `LinearLayer` covering three source weights
  for fused QKV). `emit_weight_bundle_trait` (`codegen.rs`) walks
  the SFUF across all buckets, aggregates declarations, dedupes by
  name, errors on rust_type collisions, emits the union as the
  `WeightBundle` trait. **The trait is a function of the solved
  graph, not the raw FUF.**
- **Macro pipeline drive** (`lib.rs`): `#[forward(models_dir,
  target, workloads)]` parses args, runs parse → classify →
  shape-infer → CFG → unroll → solve × workloads → schedule ×
  workloads, emits per-(arch, model) module containing pipeline
  observation constants AND codegen items.
- **Codegen module** (`codegen.rs`, `emit.rs`): walks the LOOP for
  each (model × workload bucket), emits `pub trait WeightBundle`,
  `pub unsafe fn forward_m_<N>(wm, ctx, device) -> OwnedTensor`,
  and a `pub unsafe fn forward(...)` dispatching on `num_tokens`.
  Per-subgraph emission delegates to
  `Implementation::emit_call(&EmitCtx) -> TokenStream`. `EmitCtx`
  carries `fuf, program, model, claimed_tiles, locals`; helpers:
  `primary`, `node`, `input_expr`, `all_input_exprs`,
  `input_tile_ident` (raw upstream ident for fusion impls that
  need to alias an OwnedTensor), `output_ident`, `weight_accessor`,
  `bound`.
- **`ForwardCtx`** (in `ferrite-forward` crate, cuda-gated):
  runtime args bundle the emitted forward takes (input_ids,
  positions, slot_mapping, cu_seqlens_q, seqused_k, block_table,
  max_seqlen_{q,k}, kv_cache: `&KvCachePool`, rotary:
  `&RotaryCache`).
- **Stable rebuild-on-JSON-change**: every config file the macro
  reads is registered via `const _: &str = include_str!("...")`
  in the emitted output. cargo rebuilds when JSONs change. No
  nightly, no build.rs.

### Impl library (**topology** coverage for Llama only)

> Not kernel-variant coverage. See "Real gap inventory" above —
> the old library has ~1,400 kernel variants, this has 7. Every
> DSL op in the Llama body has *an* Impl that claims it; that's
> all "topology coverage" means.

Singletons that emit real kernel calls:
- `EmbedRefImpl` → `kernels::embedding_gather`.
- `RmsNormRefImpl` → `kernels::rms_norm`. Fallback when the
  surrounding pattern isn't `(Add, RmsNorm)` — only fires for the
  first layer's `input_layernorm`, whose upstream is `embed` not
  `Add`.
- `GemmRefImpl` → `LinearLayer::forward`. Generic single-Gemm
  matcher, no phase tags. Fires for q/k/v/o/gate/up/down gemms
  that aren't claimed by a multi-tile fusion (in practice: the
  final `lm_head` gemm, and any Gemm that doesn't match the QKV
  or GateUp pattern).
- `AttentionViaCacheImpl` → `kernels::flash_attn_paged`. Claims
  `Attention` singleton-style but ignores the DSL tile's K/V
  Tile-inputs entirely; reads them from the paged cache
  (populated upstream by `FusedQkvRopeCacheImpl`). Layer index
  from the Attention tile's `Extern{kind:KvCache,index}`. Softmax
  scale = `1/sqrt(head_dim)` baked in at emit time; `is_causal =
  true` hardcoded for decoder-only.

Multi-tile fusions:
- `FusedGateUpSiluMulImpl` (4-tile). Claims `(Gemm, Gemm, Silu,
  Mul)` where two Gemms share the same activation, one feeds Silu,
  Silu feeds Mul, the other Gemm also feeds Mul. Emits fused cuBLAS
  GEMM → `silu_and_mul_fused`. Declares one packed `LinearLayer`
  accessor covering both source weights.
- `FusedAddRmsNormImpl` (2-tile). Claims `(Add, RmsNorm)` where
  the RmsNorm consumes the Add's output. Maps to
  `kernels::fused_add_rms_norm_inplace` — the kernel mutates
  `delta`'s buffer into the normed output and mutates `residual`'s
  buffer into the updated residual. Emit binds both tile outputs
  as `TensorView` aliases on the upstream OwnedTensors.
- `FusedQkvRopeCacheImpl` (4-tile). Claims `(Gemm, Gemm, Gemm,
  RopeAppend)` where the three Gemms share the same activation and
  feed the RopeAppend's first three Tile slots. Emits fused cuBLAS
  QKV GEMM → `kernels::fused_qkv_rope_cache` (applies RoPE, writes
  K/V to the paged cache at the matched layer, returns Q). Declares
  one packed `LinearLayer` accessor covering all three source
  weights. Binds RopeAppend slots 1/2 to layer-specific paged-cache
  TensorView aliases (unused — `AttentionViaCacheImpl` doesn't read
  its DSL K/V inputs).

Library registration (`starter_library()`):
1. EmbedRefImpl
2. RmsNormRefImpl
3. GemmRefImpl
4. AttentionViaCacheImpl
5. FusedGateUpSiluMulImpl
6. FusedAddRmsNormImpl
7. FusedQkvRopeCacheImpl

No singleton impls for Silu, Mul, Add, or RopeAppend — the only
kernel paths for those ops are the fusions above. A lone
Silu/Mul/Add/RopeAppend surfaces as `UnclaimedTile` (library gap
for future architectures; not a silent fallback).

### Llama/Qwen2 coverage

Per-layer tile budget in the unrolled FUF: 15 tiles (rmsnorm1, q/k/v
gemms, rope_append, attention, oproj, add1, rmsnorm2, gate_gemm,
silu, up_gemm, mul, down_gemm, add2).

Fusion savings per layer:
- SwiGLU → 1 subgraph (saves 3)
- QKV + rope → 1 (saves 3)
- Add1 + RmsNorm2 → 1 (saves 1)
- Add2 + next-layer input_layernorm (or final norm, on the last
  layer) → 1 (saves 1)

Total: **8 tiles saved per layer**. Subgraph count = `fuf.len() -
8*num_hidden_layers`. Every layer's 15 tiles collapse to 7 subgraphs.

Only one singleton RmsNorm survives (the first layer's
`input_layernorm`, whose upstream is `embed`). Everything else is
fused.

### WeightBundle emission

The emitted trait carries one method per unique accessor. For
Llama-3.2-1B:

```rust
pub trait WeightBundle {
    fn embed_tokens(&self) -> &Embedding;
    fn input_layernorm_0(&self) -> &RmsNorm;   // first layer only —
                                                // all others are fused
    fn lm_head(&self) -> &LinearLayer;
    fn norm(&self) -> &RmsNorm;
    fn mlp_down_proj_0(&self) -> &LinearLayer;
    // ... plus down_proj_1..num_layers-1
    fn post_attention_layernorm_0(&self) -> &RmsNorm;
    // ... plus post_attention_layernorm_1..etc.
    fn self_attn_o_proj_0(&self) -> &LinearLayer;
    // ... plus o_proj_1..etc.

    // Fused accessors (sorted join of source weight names):
    fn mlp_gate_proj_0__fused__mlp_up_proj_0(&self) -> &LinearLayer;
    fn self_attn_k_proj_0__fused__self_attn_q_proj_0__fused__self_attn_v_proj_0(
        &self,
    ) -> &LinearLayer;
    // ... per-layer
}
```

The caller implements this trait on whatever weight-holding struct
they want; the fused accessors must return a `LinearLayer` whose
`.weight` is the vertically-concatenated source weights
(`[gate|up]` for SwiGLU fusion; `[q|k|v]` for QKV fusion).

## Failure patterns from prior sessions (do not repeat)

These are **the actual mistakes that have eaten the user's time
across multiple sessions**. Catch yourself doing any of these
and stop.

### Adding speculative types/fields without consumers

> *"add a Layout enum / Constraint enum / Handoff enum, then we'll
> consume it later"*

Banned. The user's rule: **a commit is 100% complete if and only
if you will never have to revisit it except for bug fixes**. A
type with no live consumer will be reshaped the moment you write
the consumer, which means revisiting it, which means it wasn't
done. Land types **with their consumers**, not before.

### Inventing scaffolding instead of porting

> *"let me sketch the Impl trait with a few methods, then add
> more later"*

Banned. The user's rule: **the working old ferrite is the
exemplar**. Port verbatim, **detoxify as you go**, ship the whole
ported piece in one commit. Do not invent a smaller-than-old
interface and grow it.

### Stripping just names

> *"I'll rename `LlamaConfig` to `RopeConfig` and `Llama3RopeScaling`
> stays as a variant"*

Banned. **Llama3 anywhere in compiler code is bullshit, regardless
of whether you renamed the struct.** Detoxify means: remove
transformer-role taxonomies, remove weight-name substring matching,
remove per-layer/phase tags, remove arch-specific enum variants.
Not "rename and keep the field."

### Per-OpKind hardcoding in codegen

> *"the codegen has a `match op { OpKind::Embed => ..., OpKind::Gemm
> => ..., }` and it works for Llama"*

Not acceptable. The right shape: codegen calls
`impl.emit_call(&EmitCtx)` and each Impl emits its own kernel
call. New ops add new Impls to the library, not new arms in
codegen. **This is already in place.** Keep it that way.

### Inventing new OpKinds for fusions

> *"Qwen3 needs QK-norm, so I'll add `OpKind::QkNorm` (or
> `OpKind::QkNormRopeAppend`) to the DSL."*

Banned. **`OpKind` is the vocabulary of math primitives a model
author would write by hand** — `rmsnorm`, `gemm`, `rope_append`,
`attention`, `silu`, `gelu`, `add`, `mul`. If a proposed variant
bundles multiple math steps (`QkNorm` = rmsnorm on Q + rmsnorm
on K; `QkNormRopeAppend` = that plus RoPE plus cache write), it
is a **fusion name posing as a math op**. Fusions live in
`impl_lib.rs` as multi-tile `Implementation`s; the DSL stays
plain math.

The litmus test: **would a model paper / reference implementation
describe this as one operation?** If Qwen3's spec says "apply
RMS norm to Q and K per-head before RoPE," that is two rmsnorm
calls in the DSL — not a new op. The paper never says "qk_norm."

Mechanism:
- Keep the DSL body simple: the author writes `q = rmsnorm(q,
  q_norm_w); k = rmsnorm(k, k_norm_w);` as two ordinary rmsnorm
  tiles over the [..., head_dim] axis.
- Solver-side: add a multi-tile Impl (e.g.
  `FusedQkvQkNormRopeCacheImpl`) that claims the pattern
  `(Gemm, Gemm, Gemm, RmsNorm, RmsNorm, RopeAppend)` and emits
  the fused kernel.
- If a shape signature needs generalizing (e.g. `RmsNorm`
  admitting `[..., D]` with weight `[D]` for per-head norm),
  generalize the signature — don't spawn a new OpKind to dodge
  the shape-inference work.

Same rule applies backward: any existing *compound-sounding*
OpKind variant (`RopeAppend` couples RoPE with KV-cache write
today) is a historical compromise that should be revisited the
moment a model needs them decoupled (e.g. RoPE without cache
write). Don't compound them further.

### Treating `vllm-cuda/src/model/*.rs` as in-scope

Out of scope. **`vllm-cuda` is not touched.** The compiler replaces
only `ferrite-macros` + `ferrite-solver`. Its output plugs in at
the same call site the old ferrite output plugged in at. The only
vllm-cuda edit is a small shim to call the new signature (step 6).

### Treating the old ferrite as non-working

The old ferrite **runs Llama and Qwen2 correctly**. Its sin was
internal hardcoding, not non-functionality. Do not "rewrite" any
mechanism the old code already has — port it.

### Adding "TODO / will-do-later / for now" comments

These are bullshit markers. Either land the thing now or do not
land the surrounding piece. The only acceptable comment of this
shape is one that names the next concrete commit that will
consume the marked code.

### Claiming "100% done"

Only true once: (a) every consumer the new code needs to work
with is also in place and exercised, (b) tests observe the actual
end-user behavior (not just type-checks-and-fmt-clean), and (c)
nothing in the commit will need revisiting except for bugs. If
you're tempted to say 100% before all three hold, you're wrong.

### Runtime polymorphism for quant (recurring trap)

> *"add a `FusedLinear` enum that wraps `Packed(LinearLayer)` or
> `Split(Vec<LinearLayer>)`, with a runtime `load_auto(gw, prefixes,
> qconfig, stream)` that branches on qconfig"*

Banned. **Ferrite is a compiler.** The quant format of every weight
is known at macro-expansion time from the model's `config.json`
(`quantization_config`). Format decisions belong in the solver
(per-format Impls whose `matches()` inspects source-weight storage)
and in codegen (per-format `FieldLoad` arms emitting format-specific
loader calls). A runtime `match qconfig { ... }` inside a helper
erases what the compiler already knew, and forces the solver to
pick kernels blind to the storage format.

Additionally: **marlin is a compute kernel, not a storage format.**
Storage formats are `awq`, `gptq`, `fp8`, `bnb4bit`, `fp8_block` —
the shape of the bits on disk, fixed by the upstream HF repo.
Marlin is one of several possible kernels for 4-bit INT weights
(awq/gptq bits=4). Name Impls after the kernel they emit
(`MarlinGemmImpl`); match on the source weight's `StorageFormat`.

### Moving logic around without checking if it's really "bypassing"

When staging changes, don't reach for inflammatory framing
("bypassing", "hiding the decision") until you've checked whether
the pattern you're using matches existing ferrite code. Every
existing `FieldLoad` arm emits a call to a runtime helper
(`Embedding::load`, `LinearLayer::load_dense_concat`, …) — new quant
arms emitting `MarlinLinear::load_awq(...)` follow the same shape.
The user called this out once already; check before self-flagellating
in the next session.

## Library-gap guarantee

The library does NOT have singleton impls for Silu, Mul, Add, or
RopeAppend. The solver's behavior when any of those ops appears
without its required fusion partners:

```
SolveError::UnclaimedTile { tile, op, num_tokens }
```

This is **by design**. A new architecture whose DSL uses
standalone Silu/Mul/Add/RopeAppend (e.g. a ReLU-only model, or an
architecture with non-fused residual paths) will need a matching
Impl added to the library. The hard error points at the specific
tile that needs coverage.

## Repository layout

```
ferrite-forward/                    (worktree root)
├── PLAN.md                         the plan
├── HANDOFF.md                      this file
├── model_architectures/            arch-level per-model JSONs
│   ├── llama/                      9 Llama configs
│   └── qwen2/                      11 Qwen2/2.5 configs
├── target_profiles/                hardware metadata JSONs
│   ├── l4_sm89.json
│   └── h100_sm90.json
└── vllm-rs/crates/
    ├── ferrite-forward/            consumer crate (re-exports macro)
    │   ├── src/lib.rs              ForwardCtx (cuda-gated)
    │   └── tests/
    │       └── phase7_end_to_end.rs  6 integration tests incl. cuda
    └── ferrite-forward-macro/      proc-macro + all compiler logic
        └── src/
            ├── lib.rs              #[forward] entry point + drive
            ├── ast.rs              raw parse tree
            ├── parse.rs            syn → ast::Ast
            ├── classified.rs       classified IR types
            ├── classify.rs         ast → classified::Program
            ├── shape.rs            Dim, Shape, Solver, infer()
            ├── config.rs           ModelParams loader from dir
            ├── target.rs           TargetProfile loader
            ├── weight_conventions.rs  HF conventions
            ├── cfg.rs              classified → Cfg with u64 bounds
            ├── fuf.rs              Cfg → Fuf (numeric tile graph)
            ├── impl_lib.rs         Implementation trait + types +
            │                         all 7 Impls
            ├── solver.rs           DP solver, SFUF, WorkloadAssignments
            ├── schedule.rs         wavefront scheduler, Loop, Wave,
            │                         WorkloadLoops
            ├── concurrency.rs      ConcurrencyModel (ported verbatim)
            ├── cost.rs             contention-aware LOOP aggregator
            ├── emit.rs             EmitCtx + input_expr helpers
            └── codegen.rs          per-(model × workload) forward fn
                                      emitter; SFUF-walking
                                      WeightBundle trait emission
```

Legacy crates (the exemplars the current code was ported from)
are **deleted** as of commit `5b9fd90ef` (Step G). If you need to
look at the prior `forward!{}` / constraint-solver logic, recover
from git history:

```
git show 5b9fd90ef^:vllm-rs/crates/ferrite-solver/…
git show 5b9fd90ef^:vllm-rs/crates/ferrite-macros/…
git show 5b9fd90ef^:vllm-rs/crates/ferrite-test-harness/…
```

The live crates under `vllm-rs/crates/`:

```
ferrite-forward         consumer crate (re-exports #[forward],
                        ForwardCtx, cpu_golden)
ferrite-forward-macro   proc-macro + compiler logic (parse →
                        classify → shape → CFG → unroll → FUF →
                        DP solve → schedule → codegen)
ferrite-kernels         runtime kernel wrappers (cublas, cutlass
                        FFI, flash-attn, rms_norm, silu_and_mul,
                        rope, KV cache, LinearLayer, Embedding)
ferrite-cuda-core       GpuTensor / TensorView / OwnedTensor /
                        CachingAllocator / CUstream / GpuDevice
ferrite-cuda-builder    build pipeline for .cu sources (cached
                        under ~/.cudaforge)
ferrite-models          DSL bodies: llama.rs (`#[forward] fn llama`)
                        + qwen2.rs (`#[forward] fn qwen2`)
```

`ferrite-cuda-core/src/tensor.rs` carries an unsafe
`GpuTensor::as_view<'a>(&self) -> TensorView<'a>` method, used by
`emit.rs` to back the `(*#ident).as_view()` pattern.

## Commit history — `git log --oneline` on the worktree branch

Most recent session (AWQ quant — preparation only; no ferrite-forward
AWQ codegen yet, but the data model and the runtime loader are in
place so the next session can emit AWQ loads directly — see
"## Quantization path (in progress)" below):

```
974e9a4c8  ferrite-kernels: move AWQ→Marlin loader + helpers for codegen reuse
999076179  ferrite-forward: parse quantization_config, guard dense-only loader
```

Prior session (Granite port end-to-end: `scalar(<name>)` /
`recip_scalar(<name>)` DSL, `attention_multiplier` override,
Granite DSL body + 3 configs, GraniteFerrite cuda_worker wiring,
granite_3_3_2b golden):

```
2adc521a5  ferrite-forward: Granite end-to-end via scalar() / recip_scalar() DSL
```

Prior session (Gemma2 prompt-7 root-caused: missing BOS in
/v1/completions tokenization, not the suspected per-layer kernel
drift):

```
e2669d635  vllm-serve: prepend BOS by default on /v1/completions tokenization
```

Prior session (Gemma2 acid test: DSL `if/else`, config-driven
attention scalars, Gelu/TanhSoftCap/SlidingAttention Impls, Gemma2
DSL body + configs, Gemma `(1+w)` as DSL `w + 1.0`, vllm-executor
wiring, embed-scale + sliding-branch fix, gemma2_2b golden):

```
9bcec3611  testdata: gemma2_2b golden from Python vLLM
2e733be50  ferrite-forward: fix Gemma2 correctness — embed scale, sliding layer flip
031f7c9a9  vllm-executor: wire Gemma2Ferrite through cuda_worker
992acd9ad  ferrite-forward: Gemma `(1+w)` rmsnorm as DSL `w + 1.0`
b4887a38c  ferrite-forward: Gemma2 DSL body + configs + end-to-end tests
0494b62c5  ferrite-forward: Gelu, TanhSoftCap, SlidingAttention Impls
037d46034  ferrite-forward: config-driven attention scale + softcap
b7efd02e4  ferrite-forward: DSL `if`/`else` for compile-time layer-indexed dispatch
```

Prior session (correctness fix + Qwen2 migration + cutlass + fp8
branch + PP backbone + Step G delete):

```
838669d5f  relax dp_assigns test budget for cutlass zoo
89a408cd2  HANDOFF: landed items + trimmed remaining list
5b9fd90ef  delete legacy ferrite-macros/solver/test-harness (Step G)
9e608b256  fp8 KV-cache branch in FusedQkvRopeCacheImpl emit
b43bb85b9  cutlass GEMM zoo + CSV-driven kernel selection
67953f559  load empirical GPU cost CSVs into TargetProfile
4ad93dafa  emit forward_backbone for PP intermediate ranks
a01056410  decode attention via attention_decode_from_cache
2f661b40c  migrate Qwen2 to #[forward]
af07c6066  reshape attention output to 2D (SmolLM correctness fix)
```

## Next session starts here

**AWQ Llama/Qwen2 ferrite end-to-end landed + correctness golden
green.** Dense-bf16 + AWQ Llama/Qwen2 both route through
`#[forward]`-emitted code. The solver picks `Marlin*Impl` for any
Gemm whose weight resolves to `StorageFormat::Awq { .. }` via the
FUF-level annotation pass; the codegen-emitted `Weights::load` body
allocates one marlin workspace at the top when any AWQ accessor is
present, and the per-accessor FieldLoad routes to
`MarlinLinear::load_awq` / `load_awq_concat`.

**Five correctness goldens green via ferrite-forward.** Dense-bf16
SmolLM2-135M, Qwen2.5-0.5B, Gemma2-2B, Granite-3.3-2B plus AWQ
Llama-3.2-1B (`AMead10/Llama-3.2-1B-Instruct-AWQ`) all pass
`test_cuda_correctness_*`. Python-vLLM-generated golden JSONs under
`crates/vllm-e2e/testdata/golden/`.

### Marlin bug fix (Commit 4)

Two distinct bugs in the Rust-side Marlin wrappers, shared by AWQ
and GPTQ, hidden because the smoke tests only asserted non-empty
output:

1. **`c_tmp` was orders of magnitude undersized.** We allocated
   `[size_m, size_n]` floats; Python vLLM's `marlin.cu` reference
   allocates `sms * max_m_block_size * max_thread_n` floats. The
   kernel indexes C_tmp via `locks_off * c_size` where `locks_off`
   reaches `sms` (blockIdx.x) and `c_size = tb_m * tb_n` floats per
   lock slot. Undersized C_tmp corrupts reductions → NaN logits →
   `!!!!!!` output. Fix in `ferrite-kernels/src/kernels.rs` now
   queries SM count via `device_get_num_sm(device_id)` and sizes
   C_tmp to match Python's formula exactly (`max_thread_n = 256`
   from `marlin.cuh`).
2. **`pack_cols_4bit` / `unpack_cols_4bit` used a strided layout
   instead of Python's consecutive layout.** Our versions packed
   input cols `[p, p + cols/8, p + 2*(cols/8), …]` into output col
   `p`; Python's `quant_utils.{pack,unpack}_cols` pack/unpack
   consecutive cols `[8*p, 8*p + 1, …, 8*p + 7]`. AutoAWQ's on-disk
   `qzeros` are packed the Python way, so our unpack → undo-interleave
   → scale_perm → interleave → repack pipeline operated on
   consistently-wrong indices — every 8-wide permute touched the
   wrong 8 zero-points. Fix in `ferrite-kernels/src/layers_quant.rs`.

Both fixes are correctness-critical for any Marlin path (AWQ, GPTQ,
whatever 4-bit INT format lands next). After both, `vllm serve
Qwen/Qwen2.5-0.5B-Instruct-AWQ` produces coherent output
identical to Python vLLM (`" Paris. The capital of the United States
is Washington"` on the same prompt), and the AWQ Llama-3.2-1B
correctness golden passes end-to-end.

**Prior session: weight shapes now come from per-arch
`weights.json`; `weight_conventions.rs` is gone.** The old "global
HF convention" table masquerading as universal was in reality a
"dense-attn + SwiGLU LLM" table — only `embed_tokens` was truly
cross-arch; everything else breaks for MLA/MoE/hybrid archs. That
session replaced it with per-arch data, generated automatically:

- **`model_architectures/<arch>/weights.json`** — per-arch shape
  manifest. Keys are dotted weight paths (`self_attn.q_proj`,
  `self_attn.q_norm`); values are shape formulas in bound names
  from the arch's `config.json` (`["hidden_size",
  "head_dim * num_attention_heads"]`). Generated for the four
  existing arches: llama, qwen2, gemma2, granite.
- **`probe-weights` binary** (`ferrite-forward --features probe
  --bin probe-weights`) — bootstraps a new arch from HF. Downloads
  each size's `config.json`, range-fetches the first ~4 MB of each
  safetensors shard (header-only — the full multi-GB data is
  never downloaded), extracts weight shapes, and rewrites integer
  dims into bound-name products by cross-validating every formula
  against every listed size's config. Tied-embedding models
  (Gemma2, Granite, Qwen2.5-0.5B, SmolLM2) are detected
  structurally (`embed_tokens` present but no `lm_head`) and get a
  synthesized `lm_head` entry in compiler gemm order
  (`[hidden_size, vocab_size]`).
- **`weights_manifest.rs` + `shape.rs::infer`** — shape inference
  loads the per-arch manifest and anchors weights against it via
  **numerical-equivalence unification**. When structural unify
  fails (e.g. Qwen2.5's coincidence of `hidden_size == heads *
  head_dim`), both sides are evaluated against the first model's
  bounds; if they resolve to the same integer, accept. Genuine
  mismatches (Qwen3/Gemma3 per-head `q_norm` — declared
  `[head_dim]` vs inferred `[heads * head_dim]`) route through
  `detect_reshape_hint` which synthesizes `OpKind::Reshape` tiles.
- **`OpKind::Reshape` + `ReshapeRefImpl`** — metadata-only view op
  synthesized when the manifest declares an axis-factor mismatch.
  Emits `TensorView::reshape(&[...])` with no allocation or
  kernel launch.
- **`model_architectures/README.md`** — documents the per-arch
  recipe: `mkdir <arch>/` → commit upstream `config.json` per size
  → run `probe-weights --arch <arch> <repo-ids>` → write DSL body
  → wire into `cuda_worker.rs` → commit golden.
- **`weight_conventions.rs` deleted.** `derive_implicit_bounds`
  (head_dim/num_kv_heads defaulting) moved to `config.rs`.

**Prior session: Qwen2 bias is now first-class math in the DSL.**
Earlier sessions had Qwen2's QKV bias riding silently through
`LinearLayer::forward` — the DSL said `gemm()` but the runtime
quietly did `gemm_bias`, which is exactly the "hide math inside an
Impl" antipattern now documented in the "Inventing new OpKinds for
fusions" subsection of "Failure patterns." That session fixed it:

- **`OpKind::BiasAdd`** (new) — broadcast-add of `[D]` bias across
  `[..., D]` activation. `bias_add(x, b)` parses, shape-infers, and
  flows through classify → FUF like any other math primitive.
- **`GemmRefImpl`** — now emits **strict matmul** via
  `device.cublas.gemm(*(x), w.dense_weight(), alloc)`. Bias is no
  longer applied implicitly; `gemm()` in the DSL means matrix
  multiply, full stop.
- **`FusedGemmBiasImpl`** (new) — 2-tile fusion for `(Gemm,
  BiasAdd)` pairs. Emits `(#w).forward()` which dispatches to
  cuBLAS `gemm_bias` epilog. `debug_assert!` on
  `dense_bias().is_some()` so a DSL/safetensors mismatch panics
  loudly instead of silently skipping the bias.
- **`FusedQkvRopeCacheImpl` / `FusedQkvRopePrefillImpl`** — matcher
  now walks through an optional `BiasAdd` wrapper on each of the
  three rope inputs. Claim grows 4→7 tiles when biased. Emit is
  unchanged (packed `LinearLayer::forward` covers both cases); same
  `debug_assert!` guards.
- **`gemm_is_fusion_partner`** — extended to include `BiasAdd` so
  cutlass singletons don't strand a downstream bias.
- **`ferrite-models/src/qwen2.rs`** — rewritten with explicit
  `bias_add(q, self_attn.q_proj.bias[layer])` after each Q/K/V
  gemm. Mirrors the old `ferrite_macros::forward!` exemplar at
  `~/vllm/.claude/worktrees/claude4/vllm-rs/crates/ferrite-models/src/qwen2.rs`.
- **No singleton `BiasAddRefImpl`** — per library-gap guarantee, a
  lone `bias_add` surfaces as `UnclaimedTile`. Only fusion impls
  claim it.

Accessor names are unchanged (`source_weights` stays at the 3 Gemm
weights; bias rides through `load_dense_concat`'s auto-detect path
as a packed `LinearLayer` field). No changes to
`cuda_worker.rs::Qwen2Ferrite` needed. All four correctness
goldens still pass.

**Four correctness goldens green via ferrite-forward.** Dense-bf16
Llama (SmolLM2-135M), Qwen2 (Qwen2.5-0.5B), Gemma2 (Gemma2-2B), and
Granite (Granite-3.3-2B) all pass `test_cuda_correctness_*` through
their `*Ferrite` variants (verified by `loaded ... via ferrite-forward`
log lines on each; for Granite, also confirmed by diffing against
`FERRITE_DISABLE=1` — both paths produce identical output down to
the same tolerated prompt-1 position-2 top-N divergence).

**Prompt-7 root cause was NOT in the kernels.** The prior session's
"shared CUDA kernel drift" theory was wrong. Per-layer hidden-state
dumps comparing our hand-written Gemma2 path against HF transformers
eager showed `r ≥ 0.9998` agreement at every checkpoint — embed,
all 26 layers, final norm, lm_head, post-softcap. The actual bug:

- `crates/vllm-serve/src/engine.rs::tokenize_completion_prompts`
  hardcoded `add_special_tokens=false`, so `/v1/completions`
  tokenized prompts WITHOUT BOS (22 tokens for prompt 7 vs 23
  in the golden). Python vLLM defaults to `add_special_tokens=True`;
  the golden was generated that way, so we were silently comparing
  WITH-BOS reference output against WITHOUT-BOS engine output.
- For prompts 0-6 the BOS/no-BOS divergence stayed inside the
  top-20 tolerance window. Prompt 7's first-token prediction crossed
  the boundary: HF eager WITHOUT-BOS picked `'\n\n\n'`, golden
  (Python vLLM WITH-BOS) picked `'\n\n'`.
- Fix in commit `e2669d635`: added `add_special_tokens: bool`
  field to `CompletionRequest` defaulting to `true`, plumbed
  through to `tokenize_text(...)`. Direct struct constructors in
  `engine.rs` (test fixtures), `llm.rs` (in-process LLM API),
  and `spans.rs` (span query builder) updated to set `true`.

**Bonus pre-existing bug found, NOT yet fixed.** The chat path
(`crates/vllm-serve/src/engine.rs:2857`) does
`tok.encode(&text, true)` AFTER rendering a chat template that
literally emits `<bos>` (Gemma2) or `<|begin_of_text|>` (Llama3)
into the string — so chat completions have been double-BOS'ing
forever. Verified with HF tokenizer: Gemma2 chat template + true
encode gives `[2, 2, 106, ...]` (two BOS); + false gives `[2, 106, ...]`.
Python vLLM uses `add_special_tokens=False` for the chat path for
exactly this reason. Fix: flip line 2857 to `false`. Not done in
this session because every chat e2e test will measurably change
its first-token logits and need a wider validation pass.

`spans.rs:191, 324, 1449` correctly use `false` (template-rendered
text). `examples/src/chat.rs:51` and `examples/src/lib.rs:491`
also use `false`. So only the chat HTTP handler needs the fix.

### Prior correctness fix (kept for context)

The SmolLM correctness fix from the previous session remains.
Root cause was a 3D→2D reshape missing at the Attention output
before o_proj; `AttentionViaCacheImpl` and
`AttentionPrefillContiguousImpl` now reshape in-place via
`OwnedTensor::reshape`.

### Landed since the correctness fix

Items from the original gap list that are now done. Dropped from
the remaining-work section below; re-listed here so future
sessions don't re-port them.

1. **Qwen2 ferrite migration** — `ferrite-models/src/qwen2.rs`
   uses the same `#[forward]` DSL body as Llama plus explicit
   `bias_add(q, self_attn.q_proj.bias[layer])` tiles after each
   QKV gemm (same math the old `ferrite_macros::forward!` body
   encoded). Bias is first-class math in the DSL — not hidden
   inside `LinearLayer::forward`. The `FusedQkvRopeCache` /
   `FusedQkvRopePrefill` matchers walk through the BiasAdd
   wrappers so the packed cuBLAS `gemm_bias` epilog still runs in
   one launch. `CudaModel::Qwen2Ferrite` in `cuda_worker.rs`
   parallels `LlamaFerrite` and carries all the accessor / forward
   arms. Dense-bf16 Qwen2 now routes through ferrite; quant / TP /
   PP keep the hand-written path.
2. **Cutlass GEMM zoo + CSV costs** — `target_profiles/cost_*.csv`
   holds the three calibrated sweeps (L4/SM89, L40S/SM89,
   H100/SM90). `TargetProfile` auto-loads `cost_<name>.csv`
   alongside the JSON. `CutlassGemmImpl { tile_m, tile_n, stages }`
   is registered for every CUTLASS_TILE_ZOO entry plus
   `CutlassGemvImpl` for M=1. `target_compatible` filters by
   CSV-row presence so targets without data silently fall back to
   `GemmRefImpl`; `cost_gemm` (cuBLAS) now also consults the
   CSV so cutlass-vs-cublas comparisons are apples-to-apples.
   Matchers reject Gemms whose output feeds `RopeAppend`/`Silu`/
   `Mul` — picking cutlass for a QKV or gate/up gemm would
   orphan the fused downstream chain.
3. **Span rotation on decode attention** —
   `AttentionViaCacheImpl::emit_call` routes through
   `attention_helpers::attention_decode_from_cache`, which builds
   `cos_sin_cache_ptr` + `rotary_dim` from
   `kv_cache.block_unrotated_gpu()` and dispatches to
   `fp8_decode_attention` when `kv_cache.is_fp8()`.
4. **FP8 decode fused-QKV branch** —
   `FusedQkvRopeCacheImpl::emit_call` emits a runtime
   `if ctx.kv_cache.is_fp8() { fused_qkv_rope_cache_fp8(...) }
   else { fused_qkv_rope_cache(...) }` so fp8-KV/bf16-weight
   models work once `cuda_worker.rs` loosens its fp8 gate.
5. **PP backbone forward** — `codegen::emit_forward_backbone_for_bucket`
   emits a parallel `forward_backbone_m_<N>` per bucket that
   skips the terminal lm_head subgraph and DtoD-clones the
   penultimate tile's output into a fresh `OwnedTensor`.
   Arch-level `forward_backbone` dispatches over the Weights
   enum. `LlamaFerrite::hidden_states` /
   `Qwen2Ferrite::hidden_states` call it. Loader still gates
   ferrite off for `use_pp` — backbone is ready for when that
   flips.
6. **Delete legacy (Step G)** — `ferrite-macros`, `ferrite-solver`,
   `ferrite-test-harness` are gone (~73k lines deleted).
   `cpu_golden.rs` moved to `ferrite-forward/src/cpu_golden.rs`
   (rayon stripped; it's reference code, not a perf path).
   `target_profiles/cost_*.csv` preserves the calibrated sweeps.
7. **Gemma2 — full architecture port via DSL extensions.** The
   acid test passed for generality: the compiler stays arch-agnostic.
   Landed:
   - **Compile-time DSL conditional**: `if <ivar> % N == M { ... }
     else { ... }` / `if <ivar> < N { ... }`. Parses to a constrained
     `BoolExpr` (Modulo / Less with literal-or-sym bounds); folds at
     CFG-build to concrete `u64`s; evaluates per-iteration at unroll
     time; arms must bind the same set of names (merge_carry in the
     classifier). Also unblocks DeepSeek-V3's dense-prefix, Jamba
     hybrid-SSM layers, and any future architecture with non-uniform
     layers. See commit `b7efd02e4`.
   - **Config-driven attention params**: `ModelParams.scalars:
     BTreeMap<String, f64>` mirrors `bounds` for non-integer JSON
     fields. `AttentionViaCacheImpl`/`AttentionPrefillContiguousImpl`
     / their sliding variants read `query_pre_attn_scalar` (→
     softmax scale) and `attn_logit_softcapping` (→ softcap) per
     model; Llama/Qwen2 configs don't set them so their behavior is
     byte-identical to before. See commit `037d46034`.
   - **New OpKinds + Impls** (generic, capability-named):
     `Gelu` with `FusedGateUpGeluMulImpl` (mirrors SwiGLU fusion
     but calls `gelu_and_mul_fused`); `TanhSoftCap` with singleton
     Impl reading `final_logit_softcapping`; `SlidingAttention`
     (same shape sig as Attention) with `SlidingAttentionViaCacheImpl`
     / `SlidingAttentionPrefillContiguousImpl` passing
     `window_size_left` from `sliding_window` config. See commit
     `0494b62c5`.
   - **Gemma2 DSL body + configs**: `ferrite-models/src/gemma2.rs`
     with 4-norm layers, alternating sliding/full attention via
     `if layer % sliding_window_pattern == 0`, GELU MLP, final
     softcap. `model_architectures/gemma2/{2b,9b,27b}.json`. 2B
     = 551 tiles / 238 waves; 9B = 887/382; 27B = 971/418. See
     commits `b4887a38c`, `2e733be50`.
   - **Gemma `(1+w)` rmsnorm via DSL `w + 1.0`** (honest math,
     not a load-time hack). New `Expr::ScalarLit(f64)` / `Expr::Add`
     lowers to a scalar-Add tile fused into rmsnorm by two new
     Impls: `ScalarOffsetRmsNormImpl` (standalone) and
     `FusedAddRmsNormWithOffsetImpl` (3-tile residual-Add +
     scalar-Add + RmsNorm). Kernel gained `weight_offset: f32`
     parameter; `rms_norm_with_offset` / `fused_add_rms_norm_inplace_with_offset`
     are new thin wrappers — existing callers unchanged.
     Load-time weight mutation is NOT needed. See commit `992acd9ad`.
   - **Embed scale via DSL `sqrt(hidden_size)`**. New
     `Expr::SqrtBound(Ident)` parsed from `sqrt(<bound_name>)`,
     folded at CFG-build to a concrete `f64`. `Expr::Mul` admits
     tile×scalar; new `ScalarMulImpl` emits `scale_inplace`. Gemma2
     body: `hidden_states = embed(...) * sqrt(hidden_size)`.
   - **Gemma2Ferrite wiring**: `Gemma2FerriteModel` struct +
     `CudaModel::Gemma2Ferrite` variant + 7 match arms in
     `cuda_worker.rs`. Model-selector ferrite-gate identical to
     Llama/Qwen2. See commit `031f7c9a9`.
   - **Golden test**: `test_cuda_correctness_gemma2_2b` added;
     generator script extended. Golden JSON committed. Known-fail
     on prompt 7 per "Next session starts here". See commit
     `9bcec3611`.

### Remaining gaps vs. prior ferrite

Everything below is NOT blocked on another gap unless noted;
tackle in roughly this order.

1. **DeviceCallable / megakernel emission** — every library Impl
   is `LaunchKind::HostCallback` or `LaunchKind::RegularLaunch`.
   The compiler's codegen already dispatches on `LaunchKind` in
   principle, but there are no DeviceCallable Impls for the
   memory-bound tail (rmsnorm, silu, add, rope, elementwise) so
   nothing fuses into `__global__` megakernels. Port the device
   bodies + add the emission arm.
2. **GPTQ / FP8 / BnB quant plumbing** — analogous commits to the
   AWQ Commit 3 just landed. Each new format needs: parser arm in
   `quantization::QuantizationConfig::parse`, new `StorageFormat`
   variant (GPTQ shares Marlin with AWQ so it reuses `MarlinLinear`
   via a parallel loader), the corresponding `Marlin*Impl` /
   `Fp8*Impl` / `Bnb*Impl` matchers gated on that storage format,
   `FieldLoad` arm emitting the right loader, `is_*()` helper on
   `QuantConfig` + gate lift in `cuda_worker.rs`, a new u8 value
   assigned in `ferrite_forward_macro::quant_discriminator` +
   `ferrite_quant_kind`. The AWQ slice (Commit 3) is the template;
   each format lands as one atomic commit. GPTQ-Marlin will
   benefit automatically from the Commit 4 Marlin bug fixes.
3. **QK-norm** (Qwen3, Gemma3) — `qk_norm_inplace` +
   `rotary_embedding_q_only` path. Needs a `FusedQkvQkNormRopeImpl`.
4. **Gemma3** — another alternating-attention architecture, but
   with a 5:1 local/global ratio. Should reuse the DSL
   `if layer % sliding_window_pattern == 0 { ... }` construct
   now that it exists. Gemma3 also needs QK-norm (see gap #3)
   and different RoPE scaling. Diff should hit only
   `ferrite-models/src/gemma3.rs` + `model_architectures/gemma3/*.json`
   + any net-new kernels — NOT the compiler.
5. **MoE (Mixtral, Qwen2-MoE)** — needs a DSL construct for
   `for each of top-k experts run sub-body`. Generic language
   extension, not an MoE-specific branch.
6. **MLA (DeepSeek-V2)** — new `attention_mla` op + kernel +
   Shape signature; compiler stays out of MLA.
7. **PP weight-load plumbing** — ferrite's backbone forward is
   in place, but `cuda_worker.rs` still refuses to load
   `ferrite_models::<arch>::Weights` when `use_pp` (dense Weights
   type doesn't know about layer ranges). Extend `Weights::load`
   to accept a layer range so PP intermediate ranks can only
   materialise their owned layers.
8. **Chat-completion double-BOS fix** — `crates/vllm-serve/src/engine.rs:2857`
   does `tok.encode(&text, true)` after rendering a chat template
   that already emits `<bos>` / `<|begin_of_text|>` literally.
   Flip to `false`. Pre-existing bug, surfaced while diagnosing
   the prompt-7 completion-path BOS issue (commit `e2669d635`).
   Validation cost: every chat-completion e2e test will see its
   first-token logits shift, so plan for re-running goldens and
   possibly regenerating any chat-mode goldens that were tracking
   the double-BOS behavior.

### Already landed — do not redo

- Parser + classifier + shape inference + CFG/unroll.
- FUF/SFUF/LOOP IRs + `schedule_workloads` + `cost::refresh_predicted_us`.
- DP solver (ported from `ferrite-solver/src/lowering/solver/dp.rs`).
- Codegen delegates to `Implementation::emit_call`.
- Arch-level `Weights` enum + `Weights::load` + `forward` dispatcher
  emission. Caller integration = two calls (`load` at startup,
  `forward` per step).
- `ForwardCtx` runtime-args bundle in `ferrite-forward`.
- `include_str!` rebuild-on-change for every JSON the macro reads.
- `#[forward]` derives `models_dir` from the carrier fn name by
  walking up for `model_architectures/<name>/`.
- Tied-embedding load (lm_head reuses embed_tokens when
  `tie_word_embeddings: true`).
- Range-coalesced dispatch: `forward` accepts any `num_tokens` by
  mapping to the nearest-lower compiled bucket.
- Step D: decode/prefill attention variant split via
  `WorkloadConstraint`.
- vllm-executor wiring: `CudaModel::LlamaFerrite` variant,
  `LlamaForCausalLM::Model`-via-`forward!{}` and
  `Qwen2ForCausalLM::Model`-via-`forward!{}` invocations deleted,
  `ferrite_macros` dep dropped from vllm-cuda.
- SmolLM2-135M config in `model_architectures/llama/`.
- Dense-bf16 Llama + Qwen2 both on the ferrite path (LlamaFerrite
  / Qwen2Ferrite variants). Quant / TP / PP still hand-written.
- SmolLM-135M ferrite correctness: attention output 3D→2D reshape
  in `AttentionViaCacheImpl` / `AttentionPrefillContiguousImpl`.
- Cutlass GEMM zoo (16 tile variants + gemv) with CSV-backed cost
  comparison against CSV-anchored cuBLAS. `target_profiles/cost_*.csv`.
- Decode attention: `attention_decode_from_cache` wrapper so
  span-rotation + fp8-KV branches light up automatically.
- FP8-KV fused-QKV branch in `FusedQkvRopeCacheImpl::emit_call`.
- Arch-level `forward_backbone` dispatcher emitted alongside
  `forward`; per-bucket `forward_backbone_m_<N>` skips the
  terminal lm_head subgraph. `cuda_worker.rs::hidden_states` uses
  it for both Llama and Qwen2 ferrite variants.
- Legacy crates deleted: `ferrite-macros`, `ferrite-solver`,
  `ferrite-test-harness` (Step G). `cpu_golden.rs` preserved
  under `ferrite-forward/src/cpu_golden.rs`.
- DSL compile-time conditionals: `if ivar % N == M { ... }` /
  `if ivar < N { ... }`. Constrained predicate enum, folded
  per-iteration at unroll time. Both arms must bind the same
  names (merge-carry in classifier).
- DSL `Expr::ScalarLit(f64)` + `Expr::Add` / `Expr::Mul` admitting
  tile×scalar. New impls `ScalarOffsetRmsNormImpl`,
  `FusedAddRmsNormWithOffsetImpl`, `ScalarMulImpl`.
- DSL `sqrt(<bound_name>)` folds at CFG-build to a concrete f64
  scalar. Used by Gemma2's `embed(...) * sqrt(hidden_size)`.
- `ModelParams.scalars: BTreeMap<String, f64>` captures every
  non-integer number in config.json. `EmitCtx::scalar("key")`
  reads them; Impls read by HF convention name.
- Attention scale/softcap: `AttentionViaCacheImpl`,
  `AttentionPrefillContiguousImpl`, and both sliding variants
  read `query_pre_attn_scalar` / `attn_logit_softcapping` via
  the scalars map. Llama/Qwen2 (no such fields) fall back to
  `1/sqrt(head_dim)` / `0.0` — byte-identical prior behavior.
- Gemma2 op Impls: `OpKind::Gelu` + `FusedGateUpGeluMulImpl`;
  `OpKind::TanhSoftCap` + singleton Impl; `OpKind::SlidingAttention`
  + `SlidingAttentionViaCacheImpl` / `SlidingAttentionPrefillContiguousImpl`.
- `weight_offset: f32` added to `rms_norm_kernel` / `fused_add_rms_norm_kernel`
  in `crates/vllm-cuda/csrc/layernorm_kernels.cu`. New Rust wrappers
  `rms_norm_with_offset` / `fused_add_rms_norm_inplace_with_offset`;
  existing `rms_norm` / `fused_add_rms_norm_inplace` signatures
  unchanged (zero-offset wrappers). Gemma `(1+w)` flows in via
  these.
- Gemma2 DSL body (`ferrite-models/src/gemma2.rs`) and configs
  (`model_architectures/gemma2/{2b,9b,27b}.json`). All three
  sizes compile end-to-end through `#[forward]`.
- `CudaModel::Gemma2Ferrite` variant in `cuda_worker.rs` with
  seven match arms and model-selector ferrite-gate. Dense-bf16
  Gemma2 now routes through ferrite; quant / TP / PP keep the
  hand-written path.
- DSL `scalar(<name>)` / `recip_scalar(<name>)` folds at CFG-build
  to `ScalarLit` using `ModelParams.scalars` (non-integer config.json
  fields). `recip` form returns `1.0 / value`. Parallels `SqrtBound`
  but reads the scalars map instead of bounds. Used by Granite for
  embedding / residual / logits multipliers.
- `attention_scale_for()` prong: if the model config has
  `attention_multiplier`, use it as the direct softmax scale (no
  transform). Priority: Granite's `attention_multiplier` →
  Gemma2's `query_pre_attn_scalar.powf(-0.5)` → fallback
  `1/sqrt(head_dim)`. Llama/Qwen2/Gemma2 behavior unchanged (none
  carry `attention_multiplier`).
- Granite DSL body (`ferrite-models/src/granite.rs`) and configs
  (`model_architectures/granite/granite-3.{1-2b,1-8b,3-2b}-instruct.json`).
  Llama math + embed×embedding_multiplier, oproj/down×residual_multiplier
  before each residual add, logits×recip_scalar(logits_scaling).
  All three compile to 685 tiles / 365 waves; configs verified
  against upstream HF `config.json` verbatim.
- `CudaModel::GraniteFerrite` variant in `cuda_worker.rs` with
  seven match arms. Model-selector ferrite-gate on the existing
  `GraniteForCausalLM` arm — quant / TP / PP keep the hand-written
  Granite (which still sits inside the `LlamaForCausalLM` branch
  with post-load multiplier assignment).
- `test_cuda_correctness_granite_3_3_2b` + `granite_3_3_2b.json`
  golden. Ferrite and `FERRITE_DISABLE=1` paths produce identical
  output down to the same tolerated prompt-1 position-2 top-N
  divergence.
- **AWQ Llama + Qwen2 (Commit 3)** — compiler-native quant slice.
  `FufInput::Weight { id, index, storage: StorageFormat }` carries
  per-weight format; `Fuf::annotate_storage_formats(program, model)`
  populates it after unroll using the HF `quantization_config` +
  `tie_word_embeddings` + the Gemm-only-consumer rule. Dense impls
  (`GemmRefImpl`, `CutlassGemmImpl`, `CutlassGemvImpl`,
  `FusedGateUpSiluMulImpl`, `FusedGateUpGeluMulImpl`,
  `FusedQkvRopeCacheImpl`, `FusedQkvRopePrefillImpl`) gate
  `matches()` to reject non-Dense storage; new `MarlinGemmImpl`,
  `MarlinFusedGateUpSiluMulImpl`, `MarlinFusedQkvRopeCacheImpl`,
  `MarlinFusedQkvRopePrefillImpl` claim Awq storage instead and
  declare `rust_type = MarlinLinear`. Codegen adds
  `FieldLoad::AwqLinear { prefix, group_size }` +
  `FieldLoad::AwqLinearConcat { prefixes, group_size }` routed by
  accessor rust_type; the `Weights::load` body emits a one-shot
  `alloc_marlin_workspace` + `current_device` + `device_get_num_sm`
  prelude when any AWQ accessor is present. Arch-level dispatcher's
  match tuple gains `quant_kind: u8` (Dense=0, Awq=1) so
  dense/AWQ Llama-3.2-1B no longer share a fingerprint — see
  `ferrite_forward_macro::quant_discriminator` and
  `cuda_worker::ferrite_quant_kind`. `cuda_worker.rs` gate lifted
  to `!is_quantized() || is_awq()` for LlamaFerrite + Qwen2Ferrite.
  `model_architectures/llama/llama-3.2-1b-awq.json` +
  `model_architectures/qwen2/qwen2.5-0.5b-awq.json` committed.
  Existing `test_cuda_marlin_awq_{server_starts,completion,chat}`
  now route through the ferrite path (cuda_worker logs
  `loaded Qwen2 via ferrite-forward`) and all three pass.
  `testdata/golden/llama_3_2_1b_awq.json` committed
  (Python-vLLM-generated) but the correctness golden test is NOT
  added yet — blocked on the shared Marlin numerical bug (AWQ and
  GPTQ both produce garbage on this branch; `FERRITE_DISABLE=1`
  reproduces the same garbage, confirming it's pre-existing). The
  golden JSON is byte-for-byte reusable once marlin is fixed.

## Quantization path (in progress)

AWQ Llama / Qwen2 slice landed (see "Landed since the correctness
fix" above). Remaining AWQ work is the correctness golden test,
which is blocked on the Marlin numerical bug in gap #2. GPTQ /
FP8 / BnB / FP8-block still need their own commits — the AWQ
slice's shape (`StorageFormat::<Format>`, `<Format>*Impl`,
`FieldLoad::<Format>Linear`, u8 discriminator slot, `is_<format>()`
helper, gate lift) is the template.

### Core principle

**Ferrite is a compiler. Every quant decision is made at macro-
expansion time.** The proc-macro reads each model's `config.json`
at build time; `quantization_config` is known then. Per-weight
storage format flows through the solver via format-aware Impls
(per-format `matches()` predicate, per-format `rust_type` on the
accessor, per-format `emit_call`) and through codegen via
per-format `FieldLoad` arms that emit format-specific loader
calls. No runtime `match qconfig` on load. No `FusedLinear` enum.
No `load_auto` helper. Marlin is a compute kernel, not a format
— the storage format is `Awq { bits, group_size, zero_point,
version }` (or Gptq / Fp8 / …); marlin is one of several possible
kernels that accept 4-bit INT storage.

The emitted `Weights::load(gw, stream)` signature stays unchanged
from the dense version — if a given model has AWQ accessors, the
generated body allocates the marlin workspace inline:

```rust
pub fn load(gw, stream) -> Result<Self> {
    let __device_id = /* driver query */;
    let __num_sm = /* driver query */;
    let __marlin_ws = alloc_marlin_workspace(__num_sm, stream)?;
    // … per-weight lets, mixing:
    let embed_tokens = Embedding::load(gw, "model.embed_tokens")?;
    let q_proj_0 = MarlinLinear::load_awq(
        gw, "model.layers.0.self_attn.q_proj",
        128, __marlin_ws, __device_id,
    )?;
    let lm_head = LinearLayer::load_dense(gw, "lm_head")?;  // modules_to_not_convert
    // …
}
```

### What's landed (this session)

1. **`999076179` — `quantization_config` parser + codegen guard.**
   - `ferrite-forward-macro::quantization` module:
     - `StorageFormat { Dense, Awq { bits, group_size, zero_point,
       version } }`, `AwqVersion { Gemm, Gemv, Marlin }`.
     - `QuantizationConfig::parse(&serde_json::Value)` handles
       HF's `quant_method: "awq"` shape; unknown methods hard-error
       via `UnsupportedMethod(_)`. Honors `modules_to_not_convert`
       with suffix match.
     - `storage_format_for_weight(program, id, model)` resolves
       per-weight format on demand.
   - `ModelParams.quantization: Option<QuantizationConfig>` parsed
     in `config::load_file`.
   - Consumer: `codegen::emit_weights_struct` hard-errors via
     `compile_error!` if any accessor's source weights are non-
     `Dense` — forces future quant to ship with a matching Impl +
     FieldLoad arm, not slip through.
2. **`974e9a4c8` — AWQ→Marlin loader moved to `ferrite-kernels`.**
   - New `ferrite-kernels/src/layers_quant.rs`:
     - `alloc_marlin_workspace`, scale/zero-point permutes,
       `concat_cpu_dim1`, `fuse_bias_parts`, all previously in
       `vllm-cuda::weights_quant` / `::quant`.
     - `MarlinLinear::load_awq(gw, prefix, group_size, workspace,
       device_id)` — single-weight AWQ load + marlin repack.
     - `MarlinLinear::load_awq_concat(gw, prefixes, group_size,
       workspace, device_id)` — CPU-concat fused load (AWQ
       metadata *can* be concatenated at load time; the three
       QKV weights collapse into one packed `MarlinLinear`,
       same shape the dense fused accessor carries).
   - `vllm-cuda::weights_quant::load_awq_marlin_linear` /
     `load_fused_awq_marlin` are now 2-line adapters that unwrap
     the runtime `AwqConfig` and delegate. Hand-written Marlin
     AWQ smoke tests on Qwen2.5-0.5B-AWQ (`test_cuda_marlin_awq_*`
     in `e1_basic_serving.rs`) pass unchanged.

### Commit 3 — landed

See the "Landed since the correctness fix" bullet describing the
AWQ Llama + Qwen2 end-to-end slice for exactly what shipped and
what's still blocked (the correctness golden test, waiting on the
shared Marlin numerical bug). The design notes below stay current.

### Pitfalls to dodge (from this session's back-and-forth)

- **Don't add a `FusedLinear` enum.** We removed the one I added
  in this session's first attempt. See the "Runtime polymorphism
  for quant" failure pattern above.
- **Don't thread workspace / device_id through `Weights::load`.**
  They're emitted inline by codegen. The arch-level dispatcher
  gained one new arg (`quant_kind: u8`) so dense and AWQ variants
  of the same shape can be disambiguated at runtime — that's the
  *only* signature extension, and it's format-agnostic.
- **Don't invent an arch-level "quant config" arg.** Per-model
  storage formats are already in each `ModelParams`.
- **Don't re-fork the marlin repack logic.** The single source of
  truth is `ferrite-kernels::layers_quant::MarlinLinear::load_awq{,_concat}`.
  Hand-written and ferrite-emitted paths both call it.
- **AWQ fused-QKV is a single `MarlinLinear` (not three).** AWQ
  metadata (scales, zeros, qweight) all concatenate along dim N
  before the marlin repack; the fused accessor is one `MarlinLinear`
  exactly like the dense fused accessor is one `LinearLayer`. The
  "Split" shape I initially proposed is wrong for AWQ — it's
  relevant for FP8/BnB where per-tensor scales can't be merged,
  but those are later commits.
- **Verify `MarlinLinear::forward`'s arg list before emitting.**
  It's `(x, alloc, stream)` — no cublas handle. The dense
  equivalent `LinearLayer::forward` takes `(x, cublas, alloc,
  stream)`. Per-format emit means per-format signatures; that's
  the whole point.

### Follow-ons (new commits, not Commit 3)

- **GPTQ-marlin.** Extend the parser with `quant_method: "gptq"`,
  add a `StorageFormat::Gptq` variant, add `MarlinLinear::load_gptq{,_concat}`
  in `ferrite-kernels::layers_quant` (mirroring the current AWQ
  pair — GPTQ adds `g_idx` / `desc_act` handling per the GPTQ
  loader still in `vllm-cuda::weights_quant`). Same Impls accept
  GPTQ storage — Marlin doesn't care which INT4 format fed it
  after repack. Keep impls distinct per storage format so the
  solver can pick cost-aware in the future.
- **FP8.** `MarlinLinear::forward` isn't the path — FP8 uses
  `Fp8Linear::forward` which calls `cutlass_scaled_mm`. The FP8
  fused-QKV case is the one where `FusedLinear::Split` was
  actually applicable — per-tensor weight scales can't be merged,
  so three separate `Fp8Linear` calls + one `concat_dim1` is the
  shape. Codegen-native: the QKV FieldLoad emits three separate
  `Fp8Linear::load` lines + the fused Impl's `emit_call` emits
  three `Fp8Linear::forward` calls followed by `concat_dim1`.
  No `FusedLinear` enum needed — the generated code inlines the
  pattern.
- **BnB4bit.** Similar to FP8 split pattern. Shared dequant
  scratch buffer needs a top-of-body emit (like the marlin
  workspace).
- **FP8-block.** Per-block scales — same split pattern as FP8,
  different kernel.

## Last note

The user has been burned by prior sessions' patterns of "small
incremental scaffolding that compounds into bullshit." If you
catch yourself proposing types-first-consumers-later, hardcoding
per-OpKind, or adding `todo!()`/placeholder values, **stop and
reread the failure-patterns section above**. The user's rule is
not negotiable: every commit complete, no revisit except for bugs.
The right unit of work is the smallest **port-with-its-consumer**
that runs end-to-end, not the smallest type definition.
