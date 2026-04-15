# ferrite-forward handoff

> **Read this first if you are picking this up cold.** Paired with
> `PLAN.md`. PLAN describes what to build; this describes what's
> built, what decisions were made along the way, the landmines a
> fresh context needs to know, and — most importantly — **the
> failure patterns that have repeatedly burned the user's time.**

## TL;DR

- **Goal**: a forward-pass compiler. DSL body → solver picks
  kernels per tile → scheduler orders into wavefront LOOP → codegen
  emits Rust + CUDA. Replaces `ferrite-macros` + `ferrite-solver`.
- **End-state proof**: `timeout 60 vllm chat --model=<small-llama>`
  produces coherent output through the new compiler at perf parity
  with the existing path. **Perf parity is load-bearing — without
  the kernel-variant port in the gap table below, the new path is
  strictly slower than old ferrite.**
- **Method**: port the working old ferrite verbatim, detoxifying
  as you port. Do not rewrite mechanisms that already work.
- **Current state, honest version**: the scaffolding is in place
  (parser, classifier, shape-inference, CFG/unroll, DP solver,
  scheduler, codegen, `Weights` struct + `load` emission,
  arch-level dispatcher, decode/prefill attention variant split
  via `WorkloadConstraint`). vllm-executor routes dense bf16
  Llama through `ferrite_models::llama::forward` unconditionally —
  the old `LlamaSolver`/`Qwen2Solver` paths are deleted, and the
  legacy `ferrite_macros::forward!{}` invocations in
  `vllm-cuda/src/model/{llama,qwen2}.rs` are gone.
  **But: `vllm chat` on a Llama model runs and produces garbage
  tokens (`,,,,,,,`)** — `test_cuda_correctness_smollm_135m`
  fails. Scaffolding is NOT the problem: with `FERRITE_DISABLE=1`
  (a diagnostic toggle landed in commit `624beb4c8`) the hand-written
  `LlamaForCausalLM` path passes the same golden. The bug is
  somewhere in the ferrite-forward-emitted forward, not yet
  located. Static analysis has been exhausted — next-session
  move is granular per-layer goldens.
  The **library is still a tiny fraction of the old one**: 9
  Impls vs. the old library's ~1,400 kernel variants. Cost model
  is analytical estimates vs. the old CSV lookup against 55k
  rows of measured sweep data. See "Real gap inventory" below —
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
| 5 | **Qwen2 fused-QKV-with-bias**\* | `CublasFusedQkvGemmWithBiasImpl` | 0 | Qwen2 DSL migration blocked |
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

Old ferrite (the exemplar to port from) is at:

```
vllm-rs/crates/ferrite-solver/      ports source — 13k lines
vllm-rs/crates/ferrite-macros/      old proc macro (forward!{})
vllm-rs/crates/ferrite-kernels/     stays — runtime kernel wrappers
vllm-rs/crates/ferrite-cuda-builder/  stays — build pipeline
```

`vllm-cuda/src/model/llama.rs:3667` has a `ferrite_macros::forward!{}`
invocation today — that is what step 5 replaces with `#[forward]`.

`ferrite-cuda-core/src/tensor.rs` gained an unsafe
`GpuTensor::as_view<'a>(&self) -> TensorView<'a>` method (C4) to
make the `emit_input` pattern `(*#ident).as_view()` actually
compile — the method was aspirational before.

## Commit history (current branch, most recent first)

```
0cf1165fe  ferrite-forward: port ConcurrencyModel + contention-aware cost aggregator
2d3600a37  HANDOFF: record C1-C4 port (cuda build green; library complete)
6249c2b27  ferrite-forward: port AttentionViaCacheImpl; cuda build green
bacb93277  ferrite-forward: port FusedQkvRopeCacheImpl; delete RopeAppendRefImpl
179f6a911  ferrite-forward: port FusedAddRmsNormImpl; delete AddRefImpl
834144f44  ferrite-forward: Impl-declares-weights + port FusedGateUpSiluMulImpl
dcd8a90ed  HANDOFF: record 3/8 impls wired + multi-tile fusion plan
96b265d69  ferrite-forward: wire emit_embed / emit_rmsnorm to real ferrite-kernels symbols
4b22e722d  HANDOFF: update for emit_call refactor (75aa11312)
75aa11312  ferrite-forward: move codegen emission onto Implementation trait
987242340  ferrite-forward: emit forward fn (codegen module) — non-cuda complete
dd5e98259  ferrite-forward: port Implementation trait + library surface from old ferrite
... [phases 0-8 below this]
```

## Next session starts here

**The critical-path task: locate and fix the ferrite-path
correctness bug.** Scaffolding + framework design are done; the
ferrite-emitted forward produces garbage tokens end-to-end for
Llama. Static analysis has been exhausted; need per-layer diff
data.

### Proven debug anchor

`test_cuda_correctness_qwen2_0_5b` passes (hand-written path
works). `test_cuda_correctness_smollm_135m` fails with degenerate
output (produces `,,,,,,,`). With the diagnostic
`FERRITE_DISABLE=1` env-var gate in `cuda_worker.rs`, dense Llama
routes through `LlamaForCausalLM::load/forward` instead of
`LlamaFerrite` — SmolLM golden then passes. Bug is 100%
ferrite-path-specific.

### Concrete path to the bug: RESTORE the golden harness

**Goldens existed and worked in the prior ferrite.** They were
deleted with the megakernel retirement, not superseded. Do not
reinvent them — recover and adapt.

What was there (commit-archaeological references):
- **`crates/vllm-cuda/src/bin/gen_golden.rs`** (301 lines,
  deleted in `a13577f75`). Binary that runs hand-written
  `LlamaForCausalLM` on a fixed prompt, dumps top-k last-token
  logits to JSON. Recover with `git show
  a13577f75^:vllm-rs/crates/vllm-cuda/src/bin/gen_golden.rs`.
- **`crates/vllm-tk-test-harness/tests/op_tests.rs`** (2266
  lines, deleted in `a13577f75`). Per-op GPU-vs-CPU-golden tests.
  The pattern to copy — for each library Impl, launch its kernel
  on known bf16 inputs on GPU, call the matching `cpu_golden.rs`
  fn on CPU, diff. Recoverable the same way.
- **`cpu_golden.rs`** — pure-Rust reference per op. Still alive
  in `crates/ferrite-solver/src/cpu_golden.rs` (will move when
  Step G deletes `ferrite-solver`; move it to `ferrite-forward`
  first).
- **Commit `28a9b3acf`** ("feat(tk): add op-level test harness
  with CPU golden comparisons") documents the original design
  decisions.

What to rebuild, adapted to ferrite-forward's idiom:
1. Recover `gen_golden.rs` as-is initially. Extend output to
   include per-layer hidden-state stats (not just final logits).
   A standalone binary trips over cublasLt heuristic for some
   shapes — run it inside the live worker via a new
   `DUMP_GOLDEN=<path>` env var on `LlamaForCausalLM::forward`
   so the full worker preamble (cublas handles, streams, KV
   cache pool) is present.
2. Ferrite side: extend the macro to emit a
   `forward_with_snapshots(wm, ctx, device, num_tokens,
   snapshots: &mut Vec<(TileId, Vec<f32>)>)` parallel to
   `forward` — each emitted subgraph gets a DtoH dump of its
   boundary-output tensor(s) after the kernel call. Gate behind
   a feature/const so `forward` stays lean.
3. Diff harness (following `op_tests.rs` pattern at the layer
   granularity): loads the golden, calls
   `forward_with_snapshots`, diffs per-layer, reports first
   diverging tile with max-abs-diff.
4. Fix the bug. Confirm the new per-layer test passes AND
   `test_cuda_correctness_smollm_135m` passes.

### Perf + feature work (after correctness)

Per the gap table: Step A (cutlass tile zoo + CSV costs), Step C
(fusion parity incl. Qwen2 bias-fused QKV), Step B (DeviceCallable
fused launches), Step F (Gemma2 / Mixtral / DeepSeek-V2), Step G
(delete legacy `ferrite-solver`/`ferrite-macros`, preserving
`ferrite-solver/data/*.csv`).

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
- Qwen2 stays on hand-written path (ferrite migration blocked on
  Step C's `CublasFusedQkvGemmWithBiasImpl`).

## Last note

The user has been burned by prior sessions' patterns of "small
incremental scaffolding that compounds into bullshit." If you
catch yourself proposing types-first-consumers-later, hardcoding
per-OpKind, or adding `todo!()`/placeholder values, **stop and
reread the failure-patterns section above**. The user's rule is
not negotiable: every commit complete, no revisit except for bugs.
The right unit of work is the smallest **port-with-its-consumer**
that runs end-to-end, not the smallest type definition.
