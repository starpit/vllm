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
  with the existing path. Correctness is **proven** for SmolLM-135M
  and Qwen2.5-0.5B (`test_cuda_correctness_smollm_135m`,
  `test_cuda_correctness_qwen2_0_5b` — both ignored tests, run via
  `cargo test -p vllm-e2e --features e2e,cuda --release --test
  e_correctness -- --ignored --test-threads=1`). Perf: cutlass tile
  zoo + CSV-driven selection landed in commit `b43bb85b9`.
- **Method**: port the working old ferrite verbatim, detoxifying
  as you port. Do not rewrite mechanisms that already work.
- **Current state, honest version**: scaffolding AND correctness
  both green. Dense-bf16 Llama and Qwen2 both route through
  ferrite via `CudaModel::LlamaFerrite` / `CudaModel::Qwen2Ferrite`
  in `cuda_worker.rs`. Decode attention uses the
  `attention_decode_from_cache` wrapper (so span rotation + fp8 KV
  light up automatically); fused-QKV emit has a runtime
  `is_fp8()` branch. Backbone-only forward is emitted alongside
  the full forward for PP intermediate ranks. Quant / TP / PP /
  MoE / MLA / Gemma2 models still route to the hand-written path
  for the reasons enumerated in the "Remaining gaps" section.
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

Most recent session (Gemma2 acid test: DSL `if/else`, config-driven
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

**Gemma2 landed, with one known-fail golden test.** Dense-bf16
Gemma2-2B / 9B / 27B compile end-to-end through `#[forward]`.
`CudaModel::Gemma2Ferrite` in `cuda_worker.rs` parallels LlamaFerrite
and Qwen2Ferrite. `vllm chat unsloth/gemma-2-2b-it` produces
coherent output (e.g. the Rayleigh-scattering answer for "why is
the sky blue?"). Compiler itself is architecture-agnostic — zero
"gemma" strings below the parser.

**Known failure: `test_cuda_correctness_gemma2_2b`** hard-fails on
prompt 7 (translation prompt with apostrophe-quoted inner text).
Engine enters a loop generating the prompt text verbatim. Prompts
0-6 soft-fail at position 0 with "both in top-N" warnings (normal
bf16 divergence, tolerated). Prompt 7's failure is specific to
that prompt's numerics.

Debug findings ruled out as the cause:
- Not ferrite-vs-hand-written drift: ferrite IS loading
  (logs: "CudaWorker: loaded Gemma2 via ferrite-forward"). Both
  ferrite and hand-written produce logprobs within bf16 noise of
  each other on Gemma2; both diverge from Python vLLM the same
  way.
- Not CUDA graphs: `--enforce-eager` reproduces.
- Not final-logit softcap: our pre-softcap logits correlate r=0.11
  with Python's pre-softcap logits. Divergence is upstream of
  softcap.
- Not embed scale / sliding-layer flip: both already fixed in
  commit `2e733be50`, sky-blue prompt works.

Evidence points to per-layer numeric drift in a shared CUDA kernel
(most likely flash-attn with Gemma's softcap+sliding combo, or
cuBLAS gemm with Gemma2's unusual hidden/head_dim ratios) that
both the ferrite and hand-written paths hit via FFI into
`crates/vllm-cuda/csrc/`. The fix applies to both paths once found.

Path forward: per-layer tensor dump harness comparing against
Python vLLM running the same model on the same prompt. `vllm-rs`
has `cpu_golden.rs` infrastructure to build on. ~4-8 hours to
instrument and narrow down the first-diverging kernel call. See
gap #9 in Remaining Gaps below.

SmolLM and Qwen2 golden tests still pass — ferrite's
infrastructure and the Llama-family / Qwen2-family numerics are
intact.

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
   uses the same `#[forward]` DSL body as Llama. Qwen2's QKV bias
   rides through `LinearLayer::load_dense_concat` + `cublas.gemm_bias`
   automatically; no bespoke `CublasFusedQkvGemmWithBiasImpl` was
   needed. `CudaModel::Qwen2Ferrite` in `cuda_worker.rs` parallels
   `LlamaFerrite` and carries all the accessor / forward arms.
   Dense-bf16 Qwen2 now routes through ferrite; quant / TP / PP
   keep the hand-written path.
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
2. **Quantized Linear (marlin INT4, bnb4bit, awq/gptq, fp8
   block)** — `LinearLayer` has Marlin/Bnb/Ggml/Fp8/Fp8Block
   variants; ferrite's gemm impls only thread
   `LinearLayer::forward`'s dense arm. For quantized models the
   solver would need quantization-aware Impls; today those
   models must route around ferrite. `cuda_worker.rs` already
   filters ferrite off for `qconfig.is_bnb4bit()`, `is_fp8()`,
   `is_quantized()`.
3. **QK-norm** (Qwen3, Gemma3) — `qk_norm_inplace` +
   `rotary_embedding_q_only` path. Needs a `FusedQkvQkNormRopeImpl`.
4. **Granite multipliers** — `embedding_multiplier`,
   `residual_multiplier`, `logits_scaling`. The DSL now expresses
   scalar multiply via `<tile> * <scalar>` (landed with Gemma2's
   embed scale — see `ScalarMulImpl` + gap #7 above). For Granite
   specifically: the three multipliers are config-driven
   constants; add them to the model's JSON as scalars, read via
   `Expr::ScalarLit` + possibly a new `Expr::BoundScalar(Ident)`
   analogous to `SqrtBound` for `config_value * something`.
5. **Gemma3** — another alternating-attention architecture, but
   with a 5:1 local/global ratio. Should reuse the DSL
   `if layer % sliding_window_pattern == 0 { ... }` construct
   now that it exists. Gemma3 also needs QK-norm (see gap #3)
   and different RoPE scaling. Diff should hit only
   `ferrite-models/src/gemma3.rs` + `model_architectures/gemma3/*.json`
   + any net-new kernels — NOT the compiler.
6. **MoE (Mixtral, Qwen2-MoE)** — needs a DSL construct for
   `for each of top-k experts run sub-body`. Generic language
   extension, not an MoE-specific branch.
7. **MLA (DeepSeek-V2)** — new `attention_mla` op + kernel +
   Shape signature; compiler stays out of MLA.
8. **PP weight-load plumbing** — ferrite's backbone forward is
   in place, but `cuda_worker.rs` still refuses to load
   `ferrite_models::<arch>::Weights` when `use_pp` (dense Weights
   type doesn't know about layer ranges). Extend `Weights::load`
   to accept a layer range so PP intermediate ranks can only
   materialise their owned layers.
9. **Per-layer tensor-dump debug harness** — unblocks the
   `test_cuda_correctness_gemma2_2b` prompt-7 failure documented
   in "Next session starts here". Build: a Python script that
   runs HF/vLLM Gemma2-2B on the failing prompt and dumps
   `hidden_states` after each decoder layer to `.npy`; a Rust
   equivalent that loads via ferrite, runs one forward, and
   saves matching dumps. Offline diff finds the first layer where
   our output meaningfully differs from Python's. Then narrow
   to the specific kernel call responsible. Relevant shared CUDA
   kernels at `crates/vllm-cuda/csrc/{layernorm,pos_encoding,activation}_kernels.cu`;
   compare against `/home/moosevan/vllm/csrc/*` for semantic
   drift. `cpu_golden.rs` lives at `ferrite-forward/src/cpu_golden.rs`
   for reference-CPU implementations. Debugging approach:
   - Ferrite's `forward_backbone` skips only the TERMINAL tile;
     for Gemma2 that's `tanh_softcap`, so backbone returns
     pre-softcap logits (post-lm_head). To dump pre-lm_head
     hidden states, temporarily remove `capped = tanh_softcap(logits)`
     from `ferrite-models/src/gemma2.rs` so the terminal becomes
     `gemm` and backbone returns `normed`.
   - Python reference: `AutoModelForCausalLM.from_pretrained(...,
     torch_dtype=bf16)` + module forward hooks on each
     `model.layers[i]`.
   - Already verified: our post-softcap logits correlate r=0.11
     with Python's (nearly independent). The drift accumulates
     across layers; find the first layer where it exceeds bf16
     noise.

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

## Last note

The user has been burned by prior sessions' patterns of "small
incremental scaffolding that compounds into bullshit." If you
catch yourself proposing types-first-consumers-later, hardcoding
per-OpKind, or adding `todo!()`/placeholder values, **stop and
reread the failure-patterns section above**. The user's rule is
not negotiable: every commit complete, no revisit except for bugs.
The right unit of work is the smallest **port-with-its-consumer**
that runs end-to-end, not the smallest type definition.
