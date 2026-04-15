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
  vllm-cuda is **untouched**; the compiler's output plugs in at
  the same call site the old ferrite output plugged in at.
- **End-state proof**: `timeout 60 vllm chat --model=<small-llama>`
  produces coherent output through the new compiler at perf parity
  with the existing path. Then Qwen2.
- **Method**: port the working old ferrite (`ferrite-solver/`,
  `ferrite-macros/`, `ferrite-kernels/`) verbatim, **detoxifying
  as you port** — strip transformer-domain hardcoding, do not
  rewrite mechanisms that already work.
- **Current state**: all four fusion Impls ported. The library
  covers every tile in realistic Llama bodies (Embed, RmsNorm,
  Gemm, plus FusedGateUpSiluMul, FusedAddRmsNorm, FusedQkvRopeCache,
  AttentionViaCache). `cargo build -p ferrite-forward --features
  cuda --tests` is **green** — the emitted Llama forward fn
  type-checks across all 9 configs × 5 workload buckets.
- **Branch**: `worktree-ferrite-forward`. **HEAD**: `6249c2b27`.

## Verify current state

```bash
cd /home/moosevan/vllm/.claude/worktrees/ferrite-forward/vllm-rs
cargo test    -p ferrite-forward -p ferrite-forward-macro
cargo fmt     -p ferrite-forward -p ferrite-forward-macro --check
cargo clippy  -p ferrite-forward -p ferrite-forward-macro --lib --tests -- -D warnings
cargo build   -p ferrite-forward --features cuda --tests
```

Expected state:
- **76 unit + 6 integration tests pass.**
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
  no auto-claim fallback.
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

### Impl library (complete for Llama/Qwen2 topology)

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

## Path to `vllm chat` working

Remaining concrete steps (everything above is done):

### 5. Migrate `ferrite-models/src/llama.rs` to `#[forward]`

Current state: `ferrite-models/src/llama.rs` uses the old
`ferrite_macros::forward!{}` macro. Migration:

1. Replace the `forward!{}` invocation with `#[forward(models_dir,
   target, workloads)]` on an empty carrier fn whose body is the
   DSL.
2. Implement the emitted `WeightBundle` trait on the weight-holding
   struct. The *fused* accessors need packed `LinearLayer`s:
   - `mlp_gate_proj_<L>__fused__mlp_up_proj_<L>` → concatenate
     `gate_proj[L].weight` and `up_proj[L].weight` along the
     output dim at load time. Shape: `[2*intermediate_size,
     hidden_size]`.
   - `self_attn_k_proj_<L>__fused__self_attn_q_proj_<L>__fused__self_attn_v_proj_<L>`
     → concatenate the three weights along the output dim. Shape:
     `[q_size + 2*kv_size, hidden_size]`. Note the name is
     alphabetical-sorted (k, q, v), not positional (q, k, v); the
     user just has to return the right packed weight for that
     method name.
3. Verify the library picks the same Impls the tests expect — run
   `cargo test -p ferrite-forward-macro --lib
   swiglu_mlp_claimed_as_fused` etc. against the migrated body.

### 6. Wire into vllm-cuda's call site

`vllm-cuda/src/model/llama.rs` currently calls the old `forward!{}`
output. The new `#[forward]` emits `pub unsafe fn forward(wm, ctx,
device, num_tokens) -> OwnedTensor` inside a `pub mod llama::<model_ident>`.
Replace the old call with the new one. The signature will diverge
from the old one (new fn takes `&W: WeightBundle`, `&ForwardCtx`,
`&mut GpuDevice`, `num_tokens: u64`); expect a small shim at the
vllm-cuda call site to build the ForwardCtx and pass the right
weight bundle. **That shim is the only vllm-cuda edit.**

### 7. Validate

```bash
timeout 60 vllm chat --model=<small-llama>
```

Coherent output → correctness verified. Garbage/hang → bug in a
fusion's semantics (most likely the in-place aliasing in
`FusedAddRmsNormImpl` or the layer indexing in
`FusedQkvRopeCacheImpl` / `AttentionViaCacheImpl`).

Then `vllm bench` for perf parity vs the old ferrite path.

### 8. Migrate Qwen2

Same shape as Llama, with the addition of per-proj biases
(`qkv_bias` etc.). The current Impl library doesn't handle biases
— a `FusedQkvRopeCacheWithBiasImpl` (or `CublasGemmExWithBiasImpl`)
port from old ferrite's library.rs would be needed. Qwen2 stress-
tests the "one arch adds one Impl; zero compiler edits" rule.

### 9. Gemma2, Mixtral, DeepSeek-V2

Per PLAN.md §5/§6/§7 — each is an acid test of compiler genericity
across a different axis (pre+post norms, MoE routing, MLA).

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

## Open tasks (next session picks up here)

Ordered; each is self-contained and testable on its own.

1. **Migrate `ferrite-models/src/llama.rs` to `#[forward]`.** See §5
   above. Biggest item. Requires the caller-side WeightBundle impl
   that produces packed fused weights at load time.
2. **Wire into vllm-cuda's call site** (§6). Small shim.
3. **Run `timeout 60 vllm chat --model=<small-llama>`.** This is
   the correctness proof. If output is garbage, debug in order of
   most-likely culprit:
   - `FusedAddRmsNormImpl` in-place aliasing (residual buffer
     semantics are subtle).
   - `FusedQkvRopeCacheImpl` layer indexing + kv_cache layout
     assumptions.
   - `AttentionViaCacheImpl` softmax scale / is_causal.
   - Packed weight layout mismatch (row-major vs column-major,
     dim order).
4. **`vllm bench` parity check.**
5. **Qwen2 migration** (§8). Adds a bias-capable fused-QKV Impl.
6. **Delete `ferrite-solver` + `ferrite-macros`** once no
   `forward!{}` site remains.

## Last note

The user has been burned by prior sessions' patterns of "small
incremental scaffolding that compounds into bullshit." If you
catch yourself proposing types-first-consumers-later, hardcoding
per-OpKind, or adding `todo!()`/placeholder values, **stop and
reread the failure-patterns section above**. The user's rule is
not negotiable: every commit complete, no revisit except for bugs.
The right unit of work is the smallest **port-with-its-consumer**
that runs end-to-end, not the smallest type definition.
