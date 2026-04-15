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
- **End-state proof**: `vllm chat --model=<small-llama>` produces
  coherent output through the new compiler at perf parity with
  the existing path. Then Qwen2.
- **Method**: port the working old ferrite (`ferrite-solver/`,
  `ferrite-macros/`, `ferrite-kernels/`) verbatim, **detoxifying
  as you port** — strip transformer-domain hardcoding, do not
  rewrite mechanisms that already work.
- **Current state**: trait-based Implementation ported, codegen
  emission on the trait (`emit_call`), 3/8 starter impls wired to
  real ferrite-kernels symbols (Embed → `embedding_gather`,
  RmsNorm → `rms_norm`, Gemm → `LinearLayer::forward`). Remaining
  5 need multi-tile matchers — see "Multi-tile fusion work" below.
- **Branch**: `worktree-ferrite-forward`. **HEAD**: `96b265d69`.

## Verify current state

```bash
cd /home/moosevan/vllm/.claude/worktrees/ferrite-forward/vllm-rs
cargo test    -p ferrite-forward -p ferrite-forward-macro
cargo fmt     -p ferrite-forward -p ferrite-forward-macro --check
cargo clippy  -p ferrite-forward -p ferrite-forward-macro --lib --tests -- -D warnings
```

Should be green: **71 unit + 6 integration tests pass.** fmt + clippy
clean (without `--features cuda`).

`cargo build -p ferrite-forward --features cuda` currently **fails**
— the codegen module emits calls to wrapper symbols
(`rms_norm_forward_owned`, `rope_append_kv`, `flash_attention`,
`silu_owned`, `add_owned`, `mul_owned`) that do not exist in
`ferrite-kernels`. This is the work item below — the codegen needs
to be rewired to existing `ferrite-kernels` symbols (or wrapper
functions added).

## What's actually done

### Foundation (in place, tested)

- **Parser** (`parse.rs`, `ast.rs`): syn-based parse of the DSL
  body. Rejects `let`. Recognizes assignments, tuple destructure,
  for-loops with literal or symbolic bounds, dotted weight paths
  with optional indexing, the `*` operator, op calls.
- **Classifier** (`classify.rs`, `classified.rs`): walks AST,
  classifies free vars as Extern (fixed enum:
  `InputIds, Positions, Rotary, BlockTable, KvCache`), Weight
  (interned `WeightId` per dotted path), or Local (SSA `LocalId`
  per assignment). `OpKind` is uniform — no `GemmQ/K/V`
  sub-variants.
- **Shape inference** (`shape.rs`): `Dim = Lit | Bound(name) |
  Mul | Var`; union-find over Vars; per-op shape signatures pin
  what each op's I/O shapes must be. `Inferred { locals, weights }`.
- **HF conventions** (`weight_conventions.rs`): standard HF
  weight-name → shape lookup; `derive_implicit_bounds` fills
  config defaults like `head_dim = hidden_size / num_attention_heads`
  and `num_key_value_heads = num_attention_heads`.
- **Config loader** (`config.rs`): reads
  `model_architectures/<arch>/*.json`, captures every top-level
  integer field as a bound.
- **CFG + unroll** (`cfg.rs`, `fuf.rs`): per-model CFG with
  concrete `u64` trip counts, then unrolled to a flat
  `Fuf { nodes: Vec<FufNode> }`. Nested `Expr::Call` and
  `Expr::Mul` propagate real shapes via `apply_signature`.
- **Solver** (`solver.rs`): DP ported from
  `ferrite-solver/src/lowering/solver/dp.rs`. Topo-forward greedy
  with `claimed[]` bitmap and multi-tile claim support.
  `Assignment { cover: TileId→SubgraphId, impls: SubgraphId→ImplId,
  predicted_us }`. Unmatched tile is `SolveError::UnclaimedTile`
  — no auto-claim fallback. Workload sweep produces
  `WorkloadAssignments { per_num_tokens: BTreeMap<u64, Assignment> }`.
- **Scheduler** (`schedule.rs`): topological wavefront over
  subgraphs. `Loop { waves: Vec<Wave> }` per workload point;
  `WorkloadLoops` keyed by num_tokens. Intra-wave dep invariant
  checker.
- **Implementation trait** (`impl_lib.rs`): full method set
  ported from old ferrite (name, target_compatible,
  workload_constraint, matches, cost_us, resources, launch_kind,
  supported_input/output_handoffs, input/output_layouts,
  is_compute_bound, can_share_kernel_with). Supporting types:
  `Resources, LaunchKind, Handoff, Layout (RowMajorBf16,
  ColMajorBf16, Any — no PagedKvBf16 yet), MatchInfo,
  WorkloadConstraint`. `ImplementationLibrary` holds
  `Vec<Box<dyn Implementation>>`. `CostCtx { fuf, profile, bounds }`.
- **Starter library** (in `impl_lib.rs`): 8 trait-object impls,
  one per OpKind. All `HostCallback`, all `Layout::RowMajorBf16`,
  analytical cost (FLOPs / bandwidth). These are placeholders
  that exercise the trait surface — they are **not** the real
  fused/calibrated impls and they **don't** map cleanly to
  ferrite-kernels' actual fused kernels.
- **Macro pipeline drive** (`lib.rs`): `#[forward(models_dir,
  target, workloads)]` parses args, runs parse → classify →
  shape-infer → CFG → unroll → solve × workloads → schedule ×
  workloads, emits per-(arch, model) module containing pipeline
  observation constants AND codegen items.
- **Codegen module** (`codegen.rs` + `emit.rs`): walks the LOOP
  for each (model × workload bucket), emits `pub trait
  WeightBundle`, `pub unsafe fn forward_m_<N>(wm, ctx, device) ->
  OwnedTensor`, and a `pub unsafe fn forward(...)` dispatching on
  `num_tokens`. Per-subgraph emission is delegated to
  `Implementation::emit_call(&EmitCtx) -> TokenStream` — no
  per-OpKind match in `codegen.rs`. New kernels / ops extend the
  library, not the codegen. **Emitted calls still reference
  symbols that don't exist in ferrite-kernels** — see "Cuda build
  gap" below.
- **`ForwardCtx`** (in `ferrite-forward` crate, cuda-gated):
  runtime args bundle the emitted forward takes (input_ids,
  positions, slot_mapping, cu_seqlens_q, seqused_k, block_table,
  max_seqlen_{q,k}, kv_cache, rotary).
- **Stable rebuild-on-JSON-change**: every config file the macro
  reads is registered via `const _: &str = include_str!("...")`
  in the emitted output. cargo rebuilds when JSONs change. No
  nightly, no build.rs.

### Cuda build gap (the immediate blocker)

Each starter impl's `emit_call` in `impl_lib.rs` (`emit_embed`,
`emit_rmsnorm`, `emit_gemm`, `emit_rope_append`, `emit_attention`,
`emit_silu`, `emit_add`, `emit_mul`) emits a kernel call. Most of
those calls reference symbols that **do not exist** in
`ferrite-kernels`. Verified ferrite-kernels public surface (via
surveys, 2026-04-15):

- `kernels::embedding_gather(weight: GpuTensor, input_ids:
  GpuTensor, alloc, stream) -> OwnedTensor` — **matches Embed**.
- `kernels::rms_norm(input: GpuTensor, weight: GpuTensor, eps: f32,
  alloc, stream) -> OwnedTensor` — **matches RmsNorm**.
- `LinearLayer::forward(&self, x: TensorView<'_>, cublas, alloc,
  stream) -> OwnedTensor` — **matches Gemm** (current emit_gemm is
  already correct).
- `kernels::silu_and_mul_fused(gate_up: GpuTensor,
  intermediate_size: usize, alloc, stream) -> OwnedTensor` —
  takes **packed [gate|up]**; no plain `silu` or `mul` exists.
  Need a multi-tile `(Silu, Mul)` impl that claims the two ops
  and maps to this.
- `attention_helpers::attention_decode_from_cache(q: TensorView,
  cu_seqlens_q, seqused_k, block_table, max_seqlen_q,
  max_seqlen_k, scale, softcap, window_size_left, kv_cache,
  layer_idx, num_sm, alloc, stream, cos_sin_cache_ptr, rotary_dim,
  is_rotary_interleaved) -> OwnedTensor` — takes **Q only**, reads
  K/V from cache. DSL's Attention has K/V inputs the kernel
  doesn't want; the cleanest fix is a multi-tile
  `(RopeAppend, Attention)` impl that elides the K/V edges.
- `kernels::fused_qkv_rope_cache(qkv: GpuTensor, positions,
  cos_sin_cache, slot_mapping, key_cache, value_cache, q_size,
  kv_size, num_q_heads, head_dim, alloc, stream) -> OwnedTensor`
  — **takes packed QKV**, writes K/V to cache, returns Q. DSL's
  RopeAppend has three separate Q/K/V tensors; need a multi-tile
  `(GemmQ, GemmK, GemmV, RopeAppend)` fused impl OR a cheap
  pre-pack step.
- **No `add`, `mul`, standalone `silu`, or `rope` kernels exist.**
  Old ferrite handled residuals via `fused_add_rms_norm_inplace`
  (residual add fused into next rmsnorm) — a 2-tile `(Add,
  RmsNorm)` matcher is probably the port.

### Multi-tile fusion work (blocks `vllm chat`)

Each of the 5 remaining starter impls (`emit_silu`, `emit_mul`,
`emit_add`, `emit_rope_append`, `emit_attention` in `impl_lib.rs`)
still emits a fake symbol. They **can't** be fixed in isolation
because no standalone kernel exists for any of them; each needs a
multi-tile matcher that replaces several singleton impls:

1. **SiluAndMul**: claims `(Silu, Mul)`. Complication: the kernel
   `silu_and_mul_fused` wants `[..., 2*intermediate]` packed
   `[gate|up]`. Our DSL has gate/up as separate GEMM outputs.
   Old ferrite paired this with a `CublasFusedGateUpGemmImpl` that
   wrote into a shared `gate_up` binding — a named cross-impl
   contract. **Better approach**: a single 4-tile impl
   `(GemmGate, GemmUp, Silu, Mul)` that issues the fused cuBLAS
   call itself and then silu_and_mul_fused. Eliminates the
   cross-impl contract.
2. **FusedQkvRopeCache**: claims `(GemmQ, GemmK, GemmV,
   RopeAppend)`. Kernel `fused_qkv_rope_cache` wants packed QKV.
   Same shape of fix as above: claim the three Gemms + the rope,
   do the fused QKV cuBLAS call, pass into the fused rope+cache
   kernel. Returns only Q; K/V are written to the paged cache.
3. **AttentionViaCache**: claims `(Attention,)` but needs to know
   `layer_idx`. Kernel `attention_decode_from_cache` takes only Q
   and reads K/V from cache. Our DSL says attention has K/V
   inputs; after (2) lands, those K/V edges have their producer
   elided (K/V flow to cache, not to attention). The matcher
   sees the DSL attention tile, ignores the K/V inputs, emits a
   call with just Q + cache metadata.
4. **FusedAddRmsNorm**: claims `(Add, RmsNorm)`. Kernel
   `fused_add_rms_norm_inplace` takes hidden + residual +
   norm weight + eps. Only fires when Add's output immediately
   feeds a RmsNorm.
5. **Standalone RmsNorm** (already wired): keeps its 1-tile
   fallback when (4) doesn't match.

Config-derived values needed at emit time (intermediate_size,
num_q_heads, head_dim, num_kv_heads, scale, layer_idx): these
live in `model.bounds` / the config.json. The emit_call body
needs access. Simplest path: thread a `ModelParams` ref through
`EmitCtx`. `layer_idx` is a per-rope/per-attention tile attribute
produced by unroll (attention appears N_LAYERS times; each
occurrence has its own index). Need to surface that on FufNode
(it's already implicitly there via tile id ordering).

`WeightBundle` field types today are keyed off consuming op:
Gemm→LinearLayer, RmsNorm→RmsNorm, Embed→Embedding. Multi-tile
impls that claim multiple Gemms still reference each weight via
the same per-tile accessor; no field-type change needed.

**The deeper structural mismatch**: ferrite-kernels' kernel set is
**fused** (silu+mul fused, rope+kv-cache fused, residual-add+rmsnorm
fused). Our DSL has separate `silu`, `mul`, `add`, `rope_append`
ops. The codegen needs **multi-tile matchers** in the Impl library
that detect adjacent (silu, mul) pairs and pick the
`silu_and_mul_fused` impl over two single-op impls. Without those
multi-tile matchers, our DSL can't be lowered to existing
ferrite-kernels.

This is what the "real Impl library port" means and it's the
biggest remaining block of work. Old ferrite has these multi-tile
matchers in `ferrite-solver/src/lowering/library.rs`
(`VllmRsSiluAndMulFusedImpl`, `VllmRsFusedQkvRopeCacheImpl`, etc.).

## Path to `vllm chat` working (concrete steps)

1. **Port the real Impl library** from
   `ferrite-solver/src/lowering/library.rs`. Each impl is a
   trait object with real `matches()` (multi-tile structural
   patterns), real `cost_us()` (calibrated from CSV cost tables
   if you also port `cost_table.rs`, or analytical with
   ferrite-kernels' real shapes), real `resources()`, real
   `supported_handoffs()`, real `input/output_layouts()`. The
   minimum set for Llama:
   - `EmbedImpl` → `embedding_gather`
   - `RmsNormImpl` → `rms_norm` (or fused with prior add via
     `FusedAddRmsNormImpl`)
   - `CublasGemmImpl` → `Linear::forward` via cuBLAS (one generic
     impl, NOT per-phase; matches any `Gemm` tile)
   - `FusedQkvRopeCacheImpl` → multi-tile match (3 gemms + rope) →
     `fused_qkv_rope_cache`
   - `FlashAttentionImpl` → `flash_attn_paged`
   - `SiluAndMulFusedImpl` → multi-tile match (silu + mul) →
     `silu_and_mul_fused`
   - `ResidualAddImpl` → fused with surrounding rmsnorm via
     `fused_add_rms_norm` when applicable
2. **(Done 2026-04-15, commit `75aa11312`.)** `emit_call` is on
   the `Implementation` trait; codegen.rs dispatches via
   `imp.emit_call(&EmitCtx)`. Per-impl emission bodies live
   alongside their cost fns in `impl_lib.rs`.
3. **Migrate `ferrite-models/src/llama.rs`** from
   `forward!{}` → `#[forward]`. Implement the emitted
   `WeightBundle` trait on the existing weight-holding struct.
4. **Wire into vllm-cuda's call site** so
   `LlamaForCausalLM::forward()` calls the generated forward fn.
5. **Run** `timeout 60 vllm chat --model=<small-llama>`. Coherent
   output → Llama works. Repeat for Qwen2.
6. **Bench** to verify perf parity.

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

Half-acceptable for getting first-end-to-end running, but it's a
debt that **must come out before Gemma2 / MoE / MLA**. The right
shape: codegen calls `impl.emit_call(&match_info, &emit_ctx)` and
each Impl emits its own kernel call. New ops add new Impls to the
library, not new arms in codegen.

### Treating `vllm-cuda/src/model/*.rs` as in-scope

Out of scope. **`vllm-cuda` is not touched.** The compiler replaces
only `ferrite-macros` + `ferrite-solver`. Its output plugs in at
the same call site the old ferrite output plugged in at.

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
    │       └── phase7_end_to_end.rs  6 integration tests
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
            │                         starter library
            ├── solver.rs           DP solver, SFUF, WorkloadAssignments
            ├── schedule.rs         wavefront scheduler, Loop, Wave,
            │                         WorkloadLoops
            └── codegen.rs          per-(model × workload) forward fn
                                      emitter (cuda gap noted above)
```

Old ferrite (the exemplar to port from) is at:

```
vllm-rs/crates/ferrite-solver/      ports source — 13k lines
vllm-rs/crates/ferrite-macros/      old proc macro (forward!{})
vllm-rs/crates/ferrite-kernels/     stays — runtime kernel wrappers
vllm-rs/crates/ferrite-cuda-builder/  stays — build pipeline
```

CUTLASS kernels live in `vllm-rs/crates/vllm-cuda/csrc/` and are
already compiled + linked via `ferrite-cuda-builder`.
`vllm-cuda/src/model/llama.rs:3667` has a `ferrite_macros::forward!{}`
invocation today — that is what task #8 (later) replaces with
`#[forward]`.

## Commit history (current branch)

```
987242340  ferrite-forward: emit forward fn (codegen module) — non-cuda complete
dd5e98259  ferrite-forward: port Implementation trait + library surface from old ferrite
083b98836  Revert "ferrite-forward: Layout metadata on Implementation"
b13d49e7b  Revert "ferrite-forward: port Handoff + Resources + Constraint types"
1b39fe078  Revert "ferrite-forward: emit Model struct + Model::load per architecture"
72bf91b99  ferrite-forward: emit Model struct + Model::load per architecture     [reverted 1b39fe078]
810e87e3b  ferrite-forward: wire macro end-to-end; port DP solver; add OpKind::Mul
e5f0a9b8e  ferrite-forward: HANDOFF.md — state snapshot for a fresh context     [original handoff — replaced]
... [phases 0-8 below this]
```

The reverts are intentional — they're the speculative-types-without-
consumers commits the user pushed back on. HEAD is the clean
trait-port + codegen-skeleton state.

## Tasks (current as of this handoff)

See the live task list in the agent's task system; high-level:

- `#1` Port DP solver — **completed**.
- `#4` Scheduler produces LOOP — **completed**.
- `#7` Wire #[forward] attribute args — **completed**.
- `#6` Layout metadata + WeightModel + organize() — **superseded**;
  the Layout / WeightBundle work is now folded into the codegen
  trajectory (codegen emits a WeightBundle trait per model;
  Layout is a per-Impl declaration). Probably mark closed and
  add new tasks per the path-to-vllm-chat steps above.
- `#5` Emit forward — **partial** (codegen module exists,
  emission references nonexistent ferrite-kernels symbols).
- `#8` Switch ferrite-models/llama.rs to #[forward] — **pending**,
  blocked on real Impl library.
- `#9–#17` per the PLAN.

## Last note

The user has been burned by prior sessions' patterns of "small
incremental scaffolding that compounds into bullshit." If you
catch yourself proposing types-first-consumers-later, hardcoding
per-OpKind, or adding `todo!()`/placeholder values, **stop and
reread the failure-patterns section above**. The user's rule is
not negotiable: every commit complete, no revisit except for bugs.
The right unit of work is the smallest **port-with-its-consumer**
that runs end-to-end, not the smallest type definition.
