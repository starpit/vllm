# ff-interpreter — handoff

> Branch `worktree-ff-tp`. Read this whole file before changing
> anything. ff4 stencil rewrite is dead. The seam swap +
> universal-`Instruction<W>` pivot + FORWARD_TABLE dispatch +
> per-bucket slot metadata + dedup_quant_sig fix are all in.
> Tensor-parallel **fanout is active** at `--features nccl`
> (foundation 8 commits + activation 6 commits, tip = `85ef0006f`):
> macro fans out every variant over `{1, 2, 4, 8}`, emits per-tp
> modules with sharded `<W as CanonicalParams>::…` constants +
> `OpKind::AllReduce` rows + per-tp `inventory::submit!`. At
> default `--features cuda` (no nccl), behavior is byte-identical
> to pre-TP — see §4 for what's still ahead before live tp>1
> works (loader sharding + commandr-tp=2 verify). The branch is
> fold-ready pending a curated golden re-sweep — see §6.
>
> The original size lever ("compress per-canonical Op enum +
> __dispatch_one") was solved by lifting *everything codegen-
> independent* into a single `ferrite_forward::Instruction<W>` enum
> + `eval` match shared across every arch. That section of this
> doc was rewritten to match.

## Where we are

The branch sits at `85ef0006f` (461 commits ahead of `main`).
Eleven model crates build under both default `--features cuda` and
`--features nccl` (10 of 11 spot-checked at nccl this session;
deepseek-v2 is similar enough to v3 that it's expected to pass),
the curated 6-arch golden subset shows 15/17 pass (two qwen3
failures are the pre-existing per-head QK-norm divergence; see §6),
and the macro-test suite is **194/194 green** under both
`cargo test -p ferrite-forward-macro --lib` and
`--lib --features nccl` (the larger pre-TP count of 207/207 dropped
after the unrelated `1dcea5ce1`/`7eb8e995b` cleanup excised dead
pivot machinery; the TP foundation + activation added 12 new tests
total to that smaller baseline).

### Architectural pivot (the big one, since `5a8301a86`)

`de15e035a` lifted the per-canonical `pub enum Op` + per-arch
`__dispatch_one` body into one universal
`ferrite_forward::Instruction<W: CanonicalParams>` enum + one
`Instruction::eval` match — both shared across every emitted arch.
Every kernel-call shape (49 + Loop/Alias/Free) lives in
`ferrite-forward/src/instr.rs`. Per-arch model bounds are baked
via the `CanonicalParams` associated-const trait (HEAD_DIM,
INTERMEDIATE_SIZE, ATTN_SCALE, …) so the eval body sees them as
`<W as CanonicalParams>::HEAD_DIM`, not as a per-arm baked literal.
Result: codegen-size scales with bucket count + accessor count,
NOT with Impl count × canonical count.

Before this, the ~810-line per-canonical `__dispatch_one` was the
biggest bucket in `cargo expand`. After: gone. Commandr dropped
from ~2.6k → 1319 → 943 lines through the subsequent stack.

### Stack on top of the pivot

`7de48e428` **FORWARD_TABLE dispatch** — replaces the old O(N×M)
nested-match `pub fn forward()` + per-bucket `forward_m_<N>` /
`forward_backbone_m_<N>` shim fns with one
`static FORWARD_TABLE: &[BucketEntry<__I>]` per canonical and a
1-line forward()/forward_backbone() that calls
`find_bucket(...)` + `ferrite_forward::run(...)`. Drop the per-
bucket `pub mod m_<N>` stub modules that exposed
`NUM_SUBGRAPHS` / `NUM_WAVES` / `PREDICTED_US` constants — those
constants weren't consumed; the only test invariant they actually
captured (prefill cost ≫ decode cost) lives in `solver::tests`
now, calling `solve()` directly. Llama: 81524 → 41797 lines (-49%).

`a4362a7a3` **layer_weight_path helper** — fold each emitted
`format!("model.layers.{}.X", layer)` into
`ferrite_forward::layer_weight_path(layer, "X")`, dropping the 5-
line post-expansion `format!()` macro spew per call site. Also
trims redundant `#[cfg]` / `#[inline]` / `#[allow]` per-method
attrs on `impl Weights` (the impl block carries them). Llama
41797 → 39460.

`569b6d308` **load_layered_\* helpers** — fold each
`(0..N).map(|layer| Type::load(gw, &layer_weight_path(...), …))
.collect::<Result<Vec<_>>>()?` block emitted per layered accessor
into one `ferrite_forward::load_layered_<kind>(gw, n_layers,
suffix, …)?` call. One helper per `FieldLoad` arm
(rms_norm / cohere_layer_norm / linear_dense{,_concat} /
marlin_linear{,_concat} / fp8_linear{,_concat} /
fp8_block_linear{,_concat} / bnb4{,_concat} + embedding).
DeepSeekV2Moe stays inline (one consumer, not worth crossing the
ferrite-forward / ferrite-kernels seam). Llama 39460 → 36025
(-8.7%); `pub fn load_with` section: 8645 → 5210 (-40%).

`8a45b39d6` **per-bucket slot metadata** — `BucketEntry` gained
three u32 fields (`num_slots`, `backbone_slot`, `terminal_slot`).
Was a runtime correctness bug: per-canonical `TERMINAL_SLOT`
const was sourced from the FIRST bucket's slot map, but the
solver picks different Impls per workload point — e.g.
`CutlassGemmAdd` fuses the residual into the GEMM at prefill,
freeing a slot vs the decode bucket's separate-Add path. On
commandr the M_64 prefill bucket terminates in slot 7; M_1/M_8
decode terminate in slot 8. Graph-capture's prefill warmup
panicked with `tile slot 8 consumed but empty`. Fixed by reading
`e.6 / e.7 / e.8` from the bucket entry instead of the per-
canonical const. **Loadbearing across every arch with mixed
prefill/decode Impl picks** — assume more arches were latently
wrong and just hadn't hit graph capture yet.

`82a46a7b9` **dedup_quant_sig in canonical hashing** — a
correctness fix for FP8 dispatch. The `dedup_signature()` used
for cross-variant canonical equivalence didn't include the
QuantMethod arm. AWQ/GPTQ/CT correctly collapse to one canonical
(runtime `marlin_storage` discriminator covers their on-disk
split), but FP8 has TWO `FieldLoad` arms — `Fp8Linear` (1D scale)
vs `Fp8BlockLinear` (2D 128×128 scale) — that emit incompatible
load_with bodies. With matching Impl picks, qwen3's
`fp8-block-128x128` and `fp8-dynamic-per-tensor` hashed
identically; alphabetically `block` won canonical and the
`dynamic` shim called `Fp8BlockLinear::load` on a 1D scale
tensor → server panicked at startup. The fix adds
`dedup_quant_sig(method)` returning a stable per-arm string
(`q:fp8-block` / `q:fp8-std` / `q:awq` / `q:gptq` / `q:bnb4` /
`q:dense`) threaded into the dedup signature. Covered by 4 unit
tests in `lib.rs::tests`.

### Tensor-parallel — activation phase landed (tip `d11085d0b`)

The TP rollout is now ACTIVE at `--features nccl`. 14 commits total:
8 foundation (since `f7edb7032`) + 6 activation (since `64946efba`).
At default `--features cuda` (no nccl), behavior is byte-identical
to pre-TP — only tp=1 modules emit. At `--features nccl`,
`compile()` fans out every variant over `{1, 2, 4, 8}` and emits
per-tp modules with sharded `<W as CanonicalParams>::…` constants
+ `OpKind::AllReduce` rows after every row-parallel gemm + one
`FerriteArchRegistration` per `(arch, tp)` tuple.

**Activation commits:**

| Commit       | Lands |
|--------------|---|
| `64946efba`  | `SolvedModel.tp_world_size: u8` field threaded into `dedup_signature` + `tp_lowering::insert_all_reduces`. Pinned by `tp_world_sizes_pick_separate_canonicals`. |
| `ee0cdb9f8`  | `FerriteArchRegistration.tp_world_size: u8` + `try_load(arch_hint, tp_world_size)` matching. cuda_worker passes runtime tp. |
| `09c6c1262`  | `nccl` cargo feature on every per-arch crate forwards down through `ferrite-forward/nccl` → `ferrite-forward-macro/nccl` (next commit) → kernels/cuda-core. ferrite-models umbrella + vllm-executor forward too. |
| `b8c25e5d2`  | `emit_canonical_params_impl(model, tp)` floor-divides column-parallel dims (`num_q_heads`, `num_kv_heads`, `intermediate_size`, `q_size`, `kv_size`) by tp. 3 tests pin tp=1 identity, tp=2, tp=8. |
| `889c44b2f`  | The big one. `compile()` outer loop iterates `[1, 2, 4, 8]` when `cfg!(feature = "nccl")`. `_tp{N}` suffix on per-(model, tp) idents. Indivisible (variant, tp) tuples skipped with stderr line. `DispatchArm` struct + `Weights::load(.., tp_world_size: u8)` with per-tp match arms + per-tp `inventory::submit!`. `ferrite-forward-macro/Cargo.toml` gains its own `nccl = []` feature so cargo recompiles the proc-macro per-feature-set. |
| `d11085d0b`  | Schedule fix uncovered by activation: lowering breaks the "SubgraphId order = topological order" invariant (AllReduce tiles get high TileId but their consumers sit at low TileId). Replaced linear-walk wave assignment with memoized DFS topo. O(N+E). Surfaced as a slot-map alias-resolution panic on mistral / qwen2; commandr dodged (DSL terminal is scalar mul, not row-parallel gemm). |

**Foundation commits:**

| Commit       | Lands |
|--------------|---|
| `f7edb7032`  | `Instruction::AllReduce(u32)` variant + eval arm + `ForwardCtx::tp_group` field, all gated on a new `nccl` cargo feature on `ferrite-forward`. Stub `Self::Ferrite` arm in `cuda_worker::set_tp_group`. |
| `a43b2ccda`  | `dedup_tp_sig(u8) → "tp:N"` threaded into `SolvedModel::dedup_signature` as a `tp:1` constant. 3 unit tests on `dedup_quant_sig_*` precedent. |
| `624ca9fea`  | `coloring_allreduce_collapses_to_input_slot` regression tripwire. |
| `bc6444445`  | `cutlass_gemm_add_does_not_claim_across_intermediate_node` + `fused_add_rms_norm_*` — the comm-boundary guard is **emergent** from `consumes_tile(add, gemm)` chain-break; no explicit solver code change needed. |
| `cc75942f7`  | `OpKind::AllReduce` variant + `shape::apply_signature`/`weight_arg_ranks` arms + `AllReduceImpl` registered in `starter_library()`. |
| `a02ff0b7b`  | `tp_lowering` module + `insert_all_reduces` skeleton with no-op fast path at tp=1. Wired into `compile()` between `unroll` and `solve`. |
| `0565162c1`  | `shard_kind_for_weight_path` (HF-standard names) + actual insertion logic at tp>1: append AllReduce node, rewire all consumers of gemm output. 3 tests. |
| `03ce7668e`  | `FerriteModel.tp_group` field; `set_tp_group` arm assigns into it; both `ForwardCtx` construction sites pass `tp_group: m.tp_group.as_ref()`. Runtime plumbing connected end-to-end. |

Result: `cargo check -p vllm-executor` clean under both
`--features cuda` (variant absent, no NCCL dep) and `--features
nccl` (variant + tp_group present). Macro test suite: 194/194 pass
under both `--features ""` and `--features nccl`. At runtime, tp=1
forward calls still read-then-discard the `tp_group` field on every
`ForwardCtx` — the AllReduce eval arm is unreachable because the
lowering pass emits zero rows at tp=1. **At runtime tp>1, the
fanout is inert until task #6 (loader sharding) lands** — the
emitted `Weights::load_tp{N}` paths run, but the safetensors load
returns full-rank tensors that don't match the sharded
`CanonicalParams` sizes. See "What's left §4" below.

Verified `--features nccl` compile on commandr, mistral, qwen2,
llama, gemma2, gemma3, phi3, qwen3, granite, deepseek-v3 (10 of 11
arches). Skip-at-indivisible warnings appear on variants with
heads / intermediate not divisible by tp (qwen2.5-7b's 28 heads at
tp=8, SmolLM's 3 KV heads at any tp>3, etc.) — those (variant, tp)
tuples drop out and the umbrella build still succeeds.

### Upstream rebase that landed mid-branch

`f814ea3a8` + `9bd2ae930` (not ff-interpreter commits — they came
from the upstream `main`-tracking branch) moved per-arch JSONs out
of `model_architectures/<arch>/` and into
`crates/ferrite-model-<arch>/configs/`. Each per-arch crate is now
an optional dep behind an `arch-<name>` feature on
ferrite-models, and `FERRITE_MODELS=stem1,stem2,…` is an
env-var filter on the macro that drops llama from ~1m40s → ~2s
on a single-model build. **Use this for fast iteration** — pass
the explicit synthesized stems for quant variants (e.g.
`FERRITE_MODELS="qwen3-0.6b,qwen3-0.6b-fp8-dynamic-per-tensor,
qwen3-0.6b-fp8-block-128x128"`).

## Current expansion sizes

`python3 /tmp/expand_summary.py /tmp/<arch>_lp.rs` after
`cargo expand -p ferrite-model-<arch> --features cuda > /tmp/<arch>_lp.rs`.

| Arch | Total | Notes |
|---|---|---|
| commandr | 943 | 2 canonicals (40-layer + 1-layer) |
| llama (full) | 36025 | ~57 canonicals, biggest crate |

Llama section breakdown post-`82a46a7b9`:

| Section | Lines | Notes |
|---|---|---|
| `static BACKBONE_M_<N>` | 10723 | Per-canonical static `&[__I]` slices |
| `pub fn load_with` | 5210 | Per-canonical, calls `load_layered_*` helpers |
| `impl Weights { … }` | 5186 | Vec-compressed accessor methods |
| `pub fn fingerprint_matches` | 4048 | Per-variant dispatch |
| `static FORWARD_TABLE` | 1838 | One per canonical |
| `pub unsafe fn forward_backbone()` | 1879 | thin shim |
| `pub fn load` | 1416 | per-variant |
| `pub unsafe fn forward()` | 1219 | thin shim |
| `pub struct Weights` | 1026 | per-canonical |
| (other) | ~3400 | const decls, attrs, doc, …  |

## What's left

### 1. Fold-readiness — the immediate goal

The user asked whether the branch is ready to fold into the
upstream `ferrite-forward` line. Answer is roughly yes:

- All compression + correctness work has landed and tests pass.
- Curated 6-arch golden sweep: 15/17 pass.
- The two failures are both qwen3 — the base model and
  fp8_dynamic — and are the **same** pre-existing
  per-head-QK-norm divergence (memory:
  `project_perhead_qknorm_golden_divergence`). The fp8_dynamic
  case used to be a graph-capture *panic* before `82a46a7b9`;
  now it loads cleanly and fails the same way the base does.

Before merging, want to clear:

- Re-run the curated golden subset on a clean disk
  (sweep was run at 99% disk utilization; one previous attempt
  truncated at deepseek_v3_bzantium when the cache filled).
- Decide whether the qwen3 QK-norm divergence ships as known-bad
  or blocks the fold. It tracks separately from this branch — was
  failing before and after.

### 2. Compile-time tuning — past the lexical floor

Per memory `feedback_codegen_size_metric`: at ~36k expanded lines,
the rustc-frontend bottleneck is **item count** (structs / fns /
modules), not source bytes. Don't chase shorter accessor names or
character-level wins anymore.

Genuine remaining item-count targets:

- Per-bucket `static BACKBONE_M_<N>` + `static LM_HEAD_M_<N>` —
  ~60 statics per arch. Could potentially lift into a single
  `static SLICES: &[(&[__I], &[__I])]` per canonical, indexed by
  bucket id. Cuts ~50% of the static decls.
- Per-accessor methods on `impl Weights` — already Vec-
  compressed; further wins would require sharing the `Weights`
  struct across same-arch canonicals (the type IS already
  identical across canonicals modulo bounds — the `pub type
  Weights = super::__shared::Weights;` shim trick mentioned in
  the old §1 still applies, just at the per-arch crate level
  rather than within a single `pub mod`).
- Per-variant `fingerprint_matches` fns — already minimal but
  there's one per variant (~40+ on llama, ~30 on phi3). Could be
  collapsed into a table-driven shape match emitting one helper
  call per variant.

Don't bother with: shorter accessor names, format!() collapsing,
attribute trimming. Those are lexical and rustc doesn't care.

### 3. Per-head QK-norm divergence (unrelated to this branch)

Per `project_perhead_qknorm_golden_divergence`. Suspect: gemma3 +
qwen3 share a per-head-norm path that diverges from Python vLLM.
Not yet pinned; was failing before this branch and continues to
fail. Tracked separately. The fp8 path on qwen3 (commit
`82a46a7b9`) flushed out one *separate* fp8 dispatch bug along
the way, which is now fixed and tested — but the QK-norm
divergence sits underneath.

### 4. Tensor-parallel — runtime verify still ahead

Task #7 (canonical fanout) landed this session along with the
schedule-topo fix it surfaced. The macro side of TP is now a full
fanout; what remains is the runtime weight-loading path and the
correctness verify on a real model. Read memory
`project_tp_design_notes` for the full commit map.

**Task #6 — loader sharding.** Per-arch shard-kind table
(`shard_kind_for_weight_path` from `tp_lowering`) reused for
load-time slicing. `(rank, world)`-aware tensor load that slices
Dim0 for column-parallel weights (q/k/v/gate/up), Dim1 for
row-parallel (o/down), Replicate otherwise. Bias on row-parallel
layers moves post-AllReduce, replicated (only matters for arches
with linear bias — Llama doesn't, some Qwen2 / Granite variants
do; 4th-decimal divergence from Python vLLM at tp>1 if missed).
KV replication when `num_kv_heads < tp_size` (Llama3-8B at tp=16
etc.). Two pending tests: `loader_shards_dim0_for_qkv_dim1_for_o_proj
_at_tp_2` and `kv_replication_when_num_kv_heads_lt_tp_size`.

**Task #9 — verify on commandr at tp=2.** Per memory
`feedback_smallest_model_for_verify`. End-to-end: build vllm-cli
with `--features nccl`, run `vllm chat` with `--tensor-parallel-size
2` on commandr (943 expanded lines vs llama's 36025), then run the
curated golden subset at TP=2. Depends on #7 + #6.

### 5. Optional follow-ups noted along the way

- **Type-level shapes.** `TensorView<S: Shape>` with const-
  generic dims would have caught the in-flight shape-mismatch
  bugs at compile time (memory:
  `project_const_generic_shapes_followup`). Touches every kernel
  signature; don't bolt on mid-bugfix.
- **Item-count expand_summary.** Current `/tmp/expand_summary.py`
  groups by line count. An item-count grouping would surface the
  next-biggest target faster (e.g. count `^pub fn` / `^static `
  per arch).

## Reference points

- `crates/ferrite-forward/src/instr.rs`:
  - `pub trait CanonicalParams { const HEAD_DIM: u32; … }` —
    associated-const surface for arch-baked numerics. Each
    emitted Weights `impl CanonicalParams` plants the literals
    that the eval match reads as `<W>::HEAD_DIM` etc.
  - `pub enum Instruction<W>` — 52 tuple variants (49 Impls +
    Loop/Alias/Free) plus `AllReduce(u32)` cfg-gated on
    `feature = "nccl"`. Manual `Copy`/`Clone` impls (no `W: Copy`
    bound), no `Debug` derive. Tuple variants intentional —
    struct variants would balloon emitted size.
  - `unsafe fn run<W>(backbone, lm_head, …)` and
    `unsafe fn run_backbone<W>(backbone, …)` — universal
    interpreter drivers. Allocate a `Vec<Option<TileEntry>>` of
    `num_slots`, run each slice, return `take_owned(tiles,
    terminal_or_backbone_slot)`.
- `crates/ferrite-forward/src/lib.rs`:
  - `BucketEntry<Op>` — 9-field tuple struct.
    `(m_min, m_max_excl, sk_min, sk_max_excl, backbone, lm_head,
    num_slots, backbone_slot, terminal_slot)`. Per-bucket slot
    fields are load-bearing — see `8a45b39d6`.
  - `find_bucket` — linear-scan dispatch with `table[0]` fallback
    for out-of-range inputs.
  - `layer_weight_path(layer, suffix)` — folds `format!()`
    expansion. Used by emit + the `load_layered_*` helpers.
  - `loaders` module — `load_layered_<kind>(...)` helpers per
    `FieldLoad` arm. Each owns its own `(0..N).map().collect()`
    loop so call sites collapse to one fn call.
  - `dedup_quant_sig(method)` (private) — stable per-FieldLoad-
    arm string for cross-variant canonical hashing.
  - `dedup_tp_sig(tp_world_size)` (private) — TP analog of
    `dedup_quant_sig`. Threaded into `SolvedModel::dedup_signature`
    as `dedup_tp_sig(self.tp_world_size)`. Pinned by 3 unit tests
    + `tp_world_sizes_pick_separate_canonicals` end-to-end.
  - `tp_lowering::insert_all_reduces(fuf, program, tp_world_size)`
    — FUF-lowering pass. Walks every `OpKind::Gemm`, looks up the
    weight via `program.weights.path(...)`, and at tp>1 appends
    `OpKind::AllReduce` after every `ShardDim1` weight, rewiring
    every consumer of the gemm output to the new tile. Wired into
    `compile()` between `unroll` and `solve` with `tp_world_size`
    from the SolvedModel. Five tests cover shard-kind table +
    tp=1 no-op + tp>1 insertion + column-parallel skip + degenerate
    empty-FUF. **Caveat:** appending AllReduce nodes at high TileId
    breaks the historical "SubgraphId order = topological order"
    invariant — `schedule.rs` was rewritten to memoized DFS topo
    in `d11085d0b` to handle this. Future passes that care about
    that invariant should use the schedule's wave assignment.
  - `AllReduceImpl` (impl_lib.rs) — single-tile claim of any
    `OpKind::AllReduce`, output_alias collapses dst to input slot
    (so shape-aware coloring places the AllReduce row at the
    gemm output's slot with no `View`). No `interpreter_arm` —
    the universal eval arm in `instr.rs` is the dispatch path.
  - `trace_enabled()` — `OnceLock<bool>` reading `FERRITE_TRACE`
    env var. The eval match opens with the always-compiled
    `if ferrite_forward::trace_enabled() { … }` block; setting
    `FERRITE_TRACE=1 CUDA_LAUNCH_BLOCKING=1` enables runtime
    traces with no rebuild.
- `crates/ferrite-forward/src/tile_table.rs`:
  - `TileEntry::{Owned, View, Reshaped}`, `tile_ref`,
    `take_owned`, `view`. Runtime data model. With shape-aware
    coloring, in-place mutation Impls produce same-shape aliases
    → same-slot allocation; `View` is reserved for `Reshape`-
    style aliases (different shape, same storage).
- `crates/ferrite-forward-macro/src/codegen.rs`:
  - `emit_model` — the seam. Lowers each canonical bucket once
    (skipping terminal subgraph), computes terminal row
    separately, runs `apply_loop_compression`, emits BACKBONE +
    LM_HEAD slices, builds `FORWARD_TABLE` + `forward()` /
    `forward_backbone()`.
  - `emit_canonical_params_impl` — emits the `impl
    CanonicalParams for Weights` block with all bound-derived
    associated consts (incl. MLA YaRN logic for DeepSeek-V3).
  - `emit_layered_load_body(plan, n_layers)` — emits one fn-call
    expression to a `load_layered_*` helper.
  - `bucket_table_entries` loop — emits the per-bucket
    `__B(m_min, m_max, sk_min, sk_max, bb, lm, num_slots,
    bb_slot, term_slot)` literals.
- `crates/ferrite-forward-macro/src/interpreter_codegen.rs`:
  - `colored_slot_map` — shape-aware linear-scan allocator. Per-
    shape free pools; same-shape aliases collapse to owner's
    slot; different-shape aliases keep a `View`. **Don't add a
    constraint that lets a slot be reused across shapes.**
  - `emit_bucket_static_slice` — emits each `static BACKBONE_M_<wp>:
    &[__I] = &[…];` row using bare variant constructors (post-
    `Instruction::*` glob-import).
  - `apply_loop_compression` — generic loop detection +
    `Instruction::Loop` emission. Iter-index field set to each
    row's iter-0 baseline (NOT zeroed); arm shadow combines via
    `__layer + layer`. Per-row baselines are load-bearing.
- `crates/ferrite-forward-macro/src/lib.rs`:
  - `dedup_signature` — equivalence-class hash for canonical
    selection. Includes bounds, scalars, tie_word_embeddings,
    rope_scaling_hash, **`dedup_quant_sig`**, and the SFUF Impl
    picks per workload point. **Don't drop quant from the sig**
    — see `82a46a7b9`.
  - `compute_canonical_variants` — alphabetically-earliest stem
    wins canonical for each equivalence class.
- `crates/ferrite-kernels/src/layers.rs::Fp8AnyLinear`:
  - Wrapper enum dispatching between `Fp8Linear` (1D scale) and
    `Fp8BlockLinear` (2D scale) at runtime. The two have
    DIFFERENT `load` paths → different load_with bodies → must
    not share canonicals (see §1's `82a46a7b9` note).

## Pre-commit checklist (when you next commit)

Re-read this file. Verify:

- [ ] `cargo build -p ferrite-forward-macro` clean.
- [ ] `cargo test -p ferrite-forward-macro --lib` shows
      **194 pass / 0 fail** (drops from the larger pre-TP 207
      after `1dcea5ce1`/`7eb8e995b`'s dead-pivot excision; the
      4 `tests::dedup_quant_sig_*` regression tests +
      `fp8_block_and_std_pick_separate_canonicals` +
      `tp_world_sizes_pick_separate_canonicals` + 3
      `canonical_params_at_tp_eq_*` + 5 `tp_lowering::tests::*`
      live in the current count).
- [ ] `cargo test -p ferrite-forward-macro --lib --features nccl`
      ALSO 194/194 (cargo recompiles the proc-macro per-feature-set;
      both feature configurations must stay green).
- [ ] `cargo build -p ferrite-models --features cuda` clean
      (umbrella for every per-arch crate, default tp=1 only). Use
      `FERRITE_MODELS=<stem>` to iterate fast on a single arch.
- [ ] `cargo build -p ferrite-models --features nccl` clean
      (umbrella, fans out tp ∈ {1,2,4,8}). Indivisible
      (variant, tp) tuples drop with stderr `skip tp=N` lines —
      this is expected, not a regression.
- [ ] `cargo fmt` clean and `cargo clippy --all-targets
      -D warnings` clean on every touched crate (per
      `feedback_fmt_clippy_before_commit`).
- [ ] Curated golden subset green:
      `test_cuda_correctness_qwen2_0_5b{,_gptq,_bnb_4bit,_fp8_dynamic}`,
      `test_cuda_correctness_smollm_135m`,
      `test_cuda_correctness_gemma2_2b{,_awq,_gptq,_w4a16_ct,_fp8_dynamic,_fp8_static}`,
      `test_cuda_correctness_command_r_1l`,
      `test_cuda_correctness_granite_3_3_2b`,
      `test_cuda_correctness_qwen3_0_6b_{bnb_4bit,fp8_block}`.
      Exclude qwen3 base + qwen3 fp8_dynamic until the per-head
      QK-norm divergence is pinned.
- [ ] `vllm chat unsloth/Llama-3.2-3B-Instruct --prompt "why is
      the sky blue" --enforce-eager` produces coherent output.
      For trace, prefix with `FERRITE_TRACE=1
      CUDA_LAUNCH_BLOCKING=1` and capture stderr (no rebuild
      needed — runtime gate).
- [ ] No per-arch `Op` enum or per-arch `__dispatch_one`
      (`ferrite_forward::Instruction<W>` + `eval` is the only
      kernel-dispatch surface).
- [ ] No `_` arm or `unsafe { unreachable_unchecked() }` in any
      emitted match arm.
- [ ] No megakernel wire-format types in ferrite-forward.
- [ ] No new tests deleted (per
      `feedback_never_delete_tests` — load-bearing); new code
      lands with new invariant tests where applicable.
- [ ] If you touched `colored_slot_map`, `apply_loop_compression`,
      `dedup_signature`, or per-bucket slot emission: re-read
      the relevant §Where-we-are entry. The runtime panics
      `8a45b39d6` and `82a46a7b9` fixed are easy to reintroduce.
