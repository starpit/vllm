# ff-interpreter — handoff

> Branch `ff-interpreter`. Read this whole file before changing
> anything. ff4 stencil rewrite is dead. The seam swap +
> universal-`Instruction<W>` pivot + FORWARD_TABLE dispatch +
> per-bucket slot metadata + dedup_quant_sig fix are all in. The
> branch is fold-ready pending a curated golden re-sweep — see §6.
>
> The original size lever ("compress per-canonical Op enum +
> __dispatch_one") was solved by lifting *everything codegen-
> independent* into a single `ferrite_forward::Instruction<W>` enum
> + `eval` match shared across every arch. That section of this
> doc was rewritten to match.

## Where we are

The branch sits at `2883e7e29` (450 commits ahead of `main`).
**Thirteen** model crates build (mixtral + qwen3-moe joined the
fleet — see §"Tier-1 MoE follow-up"), the curated 6-arch golden
subset shows 15/17 pass (two qwen3 failures are the pre-existing
per-head QK-norm divergence; see §6), and the macro-test suite is
**210/210 green**.

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

**`applies_to` per-canonical Impl gate + MoE OpKind rename** — foundation for Tier-1 MoE arches (Mixtral / Qwen2-MoE / Qwen3-MoE) that ferrite doesn't support yet. Two pieces, no behavior change:

1. `Implementation::applies_to(&self, ctx: &MatchContext) -> bool` — default-`true` trait hook + `solver::solve_with_arch_filter` non-breaking sibling that production drive uses. Evaluated once per (Impl, canonical) sequentially before the parallel match loop (proc_macro2 spans on `Program` aren't `Sync`); cached as `Vec<bool>` keyed by `ImplId`. Lets future MoE Impls probe `&Program` / `&ModelParams` for arch-distinctive config keys (e.g. `num_local_experts` ⇒ Mixtral, `num_experts` + `shared_expert_intermediate_size` ⇒ Qwen-MoE, `n_routed_experts` ⇒ DeepSeek) before claiming a `OpKind::Moe` tile every BF16 MoE Impl is otherwise eligible for.

2. Rename `OpKind::DeepSeekMoe` → `OpKind::Moe`, DSL `deepseek_moe(..)` → `moe_block(..)` (HF naming match: `MixtralSparseMoeBlock`, `Qwen2MoeSparseMoeBlock`, `DeepseekV2MoE`). DSL bodies in `ferrite-model-deepseek-{v2,v3,v3-flat}` rebased. `Instruction::DeepSeekMoe` variant name kept — it carries `WtFn<W, DeepSeekV2MoELayer>` so it's type-anchored, not OpKind-anchored.

207/207 macro tests preserved; deepseek-v2-lite expansion unchanged (331 tiles, 221 waves). No existing Impl overrides `applies_to` — gating is dormant until per-arch Impls (`FusedMoeRefImpl`, `SharedFusedMoeRefImpl`, updated `DeepSeekMoeRefImpl::applies_to`) land in the Tier-1 follow-up.

### Tier-1 MoE follow-up — landed (`5e19f72dd` + `da84ed61d` + `2883e7e29`)

Wholesale across three commits, no behavior change to existing arches:

- **`5e19f72dd`** `DeepSeekMoeRefImpl::applies_to` — gates on `n_routed_experts` (DeepSeek-distinctive). No-op today; locks the contract for the peer Impls below. 207 → 208 tests.

- **`da84ed61d`** **Mixtral wholesale**: kernel `FusedMoELayer::load` (mirror of vllm-cuda's `mixtral::load_moe` at tp=1, BF16, `w1/w2/w3` naming) → `Instruction::FusedMoe` variant + eval → `FieldLoad::FusedMoe` codegen → `FusedMoeRefImpl` gating on `num_local_experts && !shared_expert_intermediate_size && !n_routed_experts` → new `ferrite-model-mixtral` crate (`block_sparse_moe[layer]` accessor; configs: `mixtral-8x7b-instruct-v0.1` 355 tiles · 195 waves, `mixtral-1-layer` 14 tiles · 9 waves) → workspace + ferrite-models registration. 208 → 209 tests.

- **`2883e7e29`** **Qwen3-MoE wholesale**: kernel `SharedFusedMoELayer::load` (mirror of vllm-cuda's `qwen3_moe::load_moe` at tp=1, BF16, `gate_proj/up_proj/down_proj` naming, optional `shared_expert.*` + `shared_expert_gate`, `renormalize: true`) → `Instruction::SharedFusedMoe` variant + eval → `FieldLoad::SharedFusedMoe` codegen → `SharedFusedMoeRefImpl` gating on `num_experts && !num_local_experts && !n_routed_experts` (covers BOTH Qwen2-MoE-with-shared and Qwen3-MoE-Instruct-without-shared — `shared_expert_intermediate_size` is a runtime-Optional on the same kernel surface, not a family discriminator) → new `ferrite-model-qwen3-moe` crate (Qwen3 attention math + `mlp[layer]` MoE accessor; configs: `qwen3-30b-a3b-instruct` 819 tiles · 483 waves, `qwen3-moe-1-layer` 20 tiles · 13 waves) → workspace + ferrite-models registration. Adds `fused_moe_ref` + `shared_fused_moe_ref` to `lib.rs::NON_GEMM_NAMES` for kernel-class summary. 209 → 210 tests.

DeepSeek-V2-Lite expansion unchanged across all three commits (331 tiles · 221 waves; 246 waves at GGML). The applies_to gate triple
(`n_routed_experts` ↔ DeepSeek, `num_local_experts` ↔ Mixtral, `num_experts` ↔ Qwen-MoE) partitions every BF16 MoE config in the fleet exactly once.

Out of scope (deferred to next follow-up): quantized MoE variants
(Mixtral Marlin / FP8, Qwen3-MoE FP8) need dedicated `Mixtral*Impl` /
`Qwen3Moe*Impl` peers + their respective FieldLoad arms; the BF16
Impls' `matches` already defer when storage is FP8/GGML, so adding
those is purely additive. Also out of scope: Qwen3-Next-style
hybrid configs with non-empty `mlp_only_layers` — would need a
layer-conditional DSL split akin to DeepSeek-V2's
`first_k_dense_replace`. None of the official Qwen3-MoE-Instruct
checkpoints ship a non-empty `mlp_only_layers`.

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

### 4. Optional follow-ups noted along the way

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
    Loop/Alias/Free). Manual `Copy`/`Clone` impls (no `W: Copy`
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
      **207 pass / 0 fail** (4 of those are the new
      `tests::dedup_quant_sig_*` regression tests + the
      end-to-end `fp8_block_and_std_pick_separate_canonicals`).
- [ ] `cargo build -p ferrite-models --features cuda` clean
      (umbrella for every per-arch crate). Use
      `FERRITE_MODELS=<stem>` to iterate fast on a single arch.
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
