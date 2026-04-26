# ff-interpreter — handoff

> Branch `ff-interpreter`. Read this whole file before changing
> anything. ff4 stencil rewrite is dead. The seam swap + codegen-
> size compression + downstream-build verification + **the runtime
> correctness fix on Llama 3.2 3B (commit `5a8301a86`)** are in.
> `vllm chat unsloth/Llama-3.2-3B-Instruct --prompt "why is the
> sky blue" --enforce-eager` produces coherent text. Next thing to
> chase is goldens (§6) + compile-time tuning (§1's cross-canonical
> dedup is the biggest lever).

## Where we are

Nine commits on top of the seam-swap checkpoint:

1. **`Vec<T>` Weights compression** (3313f8fdd) — layered accessors
   collapse from N per-layer fields + 40-arm `match layer` accessor
   methods + N per-layer load lets into one `Vec<T>` field, one
   `&self.<base>.get_unchecked(layer as usize)` accessor, one
   `(0..N).map(|layer| { format!(..., layer); …::load(…) }).collect()`
   Vec-build per layered base. `pub struct Weights`: 217→30 lines
   on commandr (-86%); `impl Weights { … }`: 578→240 (-58%);
   `pub fn load_with`: 1380→278 (-80%).
2. **Drop redundant `let _ = layer`** (58ba9b2cc) — the fn-level
   `#[allow(unused_variables)]` already suppresses warnings; the
   fn-top redundant `let layer = __layer;` setup is also gone. Per-
   arm `let layer: u32 = __layer;` shadow stays — load-bearing for
   `Op::Loop` iteration semantics.
3. **Per-bucket fn shim + manual Op Clone** (5275d036a) — per-bucket
   forward fns collapse to 1-line shims passing the bucket's static-
   slice idents + slot indices to a single `__forward_inner` /
   `__forward_backbone_inner` helper emitted once per arch module.
   `forward_backbone_m_<N>`: 30→11 lines per bucket. `Op` enum's
   `derive(Clone)` replaced with manual `*self` to skip the verbose
   `AssertParamIsClone` expansion.
4. **Shared arm-prelude lift** (2b35dca2a) —
   `extract_arch_wide_constants` records dropped `(fname, ty,
   value)` tuples in a sibling `extracted_prelude` side map on
   `ArchOpcodes`; `emit_interpreter` partitions by string-form key.
   Tuples present in 2+ variants lift to ONE fn-scope `let` at the
   top of `__dispatch_one`; per-variant residuals stay at the top
   of the arm. On commandr this lifts `cos_sin_fn`, `interleaved`,
   `biased`.
5. **Drop type annotations on extracted prelude lets** (8d96ea8d5)
   — Rust infers the type from the value, and arm-body call sites
   only need the binding's NAME. Collapses 4-line
   `for<'a> fn(&'a Weights, u32) -> &'a LinearLayer` types to one
   line per `weight_fn` / `cos_sin_fn` declaration.
6. **Trim verbose macro doc comments** (f84b116c8) — multi-paragraph
   `///` comments shortened to 1–2 line summaries (Op enum: 17→2,
   load_with: 8→2, fingerprint: 7→2, dispatcher: 8→1, etc.).
   Detailed rationale stays in macro source via `//` comments
   (prettyplease drops those from cargo expand).
7. **Three downstream-build bug fixes** (579d0644d) — surfaced when
   `cargo build -p ferrite-model-*` was run for the first time
   since the seam swap. All 11 model crates now compile clean. See
   §3 below for the bugs.
8. **Runtime correctness fix** (5a8301a86) — three coupled bugs in
   `interpreter_codegen.rs` + a debug-ergonomics change. See §5
   below; the short story is shape-aware coloring, per-row layer
   baselines in loop-compression, and a multi-valued-fname guard in
   the extracted-prelude lift. Plus `FERRITE_TRACE` runtime gate
   (was the proc-macro-time `FERRITE_DEBUG`).

`cargo expand -p ferrite-model-commandr --lib --features cuda`:
**4244 → ~2.3k lines (-45% beyond the seam-swap baseline,
~ -90% from the original ~23k pre-pivot baseline).**
Llama: **263k → ~107k lines (-59%)**, with `pub fn load_with`
dropping from 118400 to 9842 lines (-92%) — the Vec compression's
single biggest win.

Macro test count: **180 pass / 23 fail** (same 23 pre-existing
failures from before the size pass; 3 new tests added in commit
5a8301a86 pin the new invariants). All 11 model crates build
clean: llama, qwen2, qwen3, gemma2, gemma3, mistral, phi3,
granite, commandr, deepseek-v2, deepseek-v3.

The user's stretch target is <1000 lines for commandr. Cumulative
progress is in good shape but not at goal. The two remaining big
levers (cross-canonical `__dispatch_one` dedup, per-arm `Ctx`
struct refactor) are wholesale refactors — see "What's left" §1+§2
below.

## Original (pre-compression) state

Last commit before the size pass: **seam swap + coloring + Op::Loop
checkpoint**. The interpreter pivot's *core mechanism* is done:

- All 49 Impls go through `opcode_shape` / `fan_out` /
  `interpreter_arm`. `emit_call` / `EmitMode` / `EmitCtx` /
  `FragmentLibrary` are deleted.
- Per-arch enum + per-arch `__interpret` driver, dispatching via a
  `__dispatch_one(op, __layer, …)` helper. `Op::Loop(count,
  body_len)` is honored by the driver — it re-runs the next
  body_len ops `count` times, threading the iteration index to the
  arms as `__layer`.
- Linear-scan register allocation (`colored_slot_map`) packs slot
  indices so layer L's intermediates reuse the same colors as
  layer L+1's. Combined with `extract_arch_wide_constants`
  (drops fields that are byte-identical across all instances of a
  variant) + `apply_loop_compression` (generic loop detection),
  layer bodies collapse from 40× rows to one Op::Loop + one
  iteration's body.
- `forward = backbone + lm_head`. ONE backbone slice per bucket,
  ONE lm_head row. forward / forward_backbone share the backbone.
- Aliases ride at the start of the slice as `Op::Alias(dst, src)`
  rows. No per-bucket-fn alias prelude.
- Free emission dropped (coloring makes it redundant — the next
  write to a reused slot drops the old OwnedTensor automatically).
- 19 invariant tests in `interpreter_codegen::tests`.
- Fp8 family migrated via new `Fp8AnyLinear { Std(Fp8Linear),
  Block(Fp8BlockLinear) }` wrapper in `ferrite-kernels::layers`.

**`cargo expand -p ferrite-model-commandr --lib --features cuda`
went from 22,883 → 4,196 lines.** Macro crate builds in ~5s.

## What's left

The user's stretch target is <1000 lines for commandr (currently
~2.6k post-fix). Compile time is dominated by code volume — the
lever ranking is roughly: (1) cross-canonical dedup (biggest), (2)
per-arm body compression via Ctx, (3) per-arm prelude micro-
slimming. The ordering matters because (1) lifts the
bounds-independent body once per arch, multiplying any subsequent
per-arm savings by N canonicals → 1.

**Profile first.** `python3 /tmp/expand_summary.py /tmp/z2.rs`
(after `cargo expand -p ferrite-model-<arch> --lib --features
cuda > /tmp/z2.rs`) buckets expanded output by section so you can
target the heaviest one. The §1 breakdown table below came from
this script.

**Type-level shapes (deferred follow-up).** A `TensorView<S:
Shape>` with const-generic dims would have caught bug A in §5 at
compile time instead of GPU runtime — `LinearLayer::forward`
would refuse a slot whose shape doesn't match the weight's
in_features. Worth scoping when ferrite-forward gets a quiet
refactor window. Touches every kernel signature; don't bolt on
mid-bugfix. See `~/.claude/projects/-home-moosevan-vllm/memory/
project_const_generic_shapes_followup.md`.

### 1. Cross-canonical `__dispatch_one` / `Op` / `Weights` dedup

Current breakdown for commandr (cargo expand, **~2.6k total lines**
post-5a8301a86 — slight bump from the pre-fix ~2.3k due to per-arm
slot residuals that can no longer be fn-scope-lifted under the
multi-valued-fname guard, plus the always-compiled trace block.
Cross-canonical dedup is the lever that recovers this and more):

| Section | Lines | Notes |
|---|---|---|
| `fn __dispatch_one` | 810 | per-canonical, ~25 arms × ~16 lines/arm |
| `(blank lines / use / misc)` | 322 | doc comments + module wrappers + m_<N> consts |
| `pub fn load_with` | 278 | bounds-baked literals — must stay per-canonical |
| `impl Weights { … }` | 240 | accessor methods (Vec-compressed) |
| `static BACKBONE_M_<N>` | 111 | static slice rows |
| `__forward_inner` helpers | 100 | per-arch shared (already lifted) |
| `pub enum Op` | 78 | per-arch enum |
| (others smaller) | 478 | |

Commandr ships TWO canonicals (c4ai_command_r_v01 + command_r_1_layer)
because `dedup_signature` includes bounds (40 layers vs 1 layer).
Both modules emit a near-byte-identical `__dispatch_one`, `Op` enum,
`__interpret`, accessor methods, and forward helpers — just with
different bounds-baked literals in `load_with` + the static slices.

To get under 1000 lines, the bounds-INDEPENDENT parts need to lift
above the per-model `pub mod` boundary. Concretely:

- `Weights` struct + accessor methods are TYPE-IDENTICAL across
  same-arch canonicals (layered fields are `Vec<T>` regardless of
  N). One Weights definition could be shared via `pub type Weights
  = super::__shared::Weights;` — same trick as the existing shim
  mechanism.
- `Op` enum + `__dispatch_one` + `__interpret` + `__forward_inner`
  helpers reference Weights but otherwise contain no bounds-baked
  literals (the kernel-call bodies use `ctx.<...>` runtime values,
  not baked literals — except for one outlier per Impl, see below).
- `load_with` + `fingerprint_matches` + per-bucket statics +
  per-bucket forward shims STAY per-canonical (they're where bounds
  bake into literals).

The wrinkle: a few Impls bake a constant from model bounds into
their `interpreter_arm` body. E.g., `AttentionViaCacheImpl` bakes
`scale = 1/sqrt(head_dim)` as a literal `0.088388346f32` (commandr,
head_dim=128). Since the literal differs between two canonicals
with different head_dim, their `__dispatch_one` bodies aren't
byte-identical and can't be shared.

Path to fix: convert these baked constants into `OpcodeShape`
fields. `extract_arch_wide_constants` then lifts them as fn-scope
lets in `__dispatch_one`. Two canonicals with different `head_dim`
get different fn-scope `let scale = 0.0884f32;` rows, but the
arm body is byte-identical. They can share `__dispatch_one`.

Refactor scope:
- Audit every `Impl::interpreter_arm` for baked literals derived
  from model bounds/scalars. Add corresponding `OpcodeShape`
  fields. (Memory rule `feedback_no_piecemeal_codegen_migration`:
  do all Impls in one wholesale commit.)
- Add a new shim mode: `WeightsEmitMode::SharedWithCanonical` that
  emits `pub type Weights = super::<canonical>::Weights;` AND
  `pub use super::<canonical>::{Op, __dispatch_one, __interpret,
  __forward_inner, __forward_backbone_inner};` while keeping the
  variant's own `load_with` + `fingerprint_matches` + per-bucket
  statics.
- New canonicalization tier in `compute_canonical_variants`:
  partition by (impls picked + tie + rope_scaling_kind) only —
  drop bounds + scalars from this signature. Variants in the same
  partition share the bounds-independent items.

Estimated savings: commandr ~600 lines (one __dispatch_one + Op +
helpers instead of two), llama ~25k lines (57 canonicals → some
smaller number, depending on how many distinct (impl_set,
tie, rope_kind) triples exist).

### 2. Per-arm body compression via a Ctx struct (user's idea)

Each arm body is ~10-15 lines:
```
let __out = unsafe {
    let __view = ::ferrite_forward::tile_ref(__tiles, in_slot).as_view(__tiles);
    let __w = (weight_fn)(wm, layer);
    ::ferrite_kernels::kernels::xyz(*__view, __w.weight, ..., device.compute_stream)
};
__tiles[out_slot as usize] = Some(::ferrite_forward::TileEntry::Owned(__out));
```

User suggested in this session: pack the per-call state into a
`Ctx` struct with helper methods (`ctx.view(slot)`,
`ctx.write_owned(slot, t)`). Each arm body shrinks to:
```
let __view = ctx.view(in_slot);
let __w = (weight_fn)(ctx.wm, layer);
ctx.write_owned(out_slot, ::ferrite_kernels::kernels::xyz(*__view, __w.weight, ..., ctx.stream()));
```

Roughly 3-4 lines per arm × 25 arms × N canonicals = 75-100 lines
saved per canonical. Combined with #1, this would bring commandr
under 1000.

Refactor scope: similar to #1 (touch every Impl's
`interpreter_arm`). Land wholesale, commit-by-commit forbidden.

### 3. Past-but-completed: downstream-build bug fixes

Three bugs surfaced when `cargo build -p ferrite-model-*` was run
end-to-end for the first time since the seam swap. All three were
pre-existing (from commit `692d24603`); none came from this
session's compression work. Fixed in commit `579d0644d`:

- **`ReshapeRefImpl` body**: `*ndim as usize` was dereferencing a
  u8 value (`ndim: u8`). Whether the field was destructured by-
  value from the variant or extracted as a `let ndim: u8 = 2u8;`
  prelude, `*ndim` was always invalid. Fixed to `ndim as usize`.
  Affected: gemma3, qwen3, deepseek-v2, deepseek-v3 (every arch
  that emits `Op::Reshape`).
- **`TensorView` lifetime**: `as_view<'a>(&'a self, tiles: &'a
  [TileEntry]) -> TensorView<'a>` tied the returned view's
  PhantomData to `tiles`'s borrow. `TensorView` is a `Copy`
  wrapper around a raw GPU pointer — the lifetime is purely a
  marker, not a real borrow. But the borrow checker treated
  `__tiles` as immutably borrowed for the view's full scope, so
  arms that produced `Reshaped` slots from view metadata
  (`RopeAppendRefImpl`, `FusedQkvQkNormRopeCacheImpl`, …) hit
  E0502 the moment they tried to write back into `__tiles`.
  Relaxed to return `TensorView<'static>`; safety story unchanged
  (`as_view` is already `unsafe`, the drop-pass invariant keeps
  the upstream `OwnedTensor` alive).
- **`AccessorGroup` contiguity check**: my `group_accessors_by_base`
  panicked on layered families that didn't fill `0..N`
  contiguously, e.g. DeepSeek-V2's `moe` accessor (layers 1..N —
  layer 0 is dense FFN) and `mlp_down_proj` (layer 0 only —
  layers 1..N use `moe`). Replaced the `layered: bool` flag with
  an `AccessorGroupKind` enum:
  - `Unindexed` — single field, no layer arg.
  - `LayeredContiguous` — `Vec<T>`, slice-index accessor (only
    fires for layered groups starting at layer 0 with no gaps).
  - `LayeredSparse` — per-layer fields + match-arm accessor
    (legacy fallback for non-zero-start or has-gaps families).
  Also dropped the `entries.len() == num_hidden_layers`
  assertion — a contiguous-from-zero group can be a partial
  cover (e.g. `mlp_down_proj` is layer 0 only).

### 4. End-to-end build verification

**Done** (commit `579d0644d`). `cargo build -p ferrite-model-*
--features cuda` passes for all 11 arches: llama, qwen2, qwen3,
gemma2, gemma3, mistral, phi3, granite, commandr, deepseek-v2,
deepseek-v3.

### 5. Runtime correctness fix (commit 5a8301a86) — DONE

`vllm chat unsloth/Llama-3.2-3B-Instruct --prompt "why is the sky
blue" --enforce-eager` now generates coherent text. The original
crash was `cublasGemmEx [M=41, K=8192, N=5120]` from
`__dispatch_one`; root cause was three coupled bugs in
`interpreter_codegen.rs`. Only one would have produced the
crash; the other two would have silently corrupted output.

**Bug A — `colored_slot_map` was not shape-aware.** The linear-
scan allocator merged tile lifetimes that didn't overlap, with
zero regard for tensor shape. So a `[41, 8, 128]` K tile (84 KB)
and a `[41, 3072]` post-attention-residual tile (252 KB) ended up
sharing one slot. The IR's `Op::Alias(2, 0)` set up `View(0)` at
slice start, but the K tile's `Owned` write clobbered the View;
after the K tile died, the slot still held an 84 KB Owned;
`cutlass_gemm_add` then wrote 252 KB through the residual pointer
into that 84 KB buffer. Heap OOB cascaded through subsequent ops;
eventually a QKV `LinearLayer::forward` was fed a
`silu_and_mul_fused` output → cublas K=8192 N=5120 → panic.

Fix: per-shape free pools. Same-shape aliases collapse to the
owner's slot (no `View`, no `Op::Alias` row). Different-shape
aliases (`Reshape`-style) keep a `View` entry — they're naturally
in different shape pools, so the runtime indirection is
non-trivial. The `output_alias` declarations didn't need to
change; the allocator just consults shape now.

**Bug B — `apply_loop_compression` zeroed the iter-index field.**
Llama's body period spans a layer boundary: its trailing
`FusedAddRmsNorm(input_layernorm)` is the *next* layer's input ln
(fused with the residual add), so its iter-0 baseline is 1, not 0.
Zeroing all rows uniformly used layer N's input-ln weights when
computing layer N+1's input ln — accumulating drift that turned
decode to garbage after a few tokens. Same bug for post-loop ops
(static layer 27 + dispatcher's `__layer=0` → arm saw layer 0).

Fix: preserve each row's iter-0 baseline as the static field
value. Arm shadow becomes `let layer: u32 = __layer + layer;` so
each row gets `__l + baseline`. Pre/loop/post all resolve correctly
with no per-context special casing.

**Bug C — extracted-prelude lift collided fnames.** When two
variants extracted the same field name with different values
(e.g., variant A: `in_slot=2`, variant B: `in_slot=1`), both got
`let in_slot = …` at fn scope → second shadowed first → all arms
saw the wrong value. Latent before — slot indices happened to
coincide; shape-aware coloring broke the coincidence.

Fix: only fn-scope-lift when ALL variants that have an fname use
the same value. Multi-valued fnames stay in per-arm residuals.

**`FERRITE_TRACE` runtime gate.** The previous `FERRITE_DEBUG`
required a clean rebuild to enable; new design always compiles the
trace and gates it via `ferrite_forward::trace_enabled()`. Set
`FERRITE_TRACE=1` before launch (no rebuild). Disabled cost is
one atomic-load + branch per dispatched op. Op enum derives
`Debug, Copy` always so the trace can pretty-print variants.
Pair with `CUDA_LAUNCH_BLOCKING=1` for ordered output.

**Tests added** (commit 5a8301a86):
- `coloring_disjoint_lifetimes_dont_share_across_shapes`
- `coloring_same_shape_alias_collapses_to_source`
- `coloring_different_shape_alias_keeps_own_slot`
- `loop_compression_preserves_per_row_baseline`

Rewritten: `coloring_alias_dst_distinct_from_source` →
`coloring_same_shape_alias_collapses_to_source` (the old assertion
was the *cause* of bug A). Updated:
`arch_interpreter_drops_layer_drop_lines` (new arm-shadow form),
`arch_enum_uses_manual_clone_impl_to_skip_assertparamisclone`
(derive list grew Debug).

### 6. Goldens — NEXT THING

§5 fix lands runtime correctness on Llama 3.2 3B. Run the e2e
golden suite to confirm no regressions across sibling arches:

```
cargo test --release --test e_correctness -p vllm-e2e \
    --features e2e,cuda -- --ignored --test-threads=1
```

Llama subset first; match must be exact. Then Qwen2 / Qwen3 /
Mistral / Phi3 / Gemma2 / Gemma3 / Granite / CommandR / DeepSeek-V2
/ DeepSeek-V3. Any divergence is most likely a sibling-arch
manifestation of one of §5's three bugs (e.g., a different impl
mix exposes a layer-baseline corner case the M=8 / M=64 paths
don't hit).

### 7. Open warnings / cleanup

- `_protected: &HashSet<(TileId, u8)>` unused arg in `lower_bucket`
  signature. Either remove (callers update) or keep as
  documentation. Decide.
- The pre-existing `layers_moe.rs:1434` error
  (`FusedMoELayer { … }` missing fields). Not from our changes;
  should be resolved separately on `main` or by another contributor.
- 23 pre-existing macro-crate test failures unchanged: 3 named
  ones (`load_real_llama_configs`, `load_real_qwen2_configs`,
  `add_rmsnorm_pairs_claimed_as_fused_subgraph`) plus 20
  `impl_lib::tests::*` OpcodeShape round-trip tests that depend
  on Impl-side decisions made before this branch. Not in scope.

### 8. Optional follow-ups

- **Arena-backed `__tiles`.** With coloring, slot count is small
  (~10–20 for commandr) and bounded per arch. Replace `Vec<Option<
  TileEntry>>` with `[Option<TileEntry>; N_SLOTS_<ARCH>]`
  stack-allocated per per-bucket fn. Eliminates heap allocation
  per forward call.
- **Skip arch-wide-extraction's hash-string comparison.** Currently
  `to_string()` per field-value to detect identical literal tokens.
  Could pre-hash with `std::collections::hash_map::DefaultHasher`
  once per OpInstance and compare hashes for the constant-extraction
  pass too. (Already done for `detect_repeating_run`.)
- **`__dispatch_one` static slimming.** ~430 lines for commandr's
  ~25 variants × ~15 lines per arm. Each arm body has a let-rebind
  for arch-wide constants extracted from rows. Could combine
  rebinds when multiple variants share an extracted value
  (e.g., `cos_sin_fn` baked into both QkvRopeCache and
  QkvRopePrefill). Marginal gain; not worth doing before #1.

## Reference points

- `crates/ferrite-forward-macro/src/interpreter_codegen.rs`:
  - `colored_slot_map` — shape-aware linear-scan allocator. Per-
    shape free pools; same-shape aliases collapse to owner's slot;
    different-shape aliases keep a `View`. **Don't add a
    constraint that lets a slot be reused across shapes.**
  - `extract_arch_wide_constants` — drops fields with constant
    values across instances (per-variant pass; result lives in
    `extracted_prelude` side map on `ArchOpcodes`).
  - `emit_interpreter`'s prelude lift — fn-scope-lifts an fname
    only if every variant that has it uses the same value;
    multi-valued fnames stay in per-arm residuals (else they'd
    shadow each other). **Don't relax this.**
  - `detect_repeating_run` + `apply_loop_compression` — generic
    loop detection + Op::Loop emission. The iter-index field is
    set to each row's iter-0 baseline (NOT zeroed), and the arm
    shadow combines via `let layer: u32 = __layer + layer;`. Per-
    row baselines are load-bearing — Llama's body period spans a
    layer boundary.
  - `emit_enum` + `emit_interpreter` — per-arch enum (always
    `derive(Debug, Copy)` + manual Clone) + driver. The dispatcher
    body opens with the always-compiled `if
    ::ferrite_forward::trace_enabled() { … }` block, then runs
    `match __op`.
- `crates/ferrite-forward/src/lib.rs::trace_enabled`:
  - Reads `FERRITE_TRACE` env var once via `OnceLock`. Set
    `FERRITE_TRACE=1` to enable per-op tracing at runtime — no
    rebuild needed.
- `crates/ferrite-forward-macro/src/codegen.rs::emit_model`:
  - The seam. Lowers each canonical bucket once
    (skip terminal subgraph), computes terminal row separately,
    runs extraction + loop-compression passes, emits BACKBONE +
    LM_HEAD slices, emits per-bucket fns.
  - `emit_weights_struct` + `emit_weights_accessor_methods` — the
    targets of the next refactor (#1 above).
- `crates/ferrite-forward/src/tile_table.rs`:
  - `TileEntry::{Owned, View, Reshaped}`, `tile_ref`,
    `take_owned`, `view`. The interpreter's runtime data model.
  - With shape-aware coloring, in-place mutation Impls (CutlassGemmAdd,
    FusedAddRmsNorm, RopeAppend) produce same-shape aliases →
    same-slot allocation → no `View` entry at runtime. View is
    reserved for `Reshape`-style aliases (different shape, same
    storage).
- `crates/ferrite-kernels/src/layers.rs::Fp8AnyLinear`:
  - The unblocker for the Fp8 migration. Wrapper enum dispatching
    between `Fp8Linear` and `Fp8BlockLinear` at runtime.

## Pre-commit checklist (when you next commit)

Re-read this file. Verify:

- [ ] `cargo build -p ferrite-forward-macro` clean.
- [ ] `cargo test -p ferrite-forward-macro --lib` shows
      **180 pass / 23 fail** (the 23 are pre-existing —
      `load_real_llama_configs`, `load_real_qwen2_configs`,
      `add_rmsnorm_pairs_claimed_as_fused_subgraph`, plus 20
      `impl_lib::tests::*` round-trip tests that depend on
      OpcodeShape decisions made before this branch).
- [ ] `cargo build -p ferrite-model-* --features cuda` clean for
      every model crate (`llama`, `qwen2`, `qwen3`, `gemma2`,
      `gemma3`, `mistral`, `phi3`, `granite`, `commandr`,
      `deepseek-v2`, `deepseek-v3`).
- [ ] `cargo fmt` clean and `cargo clippy -D warnings` clean on
      every touched crate (per `feedback_fmt_clippy_before_commit`).
- [ ] llama golden subset green; sibling-arch goldens green.
- [ ] `vllm chat unsloth/Llama-3.2-3B-Instruct --prompt "why is
      the sky blue" --enforce-eager` produces coherent output.
      For trace, prefix with `FERRITE_TRACE=1
      CUDA_LAUNCH_BLOCKING=1` and capture stderr (no rebuild
      needed — runtime gate).
- [ ] No universal opcode enum / registry / string lookup. (Per-arch
      `Op` is the only enum; `Alias` / `Free` / `Loop` are
      universal codegen-issued, not Impl-issued.)
- [ ] No `_` arm or `unsafe { unreachable_unchecked() }` in any
      emitted match arm. (`Op::Loop(_,_) => unreachable_unchecked()`
      inside `__dispatch_one` is intentional — Loop never reaches
      dispatch, only the driver.)
- [ ] No megakernel wire-format types in ferrite-forward.
- [ ] No new tests deleted; new code lands with new
      invariant tests where applicable.
- [ ] If you touched `colored_slot_map`, `apply_loop_compression`,
      or the extracted-prelude lift: re-read §5 above. The three
      bugs there are easy to reintroduce.

If any box is unchecked, **stop and wait** — do not commit.
