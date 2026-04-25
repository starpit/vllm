# ff-interpreter — handoff

> Branch `ff-interpreter`. Read this whole file before changing
> anything. ff4 stencil rewrite is dead. The seam swap + the
> first round of codegen-size compression are in. End-to-end build
> and goldens still need to run.

## Where we are

Five commits on top of the seam-swap checkpoint focus on cargo-
expand size:

1. `Vec<T> Weights compression` — layered accessors collapse from
   N per-layer fields + 40-arm `match layer` accessor methods + N
   per-layer load lets into one `Vec<T>` field, one
   `&self.<base>.get_unchecked(layer as usize)` accessor, one
   `(0..N).map(|layer| { format!(..., layer); …::load(…) }).collect()`
   Vec-build per layered base. `pub struct Weights`: 217→30 lines on
   commandr (-86%); `impl Weights { … }`: 578→240 (-58%);
   `pub fn load_with`: 1380→278 (-80%).
2. `__dispatch_one` redundant `let _ = layer;` lines dropped (the
   fn's `unused_variables` allow already suppresses warnings); the
   fn-top redundant `let layer = __layer;` setup is gone too. Per-
   arm `let layer: u32 = __layer;` shadow stays — load-bearing for
   Op::Loop iteration semantics.
3. Per-bucket forward fns collapse to 1-line shims passing the
   bucket's static-slice idents + slot indices to a single
   `__forward_inner` / `__forward_backbone_inner` helper emitted
   once per arch module. `forward_backbone_m_<N>`: 30→11 lines per
   bucket. Op enum's `derive(Clone)` replaced with manual
   `*self` impl to skip the verbose `AssertParamIsClone` expansion.
4. Shared arm-prelude lift: `extract_arch_wide_constants` records
   dropped (fname, ty, value) tuples in a sibling `extracted_prelude`
   side map on `ArchOpcodes`; `emit_interpreter` partitions by
   string-form (key = (fname, ty, value)). Tuples present in 2+
   variants lift to ONE fn-scope `let` at the top of `__dispatch_one`;
   per-variant residuals stay at the top of the arm. `cos_sin_fn`,
   `interleaved`, `biased` lift on commandr.
5. Type annotations dropped on extracted prelude lets — Rust infers
   the type from the value, and arm-body call sites only need the
   binding's NAME. Collapses 4-line `for<'a> fn(&'a Weights, u32)
   -> &'a LinearLayer` types to one line per `weight_fn` /
   `cos_sin_fn` declaration.

`cargo expand -p ferrite-model-commandr --lib --features cuda`:
**4244 → 2417 lines (-43% beyond the seam-swap baseline).**
Llama: **263k → 111k lines (-58%)**, with `pub fn load_with`
dropping from 118400 to 9842 lines (-92%) — the Vec compression's
single biggest win.

The user's stretch target is <1000 lines for commandr. Cumulative
progress is in good shape but not at goal yet. The major remaining
levers (cross-canonical __dispatch_one dedup, per-arm body
compression via a Ctx struct + helper methods) are bigger refactors
— see "What's left" §1 below.

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

### 1. Cross-canonical `__dispatch_one` / `Op` / `Weights` dedup

Current breakdown for commandr (cargo expand, 2417 total lines):

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

### 3. Past-but-completed: Vec / shared-prelude / per-bucket dedup

Already in main on this branch as the five commits described in
"Where we are". No further action needed.

### 4. End-to-end build verification

The macro crate builds + tests pass. **`cargo build -p
ferrite-models --features cuda` has not been verified end-to-end
since the seam swap.** Earlier attempts hit two issues that are
now fixed (`Fp8AnyLinear` accessor type guard rejection, panic in
`plan_field_load` on `Fp8AnyLinear`). After landing the Weights
refactor, run a full ferrite-models build and confirm:

- Macro expansion succeeds for every model variant on disk.
- rustc can compile the emitted code (no missing `load` /
  `Weights` symbols downstream).

There are integration tests under `crates/ferrite-forward/tests/`
(phase7_end_to_end, gemma2_end_to_end) that compile every variant
through the macro — those serve as the smoke test.

### 5. Goldens

After the build is green, run:

```
cargo test --release --test e_correctness -p vllm-e2e \
    --features e2e,cuda -- --ignored --test-threads=1
```

Llama subset first; match must be exact. Then Qwen2 / Qwen3 /
Mistral / Phi3 / Gemma2 / Gemma3 / Granite / CommandR / DeepSeek-V2
/ DeepSeek-V3.

The interpreter design changes runtime behavior in subtle ways:

- Slot reuse via coloring. If any kernel reads its input AFTER
  starting to write its output (atypical), color sharing could
  expose a UAF. Currently every Impl's interpreter_arm reads the
  view before calling the kernel; the kernel queues its work then
  the OwnedTensor is dropped. Should be safe but test exercises
  this empirically.
- Op::Loop's iteration counter passed via `__layer` rather than
  per-row literal. If any arm body reads the layer index from a
  destructured field after the let-rebind shadows it, the layer
  used for kv_cache index / weight lookup would be wrong. The
  shadowing pattern is in `emit_interpreter` (post-destructure
  `let layer: u32 = __layer;`). Verify by reading the expanded arm
  bodies — none should reference the destructured `layer` field
  before the let.
- Aliases as `Op::Alias` rows execute IN slice order. Check that
  the alias prelude in the expanded output runs before any Op
  that reads the aliased slot. (Should be fine — aliases are
  prepended in `lower_bucket`.)

### 6. Open warnings / cleanup

- `_protected: &HashSet<(TileId, u8)>` unused arg in `lower_bucket`
  signature. Either remove (callers update) or keep as
  documentation. Decide.
- The pre-existing `layers_moe.rs:1434` error
  (`FusedMoELayer { … }` missing fields). Not from our changes;
  should be resolved separately on `main` or by another contributor.
- Three pre-existing macro-crate test failures
  (`load_real_llama_configs`, `load_real_qwen2_configs`,
  `add_rmsnorm_pairs_claimed_as_fused_subgraph`) unchanged. Not in
  scope.

### 7. Optional follow-ups

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
  - `colored_slot_map` — linear-scan register allocator.
  - `extract_arch_wide_constants` — drops fields with constant
    values across instances.
  - `detect_repeating_run` + `apply_loop_compression` — generic
    loop detection + Op::Loop emission. Caller passes the
    iter-index field name (`"layer"` for transformers).
  - `emit_enum` + `emit_interpreter` — per-arch enum + driver.
  - The arm-body wrapper `let layer: u32 = __layer; … #body` is the
    bridge from Op::Loop's iteration counter to arm bodies.
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
- `crates/ferrite-kernels/src/layers.rs::Fp8AnyLinear`:
  - The unblocker for the Fp8 migration. Wrapper enum dispatching
    between `Fp8Linear` and `Fp8BlockLinear` at runtime.

## Pre-commit checklist (when you next commit)

Re-read this file. Verify:

- [ ] `cargo build -p ferrite-forward-macro` clean.
- [ ] `cargo test -p ferrite-forward-macro --lib` passes
      (3 pre-existing failures don't count, see above).
- [ ] `cargo build -p ferrite-models --features cuda` clean.
- [ ] llama golden subset green; sibling-arch goldens green.
- [ ] `cargo fmt` clean on touched crates.
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

If any box is unchecked, **stop and wait** — do not commit.
