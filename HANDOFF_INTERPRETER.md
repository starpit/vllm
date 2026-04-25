# ff-interpreter — handoff

> Branch `ff-interpreter`. Read this whole file before changing
> anything. ff4 stencil rewrite is dead. Most of the original
> seam-swap migration is in place; what's left is **codegen-size
> compression** (the per-arch `Weights` struct + accessors + load
> body, plus residual cleanup) and end-to-end golden verification.

## Where we are

Last commit on `ff-interpreter`: **seam swap + coloring + Op::Loop
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

### 1. Weights / load_with compression — biggest remaining lever

~50% of the post-Op::Loop expansion is the per-arch Weights struct
+ accessor methods + `load_with` body, none of which is layer-
compressed yet. Specifically for commandr:

- `pub struct Weights`: 217 lines (40 per-layer fields × 5+ bases).
- `impl Weights { pub fn input_layernorm(&self, layer: u32) … }`:
  578 lines (40-arm `match layer` per layered accessor).
- `pub fn load_with`: ~1320 lines (one `let input_layernorm_<L> =
  …::load(gw, "model.layers.<L>.input_layernorm", eps)?;` per
  layer, per accessor).

The compression: per-layer fields → `Vec<T>` per base. Accessor
methods become `&self.<base>[layer as usize]` (one line). `load_with`
emits one `(0..NUM_HIDDEN_LAYERS).map(|layer| { format!(prefix);
…::load(…) }).collect::<Result<Vec<_>>>()?` per layered accessor
group instead of N per-layer lets.

User asked for this explicitly: "shouldn't [the layered fields]
just be slices, so that there are fewer variable declarations, and
no match garbage like this?"

Implementation outline (substantial — ~200 lines of macro-side
code):
- Add a helper `group_accessors_by_base(accessors)` that returns
  `Vec<AccessorGroup>` where `AccessorGroup { base, rust_type,
  layered: bool, entries: Vec<(Option<u64>, WeightAccessor)> }`.
- In `emit_weights_struct`:
  - Replace the per-accessor field decl loop with a per-group loop
    that emits `pub <base>: Vec<<T>>` for layered groups, `pub
    <base>: <T>` for unindexed.
  - Replace the per-accessor `lets` loop with a per-group loop. For
    layered: emit a Vec-build expression. For unindexed: keep the
    existing `let <name> = <load_call>(…)?;`.
  - The Vec-build needs to call the same `Fp8Linear::load` /
    `MarlinLinear::load` / etc., but with a runtime-formatted
    prefix string. Add a helper `emit_layered_load_expr(plan:
    &FieldLoad, source_weights: &[(WeightId, Option<u64>)],
    program: &Program) -> TokenStream` that returns a
    `TokenStream` of the form `(0..N).map(|layer| { let prefix = …;
    SomeKernel::load(gw, &prefix, …) }).collect::<Result<…>>()?`.
- In `emit_weights_accessor_methods`: drop the 40-arm match;
  layered methods become `&self.<base>[layer as usize]`,
  unindexed become `&self.<base>`.

The trickiest piece is `emit_layered_load_expr` — every FieldLoad
variant has a slightly different signature (different params,
different source-weight count semantics). Best path: factor a helper
that returns `TokenStream` for the per-iteration load call given a
`layer` ident in scope, and lift the existing per-FieldLoad arms
to use it.

The test suite should grow with this — invariants to add:
- "layered fields appear as `pub <base>: Vec<T>`, not as
  `pub <base>_0..<N>: T`."
- "load body has exactly ONE expression per layered base, not N."
- "accessor method body is `&self.<base>[layer as usize]`, not a
  match."

Estimated savings: ~2000 lines on commandr (4.2K → ~2.2K).

### 2. End-to-end build verification

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

### 3. Goldens

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

### 4. Open warnings / cleanup

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

### 5. Optional follow-ups

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
