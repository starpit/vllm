# ff-interpreter — handoff

> Read this first. This branch (`ff-interpreter`) is a **pivot off
> `ferrite-forward`** at commit `961188f5c`. ff4 (the stencil
> rewrite) is dead. The existing `HANDOFF.md` next to this file is
> the inherited ferrite-forward handoff; keep it for context, but
> the work happening *now* is what's described below.

## Scope of this refactor — host interpreter ONLY

This refactor is the **host-based interpreter** code generator.
Megakernel codegen is a separate code generator that we will
write later. Do not let megakernel concerns leak into the host
design — no `[i32; 32]` packed wire format, no KVM tp_throughput
opcode-numbering compatibility, no per-SM padding (`Noop`), no
shared opcode registry. Those belong in the megakernel codegen
when we build it.

The host interpreter is allowed to use whatever shape Rust makes
natural: typed enums with destructured payload, exhaustive
matches, no unsafe transmutes, no wire numbers in user-visible
code.

## The actual work

ferrite-forward already codegens the entire forward pass for every
(variant × workload-point): solver picks Impls, codegen walks
them, each Impl's `emit_call` produces a `let tile_X =
kernel_call(…)` Rust statement, and the macro stitches them into
a fully-unrolled forward fn. **Replace that codegen wholesale.**
Same solver. Same Impl set. Same kernel calls. The macro now emits:

1. **One per-arch enum**, codegened from the FUF the solver
   actually solved for that arch. Variants = exactly the kernel
   calls this arch's picked Impls produce, plus `Free` (drop-pass).
   Each variant carries typed payload fields (slot indices,
   layer index, etc.) — no `[i32; 32]` packing.
2. **One `static FORWARD_M_<N>: &[<Arch>Op] = &[…];` per (variant ×
   workload-point)** — each element is a per-arch enum value. No
   wire conversion at runtime; the slice is the program.
3. **One per-arch interpreter** — `for op in slice { match op { … }
   }`. The match is **closed and exhaustive** over the per-arch
   enum the macro just codegened. No `_` arm. No `unsafe`
   transmute. No `from_wire_unchecked` shenanigans.

Solver, Impl library, library invariants, fingerprinting, weights
loading, dispatcher — all unchanged. Only the *backend* of the
macro changes.

## Failure pattern this handoff explicitly rejects

> "Migrate one Impl as proof-of-concept. Then the next Impl. Then
> the next."

That is **the wrong shape**. There is no PoC layer to prove. The
existing codegen already proves every kernel call works. The work
is one wholesale change to the macro: every Impl that participates
in a real arch's forward gets its emission shape changed at the
same time. Half-migrated trait defaults that `compile_error!` at
codegen are scaffolding — they exist so the new methods can land
before the refactor is done, not so we ship a half-migrated tree.
**No transitional flag.** No "interpreter = true" opt-in. No
auto-detect-fallback. No coexistence of `emit_call` with the new
path. Wholesale or not at all.

## Locked design

### Opcodes are owned by Impls, not by a registry

There is **no universal opcode enum, no central registry, no
string lookup**. Each Impl declares its own opcode shape: the
variant name (PascalCase ident) + the typed payload fields. The
macro, processing one arch's solved FUF, collects the shapes from
the Impls the solver picked for that arch. From those shapes it
codegens a per-arch enum:

```rust
enum LlamaOp {
    AttnNorm { layer: u32, in_slot: u32, out_slot: u32 },
    QkvRopeAppend { layer: u32, in_slot: u32, out_q_slot: u32 },
    // … only the variants Llama's picked Impls declared
    Free { slot: u32 },
}
```

Two arches that happen to share an Impl get the same variant in
their respective enums (the Impl's `opcode_shape()` is one
function call, deterministic). Two arches that diverge get
disjoint variants. There is no need to reconcile across arches —
each arch has its own enum.

### Per-arch interpreter is a closed exhaustive match

```rust
for op in FORWARD_M_64 {
    match op {
        LlamaOp::AttnNorm { layer, in_slot, out_slot } => { /* arm body */ }
        LlamaOp::QkvRopeAppend { layer, in_slot, out_q_slot } => { /* arm body */ }
        // … one arm per variant in LlamaOp …
        LlamaOp::Free { slot } => { __tiles[*slot as usize] = None; }
    }
}
```

No `_` arm. No catch-all. No unsafe. The compiler enforces
exhaustiveness over `LlamaOp`. The const slice is `&[LlamaOp]`,
so by construction every element is a valid variant.

### `Free` is the only memory-management opcode

The drop-pass already computes last-reader per (tile, slot).
Where the old codegen would emit a `drop(local)`, the new codegen
emits `<Arch>Op::Free { slot }` at that point in the slice. The
arm body clears `__tiles[slot] = None`, dropping the
`OwnedTensor` (or the `View` indirection) and returning GPU
memory to the caching allocator.

`Free` is universal — every arch's enum has it, codegened by the
macro itself, not by any Impl.

### View aliasing is a fn-entry prelude, not a runtime opcode

When a claim aliases an upstream tile (today's `as_view()`
borrow), the macro emits

```rust
__tiles[dst as usize] = Some(TileEntry::View { ref_slot: src });
```

at fn entry, before the interpreter loop. Setting it before the
owner is written is fine: reads through `tile_ref` resolve the
View only when the variant arm runs, by which time the schedule's
owner-before-reader invariant has placed the owner.

No `View` variant. No runtime "alias setup" opcode.

### Tile-table is `Vec<Option<TileEntry>>` indexed by u32 slot

The runtime tile table replaces the let-bindings the old codegen
emitted. Codegen builds a dense `(TileId, output_slot) → u32`
map (call it `SlotMap`) per (variant × workload-point); slot
indices appear in the per-arch enum's payload fields and as the
table index at runtime. `TileEntry::{Owned, View { ref_slot:
u32 }}` and `tile_ref(tiles, idx) -> &TileEntry` live in
`ferrite-forward` as runtime types.

### Granularity: one slice per (variant × workload-point)

Matches today's `forward_m_<N>[_sk_<SK>]` granularity. Elevating
to one slice per variant (with workload-point selecting a
subrange or parameterizing fields at runtime) is a real
possibility for the megakernel codegen but **not in scope here**.

### Implementation trait shape

```rust
trait Implementation {
    /// Shape of this Impl's opcode — variant name + typed fields.
    /// Macro collects these from picked Impls to mint per-arch enum.
    fn opcode_shape(&self) -> OpcodeShape;

    /// One opcode-instance per kernel call this Impl makes at this
    /// (variant × workload-point). Field values resolved against
    /// the SlotMap (and bounds, ctx hints, …). Returns None if
    /// unmigrated — codegen errors with the Impl name.
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>>;

    /// The body of this Impl's match arm. Macro wraps as
    ///     <Arch>Op::<Variant> { #(field_idents),* } => { #body }
    /// using the field names declared in `opcode_shape`. The body
    /// references those idents, plus the ambient bindings
    /// `__tiles`, `wm`, `ctx`, `device`. `model` lets the body bake
    /// arch-wide config-derived literals (`hidden_size`, `head_dim`,
    /// `attention_multiplier`, `attn_logit_softcapping`, …) at
    /// codegen time; per-claim values that vary across instances of
    /// the same variant ride in `OpInstance` fields instead.
    fn interpreter_arm(&self, model: &ModelParams) -> TokenStream;
}
```

`OpcodeShape`: variant ident + ordered list of `(field_ident,
field_type: syn::Type)`. `OpInstance`: variant ident + ordered
list of field-value `TokenStream`s, in the same order as the
shape. The macro asserts shape stability (two Impls returning the
same variant ident must declare identical fields).

### What `ferrite-forward` runtime keeps / drops

**Keeps** (real runtime types the emitted interpreter uses):
- `TileEntry::{Owned(OwnedTensor), View { ref_slot: u32 }}`.
- `tile_ref(tiles, idx) -> &TileEntry`.
- `ForwardCtx`, the dispatcher, fingerprint plumbing.

**Drops** (megakernel concerns we shoehorned in earlier — not
needed by the host):
- `Instruction([i32; 32])` packed wire type.
- `INTS_PER_INSTRUCTION` constant.
- `Layout` (matmul block sizes etc. — megakernel fan-out info).
- `pub mod opcode` runtime constants — emitted code references
  per-arch enums, never wire numbers.

These deletions land in the same commit that introduces the
per-arch enum codegen, so the runtime crate never carries dead
megakernel scaffolding.

## Per-claim weight selection: fn-pointer fields

One Impl emits one variant. But two RmsNorm tiles in the same arch
claim *different* accessors (`input_layernorm` vs
`post_attention_layernorm`). The variant body can't hardcode an
accessor name; it has to be told which weight to read at this
instance. The encoding:

- Per-arch `Weights` codegen (now landed) emits
  `pub fn input_layernorm(&self, layer: u32) -> &RmsNorm { match
  layer { 0 => &self.input_layernorm_0, … } }` per accessor base
  (trailing `_<digits>` stripped). Non-layered accessors take the
  same `(&self, layer: u32) -> &Ty` signature for caller
  uniformity; their body is `&self.<field>` and ignores the layer.
- `OpInstance` carries a `weight_fn` field of type
  `for<'a> fn(&'a Weights, u32) -> &'a Ty` (or whatever return
  type fits the Impl). `fan_out` resolves it as
  `Weights::input_layernorm` (a function reference, valid in
  `const` context). The interpreter arm body calls
  `(weight_fn)(wm, layer)`.
- Multiple-weight Impls (e.g. fused QKV with separate qkv +
  rotary accessors) carry one fn-pointer field per accessor.

Codegen-time invariants the helper enforces (see
`emit_weights_accessor_methods` + its tests):

- Mixed layered + un-layered fields under one base panic at
  expansion time. There's no sensible single body for the mix and
  silently picking one would mask a solver / Impl bug.
- Two layers under the same base with different `rust_type`s
  panic at expansion time.
- Out-of-range layer at runtime panics with the accessor name
  baked in (no `_ => &self.<layer 0>` catch-all that would silently
  pick layer 0).

## Current branch state (2026-04-25 mid-session)

Branch `ff-interpreter` is at `833ddcb61` (forked from
ferrite-forward `961cb8c8`). Seven commits ahead of fork:

```
833ddcb61  ff-interpreter: Weights accessor methods for runtime layer dispatch
318949840  ff-interpreter: interpreter_arm takes &ModelParams
e7fcae89a  ff-interpreter: handoff captures end-of-session state
b2c08c8e7  ff-interpreter: per-arch enum + closed-match codegen module
f5f62f675  ff-interpreter: add OpcodeShape/OpInstance + new trait methods
afb6a804f  ff-interpreter: drop megakernel scaffolding, rewrite handoff
084828c3f  ff-interpreter: add SlotMap + ferrite-extension opcodes      ← reverted by afb6a804f
4e5174d02  ff-interpreter: scaffold Instruction IR + tile table + InstrEmit  ← reverted by afb6a804f
```

The two reverted commits were wrong-design scaffolding (universal
opcode registry + `Instruction([i32;32])` wire format), undone
by `afb6a804f`. The active scaffolding is the four most recent
commits before today plus today's two scaffolding refinements.

### What's landed and verified

- `ferrite-forward-macro/src/impl_lib.rs`:
  - `OpcodeShape { name: Ident, fields: Vec<(Ident, Type)> }`
  - `OpInstance { name: Ident, field_values: Vec<TokenStream> }`
  - `SlotMap` (compile-time `(TileId, slot) → u32` allocator)
  - Trait `Implementation` gains `opcode_shape` / `fan_out` /
    `interpreter_arm` with unmigrated defaults. **`emit_call` is
    still the active codegen path.** Both methods live on the
    trait simultaneously *as scaffolding* (handoff explicitly
    permits this so the new methods can land before the
    wholesale switch).
  - Tests: `opcode_shape_carries_typed_fields_in_declaration_order`,
    `unmigrated_shape_carries_distinctive_placeholder_name`,
    `op_instance_field_values_match_shape_field_count`,
    `slot_map_assigns_dense_indices_in_insert_order`,
    `slot_map_of_unregistered_panics`.

- `ferrite-forward-macro/src/interpreter_codegen.rs` (NEW):
  - `build_slot_map(fuf) -> SlotMap`.
  - `ArchOpcodes`: collects `(OpcodeShape, interpreter_arm)`
    across an arch's buckets; emits the per-arch Rust enum and
    the per-arch interpreter helper.
  - `LoweredBucket` + `lower_bucket(...)`: walks waves, calls
    `fan_out`, interleaves `Free` instances at drop-pass points,
    builds the alias prelude.
  - `emit_bucket_static_slice(...)`: lowers `Vec<OpInstance>` to
    `static FORWARD_<TAG>: &[<Arch>Op] = &[…];`.
  - `free_variant_shape()` / `free_instance(slot)` helpers — the
    universal `Free { slot: u32 }` variant codegen always emits.
  - Tests: 6, including `arch_interpreter_match_has_no_catchall`
    that asserts the emitted match has neither a `_` arm nor any
    `transmute` / `from_wire`.
  - **NOT WIRED** into the active codegen path. `lower_bucket`
    has zero callers in `codegen.rs`. The active path is still
    `emit_subgraph → emit_call`.

- `ferrite-forward/src/`:
  - `instruction.rs` deleted (megakernel wire format).
  - `tile_table.rs` (TileEntry + tile_ref) intact — runtime types
    the future emitted interpreter will consume.
  - `lib.rs` re-exports `TileEntry` + `tile_ref` only.

- 3 macro-crate test failures predate this branch (config variant
  counts, `add_rmsnorm_pairs_*`); not introduced here.

## What's next — the wholesale switch (one commit)

Scaffolding (steps 1–4 below) is done. The remaining work is one
wholesale commit. Per the rules: no piecemeal Impl migration, no
intermediate "some Impls migrated" tree state.

1. ~~Drop megakernel scaffolding~~ — done in `afb6a804f`.
2. ~~Define `OpcodeShape` / `OpInstance` + new trait methods~~ —
   done in `f5f62f675`. `emit_call` still active alongside the
   defaulted-`None` `fan_out`.
3. ~~`interpreter_arm` takes `&ModelParams`~~ — done in
   `318949840`. Body bakes config-derived literals
   (`hidden_size`, `head_dim`, attention scale / softcap, …) the
   same way today's `emit_call` does via `EmitCtx::bound` /
   `EmitCtx::scalar`.
4. ~~Per-arch `Weights` accessor methods~~ — done in `833ddcb61`.
   `pub fn input_layernorm(&self, layer: u32) -> &RmsNorm` etc.
   emitted alongside the per-(field × layer) fields. `OpInstance`
   `weight_fn` fields will resolve to `Weights::<base>` (`const`
   fn-pointer).
5. **Migrate every Impl in `impl_lib.rs`.** Each Impl: write
   `opcode_shape()` (variant ident + typed fields), `fan_out(...)`
   (one `OpInstance` per kernel call this Impl makes; resolves
   slot ids via `SlotMap::of`, weight selectors via
   `Weights::<base>` fn-pointer), and `interpreter_arm(model)`
   (body that references the shape's field idents + ambient
   `__tiles`, `wm`, `ctx`, `device`). Delete the Impl's
   `emit_call`.
6. **Rewrite `emit_subgraph` / `emit_forward_for_bucket` /
   `emit_forward_backbone_for_bucket` / `emit_model`** to use
   `interpreter_codegen::lower_bucket` + `ArchOpcodes::emit_enum`
   + `ArchOpcodes::emit_interpreter` + `emit_bucket_static_slice`.
   Per-arch: emit ONE enum and ONE interpreter helper; per-bucket:
   emit one static slice + a thin fn that runs the alias prelude
   and calls the helper.
7. **Delete `EmitMode::Abstract` / `FragmentLibrary` /
   Concrete-mode `EmitCtx` machinery** — all unused after the
   seam swap. `emit.rs` shrinks substantially or goes away.
8. **Delete `Implementation::emit_call` from the trait.** The
   final shape: trait has `opcode_shape` / `fan_out` /
   `interpreter_arm` + the layout/cost/handoff/etc. methods that
   were never about emission.
9. **Run llama golden** (`vllm-e2e --features e2e,cuda --release
   --test e_correctness -- --ignored --test-threads=1`, llama
   subset). Match must be exact.
10. **Run sibling-arch goldens** (qwen2/qwen3/mistral/phi3/gemma2/
    gemma3/granite/command-r/deepseek-v2/deepseek-v3). Same bar.

### Per-Impl migration recipe

For each `impl Implementation for <Foo>`:

1. Read `emit_call` carefully. Identify:
   - **Inputs** read via `ctx.input_expr(tile, slot)` → become
     `<name>_slot: u32` fields. Resolution: `slots.of(producer_tile,
     producer_slot)` for tile inputs; `ctx.input_ids` /
     `ctx.positions` etc. stay as ambient `ctx.<field>` references
     in the body (no field needed); `wm.rotary` /
     `wm.rotary_local` become `rotary_fn: fn(&Weights) -> &Rotary`
     fields when the choice is per-claim (Gemma3 mixes both).
   - **Weights** read via `ctx.weight_accessor(name)` → become
     `weight_fn: fn(&Weights, u32) -> &<Ty>` fields, value
     `Weights::<base>` where `<base>` is the accessor name with
     trailing `_<digits>` stripped. The body calls
     `(weight_fn)(wm, layer)`.
   - **Outputs** bound via `ctx.output_ident(tile, slot)` →
     become `<name>_slot: u32` fields (one per claimed output).
     Body writes `__tiles[<name>_slot as usize] =
     Some(::ferrite_forward::TileEntry::Owned(...))`.
   - **Per-claim compile-time constants** that vary across
     instances of the same variant (layer index, scalar offsets,
     interleaved-vs-NeoX flag, fp8 flag) → fields of the
     appropriate type.
   - **Arch-wide compile-time constants** (every `ctx.bound("...")`
     and `ctx.scalar("...")` call) → bake into the body via
     `model.bounds["..."]` / `model.scalars.get("...")` reads in
     `interpreter_arm(model)`. Same as today.
2. Write `opcode_shape()` returning the variant ident + ordered
   `(field_ident, field_type)` pairs.
3. Write `fan_out(m, fuf, program, bounds, slots)` returning one
   `OpInstance` per kernel call. Field-value tokens are positional,
   matching shape declaration order.
4. Write `interpreter_arm(model)`. Body destructures fields and
   calls the kernel. **Never** transmute or use `unsafe { … }`
   wrappers larger than the existing `unsafe` blocks today's
   `emit_call` already wraps. Read tile inputs via `tile_ref(__tiles,
   slot)`; for `as_view()` style use `(*entry).as_view(__tiles)`.
5. Delete `emit_call`. (This breaks compilation while the seam
   still runs the old path; the wholesale commit's last move is
   the seam swap, which makes `emit_call` an unused-method dead
   trait member, then this step deletes the trait method itself.)

### Impl inventory (47 to migrate)

Run `grep -n '^impl Implementation for' impl_lib.rs` to enumerate.
As of `833ddcb61`, ordered by location:

```
trivial_impl! → EmbedRefImpl, RmsNormRefImpl, LayerNormRefImpl,
                GemmRefImpl  (4 entries via the `trivial_impl!`
                macro at impl_lib.rs:929; the macro expands to a
                full `impl` block — convert the macro to also
                accept opcode_shape/fan_out/interpreter_arm
                callbacks, OR hand-write each as a regular impl
                so the new methods are visible.)
ReshapeRefImpl              FusedAddRmsNormImpl
FusedGemmBiasImpl           FusedAddRmsNormWithOffsetImpl
CutlassFusedGemmBiasImpl    ScalarOffsetRmsNormImpl
FusedGateUpSiluMulImpl      FusedQkvRopeCacheImpl
CutlassFusedGateUpSiluMulImpl  FusedQkvQkNormRopeCacheImpl
FusedGateUpGeluMulImpl      AttentionViaCacheImpl
ScalarMulImpl               RopeAppendRefImpl
TanhSoftCapImpl             FusedQkvRopePrefillImpl
AddRefImpl                  AttentionPrefillContiguousImpl
                            SlidingAttentionViaCacheImpl
                            SlidingAttentionPrefillContiguousImpl
CutlassGemmImpl             CutlassGemmSplitKImpl
CutlassGemmAddImpl          CutlassGemvImpl
MarlinGemmImpl              MarlinFusedGateUpSiluMulImpl
MarlinFusedGateUpGeluMulImpl  MarlinFusedQkvRopeCacheImpl
MarlinFusedQkvRopePrefillImpl
Bnb4GemmImpl                Fp8GemmImpl
Fp8FusedGemmBiasImpl        Fp8FusedGateUpSiluMulImpl
Bnb4FusedGateUpSiluMulImpl  Bnb4FusedGateUpGeluMulImpl
Fp8FusedGateUpGeluMulImpl   Fp8FusedQkvRopeCacheImpl
Fp8FusedQkvRopePrefillImpl  Bnb4FusedQkvRopeCacheImpl
Bnb4FusedQkvRopePrefillImpl
FlashInferAttentionDecodeImpl  FlashInferAttentionPrefillImpl
MlaSplitRefImpl             MlaAttentionImpl
DeepSeekMoeRefImpl
```

Within the wholesale commit, work in the same locality order as
`grep` output — keeps the diff readable. After every batch (say,
every 5 Impls), run `cargo build -p ferrite-forward-macro` to
verify the trait's still satisfied; the macro crate builds even
while `lower_bucket` is still unwired, because `emit_call` lives
alongside the new methods until step 7 fires.

### Scope estimate for the wholesale commit

- ~47 Impls in `impl_lib.rs`. Per-Impl change ~80–150 lines net.
  ≈ 5 kloc.
- `emit_subgraph` / `emit_forward_*` / `emit_model` rewrite: ~500
  lines.
- Deletions of `emit.rs::EmitMode::Abstract` /
  `FragmentLibrary` / etc.: ~500 lines removed.

This is one focused multi-day push, not a single sitting. Plan
the session for it.

## Pre-commit checklist (every commit on this branch)

Before each commit on `ff-interpreter`, re-read this handoff and
verify:

- [ ] No universal opcode enum. No central registry. No string
      opcode lookup. Each Impl owns its opcode shape.
- [ ] No `_` arm or unsafe transmute in any emitted match. The
      per-arch enum is closed; the match is exhaustive.
- [ ] No megakernel wire-format types in `ferrite-forward`
      runtime (`Instruction([i32; 32])`, `Layout`,
      `pub mod opcode`).
- [ ] No transitional `interpreter` flag. No `emit_call` /
      `fan_out` coexistence.
- [ ] The diff doesn't reach `form_regions.rs`,
      `library_invariants.rs`, `solver/`, or any loader. This is
      a backend change.
- [ ] Every migrated Impl has `opcode_shape` / `fan_out` /
      `interpreter_arm`; `emit_call` is gone from migrated Impls.

If any box is unchecked, **stop and wait** — do not commit.

## Things that must not happen

- **No universal opcode enum / registry / mirror.** The Impl owns
  the opcode shape. The macro collects shapes per arch. End.
- **No `_` arm or `unsafe { unreachable_unchecked() }` in the
  generated match.** Exhaustive over the per-arch enum or it's
  wrong.
- **No unsafe `transmute<u16, Op>` or `from_wire_unchecked`.**
  The slice is `&[<Arch>Op]`, not `&[Instruction]`.
- **No KVM wire-format leakage.** Megakernel will lower per-arch
  enum to KVM `[i32; 32]` when megakernel codegen lands. Until
  then, the host enum is the program.
- **No piecemeal Impl migration.** Wholesale or not at all.
- **No solver / loader / fingerprint edits.** Backend change only.
- **No megakernel work before the host interpreter lands one
  full llama forward.** The IR isn't proven until the host
  backend matches the golden.

## Reference points

- `vllm-rs/crates/ferrite-forward-macro/src/codegen.rs::emit_subgraph`
  — today's per-tile inlined `emit_call` site. **This is what the
  refactor rewrites.** (The handoff previously called this
  `emit_workload.rs`; that file does not exist.)
- `vllm-rs/crates/ferrite-forward-macro/src/impl_lib.rs` — the
  `Implementation` trait. `emit_call` lives at line ~503 today;
  the new methods replace it.
- `vllm-rs/crates/ferrite-forward/src/lib.rs` — runtime types the
  emitted interpreter consumes. `TileEntry` and `tile_ref` stay;
  `Instruction` / `Layout` / `opcode` get deleted in this refactor.

## Pre-existing baseline failures (not introduced by this work)

These were red on `ferrite-forward` at the fork point and remain
red here. Not blockers:

- `vllm-e2e` ignored quant variants (per inherited `HANDOFF.md`
  "Real gap inventory" section).
- `cargo check --workspace` in the top-level vllm dir trips on
  `mlx-sys` BLAS (per memory `feedback_build_flags.md`); use the
  documented `-p` builds instead.
- 3 macro-crate test-only failures (config variant counts drifted
  from earlier llama / qwen2 expansions; `add_rmsnorm_pairs_*` solver
  count). Not introduced here, not in scope to fix.
