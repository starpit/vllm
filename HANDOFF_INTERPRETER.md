# ff-interpreter — handoff

> Branch `ff-interpreter`, off `ferrite-forward@961188f5c`. ff4
> stencil rewrite is dead. Read this whole file before making
> changes.

## Goal

Replace ferrite-forward's per-(variant × workload-point) inlined
forward fns with **one per-arch interpreter** walking a
`&[<Arch>Op]` slice. Same solver, same Impl set, same kernel
calls, same outputs — only the macro's *backend* changes. **Host
interpreter ONLY**; megakernel codegen is a separate code
generator we will write later. No `[i32;32]` wire format, no KVM
opcode numbering, no shared opcode registry, no `Noop` SM
padding.

## Architecture invariants — non-negotiable

1. **Only one piece of executable code is generated per arch — the
   interpreter fn** (`for op in slice { match op { … } }`).
   Everything else the macro emits is *data*: the `Weights`
   struct + accessor methods, the per-arch `<Arch>Op` enum, the
   `static FORWARD_M_<N>: &[<Arch>Op] = &[…]` slices. Per-Impl
   `interpreter_arm(model)` produces the body of one arm of the
   interpreter's match.
2. **Each Impl owns its opcode shape.** No universal opcode enum,
   no central registry, no string lookup. The macro collects
   `OpcodeShape`s from the Impls the solver picked for one arch
   and mints that arch's enum from them. Two arches that share
   an Impl share its variant; two arches that diverge get
   disjoint variants.
3. **The match is closed and exhaustive over the per-arch enum.**
   No `_` arm. No `unsafe { unreachable_unchecked() }`. No
   `transmute<u16, Op>`, no `from_wire_unchecked`. The slice is
   `&[<Arch>Op]`, not `&[Instruction]`.
4. **Identical behavior is preserved.** Reshape stays metadata-
   only zero-copy. In-place consume kernels stay in-place.
   Aliasing stays aliasing. The only change is *how* the macro
   spells the same operations.
5. **The work isn't writing new impls.** Every existing `impl
   Implementation for <Foo>` already wires its kernel call
   through `emit_call`. Migration adds three methods —
   `opcode_shape` / `fan_out` / `interpreter_arm` — that move
   the same call into the data + interpreter form. `emit_call`
   stays alive until the seam swap, then dies.

## Wholesale rule

No piecemeal Impl migration, no transitional `interpreter` flag,
no `emit_call` / `fan_out` coexistence at the seam. The
wholesale commit migrates every Impl that participates in a
real arch's forward, swaps the seam, deletes the dead emission
machinery, and runs goldens — all in one commit. Half-migrated
trait defaults that `compile_error!` are scaffolding so the new
methods land before the switch.

## Things that must not happen

- Universal opcode enum / registry / mirror.
- `_` arm or `unsafe { unreachable_unchecked() }` in the
  generated match.
- `transmute<u16, Op>` / `from_wire_unchecked`.
- KVM `[i32;32]` wire format leaking into `ferrite-forward`.
- Piecemeal Impl migration. Wholesale or not at all.
- Solver / loader / fingerprint / region-former edits. Backend
  change only.
- Megakernel work before the host interpreter lands one full
  llama forward and matches its golden.

## Runtime types in `ferrite-forward`

**Keeps:**
- `TileEntry::{ Owned(OwnedTensor), View { ref_slot: u32 },
  Reshaped { ref_slot: u32, tensor: GpuTensor } }` —
  three variants, no more.
- `tile_ref(tiles, idx) -> &TileEntry`,
  `take_owned(tiles, idx) -> OwnedTensor`.
- `ForwardCtx`, the dispatcher, fingerprint plumbing.

**Already deleted (in `afb6a804f`):**
- `Instruction([i32;32])`, `INTS_PER_INSTRUCTION`, `Layout`,
  `pub mod opcode`. No megakernel scaffolding in-tree.

`TileEntry::Reshaped` is the only runtime-types extension this
pivot adds. See "Reshape encoding" below for why.

## Locked Implementation trait shape

```rust
trait Implementation {
    fn opcode_shape(&self) -> OpcodeShape;
    fn fan_out(
        &self, m: &MatchInfo, fuf: &Fuf, program: &Program,
        bounds: &BTreeMap<String, u64>, slots: &SlotMap,
    ) -> Option<Vec<OpInstance>>;
    fn interpreter_arm(&self, model: &ModelParams) -> TokenStream;
    // … plus the existing layout/cost/handoff/etc. methods.
    // emit_call survives until the seam swap, then is removed.
}
```

- `OpcodeShape`: variant ident + ordered `(field_ident,
  field_type: syn::Type)`. The macro asserts shape stability
  (two Impls with the same variant ident must declare identical
  fields).
- `OpInstance`: variant ident + ordered `Vec<TokenStream>` of
  field-value tokens (positional, in shape declaration order).
  Tokens must be const-eval-friendly (they end up in a `static`
  initializer).
- `interpreter_arm(model)` body: kernel call wrapped as
  `<Arch>Op::<Variant> { #(field_idents),* } => { #body }`.
  Body sees the variant's field idents plus ambient bindings
  `__tiles`, `wm`, `ctx`, `device`. Use `model` to bake
  arch-wide config-derived literals (`hidden_size`, `head_dim`,
  attention scale/softcap, …); per-claim values ride in
  `OpInstance` fields.

## Per-claim weight selection — fn-pointer fields

Two RmsNorm tiles in one arch claim different accessors
(`input_layernorm` vs `post_attention_layernorm`). The variant
body can't hardcode an accessor name; it gets told which weight
to read at this instance via a `weight_fn` field of type
`for<'a> fn(&'a Weights, u32) -> &'a Ty`. `fan_out` resolves
the value as `Weights::<base>` (a function-item reference,
const-eval-friendly); the arm calls `(weight_fn)(wm, layer)`.

Per-arch `Weights` codegen (already landed in `833ddcb61`)
emits `pub fn input_layernorm(&self, layer: u32) -> &RmsNorm`
per accessor base (trailing `_<digits>` stripped). Un-layered
accessors take the same signature for caller uniformity; their
body is `&self.<field>` and ignores `layer`.

`split_base_layer(name) -> (base, Option<u32>)` (in
`codegen.rs`, `pub(crate)`) is what `fan_out` calls to derive
the variant's `weight_fn` ident + `layer` value from a
`WeightAccessor.name`.

## Reshape encoding — instructions are data

Today's `emit_call` for Reshape produces
`(input).reshape(&[d0, d1, …])` — a `TensorView`/`GpuTensor`
with new shape metadata sharing storage with upstream. Zero
copies. The interpreter must reproduce this exactly, but
`View { ref_slot }` alone carries no shape override, so we
needed:

- New runtime variant `TileEntry::Reshaped { ref_slot: u32,
  tensor: GpuTensor }`. `tensor` carries new shape/ndim with
  the same ptr; `ref_slot` keeps the storage owner alive
  (drop-pass already aliases Reshape's output to its upstream
  via `output_alias`).
- Opcode fields `dims_lit: [u32; MAX_DIMS]`, `dims_nt_pow:
  [u8; MAX_DIMS]`, `ndim: u8`. Each axis's final dim is
  `dims_lit[i] * num_tokens.pow(dims_nt_pow[i])`.
- `decompose_reshape_dim(dim, bounds) -> (lit, nt_pow)`:
  evaluates each Dim at codegen time, folding `Lit` and every
  non-`num_tokens` `Bound` into the literal factor and counting
  `Bound("num_tokens")` occurrences into `nt_pow`. `Mul`
  recurses; `Var` panics.
- The arm reads `(*ctx.input_ids).dim(0)` for `num_tokens`,
  composes the shape, calls `upstream.reshape(&shape[..ndim])`,
  and writes `TileEntry::Reshaped { ref_slot: in_slot, tensor:
  __reshaped }`.

This is the template for any future Impl whose runtime
behavior depends on a closed-form expression over the bound
table plus `num_tokens`.

The proc-macro crate doesn't depend on `ferrite-cuda-core`, so
host-side array allocation in `fan_out` hardcodes `4` with a
comment; emitted tokens reference
`::ferrite_cuda_core::tensor::MAX_DIMS` symbolically (visible
in consumer crates).

## What's done (uncommitted, working tree)

**43 of 49 Impls migrated.** 6 Fp8 Impls deferred (see "Fp8
deferral" below). Macro crate builds clean; existing 5 + 11
new round-trip tests pass; codegen-profile line counts
unchanged in `ferrite-models` (active `emit_call` path
unperturbed).

### Migrated (43):

Singletons + simple fusions:
EmbedRef, RmsNormRef, LayerNormRef, GemmRef, ReshapeRef,
ScalarMul, TanhSoftCap, AddRef, ScalarOffsetRmsNorm.

Two-tile + multi-tile fusions:
FusedGemmBias, CutlassFusedGemmBias, FusedGateUpSiluMul,
CutlassFusedGateUpSiluMul, FusedGateUpGeluMul,
FusedAddRmsNorm, FusedAddRmsNormWithOffset.

QKV/rope/attention cluster:
FusedQkvRopeCache, FusedQkvQkNormRopeCache,
AttentionViaCache, RopeAppendRef, FusedQkvRopePrefill,
AttentionPrefillContiguous, SlidingAttentionViaCache,
SlidingAttentionPrefillContiguous.

Cutlass GEMM tile zoo (one variant per family — `tile_m`,
`tile_n`, `stages`, `split_k` ride as runtime fields):
CutlassGemm, CutlassGemmSplitK, CutlassGemmAdd, CutlassGemv.

Marlin (AWQ/GPTQ) family:
MarlinGemm, MarlinFusedGateUpSiluMul, MarlinFusedGateUpGeluMul,
MarlinFusedQkvRopeCache, MarlinFusedQkvRopePrefill.

Bnb4 family:
Bnb4Gemm, Bnb4FusedGateUpSiluMul, Bnb4FusedGateUpGeluMul,
Bnb4FusedQkvRopeCache, Bnb4FusedQkvRopePrefill.

FlashInfer + DeepSeek-V2:
FlashInferAttentionDecode, FlashInferAttentionPrefill,
MlaSplitRef, MlaAttention, DeepSeekMoeRef.

### Fp8 deferral (6):

Fp8Gemm, Fp8FusedGemmBias, Fp8FusedGateUpSiluMul,
Fp8FusedGateUpGeluMul, Fp8FusedQkvRopeCache,
Fp8FusedQkvRopePrefill.

**Why deferred:** `fp8_accessor_type_for(fuf, tile)` returns
either `Fp8Linear` or `Fp8BlockLinear` per claim depending on
storage block_size. The interpreter trait gives
`opcode_shape(&self)` no model/FUF context, so a single Impl
can't choose its variant's `weight_fn` field type per claim.

**Unblocker:** introduce `pub enum Fp8AnyLinear { Std(Fp8Linear),
Block(Fp8BlockLinear) }` in `ferrite-kernels::layers` with a
unified `unsafe fn forward(&self, x, cublas, alloc, stream)
-> OwnedTensor` that dispatches on the variant. Update
`fp8_accessor_type_for` to always return
`::ferrite_kernels::layers::Fp8AnyLinear`. Update the kernels'
loaders (`Fp8Linear::load*`, `Fp8BlockLinear::load*`) to wrap
their result in `Fp8AnyLinear::Std` / `::Block` so the
auto-generated `<arch>::Weights::load_with` body still type-
checks. With one concrete `weight_fn` type, the six Fp8 Impls
migrate identically to their Bnb4 siblings (`Bnb4bitLinear`
template). No interpreter trait changes needed.

**Active-path impact:** today's `emit_call` still produces the
right code for Fp8 because `fp8_accessor_type_for`'s decision
flows through into the emitted Weights field type. The
deferral is purely about the new interpreter path.

**Wholesale-commit gating:** the seam-swap commit cannot land
while Fp8 Impls are unmigrated AND any registered arch picks
one. None of the goldens we gate on (Llama/Qwen2/Qwen3/Mistral/
Phi3/Gemma2/Gemma3/Granite/CommandR/DeepSeek-V2/DeepSeek-V3)
use Fp8 weights, so the seam swap can land WITHOUT migrating
Fp8 — `lower_bucket` panics on `fan_out → None` only when an
Impl is actually picked. The Fp8 Impls stay registered in
`starter_library` and remain reachable through the legacy
emit path until the wrapper-enum commit lands.

Both `trivial_impl!` callsites are gone (Embed, RmsNorm,
LayerNorm, Gemm rewritten longhand) and the macro definition
itself is now deleted — the wholesale cleanup pass only has to
remove the `Implementation::emit_call` trait method, the
per-Impl `emit_call` bodies, `EmitMode::Abstract`,
`FragmentLibrary`, and Concrete-mode `EmitCtx`.

`SlotMap`, `OpcodeShape`, `OpInstance`, and the new trait
methods are in `impl_lib.rs`. `interpreter_codegen.rs` has
`build_slot_map`, `ArchOpcodes`, `lower_bucket`,
`emit_bucket_static_slice`, `free_variant_shape`,
`free_instance` — fully unit-tested but **not wired** into the
active codegen path. The active path is still
`emit_subgraph → emit_call`.

`codegen.rs` was extended with one helper: per-arch `Weights`
now exposes accessor methods `rotary_cos_sin(&self, _layer:
u32) -> GpuTensor` and (when applicable) `rotary_local_cos_sin`,
emitted alongside `accessor_methods` in canonical mode only.
Interpreter arms reference these via a `cos_sin_fn` fn-pointer
field instead of `wm.rotary_local.cos_sin_cache` — the latter
fails to type-check on Llama (no `rotary_local` field).

171/174 macro-crate tests pass. The 3 failures
(`load_real_llama_configs`, `load_real_qwen2_configs`,
`add_rmsnorm_pairs_claimed_as_fused_subgraph`) predate this
branch and are out of scope.

## Per-Impl migration recipe

For each `impl Implementation for <Foo>`:

1. Read `emit_call`. Identify:
   - **Tile inputs** (read via `ctx.input_expr(tile, slot)`) →
     `<name>_slot: u32` field. Resolve via
     `slots.of(producer_tile, producer_slot)` from
     `node.inputs[i]` (read directly off the FUF —
     `MatchInfo.boundary_inputs` loses producer-slot info).
   - **Weights** (read via `ctx.weight_accessor(name)`) →
     `weight_fn: for<'a> fn(&'a Weights, u32) -> &'a <Ty>`
     field, value `Weights::<base>` from
     `split_base_layer(acc.name)`. Add a sibling `layer: u32`
     field for the layer index. Multiple-weight Impls carry
     one fn-pointer + layer pair per accessor.
   - **Outputs** (bound via `ctx.output_ident(tile, slot)`) →
     `<name>_slot: u32` field. Body writes
     `__tiles[<name>_slot as usize] =
     Some(::ferrite_forward::TileEntry::Owned(...))` (or
     `::Reshaped{…}` for reshape-shaped Impls; `take_owned` +
     reinsert for in-place consume).
   - **Per-claim compile-time constants** that vary across
     instances of the same variant (layer, scalar offsets,
     interleaved-vs-NeoX flag, fp8 flag) → fields of the
     appropriate type.
   - **Arch-wide compile-time constants** (every `ctx.bound`,
     `ctx.scalar`) → bake into the body via
     `model.bounds["…"]` / `model.scalars.get("…")` reads in
     `interpreter_arm(model)`. Same as today.
2. Write `opcode_shape()` returning the variant ident + ordered
   `(field_ident, field_type)` pairs.
3. Write `fan_out(m, fuf, program, bounds, slots)` returning
   one `OpInstance` per kernel call. Field-value tokens are
   positional, matching declaration order.
4. Write `interpreter_arm(model)`. Body destructures the
   variant's fields and calls the kernel. Read tile inputs via
   `tile_ref(__tiles, slot).as_view(__tiles)` (or
   `as_gpu_tensor` for kernels that take `GpuTensor`).
5. Leave `emit_call` in place (it's still the active path).
6. Add a `<impl>_opcode_shape_…` round-trip test in
   `impl_lib::tests` mirroring the five existing ones —
   variant name + field names + arm body assertions + parse
   through syn + `ArchOpcodes::emit_enum` /
   `emit_interpreter` round-trip.

After every batch of 5–10, run `cargo build -p
ferrite-forward-macro`. After all migrations, run `cargo build
-p ferrite-models --features cuda`.

## Impl inventory

49 total. **Migrated (43)** — all live arch goldens covered.
**Deferred (6)** — Fp8 family, see "Fp8 deferral" above.

## What's left for the wholesale commit

1. **0 Impls remaining** for live-arch goldens. (Fp8 family
   skipped — not picked by any Llama/Qwen/Mistral/Phi3/Gemma/
   Granite/CommandR/DeepSeek arch; lands in a follow-up that
   introduces `Fp8AnyLinear`.)
2. **Seam swap** in `codegen.rs`: rewrite `emit_subgraph` /
   `emit_forward_for_bucket` /
   `emit_forward_backbone_for_bucket` / `emit_model` to use
   `interpreter_codegen::lower_bucket` +
   `ArchOpcodes::emit_enum` + `ArchOpcodes::emit_interpreter` +
   `emit_bucket_static_slice`. Per-arch: ONE enum + ONE
   interpreter helper. Per-bucket: one static slice + a thin
   fn that runs the alias prelude and calls the helper.
3. **Delete dead emission machinery**: `EmitMode::Abstract`,
   `FragmentLibrary`, Concrete-mode `EmitCtx`, the per-Impl
   `emit_call` bodies, and the `Implementation::emit_call`
   trait method. (`trivial_impl!` is already gone.)
4. **Run goldens**: llama first (`vllm-e2e --features e2e,cuda
   --release --test e_correctness -- --ignored
   --test-threads=1`, llama subset). Match must be exact. Then
   sibling-arch goldens
   (qwen2/qwen3/mistral/phi3/gemma2/gemma3/granite/command-r/
   deepseek-v2/deepseek-v3). Same bar.

Per the wholesale rule, none of this commits until the seam
swap + golden runs are in the same commit as the Impl
migrations.

## Fp8 follow-up commit (after wholesale)

A separate commit lands the Fp8AnyLinear unblocker:

1. Add `pub enum Fp8AnyLinear { Std(Fp8Linear),
   Block(Fp8BlockLinear) }` to `ferrite-kernels::layers` with
   `unsafe fn forward(&self, x, cublas, alloc, stream)` that
   dispatches on the variant.
2. Update Fp8 loaders to wrap into `Fp8AnyLinear`.
3. Change `fp8_accessor_type_for` to always return
   `::ferrite_kernels::layers::Fp8AnyLinear`.
4. Migrate the 6 Fp8 Impls following the Bnb4 family template
   (one variant per family, fixed `weight_fn` type
   `&Fp8AnyLinear`).
5. Run the Fp8 golden subset.

Constraint: between the wholesale commit and the Fp8 follow-up,
any consumer model that picks an Fp8 Impl will hit
`lower_bucket`'s "Impl X has no fan_out" panic — the seam-swap
commit moves through the same panic site. Confirm the active
goldens don't pick Fp8 before landing the wholesale commit.

## Pre-commit checklist

Re-read this file. Verify:

- [ ] No universal opcode enum / registry / string lookup.
- [ ] No `_` arm or `unsafe { unreachable_unchecked() }` in
      any emitted match.
- [ ] No megakernel wire-format types in `ferrite-forward`.
- [ ] No `interpreter` flag, no `emit_call` / `fan_out`
      coexistence at the seam.
- [ ] The diff doesn't reach `form_regions.rs`,
      `library_invariants.rs`, `solver/`, or any loader.
- [ ] Every migrated Impl has a `<impl>_opcode_shape_…` test.
- [ ] `cargo build -p ferrite-forward-macro` and `cargo build
      -p ferrite-models --features cuda` both green.

If any box is unchecked, **stop and wait** — do not commit.

## Reference points

- `vllm-rs/crates/ferrite-forward-macro/src/codegen.rs::emit_subgraph`
  — the seam to rewrite. (Earlier handoffs called this
  `emit_workload.rs`; that file does not exist.)
- `vllm-rs/crates/ferrite-forward-macro/src/impl_lib.rs` — 43
  migrated Impls live here. Embed, RmsNorm, LayerNorm, Gemm,
  Reshape are the original templates; ScalarMul/TanhSoftCap
  show in-place consume; AddRef/FusedAddRmsNorm show
  alias-prelude outputs (no `__tiles[...] = ...` write);
  Reshape/RopeAppend show `Reshaped` entries that overwrite
  the alias prelude. The `Implementation` trait + `OpcodeShape`
  / `OpInstance` / `SlotMap` types live at the top of the file.
- `vllm-rs/crates/ferrite-forward-macro/src/interpreter_codegen.rs`
  — `lower_bucket`, `ArchOpcodes`, `emit_bucket_static_slice`,
  `free_variant_shape`. Unwired until the seam swap.
- `vllm-rs/crates/ferrite-forward/src/tile_table.rs` — the
  three `TileEntry` variants and their accessors.
