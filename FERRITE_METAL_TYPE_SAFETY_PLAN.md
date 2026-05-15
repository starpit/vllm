# ferrite-metal type-safety cleanup plan

Goal: move ferrite-metal's runtime-failure surface into the type system.
"If it compiles it works" — a fresh session should be able to work
through this end-to-end without further design input.

## Status (2026-05-15, end of session)

| Phase | State    | Commit       | Notes                                                                                   |
|-------|----------|--------------|-----------------------------------------------------------------------------------------|
| 1     | DONE     | `209299aa4`  | Bugs #3 / #6 / #7 closed: LayerId, ArenaSlotIdx, BucketM, NumTokens, ConstSlot threaded |
| 2     | DONE     | `0d667e00a`  | Bug #1 closed: 16 per-kernel constants structs, every slot exhaustively listed          |
| 3     | DONE     | `5b6ab415c`  | Buffer-slot analogue of #1: attention/RoPE/QKV BindingSet structs                       |
| 4     | PARTIAL  | `3f0d3c281`  | Bug #8: MetalKernel trait + ZSTs for attention family only; see "Deferred" below        |
| 5     | DONE     | `f0136ad08`  | Bug #5 closed: PagedKvLayout — `grep` returns one hit (the doc comment)                 |
| 6     | DEFERRED |              | See "Deferred" below                                                                    |
| 7     | DONE     | `808913c6f`  | Bug #3 (axis): MScaleAxis enum replaces bare u8                                         |
| 8     | DEFERRED |              | See "Deferred" below                                                                    |
| +new  | DONE     | `bda0a5506`, `60fb075b3` | BackendCompat&lt;B&gt; — DSL tile / backend Impl mismatches are now compile errors |

`BackendCompat<B>` (commits `bda0a5506` + `60fb075b3`) wasn't in the
original plan; it surfaced from the Qwen2.5-1.5B-Instruct-4bit
garbage-output reproducer. Adds capability flags (`HAS_BIAS_ADD`,
`HAS_MOE`) to `CanonicalParams` derived by the macro from DSL
inspection (`BackendCaps::from_program` walks every `Expr::Call`'s
`OpKind`), and a `BackendCompat<B>::COMPAT_CHECK` const-eval
assertion that fails the build when a backend can't claim every
emitted tile. New `HAS_*` flags accrete via one new match arm in
`scan_expr` + one new `assert!` in `BackendCompat<Metal>` per
capability axis — reactive expansion as the next missing-Impl bug
surfaces. See `crates/ferrite-forward/src/backend_compat.rs`.

## Deferred (low priority, all bug classes either closed or low-risk)

### Phase 4 expansion — type the remaining static-symbol kernels

Currently 4 ZSTs cover `AttentionSdpaPaged{F16,Bf16}` and
`AttentionSteelPaged{F16,Bf16}`. The other static-symbol kernels —
Embed / RmsNorm / FusedAddRmsNorm / RopeAppend / FusedQkvRopeCache /
AttentionViaCache / GatherLastToken / ScatterFirstToLastRow /
SiluMul / AffineEmbed — still use the struct-literal
`LoweredCommand { kernel, library, function, … }` form.

* Catches: library/function/kernel-id drift (bug #8) on these kernels.
* Real-world incidence: zero. The slot-99 omission in steel attention
  was the only historical instance, and it's already typed.
* Cost: ~20 mechanical ZST impls + lowering-arm migration.
* Verdict: skip. Add only if a future bug demonstrates the drift.

### Phase 6 — const-generic tile shapes (steel attention, qmm_t)

Targets bug #9 (template-baked tile shapes vs runtime dispatch).
Steel attention's symbol is hardcoded at
`bq32_bk16_bd128_wm4_wn1_bs16`; the matching dispatch geometry is a
runtime tuple. A second shipping tile config would mean drift risk;
today there's only one config per kernel.

* Stable-Rust path: `const_format::concatcp!` + per-tile-config
  macro-expanded `MetalKernel` impls (one impl per shipping config).
  The session-end uncommitted attempt is in `git reflog`; revert was
  due to macro-syntax errors in the qmv/qmm_t expansion, not a
  fundamental block.
* Real-world incidence: zero — only one tile config ships per kernel.
* Cost: medium — `const_format` needs to be added to workspace deps
  (not present today); macro-expansion-time `MetalKernel` impls per
  shipping config.
* Verdict: defer until a second tile config ships, or until Phase 4
  expansion makes the const-generic mileage payoff.

### Phase 8 — build.rs reflection on `.metal` sources

The Phase 2 / 3 / 4 / 6 structs are hand-mirrors of the kernel
header. Phase 8's pitch is parsing the `.metal` files and asserting
the mirror matches at build time.

* Catches: `[[function_constant(N)]]` added to a `.metal` source
  without updating the Rust struct.
* Real-world incidence: never observed. The struct and the kernel
  header are usually edited in the same commit + visually proximate.
* Cost: ~3 days for regex parsing + a brittle dep on `.metal` syntax.
* Verdict: defer; not worth the maintenance overhead until the kernel
  set grows much further.

### Phase 3 follow-ups (low-binding-count kernels)

RmsNorm / SiluMul / Add / ScalarMul / others with 2–3 bindings stay
on hand-rolled `vec![Binding::…]` — the BindingSet boilerplate would
be larger than the literal it replaces.

## Pickup pointers for a future session

* If a NEW DSL tile starts producing garbage on a backend: add a
  match arm to `BackendCaps::scan_expr` + a flag to `CanonicalParams`
  + an `assert!` to `BackendCompat<That>`. See `bda0a5506` and
  `60fb075b3` for templates. ~5 lines of code per axis.
* If the `FusedAffineQkvRopeCacheWithBias` Impl lands on metal:
  relax the `assert!(!W::HAS_BIAS_ADD, ...)` in
  `backend_compat.rs::BackendCompat<Metal>::COMPAT_CHECK` (and add
  a more specific guard if the new Impl only handles a subset of
  the bias patterns).
* If SwitchGLU lands on metal: same shape for `HAS_MOE`.
* `ferrite-models/Cargo.toml`'s `metal` feature list silently omits
  arches that don't compile under metal (mixtral, qwen2-moe,
  qwen3-moe, phi3, deepseek-*). When their backends are added,
  include them in the list — `BackendCompat<Metal>` will surface
  the next missing Impl with a clear error.



Driven by real bugs hit in the past 12 months that a stronger type
system would have prevented. Each bug is cited inline so future-you
can sanity-check that the proposed fix actually catches it.

## Background — bugs the current type system doesn't catch

1. **Unbound function constant `ATTN_PAGED_DEBUG_MODE` (slot 99)** —
   kernel declared it, production lowering only bound 0..5. Metal left
   slot 99 undefined; if non-zero, kernel hit debug paths and produced
   degenerate output (" pr formal formal..."). Caught only by hand-
   bisecting a chat. Fixed at `b3ddb3b46` by adding `uint(99, 0)` to
   the constants vec. See [[project-metal-attention-kernel-5x-slower]].

2. **`PagedKVBlockLoader` had drifted from `BlockLoaderT`** — same
   load semantics implemented differently, triggered an Apple Metal
   compiler frag-dropping bug in the MMA loop downstream. Fixed at
   `2e3394454` by making `PagedBlockLoaderT` mirror `BlockLoaderT`
   byte-for-byte except for `next()`. See same memo.

3. **`m_scaling.axis` confusion** — bare `u8` (0/1/2), and the dispatch
   tg shape is a bare `(u32, u32, u32)`. Wiring `axis=0` to a tg shape
   whose width is the head-axis (instead of the Q-block axis) would
   silently dispatch garbage. See `worker.rs::scale_tg_for_num_tokens`.

4. **Q layout drift between contig and paged steel kernels** — both
   spell out `[total_q, num_q_heads, head_dim]` in a doc comment; if
   one drifts the cross-kernel bench AGREEs but production diverges.

5. **KV cache stride math repeated in three places** — rope_append,
   sdpa_paged, steel_paged, CPU goldens. Each computes
   `physical_block * kv_blk_stride + kv_head * kv_head_stride + ...`
   from raw multiplies. Drift between them produces hard-to-diagnose
   off-by-N errors.

6. **Layer index vs arena slot vs physical block** — all `u32` / `usize`,
   freely mixable. Cited in the macro's `lower_bucket` (where they're
   distinct concepts that share a type).

7. **Bucket m vs actual num_tokens** — both `u32`, distinguished only
   by variable name. The m_scaling formula `ceil(baseline * n / bm)`
   uses both; reversing them silently produces wrong dispatch counts.

8. **Kernel-identity drift** — `KernelId` enum + library `&'static str`
   + function-name `&'static str` are three independent fields on
   `LoweredCommand`. The lowering emits all three by hand; getting
   function name's BQ=32 wrong while leaving the symbol's `_bq32_` in
   place silently selects the wrong specialization.

9. **Tile shape baked at template time vs runtime dispatch shape** —
   the kernel is compiled with BQ=32 BK=16 etc. as template params,
   but the Rust dispatcher passes `(2, 24, 1)` threadgroups and
   `(128, 1, 1)` threads as bare numbers. Mismatch fails Metal pipeline
   build with a runtime error, not at compile time.

The plan below kills these bug classes in compile-time-checkable order.

## Phases (each independently mergeable)

Phases 1, 2, 5 are the "weekend-sized" wins that catch the highest-
recurrence bugs. Phases 3, 4, 6, 7, 8 are bigger but each one moves
another bug class into the type system.

---

### Phase 1 — Semantic numeric newtypes (~1 day)

**Target bugs:** #3, #6, #7 (m_scaling confusion; layer/slot/block mixing;
bucket-m vs num_tokens).

Add a small newtype module — one struct per semantic numeric kind.
Define in `crates/ferrite-forward/src/interpreter/metal/ids.rs`:

```rust
//! Numeric newtypes for the Metal interpreter. Distinguish kinds of
//! integers that share a type but are not interchangeable.
//!
//! Each newtype is `Copy + Eq + Hash + Debug + From<inner>` so adding
//! one to an existing call site is a one-line wrap, not a refactor.

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct LayerId(pub u8);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ArenaSlotIdx(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct PhysicalBlockIdx(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct LogicalBlockIdx(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SlotInBlock(pub u16);   // [0, BLOCK_SIZE)

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SeqIdx(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct QTokenIdx(pub u32);     // [0, total_q)

/// The static dispatch upper bound (set per bucket). Distinct from
/// `NumTokens` so the `ceil(baseline * n / bucket_m)` axis-scaling
/// math can't reverse its arguments.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct BucketM(pub u32);

/// The actual M of the in-flight forward. Always ≤ the active bucket's
/// `BucketM`.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct NumTokens(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ConstSlot(pub u16);     // function-constant slot index

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct BindingIdx(pub u8);     // buffer(N) binding index
```

Then thread through:
- `scale_tg_for_num_tokens(_, _, num_tokens: u32)` → `NumTokens`. Same
  for `MScaling { bucket_m }` → `BucketM`. This is the most direct
  catch of #3 / #7.
- `RuntimeBindingKind::KvCacheK { layer: u32 }` → `LayerId`. Forces
  every kv-cache lookup to wrap a `LayerId`.
- Same for `BlockTable` indirection sites: typed `LogicalBlockIdx`
  / `PhysicalBlockIdx` so going from one to the other has to go through
  an explicit conversion (block-table lookup).
- `ConstantValue::uint(slot: u16, ...)` → `ConstantValue::uint(ConstSlot, ...)`.

Success criterion: `cargo check -p ferrite-forward -Fmetal` passes,
zero behavior change, every site that confused layer/slot/block at
its call site is now a `LayerId(0)` / `ArenaSlotIdx(_)` literal.

---

### Phase 2 — Typed per-kernel function-constants struct (~1 day)

**Target bug:** #1 (the one we just hit).

Define one struct per kernel family in
`crates/ferrite-forward/src/interpreter/metal/kernels/constants.rs`:

```rust
/// Function constants for `attention_prefill_sdpa_v2_paged_*` and
/// `attention_steel_paged_*`. Exhaustive — adding a slot to the
/// kernel header without adding a field here is a compile error at
/// every call site.
pub struct AttentionPrefillPagedConstants {
    pub head_dim: HeadDim,
    pub num_q_heads: NumQHeads,
    pub num_kv_heads: NumKvHeads,
    pub attn_scale: AttnScale,
    pub block_size: BlockSize,
    pub max_blocks: MaxBlocksPerSeq,
    pub debug_mode: u32,   // steel kernel only; sdpa ignores
}

impl From<AttentionPrefillPagedConstants> for Vec<ConstantValue> {
    fn from(c: AttentionPrefillPagedConstants) -> Vec<ConstantValue> {
        vec![
            ConstantValue::uint(ConstSlot(0), c.head_dim.0),
            ConstantValue::uint(ConstSlot(1), c.num_q_heads.0),
            ConstantValue::uint(ConstSlot(2), c.num_kv_heads.0),
            ConstantValue::float(ConstSlot(3), c.attn_scale.0),
            ConstantValue::uint(ConstSlot(4), c.block_size.0),
            ConstantValue::uint(ConstSlot(5), c.max_blocks.0),
            ConstantValue::uint(ConstSlot(99), c.debug_mode),
        ]
    }
}
```

Existing call sites (`lowering.rs::AttentionPrefillPaged`,
`pipelines.rs::constants_for`, `attention_sweep.rs::paged_constants`,
the test helpers) now use:

```rust
constants: AttentionPrefillPagedConstants {
    head_dim: HeadDim(W::HEAD_DIM),
    num_q_heads: NumQHeads(W::NUM_Q_HEADS),
    // ...
    debug_mode: 0,
}.into(),
```

Repeat for every kernel that takes function constants:
RmsNorm, FusedAddRmsNorm, FusedGateUpSiluMul, FusedQkvRopeCache,
FusedAffineQkvRopeCache, RopeAppend, AttentionViaCache,
AttentionPrefillSdpaPaged, AffineQmv*, AffineQmmT*, AffineEmbed, etc.
About 12 structs. ~30 lines each.

Success criterion: removing a field from `AttentionPrefillPagedConstants`
breaks every call site. Adding a new `[[function_constant(N)]]` to a
kernel must be paired with a field add or the build fails.

---

### Phase 3 — Typed per-kernel bindings (~1.5 days)

**Target bugs:** Same risk class as #1 but for buffer slots. (No actual
bug hit yet; pre-emptive.)

Currently `LoweredCommand::bindings` is `Vec<Binding>` where each
`Binding` carries a numeric `binding_index: u8`. The kernel's
`[[buffer(N)]]` declarations are the source of truth; the lowering
restates them by hand. Forgetting one is silent.

Define one binding-set enum per kernel:

```rust
pub enum AttentionPrefillPagedBindings {
    Output(ArenaSlotIdx),
    Q(ArenaSlotIdx),
    CuSeqlensQ,
    SeqUsedK,
    BlockTable,
    KvCacheK(LayerId),
    KvCacheV(LayerId),
}

impl From<AttentionPrefillPagedBindings> for Binding { ... }

/// All seven required. Exhaustive struct equivalent for "I have to
/// produce all the bindings in one go" sites.
pub struct AttentionPrefillPagedBindingSet {
    pub output: ArenaSlotIdx,
    pub q: ArenaSlotIdx,
    pub kv_layer: LayerId,
}

impl From<AttentionPrefillPagedBindingSet> for Vec<Binding> {
    // CuSeqlensQ/SeqUsedK/BlockTable are runtime-only, always present
    fn from(s: AttentionPrefillPagedBindingSet) -> Vec<Binding> { ... }
}
```

Same pattern for every kernel: define a struct of just the per-kernel-
variable bindings (arena slots + layer ids); the runtime-binding ones
are baked into the conversion.

Success criterion: adding a new `[[buffer(N)]]` to a kernel breaks the
matching `BindingSet` struct's call sites. Per-kernel binding order is
declared once, in the conversion impl.

---

### Phase 4 — `MetalKernel` trait + ZST kernel identity (~2 days)

**Target bug:** #8 (kernel-identity drift).

Today `KernelId` (an enum tag) + `library: &'static str` + `function:
&'static str` are independent fields on `LoweredCommand`. Wrap them
into a single trait:

```rust
pub trait MetalKernel {
    type Constants: Into<Vec<ConstantValue>>;
    type BindingSet: Into<Vec<Binding>>;
    type Dispatch: Into<DispatchShape>;
    const LIBRARY: &'static str;
    const FUNCTION: &'static str;
    const KERNEL_ID: KernelId;   // retained for the dispatch-timing tag
}

pub struct AttentionSteelPagedBf16;
impl MetalKernel for AttentionSteelPagedBf16 {
    type Constants = AttentionPrefillPagedConstants;
    type BindingSet = AttentionPrefillPagedBindingSet;
    type Dispatch = AttentionSteelDispatch;
    const LIBRARY: &'static str = "attention_steel_paged";
    const FUNCTION: &'static str =
        "attention_steel_paged_bf16_bq32_bk16_bd128_wm4_wn1_bs16";
    const KERNEL_ID: KernelId = KernelId::AttentionPrefillSdpaPaged;
}

pub struct AttentionSdpaPagedBf16;
impl MetalKernel for AttentionSdpaPagedBf16 { ... }
```

Then the lowering site reads:

```rust
let cmd = LoweredCommand::for_kernel::<AttentionSteelPagedBf16>(
    constants, binding_set, dispatch,
);
```

— where `LoweredCommand::for_kernel<K: MetalKernel>(...)` is the only
constructor and pulls LIBRARY/FUNCTION/KERNEL_ID from `K`.

Bonus: makes the bench's `cross_check_paged_kernels` typed —
`dispatch::<AttentionSdpaPagedBf16>` vs `dispatch::<AttentionSteelPagedBf16>`
— accidentally swapping at a call site is a compile error.

Success criterion: `LoweredCommand` no longer has raw `library` /
`function` fields. Every dispatch goes through `for_kernel::<K>`.
`KernelId` is now redundant (could be removed in a follow-up, but
keep for the dispatch-timing label which needs runtime-tagged data).

---

### Phase 5 — `PagedKvLayout` newtype kills the stride-math drift (~0.5 day)

**Target bug:** #5 (KV cache stride math repeated in three places).

The layout `[num_blocks, num_kv_heads, BLOCK_SIZE, head_dim]` is the
single most-repeated invariant in the codebase. Centralize it:

```rust
#[derive(Copy, Clone)]
pub struct PagedKvLayout {
    pub num_blocks: u32,
    pub num_kv_heads: u32,
    pub block_size: u16,
    pub head_dim: u32,
}

impl PagedKvLayout {
    /// Byte offset of K[physical_block, kv_head, slot_in_block, dim]
    /// within a paged cache buffer of element size `sizeof::<T>()`.
    pub fn elem_offset(
        &self,
        block: PhysicalBlockIdx,
        kv_head: KvHead,
        slot: SlotInBlock,
        dim: u32,
    ) -> usize { ... }

    pub fn kv_blk_stride(&self) -> u32 { ... }
    pub fn kv_head_stride(&self) -> u32 { ... }
    pub fn per_token_stride(&self) -> u32 { ... }
}
```

Then `cpu_golden::attention_prefill_paged`,
`PagedBlockLoaderT` (in paged_loader.h is C++ but the *consumer* of the
strides in Rust passes them through), the cost-sweep bench, and any
other site that recomputes these strides — all read from `PagedKvLayout`.
A C++-side equivalent header is one mirror struct (`paged_kv_layout.h`)
plus a Rust↔Metal-side equivalence assertion in build.rs.

Success criterion: `grep -r "num_kv_heads \* block_size \* head_dim"`
returns one hit (inside `PagedKvLayout`). Drift between the four
consumers becomes impossible.

---

### Phase 6 — Const generics for kernel tile shapes (~2 days)

**Target bug:** #9 (template-baked tile shapes vs runtime dispatch).

The steel attention kernel is templated on `<T, BQ, BK, BD, WM, WN, BLOCK_SIZE>`.
The Rust side passes those template values as part of the symbol name
(`_bq32_bk16_bd128_wm4_wn1_bs16`) AND as runtime dispatch geometry
(`(M/BQ, num_heads, B)` threadgroups, `(WM*WN*32, 1, 1)` threads).
Drift between symbol and dispatch silently fails at pipeline build —
not at compile time.

Make the tile shape a type-level concept:

```rust
pub struct SteelAttnTile<
    const BQ: u32,
    const BK: u32,
    const BD: u32,
    const WM: u32,
    const WN: u32,
    const BLOCK_SIZE: u32,
>;

pub struct AttentionSteelPaged<T, Tile>(PhantomData<(T, Tile)>);

impl<T: MetalDtype, const BQ, BK, BD, WM, WN, BS> MetalKernel
    for AttentionSteelPaged<T, SteelAttnTile<BQ, BK, BD, WM, WN, BS>>
{
    type Dispatch = SteelAttnDispatch<BQ, WM, WN>;  // baked from Tile
    const FUNCTION: &'static str = const_format::concatcp!(
        "attention_steel_paged_", T::SYMBOL_INFIX,
        "_bq", BQ, "_bk", BK, ..., "_bs", BS,
    );
    ...
}
```

The dispatch type `SteelAttnDispatch<BQ, WM, WN>` carries the matching
shape at the type level. Constructing it computes
`tg = (ceil(M/BQ), n_q_heads, 1)` and
`threads = (WM*WN*32, 1, 1)` — impossible to dispatch with mismatched
geometry.

Success criterion: changing BQ in the kernel header without changing
the Rust tile type fails to compile (symbol name doesn't match) OR
fails at the dispatch with a clear type-level error.

This is the only phase that's load-bearing on stable Rust const
generics features — `concat!` of consts requires `const_format` or
nightly. Worth checking what level of const-generic ergonomics the
crate's MSRV allows.

---

### Phase 7 — Typed dispatch shape + m-scaling axis (~1 day)

**Target bug:** #3 (m_scaling axis confusion).

Replace `MScaling { axis: u8, bucket_m: u32 }` with:

```rust
pub enum MScaleAxis { X, Y, Z }

pub struct MScaling {
    pub axis: MScaleAxis,
    pub bucket_m: BucketM,
}

pub struct DispatchShape<S: DispatchShapeKind> {
    pub threadgroups: TgShape<S>,
    pub threads_per_threadgroup: ThreadCount<S>,
    pub m_scaling: Option<MScaling>,
}
```

The `DispatchShapeKind` trait carries the "what does each axis mean"
metadata at the type level (Q-blocks vs heads vs sequences). Then
`scale_tg_for_num_tokens` becomes typed:

```rust
fn scale_tg(
    tg: TgShape<S>,
    scaling: MScaling,
    n: NumTokens,
) -> TgShape<S>
where S: HasAxis<{ scaling.axis }>
```

Per-kernel `Dispatch` types from Phase 4/6 specify their `S` so the
m-scaling axis is checked against the dispatch geometry.

Success criterion: writing `axis: MScaleAxis::X` for a kernel whose
tid.x is the head axis (not the Q-block axis) is a compile error.

---

### Phase 8 — Build-script reflection on `.metal` sources (~3 days, stretch)

**Target bug:** Drift between Metal kernel headers and Rust constants
structs (Phases 2, 3) — those are still hand-maintained mirrors.

Add `build.rs` step that parses every `shaders/*.metal` for:

* `constant <type> <name> [[function_constant(<N>)]]` — emit a
  `KernelConstantsSpec` static.
* `[[buffer(<N>)]]` declarations on kernel functions — emit a
  `KernelBindingsSpec` static.
* `template <int BQ, int BK, ...> [[kernel]] ... <T, 32, 16, ...>` —
  emit a `KernelTemplateSpec`.

Then provide a derive (or hand-checked impl) that asserts at compile
time:

* The Phase-2 `AttentionPrefillPagedConstants::SLOTS` matches the
  kernel's `KernelConstantsSpec`.
* The Phase-3 `AttentionPrefillPagedBindingSet::BINDINGS` matches the
  kernel's `KernelBindingsSpec`.
* The Phase-6 `SteelAttnTile<32, 16, ...>` matches the kernel's
  `KernelTemplateSpec` for the instantiation being selected.

This catches "added a new function constant to the .metal source,
forgot to update the Rust struct" at build time. Adds 50ish lines of
Metal-source regex parsing.

Success criterion: editing a `.metal` file to add a function constant
without updating the matching Rust struct fails `cargo build` with a
clear error pointing at the mismatch.

---

## Order of operations

```
Phase 1 (newtypes) ──┐
                     ├── Phase 2 (constants) ──┐
                     │                         ├── Phase 4 (MetalKernel) ──┐
                     ├── Phase 3 (bindings) ───┘                           ├── Phase 6 (const generics) ──┐
                     ├── Phase 5 (PagedKvLayout) ─────────────────────────┘                              ├── Phase 8 (build.rs)
                     └── Phase 7 (dispatch shape) ───────────────────────────────────────────────────────┘
```

Phase 1, 2, 3, 5 are independently mergeable. Phase 4 needs 2 and 3.
Phase 7 is independent but most useful after Phase 1. Phase 6 needs
Phase 4. Phase 8 needs everything.

Realistic schedule for a single engineer: 1–2 weeks for Phases 1–5
(the high-value 80%). Phase 6 and 7 another 1–2 weeks. Phase 8 is
worth doing if the team is going to add many more kernels — otherwise
defer.

## What you shouldn't try to type-check (yet)

* The kernel's internal threadgroup-memory layout, shared with C++
  side via `paged_loader.h` etc. Cross-language type sharing is its
  own project. Phase 5's `PagedKvLayout` does the externally-visible
  layout; internal smem layouts stay text-comments + paired source
  inspection.

* Apple Metal compiler bugs like the `PagedKVBlockLoader::load_unsafe`
  recompute-src pattern (bug #2). That was a *codegen* surprise, not
  a type-system gap. The fix was structural mirroring (`PagedBlockLoaderT`
  inheriting `BlockLoaderT`'s field layout). A `#[repr(C)]` on the
  Rust side and `static_assert(sizeof::<Foo>() == X)` is the closest
  type-level expression of "byte-for-byte mirror", but doesn't catch
  the *behavior* divergence — only the layout divergence.

* MTL3 vs MTL4 encoder kind. Already typed via the
  `BucketStep::{Icb, Mps}` enum + the `use_mtl3` env-gated fallback.
  Could be further typed but no recurring bugs there.

## Quick reference — where things live now

| Concept | File | Notes |
|---|---|---|
| `KernelId`, `Binding`, `DispatchShape`, `MScaling`, `LoweredCommand` | `crates/ferrite-forward/src/interpreter/metal/lowered.rs` | All become typed in Phases 2–4, 7 |
| `ConstantValue`, `PipelineKey`, `SpecializedPipelineCache` | `crates/ferrite-metal-kernels/src/specialized_pipeline_cache.rs` | Phase 2 wraps ConstantValue's slot arg in `ConstSlot` |
| Lowering arms (per-instruction) | `crates/ferrite-forward/src/interpreter/metal/lowering.rs` | The main consumer of Phases 1–4 |
| RuntimeBindings | `crates/ferrite-forward/src/interpreter/metal/runtime.rs` | `kv_cache_k: Vec<Buffer>` indexed by `LayerId` after Phase 1 |
| Worker / bake / dispatch | `crates/ferrite-forward/src/interpreter/metal/worker.rs` | `scale_tg_for_num_tokens` is the Phase 7 hot spot |
| Forward / bucket pick / arena dump | `crates/ferrite-forward/src/interpreter/metal/pool.rs` | `forward()` is where the typed types meet the runtime |
| CPU goldens (paged) | `crates/ferrite-forward/src/cpu_golden.rs` | Phase 5 consumer |
| Cost-sweep bench (constants per pipeline) | `crates/ferrite-metal-cost-sweep/src/attention_sweep.rs` | Phase 2's `paged_constants(debug_mode)` becomes a struct ctor |
| Standard pipeline registration | `crates/ferrite-metal-kernels/src/specialized_pipeline_cache.rs::with_standard_shaders` | Phase 4 — register typed `MetalKernel` impls instead of a list of `("name", bytes)` tuples |
| Forward macro codegen | `crates/ferrite-forward-macro/src/codegen.rs` and friends | Where the bucket-bake / lowered-tape gets emitted; Phases 1, 7 thread through here |

## Done definition

After Phases 1–5 land:

* Adding a new `[[function_constant(N)]]` to a `.metal` file without
  updating the matching Rust struct is a compile error at every call
  site.
* Adding a new `[[buffer(N)]]` is the same.
* Confusing `LayerId` with `ArenaSlotIdx` is a compile error.
* Confusing `BucketM` with `NumTokens` is a compile error.
* Reverting a kernel's library/function/kernel-id trio piecewise is
  impossible — they're a single `MetalKernel` trait impl.
* The paged-KV-cache stride math has one definition.

After Phases 6, 7 also land:

* Dispatching a kernel with mismatched tile geometry is a compile error.
* The `m_scaling` axis can't be wired to a tg shape whose axis means
  something different.

After Phase 8:

* `.metal` source ↔ Rust constants struct drift fails the build.
