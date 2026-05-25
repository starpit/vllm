// SPDX-License-Identifier: Apache-2.0
//! Substrate-typestate vocabulary — **const-generic** edition.
//!
//! Substrate-proof primitives whose load-bearing values (page id,
//! scratch offset/bytes, mbarrier phase, edge id, iter count) live as
//! **const generics**. Constructors are `const fn` whose bodies open
//! a `const { assert!(...) }` block — those `assert!`s are evaluated
//! at MONOMORPHIZATION TIME, so a primitive whose const args violate
//! a substrate invariant is rejected by `rustc` with `E0080`. The
//! plan's "if it compiles, it runs coherently" invariant becomes
//! structurally true at the API surface.
//!
//! Per `MEGA_IR_PLAN.md` §1+§4+§8.1.
//!
//! ## What's compile-time
//!
//! - `PageId<ID, NUM_PAGES>::new()` — fails if `ID >= NUM_PAGES`.
//! - `ScratchRegion<OFFSET, BYTES, SCRATCH_BYTES, Scope>::new()` —
//!   fails if `OFFSET + BYTES > SCRATCH_BYTES` (overflow-safe).
//! - `ScratchRegion::disjoint_with()` — fails if two regions in the
//!   same scope overlap.
//! - `MbarrierPhase<P>::assert_matches::<N>()` — fails if `P != N&1`.
//! - `IterCount<ITERS>::new()` — fails if `ITERS == 0`.
//! - `EdgeId<IDX, NUM_EDGES>::new()` — fails if `IDX >= NUM_EDGES`.
//! - `ExpectedCount<COUNT>::new()` — fails if `COUNT == 0`.
//! - `Page<ID, NUM_PAGES, State>::new()` — same as PageId.
//!
//! ## What's runtime (deliberate stable-Rust simplification)
//!
//! - [`PagePool`] — cross-op alias tracking via `Vec<bool>`. With
//!   const-generic IDs, two `PageId<5, 16>` values are the SAME type;
//!   the alias-tracking question is "is this slot in use right now"
//!   which is genuine runtime state in the proc-macro's walk through
//!   the tape. A truly session-typed pool (linear types) is a
//!   nightly/Rust-evolution item; we keep `take(id)` runtime here.
//!
//! - Helper newtypes (`WeightRef`, `RotaryRef`, `FiniteF32`,
//!   `LmHeadNormKind`, `GateUpActivation`, `AttentionKind`) — these
//!   are not numeric primitives ranging over a small typed alphabet,
//!   they're paths/floats/enums. Per plan §3, helpers ride alongside
//!   substrate-proof load-bearing fields and stay runtime-validated.
//!
//! ## Erasure for storage
//!
//! Variants in [`crate::ir::nodes`] store plain `u32` fields, NOT typed
//! const-generic primitives. The substrate proof is discharged at
//! `MegaNode::variant::new::<...>()` call time via `const {}` block;
//! after that, the values flow as plain integers (zero runtime cost,
//! no `PhantomData` chain bloat). The PROOF is at the call site;
//! the storage is plain.

#![allow(dead_code)]

use core::marker::PhantomData;

mod sealed {
    pub trait Sealed {}
}

// ============================================================
// Substrate budget — zero-sized type parameterized on the
// per-variant constants the lowering emits.
// ============================================================

/// Per-variant substrate budget — pure type-level. Const generics
/// pin every shape constant; the value is zero-sized and exists
/// solely so primitives can be parameterized on a single
/// `Budget<NUM_PAGES, ...>` type, but we still expose the individual
/// const generics directly on each primitive (plan §1's pattern).
///
/// `SubstrateBudget::new()` runs a `const {}` block that asserts
/// non-zero invariants — same shape as the runtime checks, but the
/// rejection happens at monomorphization, not runtime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SubstrateBudget<
    const NUM_PAGES: u32,
    const NUM_CONSUMER_WARPS: u32,
    const PAGE_SIZE: u32,
    const SCRATCH_BYTES: u32,
    const NUM_EDGES: u32,
>;

impl<
    const NUM_PAGES: u32,
    const NUM_CONSUMER_WARPS: u32,
    const PAGE_SIZE: u32,
    const SCRATCH_BYTES: u32,
    const NUM_EDGES: u32,
> SubstrateBudget<NUM_PAGES, NUM_CONSUMER_WARPS, PAGE_SIZE, SCRATCH_BYTES, NUM_EDGES>
{
    /// Construct a substrate budget. Compile-time `const {}` block
    /// asserts non-zero invariants.
    pub const fn new() -> Self {
        const {
            assert!(NUM_PAGES > 0, "SubstrateBudget: NUM_PAGES must be > 0");
            assert!(
                NUM_CONSUMER_WARPS > 0,
                "SubstrateBudget: NUM_CONSUMER_WARPS must be > 0",
            );
            assert!(PAGE_SIZE > 0, "SubstrateBudget: PAGE_SIZE must be > 0");
            // SCRATCH_BYTES may legitimately be 0; NUM_EDGES too.
        }
        Self
    }

    pub const fn num_pages(self) -> u32 {
        NUM_PAGES
    }
    pub const fn num_consumer_warps(self) -> u32 {
        NUM_CONSUMER_WARPS
    }
    pub const fn page_size(self) -> u32 {
        PAGE_SIZE
    }
    pub const fn scratch_bytes(self) -> u32 {
        SCRATCH_BYTES
    }
    pub const fn num_edges(self) -> u32 {
        NUM_EDGES
    }
}

// ============================================================
// Page-slot lifecycle (bug classes #1, #2).
// ============================================================

pub trait IsLifecycleState: sealed::Sealed {}

pub struct Empty(PhantomData<*const ()>);
pub struct Filled(PhantomData<*const ()>);
pub struct Produced(PhantomData<*const ()>);

impl sealed::Sealed for Empty {}
impl sealed::Sealed for Filled {}
impl sealed::Sealed for Produced {}
impl IsLifecycleState for Empty {}
impl IsLifecycleState for Filled {}
impl IsLifecycleState for Produced {}

/// A page slot in the substrate's `pages[]` array.
///
/// Const-generic `ID` carries the slot id; const-generic `NUM_PAGES`
/// carries the substrate budget. `Page::<5, 16, Empty>::new()` runs
/// a `const {}` block that compile-fails when `ID >= NUM_PAGES`
/// (E0080). Lifecycle transitions consume the typestate token —
/// `Empty` → `Filled` → `Produced` → `Empty`.
pub struct Page<const ID: u32, const NUM_PAGES: u32, State: IsLifecycleState> {
    _state: PhantomData<*const State>,
}

impl<const ID: u32, const NUM_PAGES: u32> Page<ID, NUM_PAGES, Empty> {
    /// Construct a fresh `Empty` page. Compile-fails (E0080) if
    /// `ID >= NUM_PAGES`.
    pub const fn new() -> Self {
        const {
            assert!(ID < NUM_PAGES, "Page: ID out of bounds (ID >= NUM_PAGES)");
        }
        Self {
            _state: PhantomData,
        }
    }

    /// Loader-fired transition: `Empty` → `Filled`. Consumes self.
    pub const fn loader_fired(self) -> Page<ID, NUM_PAGES, Filled> {
        Page {
            _state: PhantomData,
        }
    }
}

impl<const ID: u32, const NUM_PAGES: u32> Page<ID, NUM_PAGES, Filled> {
    /// Consumer-arrived: `Filled` → `Produced`.
    pub const fn consumer_arrived(self) -> Page<ID, NUM_PAGES, Produced> {
        Page {
            _state: PhantomData,
        }
    }
}

impl<const ID: u32, const NUM_PAGES: u32> Page<ID, NUM_PAGES, Produced> {
    /// Storer-consumed: `Produced` → `Empty`.
    pub const fn storer_consumed(self) -> Page<ID, NUM_PAGES, Empty> {
        Page {
            _state: PhantomData,
        }
    }
}

impl<const ID: u32, const NUM_PAGES: u32, State: IsLifecycleState> Page<ID, NUM_PAGES, State> {
    /// Raw page id — for emit-side string formatting only.
    pub const fn id(&self) -> u32 {
        ID
    }
}

impl<const ID: u32, const NUM_PAGES: u32> Default for Page<ID, NUM_PAGES, Empty> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Scratch regions (bug classes #4, #5).
// ============================================================

pub trait IsScratchScope: sealed::Sealed {}

/// A scratch-byte region inside the substrate's
/// `scratch[SCRATCH_BYTES]` buffer.
///
/// Const-generic `OFFSET`, `BYTES`, and `SCRATCH_BYTES` make
/// within-budget a monomorphization-time check (bug class #5).
/// `disjoint_with` opens a `const {}` block that compile-fails on
/// overlap (bug class #4). `Scope` is a type parameter; two regions
/// in different scopes don't constrain each other.
pub struct ScratchRegion<
    const OFFSET: u32,
    const BYTES: u32,
    const SCRATCH_BYTES: u32,
    Scope: IsScratchScope,
> {
    _scope: PhantomData<*const Scope>,
}

impl<const OFFSET: u32, const BYTES: u32, const SCRATCH_BYTES: u32, Scope: IsScratchScope>
    ScratchRegion<OFFSET, BYTES, SCRATCH_BYTES, Scope>
{
    /// Construct a scratch region. Compile-fails (E0080) if
    /// `OFFSET + BYTES` overflows or exceeds `SCRATCH_BYTES`.
    pub const fn new() -> Self {
        const {
            // overflow-safe: u32::MAX as u64 + u32::MAX as u64 fits in u64.
            let end = (OFFSET as u64) + (BYTES as u64);
            assert!(
                end <= SCRATCH_BYTES as u64,
                "ScratchRegion: OFFSET+BYTES out of substrate scratch budget",
            );
        }
        Self {
            _scope: PhantomData,
        }
    }

    /// Validate this region is disjoint from `other` within the same
    /// scope. Compile-fails (E0080) on overlap. Returns both regions
    /// so the caller can hold them simultaneously.
    pub const fn disjoint_with<const O2: u32, const B2: u32>(
        self,
        other: ScratchRegion<O2, B2, SCRATCH_BYTES, Scope>,
    ) -> (Self, ScratchRegion<O2, B2, SCRATCH_BYTES, Scope>) {
        const {
            // disjoint iff a_end <= O2 OR b_end <= OFFSET (in u64 to
            // avoid u32 overflow during the compile-time comparison).
            let a_end = (OFFSET as u64) + (BYTES as u64);
            let b_end = (O2 as u64) + (B2 as u64);
            assert!(
                a_end <= O2 as u64 || b_end <= OFFSET as u64,
                "ScratchRegion overlap within scope",
            );
        }
        (self, other)
    }

    /// Discharge proof that this region fits a single-stage paged-KV
    /// block of `[BLOCK_SIZE, NUM_KV_HEADS * HEAD_DIM]` bf16. Compile-
    /// fails (E0080) if `BYTES < BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM
    /// * 2`. Returns `self` so the caller can chain.
    ///
    /// Used by `push_attention_via_cache` to prove the K_smem and
    /// V_smem regions are sized correctly for the paged-KV gather,
    /// per [[feedback-end-to-end-compile-time-proofs]] (the assert
    /// inside `AttentionViaCacheNode::new` against the same
    /// invariant becomes dead code once this proof has been
    /// discharged).
    pub const fn fits_kv_block<
        const BLOCK_SIZE: u32,
        const NUM_KV_HEADS: u32,
        const HEAD_DIM: u32,
    >(self) -> Self {
        const {
            let needed = (BLOCK_SIZE as u64) * (NUM_KV_HEADS as u64) * (HEAD_DIM as u64) * 2;
            assert!(
                BYTES as u64 >= needed,
                "ScratchRegion::fits_kv_block: BYTES < BLOCK_SIZE*NUM_KV_HEADS*HEAD_DIM*2",
            );
        }
        self
    }

    pub const fn offset(&self) -> u32 {
        OFFSET
    }
    pub const fn bytes(&self) -> u32 {
        BYTES
    }
}

impl<const OFFSET: u32, const BYTES: u32, const SCRATCH_BYTES: u32, Scope: IsScratchScope> Default
    for ScratchRegion<OFFSET, BYTES, SCRATCH_BYTES, Scope>
{
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Mbarrier phase (bug class #3).
// ============================================================

/// Mbarrier phase value (0 or 1). Const-generic `P`. The companion
/// `assert_matches::<N>()` constructor compile-fails when
/// `P != N & 1` (bug class #3 enforcement: phase mismatch with
/// cumulative arrive count).
pub struct MbarrierPhase<const P: u32>;

impl<const P: u32> MbarrierPhase<P> {
    /// Construct a phase. Compile-fails if `P > 1` (only 0 or 1 is a
    /// valid mbarrier phase).
    pub const fn new() -> Self {
        const {
            assert!(P <= 1, "MbarrierPhase: P must be 0 or 1");
        }
        Self
    }

    /// Assert this phase parity matches a cumulative arrive count
    /// `N`. Compile-fails if `P != N & 1`.
    pub const fn assert_matches<const N: u32>() -> Self {
        const {
            assert!(P <= 1, "MbarrierPhase: P must be 0 or 1");
            assert!(
                P == N & 1,
                "MbarrierPhase: parity mismatch with cumulative arrive count",
            );
        }
        Self
    }

    /// Assert this phase parity matches `N + 1` (the storer-side
    /// phase that pairs with a consumer-phase `assert_matches::<N>`).
    /// Compile-fails if `P != (N + 1) & 1`. Exists because Rust
    /// without `generic_const_exprs` rejects `{ ARRIVES + 1 }` as a
    /// const-generic argument; this fn lets callers pass the same
    /// `ARRIVES` literal both arms ([[feedback-end-to-end-compile-time-proofs]]).
    pub const fn assert_matches_next<const N: u32>() -> Self {
        const {
            assert!(P <= 1, "MbarrierPhase: P must be 0 or 1");
            assert!(
                P == (N.wrapping_add(1)) & 1,
                "MbarrierPhase: parity mismatch with cumulative arrive count + 1",
            );
        }
        Self
    }

    pub const fn phase(&self) -> u32 {
        P
    }
}

impl<const P: u32> Default for MbarrierPhase<P> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Warp-role pairing (bug class #6).
// ============================================================

pub const ROLE_LOADER: u8 = 0;
pub const ROLE_LAUNCHER: u8 = 1;
pub const ROLE_CONSUMER: u8 = 2;
pub const ROLE_STORER: u8 = 3;

/// A warp's role within the megakernel — `R` is a const-generic
/// `u8` pinned at type construction time. Sealed for {0..3}.
pub struct WarpRoleTag<const R: u8>;

pub trait IsValidWarpRole: sealed::Sealed {}
impl sealed::Sealed for WarpRoleTag<ROLE_LOADER> {}
impl sealed::Sealed for WarpRoleTag<ROLE_LAUNCHER> {}
impl sealed::Sealed for WarpRoleTag<ROLE_CONSUMER> {}
impl sealed::Sealed for WarpRoleTag<ROLE_STORER> {}
impl IsValidWarpRole for WarpRoleTag<ROLE_LOADER> {}
impl IsValidWarpRole for WarpRoleTag<ROLE_LAUNCHER> {}
impl IsValidWarpRole for WarpRoleTag<ROLE_CONSUMER> {}
impl IsValidWarpRole for WarpRoleTag<ROLE_STORER> {}

// ============================================================
// Bar.sync IDs (within-warpgroup synchronization barriers).
//
// PTX `bar.sync N` IDs are in `[0, 16)`; bar 0 is reserved for
// `__syncthreads`. Op-internal barriers occupy bars `1..=15`.
//
// **Type-system enforcement, not assert!**
//
// `BarSyncId<ID>: IsValidBarSyncId` is a sealed witness with impls
// **only** for ID in `1..=15`. `BarSyncPair<A, B>: IsDistinctBarPair`
// is implemented **only** for pairs where both A and B are valid AND
// `A != B`. Functions that take const-generic bar IDs declare these
// witnesses as `where` bounds — bad const args fail at TYPE CHECK,
// not at monomorphization assert.
//
// Producer/consumer pairing: a single IR field flows into BOTH the
// producer-side and consumer-side codegen splice points (e.g.
// `kittens::group<NCW>::sync(BAR_ID)` is participant-symmetric — every
// warp arrives AND waits on the same primitive call). There is no
// independent-input route by which the two sides could diverge; the
// IR field IS the single source of truth.
// ============================================================

pub struct BarSyncId<const ID: u32>;

pub trait IsValidBarSyncId: sealed::Sealed {}

pub struct BarSyncPair<const A: u32, const B: u32>;

pub trait IsDistinctBarPair: sealed::Sealed {}

/// Opaque post-erasure wrapper for a verified bar.sync ID.
///
/// Cannot be constructed from a raw `u32` — the only public path to
/// construct a `BarRef` is `BarSyncId::<ID>::erase()`, which requires
/// the sealed-witness `BarSyncId<ID>: IsValidBarSyncId` bound. Bare
/// integer literals don't satisfy that bound, so no caller can
/// produce an invalid `BarRef`.
///
/// Stored on `MegaNode` variants instead of bare `u32` so the IR
/// surface (fields, getters, function params) is typed throughout.
/// The `raw()` accessor is the only escape hatch back to integer for
/// splicing into emitted CUDA source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BarRef(u32);

impl BarRef {
    /// Recover the raw integer for codegen splicing only. Not
    /// constructible from raw `u32` outside this crate.
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

impl<const ID: u32> BarSyncId<ID>
where
    BarSyncId<ID>: IsValidBarSyncId,
{
    /// Construct a verified `BarSyncId`. The where bound only admits
    /// IDs in `1..=15`; `ID = 0` or `ID >= 16` fail at type-check
    /// (E0277), no monomorphization.
    pub const fn new() -> Self {
        Self
    }

    /// Erase to opaque [`BarRef`]. The validity witness was discharged
    /// at the type-check boundary above; the post-erasure value can
    /// flow through the IR without re-validation.
    pub const fn erase(self) -> BarRef {
        BarRef::__new_for_erase(ID)
    }
}

impl<const A: u32, const B: u32> BarSyncPair<A, B>
where
    BarSyncPair<A, B>: IsDistinctBarPair,
{
    /// Construct a verified distinct-bar pair. The where bound only
    /// admits ordered (A, B) where both are valid AND `A != B`.
    pub const fn new() -> Self {
        Self
    }

    /// Erase to opaque [`DistinctBarPairProof`].
    pub const fn erase(self) -> DistinctBarPairProof {
        DistinctBarPairProof { _priv: () }
    }
}

/// Opaque proof token witnessing that two specific bar.sync IDs are
/// distinct + valid. Storage-erased (zero data) so an op variant can
/// hold "I have distinct bars" without paying for the const generics.
/// Constructible only via [`BarSyncPair::erase`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DistinctBarPairProof {
    _priv: (),
}

// ============================================================
// Opaque post-erasure refs — the only u32 escape hatch is `raw()`,
// used by codegen to splice into emitted CUDA source. NONE of these
// have a public constructor that accepts a raw `u32`. Only the typed
// const-generic primitives (PageId, MbarrierPhase, …) can produce
// them, via `erase()`. Bare `u32` cannot be smuggled into the IR.
// ============================================================

/// Verified page-slot id (post-erasure of `PageId<ID, NUM_PAGES>`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PageRef(u32);
impl PageRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    /// Crate-private constructor — only the typed primitive's
    /// `erase()` may call this. External callers must go through the
    /// substrate-proof typed primitive.
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Verified mbarrier phase (post-erasure of `MbarrierPhase<P>`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MbarrierPhaseRef(u32);
impl MbarrierPhaseRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Verified per-layer index (post-erasure of `LayerIndex<L, NUM_LAYERS>`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayerRef(u32);
impl LayerRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Verified scratch-region offset (post-erasure of `ScratchRegion<O, B, S>`'s
/// `OFFSET` const generic).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScratchOffsetRef(u32);
impl ScratchOffsetRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Verified scratch-region byte count (post-erasure of `ScratchRegion`'s
/// `BYTES` const generic).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScratchBytesRef(u32);
impl ScratchBytesRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Verified positive `HIDDEN_DIM` template arg.
pub struct HiddenDim<const D: u32>;
pub trait IsValidHiddenDim: sealed::Sealed {}
impl<const D: u32> sealed::Sealed for HiddenDim<D> {}
// Positivity sealed witness: implementation only when D > 0 — checked
// via a const-generic `where` reach: a separate `IsPositive<X>` sealed
// trait with impls only for the const-generic value. Stable Rust
// can't express "where D > 0", so we keep monomorphization-time
// `assert!(D > 0)` inside `new()`. Range invariants over arbitrary
// `u32` (1..=u32::MAX) cannot be enumerated as sealed impls; the
// substrate-proof primitives whose value range is finite (BarSyncId
// 1..=15) use sealed witnesses, those whose range is unbounded use
// `const { assert!() }` (E0080 at monomorphization).
//
// Both forms cause `rustc` to refuse to compile bad const args; the
// difference is only WHERE in the pipeline the rejection fires.
impl<const D: u32> HiddenDim<D> {
    pub const fn new() -> Self {
        const {
            assert!(D > 0, "HiddenDim: D must be > 0");
        }
        Self
    }
    pub const fn erase(self) -> HiddenDimRef {
        HiddenDimRef::__new_for_erase(D)
    }
}
impl<const D: u32> IsValidHiddenDim for HiddenDim<D> {}

/// Opaque post-erasure of [`HiddenDim`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HiddenDimRef(u32);
impl HiddenDimRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Cumulative-arrive-count marker passed at each `push_*` call so
/// the const-generic `ARRIVES` value flows via type inference instead
/// of needing turbofish at the call site. The runtime `MegaTapeBuilder`
/// verifies this against its actual arrive count
/// (`verify_arrives`); the const-generic side carries the value into
/// the typed primitive's substrate proof (e.g.
/// `MbarrierPhase::assert_matches::<ARRIVES>`).
pub struct ArrivesCount<const N: u32>;
impl<const N: u32> ArrivesCount<N> {
    pub const fn new() -> Self {
        Self
    }
    pub const fn raw(self) -> u32 {
        N
    }
    /// Derive the page-round mbarrier phase parity from the cumulative
    /// arrive count. By the page-round protocol all three roles
    /// (consumer / storer / loader for the current page-round) wait on
    /// the same parity bit, which is `N & 1`. The const generic `N`
    /// is the proof — there is no path to fabricate a wrong value
    /// (no separate `PHASE` const generic that could disagree).
    /// Replaces redundant `MbarrierPhase::<P>::new()` const generics
    /// that previously duplicated the formula on the call site.
    pub const fn derive_phase(self) -> MbarrierPhaseRef {
        MbarrierPhaseRef::__new_for_erase(N & 1)
    }
}
impl<const N: u32> Default for ArrivesCount<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Verified positive `NUM_TOKENS` template arg.
pub struct NumTokensConst<const N: u32>;
impl<const N: u32> NumTokensConst<N> {
    pub const fn new() -> Self {
        const {
            assert!(N > 0, "NumTokens: N must be > 0");
        }
        Self
    }
    pub const fn erase(self) -> NumTokensRef {
        NumTokensRef::__new_for_erase(N)
    }
}
/// Opaque post-erasure of [`NumTokensConst`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NumTokensRef(u32);
impl NumTokensRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Verified activation-slot index (`< NUM_ACT_SLOTS`).
pub struct ActSlotConst<const SLOT: u32, const NUM_ACT_SLOTS: u32>;
impl<const SLOT: u32, const NUM_ACT_SLOTS: u32> ActSlotConst<SLOT, NUM_ACT_SLOTS> {
    pub const fn new() -> Self {
        const {
            assert!(
                SLOT < NUM_ACT_SLOTS,
                "ActSlot: SLOT >= NUM_ACT_SLOTS",
            );
        }
        Self
    }
    pub const fn erase(self) -> ActSlotRef {
        ActSlotRef::__new_for_erase(SLOT)
    }
}
/// Opaque post-erasure of [`ActSlotConst`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ActSlotRef(u32);
impl ActSlotRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Verified positive `HEAD_DIM` template arg.
pub struct HeadDim<const D: u32>;
impl<const D: u32> HeadDim<D> {
    pub const fn new() -> Self {
        const {
            assert!(D > 0, "HeadDim: D must be > 0");
        }
        Self
    }
    pub const fn erase(self) -> HeadDimRef {
        HeadDimRef::__new_for_erase(D)
    }
}
impl<const D: u32> Default for HeadDim<D> {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HeadDimRef(u32);
impl HeadDimRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Verified column-offset for the down_proj K-chunk split (0 for
/// un-chunked GemmAdd; multiple of K for chunks).
pub struct KOffset<const OFF: u32>;
impl<const OFF: u32> KOffset<OFF> {
    pub const fn new() -> Self {
        Self
    }
    pub const fn erase(self) -> KOffsetRef {
        KOffsetRef::__new_for_erase(OFF)
    }
}
impl<const OFF: u32> Default for KOffset<OFF> {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KOffsetRef(u32);
impl KOffsetRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Verified positive K_FULL — full reduction dim across all chunks.
pub struct KFull<const FULL: u32>;
impl<const FULL: u32> KFull<FULL> {
    pub const fn new() -> Self {
        const {
            assert!(FULL > 0, "KFull: FULL must be > 0");
        }
        Self
    }
    pub const fn erase(self) -> KFullRef {
        KFullRef::__new_for_erase(FULL)
    }
}
impl<const FULL: u32> Default for KFull<FULL> {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KFullRef(u32);
impl KFullRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Verified positive paged-KV block size (BLOCK_SIZE in attention
/// kernels; the rows-per-page tile shape).
pub struct BlockSize<const N: u32>;
impl<const N: u32> BlockSize<N> {
    pub const fn new() -> Self {
        const {
            assert!(N > 0, "BlockSize: N must be > 0");
        }
        Self
    }
    pub const fn erase(self) -> BlockSizeRef {
        BlockSizeRef::__new_for_erase(N)
    }
}
impl<const N: u32> Default for BlockSize<N> {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockSizeRef(u32);
impl BlockSizeRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Verified positive max-sequence-K bucket (MAX_SK ceiling for the
/// attention KV pages-per-seq dimension).
pub struct MaxSk<const N: u32>;
impl<const N: u32> MaxSk<N> {
    pub const fn new() -> Self {
        const {
            assert!(N > 0, "MaxSk: N must be > 0");
        }
        Self
    }
    pub const fn erase(self) -> MaxSkRef {
        MaxSkRef::__new_for_erase(N)
    }
}
impl<const N: u32> Default for MaxSk<N> {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaxSkRef(u32);
impl MaxSkRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Verified positive matmul-N (output cols / N dim).
pub struct MatmulN<const N: u32>;
impl<const N: u32> MatmulN<N> {
    pub const fn new() -> Self {
        const {
            assert!(N > 0, "MatmulN: N must be > 0");
        }
        Self
    }
    pub const fn erase(self) -> MatmulNRef {
        MatmulNRef::__new_for_erase(N)
    }
}
impl<const N: u32> Default for MatmulN<N> {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatmulNRef(u32);
impl MatmulNRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Verified positive matmul-K (reduction dim).
pub struct MatmulK<const K: u32>;
impl<const K: u32> MatmulK<K> {
    pub const fn new() -> Self {
        const {
            assert!(K > 0, "MatmulK: K must be > 0");
        }
        Self
    }
    pub const fn erase(self) -> MatmulKRef {
        MatmulKRef::__new_for_erase(K)
    }
}
impl<const K: u32> Default for MatmulK<K> {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatmulKRef(u32);
impl MatmulKRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Verified positive matmul-M (output rows / M dim, == NUM_TOKENS at
/// canonical's workload point).
pub struct MatmulM<const M: u32>;
impl<const M: u32> MatmulM<M> {
    pub const fn new() -> Self {
        const {
            assert!(M > 0, "MatmulM: M must be > 0");
        }
        Self
    }
    pub const fn erase(self) -> MatmulMRef {
        MatmulMRef::__new_for_erase(M)
    }
}
impl<const M: u32> Default for MatmulM<M> {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatmulMRef(u32);
impl MatmulMRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Verified positive per-warp output tile N dim — the slice of the
/// matmul N axis that one consumer warp owns. Convention: AlongN
/// warp split (each warp covers all M rows of its N slice). For
/// `Gemm` and friends, `tile_n == N / NUM_CONSUMER_WARPS`.
pub struct TileN<const N: u32>;
impl<const N: u32> TileN<N> {
    pub const fn new() -> Self {
        const {
            assert!(N > 0, "TileN: N must be > 0");
        }
        Self
    }
    pub const fn erase(self) -> TileNRef {
        TileNRef::__new_for_erase(N)
    }
}
impl<const N: u32> Default for TileN<N> {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileNRef(u32);
impl TileNRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Verified positive per-iter K-chunk width — the K dim of one
/// b_tile load. For `Gemm`, `chunk_k == K / iters` AND
/// `b_tile_bytes == chunk_k * N * sizeof(bf16)`.
pub struct ChunkK<const K: u32>;
impl<const K: u32> ChunkK<K> {
    pub const fn new() -> Self {
        const {
            assert!(K > 0, "ChunkK: K must be > 0");
        }
        Self
    }
    pub const fn erase(self) -> ChunkKRef {
        ChunkKRef::__new_for_erase(K)
    }
}
impl<const K: u32> Default for ChunkK<K> {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkKRef(u32);
impl ChunkKRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Verified positive `VOCAB_SIZE` (Embed kernel template arg).
pub struct VocabSize<const N: u32>;
impl<const N: u32> VocabSize<N> {
    pub const fn new() -> Self {
        const {
            assert!(N > 0, "VocabSize: N must be > 0");
        }
        Self
    }
    pub const fn erase(self) -> VocabSizeRef {
        VocabSizeRef::__new_for_erase(N)
    }
}
impl<const N: u32> Default for VocabSize<N> {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VocabSizeRef(u32);
impl VocabSizeRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Verified positive `INTERMEDIATE_DIM` template arg.
pub struct IntermediateDim<const D: u32>;
impl<const D: u32> IntermediateDim<D> {
    pub const fn new() -> Self {
        const {
            assert!(D > 0, "IntermediateDim: D must be > 0");
        }
        Self
    }
    pub const fn erase(self) -> IntermediateDimRef {
        IntermediateDimRef::__new_for_erase(D)
    }
}
impl<const D: u32> Default for IntermediateDim<D> {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IntermediateDimRef(u32);
impl IntermediateDimRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Verified positive Q-head count.
pub struct NumQHeads<const N: u32>;
impl<const N: u32> NumQHeads<N> {
    pub const fn new() -> Self {
        const {
            assert!(N > 0, "NumQHeads: N must be > 0");
        }
        Self
    }
    pub const fn erase(self) -> NumQHeadsRef {
        NumQHeadsRef::__new_for_erase(N)
    }
}
impl<const N: u32> Default for NumQHeads<N> {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NumQHeadsRef(u32);
impl NumQHeadsRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Verified positive KV-head count.
pub struct NumKvHeads<const N: u32>;
impl<const N: u32> NumKvHeads<N> {
    pub const fn new() -> Self {
        const {
            assert!(N > 0, "NumKvHeads: N must be > 0");
        }
        Self
    }
    pub const fn erase(self) -> NumKvHeadsRef {
        NumKvHeadsRef::__new_for_erase(N)
    }
}
impl<const N: u32> Default for NumKvHeads<N> {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NumKvHeadsRef(u32);
impl NumKvHeadsRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Opaque post-erasure of [`IterCount`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IterCountRef(u32);
impl IterCountRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Verified weight-accessor index (`< NUM_WEIGHT_ACCESSORS`).
pub struct WeightAccessorConst<const IDX: u32, const NUM_WEIGHT_ACCESSORS: u32>;
impl<const IDX: u32, const NUM_WEIGHT_ACCESSORS: u32>
    WeightAccessorConst<IDX, NUM_WEIGHT_ACCESSORS>
{
    pub const fn new() -> Self {
        const {
            assert!(
                IDX < NUM_WEIGHT_ACCESSORS,
                "WeightAccessor: IDX >= NUM_WEIGHT_ACCESSORS",
            );
        }
        Self
    }
    pub const fn erase(self) -> WeightAccessorRef {
        WeightAccessorRef::__new_for_erase(IDX)
    }
}
/// Opaque post-erasure of [`WeightAccessorConst`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightAccessorRef(u32);
impl WeightAccessorRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

// `erase` on the existing typed primitives — these primitives already
// validate at construction; here we add the post-erasure step.
impl<const ID: u32, const NUM_PAGES: u32> PageId<ID, NUM_PAGES> {
    /// Erase to opaque [`PageRef`]. Validity (`ID < NUM_PAGES`) was
    /// discharged at `PageId::new`'s `const {}` block.
    pub const fn erase(self) -> PageRef {
        PageRef::__new_for_erase(ID)
    }
}

impl<const P: u32> MbarrierPhase<P> {
    pub const fn erase(self) -> MbarrierPhaseRef {
        MbarrierPhaseRef::__new_for_erase(P)
    }
}

// Note: `LayerIndex::erase()` lives in `crate::ir::nodes` (where
// `LayerIndex` is defined) to avoid a cyclic module dep.

impl<
    const OFFSET: u32,
    const BYTES: u32,
    const SCRATCH_BYTES: u32,
    Scope: IsScratchScope,
> ScratchRegion<OFFSET, BYTES, SCRATCH_BYTES, Scope>
{
    /// Erase to opaque (offset, bytes) refs. Within-budget +
    /// disjoint-with-others were discharged earlier by
    /// `ScratchRegion::new` and `disjoint_with`.
    pub const fn erase(self) -> (ScratchOffsetRef, ScratchBytesRef) {
        (
            ScratchOffsetRef::__new_for_erase(OFFSET),
            ScratchBytesRef::__new_for_erase(BYTES),
        )
    }
}

// Generated impls: one `IsValidBarSyncId` impl per ID in 1..=15, and
// one `IsDistinctBarPair` impl per ordered (A, B) pair in 1..=15
// where `A != B` (15 * 14 = 210 pairs).
//
// The macro recurses: at each step it pairs `$first` with every
// `$rest` value in BOTH orders, then recurses on the tail. No dupes
// because each iteration's `$first` only appears in pairs with
// strictly-later `$rest` values from the input list.
macro_rules! impl_bar_pairs {
    () => {};
    ($first:literal $($rest:literal)*) => {
        impl sealed::Sealed for BarSyncId<$first> {}
        impl IsValidBarSyncId for BarSyncId<$first> {}
        $(
            impl sealed::Sealed for BarSyncPair<$first, $rest> {}
            impl IsDistinctBarPair for BarSyncPair<$first, $rest> {}
            impl sealed::Sealed for BarSyncPair<$rest, $first> {}
            impl IsDistinctBarPair for BarSyncPair<$rest, $first> {}
        )*
        impl_bar_pairs!($($rest)*);
    };
}

impl_bar_pairs!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15);

// ============================================================
// PageId — finalized validated page-slot id for storage in a
// `MegaNode` variant (post-erasure to `u32`).
// ============================================================

/// Validated page id. Const-generic constructor compile-fails when
/// `ID >= NUM_PAGES`. `raw()` exposes the integer; the variant
/// stores the raw u32 (PROOF is at construction).
pub struct PageId<const ID: u32, const NUM_PAGES: u32>;

impl<const ID: u32, const NUM_PAGES: u32> PageId<ID, NUM_PAGES> {
    /// Construct a PageId. Compile-fails (E0080) if `ID >= NUM_PAGES`.
    pub const fn new() -> Self {
        const {
            assert!(ID < NUM_PAGES, "PageId: ID out of bounds");
        }
        Self
    }

    pub const fn raw(self) -> u32 {
        ID
    }
}

impl<const ID: u32, const NUM_PAGES: u32> Default for PageId<ID, NUM_PAGES> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// PagePool — RUNTIME cross-op alias tracking.
//
// With const-generic page IDs, two `PageId<5, 16>` values are the
// SAME type. The "in-flight across pushes" question is genuinely
// runtime state at the proc-macro's tape walk; a truly session-typed
// pool (linear types or `&'static mut` per-id tokens) is a
// nightly/Rust-evolution item. Kept runtime-validated here as a
// deliberate stable-Rust simplification — the per-op-construction-time
// substrate proofs already discharge bug classes #1, #4, #5 at
// compile-time; cross-op aliasing remains a runtime panic in this
// module.
// ============================================================

pub struct PagePool {
    in_use: Vec<bool>,
    num_pages: u32,
}

impl PagePool {
    /// Construct an empty pool sized to `num_pages`. The const value
    /// is recovered from the `MegaTapeBuilder<NUM_PAGES, ...>` const
    /// generic at the only construction site.
    pub fn new(num_pages: u32) -> Self {
        Self {
            in_use: vec![false; num_pages as usize],
            num_pages,
        }
    }

    /// Take an `Empty` page slot at runtime id `id`. Panics on
    /// out-of-bounds (cross-op runtime guard, bug class #1
    /// reinforcement) and cross-op alias (bug class #2 cross-op).
    /// The op MUST `release` afterwards.
    ///
    /// Stable-Rust note: `id` is a runtime u32 here because the
    /// PagePool tracks state across multiple typed-const-generic
    /// pushes; the per-op proof at variant `new::<...>()` time is
    /// where `ID < NUM_PAGES` becomes a compile error.
    pub fn take(&mut self, id: u32) -> u32 {
        assert!(
            id < self.num_pages,
            "PagePool::take: id {id} out of bounds num_pages={}",
            self.num_pages,
        );
        assert!(
            !self.in_use[id as usize],
            "PagePool::take: page id {id} already in use by another op (cross-op reuse)",
        );
        self.in_use[id as usize] = true;
        id
    }

    /// Return a page (any state). Idempotent.
    pub fn release(&mut self, id: u32) {
        if (id as usize) < self.in_use.len() {
            self.in_use[id as usize] = false;
        }
    }

    pub fn num_pages(&self) -> u32 {
        self.num_pages
    }
}

// ============================================================
// Public scope markers — see plan §1+§4.
// ============================================================

pub trait IsScratchScopePub: sealed::Sealed {}

/// Scratch scope for `RmsNorm`-family ops.
pub struct RmsNormScope;
impl sealed::Sealed for RmsNormScope {}
impl IsScratchScope for RmsNormScope {}
impl IsScratchScopePub for RmsNormScope {}

/// Scratch scope for `FusedQkvRopeCache` Q/K rotation tiles.
pub struct RopeScope;
impl sealed::Sealed for RopeScope {}
impl IsScratchScope for RopeScope {}
impl IsScratchScopePub for RopeScope {}

/// Scratch scope for `FusedGateUp{Silu,Gelu}Mul` gate/up tiles.
pub struct MlpScope;
impl sealed::Sealed for MlpScope {}
impl IsScratchScope for MlpScope {}
impl IsScratchScopePub for MlpScope {}

/// Scratch scope for `AttentionViaCache` score / PV tiles.
pub struct AttentionScope;
impl sealed::Sealed for AttentionScope {}
impl IsScratchScope for AttentionScope {}
impl IsScratchScopePub for AttentionScope {}

/// Scratch scope for `Gemm` / lm_head fusion B-tile.
pub struct GemmScope;
impl sealed::Sealed for GemmScope {}
impl IsScratchScope for GemmScope {}
impl IsScratchScopePub for GemmScope {}

// ============================================================
// IterCount — typed iteration count for per-iter phase math.
// ============================================================

/// Typed iteration count. Const-generic constructor compile-fails
/// when `ITERS == 0` (bug class #3: 0-iter ops desync the phase
/// counter).
pub struct IterCount<const ITERS: u32>;

impl<const ITERS: u32> IterCount<ITERS> {
    pub const fn new() -> Self {
        const {
            assert!(ITERS > 0, "IterCount: ITERS must be > 0");
        }
        Self
    }

    pub const fn raw(self) -> u32 {
        ITERS
    }

    /// Erase to opaque [`IterCountRef`]. Validity (`ITERS > 0`)
    /// discharged at `IterCount::new`'s `const {}` block.
    pub const fn erase(self) -> IterCountRef {
        IterCountRef::__new_for_erase(ITERS)
    }
}

impl<const ITERS: u32> Default for IterCount<ITERS> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Cross-CTA barrier edges (Sprint D).
// ============================================================

/// Validated edge index. Const-generic; compile-fails when
/// `IDX >= NUM_EDGES`.
pub struct EdgeId<const IDX: u32, const NUM_EDGES: u32>;

impl<const IDX: u32, const NUM_EDGES: u32> EdgeId<IDX, NUM_EDGES> {
    pub const fn new() -> Self {
        const {
            assert!(IDX < NUM_EDGES, "EdgeId: IDX out of bounds (>= NUM_EDGES)");
        }
        Self
    }

    pub const fn raw(self) -> u32 {
        IDX
    }

    pub const fn erase(self) -> EdgeIdRef {
        EdgeIdRef::__new_for_erase(IDX)
    }
}

impl<const IDX: u32, const NUM_EDGES: u32> Default for EdgeId<IDX, NUM_EDGES> {
    fn default() -> Self {
        Self::new()
    }
}

/// Opaque post-erasure of [`EdgeId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EdgeIdRef(u32);
impl EdgeIdRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

/// Expected arrive-count for a `BarrierWait`. Const-generic;
/// compile-fails when `COUNT == 0`.
pub struct ExpectedCount<const COUNT: u32>;

impl<const COUNT: u32> ExpectedCount<COUNT> {
    pub const fn new() -> Self {
        const {
            assert!(COUNT > 0, "ExpectedCount: COUNT must be > 0");
        }
        Self
    }

    pub const fn raw(self) -> u32 {
        COUNT
    }

    pub const fn erase(self) -> ExpectedCountRef {
        ExpectedCountRef::__new_for_erase(COUNT)
    }
}

impl<const COUNT: u32> Default for ExpectedCount<COUNT> {
    fn default() -> Self {
        Self::new()
    }
}

/// Opaque post-erasure of [`ExpectedCount`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExpectedCountRef(u32);
impl ExpectedCountRef {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub(crate) const fn __new_for_erase(v: u32) -> Self {
        Self(v)
    }
}

// ============================================================
// Tests — confirm const-generic primitives WORK at the type level.
// Compile-fail tests live in `tests/compile-fail/*.rs` (trybuild).
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    type Budget6 = SubstrateBudget<6, 8, 32_768, 8_192, 0>;

    #[test]
    fn page_within_bounds_constructs() {
        let _p = Page::<0, 6, Empty>::new();
        let _q = Page::<5, 6, Empty>::new();
    }

    #[test]
    fn page_lifecycle_round_trips() {
        let p0 = Page::<0, 6, Empty>::new();
        let p1 = p0.loader_fired();
        let p2 = p1.consumer_arrived();
        let p3 = p2.storer_consumed();
        let _p4 = p3.loader_fired();
    }

    #[test]
    fn scratch_within_budget_constructs() {
        let _r = ScratchRegion::<0, 64, 8_192, RmsNormScope>::new();
        let _r2 = ScratchRegion::<8_128, 64, 8_192, RmsNormScope>::new();
    }

    #[test]
    fn scratch_disjoint_passes_for_non_overlap() {
        let a = ScratchRegion::<0, 64, 8_192, RmsNormScope>::new();
        let b = ScratchRegion::<64, 64, 8_192, RmsNormScope>::new();
        let _ = a.disjoint_with(b);
    }

    #[test]
    fn mbarrier_phase_construction_passes() {
        let _ = MbarrierPhase::<0>::new();
        let _ = MbarrierPhase::<1>::new();
    }

    #[test]
    fn mbarrier_phase_assert_matches_passes() {
        let _ = MbarrierPhase::<0>::assert_matches::<4>();
        let _ = MbarrierPhase::<1>::assert_matches::<5>();
        let _ = MbarrierPhase::<0>::assert_matches::<0>();
    }

    #[test]
    fn iter_count_nonzero_constructs() {
        let _ = IterCount::<1>::new();
        let _ = IterCount::<8>::new();
    }

    #[test]
    fn page_id_within_bounds_constructs() {
        let _ = PageId::<0, 6>::new();
        let _ = PageId::<5, 6>::new();
    }

    #[test]
    fn edge_id_within_bounds_constructs() {
        let _ = EdgeId::<0, 4>::new();
        let _ = EdgeId::<3, 4>::new();
    }

    #[test]
    fn expected_count_nonzero_constructs() {
        let _ = ExpectedCount::<1>::new();
        let _ = ExpectedCount::<128>::new();
    }

    #[test]
    fn substrate_budget_constructs() {
        let _ = Budget6::new();
    }

    #[test]
    fn substrate_budget_accessors() {
        let b = Budget6::new();
        assert_eq!(b.num_pages(), 6);
        assert_eq!(b.num_consumer_warps(), 8);
        assert_eq!(b.page_size(), 32_768);
        assert_eq!(b.scratch_bytes(), 8_192);
        assert_eq!(b.num_edges(), 0);
    }
}
