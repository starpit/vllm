// SPDX-License-Identifier: Apache-2.0
//! Substrate-typestate vocabulary.
//!
//! Defines the proof-carrying types that variants of the typed
//! lowered form (in [`crate::nodes`]) use as load-bearing fields.
//! Each type discharges one of the substrate bug classes the
//! `MegaTape<S>` invariant promises (see `MEGA_IR_PLAN.md` §1):
//!
//! - [`Page<State>`] — page-slot lifecycle (#1, #2)
//! - [`ScratchRegion<Scope>`] + [`Scratch::within_budget`] +
//!   [`Scratch::disjoint`] — scratch budget and overlap (#4, #5)
//! - [`MbarrierPhase`] — mbarrier parity (#3)
//! - [`WarpRoleTag<R>`] — warp-role pairing (#6)
//!
//! ## Stable-Rust shape
//!
//! The plan's `MEGA_IR_PLAN.md` §1+§4 describes the IDEAL design as
//! const-generic IDs (`Page<const ID: u32, S, State>`,
//! `ScratchRegion<const OFFSET, const BYTES, Scope>`,
//! `MbarrierPhase<const P>`, `WarpRoleTag<const R>`). On stable
//! Rust, const-generic instantiation requires the constant to be
//! known at the user's source-compile time — but the proc-macro
//! computes IDs and offsets at proc-macro RUNTIME, so they can't
//! become const-generic parameters here.
//!
//! Per `MEGA_IR_PLAN.md` §1: "If the type system can't catch it on
//! stable, document the NIGHTLY TODO at the point of use and encode
//! the closest stable approximation (sealed witness traits +
//! private constructors)."
//!
//! The stable approximation in this file:
//! - **Type-level**: `State` (`Empty`/`Filled`/`Produced`), `Scope`,
//!   `R` (warp role) are TYPE PARAMETERS. Lifecycle transitions
//!   consume the typestate token and return a new state — the
//!   compiler enforces "no consumer reads on `Page<Empty>`" at
//!   type-check time.
//! - **Construction-time**: page IDs, scratch offsets, mbarrier
//!   phases are private `u32` fields inside typed wrappers. Their
//!   constructors take the substrate budget by reference and PANIC
//!   on bounds violations. Panics surface as proc-macro errors on
//!   the user's `#[forward]`.
//! - **Sealed**: every constructor is `pub(crate)`. Outside this
//!   crate, typed wrappers can only flow through the lowering
//!   function in [`crate::lower`].
//!
//! NIGHTLY TODO at every wrapper: when `adt_const_params` and
//! `generic_const_exprs` stabilize, lift IDs / offsets / phases
//! back into const-generic parameters per the plan's ideal.

#![allow(dead_code)]

use core::marker::PhantomData;

mod sealed {
    pub trait Sealed {}
}

// ============================================================
// Substrate budget — runtime-known per-variant constants.
// The plan's `Substrate` trait used associated `const`s; on stable
// Rust those force the substrate to be a concrete type at proc-macro
// compile time, which it isn't (the proc-macro computes per-variant
// values at proc-macro RUNTIME). A plain struct holds the same data
// and is passable by reference into constructors.
// ============================================================

/// Per-variant substrate budget. The proc-macro instantiates one of
/// these per emitted `.cu` from `(num_pages, num_consumer_warps,
/// page_size, scratch_bytes)` it computes from the schedule walker.
/// Constructors of typed wrappers in this module take a borrow of
/// this struct and bounds-check against its fields.
///
/// NIGHTLY TODO: when `adt_const_params` stabilizes, replace this
/// with a sealed `Substrate` trait whose impls carry per-variant
/// associated constants (the plan's original §6 vision).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubstrateBudget {
    num_pages: u32,
    num_consumer_warps: u32,
    page_size: u32,
    scratch_bytes: u32,
}

impl SubstrateBudget {
    /// Construct a substrate budget. Panics on zero fields (a
    /// kernel with zero pages or zero scratch is nonsensical).
    pub fn new(
        num_pages: u32,
        num_consumer_warps: u32,
        page_size: u32,
        scratch_bytes: u32,
    ) -> Self {
        assert!(num_pages > 0, "SubstrateBudget: num_pages must be > 0");
        assert!(
            num_consumer_warps > 0,
            "SubstrateBudget: num_consumer_warps must be > 0",
        );
        assert!(page_size > 0, "SubstrateBudget: page_size must be > 0");
        // scratch_bytes may legitimately be 0 (ops with no scratch).
        Self {
            num_pages,
            num_consumer_warps,
            page_size,
            scratch_bytes,
        }
    }

    pub fn num_pages(&self) -> u32 {
        self.num_pages
    }
    pub fn num_consumer_warps(&self) -> u32 {
        self.num_consumer_warps
    }
    pub fn page_size(&self) -> u32 {
        self.page_size
    }
    pub fn scratch_bytes(&self) -> u32 {
        self.scratch_bytes
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
/// Type-level `State` typestate: `Empty` → loader fills →
/// `Filled` → consumer reads + arrives → `Produced` → storer
/// reads → back to `Empty`. Transitions consume the token (no
/// `Clone`, no `Copy`, no `Send`, no `Sync`). A `Page<Filled>`
/// cannot be read in the storer position; a `Page<Empty>` cannot
/// be read at all.
///
/// Runtime `id` is private and bounds-checked at construction
/// against `SubstrateBudget::num_pages`. The const-generic ID form
/// is a NIGHTLY TODO — see module docs.
pub struct Page<State: IsLifecycleState> {
    id: u32,
    _state: PhantomData<*const State>,
}

impl Page<Empty> {
    /// Construct a fresh `Empty` page. Panics if `id >=
    /// substrate.num_pages` (bug class #1 enforcement).
    /// `pub(crate)` so only the lowering can mint pages.
    pub(crate) fn new(id: u32, substrate: &SubstrateBudget) -> Self {
        assert!(
            id < substrate.num_pages,
            "Page id {id} out of substrate budget num_pages={}",
            substrate.num_pages,
        );
        Self {
            id,
            _state: PhantomData,
        }
    }

    /// Loader-fired transition: `Empty` → `Filled`. Consumes the
    /// `Empty` token; the page can no longer be filled twice (bug
    /// class #2 enforcement: double-fill).
    pub(crate) fn loader_fired(self) -> Page<Filled> {
        Page {
            id: self.id,
            _state: PhantomData,
        }
    }
}

impl Page<Filled> {
    /// Consumer-arrived transition: `Filled` → `Produced`. Consumes
    /// the `Filled` token; the page can no longer be read by the
    /// consumer (bug class #2 enforcement: read-after-arrive).
    pub(crate) fn consumer_arrived(self) -> Page<Produced> {
        Page {
            id: self.id,
            _state: PhantomData,
        }
    }
}

impl Page<Produced> {
    /// Storer-consumed transition: `Produced` → `Empty`. The page
    /// is recyclable for the next iter.
    pub(crate) fn storer_consumed(self) -> Page<Empty> {
        Page {
            id: self.id,
            _state: PhantomData,
        }
    }
}

impl<State: IsLifecycleState> Page<State> {
    /// Raw page id — for emit-side string formatting only.
    pub fn id(&self) -> u32 {
        self.id
    }
}

// ============================================================
// Scratch regions (bug classes #4, #5).
// ============================================================

pub trait IsScratchScope: sealed::Sealed {}

/// A scratch-byte region inside the substrate's
/// `scratch[scratch_bytes]` buffer.
///
/// Type-level `Scope` disambiguates regions by lifetime — two
/// regions in different scopes don't need to be disjoint. Runtime
/// `offset` and `bytes` are private and validated at construction:
///
/// - `WithinBudget`: `offset + bytes ≤ substrate.scratch_bytes`
///   (bug class #5).
/// - Inter-region disjointness within a scope: combine via
///   [`Self::disjoint_with`] which checks byte ranges don't
///   overlap (bug class #4).
pub struct ScratchRegion<Scope: IsScratchScope> {
    offset: u32,
    bytes: u32,
    _scope: PhantomData<*const Scope>,
}

impl<Scope: IsScratchScope> ScratchRegion<Scope> {
    /// Construct a scratch region. Panics if `offset + bytes`
    /// overflows `u32` or exceeds `substrate.scratch_bytes` (bug
    /// class #5). `pub(crate)` so only the lowering can mint
    /// regions.
    pub(crate) fn new(offset: u32, bytes: u32, substrate: &SubstrateBudget) -> Self {
        let end = offset
            .checked_add(bytes)
            .expect("ScratchRegion: offset + bytes overflowed u32");
        assert!(
            end <= substrate.scratch_bytes,
            "ScratchRegion out of substrate scratch budget: \
             [{offset}, {end}) > SCRATCH_BYTES={}",
            substrate.scratch_bytes,
        );
        Self {
            offset,
            bytes,
            _scope: PhantomData,
        }
    }

    /// Validate this region is disjoint from `other` within the
    /// same scope. Panics on overlap (bug class #4). Returns both
    /// regions back so the caller can hold them simultaneously.
    /// `pub(crate)` so only the lowering can declare disjointness.
    pub(crate) fn disjoint_with(self, other: Self) -> (Self, Self) {
        let a_end = self.offset + self.bytes;
        let b_end = other.offset + other.bytes;
        let overlap = self.offset < b_end && other.offset < a_end;
        assert!(
            !overlap,
            "ScratchRegion overlap within scope: \
             A=[{}, {}) B=[{}, {})",
            self.offset, a_end, other.offset, b_end,
        );
        (self, other)
    }

    pub fn offset(&self) -> u32 {
        self.offset
    }
    pub fn bytes(&self) -> u32 {
        self.bytes
    }
}

// ============================================================
// Mbarrier phase (bug class #3).
// ============================================================

/// Mbarrier phase value (0 or 1). The substrate's `wait(sem, P)`
/// blocks until phase != P; each `arrive(sem)` flips parity.
///
/// Constructor [`Self::from_arrive_count`] requires the cumulative
/// arrive count `n` and panics if the requested phase doesn't
/// match `n & 1` (bug class #3 enforcement: phase mismatch across
/// iterations).
pub struct MbarrierPhase {
    phase: u32,
}

impl MbarrierPhase {
    /// Construct a phase from a cumulative arrive count. The
    /// resulting phase is `n & 1` — the only valid pairing for a
    /// `wait` after `n` arrives. Phase value is always 0 or 1.
    /// `pub(crate)` so only the lowering can mint phases.
    pub(crate) fn from_arrive_count(n: u32) -> Self {
        Self { phase: n & 1 }
    }

    /// Validate that `requested` is the parity of `n`. Panics on
    /// mismatch (bug class #3).
    pub(crate) fn assert_matches(requested: u32, n: u32) -> Self {
        let expected = n & 1;
        assert!(
            requested == expected,
            "MbarrierPhase: requested {requested} but cumulative arrive count {n} \
             requires phase {expected}",
        );
        Self { phase: expected }
    }

    pub fn phase(&self) -> u32 {
        self.phase
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
/// `u8` pinned at type construction time. Sealed for {0..3}; a
/// `WarpRoleTag<99>` doesn't satisfy `IsValidWarpRole` and can't
/// pass the bound.
///
/// This IS const-generic on stable Rust because each variant's
/// role assignment is fixed at type-construction time (the variant
/// declares which role does what). Page IDs and scratch offsets
/// are NOT const-generic on stable because they're computed at
/// proc-macro runtime.
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
// PageId — a validated, finalized page-slot id for storage in a
// `MegaNode` variant.
//
// Page<State> tracks lifecycle DURING the lowering walk via
// typestate transitions. After the op's iter completes, the
// caller has burned through the Empty→Filled→Produced→Empty
// cycle and the variant just needs to remember which slot id was
// used (for emit-time CUDA-string formatting). PageId carries
// that — a typed wrapper around u32 with bounds validation, no
// typestate, no transitions.
// ============================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PageId(u32);

impl PageId {
    /// Construct a PageId. Panics if `id >= substrate.num_pages`
    /// (bug class #1 enforcement). `pub(crate)` so only the
    /// lowering can mint ids; outside callers receive PageIds
    /// already-validated as part of MegaNode variant fields.
    pub(crate) fn new(id: u32, substrate: &SubstrateBudget) -> Self {
        assert!(
            id < substrate.num_pages,
            "PageId {id} out of substrate budget num_pages={}",
            substrate.num_pages,
        );
        Self(id)
    }

    pub fn raw(self) -> u32 {
        self.0
    }
}

// ============================================================
// PagePool — substrate-aware page allocator the lowering uses to
// hand out Page<Empty> tokens for ops to walk through their
// lifecycle. Tracks which physical slots are currently held by
// in-flight ops; refuses to hand out a slot that's already in
// use (bug class #2: cross-op page reuse with wrong parity).
// ============================================================

pub struct PagePool {
    in_use: Vec<bool>,
    substrate: SubstrateBudget,
}

impl PagePool {
    /// Construct an empty pool sized to `substrate.num_pages`.
    pub fn new(substrate: SubstrateBudget) -> Self {
        Self {
            in_use: vec![false; substrate.num_pages as usize],
            substrate,
        }
    }

    /// Take an `Empty` page with the given id. Panics if the id is
    /// out of bounds (bug class #1) or already in use (cross-op
    /// reuse, bug class #2). The op MUST `release` it after the
    /// lifecycle walk finishes.
    pub fn take(&mut self, id: u32) -> Page<Empty> {
        assert!(
            id < self.substrate.num_pages,
            "PagePool::take: id {id} out of bounds num_pages={}",
            self.substrate.num_pages,
        );
        assert!(
            !self.in_use[id as usize],
            "PagePool::take: page id {id} already in use by another op (cross-op reuse)",
        );
        self.in_use[id as usize] = true;
        Page::<Empty>::new(id, &self.substrate)
    }

    /// Return a page (any state — caller has already walked the
    /// lifecycle and we just need the slot id). The page is now
    /// available for the next op.
    pub fn release<S: IsLifecycleState>(&mut self, page: Page<S>) {
        let id = page.id() as usize;
        // The page slot is dropped (we don't keep typestate; we
        // just record it's free). Re-marking false is idempotent.
        self.in_use[id] = false;
    }

    pub fn substrate(&self) -> &SubstrateBudget {
        &self.substrate
    }
}

// ============================================================
// Public scope markers — variants that use scratch declare which
// scope their regions live in. Two regions in the same scope must
// be `disjoint_with`; two in different scopes don't constrain
// each other (they don't coexist in time).
// ============================================================

pub trait IsScratchScopePub: sealed::Sealed {}

/// Scratch scope for `RmsNorm`-family ops. Per-iter partial-sums
/// reduction lives in this scope; no other op shares it.
pub struct RmsNormScope;
impl sealed::Sealed for RmsNormScope {}
impl IsScratchScope for RmsNormScope {}
impl IsScratchScopePub for RmsNormScope {}

// ============================================================
// Tests — confirm the substrate-proof construction-time panics
// fire as documented.
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn budget() -> SubstrateBudget {
        SubstrateBudget::new(6, 8, 32_768, 8_192)
    }

    #[test]
    fn page_within_bounds_constructs() {
        let _p = Page::<Empty>::new(0, &budget());
        let _q = Page::<Empty>::new(5, &budget());
    }

    #[test]
    #[should_panic(expected = "Page id 6 out of substrate budget num_pages=6")]
    fn page_out_of_bounds_panics() {
        let _ = Page::<Empty>::new(6, &budget());
    }

    #[test]
    fn page_lifecycle_round_trips() {
        let p0 = Page::<Empty>::new(0, &budget());
        let p1 = p0.loader_fired();
        let p2 = p1.consumer_arrived();
        let p3 = p2.storer_consumed();
        // p3 is Empty again; can transition.
        let _p4 = p3.loader_fired();
    }

    #[test]
    fn scratch_within_budget_constructs() {
        let _r = ScratchRegion::<RmsNormScope>::new(0, 64, &budget());
        let _r2 = ScratchRegion::<RmsNormScope>::new(8128, 64, &budget());
    }

    #[test]
    #[should_panic(expected = "ScratchRegion out of substrate scratch budget")]
    fn scratch_out_of_budget_panics() {
        let _ = ScratchRegion::<RmsNormScope>::new(8129, 64, &budget());
    }

    #[test]
    #[should_panic(expected = "offset + bytes overflowed u32")]
    fn scratch_overflow_panics() {
        let b = SubstrateBudget::new(1, 1, 1, u32::MAX);
        let _ = ScratchRegion::<RmsNormScope>::new(u32::MAX, 1, &b);
    }

    #[test]
    fn scratch_disjoint_passes_for_non_overlap() {
        let a = ScratchRegion::<RmsNormScope>::new(0, 64, &budget());
        let b = ScratchRegion::<RmsNormScope>::new(64, 64, &budget());
        let _ = a.disjoint_with(b);
    }

    #[test]
    #[should_panic(expected = "ScratchRegion overlap within scope")]
    fn scratch_overlap_panics() {
        let a = ScratchRegion::<RmsNormScope>::new(0, 128, &budget());
        let b = ScratchRegion::<RmsNormScope>::new(64, 64, &budget());
        let _ = a.disjoint_with(b);
    }

    #[test]
    fn mbarrier_phase_from_count() {
        assert_eq!(MbarrierPhase::from_arrive_count(0).phase(), 0);
        assert_eq!(MbarrierPhase::from_arrive_count(1).phase(), 1);
        assert_eq!(MbarrierPhase::from_arrive_count(2).phase(), 0);
        assert_eq!(MbarrierPhase::from_arrive_count(7).phase(), 1);
    }

    #[test]
    fn mbarrier_phase_assert_matches_passes() {
        let _ = MbarrierPhase::assert_matches(0, 4);
        let _ = MbarrierPhase::assert_matches(1, 5);
    }

    #[test]
    #[should_panic(expected = "MbarrierPhase: requested 0 but cumulative arrive count 5")]
    fn mbarrier_phase_assert_mismatches_panics() {
        let _ = MbarrierPhase::assert_matches(0, 5);
    }
}
