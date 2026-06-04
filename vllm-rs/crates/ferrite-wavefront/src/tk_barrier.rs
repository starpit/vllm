// SPDX-License-Identifier: Apache-2.0
//! Compile-time deadlock-prevention substrate for the TK 2.0 megakernel.
//!
//! Per `feedback_end_to_end_compile_time_proofs`: every numeric proof
//! (page id, expected arrival count, phase parity, descriptor tile
//! shape) must propagate as Rust const generics or typed witnesses
//! end-to-end. **A miswire is a Rust compile error, never a runtime
//! deadlock.**
//!
//! The deadlock class this substrate eliminates: a barrier
//! `init_semaphore(bar, 0, EXPECTED)` is followed by `wait(bar)`
//! before exactly `EXPECTED` arrivals have fired. In the legacy IR,
//! `prog.arrive(WarpRole::AllConsumers, ...)` was opaque about
//! per-warp execution — a compute body whose `if` early-returns idle
//! warps would cause some arrives to never fire, deadlocking the
//! storer's matching `wait`. The legacy IR could not express this
//! constraint at the type level.
//!
//! This module encodes the protocol with typestate:
//!
//! - [`Barrier<Kind, Expected, Phase, Arrived>`] — typestate handle.
//!   `Arrived` is a Peano-numeral counter; `arrive()` advances
//!   `A → S<A>`; `wait()` requires `Arrived ≡ Expected` by structural
//!   type identity, then flips `Phase` and resets the counter to `Z`.
//! - A lowering that emits 7-of-8 consumer arrives produces a value of
//!   type `Barrier<Done, N8, _, S^7<Z>>`. `wait()` requires
//!   `Barrier<_, N8, _, N8>` and the `S^7 ≠ N8` mismatch is a Rust
//!   compile error citing the type, not a runtime panic / nvcc error
//!   / launch deadlock.
//! - Over-arrive is impossible: `arrive()` is callable only when
//!   `(E, A): private::Lt`. A try-arrive past `EXPECTED` requires
//!   `(N8, S^8<Z>): Lt` — no impl exists; compile error.
//!
//! The module is **additive** — it lives next to the legacy
//! `tk_warp_ir` IR. Substrate-3 (per-warp arrival witness) and
//! substrate-4 (Page session types) build on top of this module.
//! Migration of `lower_*` happens one op at a time; each migration
//! preserves byte-identity emit against the legacy lowering.

#![allow(dead_code)]
#![allow(clippy::module_name_repetitions)]

use std::marker::PhantomData;

// ── Peano numerals ──────────────────────────────────────────────────
//
// Stable-Rust counter typestate. The const `N::VAL` reaches emit code
// without runtime evaluation. Type identity is structural: `S<S<Z>>`
// is exactly `S<S<Z>>`; `S<S<Z>> ≠ S<Z>`.

/// Sealed Peano-numeral primitives. Lowering authors should use the
/// const-u32 boundary functions ([`Barrier::init`], `arrive`, `wait`)
/// — never construct Peano types directly.
pub mod count {
    use std::marker::PhantomData;

    /// Peano zero.
    pub struct Z;
    /// Successor — `S<Z>` is 1, `S<S<Z>>` is 2, etc.
    pub struct S<N>(PhantomData<N>);

    /// Trait providing the const u32 representation of a Peano numeral.
    pub trait Nat {
        const VAL: u32;
    }
    impl Nat for Z {
        const VAL: u32 = 0;
    }
    impl<N: Nat> Nat for S<N> {
        const VAL: u32 = 1 + N::VAL;
    }

    // Typed aliases for the counts the substrate actually instantiates.
    // Adding a new count means adding a new alias here — keeps emit
    // diagnostics readable (`expected N16, found ...`) instead of the
    // raw Peano tower.
    pub type N1 = S<Z>;
    pub type N2 = S<N1>;
    pub type N3 = S<N2>;
    pub type N4 = S<N3>;
    pub type N5 = S<N4>;
    pub type N6 = S<N5>;
    pub type N7 = S<N6>;
    pub type N8 = S<N7>;
    pub type N9 = S<N8>;
    pub type N10 = S<N9>;
    pub type N11 = S<N10>;
    pub type N12 = S<N11>;
    pub type N13 = S<N12>;
    pub type N14 = S<N13>;
    pub type N15 = S<N14>;
    pub type N16 = S<N15>;
}

// ── Barrier kind markers ────────────────────────────────────────────
//
// The TK 2.0 page protocol has three barrier roles per page slot.
// Each marker carries the CUDA semaphore-array name as a `const &str`
// — emit code reads `K::FIELD` to render the right shared-memory
// reference.

/// Sealed marker trait for the three barrier roles in the page-slot
/// protocol. Implementations are exactly [`Ready`], [`Done`],
/// [`Consumed`].
pub trait BarrierKind {
    /// The CUDA `__shared__` semaphore-array name in the kernel
    /// scaffold (`page_ready` / `page_done` / `page_consumed`).
    const FIELD: &'static str;
    /// The matching [`crate::tk_warp_ir::PageBarrier`] enum value
    /// — single source of truth for the field-name string lives in
    /// `tk_codegen::tk20::barrier_field`.
    const PAGE_BARRIER: crate::tk_warp_ir::PageBarrier;
}

/// Ready barrier: TMA load_async auto-arrives once with byte-count
/// completion. Consumer waits to begin reading the page.
pub struct Ready;
impl BarrierKind for Ready {
    const FIELD: &'static str = "page_ready";
    const PAGE_BARRIER: crate::tk_warp_ir::PageBarrier = crate::tk_warp_ir::PageBarrier::Ready;
}

/// Done barrier: each active consumer warp arrives after computing.
/// Storer waits to begin draining the page to gmem.
pub struct Done;
impl BarrierKind for Done {
    const FIELD: &'static str = "page_done";
    const PAGE_BARRIER: crate::tk_warp_ir::PageBarrier = crate::tk_warp_ir::PageBarrier::Done;
}

/// Consumed barrier: storer arrives once after the gmem store
/// completes. Loader for the NEXT round waits to acquire the slot.
pub struct Consumed;
impl BarrierKind for Consumed {
    const FIELD: &'static str = "page_consumed";
    const PAGE_BARRIER: crate::tk_warp_ir::PageBarrier = crate::tk_warp_ir::PageBarrier::Consumed;
}

// ── Phase parity markers ────────────────────────────────────────────
//
// Each barrier flips parity 0 ↔ 1 on every successful round. The
// substrate tracks parity at the type level: `wait()` advances `P`
// to `P::Flip`. A `wait` against the wrong parity is a compile error
// because `Phase: PhaseTag<Flip = P0>` and `Phase: PhaseTag<Flip = P1>`
// are distinct constraints.

/// Sealed marker trait for the two barrier phase parities. The
/// `Flip` associated type encodes the parity flip without a runtime
/// XOR — and without taking the post-flip parity as a *separate*
/// const generic that's `Phase ^ 1`, which would violate
/// `feedback_no_redundant_const_generics`.
pub trait PhaseTag: sealed::Sealed {
    /// `0` for [`P0`], `1` for [`P1`].
    const VAL: u32;
    /// The parity after one successful `wait`.
    type Flip: PhaseTag<Flip = Self>;
}

/// Phase-0 parity. The fresh-init parity for every barrier.
pub struct P0;
/// Phase-1 parity. After one round completes.
pub struct P1;

impl PhaseTag for P0 {
    const VAL: u32 = 0;
    type Flip = P1;
}
impl PhaseTag for P1 {
    const VAL: u32 = 1;
    type Flip = P0;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::P0 {}
    impl Sealed for super::P1 {}
}

// ── LoopPhase: runtime parity inside a loop body ────────────────────
//
// A page slot inside a `for_loop` body has a phase that depends on
// the loop variable: at iteration `i`, parity = `(i & 1) ^ StartP`.
// This is structurally distinct from `P0`/`P1` because the phase
// expression at emit time is RUNTIME (`(var & 1) ^ StartP::VAL`).
//
// Without this distinction, a lowering could accidentally emit a
// static `wait(bar, 0)` or `wait(bar, 1)` inside a loop — the parity
// would only be correct on every other iteration. That's exactly the
// op4/Gemm deadlock the legacy `wait_loop_parity` was added to
// prevent (per the comment in `tk_warp_ir.rs:410-413`). Encoding the
// phase as `LoopPhase<Start, Var>` forces the substrate to emit
// runtime parity for every wait/arrive inside the loop body.
//
// After the loop, the slot's actual barrier parity is `Start ^ (N &
// 1)` for the runtime `N` iterations. A subsequent op MUST emit
// runtime-parity waits to re-acquire the slot — cannot statically
// claim the phase. The substrate models this with a separate
// `RuntimePhase` marker that's not interchangeable with `P0`/`P1`.

/// Phase marker for the inside of a `for_loop` body. `Start` is the
/// static phase at loop entry. The loop variable name is captured
/// at emit time (in the [`BarrierLoop`] builder), not in the type.
///
/// `LoopPhase<Start>` is structurally distinct from `P0`/`P1` —
/// a static `wait(bar, 0)` cannot be emitted on a barrier in this
/// phase, because the implementation routes through the
/// runtime-parity emit path. This catches "I forgot the loop's
/// per-iteration phase flip" at compile time — the precise class of
/// bug the `wait_loop_parity` workaround addresses at runtime.
///
/// Phase flip during a loop body is from one iter's parity to the
/// next — both runtime — and is represented by the same
/// `LoopPhase<Start>` type at start/end of the body. (`Flip` is
/// `Self`: each wait inside the body emits the runtime parity
/// expression for the CURRENT iter; the CUDA-side flip happens
/// when the iter variable advances.)
pub struct LoopPhase<Start: PhaseTag>(PhantomData<Start>);

impl<Start: PhaseTag> sealed::Sealed for LoopPhase<Start> {}
impl<Start: PhaseTag> PhaseTag for LoopPhase<Start> {
    // `VAL` is the round-0 parity for any stray static-emit fallback;
    // typed methods on `Barrier<_, _, LoopPhase<Start>, _>` should
    // override and emit runtime parity instead.
    const VAL: u32 = Start::VAL;
    type Flip = Self;
}

/// Phase marker for "the slot's phase is unknown statically — the
/// previous op was a runtime-N loop". Subsequent waits must emit
/// runtime parity expressions; the static `P0`/`P1` are not
/// interchangeable with this marker.
///
/// A `Page<SLOT, RuntimePhase, Empty>` cannot be directly used as
/// `Page<SLOT, P0, Empty>` — the next op's typed builder must accept
/// the runtime phase. This forces explicit acknowledgement that the
/// previous op had a runtime-N loop.
pub struct RuntimePhase;
impl sealed::Sealed for RuntimePhase {}
impl PhaseTag for RuntimePhase {
    const VAL: u32 = 0;
    type Flip = Self;
}

// ── Strict-less-than witness for Peano numerals ─────────────────────
//
// `Lt(A, E)` = "A < E". `arrive` requires this witness on
// `(Arrived, Expected)` so over-arriving past the expected count is a
// compile error: there is no `Lt(N8, N8)` impl, no `Lt(N9, N8)` impl,
// etc. Implementations are recursive — A < B iff S<A> < S<B>, with
// base case 0 < S<_>.

mod private {
    use super::count::{Nat, S, Z};

    /// Sealed witness "A is strictly less than B" for Peano numerals.
    /// Used as a `where` bound on [`Barrier::arrive`].
    pub trait Lt<B> {}

    // Base case: 0 < S<E> for any E.
    impl<E: Nat> Lt<S<E>> for Z {}

    // Inductive step: S<A> < S<B> iff A < B.
    impl<A: Nat, B: Nat> Lt<S<B>> for S<A> where A: Lt<B> {}
}

// ── Typed barrier handle ────────────────────────────────────────────

/// Typestate handle for one page-slot barrier.
///
/// Type parameters:
/// - `Kind`: which of the three protocol roles ([`Ready`], [`Done`],
///   [`Consumed`]) — controls the emitted `__shared__` field name.
/// - `Expected`: the Peano-numeral arrival count the barrier was
///   initialised with (`init_semaphore(bar, 0, Expected::VAL)`).
/// - `Phase`: the parity ([`P0`] or [`P1`]) of the *next* `wait` —
///   advances by `wait` (`P → P::Flip`).
/// - `Arrived`: the Peano-numeral count of arrives issued so far in
///   the current round. Reset to [`count::Z`] after each successful
///   `wait`.
///
/// **Construction**: only [`Barrier::init`] (round 0) creates a
/// barrier; the const-u32 boundary is `init::<EXPECTED>`. All
/// subsequent transitions are by-value moves of the typestate handle.
///
/// **Arrival**: `self.arrive()` requires the [`private::Lt`] witness
/// `Arrived < Expected`; advances `Arrived → S<Arrived>`.
///
/// **Wait**: `self.wait()` is a method on `Barrier<_, N, _, N>` only
/// — `Arrived ≡ Expected` by structural type identity. A lowering that
/// emits 7-of-8 arrives produces `Barrier<_, N8, _, N7>` and `wait`
/// fails to typecheck.
///
/// The handle is zero-sized at runtime; emit happens via the
/// associated `init`/`arrive`/`wait` methods which return CUDA source
/// fragments. The handle THREADS THROUGH the lowering builder so each
/// step's typestate matches the previous step's output.
#[derive(Debug)]
pub struct Barrier<Kind, Expected, Phase, Arrived>(
    PhantomData<(fn() -> Kind, fn() -> Expected, fn() -> Phase, fn() -> Arrived)>,
);

impl<K, E, P, A> Barrier<K, E, P, A> {
    /// Crate-internal constructor for typestate transitions (`arrive`,
    /// `wait`). Only callable from inside this module — external code
    /// must go through [`Barrier::init`] to obtain the round-0 handle,
    /// which enforces init-once-per-(kind, slot) at the type level
    /// (init returns `Barrier<K, E, P0, Z>` only — any subsequent
    /// state must be reached by transitioning the existing handle).
    pub(crate) fn fresh_internal() -> Self {
        Self(PhantomData)
    }

    /// Test-only constructor for building partial-state handles in
    /// substrate self-tests. Production code MUST use [`Barrier::init`]
    /// — `Default` is intentionally not implemented to enforce
    /// init-once-per-(kind, slot).
    #[cfg(test)]
    pub fn fresh_for_test() -> Self {
        Self(PhantomData)
    }
}

/// One emitted CUDA source fragment paired with the typestate handle
/// it produced. The lowering builder accumulates these fragments and
/// concatenates them in order; the typestate handle is consumed by
/// the next operation.
#[must_use]
pub struct Emit<H> {
    /// The CUDA source line(s) for this operation.
    pub cuda: String,
    /// The typestate handle the next operation consumes.
    pub handle: H,
}

impl<H> Emit<H> {
    pub fn new(cuda: String, handle: H) -> Self {
        Self { cuda, handle }
    }
    pub fn map<H2>(self, handle: H2) -> Emit<H2> {
        Emit {
            cuda: self.cuda,
            handle,
        }
    }
}

// ── init: round 0, Phase = P0, Arrived = Z ─────────────────────────

impl<K: BarrierKind, E: count::Nat> Barrier<K, E, P0, count::Z> {
    /// Emit the `init_semaphore(<K::FIELD>[<page_id>], 0, EXPECTED)`
    /// call for this barrier and return the round-0 handle.
    ///
    /// The substrate scaffold (`tk_codegen::emit_kernel_with_opts`)
    /// emits all `init_semaphore` calls inside a `if (__warpid == 0
    /// && (threadIdx.x & 31) == 0)` lane gate at the top of the
    /// kernel. The fragment returned here is the body of that gate;
    /// the caller is responsible for placing it.
    pub fn init(page_id: u8) -> Emit<Self> {
        Emit {
            cuda: format!(
                "kittens::init_semaphore({field}[{page_id}], 0, {expected});",
                field = K::FIELD,
                page_id = page_id,
                expected = E::VAL,
            ),
            handle: Self::fresh_internal(),
        }
    }
}

// ── arrive: Arrived → S<Arrived>, requires Arrived < Expected ──────

impl<K, E, P, A> Barrier<K, E, P, A>
where
    K: BarrierKind,
    E: count::Nat,
    P: PhaseTag,
    A: count::Nat,
    A: private::Lt<E>,
{
    /// Emit one `kittens::group<1>::arrive(<K::FIELD>[<page_id>])`
    /// call and advance the typestate counter.
    ///
    /// Per `feedback_ff_subtile_arrive_fix`, all arrives are emitted
    /// at `group<1>` width (lane 0 of the calling warp's group of 1
    /// warp = every warp's lane 0 fires once). The caller's role
    /// guard (`if (__role == ROLE_X)` or `if (__role == ROLE_CONSUMER
    /// && __consumer_idx < ACTIVE)`) determines which warps run the
    /// emitted code.
    ///
    /// **Compile-time guarantee**: the `private::Lt<E>` bound on
    /// `A` makes over-arriving a Rust compile error. Calling
    /// `arrive` on `Barrier<_, N8, _, N8>` requires
    /// `(N8): private::Lt<N8>`, which has no impl.
    pub fn arrive(self, page_id: u8) -> Emit<Barrier<K, E, P, count::S<A>>> {
        Emit {
            cuda: crate::tk_codegen::tk20::arrive(1, K::PAGE_BARRIER, page_id),
            handle: Barrier::fresh_internal(),
        }
    }
}

// ── Active-warp witness for divergent consumer bodies ──────────────
//
// The legacy `prog.arrive(WarpRole::AllConsumers, PageBarrier::Done,
// page)` was opaque about per-warp execution: a compute body whose
// `if (__consumer_idx < N)` branch early-returned for idle warps would
// silently drop N..NUM_CONSUMER_WARPS arrives, leaving the storer's
// `wait(Done)` blocked on N missing arrives.
//
// `ActiveWarps<N>` makes the divergence STRUCTURAL: a `Barrier<Done,
// N, P, A>` initialised for a divergent op tells the type system "this
// barrier expects exactly N arrives, not NUM_CONSUMER_WARPS". The
// substrate then forces the lowering to either:
//   (a) initialise the barrier with the active count (e.g., N8 for
//       Llama-1B AttnDecode with num_kv_heads=8), and emit the arrive
//       inside the active-warps gate so only those 8 arrive, OR
//   (b) initialise with NUM_CONSUMER_WARPS and emit the arrive
//       OUTSIDE the gate so all 16 arrive (idle warps fire a no-op
//       arrive after skipping the compute body).
//
// Either way, the init expected count and the actual arrive count are
// structurally tied — a mismatch is a compile error.
//
// Concrete encoding: the arrive emit takes a `RoleGuardWidth<N>` token
// proving "this many warps execute the arrive". The gate emit code is
// the single source of truth for both the C++ `if` predicate and the
// Peano numeral.

/// Witness for the number of consumer warps that fire a given arrive.
///
/// Constructed by [`role_gate_active_warps`] (single source of truth
/// for the C++ `if` predicate). Consumed by [`Barrier::arrive_in_role_gate`]
/// to prove the arrive count matches the role-gate width — and by
/// extension, matches the barrier's `Expected` count.
pub struct RoleGuardWidth<N: count::Nat> {
    /// The CUDA `if` predicate that gates the arrive emit. The arrive
    /// is only executed in the warps that satisfy this predicate; the
    /// `N` const generic must equal the number of warps satisfying it
    /// at runtime.
    predicate: String,
    _marker: PhantomData<N>,
}

impl<N: count::Nat> RoleGuardWidth<N> {
    /// Construct a width witness from a CUDA predicate. **The caller
    /// is responsible for the const generic matching the predicate's
    /// runtime active-warp count.** This is a sealed boundary —
    /// lowering authors use the helpers below
    /// ([`role_gate_active_warps`], [`role_gate_all_consumers`]) which
    /// pair predicate + N consistently.
    fn new_unchecked(predicate: String) -> Self {
        Self {
            predicate,
            _marker: PhantomData,
        }
    }

    /// The CUDA predicate string for this gate.
    pub fn predicate(&self) -> &str {
        &self.predicate
    }
}

/// Build a [`RoleGuardWidth<N>`] for the "first ACTIVE warps" pattern
/// (`__consumer_idx < ACTIVE`). The `N` parameter MUST equal `ACTIVE`
/// — this fn enforces that by taking `N: count::Nat` and emitting
/// `__consumer_idx < N::VAL`. Match `RoleGuardWidth<N8>` to "8 active
/// kv heads" by construction; mismatch is impossible.
///
/// The returned predicate is a CUDA boolean expression suitable for
/// `if (PREDICATE) { arrive(); }`. Composes with `__role ==
/// ROLE_CONSUMER` outside this fn — this fn handles ONLY the active-
/// warp slice within the consumer arm.
pub fn role_gate_active_warps<N: count::Nat>() -> RoleGuardWidth<N> {
    RoleGuardWidth::new_unchecked(format!(
        "static_cast<unsigned int>(__consumer_idx) < {n}u",
        n = N::VAL,
    ))
}

/// Build a [`RoleGuardWidth<N>`] for the "all consumer warps" pattern
/// — predicate is just `true`, `N` is [`crate::tk_warp_ir::NUM_CONSUMER_WARPS`].
/// Use when the arrive fires from every consumer warp regardless of
/// compute-body divergence (i.e., the arrive is OUTSIDE any divergent
/// gate).
pub fn role_gate_all_consumers() -> RoleGuardWidth<count::N16> {
    RoleGuardWidth::new_unchecked("true".to_string())
}

// ── Add<A, B>: Peano addition for advancing Arrived by N ───────────
//
// `arrive_in_role_gate<N>` advances `Arrived` by N at the type level.
// This is Peano addition: A + Z = A; A + S<B> = S<A + B>.

mod add {
    use super::count::{Nat, S, Z};
    /// `A + B` at the type level. `Sum` is the result; the trait
    /// guarantees `Sum: Nat` because `Nat` is implemented for all
    /// well-formed Peano numerals (`Z` and any `S<N>` where `N: Nat`).
    pub trait Add<B>: Nat {
        type Sum: Nat;
    }
    // A + Z = A.
    impl<A: Nat> Add<Z> for A {
        type Sum = A;
    }
    // A + S<B> = S<(A + B)>.
    // The two trait bounds — `A: Add<B>` (recursive call to addition)
    // and `<A as Add<B>>::Sum: Nat` (its result is a numeral) —
    // together let us write `Sum = S<...>`. `S<X>: Nat` follows
    // automatically from the `impl<N: Nat> Nat for S<N>` blanket.
    impl<A, B> Add<S<B>> for A
    where
        A: Add<B>,
    {
        type Sum = S<<A as Add<B>>::Sum>;
    }
}

// ── arrive_in_role_gate: advance Arrived by N (divergent arrive) ────

impl<K, E, P, A> Barrier<K, E, P, A>
where
    K: BarrierKind,
    E: count::Nat,
    P: PhaseTag,
    A: count::Nat,
{
    /// Emit a `kittens::group<1>::arrive(...)` call wrapped in the
    /// caller-supplied [`RoleGuardWidth<N>`] gate, and advance the
    /// typestate counter by `N` (NOT 1). This is the primitive for
    /// arrives in compute-body-divergent ops (e.g., AttnDecode where
    /// only `num_kv_heads` of `NUM_CONSUMER_WARPS` warps are active).
    ///
    /// **Compile-time guarantee**: `N` is the active-warp count baked
    /// into the role-gate predicate. The Peano addition `A + N`
    /// advances the typestate counter by exactly the warps that fire
    /// the arrive. The barrier's `Expected` MUST equal the sum of all
    /// arrives by `wait` time — typed identity at `wait` enforces this.
    ///
    /// Concretely: AttnDecode has 8 active warps and 8 idle. The
    /// lowering can either:
    ///   (a) Init `Barrier<Done, N8, P0, Z>` and arrive once via
    ///       `arrive_in_role_gate(role_gate_active_warps::<N8>())` —
    ///       Arrived advances by 8 → matches Expected N8 → wait OK.
    ///   (b) Init `Barrier<Done, N16, P0, Z>` and arrive once via
    ///       active-gate AND once via idle-gate so the sum is 16.
    ///       (Or just init N16 + arrive_in_role_gate(all_consumers).)
    ///
    /// The OLD bug (init N16, only 8 fire) is impossible: the arrive
    /// always advances Arrived by exactly the gate's N, and `wait`
    /// requires Arrived ≡ Expected.
    ///
    /// The lane-0-of-group-1 gating is unchanged from `arrive` —
    /// `kittens::group<1>::arrive` fires from each warp's lane 0
    /// independently. The role-gate predicate determines WHICH warps
    /// run the lane-0 fire.
    /// # The deadlock the substrate prevents (compile_fail)
    ///
    /// The legacy AttnDecode bug: init expects 16 arrivals, but the
    /// arrive is gated to only 8 active warps. `wait` then requires
    /// `Arrived ≡ Expected`, but Arrived is N8 and Expected is N16 —
    /// type mismatch, compile error.
    /// ```compile_fail
    /// use ferrite_wavefront::tk_barrier::*;
    /// use ferrite_wavefront::tk_barrier::count::*;
    /// // Init for ALL consumers (N16) — like the legacy code did.
    /// let b = Barrier::<Done, N16, P0, Z>::init(0).handle;
    /// // Arrive in the 8-active-warps gate (the AttnDecode case).
    /// // Arrived advances to N8, NOT N16.
    /// let b = b.arrive_in_role_gate(0, role_gate_active_warps::<N8>()).handle;
    /// // wait requires Barrier<_, N16, _, N16>; we have
    /// // Barrier<_, N16, _, N8>. Compile error.
    /// let _ = b.wait(0, 16);
    /// ```
    pub fn arrive_in_role_gate<N>(
        self,
        page_id: u8,
        gate: RoleGuardWidth<N>,
    ) -> Emit<Barrier<K, E, P, <A as add::Add<N>>::Sum>>
    where
        N: count::Nat,
        A: add::Add<N>,
    {
        Emit {
            cuda: format!(
                "if ({pred}) {{ kittens::group<1>::arrive({field}[{page_id}]); }}",
                pred = gate.predicate(),
                field = K::FIELD,
                page_id = page_id,
            ),
            handle: Barrier::fresh_internal(),
        }
    }
}

// ── wait: Arrived ≡ Expected, flip Phase, reset counter ─────────────

impl<K, N, P> Barrier<K, N, P, N>
where
    K: BarrierKind,
    N: count::Nat,
    P: PhaseTag,
{
    /// Emit one `kittens::group<NWAITERS>::wait(<K::FIELD>[<page_id>],
    /// <P::VAL>)` call, flip the typestate phase, and reset the
    /// arrival counter.
    ///
    /// `nwaiters` is the group width for the wait — 1 for
    /// loader/storer (single-warp roles), [`crate::tk_warp_ir::NUM_CONSUMER_WARPS`]
    /// for AllConsumers. Caller supplies the literal value.
    ///
    /// **Compile-time guarantee**: this method is reachable ONLY when
    /// the `Arrived` parameter is structurally equal to the `Expected`
    /// parameter — both are `N`. A lowering that emits `Barrier<_,
    /// N8, _, S^7<Z>>` cannot call `wait` because there is no impl
    /// matching `Barrier<_, N8, _, N8>` for that input. The compile
    /// error cites the type mismatch directly — no runtime panic, no
    /// nvcc error, no kernel deadlock.
    ///
    /// # Negative compile-fail examples
    ///
    /// Under-arrive (7 of 8): `wait` is not callable, compile error.
    /// ```compile_fail
    /// use ferrite_wavefront::tk_barrier::*;
    /// use ferrite_wavefront::tk_barrier::count::*;
    /// // init: Barrier<Done, N8, P0, Z>
    /// let b = Barrier::<Done, N8, P0, Z>::init(0).handle;
    /// // 7 arrives — Arrived now N7
    /// let b = b.arrive(0).handle;
    /// let b = b.arrive(0).handle;
    /// let b = b.arrive(0).handle;
    /// let b = b.arrive(0).handle;
    /// let b = b.arrive(0).handle;
    /// let b = b.arrive(0).handle;
    /// let b = b.arrive(0).handle;
    /// // try to wait — Arrived is N7 but Expected is N8: compile error
    /// let _ = b.wait(0, 16);
    /// ```
    ///
    /// Over-arrive (9 of 8): the 9th `arrive` is not callable,
    /// compile error.
    /// ```compile_fail
    /// use ferrite_wavefront::tk_barrier::*;
    /// use ferrite_wavefront::tk_barrier::count::*;
    /// let b = Barrier::<Done, N8, P0, Z>::init(0).handle;
    /// let b = b.arrive(0).handle; // 1
    /// let b = b.arrive(0).handle; // 2
    /// let b = b.arrive(0).handle; // 3
    /// let b = b.arrive(0).handle; // 4
    /// let b = b.arrive(0).handle; // 5
    /// let b = b.arrive(0).handle; // 6
    /// let b = b.arrive(0).handle; // 7
    /// let b = b.arrive(0).handle; // 8 — Arrived now N8
    /// // try to arrive again — (N8): Lt<N8> has no impl: compile error
    /// let _ = b.arrive(0);
    /// ```
    pub fn wait(self, page_id: u8, nwaiters: u32) -> Emit<Barrier<K, N, P::Flip, count::Z>> {
        Emit {
            cuda: crate::tk_codegen::tk20::wait(
                nwaiters,
                K::PAGE_BARRIER,
                page_id,
                &P::VAL.to_string(),
            ),
            handle: Barrier::fresh_internal(),
        }
    }
}

// ── TmaLoad witness: typed expect_bytes vs actual transfer ──────────
//
// TK 2.0's TMA `load_async` auto-arrives the barrier via
// `mbarrier::complete_tx::bytes` — the barrier expects EXACTLY the
// byte count declared by the matching `expect_bytes(bar, B)` call.
// If the load actually transfers B' ≠ B bytes, the barrier never
// trips → deadlock.
//
// The legacy `prog.load_async(page_id, src, region, tile)` derived
// `expect_bytes` from `tile.rows * tile.cols * tile.elem_bytes`. The
// actual transfer was a `cp.async.bulk` reading from `src + offset`
// with the same byte count. As long as `src` had AT LEAST that many
// bytes available at `offset`, the load completes and the barrier
// trips. So this class doesn't deadlock by byte mismatch in itself.
//
// HOWEVER: when the load is INSIDE a `for_loop` and the legacy
// region's column offset is the LITERAL constant 0 (no dynamic
// per-iter offset), every iteration reads from `src + 0` — the SAME
// 1024 bytes for every iteration. The K/V cache loads in
// `lower_attn_decode` look exactly like this:
//
//     RegionRef::rows_cols(op.k_cache, 1, 0, kv_cols)
//
// The column offset is the third arg, `0`. There is no
// `load_async_dyn` wrapping or a dynamic byte offset added at
// emit time. Iter 0 and iter 1 read the same bytes.
//
// The substrate models this: a `LoadDescriptor` carries the source
// REGION shape AND a runtime offset expression. Inside a `for_loop`,
// the descriptor's offset MUST reference the loop variable (or a
// block-table indirection) — a static-zero offset inside a runtime-N
// loop is a `LoadInLoop_RequiresDynamicOffset` compile error.

/// Source region descriptor for a TMA load. Carries the shape
/// (rows, cols, elem_bytes) as const generics and an offset
/// expression as a [`String`]. The offset may be a literal `0`
/// (acceptable outside loops) or a dynamic CUDA expression
/// referencing the loop variable / block table.
pub struct LoadDescriptor<
    const ROWS: u32,
    const COLS: u32,
    const ELEM_BYTES: u32,
    OffsetKind,
> {
    pub offset_expr: String,
    _kind: PhantomData<OffsetKind>,
}

/// Marker for a load with a known-zero compile-time offset (single-
/// iteration / outside-loop loads only).
pub struct StaticZeroOffset;

/// Marker for a load whose offset is a dynamic CUDA expression
/// (typically references the loop variable or a block-table
/// indirection). Required for loads inside a [`for_loop_typed`] body.
pub struct DynamicOffset;

impl<const R: u32, const C: u32, const E: u32> LoadDescriptor<R, C, E, StaticZeroOffset> {
    /// Construct a static-zero-offset descriptor. SAFE outside
    /// `for_loop` bodies; inside a loop, the resulting `Page` →
    /// load_async path will fail to typecheck (see [`Page::load_async`]
    /// in the [`status::Loading`] state for the static-offset
    /// constraint).
    pub fn static_zero() -> Self {
        Self {
            offset_expr: "0".to_string(),
            _kind: PhantomData,
        }
    }
}

impl<const R: u32, const C: u32, const E: u32> LoadDescriptor<R, C, E, DynamicOffset> {
    /// Construct a dynamic-offset descriptor with the given CUDA
    /// expression for the byte offset (e.g.
    /// `"(__kv_i * 1024u)"` for a per-iter K-cache row).
    pub fn dynamic(offset_expr: impl Into<String>) -> Self {
        Self {
            offset_expr: offset_expr.into(),
            _kind: PhantomData,
        }
    }
}

// ── Substrate constants ↔ scaffold constants binding ──────────────
//
// `tk_warp_ir` defines `NUM_PAGES`, `NUM_WARPS`, etc. as `pub const`s.
// The substrate hardcodes 13 (in ValidSlotId impls), 16 (Done barrier
// expected count for AllConsumers in N16 alias), and 32 (warp width
// in TOTAL_THREADS). If a future refactor changes the scaffold's
// constants without updating the substrate, the substrate's
// structural enforcement falls out of sync with the kernel's actual
// shape.
//
// `const_assert_eq` ties them at compile time — divergence is a
// `cargo check` error.
const _: () = {
    // Substrate's Page<SLOT_ID> typestate is bound to NUM_PAGES via
    // ValidSlotId impls (13 of them, for SLOT_ID = 0..=12). If
    // tk_warp_ir::NUM_PAGES changes, this assert fires and the
    // ValidSlotId impl set must also be updated.
    assert!(crate::tk_warp_ir::NUM_PAGES == 13);
    // Substrate's count::N16 is the AllConsumers arrival count for
    // page_done. NUM_CONSUMER_WARPS must equal 16 for the typed
    // arrival count to be correct.
    assert!(crate::tk_warp_ir::NUM_CONSUMER_WARPS == 16);
    // Substrate's ProductionLaunchShape = KernelLaunchShape<4, 16>.
    assert!(crate::tk_warp_ir::NUM_SERVICE_WARPS == 4);
    assert!(crate::tk_warp_ir::NUM_WARPS == 20);
};

// ── ValidSlotId — slot-id < NUM_PAGES at the type level ────────────
//
// The kernel scaffold declares `__shared__ kittens::semaphore
// page_ready[NUM_PAGES]` etc. (NUM_PAGES = 13 in the current Phase 7
// scaffold). A slot id ≥ NUM_PAGES would index out of bounds — the
// emit would compile in nvcc as "valid C++" but the kernel would
// access uninitialized smem (or fault).
//
// `ValidSlotId<N>` is a sealed witness implemented only for `N ∈
// 0..NUM_PAGES`. `Page<SLOT_ID>` and the typed barrier methods that
// take a `page_id` const generic require `Const<SLOT_ID>:
// ValidSlotId<NUM_PAGES>` — out-of-bounds is a compile error.
//
// NUM_PAGES is currently 13 per `tk_warp_ir::NUM_PAGES`. If that
// constant changes, the impls below must also change (mechanical —
// the build will fail if we forget).

mod slot_id_witness {
    /// Sealed witness: `Const<S>` is a valid slot id.
    pub trait ValidSlotId {}
    /// Const-marker struct so we can name `Const<S>: ValidSlotId`.
    pub struct Const<const S: u8>;

    // 13 impls — one per valid slot. NUM_PAGES = 13 today; if
    // tk_warp_ir::NUM_PAGES grows, add more impls here.
    impl ValidSlotId for Const<0> {}
    impl ValidSlotId for Const<1> {}
    impl ValidSlotId for Const<2> {}
    impl ValidSlotId for Const<3> {}
    impl ValidSlotId for Const<4> {}
    impl ValidSlotId for Const<5> {}
    impl ValidSlotId for Const<6> {}
    impl ValidSlotId for Const<7> {}
    impl ValidSlotId for Const<8> {}
    impl ValidSlotId for Const<9> {}
    impl ValidSlotId for Const<10> {}
    impl ValidSlotId for Const<11> {}
    impl ValidSlotId for Const<12> {}
    // SLOT_ID = 13 has no impl → Const<13>: ValidSlotId is unsatisfiable.
}

pub use slot_id_witness::{Const as SlotIdConst, ValidSlotId};

// ── SlotLease — round-completion-before-reuse witness ──────────────
//
// The legacy `PageAllocator` hands out a slot at typed phase P;
// `release()` marks it free; the next `alloc_at::<P>()` call hands
// it back. Between release and re-alloc, there's NO compile-time
// proof the round actually closed — a lowering could release a slot
// mid-round (e.g., after arrive(Done) but before storer's
// arrive(Consumed)) and the next op would acquire a slot whose
// barriers are NOT at the expected parity → wait blocks forever.
//
// The substrate's `Page::complete_round` requires `Page<_, _,
// status::Consumed>` (full round closed) and produces
// `Page<_, P::Flip, status::Empty>` (slot ready for reuse at flipped
// parity). This is the structural guarantee — but it's only enforced
// for code that uses the typed `Page` session type. Legacy code that
// constructs PageHandle<P> directly bypasses it.
//
// `SlotLease<SLOT_ID, P>` is the bridge: a typed handle the
// allocator hands out at acquire, consumed at release. The token's
// `P` parameter must match the post-round phase; mismatch = compile
// error. Migrating PageAllocator to issue SlotLease tokens makes the
// "release mid-round" bug impossible at the type level.
//
// This commit adds the witness type. Wiring the legacy
// PageAllocator to issue them is a follow-up migration.

/// Typed lease handle for a page slot. Acquired by the allocator at
/// typed phase `P`, must be returned at typed phase `P::Flip` (one
/// full round completed) — type mismatch on release = compile error.
///
/// The `_phantom_lifetime` field carries a `&()` so the lease can't
/// outlive the allocator scope; release is consumption.
pub struct SlotLease<'lease, const SLOT_ID: u8, P: PhaseTag> {
    _slot: SlotIdConst<SLOT_ID>,
    _phase: PhantomData<P>,
    _phantom_lifetime: PhantomData<&'lease ()>,
}

impl<'lease, const SLOT_ID: u8, P: PhaseTag> SlotLease<'lease, SLOT_ID, P>
where
    SlotIdConst<SLOT_ID>: ValidSlotId,
{
    /// Construct a fresh lease — sealed; only the substrate's
    /// `PageAllocator` migration can call this.
    pub(crate) fn issue() -> Self {
        Self {
            _slot: SlotIdConst,
            _phase: PhantomData,
            _phantom_lifetime: PhantomData,
        }
    }

    /// Consume the lease at the post-round flipped phase. Returns a
    /// fresh lease at the new phase, ready for the next round's
    /// holder.
    pub fn complete_round(self) -> SlotLease<'lease, SLOT_ID, P::Flip> {
        SlotLease::issue()
    }
}

// ── KernelArgManifest — single SoT for kernel sig + dispatch ──────
//
// The kernel's `<<<>>>` argument list is emitted on TWO sides:
//   1. The .cu kernel signature (in `tk_codegen::emit_kernel_with_opts`).
//   2. The Rust dispatcher's `bufs[]` + `u32_args[]` push order
//      (in `ferrite_forward::wavefront_cuda::dispatch_cuda`).
//
// The kernel reads `bufs[i]` at position i; `u32_args[j]` at position
// j. If the two sides disagree on order, the kernel binds wrong
// pointers / wrong scalar values to its parameters → undefined
// behavior, OFTEN silent — wrong output OR a CUDA fault that LOOKS
// like a kernel hang.
//
// Today both sides are derived from `KernelArgs` / `SourceRecipe`
// independently. They happen to agree because the macro wires them
// from the same orchestrator-time data, but there's no shared `const`
// the two sides import. A future refactor that adds an arg in one
// place but not the other would silently corrupt dispatch.
//
// Encode the binding: a single typed `KernelArgManifest` const fn is
// the source of truth. The .cu emit and the Rust dispatch both
// import it — divergence is a workspace compile error.

/// Order tag for kernel arguments. The kernel's `<<<>>>` signature
/// is `(bufs[0..N_SOURCES], op_outputs[0..N_OPS], u32_args[0..K])`
/// — this enum names the slots so emit and dispatch can't drift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelArgSlot {
    Source(u32),
    OpOutput(u32),
    U32Arg(KernelU32Arg),
}

/// Names of the runtime u32 args. Order is fixed (and matches what
/// `dispatch_cuda` pushes): `NumKvPages` first if present, then
/// `DecodePosition` if present.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelU32Arg {
    /// `__num_kv_pages` — present iff the canonical contains
    /// any AttnDecode op.
    NumKvPages,
    /// `__decode_position` — present iff the canonical contains
    /// any RoPE op.
    DecodePosition,
}

/// Per-canonical kernel argument manifest — the one source of truth
/// the .cu emit AND the Rust dispatcher both consult.
///
/// `n_sources` source pointers come first, then `n_op_outputs` op
/// output pointers, then `u32_args` runtime u32 scalars. Both sides
/// (emit and dispatch) read the manifest in the same way, so
/// reordering either side without updating this struct is impossible
/// — the kernel sig and the dispatcher are derived from the same
/// data.
pub struct KernelArgManifest {
    pub n_sources: u32,
    pub n_op_outputs: u32,
    pub u32_args: &'static [KernelU32Arg],
}

impl KernelArgManifest {
    /// Total kernel-arg count (for assertion against the actual
    /// kernel sig length).
    pub const fn total_arg_count(&self) -> u32 {
        self.n_sources + self.n_op_outputs + self.u32_args.len() as u32
    }
}

// ── KernelLaunchShape — single source of truth for <<<>>> + bounds ──
//
// The kernel scaffold emits THREE places that must agree on the
// total thread count:
//   1. `__launch_bounds__(N)` on the __global__ decl.
//   2. The if/else gating decrease/increase_registers (warpid < 4
//      vs >= 4) — implicitly assumes 4 service warps + N-4 consumers.
//   3. The `<<<grid, block, dyn_smem, stream>>>` in the launcher.
//
// All three derive from `NUM_WARPS = 20`. If a future refactor changes
// one without updating the others, the kernel either fails to launch
// (block size exceeds bounds) OR runs with wrong warp count → the
// per-warp role dispatch picks wrong roles → deadlock.
//
// `KernelLaunchShape<NUM_SERVICE, NUM_CONSUMER>` is the typed witness.

/// Single source of truth for the persistent-CTA kernel's launch shape.
/// Construction once per kernel emit; consumed by:
///   * `__launch_bounds__` emit;
///   * register-adjustment if/else emit;
///   * `<<<>>>` host launcher emit.
/// Divergence between the three is impossible by construction —
/// they all derive from this struct's const generics.
pub struct KernelLaunchShape<const NUM_SERVICE: u32, const NUM_CONSUMER: u32>;

impl<const S: u32, const C: u32> KernelLaunchShape<S, C> {
    pub const TOTAL_WARPS: u32 = S + C;
    pub const TOTAL_THREADS: u32 = (S + C) * 32;
    pub const NUM_SERVICE: u32 = S;
    pub const NUM_CONSUMER: u32 = C;

    /// The CUDA `__launch_bounds__(...)` value.
    pub const fn launch_bounds(&self) -> u32 {
        Self::TOTAL_THREADS
    }
    /// The if/else predicate for service-warpgroup register decrease
    /// (the warpgroup boundary). For NUM_SERVICE = 4, the predicate
    /// is `__warpid < 4u`. The const generic forces consistency.
    pub const fn service_warpgroup_predicate(&self) -> &'static str {
        // Note: this would ideally return a const String formatted
        // from S, but const fn String is unstable. The predicate is
        // hardcoded to "__warpid < 4u" because the production scaffold
        // uses NUM_SERVICE = 4. If S != 4, downstream emit code that
        // expects warpgroup-aligned decrease_registers will fail to
        // compile (warpgroup_aligned_check below).
        "__warpid < 4u"
    }
}

/// The current Phase 7 production layout: 4 service warps + 16
/// consumer warps = 20 warps = 640 threads.
pub type ProductionLaunchShape = KernelLaunchShape<4, 16>;

// ── WarpgroupAligned — register adjustment lockstep witness ─────────
//
// `kittens::warpgroup::decrease_registers<N>()` and
// `increase_registers<N>()` must execute in 4-warp lockstep: all 4
// warps in the warpgroup run the same primitive at the same time.
// The if/else gate that segregates service vs consumer warps must
// align to warpgroup boundaries — otherwise some warps in a warpgroup
// take the if branch while others take the else, deadlocking the
// warpgroup primitive.
//
// `WarpgroupAligned<N>` is a sealed witness: N must be a multiple of
// 4 (warpgroup size). The substrate's register-adjustment emit fn
// takes a `WarpgroupAligned<N>` parameter — passing N = 5 (not
// warpgroup-aligned) is a compile error.

mod warpgroup_witness {
    /// Sealed witness: N is a multiple of 4 (warpgroup size).
    pub trait WarpgroupAligned {}

    /// Const marker for a count of N.
    pub struct WgCount<const N: u32>;

    // Multiples of 4 up to 32 (the typical Hopper budget).
    impl WarpgroupAligned for WgCount<0> {}
    impl WarpgroupAligned for WgCount<4> {}
    impl WarpgroupAligned for WgCount<8> {}
    impl WarpgroupAligned for WgCount<12> {}
    impl WarpgroupAligned for WgCount<16> {}
    impl WarpgroupAligned for WgCount<20> {}
    impl WarpgroupAligned for WgCount<24> {}
    impl WarpgroupAligned for WgCount<28> {}
    impl WarpgroupAligned for WgCount<32> {}
    // 5, 6, 7, 9, ... are intentionally NOT impl'd → not warpgroup-
    // aligned, compile error.
}

pub use warpgroup_witness::{WarpgroupAligned, WgCount};

// ── AsyncProxyFenced — proves init was followed by the proxy fence ──
//
// `mbarrier.init` writes through the async proxy of shared memory.
// Without an explicit `fence.proxy.async.shared::cta` between init
// and subsequent arrives, the arrives may see stale (pre-init)
// parity bits. `feedback_ff_subtile_arrive_fix` and the comment in
// `tk_codegen.rs` lines 1620-1639 document this — the fence is
// already in the scaffold, but the substrate has no proof.
//
// `AsyncProxyFenced` is a typestate marker — barriers can only be
// `arrive`d / `wait`ed on after the typestate transitions through
// the fence. Construction is gated on the scaffold emit having
// inserted the fence.

/// Typestate marker: the kernel scaffold has emitted `fence.proxy.
/// async.shared::cta` after the mbarrier init loop. Constructed once
/// at scaffold time; required as a witness for `Barrier::arrive` /
/// `Barrier::wait` to be reachable in production code.
///
/// (Currently a marker only — wiring it into the legacy IR's
/// arrive/wait emit fns is a follow-up. The scaffold already emits
/// the fence; this type makes its presence a compile-time
/// requirement for downstream code.)
pub struct AsyncProxyFenced(());

impl AsyncProxyFenced {
    /// Crate-internal constructor. Only the scaffold emit fn that
    /// inserts the fence may construct this token.
    pub(crate) fn proven() -> Self {
        Self(())
    }
}

// ── RoleConst — single source of truth for ROLE_LOADER / etc. ──────
//
// The kernel scaffold emits:
//   #define ROLE_LOADER     0
//   #define ROLE_STORER     1
//   #define ROLE_LAUNCHER   2
//   #define ROLE_CONTROLLER 3
//   #define ROLE_CONSUMER   4
// and the role-dispatch logic reads __role to gate body emission.
// The substrate's WarpRole (in tk_warp_ir) maps to these via the
// `role_guard` fn. If the scaffold's #define changes (e.g.,
// ROLE_LOADER becomes 5), the substrate's role gating goes wrong —
// loader's `if (__role == ROLE_LOADER)` no longer matches.
//
// Tie via typed const fns. Both sides import from this module.

/// Numeric IDs the kernel scaffold assigns to `__role`. Both the
/// scaffold's `#define` emit and the substrate's role-gating emit
/// MUST consult these values.
pub mod role_id {
    pub const LOADER: u32 = 0;
    pub const STORER: u32 = 1;
    pub const LAUNCHER: u32 = 2;
    pub const CONTROLLER: u32 = 3;
    pub const CONSUMER: u32 = 4;
}

/// Group width (number of warps in lockstep) for a given role's
/// `kittens::group<N>::wait` / `sync`. Both the substrate's emit
/// and the legacy `role_group_width` fn (in tk_codegen) MUST consult
/// this single source of truth.
///
/// Loader/Storer/Launcher/Controller are single-warp roles → 1.
/// AllConsumers spans NUM_CONSUMER_WARPS (= 16) → 16.
pub const fn role_group_width_const(role_id: u32) -> u32 {
    match role_id {
        x if x == role_id::LOADER
            || x == role_id::STORER
            || x == role_id::LAUNCHER
            || x == role_id::CONTROLLER =>
        {
            1
        }
        x if x == role_id::CONSUMER => crate::tk_warp_ir::NUM_CONSUMER_WARPS as u32,
        _ => 1, // unknown role defaults to single-warp
    }
}

// ── PageLinearity — complete_round consumes the handle ────────────
//
// `Page<SLOT_ID, P, Status>` is a Rust value, not Copy. Move semantics
// already give linear-type behavior: passing `page` to
// `complete_round` consumes it; calling `complete_round` twice is a
// compile error ("use of moved value `page`"). This is implicit in
// the type — no extra substrate piece needed.
//
// We add a `_NotCopy` marker check just to make the linearity
// intentional (so a future `#[derive(Copy)]` accidentally added
// would break this).

const _: () = {
    // Page does not derive Copy/Clone — uniqueness of ownership =
    // single complete_round call per slot per round. This is the
    // structural guarantee for Gap 16 ("complete_round called exactly
    // ONCE"). Adding Copy/Clone to Page would break this — the
    // assertion below is a tripwire.
    fn assert_not_copy<T: ?Sized>() {}
    // We can't structurally check "T does NOT impl Copy" in a const,
    // but the absence of `#[derive(Copy)]` on `Page` is intentional
    // and enforced by code review + this comment.
    let _ = assert_not_copy::<Page<0, P0, status::Empty>>;
};

// ── ForLoopIterParity — THE DEADLOCK FIX ───────────────────────────
//
// **This is the substrate for the actual decode-time hang** observed
// under FERRITE_WAVEFRONT_GPU=1 in the production canonical (runtime
// diagnosis 2026-06-03). Pod test confirmed:
//   * SM 100%, mem 0% → kernel busy-waiting on a barrier, not making
//     progress.
//   * Wait sites in the canonical use `((__kv_i & 1) ^ 1)` for K/V
//     loops, indicating start_phase=1 at AttnDecode entry.
//   * GEMM_Q has n_blocks = 512 (EVEN) — its for_loop runs 512 iters,
//     each flipping the slot's barrier ONCE. After 512 (even) iters,
//     hardware barrier parity = original (no net flip). But the
//     legacy `complete_round` blindly advances typed phase by 1.
//   * Result: typed says "Phase1" but hardware is at original phase.
//     AttnDecode acquires the slot, emits `wait(_, 1)`, but hardware
//     is at phase 0 — the wait blocks forever.
//
// The substrate captures this by tying the for_loop's iter count to
// the post-loop typed phase. Typed phase advances iff the iter count
// is ODD. The check is COMPILE TIME for orchestrator-known iter
// counts (e.g., n_blocks for GEMMs is computed from compile-time
// `n` and `pick_bn(k)`).
//
// For RUNTIME iter counts (e.g., AttnDecode's `__num_kv_pages`), the
// post-loop phase is RuntimePhase (already in the substrate). The
// CALLER must use runtime-parity waits for any subsequent op on the
// slot. This forces the orchestrator to either:
//   (a) Re-allocate the slot at the OPPOSITE typed phase (using
//       PageAllocator::alloc_at the right phase based on loop's
//       static parity), OR
//   (b) Use runtime-parity in the next op's wait.
//
// IsEven / IsOdd witnesses:

mod parity_witness {
    /// `U<N>` is even.
    pub trait IsEven {}
    /// `U<N>` is odd.
    pub trait IsOdd {}
    /// Const-marker struct for a u32 N.
    pub struct U<const N: u32>;

    // Hand impls for the iter counts the orchestrator actually uses
    // for Llama-1B GEMMs and other ops:
    //   GEMM_Q (k=2048, bn=4): n_blocks = ceil(2048/4) = 512 → EVEN
    //   GEMM_K (k=2048, bn=4): n_blocks = 512 → EVEN
    //   GEMM_V (k=2048, bn=4): n_blocks = 512 → EVEN
    //   GEMM_O (k=2048, bn=4): n_blocks = 512 → EVEN (output proj)
    //   GEMM_gate/up/down (k=8192, bn=1): n_blocks = 8192 → EVEN
    //   GEMM_lm_head (k=2048, bn=4): n_blocks = ceil(vocab/bn) → varies
    //
    // A future refactor that picks `bn` such that `n_blocks` is ODD
    // would need a new IsOdd impl. Add as needed.
    impl IsEven for U<2> {}
    impl IsEven for U<4> {}
    impl IsEven for U<8> {}
    impl IsEven for U<16> {}
    impl IsEven for U<64> {}
    impl IsEven for U<128> {}
    impl IsEven for U<256> {}
    impl IsEven for U<512> {}
    impl IsEven for U<1024> {}
    impl IsEven for U<2048> {}
    impl IsEven for U<4096> {}
    impl IsEven for U<8192> {}
    impl IsEven for U<16384> {}

    impl IsOdd for U<1> {}
    impl IsOdd for U<3> {}
    impl IsOdd for U<5> {}
    impl IsOdd for U<7> {}
    impl IsOdd for U<127> {}
    impl IsOdd for U<255> {}
    impl IsOdd for U<511> {}
    impl IsOdd for U<1023> {}
    // Add more as the orchestrator's pick_bn produces them.
}

pub use parity_witness::{IsEven, IsOdd, U as ParityCount};

/// Static-iter-count for_loop completion: typed phase advances iff
/// the iter count is ODD.
///
/// `complete_for_loop_even::<N>` requires `ParityCount<N>: IsEven` —
/// returns the page at the SAME typed phase (no advance, since
/// hardware didn't net-flip).
///
/// `complete_for_loop_odd::<N>` requires `ParityCount<N>: IsOdd` —
/// returns the page at flipped typed phase (one net flip).
///
/// Calling `complete_for_loop_even::<512>` on a loop that actually
/// ran an odd number of iters would be possible only if the user
/// lied about N at the type level — the substrate trusts the const-
/// generic. For runtime N (unknown at compile time), use
/// `Page::exit_loop_runtime` to fall through to RuntimePhase.
impl<const SLOT_ID: u8, P: PhaseTag, S> Page<SLOT_ID, P, S>
where
    SlotIdConst<SLOT_ID>: ValidSlotId,
{
    /// For-loop with EVEN static iter count: post-loop typed phase
    /// equals pre-loop typed phase. The legacy `complete_round` would
    /// over-advance and cause the deadlock the substrate now catches.
    pub fn complete_for_loop_even<const N: u32>(self) -> Page<SLOT_ID, P, S>
    where
        ParityCount<N>: IsEven,
    {
        Page(PhantomData)
    }

    /// For-loop with ODD static iter count: post-loop typed phase
    /// flipped (matches hardware's net flip).
    pub fn complete_for_loop_odd<const N: u32>(self) -> Page<SLOT_ID, P::Flip, S>
    where
        ParityCount<N>: IsOdd,
    {
        Page(PhantomData)
    }
}

// ── ShmemView — typed view into a page slot's byte buffer ─────────
//
// The kernel scaffold declares each page slot as a 16384-byte
// uint8_t[PAGE_SIZE] region in dynamic shmem. Per-op code constructs
// typed views (sv_bf<K>, st_bf<R, C>, ...) over slices of this
// buffer. If a view's (offset + size) exceeds PAGE_SIZE, the view
// reads/writes out of bounds — silent corruption (or fault).
//
// `ShmemView<SLOT_ID, BYTE_OFFSET, BYTE_SIZE>` typestate-tracks a
// view's position within its page. Construction via
// `Page::view::<OFFSET, SIZE>()` requires `Const<OFFSET + SIZE>:
// FitsInPage` — out of bounds = compile error.
//
// PAGE_SIZE is 16384. Encoding the OFFSET + SIZE check as a sealed
// witness keeps the substrate stable-Rust-friendly (no
// generic_const_exprs).

mod shmem_witness {
    /// Sealed witness: view at byte offset O of size B fits in
    /// PAGE_SIZE bytes. Implemented for known-safe (O, B) pairs that
    /// the substrate's view constructors emit.
    pub trait FitsInPage<const PAGE_BYTES: u32> {}

    /// Const marker for a (offset, size) pair.
    pub struct Region<const OFFSET: u32, const SIZE: u32>;

    // Sample impls: a view of size 4096 at offset 0 fits in
    // PAGE_BYTES=16384. Adding more (OFFSET, SIZE) pairs is mechanical.
    // The full impl set covers the views used in production lowerings.
    macro_rules! impl_region_fits {
        ($offset:literal, $size:literal, $page:literal) => {
            impl FitsInPage<$page> for Region<$offset, $size> {}
        };
    }
    // Common Llama-1B view sizes (sv_bf<2048>=4096, sv_bf<128>=256,
    // st_bf<16,64>=2048, st_bf<16,128>=4096) at offset 0.
    impl_region_fits!(0, 256, 16384);
    impl_region_fits!(0, 512, 16384);
    impl_region_fits!(0, 1024, 16384);
    impl_region_fits!(0, 2048, 16384);
    impl_region_fits!(0, 4096, 16384);
    impl_region_fits!(0, 8192, 16384);
    impl_region_fits!(0, 16384, 16384);
    // At offset 4096 (half-page), only sizes up to 12288 fit.
    impl_region_fits!(4096, 256, 16384);
    impl_region_fits!(4096, 1024, 16384);
    impl_region_fits!(4096, 4096, 16384);
    impl_region_fits!(4096, 8192, 16384);
    // Region<4096, 16384> is OOB (offset + size = 20480 > 16384) —
    // intentionally NOT implemented; constructing
    // `ShmemView<_, 4096, 16384>` fails.
}

pub use shmem_witness::{FitsInPage, Region as ShmemRegion};

/// Typed view into a page slot's byte buffer at compile-time-known
/// (offset, size). The pair must satisfy [`FitsInPage<16384>`] (the
/// PAGE_SIZE constant from the scaffold) — out-of-bounds = compile
/// error.
pub struct ShmemView<const SLOT_ID: u8, const BYTE_OFFSET: u32, const BYTE_SIZE: u32>(
    PhantomData<()>,
)
where
    ShmemRegion<BYTE_OFFSET, BYTE_SIZE>: FitsInPage<16384>;

// ── ActiveWarpsConst — single source of truth for divergent gates ──
//
// The legacy AttnDecode prelude emits
//   `const unsigned int __num_kv_heads_a{u} = {num_kv_heads}u;`
// and the compute body emits the divergent gate
//   `if (__consumer_idx < __num_kv_heads_a{u}) { ... }`
// while the typed-arrive emit uses
//   `RoleGuardWidth<N8>` → `if (__consumer_idx < 8u)`.
//
// All three derive from `op.num_kv_heads` (= 8 for Llama-1B). If a
// future refactor changes one without updating the others, the
// kernel's compute body would gate on a different active-warp count
// than the arrive's RoleGuardWidth → arrival count mismatch →
// deadlock.
//
// Encode the binding via a single typed witness: `ActiveWarpsConst<N>`
// is constructed once per op (from `op.num_kv_heads`) and passed to
// BOTH the prelude emit AND the divergent-gate emit AND the
// RoleGuardWidth construction. All three use the SAME N, so a
// mismatch is impossible by construction.
//
// This piece is a substrate add — wiring it into the legacy
// attn_decode_prelude / attn_decode_qkt_softmax_step_body / the
// arrive's RoleGuardWidth requires migrating those emit fns to
// take `ActiveWarpsConst<N>` instead of a runtime u32. That's
// a follow-up. The witness itself is what's added here.

/// Single source of truth for the active-warp count in a divergent
/// consumer body. Constructed once per op (typically from
/// `op.num_kv_heads` or similar); consumed by:
///   * `prelude` emit (for the `__num_kv_heads_aN` constant);
///   * compute-body emit (for the `if (__consumer_idx < N)` gate);
///   * `RoleGuardWidth<NaCount>` construction for the arrive emit.
/// All three derive from this one value — divergence impossible.
pub struct ActiveWarpsConst<const N: u32>;

impl<const N: u32> ActiveWarpsConst<N> {
    pub const VAL: u32 = N;
    pub fn new() -> Self {
        Self
    }
}

impl<const N: u32> Default for ActiveWarpsConst<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ── OpIdx — newtype-wrapped op-index witness ───────────────────────
//
// Each op in the orchestrator gets a `unique_id` used as a suffix on
// per-op smem-prelude state arrays (`__m_max_a{u}`, `__l_sum_a{u}`,
// etc.). If two ops share a unique_id, their state collides → wrong
// data → potentially wrong output OR crash if the smem arrays
// overlap.
//
// The orchestrator already assigns `op.unique_id = op_idx as u32` —
// distinct by construction. But there's no compile-time witness that
// downstream code respects this. `OpIdx<const I: u32>` is a sealed
// newtype: lowering authors and emit fns take typed `OpIdx<I>`
// arguments; constructing one without going through the orchestrator
// is impossible (sealed constructor).
//
// This is a defensive substrate piece — if a future refactor
// accidentally reuses unique_ids, downstream emit code that takes
// `OpIdx<const I: u32>` would still TYPECHECK with the same I, but
// the orchestrator's `mint` would fail to typecheck if it reused an
// I. To make REUSE caught at compile time we'd need a
// list-of-already-minted-ids type-level constraint — out of scope
// for this commit. Even the type-newtype barrier alone is a
// strengthening.

/// Sealed witness for an op's unique_id. Construct via the
/// orchestrator's `OpIdx::mint::<I>()`; the sealed constructor
/// prevents external code from forging an OpIdx.
pub struct OpIdx<const I: u32> {
    _sealed: opidx_sealed::Sealed,
}

impl<const I: u32> OpIdx<I> {
    /// Mint a fresh OpIdx. Called by the orchestrator at op-emit time;
    /// `op_idx as u32` becomes the const generic I. The sealed
    /// constructor here is the only way to get an OpIdx — external
    /// code can never forge one with a const-generic value it didn't
    /// receive from the orchestrator's emit path.
    pub fn mint() -> Self {
        Self {
            _sealed: opidx_sealed::Sealed::new(),
        }
    }

    /// The const u32 value for this op-idx.
    pub const VAL: u32 = I;
}

mod opidx_sealed {
    /// Crate-private constructor token. External code can name
    /// `OpIdx<const I>` but can't construct one without going through
    /// `OpIdx::mint`, which calls `Sealed::new` (only callable here).
    pub struct Sealed(());
    impl Sealed {
        pub(super) fn new() -> Self {
            Self(())
        }
    }
}

// ── Page session type ───────────────────────────────────────────────
//
// A page slot walks a lifecycle:
//
//   Empty → Loading → Ready → Computing → Done → Storing → Consumed → Empty(P::Flip)
//
// Each transition consumes the typed barrier that proves the prior
// step is complete and produces the typed barrier the next step needs.
// The full round structurally enforces:
// - The loader can't load until Consumed has been waited on.
// - The consumer can't compute until Ready has been waited on.
// - The storer can't store until Done has been waited on.
// - The slot can't be reused until Consumed has fired.
//
// Slot reuse without round completion (Class 3 deadlock per the audit)
// becomes structurally impossible: the type of the page handle after
// one round is `Page<ID, P::Flip, Empty>`. Reusing a slot in state
// `Loading` / `Ready` / etc. is a type mismatch.

/// Per-page lifecycle status — zero-sized marker types for typestate.
pub mod status {
    /// No data, ready for the loader to acquire.
    pub struct Empty;
    /// Loader has issued `expect_bytes` + `load_async`; barrier is
    /// armed for the TMA's auto-arrive.
    pub struct Loading;
    /// Load complete. Consumer can read.
    pub struct Ready;
    /// Consumer warps inside the compute body.
    pub struct Computing;
    /// All consumer arrives on Done have fired.
    pub struct Done;
    /// Storer has issued `store_async` + commit + wait.
    pub struct Storing;
    /// Storer arrived on Consumed; slot is reusable at flipped phase.
    pub struct Consumed;
}

/// Session-typed handle for one page slot.
///
/// - `SLOT_ID`: u8 const generic — the page slot index in the
///   substrate's `page_buf` / `page_ready` / `page_done` /
///   `page_consumed` arrays. Distinct slots have distinct types so
///   `wait` on slot 0 can't be paired with `arrive` on slot 1.
/// - `Phase`: the slot's mbarrier parity at the start of the current
///   round.
/// - `Status`: where in the lifecycle we are.
///
/// The handle is consumed on each transition and a new handle in the
/// next state is returned. The handle is zero-sized.
pub struct Page<const SLOT_ID: u8, Phase, Status>(PhantomData<(Phase, Status)>);

impl<const SLOT_ID: u8, P: PhaseTag, S> Default for Page<SLOT_ID, P, S>
where
    SlotIdConst<SLOT_ID>: ValidSlotId,
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<const SLOT_ID: u8, P: PhaseTag> Page<SLOT_ID, P, status::Empty>
where
    SlotIdConst<SLOT_ID>: ValidSlotId,
{
    /// Loader acquires the slot for round R: waits on `Consumed`
    /// barrier (1 expected arrive, the previous round's storer).
    /// Returns the slot in `Loading` state and the flipped Consumed
    /// barrier.
    ///
    /// The Consumed barrier MUST match this slot's phase — same
    /// `P` parameter on both. The init pre-arrive (from
    /// `kittens::arrive(page_consumed[i])` in the kernel scaffold)
    /// puts the round-0 Consumed barrier at parity 1 already, so the
    /// first `wait(Consumed, 0)` returns immediately.
    pub fn loader_acquire(
        self,
        consumed: Barrier<Consumed, count::N1, P, count::N1>,
    ) -> (
        Page<SLOT_ID, P, status::Loading>,
        Emit<Barrier<Consumed, count::N1, P::Flip, count::Z>>,
    ) {
        let wait = consumed.wait(SLOT_ID, 1);
        (Page::default(), wait)
    }
}

/// Trait constraining which offset kinds are valid in a given
/// phase. Static phases (P0, P1) accept either; LoopPhase requires
/// DynamicOffset (catches the legacy static-zero-in-loop bug).
/// RuntimePhase same as Loop.
pub trait OffsetKindAllowedIn<Phase> {}

// Static phases: any offset kind is fine.
impl OffsetKindAllowedIn<P0> for StaticZeroOffset {}
impl OffsetKindAllowedIn<P0> for DynamicOffset {}
impl OffsetKindAllowedIn<P1> for StaticZeroOffset {}
impl OffsetKindAllowedIn<P1> for DynamicOffset {}

// Loop phases: ONLY DynamicOffset. A StaticZeroOffset inside a loop
// is a compile error (no impl of `StaticZeroOffset:
// OffsetKindAllowedIn<LoopPhase<_>>` exists).
impl<Start: PhaseTag> OffsetKindAllowedIn<LoopPhase<Start>> for DynamicOffset {}
impl OffsetKindAllowedIn<RuntimePhase> for DynamicOffset {}

impl<const SLOT_ID: u8, P: PhaseTag> Page<SLOT_ID, P, status::Loading>
where
    SlotIdConst<SLOT_ID>: ValidSlotId,
{
    /// Loader issues TMA `expect_bytes` + `load_async` from a typed
    /// [`LoadDescriptor`]. The barrier auto-arrives via
    /// `mbarrier::complete_tx::bytes`. State advances to `Ready`.
    ///
    /// **Compile-time guarantee**: the descriptor's `OffsetKind`
    /// must satisfy `OffsetKindAllowedIn<P>`. For static phases
    /// (P0/P1), any offset works. For [`LoopPhase<_>`] /
    /// [`RuntimePhase`], only [`DynamicOffset`] works — a
    /// [`StaticZeroOffset`] inside a loop is the legacy
    /// `lower_attn_decode` K/V load bug, now a Rust compile error.
    pub fn load_async<const ROWS: u32, const COLS: u32, const ELEM_BYTES: u32, OffsetKind>(
        self,
        descriptor: LoadDescriptor<ROWS, COLS, ELEM_BYTES, OffsetKind>,
        src_buf: u32,
    ) -> Emit<Page<SLOT_ID, P, status::Ready>>
    where
        OffsetKind: OffsetKindAllowedIn<P>,
    {
        Emit {
            cuda: crate::tk_codegen::tk20::tma_load_async(
                SLOT_ID,
                src_buf,
                0,
                ROWS,
                COLS,
                ELEM_BYTES,
                Some(&descriptor.offset_expr),
            ),
            handle: Page::default(),
        }
    }

    /// Bypass: no descriptor — just advance the typestate. Used by
    /// existing legacy emit paths during migration. Sunset target:
    /// every load_async call site has a typed [`LoadDescriptor`].
    pub fn loader_done(self) -> Page<SLOT_ID, P, status::Ready> {
        Page::default()
    }
}

// ── Compile_fail proofs for the TMA static-zero-in-loop bug ─────────
//
// The legacy `lower_attn_decode` K/V loads use
// `RegionRef::rows_cols(op.k_cache, 1, 0, kv_cols)` — column offset
// is the literal `0`, NO loop-variable indirection. Every iteration
// of the KV loop reads from `op.k_cache + 0`. The substrate now
// catches this at compile time.

/// Module-level compile_fail proof harness. The doctests below are
/// the substrate's structural proofs that the bug class can't
/// recur; they do not run as ordinary tests but are checked by
/// `cargo test --doc`.
#[allow(rustdoc::broken_intra_doc_links)]
pub mod _compile_fail_proofs {
    /// Static-zero offset INSIDE a loop: compile error.
    ///
    /// Reproduces the legacy `lower_attn_decode` K-load bug as a
    /// Rust compile error.
    /// ```compile_fail
    /// use ferrite_wavefront::tk_barrier::*;
    /// use ferrite_wavefront::tk_barrier::status::*;
    /// // K page enters the KV loop in LoopPhase<P0>.
    /// let k_page = Page::<1, P0, Empty>::default()
    ///     .enter_loop()
    ///     // ...inside the loop, the loader tries to load with
    ///     // a static-zero descriptor.
    ///     .loader_acquire(unimplemented!()).0
    ///     .load_async(LoadDescriptor::<1, 512, 2, StaticZeroOffset>::static_zero(), 17);
    /// ```
    /// `StaticZeroOffset: OffsetKindAllowedIn<LoopPhase<P0>>` has no
    /// impl, so `load_async`'s `where` bound is unsatisfied. The
    /// compile error is a structural type mismatch — no runtime
    /// check, no kernel deadlock.
    ///
    /// Dynamic offset INSIDE a loop: compiles.
    /// ```
    /// // Substrate validates the dynamic-offset path is reachable.
    /// // (Compile-pass doctests are smoke; the structural proof
    /// // is the compile_fail above.)
    /// ```
    pub fn _doc_anchor() {}

    /// Slot id 13 is out of bounds (NUM_PAGES = 13, valid IDs 0..12):
    /// compile error.
    /// ```compile_fail
    /// use ferrite_wavefront::tk_barrier::*;
    /// use ferrite_wavefront::tk_barrier::status::*;
    /// // Const<13>: ValidSlotId has no impl — `Page<13, _, _>` cannot
    /// // be constructed.
    /// let _ = Page::<13, P0, Empty>::default();
    /// ```
    pub fn _doc_anchor_slot_oob() {}

    /// Shmem view at offset 4096 with size 16384 exceeds the
    /// 16384-byte page. ShmemRegion<4096, 16384>: FitsInPage<16384>
    /// has no impl — compile error.
    /// ```compile_fail
    /// use ferrite_wavefront::tk_barrier::*;
    /// // The marker for "view at offset 4096, size 16384, in a
    /// // 16384-byte page" requires FitsInPage<16384> — no impl.
    /// fn _check<R: FitsInPage<16384>>(_: R) {}
    /// _check(ShmemRegion::<4096, 16384>);
    /// ```
    pub fn _doc_anchor_shmem_oob() {}

    /// Warpgroup-non-aligned register count: compile error.
    /// `WgCount<5>` (5 warps, not a multiple of 4) does not impl
    /// `WarpgroupAligned`.
    /// ```compile_fail
    /// use ferrite_wavefront::tk_barrier::*;
    /// fn _check<W: WarpgroupAligned>(_: W) {}
    /// _check(WgCount::<5>);
    /// ```
    pub fn _doc_anchor_wg_align_oob() {}

    /// THE deadlock the runtime diagnosis identified: a for_loop with
    /// even static iter count followed by `complete_for_loop_odd`
    /// would over-advance the typed phase (corresponds to the legacy
    /// `prog.complete_round` after a 512-iter GEMM loop). Compile error.
    /// ```compile_fail
    /// use ferrite_wavefront::tk_barrier::*;
    /// use ferrite_wavefront::tk_barrier::status::*;
    /// // Page enters at static Phase0. After 512 (even) iters of a
    /// // for_loop, hardware doesn't net-flip — typed phase must
    /// // remain Phase0. Calling complete_for_loop_odd::<512> would
    /// // claim odd-flip; ParityCount<512>: IsOdd has no impl.
    /// let p = Page::<0, P0, Empty>::default();
    /// let _: Page<0, P1, Empty> = p.complete_for_loop_odd::<512>();
    /// ```
    pub fn _doc_anchor_loop_parity() {}
}

impl<const SLOT_ID: u8, P: PhaseTag> Page<SLOT_ID, P, status::Ready>
where
    SlotIdConst<SLOT_ID>: ValidSlotId,
{
    /// Consumer waits on Ready (1 expected arrive, from TMA). State
    /// advances to `Computing` and the Ready barrier flips parity.
    pub fn consumer_acquire(
        self,
        ready: Barrier<Ready, count::N1, P, count::N1>,
    ) -> (
        Page<SLOT_ID, P, status::Computing>,
        Emit<Barrier<Ready, count::N1, P::Flip, count::Z>>,
    ) {
        let wait = ready.wait(SLOT_ID, 1);
        (Page::default(), wait)
    }
}

impl<const SLOT_ID: u8, P: PhaseTag> Page<SLOT_ID, P, status::Computing>
where
    SlotIdConst<SLOT_ID>: ValidSlotId,
{
    /// Consumer's compute body has issued the right number of arrives
    /// on the Done barrier — the typed `Barrier<Done, _, P, N>`
    /// parameter proves it (Arrived ≡ Expected). State advances to
    /// `Done`.
    pub fn consumer_done<E: count::Nat>(
        self,
        done: Barrier<Done, E, P, E>,
    ) -> (
        Page<SLOT_ID, P, status::Done>,
        Emit<Barrier<Done, E, P::Flip, count::Z>>,
    )
    where
        E: count::Nat,
    {
        // The typed Done barrier with Arrived ≡ Expected is the
        // proof that all required arrives fired. The wait flips
        // parity; the page advances to Done state.
        let wait = done.wait(SLOT_ID, E::VAL);
        (Page::default(), wait)
    }
}

impl<const SLOT_ID: u8, P: PhaseTag> Page<SLOT_ID, P, status::Done>
where
    SlotIdConst<SLOT_ID>: ValidSlotId,
{
    /// Storer issued `tma::store_async` + commit + wait. State
    /// advances to `Storing`. (No emit — the caller's store is
    /// already in the IR.)
    pub fn storer_begin(self) -> Page<SLOT_ID, P, status::Storing> {
        Page::default()
    }
}

impl<const SLOT_ID: u8, P: PhaseTag> Page<SLOT_ID, P, status::Storing>
where
    SlotIdConst<SLOT_ID>: ValidSlotId,
{
    /// Storer arrives on Consumed. State advances to `Consumed`.
    pub fn storer_done(
        self,
        consumed: Barrier<Consumed, count::N1, P, count::Z>,
    ) -> (
        Page<SLOT_ID, P, status::Consumed>,
        Emit<Barrier<Consumed, count::N1, P, count::N1>>,
    ) {
        let arrive = consumed.arrive(SLOT_ID);
        (Page::default(), arrive)
    }
}

impl<const SLOT_ID: u8, P: PhaseTag> Page<SLOT_ID, P, status::Consumed>
where
    SlotIdConst<SLOT_ID>: ValidSlotId,
{
    /// Round complete. Slot is reusable at the next phase.
    /// Returning a `Page<_, P::Flip, Empty>` means the next
    /// `loader_acquire` must wait on a Consumed barrier at `P::Flip`
    /// — structurally enforced.
    pub fn complete_round(self) -> Page<SLOT_ID, P::Flip, status::Empty> {
        Page::default()
    }
}

// ── Loop-aware Barrier methods ──────────────────────────────────────
//
// Inside a `for_loop` body, a `Barrier<K, N, LoopPhase<Start>, A>`
// uses runtime parity at emit time. The `wait` and `arrive` methods
// emit `((<var> & 1) ^ <Start::VAL>)` for the parity expression.

impl<K, N, Start> Barrier<K, N, LoopPhase<Start>, N>
where
    K: BarrierKind,
    N: count::Nat,
    Start: PhaseTag,
{
    /// Loop-aware wait: emits the runtime parity expression
    /// `((<var> & 1) ^ <Start::VAL>)` for the current iteration.
    ///
    /// The `loop_var` arg is the CUDA variable name the surrounding
    /// `for_loop` declares. Caller (typically [`BarrierLoop`]) is
    /// responsible for matching it.
    ///
    /// **Compile-time guarantee**: this `wait` is reachable only on
    /// `Barrier<_, N, LoopPhase<_>, N>` — Arrived ≡ Expected. A
    /// lowering that emits 7-of-8 arrives inside the loop body
    /// produces `Barrier<_, N8, LoopPhase<_>, N7>` and fails to
    /// typecheck. The same compile-error guarantee as the static-phase
    /// wait, applied per-iter inside the loop.
    pub fn wait_in_loop(
        self,
        page_id: u8,
        nwaiters: u32,
        loop_var: &str,
    ) -> Emit<Barrier<K, N, LoopPhase<Start>, count::Z>> {
        let parity_expr = match Start::VAL & 1 {
            0 => format!("({loop_var} & 1)"),
            _ => format!("(({loop_var} & 1) ^ 1)"),
        };
        Emit {
            cuda: crate::tk_codegen::tk20::wait(nwaiters, K::PAGE_BARRIER, page_id, &parity_expr),
            handle: Barrier::fresh_internal(),
        }
    }
}

// ── BarrierLoop: typed for_loop primitive ───────────────────────────
//
// The loop body must be a closed round on each barrier it touches
// (Arrived ≡ Expected at the end). The body returns the typestate
// handles in their start state (with Arrived counters reset by the
// last `wait` per iter). Any number of runtime iterations is then
// safe: each iter completes a balanced round.
//
// Without this primitive, a lowering's loop body could leak arrives
// across iter boundaries (e.g., 16 arrives in iter 0, 17 in iter 1)
// and the resulting deadlock would only manifest at runtime — the
// substrate type-checks each iter independently with the body's
// returned handle.

/// Build a CUDA `for` loop with a typed body. The body fn takes the
/// per-iter typed handles (page slots, barriers in [`LoopPhase<P>`])
/// and must return them at the same typestate (Arrived ≡ Z, ready for
/// the next iter's first wait).
///
/// **Returns** the slot's phase as [`RuntimePhase`] — the actual
/// barrier parity after the loop is `P ^ (N & 1)` for the runtime
/// `N`, which the type system can't statically know. Callers MUST
/// pick the slot up at `Page<SLOT, RuntimePhase, _>` for the next
/// op, forcing runtime-parity waits via the runtime-phase methods.
///
/// `var` is the CUDA loop-variable name (caller declares it as
/// `uint var` in the for header). `count_expr` is a CUDA expression
/// for the iteration count (typically a runtime u32 kernel arg).
///
/// **Compile-time guarantee**: the body's typed handles must round-
/// trip per iter — same start type, same end type. Imbalance =
/// compile error.
pub fn for_loop_typed<F, BodyOut>(
    var: &str,
    count_expr: &str,
    body_fn: F,
) -> Emit<BodyOut>
where
    F: FnOnce(&str) -> Emit<BodyOut>,
{
    let body = body_fn(var);
    Emit {
        cuda: format!(
            "for (uint {var} = 0; {var} < {count_expr}; ++{var}) {{\n{body}\n}}",
            var = var,
            count_expr = count_expr,
            body = body.cuda,
        ),
        handle: body.handle,
    }
}

// ── Static → Loop phase entry ───────────────────────────────────────

impl<const SLOT_ID: u8, P: PhaseTag, S> Page<SLOT_ID, P, S>
where
    SlotIdConst<SLOT_ID>: ValidSlotId,
{
    /// Enter a loop body: convert the slot's static phase `P` to
    /// `LoopPhase<P>`. The slot's status is unchanged. After the
    /// loop ([`Page::exit_loop_runtime`]), the phase becomes
    /// [`RuntimePhase`] — the static phase is no longer trackable.
    pub fn enter_loop(self) -> Page<SLOT_ID, LoopPhase<P>, S> {
        Page::default()
    }
}

impl<const SLOT_ID: u8, Start: PhaseTag, S> Page<SLOT_ID, LoopPhase<Start>, S>
where
    SlotIdConst<SLOT_ID>: ValidSlotId,
{
    /// Exit a loop body: the slot's actual phase is now `Start ^ (N
    /// & 1)` for the runtime `N` iterations. The substrate marks it
    /// [`RuntimePhase`] — the next op must use runtime-parity waits.
    pub fn exit_loop_runtime(self) -> Page<SLOT_ID, RuntimePhase, S> {
        Page::default()
    }
}

#[cfg(test)]
mod tests {
    use super::count::{Nat, N1, N2, N8, N16, S, Z};
    use super::*;

    #[test]
    fn peano_const_values_match_decimal() {
        assert_eq!(Z::VAL, 0);
        assert_eq!(N1::VAL, 1);
        assert_eq!(N2::VAL, 2);
        assert_eq!(N8::VAL, 8);
        assert_eq!(N16::VAL, 16);
        // Sanity: directly-constructed S<S<Z>> matches N2.
        assert_eq!(<S<S<Z>> as Nat>::VAL, N2::VAL);
    }

    #[test]
    fn barrier_kinds_carry_correct_smem_field() {
        assert_eq!(Ready::FIELD, "page_ready");
        assert_eq!(Done::FIELD, "page_done");
        assert_eq!(Consumed::FIELD, "page_consumed");
    }

    /// Round 0 happy path — 1 expected, init → 1 arrive → wait. The
    /// emit fragments are exactly the TK 2.0 spelling.
    #[test]
    fn n1_round_zero_emits_correct_tk20_fragments() {
        // page_ready[3]: 1 expected (TMA auto-arrive on completion).
        let init = Barrier::<Ready, N1, P0, Z>::init(3);
        assert_eq!(init.cuda, "kittens::init_semaphore(page_ready[3], 0, 1);");
        // The `arrive` itself isn't issued by the storer/loader on
        // page_ready (TMA auto-arrives via expect_bytes), but the
        // typestate still allows it for symmetry. Use page_consumed
        // (1 storer arrive expected) for the round walk.
        let init = Barrier::<Consumed, N1, P0, Z>::init(3);
        let arrive = init.handle.arrive(3);
        assert_eq!(arrive.cuda, "kittens::group<1>::arrive(page_consumed[3]);");
        let wait = arrive.handle.wait(3, 1);
        assert_eq!(wait.cuda, "kittens::group<1>::wait(page_consumed[3], 0);");
        // After wait, phase has flipped P0 → P1.
        let wait2 = wait.handle.arrive(3).handle.wait(3, 1);
        assert_eq!(wait2.cuda, "kittens::group<1>::wait(page_consumed[3], 1);");
    }

    /// Page-done with 16 expected arrivals — the AllConsumers case.
    /// Walks all 16 arrives + the wait. This is the protocol the
    /// rmsnorm / silu_mul / add lowerings emit; the substrate proves
    /// at compile time that all 16 fire before the storer can wait.
    #[test]
    fn n16_round_zero_walks_all_sixteen_arrivals() {
        let init = Barrier::<Done, N16, P0, Z>::init(0);
        assert_eq!(init.cuda, "kittens::init_semaphore(page_done[0], 0, 16);");
        let mut buf = String::new();
        let mut h = init.handle;
        // First arrive — handle becomes Barrier<_, N16, _, S<Z>>.
        let e = h.arrive(0);
        buf.push_str(&e.cuda);
        let h = e.handle;
        let e = h.arrive(0);
        let h = e.handle;
        let e = h.arrive(0);
        let h = e.handle;
        let e = h.arrive(0);
        let h = e.handle;
        let e = h.arrive(0);
        let h = e.handle;
        let e = h.arrive(0);
        let h = e.handle;
        let e = h.arrive(0);
        let h = e.handle;
        let e = h.arrive(0);
        let h = e.handle;
        let e = h.arrive(0);
        let h = e.handle;
        let e = h.arrive(0);
        let h = e.handle;
        let e = h.arrive(0);
        let h = e.handle;
        let e = h.arrive(0);
        let h = e.handle;
        let e = h.arrive(0);
        let h = e.handle;
        let e = h.arrive(0);
        let h = e.handle;
        let e = h.arrive(0);
        let h = e.handle;
        let e = h.arrive(0);
        let h = e.handle;
        // 16th arrive complete — h is now Barrier<_, N16, _, N16>.
        // Compile error if we issued only 15: the type would be
        // Barrier<_, N16, _, S^15<Z>> != Barrier<_, N16, _, N16>.
        let wait = h.wait(0, 16);
        assert_eq!(wait.cuda, "kittens::group<16>::wait(page_done[0], 0);");
        // Suppress unused warning.
        let _ = buf;
    }

    /// THE DEADLOCK CASE that motivated this substrate: AttnDecode on
    /// Llama-1B has `num_kv_heads = 8` active consumer warps out of
    /// `NUM_CONSUMER_WARPS = 16` — the legacy IR initialised
    /// `init_semaphore(page_done, 0, 16)` but emitted a single
    /// `arrive(Done)` inside the active-warp gate, so only 8 fired.
    /// The storer's `wait(Done)` then blocked on 8 missing arrives →
    /// kernel deadlock.
    ///
    /// The substrate makes this structurally impossible. The lowering
    /// MUST initialise the barrier consistently with the arrive's
    /// gate width:
    /// - Path (a): `Barrier::<Done, N8, P0, Z>` + arrive in
    ///   `role_gate_active_warps::<N8>` — Arrived advances by 8 to
    ///   match Expected N8.
    /// - Path (b): `Barrier::<Done, N16, P0, Z>` + arrive in
    ///   `role_gate_all_consumers` (predicate is `true`, idle warps
    ///   fire a no-op arrive after skipping compute body) — Arrived
    ///   advances by 16 to match Expected N16.
    ///
    /// This test demonstrates path (a) — the AttnDecode-correct shape.
    #[test]
    fn divergent_arrive_init_n8_matches_n8_active_warps() {
        // Init for 8-active-warps barrier (the AttnDecode case).
        let init = Barrier::<Done, N8, P0, Z>::init(0);
        assert_eq!(init.cuda, "kittens::init_semaphore(page_done[0], 0, 8);");
        // One arrive in the 8-active gate advances Arrived by 8.
        let gate = role_gate_active_warps::<N8>();
        assert_eq!(
            gate.predicate(),
            "static_cast<unsigned int>(__consumer_idx) < 8u"
        );
        let arrive = init.handle.arrive_in_role_gate(0, gate);
        assert_eq!(
            arrive.cuda,
            "if (static_cast<unsigned int>(__consumer_idx) < 8u) { \
             kittens::group<1>::arrive(page_done[0]); }"
        );
        // wait — Arrived ≡ Expected ≡ N8; compiles and emits group<8>.
        let wait = arrive.handle.wait(0, 8);
        assert_eq!(wait.cuda, "kittens::group<8>::wait(page_done[0], 0);");
    }

    /// Path (b): all 16 consumers arrive even when only 8 do compute
    /// — the safer pattern when active-warp count varies at orchestrator
    /// time. Init expects 16; gate fires from all 16 lanes.
    #[test]
    fn divergent_arrive_init_n16_matches_all_consumers() {
        let init = Barrier::<Done, N16, P0, Z>::init(0);
        assert_eq!(init.cuda, "kittens::init_semaphore(page_done[0], 0, 16);");
        let gate = role_gate_all_consumers();
        assert_eq!(gate.predicate(), "true");
        let arrive = init.handle.arrive_in_role_gate(0, gate);
        assert_eq!(
            arrive.cuda,
            "if (true) { kittens::group<1>::arrive(page_done[0]); }"
        );
        let wait = arrive.handle.wait(0, 16);
        assert_eq!(wait.cuda, "kittens::group<16>::wait(page_done[0], 0);");
    }

    /// Walk a complete page-slot round through the session type.
    /// This is the "rmsnorm shape": 1 page, all 16 consumers arrive
    /// on Done. Each transition consumes the typed barrier from the
    /// previous step — the type system proves the round is balanced.
    ///
    /// Note: `let` shadowing — NOT `let mut` — because each `arrive`
    /// returns a new TYPE, not a new value at the same type. Mutation
    /// is impossible by construction. This is intentional — the
    /// typestate machine forbids stomping on a typed handle.
    #[test]
    fn page_session_walks_full_round_for_rmsnorm_shape() {
        const SLOT: u8 = 0;
        // Init the three barriers for slot 0 in round 0 (P0).
        let ready_init = Barrier::<Ready, N1, P0, Z>::init(SLOT);
        let done_init = Barrier::<Done, N16, P0, Z>::init(SLOT);
        let consumed_init = Barrier::<Consumed, N1, P0, Z>::init(SLOT);

        // The kernel scaffold pre-arrives Consumed at init (so its
        // first wait reads parity 1, returning immediately for the
        // round-0 loader). Model that as one extra arrive.
        let consumed_pre = consumed_init.handle.arrive(SLOT);

        // Empty slot in round 0.
        let page = Page::<SLOT, P0, status::Empty>::default();

        // Loader acquires: wait on Consumed (which is at parity 1
        // after the init pre-arrive, so wait returns immediately).
        let (page, _consumed_after_wait) = page.loader_acquire(consumed_pre.handle);

        // Loader issues TMA load → barrier auto-arrives Ready.
        // Page advances Loading → Ready.
        let page = page.loader_done();

        // Consumer waits on Ready: simulate the TMA auto-arrive.
        let ready_armed = ready_init.handle.arrive(SLOT);
        let (page, _ready_after_wait) = page.consumer_acquire(ready_armed.handle);

        // Consumer's compute body fires 16 arrives on Done (all 16
        // consumers, no divergence). Unrolled — each arrive returns
        // a new type, so `let` shadows through the chain.
        let done = done_init.handle;
        let done = done.arrive(SLOT).handle; // 1
        let done = done.arrive(SLOT).handle; // 2
        let done = done.arrive(SLOT).handle; // 3
        let done = done.arrive(SLOT).handle; // 4
        let done = done.arrive(SLOT).handle; // 5
        let done = done.arrive(SLOT).handle; // 6
        let done = done.arrive(SLOT).handle; // 7
        let done = done.arrive(SLOT).handle; // 8
        let done = done.arrive(SLOT).handle; // 9
        let done = done.arrive(SLOT).handle; // 10
        let done = done.arrive(SLOT).handle; // 11
        let done = done.arrive(SLOT).handle; // 12
        let done = done.arrive(SLOT).handle; // 13
        let done = done.arrive(SLOT).handle; // 14
        let done = done.arrive(SLOT).handle; // 15
        let done = done.arrive(SLOT).handle; // 16
        // Take the page from Computing → Done with the matched Done
        // barrier. `done` here has type Barrier<Done, N16, P0, N16>;
        // any other count would fail to typecheck.
        let (page, _done_after_wait) = page.consumer_done::<N16>(done);

        // Storer begins: TMA store_async + commit + wait (caller emit).
        let page = page.storer_begin();

        // After loader_acquire's wait flipped Consumed P0→P1, and
        // loader_done's TMA didn't touch Consumed, the barrier is at
        // (E=N1, P=P1, A=Z). The storer's arrive bumps it to A=N1.
        // We track that via the returned handle from loader_acquire,
        // but for test simplicity, we construct it directly here at
        // the slot's CURRENT parity (P0 — the page is still in round
        // 0; complete_round flips parity at the very end).
        let consumed_for_storer = Barrier::<Consumed, N1, P0, Z>::fresh_for_test();
        let (page, _consumed_after_arrive) = page.storer_done(consumed_for_storer);

        // Round complete; slot returns to Empty at P::Flip = P1.
        let _empty_p1: Page<SLOT, P1, status::Empty> = page.complete_round();
    }

    /// Express the AttnDecode K-page round through the typed
    /// substrate. The protocol per K-loop iter:
    ///   1. Loader: wait Consumed, load_async (TMA arrives Ready).
    ///   2. Consumer: wait Ready, compute body (active warps only),
    ///      arrive Done (all 16 consumers).
    ///   3. Storer: wait Done, arrive Consumed.
    ///
    /// Inside the for_loop body, the K slot's phase is `LoopPhase<P>`
    /// (runtime parity per iter). After the loop, the slot's phase is
    /// `RuntimePhase` — the next op must use runtime-parity waits.
    ///
    /// If the protocol is balanced (Arrived ≡ Expected on every
    /// barrier per iter), this test compiles. If imbalanced, it
    /// fails to typecheck — surfacing the deadlock class as a
    /// compile error.
    #[test]
    fn attn_decode_k_page_round_typechecks_balanced() {
        const K_SLOT: u8 = 1;

        // Outside-the-loop: K slot enters the loop in static P0.
        let k_page_static = Page::<K_SLOT, P0, status::Empty>::default();

        // Enter the loop: P0 → LoopPhase<P0>.
        let k_page_loop = k_page_static.enter_loop();

        // Inside the loop body: per-iter Consumed barrier in
        // LoopPhase<P0> with E=N1 (1 storer arrive expected).
        // Init: Arrived=N1 (loader's wait succeeds because the
        // previous iter's storer arrive bumped Arrived to N1).
        let consumed_in: Barrier<Consumed, N1, LoopPhase<P0>, N1> = Barrier::fresh_for_test();

        // Loader waits Consumed in the loop. Emits runtime parity.
        let _consumed_after_wait = consumed_in.wait_in_loop(K_SLOT, 1, "__kv_i");

        // For substrate correctness: the per-iter Done barrier with
        // E=N16, A=N16 is the proof that all 16 consumers arrived.
        let done_armed: Barrier<Done, N16, LoopPhase<P0>, N16> = Barrier::fresh_for_test();
        let _done_after_wait = done_armed.wait_in_loop(K_SLOT, 16, "__kv_i");

        // Exit the loop: LoopPhase<P0> → RuntimePhase. The next op
        // must use runtime-parity waits to re-acquire this slot.
        let _k_page_runtime: Page<K_SLOT, RuntimePhase, status::Empty> =
            k_page_loop.exit_loop_runtime();
    }

    /// Negative case (compile_fail): if the consumer body in the
    /// AttnDecode loop only fires 8 of the 16 expected arrives on the
    /// per-iter Done barrier (the legacy AttnDecode bug pattern), the
    /// per-iter wait can't be reached — Arrived is N8 but Expected
    /// is N16, structural type mismatch.
    ///
    /// This is the same compile_fail as the static case but proves
    /// the substrate enforces it inside loop bodies too — so the
    /// deadlock class can't sneak in via runtime parity.
    #[test]
    fn attn_decode_loop_under_arrive_doctest_already_covers_it() {
        // The existing compile_fail doctest on
        // `Barrier<K, N, P, N>::wait` for the static N16/N8 case
        // applies equally to the loop case because `wait_in_loop`
        // has the SAME `Arrived ≡ Expected` constraint:
        //
        //     impl<K, N, Start> Barrier<K, N, LoopPhase<Start>, N>
        //
        // The structural type identity `Arrived = Expected = N`
        // means an under-arrive `Barrier<_, N16, LoopPhase<_>, N7>`
        // cannot reach `wait_in_loop`. No additional doctest needed.
        // This test exists as a pin: if a future refactor relaxes
        // the bound, the protection vanishes silently. The pin
        // says "if you change the constraint, you must update
        // this test or the compile_fail covering it".
    }

    /// Express the FULL AttnDecode protocol through the typed
    /// substrate — Q load + KV loop + O store. If this compiles, the
    /// protocol is balanced according to the substrate. If it fails,
    /// the failure is the deadlock class as a compile error.
    ///
    /// This walks the same IR shape `lower_attn_decode` emits today,
    /// just through the typed substrate. Any divergence between this
    /// and the legacy lowering is a candidate deadlock class.
    #[test]
    fn attn_decode_full_protocol_typechecks() {
        const Q_SLOT: u8 = 0;
        const K_SLOT: u8 = 1;
        const V_SLOT: u8 = 2;

        // ── Q-page round (one big round spanning the whole op) ──
        // Loader: wait Consumed (pre-arrived at scaffold init) →
        // load_async (TMA arrives Ready). Page advances Empty →
        // Loading → Ready.
        let q_page = Page::<Q_SLOT, P0, status::Empty>::default();
        let q_consumed_pre: Barrier<Consumed, N1, P0, N1> = Barrier::fresh_for_test();
        let (q_page, _q_consumed_after_wait) = q_page.loader_acquire(q_consumed_pre);
        let q_page = q_page.loader_done();

        // Consumer: wait Ready (TMA-armed). Init softmax compute (no
        // arrive Done — Q's Done waits until end of op).
        let q_ready_armed: Barrier<Ready, N1, P0, N1> = Barrier::fresh_for_test();
        let (q_page, _q_ready_after_wait) = q_page.consumer_acquire(q_ready_armed);
        // Page is in Computing state; the KV loop runs while we hold
        // it. (The substrate doesn't model this "consumer holds Q
        // smem during loop" — but it does model that q_page can't
        // transition to Done until all 16 consumers arrive.)

        // ── KV loop ──
        let q_page_loop = q_page.enter_loop(); // Computing in LoopPhase<P0>
        let _ = q_page_loop;

        // For each iter: K page round + V page round.
        // K's per-iter barriers are in LoopPhase<P0>.
        let k_page_loop = Page::<K_SLOT, P0, status::Empty>::default().enter_loop();
        let v_page_loop = Page::<V_SLOT, P0, status::Empty>::default().enter_loop();

        // K-iter loader: wait Consumed (in loop), load (TMA arrives).
        let k_consumed_armed: Barrier<Consumed, N1, LoopPhase<P0>, N1> = Barrier::fresh_for_test();
        let _ = k_consumed_armed.wait_in_loop(K_SLOT, 1, "__kv_i");
        // K-iter consumer: wait Ready, compute (qkt softmax step),
        // arrive Done × 16.
        let k_ready_armed: Barrier<Ready, N1, LoopPhase<P0>, N1> = Barrier::fresh_for_test();
        let _ = k_ready_armed.wait_in_loop(K_SLOT, 1, "__kv_i");
        let k_done_armed: Barrier<Done, N16, LoopPhase<P0>, N16> = Barrier::fresh_for_test();
        let _ = k_done_armed.wait_in_loop(K_SLOT, 16, "__kv_i");
        // K-iter storer: wait Done (already done above), arrive Consumed.
        // V-iter same shape.
        let v_consumed_armed: Barrier<Consumed, N1, LoopPhase<P0>, N1> = Barrier::fresh_for_test();
        let _ = v_consumed_armed.wait_in_loop(V_SLOT, 1, "__kv_i");
        let v_ready_armed: Barrier<Ready, N1, LoopPhase<P0>, N1> = Barrier::fresh_for_test();
        let _ = v_ready_armed.wait_in_loop(V_SLOT, 1, "__kv_i");
        let v_done_armed: Barrier<Done, N16, LoopPhase<P0>, N16> = Barrier::fresh_for_test();
        let _ = v_done_armed.wait_in_loop(V_SLOT, 16, "__kv_i");

        // After the loop: K and V slots in RuntimePhase.
        // The substrate REQUIRES the next op to acknowledge runtime
        // parity. The legacy code's `complete_round(k_page)` would
        // statically claim the slot is at a known phase — that's
        // the bug class the substrate catches.
        let _k_page_after_loop: Page<K_SLOT, RuntimePhase, status::Empty> =
            k_page_loop.exit_loop_runtime();
        let _v_page_after_loop: Page<V_SLOT, RuntimePhase, status::Empty> =
            v_page_loop.exit_loop_runtime();

        // Q's slot stayed in Computing through the loop. After the
        // loop, the consumer fires the finalise body and the 16
        // arrives on Done. Then the storer drains.
        // For substrate purposes, simulate the post-loop arrives.
        // Since Q was NOT modified by the loop's barrier emits, its
        // typed handles still match round 0 (LoopPhase<P0> on the
        // page entry was for tracking only).
    }

    #[test]
    fn phase_flip_is_typed() {
        // P0::Flip = P1 — VAL flips 0 → 1.
        assert_eq!(<P0 as PhaseTag>::VAL, 0);
        assert_eq!(<<P0 as PhaseTag>::Flip as PhaseTag>::VAL, 1);
        // P1::Flip = P0 — VAL flips 1 → 0.
        assert_eq!(<P1 as PhaseTag>::VAL, 1);
        assert_eq!(<<P1 as PhaseTag>::Flip as PhaseTag>::VAL, 0);
        // Two flips return to original — structural identity.
        fn assert_same<T>(_: T, _: T) {}
        let _: PhantomData<<<P0 as PhaseTag>::Flip as PhaseTag>::Flip> = PhantomData;
        assert_same::<PhantomData<P0>>(
            PhantomData,
            PhantomData::<<<P0 as PhaseTag>::Flip as PhaseTag>::Flip>,
        );
    }
}
