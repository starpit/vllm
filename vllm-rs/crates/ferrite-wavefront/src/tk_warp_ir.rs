// SPDX-License-Identifier: Apache-2.0
//! `TkProgram` — the warp-tier IR for one persistent TK 2.0 megakernel
//! CTA.
//!
//! This is the layer below [`subtile_ir`]: where `subtile_ir` is per-
//! WORKER (one tape per CTA on CUDA, one per threadgroup on Metal) and
//! whole-op-granular, `tk_warp_ir` is per-WARP-ROLE inside one CTA. It
//! exists so the CUDA TK megakernel codegen is a literal `match`-walk
//! instead of synthesising warp roles, mbarrier pages, and phase parity
//! at emit time. See `SUBTILE_TK_PLAN.md` at the worktree root for the
//! design rationale.
//!
//! # The phase invariant
//!
//! Every TK 2.0 mbarrier handoff has a `u32` phase parity that flips
//! `0 ↔ 1` on each round; the loader, consumer, and storer must agree
//! on which parity belongs to "this round" or the wait blocks forever.
//! Today's `MegaDispatchState::page_rounds` re-derives parity per op
//! call site; one mis-counted op → silent drift → m=1 megakernel hang.
//!
//! Here phase is a TYPE: [`Phase0`] / [`Phase1`] with [`Phase::Next`]
//! flipping under [`TkProgram::complete_round`]. A wait/arrive on the
//! wrong parity is a Rust *compile* error, not a runtime deadlock.
//!
//! # The round invariant (TK 2.0 semantics, the part easy to get wrong)
//!
//! `mbarrier.wait(P)` blocks while `mbarrier.phase == P` and returns
//! once it differs. `mbarrier.arrive()` flips the phase. Per round the
//! sequence is loader-arrive-ready, consumer-arrive-done,
//! storer-arrive-consumed — three flips, one per barrier. Within a
//! round, every wait reads the SAME parity (`R & 1`); only at the
//! round boundary does the next round's parity change to `(R+1) & 1`.
//!
//! Therefore [`TkProgram::arrive`] does NOT advance phase: the page's
//! typed phase reflects the *round parity*, not the per-barrier flip.
//! The round closes with [`TkProgram::complete_round`], which is the
//! only phase-advancing call. (See the round-walk in
//! `tk_warp_ir::tests::round_parities_match_tk20_semantics`.)

#![allow(dead_code)]

use std::marker::PhantomData;

use crate::subtile_ir::{BufId, RegionRef};

// ── Substrate constants (TK 2.0 default; see header
//    `include/kittens.cuh::page` and the persistent kernel scaffold) ──

/// Number of mbarrier pages in the persistent CTA.
pub const NUM_PAGES: u32 = 13;

/// Size of one mbarrier page in bytes (TK 2.0 default).
pub const PAGE_SIZE: u32 = 16384;

/// Bytes of CTA-level scratch outside the page pool.
pub const SCRATCH_BYTES: u32 = 1024;

/// Number of consumer warps in the persistent CTA.
pub const NUM_CONSUMER_WARPS: u8 = 8;

// ── Phase as a type ─────────────────────────────────────────────────

/// A phase parity, encoded as a marker type so phase advance is a type-
/// level move (`P → P::Next`) and a wait on the wrong parity is a
/// compile error.
pub trait Phase: Copy + 'static {
    /// The other parity: `Phase0::Next == Phase1` and vice versa.
    type Next: Phase<Next = Self>;
    /// Runtime value the codegen passes to
    /// `kittens::group<N>::wait(sem, P::VALUE)`.
    const VALUE: u32;
}

#[derive(Clone, Copy, Debug)]
pub struct Phase0;
#[derive(Clone, Copy, Debug)]
pub struct Phase1;

impl Phase for Phase0 {
    type Next = Phase1;
    const VALUE: u32 = 0;
}
impl Phase for Phase1 {
    type Next = Phase0;
    const VALUE: u32 = 1;
}

// ── Page + scratch handles ──────────────────────────────────────────

/// One mbarrier page in the persistent CTA, carrying its CURRENT phase
/// parity as a type parameter. The `id` is the runtime page slot
/// `0..NUM_PAGES`; the phase is `P::VALUE`.
///
/// Phase moves only through [`PageHandle::advance`], which is the only
/// way to obtain a `PageHandle<P::Next>` from a `PageHandle<P>`. So a
/// codegen that emits a `Wait` reading `P::VALUE` on a page whose
/// (latest) handle was `P::Next` is rejected at the *call site* of the
/// instruction constructor, not at validation, and not at runtime.
#[derive(Clone, Copy, Debug)]
pub struct PageHandle<P: Phase> {
    pub id: u8,
    _phase: PhantomData<P>,
}

impl PageHandle<Phase0> {
    /// First handle on a freshly-initialised page. The
    /// initialisation convention matches TK 2.0:
    /// `page_ready[i].init(0)`, `page_done[i].init(0)`,
    /// `page_consumed[i].arrive_pre()` so its first wait reads phase 1.
    pub const fn fresh(id: u8) -> Self {
        Self {
            id,
            _phase: PhantomData,
        }
    }
}

impl<P: Phase> PageHandle<P> {
    pub const fn id(&self) -> u8 {
        self.id
    }
    pub const fn phase(&self) -> u32 {
        P::VALUE
    }
    /// Advance the phase. Call this when the program logically performs
    /// `arrive` on this page slot.
    pub fn advance(self) -> PageHandle<P::Next> {
        PageHandle {
            id: self.id,
            _phase: PhantomData,
        }
    }
}

/// A scratch byte region, with offset+length carried as const generics
/// so a slice that overlaps another's range fails to type-check.
///
/// SAFETY (typed): the construction site is responsible for proving
/// `OFF + LEN <= SCRATCH_BYTES`. A future tightening will move this to
/// a `where (OFF + LEN <= SCRATCH_BYTES):` bound once
/// `generic_const_exprs` stabilises; for now the construction is in
/// this crate only.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScratchSlice<const OFF: u32, const LEN: u32>;

// ── Warp roles ──────────────────────────────────────────────────────

/// Which warp role inside the persistent CTA owns an instruction. The
/// CUDA emit routes the instruction to a `if (warp_role == X) { ... }`
/// arm; instructions tagged [`WarpRole::All`] are emitted unguarded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WarpRole {
    /// The producer warp that issues TMA loads into pages.
    Loader,
    /// One of the `NUM_CONSUMER_WARPS` consumer warps. The runtime
    /// `u8` is the warp index inside the consumer group; instructions
    /// that all consumer warps share use [`WarpRole::AllConsumers`].
    Consumer(u8),
    /// All consumer warps execute the instruction.
    AllConsumers,
    /// The storer warp that issues TMA stores out of pages.
    Storer,
    /// All warps in the CTA execute the instruction (init, final sync).
    All,
}

// ── Page barriers ───────────────────────────────────────────────────

/// Which TK 2.0 mbarrier of a page slot a `Wait` / `Arrive` is talking
/// to. One slot has THREE mbarriers ([`PageBarrier::Ready`],
/// [`PageBarrier::Done`], [`PageBarrier::Consumed`]) so the loader →
/// consumer → storer → next-round-loader cycle is a closed three-step
/// handshake instead of one shared phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PageBarrier {
    /// Loader signals → consumer waits. "Page filled."
    Ready,
    /// Consumer signals → storer waits. "Compute done; flush to DRAM."
    Done,
    /// Storer signals → loader waits. "Page free for next round."
    Consumed,
}

// ── Tile descriptor ─────────────────────────────────────────────────

/// A 2-D tile within a page or scratch region. Const generics carry
/// the shape so a Mma whose A.K and B.K disagree is a compile error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileShape {
    pub rows: u32,
    pub cols: u32,
    pub elem_bytes: u32,
}

// ── Instructions ────────────────────────────────────────────────────

/// One instruction in a [`TkProgram`]. The CUDA codegen is a literal
/// `match`-walk over this enum; nothing else is decided at emit time.
#[derive(Clone, Debug)]
pub enum TkInstr {
    /// Wait on `page_done[page_id]` (or `page_ready` / `page_consumed`,
    /// per `kind`) at the captured `phase`.
    Wait {
        role: WarpRole,
        page_id: u8,
        kind: PageBarrier,
        /// Either a compile-time parity from a [`PageHandle<P>`] or a
        /// runtime CUDA parity expression (TK 2.0 KV round-robin).
        phase: WaitPhase,
    },

    /// Arrive on the matching mbarrier. Caller is responsible for
    /// having already advanced the corresponding [`PageHandle`] in the
    /// program-builder state.
    Arrive {
        role: WarpRole,
        page_id: u8,
        kind: PageBarrier,
    },

    /// TMA load: fill `page_id`'s tile from `src[src_region]`. Loader
    /// role only.
    LoadAsync {
        page_id: u8,
        src: BufId,
        src_region: RegionRef,
        tile: TileShape,
    },

    /// TMA store: drain `page_id`'s tile to `dst[dst_region]`. Storer
    /// role only.
    StoreAsync {
        page_id: u8,
        dst: BufId,
        dst_region: RegionRef,
        tile: TileShape,
    },

    /// Inline compute fragment in a consumer warp. Used for ops
    /// without a TK 2.0 primitive (RMS reduction, residual add). The
    /// `body` text has been pre-resolved by the per-op atom — codegen
    /// pastes it into the role-routed arm verbatim. (Equivalent of
    /// today's atom_lib `emit_*_body` but shorter, single-warp scope.)
    Compute { role: WarpRole, body: String },

    /// Group sync (`kittens::group<NUM_CONSUMER_WARPS>::sync()`).
    Sync { role: WarpRole },

    /// `for (uint <var> = 0; <var> < <count>; ++<var>) { <body> }`.
    /// The loop emits inside the role guard each instruction in `body`
    /// already chose (so a `body` mixing loader + consumer arrives is
    /// fine: the codegen routes each body instr separately, and the
    /// `for` is hoisted around all of them — i.e. all roles execute
    /// the same trip count, which is the TK 2.0 KV-page sweep idiom).
    ForLoop {
        var: String,
        /// Loop bound. Compile-time constants are baked literally;
        /// runtime bounds are emitted as the named u32 kernel argument
        /// the persistent-CTA scaffold defines (e.g. `__num_kv_pages`).
        count: LoopBound,
        body: Vec<TkInstr>,
    },
}

/// Loop trip count for [`TkInstr::ForLoop`]: either a const baked at
/// codegen time or a named runtime u32 the kernel scaffold provides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoopBound {
    Const(u32),
    RuntimeU32(String),
}

/// The phase argument to [`TkInstr::Wait`].
///
/// `Static` is the typed-phase path: the value came from a
/// [`PageHandle<P>::VALUE`], so a wait reading the wrong parity is a
/// Rust compile error at the call site. `Runtime` is the path used
/// inside TK 2.0 KV round-robin loops, where the parity is `(i & 1)`
/// for the current iteration `i` and the kernel must compute it at
/// run time. Inside a loop, the typed model can't track per-iteration
/// flips; the lowering uses [`TkProgram::wait_loop_parity`] to bind
/// the parity expression directly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WaitPhase {
    Static(u32),
    Runtime(String),
}

impl WaitPhase {
    pub fn cuda_expr(&self) -> String {
        match self {
            WaitPhase::Static(n) => n.to_string(),
            WaitPhase::Runtime(s) => s.clone(),
        }
    }
}

impl LoopBound {
    /// CUDA expression for this bound — what the codegen drops into
    /// the `for (...; i < THIS; ...)` slot.
    pub fn cuda_expr(&self) -> String {
        match self {
            LoopBound::Const(n) => n.to_string(),
            LoopBound::RuntimeU32(name) => name.clone(),
        }
    }
}

/// One persistent-CTA tape.
#[derive(Clone, Debug, Default)]
pub struct TkProgram {
    pub instrs: Vec<TkInstr>,
}

impl TkProgram {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a typed `Wait` whose `phase` is taken from the [`PageHandle`].
    pub fn wait<P: Phase>(
        &mut self,
        role: WarpRole,
        kind: PageBarrier,
        page: PageHandle<P>,
    ) -> PageHandle<P> {
        self.instrs.push(TkInstr::Wait {
            role,
            page_id: page.id,
            kind,
            phase: WaitPhase::Static(P::VALUE),
        });
        page
    }

    /// Push a `Wait` whose phase is the *runtime* expression
    /// `(<var> & 1)` — the TK 2.0 KV round-robin parity for iteration
    /// `<var>`. Used inside [`TkProgram::for_loop`] bodies, where the
    /// typed-phase model cannot statically track per-iteration flips.
    pub fn wait_loop_parity(
        &mut self,
        role: WarpRole,
        kind: PageBarrier,
        page_id: u8,
        loop_var: &str,
    ) {
        self.instrs.push(TkInstr::Wait {
            role,
            page_id,
            kind,
            phase: WaitPhase::Runtime(format!("({loop_var} & 1)")),
        });
    }

    /// Push an `Arrive` by raw `page_id`. Used inside
    /// [`TkProgram::for_loop`] bodies where the typed `arrive` cannot
    /// be called (no typed [`PageHandle<P>`] survives the loop's
    /// per-iteration parity flip). The barrier itself flips at run
    /// time as it always did; we just emit the call.
    pub fn arrive_loop(&mut self, role: WarpRole, kind: PageBarrier, page_id: u8) {
        self.instrs.push(TkInstr::Arrive {
            role,
            page_id,
            kind,
        });
    }

    /// Push a typed `Arrive`. Phase parity is the *round parity*, not
    /// the per-barrier flip — every wait within one round reads the
    /// same `(R & 1)`, so `arrive` returns the page handle UNCHANGED.
    /// The round boundary advance is [`TkProgram::complete_round`].
    pub fn arrive<P: Phase>(
        &mut self,
        role: WarpRole,
        kind: PageBarrier,
        page: PageHandle<P>,
    ) -> PageHandle<P> {
        self.instrs.push(TkInstr::Arrive {
            role,
            page_id: page.id,
            kind,
        });
        page
    }

    /// Close a round: flip the page's typed phase parity `P → P::Next`.
    /// Emits no IR — codegen never sees this, since per round each
    /// barrier flips exactly once and all three waits read the *old*
    /// parity (the one this call flips OUT of). The next round's first
    /// wait on this page will read `P::Next::VALUE`.
    pub fn complete_round<P: Phase>(&mut self, page: PageHandle<P>) -> PageHandle<P::Next> {
        page.advance()
    }

    pub fn load_async(
        &mut self,
        page_id: u8,
        src: BufId,
        src_region: RegionRef,
        tile: TileShape,
    ) {
        self.instrs.push(TkInstr::LoadAsync {
            page_id,
            src,
            src_region,
            tile,
        });
    }

    pub fn store_async(
        &mut self,
        page_id: u8,
        dst: BufId,
        dst_region: RegionRef,
        tile: TileShape,
    ) {
        self.instrs.push(TkInstr::StoreAsync {
            page_id,
            dst,
            dst_region,
            tile,
        });
    }

    pub fn compute(&mut self, role: WarpRole, body: impl Into<String>) {
        self.instrs.push(TkInstr::Compute {
            role,
            body: body.into(),
        });
    }

    pub fn sync(&mut self, role: WarpRole) {
        self.instrs.push(TkInstr::Sync { role });
    }

    /// Build a `ForLoop` body in a sub-program. The closure receives a
    /// fresh [`TkProgram`] to populate; on return its `instrs` become
    /// the loop body. This keeps lowering code shaped like ordinary
    /// straight-line tape — no manual `Vec<TkInstr>` plumbing.
    pub fn for_loop<F>(&mut self, var: impl Into<String>, count: LoopBound, build: F)
    where
        F: FnOnce(&mut TkProgram),
    {
        let mut body = TkProgram::new();
        build(&mut body);
        self.instrs.push(TkInstr::ForLoop {
            var: var.into(),
            count,
            body: body.instrs,
        });
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// TK 2.0 round semantics: every wait within one round reads the
    /// SAME parity `(R & 1)`. Per round each of the three barriers
    /// flips exactly once, but a wait reads the parity the barrier had
    /// at the *start* of the round — i.e., `R & 1`. Phase advances
    /// only at the round boundary via [`TkProgram::complete_round`].
    #[test]
    fn round_parities_match_tk20_semantics() {
        let mut p = TkProgram::new();
        let page: PageHandle<Phase0> = PageHandle::fresh(0);

        // ── Round 0: every wait reads 0 ──
        let page = p.wait(WarpRole::Loader, PageBarrier::Consumed, page);
        let page = p.arrive(WarpRole::Loader, PageBarrier::Ready, page);
        let page = p.wait(WarpRole::AllConsumers, PageBarrier::Ready, page);
        let page = p.arrive(WarpRole::AllConsumers, PageBarrier::Done, page);
        let page = p.wait(WarpRole::Storer, PageBarrier::Done, page);
        let page = p.arrive(WarpRole::Storer, PageBarrier::Consumed, page);
        // Round boundary: typed parity now Phase1.
        let page = p.complete_round(page);

        // ── Round 1: every wait reads 1 ──
        let page = p.wait(WarpRole::Loader, PageBarrier::Consumed, page);
        let page = p.arrive(WarpRole::Loader, PageBarrier::Ready, page);
        let page = p.wait(WarpRole::AllConsumers, PageBarrier::Ready, page);
        let page = p.arrive(WarpRole::AllConsumers, PageBarrier::Done, page);
        let _page = p.wait(WarpRole::Storer, PageBarrier::Done, page);

        let phases: Vec<String> = p
            .instrs
            .iter()
            .filter_map(|i| match i {
                TkInstr::Wait { phase, .. } => Some(phase.cuda_expr()),
                _ => None,
            })
            .collect();
        assert_eq!(
            phases,
            vec!["0", "0", "0", "1", "1", "1"],
            "round 0 reads 0, round 1 reads 1"
        );
    }

    /// Page ids and roles round-trip through the program.
    #[test]
    fn instrs_carry_role_page_kind() {
        let mut p = TkProgram::new();
        let page: PageHandle<Phase0> = PageHandle::fresh(7);
        p.wait(WarpRole::Loader, PageBarrier::Consumed, page);
        match &p.instrs[0] {
            TkInstr::Wait {
                role,
                page_id,
                kind,
                phase,
            } => {
                assert_eq!(*role, WarpRole::Loader);
                assert_eq!(*page_id, 7);
                assert_eq!(*kind, PageBarrier::Consumed);
                assert_eq!(phase, &WaitPhase::Static(0));
            }
            _ => panic!("expected Wait"),
        }
    }

    /// `advance` flips parity at the type level: this test would not
    /// compile if `advance` returned the same `Phase` it consumed.
    #[test]
    fn phase_advance_is_typed() {
        let p0: PageHandle<Phase0> = PageHandle::fresh(3);
        let p1: PageHandle<Phase1> = p0.advance();
        let p0_again: PageHandle<Phase0> = p1.advance();
        assert_eq!(p0_again.id(), 3);
        assert_eq!(p0_again.phase(), 0);
    }
}
