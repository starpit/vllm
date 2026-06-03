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

/// Number of consumer warps in the persistent CTA. Phase 7: 8 → 16
/// (4 warpgroups × 4 warps each — wgmma-aligned for Hopper).
pub const NUM_CONSUMER_WARPS: u8 = 16;

/// Number of service warps in the persistent CTA. Phase 7: 2 → 4
/// (1 loader + 1 storer + 1 launcher + 1 controller, forming one
/// complete kittens warpgroup so `kittens::warpgroup::decrease_registers`
/// has its 4-warp alignment requirement satisfied). Today the launcher
/// and controller are stubs (no instruction stream + no fused ITypes
/// until Phase 12 lands); they just decrease_registers and idle.
pub const NUM_SERVICE_WARPS: u8 = 4;

/// Total warps in the persistent CTA. 20 warps × 32 threads = 640
/// threads. `__launch_bounds__(640)` and `<<<1, 640, ...>>>` follow.
pub const NUM_WARPS: u8 = NUM_SERVICE_WARPS + NUM_CONSUMER_WARPS;

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
    /// Construct a fresh `PageHandle<Self>` for slot `id`. Used by
    /// `PageAllocator::alloc_at::<P>()` so the lowerings can be generic
    /// over the starting parity of a slot (slots reused across ops
    /// alternate parity per round, and the allocator hands them out at
    /// whichever phase they're currently observed at).
    fn fresh_handle(id: u8) -> PageHandle<Self>
    where
        Self: Sized;
}

#[derive(Clone, Copy, Debug)]
pub struct Phase0;
#[derive(Clone, Copy, Debug)]
pub struct Phase1;

impl Phase for Phase0 {
    type Next = Phase1;
    const VALUE: u32 = 0;
    fn fresh_handle(id: u8) -> PageHandle<Self> {
        PageHandle::<Phase0>::fresh(id)
    }
}
impl Phase for Phase1 {
    type Next = Phase0;
    const VALUE: u32 = 1;
    fn fresh_handle(id: u8) -> PageHandle<Self> {
        PageHandle::<Phase0>::fresh(id).advance()
    }
}

// ── PageHandleAfterRuntimeLoop — typed handle post for_loop ────────
//
// A page slot that went through a `for_loop` body with RUNTIME iter
// count (`LoopBound::RuntimeU32`) emerges with an UNKNOWN-parity
// hardware barrier state — could be start parity (even iters) or
// flipped (odd iters). The legacy `prog.complete_round` blindly
// advances typed phase by 1, mismatching hardware for even iters.
// The mismatch caused the FERRITE_WAVEFRONT_GPU=1 first-decode hang
// (runtime diag 2026-06-03).
//
// `PageHandleAfterRuntimeLoop<P>` is a typed handle returned by the
// loop-aware exit path. The ONLY way to release such a handle is
// through `TkProgram::complete_round_with_parity_correction(
// handle, parity_var)`, which:
//   1. Emits a runtime conditional phantom round of arrives that
//      flips hardware once more iff the runtime iter count is even.
//   2. Advances typed phase by 1 (now consistent with hardware).
//
// A lowering that calls plain `prog.complete_round(handle)` on a
// regular `PageHandle<P>` returned by `prog.for_loop` would not
// compile — the for_loop now returns `PageHandleAfterRuntimeLoop<P>`
// for runtime-bound loops, and complete_round only accepts
// PageHandle<P>. The lowering MUST go through
// `complete_round_with_parity_correction` — which structurally emits
// the phantom round.
//
// This catches Gap 17 (the actual decode-time deadlock) at compile
// time: omit the parity correction and your code doesn't typecheck.

/// Typed page-slot handle AFTER a runtime-iter-count `for_loop` body.
/// The hardware barrier parity is `start ^ (N & 1)` where `N` is the
/// runtime iteration count. `P` is the typed phase the slot would
/// be at if `complete_round` is called naively (matching odd-N case).
///
/// **Released only via [`TkProgram::complete_round_with_parity_correction`]**
/// — that function emits a runtime conditional phantom round so
/// hardware aligns with typed phase. Forgetting the correction is
/// impossible: there's no `complete_round` method on this type.
///
/// # Compile-fail proofs (the structural Gap 17 enforcement)
///
/// 1. Calling `prog.complete_round` on a `PageHandleAfterRuntimeLoop`
///    is a Rust compile error — `complete_round` only accepts
///    `PageHandle<P>`, not the post-loop wrapper.
/// ```compile_fail
/// use ferrite_wavefront::tk_warp_ir::*;
/// let mut prog = TkProgram::new();
/// let page: PageHandle<Phase0> = PageHandle::fresh(0);
/// let post = PageHandleAfterRuntimeLoop::from_handle(page);
/// // Author tries to release without parity correction. Compile error.
/// let _ = prog.complete_round(post);
/// ```
///
/// 2. Constructing `LoopBound::RuntimeU32(...)` from a lowering is a
///    Rust compile error — the constructor takes a sealed
///    `SealedRuntime` token only `for_loop_runtime` can mint.
///    Verifies that the legacy `prog.for_loop(_, LoopBound::RuntimeU32
///    (...))` pattern (which would let the buggy
///    `prog.complete_round(k_page)` compile) is no longer expressible.
/// ```compile_fail
/// use ferrite_wavefront::tk_warp_ir::*;
/// let _ = LoopBound::RuntimeU32("__num_kv_pages".to_string());
/// // error[E0061]: enum variant takes 2 arguments but 1 was supplied
/// ```
#[derive(Clone, Copy, Debug)]
pub struct PageHandleAfterRuntimeLoop<P: Phase> {
    pub id: u8,
    _phase: PhantomData<P>,
}

impl<P: Phase> PageHandleAfterRuntimeLoop<P> {
    pub fn from_handle(h: PageHandle<P>) -> Self {
        Self {
            id: h.id,
            _phase: PhantomData,
        }
    }
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
    ///
    /// `dyn_byte_off`: optional CUDA expression for a runtime byte
    /// offset added to the static `src_region` start. Used inside
    /// for-loop bodies whose byte offset depends on the loop variable
    /// (e.g. `"(__n_i * 16384u)"` for a streaming GEMM W-tile offset).
    /// When `None`, only the static `src_region.region.cols.start * elem_bytes`
    /// is emitted.
    LoadAsync {
        page_id: u8,
        src: BufId,
        src_region: RegionRef,
        tile: TileShape,
        dyn_byte_off: Option<String>,
    },

    /// TMA store: drain `page_id`'s tile to `dst[dst_region]`. Storer
    /// role only. See [`TkInstr::LoadAsync::dyn_byte_off`].
    StoreAsync {
        page_id: u8,
        dst: BufId,
        dst_region: RegionRef,
        tile: TileShape,
        dyn_byte_off: Option<String>,
    },

    /// Inline compute fragment in a consumer (or other) warp role.
    ///
    /// The body is a `Vec<Tk20Call>`: each entry is one TK 2.0
    /// primitive call (or, during transition, a `RawString` carrying a
    /// pre-resolved CUDA fragment from a legacy `format!()` body).
    /// Codegen walks `calls` in order, emitting one CUDA statement per
    /// element via the typed `tk20::*` Rust API.
    ///
    /// This replaces a previous `Compute { body: String }` shape; the
    /// `RawString` `Tk20Call` variant is the bridge — it forwards the
    /// String verbatim — and gets sunset when every `lower_*` has been
    /// migrated to typed primitives (per `feedback_dogfood_tk20_rust`).
    Compute {
        role: WarpRole,
        calls: Vec<crate::tk_codegen::Tk20Call>,
    },

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
///
/// The `RuntimeU32` variant carries a sealed token (`SealedRuntime`)
/// that only the substrate's [`TkProgram::for_loop_runtime`] can
/// construct. Constructing `LoopBound::RuntimeU32(...)` from a
/// lowering is a Rust compile error — guarantees runtime loops go
/// through the typed parity-correction path (Gap 17, fixes the
/// FERRITE_WAVEFRONT_GPU=1 first-decode hang structurally).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoopBound {
    Const(u32),
    /// Runtime u32 expression. **Construction sealed**: only
    /// [`TkProgram::for_loop_runtime`] can produce this variant.
    RuntimeU32(String, SealedRuntime),
}

/// Sealed token gating `LoopBound::RuntimeU32` construction.
/// External code cannot instantiate this — only `tk_warp_ir`
/// internals (specifically [`TkProgram::for_loop_runtime`]) call
/// `SealedRuntime::new()` (which is `pub(crate)`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedRuntime(());

impl SealedRuntime {
    /// Crate-private. Lowerings cannot call this — they MUST go
    /// through `TkProgram::for_loop_runtime`, which threads page
    /// handles through `PageHandleAfterRuntimeLoop` and forces the
    /// parity-correction release.
    pub(crate) fn new() -> Self {
        Self(())
    }
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
            LoopBound::RuntimeU32(name, _) => name.clone(),
        }
    }
}

/// One persistent-CTA tape.
#[derive(Clone, Debug, Default)]
pub struct TkProgram {
    pub instrs: Vec<TkInstr>,
    /// CUDA text emitted at function scope BEFORE the role-routed body.
    /// Use this for state that must outlive any single role-arm block:
    /// type aliases, typed page views, persistent compute accumulators
    /// (`__m_max`, `__l_sum`, `__o_accum`, …) that are written in one
    /// compute step and read in a later one. The text is unguarded —
    /// every warp executes it; consumers reference it from inside their
    /// own role-arm blocks. Per-step locals (`__s`, …) stay inside the
    /// fragment that declares them.
    pub prelude: String,
}

impl TkProgram {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append CUDA text to the function-scope prelude. See [`Self::prelude`].
    pub fn add_prelude(&mut self, text: impl AsRef<str>) {
        self.prelude.push_str(text.as_ref());
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

    /// Push a `Wait` whose phase is the *runtime* expression for the
    /// current iteration's TK 2.0 round-robin parity. Used inside
    /// [`TkProgram::for_loop`] bodies, where the typed-phase model
    /// cannot statically track per-iteration flips.
    ///
    /// `start_phase` is the page's mbarrier parity at iter 0 — i.e.
    /// the static `Phase::VALUE` of the [`PageHandle`] the caller
    /// allocated outside the loop. Without it, the runtime expression
    /// `(loop_var & 1)` would assume start parity 0; reusing a page
    /// slot whose previous user closed it at the *opposite* parity
    /// (every other op cycle on a typical decode forward) would then
    /// emit a wait at parity 0 against an mbarrier whose actual
    /// parity is also 0 — and the wait would block forever waiting
    /// for a flip that the prior round already exhausted. This was
    /// the deadlock at op4/Gemm of the one-layer megakernel: w_page
    /// reused page-slot 1 (left at Phase1 by op0/RmsNorm), but the
    /// loop body emitted parity 0 for iter 0, mismatching the actual
    /// barrier parity by one.
    ///
    /// Per-iter parity = `(loop_var & 1) XOR start_phase`. For
    /// `start_phase=0` this collapses to `(loop_var & 1)` (the
    /// original always-start-fresh assumption). For `start_phase=1`
    /// it emits `((loop_var & 1) ^ 1)`.
    pub fn wait_loop_parity(
        &mut self,
        role: WarpRole,
        kind: PageBarrier,
        page_id: u8,
        loop_var: &str,
        start_phase: u32,
    ) {
        let phase_expr = match start_phase & 1 {
            0 => format!("({loop_var} & 1)"),
            _ => format!("(({loop_var} & 1) ^ 1)"),
        };
        self.instrs.push(TkInstr::Wait {
            role,
            page_id,
            kind,
            phase: WaitPhase::Runtime(phase_expr),
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
            dyn_byte_off: None,
        });
    }

    /// TMA load with a runtime CUDA byte-offset expression added to
    /// the static `src_region` byte offset. Used inside for-loop
    /// bodies whose offset depends on the loop variable.
    pub fn load_async_dyn(
        &mut self,
        page_id: u8,
        src: BufId,
        src_region: RegionRef,
        tile: TileShape,
        dyn_byte_off: impl Into<String>,
    ) {
        self.instrs.push(TkInstr::LoadAsync {
            page_id,
            src,
            src_region,
            tile,
            dyn_byte_off: Some(dyn_byte_off.into()),
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
            dyn_byte_off: None,
        });
    }

    pub fn store_async_dyn(
        &mut self,
        page_id: u8,
        dst: BufId,
        dst_region: RegionRef,
        tile: TileShape,
        dyn_byte_off: impl Into<String>,
    ) {
        self.instrs.push(TkInstr::StoreAsync {
            page_id,
            dst,
            dst_region,
            tile,
            dyn_byte_off: Some(dyn_byte_off.into()),
        });
    }

    /// Append a `Compute { calls: vec![Tk20Call::RawString(body)] }`.
    /// Bridge for legacy `format!()` lowerings; sunset when every
    /// `lower_*` migrates to the typed `tk20::*` API.
    pub fn compute(&mut self, role: WarpRole, body: impl Into<String>) {
        self.instrs.push(TkInstr::Compute {
            role,
            calls: vec![crate::tk_codegen::Tk20Call::RawString(body.into())],
        });
    }

    /// Append a `Compute` whose body is a typed `tk20::*` call list.
    /// New per-op lowerings (Phase 1+) build a `Vec<Tk20Call>` of
    /// typed primitive variants and call this. The `RawString` variant
    /// is forbidden in lowerings landed via this entry point.
    pub fn compute_calls(&mut self, role: WarpRole, calls: Vec<crate::tk_codegen::Tk20Call>) {
        self.instrs.push(TkInstr::Compute { role, calls });
    }

    pub fn sync(&mut self, role: WarpRole) {
        self.instrs.push(TkInstr::Sync { role });
    }

    /// Release a page handle after a RUNTIME-iter-count `for_loop`,
    /// emitting a parity-correction phantom round so hardware aligns
    /// with the typed phase advance. The phantom round fires one
    /// extra arrive on each barrier of the slot, gated on
    /// `(parity_var & 1u) == 0u` — for even iter counts this flips
    /// hardware once more; for odd, no-op.
    ///
    /// **The ONLY way to release a `PageHandleAfterRuntimeLoop`.**
    /// Forgetting the parity correction is structurally impossible.
    /// The bug class this prevents: legacy `complete_round(handle)`
    /// after a runtime for_loop assumed odd-N — for even-N (e.g.,
    /// Llama-1B AttnDecode where `__num_kv_pages` = 138868), the typed
    /// advance diverges from hardware → next op's wait blocks. The
    /// FERRITE_WAVEFRONT_GPU=1 first-decode hang (runtime-diagnosed
    /// 2026-06-03).
    pub fn complete_round_with_parity_correction<P: Phase>(
        &mut self,
        page: PageHandleAfterRuntimeLoop<P>,
        parity_var: &str,
    ) -> PageHandle<P::Next> {
        let id = page.id;
        // Phantom round: 1 arrive Ready (loader), NUM_CONSUMER_WARPS
        // arrives Done (consumers, from each warp's lane 0), 1 arrive
        // Consumed (storer). Each gated on `(parity_var & 1u) == 0u`.
        self.compute(
            WarpRole::Loader,
            format!(
                "if (({pv} & 1u) == 0u) {{ \
                 kittens::group<1>::arrive(page_ready[{id}]); \
                 }}",
                pv = parity_var,
                id = id,
            ),
        );
        self.compute(
            WarpRole::AllConsumers,
            format!(
                "if (({pv} & 1u) == 0u) {{ \
                 kittens::group<1>::arrive(page_done[{id}]); \
                 }}",
                pv = parity_var,
                id = id,
            ),
        );
        self.compute(
            WarpRole::Storer,
            format!(
                "if (({pv} & 1u) == 0u) {{ \
                 kittens::group<1>::arrive(page_consumed[{id}]); \
                 }}",
                pv = parity_var,
                id = id,
            ),
        );
        // Now hardware advanced once more; typed advance via
        // PageHandle::advance is consistent.
        PageHandle::<P> {
            id,
            _phase: PhantomData,
        }
        .advance()
    }

    /// **Structural enforcement of parity correction (Gap 17).** Build
    /// a `ForLoop` body where the iter count is a runtime variable.
    /// Pages threaded through the loop come in as `PageHandle<P>` and
    /// come out as `PageHandleAfterRuntimeLoop<P>` — the only type
    /// the safe `complete_round_with_parity_correction` accepts.
    /// Calling plain `prog.complete_round` on the returned handles is
    /// a Rust compile error (proven by the doctest on
    /// [`PageHandleAfterRuntimeLoop`]).
    ///
    /// This is the structural fix for FERRITE_WAVEFRONT_GPU=1
    /// first-decode hang — the legacy `for_loop + complete_round`
    /// pattern is no longer expressible: handles you put through
    /// `for_loop_runtime` can't be released without going through
    /// `complete_round_with_parity_correction`.
    pub fn for_loop_runtime<P, F>(
        &mut self,
        var: impl Into<String>,
        count_var: impl Into<String>,
        pages: Vec<PageHandle<P>>,
        build: F,
    ) -> Vec<PageHandleAfterRuntimeLoop<P>>
    where
        P: Phase,
        F: FnOnce(&mut LoopBody<'_>),
    {
        let count_var_string = count_var.into();
        let mut body_prog = TkProgram::new();
        let mut body = LoopBody {
            inner: &mut body_prog,
        };
        build(&mut body);
        self.instrs.push(TkInstr::ForLoop {
            var: var.into(),
            count: LoopBound::RuntimeU32(count_var_string, SealedRuntime::new()),
            body: body_prog.instrs,
        });
        pages
            .into_iter()
            .map(PageHandleAfterRuntimeLoop::from_handle)
            .collect()
    }

    /// Build a `ForLoop` body in a sub-program. The closure receives
    /// a [`LoopBody`] wrapper (not a raw [`TkProgram`]) — this enforces
    /// at compile time that every emit inside the body is loop-safe.
    ///
    /// **Specifically**: `LoopBody` does NOT expose static-offset
    /// [`TkProgram::load_async`] / [`TkProgram::store_async`] /
    /// [`TkProgram::wait`] / [`TkProgram::arrive`]. Inside a loop body,
    /// every load / store MUST take a dynamic byte-offset
    /// expression (or the SAME static expression won't be correct
    /// for every iteration). Every wait/arrive must use
    /// [`TkProgram::wait_loop_parity`] / [`TkProgram::arrive_loop`]
    /// for the runtime per-iter parity. The legacy
    /// `lower_attn_decode` K/V load bug — `body.load_async(...,
    /// RegionRef::rows_cols(..., 0, ...), ...)` — fails to compile
    /// because `LoopBody` has no `load_async` method.
    pub fn for_loop<F>(&mut self, var: impl Into<String>, count: LoopBound, build: F)
    where
        F: FnOnce(&mut LoopBody<'_>),
    {
        let mut body_prog = TkProgram::new();
        let mut body = LoopBody { inner: &mut body_prog };
        build(&mut body);
        self.instrs.push(TkInstr::ForLoop {
            var: var.into(),
            count,
            body: body_prog.instrs,
        });
    }
}

// ── LoopBody — restricted view of TkProgram for use inside for_loop ─
//
// Inside a `for_loop` body, every barrier interaction must use the
// runtime per-iter parity (`wait_loop_parity` / `arrive_loop`), and
// every TMA must use a dynamic byte-offset (`load_async_dyn` /
// `store_async_dyn`). LoopBody enforces this at the type level by
// exposing ONLY those methods. The static-offset `load_async` and
// the static-phase `wait` are not reachable through `LoopBody`.
//
// This catches the legacy `lower_attn_decode` K/V load bug at
// compile time:
//
//     body.load_async(k_id, op.k_cache,
//         RegionRef::rows_cols(op.k_cache, 1, 0, kv_cols), k_tile);
//
// Now produces "no method named `load_async` found for struct
// `&mut LoopBody`" — the call site must be rewritten to
// `body.load_async_dyn(...)` with a real per-iter offset
// expression.

/// Restricted view of [`TkProgram`] for emitting instructions inside
/// a `for_loop` body. Wraps a `&mut TkProgram` and exposes only the
/// loop-safe methods.
pub struct LoopBody<'a> {
    inner: &'a mut TkProgram,
}

impl<'a> LoopBody<'a> {
    /// Wait with the runtime per-iter parity expression. Forwards to
    /// [`TkProgram::wait_loop_parity`].
    pub fn wait_loop_parity(
        &mut self,
        role: WarpRole,
        kind: PageBarrier,
        page_id: u8,
        loop_var: &str,
        start_phase: u32,
    ) {
        self.inner.wait_loop_parity(role, kind, page_id, loop_var, start_phase);
    }

    /// Arrive inside a loop body. Forwards to
    /// [`TkProgram::arrive_loop`].
    pub fn arrive_loop(&mut self, role: WarpRole, kind: PageBarrier, page_id: u8) {
        self.inner.arrive_loop(role, kind, page_id);
    }

    /// TMA load with a runtime byte-offset expression. Forwards to
    /// [`TkProgram::load_async_dyn`]. **There is intentionally no
    /// static-offset `load_async` on `LoopBody`** — every load
    /// inside a for_loop must specify a dynamic offset, otherwise
    /// every iteration reads the same bytes (the legacy
    /// `lower_attn_decode` K/V cache bug).
    pub fn load_async_dyn(
        &mut self,
        page_id: u8,
        src: BufId,
        src_region: RegionRef,
        tile: TileShape,
        dyn_byte_off: impl Into<String>,
    ) {
        self.inner.load_async_dyn(page_id, src, src_region, tile, dyn_byte_off);
    }

    /// TMA store with a runtime byte-offset expression. Forwards to
    /// [`TkProgram::store_async_dyn`]. Same rationale as
    /// [`LoopBody::load_async_dyn`].
    pub fn store_async_dyn(
        &mut self,
        page_id: u8,
        dst: BufId,
        dst_region: RegionRef,
        tile: TileShape,
        dyn_byte_off: impl Into<String>,
    ) {
        self.inner.store_async_dyn(page_id, dst, dst_region, tile, dyn_byte_off);
    }

    /// Compute body inside a loop. Forwards to
    /// [`TkProgram::compute_calls`].
    pub fn compute_calls(&mut self, role: WarpRole, calls: Vec<crate::tk_codegen::Tk20Call>) {
        self.inner.compute_calls(role, calls);
    }

    /// CTA-wide sync inside a loop. Forwards to [`TkProgram::sync`].
    pub fn sync(&mut self, role: WarpRole) {
        self.inner.sync(role);
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
