// SPDX-License-Identifier: Apache-2.0
//! Linear, target-agnostic **SubtileTape** — a topological linearization
//! of the [`crate::subtile_ir::SubtileIR`] DAG **with every dataflow
//! edge made explicit as a slot-lifecycle instruction**.
//!
//! ## What this is (and is not)
//!
//! SubtileTape is the sequential semantics: the order in which a
//! conceptual single thread would execute the DAG, with explicit
//! `OpenLoop` / `CloseLoop` brackets for runtime-bounded loops (the
//! `AttnDecode` KV-sweep being the only such loop today).
//!
//! Every cross-Compute hazard (RAW from region overlap on op-output
//! tensors) surfaces as a **slot lifecycle**:
//!
//! ```text
//!   SlotHandle  (allocated, unwritten) ─compute_to─▶  SlotWritten (readers OK)
//!                                                         │
//!                                                   free_slot consumes
//!                                                         ▼
//!                                                       freed (id back in pool)
//! ```
//!
//! - `SlotHandle` is move-only (non-`Copy`, non-`Clone`). The Compute
//!   that writes a slot **consumes** the `SlotHandle`. Writing the same
//!   slot twice is a compile error.
//! - `SlotWritten` is move-only. Readers borrow it (`&SlotWritten`,
//!   multi-read OK). `free_slot` consumes the `SlotWritten`. Reading
//!   after free is a compile error.
//! - `compute_to` takes `&[&SlotWritten]` for reads; **reading a slot
//!   before it was written is a compile error.**
//! - `SlotId` / `SlotHandle` / `SlotWritten` are sealed — only path is
//!   the builder.
//!
//! Slot count, slot lifetime, and slot-write proof are target-agnostic
//! (a function of the DAG's liveness analysis, not the target). Only
//! **slot-physical-realization** (memory-tier capacity per slot, the
//! sync primitive that discharges the hazard) is target-specific —
//! that lives at TkTape.
//!
//! ## What does NOT live here
//!
//! Per plan §4 commit 3.b — see SUBTILE_TAPE_CONSTRAINTS.md for the
//! canonical exclusion list. All target-specific concepts
//! (execution-unit abstractions, memory-tier classification,
//! visibility primitives, pipeline-state tracking) belong on TkTape.
//! The IR-level witnesses (`KvCacheLayout`, `KvCacheProducer`,
//! `RopeForm`, online-softmax state) live on the SubtileIR `SubOp`
//! variants; the lowering looks them up by `SubtileId`.
//!
//! Plan §5 K2 is mechanically grep-checkable against this file: a
//! grep for any of the forbidden target-specific tokens must return
//! zero hits.
//!
//! For the canonical exclusion list see
//! [`crate::subtile_ir`] and `SUBTILE_TAPE_CONSTRAINTS.md`. K2
//! (target-agnostic SubtileTape, mechanically grep-checkable) is the
//! kill criterion this module enforces.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::marker::PhantomData;

use crate::subtile_ir::SubtileId;

// ── Sealed handles ──────────────────────────────────────────────────

#[doc(hidden)]
pub mod sealed {
    /// Sealing token. The inner `()` is `pub(super)`, so external code
    /// cannot construct a `Seal` value — making every type that carries
    /// a `Seal` field constructable only by code in `subtile_tape`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct Seal(pub(super) ());
}

/// Sealed slot id. The dense index identifying one writer / multi-reader
/// arena cell on the tape. Constructable only via [`TapeBuilder::alloc_slot`].
///
/// ```compile_fail
/// // Sealed; struct-literal construction is rejected.
/// let _ = ferrite_wavefront::subtile_tape::SlotId { id: 0 };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SlotId {
    id: u32,
    _seal: sealed::Seal,
}

impl SlotId {
    pub const fn index(&self) -> u32 {
        self.id
    }
}

/// Move-only proof that a slot has been **allocated but not yet
/// written**. The `Compute` that writes the slot consumes the
/// `SlotHandle` (by-value), so a slot cannot be written twice — the
/// move semantics enforce single-writer at compile time. Non-`Copy`,
/// non-`Clone` by construction.
#[derive(Debug)]
pub struct SlotHandle {
    slot: SlotId,
    _seal: sealed::Seal,
}

impl SlotHandle {
    pub const fn slot(&self) -> SlotId {
        self.slot
    }
}

/// Move-only proof that a slot has been **written** and is available
/// for reads. Borrowed (`&SlotWritten`) by readers — multi-read OK.
/// Consumed by [`TapeBuilder::free_slot`] — no use-after-free at compile
/// time. Non-`Copy`, non-`Clone` by construction.
#[derive(Debug)]
pub struct SlotWritten {
    slot: SlotId,
    _seal: sealed::Seal,
}

impl SlotWritten {
    pub const fn slot(&self) -> SlotId {
        self.slot
    }
}

/// Loop-variable handle. The id matched by [`Instr::OpenLoop`] and
/// [`Instr::CloseLoop`]. Constructable only as the return value of
/// [`TapeBuilder::open_loop`].
///
/// ```compile_fail
/// // Sealed; struct-literal construction is rejected.
/// let _ = ferrite_wavefront::subtile_tape::LoopVarId { id: 0 };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LoopVarId {
    id: u32,
    _seal: sealed::Seal,
}

impl LoopVarId {
    pub const fn index(&self) -> u32 {
        self.id
    }
}

/// Runtime-bound handle (for an `OpenLoop` whose iteration count is a
/// runtime quantity — e.g. `seq_len` for AttnDecode's KV-sweep). The
/// concrete kernel-arg slot is bound at the per-target lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RuntimeBoundId {
    id: u32,
    _seal: sealed::Seal,
}

impl RuntimeBoundId {
    pub const fn index(&self) -> u32 {
        self.id
    }
}

// ── Loop bound ──────────────────────────────────────────────────────

/// Iteration count of an [`Instr::OpenLoop`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LoopBound {
    /// Statically known iteration count.
    Const(u32),
    /// Runtime iteration count, supplied at kernel launch (e.g.
    /// `seq_len` for AttnDecode).
    Runtime(RuntimeBoundId),
}

// ── Instructions ────────────────────────────────────────────────────

/// One tape instruction. The tape is one linear stream — sequential
/// semantics. Every DAG edge surfaces as an explicit slot operation:
/// `AllocSlot` mints, `Compute { writes, reads }` writes once + reads N,
/// `FreeSlot` retires.
/// One positional input to a `Compute` Instr. Either a previously-
/// computed slot (`Computed(SlotId)`) or an external graph-source
/// tensor region (`External { tensor, region }`) that wasn't produced
/// by an upstream Compute.
///
/// Per the panic-RCA workflow (wpzaaucfk): the previous
/// `reads: Vec<SlotId>` design dropped external sources because
/// `predecessors()` skips graph-source tensors, so SubOps whose inputs
/// are all external (e.g. RmsNorm at decoder layer 0, with x and gamma
/// both external) reached `lower_compute` with an empty reads slice
/// and panicked at `reads[0]`.
///
/// Now Compute carries the FULL positional input list — both
/// computed-slot edges and external-source references — so each
/// lower_compute arm sees `node.inputs[i]` resolved to either a
/// page (Computed) or a tensor handle (External).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeInput {
    /// The input was produced by one or more earlier `Compute` writes
    /// covering the consumer's read region. With single-writer producers
    /// (the pre-N-tile baseline) the vec has length 1; once an N-tiled
    /// producer (e.g. Gemm with `nb < u32::MAX`) emits one node per
    /// col-block, every overlapping writer lands in the vec in
    /// ascending-SubtileId order. Per Patch 1 step (a) of
    /// `SPLIT_OVERSIZED_HANDOFF.md`. The vec is always non-empty
    /// (enforced at construction in `lower_dag_to_tape`).
    Computed(Vec<SlotId>),
    /// The input is a leaf graph-source (model weight, cache handle,
    /// pre-populated KV slot, etc.). Lowering side resolves the tensor
    /// + region against the SubtileIR's source manifest.
    External {
        tensor: crate::subtile_ir::TensorId,
        region: crate::subtile_ir::Region,
    },
}

impl ComputeInput {
    /// Single-writer view — returns the sole [`SlotId`] when exactly one
    /// upstream `Compute` writes this input's region. Panics with a
    /// named message for `External` inputs (lowering arms that haven't
    /// learned to resolve external sources yet) and for multi-writer
    /// `Computed` inputs (lowering arms that haven't been lifted to
    /// consume per-block predecessor pages yet — Patch 1 (b)-(d) of
    /// `SPLIT_OVERSIZED_HANDOFF.md`).
    pub fn expect_single_computed(&self, arm: &'static str, pos: usize) -> SlotId {
        match self {
            Self::Computed(slots) => {
                assert_eq!(
                    slots.len(),
                    1,
                    "lower_compute {} input[{}] is multi-writer \
                     (writers={:?}); this arm has not been lifted to \
                     multi-page consumption yet (Patch 1 step (b)-(d) \
                     of SPLIT_OVERSIZED_HANDOFF.md)",
                    arm,
                    pos,
                    slots,
                );
                slots[0]
            }
            Self::External { tensor, region } => panic!(
                "lower_compute {} input[{}] is External \
                 (tensor={:?}, region={:?}); external-source resolution \
                 is Phase A step 4+ of the panic-RCA plan",
                arm, pos, tensor, region,
            ),
        }
    }
}

/// Compile-time-arity wrapper around the positional input list of a
/// [`Instr::Compute`]. Per Phase A step 7 of the panic-RCA plan +
/// `feedback_compile_time_or_garbage` (INVIOLABLE): arity is a
/// type-level property, not a runtime check.
///
/// Each fixed-arity variant holds an `[ComputeInput; N]` — destructure-
/// matching `ComputeInputs::A2([in0, in1])` in `lower_compute` is
/// structural and rejects any other arity at rustc time. The
/// `Variadic` variant covers ops with dynamic arity (`SumReduce`'s
/// split-K combine, `AttnDecode`'s odd-arity cache pairs).
///
/// Per-arity constructors on [`TapeBuilder`] take the right number
/// of `ComputeInputBuild<'_>` arguments — wrong arity at construction
/// is a function-signature type error, not a runtime panic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeInputs {
    A1([ComputeInput; 1]),
    A2([ComputeInput; 2]),
    A3([ComputeInput; 3]),
    A4([ComputeInput; 4]),
    A5([ComputeInput; 5]),
    A6([ComputeInput; 6]),
    Variadic(Vec<ComputeInput>),
}

impl ComputeInputs {
    /// Iterate inputs in positional order regardless of arity variant.
    /// Used by validator + barrier-wait emission.
    pub fn iter(&self) -> Box<dyn Iterator<Item = &ComputeInput> + '_> {
        match self {
            Self::A1(arr) => Box::new(arr.iter()),
            Self::A2(arr) => Box::new(arr.iter()),
            Self::A3(arr) => Box::new(arr.iter()),
            Self::A4(arr) => Box::new(arr.iter()),
            Self::A5(arr) => Box::new(arr.iter()),
            Self::A6(arr) => Box::new(arr.iter()),
            Self::Variadic(v) => Box::new(v.iter()),
        }
    }
    /// Number of positional inputs.
    pub fn len(&self) -> usize {
        match self {
            Self::A1(_) => 1,
            Self::A2(_) => 2,
            Self::A3(_) => 3,
            Self::A4(_) => 4,
            Self::A5(_) => 5,
            Self::A6(_) => 6,
            Self::Variadic(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Destructure a fixed-arity inputs list into a typed array.
    /// Each arm in `lower_compute` calls the helper matching its
    /// SubOp arity — destructure-pattern-matching `[in0, in1]` gives
    /// compile-time-indexed access to the inputs without runtime
    /// bounds checks. Wrong variant → panic with a named message
    /// (structurally dead given correct `dispatch_compute_inputs`
    /// in `lower_dag_to_tape`).
    pub fn expect_a1(&self, arm: &'static str) -> &[ComputeInput; 1] {
        match self {
            Self::A1(arr) => arr,
            _ => panic!("{arm}: expected ComputeInputs::A1, got {:?}", self.len()),
        }
    }
    pub fn expect_a2(&self, arm: &'static str) -> &[ComputeInput; 2] {
        match self {
            Self::A2(arr) => arr,
            _ => panic!("{arm}: expected ComputeInputs::A2, got {:?}", self.len()),
        }
    }
    pub fn expect_a3(&self, arm: &'static str) -> &[ComputeInput; 3] {
        match self {
            Self::A3(arr) => arr,
            _ => panic!("{arm}: expected ComputeInputs::A3, got {:?}", self.len()),
        }
    }
    pub fn expect_a4(&self, arm: &'static str) -> &[ComputeInput; 4] {
        match self {
            Self::A4(arr) => arr,
            _ => panic!("{arm}: expected ComputeInputs::A4, got {:?}", self.len()),
        }
    }
    pub fn expect_a5(&self, arm: &'static str) -> &[ComputeInput; 5] {
        match self {
            Self::A5(arr) => arr,
            _ => panic!("{arm}: expected ComputeInputs::A5, got {:?}", self.len()),
        }
    }
    pub fn expect_a6(&self, arm: &'static str) -> &[ComputeInput; 6] {
        match self {
            Self::A6(arr) => arr,
            _ => panic!("{arm}: expected ComputeInputs::A6, got {:?}", self.len()),
        }
    }
    pub fn expect_variadic(&self, arm: &'static str) -> &[ComputeInput] {
        match self {
            Self::Variadic(v) => v,
            _ => panic!("{arm}: expected ComputeInputs::Variadic, got {:?}", self.len()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Instr {
    /// Mint a fresh slot id for a producer's output. Pairs (eventually)
    /// with one [`Instr::Compute`] writing this slot, and one
    /// [`Instr::FreeSlot`] retiring it.
    AllocSlot { slot: SlotId },
    /// Compute the named SubtileIR node, writing its output into `writes`
    /// and reading from `inputs` (one per positional `node.inputs[i]`).
    /// The node's input/output regions live on the SubtileIR; this
    /// instruction names node identity + the FULL positional input
    /// list (computed slots + external sources, see [`ComputeInput`]).
    Compute {
        node: SubtileId,
        writes: SlotId,
        inputs: ComputeInputs,
    },
    /// Retire the named slot — the last consumer is done. Slot id
    /// returns to the pool (the per-target lowering may recycle).
    FreeSlot { slot: SlotId },
    /// Open a runtime-bounded loop over `bound` iterations (the
    /// AttnDecode KV-sweep). Body holds Computes only; no nested loops.
    /// The matching [`Instr::CloseLoop`] takes the same `var`.
    OpenLoop { var: LoopVarId, bound: LoopBound },
    /// Close the active loop opened by `OpenLoop`.
    CloseLoop { var: LoopVarId },
}

// ── The tape ────────────────────────────────────────────────────────

/// Linear, target-agnostic per-wavefront tape. Build via [`TapeBuilder`]
/// then call [`validate_subtile_tape`] to discharge the runtime
/// invariants the typestate cannot see.
///
/// Per plan §5 K5 + audit BLOCKER (`wewpteccb`): the `instrs` field is
/// private. The ONLY constructors are [`TapeBuilder::finish`] and
/// [`lower_dag_to_tape`] (both inside this module); external code
/// reads through [`SubtileTape::instrs`]. This seals the typestate so
/// the slot-lifecycle and loop-balance invariants the
/// `TapeBuilder<S>` typestate already enforces at compile time
/// cannot be bypassed by struct-literal construction.
#[derive(Clone, Debug)]
pub struct SubtileTape {
    instrs: Vec<Instr>,
    /// High-water mark of allocated slot ids (`0..num_slots`). Live-slot
    /// count at any program point is recoverable by replay.
    pub num_slots: u32,
    pub num_loop_vars: u32,
    pub num_runtime_bounds: u32,
}

impl SubtileTape {
    /// Read-only view of the instruction stream. Per K5: external code
    /// reads through this accessor; construction is sealed to
    /// [`TapeBuilder::finish`] + [`lower_dag_to_tape`].
    pub fn instrs(&self) -> &[Instr] {
        &self.instrs
    }
}

// ── TapeBuilder<S> typestate ────────────────────────────────────────

/// Compile-time state markers for [`TapeBuilder<S>`].
///
/// Each marker implements [`BuilderState`], whose associated `Loop` type
/// names the loop-context payload that state carries:
///
/// - `Outside::Loop = ()` — no loop in flight, no payload.
/// - `InsideLoop::Loop = LoopVarId` — exactly one loop in flight, the
///   `LoopVarId` that opened it.
///
/// This shape lifts the loop-context invariant from a runtime
/// `Option<LoopVarId>` into the type system: `close_loop` reads
/// `self.cur_loop: LoopVarId` directly with no unwrap, because the
/// `InsideLoop` marker structurally cannot be constructed without one.
pub mod state {
    use super::{BuilderState, LoopVarId};

    /// No `OpenLoop` is in flight. `alloc_slot` / `free_slot` /
    /// `open_loop` / `finish` are only available here.
    #[derive(Debug)]
    pub enum Outside {}
    /// An `OpenLoop` is in flight. `close_loop` is only available here;
    /// hazard primitives (`alloc_slot`, `free_slot`) are not.
    #[derive(Debug)]
    pub enum InsideLoop {}

    impl BuilderState for Outside {
        type Loop = ();
    }
    impl BuilderState for InsideLoop {
        type Loop = LoopVarId;
    }
}

/// Sealed marker trait for [`TapeBuilder`]'s typestate parameter `S`.
/// `Loop` names the loop-context payload that state carries; see the
/// [`state`] module docs.
pub trait BuilderState: builder_state_seal::Sealed {
    /// Loop-context payload — `()` outside any loop, `LoopVarId` inside.
    type Loop: std::fmt::Debug;
}

mod builder_state_seal {
    pub trait Sealed {}
    impl Sealed for super::state::Outside {}
    impl Sealed for super::state::InsideLoop {}
}

/// Typestate-tracked builder. The `S` parameter is one of
/// [`state::Outside`] / [`state::InsideLoop`]; the same `TapeBuilder`
/// type carries different methods depending on `S`. Misuse (e.g.
/// `close_loop` on `Outside`, `finish` on `InsideLoop`, `alloc_slot`
/// inside a loop) = no matching impl, **compile error**.
///
/// ```compile_fail
/// use ferrite_wavefront::subtile_tape::TapeBuilder;
/// // close_loop is only impl'd on TapeBuilder<state::InsideLoop>.
/// let b = TapeBuilder::new();
/// let _ = b.close_loop();
/// ```
///
/// ```compile_fail
/// use ferrite_wavefront::subtile_tape::{LoopBound, TapeBuilder};
/// // finish is only impl'd on TapeBuilder<state::Outside>.
/// let b = TapeBuilder::new();
/// let (inside, _var) = b.open_loop(LoopBound::Const(8));
/// let _ = inside.finish();
/// ```
///
/// ```compile_fail
/// use ferrite_wavefront::subtile_tape::{LoopBound, TapeBuilder};
/// // alloc_slot is only impl'd on TapeBuilder<state::Outside>.
/// let b = TapeBuilder::new();
/// let (mut inside, _var) = b.open_loop(LoopBound::Const(8));
/// let _h = inside.alloc_slot();
/// ```
///
/// Hazard primitives — `SlotHandle` is move-only (non-Copy, non-Clone),
/// so single-writer / read-before-write / read-after-free become
/// use-after-move compile errors:
///
/// ```compile_fail
/// // double-write: SlotHandle is consumed by the first compute_to.
/// use ferrite_wavefront::subtile_ir::SubtileId;
/// use ferrite_wavefront::subtile_tape::TapeBuilder;
/// let mut b = TapeBuilder::new();
/// let h = b.alloc_slot();
/// let _w = b.compute_to(SubtileId(0), h, &[]);
/// // h is moved; cannot use it again.
/// let _w2 = b.compute_to(SubtileId(0), h, &[]);
/// ```
///
/// ```compile_fail
/// // read-before-write: only `&SlotWritten` is acceptable as a read.
/// // `SlotHandle` cannot be borrowed where `&SlotWritten` is expected.
/// use ferrite_wavefront::subtile_ir::SubtileId;
/// use ferrite_wavefront::subtile_tape::TapeBuilder;
/// let mut b = TapeBuilder::new();
/// let h = b.alloc_slot();
/// let _w = b.compute_to(SubtileId(0), b.alloc_slot(), &[&h]);
/// ```
///
/// ```compile_fail
/// // read-after-free: free_slot consumes SlotWritten; further reads
/// // of the same token are use-after-move.
/// use ferrite_wavefront::subtile_ir::SubtileId;
/// use ferrite_wavefront::subtile_tape::TapeBuilder;
/// let mut b = TapeBuilder::new();
/// let h0 = b.alloc_slot();
/// let w0 = b.compute_to(SubtileId(0), h0, &[]);
/// b.free_slot(w0);
/// let h1 = b.alloc_slot();
/// let _w1 = b.compute_to(SubtileId(1), h1, &[&w0]);
/// ```
pub struct TapeBuilder<S: BuilderState = state::Outside> {
    instrs: Vec<Instr>,
    next_slot: u32,
    next_loop_var: u32,
    next_runtime_bound: u32,
    /// Loop-context payload of state `S`. `()` on `Outside`,
    /// `LoopVarId` on `InsideLoop`. Per-state, not `Option<…>` — the
    /// typestate parameter structurally encodes presence.
    cur_loop: S::Loop,
    /// `PhantomData<fn() -> S>` keeps the builder `Send + Sync` without
    /// implying `S: Send` / `S: Sync`.
    _state: PhantomData<fn() -> S>,
}

impl Default for TapeBuilder<state::Outside> {
    fn default() -> Self {
        Self::new()
    }
}

impl TapeBuilder<state::Outside> {
    pub fn new() -> Self {
        Self {
            instrs: Vec::new(),
            next_slot: 0,
            next_loop_var: 0,
            next_runtime_bound: 0,
            cur_loop: (),
            _state: PhantomData,
        }
    }

    /// Allocate a fresh runtime-bound id (e.g. for AttnDecode `seq_len`).
    pub fn alloc_runtime_bound(&mut self) -> RuntimeBoundId {
        let r = RuntimeBoundId {
            id: self.next_runtime_bound,
            _seal: sealed::Seal(()),
        };
        self.next_runtime_bound += 1;
        r
    }

    /// Mint a fresh slot id and emit `Instr::AllocSlot`. Returns the
    /// move-only `SlotHandle` — the only token that can be passed to a
    /// subsequent `compute_to` as `writes`. Outside-only: slot
    /// allocation is a hazard primitive, forbidden inside a loop body.
    pub fn alloc_slot(&mut self) -> SlotHandle {
        let slot = SlotId {
            id: self.next_slot,
            _seal: sealed::Seal(()),
        };
        self.next_slot += 1;
        self.instrs.push(Instr::AllocSlot { slot });
        SlotHandle {
            slot,
            _seal: sealed::Seal(()),
        }
    }

    /// Retire a written slot. Consumes the `SlotWritten` (so no further
    /// reads are typeable) and emits `Instr::FreeSlot`. Outside-only.
    pub fn free_slot(&mut self, w: SlotWritten) {
        self.instrs.push(Instr::FreeSlot { slot: w.slot });
    }

    /// Open a runtime-bounded loop. Returns a builder in `InsideLoop`
    /// state plus a fresh [`LoopVarId`] for the matching `close_loop`.
    pub fn open_loop(
        mut self,
        bound: LoopBound,
    ) -> (TapeBuilder<state::InsideLoop>, LoopVarId) {
        let var = LoopVarId {
            id: self.next_loop_var,
            _seal: sealed::Seal(()),
        };
        self.next_loop_var += 1;
        self.instrs.push(Instr::OpenLoop { var, bound });
        let inside = TapeBuilder::<state::InsideLoop> {
            instrs: self.instrs,
            next_slot: self.next_slot,
            next_loop_var: self.next_loop_var,
            next_runtime_bound: self.next_runtime_bound,
            cur_loop: var,
            _state: PhantomData,
        };
        (inside, var)
    }

    /// Finalize: produce the linear tape. Only available with no loop
    /// in flight (the `Outside` state).
    pub fn finish(self) -> SubtileTape {
        SubtileTape {
            instrs: self.instrs,
            num_slots: self.next_slot,
            num_loop_vars: self.next_loop_var,
            num_runtime_bounds: self.next_runtime_bound,
        }
    }
}

/// `compute_to` is the single point that writes a slot. It consumes the
/// `SlotHandle` (single-writer) and borrows `&SlotWritten` for each read
/// (write-before-read). Available on both `Outside` and `InsideLoop` —
/// AttnDecode's body computes inside its KV-sweep loop, so the workload
/// instruction must be reachable from both states. (The hazard
/// primitives — `alloc_slot`, `free_slot` — stay Outside-only.)
// Helper: same regions_overlap from subtile_ir but local-scope.
fn regions_overlap_helper(a: crate::subtile_ir::Region, b: crate::subtile_ir::Region) -> bool {
    a.rows.start < a.rows.end()
        && b.rows.start < b.rows.end()
        && a.cols.start < a.cols.end()
        && b.cols.start < b.cols.end()
        && (a.rows.start < b.rows.end() && b.rows.start < a.rows.end())
        && (a.cols.start < b.cols.end() && b.cols.start < a.cols.end())
}

/// Caller-side input form for [`TapeBuilder::compute_to`]. Mirrors
/// [`ComputeInput`] but borrows the `SlotWritten` tokens for computed
/// inputs (so the SlotWritten consumed-once seal stays intact).
///
/// `Computed` carries a `Vec<&SlotWritten>` — one entry per overlapping
/// upstream writer. With single-writer producers the vec has length 1;
/// once N-tiled producers emit one node per col-block, every
/// overlapping writer's `SlotWritten` lands in the vec.
pub enum ComputeInputBuild<'a> {
    Computed(Vec<&'a SlotWritten>),
    External {
        tensor: crate::subtile_ir::TensorId,
        region: crate::subtile_ir::Region,
    },
}

fn ci_from(b: &ComputeInputBuild<'_>) -> ComputeInput {
    match b {
        ComputeInputBuild::Computed(ws) => {
            ComputeInput::Computed(ws.iter().map(|w| w.slot).collect())
        }
        ComputeInputBuild::External { tensor, region } => ComputeInput::External {
            tensor: *tensor,
            region: *region,
        },
    }
}

fn push_compute_inputs(
    instrs: &mut Vec<Instr>,
    node: SubtileId,
    write: SlotHandle,
    inputs: ComputeInputs,
) -> SlotWritten {
    let writes_id = write.slot;
    instrs.push(Instr::Compute {
        node,
        writes: writes_id,
        inputs,
    });
    SlotWritten {
        slot: writes_id,
        _seal: sealed::Seal(()),
    }
}

/// Per-arity dispatch for builder callers that have a `&[ComputeInputBuild]`
/// of caller-determined size (e.g. lower_dag_to_tape walking
/// `node.inputs` whose length depends on the SubOp).
///
/// The arity-typed constructors (`compute_a1_to`, `compute_a2_to`,
/// etc.) are the COMPILE-TIME-SAFE entry points — caller passes the
/// right number of `ComputeInputBuild` arguments. This dispatch
/// helper exists for cases where the caller already has a slice of
/// the right length but doesn't statically know which length; it
/// builds the right [`ComputeInputs`] variant.
fn dispatch_compute_inputs(inputs: &[ComputeInputBuild<'_>]) -> ComputeInputs {
    match inputs.len() {
        1 => ComputeInputs::A1([ci_from(&inputs[0])]),
        2 => ComputeInputs::A2([ci_from(&inputs[0]), ci_from(&inputs[1])]),
        3 => ComputeInputs::A3([
            ci_from(&inputs[0]),
            ci_from(&inputs[1]),
            ci_from(&inputs[2]),
        ]),
        4 => ComputeInputs::A4([
            ci_from(&inputs[0]),
            ci_from(&inputs[1]),
            ci_from(&inputs[2]),
            ci_from(&inputs[3]),
        ]),
        5 => ComputeInputs::A5([
            ci_from(&inputs[0]),
            ci_from(&inputs[1]),
            ci_from(&inputs[2]),
            ci_from(&inputs[3]),
            ci_from(&inputs[4]),
        ]),
        6 => ComputeInputs::A6([
            ci_from(&inputs[0]),
            ci_from(&inputs[1]),
            ci_from(&inputs[2]),
            ci_from(&inputs[3]),
            ci_from(&inputs[4]),
            ci_from(&inputs[5]),
        ]),
        _ => ComputeInputs::Variadic(inputs.iter().map(ci_from).collect()),
    }
}

impl TapeBuilder<state::Outside> {
    /// Compute `node`, dispatching positional inputs to the right
    /// arity-typed [`ComputeInputs`] variant.  See per-arity helpers
    /// (`compute_a1_to`, `compute_a2_to`, etc.) for callers that
    /// statically know the arity.
    pub fn compute_to(
        &mut self,
        node: SubtileId,
        write: SlotHandle,
        inputs: &[ComputeInputBuild<'_>],
    ) -> SlotWritten {
        let ci = dispatch_compute_inputs(inputs);
        push_compute_inputs(&mut self.instrs, node, write, ci)
    }
}

impl TapeBuilder<state::InsideLoop> {
    pub fn compute_to(
        &mut self,
        node: SubtileId,
        write: SlotHandle,
        inputs: &[ComputeInputBuild<'_>],
    ) -> SlotWritten {
        let ci = dispatch_compute_inputs(inputs);
        push_compute_inputs(&mut self.instrs, node, write, ci)
    }

    /// Close the active loop. Returns a builder back in the `Outside`
    /// state — `finish` is available again. `cur_loop: LoopVarId` is
    /// read directly: the `InsideLoop` state structurally cannot exist
    /// without a `LoopVarId`.
    pub fn close_loop(mut self) -> TapeBuilder<state::Outside> {
        let var = self.cur_loop;
        self.instrs.push(Instr::CloseLoop { var });
        TapeBuilder::<state::Outside> {
            instrs: self.instrs,
            next_slot: self.next_slot,
            next_loop_var: self.next_loop_var,
            next_runtime_bound: self.next_runtime_bound,
            cur_loop: (),
            _state: PhantomData,
        }
    }
}

// ── Runtime validator ───────────────────────────────────────────────

/// One class of runtime-detectable invariant violation.
///
/// Per plan §5 K5 + audit BLOCKER fix (`wewpteccb`): only relational
/// (tape, SubtileIR) properties live here. The slot-lifecycle and
/// loop-balance invariants are sealed at the type level via
/// [`TapeBuilder<S>`]'s move-only `SlotHandle` / `SlotWritten` and
/// `state::Outside` / `state::InsideLoop` typestate; with [`SubtileTape`]'s
/// `instrs` field private, struct-literal construction is no longer
/// possible and those checks would be unreachable, so they were
/// removed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidationError {
    /// `Compute` references a node id not present in the SubtileIR.
    UnknownNode { node: SubtileId },
    /// A SubtileIR node was `Compute`d more than once across the tape.
    DuplicateCompute { node: SubtileId },
    /// A SubtileIR node has no `Compute` instruction in the tape (every
    /// op-output node must be computed exactly once).
    MissingCompute { node: SubtileId },
    /// Two adjacent `Compute` instructions appear in non-ascending
    /// `SubtileId` order — the tape must be a topological linearization
    /// of the SubtileIR (which is itself ascending-id-topo).
    TopoOrderViolation {
        prev_node: SubtileId,
        next_node: SubtileId,
    },
    /// A `Compute`'s reads on the SubtileIR DAG don't match its
    /// predecessor set — one DAG edge is missing from the slot-read
    /// list, or an extra read names a node that is not a predecessor.
    EdgeMismatch {
        node: SubtileId,
        expected_preds: Vec<SubtileId>,
        actual_read_writers: Vec<SubtileId>,
    },
}

/// Runtime validator — relational (tape, SubtileIR) checks only.
/// Per audit BLOCKER fix `wewpteccb`, slot-lifecycle and loop-balance
/// were removed: with `SubtileTape::instrs` private, the
/// `TapeBuilder<S>` typestate (move-only `SlotHandle` / `SlotWritten`,
/// `state::Outside` / `state::InsideLoop`) is the sole construction
/// path and discharges those at compile time.
///
/// Three checks remain:
///
/// 1. **Compute well-formedness** — every SubtileIR node is `Compute`d
///    exactly once; no `Compute` references an out-of-range node.
/// 2. **Topo order** — adjacent Computes appear in strictly ascending
///    `SubtileId` order (the SubtileIR is already ascending-id topo;
///    the tape is a valid linearization).
/// 3. **Edge coverage** — every `Compute`'s `reads` equals (set-wise)
///    the SubtileIR predecessor set of `node`. The slots being read
///    are the slots most-recently written by the predecessors; missing
///    or extra reads = `EdgeMismatch`.
pub fn validate_subtile_tape<F: crate::subtile_ir::RopeForm, K: crate::subtile_ir::KvCacheShape>(
    tape: &SubtileTape,
    graph: &crate::subtile_ir::SubtileIR<F, K>,
) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();
    check_node_refs(tape, graph, &mut errors);
    check_compute_wellformed(tape, graph, &mut errors);
    check_edge_coverage(tape, graph, &mut errors);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn check_node_refs<F: crate::subtile_ir::RopeForm, K: crate::subtile_ir::KvCacheShape>(
    tape: &SubtileTape,
    graph: &crate::subtile_ir::SubtileIR<F, K>,
    errors: &mut Vec<ValidationError>,
) {
    let n_nodes = graph.nodes.len() as u32;
    for instr in &tape.instrs {
        if let Instr::Compute { node, .. } = instr
            && node.0 >= n_nodes
        {
            errors.push(ValidationError::UnknownNode { node: *node });
        }
    }
}

fn check_compute_wellformed<F: crate::subtile_ir::RopeForm, K: crate::subtile_ir::KvCacheShape>(
    tape: &SubtileTape,
    graph: &crate::subtile_ir::SubtileIR<F, K>,
    errors: &mut Vec<ValidationError>,
) {
    let n = graph.nodes.len();
    let mut emit_count: Vec<u32> = vec![0; n];
    let mut last_id: Option<SubtileId> = None;
    for instr in &tape.instrs {
        if let Instr::Compute { node, .. } = instr
            && (node.0 as usize) < n
        {
            emit_count[node.0 as usize] += 1;
            if let Some(prev) = last_id
                && node.0 <= prev.0
            {
                errors.push(ValidationError::TopoOrderViolation {
                    prev_node: prev,
                    next_node: *node,
                });
            }
            last_id = Some(*node);
        }
    }
    for (i, &c) in emit_count.iter().enumerate() {
        let nid = SubtileId(i as u32);
        if c == 0 {
            errors.push(ValidationError::MissingCompute { node: nid });
        } else if c > 1 {
            errors.push(ValidationError::DuplicateCompute { node: nid });
        }
    }
}

/// Edge coverage — the only relational (tape, SubtileIR) hazard
/// check that survives K5. Each `Compute`'s `reads` set must equal
/// (set-wise) the predecessor set of its node in the SubtileIR; a
/// missing or extra read is `EdgeMismatch`. Slot-lifecycle bookkeeping
/// (which slot a node wrote, which slot a node reads) is enforced
/// at compile time by `TapeBuilder<S>`'s move-only `SlotHandle` /
/// `SlotWritten` typestate; here we only re-derive the writer-of
/// each slot from the tape walk so we can compare reads to preds.
fn check_edge_coverage<F: crate::subtile_ir::RopeForm, K: crate::subtile_ir::KvCacheShape>(
    tape: &SubtileTape,
    graph: &crate::subtile_ir::SubtileIR<F, K>,
    errors: &mut Vec<ValidationError>,
) {
    let preds = crate::subtile_ir::predecessors(graph);
    let mut writer_of: BTreeMap<u32, SubtileId> = BTreeMap::new();
    for instr in &tape.instrs {
        match instr {
            Instr::AllocSlot { .. } | Instr::FreeSlot { .. } => {}
            Instr::OpenLoop { .. } | Instr::CloseLoop { .. } => {}
            Instr::Compute { node, writes, inputs } => {
                writer_of.insert(writes.id, *node);
                // Walk only Computed inputs for edge validation;
                // External inputs reference graph-source tensors that
                // have no producer node and contribute no DAG edge.
                // Per Patch 1 step (a): each Computed input carries a
                // Vec<SlotId> of overlapping writers; every entry
                // contributes one read_writer.
                let mut read_writers: Vec<SubtileId> = Vec::new();
                for ci in inputs.iter() {
                    if let ComputeInput::Computed(slots) = ci {
                        for slot in slots {
                            if let Some(w) = writer_of.get(&slot.id).copied() {
                                read_writers.push(w);
                            }
                        }
                    }
                }
                let n_idx = node.0 as usize;
                if n_idx < preds.len() {
                    let mut expected = preds[n_idx].clone();
                    expected.sort();
                    let mut actual = read_writers;
                    actual.sort();
                    actual.dedup();
                    if expected != actual {
                        errors.push(ValidationError::EdgeMismatch {
                            node: *node,
                            expected_preds: expected,
                            actual_read_writers: actual,
                        });
                    }
                }
            }
        }
    }
}

// ── Lowering: SubtileIR → SubtileTape ───────────────────────────────

/// Lower a [`crate::subtile_ir::SubtileIR`] DAG to a linear
/// [`SubtileTape`] in one deterministic pass.
///
/// Walks `graph.nodes` in ascending `SubtileId` order (the SubtileIR is
/// itself ascending-id-topo, so this is a valid topological order).
/// For each node:
///
/// 1. `alloc_slot()` mints a fresh slot for the node's output.
/// 2. `compute_to(node, slot, &[<predecessor slots>])` writes it,
///    consuming the predecessors' `&SlotWritten` tokens (multi-read OK).
/// 3. After all consumers of a predecessor have read, `free_slot`
///    retires the predecessor's slot.
///
/// `SubOp::AttnDecode` wraps in an `OpenLoop` / `CloseLoop` pair over
/// a runtime-bounded count (the KV-sweep over `seq_len` blocks); the
/// AttnDecode `Compute` lives inside the loop, the slot lifecycle
/// (alloc / free) lives outside.
///
/// **No worker assignment, no fence, no memory class.** Per-target
/// realization happens at the per-target lowering (`lower_subtile_tape_to_tk_tape`).
///
/// The returned `SubtileTape` has been validated against `graph` via
/// [`validate_subtile_tape`]; callers can assume well-formedness.
pub fn lower_dag_to_tape<F: crate::subtile_ir::RopeForm, K: crate::subtile_ir::KvCacheShape>(
    valid: &crate::subtile_ir::ValidatedGraph<'_, F, K>,
) -> SubtileTape {
    use crate::subtile_ir::{SubOp, predecessors};

    // The ValidatedGraph<F> sealed witness discharges the structural-
    // precondition gate at the type level; per §5 K5 we no longer ship
    // a runtime validate(graph).expect here.
    let graph = valid.graph();

    let preds = predecessors(graph);
    // For each node, count of yet-to-be-emitted consumers — when the
    // count hits zero, that node's slot is freed. Sources don't appear
    // (no predecessor edge means no slot).
    let mut consumer_remaining: Vec<u32> = vec![0; graph.nodes.len()];
    for ps in &preds {
        for p in ps {
            consumer_remaining[p.0 as usize] += 1;
        }
    }

    // Active SlotWritten token per node, indexed by SubtileId.
    // `Vec<Option<...>>` instead of `BTreeMap<u32, ...>`: the index
    // is bounded by graph.nodes.len() at construction (no out-of-range
    // lookup possible), and `Option::take` makes the consumer-count
    // walk's contract explicit — the only way `take` returns None on
    // a predecessor is a bug in this function's loop-invariant
    // (consumer_remaining and predecessors derived from the same
    // `preds` array, walked in ascending SubtileId order; every
    // predecessor of node N has id < N and was inserted before N).
    let mut written: Vec<Option<SlotWritten>> = (0..graph.nodes.len()).map(|_| None).collect();
    let mut builder = TapeBuilder::new();
    // For positional input plumbing: build a tensor->writer-node-id
    // index so we can find the producer of each non-source input.
    // Mirrors predecessors() but indexed by tensor instead of returning
    // a flat node-id list — we need the per-input producer to keep
    // positional order intact.
    use crate::subtile_ir::Region;
    let mut writers: Vec<Vec<(u32, Region)>> =
        vec![Vec::new(); graph.tensors.len()];
    for node in &graph.nodes {
        let nid = node.id.0;
        // Per Patch 1 step (a) of SPLIT_OVERSIZED_HANDOFF.md: collect
        // ALL overlapping writers per consumer-input (not just the
        // first one). With single-writer producers each list has
        // length 1; once N-tiled producers emit one node per
        // col-block, every overlapping writer ends up in the list,
        // matching the SSA edge validator's expected_preds.
        //
        // Empty list = External / graph source.
        let mut input_producer: Vec<Vec<u32>> = Vec::with_capacity(node.inputs.len());
        for inp in &node.inputs {
            if graph.is_source(inp.tensor) {
                input_producer.push(Vec::new());
            } else {
                let mut producers: Vec<u32> = Vec::new();
                for (wid, wreg) in &writers[inp.tensor.0 as usize] {
                    if regions_overlap_helper(*wreg, inp.region) {
                        producers.push(*wid);
                        // No `break;` — collect every overlapping writer.
                    }
                }
                assert!(
                    !producers.is_empty(),
                    "lower_dag_to_tape: non-source input \
                     (tensor={:?}, region={:?}) has no overlapping writer; \
                     SubtileIR validator should have rejected this graph",
                    inp.tensor,
                    inp.region,
                );
                input_producer.push(producers);
            }
        }
        // Take each unique computed predecessor's `SlotWritten` token
        // exactly once. Multiple positional refs to the same producer
        // (or the same producer appearing at multiple input positions)
        // share via the resolved-token map. consumer_remaining is
        // decremented once per unique producer at the end of the loop.
        let mut taken_tokens: BTreeMap<u32, SlotWritten> = BTreeMap::new();
        for prods in &input_producer {
            for pid in prods {
                if !taken_tokens.contains_key(pid) {
                    let token = written[*pid as usize].take().expect(
                        "lower_dag_to_tape: predecessor missing live SlotWritten \
                         (consumer-count walk disagrees with positional inputs)",
                    );
                    taken_tokens.insert(*pid, token);
                }
            }
        }
        // Build the ComputeInputBuild list in positional order.
        let inputs_built: Vec<ComputeInputBuild<'_>> = input_producer
            .iter()
            .zip(node.inputs.iter())
            .map(|(prods, inp)| {
                if prods.is_empty() {
                    ComputeInputBuild::External {
                        tensor: inp.tensor,
                        region: inp.region,
                    }
                } else {
                    let writers: Vec<&SlotWritten> = prods
                        .iter()
                        .map(|pid| {
                            taken_tokens.get(pid).expect("token taken above")
                        })
                        .collect();
                    ComputeInputBuild::Computed(writers)
                }
            })
            .collect();

        if matches!(node.op, SubOp::AttnDecode { .. }) {
            // Slot lifecycle for AttnDecode lives OUTSIDE the loop bracket;
            // the Compute itself lives INSIDE.
            let h = builder.alloc_slot();
            let rb = builder.alloc_runtime_bound();
            let (mut inside, _var) = builder.open_loop(LoopBound::Runtime(rb));
            let w = inside.compute_to(node.id, h, &inputs_built);
            builder = inside.close_loop();
            written[nid as usize] = Some(w);
        } else {
            let h = builder.alloc_slot();
            let w = builder.compute_to(node.id, h, &inputs_built);
            written[nid as usize] = Some(w);
        }
        // Record this node as the writer of its output tensor (mirrors
        // predecessors() so subsequent nodes can find their producers).
        writers[node.output.tensor.0 as usize].push((nid, node.output.region));

        // Decrement each predecessor's consumer count; when zero, free.
        // (Iterates the taken_tokens map populated above for unique
        // computed-input producers; multiple positional refs to the
        // same producer count as a single consume here since we only
        // took the token once.)
        for (pid, ptoken) in taken_tokens {
            let remaining = &mut consumer_remaining[pid as usize];
            *remaining -= 1;
            if *remaining == 0 {
                builder.free_slot(ptoken);
            } else {
                written[pid as usize] = Some(ptoken);
            }
        }
    }

    // Free any leaf slots (no successors) that remain — typically the
    // graph result. Their consumer_remaining is 0 from the start.
    for slot in written.iter_mut() {
        if let Some(ptoken) = slot.take() {
            builder.free_slot(ptoken);
        }
    }

    let tape = builder.finish();
    validate_subtile_tape(&tape, graph)
        .expect("lower_dag_to_tape: produced invalid SubtileTape");
    tape
}

// ── Player skeleton ─────────────────────────────────────────────────

/// One step of the trivial host-tape player. The production player
/// runs at the per-target lowering output, not at this layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlayStep {
    Allocated(SlotId),
    Computed {
        node: SubtileId,
        writes: SlotId,
        inputs: ComputeInputs,
    },
    Freed(SlotId),
    LoopOpened(u32),
    LoopClosed(u32),
}

/// Trivial replay of the tape: walk the instructions in linear order,
/// emit one [`PlayStep`] per instruction. **No compute** — the
/// production target-specific player invokes
/// [`crate::subtile_ir::eval_node`] (or the GPU kernel).
pub fn play_skeleton(tape: &SubtileTape) -> Vec<PlayStep> {
    tape.instrs
        .iter()
        .map(|i| match i {
            Instr::AllocSlot { slot } => PlayStep::Allocated(*slot),
            Instr::Compute {
                node,
                writes,
                inputs,
            } => PlayStep::Computed {
                node: *node,
                writes: *writes,
                inputs: inputs.clone(),
            },
            Instr::FreeSlot { slot } => PlayStep::Freed(*slot),
            Instr::OpenLoop { var, .. } => PlayStep::LoopOpened(var.id),
            Instr::CloseLoop { var } => PlayStep::LoopClosed(var.id),
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subtile_ir::{
        EwKind, KvCacheLayout, KvCacheProducer, NeoX, Range, Region, SoftmaxStateId, SubOp,
        SubtileIR, SubtileNode, TensorId, TensorRegion, TensorShape,
    };

    /// A minimal SubtileIR: source[1,4] → silu → result[1,4].
    fn tiny_graph() -> SubtileIR<NeoX> {
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 1, cols: 4 },
        ];
        let nodes = vec![SubtileNode {
            id: SubtileId(0),
            op: SubOp::Elementwise(EwKind::Silu),
            inputs: vec![TensorRegion {
                tensor: TensorId(0),
                region: Region {
                    rows: Range::new(0, 1),
                    cols: Range::new(0, 4),
                },
            }],
            output: TensorRegion {
                tensor: TensorId(1),
                region: Region {
                    rows: Range::new(0, 1),
                    cols: Range::new(0, 4),
                },
            },
        }];
        SubtileIR {
            tensors,
            num_sources: 1,
            nodes,
            result: TensorId(1),
        }
    }

    fn silu_node(
        id: u32,
        in_t: TensorId,
        in_cols: Range,
        out_t: TensorId,
        out_cols: Range,
    ) -> SubtileNode<NeoX> {
        SubtileNode {
            id: SubtileId(id),
            op: SubOp::Elementwise(EwKind::Silu),
            inputs: vec![TensorRegion {
                tensor: in_t,
                region: Region {
                    rows: Range::new(0, 1),
                    cols: in_cols,
                },
            }],
            output: TensorRegion {
                tensor: out_t,
                region: Region {
                    rows: Range::new(0, 1),
                    cols: out_cols,
                },
            },
        }
    }

    /// 2-node chain: source → silu(0) → silu(1).
    fn chain_graph() -> SubtileIR<NeoX> {
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 1, cols: 4 },
        ];
        SubtileIR {
            tensors,
            num_sources: 1,
            nodes: vec![
                silu_node(0, TensorId(0), Range::new(0, 4), TensorId(1), Range::new(0, 4)),
                silu_node(1, TensorId(1), Range::new(0, 4), TensorId(2), Range::new(0, 4)),
            ],
            result: TensorId(2),
        }
    }

    fn mk_slot(id: u32) -> SlotId {
        SlotId {
            id,
            _seal: sealed::Seal(()),
        }
    }
    fn mk_loop_var(id: u32) -> LoopVarId {
        LoopVarId {
            id,
            _seal: sealed::Seal(()),
        }
    }

    // ── Builder shape ─────────────────────────────────────────────

    #[test]
    fn build_finish_round_trips() {
        let mut b = TapeBuilder::new();
        let h = b.alloc_slot();
        let w = b.compute_to(SubtileId(0), h, &[]);
        b.free_slot(w);
        let tape = b.finish();
        assert_eq!(tape.num_slots, 1);
        assert!(matches!(tape.instrs[0], Instr::AllocSlot { .. }));
        assert!(matches!(
            tape.instrs[1],
            Instr::Compute { node: SubtileId(0), .. }
        ));
        assert!(matches!(tape.instrs[2], Instr::FreeSlot { .. }));
    }

    #[test]
    fn loop_open_close_round_trips_with_compute_inside() {
        let mut b = TapeBuilder::new();
        let h = b.alloc_slot();
        let (mut inside, _var) = b.open_loop(LoopBound::Const(8));
        let w = inside.compute_to(SubtileId(0), h, &[]);
        let mut outer = inside.close_loop();
        outer.free_slot(w);
        let tape = outer.finish();
        assert_eq!(tape.num_loop_vars, 1);
        assert_eq!(tape.num_slots, 1);
        assert!(matches!(
            tape.instrs.last(),
            Some(Instr::FreeSlot { .. })
        ));
    }

    #[test]
    fn runtime_loop_bound_id_increments() {
        let mut b = TapeBuilder::new();
        let r0 = b.alloc_runtime_bound();
        let r1 = b.alloc_runtime_bound();
        assert_eq!(r0.index(), 0);
        assert_eq!(r1.index(), 1);
        let tape = b.finish();
        assert_eq!(tape.num_runtime_bounds, 2);
    }

    // ── Validator: well-formedness ────────────────────────────────

    #[test]
    fn validator_accepts_well_formed_tape() {
        let g = tiny_graph();
        let mut b = TapeBuilder::new();
        let h = b.alloc_slot();
        let w = b.compute_to(SubtileId(0), h, &[]);
        b.free_slot(w);
        let tape = b.finish();
        assert_eq!(validate_subtile_tape(&tape, &g), Ok(()));
    }

    #[test]
    fn validator_flags_unknown_node() {
        let g = tiny_graph();
        let tape = SubtileTape {
            instrs: vec![
                Instr::AllocSlot { slot: mk_slot(0) },
                Instr::Compute {
                    node: SubtileId(99),
                    writes: mk_slot(0),
                    inputs: ComputeInputs::Variadic(vec![]),
                },
                Instr::FreeSlot { slot: mk_slot(0) },
            ],
            num_slots: 1,
            num_loop_vars: 0,
            num_runtime_bounds: 0,
        };
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.iter().any(|e| matches!(e, ValidationError::UnknownNode { node } if node.0 == 99)),
            "want UnknownNode(99), got {err:?}"
        );
    }

    #[test]
    fn validator_flags_missing_compute() {
        let g = chain_graph();
        let mut b = TapeBuilder::new();
        let h = b.alloc_slot();
        let w = b.compute_to(SubtileId(0), h, &[]); // node 1 never Compute'd
        b.free_slot(w);
        let tape = b.finish();
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.contains(&ValidationError::MissingCompute { node: SubtileId(1) }),
            "want MissingCompute(1), got {err:?}"
        );
    }

    #[test]
    fn validator_flags_duplicate_compute() {
        let g = tiny_graph();
        let tape = SubtileTape {
            instrs: vec![
                Instr::AllocSlot { slot: mk_slot(0) },
                Instr::Compute {
                    node: SubtileId(0),
                    writes: mk_slot(0),
                    inputs: ComputeInputs::Variadic(vec![]),
                },
                Instr::Compute {
                    node: SubtileId(0),
                    writes: mk_slot(0),
                    inputs: ComputeInputs::Variadic(vec![]),
                },
                Instr::FreeSlot { slot: mk_slot(0) },
            ],
            num_slots: 1,
            num_loop_vars: 0,
            num_runtime_bounds: 0,
        };
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.contains(&ValidationError::DuplicateCompute { node: SubtileId(0) }),
            "want DuplicateCompute(0), got {err:?}"
        );
    }

    #[test]
    fn validator_flags_topo_order_violation() {
        let g = chain_graph();
        let tape = SubtileTape {
            instrs: vec![
                Instr::AllocSlot { slot: mk_slot(0) },
                Instr::AllocSlot { slot: mk_slot(1) },
                Instr::Compute {
                    node: SubtileId(1),
                    writes: mk_slot(1),
                    inputs: ComputeInputs::Variadic(vec![]),
                },
                Instr::Compute {
                    node: SubtileId(0),
                    writes: mk_slot(0),
                    inputs: ComputeInputs::Variadic(vec![]),
                },
                Instr::FreeSlot { slot: mk_slot(0) },
                Instr::FreeSlot { slot: mk_slot(1) },
            ],
            num_slots: 2,
            num_loop_vars: 0,
            num_runtime_bounds: 0,
        };
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.iter().any(|e| matches!(e, ValidationError::TopoOrderViolation { .. })),
            "want TopoOrderViolation, got {err:?}"
        );
    }

    // ── Validator: edge coverage (relational only) ────────────────
    //
    // Loop-balance and slot-lifecycle tests were deleted per audit
    // BLOCKER fix (`wewpteccb`): K5 + feedback_compile_time_or_garbage
    // require those invariants live in the `TapeBuilder<S>` typestate,
    // not in the runtime validator. With `SubtileTape::instrs` now
    // private and the typestate-redundant ValidationError variants
    // removed, those tests would be testing dead code.

    #[test]
    fn validator_flags_edge_mismatch_extra_read() {
        // tiny_graph: node 0's predecessors set is empty; if the tape
        // claims a read of node 0's own slot, that's an edge mismatch.
        let g = tiny_graph();
        let tape = SubtileTape {
            instrs: vec![
                Instr::AllocSlot { slot: mk_slot(0) },
                Instr::Compute {
                    node: SubtileId(0),
                    writes: mk_slot(0),
                    inputs: ComputeInputs::A1([ComputeInput::Computed(vec![mk_slot(0)])]), // self-read names node 0 as a pred — bogus
                },
                Instr::FreeSlot { slot: mk_slot(0) },
            ],
            num_slots: 1,
            num_loop_vars: 0,
            num_runtime_bounds: 0,
        };
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.iter().any(|e| matches!(e, ValidationError::EdgeMismatch { .. })),
            "want EdgeMismatch on self-read, got {err:?}"
        );
    }

    // ── lower_dag_to_tape ─────────────────────────────────────────

    #[test]
    fn lower_chain_threads_slots_through() {
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 1, cols: 4 },
        ];
        let nodes = vec![
            silu_node(0, TensorId(0), Range::new(0, 4), TensorId(1), Range::new(0, 4)),
            silu_node(1, TensorId(1), Range::new(0, 4), TensorId(2), Range::new(0, 4)),
            silu_node(2, TensorId(2), Range::new(0, 4), TensorId(3), Range::new(0, 4)),
        ];
        let g: SubtileIR<NeoX> = SubtileIR {
            tensors,
            num_sources: 1,
            nodes,
            result: TensorId(3),
        };
        let valid = crate::subtile_ir::ValidatedGraph::new(&g).unwrap();
        let tape = lower_dag_to_tape(&valid);
        // Each node: 1 alloc + 1 compute + 1 free (after last consumer)
        // The chain has 3 nodes and the result-slot is freed at end.
        assert_eq!(tape.num_slots, 3);
        assert_eq!(tape.num_loop_vars, 0);
        let alloc_count = tape
            .instrs
            .iter()
            .filter(|i| matches!(i, Instr::AllocSlot { .. }))
            .count();
        let compute_count = tape
            .instrs
            .iter()
            .filter(|i| matches!(i, Instr::Compute { .. }))
            .count();
        let free_count = tape
            .instrs
            .iter()
            .filter(|i| matches!(i, Instr::FreeSlot { .. }))
            .count();
        assert_eq!(alloc_count, 3);
        assert_eq!(compute_count, 3);
        assert_eq!(free_count, 3);
        // Validator already runs at the exit of lower_dag_to_tape.
    }

    #[test]
    fn lower_attn_decode_wraps_in_runtime_loop() {
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 4, cols: 4 },
            TensorShape { rows: 4, cols: 4 },
            TensorShape { rows: 1, cols: 4 },
        ];
        use crate::subtile_ir::TestShape1x4;
        let attn: SubtileNode<NeoX, TestShape1x4> = SubtileNode {
            id: SubtileId(0),
            op: SubOp::AttnDecode {
                num_q_heads: 1,
                num_kv_heads: 1,
                head_dim: 4,
                scale: 0.5,
                layout: KvCacheLayout::<TestShape1x4>::for_cache_tensor(TensorId(1)),
                producer: KvCacheProducer::pre_populated_ext(),
                softmax_state: SoftmaxStateId::new(0),
            },
            inputs: vec![
                TensorRegion {
                    tensor: TensorId(0),
                    region: Region {
                        rows: Range::new(0, 1),
                        cols: Range::new(0, 4),
                    },
                },
                TensorRegion {
                    tensor: TensorId(1),
                    region: Region {
                        rows: Range::new(0, 4),
                        cols: Range::new(0, 4),
                    },
                },
                TensorRegion {
                    tensor: TensorId(2),
                    region: Region {
                        rows: Range::new(0, 4),
                        cols: Range::new(0, 4),
                    },
                },
            ],
            output: TensorRegion {
                tensor: TensorId(3),
                region: Region {
                    rows: Range::new(0, 1),
                    cols: Range::new(0, 4),
                },
            },
        };
        let g: SubtileIR<NeoX, TestShape1x4> = SubtileIR {
            tensors,
            num_sources: 3,
            nodes: vec![attn],
            result: TensorId(3),
        };
        let valid = crate::subtile_ir::ValidatedGraph::new(&g).unwrap();
        let tape = lower_dag_to_tape(&valid);
        assert_eq!(tape.num_loop_vars, 1);
        assert_eq!(tape.num_runtime_bounds, 1);
        assert_eq!(tape.num_slots, 1);
        // Expected stream: AllocSlot, OpenLoop, Compute, CloseLoop, FreeSlot
        assert!(matches!(tape.instrs[0], Instr::AllocSlot { .. }));
        assert!(matches!(
            tape.instrs[1],
            Instr::OpenLoop { bound: LoopBound::Runtime(_), .. }
        ));
        assert!(matches!(tape.instrs[2], Instr::Compute { .. }));
        assert!(matches!(tape.instrs[3], Instr::CloseLoop { .. }));
        assert!(matches!(tape.instrs[4], Instr::FreeSlot { .. }));
    }

    #[test]
    fn lower_diamond_threads_multi_reader_correctly() {
        // Diamond: source → silu(0) → silu(1), silu(0) → silu(2), silu(1)+silu(2) → mul(3).
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 }, // source
            TensorShape { rows: 1, cols: 4 }, // silu(0) out
            TensorShape { rows: 1, cols: 4 }, // silu(1) out
            TensorShape { rows: 1, cols: 4 }, // silu(2) out
            TensorShape { rows: 1, cols: 4 }, // mul(3) out
        ];
        let mul_3 = SubtileNode::<NeoX> {
            id: SubtileId(3),
            op: SubOp::Elementwise(EwKind::Mul),
            inputs: vec![
                TensorRegion {
                    tensor: TensorId(2),
                    region: Region {
                        rows: Range::new(0, 1),
                        cols: Range::new(0, 4),
                    },
                },
                TensorRegion {
                    tensor: TensorId(3),
                    region: Region {
                        rows: Range::new(0, 1),
                        cols: Range::new(0, 4),
                    },
                },
            ],
            output: TensorRegion {
                tensor: TensorId(4),
                region: Region {
                    rows: Range::new(0, 1),
                    cols: Range::new(0, 4),
                },
            },
        };
        let g: SubtileIR<NeoX> = SubtileIR {
            tensors,
            num_sources: 1,
            nodes: vec![
                silu_node(0, TensorId(0), Range::new(0, 4), TensorId(1), Range::new(0, 4)),
                silu_node(1, TensorId(1), Range::new(0, 4), TensorId(2), Range::new(0, 4)),
                silu_node(2, TensorId(1), Range::new(0, 4), TensorId(3), Range::new(0, 4)),
                mul_3,
            ],
            result: TensorId(4),
        };
        let valid = crate::subtile_ir::ValidatedGraph::new(&g).unwrap();
        let tape = lower_dag_to_tape(&valid);
        assert_eq!(tape.num_slots, 4);
        // Validator already ran. silu(0)'s slot has two consumers
        // (silu(1) and silu(2)); it must be freed AFTER silu(2)'s
        // Compute, not after silu(1)'s.
        // Per plan §4 lines 199-200: no `_ =>` arms (yes, even in
        // tests). Enumerate every Instr variant explicitly.
        let frees: Vec<usize> = tape
            .instrs
            .iter()
            .enumerate()
            .filter_map(|(i, instr)| match instr {
                Instr::FreeSlot { slot } if slot.id == 0 => Some(i),
                Instr::FreeSlot { .. }
                | Instr::AllocSlot { .. }
                | Instr::Compute { .. }
                | Instr::OpenLoop { .. }
                | Instr::CloseLoop { .. } => None,
            })
            .collect();
        assert_eq!(frees.len(), 1);
        // Find silu(2)'s Compute index — slot 0 free should come after.
        let silu2_compute = tape
            .instrs
            .iter()
            .position(|instr| matches!(instr, Instr::Compute { node: SubtileId(2), .. }))
            .expect("silu(2) compute should exist");
        assert!(
            frees[0] > silu2_compute,
            "slot-0 free must come after silu(2)'s compute (last reader); got free@{} silu2@{}",
            frees[0],
            silu2_compute
        );
    }

    // ── Player skeleton ───────────────────────────────────────────

    #[test]
    fn play_skeleton_round_trips_every_instr() {
        let mut b = TapeBuilder::new();
        let h = b.alloc_slot();
        let w = b.compute_to(SubtileId(0), h, &[]);
        b.free_slot(w);
        let tape = b.finish();
        let steps = play_skeleton(&tape);
        assert_eq!(steps.len(), 3);
        assert!(matches!(steps[0], PlayStep::Allocated(_)));
        assert!(matches!(
            steps[1],
            PlayStep::Computed { node: SubtileId(0), .. }
        ));
        assert!(matches!(steps[2], PlayStep::Freed(_)));
    }
}
