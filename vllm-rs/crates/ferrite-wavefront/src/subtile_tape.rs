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
//! All target-specific concepts — execution-unit abstractions
//! (per-target lowering decision), memory-tier classification
//! (TkTape concept), visibility primitives (TkTape concept),
//! pipeline-state tracking (TkTape concept). The IR-level witnesses
//! (`KvCacheLayout`, `KvCacheProducer`, `RopeForm`, online-softmax
//! state) live on the SubtileIR `SubOp` variants; the lowering looks
//! them up by `SubtileId`.
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Instr {
    /// Mint a fresh slot id for a producer's output. Pairs (eventually)
    /// with one [`Instr::Compute`] writing this slot, and one
    /// [`Instr::FreeSlot`] retiring it.
    AllocSlot { slot: SlotId },
    /// Compute the named SubtileIR node, writing its output into `writes`
    /// and reading from `reads`. The node's input/output regions live on
    /// the SubtileIR; this instruction names node identity + the
    /// dataflow edges that surface.
    Compute {
        node: SubtileId,
        writes: SlotId,
        reads: Vec<SlotId>,
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
#[derive(Clone, Debug)]
pub struct SubtileTape {
    pub instrs: Vec<Instr>,
    /// High-water mark of allocated slot ids (`0..num_slots`). Live-slot
    /// count at any program point is recoverable by replay.
    pub num_slots: u32,
    pub num_loop_vars: u32,
    pub num_runtime_bounds: u32,
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
fn push_compute(
    instrs: &mut Vec<Instr>,
    node: SubtileId,
    write: SlotHandle,
    reads: &[&SlotWritten],
) -> SlotWritten {
    let writes_id = write.slot;
    let read_ids: Vec<SlotId> = reads.iter().map(|w| w.slot).collect();
    instrs.push(Instr::Compute {
        node,
        writes: writes_id,
        reads: read_ids,
    });
    SlotWritten {
        slot: writes_id,
        _seal: sealed::Seal(()),
    }
}

impl TapeBuilder<state::Outside> {
    /// Compute `node`, writing its output into `write` (consuming the
    /// `SlotHandle`) and reading from `reads` (borrowing each
    /// `SlotWritten`). Returns the `SlotWritten` token for downstream
    /// consumers.
    pub fn compute_to(
        &mut self,
        node: SubtileId,
        write: SlotHandle,
        reads: &[&SlotWritten],
    ) -> SlotWritten {
        push_compute(&mut self.instrs, node, write, reads)
    }
}

impl TapeBuilder<state::InsideLoop> {
    /// Compute `node` inside the active loop body. Same shape as the
    /// `Outside` impl; both states share the workload instruction.
    pub fn compute_to(
        &mut self,
        node: SubtileId,
        write: SlotHandle,
        reads: &[&SlotWritten],
    ) -> SlotWritten {
        push_compute(&mut self.instrs, node, write, reads)
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

/// One class of runtime-detectable invariant violation. The typestate
/// already catches the structural ones (orphan loop brackets, reading
/// an unwritten slot, double-free) at compile time when the builder is
/// used. The runtime validator defends against hand-built tapes that
/// bypass the typestate by mutating `instrs` directly.
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
    /// A second `OpenLoop` started while another was still open.
    NestedLoop { outer_var: u32, inner_var: u32 },
    /// A `CloseLoop` appeared with no matching `OpenLoop`.
    UnmatchedCloseLoop { close_var: u32 },
    /// `OpenLoop` and matching `CloseLoop` disagree on `var`.
    MismatchedLoopVar { open_var: u32, close_var: u32 },
    /// Tape ended with a still-open loop.
    UnclosedLoop { var: u32 },
    /// A `Compute` writes into a slot that was never `AllocSlot`'d.
    WriteUnallocatedSlot { slot: u32, at: usize },
    /// A `Compute` reads from a slot that was never written.
    ReadBeforeWrite { slot: u32, at: usize },
    /// A second `Compute` writes the same slot (single-writer violated).
    DoubleWrite { slot: u32, at: usize },
    /// A `Compute` or `FreeSlot` touches a slot already freed.
    UseAfterFree { slot: u32, at: usize },
    /// `FreeSlot` retires a slot that was never `AllocSlot`'d.
    FreeUnallocatedSlot { slot: u32, at: usize },
    /// A second `FreeSlot` retires the same slot (double-free).
    DoubleFree { slot: u32, at: usize },
    /// An `AllocSlot` mints an id outside `0..num_slots` (hand-built tape).
    AllocSlotOutOfRange { slot: u32, at: usize, num_slots: u32 },
    /// A `Compute` reads from / writes to a slot id outside `0..num_slots`.
    SlotIdOutOfRange { slot: u32, at: usize, num_slots: u32 },
    /// At end-of-tape, a slot is allocated but never freed (leak).
    SlotNeverFreed { slot: u32 },
    /// A `Compute`'s reads on the SubtileIR DAG don't match its
    /// predecessor set — one DAG edge is missing from the slot-read
    /// list, or an extra read names a node that is not a predecessor.
    EdgeMismatch {
        node: SubtileId,
        expected_preds: Vec<SubtileId>,
        actual_read_writers: Vec<SubtileId>,
    },
}

/// Per-slot lifecycle state tracked by the validator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotState {
    /// `AllocSlot` seen, no `Compute` has written it yet.
    Allocated,
    /// `Compute { writes }` seen; readers OK; `FreeSlot` not yet seen.
    Written,
    /// `FreeSlot` seen; further use is `UseAfterFree`.
    Freed,
}

/// Runtime validator. Six checks:
///
/// 1. **Compute well-formedness** — every SubtileIR node is `Compute`d
///    exactly once; no `Compute` references an out-of-range node.
/// 2. **Topo order** — adjacent Computes appear in strictly ascending
///    `SubtileId` order (the SubtileIR is already ascending-id topo;
///    the tape is a valid linearization).
/// 3. **Loop balance** — every `OpenLoop` matches a `CloseLoop` with
///    the same `LoopVarId`; no nesting; no unclosed loops.
/// 4. **Slot lifecycle** — every slot id transits Allocated → Written
///    → Freed exactly once; no read-before-write; no use-after-free;
///    no double-write; no double-free; no orphan allocs.
/// 5. **Slot id range** — every slot id (`AllocSlot`/`Compute`/`FreeSlot`)
///    falls in `0..num_slots`.
/// 6. **Edge coverage** — every `Compute`'s `reads` equals (set-wise)
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
    check_loop_balance(tape, &mut errors);
    check_slot_lifecycle_and_edges(tape, graph, &mut errors);
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

fn check_loop_balance(tape: &SubtileTape, errors: &mut Vec<ValidationError>) {
    let mut open: Option<u32> = None;
    for instr in &tape.instrs {
        match instr {
            Instr::OpenLoop { var, .. } => {
                if let Some(outer_var) = open {
                    errors.push(ValidationError::NestedLoop {
                        outer_var,
                        inner_var: var.id,
                    });
                }
                open = Some(var.id);
            }
            Instr::CloseLoop { var } => match open {
                None => errors.push(ValidationError::UnmatchedCloseLoop {
                    close_var: var.id,
                }),
                Some(open_var) => {
                    if open_var != var.id {
                        errors.push(ValidationError::MismatchedLoopVar {
                            open_var,
                            close_var: var.id,
                        });
                    }
                    open = None;
                }
            },
            Instr::AllocSlot { .. } | Instr::FreeSlot { .. } | Instr::Compute { .. } => {}
        }
    }
    if let Some(var) = open {
        errors.push(ValidationError::UnclosedLoop { var });
    }
}

fn check_slot_lifecycle_and_edges<F: crate::subtile_ir::RopeForm, K: crate::subtile_ir::KvCacheShape>(
    tape: &SubtileTape,
    graph: &crate::subtile_ir::SubtileIR<F, K>,
    errors: &mut Vec<ValidationError>,
) {
    let preds = crate::subtile_ir::predecessors(graph);
    let n_slots = tape.num_slots;
    let mut phase: BTreeMap<u32, SlotState> = BTreeMap::new();
    // For each slot, the SubtileId of the node whose Compute wrote it
    // (the slot's producer). Used to map a read-slot back to its
    // predecessor for edge coverage.
    let mut writer_of: BTreeMap<u32, SubtileId> = BTreeMap::new();
    let in_range = |s: u32| s < n_slots;
    for (i, instr) in tape.instrs.iter().enumerate() {
        match instr {
            Instr::AllocSlot { slot } => {
                let s = slot.id;
                if !in_range(s) {
                    errors.push(ValidationError::AllocSlotOutOfRange {
                        slot: s,
                        at: i,
                        num_slots: n_slots,
                    });
                    continue;
                }
                phase.insert(s, SlotState::Allocated);
            }
            Instr::Compute {
                node,
                writes,
                reads,
            } => {
                let w = writes.id;
                if !in_range(w) {
                    errors.push(ValidationError::SlotIdOutOfRange {
                        slot: w,
                        at: i,
                        num_slots: n_slots,
                    });
                } else {
                    match phase.get(&w).copied() {
                        None => errors.push(ValidationError::WriteUnallocatedSlot {
                            slot: w,
                            at: i,
                        }),
                        Some(SlotState::Allocated) => {
                            phase.insert(w, SlotState::Written);
                            writer_of.insert(w, *node);
                        }
                        Some(SlotState::Written) => errors.push(ValidationError::DoubleWrite {
                            slot: w,
                            at: i,
                        }),
                        Some(SlotState::Freed) => errors.push(ValidationError::UseAfterFree {
                            slot: w,
                            at: i,
                        }),
                    }
                }
                let mut read_writers: Vec<SubtileId> = Vec::with_capacity(reads.len());
                for r in reads {
                    let s = r.id;
                    if !in_range(s) {
                        errors.push(ValidationError::SlotIdOutOfRange {
                            slot: s,
                            at: i,
                            num_slots: n_slots,
                        });
                        continue;
                    }
                    match phase.get(&s).copied() {
                        None | Some(SlotState::Allocated) => {
                            errors.push(ValidationError::ReadBeforeWrite {
                                slot: s,
                                at: i,
                            });
                        }
                        Some(SlotState::Freed) => {
                            errors.push(ValidationError::UseAfterFree {
                                slot: s,
                                at: i,
                            });
                        }
                        Some(SlotState::Written) => {
                            if let Some(wn) = writer_of.get(&s).copied() {
                                read_writers.push(wn);
                            }
                        }
                    }
                }
                let n_idx = node.0 as usize;
                if n_idx < preds.len() {
                    let mut expected = preds[n_idx].clone();
                    expected.sort();
                    let mut actual = read_writers.clone();
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
            Instr::FreeSlot { slot } => {
                let s = slot.id;
                if !in_range(s) {
                    errors.push(ValidationError::SlotIdOutOfRange {
                        slot: s,
                        at: i,
                        num_slots: n_slots,
                    });
                    continue;
                }
                match phase.get(&s).copied() {
                    None => errors.push(ValidationError::FreeUnallocatedSlot {
                        slot: s,
                        at: i,
                    }),
                    Some(SlotState::Allocated) => errors.push(ValidationError::ReadBeforeWrite {
                        slot: s,
                        at: i,
                    }),
                    Some(SlotState::Written) => {
                        phase.insert(s, SlotState::Freed);
                    }
                    Some(SlotState::Freed) => {
                        errors.push(ValidationError::DoubleFree { slot: s, at: i })
                    }
                }
            }
            Instr::OpenLoop { .. } | Instr::CloseLoop { .. } => {}
        }
    }
    for (&s, &p) in &phase {
        if !matches!(p, SlotState::Freed) {
            errors.push(ValidationError::SlotNeverFreed { slot: s });
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
/// a runtime-bounded count (the KV-sweep over `seq_len` pages); the
/// AttnDecode `Compute` lives inside the loop, the slot lifecycle
/// (alloc / free) lives outside.
///
/// **No worker assignment, no fence, no memory class.** Per-target
/// realization happens at the per-target lowering (`lower_tape_to_tk`).
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
    for node in &graph.nodes {
        let nid = node.id.0;
        let pred_ids = &preds[nid as usize];
        let mut pred_tokens: Vec<(u32, SlotWritten)> = Vec::with_capacity(pred_ids.len());
        for p in pred_ids {
            // SAFETY (algorithmic): preds[nid] lists predecessors of
            // ascending-id node nid; each predecessor p has p.0 < nid
            // and was written via `written[p.0] = Some(...)` on its
            // own iteration (ascending walk). consumer_remaining
            // re-inserts the token until last consumer — by the time
            // we observe `None` here, the function would have already
            // freed the slot, which would mean we're visiting node N
            // after N's last consumer, which violates the topo order
            // the SubtileIR's ascending-id invariant guarantees.
            let token = written[p.0 as usize]
                .take()
                .unwrap_or_else(|| unreachable!(
                    "lower_dag_to_tape: predecessor {} of node {} has no live SlotWritten — \
                     consumer-count walk disagrees with predecessors list (loop invariant)",
                    p.0, nid
                ));
            pred_tokens.push((p.0, token));
        }
        let read_refs: Vec<&SlotWritten> =
            pred_tokens.iter().map(|(_, w)| w).collect();

        if matches!(node.op, SubOp::AttnDecode { .. }) {
            // Slot lifecycle for AttnDecode lives OUTSIDE the loop bracket;
            // the Compute itself lives INSIDE.
            let h = builder.alloc_slot();
            let rb = builder.alloc_runtime_bound();
            let (mut inside, _var) = builder.open_loop(LoopBound::Runtime(rb));
            let w = inside.compute_to(node.id, h, &read_refs);
            builder = inside.close_loop();
            written[nid as usize] = Some(w);
        } else {
            let h = builder.alloc_slot();
            let w = builder.compute_to(node.id, h, &read_refs);
            written[nid as usize] = Some(w);
        }

        // Decrement each predecessor's consumer count; when zero, free.
        for (pid, ptoken) in pred_tokens {
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
        reads: Vec<SlotId>,
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
                reads,
            } => PlayStep::Computed {
                node: *node,
                writes: *writes,
                reads: reads.clone(),
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
                    reads: vec![],
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
                    reads: vec![],
                },
                Instr::Compute {
                    node: SubtileId(0),
                    writes: mk_slot(0),
                    reads: vec![],
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
                    reads: vec![],
                },
                Instr::Compute {
                    node: SubtileId(0),
                    writes: mk_slot(0),
                    reads: vec![],
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

    // ── Validator: loop balance ───────────────────────────────────

    #[test]
    fn validator_flags_unclosed_loop() {
        let g = tiny_graph();
        let tape = SubtileTape {
            instrs: vec![
                Instr::OpenLoop {
                    var: mk_loop_var(0),
                    bound: LoopBound::Const(4),
                },
                Instr::AllocSlot { slot: mk_slot(0) },
                Instr::Compute {
                    node: SubtileId(0),
                    writes: mk_slot(0),
                    reads: vec![],
                },
                Instr::FreeSlot { slot: mk_slot(0) },
                // no CloseLoop
            ],
            num_slots: 1,
            num_loop_vars: 1,
            num_runtime_bounds: 0,
        };
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.contains(&ValidationError::UnclosedLoop { var: 0 }),
            "want UnclosedLoop, got {err:?}"
        );
    }

    #[test]
    fn validator_flags_mismatched_loop_var() {
        let g = tiny_graph();
        let tape = SubtileTape {
            instrs: vec![
                Instr::OpenLoop {
                    var: mk_loop_var(0),
                    bound: LoopBound::Const(4),
                },
                Instr::AllocSlot { slot: mk_slot(0) },
                Instr::Compute {
                    node: SubtileId(0),
                    writes: mk_slot(0),
                    reads: vec![],
                },
                Instr::FreeSlot { slot: mk_slot(0) },
                Instr::CloseLoop { var: mk_loop_var(99) },
            ],
            num_slots: 1,
            num_loop_vars: 100,
            num_runtime_bounds: 0,
        };
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.iter().any(|e| matches!(
                e,
                ValidationError::MismatchedLoopVar { open_var: 0, close_var: 99 }
            )),
            "want MismatchedLoopVar, got {err:?}"
        );
    }

    #[test]
    fn validator_flags_unmatched_close_loop() {
        let g = tiny_graph();
        let tape = SubtileTape {
            instrs: vec![
                Instr::AllocSlot { slot: mk_slot(0) },
                Instr::Compute {
                    node: SubtileId(0),
                    writes: mk_slot(0),
                    reads: vec![],
                },
                Instr::FreeSlot { slot: mk_slot(0) },
                Instr::CloseLoop { var: mk_loop_var(0) },
            ],
            num_slots: 1,
            num_loop_vars: 1,
            num_runtime_bounds: 0,
        };
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.contains(&ValidationError::UnmatchedCloseLoop { close_var: 0 }),
            "want UnmatchedCloseLoop, got {err:?}"
        );
    }

    // ── Validator: slot lifecycle ─────────────────────────────────

    #[test]
    fn validator_flags_write_unallocated_slot() {
        let g = tiny_graph();
        let tape = SubtileTape {
            instrs: vec![
                Instr::Compute {
                    node: SubtileId(0),
                    writes: mk_slot(0),
                    reads: vec![],
                },
                Instr::FreeSlot { slot: mk_slot(0) },
            ],
            num_slots: 1,
            num_loop_vars: 0,
            num_runtime_bounds: 0,
        };
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.iter().any(|e| matches!(e, ValidationError::WriteUnallocatedSlot { slot: 0, .. })),
            "want WriteUnallocatedSlot(0), got {err:?}"
        );
    }

    #[test]
    fn validator_flags_read_before_write() {
        let g = chain_graph();
        let tape = SubtileTape {
            instrs: vec![
                Instr::AllocSlot { slot: mk_slot(0) },
                Instr::AllocSlot { slot: mk_slot(1) },
                Instr::Compute {
                    node: SubtileId(0),
                    writes: mk_slot(0),
                    reads: vec![mk_slot(1)],
                },
                Instr::Compute {
                    node: SubtileId(1),
                    writes: mk_slot(1),
                    reads: vec![],
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
            err.iter().any(|e| matches!(e, ValidationError::ReadBeforeWrite { slot: 1, .. })),
            "want ReadBeforeWrite(1), got {err:?}"
        );
    }

    #[test]
    fn validator_flags_double_write() {
        let g = tiny_graph();
        let tape = SubtileTape {
            instrs: vec![
                Instr::AllocSlot { slot: mk_slot(0) },
                Instr::Compute {
                    node: SubtileId(0),
                    writes: mk_slot(0),
                    reads: vec![],
                },
                Instr::Compute {
                    node: SubtileId(0),
                    writes: mk_slot(0),
                    reads: vec![],
                },
                Instr::FreeSlot { slot: mk_slot(0) },
            ],
            num_slots: 1,
            num_loop_vars: 0,
            num_runtime_bounds: 0,
        };
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.iter().any(|e| matches!(e, ValidationError::DoubleWrite { slot: 0, .. })),
            "want DoubleWrite(0), got {err:?}"
        );
    }

    #[test]
    fn validator_flags_use_after_free() {
        let g = chain_graph();
        let tape = SubtileTape {
            instrs: vec![
                Instr::AllocSlot { slot: mk_slot(0) },
                Instr::Compute {
                    node: SubtileId(0),
                    writes: mk_slot(0),
                    reads: vec![],
                },
                Instr::FreeSlot { slot: mk_slot(0) },
                Instr::AllocSlot { slot: mk_slot(1) },
                Instr::Compute {
                    node: SubtileId(1),
                    writes: mk_slot(1),
                    reads: vec![mk_slot(0)],
                },
                Instr::FreeSlot { slot: mk_slot(1) },
            ],
            num_slots: 2,
            num_loop_vars: 0,
            num_runtime_bounds: 0,
        };
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.iter().any(|e| matches!(e, ValidationError::UseAfterFree { slot: 0, .. })),
            "want UseAfterFree(0), got {err:?}"
        );
    }

    #[test]
    fn validator_flags_double_free() {
        let g = tiny_graph();
        let tape = SubtileTape {
            instrs: vec![
                Instr::AllocSlot { slot: mk_slot(0) },
                Instr::Compute {
                    node: SubtileId(0),
                    writes: mk_slot(0),
                    reads: vec![],
                },
                Instr::FreeSlot { slot: mk_slot(0) },
                Instr::FreeSlot { slot: mk_slot(0) },
            ],
            num_slots: 1,
            num_loop_vars: 0,
            num_runtime_bounds: 0,
        };
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.iter().any(|e| matches!(e, ValidationError::DoubleFree { slot: 0, .. })),
            "want DoubleFree(0), got {err:?}"
        );
    }

    #[test]
    fn validator_flags_slot_never_freed() {
        let g = tiny_graph();
        let tape = SubtileTape {
            instrs: vec![
                Instr::AllocSlot { slot: mk_slot(0) },
                Instr::Compute {
                    node: SubtileId(0),
                    writes: mk_slot(0),
                    reads: vec![],
                },
                // no FreeSlot
            ],
            num_slots: 1,
            num_loop_vars: 0,
            num_runtime_bounds: 0,
        };
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.iter().any(|e| matches!(e, ValidationError::SlotNeverFreed { slot: 0 })),
            "want SlotNeverFreed(0), got {err:?}"
        );
    }

    #[test]
    fn validator_flags_edge_mismatch_extra_read() {
        // tiny_graph: node 0's predecessors set is empty; if the tape
        // claims a read, that's an edge mismatch.
        let g = tiny_graph();
        let tape = SubtileTape {
            instrs: vec![
                Instr::AllocSlot { slot: mk_slot(0) },
                Instr::Compute {
                    node: SubtileId(0),
                    writes: mk_slot(0),
                    reads: vec![mk_slot(0)], // self-read; bogus
                },
                Instr::FreeSlot { slot: mk_slot(0) },
            ],
            num_slots: 1,
            num_loop_vars: 0,
            num_runtime_bounds: 0,
        };
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        // A self-read also trips DoubleWrite (writes seen first), so
        // primarily check the edge / use-after-* family fires.
        assert!(
            err.iter().any(|e| matches!(
                e,
                ValidationError::EdgeMismatch { .. }
                    | ValidationError::DoubleWrite { .. }
                    | ValidationError::UseAfterFree { .. }
                    | ValidationError::ReadBeforeWrite { .. }
            )),
            "want some slot/edge error on self-read, got {err:?}"
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
        let frees: Vec<usize> = tape
            .instrs
            .iter()
            .enumerate()
            .filter_map(|(i, instr)| match instr {
                Instr::FreeSlot { slot } if slot.id == 0 => Some(i),
                _ => None,
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
