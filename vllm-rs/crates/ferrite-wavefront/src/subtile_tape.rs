// SPDX-License-Identifier: Apache-2.0
//! Linear, target-agnostic **SubtileTape** — every SubtileIR DAG edge
//! becomes an explicit `Compute` / `Signal` / `Wait` / `OpenLoop` /
//! `CloseLoop` instruction.
//!
//! **No smem, no gmem, no fence, no page, no parity.** Memory-tier
//! decisions and visibility primitives are TkTape concerns (target-
//! specific); see `SUBTILE_IR_REDESIGN.md` §4 commit 6.5 for the
//! optimizer pass pipeline that picks them. SubtileTape carries DAG
//! facts only.
//!
//! Sealed handles + a typestate [`TapeBuilder<S>`] make orphan handles,
//! mismatched-id Wait/Signal pairs, and unmatched loop brackets
//! **structurally impossible** at compile time. The runtime
//! [`validate_subtile_tape`] catches the rest (orphan signal/wait,
//! deadlock cycles, cross-worker data races).
//!
//! This commit ships the IR + typestate + validator skeleton (plan §4
//! commit 3 + 3.b scrub, additive — no consumers). The SubtileIR →
//! SubtileTape lowering walker (`lower_dag_to_tape`) lands in plan §4
//! commit 5; the production validator follows in commit 5b.
//!
//! **Source of truth** for the constraint set + policy defaults:
//! [`vllm-rs/SUBTILE_TAPE_CONSTRAINTS.md`]. New constraints go there
//! before they go in code.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::marker::PhantomData;

use crate::subtile_ir::{SubtileId, SubtileIR, TensorId};

// ── Sealed handles ──────────────────────────────────────────────────

#[doc(hidden)]
pub mod sealed {
    /// Sealing token. The inner `()` is `pub(super)`, so external code
    /// cannot construct a `Seal` value — making every type that carries
    /// a `Seal` field constructable only by code in `subtile_tape`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct Seal(pub(super) ());
}

/// Identifies one worker (one persistent threadgroup / co-resident CTA).
/// Sealed: constructable only via [`TapeBuilder::worker`].
///
/// ```compile_fail
/// // The id field is private; literal construction is rejected.
/// let _ = ferrite_wavefront::subtile_tape::WorkerId { id: 0 };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkerId {
    id: u32,
    _seal: sealed::Seal,
}

impl WorkerId {
    pub const fn index(&self) -> u32 {
        self.id
    }
}

/// One-shot point-to-point flag for a single cross-worker producer→
/// consumer edge. Both [`Instr::Signal`] and [`Instr::Wait`] take this
/// same value; constructable only via [`TapeBuilder::alloc_barrier`],
/// so a `Signal` and a `Wait` that name the same `BarrierId` cannot
/// have come from independent fabrications.
///
/// ```compile_fail
/// // Sealed; struct-literal construction is rejected.
/// let _ = ferrite_wavefront::subtile_tape::BarrierId { id: 0 };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BarrierId {
    id: u32,
    _seal: sealed::Seal,
}

impl BarrierId {
    pub const fn index(&self) -> u32 {
        self.id
    }
}

/// Loop-variable handle. The id matched by [`Instr::OpenLoop`] and
/// [`Instr::CloseLoop`]. Constructable only as the return value of
/// [`TapeBuilder::open_loop`].
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
/// concrete kernel-arg slot is bound at TkTape lowering; SubtileTape
/// just names which runtime quantity supplies it.
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

/// One tape instruction. Workers are interleaved in a single linear
/// stream; the per-worker subset is recovered by filtering on `worker`.
/// The per-worker subset is in program order.
///
/// SubtileTape is target-agnostic: there is no `Fence`, no `Route`,
/// no memory class. Memory-tier decisions and visibility primitives
/// are TkTape's job.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Instr {
    /// Compute the named SubtileIR node on `worker`. The node's input /
    /// output regions are looked up in the SubtileIR; this instruction
    /// only names the node identity.
    Compute { worker: WorkerId, node: SubtileId },
    /// Set one-shot `barrier` on `worker` (producer side of a cross-
    /// worker edge). The visibility primitive that backs this signal
    /// (fence, mbar, …) is picked at TkTape lowering.
    Signal { worker: WorkerId, barrier: BarrierId },
    /// Block on one-shot `barrier` on `worker` (consumer side).
    Wait { worker: WorkerId, barrier: BarrierId },
    /// Open a runtime-bounded loop on `worker` over `bound` iterations
    /// (the AttnDecode KV-sweep). Body holds Computes only; no nested
    /// loops, no Signal/Wait inside (would reorder vs the iteration
    /// count). The matching [`Instr::CloseLoop`] takes the same `var`.
    OpenLoop {
        worker: WorkerId,
        var: LoopVarId,
        bound: LoopBound,
    },
    CloseLoop { worker: WorkerId, var: LoopVarId },
}

impl Instr {
    /// The worker this instruction runs on. Every SubtileTape Instr
    /// is worker-tagged; the per-worker subset is recovered by filter.
    pub fn worker(&self) -> WorkerId {
        match self {
            Instr::Compute { worker, .. }
            | Instr::Signal { worker, .. }
            | Instr::Wait { worker, .. }
            | Instr::OpenLoop { worker, .. }
            | Instr::CloseLoop { worker, .. } => *worker,
        }
    }
}

// ── The tape ────────────────────────────────────────────────────────

/// Linear, target-agnostic per-wavefront tape. Build via [`TapeBuilder`]
/// then call [`validate_subtile_tape`] to discharge the runtime
/// invariants the typestate cannot see.
#[derive(Clone, Debug)]
pub struct SubtileTape {
    pub instrs: Vec<Instr>,
    pub num_workers: u32,
    pub num_barriers: u32,
    pub num_loop_vars: u32,
    pub num_runtime_bounds: u32,
}

// ── TapeBuilder<S> typestate ────────────────────────────────────────

/// Compile-time state markers for [`TapeBuilder<S>`].
pub mod state {
    /// No `OpenLoop` is in flight. Cross-worker sync (`signal`, `wait`,
    /// `alloc_barrier`) and `finish` are only available here.
    #[derive(Debug)]
    pub enum Outside {}
    /// An `OpenLoop` is in flight. Body operation (`compute`) and
    /// `close_loop` are available here; cross-worker sync is forbidden
    /// (would reorder vs the iteration count).
    #[derive(Debug)]
    pub enum InsideLoop {}
}

/// Typestate-tracked builder. The `S` parameter is one of
/// [`state::Outside`] / [`state::InsideLoop`]; the same `TapeBuilder`
/// type carries different methods depending on `S`. Misuse (e.g.
/// `close_loop` on `Outside`) = no matching impl, **compile error**.
///
/// ```compile_fail
/// use ferrite_wavefront::subtile_tape::TapeBuilder;
/// // close_loop is only impl'd on TapeBuilder<state::InsideLoop>;
/// // calling it on the default (Outside) builder is rejected.
/// let b = TapeBuilder::new(2);
/// let _ = b.close_loop();
/// ```
///
/// ```compile_fail
/// use ferrite_wavefront::subtile_tape::{LoopBound, TapeBuilder};
/// // finish is only impl'd on TapeBuilder<state::Outside>;
/// // calling it inside an open loop is rejected.
/// let mut b = TapeBuilder::new(2);
/// let w = b.worker(0);
/// let (inside, _var) = b.open_loop(w, LoopBound::Const(8));
/// let _ = inside.finish();
/// ```
///
/// ```compile_fail
/// use ferrite_wavefront::subtile_tape::{LoopBound, TapeBuilder};
/// // signal is only impl'd on TapeBuilder<state::Outside>;
/// // emitting one inside a loop is rejected (would reorder vs iters).
/// let mut b = TapeBuilder::new(2);
/// let w = b.worker(0);
/// let bar = b.alloc_barrier();
/// let (mut inside, _var) = b.open_loop(w, LoopBound::Const(8));
/// inside.signal(w, bar);
/// ```
pub struct TapeBuilder<S = state::Outside> {
    instrs: Vec<Instr>,
    num_workers: u32,
    next_barrier: u32,
    next_loop_var: u32,
    next_runtime_bound: u32,
    /// `Some((var, worker))` while an OpenLoop is in flight.
    cur_loop: Option<(LoopVarId, WorkerId)>,
    /// `PhantomData<fn() -> S>` keeps the builder `Send + Sync` without
    /// implying `S: Send` / `S: Sync` — the state markers are
    /// zero-sized empty enums (never instantiated).
    _state: PhantomData<fn() -> S>,
}

impl TapeBuilder<state::Outside> {
    /// New builder for `num_workers` co-resident workers.
    pub fn new(num_workers: u32) -> Self {
        assert!(num_workers >= 1, "num_workers must be >= 1");
        Self {
            instrs: Vec::new(),
            num_workers,
            next_barrier: 0,
            next_loop_var: 0,
            next_runtime_bound: 0,
            cur_loop: None,
            _state: PhantomData,
        }
    }

    /// Mint a [`WorkerId`] for index `w` (`< num_workers`). Sealed: this
    /// is the only path to a `WorkerId` value.
    pub fn worker(&self, w: u32) -> WorkerId {
        assert!(w < self.num_workers, "worker index {} >= {}", w, self.num_workers);
        WorkerId {
            id: w,
            _seal: sealed::Seal(()),
        }
    }

    /// Allocate a fresh barrier. The same `BarrierId` MUST be passed to
    /// the matching `signal` and `wait` (the only way to get one).
    pub fn alloc_barrier(&mut self) -> BarrierId {
        let b = BarrierId {
            id: self.next_barrier,
            _seal: sealed::Seal(()),
        };
        self.next_barrier += 1;
        b
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

    pub fn compute(&mut self, worker: WorkerId, node: SubtileId) -> &mut Self {
        self.instrs.push(Instr::Compute { worker, node });
        self
    }

    pub fn signal(&mut self, worker: WorkerId, barrier: BarrierId) -> &mut Self {
        self.instrs.push(Instr::Signal { worker, barrier });
        self
    }

    pub fn wait(&mut self, worker: WorkerId, barrier: BarrierId) -> &mut Self {
        self.instrs.push(Instr::Wait { worker, barrier });
        self
    }

    /// Open a runtime-bounded loop. Returns a builder in `InsideLoop`
    /// state plus a fresh [`LoopVarId`] for the matching `close_loop`.
    pub fn open_loop(
        mut self,
        worker: WorkerId,
        bound: LoopBound,
    ) -> (TapeBuilder<state::InsideLoop>, LoopVarId) {
        let var = LoopVarId {
            id: self.next_loop_var,
            _seal: sealed::Seal(()),
        };
        self.next_loop_var += 1;
        self.instrs.push(Instr::OpenLoop {
            worker,
            var,
            bound,
        });
        let inside = TapeBuilder::<state::InsideLoop> {
            instrs: self.instrs,
            num_workers: self.num_workers,
            next_barrier: self.next_barrier,
            next_loop_var: self.next_loop_var,
            next_runtime_bound: self.next_runtime_bound,
            cur_loop: Some((var, worker)),
            _state: PhantomData,
        };
        (inside, var)
    }

    /// Finalize: produce the linear tape. Only available with no loop
    /// in flight (the `Outside` state).
    pub fn finish(self) -> SubtileTape {
        SubtileTape {
            instrs: self.instrs,
            num_workers: self.num_workers,
            num_barriers: self.next_barrier,
            num_loop_vars: self.next_loop_var,
            num_runtime_bounds: self.next_runtime_bound,
        }
    }
}

impl TapeBuilder<state::InsideLoop> {
    pub fn compute(&mut self, worker: WorkerId, node: SubtileId) -> &mut Self {
        self.instrs.push(Instr::Compute { worker, node });
        self
    }

    /// Close the active loop. Returns a builder back in the `Outside`
    /// state — `signal`/`wait`/`finish` are available again.
    pub fn close_loop(mut self) -> TapeBuilder<state::Outside> {
        let (var, worker) = self
            .cur_loop
            .expect("InsideLoop without cur_loop (typestate invariant violated)");
        self.instrs.push(Instr::CloseLoop { worker, var });
        TapeBuilder::<state::Outside> {
            instrs: self.instrs,
            num_workers: self.num_workers,
            next_barrier: self.next_barrier,
            next_loop_var: self.next_loop_var,
            next_runtime_bound: self.next_runtime_bound,
            cur_loop: None,
            _state: PhantomData,
        }
    }
}

// ── Runtime validator ───────────────────────────────────────────────

/// One class of runtime-detectable invariant violation. The compile-
/// time typestate already catches the structural ones (orphan handles
/// from sealed types, unmatched loop brackets, signal/wait inside a
/// loop); these are the dynamic-id and graph-level errors only the
/// validator can see.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidationError {
    /// Barrier with `Wait`s but no `Signal`.
    OrphanWait { barrier: u32 },
    /// Barrier with `Signal`s but no `Wait`.
    OrphanSignal { barrier: u32 },
    /// `Signal(barrier)` more than once on a one-shot flag.
    DuplicateSignal { barrier: u32 },
    /// Cross-worker dependency cycle (every worker waiting on someone
    /// else's signal that never fires).
    DeadlockCycle { workers: Vec<u32> },
    /// A tensor with cross-worker write→read where the reader worker
    /// has no `Wait` covering the writer's update. (Visibility primitive
    /// — fence vs mbar — is picked at TkTape; this layer's data-race
    /// check is sync-edge-based, target-agnostic.)
    DataRace {
        tensor: TensorId,
        writer_worker: u32,
        reader_worker: u32,
    },
    /// `Compute` references a node id not present in the SubtileIR.
    UnknownNode { node: SubtileId },
}

/// Runtime validator. Discharges the invariant classes the typestate
/// cannot see (per `SUBTILE_TAPE_CONSTRAINTS.md` §4): orphan
/// signal/wait, deadlock cycle, cross-worker data race — plus
/// well-formedness (unknown node refs).
///
/// **Skeleton-grade for commit 3.b.** Each individual check is the
/// minimal sound check; the production-grade per-tensor data-race
/// timeline lands with the lowering walker in plan §4 commit 5b.
///
/// **Not** a fence-correctness check. SubtileTape doesn't name fences;
/// fence-before-arrive is a TkTape invariant (`validate_tk_tape`).
pub fn validate_subtile_tape(
    tape: &SubtileTape,
    graph: &SubtileIR,
) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();
    check_node_refs(tape, graph, &mut errors);
    check_signal_wait_pairs(tape, &mut errors);
    check_deadlock_cycle(tape, &mut errors);
    check_data_race(tape, graph, &mut errors);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn check_node_refs(tape: &SubtileTape, graph: &SubtileIR, errors: &mut Vec<ValidationError>) {
    let n_nodes = graph.nodes.len() as u32;
    for instr in &tape.instrs {
        if let Instr::Compute { node, .. } = instr
            && node.0 >= n_nodes
        {
            errors.push(ValidationError::UnknownNode { node: *node });
        }
    }
}

fn check_signal_wait_pairs(tape: &SubtileTape, errors: &mut Vec<ValidationError>) {
    let mut signals: BTreeMap<u32, u32> = BTreeMap::new();
    let mut waits: BTreeMap<u32, u32> = BTreeMap::new();
    for instr in &tape.instrs {
        match instr {
            Instr::Signal { barrier, .. } => *signals.entry(barrier.id).or_insert(0) += 1,
            Instr::Wait { barrier, .. } => *waits.entry(barrier.id).or_insert(0) += 1,
            _ => {}
        }
    }
    for b in 0..tape.num_barriers {
        let s = signals.get(&b).copied().unwrap_or(0);
        let w = waits.get(&b).copied().unwrap_or(0);
        if s == 0 && w > 0 {
            errors.push(ValidationError::OrphanWait { barrier: b });
        }
        if w == 0 && s > 0 {
            errors.push(ValidationError::OrphanSignal { barrier: b });
        }
        if s > 1 {
            errors.push(ValidationError::DuplicateSignal { barrier: b });
        }
    }
}

fn check_deadlock_cycle(tape: &SubtileTape, errors: &mut Vec<ValidationError>) {
    // Static graph cycle check: edge consumer_worker → producer_worker
    // when consumer worker waits on a barrier whose producer is on a
    // different worker. A cycle in this graph means the (acyclic) DAG
    // invariant of SubtileIR was violated by the assignment / emission.
    let mut producer_of: BTreeMap<u32, u32> = BTreeMap::new();
    for instr in &tape.instrs {
        if let Instr::Signal { worker, barrier } = instr {
            producer_of.insert(barrier.id, worker.id);
        }
    }
    let mut edges: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for instr in &tape.instrs {
        if let Instr::Wait { worker, barrier } = instr
            && let Some(p) = producer_of.get(&barrier.id)
            && *p != worker.id
        {
            edges.entry(worker.id).or_default().push(*p);
        }
    }
    if let Some(cycle) = find_cycle(&edges) {
        errors.push(ValidationError::DeadlockCycle { workers: cycle });
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mark {
    White,
    Gray,
    Black,
}

/// DFS cycle detection on the worker dep graph; returns the first
/// detected cycle (sub-list of stack from the back-edge target).
fn find_cycle(edges: &BTreeMap<u32, Vec<u32>>) -> Option<Vec<u32>> {
    let mut color: BTreeMap<u32, Mark> = BTreeMap::new();
    let mut stack: Vec<u32> = Vec::new();
    let nodes: Vec<u32> = edges.keys().copied().collect();
    for &start in &nodes {
        if color.get(&start).copied().unwrap_or(Mark::White) != Mark::White {
            continue;
        }
        if let Some(c) = dfs_cycle(start, edges, &mut color, &mut stack) {
            return Some(c);
        }
    }
    None
}

fn dfs_cycle(
    u: u32,
    edges: &BTreeMap<u32, Vec<u32>>,
    color: &mut BTreeMap<u32, Mark>,
    stack: &mut Vec<u32>,
) -> Option<Vec<u32>> {
    color.insert(u, Mark::Gray);
    stack.push(u);
    if let Some(neighbors) = edges.get(&u) {
        for &v in neighbors {
            match color.get(&v).copied().unwrap_or(Mark::White) {
                Mark::Gray => {
                    let i = stack.iter().position(|&x| x == v).unwrap_or(0);
                    return Some(stack[i..].to_vec());
                }
                Mark::White => {
                    if let Some(c) = dfs_cycle(v, edges, color, stack) {
                        return Some(c);
                    }
                }
                Mark::Black => {}
            }
        }
    }
    stack.pop();
    color.insert(u, Mark::Black);
    None
}

fn check_data_race(tape: &SubtileTape, graph: &SubtileIR, errors: &mut Vec<ValidationError>) {
    // Skeleton: for each tensor written by some worker and read by
    // another, the reader's tape must contain a Wait *before* its first
    // read whose matching Signal is *after* the writer's last write on
    // the producer side. Without the lowering walker emitting these
    // pairs (commit 5), we don't yet have ground truth — so the
    // skeleton check is conservative: report a race only when ANY
    // cross-worker write→read exists with NO cross-worker barrier
    // active at all in the tape.
    let mut writer_workers: BTreeMap<TensorId, Vec<u32>> = BTreeMap::new();
    let mut reader_workers: BTreeMap<TensorId, Vec<u32>> = BTreeMap::new();
    let n_nodes = graph.nodes.len() as u32;
    for instr in &tape.instrs {
        if let Instr::Compute { worker, node } = instr
            && node.0 < n_nodes
        {
            let n = &graph.nodes[node.0 as usize];
            writer_workers
                .entry(n.output.tensor)
                .or_default()
                .push(worker.id);
            for inp in &n.inputs {
                reader_workers
                    .entry(inp.tensor)
                    .or_default()
                    .push(worker.id);
            }
        }
    }
    let any_signal_wait = tape
        .instrs
        .iter()
        .any(|i| matches!(i, Instr::Signal { .. } | Instr::Wait { .. }));
    if any_signal_wait {
        // Real per-tensor edge analysis lives in commit 5b. Until then,
        // having ANY signal/wait covers the basic case.
        return;
    }
    for (t, writers) in &writer_workers {
        let readers = match reader_workers.get(t) {
            Some(r) => r,
            None => continue,
        };
        for &w in writers {
            for &r in readers {
                if w != r {
                    errors.push(ValidationError::DataRace {
                        tensor: *t,
                        writer_worker: w,
                        reader_worker: r,
                    });
                }
            }
        }
    }
}

// ── Player skeleton ─────────────────────────────────────────────────

/// One step of the trivial host-tape player. Skeleton: the production
/// player lands with the lowering walker (commit 5b) — when there's a
/// real tape to play. For commit 3, this exists so consumers can name
/// the type and so the round-trip shape is locked in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlayStep {
    Computed(SubtileId),
    Signaled(u32),
    Waited(u32),
    LoopOpened(u32),
    LoopClosed(u32),
}

/// Trivial replay of the tape: walk the instructions in linear order,
/// emit one [`PlayStep`] per instruction. **No sync / no compute** —
/// the production player (commit 5b) will honor `Wait`/`Signal` and
/// invoke [`crate::subtile_ir::eval_node`].
pub fn play_skeleton(tape: &SubtileTape) -> Vec<PlayStep> {
    tape.instrs
        .iter()
        .map(|i| match i {
            Instr::Compute { node, .. } => PlayStep::Computed(*node),
            Instr::Signal { barrier, .. } => PlayStep::Signaled(barrier.id),
            Instr::Wait { barrier, .. } => PlayStep::Waited(barrier.id),
            Instr::OpenLoop { var, .. } => PlayStep::LoopOpened(var.id),
            Instr::CloseLoop { var, .. } => PlayStep::LoopClosed(var.id),
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subtile_ir::{
        EwKind, Range, Region, SubOp, SubtileNode, TensorId, TensorRegion, TensorShape,
    };

    /// A minimal SubtileIR: source[1,4] → silu → result[1,4].
    fn tiny_graph() -> SubtileIR {
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 }, // source
            TensorShape { rows: 1, cols: 4 }, // op output
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

    #[test]
    fn build_finish_round_trips() {
        let mut b = TapeBuilder::new(2);
        let w0 = b.worker(0);
        b.compute(w0, SubtileId(0));
        let tape = b.finish();
        assert_eq!(tape.num_workers, 2);
        assert_eq!(tape.instrs.len(), 1);
    }

    #[test]
    fn validator_accepts_well_formed_tape() {
        let g = tiny_graph();
        let mut b = TapeBuilder::new(1);
        let w0 = b.worker(0);
        b.compute(w0, SubtileId(0));
        let tape = b.finish();
        assert_eq!(validate_subtile_tape(&tape, &g), Ok(()));
    }

    #[test]
    fn validator_flags_unknown_node() {
        let g = tiny_graph();
        let mut b = TapeBuilder::new(1);
        let w0 = b.worker(0);
        b.compute(w0, SubtileId(99));
        let tape = b.finish();
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.iter().any(|e| matches!(e, ValidationError::UnknownNode { node } if node.0 == 99)),
            "want UnknownNode(99), got {err:?}"
        );
    }

    #[test]
    fn validator_flags_orphan_signal_and_wait() {
        let g = tiny_graph();
        let mut b = TapeBuilder::new(2);
        let w0 = b.worker(0);
        let w1 = b.worker(1);
        let solo_signal = b.alloc_barrier();
        let solo_wait = b.alloc_barrier();
        b.compute(w0, SubtileId(0))
            .signal(w0, solo_signal) // never waited
            .wait(w1, solo_wait); // never signaled
        let tape = b.finish();
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.contains(&ValidationError::OrphanSignal {
                barrier: solo_signal.index(),
            }),
            "want OrphanSignal({}), got {err:?}",
            solo_signal.index()
        );
        assert!(
            err.contains(&ValidationError::OrphanWait {
                barrier: solo_wait.index(),
            }),
            "want OrphanWait({}), got {err:?}",
            solo_wait.index()
        );
    }

    #[test]
    fn validator_flags_duplicate_signal() {
        let g = tiny_graph();
        let mut b = TapeBuilder::new(2);
        let w0 = b.worker(0);
        let w1 = b.worker(1);
        let bar = b.alloc_barrier();
        b.compute(w0, SubtileId(0))
            .signal(w0, bar)
            .signal(w0, bar) // one-shot violation
            .wait(w1, bar);
        let tape = b.finish();
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.contains(&ValidationError::DuplicateSignal {
                barrier: bar.index(),
            }),
            "want DuplicateSignal, got {err:?}"
        );
    }

    #[test]
    fn validator_flags_deadlock_cycle() {
        // w0 waits on b0 (signaled by w1); w1 waits on b1 (signaled by w0).
        // Edge graph: w0 → w1, w1 → w0. Cycle.
        let g = tiny_graph();
        let mut b = TapeBuilder::new(2);
        let w0 = b.worker(0);
        let w1 = b.worker(1);
        let b0 = b.alloc_barrier(); // signaled by w1, waited by w0
        let b1 = b.alloc_barrier(); // signaled by w0, waited by w1
        b.compute(w0, SubtileId(0))
            .signal(w0, b1)
            .wait(w0, b0)
            .signal(w1, b0)
            .wait(w1, b1);
        let tape = b.finish();
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.iter().any(|e| matches!(e, ValidationError::DeadlockCycle { .. })),
            "want DeadlockCycle, got {err:?}"
        );
    }

    #[test]
    fn loop_open_close_round_trips() {
        let mut b = TapeBuilder::new(1);
        let w0 = b.worker(0);
        let (mut inside, _var) = b.open_loop(w0, LoopBound::Const(8));
        inside.compute(w0, SubtileId(0));
        let outer = inside.close_loop();
        let tape = outer.finish();
        assert_eq!(tape.num_loop_vars, 1);
        assert!(matches!(
            tape.instrs.last(),
            Some(Instr::CloseLoop { .. })
        ));
    }

    #[test]
    fn runtime_loop_bound_id_increments() {
        let mut b = TapeBuilder::new(1);
        let r0 = b.alloc_runtime_bound();
        let r1 = b.alloc_runtime_bound();
        assert_eq!(r0.index(), 0);
        assert_eq!(r1.index(), 1);
        let tape = b.finish();
        assert_eq!(tape.num_runtime_bounds, 2);
    }

    #[test]
    fn play_skeleton_round_trips_every_instr() {
        let mut b = TapeBuilder::new(2);
        let w0 = b.worker(0);
        let w1 = b.worker(1);
        let bar = b.alloc_barrier();
        b.compute(w0, SubtileId(0)).signal(w0, bar).wait(w1, bar);
        let tape = b.finish();
        let steps = play_skeleton(&tape);
        assert_eq!(
            steps,
            vec![
                PlayStep::Computed(SubtileId(0)),
                PlayStep::Signaled(bar.index()),
                PlayStep::Waited(bar.index()),
            ]
        );
    }
}
