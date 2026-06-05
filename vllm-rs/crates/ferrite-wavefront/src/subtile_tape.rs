// SPDX-License-Identifier: Apache-2.0
//! Linear, target-agnostic **SubtileTape** — every SubtileIR DAG edge
//! becomes an explicit `Compute` / `Signal` / `Wait` / `Fence` / `Route`
//! / `OpenLoop` / `CloseLoop` instruction.
//!
//! Sealed handles + a typestate [`TapeBuilder<S>`] make orphan handles,
//! mismatched-id Wait/Signal pairs, and unmatched loop brackets
//! **structurally impossible** at compile time. The runtime
//! [`validate_subtile_tape`] catches the rest (deadlock cycles, data
//! races, missing fences, orphan signal/wait).
//!
//! This commit ships the IR + typestate + validator skeleton (plan §4
//! commit 3, additive — no consumers). The SubtileIR → SubtileTape
//! lowering walker (`lower_dag_to_tape`) lands in plan §4 commit 5;
//! the production validator + dataflow-aware play follow in commit 5b.
//!
//! **Source of truth** for the constraint set + policy defaults
//! (keep-in-smem first, chain-local placement, coarse-loops-only):
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

// ── Memory routing classification ───────────────────────────────────

/// What memory class a producer's output lives in. Set by
/// [`Instr::Route`] per producer node; the validator + the TK lowering
/// both consult it.
///
/// **Default policy (per `SUBTILE_TAPE_CONSTRAINTS.md` §6):** every
/// producer output is `Shmem` unless the routing analysis returned
/// `External` for it. Gmem is the *fallback*, not the default — every
/// gmem path on a producer that could have been shmem-carried is a
/// bandwidth round-trip the megakernel pays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryClass {
    /// Stays in shared memory; a single consumer carries the producer's
    /// smem page forward via mbar handshake.
    Shmem,
    /// Drained to global memory; consumers TMA-load it back.
    Gmem,
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Instr {
    /// Compute the named SubtileIR node on `worker`. The node's input /
    /// output regions are looked up in the SubtileIR; this instruction
    /// only names the node identity.
    Compute { worker: WorkerId, node: SubtileId },
    /// Set one-shot `barrier` on `worker` (producer side of a cross-
    /// worker edge).
    Signal { worker: WorkerId, barrier: BarrierId },
    /// Block on one-shot `barrier` on `worker` (consumer side).
    Wait { worker: WorkerId, barrier: BarrierId },
    /// Memory-hazard fence on `worker`. Required between a producer's
    /// `Signal` and any cross-worker consumer's `Wait` when the edge
    /// crossed gmem (the producer's [`Instr::Route`] class is
    /// [`MemoryClass::Gmem`]). The TK lowering picks the actual
    /// primitive (threadfence_device, etc.).
    Fence { worker: WorkerId },
    /// Routes the output of `node` through `class`. **Per producer
    /// output** (one Route per Compute, not per consumer): the
    /// producer's physical output buffer is one piece of memory all
    /// consumers see.
    Route { node: SubtileId, class: MemoryClass },
    /// Open a runtime-bounded loop on `worker` over `bound` iterations
    /// (the AttnDecode KV-sweep). Body holds Computes + Fences + Routes;
    /// no nested loops, no Signal/Wait inside (would reorder vs the
    /// iteration count). The matching [`Instr::CloseLoop`] takes the
    /// same `var`.
    OpenLoop {
        worker: WorkerId,
        var: LoopVarId,
        bound: LoopBound,
    },
    CloseLoop { worker: WorkerId, var: LoopVarId },
}

impl Instr {
    /// The worker this instruction runs on, if any. `Route` is per-
    /// producer-output, so it doesn't have its own worker tag (it
    /// follows from the matching `Compute`).
    pub fn worker(&self) -> Option<WorkerId> {
        match self {
            Instr::Compute { worker, .. }
            | Instr::Signal { worker, .. }
            | Instr::Wait { worker, .. }
            | Instr::Fence { worker }
            | Instr::OpenLoop { worker, .. }
            | Instr::CloseLoop { worker, .. } => Some(*worker),
            Instr::Route { .. } => None,
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
    /// An `OpenLoop` is in flight. Body operations (`compute`, `fence`,
    /// `route`) and `close_loop` are available here; cross-worker sync
    /// is forbidden (would reorder vs the iteration count).
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

    pub fn fence(&mut self, worker: WorkerId) -> &mut Self {
        self.instrs.push(Instr::Fence { worker });
        self
    }

    pub fn route(&mut self, node: SubtileId, class: MemoryClass) -> &mut Self {
        self.instrs.push(Instr::Route { node, class });
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

    pub fn fence(&mut self, worker: WorkerId) -> &mut Self {
        self.instrs.push(Instr::Fence { worker });
        self
    }

    pub fn route(&mut self, node: SubtileId, class: MemoryClass) -> &mut Self {
        self.instrs.push(Instr::Route { node, class });
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
    /// Cross-worker (Signal, Wait) pair whose producer wrote through
    /// gmem (Route.class == Gmem) lacks a `Fence` between producer
    /// `Compute` and `Signal`.
    MissingFence {
        producer_worker: u32,
        consumer_worker: u32,
        barrier: u32,
    },
    /// A tensor with cross-worker write→read where the reader worker
    /// has no `Wait`/`Fence` covering the writer's update.
    DataRace {
        tensor: TensorId,
        writer_worker: u32,
        reader_worker: u32,
    },
    /// `Compute` references a node id not present in the SubtileIR.
    UnknownNode { node: SubtileId },
    /// `Route` has no matching `Compute` for the same `node` on any
    /// worker.
    OrphanRoute { node: SubtileId },
    /// A node was `Compute`d but never `Route`d (every producer output
    /// must declare its memory class).
    UnroutedCompute { node: SubtileId },
}

/// Runtime validator. Discharges the four invariant classes the
/// typestate cannot see (per `SUBTILE_TAPE_CONSTRAINTS.md` §4):
/// orphan signal/wait, deadlock cycle, missing fence, data race —
/// plus a couple of basic well-formedness checks (unknown nodes,
/// orphan routes).
///
/// **Skeleton-grade for commit 3.** Each individual check is the
/// minimal sound check; the production-grade implementations land
/// with the lowering walker in plan §4 commit 5b.
pub fn validate_subtile_tape(
    tape: &SubtileTape,
    graph: &SubtileIR,
) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();
    check_node_refs(tape, graph, &mut errors);
    check_routes(tape, &mut errors);
    check_signal_wait_pairs(tape, &mut errors);
    check_deadlock_cycle(tape, &mut errors);
    check_missing_fence(tape, &mut errors);
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
        let node = match instr {
            Instr::Compute { node, .. } | Instr::Route { node, .. } => *node,
            _ => continue,
        };
        if node.0 >= n_nodes {
            errors.push(ValidationError::UnknownNode { node });
        }
    }
}

fn check_routes(tape: &SubtileTape, errors: &mut Vec<ValidationError>) {
    let mut computed: BTreeMap<SubtileId, ()> = BTreeMap::new();
    let mut routed: BTreeMap<SubtileId, ()> = BTreeMap::new();
    for instr in &tape.instrs {
        match instr {
            Instr::Compute { node, .. } => {
                computed.insert(*node, ());
            }
            Instr::Route { node, .. } => {
                routed.insert(*node, ());
            }
            _ => {}
        }
    }
    for node in routed.keys() {
        if !computed.contains_key(node) {
            errors.push(ValidationError::OrphanRoute { node: *node });
        }
    }
    for node in computed.keys() {
        if !routed.contains_key(node) {
            errors.push(ValidationError::UnroutedCompute { node: *node });
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

fn check_missing_fence(tape: &SubtileTape, errors: &mut Vec<ValidationError>) {
    // For each cross-worker barrier, scan EVERY prior Compute on the
    // producer worker (not just the most recent). If ANY of them was
    // Routed Gmem, there must be a Fence on the producer worker
    // somewhere between that Compute and the Signal — otherwise the
    // gmem write may not be visible to the consumer.
    let mut route_class: BTreeMap<SubtileId, MemoryClass> = BTreeMap::new();
    for instr in &tape.instrs {
        if let Instr::Route { node, class } = instr {
            route_class.insert(*node, *class);
        }
    }
    let mut signal_at: BTreeMap<u32, (usize, u32)> = BTreeMap::new(); // barrier → (idx, worker)
    let mut wait_worker: BTreeMap<u32, u32> = BTreeMap::new(); // barrier → consumer worker
    for (i, instr) in tape.instrs.iter().enumerate() {
        match instr {
            Instr::Signal { worker, barrier } => {
                signal_at.insert(barrier.id, (i, worker.id));
            }
            Instr::Wait { worker, barrier } => {
                wait_worker.insert(barrier.id, worker.id);
            }
            _ => {}
        }
    }
    for (b, (sig_idx, prod_w)) in &signal_at {
        let cons_w = match wait_worker.get(b) {
            Some(w) if *w != *prod_w => *w,
            _ => continue,
        };
        // Walk all instructions before the Signal on the producer worker
        // in order; track whether the most-recent Fence covers every
        // Gmem-routed Compute that follows it. If any Gmem Compute lands
        // after the latest Fence (or before any Fence), the hazard is
        // unguarded — report MissingFence.
        let mut latest_fence_idx: Option<usize> = None;
        let mut hazard_idx: Option<usize> = None;
        for (j, instr) in tape.instrs[..*sig_idx].iter().enumerate() {
            match instr {
                Instr::Fence { worker } if worker.id == *prod_w => {
                    latest_fence_idx = Some(j);
                    hazard_idx = None;
                }
                Instr::Compute { worker, node } if worker.id == *prod_w => {
                    if route_class.get(node).copied() == Some(MemoryClass::Gmem)
                        && latest_fence_idx.is_none_or(|fi| fi < j)
                    {
                        hazard_idx = Some(j);
                    }
                }
                _ => {}
            }
        }
        if hazard_idx.is_some() {
            errors.push(ValidationError::MissingFence {
                producer_worker: *prod_w,
                consumer_worker: cons_w,
                barrier: *b,
            });
        }
    }
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
    Fenced(u32),
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
        .filter_map(|i| match i {
            Instr::Compute { node, .. } => Some(PlayStep::Computed(*node)),
            Instr::Signal { barrier, .. } => Some(PlayStep::Signaled(barrier.id)),
            Instr::Wait { barrier, .. } => Some(PlayStep::Waited(barrier.id)),
            Instr::Fence { worker } => Some(PlayStep::Fenced(worker.id)),
            Instr::OpenLoop { var, .. } => Some(PlayStep::LoopOpened(var.id)),
            Instr::CloseLoop { var, .. } => Some(PlayStep::LoopClosed(var.id)),
            Instr::Route { .. } => None,
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
        b.compute(w0, SubtileId(0)).route(SubtileId(0), MemoryClass::Shmem);
        let tape = b.finish();
        assert_eq!(tape.num_workers, 2);
        assert_eq!(tape.instrs.len(), 2);
    }

    #[test]
    fn validator_accepts_well_formed_tape() {
        let g = tiny_graph();
        let mut b = TapeBuilder::new(1);
        let w0 = b.worker(0);
        b.compute(w0, SubtileId(0))
            .route(SubtileId(0), MemoryClass::Shmem);
        let tape = b.finish();
        assert_eq!(validate_subtile_tape(&tape, &g), Ok(()));
    }

    #[test]
    fn validator_flags_unrouted_compute() {
        let g = tiny_graph();
        let mut b = TapeBuilder::new(1);
        let w0 = b.worker(0);
        b.compute(w0, SubtileId(0));
        let tape = b.finish();
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.contains(&ValidationError::UnroutedCompute {
                node: SubtileId(0),
            }),
            "want UnroutedCompute, got {err:?}"
        );
    }

    #[test]
    fn validator_flags_orphan_route() {
        let g = tiny_graph();
        let mut b = TapeBuilder::new(1);
        b.route(SubtileId(0), MemoryClass::Shmem);
        let tape = b.finish();
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.contains(&ValidationError::OrphanRoute {
                node: SubtileId(0),
            }),
            "want OrphanRoute, got {err:?}"
        );
    }

    #[test]
    fn validator_flags_unknown_node() {
        let g = tiny_graph();
        let mut b = TapeBuilder::new(1);
        let w0 = b.worker(0);
        b.compute(w0, SubtileId(99))
            .route(SubtileId(99), MemoryClass::Shmem);
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
            .route(SubtileId(0), MemoryClass::Shmem)
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
            .route(SubtileId(0), MemoryClass::Shmem)
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
            .route(SubtileId(0), MemoryClass::Shmem)
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
    fn validator_flags_missing_gmem_fence() {
        // w0 Computes node 0, Routes Gmem, Signals; w1 Waits — but no
        // Fence on w0 between Compute and Signal.
        let g = tiny_graph();
        let mut b = TapeBuilder::new(2);
        let w0 = b.worker(0);
        let w1 = b.worker(1);
        let bar = b.alloc_barrier();
        b.compute(w0, SubtileId(0))
            .route(SubtileId(0), MemoryClass::Gmem)
            .signal(w0, bar)
            .wait(w1, bar);
        let tape = b.finish();
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.iter().any(|e| matches!(
                e,
                ValidationError::MissingFence { producer_worker: 0, consumer_worker: 1, .. }
            )),
            "want MissingFence(0→1), got {err:?}"
        );
    }

    #[test]
    fn validator_flags_unfenced_gmem_when_later_compute_is_shmem() {
        // Earlier Gmem-routed Compute followed by Shmem-routed Compute on
        // the same worker, then Signal — the Gmem hazard must still be
        // flagged. (Previous skeleton checked only the most-recent
        // Compute and silently passed.) Build a tiny graph with TWO
        // op-output nodes so this scenario is structurally possible.
        let g = {
            let tensors = vec![
                TensorShape { rows: 1, cols: 4 },
                TensorShape { rows: 1, cols: 4 },
                TensorShape { rows: 1, cols: 4 },
            ];
            let nodes = vec![
                SubtileNode {
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
                },
                SubtileNode {
                    id: SubtileId(1),
                    op: SubOp::Elementwise(EwKind::Silu),
                    inputs: vec![TensorRegion {
                        tensor: TensorId(1),
                        region: Region {
                            rows: Range::new(0, 1),
                            cols: Range::new(0, 4),
                        },
                    }],
                    output: TensorRegion {
                        tensor: TensorId(2),
                        region: Region {
                            rows: Range::new(0, 1),
                            cols: Range::new(0, 4),
                        },
                    },
                },
            ];
            SubtileIR {
                tensors,
                num_sources: 1,
                nodes,
                result: TensorId(2),
            }
        };
        let mut b = TapeBuilder::new(2);
        let w0 = b.worker(0);
        let w1 = b.worker(1);
        let bar = b.alloc_barrier();
        b.compute(w0, SubtileId(0))
            .route(SubtileId(0), MemoryClass::Gmem) // hazard, no fence
            .compute(w0, SubtileId(1))
            .route(SubtileId(1), MemoryClass::Shmem)
            .signal(w0, bar)
            .wait(w1, bar);
        let tape = b.finish();
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.iter().any(|e| matches!(
                e,
                ValidationError::MissingFence { producer_worker: 0, consumer_worker: 1, .. }
            )),
            "earlier Gmem hazard must be detected even when the latest Compute is Shmem; got {err:?}"
        );
    }

    #[test]
    fn validator_accepts_gmem_with_fence() {
        let g = tiny_graph();
        let mut b = TapeBuilder::new(2);
        let w0 = b.worker(0);
        let w1 = b.worker(1);
        let bar = b.alloc_barrier();
        b.compute(w0, SubtileId(0))
            .route(SubtileId(0), MemoryClass::Gmem)
            .fence(w0)
            .signal(w0, bar)
            .wait(w1, bar);
        let tape = b.finish();
        assert_eq!(validate_subtile_tape(&tape, &g), Ok(()));
    }

    #[test]
    fn loop_open_close_round_trips() {
        let mut b = TapeBuilder::new(1);
        let w0 = b.worker(0);
        let (mut inside, _var) = b.open_loop(w0, LoopBound::Const(8));
        inside.compute(w0, SubtileId(0)).fence(w0);
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
    fn play_skeleton_filters_routes() {
        let mut b = TapeBuilder::new(1);
        let w0 = b.worker(0);
        b.compute(w0, SubtileId(0))
            .route(SubtileId(0), MemoryClass::Shmem)
            .fence(w0);
        let tape = b.finish();
        let steps = play_skeleton(&tape);
        // Compute(0) + Fence(0); Route is internal-only.
        assert_eq!(
            steps,
            vec![PlayStep::Computed(SubtileId(0)), PlayStep::Fenced(0),]
        );
    }
}
