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

use crate::subtile_ir::{SubtileId, TensorId};

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

// ── Lowering: SubtileIR → SubtileTape ───────────────────────────────

/// Lower a [`crate::subtile_ir::SubtileIR`] DAG to a linear
/// [`SubtileTape`] in one deterministic pass. Plan §4 commit 5:
///
/// 1. **Validate** the SubtileIR's structural invariants
///    (`crate::subtile_ir::validate`).
/// 2. **Assign workers** with the slice-index rule (chain-local
///    placement, per `SUBTILE_TAPE_CONSTRAINTS.md` §6): worker
///    `(node.output.region.cols.start / unit) % num_workers`. The
///    decode chains preserve column slices, so this co-locates
///    `q→rope→attn→o-partial` and `gate/up→silu·mul→down-partial`
///    on one worker.
/// 3. **Topo-emit** in ascending `SubtileId` order. For each node:
///    - For every cross-worker producer, emit a `Wait` for that
///      producer's barrier (deduped per consumer).
///    - Emit the `Compute`. `SubOp::AttnDecode` wraps in an
///      `OpenLoop`/`CloseLoop` over a runtime-bounded count (the
///      KV-sweep over `seq_len` pages — bound is a fresh
///      [`RuntimeBoundId`]).
///    - If the node has any cross-worker consumer, emit a `Signal`
///      on the producer's worker.
///
/// **No `Route`, no `Fence`, no memory-class decisions** —
/// SubtileTape is target-agnostic. The conservative all-gmem
/// expansion (TMA store + fence + LoadAsync + PageBarrierWait) is
/// `lower_tape_to_tk`'s job; the shmem-promotion / fence-elimination
/// optimizations are TkTape passes (plan §4 commit 6.5).
///
/// `unit` is the column-tiling granularity the graph was lowered at
/// (typically `head_dim`, so head-structured ops and column-tiled
/// ops align). `unit == 0` is treated as `1`.
///
/// The returned `SubtileTape` has been validated against `graph` via
/// [`validate_subtile_tape`]; callers can assume well-formedness.
pub fn lower_dag_to_tape<F: crate::subtile_ir::RopeForm>(
    graph: &crate::subtile_ir::SubtileIR<F>,
    num_workers: u32,
    unit: u32,
) -> SubtileTape {
    use crate::subtile_ir::{SubOp, predecessors, validate};

    validate(graph).expect("lower_dag_to_tape: invalid SubtileIR");
    let n = graph.nodes.len();
    let p = num_workers.max(1);
    let u = unit.max(1);

    let worker_of: Vec<u32> = graph
        .nodes
        .iter()
        .map(|node| (node.output.region.cols.start / u) % p)
        .collect();

    let preds = predecessors(graph);

    // A producer needs a barrier iff some consumer lives on another worker.
    let mut needs_signal = vec![false; n];
    for (cid, plist) in preds.iter().enumerate() {
        let cw = worker_of[cid];
        for prod in plist {
            if worker_of[prod.0 as usize] != cw {
                needs_signal[prod.0 as usize] = true;
            }
        }
    }

    let mut builder = TapeBuilder::new(p);
    let mut barrier_of: Vec<Option<BarrierId>> = vec![None; n];
    for (i, &needs) in needs_signal.iter().enumerate() {
        if needs {
            barrier_of[i] = Some(builder.alloc_barrier());
        }
    }

    for node in &graph.nodes {
        let id = node.id.0 as usize;
        let w_idx = worker_of[id];
        let w = builder.worker(w_idx);

        // Wait for each distinct cross-worker producer's barrier.
        let mut waited: Vec<u32> = Vec::new();
        for prod in &preds[id] {
            let pid = prod.0 as usize;
            if worker_of[pid] != w_idx {
                let bar = barrier_of[pid]
                    .expect("cross-worker producer must have an allocated barrier");
                if !waited.contains(&bar.index()) {
                    waited.push(bar.index());
                    builder.wait(w, bar);
                }
            }
        }

        // Emit the Compute. AttnDecode wraps in a runtime-bounded loop —
        // the KV sweep over seq_len pages (a runtime quantity at decode).
        if matches!(node.op, SubOp::AttnDecode { .. }) {
            let rb = builder.alloc_runtime_bound();
            let (mut inside, _var) = builder.open_loop(w, LoopBound::Runtime(rb));
            inside.compute(w, node.id);
            builder = inside.close_loop();
        } else {
            builder.compute(w, node.id);
        }

        // Signal once if any cross-worker consumer needs it.
        if let Some(bar) = barrier_of[id] {
            builder.signal(w, bar);
        }
    }

    let tape = builder.finish();
    validate_subtile_tape(&tape, graph)
        .expect("lower_dag_to_tape: produced invalid SubtileTape");
    tape
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
    /// A SubtileIR node was `Compute`d more than once across the tape.
    DuplicateCompute { node: SubtileId },
    /// A SubtileIR node has no `Compute` instruction in the tape (every
    /// op-output node must be computed exactly once).
    MissingCompute { node: SubtileId },
    /// On `worker`, two adjacent `Compute` instructions appear in
    /// non-ascending `SubtileId` order — the per-worker subset must be
    /// a topological order of that worker's assigned nodes.
    TopoOrderViolation {
        worker: u32,
        prev_node: SubtileId,
        next_node: SubtileId,
    },
    /// A second `OpenLoop` started while another was still open
    /// (no nesting allowed at this layer).
    NestedLoop { outer_var: u32, inner_var: u32 },
    /// A `CloseLoop` appeared with no matching `OpenLoop`.
    UnmatchedCloseLoop { close_var: u32 },
    /// `OpenLoop` and matching `CloseLoop` disagree on `var`.
    MismatchedLoopVar { open_var: u32, close_var: u32 },
    /// `OpenLoop` and matching `CloseLoop` disagree on `worker`.
    MismatchedLoopWorker {
        var: u32,
        open_worker: u32,
        close_worker: u32,
    },
    /// Tape ended with a still-open loop.
    UnclosedLoop { var: u32 },
    /// `Signal` or `Wait` appeared between an `OpenLoop` and its
    /// matching `CloseLoop` (would reorder vs the iteration count).
    SyncInsideLoop { var: u32, barrier: u32 },
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
pub fn validate_subtile_tape<F: crate::subtile_ir::RopeForm>(
    tape: &SubtileTape,
    graph: &crate::subtile_ir::SubtileIR<F>,
) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();
    check_node_refs(tape, graph, &mut errors);
    check_signal_wait_pairs(tape, &mut errors);
    check_deadlock_cycle(tape, &mut errors);
    check_compute_wellformed(tape, graph, &mut errors);
    check_loop_balance(tape, &mut errors);
    check_data_race(tape, graph, &mut errors);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn check_node_refs<F: crate::subtile_ir::RopeForm>(
    tape: &SubtileTape,
    graph: &crate::subtile_ir::SubtileIR<F>,
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

fn check_data_race<F: crate::subtile_ir::RopeForm>(
    tape: &SubtileTape,
    graph: &crate::subtile_ir::SubtileIR<F>,
    errors: &mut Vec<ValidationError>,
) {
    // For every cross-worker (producer, consumer) edge in the SubtileIR's
    // region-overlap predecessor graph, verify the tape contains a
    // matching Signal/Wait pair. Specifically: there must exist a barrier
    // B such that
    //   - some Signal(B) is on worker_of[producer] at instr index > the
    //     producer's Compute index, AND
    //   - some Wait(B) is on worker_of[consumer] at instr index < the
    //     consumer's Compute index.
    // The per-worker program order on each side gives the sequencing;
    // the Signal/Wait pair establishes the cross-worker happens-before.
    // No fence needed at this layer — the visibility primitive (fence
    // vs mbar handshake) is picked at TkTape lowering.
    let preds = crate::subtile_ir::predecessors(graph);
    let n = graph.nodes.len();

    // Locate the (first) Compute for each node.
    let mut compute_at: Vec<Option<(usize, u32)>> = vec![None; n]; // (instr_idx, worker_id)
    for (i, instr) in tape.instrs.iter().enumerate() {
        if let Instr::Compute { worker, node } = instr
            && (node.0 as usize) < n
            && compute_at[node.0 as usize].is_none()
        {
            compute_at[node.0 as usize] = Some((i, worker.id));
        }
    }

    // Bucket Signal/Wait by barrier id.
    let mut signals_by_b: BTreeMap<u32, Vec<(u32, usize)>> = BTreeMap::new(); // (worker, idx)
    let mut waits_by_b: BTreeMap<u32, Vec<(u32, usize)>> = BTreeMap::new();
    for (i, instr) in tape.instrs.iter().enumerate() {
        match instr {
            Instr::Signal { worker, barrier } => signals_by_b
                .entry(barrier.id)
                .or_default()
                .push((worker.id, i)),
            Instr::Wait { worker, barrier } => waits_by_b
                .entry(barrier.id)
                .or_default()
                .push((worker.id, i)),
            _ => {}
        }
    }

    let mut reported: BTreeMap<(TensorId, u32, u32), ()> = BTreeMap::new();
    for (cid, plist) in preds.iter().enumerate() {
        let (ic, wc) = match compute_at[cid] {
            Some(x) => x,
            None => continue, // missing-compute reported separately
        };
        for prod in plist {
            let pid = prod.0 as usize;
            let (ip, wp) = match compute_at[pid] {
                Some(x) => x,
                None => continue,
            };
            if wp == wc {
                continue;
            }
            // Find a barrier whose Signal is on wp post-ip AND Wait is on wc pre-ic.
            let covered = signals_by_b.iter().any(|(b, sigs)| {
                let post_p = sigs.iter().any(|(w, idx)| *w == wp && *idx > ip);
                if !post_p {
                    return false;
                }
                waits_by_b
                    .get(b)
                    .map(|waits| waits.iter().any(|(w, idx)| *w == wc && *idx < ic))
                    .unwrap_or(false)
            });
            if !covered {
                let key = (graph.nodes[pid].output.tensor, wp, wc);
                if reported.insert(key, ()).is_none() {
                    errors.push(ValidationError::DataRace {
                        tensor: key.0,
                        writer_worker: wp,
                        reader_worker: wc,
                    });
                }
            }
        }
    }
}

fn check_compute_wellformed<F: crate::subtile_ir::RopeForm>(
    tape: &SubtileTape,
    graph: &crate::subtile_ir::SubtileIR<F>,
    errors: &mut Vec<ValidationError>,
) {
    // (a) Each SubtileIR node is Compute'd exactly once.
    // (b) Per-worker, the Compute id sequence is strictly ascending.
    let n = graph.nodes.len();
    let mut emit_count: Vec<u32> = vec![0; n];
    let mut last_per_worker: BTreeMap<u32, SubtileId> = BTreeMap::new();
    for instr in &tape.instrs {
        if let Instr::Compute { worker, node } = instr
            && (node.0 as usize) < n
        {
            emit_count[node.0 as usize] += 1;
            if let Some(prev) = last_per_worker.get(&worker.id).copied() {
                if node.0 <= prev.0 {
                    errors.push(ValidationError::TopoOrderViolation {
                        worker: worker.id,
                        prev_node: prev,
                        next_node: *node,
                    });
                }
            }
            last_per_worker.insert(worker.id, *node);
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
    // No nesting; every OpenLoop matches exactly one CloseLoop (same
    // var, same worker); no Signal/Wait between an Open and its Close.
    let mut open: Option<(u32, u32)> = None; // (var, worker)
    for instr in &tape.instrs {
        match instr {
            Instr::OpenLoop { worker, var, .. } => {
                if let Some((outer_var, _)) = open {
                    errors.push(ValidationError::NestedLoop {
                        outer_var,
                        inner_var: var.id,
                    });
                }
                open = Some((var.id, worker.id));
            }
            Instr::CloseLoop { worker, var } => match open {
                None => errors.push(ValidationError::UnmatchedCloseLoop { close_var: var.id }),
                Some((open_var, open_worker)) => {
                    if open_var != var.id {
                        errors.push(ValidationError::MismatchedLoopVar {
                            open_var,
                            close_var: var.id,
                        });
                    } else if open_worker != worker.id {
                        errors.push(ValidationError::MismatchedLoopWorker {
                            var: var.id,
                            open_worker,
                            close_worker: worker.id,
                        });
                    }
                    open = None;
                }
            },
            Instr::Signal { barrier, .. } | Instr::Wait { barrier, .. } => {
                if let Some((var, _)) = open {
                    errors.push(ValidationError::SyncInsideLoop {
                        var,
                        barrier: barrier.id,
                    });
                }
            }
            Instr::Compute { .. } => {}
        }
    }
    if let Some((var, _)) = open {
        errors.push(ValidationError::UnclosedLoop { var });
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
        EwKind, KvCacheLayout, KvCacheProducer, NeoX, Range, Region, SoftmaxStateId, SubOp,
        SubtileIR, SubtileNode, TensorId, TensorRegion, TensorShape,
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
        let b = TapeBuilder::new(1);
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

    // ── lower_dag_to_tape tests (commit 5) ──────────────────────────

    fn silu_node(id: u32, in_t: TensorId, in_cols: Range, out_t: TensorId, out_cols: Range)
        -> SubtileNode<NeoX>
    {
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

    /// Three Silu nodes chained on one worker: no Signal/Wait, three
    /// Computes in id order.
    #[test]
    fn lower_chain_single_worker_emits_no_sync() {
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 }, // source
            TensorShape { rows: 1, cols: 4 }, // op-out 0
            TensorShape { rows: 1, cols: 4 }, // op-out 1
            TensorShape { rows: 1, cols: 4 }, // op-out 2
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
        let tape = lower_dag_to_tape(&g, 1, 4);
        assert_eq!(tape.num_workers, 1);
        assert_eq!(tape.num_barriers, 0);
        assert_eq!(tape.instrs.len(), 3);
        for (i, instr) in tape.instrs.iter().enumerate() {
            match instr {
                Instr::Compute { worker, node } => {
                    assert_eq!(worker.index(), 0);
                    assert_eq!(node.0, i as u32);
                }
                _ => panic!("chain expected pure Compute stream, got {instr:?} at {i}"),
            }
        }
    }

    /// Producer on worker 0 read by one same-worker consumer AND one
    /// cross-worker consumer: ONE Signal (single barrier covers both
    /// reads), ONE Wait on the cross-worker consumer.
    #[test]
    fn lower_fork_emits_single_signal_for_multiple_consumers() {
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 }, // 0 source
            TensorShape { rows: 1, cols: 4 }, // 1 producer output (cols [0,4))
            TensorShape { rows: 1, cols: 4 }, // 2 same-worker consumer output (cols [0,4))
            TensorShape { rows: 1, cols: 8 }, // 3 cross-worker consumer output (cols [4,8))
        ];
        let nodes = vec![
            silu_node(0, TensorId(0), Range::new(0, 4), TensorId(1), Range::new(0, 4)),
            silu_node(1, TensorId(1), Range::new(0, 4), TensorId(2), Range::new(0, 4)),
            silu_node(2, TensorId(1), Range::new(0, 4), TensorId(3), Range::new(4, 4)),
        ];
        let g: SubtileIR<NeoX> = SubtileIR {
            tensors,
            num_sources: 1,
            nodes,
            result: TensorId(3),
        };
        // unit=4, num_workers=2:
        //   node 0 cols.start=0 → worker 0
        //   node 1 cols.start=0 → worker 0  (same-worker consumer)
        //   node 2 cols.start=4 → worker 1  (cross-worker consumer)
        let tape = lower_dag_to_tape(&g, 2, 4);
        assert_eq!(tape.num_barriers, 1, "one producer → one barrier");
        let signals: Vec<_> = tape
            .instrs
            .iter()
            .filter_map(|i| match i {
                Instr::Signal { worker, barrier } => Some((worker.index(), barrier.index())),
                _ => None,
            })
            .collect();
        let waits: Vec<_> = tape
            .instrs
            .iter()
            .filter_map(|i| match i {
                Instr::Wait { worker, barrier } => Some((worker.index(), barrier.index())),
                _ => None,
            })
            .collect();
        assert_eq!(signals, vec![(0, 0)], "one Signal on producer worker");
        assert_eq!(waits, vec![(1, 0)], "one Wait on cross-worker consumer");
    }

    /// Two producers on different workers, one consumer joining both.
    /// Each cross-worker producer signals; consumer emits two Waits.
    #[test]
    fn lower_join_emits_one_wait_per_cross_worker_producer() {
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 },  // 0 source
            TensorShape { rows: 1, cols: 4 },  // 1 prod-A output (cols [0,4))
            TensorShape { rows: 1, cols: 8 },  // 2 prod-B output (cols [4,8))
            TensorShape { rows: 1, cols: 12 }, // 3 consumer output (cols [8,12))
        ];
        // Consumer reads BOTH producers' outputs (Add: arity 2).
        let consumer = SubtileNode {
            id: SubtileId(2),
            op: SubOp::Elementwise(EwKind::Add),
            inputs: vec![
                TensorRegion {
                    tensor: TensorId(1),
                    region: Region {
                        rows: Range::new(0, 1),
                        cols: Range::new(0, 4),
                    },
                },
                TensorRegion {
                    tensor: TensorId(2),
                    region: Region {
                        rows: Range::new(0, 1),
                        cols: Range::new(4, 4),
                    },
                },
            ],
            output: TensorRegion {
                tensor: TensorId(3),
                region: Region {
                    rows: Range::new(0, 1),
                    cols: Range::new(8, 4),
                },
            },
        };
        let nodes = vec![
            silu_node(0, TensorId(0), Range::new(0, 4), TensorId(1), Range::new(0, 4)),
            silu_node(1, TensorId(0), Range::new(0, 4), TensorId(2), Range::new(4, 4)),
            consumer,
        ];
        let g: SubtileIR<NeoX> = SubtileIR {
            tensors,
            num_sources: 1,
            nodes,
            result: TensorId(3),
        };
        // unit=4, num_workers=3:
        //   node 0 → worker 0; node 1 → worker 1; node 2 → worker 2.
        let tape = lower_dag_to_tape(&g, 3, 4);
        assert_eq!(tape.num_barriers, 2, "two cross-worker producers");
        let waits_on_w2: Vec<_> = tape
            .instrs
            .iter()
            .filter_map(|i| match i {
                Instr::Wait { worker, barrier } if worker.index() == 2 => Some(barrier.index()),
                _ => None,
            })
            .collect();
        assert_eq!(waits_on_w2.len(), 2, "consumer waits on both producers");
        // Each barrier signaled exactly once.
        for b in 0..tape.num_barriers {
            let sigs = tape
                .instrs
                .iter()
                .filter(|i| matches!(i, Instr::Signal { barrier, .. } if barrier.index() == b))
                .count();
            assert_eq!(sigs, 1, "barrier {b} signaled exactly once");
        }
    }

    /// AttnDecode wraps in OpenLoop + Compute + CloseLoop with a
    /// runtime-bounded count (the KV-sweep over seq_len pages).
    #[test]
    fn lower_attn_decode_wraps_in_runtime_loop() {
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 }, // 0 Q (source)
            TensorShape { rows: 4, cols: 4 }, // 1 K (source, prefix)
            TensorShape { rows: 4, cols: 4 }, // 2 V (source, prefix)
            TensorShape { rows: 1, cols: 4 }, // 3 attn output
        ];
        let attn = SubtileNode {
            id: SubtileId(0),
            op: SubOp::AttnDecode {
                num_q_heads: 1,
                num_kv_heads: 1,
                head_dim: 4,
                scale: 0.5,
                layout: KvCacheLayout::for_cache_tensor(TensorId(1), 1, 4),
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
        let g: SubtileIR<NeoX> = SubtileIR {
            tensors,
            num_sources: 3,
            nodes: vec![attn],
            result: TensorId(3),
        };
        let tape = lower_dag_to_tape(&g, 1, 4);
        assert_eq!(tape.num_loop_vars, 1, "one AttnDecode → one loop var");
        assert_eq!(tape.num_runtime_bounds, 1, "one AttnDecode → one runtime bound");
        // The shape is OpenLoop + Compute + CloseLoop — three instrs.
        assert_eq!(tape.instrs.len(), 3);
        assert!(matches!(
            tape.instrs[0],
            Instr::OpenLoop { bound: LoopBound::Runtime(_), .. }
        ));
        assert!(matches!(tape.instrs[1], Instr::Compute { .. }));
        assert!(matches!(tape.instrs[2], Instr::CloseLoop { .. }));
    }

    /// Two AttnDecode nodes → two runtime bounds, two loop vars (each
    /// AttnDecode gets its own KV-sweep).
    #[test]
    fn lower_two_attn_decodes_each_get_own_runtime_loop() {
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 }, // 0 Q
            TensorShape { rows: 4, cols: 4 }, // 1 K
            TensorShape { rows: 4, cols: 4 }, // 2 V
            TensorShape { rows: 1, cols: 4 }, // 3 attn-A out
            TensorShape { rows: 1, cols: 4 }, // 4 attn-B out
        ];
        let mk_attn = |id: u32, out_t: TensorId, sm: u32| SubtileNode {
            id: SubtileId(id),
            op: SubOp::AttnDecode {
                num_q_heads: 1,
                num_kv_heads: 1,
                head_dim: 4,
                scale: 0.5,
                layout: KvCacheLayout::for_cache_tensor(TensorId(1), 1, 4),
                producer: KvCacheProducer::pre_populated_ext(),
                softmax_state: SoftmaxStateId::new(sm),
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
                tensor: out_t,
                region: Region {
                    rows: Range::new(0, 1),
                    cols: Range::new(0, 4),
                },
            },
        };
        let g: SubtileIR<NeoX> = SubtileIR {
            tensors,
            num_sources: 3,
            nodes: vec![mk_attn(0, TensorId(3), 0), mk_attn(1, TensorId(4), 1)],
            result: TensorId(4),
        };
        let tape = lower_dag_to_tape(&g, 1, 4);
        assert_eq!(tape.num_loop_vars, 2);
        assert_eq!(tape.num_runtime_bounds, 2);
    }

    // ── validate_subtile_tape full-impl tests (commit 5b) ───────────

    /// lower_dag_to_tape's output validates clean (positive control:
    /// the walker emits Signal/Wait pairs for every cross-worker edge,
    /// so no DataRace fires).
    #[test]
    fn lower_output_validates_clean_on_join_shape() {
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 1, cols: 8 },
            TensorShape { rows: 1, cols: 12 },
        ];
        let consumer = SubtileNode {
            id: SubtileId(2),
            op: SubOp::Elementwise(EwKind::Add),
            inputs: vec![
                TensorRegion {
                    tensor: TensorId(1),
                    region: Region {
                        rows: Range::new(0, 1),
                        cols: Range::new(0, 4),
                    },
                },
                TensorRegion {
                    tensor: TensorId(2),
                    region: Region {
                        rows: Range::new(0, 1),
                        cols: Range::new(4, 4),
                    },
                },
            ],
            output: TensorRegion {
                tensor: TensorId(3),
                region: Region {
                    rows: Range::new(0, 1),
                    cols: Range::new(8, 4),
                },
            },
        };
        let g: SubtileIR<NeoX> = SubtileIR {
            tensors,
            num_sources: 1,
            nodes: vec![
                silu_node(0, TensorId(0), Range::new(0, 4), TensorId(1), Range::new(0, 4)),
                silu_node(1, TensorId(0), Range::new(0, 4), TensorId(2), Range::new(4, 4)),
                consumer,
            ],
            result: TensorId(3),
        };
        let tape = lower_dag_to_tape(&g, 3, 4);
        assert_eq!(validate_subtile_tape(&tape, &g), Ok(()));
    }

    /// Cross-worker write→read with NO matching Signal/Wait pair flags
    /// DataRace. (Hand-built tape that strips the sync edges.)
    #[test]
    fn validator_flags_unguarded_cross_worker_read() {
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 1, cols: 8 },
        ];
        let g: SubtileIR<NeoX> = SubtileIR {
            tensors,
            num_sources: 1,
            nodes: vec![
                silu_node(0, TensorId(0), Range::new(0, 4), TensorId(1), Range::new(0, 4)),
                silu_node(1, TensorId(1), Range::new(0, 4), TensorId(2), Range::new(4, 4)),
            ],
            result: TensorId(2),
        };
        // Hand-build a tape: producer on w0, consumer on w1, NO Signal/Wait.
        let mut b = TapeBuilder::new(2);
        let w0 = b.worker(0);
        let w1 = b.worker(1);
        b.compute(w0, SubtileId(0)).compute(w1, SubtileId(1));
        let tape = b.finish();
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.iter().any(|e| matches!(
                e,
                ValidationError::DataRace { writer_worker: 0, reader_worker: 1, .. }
            )),
            "want DataRace(0→1), got {err:?}"
        );
    }

    /// Per-worker Compute id order violated → TopoOrderViolation.
    #[test]
    fn validator_flags_per_worker_topo_violation() {
        // Build a 2-node SubtileIR; emit Computes in descending id
        // order on one worker. Single-node graphs can't trigger the
        // adjacent-id check.
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 1, cols: 4 },
        ];
        let g: SubtileIR<NeoX> = SubtileIR {
            tensors,
            num_sources: 1,
            nodes: vec![
                silu_node(0, TensorId(0), Range::new(0, 4), TensorId(1), Range::new(0, 4)),
                silu_node(1, TensorId(1), Range::new(0, 4), TensorId(2), Range::new(0, 4)),
            ],
            result: TensorId(2),
        };
        let mut b = TapeBuilder::new(1);
        let w0 = b.worker(0);
        b.compute(w0, SubtileId(1)).compute(w0, SubtileId(0));
        let tape = b.finish();
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.iter().any(|e| matches!(
                e,
                ValidationError::TopoOrderViolation { worker: 0, .. }
            )),
            "want TopoOrderViolation, got {err:?}"
        );
    }

    /// MissingCompute fires for a SubtileIR node never Compute'd.
    #[test]
    fn validator_flags_missing_compute() {
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 1, cols: 4 },
        ];
        let g: SubtileIR<NeoX> = SubtileIR {
            tensors,
            num_sources: 1,
            nodes: vec![
                silu_node(0, TensorId(0), Range::new(0, 4), TensorId(1), Range::new(0, 4)),
                silu_node(1, TensorId(1), Range::new(0, 4), TensorId(2), Range::new(0, 4)),
            ],
            result: TensorId(2),
        };
        let mut b = TapeBuilder::new(1);
        let w0 = b.worker(0);
        b.compute(w0, SubtileId(0)); // node 1 never Computed
        let tape = b.finish();
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.contains(&ValidationError::MissingCompute { node: SubtileId(1) }),
            "want MissingCompute(1), got {err:?}"
        );
    }

    /// DuplicateCompute fires if a node is Compute'd more than once.
    #[test]
    fn validator_flags_duplicate_compute() {
        let g = tiny_graph();
        let mut b = TapeBuilder::new(1);
        let w0 = b.worker(0);
        b.compute(w0, SubtileId(0)).compute(w0, SubtileId(0));
        let tape = b.finish();
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.contains(&ValidationError::DuplicateCompute { node: SubtileId(0) }),
            "want DuplicateCompute(0), got {err:?}"
        );
    }

    /// Hand-built tape with an OpenLoop and no CloseLoop → UnclosedLoop.
    /// The TapeBuilder typestate prevents this at compile time, but the
    /// validator defends against direct Vec<Instr> mutation.
    #[test]
    fn validator_flags_unclosed_loop() {
        let g = tiny_graph();
        let tape = SubtileTape {
            instrs: vec![
                Instr::OpenLoop {
                    worker: WorkerId {
                        id: 0,
                        _seal: sealed::Seal(()),
                    },
                    var: LoopVarId {
                        id: 0,
                        _seal: sealed::Seal(()),
                    },
                    bound: LoopBound::Const(4),
                },
                Instr::Compute {
                    worker: WorkerId {
                        id: 0,
                        _seal: sealed::Seal(()),
                    },
                    node: SubtileId(0),
                },
                // no CloseLoop
            ],
            num_workers: 1,
            num_barriers: 0,
            num_loop_vars: 1,
            num_runtime_bounds: 0,
        };
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.contains(&ValidationError::UnclosedLoop { var: 0 }),
            "want UnclosedLoop, got {err:?}"
        );
    }

    /// CloseLoop on a different LoopVarId than the matching OpenLoop →
    /// MismatchedLoopVar.
    #[test]
    fn validator_flags_mismatched_loop_var() {
        let g = tiny_graph();
        let mk_w = || WorkerId {
            id: 0,
            _seal: sealed::Seal(()),
        };
        let mk_var = |id| LoopVarId {
            id,
            _seal: sealed::Seal(()),
        };
        let tape = SubtileTape {
            instrs: vec![
                Instr::OpenLoop {
                    worker: mk_w(),
                    var: mk_var(0),
                    bound: LoopBound::Const(4),
                },
                Instr::Compute {
                    worker: mk_w(),
                    node: SubtileId(0),
                },
                Instr::CloseLoop {
                    worker: mk_w(),
                    var: mk_var(99),
                },
            ],
            num_workers: 1,
            num_barriers: 0,
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

    /// Hand-built Signal between OpenLoop and CloseLoop → SyncInsideLoop.
    #[test]
    fn validator_flags_sync_inside_loop() {
        let g = tiny_graph();
        let mk_w = |id| WorkerId {
            id,
            _seal: sealed::Seal(()),
        };
        let mk_var = LoopVarId {
            id: 0,
            _seal: sealed::Seal(()),
        };
        let mk_bar = BarrierId {
            id: 0,
            _seal: sealed::Seal(()),
        };
        let tape = SubtileTape {
            instrs: vec![
                Instr::OpenLoop {
                    worker: mk_w(0),
                    var: mk_var,
                    bound: LoopBound::Const(4),
                },
                Instr::Signal {
                    worker: mk_w(0),
                    barrier: mk_bar,
                },
                Instr::Wait {
                    worker: mk_w(1),
                    barrier: mk_bar,
                },
                Instr::Compute {
                    worker: mk_w(0),
                    node: SubtileId(0),
                },
                Instr::CloseLoop {
                    worker: mk_w(0),
                    var: mk_var,
                },
            ],
            num_workers: 2,
            num_barriers: 1,
            num_loop_vars: 1,
            num_runtime_bounds: 0,
        };
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.iter().any(|e| matches!(
                e,
                ValidationError::SyncInsideLoop { var: 0, barrier: 0 }
            )),
            "want SyncInsideLoop, got {err:?}"
        );
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
