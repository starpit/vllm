// SPDX-License-Identifier: Apache-2.0
//! Linear, target-agnostic **SubtileTape** — a topological linearization
//! of the [`crate::subtile_ir::SubtileIR`] DAG.
//!
//! ## What this is (and is not)
//!
//! SubtileTape is the sequential semantics: the order in which a
//! conceptual single thread would execute the Computes, with explicit
//! `OpenLoop` / `CloseLoop` brackets for runtime-bounded loops (the
//! `AttnDecode` KV-sweep being the only such loop today). That's it.
//!
//! - **No workers, no barriers, no signals, no waits.** Parallelism
//!   lives in the SubtileIR DAG (region-overlap predecessors); the
//!   tape is one valid topological linearization. A "worker" is a
//!   target-specific parallelism dimension (CTAs / threadgroups /
//!   warp-roles); how to spread the tape's computes across them is a
//!   per-target lowering decision, not an IR concept.
//! - **No memory class, no fence.** Whether an output lives in shmem
//!   vs gmem and what visibility primitive synchronizes a write→read
//!   are target-specific decisions made at the per-target lowering
//!   (`lower_tape_to_tk` / `lower_tape_to_metal`).
//! - **No page, no parity.** Page lifecycle and phase parity are
//!   TkTape concepts.
//!
//! Every fact a downstream pass needs that isn't in `tape.instrs` it
//! derives from the SubtileIR — directly, every time. The tape is the
//! tape; analyses live in passes.
//!
//! ## Compile-time loop balance
//!
//! [`TapeBuilder<S>`] uses a typestate to make orphan loop brackets
//! structurally impossible:
//!
//! - `close_loop` is only callable on `TapeBuilder<state::InsideLoop>`.
//! - `finish` is only callable on `TapeBuilder<state::Outside>`.
//!
//! [`validate_subtile_tape`] then catches the runtime-shape errors
//! (every node Compute'd exactly once, ascending-id order, loop
//! balance for hand-built tapes that bypass the typestate).

#![allow(dead_code)]

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
/// semantics. There is no per-instruction worker tag; per-target
/// lowering decides parallelism.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Instr {
    /// Compute the named SubtileIR node. The node's input / output
    /// regions are looked up in the SubtileIR; this instruction only
    /// names the node identity.
    Compute { node: SubtileId },
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
    pub num_loop_vars: u32,
    pub num_runtime_bounds: u32,
}

// ── TapeBuilder<S> typestate ────────────────────────────────────────

/// Compile-time state markers for [`TapeBuilder<S>`].
pub mod state {
    /// No `OpenLoop` is in flight. `finish` is only available here.
    #[derive(Debug)]
    pub enum Outside {}
    /// An `OpenLoop` is in flight. `close_loop` is only available here.
    #[derive(Debug)]
    pub enum InsideLoop {}
}

/// Typestate-tracked builder. The `S` parameter is one of
/// [`state::Outside`] / [`state::InsideLoop`]; the same `TapeBuilder`
/// type carries different methods depending on `S`. Misuse (e.g.
/// `close_loop` on `Outside`, `finish` on `InsideLoop`) = no matching
/// impl, **compile error**.
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
pub struct TapeBuilder<S = state::Outside> {
    instrs: Vec<Instr>,
    next_loop_var: u32,
    next_runtime_bound: u32,
    /// `Some(var)` while an OpenLoop is in flight.
    cur_loop: Option<LoopVarId>,
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
            next_loop_var: 0,
            next_runtime_bound: 0,
            cur_loop: None,
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

    pub fn compute(&mut self, node: SubtileId) -> &mut Self {
        self.instrs.push(Instr::Compute { node });
        self
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
            next_loop_var: self.next_loop_var,
            next_runtime_bound: self.next_runtime_bound,
            cur_loop: Some(var),
            _state: PhantomData,
        };
        (inside, var)
    }

    /// Finalize: produce the linear tape. Only available with no loop
    /// in flight (the `Outside` state).
    pub fn finish(self) -> SubtileTape {
        SubtileTape {
            instrs: self.instrs,
            num_loop_vars: self.next_loop_var,
            num_runtime_bounds: self.next_runtime_bound,
        }
    }
}

impl TapeBuilder<state::InsideLoop> {
    pub fn compute(&mut self, node: SubtileId) -> &mut Self {
        self.instrs.push(Instr::Compute { node });
        self
    }

    /// Close the active loop. Returns a builder back in the `Outside`
    /// state — `finish` is available again.
    pub fn close_loop(mut self) -> TapeBuilder<state::Outside> {
        let var = self
            .cur_loop
            .expect("InsideLoop without cur_loop (typestate invariant violated)");
        self.instrs.push(Instr::CloseLoop { var });
        TapeBuilder::<state::Outside> {
            instrs: self.instrs,
            next_loop_var: self.next_loop_var,
            next_runtime_bound: self.next_runtime_bound,
            cur_loop: None,
            _state: PhantomData,
        }
    }
}

// ── Runtime validator ───────────────────────────────────────────────

/// One class of runtime-detectable invariant violation. The typestate
/// already catches the structural ones (orphan loop brackets via
/// `close_loop` / `finish` typestate gates; sealed handles); these are
/// the runtime-shape errors only the validator can see.
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
    /// A second `OpenLoop` started while another was still open
    /// (no nesting allowed at this layer).
    NestedLoop { outer_var: u32, inner_var: u32 },
    /// A `CloseLoop` appeared with no matching `OpenLoop`.
    UnmatchedCloseLoop { close_var: u32 },
    /// `OpenLoop` and matching `CloseLoop` disagree on `var`.
    MismatchedLoopVar { open_var: u32, close_var: u32 },
    /// Tape ended with a still-open loop.
    UnclosedLoop { var: u32 },
}

/// Runtime validator. Three checks:
///
/// 1. **Compute well-formedness** — every SubtileIR node is Compute'd
///    exactly once; no `Compute` references an out-of-range node.
/// 2. **Topo order** — adjacent Computes appear in strictly ascending
///    `SubtileId` order (the SubtileIR is already ascending-id topo;
///    the tape is a valid linearization).
/// 3. **Loop balance** — every `OpenLoop` matches a `CloseLoop` with
///    the same `LoopVarId`; no nesting; no unclosed loops.
pub fn validate_subtile_tape<F: crate::subtile_ir::RopeForm>(
    tape: &SubtileTape,
    graph: &crate::subtile_ir::SubtileIR<F>,
) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();
    check_node_refs(tape, graph, &mut errors);
    check_compute_wellformed(tape, graph, &mut errors);
    check_loop_balance(tape, &mut errors);
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
        if let Instr::Compute { node } = instr
            && node.0 >= n_nodes
        {
            errors.push(ValidationError::UnknownNode { node: *node });
        }
    }
}

fn check_compute_wellformed<F: crate::subtile_ir::RopeForm>(
    tape: &SubtileTape,
    graph: &crate::subtile_ir::SubtileIR<F>,
    errors: &mut Vec<ValidationError>,
) {
    let n = graph.nodes.len();
    let mut emit_count: Vec<u32> = vec![0; n];
    let mut last_id: Option<SubtileId> = None;
    for instr in &tape.instrs {
        if let Instr::Compute { node } = instr
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
            Instr::Compute { .. } => {}
        }
    }
    if let Some(var) = open {
        errors.push(ValidationError::UnclosedLoop { var });
    }
}

// ── Lowering: SubtileIR → SubtileTape ───────────────────────────────

/// Lower a [`crate::subtile_ir::SubtileIR`] DAG to a linear
/// [`SubtileTape`] in one deterministic pass.
///
/// Walks `graph.nodes` in ascending `SubtileId` order (the SubtileIR is
/// itself ascending-id-topo, so this is a valid topological order).
/// Each node emits one `Compute` instruction; `SubOp::AttnDecode` wraps
/// in an `OpenLoop` / `CloseLoop` pair over a runtime-bounded count
/// (the KV-sweep over `seq_len` pages).
///
/// **No worker assignment, no Signal/Wait emission.** Per-target work
/// distribution (across CTAs / threadgroups / warp-roles) and any
/// inter-unit synchronization happen at the per-target lowering
/// (`lower_tape_to_tk` for TK megakernel; out-of-scope for Metal).
///
/// The returned `SubtileTape` has been validated against `graph` via
/// [`validate_subtile_tape`]; callers can assume well-formedness.
pub fn lower_dag_to_tape<F: crate::subtile_ir::RopeForm>(
    graph: &crate::subtile_ir::SubtileIR<F>,
) -> SubtileTape {
    use crate::subtile_ir::{SubOp, validate};

    validate(graph).expect("lower_dag_to_tape: invalid SubtileIR");

    let mut builder = TapeBuilder::new();
    for node in &graph.nodes {
        if matches!(node.op, SubOp::AttnDecode { .. }) {
            // Runtime-bounded KV-sweep loop wraps the AttnDecode Compute.
            let rb = builder.alloc_runtime_bound();
            let (mut inside, _var) = builder.open_loop(LoopBound::Runtime(rb));
            inside.compute(node.id);
            builder = inside.close_loop();
        } else {
            builder.compute(node.id);
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
    Computed(SubtileId),
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
            Instr::Compute { node } => PlayStep::Computed(*node),
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

    // ── Builder shape ─────────────────────────────────────────────

    #[test]
    fn build_finish_round_trips() {
        let mut b = TapeBuilder::new();
        b.compute(SubtileId(0));
        let tape = b.finish();
        assert_eq!(tape.instrs.len(), 1);
        assert!(matches!(tape.instrs[0], Instr::Compute { node: SubtileId(0) }));
    }

    #[test]
    fn loop_open_close_round_trips() {
        let b = TapeBuilder::new();
        let (mut inside, _var) = b.open_loop(LoopBound::Const(8));
        inside.compute(SubtileId(0));
        let outer = inside.close_loop();
        let tape = outer.finish();
        assert_eq!(tape.num_loop_vars, 1);
        assert!(matches!(tape.instrs.last(), Some(Instr::CloseLoop { .. })));
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
        b.compute(SubtileId(0));
        let tape = b.finish();
        assert_eq!(validate_subtile_tape(&tape, &g), Ok(()));
    }

    #[test]
    fn validator_flags_unknown_node() {
        let g = tiny_graph();
        let mut b = TapeBuilder::new();
        b.compute(SubtileId(99));
        let tape = b.finish();
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.iter().any(|e| matches!(e, ValidationError::UnknownNode { node } if node.0 == 99)),
            "want UnknownNode(99), got {err:?}"
        );
    }

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
        let mut b = TapeBuilder::new();
        b.compute(SubtileId(0)); // node 1 never Compute'd
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
        let mut b = TapeBuilder::new();
        b.compute(SubtileId(0)).compute(SubtileId(0));
        let tape = b.finish();
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.contains(&ValidationError::DuplicateCompute { node: SubtileId(0) }),
            "want DuplicateCompute(0), got {err:?}"
        );
    }

    #[test]
    fn validator_flags_topo_order_violation() {
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
        let mut b = TapeBuilder::new();
        b.compute(SubtileId(1)).compute(SubtileId(0)); // descending
        let tape = b.finish();
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.iter().any(|e| matches!(e, ValidationError::TopoOrderViolation { .. })),
            "want TopoOrderViolation, got {err:?}"
        );
    }

    // ── Validator: loop balance (hand-built tapes; typestate is the
    //               compile-time backstop, validator catches direct
    //               Vec<Instr> mutation).

    #[test]
    fn validator_flags_unclosed_loop() {
        let g = tiny_graph();
        let tape = SubtileTape {
            instrs: vec![
                Instr::OpenLoop {
                    var: LoopVarId {
                        id: 0,
                        _seal: sealed::Seal(()),
                    },
                    bound: LoopBound::Const(4),
                },
                Instr::Compute { node: SubtileId(0) },
                // no CloseLoop
            ],
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
        let mk_var = |id| LoopVarId {
            id,
            _seal: sealed::Seal(()),
        };
        let tape = SubtileTape {
            instrs: vec![
                Instr::OpenLoop {
                    var: mk_var(0),
                    bound: LoopBound::Const(4),
                },
                Instr::Compute { node: SubtileId(0) },
                Instr::CloseLoop { var: mk_var(99) },
            ],
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
                Instr::Compute { node: SubtileId(0) },
                Instr::CloseLoop {
                    var: LoopVarId {
                        id: 0,
                        _seal: sealed::Seal(()),
                    },
                },
            ],
            num_loop_vars: 1,
            num_runtime_bounds: 0,
        };
        let err = validate_subtile_tape(&tape, &g).unwrap_err();
        assert!(
            err.contains(&ValidationError::UnmatchedCloseLoop { close_var: 0 }),
            "want UnmatchedCloseLoop, got {err:?}"
        );
    }

    // ── lower_dag_to_tape ──────────────────────────────────────────

    #[test]
    fn lower_chain_emits_pure_compute_stream() {
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
        let tape = lower_dag_to_tape(&g);
        assert_eq!(tape.instrs.len(), 3);
        assert_eq!(tape.num_loop_vars, 0);
        assert_eq!(tape.num_runtime_bounds, 0);
        for (i, instr) in tape.instrs.iter().enumerate() {
            match instr {
                Instr::Compute { node } => assert_eq!(node.0, i as u32),
                _ => panic!("chain expected pure Compute stream, got {instr:?} at {i}"),
            }
        }
    }

    #[test]
    fn lower_attn_decode_wraps_in_runtime_loop() {
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 },
            TensorShape { rows: 4, cols: 4 },
            TensorShape { rows: 4, cols: 4 },
            TensorShape { rows: 1, cols: 4 },
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
        let tape = lower_dag_to_tape(&g);
        assert_eq!(tape.num_loop_vars, 1);
        assert_eq!(tape.num_runtime_bounds, 1);
        assert_eq!(tape.instrs.len(), 3);
        assert!(matches!(
            tape.instrs[0],
            Instr::OpenLoop { bound: LoopBound::Runtime(_), .. }
        ));
        assert!(matches!(tape.instrs[1], Instr::Compute { .. }));
        assert!(matches!(tape.instrs[2], Instr::CloseLoop { .. }));
    }

    // ── Player skeleton ────────────────────────────────────────────

    #[test]
    fn play_skeleton_round_trips_every_instr() {
        let mut b = TapeBuilder::new();
        b.compute(SubtileId(0));
        let tape = b.finish();
        let steps = play_skeleton(&tape);
        assert_eq!(steps, vec![PlayStep::Computed(SubtileId(0))]);
    }
}
