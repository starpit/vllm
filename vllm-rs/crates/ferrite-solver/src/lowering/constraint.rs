// SPDX-License-Identifier: Apache-2.0
//! Structured [`Constraint`] enum + propagation rules.
//!
//! Constraints are **data**, not closures. Each variant is a
//! pattern the CP propagator dispatches on; the same data can be
//! linearized into ILP later. There are no `Box<dyn Fn>` constraint
//! checkers anywhere — that would make ILP encoding impossible.
//!
//! ## Constraint families
//!
//! 1. **Cover completeness** — every tile is claimed by exactly
//!    one subgraph.
//! 2. **Subgraph match** — for each (subgraph, impl) pair, the
//!    impl's `matches` predicate must hold on the claimed tiles.
//! 3. **Target compatibility** — every chosen impl must be
//!    runnable on the target.
//! 4. **Dependency order** — tile dep edges must be respected by
//!    the schedule (producer step < consumer step, or both in the
//!    same subgraph and the impl handles it internally).
//! 5. **Cooperative-launch exclusivity** — at most one
//!    `CooperativeLaunch` per step.
//! 6. **Compilation-unit reg budget** — unioned per-thread reg
//!    count of all DeviceCallable impls in one unit ≤
//!    `max_regs_per_thread`.
//! 7. **Compilation-unit shmem budget** — unioned shmem of all
//!    DeviceCallable impls in one unit ≤ `max_shmem_per_cta_bytes`.
//! 8. **Layout compatibility** — for each producer→consumer edge,
//!    the producer's output layout matches the consumer's
//!    expected input layout (or a layout-conversion impl is
//!    inserted in between).
//! 9. **Handoff compatibility** — for each scheduled handoff, the
//!    mechanism is in both impl's supported-handoff sets and
//!    available on the target.
//!
//! Each constraint variant carries the data the propagator needs.
//! The propagator's `check` returns
//! [`ConstraintStatus::Satisfied`] / `Violated` / `Unknown`.

use crate::lowering::assignment::{Assignment, CompilationUnitId, SubgraphId};
use crate::lowering::implementation::{Handoff, ImplId, LaunchKind};
use crate::lowering::library::ImplementationLibrary;
use crate::lowering::tile_graph::{TileGraph, TileId};
use crate::target_profile::TargetProfile;

/// Outcome of evaluating a constraint against a (possibly partial)
/// [`Assignment`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintStatus {
    /// All variables this constraint references are bound and the
    /// constraint holds. Safe to commit.
    Satisfied,
    /// All variables are bound and the constraint is **violated**.
    /// The current partial assignment is infeasible — backtrack.
    Violated,
    /// At least one referenced variable is still unbound. The
    /// constraint will be re-evaluated as the search proceeds.
    Unknown,
}

/// One declarative constraint. The CP propagator dispatches on
/// this enum; the ILP backend (later) linearizes each variant.
#[derive(Clone, Debug)]
pub enum Constraint {
    /// `cover` must include every TileId in `0..num_tiles`.
    /// Each tile claimed by exactly one subgraph.
    CoverComplete { num_tiles: u32 },
    /// For the given (subgraph, impl) pair, the impl's matcher
    /// must accept the subgraph's claimed tiles. Verified at
    /// commit time, when both subgraph claim and impl assignment
    /// are bound.
    SubgraphMatches { subgraph: SubgraphId },
    /// The impl assigned to this subgraph must report
    /// `target_compatible(profile) == true`.
    TargetCompatible { subgraph: SubgraphId },
    /// The impl assigned to this subgraph must have a
    /// `workload_constraint()` that accepts `profile.num_tokens()`.
    /// Distinct from `TargetCompatible` (GPU capability) — this is
    /// about workload-shape correctness requirements (e.g. "GEMV
    /// only handles M=1" or "this fused kernel requires M ≥ 64").
    WorkloadCompatible { subgraph: SubgraphId },
    /// The dep edge `producer_tile → consumer_tile` must be
    /// respected by the schedule: either both tiles are in the
    /// same subgraph (in which case the impl handles it
    /// internally), or the producer's subgraph is scheduled in a
    /// step strictly earlier than the consumer's subgraph (or in
    /// the same step with a non-Internal handoff between them).
    DependencyOrder { producer: TileId, consumer: TileId },
    /// Among all subgraphs scheduled in the same step, at most
    /// one may be a `CooperativeLaunch`. (Hardware constraint.)
    CooperativeExclusive,
    /// All subgraphs in the given compilation unit must, when
    /// their impls' `regs_per_thread` are unioned via
    /// [`crate::lowering::Resources::union_max`], stay within
    /// `cap`.
    CompilationUnitRegBudget { unit: CompilationUnitId, cap: u32 },
    /// Same as above for shmem.
    CompilationUnitShmemBudget { unit: CompilationUnitId, cap: u32 },
    /// The handoff between two subgraphs must use a mechanism
    /// supported by both impls AND available on the target.
    HandoffCompatible {
        producer: SubgraphId,
        consumer: SubgraphId,
    },
    /// If a claimed tile has consumers outside its subgraph, the
    /// tile must appear in the subgraph's `boundary_outputs` (its
    /// output must be materialized to GMEM so external consumers
    /// can read it). Without this, prologue fusion that consumes
    /// an intermediate internally would silently starve downstream
    /// tiles.
    IntermediateMaterialized {
        /// The tile whose output might be consumed externally.
        producer: TileId,
        /// A consumer of `producer` that may end up outside the
        /// subgraph claiming `producer`.
        consumer: TileId,
    },
    /// Controls whether a dependency edge's handoff must truncate
    /// the intermediate to storage dtype (bf16). When
    /// `require_truncation` is true, fusion across this edge is
    /// forbidden (the handoff must go through GMEM) — ensuring the
    /// output is bit-identical to the unfused reference.
    ///
    /// When false (default for serving), fusion can skip the
    /// truncation point, producing slightly different (but more
    /// precise) output.
    ///
    /// Generated per dependency edge where the producer is a
    /// reduction (norm) or accumulation (GEMM epilogue) — the
    /// points where dtype truncation changes the numerical path.
    PrecisionBounded {
        producer: TileId,
        consumer: TileId,
        /// If true, the handoff between these tiles must truncate
        /// to storage dtype (i.e., go through GMEM). If false,
        /// any handoff is acceptable.
        require_truncation: bool,
    },
}

impl Constraint {
    /// Evaluate this constraint against the (possibly partial)
    /// assignment. The CP solver calls this on every constraint
    /// after each variable commit; constraints that report
    /// `Violated` trigger backtracking.
    pub fn check(
        &self,
        assignment: &Assignment,
        tile_graph: &TileGraph,
        library: &ImplementationLibrary,
        profile: &TargetProfile,
    ) -> ConstraintStatus {
        match self {
            Constraint::CoverComplete { num_tiles } => {
                if assignment.cover.len() < *num_tiles as usize {
                    // Some tiles still unclaimed.
                    ConstraintStatus::Unknown
                } else if assignment.cover.len() == *num_tiles as usize {
                    // All bound; check that every TileId is present.
                    for i in 0..*num_tiles {
                        if !assignment.cover.contains_key(&TileId(i)) {
                            return ConstraintStatus::Violated;
                        }
                    }
                    ConstraintStatus::Satisfied
                } else {
                    // More entries than tiles ⇒ corrupted.
                    ConstraintStatus::Violated
                }
            }

            Constraint::SubgraphMatches { subgraph } => {
                let Some(impl_id) = assignment.impls.get(subgraph) else {
                    return ConstraintStatus::Unknown;
                };
                let claimed = assignment.tiles_in_subgraph(*subgraph);
                if claimed.is_empty() {
                    return ConstraintStatus::Unknown;
                }
                let imp = library.get(*impl_id);
                // Re-run the matcher against the claimed seed; verify
                // the returned MatchInfo's claimed_tiles equals our
                // committed tiles. (The matcher is deterministic on
                // the same seed.)
                let seed = *claimed.iter().min().unwrap();
                match imp.matches(tile_graph, seed, profile) {
                    Some(m) => {
                        let mut claim_sorted = m.claimed_tiles.clone();
                        claim_sorted.sort();
                        let mut committed_sorted = claimed.clone();
                        committed_sorted.sort();
                        if claim_sorted == committed_sorted {
                            ConstraintStatus::Satisfied
                        } else {
                            ConstraintStatus::Violated
                        }
                    }
                    None => ConstraintStatus::Violated,
                }
            }

            Constraint::TargetCompatible { subgraph } => {
                let Some(impl_id) = assignment.impls.get(subgraph) else {
                    return ConstraintStatus::Unknown;
                };
                if library.get(*impl_id).target_compatible(profile) {
                    ConstraintStatus::Satisfied
                } else {
                    ConstraintStatus::Violated
                }
            }

            Constraint::WorkloadCompatible { subgraph } => {
                let Some(impl_id) = assignment.impls.get(subgraph) else {
                    return ConstraintStatus::Unknown;
                };
                if library
                    .get(*impl_id)
                    .workload_constraint()
                    .accepts(profile.num_tokens())
                {
                    ConstraintStatus::Satisfied
                } else {
                    ConstraintStatus::Violated
                }
            }

            Constraint::DependencyOrder { producer, consumer } => {
                let p_sg = assignment.cover.get(producer);
                let c_sg = assignment.cover.get(consumer);
                let (Some(p_sg), Some(c_sg)) = (p_sg, c_sg) else {
                    return ConstraintStatus::Unknown;
                };
                if p_sg == c_sg {
                    // Same subgraph — impl handles dep internally.
                    return ConstraintStatus::Satisfied;
                }
                let p_slot = assignment.schedule.get(p_sg);
                let c_slot = assignment.schedule.get(c_sg);
                let (Some(p_slot), Some(c_slot)) = (p_slot, c_slot) else {
                    return ConstraintStatus::Unknown;
                };
                if p_slot.step < c_slot.step {
                    ConstraintStatus::Satisfied
                } else if p_slot.step == c_slot.step {
                    // Same step ⇒ they run "concurrently"; the dep
                    // is conveyed via the assigned handoff. The
                    // HandoffCompatible constraint covers feasibility.
                    if assignment.handoffs.contains_key(&(*p_sg, *c_sg)) {
                        ConstraintStatus::Satisfied
                    } else {
                        ConstraintStatus::Unknown
                    }
                } else {
                    // Producer scheduled AFTER consumer ⇒ dep violated.
                    ConstraintStatus::Violated
                }
            }

            Constraint::CooperativeExclusive => {
                // Group subgraphs by step. For each step, count
                // how many CooperativeLaunch impls are scheduled.
                use std::collections::HashMap;
                let mut by_step: HashMap<u32, u32> = HashMap::new();
                for (sg, slot) in &assignment.schedule {
                    let Some(impl_id) = assignment.impls.get(sg) else {
                        return ConstraintStatus::Unknown;
                    };
                    if matches!(
                        library.get(*impl_id).launch_kind(),
                        LaunchKind::CooperativeLaunch
                    ) {
                        let count = by_step.entry(slot.step).or_insert(0);
                        *count += 1;
                        if *count > 1 {
                            return ConstraintStatus::Violated;
                        }
                    }
                }
                ConstraintStatus::Satisfied
            }

            Constraint::CompilationUnitRegBudget { unit, cap } => {
                let subgraphs = assignment.subgraphs_in_unit(*unit);
                if subgraphs.is_empty() {
                    return ConstraintStatus::Satisfied;
                }
                let mut max_regs: u32 = 0;
                for sg in subgraphs {
                    let Some(impl_id) = assignment.impls.get(&sg) else {
                        return ConstraintStatus::Unknown;
                    };
                    let imp = library.get(*impl_id);
                    // Build a fake MatchInfo carrying just the
                    // claimed tiles for resource lookup.
                    let claimed = assignment.tiles_in_subgraph(sg);
                    let m = crate::lowering::implementation::MatchInfo {
                        claimed_tiles: claimed,
                        boundary_inputs: vec![],
                        boundary_outputs: vec![],
                    };
                    let r = imp.resources(&m);
                    max_regs = max_regs.max(r.regs_per_thread);
                }
                if max_regs > *cap {
                    ConstraintStatus::Violated
                } else {
                    ConstraintStatus::Satisfied
                }
            }

            Constraint::CompilationUnitShmemBudget { unit, cap } => {
                let subgraphs = assignment.subgraphs_in_unit(*unit);
                if subgraphs.is_empty() {
                    return ConstraintStatus::Satisfied;
                }
                let mut max_shmem: u32 = 0;
                for sg in subgraphs {
                    let Some(impl_id) = assignment.impls.get(&sg) else {
                        return ConstraintStatus::Unknown;
                    };
                    let imp = library.get(*impl_id);
                    let claimed = assignment.tiles_in_subgraph(sg);
                    let m = crate::lowering::implementation::MatchInfo {
                        claimed_tiles: claimed,
                        boundary_inputs: vec![],
                        boundary_outputs: vec![],
                    };
                    let r = imp.resources(&m);
                    max_shmem = max_shmem.max(r.shmem_bytes);
                }
                if max_shmem > *cap {
                    ConstraintStatus::Violated
                } else {
                    ConstraintStatus::Satisfied
                }
            }

            Constraint::HandoffCompatible { producer, consumer } => {
                let Some(handoff) = assignment.handoffs.get(&(*producer, *consumer)) else {
                    return ConstraintStatus::Unknown;
                };
                let p_impl = assignment.impls.get(producer);
                let c_impl = assignment.impls.get(consumer);
                let (Some(p_impl), Some(c_impl)) = (p_impl, c_impl) else {
                    return ConstraintStatus::Unknown;
                };
                let p_imp = library.get(*p_impl);
                let c_imp = library.get(*c_impl);
                let p_supports = p_imp.supported_output_handoffs().contains(handoff);
                let c_supports = c_imp.supported_input_handoffs().contains(handoff);
                if !p_supports || !c_supports {
                    return ConstraintStatus::Violated;
                }
                // Target capability check: mbarrier requires sm_90+.
                if matches!(handoff, Handoff::Mbarrier)
                    && profile.lowering.mbarrier_handoff_us.is_none()
                {
                    return ConstraintStatus::Violated;
                }
                if matches!(handoff, Handoff::DsmemRead)
                    && profile.lowering.dsmem_cluster_handoff_us.is_none()
                {
                    return ConstraintStatus::Violated;
                }
                if matches!(handoff, Handoff::SyncThreads)
                    && profile.lowering.syncthreads_handoff_us.is_none()
                {
                    return ConstraintStatus::Violated;
                }
                ConstraintStatus::Satisfied
            }

            Constraint::IntermediateMaterialized { producer, consumer } => {
                // Both tiles must be assigned before we can check.
                let p_sg = assignment.cover.get(producer);
                let c_sg = assignment.cover.get(consumer);
                let (Some(p_sg), Some(c_sg)) = (p_sg, c_sg) else {
                    return ConstraintStatus::Unknown;
                };
                if p_sg == c_sg {
                    // Same subgraph — the impl handles the dep
                    // internally. No materialization needed.
                    return ConstraintStatus::Satisfied;
                }
                // Different subgraphs: the producer's output must be
                // in its subgraph's boundary_outputs. Check via the
                // impl's match info.
                let Some(p_impl_id) = assignment.impls.get(p_sg) else {
                    return ConstraintStatus::Unknown;
                };
                let p_imp = library.get(*p_impl_id);
                let seed = assignment
                    .tiles_in_subgraph(*p_sg)
                    .into_iter()
                    .min()
                    .unwrap();
                let Some(m) = p_imp.matches(tile_graph, seed, profile) else {
                    return ConstraintStatus::Violated;
                };
                if m.boundary_outputs.contains(producer) {
                    ConstraintStatus::Satisfied
                } else {
                    // The producer is consumed internally by the
                    // fusion but an external tile needs its output.
                    // This fusion is invalid for this cover.
                    ConstraintStatus::Violated
                }
            }

            Constraint::PrecisionBounded {
                producer,
                consumer,
                require_truncation,
            } => {
                if !require_truncation {
                    // No precision constraint on this edge.
                    return ConstraintStatus::Satisfied;
                }
                let p_sg = assignment.cover.get(producer);
                let c_sg = assignment.cover.get(consumer);
                let (Some(p_sg), Some(c_sg)) = (p_sg, c_sg) else {
                    return ConstraintStatus::Unknown;
                };
                if p_sg == c_sg {
                    // Same subgraph → Internal handoff → no
                    // truncation. If truncation is required, this
                    // fusion violates precision.
                    return ConstraintStatus::Violated;
                }
                // Different subgraphs → check the actual handoff.
                if let Some(handoff) = assignment.handoffs.get(&(*p_sg, *c_sg)) {
                    if handoff.truncates_to_storage_dtype() {
                        ConstraintStatus::Satisfied
                    } else {
                        ConstraintStatus::Violated
                    }
                } else {
                    // Handoff not yet assigned.
                    ConstraintStatus::Unknown
                }
            }
        }
    }
}

/// Suppress dead-code warnings for ImplId import. The constant is
/// referenced indirectly through library.get; this re-export keeps
/// the file's `use` block minimal.
#[allow(dead_code)]
type _ImplIdAlias = ImplId;
