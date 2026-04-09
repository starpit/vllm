// SPDX-License-Identifier: Apache-2.0
//! Constraint propagation + backtracking solver with branch-and-bound.
//!
//! ## Algorithm
//!
//! Walk tiles in topological order (the input `TileGraph::nodes`
//! are already topo-sorted). At each unclaimed tile:
//!
//!   1. Enumerate every implementation in the library that
//!      `matches(seed)` and is `target_compatible(profile)`.
//!   2. Sort the matches by the implementation's `cost_us`
//!      (cheapest first — branch-and-bound benefits from finding
//!      good solutions early).
//!   3. For each match, in cheapest-first order: tentatively
//!      commit (allocate fresh SubgraphId / unit / step + claim
//!      tiles + insert handoffs to upstream subgraphs along the
//!      new claim's boundary), re-evaluate every static
//!      constraint, prune via cost lower bound, recurse on
//!      success, undo on failure.
//!   4. If no match feasibly completes, backtrack one level.
//!
//! When all tiles are claimed and every constraint reports
//! `Satisfied`, the assignment is complete and feasible. The
//! cheapest such assignment found is returned.
//!
//! ## Why this is OK for first cut
//!
//! On the L4 sm_89 starter library each tile kind has exactly one
//! or two matching impls. The branching factor at each tile is
//! ~1.5 on average. With aggressive cost-ordered branching and
//! constraint propagation, the search collapses to essentially
//! deterministic walking — milliseconds for the 16-layer Llama-1B
//! production tile graph.
//!
//! When the library grows (TK kittens fused gate-up,
//! CUTLASS prologue/epilogue specializations, sm_90 entries,
//! cubecl), branching grows but constraint propagation tightens.
//! If at some library size the CP solver becomes slow, the ILP
//! backend takes over with no other code changes.
//!
//! ## Multi-tile claims
//!
//! When an implementation matches a multi-tile subgraph (e.g.
//! `VllmRsSiluAndMulFusedImpl` claims `GateUpConcat + SiluMul`
//! together), the solver commits ALL claimed tiles under one
//! `SubgraphId`. Later in the topological walk, when the solver
//! reaches the second tile of the multi-tile claim, it sees the
//! tile is already claimed and skips it.

use crate::lowering::assignment::{Assignment, CompilationUnitId, ScheduleSlot, SubgraphId};
use crate::lowering::constraint::ConstraintStatus;
use crate::lowering::cost::cost_us;
use crate::lowering::implementation::{Handoff, ImplId};
use crate::lowering::problem::Problem;
use crate::lowering::solver::{ExecutionPlan, SolveResult, Solver};
use crate::lowering::tile_graph::TileId;

/// Constraint propagation + backtracking + branch-and-bound solver.
#[derive(Debug, Default)]
pub struct BacktrackCpSolver;

impl Solver for BacktrackCpSolver {
    fn solve(&self, problem: &Problem) -> SolveResult {
        let mut state = SearchState::new(problem);
        let starting_step: u32 = 0;
        recurse(&mut state, starting_step);
        match state.best {
            Some((assignment, predicted_us, steps)) => SolveResult::Found(ExecutionPlan {
                assignment,
                predicted_us,
                solver_steps: steps,
            }),
            None => SolveResult::Infeasible,
        }
    }
}

/// Search state carried through recursion. The current partial
/// `Assignment` is mutable; we make changes, recurse, undo on
/// backtrack.
struct SearchState<'a> {
    problem: &'a Problem<'a>,
    /// Current partial assignment.
    assignment: Assignment,
    /// Next subgraph id to allocate.
    next_subgraph: u32,
    /// Next compilation unit id to allocate.
    next_unit: u32,
    /// Next schedule step to allocate. The natural-sm89 lowering
    /// uses one step per subgraph (sequential, no overlap); CP5-D
    /// will introduce concurrent steps when DeviceCallable impls
    /// can share a step.
    next_step: u32,
    /// Best (assignment, cost, steps) found so far.
    best: Option<(Assignment, f64, u64)>,
    /// Total branch-and-bound steps explored. Used for diagnostics.
    steps: u64,
}

impl<'a> SearchState<'a> {
    fn new(problem: &'a Problem<'a>) -> Self {
        Self {
            problem,
            assignment: Assignment::default(),
            next_subgraph: 0,
            next_unit: 0,
            next_step: 0,
            best: None,
            steps: 0,
        }
    }

    /// Find the next unclaimed tile in topological order, or
    /// `None` if every tile is covered.
    fn next_unclaimed(&self) -> Option<TileId> {
        for node in self.problem.tile_graph.iter_topo() {
            if !self.assignment.cover.contains_key(&node.id) {
                return Some(node.id);
            }
        }
        None
    }

    /// Compute a lower bound on the final cost given the current
    /// partial assignment. For first cut: the cost of the bound
    /// portion only (no estimate of the unbound portion). The
    /// bound monotonically increases as the solver commits more,
    /// so branch-and-bound still prunes correctly — we may just
    /// explore more branches than a tighter bound would.
    fn lower_bound_cost_us(&self) -> f64 {
        cost_us(
            &self.assignment,
            self.problem.tile_graph,
            self.problem.library,
            self.problem.profile,
        )
    }

    /// Check every static constraint against the current partial
    /// assignment. Returns `true` if no constraint is `Violated`
    /// (Unknown is acceptable for partial assignments).
    fn no_constraint_violated(&self) -> bool {
        for c in &self.problem.static_constraints {
            let status = c.check(
                &self.assignment,
                self.problem.tile_graph,
                self.problem.library,
                self.problem.profile,
            );
            if matches!(status, ConstraintStatus::Violated) {
                return false;
            }
        }
        true
    }
}

/// Snapshot we take before tentatively committing a claim, so we
/// can undo cleanly on backtrack. Holds only the keys we mutated.
#[derive(Debug)]
struct CommitSnapshot {
    cover_keys: Vec<TileId>,
    impls_key: SubgraphId,
    schedule_key: SubgraphId,
    handoff_keys: Vec<(SubgraphId, SubgraphId)>,
    layouts_keys: Vec<TileId>,
    /// Saved counters so we can roll them back.
    saved_next_subgraph: u32,
    saved_next_unit: u32,
    saved_next_step: u32,
}

/// Recursive search step. Picks the next unclaimed tile, tries
/// every matching implementation in cheapest-first order, recurses
/// on each feasible commit. Backtracks on infeasibility or when
/// the lower-bound cost exceeds the best known.
fn recurse(state: &mut SearchState<'_>, _starting_step: u32) {
    state.steps += 1;

    // Termination: every tile claimed → check completeness + record.
    let Some(seed) = state.next_unclaimed() else {
        // Cover complete. Run a full constraint check (anything
        // that was Unknown during partial walks must now resolve).
        if !state.no_constraint_violated() {
            return;
        }
        // Compute final cost and record if it's the new best.
        let total = cost_us(
            &state.assignment,
            state.problem.tile_graph,
            state.problem.library,
            state.problem.profile,
        );
        let is_new_best = match state.best.as_ref() {
            None => true,
            Some((_, cur, _)) => total < *cur,
        };
        if is_new_best {
            state.best = Some((state.assignment.clone(), total, state.steps));
        }
        return;
    };

    // Branch-and-bound: prune if our partial cost already exceeds
    // the current best.
    let lb = state.lower_bound_cost_us();
    if let Some((_, best_cost, _)) = state.best.as_ref()
        && lb >= *best_cost
    {
        return;
    }

    // Enumerate matches at this seed across the whole library.
    let library_entries = &state.problem.library.entries;
    let profile = state.problem.profile;
    let tile_graph = state.problem.tile_graph;
    let mut candidates: Vec<(ImplId, crate::lowering::implementation::MatchInfo, f64)> = Vec::new();
    for (idx, imp) in library_entries.iter().enumerate() {
        if !imp.target_compatible(profile) {
            continue;
        }
        let Some(m) = imp.matches(tile_graph, seed, profile) else {
            continue;
        };
        // Skip matches whose claimed tiles are already partially
        // claimed by something else (would conflict with the
        // existing cover).
        let conflict = m
            .claimed_tiles
            .iter()
            .any(|t| state.assignment.cover.contains_key(t));
        if conflict {
            continue;
        }
        let cost = imp.cost_us(&m, profile);
        candidates.push((ImplId(idx as u32), m, cost));
    }
    // Cheapest first → branch-and-bound finds good solutions early.
    candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    for (impl_id, match_info, _cost) in candidates {
        let snap = commit_match(state, impl_id, &match_info);

        // Constraint propagation after the commit.
        if state.no_constraint_violated() {
            recurse(state, _starting_step);
        }

        rollback(state, snap);
    }
}

/// Tentatively commit one (impl, match) into the assignment.
/// Allocates a fresh subgraph id, compilation unit, and step.
/// Returns a snapshot of the keys that were added so the caller
/// can roll them back on backtrack.
fn commit_match(
    state: &mut SearchState<'_>,
    impl_id: ImplId,
    match_info: &crate::lowering::implementation::MatchInfo,
) -> CommitSnapshot {
    let saved_next_subgraph = state.next_subgraph;
    let saved_next_unit = state.next_unit;
    let saved_next_step = state.next_step;

    let sg = SubgraphId(state.next_subgraph);
    state.next_subgraph += 1;
    let unit = CompilationUnitId(state.next_unit);
    state.next_unit += 1;
    let step = state.next_step;
    state.next_step += 1;

    // 1. cover: claim all tiles in the match.
    let mut cover_keys = Vec::with_capacity(match_info.claimed_tiles.len());
    for t in &match_info.claimed_tiles {
        state.assignment.cover.insert(*t, sg);
        cover_keys.push(*t);
    }

    // 2. impls: subgraph → impl.
    state.assignment.impls.insert(sg, impl_id);

    // 3. schedule: assign step + unit.
    state
        .assignment
        .schedule
        .insert(sg, ScheduleSlot { step, unit });

    // 4. layouts: assign output layouts for the impl's outputs.
    let imp = state.problem.library.get(impl_id);
    let output_layouts = imp.output_layouts(match_info);
    let mut layouts_keys = Vec::new();
    for (t, layout) in match_info
        .boundary_outputs
        .iter()
        .zip(output_layouts.iter())
    {
        state.assignment.layouts.insert(*t, *layout);
        layouts_keys.push(*t);
    }

    // 5. handoffs: insert StreamOrder handoffs from each upstream
    // producer subgraph that this claim depends on. The first
    // valid handoff in the producer's supported_output_handoffs ∩
    // consumer's supported_input_handoffs wins; for the L4 sm_89
    // starter library StreamOrder is universally supported.
    let mut handoff_keys = Vec::new();
    let mut seen_producers: std::collections::HashSet<SubgraphId> = Default::default();
    for input_tile in &match_info.boundary_inputs {
        if let Some(producer_sg) = state.assignment.cover.get(input_tile).copied()
            && seen_producers.insert(producer_sg)
            && producer_sg != sg
        {
            let producer_imp = state
                .problem
                .library
                .get(state.assignment.impls[&producer_sg]);
            let supported = pick_handoff(
                producer_imp.supported_output_handoffs(),
                imp.supported_input_handoffs(),
            );
            if let Some(h) = supported {
                state.assignment.handoffs.insert((producer_sg, sg), h);
                handoff_keys.push((producer_sg, sg));
            }
        }
    }

    CommitSnapshot {
        cover_keys,
        impls_key: sg,
        schedule_key: sg,
        handoff_keys,
        layouts_keys,
        saved_next_subgraph,
        saved_next_unit,
        saved_next_step,
    }
}

fn rollback(state: &mut SearchState<'_>, snap: CommitSnapshot) {
    for t in snap.cover_keys {
        state.assignment.cover.remove(&t);
    }
    state.assignment.impls.remove(&snap.impls_key);
    state.assignment.schedule.remove(&snap.schedule_key);
    for k in snap.handoff_keys {
        state.assignment.handoffs.remove(&k);
    }
    for t in snap.layouts_keys {
        state.assignment.layouts.remove(&t);
    }
    state.next_subgraph = snap.saved_next_subgraph;
    state.next_unit = snap.saved_next_unit;
    state.next_step = snap.saved_next_step;
}

/// Pick the cheapest handoff that's in both supported sets. For
/// the L4 sm_89 starter library this returns `StreamOrder` for
/// every host-callback ↔ host-callback edge.
fn pick_handoff(producer_out: &[Handoff], consumer_in: &[Handoff]) -> Option<Handoff> {
    // Preference order — cheapest first.
    const PREFERENCES: [Handoff; 8] = [
        Handoff::Internal,
        Handoff::StreamOrder,
        Handoff::GmemFlag,
        Handoff::Mbarrier,
        Handoff::DsmemRead,
        Handoff::StreamEvent,
        Handoff::KernelBoundary,
        Handoff::InKernelGridSync,
    ];
    PREFERENCES
        .into_iter()
        .find(|h| producer_out.contains(h) && consumer_in.contains(h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lowering::library::ImplementationLibrary;
    use crate::lowering::tile_graph::{TileGraph, TileKind};
    use crate::target_profile::TargetProfile;

    #[test]
    fn solver_finds_natural_sm89_assignment() {
        let tile_graph = TileGraph::build_llama_forward(16);
        let library = ImplementationLibrary::l4_sm89_starter();
        let profile = TargetProfile::l4_sm89();
        let problem = Problem::build(&tile_graph, &library, &profile);

        let solver = BacktrackCpSolver;
        let result = solver.solve(&problem);

        let plan = match result {
            SolveResult::Found(p) => p,
            SolveResult::Infeasible => panic!("solver reported infeasible"),
        };

        eprintln!(
            "solver: {} steps, predicted {:.2} ms",
            plan.solver_steps,
            plan.predicted_us / 1000.0
        );

        // Sanity: every tile claimed.
        assert!(plan.assignment.is_cover_complete(tile_graph.len()));

        // Sanity: predicted cost is in the natural-sm89 ballpark
        // (the hand-built reference is ~43 ms).
        let pred_ms = plan.predicted_us / 1000.0;
        assert!(
            (40.0..50.0).contains(&pred_ms),
            "predicted {pred_ms} ms outside expected 40-50 ms band",
        );
    }

    #[test]
    fn solver_picks_silu_and_mul_fused_for_gate_up_concat() {
        // Verify the multi-tile claim works: the solver should
        // discover that vllm_rs_silu_and_mul_fused claims both
        // GateUpConcat AND SiluMul under one subgraph.
        let tile_graph = TileGraph::build_llama_forward(2);
        let library = ImplementationLibrary::l4_sm89_starter();
        let profile = TargetProfile::l4_sm89();
        let problem = Problem::build(&tile_graph, &library, &profile);

        let solver = BacktrackCpSolver;
        let result = solver.solve(&problem);
        let plan = match result {
            SolveResult::Found(p) => p,
            SolveResult::Infeasible => panic!("infeasible"),
        };

        // Find a GateUpConcat tile and check that its subgraph
        // also claims a SiluMul tile in the same layer.
        let concat_tile = tile_graph
            .iter_topo()
            .find(|n| n.kind == TileKind::GateUpConcat)
            .expect("test graph has GateUpConcat");
        let concat_sg = plan.assignment.cover[&concat_tile.id];
        let claimed = plan.assignment.tiles_in_subgraph(concat_sg);
        let claimed_kinds: Vec<TileKind> = claimed
            .iter()
            .map(|t| tile_graph.nodes[t.0 as usize].kind)
            .collect();
        assert!(
            claimed_kinds.contains(&TileKind::GateUpConcat)
                && claimed_kinds.contains(&TileKind::SiluMul),
            "expected the SiluAndMulFused subgraph to claim both \
             GateUpConcat AND SiluMul; claimed kinds = {claimed_kinds:?}",
        );
        let imp_id = plan.assignment.impls[&concat_sg];
        let imp_name = library.get(imp_id).name();
        assert_eq!(imp_name, "vllm_rs_silu_and_mul_fused");
    }
}
