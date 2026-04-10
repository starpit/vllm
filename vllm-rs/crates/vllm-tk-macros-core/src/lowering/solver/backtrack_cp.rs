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

use std::collections::HashMap;

use crate::lowering::assignment::{Assignment, CompilationUnitId, ScheduleSlot, SubgraphId};
use crate::lowering::constraint::ConstraintStatus;
use crate::lowering::cost::cost_us;
use crate::lowering::implementation::{Handoff, ImplId};
use crate::lowering::problem::Problem;
use crate::lowering::solver::{ExecutionPlan, SolveResult, Solver};
use crate::lowering::tile_graph::{TileId, TileKind};

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

/// Precomputed minimum cost of the cheapest impl that can match a
/// representative tile of each `TileKind`. Used to compute an
/// optimistic-but-tight remainder bound during branch-and-bound.
///
/// **Multi-tile attribution**: a multi-tile claim's full cost is
/// attributed to the *lowest-id* tile in the claim, with all other
/// claimed tiles getting 0 in the per-tile min. This way, summing
/// per-tile minima for an empty partial yields exactly the actual
/// cost of using each impl (not a divided-by-N underestimate that
/// fails to prune anything). The branch-and-bound bound becomes
/// tight enough to prune the (oproj/down) × 16 layers exponential
/// branching from `CublasGemmExWithResidualImpl`.
fn min_cost_per_tile_kind(problem: &Problem) -> HashMap<TileKind, f64> {
    let mut min_costs: HashMap<TileKind, f64> = HashMap::new();
    let kinds_present: std::collections::HashSet<TileKind> =
        problem.tile_graph.nodes.iter().map(|n| n.kind).collect();
    for kind in kinds_present {
        let representative = problem
            .tile_graph
            .iter_topo()
            .find(|n| n.kind == kind)
            .map(|n| n.id);
        let Some(seed) = representative else {
            continue;
        };
        let mut min_cost = f64::INFINITY;
        for imp in &problem.library.entries {
            if !imp.target_compatible(problem.profile) {
                continue;
            }
            if let Some(m) = imp.matches(problem.tile_graph, seed, problem.profile) {
                let c = imp.cost_us(&m, problem.profile);
                // Attribute the full multi-tile claim cost to its
                // SEED (the lowest-id claimed tile). For the seed's
                // kind that's the full cost; other tiles in the
                // claim contribute 0. This makes the sum bound
                // exact for any cover that uses this impl, which
                // is what enables branch-and-bound pruning.
                let seed_id = m.claimed_tiles.iter().copied().min().unwrap();
                let seed_kind = problem.tile_graph.nodes[seed_id.0 as usize].kind;
                let attributed = if seed_kind == kind { c } else { 0.0 };
                if attributed < min_cost {
                    min_cost = attributed;
                }
            }
        }
        // If no impl matches this kind directly with the kind being
        // the seed (e.g. SiluMul is always claimed by a fused impl
        // seeded at GateUpConcat), the per-tile contribution for
        // this kind is 0 — its cost gets attributed to the seed.
        if !min_cost.is_finite() {
            min_cost = 0.0;
        }
        min_costs.insert(kind, min_cost);
    }
    min_costs
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
    /// Precomputed cheapest cost per `TileKind`. Used by the
    /// optimistic remainder bound.
    min_cost_per_kind: HashMap<TileKind, f64>,
}

impl<'a> SearchState<'a> {
    fn new(problem: &'a Problem<'a>) -> Self {
        let min_cost_per_kind = min_cost_per_tile_kind(problem);
        Self {
            problem,
            assignment: Assignment::default(),
            next_subgraph: 0,
            next_unit: 0,
            next_step: 0,
            best: None,
            steps: 0,
            min_cost_per_kind,
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

    /// Lower bound on the final cost: partial cost (committed
    /// portion) + optimistic remainder (sum of cheapest impl per
    /// unclaimed tile). The remainder ignores constraints and
    /// fusion structure; it's a true lower bound that lets the
    /// solver prune any branch whose `lb >= best_known_cost`.
    fn lower_bound_cost_us(&self) -> f64 {
        let partial = cost_us(
            &self.assignment,
            self.problem.tile_graph,
            self.problem.library,
            self.problem.profile,
        );
        let mut remainder = 0.0;
        for node in self.problem.tile_graph.iter_topo() {
            if self.assignment.cover.contains_key(&node.id) {
                continue;
            }
            // Add the cheapest possible cost for this tile kind.
            // If unknown, fall back to 0 (won't over-prune).
            let m = self
                .min_cost_per_kind
                .get(&node.kind)
                .copied()
                .unwrap_or(0.0);
            remainder += m;
        }
        partial + remainder
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
    // Sort by amortized cost per claimed tile: a 1435µs 7-tile fused
    // claim (205µs/tile) beats a 15µs 1-tile standalone (15µs/tile)
    // at the global level even though its raw cost is higher. This
    // makes B&B find multi-tile claims early instead of greedily
    // picking standalone impls that commit the remaining tiles to
    // more expensive individual covers.
    candidates.sort_by(|a, b| {
        let a_per = a.2 / a.1.claimed_tiles.len() as f64;
        let b_per = b.2 / b.1.claimed_tiles.len() as f64;
        a_per
            .partial_cmp(&b_per)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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
    use crate::lowering::implementation::LaunchKind;
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
        let mut impl_counts: std::collections::BTreeMap<&str, u32> = Default::default();
        for sg in plan.assignment.subgraphs() {
            let name = library.get(plan.assignment.impls[&sg]).name();
            *impl_counts.entry(name).or_insert(0) += 1;
        }
        for (name, count) in &impl_counts {
            eprintln!("  {count:>3} × {name}");
        }

        // Sanity: every tile claimed.
        assert!(plan.assignment.is_cover_complete(tile_graph.len()));

        // Sanity: predicted cost is in a reasonable ballpark.
        // The analytical model (roofline) predicts ~31 ms; measured
        // is ~40 ms (the model is optimistic — doesn't account for
        // memory latency, occupancy limits, cuBLAS overhead).
        let pred_ms = plan.predicted_us / 1000.0;
        assert!(
            (25.0..50.0).contains(&pred_ms),
            "predicted {pred_ms} ms outside expected 25-50 ms band",
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

    #[test]
    fn solver_picks_tk_fused_mlp_for_decode() {
        // At seq=1 (decode), the TK fused MLP block should beat
        // separate cuBLAS calls because the fused pipeline saves
        // GMEM round-trips and launch overhead that dominate at
        // small batch sizes.
        let tile_graph = TileGraph::build_llama_forward(2);
        let library = ImplementationLibrary::l4_sm89_starter();
        let profile = TargetProfile::l4_sm89().with_seq_len(1);
        let problem = Problem::build(&tile_graph, &library, &profile);

        let solver = BacktrackCpSolver;
        let plan = match solver.solve(&problem) {
            SolveResult::Found(p) => p,
            SolveResult::Infeasible => panic!("infeasible"),
        };

        eprintln!(
            "decode solver: {} steps, predicted {:.2} ms",
            plan.solver_steps,
            plan.predicted_us / 1000.0
        );
        let mut impl_counts: std::collections::BTreeMap<&str, u32> = Default::default();
        for sg in plan.assignment.subgraphs() {
            let name = library.get(plan.assignment.impls[&sg]).name();
            *impl_counts.entry(name).or_insert(0) += 1;
        }
        for (name, count) in &impl_counts {
            eprintln!("  {count:>3} × {name}");
        }

        assert!(
            impl_counts.contains_key("tk_fused_mlp_block"),
            "expected solver to pick tk_fused_mlp_block for decode (seq=1)"
        );
    }

    /// Short abbreviation for a TileKind.
    fn tile_abbrev(kind: TileKind) -> &'static str {
        match kind {
            TileKind::RmsNorm => "norm",
            TileKind::GemmQkv => "qkv",
            TileKind::GemmOProj => "oproj",
            TileKind::GemmGate => "gate",
            TileKind::GemmUp => "up",
            TileKind::GemmDown => "down",
            TileKind::QkvSplit => "split",
            TileKind::Rope => "rope",
            TileKind::KvCacheWrite => "kv_w",
            TileKind::Attention => "attn",
            TileKind::ResidualAdd => "res",
            TileKind::GateUpConcat => "cat",
            TileKind::SiluMul => "silu",
        }
    }

    /// ANSI color for each library family.
    fn lib_color(imp_name: &str) -> &'static str {
        if imp_name.starts_with("cutlass") {
            "\x1b[31m" // red
        } else if imp_name.starts_with("cublas") {
            "\x1b[36m" // cyan
        } else if imp_name.starts_with("tk_") {
            "\x1b[33m" // yellow
        } else if imp_name.starts_with("flashinfer") {
            "\x1b[35m" // magenta
        } else if imp_name.starts_with("vllm_rs")
            || imp_name == "residual_add"
            || imp_name == "kv_cache_write"
            || imp_name == "qkv_split_free"
        {
            "\x1b[32m" // green
        } else {
            "\x1b[37m" // white/default
        }
    }
    const RESET: &str = "\x1b[0m";

    /// Short library tag from impl name. Includes tile size for CUTLASS.
    fn lib_tag(imp_name: &str) -> &'static str {
        if imp_name.starts_with("cutlass") && imp_name.contains("64x64") {
            "cl64"
        } else if imp_name.starts_with("cutlass") {
            "cl128"
        } else if imp_name.starts_with("cublas") {
            "cb"
        } else if imp_name.starts_with("tk_") {
            "tk"
        } else if imp_name.starts_with("flashinfer") {
            "fi"
        } else if imp_name.starts_with("vllm_rs")
            || imp_name == "residual_add"
            || imp_name == "kv_cache_write"
            || imp_name == "qkv_split_free"
        {
            "vr"
        } else {
            "??"
        }
    }

    /// Render one launch: `cb:l(gate)` or `tk:l([norm+gate+up+...])`.
    /// Color-coded by library family.
    fn launch_str(imp_name: &str, kinds: &[TileKind]) -> String {
        let tag = lib_tag(imp_name);
        let color = lib_color(imp_name);
        let tiles = if kinds.len() == 1 {
            tile_abbrev(kinds[0]).to_string()
        } else {
            let inner: String = kinds
                .iter()
                .map(|k| tile_abbrev(*k))
                .collect::<Vec<_>>()
                .join("+");
            format!("[{inner}]")
        };
        format!("{color}{tag}:{tiles}{RESET}")
    }

    /// Compact one-layer representation: sequence of `tag:tiles` tokens.
    fn plan_one_layer_compact(
        plan: &ExecutionPlan,
        tile_graph: &TileGraph,
        library: &ImplementationLibrary,
        layer: u16,
    ) -> String {
        let mut scheduled: Vec<_> = plan
            .assignment
            .schedule
            .iter()
            .map(|(sg, slot)| (slot.step, *sg))
            .collect();
        scheduled.sort_by_key(|(step, sg)| (*step, sg.0));

        let mut parts = Vec::new();
        for (_step, sg) in &scheduled {
            let claimed = plan.assignment.tiles_in_subgraph(*sg);
            let sg_layer = claimed
                .iter()
                .map(|t| tile_graph.nodes[t.0 as usize].layer)
                .next()
                .unwrap_or(0);
            if sg_layer != layer {
                continue;
            }
            let imp_name = library.get(plan.assignment.impls[sg]).name();
            let kinds: Vec<_> = claimed
                .iter()
                .map(|t| tile_graph.nodes[t.0 as usize].kind)
                .collect();
            parts.push(launch_str(imp_name, &kinds));
        }
        parts.join(" ")
    }

    #[test]
    fn print_plan_family_compact() {
        let tg = TileGraph::build_llama_forward(2);
        let library = ImplementationLibrary::l4_sm89_starter();
        let base = TargetProfile::l4_sm89();

        eprintln!();
        eprintln!("LLaMA 1B on L4 sm_89 — solver-discovered plans (one layer, 2-layer model)");
        eprintln!("tag:tiles = one kernel launch, [a+b] = fused tiles");
        eprintln!(
            "\x1b[31mcl128\x1b[0m=CUTLASS \x1b[36mcb\x1b[0m=cuBLAS \x1b[33mtk\x1b[0m=TK \x1b[35mfi\x1b[0m=FlashInfer \x1b[32mvr\x1b[0m=vllm-rs"
        );

        // Workload grid: (label, batch_size, seq_len)
        let workloads: &[(&str, u32, u32)] = &[
            // ── Decode (seq=1, varying BS) ──
            ("decode BS=1", 1, 1),
            ("decode BS=4", 4, 1),
            ("decode BS=16", 16, 1),
            ("decode BS=32", 32, 1),
            ("decode BS=64", 64, 1),
            ("decode BS=128", 128, 1),
            ("decode BS=256", 256, 1),
            // ── Prefill (BS=1, varying seq) ──
            ("prefill seq=64", 1, 64),
            ("prefill seq=128", 1, 128),
            ("prefill seq=256", 1, 256),
            ("prefill seq=512", 1, 512),
            ("prefill seq=1024", 1, 1024),
            ("prefill seq=4096", 1, 4096),
        ];

        eprintln!();
        eprintln!(
            "{:<20} │ {:>4} │ {:>7} │ launches",
            "workload", "M", "pred_ms"
        );
        eprintln!(
            "─────────────────────┼──────┼─────────┼─{}─",
            "─".repeat(75)
        );

        for &(label, bs, seq) in workloads {
            let profile = base.with_workload(bs, seq);
            let m = profile.num_tokens();
            let problem = Problem::build(&tg, &library, &profile);
            let plan = match BacktrackCpSolver.solve(&problem) {
                SolveResult::Found(p) => p,
                _ => {
                    eprintln!("{label:<20} │ {m:>4} │  INFEAS │");
                    continue;
                }
            };
            let compact = plan_one_layer_compact(&plan, &tg, &library, 1);
            let pred = plan.predicted_us / 1000.0;
            eprintln!("{label:<20} │ {m:>4} │ {pred:>7.2} │ {compact}");
        }
        eprintln!();
    }

    #[test]
    fn plan_family_across_seq_lens() {
        let tg = TileGraph::build_llama_forward(2);
        let library = ImplementationLibrary::l4_sm89_starter();
        let profile = TargetProfile::l4_sm89();

        let family = super::super::PlanFamily::solve_grid(
            &tg,
            &library,
            &profile,
            &BacktrackCpSolver,
            super::super::PlanFamily::DEFAULT_GRID,
        );

        eprintln!();
        eprintln!("╔══ Plan Family (2 layers, L4 sm_89) ══════════════════════════╗");
        eprintln!("║  seq_len │ predicted │ steps │ tk_fused_mlp │ cublas_gate   ║");
        eprintln!("╠──────────┼───────────┼───────┼──────────────┼──────────────╣");
        for (seq, plan) in family.iter() {
            let mut tk_count = 0u32;
            let mut gate_count = 0u32;
            for sg in plan.assignment.subgraphs() {
                let name = library.get(plan.assignment.impls[&sg]).name();
                if name == "tk_fused_mlp_block" {
                    tk_count += 1;
                }
                if name == "cublas_gemm_ex_gate" {
                    gate_count += 1;
                }
            }
            eprintln!(
                "║  {:>6} │ {:>7.2} ms│  {:>4} │ {:>12} │ {:>12} ║",
                seq,
                plan.predicted_us / 1000.0,
                plan.solver_steps,
                tk_count,
                gate_count,
            );
        }
        eprintln!("╚══════════════════════════════════════════════════════════════╝");

        assert_eq!(family.len(), super::super::PlanFamily::DEFAULT_GRID.len());

        // At seq=1, solver should pick TK fused MLP.
        let decode_plan = family.lookup(1).unwrap();
        let has_tk = decode_plan.assignment.subgraphs().any(|sg| {
            library.get(decode_plan.assignment.impls[&sg]).name() == "tk_fused_mlp_block"
        });
        assert!(has_tk, "decode plan should use tk_fused_mlp_block");

        // At seq=1024, solver should NOT pick TK fused MLP.
        let prefill_plan = family.lookup(1024).unwrap();
        let has_tk = prefill_plan.assignment.subgraphs().any(|sg| {
            library.get(prefill_plan.assignment.impls[&sg]).name() == "tk_fused_mlp_block"
        });
        assert!(!has_tk, "prefill plan should not use tk_fused_mlp_block");
    }
}
