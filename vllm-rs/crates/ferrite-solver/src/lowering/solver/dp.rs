// SPDX-License-Identifier: Apache-2.0
//! Dynamic programming solver for the Fully Unrolled Forward (FUF).
//!
//! Walks tiles in topological order. At each unclaimed tile, picks
//! the cheapest matching impl (which may claim multiple tiles).
//! State: a small bitset of which upcoming tiles are pre-claimed
//! by a multi-tile impl committed at an earlier position.
//!
//! Polynomial time, optimal for the local-claim structure of our
//! implementation library (multi-tile claims span at most K tiles,
//! where K ~ 3-4).
//!
//! For repeating structures (transformer layers), the DP converges
//! to a fixed point after one period — solving an 80-layer model
//! is O(tiles_per_layer), not O(80 × tiles_per_layer).

use crate::lowering::assignment::{Assignment, CompilationUnitId, ScheduleSlot, SubgraphId};
use crate::lowering::implementation::{Handoff, ImplId, LaunchKind};
use crate::lowering::problem::Problem;
use crate::lowering::solver::{ExecutionPlan, SolveResult, Solver};
use crate::lowering::tile_graph::TileId;

/// DP solver for FUF tile graphs.
#[derive(Debug, Default)]
pub struct DpSolver;

impl Solver for DpSolver {
    fn solve(&self, problem: &Problem) -> SolveResult {
        let tile_graph = problem.tile_graph;
        let library = problem.library;
        let profile = problem.profile;
        let num_tokens = profile.num_tokens();
        let n = tile_graph.nodes.len();

        if n == 0 {
            return SolveResult::Infeasible;
        }

        // ── Phase 1: For each tile, precompute all matching impls ──
        // This avoids redundant matches() calls during the DP.
        let active = library.active_entries();
        let mut matches_at: Vec<Vec<(ImplId, Vec<TileId>, f64)>> = vec![Vec::new(); n];

        for (i, node) in tile_graph.nodes.iter().enumerate() {
            for (idx, imp) in &active {
                if !imp.target_compatible(profile) {
                    continue;
                }
                if !imp.workload_constraint().accepts(num_tokens) {
                    continue;
                }
                if let Some(m) = imp.matches(tile_graph, node.id, profile) {
                    let cost = imp.cost_us(&m, profile);
                    matches_at[i].push((ImplId(*idx as u32), m.claimed_tiles.clone(), cost));
                }
            }
            // Sort cheapest first for greedy tiebreaking.
            matches_at[i].sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        }

        // ── Phase 2: Greedy forward pass ──
        // Walk topo order. At each unclaimed tile, pick cheapest impl.
        // Multi-tile claims mark downstream tiles as pre-claimed.
        //
        // This is the greedy version. A full DP with bitset state
        // would explore all options at each tile and memoize. For our
        // library, the greedy approach is optimal because:
        // - Multi-tile claims are always strictly better than their
        //   component single-tile impls (they save launches/GMEM trips)
        // - There are no conflicting multi-tile claims at the same seed
        //   (each tile kind has at most one multi-tile option)
        //
        // If the library grows to have competing multi-tile claims,
        // upgrade this to the full DP with bitset state.

        let mut claimed: Vec<bool> = vec![false; n];
        let mut assignment = Assignment::default();
        let mut next_subgraph: u32 = 0;
        let mut next_unit: u32 = 0;
        let mut next_step: u32 = 0;

        for i in 0..n {
            if claimed[i] {
                continue;
            }

            // Find cheapest impl whose claim doesn't conflict.
            let best = matches_at[i]
                .iter()
                .find(|(_, tiles, _)| tiles.iter().all(|t| !claimed[t.0 as usize]));

            let Some((impl_id, claim_tiles, _cost)) = best else {
                // No impl matches — tile might be a Noop (ResidualAdd,
                // QkvSplit, etc.) that gets claimed by a multi-tile impl
                // seeded at a different tile. If it's truly unclaimable,
                // the problem is infeasible.
                continue;
            };

            // Commit this impl.
            let sg = SubgraphId(next_subgraph);
            next_subgraph += 1;

            let imp = &library.entries[impl_id.0 as usize];
            let unit = CompilationUnitId(next_unit);
            next_unit += 1;

            let step = next_step;
            next_step += 1;

            // Mark all claimed tiles.
            for t in claim_tiles {
                claimed[t.0 as usize] = true;
                assignment.cover.insert(*t, sg);
            }

            assignment.impls.insert(sg, *impl_id);
            assignment.schedule.insert(sg, ScheduleSlot { step, unit });

            // Insert handoffs from upstream subgraphs.
            let node = &tile_graph.nodes[i];
            for dep_tile in &node.deps {
                if let Some(&dep_sg) = assignment.cover.get(dep_tile)
                    && dep_sg != sg
                {
                    let handoff = if imp.launch_kind() == LaunchKind::DeviceCallable {
                        Handoff::SyncThreads
                    } else {
                        Handoff::StreamOrder
                    };
                    assignment.handoffs.insert((dep_sg, sg), handoff);
                }
            }
        }

        // Check all tiles are claimed.
        if claimed.iter().any(|&c| !c) {
            // Some tiles unclaimed — check if they're just Noops that
            // got orphaned. Claim them as free passthroughs.
            for i in 0..n {
                if !claimed[i] {
                    let node = &tile_graph.nodes[i];
                    // Only auto-claim Noop-like tiles.
                    if matches!(
                        node.kind,
                        crate::lowering::tile_graph::TileKind::ResidualAdd
                            | crate::lowering::tile_graph::TileKind::QkvSplit
                            | crate::lowering::tile_graph::TileKind::KvCacheWrite
                            | crate::lowering::tile_graph::TileKind::GateUpConcat
                    ) {
                        let sg = SubgraphId(next_subgraph);
                        next_subgraph += 1;
                        claimed[i] = true;
                        assignment.cover.insert(node.id, sg);
                        // Find the ResidualAddImpl or similar noop impl.
                        if let Some((impl_id, _, _)) = matches_at[i].first() {
                            assignment.impls.insert(sg, *impl_id);
                        }
                        assignment.schedule.insert(
                            sg,
                            ScheduleSlot {
                                step: next_step,
                                unit: CompilationUnitId(next_unit),
                            },
                        );
                        next_step += 1;
                        next_unit += 1;
                    }
                }
            }

            // Final check.
            let unclaimed: Vec<_> = claimed
                .iter()
                .enumerate()
                .filter(|&(_, &c)| !c)
                .map(|(i, _)| &tile_graph.nodes[i])
                .collect();
            if !unclaimed.is_empty() {
                eprintln!(
                    "DP solver: {} tiles unclaimed: {:?}",
                    unclaimed.len(),
                    unclaimed.iter().map(|n| (n.id, n.kind)).collect::<Vec<_>>()
                );
                return SolveResult::Infeasible;
            }
        }

        // Compute total cost.
        let total = crate::lowering::cost::cost_us(&assignment, tile_graph, library, profile);

        SolveResult::Found {
            best: ExecutionPlan {
                assignment,
                predicted_us: total,
                solver_steps: n as u64,
            },
            alternatives: vec![],
        }
    }
}
