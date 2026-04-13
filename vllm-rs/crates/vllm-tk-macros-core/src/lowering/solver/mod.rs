// SPDX-License-Identifier: Apache-2.0
//! [`Solver`] trait + concrete implementations.
//!
//! The solver consumes a [`Problem`] (tile graph + library +
//! profile + static constraints) and produces an [`ExecutionPlan`]:
//! a complete [`Assignment`] plus its predicted cost.
//!
//! Two backends share the same trait, the same `Problem` type, the
//! same `Assignment` shape, and the same constraint set:
//!
//! - **`backtrack_cp`** — constraint propagation + backtracking +
//!   branch-and-bound. The first-cut backend, written in pure Rust.
//!   Tractable for the L4 sm_89 starter library because the static
//!   constraints prune the search aggressively.
//!
//! - **`ilp`** — stub for later. Will encode the same `Constraint`
//!   variants as linear inequalities and call out to a MILP solver
//!   when (and if) the CP solver proves intractable on a richer
//!   library.
//!
//! Migrating from CP to ILP is a `Box<dyn Solver>` swap. The user
//! gave the call: ship CP first, shift to ILP if we need it.

pub mod backtrack_cp;
pub mod ilp;

use crate::lowering::assignment::Assignment;
use crate::lowering::problem::Problem;

pub use backtrack_cp::BacktrackCpSolver;

/// Outcome of a solver run.
#[derive(Clone, Debug)]
pub struct ExecutionPlan {
    /// The complete assignment the solver committed to.
    pub assignment: Assignment,
    /// Predicted wall-clock for this assignment, in microseconds.
    pub predicted_us: f64,
    /// Number of branch-and-bound steps the solver explored.
    pub solver_steps: u64,
}

/// Result of attempting to solve a [`Problem`].
#[derive(Debug)]
pub enum SolveResult {
    /// Solver found a feasible assignment. Optimal under the
    /// solver's search strategy (CP backtracking finds the
    /// minimum-cost solution at the search depth it explored).
    /// `alternatives` contains additional feasible plans ranked
    /// by cost (cheapest first), excluding the best plan.
    Found {
        best: ExecutionPlan,
        alternatives: Vec<ExecutionPlan>,
    },
    /// No feasible assignment exists. Either the problem is
    /// over-constrained or the library is missing required
    /// implementations.
    Infeasible,
}

/// Search strategy that produces an [`ExecutionPlan`] from a
/// [`Problem`]. Multiple implementations share this trait so the
/// caller can swap CP for ILP without touching the rest of the
/// pipeline.
pub trait Solver {
    /// Solve the given problem. Returns the optimal feasible
    /// assignment under this solver's search strategy, or
    /// `Infeasible` if none exists.
    fn solve(&self, problem: &Problem) -> SolveResult;
}

/// Pre-solved execution plans across a grid of sequence lengths.
///
/// The runtime indexes into this table by actual `seq_len` to get
/// the plan the solver discovered for that workload shape. Plans
/// are solved offline so the hot path is a table lookup, not a
/// solver invocation.
///
/// ```ignore
/// let family = PlanFamily::solve_grid(
///     &tile_graph, &library,
///     &TargetProfile::l4_sm89(),
///     &BacktrackCpSolver::default(),
/// );
/// let plan = family.lookup(actual_seq_len);
/// ```
#[derive(Clone, Debug)]
pub struct PlanFamily {
    /// Solved plans keyed by seq_len, sorted ascending.
    plans: Vec<(u32, ExecutionPlan)>,
}

impl PlanFamily {
    /// Default grid of sequence lengths to solve for.
    /// Covers decode (1), small-batch (2-32), and prefill (64-4096).
    pub const DEFAULT_GRID: &[u32] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

    /// Solve for every seq_len in the grid. Uses the base profile's
    /// `with_seq_len()` to vary the workload shape while keeping
    /// hardware constants fixed.
    pub fn solve_grid(
        tile_graph: &crate::lowering::tile_graph::TileGraph,
        library: &mut crate::lowering::library::ImplementationLibrary,
        base_profile: &crate::target_profile::TargetProfile,
        solver: &dyn Solver,
        grid: &[u32],
    ) -> Self {
        let mut plans = Vec::with_capacity(grid.len());
        for &seq in grid {
            let profile = base_profile.with_seq_len(seq);
            // Pre-select the cheapest CUTLASS config per GEMM phase
            // for this workload. This reduces the solver's branching
            // factor from ~60 to ~1 at each GEMM tile.
            library.pruned_for_workload(tile_graph, &profile);
            let problem = Problem::build(tile_graph, library, &profile);
            match solver.solve(&problem) {
                SolveResult::Found { best: plan, .. } => plans.push((seq, plan)),
                SolveResult::Infeasible => {
                    // Skip infeasible seq_lens (shouldn't happen with
                    // a well-formed library).
                }
            }
        }
        PlanFamily { plans }
    }

    /// Look up the plan for the closest seq_len <= `target`. If
    /// `target` is smaller than the smallest solved seq_len, returns
    /// the smallest. Returns `None` only if the family is empty.
    pub fn lookup(&self, target_seq_len: u32) -> Option<&ExecutionPlan> {
        if self.plans.is_empty() {
            return None;
        }
        // Find the largest seq_len <= target.
        let idx = self
            .plans
            .partition_point(|(seq, _)| *seq <= target_seq_len);
        let idx = if idx == 0 { 0 } else { idx - 1 };
        Some(&self.plans[idx].1)
    }

    /// Iterate all (seq_len, plan) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &ExecutionPlan)> {
        self.plans.iter().map(|(s, p)| (*s, p))
    }

    /// Number of plans in the family.
    pub fn len(&self) -> usize {
        self.plans.len()
    }

    /// Whether the family is empty.
    pub fn is_empty(&self) -> bool {
        self.plans.is_empty()
    }
}
