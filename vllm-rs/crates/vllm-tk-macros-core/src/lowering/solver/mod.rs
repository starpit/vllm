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
    Found(ExecutionPlan),
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
