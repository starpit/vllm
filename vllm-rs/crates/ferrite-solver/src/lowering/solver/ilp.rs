// SPDX-License-Identifier: Apache-2.0
//! Stub ILP backend.
//!
//! Reserved for the day the CP backtracking solver proves
//! intractable. The plan: encode each `Constraint` variant as a
//! linear inequality over binary indicator variables (`x_{tile,
//! impl} = 1` if tile is claimed by impl), encode the cost
//! function linearly, hand off to a MILP solver crate
//! (`good_lp` / `russcip` / similar), decode the result back into
//! an `Assignment`.
//!
//! Sharing the same `Problem` and `Assignment` types as the CP
//! backend means the rest of the pipeline doesn't change when we
//! flip backends.

use crate::lowering::problem::Problem;
use crate::lowering::solver::{SolveResult, Solver};

/// ILP solver backend. Not implemented yet — falls through to a
/// panic so calling code that mis-routes to the ILP backend
/// doesn't silently get nothing back.
#[derive(Debug, Default)]
pub struct IlpSolver;

impl Solver for IlpSolver {
    fn solve(&self, _problem: &Problem) -> SolveResult {
        unimplemented!(
            "ILP backend not yet implemented; use BacktrackCpSolver. \
             The Problem / Assignment / Constraint types are designed \
             so the same set works for both CP and ILP — switch \
             backends with a Box<dyn Solver> swap when CP runs out \
             of headroom."
        );
    }
}
