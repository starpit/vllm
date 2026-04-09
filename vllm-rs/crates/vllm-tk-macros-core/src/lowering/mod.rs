// SPDX-License-Identifier: Apache-2.0
//! Joint constraint solver for lowering a reified DAG to a target.
//!
//! ## Why this module exists
//!
//! Lowering a model DAG to GPU code is a **joint constraint
//! problem**, not a sequence of independent decisions:
//!
//! 1. **Hardware constraints** (per [`crate::target_profile`]):
//!    register caps, shmem caps, barrier costs, launch costs,
//!    cooperative-launch exclusivity, capability flags
//!    (setmaxnreg, mbarrier, DSMEM, TMEM, ...).
//!
//! 2. **Data-dependency constraints** from the reified tile graph:
//!    every consumer must run after its producer's output is live;
//!    no two operations may write the same tile concurrently.
//!
//! 3. **Implementation constraints** from the curated kernel
//!    library: each [`Implementation`] (cuBLAS, CUTLASS multistage,
//!    CUTLASS sm_90 warp-specialized, ThunderKittens kernels,
//!    FlashInfer, vllm-rs fused ops, hand-written tile bodies, ...)
//!    matches a specific *subgraph pattern* of tiles, has its own
//!    cost / resource / launch-kind / handoff requirements, and is
//!    only valid on certain targets.
//!
//! These constraints **interlock**: picking a CUTLASS multistage
//! implementation for one subgraph constrains the layout of its
//! inputs, which constrains the implementation that produced those
//! inputs, which constrains the resource budget of the host kernel
//! they share. No choice can be made in isolation.
//!
//! The lowering solver makes one joint decision over the
//! [`Assignment`] tuple and emits an [`ExecutionPlan`] that the
//! backend turns into runnable code. Selection of an
//! `Implementation` per subgraph is the **output** of solving the
//! constraint problem, not the search axis.
//!
//! ## What's in this module
//!
//! - [`tile_graph`] — a normalized view of the reified DAG. Adds
//!   explicit residual_add, qkv_split, kv_cache_write, gate_up_concat
//!   nodes so the solver can reason about every operation as a
//!   first-class tile.
//!
//! - [`implementation`] — the [`Implementation`] trait + supporting
//!   types ([`LaunchKind`], [`Handoff`], [`Resources`], [`Layout`],
//!   [`MatchInfo`]).
//!
//! - [`library`] — the curated [`ImplementationLibrary`]. Initial
//!   entries cover what we already have linkable on L4 sm_89: cuBLAS
//!   GEMMs, vllm-rs fused norm/silu/rope, FlashInfer standalone, the
//!   existing CUTLASS small/big/narrow templates as DeviceCallable,
//!   and the hand-written norm/rope/mlp_norm tile bodies.
//!
//! - [`concurrency`] — the [`ConcurrencyModel`]: rules for what
//!   actually overlaps on the target hardware (e.g. compute-bound
//!   GEMMs on different streams don't speed up on sm_89).
//!
//! - [`constraint`] — the structured [`Constraint`] enum and the
//!   propagation rules. Constraints are *data*, not closures, so
//!   the same constraint set can be consumed by a CP backtracking
//!   solver today and an ILP encoder tomorrow.
//!
//! - [`assignment`] — the partial [`Assignment`] tuple (cover, impl,
//!   schedule, handoff, layout, compilation_unit) the solver builds
//!   incrementally.
//!
//! - [`cost`] — the wall-clock objective function over a complete
//!   assignment.
//!
//! - [`problem`] — bundles tile graph + library + profile + target
//!   constraint set into one [`Problem`] the solver consumes.
//!
//! - [`solver`] — the [`Solver`] trait. Initial implementation is
//!   constraint propagation + backtracking with branch-and-bound on
//!   the cost ([`solver::backtrack_cp`]). An ILP backend lives at
//!   [`solver::ilp`] as a stub for later, sharing the same
//!   `Problem` / `Assignment` / `Constraint` types.
//!
//! - [`backend`] — emits runnable code from a solved
//!   [`ExecutionPlan`]. The first backend ([`backend::runtime_ffi`])
//!   produces a Rust runtime function that calls into FFI for
//!   HostCallback / RegularLaunch / CooperativeLaunch implementations
//!   in dependency order.
//!
//! ## Status
//!
//! CP5-A: lays down the model types only. No solver, no backend.
//! Validation test hand-builds an [`Assignment`] equivalent to the
//! `cp4_natural_sm89_full_forward_microbench` and asserts every
//! constraint reports satisfied + the cost equals the measured
//! ~38 ms. CP5-B writes the CP backtracking solver against the same
//! types. CP5-C adds the runtime FFI backend. CP5-D extends the
//! library with CUTLASS prologue/epilogue specializations and TK
//! kittens entries so the solver finds richer fusions.

pub mod assignment;
pub mod concurrency;
pub mod constraint;
pub mod cost;
pub mod implementation;
pub mod library;
pub mod problem;
pub mod solver;
pub mod tile_graph;

pub use assignment::Assignment;
pub use concurrency::ConcurrencyModel;
pub use constraint::Constraint;
pub use cost::cost_us;
pub use implementation::{
    Handoff, ImplId, Implementation, LaunchKind, Layout, MatchInfo, Resources,
};
pub use library::ImplementationLibrary;
pub use problem::Problem;
pub use solver::{BacktrackCpSolver, ExecutionPlan, SolveResult, Solver};
pub use tile_graph::{TileGraph, TileId, TileKind};

#[cfg(test)]
mod tests;
