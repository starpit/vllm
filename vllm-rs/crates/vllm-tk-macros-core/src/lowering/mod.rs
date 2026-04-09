// SPDX-License-Identifier: Apache-2.0
//! Wave-grouping solver and reified `ExecutionPlan`.
//!
//! Takes a `WaveSchedule` (the BSP partition output of
//! `crate::schedule::partition_into_waves`) plus a `TargetProfile`'s
//! `LoweringConstraints`, and produces an `ExecutionPlan` that
//! describes how the schedule should be lowered to one-or-many
//! `__global__` functions on the target.
//!
//! See `solver.rs` for the DP recurrence and `plan.rs` for the
//! reified plan types.
//!
//! **Status**: CP1 — solver + plan types + tests + Display dump only.
//! No codegen change. The legacy `scheduled_codegen` still emits the
//! current monolithic megakernel; CP2 introduces the per-target
//! backends that consume `ExecutionPlan`.

pub mod cost_occupancy;
pub mod plan;
pub mod solver;

pub use plan::{ExecutionPlan, Handoff, KernelGroup, WaveId};
pub use solver::lower;

#[cfg(test)]
mod tests;
