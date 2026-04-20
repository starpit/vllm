// SPDX-License-Identifier: Apache-2.0
//! Stencil IR — arch-neutral intermediate representation between FUF
//! (op-level math DAG, fully unrolled) and codegen.
//!
//! A `Region` is a straight-line stencil with an N-D iteration
//! domain, role-annotated nodes, and typed dep edges carrying
//! affine offset vectors. A `RegionGraph` is a CFG of `Region`s
//! with control edges (barriers, data-dependent dispatch). The
//! IR is lowering-agnostic — the same graph serves Rust-side
//! launch dispatch (one launch per Region) and a future
//! single-kernel path.
//!
//! See `STENCIL_IR_V2_DESIGN.md` at the repo root for the full
//! design including axis vocabulary, shared-axis semantics across
//! Region boundaries, and the periodicity-detection pass that
//! re-rolls per-layer unrolling.
//!
//! This crate intentionally holds **IR types only** — no emitters,
//! no templates, no analyses. Those layer on top in separate
//! modules/crates so the IR vocabulary stays the narrow waist.

pub mod ir;

pub use ir::{
    AddrTerm, AffineOffset, Axis, AxisId, Bound, CmpOp, ControlEdge, DepKind, DepVector, Domain,
    Edge, FufOpRef, LoadAddr, Node, NodeId, Predicate, Region, RegionGraph, RegionId, Role,
    ScalarBinding, ScalarId, SmemLookup, StrideExpr, validate,
};
