// SPDX-License-Identifier: Apache-2.0
//! Stencil IR: the layer between FUF (op-level math DAG) and
//! megakernel codegen. Region = straight-line stencil with an
//! affine iteration domain, role-annotated nodes, and typed dep
//! edges. Arch differences live in the `arch` mapping table, not
//! in the IR itself.
//!
//! See `STENCIL_IR_DESIGN.md` (vocabulary freeze) and
//! `STENCIL_IR_SKETCH.md` (struct shapes) at the repo root.

pub mod arch;
pub mod emit;
pub mod emit_mega;
pub mod emit_ops;
pub mod ir;
pub mod print;
pub mod schedule;
pub mod template;
pub mod wavefront;

pub use arch::{ArchMap, BarrierPrim, HardwareUnit, sm89_fa2, sm90_fa2};
pub use emit::emit_kernel_sketch;
pub use emit_mega::{EmitError as MegaEmitError, emit_megakernel};
pub use ir::{
    AddrTerm, AffineOffset, Axis, AxisId, Bound, CmpOp, ControlEdge, DepKind, DepVector, Domain,
    Edge, FufOpRef, LoadAddr, Megakernel, Node, NodeId, Predicate, Region, RegionId, Role,
    ScalarBinding, ScalarId, SmemLookup, StrideExpr,
};
pub use schedule::{AxisKind, classify_axes, region_pipeline_depth, topo_order_within_iter};
pub use template::{AttnParams, PagedDecodeParams, Window, attn_region, attn_region_paged_decode};
pub use wavefront::{Schedule, ScheduleError, Step, schedule_wavefront};
