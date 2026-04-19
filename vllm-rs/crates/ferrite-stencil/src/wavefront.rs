// SPDX-License-Identifier: Apache-2.0
//! Wavefront scheduler. Composes `classify_axes`,
//! `region_pipeline_depth`, and `topo_order_within_iter` into a
//! concrete per-CTA schedule: preamble (run once on entry), body
//! (one steady-state iteration of the serial loop), epilogue (run
//! once on exit).
//!
//! Categorisation rule — every node falls into exactly one of:
//!   - **Preamble**: Load whose address does not mention the serial
//!     axis. Runs once when the CTA starts (e.g. load_q for FA2).
//!   - **Epilogue**: Store whose address does not mention the serial
//!     axis. The serial axis is reduced for this node; fires once
//!     on exit (e.g. store_o for FA2 after the kv_tile loop).
//!   - **Body**: everything else — Loads/Stores that iterate on the
//!     serial axis (the prefetch loads) + all Compute nodes.
//!
//! Within the body, Load nodes are emitted before Compute nodes so
//! the next iteration's prefetch kicks off before this iteration's
//! consumers wait on the pipeline. Pipeline-source nodes carry an
//! `iter_offset = +P` marking "this load is for iteration k+P, the
//! consumer at iter k reads from P iterations ago". Same-iteration
//! Raw ordering among Compute nodes comes from
//! `topo_order_within_iter`.
//!
//! Not yet emitted: a prologue that seeds the pipeline (first P
//! loads at k=0..P-1 before any consumer runs). Easy extension —
//! it's `P * |body-loads|` steps — but the body output already
//! carries the pipeline_depth needed to emit it, and keeping the v1
//! output small + testable is more valuable than lowering the prologue
//! now. Added in the next commit.

use smallvec::SmallVec;

use crate::arch::{ArchMap, BarrierPrim};
use crate::ir::{AddrTerm, AxisId, DepKind, Node, NodeId, Region, Role};
use crate::schedule::{AxisKind, classify_axes, region_pipeline_depth, topo_order_within_iter};

#[derive(Debug, Clone)]
pub struct Step {
    pub node: NodeId,
    /// Serial-axis lead. `+P` marks a pipeline-source Load that is
    /// loading for iteration `k+P` while the consumer at iter `k`
    /// reads from the P-iteration-earlier buffer. `0` otherwise.
    pub iter_offset: i32,
    /// Primitives the emitter must fence on before this step.
    /// One per incoming edge, mapped through `ArchMap::barrier`.
    pub barriers_before: SmallVec<[BarrierPrim; 2]>,
}

#[derive(Debug, Clone)]
pub struct Schedule {
    pub parallel_axes: Vec<AxisId>,
    pub serial_axis: Option<AxisId>,
    pub pipeline_depth: u32,
    pub preamble: Vec<Step>,
    pub body: Vec<Step>,
    pub epilogue: Vec<Step>,
}

#[derive(Debug)]
pub enum ScheduleError {
    MultipleSerialAxes { axes: Vec<AxisId> },
}

impl std::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScheduleError::MultipleSerialAxes { axes } => write!(
                f,
                "v1 scheduler expects at most one serial axis per region; got {:?}",
                axes
            ),
        }
    }
}

impl std::error::Error for ScheduleError {}

pub fn schedule_wavefront(region: &Region, arch: &ArchMap) -> Result<Schedule, ScheduleError> {
    let classes = classify_axes(region);
    let serial_axes: Vec<AxisId> = classes
        .iter()
        .filter(|(_, k)| *k == AxisKind::Serial)
        .map(|(id, _)| *id)
        .collect();
    let parallel_axes: Vec<AxisId> = classes
        .iter()
        .filter(|(_, k)| *k == AxisKind::Parallel)
        .map(|(id, _)| *id)
        .collect();
    if serial_axes.len() > 1 {
        return Err(ScheduleError::MultipleSerialAxes { axes: serial_axes });
    }
    let serial_axis = serial_axes.first().copied();
    let pipeline_depth = region_pipeline_depth(region);

    let topo = topo_order_within_iter(region);

    let mut preamble = Vec::new();
    let mut body_loads = Vec::new();
    let mut body_computes = Vec::new();
    let mut epilogue = Vec::new();

    for &nid in &topo {
        let node = region.node(nid);
        let on_serial = node_uses_serial_axis(node, serial_axis);
        let step = build_step(region, node, arch, pipeline_depth);

        match (node.role, on_serial) {
            (Role::Load, false) => preamble.push(step),
            (Role::Store, false) => epilogue.push(step),
            (Role::Load, true) => body_loads.push(step),
            (Role::Store, true) => body_loads.push(step),
            (Role::Compute, _) => body_computes.push(step),
        }
    }

    // Body order: loads first, then computes in same-iter topo order.
    // The loads are the next iteration's prefetch; getting them
    // kicked off early maximises overlap with this iteration's math.
    let mut body = body_loads;
    body.extend(body_computes);

    Ok(Schedule {
        parallel_axes,
        serial_axis,
        pipeline_depth,
        preamble,
        body,
        epilogue,
    })
}

fn node_uses_serial_axis(node: &Node, serial: Option<AxisId>) -> bool {
    let Some(serial) = serial else {
        return false;
    };
    let Some(addr) = node.addr.as_ref() else {
        // Compute nodes carry no address; they don't participate in
        // the serial-vs-non-serial split via addresses. Caller treats
        // Compute separately (it always goes to body).
        return false;
    };
    addr.terms.iter().any(|t| match t {
        AddrTerm::AxisStride { axis, .. }
        | AddrTerm::AxisModStride { axis, .. }
        | AddrTerm::AxisDivGather { axis, .. } => *axis == serial,
        AddrTerm::RegionEntryConst(_) => false,
    })
}

fn build_step(region: &Region, node: &Node, arch: &ArchMap, pipeline_depth: u32) -> Step {
    // iter_offset: if this node is the source of any Pipeline edge,
    // the scheduled step fires for iter k+|dep|. Max over outgoing
    // Pipeline edges (only ever one in practice but the IR permits
    // more — take the max for v1, refine if a real region needs
    // per-consumer offsets).
    let iter_offset = region
        .edges
        .iter()
        .filter(|e| e.src == node.id && e.kind == DepKind::Pipeline)
        .flat_map(|e| e.vector.0.iter().map(|(_, d)| d.unsigned_abs() as i32))
        .max()
        .unwrap_or(0);

    // Sanity: pipeline_depth from the region must be >= any single
    // edge's lead. This is the invariant that lets the arch lower
    // Pipeline dep kinds without extra buffering.
    debug_assert!(
        iter_offset as u32 <= pipeline_depth,
        "node {} has iter_offset {} > region pipeline_depth {}",
        node.id,
        iter_offset,
        pipeline_depth
    );

    let barriers_before: SmallVec<[BarrierPrim; 2]> = region
        .edges
        .iter()
        .filter(|e| e.dst == node.id)
        .map(|e| (arch.barrier)(e.kind))
        .collect();

    Step {
        node: node.id,
        iter_offset,
        barriers_before,
    }
}
