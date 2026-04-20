// SPDX-License-Identifier: Apache-2.0
//! Region formation: group sub-tiled FUF tiles into `Region`s
//! driven by the solver's Impl picks.
//!
//! One Region per subgraph in the Assignment. Each Region's axis
//! set is the union of its constituent tiles' axis sets. Edges
//! between claimed tiles inside the same subgraph become intra-
//! Region `Edge`s; cross-subgraph dataflow becomes `ControlEdge`s
//! on the enclosing `RegionGraph`.
//!
//! Still fully unrolled along every repeating dimension at this
//! point — one Region per (Impl, layer). Periodicity detection
//! (design §6.2) runs next and collapses isomorphic Region groups
//! to a single Region with an outer `repeat` axis.
//!
//! Scope limits (intentional, tightened later):
//! - Every Node in the emitted Region is `Role::Compute` with no
//!   address. Load/Compute/Store decomposition happens when the
//!   Rust / CUDA lowering that consumes this IR lands.
//! - Intra-Region edges are marked `DepKind::Raw` with
//!   zero-vector offsets. Refined when Load nodes with concrete
//!   addresses are introduced.
//! - Region axes are emitted with `Bound::Unbounded`. Concrete
//!   bounds come from the ModelParams at the codegen step that
//!   consumes `RegionGraph`; the IR stays size-invariant.
//! - `FufOpRef::tag` carries the `OpKind` name as a `'static str`.
//!   Sufficient for inspection and dedup; richer back-refs are
//!   a future concern.

#![allow(dead_code)]

use std::collections::HashMap;

use ferrite_stencil_ir::{
    Axis, AxisId, Bound, ControlEdge, DepKind, DepVector, Domain, Edge, FufOpRef, Node, NodeId,
    Region, RegionGraph, RegionId, Role,
};

use crate::classified::OpKind;
use crate::fuf::{Fuf, FufInput, TileId};
use crate::solver::{Assignment, SubgraphId};
use crate::subtile::{SubtiledFuf, TileAxis};

/// Output of `form_regions`: the `RegionGraph` + a back-map from
/// each `RegionId` to the `SubgraphId` that produced it. Consumers
/// that need the source `SubgraphId` (e.g. to look up the solver's
/// `ImplId`) index into `region_subgraphs` by `RegionId as usize`.
#[derive(Debug, Clone)]
pub struct FormedRegions {
    pub graph: RegionGraph,
    pub region_subgraphs: Vec<SubgraphId>,
}

/// Form a `RegionGraph` from a sub-tiled FUF + solver Assignment.
///
/// Emits one Region per subgraph in the Assignment. Subgraphs are
/// processed in sorted `SubgraphId` order for determinism.
pub fn form_regions(st: &SubtiledFuf<'_>, assignment: &Assignment) -> FormedRegions {
    // Sort subgraphs for deterministic Region ordering.
    let mut subgraphs: Vec<SubgraphId> = assignment.subgraphs().collect();
    subgraphs.sort();

    let mut regions: Vec<Region> = Vec::with_capacity(subgraphs.len());
    let mut sg_to_rid: HashMap<SubgraphId, RegionId> = HashMap::new();
    let mut region_subgraphs: Vec<SubgraphId> = Vec::with_capacity(subgraphs.len());

    for sg in &subgraphs {
        let tiles = assignment.tiles_in_subgraph(*sg);
        if tiles.is_empty() {
            continue;
        }
        let rid = regions.len() as RegionId;
        sg_to_rid.insert(*sg, rid);
        region_subgraphs.push(*sg);
        regions.push(form_one_region(rid, st, &tiles));
    }

    let control = derive_control_edges(st.fuf, assignment, &sg_to_rid);

    FormedRegions {
        graph: RegionGraph { regions, control },
        region_subgraphs,
    }
}

/// Build a single Region from its claimed tiles.
fn form_one_region(id: RegionId, st: &SubtiledFuf<'_>, claimed: &[TileId]) -> Region {
    // Axis union: gather every axis mentioned by any claimed tile,
    // in a stable order (first-seen).
    let mut axes_in_order: Vec<TileAxis> = Vec::new();
    for &tile in claimed {
        let annot = &st.tiles[tile.0 as usize];
        for &axis in &annot.axes {
            if !axes_in_order.contains(&axis) {
                axes_in_order.push(axis);
            }
        }
    }
    let axes: Vec<Axis> = axes_in_order
        .iter()
        .enumerate()
        .map(|(i, ta)| Axis {
            id: i as AxisId,
            name: axis_name(*ta),
            bound: Bound::Unbounded,
        })
        .collect();

    // Nodes: one per claimed tile, ordered by TileId for stability.
    // Local NodeId = index into claimed.
    let nodes: Vec<Node> = claimed
        .iter()
        .enumerate()
        .map(|(i, &tile)| {
            let op = st.fuf.get(tile).op;
            Node {
                id: i as NodeId,
                role: Role::Compute,
                op: FufOpRef {
                    tag: opkind_tag(op),
                },
                addr: None,
            }
        })
        .collect();

    // Edges: for each claimed tile's tile-inputs whose producer is
    // also in `claimed`, emit an intra-Region Raw edge.
    let claimed_to_nodeid: HashMap<TileId, NodeId> = claimed
        .iter()
        .enumerate()
        .map(|(i, &t)| (t, i as NodeId))
        .collect();
    let mut edges: Vec<Edge> = Vec::new();
    for (&tile, &dst_nid) in claimed_to_nodeid.iter() {
        let node = st.fuf.get(tile);
        for input in &node.inputs {
            if let FufInput::Tile { id, .. } = input
                && let Some(&src_nid) = claimed_to_nodeid.get(id)
            {
                edges.push(Edge {
                    src: src_nid,
                    dst: dst_nid,
                    kind: DepKind::Raw,
                    vector: DepVector::default(),
                });
            }
        }
    }
    // Stable edge order for determinism.
    edges.sort_by_key(|e| (e.src, e.dst));

    Region {
        id,
        name: region_name_from_claimed(st, claimed),
        domain: Domain {
            axes,
            predicates: Vec::new(),
        },
        entry_scalars: Vec::new(),
        nodes,
        edges,
    }
}

/// Cross-subgraph dataflow → control edges. For every claimed
/// tile's tile-input whose producer is in a *different* subgraph,
/// emit a `Barrier` control edge from the producer's Region to
/// this Region.
///
/// Dedupe (src_rid, dst_rid) pairs — one barrier suffices no
/// matter how many tile-inputs cross.
fn derive_control_edges(
    fuf: &Fuf,
    assignment: &Assignment,
    sg_to_rid: &HashMap<SubgraphId, RegionId>,
) -> Vec<ControlEdge> {
    use std::collections::BTreeSet;

    let mut pairs: BTreeSet<(RegionId, RegionId)> = BTreeSet::new();
    for node in &fuf.nodes {
        let Some(dst_sg) = assignment.subgraph_of(node.id) else {
            continue;
        };
        let Some(&dst_rid) = sg_to_rid.get(&dst_sg) else {
            continue;
        };
        for input in &node.inputs {
            if let FufInput::Tile { id, .. } = input
                && let Some(src_sg) = assignment.subgraph_of(*id)
                && src_sg != dst_sg
                && let Some(&src_rid) = sg_to_rid.get(&src_sg)
            {
                pairs.insert((src_rid, dst_rid));
            }
        }
    }

    pairs
        .into_iter()
        .map(|(src, dst)| ControlEdge {
            src,
            dst,
            kind: DepKind::Barrier,
        })
        .collect()
}

/// Stable `'static` name per axis. Matches the design doc §4.1
/// vocabulary.
fn axis_name(axis: TileAxis) -> &'static str {
    match axis {
        TileAxis::TileT => "tile_t",
        TileAxis::TileDInter => "tile_d_inter",
        TileAxis::TileDHead => "tile_d_head",
        TileAxis::HeadGroup => "head_group",
        TileAxis::QTile => "q_tile",
        TileAxis::KvTile => "kv_tile",
        TileAxis::TileDVocab => "tile_d_vocab",
    }
}

/// `'static` string tag per OpKind for `FufOpRef::tag`. Provides
/// a stable marker the emitter can match on. Names match the
/// debug-style string of the enum variant for easy log reading.
fn opkind_tag(op: OpKind) -> &'static str {
    match op {
        OpKind::Embed => "Embed",
        OpKind::RmsNorm => "RmsNorm",
        OpKind::LayerNorm => "LayerNorm",
        OpKind::Gemm => "Gemm",
        OpKind::RopeAppend => "RopeAppend",
        OpKind::RopeAppendInterleaved => "RopeAppendInterleaved",
        OpKind::Attention => "Attention",
        OpKind::SlidingAttention => "SlidingAttention",
        OpKind::Silu => "Silu",
        OpKind::Gelu => "Gelu",
        OpKind::TanhSoftCap => "TanhSoftCap",
        OpKind::Add => "Add",
        OpKind::BiasAdd => "BiasAdd",
        OpKind::Mul => "Mul",
        OpKind::Reshape => "Reshape",
    }
}

/// Single-word name for a Region: its last claimed tile's op
/// kind. Matches how subgraphs are commonly identified in logs
/// (the "seed" op). Best-effort — only for inspection / debug.
fn region_name_from_claimed(st: &SubtiledFuf<'_>, claimed: &[TileId]) -> &'static str {
    claimed
        .last()
        .map(|&t| opkind_tag(st.fuf.get(t).op))
        .unwrap_or("empty")
}

/// Pretty-print a RegionGraph for inspection. One Region per
/// block showing its id, name, axis names, and node/edge counts,
/// plus a footer with control edge summary.
pub fn pretty_print(rg: &RegionGraph) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    writeln!(
        out,
        "RegionGraph: {} regions, {} control edges",
        rg.regions.len(),
        rg.control.len()
    )
    .unwrap();
    for r in &rg.regions {
        let axes: Vec<&str> = r.domain.axes.iter().map(|a| a.name).collect();
        writeln!(
            out,
            "  R{:>3} {:<25} axes=({}) nodes={} edges={}",
            r.id,
            r.name,
            axes.join(","),
            r.nodes.len(),
            r.edges.len(),
        )
        .unwrap();
    }
    if !rg.control.is_empty() {
        writeln!(out, "  control edges:").unwrap();
        for ce in &rg.control {
            writeln!(out, "    R{} → R{} ({:?})", ce.src, ce.dst, ce.kind).unwrap();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuf::{Fuf, FufNode};
    use crate::impl_lib::ImplId;
    use crate::solver::SubgraphId;
    use crate::subtile::subtile;

    fn tile_in(id: u32) -> FufInput {
        FufInput::Tile {
            id: TileId(id),
            slot: 0,
        }
    }

    fn fuf_from_ops(ops: Vec<(OpKind, Vec<FufInput>)>) -> Fuf {
        let nodes = ops
            .into_iter()
            .enumerate()
            .map(|(i, (op, inputs))| FufNode {
                id: TileId(i as u32),
                op,
                inputs,
                outputs: vec![Vec::new()],
            })
            .collect();
        Fuf { nodes }
    }

    /// Assignment where each listed group of tiles is one
    /// subgraph bound to `ImplId(0)`. Ordering of groups sets
    /// SubgraphId numbering (0, 1, …).
    fn assignment_from_groups(groups: &[&[u32]]) -> Assignment {
        let mut a = Assignment::default();
        for (i, group) in groups.iter().enumerate() {
            let sg = SubgraphId(i as u32);
            for &t in *group {
                a.cover.insert(TileId(t), sg);
            }
            a.impls.insert(sg, ImplId(0));
        }
        a
    }

    #[test]
    fn single_region_single_tile() {
        // One rmsnorm tile → one Region, axes = (tile_t), no
        // edges, no control.
        let fuf = fuf_from_ops(vec![(OpKind::RmsNorm, vec![])]);
        let st = subtile(&fuf);
        let a = assignment_from_groups(&[&[0]]);
        let rg = form_regions(&st, &a).graph;
        assert_eq!(rg.regions.len(), 1);
        assert_eq!(rg.regions[0].domain.axes.len(), 1);
        assert_eq!(rg.regions[0].domain.axes[0].name, "tile_t");
        assert_eq!(rg.regions[0].nodes.len(), 1);
        assert_eq!(rg.regions[0].edges.len(), 0);
        assert_eq!(rg.control.len(), 0);
    }

    #[test]
    fn two_subgraphs_get_barrier() {
        // embed (sg0) → rmsnorm (sg1). Dataflow crosses subgraph
        // boundary → one barrier control edge.
        let fuf = fuf_from_ops(vec![
            (OpKind::Embed, vec![]),
            (OpKind::RmsNorm, vec![tile_in(0)]),
        ]);
        let st = subtile(&fuf);
        let a = assignment_from_groups(&[&[0], &[1]]);
        let rg = form_regions(&st, &a).graph;
        assert_eq!(rg.regions.len(), 2);
        assert_eq!(rg.control.len(), 1);
        assert_eq!(rg.control[0].src, 0);
        assert_eq!(rg.control[0].dst, 1);
        assert_eq!(rg.control[0].kind, DepKind::Barrier);
    }

    #[test]
    fn fused_multi_tile_subgraph_gets_intra_edges() {
        // A fused subgraph claiming gemm + silu + mul + gemm.
        // Axis union = (tile_t, tile_d_inter). Intra-Region
        // edges: gemm→silu, silu→mul, mul→gemm.
        let fuf = fuf_from_ops(vec![
            (OpKind::Embed, vec![]),                     // 0
            (OpKind::Gemm, vec![tile_in(0)]),            // 1 gate
            (OpKind::Silu, vec![tile_in(1)]),            // 2
            (OpKind::Gemm, vec![tile_in(0)]),            // 3 up
            (OpKind::Mul, vec![tile_in(2), tile_in(3)]), // 4
            (OpKind::Gemm, vec![tile_in(4)]),            // 5 down
            (OpKind::Add, vec![tile_in(5), tile_in(0)]), // 6
        ]);
        let st = subtile(&fuf);
        // sg 0: embed; sg 1: fused MLP block (tiles 1..=5); sg 2: add.
        let a = assignment_from_groups(&[&[0], &[1, 2, 3, 4, 5], &[6]]);
        let rg = form_regions(&st, &a).graph;
        assert_eq!(rg.regions.len(), 3);

        let mlp = &rg.regions[1];
        let mlp_axes: Vec<&str> = mlp.domain.axes.iter().map(|a| a.name).collect();
        assert!(mlp_axes.contains(&"tile_t"));
        assert!(mlp_axes.contains(&"tile_d_inter"));
        // Five claimed tiles → 5 nodes; intra-subgraph deps:
        // gate→silu, silu→mul, up→mul, mul→down = 4 edges.
        assert_eq!(mlp.nodes.len(), 5);
        assert_eq!(mlp.edges.len(), 4);

        // Control edges: embed→mlp, mlp→add (and embed→add for
        // the residual input). Exactly 3 pairs.
        let mut control_pairs: Vec<(RegionId, RegionId)> =
            rg.control.iter().map(|c| (c.src, c.dst)).collect();
        control_pairs.sort();
        assert_eq!(control_pairs, vec![(0, 1), (0, 2), (1, 2)]);
    }

    #[test]
    fn attention_region_carries_its_axes() {
        // Attention as its own subgraph — its Region must carry
        // the three attention-internal axes.
        let fuf = fuf_from_ops(vec![
            (OpKind::Embed, vec![]),
            (OpKind::Attention, vec![tile_in(0)]),
        ]);
        let st = subtile(&fuf);
        let a = assignment_from_groups(&[&[0], &[1]]);
        let rg = form_regions(&st, &a).graph;
        let attn = &rg.regions[1];
        let names: Vec<&str> = attn.domain.axes.iter().map(|a| a.name).collect();
        assert!(names.contains(&"head_group"));
        assert!(names.contains(&"q_tile"));
        assert!(names.contains(&"kv_tile"));
    }
}
