// SPDX-License-Identifier: Apache-2.0
//! Collapse plan — the shape of a `RegionGraph` after the
//! periodicity-rewrite would apply.
//!
//! `CollapsePlan` is a *description* of the collapse, not a
//! rewritten graph. It records one canonical Region per
//! equivalence class, the period of each class, and the dedup'd
//! control-edge structure between classes. The original
//! `RegionGraph` is left untouched.
//!
//! Why a plan (not a rewrite)? The final representation of
//! "Region with outer repeat axis" is a codegen-facing IR choice
//! — carrying an `Option<Bound>` on `RegionGraph`? Nesting a
//! `LoopBody` wrapper? Adding `Δrepeat` to `ControlEdge`? The
//! answer depends on how the emitter consumes it. Producing a
//! plan lets us measure the collapse and feed it to multiple
//! consumers without pre-committing to an IR surface change that
//! a future emitter might regret. When a consumer is ready the
//! plan flattens into whichever final form fits.
//!
//! Design reference: STENCIL_IR_V2_DESIGN.md §6.2 step 4-5.

#![allow(dead_code)]

use std::collections::{BTreeSet, HashMap};

use ferrite_stencil_ir::{DepKind, RegionGraph, RegionId};

use crate::periodicity::RegionClass;

/// The shape a `RegionGraph` would have if every periodic class
/// collapsed to one Region + an outer iteration.
#[derive(Debug, Clone)]
pub struct CollapsePlan {
    /// One ClassIdx per equivalence class, in the order the
    /// classes were provided to `plan_collapse`.
    pub classes: Vec<ClassInfo>,
    /// Original Region → which class index. Every original
    /// RegionId appears as a key exactly once.
    pub region_to_class: HashMap<RegionId, ClassIdx>,
    /// Cross-class control edges, de-duplicated. Each entry is
    /// `(src_class, dst_class, kind)` — if the original graph had
    /// a barrier per member pair (R_src_i → R_dst_i), all of them
    /// collapse to one edge `src_class → dst_class`. Cross-class
    /// cross-iteration edges (e.g. `add_final(k) → rmsnorm(k+1)`
    /// in a transformer) show up here too, with the same shape;
    /// `Δrepeat` resolution is left to the consumer.
    pub control: Vec<CollapsedControlEdge>,
    /// True if any class has period > 1 — i.e. the collapse is
    /// non-trivial. A RegionGraph with every class trivially of
    /// period 1 has nothing to collapse.
    pub is_periodic: bool,
}

/// Index into `CollapsePlan::classes`.
pub type ClassIdx = usize;

#[derive(Debug, Clone)]
pub struct ClassInfo {
    pub idx: ClassIdx,
    /// The Region chosen as this class's canonical instance. The
    /// others in the class are structurally identical (same
    /// canonical hash) and drop out of the collapsed graph.
    pub canonical: RegionId,
    /// How many Regions in the original graph collapse into this
    /// class. For a transformer's layer-block patterns this
    /// should roughly equal `num_hidden_layers`.
    pub period: usize,
    pub canonical_hash: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollapsedControlEdge {
    pub src_class: ClassIdx,
    pub dst_class: ClassIdx,
    pub kind: DepKind,
}

/// Build a collapse plan from a RegionGraph + its classes.
/// `classes` should be the output of [`crate::periodicity::group_regions`]
/// on the same `rg`.
pub fn plan_collapse(rg: &RegionGraph, classes: &[RegionClass]) -> CollapsePlan {
    // Assign class indices in the order classes were given. Build
    // region → class lookup.
    let mut region_to_class: HashMap<RegionId, ClassIdx> = HashMap::new();
    let mut class_infos: Vec<ClassInfo> = Vec::with_capacity(classes.len());
    for (idx, c) in classes.iter().enumerate() {
        for &r in &c.members {
            region_to_class.insert(r, idx);
        }
        class_infos.push(ClassInfo {
            idx,
            canonical: c.representative(),
            period: c.period(),
            canonical_hash: c.canonical_hash,
        });
    }

    // Cross-class control edges. Dedupe on (src_class, dst_class,
    // kind). Different kinds between the same pair stay separate.
    // Self-loops (src_class == dst_class) are dropped — a class
    // can't barrier itself in the collapsed graph (same class
    // iterates together).
    let mut uniq: BTreeSet<(ClassIdx, ClassIdx, u8)> = BTreeSet::new();
    let mut control: Vec<CollapsedControlEdge> = Vec::new();
    for ce in &rg.control {
        let Some(&src_class) = region_to_class.get(&ce.src) else {
            continue;
        };
        let Some(&dst_class) = region_to_class.get(&ce.dst) else {
            continue;
        };
        if src_class == dst_class {
            continue;
        }
        let key = (src_class, dst_class, ce.kind as u8);
        if uniq.insert(key) {
            control.push(CollapsedControlEdge {
                src_class,
                dst_class,
                kind: ce.kind,
            });
        }
    }
    // Stable order for determinism.
    control.sort_by_key(|e| (e.src_class, e.dst_class, e.kind as u8));

    let is_periodic = class_infos.iter().any(|c| c.period > 1);

    CollapsePlan {
        classes: class_infos,
        region_to_class,
        control,
        is_periodic,
    }
}

/// Pretty-print the plan for inspection. Shows the collapse
/// factor (original region count / class count), per-class
/// period, and cross-class control topology.
pub fn pretty_print(rg: &RegionGraph, plan: &CollapsePlan) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let original = rg.regions.len();
    let collapsed = plan.classes.len();
    let factor = if collapsed == 0 {
        0.0
    } else {
        original as f64 / collapsed as f64
    };
    writeln!(
        out,
        "CollapsePlan: {} regions → {} canonical ({:.1}× collapse), control {} → {}, periodic={}",
        original,
        collapsed,
        factor,
        rg.control.len(),
        plan.control.len(),
        plan.is_periodic,
    )
    .unwrap();
    for ci in &plan.classes {
        let canon = &rg.regions[ci.canonical as usize];
        writeln!(
            out,
            "  C{:>3} {:<25} period={:>3} hash={:016x}",
            ci.idx, canon.name, ci.period, ci.canonical_hash,
        )
        .unwrap();
    }
    for ce in &plan.control {
        let src = &plan.classes[ce.src_class];
        let dst = &plan.classes[ce.dst_class];
        writeln!(
            out,
            "  C{} → C{} ({:?})  [{} → {}]",
            ce.src_class,
            ce.dst_class,
            ce.kind,
            rg.regions[src.canonical as usize].name,
            rg.regions[dst.canonical as usize].name,
        )
        .unwrap();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classified::OpKind;
    use crate::fuf::{Fuf, FufInput, FufNode, TileId};
    use crate::impl_lib::ImplId;
    use crate::periodicity::group_regions;
    use crate::region_formation::form_regions;
    use crate::solver::{Assignment, SubgraphId};
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
    fn trivial_graph_is_not_periodic() {
        let fuf = fuf_from_ops(vec![
            (OpKind::Embed, vec![]),
            (OpKind::Attention, vec![tile_in(0)]),
        ]);
        let st = subtile(&fuf);
        let a = assignment_from_groups(&[&[0], &[1]]);
        let rg = form_regions(&st, &a).graph;
        let classes = group_regions(&rg);
        let plan = plan_collapse(&rg, &classes);
        assert!(!plan.is_periodic);
        assert_eq!(plan.classes.len(), 2);
    }

    #[test]
    fn layer_blocks_collapse_to_canonical() {
        // Two "layer blocks" each (rmsnorm, gemm, add) — the
        // same toy shape as the periodicity test. Three unique
        // classes, each period 2.
        let fuf = fuf_from_ops(vec![
            (OpKind::RmsNorm, vec![]),
            (OpKind::Gemm, vec![tile_in(0)]),
            (OpKind::Add, vec![tile_in(1), tile_in(0)]),
            (OpKind::RmsNorm, vec![tile_in(2)]),
            (OpKind::Gemm, vec![tile_in(3)]),
            (OpKind::Add, vec![tile_in(4), tile_in(2)]),
        ]);
        let st = subtile(&fuf);
        let a = assignment_from_groups(&[&[0], &[1], &[2], &[3], &[4], &[5]]);
        let rg = form_regions(&st, &a).graph;
        let classes = group_regions(&rg);
        let plan = plan_collapse(&rg, &classes);
        assert!(plan.is_periodic);
        assert_eq!(plan.classes.len(), 3);
        for c in &plan.classes {
            assert_eq!(c.period, 2);
        }
        // Every original Region belongs to some class.
        assert_eq!(plan.region_to_class.len(), 6);
    }

    #[test]
    fn intra_class_control_edges_drop() {
        // One class of two identical rmsnorm Regions with an
        // (artificial) barrier edge between them. The collapsed
        // plan should drop the self-loop.
        let fuf = fuf_from_ops(vec![
            (OpKind::RmsNorm, vec![]),
            (OpKind::RmsNorm, vec![tile_in(0)]),
        ]);
        let st = subtile(&fuf);
        let a = assignment_from_groups(&[&[0], &[1]]);
        let rg = form_regions(&st, &a).graph;
        let classes = group_regions(&rg);
        // Sanity: before collapse there's one barrier R0 → R1.
        assert_eq!(rg.control.len(), 1);
        let plan = plan_collapse(&rg, &classes);
        // Both Regions are in the same class → the single barrier
        // is a self-loop on the class and drops out.
        assert_eq!(plan.classes.len(), 1);
        assert_eq!(plan.classes[0].period, 2);
        assert_eq!(plan.control.len(), 0);
    }

    #[test]
    fn cross_class_edges_dedupe() {
        // Two "layers" with barrier edges embed → each layer's
        // first op. embed is period-1; rmsnorm is period-2.
        // The two embed→rmsnorm edges (one per layer) collapse
        // to one cross-class edge.
        let fuf = fuf_from_ops(vec![
            (OpKind::Embed, vec![]),
            (OpKind::RmsNorm, vec![tile_in(0)]),
            (OpKind::RmsNorm, vec![tile_in(0)]),
        ]);
        let st = subtile(&fuf);
        let a = assignment_from_groups(&[&[0], &[1], &[2]]);
        let rg = form_regions(&st, &a).graph;
        let classes = group_regions(&rg);
        let plan = plan_collapse(&rg, &classes);
        assert_eq!(plan.classes.len(), 2);
        // Only one control edge even though there were two in
        // the original graph.
        assert_eq!(plan.control.len(), 1);
    }
}
