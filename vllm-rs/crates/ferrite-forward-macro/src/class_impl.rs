// SPDX-License-Identifier: Apache-2.0
//! Class → Impl consistency check.
//!
//! Precondition for collapsing each equivalence class of Regions
//! into one: every Region in a class must have been assigned the
//! same solver `ImplId`. In practice this is expected to hold on
//! well-formed transformer forwards — the solver has no reason to
//! pick Marlin at layer 0 and Fp8 at layer 1 when weight storage
//! is uniform — but the invariant becomes *load-bearing* once
//! codegen assumes it. This module tests it before we rely on it.
//!
//! Outputs:
//! - Per class, the `Vec<ImplId>` of distinct impls observed. If
//!   length is 1, class is "clean" → codegen can collapse. If > 1,
//!   class is "heterogeneous" → collapse is blocked for that class
//!   until the solver is constrained to pick uniformly.
//!
//! Design reference: STENCIL_IR_V2_DESIGN.md §7.

#![allow(dead_code)]

use std::collections::BTreeSet;

use crate::impl_lib::ImplId;
use crate::periodicity::RegionClass;
use crate::solver::{Assignment, SubgraphId};

/// Report whether every class's members picked the same Impl.
///
/// `region_subgraphs` is the back-map produced by
/// `region_formation::form_regions` (indexed by `RegionId as usize`).
pub fn check_class_impls(
    classes: &[RegionClass],
    region_subgraphs: &[SubgraphId],
    assignment: &Assignment,
) -> ClassImplReport {
    let mut per_class: Vec<Vec<ImplId>> = Vec::with_capacity(classes.len());
    let mut inconsistent_classes: Vec<(usize, Vec<ImplId>)> = Vec::new();

    for (class_idx, class) in classes.iter().enumerate() {
        let mut impls: BTreeSet<ImplId> = BTreeSet::new();
        for &rid in &class.members {
            let sg = region_subgraphs[rid as usize];
            if let Some(imp) = assignment.impl_of(sg) {
                impls.insert(imp);
            }
        }
        let picks: Vec<ImplId> = impls.into_iter().collect();
        if picks.len() > 1 {
            inconsistent_classes.push((class_idx, picks.clone()));
        }
        per_class.push(picks);
    }

    ClassImplReport {
        per_class,
        inconsistent_classes,
    }
}

/// Result of `check_class_impls`.
#[derive(Debug, Clone)]
pub struct ClassImplReport {
    /// One entry per input class, in the same order. Each entry
    /// is the set of distinct `ImplId`s observed in the class's
    /// members (sorted, deduped). Length 1 = consistent.
    pub per_class: Vec<Vec<ImplId>>,
    /// Convenience: classes whose `per_class` entry has > 1 impl.
    /// Each entry is `(class_idx, impls)`. Empty means every class
    /// is consistent → collapse is safe.
    pub inconsistent_classes: Vec<(usize, Vec<ImplId>)>,
}

/// Resolve every class to its single `ImplId`. Panics if any
/// class is heterogeneous — call `check_class_impls` first and
/// handle the report before calling this.
///
/// Returned vec parallels the input `classes`: `class_impls[i]`
/// is the ImplId for `classes[i]`.
pub fn resolve_class_impls_strict(
    classes: &[RegionClass],
    region_subgraphs: &[SubgraphId],
    assignment: &Assignment,
) -> Vec<ImplId> {
    let report = check_class_impls(classes, region_subgraphs, assignment);
    assert!(
        report.inconsistent_classes.is_empty(),
        "heterogeneous impls per class: {:?}",
        report.inconsistent_classes
    );
    report
        .per_class
        .into_iter()
        .map(|picks| {
            *picks
                .first()
                .expect("every class has at least one member with an Impl")
        })
        .collect()
}

/// Resolve classes to Impls, tolerating heterogeneity by picking
/// the first impl in sorted order. Diagnostic — produces a plan
/// that downstream consumers can use to sketch the collapse even
/// when the solver isn't yet uniform. Unlike `_strict`, does not
/// panic; the returned `inconsistent_classes` indicates which
/// classes lost fidelity.
pub fn resolve_class_impls_tolerant(
    classes: &[RegionClass],
    region_subgraphs: &[SubgraphId],
    assignment: &Assignment,
) -> (Vec<ImplId>, Vec<(usize, Vec<ImplId>)>) {
    let report = check_class_impls(classes, region_subgraphs, assignment);
    let picks: Vec<ImplId> = report
        .per_class
        .into_iter()
        .map(|p| {
            *p.first()
                .expect("every class has at least one member with an Impl")
        })
        .collect();
    (picks, report.inconsistent_classes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classified::OpKind;
    use crate::fuf::{Fuf, FufInput, FufNode, TileId};
    use crate::impl_lib::ImplId;
    use crate::periodicity::group_regions;
    use crate::region_formation::form_regions;
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

    /// Build an Assignment directly, with explicit subgraph → impl
    /// mapping so tests control the heterogeneity story.
    fn assignment_with_impls(covers: &[(&[u32], u32 /* impl_id */)]) -> Assignment {
        let mut a = Assignment::default();
        for (i, (tiles, imp)) in covers.iter().enumerate() {
            let sg = SubgraphId(i as u32);
            for &t in *tiles {
                a.cover.insert(TileId(t), sg);
            }
            a.impls.insert(sg, ImplId(*imp));
        }
        a
    }

    #[test]
    fn consistent_class_has_single_impl() {
        // Two independent rmsnorms (no producer/consumer) share a
        // class under the neighbor-aware hash; both claimed by
        // Impl 7 → consistent.
        let fuf = fuf_from_ops(vec![(OpKind::RmsNorm, vec![]), (OpKind::RmsNorm, vec![])]);
        let st = subtile(&fuf);
        let a = assignment_with_impls(&[(&[0], 7), (&[1], 7)]);
        let fr = form_regions(&st, &a);
        let classes = group_regions(&fr.graph);
        let report = check_class_impls(&classes, &fr.region_subgraphs, &a);
        assert!(report.inconsistent_classes.is_empty());
        assert_eq!(report.per_class.len(), 1);
        assert_eq!(report.per_class[0], vec![ImplId(7)]);
    }

    #[test]
    fn heterogeneous_class_flagged() {
        // Two independent rmsnorms in the same class but claimed
        // by different impls. Report flags the class.
        let fuf = fuf_from_ops(vec![(OpKind::RmsNorm, vec![]), (OpKind::RmsNorm, vec![])]);
        let st = subtile(&fuf);
        let a = assignment_with_impls(&[(&[0], 7), (&[1], 9)]);
        let fr = form_regions(&st, &a);
        let classes = group_regions(&fr.graph);
        let report = check_class_impls(&classes, &fr.region_subgraphs, &a);
        assert_eq!(report.inconsistent_classes.len(), 1);
        let (class_idx, picks) = &report.inconsistent_classes[0];
        assert_eq!(*class_idx, 0);
        assert_eq!(*picks, vec![ImplId(7), ImplId(9)]);
    }

    #[test]
    fn resolve_strict_returns_picks_in_class_order() {
        // Two rmsnorms fanning out from a shared embed — both have
        // up=[Embed], down=[] so they share a class under the
        // neighbor-aware hash.
        let fuf = fuf_from_ops(vec![
            (OpKind::Embed, vec![]),
            (OpKind::RmsNorm, vec![tile_in(0)]),
            (OpKind::RmsNorm, vec![tile_in(0)]),
        ]);
        let st = subtile(&fuf);
        let a = assignment_with_impls(&[(&[0], 3), (&[1], 5), (&[2], 5)]);
        let fr = form_regions(&st, &a);
        let classes = group_regions(&fr.graph);
        let picks = resolve_class_impls_strict(&classes, &fr.region_subgraphs, &a);
        // Two classes: embed (Impl 3) + rmsnorm (Impl 5).
        assert_eq!(picks.len(), 2);
        // Order follows classes (by representative RegionId).
        // Embed is R0, rmsnorms are R1/R2 → embed class first.
        assert_eq!(picks, vec![ImplId(3), ImplId(5)]);
    }
}
