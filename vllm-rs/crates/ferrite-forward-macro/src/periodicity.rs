// SPDX-License-Identifier: Apache-2.0
//! Region periodicity — detection half.
//!
//! Computes a canonical hash per `Region` based on its intra-Region
//! structure (op tags, roles, axis names, edge shape) and groups
//! Regions by hash. Two Regions that come from the same pattern
//! repeated across layers — or across any other periodic dim —
//! land in the same group.
//!
//! This is the measurement that answers "how much does a re-roll
//! actually collapse?" — unique-group count vs total Region count.
//! A well-formed transformer forward should show something like
//! 480 Regions collapsing to ~15 groups.
//!
//! The actual graph rewrite (fold each group to one Region + an
//! outer `repeat` axis, re-route control edges as intra-Region
//! dep vectors with `Δrepeat ≠ 0`) is a separate pass — see
//! design §6.2 step 4-5 and module `periodicity_collapse` (later).
//!
//! Scope limits:
//! - Hash covers op tags, roles, axis names, edge shape. Addresses
//!   (`Node::addr`) and entry scalars are both empty in v1, so
//!   they trivially don't defeat grouping yet. When they
//!   populate, the hash has to canonicalize them to the symbolic
//!   form (§6.2 literal lifting).
//! - Nodes are hashed in the order `form_regions` emits them
//!   (topological by claimed TileId). Two Regions with the same
//!   FUF pattern emit nodes in the same order, so no re-canonical-
//!   ization is needed for the patterns we target. DAG-isomorphism
//!   up to relabeling would be needed to catch pathological
//!   re-orderings; not expected in transformer forwards.

#![allow(dead_code)]

use std::collections::{BTreeMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};

use ferrite_stencil_ir::{Region, RegionGraph, RegionId};

/// An equivalence class of Regions under canonical hashing.
#[derive(Debug, Clone)]
pub struct RegionClass {
    /// The hash itself — stable across a single build, not intended
    /// to survive across builds (uses `DefaultHasher`).
    pub canonical_hash: u64,
    /// Region ids in this class, in `RegionId` order. The first
    /// entry is the representative the collapse pass will use.
    pub members: Vec<RegionId>,
}

impl RegionClass {
    pub fn representative(&self) -> RegionId {
        *self.members.first().expect("class is non-empty")
    }
    pub fn period(&self) -> usize {
        self.members.len()
    }
}

/// Group the `RegionGraph`'s regions by canonical hash. Classes
/// are returned sorted by representative `RegionId` for
/// determinism.
pub fn group_regions(rg: &RegionGraph) -> Vec<RegionClass> {
    // BTreeMap keyed on hash so iteration is deterministic. Ties on
    // hash (extremely unlikely, but possible) end up in the same
    // bucket — that's semantically what we want.
    let mut buckets: BTreeMap<u64, Vec<RegionId>> = BTreeMap::new();
    for r in &rg.regions {
        let h = canonical_hash(r);
        buckets.entry(h).or_default().push(r.id);
    }
    let mut classes: Vec<RegionClass> = buckets
        .into_iter()
        .map(|(canonical_hash, members)| RegionClass {
            canonical_hash,
            members,
        })
        .collect();
    // Sort by representative for deterministic output.
    classes.sort_by_key(|c| c.representative());
    classes
}

/// Canonical hash of one Region. Covers:
/// - Region.name
/// - Axis names (stable order = the order they appear on the
///   Region, which `form_one_region` populates first-seen per
///   claimed tile — deterministic per pattern)
/// - Node sequence: (op tag, role) per node
/// - Edge sequence: (src, dst, kind) sorted
///
/// Exposed for debugging / tests; callers generally want
/// `group_regions`.
pub fn canonical_hash(r: &Region) -> u64 {
    let mut h = DefaultHasher::new();
    r.name.hash(&mut h);
    for axis in &r.domain.axes {
        axis.name.hash(&mut h);
    }
    for node in &r.nodes {
        node.op.tag.hash(&mut h);
        (node.role as u8).hash(&mut h);
    }
    // Edges: deterministic order is established by `form_one_region`
    // (sort by (src, dst)), so hashing in order is safe.
    for e in &r.edges {
        e.src.hash(&mut h);
        e.dst.hash(&mut h);
        (e.kind as u8).hash(&mut h);
    }
    h.finish()
}

/// Summary stats for reporting / inspection. Inexpensive to
/// compute on a RegionGraph; intended for eprintln! during
/// development and eventual telemetry.
#[derive(Debug, Clone, Copy)]
pub struct PeriodicitySummary {
    pub total_regions: usize,
    pub unique_classes: usize,
    /// Largest class size (= period of the dominant repeat — for
    /// a transformer forward this should roughly equal
    /// `num_hidden_layers`).
    pub max_period: usize,
    /// Median class size; a distribution hint beyond just max.
    pub median_period: usize,
}

pub fn summarize(classes: &[RegionClass]) -> PeriodicitySummary {
    let total_regions: usize = classes.iter().map(|c| c.period()).sum();
    let mut periods: Vec<usize> = classes.iter().map(|c| c.period()).collect();
    periods.sort();
    let median_period = periods.get(periods.len() / 2).copied().unwrap_or(0);
    let max_period = periods.iter().copied().max().unwrap_or(0);
    PeriodicitySummary {
        total_regions,
        unique_classes: classes.len(),
        max_period,
        median_period,
    }
}

/// Pretty-print the grouping for inspection. One line per class
/// with representative, period, and a preview of member ids.
pub fn pretty_print(rg: &RegionGraph, classes: &[RegionClass]) -> String {
    use std::fmt::Write as _;
    let sum = summarize(classes);
    let mut out = String::new();
    writeln!(
        out,
        "Region periodicity: {} total → {} unique classes (max period {}, median {})",
        sum.total_regions, sum.unique_classes, sum.max_period, sum.median_period,
    )
    .unwrap();
    for c in classes {
        let rep = &rg.regions[c.representative() as usize];
        let preview: Vec<String> = c
            .members
            .iter()
            .take(6)
            .map(|id| format!("R{id}"))
            .collect();
        let trailing = if c.members.len() > 6 {
            format!(" +{} more", c.members.len() - 6)
        } else {
            String::new()
        };
        writeln!(
            out,
            "  class {:016x} period={:>3} rep={:<25} members: {}{}",
            c.canonical_hash,
            c.period(),
            rep.name,
            preview.join(","),
            trailing,
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
    fn identical_regions_collapse() {
        // Two independent rmsnorm tiles in separate subgraphs.
        // Same canonical pattern → one class of period 2.
        let fuf = fuf_from_ops(vec![
            (OpKind::Embed, vec![]),
            (OpKind::RmsNorm, vec![tile_in(0)]),
            (OpKind::RmsNorm, vec![tile_in(1)]),
        ]);
        let st = subtile(&fuf);
        let a = assignment_from_groups(&[&[0], &[1], &[2]]);
        let rg = form_regions(&st, &a);
        let classes = group_regions(&rg);
        // Embed is its own class; the two rmsnorms share a class.
        assert_eq!(classes.len(), 2);
        let rmsnorm_class = classes
            .iter()
            .find(|c| c.period() == 2)
            .expect("two rmsnorms collapse");
        assert_eq!(rmsnorm_class.period(), 2);
    }

    #[test]
    fn different_axes_do_not_collapse() {
        // rmsnorm (axes: tile_t) vs attention (axes: head_group,
        // q_tile, kv_tile). Different axis sets → different
        // canonical hash → different classes.
        let fuf = fuf_from_ops(vec![
            (OpKind::Embed, vec![]),
            (OpKind::RmsNorm, vec![tile_in(0)]),
            (OpKind::Attention, vec![tile_in(1)]),
        ]);
        let st = subtile(&fuf);
        let a = assignment_from_groups(&[&[0], &[1], &[2]]);
        let rg = form_regions(&st, &a);
        let classes = group_regions(&rg);
        assert_eq!(classes.len(), 3);
        assert!(classes.iter().all(|c| c.period() == 1));
    }

    #[test]
    fn repeated_layer_blocks_collapse_proportionally() {
        // Simulate a tiny 3-layer "transformer" at the Region
        // level: each layer is (rmsnorm, gemm, add). Six tiles
        // make two full layer blocks; the block pattern should
        // appear as one class of period 3 (one per layer,
        // representing rmsnorm / gemm / add). Wait — that's
        // three classes each of period 3 for two layers? Let's
        // restructure:
        //
        // tiles 0..3: layer 0 — rmsnorm, gemm-o-ish, add
        // tiles 3..6: layer 1 — rmsnorm, gemm-o-ish, add
        //
        // Six singleton subgraphs. Classes:
        //   RmsNorm-only Region: members {R0, R3} — period 2
        //   Gemm Region:         members {R1, R4} — period 2
        //   Add Region:          members {R2, R5} — period 2
        // Embed optional — we drop it to keep counts tidy.
        let fuf = fuf_from_ops(vec![
            (OpKind::RmsNorm, vec![]),                   // 0
            (OpKind::Gemm, vec![tile_in(0)]),            // 1
            (OpKind::Add, vec![tile_in(1), tile_in(0)]), // 2
            (OpKind::RmsNorm, vec![tile_in(2)]),         // 3
            (OpKind::Gemm, vec![tile_in(3)]),            // 4
            (OpKind::Add, vec![tile_in(4), tile_in(2)]), // 5
        ]);
        let st = subtile(&fuf);
        let a = assignment_from_groups(&[&[0], &[1], &[2], &[3], &[4], &[5]]);
        let rg = form_regions(&st, &a);
        let classes = group_regions(&rg);
        // Three unique patterns, each period 2.
        assert_eq!(classes.len(), 3);
        for c in &classes {
            assert_eq!(
                c.period(),
                2,
                "class {:x} has {} members",
                c.canonical_hash,
                c.period()
            );
        }
    }

    #[test]
    fn summary_reports_counts() {
        let fuf = fuf_from_ops(vec![
            (OpKind::RmsNorm, vec![]),
            (OpKind::RmsNorm, vec![tile_in(0)]),
            (OpKind::RmsNorm, vec![tile_in(1)]),
            (OpKind::RmsNorm, vec![tile_in(2)]),
        ]);
        let st = subtile(&fuf);
        let a = assignment_from_groups(&[&[0], &[1], &[2], &[3]]);
        let rg = form_regions(&st, &a);
        let classes = group_regions(&rg);
        let sum = summarize(&classes);
        assert_eq!(sum.total_regions, 4);
        assert_eq!(sum.unique_classes, 1);
        assert_eq!(sum.max_period, 4);
        assert_eq!(sum.median_period, 4);
    }
}
