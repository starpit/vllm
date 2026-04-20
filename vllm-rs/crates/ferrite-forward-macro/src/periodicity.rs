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
    // Precompute 1-hop neighbor signatures. Intra-Region structure
    // alone (op tags + role + edges) does not distinguish two bare
    // `Gemm` Regions that live in different positions of a
    // transformer layer (e.g. attention-output vs MLP-down): both
    // are a single `Gemm` node over (tile_t, tile_d_out) with no
    // intra-Region edges. Their ambient role differs — attn-O
    // consumes an `Attention` Region, MLP-down consumes a `Mul`
    // (SiLU fusion). Mixing those two into one class was the root
    // cause of every heterogeneous-impl variant observed in the
    // class→impl consistency check.
    //
    // Using control-edge neighbors as a discriminator is cheap
    // (one pass over `rg.control`) and preserves layer-periodicity:
    // every layer's attn-O has the same (Attention → Gemm → Add)
    // neighborhood, every layer's MLP-down has the same (Mul →
    // Gemm → Add) neighborhood, so each splits into its own class
    // with period = num_layers.
    let neighbor_sigs = compute_neighbor_sigs(rg);

    // BTreeMap keyed on hash so iteration is deterministic. Ties on
    // hash (extremely unlikely, but possible) end up in the same
    // bucket — that's semantically what we want.
    let mut buckets: BTreeMap<u64, Vec<RegionId>> = BTreeMap::new();
    for r in &rg.regions {
        let h = canonical_hash_with_neighbors(r, &neighbor_sigs[r.id as usize]);
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

/// Canonical hash of one Region's intra-structure. Covers:
/// - Region.name
/// - Axis names (stable order = the order they appear on the
///   Region, which `form_one_region` populates first-seen per
///   claimed tile — deterministic per pattern)
/// - Node sequence: (op tag, role) per node
/// - Edge sequence: (src, dst, kind) sorted
///
/// Exposed for debugging / tests; callers generally want
/// `group_regions`, which additionally mixes in the 1-hop neighbor
/// signature for discrimination (see `canonical_hash_with_neighbors`).
pub fn canonical_hash(r: &Region) -> u64 {
    let mut h = DefaultHasher::new();
    hash_intra(r, &mut h);
    h.finish()
}

fn hash_intra(r: &Region, h: &mut DefaultHasher) {
    r.name.hash(h);
    for axis in &r.domain.axes {
        axis.name.hash(h);
    }
    for node in &r.nodes {
        node.op.tag.hash(h);
        (node.role as u8).hash(h);
    }
    for e in &r.edges {
        e.src.hash(h);
        e.dst.hash(h);
        (e.kind as u8).hash(h);
    }
}

/// Sorted op-tags of a Region's direct control-edge neighbors.
/// Deterministic (sorted), layer-invariant (neighbor ops are the
/// same across layer copies of one pattern), and cheap.
#[derive(Debug, Clone, Default)]
pub struct NeighborSig {
    pub upstream: Vec<&'static str>,
    pub downstream: Vec<&'static str>,
}

fn compute_neighbor_sigs(rg: &RegionGraph) -> Vec<NeighborSig> {
    let mut out: Vec<NeighborSig> = (0..rg.regions.len())
        .map(|_| NeighborSig::default())
        .collect();
    for ce in &rg.control {
        let src_name = rg.regions[ce.src as usize].name;
        let dst_name = rg.regions[ce.dst as usize].name;
        out[ce.dst as usize].upstream.push(src_name);
        out[ce.src as usize].downstream.push(dst_name);
    }
    for sig in &mut out {
        sig.upstream.sort();
        sig.downstream.sort();
    }
    out
}

/// Canonical hash including the Region's 1-hop neighbor signature.
/// This is the hash `group_regions` actually uses. The neighbor
/// signature splits classes that are intra-identical but live in
/// structurally-different positions of the outer graph (e.g. the
/// two bare-Gemm Regions per transformer layer).
pub fn canonical_hash_with_neighbors(r: &Region, sig: &NeighborSig) -> u64 {
    let mut h = DefaultHasher::new();
    hash_intra(r, &mut h);
    // Separator bytes so e.g. upstream=[a,b] + downstream=[] hashes
    // distinctly from upstream=[a] + downstream=[b].
    0xAAu8.hash(&mut h);
    for s in &sig.upstream {
        s.hash(&mut h);
    }
    0xBBu8.hash(&mut h);
    for s in &sig.downstream {
        s.hash(&mut h);
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
        // Two fully-independent rmsnorm tiles (no producer / consumer).
        // Same intra-structure AND same (empty) neighbor signature →
        // one class of period 2. Chaining the rmsnorms linearly would
        // *not* collapse them under the current hash because a linear
        // chain's boundary vs interior positions have different 1-hop
        // neighbor op-tags (see `boundary_positions_split_from_interior`).
        let fuf = fuf_from_ops(vec![(OpKind::RmsNorm, vec![]), (OpKind::RmsNorm, vec![])]);
        let st = subtile(&fuf);
        let a = assignment_from_groups(&[&[0], &[1]]);
        let rg = form_regions(&st, &a).graph;
        let classes = group_regions(&rg);
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0].period(), 2);
    }

    #[test]
    fn boundary_positions_split_from_interior() {
        // Linear chain: rms → rms → rms → rms. Intra-structure is
        // identical for all four, but neighbors differ:
        //   tile 0: up=[]         down=[RmsNorm]
        //   tile 1: up=[RmsNorm]  down=[RmsNorm]
        //   tile 2: up=[RmsNorm]  down=[RmsNorm]
        //   tile 3: up=[RmsNorm]  down=[]
        // Expect three classes: {0}, {1, 2}, {3}. This is the
        // load-bearing behavior that splits attn-O from MLP-down
        // in a real transformer.
        let fuf = fuf_from_ops(vec![
            (OpKind::RmsNorm, vec![]),
            (OpKind::RmsNorm, vec![tile_in(0)]),
            (OpKind::RmsNorm, vec![tile_in(1)]),
            (OpKind::RmsNorm, vec![tile_in(2)]),
        ]);
        let st = subtile(&fuf);
        let a = assignment_from_groups(&[&[0], &[1], &[2], &[3]]);
        let rg = form_regions(&st, &a).graph;
        let classes = group_regions(&rg);
        assert_eq!(classes.len(), 3);
        let mut periods: Vec<usize> = classes.iter().map(|c| c.period()).collect();
        periods.sort();
        assert_eq!(periods, vec![1, 1, 2]);
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
        let rg = form_regions(&st, &a).graph;
        let classes = group_regions(&rg);
        assert_eq!(classes.len(), 3);
        assert!(classes.iter().all(|c| c.period() == 1));
    }

    #[test]
    fn repeated_layer_blocks_collapse_proportionally() {
        // Three stacked "layer blocks" each (rmsnorm, gemm, add).
        // Nine tiles span three layer blocks; after the neighbor-
        // signature split, boundary copies (layer 0 and layer 2)
        // separate from the interior copy (layer 1). Each original
        // op position yields: two period-1 boundary classes + one
        // period-1 interior class. With three layers → 3 positions
        // × 3 classes = 9 classes total. The test proves the pass
        // runs; the interior class count is what scales with layer
        // count on a real transformer (period = num_layers − 2).
        let fuf = fuf_from_ops(vec![
            (OpKind::RmsNorm, vec![]),                   // 0
            (OpKind::Gemm, vec![tile_in(0)]),            // 1
            (OpKind::Add, vec![tile_in(1), tile_in(0)]), // 2
            (OpKind::RmsNorm, vec![tile_in(2)]),         // 3
            (OpKind::Gemm, vec![tile_in(3)]),            // 4
            (OpKind::Add, vec![tile_in(4), tile_in(2)]), // 5
            (OpKind::RmsNorm, vec![tile_in(5)]),         // 6
            (OpKind::Gemm, vec![tile_in(6)]),            // 7
            (OpKind::Add, vec![tile_in(7), tile_in(5)]), // 8
        ]);
        let st = subtile(&fuf);
        let a = assignment_from_groups(&[&[0], &[1], &[2], &[3], &[4], &[5], &[6], &[7], &[8]]);
        let rg = form_regions(&st, &a).graph;
        let classes = group_regions(&rg);
        // Every original Region is in exactly one class.
        let total: usize = classes.iter().map(|c| c.period()).sum();
        assert_eq!(total, 9);
        // Expected: 6 classes.
        //   RmsNorm: {R0} (head), {R3, R6} (interior)          → 2
        //   Gemm:    {R1, R4, R7} (upstream Rms, downstream Add) → 1
        //   Add:     {R2} (up includes Rms-shortcut), {R5}, {R8} → 3
        assert_eq!(classes.len(), 6);
    }

    #[test]
    fn four_layer_interior_collapses() {
        // Same structure as above but four layers. Interior copies
        // (layers 1 and 2) share identical neighborhoods per position
        // and merge → three period-2 classes + six period-1 boundary
        // classes = 9 total. Confirms the collapse factor grows with
        // layer count even under the stricter hash.
        let fuf = fuf_from_ops(vec![
            (OpKind::RmsNorm, vec![]),                    // 0
            (OpKind::Gemm, vec![tile_in(0)]),             // 1
            (OpKind::Add, vec![tile_in(1), tile_in(0)]),  // 2
            (OpKind::RmsNorm, vec![tile_in(2)]),          // 3
            (OpKind::Gemm, vec![tile_in(3)]),             // 4
            (OpKind::Add, vec![tile_in(4), tile_in(2)]),  // 5
            (OpKind::RmsNorm, vec![tile_in(5)]),          // 6
            (OpKind::Gemm, vec![tile_in(6)]),             // 7
            (OpKind::Add, vec![tile_in(7), tile_in(5)]),  // 8
            (OpKind::RmsNorm, vec![tile_in(8)]),          // 9
            (OpKind::Gemm, vec![tile_in(9)]),             // 10
            (OpKind::Add, vec![tile_in(10), tile_in(8)]), // 11
        ]);
        let st = subtile(&fuf);
        let a = assignment_from_groups(&[
            &[0],
            &[1],
            &[2],
            &[3],
            &[4],
            &[5],
            &[6],
            &[7],
            &[8],
            &[9],
            &[10],
            &[11],
        ]);
        let rg = form_regions(&st, &a).graph;
        let classes = group_regions(&rg);
        let total: usize = classes.iter().map(|c| c.period()).sum();
        assert_eq!(total, 12);
        // Expected classes: 6 total.
        //   RmsNorm: {R0} head, {R3,R6,R9} interior (period 3)
        //   Gemm:    {R1,R4,R7,R10} (period 4 — every Gemm has
        //            up=[Rms], down=[Add], independent of position)
        //   Add:     {R2} head (up=[Gemm,Rms]), {R5,R8} interior
        //            (period 2, up=[Add,Gemm]), {R11} tail (down=[])
        assert_eq!(classes.len(), 6);
        let mut periods: Vec<usize> = classes.iter().map(|c| c.period()).collect();
        periods.sort();
        assert_eq!(periods, vec![1, 1, 1, 2, 3, 4]);
    }

    #[test]
    fn summary_reports_counts() {
        // Four fully-independent rmsnorms — same intra-structure
        // and same (empty) neighborhood → one class of period 4.
        // A linear-chain variant would split into boundary +
        // interior classes; see `boundary_positions_split_from_interior`.
        let fuf = fuf_from_ops(vec![
            (OpKind::RmsNorm, vec![]),
            (OpKind::RmsNorm, vec![]),
            (OpKind::RmsNorm, vec![]),
            (OpKind::RmsNorm, vec![]),
        ]);
        let st = subtile(&fuf);
        let a = assignment_from_groups(&[&[0], &[1], &[2], &[3]]);
        let rg = form_regions(&st, &a).graph;
        let classes = group_regions(&rg);
        let sum = summarize(&classes);
        assert_eq!(sum.total_regions, 4);
        assert_eq!(sum.unique_classes, 1);
        assert_eq!(sum.max_period, 4);
        assert_eq!(sum.median_period, 4);
    }
}
