//! Stencil IR v2 pipeline — no-sampling dependence analysis.
//!
//! Per `STENCIL_IR_V2_DESIGN.md` §0 + §13: every correctness decision
//! derives from the iteration domain, the dependence polyhedra, and
//! the schedule. No member sampling, no ad-hoc enum variants, no
//! shape-specific heuristics.
//!
//! Pipeline:
//!
//! ```text
//! (Fuf, Assignment, class_of, class_members)
//!   └─> build_edge_dependences  →  Vec<BoundaryEdge>
//!         (enumerates every (consumer_class, consumer_member,
//!          boundary_pos) triple — NOT `members[0]` only)
//!   └─> classify_reads          →  BoundaryReads
//!         (piecewise-affine per-boundary read spec, a pure
//!          function of the BoundaryEdge distribution)
//!   └─> schedule                →  Kahn(intra-iter edges)
//!   └─> emit                    →  consumes ReadPlan, region-guarded
//! ```
//!
//! This file lands the first two stages (types + enumeration). The
//! classifier, schedule, and emitter consumer land in follow-ups;
//! they all share these types.
//!
//! Design references:
//! - TVM `src/s_tir/sblock_scope.cc` — the minimal per-buffer
//!   readers/writers dep-graph pattern.
//! - Halide `src/RealizationOrder.cpp` — the "dummy-node SCC collapse"
//!   for class-level topo sort.
//! - Halide `src/ScheduleFunctions.cpp` — `Container` stack +
//!   IfThenElse predicate threading for region guards.

// Fields on `ReadPlan`, `DepTarget`, and `BoundaryEdge` are
// intentionally declared ahead of their consumers (the classifier +
// the region-guarded emitter land in follow-up commits and read
// these fields then). Allow `dead_code` at module scope so clippy
// doesn't block the scaffolding landing.
#![allow(dead_code)]

use std::collections::{BTreeSet, HashMap};

use crate::fuf::{Fuf, FufInput, TileId};
use crate::solver::{Assignment, SubgraphId};

/// One dependence edge produced by enumerating every consumer
/// (class, member, boundary_pos) triple in the unrolled FUF. No
/// sampling — if a class has 16 members, we emit edges for all 16.
///
/// The consumer key is `(consumer_class, consumer_member,
/// boundary_pos)` — these three together identify exactly one
/// boundary input of one claimed subgraph. The producer fields
/// identify where that input comes from.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BoundaryEdge {
    pub consumer_class: usize,
    pub consumer_member: usize,
    pub boundary_pos: usize,

    pub producer_sg: SubgraphId,
    pub producer_class: usize,
    pub producer_member: usize,
    pub producer_tile: TileId,
    pub producer_slot: u8,
    /// Position of `producer_tile` within the producer subgraph's
    /// claim (sorted `TileId` order). Emitters name per-export idents
    /// by this position for multi-tile producer claims (e.g. a
    /// `FusedAddRmsNorm` claim `(Add, RmsNorm)` has pos 0 and 1 with
    /// distinct exports).
    pub producer_tile_pos: u8,
}

/// Canonical walk over a subgraph's boundary tile inputs.
///
/// One source of truth for boundary-position numbering. Any code
/// that speaks of "the Nth boundary input of subgraph S" must agree
/// with this walk. Used by ground truth, by edge enumeration, and
/// (once the rewrite lands) by the emitter's call-site argument
/// ordering.
///
/// Order: claimed tiles in their stored order; within each tile,
/// inputs in stored order; skip `FufInput::Tile` whose `id` is in
/// the claim (those are intra-subgraph); dedup on `(id, slot)` on
/// first occurrence.
///
/// The callback receives `(position, producer_tile, producer_slot)`.
pub(crate) fn walk_boundary_inputs(
    fuf: &Fuf,
    sfuf: &Assignment,
    sg: SubgraphId,
    mut visit: impl FnMut(usize, TileId, u8),
) {
    let claimed = sfuf.tiles_in_subgraph(sg);
    let claimed_set: BTreeSet<TileId> = claimed.iter().copied().collect();
    let mut seen: BTreeSet<(TileId, u8)> = BTreeSet::new();
    let mut pos = 0usize;
    for &tile in &claimed {
        for input in &fuf.get(tile).inputs {
            let FufInput::Tile { id, slot } = input else {
                continue;
            };
            if claimed_set.contains(id) {
                continue;
            }
            if !seen.insert((*id, *slot)) {
                continue;
            }
            visit(pos, *id, *slot);
            pos += 1;
        }
    }
}

/// Number of boundary inputs one subgraph has (same walk as
/// [`walk_boundary_inputs`]).
pub(crate) fn boundary_count(fuf: &Fuf, sfuf: &Assignment, sg: SubgraphId) -> usize {
    let mut count = 0;
    walk_boundary_inputs(fuf, sfuf, sg, |_, _, _| count += 1);
    count
}

/// Enumerate every dependence edge. No sampling.
///
/// For each class, for each member of that class, for each boundary
/// input of that member, emit one `BoundaryEdge` naming the producer
/// (class, member, tile, slot). A consumer input whose producer is
/// not in a known class (e.g. a raw extern like `InputIds`) yields
/// no edge.
///
/// Inputs:
/// - `fuf` / `sfuf` — the unrolled FUF + solver assignment.
/// - `class_of` — `SubgraphId → class_index`.
/// - `class_members` — `class_index → subgraphs in period order`.
///
/// Complexity: O(total tile-input count), a linear pass.
pub(crate) fn build_edge_dependences(
    fuf: &Fuf,
    sfuf: &Assignment,
    class_of: &HashMap<SubgraphId, usize>,
    class_members: &[Vec<SubgraphId>],
) -> Vec<BoundaryEdge> {
    let mut edges: Vec<BoundaryEdge> = Vec::new();
    for (consumer_class, members) in class_members.iter().enumerate() {
        for (consumer_member, &consumer_sg) in members.iter().enumerate() {
            walk_boundary_inputs(
                fuf,
                sfuf,
                consumer_sg,
                |boundary_pos, producer_tile, producer_slot| {
                    let Some(producer_sg) = sfuf.subgraph_of(producer_tile) else {
                        return;
                    };
                    if producer_sg == consumer_sg {
                        return;
                    }
                    let Some(&producer_class) = class_of.get(&producer_sg) else {
                        return;
                    };
                    let producer_member = match class_members[producer_class]
                        .iter()
                        .position(|&x| x == producer_sg)
                    {
                        Some(m) => m,
                        None => return,
                    };
                    let producer_claim = sfuf.tiles_in_subgraph(producer_sg);
                    let producer_tile_pos =
                        match producer_claim.iter().position(|&x| x == producer_tile) {
                            Some(p) => p as u8,
                            None => return,
                        };
                    edges.push(BoundaryEdge {
                        consumer_class,
                        consumer_member,
                        boundary_pos,
                        producer_sg,
                        producer_class,
                        producer_member,
                        producer_tile,
                        producer_slot,
                        producer_tile_pos,
                    });
                },
            );
        }
    }
    edges
}

// ─────────────────────────────────────────────────────────────────
// ReadPlan: piecewise-affine per-boundary read spec.
//
// A class's boundary has exactly one ReadPlan — a list of disjoint
// consumer-iter ranges, each pointing at either a pre-loop
// subgraph's local or a periodic class's output at some Δ_global.
//
// The classifier (landing in a follow-up commit) derives ReadPlans
// from the BoundaryEdge distribution and NOTHING ELSE. No sampling,
// no shape-specific heuristics, no member[0] lookups.
// ─────────────────────────────────────────────────────────────────

/// Half-open interval `[lo, hi)` over consumer *in-class* member
/// indices (NOT global __repeat). For a class `C` at `class_offset =
/// O`, the runtime guard for region `[lo, hi)` is
/// `(O + lo) <= __repeat && __repeat < (O + hi)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct IterRange {
    pub lo: usize,
    pub hi: usize,
}

impl IterRange {
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.hi - self.lo
    }
    #[allow(dead_code)]
    pub fn contains(&self, k: usize) -> bool {
        k >= self.lo && k < self.hi
    }
}

/// Where a consumer reads from.
///
/// `PreLoop` is a period-1 subgraph whose single invocation runs
/// outside the loop; its output is referenced verbatim.
///
/// `Periodic` is a periodic class's output, addressed by
/// `Δ_global = consumer_global_iter - producer_global_iter`. For a
/// well-formed transformer forward `Δ_global ∈ {0, 1}` — 0 is
/// intra-iter (producer ran earlier this iteration), 1 is a carry
/// (producer ran previous iteration). Δ_global ≥ 2 is rejected by
/// the classifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum DepTarget {
    PreLoop {
        producer_sg: SubgraphId,
        producer_tile: TileId,
        producer_slot: u8,
    },
    Periodic {
        producer_class: usize,
        producer_tile_pos: u8,
        producer_slot: u8,
        delta_global: i64,
    },
}

/// One region of a per-boundary ReadPlan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ReadRegion {
    pub range: IterRange,
    pub target: DepTarget,
}

/// Piecewise-affine read spec for one boundary of one periodic
/// consumer class.
///
/// `regions` are disjoint, sorted by `range.lo`, cover every member
/// iter from 0 to `class_members[consumer_class].len()`. Adjacent
/// regions with identical `target` are merged — the classifier
/// guarantees this canonical form.
///
/// Examples:
/// - **c2_b0 offset-gap** (the design doc's motivating case):
///     - `[0..1) → PreLoop { standalone-rmsnorm-sg }`
///     - `[1..N) → Periodic { class: c8, delta_global: 0 }`
/// - **residual-stream carry**:
///     - `[0..1) → PreLoop { embed-sg }`
///     - `[1..N) → Periodic { class: self, delta_global: 1 }`
/// - **pure intra-iter**:
///     - `[0..N) → Periodic { class: X, delta_global: 0 }`
#[derive(Debug, Clone)]
pub(crate) struct ReadPlan {
    pub consumer_class: usize,
    pub boundary_pos: usize,
    pub regions: Vec<ReadRegion>,
}

impl ReadPlan {
    /// Find the region whose `range` contains `member_k`. Returns
    /// `None` iff `member_k` is outside every region — a malformed
    /// plan that should not escape the classifier.
    #[allow(dead_code)]
    pub fn region_at(&self, member_k: usize) -> Option<&ReadRegion> {
        self.regions.iter().find(|r| r.range.contains(member_k))
    }
}

// ─────────────────────────────────────────────────────────────────
// Unit tests — narrow, self-contained. Cross-cutting invariant
// tests against fixtures live in `codegen::tests::invariants`.
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iter_range_semantics() {
        let r = IterRange { lo: 1, hi: 5 };
        assert_eq!(r.len(), 4);
        assert!(!r.contains(0));
        assert!(r.contains(1));
        assert!(r.contains(4));
        assert!(!r.contains(5));
    }

    #[test]
    fn read_plan_region_lookup() {
        let plan = ReadPlan {
            consumer_class: 2,
            boundary_pos: 0,
            regions: vec![
                ReadRegion {
                    range: IterRange { lo: 0, hi: 1 },
                    target: DepTarget::PreLoop {
                        producer_sg: SubgraphId(7),
                        producer_tile: TileId(42),
                        producer_slot: 0,
                    },
                },
                ReadRegion {
                    range: IterRange { lo: 1, hi: 16 },
                    target: DepTarget::Periodic {
                        producer_class: 8,
                        producer_tile_pos: 0,
                        producer_slot: 0,
                        delta_global: 0,
                    },
                },
            ],
        };
        assert!(matches!(
            plan.region_at(0).unwrap().target,
            DepTarget::PreLoop { .. }
        ));
        assert!(matches!(
            plan.region_at(1).unwrap().target,
            DepTarget::Periodic {
                delta_global: 0,
                ..
            }
        ));
        assert!(matches!(
            plan.region_at(15).unwrap().target,
            DepTarget::Periodic { .. }
        ));
        assert!(plan.region_at(16).is_none());
    }
}
