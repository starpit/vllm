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
// ClassDomain — explicit per-class iteration domain.
//
// Each class C fires at global iters `class_offsets[C] + k` for
// `k in 0..periods[C]`. The classifier consumes this domain to
// compute `delta_global = consumer_global_iter - producer_global_iter`
// for `Periodic` targets. Period-1 producers are classified as
// `PreLoop`.
//
// Keeping ClassDomain explicit (rather than implicit in the caller)
// is the point: the classifier is a pure function of edges + domain,
// no schedule heuristic baked in.
// ─────────────────────────────────────────────────────────────────

/// Per-class iteration domain. `periods[C]` is the number of
/// invocations of class `C` across a forward; `offsets[C]` is the
/// `__repeat` coordinate at which its first invocation runs. Parallel
/// to `class_members`.
#[derive(Debug, Clone)]
pub(crate) struct ClassDomain {
    pub periods: Vec<usize>,
    pub offsets: Vec<usize>,
}

/// Derive a `ClassDomain` from `class_members` + the existing
/// schedule's `class_offsets`.
#[allow(dead_code)]
pub(crate) fn class_domain(
    class_members: &[Vec<SubgraphId>],
    class_offsets: &[usize],
) -> ClassDomain {
    assert_eq!(
        class_members.len(),
        class_offsets.len(),
        "class_members and class_offsets must be parallel by class index"
    );
    ClassDomain {
        periods: class_members.iter().map(|m| m.len()).collect(),
        offsets: class_offsets.to_vec(),
    }
}

/// Classify the enumerated edges into per-boundary `ReadPlan`s.
///
/// A pure function of (edges, class_members, domain). No sampling,
/// no heuristic branch — every member's contribution to a boundary's
/// plan is computed from its own edge, not inferred from member[0].
///
/// Output: one entry per observed `(consumer_class, boundary_pos)`
/// pair. A boundary is omitted if not every member of the consumer
/// class has an edge (e.g. raw externs — consistent absence across
/// members is left to downstream categories to audit).
///
/// For each included boundary:
/// - `regions` cover `[0, periods[consumer_class])` contiguously.
/// - Adjacent regions with identical `DepTarget` are merged (canonical
///   form).
/// - `DepTarget::PreLoop` when the producer class has period 1.
/// - `DepTarget::Periodic` with
///   `delta_global = (offsets[c] + member_k) -
///                   (offsets[producer_class] + producer_member)`
///   otherwise.
///
/// The classifier does NOT reject `|delta_global| ≥ 2` — that is a
/// structural property of the fixture + schedule and shows up red in
/// `read_plan_classification::deltas_are_0_or_1`. Keeping the
/// classifier total means the red test names the offending
/// `(consumer_class, boundary_pos)`; a rejection at the classifier
/// would instead surface as a generic panic with less information.
#[allow(dead_code)]
pub(crate) fn classify_reads(
    edges: &[BoundaryEdge],
    class_members: &[Vec<SubgraphId>],
    domain: &ClassDomain,
) -> HashMap<(usize, usize), ReadPlan> {
    // Bucket edges by (consumer_class, boundary_pos, consumer_member).
    // Every consumer triple has at most one edge (enforced by
    // `build_edge_dependences` + unit test
    // `every_boundary_triple_yields_an_edge_or_is_raw_extern`).
    let mut by_triple: HashMap<(usize, usize, usize), &BoundaryEdge> = HashMap::new();
    let mut boundary_positions: HashMap<usize, BTreeSet<usize>> = HashMap::new();
    for e in edges {
        by_triple.insert((e.consumer_class, e.boundary_pos, e.consumer_member), e);
        boundary_positions
            .entry(e.consumer_class)
            .or_default()
            .insert(e.boundary_pos);
    }

    let mut out: HashMap<(usize, usize), ReadPlan> = HashMap::new();
    for (consumer_class, bps) in &boundary_positions {
        let period = class_members[*consumer_class].len();
        if period == 0 {
            continue;
        }
        for &bp in bps {
            // Resolve each member's target; bail if any member is
            // missing (mixed-extern boundary — a structural
            // irregularity left for downstream categories to flag).
            let mut per_member: Vec<DepTarget> = Vec::with_capacity(period);
            let mut complete = true;
            for k in 0..period {
                let Some(e) = by_triple.get(&(*consumer_class, bp, k)) else {
                    complete = false;
                    break;
                };
                per_member.push(resolve_dep_target(e, class_members, domain));
            }
            if !complete {
                continue;
            }

            // Canonical merge: collapse runs of identical targets.
            let mut regions: Vec<ReadRegion> = Vec::new();
            let mut run_start = 0usize;
            let mut run_target = per_member[0];
            for (k, &target) in per_member.iter().enumerate().skip(1) {
                if target == run_target {
                    continue;
                }
                regions.push(ReadRegion {
                    range: IterRange {
                        lo: run_start,
                        hi: k,
                    },
                    target: run_target,
                });
                run_start = k;
                run_target = target;
            }
            regions.push(ReadRegion {
                range: IterRange {
                    lo: run_start,
                    hi: period,
                },
                target: run_target,
            });

            out.insert(
                (*consumer_class, bp),
                ReadPlan {
                    consumer_class: *consumer_class,
                    boundary_pos: bp,
                    regions,
                },
            );
        }
    }
    out
}

/// Map one `BoundaryEdge` to a `DepTarget`. Period-1 producer →
/// `PreLoop`. Periodic producer → `Periodic` with `delta_global`
/// computed from the domain offsets. No sampling.
fn resolve_dep_target(
    e: &BoundaryEdge,
    class_members: &[Vec<SubgraphId>],
    domain: &ClassDomain,
) -> DepTarget {
    let producer_period = class_members[e.producer_class].len();
    if producer_period == 1 {
        return DepTarget::PreLoop {
            producer_sg: e.producer_sg,
            producer_tile: e.producer_tile,
            producer_slot: e.producer_slot,
        };
    }
    let consumer_global = domain.offsets[e.consumer_class] as i64 + e.consumer_member as i64;
    let producer_global = domain.offsets[e.producer_class] as i64 + e.producer_member as i64;
    let delta_global = consumer_global - producer_global;
    DepTarget::Periodic {
        producer_class: e.producer_class,
        producer_tile_pos: e.producer_tile_pos,
        producer_slot: e.producer_slot,
        delta_global,
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

    // Helpers for hand-building tiny fixtures to exercise the
    // classifier without running the full FUF pipeline.
    #[allow(clippy::too_many_arguments)]
    fn edge(
        consumer_class: usize,
        consumer_member: usize,
        boundary_pos: usize,
        producer_class: usize,
        producer_member: usize,
        producer_sg: u32,
        producer_tile: u32,
        producer_slot: u8,
        producer_tile_pos: u8,
    ) -> BoundaryEdge {
        BoundaryEdge {
            consumer_class,
            consumer_member,
            boundary_pos,
            producer_sg: SubgraphId(producer_sg),
            producer_class,
            producer_member,
            producer_tile: TileId(producer_tile),
            producer_slot,
            producer_tile_pos,
        }
    }

    #[test]
    fn classify_reads_intra_iter_full_range() {
        // Consumer class c2 (period=4) reads class c5 (period=4) at
        // bp 0, always at the same global iter (Δ_global=0). Expect
        // one region [0..4) → Periodic { class: 5, Δ=0 }.
        let class_members = vec![
            vec![], // c0 filler
            vec![], // c1 filler
            vec![
                SubgraphId(20),
                SubgraphId(21),
                SubgraphId(22),
                SubgraphId(23),
            ], // c2
            vec![], // c3 filler
            vec![], // c4 filler
            vec![
                SubgraphId(50),
                SubgraphId(51),
                SubgraphId(52),
                SubgraphId(53),
            ], // c5
        ];
        let offsets = vec![0, 0, 0, 0, 0, 0];
        let domain = class_domain(&class_members, &offsets);
        let edges = vec![
            edge(2, 0, 0, 5, 0, 50, 500, 0, 0),
            edge(2, 1, 0, 5, 1, 51, 510, 0, 0),
            edge(2, 2, 0, 5, 2, 52, 520, 0, 0),
            edge(2, 3, 0, 5, 3, 53, 530, 0, 0),
        ];
        let plans = classify_reads(&edges, &class_members, &domain);
        let plan = plans.get(&(2, 0)).expect("plan for (c2, bp0)");
        assert_eq!(plan.regions.len(), 1, "merged form");
        assert_eq!(plan.regions[0].range, IterRange { lo: 0, hi: 4 });
        match plan.regions[0].target {
            DepTarget::Periodic {
                producer_class,
                delta_global,
                ..
            } => {
                assert_eq!(producer_class, 5);
                assert_eq!(delta_global, 0);
            }
            _ => panic!("expected Periodic"),
        }
    }

    #[test]
    fn classify_reads_offset_gap_preloop_then_periodic() {
        // The motivating c2_b0 shape: member 0 reads a period-1
        // standalone subgraph; members 1..N read a periodic producer
        // at Δ=0. Expect two regions: [0..1) PreLoop, [1..N) Periodic.
        let class_members = vec![
            vec![SubgraphId(99)],                                 // c0 period-1 (pre-loop)
            vec![],                                               // c1 filler
            vec![SubgraphId(10), SubgraphId(11), SubgraphId(12)], // c2 period-3
            vec![],                                               // c3 filler
            vec![],                                               // c4 filler
            vec![],                                               // c5 filler
            vec![],                                               // c6 filler
            vec![],                                               // c7 filler
            vec![SubgraphId(80), SubgraphId(81), SubgraphId(82)], // c8 period-3
        ];
        let offsets = vec![0; 9];
        let domain = class_domain(&class_members, &offsets);
        let edges = vec![
            // c2 m0 bp0 → c0 m0 (period-1 → PreLoop)
            edge(2, 0, 0, 0, 0, 99, 990, 0, 0),
            // c2 m1 bp0 → c8 m1, c2 m2 bp0 → c8 m2 (periodic, Δ=0)
            edge(2, 1, 0, 8, 1, 81, 810, 0, 0),
            edge(2, 2, 0, 8, 2, 82, 820, 0, 0),
        ];
        let plans = classify_reads(&edges, &class_members, &domain);
        let plan = plans.get(&(2, 0)).expect("plan for (c2, bp0)");
        assert_eq!(plan.regions.len(), 2, "two regions after merge");
        assert_eq!(plan.regions[0].range, IterRange { lo: 0, hi: 1 });
        assert!(matches!(plan.regions[0].target, DepTarget::PreLoop { .. }));
        assert_eq!(plan.regions[1].range, IterRange { lo: 1, hi: 3 });
        match plan.regions[1].target {
            DepTarget::Periodic {
                producer_class,
                delta_global,
                ..
            } => {
                assert_eq!(producer_class, 8);
                assert_eq!(delta_global, 0);
            }
            _ => panic!("expected Periodic"),
        }
    }

    #[test]
    fn classify_reads_loop_carry_delta_one() {
        // Residual stream: member k reads self class at member k-1
        // (Δ_global = 1) for k ≥ 1, and a pre-loop producer at k=0.
        // Expect [0..1) PreLoop, [1..N) Periodic Δ=1.
        let class_members = vec![
            vec![SubgraphId(7)],                                  // c0 period-1 (embed)
            vec![SubgraphId(30), SubgraphId(31), SubgraphId(32)], // c1 period-3
        ];
        let offsets = vec![0, 0];
        let domain = class_domain(&class_members, &offsets);
        let edges = vec![
            edge(1, 0, 0, 0, 0, 7, 70, 0, 0),   // m0 ← pre-loop
            edge(1, 1, 0, 1, 0, 30, 300, 0, 0), // m1 ← self m0, Δ=1
            edge(1, 2, 0, 1, 1, 31, 310, 0, 0), // m2 ← self m1, Δ=1
        ];
        let plans = classify_reads(&edges, &class_members, &domain);
        let plan = plans.get(&(1, 0)).expect("plan for (c1, bp0)");
        assert_eq!(plan.regions.len(), 2);
        assert!(matches!(plan.regions[0].target, DepTarget::PreLoop { .. }));
        match plan.regions[1].target {
            DepTarget::Periodic {
                producer_class,
                delta_global,
                ..
            } => {
                assert_eq!(producer_class, 1);
                assert_eq!(delta_global, 1);
            }
            _ => panic!("expected Periodic"),
        }
    }

    #[test]
    fn classify_reads_skips_boundary_with_missing_member_edges() {
        // If only some members have edges (mixed-extern or
        // over-collapse), the classifier refuses to build a plan
        // rather than guessing. The boundary is absent from the
        // output map.
        let class_members = vec![
            vec![SubgraphId(10), SubgraphId(11), SubgraphId(12)],
            vec![SubgraphId(50), SubgraphId(51), SubgraphId(52)],
        ];
        let offsets = vec![0, 0];
        let domain = class_domain(&class_members, &offsets);
        let edges = vec![
            edge(0, 0, 0, 1, 0, 50, 500, 0, 0),
            // m1 deliberately missing
            edge(0, 2, 0, 1, 2, 52, 520, 0, 0),
        ];
        let plans = classify_reads(&edges, &class_members, &domain);
        assert!(
            !plans.contains_key(&(0, 0)),
            "partial boundary coverage produces no ReadPlan",
        );
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
