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
// Schedule — topological order over Δ_global=0 edges.
//
// `schedule_from_edges` is a pure function of (edges, domain). It
// builds the class-level dependence graph restricted to intra-
// iteration edges (Δ_global = 0) and returns a Kahn topological
// order over all classes, tie-breaking by ascending class index for
// determinism. Carry edges (Δ_global ≥ 1) are ignored — they do not
// constrain intra-iteration firing order.
//
// Cycle detection is structural: any class whose in-degree never
// reaches zero is reported in `ScheduleCycle::unscheduled_classes`.
// "No class with a Δ=0 cycle silently emitted" — per invariant
// category 15 (c), the caller treats an `Err` as a refusal gate;
// the rewrite never inserts a class into the emission order whose
// intra-iter deps are not satisfied.
//
// This replaces today's `ClassSchedule` offset-sort. The offset-
// sort happened to produce a topologically-valid order on shipping
// fleets because offsets were derived from periods, but it could
// not catch structural cycles — a bug the old pipeline masked by
// never tripping it. Kahn makes the contract explicit.
// ─────────────────────────────────────────────────────────────────

/// Linear topological order over classes, respecting every
/// `Δ_global = 0` dependence edge. Every class index `0..n` appears
/// exactly once.
#[derive(Debug, Clone)]
pub(crate) struct Schedule {
    pub order: Vec<usize>,
}

impl Schedule {
    #[allow(dead_code)]
    pub fn position_of(&self, class: usize) -> Option<usize> {
        self.order.iter().position(|&c| c == class)
    }
}

/// Error returned by [`schedule_from_edges`] when the Δ=0 dependence
/// graph contains a cycle. Lists every class that could not be placed
/// — at least one of them is in the cycle; the rest are transitively
/// downstream. Callers surface this as a structural refusal rather
/// than silently emitting a partial order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ScheduleCycle {
    pub unscheduled_classes: Vec<usize>,
}

/// Kahn topological sort over the Δ_global=0 sub-graph.
///
/// Edges with `Δ_global ≠ 0` (carries) are ignored — they do not
/// constrain intra-iteration firing order. Ties broken by ascending
/// class index (min-first via `BTreeSet`) so the order is
/// deterministic across invocations and implementations.
///
/// A class with no incoming Δ=0 edges is ready immediately; classes
/// with no outgoing Δ=0 edges just appear in the order without
/// further constraint. Filler classes (not present as either end of
/// any edge) appear in ascending index order at the front.
///
/// Complexity: O(|edges| + n_classes · log n_classes) — the log
/// factor is the `BTreeSet` ready-queue operations.
#[allow(dead_code)]
pub(crate) fn schedule_from_edges(
    edges: &[BoundaryEdge],
    domain: &ClassDomain,
) -> Result<Schedule, ScheduleCycle> {
    let n = domain.periods.len();

    // Class-level successor / predecessor sets, deduped. An edge
    // is included iff `delta_global = (O_c + m_c) - (O_p + m_p) = 0`.
    let mut succ: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    let mut preds: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for e in edges {
        if e.consumer_class >= n || e.producer_class >= n {
            continue;
        }
        let consumer_global = domain.offsets[e.consumer_class] as i64 + e.consumer_member as i64;
        let producer_global = domain.offsets[e.producer_class] as i64 + e.producer_member as i64;
        if consumer_global - producer_global != 0 {
            continue;
        }
        if e.producer_class == e.consumer_class {
            // Self-intra-iter (producer_sg != consumer_sg but same
            // class, same global iter) would require two distinct
            // members at the same offset+member — impossible by
            // construction. Guarded here defensively; a real
            // occurrence would indicate an upstream invariant break.
            continue;
        }
        if succ[e.producer_class].insert(e.consumer_class) {
            preds[e.consumer_class].insert(e.producer_class);
        }
    }

    let mut indeg: Vec<usize> = preds.iter().map(|s| s.len()).collect();
    let mut ready: BTreeSet<usize> = (0..n).filter(|&c| indeg[c] == 0).collect();
    let mut order: Vec<usize> = Vec::with_capacity(n);
    while let Some(&c) = ready.iter().next() {
        ready.remove(&c);
        order.push(c);
        for &s in &succ[c] {
            indeg[s] = indeg[s].saturating_sub(1);
            if indeg[s] == 0 {
                ready.insert(s);
            }
        }
    }

    if order.len() == n {
        Ok(Schedule { order })
    } else {
        let placed: BTreeSet<usize> = order.iter().copied().collect();
        let unscheduled: Vec<usize> = (0..n).filter(|c| !placed.contains(c)).collect();
        Err(ScheduleCycle {
            unscheduled_classes: unscheduled,
        })
    }
}

// ─────────────────────────────────────────────────────────────────
// Partition — project Schedule.order onto (pre_loop, periodic,
// in_loop_short, post_loop) using ClassDomain.periods + edges.
//
// This replaces today's `ClassSchedule` pre/periodic/post split.
// Four buckets, total function of (order, periods, edges):
//
//   - A class with `periods[c] > 1` is `periodic`, in Kahn order.
//   - A class with `periods[c] == 1` whose Kahn position in
//     `order` falls strictly BEFORE the first periodic class is
//     `pre_loop`.
//   - A class with `periods[c] == 1` whose Kahn position falls
//     strictly AFTER the last periodic class is `post_loop`.
//   - A class with `periods[c] == 1` whose Kahn position falls
//     between periodic classes is `in_loop_short` — it fires
//     INSIDE the loop body at a specific `__repeat` offset,
//     guarded by `if __repeat == offset { … }`. The offset is
//     derived from the `BoundaryEdge` set: for each edge between
//     the short class and a periodic neighbour at the neighbour's
//     member `m`, the candidate offset is `offsets[P] + m`. We
//     take MAX over consumer-side edges (C must fire after every
//     P-iter it reads) and MIN over producer-side edges (C must
//     fire before every P-iter that reads its output). Mixed
//     direction is feasible iff `max_consumer ≤ min_producer`.
//
// Reassembly invariant: `flat_order()` walks the Kahn interior
// using the saved `interior_classes` and concatenates pre_loop +
// interior + post_loop to rebuild `schedule.order` verbatim.
//
// This mirrors the legacy `StencilBundle::schedule` aliased-
// emittable reroute but is driven by the `BoundaryEdge` set (no
// `is_aliased_emittable` sampling, no dependence on `class_edges`).
// An interleaved period-1 class whose offset is underivable from
// edges (no edges to/from any periodic class) falls back to
// `post_loop` — degenerate, flagged by category 16 invariants
// when it breaks `flat_order == order`.
// ─────────────────────────────────────────────────────────────────

/// Partition of `Schedule.order` into four buckets by period +
/// edge-derived offset.
///
/// - `pre_loop` / `post_loop` — period-1 classes outside the
///   periodic loop window.
/// - `periodic` — period>1 classes in Kahn order.
/// - `in_loop_short` — period-1 classes interleaved with periodic,
///   each with a `(class, offset)` pair where `offset` is the
///   `__repeat` coordinate at which the class fires inside the
///   loop body.
/// - `interior_classes` — every class in the Kahn interior
///   (periodic + in_loop_short) preserving their relative Kahn
///   order. Used by `flat_order()` to reassemble `schedule.order`.
#[derive(Debug, Clone)]
pub(crate) struct PartitionedSchedule {
    pub pre_loop: Vec<usize>,
    pub periodic: Vec<usize>,
    pub in_loop_short: Vec<(usize, usize)>,
    pub post_loop: Vec<usize>,
    pub interior_classes: Vec<usize>,
}

impl PartitionedSchedule {
    /// Rebuild `schedule.order` by walking `pre_loop`, then
    /// `interior_classes` (which preserves the Kahn interleaving
    /// of `periodic` and `in_loop_short`), then `post_loop`.
    #[allow(dead_code)]
    pub fn flat_order(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(
            self.pre_loop.len() + self.interior_classes.len() + self.post_loop.len(),
        );
        out.extend(&self.pre_loop);
        out.extend(&self.interior_classes);
        out.extend(&self.post_loop);
        out
    }

    /// Offset assigned to an `in_loop_short` class. Returns `None`
    /// for classes not in `in_loop_short`.
    #[allow(dead_code)]
    pub fn short_offset(&self, class: usize) -> Option<usize> {
        self.in_loop_short
            .iter()
            .find(|(c, _)| *c == class)
            .map(|(_, off)| *off)
    }
}

/// Derive the `__repeat` offset at which a period-1 class `c`
/// fires, from `BoundaryEdge`s between `c` and any periodic class.
///
/// Legacy semantics (mirroring `StencilBundle::schedule`'s aliased-
/// emittable reroute but keyed on `BoundaryEdge` instead of
/// `class_edges`):
///
/// - `c` as CONSUMER of periodic `P` at P-member `m_P`: `c` must
///   fire at or after `offsets[P] + m_P` so that P's output is
///   available. Take the MAX over all consumer-side candidates.
/// - `c` as PRODUCER for periodic `P` at P-member `m_P`: `c` must
///   fire at or before `offsets[P] + m_P` so that every consumer
///   iter can read `c`'s output (directly, or carried forward).
///   Take the MIN over all producer-side candidates.
/// - Mixed-direction feasibility: `max_consumer ≤ min_producer`.
///   Picks `max(max_consumer, min_producer)` so both sides satisfy.
///
/// Returns `None` if the offset is underivable (no edges to any
/// periodic class, or infeasible mixed-direction constraint, or a
/// candidate falls outside `[0, max_period)`). Callers treat `None`
/// as "leave the class in its current bucket" — the structural
/// invariant tests flag the downstream emission gap.
fn derive_short_offset(
    c: usize,
    edges: &[BoundaryEdge],
    domain: &ClassDomain,
    periodic_set: &BTreeSet<usize>,
) -> Option<usize> {
    let max_period = *domain.periods.iter().max().unwrap_or(&0);
    if max_period == 0 {
        return None;
    }
    let mut min_producer: Option<usize> = None;
    let mut max_consumer: Option<usize> = None;
    for e in edges {
        let (other_class, other_member, c_is_consumer) =
            if e.consumer_class == c && periodic_set.contains(&e.producer_class) {
                (e.producer_class, e.producer_member, true)
            } else if e.producer_class == c && periodic_set.contains(&e.consumer_class) {
                (e.consumer_class, e.consumer_member, false)
            } else {
                continue;
            };
        let candidate = domain.offsets[other_class] + other_member;
        if candidate >= max_period {
            return None;
        }
        if c_is_consumer {
            max_consumer = Some(max_consumer.map_or(candidate, |m| m.max(candidate)));
        } else {
            min_producer = Some(min_producer.map_or(candidate, |m| m.min(candidate)));
        }
    }
    match (max_consumer, min_producer) {
        (None, None) => None,
        (Some(cmax), None) => Some(cmax),
        (None, Some(pmin)) => Some(pmin),
        (Some(cmax), Some(pmin)) => {
            if cmax > pmin {
                None
            } else {
                Some(cmax.max(pmin))
            }
        }
    }
}

/// Project `schedule.order` onto four buckets with edge-derived
/// `in_loop_short` offsets.
///
/// Walk `order` once, computing first / last periodic positions
/// in a first pass, then classifying each class in a second pass.
/// A period-1 class whose Kahn position is strictly between the
/// first and last periodic positions goes to `in_loop_short` with
/// the offset from `derive_short_offset`; if that derivation
/// returns `None`, the class falls back to `post_loop` (degenerate;
/// `flat_order != schedule.order` is the red invariant that
/// surfaces the gap).
///
/// Complexity: O(n_classes + |edges|).
#[allow(dead_code)]
pub(crate) fn partition_schedule(
    schedule: &Schedule,
    domain: &ClassDomain,
    edges: &[BoundaryEdge],
) -> PartitionedSchedule {
    // First pass: find the Kahn positions of the first and last
    // periodic classes.
    let mut first_periodic_pos: Option<usize> = None;
    let mut last_periodic_pos: Option<usize> = None;
    for (pos, &c) in schedule.order.iter().enumerate() {
        if c >= domain.periods.len() {
            continue;
        }
        if domain.periods[c] > 1 {
            if first_periodic_pos.is_none() {
                first_periodic_pos = Some(pos);
            }
            last_periodic_pos = Some(pos);
        }
    }

    // Precompute the set of periodic classes for
    // `derive_short_offset`.
    let periodic_set: BTreeSet<usize> = schedule
        .order
        .iter()
        .copied()
        .filter(|&c| c < domain.periods.len() && domain.periods[c] > 1)
        .collect();

    let mut pre_loop: Vec<usize> = Vec::new();
    let mut periodic: Vec<usize> = Vec::new();
    let mut in_loop_short: Vec<(usize, usize)> = Vec::new();
    let mut post_loop: Vec<usize> = Vec::new();
    let mut interior_classes: Vec<usize> = Vec::new();

    for (pos, &c) in schedule.order.iter().enumerate() {
        if c >= domain.periods.len() {
            continue;
        }
        let period = domain.periods[c];
        if period > 1 {
            periodic.push(c);
            interior_classes.push(c);
            continue;
        }
        // period == 1 path.
        let before_first = match first_periodic_pos {
            Some(fp) => pos < fp,
            None => true, // no periodic class exists → everything pre_loop
        };
        let after_last = match last_periodic_pos {
            Some(lp) => pos > lp,
            None => false,
        };
        if before_first {
            pre_loop.push(c);
        } else if after_last {
            post_loop.push(c);
        } else {
            // Interleaved: derive offset from edges; fall back to
            // post_loop when underivable (structural flag caught by
            // category 16 flat_order invariants).
            match derive_short_offset(c, edges, domain, &periodic_set) {
                Some(offset) => {
                    in_loop_short.push((c, offset));
                    interior_classes.push(c);
                }
                None => {
                    post_loop.push(c);
                }
            }
        }
    }

    PartitionedSchedule {
        pre_loop,
        periodic,
        in_loop_short,
        post_loop,
        interior_classes,
    }
}

// ─────────────────────────────────────────────────────────────────
// StencilPipeline — one bundle the emitter consumes as its sole
// source of truth.
//
// The rewrite's step 6 replaces `class_input_provenance` (per-rep,
// `InputOrigin`-typed, heuristic) + `ClassSchedule` (offset-sort
// partition) with two lookups off this bundle:
//
//   - `boundary_plan(class, bp)` → the full ReadPlan for a class's
//     boundary — used to emit region-guarded reads.
//   - `target_for(class, bp, member_k)` → the DepTarget for one
//     member's read — used to emit one fragment call-site's
//     argument list.
//
// The bundle owns all four stages so downstream callers borrow
// selectively. No emitter changes land in the commit that
// introduces this type; the category 17 invariants prove the API
// is TOTAL over the queries the emitter will make before the
// emission rewire starts.
// ─────────────────────────────────────────────────────────────────

/// All four stages of the pipeline bundled for one fixture: edge
/// enumeration, domain, topological schedule, partition, and the
/// per-boundary ReadPlan map.
///
/// Built from `(fuf, sfuf, class_of, class_members, class_offsets)`
/// via [`StencilPipeline::build`]. Returns `Err(ScheduleCycle)` iff
/// the Δ_global=0 edge sub-graph has a cycle — the caller surfaces
/// this as a structural refusal rather than silently emitting a
/// partial order.
#[derive(Debug, Clone)]
pub(crate) struct StencilPipeline {
    pub edges: Vec<BoundaryEdge>,
    pub domain: ClassDomain,
    pub schedule: Schedule,
    pub partition: PartitionedSchedule,
    pub plans: HashMap<(usize, usize), ReadPlan>,
}

impl StencilPipeline {
    /// Compose every pipeline stage in order. Each stage is a pure
    /// function of its predecessors' outputs — the whole bundle is
    /// deterministic.
    #[allow(dead_code)]
    pub fn build(
        fuf: &Fuf,
        sfuf: &Assignment,
        class_of: &HashMap<SubgraphId, usize>,
        class_members: &[Vec<SubgraphId>],
        class_offsets: &[usize],
    ) -> Result<Self, ScheduleCycle> {
        let edges = build_edge_dependences(fuf, sfuf, class_of, class_members);
        let domain = class_domain(class_members, class_offsets);
        let schedule = schedule_from_edges(&edges, &domain)?;
        let partition = partition_schedule(&schedule, &domain, &edges);
        let plans = classify_reads(&edges, class_members, &domain);
        Ok(StencilPipeline {
            edges,
            domain,
            schedule,
            partition,
            plans,
        })
    }

    /// The ReadPlan for one `(consumer_class, boundary_pos)` pair,
    /// or `None` if no plan exists — the classifier omits boundaries
    /// whose members do not all have a dependence edge (raw externs
    /// or over-collapsed shapes). Category 14 tracks boundaries that
    /// SHOULD have a plan; category 17 tracks boundaries the emitter
    /// will query.
    #[allow(dead_code)]
    pub fn boundary_plan(&self, class: usize, boundary_pos: usize) -> Option<&ReadPlan> {
        self.plans.get(&(class, boundary_pos))
    }

    /// The `DepTarget` that one member's read of a boundary resolves
    /// to. `None` iff no plan covers `(class, boundary_pos)` or
    /// `member_k` lies outside the plan's covered range.
    ///
    /// This is the single lookup the emitter (step 6 follow-up) will
    /// make for every boundary-input argument it emits, replacing
    /// `class_input_provenance`'s per-slot `InputOrigin`. The
    /// category 17 invariants pin the API as TOTAL over the
    /// emitter's query set before the rewire lands.
    #[allow(dead_code)]
    pub fn target_for(
        &self,
        class: usize,
        boundary_pos: usize,
        member_k: usize,
    ) -> Option<DepTarget> {
        self.boundary_plan(class, boundary_pos)?
            .region_at(member_k)
            .map(|r| r.target)
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
    fn schedule_ignores_carry_edges() {
        // c0 period-3 carries from itself (Δ=1). No intra-iter edges.
        // Expect linear order [0] — carry does not constrain class
        // ordering.
        let class_members = vec![vec![SubgraphId(10), SubgraphId(11), SubgraphId(12)]];
        let offsets = vec![0];
        let domain = class_domain(&class_members, &offsets);
        let edges = vec![
            // m1 ← m0 (Δ=1, carry), m2 ← m1 (Δ=1, carry).
            edge(0, 1, 0, 0, 0, 10, 100, 0, 0),
            edge(0, 2, 0, 0, 1, 11, 110, 0, 0),
        ];
        let sched = schedule_from_edges(&edges, &domain).expect("no cycle");
        assert_eq!(sched.order, vec![0]);
    }

    #[test]
    fn schedule_respects_intra_iter_edge() {
        // c0 feeds c1 at Δ=0 on every iter. c1 must appear after c0.
        let class_members = vec![
            vec![SubgraphId(10), SubgraphId(11)],
            vec![SubgraphId(20), SubgraphId(21)],
        ];
        let offsets = vec![0, 0];
        let domain = class_domain(&class_members, &offsets);
        let edges = vec![
            edge(1, 0, 0, 0, 0, 10, 100, 0, 0),
            edge(1, 1, 0, 0, 1, 11, 110, 0, 0),
        ];
        let sched = schedule_from_edges(&edges, &domain).expect("no cycle");
        assert_eq!(sched.order, vec![0, 1]);
        assert!(sched.position_of(0).unwrap() < sched.position_of(1).unwrap());
    }

    #[test]
    fn schedule_ties_break_by_class_index() {
        // No edges at all — all classes ready. Order must be ascending
        // class index so the output is deterministic across runs.
        let class_members = vec![
            vec![SubgraphId(10)],
            vec![SubgraphId(20)],
            vec![SubgraphId(30)],
        ];
        let offsets = vec![0, 0, 0];
        let domain = class_domain(&class_members, &offsets);
        let sched = schedule_from_edges(&[], &domain).expect("no cycle");
        assert_eq!(sched.order, vec![0, 1, 2]);
    }

    #[test]
    fn schedule_detects_cycle() {
        // c0 ←→ c1 at Δ=0. Both classes stay unscheduled.
        let class_members = vec![
            vec![SubgraphId(10), SubgraphId(11)],
            vec![SubgraphId(20), SubgraphId(21)],
        ];
        let offsets = vec![0, 0];
        let domain = class_domain(&class_members, &offsets);
        let edges = vec![
            // c1 ← c0 (Δ=0)
            edge(1, 0, 0, 0, 0, 10, 100, 0, 0),
            edge(1, 1, 0, 0, 1, 11, 110, 0, 0),
            // c0 ← c1 (Δ=0) — cycle
            edge(0, 0, 0, 1, 0, 20, 200, 0, 0),
            edge(0, 1, 0, 1, 1, 21, 210, 0, 0),
        ];
        let err = schedule_from_edges(&edges, &domain).unwrap_err();
        assert_eq!(err.unscheduled_classes, vec![0, 1]);
    }

    #[test]
    fn schedule_mixes_ready_and_constrained_classes() {
        // c2 depends on c0 at Δ=0; c1 is unconstrained. Expect c0, c1
        // ready immediately (index tie-break), c2 released by c0.
        let class_members = vec![
            vec![SubgraphId(10)],
            vec![SubgraphId(20)],
            vec![SubgraphId(30)],
        ];
        let offsets = vec![0, 0, 0];
        let domain = class_domain(&class_members, &offsets);
        let edges = vec![edge(2, 0, 0, 0, 0, 10, 100, 0, 0)];
        let sched = schedule_from_edges(&edges, &domain).expect("no cycle");
        // c0 and c1 both ready at start; tie-break index chooses 0 then 1.
        // c2 released by c0 so appears at index 2.
        assert_eq!(sched.order, vec![0, 1, 2]);
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

    fn dom(periods: Vec<usize>) -> ClassDomain {
        let offsets = vec![0usize; periods.len()];
        ClassDomain { periods, offsets }
    }

    #[test]
    fn partition_all_pre_loop_when_no_periodic() {
        // All classes are period-1. Every one lands in pre_loop.
        let sched = Schedule {
            order: vec![0, 1, 2],
        };
        let domain = dom(vec![1, 1, 1]);
        let part = partition_schedule(&sched, &domain, &[]);
        assert_eq!(part.pre_loop, vec![0, 1, 2]);
        assert!(part.periodic.is_empty());
        assert!(part.in_loop_short.is_empty());
        assert!(part.post_loop.is_empty());
    }

    #[test]
    fn partition_typical_shape() {
        // order: [c0 (p=1), c1 (p=1), c2 (p=16), c3 (p=16), c4 (p=1)]
        // expect: pre_loop=[0,1], periodic=[2,3], post_loop=[4].
        let sched = Schedule {
            order: vec![0, 1, 2, 3, 4],
        };
        let domain = dom(vec![1, 1, 16, 16, 1]);
        let part = partition_schedule(&sched, &domain, &[]);
        assert_eq!(part.pre_loop, vec![0, 1]);
        assert_eq!(part.periodic, vec![2, 3]);
        assert!(part.in_loop_short.is_empty());
        assert_eq!(part.post_loop, vec![4]);
    }

    #[test]
    fn partition_period_one_interleaved_reroutes_to_in_loop_short() {
        // A period-1 class between two periodic classes is an
        // aliased-emittable reroute candidate. With edges that
        // identify the offset (c1 reads c0 at iter 5, so c1 fires
        // at __repeat = 5 = offset(c0) + 5), c1 moves to
        // `in_loop_short` with offset 5 and c3 (post the last
        // periodic) stays in `post_loop`.
        let sched = Schedule {
            order: vec![0, 1, 2, 3],
        };
        let periods = vec![16, 1, 16, 1];
        // Offsets mirror today's "short classes start late" rule
        // so that offsets[0]=0 and c1's derived offset is directly
        // the consumer iter.
        let offsets = vec![0, 0, 0, 0];
        let class_members = vec![
            (0..16).map(SubgraphId).collect::<Vec<_>>(),
            vec![SubgraphId(100)],
            (0..16).map(|i| SubgraphId(200 + i)).collect::<Vec<_>>(),
            vec![SubgraphId(300)],
        ];
        let domain = ClassDomain { periods, offsets };
        // Single edge: c1 consumes c0 at c0's member 5. Offset = 5.
        let edges = vec![edge(1, 0, 0, 0, 5, 5, 50, 0, 0)];
        let part = partition_schedule(&sched, &domain, &edges);
        let _ = class_members; // silence unused (domain carries lengths)
        assert_eq!(part.pre_loop, Vec::<usize>::new());
        assert_eq!(part.periodic, vec![0, 2]);
        assert_eq!(part.in_loop_short, vec![(1, 5)]);
        assert_eq!(part.post_loop, vec![3]);
        assert_eq!(part.interior_classes, vec![0, 1, 2]);
        assert_eq!(part.flat_order(), sched.order);
    }

    #[test]
    fn partition_interleaved_without_edges_falls_back_to_post_loop() {
        // When offset is underivable (no edges to/from a periodic
        // neighbour), the interleaved period-1 class degrades into
        // `post_loop`. flat_order then differs from schedule.order
        // — category 16 (c) is the red that catches this on the
        // real fixture.
        let sched = Schedule {
            order: vec![0, 1, 2],
        };
        let domain = dom(vec![16, 1, 16]);
        let part = partition_schedule(&sched, &domain, &[]);
        assert!(part.in_loop_short.is_empty());
        assert_eq!(part.periodic, vec![0, 2]);
        assert_eq!(part.post_loop, vec![1]);
        assert_ne!(part.flat_order(), sched.order);
    }

    #[test]
    fn partition_every_class_appears_exactly_once() {
        let sched = Schedule {
            order: vec![3, 1, 0, 4, 2],
        };
        let domain = dom(vec![1, 12, 1, 12, 1]);
        let part = partition_schedule(&sched, &domain, &[]);
        let mut seen: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for &c in part
            .pre_loop
            .iter()
            .chain(part.periodic.iter())
            .chain(part.in_loop_short.iter().map(|(c, _)| c))
            .chain(part.post_loop.iter())
        {
            assert!(seen.insert(c), "class {c} appeared twice");
        }
        assert_eq!(seen.len(), 5);
    }

    #[test]
    fn partition_flat_order_matches_schedule_when_no_interleave() {
        // When period-1 classes sit cleanly around periodic ones,
        // flat_order reassembles the schedule verbatim.
        let sched = Schedule {
            order: vec![0, 1, 2, 3, 4, 5],
        };
        let domain = dom(vec![1, 1, 8, 8, 1, 1]);
        let part = partition_schedule(&sched, &domain, &[]);
        assert_eq!(part.flat_order(), sched.order);
    }

    #[test]
    fn partition_all_periodic() {
        let sched = Schedule {
            order: vec![0, 1, 2],
        };
        let domain = dom(vec![4, 4, 4]);
        let part = partition_schedule(&sched, &domain, &[]);
        assert!(part.pre_loop.is_empty());
        assert_eq!(part.periodic, vec![0, 1, 2]);
        assert!(part.in_loop_short.is_empty());
        assert!(part.post_loop.is_empty());
    }

    #[test]
    fn partition_short_offset_mixed_feasibility() {
        // c1 is period-1 interleaved. c0 is periodic period-16;
        // c1 consumes c0 at iter 3 (lower bound 3) AND produces
        // for c0 at iter 10 (upper bound 10). Offset = max(3, 10 → from
        // producer min) — wait: producer-side takes MIN (c1 must
        // fire before iter 10), consumer-side takes MAX (c1 must
        // fire after iter 3). Feasibility: 3 ≤ 10. Pick max(3,10)=10.
        let sched = Schedule {
            order: vec![0, 1, 2],
        };
        let domain = ClassDomain {
            periods: vec![16, 1, 16],
            offsets: vec![0, 0, 0],
        };
        let edges = vec![
            // c1 consumes from c0 at m=3
            edge(1, 0, 0, 0, 3, 3, 30, 0, 0),
            // c0 at m=10 consumes from c1
            edge(0, 10, 0, 1, 0, 1, 11, 0, 0),
        ];
        let part = partition_schedule(&sched, &domain, &edges);
        assert_eq!(part.in_loop_short, vec![(1, 10)]);
    }

    #[test]
    fn partition_short_offset_infeasible_falls_back() {
        // c1 would need to fire AFTER iter 10 (consumer) but BEFORE
        // iter 3 (producer). Infeasible → None → post_loop fallback.
        let sched = Schedule {
            order: vec![0, 1, 2],
        };
        let domain = ClassDomain {
            periods: vec![16, 1, 16],
            offsets: vec![0, 0, 0],
        };
        let edges = vec![
            // c1 consumes c0 at m=10 → lower bound 10
            edge(1, 0, 0, 0, 10, 10, 100, 0, 0),
            // c0 at m=3 consumes from c1 → upper bound 3
            edge(0, 3, 0, 1, 0, 1, 11, 0, 0),
        ];
        let part = partition_schedule(&sched, &domain, &edges);
        assert!(part.in_loop_short.is_empty());
        assert_eq!(part.post_loop, vec![1]);
    }

    /// Hand-built `StencilPipeline` with one periodic class that
    /// reads a PreLoop producer at m=0 and a Periodic carry at m≥1.
    /// `target_for` must return the right DepTarget at every
    /// member_k and `None` past the end.
    #[test]
    fn pipeline_lookup_target_for() {
        // Mirror of classify_reads_loop_carry_delta_one's shape
        // (period-3 self-carry with pre-loop init at m=0).
        let mut plans: HashMap<(usize, usize), ReadPlan> = HashMap::new();
        plans.insert(
            (1, 0),
            ReadPlan {
                consumer_class: 1,
                boundary_pos: 0,
                regions: vec![
                    ReadRegion {
                        range: IterRange { lo: 0, hi: 1 },
                        target: DepTarget::PreLoop {
                            producer_sg: SubgraphId(7),
                            producer_tile: TileId(70),
                            producer_slot: 0,
                        },
                    },
                    ReadRegion {
                        range: IterRange { lo: 1, hi: 3 },
                        target: DepTarget::Periodic {
                            producer_class: 1,
                            producer_tile_pos: 0,
                            producer_slot: 0,
                            delta_global: 1,
                        },
                    },
                ],
            },
        );
        let pipe = StencilPipeline {
            edges: vec![],
            domain: ClassDomain {
                periods: vec![1, 3],
                offsets: vec![0, 0],
            },
            schedule: Schedule { order: vec![0, 1] },
            partition: PartitionedSchedule {
                pre_loop: vec![0],
                periodic: vec![1],
                in_loop_short: vec![],
                post_loop: vec![],
                interior_classes: vec![1],
            },
            plans,
        };
        assert!(matches!(
            pipe.target_for(1, 0, 0),
            Some(DepTarget::PreLoop { .. })
        ));
        assert!(matches!(
            pipe.target_for(1, 0, 1),
            Some(DepTarget::Periodic {
                delta_global: 1,
                ..
            })
        ));
        assert!(matches!(
            pipe.target_for(1, 0, 2),
            Some(DepTarget::Periodic {
                delta_global: 1,
                ..
            })
        ));
        // Past the end of the plan.
        assert!(pipe.target_for(1, 0, 3).is_none());
        // Boundary_pos with no plan.
        assert!(pipe.target_for(1, 1, 0).is_none());
        // Class with no plan (pre-loop class c0 has no ReadPlan).
        assert!(pipe.target_for(0, 0, 0).is_none());
    }

    /// `boundary_plan` returns the stored plan by key and `None`
    /// otherwise.
    #[test]
    fn pipeline_lookup_boundary_plan() {
        let mut plans: HashMap<(usize, usize), ReadPlan> = HashMap::new();
        let expected = ReadPlan {
            consumer_class: 2,
            boundary_pos: 0,
            regions: vec![ReadRegion {
                range: IterRange { lo: 0, hi: 4 },
                target: DepTarget::Periodic {
                    producer_class: 5,
                    producer_tile_pos: 0,
                    producer_slot: 0,
                    delta_global: 0,
                },
            }],
        };
        plans.insert((2, 0), expected);
        let pipe = StencilPipeline {
            edges: vec![],
            domain: ClassDomain {
                periods: vec![4, 4],
                offsets: vec![0, 0],
            },
            schedule: Schedule { order: vec![0, 1] },
            partition: PartitionedSchedule {
                pre_loop: vec![],
                periodic: vec![0, 1],
                in_loop_short: vec![],
                post_loop: vec![],
                interior_classes: vec![0, 1],
            },
            plans,
        };
        let plan = pipe.boundary_plan(2, 0).expect("plan present");
        assert_eq!(plan.consumer_class, 2);
        assert_eq!(plan.regions.len(), 1);
        assert!(pipe.boundary_plan(2, 1).is_none());
        assert!(pipe.boundary_plan(999, 0).is_none());
    }
}
