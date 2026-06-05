// SPDX-License-Identifier: Apache-2.0
//! Bin-pack a tensor-region SSA graph ([`crate::subtile_ir::SubtileIR`]) into
//! `P` co-resident worker tapes for the persistent decode megakernel, and
//! stamp every cross-worker producer→consumer edge as a point-to-point
//! `Wait`/`Signal` flag.
//!
//! This is the GPU-accurate sibling of [`crate::tape`] + [`crate::scheduler`]
//! (which schedule the coarse `Operand::Sub` graph in [`crate::subtile`]).
//! The region graph is the one the megakernel actually runs: every matmul is
//! N-block-tiled, so its blocks can be **spread across the 10 M4 cores** (the
//! bandwidth argument — single-TG is 1/10 BW), and the consumer of the whole
//! output `Wait`s on every block's flag.
//!
//! Edges come from [`crate::subtile_ir::predecessors`] (region overlap on
//! op-output tensors; leaf-source reads have none). Same point-to-point design
//! as `tape.rs`: each surviving cross-worker edge is **one flag** (measured
//! 0.18 µs/hop, flat in core count), NOT a global barrier; the co-resident
//! threadgroups spin on the producer's flag.
//!
//! Host-first correctness: [`play`] replays a schedule honoring the flags,
//! computing each node with [`crate::subtile_ir::eval_node`], and Tier A asserts
//! it equals [`crate::subtile_ir::eval_dag`] bit-for-bit — proving the assignment
//! + flag stamping are deadlock-free and order-correct before any GPU/MSL.

use crate::subtile_ir::{
    SubtileId, SubtileIR, SubtileNode, eval_node, predecessors, scatter,
};

// ── Scheduled form ──────────────────────────────────────────────────

/// One instruction in a worker's tape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TapeInstr {
    /// Compute this node (its inputs are ready: same-worker producers ran
    /// earlier in this tape; cross-worker producers are gated by a `Wait`).
    Compute(SubtileId),
    /// Set one-shot flag `f` — the producer side of a cross-worker edge.
    Signal(u32),
    /// Block until one-shot flag `f` is set — the consumer side.
    Wait(u32),
}

/// One worker's ordered instruction stream (one co-resident threadgroup).
#[derive(Clone, Debug, Default)]
pub struct Worker {
    pub tape: Vec<TapeInstr>,
}

/// A full schedule: `P` worker tapes plus the flag count they reference.
#[derive(Clone, Debug)]
pub struct Schedule {
    pub workers: Vec<Worker>,
    pub num_flags: u32,
}

impl Schedule {
    /// Total `Compute` instructions across all workers — must equal the
    /// node count (every node computed exactly once).
    pub fn total_computes(&self) -> usize {
        self.workers
            .iter()
            .flat_map(|w| &w.tape)
            .filter(|i| matches!(i, TapeInstr::Compute(_)))
            .count()
    }
}

/// Scheduler knobs (mirrors [`crate::scheduler::ScheduleParams`]).
#[derive(Clone, Copy, Debug)]
pub struct ScheduleParams {
    /// Number of co-resident workers (M4 = 10 GPU cores).
    pub num_workers: u32,
    /// Microsecond cost charged per surviving cross-worker edge (the p2p hop
    /// latency, measured ~0.18 µs on M4). Ties the edge-cut objective into
    /// the load-balance objective's units so they're commensurable.
    pub wait_cost_us: f64,
}

/// Quality metrics read back off a [`Schedule`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScheduleMetrics {
    /// P1 — busiest worker's total compute (µs); bounds the makespan.
    pub max_stack_us: f64,
    /// Sum of every worker's compute (µs); makespan floor is
    /// `total_us / num_workers`.
    pub total_us: f64,
    /// P2 — number of `Wait`s = cross-worker edges crossed.
    pub edge_cut: u32,
}

// ── Assignment → deadlock-free tape ─────────────────────────────────

/// Build a [`Schedule`] from a node→worker assignment (`worker_of[id]`) and
/// the precomputed `preds`. Each worker computes its nodes in ascending-id
/// (topological) order; a one-shot flag is allocated per producer that has
/// any cross-worker consumer; consumers `Wait` on it, the producer `Signal`s.
///
/// **Deadlock-free for *any* assignment:** [`crate::subtile_ir::validate`]
/// guarantees every predecessor has a strictly smaller id than its consumer,
/// and every worker emits in ascending-id order, so a producer's `Signal`
/// always precedes — across the whole emission — the point any consumer could
/// block on it. The scheduler therefore only has to choose a good `worker_of`.
pub fn schedule_from_assignment(
    graph: &SubtileIR,
    preds: &[Vec<SubtileId>],
    worker_of: &[u32],
    num_workers: usize,
) -> Schedule {
    let n = graph.nodes.len();
    assert_eq!(worker_of.len(), n, "worker_of must cover every node");
    assert_eq!(preds.len(), n, "preds must cover every node");

    // A producer needs a flag iff some consumer lives on another worker.
    let mut needs_flag = vec![false; n];
    for (cid, p) in preds.iter().enumerate() {
        let cw = worker_of[cid];
        for prod in p {
            if worker_of[prod.0 as usize] != cw {
                needs_flag[prod.0 as usize] = true;
            }
        }
    }
    let mut flag_of = vec![u32::MAX; n];
    let mut num_flags = 0u32;
    for (i, &nf) in needs_flag.iter().enumerate() {
        if nf {
            flag_of[i] = num_flags;
            num_flags += 1;
        }
    }

    let mut workers: Vec<Worker> = (0..num_workers).map(|_| Worker::default()).collect();
    for node in &graph.nodes {
        let id = node.id.0 as usize;
        let w = worker_of[id] as usize;
        // One Wait per distinct cross-worker producer flag for THIS node. (Not
        // deduped across the worker's whole tape: a measured regression — the
        // re-issued `Wait` before each consumer, though redundant for
        // correctness, keeps the megakernel ~7× faster on M5. The device
        // barrier each `Wait` carries appears to pace the worker against the
        // producers; deduping lets a consumer worker rush to a join and busy-
        // spin, contending the bus with the producers' weight loads. Keep the
        // re-waits.)
        let mut waited: Vec<u32> = Vec::new();
        for prod in &preds[id] {
            if worker_of[prod.0 as usize] != worker_of[id] {
                let f = flag_of[prod.0 as usize];
                debug_assert_ne!(f, u32::MAX, "cross-worker producer must have a flag");
                if !waited.contains(&f) {
                    waited.push(f);
                    workers[w].tape.push(TapeInstr::Wait(f));
                }
            }
        }
        workers[w].tape.push(TapeInstr::Compute(node.id));
        if needs_flag[id] {
            workers[w].tape.push(TapeInstr::Signal(flag_of[id]));
        }
    }

    Schedule { workers, num_flags }
}

/// Round-robin partition (node `i` → worker `i % p`) — a test fixture; the
/// real assignment is [`schedule_wavefront`].
pub fn partition_roundrobin(graph: &SubtileIR, p: u32) -> Schedule {
    let p = p.max(1) as usize;
    let preds = predecessors(graph);
    let worker_of: Vec<u32> = (0..graph.nodes.len()).map(|i| (i % p) as u32).collect();
    schedule_from_assignment(graph, &preds, &worker_of, p)
}

// ── The wavefront scheduler ─────────────────────────────────────────

/// Greedy cost-aware assignment of region nodes to `P` workers, then a
/// deadlock-free tape. Each node goes to the worker minimizing
/// `new_load + wait_cost · (predecessors not already there)` — balancing
/// per-worker compute (P1) against cross-worker edges (P2). Cost is injected
/// (`Fn(&SubtileNode) -> f64` in µs) so this crate stays decoupled from the
/// target's cost tables; the metal compiler supplies the real `cost_us`.
pub fn schedule_wavefront(
    graph: &SubtileIR,
    cost: impl Fn(&SubtileNode) -> f64,
    params: ScheduleParams,
) -> Schedule {
    let preds = predecessors(graph);
    let p = params.num_workers.max(1) as usize;
    let n = graph.nodes.len();
    let mut load = vec![0f64; p];
    let mut worker_of = vec![0u32; n];

    // Topological (ascending-id) order: a node's predecessors are placed
    // before it, so the affinity term sees their final workers.
    for node in &graph.nodes {
        let id = node.id.0 as usize;
        let c = cost(node);
        let mut best_w = 0usize;
        let mut best_score = f64::INFINITY;
        for (w, &load_w) in load.iter().enumerate() {
            // P2 term: predecessors not on w become waits into this node.
            let cut = preds[id]
                .iter()
                .filter(|prod| worker_of[prod.0 as usize] as usize != w)
                .count() as u32;
            let score = load_w + c + params.wait_cost_us * cut as f64;
            if score < best_score {
                best_score = score;
                best_w = w;
            }
        }
        worker_of[id] = best_w as u32;
        load[best_w] += c;
    }

    schedule_from_assignment(graph, &preds, &worker_of, p)
}

/// **Slice-index owner assignment** — the partition's placement rule. Worker
/// `w` owns column-slice `w` of EVERY op: `owner = (output column start / unit)
/// mod P`. Because the decode chains preserve columns — matmul N-block →
/// rope → attn (one q-head group), and gate/up → silu·mul (one intermediate
/// block) — every op in a chain lands on the SAME worker, so the chain is
/// local (no cross-worker wait). This replaces the greedy scatter of
/// [`schedule_wavefront`], which balances load by splitting chains across
/// workers (the source of the cross-worker dependency-wait the megakernel
/// pays). `unit` is the tiling granularity the graph was lowered at (the
/// `head_dim`, so head-structured ops and column-tiled ops align).
///
/// Cross-worker edges survive only at the genuine joins — the o_proj/down
/// reductions and the rmsnorm broadcast — which the split-K + replication
/// transforms remove ([`crate::partition`]).
pub fn assign_owners_slice_index(graph: &SubtileIR, unit: u32, num_workers: u32) -> Vec<u32> {
    let p = num_workers.max(1);
    let unit = unit.max(1);
    graph
        .nodes
        .iter()
        .map(|n| (n.output.region.cols.start / unit) % p)
        .collect()
}

/// Build a [`Schedule`] from the slice-index owner assignment.
pub fn schedule_slice_index(graph: &SubtileIR, unit: u32, num_workers: u32) -> Schedule {
    let preds = predecessors(graph);
    let worker_of = assign_owners_slice_index(graph, unit, num_workers);
    schedule_from_assignment(graph, &preds, &worker_of, num_workers.max(1) as usize)
}

/// Read P1 / P2 / total off a schedule under the given cost model.
pub fn measure(
    graph: &SubtileIR,
    schedule: &Schedule,
    cost: impl Fn(&SubtileNode) -> f64,
) -> ScheduleMetrics {
    let mut stack = vec![0f64; schedule.workers.len()];
    let mut edge_cut = 0u32;
    for (wi, w) in schedule.workers.iter().enumerate() {
        for instr in &w.tape {
            match instr {
                TapeInstr::Compute(id) => stack[wi] += cost(&graph.nodes[id.0 as usize]),
                TapeInstr::Wait(_) => edge_cut += 1,
                TapeInstr::Signal(_) => {}
            }
        }
    }
    ScheduleMetrics {
        max_stack_us: stack.iter().cloned().fold(0.0, f64::max),
        total_us: stack.iter().sum(),
        edge_cut,
    }
}

// ── Host replay (Tier A) ────────────────────────────────────────────

/// Replay a schedule on the host, honoring `Wait`/`Signal`, and return the
/// backing buffer of every tensor (indexed by `TensorId`) — identical to
/// [`crate::subtile_ir::eval_dag`] when the schedule's sync is correct. Panics on
/// deadlock (a missing `Signal` or cyclic `Wait`s) and on any node computed
/// before a producer it reads (a missing `Wait` edge) — the bugs host-first
/// is meant to catch.
pub fn play(graph: &SubtileIR, schedule: &Schedule, sources: &[&[f32]]) -> Vec<Vec<f32>> {
    assert_eq!(
        sources.len(),
        graph.num_sources as usize,
        "source count mismatch"
    );
    let n = graph.nodes.len();
    let mut bufs: Vec<Vec<f32>> = graph
        .tensors
        .iter()
        .map(|t| vec![0f32; (t.rows * t.cols) as usize])
        .collect();
    for (s, src) in sources.iter().enumerate() {
        assert_eq!(src.len(), bufs[s].len(), "source {s} buffer size mismatch");
        bufs[s].copy_from_slice(src);
    }

    let preds = predecessors(graph);
    let mut computed = vec![false; n];
    let mut flags = vec![false; schedule.num_flags as usize];
    let mut pc = vec![0usize; schedule.workers.len()];

    loop {
        let mut progressed = false;
        let mut all_done = true;
        for (wi, worker) in schedule.workers.iter().enumerate() {
            if pc[wi] >= worker.tape.len() {
                continue;
            }
            all_done = false;
            match worker.tape[pc[wi]] {
                TapeInstr::Wait(f) => {
                    if flags[f as usize] {
                        pc[wi] += 1;
                        progressed = true;
                    }
                    // else: this worker stays blocked this round.
                }
                TapeInstr::Signal(f) => {
                    flags[f as usize] = true;
                    pc[wi] += 1;
                    progressed = true;
                }
                TapeInstr::Compute(id) => {
                    let node = &graph.nodes[id.0 as usize];
                    for prod in &preds[id.0 as usize] {
                        assert!(
                            computed[prod.0 as usize],
                            "node {} computed before producer {} (missing Wait edge)",
                            id.0, prod.0
                        );
                    }
                    let out = eval_node(node, graph, &bufs);
                    let shape = graph.shape(node.output.tensor);
                    scatter(
                        &mut bufs,
                        node.output.tensor,
                        node.output.region,
                        &out,
                        shape,
                    );
                    computed[id.0 as usize] = true;
                    pc[wi] += 1;
                    progressed = true;
                }
            }
        }
        if all_done {
            break;
        }
        assert!(
            progressed,
            "deadlock: no worker advanced (missing Signal or cyclic Waits)"
        );
    }

    debug_assert!(computed.iter().all(|&c| c), "some node never computed");
    bufs
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::{InputRef, LoweredOp, LoweringInput, OpDesc};
    use crate::subtile_ir::{SourceShape, lower_region, result_buffer};

    fn rng_fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (s >> 33) as u32;
                (bits as f32 / 2147483648.0) * 2.0 - 1.0
            })
            .collect()
    }

    /// gemm1(N1×K) → silu → gemm2(N2×N1): a chain whose matmuls N-block-tile
    /// and whose elementwise/second-matmul consumers join on every block — so
    /// scheduling it across `P` workers exercises the cross-worker producer→
    /// consumer flags (block-spread + whole-output join), not just balance.
    fn chain_input(k: u32, n1: u32, n2: u32) -> (LoweringInput, Vec<Vec<f32>>) {
        let act = rng_fill(k as usize, 1);
        let w1 = rng_fill((n1 * k) as usize, 2);
        let w2 = rng_fill((n2 * n1) as usize, 3);
        let input = LoweringInput {
            sources: vec![
                SourceShape { rows: 1, cols: k },   // 0: act
                SourceShape { rows: n1, cols: k },  // 1: W1 [n1,k]
                SourceShape { rows: n2, cols: n1 }, // 2: W2 [n2,n1]
            ],
            ops: vec![
                OpDesc {
                    op: LoweredOp::Gemm { n: n1 },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: LoweredOp::Silu,
                    m: 1,
                    inputs: vec![InputRef::Op(0)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: n2 },
                    m: 1,
                    inputs: vec![InputRef::Op(1), InputRef::Ext(2)],
                },
            ],
            result: 2,
        };
        (input, vec![act, w1, w2])
    }

    fn cost_area(node: &SubtileNode) -> f64 {
        (node.output.region.rows.len * node.output.region.cols.len) as f64
    }

    /// **Tier A on the region graph:** a wavefront-scheduled tape of `P=10`
    /// (and 1/2/4) workers replays bit-for-bit identical to direct topo eval,
    /// across N-block widths — so bin-packing + cross-worker flag stamping are
    /// proven deadlock-free and order-correct.
    #[test]
    fn region_schedule_replays_bit_exact() {
        let (input, data) = chain_input(24, 16, 20);
        let srcs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        for nb in [4u32, 8, 1000] {
            let g = lower_region(&input, std::num::NonZeroU32::new(nb).unwrap());
            let want = result_buffer(&g, &crate::subtile_ir::eval_dag(&g, &srcs)).to_vec();
            for p in [1u32, 2, 4, 10] {
                let s = schedule_wavefront(
                    &g,
                    cost_area,
                    ScheduleParams {
                        num_workers: p,
                        wait_cost_us: 0.18,
                    },
                );
                assert_eq!(s.workers.len(), p as usize, "P workers (incl. empties)");
                assert_eq!(
                    s.total_computes(),
                    g.nodes.len(),
                    "every node scheduled once (nb={nb}, p={p})"
                );
                let got = play(&g, &s, &srcs);
                assert_eq!(
                    result_buffer(&g, &got),
                    &want[..],
                    "wavefront replay nb={nb} p={p}"
                );
                // Round-robin must also replay (correctness ⟂ assignment).
                let rr = partition_roundrobin(&g, p);
                assert_eq!(
                    result_buffer(&g, &play(&g, &rr, &srcs)),
                    &want[..],
                    "round-robin replay nb={nb} p={p}"
                );
            }
        }
    }

    /// **Slice-index scheduling co-locates chains.** gemm1(N-block) → silu
    /// (tiled) → gemm2: with `unit = nb`, silu tile `i` shares columns with
    /// gemm1 block `i`, so the slice-index puts them on the SAME worker (the
    /// chain is local — no cross-worker wait), and the tape still replays
    /// bit-exact (`play` is correct for any valid assignment).
    #[test]
    fn slice_index_co_locates_chains() {
        let (input, data) = chain_input(24, 16, 20);
        let srcs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let nb = 4u32;
        let g = lower_region(&input, std::num::NonZeroU32::new(nb).unwrap());
        let want = result_buffer(&g, &crate::subtile_ir::eval_dag(&g, &srcs)).to_vec();
        // gemm1 → 4 blocks (ids 0..4, cols 0,4,8,12); silu → 4 tiles (ids 4..8,
        // same cols); gemm2 → 5 blocks reading whole silu.
        assert_eq!(g.nodes.len(), 13);
        for p in [2u32, 4, 10] {
            let owner = assign_owners_slice_index(&g, nb, p);
            for i in 0..4 {
                assert_eq!(
                    owner[i],
                    owner[4 + i],
                    "silu tile {i} co-located with its matmul block (p={p})"
                );
            }
            let si = schedule_slice_index(&g, nb, p);
            assert_eq!(si.total_computes(), g.nodes.len());
            assert_eq!(
                result_buffer(&g, &play(&g, &si, &srcs)),
                &want[..],
                "slice-index replay p={p}"
            );
        }
    }

    /// Flag discipline: each one-shot flag is signaled exactly once and
    /// waited at least once; every node computed exactly once; `p=1` has no
    /// cross-worker edges so no flags.
    #[test]
    fn flag_invariants() {
        let (input, _) = chain_input(24, 16, 20);
        let g = lower_region(&input, std::num::NonZeroU32::new(4).unwrap()); // n-block so producers split

        let s1 = partition_roundrobin(&g, 1);
        assert_eq!(s1.num_flags, 0, "p=1 → no cross-worker edges");
        assert!(
            s1.workers
                .iter()
                .flat_map(|w| &w.tape)
                .all(|i| matches!(i, TapeInstr::Compute(_))),
            "p=1 tape is pure computes"
        );

        let s = schedule_wavefront(
            &g,
            cost_area,
            ScheduleParams {
                num_workers: 10,
                wait_cost_us: 0.18,
            },
        );
        assert_eq!(s.total_computes(), g.nodes.len());
        let mut signals = vec![0u32; s.num_flags as usize];
        let mut waits = vec![0u32; s.num_flags as usize];
        for w in &s.workers {
            for i in &w.tape {
                match i {
                    TapeInstr::Signal(f) => signals[*f as usize] += 1,
                    TapeInstr::Wait(f) => waits[*f as usize] += 1,
                    TapeInstr::Compute(_) => {}
                }
            }
        }
        for f in 0..s.num_flags as usize {
            assert_eq!(signals[f], 1, "flag {f} signaled exactly once");
            assert!(waits[f] >= 1, "flag {f} waited at least once");
        }
    }

    /// On a wide N-block matmul (one op, all blocks independent), the
    /// scheduler spreads the blocks across the workers — the bandwidth point.
    /// With no edges (one gemm, blocks read only sources) it's pure balance:
    /// the busiest worker's stack is far below the single-worker total.
    #[test]
    fn wide_matmul_spreads_across_workers() {
        // act[1,8] @ W[120,8] with nb=4 → 30 independent blocks, no edges.
        let act = rng_fill(8, 10);
        let w = rng_fill(120 * 8, 11);
        let input = LoweringInput {
            sources: vec![
                SourceShape { rows: 1, cols: 8 },
                SourceShape { rows: 120, cols: 8 },
            ],
            ops: vec![OpDesc {
                op: LoweredOp::Gemm { n: 120 },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
            }],
            result: 0,
        };
        let g = lower_region(&input, std::num::NonZeroU32::new(4).unwrap());
        assert_eq!(g.nodes.len(), 30, "ceil(120/4) blocks");
        let s = schedule_wavefront(
            &g,
            cost_area,
            ScheduleParams {
                num_workers: 10,
                wait_cost_us: 0.18,
            },
        );
        let m = measure(&g, &s, cost_area);
        assert_eq!(m.edge_cut, 0, "independent blocks → no waits");
        // 30 equal blocks over 10 workers → 3 each → max_stack == total/10.
        assert!(
            (m.max_stack_us - m.total_us / 10.0).abs() < 1e-9,
            "blocks spread evenly: max_stack={} total/10={}",
            m.max_stack_us,
            m.total_us / 10.0
        );
        let srcs: Vec<&[f32]> = vec![&act, &w];
        assert_eq!(
            result_buffer(&g, &play(&g, &s, &srcs)),
            result_buffer(&g, &crate::subtile_ir::eval_dag(&g, &srcs)),
            "spread matmul replays bit-exact"
        );
    }
}
