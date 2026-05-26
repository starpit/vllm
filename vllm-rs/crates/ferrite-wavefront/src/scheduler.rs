// SPDX-License-Identifier: Apache-2.0
//! The wavefront scheduler — assign subtile nodes to P co-resident
//! workers, then stamp cross-worker edges as point-to-point `Wait`/
//! `Signal` (via [`tape::schedule_from_assignment`], which is deadlock-
//! free for any assignment). The scheduler therefore only has to choose a
//! good node→worker assignment; the design's two objectives, both in
//! microseconds so they're commensurable:
//!
//!   - **P1 = max stack height** — the busiest worker's total compute,
//!     which bounds the makespan. Minimized by balancing per-worker load.
//!   - **P2 = cross-tape edge cut** — each surviving cross-worker edge is
//!     one p2p wait (~0.18 µs measured, flat in core count) plus any
//!     stall-if-early. Minimized by keeping a consumer with its producers.
//!
//! Cost is an **injected** `Fn(&SubtileNode) -> f64` (µs): the macro crate
//! supplies real `TargetProfile::cost_us`; tests supply synthetic costs.
//! This keeps `ferrite-wavefront` decoupled from the cost tables.
//!
//! This first cut is a single-pass greedy list scheduler over the DAG's
//! topological (ascending-id) order: each node goes to the worker
//! minimizing `new_load + wait_cost · (predecessors not already there)`.
//! Modulo scheduling across layer iterations (II = ResMII, so weight loads
//! of the next layer overlap this layer's serial drain) builds on this.

use crate::subtile::{Operand, SubtileGraph, SubtileNode};
use crate::tape::{Schedule, TapeInstr, schedule_from_assignment};

/// Scheduler knobs.
#[derive(Clone, Copy, Debug)]
pub struct ScheduleParams {
    pub num_workers: u32,
    /// Microsecond cost charged per surviving cross-worker edge (the p2p
    /// hop latency — measured ~0.18 µs on M4). Ties P2 into P1's units.
    pub wait_cost_us: f64,
}

/// Quality metrics read back off a [`Schedule`] (see module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScheduleMetrics {
    /// P1 — busiest worker's total compute (µs).
    pub max_stack_us: f64,
    /// Sum of all workers' compute (µs) — makespan lower bound is
    /// `total_us / num_workers`.
    pub total_us: f64,
    /// P2 — number of `Wait` instructions = cross-worker edges crossed.
    pub edge_cut: u32,
}

/// Greedy cost-aware assignment, then a deadlock-free tape.
pub fn schedule_wavefront(
    graph: &SubtileGraph,
    cost: impl Fn(&SubtileNode) -> f64,
    params: ScheduleParams,
) -> Schedule {
    let p = params.num_workers.max(1) as usize;
    let n = graph.nodes.len();
    let mut load = vec![0f64; p];
    let mut worker_of = vec![0u32; n];

    // Topological (ascending-id) order: a node's predecessors are placed
    // before it, so the affinity term sees their final workers.
    for node in &graph.nodes {
        let c = cost(node);
        let mut best_w = 0usize;
        let mut best_score = f64::INFINITY;
        for (w, &load_w) in load.iter().enumerate() {
            // P2 term: predecessors not on w become waits into this node.
            let mut cut = 0u32;
            for inp in &node.inputs {
                if let Operand::Sub(prod) = inp
                    && worker_of[prod.0 as usize] as usize != w
                {
                    cut += 1;
                }
            }
            // Balance (new load on w) + communication into this node.
            let score = load_w + c + params.wait_cost_us * cut as f64;
            if score < best_score {
                best_score = score;
                best_w = w;
            }
        }
        worker_of[node.id.0 as usize] = best_w as u32;
        load[best_w] += c;
    }

    schedule_from_assignment(graph, &worker_of, p)
}

/// Read P1 / P2 / total off a schedule under the given cost model.
pub fn measure(
    graph: &SubtileGraph,
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

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subtile::{
        EwKind, OutputSlot, Range, Region, SourceId, SourceShape, SubOp, SubtileGraph, SubtileId,
        SubtileNode, TilingPolicy, eval_dag, lower_gate_up_silu_mul_standalone,
        lower_gemm_standalone,
    };
    use crate::tape::{partition_roundrobin, play};

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

    /// The scheduled tape must still replay bit-exact (correctness is
    /// independent of the assignment — only the schedule's quality changes).
    #[test]
    fn wavefront_schedule_replays_bit_exact() {
        // A graph with real inter-node edges (split-K reduces + the
        // gate/up chain).
        let (m, n, k) = (2u32, 64, 192);
        let a = rng_fill((m * k) as usize, 1);
        let w = rng_fill((n * k) as usize, 2);
        let g = lower_gemm_standalone(m, n, k, TilingPolicy { nb: 8, k_chunks: 4 });
        let want = eval_dag(&g, &[&a, &w]);
        let cost = |node: &SubtileNode| (node.out_rows * node.out_cols) as f64;
        for p in [1u32, 2, 3, 4, 8] {
            let s = schedule_wavefront(
                &g,
                cost,
                ScheduleParams {
                    num_workers: p,
                    wait_cost_us: 0.18,
                },
            );
            assert_eq!(
                s.total_computes(),
                g.nodes.len(),
                "every node scheduled once"
            );
            assert_eq!(play(&g, &s, &[&a, &w]), want, "wavefront replay p={p}");
        }

        // And on a graph with chained Sub edges.
        let (gm, gd) = (2u32, 96u32);
        let gate = rng_fill((gm * gd) as usize, 3);
        let up = rng_fill((gm * gd) as usize, 4);
        let gg = lower_gate_up_silu_mul_standalone(gm, gd, 8);
        let gwant = eval_dag(&gg, &[&gate, &up]);
        let s = schedule_wavefront(
            &gg,
            cost,
            ScheduleParams {
                num_workers: 4,
                wait_cost_us: 0.18,
            },
        );
        assert_eq!(
            play(&gg, &s, &[&gate, &up]),
            gwant,
            "gate/up wavefront replay"
        );
    }

    /// On a deliberately imbalanced workload, the cost-aware scheduler
    /// achieves a strictly lower max-stack (P1) than round-robin.
    #[test]
    fn wavefront_beats_roundrobin_on_imbalance() {
        // 8 independent Silu nodes over disjoint slices of one source, so
        // partitioning is pure load balancing (no edges → no waits). Costs
        // alternate heavy/light: round-robin (i % 2) piles all the heavy
        // ones on worker 0.
        let width = 4u32;
        let n = 8usize;
        let total = n as u32 * width;
        let mut nodes = Vec::new();
        let mut outputs = Vec::new();
        for i in 0..n {
            let region = Region {
                rows: Range::new(0, 1),
                cols: Range::new(i as u32 * width, width),
            };
            nodes.push(SubtileNode {
                id: SubtileId(i as u32),
                op: SubOp::Elementwise(EwKind::Silu),
                inputs: vec![Operand::Source {
                    id: SourceId(0),
                    region,
                }],
                out_rows: 1,
                out_cols: width,
            });
            outputs.push(OutputSlot {
                node: SubtileId(i as u32),
                dest: region,
            });
        }
        let g = SubtileGraph {
            nodes,
            sources: vec![SourceShape {
                rows: 1,
                cols: total,
            }],
            result_rows: 1,
            result_cols: total,
            outputs,
        };

        // Even ids heavy (10), odd ids light (1).
        let cost = |node: &SubtileNode| {
            if node.id.0.is_multiple_of(2) {
                10.0
            } else {
                1.0
            }
        };
        let params = ScheduleParams {
            num_workers: 2,
            wait_cost_us: 0.0,
        };

        let rr = partition_roundrobin(&g, 2);
        let wf = schedule_wavefront(&g, cost, params);
        let m_rr = measure(&g, &rr, cost);
        let m_wf = measure(&g, &wf, cost);

        // Round-robin: worker 0 gets all four heavy nodes (ids 0,2,4,6) =
        // 40; the balanced optimum is 22 per worker.
        assert_eq!(m_rr.max_stack_us, 40.0, "round-robin piles heavies");
        assert!(
            m_wf.max_stack_us < m_rr.max_stack_us,
            "wavefront should balance better: wf={} rr={}",
            m_wf.max_stack_us,
            m_rr.max_stack_us
        );
        assert_eq!(m_wf.total_us, m_rr.total_us, "same total work either way");
        // No edges in this graph, so no p2p waits regardless.
        assert_eq!(m_wf.edge_cut, 0);

        // Correctness still holds.
        let x = rng_fill(total as usize, 9);
        let want = eval_dag(&g, &[&x]);
        assert_eq!(play(&g, &wf, &[&x]), want, "imbalanced graph replays");
    }

    /// Trading P2 against P1: a positive `wait_cost` pulls a chain of
    /// dependent nodes onto one worker (zero edge cut), where round-robin
    /// would scatter it.
    #[test]
    fn wait_cost_pulls_chains_together() {
        // A linear chain a→b→c→d (each reads the previous), via Add nodes
        // over the same source region so they're shape-compatible.
        let width = 4u32;
        let src = SourceShape {
            rows: 1,
            cols: width,
        };
        let region = Region {
            rows: Range::new(0, 1),
            cols: Range::new(0, width),
        };
        let mut nodes = vec![SubtileNode {
            id: SubtileId(0),
            op: SubOp::Elementwise(EwKind::Silu),
            inputs: vec![Operand::Source {
                id: SourceId(0),
                region,
            }],
            out_rows: 1,
            out_cols: width,
        }];
        for i in 1..5u32 {
            nodes.push(SubtileNode {
                id: SubtileId(i),
                op: SubOp::Elementwise(EwKind::Add),
                inputs: vec![
                    Operand::Sub(SubtileId(i - 1)),
                    Operand::Source {
                        id: SourceId(0),
                        region,
                    },
                ],
                out_rows: 1,
                out_cols: width,
            });
        }
        let g = SubtileGraph {
            nodes,
            sources: vec![src],
            result_rows: 1,
            result_cols: width,
            outputs: vec![OutputSlot {
                node: SubtileId(4),
                dest: region,
            }],
        };
        let cost = |_: &SubtileNode| 1.0;

        // High wait cost → keep the chain together → zero edge cut.
        let s = schedule_wavefront(
            &g,
            cost,
            ScheduleParams {
                num_workers: 4,
                wait_cost_us: 100.0,
            },
        );
        assert_eq!(
            measure(&g, &s, cost).edge_cut,
            0,
            "chain kept on one worker"
        );

        // Correctness.
        let x = rng_fill(width as usize, 7);
        assert_eq!(play(&g, &s, &[&x]), eval_dag(&g, &[&x]), "chain replays");
    }
}
