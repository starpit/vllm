// SPDX-License-Identifier: Apache-2.0
//! Static greedy list-scheduler for the reified DAG.
//!
//! Phase 2 of the static-scheduler pivot. Takes a `ReifiedDag` plus a CTA
//! pool size and a per-node cost model, produces a `Schedule` that assigns
//! every node to a (cta_id, start_cycle, end_cycle) slot respecting all
//! data dependencies.
//!
//! Algorithm: standard list-scheduling with critical-path-from-sink priority.
//! 1. Compute `cp_remaining[n]` = longest path of cumulative cost from n to
//!    any sink. This is the priority — higher = scheduled earlier.
//! 2. Maintain a ready set (nodes with all deps satisfied) ordered by
//!    `cp_remaining` desc.
//! 3. Maintain `cta_free[c]` = next free cycle for CTA c.
//! 4. Each step: pop highest-priority ready node N, pick CTA with smallest
//!    `cta_free`, schedule N at `max(cta_free[c], max_dep_finish)`.
//! 5. Update successors' remaining-dep counts; promote any that hit 0.
//!
//! The cost model is intentionally crude — Phase 2 cares about *structure*
//! and *predicted shape*, not absolute calibration. Phase 3+ refines the
//! model and validates against measured timings.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::reified_dag::{NodeId, Phase, ReifiedDag};

/// Per-node cost in abstract "tensor-core mma units." One unit is roughly
/// the time of one m16n8k16 bf16 MMA (~16 sm89 cycles). Memory phases get
/// translated into the same unit so the scheduler sees a single timeline.
#[derive(Clone, Debug)]
pub struct CostModel {
    /// Tile shape used by the reifier (needed to size GEMM costs).
    pub row_tile: u32,
    pub qkv_col_tile: u32,
    pub o_col_tile: u32,
    pub gate_up_col_tile: u32,
    pub down_col_tile: u32,
    pub hidden_dim: u32,
    pub intermediate_dim: u32,
    pub seq_len: u32,
    pub head_dim: u32,
    pub num_attn_heads: u32,
    pub num_kv_heads: u32,
}

impl CostModel {
    pub fn from_dag(dag: &ReifiedDag) -> Self {
        Self {
            row_tile: dag.tiles.row_tile,
            qkv_col_tile: dag.tiles.qkv_col_tile,
            o_col_tile: dag.tiles.o_col_tile,
            gate_up_col_tile: dag.tiles.gate_up_col_tile,
            down_col_tile: dag.tiles.down_col_tile,
            hidden_dim: dag.dims.hidden_dim,
            intermediate_dim: dag.dims.intermediate_dim,
            seq_len: dag.dims.seq_len,
            head_dim: dag.dims.head_dim,
            num_attn_heads: dag.dims.num_attn_heads,
            num_kv_heads: dag.dims.num_kv_heads,
        }
    }

    /// Compute cost in "mma units" for a single GEMM tile of shape (M, N, K).
    /// One m16n8k16 bf16 mma covers 16*8*16 = 2048 fma → use that as the unit.
    fn gemm_compute(m: u32, n: u32, k: u32) -> u32 {
        let mma_m = m.div_ceil(16);
        let mma_n = n.div_ceil(8);
        let mma_k = k.div_ceil(16);
        mma_m * mma_n * mma_k
    }

    /// Memory cost in mma units, given bytes loaded by a single CTA running
    /// this tile. Assumes global memory bandwidth is fairly shared across
    /// `num_sms`-many concurrent CTAs.
    ///
    /// L4 numbers: 300 GB/s global BW, 58 SMs, ~1.5 GHz boost. Per-SM share
    /// = 300/58 ≈ 5.17 GB/s = ~3.45 B/cycle. One mma unit ≈ 16 cycles, so
    /// ~55 B/mma-unit. We bake the constant in for sm89/L4; future targets
    /// can plug a different number.
    const BYTES_PER_MMA_UNIT_L4: u32 = 55;

    fn mem_cost(bytes: u32) -> u32 {
        bytes / Self::BYTES_PER_MMA_UNIT_L4 + 1
    }

    /// GEMM tile cost = max(compute, A_bytes + B_bytes + C_bytes loaded).
    /// Assumes cp.async pipelining overlaps memory with compute, so the
    /// steady-state bound is whichever is larger.
    fn gemm_total(m: u32, n: u32, k: u32) -> u32 {
        let compute = Self::gemm_compute(m, n, k);
        // bf16 = 2 bytes per element. A=[M,K], B=[K,N], output=[M,N].
        // Output write counted at 1x (no read for plain GEMM; residual add
        // would add another M*N read but we ignore that ε contribution).
        let bytes = 2 * (m * k + k * n + m * n);
        let memory = Self::mem_cost(bytes);
        compute.max(memory)
    }

    pub fn cost(&self, phase: Phase) -> u32 {
        let m = self.row_tile;
        match phase {
            // RMSNorm: load M*HD activations + HD weights, write M*HD. Pure BW.
            Phase::AttnNorm | Phase::MlpNorm => {
                let bytes = 2 * (m * self.hidden_dim * 2 + self.hidden_dim);
                Self::mem_cost(bytes) + 8
            }
            Phase::Qkv => Self::gemm_total(m, self.qkv_col_tile, self.hidden_dim),
            Phase::Rope => {
                // Load Q+K+V tile (~M*qkv_dim), write same. Memory bound.
                let qkv = (self.num_attn_heads + 2 * self.num_kv_heads) * self.head_dim;
                Self::mem_cost(2 * 2 * m * qkv) + 4
            }
            Phase::Attention => {
                // Loads full K/V across seq_len (M*K_bytes + M*V_bytes per
                // attn group) and computes scaled-dot-product. Use the larger
                // of compute and BW.
                let kv_bytes_per_row = 2 * 2 * self.seq_len * self.num_kv_heads * self.head_dim;
                let mem = Self::mem_cost(kv_bytes_per_row + 2 * m * self.hidden_dim);
                let compute = (2 * m * self.seq_len * self.head_dim * self.num_attn_heads
                    / self.num_kv_heads)
                    / 2048;
                mem.max(compute.max(64))
            }
            Phase::OProj => Self::gemm_total(m, self.o_col_tile, self.hidden_dim),
            Phase::GateUp => Self::gemm_total(m, self.gate_up_col_tile, self.hidden_dim),
            Phase::Down => Self::gemm_total(m, self.down_col_tile, self.intermediate_dim),
        }
    }
}

/// Where one node ended up in the schedule.
#[derive(Clone, Copy, Debug)]
pub struct ScheduledNode {
    pub node: NodeId,
    pub cta: u32,
    pub start: u64,
    pub end: u64,
}

#[derive(Debug)]
pub struct Schedule {
    pub num_ctas: u32,
    /// Per-node placement, indexed by NodeId.
    pub placements: Vec<ScheduledNode>,
    /// Total wall-clock cycles (max end across all nodes).
    pub makespan: u64,
    /// Lower bound from the longest weighted path (no resource constraint).
    pub critical_path_cost: u64,
}

impl Schedule {
    pub fn ms_at(&self, gpu_clock_hz: f64) -> f64 {
        self.makespan as f64 / gpu_clock_hz * 1000.0
    }

    pub fn cp_ms_at(&self, gpu_clock_hz: f64) -> f64 {
        self.critical_path_cost as f64 / gpu_clock_hz * 1000.0
    }

    /// Per-CTA utilization: busy_cycles / makespan.
    pub fn utilization(&self) -> Vec<f64> {
        let mut busy = vec![0u64; self.num_ctas as usize];
        for p in &self.placements {
            busy[p.cta as usize] += p.end - p.start;
        }
        busy.into_iter()
            .map(|b| b as f64 / self.makespan as f64)
            .collect()
    }
}

/// Heap entry for the ready queue. Higher priority comes out first.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReadyEntry {
    priority: u64,
    node: u32,
}

impl Ord for ReadyEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.node.cmp(&self.node))
    }
}

impl PartialOrd for ReadyEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Build the schedule.
pub fn schedule(dag: &ReifiedDag, num_ctas: u32, cost: &CostModel) -> Schedule {
    let n = dag.nodes.len();
    let costs: Vec<u32> = dag.nodes.iter().map(|nd| cost.cost(nd.phase)).collect();

    // ── Build forward successor list ──
    let mut successors: Vec<Vec<u32>> = vec![Vec::new(); n];
    for nd in &dag.nodes {
        for d in &nd.deps {
            successors[d.0 as usize].push(nd.id.0);
        }
    }

    // ── Critical path remaining cost (longest weighted path from node to sink) ──
    // Process in reverse topological order. Since nodes are emitted in topo
    // order, iterating high-index → low-index works.
    let mut cp_remaining: Vec<u64> = vec![0; n];
    for i in (0..n).rev() {
        let mut best: u64 = 0;
        for &s in &successors[i] {
            let v = cp_remaining[s as usize];
            if v > best {
                best = v;
            }
        }
        cp_remaining[i] = best + costs[i] as u64;
    }
    let critical_path_cost = *cp_remaining.iter().max().unwrap_or(&0);

    // ── List schedule ──
    let mut remaining_deps: Vec<u32> = dag.nodes.iter().map(|nd| nd.deps.len() as u32).collect();
    let mut node_finish: Vec<u64> = vec![0; n];
    let mut placements: Vec<ScheduledNode> = vec![
        ScheduledNode {
            node: NodeId(0),
            cta: 0,
            start: 0,
            end: 0,
        };
        n
    ];
    let mut cta_free: Vec<u64> = vec![0; num_ctas as usize];

    let mut ready: BinaryHeap<ReadyEntry> = BinaryHeap::new();
    for i in 0..n {
        if remaining_deps[i] == 0 {
            ready.push(ReadyEntry {
                priority: cp_remaining[i],
                node: i as u32,
            });
        }
    }

    while let Some(ReadyEntry { node, .. }) = ready.pop() {
        let nd_idx = node as usize;
        // Pick CTA with smallest free-time. Linear scan is fine for L4-class
        // SM counts (~58) and Phase 2 doesn't need to be fast.
        let mut best_cta = 0u32;
        let mut best_free = cta_free[0];
        for c in 1..num_ctas {
            let f = cta_free[c as usize];
            if f < best_free {
                best_free = f;
                best_cta = c;
            }
        }

        // Earliest start = max(cta_free, max_dep_finish).
        let mut start = best_free;
        for d in &dag.nodes[nd_idx].deps {
            let f = node_finish[d.0 as usize];
            if f > start {
                start = f;
            }
        }
        let end = start + costs[nd_idx] as u64;

        placements[nd_idx] = ScheduledNode {
            node: NodeId(node),
            cta: best_cta,
            start,
            end,
        };
        node_finish[nd_idx] = end;
        cta_free[best_cta as usize] = end;

        for &s in &successors[nd_idx] {
            let s_idx = s as usize;
            remaining_deps[s_idx] -= 1;
            if remaining_deps[s_idx] == 0 {
                ready.push(ReadyEntry {
                    priority: cp_remaining[s_idx],
                    node: s,
                });
            }
        }
    }

    let makespan = *cta_free.iter().max().unwrap_or(&0);
    Schedule {
        num_ctas,
        placements,
        makespan,
        critical_path_cost,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reified_dag::{LlamaDims, TileSizes};

    fn llama_1b_dims(seq: u32) -> LlamaDims {
        LlamaDims {
            num_layers: 16,
            hidden_dim: 2048,
            intermediate_dim: 8192,
            num_attn_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            seq_len: seq,
        }
    }

    #[test]
    fn schedule_1b_seq1024_smoke() {
        let dims = llama_1b_dims(1024);
        let dag = ReifiedDag::reify_llama(dims, TileSizes::default_v1());
        let cost = CostModel::from_dag(&dag);
        let sched = schedule(&dag, 58, &cost);

        // Every node placed exactly once.
        assert_eq!(sched.placements.len(), dag.nodes.len());
        for (i, p) in sched.placements.iter().enumerate() {
            assert_eq!(p.node.0 as usize, i);
            assert!(p.end >= p.start);
            assert!(p.cta < sched.num_ctas);
        }

        // Makespan is at least the critical path (cannot finish faster).
        assert!(sched.makespan >= sched.critical_path_cost);
        // Sanity: the work-conservation lower bound also holds.
        let total_work: u64 = sched.placements.iter().map(|p| p.end - p.start).sum();
        let work_lower_bound = total_work / sched.num_ctas as u64;
        assert!(sched.makespan >= work_lower_bound);

        // Print the prediction for human inspection.
        let l4_clock_hz = 1.5e9;
        eprintln!(
            "schedule: nodes={}, makespan={} cycles, cp={} cycles, predicted={:.2}ms (cp={:.2}ms) @ 1.5GHz",
            dag.nodes.len(),
            sched.makespan,
            sched.critical_path_cost,
            sched.ms_at(l4_clock_hz),
            sched.cp_ms_at(l4_clock_hz),
        );
        let util = sched.utilization();
        let avg = util.iter().sum::<f64>() / util.len() as f64;
        let min = util.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = util.iter().cloned().fold(0.0_f64, f64::max);
        eprintln!(
            "cta utilization: avg={:.1}% min={:.1}% max={:.1}%",
            avg * 100.0,
            min * 100.0,
            max * 100.0
        );
    }

    #[test]
    fn schedule_respects_dependencies() {
        let dims = llama_1b_dims(64);
        let dag = ReifiedDag::reify_llama(dims, TileSizes::default_v1());
        let cost = CostModel::from_dag(&dag);
        let sched = schedule(&dag, 8, &cost);

        for nd in &dag.nodes {
            let p = &sched.placements[nd.id.0 as usize];
            for d in &nd.deps {
                let dp = &sched.placements[d.0 as usize];
                assert!(
                    p.start >= dp.end,
                    "node {} starts at {} before dep {} ends at {}",
                    nd.id.0,
                    p.start,
                    d.0,
                    dp.end
                );
            }
        }
    }
}
