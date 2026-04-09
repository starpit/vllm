// SPDX-License-Identifier: Apache-2.0
//! BSP wave-front partitioner for the reified DAG.
//!
//! Input:  `ReifiedDag` + num_ctas + `CostModel` + barrier cost.
//! Output: `WaveSchedule` — a sequence of waves where each wave is a
//!         per-CTA list of NodeIds. Inside a wave, no synchronization is
//!         needed (no two nodes in the same wave depend on each other).
//!         Between waves, one global grid barrier.
//!
//! Algorithm:
//!   1. Compute `earliest_wave[n]` = `1 + max(earliest_wave[d] for d in n.deps)`,
//!      or 0 for sources. The minimum number of waves is exactly the longest
//!      such depth (the critical path in node count).
//!   2. Bucket nodes by their wave index.
//!   3. Within each wave, distribute nodes across CTAs greedily by current
//!      load (each node assigned to the CTA with the lowest accumulated cost
//!      so far in this wave).
//!
//! Cost model is BW-aware: each tile costs `max(compute, mem_bw)`. The
//! predicted makespan is `sum_over_waves(max_per_cta_cost) + K * barrier_cost`.
//! See the field comments on `WaveSchedule` for the breakdown.
//!
//! This is the "minimum K" partition (one wave per critical-path level).
//! A future pass can coarsen by merging adjacent waves if doing so reduces
//! `K * barrier_cost` more than it increases `sum_max_per_cta`.

use crate::kernel_library::CoalescedDag;
use crate::reified_dag::{NodeId, Phase};

/// Per-grid-barrier cost in mma units. Calibrated against the L4
/// gmem-flag spin barrier (~115 µs measured at seq=1024 = ~170k cycles
/// = ~10.6k mma units), but rounded down to 100 because the cost-model
/// scoring should err on the side of *under*-counting barrier cost so
/// the cost-gated coalesce passes are conservative — they only fire
/// when the savings clearly outweigh a barrier worth of work, even at
/// the optimistic barrier-cost estimate.
///
/// All `partition_into_waves` callers in production use this constant.
/// Tests can pass their own value to exercise the predicate logic.
pub const BARRIER_COST_MMA_UNITS: u64 = 100;

/// Score a DAG by simulating its schedule and returning the predicted
/// makespan. This is the cost function the cost-gated coalesce passes
/// minimize over: a coalesce pass is accepted only if applying it
/// reduces this score.
///
/// Builds a fresh `CostModel` from the DAG (cheap — just a struct copy
/// of `dims` + `tiles`) and runs `partition_into_waves` with the
/// production `barrier_cost`. The returned value is in the same
/// mma-unit domain as `WaveSchedule::predicted_cost`.
pub fn score_dag(dag: &CoalescedDag, num_ctas: u32) -> u64 {
    let cost = CostModel::from_dag(dag, num_ctas);
    let sched = partition_into_waves(dag, num_ctas, &cost, BARRIER_COST_MMA_UNITS);
    sched.predicted_cost
}

/// Per-node compute / memory cost in "mma units" (≈16 sm89 cycles each).
/// Identical to the previous list-scheduler model — the cost domain doesn't
/// change when we switch from list scheduling to wave partitioning.
#[derive(Clone, Debug)]
pub struct CostModel {
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
    /// Persistent grid CTA count. Threaded through so the cost
    /// model can compute *per-CTA* cost for wave-cooperative
    /// bindings, including polyalgorithmic GEMM tile-shape choices
    /// where bin-pack rounding (`ceil(work_units / num_ctas)`)
    /// distinguishes one shape from another.
    pub num_ctas: u32,
}

impl CostModel {
    pub fn from_dag(dag: &CoalescedDag, num_ctas: u32) -> Self {
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
            num_ctas,
        }
    }

    /// Total work for a wave-cooperative cutlass GEMM at a given
    /// (M, K, N, tile_M, tile_N, tile_K) shape, in mma units. Returns
    /// the *post-scheduling* total: `num_ctas × ceil(work_units /
    /// num_ctas) × per_tile_mma`. This bakes in the bin-pack rounding
    /// waste explicitly so the cost model can distinguish tile shapes
    /// — smaller tiles tend to leave fewer idle CTA slots when
    /// `work_units` is small relative to `num_ctas`.
    ///
    /// **Convention**: `BoundKernel::cost` returns TOTAL work that
    /// `partition_into_waves` then divides by `num_ctas` for
    /// wave-cooperative bindings (`per_cta = ceil(total / num_ctas)`).
    /// We multiply by `num_ctas` at the end here so the partitioner's
    /// later division gives back the per-CTA value we actually want.
    pub fn cutlass_gemm_total(
        &self,
        m: u32,
        n: u32,
        k: u32,
        tile_m: u32,
        tile_n: u32,
        tile_k: u32,
    ) -> u32 {
        let row_tiles = m.div_ceil(tile_m);
        let col_tiles = n.div_ceil(tile_n);
        let work_units = row_tiles * col_tiles;
        let k_iters = k.div_ceil(tile_k);
        // Per-tile mma instruction count: tile_m/16 × tile_n/8 × tile_k/16
        // (sm89 mma is m16n8k16).
        let mma_per_iter = (tile_m / 16) * (tile_n / 8) * (tile_k / 16);
        let per_tile_mma = k_iters * mma_per_iter;
        let per_cta_units = work_units.div_ceil(self.num_ctas);
        // Pre-multiply by num_ctas so partition_into_waves' div_ceil
        // recovers per_cta_units * per_tile_mma per CTA — i.e. we
        // explicitly account for the bin-pack rounding waste here
        // and the partitioner's later division is exact.
        per_cta_units * per_tile_mma * self.num_ctas
    }

    fn gemm_compute(m: u32, n: u32, k: u32) -> u32 {
        let mma_m = m.div_ceil(16);
        let mma_n = n.div_ceil(8);
        let mma_k = k.div_ceil(16);
        mma_m * mma_n * mma_k
    }

    /// L4 numbers: 300 GB/s global BW, 58 SMs, ~1.5 GHz boost. Per-SM share
    /// ≈ 5.17 GB/s ≈ 3.45 B/cycle. One mma unit ≈ 16 cycles ⇒ ~55 B/mma-unit.
    const BYTES_PER_MMA_UNIT_L4: u32 = 55;

    fn mem_cost(bytes: u32) -> u32 {
        bytes / Self::BYTES_PER_MMA_UNIT_L4 + 1
    }

    fn gemm_total(m: u32, n: u32, k: u32) -> u32 {
        let compute = Self::gemm_compute(m, n, k);
        let bytes = 2 * (m * k + k * n + m * n);
        let memory = Self::mem_cost(bytes);
        compute.max(memory)
    }

    pub fn cost(&self, phase: Phase) -> u32 {
        let m = self.row_tile;
        match phase {
            Phase::AttnNorm | Phase::MlpNorm => {
                let bytes = 2 * (m * self.hidden_dim * 2 + self.hidden_dim);
                Self::mem_cost(bytes) + 8
            }
            Phase::Qkv => Self::gemm_total(m, self.qkv_col_tile, self.hidden_dim),
            Phase::Rope => {
                let qkv = (self.num_attn_heads + 2 * self.num_kv_heads) * self.head_dim;
                Self::mem_cost(2 * 2 * m * qkv) + 4
            }
            Phase::Attention => {
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

/// One wave's worth of work: a per-CTA ordered list of NodeIds. Inside a
/// wave, no synchronization is needed.
#[derive(Clone, Debug)]
pub struct Wave {
    /// `cta_nodes[c]` is the ordered list of NodeIds CTA c executes in this wave.
    pub cta_nodes: Vec<Vec<NodeId>>,
    /// Predicted cost of this wave (max over CTAs of sum of node costs).
    pub max_cta_cost: u64,
    /// Sum of all node costs in this wave (work-conservation lower bound).
    pub total_cost: u64,
}

/// The full BSP schedule: K waves, plus a cost breakdown.
#[derive(Clone, Debug)]
pub struct WaveSchedule {
    pub num_ctas: u32,
    pub waves: Vec<Wave>,
    /// Predicted makespan in mma-unit cycles. Equals
    /// `sum(wave.max_cta_cost) + num_waves * barrier_cost`.
    pub predicted_cost: u64,
    /// Critical-path lower bound (sum of costs along longest weighted path).
    pub critical_path_cost: u64,
    /// Per-barrier cost used for the prediction.
    pub barrier_cost: u64,
}

impl WaveSchedule {
    pub fn num_waves(&self) -> usize {
        self.waves.len()
    }
    pub fn ms_at(&self, gpu_clock_hz: f64) -> f64 {
        self.predicted_cost as f64 / gpu_clock_hz * 1000.0
    }
    pub fn cp_ms_at(&self, gpu_clock_hz: f64) -> f64 {
        self.critical_path_cost as f64 / gpu_clock_hz * 1000.0
    }
}

/// Build a BSP wave schedule.
///
/// `barrier_cost` is in the same mma-unit domain as the cost model. On L4
/// a gmem-flag grid barrier is ~1-2 µs ≈ 1500-3000 cycles ≈ 100-200 mma units.
pub fn partition_into_waves(
    dag: &CoalescedDag,
    num_ctas: u32,
    cost: &CostModel,
    barrier_cost: u64,
) -> WaveSchedule {
    let n = dag.nodes.len();
    // Cost is now per-binding, not per-phase. With only `HandWrittenRowTile`
    // registered the result is identical to the old `cost.cost(nd.phase)`
    // path; new library entries (FlashInferAttentionLayer, …) compute it
    // their own way inside `BoundKernel::cost`.
    let costs: Vec<u32> = dag.nodes.iter().map(|nd| nd.kernel.cost(cost)).collect();

    // ── Step 1: earliest_wave[i] = 1 + max(earliest_wave[d] for d in deps), or 0 ──
    // Topological order is implicit (deps point backward by Phase 1 invariant).
    let mut wave_idx: Vec<u32> = vec![0; n];
    let mut max_wave: u32 = 0;
    for i in 0..n {
        let mut w = 0u32;
        for d in &dag.nodes[i].deps {
            let dw = wave_idx[d.0 as usize] + 1;
            if dw > w {
                w = dw;
            }
        }
        wave_idx[i] = w;
        if w > max_wave {
            max_wave = w;
        }
    }
    let num_waves = (max_wave + 1) as usize;

    // ── Step 2: bucket nodes by wave ──
    let mut nodes_per_wave: Vec<Vec<u32>> = vec![Vec::new(); num_waves];
    for i in 0..n {
        nodes_per_wave[wave_idx[i] as usize].push(i as u32);
    }

    // ── Step 3: load-balance each wave across CTAs ──
    // Sort each wave's nodes by descending cost so the heaviest get placed
    // first (LPT — longest processing time first — gives a known 4/3-OPT bound
    // for makespan minimization on identical machines).
    let mut waves: Vec<Wave> = Vec::with_capacity(num_waves);
    for wave_nodes in nodes_per_wave.iter_mut() {
        wave_nodes.sort_by_key(|&i| std::cmp::Reverse(costs[i as usize]));

        let mut cta_nodes: Vec<Vec<NodeId>> = vec![Vec::new(); num_ctas as usize];
        let mut cta_load: Vec<u64> = vec![0; num_ctas as usize];

        for &node in wave_nodes.iter() {
            let cnode = &dag.nodes[node as usize];
            if cnode.kernel.is_wave_cooperative() {
                // Wave-cooperative bindings (e.g. FlashInferAttentionLayer)
                // are not bin-packed onto one CTA. The runner internally
                // partitions the work across all CTAs in the persistent
                // grid via its own work indptr — we mirror that by
                // placing the same NodeId into every CTA's op stream.
                // The per-CTA load contribution is `cost / num_ctas`
                // so the wave's makespan rollup matches reality
                // (parallel execution, not serialized).
                let per_cta = (costs[node as usize] as u64).div_ceil(num_ctas as u64);
                for cta_id in 0..(num_ctas as usize) {
                    cta_nodes[cta_id].push(NodeId(node));
                    cta_load[cta_id] += per_cta;
                }
            } else {
                // Standard LPT: pick the CTA with lowest current load.
                let (best_cta, _) = cta_load
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, l)| **l)
                    .unwrap();
                cta_nodes[best_cta].push(NodeId(node));
                cta_load[best_cta] += costs[node as usize] as u64;
            }
        }

        let max_cta_cost = *cta_load.iter().max().unwrap_or(&0);
        let total_cost: u64 = cta_load.iter().sum();
        waves.push(Wave {
            cta_nodes,
            max_cta_cost,
            total_cost,
        });
    }

    // ── Step 4: cost rollups ──
    let work_cost: u64 = waves.iter().map(|w| w.max_cta_cost).sum();
    let predicted_cost = work_cost + (num_waves as u64) * barrier_cost;

    // Critical path: longest weighted dep chain. Single forward sweep since
    // nodes are in topo order.
    let mut cp_remaining: Vec<u64> = vec![0; n];
    let mut successors: Vec<Vec<u32>> = vec![Vec::new(); n];
    for nd in &dag.nodes {
        for d in &nd.deps {
            successors[d.0 as usize].push(nd.id.0);
        }
    }
    for i in (0..n).rev() {
        let best = successors[i]
            .iter()
            .map(|&s| cp_remaining[s as usize])
            .max()
            .unwrap_or(0);
        cp_remaining[i] = best + costs[i] as u64;
    }
    let critical_path_cost = *cp_remaining.iter().max().unwrap_or(&0);

    WaveSchedule {
        num_ctas,
        waves,
        predicted_cost,
        critical_path_cost,
        barrier_cost,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_library::coalesce;
    use crate::reified_dag::{LlamaDims, ReifiedDag, TileSizes};
    use crate::target_profile::TargetProfile;

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
    fn waves_are_dependency_safe() {
        // No two nodes in the same wave may depend on each other, and every
        // node's deps must lie in strictly earlier waves.
        let reified = ReifiedDag::reify_llama(llama_1b_dims(64), TileSizes::default_v1());
        let dag = coalesce(&reified);
        let cost = CostModel::from_dag(&dag, 8);
        let sched = partition_into_waves(&dag, 8, &cost, 100);

        // Build node → wave_idx lookup.
        let mut node_wave = vec![u32::MAX; dag.nodes.len()];
        for (w, wave) in sched.waves.iter().enumerate() {
            for cta in &wave.cta_nodes {
                for nid in cta {
                    node_wave[nid.0 as usize] = w as u32;
                }
            }
        }
        // Every node placed.
        assert!(node_wave.iter().all(|&w| w != u32::MAX));

        // Strict-earlier dep invariant.
        for nd in &dag.nodes {
            let my_w = node_wave[nd.id.0 as usize];
            for d in &nd.deps {
                let dep_w = node_wave[d.0 as usize];
                assert!(
                    dep_w < my_w,
                    "node {} in wave {} has dep {} in wave {}",
                    nd.id.0,
                    my_w,
                    d.0,
                    dep_w
                );
            }
        }
    }

    #[test]
    fn num_waves_equals_critical_path_in_nodes() {
        // The minimum-K partition has exactly critical-path-depth waves.
        let profile = TargetProfile::l4_sm89();
        let reified = ReifiedDag::reify_llama(llama_1b_dims(1024), TileSizes::default_v1());
        let dag = coalesce(&reified);
        let cost = CostModel::from_dag(&dag, profile.cooperative_grid_size());
        let sched = partition_into_waves(&dag, profile.cooperative_grid_size(), &cost, 100);
        let cp_nodes = reified.critical_path_depth();
        assert_eq!(sched.num_waves(), cp_nodes as usize);
    }

    #[test]
    fn schedule_1b_seq1024_smoke() {
        let profile = TargetProfile::l4_sm89();
        let reified = ReifiedDag::reify_llama(llama_1b_dims(1024), TileSizes::default_v1());
        let dag = coalesce(&reified);
        let cost = CostModel::from_dag(&dag, profile.cooperative_grid_size());
        // 100 mma units ≈ 1.0 µs at 1.5 GHz — ballpark for an L4 gmem-flag barrier.
        let sched = partition_into_waves(&dag, profile.cooperative_grid_size(), &cost, 100);

        let l4_clock_hz = 1.5e9;
        eprintln!(
            "BSP schedule: nodes={}, waves={}, predicted={:.2}ms (cp={:.2}ms, barrier={:.2}ms) @ 1.5GHz",
            dag.nodes.len(),
            sched.num_waves(),
            sched.ms_at(l4_clock_hz),
            sched.cp_ms_at(l4_clock_hz),
            (sched.num_waves() as u64 * sched.barrier_cost) as f64 / l4_clock_hz * 1000.0,
        );
        // Sanity: predicted >= critical path.
        assert!(sched.predicted_cost >= sched.critical_path_cost);
    }
}
