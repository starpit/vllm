// SPDX-License-Identifier: Apache-2.0
//! Curated implementation library.
//!
//! Each entry is a hand-curated [`Implementation`] corresponding to
//! one specific kernel from one specific source: cuBLAS, vllm-rs's
//! fused-op kernels, FlashInfer, the CUTLASS templates already
//! vendored into the megakernel, and the hand-written
//! `tile_attn_norm` / `tile_rope` / `tile_mlp_norm` device helpers.
//!
//! ## Cost calibration
//!
//! Per-shape wall-clock costs come from the CP4 microbenches in
//! `crates/vllm-tk-test-harness/tests/scheduled_megakernel_test.rs`:
//!
//! - `cp4_cublas_gemm_only_microbench` measured 36.6 ms total for
//!   5 cuBLAS GEMMs × 16 layers at the Llama-1B seq=1024 shape.
//!   Per layer: ~2.29 ms; per GEMM averaged: ~0.46 ms. Different
//!   shapes have different per-call costs which we encode below.
//!
//! - `cp4_natural_sm89_full_forward_microbench` measured 37.8 ms
//!   for the same forward minus attention. The non-GEMM ops add
//!   ~1.2 ms total — vllm-rs's fused norm/silu/rope are nearly
//!   free at this shape.
//!
//! - The megakernel's `fanin rope+at` clock measures FlashInfer
//!   attention at ~10 ms total per forward (~625 µs per layer).
//!
//! - The legacy megakernel's per-arm clocks give us cutlass-device
//!   timings: qkv ~470 µs/layer, oproj ~150 µs/layer, gate+up
//!   ~1380 µs/layer (sum of two), down ~610 µs/layer.
//!
//! These numbers are encoded as constants in the `cost_us`
//! implementations below. When better measurements arrive, update
//! the constants in one place.
//!
//! ## What's in the library today
//!
//! The L4 sm_89 entries the solver can pick from at CP5-A:
//!
//! - **cuBLAS** entries (HostCallback):
//!   `CublasGemmExImpl` for each of qkv, oproj, gate, up, down
//! - **vllm-rs fused** entries (HostCallback):
//!   `RmsNormImpl` (matches RmsNorm),
//!   `SiluAndMulFusedImpl` (matches GateUpConcat + SiluMul as a
//!   two-tile claim because vllm-rs's silu_and_mul_fused expects
//!   the pre-concat layout),
//!   `RotaryEmbeddingImpl` (matches Rope; the kv_cache_write tile
//!   is claimed separately by a lightweight helper for now)
//! - **FlashInfer standalone** entry (HostCallback):
//!   `FlashInferStandaloneImpl` matches Attention
//! - **Hand-written passthroughs** (HostCallback, ~free):
//!   `QkvSplitFreeImpl` for QkvSplit (a no-op; the qkv buffer is
//!   already laid out as Q|K|V and downstream consumers compute
//!   their own offsets),
//!   `KvCacheWriteImpl` for KvCacheWrite (a memcpy),
//!   `ResidualAddImpl` for ResidualAdd (a single elementwise op)
//!
//! Future entries (CP5-D and beyond):
//!
//! - CUTLASS multistage with custom epilogue (residual_add fusion)
//! - CUTLASS sm_90 warp-specialized + TMA
//! - ThunderKittens fused gate-up dual-B (claims a 4-tile subgraph)
//! - cubecl entries
//!
//! Adding a new entry is one new struct + one `register` call below.
//! No solver / framework changes.

use crate::lowering::implementation::{
    Handoff, ImplId, Implementation, LaunchKind, Layout, MatchInfo, Resources, WorkloadConstraint,
};
use crate::lowering::tile_graph::{TileGraph, TileId, TileKind};
use crate::target_profile::TargetProfile;

/// Curated set of [`Implementation`] entries available to the
/// lowering solver.
pub struct ImplementationLibrary {
    pub entries: Vec<Box<dyn Implementation>>,
    /// When set, the solver only considers entries at these indices.
    /// Used by `pruned_for_workload` to pre-select the cheapest
    /// CUTLASS config per GEMM phase, reducing solver branching from
    /// ~60 CUTLASS configs per tile to ~1.
    pub active_indices: Option<Vec<usize>>,
}

impl ImplementationLibrary {
    /// Return the entries the solver should iterate over: either the
    /// active subset (if pruned) or all entries.
    pub fn active_entries(&self) -> Vec<(usize, &dyn Implementation)> {
        match &self.active_indices {
            Some(indices) => indices
                .iter()
                .map(|&i| (i, self.entries[i].as_ref()))
                .collect(),
            None => self
                .entries
                .iter()
                .enumerate()
                .map(|(i, e)| (i, e.as_ref()))
                .collect(),
        }
    }

    /// Pre-select the cheapest CUTLASS config per GEMM phase for the
    /// given workload, returning a library view that only exposes those
    /// winners plus all non-CUTLASS impls. This reduces the solver's
    /// branching factor from ~60 CUTLASS configs per GEMM tile to ~1.
    pub fn pruned_for_workload(&mut self, tile_graph: &TileGraph, profile: &TargetProfile) {
        use std::collections::HashMap;

        let num_tokens = profile.num_tokens();

        // Find one representative seed tile per GEMM phase.
        let mut phase_seeds: HashMap<TileKind, TileId> = HashMap::new();
        for node in tile_graph.iter_topo() {
            if node.kind.is_gemm() && !phase_seeds.contains_key(&node.kind) {
                phase_seeds.insert(node.kind, node.id);
            }
        }

        // For each GEMM phase, find the cheapest single-tile CUTLASS impl.
        // Key: TileKind, Value: (entry index, cost).
        let mut best_cutlass: HashMap<TileKind, (usize, f64)> = HashMap::new();

        for (idx, imp) in self.entries.iter().enumerate() {
            if !imp.name().starts_with("cutlass_") {
                continue;
            }
            if !imp.target_compatible(profile) {
                continue;
            }
            if !imp.workload_constraint().accepts(num_tokens) {
                continue;
            }
            for (&kind, &seed) in &phase_seeds {
                if let Some(m) = imp.matches(tile_graph, seed, profile)
                    && m.claimed_tiles.len() == 1
                {
                    let cost = imp.cost_us(&m, profile);
                    let entry = best_cutlass.entry(kind).or_insert((idx, f64::INFINITY));
                    if cost < entry.1 {
                        *entry = (idx, cost);
                    }
                }
            }
        }

        let winner_set: std::collections::HashSet<usize> =
            best_cutlass.values().map(|(idx, _)| *idx).collect();

        let mut indices = Vec::new();
        for (idx, imp) in self.entries.iter().enumerate() {
            let name = imp.name();
            if name.starts_with("cutlass_")
                && !winner_set.contains(&idx)
                && !name.contains("_residual")
                && !name.contains("_bias")
            {
                // Non-winning single-tile CUTLASS config — skip.
                continue;
            }
            indices.push(idx);
        }

        self.active_indices = Some(indices);
    }

    /// Clear the active indices filter.
    pub fn clear_pruning(&mut self) {
        self.active_indices = None;
    }
}

impl ImplementationLibrary {
    /// Convenience: starter library with Llama 3.2 1B dims (for tests).
    pub fn l4_sm89_starter_default() -> Self {
        Self::l4_sm89_starter(crate::lowering::tile_graph::ModelDims::LLAMA_3_2_1B)
    }

    /// Construct the L40S sm_89 starter library. Same implementation
    /// set as L4 (same ISA), but initializes the cost model with
    /// L40S-measured microbenchmarks so the solver picks optimal tile
    /// configs for the L40S's 142 SMs / 2.52 GHz clock.
    pub fn l40s_sm89_starter(dims: crate::lowering::tile_graph::ModelDims) -> Self {
        l4_cost_model::init_for_target(
            crate::lowering::cost_table::load_l40s_sm89()
                .expect("L40S cost CSV not found — run gpu_cost_sweep on L40S"),
            GpuCostTable::l4_sm89(), // elementwise/attention not yet in CSV; use L4 estimates scaled by SM ratio
        );
        let mut lib = Self::sm89_starter_common(dims);
        // DeviceCallable variants for persistent-kernel grouping.
        // On sm89, uses __syncthreads() handoffs (not mbarrier).
        // The solver groups ops when launch-overhead savings exceed
        // the per-handoff cost (~0.5µs syncthreads vs ~3µs launch).
        lib.add_device_callable_variants(dims);
        lib
    }

    /// Construct the H100 sm_90 starter library. Same implementation
    /// set as sm_89 for now (cuBLAS, CUTLASS 2.x, vllm-rs fused,
    /// FlashInfer), but uses H100 cost data so the solver picks
    /// optimal kernels for Hopper. Future: add sm90 CUTLASS 3.x
    /// and TK persistent-grid implementations.
    pub fn h100_sm90_starter(dims: crate::lowering::tile_graph::ModelDims) -> Self {
        l4_cost_model::init_for_target(
            crate::lowering::cost_table::load_h100_sm90()
                .expect("H100 cost CSV not found — run gpu_cost_sweep on H100"),
            GpuCostTable::l4_sm89(), // elementwise/attention not yet in CSV; use L4 estimates
        );
        let mut lib = Self::sm89_starter_common(dims);
        // Add DeviceCallable variants of all ops for megakernel grouping.
        // These have the same compute cost as standalone (setmaxnreg
        // eliminates register-union penalty on sm90) but use Mbarrier
        // handoffs instead of StreamOrder. The solver groups them into
        // one CompilationUnitId, saving per_launch_overhead_us per op.
        lib.add_device_callable_variants(dims);
        lib
    }

    /// Convenience: H100 starter library with Llama 3.2 1B dims.
    pub fn h100_sm90_starter_default() -> Self {
        Self::h100_sm90_starter(crate::lowering::tile_graph::ModelDims::LLAMA_3_2_1B)
    }

    /// Construct the L4 sm_89 starter library: cuBLAS, vllm-rs
    /// fused, FlashInfer standalone, plus the small passthroughs
    /// for QkvSplit / KvCacheWrite / ResidualAdd. CP5-D extends.
    pub fn l4_sm89_starter(dims: crate::lowering::tile_graph::ModelDims) -> Self {
        // L4 cost model is the default (lazy-initialized), no explicit init needed.
        Self::sm89_starter_common(dims)
    }

    /// Shared implementation library for all sm_89 targets.
    fn sm89_starter_common(dims: crate::lowering::tile_graph::ModelDims) -> Self {
        let mut entries: Vec<Box<dyn Implementation>> = vec![
            // ── cuBLAS GEMM with fused residual (claims GEMM + ResidualAdd
            //    as a two-tile subgraph; uses cublasGemmEx beta=1.0 to
            //    fold the residual add into the GEMM epilogue for free).
            //    Listed BEFORE the standalone CublasGemmExImpl entries
            //    so the solver's cheapest-first tie-break (when both
            //    cost 145 µs at the per-call level) prefers the fused
            //    variant. The fused variant saves a downstream
            Box::new(CublasGemmExWithResidualImpl::new(TileKind::GemmOProj)),
            Box::new(CublasGemmExWithResidualImpl::new(TileKind::GemmDown)),
            // ── Fused QKV GEMM ──
            // Claims {GemmQ, GemmK, GemmV} as one 3-tile subgraph.
            // Listed before individual Q/K/V so the solver prefers it.
            Box::new(CublasFusedQkvGemmImpl),
            // ── Fused QKV GEMM + bias ──
            // Claims all 6 tiles {GemmQ, BiasAdd, GemmK, BiasAdd, GemmV, BiasAdd}.
            // One cuBLAS gemm_bias call with concatenated weight + bias.
            // Listed before individual GEMM+bias so the solver prefers it.
            Box::new(CublasFusedQkvGemmWithBiasImpl::new(dims)),
            // ── cuBLAS GEMM with fused bias epilogue ──
            // For biased models (Qwen2, Qwen2.5, ...). Registered
            // before the standalone CublasGemmExImpl variants so the
            // solver prefers the fused {GemmQ + BiasAdd} cover over
            // the split cover when both are feasible.
            Box::new(CublasGemmExWithBiasImpl::new(TileKind::GemmQ)),
            Box::new(CublasGemmExWithBiasImpl::new(TileKind::GemmK)),
            Box::new(CublasGemmExWithBiasImpl::new(TileKind::GemmV)),
            // ── Fused gate+up GEMM ──
            Box::new(CublasFusedGateUpGemmImpl),
            // ── CUTLASS norm+GEMM prologue fusion (D-3) ──
            Box::new(CutlassNormGemmImpl::new(TileKind::GemmQ, 128, 128)),
            Box::new(CutlassNormGemmImpl::new(TileKind::GemmK, 128, 128)),
            Box::new(CutlassNormGemmImpl::new(TileKind::GemmV, 128, 128)),
            Box::new(CutlassNormGemmImpl::new(TileKind::GemmGate, 128, 128)),
            Box::new(CutlassNormGemmImpl::new(TileKind::GemmUp, 128, 128)),
            Box::new(CublasGemmExImpl::new(TileKind::GemmQ)),
            Box::new(CublasGemmExImpl::new(TileKind::GemmK)),
            Box::new(CublasGemmExImpl::new(TileKind::GemmV)),
            Box::new(CublasGemmExImpl::new(TileKind::GemmOProj)),
            Box::new(CublasGemmExImpl::new(TileKind::GemmGate)),
            Box::new(CublasGemmExImpl::new(TileKind::GemmUp)),
            Box::new(CublasGemmExImpl::new(TileKind::GemmDown)),
            // ── vllm-rs fused ops ──
            Box::new(VllmRsRmsNormImpl),
            // Fused 3-tile claim (QkvSplit + Rope + KvCacheWrite) — listed
            // before the standalone variants so the solver picks it on the
            // first branch.
            // Decode: fused 3-tile (QkvSplit + Rope + KvCacheWrite).
            // Only matches at seq_len <= 1.
            Box::new(VllmRsFusedQkvRopeCacheImpl),
            // Prefill: 3-tile (QkvSplit + Rope + KvCacheWrite) with
            // split_qkv + rotary_both + write_kv_cache.
            // Only matches at seq_len > 1.
            Box::new(VllmRsPrefillRopeCacheImpl),
            Box::new(VllmRsRotaryEmbeddingImpl),
            Box::new(VllmRsSiluAndMulFusedImpl),
            // ── FlashInfer ──
            // Decode: attention_decode_from_cache (seq_len <= 1).
            Box::new(FlashInferStandaloneImpl),
            // Prefill: attention_standard with explicit Q, K, V (seq_len > 1).
            Box::new(FlashInferStandardImpl),
            // ── Free / cheap passthroughs ──
            Box::new(EmbedImpl),
            Box::new(KvCacheWriteImpl),
            Box::new(ResidualAddImpl),
            // Standalone bias_add (covers BiasAdd tiles when the solver
            // picks a bias-less GEMM for the preceding Gemm{phase}).
            Box::new(StandaloneBiasAddImpl),
        ];
        // ── CUTLASS GEMMs (full tile config grid) ──
        // Every (phase × tile_m × tile_n × stages) combo. The solver
        // picks the config that minimizes measured cost per workload.
        entries.extend(CutlassGemmImpl::all_configs(dims));
        entries.extend(CutlassGemmWithResidualImpl::all_configs(dims));
        entries.extend(CutlassGemvImpl::all_configs(dims));
        ImplementationLibrary {
            entries,
            active_indices: None,
        }
    }

    /// Add DeviceCallable (megakernel-embeddable) variants of the key
    /// ops. Each wraps a standalone impl with the DeviceCallableWrapper
    /// so it uses Mbarrier handoffs and can share a CompilationUnitId.
    fn add_device_callable_variants(&mut self, dims: crate::lowering::tile_graph::ModelDims) {
        // DeviceCallable GEMM: use cuBLAS-equivalent costs from CSV.
        // The solver will pick these when grouping saves enough launch overhead.
        for &phase in &[
            TileKind::GemmQ,
            TileKind::GemmK,
            TileKind::GemmV,
            TileKind::GemmOProj,
            TileKind::GemmGate,
            TileKind::GemmUp,
            TileKind::GemmDown,
            TileKind::GemmLmHead,
        ] {
            self.entries
                .push(Box::new(DeviceCallableWrapper::new(Box::new(
                    CublasGemmExImpl::new(phase),
                ))));
        }
        // DeviceCallable fused GEMM+residual (oproj, down).
        for &phase in &[TileKind::GemmOProj, TileKind::GemmDown] {
            self.entries
                .push(Box::new(DeviceCallableWrapper::new(Box::new(
                    CublasGemmExWithResidualImpl::new(phase),
                ))));
        }
        // DeviceCallable elementwise ops.
        self.entries
            .push(Box::new(DeviceCallableWrapper::new(Box::new(
                VllmRsRmsNormImpl,
            ))));
        self.entries
            .push(Box::new(DeviceCallableWrapper::new(Box::new(
                VllmRsSiluAndMulFusedImpl,
            ))));
        // DeviceCallable fused QkvRopeCache (decode).
        self.entries
            .push(Box::new(DeviceCallableWrapper::new(Box::new(
                VllmRsFusedQkvRopeCacheImpl,
            ))));
        // DeviceCallable prefill QkvRopeCache.
        self.entries
            .push(Box::new(DeviceCallableWrapper::new(Box::new(
                VllmRsPrefillRopeCacheImpl,
            ))));
        // DeviceCallable attention (TK native, sm90+ wgmma-based).
        self.entries.push(Box::new(TkAttentionDecodeImpl));
        self.entries.push(Box::new(TkAttentionPrefillImpl));
        // DeviceCallable GEMV (TK matvec for BS=1 decode).
        for &phase in &[
            TileKind::GemmQ,
            TileKind::GemmK,
            TileKind::GemmV,
            TileKind::GemmGate,
            TileKind::GemmUp,
            TileKind::GemmLmHead,
        ] {
            self.entries
                .push(Box::new(DeviceCallableWrapper::new(Box::new(
                    CutlassGemvImpl::new(phase, dims),
                ))));
        }
        // DeviceCallable CUTLASS GEMM configs (best tile per shape).
        // Add a subset of tile configs — the solver picks the best.
        let best_tiles: &[(u32, u32, u32)] = &[
            (128, 128, 4),
            (128, 128, 3),
            (64, 128, 4),
            (64, 128, 3),
            (128, 256, 3),
        ];
        for &phase in &[
            TileKind::GemmQ,
            TileKind::GemmK,
            TileKind::GemmV,
            TileKind::GemmOProj,
            TileKind::GemmGate,
            TileKind::GemmUp,
            TileKind::GemmDown,
            TileKind::GemmLmHead,
        ] {
            for &(tm, tn, s) in best_tiles {
                self.entries
                    .push(Box::new(DeviceCallableWrapper::new(Box::new(
                        CutlassGemmImpl::new(phase, tm, tn, s, dims),
                    ))));
            }
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, id: ImplId) -> &dyn Implementation {
        &*self.entries[id.0 as usize]
    }

    /// Iterate (id, impl) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (ImplId, &dyn Implementation)> {
        self.entries
            .iter()
            .enumerate()
            .map(|(i, b)| (ImplId(i as u32), &**b))
    }
}

// ── Per-GPU microbench-calibrated cost model ──
//
// Costs are interpolated from measured data points at a grid of M
// (num_tokens) values. Each GPU has its own CostTable, populated by
// a one-time microbench sweep. The solver reads costs via `lookup()`
// which lerps between the two nearest grid points.
//
// To add a new GPU: run the `cublas_gemm_sweep_microbench` test on
// the target, then construct a CostTable from the output.

/// One measured data point: (num_tokens, cost_us).
type CostPoint = (u32, f64);

/// Piecewise-linear cost curve from microbench data.
/// `lookup(m)` interpolates between the two nearest grid points.
#[derive(Clone, Debug)]
pub struct CostCurve {
    points: Vec<CostPoint>,
}

impl CostCurve {
    pub fn new(mut points: Vec<CostPoint>) -> Self {
        points.sort_by_key(|(m, _)| *m);
        assert!(!points.is_empty(), "CostCurve needs at least one point");
        Self { points }
    }

    /// Interpolate cost at `m` tokens. Clamps to endpoints.
    pub fn lookup(&self, m: u32) -> f64 {
        if m <= self.points[0].0 {
            return self.points[0].1;
        }
        let last = self.points.len() - 1;
        if m >= self.points[last].0 {
            // Extrapolate linearly from last two points.
            if last == 0 {
                return self.points[0].1;
            }
            let (m1, c1) = self.points[last - 1];
            let (m2, c2) = self.points[last];
            let slope = (c2 - c1) / (m2 - m1) as f64;
            return c2 + slope * (m - m2) as f64;
        }
        // Binary search for the interval.
        let idx = self.points.partition_point(|(pm, _)| *pm <= m);
        let (m0, c0) = self.points[idx - 1];
        let (m1, c1) = self.points[idx];
        let t = (m - m0) as f64 / (m1 - m0) as f64;
        c0 + t * (c1 - c0)
    }
}

/// Per-GPU cost table for all kernel families in the library.
/// Populated by running microbench sweeps on the target GPU.
#[derive(Clone, Debug)]
pub struct GpuCostTable {
    pub gpu_name: String,
    /// cuBLAS GEMM costs per phase.
    pub cublas_qkv: CostCurve,
    pub cublas_oproj: CostCurve,
    pub cublas_gate: CostCurve,
    pub cublas_up: CostCurve,
    pub cublas_down: CostCurve,
    /// Elementwise op costs (all BW-bound, same curve shape).
    pub elementwise_hd: CostCurve, // dim=hidden_dim (norm, res_add)
    pub elementwise_id: CostCurve,  // dim=intermediate_dim (silu)
    pub elementwise_qkv: CostCurve, // dim=qkv_dim (fused rope+cache)
    /// CUTLASS 128×128 GEMM — gate shape (N=8192, K=2048).
    /// Other phases scale proportionally via the analytical model.
    pub cutlass128_gate: CostCurve,
    /// CUTLASS 64×64 GEMM — gate shape.
    pub cutlass64_gate: CostCurve,
    /// FlashInfer attention per-layer cost.
    pub attention: CostCurve,
}

impl GpuCostTable {
    /// L4 sm_89 cost table from microbench sweep (cublas_gemm_sweep_microbench).
    pub fn l4_sm89() -> Self {
        Self {
            gpu_name: "L4 sm_89".into(),
            cublas_qkv: CostCurve::new(vec![
                (1, 13.5),
                (4, 14.5),
                (8, 14.4),
                (16, 14.8),
                (32, 17.6),
                (64, 15.3),
                (128, 21.4),
                (256, 38.1),
                (512, 71.2),
                (1024, 156.5),
                (2048, 339.0),
                (4096, 584.8),
            ]),
            cublas_oproj: CostCurve::new(vec![
                (1, 9.6),
                (4, 13.8),
                (8, 14.0),
                (16, 14.5),
                (32, 17.2),
                (64, 12.6),
                (128, 20.5),
                (256, 34.3),
                (512, 56.7),
                (1024, 108.1),
                (2048, 225.3),
                (4096, 372.0),
            ]),
            cublas_gate: CostCurve::new(vec![
                (1, 27.2),
                (4, 25.8),
                (8, 26.7),
                (16, 28.6),
                (32, 39.0),
                (64, 77.6),
                (128, 71.7),
                (256, 95.9),
                (512, 172.4),
                (1024, 392.5),
                (2048, 783.9),
                (4096, 1549.5),
            ]),
            cublas_up: CostCurve::new(vec![
                (1, 27.6),
                (4, 25.7),
                (8, 26.4),
                (16, 28.2),
                (32, 39.0),
                (64, 77.6),
                (128, 71.6),
                (256, 95.8),
                (512, 172.5),
                (1024, 434.5),
                (2048, 760.2),
                (4096, 1557.6),
            ]),
            cublas_down: CostCurve::new(vec![
                (1, 24.5),
                (4, 44.7),
                (8, 44.7),
                (16, 44.0),
                (32, 37.6),
                (64, 38.2),
                (128, 53.4),
                (256, 99.1),
                (512, 211.7),
                (1024, 472.1),
                (2048, 764.1),
                (4096, 1626.5),
            ]),
            // Elementwise: rough BW model calibrated to ~15µs at M=1024
            elementwise_hd: CostCurve::new(vec![
                (1, 3.0),
                (32, 3.5),
                (128, 5.0),
                (512, 10.0),
                (1024, 15.0),
                (4096, 55.0),
            ]),
            elementwise_id: CostCurve::new(vec![
                (1, 3.0),
                (32, 4.0),
                (128, 8.0),
                (512, 20.0),
                (1024, 35.0),
                (4096, 130.0),
            ]),
            elementwise_qkv: CostCurve::new(vec![
                (1, 3.0),
                (32, 4.0),
                (128, 7.0),
                (512, 15.0),
                (1024, 25.0),
                (4096, 90.0),
            ]),
            // CUTLASS measured on gate shape (N=8192, K=2048).
            // Other phases: scale by measured cuBLAS ratio.
            cutlass128_gate: CostCurve::new(vec![
                (1, 70.8),
                (4, 70.9),
                (16, 70.9),
                (32, 71.1),
                (64, 71.4),
                (128, 72.4),
                (256, 109.0),
                (512, 204.6),
                (1024, 429.0),
            ]),
            cutlass64_gate: CostCurve::new(vec![
                (1, 29.5),
                (4, 29.5),
                (16, 29.8),
                (32, 30.5),
                (64, 37.8),
                (128, 75.4),
                (256, 130.3),
                (512, 274.9),
                (1024, 621.2),
            ]),
            attention: CostCurve::new(vec![
                (1, 50.0),
                (64, 90.0),
                (256, 200.0),
                (1024, 625.0),
                (4096, 2400.0),
            ]),
        }
    }
}

// Cost model backed by the CSV grid. Looks up costs by (M, N, K)
// with interpolation — no hardcoded per-phase shapes.
//
// The active grid is set once via `init_for_target()` and thereafter
// used by every `Implementation::cost_us()` call. Default: L4 sm_89.
mod l4_cost_model {
    use super::GpuCostTable;
    use crate::lowering::cost_table::GpuCostGrid;
    use std::sync::OnceLock;

    static GRID: OnceLock<GpuCostGrid> = OnceLock::new();
    static TABLE: OnceLock<GpuCostTable> = OnceLock::new();

    /// Initialize the cost model for a specific target. Call once before
    /// solving. Subsequent calls are no-ops (first writer wins).
    pub fn init_for_target(grid: GpuCostGrid, table: GpuCostTable) {
        let _ = GRID.set(grid);
        let _ = TABLE.set(table);
    }

    fn grid() -> &'static GpuCostGrid {
        GRID.get_or_init(|| {
            crate::lowering::cost_table::load_l4_sm89()
                .expect("L4 cost CSV not found — run gpu_cost_sweep")
        })
    }

    fn table() -> &'static GpuCostTable {
        TABLE.get_or_init(GpuCostTable::l4_sm89)
    }

    pub fn gemm_us(m: u32, n: u32, k: u32) -> f64 {
        grid().lookup("cublas", m, n, k)
    }

    pub fn elementwise_us(m: u32, dim: u32) -> f64 {
        // Prefer measured data from GPU cost sweep CSV if available.
        // The sweep stores rms_norm and silu_mul with N=dim, K=0.
        // Use rms_norm as the representative elementwise cost (it's
        // the more expensive of the two).
        let g = grid();
        if g.has_data("rms_norm") {
            return g.lookup("rms_norm", m, dim, 0);
        }
        // Fallback: L4 hardcoded curves.
        let t = table();
        match dim {
            2048 => t.elementwise_hd.lookup(m),
            8192 => t.elementwise_id.lookup(m),
            3072 => t.elementwise_qkv.lookup(m),
            512 => t.elementwise_hd.lookup(m) * 0.25,
            _ => t.elementwise_hd.lookup(m) * (dim as f64 / 2048.0),
        }
    }

    /// SiLU+Mul cost — uses measured data if available.
    pub fn silu_mul_us(m: u32, dim: u32) -> f64 {
        let g = grid();
        if g.has_data("silu_mul") {
            return g.lookup("silu_mul", m, dim, 0);
        }
        // Fallback: same as elementwise (silu is similar cost).
        elementwise_us(m, dim)
    }

    /// Fused QKV RoPE cost — uses measured data if available.
    pub fn rope_us(m: u32, total_dim: u32, rotary_dim: u32) -> f64 {
        let g = grid();
        // Try model-specific config first, then generic.
        for name in &["rope_1b", "rope_7b", "rope_70b"] {
            if g.has_data(name) {
                // Match by total_dim (N column).
                let cost = g.lookup(name, m, total_dim, rotary_dim);
                if cost > 0.0 {
                    return cost;
                }
            }
        }
        // Fallback: elementwise at QKV dim.
        elementwise_us(m, total_dim)
    }

    pub fn attention_us(seq: u32) -> f64 {
        table().attention.lookup(seq)
    }

    /// Look up measured CUTLASS cost by tile config.
    /// CSV kernel names: `cutlass_{M}x{N}_s{stages}`.
    pub fn cutlass_gemm_us(m: u32, n: u32, k: u32, tile_m: u32, tile_n: u32, stages: u32) -> f64 {
        let name = format!("cutlass_{}x{}_s{}", tile_m, tile_n, stages);
        grid().lookup(&name, m, n, k)
    }

    /// Fast path: caller pre-computed the cost table key as `&'static str`.
    /// Avoids the format! allocation on every solver backtrack.
    pub fn cutlass_gemm_us_by_key(key: &str, m: u32, n: u32, k: u32) -> f64 {
        grid().lookup(key, m, n, k)
    }

    /// Look up measured CUTLASS GEMV cost (only valid at M=1).
    pub fn gemv_us(m: u32, n: u32, k: u32) -> f64 {
        grid().lookup("cutlass_gemv", m, n, k)
    }
}

// Re-export fixed constants for legacy code.
mod l4_llama_1b_seq1024_costs {
    pub const RMS_NORM_US: f64 = 15.0;
    pub const ROTARY_EMBEDDING_US: f64 = 20.0;
    pub const SILU_AND_MUL_US: f64 = 12.0;
    pub const KV_CACHE_WRITE_US: f64 = 10.0;
    pub const RESIDUAL_ADD_US: f64 = 8.0;
    pub const QKV_SPLIT_US: f64 = 0.0;
    pub const FLASHINFER_ATTENTION_US: f64 = 625.0;
    pub const CUBLAS_GATE_US: f64 = 530.0;
    pub const CUBLAS_UP_US: f64 = 530.0;
    pub const CUBLAS_DOWN_US: f64 = 540.0;
}

// ── cuBLAS GEMM implementation ──

/// Wraps `cublasGemmEx` for one specific GEMM phase. We use one
/// entry per phase rather than one generic GEMM entry so the cost
/// lookup is a direct match — no shape inference needed.
#[derive(Debug)]
pub struct CublasGemmExImpl {
    phase: TileKind,
}

impl CublasGemmExImpl {
    pub fn new(phase: TileKind) -> Self {
        debug_assert!(phase.is_gemm(), "CublasGemmExImpl needs a GEMM tile kind");
        Self { phase }
    }
}

impl Implementation for CublasGemmExImpl {
    fn name(&self) -> &'static str {
        match self.phase {
            TileKind::GemmQ | TileKind::GemmK | TileKind::GemmV => "cublas_gemm_ex_qkv",
            TileKind::GemmOProj => "cublas_gemm_ex_oproj",
            TileKind::GemmGate => "cublas_gemm_ex_gate",
            TileKind::GemmUp => "cublas_gemm_ex_up",
            TileKind::GemmDown => "cublas_gemm_ex_down",
            _ => "cublas_gemm_ex_unknown",
        }
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        // cuBLAS works on every CUDA target we care about.
        true
    }

    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        let node = &tile_graph.nodes[seed.0 as usize];
        if node.kind != self.phase {
            return None;
        }
        // Single-tile claim. Boundary inputs = node.deps; boundary
        // outputs = [seed] (downstream consumers read the GEMM
        // output as a tile).
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: node.deps.clone(),
            boundary_outputs: vec![seed],
            layer: node.layer,
        })
    }

    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        // LLaMA 1B shapes: HD=2048, ID=8192, QKV_DIM=3072
        let m = profile.num_tokens();
        match self.phase {
            TileKind::GemmQ | TileKind::GemmK | TileKind::GemmV => {
                l4_cost_model::gemm_us(m, 3072, 2048)
            }
            TileKind::GemmOProj => l4_cost_model::gemm_us(m, 2048, 2048),
            TileKind::GemmGate => l4_cost_model::gemm_us(m, 8192, 2048),
            TileKind::GemmUp => l4_cost_model::gemm_us(m, 8192, 2048),
            TileKind::GemmDown => l4_cost_model::gemm_us(m, 2048, 8192),
            _ => 0.0,
        }
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        // cuBLAS picks its own internal kernel — we don't share its
        // register / shmem budget. Report a sentinel that says
        // "this is a host-callback library kernel; its resource
        // usage doesn't affect any other implementation's budget."
        // The constraint solver special-cases HostCallback impls
        // and doesn't union their resources with anything.
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 0,
            threads_per_cta: 0,
        }
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        // Stream-ordered or via cudaEvent across streams.
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }
}

/// Fused QKV GEMM via cuBLAS. Claims {GemmQ, GemmK, GemmV} as a
/// single three-tile subgraph. The loader concatenates the three
/// separate weight tensors into one contiguous [q_size+2*kv_size, hidden]
/// buffer; the generated forward code runs one GEMM instead of three.
///
/// The solver prefers this over 3× `CublasGemmExImpl` because one
/// large GEMM is cheaper than three small ones (better GPU utilization,
/// one launch instead of three).
#[derive(Debug)]
pub struct CublasFusedQkvGemmImpl;

impl Implementation for CublasFusedQkvGemmImpl {
    fn name(&self) -> &'static str {
        "cublas_fused_qkv_gemm"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        let node = &tile_graph.nodes[seed.0 as usize];
        // Only seed on GemmQ — avoids triple-matching the same group.
        if node.kind != TileKind::GemmQ {
            return None;
        }
        let layer = node.layer;
        let deps = &node.deps;

        // Find sibling GemmK and GemmV on the same layer with the
        // same dependencies (i.e. same input activation).
        let k_tile = tile_graph
            .nodes
            .iter()
            .find(|n| n.kind == TileKind::GemmK && n.layer == layer && n.deps == *deps)?;
        let v_tile = tile_graph
            .nodes
            .iter()
            .find(|n| n.kind == TileKind::GemmV && n.layer == layer && n.deps == *deps)?;

        Some(MatchInfo {
            claimed_tiles: vec![seed, k_tile.id, v_tile.id],
            boundary_inputs: deps.clone(),
            boundary_outputs: vec![seed, k_tile.id, v_tile.id],
            layer,
        })
    }

    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        // One fused GEMM: [M, hidden] × [qkv_dim, hidden]^T.
        // Cheaper than 3 separate GEMMs due to better utilization.
        // Uses hardcoded Llama 3B dims (qkv=3072, hidden=3072) — the
        // cost model is approximate; exact dims don't change the
        // solver's relative preference for fused vs unfused.
        let m = profile.num_tokens();
        l4_cost_model::gemm_us(m, 3072, 3072)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 0,
            threads_per_cta: 0,
        }
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        // One fused output: [M, qkv_dim]
        vec![Layout::RowMajorBf16]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }
}

/// Fused QKV GEMM + bias via cuBLAS. Claims 6 tiles:
/// `{GemmQ, BiasAdd, GemmK, BiasAdd, GemmV, BiasAdd}`.
///
/// The loader concatenates Q/K/V weights into one `[qkv_dim, hidden]`
/// buffer and Q/K/V biases into one `[qkv_dim]` vector. cuBLAS runs
/// one `gemm_bias` call (cublasLt with BIAS_POINTER epilogue) that
/// produces a concatenated `[M, qkv_dim]` output with bias folded in.
///
/// This is the sm89 fast path for biased models (Qwen2). On sm90+
/// the megakernel may prefer unfused CUTLASS GEMMs + separate bias,
/// since device-callable launches have no overhead.
#[derive(Debug)]
pub struct CublasFusedQkvGemmWithBiasImpl {
    dims: crate::lowering::tile_graph::ModelDims,
}

impl CublasFusedQkvGemmWithBiasImpl {
    pub fn new(dims: crate::lowering::tile_graph::ModelDims) -> Self {
        Self { dims }
    }
}

impl Implementation for CublasFusedQkvGemmWithBiasImpl {
    fn name(&self) -> &'static str {
        "cublas_fused_qkv_gemm_with_bias"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        let node = &tile_graph.nodes[seed.0 as usize];
        // Only seed on GemmQ — avoids triple-matching the same group.
        if node.kind != TileKind::GemmQ {
            return None;
        }
        let layer = node.layer;
        let deps = &node.deps;

        // Find sibling GemmK and GemmV on the same layer with the
        // same dependencies (i.e. same input activation).
        let k_tile = tile_graph
            .nodes
            .iter()
            .find(|n| n.kind == TileKind::GemmK && n.layer == layer && n.deps == *deps)?;
        let v_tile = tile_graph
            .nodes
            .iter()
            .find(|n| n.kind == TileKind::GemmV && n.layer == layer && n.deps == *deps)?;

        // Find downstream BiasAdd for each GEMM tile.
        let q_bias = tile_graph
            .nodes
            .iter()
            .find(|n| n.kind == TileKind::BiasAdd && n.layer == layer && n.deps.contains(&seed))?;
        let k_bias = tile_graph.nodes.iter().find(|n| {
            n.kind == TileKind::BiasAdd && n.layer == layer && n.deps.contains(&k_tile.id)
        })?;
        let v_bias = tile_graph.nodes.iter().find(|n| {
            n.kind == TileKind::BiasAdd && n.layer == layer && n.deps.contains(&v_tile.id)
        })?;

        Some(MatchInfo {
            claimed_tiles: vec![seed, q_bias.id, k_tile.id, k_bias.id, v_tile.id, v_bias.id],
            boundary_inputs: deps.clone(),
            // Outputs are the BiasAdd tiles (downstream consumers
            // depend on these, not the raw GEMM tiles).
            boundary_outputs: vec![q_bias.id, k_bias.id, v_bias.id],
            layer,
        })
    }

    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        // One fused GEMM: [M, hidden] × [qkv_dim, hidden]^T.
        // Bias epilogue is free (folded into cublasLt writeback).
        let m = profile.num_tokens();
        let n = self.dims.qkv_dim();
        let k = self.dims.hidden_size;
        l4_cost_model::gemm_us(m, n, k)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 0,
            threads_per_cta: 0,
        }
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }
}

/// Fused gate+up GEMM via cuBLAS. Claims {GemmGate, GemmUp} as a
/// single two-tile subgraph. The loader concatenates the gate and up
/// weight tensors into one contiguous [2*intermediate, hidden] buffer;
/// the generated forward code runs one GEMM instead of two, then
/// `silu_and_mul_fused` splits the output.
#[derive(Debug)]
pub struct CublasFusedGateUpGemmImpl;

impl Implementation for CublasFusedGateUpGemmImpl {
    fn name(&self) -> &'static str {
        "cublas_fused_gate_up_gemm"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        let node = &tile_graph.nodes[seed.0 as usize];
        // Only seed on GemmGate — avoids double-matching.
        if node.kind != TileKind::GemmGate {
            return None;
        }
        let layer = node.layer;
        let deps = &node.deps;

        // Find sibling GemmUp on the same layer with the same deps.
        let up_tile = tile_graph
            .nodes
            .iter()
            .find(|n| n.kind == TileKind::GemmUp && n.layer == layer && n.deps == *deps)?;

        Some(MatchInfo {
            claimed_tiles: vec![seed, up_tile.id],
            boundary_inputs: deps.clone(),
            boundary_outputs: vec![seed, up_tile.id],
            layer,
        })
    }

    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        // One fused GEMM instead of two: saves a launch + better
        // utilization. Cost < sum of two separate gate/up GEMMs.
        let m = profile.num_tokens();
        let separate_cost =
            l4_cost_model::gemm_us(m, 8192, 2048) + l4_cost_model::gemm_us(m, 8192, 2048);
        // Fused is ~10-20% cheaper than the sum (one launch, better
        // memory coalescing on the weight read).
        // Always cheaper than 2 separate GEMMs — ensures uniform
        // fusion decision across all buckets (the struct is shared).
        separate_cost * 0.5
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 0,
            threads_per_cta: 0,
        }
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }
}

/// `cublasGemmEx` with `beta=1.0` epilogue. Claims a two-tile
/// subgraph: `(GemmOProj + ResidualAdd)` or `(GemmDown + ResidualAdd)`.
/// The residual add comes "free" via cuBLAS's beta parameter — the
/// solver should prefer this entry over the separate
/// `(CublasGemmExImpl, ResidualAddImpl)` cover because it eliminates
/// the standalone residual_add cost AND a launch boundary.
///
/// **Why only oproj and down**: those are the two GEMMs in a Llama
/// layer that have a residual operand on their output buffer. qkv,
/// gate, up GEMMs write to fresh buffers (no residual to fold).
#[derive(Debug)]
pub struct CublasGemmExWithResidualImpl {
    phase: TileKind,
}

impl CublasGemmExWithResidualImpl {
    pub fn new(phase: TileKind) -> Self {
        debug_assert!(matches!(phase, TileKind::GemmOProj | TileKind::GemmDown));
        Self { phase }
    }
}

impl Implementation for CublasGemmExWithResidualImpl {
    fn name(&self) -> &'static str {
        match self.phase {
            TileKind::GemmOProj => "cublas_gemm_ex_oproj_with_residual",
            TileKind::GemmDown => "cublas_gemm_ex_down_with_residual",
            _ => "cublas_gemm_ex_with_residual_unknown",
        }
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        let node = &tile_graph.nodes[seed.0 as usize];
        // Match starting from either the GEMM or the ResidualAdd
        // end of the pattern. Find the (GEMM, ResidualAdd) pair.
        let (gemm_id, gemm_node, residual_id) = match node.kind {
            kind if kind == self.phase => {
                // Find a downstream ResidualAdd that consumes this GEMM.
                let residual = tile_graph
                    .nodes
                    .iter()
                    .find(|n| n.kind == TileKind::ResidualAdd && n.deps.contains(&seed))?;
                (seed, node, residual.id)
            }
            TileKind::ResidualAdd => {
                // Find a GEMM dep of the right phase.
                let gemm = node
                    .deps
                    .iter()
                    .copied()
                    .find(|d| tile_graph.nodes[d.0 as usize].kind == self.phase)?;
                (gemm, &tile_graph.nodes[gemm.0 as usize], seed)
            }
            _ => return None,
        };

        // The ResidualAdd's deps should be (hidden_in, gemm_output).
        // The GEMM's deps are the GEMM operands.
        // Boundary inputs include the GEMM's deps + the ResidualAdd's
        // OTHER dep (the hidden_in residual operand).
        let residual_node = &tile_graph.nodes[residual_id.0 as usize];
        let mut boundary_inputs = gemm_node.deps.clone();
        for d in &residual_node.deps {
            if *d != gemm_id && !boundary_inputs.contains(d) {
                boundary_inputs.push(*d);
            }
        }
        Some(MatchInfo {
            claimed_tiles: vec![gemm_id, residual_id],
            boundary_inputs,
            // The output is the ResidualAdd tile (hidden_states write).
            boundary_outputs: vec![residual_id],
            layer: gemm_node.layer,
        })
    }

    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        // Same as the plain cuBLAS GEMM — beta=1 doesn't change wall
        // clock vs beta=0 in cuBLAS's mainloop.
        let m = profile.num_tokens();
        match self.phase {
            TileKind::GemmOProj => l4_cost_model::gemm_us(m, 2048, 2048),
            TileKind::GemmDown => l4_cost_model::gemm_us(m, 2048, 8192),
            _ => 0.0,
        }
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 0,
            threads_per_cta: 0,
        }
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }
}

// ── vllm-rs fused ops ──

#[derive(Debug)]
pub struct VllmRsRmsNormImpl;

impl Implementation for VllmRsRmsNormImpl {
    fn name(&self) -> &'static str {
        "vllm_rs_rms_norm"
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        let node = &tile_graph.nodes[seed.0 as usize];
        if node.kind != TileKind::RmsNorm {
            return None;
        }
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: node.deps.clone(),
            boundary_outputs: vec![seed],
            layer: node.layer,
        })
    }
    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        l4_cost_model::elementwise_us(profile.num_tokens(), 2048)
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 0,
            threads_per_cta: 0,
        }
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16]
    }
}

#[derive(Debug)]
pub struct VllmRsRotaryEmbeddingImpl;

impl Implementation for VllmRsRotaryEmbeddingImpl {
    fn name(&self) -> &'static str {
        "vllm_rs_rotary_embedding"
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        let node = &tile_graph.nodes[seed.0 as usize];
        if node.kind != TileKind::Rope {
            return None;
        }
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: node.deps.clone(),
            boundary_outputs: vec![seed],
            layer: node.layer,
        })
    }
    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        l4_cost_model::rope_us(profile.num_tokens(), 2048, 64)
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 0,
            threads_per_cta: 0,
        }
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16]
    }
}

/// vllm-rs's `silu_and_mul_fused` operates on a packed
/// `[seq, 2*intermediate]` buffer, so it CLAIMS the
/// `(GateUpConcat + SiluMul)` two-tile subgraph as a single
/// invocation.
#[derive(Debug)]
pub struct VllmRsSiluAndMulFusedImpl;

impl Implementation for VllmRsSiluAndMulFusedImpl {
    fn name(&self) -> &'static str {
        "vllm_rs_silu_and_mul_fused"
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        let node = &tile_graph.nodes[seed.0 as usize];
        // We seed-match either the GateUpConcat or the SiluMul end
        // of the pattern. The seed-rooted enumeration in the solver
        // will call us once per tile in the graph; we accept whichever
        // tile starts the match.
        let (concat_id, silu_id) = match node.kind {
            TileKind::GateUpConcat => {
                // Find the SiluMul that consumes this concat.
                let silu = tile_graph
                    .nodes
                    .iter()
                    .find(|n| n.kind == TileKind::SiluMul && n.deps.contains(&seed))?;
                (seed, silu.id)
            }
            TileKind::SiluMul => {
                // The dep is the GateUpConcat.
                let concat = node
                    .deps
                    .iter()
                    .copied()
                    .find(|d| tile_graph.nodes[d.0 as usize].kind == TileKind::GateUpConcat)?;
                (concat, seed)
            }
            _ => return None,
        };
        let concat_node = &tile_graph.nodes[concat_id.0 as usize];
        Some(MatchInfo {
            claimed_tiles: vec![concat_id, silu_id],
            // Inputs are GateGemm + UpGemm (deps of GateUpConcat).
            boundary_inputs: concat_node.deps.clone(),
            // Output is the SiluMul tile (downstream consumer).
            boundary_outputs: vec![silu_id],
            layer: node.layer,
        })
    }
    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        l4_cost_model::silu_mul_us(profile.num_tokens(), 8192) // intermediate_dim
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 0,
            threads_per_cta: 0,
        }
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16]
    }
}

// ── DeviceCallable attention (sm90+) — TK-native ──
//
// Attention that runs inside the persistent megakernel using
// ThunderKittens' native attention primitives (`mma_ABt` for Q×K^T
// scores, online softmax via `exp2`/`max`/`sub_row`, `mma_AB` for
// attn×V accumulation). Fully device-callable — no separate kernel
// launch needed.
//
// For decode (seq_len=1): partial attention per SM with TMA-streamed
// K/V pages from the paged cache, plus a cross-SM log-sum-exp
// reduction step (see Megakernels `attention_partial.cu` +
// `attention_reduction.cu`).
//
// For prefill: 64-row Q blocks × 128-token KV pages with causal
// masking and 3-stage TMA pipeline (see Megakernels
// `attention_prefill.cu`).
//
// Both variants receive input via mbarrier from the preceding
// rope/split op and hand off output to oproj via mbarrier.

/// Decode attention via TK-native `mma_ABt`/`mma_AB` with online
/// softmax. Paged KV cache, seq_len <= 1.
#[derive(Debug)]
pub struct TkAttentionDecodeImpl;

impl Implementation for TkAttentionDecodeImpl {
    fn name(&self) -> &'static str {
        "tk_attention_decode"
    }
    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        // sm90+ only (needs wgmma + mbarrier).
        profile.lowering.regs_dynamic_per_warpgroup
            && profile.lowering.mbarrier_handoff_us.is_some()
    }
    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange { min: 0, max: 1 }
    }
    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        let node = &tile_graph.nodes[seed.0 as usize];
        if node.kind != TileKind::Attention {
            return None;
        }
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: node.deps.clone(),
            boundary_outputs: vec![seed],
            layer: node.layer,
        })
    }
    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        // TK-native attention (mma_ABt + online softmax + mma_AB).
        // TODO: replace with measured TK attention cost from B2 benchmarks.
        l4_cost_model::attention_us(profile.num_tokens())
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 228 * 1024, // 228 KiB for Q/K/V tiles + softmax scratch
            regs_per_thread: 232,    // wgmma consumer warpgroup
            threads_per_cta: 128,    // 1 warpgroup
        }
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::DeviceCallable
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::Mbarrier, Handoff::Internal, Handoff::StreamOrder]
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::Mbarrier, Handoff::Internal, Handoff::StreamOrder]
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::Any; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16]
    }
}

/// Prefill attention via TK-native primitives (64-row Q blocks,
/// causal masking, 3-stage TMA pipeline). Explicit Q, K, V, seq_len > 1.
#[derive(Debug)]
pub struct TkAttentionPrefillImpl;

impl Implementation for TkAttentionPrefillImpl {
    fn name(&self) -> &'static str {
        "tk_attention_prefill"
    }
    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.lowering.regs_dynamic_per_warpgroup
            && profile.lowering.mbarrier_handoff_us.is_some()
    }
    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange {
            min: 2,
            max: u32::MAX,
        }
    }
    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        let node = &tile_graph.nodes[seed.0 as usize];
        if node.kind != TileKind::Attention {
            return None;
        }
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: node.deps.clone(),
            boundary_outputs: vec![seed],
            layer: node.layer,
        })
    }
    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        l4_cost_model::attention_us(profile.num_tokens())
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 228 * 1024,
            regs_per_thread: 232,
            threads_per_cta: 128,
        }
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::DeviceCallable
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::Mbarrier, Handoff::Internal, Handoff::StreamOrder]
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::Mbarrier, Handoff::Internal, Handoff::StreamOrder]
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::Any; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16]
    }
}

// ── FlashInfer standalone attention ──

#[derive(Debug)]
pub struct FlashInferStandaloneImpl;

impl Implementation for FlashInferStandaloneImpl {
    fn name(&self) -> &'static str {
        "flashinfer_standalone_fa2"
    }
    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        // Decode path: attention reads from paged KV cache.
        profile.seq_len <= 1
    }
    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        let node = &tile_graph.nodes[seed.0 as usize];
        if node.kind != TileKind::Attention {
            return None;
        }
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: node.deps.clone(),
            boundary_outputs: vec![seed],
            layer: node.layer,
        })
    }
    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        l4_cost_model::attention_us(profile.num_tokens())
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 0,
            threads_per_cta: 0,
        }
    }
    fn launch_kind(&self) -> LaunchKind {
        // FlashInfer's standalone wrapper is a regular launch from
        // the host's perspective (cudaLaunchKernel under the hood).
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        // Q is row-major; KV cache is paged.
        let mut layouts = Vec::with_capacity(m.boundary_inputs.len());
        for _ in &m.boundary_inputs {
            layouts.push(Layout::Any);
        }
        layouts
    }
    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16]
    }
}

// ── Free / cheap passthroughs ──

/// QkvSplit is a logical-only operation: the qkv buffer is laid
/// out as `[Q | K | V]` and downstream consumers compute their own
/// pointer offsets into it. The "split" is free.
#[derive(Debug)]
pub struct QkvSplitFreeImpl;

impl Implementation for QkvSplitFreeImpl {
    fn name(&self) -> &'static str {
        "qkv_split_free"
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        let node = &tile_graph.nodes[seed.0 as usize];
        if node.kind != TileKind::QkvSplit {
            return None;
        }
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: node.deps.clone(),
            boundary_outputs: vec![seed],
            layer: node.layer,
        })
    }
    fn cost_us(&self, _m: &MatchInfo, _profile: &TargetProfile) -> f64 {
        l4_llama_1b_seq1024_costs::QKV_SPLIT_US
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 0,
            threads_per_cta: 0,
        }
    }
    fn launch_kind(&self) -> LaunchKind {
        // Pseudo-launch — no kernel actually issued. The runtime
        // backend just records the boundary and moves on.
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::Internal]
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::Internal]
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::Any; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::Any]
    }
}

/// Embedding-table gather: `out[i] = embed_tokens[input_ids[i]]`.
/// Claims a single `TileKind::Embed` tile (always the pre-loop
/// tile at `layer == PRE_LOOP_LAYER`). The codegen emits a call
/// to `kernels::embedding_gather`.
#[derive(Debug)]
pub struct EmbedImpl;

impl Implementation for EmbedImpl {
    fn name(&self) -> &'static str {
        "embedding_gather"
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        let node = &tile_graph.nodes[seed.0 as usize];
        if node.kind != TileKind::Embed {
            return None;
        }
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: node.deps.clone(),
            boundary_outputs: vec![seed],
            layer: node.layer,
        })
    }
    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        // Memory-bound gather, ~1-2µs for small seq lens, scales
        // with num_tokens * hidden. Reuse the hd elementwise curve
        // as a rough proxy.
        l4_cost_model::elementwise_us(profile.num_tokens(), 2048)
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 0,
            threads_per_cta: 0,
        }
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::Any; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16]
    }
}

#[derive(Debug)]
pub struct KvCacheWriteImpl;

impl Implementation for KvCacheWriteImpl {
    fn name(&self) -> &'static str {
        "kv_cache_write"
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        let node = &tile_graph.nodes[seed.0 as usize];
        if node.kind != TileKind::KvCacheWrite {
            return None;
        }
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: node.deps.clone(),
            boundary_outputs: vec![seed],
            layer: node.layer,
        })
    }
    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        // KV cache scatter: seq * kv_size * 2 bytes read + write
        l4_cost_model::elementwise_us(profile.num_tokens(), 512) // kv_dim = 8*64 = 512
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 0,
            threads_per_cta: 0,
        }
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::Any; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::PagedKvBf16]
    }
}

/// Standalone per-column bias add: `out[row, col] += bias[col]`.
/// Claims a single `BiasAdd` tile. Dispatches
/// `kernels::bias_add_inplace` on the GEMM output buffer. The tile
/// preceding it in the DAG is the `Gemm*` that produced the output,
/// and it's claimed by a separate (unbiased) GEMM impl — so the
/// solver's cover for `gemm_bias` becomes `CutlassGemm* + BiasAdd`
/// (two launches) or `CublasGemmExWithBiasImpl` (one launch, fused).
#[derive(Debug)]
pub struct StandaloneBiasAddImpl;

impl Implementation for StandaloneBiasAddImpl {
    fn name(&self) -> &'static str {
        "standalone_bias_add"
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        let node = &tile_graph.nodes[seed.0 as usize];
        if node.kind != TileKind::BiasAdd {
            return None;
        }
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: node.deps.clone(),
            boundary_outputs: vec![seed],
            layer: node.layer,
        })
    }
    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        // Memory-bound broadcast add on the GEMM output. BW is
        // proportional to M*N (the output buffer); for QKV, N ≈
        // qkv_dim ≈ 3072, same order as the fused rope+cache path.
        l4_cost_model::elementwise_us(profile.num_tokens(), 3072)
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 0,
            threads_per_cta: 0,
        }
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16]
    }
}

/// `cublasGemmEx` with a fused bias epilogue (`cublas.gemm_bias`).
/// Claims a two-tile subgraph `(Gemm{phase}, BiasAdd)`. The solver
/// should prefer this over `CublasGemmExImpl + StandaloneBiasAddImpl`
/// when both are feasible, because one launch is cheaper than two
/// and the bias epilogue is ~free in cuBLAS's mainloop.
///
/// Registered per phase — for Qwen2 only Q/K/V are needed today,
/// but the impl is parameterized so future models with biased
/// gate/up/down/o_proj projections can reuse it.
#[derive(Debug)]
pub struct CublasGemmExWithBiasImpl {
    phase: TileKind,
}

impl CublasGemmExWithBiasImpl {
    pub fn new(phase: TileKind) -> Self {
        debug_assert!(
            phase.is_gemm(),
            "CublasGemmExWithBiasImpl needs a GEMM tile kind"
        );
        Self { phase }
    }
}

impl Implementation for CublasGemmExWithBiasImpl {
    fn name(&self) -> &'static str {
        match self.phase {
            TileKind::GemmQ | TileKind::GemmK | TileKind::GemmV => "cublas_gemm_ex_qkv_with_bias",
            TileKind::GemmOProj => "cublas_gemm_ex_oproj_with_bias",
            TileKind::GemmGate => "cublas_gemm_ex_gate_with_bias",
            TileKind::GemmUp => "cublas_gemm_ex_up_with_bias",
            TileKind::GemmDown => "cublas_gemm_ex_down_with_bias",
            TileKind::GemmLmHead => "cublas_gemm_ex_lm_head_with_bias",
            _ => "cublas_gemm_ex_with_bias_unknown",
        }
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        let node = &tile_graph.nodes[seed.0 as usize];
        // Seedable from either end: the GEMM or the BiasAdd.
        let (gemm_id, gemm_node, bias_id) = match node.kind {
            kind if kind == self.phase => {
                let bias = tile_graph
                    .nodes
                    .iter()
                    .find(|n| n.kind == TileKind::BiasAdd && n.deps.contains(&seed))?;
                (seed, node, bias.id)
            }
            TileKind::BiasAdd => {
                let gemm = node
                    .deps
                    .iter()
                    .copied()
                    .find(|d| tile_graph.nodes[d.0 as usize].kind == self.phase)?;
                (gemm, &tile_graph.nodes[gemm.0 as usize], seed)
            }
            _ => return None,
        };
        Some(MatchInfo {
            claimed_tiles: vec![gemm_id, bias_id],
            boundary_inputs: gemm_node.deps.clone(),
            boundary_outputs: vec![bias_id],
            layer: gemm_node.layer,
        })
    }
    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        // Identical to the plain cuBLAS GEMM — epilogue bias is
        // folded into the mainloop and doesn't change wall clock.
        let m = profile.num_tokens();
        match self.phase {
            TileKind::GemmQ | TileKind::GemmK | TileKind::GemmV => {
                l4_cost_model::gemm_us(m, 3072, 2048)
            }
            TileKind::GemmOProj => l4_cost_model::gemm_us(m, 2048, 2048),
            TileKind::GemmGate => l4_cost_model::gemm_us(m, 8192, 2048),
            TileKind::GemmUp => l4_cost_model::gemm_us(m, 8192, 2048),
            TileKind::GemmDown => l4_cost_model::gemm_us(m, 2048, 8192),
            _ => 0.0,
        }
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 0,
            threads_per_cta: 0,
        }
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16]
    }
    fn is_compute_bound(&self) -> bool {
        true
    }
}

#[derive(Debug)]
pub struct ResidualAddImpl;

impl Implementation for ResidualAddImpl {
    fn name(&self) -> &'static str {
        "residual_add"
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        let node = &tile_graph.nodes[seed.0 as usize];
        if node.kind != TileKind::ResidualAdd {
            return None;
        }
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: node.deps.clone(),
            boundary_outputs: vec![seed],
            layer: node.layer,
        })
    }
    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        l4_cost_model::elementwise_us(profile.num_tokens(), 2048)
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 0,
            threads_per_cta: 0,
        }
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16]
    }
}

/// vllm-rs `fused_qkv_rope_cache_bf16` claims the 3-tile subgraph
/// `(QkvSplit → Rope → KvCacheWrite)` as a single host-callback
/// kernel: it reads the packed `[Q | K | V]` qkv buffer, applies
/// RoPE in place to Q and K, writes Q to a separate `q_post_rope`
/// buffer, and scatters K/V into the paged cache via `slot_mapping`
/// — exactly what the three logical tiles do separately.
#[derive(Debug)]
pub struct VllmRsFusedQkvRopeCacheImpl;

impl Implementation for VllmRsFusedQkvRopeCacheImpl {
    fn name(&self) -> &'static str {
        "vllm_rs_fused_qkv_rope_cache"
    }
    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        // Decode only: fused split + Q rope + cache write.
        // Prefill uses split_qkv + rotary_both + write_kv_cache.
        profile.seq_len <= 1
    }
    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        let split_node = &tile_graph.nodes[seed.0 as usize];
        if split_node.kind != TileKind::QkvSplit {
            return None;
        }
        let rope = tile_graph
            .nodes
            .iter()
            .find(|n| n.kind == TileKind::Rope && n.deps.contains(&seed))?;
        let kvw = tile_graph
            .nodes
            .iter()
            .find(|n| n.kind == TileKind::KvCacheWrite && n.deps.contains(&rope.id))?;
        Some(MatchInfo {
            claimed_tiles: vec![seed, rope.id, kvw.id],
            boundary_inputs: split_node.deps.clone(),
            boundary_outputs: vec![rope.id, kvw.id],
            layer: split_node.layer,
        })
    }
    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        // Fused rope + qkv split + kv cache write: reads qkv_dim,
        // writes q_dim + kv scatter. One kernel instead of three.
        l4_cost_model::rope_us(profile.num_tokens(), 3072, 64) // LLaMA 1B: total_dim=3072, rotary=64
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 0,
            threads_per_cta: 0,
        }
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16, Layout::PagedKvBf16]
    }
}

/// Prefill rope + cache: claims [QkvSplit + Rope + KvCacheWrite].
/// Uses split_qkv + rotary_embedding_inplace (both Q and K) + write_kv_cache.
/// Only matches at seq_len > 1 (prefill).
#[derive(Debug)]
pub struct VllmRsPrefillRopeCacheImpl;

impl Implementation for VllmRsPrefillRopeCacheImpl {
    fn name(&self) -> &'static str {
        "vllm_rs_prefill_rope_cache"
    }
    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.seq_len > 1
    }
    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        // Same subgraph as the decode variant.
        let split_node = &tile_graph.nodes[seed.0 as usize];
        if split_node.kind != TileKind::QkvSplit {
            return None;
        }
        let rope = tile_graph
            .nodes
            .iter()
            .find(|n| n.kind == TileKind::Rope && n.deps.contains(&seed))?;
        let kvw = tile_graph
            .nodes
            .iter()
            .find(|n| n.kind == TileKind::KvCacheWrite && n.deps.contains(&rope.id))?;
        Some(MatchInfo {
            claimed_tiles: vec![seed, rope.id, kvw.id],
            boundary_inputs: split_node.deps.clone(),
            boundary_outputs: vec![rope.id, kvw.id],
            layer: split_node.layer,
        })
    }
    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        // Prefill: split_qkv + rotary_both + write_kv_cache.
        // More work than decode fused variant.
        l4_cost_model::rope_us(profile.num_tokens(), 3072, 64) * 1.5 // prefill: ~1.5× decode rope
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 0,
            threads_per_cta: 0,
        }
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16, Layout::PagedKvBf16]
    }
}

/// FlashInfer attention_standard: prefill attention with explicit Q, K, V.
/// Only matches at seq_len > 1 (prefill).
#[derive(Debug)]
pub struct FlashInferStandardImpl;

impl Implementation for FlashInferStandardImpl {
    fn name(&self) -> &'static str {
        "flashinfer_standard_fa2"
    }
    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.seq_len > 1
    }
    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        let node = &tile_graph.nodes[seed.0 as usize];
        if node.kind != TileKind::Attention {
            return None;
        }
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: node.deps.clone(),
            boundary_outputs: vec![seed],
            layer: node.layer,
        })
    }
    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        l4_cost_model::attention_us(profile.num_tokens())
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 0,
            threads_per_cta: 0,
        }
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        let mut layouts = Vec::with_capacity(m.boundary_inputs.len());
        for _ in &m.boundary_inputs {
            layouts.push(Layout::Any);
        }
        layouts
    }
    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16]
    }
}

// ── CUTLASS GEMM with explicit tile config ──

/// CUTLASS GEMM with a specific tile shape. The solver explores
/// multiple tile configs per GEMM phase and picks the one that
/// minimizes cost at the given `num_tokens`. Not yet backed by
/// actual CUTLASS kernels — analytical cost model only. When
/// CUTLASS dispatch is wired, the interpreter uses `tile_m/tile_n`
/// to select the right template instantiation.
#[derive(Debug)]
pub struct CutlassGemmImpl {
    phase: TileKind,
    tile_m: u32,
    tile_n: u32,
    stages: u32,
    dims: crate::lowering::tile_graph::ModelDims,
    /// Pre-computed "cutlass_{phase}_{M}x{N}_s{stages}" — stable name.
    name: &'static str,
    /// Pre-computed "cutlass_{M}x{N}_s{stages}" — cost table lookup key.
    cost_key: &'static str,
}

impl CutlassGemmImpl {
    pub fn new(
        phase: TileKind,
        tile_m: u32,
        tile_n: u32,
        stages: u32,
        dims: crate::lowering::tile_graph::ModelDims,
    ) -> Self {
        debug_assert!(phase.is_gemm(), "CutlassGemmImpl needs a GEMM tile kind");
        let phase_str = match phase {
            TileKind::GemmQ | TileKind::GemmK | TileKind::GemmV => "qkv",
            TileKind::GemmOProj => "oproj",
            TileKind::GemmGate => "gate",
            TileKind::GemmUp => "up",
            TileKind::GemmDown => "down",
            TileKind::GemmLmHead => "lm_head",
            _ => "unknown",
        };
        let name: &'static str = Box::leak(
            format!("cutlass_{}_{}x{}_s{}", phase_str, tile_m, tile_n, stages).into_boxed_str(),
        );
        let cost_key: &'static str =
            Box::leak(format!("cutlass_{}x{}_s{}", tile_m, tile_n, stages).into_boxed_str());
        Self {
            phase,
            tile_m,
            tile_n,
            stages,
            dims,
            name,
            cost_key,
        }
    }

    /// Create a splitK variant. Same tile config but different cost_key
    /// (e.g. `cutlass_64x64_s4_sk8`) so it looks up splitK-specific costs.
    pub fn new_splitk(
        phase: TileKind,
        tile_m: u32,
        tile_n: u32,
        stages: u32,
        split_k: u32,
        dims: crate::lowering::tile_graph::ModelDims,
    ) -> Self {
        let phase_str = match phase {
            TileKind::GemmQ => "q",
            TileKind::GemmK => "k",
            TileKind::GemmV => "v",
            TileKind::GemmOProj => "oproj",
            TileKind::GemmGate => "gate",
            TileKind::GemmUp => "up",
            TileKind::GemmDown => "down",
            TileKind::GemmLmHead => "lm_head",
            _ => "unknown",
        };
        let name: &'static str = Box::leak(
            format!(
                "cutlass_{}_{}x{}_s{}_sk{}",
                phase_str, tile_m, tile_n, stages, split_k
            )
            .into_boxed_str(),
        );
        let cost_key: &'static str = Box::leak(
            format!("cutlass_{}x{}_s{}_sk{}", tile_m, tile_n, stages, split_k).into_boxed_str(),
        );
        Self {
            phase,
            tile_m,
            tile_n,
            stages,
            dims,
            name,
            cost_key,
        }
    }

    /// Create a TB_K=64 variant with cost_key like `cutlass_64x64_k64_s4`.
    pub fn new_k64(
        phase: TileKind,
        tile_m: u32,
        tile_n: u32,
        stages: u32,
        dims: crate::lowering::tile_graph::ModelDims,
    ) -> Self {
        let phase_str = match phase {
            TileKind::GemmQ => "q",
            TileKind::GemmK => "k",
            TileKind::GemmV => "v",
            TileKind::GemmOProj => "oproj",
            TileKind::GemmGate => "gate",
            TileKind::GemmUp => "up",
            TileKind::GemmDown => "down",
            TileKind::GemmLmHead => "lm_head",
            _ => "unknown",
        };
        let name: &'static str = Box::leak(
            format!(
                "cutlass_{}_{}x{}_k64_s{}",
                phase_str, tile_m, tile_n, stages
            )
            .into_boxed_str(),
        );
        let cost_key: &'static str =
            Box::leak(format!("cutlass_{}x{}_k64_s{}", tile_m, tile_n, stages).into_boxed_str());
        Self {
            phase,
            tile_m,
            tile_n,
            stages,
            dims,
            name,
            cost_key,
        }
    }

    /// Generate all (phase × tile_config) entries for the library.
    pub fn all_configs(
        dims: crate::lowering::tile_graph::ModelDims,
    ) -> Vec<Box<dyn Implementation>> {
        let phases = [
            TileKind::GemmQ,
            TileKind::GemmK,
            TileKind::GemmV,
            TileKind::GemmOProj,
            TileKind::GemmGate,
            TileKind::GemmUp,
            TileKind::GemmDown,
            TileKind::GemmLmHead,
        ];
        // (tile_m, tile_n, stages) — matches the CUDA macro instantiations
        let tiles: &[(u32, u32, u32)] = &[
            (32, 64, 4),
            (32, 64, 3),
            (32, 128, 4),
            (32, 128, 3),
            (32, 256, 3),
            (64, 64, 4),
            (64, 64, 3),
            (64, 128, 4),
            (64, 128, 3),
            (128, 64, 4),
            (128, 64, 3),
            (128, 128, 4),
            (128, 128, 3),
            (128, 256, 3),
            (256, 64, 4),
            (256, 64, 3),
            // stages=2 variants — disabled until CUDA kernel instantiations
            // are added to cutlass_standalone_gemm.cu.
            // (64, 64, 2), (64, 128, 2), (128, 64, 2),
            // (128, 128, 2), (128, 256, 2), (256, 64, 2),
        ];
        // TB_K=64 configs — cost_key like "cutlass_64x64_k64_s4"
        // Note: 128x128_k64_s4, 256x64_k64_s{3,4}, 128x256_k64_s3 exceed
        // sm89 SMEM and are excluded.
        let k64_tiles: &[(u32, u32, u32)] = &[
            (64, 64, 4),
            (64, 64, 3),
            // (64, 64, 2),  // stages=2 disabled — no CUDA kernel
            (64, 128, 4),
            (64, 128, 3),
            // (64, 128, 2),
            (128, 64, 4),
            (128, 64, 3),
            // (128, 64, 2),
            (128, 128, 3),
            // (128, 128, 2),
            // (256, 64, 2),
            (32, 64, 4),
            (32, 128, 4),
        ];
        // SplitK configs — (tile_m, tile_n, stages, split_k_slices)
        let splitk_tiles: &[(u32, u32, u32, u32)] = &[
            (64, 64, 4, 2),
            (64, 64, 4, 4),
            (64, 64, 4, 8),
            (64, 64, 4, 16),
            (128, 128, 3, 2),
            (128, 128, 3, 4),
            (128, 128, 3, 8),
            (64, 128, 4, 2),
            (64, 128, 4, 4),
            (64, 128, 4, 8),
            (64, 128, 4, 16),
            (32, 64, 4, 4),
            (32, 64, 4, 8),
            (32, 64, 4, 16),
            (128, 128, 4, 2),
            (128, 128, 4, 4),
            (128, 128, 4, 8),
            (128, 64, 4, 2),
            (128, 64, 4, 4),
            (128, 64, 4, 8),
            (256, 64, 4, 2),
            (256, 64, 4, 4),
        ];
        // k64 splitK configs
        let k64_splitk_tiles: &[(u32, u32, u32, u32)] = &[
            (64, 64, 4, 2),
            (64, 64, 4, 4),
            (64, 64, 4, 8),
            (128, 128, 3, 2),
            (128, 128, 3, 4),
        ];
        let mut out: Vec<Box<dyn Implementation>> = Vec::new();
        for &phase in &phases {
            for &(m, n, s) in tiles {
                out.push(Box::new(Self::new(phase, m, n, s, dims)));
            }
            for &(m, n, s) in k64_tiles {
                out.push(Box::new(Self::new_k64(phase, m, n, s, dims)));
            }
            for &(m, n, s, sk) in splitk_tiles {
                out.push(Box::new(Self::new_splitk(phase, m, n, s, sk, dims)));
            }
            for &(m, n, s, sk) in k64_splitk_tiles {
                // k64 splitK: cost_key like "cutlass_64x64_k64_s4_sk2"
                let phase_str = match phase {
                    TileKind::GemmQ => "q",
                    TileKind::GemmK => "k",
                    TileKind::GemmV => "v",
                    TileKind::GemmOProj => "oproj",
                    TileKind::GemmGate => "gate",
                    TileKind::GemmUp => "up",
                    TileKind::GemmDown => "down",
                    TileKind::GemmLmHead => "lm_head",
                    _ => "unknown",
                };
                let name: &'static str = Box::leak(
                    format!("cutlass_{}_{}x{}_k64_s{}_sk{}", phase_str, m, n, s, sk)
                        .into_boxed_str(),
                );
                let cost_key: &'static str =
                    Box::leak(format!("cutlass_{}x{}_k64_s{}_sk{}", m, n, s, sk).into_boxed_str());
                out.push(Box::new(Self {
                    phase,
                    tile_m: m,
                    tile_n: n,
                    stages: s,
                    dims,
                    name,
                    cost_key,
                }));
            }
        }
        out
    }
}

/// Compute (N, K) for a GEMM phase from the model dims.
/// - QKV: N = (num_q + 2*num_kv) * head_dim, K = hidden
/// - OProj: N = hidden, K = num_q * head_dim  (input is attention output)
/// - Gate/Up: N = intermediate, K = hidden
/// - Down: N = hidden, K = intermediate
/// - LmHead: N = vocab_size, K = hidden
fn gemm_nk(phase: TileKind, dims: crate::lowering::tile_graph::ModelDims) -> (u32, u32) {
    match phase {
        TileKind::GemmQ | TileKind::GemmK | TileKind::GemmV => (dims.qkv_dim(), dims.hidden_size),
        TileKind::GemmOProj => (dims.hidden_size, dims.num_attention_heads * dims.head_dim),
        TileKind::GemmGate | TileKind::GemmUp => (dims.intermediate_size, dims.hidden_size),
        TileKind::GemmDown => (dims.hidden_size, dims.intermediate_size),
        TileKind::GemmLmHead => (dims.vocab_size, dims.hidden_size),
        _ => (1, 1),
    }
}

impl Implementation for CutlassGemmImpl {
    fn name(&self) -> &'static str {
        self.name
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        let node = &tile_graph.nodes[seed.0 as usize];
        if node.kind != self.phase {
            return None;
        }
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: node.deps.clone(),
            boundary_outputs: vec![seed],
            layer: node.layer,
        })
    }
    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        let m = profile.num_tokens();
        let (n, k) = gemm_nk(self.phase, self.dims);
        l4_cost_model::cutlass_gemm_us_by_key(self.cost_key, m, n, k)
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        // CUTLASS tile config determines shmem: roughly
        // 2 * (tile_m * tile_k + tile_n * tile_k) * 2 bytes per stage.
        let tile_k = 32u32;
        let per_stage = (self.tile_m * tile_k + self.tile_n * tile_k) * 2;
        Resources {
            shmem_bytes: per_stage * 4, // 4 pipeline stages
            regs_per_thread: if self.tile_m >= 128 { 128 } else { 80 },
            threads_per_cta: 256,
        }
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16]
    }
}

// ── CUTLASS GEMV (M=1 specialization) ──

/// CUTLASS SIMT GEMV — handles the `y[N] = W[N,K] @ x[K]` pattern
/// that shows up at BS=1 decode. Tensor-op GEMMs waste 63/64 of their
/// M tile at M=1; this dedicated GEMV avoids that and beats cuBLAS
/// by 1.5-1.9× on the shapes that matter for small LLMs.
///
/// Only matches at M=1. At M>1, the regular CUTLASS GEMMs win.
#[derive(Debug)]
pub struct CutlassGemvImpl {
    phase: TileKind,
    dims: crate::lowering::tile_graph::ModelDims,
}

impl CutlassGemvImpl {
    pub fn new(phase: TileKind, dims: crate::lowering::tile_graph::ModelDims) -> Self {
        debug_assert!(phase.is_gemm(), "CutlassGemvImpl needs a GEMM tile kind");
        Self { phase, dims }
    }

    pub fn all_configs(
        dims: crate::lowering::tile_graph::ModelDims,
    ) -> Vec<Box<dyn Implementation>> {
        // Only phases that don't benefit from fused residual (beta=1 epilogue).
        // OProj and Down use fused-residual GEMMs — replacing them with
        // standalone GEMV + separate residual add adds 2 launches per layer.
        // LmHead runs once per forward pass with no residual, so GEMV wins
        // at BS=1 where the vocab×hidden projection is memory-bound.
        vec![
            Box::new(Self::new(TileKind::GemmQ, dims)),
            Box::new(Self::new(TileKind::GemmK, dims)),
            Box::new(Self::new(TileKind::GemmV, dims)),
            Box::new(Self::new(TileKind::GemmGate, dims)),
            Box::new(Self::new(TileKind::GemmUp, dims)),
            Box::new(Self::new(TileKind::GemmLmHead, dims)),
        ]
    }
}

impl Implementation for CutlassGemvImpl {
    fn name(&self) -> &'static str {
        match self.phase {
            TileKind::GemmQ | TileKind::GemmK | TileKind::GemmV => "cutlass_gemv_qkv",
            TileKind::GemmOProj => "cutlass_gemv_oproj",
            TileKind::GemmGate => "cutlass_gemv_gate",
            TileKind::GemmUp => "cutlass_gemv_up",
            TileKind::GemmDown => "cutlass_gemv_down",
            TileKind::GemmLmHead => "cutlass_gemv_lm_head",
            _ => "cutlass_gemv_unknown",
        }
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    /// CUTLASS GEMV mathematically only produces a matrix-vector
    /// product: y[N] = W[N,K] @ x[K]. It has no notion of "batch"
    /// beyond calling it M times, so using it at M > 1 is a
    /// correctness requirement violation (the launched kernel
    /// only writes one row of output). This is distinct from
    /// "GEMV is slow at M > 1" — it's "GEMV is wrong at M > 1".
    fn workload_constraint(&self) -> crate::lowering::implementation::WorkloadConstraint {
        crate::lowering::implementation::WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }
    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        let node = &tile_graph.nodes[seed.0 as usize];
        if node.kind != self.phase {
            return None;
        }
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: node.deps.clone(),
            boundary_outputs: vec![seed],
            layer: node.layer,
        })
    }
    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        let m = profile.num_tokens();
        let (n, k) = gemm_nk(self.phase, self.dims);
        l4_cost_model::gemv_us(m, n, k)
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 4096,
            regs_per_thread: 64,
            threads_per_cta: 128,
        }
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16]
    }
}

// ── CUTLASS GEMM with fused residual (beta=1 epilogue) ──

/// CUTLASS GEMM claiming `[GEMM + ResidualAdd]` as a 2-tile subgraph,
/// using a beta=1 linear combination epilogue (same trick as cuBLAS).
/// The residual add is free — it happens in the epilogue store.
#[derive(Debug)]
pub struct CutlassGemmWithResidualImpl {
    phase: TileKind,
    tile_m: u32,
    tile_n: u32,
    stages: u32,
    dims: crate::lowering::tile_graph::ModelDims,
    name: &'static str,
    cost_key: &'static str,
}

impl CutlassGemmWithResidualImpl {
    pub fn new(
        phase: TileKind,
        tile_m: u32,
        tile_n: u32,
        stages: u32,
        dims: crate::lowering::tile_graph::ModelDims,
    ) -> Self {
        debug_assert!(
            phase == TileKind::GemmOProj || phase == TileKind::GemmDown,
            "only oproj and down have residual add"
        );
        let phase_str = match phase {
            TileKind::GemmOProj => "oproj",
            TileKind::GemmDown => "down",
            _ => "unknown",
        };
        let name: &'static str = Box::leak(
            format!(
                "cutlass_{}_{}x{}_s{}_res",
                phase_str, tile_m, tile_n, stages
            )
            .into_boxed_str(),
        );
        let cost_key: &'static str =
            Box::leak(format!("cutlass_{}x{}_s{}", tile_m, tile_n, stages).into_boxed_str());
        Self {
            phase,
            tile_m,
            tile_n,
            stages,
            dims,
            name,
            cost_key,
        }
    }

    pub fn new_splitk(
        phase: TileKind,
        tile_m: u32,
        tile_n: u32,
        stages: u32,
        split_k: u32,
        dims: crate::lowering::tile_graph::ModelDims,
    ) -> Self {
        debug_assert!(
            phase == TileKind::GemmOProj || phase == TileKind::GemmDown,
            "only oproj and down have residual add"
        );
        let phase_str = match phase {
            TileKind::GemmOProj => "oproj",
            TileKind::GemmDown => "down",
            _ => "unknown",
        };
        let name: &'static str = Box::leak(
            format!(
                "cutlass_{}_{}x{}_s{}_sk{}_res",
                phase_str, tile_m, tile_n, stages, split_k
            )
            .into_boxed_str(),
        );
        let cost_key: &'static str = Box::leak(
            format!("cutlass_{}x{}_s{}_sk{}", tile_m, tile_n, stages, split_k).into_boxed_str(),
        );
        Self {
            phase,
            tile_m,
            tile_n,
            stages,
            dims,
            name,
            cost_key,
        }
    }

    pub fn new_k64(
        phase: TileKind,
        tile_m: u32,
        tile_n: u32,
        stages: u32,
        dims: crate::lowering::tile_graph::ModelDims,
    ) -> Self {
        debug_assert!(
            phase == TileKind::GemmOProj || phase == TileKind::GemmDown,
            "only oproj and down have residual add"
        );
        let phase_str = match phase {
            TileKind::GemmOProj => "oproj",
            TileKind::GemmDown => "down",
            _ => "unknown",
        };
        let name: &'static str = Box::leak(
            format!(
                "cutlass_{}_{}x{}_k64_s{}_res",
                phase_str, tile_m, tile_n, stages
            )
            .into_boxed_str(),
        );
        let cost_key: &'static str =
            Box::leak(format!("cutlass_{}x{}_k64_s{}", tile_m, tile_n, stages).into_boxed_str());
        Self {
            phase,
            tile_m,
            tile_n,
            stages,
            dims,
            name,
            cost_key,
        }
    }

    /// Generate all (phase × tile_config) entries for residual-fused GEMMs.
    pub fn all_configs(
        dims: crate::lowering::tile_graph::ModelDims,
    ) -> Vec<Box<dyn Implementation>> {
        let phases = [TileKind::GemmOProj, TileKind::GemmDown];
        let tiles: &[(u32, u32, u32)] = &[
            (32, 64, 4),
            (32, 64, 3),
            (32, 128, 4),
            (32, 128, 3),
            (32, 256, 3),
            (64, 64, 4),
            (64, 64, 3),
            (64, 128, 4),
            (64, 128, 3),
            (128, 64, 4),
            (128, 64, 3),
            (128, 128, 4),
            (128, 128, 3),
            (128, 256, 3),
            (256, 64, 4),
            (256, 64, 3),
            // stages=2 variants — disabled until CUDA kernel instantiations
            // are added to cutlass_standalone_gemm.cu.
            // (64, 64, 2), (64, 128, 2), (128, 64, 2),
            // (128, 128, 2), (128, 256, 2), (256, 64, 2),
        ];
        let k64_tiles: &[(u32, u32, u32)] = &[
            (64, 64, 4),
            (64, 64, 3),
            // (64, 64, 2),  // stages=2 disabled — no CUDA kernel
            (64, 128, 4),
            (64, 128, 3),
            // (64, 128, 2),
            (128, 64, 4),
            (128, 64, 3),
            // (128, 64, 2),
            (128, 128, 3),
            // (128, 128, 2),
            // (256, 64, 2),
            (32, 64, 4),
            (32, 128, 4),
        ];
        let splitk_tiles: &[(u32, u32, u32, u32)] = &[
            (64, 64, 4, 2),
            (64, 64, 4, 4),
            (64, 64, 4, 8),
            (64, 64, 4, 16),
            (128, 128, 3, 2),
            (128, 128, 3, 4),
            (128, 128, 3, 8),
            (64, 128, 4, 2),
            (64, 128, 4, 4),
            (64, 128, 4, 8),
            (64, 128, 4, 16),
            (32, 64, 4, 4),
            (32, 64, 4, 8),
            (32, 64, 4, 16),
            (128, 128, 4, 2),
            (128, 128, 4, 4),
            (128, 128, 4, 8),
            (128, 64, 4, 2),
            (128, 64, 4, 4),
            (128, 64, 4, 8),
            (256, 64, 4, 2),
            (256, 64, 4, 4),
        ];
        let k64_splitk_tiles: &[(u32, u32, u32, u32)] = &[
            (64, 64, 4, 2),
            (64, 64, 4, 4),
            (64, 64, 4, 8),
            (128, 128, 3, 2),
            (128, 128, 3, 4),
        ];
        let mut out: Vec<Box<dyn Implementation>> = Vec::new();
        for &phase in &phases {
            for &(m, n, s) in tiles {
                out.push(Box::new(Self::new(phase, m, n, s, dims)));
            }
            for &(m, n, s) in k64_tiles {
                out.push(Box::new(Self::new_k64(phase, m, n, s, dims)));
            }
            for &(m, n, s, sk) in splitk_tiles {
                out.push(Box::new(Self::new_splitk(phase, m, n, s, sk, dims)));
            }
            for &(m, n, s, sk) in k64_splitk_tiles {
                let phase_str = match phase {
                    TileKind::GemmOProj => "oproj",
                    TileKind::GemmDown => "down",
                    _ => "unknown",
                };
                let name: &'static str = Box::leak(
                    format!("cutlass_{}_{}x{}_k64_s{}_sk{}_res", phase_str, m, n, s, sk)
                        .into_boxed_str(),
                );
                let cost_key: &'static str =
                    Box::leak(format!("cutlass_{}x{}_k64_s{}_sk{}", m, n, s, sk).into_boxed_str());
                out.push(Box::new(Self {
                    phase,
                    tile_m: m,
                    tile_n: n,
                    stages: s,
                    dims,
                    name,
                    cost_key,
                }));
            }
        }
        out
    }
}

impl Implementation for CutlassGemmWithResidualImpl {
    fn name(&self) -> &'static str {
        self.name
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        // Same 2-tile claim pattern as CublasGemmExWithResidualImpl.
        let gemm_node = &tile_graph.nodes[seed.0 as usize];
        if gemm_node.kind != self.phase {
            return None;
        }
        let gemm_id = seed;
        let residual = tile_graph.nodes.iter().find(|n| {
            n.kind == TileKind::ResidualAdd
                && n.layer == gemm_node.layer
                && n.deps.contains(&gemm_id)
        })?;
        let residual_id = residual.id;
        let residual_node = &tile_graph.nodes[residual_id.0 as usize];
        let mut boundary_inputs = gemm_node.deps.clone();
        for d in &residual_node.deps {
            if *d != gemm_id && !boundary_inputs.contains(d) {
                boundary_inputs.push(*d);
            }
        }
        Some(MatchInfo {
            claimed_tiles: vec![gemm_id, residual_id],
            boundary_inputs,
            boundary_outputs: vec![residual_id],
            layer: gemm_node.layer,
        })
    }
    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        // Same as standalone CUTLASS GEMM — beta=1 is free in the epilogue.
        let m = profile.num_tokens();
        let (n, k) = gemm_nk(self.phase, self.dims);
        l4_cost_model::cutlass_gemm_us_by_key(self.cost_key, m, n, k)
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        let tile_k = 32u32;
        let per_stage = (self.tile_m * tile_k + self.tile_n * tile_k) * 2;
        Resources {
            shmem_bytes: per_stage * 4,
            regs_per_thread: if self.tile_m >= 128 { 128 } else { 80 },
            threads_per_cta: 256,
        }
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16]
    }
}

// ── DeviceCallable wrapper (sm90+ megakernel-embeddable ops) ──

/// Wraps any existing `Implementation` and makes it `DeviceCallable`
/// with `Mbarrier` handoffs. Used to create megakernel-embeddable
/// variants of standalone ops for the H100 library.
///
/// On sm90+ with `setmaxnreg`, grouping DeviceCallable ops into one
/// persistent kernel has minimal occupancy penalty — each warpgroup
/// picks its own register budget. The cost is the same as standalone
/// (validated assumption; to be confirmed by B2 benchmarks).
#[derive(Debug)]
pub struct DeviceCallableWrapper {
    inner: Box<dyn Implementation>,
    name: &'static str,
}

impl DeviceCallableWrapper {
    pub fn new(inner: Box<dyn Implementation>) -> Self {
        let name: &'static str = Box::leak(format!("dc_{}", inner.name()).into_boxed_str());
        Self { inner, name }
    }
}

impl Implementation for DeviceCallableWrapper {
    fn name(&self) -> &'static str {
        self.name
    }
    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        // Need at least one intra-kernel handoff mechanism:
        // mbarrier (sm90+) or __syncthreads() (all arches).
        let has_intra_kernel_handoff = profile.lowering.mbarrier_handoff_us.is_some()
            || profile.lowering.syncthreads_handoff_us.is_some();
        has_intra_kernel_handoff && self.inner.target_compatible(profile)
    }
    fn workload_constraint(&self) -> WorkloadConstraint {
        self.inner.workload_constraint()
    }
    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        self.inner.matches(tile_graph, seed, profile)
    }
    fn cost_us(&self, m: &MatchInfo, profile: &TargetProfile) -> f64 {
        self.inner.cost_us(m, profile)
    }
    fn resources(&self, m: &MatchInfo) -> Resources {
        self.inner.resources(m)
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::DeviceCallable
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        // Mbarrier (sm90+) or SyncThreads (all arches) for
        // intra-megakernel handoffs. StreamOrder for receiving
        // from a preceding HostCallback.
        &[
            Handoff::Mbarrier,
            Handoff::SyncThreads,
            Handoff::Internal,
            Handoff::StreamOrder,
        ]
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[
            Handoff::Mbarrier,
            Handoff::SyncThreads,
            Handoff::Internal,
            Handoff::StreamOrder,
        ]
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        self.inner.input_layouts(m)
    }
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        self.inner.output_layouts(m)
    }
    fn is_compute_bound(&self) -> bool {
        self.inner.is_compute_bound()
    }
    fn can_share_kernel_with(&self, _other: &dyn Implementation) -> bool {
        true
    }
}

// ── CUTLASS norm+GEMM prologue fusion ──

/// Claims `[RmsNorm + GEMM]` as a 2-tile subgraph. The RMS norm
/// runs in the CUTLASS prologue iterator: each threadblock loads
/// its tile of A from GMEM, applies row-wise RMS norm in shared
/// memory, then feeds the normalized values into the MMA pipeline.
/// Saves one kernel launch + one full GMEM write+read of the
/// normalized activation buffer.
#[derive(Debug)]
pub struct CutlassNormGemmImpl {
    gemm_phase: TileKind,
    tile_m: u32,
    tile_n: u32,
}

impl CutlassNormGemmImpl {
    pub fn new(gemm_phase: TileKind, tile_m: u32, tile_n: u32) -> Self {
        Self {
            gemm_phase,
            tile_m,
            tile_n,
        }
    }
}

impl Implementation for CutlassNormGemmImpl {
    fn name(&self) -> &'static str {
        match self.gemm_phase {
            TileKind::GemmQ | TileKind::GemmK | TileKind::GemmV => "cutlass_norm_qkv_128",
            TileKind::GemmGate => "cutlass_norm_gate_128",
            TileKind::GemmUp => "cutlass_norm_up_128",
            _ => "cutlass_norm_unknown",
        }
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(
        &self,
        tile_graph: &TileGraph,
        seed: TileId,
        _profile: &TargetProfile,
    ) -> Option<MatchInfo> {
        // Seed on the GEMM tile so this competes directly with
        // standalone CutlassGemmImpl at the same branching point.
        // Walk backward to find the feeding RmsNorm.
        let gemm = &tile_graph.nodes[seed.0 as usize];
        if gemm.kind != self.gemm_phase {
            return None;
        }
        let norm = gemm.deps.iter().find_map(|dep| {
            let n = &tile_graph.nodes[dep.0 as usize];
            if n.kind == TileKind::RmsNorm && n.layer == gemm.layer {
                Some(n)
            } else {
                None
            }
        })?;
        Some(MatchInfo {
            claimed_tiles: vec![norm.id, seed],
            boundary_inputs: norm.deps.clone(),
            boundary_outputs: vec![seed],
            layer: gemm.layer,
        })
    }
    fn cost_us(&self, _m: &MatchInfo, profile: &TargetProfile) -> f64 {
        // The norm is "free" — it runs in the prologue while waiting
        // for the B-operand load. Cost ≈ CUTLASS GEMM cost alone
        // (maybe 2-5% overhead for the norm compute in the prologue).
        let m = profile.num_tokens();
        let (n, k) = match self.gemm_phase {
            TileKind::GemmQ | TileKind::GemmK | TileKind::GemmV => (3072, 2048),
            TileKind::GemmGate | TileKind::GemmUp => (8192, 2048),
            _ => (1, 1),
        };
        l4_cost_model::cutlass_gemm_us(m, n, k, self.tile_m, self.tile_n, 4) * 1.03 // 3% overhead for prologue norm
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        let tile_k = 32u32;
        let per_stage = (self.tile_m * tile_k + self.tile_n * tile_k) * 2;
        Resources {
            // Extra shmem for norm: hidden_dim * 4 bytes (f32 accumulator)
            shmem_bytes: per_stage * 4 + 2048 * 4,
            regs_per_thread: 140, // slightly more than plain GEMM
            threads_per_cta: 256,
        }
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder, Handoff::StreamEvent]
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }
    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16]
    }
}
