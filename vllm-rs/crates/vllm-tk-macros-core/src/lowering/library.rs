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
    Handoff, ImplId, Implementation, LaunchKind, Layout, MatchInfo, Resources,
};
use crate::lowering::tile_graph::{TileGraph, TileId, TileKind};
use crate::target_profile::TargetProfile;

/// Curated set of [`Implementation`] entries available to the
/// lowering solver.
pub struct ImplementationLibrary {
    pub entries: Vec<Box<dyn Implementation>>,
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
        Self::sm89_starter_common(dims)
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
            // ── cuBLAS GEMM with fused bias epilogue ──
            // For biased models (Qwen2, Qwen2.5, ...). Registered
            // before the standalone CublasGemmExImpl variants so the
            // solver prefers the fused {GemmQkv + BiasAdd} cover over
            // the split cover when both are feasible.
            Box::new(CublasGemmExWithBiasImpl::new(TileKind::GemmQkv)),
            // ── CUTLASS norm+GEMM prologue fusion (D-3) ──
            Box::new(CutlassNormGemmImpl::new(TileKind::GemmQkv, 128, 128)),
            Box::new(CutlassNormGemmImpl::new(TileKind::GemmGate, 128, 128)),
            Box::new(CutlassNormGemmImpl::new(TileKind::GemmUp, 128, 128)),
            Box::new(CublasGemmExImpl::new(TileKind::GemmQkv)),
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
        ImplementationLibrary { entries }
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
        let t = table();
        match dim {
            2048 => t.elementwise_hd.lookup(m),
            8192 => t.elementwise_id.lookup(m),
            3072 => t.elementwise_qkv.lookup(m),
            512 => t.elementwise_hd.lookup(m) * 0.25,
            _ => t.elementwise_hd.lookup(m) * (dim as f64 / 2048.0),
        }
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
            TileKind::GemmQkv => "cublas_gemm_ex_qkv",
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
            TileKind::GemmQkv => l4_cost_model::gemm_us(m, 3072, 2048),
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
    fn cost_us(&self, _m: &MatchInfo, _profile: &TargetProfile) -> f64 {
        l4_llama_1b_seq1024_costs::ROTARY_EMBEDDING_US
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
        l4_cost_model::elementwise_us(profile.num_tokens(), 8192) // intermediate_dim
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
/// Registered per phase — for Qwen2 only `GemmQkv` is needed today,
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
            TileKind::GemmQkv => "cublas_gemm_ex_qkv_with_bias",
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
            TileKind::GemmQkv => l4_cost_model::gemm_us(m, 3072, 2048),
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
        l4_cost_model::elementwise_us(profile.num_tokens(), 3072) // qkv_dim
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
        l4_cost_model::elementwise_us(profile.num_tokens(), 3072) * 1.5
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
            TileKind::GemmQkv => "qkv",
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

    /// Generate all (phase × tile_config) entries for the library.
    pub fn all_configs(
        dims: crate::lowering::tile_graph::ModelDims,
    ) -> Vec<Box<dyn Implementation>> {
        let phases = [
            TileKind::GemmQkv,
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
            // (256, 128, 3) — exceeds sm89 SMEM
        ];
        let mut out: Vec<Box<dyn Implementation>> = Vec::new();
        for &phase in &phases {
            for &(m, n, s) in tiles {
                out.push(Box::new(Self::new(phase, m, n, s, dims)));
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
        TileKind::GemmQkv => (dims.qkv_dim(), dims.hidden_size),
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
            Box::new(Self::new(TileKind::GemmQkv, dims)),
            Box::new(Self::new(TileKind::GemmGate, dims)),
            Box::new(Self::new(TileKind::GemmUp, dims)),
            Box::new(Self::new(TileKind::GemmLmHead, dims)),
        ]
    }
}

impl Implementation for CutlassGemvImpl {
    fn name(&self) -> &'static str {
        match self.phase {
            TileKind::GemmQkv => "cutlass_gemv_qkv",
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
            // (256, 128, 3) — exceeds sm89 SMEM
        ];
        let mut out: Vec<Box<dyn Implementation>> = Vec::new();
        for &phase in &phases {
            for &(m, n, s) in tiles {
                out.push(Box::new(Self::new(phase, m, n, s, dims)));
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
            TileKind::GemmQkv => "cutlass_norm_qkv_128",
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
            TileKind::GemmQkv => (3072, 2048),
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
