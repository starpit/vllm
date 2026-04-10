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
    /// Construct the L4 sm_89 starter library: cuBLAS, vllm-rs
    /// fused, FlashInfer standalone, plus the small passthroughs
    /// for QkvSplit / KvCacheWrite / ResidualAdd. CP5-D extends.
    pub fn l4_sm89_starter() -> Self {
        let entries: Vec<Box<dyn Implementation>> = vec![
            // ── TK fused MLP block (7-tile claim) ──
            Box::new(TkFusedMlpBlockImpl),
            // ── cuBLAS GEMM with fused residual (claims GEMM + ResidualAdd
            //    as a two-tile subgraph; uses cublasGemmEx beta=1.0 to
            //    fold the residual add into the GEMM epilogue for free).
            //    Listed BEFORE the standalone CublasGemmExImpl entries
            //    so the solver's cheapest-first tie-break (when both
            //    cost 145 µs at the per-call level) prefers the fused
            //    variant. The fused variant saves a downstream
            //    ResidualAdd cost so its full-path cost is strictly
            //    lower; this ordering just makes the solver find it
            //    on the first branch instead of after backtracking.
            Box::new(CublasGemmExWithResidualImpl::new(TileKind::GemmOProj)),
            Box::new(CublasGemmExWithResidualImpl::new(TileKind::GemmDown)),
            // ── cuBLAS GEMMs (one entry per phase for clean cost lookup) ──
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
            Box::new(VllmRsFusedQkvRopeCacheImpl),
            Box::new(VllmRsRotaryEmbeddingImpl),
            Box::new(VllmRsSiluAndMulFusedImpl),
            // ── FlashInfer standalone ──
            Box::new(FlashInferStandaloneImpl),
            // ── Free / cheap passthroughs ──
            // Note: QkvSplitFreeImpl is intentionally omitted. The fused
            // VllmRsFusedQkvRopeCacheImpl is the only cover for QkvSplit
            // (and folds in Rope + KvCacheWrite as a side effect), which
            // forces the solver to use it. The cheapest-first per-seed
            // greedy with the loose remainder bound otherwise picks
            // free-split for the QkvSplit tile and never recovers.
            Box::new(KvCacheWriteImpl),
            Box::new(ResidualAddImpl),
        ];
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

// ── Cost calibration constants (from CP4 microbench) ──
//
// All numbers are wall-clock microseconds for ONE invocation on the
// Llama-1B seq=1024 production shape on L4 sm_89.

mod l4_llama_1b_seq1024_costs {
    /// cuBLAS per-call costs for each GEMM phase. From the CP4
    /// gemm-only microbench (36.6 ms total / 80 calls = 458 µs avg,
    /// distributed unevenly per shape — bigger N gets more time).
    pub const CUBLAS_QKV_US: f64 = 220.0; // M=1024 K=2048 N=3072
    pub const CUBLAS_OPROJ_US: f64 = 145.0; // M=1024 K=2048 N=2048
    pub const CUBLAS_GATE_US: f64 = 530.0; // M=1024 K=2048 N=8192
    pub const CUBLAS_UP_US: f64 = 530.0; // M=1024 K=2048 N=8192
    pub const CUBLAS_DOWN_US: f64 = 540.0; // M=1024 K=8192 N=2048

    /// vllm-rs fused-op per-call costs (negligible relative to GEMMs;
    /// the natural microbench measured 1.2 ms total for all 16 layers
    /// of norm + rope + silu_mul, distributed across 5 calls per
    /// layer = ~15 µs per call avg).
    pub const RMS_NORM_US: f64 = 15.0;
    pub const ROTARY_EMBEDDING_US: f64 = 20.0;
    pub const SILU_AND_MUL_US: f64 = 12.0;
    pub const KV_CACHE_WRITE_US: f64 = 10.0;
    pub const RESIDUAL_ADD_US: f64 = 8.0;
    pub const QKV_SPLIT_US: f64 = 0.0; // free — the buffer is already laid out

    /// FlashInfer attention per-layer cost from the megakernel's
    /// `fanin rope+at` clock (~10 ms / 16 layers = ~625 µs/layer).
    pub const FLASHINFER_ATTENTION_US: f64 = 625.0;
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

    fn cost_us(&self, _m: &MatchInfo, _profile: &TargetProfile) -> f64 {
        use l4_llama_1b_seq1024_costs::*;
        match self.phase {
            TileKind::GemmQkv => CUBLAS_QKV_US,
            TileKind::GemmOProj => CUBLAS_OPROJ_US,
            TileKind::GemmGate => CUBLAS_GATE_US,
            TileKind::GemmUp => CUBLAS_UP_US,
            TileKind::GemmDown => CUBLAS_DOWN_US,
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

    fn cost_us(&self, _m: &MatchInfo, _profile: &TargetProfile) -> f64 {
        // Same as the plain cuBLAS GEMM — beta=1 doesn't change wall
        // clock vs beta=0 in cuBLAS's mainloop.
        use l4_llama_1b_seq1024_costs::*;
        match self.phase {
            TileKind::GemmOProj => CUBLAS_OPROJ_US,
            TileKind::GemmDown => CUBLAS_DOWN_US,
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
    fn cost_us(&self, _m: &MatchInfo, _profile: &TargetProfile) -> f64 {
        l4_llama_1b_seq1024_costs::RMS_NORM_US
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
    fn cost_us(&self, _m: &MatchInfo, _profile: &TargetProfile) -> f64 {
        l4_llama_1b_seq1024_costs::SILU_AND_MUL_US
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
    fn cost_us(&self, _m: &MatchInfo, _profile: &TargetProfile) -> f64 {
        l4_llama_1b_seq1024_costs::FLASHINFER_ATTENTION_US
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
    fn cost_us(&self, _m: &MatchInfo, _profile: &TargetProfile) -> f64 {
        l4_llama_1b_seq1024_costs::KV_CACHE_WRITE_US
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
    fn cost_us(&self, _m: &MatchInfo, _profile: &TargetProfile) -> f64 {
        l4_llama_1b_seq1024_costs::RESIDUAL_ADD_US
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
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
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
    fn cost_us(&self, _m: &MatchInfo, _profile: &TargetProfile) -> f64 {
        l4_llama_1b_seq1024_costs::ROTARY_EMBEDDING_US
            + l4_llama_1b_seq1024_costs::KV_CACHE_WRITE_US
            + l4_llama_1b_seq1024_costs::QKV_SPLIT_US
            - 5.0
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

/// TK fused MLP block: claims the 7-tile subgraph
/// `(RmsNorm_mlp → GemmGate → GemmUp → GateUpConcat → SiluMul → GemmDown → ResidualAdd_mlp)`
/// as a single kernel launch via the grid-dispatched `cp5_fused_mlp`.
#[derive(Debug)]
pub struct TkFusedMlpBlockImpl;

impl Implementation for TkFusedMlpBlockImpl {
    fn name(&self) -> &'static str {
        "tk_fused_mlp_block"
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
        let norm = &tile_graph.nodes[seed.0 as usize];
        if norm.kind != TileKind::RmsNorm {
            return None;
        }
        let consumers: Vec<_> = tile_graph
            .nodes
            .iter()
            .filter(|n| n.deps.contains(&seed) && n.layer == norm.layer)
            .collect();
        let gate = consumers.iter().find(|n| n.kind == TileKind::GemmGate)?;
        let up = consumers.iter().find(|n| n.kind == TileKind::GemmUp)?;
        let concat = tile_graph.nodes.iter().find(|n| {
            n.kind == TileKind::GateUpConcat
                && n.layer == norm.layer
                && n.deps.contains(&gate.id)
                && n.deps.contains(&up.id)
        })?;
        let silu = tile_graph.nodes.iter().find(|n| {
            n.kind == TileKind::SiluMul && n.layer == norm.layer && n.deps.contains(&concat.id)
        })?;
        let down = tile_graph.nodes.iter().find(|n| {
            n.kind == TileKind::GemmDown && n.layer == norm.layer && n.deps.contains(&silu.id)
        })?;
        let residual = tile_graph.nodes.iter().find(|n| {
            n.kind == TileKind::ResidualAdd && n.layer == norm.layer && n.deps.contains(&down.id)
        })?;
        Some(MatchInfo {
            claimed_tiles: vec![
                seed,
                gate.id,
                up.id,
                concat.id,
                silu.id,
                down.id,
                residual.id,
            ],
            boundary_inputs: norm.deps.clone(),
            boundary_outputs: vec![residual.id],
            layer: norm.layer,
        })
    }
    fn cost_us(&self, _m: &MatchInfo, _profile: &TargetProfile) -> f64 {
        // Sum of individual costs minus GMEM savings from fusion.
        // Placeholder — calibrate via microbench.
        l4_llama_1b_seq1024_costs::RMS_NORM_US
            + l4_llama_1b_seq1024_costs::CUBLAS_GATE_US
            + l4_llama_1b_seq1024_costs::CUBLAS_UP_US
            + l4_llama_1b_seq1024_costs::SILU_AND_MUL_US
            + l4_llama_1b_seq1024_costs::CUBLAS_DOWN_US
            + l4_llama_1b_seq1024_costs::RESIDUAL_ADD_US
            - 200.0
    }
    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 32768,
            regs_per_thread: 128,
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
