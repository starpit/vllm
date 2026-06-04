// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal fused kernel implementation adapters.
//!
//! Wraps Metal fused kernels (Add+RMSNorm, Gate-Up-SiLU-Mul) to satisfy ferrite's Implementation trait.

use std::collections::BTreeMap;

use crate::classified::{OpKind, Program};
use crate::codegen::split_base_layer;
use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::{
    CostCtx, FusedAddRmsNormImpl, FusedGateUpGeluMulImpl, FusedGateUpSiluMulImpl, Handoff,
    Implementation, LaunchKind, Layout, MatchInfo, OpcodeShape, Resources, SlotMap, WeightAccessor,
    WorkloadConstraint, consumes_tile, default_required_weights, first_tile_input,
    first_weight_ref, gemm_nk_from_fuf, weight_storage_of,
};
use crate::metal::affine_qmm::{affine_qmm_opcode_shape, affine_qmm_vector_limit};
use crate::metal::nvfp4_qmm::nvfp4_qmm_opcode_shape;
use crate::quantization::StorageFormat;
use crate::target::{Backend, TargetProfile};

/// Adapter that wraps Metal Fused Add+RMSNorm to satisfy ferrite's Implementation trait.
///
/// This fusion eliminates one memory round-trip by computing the residual add and
/// normalization in a single pass: output = rmsnorm(input + residual, weight, eps)
#[derive(Debug)]
pub struct MetalFusedAddRmsNormImpl {
    /// Kernel name for cost table lookup
    kernel_name: &'static str,
    /// Data type (fp16 or bf16)
    dtype: &'static str,
}

impl MetalFusedAddRmsNormImpl {
    pub fn new_fp16() -> Self {
        Self {
            kernel_name: "fused_add_rmsnorm_f16",
            dtype: "fp16",
        }
    }

    pub fn new_bf16() -> Self {
        Self {
            kernel_name: "fused_add_rmsnorm_bf16",
            dtype: "bf16",
        }
    }

    /// Analytical cost model for fused Add+RMSNorm (memory-bound operation).
    /// Cost = (bytes_read + bytes_written) / bandwidth
    /// Reads: input + residual + weight
    /// Writes: output (+ optional residual_out)
    fn analytical_cost_us(
        &self,
        m: u32,
        n: u32,
        bandwidth_gbps: f64,
        has_residual_out: bool,
    ) -> f64 {
        let bytes_per_element = 2.0; // fp16/bf16
        let total_elements = (m as f64) * (n as f64);

        // Reads: input [M,N] + residual [M,N] + weight [N]
        let bytes_read = total_elements * bytes_per_element * 2.0 + (n as f64) * bytes_per_element;

        // Writes: output [M,N] + optional residual_out [M,N]
        let write_multiplier = if has_residual_out { 2.0 } else { 1.0 };
        let bytes_written = total_elements * bytes_per_element * write_multiplier;

        let total_bytes = bytes_read + bytes_written;
        let total_gb = total_bytes / 1e9;
        let time_seconds = total_gb / bandwidth_gbps;
        time_seconds * 1e6 // convert to microseconds
    }
}

impl Implementation for MetalFusedAddRmsNormImpl {
    fn name(&self) -> &'static str {
        match self.dtype {
            "fp16" => "metal_fused_add_rmsnorm_f16",
            "bf16" => "metal_fused_add_rmsnorm_bf16",
            _ => "metal_fused_add_rmsnorm",
        }
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        // Only compatible with Metal targets
        profile.backend == Backend::Metal
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::Any
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        // Defer the (Add, RmsNorm) claim shape to the canonical CUDA
        // impl — same tile pattern, same boundary inputs, same
        // both-tiles-aliased output semantics. Only the kernel cost
        // model and the bound `target_compatible` differ between
        // backends.
        FusedAddRmsNormImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, match_info: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Get the RMSNorm tile (second in claimed_tiles)
        let rmsnorm_tile = match_info.claimed_tiles[1];
        let node = ctx.fuf.get(rmsnorm_tile);

        // Get shape: RMSNorm operates on [M, N]
        let shape = &node.outputs[0];
        let dims = ctx.eval_shape(shape);

        if let Some(dims) = dims
            && dims.len() >= 2
        {
            let m = dims[0] as u32;
            let n = dims[1] as u32;

            // Check if Add has multiple consumers (indicates residual_out needed)
            let add_tile = match_info.claimed_tiles[0];
            let has_residual_out = ctx
                .fuf
                .nodes
                .iter()
                .filter(|node| {
                    node.inputs.iter().any(|input| {
                        if let crate::fuf::FufInput::Tile { id, slot: _ } = input {
                            id == &add_tile
                        } else {
                            false
                        }
                    })
                })
                .count()
                > 1;

            // Try empirical cost first
            let k = if has_residual_out { 1 } else { 0 };
            if let Some(cost) = ctx.profile.cost_us_for(self.kernel_name, m, n, k) {
                return cost;
            }

            // Fall back to analytical model
            return self.analytical_cost_us(
                m,
                n,
                ctx.profile.memory_bandwidth_gbps,
                has_residual_out,
            );
        }

        // Fallback: conservative estimate
        150.0
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0, // Metal uses threadgroup memory
            regs_per_thread: 32,
            threads_per_cta: 256,
        }
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder]
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder]
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::Any; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::Any; m.boundary_outputs.len()]
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        default_required_weights(claimed_tiles, fuf, program)
    }

    // `fused_add_rms_norm_inplace` mutates both the residual buffer
    // (Add output → updated residual) and the delta buffer (RmsNorm
    // output, normed-in-place); both outputs are TensorView aliases
    // of the upstream Add inputs. Mirror the CUDA contract so the
    // codegen drop pass preserves both upstreams correctly.
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        FusedAddRmsNormImpl.output_alias(claimed_tiles, fuf)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        FusedAddRmsNormImpl.opcode_shape()
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<ferrite_forward::Instruction>> {
        FusedAddRmsNormImpl.fan_out(m, fuf, program, bounds, slots)
    }
}

/// Adapter that wraps Metal Fused Gate-Up-SiLU-Mul (SwiGLU) to satisfy ferrite's Implementation trait.
///
/// This fusion computes: output = silu(gate) * up
/// Where: silu(x) = x * sigmoid(x)
///
/// Eliminates intermediate memory traffic by fusing the activation and multiplication.
#[derive(Debug)]
pub struct MetalFusedGateUpSiluMulImpl {
    /// Kernel name for cost table lookup
    kernel_name: &'static str,
    /// Data type (fp16 or bf16)
    dtype: &'static str,
    /// Whether this is GELU variant (for Gemma models)
    is_gelu: bool,
}

impl MetalFusedGateUpSiluMulImpl {
    pub fn new_fp16() -> Self {
        Self {
            kernel_name: "fused_gate_up_silu_mul_f16",
            dtype: "fp16",
            is_gelu: false,
        }
    }

    pub fn new_bf16() -> Self {
        Self {
            kernel_name: "fused_gate_up_silu_mul_bf16",
            dtype: "bf16",
            is_gelu: false,
        }
    }

    pub fn new_gelu_fp16() -> Self {
        Self {
            kernel_name: "fused_gate_up_gelu_mul_f16",
            dtype: "fp16",
            is_gelu: true,
        }
    }

    /// Decomposed cost for Affine 4-bit Gate-Up-SiLU-Mul. `fan_out`
    /// emits 3 separate Instructions for this storage (gate AffineQmm,
    /// up AffineQmm, SiluMul) — there is no fused affine kernel —
    /// so the cost must equal `cost(qmm gate) + cost(qmm up) + cost(silu_and_mul)`,
    /// not the analytical "single fused bandwidth pass" estimate that
    /// applies to the Dense path. Without this branch the impl claims
    /// the 4-tile region at a fictitiously low cost and starves
    /// SynthMlpPreDown at decode (project_metal_mlppredown_decode_cost).
    fn affine_decomposed_cost_us(
        &self,
        gate_node: &crate::fuf::FufNode,
        m: u32,
        n: u32,
        _group_size: u32,
        _bits: u32,
        ctx: &CostCtx,
    ) -> f64 {
        // K from gate's activation input.
        let k = gate_node
            .inputs
            .iter()
            .find_map(|i| match i {
                crate::fuf::FufInput::Tile { id, slot } => ctx
                    .fuf
                    .get(*id)
                    .outputs
                    .get(*slot as usize)
                    .and_then(|s| ctx.eval_shape(s))
                    .and_then(|v| v.last().copied())
                    .map(|x| x as u32),
                _ => None,
            })
            .unwrap_or(2048);

        let qmm = crate::metal::affine_qmm::empirical_cost_us(self.dtype, m, n, k, gate_node, ctx);
        let one_qmm = qmm.unwrap_or_else(|| {
            // Analytical fallback (compute-bound roofline for the gemm).
            let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
            flops / (ctx.profile.peak_tflops_fp16 * 1e12) * 1e6
        });

        // SiluMul: bandwidth-bound, reads gate+up [M,N] and writes [M,N].
        let act_bytes = 2.0_f64;
        let silu_bytes = 3.0 * (m as f64) * (n as f64) * act_bytes;
        let silu_us = silu_bytes / 1e9 / ctx.profile.memory_bandwidth_gbps * 1e6;

        2.0 * one_qmm + silu_us
    }

    /// Analytical cost model for fused Gate-Up-SiLU-Mul (memory-bound operation).
    /// Cost = (bytes_read + bytes_written) / bandwidth
    /// Reads: gate [M,N] + up [M,N]
    /// Writes: output [M,N]
    fn analytical_cost_us(&self, m: u32, n: u32, bandwidth_gbps: f64) -> f64 {
        let bytes_per_element = 2.0; // fp16/bf16
        let total_elements = (m as f64) * (n as f64);

        // Reads: gate [M,N] + up [M,N]
        let bytes_read = total_elements * bytes_per_element * 2.0;

        // Writes: output [M,N]
        let bytes_written = total_elements * bytes_per_element;

        let total_bytes = bytes_read + bytes_written;
        let total_gb = total_bytes / 1e9;
        let time_seconds = total_gb / bandwidth_gbps;
        time_seconds * 1e6 // convert to microseconds
    }
}

impl Implementation for MetalFusedGateUpSiluMulImpl {
    fn name(&self) -> &'static str {
        if self.is_gelu {
            match self.dtype {
                "fp16" => "metal_fused_gate_up_gelu_mul_f16",
                "bf16" => "metal_fused_gate_up_gelu_mul_bf16",
                _ => "metal_fused_gate_up_gelu_mul",
            }
        } else {
            match self.dtype {
                "fp16" => "metal_fused_gate_up_silu_mul_f16",
                "bf16" => "metal_fused_gate_up_silu_mul_bf16",
                _ => "metal_fused_gate_up_silu_mul",
            }
        }
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        // Only compatible with Metal targets
        profile.backend == Backend::Metal
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::Any
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        // GELU variant has no affine decomposition path today —
        // delegate to the canonical Dense-only CUDA matcher.
        if self.is_gelu {
            return FusedGateUpGeluMulImpl.matches(fuf, seed, profile);
        }
        // SiLU variant: same 4-tile `(Gemm, Gemm, Silu, Mul)` claim
        // as the CUDA matcher, but accept Dense (existing fused
        // kernel) AND MLX-affine (decomposed q-MLP — fan_out emits
        // `AffineQmm` + `AffineQmm` + `SiluMul`). The CUDA matcher's
        // storage gate at `impl_lib.rs:2971` is restricted to Dense,
        // so the walk has to live here for the Affine path.
        let gate_gemm = fuf.get(seed);
        if gate_gemm.op != OpKind::Gemm {
            return None;
        }
        let gate_storage = weight_storage_of(gate_gemm);
        let gate_is_dense = matches!(gate_storage, Some(StorageFormat::Dense));
        let gate_is_affine = matches!(gate_storage, Some(StorageFormat::Affine { .. }));
        let gate_is_nvfp4 = matches!(gate_storage, Some(StorageFormat::Nvfp4 { .. }));
        if !gate_is_dense && !gate_is_affine && !gate_is_nvfp4 {
            return None;
        }

        let silu_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Silu && consumes_tile(n, seed))?;
        let silu_id = silu_node.id;
        let mul_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Mul && consumes_tile(n, silu_id))?;
        let mul_id = mul_node.id;

        let up_gemm_id = mul_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != silu_id => Some(*id),
            _ => None,
        })?;
        let up_gemm = fuf.get(up_gemm_id);
        if up_gemm.op != OpKind::Gemm {
            return None;
        }
        // Storage must agree between gate & up — mixed quant on a
        // single MLP isn't a real checkpoint shape, and fanout would
        // pick a single decomposition mode.
        let up_storage = weight_storage_of(up_gemm);
        let up_is_dense = matches!(up_storage, Some(StorageFormat::Dense));
        let up_is_affine = matches!(up_storage, Some(StorageFormat::Affine { .. }));
        let up_is_nvfp4 = matches!(up_storage, Some(StorageFormat::Nvfp4 { .. }));
        if gate_is_dense != up_is_dense
            || gate_is_affine != up_is_affine
            || gate_is_nvfp4 != up_is_nvfp4
        {
            return None;
        }
        if first_tile_input(gate_gemm)? != first_tile_input(up_gemm)? {
            return None;
        }

        let mut claimed = [seed, up_gemm_id, silu_id, mul_id];
        claimed.sort();
        let claimed = claimed.to_vec();

        let activation_tile = first_tile_input(gate_gemm)?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation_tile],
            boundary_outputs: vec![mul_id],
        })
    }

    fn cost_us(&self, match_info: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Get the Mul tile (last in claimed_tiles)
        let mul_tile = *match_info.claimed_tiles.last().unwrap();
        let node = ctx.fuf.get(mul_tile);

        // Get shape: Mul operates on [M, N]
        let shape = &node.outputs[0];
        let dims = ctx.eval_shape(shape);

        if let Some(dims) = dims
            && dims.len() >= 2
        {
            let m = dims[0] as u32;
            let n = dims[1] as u32;

            // Storage-aware cost. For Dense weights, a real fused
            // kernel runs (silu+mul+matmul stays in registers across
            // the bandwidth boundary), so the analytical bandwidth
            // model applies. For Affine 4-bit weights, `fan_out`
            // emits 3 SEPARATE Instructions (AffineQmm gate + AffineQmm
            // up + SiluMul) — there is no fused affine variant — so
            // the cost must sum the unfused components or it tells
            // the solver a lie. The lie was: at M=1 affine, this
            // returned ~20 µs (analytical fused) + startup that
            // amortizes to ~314 µs/call, beating SynthMlpPreDown's
            // real CSV cost of 670 µs and starving the synth at
            // decode (project_metal_mlppredown_decode_cost).
            let gate_tile = match_info.claimed_tiles[0];
            let gate_node = ctx.fuf.get(gate_tile);
            let storage = weight_storage_of(gate_node);
            if let Some(StorageFormat::Affine { group_size, bits }) = storage {
                return self.affine_decomposed_cost_us(gate_node, m, n, *group_size, *bits, ctx);
            }
            // NVFP4 decomposes the same way (Nvfp4Qmm gate + up + SiluMul)
            // and has no competing fused kernel, so an analytical roofline
            // (no nvfp4 cost-sweep rows yet) is sufficient for the solver.
            if matches!(storage, Some(StorageFormat::Nvfp4 { .. })) {
                return self.analytical_cost_us(m, n, ctx.profile.memory_bandwidth_gbps);
            }

            // Dense path: existing fused-kernel cost model.
            // Try empirical cost first
            if let Some(cost) = ctx.profile.cost_us_for(self.kernel_name, m, n, 0) {
                return cost;
            }

            // Fall back to analytical model
            return self.analytical_cost_us(m, n, ctx.profile.memory_bandwidth_gbps);
        }

        // Fallback: conservative estimate
        120.0
    }

    fn startup_us(&self, _match_info: &MatchInfo, ctx: &CostCtx) -> f64 {
        // The fused MLP requires `[gate|up]` packed at load time —
        // `required_weights` declares one accessor whose source list
        // covers both `gate_proj.weight` and `up_proj.weight`, and
        // `LinearLayer::load_dense_concat_packed` does mmap → heap
        // Vec → arena MTLBuffer (two synchronous CPU memcpies of
        // `2 × intermediate × hidden × elem_bytes`). On Apple Silicon
        // the loader path runs at ~5 GB/s effective (allocator
        // overhead dominates over peak memcpy), enough to be visible
        // in init engine time on Llama-3.2-3B (~1s across 28 layers).
        //
        // Returning a positive cost here lets the solver weigh the
        // packed fused impl against the unpacked alternative
        // (separate gate gemm + up gemm + silu_and_mul) — the
        // unpacked composition pays no startup but spends extra
        // per-call dispatches. At metal's interactive default
        // `expected_calls_per_load = 64`, the crossover lands where
        // it should: chat workloads prefer unpacked; long batches
        // prefer fused.
        //
        // Shape comes from the model's HF-config bounds rather than
        // walking tile inputs because the per-tile pack cost is
        // shape-independent of `num_tokens` (it's a load-time CPU
        // memcpy, not a per-call dispatch). Per-tile — the layered
        // loader instantiates this once per layer, and the solver
        // sums across the per-layer tile assignments.
        let intermediate = ctx.bounds.get("intermediate_size").copied().unwrap_or(0);
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0);
        if intermediate == 0 || hidden == 0 {
            return 0.0;
        }
        let elem_bytes: u64 = match self.dtype {
            "fp16" | "bf16" => 2,
            _ => 4,
        };
        let packed_bytes = 2 * intermediate * hidden * elem_bytes;
        const PACK_BANDWIDTH_GBPS: f64 = 5.0;
        let bytes_per_us = PACK_BANDWIDTH_GBPS * 1_000.0;
        packed_bytes as f64 / bytes_per_us
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0, // Metal uses threadgroup memory
            regs_per_thread: 32,
            threads_per_cta: 256,
        }
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder]
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder]
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::Any; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::Any; m.boundary_outputs.len()]
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        if self.is_gelu {
            return FusedGateUpGeluMulImpl.required_weights(claimed_tiles, fuf, program);
        }
        // SiLU: storage-polymorphic accessor shape.
        //   * Dense → one fused accessor whose source aggregates
        //     gate_proj + up_proj weight refs (the loader concats
        //     them into a single `[gate|up]` LinearLayer at load
        //     time — the existing CUDA path).
        //   * Affine → two single-source accessors. Codegen's
        //     `linear_field_load` resolver sees each as
        //     `FieldLoad::LinearAffine` and emits a separate
        //     `LinearLayer::load_affine_quant` per source so the
        //     decomposed `AffineQmm` instructions emitted in
        //     `fan_out` can each address their own Linear.
        if storage_of_first_gemm(claimed_tiles, fuf).is_decomposed_quant() {
            return default_required_weights(claimed_tiles, fuf, program);
        }
        FusedGateUpSiluMulImpl.required_weights(claimed_tiles, fuf, program)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        if self.is_gelu {
            FusedGateUpGeluMulImpl.opcode_shape()
        } else {
            FusedGateUpSiluMulImpl.opcode_shape()
        }
    }

    fn extra_opcode_shapes(&self) -> Vec<OpcodeShape> {
        if self.is_gelu {
            // No affine GELU decomposition path today.
            return Vec::new();
        }
        // SiLU: the Affine path fans out into `AffineQmm` ×2 +
        // `SiluMul`, and the NVFP4 path into `Nvfp4Qmm` ×2 + `SiluMul`.
        // Register all decomposition shapes here so the per-bucket
        // static-slice validator typechecks them regardless of whether
        // `MetalAffineQmmImpl` / `MetalNvfp4QmmImpl` claimed any
        // standalone Gemm in this model. The `*Qmm` shapes must match
        // those impls' `opcode_shape` bit-exactly — they come from the
        // same helpers to enforce that.
        vec![
            affine_qmm_opcode_shape(),
            nvfp4_qmm_opcode_shape(),
            silu_mul_opcode_shape(),
        ]
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<ferrite_forward::Instruction>> {
        if self.is_gelu {
            return FusedGateUpGeluMulImpl.fan_out(m, fuf, program, bounds, slots);
        }
        // Discriminate Dense (single fused emit) vs Affine
        // (decomposed `AffineQmm` + `AffineQmm` + `SiluMul`) on the
        // gate Gemm's weight storage. `matches()` already enforces
        // gate/up agreement, so inspecting one suffices.
        if !storage_of_first_gemm(&m.claimed_tiles, fuf).is_decomposed_quant() {
            return FusedGateUpSiluMulImpl.fan_out(m, fuf, program, bounds, slots);
        }
        Some(quant_decomposed_fan_out(m, fuf, program, bounds, slots))
    }
}

/// Compact descriptor of the gate-Gemm's storage format used by the
/// SiLU variant's `required_weights` and `fan_out` to discriminate
/// Dense (fused emit) from MLX-affine (decomposed emit).
#[derive(Clone, Copy, Debug)]
enum GateStorage {
    Dense,
    Affine,
    Nvfp4,
    Other,
}

impl GateStorage {
    fn is_affine(self) -> bool {
        matches!(self, GateStorage::Affine)
    }
    fn is_nvfp4(self) -> bool {
        matches!(self, GateStorage::Nvfp4)
    }
    /// Both 4-bit dequant-on-read formats decompose the fused MLP into
    /// `(qmm gate, qmm up, SiluMul)` — they share `required_weights` /
    /// `fan_out` handling, differing only in the emitted opcode.
    fn is_decomposed_quant(self) -> bool {
        self.is_affine() || self.is_nvfp4()
    }
}

fn storage_of_first_gemm(claimed_tiles: &[TileId], fuf: &Fuf) -> GateStorage {
    for &t in claimed_tiles {
        let n = fuf.get(t);
        if n.op == OpKind::Gemm {
            return match weight_storage_of(n) {
                Some(StorageFormat::Dense) => GateStorage::Dense,
                Some(StorageFormat::Affine { .. }) => GateStorage::Affine,
                Some(StorageFormat::Nvfp4 { .. }) => GateStorage::Nvfp4,
                _ => GateStorage::Other,
            };
        }
    }
    GateStorage::Other
}

/// `Instruction::SiluMul` variant shape: `(gate_slot, up_slot,
/// out_slot, width)`. The three slots feed the `silu_mul_<dtype>`
/// kernel's three buffer bindings; `width` (the per-row element count
/// = the claim's gate/up Gemm N) sizes the dispatch — carried on the
/// instruction rather than derived from `W::INTERMEDIATE_SIZE` so
/// SwiGLU blocks of any width (dense MLP, MoE shared expert) lower
/// correctly in the same model.
fn silu_mul_opcode_shape() -> OpcodeShape {
    OpcodeShape::new(
        "SiluMul",
        vec![
            ("gate_slot", syn::parse_quote!(u32)),
            ("up_slot", syn::parse_quote!(u32)),
            ("out_slot", syn::parse_quote!(u32)),
            ("width", syn::parse_quote!(u32)),
        ],
    )
}

/// Build the three-instruction decomposition (`qmm` gate, `qmm` up,
/// SiluMul) used when the fused gate-up SiLU MLP claim's Gemms have a
/// dequant-on-read 4-bit storage (MLX-affine → `AffineQmm`, NVFP4 →
/// `Nvfp4Qmm`; chosen per-Gemm by [`decomposed_qmm_inst`]). Mirrors
/// plan P12 branch (i): the macro composes existing primitives instead
/// of a hand-rolled fused-quant-MLP kernel (`feedback_no_handcoded_fusion`).
fn quant_decomposed_fan_out(
    m: &MatchInfo,
    fuf: &Fuf,
    program: &Program,
    bounds: &BTreeMap<String, u64>,
    slots: &SlotMap,
) -> Vec<ferrite_forward::Instruction> {
    let silu_id = *m
        .claimed_tiles
        .iter()
        .find(|t| fuf.get(**t).op == OpKind::Silu)
        .expect("MetalFusedGateUpSiluMul(Affine): claim contains Silu");
    let mul_id = *m
        .claimed_tiles
        .iter()
        .find(|t| fuf.get(**t).op == OpKind::Mul)
        .expect("MetalFusedGateUpSiluMul(Affine): claim contains Mul");
    let gate_id = match fuf.get(silu_id).inputs.first() {
        Some(FufInput::Tile { id, .. }) => *id,
        other => panic!(
            "MetalFusedGateUpSiluMul(Affine): Silu's first input must be a Tile (got {other:?})"
        ),
    };
    let up_id = fuf
        .get(mul_id)
        .inputs
        .iter()
        .find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != silu_id => Some(*id),
            _ => None,
        })
        .expect("MetalFusedGateUpSiluMul(Affine): Mul's non-Silu Tile input must be the up Gemm");

    let gate_node = fuf.get(gate_id);
    let up_node = fuf.get(up_id);
    let (in_id, in_slot) = match gate_node.inputs.first() {
        Some(FufInput::Tile { id, slot }) => (*id, *slot),
        other => panic!(
            "MetalFusedGateUpSiluMul(Affine): gate Gemm's first input must be a Tile (got {other:?})"
        ),
    };
    let in_slot_idx = slots.of(in_id, in_slot);
    let gate_out_idx = slots.of(gate_id, 0);
    let up_out_idx = slots.of(up_id, 0);
    let final_out_idx = slots.of(mul_id, 0);

    let gate_wref = first_weight_ref(gate_node)
        .expect("MetalFusedGateUpSiluMul(Affine): gate Gemm has no weight ref");
    let up_wref = first_weight_ref(up_node)
        .expect("MetalFusedGateUpSiluMul(Affine): up Gemm has no weight ref");

    // Resolve accessor names by matching `source_weights` back to the
    // per-Gemm weight ref. `required_weights` for Affine returns the
    // default (per-weight) accessor list, so each accessor has one
    // source — matching is just `source_weights == [wref]`.
    let accessors = default_required_weights(&m.claimed_tiles, fuf, program);
    let gate_acc = accessors
        .iter()
        .find(|a| a.source_weights == vec![gate_wref])
        .expect("MetalFusedGateUpSiluMul(Affine): gate accessor missing from required_weights");
    let up_acc = accessors
        .iter()
        .find(|a| a.source_weights == vec![up_wref])
        .expect("MetalFusedGateUpSiluMul(Affine): up accessor missing from required_weights");

    let (gate_base, gate_layer) = split_base_layer(&gate_acc.name.to_string());
    let gate_layer_lit = gate_layer.unwrap_or(0) as u32;
    let gate_base_ident = syn::Ident::new(&gate_base, proc_macro2::Span::call_site());
    let (up_base, up_layer) = split_base_layer(&up_acc.name.to_string());
    let up_layer_lit = up_layer.unwrap_or(0) as u32;
    let up_base_ident = syn::Ident::new(&up_base, proc_macro2::Span::call_site());

    let (gate_n, gate_k) = gemm_nk_from_fuf(fuf, gate_node, bounds)
        .expect("MetalFusedGateUpSiluMul(Affine): gate (N, K) must resolve from FUF + bounds");
    let (up_n, up_k) = gemm_nk_from_fuf(fuf, up_node, bounds)
        .expect("MetalFusedGateUpSiluMul(Affine): up (N, K) must resolve from FUF + bounds");
    // Weight info (LinearLayer for gate/up) flows through
    // `required_weights()` — codegen attaches slots in declaration
    // order, so each `*Qmm` row gets its own `(tape_index, op_idx,
    // slot=0)` arm against the right `Weights::<gate|up>_proj` base.
    // The opcode (AffineQmm vs Nvfp4Qmm) is chosen per-Gemm from its
    // weight storage by `decomposed_qmm_inst`.
    let _ = (gate_base_ident, up_base_ident);
    let gate_inst = decomposed_qmm_inst(
        gate_node,
        in_slot_idx,
        gate_out_idx,
        gate_layer_lit,
        gate_n,
        gate_k,
    );
    let up_inst = decomposed_qmm_inst(up_node, in_slot_idx, up_out_idx, up_layer_lit, up_n, up_k);
    // SwiGLU invariant: gate and up project to the same width; that
    // width sizes the elementwise SiluMul tail.
    assert_eq!(
        gate_n, up_n,
        "MetalFusedGateUpSiluMul(Affine): gate N ({gate_n}) != up N ({up_n})"
    );
    // Catch the `intermediate_size: 0` config-bug class at macro
    // expansion (a 0-width SiluMul is a silent no-op kernel — the
    // runtime lowering also asserts, but failing the build beats
    // failing the first forward).
    assert!(
        gate_n > 0,
        "MetalFusedGateUpSiluMul(Affine): SwiGLU width is 0 — \
         `intermediate_size` (or the all-MoE shared-expert derivation) \
         resolved to 0 for a body that has a dense SwiGLU MLP"
    );
    let silu_mul_inst =
        ferrite_forward::Instruction::SiluMul(gate_out_idx, up_out_idx, final_out_idx, gate_n);
    vec![gate_inst, up_inst, silu_mul_inst]
}

/// Emit the per-Gemm decode instruction for the decomposed quant MLP,
/// choosing the opcode from the Gemm's weight storage: MLX-affine →
/// `AffineQmm` (carries its `group_size`/`bits`), NVFP4 → `Nvfp4Qmm`
/// (group_size from storage, bits always 4). The `vector_limit`
/// (decode↔prefill boundary) comes from the same `(K, N)` helper both
/// formats share.
fn decomposed_qmm_inst(
    node: &crate::fuf::FufNode,
    in_slot: u32,
    out_slot: u32,
    layer: u32,
    n: u32,
    k: u32,
) -> ferrite_forward::Instruction {
    let vl = affine_qmm_vector_limit(k, n);
    match weight_storage_of(node) {
        Some(StorageFormat::Affine { group_size, bits }) => {
            ferrite_forward::Instruction::AffineQmm(
                in_slot,
                out_slot,
                layer,
                n,
                k,
                *group_size,
                *bits,
                vl,
            )
        }
        Some(StorageFormat::Nvfp4 { group_size }) => ferrite_forward::Instruction::Nvfp4Qmm(
            in_slot,
            out_slot,
            layer,
            n,
            k,
            *group_size,
            4,
            vl,
        ),
        other => {
            panic!("decomposed_qmm_inst: gate/up Gemm storage isn't a decomposed quant ({other:?})")
        }
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::target::from_metal_profile;

    #[test]
    fn metal_fused_add_rmsnorm_only_compatible_with_metal_targets() {
        let metal_impl = MetalFusedAddRmsNormImpl::new_fp16();

        // Metal target - should be compatible
        let metal_profile = from_metal_profile(&ferrite_metal_targets::M1_8CORE);
        assert!(metal_impl.target_compatible(&metal_profile));

        // CUDA target - should NOT be compatible
        let cuda_profile = crate::target::from_profile_def(&ferrite_cuda_targets::L4_SM89);
        assert!(!metal_impl.target_compatible(&cuda_profile));
    }

    #[test]
    fn metal_fused_add_rmsnorm_cost_accounts_for_residual_out() {
        let impl_fp16 = MetalFusedAddRmsNormImpl::new_fp16();

        // Without residual_out
        let cost_no_res = impl_fp16.analytical_cost_us(1024, 4096, 68.25, false);

        // With residual_out (extra write)
        let cost_with_res = impl_fp16.analytical_cost_us(1024, 4096, 68.25, true);

        // Cost with residual_out should be higher
        assert!(
            cost_with_res > cost_no_res,
            "Expected cost_with_res ({}) > cost_no_res ({})",
            cost_with_res,
            cost_no_res
        );

        // Should be roughly 1.33× (4 reads + 2 writes vs 4 reads + 1 write)
        let ratio = cost_with_res / cost_no_res;
        assert!(
            (ratio - 1.2).abs() < 0.2,
            "Expected ratio ~1.2-1.4, got {}",
            ratio
        );
    }

    #[test]
    fn metal_fused_gate_up_silu_mul_only_compatible_with_metal_targets() {
        let metal_impl = MetalFusedGateUpSiluMulImpl::new_fp16();

        // Metal target - should be compatible
        let metal_profile = from_metal_profile(&ferrite_metal_targets::M1_8CORE);
        assert!(metal_impl.target_compatible(&metal_profile));

        // CUDA target - should NOT be compatible
        let cuda_profile = crate::target::from_profile_def(&ferrite_cuda_targets::L4_SM89);
        assert!(!metal_impl.target_compatible(&cuda_profile));
    }

    #[test]
    fn metal_fused_gate_up_silu_mul_analytical_cost() {
        let impl_fp16 = MetalFusedGateUpSiluMulImpl::new_fp16();

        // Cost should scale with M*N (memory-bound)
        let cost_small = impl_fp16.analytical_cost_us(512, 2048, 68.25);
        let cost_large = impl_fp16.analytical_cost_us(1024, 4096, 68.25);

        // Large should be ~8× more expensive (2× M, 2× N = 4× elements, 2× for gate+up)
        let ratio = cost_large / cost_small;
        assert!(
            (ratio - 8.0).abs() < 1.0,
            "Expected ratio ~8.0, got {}",
            ratio
        );
    }
}
