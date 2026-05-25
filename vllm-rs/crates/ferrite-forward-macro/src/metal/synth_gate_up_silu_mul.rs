// SPDX-License-Identifier: Apache-2.0
//! Solver-driven `SynthGateUpSiluMul` tile-GEMM megakernel Impl.
//!
//! Claims the 4-tile chain `(Gemm_gate + Gemm_up + Silu + Mul)` at the FUF
//! level for large-M prefill buckets (M ≥ 8). Uses simdgroup_matrix 8×8
//! tiles (BM=BN=BK=32, TM=TN=2, 128 threads) instead of the GEMV-based
//! `mk_qmv_fast` used by decode-path kernels.
//!
//! The AddRmsNorm step is NOT claimed here — it runs first via
//! `MetalFusedAddRmsNormImpl` and writes `x_norm` to device memory.
//! This impl fuses gate + up projections + SiluMul into one kernel.

use std::collections::BTreeMap;

use crate::classified::{OpKind, Program};
use crate::codegen::split_base_layer;
use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::{
    CostCtx, Handoff, Implementation, LaunchKind, Layout, MatchInfo, OpcodeShape, Resources,
    SlotMap, WeightAccessor, WorkloadConstraint, consumes_tile, default_required_weights,
    weight_storage_of,
};
use crate::quantization::StorageFormat;
use crate::target::{Backend, TargetProfile};

use ::ferrite_fusion_synth::fuse_pass::{
    ChunkConstants, SynthesisBackend, synthesize_gate_up_silu_mul_large_chunk,
};

#[derive(Debug)]
pub struct MetalSynthGateUpSiluMulImpl {
    pub act_tag: &'static str,
    pub scale_tag: &'static str,
    pub group_size: u32,
    pub bits: u32,
}

impl MetalSynthGateUpSiluMulImpl {
    /// Llama-3.x / Qwen2.5 / SmolLM mlx-community 4bit: F16 scales.
    pub fn bf16_gs64() -> Self {
        Self {
            act_tag: "bfloat",
            scale_tag: "half",
            group_size: 64,
            bits: 4,
        }
    }
    /// Qwen3 family mlx-community 4bit: BF16 scales.
    pub fn bf16_gs64_s_bf16() -> Self {
        Self {
            act_tag: "bfloat",
            scale_tag: "bfloat",
            group_size: 64,
            bits: 4,
        }
    }
}

/// True iff the model's HF architecture string is one of the Qwen3
/// family (`Qwen3ForCausalLM` / `Qwen3MoeForCausalLM`). Their
/// mlx-community 4bit checkpoints ship BF16 scales/biases (probed
/// across cached HF snapshots) while Llama-3.x / Qwen2.5 / SmolLM
/// ship F16 — drives the synth Impl's `applies_to` gate.
pub(crate) fn is_qwen3_arch(model: &crate::config::ModelParams) -> bool {
    model
        .architectures
        .iter()
        .any(|a| matches!(a.as_str(), "Qwen3ForCausalLM" | "Qwen3MoeForCausalLM"))
}

impl Implementation for MetalSynthGateUpSiluMulImpl {
    fn name(&self) -> &'static str {
        "metal_synth_gate_up_silu_mul"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.backend == Backend::Metal
    }

    fn applies_to(&self, ctx: &crate::impl_lib::MatchContext) -> bool {
        // Gate by scale dtype: only the synth variant whose
        // `scale_tag` matches the model's on-disk scale convention
        // claims a tile. The other variant returns false so the
        // solver doesn't see a duplicate match.
        let is_qwen3 = is_qwen3_arch(ctx.model);
        matches!(
            (is_qwen3, self.scale_tag),
            (true, "bfloat") | (false, "half")
        )
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange {
            min: 8,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let gate_node = fuf.get(seed);
        if gate_node.op != OpKind::Gemm {
            return None;
        }
        if !matches!(
            weight_storage_of(gate_node),
            Some(StorageFormat::Affine { group_size, bits })
                if *group_size == self.group_size && *bits == self.bits
        ) {
            return None;
        }

        let silu_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Silu && consumes_tile(n, seed))?;
        let silu_tile = silu_node.id;

        let mul_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Mul && consumes_tile(n, silu_tile))?;
        let mul_tile = mul_node.id;

        let up_tile = mul_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != silu_tile => Some(*id),
            _ => None,
        })?;
        let up_node = fuf.get(up_tile);
        if up_node.op != OpKind::Gemm {
            return None;
        }
        if !matches!(
            weight_storage_of(up_node),
            Some(StorageFormat::Affine { group_size, bits })
                if *group_size == self.group_size && *bits == self.bits
        ) {
            return None;
        }

        let gate_in = gate_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } => Some(*id),
            _ => None,
        })?;
        let up_in = up_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } => Some(*id),
            _ => None,
        })?;
        if gate_in != up_in {
            return None;
        }

        let mut claimed = vec![seed, up_tile, silu_tile, mul_tile];
        claimed.sort();

        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![gate_in],
            boundary_outputs: vec![mul_tile],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let num_tokens = ctx.num_tokens() as u32;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as u32;

        // SAFETY GATE — this fused-large-M kernel uses 32×32 simdgroup
        // tiles per threadgroup without M-direction weight reuse, so its
        // effective bandwidth is roughly `peak_bw / (M/32)` at large M.
        // At M=1024 we measure ~8× slower than per-op AffineQmm + Silu +
        // Mul. Until the kernel is redesigned with M-blocked tiling that
        // amortizes weight reads across multiple M-tiles per TG, gate it
        // off above the M ranges where it's been validated. Returning a
        // very large finite cost (not INFINITY — the solver rejects
        // non-finite costs as fatal, see `solver.rs:477`) lets the
        // solver fall through to the unfused per-op chain at prefill.
        if num_tokens > 64 {
            return 1.0e15;
        }

        // Cost requires measured data — the analytic fallback used to
        // claim `silu_us * 0.90`, a hardware-agnostic lie that beats
        // the unfused AffineQmm + Silu + Mul chain on every platform
        // regardless of whether the kernel is actually fast there
        // (62318fb52 exposed the bug on M1 Max once the fused-MLP
        // path stopped winning). Refuse to claim a low cost without
        // a measurement; populate the CSV via
        // `cargo run -p ferrite-metal-cost-sweep --release`.
        let synth_name = format!(
            "synth_gate_up_silu_mul_large_{}_{}_gs{}",
            self.act_tag, self.scale_tag, self.group_size,
        );
        if let Some(cost) = ctx.profile.cost_us_for(&synth_name, num_tokens, hidden, 0) {
            return cost;
        }
        1.0e15
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 8 * 9 * 2 * 3,
            regs_per_thread: 64,
            threads_per_cta: 32,
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

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "SynthGateUpSiluMul",
            vec![
                ("x_norm_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("group_size", syn::parse_quote!(u32)),
                ("bits", syn::parse_quote!(u32)),
                ("kernel_symbol", syn::parse_quote!(&'static str)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<ferrite_forward::Instruction>> {
        let mut silu_tile_id = None;
        let mut mul_tile = None;
        for &t in &m.claimed_tiles {
            match fuf.get(t).op {
                OpKind::Silu => silu_tile_id = Some(t),
                OpKind::Mul => mul_tile = Some(t),
                _ => {}
            }
        }
        let silu_tile_id = silu_tile_id?;
        let mul_tile: TileId = mul_tile?;

        let gate_tile = fuf.get(silu_tile_id).inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if fuf.get(*id).op == OpKind::Gemm => Some(*id),
            _ => None,
        })?;
        let up_tile = fuf.get(mul_tile).inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != silu_tile_id && fuf.get(*id).op == OpKind::Gemm => {
                Some(*id)
            }
            _ => None,
        })?;
        let x_norm_slot = slots.of(m.boundary_inputs[0], 0);
        let out_slot = slots.of(mul_tile, 0);

        let layer = m.claimed_tiles.iter().find_map(|&t| {
            fuf.get(t).inputs.iter().find_map(|i| {
                if let FufInput::Weight { index: Some(l), .. } = i {
                    Some(*l as u32)
                } else {
                    None
                }
            })
        })?;

        let acc_for = |tile: TileId| -> Option<WeightAccessor> {
            default_required_weights(&[tile], fuf, program)
                .into_iter()
                .next()
        };
        let gate_acc = acc_for(gate_tile)?;
        let up_acc = acc_for(up_tile)?;

        let to_base = |acc: &WeightAccessor| -> syn::Ident {
            let (base, _) = split_base_layer(&acc.name.to_string());
            syn::Ident::new(&base, proc_macro2::Span::call_site())
        };
        let gate_base = to_base(&gate_acc);
        let up_base = to_base(&up_acc);

        let (gs, bits) = match weight_storage_of(fuf.get(gate_tile)) {
            Some(StorageFormat::Affine { group_size, bits }) => (*group_size, *bits),
            _ => return None,
        };

        let hidden = *bounds.get("hidden_size").unwrap_or(&0) as u32;
        let intermediate = *bounds.get("intermediate_size").unwrap_or(&0) as u32;
        let head_dim = *bounds.get("head_dim").unwrap_or(&128) as u32;
        let eps = 1e-5_f32;
        let consts = ChunkConstants {
            hidden,
            intermediate,
            head_dim,
            num_q_heads: 0,
            num_kv_heads: 0,
            rot_dim: 0,
            block_size: 0,
            m: 0,
            group_size: gs,
            rms_norm_eps: eps,
            // MLP gate/up have no per-row linear bias on any model
            // ferrite-metal supports; only the QKV `bias_add` is wired
            // through the synth path. Stays `false` here.
            has_linear_bias: false,
        };
        let kernel = synthesize_gate_up_silu_mul_large_chunk(
            SynthesisBackend::Metal,
            self.act_tag,
            self.scale_tag,
            &consts,
        );
        let _ = bits;
        // Gate/up LinearLayers flow through `required_weights()`;
        // codegen assigns sub-slots 0/1.
        let _ = (gate_base, up_base);
        let kernel_symbol: &'static str = Box::leak(kernel.symbol.into_boxed_str());
        #[cfg(feature = "metal")]
        return Some(vec![ferrite_forward::Instruction::SynthGateUpSiluMul(
            x_norm_slot,
            out_slot,
            layer,
            gs,
            self.bits,
            kernel_symbol,
        )]);
        #[cfg(not(feature = "metal"))]
        {
            let _ = (x_norm_slot, out_slot, layer, gs, kernel_symbol);
            return None;
        }
    }
}
