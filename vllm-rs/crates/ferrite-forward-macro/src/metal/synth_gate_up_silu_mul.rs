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

use quote::quote;

use crate::classified::{OpKind, Program};
use crate::codegen::split_base_layer;
use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::{
    consumes_tile, default_required_weights, weight_storage_of, CostCtx, Handoff, Implementation,
    LaunchKind, Layout, MatchInfo, OpInstance, OpcodeShape, Resources, SlotMap, WeightAccessor,
    WorkloadConstraint,
};
use crate::quantization::StorageFormat;
use crate::target::{Backend, TargetProfile};

use ::ferrite_fusion_synth::fuse_pass::{
    synthesize_gate_up_silu_mul_large_chunk, ChunkConstants, SynthesisBackend,
};

#[derive(Debug)]
pub struct MetalSynthGateUpSiluMulImpl {
    pub act_tag: &'static str,
    pub scale_tag: &'static str,
    pub group_size: u32,
    pub bits: u32,
}

impl MetalSynthGateUpSiluMulImpl {
    pub fn bf16_gs64() -> Self {
        Self { act_tag: "bfloat", scale_tag: "half", group_size: 64, bits: 4 }
    }
}

impl Implementation for MetalSynthGateUpSiluMulImpl {
    fn name(&self) -> &'static str {
        "metal_synth_gate_up_silu_mul"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.backend == Backend::Metal
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange { min: 8, max: u32::MAX }
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

        let silu_node = fuf.nodes.iter().find(|n| {
            n.op == OpKind::Silu && consumes_tile(n, seed)
        })?;
        let silu_tile = silu_node.id;

        let mul_node = fuf.nodes.iter().find(|n| {
            n.op == OpKind::Mul && consumes_tile(n, silu_tile)
        })?;
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
        let intermediate = ctx.bounds.get("intermediate_size").copied().unwrap_or(0) as u32;

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

        let synth_name = format!(
            "synth_gate_up_silu_mul_large_{}_{}_gs{}",
            self.act_tag, self.scale_tag, self.group_size,
        );
        if let Some(cost) = ctx.profile.cost_us_for(&synth_name, num_tokens, hidden, 0) {
            return cost;
        }

        if hidden == 0 || intermediate == 0 { return 1.0e9; }
        let bw = ctx.profile.memory_bandwidth_gbps;
        if bw <= 0.0 { return 1.0e9; }

        let mf = num_tokens.max(1) as f64;
        let im = intermediate as f64;
        let act_bytes = 2.0_f64;

        let silu_bytes = 3.0 * mf * im * act_bytes;
        let silu_us    = silu_bytes / 1e9 / bw * 1e6;

        silu_us * 0.90
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources { shmem_bytes: 8 * 9 * 2 * 3, regs_per_thread: 64, threads_per_cta: 32 }
    }

    fn launch_kind(&self) -> LaunchKind { LaunchKind::HostCallback }

    fn supported_input_handoffs(&self) -> &[Handoff] { &[Handoff::StreamOrder] }
    fn supported_output_handoffs(&self) -> &[Handoff] { &[Handoff::StreamOrder] }

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
                ("out_slot",    syn::parse_quote!(u32)),
                ("layer",       syn::parse_quote!(u32)),
                (
                    "gate_wt_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                (
                    "up_wt_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                ("group_size", syn::parse_quote!(u32)),
                ("bits",       syn::parse_quote!(u32)),
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
    ) -> Option<Vec<OpInstance>> {
        let mut silu_tile_id = None;
        let mut mul_tile     = None;
        for &t in &m.claimed_tiles {
            match fuf.get(t).op {
                OpKind::Silu => silu_tile_id = Some(t),
                OpKind::Mul  => mul_tile = Some(t),
                _ => {}
            }
        }
        let silu_tile_id = silu_tile_id?;
        let mul_tile: TileId = mul_tile?;

        let gate_tile = fuf.get(silu_tile_id).inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. }
                if fuf.get(*id).op == OpKind::Gemm => Some(*id),
            _ => None,
        })?;
        let up_tile = fuf.get(mul_tile).inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. }
                if *id != silu_tile_id && fuf.get(*id).op == OpKind::Gemm => Some(*id),
            _ => None,
        })?;
        let x_norm_slot = slots.of(m.boundary_inputs[0], 0);
        let out_slot    = slots.of(mul_tile, 0);

        let layer = m.claimed_tiles.iter().find_map(|&t| {
            fuf.get(t).inputs.iter().find_map(|i| {
                if let FufInput::Weight { index: Some(l), .. } = i { Some(*l as u32) }
                else { None }
            })
        })?;

        let acc_for = |tile: TileId| -> Option<WeightAccessor> {
            default_required_weights(&[tile], fuf, program).into_iter().next()
        };
        let gate_acc = acc_for(gate_tile)?;
        let up_acc   = acc_for(up_tile)?;

        let to_wt = |acc: &WeightAccessor| -> proc_macro2::TokenStream {
            let (base, _) = split_base_layer(&acc.name.to_string());
            let ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
            quote! { Weights::#ident }
        };
        let gate_wt = to_wt(&gate_acc);
        let up_wt   = to_wt(&up_acc);

        let (gs, bits) = match weight_storage_of(fuf.get(gate_tile)) {
            Some(StorageFormat::Affine { group_size, bits }) => (*group_size, *bits),
            _ => return None,
        };

        let hidden       = *bounds.get("hidden_size").unwrap_or(&0) as u32;
        let intermediate = *bounds.get("intermediate_size").unwrap_or(&0) as u32;
        let head_dim     = *bounds.get("head_dim").unwrap_or(&128) as u32;
        let eps          = 1e-5_f32;
        let consts = ChunkConstants {
            hidden, intermediate, head_dim,
            num_q_heads: 0, num_kv_heads: 0,
            rot_dim: 0, block_size: 0,
            m: 0, group_size: gs,
            rms_norm_eps: eps,
            // MLP gate/up have no per-row linear bias on any model
            // ferrite-metal supports; only the QKV `bias_add` is wired
            // through the synth path. Stays `false` here.
            has_linear_bias: false,
        };
        let kernel = synthesize_gate_up_silu_mul_large_chunk(
            SynthesisBackend::Metal, self.act_tag, self.scale_tag, &consts,
        );
        let symbol_lit = syn::LitStr::new(&kernel.symbol, proc_macro2::Span::call_site());

        let _ = bits;
        let bits_lit = self.bits;
        let gs_lit   = gs;
        let layer_lit = layer;

        Some(vec![OpInstance::new(
            syn::Ident::new("SynthGateUpSiluMul", proc_macro2::Span::call_site()),
            vec![
                quote! { #x_norm_slot },
                quote! { #out_slot },
                quote! { #layer_lit },
                gate_wt,
                up_wt,
                quote! { #gs_lit },
                quote! { #bits_lit },
                quote! { #symbol_lit },
            ],
        )])
    }
}
