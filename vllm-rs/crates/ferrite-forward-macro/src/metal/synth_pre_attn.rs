// SPDX-License-Identifier: Apache-2.0
//! Solver-driven `SynthPreAttn` megakernel Impl.
//!
//! Claims the 6-tile chain `(Add → RmsNorm → Gemm_Q + Gemm_K + Gemm_V →
//! RopeAppend)` (or 5-tile `(RmsNorm → 3×Gemm → RopeAppend)` for the
//! layer-0 init variant) at the FUF level. When the solver DP picks
//! this Impl over the unfused alternative
//! `(FusedAddRmsNormImpl + 3×MetalAffineQmmImpl + RopeAppendImpl)`,
//! `fan_out` emits a single `Instruction::SynthPreAttn` so the runtime
//! dispatches one synthesized kernel instead of five separate ones.
//!
//! Cost source: `synth_pre_attn_<act>_<scale>_gs<gs>` row in the
//! `ferrite-metal-targets` CSV when present; falls back to the
//! component cost sum (the same thing the unfused alternative would
//! score, so the solver picks either way on tie — slightly biased
//! toward `Synth` to break the tie since it saves dispatch overhead).
//!
//! Replaces `interpreter_codegen::apply_synth_replacement{,_init}` —
//! both of those ran AFTER the solver and rewrote the instruction
//! stream unconditionally, bypassing cost-driven selection. With the
//! Impl in the solver DP, the pick is cost-driven per bucket.

use std::collections::BTreeMap;

use crate::classified::{OpKind, Program};
use crate::codegen::split_base_layer;
use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::{
    consumes_tile, default_required_weights, first_tile_input, weight_storage_of, CostCtx,
    Handoff, Implementation, LaunchKind, Layout, MatchInfo, OpInstance, OpcodeShape, Resources,
    SlotMap, WeightAccessor, WorkloadConstraint,
};
use crate::quantization::StorageFormat;
use crate::target::{Backend, TargetProfile};

use quote::quote;

/// SynthPreAttn megakernel claim. `init=false` matches the 6-tile
/// chain headed by `Add` (layers ≥1); `init=true` matches the 5-tile
/// chain headed by `RmsNorm` (layer 0, no preceding residual add).
#[derive(Debug)]
pub struct MetalSynthPreAttnImpl {
    /// Activation dtype tag used in the synth kernel symbol name.
    /// `"bfloat"` or `"half"`; chosen at construction to match the
    /// canonical's `W::METAL_DTYPE`.
    pub act_tag: &'static str,
    /// Scale dtype tag (always `"half"` for the affine-int4 path
    /// today; RMSNorm gains + group scales are F16 on disk).
    pub scale_tag: &'static str,
    /// Affine quant group size — must match what the Affine-storage
    /// weights ship with. Llama-3.2 4bit ships gs=64.
    pub group_size: u32,
    /// Affine quant bits — only `4` is wired in the int4 production
    /// path.
    pub bits: u32,
    pub init: bool,
}

impl MetalSynthPreAttnImpl {
    pub fn bf16_gs64() -> Self {
        Self {
            act_tag: "bfloat",
            scale_tag: "half",
            group_size: 64,
            bits: 4,
            init: false,
        }
    }
    pub fn bf16_gs64_init() -> Self {
        Self {
            act_tag: "bfloat",
            scale_tag: "half",
            group_size: 64,
            bits: 4,
            init: true,
        }
    }
}

impl Implementation for MetalSynthPreAttnImpl {
    fn name(&self) -> &'static str {
        if self.init {
            "metal_synth_pre_attn_init"
        } else {
            "metal_synth_pre_attn"
        }
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.backend == Backend::Metal
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::Any
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Layer ≥1: seed must be the residual `Add`. Layer 0 init
        // variant: seed must be the standalone `RmsNorm` whose input
        // is *not* an `Add` (otherwise the non-init impl claims it).
        let seed_node = fuf.get(seed);
        let (add_tile, rmsnorm_tile) = if self.init {
            if seed_node.op != OpKind::RmsNorm {
                return None;
            }
            // Reject if seed's first input is an Add (the non-init
            // impl owns that chain).
            let (in_tile, _slot) = first_tile_input(seed_node)?;
            if fuf.get(in_tile).op == OpKind::Add {
                return None;
            }
            (None, seed)
        } else {
            if seed_node.op != OpKind::Add {
                return None;
            }
            // Residual-stream `Add` only — both operands are tiles.
            if !seed_node
                .inputs
                .iter()
                .all(|i| matches!(i, FufInput::Tile { .. }))
            {
                return None;
            }
            let rmsnorm = fuf
                .nodes
                .iter()
                .find(|n| n.op == OpKind::RmsNorm && consumes_tile(n, seed))?;
            (Some(seed), rmsnorm.id)
        };

        // Find three Gemm consumers of the RmsNorm.
        let gemms: Vec<TileId> = fuf
            .nodes
            .iter()
            .filter(|n| n.op == OpKind::Gemm && consumes_tile(n, rmsnorm_tile))
            .map(|n| n.id)
            .collect();
        if gemms.len() != 3 {
            return None;
        }
        // All three Gemms must be Affine-storage (int4 path).
        for &g in &gemms {
            if !matches!(
                weight_storage_of(fuf.get(g)),
                Some(StorageFormat::Affine { group_size, bits })
                    if *group_size == self.group_size && *bits == self.bits
            ) {
                return None;
            }
        }

        // Find the RopeAppend whose first three Tile inputs are these
        // three Gemms (in any order — we'll sort by NUM_Q vs NUM_KV
        // shapes downstream in fan_out).
        let rope_node = fuf.nodes.iter().find(|n| {
            n.op != OpKind::RopeAppend && return false;
            let tile_inputs: Vec<TileId> = n
                .inputs
                .iter()
                .filter_map(|i| match i {
                    FufInput::Tile { id, .. } => Some(*id),
                    _ => None,
                })
                .take(3)
                .collect();
            tile_inputs.len() == 3 && gemms.iter().all(|g| tile_inputs.contains(g))
        })?;
        let rope_tile = rope_node.id;

        let mut claimed: Vec<TileId> = Vec::with_capacity(6);
        if let Some(a) = add_tile {
            claimed.push(a);
        }
        claimed.push(rmsnorm_tile);
        claimed.extend(&gemms);
        claimed.push(rope_tile);
        claimed.sort();

        // Boundary inputs: residual + delta (from Add) for non-init,
        // or just the upstream norm input for init.
        let boundary_inputs: Vec<TileId> = match add_tile {
            Some(a) => fuf
                .get(a)
                .inputs
                .iter()
                .filter_map(|i| match i {
                    FufInput::Tile { id, .. } => Some(*id),
                    _ => None,
                })
                .collect(),
            None => {
                let (in_tile, _) = first_tile_input(fuf.get(rmsnorm_tile))?;
                vec![in_tile]
            }
        };

        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs,
            // Q goes out via rope (used by attention); K/V land in the
            // KV cache (no live output). Add's residual-update output
            // is also live for the next layer's chain. Declare both
            // here for dep tracking.
            boundary_outputs: match add_tile {
                Some(a) => vec![a, rope_tile],
                None => vec![rope_tile],
            },
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Read the swept synth cost when the CSV has a row for this
        // bucket. The sweep wires through to
        // `fuse_pass::synthesize_pre_attn_chunk` to compile + bench
        // the actual synth kernel; absent that row, fall back to the
        // sum of the unfused component costs (FusedAddRmsNorm +
        // 3×AffineQmm + RopeAppend). The solver then picks via tie-
        // break on (a) the small bias below favoring Synth for one
        // fewer dispatch's worth of host-side overhead, or (b) any
        // explicit CSV row if calibrated.
        let num_tokens = ctx.num_tokens() as u32;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as u32;
        let head_dim = ctx.bounds.get("head_dim").copied().unwrap_or(0) as u32;
        let num_q = ctx.bounds.get("num_attention_heads").copied().unwrap_or(0) as u32;
        let num_kv = ctx
            .bounds
            .get("num_key_value_heads")
            .copied()
            .unwrap_or(0) as u32;
        let q_n = num_q.saturating_mul(head_dim);
        let kv_n = num_kv.saturating_mul(head_dim);

        let _ = (q_n, kv_n, hidden, num_tokens);
        let synth_name = format!(
            "synth_pre_attn_{}_{}_gs{}",
            self.act_tag, self.scale_tag, self.group_size,
        );
        if let Some(cost) = ctx.profile.cost_us_for(&synth_name, num_tokens, hidden, 0) {
            return cost;
        }
        // No swept synth cost for this chip yet. Return a very
        // large finite cost so the solver never picks this Impl in
        // the absence of real measurement — `apply_synth_replacement`
        // (post-pass) + `bucket_m < 2` gate (2393cf820) continue to
        // drive the fusion decision until the sweep emits
        // synth_pre_attn rows. Once it does, this returns measured
        // cost and the solver takes over (and the post-pass can be
        // removed). f64::INFINITY trips the solver's finite-cost
        // invariant ("cost_fn returned None"), so use a large but
        // bounded value instead.
        1.0e9
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 16 * 1024,
            regs_per_thread: 64,
            threads_per_cta: 1024,
        }
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::Any; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::Any; m.boundary_outputs.len()]
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "SynthPreAttn",
            vec![
                ("residual_slot", syn::parse_quote!(u32)),
                ("delta_slot", syn::parse_quote!(u32)),
                ("q_out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "q_wt_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                (
                    "k_wt_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                (
                    "v_wt_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                (
                    "rms_wt_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::RmsNorm
                    ),
                ),
                (
                    "cos_sin_fn",
                    syn::parse_quote!(
                        for<'a> fn(
                            &'a Weights,
                            u32,
                        )
                            -> (&'a ::ferrite_kernels::OwnedTensor, &'a ::ferrite_kernels::OwnedTensor)
                    ),
                ),
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
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        // Find the four key tiles within the claim by op-kind. The
        // claim is sorted by TileId so we walk it and pick.
        let mut add_tile: Option<TileId> = None;
        let mut rmsnorm_tile: Option<TileId> = None;
        let mut gemm_tiles: Vec<TileId> = Vec::new();
        let mut rope_tile: Option<TileId> = None;
        for &t in &m.claimed_tiles {
            match fuf.get(t).op {
                OpKind::Add => add_tile = Some(t),
                OpKind::RmsNorm => rmsnorm_tile = Some(t),
                OpKind::Gemm => gemm_tiles.push(t),
                OpKind::RopeAppend => rope_tile = Some(t),
                _ => {}
            }
        }
        let rmsnorm_tile = rmsnorm_tile?;
        let rope_tile = rope_tile?;
        if gemm_tiles.len() != 3 {
            return None;
        }

        // residual_slot / delta_slot:
        //   - Non-init: the two `FufInput::Tile` operands of the Add.
        //     By DSL convention `add(delta, residual)` (delta is the
        //     o_proj output, residual is the running stream); but to
        //     stay tolerant we just take the two tile inputs in order
        //     and trust the runtime kernel's binding contract
        //     (binding(0)=residual, binding(1)=delta).
        //   - Init: residual_slot == delta_slot == norm's first tile
        //     input. The kernel ignores the delta read in init mode.
        let (residual_slot_idx, delta_slot_idx) = match add_tile {
            Some(a) => {
                let add_node = fuf.get(a);
                let tile_inputs: Vec<(TileId, u8)> = add_node
                    .inputs
                    .iter()
                    .filter_map(|i| match i {
                        FufInput::Tile { id, slot } => Some((*id, *slot)),
                        _ => None,
                    })
                    .collect();
                if tile_inputs.len() != 2 {
                    return None;
                }
                (
                    slots.of(tile_inputs[0].0, tile_inputs[0].1),
                    slots.of(tile_inputs[1].0, tile_inputs[1].1),
                )
            }
            None => {
                let (norm_in, norm_slot) = first_tile_input(fuf.get(rmsnorm_tile))?;
                let idx = slots.of(norm_in, norm_slot);
                (idx, idx)
            }
        };

        // q_out_slot: the Q-projection Gemm's output. Identify Q
        // among the 3 Gemms by the RopeAppend's input ordering —
        // rope's first tile input is the Q-gemm, second is K, third
        // is V.
        let rope_node = fuf.get(rope_tile);
        let rope_tile_inputs: Vec<TileId> = rope_node
            .inputs
            .iter()
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .take(3)
            .collect();
        if rope_tile_inputs.len() != 3 {
            return None;
        }
        let q_tile = rope_tile_inputs[0];
        let k_tile = rope_tile_inputs[1];
        let v_tile = rope_tile_inputs[2];
        if !gemm_tiles.contains(&q_tile)
            || !gemm_tiles.contains(&k_tile)
            || !gemm_tiles.contains(&v_tile)
        {
            return None;
        }
        let q_out_slot_idx = slots.of(q_tile, 0);

        // Layer index — recover from any of the per-layer weight
        // inputs. The Gemm tiles' weight has an `index` field;
        // RmsNorm/RopeAppend also have one each.
        let layer = (|| {
            for &t in &m.claimed_tiles {
                for input in &fuf.get(t).inputs {
                    if let FufInput::Weight {
                        index: Some(layer), ..
                    } = input
                    {
                        return Some(*layer as u32);
                    }
                }
            }
            None
        })()?;

        // WeightAccessors per tile. Use `default_required_weights`
        // on each individually so we can route by source tile.
        let acc_for = |tile: TileId| -> Option<WeightAccessor> {
            default_required_weights(&[tile], fuf, program)
                .into_iter()
                .next()
        };
        let q_acc = acc_for(q_tile)?;
        let k_acc = acc_for(k_tile)?;
        let v_acc = acc_for(v_tile)?;
        let rms_acc = acc_for(rmsnorm_tile)?;
        let rope_acc = acc_for(rope_tile)?;

        let to_weights_path = |acc: &WeightAccessor| -> proc_macro2::TokenStream {
            let (base, _layer) = split_base_layer(&acc.name.to_string());
            let ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
            quote! { Weights::#ident }
        };
        let q_wt = to_weights_path(&q_acc);
        let k_wt = to_weights_path(&k_acc);
        let v_wt = to_weights_path(&v_acc);
        let rms_wt = to_weights_path(&rms_acc);
        let cs_fn = to_weights_path(&rope_acc);

        // group_size + bits from the Affine storage on any Gemm.
        let (gs, bits) = match weight_storage_of(fuf.get(q_tile)) {
            Some(StorageFormat::Affine { group_size, bits }) => (*group_size, *bits),
            _ => return None,
        };

        // Kernel symbol matches `fuse_pass::synthesize_pre_attn{,_init}_chunk`.
        let symbol = if self.init {
            format!(
                "synth_pre_attn_init_{}_{}_gs{}",
                self.act_tag, self.scale_tag, self.group_size,
            )
        } else {
            format!(
                "synth_pre_attn_{}_{}_gs{}",
                self.act_tag, self.scale_tag, self.group_size,
            )
        };
        let symbol_lit = syn::LitStr::new(&symbol, proc_macro2::Span::call_site());

        let _ = bits;
        let bits_lit = self.bits;
        let gs_lit = gs;
        let layer_lit = layer;

        Some(vec![OpInstance::new(
            syn::Ident::new("SynthPreAttn", proc_macro2::Span::call_site()),
            vec![
                quote! { #residual_slot_idx },
                quote! { #delta_slot_idx },
                quote! { #q_out_slot_idx },
                quote! { #layer_lit },
                q_wt,
                k_wt,
                v_wt,
                rms_wt,
                cs_fn,
                quote! { #gs_lit },
                quote! { #bits_lit },
                quote! { #symbol_lit },
            ],
        )])
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // Aggregate accessors from every claimed tile so the loader
        // pulls every weight the synth kernel reads (rms gain, three
        // QKV LinearLayer triples, rotary cos/sin).
        default_required_weights(claimed_tiles, fuf, program)
    }
}
