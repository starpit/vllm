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
use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::{
    consumes_tile, first_tile_input, weight_storage_of, CostCtx, Handoff, Implementation,
    LaunchKind, Layout, MatchInfo, OpInstance, OpcodeShape, Resources, SlotMap, WeightAccessor,
    WorkloadConstraint,
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

        let synth_name = format!(
            "synth_pre_attn_{}_{}_gs{}",
            self.act_tag, self.scale_tag, self.group_size,
        );
        if let Some(cost) = ctx.profile.cost_us_for(&synth_name, num_tokens, hidden, 0) {
            return cost;
        }

        // Component-sum fallback.
        let qmv_dt = if self.act_tag == "bfloat" { "bf16" } else { "f16" };
        let qmv_name = format!(
            "affine_qmv_fast_{}_gs{}",
            qmv_dt, self.group_size,
        );
        let qmv_q = ctx
            .profile
            .cost_us_for(&qmv_name, num_tokens, q_n, hidden)
            .unwrap_or(0.0);
        let qmv_kv = ctx
            .profile
            .cost_us_for(&qmv_name, num_tokens, kv_n, hidden)
            .unwrap_or(0.0);
        let norm_name = if self.act_tag == "bfloat" {
            "metal_fused_add_rmsnorm_bf16"
        } else {
            "metal_fused_add_rmsnorm_f16"
        };
        let norm = ctx
            .profile
            .cost_us_for(norm_name, num_tokens, hidden, 0)
            .unwrap_or(0.0);
        let rope = ctx
            .profile
            .cost_us_for("metal_rope_append_bf16", num_tokens, hidden, 0)
            .unwrap_or(0.0);

        // Bias the fused alternative slightly cheaper on tie so the
        // solver picks Synth when components are equivalent and Synth
        // saves dispatch-launch overhead (~50µs × 4 saved launches).
        let bias_us = 50.0 * 4.0;
        (norm + qmv_q + 2.0 * qmv_kv + rope) - bias_us
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
        _m: &MatchInfo,
        _fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        _slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        // TODO(solver-driven-synth): build the SynthPreAttn OpInstance
        // directly from the claimed tiles' weight accessors.
        //
        // Current state: `interpreter_codegen::apply_synth_replacement`
        // still owns the post-pass rewrite, gated by the `bucket_m < 2`
        // heuristic. This Impl is registered so the solver tracks it
        // (cost comparisons stay correct), but `fan_out` returns
        // `None` to fall back to the unfused chain emission while the
        // weight-accessor → token-stream plumbing lands.
        //
        // The full implementation needs to:
        //  1. Pull WeightAccessors from each claimed Gemm (Q/K/V),
        //     the RmsNorm, and the RopeAppend (cos_sin).
        //  2. Convert each accessor name to `Weights::<base>` via
        //     `crate::codegen::split_base_layer`.
        //  3. Resolve slot ids for residual_slot, delta_slot,
        //     q_out_slot via `slots.of(tile, output_slot)`.
        //  4. Build the 12-field OpInstance matching the variant in
        //     `ferrite_forward::instr::Instruction::SynthPreAttn`.
        //
        // Until that lands, returning `None` means the solver doesn't
        // pick this Impl at codegen (no fan_out → no emission), and
        // the legacy `apply_synth_replacement` continues to work as
        // before. The cost_us above still participates in solver
        // scoring, so when this fan_out lands the right pick is
        // already in place.
        None
    }

    fn required_weights(
        &self,
        _claimed_tiles: &[TileId],
        _fuf: &Fuf,
        _program: &Program,
    ) -> Vec<WeightAccessor> {
        // See `fan_out` TODO — once fan_out lands, this returns the
        // five accessors (q_proj, k_proj, v_proj, input_layernorm,
        // rotary) collected from the claimed tiles' Weight inputs.
        Vec::new()
    }
}
