// SPDX-License-Identifier: Apache-2.0

//! Metal-target Implementations for `OpKind::Moe`.
//!
//! Two singleton Impls cover the BF16-activations × MLX-affine-int4-expert-
//! weights MoE archs we ship on Metal:
//!
//! * [`MetalFusedMoeImpl`] — Mixtral-family (no shared expert,
//!   `topk → softmax(scores)` order).
//! * [`MetalSharedFusedMoeImpl`] — Qwen2/3-MoE family (optional shared
//!   expert + sigmoid gate, `softmax → topk → take_along_axis` order,
//!   optional `norm_topk_prob` renorm).
//!
//! Both differ from their CUDA-target peers (`FusedMoeRefImpl` /
//! `SharedFusedMoeRefImpl` in `impl_lib.rs`) only in the emitted
//! `Instruction`: the Metal Impls emit [`ferrite_forward::Instruction::
//! MetalFusedMoe`] / [`MetalSharedFusedMoe`], which carry the full
//! macro-baked MoE shape (num_experts, top_k, moe_intermediate_size,
//! hidden_size, shared_intermediate_size, group_size, bits,
//! norm_topk_prob). The Metal lowering pass (`lower_one`) then
//! specializes pipelines + dispatch grids + `Binding::Inline` u32s +
//! `Binding::MoeScratch` byte offsets directly from those literals,
//! without a runtime shape-resolution detour.
//!
//! Storage gate: both Impls require the per-expert weights to be
//! `StorageFormat::Affine` (mlx-community 4bit checkpoint layout).
//! Dense BF16 MoE checkpoints don't have a Metal Impl yet — they'd
//! need a separate per-expert dense-matmul path. The CUDA-target
//! Impls remain target-gated `Backend::Cuda` so they don't fire on
//! Metal builds; an unclaimed `OpKind::Moe` tile on a Dense MoE
//! checkpoint surfaces as a `BackendCompat<Metal>` failure (which is
//! the right outcome — we don't silently lower MoE on a path we
//! haven't ported).

use std::collections::BTreeMap;

use crate::classified::{OpKind, Program};
use crate::codegen::split_base_layer;
use crate::emit::weight_field_name;
use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::{
    CostCtx, Handoff, Implementation, LaunchKind, Layout, MatchContext, MatchInfo, OpcodeShape,
    Resources, SlotMap, WeightAccessor, weight_storage_of,
};
use crate::quantization::StorageFormat;
use crate::target::{Backend, TargetProfile};

use quote::quote;

const UNCALIBRATED_COST_US: f64 = 1_000_000.0;

/// Look up a `u64` bound by name, or panic with a clear "macro-time
/// MoE shape lookup failed" message. Used at `fan_out` time — every
/// MoE shape field listed here must be in the bounds map by the
/// time fan_out is called, because the per-arch
/// `forward!` macro is expected to populate them from `config.json`.
fn bound_or_die(bounds: &BTreeMap<String, u64>, key: &str, ctx: &'static str) -> u64 {
    *bounds.get(key).unwrap_or_else(|| {
        panic!(
            "{ctx}: required macro-time bound `{key}` not present in model.bounds. \
             Available keys: {:?}",
            bounds.keys().collect::<Vec<_>>()
        )
    })
}

/// Read `num_experts` covering Mixtral's `num_local_experts` and the
/// Qwen-MoE family's `num_experts`. Panics if neither is present.
fn read_num_experts(bounds: &BTreeMap<String, u64>, ctx: &'static str) -> u64 {
    if let Some(&v) = bounds.get("num_local_experts") {
        return v;
    }
    if let Some(&v) = bounds.get("num_experts") {
        return v;
    }
    panic!(
        "{ctx}: neither `num_local_experts` (Mixtral) nor `num_experts` (Qwen-MoE) found in \
         model.bounds; available: {:?}",
        bounds.keys().collect::<Vec<_>>()
    )
}

/// Read `moe_intermediate_size`. Qwen-MoE configs ship this directly;
/// Mixtral configs reuse `intermediate_size` (Mixtral has no separate
/// MoE intermediate dim). Try the Qwen key first.
fn read_moe_intermediate(bounds: &BTreeMap<String, u64>, ctx: &'static str) -> u64 {
    if let Some(&v) = bounds.get("moe_intermediate_size") {
        return v;
    }
    if let Some(&v) = bounds.get("intermediate_size") {
        return v;
    }
    panic!(
        "{ctx}: neither `moe_intermediate_size` (Qwen) nor `intermediate_size` (Mixtral) found \
         in model.bounds; available: {:?}",
        bounds.keys().collect::<Vec<_>>()
    )
}

/// Extract group_size + bits when the matched MoE block uses
/// MLX-affine int4 per-expert weights. Returns `None` for Dense /
/// other formats — the Metal Impl declines those. Both Mixtral and
/// Qwen-MoE share storage across all per-expert tensors, so any
/// weight edge's format is the format for the whole MoE block;
/// `weight_storage_of` walks the node's `Weight` edges and returns
/// the first one it can classify.
fn affine_gs_bits(node: &crate::fuf::FufNode) -> Option<(u32, u32)> {
    match weight_storage_of(node)? {
        StorageFormat::Affine { group_size, bits } => Some((*group_size, *bits)),
        _ => None,
    }
}

/// Walk the node's `Weight` edges, deduplicate by base name, and
/// build one `WeightAccessor` of the requested layer struct type per
/// distinct base. Mirrors `FusedMoeRefImpl::required_weights` /
/// `SharedFusedMoeRefImpl::required_weights` — the Metal Impl loads
/// the same `FusedMoELayer` / `SharedFusedMoELayer` struct as cuda,
/// just through a Metal-specific load arm (§2 of
/// `project_metal_moe_kernels_landed`).
fn moe_required_weights(
    claimed_tiles: &[TileId],
    fuf: &Fuf,
    program: &Program,
    layer_struct: proc_macro2::TokenStream,
) -> Vec<WeightAccessor> {
    let tile = claimed_tiles[0];
    let node = fuf.get(tile);
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for input in &node.inputs {
        if let FufInput::Weight { id, index, .. } = input {
            let name = weight_field_name(program, *id, *index);
            if !seen.insert(name.to_string()) {
                continue;
            }
            out.push(WeightAccessor {
                name,
                rust_type: layer_struct.clone(),
                source_weights: vec![(*id, *index)],
            });
        }
    }
    out
}

fn moe_single_tile_match(fuf: &Fuf, seed: TileId) -> Option<MatchInfo> {
    let node = fuf.get(seed);
    if node.op != OpKind::Moe {
        return None;
    }
    Some(MatchInfo {
        claimed_tiles: vec![seed],
        boundary_inputs: vec![],
        boundary_outputs: vec![seed],
    })
}

// ── MetalFusedMoeImpl ────────────────────────────────────────────────
//
// Mixtral-style. `applies_to` gates on `num_local_experts` + no shared
// expert + no DeepSeek `n_routed_experts` — same shape as
// `FusedMoeRefImpl::applies_to` but Backend=Metal and Affine storage
// only.

/// Solver-side singleton for Metal MLX-affine int4 fused MoE
/// (Mixtral family).
#[derive(Debug, Default)]
pub struct MetalFusedMoeImpl;

impl Implementation for MetalFusedMoeImpl {
    fn name(&self) -> &'static str {
        "metal_fused_moe"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.backend == Backend::Metal
    }

    fn applies_to(&self, ctx: &MatchContext) -> bool {
        let b = &ctx.model.bounds;
        b.contains_key("num_local_experts")
            && !b.contains_key("shared_expert_intermediate_size")
            && !b.contains_key("n_routed_experts")
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        let m = moe_single_tile_match(fuf, seed)?;
        // Storage gate: Affine-only. Dense MoE falls through and is
        // currently unclaimed on Metal (surfaces as BackendCompat).
        affine_gs_bits(node)?;
        Some(m)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        UNCALIBRATED_COST_US
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        // MoE on Metal is a multi-`LoweredCommand` decomposition baked
        // into the per-bucket ICB (router softmax → argpartition →
        // top-k slice → take_along_axis → affine_gather_qmv × 3 →
        // moe_weighted_sum). The Metal lowering pass owns the
        // decomposition — see `project_metal_moe_switchglu` and the
        // §3b lowering arm. `HostCallback` here matches the CUDA-side
        // FusedMoeRefImpl; for Metal it's an architecture-level tag,
        // not a runtime dispatch route (the worker stamps the
        // sub-commands at record time).
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

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        moe_required_weights(
            claimed_tiles,
            fuf,
            program,
            quote! { ::ferrite_kernels::layers_moe::FusedMoELayer },
        )
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "MetalFusedMoe",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("num_experts", syn::parse_quote!(u32)),
                ("top_k", syn::parse_quote!(u32)),
                ("moe_intermediate_size", syn::parse_quote!(u32)),
                ("hidden_size", syn::parse_quote!(u32)),
                ("group_size", syn::parse_quote!(u32)),
                ("bits", syn::parse_quote!(u32)),
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
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("MetalFusedMoe: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);

        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("MetalFusedMoe: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;

        let num_experts = read_num_experts(bounds, "MetalFusedMoe") as u32;
        let top_k = bound_or_die(bounds, "num_experts_per_tok", "MetalFusedMoe") as u32;
        let moe_inter = read_moe_intermediate(bounds, "MetalFusedMoe") as u32;
        let hidden = bound_or_die(bounds, "hidden_size", "MetalFusedMoe") as u32;
        let (group_size, bits) =
            affine_gs_bits(node).expect("MetalFusedMoe: matches() admitted a non-Affine MoE tile");

        Some(vec![ferrite_forward::Instruction::MetalFusedMoe(
            in_slot_idx,
            out_slot_idx,
            layer,
            num_experts,
            top_k,
            moe_inter,
            hidden,
            group_size,
            bits,
        )])
    }
}

// ── MetalSharedFusedMoeImpl ───────────────────────────────────────────
//
// Qwen2/3-MoE-style. Gates on `num_experts` and absence of Mixtral's
// `num_local_experts` / DeepSeek's `n_routed_experts`. The
// `shared_expert_intermediate_size` bound is optional and may be 0
// (Qwen3-MoE-30B-A3B-Instruct ships shared_inter=0); the variant
// payload's `shared_intermediate_size` field carries the value
// verbatim and the lowering arm conditions the shared-expert tail on
// `shared_intermediate_size > 0`.

#[derive(Debug, Default)]
pub struct MetalSharedFusedMoeImpl;

impl Implementation for MetalSharedFusedMoeImpl {
    fn name(&self) -> &'static str {
        "metal_shared_fused_moe"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.backend == Backend::Metal
    }

    fn applies_to(&self, ctx: &MatchContext) -> bool {
        let b = &ctx.model.bounds;
        b.contains_key("num_experts")
            && !b.contains_key("num_local_experts")
            && !b.contains_key("n_routed_experts")
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        let m = moe_single_tile_match(fuf, seed)?;
        affine_gs_bits(node)?;
        Some(m)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        UNCALIBRATED_COST_US
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
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

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        moe_required_weights(
            claimed_tiles,
            fuf,
            program,
            quote! { ::ferrite_kernels::layers_moe::SharedFusedMoELayer },
        )
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "MetalSharedFusedMoe",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("num_experts", syn::parse_quote!(u32)),
                ("top_k", syn::parse_quote!(u32)),
                ("moe_intermediate_size", syn::parse_quote!(u32)),
                ("hidden_size", syn::parse_quote!(u32)),
                ("shared_intermediate_size", syn::parse_quote!(u32)),
                ("group_size", syn::parse_quote!(u32)),
                ("bits", syn::parse_quote!(u32)),
                ("norm_topk_prob", syn::parse_quote!(bool)),
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
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("MetalSharedFusedMoe: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);

        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("MetalSharedFusedMoe: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;

        let num_experts = read_num_experts(bounds, "MetalSharedFusedMoe") as u32;
        let top_k = bound_or_die(bounds, "num_experts_per_tok", "MetalSharedFusedMoe") as u32;
        let moe_inter = read_moe_intermediate(bounds, "MetalSharedFusedMoe") as u32;
        let hidden = bound_or_die(bounds, "hidden_size", "MetalSharedFusedMoe") as u32;
        // shared_expert_intermediate_size is optional; 0 means no
        // shared expert (modern Qwen3-MoE-30B-A3B-Instruct ships 0).
        let shared_inter = bounds
            .get("shared_expert_intermediate_size")
            .copied()
            .unwrap_or(0) as u32;
        let (group_size, bits) = affine_gs_bits(node)
            .expect("MetalSharedFusedMoe: matches() admitted a non-Affine MoE tile");
        // norm_topk_prob is a Qwen3-MoE config knob. Older Qwen2-MoE
        // configs omit it; treat as false there. HF stores booleans
        // as 0/1 in the bounds u64 map.
        let norm_topk_prob = bounds.get("norm_topk_prob").copied().unwrap_or(0) != 0;

        Some(vec![ferrite_forward::Instruction::MetalSharedFusedMoe(
            in_slot_idx,
            out_slot_idx,
            layer,
            num_experts,
            top_k,
            moe_inter,
            hidden,
            shared_inter,
            group_size,
            bits,
            norm_topk_prob,
        )])
    }
}
