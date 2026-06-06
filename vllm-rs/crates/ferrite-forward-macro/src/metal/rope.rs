// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal Implementation adapters for RoPE operations

use std::collections::BTreeMap;

use crate::classified::{ExternKind, OpKind, Program};
use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::{
    CostCtx, Handoff, Implementation, LaunchKind, Layout, MatchInfo, OpcodeShape, Resources,
    RopeAppendRefImpl, SlotMap, WeightAccessor, WorkloadConstraint, consumes_tile,
    default_required_weights,
};
use crate::target::{Backend, TargetProfile};

/// Metal implementation for RopeAppend (NeoX-style)
///
/// Applies rotary position embedding to query and key tensors.
/// NeoX style: pairs element i with i + half_dim.
///
/// Used by: Llama, GPT-NeoX, Mistral, Qwen, most models
#[derive(Debug)]
pub struct MetalRopeAppendImpl {
    dtype: &'static str, // "fp16" or "bf16"
}

impl MetalRopeAppendImpl {
    pub fn new_fp16() -> Self {
        Self { dtype: "fp16" }
    }

    pub fn new_bf16() -> Self {
        Self { dtype: "bf16" }
    }

    /// Analytical cost model for RoPE (memory-bound operation).
    /// Memory traffic: 2 reads (Q, K) + 1 read (cache) + 2 writes (Q', K')
    fn analytical_cost_us(&self, num_elements: u32, bandwidth_gbps: f64) -> f64 {
        let bytes_per_element = 2.0; // fp16/bf16
        let total_elements = num_elements as f64;

        // RoPE reads Q, K, cos_sin_cache and writes Q', K'
        // Assuming Q and K have same size
        let bytes_read = 3.0 * total_elements * bytes_per_element; // Q, K, cache
        let bytes_written = 2.0 * total_elements * bytes_per_element; // Q', K'
        let total_bytes = bytes_read + bytes_written;

        let total_gb = total_bytes / 1e9;
        let time_seconds = total_gb / bandwidth_gbps;

        // Add small compute overhead for rotation math (2 muls + 2 adds per pair)
        let num_pairs = num_elements / 2;
        let flops = (num_pairs * 4) as f64;
        let compute_overhead = flops / (400.0 * 1e12) * 0.1; // 10% of compute time

        (time_seconds + compute_overhead) * 1e6 // convert to microseconds
    }
}

impl Implementation for MetalRopeAppendImpl {
    fn name(&self) -> &'static str {
        match self.dtype {
            "fp16" => "metal_rope_append_f16",
            "bf16" => "metal_rope_append_bf16",
            _ => "metal_rope_append",
        }
    }

    fn kv_layer_io(
        &self,
        claimed_tiles: &[crate::fuf::TileId],
        fuf: &crate::fuf::Fuf,
    ) -> (Option<u32>, Option<u32>) {
        // RopeAppend writes the per-layer paged KV cache.
        (
            crate::impl_lib::kv_cache_extern_layer(claimed_tiles, fuf),
            None,
        )
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.backend == Backend::Metal
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::Any
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        if node.op != OpKind::RopeAppend {
            return None;
        }

        // Singleton claim - just this RopeAppend tile
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: vec![],
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let tile = m.claimed_tiles[0];
        let node = ctx.fuf.get(tile);

        let shape = &node.outputs[0];
        let dims = ctx.eval_shape(shape);

        if let Some(dims) = dims {
            let num_elements: u32 = dims.iter().copied().product::<u64>() as u32;

            let kernel_name = match self.dtype {
                "fp16" => "rope_append_f16",
                "bf16" => "rope_append_bf16",
                _ => "rope_append_f16",
            };

            if let Some(cost) = ctx.profile.cost_us_for(kernel_name, num_elements, 1, 0) {
                return cost;
            }

            return self.analytical_cost_us(num_elements, ctx.profile.memory_bandwidth_gbps);
        }

        10.0 // conservative fallback
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 16,
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

    // RopeAppend's three outputs (q', k', v') are 3D `TensorView`
    // reshapes over the upstream Q/K/V buffers — no allocation, no
    // move. Mirror the CUDA contract.
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        RopeAppendRefImpl.output_alias(claimed_tiles, fuf)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        RopeAppendRefImpl.opcode_shape()
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<ferrite_forward::Instruction>> {
        RopeAppendRefImpl.fan_out(m, fuf, program, bounds, slots)
    }

    fn as_atom(&self, _m: &MatchInfo, _fuf: &Fuf) -> Option<Box<dyn crate::atom::Atom>> {
        Some(Box::new(crate::atom_lib::RopeAppendAtom))
    }
}

/// Metal implementation for RopeAppendInterleaved (GPT-J style)
///
/// Applies rotary position embedding with interleaved pairing.
/// Interleaved style: pairs element 2i with 2i+1.
///
/// Used by: Cohere CommandR family
#[derive(Debug)]
pub struct MetalRopeAppendInterleavedImpl {
    dtype: &'static str,
}

impl MetalRopeAppendInterleavedImpl {
    pub fn new_fp16() -> Self {
        Self { dtype: "fp16" }
    }

    pub fn new_bf16() -> Self {
        Self { dtype: "bf16" }
    }

    fn analytical_cost_us(&self, num_elements: u32, bandwidth_gbps: f64) -> f64 {
        // Same cost model as standard RoPE - only element pairing differs
        let bytes_per_element = 2.0;
        let total_elements = num_elements as f64;
        let bytes_read = 3.0 * total_elements * bytes_per_element;
        let bytes_written = 2.0 * total_elements * bytes_per_element;
        let total_bytes = bytes_read + bytes_written;
        let total_gb = total_bytes / 1e9;
        let time_seconds = total_gb / bandwidth_gbps;
        let num_pairs = num_elements / 2;
        let flops = (num_pairs * 4) as f64;
        let compute_overhead = flops / (400.0 * 1e12) * 0.1;
        (time_seconds + compute_overhead) * 1e6
    }
}

impl Implementation for MetalRopeAppendInterleavedImpl {
    fn name(&self) -> &'static str {
        match self.dtype {
            "fp16" => "metal_rope_append_interleaved_f16",
            "bf16" => "metal_rope_append_interleaved_bf16",
            _ => "metal_rope_append_interleaved",
        }
    }

    fn kv_layer_io(
        &self,
        claimed_tiles: &[crate::fuf::TileId],
        fuf: &crate::fuf::Fuf,
    ) -> (Option<u32>, Option<u32>) {
        (
            crate::impl_lib::kv_cache_extern_layer(claimed_tiles, fuf),
            None,
        )
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.backend == Backend::Metal
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::Any
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        if node.op != OpKind::RopeAppendInterleaved {
            return None;
        }

        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: vec![],
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let tile = m.claimed_tiles[0];
        let node = ctx.fuf.get(tile);
        let shape = &node.outputs[0];
        let dims = ctx.eval_shape(shape);

        if let Some(dims) = dims {
            let num_elements: u32 = dims.iter().copied().product::<u64>() as u32;
            let kernel_name = match self.dtype {
                "fp16" => "rope_append_interleaved_f16",
                "bf16" => "rope_append_interleaved_bf16",
                _ => "rope_append_interleaved_f16",
            };

            if let Some(cost) = ctx.profile.cost_us_for(kernel_name, num_elements, 1, 0) {
                return cost;
            }

            return self.analytical_cost_us(num_elements, ctx.profile.memory_bandwidth_gbps);
        }

        10.0
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 16,
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

    // Same alias contract + same `Instruction::RopeAppend` shape as
    // the NeoX variant; the `interleaved: bool` field on the variant
    // distinguishes them at codegen time, so one shape covers both.
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        RopeAppendRefImpl.output_alias(claimed_tiles, fuf)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        RopeAppendRefImpl.opcode_shape()
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<ferrite_forward::Instruction>> {
        RopeAppendRefImpl.fan_out(m, fuf, program, bounds, slots)
    }
}


/// Gemma4 pre-attention tail fusion: claims the 4-tile chain
///
///   `RmsNorm(q_raw, q_gains) / RmsNorm(k_raw, k_gains) /
///    RmsNormUnit(v_raw)  →  RopeAppend`
///
/// and emits one `Instruction::RopeAppendNormed` — the per-head norm
/// prologues run inside the rope dispatch (`rope_append_normed_*` in
/// `rope.metal`), replicating the standalone kernels' 256-thread
/// reduction order and per-op rounding boundaries exactly, so the
/// fusion is BIT-IDENTICAL to the 4-kernel sequence at every M.
///
/// The v-input must be `RmsNormUnit` (Gemma4's unweighted V norm), so
/// the matcher can never fire on Qwen3-style qk-norm models or any
/// other arch.
#[derive(Debug)]
pub struct MetalRopeAppendNormedImpl {
    kernel_name: &'static str,
    dtype: &'static str,
}

impl MetalRopeAppendNormedImpl {
    pub fn new_fp16() -> Self {
        Self {
            kernel_name: "rope_append_normed_f16",
            dtype: "fp16",
        }
    }
    pub fn new_bf16() -> Self {
        Self {
            kernel_name: "rope_append_normed_bf16",
            dtype: "bf16",
        }
    }
}

/// Peel an optional single-consumer flatten-back `Reshape` between a
/// rope input and its norm (Gemma4 q/k norms run per-head on a
/// `[m, heads, head_dim]` view, then a Reshape flattens back for the
/// rope tile; the global-class k path has NO such reshape). Returns
/// `(norm_tile, reshape_tile_if_any)`.
fn peel_reshape(fuf: &Fuf, t: TileId) -> (TileId, Option<TileId>) {
    let node = fuf.get(t);
    if node.op == OpKind::Reshape {
        if let Some(inner) = node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } => Some(*id),
            _ => None,
        }) {
            return (inner, Some(t));
        }
    }
    (t, None)
}

/// Validate the full Gemma4 pre-attn pattern anchored at the rope tile
/// and build the claim. Called from `matches` with the q_norm seed
/// already resolved to its downstream rope.
fn match_rope_normed_at(fuf: &Fuf, seed: TileId) -> Option<MatchInfo> {
    let rope = fuf.get(seed);
    if rope.op != OpKind::RopeAppend {
        return None;
    }
    {
        // q / k / v upstream tiles (rope inputs 0, 1, 2).
        let qkv: Vec<TileId> = rope
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let [q_in, k_in, v_t] = qkv.as_slice() else {
            return None;
        };
        let (q_in, k_in, v_t) = (*q_in, *k_in, *v_t);
        let single_consumer = |t: TileId| -> bool {
            fuf.nodes.iter().filter(|n| consumes_tile(n, t)).count() == 1
        };
        // q/k may reach the rope through a flatten-back Reshape over
        // the per-head norm; peel it (and claim it) when present.
        let (q_t, q_rs) = peel_reshape(fuf, q_in);
        let (k_t, k_rs) = peel_reshape(fuf, k_in);
        if q_rs.is_some() && !single_consumer(q_in) {
            return None;
        }
        if k_rs.is_some() && !single_consumer(k_in) {
            return None;
        }
        // q/k are weighted per-head norms; v is the unit norm — each
        // consumed ONLY inside this claim (their pre-norm values must
        // not escape).
        if fuf.get(q_t).op != OpKind::RmsNorm || !single_consumer(q_t) {
            return None;
        }
        if fuf.get(k_t).op != OpKind::RmsNorm || !single_consumer(k_t) {
            return None;
        }
        if fuf.get(v_t).op != OpKind::RmsNormUnit || !single_consumer(v_t) {
            return None;
        }
        // Raw (pre-norm) tiles — the claim boundary. On k_eq_v global
        // layers k_raw == v_raw; dedup for the boundary list.
        let raw_of = |t: TileId| -> Option<TileId> {
            fuf.get(t).inputs.iter().find_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
        };
        let q_raw = raw_of(q_t)?;
        let k_raw = raw_of(k_t)?;
        let v_raw = raw_of(v_t)?;
        let mut boundary_inputs = vec![q_raw, k_raw];
        if v_raw != k_raw {
            boundary_inputs.push(v_raw);
        }

        let mut claimed = vec![q_t, k_t, v_t, seed];
        claimed.extend(q_rs);
        claimed.extend(k_rs);
        claimed.sort();
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_outputs: vec![seed],
            boundary_inputs,
        })
    }
}

impl Implementation for MetalRopeAppendNormedImpl {
    fn name(&self) -> &'static str {
        match self.dtype {
            "fp16" => "metal_rope_append_normed_f16",
            _ => "metal_rope_append_normed_bf16",
        }
    }

    fn kv_layer_io(
        &self,
        claimed_tiles: &[crate::fuf::TileId],
        fuf: &crate::fuf::Fuf,
    ) -> (Option<u32>, Option<u32>) {
        // Writes the per-layer paged KV cache (same as RopeAppend).
        (
            crate::impl_lib::kv_cache_extern_layer(claimed_tiles, fuf),
            None,
        )
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.backend == Backend::Metal
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::Any
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // The solver DP only accepts claims whose seed is the claim's
        // MINIMUM tile (phase 2 rejects backward-reaching claims), so
        // this impl seeds at the Q-NORM — the earliest tile of the
        // chain — and walks FORWARD to the rope tile, then validates
        // the full pattern from there.
        if fuf.get(seed).op != OpKind::RmsNorm {
            return None;
        }
        // seed → (optional flatten-back Reshape) → RopeAppend.
        let mut cur = seed;
        let mut hops = 0;
        let rope_id = loop {
            let consumers: Vec<&crate::fuf::FufNode> = fuf
                .nodes
                .iter()
                .filter(|n| consumes_tile(n, cur))
                .collect();
            let [next] = consumers.as_slice() else {
                return None;
            };
            match next.op {
                OpKind::RopeAppend => break next.id,
                OpKind::Reshape if hops == 0 => {
                    cur = next.id;
                    hops += 1;
                }
                _ => return None,
            }
        };
        let info = match_rope_normed_at(fuf, rope_id)?;
        // Exactly one seed produces the claim: its minimum tile.
        if *info.claimed_tiles.first()? != seed {
            return None;
        }
        Some(info)
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let m = ctx.num_tokens() as u32;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as u32;
        if let Some(cost) = ctx.profile.cost_us_for(self.kernel_name, m, hidden, 0) {
            return cost;
        }
        // Same traffic as the bare rope (norm reads are absorbed into
        // the rope's own row passes), so this analytical estimate is
        // strictly below the unfused 4-op sum and the DP always
        // prefers the fusion when the pattern matches.
        let mf = m as f64;
        let q_size = ctx.bounds.get("num_attention_heads").copied().unwrap_or(0)
            * ctx.bounds.get("head_dim").copied().unwrap_or(0);
        let kv_size = ctx.bounds.get("num_key_value_heads").copied().unwrap_or(0)
            * ctx.bounds.get("head_dim").copied().unwrap_or(0);
        let bytes = mf * (2.0 * q_size as f64 + 4.0 * kv_size as f64) * 2.0;
        let bw = ctx.profile.memory_bandwidth_gbps;
        if bw <= 0.0 {
            return 1.0e9;
        }
        bytes / 1e9 / bw * 1e6
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            // scratch[256] f32 + q_tg/k_tg[512] T_act.
            shmem_bytes: 3 * 1024,
            regs_per_thread: 24,
            threads_per_cta: 512,
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
        // Claimed order [q_norm, k_norm, v_unit, rope] → accessors
        // [q_gains, k_gains] → RmsNorm-kind sub-slots 0 / 1 (v_unit
        // and rope carry no weight refs; CosSin is auto-injected).
        default_required_weights(claimed_tiles, fuf, program)
    }

    #[allow(clippy::type_complexity)]
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        // Rope outputs alias the RAW projection buffers (q is normed +
        // rotated in place; k'/v' tiles are dead — attention reads the
        // cache — but the alias keeps lifetimes conservative).
        let rope_id = *claimed_tiles
            .iter()
            .find(|t| matches!(fuf.get(**t).op, OpKind::RopeAppend))
            .expect("RopeAppendNormed claim contains RopeAppend");
        let norm_in = |t: TileId| -> Option<(TileId, u8)> {
            fuf.get(t).inputs.iter().find_map(|i| match i {
                FufInput::Tile { id, slot } => Some((*id, *slot)),
                _ => None,
            })
        };
        let rope_node = fuf.get(rope_id);
        let upstream = |idx: usize| -> Option<(TileId, u8)> {
            match rope_node.inputs.get(idx) {
                Some(FufInput::Tile { id, .. }) => {
                    let (norm_t, _) = peel_reshape(fuf, *id);
                    norm_in(norm_t)
                }
                _ => None,
            }
        };
        vec![
            ((rope_id, 0), upstream(0)),
            ((rope_id, 1), upstream(1)),
            ((rope_id, 2), upstream(2)),
        ]
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "RopeAppendNormed",
            vec![
                ("q_slot", syn::parse_quote!(u32)),
                ("k_slot", syn::parse_quote!(u32)),
                ("v_slot", syn::parse_quote!(u32)),
                ("q_out_slot", syn::parse_quote!(u32)),
                ("k_out_slot", syn::parse_quote!(u32)),
                ("v_out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("interleaved", syn::parse_quote!(bool)),
                ("is_global", syn::parse_quote!(bool)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<ferrite_forward::Instruction>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| matches!(fuf.get(**t).op, OpKind::RopeAppend))?;
        let rope_node = fuf.get(rope_id);
        // Geometry class — same detection as RopeAppendRefImpl.
        let is_global = !fuf.nodes.iter().any(|n| {
            n.op == OpKind::SlidingAttention && consumes_tile(n, rope_id)
        });
        // RAW (pre-norm) slots: the rope input tiles are the claimed
        // norms; their tile inputs are the projection outputs.
        let raw_slot = |idx: usize| -> Option<u32> {
            let in_id = match rope_node.inputs.get(idx) {
                Some(FufInput::Tile { id, .. }) => *id,
                _ => return None,
            };
            let (norm_id, _) = peel_reshape(fuf, in_id);
            let (raw_id, raw_sub) = fuf.get(norm_id).inputs.iter().find_map(|i| match i {
                FufInput::Tile { id, slot } => Some((*id, *slot)),
                _ => None,
            })?;
            Some(slots.of(raw_id, raw_sub))
        };
        let q_slot = raw_slot(0)?;
        let k_slot = raw_slot(1)?;
        let v_slot = raw_slot(2)?;
        let q_out_slot = slots.of(rope_id, 0);
        let k_out_slot = slots.of(rope_id, 1);
        let v_out_slot = slots.of(rope_id, 2);
        let layer = rope_node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Extern {
                    kind: ExternKind::KvCache,
                    index: Some(layer),
                } => Some(*layer),
                _ => None,
            })
            .expect("RopeAppendNormed: kv_cache extern with concrete layer index")
            as u32;

        Some(vec![ferrite_forward::Instruction::RopeAppendNormed(
            q_slot,
            k_slot,
            v_slot,
            q_out_slot,
            k_out_slot,
            v_out_slot,
            layer,
            false,
            is_global,
        )])
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::target::from_metal_profile;

    #[test]
    fn metal_rope_only_compatible_with_metal_targets() {
        let metal_impl = MetalRopeAppendImpl::new_fp16();
        let metal_profile = from_metal_profile(&ferrite_metal_targets::M1_8CORE);
        assert!(metal_impl.target_compatible(&metal_profile));

        let cuda_profile = crate::target::from_profile_def(&ferrite_cuda_targets::L4_SM89);
        assert!(!metal_impl.target_compatible(&cuda_profile));
    }

    #[test]
    fn metal_rope_interleaved_only_compatible_with_metal() {
        let impl_interleaved = MetalRopeAppendInterleavedImpl::new_fp16();
        let metal_profile = from_metal_profile(&ferrite_metal_targets::M1_8CORE);
        assert!(impl_interleaved.target_compatible(&metal_profile));
    }

    #[test]
    fn metal_rope_analytical_cost_scales_with_size() {
        let impl_fp16 = MetalRopeAppendImpl::new_fp16();

        // Small: 4096 elements
        let small_cost = impl_fp16.analytical_cost_us(4096, 68.25);

        // Large: 262144 elements (64x larger)
        let large_cost = impl_fp16.analytical_cost_us(262144, 68.25);

        // Cost should scale roughly linearly with size
        let ratio = large_cost / small_cost;
        assert!(ratio > 50.0 && ratio < 80.0, "Cost ratio: {}", ratio);

        // Costs should be in reasonable microsecond range
        assert!(
            small_cost > 0.1 && small_cost < 100.0,
            "Small cost: {}",
            small_cost
        );
        assert!(
            large_cost > 1.0 && large_cost < 10000.0,
            "Large cost: {}",
            large_cost
        );
    }

    #[test]
    fn metal_rope_cost_accounts_for_memory_traffic() {
        let impl_fp16 = MetalRopeAppendImpl::new_fp16();

        // RoPE reads 3 buffers (Q, K, cache) + writes 2 (Q', K') = 5× traffic
        // vs Add which reads 2 + writes 1 = 3× traffic
        let num_elements = 1_000_000;
        let bandwidth = 100.0; // GB/s

        let cost = impl_fp16.analytical_cost_us(num_elements, bandwidth);

        // Expected: (3*1M*2 + 2*1M*2) bytes / 100 GB/s = 10 MB / 100 GB/s = 100 µs
        let expected = 100.0;
        assert!(
            (cost - expected).abs() < 10.0,
            "Expected ~{}µs, got {}µs",
            expected,
            cost
        );
    }
}
