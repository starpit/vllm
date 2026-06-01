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

use crate::classified::{ExternKind, OpKind, Program};
use crate::codegen::split_base_layer;
use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::{
    CostCtx, Handoff, Implementation, LaunchKind, Layout, MatchInfo, OpcodeShape, Resources,
    SlotMap, WeightAccessor, WorkloadConstraint, consumes_tile, default_required_weights,
    first_tile_input, kv_cache_extern_layer, weight_storage_of,
};
use crate::quantization::StorageFormat;
use crate::target::{Backend, TargetProfile};

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
    /// Qwen3-family BF16-scale variant. See [`is_qwen3_arch`] for the gate.
    pub fn bf16_gs64_s_bf16() -> Self {
        Self {
            act_tag: "bfloat",
            scale_tag: "bfloat",
            group_size: 64,
            bits: 4,
            init: false,
        }
    }
    pub fn bf16_gs64_s_bf16_init() -> Self {
        Self {
            act_tag: "bfloat",
            scale_tag: "bfloat",
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

    fn applies_to(&self, ctx: &crate::impl_lib::MatchContext) -> bool {
        let is_qwen3 = crate::metal::synth_gate_up_silu_mul::is_qwen3_arch(ctx.model);
        matches!(
            (is_qwen3, self.scale_tag),
            (true, "bfloat") | (false, "half")
        )
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        // Restrict to single-token decode (mirrors SynthMlpPreDown).
        // At M>=2 per-row megakernel pays the same within-TG serial
        // cost without launch-overhead-saving benefit; unfused chain
        // wins via GPU-pipelined kernel overlap.
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
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

        // Find the RopeAppend whose first three Tile inputs resolve
        // (directly or via an interposing `BiasAdd`) to the three
        // gemms above. Qwen2/Qwen2.5 emit `bias_add(q, q_proj.bias)`
        // between each Gemm and the rope; the matcher walks one hop
        // through that BiasAdd to recover the underlying Gemm. The
        // resolved tile is also recorded so fan_out can collect the
        // bias weight refs.
        //
        // Uniform-bias check (mirrors the cuda `FusedQkvRopeCacheImpl`
        // claim): all three slots must go via BiasAdd or none. Mixed
        // shapes don't appear in production DSLs and the synth
        // kernel signature can't accommodate them without a new
        // axis.
        let resolve = |tid: TileId| -> Option<(TileId, Option<TileId>)> {
            // Returns (gemm_tile, optional_bias_add_in_path).
            let n = fuf.get(tid);
            match n.op {
                OpKind::Gemm => Some((tid, None)),
                OpKind::BiasAdd => {
                    let (upstream, _) = first_tile_input(n)?;
                    if fuf.get(upstream).op == OpKind::Gemm {
                        Some((upstream, Some(tid)))
                    } else {
                        None
                    }
                }
                _ => None,
            }
        };
        let (rope_node, resolved): (&crate::fuf::FufNode, [(TileId, Option<TileId>); 3]) = {
            #[allow(clippy::type_complexity)]
            let mut found: Option<(_, [(TileId, Option<TileId>); 3])> = None;
            for n in &fuf.nodes {
                if n.op != OpKind::RopeAppend {
                    continue;
                }
                let tile_inputs: Vec<TileId> = n
                    .inputs
                    .iter()
                    .filter_map(|i| match i {
                        FufInput::Tile { id, .. } => Some(*id),
                        _ => None,
                    })
                    .take(3)
                    .collect();
                if tile_inputs.len() != 3 {
                    continue;
                }
                let resolved: Option<[(TileId, Option<TileId>); 3]> = (|| {
                    Some([
                        resolve(tile_inputs[0])?,
                        resolve(tile_inputs[1])?,
                        resolve(tile_inputs[2])?,
                    ])
                })();
                let Some(resolved) = resolved else { continue };
                if !gemms
                    .iter()
                    .all(|g| resolved.iter().any(|(gid, _)| gid == g))
                {
                    continue;
                }
                let bias_count = resolved.iter().filter(|(_, b)| b.is_some()).count();
                if bias_count != 0 && bias_count != 3 {
                    // Mixed: reject so the singleton path picks up
                    // the chain.
                    continue;
                }
                found = Some((n, resolved));
                break;
            }
            found?
        };
        let rope_tile = rope_node.id;
        let bias_tiles: Vec<TileId> = resolved.iter().filter_map(|(_, b)| *b).collect();

        let mut claimed: Vec<TileId> = Vec::with_capacity(6 + bias_tiles.len());
        if let Some(a) = add_tile {
            claimed.push(a);
        }
        claimed.push(rmsnorm_tile);
        claimed.extend(&gemms);
        claimed.extend(&bias_tiles);
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

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Prefer the swept synth-kernel row when the CSV has one for
        // this chip + bucket. Absent that, synthesize a component-sum
        // analytical estimate — the same shape `FusedAddRmsNormImpl`
        // / `MetalAffineQmmImpl` / `MetalRopeAppendImpl` each fall
        // back to. The solver then picks fused vs unfused on a tiny
        // bias toward Synth (one fewer dispatch's worth of host
        // overhead) rather than the 1e9 sentinel that used to keep
        // this Impl out of the running on uncalibrated chips.
        let num_tokens = ctx.num_tokens() as u32;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as u32;
        let head_dim = ctx.bounds.get("head_dim").copied().unwrap_or(0) as u32;
        let num_q = ctx.bounds.get("num_attention_heads").copied().unwrap_or(0) as u32;
        let num_kv = ctx.bounds.get("num_key_value_heads").copied().unwrap_or(0) as u32;
        let q_n = num_q.saturating_mul(head_dim);
        let kv_n = num_kv.saturating_mul(head_dim);

        // SAFETY GATE — same M-direction bandwidth issue as
        // SynthGateUpSiluMul: this synth was swept only at M ∈ {1, 2,
        // 4, 8, 16} (see `cost_m4.csv` `synth_pre_attn_*` rows). At
        // M=1024 the analytical fallback underestimates real cost by
        // ~100×, and the actual kernel is ~8× slower than per-op
        // AffineQmm + RmsNorm + RopeAppend. Gate it off above the
        // validated range until the kernel is redesigned with
        // M-blocked tiling.
        if num_tokens > 64 {
            return 1.0e15;
        }

        let synth_name = format!(
            "synth_pre_attn_{}_{}_gs{}",
            self.act_tag, self.scale_tag, self.group_size,
        );
        if let Some(cost) = ctx.profile.cost_us_for(&synth_name, num_tokens, hidden, 0) {
            return cost;
        }

        if hidden == 0 || q_n == 0 || kv_n == 0 {
            return 1.0e9;
        }

        let bw = ctx.profile.memory_bandwidth_gbps;
        let tflops = ctx.profile.peak_tflops_fp16;
        if bw <= 0.0 || tflops <= 0.0 {
            return 1.0e9;
        }
        let act_bytes = 2.0_f64;
        let mf = num_tokens.max(1) as f64;
        let h = hidden as f64;
        let qn = q_n as f64;
        let kvn = kv_n as f64;

        // Norm step. `init=true` is a singleton RmsNorm (one [M,N]
        // read + one [M,N] write); `init=false` is FusedAddRmsNorm
        // with the residual read + residual_out write
        // (matches `FusedAddRmsNormImpl::analytical_cost_us` with
        // `has_residual_out=true` — synth always emits the writeback
        // for the next layer's chain).
        let norm_bytes = if self.init {
            mf * h * act_bytes * 2.0 + h * act_bytes
        } else {
            mf * h * act_bytes * 4.0 + h * act_bytes
        };
        let norm_us = norm_bytes / 1e9 / bw * 1e6;

        // 3×AffineQmm (Q+K+V). Mirror `MetalAffineQmmImpl::
        // analytical_cost_us` (compute-roofline at peak_tflops_fp16)
        // summed across the three N's. The qmv/qmm_t kernels
        // dequantize in-register so peak compute is the right
        // roofline regardless of the int4 BW saving.
        let total_n = qn + 2.0 * kvn;
        let gemm_flops = 2.0 * mf * h * total_n;
        let gemm_us = gemm_flops / (tflops * 1e12) * 1e6;

        // RopeAppend. Mirror `MetalRopeAppendImpl::analytical_cost_us`
        // — bandwidth-bound on `2 * (q_n + kv_n) * M` elements (Q+K
        // input/output streams). cos_sin cache adds the third read
        // term.
        let rope_elems = mf * (qn + kvn);
        let rope_bytes = 5.0 * rope_elems * act_bytes;
        let rope_us = rope_bytes / 1e9 / bw * 1e6;

        // Small bias toward Synth so the solver picks fused on tie
        // (one fewer dispatch's worth of host overhead, ~2-5 µs on
        // Apple silicon).
        (norm_us + gemm_us + rope_us) * 0.95
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

    #[allow(clippy::type_complexity)]
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        // Mirrors `FusedQkvRopeCacheImpl::output_alias`: the only
        // externally-visible output is rotated-Q (rope slot 0). Slots
        // 1/2 (K/V) are paged-cache views, owned by the kv_cache pool
        // — absent from the alias map (untracked). The intermediate
        // rms-out + qmv-band outputs live in threadgroup memory inside
        // the megakernel and never escape to a runtime slot, so we
        // omit them too — declaring them as standalone Impl outputs
        // (the default behavior) would have the slot allocator size
        // arena buffers for ghost slots no kernel binds.
        let rope_id = *claimed_tiles
            .iter()
            .find(|t| matches!(fuf.get(**t).op, OpKind::RopeAppend))
            .expect("SynthPreAttn claim contains RopeAppend");
        vec![((rope_id, 0), None)]
    }

    fn kv_layer_io(&self, claimed_tiles: &[TileId], fuf: &Fuf) -> (Option<u32>, Option<u32>) {
        // SynthPreAttn writes the per-layer paged KV cache (its
        // RopeAppend sub-tile carries the `ExternKind::KvCache`
        // extern). Reads nothing from KV. Without this override the
        // hazard analyzer never inserts a barrier between this
        // dispatch and the immediately-following AttentionViaCache
        // that reads the same KV slot — Metal then runs the attention
        // concurrently with the KV write and produces garbage.
        (kv_cache_extern_layer(claimed_tiles, fuf), None)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        // MUST stay byte-identical to
        // `interpreter_codegen::synth_pre_attn_opcode_shape` —
        // arch_opcodes registers under one logical "SynthPreAttn"
        // name and the macro panics on cross-registrar disagreement.
        // Canonical names/types match the
        // `ferrite_forward::Instruction::SynthPreAttn` tuple variant
        // declared in `ferrite-forward::instr`.
        OpcodeShape::new(
            "SynthPreAttn",
            vec![
                ("residual_slot", syn::parse_quote!(u32)),
                ("delta_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("residual_out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("group_size", syn::parse_quote!(u32)),
                ("bits", syn::parse_quote!(u32)),
                ("kernel_symbol", syn::parse_quote!(&'static str)),
                // Set when the claim absorbed BiasAdd tiles between
                // each Gemm and the RopeAppend (Qwen2 QKV biases).
                // Selects the `_bias` kernel variant and tells the
                // metal lowering arm to bind the 3 `AffineLinearBias`
                // weights at buffers 18/19/20.
                ("has_linear_bias", syn::parse_quote!(bool)),
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
    ) -> Option<Vec<ferrite_forward::Instruction>> {
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
        //     DSL convention `Add(delta, residual)` — inputs[0] is the
        //     o_proj/mlp-down delta, inputs[1] is the running residual.
        //     This MUST match `FusedAddRmsNormImpl::fan_out` /
        //     `output_alias` (impl_lib.rs ~4740) so that solver-picked
        //     synth kernels at L≥1 land their slot indices in the same
        //     order as the post-pass `apply_synth_replacement` would
        //     have emitted them (post-pass reads `field_values[0]` as
        //     delta and `field_values[1]` as residual). Getting this
        //     backwards swaps `residual_slot`/`delta_slot` in the loop
        //     body once cost rows make the solver pick this Impl: the
        //     kernel writes the in-place residual update into the
        //     wrong arena buffer (the one a later `AffineQmm o_proj`
        //     clobbers), producing garbage decode.
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
                let delta = tile_inputs[0];
                let residual = tile_inputs[1];
                (slots.of(residual.0, residual.1), slots.of(delta.0, delta.1))
            }
            None => {
                let (norm_in, norm_slot) = first_tile_input(fuf.get(rmsnorm_tile))?;
                let idx = slots.of(norm_in, norm_slot);
                (idx, idx)
            }
        };

        // residual_out: the updated residual (`residual_in + delta`) the
        // kernel writes. Non-init = the `Add` tile's own output value
        // (coloring gives it a slot distinct from `residual_slot_idx`, so
        // the kernel reads the input and writes this — never in place).
        // Init = no residual add / no write, so it aliases the input slot
        // (downstream reads the unchanged embedding); the kernel leaves it
        // untouched in init mode.
        let residual_out_slot_idx = match add_tile {
            Some(a) => slots.of(a, 0),
            None => residual_slot_idx,
        };

        // q_out_slot: the Q-projection Gemm's output. Identify Q
        // among the 3 Gemms by the RopeAppend's input ordering —
        // rope's first tile input is the Q (gemm or biased), second
        // is K, third is V. Walk through the optional `BiasAdd`
        // between each Gemm and the rope so the same fan_out covers
        // both the bias-free Llama chain and the biased Qwen2 chain.
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
        let resolve_gemm = |tid: TileId| -> Option<(TileId, Option<TileId>)> {
            let n = fuf.get(tid);
            match n.op {
                OpKind::Gemm => Some((tid, None)),
                OpKind::BiasAdd => {
                    let (upstream, _) = first_tile_input(n)?;
                    if fuf.get(upstream).op == OpKind::Gemm {
                        Some((upstream, Some(tid)))
                    } else {
                        None
                    }
                }
                _ => None,
            }
        };
        let (q_tile, q_bias_tile) = resolve_gemm(rope_tile_inputs[0])?;
        let (k_tile, k_bias_tile) = resolve_gemm(rope_tile_inputs[1])?;
        let (v_tile, v_bias_tile) = resolve_gemm(rope_tile_inputs[2])?;
        if !gemm_tiles.contains(&q_tile)
            || !gemm_tiles.contains(&k_tile)
            || !gemm_tiles.contains(&v_tile)
        {
            return None;
        }
        // Uniform-bias check (same gate the matcher applied — guards
        // against a stale match → fan_out shape mismatch).
        let has_linear_bias = q_bias_tile.is_some();
        if has_linear_bias != (k_bias_tile.is_some() && v_bias_tile.is_some())
            || k_bias_tile.is_some() != v_bias_tile.is_some()
        {
            return None;
        }
        // q_out_slot: the *post-rope* Q output, i.e. RopeAppend's
        // output slot 0 — that's where the downstream attention reads
        // Q from. The pre-rope Q-gemm's output is internal to the
        // synth kernel and lives in TG memory.
        let q_out_slot_idx = slots.of(rope_tile, 0);

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
        // RopeAppend has no FufInput::Weight inputs — its cos_sin cache
        // is plumbed as an `ExternKind::Rotary` / `RotaryLocal` extern.
        // Mirror `RopeAppendRefImpl::fan_out`: pick the per-arch
        // `Weights::rotary_cos_sin{,_local}` field accessor based on
        // which extern kind appears in the claimed tiles.
        let uses_local_rotary = m.claimed_tiles.iter().any(|&tid| {
            fuf.get(tid).inputs.iter().any(|i| {
                matches!(
                    i,
                    FufInput::Extern {
                        kind: ExternKind::RotaryLocal,
                        ..
                    }
                )
            })
        });
        let cos_sin_ident = if uses_local_rotary {
            syn::Ident::new("rotary_local_cos_sin", proc_macro2::Span::call_site())
        } else {
            syn::Ident::new("rotary_cos_sin", proc_macro2::Span::call_site())
        };

        let to_base_ident = |acc: &WeightAccessor| -> syn::Ident {
            let (base, _layer) = split_base_layer(&acc.name.to_string());
            syn::Ident::new(&base, proc_macro2::Span::call_site())
        };
        let q_base = to_base_ident(&q_acc);
        let k_base = to_base_ident(&k_acc);
        let v_base = to_base_ident(&v_acc);
        let rms_base = to_base_ident(&rms_acc);

        // group_size + bits from the Affine storage on any Gemm.
        let (gs, bits) = match weight_storage_of(fuf.get(q_tile)) {
            Some(StorageFormat::Affine { group_size, bits }) => (*group_size, *bits),
            _ => return None,
        };

        // Kernel symbol matches `fuse_pass::synthesize_pre_attn{,_init}_chunk`,
        // including the `_bias` suffix when biased.
        let bias_suffix = if has_linear_bias { "_bias" } else { "" };
        let symbol = if self.init {
            format!(
                "synth_pre_attn_init_{}_{}_gs{}{}",
                self.act_tag, self.scale_tag, self.group_size, bias_suffix,
            )
        } else {
            format!(
                "synth_pre_attn_{}_{}_gs{}{}",
                self.act_tag, self.scale_tag, self.group_size, bias_suffix,
            )
        };
        let _ = bits;
        // Weight order — Q/K/V LinearLayers, RmsNorm, CosSin — flows
        // through `required_weights()` as a parallel array; the macro
        // (`emit_weight_accessors_impl`) walks it to build the
        // per-arch `WeightAccessors` match arms with sub-slot indices
        // 0/1/2 for Q/K/V LinearLayer, 0 for RmsNorm, 0 for CosSin.
        let _ = (q_base, k_base, v_base, rms_base, cos_sin_ident);
        // SynthPreAttn's `kernel_symbol: &'static str` lives on the
        // emitted `Instruction`. Leak the formatted symbol so the
        // `&'static str` reference outlives the macro invocation —
        // it's serialised back into the per-arch static slice by
        // `instruction_to_tokens`.
        let kernel_symbol: &'static str = Box::leak(symbol.into_boxed_str());
        Some(vec![ferrite_forward::Instruction::SynthPreAttn(
            residual_slot_idx,
            delta_slot_idx,
            q_out_slot_idx,
            residual_out_slot_idx,
            layer,
            gs,
            self.bits,
            kernel_symbol,
            has_linear_bias,
        )])
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // Aggregate accessors from every claimed tile *except* the
        // optional `BiasAdd` per QKV branch — their bias weight refs
        // would produce standalone `GpuTensor`-typed accessors (per
        // `rust_type_for_weight_consumed_by`) that collide with the
        // `LinearLayer` accessor declared from the upstream Gemm.
        let filtered: Vec<TileId> = claimed_tiles
            .iter()
            .copied()
            .filter(|&t| fuf.get(t).op != OpKind::BiasAdd)
            .collect();
        default_required_weights(&filtered, fuf, program)
    }
}
