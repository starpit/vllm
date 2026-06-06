// SPDX-License-Identifier: Apache-2.0
//! Solver-driven `SynthMlpPreDown` megakernel Impl.
//!
//! Claims the 5-tile post-attention MLP chain
//! `(Add → RmsNorm → Gemm gate + Gemm up → SiluMul)` at the FUF level.
//! When the solver DP picks this Impl over the unfused alternative
//! `(FusedAddRmsNormImpl + 2×MetalAffineQmmImpl + SiluMulImpl)`,
//! `fan_out` emits a single `Instruction::SynthMlpPreDown` so the
//! runtime dispatches one synthesized kernel instead of four
//! separate ones.
//!
//! Mirrors `MetalSynthPreAttnImpl`: same compile-once AOT symbol
//! (`synth_mlp_pre_down_<act>_<scale>_gs<gs>`), same CSV-first /
//! analytical-fallback cost model. Replaces
//! `interpreter_codegen::apply_synth_replacement_mlp` — that
//! post-pass ran AFTER the solver and rewrote the instruction stream
//! unconditionally; with the Impl in the solver DP the pick is
//! cost-driven per bucket.

use std::collections::BTreeMap;

use crate::classified::{OpKind, Program};
use crate::codegen::split_base_layer;
use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::{
    CostCtx, Handoff, Implementation, LaunchKind, Layout, MatchInfo, OpcodeShape, Resources,
    SlotMap, WeightAccessor, WorkloadConstraint, consumes_tile, default_required_weights,
};
use crate::quantization::StorageFormat;
use crate::target::{Backend, TargetProfile};

fn weight_storage_of(node: &crate::fuf::FufNode) -> Option<&StorageFormat> {
    for input in &node.inputs {
        if let FufInput::Weight { storage, .. } = input {
            return Some(storage);
        }
    }
    None
}

/// SynthMlpPreDown megakernel claim. Always seeds at the
/// post-attention `Add` whose downstream consumers are the
/// gate/up Gemms (no init variant — the post-attn norm always has
/// a preceding residual add).
#[derive(Debug)]
pub struct MetalSynthMlpPreDownImpl {
    pub act_tag: &'static str,
    pub scale_tag: &'static str,
    pub group_size: u32,
    pub bits: u32,
    /// MLP gate activation this variant claims: `OpKind::Silu`
    /// (Llama/Qwen GLU) or `OpKind::Gelu` (Gemma GeGLU). Drives both
    /// the FUF match and the kernel-symbol variant tag — mirrors
    /// `fuse_pass::synthesize_mlp_pre_down_chunk`.
    pub act_op: OpKind,
}

impl MetalSynthMlpPreDownImpl {
    pub fn bf16_gs64() -> Self {
        Self {
            act_tag: "bfloat",
            scale_tag: "half",
            group_size: 64,
            bits: 4,
            act_op: OpKind::Silu,
        }
    }
    /// Qwen3-family BF16-scale variant.
    pub fn bf16_gs64_s_bf16() -> Self {
        Self {
            act_tag: "bfloat",
            scale_tag: "bfloat",
            group_size: 64,
            bits: 4,
            act_op: OpKind::Silu,
        }
    }
    /// Gemma4 GeGLU variant: tanh-GELU gate over 8-bit MLP
    /// projections (`mlx-affine-b4-g64-mlp8` preset), BF16 scales.
    pub fn gelu_b8_gs64_s_bf16() -> Self {
        Self {
            act_tag: "bfloat",
            scale_tag: "bfloat",
            group_size: 64,
            bits: 8,
            act_op: OpKind::Gelu,
        }
    }

    /// Kernel symbol — MUST stay byte-identical to the naming in
    /// `fuse_pass::synthesize_mlp_pre_down_chunk` (legacy untagged
    /// name for the (Silu, b4) variant keeps existing cost-CSV rows
    /// and metallib registrations valid).
    fn symbol(&self) -> String {
        let variant_tag = match (self.act_op, self.bits) {
            (OpKind::Gelu, b) => format!("gelu_b{b}_"),
            (_, 4) => String::new(),
            (_, b) => format!("b{b}_"),
        };
        format!(
            "synth_mlp_pre_down_{}{}_{}_gs{}",
            variant_tag, self.act_tag, self.scale_tag, self.group_size,
        )
    }
}

impl Implementation for MetalSynthMlpPreDownImpl {
    fn name(&self) -> &'static str {
        "metal_synth_mlp_pre_down"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        // SynthMlpPreDown is unsafe on M1 Max at every M tested:
        //   - M=1 (single-stream decode): threadgroup barrier deadlock
        //     in mk_tg_rmsnorm_scale, MTLCommandBufferStatus(5) abort
        //     after 510 ms (9419ca204).
        //   - M>=2 (batched decode / prefill): kernel runs to
        //     completion but produces silently-wrong output — chat
        //     emits garbage like "ctorctorctorctor..." or "!!!!!!".
        //     Same failure mode the FusedGateUpSiluMul Affine path had
        //     before 62318fb52, suggesting a shared root cause in
        //     either the AddRmsNormAtom or AffineQmvAtom Metal emit
        //     when run on M1 hardware. Reproduces on M1 only — M3/M4
        //     produce correct output.
        // Gate it off on M1 entirely until either the unsafe atom
        // path is identified or the M1-specific divergence is fixed.
        // FERRITE_NO_SYNTH=1 disables the M=1 fused synth megakernels so
        // ALL M route through the unfused chain — makes decode
        // batch-invariant (M=1 == M>1) and sidesteps quant-synth metallib
        // gaps (e.g. the bf16-scale `synth_mlp_pre_down_*` library that
        // isn't registered for quantized Qwen3.5). See `workload_constraint`.
        if std::env::var_os("FERRITE_NO_SYNTH").is_some() {
            return false;
        }
        if profile.backend != Backend::Metal {
            return false;
        }
        match &profile.backend_spec {
            crate::target::BackendSpec::Metal(m) => !m.generation.starts_with("M1"),
            _ => false,
        }
    }

    fn applies_to(&self, ctx: &crate::impl_lib::MatchContext) -> bool {
        // mlx-affine checkpoints: the M=1 synth megakernel's fused rmsnorm
        // double-counts the pre-applied zero-centered offset → degenerate
        // output. Route to the (correct) unfused chain.
        //
        // Gelu-variant exemption (Gemma4): the synth AddRmsNormAtom
        // applies plain `w·x̂` (offset 0). Models whose runtime norm
        // offset is 0 — Gemma4 stores FULL gains, `norm_weight_runtime_
        // offset` returns 0.0 — match that exactly, so the double-count
        // hazard doesn't exist. Scoped to the Gelu variant so the Silu
        // (Llama/Qwen) routing is untouched.
        if crate::metal::synth_gate_up_silu_mul::is_mlx_affine(ctx.model) {
            let gelu_offset0 = self.act_op == OpKind::Gelu
                && crate::codegen::norm_weight_runtime_offset(ctx.model) == 0.0;
            if !gelu_offset0 {
                return false;
            }
        }
        let is_qwen3 = crate::metal::synth_gate_up_silu_mul::is_qwen3_arch(ctx.model);
        matches!(
            (is_qwen3, self.scale_tag),
            (true, "bfloat") | (false, "half")
        )
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        // Restrict to single-token decode. At M>=2 the per-row
        // megakernel runs one TG per (m_row, tile), filling SMs with
        // serial sequential gate-qmv→up-qmv→silu_mul work inside
        // each TG with no inter-kernel overlap the unfused chain
        // gets from GPU pipelining successive dispatches.
        //
        // KNOWN CONSEQUENCE — BATCH NON-INVARIANCE (diagnosed 2026-06-02,
        // accepted/documented, not a bug): because this fused megakernel
        // fires ONLY at M==1 while M>=2 routes through the unfused chain
        // (FusedAddRmsNorm + AffineQmm gate + AffineQmm up + SiluMul), the
        // two paths keep different intermediate precision (fused holds f32
        // across the chain; unfused round-trips bf16 between dispatches).
        // So a token's decode logits are NOT bit-identical across batch
        // sizes M. For an UNCERTAIN greedy token this can flip the argmax,
        // so batched decode of identical prompts may produce a different
        // (still coherent) token than single-seq, and — combined with the
        // engine's non-lockstep scheduling (a seq lands at varying M each
        // run) — vary run-to-run / row-to-row. NOT a race: lockstep rows
        // (same M, same position) are byte-identical; survives all GPU
        // barriers. Attention (`attention_via_cache_v2` vs
        // `attention_prefill_sdpa_v2_paged`) is bit-identical; lm_head is
        // per-row qmv. The ONLY M-dependence is this synth-vs-unfused gate
        // (here + `synth_pre_attn.rs`). To make batched decode bit-exact
        // to single-seq, the fused and unfused paths must agree
        // numerically (e.g. route M>1 through the fused path — at the TPOT
        // cost this gate exists to avoid, see commit c7411077b).
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let seed_node = fuf.get(seed);
        if seed_node.op != OpKind::Add {
            return None;
        }
        // Residual-stream Add only — both operands are tiles.
        if !seed_node
            .inputs
            .iter()
            .all(|i| matches!(i, FufInput::Tile { .. }))
        {
            return None;
        }
        // RmsNorm consumes the Add.
        let rmsnorm = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::RmsNorm && consumes_tile(n, seed))?;
        let rmsnorm_tile = rmsnorm.id;

        // Exactly two Gemm consumers of the RmsNorm — gate and up.
        let gemms: Vec<TileId> = fuf
            .nodes
            .iter()
            .filter(|n| n.op == OpKind::Gemm && consumes_tile(n, rmsnorm_tile))
            .map(|n| n.id)
            .collect();
        if gemms.len() != 2 {
            return None;
        }
        // Both Gemms must be Affine-storage (int4 path).
        for &g in &gemms {
            if !matches!(
                weight_storage_of(fuf.get(g)),
                Some(StorageFormat::Affine { group_size, bits })
                    if *group_size == self.group_size && *bits == self.bits
            ) {
                return None;
            }
        }

        // The gate activation (Silu or Gelu, per variant) consumes
        // exactly one of the Gemms (the gate).
        let act_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == self.act_op && gemms.iter().any(|g| consumes_tile(n, *g)))?;
        let silu_tile = act_node.id;
        // Mul consumes the activation and the other (up) Gemm.
        let mul_node = fuf.nodes.iter().find(|n| {
            if n.op != OpKind::Mul || !consumes_tile(n, silu_tile) {
                return false;
            }
            // Mul's other Tile input must be the up Gemm.
            n.inputs.iter().any(|i| match i {
                FufInput::Tile { id, .. } => *id != silu_tile && gemms.contains(id),
                _ => false,
            })
        })?;
        let mul_tile = mul_node.id;

        let mut claimed: Vec<TileId> = Vec::with_capacity(6);
        claimed.push(seed);
        claimed.push(rmsnorm_tile);
        claimed.extend(&gemms);
        claimed.push(silu_tile);
        claimed.push(mul_tile);
        claimed.sort();

        // Boundary inputs: residual + delta (from Add).
        let boundary_inputs: Vec<TileId> = fuf
            .get(seed)
            .inputs
            .iter()
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();

        Some(MatchInfo {
            claimed_tiles: claimed,
            // Add's residual update is live for the next layer; Mul
            // output feeds the standalone down-projection Gemm.
            boundary_outputs: vec![seed, mul_tile],
            boundary_inputs,
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Mirrors `MetalSynthPreAttnImpl::cost_us`: CSV-first, then
        // analytical component-sum fallback so chips without
        // `synth_mlp_pre_down_*` rows still get the Impl picked.
        let num_tokens = ctx.num_tokens() as u32;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as u32;
        let intermediate = ctx.bounds.get("intermediate_size").copied().unwrap_or(0) as u32;

        // SAFETY GATE — mirror `MetalSynthPreAttnImpl::cost_us`. This
        // synth's kernel is decode-shaped (one threadgroup per token,
        // no M-direction weight reuse). Measured 2026-05-25 on M4,
        // Llama-3.2-3B-Instruct-4bit, M=1024: ~6940 µs/call — ~85% of
        // all prefill GPU time and ~6× slower than the per-op M-tiled
        // qmm_t gate/up/down chain. Both the swept CSV row and the
        // analytical fallback below underprice it badly, so the gate
        // must sit BEFORE the CSV lookup. Above M=64, force the per-op
        // path; the M=1..64 decode/small-prefill win is unaffected.
        if num_tokens > 64 {
            return 1.0e15;
        }

        let synth_name = self.symbol();
        if let Some(cost) = ctx.profile.cost_us_for(&synth_name, num_tokens, hidden, 0) {
            return cost;
        }

        if hidden == 0 || intermediate == 0 {
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
        let im = intermediate as f64;

        // FusedAddRmsNorm with residual_out — same shape as
        // `MetalFusedAddRmsNormImpl::analytical_cost_us`
        // (has_residual_out=true).
        let norm_bytes = mf * h * act_bytes * 4.0 + h * act_bytes;
        let norm_us = norm_bytes / 1e9 / bw * 1e6;

        // 2×AffineQmm (gate + up). Compute-bound roofline mirrors
        // `MetalAffineQmmImpl::analytical_cost_us` summed across both.
        let gemm_flops = 2.0 * 2.0 * mf * h * im;
        let gemm_us = gemm_flops / (tflops * 1e12) * 1e6;

        // SiluMul: bandwidth-bound over `M*intermediate` elements
        // (reads gate + up, writes one product output).
        let silu_bytes = 3.0 * mf * im * act_bytes;
        let silu_us = silu_bytes / 1e9 / bw * 1e6;

        (norm_us + gemm_us + silu_us) * 0.95
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
        // SiluMul output is the only externally-visible product
        // (consumed by the standalone down-projection Gemm). The
        // residual `Add`'s output is aliased back into the residual
        // slot in-place by the kernel and the slot allocator already
        // tracks it via the residual boundary input — no separate
        // alias entry needed (mirrors `FusedAddRmsNormImpl::output_alias`
        // semantics where the slot keeps its identity across the op).
        let mul_id = *claimed_tiles
            .iter()
            .find(|t| matches!(fuf.get(**t).op, OpKind::Mul))
            .expect("SynthMlpPreDown claim contains Mul");
        vec![((mul_id, 0), None)]
    }

    fn opcode_shape(&self) -> OpcodeShape {
        // MUST stay byte-identical to
        // `interpreter_codegen::synth_mlp_pre_down_opcode_shape`.
        OpcodeShape::new(
            "SynthMlpPreDown",
            vec![
                ("residual_slot", syn::parse_quote!(u32)),
                ("delta_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("residual_out_slot", syn::parse_quote!(u32)),
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
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<ferrite_forward::Instruction>> {
        let mut add_tile: Option<TileId> = None;
        let mut rmsnorm_tile: Option<TileId> = None;
        let mut gemm_tiles: Vec<TileId> = Vec::new();
        let mut silu_tile: Option<TileId> = None;
        let mut mul_tile: Option<TileId> = None;
        for &t in &m.claimed_tiles {
            match fuf.get(t).op {
                OpKind::Add => add_tile = Some(t),
                OpKind::RmsNorm => rmsnorm_tile = Some(t),
                OpKind::Gemm => gemm_tiles.push(t),
                op if op == self.act_op => silu_tile = Some(t),
                OpKind::Mul => mul_tile = Some(t),
                _ => {}
            }
        }
        let add_tile = add_tile?;
        let rmsnorm_tile = rmsnorm_tile?;
        let silu_tile = silu_tile?;
        let mul_tile = mul_tile?;
        if gemm_tiles.len() != 2 {
            return None;
        }

        // residual_slot / delta_slot: same convention as
        // `FusedAddRmsNormImpl::fan_out` — `Add(delta, residual)`,
        // inputs[0]=delta, inputs[1]=residual.
        let add_node = fuf.get(add_tile);
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
        let residual_slot_idx = slots.of(residual.0, residual.1);
        let delta_slot_idx = slots.of(delta.0, delta.1);

        // gate = the Gemm consumed by Silu. up = the other Gemm
        // (Mul's other tile input). Matches the field-ordering
        // `apply_synth_replacement_mlp` produces (gate then up).
        let silu_in = fuf.get(silu_tile).inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } => Some(*id),
            _ => None,
        })?;
        if !gemm_tiles.contains(&silu_in) {
            return None;
        }
        let gate_tile = silu_in;
        let up_tile = *gemm_tiles.iter().find(|&&g| g != gate_tile)?;

        // out_slot = Mul's output. The downstream down-projection
        // reads from this slot.
        let out_slot_idx = slots.of(mul_tile, 0);

        // Layer index — pull from any per-layer weight input.
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

        let acc_for = |tile: TileId| -> Option<WeightAccessor> {
            default_required_weights(&[tile], fuf, program)
                .into_iter()
                .next()
        };
        let gate_acc = acc_for(gate_tile)?;
        let up_acc = acc_for(up_tile)?;
        let rms_acc = acc_for(rmsnorm_tile)?;

        let to_base = |acc: &WeightAccessor| -> syn::Ident {
            let (base, _layer) = split_base_layer(&acc.name.to_string());
            syn::Ident::new(&base, proc_macro2::Span::call_site())
        };
        let gate_base = to_base(&gate_acc);
        let up_base = to_base(&up_acc);
        let rms_base = to_base(&rms_acc);

        let (gs, bits) = match weight_storage_of(fuf.get(gate_tile)) {
            Some(StorageFormat::Affine { group_size, bits }) => (*group_size, *bits),
            _ => return None,
        };

        let symbol = self.symbol();
        let _ = bits;
        // Gate/up LinearLayers and RmsNorm flow through
        // `required_weights()`; codegen assigns sub-slots
        // 0/1 (Linear) and 0 (RmsNorm).
        let _ = (gate_base, up_base, rms_base);
        let kernel_symbol: &'static str = Box::leak(symbol.into_boxed_str());
        // residual_out: the updated residual (`residual_in + delta`) the
        // kernel writes — the `Add` tile's own output value. The coloring
        // gives it a slot distinct from `residual_slot_idx` (fused-
        // subgraph inputs stay live to the subgraph end), so the kernel
        // reads the input slot and writes this one — never in place.
        let residual_out_slot_idx = slots.of(add_tile, 0);
        Some(vec![ferrite_forward::Instruction::SynthMlpPreDown(
            residual_slot_idx,
            delta_slot_idx,
            out_slot_idx,
            residual_out_slot_idx,
            layer,
            gs,
            self.bits,
            kernel_symbol,
        )])
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        default_required_weights(claimed_tiles, fuf, program)
    }
}
