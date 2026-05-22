// SPDX-License-Identifier: Apache-2.0
//! Solver-driven whole-forward decode megakernel Impl.
//!
//! Claims the ENTIRE decode forward pass at the FUF level: every
//! layer's `(input_layernorm → q/k/v Gemms → RopeAppend → Attention
//! → o_proj Gemm → attn-residual Add → post_attention_layernorm →
//! gate/up Gemms → Silu → Mul → down_proj Gemm → mlp-residual Add)`
//! plus the trailing `(final RmsNorm → lm_head Gemm)`. The Embed at
//! the top of the forward stays outside the claim (its `[M, HIDDEN]`
//! output IS the persistent kernel's entry `__residual` buffer).
//!
//! `fan_out` emits one `Instruction::ForwardDecodePersistent`. The
//! metal lowering arm (see
//! `ferrite-forward/src/interpreter/metal/lowering.rs`) translates that
//! into a single LoweredCommand whose argument-buffer binding the
//! worker pre-builds at bake time (see
//! `build_per_layer_arg_buffer` in `worker.rs`).
//!
//! **Env-gated** via `FERRITE_PERSISTENT_FORWARD=1`. Without the gate
//! the Impl is `target_compatible: false` and never enters the solver
//! pool — production behavior is unchanged. With the gate on, the
//! cost is pinned to `1.0` so the solver picks this over the
//! per-phase synth siblings + per-op kernels that would otherwise
//! cover the same tiles. Decode-only (`WorkloadConstraint::
//! NumTokensRange { min: 1, max: 1 }`) since `synthesize_forward_decode`
//! bakes M=1 today.
//!
//! See [[persistent-decode-handoff]] for the kernel-side details
//! (`ferrite-fusion-synth::synthesize_forward_decode`) and the
//! Binding / lowering / worker pieces this Impl plugs into.
//!
//! ## Solver routing — `ClaimClass::Wide`
//!
//! The whole-forward claim spans ~243 tiles for Llama-3.2-1B, far
//! beyond the per-seed `ClaimMask::WINDOW` the Local bitmask DP can
//! represent. This Impl overrides `claim_class()` to
//! [`ClaimClass::Wide`], routing it through the solver's Wide lane:
//! at most one Wide candidate picks per workload point, and the
//! solver scores `wide.cost + constrained_local_dp(uncovered_tiles)`
//! against the baseline Local-only total before picking the cheaper.

use std::collections::{BTreeMap, HashSet};

use crate::classified::{OpKind, Program};
use crate::codegen::split_base_layer;
use crate::emit::weight_field_name;
use crate::fuf::{Fuf, FufInput, FufNode, TileId};
use crate::impl_lib::{
    consumes_tile, first_tile_input, rust_type_for_weight_consumed_by, ClaimClass, CostCtx,
    Handoff, Implementation, LaunchKind, Layout, MatchInfo, OpcodeShape, Resources, SlotMap,
    WeightAccessor, WorkloadConstraint,
};
use crate::quantization::StorageFormat;
use crate::target::{Backend, TargetProfile};

use std::collections::HashSet as StdHashSet;

/// Pull the storage format of `node`'s first weight input (if any).
/// Mirrors the local helpers in the sibling synth Impls.
fn weight_storage_of(node: &FufNode) -> Option<&StorageFormat> {
    for input in &node.inputs {
        if let FufInput::Weight { storage, .. } = input {
            return Some(storage);
        }
    }
    None
}

/// Whole-forward decode megakernel claim. One instance per
/// (act, scale, group_size, bits) tuple in the impl library; the
/// matcher gates on the lm_head Gemm's storage format so e.g. a
/// future int8 sibling would compete with int4 on quant-aware pick.
#[derive(Debug)]
pub struct MetalForwardDecodePersistentImpl {
    /// Activation dtype tag (`"bfloat"` or `"half"`) — must match the
    /// canonical's `W::METAL_DTYPE`. Used in the kernel symbol name
    /// emitted by `fan_out`; the AOT-embedded metallib bytes live at
    /// the same key in `synthesized_kernel_metallibs()`.
    pub act_tag: &'static str,
    /// Scale dtype tag — always `"half"` on the affine-int4 path
    /// today (F16 group scales).
    pub scale_tag: &'static str,
    /// Affine quant group size; must match what `LinearLayer::AffineQuant`
    /// ships with.
    pub group_size: u32,
    /// Affine quant bits; only 4 is wired today.
    pub bits: u32,
}

impl MetalForwardDecodePersistentImpl {
    /// Standard Llama-3.2 4bit shape: bf16 activations, half-precision
    /// group scales, gs=64, bits=4.
    pub fn bf16_gs64() -> Self {
        Self {
            act_tag: "bfloat",
            scale_tag: "half",
            group_size: 64,
            bits: 4,
        }
    }

    fn is_enabled() -> bool {
        std::env::var("FERRITE_PERSISTENT_FORWARD")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }
}

/// Walk a delta sub-tree (the non-residual operand of a residual Add)
/// and claim every tile reachable through `FufInput::Tile` edges.
/// Stops short of `residual_skip` (the previous-residual tile the
/// outer loop walks separately) and of any `Add` / `Embed` tile
/// (residual-chain anchors that belong to a different walk segment).
///
/// Deterministic-output by ordering claimed tiles before they leave
/// the matcher; the walk itself uses a `HashSet` for O(1) dedup.
fn walk_and_claim_delta(
    start: TileId,
    residual_skip: TileId,
    fuf: &Fuf,
    claimed: &mut HashSet<TileId>,
) {
    let mut stack = vec![start];
    while let Some(t) = stack.pop() {
        if t == residual_skip {
            continue;
        }
        let n = fuf.get(t);
        if matches!(n.op, OpKind::Add | OpKind::Embed) {
            continue;
        }
        if !claimed.insert(t) {
            continue;
        }
        for input in &n.inputs {
            if let FufInput::Tile { id, .. } = input {
                stack.push(*id);
            }
        }
    }
}

/// Identify the (residual, delta) pair in a residual-stream Add. The
/// residual input is itself an Add (interior layer's prior Add) or
/// the Embed (first layer's pre-attention Add). The other input is
/// the delta root — a Gemm in Llama-family forwards (o_proj for the
/// attn Add, down_proj for the mlp Add).
fn split_residual_delta(add_node: &FufNode, fuf: &Fuf) -> Option<(TileId, TileId)> {
    let tile_inputs: Vec<TileId> = add_node
        .inputs
        .iter()
        .filter_map(|i| match i {
            FufInput::Tile { id, .. } => Some(*id),
            _ => None,
        })
        .collect();
    if tile_inputs.len() != 2 {
        return None;
    }
    let a = tile_inputs[0];
    let b = tile_inputs[1];
    let a_op = fuf.get(a).op;
    let b_op = fuf.get(b).op;
    match (a_op, b_op) {
        (OpKind::Add | OpKind::Embed, _) => Some((a, b)),
        (_, OpKind::Add | OpKind::Embed) => Some((b, a)),
        _ => None,
    }
}

impl Implementation for MetalForwardDecodePersistentImpl {
    fn name(&self) -> &'static str {
        "metal_forward_decode_persistent"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        if profile.backend != Backend::Metal {
            return false;
        }
        Self::is_enabled()
    }

    fn claim_class(&self) -> ClaimClass {
        // Whole-forward claim spans ~243 tiles for Llama-3.2-1B; it
        // cannot fit `ClaimMask::WINDOW`. Routes through the solver's
        // Wide lane: at most one Wide candidate picks per workload
        // point, scored as `cost + constrained_local_dp(uncovered)`.
        ClaimClass::Wide
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        // Single-token decode only — synthesize_forward_decode pins
        // M=1 today (the in-kernel BN=8 attention atom is decode-shaped
        // + the residual buffer is M*HIDDEN sized for one token). Lift
        // when chunked-decode lands the M-axis loop kernel-side.
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        if !Self::is_enabled() || profile.backend != Backend::Metal {
            return None;
        }

        // Seed must be the unique terminal Gemm — the lm_head whose
        // output is consumed by no other tile. That gives us ONE
        // deterministic seeding point across the whole FUF (the solver
        // tries every tile as a seed; everything except lm_head's
        // Gemm fails the consumer check below).
        let seed_node = fuf.get(seed);
        if seed_node.op != OpKind::Gemm {
            return None;
        }
        // lm_head must be Affine-storage (matches the persistent
        // kernel's int4 lm_head phase).
        if !matches!(
            weight_storage_of(seed_node),
            Some(StorageFormat::Affine { group_size, bits })
                if *group_size == self.group_size && *bits == self.bits
        ) {
            return None;
        }
        if fuf.nodes.iter().any(|n| consumes_tile(n, seed)) {
            return None;
        }

        let mut claimed: HashSet<TileId> = HashSet::new();
        claimed.insert(seed); // lm_head Gemm

        // lm_head's tile input must be the final RmsNorm.
        let (final_rms_id, _) = first_tile_input(seed_node)?;
        let final_rms_node = fuf.get(final_rms_id);
        if final_rms_node.op != OpKind::RmsNorm {
            return None;
        }
        claimed.insert(final_rms_id);

        // Walk the residual chain backwards through `2 * num_layers`
        // Adds. Each layer contributes two: one after attention, one
        // after MLP. The matcher doesn't need num_layers passed in —
        // it counts as it goes and verifies the count is non-zero
        // and even.
        let (mut residual, _) = first_tile_input(final_rms_node)?;
        let mut num_adds: u32 = 0;
        loop {
            let n = fuf.get(residual);
            if matches!(n.op, OpKind::Embed) {
                break;
            }
            if n.op != OpKind::Add {
                // Unexpected residual-chain shape — bail out cleanly
                // rather than partial-claim the FUF.
                return None;
            }
            claimed.insert(residual);
            num_adds += 1;
            let (prev, delta) = split_residual_delta(n, fuf)?;
            walk_and_claim_delta(delta, prev, fuf, &mut claimed);
            residual = prev;
        }
        if num_adds == 0 || num_adds % 2 != 0 {
            return None;
        }

        // residual now points at the Embed tile (the loop's break).
        // Don't claim it — MetalEmbedImpl / MetalAffineEmbedImpl owns
        // it. Its output IS our entry `__residual` buffer.
        let embed_tile = residual;
        debug_assert_eq!(fuf.get(embed_tile).op, OpKind::Embed);

        let mut claimed_vec: Vec<TileId> = claimed.into_iter().collect();
        claimed_vec.sort();
        Some(MatchInfo {
            claimed_tiles: claimed_vec,
            boundary_inputs: vec![embed_tile],
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        // Env-gated explicit override — when FERRITE_PERSISTENT_FORWARD
        // is on, pin the cost to a constant well below any plausible
        // sum-of-per-phase alternative (16 layers × ~5 phases ×
        // ~50-200us per phase = ~4-16 ms baseline at decode). The
        // solver then picks this over the sibling synth + per-op
        // claimers covering the same tiles.
        //
        // Use 1.0 rather than 0.0 so the cost histogram doesn't
        // confuse this with an uninitialized entry.
        1.0
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        // 256 threads/TG paired with the BN=8 in-kernel attention
        // atom (synth pins NUM_SIMDGROUPS=8). TG-mem dominated by the
        // HIDDEN-wide __x_norm staging buffer + HEAD_DIM-wide
        // __qmv_smem / __gate_smem / __up_smem (one each). At
        // Llama-3.2-1B (HIDDEN=2048, HEAD_DIM=64) this is
        // ~2048*2 + 64*4*3 ≈ 5 KiB; round to 16 KiB headroom for the
        // 3B + Qwen2 shapes.
        Resources {
            shmem_bytes: 16 * 1024,
            regs_per_thread: 64,
            threads_per_cta: 256,
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
        // Only the lm_head Gemm's output (logits) is externally
        // consumed. Internal residual / scratch slots are kernel-private;
        // the slot allocator handles them through the boundary inputs.
        let lm_head = *claimed_tiles
            .iter()
            .find(|&&t| {
                let n = fuf.get(t);
                n.op == OpKind::Gemm && !fuf.nodes.iter().any(|m| consumes_tile(m, t))
            })
            .expect("ForwardDecodePersistent claim contains a terminal lm_head Gemm");
        vec![((lm_head, 0), None)]
    }

    fn opcode_shape(&self) -> OpcodeShape {
        // MUST stay byte-identical to the 9-tuple
        // `Instruction::ForwardDecodePersistent` shape baked into
        // `interpreter_codegen` (see the `I::ForwardDecodePersistent`
        // arms in `interpreter_codegen.rs` around lines 733 / 1512 /
        // 2179).
        OpcodeShape::new(
            "ForwardDecodePersistent",
            vec![
                ("residual_slot", syn::parse_quote!(u32)),
                ("q_scratch_slot", syn::parse_quote!(u32)),
                ("attn_scratch_slot", syn::parse_quote!(u32)),
                ("mlp_scratch_slot", syn::parse_quote!(u32)),
                ("logits_out_slot", syn::parse_quote!(u32)),
                ("num_layers", syn::parse_quote!(u32)),
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
        _program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<ferrite_forward::Instruction>> {
        // residual_slot: Embed output (= boundary_inputs[0]). The
        // persistent kernel reads this as `__residual` at entry and
        // updates it in place across all layers.
        let embed_tile = *m.boundary_inputs.first()?;
        let residual_slot = slots.of(embed_tile, 0);

        // logits_out_slot: the terminal lm_head Gemm's output slot.
        // NOTE: `boundary_outputs[0]` is NOT necessarily the lm_head
        // Gemm — codegen invokes terminal-impl fan_out with
        // `boundary_outputs = tiles_in_subgraph(terminal_sg)` (the
        // entire claimed subgraph, not just the matcher's seed), and
        // for our WIDE claim that list's first element is layer 0's
        // RmsNorm. Find the terminal Gemm explicitly — the same way
        // `output_alias` already does — so the slot we report matches
        // `codegen::terminal_slot = slots.of(<lm_head Gemm>, 0)`. If
        // they disagree, the sampler reads `arena[terminal_slot]`
        // while the kernel writes `arena[logits_out_slot]` and the
        // model produces stale logits.
        let lm_head_tile = *m
            .claimed_tiles
            .iter()
            .find(|&&t| {
                let n = fuf.get(t);
                n.op == OpKind::Gemm && !fuf.nodes.iter().any(|m| consumes_tile(m, t))
            })?;
        let logits_out_slot = slots.of(lm_head_tile, 0);

        // q_scratch / attn_scratch / mlp_scratch are cross-layer
        // scratch areas that the linear-scan slot allocator already
        // coalesces (each layer's `q_proj` Gemm output dies before the
        // next layer's runs, so layer-0's slot == layer-1's slot ==
        // ... == one physical buffer in the arena). Pick the
        // representative tile from the claimed set; the slot id we
        // resolve here is the shared color the persistent kernel
        // reuses across every layer.
        let q_repr = pick_role_tile(&m.claimed_tiles, fuf, |n| {
            n.op == OpKind::Gemm && is_q_proj_role(n.id, fuf)
        })?;
        let q_scratch_slot = slots.of(q_repr, 0);

        let attn_repr = pick_role_tile(&m.claimed_tiles, fuf, |n| {
            n.op == OpKind::Gemm && is_o_proj_role(n, fuf)
        })?;
        let attn_scratch_slot = slots.of(attn_repr, 0);

        let mlp_repr =
            pick_role_tile(&m.claimed_tiles, fuf, |n| n.op == OpKind::Mul)?;
        let mlp_scratch_slot = slots.of(mlp_repr, 0);

        let num_layers = *bounds.get("num_hidden_layers")? as u32;
        if num_layers == 0 {
            return None;
        }
        let head_dim = *bounds.get("head_dim")? as u32;
        if head_dim == 0 || head_dim % 64 != 0 {
            return None;
        }

        // Symbol must match the synth output. V1 vs V2 swap is keyed on
        // FERRITE_PD_V2 env at proc-macro time (same key the codegen uses).
        let use_v2 = std::env::var("FERRITE_PD_V2")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let v2_threads: u32 = std::env::var("FERRITE_PD_V2_THREADS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(256);
        let symbol = if use_v2 {
            format!(
                "forward_decode_persistent_v2_{}_{}_gs{}_hd{}_t{}_L{}",
                self.act_tag, self.scale_tag, self.group_size, head_dim, v2_threads, num_layers,
            )
        } else {
            format!(
                "forward_decode_persistent_{}_{}_gs{}_hd{}_t256_L{}",
                self.act_tag, self.scale_tag, self.group_size, head_dim, num_layers,
            )
        };
        let kernel_symbol: &'static str = Box::leak(symbol.into_boxed_str());

        Some(vec![ferrite_forward::Instruction::ForwardDecodePersistent(
            residual_slot,
            q_scratch_slot,
            attn_scratch_slot,
            mlp_scratch_slot,
            logits_out_slot,
            num_layers,
            self.group_size,
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
        // **NOT** `default_required_weights`. The default walker emits
        // one accessor per `(WeightId, index)` pair, so a 16-layer
        // claim returns 16 × N per-layer entries (`input_layernorm_0`,
        // `input_layernorm_1`, …) interleaved by layer's position
        // through the FUF. The macro's accessor codegen counts slots
        // per `(op_idx, kind)`, so those 16 entries become slots
        // 0..15 of `RmsNorm` at op_idx 0 — even though every entry
        // calls the same loop-compressed `self.input_layernorm(layer)`
        // method. The metal lowering arm assumes ROLES map to FIXED
        // slot ids (RmsNorm slot 0 = input_ln, slot 1 = postattn_ln,
        // slot 2 = model_norm; Linear slot 0..6 = q/k/v/o/g/u/d, slot
        // 7 = lm_head). A per-layer interleaving makes slot 2 hit
        // `input_layernorm` (layer-1's instance) instead of
        // `model_norm`, and the worker resolves the wrong weight.
        //
        // Walk in FUF order (claimed_tiles arrives sorted by TileId =
        // FUF position) and dedupe by `split_base_layer(name).0` so
        // each ROLE contributes exactly one accessor in role-first
        // order — Llama's per-block dataflow naturally produces the
        // role sequence the lowering arm expects:
        //
        //   input_ln (RmsNorm slot 0)
        //   q_proj, k_proj, v_proj (Linear slots 0..2)
        //   o_proj (Linear slot 3)
        //   postattn_ln (RmsNorm slot 1)
        //   gate, up, down (Linear slots 4..6)
        //   model.norm (RmsNorm slot 2)
        //   lm_head (Linear slot 7)
        //
        // Per-layer accessor families (`input_layernorm_0..15`) all
        // collapse to one slot since they share the base; the macro's
        // base-collapse via `split_base_layer` in
        // `weight_accessors_to_slots` then routes runtime calls
        // through one `self.<base>(layer)` method.
        let mut out: Vec<WeightAccessor> = Vec::new();
        let mut seen_bases: StdHashSet<String> = StdHashSet::new();
        for &tid in claimed_tiles {
            let node = fuf.get(tid);
            for input in &node.inputs {
                if let FufInput::Weight { id, index, .. } = input {
                    let name = weight_field_name(program, *id, *index);
                    let (base, _layer) = split_base_layer(&name.to_string());
                    if !seen_bases.insert(base) {
                        continue;
                    }
                    out.push(WeightAccessor {
                        name,
                        rust_type: rust_type_for_weight_consumed_by(node.op),
                        source_weights: vec![(*id, *index)],
                    });
                }
            }
        }
        out
    }
}

/// Find a tile in `claimed_tiles` matching `pred`. Returns the first
/// hit since the slot allocator coalesces every layer's instances of
/// the same role into one color — so "any layer's q_proj Gemm" is
/// equivalent for slot-resolution purposes.
fn pick_role_tile<F: Fn(&FufNode) -> bool>(
    claimed_tiles: &[TileId],
    fuf: &Fuf,
    pred: F,
) -> Option<TileId> {
    claimed_tiles
        .iter()
        .copied()
        .find(|&t| pred(fuf.get(t)))
}

/// True iff tile `t` is consumed at the Q-slot of some `RopeAppend`
/// (or `RopeAppendInterleaved`). The Q-projection Gemm in the FUF is
/// distinguished from K/V by its position in RopeAppend's first
/// tile-input slot.
fn is_q_proj_role(t: TileId, fuf: &Fuf) -> bool {
    fuf.nodes.iter().any(|n| {
        if !matches!(n.op, OpKind::RopeAppend | OpKind::RopeAppendInterleaved) {
            return false;
        }
        let first_tile = n.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } => Some(*id),
            _ => None,
        });
        first_tile == Some(t)
    })
}

/// True iff Gemm `n` is the o_proj — its tile input is an Attention
/// (or VarlenAttention/SlidingAttention sibling) output.
fn is_o_proj_role(n: &FufNode, fuf: &Fuf) -> bool {
    n.inputs.iter().any(|i| {
        if let FufInput::Tile { id, .. } = i {
            matches!(
                fuf.get(*id).op,
                OpKind::Attention | OpKind::SlidingAttention | OpKind::VarlenAttention
            )
        } else {
            false
        }
    })
}
