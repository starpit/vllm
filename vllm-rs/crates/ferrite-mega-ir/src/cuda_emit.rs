// SPDX-License-Identifier: Apache-2.0
//! Phase C step 2 — purely syntactic emit from `MegaTape` to CUDA
//! source.
//!
//! Per `MEGA_IR_PLAN.md` §5+§8.4, this is a single
//! `lower_to_cuda(tape: &MegaTape, …) -> CuVariant` function that
//! pattern-matches the typed tape. No per-op dispatcher, no
//! role-string accumulator, no four-string `WalkerLines` struct (the
//! retired walker naming is forbidden).
//!
//! Step 1 (already shipping in 605/659 canonicals) emits
//! `build_mega_tape_<canonical>()` Rust fns whose literal-const-arg
//! `MegaTapeBuilder::push_*::<...>(...)` calls discharge the
//! substrate-proof const-asserts at user-build time. By the time the
//! `MegaTape` value is in hand, every load-bearing field has already
//! been validated; emit only has to splice numbers and
//! string/path/enum helper newtypes into a `.cu` source skeleton.
//!
//! ## Sprint sequencing
//!
//! Per `MEGA_IR_PLAN.md` §10, Phase C step 2 lands sprint by sprint:
//!
//! | Sprint | Variant | Done = |
//! |---|---|---|
//! | A | `RmsNorm` | E2E coherent |
//! | B | `FusedQkvRopeCache` | E2E coherent |
//! | C | `FusedAddRmsNorm`, `FusedGateUpActivateMul`, `FusedCublasGemmAdd`, … | E2E coherent |
//! | D | remaining variants + lm_head fusions + barriers + sliding attn + scalar/offset/softcap | E2E coherent |
//!
//! This module's first commit is the *scaffold* — every variant has
//! a placeholder body that emits the typed-getter values as a
//! comment. Subsequent sprints replace placeholders with real
//! per-variant CUDA. The scaffold's purpose is to prove the
//! `MegaTape -> CuVariant` plumbing, not to produce coherent
//! output.
//!
//! ## ABI tier inference
//!
//! The emitted launcher signature mirrors one of the three runtime
//! ABI tiers in
//! `ferrite-forward::interpreter::mega::{LaunchFn,LaunchFnQkv,LaunchFnAttn}`:
//!
//! - `Attn` — any `AttentionViaCache` node in the tape. Layers
//!   `seq_lens` + `block_table` + `block_table_stride` on top of the
//!   QKV prefix. Implies QKV.
//! - `Qkv` — any `FusedQkvRopeCache` node in the tape. Layers
//!   `input_ids` + `positions` + `slot_mapping` + `key_cache_ptrs`
//!   + `value_cache_ptrs` on top of the base prefix.
//! - `Base` — neither of the above. Just `act_ptrs` + `weight_ptrs`
//!   + `barriers` + `trace_level`.
//!
//! Inference is conservative: presence of an `AttentionViaCache`
//! anywhere in the tape promotes the canonical to `Attn`, even if
//! some other tier could in principle suffice. Matches
//! `MEGA_IR_PLAN.md` §3 helper-newtype rule — runtime-validated
//! at-most-one summarization, not field-validity scaffolding.

#![allow(dead_code)]

use crate::nodes::{
    Add, AttentionKind, AttentionViaCacheNode, BarrierSignal, BarrierWait, CutlassFusedNormGemm,
    Embed, FusedAddRmsNorm, FusedCublasGemmAdd, FusedGateUpActivateMul, FusedQkvRopeCache,
    GateUpActivation, Gemm, LmHeadNormKind, MegaNode, RmsNorm, ScalarMul, ScalarOffsetRmsNorm,
    SpliceMmEmbeds, TanhSoftCap,
};
use crate::tape::MegaTape;

/// Which positional `extern "C" ferrite_<canonical>_launch` ABI the
/// emitted `.cu` exposes. Mirrors the runtime tier tags in
/// `ferrite-forward::interpreter::mega`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CuAbiTier {
    Base,
    Qkv,
    Attn,
}

impl CuAbiTier {
    /// Stable string spelling for embedding in source comments and
    /// downstream cudaforge-cache keys.
    pub const fn as_str(self) -> &'static str {
        match self {
            CuAbiTier::Base => "Base",
            CuAbiTier::Qkv => "Qkv",
            CuAbiTier::Attn => "Attn",
        }
    }
}

/// Substrate budget + dispatch counts that the emit step needs to
/// stamp into the .cu's banner / launcher signature.
///
/// Values come from the same proc-macro-time substrate budget that
/// `emit_canonical_build_fn` uses to instantiate `MegaTapeBuilder`'s
/// const generics — every field here is what the user-build's
/// `build_mega_tape_<canonical>` was monomorphized against, so the
/// emitted `.cu` is structurally consistent with the substrate
/// proofs that fired at user-build time.
#[derive(Clone, Copy, Debug)]
pub struct CuLowerCtx {
    pub num_pages: u32,
    pub num_consumer_warps: u32,
    pub page_size: u32,
    pub scratch_bytes: u32,
    pub num_layers: u32,
    pub num_edges: u32,
    /// Number of activation slots the host stages into `act_ptrs`.
    /// Recorded in the `.cu` banner so cudaforge / runtime sizing
    /// matches the codegen-time count. Computed by the caller from
    /// the canonical's slot map; the emit step doesn't infer it.
    pub num_act_slots: u32,
    /// Number of weight accessors the host stages into `weight_ptrs`
    /// (across all layers). Recorded in the `.cu` banner.
    pub num_weight_accessors: u32,
}

/// Output of `lower_to_cuda` for one canonical.
///
/// `source` is a complete `.cu` file containing:
///   - file banner with substrate budget + ABI tier + variant counts,
///   - `extern "C"` launcher signature for the inferred tier,
///   - per-node placeholder body (sprint A onward replaces these
///     with real CUDA per the §10 sprint table).
///
/// The downstream cudaforge integration consumes `(source,
/// canonical_name, abi_tier)` to stage one `.cu` per canonical and
/// link a per-variant `.a`. That wiring is the next commit-unit per
/// the handoff sequence ("template wired, first canonical CUDA
/// byte-identical, MEGA_FORWARD_TABLE populated, runtime dispatch
/// live").
#[derive(Clone, Debug)]
pub struct CuVariant {
    pub canonical_name: String,
    pub source: String,
    pub abi_tier: CuAbiTier,
    pub num_act_slots: u32,
    pub num_weight_accessors: u32,
    pub num_layers: u32,
    pub num_edges: u32,
}

/// Pattern-match `tape.nodes()` and produce a `.cu` source
/// `CuVariant`. The function is the entire emit step — every
/// per-variant body is inlined into one match expression here,
/// satisfying §8.4's "no per-op dispatcher, no role-string
/// accumulator" rule.
///
/// Phase C step 2 SCAFFOLD: each match arm emits a
/// `// MEGA_NODE: <Variant> { … }` comment line with the typed
/// getter values, plus a `// TODO(sprint X)` marker pointing at the
/// §10 sprint that owns the variant's real CUDA. The scaffold
/// commit's job is to prove the `MegaTape -> .cu` plumbing; sprints
/// A-D replace placeholder bodies with real per-variant CUDA without
/// touching the surrounding harness.
pub fn lower_to_cuda(tape: &MegaTape, canonical_name: &str, ctx: &CuLowerCtx) -> CuVariant {
    let abi_tier = infer_abi_tier(tape);
    let mut body = String::with_capacity(8 * 1024);
    for (idx, node) in tape.nodes().iter().enumerate() {
        body.push_str(&emit_node_placeholder(idx, node));
    }
    let source = render_cu_source(canonical_name, abi_tier, ctx, &body);
    CuVariant {
        canonical_name: canonical_name.to_string(),
        source,
        abi_tier,
        num_act_slots: ctx.num_act_slots,
        num_weight_accessors: ctx.num_weight_accessors,
        num_layers: ctx.num_layers,
        num_edges: ctx.num_edges,
    }
}

fn infer_abi_tier(tape: &MegaTape) -> CuAbiTier {
    let mut has_attention = false;
    let mut has_qkv = false;
    for node in tape.nodes() {
        match node {
            MegaNode::AttentionViaCache(_) => has_attention = true,
            MegaNode::FusedQkvRopeCache(_) => has_qkv = true,
            _ => {}
        }
    }
    if has_attention {
        CuAbiTier::Attn
    } else if has_qkv {
        CuAbiTier::Qkv
    } else {
        CuAbiTier::Base
    }
}

fn render_cu_source(
    canonical_name: &str,
    abi_tier: CuAbiTier,
    ctx: &CuLowerCtx,
    body: &str,
) -> String {
    let signature = match abi_tier {
        CuAbiTier::Base => render_base_signature(canonical_name),
        CuAbiTier::Qkv => render_qkv_signature(canonical_name),
        CuAbiTier::Attn => render_attn_signature(canonical_name),
    };
    let mut s = String::with_capacity(body.len() + 4 * 1024);
    s.push_str("// SPDX-License-Identifier: Apache-2.0\n");
    s.push_str("// Generated by ferrite_mega_ir::cuda_emit::lower_to_cuda — do NOT edit.\n");
    s.push_str("// Phase C step 2 scaffold: per-variant body is a placeholder; sprints A-D\n");
    s.push_str("// replace placeholders with real CUDA. See MEGA_IR_PLAN.md §10.\n");
    s.push_str("//\n");
    s.push_str(&format!("// canonical:            {canonical_name}\n"));
    s.push_str(&format!("// abi_tier:             {}\n", abi_tier.as_str()));
    s.push_str(&format!("// num_pages:            {}\n", ctx.num_pages));
    s.push_str(&format!(
        "// num_consumer_warps:   {}\n",
        ctx.num_consumer_warps
    ));
    s.push_str(&format!("// page_size:            {}\n", ctx.page_size));
    s.push_str(&format!(
        "// scratch_bytes:        {}\n",
        ctx.scratch_bytes
    ));
    s.push_str(&format!("// num_layers:           {}\n", ctx.num_layers));
    s.push_str(&format!("// num_edges:            {}\n", ctx.num_edges));
    s.push_str(&format!(
        "// num_act_slots:        {}\n",
        ctx.num_act_slots
    ));
    s.push_str(&format!(
        "// num_weight_accessors: {}\n",
        ctx.num_weight_accessors
    ));
    s.push('\n');
    s.push_str("#include <cuda_runtime.h>\n");
    s.push('\n');
    s.push_str(&signature);
    s.push_str("{\n");
    s.push_str("    // ===== Phase C step 2 placeholder body =====\n");
    s.push_str("    // The four-warp-role mega kernel (loader/launcher/consumer/storer)\n");
    s.push_str("    // wires up over the per-variant `.cuh` includes (see\n");
    s.push_str("    // crates/ferrite-kernels/csrc/tk/ferrite_kernels/) once the\n");
    s.push_str("    // matching sprint lands. Today this scaffold returns success so\n");
    s.push_str("    // the canonical's linker symbol resolves; the host gates\n");
    s.push_str("    // dispatch via `MEGA_FORWARD_TABLE` (still empty per Phase C\n");
    s.push_str("    // step 3) — the runtime never actually invokes this scaffold.\n");
    s.push_str(body);
    s.push_str("    return cudaSuccess;\n");
    s.push_str("}\n");
    s
}

fn render_base_signature(canonical_name: &str) -> String {
    format!(
        "extern \"C\" cudaError_t ferrite_{canonical_name}_launch(\n\
         \x20   __nv_bfloat16* const*       act_ptrs,\n\
         \x20   const __nv_bfloat16* const* weight_ptrs,\n\
         \x20   int*                        barriers,\n\
         \x20   int                         trace_level,\n\
         \x20   cudaStream_t                stream)\n"
    )
}

fn render_qkv_signature(canonical_name: &str) -> String {
    format!(
        "extern \"C\" cudaError_t ferrite_{canonical_name}_launch(\n\
         \x20   __nv_bfloat16* const*       act_ptrs,\n\
         \x20   const __nv_bfloat16* const* weight_ptrs,\n\
         \x20   const uint32_t*             input_ids,\n\
         \x20   const uint32_t*             positions,\n\
         \x20   const int64_t*              slot_mapping,\n\
         \x20   const __nv_bfloat16* const* key_cache_ptrs,\n\
         \x20   const __nv_bfloat16* const* value_cache_ptrs,\n\
         \x20   int*                        barriers,\n\
         \x20   int                         trace_level,\n\
         \x20   cudaStream_t                stream)\n"
    )
}

fn render_attn_signature(canonical_name: &str) -> String {
    format!(
        "extern \"C\" cudaError_t ferrite_{canonical_name}_launch(\n\
         \x20   __nv_bfloat16* const*       act_ptrs,\n\
         \x20   const __nv_bfloat16* const* weight_ptrs,\n\
         \x20   const uint32_t*             input_ids,\n\
         \x20   const uint32_t*             positions,\n\
         \x20   const int64_t*              slot_mapping,\n\
         \x20   const __nv_bfloat16* const* key_cache_ptrs,\n\
         \x20   const __nv_bfloat16* const* value_cache_ptrs,\n\
         \x20   const int32_t*              seq_lens,\n\
         \x20   const uint32_t*             block_table,\n\
         \x20   uint32_t                    block_table_stride,\n\
         \x20   int*                        barriers,\n\
         \x20   int                         trace_level,\n\
         \x20   cudaStream_t                stream)\n"
    )
}

fn emit_node_placeholder(idx: usize, node: &MegaNode) -> String {
    match node {
        MegaNode::RmsNorm(n) => emit_rms_norm(idx, n),
        MegaNode::FusedQkvRopeCache(n) => emit_fused_qkv_rope_cache(idx, n),
        MegaNode::Add(n) => emit_add(idx, n),
        MegaNode::FusedAddRmsNorm(n) => emit_fused_add_rms_norm(idx, n),
        MegaNode::FusedGateUpActivateMul(n) => emit_fused_gate_up_activate_mul(idx, n),
        MegaNode::Embed(n) => emit_embed(idx, n),
        MegaNode::ScalarMul(n) => emit_scalar_mul(idx, n),
        MegaNode::TanhSoftCap(n) => emit_tanh_soft_cap(idx, n),
        MegaNode::ScalarOffsetRmsNorm(n) => emit_scalar_offset_rms_norm(idx, n),
        MegaNode::Gemm(n) => emit_gemm(idx, n),
        MegaNode::FusedCublasGemmAdd(n) => emit_fused_cublas_gemm_add(idx, n),
        MegaNode::CutlassFusedNormGemm(n) => emit_cutlass_fused_norm_gemm(idx, n),
        MegaNode::AttentionViaCache(n) => emit_attention_via_cache(idx, n),
        MegaNode::BarrierSignal(n) => emit_barrier_signal(idx, n),
        MegaNode::BarrierWait(n) => emit_barrier_wait(idx, n),
        MegaNode::SpliceMmEmbeds(n) => emit_splice_mm_embeds(idx, n),
    }
}

fn emit_rms_norm(idx: usize, n: &RmsNorm) -> String {
    format!(
        "    // [{idx}] RmsNorm {{ in_page={} weight_page={} partial=[{}+{}B] phase=cs/{}/{} layer={} weight={:?} }} // TODO(sprint A)\n",
        n.in_page_id(),
        n.weight_page_id(),
        n.partial_offset(),
        n.partial_bytes(),
        n.consumer_phase(),
        n.storer_phase(),
        n.layer(),
        n.weight.path(),
    )
}

fn emit_fused_qkv_rope_cache(idx: usize, n: &FusedQkvRopeCache) -> String {
    format!(
        "    // [{idx}] FusedQkvRopeCache {{ in={} qkv={} cos_sin={} q={} k={} v={} q_rope=[{}+{}B] k_rope=[{}+{}B] phase=cs/{}/{} iters={} layer={} biased={} interleaved={} weight={:?} rotary={:?} }} // TODO(sprint B)\n",
        n.in_page_id(),
        n.qkv_weight_page_id(),
        n.cos_sin_page_id(),
        n.q_out_page_id(),
        n.k_out_page_id(),
        n.v_out_page_id(),
        n.q_rope_offset(),
        n.q_rope_bytes(),
        n.k_rope_offset(),
        n.k_rope_bytes(),
        n.consumer_phase(),
        n.storer_phase(),
        n.iters(),
        n.layer(),
        n.biased,
        n.interleaved,
        n.qkv_weight.path(),
        n.rotary.path(),
    )
}

fn emit_add(idx: usize, n: &Add) -> String {
    format!(
        "    // [{idx}] Add {{ delta={} residual={} phase=cs/{}/{} }} // TODO(sprint C)\n",
        n.delta_page_id(),
        n.residual_page_id(),
        n.consumer_phase(),
        n.storer_phase(),
    )
}

fn emit_fused_add_rms_norm(idx: usize, n: &FusedAddRmsNorm) -> String {
    format!(
        "    // [{idx}] FusedAddRmsNorm {{ delta={} residual={} weight_page={} partial=[{}+{}B] phase=cs/{}/{} layer={} weight={:?} }} // TODO(sprint C)\n",
        n.delta_page_id(),
        n.residual_page_id(),
        n.weight_page_id(),
        n.partial_offset(),
        n.partial_bytes(),
        n.consumer_phase(),
        n.storer_phase(),
        n.layer(),
        n.weight.path(),
    )
}

fn emit_fused_gate_up_activate_mul(idx: usize, n: &FusedGateUpActivateMul) -> String {
    let act = match n.activation {
        GateUpActivation::Silu => "Silu",
        GateUpActivation::Gelu => "Gelu",
    };
    format!(
        "    // [{idx}] FusedGateUp{act}Mul {{ in={} gate_up={} out={} gate=[{}+{}B] up=[{}+{}B] phase=cs/{}/{} iters={} layer={} weight={:?} }} // TODO(sprint C)\n",
        n.in_page_id(),
        n.gate_up_weight_page_id(),
        n.out_page_id(),
        n.gate_offset(),
        n.gate_bytes(),
        n.up_offset(),
        n.up_bytes(),
        n.consumer_phase(),
        n.storer_phase(),
        n.iters(),
        n.layer(),
        n.weight.path(),
    )
}

fn emit_embed(idx: usize, n: &Embed) -> String {
    format!(
        "    // [{idx}] Embed {{ out={} weight_page={} phase=cs/{}/{} weight={:?} }} // TODO(sprint D)\n",
        n.out_page_id(),
        n.embed_weight_page_id(),
        n.consumer_phase(),
        n.storer_phase(),
        n.embed_weight.path(),
    )
}

fn emit_scalar_mul(idx: usize, n: &ScalarMul) -> String {
    format!(
        "    // [{idx}] ScalarMul {{ in={} out={} phase=cs/{}/{} scale={} }} // TODO(sprint D)\n",
        n.in_page_id(),
        n.out_page_id(),
        n.consumer_phase(),
        n.storer_phase(),
        n.scale.raw(),
    )
}

fn emit_tanh_soft_cap(idx: usize, n: &TanhSoftCap) -> String {
    format!(
        "    // [{idx}] TanhSoftCap {{ in={} out={} phase=cs/{}/{} }} // TODO(sprint D)\n",
        n.in_page_id(),
        n.out_page_id(),
        n.consumer_phase(),
        n.storer_phase(),
    )
}

fn emit_scalar_offset_rms_norm(idx: usize, n: &ScalarOffsetRmsNorm) -> String {
    format!(
        "    // [{idx}] ScalarOffsetRmsNorm {{ in={} weight_page={} partial=[{}+{}B] phase=cs/{}/{} layer={} weight={:?} offset={} }} // TODO(sprint D)\n",
        n.in_page_id(),
        n.weight_page_id(),
        n.partial_offset(),
        n.partial_bytes(),
        n.consumer_phase(),
        n.storer_phase(),
        n.layer(),
        n.weight.path(),
        n.offset.raw(),
    )
}

fn emit_gemm(idx: usize, n: &Gemm) -> String {
    format!(
        "    // [{idx}] Gemm {{ in={} weight_page={} out={} b_tile=[{}+{}B] phase=cs/{}/{} iters={} layer={} n={} k={} weight={:?} }} // TODO(sprint D)\n",
        n.in_page_id(),
        n.weight_page_id(),
        n.out_page_id(),
        n.b_tile_offset(),
        n.b_tile_bytes(),
        n.consumer_phase(),
        n.storer_phase(),
        n.iters(),
        n.layer(),
        n.n(),
        n.k(),
        n.weight.path(),
    )
}

fn emit_fused_cublas_gemm_add(idx: usize, n: &FusedCublasGemmAdd) -> String {
    format!(
        "    // [{idx}] FusedCublasGemmAdd {{ in={} weight_page={} residual={} b_tile=[{}+{}B] phase=cs/{}/{} iters={} layer={} n={} k={} weight={:?} }} // TODO(sprint C)\n",
        n.in_page_id(),
        n.weight_page_id(),
        n.residual_page_id(),
        n.b_tile_offset(),
        n.b_tile_bytes(),
        n.consumer_phase(),
        n.storer_phase(),
        n.iters(),
        n.layer(),
        n.n(),
        n.k(),
        n.weight.path(),
    )
}

fn emit_cutlass_fused_norm_gemm(idx: usize, n: &CutlassFusedNormGemm) -> String {
    let kind = match n.norm_kind {
        LmHeadNormKind::RmsNorm => "RmsNorm",
        LmHeadNormKind::AddRmsNorm => "AddRmsNorm",
        LmHeadNormKind::AddScalarOffsetRmsNorm => "AddScalarOffsetRmsNorm",
        LmHeadNormKind::MeanSubRmsNorm => "MeanSubRmsNorm",
    };
    let delta_str = match n.delta_page_id() {
        Some(d) => format!("delta={d} "),
        None => String::new(),
    };
    let offset_str = match n.offset {
        Some(o) => format!("offset={} ", o.raw()),
        None => String::new(),
    };
    format!(
        "    // [{idx}] CutlassFused{kind}Gemm {{ in={} {delta_str}norm={} linear={} out={} partial=[{}+{}B] b_tile=[{}+{}B] phase=cs/{}/{} iters={} layer={} n={} k={} {offset_str}norm_w={:?} lin_w={:?} }} // TODO(sprint D)\n",
        n.in_page_id(),
        n.norm_weight_page_id(),
        n.linear_weight_page_id(),
        n.out_page_id(),
        n.partial_offset(),
        n.partial_bytes(),
        n.b_tile_offset(),
        n.b_tile_bytes(),
        n.consumer_phase(),
        n.storer_phase(),
        n.iters(),
        n.layer(),
        n.n(),
        n.k(),
        n.norm_weight.path(),
        n.linear_weight.path(),
    )
}

fn emit_attention_via_cache(idx: usize, n: &AttentionViaCacheNode) -> String {
    let kind = match n.kind {
        AttentionKind::Full => "Full".to_string(),
        AttentionKind::Sliding(w) => format!("Sliding(window={w})"),
    };
    format!(
        "    // [{idx}] AttentionViaCache {{ q_in={} attn_out={} score=[{}+{}B] pv=[{}+{}B] phase=cs/{}/{} iters={} layer={} kind={kind} interleaved={} }} // TODO(sprint D)\n",
        n.q_in_page_id(),
        n.attn_out_page_id(),
        n.score_offset(),
        n.score_bytes(),
        n.pv_offset(),
        n.pv_bytes(),
        n.consumer_phase(),
        n.storer_phase(),
        n.iters(),
        n.kv_cache_layer(),
        n.interleaved,
    )
}

fn emit_barrier_signal(idx: usize, n: &BarrierSignal) -> String {
    format!(
        "    // [{idx}] BarrierSignal {{ edge={} }} // TODO(sprint D)\n",
        n.edge(),
    )
}

fn emit_barrier_wait(idx: usize, n: &BarrierWait) -> String {
    format!(
        "    // [{idx}] BarrierWait {{ edge={} expected={} }} // TODO(sprint D)\n",
        n.edge(),
        n.expected(),
    )
}

fn emit_splice_mm_embeds(idx: usize, n: &SpliceMmEmbeds) -> String {
    format!(
        "    // [{idx}] SpliceMmEmbeds {{ slot_id={} phase=cs/{}/{} }} // TODO(sprint D)\n",
        n.slot_id(),
        n.consumer_phase(),
        n.storer_phase(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodes::{FiniteF32, RotaryRef, WeightRef};

    fn ctx() -> CuLowerCtx {
        CuLowerCtx {
            num_pages: 32,
            num_consumer_warps: 8,
            page_size: 32_768,
            scratch_bytes: 32_768,
            num_layers: 32,
            num_edges: 0,
            num_act_slots: 16,
            num_weight_accessors: 9,
        }
    }

    #[test]
    fn empty_tape_emits_base_tier_with_banner() {
        let tape = MegaTape::__build_from_nodes(Vec::new());
        let v = lower_to_cuda(&tape, "test_empty", &ctx());
        assert_eq!(v.canonical_name, "test_empty");
        assert_eq!(v.abi_tier, CuAbiTier::Base);
        assert!(v.source.contains("ferrite_test_empty_launch"));
        assert!(v.source.contains("// canonical:            test_empty"));
        assert!(v.source.contains("// abi_tier:             Base"));
        assert!(v.source.contains("return cudaSuccess"));
    }

    #[test]
    fn rms_norm_node_emits_typed_getter_values() {
        let n = RmsNorm::__new_for_emit(
            0,
            1,
            0,
            1024,
            0,
            1,
            5,
            WeightRef::new("model.layers.5.input_layernorm.weight".to_string()),
        );
        let tape = MegaTape::__build_from_nodes(vec![MegaNode::RmsNorm(n)]);
        let v = lower_to_cuda(&tape, "test_rms", &ctx());
        assert_eq!(v.abi_tier, CuAbiTier::Base);
        assert!(v.source.contains("RmsNorm {"));
        assert!(v.source.contains("in_page=0"));
        assert!(v.source.contains("weight_page=1"));
        assert!(v.source.contains("partial=[0+1024B]"));
        assert!(v.source.contains("layer=5"));
        assert!(v.source.contains("input_layernorm.weight"));
        assert!(v.source.contains("// TODO(sprint A)"));
    }

    #[test]
    fn qkv_node_promotes_tier_to_qkv() {
        let n = FusedQkvRopeCache::__new_for_emit(
            0, 1, 2, 3, 4, 5, 0, 256, 256, 256, 0, 1, 1, 7,
            WeightRef::new("qkv".into()),
            RotaryRef::new("rope".into()),
            false,
            true,
        );
        let tape = MegaTape::__build_from_nodes(vec![MegaNode::FusedQkvRopeCache(n)]);
        let v = lower_to_cuda(&tape, "test_qkv", &ctx());
        assert_eq!(v.abi_tier, CuAbiTier::Qkv);
        assert!(v.source.contains("const uint32_t*             positions"));
        assert!(v.source.contains("FusedQkvRopeCache"));
        assert!(v.source.contains("// TODO(sprint B)"));
    }

    #[test]
    fn attention_node_promotes_tier_to_attn() {
        let n = AttentionViaCacheNode::__new_for_emit(
            0,
            0,
            0,
            512,
            512,
            512,
            0,
            1,
            1,
            3,
            AttentionKind::Full,
            false,
        );
        let tape = MegaTape::__build_from_nodes(vec![MegaNode::AttentionViaCache(n)]);
        let v = lower_to_cuda(&tape, "test_attn", &ctx());
        assert_eq!(v.abi_tier, CuAbiTier::Attn);
        assert!(v.source.contains("const int32_t*              seq_lens"));
        assert!(v.source.contains("const uint32_t*             block_table"));
        assert!(v.source.contains("AttentionViaCache"));
    }

    #[test]
    fn attention_overrides_qkv_when_both_present() {
        let qkv = FusedQkvRopeCache::__new_for_emit(
            0, 1, 2, 3, 4, 5, 0, 256, 256, 256, 0, 1, 1, 7,
            WeightRef::new("qkv".into()),
            RotaryRef::new("rope".into()),
            false,
            false,
        );
        let attn = AttentionViaCacheNode::__new_for_emit(
            6, 6, 0, 512, 512, 512, 0, 1, 1, 7, AttentionKind::Full, false,
        );
        let tape = MegaTape::__build_from_nodes(vec![
            MegaNode::FusedQkvRopeCache(qkv),
            MegaNode::AttentionViaCache(attn),
        ]);
        let v = lower_to_cuda(&tape, "test_qkv_then_attn", &ctx());
        assert_eq!(v.abi_tier, CuAbiTier::Attn);
    }

    #[test]
    fn many_node_kinds_all_emit() {
        let nodes = vec![
            MegaNode::Embed(Embed::__new_for_emit(
                0,
                1,
                0,
                1,
                WeightRef::new("embed".into()),
            )),
            MegaNode::ScalarMul(ScalarMul::__new_for_emit(
                0,
                0,
                0,
                1,
                FiniteF32::new(0.5),
            )),
            MegaNode::TanhSoftCap(TanhSoftCap::__new_for_emit(0, 0, 0, 1)),
            MegaNode::Add(Add::__new_for_emit(0, 1, 0, 1)),
            MegaNode::BarrierSignal(BarrierSignal::__new_for_emit(0)),
            MegaNode::BarrierWait(BarrierWait::__new_for_emit(0, 8)),
            MegaNode::SpliceMmEmbeds(SpliceMmEmbeds::__new_for_emit(2, 0, 1)),
        ];
        let tape = MegaTape::__build_from_nodes(nodes);
        let v = lower_to_cuda(&tape, "test_many", &ctx());
        assert_eq!(v.abi_tier, CuAbiTier::Base);
        for needle in [
            "Embed {", "ScalarMul {", "TanhSoftCap {", "Add {", "BarrierSignal {",
            "BarrierWait {", "SpliceMmEmbeds {",
        ] {
            assert!(v.source.contains(needle), "missing node body: {needle}");
        }
    }
}
