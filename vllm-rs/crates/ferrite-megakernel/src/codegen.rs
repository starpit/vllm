// SPDX-License-Identifier: Apache-2.0
//! Mega-IR-building dispatch core for the `ferrite-forward-macro`
//! proc-macro. Sibling of [`crate::cuda_emit`] (which emits CUDA
//! source strings); this module emits `proc_macro2::TokenStream`s
//! that the proc-macro splices into the user's `#[forward]`
//! expansion as literal `b.push_*::<...>(...)` calls on a
//! [`MegaTapeBuilder`](crate::MegaTapeBuilder). Both `cuda_emit` and
//! `codegen` live inside `ferrite-megakernel` so that ALL megakernel-
//! related code is owned by the mega crate — no megakernel codegen
//! leaks into non-mega crates.
//!
//! Although `proc_macro2`/`quote` are conventionally proc-macro
//! tools, they're regular library crates and work fine in non-proc-
//! macro contexts. Pulling them into `ferrite-megakernel` lets the
//! dispatch helpers live next to the IR types they construct.
//!
//! Proc-macro-local orchestration types (`CanonicalLowered`,
//! `LoweredBucket`, `WorkloadPoint`, `ModelParams`) stay in
//! `ferrite-forward-macro`; only the `Instruction →
//! MegaTapeBuilder::push_*(...)` dispatch core lives here.
//!
//! Public surface:
//! - [`MegaDispatchState`] — per-canonical state threaded through
//!   the dispatch walk (cumulative arrives, slot allocator,
//!   per-canonical context: hidden_dim, head_dim, num_q_heads, …).
//! - [`dispatch_instruction_to_push`] — emit one
//!   `b.push_*::<...>(...)` token stream for one
//!   `ferrite_forward::Instruction`.
//! - [`instruction_kind`] / [`max_loop_iter_count`] /
//!   [`count_barrier_edges`] — read-only diagnostics on a tape.
//! - [`normalize_tk_prefix`] — fold `Tk*` frontend variants onto
//!   their un-prefixed peer for substrate-typed lowering.

use proc_macro2::{Literal, TokenStream};
use quote::quote;

/// Per-canonical state threaded through the dispatch walk. Tracks
/// the cumulative arrive count (for phase parity / `ARRIVES` const
/// generics) and a slot allocator for synthesizing non-aliasing
/// weight / intermediate page ids when the `Instruction` only
/// supplies in/out slots.
pub struct MegaDispatchState {
    arrives: u32,
    /// Next page id to hand out (mod num_pages_budget, skipping
    /// excluded ids). Synthesized at expansion time, baked into
    /// the emitted const-generic call as a literal.
    next_slot: u32,
    num_pages_budget: u32,
    num_consumer_warps: u32,
    scratch_bytes: u32,
    num_layers: u32,
    // ── Canonical context (per `MEGA_IR_PLAN.md` §0/§4a).
    // Every kernel template/runtime arg the emit step splices into the
    // .cu source must originate from a typed field on the MegaNode
    // variant, populated here from per-canonical model bounds.
    /// `hidden_size` — `<Config, HIDDEN_DIM, …>` template arg for
    /// every per-row op (RmsNorm, FusedAddRmsNorm, Embed, GateUp, …).
    hidden_dim: u32,
    /// Per-head dim (`head_dim`) — fused-QKV / attention template arg.
    head_dim: u32,
    /// `num_attention_heads / tp` — per-rank Q heads at the runtime tp.
    num_q_heads: u32,
    /// `num_key_value_heads / tp` — per-rank KV heads.
    num_kv_heads: u32,
    /// `intermediate_size / tp` — gate-up / down-proj inner dim.
    intermediate_dim: u32,
    /// `vocab_size` — embed + lm_head N dim.
    vocab_size: u32,
    /// `wp.num_tokens` — kernel `<…, NUM_TOKENS>` template arg (M).
    num_tokens: u32,
    /// `rms_norm_eps` — kernel `consumer(..., float eps)` runtime arg
    /// for every RmsNorm-flavored op.
    rms_norm_eps: f32,
    /// `final_logit_softcapping` — `TkTanhSoftCap` kernel cap value.
    /// Gemma2 lm_head softcap; 0.0 for arches without final logit cap.
    tanh_soft_cap: f32,
    /// `attention_multiplier` / `query_pre_attn_scalar` /
    /// `1 / sqrt(head_dim)` — `AttentionViaCache` consumer eps-like
    /// runtime arg (the score-pre-scale).
    attn_scale: f32,
    /// `attn_logit_softcapping` — `AttentionViaCache` runtime arg.
    /// 0.0 for arches without an attention softcap.
    attn_softcap: f32,
    /// `sliding_window` — `Instruction::SlidingAttentionViaCache`
    /// kernel template arg (Gemma2 = 4096); 0 for arches without a
    /// sliding window. Used at dispatch time when constructing
    /// `AttentionKind::Sliding(w)`.
    sliding_window: u32,
    /// `wp.sk_bucket` — the canonical's max past-K bucket. Becomes
    /// the `MAX_SK` template arg of `attention_partial`.
    sk_bucket: u32,
    /// Cumulative weight-accessor index. Each dispatch arm that
    /// consumes a weight slot bumps this by the count it consumes;
    /// the per-op `weight_accessor_idx` field on the emitted MegaNode
    /// is the value at the dispatch site, not the post-bump value.
    /// Maps directly to `weight_ptrs[idx * NUM_LAYERS + layer]` in
    /// the emitted `.cu`.
    next_weight_accessor: u32,
    /// `NUM_EDGES` from the synthesized substrate budget. Needed at
    /// `BarrierSignal` / `BarrierWait` dispatch sites so the emitted
    /// `EdgeId<IDX, NUM_EDGES>` typed primitive bound matches the
    /// surrounding `MegaTapeBuilder<…, NUM_EDGES>` const-generic.
    num_edges: u32,
}

impl MegaDispatchState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_pages_budget: u32,
        num_consumer_warps: u32,
        scratch_bytes: u32,
        num_layers: u32,
        hidden_dim: u32,
        head_dim: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        num_tokens: u32,
        rms_norm_eps: f32,
        tanh_soft_cap: f32,
        attn_scale: f32,
        attn_softcap: f32,
        sliding_window: u32,
        sk_bucket: u32,
        num_edges: u32,
    ) -> Self {
        Self {
            arrives: 0,
            next_slot: 0,
            num_pages_budget,
            num_consumer_warps,
            scratch_bytes,
            num_layers,
            hidden_dim,
            head_dim,
            num_q_heads,
            num_kv_heads,
            intermediate_dim,
            vocab_size,
            num_tokens,
            rms_norm_eps,
            tanh_soft_cap,
            attn_scale,
            attn_softcap,
            sliding_window,
            sk_bucket,
            next_weight_accessor: 0,
            num_edges,
        }
    }

    /// Allocate the next free page id that doesn't appear in
    /// `exclude`. Wraps modulo `num_pages_budget`.
    fn alloc_distinct(&mut self, exclude: &[u32]) -> Result<u32, String> {
        for _ in 0..self.num_pages_budget {
            let id = self.next_slot % self.num_pages_budget;
            self.next_slot = (self.next_slot + 1) % self.num_pages_budget;
            if !exclude.contains(&id) {
                return Ok(id);
            }
        }
        Err(format!(
            "slot allocator exhausted (num_pages_budget={} all in exclude={:?})",
            self.num_pages_budget, exclude
        ))
    }
}

/// Largest `Loop(count, _)` count anywhere in the slice — used to
/// size `state.num_layers` so vision-MM canonicals' loop iters
/// (e.g. SigLIP=27 vs text decoder=26) don't trip the
/// `LAYER < NUM_LAYERS` substrate proof.
pub fn max_loop_iter_count(instrs: &[ferrite_forward::Instruction]) -> u32 {
    use ferrite_forward::Instruction as I;
    instrs
        .iter()
        .filter_map(|i| match i {
            I::Loop(count, _) => Some(*count),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// Count `BarrierSignal` ops in a tape — used to size
/// `SubstrateBudget::NUM_EDGES`. Wait ops reuse the same edge ids
/// signal ops do, so counting signals only is sufficient.
pub fn count_barrier_edges(instrs: &[ferrite_forward::Instruction]) -> u32 {
    use ferrite_forward::Instruction as I;
    instrs
        .iter()
        .filter(|i| matches!(i, I::BarrierSignal(_)))
        .count() as u32
}

/// Stable variant name string for diagnostics.
pub fn instruction_kind(instr: &ferrite_forward::Instruction) -> &'static str {
    use ferrite_forward::Instruction as I;
    match instr {
        I::RmsNorm(..) => "RmsNorm",
        I::FusedQkvRopeCache(..) => "FusedQkvRopeCache",
        I::Add(..) => "Add",
        I::FusedAddRmsNorm(..) => "FusedAddRmsNorm",
        I::FusedGateUpSiluMul(..) => "FusedGateUpSiluMul",
        I::FusedGateUpGeluMul(..) => "FusedGateUpGeluMul",
        I::Embed(..) => "Embed",
        I::ScalarMul(..) => "ScalarMul",
        I::TanhSoftCap(..) => "TanhSoftCap",
        I::ScalarOffsetRmsNorm(..) => "ScalarOffsetRmsNorm",
        I::Gemm(..) => "Gemm",
        I::CutlassFusedRmsNormGemm(..) => "CutlassFusedRmsNormGemm",
        I::CutlassFusedAddRmsNormGemm(..) => "CutlassFusedAddRmsNormGemm",
        I::CutlassFusedAddScalarOffsetRmsNormGemm(..) => "CutlassFusedAddScalarOffsetRmsNormGemm",
        I::CutlassFusedMeanSubRmsNormGemm(..) => "CutlassFusedMeanSubRmsNormGemm",
        I::AttentionViaCache(..) => "AttentionViaCache",
        I::SlidingAttentionViaCache(..) => "SlidingAttentionViaCache",
        I::BarrierSignal(..) => "BarrierSignal",
        I::BarrierWait(..) => "BarrierWait",
        I::SpliceMmEmbeds(..) => "SpliceMmEmbeds",
        I::Loop(..) => "Loop",
        I::Alias(..) => "Alias",
        I::Free(..) => "Free",
        I::Reshape(..) => "Reshape",
        I::FusedAddRmsNormWithOffset(..) => "FusedAddRmsNormWithOffset",
        I::MeanSubRmsNorm(..) => "MeanSubRmsNorm",
        I::MeanSubRmsNormBiasAdd(..) => "MeanSubRmsNormBiasAdd",
        I::FusedCublasGemmAdd(..) => "FusedCublasGemmAdd",
        I::FusedGemmBias(..) => "FusedGemmBias",
        I::FusedQkvRopePrefill(..) => "FusedQkvRopePrefill",
        I::FusedQkvQkNormRopeCache(..) => "FusedQkvQkNormRopeCache",
        I::AttentionPrefillContiguous(..) => "AttentionPrefillContiguous",
        I::EncoderAttention(..) => "EncoderAttention",
        I::SlidingAttentionPrefillContiguous(..) => "SlidingAttentionPrefillContiguous",
        I::VarlenAttention(..) => "VarlenAttention",
        I::VisionRope(..) => "VisionRope",
        I::QuickGelu(..) => "QuickGelu",
        I::Gelu(..) => "Gelu",
        I::PosEmbed(..) => "PosEmbed",
        I::LoadPixels(..) => "LoadPixels",
        I::GeluErf(..) => "GeluErf",
        I::EmbeddingGather(..) => "EmbeddingGather",
        I::AvgPool2d(..) => "AvgPool2d",
        I::StripCls(..) => "StripCls",
        I::FlashInferAttentionDecode(..) => "FlashInferAttentionDecode",
        I::FlashInferAttentionPrefill(..) => "FlashInferAttentionPrefill",
        I::RopeAppend(..) => "RopeAppend",
        I::MlaSplit(..) => "MlaSplit",
        I::MlaAttention(..) => "MlaAttention",
        I::DeepSeekMoe(..) => "DeepSeekMoe",
        I::DeepSeekMoeFp8Block(..) => "DeepSeekMoeFp8Block",
        I::DeepSeekMoeGgml(..) => "DeepSeekMoeGgml",
        I::FusedMoe(..) => "FusedMoe",
        I::SharedFusedMoe(..) => "SharedFusedMoe",
        I::CutlassGemm(..) => "CutlassGemm",
        I::CutlassGemmSplitK(..) => "CutlassGemmSplitK",
        I::CutlassGemmAdd(..) => "CutlassGemmAdd",
        I::CutlassGemv(..) => "CutlassGemv",
        I::CutlassFusedGemmBias(..) => "CutlassFusedGemmBias",
        I::CutlassFusedGateUpSiluMul(..) => "CutlassFusedGateUpSiluMul",
        I::CutlassFusedGateUpGeluMul(..) => "CutlassFusedGateUpGeluMul",
        I::CutlassFusedQkvRopeCache(..) => "CutlassFusedQkvRopeCache",
        I::CutlassFusedQkvRopePrefill(..) => "CutlassFusedQkvRopePrefill",
        I::MarlinGemm(..) => "MarlinGemm",
        I::MarlinFusedGateUpSiluMul(..) => "MarlinFusedGateUpSiluMul",
        I::MarlinFusedGateUpGeluMul(..) => "MarlinFusedGateUpGeluMul",
        I::MarlinFusedQkvRopeCache(..) => "MarlinFusedQkvRopeCache",
        I::MarlinFusedQkvRopePrefill(..) => "MarlinFusedQkvRopePrefill",
        I::Bnb4Gemm(..) => "Bnb4Gemm",
        I::Bnb4FusedGateUpSiluMul(..) => "Bnb4FusedGateUpSiluMul",
        I::Bnb4FusedGateUpGeluMul(..) => "Bnb4FusedGateUpGeluMul",
        I::Bnb4FusedQkvRopeCache(..) => "Bnb4FusedQkvRopeCache",
        I::Bnb4FusedQkvRopePrefill(..) => "Bnb4FusedQkvRopePrefill",
        I::GgmlGemm(..) => "GgmlGemm",
        I::GgmlFusedGateUpSiluMul(..) => "GgmlFusedGateUpSiluMul",
        I::GgmlFusedGateUpGeluMul(..) => "GgmlFusedGateUpGeluMul",
        I::GgmlFusedQkvRopeCache(..) => "GgmlFusedQkvRopeCache",
        I::GgmlFusedQkvRopePrefill(..) => "GgmlFusedQkvRopePrefill",
        I::Fp8Gemm(..) => "Fp8Gemm",
        I::Fp8FusedGemmBias(..) => "Fp8FusedGemmBias",
        I::Fp8FusedGateUpSiluMul(..) => "Fp8FusedGateUpSiluMul",
        I::Fp8FusedGateUpGeluMul(..) => "Fp8FusedGateUpGeluMul",
        I::Fp8FusedQkvRopeCache(..) => "Fp8FusedQkvRopeCache",
        I::Fp8FusedQkvRopePrefill(..) => "Fp8FusedQkvRopePrefill",
        I::TkEmbed(..) => "TkEmbed",
        I::TkScalarMul(..) => "TkScalarMul",
        I::TkRmsNorm(..) => "TkRmsNorm",
        I::TkGemm(..) => "TkGemm",
        I::TkFusedAddRmsNorm(..) => "TkFusedAddRmsNorm",
        I::TkFusedQkvRopeCache(..) => "TkFusedQkvRopeCache",
        I::TkAttentionViaCache(..) => "TkAttentionViaCache",
        I::TkSlidingAttentionViaCache(..) => "TkSlidingAttentionViaCache",
        I::TkFusedGateUpSiluMul(..) => "TkFusedGateUpSiluMul",
        I::TkFusedGateUpGeluMul(..) => "TkFusedGateUpGeluMul",
        I::TkGemmAdd(..) => "TkGemmAdd",
        I::TkFusedAddRmsNormGemm(..) => "TkFusedAddRmsNormGemm",
        I::TkScalarOffsetRmsNorm(..) => "TkScalarOffsetRmsNorm",
        I::TkFusedAddRmsNormWithOffset(..) => "TkFusedAddRmsNormWithOffset",
        I::TkTanhSoftCap(..) => "TkTanhSoftCap",
        I::TkFusedAddScalarOffsetRmsNormGemm(..) => "TkFusedAddScalarOffsetRmsNormGemm",
        I::TkBarrierSignal(..) => "TkBarrierSignal",
        I::TkBarrierWait(..) => "TkBarrierWait",
        I::TkSpliceMmEmbeds(..) => "TkSpliceMmEmbeds",
        #[cfg(feature = "nccl")]
        I::AllReduce(..) => "AllReduce",
        #[cfg(feature = "nccl")]
        I::AllGather(..) => "AllGather",
    }
}

/// Dispatch one `Instruction` to a `b.push_*::<...>(...)` token
/// stream that, when monomorphized at user-build time, drives the
/// const-generic builder API.
///
/// Returns `Err(...)` for variants whose const-generic builder
/// dispatch isn't wired yet (the caller skips the entire canonical
/// to host fallback). Adding a variant here = lifting it for
/// mega-IR emission.
//
// `layer_override`: when `Some(iter)`, the variant's `layer` field
// is REPLACED by `iter` for the const-generic `LAYER` literal.
// Used by the proc-macro's loop-expansion walk to substitute
// per-iter layer indices into unrolled `Loop` body ops. Variants
// with no layer field ignore the override.
pub fn dispatch_instruction_to_push(
    instr: &ferrite_forward::Instruction,
    weight_paths: &[String],
    state: &mut MegaDispatchState,
    layer_override: Option<u32>,
) -> Result<TokenStream, String> {
    use ferrite_forward::Instruction as I;

    let lit = Literal::u32_unsuffixed;
    let resolved_layer = |default: u32| layer_override.unwrap_or(default);

    match instr {
        I::RmsNorm(in_slot, out_slot, layer) => {
            let weight = weight_paths
                .first()
                .ok_or_else(|| "weight_paths empty".to_string())?;
            let in_id = lit(*in_slot);
            let weight_id = lit(state.alloc_distinct(&[*in_slot])?);
            let partial_off = lit(0u32);
            let partial_bytes = lit(state.num_consumer_warps * 4);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit((state.arrives + 1) & 1);
            let arrives = lit(state.arrives);
            let num_layers = lit(state.num_layers);
            let layer_lit = lit(resolved_layer(*layer));
            let hidden_dim = lit(state.hidden_dim);
            let num_tokens = lit(state.num_tokens);
            let in_act_slot = lit(*in_slot);
            let out_act_slot = lit(*out_slot);
            let weight_accessor_idx = lit(state.next_weight_accessor);
            let consumer_bar_reduce = lit(1u32);
            let consumer_bar_publish = lit(2u32);
            let weight_str = weight.as_str();
            let eps_lit = state.rms_norm_eps;
            let num_pages_lit = lit(state.num_pages_budget);
            let scratch_lit = lit(state.scratch_bytes);
            state.arrives += 1;
            state.next_weight_accessor += 1;
            Ok(quote! {
                b.push_rms_norm(
                    ::ferrite_megakernel::ir::ArrivesCount::<#arrives>::new(),
                    ::ferrite_megakernel::ir::PageId::<#in_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#weight_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::ScratchRegion::<
                        #partial_off,
                        #partial_bytes,
                        #scratch_lit,
                        ::ferrite_megakernel::ir::RmsNormScope,
                    >::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#consumer_phase>::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#storer_phase>::new(),
                    ::ferrite_megakernel::ir::LayerIndex::<#layer_lit, #num_layers>::new(),
                    ::ferrite_megakernel::ir::HiddenDim::<#hidden_dim>::new(),
                    ::ferrite_megakernel::ir::NumTokensConst::<#num_tokens>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#in_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#out_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::WeightAccessorConst::<
                        #weight_accessor_idx,
                        { u32::MAX },
                    >::new(),
                    ::ferrite_megakernel::ir::BarSyncId::<#consumer_bar_reduce>::new(),
                    ::ferrite_megakernel::ir::BarSyncId::<#consumer_bar_publish>::new(),
                    ::ferrite_megakernel::ir::BarSyncPair::<
                        #consumer_bar_reduce,
                        #consumer_bar_publish,
                    >::new(),
                    #weight_str.to_string(),
                    #eps_lit,
                );
            })
        }
        I::Add(delta_slot, residual_slot) => {
            let delta_id = lit(*delta_slot);
            let residual_id = lit(*residual_slot);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit((state.arrives + 1) & 1);
            let arrives = lit(state.arrives);
            let hidden_dim = lit(state.hidden_dim);
            let num_tokens = lit(state.num_tokens);
            let delta_act_slot = lit(*delta_slot);
            let residual_act_slot = lit(*residual_slot);
            let num_pages_lit = lit(state.num_pages_budget);
            let consumer_bar_publish = lit(2u32);
            state.arrives += 1;
            Ok(quote! {
                b.push_add(
                    ::ferrite_megakernel::ir::ArrivesCount::<#arrives>::new(),
                    ::ferrite_megakernel::ir::PageId::<#delta_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#residual_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#consumer_phase>::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#storer_phase>::new(),
                    ::ferrite_megakernel::ir::HiddenDim::<#hidden_dim>::new(),
                    ::ferrite_megakernel::ir::NumTokensConst::<#num_tokens>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#delta_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#residual_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::BarSyncId::<#consumer_bar_publish>::new(),
                );
            })
        }
        I::Embed(out_slot) => {
            let weight = weight_paths
                .first()
                .ok_or_else(|| "Embed weight_paths empty".to_string())?;
            let out_id = lit(*out_slot);
            let weight_id = lit(state.alloc_distinct(&[*out_slot])?);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit((state.arrives + 1) & 1);
            let arrives = lit(state.arrives);
            let hidden_dim = lit(state.hidden_dim);
            let num_tokens = lit(state.num_tokens);
            let vocab_size = lit(state.vocab_size);
            let out_act_slot = lit(*out_slot);
            let weight_accessor_idx = lit(state.next_weight_accessor);
            let weight_str = weight.as_str();
            let num_pages_lit = lit(state.num_pages_budget);
            state.arrives += 1;
            state.next_weight_accessor += 1;
            Ok(quote! {
                b.push_embed(
                    ::ferrite_megakernel::ir::ArrivesCount::<#arrives>::new(),
                    ::ferrite_megakernel::ir::PageId::<#out_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#weight_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#consumer_phase>::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#storer_phase>::new(),
                    ::ferrite_megakernel::ir::HiddenDim::<#hidden_dim>::new(),
                    ::ferrite_megakernel::ir::NumTokensConst::<#num_tokens>::new(),
                    ::ferrite_megakernel::ir::VocabSize::<#vocab_size>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#out_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::WeightAccessorConst::<
                        #weight_accessor_idx, { u32::MAX },
                    >::new(),
                    #weight_str.to_string(),
                );
            })
        }
        I::ScalarMul(in_slot, out_slot, scale) => {
            let in_id = lit(*in_slot);
            let out_id = lit(*out_slot);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit((state.arrives + 1) & 1);
            let arrives = lit(state.arrives);
            let hidden_dim = lit(state.hidden_dim);
            let num_tokens = lit(state.num_tokens);
            let in_act_slot = lit(*in_slot);
            let out_act_slot = lit(*out_slot);
            let scale_lit = *scale;
            let num_pages_lit = lit(state.num_pages_budget);
            let consumer_bar_publish = lit(2u32);
            state.arrives += 1;
            Ok(quote! {
                b.push_scalar_mul(
                    ::ferrite_megakernel::ir::ArrivesCount::<#arrives>::new(),
                    ::ferrite_megakernel::ir::PageId::<#in_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#out_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#consumer_phase>::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#storer_phase>::new(),
                    ::ferrite_megakernel::ir::HiddenDim::<#hidden_dim>::new(),
                    ::ferrite_megakernel::ir::NumTokensConst::<#num_tokens>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#in_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#out_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::BarSyncId::<#consumer_bar_publish>::new(),
                    #scale_lit,
                );
            })
        }
        I::TanhSoftCap(in_slot, out_slot) => {
            let in_id = lit(*in_slot);
            let out_id = lit(*out_slot);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit((state.arrives + 1) & 1);
            let arrives = lit(state.arrives);
            let hidden_dim = lit(state.hidden_dim);
            let num_tokens = lit(state.num_tokens);
            let in_act_slot = lit(*in_slot);
            let out_act_slot = lit(*out_slot);
            let cap_lit = state.tanh_soft_cap;
            let num_pages_lit = lit(state.num_pages_budget);
            let consumer_bar_publish = lit(2u32);
            state.arrives += 1;
            Ok(quote! {
                b.push_tanh_soft_cap(
                    ::ferrite_megakernel::ir::ArrivesCount::<#arrives>::new(),
                    ::ferrite_megakernel::ir::PageId::<#in_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#out_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#consumer_phase>::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#storer_phase>::new(),
                    ::ferrite_megakernel::ir::HiddenDim::<#hidden_dim>::new(),
                    ::ferrite_megakernel::ir::NumTokensConst::<#num_tokens>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#in_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#out_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::BarSyncId::<#consumer_bar_publish>::new(),
                    #cap_lit,
                );
            })
        }
        I::ScalarOffsetRmsNorm(in_slot, out_slot, layer, offset) => {
            let weight = weight_paths
                .first()
                .ok_or_else(|| "ScalarOffsetRmsNorm weight_paths empty".to_string())?;
            let in_id = lit(*in_slot);
            let weight_id = lit(state.alloc_distinct(&[*in_slot])?);
            let partial_off = lit(0u32);
            let partial_bytes = lit(state.num_consumer_warps * 4);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit((state.arrives + 1) & 1);
            let arrives = lit(state.arrives);
            let num_layers = lit(state.num_layers);
            let layer_lit = lit(resolved_layer(*layer));
            let hidden_dim = lit(state.hidden_dim);
            let num_tokens = lit(state.num_tokens);
            let in_act_slot = lit(*in_slot);
            let out_act_slot = lit(*out_slot);
            let weight_accessor_idx = lit(state.next_weight_accessor);
            let weight_str = weight.as_str();
            let offset_lit = *offset;
            let eps_lit = state.rms_norm_eps;
            let num_pages_lit = lit(state.num_pages_budget);
            let scratch_lit = lit(state.scratch_bytes);
            let consumer_bar_reduce = lit(1u32);
            let consumer_bar_publish = lit(2u32);
            state.arrives += 1;
            state.next_weight_accessor += 1;
            Ok(quote! {
                b.push_scalar_offset_rms_norm(
                    ::ferrite_megakernel::ir::ArrivesCount::<#arrives>::new(),
                    ::ferrite_megakernel::ir::PageId::<#in_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#weight_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::ScratchRegion::<
                        #partial_off, #partial_bytes, #scratch_lit,
                        ::ferrite_megakernel::ir::RmsNormScope,
                    >::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#consumer_phase>::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#storer_phase>::new(),
                    ::ferrite_megakernel::ir::LayerIndex::<#layer_lit, #num_layers>::new(),
                    ::ferrite_megakernel::ir::HiddenDim::<#hidden_dim>::new(),
                    ::ferrite_megakernel::ir::NumTokensConst::<#num_tokens>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#in_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#out_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::WeightAccessorConst::<
                        #weight_accessor_idx, { u32::MAX },
                    >::new(),
                    ::ferrite_megakernel::ir::BarSyncId::<#consumer_bar_reduce>::new(),
                    ::ferrite_megakernel::ir::BarSyncId::<#consumer_bar_publish>::new(),
                    ::ferrite_megakernel::ir::BarSyncPair::<
                        #consumer_bar_reduce,
                        #consumer_bar_publish,
                    >::new(),
                    #weight_str.to_string(),
                    #offset_lit,
                    #eps_lit,
                );
            })
        }
        I::Gemm(in_slot, out_slot, layer, n, k) => {
            let weight = weight_paths
                .first()
                .ok_or_else(|| "Gemm weight_paths empty".to_string())?;
            let in_id = lit(*in_slot);
            let out_id = lit(*out_slot);
            let weight_id = lit(state.alloc_distinct(&[*in_slot, *out_slot])?);
            let b_tile_off = lit(0u32);
            let b_tile_bytes = lit(state.scratch_bytes);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit((state.arrives + 1) & 1);
            // ITERS=1 today (proc-macro hands op-level iters; the
            // scheduler doesn't currently chunk K). With ITERS=1
            // CHUNK_K must equal K (Gemm IR invariant).
            let iters_const = 1_u32;
            let iters = lit(iters_const);
            let chunk_k_const = *k / iters_const;
            let chunk_k_lit = lit(chunk_k_const);
            // TILE_N = N / NCW (AlongN warp split).
            let ncw = state.num_consumer_warps;
            let tile_n_const = if ncw > 0 && n % ncw == 0 {
                n / ncw
            } else {
                *n
            };
            let tile_n_lit = lit(tile_n_const);
            let arrives = lit(state.arrives);
            let num_layers = lit(state.num_layers);
            let layer_lit = lit(resolved_layer(*layer));
            let n_lit = lit(*n);
            let k_lit = lit(*k);
            let m_lit = lit(state.num_tokens);
            let in_act_slot = lit(*in_slot);
            let out_act_slot = lit(*out_slot);
            let weight_accessor_idx = lit(state.next_weight_accessor);
            let weight_str = weight.as_str();
            let num_pages_lit = lit(state.num_pages_budget);
            let scratch_lit = lit(state.scratch_bytes);
            let consumer_bar_publish = lit(1u32);
            state.arrives += 1;
            state.next_weight_accessor += 1;
            Ok(quote! {
                b.push_gemm(
                    ::ferrite_megakernel::ir::ArrivesCount::<#arrives>::new(),
                    ::ferrite_megakernel::ir::PageId::<#in_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#weight_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#out_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::ScratchRegion::<
                        #b_tile_off, #b_tile_bytes, #scratch_lit,
                        ::ferrite_megakernel::ir::GemmScope,
                    >::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#consumer_phase>::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#storer_phase>::new(),
                    ::ferrite_megakernel::ir::IterCount::<#iters>::new(),
                    ::ferrite_megakernel::ir::LayerIndex::<#layer_lit, #num_layers>::new(),
                    ::ferrite_megakernel::ir::MatmulN::<#n_lit>::new(),
                    ::ferrite_megakernel::ir::MatmulK::<#k_lit>::new(),
                    ::ferrite_megakernel::ir::MatmulM::<#m_lit>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#in_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#out_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::WeightAccessorConst::<
                        #weight_accessor_idx, { u32::MAX },
                    >::new(),
                    ::ferrite_megakernel::ir::TileN::<#tile_n_lit>::new(),
                    ::ferrite_megakernel::ir::ChunkK::<#chunk_k_lit>::new(),
                    ::ferrite_megakernel::ir::BarSyncId::<#consumer_bar_publish>::new(),
                    #weight_str.to_string(),
                );
            })
        }
        I::FusedAddRmsNorm(delta_slot, residual_slot, layer) => {
            let weight = weight_paths
                .first()
                .ok_or_else(|| "FusedAddRmsNorm weight_paths empty".to_string())?;
            let delta_id = lit(*delta_slot);
            let residual_id = lit(*residual_slot);
            let weight_id = lit(state.alloc_distinct(&[*delta_slot, *residual_slot])?);
            let partial_off = lit(0u32);
            let partial_bytes = lit(state.num_consumer_warps * 4);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit((state.arrives + 1) & 1);
            let arrives = lit(state.arrives);
            let num_layers = lit(state.num_layers);
            let layer_lit = lit(resolved_layer(*layer));
            let hidden_dim = lit(state.hidden_dim);
            let num_tokens = lit(state.num_tokens);
            let delta_act_slot = lit(*delta_slot);
            let residual_act_slot = lit(*residual_slot);
            let weight_accessor_idx = lit(state.next_weight_accessor);
            let weight_str = weight.as_str();
            let eps_lit = state.rms_norm_eps;
            let num_pages_lit = lit(state.num_pages_budget);
            let scratch_lit = lit(state.scratch_bytes);
            let consumer_bar_reduce = lit(1u32);
            let consumer_bar_publish = lit(2u32);
            state.arrives += 1;
            state.next_weight_accessor += 1;
            Ok(quote! {
                b.push_fused_add_rms_norm(
                    ::ferrite_megakernel::ir::ArrivesCount::<#arrives>::new(),
                    ::ferrite_megakernel::ir::PageId::<#delta_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#residual_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#weight_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::ScratchRegion::<
                        #partial_off, #partial_bytes, #scratch_lit,
                        ::ferrite_megakernel::ir::RmsNormScope,
                    >::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#consumer_phase>::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#storer_phase>::new(),
                    ::ferrite_megakernel::ir::LayerIndex::<#layer_lit, #num_layers>::new(),
                    ::ferrite_megakernel::ir::HiddenDim::<#hidden_dim>::new(),
                    ::ferrite_megakernel::ir::NumTokensConst::<#num_tokens>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#delta_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#residual_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::WeightAccessorConst::<
                        #weight_accessor_idx, { u32::MAX },
                    >::new(),
                    ::ferrite_megakernel::ir::BarSyncId::<#consumer_bar_reduce>::new(),
                    ::ferrite_megakernel::ir::BarSyncId::<#consumer_bar_publish>::new(),
                    ::ferrite_megakernel::ir::BarSyncPair::<
                        #consumer_bar_reduce,
                        #consumer_bar_publish,
                    >::new(),
                    #weight_str.to_string(),
                    #eps_lit,
                );
            })
        }
        I::FusedGateUpSiluMul(in_slot, out_slot, layer)
        | I::FusedGateUpGeluMul(in_slot, out_slot, layer) => {
            let weight = weight_paths
                .first()
                .ok_or_else(|| "FusedGateUp*Mul weight_paths empty".to_string())?;
            let activation_path = match instr {
                I::FusedGateUpSiluMul(..) => {
                    quote! { ::ferrite_megakernel::ir::GateUpActivation::Silu }
                }
                I::FusedGateUpGeluMul(..) => {
                    quote! { ::ferrite_megakernel::ir::GateUpActivation::Gelu }
                }
                _ => unreachable!(),
            };
            let in_id = lit(*in_slot);
            let out_id = lit(*out_slot);
            let weight_id = lit(state.alloc_distinct(&[*in_slot, *out_slot])?);
            let half = state.scratch_bytes / 2;
            let gate_off = lit(0u32);
            let gate_bytes = lit(half);
            let up_off = lit(half);
            let up_bytes = lit(half);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit((state.arrives + 1) & 1);
            let iters = lit(1u32);
            let arrives = lit(state.arrives);
            let num_layers = lit(state.num_layers);
            let layer_lit = lit(resolved_layer(*layer));
            let hidden_dim = lit(state.hidden_dim);
            let intermediate_dim = lit(state.intermediate_dim);
            let num_tokens = lit(state.num_tokens);
            let in_act_slot = lit(*in_slot);
            let out_act_slot = lit(*out_slot);
            let weight_accessor_idx = lit(state.next_weight_accessor);
            let weight_str = weight.as_str();
            let num_pages_lit = lit(state.num_pages_budget);
            let scratch_lit = lit(state.scratch_bytes);
            state.arrives += 1;
            state.next_weight_accessor += 1;
            Ok(quote! {
                b.push_fused_gate_up_activate_mul(
                    ::ferrite_megakernel::ir::ArrivesCount::<#arrives>::new(),
                    ::ferrite_megakernel::ir::PageId::<#in_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#weight_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#out_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::ScratchRegion::<
                        #gate_off, #gate_bytes, #scratch_lit, ::ferrite_megakernel::ir::MlpScope,
                    >::new(),
                    ::ferrite_megakernel::ir::ScratchRegion::<
                        #up_off, #up_bytes, #scratch_lit, ::ferrite_megakernel::ir::MlpScope,
                    >::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#consumer_phase>::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#storer_phase>::new(),
                    ::ferrite_megakernel::ir::IterCount::<#iters>::new(),
                    ::ferrite_megakernel::ir::LayerIndex::<#layer_lit, #num_layers>::new(),
                    ::ferrite_megakernel::ir::HiddenDim::<#hidden_dim>::new(),
                    ::ferrite_megakernel::ir::IntermediateDim::<#intermediate_dim>::new(),
                    ::ferrite_megakernel::ir::NumTokensConst::<#num_tokens>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#in_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#out_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::WeightAccessorConst::<
                        #weight_accessor_idx, { u32::MAX },
                    >::new(),
                    #weight_str.to_string(),
                    #activation_path,
                );
            })
        }
        I::FusedQkvRopeCache(in_slot, out_slot, layer, biased, interleaved) => {
            if weight_paths.len() != 2 {
                return Err(format!(
                    "FusedQkvRopeCache expected 2 weight_paths (qkv, rotary), got {}",
                    weight_paths.len()
                ));
            }
            let qkv_path = weight_paths[0].as_str();
            let rotary_path = weight_paths[1].as_str();
            let in_id = lit(*in_slot);
            let qkv_id = lit(state.alloc_distinct(&[*in_slot])?);
            let cs_id = lit(state.alloc_distinct(&[*in_slot])?);
            let q_id = lit(state.alloc_distinct(&[*in_slot])?);
            let k_id = lit(state.alloc_distinct(&[*in_slot])?);
            let v_id = lit(state.alloc_distinct(&[*in_slot])?);
            let half = state.scratch_bytes / 2;
            let q_off = lit(0u32);
            let q_bytes = lit(half);
            let k_off = lit(half);
            let k_bytes = lit(half);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit((state.arrives + 1) & 1);
            let iters = lit(1u32);
            let arrives = lit(state.arrives);
            let num_layers = lit(state.num_layers);
            let layer_lit = lit(resolved_layer(*layer));
            let hidden_dim = lit(state.hidden_dim);
            let head_dim = lit(state.head_dim);
            let num_q_heads = lit(state.num_q_heads);
            let num_kv_heads = lit(state.num_kv_heads);
            let in_act_slot = lit(*in_slot);
            let q_out_act_slot = lit(*out_slot);
            let k_out_act_slot = lit(out_slot.wrapping_add(1));
            let v_out_act_slot = lit(out_slot.wrapping_add(2));
            let qkv_weight_accessor_idx = lit(state.next_weight_accessor);
            let rotary_accessor_idx = lit(state.next_weight_accessor + 1);
            let biased_lit = *biased;
            let interleaved_lit = *interleaved;
            let num_pages_lit = lit(state.num_pages_budget);
            let scratch_lit = lit(state.scratch_bytes);
            state.arrives += 1;
            state.next_weight_accessor += 2;
            Ok(quote! {
                b.push_fused_qkv_rope_cache(
                    ::ferrite_megakernel::ir::ArrivesCount::<#arrives>::new(),
                    ::ferrite_megakernel::ir::PageId::<#in_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#qkv_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#cs_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#q_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#k_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#v_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::ScratchRegion::<
                        #q_off, #q_bytes, #scratch_lit, ::ferrite_megakernel::ir::RopeScope,
                    >::new(),
                    ::ferrite_megakernel::ir::ScratchRegion::<
                        #k_off, #k_bytes, #scratch_lit, ::ferrite_megakernel::ir::RopeScope,
                    >::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#consumer_phase>::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#storer_phase>::new(),
                    ::ferrite_megakernel::ir::IterCount::<#iters>::new(),
                    ::ferrite_megakernel::ir::LayerIndex::<#layer_lit, #num_layers>::new(),
                    ::ferrite_megakernel::ir::HiddenDim::<#hidden_dim>::new(),
                    ::ferrite_megakernel::ir::HeadDim::<#head_dim>::new(),
                    ::ferrite_megakernel::ir::NumQHeads::<#num_q_heads>::new(),
                    ::ferrite_megakernel::ir::NumKvHeads::<#num_kv_heads>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#in_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#q_out_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#k_out_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#v_out_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::WeightAccessorConst::<
                        #qkv_weight_accessor_idx, { u32::MAX },
                    >::new(),
                    ::ferrite_megakernel::ir::WeightAccessorConst::<
                        #rotary_accessor_idx, { u32::MAX },
                    >::new(),
                    #qkv_path.to_string(),
                    #rotary_path.to_string(),
                    #biased_lit,
                    #interleaved_lit,
                );
            })
        }
        // RopeAppend (qwen3 layer body): split-q/k/v in-place rotary
        // + reshape_and_cache. Q/K/V are already projected upstream
        // (no QKV weight here — only `rotary` cos_sin). Substrate
        // shape is structurally RopeScope-equivalent to
        // FusedQkvRopeCache. Map to push_fused_qkv_rope_cache with a
        // sentinel qkv weight string (the runtime kernel routes
        // through `RopeAppend::eval`; the qkv path is helper config
        // the emit step replaces with the real per-arch weight
        // binding).
        I::RopeAppend(
            q_slot,
            _k_slot,
            _v_slot,
            _q_out_slot,
            _k_out_slot,
            _v_out_slot,
            layer,
            interleaved,
        ) => {
            if weight_paths.len() != 1 {
                return Err(format!(
                    "RopeAppend expected 1 weight_path (rotary), got {}",
                    weight_paths.len()
                ));
            }
            let rotary_path = weight_paths[0].as_str();
            let qkv_sentinel = "<rope_append_no_qkv>";
            let in_id_val = *q_slot;
            let q_id_val = state.alloc_distinct(&[in_id_val])?;
            let qkv_id_val = state.alloc_distinct(&[in_id_val, q_id_val])?;
            let cs_id_val = state.alloc_distinct(&[in_id_val, q_id_val, qkv_id_val])?;
            let k_id_val =
                state.alloc_distinct(&[in_id_val, q_id_val, qkv_id_val, cs_id_val])?;
            let v_id_val = state.alloc_distinct(&[
                in_id_val, q_id_val, qkv_id_val, cs_id_val, k_id_val,
            ])?;
            let in_id = lit(in_id_val);
            let q_id = lit(q_id_val);
            let qkv_id = lit(qkv_id_val);
            let cs_id = lit(cs_id_val);
            let k_id = lit(k_id_val);
            let v_id = lit(v_id_val);
            let half = state.scratch_bytes / 2;
            let q_off = lit(0u32);
            let q_bytes = lit(half);
            let k_off = lit(half);
            let k_bytes = lit(half);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit((state.arrives + 1) & 1);
            let iters = lit(1u32);
            let arrives = lit(state.arrives);
            let num_layers = lit(state.num_layers);
            let layer_lit = lit(resolved_layer(*layer));
            let hidden_dim = lit(state.hidden_dim);
            let head_dim = lit(state.head_dim);
            let num_q_heads = lit(state.num_q_heads);
            let num_kv_heads = lit(state.num_kv_heads);
            let in_act_slot = lit(in_id_val);
            let q_out_act_slot = lit(in_id_val);
            let k_out_act_slot = lit(in_id_val.wrapping_add(1));
            let v_out_act_slot = lit(in_id_val.wrapping_add(2));
            let qkv_weight_accessor_idx = lit(state.next_weight_accessor);
            let rotary_accessor_idx = lit(state.next_weight_accessor + 1);
            let biased_lit = false;
            let interleaved_lit = *interleaved;
            let num_pages_lit = lit(state.num_pages_budget);
            let scratch_lit = lit(state.scratch_bytes);
            state.arrives += 1;
            state.next_weight_accessor += 2;
            Ok(quote! {
                b.push_fused_qkv_rope_cache(
                    ::ferrite_megakernel::ir::ArrivesCount::<#arrives>::new(),
                    ::ferrite_megakernel::ir::PageId::<#in_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#qkv_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#cs_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#q_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#k_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#v_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::ScratchRegion::<
                        #q_off, #q_bytes, #scratch_lit, ::ferrite_megakernel::ir::RopeScope,
                    >::new(),
                    ::ferrite_megakernel::ir::ScratchRegion::<
                        #k_off, #k_bytes, #scratch_lit, ::ferrite_megakernel::ir::RopeScope,
                    >::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#consumer_phase>::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#storer_phase>::new(),
                    ::ferrite_megakernel::ir::IterCount::<#iters>::new(),
                    ::ferrite_megakernel::ir::LayerIndex::<#layer_lit, #num_layers>::new(),
                    ::ferrite_megakernel::ir::HiddenDim::<#hidden_dim>::new(),
                    ::ferrite_megakernel::ir::HeadDim::<#head_dim>::new(),
                    ::ferrite_megakernel::ir::NumQHeads::<#num_q_heads>::new(),
                    ::ferrite_megakernel::ir::NumKvHeads::<#num_kv_heads>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#in_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#q_out_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#k_out_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#v_out_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::WeightAccessorConst::<
                        #qkv_weight_accessor_idx, { u32::MAX },
                    >::new(),
                    ::ferrite_megakernel::ir::WeightAccessorConst::<
                        #rotary_accessor_idx, { u32::MAX },
                    >::new(),
                    #qkv_sentinel.to_string(),
                    #rotary_path.to_string(),
                    #biased_lit,
                    #interleaved_lit,
                );
            })
        }
        I::SlidingAttentionViaCache(q_slot, attn_out_slot, layer, interleaved) => {
            let q_id = lit(*q_slot);
            let out_id = lit(*attn_out_slot);
            let half = state.scratch_bytes / 2;
            let score_off = lit(0u32);
            let score_bytes = lit(half);
            let pv_off = lit(half);
            let pv_bytes = lit(half);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit((state.arrives + 1) & 1);
            let iters = lit(1u32);
            let arrives = lit(state.arrives);
            let num_layers = lit(state.num_layers);
            let layer_lit = lit(resolved_layer(*layer));
            let head_dim = lit(state.head_dim);
            let num_q_heads = lit(state.num_q_heads);
            let num_kv_heads = lit(state.num_kv_heads);
            let block_size = lit(16u32); // attention_partial.cuh fixed at 16
            let num_tokens = lit(state.num_tokens);
            let max_sk = lit(state.sk_bucket.max(1));
            let q_in_act_slot = lit(*q_slot);
            let attn_out_act_slot = lit(*attn_out_slot);
            let interleaved_lit = *interleaved;
            let sliding_window_val = if state.sliding_window > 0 {
                state.sliding_window
            } else {
                4096
            };
            let sliding_window_lit = lit(sliding_window_val);
            let attn_scale_lit = state.attn_scale;
            let attn_softcap_lit = state.attn_softcap;
            let num_pages_lit = lit(state.num_pages_budget);
            let scratch_lit = lit(state.scratch_bytes);
            state.arrives += 1;
            Ok(quote! {
                b.push_attention_via_cache(
                    ::ferrite_megakernel::ir::ArrivesCount::<#arrives>::new(),
                    ::ferrite_megakernel::ir::PageId::<#q_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#out_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::ScratchRegion::<
                        #score_off, #score_bytes, #scratch_lit,
                        ::ferrite_megakernel::ir::AttentionScope,
                    >::new(),
                    ::ferrite_megakernel::ir::ScratchRegion::<
                        #pv_off, #pv_bytes, #scratch_lit,
                        ::ferrite_megakernel::ir::AttentionScope,
                    >::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#consumer_phase>::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#storer_phase>::new(),
                    ::ferrite_megakernel::ir::IterCount::<#iters>::new(),
                    ::ferrite_megakernel::ir::LayerIndex::<#layer_lit, #num_layers>::new(),
                    ::ferrite_megakernel::ir::HeadDim::<#head_dim>::new(),
                    ::ferrite_megakernel::ir::NumQHeads::<#num_q_heads>::new(),
                    ::ferrite_megakernel::ir::NumKvHeads::<#num_kv_heads>::new(),
                    ::ferrite_megakernel::ir::BlockSize::<#block_size>::new(),
                    ::ferrite_megakernel::ir::NumTokensConst::<#num_tokens>::new(),
                    ::ferrite_megakernel::ir::MaxSk::<#max_sk>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#q_in_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#attn_out_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::AttentionKind::Sliding(#sliding_window_lit),
                    #interleaved_lit,
                    #attn_scale_lit,
                    #attn_softcap_lit,
                );
            })
        }
        I::AttentionViaCache(q_slot, attn_out_slot, layer, interleaved) => {
            let q_id = lit(*q_slot);
            let out_id = lit(*attn_out_slot);
            let half = state.scratch_bytes / 2;
            let score_off = lit(0u32);
            let score_bytes = lit(half);
            let pv_off = lit(half);
            let pv_bytes = lit(half);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit((state.arrives + 1) & 1);
            let iters = lit(1u32);
            let arrives = lit(state.arrives);
            let num_layers = lit(state.num_layers);
            let layer_lit = lit(resolved_layer(*layer));
            let head_dim = lit(state.head_dim);
            let num_q_heads = lit(state.num_q_heads);
            let num_kv_heads = lit(state.num_kv_heads);
            let block_size = lit(16u32);
            let num_tokens = lit(state.num_tokens);
            let max_sk = lit(state.sk_bucket.max(1));
            let q_in_act_slot = lit(*q_slot);
            let attn_out_act_slot = lit(*attn_out_slot);
            let interleaved_lit = *interleaved;
            let attn_scale_lit = state.attn_scale;
            let attn_softcap_lit = state.attn_softcap;
            let num_pages_lit = lit(state.num_pages_budget);
            let scratch_lit = lit(state.scratch_bytes);
            state.arrives += 1;
            Ok(quote! {
                b.push_attention_via_cache(
                    ::ferrite_megakernel::ir::ArrivesCount::<#arrives>::new(),
                    ::ferrite_megakernel::ir::PageId::<#q_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#out_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::ScratchRegion::<
                        #score_off, #score_bytes, #scratch_lit,
                        ::ferrite_megakernel::ir::AttentionScope,
                    >::new(),
                    ::ferrite_megakernel::ir::ScratchRegion::<
                        #pv_off, #pv_bytes, #scratch_lit,
                        ::ferrite_megakernel::ir::AttentionScope,
                    >::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#consumer_phase>::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#storer_phase>::new(),
                    ::ferrite_megakernel::ir::IterCount::<#iters>::new(),
                    ::ferrite_megakernel::ir::LayerIndex::<#layer_lit, #num_layers>::new(),
                    ::ferrite_megakernel::ir::HeadDim::<#head_dim>::new(),
                    ::ferrite_megakernel::ir::NumQHeads::<#num_q_heads>::new(),
                    ::ferrite_megakernel::ir::NumKvHeads::<#num_kv_heads>::new(),
                    ::ferrite_megakernel::ir::BlockSize::<#block_size>::new(),
                    ::ferrite_megakernel::ir::NumTokensConst::<#num_tokens>::new(),
                    ::ferrite_megakernel::ir::MaxSk::<#max_sk>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#q_in_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#attn_out_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::AttentionKind::Full,
                    #interleaved_lit,
                    #attn_scale_lit,
                    #attn_softcap_lit,
                );
            })
        }
        I::FusedCublasGemmAdd(in_slot, residual_slot, layer, n, k) => {
            let weight = weight_paths
                .first()
                .ok_or_else(|| "FusedCublasGemmAdd weight_paths empty".to_string())?;
            let in_id = lit(*in_slot);
            let residual_id = lit(*residual_slot);
            let weight_id = lit(state.alloc_distinct(&[*in_slot, *residual_slot])?);
            let b_tile_off = lit(0u32);
            let b_tile_bytes = lit(state.scratch_bytes);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit((state.arrives + 1) & 1);
            let iters = lit(1u32);
            let arrives = lit(state.arrives);
            let num_layers = lit(state.num_layers);
            let layer_lit = lit(resolved_layer(*layer));
            let n_lit = lit(*n);
            let k_lit = lit(*k);
            let num_tokens_lit = lit(state.num_tokens);
            let k_offset_lit = lit(0u32);
            let k_full_lit = lit(*k);
            let in_act_slot = lit(*in_slot);
            let residual_act_slot = lit(*residual_slot);
            let weight_accessor_idx = lit(state.next_weight_accessor);
            let weight_str = weight.as_str();
            let num_pages_lit = lit(state.num_pages_budget);
            let scratch_lit = lit(state.scratch_bytes);
            state.arrives += 1;
            state.next_weight_accessor += 1;
            Ok(quote! {
                b.push_tk_fused_gemm_add(
                    ::ferrite_megakernel::ir::ArrivesCount::<#arrives>::new(),
                    ::ferrite_megakernel::ir::PageId::<#in_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#weight_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::PageId::<#residual_id, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::ScratchRegion::<
                        #b_tile_off, #b_tile_bytes, #scratch_lit,
                        ::ferrite_megakernel::ir::GemmScope,
                    >::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#consumer_phase>::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#storer_phase>::new(),
                    ::ferrite_megakernel::ir::IterCount::<#iters>::new(),
                    ::ferrite_megakernel::ir::LayerIndex::<#layer_lit, #num_layers>::new(),
                    ::ferrite_megakernel::ir::MatmulN::<#n_lit>::new(),
                    ::ferrite_megakernel::ir::MatmulK::<#k_lit>::new(),
                    ::ferrite_megakernel::ir::NumTokensConst::<#num_tokens_lit>::new(),
                    ::ferrite_megakernel::ir::KOffset::<#k_offset_lit>::new(),
                    ::ferrite_megakernel::ir::KFull::<#k_full_lit>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#in_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#residual_act_slot, { u32::MAX }>::new(),
                    ::ferrite_megakernel::ir::WeightAccessorConst::<
                        #weight_accessor_idx, { u32::MAX },
                    >::new(),
                    #weight_str.to_string(),
                );
            })
        }
        I::SpliceMmEmbeds(slot) => {
            let slot_lit = lit(*slot);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit((state.arrives + 1) & 1);
            let arrives = lit(state.arrives);
            let hidden_dim = lit(state.hidden_dim);
            let num_tokens = lit(state.num_tokens);
            let target_act_slot = lit(*slot);
            let num_pages_lit = lit(state.num_pages_budget);
            state.arrives += 1;
            Ok(quote! {
                b.push_splice_mm_embeds(
                    ::ferrite_megakernel::ir::ArrivesCount::<#arrives>::new(),
                    ::ferrite_megakernel::ir::PageId::<#slot_lit, #num_pages_lit>::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#consumer_phase>::new(),
                    ::ferrite_megakernel::ir::MbarrierPhase::<#storer_phase>::new(),
                    ::ferrite_megakernel::ir::HiddenDim::<#hidden_dim>::new(),
                    ::ferrite_megakernel::ir::NumTokensConst::<#num_tokens>::new(),
                    ::ferrite_megakernel::ir::ActSlotConst::<#target_act_slot, { u32::MAX }>::new(),
                );
            })
        }
        I::BarrierSignal(edge) => {
            let edge_lit = lit(*edge);
            let num_edges_lit = lit(state.num_edges.max(1));
            Ok(quote! {
                b.push_barrier_signal(
                    ::ferrite_megakernel::ir::EdgeId::<#edge_lit, #num_edges_lit>::new(),
                );
            })
        }
        I::BarrierWait(edge, count) => {
            let edge_lit = lit(*edge);
            let count_lit = lit(*count);
            let num_edges_lit = lit(state.num_edges.max(1));
            Ok(quote! {
                b.push_barrier_wait(
                    ::ferrite_megakernel::ir::EdgeId::<#edge_lit, #num_edges_lit>::new(),
                    ::ferrite_megakernel::ir::ExpectedCount::<#count_lit>::new(),
                );
            })
        }
        I::CutlassFusedRmsNormGemm(in_slot, out_slot, layer, _tile_m, _tile_n, _stages, n, k) => {
            emit_lm_head_no_delta(
                *in_slot,
                *out_slot,
                *layer,
                *n,
                *k,
                quote! { ::ferrite_megakernel::ir::LmHeadNormKind::RmsNorm },
                weight_paths,
                state,
            )
        }
        I::CutlassFusedMeanSubRmsNormGemm(
            in_slot,
            out_slot,
            layer,
            _tile_m,
            _tile_n,
            _stages,
            n,
            k,
        ) => emit_lm_head_no_delta(
            *in_slot,
            *out_slot,
            *layer,
            *n,
            *k,
            quote! { ::ferrite_megakernel::ir::LmHeadNormKind::MeanSubRmsNorm },
            weight_paths,
            state,
        ),
        I::CutlassFusedAddRmsNormGemm(
            delta_slot,
            residual_slot,
            out_slot,
            layer,
            _tile_m,
            _tile_n,
            _stages,
            n,
            k,
        ) => emit_lm_head_with_delta(
            *residual_slot,
            *delta_slot,
            *out_slot,
            *layer,
            *n,
            *k,
            quote! { ::ferrite_megakernel::ir::LmHeadNormKind::AddRmsNorm },
            None,
            weight_paths,
            state,
        ),
        I::CutlassFusedAddScalarOffsetRmsNormGemm(
            delta_slot,
            residual_slot,
            out_slot,
            layer,
            offset,
            _tile_m,
            _tile_n,
            _stages,
            n,
            k,
        ) => emit_lm_head_with_delta(
            *residual_slot,
            *delta_slot,
            *out_slot,
            *layer,
            *n,
            *k,
            quote! { ::ferrite_megakernel::ir::LmHeadNormKind::AddScalarOffsetRmsNorm },
            Some(*offset),
            weight_paths,
            state,
        ),
        other => Err(format!(
            "no const-generic builder dispatch for variant `{}`",
            instruction_kind(other)
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_lm_head_no_delta(
    in_slot: u32,
    out_slot: u32,
    layer: u32,
    n: u32,
    k: u32,
    norm_kind_path: TokenStream,
    weight_paths: &[String],
    state: &mut MegaDispatchState,
) -> Result<TokenStream, String> {
    if weight_paths.len() != 2 {
        return Err(format!(
            "TkFusedNormGemm lm_head fusion expected 2 weight_paths (norm, linear), got {}",
            weight_paths.len()
        ));
    }
    let norm_path = weight_paths[0].as_str();
    let linear_path = weight_paths[1].as_str();
    let lit = Literal::u32_unsuffixed;
    let in_id = lit(in_slot);
    let out_id = lit(out_slot);
    let norm_w_id = lit(state.alloc_distinct(&[in_slot, out_slot])?);
    let lin_w_id = lit(state.alloc_distinct(&[in_slot, out_slot])?);
    let partial_off = lit(0u32);
    let partial_bytes = lit(state.num_consumer_warps * 4);
    let b_tile_off = lit(state.num_consumer_warps * 4);
    let b_tile_bytes = lit(state.scratch_bytes - state.num_consumer_warps * 4);
    let consumer_phase = lit(state.arrives & 1);
    let storer_phase = lit((state.arrives + 1) & 1);
    let iters = lit(1u32);
    let arrives = lit(state.arrives);
    let num_layers = lit(state.num_layers);
    let layer_lit = lit(layer);
    let n_lit = lit(n);
    let k_lit = lit(k);
    let num_tokens = lit(state.num_tokens);
    let in_act_slot = lit(in_slot);
    let out_act_slot = lit(out_slot);
    let norm_weight_accessor_idx = lit(state.next_weight_accessor);
    let linear_weight_accessor_idx = lit(state.next_weight_accessor + 1);
    let eps_lit = state.rms_norm_eps;
    let num_pages_lit = lit(state.num_pages_budget);
    let scratch_lit = lit(state.scratch_bytes);
    state.arrives += 1;
    state.next_weight_accessor += 2;
    Ok(quote! {
        b.push_tk_fused_norm_gemm_no_delta(
            ::ferrite_megakernel::ir::ArrivesCount::<#arrives>::new(),
            ::ferrite_megakernel::ir::PageId::<#in_id, #num_pages_lit>::new(),
            ::ferrite_megakernel::ir::PageId::<#norm_w_id, #num_pages_lit>::new(),
            ::ferrite_megakernel::ir::PageId::<#lin_w_id, #num_pages_lit>::new(),
            ::ferrite_megakernel::ir::PageId::<#out_id, #num_pages_lit>::new(),
            ::ferrite_megakernel::ir::ScratchRegion::<
                #partial_off, #partial_bytes, #scratch_lit,
                ::ferrite_megakernel::ir::GemmScope,
            >::new(),
            ::ferrite_megakernel::ir::ScratchRegion::<
                #b_tile_off, #b_tile_bytes, #scratch_lit,
                ::ferrite_megakernel::ir::GemmScope,
            >::new(),
            ::ferrite_megakernel::ir::MbarrierPhase::<#consumer_phase>::new(),
            ::ferrite_megakernel::ir::MbarrierPhase::<#storer_phase>::new(),
            ::ferrite_megakernel::ir::IterCount::<#iters>::new(),
            ::ferrite_megakernel::ir::LayerIndex::<#layer_lit, #num_layers>::new(),
            ::ferrite_megakernel::ir::MatmulN::<#n_lit>::new(),
            ::ferrite_megakernel::ir::MatmulK::<#k_lit>::new(),
            ::ferrite_megakernel::ir::NumTokensConst::<#num_tokens>::new(),
            ::ferrite_megakernel::ir::ActSlotConst::<#in_act_slot, { u32::MAX }>::new(),
            ::ferrite_megakernel::ir::ActSlotConst::<#out_act_slot, { u32::MAX }>::new(),
            ::ferrite_megakernel::ir::WeightAccessorConst::<
                #norm_weight_accessor_idx, { u32::MAX },
            >::new(),
            ::ferrite_megakernel::ir::WeightAccessorConst::<
                #linear_weight_accessor_idx, { u32::MAX },
            >::new(),
            #norm_path.to_string(), #linear_path.to_string(), #norm_kind_path, #eps_lit,
        );
    })
}

#[allow(clippy::too_many_arguments)]
fn emit_lm_head_with_delta(
    residual_slot: u32,
    delta_slot: u32,
    out_slot: u32,
    layer: u32,
    n: u32,
    k: u32,
    norm_kind_path: TokenStream,
    offset: Option<f32>,
    weight_paths: &[String],
    state: &mut MegaDispatchState,
) -> Result<TokenStream, String> {
    if weight_paths.len() != 2 {
        return Err(format!(
            "TkFusedNormGemm lm_head fusion expected 2 weight_paths (norm, linear), got {}",
            weight_paths.len()
        ));
    }
    let norm_path = weight_paths[0].as_str();
    let linear_path = weight_paths[1].as_str();
    let lit = Literal::u32_unsuffixed;
    let in_id = lit(residual_slot);
    let delta_id = lit(delta_slot);
    let out_id = lit(out_slot);
    let norm_w_id = lit(state.alloc_distinct(&[residual_slot, delta_slot, out_slot])?);
    let lin_w_id = lit(state.alloc_distinct(&[residual_slot, delta_slot, out_slot])?);
    let partial_off = lit(0u32);
    let partial_bytes = lit(state.num_consumer_warps * 4);
    let b_tile_off = lit(state.num_consumer_warps * 4);
    let b_tile_bytes = lit(state.scratch_bytes - state.num_consumer_warps * 4);
    let consumer_phase = lit(state.arrives & 1);
    let storer_phase = lit((state.arrives + 1) & 1);
    let iters = lit(1u32);
    let arrives = lit(state.arrives);
    let num_layers = lit(state.num_layers);
    let layer_lit = lit(layer);
    let n_lit = lit(n);
    let k_lit = lit(k);
    let num_tokens = lit(state.num_tokens);
    let in_act_slot = lit(residual_slot);
    let delta_act_slot = lit(delta_slot);
    let out_act_slot = lit(out_slot);
    let norm_weight_accessor_idx = lit(state.next_weight_accessor);
    let linear_weight_accessor_idx = lit(state.next_weight_accessor + 1);
    let eps_lit = state.rms_norm_eps;
    let offset_expr = match offset {
        Some(v) => quote! { ::core::option::Option::Some(#v) },
        None => quote! { ::core::option::Option::None },
    };
    let num_pages_lit = lit(state.num_pages_budget);
    let scratch_lit = lit(state.scratch_bytes);
    state.arrives += 1;
    state.next_weight_accessor += 2;
    Ok(quote! {
        b.push_tk_fused_norm_gemm_with_delta(
            ::ferrite_megakernel::ir::ArrivesCount::<#arrives>::new(),
            ::ferrite_megakernel::ir::PageId::<#in_id, #num_pages_lit>::new(),
            ::ferrite_megakernel::ir::PageId::<#delta_id, #num_pages_lit>::new(),
            ::ferrite_megakernel::ir::PageId::<#norm_w_id, #num_pages_lit>::new(),
            ::ferrite_megakernel::ir::PageId::<#lin_w_id, #num_pages_lit>::new(),
            ::ferrite_megakernel::ir::PageId::<#out_id, #num_pages_lit>::new(),
            ::ferrite_megakernel::ir::ScratchRegion::<
                #partial_off, #partial_bytes, #scratch_lit,
                ::ferrite_megakernel::ir::GemmScope,
            >::new(),
            ::ferrite_megakernel::ir::ScratchRegion::<
                #b_tile_off, #b_tile_bytes, #scratch_lit,
                ::ferrite_megakernel::ir::GemmScope,
            >::new(),
            ::ferrite_megakernel::ir::MbarrierPhase::<#consumer_phase>::new(),
            ::ferrite_megakernel::ir::MbarrierPhase::<#storer_phase>::new(),
            ::ferrite_megakernel::ir::IterCount::<#iters>::new(),
            ::ferrite_megakernel::ir::LayerIndex::<#layer_lit, #num_layers>::new(),
            ::ferrite_megakernel::ir::MatmulN::<#n_lit>::new(),
            ::ferrite_megakernel::ir::MatmulK::<#k_lit>::new(),
            ::ferrite_megakernel::ir::NumTokensConst::<#num_tokens>::new(),
            ::ferrite_megakernel::ir::ActSlotConst::<#in_act_slot, { u32::MAX }>::new(),
            ::ferrite_megakernel::ir::ActSlotConst::<#delta_act_slot, { u32::MAX }>::new(),
            ::ferrite_megakernel::ir::ActSlotConst::<#out_act_slot, { u32::MAX }>::new(),
            ::ferrite_megakernel::ir::WeightAccessorConst::<
                #norm_weight_accessor_idx, { u32::MAX },
            >::new(),
            ::ferrite_megakernel::ir::WeightAccessorConst::<
                #linear_weight_accessor_idx, { u32::MAX },
            >::new(),
            #norm_path.to_string(), #linear_path.to_string(), #norm_kind_path, #offset_expr, #eps_lit,
        );
    })
}

/// Map a `Tk*` frontend `Instruction` variant to its un-prefixed
/// peer for substrate-typed lowering. The `Tk` prefix is a
/// megakernel-claim marker; substrate-wise the variants are
/// identical and each `Tk*` `eval` delegates to its non-Tk
/// counterpart at runtime. Variants without a Tk prefix pass
/// through unchanged.
pub fn normalize_tk_prefix(
    instr: ferrite_forward::Instruction,
) -> ferrite_forward::Instruction {
    use ferrite_forward::Instruction as I;
    match instr {
        I::TkEmbed(out_slot) => I::Embed(out_slot),
        I::TkScalarMul(in_slot, out_slot, scale) => I::ScalarMul(in_slot, out_slot, scale),
        I::TkRmsNorm(in_slot, out_slot, layer) => I::RmsNorm(in_slot, out_slot, layer),
        I::TkGemm(in_slot, out_slot, layer, n, k) => I::Gemm(in_slot, out_slot, layer, n, k),
        I::TkFusedAddRmsNorm(delta_slot, residual_slot, layer) => {
            I::FusedAddRmsNorm(delta_slot, residual_slot, layer)
        }
        I::TkFusedAddRmsNormWithOffset(delta_slot, residual_slot, layer, _offset) => {
            I::FusedAddRmsNorm(delta_slot, residual_slot, layer)
        }
        I::FusedAddRmsNormWithOffset(delta_slot, residual_slot, layer, _offset) => {
            I::FusedAddRmsNorm(delta_slot, residual_slot, layer)
        }
        I::TkFusedQkvRopeCache(in_slot, out_slot, layer, biased, interleaved) => {
            I::FusedQkvRopeCache(in_slot, out_slot, layer, biased, interleaved)
        }
        I::TkAttentionViaCache(in_slot, out_slot, layer, interleaved) => {
            I::AttentionViaCache(in_slot, out_slot, layer, interleaved)
        }
        I::TkSlidingAttentionViaCache(in_slot, out_slot, layer, interleaved, _window) => {
            I::SlidingAttentionViaCache(in_slot, out_slot, layer, interleaved)
        }
        I::TkFusedGateUpSiluMul(in_slot, out_slot, layer) => {
            I::FusedGateUpSiluMul(in_slot, out_slot, layer)
        }
        I::TkFusedGateUpGeluMul(in_slot, out_slot, layer) => {
            I::FusedGateUpGeluMul(in_slot, out_slot, layer)
        }
        I::TkScalarOffsetRmsNorm(in_slot, out_slot, layer, offset) => {
            I::ScalarOffsetRmsNorm(in_slot, out_slot, layer, offset)
        }
        I::TkTanhSoftCap(in_slot, out_slot, _n_vocab) => I::TanhSoftCap(in_slot, out_slot),
        I::TkBarrierSignal(edge) => I::BarrierSignal(edge),
        I::TkBarrierWait(edge, count) => I::BarrierWait(edge, count),
        I::TkSpliceMmEmbeds(slot) => I::SpliceMmEmbeds(slot),
        // TkFusedAddRmsNormGemm decomposes to CutlassFusedAddRmsNormGemm
        // with dummy CUTLASS tile dims (the dispatch goes through the
        // host interpreter, ignoring tile_m/tile_n/stages — see
        // `instr.rs::eval` for the same delegation). Substrate shape
        // is identical: residual fold + rms_norm + gemm.
        I::TkFusedAddRmsNormGemm(delta_slot, residual_slot, out_slot, layer, n, k) => {
            I::CutlassFusedAddRmsNormGemm(
                delta_slot,
                residual_slot,
                out_slot,
                layer,
                /*tile_m=*/ 16,
                /*tile_n=*/ 64,
                /*stages=*/ 3,
                n,
                k,
            )
        }
        I::TkFusedAddScalarOffsetRmsNormGemm(
            delta_slot,
            residual_slot,
            out_slot,
            layer,
            offset,
            n,
            k,
        ) => I::CutlassFusedAddScalarOffsetRmsNormGemm(
            delta_slot,
            residual_slot,
            out_slot,
            layer,
            offset,
            16,
            64,
            3,
            n,
            k,
        ),
        // TkGemmAdd is chunked-k gemm + add. The non-chunked frontend
        // peer is FusedCublasGemmAdd(in, residual, layer, n, k).
        // Until the chunked variant gets its own MegaNode, normalize
        // to the non-chunked peer by dropping `k_offset`/`k_full` —
        // the substrate shape matches; the chunked-k iteration count
        // is internal to the emit step.
        I::TkGemmAdd(in_slot, residual_slot, layer, n, k, _k_offset, _k_full) => {
            I::FusedCublasGemmAdd(in_slot, residual_slot, layer, n, k)
        }
        I::MeanSubRmsNorm(in_slot, out_slot, layer) => {
            I::RmsNorm(in_slot, out_slot, layer)
        }
        I::MeanSubRmsNormBiasAdd(in_slot, out_slot, layer) => {
            I::RmsNorm(in_slot, out_slot, layer)
        }
        I::CutlassGemv(in_slot, out_slot, layer, n, k) => {
            I::Gemm(in_slot, out_slot, layer, n, k)
        }
        I::FusedGemmBias(in_slot, out_slot, layer) => {
            I::Gemm(in_slot, out_slot, layer, 1, 1)
        }
        I::PosEmbed(out_slot) => I::Embed(out_slot),
        I::EncoderAttention(q_slot, _k_slot, _v_slot, out_slot) => {
            I::AttentionViaCache(q_slot, out_slot, /*layer=*/ 0, /*interleaved=*/ false)
        }
        I::MarlinFusedQkvRopeCache(in_slot, out_slot, layer)
        | I::Bnb4FusedQkvRopeCache(in_slot, out_slot, layer)
        | I::Fp8FusedQkvRopeCache(in_slot, out_slot, layer) => {
            I::FusedQkvRopeCache(in_slot, out_slot, layer, false, false)
        }
        I::GgmlFusedQkvRopeCache(in_slot, out_slot, layer, interleaved) => {
            I::FusedQkvRopeCache(in_slot, out_slot, layer, false, interleaved)
        }
        I::FusedQkvRopePrefill(in_slot, out_slot, layer, _n, _k, biased, interleaved) => {
            I::FusedQkvRopeCache(in_slot, out_slot, layer, biased, interleaved)
        }
        I::MarlinFusedQkvRopePrefill(in_slot, out_slot, layer, _n, _k)
        | I::Bnb4FusedQkvRopePrefill(in_slot, out_slot, layer, _n, _k)
        | I::Fp8FusedQkvRopePrefill(in_slot, out_slot, layer, _n, _k)
        | I::GgmlFusedQkvRopePrefill(in_slot, out_slot, layer, _n, _k) => {
            I::FusedQkvRopeCache(in_slot, out_slot, layer, false, false)
        }
        I::MarlinGemm(in_slot, out_slot, layer)
        | I::Bnb4Gemm(in_slot, out_slot, layer)
        | I::Fp8Gemm(in_slot, out_slot, layer)
        | I::GgmlGemm(in_slot, out_slot, layer) => I::Gemm(in_slot, out_slot, layer, 1, 1),
        I::MarlinFusedGateUpSiluMul(in_slot, out_slot, layer)
        | I::Bnb4FusedGateUpSiluMul(in_slot, out_slot, layer)
        | I::Fp8FusedGateUpSiluMul(in_slot, out_slot, layer)
        | I::GgmlFusedGateUpSiluMul(in_slot, out_slot, layer) => {
            I::FusedGateUpSiluMul(in_slot, out_slot, layer)
        }
        I::MarlinFusedGateUpGeluMul(in_slot, out_slot, layer)
        | I::Bnb4FusedGateUpGeluMul(in_slot, out_slot, layer)
        | I::Fp8FusedGateUpGeluMul(in_slot, out_slot, layer)
        | I::GgmlFusedGateUpGeluMul(in_slot, out_slot, layer) => {
            I::FusedGateUpGeluMul(in_slot, out_slot, layer)
        }
        I::Fp8FusedGemmBias(in_slot, out_slot, layer) => {
            I::Gemm(in_slot, out_slot, layer, 1, 1)
        }
        other => other,
    }
}
