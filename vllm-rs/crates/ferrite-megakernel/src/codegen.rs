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

/// Stable variant name string for diagnostics on mega-reachable
/// `Instruction` variants. Mega-claimed tapes only contain `Tk*`
/// peers (the `Tk*Impl` cost-DP winners' Instruction emission) plus
/// shape-named structural variants. Backend-vendor-tagged variants
/// (the host-interpreter peers that have no megakernel fast path)
/// are NEVER reachable from megakernel dispatch and are bucketed
/// under `<non-mega>` rather than enumerated. See
/// [[feedback-no-cutlass-in-mega]].
pub fn instruction_kind(instr: &ferrite_forward::Instruction) -> &'static str {
    use ferrite_forward::Instruction as I;
    match instr {
        // ── Tk* peers (frontend-emitted by `Tk*Impl::op_emit`).
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
        // ── Shape-named variants (no fast-path peer, may appear
        // post-`normalize_tk_prefix` strip).
        I::RmsNorm(..) => "RmsNorm",
        I::Add(..) => "Add",
        I::Embed(..) => "Embed",
        I::ScalarMul(..) => "ScalarMul",
        I::Gemm(..) => "Gemm",
        I::FusedAddRmsNorm(..) => "FusedAddRmsNorm",
        I::FusedAddRmsNormWithOffset(..) => "FusedAddRmsNormWithOffset",
        I::FusedGateUpSiluMul(..) => "FusedGateUpSiluMul",
        I::FusedGateUpGeluMul(..) => "FusedGateUpGeluMul",
        I::FusedQkvRopeCache(..) => "FusedQkvRopeCache",
        I::AttentionViaCache(..) => "AttentionViaCache",
        I::SlidingAttentionViaCache(..) => "SlidingAttentionViaCache",
        I::TanhSoftCap(..) => "TanhSoftCap",
        I::ScalarOffsetRmsNorm(..) => "ScalarOffsetRmsNorm",
        I::SpliceMmEmbeds(..) => "SpliceMmEmbeds",
        I::BarrierSignal(..) => "BarrierSignal",
        I::BarrierWait(..) => "BarrierWait",
        // ── Structural / control flow.
        I::Loop(..) => "Loop",
        I::Alias(..) => "Alias",
        I::Free(..) => "Free",
        I::Reshape(..) => "Reshape",
        #[cfg(feature = "nccl")]
        I::AllReduce(..) => "AllReduce",
        #[cfg(feature = "nccl")]
        I::AllGather(..) => "AllGather",
        // ── Vendor-named or non-mega-eligible variants. Reaching
        // this arm from megakernel dispatch is itself the bug
        // (the tape claimer should have rejected the tape).
        _ => "<non-mega>",
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
///
/// Per `MEGA_IR_PLAN.md` §8.0b "Implementing end-to-end const
/// generics: proc-macro-time dispatch", a parallel
/// [`dispatch_instruction_to_render`] walks the same Instruction
/// list and produces `render_*::<const-generic-args>(...)` token
/// streams. Both walks must use the SAME state semantics so the
/// IR's substrate proofs and the emitted `.cu` shapes share one
/// source of truth.
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
            // S12a IR ext: TILE_N = INTERMEDIATE_DIM / NCW
            // (AlongN warp split). Mirrors Gemm/TkFusedGemmAdd.
            let ncw = state.num_consumer_warps;
            let tile_n_const =
                if ncw > 0 && state.intermediate_dim % ncw == 0 {
                    state.intermediate_dim / ncw
                } else {
                    state.intermediate_dim
                };
            let tile_n_lit = lit(tile_n_const);
            let consumer_bar_publish = lit(1u32);
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
                    ::ferrite_megakernel::ir::TileN::<#tile_n_lit>::new(),
                    ::ferrite_megakernel::ir::BarSyncId::<#consumer_bar_publish>::new(),
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
            // S15c partition: scratch is split four ways. Q rope and
            // K rope each get a quarter (RopeScope, disjoint within
            // scope by offset proof); the qkv b_tile gets the other
            // half (GemmScope, cross-scope so disjoint by tag, no
            // offset proof needed against rope).
            let quarter = state.scratch_bytes / 4;
            let half = state.scratch_bytes / 2;
            let q_off = lit(0u32);
            let q_bytes = lit(quarter);
            let k_off = lit(quarter);
            let k_bytes = lit(quarter);
            let b_tile_off = lit(2 * quarter);
            let b_tile_bytes = lit(half);
            let iters_const = 1_u32;
            let iters = lit(iters_const);
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
            // S15a: matmul layout. M = num_tokens; K = hidden_dim;
            // N = qkv_n = (num_q_heads + 2*num_kv_heads) * head_dim.
            // TILE_N = qkv_n / NCW (AlongN warp split). Fall back to
            // qkv_n when NCW doesn't divide; the IR only requires
            // TILE_N > 0. Mirrors S10 / S11a / S12a / S13a.
            let num_tokens_const = state.num_tokens;
            let num_tokens_lit = lit(num_tokens_const);
            let qkv_n =
                (state.num_q_heads + 2 * state.num_kv_heads) * state.head_dim;
            let ncw = state.num_consumer_warps;
            let tile_n_const = if ncw > 0 && qkv_n % ncw == 0 { qkv_n / ncw } else { qkv_n };
            let tile_n_lit = lit(tile_n_const);
            // ITERS=1 today → CHUNK_K must equal HIDDEN_DIM (S15a IR invariant).
            let chunk_k_lit = lit(state.hidden_dim / iters_const);
            // Single named bar for the post-mma + RoPE publish.
            // Mirror of FusedGateUpActivateMul S12a (BarSyncId=1).
            let consumer_bar_publish = lit(1u32);
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
                    ::ferrite_megakernel::ir::ScratchRegion::<
                        #b_tile_off, #b_tile_bytes, #scratch_lit, ::ferrite_megakernel::ir::GemmScope,
                    >::new(),
                    ::ferrite_megakernel::ir::IterCount::<#iters>::new(),
                    ::ferrite_megakernel::ir::LayerIndex::<#layer_lit, #num_layers>::new(),
                    ::ferrite_megakernel::ir::HiddenDim::<#hidden_dim>::new(),
                    ::ferrite_megakernel::ir::HeadDim::<#head_dim>::new(),
                    ::ferrite_megakernel::ir::NumQHeads::<#num_q_heads>::new(),
                    ::ferrite_megakernel::ir::NumKvHeads::<#num_kv_heads>::new(),
                    ::ferrite_megakernel::ir::NumTokensConst::<#num_tokens_lit>::new(),
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
                    ::ferrite_megakernel::ir::TileN::<#tile_n_lit>::new(),
                    ::ferrite_megakernel::ir::ChunkK::<#chunk_k_lit>::new(),
                    ::ferrite_megakernel::ir::BarSyncId::<#consumer_bar_publish>::new(),
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
            // S15c partition (mirror of FusedQkvRopeCache arm). Even
            // though RopeAppend doesn't run a QKV matmul (its qkv
            // weight is the sentinel), the IR field is required, so
            // the proc-macro reserves the same scratch layout.
            let quarter = state.scratch_bytes / 4;
            let half = state.scratch_bytes / 2;
            let q_off = lit(0u32);
            let q_bytes = lit(quarter);
            let k_off = lit(quarter);
            let k_bytes = lit(quarter);
            let b_tile_off = lit(2 * quarter);
            let b_tile_bytes = lit(half);
            let iters_const = 1_u32;
            let iters = lit(iters_const);
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
            // S15a: same matmul-layout defaults as FusedQkvRopeCache.
            // RopeAppend is structurally identical at the IR level
            // (the runtime kernel handles the no-qkv-matmul shape via
            // the qkv sentinel path).
            let num_tokens_lit = lit(state.num_tokens);
            let qkv_n =
                (state.num_q_heads + 2 * state.num_kv_heads) * state.head_dim;
            let ncw = state.num_consumer_warps;
            let tile_n_const = if ncw > 0 && qkv_n % ncw == 0 { qkv_n / ncw } else { qkv_n };
            let tile_n_lit = lit(tile_n_const);
            let chunk_k_lit = lit(state.hidden_dim / iters_const);
            let consumer_bar_publish = lit(1u32);
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
                    ::ferrite_megakernel::ir::ScratchRegion::<
                        #b_tile_off, #b_tile_bytes, #scratch_lit, ::ferrite_megakernel::ir::GemmScope,
                    >::new(),
                    ::ferrite_megakernel::ir::IterCount::<#iters>::new(),
                    ::ferrite_megakernel::ir::LayerIndex::<#layer_lit, #num_layers>::new(),
                    ::ferrite_megakernel::ir::HiddenDim::<#hidden_dim>::new(),
                    ::ferrite_megakernel::ir::HeadDim::<#head_dim>::new(),
                    ::ferrite_megakernel::ir::NumQHeads::<#num_q_heads>::new(),
                    ::ferrite_megakernel::ir::NumKvHeads::<#num_kv_heads>::new(),
                    ::ferrite_megakernel::ir::NumTokensConst::<#num_tokens_lit>::new(),
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
                    ::ferrite_megakernel::ir::TileN::<#tile_n_lit>::new(),
                    ::ferrite_megakernel::ir::ChunkK::<#chunk_k_lit>::new(),
                    ::ferrite_megakernel::ir::BarSyncId::<#consumer_bar_publish>::new(),
                    #qkv_sentinel.to_string(),
                    #rotary_path.to_string(),
                    #biased_lit,
                    #interleaved_lit,
                );
            })
        }
        I::SlidingAttentionViaCache(q_slot, attn_out_slot, layer, interleaved) => {
            emit_attention_via_cache_push(
                *q_slot,
                *attn_out_slot,
                resolved_layer(*layer),
                *interleaved,
                /*is_sliding=*/ true,
                state,
            )
        }
        I::AttentionViaCache(q_slot, attn_out_slot, layer, interleaved) => {
            emit_attention_via_cache_push(
                *q_slot,
                *attn_out_slot,
                resolved_layer(*layer),
                *interleaved,
                /*is_sliding=*/ false,
                state,
            )
        }
        I::TkGemmAdd(in_slot, residual_slot, layer, n, k, _k_offset, _k_full) => {
            let weight = weight_paths
                .first()
                .ok_or_else(|| "TkGemmAdd weight_paths empty".to_string())?;
            let in_id = lit(*in_slot);
            let residual_id = lit(*residual_slot);
            let weight_id = lit(state.alloc_distinct(&[*in_slot, *residual_slot])?);
            let b_tile_off = lit(0u32);
            let b_tile_bytes = lit(state.scratch_bytes);
            // ITERS=1 today (proc-macro hands op-level iters; the
            // scheduler doesn't currently chunk K). With ITERS=1
            // CHUNK_K must equal K (TkFusedGemmAdd IR invariant).
            let iters_const = 1_u32;
            let iters = lit(iters_const);
            let chunk_k_const = *k / iters_const;
            let chunk_k_lit = lit(chunk_k_const);
            // TILE_N = N / NCW (AlongN warp split). Mirror of the
            // Gemm proc-macro logic from Sprint 9.
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
            let num_tokens_lit = lit(state.num_tokens);
            let k_offset_lit = lit(0u32);
            let k_full_lit = lit(*k);
            let in_act_slot = lit(*in_slot);
            let residual_act_slot = lit(*residual_slot);
            let weight_accessor_idx = lit(state.next_weight_accessor);
            let weight_str = weight.as_str();
            let num_pages_lit = lit(state.num_pages_budget);
            let scratch_lit = lit(state.scratch_bytes);
            // Fixed BAR ID in 1..=15 (bar 0 is __syncthreads).
            // Mirror of Gemm Sprint 10a: AlongN warp split has no
            // cross-warp reduction, so only one publish bar.
            let consumer_bar_publish = lit(1u32);
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
                    ::ferrite_megakernel::ir::TileN::<#tile_n_lit>::new(),
                    ::ferrite_megakernel::ir::ChunkK::<#chunk_k_lit>::new(),
                    ::ferrite_megakernel::ir::BarSyncId::<#consumer_bar_publish>::new(),
                    #weight_str.to_string(),
                );
            })
        }
        I::SpliceMmEmbeds(slot) => {
            let slot_lit = lit(*slot);
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
        I::TkFusedAddRmsNormGemm(delta_slot, residual_slot, out_slot, layer, n, k) => {
            emit_lm_head_with_delta(
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

/// Per-accessor weight base name for the megakernel `weight_ptrs[acc]`
/// grid emitted by [`dispatch_instruction_to_push`] for `instr`. Length
/// equals the number of accessors that arm bumps `state.next_weight_accessor`
/// by, in the same order, so the i-th element is the base name (i.e. the
/// `wm.<base>(layer)` accessor) the kernel will read for accessor `i`.
///
/// `None` means the accessor slot is a sentinel — currently only used
/// for `I::RopeAppend`'s qkv-projection slot, which carries no weight
/// (the rotary table is the only real input). The wrapper fills these
/// with null pointers; the kernel never dereferences them.
///
/// Mirrors `dispatch_instruction_to_push` arm-for-arm. Add an arm here
/// whenever `dispatch_instruction_to_push` grows a new accessor-allocating
/// variant, or the `forward_mega_<canonical>` wrapper will mis-stage
/// `weight_ptrs` for that op.
pub fn instruction_weight_bases(
    instr: &ferrite_forward::Instruction,
    weight_paths: &[String],
) -> Vec<Option<String>> {
    use ferrite_forward::Instruction as I;
    let normalized = normalize_tk_prefix(*instr);
    match normalized {
        // 1-accessor arms: weight_paths[0] is the base.
        I::RmsNorm(..)
        | I::Embed(..)
        | I::ScalarOffsetRmsNorm(..)
        | I::Gemm(..)
        | I::FusedAddRmsNorm(..)
        | I::FusedGateUpSiluMul(..)
        | I::FusedGateUpGeluMul(..)
        | I::TkGemmAdd(..) => vec![weight_paths.first().cloned()],
        // 2-accessor arms: weight_paths[0] then weight_paths[1].
        I::FusedQkvRopeCache(..)
        | I::TkFusedAddRmsNormGemm(..)
        | I::TkFusedAddScalarOffsetRmsNormGemm(..) => {
            vec![weight_paths.first().cloned(), weight_paths.get(1).cloned()]
        }
        // RopeAppend allocates 2 accessors but carries 1 weight_path:
        // accessor 0 is the qkv sentinel (no real weight, never read),
        // accessor 1 is the rotary cos/sin table.
        I::RopeAppend(..) => vec![None, weight_paths.first().cloned()],
        // 0-accessor arms.
        I::Add(..)
        | I::ScalarMul(..)
        | I::TanhSoftCap(..)
        | I::AttentionViaCache(..)
        | I::SlidingAttentionViaCache(..)
        | I::SpliceMmEmbeds(..)
        | I::BarrierSignal(..)
        | I::BarrierWait(..) => vec![],
        // Control / view ops have no megakernel substrate effect.
        I::Loop(..)
        | I::Alias(..)
        | I::Free(..)
        | I::Reshape(..)
        | I::LoadPixels(..)
        | I::EmbeddingGather(..)
        | I::StripCls(..) => vec![],
        _ => vec![],
    }
}

/// Build the `b.push_attention_via_cache::<…>(…)` token stream for
/// either `I::AttentionViaCache` or `I::SlidingAttentionViaCache`.
/// Both arms share the same const-generic + scratch-layout logic;
/// only the `AttentionKind` runtime arg differs.
///
/// AttentionScope scratch layout (committed here, transcribed by the
/// render fn):
///   `score`   `[0,                       SCORE_BYTES)`
///   `pv`      `[SCORE_BYTES,             SCORE_BYTES + PV_BYTES)`
///   `k_smem`  `[SCORE_BYTES + PV_BYTES,  + KV_BLOCK_BYTES)`
///   `v_smem`  next `KV_BLOCK_BYTES` bytes after k_smem
/// where `KV_BLOCK_BYTES = BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM * 2`.
/// Score / pv are sized `state.scratch_bytes` minus 2 KV blocks then
/// split in half. The `disjoint_with` chain in
/// `push_attention_via_cache` discharges all 6 pairwise-disjoint
/// proofs at proc-macro construction time.
fn emit_attention_via_cache_push(
    q_slot: u32,
    attn_out_slot: u32,
    layer: u32,
    interleaved: bool,
    is_sliding: bool,
    state: &mut MegaDispatchState,
) -> Result<TokenStream, String> {
    let lit = Literal::u32_unsuffixed;
    let q_id = lit(q_slot);
    let out_id = lit(attn_out_slot);

    // BLOCK_SIZE is fixed at 16 by the Attn-tier kernel signature
    // (block_table indexes into pages of 16-token granularity).
    let block_size_const: u32 = 16;

    // K_smem and V_smem each occupy a full substrate page (TK 2.0
    // layout — see `feedback_ff_mega_skip_guards` follow-on).
    // The Node's const assert verifies PAGE_SIZE >= KV block bytes.
    let k_smem_page_id_const = state.alloc_distinct(&[q_slot, attn_out_slot])?;
    let v_smem_page_id_const =
        state.alloc_distinct(&[q_slot, attn_out_slot, k_smem_page_id_const])?;

    // Score / PV scratch tiles — small reduction buffers carried by
    // the substrate but not addressed by the current render. Cap at
    // 256 B each (fits TK's 1024 B scratch with headroom).
    let score_off_const: u32 = 0;
    let score_bytes_const: u32 = 256;
    let pv_off_const: u32 = score_off_const + score_bytes_const;
    let pv_bytes_const: u32 = 256;

    let score_off = lit(score_off_const);
    let score_bytes = lit(score_bytes_const);
    let pv_off = lit(pv_off_const);
    let pv_bytes = lit(pv_bytes_const);
    let k_smem_page_id = lit(k_smem_page_id_const);
    let v_smem_page_id = lit(v_smem_page_id_const);

    let iters = lit(1u32);
    let arrives = lit(state.arrives);
    let num_layers = lit(state.num_layers);
    let layer_lit = lit(layer);
    let head_dim = lit(state.head_dim);
    let num_q_heads = lit(state.num_q_heads);
    let num_kv_heads = lit(state.num_kv_heads);
    let block_size = lit(block_size_const);
    let num_tokens = lit(state.num_tokens);
    let max_sk = lit(state.sk_bucket.max(1));
    let q_in_act_slot = lit(q_slot);
    let attn_out_act_slot = lit(attn_out_slot);
    let interleaved_lit = interleaved;
    let attn_scale_lit = state.attn_scale;
    let attn_softcap_lit = state.attn_softcap;
    let num_pages_lit = lit(state.num_pages_budget);
    let scratch_lit = lit(state.scratch_bytes);

    let kind_expr = if is_sliding {
        let sliding_window_val = if state.sliding_window > 0 {
            state.sliding_window
        } else {
            4096
        };
        let sliding_window_lit = lit(sliding_window_val);
        quote! { ::ferrite_megakernel::ir::AttentionKind::Sliding(#sliding_window_lit) }
    } else {
        quote! { ::ferrite_megakernel::ir::AttentionKind::Full }
    };

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
            ::ferrite_megakernel::ir::PageId::<#k_smem_page_id, #num_pages_lit>::new(),
            ::ferrite_megakernel::ir::PageId::<#v_smem_page_id, #num_pages_lit>::new(),
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
            #kind_expr,
            #interleaved_lit,
            #attn_scale_lit,
            #attn_softcap_lit,
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
    let iters_const = 1_u32;
    let iters = lit(iters_const);
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
    let ncw = state.num_consumer_warps;
    let tile_n_const = if ncw > 0 && n % ncw == 0 { n / ncw } else { n };
    let tile_n_lit = lit(tile_n_const);
    let chunk_k_lit = lit(k / iters_const);
    let consumer_bar_reduce = lit(1u32);
    let consumer_bar_publish = lit(2u32);
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
            ::ferrite_megakernel::ir::TileN::<#tile_n_lit>::new(),
            ::ferrite_megakernel::ir::ChunkK::<#chunk_k_lit>::new(),
            ::ferrite_megakernel::ir::BarSyncId::<#consumer_bar_reduce>::new(),
            ::ferrite_megakernel::ir::BarSyncId::<#consumer_bar_publish>::new(),
            ::ferrite_megakernel::ir::BarSyncPair::<
                #consumer_bar_reduce,
                #consumer_bar_publish,
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
        // TkFusedAddRmsNormGemm / TkFusedAddScalarOffsetRmsNormGemm /
        // TkGemmAdd pass through unchanged — there is no shape-named
        // canonical for these in the frontend Instruction enum, and
        // megakernel must not match on vendor-named variants. Dispatch
        // matches `I::Tk*` directly.
        passthrough @ (I::TkFusedAddRmsNormGemm(..)
            | I::TkFusedAddScalarOffsetRmsNormGemm(..)
            | I::TkGemmAdd(..)) => passthrough,
        // ── Shape-renames within the shape-named family. These fold
        // semantically-equivalent shape-named variants onto the
        // canonical that megakernel dispatches.
        I::MeanSubRmsNorm(in_slot, out_slot, layer) => {
            I::RmsNorm(in_slot, out_slot, layer)
        }
        I::MeanSubRmsNormBiasAdd(in_slot, out_slot, layer) => {
            I::RmsNorm(in_slot, out_slot, layer)
        }
        I::FusedGemmBias(in_slot, out_slot, layer) => {
            I::Gemm(in_slot, out_slot, layer, 1, 1)
        }
        I::PosEmbed(out_slot) => I::Embed(out_slot),
        I::EncoderAttention(q_slot, _k_slot, _v_slot, out_slot) => {
            I::AttentionViaCache(q_slot, out_slot, /*layer=*/ 0, /*interleaved=*/ false)
        }
        I::FusedQkvRopePrefill(in_slot, out_slot, layer, _n, _k, biased, interleaved) => {
            I::FusedQkvRopeCache(in_slot, out_slot, layer, biased, interleaved)
        }
        // ── Vendor-named variants are never mega-reachable (their
        // backend Impls don't have Tk peers, so the cost DP picks
        // them only when megakernel is OFF; mega tape claimer
        // rejects any tape containing them). Per
        // [[feedback-no-cutlass-in-mega]], megakernel must not
        // pattern-match on those names. They pass through unchanged
        // here; if one ever reaches dispatch, dispatch's `_` arm
        // surfaces it as `<non-mega>`.
        other => other,
    }
}

/// Dispatch one `Instruction` to a `bodies.push(::ferrite_megakernel::
/// cuda_emit::render::render_*::<...>(...))` token stream paralleling
/// [`dispatch_instruction_to_push`]. The proc-macro emits these calls
/// into a per-canonical `emit_for_canonical_<canonical>()` fn whose
/// body is a sequence of literal `render_*` invocations sharing the
/// same const-generic literals as the parallel `b.push_*::<...>(...)`
/// tape builder calls.
///
/// Returns `Ok(None)` for variants whose `render_*` fn isn't yet
/// written — the caller skips the entire canonical's emit fn when
/// ANY Instruction returns `None` (the `.cu` artifact is only
/// meaningful when every node has a render binding).
///
/// State semantics MIRROR [`dispatch_instruction_to_push`]: same
/// `state.alloc_distinct` order, same `state.arrives` /
/// `state.next_weight_accessor` bumps. The proc-macro walks the
/// Instruction list TWICE — once for push (writes
/// `build_mega_tape_<canonical>` body), once for render (writes
/// `emit_for_canonical_<canonical>` body) — sharing a single
/// [`MegaDispatchState`] checkpoint between walks.
pub fn dispatch_instruction_to_render(
    instr: &ferrite_forward::Instruction,
    weight_paths: &[String],
    state: &mut MegaDispatchState,
    layer_override: Option<u32>,
) -> Result<Option<TokenStream>, String> {
    use ferrite_forward::Instruction as I;

    let lit = Literal::u32_unsuffixed;
    let resolved_layer = |default: u32| layer_override.unwrap_or(default);

    match instr {
        I::RmsNorm(in_slot, out_slot, layer) => {
            let _weight = weight_paths
                .first()
                .ok_or_else(|| "weight_paths empty".to_string())?;
            let in_id = lit(*in_slot);
            let weight_id = lit(state.alloc_distinct(&[*in_slot])?);
            let layer_lit = lit(resolved_layer(*layer));
            let hidden_dim = lit(state.hidden_dim);
            let num_tokens = lit(state.num_tokens);
            let in_act_slot = lit(*in_slot);
            let out_act_slot = lit(*out_slot);
            let weight_accessor_idx = lit(state.next_weight_accessor);
            let ncw = state.num_consumer_warps;
            let k_per_warp = lit(state.hidden_dim / ncw.max(1));
            let ncw_lit = lit(ncw);
            let num_layers = lit(state.num_layers);
            let bar_reduce = lit(1u32);
            let bar_publish = lit(2u32);
            let partial_offset = lit(0u32);
            let eps_lit = state.rms_norm_eps;
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit(state.arrives & 1);
            state.arrives += 1;
            state.next_weight_accessor += 1;
            Ok(Some(quote! {
                bodies.push(::ferrite_megakernel::cuda_emit::render::render_rms_norm::<
                    #hidden_dim, #num_tokens, #ncw_lit, #k_per_warp, #num_layers,
                >(
                    #in_id, #weight_id,
                    #consumer_phase, #storer_phase,
                    #layer_lit,
                    #in_act_slot, #out_act_slot,
                    #weight_accessor_idx,
                    #bar_reduce, #bar_publish,
                    #partial_offset,
                    #eps_lit,
                ));
            }))
        }
        I::Add(delta_slot, residual_slot) => {
            let delta_id = lit(*delta_slot);
            let residual_id = lit(*residual_slot);
            let hidden_dim = lit(state.hidden_dim);
            let num_tokens = lit(state.num_tokens);
            let delta_act_slot = lit(*delta_slot);
            let residual_act_slot = lit(*residual_slot);
            let ncw = state.num_consumer_warps;
            let k_per_warp = lit(state.hidden_dim / ncw.max(1));
            let ncw_lit = lit(ncw);
            let bar_publish = lit(2u32);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit(state.arrives & 1);
            state.arrives += 1;
            Ok(Some(quote! {
                bodies.push(::ferrite_megakernel::cuda_emit::render::render_add::<
                    #hidden_dim, #num_tokens, #ncw_lit, #k_per_warp,
                >(
                    #delta_id, #residual_id,
                    #consumer_phase, #storer_phase,
                    #delta_act_slot, #residual_act_slot,
                    #bar_publish,
                ));
            }))
        }
        I::Embed(out_slot) => {
            let _weight = weight_paths
                .first()
                .ok_or_else(|| "Embed weight_paths empty".to_string())?;
            let out_id = lit(*out_slot);
            let _weight_id = lit(state.alloc_distinct(&[*out_slot])?);
            let hidden_dim = lit(state.hidden_dim);
            let num_tokens = lit(state.num_tokens);
            let out_act_slot = lit(*out_slot);
            let weight_accessor_idx = lit(state.next_weight_accessor);
            let num_layers = lit(state.num_layers);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit(state.arrives & 1);
            state.arrives += 1;
            state.next_weight_accessor += 1;
            Ok(Some(quote! {
                bodies.push(::ferrite_megakernel::cuda_emit::render::render_embed::<
                    #hidden_dim, #num_tokens, #num_layers,
                >(
                    #out_id,
                    #consumer_phase, #storer_phase,
                    #out_act_slot,
                    #weight_accessor_idx,
                ));
            }))
        }
        I::ScalarMul(in_slot, out_slot, scale) => {
            let in_id = lit(*in_slot);
            let out_id = lit(*out_slot);
            let hidden_dim = lit(state.hidden_dim);
            let num_tokens = lit(state.num_tokens);
            let in_act_slot = lit(*in_slot);
            let out_act_slot = lit(*out_slot);
            let scale_lit = *scale;
            let ncw = state.num_consumer_warps;
            let k_per_warp = lit(state.hidden_dim / ncw.max(1));
            let ncw_lit = lit(ncw);
            let bar_publish = lit(2u32);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit(state.arrives & 1);
            state.arrives += 1;
            Ok(Some(quote! {
                bodies.push(::ferrite_megakernel::cuda_emit::render::render_scalar_mul::<
                    #hidden_dim, #num_tokens, #ncw_lit, #k_per_warp,
                >(
                    #in_id, #out_id,
                    #consumer_phase, #storer_phase,
                    #in_act_slot, #out_act_slot,
                    #bar_publish,
                    #scale_lit,
                ));
            }))
        }
        I::TanhSoftCap(in_slot, out_slot) => {
            let in_id = lit(*in_slot);
            let out_id = lit(*out_slot);
            let hidden_dim = lit(state.hidden_dim);
            let num_tokens = lit(state.num_tokens);
            let in_act_slot = lit(*in_slot);
            let out_act_slot = lit(*out_slot);
            let cap_lit = state.tanh_soft_cap;
            let ncw = state.num_consumer_warps;
            let k_per_warp = lit(state.hidden_dim / ncw.max(1));
            let ncw_lit = lit(ncw);
            let bar_publish = lit(2u32);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit(state.arrives & 1);
            state.arrives += 1;
            Ok(Some(quote! {
                bodies.push(::ferrite_megakernel::cuda_emit::render::render_tanh_soft_cap::<
                    #hidden_dim, #num_tokens, #ncw_lit, #k_per_warp,
                >(
                    #in_id, #out_id,
                    #consumer_phase, #storer_phase,
                    #in_act_slot, #out_act_slot,
                    #bar_publish,
                    #cap_lit,
                ));
            }))
        }
        I::ScalarOffsetRmsNorm(in_slot, out_slot, layer, offset) => {
            let _weight = weight_paths
                .first()
                .ok_or_else(|| "ScalarOffsetRmsNorm weight_paths empty".to_string())?;
            let in_id = lit(*in_slot);
            let weight_id = lit(state.alloc_distinct(&[*in_slot])?);
            let layer_lit = lit(resolved_layer(*layer));
            let hidden_dim = lit(state.hidden_dim);
            let num_tokens = lit(state.num_tokens);
            let in_act_slot = lit(*in_slot);
            let out_act_slot = lit(*out_slot);
            let weight_accessor_idx = lit(state.next_weight_accessor);
            let offset_lit = *offset;
            let eps_lit = state.rms_norm_eps;
            let ncw = state.num_consumer_warps;
            let k_per_warp = lit(state.hidden_dim / ncw.max(1));
            let ncw_lit = lit(ncw);
            let num_layers = lit(state.num_layers);
            let bar_reduce = lit(1u32);
            let bar_publish = lit(2u32);
            let partial_offset = lit(0u32);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit(state.arrives & 1);
            state.arrives += 1;
            state.next_weight_accessor += 1;
            Ok(Some(quote! {
                bodies.push(::ferrite_megakernel::cuda_emit::render::render_scalar_offset_rms_norm::<
                    #hidden_dim, #num_tokens, #ncw_lit, #k_per_warp, #num_layers,
                >(
                    #in_id, #weight_id,
                    #consumer_phase, #storer_phase,
                    #layer_lit,
                    #in_act_slot, #out_act_slot,
                    #weight_accessor_idx,
                    #bar_reduce, #bar_publish,
                    #partial_offset,
                    #eps_lit, #offset_lit,
                ));
            }))
        }
        I::Gemm(in_slot, out_slot, layer, n, k) => {
            let _weight = weight_paths
                .first()
                .ok_or_else(|| "Gemm weight_paths empty".to_string())?;
            let in_id = lit(*in_slot);
            let out_id = lit(*out_slot);
            let weight_id = lit(state.alloc_distinct(&[*in_slot, *out_slot])?);
            let iters_const = 1_u32;
            let iters = lit(iters_const);
            let ncw = state.num_consumer_warps;
            let tile_n_const = if ncw > 0 && n % ncw == 0 { n / ncw } else { *n };
            let tile_n_lit = lit(tile_n_const);
            let layer_lit = lit(resolved_layer(*layer));
            let n_lit = lit(*n);
            let k_lit = lit(*k);
            let m_lit = lit(state.num_tokens);
            let in_act_slot = lit(*in_slot);
            let out_act_slot = lit(*out_slot);
            let weight_accessor_idx = lit(state.next_weight_accessor);
            let num_layers = lit(state.num_layers);
            let ncw_lit = lit(ncw);
            let bar_publish = lit(1u32);
            let b_tile_offset = lit(0u32);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit(state.arrives & 1);
            state.arrives += 1;
            state.next_weight_accessor += 1;
            Ok(Some(quote! {
                bodies.push(::ferrite_megakernel::cuda_emit::render::render_gemm::<
                    #m_lit, #k_lit, #n_lit, #tile_n_lit, #ncw_lit, #num_layers, #iters,
                >(
                    #in_id, #weight_id, #out_id,
                    #consumer_phase, #storer_phase,
                    #layer_lit,
                    #in_act_slot, #out_act_slot,
                    #weight_accessor_idx,
                    #bar_publish,
                    #b_tile_offset,
                ));
            }))
        }
        I::FusedAddRmsNorm(delta_slot, residual_slot, layer) => {
            let _weight = weight_paths
                .first()
                .ok_or_else(|| "FusedAddRmsNorm weight_paths empty".to_string())?;
            let delta_id = lit(*delta_slot);
            let residual_id = lit(*residual_slot);
            let weight_id = lit(state.alloc_distinct(&[*delta_slot, *residual_slot])?);
            let layer_lit = lit(resolved_layer(*layer));
            let hidden_dim = lit(state.hidden_dim);
            let num_tokens = lit(state.num_tokens);
            let delta_act_slot = lit(*delta_slot);
            let residual_act_slot = lit(*residual_slot);
            let weight_accessor_idx = lit(state.next_weight_accessor);
            let eps_lit = state.rms_norm_eps;
            let ncw = state.num_consumer_warps;
            let k_per_warp = lit(state.hidden_dim / ncw.max(1));
            let ncw_lit = lit(ncw);
            let num_layers = lit(state.num_layers);
            let bar_reduce = lit(1u32);
            let bar_publish = lit(2u32);
            let partial_offset = lit(0u32);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit(state.arrives & 1);
            state.arrives += 1;
            state.next_weight_accessor += 1;
            Ok(Some(quote! {
                bodies.push(::ferrite_megakernel::cuda_emit::render::render_fused_add_rms_norm::<
                    #hidden_dim, #num_tokens, #ncw_lit, #k_per_warp, #num_layers,
                >(
                    #delta_id, #residual_id, #weight_id,
                    #consumer_phase, #storer_phase,
                    #layer_lit,
                    #delta_act_slot, #residual_act_slot,
                    #weight_accessor_idx,
                    #bar_reduce, #bar_publish,
                    #partial_offset,
                    #eps_lit,
                ));
            }))
        }
        I::FusedGateUpSiluMul(in_slot, out_slot, layer)
        | I::FusedGateUpGeluMul(in_slot, out_slot, layer) => {
            let _weight = weight_paths
                .first()
                .ok_or_else(|| "FusedGateUp*Mul weight_paths empty".to_string())?;
            let activation_path = match instr {
                I::FusedGateUpSiluMul(..) => {
                    quote! { ::ferrite_megakernel::ir::nodes::GateUpActivation::Silu }
                }
                I::FusedGateUpGeluMul(..) => {
                    quote! { ::ferrite_megakernel::ir::nodes::GateUpActivation::Gelu }
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
            let iters = lit(1u32);
            let layer_lit = lit(resolved_layer(*layer));
            let hidden_dim = lit(state.hidden_dim);
            let intermediate_dim = lit(state.intermediate_dim);
            let m_lit = lit(state.num_tokens);
            let in_act_slot = lit(*in_slot);
            let out_act_slot = lit(*out_slot);
            let weight_accessor_idx = lit(state.next_weight_accessor);
            let ncw = state.num_consumer_warps;
            let tile_n_const = if ncw > 0 && state.intermediate_dim % ncw == 0 {
                state.intermediate_dim / ncw
            } else {
                state.intermediate_dim
            };
            let tile_n_lit = lit(tile_n_const);
            let ncw_lit = lit(ncw);
            let num_layers = lit(state.num_layers);
            let bar_publish = lit(1u32);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit(state.arrives & 1);
            state.arrives += 1;
            state.next_weight_accessor += 1;
            Ok(Some(quote! {
                bodies.push(::ferrite_megakernel::cuda_emit::render::render_fused_gate_up_activate_mul::<
                    #m_lit, #hidden_dim, #intermediate_dim, #tile_n_lit, #ncw_lit, #num_layers, #iters,
                >(
                    #in_id, #weight_id, #out_id,
                    #consumer_phase, #storer_phase,
                    #layer_lit,
                    #in_act_slot, #out_act_slot,
                    #weight_accessor_idx,
                    #bar_publish,
                    #gate_off, #up_off, #gate_bytes, #up_bytes,
                    #activation_path,
                ));
            }))
        }
        I::FusedQkvRopeCache(in_slot, out_slot, layer, _biased, _interleaved) => {
            if weight_paths.len() != 2 {
                return Err(format!(
                    "FusedQkvRopeCache expected 2 weight_paths (qkv, rotary), got {}",
                    weight_paths.len()
                ));
            }
            let in_id = lit(*in_slot);
            let qkv_id = lit(state.alloc_distinct(&[*in_slot])?);
            let cs_id = lit(state.alloc_distinct(&[*in_slot])?);
            let q_id = lit(state.alloc_distinct(&[*in_slot])?);
            let k_id = lit(state.alloc_distinct(&[*in_slot])?);
            let v_id = lit(state.alloc_distinct(&[*in_slot])?);
            let quarter = state.scratch_bytes / 4;
            let q_off = lit(0u32);
            let k_off = lit(quarter);
            let b_tile_off = lit(2 * quarter);
            let iters_const = 1_u32;
            let iters = lit(iters_const);
            let layer_lit = lit(resolved_layer(*layer));
            let hidden_dim = lit(state.hidden_dim);
            let head_dim = lit(state.head_dim);
            let num_q_heads_lit = lit(state.num_q_heads);
            let num_kv_heads_lit = lit(state.num_kv_heads);
            let q_dim_lit = lit(state.num_q_heads * state.head_dim);
            let kv_dim_lit = lit(state.num_kv_heads * state.head_dim);
            let qkv_n_val = (state.num_q_heads + 2 * state.num_kv_heads) * state.head_dim;
            let qkv_n_lit = lit(qkv_n_val);
            let m_lit = lit(state.num_tokens);
            let in_act_slot = lit(*in_slot);
            let q_out_act_slot = lit(*out_slot);
            let k_out_act_slot = lit(out_slot.wrapping_add(1));
            let v_out_act_slot = lit(out_slot.wrapping_add(2));
            let qkv_weight_accessor = lit(state.next_weight_accessor);
            let rotary_accessor = lit(state.next_weight_accessor + 1);
            let ncw = state.num_consumer_warps;
            let tile_n_const = if ncw > 0 && qkv_n_val % ncw == 0 {
                qkv_n_val / ncw
            } else {
                qkv_n_val
            };
            let tile_n_lit = lit(tile_n_const);
            let heads_per_warp_const = if state.head_dim > 0 {
                tile_n_const / state.head_dim
            } else {
                0
            };
            let heads_per_warp_lit = lit(heads_per_warp_const);
            let ncw_lit = lit(ncw);
            let num_layers = lit(state.num_layers);
            let bar_publish = lit(1u32);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit(state.arrives & 1);
            state.arrives += 1;
            state.next_weight_accessor += 2;
            Ok(Some(quote! {
                bodies.push(::ferrite_megakernel::cuda_emit::render::render_fused_qkv_rope_cache::<
                    #m_lit, #hidden_dim, #head_dim, #num_q_heads_lit, #num_kv_heads_lit,
                    #q_dim_lit, #kv_dim_lit, #qkv_n_lit, #tile_n_lit, #heads_per_warp_lit,
                    #ncw_lit, #num_layers, #iters,
                >(
                    #in_id, #qkv_id, #cs_id, #q_id, #k_id, #v_id,
                    #consumer_phase, #storer_phase,
                    #layer_lit,
                    #in_act_slot, #q_out_act_slot, #k_out_act_slot, #v_out_act_slot,
                    #qkv_weight_accessor, #rotary_accessor,
                    #bar_publish,
                    #q_off, #k_off, #b_tile_off,
                ));
            }))
        }
        I::SpliceMmEmbeds(slot) => {
            let slot_id = lit(*slot);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit(state.arrives & 1);
            state.arrives += 1;
            Ok(Some(quote! {
                bodies.push(::ferrite_megakernel::cuda_emit::render::render_splice_mm_embeds(
                    #slot_id, #consumer_phase, #storer_phase,
                ));
            }))
        }
        I::BarrierSignal(edge) => {
            let edge_lit = lit(*edge);
            Ok(Some(quote! {
                bodies.push(::ferrite_megakernel::cuda_emit::render::render_barrier_signal(
                    #edge_lit,
                ));
            }))
        }
        I::BarrierWait(edge, count) => {
            let edge_lit = lit(*edge);
            let count_lit = lit(*count);
            Ok(Some(quote! {
                bodies.push(::ferrite_megakernel::cuda_emit::render::render_barrier_wait(
                    #edge_lit, #count_lit,
                ));
            }))
        }
        I::TkGemmAdd(in_slot, residual_slot, layer, n, k, _k_offset, _k_full) => Ok(Some(
            render_tk_gemm_add(
                *in_slot,
                *residual_slot,
                resolved_layer(*layer),
                *n,
                *k,
                weight_paths,
                state,
            )?,
        )),
        I::TkFusedAddRmsNormGemm(delta_slot, residual_slot, out_slot, layer, n, k) => {
            Ok(Some(render_lm_head_with_delta(
                *residual_slot,
                *delta_slot,
                *out_slot,
                resolved_layer(*layer),
                *n,
                *k,
                quote! { ::ferrite_megakernel::ir::LmHeadNormKind::AddRmsNorm },
                None,
                weight_paths,
                state,
            )?))
        }
        I::TkFusedAddScalarOffsetRmsNormGemm(
            delta_slot,
            residual_slot,
            out_slot,
            layer,
            offset,
            n,
            k,
        ) => Ok(Some(render_lm_head_with_delta(
            *residual_slot,
            *delta_slot,
            *out_slot,
            resolved_layer(*layer),
            *n,
            *k,
            quote! { ::ferrite_megakernel::ir::LmHeadNormKind::AddScalarOffsetRmsNorm },
            Some(*offset),
            weight_paths,
            state,
        )?)),
        I::AttentionViaCache(q_slot, attn_out_slot, layer, interleaved) => Ok(Some(
            render_attention_via_cache_dispatch(
                *q_slot,
                *attn_out_slot,
                resolved_layer(*layer),
                *interleaved,
                /*is_sliding=*/ false,
                state,
            )?,
        )),
        I::SlidingAttentionViaCache(q_slot, attn_out_slot, layer, interleaved) => Ok(Some(
            render_attention_via_cache_dispatch(
                *q_slot,
                *attn_out_slot,
                resolved_layer(*layer),
                *interleaved,
                /*is_sliding=*/ true,
                state,
            )?,
        )),
        // RopeAppend mirrors the push side, which routes through
        // push_fused_qkv_rope_cache with a sentinel qkv weight path
        // (the runtime kernel handles the no-qkv-matmul shape via the
        // sentinel; emit shares the FusedQkvRopeCache render fn).
        // State bumps and slot alloc order MUST match the push arm
        // verbatim — see codegen.rs RopeAppend push (alloc sequence
        // q_id, qkv_id, cs_id, k_id, v_id with progressive
        // exclude lists from in_id_val = q_slot).
        I::RopeAppend(
            q_slot,
            _k_slot,
            _v_slot,
            _q_out_slot,
            _k_out_slot,
            _v_out_slot,
            layer,
            _interleaved,
        ) => {
            if weight_paths.len() != 1 {
                return Err(format!(
                    "RopeAppend expected 1 weight_path (rotary), got {}",
                    weight_paths.len()
                ));
            }
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
            let qkv_id = lit(qkv_id_val);
            let cs_id = lit(cs_id_val);
            let q_id = lit(q_id_val);
            let k_id = lit(k_id_val);
            let v_id = lit(v_id_val);
            let quarter = state.scratch_bytes / 4;
            let q_off = lit(0u32);
            let k_off = lit(quarter);
            let b_tile_off = lit(2 * quarter);
            let iters_const = 1_u32;
            let iters = lit(iters_const);
            let layer_lit = lit(resolved_layer(*layer));
            let hidden_dim = lit(state.hidden_dim);
            let head_dim = lit(state.head_dim);
            let num_q_heads_lit = lit(state.num_q_heads);
            let num_kv_heads_lit = lit(state.num_kv_heads);
            let q_dim_lit = lit(state.num_q_heads * state.head_dim);
            let kv_dim_lit = lit(state.num_kv_heads * state.head_dim);
            let qkv_n_val = (state.num_q_heads + 2 * state.num_kv_heads) * state.head_dim;
            let qkv_n_lit = lit(qkv_n_val);
            let m_lit = lit(state.num_tokens);
            let in_act_slot = lit(in_id_val);
            let q_out_act_slot = lit(in_id_val);
            let k_out_act_slot = lit(in_id_val.wrapping_add(1));
            let v_out_act_slot = lit(in_id_val.wrapping_add(2));
            let qkv_weight_accessor = lit(state.next_weight_accessor);
            let rotary_accessor = lit(state.next_weight_accessor + 1);
            let ncw = state.num_consumer_warps;
            let tile_n_const = if ncw > 0 && qkv_n_val % ncw == 0 {
                qkv_n_val / ncw
            } else {
                qkv_n_val
            };
            let tile_n_lit = lit(tile_n_const);
            let heads_per_warp_const = if state.head_dim > 0 {
                tile_n_const / state.head_dim
            } else {
                0
            };
            let heads_per_warp_lit = lit(heads_per_warp_const);
            let ncw_lit = lit(ncw);
            let num_layers = lit(state.num_layers);
            let bar_publish = lit(1u32);
            let consumer_phase = lit(state.arrives & 1);
            let storer_phase = lit(state.arrives & 1);
            state.arrives += 1;
            state.next_weight_accessor += 2;
            Ok(Some(quote! {
                bodies.push(::ferrite_megakernel::cuda_emit::render::render_fused_qkv_rope_cache::<
                    #m_lit, #hidden_dim, #head_dim, #num_q_heads_lit, #num_kv_heads_lit,
                    #q_dim_lit, #kv_dim_lit, #qkv_n_lit, #tile_n_lit, #heads_per_warp_lit,
                    #ncw_lit, #num_layers, #iters,
                >(
                    #in_id, #qkv_id, #cs_id, #q_id, #k_id, #v_id,
                    #consumer_phase, #storer_phase,
                    #layer_lit,
                    #in_act_slot, #q_out_act_slot, #k_out_act_slot, #v_out_act_slot,
                    #qkv_weight_accessor, #rotary_accessor,
                    #bar_publish,
                    #q_off, #k_off, #b_tile_off,
                ));
            }))
        }
        // Variants wired in dispatch_to_push but whose render_*
        // counterpart hasn't been written yet — the proc-macro skips
        // emit_for_canonical for any tape that contains them.
        _ => Ok(None),
    }
}

/// Build the
/// `bodies.push(render_attention_via_cache::<…>(…))` /
/// `bodies.push(render_sliding_attention_via_cache::<…>(…))`
/// token stream. Mirrors the exact const-generic + scratch-layout
/// arithmetic in [`emit_attention_via_cache_push`] (the proc-macro
/// walks the Instruction list TWICE — once for push, once for
/// render — sharing the same [`MegaDispatchState`] checkpoint, so
/// the const-generic args must agree to the literal).
fn render_attention_via_cache_dispatch(
    q_slot: u32,
    attn_out_slot: u32,
    layer: u32,
    interleaved: bool,
    is_sliding: bool,
    state: &mut MegaDispatchState,
) -> Result<TokenStream, String> {
    let lit = Literal::u32_unsuffixed;
    let q_id = lit(q_slot);
    let out_id = lit(attn_out_slot);

    // Page-id allocator order MUST match `emit_attention_via_cache_push`
    // verbatim — the proc-macro walks the Instruction list TWICE (once
    // for push, once for render) sharing the same `MegaDispatchState`
    // checkpoint. K_smem and V_smem each occupy a full substrate page
    // (TK 2.0 PAGE_SIZE=16384 ≥ KV block bytes; verified by Node const
    // assert).
    let block_size_const: u32 = 16;
    let k_smem_page_id_const = state.alloc_distinct(&[q_slot, attn_out_slot])?;
    let v_smem_page_id_const =
        state.alloc_distinct(&[q_slot, attn_out_slot, k_smem_page_id_const])?;

    // Score / PV scratch — small reduction buffers, 256 B each (must
    // match push side).
    let score_off_const: u32 = 0;
    let pv_off_const: u32 = 256;

    let score_off = lit(score_off_const);
    let pv_off = lit(pv_off_const);
    let k_smem_page_id = lit(k_smem_page_id_const);
    let v_smem_page_id = lit(v_smem_page_id_const);

    let iters = lit(1u32);
    let layer_lit = lit(layer);
    let head_dim = lit(state.head_dim);
    let num_q_heads = lit(state.num_q_heads);
    let num_kv_heads = lit(state.num_kv_heads);
    let block_size = lit(block_size_const);
    let m_lit = lit(state.num_tokens);
    let max_sk = lit(state.sk_bucket.max(1));
    let q_in_act_slot = lit(q_slot);
    let attn_out_act_slot = lit(attn_out_slot);
    let interleaved_lit = interleaved;
    let attn_scale_lit = state.attn_scale;
    let attn_softcap_lit = state.attn_softcap;
    let ncw = state.num_consumer_warps;
    let ncw_lit = lit(ncw);
    let num_layers = lit(state.num_layers);

    let consumer_phase = lit(state.arrives & 1);
    let storer_phase = lit(state.arrives & 1);
    state.arrives += 1;

    if is_sliding {
        let sliding_window_val = if state.sliding_window > 0 {
            state.sliding_window
        } else {
            4096
        };
        let sliding_window_lit = lit(sliding_window_val);
        Ok(quote! {
            bodies.push(::ferrite_megakernel::cuda_emit::render::render_sliding_attention_via_cache::<
                #m_lit, #head_dim, #num_q_heads, #num_kv_heads, #block_size,
                #max_sk, #ncw_lit, #num_layers, #iters,
            >(
                #q_id, #out_id,
                #consumer_phase, #storer_phase,
                #layer_lit,
                #q_in_act_slot, #attn_out_act_slot,
                #score_off, #pv_off, #k_smem_page_id, #v_smem_page_id,
                #attn_scale_lit, #attn_softcap_lit,
                #interleaved_lit,
                #sliding_window_lit,
            ));
        })
    } else {
        Ok(quote! {
            bodies.push(::ferrite_megakernel::cuda_emit::render::render_attention_via_cache::<
                #m_lit, #head_dim, #num_q_heads, #num_kv_heads, #block_size,
                #max_sk, #ncw_lit, #num_layers, #iters,
            >(
                #q_id, #out_id,
                #consumer_phase, #storer_phase,
                #layer_lit,
                #q_in_act_slot, #attn_out_act_slot,
                #score_off, #pv_off, #k_smem_page_id, #v_smem_page_id,
                #attn_scale_lit, #attn_softcap_lit,
                #interleaved_lit,
            ));
        })
    }
}

/// Render-side mirror of the `TkFusedAddRmsNormGemm` /
/// `TkFusedAddScalarOffsetRmsNormGemm` push arms (lm_head with
/// residual fold). Emits a
/// `bodies.push(::ferrite_megakernel::cuda_emit::render::render_tk_fused_norm_gemm::<…>(…));`
/// with const generics and runtime args reflecting the same
/// per-canonical [`MegaDispatchState`] decisions the push walk made
/// (page id allocator, arrives++, weight_accessor++, scratch layout,
/// bar IDs).
#[allow(clippy::too_many_arguments)]
fn render_lm_head_with_delta(
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
    let lit = Literal::u32_unsuffixed;
    let in_id = lit(residual_slot);
    let delta_id = lit(delta_slot);
    let out_id = lit(out_slot);
    let norm_w_id = lit(state.alloc_distinct(&[residual_slot, delta_slot, out_slot])?);
    let lin_w_id = lit(state.alloc_distinct(&[residual_slot, delta_slot, out_slot])?);
    let partial_off = lit(0u32);
    let b_tile_off = lit(state.num_consumer_warps * 4);
    let iters_const = 1_u32;
    let iters = lit(iters_const);
    let layer_lit = lit(layer);
    let n_lit = lit(n);
    let k_lit = lit(k);
    let m_lit = lit(state.num_tokens);
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
    let ncw = state.num_consumer_warps;
    let tile_n_const = if ncw > 0 && n % ncw == 0 { n / ncw } else { n };
    let tile_n_lit = lit(tile_n_const);
    let k_per_warp = lit(k / ncw.max(1));
    let ncw_lit = lit(ncw);
    let num_layers = lit(state.num_layers);
    let bar_reduce = lit(1u32);
    let bar_publish = lit(2u32);
    let consumer_phase = lit(state.arrives & 1);
    let storer_phase = lit(state.arrives & 1);
    state.arrives += 1;
    state.next_weight_accessor += 2;
    Ok(quote! {
        bodies.push(::ferrite_megakernel::cuda_emit::render::render_tk_fused_norm_gemm::<
            #m_lit, #k_lit, #n_lit, #tile_n_lit, #ncw_lit, #k_per_warp, #num_layers, #iters,
        >(
            #in_id,
            ::core::option::Option::Some(#delta_id),
            #norm_w_id, #lin_w_id, #out_id,
            #consumer_phase, #storer_phase,
            #layer_lit,
            #in_act_slot,
            ::core::option::Option::Some(#delta_act_slot),
            #out_act_slot,
            #norm_weight_accessor_idx, #linear_weight_accessor_idx,
            #bar_reduce, #bar_publish,
            #eps_lit,
            #norm_kind_path,
            #offset_expr,
            #b_tile_off,
            #partial_off,
        ));
    })
}

/// Render-side mirror of the `TkGemmAdd` push arm. Emits a
/// `bodies.push(::ferrite_megakernel::cuda_emit::render::render_tk_fused_gemm_add::<…>(…));`.
/// State semantics mirror the push side: one
/// `alloc_distinct(&[in, residual])` for the weight page,
/// `arrives += 1`, `next_weight_accessor += 1`. ITERS=1 per the
/// substrate invariant; the chunked-K `k_offset`/`k_full` fields on
/// the Tk variant are not consumed today (one chunk = full K).
#[allow(clippy::too_many_arguments)]
fn render_tk_gemm_add(
    in_slot: u32,
    residual_slot: u32,
    layer: u32,
    n: u32,
    k: u32,
    weight_paths: &[String],
    state: &mut MegaDispatchState,
) -> Result<TokenStream, String> {
    let _weight = weight_paths
        .first()
        .ok_or_else(|| "TkGemmAdd weight_paths empty".to_string())?;
    let lit = Literal::u32_unsuffixed;
    let in_id = lit(in_slot);
    let residual_id = lit(residual_slot);
    let weight_id = lit(state.alloc_distinct(&[in_slot, residual_slot])?);
    let iters_const = 1_u32;
    let iters = lit(iters_const);
    let ncw = state.num_consumer_warps;
    let tile_n_const = if ncw > 0 && n % ncw == 0 { n / ncw } else { n };
    let tile_n_lit = lit(tile_n_const);
    let layer_lit = lit(layer);
    let n_lit = lit(n);
    let k_lit = lit(k);
    let m_lit = lit(state.num_tokens);
    let in_act_slot = lit(in_slot);
    let residual_act_slot = lit(residual_slot);
    let weight_accessor_idx = lit(state.next_weight_accessor);
    let num_layers = lit(state.num_layers);
    let ncw_lit = lit(ncw);
    let bar_publish = lit(1u32);
    let b_tile_offset = lit(0u32);
    let consumer_phase = lit(state.arrives & 1);
    let storer_phase = lit(state.arrives & 1);
    state.arrives += 1;
    state.next_weight_accessor += 1;
    Ok(quote! {
        bodies.push(::ferrite_megakernel::cuda_emit::render::render_tk_fused_gemm_add::<
            #m_lit, #k_lit, #n_lit, #tile_n_lit, #ncw_lit, #num_layers, #iters,
        >(
            #in_id, #weight_id, #residual_id,
            #consumer_phase, #storer_phase,
            #layer_lit,
            #in_act_slot, #residual_act_slot,
            #weight_accessor_idx,
            #bar_publish,
            #b_tile_offset,
        ));
    })
}

/// Compute the [`LaunchTier`](crate::cuda_emit::LaunchTier) implied
/// by a list of Instructions — Attn ⊃ Qkv ⊃ Base. Used by the
/// proc-macro at user-build time to pick the kernel signature for
/// `emit_for_canonical_<canonical>`.
pub fn launch_tier_for_instructions(
    instructions: &[ferrite_forward::Instruction],
) -> crate::cuda_emit::LaunchTier {
    use ferrite_forward::Instruction as I;
    let mut needs_attn = false;
    let mut needs_qkv = false;
    for instr in instructions {
        // Both the host-interpreter (`AttentionViaCache`,
        // `FusedQkvRopeCache`, ...) AND the TK 2.0-decode
        // (`TkAttentionViaCache`, `TkFusedQkvRopeCache`, ...)
        // variants emit the same kernel signature args at the
        // megakernel boundary — the Tk* variants get normalized
        // to their non-Tk equivalents in
        // `dispatch_instruction_to_push` (see line 1436+).
        // Missing the Tk variants here meant Attn-tier kernels
        // were emitted at Qkv-tier signatures, surfacing as
        // `seq_lens` / `block_table` undefined at nvcc time on
        // llama and friends (their `attention(...)` lowers to
        // `TkAttentionViaCache` for decode-role canonicals).
        match instr {
            I::AttentionViaCache(..)
            | I::SlidingAttentionViaCache(..)
            | I::TkAttentionViaCache(..)
            | I::TkSlidingAttentionViaCache(..) => needs_attn = true,
            I::FusedQkvRopeCache(..) | I::TkFusedQkvRopeCache(..) | I::RopeAppend(..) => {
                needs_qkv = true
            }
            _ => {}
        }
    }
    if needs_attn {
        crate::cuda_emit::LaunchTier::Attn
    } else if needs_qkv {
        crate::cuda_emit::LaunchTier::Qkv
    } else {
        crate::cuda_emit::LaunchTier::Base
    }
}
