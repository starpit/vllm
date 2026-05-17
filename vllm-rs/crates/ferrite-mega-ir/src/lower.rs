// SPDX-License-Identifier: Apache-2.0
//! `MegaTape` builder + lowering — **const-generic** edition.
//!
//! ## Compile-time substrate proofs at the API surface
//!
//! [`MegaTapeBuilder<NUM_PAGES, NUM_CONSUMER_WARPS, PAGE_SIZE,
//! SCRATCH_BYTES, NUM_EDGES>`] is the const-generic-parameterized
//! builder. Every `push_*::<const ...>` method takes the per-op
//! load-bearing values as const generic args, threads them into the
//! variant constructor's `const {}` block, and gets monomorphization-
//! time substrate-proof discharge for free. Bad const args → rustc
//! E0080 compile error at the call site.
//!
//! ## Cross-op runtime state
//!
//! [`PagePool`] tracks "is page id N in flight right now?" — that's
//! genuinely runtime state in the proc-macro's tape walk and stays
//! runtime here (panic on cross-op alias, bug class #2 cross-op).
//! See [`crate::substrate::PagePool`] for the rationale (stable-Rust
//! limitation; truly session-typed pool needs linear types).
//!
//! ## Phase tracking
//!
//! Cumulative arrive count is runtime in the builder ([`ArriveCount`]).
//! BUT — each push method takes `ARRIVES` as a const generic AND the
//! builder verifies at runtime that `ARRIVES` matches the current
//! count, so phase parity is BOTH compile-time-checked (against the
//! caller's arrive prediction) AND runtime-verified (against the
//! actual cumulative count). The proc-macro is the ONLY caller and
//! tracks cumulative arrives at expansion time, so the compile-time
//! check is the primary line of defense; the runtime check catches
//! a hand-written builder caller's arrive miscount.
//!
//! ## `lower(...)` runtime-tape entry point
//!
//! The original `lower(&[OpInput], num_layers, substrate)` ate
//! runtime `Instruction` values and produced typed `MegaNode`s. It
//! cannot exist in the new world: const-generic constructors require
//! literal const args at the call site, so a function dispatching on
//! a runtime `Instruction` can't pass `op.in_slot` (runtime u32) as
//! a const-generic arg.
//!
//! Per the plan §C, the proc-macro will eventually emit literal
//! `builder.push_rms_norm::<0, 1, 0, 32, 0, 1, 0>(weight)` calls
//! inside the user's `#[forward]` expansion, where the proc-macro
//! tracks cumulative arrives and slot allocation at expansion time.
//! For now, [`lower`] is stubbed: it returns
//! `LowerError::NotYetLifted` for every input. The proc-macro side
//! (`emit_mega_artifacts_inline` in ferrite-forward-macro) is
//! correspondingly stubbed.

#![allow(dead_code)]

use crate::nodes::{
    Add, AttentionKind, AttentionViaCacheNode, BarrierSignal, BarrierWait, CutlassFusedNormGemm,
    Embed, FiniteF32, FusedAddRmsNorm, FusedCublasGemmAdd, FusedGateUpActivateMul,
    FusedQkvRopeCache, GateUpActivation, Gemm, LmHeadNormKind, MegaNode, RmsNorm, RotaryRef,
    ScalarMul, ScalarOffsetRmsNorm, SpliceMmEmbeds, TanhSoftCap, WeightRef,
};
use crate::substrate::{PagePool, SubstrateBudget};
use crate::tape::MegaTape;

/// Cumulative mbarrier arrive count tracked across the tape.
#[derive(Clone, Copy, Debug, Default)]
pub struct ArriveCount(u32);

impl ArriveCount {
    pub fn current(&self) -> u32 {
        self.0
    }
    pub fn bump(&mut self) {
        self.0 += 1;
    }
}

/// Builder, parameterized on the substrate constants.
///
/// Each push method takes per-op const generics and runs the
/// monomorphization-time substrate-proof discharge inside the
/// corresponding variant's `new::<...>` constructor. The runtime
/// page pool and arrive count thread cross-op state.
pub struct MegaTapeBuilder<
    const NUM_PAGES: u32,
    const NUM_CONSUMER_WARPS: u32,
    const PAGE_SIZE: u32,
    const SCRATCH_BYTES: u32,
    const NUM_EDGES: u32,
> {
    nodes: Vec<MegaNode>,
    pool: PagePool,
    arrives: ArriveCount,
}

impl<
    const NUM_PAGES: u32,
    const NUM_CONSUMER_WARPS: u32,
    const PAGE_SIZE: u32,
    const SCRATCH_BYTES: u32,
    const NUM_EDGES: u32,
> Default for MegaTapeBuilder<NUM_PAGES, NUM_CONSUMER_WARPS, PAGE_SIZE, SCRATCH_BYTES, NUM_EDGES>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
    const NUM_PAGES: u32,
    const NUM_CONSUMER_WARPS: u32,
    const PAGE_SIZE: u32,
    const SCRATCH_BYTES: u32,
    const NUM_EDGES: u32,
> MegaTapeBuilder<NUM_PAGES, NUM_CONSUMER_WARPS, PAGE_SIZE, SCRATCH_BYTES, NUM_EDGES>
{
    /// Construct a builder. The `SubstrateBudget<...>::new()`
    /// const constructor compile-fails if any of the substrate
    /// invariants is violated (NUM_PAGES > 0, etc).
    pub fn new() -> Self {
        // Run the substrate-budget invariants compile check — even
        // though we don't store the value, we do want the const
        // assertions to fire if a caller picks NUM_PAGES = 0.
        let _budget: SubstrateBudget<
            NUM_PAGES,
            NUM_CONSUMER_WARPS,
            PAGE_SIZE,
            SCRATCH_BYTES,
            NUM_EDGES,
        > = SubstrateBudget::new();
        Self {
            nodes: Vec::new(),
            pool: PagePool::new(NUM_PAGES),
            arrives: ArriveCount::default(),
        }
    }

    /// Cumulative arrive count visible to the caller — used by the
    /// proc-macro to compute the next op's `ARRIVES` const generic.
    pub fn arrives(&self) -> u32 {
        self.arrives.current()
    }

    /// Verify the caller's `ARRIVES` const-generic matches the
    /// builder's cumulative count. Runtime backstop in case the
    /// caller mistracks (the const-generic side proves phase parity
    /// against THIS const, so if the const is wrong relative to the
    /// real arrive count we'd silently emit wrong CUDA).
    fn verify_arrives(&self, arrives_const: u32, op_name: &'static str) {
        assert_eq!(
            arrives_const,
            self.arrives.current(),
            "{op_name}: ARRIVES const generic ({arrives_const}) doesn't match builder cumulative count ({})",
            self.arrives.current(),
        );
    }

    /// Push a typed RmsNorm op onto the tape.
    ///
    /// Compile-time substrate proofs (via `RmsNorm::new::<...>`
    /// const-generic constructor):
    /// - `IN_ID < NUM_PAGES`, `WEIGHT_ID < NUM_PAGES`
    /// - `IN_ID != WEIGHT_ID`
    /// - `PARTIAL_OFF + PARTIAL_BYTES <= SCRATCH_BYTES`
    /// - `LAYER < NUM_LAYERS`
    /// - `CONSUMER_PHASE == ARRIVES & 1`
    /// - `STORER_PHASE == (ARRIVES + 1) & 1`
    /// - `HIDDEN_DIM > 0`, `NUM_TOKENS > 0` (kernel-AST shape)
    ///
    /// Kernel-AST const generics (per `MEGA_IR_PLAN.md` §0/§4a/§8.0
    /// — splice into `ferrite::ops::rms_norm::<role><Config,
    /// HIDDEN_DIM, NUM_TOKENS>(...)`):
    /// - `HIDDEN_DIM`, `NUM_TOKENS` — kernel template args.
    /// - `IN_ACT_SLOT` — `act_ptrs[]` index (input row gmem ptr).
    /// - `OUT_ACT_SLOT` — `act_ptrs[]` index (storer's output ptr).
    /// - `WEIGHT_ACCESSOR_IDX` — flat index into `weight_ptrs[acc *
    ///   NUM_LAYERS + layer]` for the rms weight.
    ///
    /// Runtime args:
    /// - `weight_path` — `WeightRef` for emit-time accessor sanity.
    /// - `eps` — `FiniteF32` for the kernel's `consumer(..., float eps)`
    ///   runtime arg (rms denominator stabilizer).
    ///
    /// Runtime checks (cross-op):
    /// - PagePool refuses cross-op alias.
    /// - `ARRIVES` const generic matches the builder's cumulative count.
    #[allow(clippy::too_many_arguments)]
    pub fn push_rms_norm<
        // ARRIVES is the only const generic NOT carried by a typed
        // primitive arg (it's the builder's runtime arrive count
        // forecast). Caller turbofishes JUST `::<ARRIVES>`; the
        // remaining const generics are inferred from arg types.
        const ARRIVES: u32,
        const IN_ID: u32,
        const WEIGHT_ID: u32,
        const PARTIAL_OFF: u32,
        const PARTIAL_BYTES: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const LAYER: u32,
        const NUM_LAYERS: u32,
        const HIDDEN_DIM: u32,
        const NUM_TOKENS: u32,
        const IN_ACT_SLOT: u32,
        const OUT_ACT_SLOT: u32,
        const WEIGHT_ACCESSOR_IDX: u32,
        const CONSUMER_BAR_REDUCE: u32,
        const CONSUMER_BAR_PUBLISH: u32,
    >(
        &mut self,
        // ===== Typed primitives — each carries its substrate proof
        // in its TYPE. EVERY const generic on this method is inferred
        // from the arg types via Rust type inference; callers don't
        // turbofish, they construct the typed primitives. No
        // `/*FOO=*/` positional u32 comments — the type IS the
        // documentation.
        _arrives: crate::substrate::ArrivesCount<ARRIVES>,
        _in_page: crate::substrate::PageId<IN_ID, NUM_PAGES>,
        _weight_page: crate::substrate::PageId<WEIGHT_ID, NUM_PAGES>,
        _partial: crate::substrate::ScratchRegion<
            PARTIAL_OFF,
            PARTIAL_BYTES,
            SCRATCH_BYTES,
            crate::substrate::RmsNormScope,
        >,
        _consumer_phase: crate::substrate::MbarrierPhase<CONSUMER_PHASE>,
        _storer_phase: crate::substrate::MbarrierPhase<STORER_PHASE>,
        _layer: crate::nodes::LayerIndex<LAYER, NUM_LAYERS>,
        _hidden_dim: crate::substrate::HiddenDim<HIDDEN_DIM>,
        _num_tokens: crate::substrate::NumTokensConst<NUM_TOKENS>,
        // ActSlot/WeightAccessor use sentinel `u32::MAX` upper bound
        // until the builder's substrate budget gets `NUM_ACT_SLOTS` /
        // `NUM_WEIGHT_ACCESSORS` const generics propagated. The
        // typed-discipline (no bare u32) holds today; the proper
        // bound check is the next sprint.
        _in_act_slot: crate::substrate::ActSlotConst<IN_ACT_SLOT, { u32::MAX }>,
        _out_act_slot: crate::substrate::ActSlotConst<OUT_ACT_SLOT, { u32::MAX }>,
        _weight_accessor_idx: crate::substrate::WeightAccessorConst<
            WEIGHT_ACCESSOR_IDX,
            { u32::MAX },
        >,
        _consumer_bar_reduce: crate::substrate::BarSyncId<CONSUMER_BAR_REDUCE>,
        _consumer_bar_publish: crate::substrate::BarSyncId<CONSUMER_BAR_PUBLISH>,
        _bar_pair: crate::substrate::BarSyncPair<CONSUMER_BAR_REDUCE, CONSUMER_BAR_PUBLISH>,
        weight_path: String,
        eps: f32,
    ) -> &mut Self
    where
        // Sealed-witness type-check — ill-formed bar IDs (range or
        // aliased) fail at TYPE CHECK (E0277), not at any runtime
        // assert.
        crate::substrate::BarSyncId<CONSUMER_BAR_REDUCE>:
            crate::substrate::IsValidBarSyncId,
        crate::substrate::BarSyncId<CONSUMER_BAR_PUBLISH>:
            crate::substrate::IsValidBarSyncId,
        crate::substrate::BarSyncPair<CONSUMER_BAR_REDUCE, CONSUMER_BAR_PUBLISH>:
            crate::substrate::IsDistinctBarPair,
    {
        self.verify_arrives(ARRIVES, "push_rms_norm");
        // Cross-op alias check via runtime PagePool.
        let _ = self.pool.take(IN_ID);
        let _ = self.pool.take(WEIGHT_ID);
        let weight = WeightRef::new(weight_path);
        let eps = FiniteF32::new(eps);
        let node = RmsNorm::new::<
            IN_ID,
            WEIGHT_ID,
            PARTIAL_OFF,
            PARTIAL_BYTES,
            CONSUMER_PHASE,
            STORER_PHASE,
            LAYER,
            NUM_PAGES,
            NUM_LAYERS,
            SCRATCH_BYTES,
            ARRIVES,
            HIDDEN_DIM,
            NUM_TOKENS,
            IN_ACT_SLOT,
            OUT_ACT_SLOT,
            WEIGHT_ACCESSOR_IDX,
            CONSUMER_BAR_REDUCE,
            CONSUMER_BAR_PUBLISH,
        >(weight, eps);
        self.nodes.push(MegaNode::RmsNorm(node));
        self.pool.release(IN_ID);
        self.pool.release(WEIGHT_ID);
        self.arrives.bump();
        self
    }

    /// Push a typed FusedQkvRopeCache op onto the tape.
    ///
    /// Typed-args API: each typed primitive carries its substrate
    /// proof in its TYPE; the const generics on this method are
    /// inferred from the arg types via Rust type inference. No
    /// turbofish; no `/*FOO=*/` positional u32 comments at the call
    /// site.
    #[allow(clippy::too_many_arguments)]
    pub fn push_fused_qkv_rope_cache<
        const IN_ID: u32,
        const QKV_ID: u32,
        const COS_SIN_ID: u32,
        const Q_ID: u32,
        const K_ID: u32,
        const V_ID: u32,
        const Q_OFF: u32,
        const Q_BYTES: u32,
        const K_OFF: u32,
        const K_BYTES: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const ITERS: u32,
        const LAYER: u32,
        const NUM_LAYERS: u32,
        const ARRIVES: u32,
        const HIDDEN_DIM: u32,
        const HEAD_DIM: u32,
        const NUM_Q_HEADS: u32,
        const NUM_KV_HEADS: u32,
        const IN_ACT_SLOT: u32,
        const Q_OUT_ACT_SLOT: u32,
        const K_OUT_ACT_SLOT: u32,
        const V_OUT_ACT_SLOT: u32,
        const QKV_WEIGHT_ACCESSOR_IDX: u32,
        const ROTARY_ACCESSOR_IDX: u32,
    >(
        &mut self,
        _arrives: crate::substrate::ArrivesCount<ARRIVES>,
        _in_page: crate::substrate::PageId<IN_ID, NUM_PAGES>,
        _qkv_page: crate::substrate::PageId<QKV_ID, NUM_PAGES>,
        _cos_sin_page: crate::substrate::PageId<COS_SIN_ID, NUM_PAGES>,
        _q_out_page: crate::substrate::PageId<Q_ID, NUM_PAGES>,
        _k_out_page: crate::substrate::PageId<K_ID, NUM_PAGES>,
        _v_out_page: crate::substrate::PageId<V_ID, NUM_PAGES>,
        _q_rope: crate::substrate::ScratchRegion<
            Q_OFF,
            Q_BYTES,
            SCRATCH_BYTES,
            crate::substrate::RopeScope,
        >,
        _k_rope: crate::substrate::ScratchRegion<
            K_OFF,
            K_BYTES,
            SCRATCH_BYTES,
            crate::substrate::RopeScope,
        >,
        _consumer_phase: crate::substrate::MbarrierPhase<CONSUMER_PHASE>,
        _storer_phase: crate::substrate::MbarrierPhase<STORER_PHASE>,
        _iters: crate::substrate::IterCount<ITERS>,
        _layer: crate::nodes::LayerIndex<LAYER, NUM_LAYERS>,
        _hidden_dim: crate::substrate::HiddenDim<HIDDEN_DIM>,
        _head_dim: crate::substrate::HeadDim<HEAD_DIM>,
        _num_q_heads: crate::substrate::NumQHeads<NUM_Q_HEADS>,
        _num_kv_heads: crate::substrate::NumKvHeads<NUM_KV_HEADS>,
        _in_act_slot: crate::substrate::ActSlotConst<IN_ACT_SLOT, { u32::MAX }>,
        _q_out_act_slot: crate::substrate::ActSlotConst<Q_OUT_ACT_SLOT, { u32::MAX }>,
        _k_out_act_slot: crate::substrate::ActSlotConst<K_OUT_ACT_SLOT, { u32::MAX }>,
        _v_out_act_slot: crate::substrate::ActSlotConst<V_OUT_ACT_SLOT, { u32::MAX }>,
        _qkv_weight_accessor_idx: crate::substrate::WeightAccessorConst<
            QKV_WEIGHT_ACCESSOR_IDX,
            { u32::MAX },
        >,
        _rotary_accessor_idx: crate::substrate::WeightAccessorConst<
            ROTARY_ACCESSOR_IDX,
            { u32::MAX },
        >,
        qkv_weight_path: String,
        rotary_path: String,
        biased: bool,
        interleaved: bool,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_fused_qkv_rope_cache");
        let _ = self.pool.take(IN_ID);
        let _ = self.pool.take(QKV_ID);
        let _ = self.pool.take(COS_SIN_ID);
        let _ = self.pool.take(Q_ID);
        let _ = self.pool.take(K_ID);
        let _ = self.pool.take(V_ID);
        let qkv_weight = WeightRef::new(qkv_weight_path);
        let rotary = RotaryRef::new(rotary_path);
        let node = FusedQkvRopeCache::new::<
            IN_ID,
            QKV_ID,
            COS_SIN_ID,
            Q_ID,
            K_ID,
            V_ID,
            Q_OFF,
            Q_BYTES,
            K_OFF,
            K_BYTES,
            CONSUMER_PHASE,
            STORER_PHASE,
            ITERS,
            LAYER,
            NUM_PAGES,
            NUM_LAYERS,
            SCRATCH_BYTES,
            ARRIVES,
            HIDDEN_DIM,
            HEAD_DIM,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            IN_ACT_SLOT,
            Q_OUT_ACT_SLOT,
            K_OUT_ACT_SLOT,
            V_OUT_ACT_SLOT,
            QKV_WEIGHT_ACCESSOR_IDX,
            ROTARY_ACCESSOR_IDX,
        >(qkv_weight, rotary, biased, interleaved);
        self.nodes.push(MegaNode::FusedQkvRopeCache(node));
        self.pool.release(IN_ID);
        self.pool.release(QKV_ID);
        self.pool.release(COS_SIN_ID);
        self.pool.release(Q_ID);
        self.pool.release(K_ID);
        self.pool.release(V_ID);
        for _ in 0..ITERS {
            self.arrives.bump();
        }
        self
    }

    /// Push a typed `Add` (residual fold). Typed-args API.
    #[allow(clippy::too_many_arguments)]
    pub fn push_add<
        const DELTA_ID: u32,
        const RESIDUAL_ID: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const ARRIVES: u32,
        const HIDDEN_DIM: u32,
        const NUM_TOKENS: u32,
        const DELTA_ACT_SLOT: u32,
        const RESIDUAL_ACT_SLOT: u32,
    >(
        &mut self,
        _arrives: crate::substrate::ArrivesCount<ARRIVES>,
        _delta_page: crate::substrate::PageId<DELTA_ID, NUM_PAGES>,
        _residual_page: crate::substrate::PageId<RESIDUAL_ID, NUM_PAGES>,
        _consumer_phase: crate::substrate::MbarrierPhase<CONSUMER_PHASE>,
        _storer_phase: crate::substrate::MbarrierPhase<STORER_PHASE>,
        _hidden_dim: crate::substrate::HiddenDim<HIDDEN_DIM>,
        _num_tokens: crate::substrate::NumTokensConst<NUM_TOKENS>,
        _delta_act_slot: crate::substrate::ActSlotConst<DELTA_ACT_SLOT, { u32::MAX }>,
        _residual_act_slot: crate::substrate::ActSlotConst<RESIDUAL_ACT_SLOT, { u32::MAX }>,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_add");
        let _ = self.pool.take(DELTA_ID);
        let _ = self.pool.take(RESIDUAL_ID);
        let node = Add::new::<
            DELTA_ID,
            RESIDUAL_ID,
            CONSUMER_PHASE,
            STORER_PHASE,
            NUM_PAGES,
            ARRIVES,
            HIDDEN_DIM,
            NUM_TOKENS,
            DELTA_ACT_SLOT,
            RESIDUAL_ACT_SLOT,
        >();
        self.nodes.push(MegaNode::Add(node));
        self.pool.release(DELTA_ID);
        self.pool.release(RESIDUAL_ID);
        self.arrives.bump();
        self
    }

    /// Push a typed `FusedAddRmsNorm`. Typed-args API.
    #[allow(clippy::too_many_arguments)]
    pub fn push_fused_add_rms_norm<
        const DELTA_ID: u32,
        const RESIDUAL_ID: u32,
        const WEIGHT_ID: u32,
        const PARTIAL_OFF: u32,
        const PARTIAL_BYTES: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const LAYER: u32,
        const NUM_LAYERS: u32,
        const ARRIVES: u32,
        const HIDDEN_DIM: u32,
        const NUM_TOKENS: u32,
        const DELTA_ACT_SLOT: u32,
        const RESIDUAL_ACT_SLOT: u32,
        const WEIGHT_ACCESSOR_IDX: u32,
    >(
        &mut self,
        _arrives: crate::substrate::ArrivesCount<ARRIVES>,
        _delta_page: crate::substrate::PageId<DELTA_ID, NUM_PAGES>,
        _residual_page: crate::substrate::PageId<RESIDUAL_ID, NUM_PAGES>,
        _weight_page: crate::substrate::PageId<WEIGHT_ID, NUM_PAGES>,
        _partial: crate::substrate::ScratchRegion<
            PARTIAL_OFF,
            PARTIAL_BYTES,
            SCRATCH_BYTES,
            crate::substrate::RmsNormScope,
        >,
        _consumer_phase: crate::substrate::MbarrierPhase<CONSUMER_PHASE>,
        _storer_phase: crate::substrate::MbarrierPhase<STORER_PHASE>,
        _layer: crate::nodes::LayerIndex<LAYER, NUM_LAYERS>,
        _hidden_dim: crate::substrate::HiddenDim<HIDDEN_DIM>,
        _num_tokens: crate::substrate::NumTokensConst<NUM_TOKENS>,
        _delta_act_slot: crate::substrate::ActSlotConst<DELTA_ACT_SLOT, { u32::MAX }>,
        _residual_act_slot: crate::substrate::ActSlotConst<RESIDUAL_ACT_SLOT, { u32::MAX }>,
        _weight_accessor_idx: crate::substrate::WeightAccessorConst<
            WEIGHT_ACCESSOR_IDX,
            { u32::MAX },
        >,
        weight_path: String,
        eps: f32,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_fused_add_rms_norm");
        let _ = self.pool.take(DELTA_ID);
        let _ = self.pool.take(RESIDUAL_ID);
        let _ = self.pool.take(WEIGHT_ID);
        let weight = WeightRef::new(weight_path);
        let eps = FiniteF32::new(eps);
        let node = FusedAddRmsNorm::new::<
            DELTA_ID,
            RESIDUAL_ID,
            WEIGHT_ID,
            PARTIAL_OFF,
            PARTIAL_BYTES,
            CONSUMER_PHASE,
            STORER_PHASE,
            LAYER,
            NUM_PAGES,
            NUM_LAYERS,
            SCRATCH_BYTES,
            ARRIVES,
            HIDDEN_DIM,
            NUM_TOKENS,
            DELTA_ACT_SLOT,
            RESIDUAL_ACT_SLOT,
            WEIGHT_ACCESSOR_IDX,
        >(weight, eps);
        self.nodes.push(MegaNode::FusedAddRmsNorm(node));
        self.pool.release(DELTA_ID);
        self.pool.release(RESIDUAL_ID);
        self.pool.release(WEIGHT_ID);
        self.arrives.bump();
        self
    }

    /// Push a typed `FusedGateUp{Silu,Gelu}Mul`. Adds HIDDEN_DIM /
    /// INTERMEDIATE_DIM / NUM_TOKENS / IN_ACT_SLOT / OUT_ACT_SLOT /
    /// WEIGHT_ACCESSOR_IDX const generics for the kernel-AST emit
    /// (per `MEGA_IR_PLAN.md` §0/§4a).
    #[allow(clippy::too_many_arguments)]
    pub fn push_fused_gate_up_activate_mul<
        const IN_ID: u32,
        const WEIGHT_ID: u32,
        const OUT_ID: u32,
        const GATE_OFF: u32,
        const GATE_BYTES: u32,
        const UP_OFF: u32,
        const UP_BYTES: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const ITERS: u32,
        const LAYER: u32,
        const NUM_LAYERS: u32,
        const ARRIVES: u32,
        const HIDDEN_DIM: u32,
        const INTERMEDIATE_DIM: u32,
        const NUM_TOKENS: u32,
        const IN_ACT_SLOT: u32,
        const OUT_ACT_SLOT: u32,
        const WEIGHT_ACCESSOR_IDX: u32,
    >(
        &mut self,
        _arrives: crate::substrate::ArrivesCount<ARRIVES>,
        _in_page: crate::substrate::PageId<IN_ID, NUM_PAGES>,
        _weight_page: crate::substrate::PageId<WEIGHT_ID, NUM_PAGES>,
        _out_page: crate::substrate::PageId<OUT_ID, NUM_PAGES>,
        _gate: crate::substrate::ScratchRegion<
            GATE_OFF, GATE_BYTES, SCRATCH_BYTES, crate::substrate::MlpScope,
        >,
        _up: crate::substrate::ScratchRegion<
            UP_OFF, UP_BYTES, SCRATCH_BYTES, crate::substrate::MlpScope,
        >,
        _consumer_phase: crate::substrate::MbarrierPhase<CONSUMER_PHASE>,
        _storer_phase: crate::substrate::MbarrierPhase<STORER_PHASE>,
        _iters: crate::substrate::IterCount<ITERS>,
        _layer: crate::nodes::LayerIndex<LAYER, NUM_LAYERS>,
        _hidden_dim: crate::substrate::HiddenDim<HIDDEN_DIM>,
        _intermediate_dim: crate::substrate::IntermediateDim<INTERMEDIATE_DIM>,
        _num_tokens: crate::substrate::NumTokensConst<NUM_TOKENS>,
        _in_act_slot: crate::substrate::ActSlotConst<IN_ACT_SLOT, { u32::MAX }>,
        _out_act_slot: crate::substrate::ActSlotConst<OUT_ACT_SLOT, { u32::MAX }>,
        _weight_accessor_idx: crate::substrate::WeightAccessorConst<
            WEIGHT_ACCESSOR_IDX, { u32::MAX },
        >,
        weight_path: String,
        activation: GateUpActivation,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_fused_gate_up_activate_mul");
        let _ = self.pool.take(IN_ID);
        let _ = self.pool.take(WEIGHT_ID);
        let _ = self.pool.take(OUT_ID);
        let weight = WeightRef::new(weight_path);
        let node = FusedGateUpActivateMul::new::<
            IN_ID,
            WEIGHT_ID,
            OUT_ID,
            GATE_OFF,
            GATE_BYTES,
            UP_OFF,
            UP_BYTES,
            CONSUMER_PHASE,
            STORER_PHASE,
            ITERS,
            LAYER,
            NUM_PAGES,
            NUM_LAYERS,
            SCRATCH_BYTES,
            ARRIVES,
            HIDDEN_DIM,
            INTERMEDIATE_DIM,
            NUM_TOKENS,
            IN_ACT_SLOT,
            OUT_ACT_SLOT,
            WEIGHT_ACCESSOR_IDX,
        >(weight, activation);
        self.nodes.push(MegaNode::FusedGateUpActivateMul(node));
        self.pool.release(IN_ID);
        self.pool.release(WEIGHT_ID);
        self.pool.release(OUT_ID);
        for _ in 0..ITERS {
            self.arrives.bump();
        }
        self
    }

    /// Push a typed `Embed`. Adds `HIDDEN_DIM` / `NUM_TOKENS` /
    /// `VOCAB_SIZE` / `OUT_ACT_SLOT` / `WEIGHT_ACCESSOR_IDX` const
    /// generics — every kernel template + slot index field per §4a.
    #[allow(clippy::too_many_arguments)]
    pub fn push_embed<
        const OUT_ID: u32,
        const WEIGHT_ID: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const ARRIVES: u32,
        const HIDDEN_DIM: u32,
        const NUM_TOKENS: u32,
        const VOCAB_SIZE: u32,
        const OUT_ACT_SLOT: u32,
        const WEIGHT_ACCESSOR_IDX: u32,
    >(
        &mut self,
        _arrives: crate::substrate::ArrivesCount<ARRIVES>,
        _out_page: crate::substrate::PageId<OUT_ID, NUM_PAGES>,
        _weight_page: crate::substrate::PageId<WEIGHT_ID, NUM_PAGES>,
        _consumer_phase: crate::substrate::MbarrierPhase<CONSUMER_PHASE>,
        _storer_phase: crate::substrate::MbarrierPhase<STORER_PHASE>,
        _hidden_dim: crate::substrate::HiddenDim<HIDDEN_DIM>,
        _num_tokens: crate::substrate::NumTokensConst<NUM_TOKENS>,
        _vocab_size: crate::substrate::VocabSize<VOCAB_SIZE>,
        _out_act_slot: crate::substrate::ActSlotConst<OUT_ACT_SLOT, { u32::MAX }>,
        _weight_accessor_idx: crate::substrate::WeightAccessorConst<
            WEIGHT_ACCESSOR_IDX, { u32::MAX },
        >,
        embed_weight_path: String,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_embed");
        let _ = self.pool.take(OUT_ID);
        let _ = self.pool.take(WEIGHT_ID);
        let weight = WeightRef::new(embed_weight_path);
        let node = Embed::new::<
            OUT_ID,
            WEIGHT_ID,
            CONSUMER_PHASE,
            STORER_PHASE,
            NUM_PAGES,
            ARRIVES,
            HIDDEN_DIM,
            NUM_TOKENS,
            VOCAB_SIZE,
            OUT_ACT_SLOT,
            WEIGHT_ACCESSOR_IDX,
        >(weight);
        self.nodes.push(MegaNode::Embed(node));
        self.pool.release(OUT_ID);
        self.pool.release(WEIGHT_ID);
        self.arrives.bump();
        self
    }

    /// Push a typed `ScalarMul`. Adds `HIDDEN_DIM` / `NUM_TOKENS` /
    /// `IN_ACT_SLOT` / `OUT_ACT_SLOT` for kernel-AST emit (§4a).
    #[allow(clippy::too_many_arguments)]
    pub fn push_scalar_mul<
        const IN_ID: u32,
        const OUT_ID: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const ARRIVES: u32,
        const HIDDEN_DIM: u32,
        const NUM_TOKENS: u32,
        const IN_ACT_SLOT: u32,
        const OUT_ACT_SLOT: u32,
    >(
        &mut self,
        _arrives: crate::substrate::ArrivesCount<ARRIVES>,
        _in_page: crate::substrate::PageId<IN_ID, NUM_PAGES>,
        _out_page: crate::substrate::PageId<OUT_ID, NUM_PAGES>,
        _consumer_phase: crate::substrate::MbarrierPhase<CONSUMER_PHASE>,
        _storer_phase: crate::substrate::MbarrierPhase<STORER_PHASE>,
        _hidden_dim: crate::substrate::HiddenDim<HIDDEN_DIM>,
        _num_tokens: crate::substrate::NumTokensConst<NUM_TOKENS>,
        _in_act_slot: crate::substrate::ActSlotConst<IN_ACT_SLOT, { u32::MAX }>,
        _out_act_slot: crate::substrate::ActSlotConst<OUT_ACT_SLOT, { u32::MAX }>,
        scale: f32,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_scalar_mul");
        let _ = self.pool.take(IN_ID);
        let _ = self.pool.take(OUT_ID);
        let scale = FiniteF32::new(scale);
        let node = ScalarMul::new::<
            IN_ID,
            OUT_ID,
            CONSUMER_PHASE,
            STORER_PHASE,
            NUM_PAGES,
            ARRIVES,
            HIDDEN_DIM,
            NUM_TOKENS,
            IN_ACT_SLOT,
            OUT_ACT_SLOT,
        >(scale);
        self.nodes.push(MegaNode::ScalarMul(node));
        self.pool.release(IN_ID);
        self.pool.release(OUT_ID);
        self.arrives.bump();
        self
    }

    /// Push a typed `TanhSoftCap`. Adds `HIDDEN_DIM` / `NUM_TOKENS` /
    /// `IN_ACT_SLOT` / `OUT_ACT_SLOT` const generics + runtime `cap`
    /// (Gemma2 final-logit softcap; 0.0 = identity for non-Gemma2).
    #[allow(clippy::too_many_arguments)]
    pub fn push_tanh_soft_cap<
        const IN_ID: u32,
        const OUT_ID: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const ARRIVES: u32,
        const HIDDEN_DIM: u32,
        const NUM_TOKENS: u32,
        const IN_ACT_SLOT: u32,
        const OUT_ACT_SLOT: u32,
    >(
        &mut self,
        _arrives: crate::substrate::ArrivesCount<ARRIVES>,
        _in_page: crate::substrate::PageId<IN_ID, NUM_PAGES>,
        _out_page: crate::substrate::PageId<OUT_ID, NUM_PAGES>,
        _consumer_phase: crate::substrate::MbarrierPhase<CONSUMER_PHASE>,
        _storer_phase: crate::substrate::MbarrierPhase<STORER_PHASE>,
        _hidden_dim: crate::substrate::HiddenDim<HIDDEN_DIM>,
        _num_tokens: crate::substrate::NumTokensConst<NUM_TOKENS>,
        _in_act_slot: crate::substrate::ActSlotConst<IN_ACT_SLOT, { u32::MAX }>,
        _out_act_slot: crate::substrate::ActSlotConst<OUT_ACT_SLOT, { u32::MAX }>,
        cap: f32,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_tanh_soft_cap");
        let _ = self.pool.take(IN_ID);
        let _ = self.pool.take(OUT_ID);
        let cap = FiniteF32::new(cap);
        let node = TanhSoftCap::new::<
            IN_ID,
            OUT_ID,
            CONSUMER_PHASE,
            STORER_PHASE,
            NUM_PAGES,
            ARRIVES,
            HIDDEN_DIM,
            NUM_TOKENS,
            IN_ACT_SLOT,
            OUT_ACT_SLOT,
        >(cap);
        self.nodes.push(MegaNode::TanhSoftCap(node));
        self.pool.release(IN_ID);
        self.pool.release(OUT_ID);
        self.arrives.bump();
        self
    }

    /// Push a typed `ScalarOffsetRmsNorm`. Adds `HIDDEN_DIM` /
    /// `NUM_TOKENS` / `IN_ACT_SLOT` / `OUT_ACT_SLOT` /
    /// `WEIGHT_ACCESSOR_IDX` + runtime `eps` (§4a).
    #[allow(clippy::too_many_arguments)]
    pub fn push_scalar_offset_rms_norm<
        const IN_ID: u32,
        const WEIGHT_ID: u32,
        const PARTIAL_OFF: u32,
        const PARTIAL_BYTES: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const LAYER: u32,
        const NUM_LAYERS: u32,
        const ARRIVES: u32,
        const HIDDEN_DIM: u32,
        const NUM_TOKENS: u32,
        const IN_ACT_SLOT: u32,
        const OUT_ACT_SLOT: u32,
        const WEIGHT_ACCESSOR_IDX: u32,
    >(
        &mut self,
        _arrives: crate::substrate::ArrivesCount<ARRIVES>,
        _in_page: crate::substrate::PageId<IN_ID, NUM_PAGES>,
        _weight_page: crate::substrate::PageId<WEIGHT_ID, NUM_PAGES>,
        _partial: crate::substrate::ScratchRegion<
            PARTIAL_OFF, PARTIAL_BYTES, SCRATCH_BYTES, crate::substrate::RmsNormScope,
        >,
        _consumer_phase: crate::substrate::MbarrierPhase<CONSUMER_PHASE>,
        _storer_phase: crate::substrate::MbarrierPhase<STORER_PHASE>,
        _layer: crate::nodes::LayerIndex<LAYER, NUM_LAYERS>,
        _hidden_dim: crate::substrate::HiddenDim<HIDDEN_DIM>,
        _num_tokens: crate::substrate::NumTokensConst<NUM_TOKENS>,
        _in_act_slot: crate::substrate::ActSlotConst<IN_ACT_SLOT, { u32::MAX }>,
        _out_act_slot: crate::substrate::ActSlotConst<OUT_ACT_SLOT, { u32::MAX }>,
        _weight_accessor_idx: crate::substrate::WeightAccessorConst<
            WEIGHT_ACCESSOR_IDX, { u32::MAX },
        >,
        weight_path: String,
        offset: f32,
        eps: f32,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_scalar_offset_rms_norm");
        let _ = self.pool.take(IN_ID);
        let _ = self.pool.take(WEIGHT_ID);
        let weight = WeightRef::new(weight_path);
        let offset = FiniteF32::new(offset);
        let eps = FiniteF32::new(eps);
        let node = ScalarOffsetRmsNorm::new::<
            IN_ID,
            WEIGHT_ID,
            PARTIAL_OFF,
            PARTIAL_BYTES,
            CONSUMER_PHASE,
            STORER_PHASE,
            LAYER,
            NUM_PAGES,
            NUM_LAYERS,
            SCRATCH_BYTES,
            ARRIVES,
            HIDDEN_DIM,
            NUM_TOKENS,
            IN_ACT_SLOT,
            OUT_ACT_SLOT,
            WEIGHT_ACCESSOR_IDX,
        >(weight, offset, eps);
        self.nodes.push(MegaNode::ScalarOffsetRmsNorm(node));
        self.pool.release(IN_ID);
        self.pool.release(WEIGHT_ID);
        self.arrives.bump();
        self
    }

    /// Push a typed `Gemm`. Adds `M` (NUM_TOKENS), `IN_ACT_SLOT`,
    /// `OUT_ACT_SLOT`, `WEIGHT_ACCESSOR_IDX` for §0/§4a kernel-AST
    /// emit (`<Config, K, N, M>` template + `act_ptrs[]` /
    /// `weight_ptrs[]` indices).
    #[allow(clippy::too_many_arguments)]
    pub fn push_gemm<
        const IN_ID: u32,
        const WEIGHT_ID: u32,
        const OUT_ID: u32,
        const B_TILE_OFF: u32,
        const B_TILE_BYTES: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const ITERS: u32,
        const LAYER: u32,
        const N: u32,
        const K: u32,
        const NUM_LAYERS: u32,
        const ARRIVES: u32,
        const M: u32,
        const IN_ACT_SLOT: u32,
        const OUT_ACT_SLOT: u32,
        const WEIGHT_ACCESSOR_IDX: u32,
    >(
        &mut self,
        weight_path: String,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_gemm");
        let _ = self.pool.take(IN_ID);
        let _ = self.pool.take(WEIGHT_ID);
        let _ = self.pool.take(OUT_ID);
        let weight = WeightRef::new(weight_path);
        let node = Gemm::new::<
            IN_ID,
            WEIGHT_ID,
            OUT_ID,
            B_TILE_OFF,
            B_TILE_BYTES,
            CONSUMER_PHASE,
            STORER_PHASE,
            ITERS,
            LAYER,
            N,
            K,
            NUM_PAGES,
            NUM_LAYERS,
            SCRATCH_BYTES,
            ARRIVES,
            M,
            IN_ACT_SLOT,
            OUT_ACT_SLOT,
            WEIGHT_ACCESSOR_IDX,
        >(weight);
        self.nodes.push(MegaNode::Gemm(node));
        self.pool.release(IN_ID);
        self.pool.release(WEIGHT_ID);
        self.pool.release(OUT_ID);
        for _ in 0..ITERS {
            self.arrives.bump();
        }
        self
    }

    /// Push a typed `FusedCublasGemmAdd` onto the tape. Substrate
    /// shape: 3 pages (in / weight / residual=output), `GemmScope`
    /// B-tile, multi-iter. The residual page is read AND written
    /// (in-place residual fold after the gemm).
    ///
    /// AST const generics (§0/§4a): NUM_TOKENS, K_OFFSET, K_FULL +
    /// IN_ACT_SLOT, RESIDUAL_ACT_SLOT, WEIGHT_ACCESSOR_IDX. K_OFFSET
    /// / K_FULL drive the down_proj 4-chunk split (TkGemmAdd path);
    /// the un-split case has K_OFFSET = 0, K_FULL = K.
    #[allow(clippy::too_many_arguments)]
    pub fn push_fused_cublas_gemm_add<
        const IN_ID: u32,
        const WEIGHT_ID: u32,
        const RESIDUAL_ID: u32,
        const B_TILE_OFF: u32,
        const B_TILE_BYTES: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const ITERS: u32,
        const LAYER: u32,
        const N: u32,
        const K: u32,
        const NUM_LAYERS: u32,
        const ARRIVES: u32,
        const NUM_TOKENS: u32,
        const K_OFFSET: u32,
        const K_FULL: u32,
        const IN_ACT_SLOT: u32,
        const RESIDUAL_ACT_SLOT: u32,
        const WEIGHT_ACCESSOR_IDX: u32,
    >(
        &mut self,
        weight_path: String,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_fused_cublas_gemm_add");
        let _ = self.pool.take(IN_ID);
        let _ = self.pool.take(WEIGHT_ID);
        let _ = self.pool.take(RESIDUAL_ID);
        let weight = WeightRef::new(weight_path);
        let node = FusedCublasGemmAdd::new::<
            IN_ID,
            WEIGHT_ID,
            RESIDUAL_ID,
            B_TILE_OFF,
            B_TILE_BYTES,
            CONSUMER_PHASE,
            STORER_PHASE,
            ITERS,
            LAYER,
            N,
            K,
            NUM_PAGES,
            NUM_LAYERS,
            SCRATCH_BYTES,
            ARRIVES,
            NUM_TOKENS,
            K_OFFSET,
            K_FULL,
            IN_ACT_SLOT,
            RESIDUAL_ACT_SLOT,
            WEIGHT_ACCESSOR_IDX,
        >(weight);
        self.nodes.push(MegaNode::FusedCublasGemmAdd(node));
        self.pool.release(IN_ID);
        self.pool.release(WEIGHT_ID);
        self.pool.release(RESIDUAL_ID);
        for _ in 0..ITERS {
            self.arrives.bump();
        }
        self
    }

    /// Push a typed `CutlassFusedNormGemm` for the no-delta flavors
    /// (RmsNorm, MeanSubRmsNorm). AST-shape const generics:
    /// NUM_TOKENS, IN_ACT_SLOT, OUT_ACT_SLOT,
    /// NORM_WEIGHT_ACCESSOR_IDX, LINEAR_WEIGHT_ACCESSOR_IDX. Runtime
    /// `eps` for the kernel's consumer eps arg.
    #[allow(clippy::too_many_arguments)]
    pub fn push_cutlass_fused_norm_gemm_no_delta<
        const IN_ID: u32,
        const NORM_W_ID: u32,
        const LIN_W_ID: u32,
        const OUT_ID: u32,
        const PARTIAL_OFF: u32,
        const PARTIAL_BYTES: u32,
        const B_TILE_OFF: u32,
        const B_TILE_BYTES: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const ITERS: u32,
        const LAYER: u32,
        const N: u32,
        const K: u32,
        const NUM_LAYERS: u32,
        const ARRIVES: u32,
        const NUM_TOKENS: u32,
        const IN_ACT_SLOT: u32,
        const OUT_ACT_SLOT: u32,
        const NORM_WEIGHT_ACCESSOR_IDX: u32,
        const LINEAR_WEIGHT_ACCESSOR_IDX: u32,
    >(
        &mut self,
        norm_weight_path: String,
        linear_weight_path: String,
        norm_kind: LmHeadNormKind,
        eps: f32,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_cutlass_fused_norm_gemm_no_delta");
        let _ = self.pool.take(IN_ID);
        let _ = self.pool.take(NORM_W_ID);
        let _ = self.pool.take(LIN_W_ID);
        let _ = self.pool.take(OUT_ID);
        let norm_weight = WeightRef::new(norm_weight_path);
        let linear_weight = WeightRef::new(linear_weight_path);
        let eps = FiniteF32::new(eps);
        let node = CutlassFusedNormGemm::new_no_delta::<
            IN_ID,
            NORM_W_ID,
            LIN_W_ID,
            OUT_ID,
            PARTIAL_OFF,
            PARTIAL_BYTES,
            B_TILE_OFF,
            B_TILE_BYTES,
            CONSUMER_PHASE,
            STORER_PHASE,
            ITERS,
            LAYER,
            N,
            K,
            NUM_PAGES,
            NUM_LAYERS,
            SCRATCH_BYTES,
            ARRIVES,
            NUM_TOKENS,
            IN_ACT_SLOT,
            OUT_ACT_SLOT,
            NORM_WEIGHT_ACCESSOR_IDX,
            LINEAR_WEIGHT_ACCESSOR_IDX,
        >(norm_weight, linear_weight, norm_kind, eps);
        self.nodes.push(MegaNode::CutlassFusedNormGemm(node));
        self.pool.release(IN_ID);
        self.pool.release(NORM_W_ID);
        self.pool.release(LIN_W_ID);
        self.pool.release(OUT_ID);
        for _ in 0..ITERS {
            self.arrives.bump();
        }
        self
    }

    /// Push a typed `CutlassFusedNormGemm` for the residual-fold
    /// flavors (AddRmsNorm, AddScalarOffsetRmsNorm). AST-shape:
    /// NUM_TOKENS, IN_ACT_SLOT, DELTA_ACT_SLOT, OUT_ACT_SLOT,
    /// NORM_WEIGHT_ACCESSOR_IDX, LINEAR_WEIGHT_ACCESSOR_IDX. Runtime
    /// `eps`; offset present only for AddScalarOffsetRmsNorm.
    #[allow(clippy::too_many_arguments)]
    pub fn push_cutlass_fused_norm_gemm_with_delta<
        const IN_ID: u32,
        const DELTA_ID: u32,
        const NORM_W_ID: u32,
        const LIN_W_ID: u32,
        const OUT_ID: u32,
        const PARTIAL_OFF: u32,
        const PARTIAL_BYTES: u32,
        const B_TILE_OFF: u32,
        const B_TILE_BYTES: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const ITERS: u32,
        const LAYER: u32,
        const N: u32,
        const K: u32,
        const NUM_LAYERS: u32,
        const ARRIVES: u32,
        const NUM_TOKENS: u32,
        const IN_ACT_SLOT: u32,
        const DELTA_ACT_SLOT: u32,
        const OUT_ACT_SLOT: u32,
        const NORM_WEIGHT_ACCESSOR_IDX: u32,
        const LINEAR_WEIGHT_ACCESSOR_IDX: u32,
    >(
        &mut self,
        norm_weight_path: String,
        linear_weight_path: String,
        norm_kind: LmHeadNormKind,
        offset: Option<f32>,
        eps: f32,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_cutlass_fused_norm_gemm_with_delta");
        let _ = self.pool.take(IN_ID);
        let _ = self.pool.take(DELTA_ID);
        let _ = self.pool.take(NORM_W_ID);
        let _ = self.pool.take(LIN_W_ID);
        let _ = self.pool.take(OUT_ID);
        let norm_weight = WeightRef::new(norm_weight_path);
        let linear_weight = WeightRef::new(linear_weight_path);
        let offset = offset.map(FiniteF32::new);
        let eps = FiniteF32::new(eps);
        let node = CutlassFusedNormGemm::new_with_delta::<
            IN_ID,
            DELTA_ID,
            NORM_W_ID,
            LIN_W_ID,
            OUT_ID,
            PARTIAL_OFF,
            PARTIAL_BYTES,
            B_TILE_OFF,
            B_TILE_BYTES,
            CONSUMER_PHASE,
            STORER_PHASE,
            ITERS,
            LAYER,
            N,
            K,
            NUM_PAGES,
            NUM_LAYERS,
            SCRATCH_BYTES,
            ARRIVES,
            NUM_TOKENS,
            IN_ACT_SLOT,
            DELTA_ACT_SLOT,
            OUT_ACT_SLOT,
            NORM_WEIGHT_ACCESSOR_IDX,
            LINEAR_WEIGHT_ACCESSOR_IDX,
        >(norm_weight, linear_weight, norm_kind, offset, eps);
        self.nodes.push(MegaNode::CutlassFusedNormGemm(node));
        self.pool.release(IN_ID);
        self.pool.release(DELTA_ID);
        self.pool.release(NORM_W_ID);
        self.pool.release(LIN_W_ID);
        self.pool.release(OUT_ID);
        for _ in 0..ITERS {
            self.arrives.bump();
        }
        self
    }

    /// Push a typed `AttentionViaCacheNode`. AST-shape const
    /// generics: HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS, BLOCK_SIZE,
    /// NUM_TOKENS, MAX_SK + Q_IN_ACT_SLOT, ATTN_OUT_ACT_SLOT.
    /// Runtime: attn_scale, attn_softcap (zero softcap = identity).
    #[allow(clippy::too_many_arguments)]
    pub fn push_attention_via_cache<
        const Q_IN_ID: u32,
        const ATTN_OUT_ID: u32,
        const SCORE_OFF: u32,
        const SCORE_BYTES: u32,
        const PV_OFF: u32,
        const PV_BYTES: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const ITERS: u32,
        const LAYER: u32,
        const NUM_LAYERS: u32,
        const ARRIVES: u32,
        const HEAD_DIM: u32,
        const NUM_Q_HEADS: u32,
        const NUM_KV_HEADS: u32,
        const BLOCK_SIZE: u32,
        const NUM_TOKENS: u32,
        const MAX_SK: u32,
        const Q_IN_ACT_SLOT: u32,
        const ATTN_OUT_ACT_SLOT: u32,
    >(
        &mut self,
        kind: AttentionKind,
        interleaved: bool,
        attn_scale: f32,
        attn_softcap: f32,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_attention_via_cache");
        let _ = self.pool.take(Q_IN_ID);
        let _ = self.pool.take(ATTN_OUT_ID);
        let attn_scale = FiniteF32::new(attn_scale);
        let attn_softcap = FiniteF32::new(attn_softcap);
        let node = AttentionViaCacheNode::new::<
            Q_IN_ID,
            ATTN_OUT_ID,
            SCORE_OFF,
            SCORE_BYTES,
            PV_OFF,
            PV_BYTES,
            CONSUMER_PHASE,
            STORER_PHASE,
            ITERS,
            LAYER,
            NUM_PAGES,
            NUM_LAYERS,
            SCRATCH_BYTES,
            ARRIVES,
            HEAD_DIM,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            BLOCK_SIZE,
            NUM_TOKENS,
            MAX_SK,
            Q_IN_ACT_SLOT,
            ATTN_OUT_ACT_SLOT,
        >(kind, interleaved, attn_scale, attn_softcap);
        self.nodes.push(MegaNode::AttentionViaCache(node));
        self.pool.release(Q_IN_ID);
        self.pool.release(ATTN_OUT_ID);
        for _ in 0..ITERS {
            self.arrives.bump();
        }
        self
    }

    /// Push a typed `SpliceMmEmbeds`. Adds HIDDEN_DIM, NUM_TOKENS,
    /// TARGET_ACT_SLOT for §0/§4a kernel-AST D2D-copy emit.
    #[allow(clippy::too_many_arguments)]
    pub fn push_splice_mm_embeds<
        const SLOT_ID: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const ARRIVES: u32,
        const HIDDEN_DIM: u32,
        const NUM_TOKENS: u32,
        const TARGET_ACT_SLOT: u32,
    >(
        &mut self,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_splice_mm_embeds");
        let _ = self.pool.take(SLOT_ID);
        let node = SpliceMmEmbeds::new::<
            SLOT_ID,
            CONSUMER_PHASE,
            STORER_PHASE,
            NUM_PAGES,
            ARRIVES,
            HIDDEN_DIM,
            NUM_TOKENS,
            TARGET_ACT_SLOT,
        >();
        self.nodes.push(MegaNode::SpliceMmEmbeds(node));
        self.pool.release(SLOT_ID);
        self.arrives.bump();
        self
    }

    /// Push a typed `BarrierSignal`.
    pub fn push_barrier_signal<const IDX: u32>(&mut self) -> &mut Self {
        let node = BarrierSignal::new::<IDX, NUM_EDGES>();
        self.nodes.push(MegaNode::BarrierSignal(node));
        self
    }

    /// Push a typed `BarrierWait`.
    pub fn push_barrier_wait<const IDX: u32, const COUNT: u32>(&mut self) -> &mut Self {
        let node = BarrierWait::new::<IDX, COUNT, NUM_EDGES>();
        self.nodes.push(MegaNode::BarrierWait(node));
        self
    }

    /// Consume the builder and return the typed tape.
    pub fn finish(self) -> MegaTape {
        MegaTape::__build_from_nodes(self.nodes)
    }
}

#[derive(Debug)]
pub enum LowerError {
    /// Reserved for variants that haven't been substrate-lifted
    /// yet.
    NotYetLifted {
        op: &'static str,
    },
    WrongWeightArity {
        op: &'static str,
        expected: u32,
        got: u32,
    },
    SubstrateBudgetTooSmall {
        need: u32,
        have: u32,
    },
    MissingSlidingWindow,
}

// `OpInput` and the runtime-walking `lower(&[OpInput])` entry point
// were retired with the const-generic refactor — passing runtime
// u32 values as const-generic args isn't legal Rust. Per
// `MEGA_IR_PLAN.md` §C, the proc-macro builds the typed `MegaTape`
// by EMITTING literal `MegaTapeBuilder::push_*::<...>(weight_path)`
// calls at expansion time; user-build monomorphization fires the
// `const {}` blocks. No runtime-dispatch entry point exists, and
// `ferrite-mega-ir` no longer depends on `ferrite-forward`.

// ============================================================
// Tests — exercise the const-generic builder at every variant.
// Compile-fail tests for the substrate proofs live in
// `tests/compile-fail/*.rs` (trybuild).
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    type Builder6 = MegaTapeBuilder<6, 8, 32_768, 8_192, 0>;
    type Builder8 = MegaTapeBuilder<8, 8, 32_768, 8_192, 0>;
    type BuilderD = MegaTapeBuilder<8, 8, 32_768, 32_768, 4>;

    #[test]
    fn lowers_well_formed_rms_norm() {
        let mut b = Builder6::new();
        // Each typed-primitive arg is self-documenting via its type
        // — no `/*FOO=*/` positional u32 comments needed. Substrate
        // proofs are discharged at each typed primitive's
        // construction (e.g. `PageId::<0, 6>` proves `0 < 6`;
        // `BarSyncId<1>` proves bar.sync ID is in 1..=15;
        // `BarSyncPair<1, 2>` proves the two bars are distinct).
        use crate::nodes::LayerIndex;
        use crate::substrate::{
            ActSlotConst, ArrivesCount, BarSyncId, BarSyncPair, HiddenDim, MbarrierPhase,
            NumTokensConst, PageId, RmsNormScope, ScratchRegion, WeightAccessorConst,
        };
        b.push_rms_norm(
            ArrivesCount::<0>::new(),
            PageId::<0, 6>::new(),
            PageId::<1, 6>::new(),
            ScratchRegion::<0, 32, 8192, RmsNormScope>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            LayerIndex::<0, 16>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<8>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<1, { u32::MAX }>::new(),
            WeightAccessorConst::<0, { u32::MAX }>::new(),
            BarSyncId::<1>::new(),
            BarSyncId::<2>::new(),
            BarSyncPair::<1, 2>::new(),
            "W::norm".to_string(),
            1.0e-5_f32,
        );
        let tape = b.finish();
        assert_eq!(tape.nodes().len(), 1);
        let MegaNode::RmsNorm(n) = &tape.nodes()[0] else {
            panic!("expected RmsNorm");
        };
        assert_eq!(n.in_page().raw(), 0);
        assert_eq!(n.weight_page().raw(), 1);
        assert_eq!(n.partial_offset().raw(), 0);
        assert_eq!(n.partial_bytes().raw(), 32);
        assert_eq!(n.consumer_phase().raw(), 0);
        assert_eq!(n.storer_phase().raw(), 1);
        assert_eq!(n.layer().raw(), 0);
        assert_eq!(n.hidden_dim().raw(), 2048);
        assert_eq!(n.num_tokens().raw(), 8);
        assert_eq!(n.in_act_slot().raw(), 0);
        assert_eq!(n.out_act_slot().raw(), 1);
        assert_eq!(n.weight_accessor_idx().raw(), 0);
        assert_eq!(n.consumer_bar_reduce().raw(), 1);
        assert_eq!(n.consumer_bar_publish().raw(), 2);
        assert!((n.eps().raw() - 1.0e-5_f32).abs() < 1e-9);
        assert_eq!(n.weight.path(), "W::norm");
    }

    #[test]
    fn lowers_two_rms_norms_with_phase_advance() {
        use crate::nodes::LayerIndex;
        use crate::substrate::{
            ActSlotConst, ArrivesCount, BarSyncId, BarSyncPair, HiddenDim, MbarrierPhase,
            NumTokensConst, PageId, RmsNormScope, ScratchRegion, WeightAccessorConst,
        };
        let mut b = Builder6::new();
        b.push_rms_norm(
            ArrivesCount::<0>::new(),
            PageId::<0, 6>::new(),
            PageId::<1, 6>::new(),
            ScratchRegion::<0, 32, 8192, RmsNormScope>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            LayerIndex::<0, 16>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<8>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<1, { u32::MAX }>::new(),
            WeightAccessorConst::<0, { u32::MAX }>::new(),
            BarSyncId::<1>::new(),
            BarSyncId::<2>::new(),
            BarSyncPair::<1, 2>::new(),
            "W::n0".to_string(),
            1.0e-5_f32,
        );
        // After first op, ARRIVES = 1; CONSUMER_PHASE = 1, STORER_PHASE = 0.
        // OUT_ACT_SLOT = 2 to avoid runtime PagePool aliasing on slot 1.
        b.push_rms_norm(
            ArrivesCount::<1>::new(),
            PageId::<0, 6>::new(),
            PageId::<1, 6>::new(),
            ScratchRegion::<0, 32, 8192, RmsNormScope>::new(),
            MbarrierPhase::<1>::new(),
            MbarrierPhase::<0>::new(),
            LayerIndex::<1, 16>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<8>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<2, { u32::MAX }>::new(),
            WeightAccessorConst::<1, { u32::MAX }>::new(),
            BarSyncId::<1>::new(),
            BarSyncId::<2>::new(),
            BarSyncPair::<1, 2>::new(),
            "W::n1".to_string(),
            1.0e-5_f32,
        );
        let tape = b.finish();
        let MegaNode::RmsNorm(n0) = &tape.nodes()[0] else {
            panic!();
        };
        let MegaNode::RmsNorm(n1) = &tape.nodes()[1] else {
            panic!();
        };
        assert_eq!(n0.consumer_phase().raw(), 0);
        assert_eq!(n0.storer_phase().raw(), 1);
        assert_eq!(n1.consumer_phase().raw(), 1);
        assert_eq!(n1.storer_phase().raw(), 0);
    }

    #[test]
    #[should_panic(expected = "PagePool::take: page id 1 already in use")]
    fn rejects_cross_op_alias_runtime() {
        // PagePool runtime cross-op alias guard. The push_* methods
        // release pages before returning, so a true cross-op alias
        // can't be triggered through the high-level API alone — we
        // drive PagePool directly here to exercise the guard.
        let mut pool = crate::substrate::PagePool::new(6);
        let _ = pool.take(1);
        let _ = pool.take(1); // cross-op alias panic
    }

    #[test]
    fn lowers_fused_qkv_rope_cache() {
        use crate::nodes::LayerIndex;
        use crate::substrate::{
            ActSlotConst, ArrivesCount, HeadDim, HiddenDim, IterCount, MbarrierPhase,
            NumKvHeads, NumQHeads, PageId, RopeScope, ScratchRegion, WeightAccessorConst,
        };
        let mut b = Builder8::new();
        // 6 distinct page ids, q_rope (0,2048) + k_rope (2048,2048) —
        // disjoint, within 8192 SCRATCH_BYTES. ITERS=4, LAYER=0,
        // NUM_LAYERS=16, ARRIVES=0, CONSUMER_PHASE=0, STORER_PHASE=1.
        b.push_fused_qkv_rope_cache(
            ArrivesCount::<0>::new(),
            PageId::<0, 8>::new(),
            PageId::<1, 8>::new(),
            PageId::<2, 8>::new(),
            PageId::<3, 8>::new(),
            PageId::<4, 8>::new(),
            PageId::<5, 8>::new(),
            ScratchRegion::<0, 2048, 8192, RopeScope>::new(),
            ScratchRegion::<2048, 2048, 8192, RopeScope>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            IterCount::<4>::new(),
            LayerIndex::<0, 16>::new(),
            HiddenDim::<2048>::new(),
            HeadDim::<64>::new(),
            NumQHeads::<32>::new(),
            NumKvHeads::<8>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<1, { u32::MAX }>::new(),
            ActSlotConst::<2, { u32::MAX }>::new(),
            ActSlotConst::<3, { u32::MAX }>::new(),
            WeightAccessorConst::<0, { u32::MAX }>::new(),
            WeightAccessorConst::<1, { u32::MAX }>::new(),
            "W::qkv".to_string(),
            "W::rot".to_string(),
            true,
            false,
        );
        let tape = b.finish();
        let MegaNode::FusedQkvRopeCache(n) = &tape.nodes()[0] else {
            panic!();
        };
        assert_eq!(n.in_page().raw(), 0);
        assert_eq!(n.qkv_weight_page().raw(), 1);
        assert_eq!(n.cos_sin_page().raw(), 2);
        assert_eq!(n.q_out_page().raw(), 3);
        assert_eq!(n.k_out_page().raw(), 4);
        assert_eq!(n.v_out_page().raw(), 5);
        assert_eq!(n.q_rope_offset().raw(), 0);
        assert_eq!(n.q_rope_bytes().raw(), 2048);
        assert_eq!(n.k_rope_offset().raw(), 2048);
        assert_eq!(n.k_rope_bytes().raw(), 2048);
        assert_eq!(n.iters().raw(), 4);
        assert!(n.biased);
    }

    #[test]
    fn lowers_add_minimal() {
        use crate::substrate::{
            ActSlotConst, ArrivesCount, HiddenDim, MbarrierPhase, NumTokensConst, PageId,
        };
        let mut b = Builder6::new();
        b.push_add(
            ArrivesCount::<0>::new(),
            PageId::<0, 6>::new(),
            PageId::<1, 6>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<8>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<1, { u32::MAX }>::new(),
        );
        let tape = b.finish();
        let MegaNode::Add(n) = &tape.nodes()[0] else {
            panic!();
        };
        assert_eq!(n.delta_page().raw(), 0);
        assert_eq!(n.residual_page().raw(), 1);
        assert_eq!(n.hidden_dim().raw(), 2048);
        assert_eq!(n.num_tokens().raw(), 8);
        assert_eq!(n.delta_act_slot().raw(), 0);
        assert_eq!(n.residual_act_slot().raw(), 1);
    }

    #[test]
    fn lowers_fused_add_rms_norm() {
        use crate::nodes::LayerIndex;
        use crate::substrate::{
            ActSlotConst, ArrivesCount, HiddenDim, MbarrierPhase, NumTokensConst, PageId,
            RmsNormScope, ScratchRegion, WeightAccessorConst,
        };
        let mut b = Builder6::new();
        b.push_fused_add_rms_norm(
            ArrivesCount::<0>::new(),
            PageId::<0, 6>::new(),
            PageId::<1, 6>::new(),
            PageId::<2, 6>::new(),
            ScratchRegion::<0, 32, 8192, RmsNormScope>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            LayerIndex::<3, 16>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<8>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<1, { u32::MAX }>::new(),
            WeightAccessorConst::<0, { u32::MAX }>::new(),
            "W::norm".to_string(),
            1.0e-5_f32,
        );
        let tape = b.finish();
        let MegaNode::FusedAddRmsNorm(n) = &tape.nodes()[0] else {
            panic!();
        };
        assert_eq!(n.layer().raw(), 3);
        assert_eq!(n.partial_bytes().raw(), 32);
    }

    #[test]
    fn lowers_fused_gate_up_silu_mul() {
        use crate::nodes::LayerIndex;
        use crate::substrate::{
            ActSlotConst, ArrivesCount, HiddenDim, IntermediateDim, IterCount, MbarrierPhase,
            MlpScope, NumTokensConst, PageId, ScratchRegion, WeightAccessorConst,
        };
        let mut b = Builder8::new();
        b.push_fused_gate_up_activate_mul(
            ArrivesCount::<0>::new(),
            PageId::<0, 8>::new(),
            PageId::<1, 8>::new(),
            PageId::<2, 8>::new(),
            ScratchRegion::<0, 2048, 8192, MlpScope>::new(),
            ScratchRegion::<2048, 2048, 8192, MlpScope>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            IterCount::<8>::new(),
            LayerIndex::<5, 16>::new(),
            HiddenDim::<2048>::new(),
            IntermediateDim::<8192>::new(),
            NumTokensConst::<8>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<2, { u32::MAX }>::new(),
            WeightAccessorConst::<0, { u32::MAX }>::new(),
            "W::mlp".to_string(),
            GateUpActivation::Silu,
        );
        let tape = b.finish();
        let MegaNode::FusedGateUpActivateMul(n) = &tape.nodes()[0] else {
            panic!();
        };
        assert_eq!(n.iters().raw(), 8);
        assert_eq!(n.activation, GateUpActivation::Silu);
        assert_eq!(n.hidden_dim().raw(), 2048);
        assert_eq!(n.intermediate_dim().raw(), 8192);
        assert_eq!(n.num_tokens().raw(), 8);
        assert_eq!(n.in_act_slot().raw(), 0);
        assert_eq!(n.out_act_slot().raw(), 2);
        assert_eq!(n.weight_accessor_idx().raw(), 0);
    }

    #[test]
    fn lowers_embed() {
        use crate::substrate::{
            ActSlotConst, ArrivesCount, HiddenDim, MbarrierPhase, NumTokensConst, PageId,
            VocabSize, WeightAccessorConst,
        };
        let mut b = BuilderD::new();
        b.push_embed(
            ArrivesCount::<0>::new(),
            PageId::<0, 8>::new(),
            PageId::<1, 8>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<8>::new(),
            VocabSize::<128_000>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            WeightAccessorConst::<0, { u32::MAX }>::new(),
            "W::embed".to_string(),
        );
        let tape = b.finish();
        let MegaNode::Embed(n) = &tape.nodes()[0] else {
            panic!();
        };
        assert_eq!(n.out_page().raw(), 0);
        assert_eq!(n.embed_weight_page().raw(), 1);
        assert_eq!(n.hidden_dim().raw(), 2048);
        assert_eq!(n.num_tokens().raw(), 8);
        assert_eq!(n.vocab_size().raw(), 128_000);
        assert_eq!(n.out_act_slot().raw(), 0);
        assert_eq!(n.weight_accessor_idx().raw(), 0);
    }

    #[test]
    fn lowers_scalar_mul_finite_scale() {
        use crate::substrate::{
            ActSlotConst, ArrivesCount, HiddenDim, MbarrierPhase, NumTokensConst, PageId,
        };
        let mut b = BuilderD::new();
        b.push_scalar_mul(
            ArrivesCount::<0>::new(),
            PageId::<0, 8>::new(),
            PageId::<1, 8>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<8>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<1, { u32::MAX }>::new(),
            0.5,
        );
        let tape = b.finish();
        let MegaNode::ScalarMul(n) = &tape.nodes()[0] else {
            panic!();
        };
        assert_eq!(n.scale.raw(), 0.5);
        assert_eq!(n.hidden_dim().raw(), 2048);
        assert_eq!(n.in_act_slot().raw(), 0);
        assert_eq!(n.out_act_slot().raw(), 1);
    }

    #[test]
    #[should_panic(expected = "FiniteF32 rejects non-finite value: NaN")]
    fn scalar_mul_rejects_nan_scale() {
        use crate::substrate::{
            ActSlotConst, ArrivesCount, HiddenDim, MbarrierPhase, NumTokensConst, PageId,
        };
        let mut b = BuilderD::new();
        b.push_scalar_mul(
            ArrivesCount::<0>::new(),
            PageId::<0, 8>::new(),
            PageId::<1, 8>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<8>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<1, { u32::MAX }>::new(),
            f32::NAN,
        );
    }

    #[test]
    fn lowers_tanh_soft_cap() {
        use crate::substrate::{
            ActSlotConst, ArrivesCount, HiddenDim, MbarrierPhase, NumTokensConst, PageId,
        };
        let mut b = BuilderD::new();
        b.push_tanh_soft_cap(
            ArrivesCount::<0>::new(),
            PageId::<0, 8>::new(),
            PageId::<1, 8>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<8>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<1, { u32::MAX }>::new(),
            30.0,
        );
        let tape = b.finish();
        let MegaNode::TanhSoftCap(n) = &tape.nodes()[0] else {
            panic!();
        };
        assert_eq!(n.cap.raw(), 30.0);
        assert_eq!(n.hidden_dim().raw(), 2048);
        assert_eq!(n.num_tokens().raw(), 8);
    }

    #[test]
    fn lowers_scalar_offset_rms_norm() {
        use crate::nodes::LayerIndex;
        use crate::substrate::{
            ActSlotConst, ArrivesCount, HiddenDim, MbarrierPhase, NumTokensConst, PageId,
            RmsNormScope, ScratchRegion, WeightAccessorConst,
        };
        let mut b = BuilderD::new();
        b.push_scalar_offset_rms_norm(
            ArrivesCount::<0>::new(),
            PageId::<0, 8>::new(),
            PageId::<1, 8>::new(),
            ScratchRegion::<0, 32, 32_768, RmsNormScope>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            LayerIndex::<5, 16>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<8>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<1, { u32::MAX }>::new(),
            WeightAccessorConst::<0, { u32::MAX }>::new(),
            "W::norm".to_string(),
            1.0,
            1.0e-5_f32,
        );
        let tape = b.finish();
        let MegaNode::ScalarOffsetRmsNorm(n) = &tape.nodes()[0] else {
            panic!();
        };
        assert_eq!(n.layer().raw(), 5);
        assert_eq!(n.offset.raw(), 1.0);
        assert_eq!(n.hidden_dim().raw(), 2048);
        assert_eq!(n.num_tokens().raw(), 8);
        assert_eq!(n.in_act_slot().raw(), 0);
        assert_eq!(n.out_act_slot().raw(), 1);
        assert_eq!(n.weight_accessor_idx().raw(), 0);
        assert!((n.eps().raw() - 1.0e-5_f32).abs() < 1e-9);
    }

    #[test]
    fn lowers_gemm() {
        let mut b = BuilderD::new();
        b.push_gemm::<0, 1, 2, 0, 4096, 0, 1, 4, 3, 4096, 2048, 16, 0, 8, 0, 2, 0>(
            "W::gemm".to_string(),
        );
        let tape = b.finish();
        let MegaNode::Gemm(n) = &tape.nodes()[0] else {
            panic!();
        };
        assert_eq!(n.n(), 4096);
        assert_eq!(n.k(), 2048);
        assert_eq!(n.iters(), 4);
    }

    #[test]
    fn lowers_lm_head_rms_norm_no_delta() {
        let mut b = BuilderD::new();
        b.push_cutlass_fused_norm_gemm_no_delta::<
            0, 1, 2, 3,
            0, 32, 4096, 4096,
            0, 1, 1, 0,
            128_000, 4_096,
            16, 0,
            8, 0, 1, 0, 1,
        >(
            "W::norm".to_string(),
            "W::lm_head".to_string(),
            LmHeadNormKind::RmsNorm,
            1.0e-5_f32,
        );
        let tape = b.finish();
        let MegaNode::CutlassFusedNormGemm(n) = &tape.nodes()[0] else {
            panic!();
        };
        assert_eq!(n.norm_kind, LmHeadNormKind::RmsNorm);
        assert!(n.delta_page_id().is_none());
    }

    #[test]
    fn lowers_lm_head_add_scalar_offset_rms_norm_with_delta() {
        let mut b = BuilderD::new();
        b.push_cutlass_fused_norm_gemm_with_delta::<
            0, 1, 2, 3, 4,
            0, 32, 4096, 4096,
            0, 1, 1, 0,
            128_000, 4_096,
            16, 0,
            8, 0, 1, 2, 0, 1,
        >(
            "W::norm".to_string(),
            "W::lm_head".to_string(),
            LmHeadNormKind::AddScalarOffsetRmsNorm,
            Some(1.0),
            1.0e-5_f32,
        );
        let tape = b.finish();
        let MegaNode::CutlassFusedNormGemm(n) = &tape.nodes()[0] else {
            panic!();
        };
        assert_eq!(n.norm_kind, LmHeadNormKind::AddScalarOffsetRmsNorm);
        assert_eq!(n.delta_page_id(), Some(1));
        assert_eq!(n.offset.unwrap().raw(), 1.0);
    }

    #[test]
    #[should_panic(expected = "AddScalarOffsetRmsNorm requires Some(offset)")]
    fn lm_head_rejects_missing_offset_for_scalar_offset_kind() {
        let mut b = BuilderD::new();
        b.push_cutlass_fused_norm_gemm_with_delta::<
            0, 1, 2, 3, 4,
            0, 32, 4096, 4096,
            0, 1, 1, 0,
            128_000, 4_096,
            16, 0,
            8, 0, 1, 2, 0, 1,
        >(
            "W::norm".to_string(),
            "W::lm_head".to_string(),
            LmHeadNormKind::AddScalarOffsetRmsNorm,
            None,
            1.0e-5_f32,
        );
    }

    #[test]
    fn lowers_attention_via_cache_full() {
        let mut b = BuilderD::new();
        b.push_attention_via_cache::<
            0, 1, 0, 4096, 4096, 4096, 0, 1, 8, 5, 16, 0,
            64, 32, 8, 16, 8, 8192, 0, 1,
        >(
            AttentionKind::Full,
            false,
            0.125_f32,
            0.0_f32,
        );
        let tape = b.finish();
        let MegaNode::AttentionViaCache(n) = &tape.nodes()[0] else {
            panic!();
        };
        assert_eq!(n.kv_cache_layer(), 5);
        assert_eq!(n.iters(), 8);
        assert!(matches!(n.kind, AttentionKind::Full));
    }

    #[test]
    fn lowers_barrier_signal_and_wait() {
        let mut b = BuilderD::new();
        b.push_barrier_signal::<0>();
        b.push_barrier_wait::<0, 4>();
        let tape = b.finish();
        let MegaNode::BarrierSignal(s) = &tape.nodes()[0] else {
            panic!();
        };
        assert_eq!(s.edge(), 0);
        let MegaNode::BarrierWait(w) = &tape.nodes()[1] else {
            panic!();
        };
        assert_eq!(w.edge(), 0);
        assert_eq!(w.expected(), 4);
    }

    #[test]
    fn barriers_do_not_advance_arrives() {
        use crate::nodes::LayerIndex;
        use crate::substrate::{
            ActSlotConst, ArrivesCount, BarSyncId, BarSyncPair, HiddenDim, MbarrierPhase,
            NumTokensConst, PageId, RmsNormScope, ScratchRegion, WeightAccessorConst,
        };
        // Barriers don't bump the per-CTA mbarrier count, so the
        // next op's ARRIVES const stays at 0 even after two barriers.
        let mut b = BuilderD::new();
        b.push_barrier_signal::<0>();
        b.push_barrier_wait::<0, 4>();
        b.push_rms_norm(
            ArrivesCount::<0>::new(),
            PageId::<0, 8>::new(),
            PageId::<1, 8>::new(),
            ScratchRegion::<0, 32, 32_768, RmsNormScope>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            LayerIndex::<0, 16>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<8>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<1, { u32::MAX }>::new(),
            WeightAccessorConst::<0, { u32::MAX }>::new(),
            BarSyncId::<1>::new(),
            BarSyncId::<2>::new(),
            BarSyncPair::<1, 2>::new(),
            "W::n".to_string(),
            1.0e-5_f32,
        );
        let tape = b.finish();
        assert_eq!(tape.nodes().len(), 3);
    }

    #[test]
    fn builder_arrives_visible_to_caller() {
        use crate::nodes::LayerIndex;
        use crate::substrate::{
            ActSlotConst, ArrivesCount, BarSyncId, BarSyncPair, HiddenDim, MbarrierPhase,
            NumTokensConst, PageId, RmsNormScope, ScratchRegion, WeightAccessorConst,
        };
        let mut b = Builder6::new();
        assert_eq!(b.arrives(), 0);
        b.push_rms_norm(
            ArrivesCount::<0>::new(),
            PageId::<0, 6>::new(),
            PageId::<1, 6>::new(),
            ScratchRegion::<0, 32, 8192, RmsNormScope>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            LayerIndex::<0, 16>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<8>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<1, { u32::MAX }>::new(),
            WeightAccessorConst::<0, { u32::MAX }>::new(),
            BarSyncId::<1>::new(),
            BarSyncId::<2>::new(),
            BarSyncPair::<1, 2>::new(),
            "W::n".to_string(),
            1.0e-5_f32,
        );
        assert_eq!(b.arrives(), 1);
        b.push_rms_norm(
            ArrivesCount::<1>::new(),
            PageId::<0, 6>::new(),
            PageId::<1, 6>::new(),
            ScratchRegion::<0, 32, 8192, RmsNormScope>::new(),
            MbarrierPhase::<1>::new(),
            MbarrierPhase::<0>::new(),
            LayerIndex::<0, 16>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<8>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<1, { u32::MAX }>::new(),
            WeightAccessorConst::<0, { u32::MAX }>::new(),
            BarSyncId::<1>::new(),
            BarSyncId::<2>::new(),
            BarSyncPair::<1, 2>::new(),
            "W::n".to_string(),
            1.0e-5_f32,
        );
        assert_eq!(b.arrives(), 2);
    }
}
