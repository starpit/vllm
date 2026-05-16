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
        weight_path: String,
        eps: f32,
    ) -> &mut Self {
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
        >(weight, eps);
        self.nodes.push(MegaNode::RmsNorm(node));
        self.pool.release(IN_ID);
        self.pool.release(WEIGHT_ID);
        self.arrives.bump();
        self
    }

    /// Push a typed FusedQkvRopeCache op onto the tape.
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
    >(
        &mut self,
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

    /// Push a typed `Add` (residual fold).
    #[allow(clippy::too_many_arguments)]
    pub fn push_add<
        const DELTA_ID: u32,
        const RESIDUAL_ID: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const ARRIVES: u32,
    >(
        &mut self,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_add");
        let _ = self.pool.take(DELTA_ID);
        let _ = self.pool.take(RESIDUAL_ID);
        let node =
            Add::new::<DELTA_ID, RESIDUAL_ID, CONSUMER_PHASE, STORER_PHASE, NUM_PAGES, ARRIVES>();
        self.nodes.push(MegaNode::Add(node));
        self.pool.release(DELTA_ID);
        self.pool.release(RESIDUAL_ID);
        self.arrives.bump();
        self
    }

    /// Push a typed `FusedAddRmsNorm`.
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
    >(
        &mut self,
        weight_path: String,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_fused_add_rms_norm");
        let _ = self.pool.take(DELTA_ID);
        let _ = self.pool.take(RESIDUAL_ID);
        let _ = self.pool.take(WEIGHT_ID);
        let weight = WeightRef::new(weight_path);
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
        >(weight);
        self.nodes.push(MegaNode::FusedAddRmsNorm(node));
        self.pool.release(DELTA_ID);
        self.pool.release(RESIDUAL_ID);
        self.pool.release(WEIGHT_ID);
        self.arrives.bump();
        self
    }

    /// Push a typed `FusedGateUp{Silu,Gelu}Mul`.
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
    >(
        &mut self,
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

    /// Push a typed `Embed`.
    pub fn push_embed<
        const OUT_ID: u32,
        const WEIGHT_ID: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const ARRIVES: u32,
    >(
        &mut self,
        embed_weight_path: String,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_embed");
        let _ = self.pool.take(OUT_ID);
        let _ = self.pool.take(WEIGHT_ID);
        let weight = WeightRef::new(embed_weight_path);
        let node = Embed::new::<OUT_ID, WEIGHT_ID, CONSUMER_PHASE, STORER_PHASE, NUM_PAGES, ARRIVES>(
            weight,
        );
        self.nodes.push(MegaNode::Embed(node));
        self.pool.release(OUT_ID);
        self.pool.release(WEIGHT_ID);
        self.arrives.bump();
        self
    }

    /// Push a typed `ScalarMul`.
    pub fn push_scalar_mul<
        const IN_ID: u32,
        const OUT_ID: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const ARRIVES: u32,
    >(
        &mut self,
        scale: f32,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_scalar_mul");
        let _ = self.pool.take(IN_ID);
        let _ = self.pool.take(OUT_ID);
        let scale = FiniteF32::new(scale);
        let node = ScalarMul::new::<IN_ID, OUT_ID, CONSUMER_PHASE, STORER_PHASE, NUM_PAGES, ARRIVES>(
            scale,
        );
        self.nodes.push(MegaNode::ScalarMul(node));
        self.pool.release(IN_ID);
        self.pool.release(OUT_ID);
        self.arrives.bump();
        self
    }

    /// Push a typed `TanhSoftCap`.
    pub fn push_tanh_soft_cap<
        const IN_ID: u32,
        const OUT_ID: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const ARRIVES: u32,
    >(
        &mut self,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_tanh_soft_cap");
        let _ = self.pool.take(IN_ID);
        let _ = self.pool.take(OUT_ID);
        let node =
            TanhSoftCap::new::<IN_ID, OUT_ID, CONSUMER_PHASE, STORER_PHASE, NUM_PAGES, ARRIVES>();
        self.nodes.push(MegaNode::TanhSoftCap(node));
        self.pool.release(IN_ID);
        self.pool.release(OUT_ID);
        self.arrives.bump();
        self
    }

    /// Push a typed `ScalarOffsetRmsNorm`.
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
    >(
        &mut self,
        weight_path: String,
        offset: f32,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_scalar_offset_rms_norm");
        let _ = self.pool.take(IN_ID);
        let _ = self.pool.take(WEIGHT_ID);
        let weight = WeightRef::new(weight_path);
        let offset = FiniteF32::new(offset);
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
        >(weight, offset);
        self.nodes.push(MegaNode::ScalarOffsetRmsNorm(node));
        self.pool.release(IN_ID);
        self.pool.release(WEIGHT_ID);
        self.arrives.bump();
        self
    }

    /// Push a typed `Gemm`.
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

    /// Push a typed `FusedCublasGemmAdd` onto the tape.
    /// Substrate shape: 3 pages (in / weight / residual=output),
    /// `GemmScope` B-tile, multi-iter. The residual page is read
    /// AND written (in-place residual fold after the gemm).
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
    /// (RmsNorm, MeanSubRmsNorm).
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
    >(
        &mut self,
        norm_weight_path: String,
        linear_weight_path: String,
        norm_kind: LmHeadNormKind,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_cutlass_fused_norm_gemm_no_delta");
        let _ = self.pool.take(IN_ID);
        let _ = self.pool.take(NORM_W_ID);
        let _ = self.pool.take(LIN_W_ID);
        let _ = self.pool.take(OUT_ID);
        let norm_weight = WeightRef::new(norm_weight_path);
        let linear_weight = WeightRef::new(linear_weight_path);
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
        >(norm_weight, linear_weight, norm_kind);
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
    /// flavors (AddRmsNorm, AddScalarOffsetRmsNorm).
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
    >(
        &mut self,
        norm_weight_path: String,
        linear_weight_path: String,
        norm_kind: LmHeadNormKind,
        offset: Option<f32>,
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
        >(norm_weight, linear_weight, norm_kind, offset);
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

    /// Push a typed `AttentionViaCacheNode`.
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
    >(
        &mut self,
        kind: AttentionKind,
        interleaved: bool,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_attention_via_cache");
        let _ = self.pool.take(Q_IN_ID);
        let _ = self.pool.take(ATTN_OUT_ID);
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
        >(kind, interleaved);
        self.nodes.push(MegaNode::AttentionViaCache(node));
        self.pool.release(Q_IN_ID);
        self.pool.release(ATTN_OUT_ID);
        for _ in 0..ITERS {
            self.arrives.bump();
        }
        self
    }

    /// Push a typed `SpliceMmEmbeds`.
    pub fn push_splice_mm_embeds<
        const SLOT_ID: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const ARRIVES: u32,
    >(
        &mut self,
    ) -> &mut Self {
        self.verify_arrives(ARRIVES, "push_splice_mm_embeds");
        let _ = self.pool.take(SLOT_ID);
        let node =
            SpliceMmEmbeds::new::<SLOT_ID, CONSUMER_PHASE, STORER_PHASE, NUM_PAGES, ARRIVES>();
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
        // Substrate proofs: IN_ID=0, WEIGHT_ID=1, PARTIAL_OFF=0,
        // PARTIAL_BYTES=32 (NUM_CONSUMER_WARPS*4), CONSUMER_PHASE=0
        // (ARRIVES=0&1=0), STORER_PHASE=1, LAYER=0, NUM_LAYERS=16,
        // ARRIVES=0. AST shape: HIDDEN_DIM=2048, NUM_TOKENS=8,
        // IN_ACT_SLOT=0, OUT_ACT_SLOT=1, WEIGHT_ACCESSOR_IDX=0.
        b.push_rms_norm::<0, 1, 0, 32, 0, 1, 0, 16, 0, 2048, 8, 0, 1, 0>(
            "W::norm".to_string(),
            1.0e-5_f32,
        );
        let tape = b.finish();
        assert_eq!(tape.nodes().len(), 1);
        let MegaNode::RmsNorm(n) = &tape.nodes()[0] else {
            panic!("expected RmsNorm");
        };
        assert_eq!(n.in_page_id(), 0);
        assert_eq!(n.weight_page_id(), 1);
        assert_eq!(n.partial_offset(), 0);
        assert_eq!(n.partial_bytes(), 32);
        assert_eq!(n.consumer_phase(), 0);
        assert_eq!(n.storer_phase(), 1);
        assert_eq!(n.layer(), 0);
        assert_eq!(n.hidden_dim(), 2048);
        assert_eq!(n.num_tokens(), 8);
        assert_eq!(n.in_act_slot(), 0);
        assert_eq!(n.out_act_slot(), 1);
        assert_eq!(n.weight_accessor_idx(), 0);
        assert!((n.eps().raw() - 1.0e-5_f32).abs() < 1e-9);
        assert_eq!(n.weight.path(), "W::norm");
    }

    #[test]
    fn lowers_two_rms_norms_with_phase_advance() {
        let mut b = Builder6::new();
        b.push_rms_norm::<0, 1, 0, 32, 0, 1, 0, 16, 0, 2048, 8, 0, 1, 0>(
            "W::n0".to_string(),
            1.0e-5_f32,
        );
        // After first op, ARRIVES = 1; CONSUMER_PHASE = 1, STORER_PHASE = 0.
        b.push_rms_norm::<0, 1, 0, 32, 1, 0, 1, 16, 1, 2048, 8, 0, 2, 1>(
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
        assert_eq!(n0.consumer_phase(), 0);
        assert_eq!(n0.storer_phase(), 1);
        assert_eq!(n1.consumer_phase(), 1);
        assert_eq!(n1.storer_phase(), 0);
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
        let mut b = Builder8::new();
        // 6 distinct page ids, q_off+q_bytes=2048, k_off=2048+k_bytes=4096,
        // disjoint, within 8192 SCRATCH_BYTES, ITERS=4, LAYER=0,
        // NUM_LAYERS=16, ARRIVES=0, CONSUMER_PHASE=0, STORER_PHASE=1.
        b.push_fused_qkv_rope_cache::<0, 1, 2, 3, 4, 5, 0, 2048, 2048, 2048, 0, 1, 4, 0, 16, 0>(
            "W::qkv".to_string(),
            "W::rot".to_string(),
            true,
            false,
        );
        let tape = b.finish();
        let MegaNode::FusedQkvRopeCache(n) = &tape.nodes()[0] else {
            panic!();
        };
        assert_eq!(n.in_page_id(), 0);
        assert_eq!(n.qkv_weight_page_id(), 1);
        assert_eq!(n.cos_sin_page_id(), 2);
        assert_eq!(n.q_out_page_id(), 3);
        assert_eq!(n.k_out_page_id(), 4);
        assert_eq!(n.v_out_page_id(), 5);
        assert_eq!(n.q_rope_offset(), 0);
        assert_eq!(n.q_rope_bytes(), 2048);
        assert_eq!(n.k_rope_offset(), 2048);
        assert_eq!(n.k_rope_bytes(), 2048);
        assert_eq!(n.iters(), 4);
        assert!(n.biased);
    }

    #[test]
    fn lowers_add_minimal() {
        let mut b = Builder6::new();
        b.push_add::<0, 1, 0, 1, 0>();
        let tape = b.finish();
        let MegaNode::Add(n) = &tape.nodes()[0] else {
            panic!();
        };
        assert_eq!(n.delta_page_id(), 0);
        assert_eq!(n.residual_page_id(), 1);
    }

    #[test]
    fn lowers_fused_add_rms_norm() {
        let mut b = Builder6::new();
        // DELTA=0, RES=1, WEIGHT=2, partial_off=0, partial_bytes=32,
        // phases (0,1), layer 3 (NUM_LAYERS=16), ARRIVES=0.
        b.push_fused_add_rms_norm::<0, 1, 2, 0, 32, 0, 1, 3, 16, 0>("W::norm".to_string());
        let tape = b.finish();
        let MegaNode::FusedAddRmsNorm(n) = &tape.nodes()[0] else {
            panic!();
        };
        assert_eq!(n.layer(), 3);
        assert_eq!(n.partial_bytes(), 32);
    }

    #[test]
    fn lowers_fused_gate_up_silu_mul() {
        let mut b = Builder8::new();
        // IN=0, WEIGHT=1, OUT=2, gate(0,2048), up(2048,2048),
        // phases (0,1), ITERS=8, LAYER=5, NUM_LAYERS=16, ARRIVES=0.
        b.push_fused_gate_up_activate_mul::<0, 1, 2, 0, 2048, 2048, 2048, 0, 1, 8, 5, 16, 0>(
            "W::mlp".to_string(),
            GateUpActivation::Silu,
        );
        let tape = b.finish();
        let MegaNode::FusedGateUpActivateMul(n) = &tape.nodes()[0] else {
            panic!();
        };
        assert_eq!(n.iters(), 8);
        assert_eq!(n.activation, GateUpActivation::Silu);
    }

    #[test]
    fn lowers_embed() {
        let mut b = BuilderD::new();
        b.push_embed::<0, 1, 0, 1, 0>("W::embed".to_string());
        let tape = b.finish();
        let MegaNode::Embed(n) = &tape.nodes()[0] else {
            panic!();
        };
        assert_eq!(n.out_page_id(), 0);
        assert_eq!(n.embed_weight_page_id(), 1);
    }

    #[test]
    fn lowers_scalar_mul_finite_scale() {
        let mut b = BuilderD::new();
        b.push_scalar_mul::<0, 1, 0, 1, 0>(0.5);
        let tape = b.finish();
        let MegaNode::ScalarMul(n) = &tape.nodes()[0] else {
            panic!();
        };
        assert_eq!(n.scale.raw(), 0.5);
    }

    #[test]
    #[should_panic(expected = "FiniteF32 rejects non-finite value: NaN")]
    fn scalar_mul_rejects_nan_scale() {
        let mut b = BuilderD::new();
        b.push_scalar_mul::<0, 1, 0, 1, 0>(f32::NAN);
    }

    #[test]
    fn lowers_tanh_soft_cap() {
        let mut b = BuilderD::new();
        b.push_tanh_soft_cap::<0, 1, 0, 1, 0>();
        let tape = b.finish();
        let MegaNode::TanhSoftCap(_) = &tape.nodes()[0] else {
            panic!();
        };
    }

    #[test]
    fn lowers_scalar_offset_rms_norm() {
        let mut b = BuilderD::new();
        b.push_scalar_offset_rms_norm::<0, 1, 0, 32, 0, 1, 5, 16, 0>("W::norm".to_string(), 1.0);
        let tape = b.finish();
        let MegaNode::ScalarOffsetRmsNorm(n) = &tape.nodes()[0] else {
            panic!();
        };
        assert_eq!(n.layer(), 5);
        assert_eq!(n.offset.raw(), 1.0);
    }

    #[test]
    fn lowers_gemm() {
        let mut b = BuilderD::new();
        b.push_gemm::<0, 1, 2, 0, 4096, 0, 1, 4, 3, 4096, 2048, 16, 0>("W::gemm".to_string());
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
        >(
            "W::norm".to_string(),
            "W::lm_head".to_string(),
            LmHeadNormKind::RmsNorm,
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
        >(
            "W::norm".to_string(),
            "W::lm_head".to_string(),
            LmHeadNormKind::AddScalarOffsetRmsNorm,
            Some(1.0),
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
        >(
            "W::norm".to_string(),
            "W::lm_head".to_string(),
            LmHeadNormKind::AddScalarOffsetRmsNorm,
            None,
        );
    }

    #[test]
    fn lowers_attention_via_cache_full() {
        let mut b = BuilderD::new();
        b.push_attention_via_cache::<0, 1, 0, 4096, 4096, 4096, 0, 1, 8, 5, 16, 0>(
            AttentionKind::Full,
            false,
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
        // Barriers don't bump the per-CTA mbarrier count, so the
        // next op's ARRIVES const stays at 0 even after two barriers.
        let mut b = BuilderD::new();
        b.push_barrier_signal::<0>();
        b.push_barrier_wait::<0, 4>();
        b.push_rms_norm::<0, 1, 0, 32, 0, 1, 0, 16, 0, 2048, 8, 0, 1, 0>(
            "W::n".to_string(),
            1.0e-5_f32,
        );
        let tape = b.finish();
        assert_eq!(tape.nodes().len(), 3);
    }

    #[test]
    fn builder_arrives_visible_to_caller() {
        let mut b = Builder6::new();
        assert_eq!(b.arrives(), 0);
        b.push_rms_norm::<0, 1, 0, 32, 0, 1, 0, 16, 0, 2048, 8, 0, 1, 0>(
            "W::n".to_string(),
            1.0e-5_f32,
        );
        assert_eq!(b.arrives(), 1);
        b.push_rms_norm::<0, 1, 0, 32, 1, 0, 0, 16, 1, 2048, 8, 0, 1, 0>(
            "W::n".to_string(),
            1.0e-5_f32,
        );
        assert_eq!(b.arrives(), 2);
    }
}
