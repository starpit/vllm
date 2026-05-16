// SPDX-License-Identifier: Apache-2.0
//! Typed lowered-form variants.
//!
//! One variant per `Instruction<W>` kind. Each variant's
//! load-bearing fields are substrate proofs from
//! [`crate::substrate`] — `PageId` (validated against
//! `num_pages`), `MbarrierPhase` (parity-checked against cumulative
//! arrive count), `ScratchRegion<Scope>` (within-budget +
//! disjoint), `WarpRoleTag<R>` (sealed valid-role) — chosen so that
//! constructing a variant value requires the substrate-layout
//! invariants the corresponding bug class targets (see
//! `MEGA_IR_PLAN.md` §1, §4).
//!
//! Helper newtypes (e.g. [`LayerIndex`], [`WeightRef`]) MAY ride
//! alongside for runtime weight-pointer resolution; they are NOT
//! load-bearing on their own (per `MEGA_IR_PLAN.md` §3, §4).

#![allow(dead_code)]

use crate::substrate::{
    AttentionScope, EdgeId, ExpectedCount, GemmScope, IterCount, MbarrierPhase, MlpScope, PageId,
    ROLE_CONSUMER, ROLE_LAUNCHER, ROLE_LOADER, ROLE_STORER, RmsNormScope, RopeScope, ScratchRegion,
    WarpRoleTag,
};

/// Helper newtype: layer index for runtime weight-pointer
/// resolution. Variant fields use this ALONGSIDE substrate-proof
/// fields; never as the load-bearing field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayerIndex(u32);

impl LayerIndex {
    pub fn new(idx: u32, num_layers: u32) -> Self {
        assert!(
            idx < num_layers,
            "LayerIndex out of range: {idx} >= {num_layers}",
        );
        Self(idx)
    }

    pub fn raw(self) -> u32 {
        self.0
    }
}

/// Helper newtype: per-layer weight accessor path string.
/// Construction enforces non-empty / non-whitespace.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WeightRef(String);

impl WeightRef {
    pub fn new(path: String) -> Self {
        assert!(
            !path.trim().is_empty(),
            "WeightRef must be a non-empty path string",
        );
        Self(path)
    }

    pub fn path(&self) -> &str {
        &self.0
    }
}

/// Per-iter mbarrier wait/arrive sequence for one warp role.
/// Sprint A: RmsNorm has two consumer waits (input page,
/// weight page) per iter at the same phase, one consumer arrive
/// (output ready), and one storer wait (output) per iter.
///
/// `WarpRoleTag<R>` pins the role at the type level (bug class
/// #6). `MbarrierPhase` carries the parity validated against the
/// cumulative arrive count up to this op (bug class #3).
pub struct PerIterPhases<const R: u8>
where
    WarpRoleTag<R>: crate::substrate::IsValidWarpRole,
{
    pub wait_phase: MbarrierPhase,
    pub _role: WarpRoleTag<R>,
}

/// The typed lowered RmsNorm variant.
///
/// Substrate proofs (load-bearing):
/// - `in_page`: the input row's page slot id (validated within
///   `num_pages`). Lifecycle: was Empty at op start, was filled by
///   the loader, was read by the consumer who then arrived, was
///   read by the storer, is back to Empty at op end. The lowering
///   walks Page<State> typestate transitions to prove this; the
///   stored PageId records the slot id only.
/// - `weight_page`: the weight row's page slot id. Lifecycle: was
///   Empty, filled by loader, read by consumer (no storer reads
///   weight). Returns to Empty at op end.
/// - `partial_sums`: per-iter scratch region for the cross-warp
///   sum-of-squares reduction. ScratchRegion<RmsNormScope> proves
///   `[offset, offset+bytes) ⊆ [0, num_consumer_warps * 4)` AND
///   within `substrate.scratch_bytes` (bug classes #4, #5).
/// - `consumer_phase`, `storer_phase`: phase parity for each
///   wait site, validated against the cumulative arrive count
///   (bug class #3).
/// - `_loader_role`, `_launcher_role`, `_consumer_role`,
///   `_storer_role`: type-level role tags (bug class #6 — the
///   role pairing is fixed at variant construction; cross-role
///   action calls in the emit step have no way to refer to a
///   role this variant doesn't claim).
///
/// Helper fields (alongside, not load-bearing):
/// - `layer`: for emit-time weight-pointer resolution.
/// - `weight`: weight accessor path.
pub struct RmsNorm {
    pub in_page: PageId,
    pub weight_page: PageId,
    pub partial_sums: ScratchRegion<RmsNormScope>,
    pub consumer_phase: MbarrierPhase,
    pub storer_phase: MbarrierPhase,
    pub _loader_role: WarpRoleTag<ROLE_LOADER>,
    pub _launcher_role: WarpRoleTag<ROLE_LAUNCHER>,
    pub _consumer_role: WarpRoleTag<ROLE_CONSUMER>,
    pub _storer_role: WarpRoleTag<ROLE_STORER>,
    pub layer: LayerIndex,
    pub weight: WeightRef,
}

/// Helper newtype: per-arch rotary cache accessor — distinguishes
/// the global rotary (`wm.rotary.cos_sin_cache`) from per-layer
/// local rotary (`wm.rotary_local.cos_sin_cache`) on archs that
/// carry both (Gemma3). Construction enforces non-empty path.
/// Not load-bearing on its own (per `MEGA_IR_PLAN.md` §3, §4).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RotaryRef(String);

impl RotaryRef {
    pub fn new(path: String) -> Self {
        assert!(
            !path.trim().is_empty(),
            "RotaryRef must be a non-empty path string",
        );
        Self(path)
    }

    pub fn path(&self) -> &str {
        &self.0
    }
}

/// The typed lowered FusedQkvRopeCache variant.
///
/// Substrate proofs (load-bearing):
///
/// **Six page slots, each walked Empty → Filled → Produced → Empty**
/// at proc-macro construction time, so the recorded `PageId`s are
/// witness to a valid lifecycle (bug class #2):
///
/// - `in_page`: post-norm activation row feeding the QKV gemm.
/// - `qkv_weight_page`: packed Q/K/V projection weights (single
///   accessor; the LinearLayer wrapper packs the three projections
///   into one buffer at load time).
/// - `cos_sin_page`: rotary cos/sin cache page.
/// - `q_out_page`, `k_out_page`, `v_out_page`: per-token outputs.
///   Q goes to the next op (attention); K/V additionally write to
///   the global paged KV cache (the `kv_cache_layer` extern; not
///   substrate-managed because it lives in HBM, not shmem-pages).
///
/// `PageId::raw()` is bounds-validated against
/// `SubstrateBudget::num_pages` (bug class #1).
///
/// **Two scratch regions, both in `RopeScope`, validated disjoint**
/// at proc-macro time (bug class #4):
///
/// - `q_rope_buf`: per-token Q rotation tile in shmem.
/// - `k_rope_buf`: per-token K rotation tile in shmem.
///
/// Each is independently validated within `SubstrateBudget::scratch_bytes`
/// (bug class #5); their disjointness is checked via
/// [`ScratchRegion::disjoint_with`].
///
/// **Per-iter mbarrier phases** (bug class #3):
///
/// - `consumer_phase`: parity at iter 0 of the consumer wait. The
///   lowering proves this matches the cumulative arrive count at
///   op start.
/// - `storer_phase`: parity at iter 0 of the storer wait. Lowered
///   to `(arrives + 1) & 1` — the storer waits AFTER the consumer
///   arrives.
/// - `iters`: per-token iteration count (one consumer arrive per
///   token-iter, so the cumulative arrive count advances by
///   `iters.raw() * 1` after this op completes).
///
/// The kernel-internal `phase ^= 1` advances per iter; given the
/// initial phase parity matches the start-of-op cumulative count
/// and `arrives_per_iter == 1`, the kernel-internal phase at iter
/// `t` matches `(C_start + t) & 1` — the parity the hardware
/// barrier expects (proof: `(a+b) & 1 == (a&1) ^ (b&1)`).
///
/// **Warp role tags** (bug class #6): all four roles claimed; mis-
/// pairing across roles in the variant declaration is rejected by
/// the sealed-trait bound on `WarpRoleTag<R>` at the field level.
///
/// Helper fields (alongside, not load-bearing):
/// - `layer`: emit-time layer index for weight-pointer resolution.
/// - `qkv_weight`: weight accessor path (LinearLayer holding packed Q/K/V).
/// - `rotary`: rotary accessor path (global or local cos/sin cache).
/// - `biased`: dense bias attached (Qwen2-family) vs no bias (Llama).
/// - `interleaved`: interleaved-rope half-rotation pattern.
pub struct FusedQkvRopeCache {
    pub in_page: PageId,
    pub qkv_weight_page: PageId,
    pub cos_sin_page: PageId,
    pub q_out_page: PageId,
    pub k_out_page: PageId,
    pub v_out_page: PageId,
    pub q_rope_buf: ScratchRegion<RopeScope>,
    pub k_rope_buf: ScratchRegion<RopeScope>,
    pub consumer_phase: MbarrierPhase,
    pub storer_phase: MbarrierPhase,
    pub iters: IterCount,
    pub _loader_role: WarpRoleTag<ROLE_LOADER>,
    pub _launcher_role: WarpRoleTag<ROLE_LAUNCHER>,
    pub _consumer_role: WarpRoleTag<ROLE_CONSUMER>,
    pub _storer_role: WarpRoleTag<ROLE_STORER>,
    pub layer: LayerIndex,
    pub qkv_weight: WeightRef,
    pub rotary: RotaryRef,
    pub biased: bool,
    pub interleaved: bool,
}

/// The typed lowered `Add` (residual fold) variant.
///
/// `Instruction::Add(delta_slot, residual_slot)` semantics:
/// `residual_slot += delta_slot` in place. The plan §10 calls this
/// shape "DownProjResidual" because the down-projection's output is
/// added back into the layer's residual stream; in the
/// `Instruction` enum it's the universal element-wise residual fold
/// (used after every MLP and attention block).
///
/// Substrate proofs (load-bearing):
/// - `delta_page`, `residual_page`: two `PageId`s, each
///   bounds-validated (#1) and lifecycle-walked Empty → Filled →
///   Produced → Empty during construction (#2). Cross-op aliasing
///   (#2 cross-op) is guarded by `PagePool::take`.
/// - `consumer_phase`, `storer_phase`: parity-validated against
///   the cumulative arrive count at op boundary (#3).
/// - 4 `WarpRoleTag<R>`: the four roles claimed (#6). The `Add`
///   kernel doesn't actually use a launcher (no TMA), but every
///   variant claims all four for consistency — emit-side dispatch
///   skips unused roles.
///
/// No scratch (element-wise add fits in registers; no shmem
/// reduction). No weight (purely two-input). No iters (single
/// op-level arrive).
pub struct Add {
    pub delta_page: PageId,
    pub residual_page: PageId,
    pub consumer_phase: MbarrierPhase,
    pub storer_phase: MbarrierPhase,
    pub _loader_role: WarpRoleTag<ROLE_LOADER>,
    pub _launcher_role: WarpRoleTag<ROLE_LAUNCHER>,
    pub _consumer_role: WarpRoleTag<ROLE_CONSUMER>,
    pub _storer_role: WarpRoleTag<ROLE_STORER>,
}

/// The typed lowered `FusedAddRmsNorm` variant.
///
/// `Instruction::FusedAddRmsNorm(delta_slot, residual_slot, layer)`
/// semantics: `residual_slot += delta_slot; out = rms_norm(residual,
/// weight[layer])` — the result lives back in `residual_slot`
/// (in-place norm). Used as the pre-attention norm + folded prior
/// residual (and the pre-MLP norm + folded attention residual) in
/// every transformer layer.
///
/// Substrate proofs (load-bearing):
/// - `delta_page`, `residual_page`, `weight_page`: three
///   lifecycle-walked + bounds-validated `PageId`s (#1, #2).
/// - `partial_sums`: per-iter cross-warp sum-of-squares reduction
///   in `RmsNormScope` (sibling regions on a future RmsNorm op land
///   in the same scope and get `disjoint_with`-discharged) — #4 +
///   #5.
/// - `consumer_phase`, `storer_phase`: #3.
/// - 4 `WarpRoleTag<R>`: #6.
///
/// One op-level arrive (the kernel-internal per-token reduction
/// folds into a single consumer arrive at the variant boundary).
pub struct FusedAddRmsNorm {
    pub delta_page: PageId,
    pub residual_page: PageId,
    pub weight_page: PageId,
    pub partial_sums: ScratchRegion<RmsNormScope>,
    pub consumer_phase: MbarrierPhase,
    pub storer_phase: MbarrierPhase,
    pub _loader_role: WarpRoleTag<ROLE_LOADER>,
    pub _launcher_role: WarpRoleTag<ROLE_LAUNCHER>,
    pub _consumer_role: WarpRoleTag<ROLE_CONSUMER>,
    pub _storer_role: WarpRoleTag<ROLE_STORER>,
    pub layer: LayerIndex,
    pub weight: WeightRef,
}

/// Activation choice for the gate-up MLP fusion. Two `Instruction`
/// variants (`FusedGateUpSiluMul`, `FusedGateUpGeluMul`) lower to
/// the same substrate shape — three pages, two disjoint
/// `MlpScope` tiles, one weight (packed gate+up `LinearLayer`).
/// The activation flag rides alongside as a helper field; emit
/// picks the kernel template, no substrate consequences.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GateUpActivation {
    Silu,
    Gelu,
}

/// The typed lowered `FusedGateUp{Silu,Gelu}Mul` variant.
///
/// Instruction semantics:
/// `out = silu(gemm(in, gate_w)) * gemm(in, up_w)` (or `gelu` for
/// the GeluMul peer). The `gate_w` and `up_w` are packed into a
/// single `LinearLayer` accessor at load time; the kernel issues
/// one cublas matmul and slices the output along the inner dim.
///
/// Substrate proofs (load-bearing):
/// - `in_page`, `gate_up_weight_page`, `out_page`: three pages
///   lifecycle-walked at construction (#1, #2). Aliasing across
///   them (in == out, in == weight, …) is rejected by `PagePool`.
/// - `gate_buf`, `up_buf`: two per-tok-iter activation tiles in
///   `MlpScope`, validated `disjoint_with` at construction (#4)
///   and each within budget (#5).
/// - `consumer_phase`, `storer_phase`: per-iter boundary parity
///   (#3 boundary; per-iter math falls out the same way as Sprint
///   B's Rope — `(C+t)&1 == (C&1)^(t&1)`).
/// - `iters`: per-token iteration count. The kernel issues two
///   gemms + one elementwise activate-mul per iter, with one
///   consumer arrive per iter; cumulative arrive count advances
///   by `iters.raw()`.
/// - 4 `WarpRoleTag<R>`: #6.
///
/// Helper fields:
/// - `layer`, `weight` (packed `gate_up`).
/// - `activation`: `Silu` or `Gelu`.
pub struct FusedGateUpActivateMul {
    pub in_page: PageId,
    pub gate_up_weight_page: PageId,
    pub out_page: PageId,
    pub gate_buf: ScratchRegion<MlpScope>,
    pub up_buf: ScratchRegion<MlpScope>,
    pub consumer_phase: MbarrierPhase,
    pub storer_phase: MbarrierPhase,
    pub iters: IterCount,
    pub _loader_role: WarpRoleTag<ROLE_LOADER>,
    pub _launcher_role: WarpRoleTag<ROLE_LAUNCHER>,
    pub _consumer_role: WarpRoleTag<ROLE_CONSUMER>,
    pub _storer_role: WarpRoleTag<ROLE_STORER>,
    pub layer: LayerIndex,
    pub weight: WeightRef,
    pub activation: GateUpActivation,
}

/// Helper newtype: finite f32 (rejects NaN / ±∞). Used alongside
/// substrate proofs by variants that carry scalar coefficients
/// (`ScalarMul`'s `scale`, `ScalarOffsetRmsNorm`'s `offset`,
/// `CutlassFusedAddScalarOffsetRmsNormGemm`'s `offset`). Not
/// load-bearing on its own (per `MEGA_IR_PLAN.md` §3, §4); the
/// load-bearing fields are still substrate proofs.
#[derive(Clone, Copy, Debug)]
pub struct FiniteF32(f32);

impl FiniteF32 {
    pub fn new(v: f32) -> Self {
        assert!(v.is_finite(), "FiniteF32 rejects non-finite value: {v}");
        Self(v)
    }

    pub fn raw(self) -> f32 {
        self.0
    }
}

impl PartialEq for FiniteF32 {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for FiniteF32 {}

impl std::hash::Hash for FiniteF32 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// Helper newtype: validated `(n, k)` matmul shape. Construction
/// enforces both dims > 0. Used alongside substrate proofs on
/// `Gemm` and the lm_head Cutlass fusion variants. Not
/// load-bearing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatmulShape {
    pub n: u32,
    pub k: u32,
}

impl MatmulShape {
    pub fn new(n: u32, k: u32) -> Self {
        assert!(n > 0, "MatmulShape: n must be > 0 (got {n})");
        assert!(k > 0, "MatmulShape: k must be > 0 (got {k})");
        Self { n, k }
    }
}

/// Helper newtype: typed sliding-window size. Validates `> 0` —
/// a 0-window degenerates to "attend to nothing" which the
/// attention kernel can't represent. Used alongside substrate
/// proofs on `SlidingAttentionViaCache`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SlidingWindow(u32);

impl SlidingWindow {
    pub fn new(window: u32) -> Self {
        assert!(window > 0, "SlidingWindow must be > 0 (got {window})",);
        Self(window)
    }

    pub fn raw(self) -> u32 {
        self.0
    }
}

// ── Sprint D variants ────────────────────────────────────────

/// `Instruction::Embed(out_slot)`: vocab-table lookup using the
/// `input_ids` extern (per-token id) → `embed_tokens` weight tile.
/// Single-arrive, single-output-page. The embed table is large
/// (vocab × hidden) so loader streams the per-token rows from gmem
/// rather than staging the full table; the substrate proof is just
/// the output page lifecycle plus the weight-table accessor path.
pub struct Embed {
    pub out_page: PageId,
    pub embed_weight_page: PageId,
    pub consumer_phase: MbarrierPhase,
    pub storer_phase: MbarrierPhase,
    pub _loader_role: WarpRoleTag<ROLE_LOADER>,
    pub _launcher_role: WarpRoleTag<ROLE_LAUNCHER>,
    pub _consumer_role: WarpRoleTag<ROLE_CONSUMER>,
    pub _storer_role: WarpRoleTag<ROLE_STORER>,
    pub embed_weight: WeightRef,
}

/// `Instruction::ScalarMul(in_slot, out_slot, scale)`: elementwise
/// `out = in * scale`. Single-arrive. No weight, no scratch — the
/// scalar lives in a register. `scale` validated finite via
/// `FiniteF32`.
pub struct ScalarMul {
    pub in_page: PageId,
    pub out_page: PageId,
    pub consumer_phase: MbarrierPhase,
    pub storer_phase: MbarrierPhase,
    pub _loader_role: WarpRoleTag<ROLE_LOADER>,
    pub _launcher_role: WarpRoleTag<ROLE_LAUNCHER>,
    pub _consumer_role: WarpRoleTag<ROLE_CONSUMER>,
    pub _storer_role: WarpRoleTag<ROLE_STORER>,
    pub scale: FiniteF32,
}

/// `Instruction::TanhSoftCap(in_slot, out_slot)`: elementwise
/// `out = tanh(in / cap) * cap` with the soft-cap value baked into
/// the kernel template (no runtime scalar — Gemma2/3's softcap
/// constants are config-known). Same substrate shape as ScalarMul
/// minus the scalar.
pub struct TanhSoftCap {
    pub in_page: PageId,
    pub out_page: PageId,
    pub consumer_phase: MbarrierPhase,
    pub storer_phase: MbarrierPhase,
    pub _loader_role: WarpRoleTag<ROLE_LOADER>,
    pub _launcher_role: WarpRoleTag<ROLE_LAUNCHER>,
    pub _consumer_role: WarpRoleTag<ROLE_CONSUMER>,
    pub _storer_role: WarpRoleTag<ROLE_STORER>,
}

/// `Instruction::ScalarOffsetRmsNorm(in_slot, out_slot, layer,
/// offset)`: Gemma2 / Gemma3 variant of RmsNorm: weight is `weight
/// + offset * 1.0` instead of `weight`. Substrate shape is
/// identical to RmsNorm — partial-sums reduction tile in
/// `RmsNormScope`, two pages (in, weight), boundary phase parity.
/// `offset` rides as a `FiniteF32` helper.
pub struct ScalarOffsetRmsNorm {
    pub in_page: PageId,
    pub weight_page: PageId,
    pub partial_sums: ScratchRegion<RmsNormScope>,
    pub consumer_phase: MbarrierPhase,
    pub storer_phase: MbarrierPhase,
    pub _loader_role: WarpRoleTag<ROLE_LOADER>,
    pub _launcher_role: WarpRoleTag<ROLE_LAUNCHER>,
    pub _consumer_role: WarpRoleTag<ROLE_CONSUMER>,
    pub _storer_role: WarpRoleTag<ROLE_STORER>,
    pub layer: LayerIndex,
    pub weight: WeightRef,
    pub offset: FiniteF32,
}

/// `Instruction::Gemm(in_slot, out_slot, layer, n, k)`: standalone
/// matmul against a per-layer `LinearLayer` weight. Multi-iter
/// (per-token TMA-loaded B-tiles) — `iters` arrives bumped per call.
/// `b_tile` lives in `GemmScope`.
pub struct Gemm {
    pub in_page: PageId,
    pub weight_page: PageId,
    pub out_page: PageId,
    pub b_tile: ScratchRegion<GemmScope>,
    pub consumer_phase: MbarrierPhase,
    pub storer_phase: MbarrierPhase,
    pub iters: IterCount,
    pub _loader_role: WarpRoleTag<ROLE_LOADER>,
    pub _launcher_role: WarpRoleTag<ROLE_LAUNCHER>,
    pub _consumer_role: WarpRoleTag<ROLE_CONSUMER>,
    pub _storer_role: WarpRoleTag<ROLE_STORER>,
    pub layer: LayerIndex,
    pub weight: WeightRef,
    pub shape: MatmulShape,
}

/// Norm-flavor for the lm_head Cutlass fusion variants — selects
/// the pre-gemm transformation kernel template. Helper field; no
/// substrate consequences.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LmHeadNormKind {
    /// `Instruction::CutlassFusedRmsNormGemm`: rms_norm(in) then
    /// gemm.
    RmsNorm,
    /// `Instruction::CutlassFusedAddRmsNormGemm`: residual fold
    /// then rms_norm then gemm.
    AddRmsNorm,
    /// `Instruction::CutlassFusedAddScalarOffsetRmsNormGemm`:
    /// add then rms_norm with scalar offset then gemm. Carries
    /// the offset alongside in the variant struct.
    AddScalarOffsetRmsNorm,
    /// `Instruction::CutlassFusedMeanSubRmsNormGemm`: mean
    /// subtraction (CommandR-flavored) then rms_norm then gemm.
    MeanSubRmsNorm,
}

/// Unified lm_head Cutlass fusion variant. The four
/// `Instruction::CutlassFused*` variants share substrate shape:
/// 2 weight pages (norm weight + linear weight), 1 input page (or
/// 2 for the residual-folding flavors — delta + residual), 1 output
/// page, partial sums in `RmsNormScope` for the reduction, B-tile
/// in `GemmScope` for the gemm. The norm flavor flag picks the
/// kernel template at emit time.
///
/// `delta_page` is `Some` for `AddRmsNorm` / `AddScalarOffsetRmsNorm`
/// (the residual-fold flavors); `None` for `RmsNorm` / `MeanSubRmsNorm`
/// (the input row IS the residual, no fold).
///
/// `offset` is `Some(FiniteF32)` only for `AddScalarOffsetRmsNorm`.
pub struct CutlassFusedNormGemm {
    pub in_page: PageId,
    pub delta_page: Option<PageId>,
    pub norm_weight_page: PageId,
    pub linear_weight_page: PageId,
    pub out_page: PageId,
    pub partial_sums: ScratchRegion<RmsNormScope>,
    pub b_tile: ScratchRegion<GemmScope>,
    pub consumer_phase: MbarrierPhase,
    pub storer_phase: MbarrierPhase,
    pub iters: IterCount,
    pub _loader_role: WarpRoleTag<ROLE_LOADER>,
    pub _launcher_role: WarpRoleTag<ROLE_LAUNCHER>,
    pub _consumer_role: WarpRoleTag<ROLE_CONSUMER>,
    pub _storer_role: WarpRoleTag<ROLE_STORER>,
    pub layer: LayerIndex,
    pub norm_weight: WeightRef,
    pub linear_weight: WeightRef,
    pub shape: MatmulShape,
    pub norm_kind: LmHeadNormKind,
    pub offset: Option<FiniteF32>,
}

/// Sliding-window kind for the attention variants. Helper field
/// distinguishing `AttentionViaCache` (no window — full causal)
/// from `SlidingAttentionViaCache` (windowed).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttentionKind {
    /// Full causal attention — no window restriction.
    Full,
    /// Windowed causal attention — local-context only. Carries the
    /// `SlidingWindow` size alongside.
    Sliding(SlidingWindow),
}

/// Unified attention-via-paged-cache variant. Both
/// `Instruction::AttentionViaCache` and `SlidingAttentionViaCache`
/// lower here, distinguished by `kind`.
///
/// Substrate proofs:
/// - 2 PageIds: `q_in_page` (input Q rows) and `attn_out_page`
///   (post-softmax output). K/V live in the global paged KV cache
///   which isn't a substrate-managed shmem page — `kv_cache_layer`
///   is the LayerIndex helper recording which layer's cache to
///   read.
/// - 2 disjoint `AttentionScope` scratch tiles: `score_tile` (QKT
///   softmax-input tile) and `pv_tile` (post-softmax PV
///   accumulator).
/// - Boundary phase parity (`consumer_phase`, `storer_phase`) +
///   `iters` (per-token attention iter count, one consumer arrive
///   per iter).
/// - 4 warp role tags.
pub struct AttentionViaCacheNode {
    pub q_in_page: PageId,
    pub attn_out_page: PageId,
    pub score_tile: ScratchRegion<AttentionScope>,
    pub pv_tile: ScratchRegion<AttentionScope>,
    pub consumer_phase: MbarrierPhase,
    pub storer_phase: MbarrierPhase,
    pub iters: IterCount,
    pub _loader_role: WarpRoleTag<ROLE_LOADER>,
    pub _launcher_role: WarpRoleTag<ROLE_LAUNCHER>,
    pub _consumer_role: WarpRoleTag<ROLE_CONSUMER>,
    pub _storer_role: WarpRoleTag<ROLE_STORER>,
    pub kv_cache_layer: LayerIndex,
    pub interleaved: bool,
    pub kind: AttentionKind,
}

/// `Instruction::BarrierSignal(edge_idx)`: producer-side cross-CTA
/// barrier signal. Fires `atomicAdd(&barriers[edge_idx], 1)` from
/// every CTA's storer role.
///
/// Substrate proofs:
/// - `edge`: validated `EdgeId` (bug class #1 against
///   `SubstrateBudget::num_edges`).
/// - `_storer_role`: the role doing the atomic add — pinned at
///   the type level (bug class #6 — only the storer can signal).
///
/// No pages, no scratch, no phases at the per-CTA-shmem level
/// (cross-CTA sync is gmem-bound; the per-CTA mbarrier phases
/// don't apply to the global atomic counter).
pub struct BarrierSignal {
    pub edge: EdgeId,
    pub _storer_role: WarpRoleTag<ROLE_STORER>,
}

/// `Instruction::BarrierWait(edge_idx, expected_count)`:
/// consumer-side cross-CTA barrier wait. Spins on
/// `barriers[edge_idx] >= expected_count` from the loader role
/// before consuming the produced data.
///
/// Substrate proofs:
/// - `edge`: validated `EdgeId` (#1).
/// - `expected`: validated `ExpectedCount` (must be > 0).
/// - `_loader_role`: the role doing the spin (#6).
pub struct BarrierWait {
    pub edge: EdgeId,
    pub expected: ExpectedCount,
    pub _loader_role: WarpRoleTag<ROLE_LOADER>,
}

/// The typed lowered MegaNode enum. One variant per migrated op.
///
/// - Sprint A: `RmsNorm`.
/// - Sprint B: `FusedQkvRopeCache`.
/// - Sprint C: `Add`, `FusedAddRmsNorm`, `FusedGateUp{Silu,Gelu}Mul`
///   (collapsed into `FusedGateUpActivateMul`).
/// - Sprint D: `Embed`, `ScalarMul`, `TanhSoftCap`,
///   `ScalarOffsetRmsNorm`, `Gemm`, `CutlassFused*Gemm`
///   (collapsed into `CutlassFusedNormGemm`),
///   `AttentionViaCache` and `SlidingAttentionViaCache`
///   (collapsed into `AttentionViaCacheNode`),
///   `BarrierSignal`, `BarrierWait`.
pub enum MegaNode {
    RmsNorm(RmsNorm),
    FusedQkvRopeCache(FusedQkvRopeCache),
    Add(Add),
    FusedAddRmsNorm(FusedAddRmsNorm),
    FusedGateUpActivateMul(FusedGateUpActivateMul),
    Embed(Embed),
    ScalarMul(ScalarMul),
    TanhSoftCap(TanhSoftCap),
    ScalarOffsetRmsNorm(ScalarOffsetRmsNorm),
    Gemm(Gemm),
    CutlassFusedNormGemm(CutlassFusedNormGemm),
    AttentionViaCache(AttentionViaCacheNode),
    BarrierSignal(BarrierSignal),
    BarrierWait(BarrierWait),
}
