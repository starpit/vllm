// SPDX-License-Identifier: Apache-2.0
//! Typed lowered-form variants — **const-generic** edition.
//!
//! Each `MegaNode` variant stores plain `u32` fields (post-erasure)
//! but its `new::<const ...>()` constructor takes the load-bearing
//! values as **const generics** and opens a `const {}` block that
//! validates every substrate invariant at MONOMORPHIZATION time.
//! Construct with bad const args → `rustc` E0080 compile error.
//!
//! Helper newtypes (`WeightRef`, `RotaryRef`, `FiniteF32`,
//! `LmHeadNormKind`, `GateUpActivation`, `AttentionKind`,
//! `MatmulShape`, `SlidingWindow`, `LayerIndex`) stay
//! runtime-validated — they're either path strings, floats, or
//! enum tags, which are not numeric primitives ranging over a small
//! typed alphabet. Per `MEGA_IR_PLAN.md` §3, helpers ride alongside
//! substrate-proof load-bearing fields and stay runtime-validated;
//! the substrate-proof fields are the load-bearing ones, and those
//! ARE compile-time-checked.

#![allow(dead_code)]

// Helper-newtype path / float / enum imports come from this crate.

/// Helper newtype: layer index. Now const-generic — compile-fails
/// when `LAYER >= NUM_LAYERS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayerIndex<const LAYER: u32, const NUM_LAYERS: u32>;

impl<const LAYER: u32, const NUM_LAYERS: u32> LayerIndex<LAYER, NUM_LAYERS> {
    pub const fn new() -> Self {
        const {
            assert!(LAYER < NUM_LAYERS, "LayerIndex: LAYER out of range");
        }
        Self
    }

    pub const fn raw(self) -> u32 {
        LAYER
    }

    /// Erase to opaque [`crate::ir::substrate::LayerRef`].
    pub const fn erase(self) -> crate::ir::substrate::LayerRef {
        // Safety: validity discharged in `new`'s `const {}` block.
        // The opaque ref's only public accessor is `raw()`.
        crate::ir::substrate::LayerRef::__new_for_erase(LAYER)
    }
}

impl<const LAYER: u32, const NUM_LAYERS: u32> Default for LayerIndex<LAYER, NUM_LAYERS> {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper newtype: weight accessor path string. Runtime-validated
/// (not a numeric primitive — can't be const-generic on stable Rust).
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

/// Helper newtype: rotary cos/sin cache accessor path. Same shape
/// as `WeightRef`, distinguished by the type so emit-side sites
/// can't mix them up. Runtime-validated.
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

/// Helper newtype: finite f32 (rejects NaN / ±∞). Runtime-validated.
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

/// Helper newtype: validated `(n, k)` matmul shape. Const-generic —
/// compile-fails when either dim is 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatmulShape<const N: u32, const K: u32>;

impl<const N: u32, const K: u32> MatmulShape<N, K> {
    pub const fn new() -> Self {
        const {
            assert!(N > 0, "MatmulShape: N must be > 0");
            assert!(K > 0, "MatmulShape: K must be > 0");
        }
        Self
    }

    pub const fn n(self) -> u32 {
        N
    }
    pub const fn k(self) -> u32 {
        K
    }
}

impl<const N: u32, const K: u32> Default for MatmulShape<N, K> {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper newtype: typed sliding-window size. Const-generic;
/// compile-fails on 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SlidingWindow<const WINDOW: u32>;

impl<const WINDOW: u32> SlidingWindow<WINDOW> {
    pub const fn new() -> Self {
        const {
            assert!(WINDOW > 0, "SlidingWindow: WINDOW must be > 0");
        }
        Self
    }

    pub const fn raw(self) -> u32 {
        WINDOW
    }
}

impl<const WINDOW: u32> Default for SlidingWindow<WINDOW> {
    fn default() -> Self {
        Self::new()
    }
}

/// Activation choice for the gate-up MLP fusion. Helper enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GateUpActivation {
    Silu,
    Gelu,
}

/// Norm-flavor for the lm_head Cutlass fusion. Helper enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LmHeadNormKind {
    RmsNorm,
    AddRmsNorm,
    AddScalarOffsetRmsNorm,
    MeanSubRmsNorm,
}

/// Sliding-window kind for the attention variants. Helper enum.
/// `Sliding` carries a runtime u32 — `SlidingWindow<W>` would force
/// the kind enum itself to be const-generic, which doesn't compose
/// with the variant struct. Window value is validated at builder
/// time via the const-generic primitive then erased here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttentionKind {
    Full,
    Sliding(u32),
}

// ============================================================
// MegaNode variants — store plain u32 fields, constructors take
// const-generic substrate args + run `const {}` blocks.
//
// Each variant has a getter per field so the eventual emit step
// can read the integers without re-validating.
// ============================================================

/// The typed lowered RmsNorm variant. Storage: plain u32s + helper
/// newtypes. Construction via [`RmsNorm::new::<...>`] discharges
/// substrate proofs at compile time.
///
/// ## AST contract (`MEGA_IR_PLAN.md` §0 / §4a / §8.0)
///
/// Codegen inlines the four role bodies (loader / consumer /
/// launcher / storer) directly into the emitted kernel `.cu`,
/// calling TK primitives (`kittens::*`, `ferrite::tk::*`) and the
/// substrate (`ferrite::SharedState`, `ss.pages`, `ss.page_ready`,
/// `ss.page_done`, `ss.scratch`). NO ferrite-owned per-op wrapper
/// in scope — every value spliced into the `.cu` source comes from
/// a typed getter on this variant.
///
/// ## Substrate-proof fields
///
/// `in_page_id`, `weight_page_id`, `partial_offset/bytes`,
/// `consumer_phase`, `storer_phase`, `layer` — discharged in
/// `new::<...>`'s `const {}` block.
///
/// ## Kernel-AST fields
///
/// - `hidden_dim` — `<Config, HIDDEN_DIM, NUM_TOKENS>` template.
/// - `num_tokens` — same.
/// - `eps` — `consumer(..., float eps)` runtime arg.
/// - `in_act_slot` — `act_ptrs[in_act_slot]` (kernel input row).
/// - `out_act_slot` — `act_ptrs[out_act_slot]` (storer output row).
/// - `weight_accessor_idx` — `weight_ptrs[idx * NUM_LAYERS + layer]`.
pub struct TkRmsNorm {
    in_page: crate::ir::substrate::PageRef,
    weight_page: crate::ir::substrate::PageRef,
    partial_offset: crate::ir::substrate::ScratchOffsetRef,
    partial_bytes: crate::ir::substrate::ScratchBytesRef,
    consumer_phase: crate::ir::substrate::MbarrierPhaseRef,
    storer_phase: crate::ir::substrate::MbarrierPhaseRef,
    layer: crate::ir::substrate::LayerRef,
    hidden_dim: crate::ir::substrate::HiddenDimRef,
    num_tokens: crate::ir::substrate::NumTokensRef,
    in_act_slot: crate::ir::substrate::ActSlotRef,
    out_act_slot: crate::ir::substrate::ActSlotRef,
    weight_accessor_idx: crate::ir::substrate::WeightAccessorRef,
    /// Cross-warp `bar.sync` ID for the consumer's sum-of-squares
    /// reduction. PTX `bar.sync` IDs are `[0, 16)`; bar 0 is
    /// reserved for `__syncthreads`. Constructed only via
    /// `BarSyncId<ID>::erase()` — sealed-witness type-check
    /// guarantees ID ∈ 1..=15.
    consumer_bar_reduce: crate::ir::substrate::BarRef,
    /// Cross-warp `bar.sync` ID for the consumer's "all warps wrote
    /// their output slice" publish before warp 0 arrives on the
    /// page_done semaphore.
    consumer_bar_publish: crate::ir::substrate::BarRef,
    /// Witness that `consumer_bar_reduce != consumer_bar_publish`,
    /// constructed only via `BarSyncPair<A, B>::erase()` (sealed
    /// for distinct ordered pairs in 1..=15). Storage-erased: a
    /// zero-sized token whose existence is the proof.
    bar_pair_proof: crate::ir::substrate::DistinctBarPairProof,
    eps: FiniteF32,
    pub weight: WeightRef,
}

impl TkRmsNorm {
    /// Const-generic constructor. Compile-time substrate proofs:
    /// - `IN_ID < NUM_PAGES`, `WEIGHT_ID < NUM_PAGES` (#1)
    /// - `IN_ID != WEIGHT_ID` (within-op alias #2)
    /// - `PARTIAL_OFF + PARTIAL_BYTES <= SCRATCH_BYTES` (#5)
    /// - `LAYER < NUM_LAYERS`
    /// - `CONSUMER_PHASE == ARRIVES & 1` (#3)
    /// - `STORER_PHASE == (ARRIVES + 1) & 1` (#3)
    /// - `HIDDEN_DIM > 0`, `NUM_TOKENS > 0` (kernel-AST shape)
    #[allow(clippy::too_many_arguments)]
    pub fn new<
        const IN_ID: u32,
        const WEIGHT_ID: u32,
        const PARTIAL_OFF: u32,
        const PARTIAL_BYTES: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const LAYER: u32,
        const NUM_PAGES: u32,
        const NUM_LAYERS: u32,
        const SCRATCH_BYTES: u32,
        const ARRIVES: u32,
        const HIDDEN_DIM: u32,
        const NUM_TOKENS: u32,
        const IN_ACT_SLOT: u32,
        const OUT_ACT_SLOT: u32,
        const WEIGHT_ACCESSOR_IDX: u32,
        const CONSUMER_BAR_REDUCE: u32,
        const CONSUMER_BAR_PUBLISH: u32,
    >(
        weight: WeightRef,
        eps: FiniteF32,
    ) -> Self
    where
        // Sealed-witness type-check: `BarSyncId<ID>: IsValidBarSyncId`
        // is implemented only for ID in 1..=15 (bar 0 is
        // `__syncthreads`, IDs >= 16 are out of PTX range).
        // `BarSyncPair<A, B>: IsDistinctBarPair` is implemented only
        // for ordered (A, B) where both are valid AND A != B.
        //
        // If the caller passes BAR_REDUCE = 0, BAR_PUBLISH = 16, or
        // BAR_REDUCE == BAR_PUBLISH, the corresponding bound has no
        // matching impl and the compiler rejects the call at
        // TYPE-CHECK time (E0277), before any monomorphization. No
        // runtime check, no `assert!`, no path that could deadlock.
        crate::ir::substrate::BarSyncId<CONSUMER_BAR_REDUCE>:
            crate::ir::substrate::IsValidBarSyncId,
        crate::ir::substrate::BarSyncId<CONSUMER_BAR_PUBLISH>:
            crate::ir::substrate::IsValidBarSyncId,
        crate::ir::substrate::BarSyncPair<CONSUMER_BAR_REDUCE, CONSUMER_BAR_PUBLISH>:
            crate::ir::substrate::IsDistinctBarPair,
    {
        const {
            assert!(IN_ID < NUM_PAGES, "RmsNorm: IN_ID out of bounds");
            assert!(WEIGHT_ID < NUM_PAGES, "RmsNorm: WEIGHT_ID out of bounds");
            assert!(IN_ID != WEIGHT_ID, "RmsNorm: IN_ID and WEIGHT_ID alias");
            let end = (PARTIAL_OFF as u64) + (PARTIAL_BYTES as u64);
            assert!(
                end <= SCRATCH_BYTES as u64,
                "RmsNorm: partial_sums region out of substrate scratch budget",
            );
            assert!(LAYER < NUM_LAYERS, "RmsNorm: LAYER out of range");
            assert!(
                CONSUMER_PHASE == ARRIVES & 1,
                "RmsNorm: CONSUMER_PHASE parity mismatch with cumulative arrives",
            );
            assert!(
                STORER_PHASE == (ARRIVES + 1) & 1,
                "RmsNorm: STORER_PHASE parity mismatch with cumulative arrives + 1",
            );
            assert!(HIDDEN_DIM > 0, "RmsNorm: HIDDEN_DIM must be > 0");
            assert!(NUM_TOKENS > 0, "RmsNorm: NUM_TOKENS must be > 0");
            // CONSUMER_BAR_REDUCE / CONSUMER_BAR_PUBLISH range +
            // distinctness are enforced by the sealed-witness `where`
            // bounds above — no asserts here.
        }
        // Construct each typed primitive (each runs its substrate
        // proof — e.g. `PageId::new` asserts ID < NUM_PAGES;
        // `BarSyncId::new` requires `IsValidBarSyncId`), then erase
        // to the opaque post-erasure refs stored on the variant.
        // Bare integer literals cannot reach the storage — they
        // travel through `PageId<>` / `BarSyncId<>` / etc., the
        // typed primitives' constructors validate them.
        use crate::ir::substrate::{
            ActSlotConst, BarSyncId, BarSyncPair, HiddenDim, MbarrierPhase, NumTokensConst,
            PageId, ScratchBytesRef, ScratchOffsetRef, WeightAccessorConst,
        };
        let in_page = PageId::<IN_ID, NUM_PAGES>::new().erase();
        let weight_page = PageId::<WEIGHT_ID, NUM_PAGES>::new().erase();
        // Scratch offset/bytes — within-budget proof from the
        // `const {}` block above; the typed `ScratchRegion` would be
        // overkill for a bare offset/bytes pair, so we use the
        // `__new_for_erase` ctor directly with the already-validated
        // values.
        let partial_offset = ScratchOffsetRef::__new_for_erase(PARTIAL_OFF);
        let partial_bytes = ScratchBytesRef::__new_for_erase(PARTIAL_BYTES);
        let consumer_phase = MbarrierPhase::<CONSUMER_PHASE>::new().erase();
        let storer_phase = MbarrierPhase::<STORER_PHASE>::new().erase();
        let layer = LayerIndex::<LAYER, NUM_LAYERS>::new().erase();
        let hidden_dim = HiddenDim::<HIDDEN_DIM>::new().erase();
        let num_tokens = NumTokensConst::<NUM_TOKENS>::new().erase();
        // Activation slots / weight accessor: range checks against
        // a substrate-budget bound that the caller propagates via
        // these const generics. For now we use `IN_ACT_SLOT < u32::MAX`
        // which is trivially true; the real bound flows from the
        // `MegaTapeBuilder` via `NUM_ACT_SLOTS` once that const
        // generic is added (next sprint). Today, ActSlotConst pins
        // ID < NUM_ACT_SLOTS = u32::MAX (no real check); this
        // placeholder retains the typed-ref discipline so storage
        // and getters are typed even before NUM_ACT_SLOTS lands.
        let in_act_slot = ActSlotConst::<IN_ACT_SLOT, { u32::MAX }>::new().erase();
        let out_act_slot = ActSlotConst::<OUT_ACT_SLOT, { u32::MAX }>::new().erase();
        let weight_accessor_idx =
            WeightAccessorConst::<WEIGHT_ACCESSOR_IDX, { u32::MAX }>::new().erase();
        let consumer_bar_reduce = BarSyncId::<CONSUMER_BAR_REDUCE>::new().erase();
        let consumer_bar_publish = BarSyncId::<CONSUMER_BAR_PUBLISH>::new().erase();
        let bar_pair_proof =
            BarSyncPair::<CONSUMER_BAR_REDUCE, CONSUMER_BAR_PUBLISH>::new().erase();
        Self {
            in_page,
            weight_page,
            partial_offset,
            partial_bytes,
            consumer_phase,
            storer_phase,
            layer,
            hidden_dim,
            num_tokens,
            in_act_slot,
            out_act_slot,
            weight_accessor_idx,
            consumer_bar_reduce,
            consumer_bar_publish,
            bar_pair_proof,
            eps,
            weight,
        }
    }

    pub const fn in_page(&self) -> crate::ir::substrate::PageRef {
        self.in_page
    }
    pub const fn weight_page(&self) -> crate::ir::substrate::PageRef {
        self.weight_page
    }
    pub const fn partial_offset(&self) -> crate::ir::substrate::ScratchOffsetRef {
        self.partial_offset
    }
    pub const fn partial_bytes(&self) -> crate::ir::substrate::ScratchBytesRef {
        self.partial_bytes
    }
    pub const fn consumer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.storer_phase
    }
    pub const fn layer(&self) -> crate::ir::substrate::LayerRef {
        self.layer
    }
    /// Kernel `<Config, HIDDEN_DIM, NUM_TOKENS>` template arg.
    pub const fn hidden_dim(&self) -> crate::ir::substrate::HiddenDimRef {
        self.hidden_dim
    }
    /// Kernel `<Config, HIDDEN_DIM, NUM_TOKENS>` template arg.
    pub const fn num_tokens(&self) -> crate::ir::substrate::NumTokensRef {
        self.num_tokens
    }
    /// `act_ptrs[in_act_slot]` — kernel input row gmem ptr.
    pub const fn in_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.in_act_slot
    }
    /// `act_ptrs[out_act_slot]` — kernel output row gmem ptr (storer).
    pub const fn out_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.out_act_slot
    }
    /// `weight_ptrs[weight_accessor_idx * NUM_LAYERS + layer]` —
    /// flat-table index for the per-layer rms weight.
    pub const fn weight_accessor_idx(&self) -> crate::ir::substrate::WeightAccessorRef {
        self.weight_accessor_idx
    }
    /// `bar.sync` ID for the consumer's sum-of-squares reduction.
    /// Validity (1..=15) was discharged by sealed-witness type-check
    /// at construction. `BarRef` cannot be constructed from raw `u32`
    /// outside this crate.
    pub const fn consumer_bar_reduce(&self) -> crate::ir::substrate::BarRef {
        self.consumer_bar_reduce
    }
    /// `bar.sync` ID for the consumer's "all warps wrote their
    /// output slice" publish before warp 0 arrives on page_done.
    pub const fn consumer_bar_publish(&self) -> crate::ir::substrate::BarRef {
        self.consumer_bar_publish
    }
    /// Witness that `consumer_bar_reduce != consumer_bar_publish`.
    /// Existence of the value IS the proof.
    pub const fn bar_pair_proof(&self) -> crate::ir::substrate::DistinctBarPairProof {
        self.bar_pair_proof
    }
    /// Kernel `consumer(..., float eps)` runtime arg.
    pub fn eps(&self) -> FiniteF32 {
        self.eps
    }
}

/// The typed lowered FusedQkvRopeCache variant.
///
/// Codegen inlines the four role bodies directly into the kernel
/// `.cu`, calling TK + substrate primitives. Kernel-shape template
/// args (HIDDEN_DIM, HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS, BIASED,
/// INTERLEAVED) come from typed getters; runtime args (positions,
/// cos_sin_cache, KV cache pages, slot_mapping) come from the
/// per-variant `KernelExtras` flag set propagated to the kernel
/// signature.
///
/// Sprint 15a (S15a) IR ext: `num_tokens` (M dim) + `tile_n` +
/// `chunk_k` (mirror Gemm S9 / TkFusedGemmAdd S11a / TkFusedNormGemm
/// S13a) cover the AlongN + per-iter-K layout for the QKV linear
/// projection. `consumer_bar_publish` (mirror FusedGateUpActivateMul
/// S12a) covers the cross-warp publish before warp 0 arrives on
/// `page_done` — only a single bar (no cross-warp reduction; the
/// per-warp [M, TILE_N] mma + per-row RoPE rotates each warp's
/// disjoint output cols independently).
///
/// Sprint 15c (S15c) IR ext: `qkv_b_tile_offset` /
/// `qkv_b_tile_bytes` carry the GemmScope scratch region for staging
/// the QKV linear weight `[K, qkv_n]` as the mma_AB B operand.
/// Mirrors `TkFusedNormGemm::b_tile_offset` / `Gemm::b_tile_offset`.
/// `qkv_weight_page` was load-bearing for the page-handoff barrier
/// only; the actual weight tile lives in scratch (the page is too
/// small — typical `PAGE_SIZE=32KB` cannot fit a multi-MB weight).
pub struct TkFusedQkvRopeCache {
    in_page: crate::ir::substrate::PageRef,
    qkv_weight_page: crate::ir::substrate::PageRef,
    cos_sin_page: crate::ir::substrate::PageRef,
    q_out_page: crate::ir::substrate::PageRef,
    k_out_page: crate::ir::substrate::PageRef,
    v_out_page: crate::ir::substrate::PageRef,
    q_rope_offset: crate::ir::substrate::ScratchOffsetRef,
    q_rope_bytes: crate::ir::substrate::ScratchBytesRef,
    k_rope_offset: crate::ir::substrate::ScratchOffsetRef,
    k_rope_bytes: crate::ir::substrate::ScratchBytesRef,
    /// QKV linear weight b_tile staging region in `GemmScope` scratch.
    /// Sized for one `[chunk_k, qkv_n] * sizeof(bf16)` tile (per-iter
    /// b_tile when `iters > 1`; full weight when `iters == 1`).
    /// Mirrors `TkFusedNormGemm::b_tile_offset` (S13a). Disjoint from
    /// `q_rope` / `k_rope` by the scope tag (RopeScope vs GemmScope —
    /// no within-scope overlap proof needed).
    qkv_b_tile_offset: crate::ir::substrate::ScratchOffsetRef,
    qkv_b_tile_bytes: crate::ir::substrate::ScratchBytesRef,
    consumer_phase: crate::ir::substrate::MbarrierPhaseRef,
    storer_phase: crate::ir::substrate::MbarrierPhaseRef,
    iters: crate::ir::substrate::IterCountRef,
    layer: crate::ir::substrate::LayerRef,
    hidden_dim: crate::ir::substrate::HiddenDimRef,
    head_dim: crate::ir::substrate::HeadDimRef,
    num_q_heads: crate::ir::substrate::NumQHeadsRef,
    num_kv_heads: crate::ir::substrate::NumKvHeadsRef,
    num_tokens: crate::ir::substrate::NumTokensRef,
    /// Per-warp output N slice for the QKV linear projection
    /// (AlongN warp split). Mirrors `Gemm::tile_n` /
    /// `TkFusedGemmAdd::tile_n` / `FusedGateUpActivateMul::tile_n`.
    /// Equality with `qkv_n / NCW` (where `qkv_n =
    /// (num_q_heads + 2 * num_kv_heads) * head_dim`) is enforced at
    /// emit time (NCW isn't a const generic on the builder).
    tile_n: crate::ir::substrate::TileNRef,
    /// Per-iter K-chunk width for the b_tile. Mirrors
    /// `Gemm::chunk_k` / `TkFusedGemmAdd::chunk_k` /
    /// `TkFusedNormGemm::chunk_k`. `chunk_k * iters == hidden_dim`
    /// is enforced at construction.
    chunk_k: crate::ir::substrate::ChunkKRef,
    in_act_slot: crate::ir::substrate::ActSlotRef,
    q_out_act_slot: crate::ir::substrate::ActSlotRef,
    k_out_act_slot: crate::ir::substrate::ActSlotRef,
    v_out_act_slot: crate::ir::substrate::ActSlotRef,
    qkv_weight_accessor_idx: crate::ir::substrate::WeightAccessorRef,
    rotary_accessor_idx: crate::ir::substrate::WeightAccessorRef,
    /// Cross-warp `bar.sync` ID for the consumer's "all warps wrote
    /// their `[M, TILE_N]` matmul + RoPE-rotated output slice"
    /// publish before warp 0 arrives on `page_done[q_out|k_out|v_out]`.
    /// Mirrors `FusedGateUpActivateMul::consumer_bar_publish` (S12a).
    /// Validity (1..=15) is enforced at construction by `BarSyncId<ID>:
    /// IsValidBarSyncId`. AlongN split → no cross-warp reduction over
    /// the tile, so only one bar is needed.
    consumer_bar_publish: crate::ir::substrate::BarRef,
    pub qkv_weight: WeightRef,
    pub rotary: RotaryRef,
    pub biased: bool,
    pub interleaved: bool,
}

impl TkFusedQkvRopeCache {
    /// Const-generic constructor with all substrate proofs at
    /// compile time. Six page bounds, six pairwise non-aliases,
    /// two scratch within-budget, two scratch disjoint, two phase
    /// parities, layer in range, iters > 0.
    ///
    /// Kernel-AST const generics (per `MEGA_IR_PLAN.md` §0/§4a):
    /// HIDDEN_DIM, HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS — kernel
    /// template args. IN_ACT_SLOT, Q_OUT_ACT_SLOT, K_OUT_ACT_SLOT,
    /// V_OUT_ACT_SLOT, QKV_WEIGHT_ACCESSOR_IDX,
    /// ROTARY_ACCESSOR_IDX — host-slot indices for `act_ptrs[]` /
    /// `weight_ptrs[]` / cos-sin gmem ptr.
    #[allow(clippy::too_many_arguments)]
    pub fn new<
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
        const B_TILE_OFF: u32,
        const B_TILE_BYTES: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const ITERS: u32,
        const LAYER: u32,
        const NUM_PAGES: u32,
        const NUM_LAYERS: u32,
        const SCRATCH_BYTES: u32,
        const ARRIVES: u32,
        const HIDDEN_DIM: u32,
        const HEAD_DIM: u32,
        const NUM_Q_HEADS: u32,
        const NUM_KV_HEADS: u32,
        const NUM_TOKENS: u32,
        const IN_ACT_SLOT: u32,
        const Q_OUT_ACT_SLOT: u32,
        const K_OUT_ACT_SLOT: u32,
        const V_OUT_ACT_SLOT: u32,
        const QKV_WEIGHT_ACCESSOR_IDX: u32,
        const ROTARY_ACCESSOR_IDX: u32,
        const TILE_N: u32,
        const CHUNK_K: u32,
        const CONSUMER_BAR_PUBLISH: u32,
    >(
        qkv_weight: WeightRef,
        rotary: RotaryRef,
        biased: bool,
        interleaved: bool,
    ) -> Self
    where
        // Sealed-witness: BAR_PUBLISH ∈ 1..=15. Mirrors S12a precedent.
        crate::ir::substrate::BarSyncId<CONSUMER_BAR_PUBLISH>:
            crate::ir::substrate::IsValidBarSyncId,
    {
        const {
            assert!(IN_ID < NUM_PAGES, "FusedQkvRopeCache: IN_ID out of bounds");
            assert!(
                QKV_ID < NUM_PAGES,
                "FusedQkvRopeCache: QKV_ID out of bounds"
            );
            assert!(
                COS_SIN_ID < NUM_PAGES,
                "FusedQkvRopeCache: COS_SIN_ID out of bounds"
            );
            assert!(Q_ID < NUM_PAGES, "FusedQkvRopeCache: Q_ID out of bounds");
            assert!(K_ID < NUM_PAGES, "FusedQkvRopeCache: K_ID out of bounds");
            assert!(V_ID < NUM_PAGES, "FusedQkvRopeCache: V_ID out of bounds");

            // Pairwise non-alias check across all six pages. With 6
            // values that's 15 pairs; we list them out so the
            // compile error names which pair collides.
            assert!(
                IN_ID != QKV_ID
                    && IN_ID != COS_SIN_ID
                    && IN_ID != Q_ID
                    && IN_ID != K_ID
                    && IN_ID != V_ID,
                "FusedQkvRopeCache: IN_ID aliases another page"
            );
            assert!(
                QKV_ID != COS_SIN_ID && QKV_ID != Q_ID && QKV_ID != K_ID && QKV_ID != V_ID,
                "FusedQkvRopeCache: QKV_ID aliases another page"
            );
            assert!(
                COS_SIN_ID != Q_ID && COS_SIN_ID != K_ID && COS_SIN_ID != V_ID,
                "FusedQkvRopeCache: COS_SIN_ID aliases another page"
            );
            assert!(
                Q_ID != K_ID && Q_ID != V_ID,
                "FusedQkvRopeCache: Q_ID aliases another page"
            );
            assert!(K_ID != V_ID, "FusedQkvRopeCache: K_ID and V_ID alias");

            // Scratch within-budget.
            let q_end = (Q_OFF as u64) + (Q_BYTES as u64);
            let k_end = (K_OFF as u64) + (K_BYTES as u64);
            let b_end = (B_TILE_OFF as u64) + (B_TILE_BYTES as u64);
            assert!(
                q_end <= SCRATCH_BYTES as u64,
                "FusedQkvRopeCache: Q rope buf out of scratch budget"
            );
            assert!(
                k_end <= SCRATCH_BYTES as u64,
                "FusedQkvRopeCache: K rope buf out of scratch budget"
            );
            assert!(
                b_end <= SCRATCH_BYTES as u64,
                "FusedQkvRopeCache: qkv b_tile out of scratch budget"
            );
            // Scratch disjoint within RopeScope (Q vs K rope bufs).
            // The qkv b_tile lives in GemmScope, so cross-scope
            // disjointness is by typed-tag, no offset proof needed.
            assert!(
                q_end <= K_OFF as u64 || k_end <= Q_OFF as u64,
                "FusedQkvRopeCache: Q and K rope bufs overlap within RopeScope"
            );
            assert!(B_TILE_BYTES > 0, "FusedQkvRopeCache: B_TILE_BYTES must be > 0");

            assert!(ITERS > 0, "FusedQkvRopeCache: ITERS must be > 0");
            assert!(LAYER < NUM_LAYERS, "FusedQkvRopeCache: LAYER out of range");
            assert!(
                CONSUMER_PHASE == ARRIVES & 1,
                "FusedQkvRopeCache: CONSUMER_PHASE parity mismatch"
            );
            assert!(
                STORER_PHASE == (ARRIVES + 1) & 1,
                "FusedQkvRopeCache: STORER_PHASE parity mismatch"
            );
            assert!(HIDDEN_DIM > 0, "FusedQkvRopeCache: HIDDEN_DIM must be > 0");
            assert!(HEAD_DIM > 0, "FusedQkvRopeCache: HEAD_DIM must be > 0");
            assert!(
                NUM_Q_HEADS > 0,
                "FusedQkvRopeCache: NUM_Q_HEADS must be > 0"
            );
            assert!(
                NUM_KV_HEADS > 0,
                "FusedQkvRopeCache: NUM_KV_HEADS must be > 0"
            );
            assert!(
                NUM_TOKENS > 0,
                "FusedQkvRopeCache: NUM_TOKENS must be > 0"
            );
            // Tile-layout invariants (mirror Gemm S9 / TkFusedGemmAdd
            // S11a / TkFusedNormGemm S13a). AlongN warp split: each
            // consumer warp covers TILE_N output cols. tile_n * NCW
            // == qkv_n is enforced at emit time (NCW + qkv_n aren't
            // const generics here). tile_n > 0 / chunk_k > 0 are.
            assert!(TILE_N > 0, "FusedQkvRopeCache: TILE_N must be > 0");
            assert!(CHUNK_K > 0, "FusedQkvRopeCache: CHUNK_K must be > 0");
            // Per-iter K coverage: chunk_k * iters == hidden_dim.
            assert!(
                CHUNK_K * ITERS == HIDDEN_DIM,
                "FusedQkvRopeCache: CHUNK_K * ITERS must equal HIDDEN_DIM"
            );
        }
        // Each typed primitive's `new()` runs its substrate proof
        // (PageId<>: ID < NUM_PAGES; MbarrierPhase<>: P <= 1; etc.);
        // erase to opaque refs for storage. Bare `u32` cannot reach
        // the IR — they flow through typed primitive constructors.
        use crate::ir::substrate::{
            ActSlotConst, BarSyncId, ChunkK, HeadDim, HiddenDim, IterCount, MbarrierPhase,
            NumKvHeads, NumQHeads, NumTokensConst, PageId, ScratchBytesRef, ScratchOffsetRef,
            TileN, WeightAccessorConst,
        };
        Self {
            in_page: PageId::<IN_ID, NUM_PAGES>::new().erase(),
            qkv_weight_page: PageId::<QKV_ID, NUM_PAGES>::new().erase(),
            cos_sin_page: PageId::<COS_SIN_ID, NUM_PAGES>::new().erase(),
            q_out_page: PageId::<Q_ID, NUM_PAGES>::new().erase(),
            k_out_page: PageId::<K_ID, NUM_PAGES>::new().erase(),
            v_out_page: PageId::<V_ID, NUM_PAGES>::new().erase(),
            // Scratch offset/bytes — within-budget proof from the
            // const{} block above; ScratchRegion would be one
            // typed primitive but FusedQkvRopeCache uses two
            // disjoint scratch regions (Q rope + K rope), each its
            // own offset + bytes. The `__new_for_erase` ctor stores
            // the already-validated values via crate-private path.
            q_rope_offset: ScratchOffsetRef::__new_for_erase(Q_OFF),
            q_rope_bytes: ScratchBytesRef::__new_for_erase(Q_BYTES),
            k_rope_offset: ScratchOffsetRef::__new_for_erase(K_OFF),
            k_rope_bytes: ScratchBytesRef::__new_for_erase(K_BYTES),
            qkv_b_tile_offset: ScratchOffsetRef::__new_for_erase(B_TILE_OFF),
            qkv_b_tile_bytes: ScratchBytesRef::__new_for_erase(B_TILE_BYTES),
            consumer_phase: MbarrierPhase::<CONSUMER_PHASE>::new().erase(),
            storer_phase: MbarrierPhase::<STORER_PHASE>::new().erase(),
            iters: IterCount::<ITERS>::new().erase(),
            layer: LayerIndex::<LAYER, NUM_LAYERS>::new().erase(),
            hidden_dim: HiddenDim::<HIDDEN_DIM>::new().erase(),
            head_dim: HeadDim::<HEAD_DIM>::new().erase(),
            num_q_heads: NumQHeads::<NUM_Q_HEADS>::new().erase(),
            num_kv_heads: NumKvHeads::<NUM_KV_HEADS>::new().erase(),
            num_tokens: NumTokensConst::<NUM_TOKENS>::new().erase(),
            tile_n: TileN::<TILE_N>::new().erase(),
            chunk_k: ChunkK::<CHUNK_K>::new().erase(),
            in_act_slot: ActSlotConst::<IN_ACT_SLOT, { u32::MAX }>::new().erase(),
            q_out_act_slot: ActSlotConst::<Q_OUT_ACT_SLOT, { u32::MAX }>::new().erase(),
            k_out_act_slot: ActSlotConst::<K_OUT_ACT_SLOT, { u32::MAX }>::new().erase(),
            v_out_act_slot: ActSlotConst::<V_OUT_ACT_SLOT, { u32::MAX }>::new().erase(),
            qkv_weight_accessor_idx: WeightAccessorConst::<
                QKV_WEIGHT_ACCESSOR_IDX,
                { u32::MAX },
            >::new()
            .erase(),
            rotary_accessor_idx: WeightAccessorConst::<ROTARY_ACCESSOR_IDX, { u32::MAX }>::new()
                .erase(),
            consumer_bar_publish: BarSyncId::<CONSUMER_BAR_PUBLISH>::new().erase(),
            qkv_weight,
            rotary,
            biased,
            interleaved,
        }
    }

    pub const fn in_page(&self) -> crate::ir::substrate::PageRef {
        self.in_page
    }
    pub const fn qkv_weight_page(&self) -> crate::ir::substrate::PageRef {
        self.qkv_weight_page
    }
    pub const fn cos_sin_page(&self) -> crate::ir::substrate::PageRef {
        self.cos_sin_page
    }
    pub const fn q_out_page(&self) -> crate::ir::substrate::PageRef {
        self.q_out_page
    }
    pub const fn k_out_page(&self) -> crate::ir::substrate::PageRef {
        self.k_out_page
    }
    pub const fn v_out_page(&self) -> crate::ir::substrate::PageRef {
        self.v_out_page
    }
    pub const fn q_rope_offset(&self) -> crate::ir::substrate::ScratchOffsetRef {
        self.q_rope_offset
    }
    pub const fn q_rope_bytes(&self) -> crate::ir::substrate::ScratchBytesRef {
        self.q_rope_bytes
    }
    pub const fn k_rope_offset(&self) -> crate::ir::substrate::ScratchOffsetRef {
        self.k_rope_offset
    }
    pub const fn k_rope_bytes(&self) -> crate::ir::substrate::ScratchBytesRef {
        self.k_rope_bytes
    }
    /// QKV b_tile staging region offset (GemmScope). The mma_AB B
    /// operand is staged here as `[K, qkv_n] bf16` (or `[CHUNK_K,
    /// qkv_n]` per iter once iters > 1). Mirrors
    /// `TkFusedNormGemm::b_tile_offset` (S13a).
    pub const fn qkv_b_tile_offset(&self) -> crate::ir::substrate::ScratchOffsetRef {
        self.qkv_b_tile_offset
    }
    pub const fn qkv_b_tile_bytes(&self) -> crate::ir::substrate::ScratchBytesRef {
        self.qkv_b_tile_bytes
    }
    pub const fn consumer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.storer_phase
    }
    pub const fn iters(&self) -> crate::ir::substrate::IterCountRef {
        self.iters
    }
    pub const fn layer(&self) -> crate::ir::substrate::LayerRef {
        self.layer
    }
    pub const fn hidden_dim(&self) -> crate::ir::substrate::HiddenDimRef {
        self.hidden_dim
    }
    pub const fn head_dim(&self) -> crate::ir::substrate::HeadDimRef {
        self.head_dim
    }
    pub const fn num_q_heads(&self) -> crate::ir::substrate::NumQHeadsRef {
        self.num_q_heads
    }
    pub const fn num_kv_heads(&self) -> crate::ir::substrate::NumKvHeadsRef {
        self.num_kv_heads
    }
    pub const fn num_tokens(&self) -> crate::ir::substrate::NumTokensRef {
        self.num_tokens
    }
    pub const fn tile_n(&self) -> crate::ir::substrate::TileNRef {
        self.tile_n
    }
    pub const fn chunk_k(&self) -> crate::ir::substrate::ChunkKRef {
        self.chunk_k
    }
    pub const fn in_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.in_act_slot
    }
    pub const fn q_out_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.q_out_act_slot
    }
    pub const fn k_out_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.k_out_act_slot
    }
    pub const fn v_out_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.v_out_act_slot
    }
    pub const fn qkv_weight_accessor_idx(&self) -> crate::ir::substrate::WeightAccessorRef {
        self.qkv_weight_accessor_idx
    }
    pub const fn rotary_accessor_idx(&self) -> crate::ir::substrate::WeightAccessorRef {
        self.rotary_accessor_idx
    }
    /// `bar.sync` ID for the consumer's "all warps wrote their
    /// `[M, TILE_N]` matmul + RoPE-rotated output slice" publish
    /// before warp 0 arrives on `page_done`. Validity (1..=15) was
    /// discharged by sealed-witness type-check at construction.
    pub const fn consumer_bar_publish(&self) -> crate::ir::substrate::BarRef {
        self.consumer_bar_publish
    }
}

/// The typed lowered `Add` (residual fold) variant.
///
/// Kernel ABI: bf16 elementwise per-row residual add — emit splices a
/// per-row load/add/store loop with `<HIDDEN_DIM, NUM_TOKENS>` shape.
pub struct TkAdd {
    delta_page: crate::ir::substrate::PageRef,
    residual_page: crate::ir::substrate::PageRef,
    consumer_phase: crate::ir::substrate::MbarrierPhaseRef,
    storer_phase: crate::ir::substrate::MbarrierPhaseRef,
    hidden_dim: crate::ir::substrate::HiddenDimRef,
    num_tokens: crate::ir::substrate::NumTokensRef,
    delta_act_slot: crate::ir::substrate::ActSlotRef,
    residual_act_slot: crate::ir::substrate::ActSlotRef,
    /// Cross-warp `bar.sync` ID for the consumer's "all warps wrote
    /// their output slice" publish before warp 0 arrives on
    /// `page_done`. Type-checked in 1..=15 by `BarSyncId`.
    consumer_bar_publish: crate::ir::substrate::BarRef,
}

impl TkAdd {
    #[allow(clippy::too_many_arguments)]
    pub fn new<
        const DELTA_ID: u32,
        const RESIDUAL_ID: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const NUM_PAGES: u32,
        const ARRIVES: u32,
        const HIDDEN_DIM: u32,
        const NUM_TOKENS: u32,
        const DELTA_ACT_SLOT: u32,
        const RESIDUAL_ACT_SLOT: u32,
        const CONSUMER_BAR_PUBLISH: u32,
    >() -> Self
    where
        crate::ir::substrate::BarSyncId<CONSUMER_BAR_PUBLISH>: crate::ir::substrate::IsValidBarSyncId,
    {
        const {
            // Cross-field invariants — page non-alias, phase parity.
            // Stable Rust can't enumerate sealed witnesses for these
            // (page non-alias would need `NUM_PAGES * (NUM_PAGES-1)`
            // impls; phase parity depends on unbounded ARRIVES).
            assert!(
                DELTA_ID != RESIDUAL_ID,
                "Add: DELTA_ID and RESIDUAL_ID alias"
            );
            assert!(
                CONSUMER_PHASE == ARRIVES & 1,
                "Add: CONSUMER_PHASE parity mismatch"
            );
            assert!(
                STORER_PHASE == (ARRIVES + 1) & 1,
                "Add: STORER_PHASE parity mismatch"
            );
        }
        use crate::ir::substrate::{
            ActSlotConst, BarSyncId, HiddenDim, MbarrierPhase, NumTokensConst, PageId,
        };
        Self {
            delta_page: PageId::<DELTA_ID, NUM_PAGES>::new().erase(),
            residual_page: PageId::<RESIDUAL_ID, NUM_PAGES>::new().erase(),
            consumer_phase: MbarrierPhase::<CONSUMER_PHASE>::new().erase(),
            storer_phase: MbarrierPhase::<STORER_PHASE>::new().erase(),
            hidden_dim: HiddenDim::<HIDDEN_DIM>::new().erase(),
            num_tokens: NumTokensConst::<NUM_TOKENS>::new().erase(),
            delta_act_slot: ActSlotConst::<DELTA_ACT_SLOT, { u32::MAX }>::new().erase(),
            residual_act_slot: ActSlotConst::<RESIDUAL_ACT_SLOT, { u32::MAX }>::new().erase(),
            consumer_bar_publish: BarSyncId::<CONSUMER_BAR_PUBLISH>::new().erase(),
        }
    }

    pub const fn delta_page(&self) -> crate::ir::substrate::PageRef {
        self.delta_page
    }
    pub const fn residual_page(&self) -> crate::ir::substrate::PageRef {
        self.residual_page
    }
    pub const fn consumer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.storer_phase
    }
    pub const fn hidden_dim(&self) -> crate::ir::substrate::HiddenDimRef {
        self.hidden_dim
    }
    pub const fn num_tokens(&self) -> crate::ir::substrate::NumTokensRef {
        self.num_tokens
    }
    pub const fn delta_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.delta_act_slot
    }
    pub const fn residual_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.residual_act_slot
    }
    pub const fn consumer_bar_publish(&self) -> crate::ir::substrate::BarRef {
        self.consumer_bar_publish
    }
}

/// The typed lowered `FusedAddRmsNorm` variant.
///
/// Codegen inlines the four role bodies directly into the kernel
/// `.cu`. Template args `<HIDDEN_DIM, NUM_TOKENS>` come from typed
/// getters; runtime arg `eps` from `eps()`.
pub struct TkFusedAddRmsNorm {
    delta_page: crate::ir::substrate::PageRef,
    residual_page: crate::ir::substrate::PageRef,
    weight_page: crate::ir::substrate::PageRef,
    partial_offset: crate::ir::substrate::ScratchOffsetRef,
    partial_bytes: crate::ir::substrate::ScratchBytesRef,
    consumer_phase: crate::ir::substrate::MbarrierPhaseRef,
    storer_phase: crate::ir::substrate::MbarrierPhaseRef,
    layer: crate::ir::substrate::LayerRef,
    hidden_dim: crate::ir::substrate::HiddenDimRef,
    num_tokens: crate::ir::substrate::NumTokensRef,
    delta_act_slot: crate::ir::substrate::ActSlotRef,
    residual_act_slot: crate::ir::substrate::ActSlotRef,
    weight_accessor_idx: crate::ir::substrate::WeightAccessorRef,
    /// Cross-warp `bar.sync` ID for the consumer's sum-of-squares
    /// reduction across consumer warps. Type-checked in 1..=15.
    consumer_bar_reduce: crate::ir::substrate::BarRef,
    /// Cross-warp `bar.sync` ID for the consumer's "all warps wrote
    /// their output slice" publish before warp 0 arrives on
    /// `page_done`. Distinct from `consumer_bar_reduce`.
    consumer_bar_publish: crate::ir::substrate::BarRef,
    /// Witness that `consumer_bar_reduce != consumer_bar_publish`.
    /// Storage-erased zero-sized token whose existence is the proof.
    bar_pair_proof: crate::ir::substrate::DistinctBarPairProof,
    eps: FiniteF32,
    pub weight: WeightRef,
}

impl TkFusedAddRmsNorm {
    #[allow(clippy::too_many_arguments)]
    pub fn new<
        const DELTA_ID: u32,
        const RESIDUAL_ID: u32,
        const WEIGHT_ID: u32,
        const PARTIAL_OFF: u32,
        const PARTIAL_BYTES: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const LAYER: u32,
        const NUM_PAGES: u32,
        const NUM_LAYERS: u32,
        const SCRATCH_BYTES: u32,
        const ARRIVES: u32,
        const HIDDEN_DIM: u32,
        const NUM_TOKENS: u32,
        const DELTA_ACT_SLOT: u32,
        const RESIDUAL_ACT_SLOT: u32,
        const WEIGHT_ACCESSOR_IDX: u32,
        const CONSUMER_BAR_REDUCE: u32,
        const CONSUMER_BAR_PUBLISH: u32,
    >(
        weight: WeightRef,
        eps: FiniteF32,
    ) -> Self
    where
        crate::ir::substrate::BarSyncId<CONSUMER_BAR_REDUCE>: crate::ir::substrate::IsValidBarSyncId,
        crate::ir::substrate::BarSyncId<CONSUMER_BAR_PUBLISH>: crate::ir::substrate::IsValidBarSyncId,
        crate::ir::substrate::BarSyncPair<CONSUMER_BAR_REDUCE, CONSUMER_BAR_PUBLISH>:
            crate::ir::substrate::IsDistinctBarPair,
    {
        const {
            assert!(DELTA_ID < NUM_PAGES, "FusedAddRmsNorm: DELTA_ID OOB");
            assert!(RESIDUAL_ID < NUM_PAGES, "FusedAddRmsNorm: RESIDUAL_ID OOB");
            assert!(WEIGHT_ID < NUM_PAGES, "FusedAddRmsNorm: WEIGHT_ID OOB");
            assert!(
                DELTA_ID != RESIDUAL_ID && DELTA_ID != WEIGHT_ID && RESIDUAL_ID != WEIGHT_ID,
                "FusedAddRmsNorm: page alias",
            );
            let end = (PARTIAL_OFF as u64) + (PARTIAL_BYTES as u64);
            assert!(
                end <= SCRATCH_BYTES as u64,
                "FusedAddRmsNorm: partial_sums OOB scratch budget",
            );
            assert!(LAYER < NUM_LAYERS, "FusedAddRmsNorm: LAYER OOB");
            assert!(
                CONSUMER_PHASE == ARRIVES & 1,
                "FusedAddRmsNorm: CONSUMER_PHASE parity mismatch",
            );
            assert!(
                STORER_PHASE == (ARRIVES + 1) & 1,
                "FusedAddRmsNorm: STORER_PHASE parity mismatch",
            );
            assert!(HIDDEN_DIM > 0, "FusedAddRmsNorm: HIDDEN_DIM must be > 0");
            assert!(NUM_TOKENS > 0, "FusedAddRmsNorm: NUM_TOKENS must be > 0");
        }
        use crate::ir::substrate::{
            ActSlotConst, BarSyncId, BarSyncPair, HiddenDim, MbarrierPhase, NumTokensConst,
            PageId, ScratchBytesRef, ScratchOffsetRef, WeightAccessorConst,
        };
        Self {
            delta_page: PageId::<DELTA_ID, NUM_PAGES>::new().erase(),
            residual_page: PageId::<RESIDUAL_ID, NUM_PAGES>::new().erase(),
            weight_page: PageId::<WEIGHT_ID, NUM_PAGES>::new().erase(),
            partial_offset: ScratchOffsetRef::__new_for_erase(PARTIAL_OFF),
            partial_bytes: ScratchBytesRef::__new_for_erase(PARTIAL_BYTES),
            consumer_phase: MbarrierPhase::<CONSUMER_PHASE>::new().erase(),
            storer_phase: MbarrierPhase::<STORER_PHASE>::new().erase(),
            layer: LayerIndex::<LAYER, NUM_LAYERS>::new().erase(),
            hidden_dim: HiddenDim::<HIDDEN_DIM>::new().erase(),
            num_tokens: NumTokensConst::<NUM_TOKENS>::new().erase(),
            delta_act_slot: ActSlotConst::<DELTA_ACT_SLOT, { u32::MAX }>::new().erase(),
            residual_act_slot: ActSlotConst::<RESIDUAL_ACT_SLOT, { u32::MAX }>::new().erase(),
            weight_accessor_idx: WeightAccessorConst::<WEIGHT_ACCESSOR_IDX, { u32::MAX }>::new()
                .erase(),
            consumer_bar_reduce: BarSyncId::<CONSUMER_BAR_REDUCE>::new().erase(),
            consumer_bar_publish: BarSyncId::<CONSUMER_BAR_PUBLISH>::new().erase(),
            bar_pair_proof: BarSyncPair::<CONSUMER_BAR_REDUCE, CONSUMER_BAR_PUBLISH>::new()
                .erase(),
            eps,
            weight,
        }
    }

    pub const fn delta_page(&self) -> crate::ir::substrate::PageRef {
        self.delta_page
    }
    pub const fn residual_page(&self) -> crate::ir::substrate::PageRef {
        self.residual_page
    }
    pub const fn weight_page(&self) -> crate::ir::substrate::PageRef {
        self.weight_page
    }
    pub const fn partial_offset(&self) -> crate::ir::substrate::ScratchOffsetRef {
        self.partial_offset
    }
    pub const fn partial_bytes(&self) -> crate::ir::substrate::ScratchBytesRef {
        self.partial_bytes
    }
    pub const fn consumer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.storer_phase
    }
    pub const fn layer(&self) -> crate::ir::substrate::LayerRef {
        self.layer
    }
    pub const fn hidden_dim(&self) -> crate::ir::substrate::HiddenDimRef {
        self.hidden_dim
    }
    pub const fn num_tokens(&self) -> crate::ir::substrate::NumTokensRef {
        self.num_tokens
    }
    pub const fn delta_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.delta_act_slot
    }
    pub const fn residual_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.residual_act_slot
    }
    pub const fn weight_accessor_idx(&self) -> crate::ir::substrate::WeightAccessorRef {
        self.weight_accessor_idx
    }
    pub const fn consumer_bar_reduce(&self) -> crate::ir::substrate::BarRef {
        self.consumer_bar_reduce
    }
    pub const fn consumer_bar_publish(&self) -> crate::ir::substrate::BarRef {
        self.consumer_bar_publish
    }
    pub const fn bar_pair_proof(&self) -> crate::ir::substrate::DistinctBarPairProof {
        self.bar_pair_proof
    }
    pub fn eps(&self) -> FiniteF32 {
        self.eps
    }
}

/// The typed lowered `FusedGateUp{Silu,Gelu}Mul` variant.
///
/// Codegen inlines the role bodies directly. Template args
/// `<HIDDEN_DIM, INTERMEDIATE_DIM, NUM_TOKENS>` come from typed
/// getters; activation enum (`GateUpActivation::{Silu,Gelu}`)
/// selects which TK helper sequence the codegen emits.
///
/// Sprint 12a (S12a) IR ext: `tile_n` and `consumer_bar_publish`
/// added to mirror the S10a (Gemm) / S11a (TkFusedGemmAdd)
/// pattern. The AlongN warp split needs `tile_n * NCW ==
/// intermediate_dim`; the consumer's cross-warp publish before
/// `page_done[out]` needs a `bar.sync` ID in 1..=15.
pub struct TkFusedGateUpActivateMul {
    in_page: crate::ir::substrate::PageRef,
    gate_up_weight_page: crate::ir::substrate::PageRef,
    out_page: crate::ir::substrate::PageRef,
    gate_offset: crate::ir::substrate::ScratchOffsetRef,
    gate_bytes: crate::ir::substrate::ScratchBytesRef,
    up_offset: crate::ir::substrate::ScratchOffsetRef,
    up_bytes: crate::ir::substrate::ScratchBytesRef,
    consumer_phase: crate::ir::substrate::MbarrierPhaseRef,
    storer_phase: crate::ir::substrate::MbarrierPhaseRef,
    iters: crate::ir::substrate::IterCountRef,
    layer: crate::ir::substrate::LayerRef,
    hidden_dim: crate::ir::substrate::HiddenDimRef,
    intermediate_dim: crate::ir::substrate::IntermediateDimRef,
    num_tokens: crate::ir::substrate::NumTokensRef,
    /// Per-warp output N slice (AlongN warp split). Mirrors the
    /// `Gemm::tile_n` field added in Sprint 9. Equality with
    /// `intermediate_dim / NCW` is enforced at emit time (NCW
    /// isn't a const generic on the builder).
    tile_n: crate::ir::substrate::TileNRef,
    in_act_slot: crate::ir::substrate::ActSlotRef,
    out_act_slot: crate::ir::substrate::ActSlotRef,
    weight_accessor_idx: crate::ir::substrate::WeightAccessorRef,
    /// Cross-warp `bar.sync` ID for the consumer's "all warps
    /// wrote their `[M, TILE_N]` accumulator slice into out_smem"
    /// publish before warp 0 arrives on `page_done[out]`. Mirrors
    /// `Gemm::consumer_bar_publish` (Sprint 10a). Validity
    /// (1..=15) is enforced at construction by `BarSyncId<ID>:
    /// IsValidBarSyncId`. AlongN split → no cross-warp reduction
    /// over the tile, so only one bar is needed.
    consumer_bar_publish: crate::ir::substrate::BarRef,
    pub weight: WeightRef,
    pub activation: GateUpActivation,
}

impl TkFusedGateUpActivateMul {
    #[allow(clippy::too_many_arguments)]
    pub fn new<
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
        const NUM_PAGES: u32,
        const NUM_LAYERS: u32,
        const SCRATCH_BYTES: u32,
        const ARRIVES: u32,
        const HIDDEN_DIM: u32,
        const INTERMEDIATE_DIM: u32,
        const NUM_TOKENS: u32,
        const IN_ACT_SLOT: u32,
        const OUT_ACT_SLOT: u32,
        const WEIGHT_ACCESSOR_IDX: u32,
        const TILE_N: u32,
        const CONSUMER_BAR_PUBLISH: u32,
    >(
        weight: WeightRef,
        activation: GateUpActivation,
    ) -> Self
    where
        // Sealed-witness: BAR_PUBLISH ∈ 1..=15. Mirrors the
        // S10a/S11a precedent (`BarSyncId` validity check).
        crate::ir::substrate::BarSyncId<CONSUMER_BAR_PUBLISH>:
            crate::ir::substrate::IsValidBarSyncId,
    {
        const {
            assert!(IN_ID < NUM_PAGES, "FusedGateUp: IN_ID OOB");
            assert!(WEIGHT_ID < NUM_PAGES, "FusedGateUp: WEIGHT_ID OOB");
            assert!(OUT_ID < NUM_PAGES, "FusedGateUp: OUT_ID OOB");
            assert!(
                IN_ID != WEIGHT_ID && IN_ID != OUT_ID && WEIGHT_ID != OUT_ID,
                "FusedGateUp: page alias",
            );
            let g_end = (GATE_OFF as u64) + (GATE_BYTES as u64);
            let u_end = (UP_OFF as u64) + (UP_BYTES as u64);
            assert!(
                g_end <= SCRATCH_BYTES as u64,
                "FusedGateUp: gate_buf OOB scratch budget"
            );
            assert!(
                u_end <= SCRATCH_BYTES as u64,
                "FusedGateUp: up_buf OOB scratch budget"
            );
            assert!(
                g_end <= UP_OFF as u64 || u_end <= GATE_OFF as u64,
                "FusedGateUp: gate_buf and up_buf overlap within MlpScope"
            );
            assert!(ITERS > 0, "FusedGateUp: ITERS must be > 0");
            assert!(LAYER < NUM_LAYERS, "FusedGateUp: LAYER OOB");
            assert!(
                CONSUMER_PHASE == ARRIVES & 1,
                "FusedGateUp: CONSUMER_PHASE parity mismatch",
            );
            assert!(
                STORER_PHASE == (ARRIVES + 1) & 1,
                "FusedGateUp: STORER_PHASE parity mismatch",
            );
            assert!(HIDDEN_DIM > 0, "FusedGateUp: HIDDEN_DIM must be > 0");
            assert!(
                INTERMEDIATE_DIM > 0,
                "FusedGateUp: INTERMEDIATE_DIM must be > 0"
            );
            assert!(NUM_TOKENS > 0, "FusedGateUp: NUM_TOKENS must be > 0");
            // Tile-layout invariant (mirror of Gemm Sprint 9).
            // AlongN warp split: each consumer warp covers TILE_N
            // output cols. tile_n * NCW equality with
            // intermediate_dim is enforced by the proc-macro
            // (NCW isn't a const generic here); tile_n > 0 is.
            assert!(TILE_N > 0, "FusedGateUp: TILE_N must be > 0");
        }
        use crate::ir::substrate::{
            ActSlotConst, BarSyncId, HiddenDim, IntermediateDim, IterCount, MbarrierPhase,
            NumTokensConst, PageId, ScratchBytesRef, ScratchOffsetRef, TileN,
            WeightAccessorConst,
        };
        Self {
            in_page: PageId::<IN_ID, NUM_PAGES>::new().erase(),
            gate_up_weight_page: PageId::<WEIGHT_ID, NUM_PAGES>::new().erase(),
            out_page: PageId::<OUT_ID, NUM_PAGES>::new().erase(),
            gate_offset: ScratchOffsetRef::__new_for_erase(GATE_OFF),
            gate_bytes: ScratchBytesRef::__new_for_erase(GATE_BYTES),
            up_offset: ScratchOffsetRef::__new_for_erase(UP_OFF),
            up_bytes: ScratchBytesRef::__new_for_erase(UP_BYTES),
            consumer_phase: MbarrierPhase::<CONSUMER_PHASE>::new().erase(),
            storer_phase: MbarrierPhase::<STORER_PHASE>::new().erase(),
            iters: IterCount::<ITERS>::new().erase(),
            layer: LayerIndex::<LAYER, NUM_LAYERS>::new().erase(),
            hidden_dim: HiddenDim::<HIDDEN_DIM>::new().erase(),
            intermediate_dim: IntermediateDim::<INTERMEDIATE_DIM>::new().erase(),
            num_tokens: NumTokensConst::<NUM_TOKENS>::new().erase(),
            tile_n: TileN::<TILE_N>::new().erase(),
            in_act_slot: ActSlotConst::<IN_ACT_SLOT, { u32::MAX }>::new().erase(),
            out_act_slot: ActSlotConst::<OUT_ACT_SLOT, { u32::MAX }>::new().erase(),
            weight_accessor_idx: WeightAccessorConst::<WEIGHT_ACCESSOR_IDX, { u32::MAX }>::new()
                .erase(),
            consumer_bar_publish: BarSyncId::<CONSUMER_BAR_PUBLISH>::new().erase(),
            weight,
            activation,
        }
    }

    pub const fn in_page(&self) -> crate::ir::substrate::PageRef {
        self.in_page
    }
    pub const fn gate_up_weight_page(&self) -> crate::ir::substrate::PageRef {
        self.gate_up_weight_page
    }
    pub const fn out_page(&self) -> crate::ir::substrate::PageRef {
        self.out_page
    }
    pub const fn gate_offset(&self) -> crate::ir::substrate::ScratchOffsetRef {
        self.gate_offset
    }
    pub const fn gate_bytes(&self) -> crate::ir::substrate::ScratchBytesRef {
        self.gate_bytes
    }
    pub const fn up_offset(&self) -> crate::ir::substrate::ScratchOffsetRef {
        self.up_offset
    }
    pub const fn up_bytes(&self) -> crate::ir::substrate::ScratchBytesRef {
        self.up_bytes
    }
    pub const fn consumer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.storer_phase
    }
    pub const fn iters(&self) -> crate::ir::substrate::IterCountRef {
        self.iters
    }
    pub const fn layer(&self) -> crate::ir::substrate::LayerRef {
        self.layer
    }
    pub const fn hidden_dim(&self) -> crate::ir::substrate::HiddenDimRef {
        self.hidden_dim
    }
    pub const fn intermediate_dim(&self) -> crate::ir::substrate::IntermediateDimRef {
        self.intermediate_dim
    }
    pub const fn num_tokens(&self) -> crate::ir::substrate::NumTokensRef {
        self.num_tokens
    }
    pub const fn in_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.in_act_slot
    }
    pub const fn out_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.out_act_slot
    }
    pub const fn weight_accessor_idx(&self) -> crate::ir::substrate::WeightAccessorRef {
        self.weight_accessor_idx
    }
    pub const fn tile_n(&self) -> crate::ir::substrate::TileNRef {
        self.tile_n
    }
    /// `bar.sync` ID for the consumer's "all warps wrote their
    /// `[M, TILE_N]` accumulator slice" publish before warp 0
    /// arrives on `page_done[out]`. Validity (1..=15) was
    /// discharged by sealed-witness type-check at construction.
    pub const fn consumer_bar_publish(&self) -> crate::ir::substrate::BarRef {
        self.consumer_bar_publish
    }
}

/// `Embed` (vocab table lookup) variant.
///
/// Codegen inlines the role bodies; loader pulls one row of
/// `weight_ptrs[weight_accessor_idx * NUM_LAYERS + 0]` per token
/// (LAYER is always 0 for Embed). The vocab table is sized
/// `VOCAB_SIZE × HIDDEN_DIM`. Loader needs `input_ids` (uint32_t*)
/// as a kernel-level extra ptr (see `KernelExtras::needs_input_ids`).
pub struct TkEmbed {
    out_page: crate::ir::substrate::PageRef,
    embed_weight_page: crate::ir::substrate::PageRef,
    consumer_phase: crate::ir::substrate::MbarrierPhaseRef,
    storer_phase: crate::ir::substrate::MbarrierPhaseRef,
    hidden_dim: crate::ir::substrate::HiddenDimRef,
    num_tokens: crate::ir::substrate::NumTokensRef,
    vocab_size: crate::ir::substrate::VocabSizeRef,
    out_act_slot: crate::ir::substrate::ActSlotRef,
    weight_accessor_idx: crate::ir::substrate::WeightAccessorRef,
    pub embed_weight: WeightRef,
}

impl TkEmbed {
    #[allow(clippy::too_many_arguments)]
    pub fn new<
        const OUT_ID: u32,
        const WEIGHT_ID: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const NUM_PAGES: u32,
        const ARRIVES: u32,
        const HIDDEN_DIM: u32,
        const NUM_TOKENS: u32,
        const VOCAB_SIZE: u32,
        const OUT_ACT_SLOT: u32,
        const WEIGHT_ACCESSOR_IDX: u32,
    >(
        embed_weight: WeightRef,
    ) -> Self {
        const {
            assert!(OUT_ID != WEIGHT_ID, "Embed: page alias");
            assert!(
                CONSUMER_PHASE == ARRIVES & 1,
                "Embed: CONSUMER_PHASE parity"
            );
            assert!(
                STORER_PHASE == (ARRIVES + 1) & 1,
                "Embed: STORER_PHASE parity"
            );
        }
        use crate::ir::substrate::{
            ActSlotConst, HiddenDim, MbarrierPhase, NumTokensConst, PageId, VocabSize,
            WeightAccessorConst,
        };
        Self {
            out_page: PageId::<OUT_ID, NUM_PAGES>::new().erase(),
            embed_weight_page: PageId::<WEIGHT_ID, NUM_PAGES>::new().erase(),
            consumer_phase: MbarrierPhase::<CONSUMER_PHASE>::new().erase(),
            storer_phase: MbarrierPhase::<STORER_PHASE>::new().erase(),
            hidden_dim: HiddenDim::<HIDDEN_DIM>::new().erase(),
            num_tokens: NumTokensConst::<NUM_TOKENS>::new().erase(),
            vocab_size: VocabSize::<VOCAB_SIZE>::new().erase(),
            out_act_slot: ActSlotConst::<OUT_ACT_SLOT, { u32::MAX }>::new().erase(),
            weight_accessor_idx: WeightAccessorConst::<WEIGHT_ACCESSOR_IDX, { u32::MAX }>::new()
                .erase(),
            embed_weight,
        }
    }

    pub const fn out_page(&self) -> crate::ir::substrate::PageRef {
        self.out_page
    }
    pub const fn embed_weight_page(&self) -> crate::ir::substrate::PageRef {
        self.embed_weight_page
    }
    pub const fn consumer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.storer_phase
    }
    pub const fn hidden_dim(&self) -> crate::ir::substrate::HiddenDimRef {
        self.hidden_dim
    }
    pub const fn num_tokens(&self) -> crate::ir::substrate::NumTokensRef {
        self.num_tokens
    }
    pub const fn vocab_size(&self) -> crate::ir::substrate::VocabSizeRef {
        self.vocab_size
    }
    pub const fn out_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.out_act_slot
    }
    pub const fn weight_accessor_idx(&self) -> crate::ir::substrate::WeightAccessorRef {
        self.weight_accessor_idx
    }
}

/// `ScalarMul` variant.
///
/// Kernel ABI: bf16 elementwise per-row scale — emit splices a per-row
/// load/mul/store loop with `<HIDDEN_DIM, NUM_TOKENS>` shape.
pub struct TkScalarMul {
    in_page: crate::ir::substrate::PageRef,
    out_page: crate::ir::substrate::PageRef,
    consumer_phase: crate::ir::substrate::MbarrierPhaseRef,
    storer_phase: crate::ir::substrate::MbarrierPhaseRef,
    hidden_dim: crate::ir::substrate::HiddenDimRef,
    num_tokens: crate::ir::substrate::NumTokensRef,
    in_act_slot: crate::ir::substrate::ActSlotRef,
    out_act_slot: crate::ir::substrate::ActSlotRef,
    /// Cross-warp `bar.sync` ID for the consumer's "all warps wrote
    /// their output slice" publish before warp 0 arrives on
    /// `page_done`. Sealed-witness BarSyncId in 1..=15.
    consumer_bar_publish: crate::ir::substrate::BarRef,
    pub scale: FiniteF32,
}

impl TkScalarMul {
    #[allow(clippy::too_many_arguments)]
    pub fn new<
        const IN_ID: u32,
        const OUT_ID: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const NUM_PAGES: u32,
        const ARRIVES: u32,
        const HIDDEN_DIM: u32,
        const NUM_TOKENS: u32,
        const IN_ACT_SLOT: u32,
        const OUT_ACT_SLOT: u32,
        const CONSUMER_BAR_PUBLISH: u32,
    >(
        scale: FiniteF32,
    ) -> Self
    where
        crate::ir::substrate::BarSyncId<CONSUMER_BAR_PUBLISH>: crate::ir::substrate::IsValidBarSyncId,
    {
        const {
            // ScalarMul is elementwise; in-place (IN_ID == OUT_ID) is
            // valid (gemma2 post-attn `* hidden`).
            assert!(
                CONSUMER_PHASE == ARRIVES & 1,
                "ScalarMul: CONSUMER_PHASE parity"
            );
            assert!(
                STORER_PHASE == (ARRIVES + 1) & 1,
                "ScalarMul: STORER_PHASE parity"
            );
        }
        use crate::ir::substrate::{
            ActSlotConst, BarSyncId, HiddenDim, MbarrierPhase, NumTokensConst, PageId,
        };
        Self {
            in_page: PageId::<IN_ID, NUM_PAGES>::new().erase(),
            out_page: PageId::<OUT_ID, NUM_PAGES>::new().erase(),
            consumer_phase: MbarrierPhase::<CONSUMER_PHASE>::new().erase(),
            storer_phase: MbarrierPhase::<STORER_PHASE>::new().erase(),
            hidden_dim: HiddenDim::<HIDDEN_DIM>::new().erase(),
            num_tokens: NumTokensConst::<NUM_TOKENS>::new().erase(),
            in_act_slot: ActSlotConst::<IN_ACT_SLOT, { u32::MAX }>::new().erase(),
            out_act_slot: ActSlotConst::<OUT_ACT_SLOT, { u32::MAX }>::new().erase(),
            consumer_bar_publish: BarSyncId::<CONSUMER_BAR_PUBLISH>::new().erase(),
            scale,
        }
    }

    pub const fn in_page(&self) -> crate::ir::substrate::PageRef {
        self.in_page
    }
    pub const fn out_page(&self) -> crate::ir::substrate::PageRef {
        self.out_page
    }
    pub const fn consumer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.storer_phase
    }
    pub const fn hidden_dim(&self) -> crate::ir::substrate::HiddenDimRef {
        self.hidden_dim
    }
    pub const fn num_tokens(&self) -> crate::ir::substrate::NumTokensRef {
        self.num_tokens
    }
    pub const fn in_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.in_act_slot
    }
    pub const fn out_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.out_act_slot
    }
    pub const fn consumer_bar_publish(&self) -> crate::ir::substrate::BarRef {
        self.consumer_bar_publish
    }
}

/// `TanhSoftCap` variant. Same shape as `ScalarMul` plus a runtime
/// `cap` value. Kernel: emit splices a per-row load/tanh-cap/store
/// loop with `<HIDDEN_DIM, NUM_TOKENS>` shape and the runtime cap
/// (Gemma2 final-logit softcap; 0.0 = identity for arches without).
pub struct TkTanhSoftCap {
    in_page: crate::ir::substrate::PageRef,
    out_page: crate::ir::substrate::PageRef,
    consumer_phase: crate::ir::substrate::MbarrierPhaseRef,
    storer_phase: crate::ir::substrate::MbarrierPhaseRef,
    hidden_dim: crate::ir::substrate::HiddenDimRef,
    num_tokens: crate::ir::substrate::NumTokensRef,
    in_act_slot: crate::ir::substrate::ActSlotRef,
    out_act_slot: crate::ir::substrate::ActSlotRef,
    /// Cross-warp `bar.sync` ID for the consumer publish before
    /// warp 0 arrives on `page_done`. Sealed-witness 1..=15.
    consumer_bar_publish: crate::ir::substrate::BarRef,
    pub cap: FiniteF32,
}

impl TkTanhSoftCap {
    #[allow(clippy::too_many_arguments)]
    pub fn new<
        const IN_ID: u32,
        const OUT_ID: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const NUM_PAGES: u32,
        const ARRIVES: u32,
        const HIDDEN_DIM: u32,
        const NUM_TOKENS: u32,
        const IN_ACT_SLOT: u32,
        const OUT_ACT_SLOT: u32,
        const CONSUMER_BAR_PUBLISH: u32,
    >(
        cap: FiniteF32,
    ) -> Self
    where
        crate::ir::substrate::BarSyncId<CONSUMER_BAR_PUBLISH>: crate::ir::substrate::IsValidBarSyncId,
    {
        const {
            // TanhSoftCap is elementwise; in-place (IN_ID == OUT_ID)
            // is valid (gemma2 final logit cap).
            assert!(
                CONSUMER_PHASE == ARRIVES & 1,
                "TanhSoftCap: CONSUMER_PHASE parity"
            );
            assert!(
                STORER_PHASE == (ARRIVES + 1) & 1,
                "TanhSoftCap: STORER_PHASE parity"
            );
        }
        use crate::ir::substrate::{
            ActSlotConst, BarSyncId, HiddenDim, MbarrierPhase, NumTokensConst, PageId,
        };
        Self {
            in_page: PageId::<IN_ID, NUM_PAGES>::new().erase(),
            out_page: PageId::<OUT_ID, NUM_PAGES>::new().erase(),
            consumer_phase: MbarrierPhase::<CONSUMER_PHASE>::new().erase(),
            storer_phase: MbarrierPhase::<STORER_PHASE>::new().erase(),
            hidden_dim: HiddenDim::<HIDDEN_DIM>::new().erase(),
            num_tokens: NumTokensConst::<NUM_TOKENS>::new().erase(),
            in_act_slot: ActSlotConst::<IN_ACT_SLOT, { u32::MAX }>::new().erase(),
            out_act_slot: ActSlotConst::<OUT_ACT_SLOT, { u32::MAX }>::new().erase(),
            consumer_bar_publish: BarSyncId::<CONSUMER_BAR_PUBLISH>::new().erase(),
            cap,
        }
    }

    pub const fn in_page(&self) -> crate::ir::substrate::PageRef {
        self.in_page
    }
    pub const fn out_page(&self) -> crate::ir::substrate::PageRef {
        self.out_page
    }
    pub const fn consumer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.storer_phase
    }
    pub const fn hidden_dim(&self) -> crate::ir::substrate::HiddenDimRef {
        self.hidden_dim
    }
    pub const fn num_tokens(&self) -> crate::ir::substrate::NumTokensRef {
        self.num_tokens
    }
    pub const fn in_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.in_act_slot
    }
    pub const fn out_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.out_act_slot
    }
    pub const fn consumer_bar_publish(&self) -> crate::ir::substrate::BarRef {
        self.consumer_bar_publish
    }
}

/// `ScalarOffsetRmsNorm` variant.
///
/// Codegen inlines the role bodies. Same TK + substrate primitives
/// as RmsNorm plus a `float offset` runtime arg in the consumer's
/// scale-multiply step.
pub struct TkScalarOffsetRmsNorm {
    in_page: crate::ir::substrate::PageRef,
    weight_page: crate::ir::substrate::PageRef,
    partial_offset: crate::ir::substrate::ScratchOffsetRef,
    partial_bytes: crate::ir::substrate::ScratchBytesRef,
    consumer_phase: crate::ir::substrate::MbarrierPhaseRef,
    storer_phase: crate::ir::substrate::MbarrierPhaseRef,
    layer: crate::ir::substrate::LayerRef,
    hidden_dim: crate::ir::substrate::HiddenDimRef,
    num_tokens: crate::ir::substrate::NumTokensRef,
    in_act_slot: crate::ir::substrate::ActSlotRef,
    out_act_slot: crate::ir::substrate::ActSlotRef,
    weight_accessor_idx: crate::ir::substrate::WeightAccessorRef,
    /// Cross-warp `bar.sync` ID for the consumer's RMS sum-of-squares
    /// reduction. Sealed-witness type-checked in 1..=15.
    consumer_bar_reduce: crate::ir::substrate::BarRef,
    /// Cross-warp `bar.sync` ID for the consumer's publish before
    /// warp 0 arrives on `page_done`. Distinct from
    /// `consumer_bar_reduce`.
    consumer_bar_publish: crate::ir::substrate::BarRef,
    /// Witness that the two BAR IDs are distinct.
    bar_pair_proof: crate::ir::substrate::DistinctBarPairProof,
    eps: FiniteF32,
    pub weight: WeightRef,
    pub offset: FiniteF32,
}

impl TkScalarOffsetRmsNorm {
    #[allow(clippy::too_many_arguments)]
    pub fn new<
        const IN_ID: u32,
        const WEIGHT_ID: u32,
        const PARTIAL_OFF: u32,
        const PARTIAL_BYTES: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const LAYER: u32,
        const NUM_PAGES: u32,
        const NUM_LAYERS: u32,
        const SCRATCH_BYTES: u32,
        const ARRIVES: u32,
        const HIDDEN_DIM: u32,
        const NUM_TOKENS: u32,
        const IN_ACT_SLOT: u32,
        const OUT_ACT_SLOT: u32,
        const WEIGHT_ACCESSOR_IDX: u32,
        const CONSUMER_BAR_REDUCE: u32,
        const CONSUMER_BAR_PUBLISH: u32,
    >(
        weight: WeightRef,
        offset: FiniteF32,
        eps: FiniteF32,
    ) -> Self
    where
        crate::ir::substrate::BarSyncId<CONSUMER_BAR_REDUCE>: crate::ir::substrate::IsValidBarSyncId,
        crate::ir::substrate::BarSyncId<CONSUMER_BAR_PUBLISH>: crate::ir::substrate::IsValidBarSyncId,
        crate::ir::substrate::BarSyncPair<CONSUMER_BAR_REDUCE, CONSUMER_BAR_PUBLISH>:
            crate::ir::substrate::IsDistinctBarPair,
    {
        const {
            assert!(IN_ID < NUM_PAGES, "ScalarOffsetRmsNorm: IN_ID OOB");
            assert!(WEIGHT_ID < NUM_PAGES, "ScalarOffsetRmsNorm: WEIGHT_ID OOB");
            assert!(IN_ID != WEIGHT_ID, "ScalarOffsetRmsNorm: page alias");
            let end = (PARTIAL_OFF as u64) + (PARTIAL_BYTES as u64);
            assert!(
                end <= SCRATCH_BYTES as u64,
                "ScalarOffsetRmsNorm: partial_sums OOB"
            );
            assert!(LAYER < NUM_LAYERS, "ScalarOffsetRmsNorm: LAYER OOB");
            assert!(
                CONSUMER_PHASE == ARRIVES & 1,
                "ScalarOffsetRmsNorm: CONSUMER_PHASE parity"
            );
            assert!(
                STORER_PHASE == (ARRIVES + 1) & 1,
                "ScalarOffsetRmsNorm: STORER_PHASE parity"
            );
            assert!(
                HIDDEN_DIM > 0,
                "ScalarOffsetRmsNorm: HIDDEN_DIM must be > 0"
            );
            assert!(
                NUM_TOKENS > 0,
                "ScalarOffsetRmsNorm: NUM_TOKENS must be > 0"
            );
        }
        use crate::ir::substrate::{
            ActSlotConst, BarSyncId, BarSyncPair, HiddenDim, MbarrierPhase, NumTokensConst,
            PageId, ScratchBytesRef, ScratchOffsetRef, WeightAccessorConst,
        };
        Self {
            in_page: PageId::<IN_ID, NUM_PAGES>::new().erase(),
            weight_page: PageId::<WEIGHT_ID, NUM_PAGES>::new().erase(),
            partial_offset: ScratchOffsetRef::__new_for_erase(PARTIAL_OFF),
            partial_bytes: ScratchBytesRef::__new_for_erase(PARTIAL_BYTES),
            consumer_phase: MbarrierPhase::<CONSUMER_PHASE>::new().erase(),
            storer_phase: MbarrierPhase::<STORER_PHASE>::new().erase(),
            layer: LayerIndex::<LAYER, NUM_LAYERS>::new().erase(),
            hidden_dim: HiddenDim::<HIDDEN_DIM>::new().erase(),
            num_tokens: NumTokensConst::<NUM_TOKENS>::new().erase(),
            in_act_slot: ActSlotConst::<IN_ACT_SLOT, { u32::MAX }>::new().erase(),
            out_act_slot: ActSlotConst::<OUT_ACT_SLOT, { u32::MAX }>::new().erase(),
            weight_accessor_idx: WeightAccessorConst::<WEIGHT_ACCESSOR_IDX, { u32::MAX }>::new()
                .erase(),
            consumer_bar_reduce: BarSyncId::<CONSUMER_BAR_REDUCE>::new().erase(),
            consumer_bar_publish: BarSyncId::<CONSUMER_BAR_PUBLISH>::new().erase(),
            bar_pair_proof: BarSyncPair::<CONSUMER_BAR_REDUCE, CONSUMER_BAR_PUBLISH>::new()
                .erase(),
            eps,
            weight,
            offset,
        }
    }

    pub const fn in_page(&self) -> crate::ir::substrate::PageRef {
        self.in_page
    }
    pub const fn weight_page(&self) -> crate::ir::substrate::PageRef {
        self.weight_page
    }
    pub const fn partial_offset(&self) -> crate::ir::substrate::ScratchOffsetRef {
        self.partial_offset
    }
    pub const fn partial_bytes(&self) -> crate::ir::substrate::ScratchBytesRef {
        self.partial_bytes
    }
    pub const fn consumer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.storer_phase
    }
    pub const fn layer(&self) -> crate::ir::substrate::LayerRef {
        self.layer
    }
    pub const fn hidden_dim(&self) -> crate::ir::substrate::HiddenDimRef {
        self.hidden_dim
    }
    pub const fn num_tokens(&self) -> crate::ir::substrate::NumTokensRef {
        self.num_tokens
    }
    pub const fn in_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.in_act_slot
    }
    pub const fn out_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.out_act_slot
    }
    pub const fn weight_accessor_idx(&self) -> crate::ir::substrate::WeightAccessorRef {
        self.weight_accessor_idx
    }
    pub const fn consumer_bar_reduce(&self) -> crate::ir::substrate::BarRef {
        self.consumer_bar_reduce
    }
    pub const fn consumer_bar_publish(&self) -> crate::ir::substrate::BarRef {
        self.consumer_bar_publish
    }
    pub const fn bar_pair_proof(&self) -> crate::ir::substrate::DistinctBarPairProof {
        self.bar_pair_proof
    }
    pub fn eps(&self) -> FiniteF32 {
        self.eps
    }
}

/// `Gemm` variant. Storage erases `(n, k)` to plain u32 fields.
///
/// Codegen inlines the role bodies. Template `<K, N, M>` with
/// M = NUM_TOKENS at the canonical's workload point. The consumer's
/// inner-product loop uses TK `wgmma`/`mma_ABt` primitives (Hopper)
/// or warp-level register tiles (Ampere).
pub struct TkGemm {
    in_page: crate::ir::substrate::PageRef,
    weight_page: crate::ir::substrate::PageRef,
    out_page: crate::ir::substrate::PageRef,
    b_tile_offset: crate::ir::substrate::ScratchOffsetRef,
    b_tile_bytes: crate::ir::substrate::ScratchBytesRef,
    consumer_phase: crate::ir::substrate::MbarrierPhaseRef,
    storer_phase: crate::ir::substrate::MbarrierPhaseRef,
    iters: crate::ir::substrate::IterCountRef,
    layer: crate::ir::substrate::LayerRef,
    n: crate::ir::substrate::MatmulNRef,
    k: crate::ir::substrate::MatmulKRef,
    m: crate::ir::substrate::MatmulMRef,
    /// Per-warp output N slice. AlongN warp split convention: each
    /// consumer warp covers all M rows of `tile_n` output cols.
    /// Required `tile_n * NCW == n` (proc-macro emits consistent
    /// values; substrate-level enforcement is a future sprint).
    tile_n: crate::ir::substrate::TileNRef,
    /// Per-iter K-chunk width loaded into the b_tile. Required
    /// `chunk_k * iters == k` AND
    /// `b_tile_bytes == chunk_k * n * sizeof(bf16)`.
    chunk_k: crate::ir::substrate::ChunkKRef,
    in_act_slot: crate::ir::substrate::ActSlotRef,
    out_act_slot: crate::ir::substrate::ActSlotRef,
    weight_accessor_idx: crate::ir::substrate::WeightAccessorRef,
    /// Cross-warp `bar.sync` ID for the consumer's "all warps wrote
    /// their `[M, TILE_N]` output slice" publish before warp 0
    /// arrives on `page_done[out_page]`. Validity (1..=15) is
    /// enforced at construction by `BarSyncId<ID>:
    /// IsValidBarSyncId`. Gemm has no cross-warp reduction (each
    /// warp owns disjoint output cols under the AlongN split), so
    /// only one bar is needed.
    consumer_bar_publish: crate::ir::substrate::BarRef,
    pub weight: WeightRef,
}

impl TkGemm {
    #[allow(clippy::too_many_arguments)]
    pub fn new<
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
        const NUM_PAGES: u32,
        const NUM_LAYERS: u32,
        const SCRATCH_BYTES: u32,
        const ARRIVES: u32,
        const M: u32,
        const IN_ACT_SLOT: u32,
        const OUT_ACT_SLOT: u32,
        const WEIGHT_ACCESSOR_IDX: u32,
        const TILE_N: u32,
        const CHUNK_K: u32,
        const CONSUMER_BAR_PUBLISH: u32,
    >(
        weight: WeightRef,
    ) -> Self
    where
        // Sealed-witness: BAR_PUBLISH ∈ 1..=15. ID 0 is reserved
        // for `__syncthreads`; 16+ is out of PTX range. The where
        // bound has no matching impl outside that range, so the
        // call is rejected at type-check time.
        crate::ir::substrate::BarSyncId<CONSUMER_BAR_PUBLISH>:
            crate::ir::substrate::IsValidBarSyncId,
    {
        const {
            assert!(IN_ID < NUM_PAGES, "Gemm: IN_ID OOB");
            assert!(WEIGHT_ID < NUM_PAGES, "Gemm: WEIGHT_ID OOB");
            assert!(OUT_ID < NUM_PAGES, "Gemm: OUT_ID OOB");
            assert!(
                IN_ID != WEIGHT_ID && IN_ID != OUT_ID && WEIGHT_ID != OUT_ID,
                "Gemm: page alias"
            );
            let end = (B_TILE_OFF as u64) + (B_TILE_BYTES as u64);
            assert!(
                end <= SCRATCH_BYTES as u64,
                "Gemm: b_tile OOB scratch budget"
            );
            assert!(ITERS > 0, "Gemm: ITERS must be > 0");
            assert!(LAYER < NUM_LAYERS, "Gemm: LAYER OOB");
            assert!(N > 0, "Gemm: N must be > 0");
            assert!(K > 0, "Gemm: K must be > 0");
            assert!(M > 0, "Gemm: M (NUM_TOKENS) must be > 0");
            assert!(CONSUMER_PHASE == ARRIVES & 1, "Gemm: CONSUMER_PHASE parity");
            assert!(
                STORER_PHASE == (ARRIVES + 1) & 1,
                "Gemm: STORER_PHASE parity"
            );
            // Tile-layout consistency. AlongN warp split: each
            // consumer warp covers TILE_N output cols. tile_n * NCW
            // equality with N is enforced by the proc-macro (NCW
            // isn't a const generic here); tile_n > 0 is.
            assert!(TILE_N > 0, "Gemm: TILE_N must be > 0");
            assert!(CHUNK_K > 0, "Gemm: CHUNK_K must be > 0");
            // Per-iter K coverage: chunk_k * iters == K. Hard
            // mathematical invariant — the kernel covers the full
            // K dim in `iters` steps of `chunk_k` each.
            assert!(
                CHUNK_K * ITERS == K,
                "Gemm: CHUNK_K * ITERS must equal K"
            );
            // (b_tile_bytes layout is an emit-side decision — could
            // be [chunk_k, N] CTA-shared or [chunk_k, tile_n]
            // per-warp. Don't pre-commit at the IR level.)
        }
        use crate::ir::substrate::{
            ActSlotConst, BarSyncId, ChunkK, IterCount, MatmulK, MatmulM, MatmulN,
            MbarrierPhase, PageId, ScratchBytesRef, ScratchOffsetRef, TileN, WeightAccessorConst,
        };
        Self {
            in_page: PageId::<IN_ID, NUM_PAGES>::new().erase(),
            weight_page: PageId::<WEIGHT_ID, NUM_PAGES>::new().erase(),
            out_page: PageId::<OUT_ID, NUM_PAGES>::new().erase(),
            b_tile_offset: ScratchOffsetRef::__new_for_erase(B_TILE_OFF),
            b_tile_bytes: ScratchBytesRef::__new_for_erase(B_TILE_BYTES),
            consumer_phase: MbarrierPhase::<CONSUMER_PHASE>::new().erase(),
            storer_phase: MbarrierPhase::<STORER_PHASE>::new().erase(),
            iters: IterCount::<ITERS>::new().erase(),
            layer: LayerIndex::<LAYER, NUM_LAYERS>::new().erase(),
            n: MatmulN::<N>::new().erase(),
            k: MatmulK::<K>::new().erase(),
            m: MatmulM::<M>::new().erase(),
            tile_n: TileN::<TILE_N>::new().erase(),
            chunk_k: ChunkK::<CHUNK_K>::new().erase(),
            in_act_slot: ActSlotConst::<IN_ACT_SLOT, { u32::MAX }>::new().erase(),
            out_act_slot: ActSlotConst::<OUT_ACT_SLOT, { u32::MAX }>::new().erase(),
            weight_accessor_idx: WeightAccessorConst::<WEIGHT_ACCESSOR_IDX, { u32::MAX }>::new()
                .erase(),
            consumer_bar_publish: BarSyncId::<CONSUMER_BAR_PUBLISH>::new().erase(),
            weight,
        }
    }

    pub const fn in_page(&self) -> crate::ir::substrate::PageRef {
        self.in_page
    }
    pub const fn weight_page(&self) -> crate::ir::substrate::PageRef {
        self.weight_page
    }
    pub const fn out_page(&self) -> crate::ir::substrate::PageRef {
        self.out_page
    }
    pub const fn tile_n(&self) -> crate::ir::substrate::TileNRef {
        self.tile_n
    }
    pub const fn chunk_k(&self) -> crate::ir::substrate::ChunkKRef {
        self.chunk_k
    }
    pub const fn b_tile_offset(&self) -> crate::ir::substrate::ScratchOffsetRef {
        self.b_tile_offset
    }
    pub const fn b_tile_bytes(&self) -> crate::ir::substrate::ScratchBytesRef {
        self.b_tile_bytes
    }
    pub const fn consumer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.storer_phase
    }
    pub const fn iters(&self) -> crate::ir::substrate::IterCountRef {
        self.iters
    }
    pub const fn layer(&self) -> crate::ir::substrate::LayerRef {
        self.layer
    }
    pub const fn n(&self) -> crate::ir::substrate::MatmulNRef {
        self.n
    }
    pub const fn k(&self) -> crate::ir::substrate::MatmulKRef {
        self.k
    }
    pub const fn m(&self) -> crate::ir::substrate::MatmulMRef {
        self.m
    }
    pub const fn in_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.in_act_slot
    }
    pub const fn out_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.out_act_slot
    }
    pub const fn weight_accessor_idx(&self) -> crate::ir::substrate::WeightAccessorRef {
        self.weight_accessor_idx
    }
    /// `bar.sync` ID for the consumer's "all warps wrote their
    /// `[M, TILE_N]` output slice" publish before warp 0 arrives
    /// on `page_done[out_page]`. Validity (1..=15) was discharged
    /// by sealed-witness type-check at construction.
    pub const fn consumer_bar_publish(&self) -> crate::ir::substrate::BarRef {
        self.consumer_bar_publish
    }
}

/// TK-emit backend variant for the frontend
/// `Instruction::FusedCublasGemmAdd(in, residual, layer, n, k)` →
/// `residual += gemm(in, weight[layer])` in place. Substrate shape
/// is `Gemm` plus a `residual_page` that's read AND written (the
/// output writes back to the residual buffer; no separate out_page).
///
/// Codegen inlines the role bodies. Template
/// `<K, N, NUM_TOKENS, K_OFFSET, K_FULL>`. K_OFFSET / K_FULL
/// support the 4-chunk down_proj split (`TkGemmAdd` frontend path);
/// the un-split case has K_OFFSET = 0, K_FULL = K.
pub struct TkFusedGemmAdd {
    in_page: crate::ir::substrate::PageRef,
    weight_page: crate::ir::substrate::PageRef,
    residual_page: crate::ir::substrate::PageRef,
    b_tile_offset: crate::ir::substrate::ScratchOffsetRef,
    b_tile_bytes: crate::ir::substrate::ScratchBytesRef,
    consumer_phase: crate::ir::substrate::MbarrierPhaseRef,
    storer_phase: crate::ir::substrate::MbarrierPhaseRef,
    iters: crate::ir::substrate::IterCountRef,
    layer: crate::ir::substrate::LayerRef,
    n: crate::ir::substrate::MatmulNRef,
    k: crate::ir::substrate::MatmulKRef,
    num_tokens: crate::ir::substrate::NumTokensRef,
    k_offset: crate::ir::substrate::KOffsetRef,
    k_full: crate::ir::substrate::KFullRef,
    /// Per-warp output N slice (AlongN warp split). Mirrors the
    /// `Gemm::tile_n` field added in Sprint 9.
    tile_n: crate::ir::substrate::TileNRef,
    /// Per-iter K-chunk width loaded into the b_tile. Mirrors the
    /// `Gemm::chunk_k` field added in Sprint 9.
    chunk_k: crate::ir::substrate::ChunkKRef,
    in_act_slot: crate::ir::substrate::ActSlotRef,
    residual_act_slot: crate::ir::substrate::ActSlotRef,
    weight_accessor_idx: crate::ir::substrate::WeightAccessorRef,
    /// Cross-warp `bar.sync` ID for the consumer's "all warps wrote
    /// their `[M, TILE_N]` accumulator slice into the residual page"
    /// publish before warp 0 arrives on `page_done[residual_page]`.
    /// Mirrors `Gemm::consumer_bar_publish` (Sprint 10a). Validity
    /// (1..=15) is enforced at construction by `BarSyncId<ID>:
    /// IsValidBarSyncId`. TkFusedGemmAdd has no cross-warp
    /// reduction (each warp owns disjoint output cols under the
    /// AlongN split), so only one bar is needed.
    consumer_bar_publish: crate::ir::substrate::BarRef,
    pub weight: WeightRef,
}

impl TkFusedGemmAdd {
    #[allow(clippy::too_many_arguments)]
    pub fn new<
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
        const NUM_PAGES: u32,
        const NUM_LAYERS: u32,
        const SCRATCH_BYTES: u32,
        const ARRIVES: u32,
        const NUM_TOKENS: u32,
        const K_OFFSET: u32,
        const K_FULL: u32,
        const IN_ACT_SLOT: u32,
        const RESIDUAL_ACT_SLOT: u32,
        const WEIGHT_ACCESSOR_IDX: u32,
        const TILE_N: u32,
        const CHUNK_K: u32,
        const CONSUMER_BAR_PUBLISH: u32,
    >(
        weight: WeightRef,
    ) -> Self
    where
        // Sealed-witness: BAR_PUBLISH ∈ 1..=15. ID 0 is reserved
        // for `__syncthreads`; 16+ is out of PTX range. Mirrors
        // `Gemm::new` (Sprint 10a).
        crate::ir::substrate::BarSyncId<CONSUMER_BAR_PUBLISH>:
            crate::ir::substrate::IsValidBarSyncId,
    {
        const {
            assert!(IN_ID < NUM_PAGES, "TkFusedGemmAdd: IN_ID OOB");
            assert!(WEIGHT_ID < NUM_PAGES, "TkFusedGemmAdd: WEIGHT_ID OOB");
            assert!(
                RESIDUAL_ID < NUM_PAGES,
                "TkFusedGemmAdd: RESIDUAL_ID OOB"
            );
            assert!(
                IN_ID != WEIGHT_ID && IN_ID != RESIDUAL_ID && WEIGHT_ID != RESIDUAL_ID,
                "TkFusedGemmAdd: page alias"
            );
            let end = (B_TILE_OFF as u64) + (B_TILE_BYTES as u64);
            assert!(
                end <= SCRATCH_BYTES as u64,
                "TkFusedGemmAdd: b_tile OOB scratch budget"
            );
            assert!(ITERS > 0, "TkFusedGemmAdd: ITERS must be > 0");
            assert!(LAYER < NUM_LAYERS, "TkFusedGemmAdd: LAYER OOB");
            assert!(N > 0, "TkFusedGemmAdd: N must be > 0");
            assert!(K > 0, "TkFusedGemmAdd: K must be > 0");
            assert!(
                CONSUMER_PHASE == ARRIVES & 1,
                "TkFusedGemmAdd: CONSUMER_PHASE parity"
            );
            assert!(
                STORER_PHASE == (ARRIVES + 1) & 1,
                "TkFusedGemmAdd: STORER_PHASE parity"
            );
            assert!(
                NUM_TOKENS > 0,
                "TkFusedGemmAdd: NUM_TOKENS must be > 0"
            );
            assert!(K_FULL > 0, "TkFusedGemmAdd: K_FULL must be > 0");
            assert!(
                (K_OFFSET as u64) + (K as u64) <= K_FULL as u64,
                "TkFusedGemmAdd: K_OFFSET + K must be <= K_FULL"
            );
            // Tile-layout invariants (mirror of Gemm Sprint 9).
            // AlongN warp split: each consumer warp covers TILE_N
            // output cols. tile_n * NCW equality with N is enforced
            // by the proc-macro (NCW isn't a const generic here);
            // tile_n > 0 is.
            assert!(TILE_N > 0, "TkFusedGemmAdd: TILE_N must be > 0");
            assert!(CHUNK_K > 0, "TkFusedGemmAdd: CHUNK_K must be > 0");
            // Per-iter K coverage: chunk_k * iters == K. Hard
            // mathematical invariant — the kernel covers the full
            // K dim in `iters` steps of `chunk_k` each.
            assert!(
                CHUNK_K * ITERS == K,
                "TkFusedGemmAdd: CHUNK_K * ITERS must equal K"
            );
        }
        use crate::ir::substrate::{
            ActSlotConst, BarSyncId, ChunkK, IterCount, KFull, KOffset, MatmulK, MatmulN,
            MbarrierPhase, NumTokensConst, PageId, ScratchBytesRef, ScratchOffsetRef, TileN,
            WeightAccessorConst,
        };
        Self {
            in_page: PageId::<IN_ID, NUM_PAGES>::new().erase(),
            weight_page: PageId::<WEIGHT_ID, NUM_PAGES>::new().erase(),
            residual_page: PageId::<RESIDUAL_ID, NUM_PAGES>::new().erase(),
            b_tile_offset: ScratchOffsetRef::__new_for_erase(B_TILE_OFF),
            b_tile_bytes: ScratchBytesRef::__new_for_erase(B_TILE_BYTES),
            consumer_phase: MbarrierPhase::<CONSUMER_PHASE>::new().erase(),
            storer_phase: MbarrierPhase::<STORER_PHASE>::new().erase(),
            iters: IterCount::<ITERS>::new().erase(),
            layer: LayerIndex::<LAYER, NUM_LAYERS>::new().erase(),
            n: MatmulN::<N>::new().erase(),
            k: MatmulK::<K>::new().erase(),
            num_tokens: NumTokensConst::<NUM_TOKENS>::new().erase(),
            k_offset: KOffset::<K_OFFSET>::new().erase(),
            k_full: KFull::<K_FULL>::new().erase(),
            tile_n: TileN::<TILE_N>::new().erase(),
            chunk_k: ChunkK::<CHUNK_K>::new().erase(),
            in_act_slot: ActSlotConst::<IN_ACT_SLOT, { u32::MAX }>::new().erase(),
            residual_act_slot: ActSlotConst::<RESIDUAL_ACT_SLOT, { u32::MAX }>::new().erase(),
            weight_accessor_idx: WeightAccessorConst::<WEIGHT_ACCESSOR_IDX, { u32::MAX }>::new()
                .erase(),
            consumer_bar_publish: BarSyncId::<CONSUMER_BAR_PUBLISH>::new().erase(),
            weight,
        }
    }

    pub const fn in_page(&self) -> crate::ir::substrate::PageRef {
        self.in_page
    }
    pub const fn weight_page(&self) -> crate::ir::substrate::PageRef {
        self.weight_page
    }
    pub const fn residual_page(&self) -> crate::ir::substrate::PageRef {
        self.residual_page
    }
    pub const fn tile_n(&self) -> crate::ir::substrate::TileNRef {
        self.tile_n
    }
    pub const fn chunk_k(&self) -> crate::ir::substrate::ChunkKRef {
        self.chunk_k
    }
    pub const fn b_tile_offset(&self) -> crate::ir::substrate::ScratchOffsetRef {
        self.b_tile_offset
    }
    pub const fn b_tile_bytes(&self) -> crate::ir::substrate::ScratchBytesRef {
        self.b_tile_bytes
    }
    pub const fn consumer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.storer_phase
    }
    pub const fn iters(&self) -> crate::ir::substrate::IterCountRef {
        self.iters
    }
    pub const fn layer(&self) -> crate::ir::substrate::LayerRef {
        self.layer
    }
    pub const fn n(&self) -> crate::ir::substrate::MatmulNRef {
        self.n
    }
    pub const fn k(&self) -> crate::ir::substrate::MatmulKRef {
        self.k
    }
    pub const fn num_tokens(&self) -> crate::ir::substrate::NumTokensRef {
        self.num_tokens
    }
    pub const fn k_offset(&self) -> crate::ir::substrate::KOffsetRef {
        self.k_offset
    }
    pub const fn k_full(&self) -> crate::ir::substrate::KFullRef {
        self.k_full
    }
    pub const fn in_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.in_act_slot
    }
    pub const fn residual_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.residual_act_slot
    }
    pub const fn weight_accessor_idx(&self) -> crate::ir::substrate::WeightAccessorRef {
        self.weight_accessor_idx
    }
    /// `bar.sync` ID for the consumer's "all warps wrote their
    /// `[M, TILE_N]` accumulator slice" publish before warp 0
    /// arrives on `page_done[residual_page]`. Validity (1..=15) was
    /// discharged by sealed-witness type-check at construction.
    pub const fn consumer_bar_publish(&self) -> crate::ir::substrate::BarRef {
        self.consumer_bar_publish
    }
}

/// `TkFusedNormGemm` (lm_head fusion) variant. TK-emit-only union
/// over the four frontend `Cutlass*` variants
/// (`CutlassFusedRmsNormGemm`, `CutlassFusedMeanSubRmsNormGemm`,
/// `CutlassFusedAddRmsNormGemm`, `CutlassFusedAddScalarOffsetRmsNormGemm`)
/// — the `norm_kind` enum + optional `delta_page` / `offset` select
/// which frontend flavor lowers to this backend node.
///
/// `delta_page_id`: `Some` for AddRmsNorm / AddScalarOffsetRmsNorm.
/// `offset`: `Some` only for AddScalarOffsetRmsNorm.
///
/// The Option<u32> for `delta_page_id` and Option<FiniteF32> for
/// `offset` are runtime — the cross-field invariant
/// `(norm_kind == AddScalarOffsetRmsNorm) <=> offset.is_some()`
/// requires runtime branching at the builder. The substrate-proof
/// fields (page bounds, scratch budget, phase parity, n/k > 0) are
/// const-generic.
///
/// Codegen inlines the role bodies. Template `<K, N, NUM_TOKENS>`
/// with NUM_TOKENS = 1 today. The `norm_kind` enum selects which
/// fused-norm sequence the codegen emits (RmsNorm / AddRmsNorm /
/// MeanSubRmsNorm / AddScalarOffsetRmsNorm).
///
/// Sprint 13a (S13a) IR ext: `tile_n` + `chunk_k` (mirror Gemm
/// Sprint 9 / TkFusedGemmAdd Sprint 11a) cover the AlongN +
/// per-iter-K layout for the linear projection. `consumer_bar_reduce`
/// + `consumer_bar_publish` (mirror FusedAddRmsNorm) cover the
/// cross-warp norm reduce + the post-gemm publish before warp 0
/// arrives on `page_done`.
pub struct TkFusedNormGemm {
    in_page: crate::ir::substrate::PageRef,
    delta_page: Option<crate::ir::substrate::PageRef>,
    norm_weight_page: crate::ir::substrate::PageRef,
    linear_weight_page: crate::ir::substrate::PageRef,
    out_page: crate::ir::substrate::PageRef,
    partial_offset: crate::ir::substrate::ScratchOffsetRef,
    partial_bytes: crate::ir::substrate::ScratchBytesRef,
    b_tile_offset: crate::ir::substrate::ScratchOffsetRef,
    b_tile_bytes: crate::ir::substrate::ScratchBytesRef,
    consumer_phase: crate::ir::substrate::MbarrierPhaseRef,
    storer_phase: crate::ir::substrate::MbarrierPhaseRef,
    iters: crate::ir::substrate::IterCountRef,
    layer: crate::ir::substrate::LayerRef,
    n: crate::ir::substrate::MatmulNRef,
    k: crate::ir::substrate::MatmulKRef,
    num_tokens: crate::ir::substrate::NumTokensRef,
    /// Per-warp output N slice for the linear projection (AlongN
    /// warp split). Mirrors `Gemm::tile_n` / `TkFusedGemmAdd::tile_n`.
    tile_n: crate::ir::substrate::TileNRef,
    /// Per-iter K-chunk width for the b_tile. Mirrors
    /// `Gemm::chunk_k` / `TkFusedGemmAdd::chunk_k`. ITERS=1 today so
    /// `chunk_k == k` is the only legal config.
    chunk_k: crate::ir::substrate::ChunkKRef,
    in_act_slot: crate::ir::substrate::ActSlotRef,
    delta_act_slot: Option<crate::ir::substrate::ActSlotRef>,
    out_act_slot: crate::ir::substrate::ActSlotRef,
    norm_weight_accessor_idx: crate::ir::substrate::WeightAccessorRef,
    linear_weight_accessor_idx: crate::ir::substrate::WeightAccessorRef,
    /// Cross-warp `bar.sync` ID for the consumer's sum-of-squares
    /// reduction across consumer warps. Type-checked in 1..=15.
    /// Mirror of `FusedAddRmsNorm::consumer_bar_reduce`.
    consumer_bar_reduce: crate::ir::substrate::BarRef,
    /// Cross-warp `bar.sync` ID used twice: once after the norm
    /// writeback so all warps see the full normalized A in shared
    /// memory before the linear matmul, and once after the matmul
    /// writeback so warp 0 can safely arrive on `page_done`.
    /// Distinct from `consumer_bar_reduce`.
    consumer_bar_publish: crate::ir::substrate::BarRef,
    /// Witness that `consumer_bar_reduce != consumer_bar_publish`.
    /// Storage-erased zero-sized token whose existence is the proof.
    bar_pair_proof: crate::ir::substrate::DistinctBarPairProof,
    eps: FiniteF32,
    pub norm_weight: WeightRef,
    pub linear_weight: WeightRef,
    pub norm_kind: LmHeadNormKind,
    pub offset: Option<FiniteF32>,
}

impl TkFusedNormGemm {
    /// Const-generic constructor for the residual-fold-free flavors
    /// (RmsNorm / MeanSubRmsNorm — `delta_page_id` = None,
    /// `offset` = None).
    #[allow(clippy::too_many_arguments)]
    pub fn new_no_delta<
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
        const NUM_PAGES: u32,
        const NUM_LAYERS: u32,
        const SCRATCH_BYTES: u32,
        const ARRIVES: u32,
        const NUM_TOKENS: u32,
        const IN_ACT_SLOT: u32,
        const OUT_ACT_SLOT: u32,
        const NORM_WEIGHT_ACCESSOR_IDX: u32,
        const LINEAR_WEIGHT_ACCESSOR_IDX: u32,
        const TILE_N: u32,
        const CHUNK_K: u32,
        const CONSUMER_BAR_REDUCE: u32,
        const CONSUMER_BAR_PUBLISH: u32,
    >(
        norm_weight: WeightRef,
        linear_weight: WeightRef,
        norm_kind: LmHeadNormKind,
        eps: FiniteF32,
    ) -> Self
    where
        crate::ir::substrate::BarSyncId<CONSUMER_BAR_REDUCE>:
            crate::ir::substrate::IsValidBarSyncId,
        crate::ir::substrate::BarSyncId<CONSUMER_BAR_PUBLISH>:
            crate::ir::substrate::IsValidBarSyncId,
        crate::ir::substrate::BarSyncPair<CONSUMER_BAR_REDUCE, CONSUMER_BAR_PUBLISH>:
            crate::ir::substrate::IsDistinctBarPair,
    {
        const {
            assert!(IN_ID < NUM_PAGES, "TkFusedNormGemm: IN_ID OOB");
            assert!(NORM_W_ID < NUM_PAGES, "TkFusedNormGemm: NORM_W_ID OOB");
            assert!(LIN_W_ID < NUM_PAGES, "TkFusedNormGemm: LIN_W_ID OOB");
            assert!(OUT_ID < NUM_PAGES, "TkFusedNormGemm: OUT_ID OOB");
            assert!(
                IN_ID != NORM_W_ID
                    && IN_ID != LIN_W_ID
                    && IN_ID != OUT_ID
                    && NORM_W_ID != LIN_W_ID
                    && NORM_W_ID != OUT_ID
                    && LIN_W_ID != OUT_ID,
                "TkFusedNormGemm: page alias"
            );
            let p_end = (PARTIAL_OFF as u64) + (PARTIAL_BYTES as u64);
            let b_end = (B_TILE_OFF as u64) + (B_TILE_BYTES as u64);
            assert!(
                p_end <= SCRATCH_BYTES as u64,
                "TkFusedNormGemm: partial_sums OOB"
            );
            assert!(
                b_end <= SCRATCH_BYTES as u64,
                "TkFusedNormGemm: b_tile OOB"
            );
            assert!(ITERS > 0, "TkFusedNormGemm: ITERS must be > 0");
            assert!(LAYER < NUM_LAYERS, "TkFusedNormGemm: LAYER OOB");
            assert!(N > 0, "TkFusedNormGemm: N must be > 0");
            assert!(K > 0, "TkFusedNormGemm: K must be > 0");
            assert!(
                CONSUMER_PHASE == ARRIVES & 1,
                "TkFusedNormGemm: CONSUMER_PHASE parity"
            );
            assert!(
                STORER_PHASE == (ARRIVES + 1) & 1,
                "TkFusedNormGemm: STORER_PHASE parity"
            );
            assert!(
                NUM_TOKENS > 0,
                "TkFusedNormGemm: NUM_TOKENS must be > 0"
            );
            // Tile-layout invariants (mirror of Gemm Sprint 9).
            // AlongN warp split: each consumer warp covers TILE_N
            // output cols. tile_n * NCW equality with N is enforced
            // by the proc-macro (NCW isn't a const generic here);
            // tile_n > 0 is.
            assert!(TILE_N > 0, "TkFusedNormGemm: TILE_N must be > 0");
            assert!(CHUNK_K > 0, "TkFusedNormGemm: CHUNK_K must be > 0");
            // Per-iter K coverage: chunk_k * iters == K.
            assert!(
                CHUNK_K * ITERS == K,
                "TkFusedNormGemm: CHUNK_K * ITERS must equal K"
            );
        }
        // Runtime cross-field invariant: norm_kind must NOT carry
        // residual fold or scalar offset for this constructor.
        match norm_kind {
            LmHeadNormKind::RmsNorm | LmHeadNormKind::MeanSubRmsNorm => {}
            LmHeadNormKind::AddRmsNorm | LmHeadNormKind::AddScalarOffsetRmsNorm => {
                panic!(
                    "TkFusedNormGemm::new_no_delta: norm_kind requires a delta page; use new_with_delta",
                );
            }
        }
        use crate::ir::substrate::{
            ActSlotConst, BarSyncId, BarSyncPair, ChunkK, IterCount, MatmulK, MatmulN,
            MbarrierPhase, NumTokensConst, PageId, ScratchBytesRef, ScratchOffsetRef, TileN,
            WeightAccessorConst,
        };
        Self {
            in_page: PageId::<IN_ID, NUM_PAGES>::new().erase(),
            delta_page: None,
            norm_weight_page: PageId::<NORM_W_ID, NUM_PAGES>::new().erase(),
            linear_weight_page: PageId::<LIN_W_ID, NUM_PAGES>::new().erase(),
            out_page: PageId::<OUT_ID, NUM_PAGES>::new().erase(),
            partial_offset: ScratchOffsetRef::__new_for_erase(PARTIAL_OFF),
            partial_bytes: ScratchBytesRef::__new_for_erase(PARTIAL_BYTES),
            b_tile_offset: ScratchOffsetRef::__new_for_erase(B_TILE_OFF),
            b_tile_bytes: ScratchBytesRef::__new_for_erase(B_TILE_BYTES),
            consumer_phase: MbarrierPhase::<CONSUMER_PHASE>::new().erase(),
            storer_phase: MbarrierPhase::<STORER_PHASE>::new().erase(),
            iters: IterCount::<ITERS>::new().erase(),
            layer: LayerIndex::<LAYER, NUM_LAYERS>::new().erase(),
            n: MatmulN::<N>::new().erase(),
            k: MatmulK::<K>::new().erase(),
            num_tokens: NumTokensConst::<NUM_TOKENS>::new().erase(),
            tile_n: TileN::<TILE_N>::new().erase(),
            chunk_k: ChunkK::<CHUNK_K>::new().erase(),
            in_act_slot: ActSlotConst::<IN_ACT_SLOT, { u32::MAX }>::new().erase(),
            delta_act_slot: None,
            out_act_slot: ActSlotConst::<OUT_ACT_SLOT, { u32::MAX }>::new().erase(),
            norm_weight_accessor_idx: WeightAccessorConst::<
                NORM_WEIGHT_ACCESSOR_IDX,
                { u32::MAX },
            >::new()
            .erase(),
            linear_weight_accessor_idx: WeightAccessorConst::<
                LINEAR_WEIGHT_ACCESSOR_IDX,
                { u32::MAX },
            >::new()
            .erase(),
            consumer_bar_reduce: BarSyncId::<CONSUMER_BAR_REDUCE>::new().erase(),
            consumer_bar_publish: BarSyncId::<CONSUMER_BAR_PUBLISH>::new().erase(),
            bar_pair_proof: BarSyncPair::<CONSUMER_BAR_REDUCE, CONSUMER_BAR_PUBLISH>::new()
                .erase(),
            eps,
            norm_weight,
            linear_weight,
            norm_kind,
            offset: None,
        }
    }

    /// Const-generic constructor for residual-fold flavors
    /// (AddRmsNorm / AddScalarOffsetRmsNorm).
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_delta<
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
        const NUM_PAGES: u32,
        const NUM_LAYERS: u32,
        const SCRATCH_BYTES: u32,
        const ARRIVES: u32,
        const NUM_TOKENS: u32,
        const IN_ACT_SLOT: u32,
        const DELTA_ACT_SLOT: u32,
        const OUT_ACT_SLOT: u32,
        const NORM_WEIGHT_ACCESSOR_IDX: u32,
        const LINEAR_WEIGHT_ACCESSOR_IDX: u32,
        const TILE_N: u32,
        const CHUNK_K: u32,
        const CONSUMER_BAR_REDUCE: u32,
        const CONSUMER_BAR_PUBLISH: u32,
    >(
        norm_weight: WeightRef,
        linear_weight: WeightRef,
        norm_kind: LmHeadNormKind,
        offset: Option<FiniteF32>,
        eps: FiniteF32,
    ) -> Self
    where
        crate::ir::substrate::BarSyncId<CONSUMER_BAR_REDUCE>:
            crate::ir::substrate::IsValidBarSyncId,
        crate::ir::substrate::BarSyncId<CONSUMER_BAR_PUBLISH>:
            crate::ir::substrate::IsValidBarSyncId,
        crate::ir::substrate::BarSyncPair<CONSUMER_BAR_REDUCE, CONSUMER_BAR_PUBLISH>:
            crate::ir::substrate::IsDistinctBarPair,
    {
        const {
            assert!(IN_ID < NUM_PAGES, "TkFusedNormGemm: IN_ID OOB");
            assert!(DELTA_ID < NUM_PAGES, "TkFusedNormGemm: DELTA_ID OOB");
            assert!(NORM_W_ID < NUM_PAGES, "TkFusedNormGemm: NORM_W_ID OOB");
            assert!(LIN_W_ID < NUM_PAGES, "TkFusedNormGemm: LIN_W_ID OOB");
            assert!(OUT_ID < NUM_PAGES, "TkFusedNormGemm: OUT_ID OOB");
            assert!(
                IN_ID != DELTA_ID
                    && IN_ID != NORM_W_ID
                    && IN_ID != LIN_W_ID
                    && IN_ID != OUT_ID
                    && DELTA_ID != NORM_W_ID
                    && DELTA_ID != LIN_W_ID
                    && DELTA_ID != OUT_ID
                    && NORM_W_ID != LIN_W_ID
                    && NORM_W_ID != OUT_ID
                    && LIN_W_ID != OUT_ID,
                "TkFusedNormGemm: page alias"
            );
            let p_end = (PARTIAL_OFF as u64) + (PARTIAL_BYTES as u64);
            let b_end = (B_TILE_OFF as u64) + (B_TILE_BYTES as u64);
            assert!(
                p_end <= SCRATCH_BYTES as u64,
                "TkFusedNormGemm: partial_sums OOB"
            );
            assert!(
                b_end <= SCRATCH_BYTES as u64,
                "TkFusedNormGemm: b_tile OOB"
            );
            assert!(ITERS > 0, "TkFusedNormGemm: ITERS must be > 0");
            assert!(LAYER < NUM_LAYERS, "TkFusedNormGemm: LAYER OOB");
            assert!(N > 0, "TkFusedNormGemm: N must be > 0");
            assert!(K > 0, "TkFusedNormGemm: K must be > 0");
            assert!(
                CONSUMER_PHASE == ARRIVES & 1,
                "TkFusedNormGemm: CONSUMER_PHASE parity"
            );
            assert!(
                STORER_PHASE == (ARRIVES + 1) & 1,
                "TkFusedNormGemm: STORER_PHASE parity"
            );
            assert!(
                NUM_TOKENS > 0,
                "TkFusedNormGemm: NUM_TOKENS must be > 0"
            );
            assert!(TILE_N > 0, "TkFusedNormGemm: TILE_N must be > 0");
            assert!(CHUNK_K > 0, "TkFusedNormGemm: CHUNK_K must be > 0");
            assert!(
                CHUNK_K * ITERS == K,
                "TkFusedNormGemm: CHUNK_K * ITERS must equal K"
            );
        }
        match (norm_kind, offset.is_some()) {
            (LmHeadNormKind::AddScalarOffsetRmsNorm, true)
            | (LmHeadNormKind::AddRmsNorm, false) => {}
            (LmHeadNormKind::AddScalarOffsetRmsNorm, false) => {
                panic!(
                    "TkFusedNormGemm::new_with_delta: AddScalarOffsetRmsNorm requires Some(offset)"
                );
            }
            (LmHeadNormKind::AddRmsNorm, true) => {
                panic!("TkFusedNormGemm::new_with_delta: AddRmsNorm must not carry an offset");
            }
            (kind, _) => {
                let _ = kind;
                panic!("TkFusedNormGemm::new_with_delta: norm_kind cannot carry a delta page");
            }
        }
        use crate::ir::substrate::{
            ActSlotConst, BarSyncId, BarSyncPair, ChunkK, IterCount, MatmulK, MatmulN,
            MbarrierPhase, NumTokensConst, PageId, ScratchBytesRef, ScratchOffsetRef, TileN,
            WeightAccessorConst,
        };
        Self {
            in_page: PageId::<IN_ID, NUM_PAGES>::new().erase(),
            delta_page: Some(PageId::<DELTA_ID, NUM_PAGES>::new().erase()),
            norm_weight_page: PageId::<NORM_W_ID, NUM_PAGES>::new().erase(),
            linear_weight_page: PageId::<LIN_W_ID, NUM_PAGES>::new().erase(),
            out_page: PageId::<OUT_ID, NUM_PAGES>::new().erase(),
            partial_offset: ScratchOffsetRef::__new_for_erase(PARTIAL_OFF),
            partial_bytes: ScratchBytesRef::__new_for_erase(PARTIAL_BYTES),
            b_tile_offset: ScratchOffsetRef::__new_for_erase(B_TILE_OFF),
            b_tile_bytes: ScratchBytesRef::__new_for_erase(B_TILE_BYTES),
            consumer_phase: MbarrierPhase::<CONSUMER_PHASE>::new().erase(),
            storer_phase: MbarrierPhase::<STORER_PHASE>::new().erase(),
            iters: IterCount::<ITERS>::new().erase(),
            layer: LayerIndex::<LAYER, NUM_LAYERS>::new().erase(),
            n: MatmulN::<N>::new().erase(),
            k: MatmulK::<K>::new().erase(),
            num_tokens: NumTokensConst::<NUM_TOKENS>::new().erase(),
            tile_n: TileN::<TILE_N>::new().erase(),
            chunk_k: ChunkK::<CHUNK_K>::new().erase(),
            in_act_slot: ActSlotConst::<IN_ACT_SLOT, { u32::MAX }>::new().erase(),
            delta_act_slot: Some(ActSlotConst::<DELTA_ACT_SLOT, { u32::MAX }>::new().erase()),
            out_act_slot: ActSlotConst::<OUT_ACT_SLOT, { u32::MAX }>::new().erase(),
            norm_weight_accessor_idx: WeightAccessorConst::<
                NORM_WEIGHT_ACCESSOR_IDX,
                { u32::MAX },
            >::new()
            .erase(),
            linear_weight_accessor_idx: WeightAccessorConst::<
                LINEAR_WEIGHT_ACCESSOR_IDX,
                { u32::MAX },
            >::new()
            .erase(),
            consumer_bar_reduce: BarSyncId::<CONSUMER_BAR_REDUCE>::new().erase(),
            consumer_bar_publish: BarSyncId::<CONSUMER_BAR_PUBLISH>::new().erase(),
            bar_pair_proof: BarSyncPair::<CONSUMER_BAR_REDUCE, CONSUMER_BAR_PUBLISH>::new()
                .erase(),
            eps,
            norm_weight,
            linear_weight,
            norm_kind,
            offset,
        }
    }

    pub const fn in_page(&self) -> crate::ir::substrate::PageRef {
        self.in_page
    }
    pub const fn delta_page(&self) -> Option<crate::ir::substrate::PageRef> {
        self.delta_page
    }
    pub const fn norm_weight_page(&self) -> crate::ir::substrate::PageRef {
        self.norm_weight_page
    }
    pub const fn linear_weight_page(&self) -> crate::ir::substrate::PageRef {
        self.linear_weight_page
    }
    pub const fn out_page(&self) -> crate::ir::substrate::PageRef {
        self.out_page
    }
    pub const fn partial_offset(&self) -> crate::ir::substrate::ScratchOffsetRef {
        self.partial_offset
    }
    pub const fn partial_bytes(&self) -> crate::ir::substrate::ScratchBytesRef {
        self.partial_bytes
    }
    pub const fn b_tile_offset(&self) -> crate::ir::substrate::ScratchOffsetRef {
        self.b_tile_offset
    }
    pub const fn b_tile_bytes(&self) -> crate::ir::substrate::ScratchBytesRef {
        self.b_tile_bytes
    }
    pub const fn consumer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.storer_phase
    }
    pub const fn iters(&self) -> crate::ir::substrate::IterCountRef {
        self.iters
    }
    pub const fn layer(&self) -> crate::ir::substrate::LayerRef {
        self.layer
    }
    pub const fn n(&self) -> crate::ir::substrate::MatmulNRef {
        self.n
    }
    pub const fn k(&self) -> crate::ir::substrate::MatmulKRef {
        self.k
    }
    pub const fn num_tokens(&self) -> crate::ir::substrate::NumTokensRef {
        self.num_tokens
    }
    pub const fn in_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.in_act_slot
    }
    pub const fn delta_act_slot(&self) -> Option<crate::ir::substrate::ActSlotRef> {
        self.delta_act_slot
    }
    pub const fn out_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.out_act_slot
    }
    pub const fn norm_weight_accessor_idx(&self) -> crate::ir::substrate::WeightAccessorRef {
        self.norm_weight_accessor_idx
    }
    pub const fn linear_weight_accessor_idx(&self) -> crate::ir::substrate::WeightAccessorRef {
        self.linear_weight_accessor_idx
    }
    pub const fn tile_n(&self) -> crate::ir::substrate::TileNRef {
        self.tile_n
    }
    pub const fn chunk_k(&self) -> crate::ir::substrate::ChunkKRef {
        self.chunk_k
    }
    pub const fn consumer_bar_reduce(&self) -> crate::ir::substrate::BarRef {
        self.consumer_bar_reduce
    }
    pub const fn consumer_bar_publish(&self) -> crate::ir::substrate::BarRef {
        self.consumer_bar_publish
    }
    pub const fn bar_pair_proof(&self) -> crate::ir::substrate::DistinctBarPairProof {
        self.bar_pair_proof
    }
    pub fn eps(&self) -> FiniteF32 {
        self.eps
    }
}

/// `AttentionViaCacheNode` (covers `AttentionViaCache` and
/// `SlidingAttentionViaCache`).
///
/// Codegen inlines the four role bodies. Template arg list:
/// `<HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS, BLOCK_SIZE, NUM_TOKENS,
/// SPLITS, SLIDING_WINDOW, HAS_SOFTCAP, MAX_SK>`. `kind`
/// (Full/Sliding) → SLIDING_WINDOW; `attn_softcap > 0` →
/// HAS_SOFTCAP. SPLITS = 1 today; SPLITS > 1 fans through a
/// reduction step that the codegen will splice as a second per-tile
/// pass.
pub struct TkAttentionViaCacheNode {
    q_in_page: crate::ir::substrate::PageRef,
    attn_out_page: crate::ir::substrate::PageRef,
    score_offset: crate::ir::substrate::ScratchOffsetRef,
    score_bytes: crate::ir::substrate::ScratchBytesRef,
    pv_offset: crate::ir::substrate::ScratchOffsetRef,
    pv_bytes: crate::ir::substrate::ScratchBytesRef,
    // K_smem and V_smem staging buffers — single-stage paged-KV
    // gather buffers used by the loader role to TMA-load one KV
    // block at a time. Sized [BLOCK_SIZE, NUM_KV_HEADS * HEAD_DIM]
    // bf16 = `BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM * 2` bytes
    // each. Live in `AttentionScope` and are disjoint from
    // `score_*` / `pv_*` (proven via `ScratchRegion::disjoint_with`
    // at proc-macro construction time, per
    // [[feedback-end-to-end-compile-time-proofs]]).
    k_smem_offset: crate::ir::substrate::ScratchOffsetRef,
    k_smem_bytes: crate::ir::substrate::ScratchBytesRef,
    v_smem_offset: crate::ir::substrate::ScratchOffsetRef,
    v_smem_bytes: crate::ir::substrate::ScratchBytesRef,
    consumer_phase: crate::ir::substrate::MbarrierPhaseRef,
    storer_phase: crate::ir::substrate::MbarrierPhaseRef,
    iters: crate::ir::substrate::IterCountRef,
    kv_cache_layer: crate::ir::substrate::LayerRef,
    head_dim: crate::ir::substrate::HeadDimRef,
    num_q_heads: crate::ir::substrate::NumQHeadsRef,
    num_kv_heads: crate::ir::substrate::NumKvHeadsRef,
    block_size: crate::ir::substrate::BlockSizeRef,
    num_tokens: crate::ir::substrate::NumTokensRef,
    max_sk: crate::ir::substrate::MaxSkRef,
    q_in_act_slot: crate::ir::substrate::ActSlotRef,
    attn_out_act_slot: crate::ir::substrate::ActSlotRef,
    attn_scale: FiniteF32,
    attn_softcap: FiniteF32,
    pub interleaved: bool,
    pub kind: AttentionKind,
}

impl TkAttentionViaCacheNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new<
        const Q_IN_ID: u32,
        const ATTN_OUT_ID: u32,
        const SCORE_OFF: u32,
        const SCORE_BYTES: u32,
        const PV_OFF: u32,
        const PV_BYTES: u32,
        const K_SMEM_OFF: u32,
        const K_SMEM_BYTES: u32,
        const V_SMEM_OFF: u32,
        const V_SMEM_BYTES: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const ITERS: u32,
        const LAYER: u32,
        const NUM_PAGES: u32,
        const NUM_LAYERS: u32,
        const SCRATCH_BYTES: u32,
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
        kind: AttentionKind,
        interleaved: bool,
        attn_scale: FiniteF32,
        attn_softcap: FiniteF32,
    ) -> Self {
        const {
            assert!(Q_IN_ID < NUM_PAGES, "AttentionViaCache: Q_IN_ID OOB");
            assert!(
                ATTN_OUT_ID < NUM_PAGES,
                "AttentionViaCache: ATTN_OUT_ID OOB"
            );
            // NOTE: Q_IN_ID == ATTN_OUT_ID is intentional for the
            // in-place attention pattern — the kernel reads Q from
            // the page, computes attention via the global paged KV
            // cache, then writes the output back into the same page.
            // Lifecycle: Empty (stale) → Filled (Q loaded) →
            // Produced (consumer wrote attn output) → Empty (drained).
            // No aliasing problem within the op.
            let s_end = (SCORE_OFF as u64) + (SCORE_BYTES as u64);
            let p_end = (PV_OFF as u64) + (PV_BYTES as u64);
            let k_end = (K_SMEM_OFF as u64) + (K_SMEM_BYTES as u64);
            let v_end = (V_SMEM_OFF as u64) + (V_SMEM_BYTES as u64);
            assert!(
                s_end <= SCRATCH_BYTES as u64,
                "AttentionViaCache: score_tile OOB"
            );
            assert!(
                p_end <= SCRATCH_BYTES as u64,
                "AttentionViaCache: pv_tile OOB"
            );
            assert!(
                k_end <= SCRATCH_BYTES as u64,
                "AttentionViaCache: k_smem OOB"
            );
            assert!(
                v_end <= SCRATCH_BYTES as u64,
                "AttentionViaCache: v_smem OOB"
            );
            // All four AttentionScope regions must be pairwise
            // disjoint. (n*(n-1)/2 = 6 pairs for n=4.)
            let pairs: [(u64, u64, u64, u64); 6] = [
                (SCORE_OFF as u64, s_end, PV_OFF as u64, p_end),
                (SCORE_OFF as u64, s_end, K_SMEM_OFF as u64, k_end),
                (SCORE_OFF as u64, s_end, V_SMEM_OFF as u64, v_end),
                (PV_OFF as u64, p_end, K_SMEM_OFF as u64, k_end),
                (PV_OFF as u64, p_end, V_SMEM_OFF as u64, v_end),
                (K_SMEM_OFF as u64, k_end, V_SMEM_OFF as u64, v_end),
            ];
            let mut i = 0;
            while i < pairs.len() {
                let (a_off, a_end, b_off, b_end) = pairs[i];
                assert!(
                    a_end <= b_off || b_end <= a_off,
                    "AttentionViaCache: AttentionScope regions overlap"
                );
                i += 1;
            }
            assert!(ITERS > 0, "AttentionViaCache: ITERS must be > 0");
            assert!(LAYER < NUM_LAYERS, "AttentionViaCache: LAYER OOB");
            assert!(
                CONSUMER_PHASE == ARRIVES & 1,
                "AttentionViaCache: CONSUMER_PHASE parity"
            );
            assert!(
                STORER_PHASE == (ARRIVES + 1) & 1,
                "AttentionViaCache: STORER_PHASE parity"
            );
            assert!(HEAD_DIM > 0, "AttentionViaCache: HEAD_DIM must be > 0");
            assert!(
                NUM_Q_HEADS > 0,
                "AttentionViaCache: NUM_Q_HEADS must be > 0"
            );
            assert!(
                NUM_KV_HEADS > 0,
                "AttentionViaCache: NUM_KV_HEADS must be > 0"
            );
            assert!(
                BLOCK_SIZE > 0,
                "AttentionViaCache: BLOCK_SIZE must be > 0"
            );
            assert!(
                NUM_TOKENS > 0,
                "AttentionViaCache: NUM_TOKENS must be > 0"
            );
            assert!(MAX_SK > 0, "AttentionViaCache: MAX_SK must be > 0");
            // K_smem and V_smem must each fit
            // `BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM * sizeof(bf16)`
            // bytes (one paged-KV block, single-stage, bf16).
            let kv_block_bytes = (BLOCK_SIZE as u64)
                * (NUM_KV_HEADS as u64)
                * (HEAD_DIM as u64)
                * 2;
            assert!(
                K_SMEM_BYTES as u64 >= kv_block_bytes,
                "AttentionViaCache: K_SMEM_BYTES < BLOCK_SIZE*NUM_KV_HEADS*HEAD_DIM*2"
            );
            assert!(
                V_SMEM_BYTES as u64 >= kv_block_bytes,
                "AttentionViaCache: V_SMEM_BYTES < BLOCK_SIZE*NUM_KV_HEADS*HEAD_DIM*2"
            );
        }
        // Runtime: SlidingWindow value > 0 was discharged by the
        // const-generic SlidingWindow<W> primitive; here we just
        // store the runtime u32 carried in AttentionKind::Sliding.
        use crate::ir::substrate::{
            ActSlotConst, BlockSize, HeadDim, IterCount, MaxSk, MbarrierPhase, NumKvHeads,
            NumQHeads, NumTokensConst, PageId, ScratchBytesRef, ScratchOffsetRef,
        };
        Self {
            q_in_page: PageId::<Q_IN_ID, NUM_PAGES>::new().erase(),
            attn_out_page: PageId::<ATTN_OUT_ID, NUM_PAGES>::new().erase(),
            score_offset: ScratchOffsetRef::__new_for_erase(SCORE_OFF),
            score_bytes: ScratchBytesRef::__new_for_erase(SCORE_BYTES),
            pv_offset: ScratchOffsetRef::__new_for_erase(PV_OFF),
            pv_bytes: ScratchBytesRef::__new_for_erase(PV_BYTES),
            k_smem_offset: ScratchOffsetRef::__new_for_erase(K_SMEM_OFF),
            k_smem_bytes: ScratchBytesRef::__new_for_erase(K_SMEM_BYTES),
            v_smem_offset: ScratchOffsetRef::__new_for_erase(V_SMEM_OFF),
            v_smem_bytes: ScratchBytesRef::__new_for_erase(V_SMEM_BYTES),
            consumer_phase: MbarrierPhase::<CONSUMER_PHASE>::new().erase(),
            storer_phase: MbarrierPhase::<STORER_PHASE>::new().erase(),
            iters: IterCount::<ITERS>::new().erase(),
            kv_cache_layer: LayerIndex::<LAYER, NUM_LAYERS>::new().erase(),
            head_dim: HeadDim::<HEAD_DIM>::new().erase(),
            num_q_heads: NumQHeads::<NUM_Q_HEADS>::new().erase(),
            num_kv_heads: NumKvHeads::<NUM_KV_HEADS>::new().erase(),
            block_size: BlockSize::<BLOCK_SIZE>::new().erase(),
            num_tokens: NumTokensConst::<NUM_TOKENS>::new().erase(),
            max_sk: MaxSk::<MAX_SK>::new().erase(),
            q_in_act_slot: ActSlotConst::<Q_IN_ACT_SLOT, { u32::MAX }>::new().erase(),
            attn_out_act_slot: ActSlotConst::<ATTN_OUT_ACT_SLOT, { u32::MAX }>::new().erase(),
            attn_scale,
            attn_softcap,
            interleaved,
            kind,
        }
    }

    pub const fn q_in_page(&self) -> crate::ir::substrate::PageRef {
        self.q_in_page
    }
    pub const fn attn_out_page(&self) -> crate::ir::substrate::PageRef {
        self.attn_out_page
    }
    pub const fn score_offset(&self) -> crate::ir::substrate::ScratchOffsetRef {
        self.score_offset
    }
    pub const fn score_bytes(&self) -> crate::ir::substrate::ScratchBytesRef {
        self.score_bytes
    }
    pub const fn pv_offset(&self) -> crate::ir::substrate::ScratchOffsetRef {
        self.pv_offset
    }
    pub const fn pv_bytes(&self) -> crate::ir::substrate::ScratchBytesRef {
        self.pv_bytes
    }
    pub const fn k_smem_offset(&self) -> crate::ir::substrate::ScratchOffsetRef {
        self.k_smem_offset
    }
    pub const fn k_smem_bytes(&self) -> crate::ir::substrate::ScratchBytesRef {
        self.k_smem_bytes
    }
    pub const fn v_smem_offset(&self) -> crate::ir::substrate::ScratchOffsetRef {
        self.v_smem_offset
    }
    pub const fn v_smem_bytes(&self) -> crate::ir::substrate::ScratchBytesRef {
        self.v_smem_bytes
    }
    pub const fn consumer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.storer_phase
    }
    pub const fn iters(&self) -> crate::ir::substrate::IterCountRef {
        self.iters
    }
    pub const fn kv_cache_layer(&self) -> crate::ir::substrate::LayerRef {
        self.kv_cache_layer
    }
    pub const fn head_dim(&self) -> crate::ir::substrate::HeadDimRef {
        self.head_dim
    }
    pub const fn num_q_heads(&self) -> crate::ir::substrate::NumQHeadsRef {
        self.num_q_heads
    }
    pub const fn num_kv_heads(&self) -> crate::ir::substrate::NumKvHeadsRef {
        self.num_kv_heads
    }
    pub const fn block_size(&self) -> crate::ir::substrate::BlockSizeRef {
        self.block_size
    }
    pub const fn num_tokens(&self) -> crate::ir::substrate::NumTokensRef {
        self.num_tokens
    }
    pub const fn max_sk(&self) -> crate::ir::substrate::MaxSkRef {
        self.max_sk
    }
    pub const fn q_in_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.q_in_act_slot
    }
    pub const fn attn_out_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.attn_out_act_slot
    }
    pub fn attn_scale(&self) -> FiniteF32 {
        self.attn_scale
    }
    pub fn attn_softcap(&self) -> FiniteF32 {
        self.attn_softcap
    }
}

/// `BarrierSignal` variant.
pub struct TkBarrierSignal {
    edge: crate::ir::substrate::EdgeIdRef,
}

impl TkBarrierSignal {
    pub fn new<const IDX: u32, const NUM_EDGES: u32>() -> Self {
        Self {
            edge: crate::ir::substrate::EdgeId::<IDX, NUM_EDGES>::new().erase(),
        }
    }

    pub const fn edge(&self) -> crate::ir::substrate::EdgeIdRef {
        self.edge
    }
}

/// `BarrierWait` variant.
pub struct TkBarrierWait {
    edge: crate::ir::substrate::EdgeIdRef,
    expected: crate::ir::substrate::ExpectedCountRef,
}

impl TkBarrierWait {
    pub fn new<const IDX: u32, const COUNT: u32, const NUM_EDGES: u32>() -> Self {
        Self {
            edge: crate::ir::substrate::EdgeId::<IDX, NUM_EDGES>::new().erase(),
            expected: crate::ir::substrate::ExpectedCount::<COUNT>::new().erase(),
        }
    }

    pub const fn edge(&self) -> crate::ir::substrate::EdgeIdRef {
        self.edge
    }
    pub const fn expected(&self) -> crate::ir::substrate::ExpectedCountRef {
        self.expected
    }
}

/// The typed lowered `SpliceMmEmbeds` variant — multimodal
/// placeholder splice. The kernel D2D-copies projected vision
/// embeddings into the placeholder positions of an in-flight
/// activation page; substrate shape is one in-place page touch.
///
/// AST shape: per-row D2D copy with `<HIDDEN_DIM, NUM_TOKENS>` shape
/// and the target activation slot.
pub struct TkSpliceMmEmbeds {
    slot: crate::ir::substrate::PageRef,
    consumer_phase: crate::ir::substrate::MbarrierPhaseRef,
    storer_phase: crate::ir::substrate::MbarrierPhaseRef,
    hidden_dim: crate::ir::substrate::HiddenDimRef,
    num_tokens: crate::ir::substrate::NumTokensRef,
    target_act_slot: crate::ir::substrate::ActSlotRef,
}

impl TkSpliceMmEmbeds {
    #[allow(clippy::too_many_arguments)]
    pub fn new<
        const SLOT_ID: u32,
        const CONSUMER_PHASE: u32,
        const STORER_PHASE: u32,
        const NUM_PAGES: u32,
        const ARRIVES: u32,
        const HIDDEN_DIM: u32,
        const NUM_TOKENS: u32,
        const TARGET_ACT_SLOT: u32,
    >() -> Self {
        const {
            assert!(SLOT_ID < NUM_PAGES, "SpliceMmEmbeds: SLOT_ID out of bounds");
            assert!(
                CONSUMER_PHASE == ARRIVES & 1,
                "SpliceMmEmbeds: CONSUMER_PHASE parity mismatch"
            );
            assert!(
                STORER_PHASE == (ARRIVES + 1) & 1,
                "SpliceMmEmbeds: STORER_PHASE parity mismatch"
            );
            assert!(HIDDEN_DIM > 0, "SpliceMmEmbeds: HIDDEN_DIM must be > 0");
            assert!(NUM_TOKENS > 0, "SpliceMmEmbeds: NUM_TOKENS must be > 0");
        }
        use crate::ir::substrate::{
            ActSlotConst, HiddenDim, MbarrierPhase, NumTokensConst, PageId,
        };
        Self {
            slot: PageId::<SLOT_ID, NUM_PAGES>::new().erase(),
            consumer_phase: MbarrierPhase::<CONSUMER_PHASE>::new().erase(),
            storer_phase: MbarrierPhase::<STORER_PHASE>::new().erase(),
            hidden_dim: HiddenDim::<HIDDEN_DIM>::new().erase(),
            num_tokens: NumTokensConst::<NUM_TOKENS>::new().erase(),
            target_act_slot: ActSlotConst::<TARGET_ACT_SLOT, { u32::MAX }>::new().erase(),
        }
    }

    pub const fn slot(&self) -> crate::ir::substrate::PageRef {
        self.slot
    }
    pub const fn consumer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> crate::ir::substrate::MbarrierPhaseRef {
        self.storer_phase
    }
    pub const fn hidden_dim(&self) -> crate::ir::substrate::HiddenDimRef {
        self.hidden_dim
    }
    pub const fn num_tokens(&self) -> crate::ir::substrate::NumTokensRef {
        self.num_tokens
    }
    pub const fn target_act_slot(&self) -> crate::ir::substrate::ActSlotRef {
        self.target_act_slot
    }
}

/// The typed lowered MegaNode enum.
pub enum MegaNode {
    TkRmsNorm(TkRmsNorm),
    TkFusedQkvRopeCache(TkFusedQkvRopeCache),
    TkAdd(TkAdd),
    TkFusedAddRmsNorm(TkFusedAddRmsNorm),
    TkFusedGateUpActivateMul(TkFusedGateUpActivateMul),
    TkEmbed(TkEmbed),
    TkScalarMul(TkScalarMul),
    TkTanhSoftCap(TkTanhSoftCap),
    TkScalarOffsetRmsNorm(TkScalarOffsetRmsNorm),
    TkGemm(TkGemm),
    TkFusedGemmAdd(TkFusedGemmAdd),
    TkFusedNormGemm(TkFusedNormGemm),
    TkAttentionViaCache(TkAttentionViaCacheNode),
    TkBarrierSignal(TkBarrierSignal),
    TkBarrierWait(TkBarrierWait),
    TkSpliceMmEmbeds(TkSpliceMmEmbeds),
}
