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
pub struct RmsNorm {
    in_page_id: u32,
    weight_page_id: u32,
    partial_offset: u32,
    partial_bytes: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    hidden_dim: u32,
    num_tokens: u32,
    in_act_slot: u32,
    out_act_slot: u32,
    weight_accessor_idx: u32,
    eps: FiniteF32,
    pub weight: WeightRef,
}

impl RmsNorm {
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
    >(
        weight: WeightRef,
        eps: FiniteF32,
    ) -> Self {
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
        }
        Self {
            in_page_id: IN_ID,
            weight_page_id: WEIGHT_ID,
            partial_offset: PARTIAL_OFF,
            partial_bytes: PARTIAL_BYTES,
            consumer_phase: CONSUMER_PHASE,
            storer_phase: STORER_PHASE,
            layer: LAYER,
            hidden_dim: HIDDEN_DIM,
            num_tokens: NUM_TOKENS,
            in_act_slot: IN_ACT_SLOT,
            out_act_slot: OUT_ACT_SLOT,
            weight_accessor_idx: WEIGHT_ACCESSOR_IDX,
            eps,
            weight,
        }
    }

    pub const fn in_page_id(&self) -> u32 {
        self.in_page_id
    }
    pub const fn weight_page_id(&self) -> u32 {
        self.weight_page_id
    }
    pub const fn partial_offset(&self) -> u32 {
        self.partial_offset
    }
    pub const fn partial_bytes(&self) -> u32 {
        self.partial_bytes
    }
    pub const fn consumer_phase(&self) -> u32 {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> u32 {
        self.storer_phase
    }
    pub const fn layer(&self) -> u32 {
        self.layer
    }
    /// Kernel `<Config, HIDDEN_DIM, NUM_TOKENS>` template arg.
    pub const fn hidden_dim(&self) -> u32 {
        self.hidden_dim
    }
    /// Kernel `<Config, HIDDEN_DIM, NUM_TOKENS>` template arg.
    pub const fn num_tokens(&self) -> u32 {
        self.num_tokens
    }
    /// `act_ptrs[in_act_slot]` — kernel input row gmem ptr.
    pub const fn in_act_slot(&self) -> u32 {
        self.in_act_slot
    }
    /// `act_ptrs[out_act_slot]` — kernel output row gmem ptr (storer).
    pub const fn out_act_slot(&self) -> u32 {
        self.out_act_slot
    }
    /// `weight_ptrs[weight_accessor_idx * NUM_LAYERS + layer]` —
    /// flat-table index for the per-layer rms weight.
    pub const fn weight_accessor_idx(&self) -> u32 {
        self.weight_accessor_idx
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
pub struct FusedQkvRopeCache {
    in_page_id: u32,
    qkv_weight_page_id: u32,
    cos_sin_page_id: u32,
    q_out_page_id: u32,
    k_out_page_id: u32,
    v_out_page_id: u32,
    q_rope_offset: u32,
    q_rope_bytes: u32,
    k_rope_offset: u32,
    k_rope_bytes: u32,
    consumer_phase: u32,
    storer_phase: u32,
    iters: u32,
    layer: u32,
    hidden_dim: u32,
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    in_act_slot: u32,
    q_out_act_slot: u32,
    k_out_act_slot: u32,
    v_out_act_slot: u32,
    qkv_weight_accessor_idx: u32,
    rotary_accessor_idx: u32,
    pub qkv_weight: WeightRef,
    pub rotary: RotaryRef,
    pub biased: bool,
    pub interleaved: bool,
}

impl FusedQkvRopeCache {
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
    pub const fn new<
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
        const NUM_PAGES: u32,
        const NUM_LAYERS: u32,
        const SCRATCH_BYTES: u32,
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
        qkv_weight: WeightRef,
        rotary: RotaryRef,
        biased: bool,
        interleaved: bool,
    ) -> Self {
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
            assert!(
                q_end <= SCRATCH_BYTES as u64,
                "FusedQkvRopeCache: Q rope buf out of scratch budget"
            );
            assert!(
                k_end <= SCRATCH_BYTES as u64,
                "FusedQkvRopeCache: K rope buf out of scratch budget"
            );
            // Scratch disjoint.
            assert!(
                q_end <= K_OFF as u64 || k_end <= Q_OFF as u64,
                "FusedQkvRopeCache: Q and K rope bufs overlap within RopeScope"
            );

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
        }
        Self {
            in_page_id: IN_ID,
            qkv_weight_page_id: QKV_ID,
            cos_sin_page_id: COS_SIN_ID,
            q_out_page_id: Q_ID,
            k_out_page_id: K_ID,
            v_out_page_id: V_ID,
            q_rope_offset: Q_OFF,
            q_rope_bytes: Q_BYTES,
            k_rope_offset: K_OFF,
            k_rope_bytes: K_BYTES,
            consumer_phase: CONSUMER_PHASE,
            storer_phase: STORER_PHASE,
            iters: ITERS,
            layer: LAYER,
            hidden_dim: HIDDEN_DIM,
            head_dim: HEAD_DIM,
            num_q_heads: NUM_Q_HEADS,
            num_kv_heads: NUM_KV_HEADS,
            in_act_slot: IN_ACT_SLOT,
            q_out_act_slot: Q_OUT_ACT_SLOT,
            k_out_act_slot: K_OUT_ACT_SLOT,
            v_out_act_slot: V_OUT_ACT_SLOT,
            qkv_weight_accessor_idx: QKV_WEIGHT_ACCESSOR_IDX,
            rotary_accessor_idx: ROTARY_ACCESSOR_IDX,
            qkv_weight,
            rotary,
            biased,
            interleaved,
        }
    }

    pub const fn in_page_id(&self) -> u32 {
        self.in_page_id
    }
    pub const fn qkv_weight_page_id(&self) -> u32 {
        self.qkv_weight_page_id
    }
    pub const fn cos_sin_page_id(&self) -> u32 {
        self.cos_sin_page_id
    }
    pub const fn q_out_page_id(&self) -> u32 {
        self.q_out_page_id
    }
    pub const fn k_out_page_id(&self) -> u32 {
        self.k_out_page_id
    }
    pub const fn v_out_page_id(&self) -> u32 {
        self.v_out_page_id
    }
    pub const fn q_rope_offset(&self) -> u32 {
        self.q_rope_offset
    }
    pub const fn q_rope_bytes(&self) -> u32 {
        self.q_rope_bytes
    }
    pub const fn k_rope_offset(&self) -> u32 {
        self.k_rope_offset
    }
    pub const fn k_rope_bytes(&self) -> u32 {
        self.k_rope_bytes
    }
    pub const fn consumer_phase(&self) -> u32 {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> u32 {
        self.storer_phase
    }
    pub const fn iters(&self) -> u32 {
        self.iters
    }
    pub const fn layer(&self) -> u32 {
        self.layer
    }
    pub const fn hidden_dim(&self) -> u32 {
        self.hidden_dim
    }
    pub const fn head_dim(&self) -> u32 {
        self.head_dim
    }
    pub const fn num_q_heads(&self) -> u32 {
        self.num_q_heads
    }
    pub const fn num_kv_heads(&self) -> u32 {
        self.num_kv_heads
    }
    pub const fn in_act_slot(&self) -> u32 {
        self.in_act_slot
    }
    pub const fn q_out_act_slot(&self) -> u32 {
        self.q_out_act_slot
    }
    pub const fn k_out_act_slot(&self) -> u32 {
        self.k_out_act_slot
    }
    pub const fn v_out_act_slot(&self) -> u32 {
        self.v_out_act_slot
    }
    pub const fn qkv_weight_accessor_idx(&self) -> u32 {
        self.qkv_weight_accessor_idx
    }
    pub const fn rotary_accessor_idx(&self) -> u32 {
        self.rotary_accessor_idx
    }
}

/// The typed lowered `Add` (residual fold) variant.
///
/// Kernel ABI: bf16 elementwise per-row residual add — emit splices a
/// per-row load/add/store loop with `<HIDDEN_DIM, NUM_TOKENS>` shape.
pub struct Add {
    delta_page_id: u32,
    residual_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    hidden_dim: u32,
    num_tokens: u32,
    delta_act_slot: u32,
    residual_act_slot: u32,
}

impl Add {
    #[allow(clippy::too_many_arguments)]
    pub const fn new<
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
    >() -> Self {
        const {
            assert!(DELTA_ID < NUM_PAGES, "Add: DELTA_ID out of bounds");
            assert!(RESIDUAL_ID < NUM_PAGES, "Add: RESIDUAL_ID out of bounds");
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
            assert!(HIDDEN_DIM > 0, "Add: HIDDEN_DIM must be > 0");
            assert!(NUM_TOKENS > 0, "Add: NUM_TOKENS must be > 0");
        }
        Self {
            delta_page_id: DELTA_ID,
            residual_page_id: RESIDUAL_ID,
            consumer_phase: CONSUMER_PHASE,
            storer_phase: STORER_PHASE,
            hidden_dim: HIDDEN_DIM,
            num_tokens: NUM_TOKENS,
            delta_act_slot: DELTA_ACT_SLOT,
            residual_act_slot: RESIDUAL_ACT_SLOT,
        }
    }

    pub const fn delta_page_id(&self) -> u32 {
        self.delta_page_id
    }
    pub const fn residual_page_id(&self) -> u32 {
        self.residual_page_id
    }
    pub const fn consumer_phase(&self) -> u32 {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> u32 {
        self.storer_phase
    }
    pub const fn hidden_dim(&self) -> u32 {
        self.hidden_dim
    }
    pub const fn num_tokens(&self) -> u32 {
        self.num_tokens
    }
    pub const fn delta_act_slot(&self) -> u32 {
        self.delta_act_slot
    }
    pub const fn residual_act_slot(&self) -> u32 {
        self.residual_act_slot
    }
}

/// The typed lowered `FusedAddRmsNorm` variant.
///
/// Codegen inlines the four role bodies directly into the kernel
/// `.cu`. Template args `<HIDDEN_DIM, NUM_TOKENS>` come from typed
/// getters; runtime arg `eps` from `eps()`.
pub struct FusedAddRmsNorm {
    delta_page_id: u32,
    residual_page_id: u32,
    weight_page_id: u32,
    partial_offset: u32,
    partial_bytes: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    hidden_dim: u32,
    num_tokens: u32,
    delta_act_slot: u32,
    residual_act_slot: u32,
    weight_accessor_idx: u32,
    eps: FiniteF32,
    pub weight: WeightRef,
}

impl FusedAddRmsNorm {
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
    >(
        weight: WeightRef,
        eps: FiniteF32,
    ) -> Self {
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
        Self {
            delta_page_id: DELTA_ID,
            residual_page_id: RESIDUAL_ID,
            weight_page_id: WEIGHT_ID,
            partial_offset: PARTIAL_OFF,
            partial_bytes: PARTIAL_BYTES,
            consumer_phase: CONSUMER_PHASE,
            storer_phase: STORER_PHASE,
            layer: LAYER,
            hidden_dim: HIDDEN_DIM,
            num_tokens: NUM_TOKENS,
            delta_act_slot: DELTA_ACT_SLOT,
            residual_act_slot: RESIDUAL_ACT_SLOT,
            weight_accessor_idx: WEIGHT_ACCESSOR_IDX,
            eps,
            weight,
        }
    }

    pub const fn delta_page_id(&self) -> u32 {
        self.delta_page_id
    }
    pub const fn residual_page_id(&self) -> u32 {
        self.residual_page_id
    }
    pub const fn weight_page_id(&self) -> u32 {
        self.weight_page_id
    }
    pub const fn partial_offset(&self) -> u32 {
        self.partial_offset
    }
    pub const fn partial_bytes(&self) -> u32 {
        self.partial_bytes
    }
    pub const fn consumer_phase(&self) -> u32 {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> u32 {
        self.storer_phase
    }
    pub const fn layer(&self) -> u32 {
        self.layer
    }
    pub const fn hidden_dim(&self) -> u32 {
        self.hidden_dim
    }
    pub const fn num_tokens(&self) -> u32 {
        self.num_tokens
    }
    pub const fn delta_act_slot(&self) -> u32 {
        self.delta_act_slot
    }
    pub const fn residual_act_slot(&self) -> u32 {
        self.residual_act_slot
    }
    pub const fn weight_accessor_idx(&self) -> u32 {
        self.weight_accessor_idx
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
pub struct FusedGateUpActivateMul {
    in_page_id: u32,
    gate_up_weight_page_id: u32,
    out_page_id: u32,
    gate_offset: u32,
    gate_bytes: u32,
    up_offset: u32,
    up_bytes: u32,
    consumer_phase: u32,
    storer_phase: u32,
    iters: u32,
    layer: u32,
    hidden_dim: u32,
    intermediate_dim: u32,
    num_tokens: u32,
    in_act_slot: u32,
    out_act_slot: u32,
    weight_accessor_idx: u32,
    pub weight: WeightRef,
    pub activation: GateUpActivation,
}

impl FusedGateUpActivateMul {
    #[allow(clippy::too_many_arguments)]
    pub const fn new<
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
    >(
        weight: WeightRef,
        activation: GateUpActivation,
    ) -> Self {
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
        }
        Self {
            in_page_id: IN_ID,
            gate_up_weight_page_id: WEIGHT_ID,
            out_page_id: OUT_ID,
            gate_offset: GATE_OFF,
            gate_bytes: GATE_BYTES,
            up_offset: UP_OFF,
            up_bytes: UP_BYTES,
            consumer_phase: CONSUMER_PHASE,
            storer_phase: STORER_PHASE,
            iters: ITERS,
            layer: LAYER,
            hidden_dim: HIDDEN_DIM,
            intermediate_dim: INTERMEDIATE_DIM,
            num_tokens: NUM_TOKENS,
            in_act_slot: IN_ACT_SLOT,
            out_act_slot: OUT_ACT_SLOT,
            weight_accessor_idx: WEIGHT_ACCESSOR_IDX,
            weight,
            activation,
        }
    }

    pub const fn in_page_id(&self) -> u32 {
        self.in_page_id
    }
    pub const fn gate_up_weight_page_id(&self) -> u32 {
        self.gate_up_weight_page_id
    }
    pub const fn out_page_id(&self) -> u32 {
        self.out_page_id
    }
    pub const fn gate_offset(&self) -> u32 {
        self.gate_offset
    }
    pub const fn gate_bytes(&self) -> u32 {
        self.gate_bytes
    }
    pub const fn up_offset(&self) -> u32 {
        self.up_offset
    }
    pub const fn up_bytes(&self) -> u32 {
        self.up_bytes
    }
    pub const fn consumer_phase(&self) -> u32 {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> u32 {
        self.storer_phase
    }
    pub const fn iters(&self) -> u32 {
        self.iters
    }
    pub const fn layer(&self) -> u32 {
        self.layer
    }
    pub const fn hidden_dim(&self) -> u32 {
        self.hidden_dim
    }
    pub const fn intermediate_dim(&self) -> u32 {
        self.intermediate_dim
    }
    pub const fn num_tokens(&self) -> u32 {
        self.num_tokens
    }
    pub const fn in_act_slot(&self) -> u32 {
        self.in_act_slot
    }
    pub const fn out_act_slot(&self) -> u32 {
        self.out_act_slot
    }
    pub const fn weight_accessor_idx(&self) -> u32 {
        self.weight_accessor_idx
    }
}

/// `Embed` (vocab table lookup) variant.
///
/// Codegen inlines the role bodies; loader pulls one row of
/// `weight_ptrs[weight_accessor_idx * NUM_LAYERS + 0]` per token
/// (LAYER is always 0 for Embed). The vocab table is sized
/// `VOCAB_SIZE × HIDDEN_DIM`. Loader needs `input_ids` (uint32_t*)
/// as a kernel-level extra ptr (see `KernelExtras::needs_input_ids`).
pub struct Embed {
    out_page_id: u32,
    embed_weight_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    hidden_dim: u32,
    num_tokens: u32,
    vocab_size: u32,
    out_act_slot: u32,
    weight_accessor_idx: u32,
    pub embed_weight: WeightRef,
}

impl Embed {
    #[allow(clippy::too_many_arguments)]
    pub const fn new<
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
            assert!(OUT_ID < NUM_PAGES, "Embed: OUT_ID OOB");
            assert!(WEIGHT_ID < NUM_PAGES, "Embed: WEIGHT_ID OOB");
            assert!(OUT_ID != WEIGHT_ID, "Embed: page alias");
            assert!(
                CONSUMER_PHASE == ARRIVES & 1,
                "Embed: CONSUMER_PHASE parity"
            );
            assert!(
                STORER_PHASE == (ARRIVES + 1) & 1,
                "Embed: STORER_PHASE parity"
            );
            assert!(HIDDEN_DIM > 0, "Embed: HIDDEN_DIM must be > 0");
            assert!(NUM_TOKENS > 0, "Embed: NUM_TOKENS must be > 0");
            assert!(VOCAB_SIZE > 0, "Embed: VOCAB_SIZE must be > 0");
        }
        Self {
            out_page_id: OUT_ID,
            embed_weight_page_id: WEIGHT_ID,
            consumer_phase: CONSUMER_PHASE,
            storer_phase: STORER_PHASE,
            hidden_dim: HIDDEN_DIM,
            num_tokens: NUM_TOKENS,
            vocab_size: VOCAB_SIZE,
            out_act_slot: OUT_ACT_SLOT,
            weight_accessor_idx: WEIGHT_ACCESSOR_IDX,
            embed_weight,
        }
    }

    pub const fn out_page_id(&self) -> u32 {
        self.out_page_id
    }
    pub const fn embed_weight_page_id(&self) -> u32 {
        self.embed_weight_page_id
    }
    pub const fn consumer_phase(&self) -> u32 {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> u32 {
        self.storer_phase
    }
    pub const fn hidden_dim(&self) -> u32 {
        self.hidden_dim
    }
    pub const fn num_tokens(&self) -> u32 {
        self.num_tokens
    }
    pub const fn vocab_size(&self) -> u32 {
        self.vocab_size
    }
    pub const fn out_act_slot(&self) -> u32 {
        self.out_act_slot
    }
    pub const fn weight_accessor_idx(&self) -> u32 {
        self.weight_accessor_idx
    }
}

/// `ScalarMul` variant.
///
/// Kernel ABI: bf16 elementwise per-row scale — emit splices a per-row
/// load/mul/store loop with `<HIDDEN_DIM, NUM_TOKENS>` shape.
pub struct ScalarMul {
    in_page_id: u32,
    out_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    hidden_dim: u32,
    num_tokens: u32,
    in_act_slot: u32,
    out_act_slot: u32,
    pub scale: FiniteF32,
}

impl ScalarMul {
    #[allow(clippy::too_many_arguments)]
    pub const fn new<
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
    >(
        scale: FiniteF32,
    ) -> Self {
        const {
            assert!(IN_ID < NUM_PAGES, "ScalarMul: IN_ID OOB");
            assert!(OUT_ID < NUM_PAGES, "ScalarMul: OUT_ID OOB");
            // ScalarMul is elementwise; in-place (IN_ID == OUT_ID) is
            // a valid substrate pattern (gemma2 post-attn `* hidden`).
            assert!(
                CONSUMER_PHASE == ARRIVES & 1,
                "ScalarMul: CONSUMER_PHASE parity"
            );
            assert!(
                STORER_PHASE == (ARRIVES + 1) & 1,
                "ScalarMul: STORER_PHASE parity"
            );
            assert!(HIDDEN_DIM > 0, "ScalarMul: HIDDEN_DIM must be > 0");
            assert!(NUM_TOKENS > 0, "ScalarMul: NUM_TOKENS must be > 0");
        }
        Self {
            in_page_id: IN_ID,
            out_page_id: OUT_ID,
            consumer_phase: CONSUMER_PHASE,
            storer_phase: STORER_PHASE,
            hidden_dim: HIDDEN_DIM,
            num_tokens: NUM_TOKENS,
            in_act_slot: IN_ACT_SLOT,
            out_act_slot: OUT_ACT_SLOT,
            scale,
        }
    }

    pub const fn in_page_id(&self) -> u32 {
        self.in_page_id
    }
    pub const fn out_page_id(&self) -> u32 {
        self.out_page_id
    }
    pub const fn consumer_phase(&self) -> u32 {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> u32 {
        self.storer_phase
    }
    pub const fn hidden_dim(&self) -> u32 {
        self.hidden_dim
    }
    pub const fn num_tokens(&self) -> u32 {
        self.num_tokens
    }
    pub const fn in_act_slot(&self) -> u32 {
        self.in_act_slot
    }
    pub const fn out_act_slot(&self) -> u32 {
        self.out_act_slot
    }
}

/// `TanhSoftCap` variant. Same shape as `ScalarMul` plus a runtime
/// `cap` value. Kernel: emit splices a per-row load/tanh-cap/store
/// loop with `<HIDDEN_DIM, NUM_TOKENS>` shape and the runtime cap
/// (Gemma2 final-logit softcap; 0.0 = identity for arches without).
pub struct TanhSoftCap {
    in_page_id: u32,
    out_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    hidden_dim: u32,
    num_tokens: u32,
    in_act_slot: u32,
    out_act_slot: u32,
    pub cap: FiniteF32,
}

impl TanhSoftCap {
    #[allow(clippy::too_many_arguments)]
    pub const fn new<
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
    >(
        cap: FiniteF32,
    ) -> Self {
        const {
            assert!(IN_ID < NUM_PAGES, "TanhSoftCap: IN_ID OOB");
            assert!(OUT_ID < NUM_PAGES, "TanhSoftCap: OUT_ID OOB");
            // TanhSoftCap is elementwise; in-place (IN_ID == OUT_ID) is
            // a valid substrate pattern (gemma2 final logit cap).
            assert!(
                CONSUMER_PHASE == ARRIVES & 1,
                "TanhSoftCap: CONSUMER_PHASE parity"
            );
            assert!(
                STORER_PHASE == (ARRIVES + 1) & 1,
                "TanhSoftCap: STORER_PHASE parity"
            );
            assert!(HIDDEN_DIM > 0, "TanhSoftCap: HIDDEN_DIM must be > 0");
            assert!(NUM_TOKENS > 0, "TanhSoftCap: NUM_TOKENS must be > 0");
        }
        Self {
            in_page_id: IN_ID,
            out_page_id: OUT_ID,
            consumer_phase: CONSUMER_PHASE,
            storer_phase: STORER_PHASE,
            hidden_dim: HIDDEN_DIM,
            num_tokens: NUM_TOKENS,
            in_act_slot: IN_ACT_SLOT,
            out_act_slot: OUT_ACT_SLOT,
            cap,
        }
    }

    pub const fn in_page_id(&self) -> u32 {
        self.in_page_id
    }
    pub const fn out_page_id(&self) -> u32 {
        self.out_page_id
    }
    pub const fn consumer_phase(&self) -> u32 {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> u32 {
        self.storer_phase
    }
    pub const fn hidden_dim(&self) -> u32 {
        self.hidden_dim
    }
    pub const fn num_tokens(&self) -> u32 {
        self.num_tokens
    }
    pub const fn in_act_slot(&self) -> u32 {
        self.in_act_slot
    }
    pub const fn out_act_slot(&self) -> u32 {
        self.out_act_slot
    }
}

/// `ScalarOffsetRmsNorm` variant.
///
/// Codegen inlines the role bodies. Same TK + substrate primitives
/// as RmsNorm plus a `float offset` runtime arg in the consumer's
/// scale-multiply step.
pub struct ScalarOffsetRmsNorm {
    in_page_id: u32,
    weight_page_id: u32,
    partial_offset: u32,
    partial_bytes: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    hidden_dim: u32,
    num_tokens: u32,
    in_act_slot: u32,
    out_act_slot: u32,
    weight_accessor_idx: u32,
    eps: FiniteF32,
    pub weight: WeightRef,
    pub offset: FiniteF32,
}

impl ScalarOffsetRmsNorm {
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
    >(
        weight: WeightRef,
        offset: FiniteF32,
        eps: FiniteF32,
    ) -> Self {
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
        Self {
            in_page_id: IN_ID,
            weight_page_id: WEIGHT_ID,
            partial_offset: PARTIAL_OFF,
            partial_bytes: PARTIAL_BYTES,
            consumer_phase: CONSUMER_PHASE,
            storer_phase: STORER_PHASE,
            layer: LAYER,
            hidden_dim: HIDDEN_DIM,
            num_tokens: NUM_TOKENS,
            in_act_slot: IN_ACT_SLOT,
            out_act_slot: OUT_ACT_SLOT,
            weight_accessor_idx: WEIGHT_ACCESSOR_IDX,
            eps,
            weight,
            offset,
        }
    }

    pub const fn in_page_id(&self) -> u32 {
        self.in_page_id
    }
    pub const fn weight_page_id(&self) -> u32 {
        self.weight_page_id
    }
    pub const fn partial_offset(&self) -> u32 {
        self.partial_offset
    }
    pub const fn partial_bytes(&self) -> u32 {
        self.partial_bytes
    }
    pub const fn consumer_phase(&self) -> u32 {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> u32 {
        self.storer_phase
    }
    pub const fn layer(&self) -> u32 {
        self.layer
    }
    pub const fn hidden_dim(&self) -> u32 {
        self.hidden_dim
    }
    pub const fn num_tokens(&self) -> u32 {
        self.num_tokens
    }
    pub const fn in_act_slot(&self) -> u32 {
        self.in_act_slot
    }
    pub const fn out_act_slot(&self) -> u32 {
        self.out_act_slot
    }
    pub const fn weight_accessor_idx(&self) -> u32 {
        self.weight_accessor_idx
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
pub struct Gemm {
    in_page_id: u32,
    weight_page_id: u32,
    out_page_id: u32,
    b_tile_offset: u32,
    b_tile_bytes: u32,
    consumer_phase: u32,
    storer_phase: u32,
    iters: u32,
    layer: u32,
    n: u32,
    k: u32,
    m: u32,
    in_act_slot: u32,
    out_act_slot: u32,
    weight_accessor_idx: u32,
    pub weight: WeightRef,
}

impl Gemm {
    #[allow(clippy::too_many_arguments)]
    pub const fn new<
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
    >(
        weight: WeightRef,
    ) -> Self {
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
        }
        Self {
            in_page_id: IN_ID,
            weight_page_id: WEIGHT_ID,
            out_page_id: OUT_ID,
            b_tile_offset: B_TILE_OFF,
            b_tile_bytes: B_TILE_BYTES,
            consumer_phase: CONSUMER_PHASE,
            storer_phase: STORER_PHASE,
            iters: ITERS,
            layer: LAYER,
            n: N,
            k: K,
            m: M,
            in_act_slot: IN_ACT_SLOT,
            out_act_slot: OUT_ACT_SLOT,
            weight_accessor_idx: WEIGHT_ACCESSOR_IDX,
            weight,
        }
    }

    pub const fn in_page_id(&self) -> u32 {
        self.in_page_id
    }
    pub const fn weight_page_id(&self) -> u32 {
        self.weight_page_id
    }
    pub const fn out_page_id(&self) -> u32 {
        self.out_page_id
    }
    pub const fn b_tile_offset(&self) -> u32 {
        self.b_tile_offset
    }
    pub const fn b_tile_bytes(&self) -> u32 {
        self.b_tile_bytes
    }
    pub const fn consumer_phase(&self) -> u32 {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> u32 {
        self.storer_phase
    }
    pub const fn iters(&self) -> u32 {
        self.iters
    }
    pub const fn layer(&self) -> u32 {
        self.layer
    }
    pub const fn n(&self) -> u32 {
        self.n
    }
    pub const fn k(&self) -> u32 {
        self.k
    }
    pub const fn m(&self) -> u32 {
        self.m
    }
    pub const fn in_act_slot(&self) -> u32 {
        self.in_act_slot
    }
    pub const fn out_act_slot(&self) -> u32 {
        self.out_act_slot
    }
    pub const fn weight_accessor_idx(&self) -> u32 {
        self.weight_accessor_idx
    }
}

/// `Instruction::FusedCublasGemmAdd(in, residual, layer, n, k)` —
/// `residual += gemm(in, weight[layer])` in place. Substrate shape
/// is `Gemm` plus a `residual_page` that's read AND written
/// (the output writes back to the residual buffer; no separate
/// out_page).
///
/// Codegen inlines the role bodies. Template
/// `<K, N, NUM_TOKENS, K_OFFSET, K_FULL>`. K_OFFSET / K_FULL
/// support the 4-chunk down_proj split (TkGemmAdd path); the
/// un-split case has K_OFFSET = 0, K_FULL = K.
pub struct FusedCublasGemmAdd {
    in_page_id: u32,
    weight_page_id: u32,
    residual_page_id: u32,
    b_tile_offset: u32,
    b_tile_bytes: u32,
    consumer_phase: u32,
    storer_phase: u32,
    iters: u32,
    layer: u32,
    n: u32,
    k: u32,
    num_tokens: u32,
    k_offset: u32,
    k_full: u32,
    in_act_slot: u32,
    residual_act_slot: u32,
    weight_accessor_idx: u32,
    pub weight: WeightRef,
}

impl FusedCublasGemmAdd {
    #[allow(clippy::too_many_arguments)]
    pub const fn new<
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
    >(
        weight: WeightRef,
    ) -> Self {
        const {
            assert!(IN_ID < NUM_PAGES, "FusedCublasGemmAdd: IN_ID OOB");
            assert!(WEIGHT_ID < NUM_PAGES, "FusedCublasGemmAdd: WEIGHT_ID OOB");
            assert!(
                RESIDUAL_ID < NUM_PAGES,
                "FusedCublasGemmAdd: RESIDUAL_ID OOB"
            );
            assert!(
                IN_ID != WEIGHT_ID && IN_ID != RESIDUAL_ID && WEIGHT_ID != RESIDUAL_ID,
                "FusedCublasGemmAdd: page alias"
            );
            let end = (B_TILE_OFF as u64) + (B_TILE_BYTES as u64);
            assert!(
                end <= SCRATCH_BYTES as u64,
                "FusedCublasGemmAdd: b_tile OOB scratch budget"
            );
            assert!(ITERS > 0, "FusedCublasGemmAdd: ITERS must be > 0");
            assert!(LAYER < NUM_LAYERS, "FusedCublasGemmAdd: LAYER OOB");
            assert!(N > 0, "FusedCublasGemmAdd: N must be > 0");
            assert!(K > 0, "FusedCublasGemmAdd: K must be > 0");
            assert!(
                CONSUMER_PHASE == ARRIVES & 1,
                "FusedCublasGemmAdd: CONSUMER_PHASE parity"
            );
            assert!(
                STORER_PHASE == (ARRIVES + 1) & 1,
                "FusedCublasGemmAdd: STORER_PHASE parity"
            );
            assert!(
                NUM_TOKENS > 0,
                "FusedCublasGemmAdd: NUM_TOKENS must be > 0"
            );
            assert!(K_FULL > 0, "FusedCublasGemmAdd: K_FULL must be > 0");
            assert!(
                (K_OFFSET as u64) + (K as u64) <= K_FULL as u64,
                "FusedCublasGemmAdd: K_OFFSET + K must be <= K_FULL"
            );
        }
        Self {
            in_page_id: IN_ID,
            weight_page_id: WEIGHT_ID,
            residual_page_id: RESIDUAL_ID,
            b_tile_offset: B_TILE_OFF,
            b_tile_bytes: B_TILE_BYTES,
            consumer_phase: CONSUMER_PHASE,
            storer_phase: STORER_PHASE,
            iters: ITERS,
            layer: LAYER,
            n: N,
            k: K,
            num_tokens: NUM_TOKENS,
            k_offset: K_OFFSET,
            k_full: K_FULL,
            in_act_slot: IN_ACT_SLOT,
            residual_act_slot: RESIDUAL_ACT_SLOT,
            weight_accessor_idx: WEIGHT_ACCESSOR_IDX,
            weight,
        }
    }

    pub const fn in_page_id(&self) -> u32 {
        self.in_page_id
    }
    pub const fn weight_page_id(&self) -> u32 {
        self.weight_page_id
    }
    pub const fn residual_page_id(&self) -> u32 {
        self.residual_page_id
    }
    pub const fn b_tile_offset(&self) -> u32 {
        self.b_tile_offset
    }
    pub const fn b_tile_bytes(&self) -> u32 {
        self.b_tile_bytes
    }
    pub const fn consumer_phase(&self) -> u32 {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> u32 {
        self.storer_phase
    }
    pub const fn iters(&self) -> u32 {
        self.iters
    }
    pub const fn layer(&self) -> u32 {
        self.layer
    }
    pub const fn n(&self) -> u32 {
        self.n
    }
    pub const fn k(&self) -> u32 {
        self.k
    }
    pub const fn num_tokens(&self) -> u32 {
        self.num_tokens
    }
    pub const fn k_offset(&self) -> u32 {
        self.k_offset
    }
    pub const fn k_full(&self) -> u32 {
        self.k_full
    }
    pub const fn in_act_slot(&self) -> u32 {
        self.in_act_slot
    }
    pub const fn residual_act_slot(&self) -> u32 {
        self.residual_act_slot
    }
    pub const fn weight_accessor_idx(&self) -> u32 {
        self.weight_accessor_idx
    }
}

/// `CutlassFusedNormGemm` (lm_head fusion) variant.
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
pub struct CutlassFusedNormGemm {
    in_page_id: u32,
    delta_page_id: Option<u32>,
    norm_weight_page_id: u32,
    linear_weight_page_id: u32,
    out_page_id: u32,
    partial_offset: u32,
    partial_bytes: u32,
    b_tile_offset: u32,
    b_tile_bytes: u32,
    consumer_phase: u32,
    storer_phase: u32,
    iters: u32,
    layer: u32,
    n: u32,
    k: u32,
    num_tokens: u32,
    in_act_slot: u32,
    delta_act_slot: Option<u32>,
    out_act_slot: u32,
    norm_weight_accessor_idx: u32,
    linear_weight_accessor_idx: u32,
    eps: FiniteF32,
    pub norm_weight: WeightRef,
    pub linear_weight: WeightRef,
    pub norm_kind: LmHeadNormKind,
    pub offset: Option<FiniteF32>,
}

impl CutlassFusedNormGemm {
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
    >(
        norm_weight: WeightRef,
        linear_weight: WeightRef,
        norm_kind: LmHeadNormKind,
        eps: FiniteF32,
    ) -> Self {
        const {
            assert!(IN_ID < NUM_PAGES, "CutlassFusedNormGemm: IN_ID OOB");
            assert!(NORM_W_ID < NUM_PAGES, "CutlassFusedNormGemm: NORM_W_ID OOB");
            assert!(LIN_W_ID < NUM_PAGES, "CutlassFusedNormGemm: LIN_W_ID OOB");
            assert!(OUT_ID < NUM_PAGES, "CutlassFusedNormGemm: OUT_ID OOB");
            assert!(
                IN_ID != NORM_W_ID
                    && IN_ID != LIN_W_ID
                    && IN_ID != OUT_ID
                    && NORM_W_ID != LIN_W_ID
                    && NORM_W_ID != OUT_ID
                    && LIN_W_ID != OUT_ID,
                "CutlassFusedNormGemm: page alias"
            );
            let p_end = (PARTIAL_OFF as u64) + (PARTIAL_BYTES as u64);
            let b_end = (B_TILE_OFF as u64) + (B_TILE_BYTES as u64);
            assert!(
                p_end <= SCRATCH_BYTES as u64,
                "CutlassFusedNormGemm: partial_sums OOB"
            );
            assert!(
                b_end <= SCRATCH_BYTES as u64,
                "CutlassFusedNormGemm: b_tile OOB"
            );
            assert!(ITERS > 0, "CutlassFusedNormGemm: ITERS must be > 0");
            assert!(LAYER < NUM_LAYERS, "CutlassFusedNormGemm: LAYER OOB");
            assert!(N > 0, "CutlassFusedNormGemm: N must be > 0");
            assert!(K > 0, "CutlassFusedNormGemm: K must be > 0");
            assert!(
                CONSUMER_PHASE == ARRIVES & 1,
                "CutlassFusedNormGemm: CONSUMER_PHASE parity"
            );
            assert!(
                STORER_PHASE == (ARRIVES + 1) & 1,
                "CutlassFusedNormGemm: STORER_PHASE parity"
            );
            assert!(
                NUM_TOKENS > 0,
                "CutlassFusedNormGemm: NUM_TOKENS must be > 0"
            );
        }
        // Runtime cross-field invariant: norm_kind must NOT carry
        // residual fold or scalar offset for this constructor.
        match norm_kind {
            LmHeadNormKind::RmsNorm | LmHeadNormKind::MeanSubRmsNorm => {}
            LmHeadNormKind::AddRmsNorm | LmHeadNormKind::AddScalarOffsetRmsNorm => {
                panic!(
                    "CutlassFusedNormGemm::new_no_delta: norm_kind requires a delta page; use new_with_delta",
                );
            }
        }
        Self {
            in_page_id: IN_ID,
            delta_page_id: None,
            norm_weight_page_id: NORM_W_ID,
            linear_weight_page_id: LIN_W_ID,
            out_page_id: OUT_ID,
            partial_offset: PARTIAL_OFF,
            partial_bytes: PARTIAL_BYTES,
            b_tile_offset: B_TILE_OFF,
            b_tile_bytes: B_TILE_BYTES,
            consumer_phase: CONSUMER_PHASE,
            storer_phase: STORER_PHASE,
            iters: ITERS,
            layer: LAYER,
            n: N,
            k: K,
            num_tokens: NUM_TOKENS,
            in_act_slot: IN_ACT_SLOT,
            delta_act_slot: None,
            out_act_slot: OUT_ACT_SLOT,
            norm_weight_accessor_idx: NORM_WEIGHT_ACCESSOR_IDX,
            linear_weight_accessor_idx: LINEAR_WEIGHT_ACCESSOR_IDX,
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
    >(
        norm_weight: WeightRef,
        linear_weight: WeightRef,
        norm_kind: LmHeadNormKind,
        offset: Option<FiniteF32>,
        eps: FiniteF32,
    ) -> Self {
        const {
            assert!(IN_ID < NUM_PAGES, "CutlassFusedNormGemm: IN_ID OOB");
            assert!(DELTA_ID < NUM_PAGES, "CutlassFusedNormGemm: DELTA_ID OOB");
            assert!(NORM_W_ID < NUM_PAGES, "CutlassFusedNormGemm: NORM_W_ID OOB");
            assert!(LIN_W_ID < NUM_PAGES, "CutlassFusedNormGemm: LIN_W_ID OOB");
            assert!(OUT_ID < NUM_PAGES, "CutlassFusedNormGemm: OUT_ID OOB");
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
                "CutlassFusedNormGemm: page alias"
            );
            let p_end = (PARTIAL_OFF as u64) + (PARTIAL_BYTES as u64);
            let b_end = (B_TILE_OFF as u64) + (B_TILE_BYTES as u64);
            assert!(
                p_end <= SCRATCH_BYTES as u64,
                "CutlassFusedNormGemm: partial_sums OOB"
            );
            assert!(
                b_end <= SCRATCH_BYTES as u64,
                "CutlassFusedNormGemm: b_tile OOB"
            );
            assert!(ITERS > 0, "CutlassFusedNormGemm: ITERS must be > 0");
            assert!(LAYER < NUM_LAYERS, "CutlassFusedNormGemm: LAYER OOB");
            assert!(N > 0, "CutlassFusedNormGemm: N must be > 0");
            assert!(K > 0, "CutlassFusedNormGemm: K must be > 0");
            assert!(
                CONSUMER_PHASE == ARRIVES & 1,
                "CutlassFusedNormGemm: CONSUMER_PHASE parity"
            );
            assert!(
                STORER_PHASE == (ARRIVES + 1) & 1,
                "CutlassFusedNormGemm: STORER_PHASE parity"
            );
            assert!(
                NUM_TOKENS > 0,
                "CutlassFusedNormGemm: NUM_TOKENS must be > 0"
            );
        }
        match (norm_kind, offset.is_some()) {
            (LmHeadNormKind::AddScalarOffsetRmsNorm, true)
            | (LmHeadNormKind::AddRmsNorm, false) => {}
            (LmHeadNormKind::AddScalarOffsetRmsNorm, false) => {
                panic!(
                    "CutlassFusedNormGemm::new_with_delta: AddScalarOffsetRmsNorm requires Some(offset)"
                );
            }
            (LmHeadNormKind::AddRmsNorm, true) => {
                panic!("CutlassFusedNormGemm::new_with_delta: AddRmsNorm must not carry an offset");
            }
            (kind, _) => {
                let _ = kind;
                panic!("CutlassFusedNormGemm::new_with_delta: norm_kind cannot carry a delta page");
            }
        }
        Self {
            in_page_id: IN_ID,
            delta_page_id: Some(DELTA_ID),
            norm_weight_page_id: NORM_W_ID,
            linear_weight_page_id: LIN_W_ID,
            out_page_id: OUT_ID,
            partial_offset: PARTIAL_OFF,
            partial_bytes: PARTIAL_BYTES,
            b_tile_offset: B_TILE_OFF,
            b_tile_bytes: B_TILE_BYTES,
            consumer_phase: CONSUMER_PHASE,
            storer_phase: STORER_PHASE,
            iters: ITERS,
            layer: LAYER,
            n: N,
            k: K,
            num_tokens: NUM_TOKENS,
            in_act_slot: IN_ACT_SLOT,
            delta_act_slot: Some(DELTA_ACT_SLOT),
            out_act_slot: OUT_ACT_SLOT,
            norm_weight_accessor_idx: NORM_WEIGHT_ACCESSOR_IDX,
            linear_weight_accessor_idx: LINEAR_WEIGHT_ACCESSOR_IDX,
            eps,
            norm_weight,
            linear_weight,
            norm_kind,
            offset,
        }
    }

    pub const fn in_page_id(&self) -> u32 {
        self.in_page_id
    }
    pub const fn delta_page_id(&self) -> Option<u32> {
        self.delta_page_id
    }
    pub const fn norm_weight_page_id(&self) -> u32 {
        self.norm_weight_page_id
    }
    pub const fn linear_weight_page_id(&self) -> u32 {
        self.linear_weight_page_id
    }
    pub const fn out_page_id(&self) -> u32 {
        self.out_page_id
    }
    pub const fn partial_offset(&self) -> u32 {
        self.partial_offset
    }
    pub const fn partial_bytes(&self) -> u32 {
        self.partial_bytes
    }
    pub const fn b_tile_offset(&self) -> u32 {
        self.b_tile_offset
    }
    pub const fn b_tile_bytes(&self) -> u32 {
        self.b_tile_bytes
    }
    pub const fn consumer_phase(&self) -> u32 {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> u32 {
        self.storer_phase
    }
    pub const fn iters(&self) -> u32 {
        self.iters
    }
    pub const fn layer(&self) -> u32 {
        self.layer
    }
    pub const fn n(&self) -> u32 {
        self.n
    }
    pub const fn k(&self) -> u32 {
        self.k
    }
    pub const fn num_tokens(&self) -> u32 {
        self.num_tokens
    }
    pub const fn in_act_slot(&self) -> u32 {
        self.in_act_slot
    }
    pub const fn delta_act_slot(&self) -> Option<u32> {
        self.delta_act_slot
    }
    pub const fn out_act_slot(&self) -> u32 {
        self.out_act_slot
    }
    pub const fn norm_weight_accessor_idx(&self) -> u32 {
        self.norm_weight_accessor_idx
    }
    pub const fn linear_weight_accessor_idx(&self) -> u32 {
        self.linear_weight_accessor_idx
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
pub struct AttentionViaCacheNode {
    q_in_page_id: u32,
    attn_out_page_id: u32,
    score_offset: u32,
    score_bytes: u32,
    pv_offset: u32,
    pv_bytes: u32,
    consumer_phase: u32,
    storer_phase: u32,
    iters: u32,
    kv_cache_layer: u32,
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    block_size: u32,
    num_tokens: u32,
    max_sk: u32,
    q_in_act_slot: u32,
    attn_out_act_slot: u32,
    attn_scale: FiniteF32,
    attn_softcap: FiniteF32,
    pub interleaved: bool,
    pub kind: AttentionKind,
}

impl AttentionViaCacheNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new<
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
            assert!(
                s_end <= SCRATCH_BYTES as u64,
                "AttentionViaCache: score_tile OOB"
            );
            assert!(
                p_end <= SCRATCH_BYTES as u64,
                "AttentionViaCache: pv_tile OOB"
            );
            assert!(
                s_end <= PV_OFF as u64 || p_end <= SCORE_OFF as u64,
                "AttentionViaCache: score and PV tiles overlap within AttentionScope"
            );
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
        }
        // Runtime: SlidingWindow value > 0 was discharged by the
        // const-generic SlidingWindow<W> primitive; here we just
        // store the runtime u32 carried in AttentionKind::Sliding.
        Self {
            q_in_page_id: Q_IN_ID,
            attn_out_page_id: ATTN_OUT_ID,
            score_offset: SCORE_OFF,
            score_bytes: SCORE_BYTES,
            pv_offset: PV_OFF,
            pv_bytes: PV_BYTES,
            consumer_phase: CONSUMER_PHASE,
            storer_phase: STORER_PHASE,
            iters: ITERS,
            kv_cache_layer: LAYER,
            head_dim: HEAD_DIM,
            num_q_heads: NUM_Q_HEADS,
            num_kv_heads: NUM_KV_HEADS,
            block_size: BLOCK_SIZE,
            num_tokens: NUM_TOKENS,
            max_sk: MAX_SK,
            q_in_act_slot: Q_IN_ACT_SLOT,
            attn_out_act_slot: ATTN_OUT_ACT_SLOT,
            attn_scale,
            attn_softcap,
            interleaved,
            kind,
        }
    }

    pub const fn q_in_page_id(&self) -> u32 {
        self.q_in_page_id
    }
    pub const fn attn_out_page_id(&self) -> u32 {
        self.attn_out_page_id
    }
    pub const fn score_offset(&self) -> u32 {
        self.score_offset
    }
    pub const fn score_bytes(&self) -> u32 {
        self.score_bytes
    }
    pub const fn pv_offset(&self) -> u32 {
        self.pv_offset
    }
    pub const fn pv_bytes(&self) -> u32 {
        self.pv_bytes
    }
    pub const fn consumer_phase(&self) -> u32 {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> u32 {
        self.storer_phase
    }
    pub const fn iters(&self) -> u32 {
        self.iters
    }
    pub const fn kv_cache_layer(&self) -> u32 {
        self.kv_cache_layer
    }
    pub const fn head_dim(&self) -> u32 {
        self.head_dim
    }
    pub const fn num_q_heads(&self) -> u32 {
        self.num_q_heads
    }
    pub const fn num_kv_heads(&self) -> u32 {
        self.num_kv_heads
    }
    pub const fn block_size(&self) -> u32 {
        self.block_size
    }
    pub const fn num_tokens(&self) -> u32 {
        self.num_tokens
    }
    pub const fn max_sk(&self) -> u32 {
        self.max_sk
    }
    pub const fn q_in_act_slot(&self) -> u32 {
        self.q_in_act_slot
    }
    pub const fn attn_out_act_slot(&self) -> u32 {
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
pub struct BarrierSignal {
    edge: u32,
}

impl BarrierSignal {
    pub const fn new<const IDX: u32, const NUM_EDGES: u32>() -> Self {
        const {
            assert!(IDX < NUM_EDGES, "BarrierSignal: IDX OOB");
        }
        Self { edge: IDX }
    }

    pub const fn edge(&self) -> u32 {
        self.edge
    }
}

/// `BarrierWait` variant.
pub struct BarrierWait {
    edge: u32,
    expected: u32,
}

impl BarrierWait {
    pub const fn new<const IDX: u32, const COUNT: u32, const NUM_EDGES: u32>() -> Self {
        const {
            assert!(IDX < NUM_EDGES, "BarrierWait: IDX OOB");
            assert!(COUNT > 0, "BarrierWait: COUNT must be > 0");
        }
        Self {
            edge: IDX,
            expected: COUNT,
        }
    }

    pub const fn edge(&self) -> u32 {
        self.edge
    }
    pub const fn expected(&self) -> u32 {
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
pub struct SpliceMmEmbeds {
    slot_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    hidden_dim: u32,
    num_tokens: u32,
    target_act_slot: u32,
}

impl SpliceMmEmbeds {
    #[allow(clippy::too_many_arguments)]
    pub const fn new<
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
        Self {
            slot_id: SLOT_ID,
            consumer_phase: CONSUMER_PHASE,
            storer_phase: STORER_PHASE,
            hidden_dim: HIDDEN_DIM,
            num_tokens: NUM_TOKENS,
            target_act_slot: TARGET_ACT_SLOT,
        }
    }

    pub const fn slot_id(&self) -> u32 {
        self.slot_id
    }
    pub const fn consumer_phase(&self) -> u32 {
        self.consumer_phase
    }
    pub const fn storer_phase(&self) -> u32 {
        self.storer_phase
    }
    pub const fn hidden_dim(&self) -> u32 {
        self.hidden_dim
    }
    pub const fn num_tokens(&self) -> u32 {
        self.num_tokens
    }
    pub const fn target_act_slot(&self) -> u32 {
        self.target_act_slot
    }
}

/// The typed lowered MegaNode enum.
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
    FusedCublasGemmAdd(FusedCublasGemmAdd),
    CutlassFusedNormGemm(CutlassFusedNormGemm),
    AttentionViaCache(AttentionViaCacheNode),
    BarrierSignal(BarrierSignal),
    BarrierWait(BarrierWait),
    SpliceMmEmbeds(SpliceMmEmbeds),
}
