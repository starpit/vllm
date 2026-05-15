// SPDX-License-Identifier: Apache-2.0
//! Numeric newtypes for the Metal interpreter.
//!
//! Distinguish kinds of integers that share a primitive type but are
//! not interchangeable. Each newtype is
//! `Copy + Clone + Eq + Hash + Debug` plus a `From<inner>` impl so adding
//! one to an existing call site is a one-line wrap, not a refactor.
//!
//! The goal is compile-time prevention of bug classes already hit on
//! ferrite-metal — see `FERRITE_METAL_TYPE_SAFETY_PLAN.md` for the
//! enumerated bugs each newtype guards against.

/// Per-layer index into the model's transformer stack.
///
/// Distinct from `ArenaSlotIdx` (a colored tile-arena slot), from
/// `PhysicalBlockIdx` / `LogicalBlockIdx` (paged-cache block numbers),
/// and from `SeqIdx` (per-batch sequence id). The inner width is `u32`
/// to match the existing `RuntimeBindingKind::KvCacheK { layer: u32 }`
/// arithmetic (`*layer + layer_offset` during loop unrolling).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct LayerId(pub u32);

impl From<u32> for LayerId {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl LayerId {
    pub fn get(self) -> u32 {
        self.0
    }
}

/// Colored arena slot id (the worker's per-shape-class tile arena).
///
/// Distinct from `LayerId` and from raw kernel binding indices. The
/// post-coloring linear-scan reg allocator (`colored_slot_map()` in
/// `ferrite-forward-macro/src/interpreter_codegen.rs`) emits these.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ArenaSlotIdx(pub u32);

impl From<u32> for ArenaSlotIdx {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl ArenaSlotIdx {
    pub fn get(self) -> u32 {
        self.0
    }
}

/// Physical block index inside the paged KV cache pool.
///
/// Distinct from `LogicalBlockIdx`: lookup goes
/// `block_table[seq][logical] → physical`. Confusing the two silently
/// reads or writes the wrong sequence's cache memory.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct PhysicalBlockIdx(pub u32);

impl From<u32> for PhysicalBlockIdx {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

/// Per-sequence logical block index (0..max_blocks_per_seq).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct LogicalBlockIdx(pub u32);

impl From<u32> for LogicalBlockIdx {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

/// Token index inside one paged block (`[0, BLOCK_SIZE)`).
///
/// `u16` is plenty — block sizes are 16/32/64 in practice.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SlotInBlock(pub u16);

impl From<u16> for SlotInBlock {
    fn from(v: u16) -> Self {
        Self(v)
    }
}

/// Per-batch sequence id (0..num_seqs in the forward batch).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SeqIdx(pub u32);

impl From<u32> for SeqIdx {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

/// Q-token index inside a forward call (`[0, total_q)`).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct QTokenIdx(pub u32);

impl From<u32> for QTokenIdx {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

/// The static dispatch upper bound (set per bucket).
///
/// Distinct from `NumTokens` so the `ceil(baseline * n / bucket_m)`
/// axis-scaling math (see
/// `worker::scale_tg_for_num_tokens`) can't reverse its arguments.
/// This catches `FERRITE_METAL_TYPE_SAFETY_PLAN.md` bug #7.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct BucketM(pub u32);

impl From<u32> for BucketM {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl BucketM {
    pub fn get(self) -> u32 {
        self.0
    }
}

/// The actual M of the in-flight forward.
///
/// Always `<= bucket_m` of the active bucket (the bucket picker
/// guarantees this).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct NumTokens(pub u32);

impl From<u32> for NumTokens {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl NumTokens {
    pub fn get(self) -> u32 {
        self.0
    }
}

/// `[[buffer(N)]]` binding index on a kernel function signature.
///
/// Distinct from `ConstSlot` (function-constant index, a different
/// Metal-level concept). Distinct from `ArenaSlotIdx` (worker-arena
/// slot id used to resolve the buffer pointer).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct BindingIdx(pub u8);

impl From<u8> for BindingIdx {
    fn from(v: u8) -> Self {
        Self(v)
    }
}

impl BindingIdx {
    pub fn get(self) -> u8 {
        self.0
    }
}

/// Function-constant slot id (`[[function_constant(N)]]`).
///
/// Re-exported from `ferrite-metal-kernels` so the type lives next to
/// the `ConstantValue` constructor it parameterizes.
pub use ferrite_metal_kernels::specialized_pipeline_cache::ConstSlot;

// ── Per-dim newtypes for kernel function-constant fields ────────────
//
// Each is a transparent newtype wrapping the primitive Metal expects
// (`u32` for `[[function_constant(N)]] constant uint`,
//  `i32` for `… int`,
//  `f32` for `… float`).
//
// Phase 2's per-kernel constants structs use these so swapping
// `head_dim` and `num_q_heads` at a call site requires a visually
// obvious mistake (`HeadDim(W::NUM_Q_HEADS)`) rather than silent
// reordering of two interchangeable `u32`s.

macro_rules! u32_newtype {
    ($($(#[$m:meta])* $name:ident),* $(,)?) => {$(
        $(#[$m])*
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
        pub struct $name(pub u32);
        impl From<u32> for $name { fn from(v: u32) -> Self { Self(v) } }
        impl $name { pub fn get(self) -> u32 { self.0 } }
    )*}
}

macro_rules! i32_newtype {
    ($($(#[$m:meta])* $name:ident),* $(,)?) => {$(
        $(#[$m])*
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
        pub struct $name(pub i32);
        impl From<i32> for $name { fn from(v: i32) -> Self { Self(v) } }
        impl $name { pub fn get(self) -> i32 { self.0 } }
    )*}
}

macro_rules! f32_newtype {
    ($($(#[$m:meta])* $name:ident),* $(,)?) => {$(
        $(#[$m])*
        #[derive(Copy, Clone, PartialEq, Debug)]
        pub struct $name(pub f32);
        impl From<f32> for $name { fn from(v: f32) -> Self { Self(v) } }
        impl $name { pub fn get(self) -> f32 { self.0 } }
    )*}
}

u32_newtype!(
    /// Per-head feature width (`W::HEAD_DIM`).
    HeadDim,
    /// Number of query heads (`W::NUM_Q_HEADS`).
    NumQHeads,
    /// Number of key/value heads (`W::NUM_KV_HEADS`), pre-GQA fan-out.
    NumKvHeads,
    /// Number of `head_dim` positions touched by RoPE (`W::ROT_DIM`).
    /// Equal to `head_dim` for full RoPE, smaller for partial RoPE.
    RotDim,
    /// Tokens per paged KV-cache block (`W::BLOCK_SIZE`).
    BlockSize,
    /// Block-table fanout per sequence (`W::MAX_BLOCKS_PER_SEQ`).
    MaxBlocksPerSeq,
    /// Hidden / Q-projection size (`W::Q_SIZE` — `num_q_heads * head_dim`).
    QSize,
    /// MLP intermediate size (`W::INTERMEDIATE_SIZE`).
    IntermediateSize,
    /// Generic hidden size in elements (used by elementwise kernels —
    /// `AffineEmbed.hidden_size`, `GatherLastToken.row_stride`).
    HiddenSize,
    /// MLX-affine packed-K (input cols) for a qmv / qmm_t dispatch.
    KDim,
    /// MLX-affine output cols / weight rows (qmv / qmm_t `n_v`).
    NDim,
    /// Split-K partition count (for `affine_qmm_t_splitk`).
    SplitK,
    /// Steel attention `[[function_constant(99)]]` debug-mode toggle.
    /// Bound to `0` for production; `>0` selects diagnostic paths.
    AttnDebugMode,
);

i32_newtype!(
    /// MLX-affine `[[function_constant]] constant int` packed-K size.
    /// Same semantic as `KDim`, but the qmv / qmm_t MLX-port shaders
    /// declare the constant as signed `int`.
    KDimI32,
    /// MLX-affine `[[function_constant]] constant int` output cols.
    NDimI32,
    /// MLX-affine `[[function_constant]] constant int` per-call M (=
    /// bucket_m). Distinct from [`BucketM`] because the qmm_t shader
    /// declares the constant as `int`.
    MDimI32,
    /// MLX-affine `[[function_constant]] constant int` split-K
    /// partition stride (`affine_qmm_t_splitk` only).
    KPartitionSizeI32,
);

f32_newtype!(
    /// Pre-softmax scale applied per Q*K dot product
    /// (`W::ATTN_SCALE`, typically `1/sqrt(head_dim)`).
    AttnScale,
    /// RMSNorm epsilon (`W::RMS_NORM_EPS`).
    RmsNormEps,
);
