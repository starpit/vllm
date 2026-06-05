// SPDX-License-Identifier: Apache-2.0
//! `metal_tape` — v1 carcass, post §10 dead-arm scrub.
//!
//! Per SUBTILE_IR_REDESIGN.md §4 commit 10: the v1 Metal-runtime
//! ISA (MetalTape, Dispatch, SubtileInstr, play, validate,
//! MetalTapeBuilder, tile_qmv, PipelineSpec, PipelineInterner,
//! QmvOperands/Shape/KernelInfo, Binding, RegionRef, Grid,
//! ConstValue, FnConst, OpKind, OpDataflow, Executor) is dead —
//! deleted in this commit. `lower_tape_to_tk` + `tk_player` own the
//! GPU emit surface now. The §7 net-surface table targeted ~900 LOC
//! after this scrub; the actual cut is steeper because the runtime
//! is fully gone.
//!
//! What survives, until the parent crate's commit 7 retargets
//! `to_wavefront.rs` to build SubtileIR directly: just the **weight-
//! typing types** that mega.rs and partition.rs still consume —
//! [`BufId`], [`PipeId`], [`FlagId`], [`WeightRole`], [`WeightBundle`],
//! [`WeightLoc`], [`InputKind`], [`BufferRef`] — plus the two row-
//! stride helpers ([`packed_weight_row_bytes`], [`affine_scale_row_bytes`])
//! used by mega's QMV planning.
//!
//! Slated for full deletion alongside `lower.rs` once the proc-macro
//! crate's `to_wavefront.rs` is wired.

#![allow(dead_code)]

// ── Handles ─────────────────────────────────────────────────────────

/// Stable buffer-handle newtype. Pre-§10 it indexed into
/// `MetalTape::buffers`; today it survives only because mega.rs /
/// partition.rs still index logical operands by it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufId(pub u32);

/// Stable pipeline-handle newtype (v1 carcass; survives for ABI parity
/// with mega.rs's QMV planning code).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PipeId(pub u32);

/// A point-to-point sync flag (v1 Wait/Signal handle; carcass).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FlagId(pub u32);

// ── Logical operands ────────────────────────────────────────────────

/// Which tensor of a weight bundle a [`BufferRef::Weight`] names. Mirrors
/// the interpreters' shared `WeightTensor` 1:1 (decode subset) so the
/// compiler's `LoweredCommand`→`BufferRef` map is unambiguous both ways.
/// Backend-neutral. `Weight` is the matrix — the resolver picks the
/// packed-quant vs dense tensor from the layer type, not from this tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightRole {
    /// The weight matrix / norm gain (`WeightTensor::Weight`).
    Weight,
    /// Per-output dense bias (`WeightTensor::Bias`).
    Bias,
    /// Affine dequant scales (`WeightTensor::AffineScales`).
    AffineScales,
    /// Affine dequant biases (`WeightTensor::AffineBiases`).
    AffineBiases,
    /// Affine quant's per-output linear bias (`WeightTensor::AffineLinearBias`).
    AffineLinearBias,
}

/// Which weight-bundle accessor resolves a [`BufferRef::Weight`]. Mirrors
/// the interpreters' shared `WeightBundleKind` (the subset the decode
/// path needs); the Executor maps it to the concrete `WeightAccessors`
/// call. Backend-neutral — both Metal and CUDA resolve through the same
/// accessor trait.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightBundle {
    RmsNorm,
    Embedding,
    LinearLayer,
    /// Rotary cos/sin table bundle (`WeightBundleKind::CosSin`).
    CosSin,
    /// Quantized (affine) token-embedding bundle.
    AffineQuantEmbedding,
}

/// The `WeightAccessors` locator both interpreters use to recover a
/// per-layer weight tensor: the macro-baked `(bucket, op_idx, slot)`
/// triple plus the unrolled `layer`. Plain data; no backend types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightLoc {
    pub layer: u32,
    pub bucket: u32,
    pub op_idx: u32,
    pub slot: u32,
}

/// A runtime per-forward input buffer. Mirrors the interpreters' shared
/// `RuntimeBindingKind` so the compiler maps it 1:1. (Cos/sin are a
/// `WeightBundle::CosSin` weight, not a runtime input.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputKind {
    InputIds,
    Positions,
    SlotMapping,
    CuSeqlensQ,
    SeqUsedK,
    BlockTable,
    /// Paged KV cache K half for `layer` (resolver picks the offset).
    KvCacheK { layer: u32 },
    /// Paged KV cache V half for `layer`.
    KvCacheV { layer: u32 },
    /// `[1]` u32 — the forward's actual `num_tokens`.
    NumTokens,
}

/// What a logical buffer is bound to. Pre-§10 the metal `Executor`
/// resolved each to a `(MTLBuffer, base_offset)` at setup (weights via
/// the per-arch `WeightAccessors`, arena slots from the worker arena,
/// etc.). The runtime is gone; this enum survives as a typed identifier
/// for mega.rs / partition.rs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BufferRef {
    /// A model weight tensor, resolved via the shared `WeightAccessors`
    /// trait: `bundle` picks the accessor, `loc` the per-layer tensor,
    /// `role` which tensor of the bundle (packed / scales / biases / …).
    Weight {
        bundle: WeightBundle,
        role: WeightRole,
        loc: WeightLoc,
    },
    /// A colored arena activation slot.
    ArenaSlot(u32),
    /// A shared scratch buffer (e.g. split-K partials).
    Scratch(u32),
    /// A runtime per-forward input.
    Input(InputKind),
    /// The host-gathered embedded hidden-state row `[1, hidden]` the runtime
    /// supplies as the decode's first activation. Per the plan, embed is a
    /// host gather ("embed-as-source"), NOT a megakernel op and NOT the
    /// embedding weight — so it resolves neither through `WeightAccessors`
    /// nor `RuntimeBindings`. The Metal glue binds it to the per-op forward's
    /// embed-output buffer (the megakernel runs as an alt path right after).
    EmbeddedHidden,
}

// ── Row-stride helpers (used by mega.rs's QMV planning) ─────────────

/// Byte stride of one output row of the packed quantized weight
/// (`[out, in*bits/8]`): `k * bits / 8`. (4-bit, k=2048 ⇒ 1024 B/row.)
pub fn packed_weight_row_bytes(k: u32, bits: u32) -> u64 {
    (k as u64 * bits as u64) / 8
}

/// Byte stride of one output row of the affine scales/biases
/// (`[out, in/group_size]` at `scale_elem` bytes each).
pub fn affine_scale_row_bytes(k: u32, group_size: u32, scale_elem: u64) -> u64 {
    (k / group_size) as u64 * scale_elem
}
