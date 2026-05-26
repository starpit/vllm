// SPDX-License-Identifier: Apache-2.0
//! Megakernel IR — typed `MegaTape` + substrate primitives + the
//! `MegaTapeBuilder` const-generic API the proc-macro emits calls
//! against.
//!
//! Sealed `Substrate` trait + sealed witnesses on every push fn
//! ensure a `MegaTape` value cannot exist unless the megakernel it
//! lowers to is provably free of page-slot lifecycle, mbarrier
//! phase, scratch overlap, and warp-role bug classes — checked at
//! `rustc` monomorphization time, not runtime.
//!
//! This module exposes pure runtime / IR types. Codegen (Rust
//! TokenStreams from frontend `Instruction`s, CUDA source from
//! `MegaNode`s) lives in sibling [`crate::codegen`] and
//! [`crate::cuda_emit`] modules.

pub mod lower;
pub mod nodes;
pub mod substrate;
pub mod tape;

// Re-export every IR type at the `ir::` level so consumers (and
// the proc-macro emit paths) can write `ferrite_megakernel::ir::X`
// without descending into the per-file submodule. Mirrors the
// crate-root re-exports in `lib.rs`.
pub use lower::{ArriveCount, LowerError, MegaTapeBuilder};
pub use nodes::{
    TkAdd, AttentionKind, TkAttentionViaCacheNode, TkBarrierSignal, TkBarrierWait, TkEmbed, FiniteF32,
    TkFusedAddRmsNorm, TkFusedGateUpActivateMul, TkFusedQkvRopeCache, GateUpActivation, TkGemm,
    LayerIndex, LmHeadNormKind, MatmulShape, MegaNode, TkRmsNorm, RotaryRef, TkScalarMul,
    TkScalarOffsetRmsNorm, SlidingWindow, TkSpliceMmEmbeds, TkTanhSoftCap, TkFusedGemmAdd,
    TkFusedNormGemm, WeightRef,
};
pub use substrate::{
    ActSlotConst, ActSlotRef, ArrivesCount, AttentionScope, BarRef, BarSyncId, BarSyncPair,
    BlockSize, BlockSizeRef, ChunkK, ChunkKRef, DistinctBarPairProof, EdgeId, EdgeIdRef, Empty,
    ExpectedCount, ExpectedCountRef, Filled, GemmScope, HeadDim, HeadDimRef, HiddenDim,
    HiddenDimRef, InPageStagingFits, IntermediateDim, IntermediateDimRef, IsDistinctBarPair, IsLifecycleState,
    IsScratchScope, IsScratchScopePub, IsValidBarSyncId, IsValidHiddenDim, IsValidWarpRole,
    IterCount, IterCountRef, KFull, KFullRef, KOffset, KOffsetRef, LayerRef, MatmulK, MatmulKRef,
    MatmulM, MatmulMRef, MatmulN, MatmulNRef, MaxSk, MaxSkRef, MbarrierPhase, MbarrierPhaseRef,
    MlpScope, NumKvHeads, NumKvHeadsRef, NumQHeads, NumQHeadsRef, NumTokensConst, NumTokensRef,
    Page, PageId, PagePool, PageRef, Produced, ROLE_CONSUMER, ROLE_LAUNCHER, ROLE_LOADER,
    ROLE_STORER, RmsNormScope, RopeScope, ScratchBytesRef, ScratchOffsetRef, ScratchRegion,
    SubstrateBudget, TileN, TileNRef, VocabSize, VocabSizeRef, WarpRoleTag, WeightAccessorConst,
    WeightAccessorRef,
};
pub use tape::{MegaTape, TapeBudget};
