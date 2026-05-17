// SPDX-License-Identifier: Apache-2.0
//! # ferrite-megakernel
//!
//! Single home for ALL megakernel logic — typed correct-by-
//! construction substrate IR, plus the codegen modules that emit
//! into both the user's Rust build and the runtime CUDA source.
//! Megakernel-related work is intentionally NOT split across
//! `ferrite-forward` / `ferrite-forward-macro` / a separate
//! mega-codegen crate; everything lives here so the boundary is
//! one rename or one `mv` away.
//!
//! ## Modules
//!
//! - [`ir`] — typed `MegaTape` + substrate primitives + the
//!   `MegaTapeBuilder` const-generic API. Pure runtime / IR types.
//! - [`codegen`] — `Instruction → b.push_*::<...>(...)` Rust
//!   TokenStream emission for the proc-macro's `#[forward]`
//!   expansion. Re-uses `proc-macro2` / `quote` from a regular
//!   library context (those crates aren't proc-macro-only).
//! - [`cuda_emit`] — `MegaNode → .cu source` CUDA string emission
//!   per warp role. Cites every TK 2.0 primitive call to its source
//!   line in `third_party/thunderkittens/include/`.
//!
//! ## Compile-time substrate-proof discharge
//!
//! Every substrate-proof primitive (page id, scratch region offset/
//! bytes, mbarrier phase, edge id, iter count, layer index) is a
//! **const-generic** type whose `new()` constructors open
//! `const { assert!(...) }` blocks. Those `assert!`s fire at
//! MONOMORPHIZATION TIME — bad const args → `rustc` E0080 compile
//! error at the call site, not a runtime panic.
//!
//! "If it compiles, it runs coherently." — structurally true at
//! the API surface.
//!
//! ## What stays runtime
//!
//! - [`ir::substrate::PagePool`] cross-op alias tracking (linear
//!   types would be needed for a truly session-typed pool;
//!   deliberate stable-Rust simplification — runtime panic on
//!   cross-op alias).
//! - Helper newtypes (`WeightRef`, `RotaryRef`, `FiniteF32`,
//!   `LmHeadNormKind`, `GateUpActivation`, `AttentionKind`) —
//!   path strings / floats / enums; not numeric primitives that
//!   can be const-generic.
//! - The runtime-walking [`ir::lower::lower`] entry point —
//!   STUBBED. Const-generic constructors require literal const
//!   args at the call site, so a function dispatching on a runtime
//!   `Instruction` can't pass `op.in_slot: u32` (runtime) as a
//!   const-generic. The proc-macro emits literal const-arg
//!   `b.push_*::<...>(...)` calls at expansion time via
//!   [`codegen::dispatch_instruction_to_push`].

pub mod codegen;
pub mod cuda_emit;
pub mod ir;

// Crate-root re-exports of the most-used IR types so consumers can
// write `ferrite_megakernel::MegaTape` instead of
// `ferrite_megakernel::ir::tape::MegaTape`. Proc-macro emit
// anchors at the explicit `::ferrite_megakernel::ir::*` paths so
// the structure is visible in expanded source.
pub use ir::lower::{ArriveCount, LowerError, MegaTapeBuilder};
pub use ir::nodes::{
    Add, AttentionKind, AttentionViaCacheNode, BarrierSignal, BarrierWait, Embed, FiniteF32,
    FusedAddRmsNorm, FusedGateUpActivateMul, FusedQkvRopeCache, GateUpActivation, Gemm,
    LayerIndex, LmHeadNormKind, MatmulShape, MegaNode, RmsNorm, RotaryRef, ScalarMul,
    ScalarOffsetRmsNorm, SlidingWindow, SpliceMmEmbeds, TanhSoftCap, TkFusedGemmAdd,
    TkFusedNormGemm, WeightRef,
};
pub use ir::substrate::{
    ActSlotConst, ActSlotRef, ArrivesCount, AttentionScope, BarRef, BarSyncId, BarSyncPair,
    BlockSize, BlockSizeRef, ChunkK, ChunkKRef, DistinctBarPairProof, EdgeId, EdgeIdRef, Empty,
    ExpectedCount, ExpectedCountRef, Filled, GemmScope, HeadDim, HeadDimRef, HiddenDim,
    HiddenDimRef, IntermediateDim, IntermediateDimRef, IsDistinctBarPair, IsLifecycleState,
    IsScratchScope, IsScratchScopePub, IsValidBarSyncId, IsValidHiddenDim, IsValidWarpRole,
    IterCount, IterCountRef, KFull, KFullRef, KOffset, KOffsetRef, LayerRef, MatmulK, MatmulKRef,
    MatmulM, MatmulMRef, MatmulN, MatmulNRef, MaxSk, MaxSkRef, MbarrierPhase, MbarrierPhaseRef,
    MlpScope, NumKvHeads, NumKvHeadsRef, NumQHeads, NumQHeadsRef, NumTokensConst, NumTokensRef,
    Page, PageId, PagePool, PageRef, Produced, ROLE_CONSUMER, ROLE_LAUNCHER, ROLE_LOADER,
    ROLE_STORER, RmsNormScope, RopeScope, ScratchBytesRef, ScratchOffsetRef, ScratchRegion,
    SubstrateBudget, TileN, TileNRef, VocabSize, VocabSizeRef, WarpRoleTag, WeightAccessorConst,
    WeightAccessorRef,
};
pub use ir::tape::MegaTape;
