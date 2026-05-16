// SPDX-License-Identifier: Apache-2.0
//! # ferrite-mega-ir — const-generic edition
//!
//! Substrate-aware typed lowered form for ferrite megakernel
//! emission. See `MEGA_IR_PLAN.md` at the worktree root for the
//! contract — what the IR is, what it isn't, what counts as done.
//!
//! ## Compile-time substrate-proof discharge (this edition)
//!
//! Every substrate-proof primitive (page id, scratch region offset/
//! bytes, mbarrier phase, edge id, iter count, layer index) is a
//! **const-generic** type whose `new()` (and method) constructors
//! open `const { assert!(...) }` blocks. Those `assert!`s fire at
//! MONOMORPHIZATION TIME — bad const args → `rustc` E0080 compile
//! error at the call site, not a runtime panic.
//!
//! Per `MEGA_IR_PLAN.md` §1+§4+§8.1: "If it compiles, it runs
//! coherently." — structurally true at the API surface.
//!
//! ## What stays runtime
//!
//! - [`PagePool`] cross-op alias tracking (linear types would be
//!   needed for a truly session-typed pool; deliberate stable-Rust
//!   simplification — runtime panic on cross-op alias).
//! - Helper newtypes (`WeightRef`, `RotaryRef`, `FiniteF32`,
//!   `LmHeadNormKind`, `GateUpActivation`, `AttentionKind`) —
//!   path strings / floats / enums; not numeric primitives that
//!   can be const-generic.
//! - The runtime-walking [`lower`] entry point — STUBBED. Const-
//!   generic constructors require literal const args at the call
//!   site, so a function dispatching on a runtime `Instruction`
//!   can't pass `op.in_slot: u32` (runtime) as a const-generic.
//!   Phase C of the plan replaces this with literal const-arg
//!   emission at proc-macro expansion time.

pub mod lower;
pub mod nodes;
pub mod substrate;
pub mod tape;

pub use lower::{ArriveCount, LowerError, MegaTapeBuilder};
pub use nodes::{
    Add, AttentionKind, AttentionViaCacheNode, BarrierSignal, BarrierWait, CutlassFusedNormGemm,
    Embed, FiniteF32, FusedAddRmsNorm, FusedCublasGemmAdd, FusedGateUpActivateMul,
    FusedQkvRopeCache, GateUpActivation, Gemm, LayerIndex, LmHeadNormKind, MatmulShape, MegaNode,
    RmsNorm, RotaryRef, ScalarMul, ScalarOffsetRmsNorm, SlidingWindow, SpliceMmEmbeds, TanhSoftCap,
    WeightRef,
};
pub use substrate::{
    AttentionScope, EdgeId, Empty, ExpectedCount, Filled, GemmScope, IsLifecycleState,
    IsScratchScope, IsScratchScopePub, IsValidWarpRole, IterCount, MbarrierPhase, MlpScope, Page,
    PageId, PagePool, Produced, ROLE_CONSUMER, ROLE_LAUNCHER, ROLE_LOADER, ROLE_STORER,
    RmsNormScope, RopeScope, ScratchRegion, SubstrateBudget, WarpRoleTag,
};
pub use tape::MegaTape;
