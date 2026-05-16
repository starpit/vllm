// SPDX-License-Identifier: Apache-2.0
//! # ferrite-mega-ir
//!
//! Substrate-aware typed lowered form for ferrite megakernel
//! emission. See `MEGA_IR_PLAN.md` at the worktree root for the
//! contract — what the IR is, what it isn't, what counts as done.
//!
//! At the proc-macro level, this crate's `lower` function consumes
//! a semantic `Vec<Instruction<W>>` Tape (defined in
//! `ferrite-forward::instr`) and produces a typed `MegaTape`.
//! The output tape's variants carry **substrate-layout proofs** as
//! their load-bearing fields — `Page<State>` typestate,
//! `MbarrierPhase`, `ScratchRegion<Scope>`, `WarpRoleTag<R>`.
//! Constructing such a value is sealed and only succeeds when the
//! substrate's invariants (`num_pages`, `scratch_bytes`, phase
//! parity, role pairing) are satisfied for the input op. Any
//! unsatisfied invariant panics at proc-macro construction time —
//! surfaces as a compile error on the user's `#[forward]`.
//!
//! Status: SKELETON. The `substrate` module's typestate vocabulary
//! is in place and tested; `nodes::MegaNode` is uninhabited
//! (Sprint A lands the first variant, RmsNorm).
//! `tape::MegaTape` and `lower::lower` exist as structural
//! placeholders for the work each sprint adds.
//!
//! ## Forbidden patterns (per `MEGA_IR_PLAN.md` §3, §8)
//!
//! - Variants whose load-bearing fields are field-validity
//!   newtypes (`ActivationSlotId(u32)`, `WeightRef(String)`,
//!   `NormEps(f32)`, etc.) without substrate proofs. Helper
//!   newtypes still ride alongside, but substrate proofs must be
//!   present.
//! - Parallel typed enums for the op set when `Instruction<W>`
//!   already covers it. `MegaNode` shares variant *names* with
//!   `Instruction<W>` but at a different abstraction layer
//!   (substrate proofs as fields), not as a duplicate enum.
//! - Token-stream erasure between typed values. If both ends are
//!   typed, the value travels typed.
//! - "Walker" naming. The previous codegen is dead.

pub mod lower;
pub mod nodes;
pub mod substrate;
pub mod tape;

pub use lower::{ArriveCount, LowerError, MegaTapeBuilder, OpInput, lower};
pub use nodes::{LayerIndex, MegaNode, RmsNorm, WeightRef};
pub use substrate::{
    Empty, Filled, IsLifecycleState, IsScratchScope, IsScratchScopePub, IsValidWarpRole,
    MbarrierPhase, Page, PageId, PagePool, Produced, ROLE_CONSUMER, ROLE_LAUNCHER, ROLE_LOADER,
    ROLE_STORER, RmsNormScope, ScratchRegion, SubstrateBudget, WarpRoleTag,
};
pub use tape::MegaTape;
