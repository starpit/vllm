// SPDX-License-Identifier: Apache-2.0
//! Instruction interpreters for different backends.
//!
//! This module contains runtime interpreters that execute ferrite-forward
//! instruction tapes on different hardware backends:
//!
//! - CUDA: Direct kernel dispatch via `Instruction::eval()` (in `instr.rs`)
//! - Metal: ICB recording + execution via `MetalExecutor` (in `metal.rs`)
//!
//! Each interpreter walks the instruction tape and dispatches to backend-specific
//! kernel implementations.

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(all(test, feature = "metal"))]
mod metal_tests;

#[cfg(feature = "metal")]
pub use metal::MetalExecutor;
