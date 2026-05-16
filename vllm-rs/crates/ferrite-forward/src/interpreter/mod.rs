// SPDX-License-Identifier: Apache-2.0
//! Rust-side runtime for the device-interpreter megakernels emitted
//! by `ferrite-forward-macro/src/interpreter/mega.rs`.
//!
//! Ferrite has two interpreters of the same `Instruction<W>` set:
//!
//! - **Host interpreter** — `crate::instr::Instruction::eval`. Runs
//!   on the CPU; each instruction arm issues a kernel launch
//!   (`kernels::rms_norm`, `cutlass::gemm`, …).
//! - **Device interpreter** — one per-variant `.cu` megakernel,
//!   codegen'd by the macro side under `FERRITE_TK_PLAN.md`. The
//!   same instructions, but inlined into a single kernel body with
//!   four warp-role walkers calling TK 2.0 op primitives.
//!
//! This module exposes the Rust-side ABI the device interpreter
//! expects: a `#[repr(C)]` `LaunchArgs` mirroring the emitted
//! `extern "C" ferrite_<variant>_launch(...)` entry point, plus a
//! thin [`launch`] wrapper. Per-canonical wiring (the extern
//! declarations and call sites) is emitted by the proc-macro.

#![cfg(feature = "cuda")]

pub mod mega;
