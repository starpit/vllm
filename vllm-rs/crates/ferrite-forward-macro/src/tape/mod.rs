// SPDX-License-Identifier: Apache-2.0
//! Tape-level claimers — the second solve in ferrite's two-layer
//! architecture.
//!
//! After the Instruction-level solver (FUF → Tape via the
//! [`crate::impl_lib::ImplementationLibrary`] DP) settles on a
//! tape of backend-specific [`crate::impl_lib::OpInstance`]s, a
//! second solve decides how to run the tape. This module holds the
//! claimers that compete at that stage:
//!
//! - [`host_interp::HostInterpreterTapeClaimer`] — runs any tape
//!   via `Instruction::eval` at runtime; universal fallback. No
//!   compile-time artifacts.
//!
//! The previous `tk_mega::TkMegaTapeClaimer` codegen path was
//! removed when the typed `MegaTape<S>` substrate-aware lowering
//! took over; the `Tk*` op claims now flow through
//! `ferrite_mega_ir::emit` directly.
//!
//! See [`crate::tape_claim`] for the trait + library surface.

pub mod host_interp;
