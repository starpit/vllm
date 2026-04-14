// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]
//! Core logic shared between the `forward!` proc macro and its runtime
//! consumers.
//!
//! The `forward!` macro expands at `cargo build` time by parsing a DSL,
//! building a control-flow graph ([`cfg::Cfg`]), unrolling loops
//! ([`unroll::unroll_tagged`]), lifting the unrolled instruction
//! stream to a [`lowering::tile_graph::TileGraph`]
//! ([`fuf::build_fuf`]), running the constraint solver against an
//! [`lowering::ImplementationLibrary`], and emitting Rust code that
//! calls individual kernels via FFI. This crate hosts the
//! non-proc-macro half of that pipeline so it's available to both
//! `ferrite-macros` (the proc-macro crate) and tests.

#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::useless_vec
)]
pub mod cfg;
pub mod cfg_analysis;
pub mod cpu_golden;
pub mod fuf;
pub mod lowering;
pub mod parse;
pub mod target_profile;
pub mod unroll;
