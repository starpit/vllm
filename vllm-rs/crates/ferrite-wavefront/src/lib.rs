// SPDX-License-Identifier: Apache-2.0
//! ferrite-wavefront — the subtile-wavefront scheduling layer for the
//! persistent decode megakernel.
//!
//! This is a normal library crate (not the proc-macro), so it is
//! unit-testable on any host — the host-first correctness path runs
//! here. The proc-macro crate `ferrite-forward-macro` depends on this
//! crate and, at expansion time, walks its (internal) solved FUF and
//! calls the lowering here to build a [`subtile::SubtileGraph`], then
//! emits GPU tape players. This mirrors how `ferrite-forward-macro`
//! already calls the normal `ferrite-fusion-synth` crate for fusion
//! logic; the subtile IR and `Fuf`/`Assignment` can't live in the
//! proc-macro crate because proc-macro crates export only macros.
//!
//! `cpu_golden` (from `ferrite-forward`) is the host *calculator* the
//! validation computes with — not the correctness *oracle* (that is
//! ferrite-metal non-mega at temp=0).

pub mod fixtures;
#[cfg(feature = "cuda")]
pub mod dispatch;
#[cfg(feature = "cuda")]
pub mod launcher;
pub mod lower;
pub mod mega;
pub mod partition;
pub mod region_schedule;
pub mod routing;
pub mod subtile_ir;
pub mod subtile_tape;
pub mod metal_tape;
pub mod tk_player;
pub mod tk_tape;
