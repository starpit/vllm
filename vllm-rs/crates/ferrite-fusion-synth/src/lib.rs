// SPDX-License-Identifier: Apache-2.0
//
// Atom-driven megakernel synthesis. Extracted from
// `ferrite-forward-macro` so non-proc-macro crates (e.g.
// `ferrite-metal-cost-sweep`) can call the synth-pass entry points
// — `synthesize_pre_attn_chunk`, `synthesize_pre_attn_init_chunk`,
// `synthesize_mlp_pre_down_chunk` — to bench the generated kernels.
//
// `ferrite-forward-macro` re-exports these modules so its existing
// `crate::atom::*` / `crate::atom_lib::*` / `crate::fuse_pass::*`
// paths keep working unchanged.

pub mod aot;
pub mod atom;
pub mod atom_lib;
pub mod fuse_pass;
