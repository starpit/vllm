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

/// Number of paged-KV-cache blocks backed by one physical chunk buffer
/// in the metal reactive (chunked) KV pool.
///
/// SINGLE SOURCE OF TRUTH for the chunk granularity. It is consumed in
/// two compilation contexts that MUST agree, or the KV cache corrupts:
///
///   1. **Synth megakernel (macro-expansion time):** `fuse_pass` bakes
///      this as a `constant constexpr uint BLOCKS_PER_CHUNK` literal in
///      the generated `SynthPreAttn` MSL (same mechanism as `BLOCK_SIZE`).
///   2. **Hand-written kernels (pipeline-bake time):** the metal lowering
///      sets this as a `[[function_constant]]` on `rope_append`,
///      `fused_qkv_rope_cache`, `attention_via_cache`, and
///      `attention_*_paged` (via `ids::BlocksPerChunk`).
///
/// The chunked pool allocates `ceil(num_gpu_blocks / BLOCKS_PER_CHUNK)`
/// chunk buffers per layer per K/V; a physical block id `pb` addresses
/// `chunk_table[pb / BLOCKS_PER_CHUNK]` then `pb % BLOCKS_PER_CHUNK`
/// within that chunk (see `PagedKvLayout::chunk_decompose`). `num_gpu_blocks`
/// is rounded to a multiple of this at the source the scheduler shares.
///
/// 128 blocks × block_size(16) = 2048 tokens of KV per chunk. This is
/// the reactive pool's growth granularity AND its minimum resident
/// footprint (one chunk allocated at init, the rest grow on demand).
/// 128 keeps the floor small (a short prompt holds one chunk) while
/// coarse enough that a typical short generation never crosses a chunk
/// boundary — so no per-growth `residency.commit` cost on the hot path.
/// Longer sequences grow one chunk (one commit) per 2048 tokens.
pub const BLOCKS_PER_CHUNK: u32 = 128;
