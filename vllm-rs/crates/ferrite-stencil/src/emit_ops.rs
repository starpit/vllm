// SPDX-License-Identifier: Apache-2.0
//! Per-op intrinsic expansion table.
//!
//! [`expand`] maps `(op tag, arch)` → inline CUDA pseudocode for the
//! node body, replacing the stub `tag();` line the sketch emitter
//! would otherwise produce. Pseudocode, not yet compilable: helpers
//! like `tma_load_2d(...)` / `cp_async_128(...)` stand in for the
//! real intrinsic sequences that a later commit will macro-expand.
//!
//! This commit wires the dispatch and fills in `load_q_tile` for both
//! SM90 (TMA) and SM89 (cp.async.ca). Subsequent commits fill in the
//! remaining FA2 ops (load_k/v_tile, qk_matmul, softmax_update,
//! pv_matmul, store_o_tile) and the paged-decode variants.

use std::fmt::Write;

/// Context the expansion sees for each node. Keep this small and
/// additive; the table below is the single source of truth for what
/// a tag compiles to.
#[derive(Debug, Clone, Copy)]
pub struct ExpandCtx<'a> {
    pub arch_name: &'a str,
    /// `+P` for pipeline-source loads (the consumer at iter k reads
    /// from iter k−P, so the loader writes into slot `k+P mod P`).
    /// `0` for non-pipelined nodes.
    pub iter_offset: i32,
    pub pipeline_depth: u32,
}

/// Try to expand `(tag, arch)` into concrete CUDA pseudocode. Returns
/// `None` when no entry exists yet; callers fall back to the stub
/// `tag();` line.
pub fn expand(tag: &str, ctx: &ExpandCtx<'_>) -> Option<String> {
    match tag {
        "load_q_tile" => Some(load_q_tile(ctx)),
        "load_k_tile" => Some(load_kv_tile(ctx, "k")),
        "load_v_tile" => Some(load_kv_tile(ctx, "v")),
        _ => None,
    }
}

fn load_q_tile(ctx: &ExpandCtx<'_>) -> String {
    // `load_q_tile` runs in the preamble (no iter_offset). It reads
    // the Q tile for this CTA's (q_tile, head_group) coordinate into
    // smem once. On SM90 this is a single TMA `cp.async.bulk.tensor`
    // with an mbarrier arrive; on SM89 it's a cp.async.ca loop over
    // the tile's elements.
    let mut s = String::new();
    writeln!(s, "{{").unwrap();
    writeln!(s, "  // load_q_tile: Q[q_tile, head_group, :, :] → smem").unwrap();
    match ctx.arch_name {
        "sm90_fa2" => {
            writeln!(s, "  if (wg == LOADER_WG) {{").unwrap();
            writeln!(s, "    tma_load_2d(smem_q, Q_gmem, q_tile, head_group);").unwrap();
            writeln!(s, "    mbarrier_arrive(&bar_q);").unwrap();
            writeln!(s, "  }}").unwrap();
        }
        "sm89_fa2" => {
            writeln!(s, "  cp_async_128(smem_q, Q_gmem, q_tile, head_group);").unwrap();
            writeln!(s, "  cp_async_commit_group();").unwrap();
        }
        other => {
            writeln!(s, "  // unknown arch {} — fall back to generic load", other).unwrap();
            writeln!(s, "  generic_load(smem_q, Q_gmem, q_tile, head_group);").unwrap();
        }
    }
    // Suppress "unused" for iter_offset / pipeline_depth; both matter
    // for pipelined loads but `load_q_tile` is preamble-only.
    let _ = (ctx.iter_offset, ctx.pipeline_depth);
    write!(s, "}}").unwrap();
    s
}

/// Pipeline-source loads for K and V. Distinguished from `load_q_tile`
/// by two things: (a) they run inside the serial loop, so the smem
/// destination rotates through a `P`-deep ring (`slot = (kv_tile + P)
/// % P`, where `P = pipeline_depth` and the `+P` matches the
/// `iter_offset` the scheduler stamped on the step); (b) they arrive
/// on a per-slot mbarrier / cp.async group so the consumer at iter
/// `kv_tile` can wait on its corresponding slot.
fn load_kv_tile(ctx: &ExpandCtx<'_>, which: &str) -> String {
    debug_assert!(ctx.iter_offset > 0, "pipeline-source load must be +P");
    let p = ctx.pipeline_depth;
    let buf = format!("smem_{}", which);
    let gmem = format!("{}_gmem", which.to_uppercase());
    let mut s = String::new();
    writeln!(s, "{{").unwrap();
    writeln!(
        s,
        "  // load_{}_tile: {}[kv_tile + {}, head_group, :, :] → {}[slot]",
        which, gmem, ctx.iter_offset, buf,
    )
    .unwrap();
    writeln!(
        s,
        "  uint32_t slot = (kv_tile + {}) % {};",
        ctx.iter_offset, p
    )
    .unwrap();
    match ctx.arch_name {
        "sm90_fa2" => {
            writeln!(s, "  if (wg == LOADER_WG) {{").unwrap();
            writeln!(
                s,
                "    tma_load_2d({}[slot], {}, kv_tile + {}, head_group);",
                buf, gmem, ctx.iter_offset,
            )
            .unwrap();
            writeln!(s, "    mbarrier_arrive(&bar_kv[slot]);").unwrap();
            writeln!(s, "  }}").unwrap();
        }
        "sm89_fa2" => {
            writeln!(
                s,
                "  cp_async_128({}[slot], {}, kv_tile + {}, head_group);",
                buf, gmem, ctx.iter_offset,
            )
            .unwrap();
            writeln!(s, "  cp_async_commit_group();").unwrap();
        }
        other => {
            writeln!(s, "  // unknown arch {} — fall back to generic load", other).unwrap();
            writeln!(
                s,
                "  generic_load({}[slot], {}, kv_tile + {}, head_group);",
                buf, gmem, ctx.iter_offset,
            )
            .unwrap();
        }
    }
    write!(s, "}}").unwrap();
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(arch: &'static str) -> ExpandCtx<'static> {
        ExpandCtx {
            arch_name: arch,
            iter_offset: 0,
            pipeline_depth: 3,
        }
    }

    #[test]
    fn load_q_tile_sm90_uses_tma_and_mbarrier_arrive() {
        let s = expand("load_q_tile", &ctx("sm90_fa2")).unwrap();
        assert!(s.contains("tma_load_2d(smem_q, Q_gmem"));
        assert!(s.contains("mbarrier_arrive(&bar_q)"));
        // Loader-wg guard is present.
        assert!(s.contains("if (wg == LOADER_WG)"));
    }

    #[test]
    fn load_q_tile_sm89_uses_cp_async() {
        let s = expand("load_q_tile", &ctx("sm89_fa2")).unwrap();
        assert!(s.contains("cp_async_128(smem_q, Q_gmem"));
        assert!(s.contains("cp_async_commit_group"));
        // No warpgroup guard on SM89 (AllWarps).
        assert!(!s.contains("LOADER_WG"));
    }

    fn pipe_ctx(arch: &'static str) -> ExpandCtx<'static> {
        ExpandCtx {
            arch_name: arch,
            iter_offset: 3,
            pipeline_depth: 3,
        }
    }

    #[test]
    fn load_kv_tile_sm90_rotates_slot_and_arrives_on_bar_kv() {
        let k = expand("load_k_tile", &pipe_ctx("sm90_fa2")).unwrap();
        assert!(k.contains("uint32_t slot = (kv_tile + 3) % 3;"));
        assert!(k.contains("tma_load_2d(smem_k[slot], K_gmem, kv_tile + 3"));
        assert!(k.contains("mbarrier_arrive(&bar_kv[slot]);"));
        assert!(k.contains("if (wg == LOADER_WG)"));

        let v = expand("load_v_tile", &pipe_ctx("sm90_fa2")).unwrap();
        assert!(v.contains("tma_load_2d(smem_v[slot], V_gmem"));
    }

    #[test]
    fn load_kv_tile_sm89_uses_cp_async_into_ring_slot() {
        let k = expand("load_k_tile", &pipe_ctx("sm89_fa2")).unwrap();
        assert!(k.contains("uint32_t slot = (kv_tile + 3) % 3;"));
        assert!(k.contains("cp_async_128(smem_k[slot], K_gmem, kv_tile + 3"));
        assert!(k.contains("cp_async_commit_group"));
        assert!(!k.contains("LOADER_WG"));
    }

    #[test]
    fn unknown_tag_returns_none() {
        assert!(expand("qk_matmul", &ctx("sm90_fa2")).is_none());
        assert!(expand("not_a_real_op", &ctx("sm89_fa2")).is_none());
    }
}
