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
        "qk_matmul" => Some(qk_matmul(ctx)),
        "softmax_update" => Some(softmax_update(ctx)),
        "pv_matmul" => Some(pv_matmul(ctx)),
        "store_o_tile" => Some(store_o_tile(ctx)),
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

/// Guard a block on the consumer warpgroup on SM90, or run it on
/// AllWarps elsewhere. The compute ops all run on the consumer;
/// centralising the dispatch keeps the per-op bodies focused on the
/// math.
fn compute_guarded(ctx: &ExpandCtx<'_>, body: &str) -> String {
    let mut s = String::new();
    writeln!(s, "{{").unwrap();
    match ctx.arch_name {
        "sm90_fa2" => {
            writeln!(s, "  if (wg == CONSUMER_WG) {{").unwrap();
            for line in body.lines() {
                writeln!(s, "    {}", line).unwrap();
            }
            writeln!(s, "  }}").unwrap();
        }
        _ => {
            for line in body.lines() {
                writeln!(s, "  {}", line).unwrap();
            }
        }
    }
    write!(s, "}}").unwrap();
    s
}

/// S_frag = Q_tile @ K_tile^T, accumulating into registers.
///
/// On SM90 the consumer issues a single `wgmma.mma_async` across the
/// warpgroup; the smem-K slot corresponds to the iter's current kv
/// tile. On SM89 every warp participates in a tiled `mma.sync`
/// sweep over head_dim / k.
fn qk_matmul(ctx: &ExpandCtx<'_>) -> String {
    debug_assert_eq!(ctx.iter_offset, 0, "compute nodes carry no iter_offset");
    let body = match ctx.arch_name {
        "sm90_fa2" => concat!(
            "// S_frag = Q_tile @ K_tile^T\n",
            "uint32_t slot = kv_tile % PIPE;\n",
            "wgmma_fence();\n",
            "wgmma_mma_async(S_frag, smem_q, smem_k[slot]);\n",
            "wgmma_commit_group();\n",
            "wgmma_wait_group<0>();\n",
        ),
        "sm89_fa2" => concat!(
            "// S_frag = Q_tile @ K_tile^T (tiled mma.sync over head_dim)\n",
            "uint32_t slot = kv_tile % PIPE;\n",
            "mma_sync_accumulate(S_frag, smem_q, smem_k[slot]);\n",
        ),
        _ => "mma_accumulate(S_frag, smem_q, smem_k);\n",
    };
    compute_guarded(ctx, body)
}

/// Online softmax rescale: update per-row (m, l); rescale the O
/// accumulator by exp(m_old - m_new); produce P_frag = exp(S - m_new).
/// Same math on both archs; SM90 still guards on the consumer wg so
/// only one warpgroup touches the accumulators.
fn softmax_update(ctx: &ExpandCtx<'_>) -> String {
    debug_assert_eq!(ctx.iter_offset, 0, "compute nodes carry no iter_offset");
    let body = concat!(
        "// online softmax: (m, l, O) ← update(S_frag)\n",
        "float m_new = row_max(S_frag, m);\n",
        "float scale = exp2f(m - m_new);\n",
        "P_frag = exp2f_frag(S_frag - m_new);\n",
        "l      = scale * l + row_sum(P_frag);\n",
        "O_frag = scale * O_frag;  // rescale prior accumulator\n",
        "m      = m_new;\n",
    );
    compute_guarded(ctx, body)
}

/// O_frag += P_frag @ V_tile. Same warpgroup/warp split as
/// `qk_matmul`, pipelining into the V slot.
fn pv_matmul(ctx: &ExpandCtx<'_>) -> String {
    debug_assert_eq!(ctx.iter_offset, 0, "compute nodes carry no iter_offset");
    let body = match ctx.arch_name {
        "sm90_fa2" => concat!(
            "// O_frag += P_frag @ V_tile\n",
            "uint32_t slot = kv_tile % PIPE;\n",
            "wgmma_fence();\n",
            "wgmma_mma_async(O_frag, P_frag, smem_v[slot]);\n",
            "wgmma_commit_group();\n",
            "wgmma_wait_group<0>();\n",
            "// release the kv slot back to the loader\n",
            "mbarrier_arrive(&bar_kv_consumed[slot]);\n",
        ),
        "sm89_fa2" => concat!(
            "// O_frag += P_frag @ V_tile\n",
            "uint32_t slot = kv_tile % PIPE;\n",
            "mma_sync_accumulate(O_frag, P_frag, smem_v[slot]);\n",
        ),
        _ => "mma_accumulate(O_frag, P_frag, smem_v);\n",
    };
    compute_guarded(ctx, body)
}

/// Final store: divide O_frag by l, write to gmem at (q_tile, head_group).
///
/// On SM90 the consumer stages the fragment into smem and signals the
/// storer warpgroup, which performs a TMA store; on SM89 the consumer
/// writes directly to gmem with STG.
fn store_o_tile(ctx: &ExpandCtx<'_>) -> String {
    debug_assert_eq!(ctx.iter_offset, 0, "store nodes carry no iter_offset");
    let mut s = String::new();
    writeln!(s, "{{").unwrap();
    writeln!(
        s,
        "  // store_o_tile: O_gmem[q_tile, head_group] ← O_frag / l"
    )
    .unwrap();
    match ctx.arch_name {
        "sm90_fa2" => {
            writeln!(s, "  if (wg == CONSUMER_WG) {{").unwrap();
            writeln!(s, "    O_frag = O_frag * rcp(l);").unwrap();
            writeln!(s, "    stmatrix_smem(smem_o, O_frag);").unwrap();
            writeln!(s, "    mbarrier_arrive(&bar_o_ready);").unwrap();
            writeln!(s, "  }} else if (wg == STORER_WG) {{").unwrap();
            writeln!(s, "    mbarrier_wait(&bar_o_ready);").unwrap();
            writeln!(s, "    tma_store_2d(O_gmem, smem_o, q_tile, head_group);").unwrap();
            writeln!(s, "  }}").unwrap();
        }
        "sm89_fa2" => {
            writeln!(s, "  O_frag = O_frag * rcp(l);").unwrap();
            writeln!(s, "  stg_128(O_gmem, O_frag, q_tile, head_group);").unwrap();
        }
        other => {
            writeln!(
                s,
                "  // unknown arch {} — fall back to generic store",
                other
            )
            .unwrap();
            writeln!(s, "  generic_store(O_gmem, O_frag, l, q_tile, head_group);").unwrap();
        }
    }
    let _ = ctx.pipeline_depth;
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
    fn qk_matmul_sm90_uses_wgmma_under_consumer_guard() {
        let s = expand("qk_matmul", &ctx("sm90_fa2")).unwrap();
        assert!(s.contains("if (wg == CONSUMER_WG)"));
        assert!(s.contains("wgmma_mma_async(S_frag, smem_q, smem_k[slot])"));
        assert!(s.contains("wgmma_commit_group"));
        assert!(s.contains("wgmma_wait_group<0>"));
    }

    #[test]
    fn qk_matmul_sm89_uses_mma_sync_no_wg_guard() {
        let s = expand("qk_matmul", &ctx("sm89_fa2")).unwrap();
        assert!(s.contains("mma_sync_accumulate(S_frag, smem_q, smem_k[slot])"));
        assert!(!s.contains("CONSUMER_WG"));
        assert!(!s.contains("wgmma"));
    }

    #[test]
    fn softmax_update_emits_online_rescale() {
        for arch in ["sm90_fa2", "sm89_fa2"] {
            let s = expand("softmax_update", &ctx(arch)).unwrap();
            assert!(s.contains("m_new = row_max(S_frag, m)"));
            assert!(s.contains("scale = exp2f(m - m_new)"));
            assert!(s.contains("l      = scale * l + row_sum(P_frag)"));
            assert!(s.contains("O_frag = scale * O_frag"));
        }
        // SM90 guards it on consumer wg; SM89 does not.
        assert!(
            expand("softmax_update", &ctx("sm90_fa2"))
                .unwrap()
                .contains("CONSUMER_WG")
        );
        assert!(
            !expand("softmax_update", &ctx("sm89_fa2"))
                .unwrap()
                .contains("CONSUMER_WG")
        );
    }

    #[test]
    fn pv_matmul_sm90_releases_kv_slot_to_loader() {
        let s = expand("pv_matmul", &ctx("sm90_fa2")).unwrap();
        assert!(s.contains("wgmma_mma_async(O_frag, P_frag, smem_v[slot])"));
        // Consumer signals the loader that the kv slot is free again.
        assert!(s.contains("mbarrier_arrive(&bar_kv_consumed[slot])"));
    }

    #[test]
    fn store_o_sm90_splits_consumer_staging_and_storer_tma() {
        let s = expand("store_o_tile", &ctx("sm90_fa2")).unwrap();
        assert!(s.contains("if (wg == CONSUMER_WG)"));
        assert!(s.contains("O_frag = O_frag * rcp(l)"));
        assert!(s.contains("stmatrix_smem(smem_o, O_frag)"));
        assert!(s.contains("mbarrier_arrive(&bar_o_ready)"));
        assert!(s.contains("} else if (wg == STORER_WG)"));
        assert!(s.contains("tma_store_2d(O_gmem, smem_o, q_tile, head_group)"));
    }

    #[test]
    fn store_o_sm89_uses_direct_stg() {
        let s = expand("store_o_tile", &ctx("sm89_fa2")).unwrap();
        assert!(s.contains("O_frag = O_frag * rcp(l)"));
        assert!(s.contains("stg_128(O_gmem, O_frag, q_tile, head_group)"));
        assert!(!s.contains("STORER_WG"));
    }

    #[test]
    fn unknown_tag_returns_none() {
        assert!(expand("not_a_real_op", &ctx("sm89_fa2")).is_none());
        assert!(expand("paged_gather_kv", &ctx("sm90_fa2")).is_none());
    }
}
