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

/// How a tag touches a gmem tensor: read-only, write-only, or both.
/// Consumed by `emit_megakernel` to decide whether the kernel param
/// needs the `const` qualifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GmemAccess {
    Read,
    Write,
    ReadWrite,
}

impl GmemAccess {
    pub fn union(self, other: Self) -> Self {
        match (self, other) {
            (Self::ReadWrite, _) | (_, Self::ReadWrite) => Self::ReadWrite,
            (Self::Read, Self::Write) | (Self::Write, Self::Read) => Self::ReadWrite,
            (Self::Read, Self::Read) => Self::Read,
            (Self::Write, Self::Write) => Self::Write,
        }
    }
}

/// Gmem tensor names + access kinds each tag touches. Companion to
/// `expand`: the emitter uses this to assemble the kernel signature
/// (one pointer param per unique tensor) while `expand` emits the
/// body that references them. Compute-only tags return an empty
/// slice — they work on register fragments, not gmem.
pub fn gmem_refs(tag: &str) -> &'static [(&'static str, GmemAccess)] {
    use GmemAccess::*;
    match tag {
        // ── Attention (FA2 + paged decode) ──
        "load_q_tile" => &[("Q_gmem", Read)],
        "load_k_tile" => &[("K_gmem", Read)],
        "load_v_tile" => &[("V_gmem", Read)],
        "store_o_tile" => &[("O_gmem", Write)],
        // ── GEMM ──
        "load_a_tile" => &[("A_gmem", Read)],
        "load_b_tile" => &[("B_gmem", Read)],
        "store_c_tile" => &[("C_gmem", Write)],
        // ── Norm / elementwise (row-wise; load_x_row etc. use the
        //    same name across regions because the region's regional
        //    gmem context lives in emit_mega, not here). ──
        "load_x_row" => &[("X_gmem", Read)],
        "load_weight" => &[("W_gmem", Read)],
        "store_y_row" => &[("Y_gmem", Write)],
        "load_a_row" => &[("A_gmem", Read)],
        "load_b_row" => &[("B_gmem", Read)],
        "store_sum_row" => &[("Sum_gmem", Write)],
        // ── QKV + RoPE ──
        "load_wqkv_tile" => &[("Wqkv_gmem", Read)],
        "load_rope_coef" => &[("RopeCoef_gmem", Read)],
        "store_q_row" => &[("Q_gmem", Write)],
        "store_k_row" => &[("K_gmem", Write)],
        "store_v_row" => &[("V_gmem", Write)],
        "store_k_cache" => &[("K_cache_gmem", Write)],
        "store_v_cache" => &[("V_cache_gmem", Write)],
        // ── Gate + Up + SiLU/GeLU + Mul ──
        "load_wgate_tile" => &[("Wgate_gmem", Read)],
        "load_wup_tile" => &[("Wup_gmem", Read)],
        "store_inter_tile" => &[("Inter_gmem", Write)],
        // ── Embedding lookup ──
        "load_embed_row" => &[
            ("Embed_gmem", Read),
            // Indirection index lives in gmem as well (token_ids[]).
            ("TokenIds_gmem", Read),
        ],
        "store_embed_row" => &[("Y_gmem", Write)],
        // Compute-only tags: no gmem touch.
        _ => &[],
    }
}

/// Try to expand `(tag, arch)` into concrete CUDA pseudocode. Returns
/// `None` when no entry exists yet; callers fall back to the stub
/// `tag();` line.
pub fn expand(tag: &str, ctx: &ExpandCtx<'_>) -> Option<String> {
    match tag {
        // ── Attention (FA2 + paged decode) ──
        "load_q_tile" => Some(load_q_tile(ctx)),
        "load_k_tile" => Some(load_kv_tile(ctx, "k")),
        "load_v_tile" => Some(load_kv_tile(ctx, "v")),
        "qk_matmul" => Some(qk_matmul(ctx)),
        "softmax_update" => Some(softmax_update(ctx)),
        "pv_matmul" => Some(pv_matmul(ctx)),
        "store_o_tile" => Some(store_o_tile(ctx)),
        // ── GEMM (+ quant variants share this) ──
        "load_a_tile" => Some(generic_pipeline_load(
            ctx, "smem_a", "A_gmem", "m_tile", "k_tile",
        )),
        "load_b_tile" => Some(generic_pipeline_load(
            ctx, "smem_b", "B_gmem", "k_tile", "n_tile",
        )),
        "gemm_accumulate" => Some(generic_gemm(ctx, "C_frag", "smem_a", "smem_b")),
        "store_c_tile" => Some(generic_store(ctx, "C_gmem", "C_frag", "m_tile", "n_tile")),
        // ── RMSNorm / LayerNorm / Add ──
        "load_x_row" => Some(generic_preamble_load(ctx, "smem_x", "X_gmem", "token_tile")),
        "load_weight" => Some(generic_preamble_load(ctx, "smem_w", "W_gmem", "token_tile")),
        "rmsnorm_compute" => Some(rmsnorm_compute(ctx)),
        "store_y_row" => Some(generic_store(
            ctx,
            "Y_gmem",
            "Y_frag",
            "token_tile",
            "/*hidden*/",
        )),
        "load_a_row" => Some(generic_preamble_load(ctx, "smem_a", "A_gmem", "token_tile")),
        "load_b_row" => Some(generic_preamble_load(ctx, "smem_b", "B_gmem", "token_tile")),
        "elementwise_add" => Some(elementwise_add(ctx)),
        "store_sum_row" => Some(generic_store(
            ctx,
            "Sum_gmem",
            "sum_frag",
            "token_tile",
            "/*hidden*/",
        )),
        // ── QKV + RoPE ──
        "load_wqkv_tile" => Some(generic_pipeline_load(
            ctx,
            "smem_wqkv",
            "Wqkv_gmem",
            "head_tile",
            "k_tile",
        )),
        "qkv_matmul" => Some(generic_gemm(ctx, "QKV_frag", "smem_x", "smem_wqkv")),
        "load_rope_coef" => Some(generic_preamble_load(
            ctx,
            "smem_rope",
            "RopeCoef_gmem",
            "token_tile",
        )),
        "apply_rope" => Some(apply_rope(ctx)),
        "store_q_row" => Some(generic_store(
            ctx,
            "Q_gmem",
            "Q_frag",
            "token_tile",
            "head_tile",
        )),
        "store_k_row" => Some(generic_store(
            ctx,
            "K_gmem",
            "K_frag",
            "token_tile",
            "head_tile",
        )),
        "store_v_row" => Some(generic_store(
            ctx,
            "V_gmem",
            "V_frag",
            "token_tile",
            "head_tile",
        )),
        "store_k_cache" => Some(generic_cache_store(ctx, "K_cache_gmem", "K_frag")),
        "store_v_cache" => Some(generic_cache_store(ctx, "V_cache_gmem", "V_frag")),
        // ── Gate + Up + SiLU/GeLU + Mul (MLP input) ──
        "load_wgate_tile" => Some(generic_pipeline_load(
            ctx,
            "smem_wgate",
            "Wgate_gmem",
            "inter_tile",
            "k_tile",
        )),
        "load_wup_tile" => Some(generic_pipeline_load(
            ctx,
            "smem_wup",
            "Wup_gmem",
            "inter_tile",
            "k_tile",
        )),
        "gate_gemm_accumulate" => Some(generic_gemm(ctx, "Gate_frag", "smem_x", "smem_wgate")),
        "up_gemm_accumulate" => Some(generic_gemm(ctx, "Up_frag", "smem_x", "smem_wup")),
        "silu_mul_fuse" => Some(silu_mul_fuse(ctx)),
        "store_inter_tile" => Some(generic_store(
            ctx,
            "Inter_gmem",
            "Inter_frag",
            "token_tile",
            "inter_tile",
        )),
        // ── Unary in-place ──
        "scalar_mul" => Some(generic_unary(ctx, "scalar_mul", "X_frag", "X_frag * scale")),
        "tanh_softcap" => Some(generic_unary(
            ctx,
            "tanh_softcap",
            "X_frag",
            "cap * __tanhf(X_frag * rcp_cap)",
        )),
        // ── Embedding lookup ──
        "load_embed_row" => Some(embed_gather(ctx)),
        "store_embed_row" => Some(generic_store(
            ctx,
            "Y_gmem",
            "Embed_frag",
            "token_tile",
            "/*hidden*/",
        )),
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

// ─── Generic expansion helpers for non-attention tags ──────────
//
// Each helper renders a small CUDA-pseudocode block that mirrors the
// per-arch patterns of the FA2 expansions above (TMA + mbarrier on
// SM90; cp.async.ca + cp.async_commit on SM89). Bodies are
// pseudocode — `wgmma_mma_async`, `mma_sync_accumulate`, etc. — not
// yet real intrinsic calls; the intent is to render the shape of the
// kernel so the megakernel composition is reviewable. Real intrinsic
// lowering lands once the runtime launch glue can compile the output.

/// Pipelined load into a `P`-deep ring slot. Same shape pattern as
/// load_kv_tile but parameterized on the smem/gmem buffer names and
/// the axis labels the caller uses. One call per Load node in a
/// pipeline-edge region.
fn generic_pipeline_load(
    ctx: &ExpandCtx<'_>,
    smem: &str,
    gmem: &str,
    row_axis: &str,
    col_axis: &str,
) -> String {
    let p = ctx.pipeline_depth;
    let mut s = String::new();
    writeln!(s, "{{").unwrap();
    writeln!(
        s,
        "  // {}: {}[{} + {}] → {}[slot]",
        gmem, gmem, row_axis, ctx.iter_offset, smem,
    )
    .unwrap();
    writeln!(
        s,
        "  uint32_t slot = ({} + {}) % {};",
        row_axis, ctx.iter_offset, p
    )
    .unwrap();
    match ctx.arch_name {
        "sm90_fa2" => {
            writeln!(s, "  if (wg == LOADER_WG) {{").unwrap();
            writeln!(
                s,
                "    tma_load_2d({}[slot], {}, {} + {}, {});",
                smem, gmem, row_axis, ctx.iter_offset, col_axis,
            )
            .unwrap();
            writeln!(s, "    mbarrier_arrive(&bar_{}[slot]);", smem).unwrap();
            writeln!(s, "  }}").unwrap();
        }
        "sm89_fa2" => {
            writeln!(
                s,
                "  cp_async_128({}[slot], {}, {} + {}, {});",
                smem, gmem, row_axis, ctx.iter_offset, col_axis,
            )
            .unwrap();
            writeln!(s, "  cp_async_commit_group();").unwrap();
        }
        other => {
            writeln!(s, "  // unknown arch {} — fall back to generic load", other).unwrap();
            writeln!(
                s,
                "  generic_load({}[slot], {}, {} + {}, {});",
                smem, gmem, row_axis, ctx.iter_offset, col_axis,
            )
            .unwrap();
        }
    }
    write!(s, "}}").unwrap();
    s
}

/// Straight-line preamble load: no ring slot, no pipeline semantics.
/// Used by region templates whose Load nodes don't sit on a Pipeline
/// edge (rmsnorm, elementwise ops, rope coefficients).
fn generic_preamble_load(ctx: &ExpandCtx<'_>, smem: &str, gmem: &str, axis: &str) -> String {
    let mut s = String::new();
    writeln!(s, "{{").unwrap();
    writeln!(s, "  // {}: {}[{}] → {}", gmem, gmem, axis, smem).unwrap();
    match ctx.arch_name {
        "sm90_fa2" => {
            writeln!(s, "  if (wg == LOADER_WG) {{").unwrap();
            writeln!(s, "    tma_load_2d({}, {}, {});", smem, gmem, axis).unwrap();
            writeln!(s, "    mbarrier_arrive(&bar_{});", smem).unwrap();
            writeln!(s, "  }}").unwrap();
        }
        "sm89_fa2" => {
            writeln!(s, "  cp_async_128({}, {}, {});", smem, gmem, axis).unwrap();
            writeln!(s, "  cp_async_commit_group();").unwrap();
        }
        other => {
            writeln!(s, "  // unknown arch {} — fall back to generic load", other).unwrap();
        }
    }
    let _ = (ctx.iter_offset, ctx.pipeline_depth);
    write!(s, "}}").unwrap();
    s
}

/// Accumulating GEMM inside a serial-K loop. `acc` accumulates in
/// registers across `k_tile` iterations; smem_a / smem_b rotate
/// through a `P`-deep ring via `slot = k_tile % PIPE`.
fn generic_gemm(ctx: &ExpandCtx<'_>, acc: &str, smem_a: &str, smem_b: &str) -> String {
    debug_assert_eq!(ctx.iter_offset, 0, "compute nodes carry no iter_offset");
    let body = match ctx.arch_name {
        "sm90_fa2" => format!(
            concat!(
                "// {acc} += {a} @ {b}\n",
                "uint32_t slot = k_tile % PIPE;\n",
                "wgmma_fence();\n",
                "wgmma_mma_async({acc}, {a}[slot], {b}[slot]);\n",
                "wgmma_commit_group();\n",
                "wgmma_wait_group<0>();\n",
            ),
            acc = acc,
            a = smem_a,
            b = smem_b,
        ),
        "sm89_fa2" => format!(
            concat!(
                "// {acc} += {a} @ {b} (tiled mma.sync)\n",
                "uint32_t slot = k_tile % PIPE;\n",
                "mma_sync_accumulate({acc}, {a}[slot], {b}[slot]);\n",
            ),
            acc = acc,
            a = smem_a,
            b = smem_b,
        ),
        _ => format!("mma_accumulate({}, {}, {});\n", acc, smem_a, smem_b),
    };
    compute_guarded(ctx, &body)
}

/// Final store of a fragment: SM90 stages through smem + TMA store,
/// SM89 emits STG.128 directly. One helper across every store_* tag
/// in the non-attention templates.
fn generic_store(
    ctx: &ExpandCtx<'_>,
    gmem: &str,
    frag: &str,
    row_axis: &str,
    col_axis: &str,
) -> String {
    debug_assert_eq!(ctx.iter_offset, 0, "store nodes carry no iter_offset");
    let mut s = String::new();
    writeln!(s, "{{").unwrap();
    writeln!(
        s,
        "  // {}: {}[{}, {}] ← {}",
        gmem, gmem, row_axis, col_axis, frag
    )
    .unwrap();
    match ctx.arch_name {
        "sm90_fa2" => {
            writeln!(s, "  if (wg == CONSUMER_WG) {{").unwrap();
            writeln!(s, "    stmatrix_smem(smem_out, {});", frag).unwrap();
            writeln!(s, "    mbarrier_arrive(&bar_{}_ready);", gmem).unwrap();
            writeln!(s, "  }} else if (wg == STORER_WG) {{").unwrap();
            writeln!(s, "    mbarrier_wait(&bar_{}_ready);", gmem).unwrap();
            writeln!(
                s,
                "    tma_store_2d({}, smem_out, {}, {});",
                gmem, row_axis, col_axis,
            )
            .unwrap();
            writeln!(s, "  }}").unwrap();
        }
        "sm89_fa2" => {
            writeln!(
                s,
                "  stg_128({}, {}, {}, {});",
                gmem, frag, row_axis, col_axis,
            )
            .unwrap();
        }
        other => {
            writeln!(
                s,
                "  // unknown arch {} — fall back to generic store",
                other
            )
            .unwrap();
        }
    }
    let _ = ctx.pipeline_depth;
    write!(s, "}}").unwrap();
    s
}

/// KV cache store — same signal shape as generic_store but through a
/// paged-KV address computation (block_table gather).
fn generic_cache_store(ctx: &ExpandCtx<'_>, gmem: &str, frag: &str) -> String {
    let mut s = String::new();
    writeln!(s, "{{").unwrap();
    writeln!(
        s,
        "  // {}: KV cache slot (token_tile → block_table lookup) ← {}",
        gmem, frag,
    )
    .unwrap();
    writeln!(
        s,
        "  uint32_t page = block_table[token_tile / blocks_per_tile];"
    )
    .unwrap();
    writeln!(s, "  uint32_t slot = token_tile % blocks_per_tile;").unwrap();
    match ctx.arch_name {
        "sm90_fa2" => {
            writeln!(s, "  if (wg == STORER_WG) {{").unwrap();
            writeln!(
                s,
                "    tma_store_2d({}[page, slot, head_tile], {});",
                gmem, frag,
            )
            .unwrap();
            writeln!(s, "  }}").unwrap();
        }
        "sm89_fa2" => {
            writeln!(s, "  stg_128({}[page, slot, head_tile], {});", gmem, frag,).unwrap();
        }
        _ => {}
    }
    let _ = (ctx.iter_offset, ctx.pipeline_depth);
    write!(s, "}}").unwrap();
    s
}

/// Element-wise unary op. `expr` is the per-lane expression (e.g.
/// `X_frag * scale` for scalar_mul).
fn generic_unary(ctx: &ExpandCtx<'_>, tag: &str, frag: &str, expr: &str) -> String {
    debug_assert_eq!(ctx.iter_offset, 0, "compute nodes carry no iter_offset");
    let body = format!("// {tag}: {frag} = {expr};\n{frag} = {expr};\n");
    compute_guarded(ctx, &body)
}

/// Warp-/wg-wide RMSNorm: rms = rsqrt(mean(x·x) + ε); y = x * rms * w.
/// Reduction internals are intrinsic-level (shuffle / shared-memory);
/// pseudocode here.
fn rmsnorm_compute(ctx: &ExpandCtx<'_>) -> String {
    debug_assert_eq!(ctx.iter_offset, 0, "compute nodes carry no iter_offset");
    let body = concat!(
        "// Y_frag = X_row * rsqrt(mean(X_row^2) + eps) * W_row\n",
        "float sumsq = warp_reduce_sum_of_squares(smem_x);\n",
        "float rms   = rsqrtf(sumsq / hidden_dim + eps);\n",
        "Y_frag      = frag_mul(frag_mul(smem_x, smem_w), rms);\n",
    );
    compute_guarded(ctx, body)
}

/// Element-wise add of two preloaded rows.
fn elementwise_add(ctx: &ExpandCtx<'_>) -> String {
    debug_assert_eq!(ctx.iter_offset, 0, "compute nodes carry no iter_offset");
    let body = "// sum_frag = smem_a + smem_b\nsum_frag = frag_add(smem_a, smem_b);\n";
    compute_guarded(ctx, body)
}

/// Apply rotary position embedding to the (Q, K) projections. V
/// passes through unchanged.
fn apply_rope(ctx: &ExpandCtx<'_>) -> String {
    debug_assert_eq!(ctx.iter_offset, 0, "compute nodes carry no iter_offset");
    let body = concat!(
        "// Apply RoPE in-place to Q_frag and K_frag using smem_rope.\n",
        "Q_frag = rope_rotate(QKV_frag.q, smem_rope);\n",
        "K_frag = rope_rotate(QKV_frag.k, smem_rope);\n",
        "V_frag = QKV_frag.v;\n",
    );
    compute_guarded(ctx, body)
}

/// SiLU on gate, then multiply by up. One compute guard around both.
fn silu_mul_fuse(ctx: &ExpandCtx<'_>) -> String {
    debug_assert_eq!(ctx.iter_offset, 0, "compute nodes carry no iter_offset");
    let body = concat!(
        "// Inter_frag = silu(Gate_frag) * Up_frag\n",
        "Inter_frag = frag_mul(silu(Gate_frag), Up_frag);\n",
    );
    compute_guarded(ctx, body)
}

/// Embedding lookup: gather row from embed_table using token_ids.
/// One Load, no Compute — this is pure data movement.
fn embed_gather(ctx: &ExpandCtx<'_>) -> String {
    let mut s = String::new();
    writeln!(s, "{{").unwrap();
    writeln!(s, "  // Embed_frag = embed_table[token_ids[token_tile]];").unwrap();
    match ctx.arch_name {
        "sm90_fa2" => {
            writeln!(s, "  if (wg == LOADER_WG) {{").unwrap();
            writeln!(s, "    uint32_t row = token_ids[token_tile];").unwrap();
            writeln!(
                s,
                "    tma_load_2d(Embed_frag, Embed_gmem + row * hidden_stride);"
            )
            .unwrap();
            writeln!(s, "    mbarrier_arrive(&bar_embed);").unwrap();
            writeln!(s, "  }}").unwrap();
        }
        "sm89_fa2" => {
            writeln!(s, "  uint32_t row = token_ids[token_tile];").unwrap();
            writeln!(
                s,
                "  cp_async_128(Embed_frag, Embed_gmem + row * hidden_stride);"
            )
            .unwrap();
            writeln!(s, "  cp_async_commit_group();").unwrap();
        }
        other => {
            writeln!(s, "  // unknown arch {}", other).unwrap();
        }
    }
    let _ = (ctx.iter_offset, ctx.pipeline_depth);
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

    // ── New-tag expansions ────────────────────────────────────────

    fn pipe_ctx_nonzero(arch: &'static str) -> ExpandCtx<'static> {
        ExpandCtx {
            arch_name: arch,
            iter_offset: 3,
            pipeline_depth: 3,
        }
    }

    #[test]
    fn gemm_accumulate_sm90_uses_wgmma() {
        let s = expand("gemm_accumulate", &ctx("sm90_fa2")).unwrap();
        assert!(s.contains("wgmma_mma_async(C_frag, smem_a[slot], smem_b[slot])"));
        assert!(s.contains("CONSUMER_WG"));
    }

    #[test]
    fn load_a_tile_sm89_uses_cp_async_into_slot() {
        let s = expand("load_a_tile", &pipe_ctx_nonzero("sm89_fa2")).unwrap();
        assert!(s.contains("uint32_t slot = (m_tile + 3) % 3;"));
        assert!(s.contains("cp_async_128(smem_a[slot], A_gmem"));
    }

    #[test]
    fn store_c_tile_sm89_uses_stg() {
        let s = expand("store_c_tile", &ctx("sm89_fa2")).unwrap();
        assert!(s.contains("stg_128(C_gmem, C_frag, m_tile, n_tile)"));
    }

    #[test]
    fn rmsnorm_compute_has_reduce_and_rsqrt() {
        for arch in ["sm90_fa2", "sm89_fa2"] {
            let s = expand("rmsnorm_compute", &ctx(arch)).unwrap();
            assert!(s.contains("warp_reduce_sum_of_squares"));
            assert!(s.contains("rsqrtf"));
            assert!(s.contains("Y_frag"));
        }
    }

    #[test]
    fn apply_rope_rotates_q_and_k_not_v() {
        let s = expand("apply_rope", &ctx("sm90_fa2")).unwrap();
        assert!(s.contains("Q_frag = rope_rotate(QKV_frag.q"));
        assert!(s.contains("K_frag = rope_rotate(QKV_frag.k"));
        // V passes through unrotated.
        assert!(s.contains("V_frag = QKV_frag.v"));
    }

    #[test]
    fn silu_mul_fuse_applies_silu_to_gate_and_mul_up() {
        let s = expand("silu_mul_fuse", &ctx("sm90_fa2")).unwrap();
        assert!(s.contains("silu(Gate_frag)"));
        assert!(s.contains("frag_mul(silu(Gate_frag), Up_frag)"));
    }

    #[test]
    fn scalar_mul_and_tanh_softcap_expand_to_inline_expr() {
        let m = expand("scalar_mul", &ctx("sm89_fa2")).unwrap();
        assert!(m.contains("X_frag = X_frag * scale"));
        let t = expand("tanh_softcap", &ctx("sm89_fa2")).unwrap();
        assert!(t.contains("__tanhf(X_frag * rcp_cap)"));
    }

    #[test]
    fn embed_gather_indexes_through_token_ids() {
        for arch in ["sm90_fa2", "sm89_fa2"] {
            let s = expand("load_embed_row", &ctx(arch)).unwrap();
            assert!(s.contains("uint32_t row = token_ids[token_tile]"));
            assert!(s.contains("Embed_gmem + row * hidden_stride"));
        }
    }

    #[test]
    fn all_new_template_tags_have_expansions() {
        // Every tag the new region templates emit should expand, not
        // fall through to the stub. Guards against silently shipping
        // regions whose bodies render as `tag();`.
        let required = [
            "load_a_tile",
            "load_b_tile",
            "gemm_accumulate",
            "store_c_tile",
            "load_x_row",
            "load_weight",
            "rmsnorm_compute",
            "store_y_row",
            "load_a_row",
            "load_b_row",
            "elementwise_add",
            "store_sum_row",
            "load_wqkv_tile",
            "qkv_matmul",
            "load_rope_coef",
            "apply_rope",
            "store_q_row",
            "store_k_row",
            "store_v_row",
            "store_k_cache",
            "store_v_cache",
            "load_wgate_tile",
            "load_wup_tile",
            "gate_gemm_accumulate",
            "up_gemm_accumulate",
            "silu_mul_fuse",
            "store_inter_tile",
            "scalar_mul",
            "tanh_softcap",
            "load_embed_row",
            "store_embed_row",
        ];
        for tag in required {
            for arch in ["sm90_fa2", "sm89_fa2"] {
                assert!(
                    expand(tag, &ctx(arch)).is_some(),
                    "tag {} on arch {} has no expansion",
                    tag,
                    arch,
                );
            }
        }
    }
}
