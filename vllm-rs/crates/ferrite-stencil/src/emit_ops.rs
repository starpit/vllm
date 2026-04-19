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
    /// Region's parallel-axis names in outer-to-inner order (the
    /// `sched.parallel_axes` resolved against `region.axis(id).name`).
    /// FA2 = `["q_tile", "head_group"]`, paged decode = `["b",
    /// "head_group"]`, GEMM = `["m_tile", "n_tile"]`, qkv_rope =
    /// `["token_tile", "head_tile"]`, rmsnorm = `["token_tile"]`.
    pub parallel_axes: &'a [&'static str],
    /// Region's serial (reduction) axis name. FA2/paged-decode =
    /// `Some("kv_tile")`, GEMM/qkv/gate_up_silu = `Some("k_tile")`,
    /// rmsnorm/embed/unary = `None`.
    pub serial_axis: Option<&'static str>,
}

impl<'a> ExpandCtx<'a> {
    /// `parallel_axes[i]` or a stable placeholder when the region has
    /// fewer axes than the tag expects. Returns `"0u"` so the emitted
    /// address is still legal C++ even if a tag is mis-applied.
    pub fn par(&self, i: usize) -> &'static str {
        self.parallel_axes.get(i).copied().unwrap_or("0u")
    }
    /// `serial_axis.unwrap_or("0u")`. Serial-use tags (qk_matmul,
    /// pv_matmul, generic_gemm slot calc) assume one exists; the
    /// default keeps the emission legal if the region lacks one.
    pub fn ser(&self) -> &'static str {
        self.serial_axis.unwrap_or("0u")
    }
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

/// Region-local declaration a tag contributes. The emitter unions the
/// `local_refs` of every node in a region, dedupes by name (widening
/// `SmemPlain` to `SmemRing` when both forms appear), and emits the
/// declarations at the top of the region's `{}` scope. Item 2 in
/// STENCIL_IR_STATUS.md: regions used to share file-scope placeholders,
/// which is correctness-breaking once multiple GEMM/norm regions
/// coexist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalKind {
    /// `__shared__ StencilFrag name;` — single staging buffer.
    SmemPlain,
    /// `__shared__ StencilFrag name[PIPE];` — P-deep ring for
    /// pipeline-source loads (consumer reads slot `k`, loader writes
    /// slot `k + P`).
    SmemRing,
    /// `StencilFrag name;` — per-thread register fragment. Real
    /// lowering swaps for concrete mma accumulator types.
    Frag,
    /// `struct { StencilFrag q, k, v; } name;` — QKV trifecta held
    /// together so the RoPE compute can address its parts.
    FragQkv,
    /// `__shared__ Mbarrier name;` — single mbarrier for a plain
    /// preamble load.
    MbarrierPlain,
    /// `__shared__ Mbarrier name[PIPE];` — P-deep mbarrier ring.
    MbarrierRing,
    /// `float name = <init>;` — per-thread register scalar with an
    /// initializer. FA2's online softmax uses m/l this way.
    FloatInit(&'static str),
}

#[derive(Debug, Clone, Copy)]
pub struct LocalRef {
    pub name: &'static str,
    pub kind: LocalKind,
}

/// Names + kinds of every region-local that the expansion for `tag`
/// references. Compute-only tags still declare the fragments they
/// read+write (e.g. `gemm_accumulate` declares `C_frag`); Load/Store
/// tags declare their smem staging buffers and mbarriers.
pub fn local_refs(tag: &str) -> &'static [LocalRef] {
    use LocalKind::*;
    match tag {
        // ── Attention (FA2) ──
        "load_q_tile" => &[
            LocalRef {
                name: "smem_q",
                kind: SmemPlain,
            },
            LocalRef {
                name: "bar_q",
                kind: MbarrierPlain,
            },
        ],
        "load_k_tile" => &[
            LocalRef {
                name: "smem_k",
                kind: SmemRing,
            },
            LocalRef {
                name: "bar_kv",
                kind: MbarrierRing,
            },
        ],
        "load_v_tile" => &[
            LocalRef {
                name: "smem_v",
                kind: SmemRing,
            },
            LocalRef {
                name: "bar_kv",
                kind: MbarrierRing,
            },
        ],
        "qk_matmul" => &[LocalRef {
            name: "S_frag",
            kind: Frag,
        }],
        "softmax_update" => &[
            LocalRef {
                name: "P_frag",
                kind: Frag,
            },
            LocalRef {
                name: "O_frag",
                kind: Frag,
            },
            LocalRef {
                name: "m",
                kind: FloatInit("-INFINITY"),
            },
            LocalRef {
                name: "l",
                kind: FloatInit("0.0f"),
            },
        ],
        "pv_matmul" => &[
            LocalRef {
                name: "O_frag",
                kind: Frag,
            },
            LocalRef {
                name: "bar_kv_consumed",
                kind: MbarrierRing,
            },
        ],
        "store_o_tile" => &[
            LocalRef {
                name: "smem_o",
                kind: SmemPlain,
            },
            LocalRef {
                name: "bar_o_ready",
                kind: MbarrierPlain,
            },
        ],
        // ── GEMM (quant variants share) ──
        "load_a_tile" => &[
            LocalRef {
                name: "smem_a",
                kind: SmemRing,
            },
            LocalRef {
                name: "bar_smem_a",
                kind: MbarrierRing,
            },
        ],
        "load_b_tile" => &[
            LocalRef {
                name: "smem_b",
                kind: SmemRing,
            },
            LocalRef {
                name: "bar_smem_b",
                kind: MbarrierRing,
            },
        ],
        "gemm_accumulate" => &[LocalRef {
            name: "C_frag",
            kind: Frag,
        }],
        "store_c_tile" => &[
            LocalRef {
                name: "C_frag",
                kind: Frag,
            },
            LocalRef {
                name: "smem_out",
                kind: SmemPlain,
            },
            LocalRef {
                name: "bar_C_gmem_ready",
                kind: MbarrierPlain,
            },
        ],
        // ── RMSNorm / add (preamble-loaded rows, one staging buffer) ──
        "load_x_row" => &[
            LocalRef {
                name: "smem_x",
                kind: SmemPlain,
            },
            LocalRef {
                name: "bar_smem_x",
                kind: MbarrierPlain,
            },
        ],
        "load_weight" => &[
            LocalRef {
                name: "smem_w",
                kind: SmemPlain,
            },
            LocalRef {
                name: "bar_smem_w",
                kind: MbarrierPlain,
            },
        ],
        "rmsnorm_compute" => &[LocalRef {
            name: "Y_frag",
            kind: Frag,
        }],
        "store_y_row" => &[
            LocalRef {
                name: "Y_frag",
                kind: Frag,
            },
            LocalRef {
                name: "smem_out",
                kind: SmemPlain,
            },
            LocalRef {
                name: "bar_Y_gmem_ready",
                kind: MbarrierPlain,
            },
        ],
        "load_a_row" => &[
            LocalRef {
                name: "smem_a",
                kind: SmemPlain,
            },
            LocalRef {
                name: "bar_smem_a",
                kind: MbarrierPlain,
            },
        ],
        "load_b_row" => &[
            LocalRef {
                name: "smem_b",
                kind: SmemPlain,
            },
            LocalRef {
                name: "bar_smem_b",
                kind: MbarrierPlain,
            },
        ],
        "elementwise_add" => &[LocalRef {
            name: "sum_frag",
            kind: Frag,
        }],
        "store_sum_row" => &[
            LocalRef {
                name: "sum_frag",
                kind: Frag,
            },
            LocalRef {
                name: "smem_out",
                kind: SmemPlain,
            },
            LocalRef {
                name: "bar_Sum_gmem_ready",
                kind: MbarrierPlain,
            },
        ],
        // ── QKV + RoPE ──
        "load_wqkv_tile" => &[
            LocalRef {
                name: "smem_wqkv",
                kind: SmemRing,
            },
            LocalRef {
                name: "bar_smem_wqkv",
                kind: MbarrierRing,
            },
        ],
        "qkv_matmul" => &[
            LocalRef {
                name: "QKV_frag",
                kind: FragQkv,
            },
            // generic_gemm indexes both operands as `[slot]`; even
            // though load_x_row declares smem_x as plain for the
            // preamble load, the ring widening ensures the compute's
            // `smem_x[slot]` parses. Runtime correctness (plain buffer
            // loaded once but indexed per-slot) is item 1's problem.
            LocalRef {
                name: "smem_x",
                kind: SmemRing,
            },
            LocalRef {
                name: "smem_wqkv",
                kind: SmemRing,
            },
        ],
        "load_rope_coef" => &[
            LocalRef {
                name: "smem_rope",
                kind: SmemPlain,
            },
            LocalRef {
                name: "bar_smem_rope",
                kind: MbarrierPlain,
            },
        ],
        "apply_rope" => &[
            LocalRef {
                name: "Q_frag",
                kind: Frag,
            },
            LocalRef {
                name: "K_frag",
                kind: Frag,
            },
            LocalRef {
                name: "V_frag",
                kind: Frag,
            },
            LocalRef {
                name: "QKV_frag",
                kind: FragQkv,
            },
        ],
        "store_q_row" => &[
            LocalRef {
                name: "Q_frag",
                kind: Frag,
            },
            LocalRef {
                name: "smem_out",
                kind: SmemPlain,
            },
            LocalRef {
                name: "bar_Q_gmem_ready",
                kind: MbarrierPlain,
            },
        ],
        "store_k_row" => &[
            LocalRef {
                name: "K_frag",
                kind: Frag,
            },
            LocalRef {
                name: "smem_out",
                kind: SmemPlain,
            },
            LocalRef {
                name: "bar_K_gmem_ready",
                kind: MbarrierPlain,
            },
        ],
        "store_v_row" => &[
            LocalRef {
                name: "V_frag",
                kind: Frag,
            },
            LocalRef {
                name: "smem_out",
                kind: SmemPlain,
            },
            LocalRef {
                name: "bar_V_gmem_ready",
                kind: MbarrierPlain,
            },
        ],
        "store_k_cache" => &[LocalRef {
            name: "K_frag",
            kind: Frag,
        }],
        "store_v_cache" => &[LocalRef {
            name: "V_frag",
            kind: Frag,
        }],
        // ── Gate + Up + SiLU/GeLU + Mul (MLP input) ──
        "load_wgate_tile" => &[
            LocalRef {
                name: "smem_wgate",
                kind: SmemRing,
            },
            LocalRef {
                name: "bar_smem_wgate",
                kind: MbarrierRing,
            },
        ],
        "load_wup_tile" => &[
            LocalRef {
                name: "smem_wup",
                kind: SmemRing,
            },
            LocalRef {
                name: "bar_smem_wup",
                kind: MbarrierRing,
            },
        ],
        "gate_gemm_accumulate" => &[
            LocalRef {
                name: "Gate_frag",
                kind: Frag,
            },
            LocalRef {
                name: "smem_x",
                kind: SmemRing,
            },
            LocalRef {
                name: "smem_wgate",
                kind: SmemRing,
            },
        ],
        "up_gemm_accumulate" => &[
            LocalRef {
                name: "Up_frag",
                kind: Frag,
            },
            LocalRef {
                name: "smem_x",
                kind: SmemRing,
            },
            LocalRef {
                name: "smem_wup",
                kind: SmemRing,
            },
        ],
        "silu_mul_fuse" => &[
            LocalRef {
                name: "Inter_frag",
                kind: Frag,
            },
            LocalRef {
                name: "Gate_frag",
                kind: Frag,
            },
            LocalRef {
                name: "Up_frag",
                kind: Frag,
            },
        ],
        "store_inter_tile" => &[
            LocalRef {
                name: "Inter_frag",
                kind: Frag,
            },
            LocalRef {
                name: "smem_out",
                kind: SmemPlain,
            },
            LocalRef {
                name: "bar_Inter_gmem_ready",
                kind: MbarrierPlain,
            },
        ],
        // ── Unary in-place (scalar_mul, tanh_softcap) ──
        "scalar_mul" | "tanh_softcap" => &[LocalRef {
            name: "X_frag",
            kind: Frag,
        }],
        // ── Embedding lookup ──
        "load_embed_row" => &[
            LocalRef {
                name: "Embed_frag",
                kind: Frag,
            },
            LocalRef {
                name: "bar_embed",
                kind: MbarrierPlain,
            },
        ],
        "store_embed_row" => &[
            LocalRef {
                name: "Embed_frag",
                kind: Frag,
            },
            LocalRef {
                name: "smem_out",
                kind: SmemPlain,
            },
            LocalRef {
                name: "bar_Y_gmem_ready",
                kind: MbarrierPlain,
            },
        ],
        _ => &[],
    }
}

impl LocalKind {
    /// Widen one declaration against another for the same name.
    /// `SmemPlain` + `SmemRing` → `SmemRing`: the ring storage
    /// subsumes plain since scalar loads decay to slot 0. Same rule for
    /// mbarriers. Conflicting kinds (e.g. `SmemPlain` vs `Frag` under
    /// one name) are caller bugs — we return the left operand as a
    /// deterministic choice rather than panicking during emission.
    pub fn widen(self, other: Self) -> Self {
        use LocalKind::*;
        match (self, other) {
            (SmemPlain, SmemRing) | (SmemRing, SmemPlain) => SmemRing,
            (MbarrierPlain, MbarrierRing) | (MbarrierRing, MbarrierPlain) => MbarrierRing,
            (a, b) if a == b => a,
            (a, _) => a,
        }
    }

    /// The C++ declaration template. `name` is substituted; ring
    /// depths are always `PIPE` (the prelude's compile-time macro).
    pub fn decl(self, name: &str) -> String {
        use LocalKind::*;
        match self {
            SmemPlain => format!("__shared__ StencilFrag {};", name),
            SmemRing => format!("__shared__ StencilFrag {}[PIPE];", name),
            Frag => format!("StencilFrag {};", name),
            FragQkv => format!("struct {{ StencilFrag q, k, v; }} {};", name),
            MbarrierPlain => format!("__shared__ Mbarrier {};", name),
            MbarrierRing => format!("__shared__ Mbarrier {}[PIPE];", name),
            FloatInit(init) => format!("float {} = {};", name, init),
        }
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
        "store_y_row" => Some(generic_store(ctx, "Y_gmem", "Y_frag", "token_tile", "0u")),
        "load_a_row" => Some(generic_preamble_load(ctx, "smem_a", "A_gmem", "token_tile")),
        "load_b_row" => Some(generic_preamble_load(ctx, "smem_b", "B_gmem", "token_tile")),
        "elementwise_add" => Some(elementwise_add(ctx)),
        "store_sum_row" => Some(generic_store(
            ctx,
            "Sum_gmem",
            "sum_frag",
            "token_tile",
            "0u",
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
            "0u",
        )),
        _ => None,
    }
}

fn load_q_tile(ctx: &ExpandCtx<'_>) -> String {
    // `load_q_tile` runs in the preamble (no iter_offset). It reads
    // the Q tile for this CTA's (parallel[0], parallel[1]) coordinate
    // into smem once. On SM90 this is a single TMA `cp.async.bulk.
    // tensor` with an mbarrier arrive; on SM89 it's a cp.async.ca
    // loop over the tile's elements. Row axis varies by template: FA2
    // uses `q_tile`, paged decode uses `b` — both resolve through
    // `ctx.parallel_axes`.
    let row = ctx.par(0);
    let col = ctx.par(1);
    let mut s = String::new();
    writeln!(s, "{{").unwrap();
    writeln!(s, "  // load_q_tile: Q[{}, {}, :, :] → smem", row, col).unwrap();
    match ctx.arch_name {
        "sm90_fa2" => {
            writeln!(s, "  if (wg == LOADER_WG) {{").unwrap();
            writeln!(s, "    tma_load_2d(smem_q, Q_gmem, {}, {});", row, col).unwrap();
            writeln!(s, "    mbarrier_arrive(&bar_q);").unwrap();
            writeln!(s, "  }}").unwrap();
        }
        "sm89_fa2" => {
            writeln!(s, "  cp_async_128(smem_q, Q_gmem, {}, {});", row, col).unwrap();
            writeln!(s, "  cp_async_commit_group();").unwrap();
        }
        other => {
            writeln!(s, "  // unknown arch {} — fall back to generic load", other).unwrap();
            writeln!(s, "  generic_load(smem_q, Q_gmem, {}, {});", row, col).unwrap();
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
    let ser = ctx.ser();
    let col = ctx.par(1);
    let mut s = String::new();
    writeln!(s, "{{").unwrap();
    writeln!(
        s,
        "  // load_{}_tile: {}[{} + {}, {}, :, :] → {}[slot]",
        which, gmem, ser, ctx.iter_offset, col, buf,
    )
    .unwrap();
    writeln!(
        s,
        "  uint32_t slot = ({} + {}) % {};",
        ser, ctx.iter_offset, p
    )
    .unwrap();
    match ctx.arch_name {
        "sm90_fa2" => {
            writeln!(s, "  if (wg == LOADER_WG) {{").unwrap();
            writeln!(
                s,
                "    tma_load_2d({}[slot], {}, {} + {}, {});",
                buf, gmem, ser, ctx.iter_offset, col,
            )
            .unwrap();
            writeln!(s, "    mbarrier_arrive(&bar_kv[slot]);").unwrap();
            writeln!(s, "  }}").unwrap();
        }
        "sm89_fa2" => {
            writeln!(
                s,
                "  cp_async_128({}[slot], {}, {} + {}, {});",
                buf, gmem, ser, ctx.iter_offset, col,
            )
            .unwrap();
            writeln!(s, "  cp_async_commit_group();").unwrap();
        }
        other => {
            writeln!(s, "  // unknown arch {} — fall back to generic load", other).unwrap();
            writeln!(
                s,
                "  generic_load({}[slot], {}, {} + {}, {});",
                buf, gmem, ser, ctx.iter_offset, col,
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
    let ser = ctx.ser();
    let body = match ctx.arch_name {
        "sm90_fa2" => format!(
            concat!(
                "// S_frag = Q_tile @ K_tile^T\n",
                "uint32_t slot = {ser} % PIPE;\n",
                "wgmma_fence();\n",
                "wgmma_mma_async(S_frag, smem_q, smem_k[slot]);\n",
                "wgmma_commit_group();\n",
                "wgmma_wait_group<0>();\n",
            ),
            ser = ser,
        ),
        "sm89_fa2" => format!(
            concat!(
                "// S_frag = Q_tile @ K_tile^T (tiled mma.sync over head_dim)\n",
                "uint32_t slot = {ser} % PIPE;\n",
                "mma_sync_accumulate(S_frag, smem_q, smem_k[slot]);\n",
            ),
            ser = ser,
        ),
        _ => "mma_accumulate(S_frag, smem_q, smem_k);\n".to_string(),
    };
    compute_guarded(ctx, &body)
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
    let ser = ctx.ser();
    let body = match ctx.arch_name {
        "sm90_fa2" => format!(
            concat!(
                "// O_frag += P_frag @ V_tile\n",
                "uint32_t slot = {ser} % PIPE;\n",
                "wgmma_fence();\n",
                "wgmma_mma_async(O_frag, P_frag, smem_v[slot]);\n",
                "wgmma_commit_group();\n",
                "wgmma_wait_group<0>();\n",
                "// release the kv slot back to the loader\n",
                "mbarrier_arrive(&bar_kv_consumed[slot]);\n",
            ),
            ser = ser,
        ),
        "sm89_fa2" => format!(
            concat!(
                "// O_frag += P_frag @ V_tile\n",
                "uint32_t slot = {ser} % PIPE;\n",
                "mma_sync_accumulate(O_frag, P_frag, smem_v[slot]);\n",
            ),
            ser = ser,
        ),
        _ => "mma_accumulate(O_frag, P_frag, smem_v);\n".to_string(),
    };
    compute_guarded(ctx, &body)
}

/// Final store: divide O_frag by l, write to gmem at (q_tile, head_group).
///
/// On SM90 the consumer stages the fragment into smem and signals the
/// storer warpgroup, which performs a TMA store; on SM89 the consumer
/// writes directly to gmem with STG.
fn store_o_tile(ctx: &ExpandCtx<'_>) -> String {
    debug_assert_eq!(ctx.iter_offset, 0, "store nodes carry no iter_offset");
    let row = ctx.par(0);
    let col = ctx.par(1);
    let mut s = String::new();
    writeln!(s, "{{").unwrap();
    writeln!(
        s,
        "  // store_o_tile: O_gmem[{}, {}] ← O_frag / l",
        row, col
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
            writeln!(s, "    tma_store_2d(O_gmem, smem_o, {}, {});", row, col).unwrap();
            writeln!(s, "  }}").unwrap();
        }
        "sm89_fa2" => {
            writeln!(s, "  O_frag = O_frag * rcp(l);").unwrap();
            writeln!(s, "  stg_128(O_gmem, O_frag, {}, {});", row, col).unwrap();
        }
        other => {
            writeln!(
                s,
                "  // unknown arch {} — fall back to generic store",
                other
            )
            .unwrap();
            writeln!(s, "  generic_store(O_gmem, O_frag, l, {}, {});", row, col).unwrap();
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
    // Serial axis drives the slot rotation. Templates that host GEMM
    // accumulate tags consistently name it `k_tile`, but paged-decode
    // variants / future regions could pick a different reduction axis
    // — using `ctx.ser()` keeps us honest either way.
    let ser = ctx.ser();
    let body = match ctx.arch_name {
        "sm90_fa2" => format!(
            concat!(
                "// {acc} += {a} @ {b}\n",
                "uint32_t slot = {ser} % PIPE;\n",
                "wgmma_fence();\n",
                "wgmma_mma_async({acc}, {a}[slot], {b}[slot]);\n",
                "wgmma_commit_group();\n",
                "wgmma_wait_group<0>();\n",
            ),
            ser = ser,
            acc = acc,
            a = smem_a,
            b = smem_b,
        ),
        "sm89_fa2" => format!(
            concat!(
                "// {acc} += {a} @ {b} (tiled mma.sync)\n",
                "uint32_t slot = {ser} % PIPE;\n",
                "mma_sync_accumulate({acc}, {a}[slot], {b}[slot]);\n",
            ),
            ser = ser,
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
                "    tma_store_2d({}, {}, page, slot, head_tile);",
                gmem, frag,
            )
            .unwrap();
            writeln!(s, "  }}").unwrap();
        }
        "sm89_fa2" => {
            writeln!(s, "  stg_128({}, {}, page, slot, head_tile);", gmem, frag).unwrap();
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

    /// FA2-style axis layout: `["q_tile", "head_group"]` parallel +
    /// `Some("kv_tile")` serial. Used by the bulk of the expansion
    /// tests below; attention-specific tags already hardcoded these
    /// names, so keeping them here preserves the existing assertions
    /// while exercising the threading path.
    const FA2_PAR: &[&str] = &["q_tile", "head_group"];
    const FA2_SER: Option<&str> = Some("kv_tile");

    fn ctx(arch: &'static str) -> ExpandCtx<'static> {
        ExpandCtx {
            arch_name: arch,
            iter_offset: 0,
            pipeline_depth: 3,
            parallel_axes: FA2_PAR,
            serial_axis: FA2_SER,
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
            parallel_axes: FA2_PAR,
            serial_axis: FA2_SER,
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
        // GEMM-style axis layout for the A/B/C pipeline load tests:
        // parallel = (m_tile, n_tile), serial = k_tile.
        ExpandCtx {
            arch_name: arch,
            iter_offset: 3,
            pipeline_depth: 3,
            parallel_axes: &["m_tile", "n_tile"],
            serial_axis: Some("k_tile"),
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

    // ── local_refs + LocalKind::widen / decl ──────────────────────

    #[test]
    fn load_q_tile_declares_plain_smem_and_mbarrier() {
        let refs = local_refs("load_q_tile");
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].name, "smem_q");
        assert_eq!(refs[0].kind, LocalKind::SmemPlain);
        assert_eq!(refs[1].name, "bar_q");
        assert_eq!(refs[1].kind, LocalKind::MbarrierPlain);
    }

    #[test]
    fn load_k_and_v_share_bar_kv_ring() {
        let k = local_refs("load_k_tile");
        let v = local_refs("load_v_tile");
        // Both declare a ring mbarrier `bar_kv` — collect_locals
        // dedupes by name so one `bar_kv[PIPE]` is emitted per region.
        assert!(
            k.iter()
                .any(|r| r.name == "bar_kv" && r.kind == LocalKind::MbarrierRing)
        );
        assert!(
            v.iter()
                .any(|r| r.name == "bar_kv" && r.kind == LocalKind::MbarrierRing)
        );
    }

    #[test]
    fn softmax_update_carries_float_accumulators_with_inits() {
        let refs = local_refs("softmax_update");
        let m = refs.iter().find(|r| r.name == "m").unwrap();
        assert_eq!(m.kind, LocalKind::FloatInit("-INFINITY"));
        let l = refs.iter().find(|r| r.name == "l").unwrap();
        assert_eq!(l.kind, LocalKind::FloatInit("0.0f"));
    }

    #[test]
    fn local_kind_widen_promotes_plain_to_ring() {
        use LocalKind::*;
        assert_eq!(SmemPlain.widen(SmemRing), SmemRing);
        assert_eq!(SmemRing.widen(SmemPlain), SmemRing);
        assert_eq!(MbarrierPlain.widen(MbarrierRing), MbarrierRing);
        assert_eq!(MbarrierRing.widen(MbarrierPlain), MbarrierRing);
        // Same-kind is idempotent.
        assert_eq!(SmemRing.widen(SmemRing), SmemRing);
        assert_eq!(Frag.widen(Frag), Frag);
    }

    #[test]
    fn local_kind_decl_renders_expected_cpp() {
        assert_eq!(
            LocalKind::SmemPlain.decl("smem_x"),
            "__shared__ StencilFrag smem_x;"
        );
        assert_eq!(
            LocalKind::SmemRing.decl("smem_a"),
            "__shared__ StencilFrag smem_a[PIPE];"
        );
        assert_eq!(LocalKind::Frag.decl("C_frag"), "StencilFrag C_frag;");
        assert_eq!(
            LocalKind::FragQkv.decl("QKV_frag"),
            "struct { StencilFrag q, k, v; } QKV_frag;"
        );
        assert_eq!(
            LocalKind::MbarrierPlain.decl("bar_q"),
            "__shared__ Mbarrier bar_q;"
        );
        assert_eq!(
            LocalKind::MbarrierRing.decl("bar_kv"),
            "__shared__ Mbarrier bar_kv[PIPE];"
        );
        assert_eq!(
            LocalKind::FloatInit("-INFINITY").decl("m"),
            "float m = -INFINITY;"
        );
    }

    // ── Axis-name threading (item 3) ──────────────────────────────

    #[test]
    fn load_q_tile_paged_decode_uses_b_axis_not_q_tile() {
        // paged_decode's parallel axes are (b, head_group). load_q_tile
        // hard-coded `q_tile` in prior revisions; item 3 routes it
        // through `ctx.par(0)` so paged decode emits `b`.
        let ctx = ExpandCtx {
            arch_name: "sm90_fa2",
            iter_offset: 0,
            pipeline_depth: 3,
            parallel_axes: &["b", "head_group"],
            serial_axis: Some("kv_tile"),
        };
        let s = expand("load_q_tile", &ctx).unwrap();
        assert!(s.contains("tma_load_2d(smem_q, Q_gmem, b, head_group)"));
        assert!(!s.contains("tma_load_2d(smem_q, Q_gmem, q_tile"));
    }

    #[test]
    fn store_o_tile_paged_decode_uses_b_axis() {
        let ctx = ExpandCtx {
            arch_name: "sm90_fa2",
            iter_offset: 0,
            pipeline_depth: 3,
            parallel_axes: &["b", "head_group"],
            serial_axis: Some("kv_tile"),
        };
        let s = expand("store_o_tile", &ctx).unwrap();
        assert!(s.contains("tma_store_2d(O_gmem, smem_o, b, head_group)"));
    }

    #[test]
    fn qk_matmul_slot_calc_follows_serial_axis() {
        // Same tag, two regions, two slot calcs — FA2 uses kv_tile,
        // but a theoretical kv_tile-renamed region would get the
        // correct name.
        let fa2 = ExpandCtx {
            arch_name: "sm90_fa2",
            iter_offset: 0,
            pipeline_depth: 3,
            parallel_axes: FA2_PAR,
            serial_axis: Some("kv_tile"),
        };
        assert!(
            expand("qk_matmul", &fa2)
                .unwrap()
                .contains("uint32_t slot = kv_tile % PIPE;")
        );

        let renamed = ExpandCtx {
            arch_name: "sm90_fa2",
            iter_offset: 0,
            pipeline_depth: 3,
            parallel_axes: FA2_PAR,
            serial_axis: Some("kv_chunk"),
        };
        assert!(
            expand("qk_matmul", &renamed)
                .unwrap()
                .contains("uint32_t slot = kv_chunk % PIPE;")
        );
    }

    #[test]
    fn generic_gemm_slot_calc_follows_region_serial_axis() {
        // GEMM regions use k_tile as the serial axis — threading
        // shows up in gemm_accumulate too.
        let gemm_ctx = ExpandCtx {
            arch_name: "sm90_fa2",
            iter_offset: 0,
            pipeline_depth: 3,
            parallel_axes: &["m_tile", "n_tile"],
            serial_axis: Some("k_tile"),
        };
        let s = expand("gemm_accumulate", &gemm_ctx).unwrap();
        assert!(s.contains("uint32_t slot = k_tile % PIPE;"));
    }

    #[test]
    fn qkv_matmul_carries_ring_smem_operands() {
        // qkv_matmul indexes `smem_x[slot]` and `smem_wqkv[slot]` via
        // generic_gemm. Its local_refs declare both as SmemRing so
        // that in a region that also contains `load_x_row`
        // (SmemPlain), widening produces a single ring declaration.
        let refs = local_refs("qkv_matmul");
        assert!(
            refs.iter()
                .any(|r| r.name == "smem_x" && r.kind == LocalKind::SmemRing)
        );
        assert!(
            refs.iter()
                .any(|r| r.name == "smem_wqkv" && r.kind == LocalKind::SmemRing)
        );
        assert!(
            refs.iter()
                .any(|r| r.name == "QKV_frag" && r.kind == LocalKind::FragQkv)
        );
    }
}
