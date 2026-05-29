// SPDX-License-Identifier: Apache-2.0
//! Orchestrator: [`crate::lower::LoweringInput`] → [`TkProgram`].
//!
//! Walks the topologically-ordered op list and dispatches each op to
//! the matching `lower_*` in [`crate::tk_lower`]. The orchestrator's
//! only job is to:
//!
//! 1. Assign deterministic [`BufId`]s to every external source and
//!    every op's output staging buffer.
//! 2. Resolve each op's input/output [`BufId`]s via [`InputRef`]
//!    indirection.
//! 3. Infer the few remaining shape parameters the per-op lowerings
//!    need but `LoweredOp` doesn't carry directly (e.g. `hidden` from a
//!    source's `cols`).
//!
//! BufId convention: external sources occupy `BufId(0..n_sources)`;
//! op output staging buffers occupy
//! `BufId(n_sources..n_sources + n_ops)`.
//!
//! GEMM `BN` selection is derived per call from the page-fit constraint
//! `bn * k * act_elem <= PAGE_SIZE` (see [`pick_bn`]) — independent of
//! any specific model. The orchestrator refuses any GEMM whose `(bn, k)`
//! pair falls outside that envelope, making the page-fit constraint a
//! structural property of the IR.

use crate::lower::{InputRef, LoweredOp, LoweringInput};
use crate::subtile_ir::BufId;
use crate::tk_lower::{
    lower_attn_decode, lower_gemm_m1, lower_residual_add, lower_rmsnorm, lower_rope_rotate,
    lower_silu_mul, AddOp, AttnDecodeOp, GemmM1Op, PageAllocator, RmsNormOp, RopeRotateOp,
    SiluMulOp,
};
use crate::tk_warp_ir::{Phase0, Phase1, TkProgram, WarpRole, PAGE_SIZE};

/// Activation element bytes assumed across the decode forward.
/// bf16 = 2 bytes. The substrate is bf16-only today.
pub const ACT_ELEM: u32 = 2;

/// Short kind tag for each [`LoweredOp`] — used by the per-op trace
/// marker the orchestrator inserts before every op's lowering, so
/// `EmitOpts::debug_handshake` printf traces can be partitioned by op.
fn op_kind_name(op: &LoweredOp) -> &'static str {
    match op {
        LoweredOp::Gemm { .. } => "Gemm",
        LoweredOp::RmsNorm { .. } => "RmsNorm",
        LoweredOp::Silu => "Silu",
        LoweredOp::Mul => "Mul",
        LoweredOp::SiluMul => "SiluMul",
        LoweredOp::Add => "Add",
        LoweredOp::RopeRotate { .. } => "RopeRotate",
        LoweredOp::RopeAppend { .. } => "RopeAppend",
        LoweredOp::AttnDecode { .. } => "AttnDecode",
    }
}

/// Pick `bn` (W-tile rows per page load) for a given GEMM `k`.
/// Constraint: `bn * k * ACT_ELEM <= PAGE_SIZE` AND `n_blocks = ceil(n / bn)`
/// must be even (the orchestrator can't statically track post-loop parity
/// for an odd-N count). The caller is responsible for shaping `n` so the
/// resulting `n_blocks` is even.
fn pick_bn(k: u32) -> u32 {
    let bytes_per_row = k * ACT_ELEM;
    let max_bn = PAGE_SIZE / bytes_per_row;
    // Cap at 8 — there are only 8 consumer warps and our compute body
    // assigns one warp per output value (warps with `c >= bn` idle).
    max_bn.min(8).max(1)
}

/// Lower an entire forward (a [`LoweringInput`]) to a single
/// [`TkProgram`] — the persistent megakernel body.
///
/// Returns the program plus the count of buffer ids used (so the
/// kernel scaffold knows how many pointer parameters to emit).
pub fn lower_to_tk(input: &LoweringInput) -> (TkProgram, u32) {
    let mut prog = TkProgram::new();
    let mut pages = PageAllocator::new();
    let n_sources = input.sources.len() as u32;
    let mut op_out_buf: Vec<BufId> = Vec::with_capacity(input.ops.len());

    let buf_for = |r: InputRef, op_out_buf: &[BufId]| -> BufId {
        match r {
            InputRef::Ext(e) => BufId(e as u32),
            InputRef::Op(j) => op_out_buf[j],
        }
    };

    let shape_for =
        |r: InputRef, op_out_shape: &[(u32, u32)], sources: &[crate::subtile::SourceShape]| -> (u32, u32) {
            match r {
                InputRef::Ext(e) => (sources[e].rows, sources[e].cols),
                InputRef::Op(j) => op_out_shape[j],
            }
        };

    let mut op_out_shape: Vec<(u32, u32)> = Vec::with_capacity(input.ops.len());

    /// Dispatch one `lower_X<P>` based on which parity has enough free
    /// slots. Phase0 is preferred (fresh allocator state); Phase1 is
    /// used for slot reuse after the first round of any slot. If neither
    /// parity has `n_pages` free, falls back to whichever has more — the
    /// inner `alloc_at::<P>()` will then panic with the offending op
    /// name, which is the visible failure we want.
    macro_rules! dispatch_phase {
        ($n_pages:expr, $f:ident, $op:expr) => {{
            let n: usize = $n_pages;
            if pages.count_at(0) >= n {
                $f::<Phase0>($op, &mut pages, &mut prog);
            } else {
                $f::<Phase1>($op, &mut pages, &mut prog);
            }
        }};
    }

    for (op_idx, desc) in input.ops.iter().enumerate() {
        let out_buf = BufId(n_sources + op_idx as u32);
        op_out_buf.push(out_buf);

        // Per-op trace marker — fires from thread 0 (warp 0 lane 0) so
        // the printf trace tagged by `EmitOpts::debug_handshake` can be
        // partitioned by op. Wrapped in `#ifdef TK_DEBUG_HANDSHAKE` so
        // it's a compile-time no-op when the dbg-handshake instrumentation
        // is off (the same macro `tk_codegen` toggles for the per-op
        // wait/arrive printfs). One thread, one line per op — negligible
        // even if always on.
        let op_tag = format!("op{op_idx}/{}", op_kind_name(&desc.op));
        prog.compute(
            WarpRole::All,
            format!(
                "#ifdef TK_DEBUG_HANDSHAKE\n        \
                 if (threadIdx.x == 0) {{ \
                 printf(\"[BEGIN {op_tag}]\\n\"); }}\n        \
                 #endif"
            ),
        );

        match desc.op {
            LoweredOp::RmsNorm { eps } => {
                let x = buf_for(desc.inputs[0], &op_out_buf);
                let weight = buf_for(desc.inputs[1], &op_out_buf);
                let hidden = shape_for(desc.inputs[0], &op_out_shape, &input.sources).1;
                dispatch_phase!(
                    2,
                    lower_rmsnorm,
                    RmsNormOp {
                        x,
                        weight,
                        out: out_buf,
                        hidden,
                        m: desc.m,
                        act_elem: ACT_ELEM,
                        eps,
                        init: false,
                    }
                );
                op_out_shape.push((desc.m, hidden));
            }

            LoweredOp::Gemm { n, k } => {
                let x = buf_for(desc.inputs[0], &op_out_buf);
                let w = buf_for(desc.inputs[1], &op_out_buf);
                let bn = pick_bn(k);
                dispatch_phase!(
                    3,
                    lower_gemm_m1,
                    GemmM1Op {
                        x,
                        w,
                        out: out_buf,
                        k,
                        n,
                        bn,
                        act_elem: ACT_ELEM,
                    }
                );
                op_out_shape.push((desc.m, n));
            }

            LoweredOp::SiluMul => {
                let gate = buf_for(desc.inputs[0], &op_out_buf);
                let up = buf_for(desc.inputs[1], &op_out_buf);
                let intermediate = shape_for(desc.inputs[0], &op_out_shape, &input.sources).1;
                dispatch_phase!(
                    2,
                    lower_silu_mul,
                    SiluMulOp {
                        gate,
                        up,
                        out: out_buf,
                        intermediate,
                        m: desc.m,
                        act_elem: ACT_ELEM,
                    }
                );
                op_out_shape.push((desc.m, intermediate));
            }

            LoweredOp::Add => {
                let a = buf_for(desc.inputs[0], &op_out_buf);
                let b = buf_for(desc.inputs[1], &op_out_buf);
                let hidden = shape_for(desc.inputs[0], &op_out_shape, &input.sources).1;
                dispatch_phase!(
                    2,
                    lower_residual_add,
                    AddOp {
                        a,
                        b,
                        out: out_buf,
                        hidden,
                        m: desc.m,
                        act_elem: ACT_ELEM,
                    }
                );
                op_out_shape.push((desc.m, hidden));
            }

            LoweredOp::RopeRotate { head_dim }
            | LoweredOp::RopeAppend { head_dim, layer: _ } => {
                let x = buf_for(desc.inputs[0], &op_out_buf);
                let cos = buf_for(desc.inputs[1], &op_out_buf);
                let sin = buf_for(desc.inputs[2], &op_out_buf);
                let cols = shape_for(desc.inputs[0], &op_out_shape, &input.sources).1;
                let num_heads = cols / head_dim;
                dispatch_phase!(
                    3,
                    lower_rope_rotate,
                    RopeRotateOp {
                        x,
                        cos,
                        sin,
                        out: out_buf,
                        head_dim,
                        num_heads,
                        m: desc.m,
                        act_elem: ACT_ELEM,
                    }
                );
                op_out_shape.push((desc.m, cols));
            }

            LoweredOp::AttnDecode {
                num_q_heads,
                num_kv_heads,
                head_dim,
                scale,
            } => {
                // Multi-head GQA: q has `num_q_heads` heads of `head_dim`
                // each (`[1, num_q_heads * head_dim]`); kv has
                // `num_kv_heads` heads (`num_q_heads / num_kv_heads`
                // q-heads share each kv-head). Output is per-head
                // attention concatenated.
                let q = buf_for(desc.inputs[0], &op_out_buf);
                let k_cache = buf_for(desc.inputs[1], &op_out_buf);
                let v_cache = buf_for(desc.inputs[2], &op_out_buf);
                dispatch_phase!(
                    3,
                    lower_attn_decode,
                    AttnDecodeOp {
                        q,
                        k_cache,
                        v_cache,
                        out: out_buf,
                        head_dim,
                        num_q_heads,
                        num_kv_heads,
                        act_elem: ACT_ELEM,
                        softmax_scale: scale,
                        num_kv_pages_arg: "__num_kv_pages",
                        unique_id: op_idx as u32,
                    }
                );
                op_out_shape.push((desc.m, num_q_heads * head_dim));
            }

            // Standalone Silu/Mul should be fused into SiluMul before
            // orchestration (see crate::lower::fuse_silu_mul). The
            // megakernel substrate has no standalone-Silu primitive.
            LoweredOp::Silu | LoweredOp::Mul => {
                panic!(
                    "lower_to_tk: standalone {:?} reached the orchestrator; \
                     fuse_silu_mul must run before lower_to_tk.",
                    desc.op
                );
            }
        }
    }

    let n_bufs = n_sources + input.ops.len() as u32;
    (prog, n_bufs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::one_layer_input;
    use crate::tk_codegen::emit_body;

    #[test]
    fn pick_bn_keeps_w_tile_inside_one_page() {
        // K=2048 → bn ≤ floor(16384 / (2048*2)) = 4
        assert_eq!(pick_bn(2048), 4);
        // K=8192 → bn ≤ 1
        assert_eq!(pick_bn(8192), 1);
        // bn is always at least 1 even if k * ACT_ELEM > PAGE_SIZE
        // (caller is responsible for ensuring page fit).
        assert_eq!(pick_bn(16384), 1);
    }

    /// One-layer forward lowers to a TkProgram with no panics, and the
    /// emitted body contains markers from each per-op lowering.
    #[test]
    fn one_layer_forward_lowers_end_to_end() {
        let input = one_layer_input();
        let (prog, n_bufs) = lower_to_tk(&input);
        let src = emit_body(&prog);

        // 14 sources + 12 ops = 26 buffer ids.
        assert_eq!(n_bufs, 14 + 12);

        // Each op's body comment fired at least once.
        assert!(src.contains("RmsNorm"), "{src}");
        assert!(src.contains("GemmM1"), "{src}");
        assert!(src.contains("RoPE rotate"), "{src}");
        assert!(src.contains("AttnDecode"), "{src}");
        assert!(src.contains("SiluMul"), "{src}");
        assert!(src.contains("Residual Add"), "{src}");
    }

    /// Sanity: every BufId referenced in an emitted load/store points
    /// inside `[0, n_bufs)` so the kernel scaffold can bind them all.
    #[test]
    fn emitted_buf_ids_stay_in_range() {
        use crate::tk_warp_ir::TkInstr;
        let input = one_layer_input();
        let (prog, n_bufs) = lower_to_tk(&input);
        let mut walk: Vec<&TkInstr> = prog.instrs.iter().collect();
        let mut i = 0;
        while i < walk.len() {
            if let TkInstr::ForLoop { body, .. } = walk[i] {
                walk.extend(body.iter());
            }
            i += 1;
        }
        for instr in &walk {
            match instr {
                TkInstr::LoadAsync { src, .. } => {
                    assert!(src.0 < n_bufs, "LoadAsync src buf {} >= n_bufs={n_bufs}", src.0);
                }
                TkInstr::StoreAsync { dst, .. } => {
                    assert!(dst.0 < n_bufs, "StoreAsync dst buf {} >= n_bufs={n_bufs}", dst.0);
                }
                _ => {}
            }
        }
    }
}
