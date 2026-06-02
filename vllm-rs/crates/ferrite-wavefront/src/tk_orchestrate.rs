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
use crate::routing::{
    classify_inputs, classify_outputs, coalesce_carry_forwards, InputRouting, OutputRouting,
};
use crate::subtile_ir::BufId;
use crate::tk_lower::{
    lower_attn_decode, lower_attn_decode_routed, lower_gemm_m1, lower_gemm_m1_routed,
    lower_residual_add, lower_residual_add_routed, lower_rmsnorm, lower_rmsnorm_routed,
    lower_rope_rotate, lower_rope_rotate_routed, lower_silu_mul, lower_silu_mul_routed, AddOp,
    AttnDecodeOp, CarriedHandle, GemmM1Op, PageAllocator, RmsNormOp, RopeRotateOp, RoutingHints,
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

/// Phase 12 — build routing hints for op `op_idx` from the
/// pre-computed analysis vectors and the running `carried_table`
/// side table. For each input slot:
///   - `InputRouting::CarryForward { producer_op_idx }` →
///     `Some(carried_table[producer_op_idx])` (the producer must
///     have stored its `CarriedHandle` already; producers are visited
///     before consumers in topo order).
///   - `InputRouting::GmemLoad` → `None`.
fn build_routing_hints(
    op_idx: usize,
    input_routing: &[Vec<InputRouting>],
    output_routing: &[OutputRouting],
    carried_table: &[Option<CarriedHandle>],
) -> RoutingHints {
    let inputs: Vec<Option<CarriedHandle>> = input_routing[op_idx]
        .iter()
        .map(|ir| match ir {
            InputRouting::CarryForward { producer_op_idx } => {
                carried_table.get(*producer_op_idx).copied().flatten()
            }
            InputRouting::GmemLoad => None,
        })
        .collect();
    RoutingHints {
        inputs,
        output_internal: output_routing[op_idx].is_internal(),
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

    // Phase 12 — routing: when `FERRITE_NEW_ROUTING` is set, walk the
    // DAG once to classify each op's outputs as internal/external and
    // each op's input slots as carry-forward/gmem-load. Producer
    // CarriedHandles thread through `carried_table[op_idx]` to the
    // consuming op's hints. Off by default → byte-identity to the
    // legacy lowerings.
    let routing_on = std::env::var_os("FERRITE_NEW_ROUTING").is_some();
    let (output_routing, input_routing) = if routing_on {
        let mut outs = classify_outputs(input);
        let ins = classify_inputs(input, &outs);
        // Demote producers whose CarryForward got dropped by the
        // "at most one carry-forward per consumer" rule. They must
        // drain to gmem so the consumer can TMA-load safely.
        coalesce_carry_forwards(&mut outs, &ins);
        (Some(outs), Some(ins))
    } else {
        (None, None)
    };
    let mut carried_table: Vec<Option<CarriedHandle>> = vec![None; input.ops.len()];

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

    /// Phase 12 — routed dispatch. Returns `RoutingResult`. Picks
    /// `Phase0` vs `Phase1` from carry-forward inputs' parities (all
    /// must agree); falls back to free-slot count when no carry-
    /// forward inputs.
    #[allow(unused_macros)]
    macro_rules! dispatch_phase_routed {
        ($n_pages:expr, $f:ident, $op:expr, $hints:expr) => {{
            let n: usize = $n_pages;
            let hints_ref: &RoutingHints = $hints;
            let parities: Vec<u32> = hints_ref
                .inputs
                .iter()
                .filter_map(|h| h.as_ref().map(|c| c.phase))
                .collect();
            if !parities.is_empty() {
                debug_assert!(
                    parities.iter().all(|&p| p == parities[0]),
                    "lower_to_tk: carry-forward inputs have inconsistent parities for {}: {:?}",
                    stringify!($f),
                    parities,
                );
            }
            let chosen = parities.first().copied();
            match chosen {
                Some(0) => $f::<Phase0>($op, hints_ref, &mut pages, &mut prog),
                Some(1) => $f::<Phase1>($op, hints_ref, &mut pages, &mut prog),
                Some(other) => panic!(
                    "lower_to_tk: invalid carry-forward parity {} for {}",
                    other,
                    stringify!($f),
                ),
                None => {
                    if pages.count_at(0) >= n {
                        $f::<Phase0>($op, hints_ref, &mut pages, &mut prog)
                    } else {
                        $f::<Phase1>($op, hints_ref, &mut pages, &mut prog)
                    }
                }
            }
        }};
    }

    /// Combined dispatch: routed when `hints_opt` is `Some`; legacy
    /// otherwise. Stores any returned `output_carried` into
    /// `carried_table[op_idx]`. Caller passes `op_idx`, `hints_opt`,
    /// and `carried_table` explicitly so macro hygiene doesn't trip
    /// on the captured for-loop variable.
    macro_rules! dispatch_phase_maybe_routed {
        ($op_idx:expr, $hints_opt:expr, $carried:expr,
         $n_pages:expr, $f_legacy:ident, $f_routed:ident, $op:expr) => {{
            if let Some(ref hints) = $hints_opt {
                let result = dispatch_phase_routed!($n_pages, $f_routed, $op, hints);
                $carried[$op_idx] = result.output_carried;
            } else {
                dispatch_phase!($n_pages, $f_legacy, $op);
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

        // Phase 12: build per-op routing hints from the analysis +
        // running side table. None when routing is off (legacy path).
        let hints_opt: Option<RoutingHints> =
            if let (Some(ir), Some(or_)) = (input_routing.as_ref(), output_routing.as_ref()) {
                Some(build_routing_hints(op_idx, ir, or_, &carried_table))
            } else {
                None
            };

        match desc.op {
            LoweredOp::RmsNorm { eps } => {
                let x = buf_for(desc.inputs[0], &op_out_buf);
                let weight = buf_for(desc.inputs[1], &op_out_buf);
                let hidden = shape_for(desc.inputs[0], &op_out_shape, &input.sources).1;
                dispatch_phase_maybe_routed!(
                    op_idx, hints_opt, carried_table,
                    2,
                    lower_rmsnorm,
                    lower_rmsnorm_routed,
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
                dispatch_phase_maybe_routed!(
                    op_idx, hints_opt, carried_table,
                    3,
                    lower_gemm_m1,
                    lower_gemm_m1_routed,
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
                dispatch_phase_maybe_routed!(
                    op_idx, hints_opt, carried_table,
                    2,
                    lower_silu_mul,
                    lower_silu_mul_routed,
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
                dispatch_phase_maybe_routed!(
                    op_idx, hints_opt, carried_table,
                    2,
                    lower_residual_add,
                    lower_residual_add_routed,
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
                // RopeAppend's `desc.inputs` is `[K, cos, sin, V]`
                // (4 entries) but the rotate lowering only consumes
                // x/cos/sin. Truncate hints to match before dispatch.
                let hints_opt = hints_opt.as_ref().map(|h| RoutingHints {
                    inputs: h.inputs.iter().take(3).cloned().collect(),
                    output_internal: h.output_internal,
                });
                dispatch_phase_maybe_routed!(
                    op_idx, hints_opt, carried_table,
                    3,
                    lower_rope_rotate,
                    lower_rope_rotate_routed,
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
                // AttnDecode's `desc.inputs` is `[Q, (K_seg, V_seg)...]`
                // — Q + paged-cache segment pairs (often >3). The
                // routed lowering only consumes Q/k_cache/v_cache (3).
                // Truncate hints to match.
                let hints_opt = hints_opt.as_ref().map(|h| RoutingHints {
                    inputs: h.inputs.iter().take(3).cloned().collect(),
                    output_internal: h.output_internal,
                });
                dispatch_phase_maybe_routed!(
                    op_idx, hints_opt, carried_table,
                    3,
                    lower_attn_decode,
                    lower_attn_decode_routed,
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
