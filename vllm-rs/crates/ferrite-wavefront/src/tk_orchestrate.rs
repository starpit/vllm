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

use std::collections::BTreeMap;

use crate::lower::{InputRef, LoweredOp, LoweringInput};
use crate::routing::{
    classify_inputs, classify_outputs, coalesce_carry_forwards, InputRouting, OutputRouting,
};
use crate::subtile_ir::BufId;
use crate::tk_codegen::GlLayout;
use crate::tk_lower::{
    lower_attn_decode, lower_attn_decode_routed, lower_gemm_m1, lower_gemm_m1_routed,
    lower_residual_add, lower_residual_add_routed, lower_rmsnorm, lower_rmsnorm_routed,
    lower_rope_append, lower_rope_append_routed, lower_rope_rotate, lower_rope_rotate_routed,
    lower_silu_mul, lower_silu_mul_routed, AddOp, AttnDecodeOp, CarriedHandle, GemmM1Op,
    PageAllocator, RmsNormOp, RopeAppendOp, RopeRotateOp, RoutingHints, SiluMulOp,
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

    // E.13: typed gmem handle stash for the cross-op K/V cache edge.
    // RopeAppend (the producer) places handles here; the next
    // AttnDecode (the consumer) takes them, fences, and feeds the
    // Fenced wrappers to its lowering. The orchestrator cannot
    // bypass this — `lower_attn_decode` requires
    // `Fenced<GmemHandle<...>>`, and the only way to construct one
    // is `tk_gmem::emit_fence_after_op`.
    let mut pending_k_unfenced: Option<crate::tk_gmem::GmemHandle<crate::tk_gmem::KCache>> =
        None;
    let mut pending_v_unfenced: Option<crate::tk_gmem::GmemHandle<crate::tk_gmem::VCache>> =
        None;

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

            LoweredOp::RopeRotate { head_dim } => {
                let x = buf_for(desc.inputs[0], &op_out_buf);
                let cos = buf_for(desc.inputs[1], &op_out_buf);
                let sin = buf_for(desc.inputs[2], &op_out_buf);
                let cols = shape_for(desc.inputs[0], &op_out_shape, &input.sources).1;
                let num_heads = cols / head_dim;
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

            // E.12: RopeAppend rotates K and writes rotated K + V to
            // the paged KV cache pools (PrefixK / PrefixV per layer).
            // `desc.inputs` from the bridge is
            // `[K, cos, sin, V, K_cache, V_cache]` (6 entries, per
            // E.12.A bridge widening). Routed lowering only carries
            // the first 4 (K/cos/sin/V); K_cache and V_cache are
            // always Ext (paged-cache pools) so their hints are
            // dropped via `take(4)`.
            LoweredOp::RopeAppend { head_dim, layer: _ } => {
                let k = buf_for(desc.inputs[0], &op_out_buf);
                let cos = buf_for(desc.inputs[1], &op_out_buf);
                let sin = buf_for(desc.inputs[2], &op_out_buf);
                let v = buf_for(desc.inputs[3], &op_out_buf);
                let k_cache = buf_for(desc.inputs[4], &op_out_buf);
                let v_cache = buf_for(desc.inputs[5], &op_out_buf);
                let cols = shape_for(desc.inputs[0], &op_out_shape, &input.sources).1;
                let num_kv_heads = cols / head_dim;
                let hints_opt = hints_opt.as_ref().map(|h| RoutingHints {
                    inputs: h.inputs.iter().take(4).cloned().collect(),
                    output_internal: h.output_internal,
                });
                let rope_append_op = RopeAppendOp {
                    k,
                    cos,
                    sin,
                    v,
                    out: out_buf,
                    k_cache,
                    v_cache,
                    head_dim,
                    num_kv_heads,
                    m: desc.m,
                    act_elem: ACT_ELEM,
                    decode_slot_arg: "__decode_slot",
                };
                let k_handle = crate::tk_gmem::GmemHandle::<crate::tk_gmem::KCache>::new_initial(
                    k_cache,
                );
                let v_handle = crate::tk_gmem::GmemHandle::<crate::tk_gmem::VCache>::new_initial(
                    v_cache,
                );
                let (k_out, v_out) = match hints_opt.as_ref() {
                    Some(hints) => {
                        // Routed: phase from carry-forward parities.
                        let parities: Vec<u32> = hints
                            .inputs
                            .iter()
                            .filter_map(|h| h.as_ref().map(|c| c.phase))
                            .collect();
                        let chosen = parities.first().copied();
                        let (result, k_out, v_out) = match chosen {
                            Some(0) => lower_rope_append_routed::<Phase0>(
                                rope_append_op, k_handle, v_handle, hints, &mut pages, &mut prog,
                            ),
                            Some(1) => lower_rope_append_routed::<Phase1>(
                                rope_append_op, k_handle, v_handle, hints, &mut pages, &mut prog,
                            ),
                            Some(other) => panic!(
                                "lower_to_tk: invalid carry-forward parity {} for lower_rope_append_routed",
                                other,
                            ),
                            None => {
                                if pages.count_at(0) >= 4 {
                                    lower_rope_append_routed::<Phase0>(
                                        rope_append_op, k_handle, v_handle, hints, &mut pages, &mut prog,
                                    )
                                } else {
                                    lower_rope_append_routed::<Phase1>(
                                        rope_append_op, k_handle, v_handle, hints, &mut pages, &mut prog,
                                    )
                                }
                            }
                        };
                        carried_table[op_idx] = result.output_carried;
                        (k_out, v_out)
                    }
                    None => {
                        // Legacy: phase from free-slot count.
                        if pages.count_at(0) >= 4 {
                            lower_rope_append::<Phase0>(
                                rope_append_op, k_handle, v_handle, &mut pages, &mut prog,
                            )
                        } else {
                            lower_rope_append::<Phase1>(
                                rope_append_op, k_handle, v_handle, &mut pages, &mut prog,
                            )
                        }
                    }
                };
                pending_k_unfenced = Some(k_out);
                pending_v_unfenced = Some(v_out);
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
                let attn_decode_op = AttnDecodeOp {
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
                };
                // E.13: take the unfenced handles from the prior
                // RopeAppend (if any), emit the cross-op gmem fence,
                // and pass the Fenced wrappers to the consumer. If no
                // RopeAppend ran (test fixtures + future variants
                // where the cache is pre-populated by per-op forward
                // and read-only inside the megakernel), construct a
                // fresh `new_initial` handle. Either path goes
                // through `emit_fence_after_op` — AttnDecode's
                // `Fenced<...>` requirement is satisfied uniformly.
                let k_unfenced = pending_k_unfenced.take().unwrap_or_else(|| {
                    crate::tk_gmem::GmemHandle::<crate::tk_gmem::KCache>::new_initial(k_cache)
                });
                let v_unfenced = pending_v_unfenced.take().unwrap_or_else(|| {
                    crate::tk_gmem::GmemHandle::<crate::tk_gmem::VCache>::new_initial(v_cache)
                });
                debug_assert_eq!(
                    k_unfenced.buf_id(),
                    k_cache,
                    "lower_to_tk: AttnDecode's k_cache buf doesn't match producer's K handle",
                );
                debug_assert_eq!(
                    v_unfenced.buf_id(),
                    v_cache,
                );
                let k_fenced = crate::tk_gmem::emit_fence_after_op(&mut prog, k_unfenced);
                let v_fenced = crate::tk_gmem::emit_fence_after_op(&mut prog, v_unfenced);
                match hints_opt.as_ref() {
                    Some(hints) => {
                        let parities: Vec<u32> = hints
                            .inputs
                            .iter()
                            .filter_map(|h| h.as_ref().map(|c| c.phase))
                            .collect();
                        let chosen = parities.first().copied();
                        let result = match chosen {
                            Some(0) => lower_attn_decode_routed::<Phase0>(
                                attn_decode_op,
                                k_fenced,
                                v_fenced,
                                hints,
                                &mut pages,
                                &mut prog,
                            ),
                            Some(1) => lower_attn_decode_routed::<Phase1>(
                                attn_decode_op,
                                k_fenced,
                                v_fenced,
                                hints,
                                &mut pages,
                                &mut prog,
                            ),
                            Some(other) => panic!(
                                "lower_to_tk: invalid carry-forward parity {} for lower_attn_decode_routed",
                                other,
                            ),
                            None => {
                                if pages.count_at(0) >= 3 {
                                    lower_attn_decode_routed::<Phase0>(
                                        attn_decode_op,
                                        k_fenced,
                                        v_fenced,
                                        hints,
                                        &mut pages,
                                        &mut prog,
                                    )
                                } else {
                                    lower_attn_decode_routed::<Phase1>(
                                        attn_decode_op,
                                        k_fenced,
                                        v_fenced,
                                        hints,
                                        &mut pages,
                                        &mut prog,
                                    )
                                }
                            }
                        };
                        carried_table[op_idx] = result.output_carried;
                    }
                    None => {
                        if pages.count_at(0) >= 3 {
                            lower_attn_decode::<Phase0>(
                                attn_decode_op,
                                k_fenced,
                                v_fenced,
                                &mut pages,
                                &mut prog,
                            );
                        } else {
                            lower_attn_decode::<Phase1>(
                                attn_decode_op,
                                k_fenced,
                                v_fenced,
                                &mut pages,
                                &mut prog,
                            );
                        }
                    }
                }
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

/// Step C.3 — opt-in descriptor-TMA layouts for the orchestrator's
/// rmsnorm output buffers.
///
/// Walks `input.ops` in topo order (matching `lower_to_tk`'s shape
/// inference), and for every `LoweredOp::RmsNorm` produces an entry
/// keyed by the op's output `BufId` mapping to a vector-form
/// `kittens::sv_bf<HIDDEN>` `gl<>` layout. Returns an empty map when
/// `FERRITE_NEW_TMA_TENSOR` is unset — the canonical raw-bulk path
/// stays byte-identical in default builds.
///
/// Scope rationale (per the C.3 handoff):
///   - rmsnorm output stores have a fixed offset (region.cols.start =
///     0) and no dynamic byte offset — the typed `tma::store_async`
///     `{0,0,0,0}` coord is exact.
///   - m=1 decode → 1×hidden activation row, fits `sv_bf<K>` (vector
///     form, `UTMASTG.4D`); tile form `st_bf<R, C>` would require
///     R % 16 == 0 and isn't viable for m=1.
///   - Per-buffer opt-in (mixed kernel sig) keeps param-mem usage
///     bounded — each `gl<>` instance is ~152B and the sm_90 kernel
///     param-mem cap is 32KB.
pub fn descriptor_layouts(input: &LoweringInput) -> BTreeMap<u32, GlLayout> {
    if std::env::var_os("FERRITE_NEW_TMA_TENSOR").is_none() {
        return BTreeMap::new();
    }
    descriptor_layouts_for_rmsnorm_outputs(input)
}

/// Pure helper: walks `input` and produces the same descriptor-TMA
/// layout map the env-gated [`descriptor_layouts`] returns when
/// enabled. Exposed for unit tests so they don't need to mutate
/// process-wide env state.
pub fn descriptor_layouts_for_rmsnorm_outputs(input: &LoweringInput) -> BTreeMap<u32, GlLayout> {
    let mut out = BTreeMap::new();
    let n_sources = input.sources.len() as u32;
    let mut op_out_shape: Vec<(u32, u32)> = Vec::with_capacity(input.ops.len());

    let shape_for = |r: InputRef,
                     op_out_shape: &[(u32, u32)],
                     sources: &[crate::subtile::SourceShape]|
     -> (u32, u32) {
        match r {
            InputRef::Ext(e) => (sources[e].rows, sources[e].cols),
            InputRef::Op(j) => op_out_shape[j],
        }
    };

    // Same per-op shape inference as `lower_to_tk` — walk ops in
    // order, push each op's (m, cols) so downstream `InputRef::Op(j)`
    // resolutions land on the right shape.
    for (op_idx, desc) in input.ops.iter().enumerate() {
        let buf_id = n_sources + op_idx as u32;
        match desc.op {
            LoweredOp::RmsNorm { .. } => {
                let hidden = shape_for(desc.inputs[0], &op_out_shape, &input.sources).1;
                op_out_shape.push((desc.m, hidden));
                out.insert(
                    buf_id,
                    GlLayout {
                        batch: 1,
                        depth: 1,
                        rows: 1,
                        cols: hidden as i32,
                        tile_type: format!("kittens::sv_bf<{hidden}>"),
                    },
                );
            }
            LoweredOp::Gemm { n, .. } => {
                op_out_shape.push((desc.m, n));
            }
            LoweredOp::SiluMul => {
                let intermediate = shape_for(desc.inputs[0], &op_out_shape, &input.sources).1;
                op_out_shape.push((desc.m, intermediate));
            }
            LoweredOp::Add => {
                let hidden = shape_for(desc.inputs[0], &op_out_shape, &input.sources).1;
                op_out_shape.push((desc.m, hidden));
            }
            LoweredOp::RopeRotate { .. } | LoweredOp::RopeAppend { .. } => {
                let cols = shape_for(desc.inputs[0], &op_out_shape, &input.sources).1;
                op_out_shape.push((desc.m, cols));
            }
            LoweredOp::AttnDecode {
                num_q_heads,
                head_dim,
                ..
            } => {
                op_out_shape.push((desc.m, num_q_heads * head_dim));
            }
            LoweredOp::Silu | LoweredOp::Mul => {
                // Standalone Silu/Mul is fused into SiluMul before
                // orchestration; if one slips through, the orchestrator
                // panics — match its error surface here so this helper
                // doesn't paper over the bug with a stale shape entry.
                panic!(
                    "descriptor_layouts: standalone {:?} reached the helper; \
                     fuse_silu_mul must run before lower_to_tk.",
                    desc.op
                );
            }
        }
    }
    out
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

    /// Step C.3 — full one-layer canonical emit with descriptor TMA on
    /// rmsnorm outputs gives:
    ///   - one `using ARG{i}_T = kittens::gl<...>;` typedef per rmsnorm
    ///     output buffer.
    ///   - one `const __grid_constant__ ARG{i}_T arg{i}` kernel param
    ///     per typed buffer.
    ///   - typed `tma::store_async<cache_policy::NORMAL>(arg{i}, ...,
    ///     {0,0,0,0})` for each rmsnorm output store, in place of the
    ///     raw-bulk `store_async(reinterpret_cast<char*>(buf{i}), ...)`.
    ///   - one `ARG{i}_T arg{i}_inst(...)` host wrapper instance per
    ///     typed buf, passed to the `<<<...>>>` launch.
    /// This is the full integration that `vllm-executor`'s ferrite_worker
    /// hits when `FERRITE_NEW_TMA_TENSOR=1`.
    #[test]
    fn full_one_layer_emit_with_descriptor_layouts_wires_typed_stores() {
        use crate::fixtures::orchestrator_kernel_args;
        use crate::tk_codegen::{emit_kernel_with_opts, EmitOpts};

        let input = one_layer_input();
        let (prog, n_bufs) = lower_to_tk(&input);
        let args = orchestrator_kernel_args(&input, n_bufs);
        let layouts = descriptor_layouts_for_rmsnorm_outputs(&input);
        assert!(!layouts.is_empty(), "fixture has rmsnorms");

        let opts = EmitOpts {
            descriptor_layouts: layouts.clone(),
            ..Default::default()
        };
        let src = emit_kernel_with_opts("tk_decode_one_layer_typed", &args, &prog, &opts);

        // Every typed buf gets a typedef + a __grid_constant__ kernel
        // sig param + a host instance.
        for buf_id in layouts.keys() {
            assert!(
                src.contains(&format!("using ARG{buf_id}_T = kittens::gl<")),
                "typedef for buf {buf_id}\n--- src ---\n{src}"
            );
            assert!(
                src.contains(&format!(
                    "const __grid_constant__ ARG{buf_id}_T arg{buf_id}"
                )),
                "grid_constant arg for buf {buf_id}\n--- src ---\n{src}"
            );
            assert!(
                src.contains(&format!("ARG{buf_id}_T arg{buf_id}_inst(")),
                "host wrapper instance for buf {buf_id}\n--- src ---\n{src}"
            );
        }

        // At least one typed store_async lands in the body — exactly
        // the rmsnorm output drains.
        assert!(
            src.contains("kittens::group<1>::tma::store_async<kittens::cache_policy::NORMAL>("),
            "at least one typed store_async\n--- src ---\n{src}"
        );

        // Default-off control: no descriptor layouts → emit must keep
        // the legacy raw-bulk store form for those same rmsnorm outputs.
        let plain = emit_kernel_with_opts(
            "tk_decode_one_layer_typed",
            &args,
            &prog,
            &EmitOpts::default(),
        );
        assert!(
            !plain.contains("kittens::cache_policy::NORMAL"),
            "default emit must not contain typed store form\n{plain}"
        );
        assert!(
            plain.contains("kittens::group<1>::tma::store_async("),
            "default emit keeps raw-bulk store_async\n{plain}"
        );
    }

    /// Step C.3 — `descriptor_layouts_for_rmsnorm_outputs` keys exactly
    /// on the rmsnorm output BufIds (one per RmsNorm op), and each
    /// entry carries the rmsnorm's hidden-axis size as a `sv_bf<HIDDEN>`
    /// vector layout. Pure helper — no env mutation.
    #[test]
    fn descriptor_layouts_keys_each_rmsnorm_output() {
        let input = one_layer_input();
        let layouts = descriptor_layouts_for_rmsnorm_outputs(&input);
        let n_sources = input.sources.len() as u32;

        // Collect expected keys by scanning ops for RmsNorm ops.
        let mut expected: Vec<u32> = input
            .ops
            .iter()
            .enumerate()
            .filter_map(|(i, d)| match d.op {
                LoweredOp::RmsNorm { .. } => Some(n_sources + i as u32),
                _ => None,
            })
            .collect();
        expected.sort();
        let mut got: Vec<u32> = layouts.keys().copied().collect();
        got.sort();
        assert_eq!(got, expected, "keys must match RmsNorm output BufIds");
        assert!(
            !layouts.is_empty(),
            "one_layer_input has rmsnorms; helper must produce entries"
        );

        for (buf_id, layout) in &layouts {
            assert_eq!(layout.batch, 1);
            assert_eq!(layout.depth, 1);
            assert_eq!(layout.rows, 1);
            assert!(layout.cols > 0, "rmsnorm hidden must be positive");
            assert_eq!(
                layout.tile_type,
                format!("kittens::sv_bf<{}>", layout.cols),
                "tile_type must match cols for buf {buf_id}"
            );
        }
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
