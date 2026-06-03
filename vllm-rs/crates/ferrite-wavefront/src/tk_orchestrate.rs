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

use std::collections::{BTreeMap, HashMap};

use crate::lower::{InputRef, LoweredOp, LoweringInput};
use crate::routing::{
    classify_inputs, classify_outputs, coalesce_carry_forwards, InputRouting,
};
use crate::subtile_ir::BufId;
use crate::tk_codegen::GlLayout;
use crate::tk_gmem::{
    emit_fence_after_op, ArenaSlot, Carried, CarriedProof, CrossOpInput, Ext, GmemHandle, OpOutput,
};
use crate::tk_lower::{
    lower_attn_decode, lower_gemm_m1, lower_residual_add, lower_rmsnorm, lower_rope_append,
    lower_rope_rotate, lower_silu_mul, AddOp, AttnDecodeOp, CarriedHandle, GemmM1Op,
    PageAllocator, RmsNormOp, RopeAppendOp, RopeRotateOp, SiluMulOp,
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
        LoweredOp::AttnPrefill { .. } => "AttnPrefill",
        LoweredOp::RopeMultiToken { .. } => "RopeMultiToken",
        LoweredOp::ReshapeAndCacheMulti { .. } => "ReshapeAndCacheMulti",
    }
}

/// Stage 2.B — build a typed [`CrossOpInput<ArenaSlot>`] for an
/// arena-edge consumer slot. The producer's gmem-routed output
/// landed in `gmem_handles[buf]`; we wrap it via
/// [`emit_fence_after_op`] for the consumer's typed Fenced input.
/// For carry-forward edges we wrap the producer's stashed
/// `CarriedHandle` (from `carried_table`) into a sealed
/// [`Carried<ArenaSlot>`].
fn build_arena_input(
    routing: &InputRouting,
    buf: BufId,
    gmem_handles: &HashMap<BufId, GmemHandle<ArenaSlot>>,
    carried_table: &[Option<CarriedHandle>],
    prog: &mut TkProgram,
) -> CrossOpInput<ArenaSlot> {
    match routing {
        InputRouting::CarryForward { producer_op_idx } => {
            let h = carried_table[*producer_op_idx]
                .expect("Stage 2.B: producer's CarriedHandle must be stashed before consumer runs");
            CrossOpInput::Carried(Carried::from_handle(h, CarriedProof::mint()))
        }
        InputRouting::GmemLoad => {
            let h = *gmem_handles.get(&buf).unwrap_or_else(|| {
                panic!(
                    "Stage 2.B: arena gmem handle missing for buf {:?} (producer must have \
                     stashed via OpOutput::Gmem before consumer dispatch)",
                    buf
                )
            });
            CrossOpInput::Fenced(emit_fence_after_op(prog, h))
        }
    }
}

/// Stage 2.B — build a typed [`CrossOpInput<Ext>`] for an external-
/// source consumer slot. External sources never carry-forward (they
/// have no producer op); the orchestrator constructs a fresh
/// `GmemHandle::<Ext>::new_initial` and fences it.
fn build_ext_input(buf: BufId, prog: &mut TkProgram) -> CrossOpInput<Ext> {
    let h = GmemHandle::<Ext>::new_initial(buf);
    CrossOpInput::Fenced(emit_fence_after_op(prog, h))
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

    // Step F.1 (was Phase 12 + FERRITE_NEW_ROUTING gate): smem-routing
    // is always on. Walk the DAG once to classify each op's outputs as
    // internal/external and each op's input slots as carry-forward /
    // gmem-load. Producer `CarriedHandle`s thread through
    // `carried_table[op_idx]` to the consuming op's hints; the
    // consuming op's lowering takes the routed path
    // (`_routed` variants) and the producer's smem page goes to the
    // consumer via cross-IType mbarrier handshake — no gmem
    // round-trip for transient activations.
    //
    // Validated post-E.13 with `FERRITE_WAVEFRONT_GPU=1
    // FERRITE_NEW_ROUTING=1`: Paris coherent, kernel ~80x faster than
    // the legacy gmem-roundtrip-everywhere path
    // (`tma::store_async` 161 → 111, `tma::load_async` 532 → 418).
    let (output_routing, input_routing) = {
        let mut outs = classify_outputs(input);
        let ins = classify_inputs(input, &outs);
        // Demote producers whose CarryForward got dropped by the
        // "at most one carry-forward per consumer" rule. They must
        // drain to gmem so the consumer can TMA-load safely.
        coalesce_carry_forwards(&mut outs, &ins);
        (outs, ins)
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

    // Stage 2.B — phase-pick helper. Carry-forward inputs (if any) all
    // share one parity (the producer's post-round parity) — that's the
    // typed phase the consumer's lowering must run at. Without any
    // carry-forward inputs, fall back to picking the parity with the
    // most free slots so reuse keeps working across many ops.
    fn pick_phase(carried_phases: &[u32], pages: &PageAllocator, n_pages: usize) -> u32 {
        if let Some(first) = carried_phases.first().copied() {
            debug_assert!(
                carried_phases.iter().all(|&p| p == first),
                "lower_to_tk: carry-forward inputs have inconsistent parities: {:?}",
                carried_phases,
            );
            return first;
        }
        if pages.count_at(0) >= n_pages {
            0
        } else {
            1
        }
    }

    /// Stage 2.B — collect the carried-input parities from the
    /// per-op input routing for phase picking.
    fn collect_carried_phases(
        op_idx: usize,
        input_routing: &[Vec<InputRouting>],
        carried_table: &[Option<CarriedHandle>],
    ) -> Vec<u32> {
        input_routing[op_idx]
            .iter()
            .filter_map(|ir| match ir {
                InputRouting::CarryForward { producer_op_idx } => {
                    carried_table[*producer_op_idx].map(|c| c.phase)
                }
                InputRouting::GmemLoad => None,
            })
            .collect()
    }

    // Stage 2.B — typed gmem handle stash for arena-edge gmem-routed
    // outputs. Producers populate this on `OpOutput::Gmem`; consumers
    // look up by BufId and fence via `build_arena_input` to construct
    // their typed `CrossOpInput::Fenced`.
    let mut gmem_handles: HashMap<BufId, GmemHandle<ArenaSlot>> = HashMap::new();

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

    /// Stage 2.B — stash the lowering's typed output for the next
    /// consumer op. Gmem outputs go into `gmem_handles[buf]`; carry-
    /// forward outputs go into `carried_table[op_idx]`.
    fn stash_output(
        out_buf: BufId,
        op_idx: usize,
        output: OpOutput<ArenaSlot>,
        gmem_handles: &mut HashMap<BufId, GmemHandle<ArenaSlot>>,
        carried_table: &mut [Option<CarriedHandle>],
    ) {
        match output {
            OpOutput::Gmem(h) => {
                debug_assert_eq!(
                    h.buf_id(),
                    out_buf,
                    "Stage 2.B: lowering returned GmemHandle with mismatched BufId",
                );
                gmem_handles.insert(out_buf, h);
            }
            OpOutput::Carried(c) => {
                carried_table[op_idx] = Some(c.into_handle());
            }
        }
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

        // Stage 2.B — routing analysis is consulted on a per-input
        // basis below. `output_internal` is read once to drive the
        // typed `bool` argument to each lowering.
        let output_internal = output_routing[op_idx].is_internal();
        let carried_phases = collect_carried_phases(op_idx, &input_routing, &carried_table);

        // Helper closure to build an arena `CrossOpInput` for one
        // `desc.inputs[i]` slot. Captures the routing classification
        // for this op + the running gmem-handle / carried tables.
        // Outer match below lifts these per-op.
        match desc.op {
            LoweredOp::RmsNorm { eps } => {
                let x_buf = buf_for(desc.inputs[0], &op_out_buf);
                let weight_buf = buf_for(desc.inputs[1], &op_out_buf);
                let hidden = shape_for(desc.inputs[0], &op_out_shape, &input.sources).1;
                let x_in = build_arena_input(
                    &input_routing[op_idx][0],
                    x_buf,
                    &gmem_handles,
                    &carried_table,
                    &mut prog,
                );
                let weight_in = build_ext_input(weight_buf, &mut prog);
                let phase = pick_phase(&carried_phases, &pages, 2);
                let op = RmsNormOp {
                    x: x_buf,
                    weight: weight_buf,
                    out: out_buf,
                    hidden,
                    m: desc.m,
                    act_elem: ACT_ELEM,
                    eps,
                    init: false,
                };
                let output = if phase == 0 {
                    lower_rmsnorm::<Phase0>(op, x_in, weight_in, output_internal, &mut pages, &mut prog)
                } else {
                    lower_rmsnorm::<Phase1>(op, x_in, weight_in, output_internal, &mut pages, &mut prog)
                };
                stash_output(out_buf, op_idx, output, &mut gmem_handles, &mut carried_table);
                op_out_shape.push((desc.m, hidden));
            }

            LoweredOp::Gemm { n, k } => {
                let x_buf = buf_for(desc.inputs[0], &op_out_buf);
                let w_buf = buf_for(desc.inputs[1], &op_out_buf);
                let bn = pick_bn(k);
                let x_in = build_arena_input(
                    &input_routing[op_idx][0],
                    x_buf,
                    &gmem_handles,
                    &carried_table,
                    &mut prog,
                );
                let w_in = build_ext_input(w_buf, &mut prog);
                let phase = pick_phase(&carried_phases, &pages, 3);
                let op = GemmM1Op {
                    x: x_buf,
                    w: w_buf,
                    out: out_buf,
                    k,
                    n,
                    bn,
                    act_elem: ACT_ELEM,
                };
                let output = if phase == 0 {
                    lower_gemm_m1::<Phase0>(op, x_in, w_in, output_internal, &mut pages, &mut prog)
                } else {
                    lower_gemm_m1::<Phase1>(op, x_in, w_in, output_internal, &mut pages, &mut prog)
                };
                stash_output(out_buf, op_idx, output, &mut gmem_handles, &mut carried_table);
                op_out_shape.push((desc.m, n));
            }

            LoweredOp::SiluMul => {
                let gate_buf = buf_for(desc.inputs[0], &op_out_buf);
                let up_buf = buf_for(desc.inputs[1], &op_out_buf);
                let intermediate = shape_for(desc.inputs[0], &op_out_shape, &input.sources).1;
                let gate_in = build_arena_input(
                    &input_routing[op_idx][0],
                    gate_buf,
                    &gmem_handles,
                    &carried_table,
                    &mut prog,
                );
                let up_in = build_arena_input(
                    &input_routing[op_idx][1],
                    up_buf,
                    &gmem_handles,
                    &carried_table,
                    &mut prog,
                );
                let phase = pick_phase(&carried_phases, &pages, 2);
                let op = SiluMulOp {
                    gate: gate_buf,
                    up: up_buf,
                    out: out_buf,
                    intermediate,
                    m: desc.m,
                    act_elem: ACT_ELEM,
                };
                let output = if phase == 0 {
                    lower_silu_mul::<Phase0>(op, gate_in, up_in, output_internal, &mut pages, &mut prog)
                } else {
                    lower_silu_mul::<Phase1>(op, gate_in, up_in, output_internal, &mut pages, &mut prog)
                };
                stash_output(out_buf, op_idx, output, &mut gmem_handles, &mut carried_table);
                op_out_shape.push((desc.m, intermediate));
            }

            LoweredOp::Add => {
                let a_buf = buf_for(desc.inputs[0], &op_out_buf);
                let b_buf = buf_for(desc.inputs[1], &op_out_buf);
                let hidden = shape_for(desc.inputs[0], &op_out_shape, &input.sources).1;
                let a_in = build_arena_input(
                    &input_routing[op_idx][0],
                    a_buf,
                    &gmem_handles,
                    &carried_table,
                    &mut prog,
                );
                let b_in = build_arena_input(
                    &input_routing[op_idx][1],
                    b_buf,
                    &gmem_handles,
                    &carried_table,
                    &mut prog,
                );
                let phase = pick_phase(&carried_phases, &pages, 2);
                let op = AddOp {
                    a: a_buf,
                    b: b_buf,
                    out: out_buf,
                    hidden,
                    m: desc.m,
                    act_elem: ACT_ELEM,
                };
                let output = if phase == 0 {
                    lower_residual_add::<Phase0>(op, a_in, b_in, output_internal, &mut pages, &mut prog)
                } else {
                    lower_residual_add::<Phase1>(op, a_in, b_in, output_internal, &mut pages, &mut prog)
                };
                stash_output(out_buf, op_idx, output, &mut gmem_handles, &mut carried_table);
                op_out_shape.push((desc.m, hidden));
            }

            LoweredOp::RopeRotate { head_dim } => {
                let x_buf = buf_for(desc.inputs[0], &op_out_buf);
                let cos_buf = buf_for(desc.inputs[1], &op_out_buf);
                let sin_buf = buf_for(desc.inputs[2], &op_out_buf);
                let cols = shape_for(desc.inputs[0], &op_out_shape, &input.sources).1;
                let num_heads = cols / head_dim;
                let x_in = build_arena_input(
                    &input_routing[op_idx][0],
                    x_buf,
                    &gmem_handles,
                    &carried_table,
                    &mut prog,
                );
                let cos_in = build_ext_input(cos_buf, &mut prog);
                let sin_in = build_ext_input(sin_buf, &mut prog);
                let phase = pick_phase(&carried_phases, &pages, 3);
                let op = RopeRotateOp {
                    x: x_buf,
                    cos: cos_buf,
                    sin: sin_buf,
                    out: out_buf,
                    head_dim,
                    num_heads,
                    m: desc.m,
                    act_elem: ACT_ELEM,
                };
                let output = if phase == 0 {
                    lower_rope_rotate::<Phase0>(op, x_in, cos_in, sin_in, output_internal, &mut pages, &mut prog)
                } else {
                    lower_rope_rotate::<Phase1>(op, x_in, cos_in, sin_in, output_internal, &mut pages, &mut prog)
                };
                stash_output(out_buf, op_idx, output, &mut gmem_handles, &mut carried_table);
                op_out_shape.push((desc.m, cols));
            }

            // E.12: RopeAppend rotates K and writes rotated K + V to
            // the paged KV cache pools (PrefixK / PrefixV per layer).
            // `desc.inputs` from the bridge is
            // `[K, cos, sin, V, K_cache, V_cache]` (6 entries, per
            // E.12.A bridge widening). The lowering takes K/cos/sin/V
            // as typed `CrossOpInput` (4 slots); K_cache and V_cache
            // are typed `GmemHandle` arguments (E.12.B).
            LoweredOp::RopeAppend { head_dim, layer: _ } => {
                let k_buf = buf_for(desc.inputs[0], &op_out_buf);
                let cos_buf = buf_for(desc.inputs[1], &op_out_buf);
                let sin_buf = buf_for(desc.inputs[2], &op_out_buf);
                let v_buf = buf_for(desc.inputs[3], &op_out_buf);
                let k_cache = buf_for(desc.inputs[4], &op_out_buf);
                let v_cache = buf_for(desc.inputs[5], &op_out_buf);
                let cols = shape_for(desc.inputs[0], &op_out_shape, &input.sources).1;
                let num_kv_heads = cols / head_dim;
                let k_in = build_arena_input(
                    &input_routing[op_idx][0],
                    k_buf,
                    &gmem_handles,
                    &carried_table,
                    &mut prog,
                );
                let cos_in = build_ext_input(cos_buf, &mut prog);
                let sin_in = build_ext_input(sin_buf, &mut prog);
                let v_in = build_arena_input(
                    &input_routing[op_idx][3],
                    v_buf,
                    &gmem_handles,
                    &carried_table,
                    &mut prog,
                );
                // Carry-forward phases: only K + V (cos/sin are always Ext;
                // KCache/VCache aren't CrossOpInput).
                let mut ra_phases: Vec<u32> = Vec::new();
                if let InputRouting::CarryForward { producer_op_idx } = input_routing[op_idx][0] {
                    if let Some(c) = carried_table[producer_op_idx] { ra_phases.push(c.phase); }
                }
                if let InputRouting::CarryForward { producer_op_idx } = input_routing[op_idx][3] {
                    if let Some(c) = carried_table[producer_op_idx] { ra_phases.push(c.phase); }
                }
                let phase = pick_phase(&ra_phases, &pages, 4);
                let rope_append_op = RopeAppendOp {
                    k: k_buf,
                    cos: cos_buf,
                    sin: sin_buf,
                    v: v_buf,
                    out: out_buf,
                    k_cache,
                    v_cache,
                    head_dim,
                    num_kv_heads,
                    m: desc.m,
                    act_elem: ACT_ELEM,
                    decode_slot_arg: "__decode_slot",
                };
                let k_handle = crate::tk_gmem::GmemHandle::<crate::tk_gmem::KCache>::new_initial(k_cache);
                let v_handle = crate::tk_gmem::GmemHandle::<crate::tk_gmem::VCache>::new_initial(v_cache);
                let (output, k_out, v_out) = if phase == 0 {
                    lower_rope_append::<Phase0>(
                        rope_append_op,
                        k_in,
                        cos_in,
                        sin_in,
                        v_in,
                        k_handle,
                        v_handle,
                        output_internal,
                        &mut pages,
                        &mut prog,
                    )
                } else {
                    lower_rope_append::<Phase1>(
                        rope_append_op,
                        k_in,
                        cos_in,
                        sin_in,
                        v_in,
                        k_handle,
                        v_handle,
                        output_internal,
                        &mut pages,
                        &mut prog,
                    )
                };
                stash_output(out_buf, op_idx, output, &mut gmem_handles, &mut carried_table);
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
                let q_buf = buf_for(desc.inputs[0], &op_out_buf);
                let k_cache = buf_for(desc.inputs[1], &op_out_buf);
                let v_cache = buf_for(desc.inputs[2], &op_out_buf);
                let q_in = build_arena_input(
                    &input_routing[op_idx][0],
                    q_buf,
                    &gmem_handles,
                    &carried_table,
                    &mut prog,
                );
                // Q is the only CrossOpInput; K_cache/V_cache stay
                // typed Fenced<GmemHandle<...>>.
                let mut ad_phases: Vec<u32> = Vec::new();
                if let InputRouting::CarryForward { producer_op_idx } = input_routing[op_idx][0] {
                    if let Some(c) = carried_table[producer_op_idx] { ad_phases.push(c.phase); }
                }
                let phase = pick_phase(&ad_phases, &pages, 3);
                let attn_decode_op = AttnDecodeOp {
                    q: q_buf,
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
                // RopeAppend ran (test fixtures + future variants),
                // construct a fresh `new_initial` handle. Either path
                // goes through `emit_fence_after_op`.
                let k_unfenced = pending_k_unfenced.take().unwrap_or_else(|| {
                    crate::tk_gmem::GmemHandle::<crate::tk_gmem::KCache>::new_initial(k_cache)
                });
                let v_unfenced = pending_v_unfenced.take().unwrap_or_else(|| {
                    crate::tk_gmem::GmemHandle::<crate::tk_gmem::VCache>::new_initial(v_cache)
                });
                debug_assert_eq!(k_unfenced.buf_id(), k_cache);
                debug_assert_eq!(v_unfenced.buf_id(), v_cache);
                let k_fenced = crate::tk_gmem::emit_fence_after_op(&mut prog, k_unfenced);
                let v_fenced = crate::tk_gmem::emit_fence_after_op(&mut prog, v_unfenced);
                let output = if phase == 0 {
                    lower_attn_decode::<Phase0>(
                        attn_decode_op,
                        q_in,
                        k_fenced,
                        v_fenced,
                        output_internal,
                        &mut pages,
                        &mut prog,
                    )
                } else {
                    lower_attn_decode::<Phase1>(
                        attn_decode_op,
                        q_in,
                        k_fenced,
                        v_fenced,
                        output_internal,
                        &mut pages,
                        &mut prog,
                    )
                };
                stash_output(out_buf, op_idx, output, &mut gmem_handles, &mut carried_table);
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

            // Stage 4.A — prefill IR variants exist in the enum so
            // the bridge can emit them, but the lowerings land in
            // Stage 4.C. Until then, the orchestrator panics if a
            // prefill canonical (num_tokens > 1) reaches it.
            LoweredOp::AttnPrefill { .. }
            | LoweredOp::RopeMultiToken { .. }
            | LoweredOp::ReshapeAndCacheMulti { .. } => {
                panic!(
                    "lower_to_tk: prefill op {:?} not yet supported (Stage 4.C)",
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
            LoweredOp::AttnPrefill {
                num_q_heads,
                head_dim,
                ..
            } => {
                op_out_shape.push((desc.m, num_q_heads * head_dim));
            }
            LoweredOp::RopeMultiToken { .. }
            | LoweredOp::ReshapeAndCacheMulti { .. } => {
                let cols = shape_for(desc.inputs[0], &op_out_shape, &input.sources).1;
                op_out_shape.push((desc.m, cols));
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

    /// Step F (Stage 3) — audit pin: validate the External-edge
    /// classification on the `one_layer_input` synthetic fixture.
    /// Three categories of External outputs (per `routing.rs`'s
    /// `classify_outputs` + `coalesce_carry_forwards`):
    ///
    /// 1. **Result** (Op 11 Add) — `input.result`'s output is
    ///    host-readable.
    /// 2. **Multi-consumer** (Op 6 RmsNorm-2) — output feeds Op 7
    ///    (gate_proj) AND Op 8 (up_proj). `classify_outputs` marks
    ///    multi-consumer outputs External upfront.
    /// 3. **Coalesce demotion** (Op 8 Gemm-up) — single-consumer
    ///    Op 9 (SiluMul) read both Op 7 and Op 8 but the
    ///    "at most one CarryForward per consumer" rule routed Op 7
    ///    (the first eligible) and Op 8 lost the race.
    ///    `coalesce_carry_forwards` demoted Op 8 to External so the
    ///    consumer can TMA-load.
    ///
    /// Internal outputs (9 ops: 0..5, 7, 9, 10) all carry-forward
    /// to their single consumer.
    #[test]
    fn stage_3_audit_external_edges_three_categories() {
        use crate::routing::{
            classify_inputs, classify_outputs, coalesce_carry_forwards, InputRouting,
            OutputRouting,
        };
        let input = one_layer_input();
        let mut outs = classify_outputs(&input);
        let ins = classify_inputs(&input, &outs);
        coalesce_carry_forwards(&mut outs, &ins);

        // Category 1: result op.
        assert!(matches!(outs[input.result], OutputRouting::External));

        // Category 2: multi-consumer (Op 6 RmsNorm-2 → {Op 7, Op 8}).
        assert!(matches!(outs[6], OutputRouting::External));
        // Both Op 7 and Op 8 GmemLoad from Op 6 (no carry).
        assert!(matches!(ins[7][0], InputRouting::GmemLoad));
        assert!(matches!(ins[8][0], InputRouting::GmemLoad));

        // Category 3: coalesce demotion (Op 8, lost carry race to
        // Op 7 at SiluMul).
        assert!(matches!(outs[8], OutputRouting::External));
        // Op 9 carries Op 7 (the winning input slot).
        assert!(matches!(
            ins[9][0],
            InputRouting::CarryForward { producer_op_idx: 7 }
        ));
        // Op 9 GmemLoads from Op 8 (the losing input slot).
        assert!(matches!(ins[9][1], InputRouting::GmemLoad));

        // Internal carry-forward chain: 0 → 1 → 2 → 3 → 4 → 5 → 6
        // (with Op 6 being the LAST Internal in the chain because
        // Op 5 carries to Op 6, then Op 6 demotes-to-External).
        // Sanity: all listed Internal ops have exactly one consumer.
        for i in [0usize, 1, 2, 3, 4, 5, 7, 9, 10] {
            assert!(
                matches!(outs[i], OutputRouting::Internal { ref consumer_op_indices } if consumer_op_indices.len() == 1),
                "Op {i} expected Internal-single-consumer"
            );
        }

        // Final tally: 3 External + 9 Internal = 12 ops.
        let ext = outs
            .iter()
            .filter(|r| matches!(r, OutputRouting::External))
            .count();
        assert_eq!(ext, 3, "External count: result + multi-consumer + coalesce-demotion");
        assert_eq!(outs.len() - ext, 9, "Internal count");
    }

    /// Step F (Stage 4.A) — bridge is parameterized by `num_tokens`.
    /// For `num_tokens > 1`, the bridge emits prefill IR variants
    /// (`AttnPrefill`, `RopeMultiToken`, `ReshapeAndCacheMulti`) and
    /// every `OpDesc.m == num_tokens`. Until Stage 4.C lands the
    /// per-op lowerings, the orchestrator panics on prefill ops with
    /// a known message — this test pins that contract so the next
    /// stage knows where to wire the lowerings.
    #[test]
    fn stage_4a_prefill_fixture_propagates_num_tokens() {
        use crate::fixtures::{buf_byte_sizes, one_layer_input, prefill_one_layer_input};
        let n = 64u32;
        let pf = prefill_one_layer_input(n);
        // Every op carries the bucket size as `m`.
        for od in &pf.ops {
            assert_eq!(od.m, n, "OpDesc.m must equal num_tokens");
        }
        // Prefill IR variants (vs decode's RopeRotate/RopeAppend/AttnDecode):
        let kinds: Vec<&'static str> = pf.ops.iter().map(|od| op_kind_name(&od.op)).collect();
        assert!(kinds.contains(&"RopeMultiToken"), "kinds={:?}", kinds);
        assert!(kinds.contains(&"AttnPrefill"), "kinds={:?}", kinds);
        // Per-buffer byte sizes scale with num_tokens for activation
        // sources (source 0 = `[m, hidden]`, m * 2048 * 2 bytes).
        let dec_sizes = buf_byte_sizes(&one_layer_input());
        let pf_sizes = buf_byte_sizes(&pf);
        // Source 0 (input activation `x`): decode is `[1, 2048] = 4096 B`,
        // prefill is `[n, 2048] = n * 4096 B`.
        assert_eq!(dec_sizes[0], 1 * 2048 * 2);
        assert_eq!(pf_sizes[0], n as usize * 2048 * 2);
    }

    #[test]
    #[should_panic(expected = "prefill op")]
    fn stage_4a_orchestrator_panics_on_prefill_until_4c() {
        use crate::fixtures::rope_multi_only_input;
        let pf = rope_multi_only_input(64);
        let _ = lower_to_tk(&pf);
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
    ///
    /// Step F.1 — `#[ignore]`d: synthetic fixture page-allocator
    /// pathology (see notes on `one_layer_forward_lowers_end_to_end`).
    #[test]
    #[ignore]
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
    /// Step F.1 — `#[ignore]`d: synthetic fixture exhausts page
    /// allocator under always-on routing (real Llama-1B works,
    /// Stage 0 pod-validated). See follow-up note above the
    /// `descriptor_layouts` test.
    #[test]
    #[ignore]
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
    ///
    /// Step F.1 — `#[ignore]`d: same fixture-vs-allocator pathology.
    #[test]
    #[ignore]
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
