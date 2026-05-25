// SPDX-License-Identifier: Apache-2.0
//! Per-`MegaNode` const-generic render functions.
//!
//! Each `render_<variant><const ...>(/* per-op runtime args */)`
//! returns four [`CuBlock`](super::cu::CuBlock)s — one per warp role
//! (loader / launcher / consumer / storer). The proc-macro emits
//! per-Instruction `render_<variant>::<const-generic-args>(...)`
//! calls into a per-canonical `emit_for_canonical_<canonical>()` fn,
//! one source of truth for the const generics shared with the
//! parallel `b.push_<variant>::<const-generic-args>(...)` tape
//! builder calls.
//!
//! Per [[feedback-end-to-end-compile-time-proofs]] (`MEGA_IR_PLAN.md`
//! §8.0b), every shape const generic the IR encodes propagates
//! through to `tk20::*` and `handles::*` calls here as a Rust const
//! generic. Wrong shapes are Rust type errors at user-build time,
//! not runtime panics or nvcc errors.

#![allow(clippy::too_many_arguments)]

use crate::ir::nodes::{GateUpActivation, LmHeadNormKind};
use crate::ir::substrate::{PageRef, ScratchOffsetRef};

use super::cu::{CuBlock, CuExpr, CuStmt};
use super::handles::{
    Bf16, F32, RtRow, St, gmem_act_ptr_raw, gmem_barrier_slot_ptr, gmem_input_ids, gmem_positions,
    gmem_weight_ptr_raw, gmem_weight_ptr_raw_offset, page_as_byte_ptr, page_as_st_bf,
    page_as_sv_bf, page_consumed_sem, page_done_sem, page_ready_sem, page_row_as_sv_bf,
    scratch_as,
    scratch_as_st_bf,
};
use super::tk20;

/// Four role-body chunks for a single MegaNode (mirror of the prior
/// `roles::RoleBodies` shape).
#[derive(Debug, Default)]
pub struct RoleBodies {
    pub loader: CuBlock,
    pub launcher: CuBlock,
    pub consumer: CuBlock,
    pub storer: CuBlock,
    pub skipped: Option<&'static str>,
}

impl RoleBodies {
    pub fn skipped(variant: &'static str) -> Self {
        Self {
            skipped: Some(variant),
            ..Self::default()
        }
    }
}

const BF16_BYTES: u32 = 2;

#[inline]
fn page(id: u32) -> PageRef {
    PageRef::__new_for_erase(id)
}

#[inline]
fn off(o: u32) -> ScratchOffsetRef {
    ScratchOffsetRef::__new_for_erase(o)
}

// ============================================================
// BarrierSignal / BarrierWait — single-line cross-CTA barriers.
// No shape const generics; edge id is runtime.
// ============================================================

pub fn render_barrier_signal(edge: u32) -> RoleBodies {
    let slot_ptr = gmem_barrier_slot_ptr(edge);
    let mut loader = CuBlock::new();
    loader.push(tk20::ferrite_barrier_signal(&slot_ptr, 1));
    RoleBodies {
        loader,
        ..Default::default()
    }
}

pub fn render_barrier_wait(edge: u32, expected: u32) -> RoleBodies {
    let slot_ptr = gmem_barrier_slot_ptr(edge);
    let mut loader = CuBlock::new();
    loader.push(tk20::ferrite_barrier_wait(&slot_ptr, expected));
    RoleBodies {
        loader,
        ..Default::default()
    }
}

// ============================================================
// SpliceMmEmbeds — passthrough; advance per-page barrier cycle.
// ============================================================

pub fn render_splice_mm_embeds(
    slot_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
) -> RoleBodies {
    let loader_phase = storer_phase;
    let slot = page(slot_id);
    let slot_ready = page_ready_sem(slot);
    let slot_done = page_done_sem(slot);
    let slot_consumed = page_consumed_sem(slot);

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait::<1>(&slot_consumed, loader_phase));
    loader.push(tk20::group_arrive::<1>(&slot_ready));

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait::<1>(&slot_ready, consumer_phase));
    consumer.push(tk20::block_warp_zero(&[tk20::group_arrive::<1>(&slot_done)]));

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&slot_done, storer_phase));
    storer.push(tk20::group_arrive::<1>(&slot_consumed));

    RoleBodies {
        loader,
        consumer,
        storer,
        ..Default::default()
    }
}

// ============================================================
// RmsNorm — in-place per-row normalization.
// ============================================================

pub fn render_rms_norm<
    const HIDDEN_DIM: u32,
    const NUM_TOKENS: u32,
    const NCW: u32,
    const K_PER_WARP: u32,
    const NUM_LAYERS: u32,
>(
    in_page_id: u32,
    weight_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    in_act_slot: u32,
    out_act_slot: u32,
    weight_accessor: u32,
    bar_reduce: u32,
    bar_publish: u32,
    partial_offset: u32,
    eps: f32,
) -> RoleBodies {
    let loader_phase = storer_phase;
    let in_page_r = page(in_page_id);
    let weight_page_r = page(weight_page_id);

    let in_smem = page_as_sv_bf::<HIDDEN_DIM>(in_page_r);
    let weight_smem = page_as_sv_bf::<HIDDEN_DIM>(weight_page_r);
    let in_smem_row = page_row_as_sv_bf::<HIDDEN_DIM>(in_page_r, "__row");
    let in_ready = page_ready_sem(in_page_r);
    let weight_ready = page_ready_sem(weight_page_r);
    let in_done = page_done_sem(in_page_r);
    let in_consumed = page_consumed_sem(in_page_r);
    let weight_consumed = page_consumed_sem(weight_page_r);
    let partial = scratch_as::<F32>(off(partial_offset));
    let in_gmem = gmem_act_ptr_raw(in_act_slot);
    let out_gmem = gmem_act_ptr_raw(out_act_slot);
    let weight_gmem = gmem_weight_ptr_raw(weight_accessor, layer, NUM_LAYERS);

    let act_bytes = HIDDEN_DIM * NUM_TOKENS * BF16_BYTES;
    let weight_bytes = HIDDEN_DIM * BF16_BYTES;

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait::<1>(&in_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&weight_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes::<1>(&in_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw::<1, HIDDEN_DIM>(
        &in_smem, &in_gmem, act_bytes, &in_ready,
    ));
    loader.push(tk20::group_tma_expect_bytes::<1>(&weight_ready, weight_bytes));
    loader.push(tk20::group_tma_load_async_raw::<1, HIDDEN_DIM>(
        &weight_smem,
        &weight_gmem,
        weight_bytes,
        &weight_ready,
    ));

    let launcher = CuBlock::new();

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait::<1>(&in_ready, consumer_phase));
    consumer.push(tk20::group_wait::<1>(&weight_ready, consumer_phase));

    let (decl_act, act_rv) = tk20::decl_rv_fl::<K_PER_WARP>("__rms_act_rv");
    let (decl_sq, sq_rv) = tk20::decl_rv_fl::<K_PER_WARP>("__rms_sq_rv");
    let (decl_w, weight_rv) = tk20::decl_rv_fl::<K_PER_WARP>("__rms_weight_rv");
    consumer.push(decl_act);
    consumer.push(decl_sq);
    consumer.push(decl_w);

    // Weight is shared across rows — load once outside the per-row loop.
    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &weight_rv,
        &weight_smem,
    ));

    let mut per_row = CuBlock::new();
    per_row.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &act_rv, &in_smem_row,
    ));
    per_row.push(tk20::warp_copy_rv::<F32, K_PER_WARP, _>(&sq_rv, &act_rv));
    per_row.push(tk20::warp_mul_rv_rv::<K_PER_WARP, _>(&sq_rv, &sq_rv, &sq_rv));
    let (decl_partial, partial_sum_expr) = tk20::decl_local_f32("__rms_partial_sum", "0.0f");
    per_row.push(decl_partial);
    per_row.push(tk20::warp_sum_to_scalar_f32::<K_PER_WARP>(&partial_sum_expr, &sq_rv));

    let (decl_full, full_sum_expr) = tk20::decl_local_f32("__rms_full_sum", "0.0f");
    per_row.push(decl_full);
    per_row.push(tk20::cross_warp_reduce_sum_f32::<NCW>(
        full_sum_expr.as_str(),
        partial_sum_expr.as_str(),
        &partial,
        bar_reduce,
    )); // BAR_REDUCE pinned to 1 by const generic

    let (decl_scale, scale_expr) = tk20::decl_rms_scale_local::<HIDDEN_DIM>(
        "__rms_scale",
        full_sum_expr.as_str(),
        eps,
    );
    per_row.push(decl_scale);

    per_row.push(tk20::warp_mul_rv_scalar_f32::<K_PER_WARP>(&act_rv, &act_rv, &scale_expr));
    per_row.push(tk20::warp_mul_rv_rv::<K_PER_WARP, _>(&act_rv, &act_rv, &weight_rv));

    per_row.push(tk20::group_store_rv_to_sv_f32_to_bf16::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &in_smem_row, &act_rv,
    ));

    consumer.push(tk20::for_loop_no_unroll(
        &format!("int __row = 0; __row < {NUM_TOKENS}; ++__row"),
        &per_row,
    ));

    consumer.push(tk20::group_sync_named::<NCW>(bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive::<1>(&in_done),
        tk20::group_arrive::<1>(&weight_consumed),
    ]));

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&in_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw::<1, HIDDEN_DIM>(
        &out_gmem, &in_smem, act_bytes,
    ));
    storer.push(tk20::group_tma_store_async_wait::<1>());
    storer.push(tk20::group_arrive::<1>(&in_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}

// ============================================================
// Add — in-place residual fold (residual_smem += delta_smem).
// ============================================================

pub fn render_add<
    const HIDDEN_DIM: u32,
    const NUM_TOKENS: u32,
    const NCW: u32,
    const K_PER_WARP: u32,
>(
    delta_page_id: u32,
    residual_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    delta_act_slot: u32,
    residual_act_slot: u32,
    bar_publish: u32,
) -> RoleBodies {
    let loader_phase = storer_phase;
    let delta_p = page(delta_page_id);
    let residual_p = page(residual_page_id);

    let delta_smem = page_as_sv_bf::<HIDDEN_DIM>(delta_p);
    let residual_smem = page_as_sv_bf::<HIDDEN_DIM>(residual_p);
    let delta_smem_row = page_row_as_sv_bf::<HIDDEN_DIM>(delta_p, "__row");
    let residual_smem_row = page_row_as_sv_bf::<HIDDEN_DIM>(residual_p, "__row");
    let delta_ready = page_ready_sem(delta_p);
    let residual_ready = page_ready_sem(residual_p);
    let residual_done = page_done_sem(residual_p);
    let delta_consumed = page_consumed_sem(delta_p);
    let residual_consumed = page_consumed_sem(residual_p);
    let delta_gmem = gmem_act_ptr_raw(delta_act_slot);
    let residual_gmem = gmem_act_ptr_raw(residual_act_slot);

    let act_bytes = HIDDEN_DIM * NUM_TOKENS * BF16_BYTES;

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait::<1>(&delta_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&residual_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes::<1>(&delta_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw::<1, HIDDEN_DIM>(
        &delta_smem, &delta_gmem, act_bytes, &delta_ready,
    ));
    loader.push(tk20::group_tma_expect_bytes::<1>(&residual_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw::<1, HIDDEN_DIM>(
        &residual_smem, &residual_gmem, act_bytes, &residual_ready,
    ));

    let launcher = CuBlock::new();

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait::<1>(&delta_ready, consumer_phase));
    consumer.push(tk20::group_wait::<1>(&residual_ready, consumer_phase));

    let (decl_delta, delta_rv) = tk20::decl_rv_fl::<K_PER_WARP>("__add_delta_rv");
    let (decl_res, res_rv) = tk20::decl_rv_fl::<K_PER_WARP>("__add_res_rv");
    consumer.push(decl_delta);
    consumer.push(decl_res);

    let mut per_row = CuBlock::new();
    per_row.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &delta_rv, &delta_smem_row,
    ));
    per_row.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &res_rv, &residual_smem_row,
    ));
    per_row.push(tk20::warp_add_rv_rv::<K_PER_WARP, _>(&res_rv, &res_rv, &delta_rv));
    per_row.push(tk20::group_store_rv_to_sv_f32_to_bf16::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &residual_smem_row, &res_rv,
    ));

    consumer.push(tk20::for_loop_no_unroll(
        &format!("int __row = 0; __row < {NUM_TOKENS}; ++__row"),
        &per_row,
    ));

    consumer.push(tk20::group_sync_named::<NCW>(bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive::<1>(&residual_done),
        tk20::group_arrive::<1>(&delta_consumed),
    ]));

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&residual_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw::<1, HIDDEN_DIM>(
        &residual_gmem, &residual_smem, act_bytes,
    ));
    storer.push(tk20::group_tma_store_async_wait::<1>());
    storer.push(tk20::group_arrive::<1>(&residual_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}

// ============================================================
// ScalarMul — in-place per-row scale (out = in * scale).
// ============================================================

pub fn render_scalar_mul<
    const HIDDEN_DIM: u32,
    const NUM_TOKENS: u32,
    const NCW: u32,
    const K_PER_WARP: u32,
>(
    in_page_id: u32,
    out_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    in_act_slot: u32,
    out_act_slot: u32,
    bar_publish: u32,
    scale: f32,
) -> RoleBodies {
    let loader_phase = storer_phase;
    let in_p = page(in_page_id);
    let out_p = page(out_page_id);
    let in_place = in_page_id == out_page_id;

    let in_smem = page_as_sv_bf::<HIDDEN_DIM>(in_p);
    let out_smem = page_as_sv_bf::<HIDDEN_DIM>(out_p);
    let in_smem_row = page_row_as_sv_bf::<HIDDEN_DIM>(in_p, "__row");
    let out_smem_row = page_row_as_sv_bf::<HIDDEN_DIM>(out_p, "__row");
    let in_ready = page_ready_sem(in_p);
    let out_done = page_done_sem(out_p);
    let in_consumed = page_consumed_sem(in_p);
    let out_consumed = page_consumed_sem(out_p);
    let in_gmem = gmem_act_ptr_raw(in_act_slot);
    let out_gmem = gmem_act_ptr_raw(out_act_slot);

    let act_bytes = HIDDEN_DIM * NUM_TOKENS * BF16_BYTES;

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait::<1>(&in_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes::<1>(&in_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw::<1, HIDDEN_DIM>(
        &in_smem, &in_gmem, act_bytes, &in_ready,
    ));

    let launcher = CuBlock::new();

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait::<1>(&in_ready, consumer_phase));
    let (decl_act, act_rv) = tk20::decl_rv_fl::<K_PER_WARP>("__smul_rv");
    consumer.push(decl_act);

    let scale_lit = CuExpr::new(format!("{:e}f", scale));
    let mut per_row = CuBlock::new();
    per_row.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &act_rv, &in_smem_row,
    ));
    per_row.push(tk20::warp_mul_rv_scalar_f32::<K_PER_WARP>(&act_rv, &act_rv, &scale_lit));
    per_row.push(tk20::group_store_rv_to_sv_f32_to_bf16::<NCW, K_PER_WARP, HIDDEN_DIM>(
        if in_place { &in_smem_row } else { &out_smem_row },
        &act_rv,
    ));
    consumer.push(tk20::for_loop_no_unroll(
        &format!("int __row = 0; __row < {NUM_TOKENS}; ++__row"),
        &per_row,
    ));

    consumer.push(tk20::group_sync_named::<NCW>(bar_publish));
    let mut publish_stmts = vec![tk20::group_arrive::<1>(&out_done)];
    if !in_place {
        publish_stmts.push(tk20::group_arrive::<1>(&in_consumed));
    }
    consumer.push(tk20::block_warp_zero(&publish_stmts));

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&out_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw::<1, HIDDEN_DIM>(
        &out_gmem,
        if in_place { &in_smem } else { &out_smem },
        act_bytes,
    ));
    storer.push(tk20::group_tma_store_async_wait::<1>());
    storer.push(tk20::group_arrive::<1>(&out_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}

// ============================================================
// TanhSoftCap — out = tanhf(in / cap) * cap, per-lane.
// ============================================================

pub fn render_tanh_soft_cap<
    const HIDDEN_DIM: u32,
    const NUM_TOKENS: u32,
    const NCW: u32,
    const K_PER_WARP: u32,
>(
    in_page_id: u32,
    out_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    in_act_slot: u32,
    out_act_slot: u32,
    bar_publish: u32,
    cap: f32,
) -> RoleBodies {
    let loader_phase = storer_phase;
    let in_p = page(in_page_id);
    let out_p = page(out_page_id);
    let in_place = in_page_id == out_page_id;

    let in_smem = page_as_sv_bf::<HIDDEN_DIM>(in_p);
    let out_smem = page_as_sv_bf::<HIDDEN_DIM>(out_p);
    let in_smem_row = page_row_as_sv_bf::<HIDDEN_DIM>(in_p, "__row");
    let out_smem_row = page_row_as_sv_bf::<HIDDEN_DIM>(out_p, "__row");
    let in_ready = page_ready_sem(in_p);
    let out_done = page_done_sem(out_p);
    let in_consumed = page_consumed_sem(in_p);
    let out_consumed = page_consumed_sem(out_p);
    let in_gmem = gmem_act_ptr_raw(in_act_slot);
    let out_gmem = gmem_act_ptr_raw(out_act_slot);

    let act_bytes = HIDDEN_DIM * NUM_TOKENS * BF16_BYTES;

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait::<1>(&in_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes::<1>(&in_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw::<1, HIDDEN_DIM>(
        &in_smem, &in_gmem, act_bytes, &in_ready,
    ));

    let launcher = CuBlock::new();

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait::<1>(&in_ready, consumer_phase));
    let (decl_act, act_rv) = tk20::decl_rv_fl::<K_PER_WARP>("__tanh_rv");
    consumer.push(decl_act);

    let lambda_body = format!("tanhf(x * (1.0f / {cap:e}f)) * {cap:e}f");
    let mut per_row = CuBlock::new();
    per_row.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &act_rv, &in_smem_row,
    ));
    per_row.push(tk20::warp_apply_f32_lambda::<K_PER_WARP>(&act_rv, &act_rv, &lambda_body));
    per_row.push(tk20::group_store_rv_to_sv_f32_to_bf16::<NCW, K_PER_WARP, HIDDEN_DIM>(
        if in_place { &in_smem_row } else { &out_smem_row },
        &act_rv,
    ));
    consumer.push(tk20::for_loop_no_unroll(
        &format!("int __row = 0; __row < {NUM_TOKENS}; ++__row"),
        &per_row,
    ));

    consumer.push(tk20::group_sync_named::<NCW>(bar_publish));
    let mut publish_stmts = vec![tk20::group_arrive::<1>(&out_done)];
    if !in_place {
        publish_stmts.push(tk20::group_arrive::<1>(&in_consumed));
    }
    consumer.push(tk20::block_warp_zero(&publish_stmts));

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&out_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw::<1, HIDDEN_DIM>(
        &out_gmem,
        if in_place { &in_smem } else { &out_smem },
        act_bytes,
    ));
    storer.push(tk20::group_tma_store_async_wait::<1>());
    storer.push(tk20::group_arrive::<1>(&out_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}

// ============================================================
// FusedAddRmsNorm — residual += delta, then RmsNorm in-place.
// ============================================================

pub fn render_fused_add_rms_norm<
    const HIDDEN_DIM: u32,
    const NUM_TOKENS: u32,
    const NCW: u32,
    const K_PER_WARP: u32,
    const NUM_LAYERS: u32,
>(
    delta_page_id: u32,
    residual_page_id: u32,
    weight_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    delta_act_slot: u32,
    residual_act_slot: u32,
    weight_accessor: u32,
    bar_reduce: u32,
    bar_publish: u32,
    partial_offset: u32,
    eps: f32,
) -> RoleBodies {
    let loader_phase = storer_phase;
    let delta_p = page(delta_page_id);
    let residual_p = page(residual_page_id);
    let weight_p = page(weight_page_id);

    let delta_smem = page_as_sv_bf::<HIDDEN_DIM>(delta_p);
    let residual_smem = page_as_sv_bf::<HIDDEN_DIM>(residual_p);
    let weight_smem = page_as_sv_bf::<HIDDEN_DIM>(weight_p);
    let delta_smem_row = page_row_as_sv_bf::<HIDDEN_DIM>(delta_p, "__row");
    let residual_smem_row = page_row_as_sv_bf::<HIDDEN_DIM>(residual_p, "__row");
    let delta_ready = page_ready_sem(delta_p);
    let residual_ready = page_ready_sem(residual_p);
    let weight_ready = page_ready_sem(weight_p);
    let residual_done = page_done_sem(residual_p);
    let delta_consumed = page_consumed_sem(delta_p);
    let residual_consumed = page_consumed_sem(residual_p);
    let weight_consumed = page_consumed_sem(weight_p);
    let partial = scratch_as::<F32>(off(partial_offset));
    let delta_gmem = gmem_act_ptr_raw(delta_act_slot);
    let residual_gmem = gmem_act_ptr_raw(residual_act_slot);
    let weight_gmem = gmem_weight_ptr_raw(weight_accessor, layer, NUM_LAYERS);

    let act_bytes = HIDDEN_DIM * NUM_TOKENS * BF16_BYTES;
    let weight_bytes = HIDDEN_DIM * BF16_BYTES;

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait::<1>(&delta_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&residual_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&weight_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes::<1>(&delta_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw::<1, HIDDEN_DIM>(
        &delta_smem, &delta_gmem, act_bytes, &delta_ready,
    ));
    loader.push(tk20::group_tma_expect_bytes::<1>(&residual_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw::<1, HIDDEN_DIM>(
        &residual_smem, &residual_gmem, act_bytes, &residual_ready,
    ));
    loader.push(tk20::group_tma_expect_bytes::<1>(&weight_ready, weight_bytes));
    loader.push(tk20::group_tma_load_async_raw::<1, HIDDEN_DIM>(
        &weight_smem, &weight_gmem, weight_bytes, &weight_ready,
    ));

    let launcher = CuBlock::new();

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait::<1>(&delta_ready, consumer_phase));
    consumer.push(tk20::group_wait::<1>(&residual_ready, consumer_phase));
    consumer.push(tk20::group_wait::<1>(&weight_ready, consumer_phase));

    let (decl_delta, delta_rv) = tk20::decl_rv_fl::<K_PER_WARP>("__farn_delta_rv");
    let (decl_res, res_rv) = tk20::decl_rv_fl::<K_PER_WARP>("__farn_res_rv");
    let (decl_sq, sq_rv) = tk20::decl_rv_fl::<K_PER_WARP>("__farn_sq_rv");
    let (decl_w, weight_rv) = tk20::decl_rv_fl::<K_PER_WARP>("__farn_weight_rv");
    consumer.push(decl_delta);
    consumer.push(decl_res);
    consumer.push(decl_sq);
    consumer.push(decl_w);

    // Weight is shared across rows — load once outside the per-row loop.
    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &weight_rv, &weight_smem,
    ));

    let mut per_row = CuBlock::new();
    per_row.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &delta_rv, &delta_smem_row,
    ));
    per_row.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &res_rv, &residual_smem_row,
    ));
    per_row.push(tk20::warp_add_rv_rv::<K_PER_WARP, _>(&res_rv, &res_rv, &delta_rv));

    per_row.push(tk20::warp_copy_rv::<F32, K_PER_WARP, _>(&sq_rv, &res_rv));
    per_row.push(tk20::warp_mul_rv_rv::<K_PER_WARP, _>(&sq_rv, &sq_rv, &sq_rv));
    let (decl_partial, partial_sum_expr) = tk20::decl_local_f32("__farn_partial_sum", "0.0f");
    per_row.push(decl_partial);
    per_row.push(tk20::warp_sum_to_scalar_f32::<K_PER_WARP>(&partial_sum_expr, &sq_rv));

    let (decl_full, full_sum_expr) = tk20::decl_local_f32("__farn_full_sum", "0.0f");
    per_row.push(decl_full);
    per_row.push(tk20::cross_warp_reduce_sum_f32::<NCW>(
        full_sum_expr.as_str(),
        partial_sum_expr.as_str(),
        &partial,
        bar_reduce,
    ));
    let (decl_scale, scale_expr) = tk20::decl_rms_scale_local::<HIDDEN_DIM>(
        "__farn_scale",
        full_sum_expr.as_str(),
        eps,
    );
    per_row.push(decl_scale);
    per_row.push(tk20::warp_mul_rv_scalar_f32::<K_PER_WARP>(&res_rv, &res_rv, &scale_expr));
    per_row.push(tk20::warp_mul_rv_rv::<K_PER_WARP, _>(&res_rv, &res_rv, &weight_rv));

    per_row.push(tk20::group_store_rv_to_sv_f32_to_bf16::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &residual_smem_row, &res_rv,
    ));

    consumer.push(tk20::for_loop_no_unroll(
        &format!("int __row = 0; __row < {NUM_TOKENS}; ++__row"),
        &per_row,
    ));

    consumer.push(tk20::group_sync_named::<NCW>(bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive::<1>(&residual_done),
        tk20::group_arrive::<1>(&delta_consumed),
        tk20::group_arrive::<1>(&weight_consumed),
    ]));

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&residual_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw::<1, HIDDEN_DIM>(
        &residual_gmem, &residual_smem, act_bytes,
    ));
    storer.push(tk20::group_tma_store_async_wait::<1>());
    storer.push(tk20::group_arrive::<1>(&residual_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}

// ============================================================
// ScalarOffsetRmsNorm — out = (act * scale) * (weight + offset).
// ============================================================

pub fn render_scalar_offset_rms_norm<
    const HIDDEN_DIM: u32,
    const NUM_TOKENS: u32,
    const NCW: u32,
    const K_PER_WARP: u32,
    const NUM_LAYERS: u32,
>(
    in_page_id: u32,
    weight_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    in_act_slot: u32,
    out_act_slot: u32,
    weight_accessor: u32,
    bar_reduce: u32,
    bar_publish: u32,
    partial_offset: u32,
    eps: f32,
    offset: f32,
) -> RoleBodies {
    let loader_phase = storer_phase;
    let in_p = page(in_page_id);
    let weight_p = page(weight_page_id);

    let in_smem = page_as_sv_bf::<HIDDEN_DIM>(in_p);
    let weight_smem = page_as_sv_bf::<HIDDEN_DIM>(weight_p);
    let in_smem_row = page_row_as_sv_bf::<HIDDEN_DIM>(in_p, "__row");
    let in_ready = page_ready_sem(in_p);
    let weight_ready = page_ready_sem(weight_p);
    let in_done = page_done_sem(in_p);
    let in_consumed = page_consumed_sem(in_p);
    let weight_consumed = page_consumed_sem(weight_p);
    let partial = scratch_as::<F32>(off(partial_offset));
    let in_gmem = gmem_act_ptr_raw(in_act_slot);
    let out_gmem = gmem_act_ptr_raw(out_act_slot);
    let weight_gmem = gmem_weight_ptr_raw(weight_accessor, layer, NUM_LAYERS);

    let act_bytes = HIDDEN_DIM * NUM_TOKENS * BF16_BYTES;
    let weight_bytes = HIDDEN_DIM * BF16_BYTES;

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait::<1>(&in_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&weight_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes::<1>(&in_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw::<1, HIDDEN_DIM>(
        &in_smem, &in_gmem, act_bytes, &in_ready,
    ));
    loader.push(tk20::group_tma_expect_bytes::<1>(&weight_ready, weight_bytes));
    loader.push(tk20::group_tma_load_async_raw::<1, HIDDEN_DIM>(
        &weight_smem, &weight_gmem, weight_bytes, &weight_ready,
    ));

    let launcher = CuBlock::new();

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait::<1>(&in_ready, consumer_phase));
    consumer.push(tk20::group_wait::<1>(&weight_ready, consumer_phase));

    let (decl_act, act_rv) = tk20::decl_rv_fl::<K_PER_WARP>("__sors_act_rv");
    let (decl_sq, sq_rv) = tk20::decl_rv_fl::<K_PER_WARP>("__sors_sq_rv");
    let (decl_w, weight_rv) = tk20::decl_rv_fl::<K_PER_WARP>("__sors_weight_rv");
    consumer.push(decl_act);
    consumer.push(decl_sq);
    consumer.push(decl_w);

    // Weight + offset is shared across rows — load + bias once outside the per-row loop.
    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &weight_rv, &weight_smem,
    ));
    let offset_lit = CuExpr::new(format!("{:e}f", offset));
    consumer.push(tk20::warp_add_rv_scalar_f32::<K_PER_WARP>(&weight_rv, &weight_rv, &offset_lit));

    let mut per_row = CuBlock::new();
    per_row.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &act_rv, &in_smem_row,
    ));
    per_row.push(tk20::warp_copy_rv::<F32, K_PER_WARP, _>(&sq_rv, &act_rv));
    per_row.push(tk20::warp_mul_rv_rv::<K_PER_WARP, _>(&sq_rv, &sq_rv, &sq_rv));
    let (decl_partial, partial_sum_expr) = tk20::decl_local_f32("__sors_partial_sum", "0.0f");
    per_row.push(decl_partial);
    per_row.push(tk20::warp_sum_to_scalar_f32::<K_PER_WARP>(&partial_sum_expr, &sq_rv));
    let (decl_full, full_sum_expr) = tk20::decl_local_f32("__sors_full_sum", "0.0f");
    per_row.push(decl_full);
    per_row.push(tk20::cross_warp_reduce_sum_f32::<NCW>(
        full_sum_expr.as_str(),
        partial_sum_expr.as_str(),
        &partial,
        bar_reduce,
    ));
    let (decl_scale, scale_expr) = tk20::decl_rms_scale_local::<HIDDEN_DIM>(
        "__sors_scale",
        full_sum_expr.as_str(),
        eps,
    );
    per_row.push(decl_scale);
    per_row.push(tk20::warp_mul_rv_scalar_f32::<K_PER_WARP>(&act_rv, &act_rv, &scale_expr));
    per_row.push(tk20::warp_mul_rv_rv::<K_PER_WARP, _>(&act_rv, &act_rv, &weight_rv));
    per_row.push(tk20::group_store_rv_to_sv_f32_to_bf16::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &in_smem_row, &act_rv,
    ));

    consumer.push(tk20::for_loop_no_unroll(
        &format!("int __row = 0; __row < {NUM_TOKENS}; ++__row"),
        &per_row,
    ));

    consumer.push(tk20::group_sync_named::<NCW>(bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive::<1>(&in_done),
        tk20::group_arrive::<1>(&weight_consumed),
    ]));

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&in_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw::<1, HIDDEN_DIM>(
        &out_gmem, &in_smem, act_bytes,
    ));
    storer.push(tk20::group_tma_store_async_wait::<1>());
    storer.push(tk20::group_arrive::<1>(&in_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}

// ============================================================
// Embed — per-token vocab table gather.
// ============================================================

pub fn render_embed<
    const HIDDEN_DIM: u32,
    const NUM_TOKENS: u32,
    const NUM_LAYERS: u32,
>(
    out_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    out_act_slot: u32,
    weight_accessor: u32,
) -> RoleBodies {
    let loader_phase = storer_phase;
    let layer: u32 = 0;
    let out_p = page(out_page_id);

    let out_byte = page_as_byte_ptr(out_p);
    let out_ready = page_ready_sem(out_p);
    let out_done = page_done_sem(out_p);
    let out_consumed = page_consumed_sem(out_p);
    let embed_table = gmem_weight_ptr_raw(weight_accessor, layer, NUM_LAYERS);
    let input_ids = gmem_input_ids();
    let out_gmem = gmem_act_ptr_raw(out_act_slot);

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait::<1>(&out_consumed, loader_phase));
    loader.push(tk20::embed_per_token_gather::<HIDDEN_DIM, NUM_TOKENS>(
        &out_byte,
        &embed_table,
        &input_ids,
        &out_ready,
    ));

    let launcher = CuBlock::new();

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait::<1>(&out_ready, consumer_phase));
    consumer.push(tk20::block_warp_zero(&[tk20::group_arrive::<1>(&out_done)]));

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&out_done, storer_phase));
    storer.push(tk20::per_token_tma_store::<HIDDEN_DIM, NUM_TOKENS>(
        &out_gmem,
        &out_byte,
    ));
    storer.push(tk20::group_tma_store_async_wait::<1>());
    storer.push(tk20::group_arrive::<1>(&out_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}

// ============================================================
// Gemm — `D = A * B + C` (C zero) under AlongN warp split.
// ============================================================

pub fn render_gemm<
    const M: u32,
    const K: u32,
    const N: u32,
    const TILE_N: u32,
    const NCW: u32,
    const NUM_LAYERS: u32,
    const ITERS: u32,
>(
    in_page_id: u32,
    weight_page_id: u32,
    out_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    in_act_slot: u32,
    out_act_slot: u32,
    weight_accessor: u32,
    bar_publish: u32,
    b_tile_offset: u32,
) -> RoleBodies {
    if ITERS != 1 {
        return RoleBodies::skipped("TkGemm");
    }
    // M%16!=0 → vec-mat decode-shape (small-batch decode, M ∈ {1, 8}).
    // M%16==0 → tile-MMA prefill-shape (M ∈ {64, 512, 4096}).
    if M % 16 != 0 {
        return render_gemm_decode::<M, K, N, TILE_N, NCW, NUM_LAYERS, ITERS>(
            in_page_id,
            weight_page_id,
            out_page_id,
            consumer_phase,
            storer_phase,
            layer,
            in_act_slot,
            out_act_slot,
            weight_accessor,
            bar_publish,
            b_tile_offset,
        );
    }

    // M%16==0 prefill (M ∈ {64, 512, 4096}). Dispatch to a
    // function parameterized over M_BLOCKS = M / 64 (so the inner
    // wgmma A tile is always st_bf<64, K>). The dispatch fans
    // M onto fixed M_BLOCKS const-generic instantiations so that
    // Rust monomorphizes `render_gemm` at decode-shape M ∈ {1, 8}
    // without triggering the M_BLOCKS const-asserts (which fire
    // only when those branches are actually instantiated, and
    // they always are at fixed M_BLOCKS values where the asserts
    // pass).
    // Dispatch on (M, TILE_N) → concrete (M_BLOCKS, M_TOTAL, WGMMA_N, N_MICRO).
    // wgmma::base specializations exist only for cols ∈ [16..256, step 16]
    // (see TK 2.0 base.cuh:28-43), so each wgmma instruction takes WGMMA_N
    // ≤ 256 and we walk N_MICRO of them per warpgroup-N-tile.
    macro_rules! dispatch_m {
        ($mb:expr, $mt:expr) => {
            if TILE_N == 64 {
                render_gemm_prefill_wgmma::<$mb, $mt, K, N, 64, 1, NCW, NUM_LAYERS, ITERS>(
                    in_page_id, weight_page_id, out_page_id,
                    consumer_phase, storer_phase, layer,
                    in_act_slot, out_act_slot, weight_accessor,
                    bar_publish, b_tile_offset,
                )
            } else if TILE_N == 128 {
                render_gemm_prefill_wgmma::<$mb, $mt, K, N, 128, 1, NCW, NUM_LAYERS, ITERS>(
                    in_page_id, weight_page_id, out_page_id,
                    consumer_phase, storer_phase, layer,
                    in_act_slot, out_act_slot, weight_accessor,
                    bar_publish, b_tile_offset,
                )
            } else if TILE_N == 256 {
                render_gemm_prefill_wgmma::<$mb, $mt, K, N, 256, 1, NCW, NUM_LAYERS, ITERS>(
                    in_page_id, weight_page_id, out_page_id,
                    consumer_phase, storer_phase, layer,
                    in_act_slot, out_act_slot, weight_accessor,
                    bar_publish, b_tile_offset,
                )
            } else if TILE_N == 512 {
                render_gemm_prefill_wgmma::<$mb, $mt, K, N, 256, 2, NCW, NUM_LAYERS, ITERS>(
                    in_page_id, weight_page_id, out_page_id,
                    consumer_phase, storer_phase, layer,
                    in_act_slot, out_act_slot, weight_accessor,
                    bar_publish, b_tile_offset,
                )
            } else if TILE_N == 1024 {
                render_gemm_prefill_wgmma::<$mb, $mt, K, N, 256, 4, NCW, NUM_LAYERS, ITERS>(
                    in_page_id, weight_page_id, out_page_id,
                    consumer_phase, storer_phase, layer,
                    in_act_slot, out_act_slot, weight_accessor,
                    bar_publish, b_tile_offset,
                )
            } else if TILE_N == 2048 {
                render_gemm_prefill_wgmma::<$mb, $mt, K, N, 256, 8, NCW, NUM_LAYERS, ITERS>(
                    in_page_id, weight_page_id, out_page_id,
                    consumer_phase, storer_phase, layer,
                    in_act_slot, out_act_slot, weight_accessor,
                    bar_publish, b_tile_offset,
                )
            } else if TILE_N == 4096 {
                render_gemm_prefill_wgmma::<$mb, $mt, K, N, 256, 16, NCW, NUM_LAYERS, ITERS>(
                    in_page_id, weight_page_id, out_page_id,
                    consumer_phase, storer_phase, layer,
                    in_act_slot, out_act_slot, weight_accessor,
                    bar_publish, b_tile_offset,
                )
            } else {
                RoleBodies::skipped("TkGemm")
            }
        };
    }
    if M == 64 {
        dispatch_m!(1, 64)
    } else if M == 512 {
        dispatch_m!(8, 512)
    } else if M == 4096 {
        dispatch_m!(64, 4096)
    } else {
        panic!("render_gemm prefill: unsupported M (must be 64/512/4096); got {M}");
    }
}

// ============================================================
// Gemm (prefill shape — TK 2.0 H100 wgmma SMEM+SMEM).
//
// Parameterized over `M_BLOCKS` = M_TOTAL / 64. The inner wgmma
// always operates on a `st_bf<64, K>` A subtile (TK 2.0's
// `warpgroup::mma_AB` enforces A::rows == 64 because
// `M = A::rows / TILE_ROW_DIM<bf16=16> == 4`). For M > 64 we
// emit an outer m-block loop in CUDA that subtiles the
// activation/output along the row dimension.
//
// Lifted out of `render_gemm` so that Rust's monomorphization
// of `render_gemm` at decode-shape M ∈ {1, 8} does not
// instantiate the wgmma binding (which fails the
// M==64 const-assert in tk20::warpgroup_mma_AB at those Ms).
//
// D accumulator: per-warp `rt_fl<16, TILE_N>` — each warp of
// the 4-warp warpgroup holds 16 of the 64 rows.
//
// N partitioning: NCW consumer warps form NCW/4 warpgroups
// (NCW=8 → 2 warpgroups). Full N = NCW * TILE_N. Each
// warpgroup covers 4 of the NCW N-tiles by looping
// `n_in_wg ∈ 0..4` and computing column block
// `groupid()*4 + n_in_wg`.
//
// Note: at M_BLOCKS > 1 the activation/output pages are
// over-committed (a `st_bf<M_TOTAL, K>` view of a 32 KB page).
// Prefill is not invoked at runtime by the megakernel persistent
// interpreter — these renders exist to nvcc-compile clean as
// part of the substrate proof, with no runtime launch.
// ============================================================

#[allow(clippy::too_many_arguments)]
pub fn render_gemm_prefill_wgmma<
    const M_BLOCKS: u32,
    const M_TOTAL: u32,
    const K: u32,
    const N: u32,
    const WGMMA_N: u32,
    const N_MICRO: u32,
    const NCW: u32,
    const NUM_LAYERS: u32,
    const ITERS: u32,
>(
    in_page_id: u32,
    weight_page_id: u32,
    out_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    in_act_slot: u32,
    out_act_slot: u32,
    weight_accessor: u32,
    bar_publish: u32,
    b_tile_offset: u32,
) -> RoleBodies {
    const { assert!(NCW % 4 == 0, "render_gemm prefill: NCW must be a multiple of 4 (warpgroup size)"); }
    const { assert!(M_BLOCKS >= 1, "render_gemm prefill: M_BLOCKS must be >= 1"); }
    const { assert!(M_TOTAL == M_BLOCKS * 64, "render_gemm prefill: M_TOTAL must equal M_BLOCKS * 64"); }
    const { assert!(WGMMA_N >= 16 && WGMMA_N <= 256 && WGMMA_N % 16 == 0,
        "render_gemm prefill: WGMMA_N must be a multiple of 16 in [16, 256] (TK 2.0 wgmma::base specializations)"); }
    const { assert!(N_MICRO >= 1, "render_gemm prefill: N_MICRO must be >= 1"); }

    const M_TILE: u32 = 64;

    let loader_phase = storer_phase;
    let in_p = page(in_page_id);
    let weight_p = page(weight_page_id);
    let out_p = page(out_page_id);

    let in_smem = page_as_st_bf::<M_TOTAL, K>(in_p);
    let out_smem = page_as_st_bf::<M_TOTAL, N>(out_p);
    let b_tile = scratch_as_st_bf::<K, N>(off(b_tile_offset));

    let in_ready = page_ready_sem(in_p);
    let weight_ready = page_ready_sem(weight_p);
    let out_done = page_done_sem(out_p);
    let in_consumed = page_consumed_sem(in_p);
    let weight_consumed = page_consumed_sem(weight_p);
    let out_consumed = page_consumed_sem(out_p);

    let in_gmem = gmem_act_ptr_raw(in_act_slot);
    let out_gmem = gmem_act_ptr_raw(out_act_slot);
    let weight_gmem = gmem_weight_ptr_raw(weight_accessor, layer, NUM_LAYERS);

    let act_bytes = M_TOTAL * K * BF16_BYTES;
    let weight_bytes = K * N * BF16_BYTES;
    let out_bytes = M_TOTAL * N * BF16_BYTES;

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait::<1>(&in_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&weight_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&out_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes::<1>(&in_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf::<1, M_TOTAL, K>(
        &in_smem, &in_gmem, act_bytes, &in_ready,
    ));
    loader.push(tk20::group_tma_expect_bytes::<1>(&weight_ready, weight_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf::<1, K, N>(
        &b_tile, &weight_gmem, weight_bytes, &weight_ready,
    ));

    let launcher = CuBlock::new();

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait::<1>(&in_ready, consumer_phase));
    consumer.push(tk20::group_wait::<1>(&weight_ready, consumer_phase));

    let n_tiles_per_wg: u32 = 4;
    let groupid = tk20::warpgroup_groupid_expr();

    // A subtile: 64-row slice of in_smem along the M dim, declared
    // once per m-block (shared across all N-microtiles in this block).
    let (decl_a_sub, a_sub_outer) = tk20::decl_st_bf_subtile::<M_TOTAL, K, M_TILE, K>(
        "__gemm_a_sub", &in_smem, "__gemm_m_block", "0",
    );

    // Innermost: per-microtile wgmma over WGMMA_N cols. acc_rt is
    // declared inside the loop body so each iteration gets its own
    // (compiler reuses registers).
    let mut micro_body = CuBlock::new();
    micro_body.push(CuStmt::new(format!(
        "int __gemm_b_col = __gemm_n_idx * {N_MICRO} + __gemm_nm;"
    )));
    let (decl_acc, acc_rt) = tk20::decl_rt_fl_warpgroup::<WGMMA_N>("__gemm_acc");
    let (decl_b_sub, b_sub) = tk20::decl_st_bf_subtile::<K, N, K, WGMMA_N>(
        "__gemm_b_sub", &b_tile, "0", "__gemm_b_col",
    );
    let (decl_out_sub, out_sub) =
        tk20::decl_st_bf_subtile::<M_TOTAL, N, M_TILE, WGMMA_N>(
            "__gemm_out_sub", &out_smem, "__gemm_m_block", "__gemm_b_col",
        );
    micro_body.push(decl_acc);
    micro_body.push(decl_b_sub);
    micro_body.push(decl_out_sub);
    micro_body.push(tk20::warpgroup_zero_rt_fl::<WGMMA_N>(&acc_rt));
    micro_body.push(tk20::warpgroup_mma_AB::<M_TILE, K, WGMMA_N>(&acc_rt, &a_sub_outer, &b_sub));
    micro_body.push(tk20::warpgroup_mma_async_wait());
    micro_body.push(tk20::warpgroup_store_st_bf_from_rt_fl::<M_TILE, WGMMA_N>(&out_sub, &acc_rt));

    // Inner N-tile loop body (per m-block). Each warpgroup
    // owns 4 of the NCW N-tiles. A subtile is shared across all
    // microtiles within a warpgroup-N-tile (same A rows).
    let mut n_loop_body = CuBlock::new();
    n_loop_body.push(CuStmt::new(format!(
        "int __gemm_n_idx = {groupid} * {n_tiles_per_wg} + __gemm_n_in_wg;"
    )));
    n_loop_body.push(tk20::for_loop_no_unroll(
        &format!("int __gemm_nm = 0; __gemm_nm < {N_MICRO}; ++__gemm_nm"),
        &micro_body,
    ));

    let mut m_loop_body = CuBlock::new();
    m_loop_body.push(decl_a_sub);
    m_loop_body.push(tk20::for_loop_no_unroll(
        &format!("int __gemm_n_in_wg = 0; __gemm_n_in_wg < {n_tiles_per_wg}; ++__gemm_n_in_wg"),
        &n_loop_body,
    ));

    consumer.push(tk20::for_loop_no_unroll(
        &format!("int __gemm_m_block = 0; __gemm_m_block < {M_BLOCKS}; ++__gemm_m_block"),
        &m_loop_body,
    ));

    consumer.push(tk20::group_sync_named::<NCW>(bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive::<1>(&out_done),
        tk20::group_arrive::<1>(&in_consumed),
        tk20::group_arrive::<1>(&weight_consumed),
    ]));

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&out_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw_st_bf::<1, M_TOTAL, N>(
        &out_gmem, &out_smem, out_bytes,
    ));
    storer.push(tk20::group_tma_store_async_wait::<1>());
    storer.push(tk20::group_arrive::<1>(&out_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}

// ============================================================
// Gemm (decode shape, M%16!=0 — small-batch decode).
//
// For M ∈ {1, 8} the prefill render's `st_bf<M, K>` / `rt_bf<M, K>`
// tiles fail TK 2.0's `static_assert(rows % TILE_ROW_DIM == 0)`.
// Padding M to 16 in scratch doesn't fit (16 * K * 2 = 64 KB at
// K=2048 vs SCRATCH_BYTES=32 KB). The prefill K*N b_tile staging
// also doesn't fit (8 MB at K=N=2048). So we emit a per-thread
// vec-mat compute path: each thread owns `TILE_N / 32` output
// columns and accumulates `M` floats per col, sharing each B-load
// across the M rows. B is read directly from gmem (no scratch
// staging — decode is bandwidth-bound, ~8 MB B per GEMM ≈ 2.7 µs
// at H100's 3 TB/s, and the B-share across M rows still keeps
// the pattern memory-bound at small M).
//
// This is a correctness-first emit (no tensor cores), matching
// the substrate's current sizing. When SCRATCH_BYTES / PAGE_SIZE
// are right-sized for an actual H100 launch (228 KB shmem
// budget), this can be re-emitted with TK warp::mma over a
// split-K rt<16, K_CHUNK> path.
// ============================================================

#[allow(clippy::too_many_arguments)]
pub fn render_gemm_decode<
    const M: u32,
    const K: u32,
    const N: u32,
    const TILE_N: u32,
    const NCW: u32,
    const NUM_LAYERS: u32,
    const ITERS: u32,
>(
    in_page_id: u32,
    _weight_page_id: u32,
    out_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    in_act_slot: u32,
    out_act_slot: u32,
    weight_accessor: u32,
    bar_publish: u32,
    _b_tile_offset: u32,
) -> RoleBodies {
    if ITERS != 1 {
        return RoleBodies::skipped("TkGemmDecode");
    }
    if M == 0 {
        return RoleBodies::skipped("TkGemmDecode");
    }
    if NCW == 0 || N % NCW != 0 {
        return RoleBodies::skipped("TkGemmDecode");
    }
    if TILE_N != N / NCW {
        return RoleBodies::skipped("TkGemmDecode");
    }
    if TILE_N % 32 != 0 {
        return RoleBodies::skipped("TkGemmDecode");
    }

    let loader_phase = storer_phase;
    let in_p = page(in_page_id);
    let out_p = page(out_page_id);

    let in_ready = page_ready_sem(in_p);
    let out_done = page_done_sem(out_p);
    let in_consumed = page_consumed_sem(in_p);
    let out_consumed = page_consumed_sem(out_p);

    let in_gmem = gmem_act_ptr_raw(in_act_slot);
    let out_gmem = gmem_act_ptr_raw(out_act_slot);
    let weight_gmem = gmem_weight_ptr_raw(weight_accessor, layer, NUM_LAYERS);

    let act_bytes: u32 = M * K * BF16_BYTES;
    let out_bytes: u32 = M * N * BF16_BYTES;

    // -----------------------------------------------------------
    // LOADER role — TMA-load A from gmem to in_page. Weights are
    // read directly by the consumer (per-thread gmem loads), so
    // no weight TMA stage here.
    // -----------------------------------------------------------
    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait::<1>(&in_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&out_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes::<1>(&in_ready, act_bytes));
    loader.push(CuStmt::new(format!(
        "{{ \
         __nv_bfloat16* __gd_a_dst = reinterpret_cast<__nv_bfloat16*>(ss.pages[{in_id}]); \
         kittens::group<1>::tma::load_async(\
         reinterpret_cast<void*>(__gd_a_dst), \
         reinterpret_cast<void*>({a_gmem}), \
         {act_bytes}, {a_ready}); \
         }}",
        in_id = in_page_id,
        a_gmem = in_gmem.expr(),
        act_bytes = act_bytes,
        a_ready = in_ready.expr(),
    )));

    let launcher = CuBlock::new();

    // -----------------------------------------------------------
    // CONSUMER role — per-thread vec-mat compute.
    // Each warp owns TILE_N output cols (= N / NCW); each thread
    // owns `cols_per_thread = TILE_N / 32` cols. Direct gmem reads
    // of B; reduce per-col sum in fp32; bf16 cast on store.
    // -----------------------------------------------------------
    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait::<1>(&in_ready, consumer_phase));
    consumer.push(CuStmt::new(format!(
        "{{ \
         const __nv_bfloat16* __gd_a_in = reinterpret_cast<const __nv_bfloat16*>(\
             ss.pages[{in_id}]); \
         __nv_bfloat16* __gd_a_out = reinterpret_cast<__nv_bfloat16*>(\
             ss.pages[{out_id}]); \
         const __nv_bfloat16* __gd_b_gmem = {b_gmem}; \
         int __gd_warp_id = static_cast<int>(kittens::warpid()); \
         int __gd_lane = static_cast<int>(kittens::laneid()); \
         int __gd_warp_col_base = __gd_warp_id * {tile_n}; \
         constexpr int __gd_cols_per_thread = {tile_n} / 32; \
         for (int __gd_c = 0; __gd_c < __gd_cols_per_thread; __gd_c++) {{ \
             int __gd_n = __gd_warp_col_base + __gd_c * 32 + __gd_lane; \
             float __gd_acc[{m}]; \
             _Pragma(\"unroll\") \
             for (int __gd_m = 0; __gd_m < {m}; __gd_m++) __gd_acc[__gd_m] = 0.0f; \
             for (int __gd_k = 0; __gd_k < {k}; __gd_k++) {{ \
                 float __gd_b = __bfloat162float(__gd_b_gmem[__gd_k * {n} + __gd_n]); \
                 _Pragma(\"unroll\") \
                 for (int __gd_m = 0; __gd_m < {m}; __gd_m++) {{ \
                     float __gd_a = __bfloat162float(\
                         __gd_a_in[__gd_m * {k} + __gd_k]); \
                     __gd_acc[__gd_m] += __gd_a * __gd_b; \
                 }} \
             }} \
             _Pragma(\"unroll\") \
             for (int __gd_m = 0; __gd_m < {m}; __gd_m++) {{ \
                 __gd_a_out[__gd_m * {n} + __gd_n] = __float2bfloat16(__gd_acc[__gd_m]); \
             }} \
         }} \
         }}",
        in_id = in_page_id,
        out_id = out_page_id,
        b_gmem = weight_gmem.expr(),
        tile_n = TILE_N,
        m = M,
        k = K,
        n = N,
    )));

    consumer.push(tk20::group_sync_named::<NCW>(bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive::<1>(&out_done),
        tk20::group_arrive::<1>(&in_consumed),
    ]));

    // -----------------------------------------------------------
    // STORER role — TMA-store output row to gmem.
    // -----------------------------------------------------------
    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&out_done, storer_phase));
    storer.push(CuStmt::new(format!(
        "{{ \
         __nv_bfloat16* __gd_out_src = reinterpret_cast<__nv_bfloat16*>(ss.pages[{out_id}]); \
         kittens::group<1>::tma::store_async(\
             reinterpret_cast<void*>({out_gmem}), \
             reinterpret_cast<void*>(__gd_out_src), \
             {bytes}); \
         }}",
        out_id = out_page_id,
        out_gmem = out_gmem.expr(),
        bytes = out_bytes,
    )));
    storer.push(tk20::group_tma_store_async_wait::<1>());
    storer.push(tk20::group_arrive::<1>(&out_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}

// ============================================================
// TkFusedGemmAdd (decode shape, M%16!=0 — small-batch decode).
//
// Same per-thread vec-mat structure as `render_gemm_decode`, plus
// a residual fold: TMA-load the residual into the out-page (which
// is reused as the residual page in the prefill render), each
// thread reads its M residual elements from the out-page (one per
// row), folds in the M dot products, bf16-stores back to the
// out-page. B-load is shared across the M rows (one fp32 mul-add
// per (m, k, n)).
// ============================================================

#[allow(clippy::too_many_arguments)]
pub fn render_fused_gemm_add_decode<
    const M: u32,
    const K: u32,
    const N: u32,
    const TILE_N: u32,
    const NCW: u32,
    const NUM_LAYERS: u32,
    const ITERS: u32,
>(
    in_page_id: u32,
    _weight_page_id: u32,
    residual_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    in_act_slot: u32,
    residual_act_slot: u32,
    weight_accessor: u32,
    bar_publish: u32,
    _b_tile_offset: u32,
) -> RoleBodies {
    if ITERS != 1 {
        return RoleBodies::skipped("TkFusedGemmAddDecode");
    }
    if M == 0 {
        return RoleBodies::skipped("TkFusedGemmAddDecode");
    }
    if NCW == 0 || N % NCW != 0 {
        return RoleBodies::skipped("TkFusedGemmAddDecode");
    }
    if TILE_N != N / NCW {
        return RoleBodies::skipped("TkFusedGemmAddDecode");
    }
    if TILE_N % 32 != 0 {
        return RoleBodies::skipped("TkFusedGemmAddDecode");
    }

    let loader_phase = storer_phase;
    let in_p = page(in_page_id);
    let residual_p = page(residual_page_id);

    let in_ready = page_ready_sem(in_p);
    let residual_ready = page_ready_sem(residual_p);
    let residual_done = page_done_sem(residual_p);
    let in_consumed = page_consumed_sem(in_p);
    let residual_consumed = page_consumed_sem(residual_p);

    let in_gmem = gmem_act_ptr_raw(in_act_slot);
    let residual_gmem = gmem_act_ptr_raw(residual_act_slot);
    let weight_gmem = gmem_weight_ptr_raw(weight_accessor, layer, NUM_LAYERS);

    let act_bytes: u32 = M * K * BF16_BYTES;
    let residual_bytes: u32 = M * N * BF16_BYTES;

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait::<1>(&in_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&residual_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes::<1>(&in_ready, act_bytes));
    loader.push(CuStmt::new(format!(
        "{{ \
         __nv_bfloat16* __ga_a_dst = reinterpret_cast<__nv_bfloat16*>(ss.pages[{in_id}]); \
         kittens::group<1>::tma::load_async(\
         reinterpret_cast<void*>(__ga_a_dst), \
         reinterpret_cast<void*>({a_gmem}), \
         {act_bytes}, {a_ready}); \
         }}",
        in_id = in_page_id,
        a_gmem = in_gmem.expr(),
        act_bytes = act_bytes,
        a_ready = in_ready.expr(),
    )));
    loader.push(tk20::group_tma_expect_bytes::<1>(&residual_ready, residual_bytes));
    loader.push(CuStmt::new(format!(
        "{{ \
         __nv_bfloat16* __ga_r_dst = reinterpret_cast<__nv_bfloat16*>(ss.pages[{res_id}]); \
         kittens::group<1>::tma::load_async(\
         reinterpret_cast<void*>(__ga_r_dst), \
         reinterpret_cast<void*>({r_gmem}), \
         {res_bytes}, {r_ready}); \
         }}",
        res_id = residual_page_id,
        r_gmem = residual_gmem.expr(),
        res_bytes = residual_bytes,
        r_ready = residual_ready.expr(),
    )));

    let launcher = CuBlock::new();

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait::<1>(&in_ready, consumer_phase));
    consumer.push(tk20::group_wait::<1>(&residual_ready, consumer_phase));
    consumer.push(CuStmt::new(format!(
        "{{ \
         const __nv_bfloat16* __ga_a_in = reinterpret_cast<const __nv_bfloat16*>(\
             ss.pages[{in_id}]); \
         __nv_bfloat16* __ga_r = reinterpret_cast<__nv_bfloat16*>(\
             ss.pages[{res_id}]); \
         const __nv_bfloat16* __ga_b_gmem = {b_gmem}; \
         int __ga_warp_id = static_cast<int>(kittens::warpid()); \
         int __ga_lane = static_cast<int>(kittens::laneid()); \
         int __ga_warp_col_base = __ga_warp_id * {tile_n}; \
         constexpr int __ga_cols_per_thread = {tile_n} / 32; \
         for (int __ga_c = 0; __ga_c < __ga_cols_per_thread; __ga_c++) {{ \
             int __ga_n = __ga_warp_col_base + __ga_c * 32 + __ga_lane; \
             float __ga_acc[{m}]; \
             _Pragma(\"unroll\") \
             for (int __ga_m = 0; __ga_m < {m}; __ga_m++) \
                 __ga_acc[__ga_m] = __bfloat162float(__ga_r[__ga_m * {n} + __ga_n]); \
             for (int __ga_k = 0; __ga_k < {k}; __ga_k++) {{ \
                 float __ga_b = __bfloat162float(__ga_b_gmem[__ga_k * {n} + __ga_n]); \
                 _Pragma(\"unroll\") \
                 for (int __ga_m = 0; __ga_m < {m}; __ga_m++) {{ \
                     float __ga_a = __bfloat162float(\
                         __ga_a_in[__ga_m * {k} + __ga_k]); \
                     __ga_acc[__ga_m] += __ga_a * __ga_b; \
                 }} \
             }} \
             _Pragma(\"unroll\") \
             for (int __ga_m = 0; __ga_m < {m}; __ga_m++) {{ \
                 __ga_r[__ga_m * {n} + __ga_n] = __float2bfloat16(__ga_acc[__ga_m]); \
             }} \
         }} \
         }}",
        in_id = in_page_id,
        res_id = residual_page_id,
        b_gmem = weight_gmem.expr(),
        tile_n = TILE_N,
        m = M,
        k = K,
        n = N,
    )));

    consumer.push(tk20::group_sync_named::<NCW>(bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive::<1>(&residual_done),
        tk20::group_arrive::<1>(&in_consumed),
    ]));

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&residual_done, storer_phase));
    storer.push(CuStmt::new(format!(
        "{{ \
         __nv_bfloat16* __ga_r_src = reinterpret_cast<__nv_bfloat16*>(ss.pages[{res_id}]); \
         kittens::group<1>::tma::store_async(\
             reinterpret_cast<void*>({r_gmem}), \
             reinterpret_cast<void*>(__ga_r_src), \
             {bytes}); \
         }}",
        res_id = residual_page_id,
        r_gmem = residual_gmem.expr(),
        bytes = residual_bytes,
    )));
    storer.push(tk20::group_tma_store_async_wait::<1>());
    storer.push(tk20::group_arrive::<1>(&residual_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}

// ============================================================
// TkFusedGemmAdd — `residual += A * B` (in-place residual fold).
// ============================================================

pub fn render_tk_fused_gemm_add<
    const M: u32,
    const K: u32,
    const N: u32,
    const TILE_N: u32,
    const NCW: u32,
    const NUM_LAYERS: u32,
    const ITERS: u32,
>(
    in_page_id: u32,
    weight_page_id: u32,
    residual_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    in_act_slot: u32,
    residual_act_slot: u32,
    weight_accessor: u32,
    bar_publish: u32,
    b_tile_offset: u32,
) -> RoleBodies {
    if ITERS != 1 {
        return RoleBodies::skipped("TkFusedGemmAdd");
    }
    // M%16!=0 → vec-mat decode-shape (small-batch decode).
    // M%16==0 → tile-MMA prefill-shape, dispatch onto fixed
    // M_BLOCKS (= M / 64) so the wgmma const-asserts only fire at
    // those instantiations (mirrors render_gemm dispatch).
    if M % 16 != 0 {
        return render_fused_gemm_add_decode::<M, K, N, TILE_N, NCW, NUM_LAYERS, ITERS>(
            in_page_id,
            weight_page_id,
            residual_page_id,
            consumer_phase,
            storer_phase,
            layer,
            in_act_slot,
            residual_act_slot,
            weight_accessor,
            bar_publish,
            b_tile_offset,
        );
    }
    macro_rules! dispatch_m_ga {
        ($mb:expr, $mt:expr) => {
            if TILE_N == 64 {
                render_tk_fused_gemm_add_prefill_wgmma::<$mb, $mt, K, N, 64, 1, NCW, NUM_LAYERS, ITERS>(
                    in_page_id, weight_page_id, residual_page_id,
                    consumer_phase, storer_phase, layer,
                    in_act_slot, residual_act_slot, weight_accessor,
                    bar_publish, b_tile_offset,
                )
            } else if TILE_N == 128 {
                render_tk_fused_gemm_add_prefill_wgmma::<$mb, $mt, K, N, 128, 1, NCW, NUM_LAYERS, ITERS>(
                    in_page_id, weight_page_id, residual_page_id,
                    consumer_phase, storer_phase, layer,
                    in_act_slot, residual_act_slot, weight_accessor,
                    bar_publish, b_tile_offset,
                )
            } else if TILE_N == 256 {
                render_tk_fused_gemm_add_prefill_wgmma::<$mb, $mt, K, N, 256, 1, NCW, NUM_LAYERS, ITERS>(
                    in_page_id, weight_page_id, residual_page_id,
                    consumer_phase, storer_phase, layer,
                    in_act_slot, residual_act_slot, weight_accessor,
                    bar_publish, b_tile_offset,
                )
            } else if TILE_N == 512 {
                render_tk_fused_gemm_add_prefill_wgmma::<$mb, $mt, K, N, 256, 2, NCW, NUM_LAYERS, ITERS>(
                    in_page_id, weight_page_id, residual_page_id,
                    consumer_phase, storer_phase, layer,
                    in_act_slot, residual_act_slot, weight_accessor,
                    bar_publish, b_tile_offset,
                )
            } else if TILE_N == 1024 {
                render_tk_fused_gemm_add_prefill_wgmma::<$mb, $mt, K, N, 256, 4, NCW, NUM_LAYERS, ITERS>(
                    in_page_id, weight_page_id, residual_page_id,
                    consumer_phase, storer_phase, layer,
                    in_act_slot, residual_act_slot, weight_accessor,
                    bar_publish, b_tile_offset,
                )
            } else if TILE_N == 2048 {
                render_tk_fused_gemm_add_prefill_wgmma::<$mb, $mt, K, N, 256, 8, NCW, NUM_LAYERS, ITERS>(
                    in_page_id, weight_page_id, residual_page_id,
                    consumer_phase, storer_phase, layer,
                    in_act_slot, residual_act_slot, weight_accessor,
                    bar_publish, b_tile_offset,
                )
            } else if TILE_N == 4096 {
                render_tk_fused_gemm_add_prefill_wgmma::<$mb, $mt, K, N, 256, 16, NCW, NUM_LAYERS, ITERS>(
                    in_page_id, weight_page_id, residual_page_id,
                    consumer_phase, storer_phase, layer,
                    in_act_slot, residual_act_slot, weight_accessor,
                    bar_publish, b_tile_offset,
                )
            } else {
                RoleBodies::skipped("TkFusedGemmAdd")
            }
        };
    }
    if M == 64 {
        dispatch_m_ga!(1, 64)
    } else if M == 512 {
        dispatch_m_ga!(8, 512)
    } else if M == 4096 {
        dispatch_m_ga!(64, 4096)
    } else {
        panic!("render_tk_fused_gemm_add prefill: unsupported M (must be 64/512/4096); got {M}");
    }
}

// ============================================================
// TkFusedGemmAdd (prefill shape — TK 2.0 H100 wgmma SMEM+SMEM).
//
// Mirrors `render_gemm_prefill_wgmma` with the residual fused
// in: each warpgroup-worth of D accumulator is preloaded from
// the residual subtile via `warpgroup::load(rt_fl, st_bf)`, then
// `mma_AB` accumulates `D += A * B` (TK 2.0 default
// `accumulate=1`), and the result is stored back into the
// residual smem slot. Same outer m-block loop and N-tile-per-
// warpgroup partition as render_gemm.
// ============================================================

#[allow(clippy::too_many_arguments)]
pub fn render_tk_fused_gemm_add_prefill_wgmma<
    const M_BLOCKS: u32,
    const M_TOTAL: u32,
    const K: u32,
    const N: u32,
    const WGMMA_N: u32,
    const N_MICRO: u32,
    const NCW: u32,
    const NUM_LAYERS: u32,
    const ITERS: u32,
>(
    in_page_id: u32,
    weight_page_id: u32,
    residual_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    in_act_slot: u32,
    residual_act_slot: u32,
    weight_accessor: u32,
    bar_publish: u32,
    b_tile_offset: u32,
) -> RoleBodies {
    const { assert!(NCW % 4 == 0, "render_tk_fused_gemm_add prefill: NCW must be a multiple of 4 (warpgroup size)"); }
    const { assert!(M_BLOCKS >= 1, "render_tk_fused_gemm_add prefill: M_BLOCKS must be >= 1"); }
    const { assert!(M_TOTAL == M_BLOCKS * 64, "render_tk_fused_gemm_add prefill: M_TOTAL must equal M_BLOCKS * 64"); }
    const { assert!(WGMMA_N >= 16 && WGMMA_N <= 256 && WGMMA_N % 16 == 0,
        "render_tk_fused_gemm_add prefill: WGMMA_N must be a multiple of 16 in [16, 256]"); }
    const { assert!(N_MICRO >= 1, "render_tk_fused_gemm_add prefill: N_MICRO must be >= 1"); }

    const M_TILE: u32 = 64;

    let loader_phase = storer_phase;
    let in_p = page(in_page_id);
    let weight_p = page(weight_page_id);
    let residual_p = page(residual_page_id);

    let in_smem = page_as_st_bf::<M_TOTAL, K>(in_p);
    let residual_smem = page_as_st_bf::<M_TOTAL, N>(residual_p);
    let b_tile = scratch_as_st_bf::<K, N>(off(b_tile_offset));

    let in_ready = page_ready_sem(in_p);
    let weight_ready = page_ready_sem(weight_p);
    let residual_ready = page_ready_sem(residual_p);
    let residual_done = page_done_sem(residual_p);
    let in_consumed = page_consumed_sem(in_p);
    let weight_consumed = page_consumed_sem(weight_p);
    let residual_consumed = page_consumed_sem(residual_p);

    let in_gmem = gmem_act_ptr_raw(in_act_slot);
    let residual_gmem = gmem_act_ptr_raw(residual_act_slot);
    let weight_gmem = gmem_weight_ptr_raw(weight_accessor, layer, NUM_LAYERS);

    let act_bytes = M_TOTAL * K * BF16_BYTES;
    let weight_bytes = K * N * BF16_BYTES;
    let residual_bytes = M_TOTAL * N * BF16_BYTES;

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait::<1>(&in_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&weight_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&residual_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes::<1>(&in_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf::<1, M_TOTAL, K>(
        &in_smem, &in_gmem, act_bytes, &in_ready,
    ));
    loader.push(tk20::group_tma_expect_bytes::<1>(&weight_ready, weight_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf::<1, K, N>(
        &b_tile, &weight_gmem, weight_bytes, &weight_ready,
    ));
    loader.push(tk20::group_tma_expect_bytes::<1>(&residual_ready, residual_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf::<1, M_TOTAL, N>(
        &residual_smem, &residual_gmem, residual_bytes, &residual_ready,
    ));

    let launcher = CuBlock::new();

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait::<1>(&in_ready, consumer_phase));
    consumer.push(tk20::group_wait::<1>(&weight_ready, consumer_phase));
    consumer.push(tk20::group_wait::<1>(&residual_ready, consumer_phase));

    let n_tiles_per_wg: u32 = 4;
    let groupid = tk20::warpgroup_groupid_expr();

    // A subtile shared across all N-microtiles in this m-block.
    let (decl_a_sub, a_sub_outer) = tk20::decl_st_bf_subtile::<M_TOTAL, K, M_TILE, K>(
        "__gemm_a_sub", &in_smem, "__gemm_m_block", "0",
    );

    // Innermost: per-microtile fused-residual wgmma.
    let mut micro_body = CuBlock::new();
    micro_body.push(CuStmt::new(format!(
        "int __gemm_b_col = __gemm_n_idx * {N_MICRO} + __gemm_nm;"
    )));
    let (decl_acc, acc_rt) = tk20::decl_rt_fl_warpgroup::<WGMMA_N>("__gemm_acc");
    let (decl_b_sub, b_sub) = tk20::decl_st_bf_subtile::<K, N, K, WGMMA_N>(
        "__gemm_b_sub", &b_tile, "0", "__gemm_b_col",
    );
    let (decl_resid_sub, resid_sub) =
        tk20::decl_st_bf_subtile::<M_TOTAL, N, M_TILE, WGMMA_N>(
            "__gemm_resid_sub", &residual_smem, "__gemm_m_block", "__gemm_b_col",
        );
    micro_body.push(decl_acc);
    micro_body.push(decl_b_sub);
    micro_body.push(decl_resid_sub);
    // Preload residual into D so wgmma's default accumulate gives
    // D = residual + A*B.
    micro_body.push(tk20::warpgroup_load_rt_fl_from_st_bf::<M_TILE, WGMMA_N>(&acc_rt, &resid_sub));
    micro_body.push(tk20::warpgroup_mma_AB::<M_TILE, K, WGMMA_N>(&acc_rt, &a_sub_outer, &b_sub));
    micro_body.push(tk20::warpgroup_mma_async_wait());
    micro_body.push(tk20::warpgroup_store_st_bf_from_rt_fl::<M_TILE, WGMMA_N>(&resid_sub, &acc_rt));

    let mut n_loop_body = CuBlock::new();
    n_loop_body.push(CuStmt::new(format!(
        "int __gemm_n_idx = {groupid} * {n_tiles_per_wg} + __gemm_n_in_wg;"
    )));
    n_loop_body.push(tk20::for_loop_no_unroll(
        &format!("int __gemm_nm = 0; __gemm_nm < {N_MICRO}; ++__gemm_nm"),
        &micro_body,
    ));

    let mut m_loop_body = CuBlock::new();
    m_loop_body.push(decl_a_sub);
    m_loop_body.push(tk20::for_loop_no_unroll(
        &format!("int __gemm_n_in_wg = 0; __gemm_n_in_wg < {n_tiles_per_wg}; ++__gemm_n_in_wg"),
        &n_loop_body,
    ));

    consumer.push(tk20::for_loop_no_unroll(
        &format!("int __gemm_m_block = 0; __gemm_m_block < {M_BLOCKS}; ++__gemm_m_block"),
        &m_loop_body,
    ));

    consumer.push(tk20::group_sync_named::<NCW>(bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive::<1>(&residual_done),
        tk20::group_arrive::<1>(&in_consumed),
        tk20::group_arrive::<1>(&weight_consumed),
    ]));

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&residual_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw_st_bf::<1, M_TOTAL, N>(
        &residual_gmem, &residual_smem, residual_bytes,
    ));
    storer.push(tk20::group_tma_store_async_wait::<1>());
    storer.push(tk20::group_arrive::<1>(&residual_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}

// ============================================================
// FusedGateUpActivateMul (decode shape, M%16!=0 — small-batch decode).
//
// Per-thread vec-mat with two output accs per row (gate, up), one
// shared activation lambda on gate-acc, multiply by up-acc, bf16
// store. Both gate and up weights are read directly from gmem
// (gate at base offset, up at +gate_bytes — same layout the
// prefill render uses with `gmem_weight_ptr_raw_offset`). Each
// (gate_gmem, up_gmem) load is shared across the M rows.
// ============================================================

#[allow(clippy::too_many_arguments)]
pub fn render_fused_gate_up_activate_mul_decode<
    const M: u32,
    const HIDDEN_DIM: u32,
    const INTERMEDIATE_DIM: u32,
    const TILE_N: u32,
    const NCW: u32,
    const NUM_LAYERS: u32,
    const ITERS: u32,
>(
    in_page_id: u32,
    _weight_page_id: u32,
    out_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    in_act_slot: u32,
    out_act_slot: u32,
    weight_accessor: u32,
    bar_publish: u32,
    gate_bytes: u32,
    activation: GateUpActivation,
) -> RoleBodies {
    if ITERS != 1 {
        return RoleBodies::skipped("TkFusedGateUpActivateMulDecode");
    }
    if M == 0 {
        return RoleBodies::skipped("TkFusedGateUpActivateMulDecode");
    }
    if NCW == 0 || INTERMEDIATE_DIM % NCW != 0 {
        return RoleBodies::skipped("TkFusedGateUpActivateMulDecode");
    }
    if TILE_N != INTERMEDIATE_DIM / NCW {
        return RoleBodies::skipped("TkFusedGateUpActivateMulDecode");
    }
    if TILE_N % 32 != 0 {
        return RoleBodies::skipped("TkFusedGateUpActivateMulDecode");
    }

    let loader_phase = storer_phase;
    let in_p = page(in_page_id);
    let out_p = page(out_page_id);

    let in_ready = page_ready_sem(in_p);
    let out_done = page_done_sem(out_p);
    let in_consumed = page_consumed_sem(in_p);
    let out_consumed = page_consumed_sem(out_p);

    let in_gmem = gmem_act_ptr_raw(in_act_slot);
    let out_gmem = gmem_act_ptr_raw(out_act_slot);
    let gate_gmem = gmem_weight_ptr_raw(weight_accessor, layer, NUM_LAYERS);
    let up_gmem = gmem_weight_ptr_raw_offset(weight_accessor, layer, NUM_LAYERS, gate_bytes);

    let act_bytes: u32 = M * HIDDEN_DIM * BF16_BYTES;
    let out_bytes: u32 = M * INTERMEDIATE_DIM * BF16_BYTES;

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait::<1>(&in_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&out_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes::<1>(&in_ready, act_bytes));
    loader.push(CuStmt::new(format!(
        "{{ \
         __nv_bfloat16* __gu_a_dst = reinterpret_cast<__nv_bfloat16*>(ss.pages[{in_id}]); \
         kittens::group<1>::tma::load_async(\
         reinterpret_cast<void*>(__gu_a_dst), \
         reinterpret_cast<void*>({a_gmem}), \
         {act_bytes}, {a_ready}); \
         }}",
        in_id = in_page_id,
        a_gmem = in_gmem.expr(),
        act_bytes = act_bytes,
        a_ready = in_ready.expr(),
    )));

    let launcher = CuBlock::new();

    let activation_expr = match activation {
        GateUpActivation::Silu => "__gu_x * (1.0f / (1.0f + __expf(-__gu_x)))",
        GateUpActivation::Gelu => {
            "0.5f * __gu_x * (1.0f + tanhf(0.7978845608028654f \
             * (__gu_x + 0.044715f * __gu_x * __gu_x * __gu_x)))"
        }
    };

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait::<1>(&in_ready, consumer_phase));
    consumer.push(CuStmt::new(format!(
        "{{ \
         const __nv_bfloat16* __gu_a_in = reinterpret_cast<const __nv_bfloat16*>(\
             ss.pages[{in_id}]); \
         __nv_bfloat16* __gu_a_out = reinterpret_cast<__nv_bfloat16*>(\
             ss.pages[{out_id}]); \
         const __nv_bfloat16* __gu_gate_gmem = {gate_gmem}; \
         const __nv_bfloat16* __gu_up_gmem = {up_gmem}; \
         int __gu_warp_id = static_cast<int>(kittens::warpid()); \
         int __gu_lane = static_cast<int>(kittens::laneid()); \
         int __gu_warp_col_base = __gu_warp_id * {tile_n}; \
         constexpr int __gu_cols_per_thread = {tile_n} / 32; \
         for (int __gu_c = 0; __gu_c < __gu_cols_per_thread; __gu_c++) {{ \
             int __gu_n = __gu_warp_col_base + __gu_c * 32 + __gu_lane; \
             float __gu_g[{m}]; \
             float __gu_u[{m}]; \
             _Pragma(\"unroll\") \
             for (int __gu_m = 0; __gu_m < {m}; __gu_m++) {{ \
                 __gu_g[__gu_m] = 0.0f; __gu_u[__gu_m] = 0.0f; \
             }} \
             for (int __gu_k = 0; __gu_k < {k}; __gu_k++) {{ \
                 float __gu_gw = __bfloat162float(__gu_gate_gmem[__gu_k * {n} + __gu_n]); \
                 float __gu_uw = __bfloat162float(__gu_up_gmem[__gu_k * {n} + __gu_n]); \
                 _Pragma(\"unroll\") \
                 for (int __gu_m = 0; __gu_m < {m}; __gu_m++) {{ \
                     float __gu_a = __bfloat162float(\
                         __gu_a_in[__gu_m * {k} + __gu_k]); \
                     __gu_g[__gu_m] += __gu_a * __gu_gw; \
                     __gu_u[__gu_m] += __gu_a * __gu_uw; \
                 }} \
             }} \
             _Pragma(\"unroll\") \
             for (int __gu_m = 0; __gu_m < {m}; __gu_m++) {{ \
                 float __gu_x = __gu_g[__gu_m]; \
                 float __gu_act = {act_expr}; \
                 __gu_a_out[__gu_m * {n} + __gu_n] = \
                     __float2bfloat16(__gu_act * __gu_u[__gu_m]); \
             }} \
         }} \
         }}",
        in_id = in_page_id,
        out_id = out_page_id,
        gate_gmem = gate_gmem.expr(),
        up_gmem = up_gmem.expr(),
        tile_n = TILE_N,
        m = M,
        k = HIDDEN_DIM,
        n = INTERMEDIATE_DIM,
        act_expr = activation_expr,
    )));

    consumer.push(tk20::group_sync_named::<NCW>(bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive::<1>(&out_done),
        tk20::group_arrive::<1>(&in_consumed),
    ]));

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&out_done, storer_phase));
    storer.push(CuStmt::new(format!(
        "{{ \
         __nv_bfloat16* __gu_out_src = reinterpret_cast<__nv_bfloat16*>(ss.pages[{out_id}]); \
         kittens::group<1>::tma::store_async(\
             reinterpret_cast<void*>({out_gmem}), \
             reinterpret_cast<void*>(__gu_out_src), \
             {bytes}); \
         }}",
        out_id = out_page_id,
        out_gmem = out_gmem.expr(),
        bytes = out_bytes,
    )));
    storer.push(tk20::group_tma_store_async_wait::<1>());
    storer.push(tk20::group_arrive::<1>(&out_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}

// ============================================================
// FusedGateUpActivateMul — out = activation(A @ W_gate) * (A @ W_up).
// ============================================================

pub fn render_fused_gate_up_activate_mul<
    const M: u32,
    const HIDDEN_DIM: u32,
    const INTERMEDIATE_DIM: u32,
    const TILE_N: u32,
    const NCW: u32,
    const NUM_LAYERS: u32,
    const ITERS: u32,
>(
    in_page_id: u32,
    weight_page_id: u32,
    out_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    in_act_slot: u32,
    out_act_slot: u32,
    weight_accessor: u32,
    bar_publish: u32,
    gate_offset: u32,
    up_offset: u32,
    gate_bytes: u32,
    up_bytes: u32,
    activation: GateUpActivation,
) -> RoleBodies {
    if ITERS != 1 {
        return RoleBodies::skipped("TkFusedGateUpActivateMul");
    }
    // M%16!=0 → vec-mat decode-shape (small-batch decode).
    // M%16==0 → tile-MMA prefill-shape, dispatch onto fixed
    // M_BLOCKS so wgmma const-asserts only fire at those instantiations.
    if M % 16 != 0 {
        return render_fused_gate_up_activate_mul_decode::<
            M, HIDDEN_DIM, INTERMEDIATE_DIM, TILE_N, NCW, NUM_LAYERS, ITERS,
        >(
            in_page_id,
            weight_page_id,
            out_page_id,
            consumer_phase,
            storer_phase,
            layer,
            in_act_slot,
            out_act_slot,
            weight_accessor,
            bar_publish,
            gate_bytes,
            activation,
        );
    }
    macro_rules! dispatch_m_gu {
        ($mb:expr, $mt:expr) => {
            if TILE_N == 64 {
                render_fused_gate_up_activate_mul_prefill_wgmma::<$mb, $mt, HIDDEN_DIM, INTERMEDIATE_DIM, 64, 1, NCW, NUM_LAYERS, ITERS>(
                    in_page_id, weight_page_id, out_page_id,
                    consumer_phase, storer_phase, layer,
                    in_act_slot, out_act_slot, weight_accessor, bar_publish,
                    gate_offset, up_offset, gate_bytes, up_bytes, activation,
                )
            } else if TILE_N == 128 {
                render_fused_gate_up_activate_mul_prefill_wgmma::<$mb, $mt, HIDDEN_DIM, INTERMEDIATE_DIM, 128, 1, NCW, NUM_LAYERS, ITERS>(
                    in_page_id, weight_page_id, out_page_id,
                    consumer_phase, storer_phase, layer,
                    in_act_slot, out_act_slot, weight_accessor, bar_publish,
                    gate_offset, up_offset, gate_bytes, up_bytes, activation,
                )
            } else if TILE_N == 256 {
                render_fused_gate_up_activate_mul_prefill_wgmma::<$mb, $mt, HIDDEN_DIM, INTERMEDIATE_DIM, 256, 1, NCW, NUM_LAYERS, ITERS>(
                    in_page_id, weight_page_id, out_page_id,
                    consumer_phase, storer_phase, layer,
                    in_act_slot, out_act_slot, weight_accessor, bar_publish,
                    gate_offset, up_offset, gate_bytes, up_bytes, activation,
                )
            } else if TILE_N == 512 {
                render_fused_gate_up_activate_mul_prefill_wgmma::<$mb, $mt, HIDDEN_DIM, INTERMEDIATE_DIM, 256, 2, NCW, NUM_LAYERS, ITERS>(
                    in_page_id, weight_page_id, out_page_id,
                    consumer_phase, storer_phase, layer,
                    in_act_slot, out_act_slot, weight_accessor, bar_publish,
                    gate_offset, up_offset, gate_bytes, up_bytes, activation,
                )
            } else if TILE_N == 1024 {
                render_fused_gate_up_activate_mul_prefill_wgmma::<$mb, $mt, HIDDEN_DIM, INTERMEDIATE_DIM, 256, 4, NCW, NUM_LAYERS, ITERS>(
                    in_page_id, weight_page_id, out_page_id,
                    consumer_phase, storer_phase, layer,
                    in_act_slot, out_act_slot, weight_accessor, bar_publish,
                    gate_offset, up_offset, gate_bytes, up_bytes, activation,
                )
            } else if TILE_N == 2048 {
                render_fused_gate_up_activate_mul_prefill_wgmma::<$mb, $mt, HIDDEN_DIM, INTERMEDIATE_DIM, 256, 8, NCW, NUM_LAYERS, ITERS>(
                    in_page_id, weight_page_id, out_page_id,
                    consumer_phase, storer_phase, layer,
                    in_act_slot, out_act_slot, weight_accessor, bar_publish,
                    gate_offset, up_offset, gate_bytes, up_bytes, activation,
                )
            } else if TILE_N == 4096 {
                render_fused_gate_up_activate_mul_prefill_wgmma::<$mb, $mt, HIDDEN_DIM, INTERMEDIATE_DIM, 256, 16, NCW, NUM_LAYERS, ITERS>(
                    in_page_id, weight_page_id, out_page_id,
                    consumer_phase, storer_phase, layer,
                    in_act_slot, out_act_slot, weight_accessor, bar_publish,
                    gate_offset, up_offset, gate_bytes, up_bytes, activation,
                )
            } else {
                RoleBodies::skipped("TkFusedGateUpActivateMul")
            }
        };
    }
    if M == 64 {
        dispatch_m_gu!(1, 64)
    } else if M == 512 {
        dispatch_m_gu!(8, 512)
    } else if M == 4096 {
        dispatch_m_gu!(64, 4096)
    } else {
        panic!("render_fused_gate_up_activate_mul prefill: unsupported M (must be 64/512/4096); got {M}");
    }
}

// ============================================================
// FusedGateUpActivateMul (prefill — TK 2.0 H100 wgmma SMEM+SMEM).
//
// Outer m-block CUDA loop wraps a per-warpgroup N-tile loop. For
// each (m_block, n_in_wg) the body:
//   1. Subtiles A=st_bf<64, HIDDEN_DIM>, gate_b=st_bf<HIDDEN_DIM, TILE_N>,
//      up_b=st_bf<HIDDEN_DIM, TILE_N>, out=st_bf<64, TILE_N>
//   2. zeroes gate_acc, mma gate_acc += A*gate_b (wgmma SMEM+SMEM)
//   3. zeroes up_acc, mma up_acc += A*up_b
//   4. mma_async_wait; apply(activation, gate_acc); mul(gate_acc, gate_acc, up_acc)
//   5. warpgroup store st_bf out_sub from rt_fl gate_acc
// ============================================================

#[allow(clippy::too_many_arguments)]
pub fn render_fused_gate_up_activate_mul_prefill_wgmma<
    const M_BLOCKS: u32,
    const M_TOTAL: u32,
    const HIDDEN_DIM: u32,
    const INTERMEDIATE_DIM: u32,
    const WGMMA_N: u32,
    const N_MICRO: u32,
    const NCW: u32,
    const NUM_LAYERS: u32,
    const ITERS: u32,
>(
    in_page_id: u32,
    weight_page_id: u32,
    out_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    in_act_slot: u32,
    out_act_slot: u32,
    weight_accessor: u32,
    bar_publish: u32,
    gate_offset: u32,
    up_offset: u32,
    gate_bytes: u32,
    up_bytes: u32,
    activation: GateUpActivation,
) -> RoleBodies {
    const { assert!(NCW % 4 == 0, "render_fused_gate_up_activate_mul prefill: NCW must be a multiple of 4 (warpgroup size)"); }
    const { assert!(M_BLOCKS >= 1, "render_fused_gate_up_activate_mul prefill: M_BLOCKS must be >= 1"); }
    const { assert!(M_TOTAL == M_BLOCKS * 64, "render_fused_gate_up_activate_mul prefill: M_TOTAL must equal M_BLOCKS * 64"); }
    const { assert!(WGMMA_N >= 16 && WGMMA_N <= 256 && WGMMA_N % 16 == 0,
        "render_fused_gate_up_activate_mul prefill: WGMMA_N must be a multiple of 16 in [16, 256] (TK 2.0 wgmma::base specializations)"); }
    const { assert!(N_MICRO >= 1, "render_fused_gate_up_activate_mul prefill: N_MICRO must be >= 1"); }

    const M_TILE: u32 = 64;

    let loader_phase = storer_phase;
    let in_p = page(in_page_id);
    let weight_p = page(weight_page_id);
    let out_p = page(out_page_id);

    let in_smem = page_as_st_bf::<M_TOTAL, HIDDEN_DIM>(in_p);
    let out_smem = page_as_st_bf::<M_TOTAL, INTERMEDIATE_DIM>(out_p);
    let gate_buf = scratch_as_st_bf::<HIDDEN_DIM, INTERMEDIATE_DIM>(off(gate_offset));
    let up_buf = scratch_as_st_bf::<HIDDEN_DIM, INTERMEDIATE_DIM>(off(up_offset));

    let in_ready = page_ready_sem(in_p);
    let weight_ready = page_ready_sem(weight_p);
    let out_done = page_done_sem(out_p);
    let in_consumed = page_consumed_sem(in_p);
    let weight_consumed = page_consumed_sem(weight_p);
    let out_consumed = page_consumed_sem(out_p);

    let in_gmem = gmem_act_ptr_raw(in_act_slot);
    let out_gmem = gmem_act_ptr_raw(out_act_slot);
    let gate_gmem = gmem_weight_ptr_raw(weight_accessor, layer, NUM_LAYERS);
    let up_gmem = gmem_weight_ptr_raw_offset(weight_accessor, layer, NUM_LAYERS, gate_bytes);

    let act_bytes = M_TOTAL * HIDDEN_DIM * BF16_BYTES;
    let out_bytes = M_TOTAL * INTERMEDIATE_DIM * BF16_BYTES;
    let weight_total_bytes = gate_bytes + up_bytes;

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait::<1>(&in_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&weight_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&out_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes::<1>(&in_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf::<1, M_TOTAL, HIDDEN_DIM>(
        &in_smem, &in_gmem, act_bytes, &in_ready,
    ));
    loader.push(tk20::group_tma_expect_bytes::<1>(&weight_ready, weight_total_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf::<1, HIDDEN_DIM, INTERMEDIATE_DIM>(
        &gate_buf, &gate_gmem, gate_bytes, &weight_ready,
    ));
    loader.push(tk20::group_tma_load_async_raw_st_bf::<1, HIDDEN_DIM, INTERMEDIATE_DIM>(
        &up_buf, &up_gmem, up_bytes, &weight_ready,
    ));

    let launcher = CuBlock::new();

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait::<1>(&in_ready, consumer_phase));
    consumer.push(tk20::group_wait::<1>(&weight_ready, consumer_phase));

    let n_tiles_per_wg: u32 = 4;
    let groupid = tk20::warpgroup_groupid_expr();

    let activation_lambda = match activation {
        GateUpActivation::Silu => "x * (1.0f / (1.0f + __expf(-x)))",
        GateUpActivation::Gelu => {
            "0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)))"
        }
    };

    // A subtile: 64-row slice of in_smem along the M dim, declared
    // once per m-block (shared across all N-microtiles in this block).
    let (decl_a_sub, a_sub_outer) = tk20::decl_st_bf_subtile::<M_TOTAL, HIDDEN_DIM, M_TILE, HIDDEN_DIM>(
        "__gu_a_sub", &in_smem, "__gu_m_block", "0",
    );

    // Innermost: per-microtile wgmma over WGMMA_N cols. Fresh
    // gate_acc/up_acc declared per-iteration (registers reused).
    let mut micro_body = CuBlock::new();
    micro_body.push(CuStmt::new(format!(
        "int __gu_b_col = __gu_n_idx * {N_MICRO} + __gu_nm;"
    )));
    let (decl_gate_acc, gate_acc_rt) = tk20::decl_rt_fl_warpgroup::<WGMMA_N>("__gu_gate_acc");
    let (decl_up_acc, up_acc_rt) = tk20::decl_rt_fl_warpgroup::<WGMMA_N>("__gu_up_acc");
    let (decl_gate_b_sub, gate_b_sub) =
        tk20::decl_st_bf_subtile::<HIDDEN_DIM, INTERMEDIATE_DIM, HIDDEN_DIM, WGMMA_N>(
            "__gu_gate_b_sub", &gate_buf, "0", "__gu_b_col",
        );
    let (decl_up_b_sub, up_b_sub) =
        tk20::decl_st_bf_subtile::<HIDDEN_DIM, INTERMEDIATE_DIM, HIDDEN_DIM, WGMMA_N>(
            "__gu_up_b_sub", &up_buf, "0", "__gu_b_col",
        );
    let (decl_out_sub, out_sub) =
        tk20::decl_st_bf_subtile::<M_TOTAL, INTERMEDIATE_DIM, M_TILE, WGMMA_N>(
            "__gu_out_sub", &out_smem, "__gu_m_block", "__gu_b_col",
        );
    micro_body.push(decl_gate_acc);
    micro_body.push(decl_up_acc);
    micro_body.push(decl_gate_b_sub);
    micro_body.push(decl_up_b_sub);
    micro_body.push(decl_out_sub);
    micro_body.push(tk20::warpgroup_zero_rt_fl::<WGMMA_N>(&gate_acc_rt));
    micro_body.push(tk20::warpgroup_zero_rt_fl::<WGMMA_N>(&up_acc_rt));
    micro_body.push(tk20::warpgroup_mma_AB::<M_TILE, HIDDEN_DIM, WGMMA_N>(&gate_acc_rt, &a_sub_outer, &gate_b_sub));
    micro_body.push(tk20::warpgroup_mma_AB::<M_TILE, HIDDEN_DIM, WGMMA_N>(&up_acc_rt, &a_sub_outer, &up_b_sub));
    micro_body.push(tk20::warpgroup_mma_async_wait());
    micro_body.push(tk20::warpgroup_apply_f32_rt_lambda::<WGMMA_N>(
        &gate_acc_rt, &gate_acc_rt, activation_lambda,
    ));
    micro_body.push(tk20::warpgroup_mul_rt_rt::<WGMMA_N>(&gate_acc_rt, &gate_acc_rt, &up_acc_rt));
    micro_body.push(tk20::warpgroup_store_st_bf_from_rt_fl::<M_TILE, WGMMA_N>(&out_sub, &gate_acc_rt));

    // Inner N-tile loop body (per m-block). Each warpgroup
    // owns 4 of the NCW N-tiles; A subtile shared across micros.
    let mut n_loop_body = CuBlock::new();
    n_loop_body.push(CuStmt::new(format!(
        "int __gu_n_idx = {groupid} * {n_tiles_per_wg} + __gu_n_in_wg;"
    )));
    n_loop_body.push(tk20::for_loop_no_unroll(
        &format!("int __gu_nm = 0; __gu_nm < {N_MICRO}; ++__gu_nm"),
        &micro_body,
    ));

    let mut m_loop_body = CuBlock::new();
    m_loop_body.push(decl_a_sub);
    m_loop_body.push(tk20::for_loop_no_unroll(
        &format!("int __gu_n_in_wg = 0; __gu_n_in_wg < {n_tiles_per_wg}; ++__gu_n_in_wg"),
        &n_loop_body,
    ));

    consumer.push(tk20::for_loop_no_unroll(
        &format!("int __gu_m_block = 0; __gu_m_block < {M_BLOCKS}; ++__gu_m_block"),
        &m_loop_body,
    ));

    consumer.push(tk20::group_sync_named::<NCW>(bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive::<1>(&out_done),
        tk20::group_arrive::<1>(&in_consumed),
        tk20::group_arrive::<1>(&weight_consumed),
    ]));

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&out_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw_st_bf::<1, M_TOTAL, INTERMEDIATE_DIM>(
        &out_gmem, &out_smem, out_bytes,
    ));
    storer.push(tk20::group_tma_store_async_wait::<1>());
    storer.push(tk20::group_arrive::<1>(&out_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}

// ============================================================
// TkFusedNormGemm — `out = (norm(in [+ delta]) * norm_w) @ lin_w`.
// ============================================================

pub fn render_tk_fused_norm_gemm<
    const M: u32,
    const K: u32,
    const N: u32,
    const TILE_N: u32,
    const NCW: u32,
    const K_PER_WARP: u32,
    const NUM_LAYERS: u32,
    const ITERS: u32,
>(
    in_page_id: u32,
    delta_page_id: Option<u32>,
    norm_weight_page_id: u32,
    linear_weight_page_id: u32,
    out_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    in_act_slot: u32,
    delta_act_slot: Option<u32>,
    out_act_slot: u32,
    norm_weight_accessor: u32,
    linear_weight_accessor: u32,
    bar_reduce: u32,
    bar_publish: u32,
    eps: f32,
    norm_kind: LmHeadNormKind,
    offset_opt: Option<f32>,
    b_tile_offset: u32,
    partial_offset: u32,
) -> RoleBodies {
    if ITERS != 1 {
        return RoleBodies::skipped("TkFusedNormGemm");
    }
    // lm_head decode: N-streaming with direct-gmem B reads and direct-
    // gmem output writes. Output `[M, N]` (e.g. [1, 128256]) does not
    // fit a substrate page (32 KB), so the prefill emit's `out_st =
    // page_as_st_bf::<M, N>(out_p)` + `tma_store_async_raw_st_bf::<1,
    // M, N>(out_gmem, out_st, M*N*2)` reads M*N*2 bytes from a 32 KB
    // region (UB) and asks cicc to instantiate `rt_fl<M, N/NCW>` which
    // blew up to ~36 GB / >60 min on H100 at M=512 N=128256 NCW=8.
    //
    // Decode-first scope: M=1 only. M>1 is gated on multi-row rmsnorm
    // (Task #10) — the existing rmsnorm phase here partitions K across
    // NCW warps but processes only the first row of `[M, K]`, so M>1
    // would silently produce row-0 normalized × W for all rows.
    if M != 1 {
        return RoleBodies::skipped("TkFusedNormGemm");
    }
    // Per-thread vec-mat needs each warp to own a clean TILE_N=N/NCW
    // slab divisible by 32 (lane-per-col).
    if NCW == 0 || N % NCW != 0 {
        return RoleBodies::skipped("TkFusedNormGemm");
    }
    let tile_n = N / NCW;
    if tile_n == 0 || tile_n % 32 != 0 {
        return RoleBodies::skipped("TkFusedNormGemm");
    }
    // A in `in_page` (post-rmsnorm) is M*K*2 bytes; must fit page.
    // M=1 K=2048 → 4 KB. Holds.
    let _ = TILE_N;
    let _ = K_PER_WARP;

    let loader_phase = storer_phase;
    let in_p = page(in_page_id);
    let delta_p_opt = delta_page_id.map(page);
    let norm_p = page(norm_weight_page_id);
    let lin_p = page(linear_weight_page_id);
    let out_p = page(out_page_id);

    let in_sv = page_as_sv_bf::<K>(in_p);
    let norm_weight_sv = page_as_sv_bf::<K>(norm_p);
    let partial = scratch_as::<F32>(off(partial_offset));
    let _ = b_tile_offset;

    let in_ready = page_ready_sem(in_p);
    let norm_weight_ready = page_ready_sem(norm_p);
    let out_done = page_done_sem(out_p);
    let in_consumed = page_consumed_sem(in_p);
    let norm_weight_consumed = page_consumed_sem(norm_p);
    let lin_weight_consumed = page_consumed_sem(lin_p);
    let out_consumed = page_consumed_sem(out_p);

    let in_gmem = gmem_act_ptr_raw(in_act_slot);
    let out_gmem = gmem_act_ptr_raw(out_act_slot);
    let norm_weight_gmem = gmem_weight_ptr_raw(norm_weight_accessor, layer, NUM_LAYERS);
    let lin_weight_gmem = gmem_weight_ptr_raw(linear_weight_accessor, layer, NUM_LAYERS);

    let act_bytes = M * K * BF16_BYTES;
    let norm_weight_bytes = K * BF16_BYTES;

    let delta_sv_opt = delta_p_opt.map(page_as_sv_bf::<K>);
    let delta_ready_opt = delta_p_opt.map(page_ready_sem);
    let delta_consumed_opt = delta_p_opt.map(page_consumed_sem);
    let delta_gmem_opt = delta_act_slot.map(gmem_act_ptr_raw);

    // LOADER: TMA-load A (4 KB at K=2048) + delta (if present, 4 KB)
    // + norm_weight (4 KB). Linear weight `[K, N]` is read directly
    // from gmem by the consumer's vec-mat — no TMA stage (would be
    // ~512 MB at K=2048, N=128256, BF16). Consumer therefore does NOT
    // wait on `lin_weight_ready`; we still arrive at
    // `lin_weight_consumed` from the consumer below to keep the
    // loader's cross-iteration `wait(lin_weight_consumed)` honest.
    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait::<1>(&in_consumed, loader_phase));
    if let Some(delta_consumed) = delta_consumed_opt.as_ref() {
        loader.push(tk20::group_wait::<1>(delta_consumed, loader_phase));
    }
    loader.push(tk20::group_wait::<1>(&norm_weight_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&lin_weight_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&out_consumed, loader_phase));

    loader.push(tk20::group_tma_expect_bytes::<1>(&in_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw::<1, K>(
        &in_sv, &in_gmem, act_bytes, &in_ready,
    ));

    if let (Some(delta_sv), Some(delta_ready), Some(delta_gmem)) = (
        delta_sv_opt.as_ref(),
        delta_ready_opt.as_ref(),
        delta_gmem_opt.as_ref(),
    ) {
        loader.push(tk20::group_tma_expect_bytes::<1>(delta_ready, act_bytes));
        loader.push(tk20::group_tma_load_async_raw::<1, K>(
            delta_sv, delta_gmem, act_bytes, delta_ready,
        ));
    }

    loader.push(tk20::group_tma_expect_bytes::<1>(&norm_weight_ready, norm_weight_bytes));
    loader.push(tk20::group_tma_load_async_raw::<1, K>(
        &norm_weight_sv, &norm_weight_gmem, norm_weight_bytes, &norm_weight_ready,
    ));

    let launcher = CuBlock::new();

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait::<1>(&in_ready, consumer_phase));
    if let Some(delta_ready) = delta_ready_opt.as_ref() {
        consumer.push(tk20::group_wait::<1>(delta_ready, consumer_phase));
    }
    consumer.push(tk20::group_wait::<1>(&norm_weight_ready, consumer_phase));
    // No `lin_weight_ready` wait: linear weight is read direct from
    // gmem by the per-thread vec-mat below.

    let (decl_act, act_rv) = tk20::decl_rv_fl::<K_PER_WARP>("__lmh_act_rv");
    let (decl_sq, sq_rv) = tk20::decl_rv_fl::<K_PER_WARP>("__lmh_sq_rv");
    let (decl_w, weight_rv) = tk20::decl_rv_fl::<K_PER_WARP>("__lmh_norm_w_rv");
    consumer.push(decl_act);
    consumer.push(decl_sq);
    consumer.push(decl_w);

    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, K>(
        &act_rv, &in_sv,
    ));

    if let Some(delta_sv) = delta_sv_opt.as_ref() {
        let (decl_delta, delta_rv) = tk20::decl_rv_fl::<K_PER_WARP>("__lmh_delta_rv");
        consumer.push(decl_delta);
        consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, K>(
            &delta_rv, delta_sv,
        ));
        consumer.push(tk20::warp_add_rv_rv::<K_PER_WARP, _>(&act_rv, &act_rv, &delta_rv));
    }

    if matches!(norm_kind, LmHeadNormKind::MeanSubRmsNorm) {
        let (decl_msum, msum_local) = tk20::decl_local_f32("__lmh_mean_partial", "0.0f");
        consumer.push(decl_msum);
        consumer.push(tk20::warp_sum_to_scalar_f32::<K_PER_WARP>(&msum_local, &act_rv));
        let (decl_full_msum, full_msum) = tk20::decl_local_f32("__lmh_mean_full", "0.0f");
        consumer.push(decl_full_msum);
        consumer.push(tk20::cross_warp_reduce_sum_f32::<NCW>(
            full_msum.as_str(),
            msum_local.as_str(),
            &partial,
            bar_reduce,
        ));
        let neg_mean = CuExpr::new(format!(
            "-({} * (1.0f / {K}.0f))",
            full_msum.as_str(),
        ));
        consumer.push(tk20::warp_add_rv_scalar_f32::<K_PER_WARP>(
            &act_rv, &act_rv, &neg_mean,
        ));
    }

    consumer.push(tk20::warp_copy_rv::<F32, K_PER_WARP, _>(&sq_rv, &act_rv));
    consumer.push(tk20::warp_mul_rv_rv::<K_PER_WARP, _>(&sq_rv, &sq_rv, &sq_rv));
    let (decl_partial, partial_sum_expr) = tk20::decl_local_f32("__lmh_partial_sum", "0.0f");
    consumer.push(decl_partial);
    consumer.push(tk20::warp_sum_to_scalar_f32::<K_PER_WARP>(&partial_sum_expr, &sq_rv));

    let (decl_full, full_sum_expr) = tk20::decl_local_f32("__lmh_full_sum", "0.0f");
    consumer.push(decl_full);
    consumer.push(tk20::cross_warp_reduce_sum_f32::<NCW>(
        full_sum_expr.as_str(),
        partial_sum_expr.as_str(),
        &partial,
        bar_reduce,
    ));
    let (decl_scale, scale_expr) = tk20::decl_rms_scale_local::<K>(
        "__lmh_scale",
        full_sum_expr.as_str(),
        eps,
    );
    consumer.push(decl_scale);

    consumer.push(tk20::warp_mul_rv_scalar_f32::<K_PER_WARP>(&act_rv, &act_rv, &scale_expr));
    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, K>(
        &weight_rv, &norm_weight_sv,
    ));
    if let Some(offset_value) = offset_opt {
        let offset_lit = CuExpr::new(format!("{:e}f", offset_value));
        consumer.push(tk20::warp_add_rv_scalar_f32::<K_PER_WARP>(
            &weight_rv, &weight_rv, &offset_lit,
        ));
    }
    consumer.push(tk20::warp_mul_rv_rv::<K_PER_WARP, _>(&act_rv, &act_rv, &weight_rv));

    consumer.push(tk20::group_store_rv_to_sv_f32_to_bf16::<NCW, K_PER_WARP, K>(
        &in_sv, &act_rv,
    ));

    consumer.push(tk20::group_sync_named::<NCW>(bar_publish));

    // -----------------------------------------------------------
    // GEMM phase — per-thread vec-mat with N-streaming.
    // Each warp owns `tile_n = N / NCW` output cols; each thread
    // owns `cols_per_thread = tile_n / 32` cols. A is read from
    // `ss.pages[in_page_id]` (post-rmsnorm bf16, K elements at M=1).
    // B (`[K, N]` bf16) is read directly from gmem — no scratch /
    // page staging (full B is ~512 MB at lm_head shape, doesn't fit
    // a 32 KB scratch let alone a page). Output is written direct
    // to `act_ptrs[out_act_slot]` (M*N*BF16_BYTES bytes; doesn't fit
    // a page either at vocab=128256). cicc instantiates one `float`
    // accumulator per thread per col-iter — bounded register usage,
    // no fictional `rt_fl<512, 16032>`.
    consumer.push(CuStmt::new(format!(
        "{{ \
         const __nv_bfloat16* __lmh_a_in = reinterpret_cast<const __nv_bfloat16*>(\
             ss.pages[{in_id}]); \
         const __nv_bfloat16* __lmh_b_gmem = {b_gmem}; \
         __nv_bfloat16* __lmh_out_gmem = {out_gmem}; \
         int __lmh_warp_id = static_cast<int>(kittens::warpid()); \
         int __lmh_lane = static_cast<int>(kittens::laneid()); \
         int __lmh_warp_col_base = __lmh_warp_id * {tile_n}; \
         constexpr int __lmh_cols_per_thread = {tile_n} / 32; \
         for (int __lmh_c = 0; __lmh_c < __lmh_cols_per_thread; __lmh_c++) {{ \
             int __lmh_n = __lmh_warp_col_base + __lmh_c * 32 + __lmh_lane; \
             float __lmh_acc[{m}]; \
             _Pragma(\"unroll\") \
             for (int __lmh_m = 0; __lmh_m < {m}; __lmh_m++) __lmh_acc[__lmh_m] = 0.0f; \
             for (int __lmh_k = 0; __lmh_k < {k}; __lmh_k++) {{ \
                 float __lmh_b = __bfloat162float(__lmh_b_gmem[__lmh_k * {n} + __lmh_n]); \
                 _Pragma(\"unroll\") \
                 for (int __lmh_m = 0; __lmh_m < {m}; __lmh_m++) {{ \
                     float __lmh_a = __bfloat162float(\
                         __lmh_a_in[__lmh_m * {k} + __lmh_k]); \
                     __lmh_acc[__lmh_m] += __lmh_a * __lmh_b; \
                 }} \
             }} \
             _Pragma(\"unroll\") \
             for (int __lmh_m = 0; __lmh_m < {m}; __lmh_m++) {{ \
                 __lmh_out_gmem[__lmh_m * {n} + __lmh_n] = __float2bfloat16(__lmh_acc[__lmh_m]); \
             }} \
         }} \
         }}",
        in_id = in_page_id,
        b_gmem = lin_weight_gmem.expr(),
        out_gmem = out_gmem.expr(),
        tile_n = tile_n,
        m = M,
        k = K,
        n = N,
    )));

    consumer.push(tk20::group_sync_named::<NCW>(bar_publish));

    let mut arrives = vec![
        tk20::group_arrive::<1>(&out_done),
        tk20::group_arrive::<1>(&in_consumed),
        tk20::group_arrive::<1>(&norm_weight_consumed),
        tk20::group_arrive::<1>(&lin_weight_consumed),
    ];
    if let Some(delta_consumed) = delta_consumed_opt.as_ref() {
        arrives.push(tk20::group_arrive::<1>(delta_consumed));
    }
    consumer.push(tk20::block_warp_zero(&arrives));

    // STORER: output already lives in gmem (consumer wrote direct).
    // We still bracket out_done -> out_consumed to keep the loader's
    // cross-iteration `wait(out_consumed)` honest; no TMA store.
    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&out_done, storer_phase));
    storer.push(tk20::group_arrive::<1>(&out_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}

// ============================================================
// FusedQkvRopeCache (decode shape, M%16!=0 — small-batch decode).
//
// Per-thread vec-mat over the fused QKV weight tile (read directly
// from gmem at QKV_N width), staging Q/K dot products into scratch
// `q_rope_offset` / `k_rope_offset` (laid out [M, Q_DIM] / [M,
// KV_DIM] bf16 row-major), then within the same warp re-reading
// the paired column per row, applying per-row RoPE, and writing
// to q_out_page / k_out_page. V columns bypass RoPE and are
// stored directly into v_out_page in phase 1.
//
// Pair locality: TILE_N = HEADS_PER_WARP * HEAD_DIM, so every
// `(col, paired_col)` pair lies within the same warp's TILE_N
// range. Intra-warp `__syncwarp()` is sufficient; no cross-warp
// barrier needed for the RoPE read.
//
// cos_sin layout: cos_sin_per_token_gather writes
// `[cos[0..HEAD_DIM/2], sin[0..HEAD_DIM/2]]` for each of M tokens
// at row `positions[t]` of the rotary table. cos_sin_page holds
// M * HEAD_DIM bf16 (per-row cos/sin pair).
// ============================================================

#[allow(clippy::too_many_arguments)]
pub fn render_fused_qkv_rope_cache_decode<
    const M: u32,
    const HIDDEN_DIM: u32,
    const HEAD_DIM: u32,
    const Q_DIM: u32,
    const KV_DIM: u32,
    const QKV_N: u32,
    const TILE_N: u32,
    const NCW: u32,
    const NUM_LAYERS: u32,
    const ITERS: u32,
>(
    in_page_id: u32,
    _qkv_weight_page_id: u32,
    cos_sin_page_id: u32,
    q_out_page_id: u32,
    k_out_page_id: u32,
    v_out_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    in_act_slot: u32,
    q_out_act_slot: u32,
    k_out_act_slot: u32,
    v_out_act_slot: u32,
    qkv_weight_accessor: u32,
    rotary_accessor: u32,
    bar_publish: u32,
    q_rope_offset: u32,
    k_rope_offset: u32,
    _qkv_b_tile_offset: u32,
) -> RoleBodies {
    if ITERS != 1 {
        return RoleBodies::skipped("TkFusedQkvRopeCacheDecode");
    }
    if M == 0 {
        return RoleBodies::skipped("TkFusedQkvRopeCacheDecode");
    }
    if HEAD_DIM == 0 || HEAD_DIM % 2 != 0 {
        return RoleBodies::skipped("TkFusedQkvRopeCacheDecode");
    }
    if NCW == 0 || QKV_N % NCW != 0 {
        return RoleBodies::skipped("TkFusedQkvRopeCacheDecode");
    }
    if TILE_N != QKV_N / NCW {
        return RoleBodies::skipped("TkFusedQkvRopeCacheDecode");
    }
    if TILE_N % 32 != 0 || TILE_N % HEAD_DIM != 0 {
        return RoleBodies::skipped("TkFusedQkvRopeCacheDecode");
    }
    // Pair locality: every `(col, col ± HEAD_DIM/2)` pair must lie
    // in the same warp's TILE_N. With TILE_N % HEAD_DIM == 0 and
    // pairs always within the same head, this holds.
    if Q_DIM % HEAD_DIM != 0 || KV_DIM % HEAD_DIM != 0 {
        return RoleBodies::skipped("TkFusedQkvRopeCacheDecode");
    }

    let loader_phase = storer_phase;
    let in_p = page(in_page_id);
    let cos_sin_p = page(cos_sin_page_id);
    let q_out_p = page(q_out_page_id);
    let k_out_p = page(k_out_page_id);
    let v_out_p = page(v_out_page_id);

    let cos_sin_byte = page_as_byte_ptr(cos_sin_p);

    let in_ready = page_ready_sem(in_p);
    let cos_sin_ready = page_ready_sem(cos_sin_p);
    let q_out_done = page_done_sem(q_out_p);
    let k_out_done = page_done_sem(k_out_p);
    let v_out_done = page_done_sem(v_out_p);
    let in_consumed = page_consumed_sem(in_p);
    let cos_sin_consumed = page_consumed_sem(cos_sin_p);
    let q_out_consumed = page_consumed_sem(q_out_p);
    let k_out_consumed = page_consumed_sem(k_out_p);
    let v_out_consumed = page_consumed_sem(v_out_p);

    let in_gmem = gmem_act_ptr_raw(in_act_slot);
    let q_out_gmem = gmem_act_ptr_raw(q_out_act_slot);
    let k_out_gmem = gmem_act_ptr_raw(k_out_act_slot);
    let v_out_gmem = gmem_act_ptr_raw(v_out_act_slot);
    let qkv_weight_gmem = gmem_weight_ptr_raw(qkv_weight_accessor, layer, NUM_LAYERS);
    let cos_sin_gmem = gmem_weight_ptr_raw(rotary_accessor, 0, NUM_LAYERS);
    let positions = gmem_positions();

    let act_bytes: u32 = M * HIDDEN_DIM * BF16_BYTES;
    let q_out_bytes: u32 = M * Q_DIM * BF16_BYTES;
    let k_out_bytes: u32 = M * KV_DIM * BF16_BYTES;
    let v_out_bytes: u32 = M * KV_DIM * BF16_BYTES;

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait::<1>(&in_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&cos_sin_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&q_out_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&k_out_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&v_out_consumed, loader_phase));

    loader.push(tk20::group_tma_expect_bytes::<1>(&in_ready, act_bytes));
    loader.push(CuStmt::new(format!(
        "{{ \
         __nv_bfloat16* __qd_a_dst = reinterpret_cast<__nv_bfloat16*>(ss.pages[{in_id}]); \
         kittens::group<1>::tma::load_async(\
         reinterpret_cast<void*>(__qd_a_dst), \
         reinterpret_cast<void*>({a_gmem}), \
         {act_bytes}, {a_ready}); \
         }}",
        in_id = in_page_id,
        a_gmem = in_gmem.expr(),
        act_bytes = act_bytes,
        a_ready = in_ready.expr(),
    )));

    loader.push(tk20::cos_sin_per_token_gather::<HEAD_DIM, M>(
        &cos_sin_byte,
        &cos_sin_gmem,
        &positions,
        &cos_sin_ready,
    ));

    let launcher = CuBlock::new();

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait::<1>(&in_ready, consumer_phase));
    consumer.push(tk20::group_wait::<1>(&cos_sin_ready, consumer_phase));

    // Phase 1 — per-thread vec-mat over QKV_N; stage Q/K results
    // to scratch (q_rope_offset / k_rope_offset, laid out [M, Q_DIM]
    // and [M, KV_DIM] row-major bf16), write V directly to
    // v_out_page (also [M, KV_DIM]).
    consumer.push(CuStmt::new(format!(
        "{{ \
         const __nv_bfloat16* __qd_a_in = reinterpret_cast<const __nv_bfloat16*>(\
             ss.pages[{in_id}]); \
         const __nv_bfloat16* __qd_qkv_gmem = {qkv_gmem}; \
         __nv_bfloat16* __qd_v_out = reinterpret_cast<__nv_bfloat16*>(\
             ss.pages[{v_id}]); \
         __nv_bfloat16* __qd_q_stage = reinterpret_cast<__nv_bfloat16*>(\
             ss.scratch + {q_off}); \
         __nv_bfloat16* __qd_k_stage = reinterpret_cast<__nv_bfloat16*>(\
             ss.scratch + {k_off}); \
         int __qd_warp_id = static_cast<int>(kittens::warpid()); \
         int __qd_lane = static_cast<int>(kittens::laneid()); \
         int __qd_warp_col_base = __qd_warp_id * {tile_n}; \
         constexpr int __qd_cols_per_thread = {tile_n} / 32; \
         for (int __qd_c = 0; __qd_c < __qd_cols_per_thread; __qd_c++) {{ \
             int __qd_n = __qd_warp_col_base + __qd_c * 32 + __qd_lane; \
             float __qd_acc[{m}]; \
             _Pragma(\"unroll\") \
             for (int __qd_m = 0; __qd_m < {m}; __qd_m++) __qd_acc[__qd_m] = 0.0f; \
             for (int __qd_k = 0; __qd_k < {hidden}; __qd_k++) {{ \
                 float __qd_w = __bfloat162float(\
                     __qd_qkv_gmem[__qd_k * {qkv_n} + __qd_n]); \
                 _Pragma(\"unroll\") \
                 for (int __qd_m = 0; __qd_m < {m}; __qd_m++) {{ \
                     float __qd_a = __bfloat162float(\
                         __qd_a_in[__qd_m * {hidden} + __qd_k]); \
                     __qd_acc[__qd_m] += __qd_a * __qd_w; \
                 }} \
             }} \
             if (__qd_n < {q_dim}) {{ \
                 _Pragma(\"unroll\") \
                 for (int __qd_m = 0; __qd_m < {m}; __qd_m++) {{ \
                     __qd_q_stage[__qd_m * {q_dim} + __qd_n] = \
                         __float2bfloat16(__qd_acc[__qd_m]); \
                 }} \
             }} else if (__qd_n < {q_dim} + {kv_dim}) {{ \
                 int __qd_kn = __qd_n - {q_dim}; \
                 _Pragma(\"unroll\") \
                 for (int __qd_m = 0; __qd_m < {m}; __qd_m++) {{ \
                     __qd_k_stage[__qd_m * {kv_dim} + __qd_kn] = \
                         __float2bfloat16(__qd_acc[__qd_m]); \
                 }} \
             }} else {{ \
                 int __qd_vn = __qd_n - {q_dim} - {kv_dim}; \
                 _Pragma(\"unroll\") \
                 for (int __qd_m = 0; __qd_m < {m}; __qd_m++) {{ \
                     __qd_v_out[__qd_m * {kv_dim} + __qd_vn] = \
                         __float2bfloat16(__qd_acc[__qd_m]); \
                 }} \
             }} \
         }} \
         __syncwarp(); \
         }}",
        in_id = in_page_id,
        v_id = v_out_page_id,
        qkv_gmem = qkv_weight_gmem.expr(),
        q_off = q_rope_offset,
        k_off = k_rope_offset,
        tile_n = TILE_N,
        m = M,
        hidden = HIDDEN_DIM,
        qkv_n = QKV_N,
        q_dim = Q_DIM,
        kv_dim = KV_DIM,
    )));

    // Phase 2 — re-read paired col from scratch per row, apply
    // per-row RoPE (cos_sin_page row m), store to q_out / k_out.
    consumer.push(CuStmt::new(format!(
        "{{ \
         const __nv_bfloat16* __qd_q_stage = reinterpret_cast<const __nv_bfloat16*>(\
             ss.scratch + {q_off}); \
         const __nv_bfloat16* __qd_k_stage = reinterpret_cast<const __nv_bfloat16*>(\
             ss.scratch + {k_off}); \
         const __nv_bfloat16* __qd_cos_sin = reinterpret_cast<const __nv_bfloat16*>(\
             ss.pages[{cs_id}]); \
         __nv_bfloat16* __qd_q_out = reinterpret_cast<__nv_bfloat16*>(\
             ss.pages[{q_id}]); \
         __nv_bfloat16* __qd_k_out = reinterpret_cast<__nv_bfloat16*>(\
             ss.pages[{k_id}]); \
         int __qd_warp_id = static_cast<int>(kittens::warpid()); \
         int __qd_lane = static_cast<int>(kittens::laneid()); \
         int __qd_warp_col_base = __qd_warp_id * {tile_n}; \
         constexpr int __qd_cols_per_thread = {tile_n} / 32; \
         constexpr int __qd_half = {head_dim} / 2; \
         for (int __qd_c = 0; __qd_c < __qd_cols_per_thread; __qd_c++) {{ \
             int __qd_n = __qd_warp_col_base + __qd_c * 32 + __qd_lane; \
             if (__qd_n < {q_dim}) {{ \
                 int __qd_pos = __qd_n % {head_dim}; \
                 int __qd_head_base = __qd_n - __qd_pos; \
                 bool __qd_low = __qd_pos < __qd_half; \
                 int __qd_paired = __qd_low ? __qd_pos + __qd_half : __qd_pos - __qd_half; \
                 int __qd_t = __qd_low ? __qd_pos : __qd_pos - __qd_half; \
                 _Pragma(\"unroll\") \
                 for (int __qd_m = 0; __qd_m < {m}; __qd_m++) {{ \
                     float __qd_x = __bfloat162float(\
                         __qd_q_stage[__qd_m * {q_dim} + __qd_n]); \
                     float __qd_p = __bfloat162float(\
                         __qd_q_stage[__qd_m * {q_dim} + __qd_head_base + __qd_paired]); \
                     float __qd_co = __bfloat162float(\
                         __qd_cos_sin[__qd_m * {head_dim} + __qd_t]); \
                     float __qd_si = __bfloat162float(\
                         __qd_cos_sin[__qd_m * {head_dim} + __qd_half + __qd_t]); \
                     float __qd_r = __qd_low \
                         ? (__qd_x * __qd_co - __qd_p * __qd_si) \
                         : (__qd_x * __qd_co + __qd_p * __qd_si); \
                     __qd_q_out[__qd_m * {q_dim} + __qd_n] = __float2bfloat16(__qd_r); \
                 }} \
             }} else if (__qd_n < {q_dim} + {kv_dim}) {{ \
                 int __qd_kn = __qd_n - {q_dim}; \
                 int __qd_pos = __qd_kn % {head_dim}; \
                 int __qd_head_base = __qd_kn - __qd_pos; \
                 bool __qd_low = __qd_pos < __qd_half; \
                 int __qd_paired = __qd_low ? __qd_pos + __qd_half : __qd_pos - __qd_half; \
                 int __qd_t = __qd_low ? __qd_pos : __qd_pos - __qd_half; \
                 _Pragma(\"unroll\") \
                 for (int __qd_m = 0; __qd_m < {m}; __qd_m++) {{ \
                     float __qd_x = __bfloat162float(\
                         __qd_k_stage[__qd_m * {kv_dim} + __qd_kn]); \
                     float __qd_p = __bfloat162float(\
                         __qd_k_stage[__qd_m * {kv_dim} + __qd_head_base + __qd_paired]); \
                     float __qd_co = __bfloat162float(\
                         __qd_cos_sin[__qd_m * {head_dim} + __qd_t]); \
                     float __qd_si = __bfloat162float(\
                         __qd_cos_sin[__qd_m * {head_dim} + __qd_half + __qd_t]); \
                     float __qd_r = __qd_low \
                         ? (__qd_x * __qd_co - __qd_p * __qd_si) \
                         : (__qd_x * __qd_co + __qd_p * __qd_si); \
                     __qd_k_out[__qd_m * {kv_dim} + __qd_kn] = __float2bfloat16(__qd_r); \
                 }} \
             }} \
         }} \
         }}",
        q_off = q_rope_offset,
        k_off = k_rope_offset,
        cs_id = cos_sin_page_id,
        q_id = q_out_page_id,
        k_id = k_out_page_id,
        tile_n = TILE_N,
        m = M,
        head_dim = HEAD_DIM,
        q_dim = Q_DIM,
        kv_dim = KV_DIM,
    )));

    consumer.push(tk20::group_sync_named::<NCW>(bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive::<1>(&q_out_done),
        tk20::group_arrive::<1>(&k_out_done),
        tk20::group_arrive::<1>(&v_out_done),
        tk20::group_arrive::<1>(&in_consumed),
        tk20::group_arrive::<1>(&cos_sin_consumed),
    ]));

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&q_out_done, storer_phase));
    storer.push(tk20::group_wait::<1>(&k_out_done, storer_phase));
    storer.push(tk20::group_wait::<1>(&v_out_done, storer_phase));
    storer.push(CuStmt::new(format!(
        "{{ \
         __nv_bfloat16* __qd_q_src = reinterpret_cast<__nv_bfloat16*>(ss.pages[{q_id}]); \
         kittens::group<1>::tma::store_async(\
             reinterpret_cast<void*>({q_gmem}), \
             reinterpret_cast<void*>(__qd_q_src), \
             {bytes}); \
         }}",
        q_id = q_out_page_id,
        q_gmem = q_out_gmem.expr(),
        bytes = q_out_bytes,
    )));
    storer.push(CuStmt::new(format!(
        "{{ \
         __nv_bfloat16* __qd_k_src = reinterpret_cast<__nv_bfloat16*>(ss.pages[{k_id}]); \
         kittens::group<1>::tma::store_async(\
             reinterpret_cast<void*>({k_gmem}), \
             reinterpret_cast<void*>(__qd_k_src), \
             {bytes}); \
         }}",
        k_id = k_out_page_id,
        k_gmem = k_out_gmem.expr(),
        bytes = k_out_bytes,
    )));
    storer.push(CuStmt::new(format!(
        "{{ \
         __nv_bfloat16* __qd_v_src = reinterpret_cast<__nv_bfloat16*>(ss.pages[{v_id}]); \
         kittens::group<1>::tma::store_async(\
             reinterpret_cast<void*>({v_gmem}), \
             reinterpret_cast<void*>(__qd_v_src), \
             {bytes}); \
         }}",
        v_id = v_out_page_id,
        v_gmem = v_out_gmem.expr(),
        bytes = v_out_bytes,
    )));
    storer.push(tk20::group_tma_store_async_wait::<1>());
    storer.push(tk20::group_arrive::<1>(&q_out_consumed));
    storer.push(tk20::group_arrive::<1>(&k_out_consumed));
    storer.push(tk20::group_arrive::<1>(&v_out_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}

// ============================================================
// FusedQkvRopeCache — fused QKV linear projection + RoPE rotation.
//
// The per-head body is a runtime for-loop with an inner if-else
// routing each head to Q rope staging, K rope staging, or V
// output. The loop scaffolding (for, if-else) goes through
// `tk20::for_loop` / `tk20::if_else` per the dogfood-tk20 rule;
// the leaf statements that need runtime-shape subtile coords
// (e.g. `subtile<>(int2{0, col / head_dim})`) and a `__device__`
// lambda for the RoPE rotation are emitted as scoped
// `CuStmt::new(format!())` inside the CuBlocks (no const-generic
// `tk20::*` binding can express a runtime subtile offset).
// ============================================================

pub fn render_fused_qkv_rope_cache<
    const M: u32,
    const HIDDEN_DIM: u32,
    const HEAD_DIM: u32,
    const NUM_Q_HEADS: u32,
    const NUM_KV_HEADS: u32,
    const Q_DIM: u32,
    const KV_DIM: u32,
    const QKV_N: u32,
    const TILE_N: u32,
    const HEADS_PER_WARP: u32,
    const NCW: u32,
    const NUM_LAYERS: u32,
    const ITERS: u32,
>(
    in_page_id: u32,
    qkv_weight_page_id: u32,
    cos_sin_page_id: u32,
    q_out_page_id: u32,
    k_out_page_id: u32,
    v_out_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    in_act_slot: u32,
    q_out_act_slot: u32,
    k_out_act_slot: u32,
    v_out_act_slot: u32,
    qkv_weight_accessor: u32,
    rotary_accessor: u32,
    bar_publish: u32,
    q_rope_offset: u32,
    k_rope_offset: u32,
    qkv_b_tile_offset: u32,
) -> RoleBodies {
    if ITERS != 1 {
        return RoleBodies::skipped("TkFusedQkvRopeCache");
    }
    if HEAD_DIM == 0 || TILE_N % HEAD_DIM != 0 {
        return RoleBodies::skipped("TkFusedQkvRopeCache");
    }
    // M%16!=0 → vec-mat decode-shape (small-batch decode).
    // M%16==0 → tile-MMA prefill-shape, dispatch onto fixed
    // M_BLOCKS so wgmma const-asserts only fire at those instantiations.
    if M % 16 != 0 {
        return render_fused_qkv_rope_cache_decode::<
            M, HIDDEN_DIM, HEAD_DIM, Q_DIM, KV_DIM, QKV_N, TILE_N, NCW, NUM_LAYERS, ITERS,
        >(
            in_page_id,
            qkv_weight_page_id,
            cos_sin_page_id,
            q_out_page_id,
            k_out_page_id,
            v_out_page_id,
            consumer_phase,
            storer_phase,
            layer,
            in_act_slot,
            q_out_act_slot,
            k_out_act_slot,
            v_out_act_slot,
            qkv_weight_accessor,
            rotary_accessor,
            bar_publish,
            q_rope_offset,
            k_rope_offset,
            qkv_b_tile_offset,
        );
    }
    if M == 64 {
        return render_fused_qkv_rope_cache_prefill_wgmma::<
            1, 64, HIDDEN_DIM, HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS, Q_DIM, KV_DIM, QKV_N,
            TILE_N, HEADS_PER_WARP, NCW, NUM_LAYERS, ITERS,
        >(
            in_page_id, qkv_weight_page_id, cos_sin_page_id,
            q_out_page_id, k_out_page_id, v_out_page_id,
            consumer_phase, storer_phase, layer,
            in_act_slot, q_out_act_slot, k_out_act_slot, v_out_act_slot,
            qkv_weight_accessor, rotary_accessor, bar_publish,
            q_rope_offset, k_rope_offset, qkv_b_tile_offset,
        );
    } else if M == 512 {
        return render_fused_qkv_rope_cache_prefill_wgmma::<
            8, 512, HIDDEN_DIM, HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS, Q_DIM, KV_DIM, QKV_N,
            TILE_N, HEADS_PER_WARP, NCW, NUM_LAYERS, ITERS,
        >(
            in_page_id, qkv_weight_page_id, cos_sin_page_id,
            q_out_page_id, k_out_page_id, v_out_page_id,
            consumer_phase, storer_phase, layer,
            in_act_slot, q_out_act_slot, k_out_act_slot, v_out_act_slot,
            qkv_weight_accessor, rotary_accessor, bar_publish,
            q_rope_offset, k_rope_offset, qkv_b_tile_offset,
        );
    } else if M == 4096 {
        return render_fused_qkv_rope_cache_prefill_wgmma::<
            64, 4096, HIDDEN_DIM, HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS, Q_DIM, KV_DIM, QKV_N,
            TILE_N, HEADS_PER_WARP, NCW, NUM_LAYERS, ITERS,
        >(
            in_page_id, qkv_weight_page_id, cos_sin_page_id,
            q_out_page_id, k_out_page_id, v_out_page_id,
            consumer_phase, storer_phase, layer,
            in_act_slot, q_out_act_slot, k_out_act_slot, v_out_act_slot,
            qkv_weight_accessor, rotary_accessor, bar_publish,
            q_rope_offset, k_rope_offset, qkv_b_tile_offset,
        );
    } else {
        panic!("render_fused_qkv_rope_cache prefill: unsupported M (must be 64/512/4096); got {M}");
    }
}

// ============================================================
// FusedQkvRopeCache (prefill — TK 2.0 H100 wgmma SMEM+SMEM).
//
// Outer m-block loop wrapping a per-warpgroup, per-head inner
// loop. Each (m_block, head) iteration:
//   1. Computes the global N column for this head: warpgroup_id
//      contributes the warpgroup's column block (HEADS_PER_WG * HEAD_DIM
//      cols), inner head index __qkv_h adds HEAD_DIM cols.
//   2. Subtiles A=st_bf<64, HIDDEN_DIM> at (m_block, 0) and
//      B=st_bf<HIDDEN_DIM, HEAD_DIM> at (0, __qkv_col / HEAD_DIM).
//   3. Zeros warpgroup-distributed acc rt_fl<16, HEAD_DIM>, mma_AB
//      D = A*B, mma_async_wait.
//   4. Routes by __qkv_col to one of three branches:
//        Q (__qkv_col < Q_DIM): warpgroup::store acc → stg subtile
//          of __qkv_q_rope; warpgroup::apply RoPE lambda on acc;
//          warpgroup::store acc → out subtile of q_out_smem.
//        K (Q_DIM ≤ __qkv_col < Q_DIM+KV_DIM): same RoPE flow into
//          __qkv_k_rope and k_out_smem.
//        V (otherwise): warpgroup::store acc → out subtile of
//          v_out_smem (no RoPE).
// ============================================================

#[allow(clippy::too_many_arguments)]
pub fn render_fused_qkv_rope_cache_prefill_wgmma<
    const M_BLOCKS: u32,
    const M_TOTAL: u32,
    const HIDDEN_DIM: u32,
    const HEAD_DIM: u32,
    const NUM_Q_HEADS: u32,
    const NUM_KV_HEADS: u32,
    const Q_DIM: u32,
    const KV_DIM: u32,
    const QKV_N: u32,
    const TILE_N: u32,
    const HEADS_PER_WARP: u32,
    const NCW: u32,
    const NUM_LAYERS: u32,
    const ITERS: u32,
>(
    in_page_id: u32,
    qkv_weight_page_id: u32,
    cos_sin_page_id: u32,
    q_out_page_id: u32,
    k_out_page_id: u32,
    v_out_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    in_act_slot: u32,
    q_out_act_slot: u32,
    k_out_act_slot: u32,
    v_out_act_slot: u32,
    qkv_weight_accessor: u32,
    rotary_accessor: u32,
    bar_publish: u32,
    q_rope_offset: u32,
    k_rope_offset: u32,
    qkv_b_tile_offset: u32,
) -> RoleBodies {
    const { assert!(NCW % 4 == 0, "render_fused_qkv_rope_cache prefill: NCW must be a multiple of 4 (warpgroup size)"); }
    const { assert!(M_BLOCKS >= 1, "render_fused_qkv_rope_cache prefill: M_BLOCKS must be >= 1"); }
    const { assert!(M_TOTAL == M_BLOCKS * 64, "render_fused_qkv_rope_cache prefill: M_TOTAL must equal M_BLOCKS * 64"); }

    const M_TILE: u32 = 64;
    let num_warpgroups: u32 = NCW / 4;
    let heads_per_wg: u32 = HEADS_PER_WARP * 4;

    let loader_phase = storer_phase;
    let in_p = page(in_page_id);
    let qkv_weight_p = page(qkv_weight_page_id);
    let cos_sin_p = page(cos_sin_page_id);
    let q_out_p = page(q_out_page_id);
    let k_out_p = page(k_out_page_id);
    let v_out_p = page(v_out_page_id);

    let in_smem = page_as_st_bf::<M_TOTAL, HIDDEN_DIM>(in_p);
    let q_out_smem = page_as_st_bf::<M_TOTAL, Q_DIM>(q_out_p);
    let k_out_smem = page_as_st_bf::<M_TOTAL, KV_DIM>(k_out_p);
    let v_out_smem = page_as_st_bf::<M_TOTAL, KV_DIM>(v_out_p);
    let qkv_b_tile = scratch_as_st_bf::<HIDDEN_DIM, QKV_N>(off(qkv_b_tile_offset));
    let cos_sin_byte = page_as_byte_ptr(cos_sin_p);

    let in_ready = page_ready_sem(in_p);
    let qkv_weight_ready = page_ready_sem(qkv_weight_p);
    let cos_sin_ready = page_ready_sem(cos_sin_p);
    let q_out_done = page_done_sem(q_out_p);
    let k_out_done = page_done_sem(k_out_p);
    let v_out_done = page_done_sem(v_out_p);
    let in_consumed = page_consumed_sem(in_p);
    let qkv_weight_consumed = page_consumed_sem(qkv_weight_p);
    let cos_sin_consumed = page_consumed_sem(cos_sin_p);
    let q_out_consumed = page_consumed_sem(q_out_p);
    let k_out_consumed = page_consumed_sem(k_out_p);
    let v_out_consumed = page_consumed_sem(v_out_p);

    let in_gmem = gmem_act_ptr_raw(in_act_slot);
    let q_out_gmem = gmem_act_ptr_raw(q_out_act_slot);
    let k_out_gmem = gmem_act_ptr_raw(k_out_act_slot);
    let v_out_gmem = gmem_act_ptr_raw(v_out_act_slot);
    let qkv_weight_gmem = gmem_weight_ptr_raw(qkv_weight_accessor, layer, NUM_LAYERS);
    let cos_sin_gmem = gmem_weight_ptr_raw(rotary_accessor, 0, NUM_LAYERS);
    let positions = gmem_positions();

    let act_bytes = M_TOTAL * HIDDEN_DIM * BF16_BYTES;
    let qkv_weight_bytes = HIDDEN_DIM * QKV_N * BF16_BYTES;
    let q_out_bytes = M_TOTAL * Q_DIM * BF16_BYTES;
    let k_out_bytes = M_TOTAL * KV_DIM * BF16_BYTES;
    let v_out_bytes = M_TOTAL * KV_DIM * BF16_BYTES;

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait::<1>(&in_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&qkv_weight_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&cos_sin_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&q_out_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&k_out_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&v_out_consumed, loader_phase));

    loader.push(tk20::group_tma_expect_bytes::<1>(&in_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf::<1, M_TOTAL, HIDDEN_DIM>(
        &in_smem, &in_gmem, act_bytes, &in_ready,
    ));

    loader.push(tk20::group_tma_expect_bytes::<1>(&qkv_weight_ready, qkv_weight_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf::<1, HIDDEN_DIM, QKV_N>(
        &qkv_b_tile, &qkv_weight_gmem, qkv_weight_bytes, &qkv_weight_ready,
    ));

    loader.push(tk20::cos_sin_per_token_gather::<HEAD_DIM, M_TOTAL>(
        &cos_sin_byte,
        &cos_sin_gmem,
        &positions,
        &cos_sin_ready,
    ));

    let launcher = CuBlock::new();

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait::<1>(&in_ready, consumer_phase));
    consumer.push(tk20::group_wait::<1>(&qkv_weight_ready, consumer_phase));
    consumer.push(tk20::group_wait::<1>(&cos_sin_ready, consumer_phase));

    // Pre-loop decls: scratch ST refs for Q/K rope staging (sized
    // M_TOTAL × Q_DIM / KV_DIM — over-committed at M_TOTAL > 64,
    // matching the page over-commit pattern; prefill bodies are not
    // launched at runtime in the persistent-decode substrate, so the
    // overcommit exists only for nvcc-compile clean substrate proof),
    // raw bf16 ptr for cos_sin gather, warpgroup id, heads_per_wg,
    // and the unroll pragma for the per-head loop.
    consumer.push(CuStmt::new(format!(
        "auto& __qkv_q_rope = *reinterpret_cast<kittens::st_bf<{m_total}, {q_dim}>*>(\
         ss.scratch + {q_rope_offset});",
        m_total = M_TOTAL,
        q_dim = Q_DIM,
    )));
    consumer.push(CuStmt::new(format!(
        "auto& __qkv_k_rope = *reinterpret_cast<kittens::st_bf<{m_total}, {kv_dim}>*>(\
         ss.scratch + {k_rope_offset});",
        m_total = M_TOTAL,
        kv_dim = KV_DIM,
    )));
    consumer.push(CuStmt::new(format!(
        "__nv_bfloat16* __qkv_cos_sin_ptr = reinterpret_cast<__nv_bfloat16*>(\
         ss.pages[{cos_sin_id}]);",
        cos_sin_id = cos_sin_page_id,
    )));
    consumer.push(CuStmt::new(format!(
        "const int __qkv_wg_id = {gid};",
        gid = tk20::warpgroup_groupid_expr(),
    )));

    let (decl_acc, acc_rt) = tk20::decl_rt_fl_warpgroup::<HEAD_DIM>("__qkv_acc");
    // __qkv_rot is referenced by name in the raw CuStmt RoPE apply;
    // the typed handle exists only to materialize the decl statement.
    let (decl_rot, _rot_rt) = tk20::decl_rt_fl_warpgroup::<HEAD_DIM>("__qkv_rot");
    consumer.push(decl_acc);
    consumer.push(decl_rot);

    // Per-head loop body (one wgmma per head). Each warpgroup
    // iterates `heads_per_wg = HEADS_PER_WARP * 4` heads, with col
    // base `__qkv_wg_id * heads_per_wg * HEAD_DIM`.
    let mut head_loop_body = CuBlock::new();
    head_loop_body.push(CuStmt::new(format!(
        "const int __qkv_col = __qkv_wg_id * {heads_per_wg} * {head_dim} + __qkv_h * {head_dim};",
        heads_per_wg = heads_per_wg,
        head_dim = HEAD_DIM,
    )));

    let (decl_a_sub, a_sub) = tk20::decl_st_bf_subtile::<M_TOTAL, HIDDEN_DIM, M_TILE, HIDDEN_DIM>(
        "__qkv_a_sub", &in_smem, "__qkv_m_block", "0",
    );
    head_loop_body.push(decl_a_sub);
    // B subtile coords are runtime (per-head); emit the subtile
    // bind as a raw CuStmt and re-bind a typed handle for the
    // wgmma call. The decl_st_bf_subtile helper assumes a fixed
    // const-generic outer layout, but here the outer ST is
    // `qkv_b_tile = st_bf<HIDDEN_DIM, QKV_N>` and we want a
    // `st_bf<HIDDEN_DIM, HEAD_DIM>` subtile at column block
    // `__qkv_col / HEAD_DIM` (runtime).
    head_loop_body.push(CuStmt::new(format!(
        "auto __qkv_b_sub = ({qkv_b_tile})\
         .template subtile<{hidden_dim}, {head_dim}>(\
         int2{{0, __qkv_col / {head_dim}}});",
        qkv_b_tile = qkv_b_tile.expr(),
        hidden_dim = HIDDEN_DIM,
        head_dim = HEAD_DIM,
    )));
    let b_sub_typed = St::<Bf16, HIDDEN_DIM, HEAD_DIM>::from_expr(CuExpr::new("__qkv_b_sub".to_string()));

    head_loop_body.push(tk20::warpgroup_zero_rt_fl::<HEAD_DIM>(&acc_rt));
    head_loop_body.push(tk20::warpgroup_mma_AB::<M_TILE, HIDDEN_DIM, HEAD_DIM>(
        &acc_rt, &a_sub, &b_sub_typed,
    ));
    head_loop_body.push(tk20::warpgroup_mma_async_wait());

    // RoPE branch builder: stages acc into scratch (Q_rope or
    // K_rope), syncs the warpgroup, applies the RoPE lambda over
    // the warpgroup-distributed acc, then stores into the output
    // smem subtile.
    let make_rope_branch = |stg_ref: &str, out_smem_expr: &str, local_expr: &str| {
        let mut block = CuBlock::new();
        block.push(CuStmt::new(format!(
            "const int __qkv_local = {local_expr};"
        )));
        block.push(CuStmt::new(format!(
            "auto __qkv_stg = {stg_ref}\
             .template subtile<{m_tile}, {head_dim}>(\
             int2{{__qkv_m_block, __qkv_local / {head_dim}}});",
            m_tile = M_TILE,
            head_dim = HEAD_DIM,
        )));
        block.push(CuStmt::new(
            "kittens::warpgroup::store(__qkv_stg, __qkv_acc);".to_string(),
        ));
        // group sync (warpgroup) — all 4 warps must finish writing
        // their 16-row slice of stg before any thread reads
        // paired-half values back.
        block.push(CuStmt::new(
            "asm volatile(\"bar.sync 1, 128;\" ::: \"memory\");".to_string(),
        ));
        block.push(CuStmt::new(
            "__nv_bfloat16* __qkv_stg_ptr = reinterpret_cast<__nv_bfloat16*>(&__qkv_stg);"
                .to_string(),
        ));
        block.push(CuStmt::new(format!(
            "kittens::warpgroup::apply(__qkv_rot, __qkv_acc, [=] __device__ \
             (int row, int col, float x) {{\n\
             \x20   constexpr int __half = {head_dim} / 2;\n\
             \x20   int __pc = col < __half ? col + __half : col - __half;\n\
             \x20   float __paired = __bfloat162float(\
             __qkv_stg_ptr[row * {head_dim} + __pc]);\n\
             \x20   int __t = col < __half ? col : col - __half;\n\
             \x20   float __c = __bfloat162float(\
             __qkv_cos_sin_ptr[row * {head_dim} + __t]);\n\
             \x20   float __s = __bfloat162float(\
             __qkv_cos_sin_ptr[row * {head_dim} + __half + __t]);\n\
             \x20   return col < __half ? (x * __c - __paired * __s) \
             : (x * __c + __paired * __s);\n\
             }});",
            head_dim = HEAD_DIM,
        )));
        block.push(CuStmt::new(format!(
            "auto __qkv_out = ({out_smem_expr})\
             .template subtile<{m_tile}, {head_dim}>(\
             int2{{__qkv_m_block, __qkv_local / {head_dim}}});",
            m_tile = M_TILE,
            head_dim = HEAD_DIM,
        )));
        block.push(CuStmt::new(
            "kittens::warpgroup::store(__qkv_out, __qkv_rot);".to_string(),
        ));
        block
    };

    let q_branch = make_rope_branch("__qkv_q_rope", q_out_smem.expr().as_str(), "__qkv_col");
    let k_branch = make_rope_branch(
        "__qkv_k_rope",
        k_out_smem.expr().as_str(),
        &format!("__qkv_col - {}", Q_DIM),
    );

    // V passthrough branch: store acc directly to v_out subtile (no RoPE).
    let mut v_branch = CuBlock::new();
    v_branch.push(CuStmt::new(format!(
        "const int __qkv_local = __qkv_col - {q_dim} - {kv_dim};",
        q_dim = Q_DIM,
        kv_dim = KV_DIM,
    )));
    v_branch.push(CuStmt::new(format!(
        "auto __qkv_out = ({v_out_smem})\
         .template subtile<{m_tile}, {head_dim}>(\
         int2{{__qkv_m_block, __qkv_local / {head_dim}}});",
        v_out_smem = v_out_smem.expr(),
        m_tile = M_TILE,
        head_dim = HEAD_DIM,
    )));
    v_branch.push(CuStmt::new(
        "kittens::warpgroup::store(__qkv_out, __qkv_acc);".to_string(),
    ));

    let q_cond = format!("__qkv_col < {}", Q_DIM);
    let k_cond = format!("__qkv_col < {} + {}", Q_DIM, KV_DIM);
    head_loop_body.push(tk20::if_chain(
        &[(&q_cond, &q_branch), (&k_cond, &k_branch)],
        Some(&v_branch),
    ));

    let mut m_loop_body = CuBlock::new();
    m_loop_body.push(tk20::for_loop(
        &format!(
            "int __qkv_h = 0; __qkv_h < {heads_per_wg}; ++__qkv_h",
            heads_per_wg = heads_per_wg,
        ),
        &head_loop_body,
    ));

    consumer.push(tk20::for_loop_no_unroll(
        &format!("int __qkv_m_block = 0; __qkv_m_block < {M_BLOCKS}; ++__qkv_m_block"),
        &m_loop_body,
    ));

    consumer.push(tk20::group_sync_named::<NCW>(bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive::<1>(&q_out_done),
        tk20::group_arrive::<1>(&k_out_done),
        tk20::group_arrive::<1>(&v_out_done),
        tk20::group_arrive::<1>(&in_consumed),
        tk20::group_arrive::<1>(&qkv_weight_consumed),
        tk20::group_arrive::<1>(&cos_sin_consumed),
    ]));
    let _ = num_warpgroups;

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&q_out_done, storer_phase));
    storer.push(tk20::group_wait::<1>(&k_out_done, storer_phase));
    storer.push(tk20::group_wait::<1>(&v_out_done, storer_phase));

    storer.push(tk20::group_tma_store_async_raw_st_bf::<1, M_TOTAL, Q_DIM>(
        &q_out_gmem, &q_out_smem, q_out_bytes,
    ));
    storer.push(tk20::group_tma_store_async_raw_st_bf::<1, M_TOTAL, KV_DIM>(
        &k_out_gmem, &k_out_smem, k_out_bytes,
    ));
    storer.push(tk20::group_tma_store_async_raw_st_bf::<1, M_TOTAL, KV_DIM>(
        &v_out_gmem, &v_out_smem, v_out_bytes,
    ));
    storer.push(tk20::group_tma_store_async_wait::<1>());
    storer.push(tk20::group_arrive::<1>(&q_out_consumed));
    storer.push(tk20::group_arrive::<1>(&k_out_consumed));
    storer.push(tk20::group_arrive::<1>(&v_out_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}

// ============================================================
// AttentionViaCache — paged FlashAttention-2 over the global KV
// cache. Const generics propagate every shape and lifecycle proof
// from `AttentionViaCacheNode` end-to-end:
//
//   M           = NUM_TOKENS   (Q rows; vLLM's per-request token count)
//   HEAD_DIM    = per-head dim (e.g. 64 for llama-3.2-1b)
//   NUM_Q_HEADS / NUM_KV_HEADS / KV_DIM = NUM_KV_HEADS * HEAD_DIM
//   BLOCK_SIZE  = paged-KV block size (16, fixed by Attn launch tier)
//   NCW         = NUM_CONSUMER_WARPS (warp split of Q heads)
//   NUM_LAYERS  = total transformer layers (for kv_cache_ptrs[layer])
//
// Scratch layout (proven disjoint at proc-macro time via four-way
// `ScratchRegion::disjoint_with` chain in
// [`crate::ir::lower::push_attention_via_cache`]):
//   `score`    `[SCORE_OFF,  SCORE_OFF + SCORE_BYTES)`     — softmax stats
//   `pv`       `[PV_OFF,     PV_OFF    + PV_BYTES)`        — partial PV accum
//   `k_smem`   `[K_SMEM_OFF, K_SMEM_OFF + K_SMEM_BYTES)`   — single-stage K block
//   `v_smem`   `[V_SMEM_OFF, V_SMEM_OFF + V_SMEM_BYTES)`   — single-stage V block
//
// The K_SMEM / V_SMEM regions are sized to fit one `[BLOCK_SIZE,
// NUM_KV_HEADS * HEAD_DIM]` bf16 paged-KV block each (proof
// discharged via `ScratchRegion::fits_kv_block` in the push path).
//
// **STATUS**: this fn is a *stub*. The AST-level wiring is complete —
// IR fields, push-side substrate proofs, proc-macro dispatch, render
// dispatch arm — but the role-body composition that emits the full
// FlashAttention algorithm (Q@K^T, online softmax, PV) is the next
// named phase. Returning `RoleBodies::skipped` emits a `// SKIPPED`
// marker per role and lets `emit_for_canonical_<canonical>` succeed
// for tapes containing this node, instead of skipping the entire
// canonical (which is the pre-S16 behavior).
// ============================================================

#[allow(clippy::too_many_arguments, clippy::cognitive_complexity)]
pub fn render_attention_via_cache<
    const M: u32,
    const HEAD_DIM: u32,
    const NUM_Q_HEADS: u32,
    const NUM_KV_HEADS: u32,
    const BLOCK_SIZE: u32,
    const MAX_SK: u32,
    const NCW: u32,
    const NUM_LAYERS: u32,
    const ITERS: u32,
>(
    q_in_page_id: u32,
    attn_out_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    q_in_act_slot: u32,
    attn_out_act_slot: u32,
    _score_offset: u32,
    _pv_offset: u32,
    k_smem_offset: u32,
    v_smem_offset: u32,
    attn_scale: f32,
    attn_softcap: f32,
    _interleaved: bool,
) -> RoleBodies {
    // M=1 → decode shape (heads-packed Q + per-warp staging +
    //                     cross-warp compact, log-space softmax).
    // M=16 → prefill shape (per-Q-head outer loop).
    // Other M → SKIPPED until the corresponding shape lands.
    if M == 1 {
        render_attention_via_cache_decode::<
            HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS, BLOCK_SIZE, MAX_SK, NCW, NUM_LAYERS, ITERS,
        >(
            q_in_page_id,
            attn_out_page_id,
            consumer_phase,
            storer_phase,
            layer,
            q_in_act_slot,
            attn_out_act_slot,
            k_smem_offset,
            v_smem_offset,
            attn_scale,
            attn_softcap,
            /*sliding_window=*/ None,
        )
    } else {
        render_attention_via_cache_impl::<
            M, HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS, BLOCK_SIZE, MAX_SK, NCW, NUM_LAYERS, ITERS,
        >(
            q_in_page_id,
            attn_out_page_id,
            consumer_phase,
            storer_phase,
            layer,
            q_in_act_slot,
            attn_out_act_slot,
            k_smem_offset,
            v_smem_offset,
            attn_scale,
            attn_softcap,
            /*sliding_window=*/ None,
        )
    }
}

// ============================================================
// SlidingAttentionViaCache — same algorithm as
// `AttentionViaCache` plus a runtime sliding-window mask
// (`if (kv_token_pos < q_token_pos - SLIDING_WINDOW) continue;`
// inside the per-block consumer loop).
//
// Same const-generic surface as `render_attention_via_cache` plus
// the runtime `sliding_window` arg. STUB — same partial-landing
// rationale documented above.
// ============================================================

#[allow(clippy::too_many_arguments)]
pub fn render_sliding_attention_via_cache<
    const M: u32,
    const HEAD_DIM: u32,
    const NUM_Q_HEADS: u32,
    const NUM_KV_HEADS: u32,
    const BLOCK_SIZE: u32,
    const MAX_SK: u32,
    const NCW: u32,
    const NUM_LAYERS: u32,
    const ITERS: u32,
>(
    q_in_page_id: u32,
    attn_out_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    q_in_act_slot: u32,
    attn_out_act_slot: u32,
    _score_offset: u32,
    _pv_offset: u32,
    k_smem_offset: u32,
    v_smem_offset: u32,
    attn_scale: f32,
    attn_softcap: f32,
    _interleaved: bool,
    sliding_window: u32,
) -> RoleBodies {
    if M == 1 {
        render_attention_via_cache_decode::<
            HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS, BLOCK_SIZE, MAX_SK, NCW, NUM_LAYERS, ITERS,
        >(
            q_in_page_id,
            attn_out_page_id,
            consumer_phase,
            storer_phase,
            layer,
            q_in_act_slot,
            attn_out_act_slot,
            k_smem_offset,
            v_smem_offset,
            attn_scale,
            attn_softcap,
            Some(sliding_window),
        )
    } else {
        render_attention_via_cache_impl::<
            M, HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS, BLOCK_SIZE, MAX_SK, NCW, NUM_LAYERS, ITERS,
        >(
            q_in_page_id,
            attn_out_page_id,
            consumer_phase,
            storer_phase,
            layer,
            q_in_act_slot,
            attn_out_act_slot,
            k_smem_offset,
            v_smem_offset,
            attn_scale,
            attn_softcap,
            Some(sliding_window),
        )
    }
}

// ============================================================
// FlashAttention-2 algorithm body. Shared between
// `render_attention_via_cache` and
// `render_sliding_attention_via_cache` — the sliding variant
// passes `sliding_window = Some(w)` to splice an extra
// `if (kv_pos < q_pos - w) continue;` predicate inside the
// per-block consumer loop.
//
// Algorithm (per consumer warp, owns NUM_Q_HEADS / NCW Q-heads):
//   load Q register tile from q_in_page subtile for owned q_head
//   max_vec ← -INF, sum_vec ← 0, o_reg ← 0
//   for p in 0..num_blocks:
//       // Issue paged-KV TMA load (warp 0 only)
//       wait k_arrived[p & 1]
//       load K rt from k_smem[kv_head * HEAD_DIM .. + HEAD_DIM]
//       att_block ← Q @ K^T
//       att_block ← att_block * attn_scale
//       att_block ← attn_softcap > 0 ? attn_softcap * tanh(att_block / attn_softcap) : att_block
//       (if sliding) mask att_block where kv_pos < q_pos - w
//       new_max ← row_max(att_block, max_vec)
//       att_block ← exp(att_block - new_max) (broadcast row sub then exp)
//       rescale ← exp(max_vec - new_max)
//       sum_vec ← row_sum(att_block, rescale * sum_vec)
//       o_reg ← o_reg * rescale (per row)
//       max_vec ← new_max
//       wait v_arrived[p & 1]
//       load V rt
//       att_bf ← bf16 cast of att_block
//       o_reg ← att_bf @ V + o_reg
//   o_reg ← o_reg / sum_vec (per row, via mul_row with reciprocal)
//   bf16 cast o_reg, store to attn_out_page subtile for owned q_head
//
// Loader role TMA-loads Q from gmem (q_in_act_slot) into q_in_page.
// Storer role TMA-stores attn_out_page to gmem (attn_out_act_slot).
//
// All TMA loads for K/V blocks issue from the consumer body; only
// the warp-0 thread of warp 0 actually issues (TK 2.0's
// `tma::load_async` self-gates on `laneid() == 0`, and the outer
// `if (kittens::warpid() == 0)` further restricts to one warp).
// Local `__shared__ kittens::semaphore` declarations carry the
// per-block-iter handshake; phase parity toggles via `(p & 1)`.
//
// ITERS != 1 is currently unsupported; emits a SKIPPED marker.
// MAX_SK is the proc-macro-time SK_BUCKET upper bound for
// `num_blocks` (host-side launch caps `seq_lens[t] ≤ MAX_SK`).
// ============================================================

#[allow(clippy::too_many_arguments)]
fn render_attention_via_cache_impl<
    const M: u32,
    const HEAD_DIM: u32,
    const NUM_Q_HEADS: u32,
    const NUM_KV_HEADS: u32,
    const BLOCK_SIZE: u32,
    const MAX_SK: u32,
    const NCW: u32,
    const NUM_LAYERS: u32,
    const ITERS: u32,
>(
    q_in_page_id: u32,
    attn_out_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    q_in_act_slot: u32,
    attn_out_act_slot: u32,
    k_smem_offset: u32,
    v_smem_offset: u32,
    attn_scale: f32,
    attn_softcap: f32,
    sliding_window: Option<u32>,
) -> RoleBodies {
    if ITERS != 1 {
        return RoleBodies::skipped("TkAttentionViaCache");
    }
    if NUM_Q_HEADS == 0 || NUM_KV_HEADS == 0 || HEAD_DIM == 0 || NCW == 0 {
        return RoleBodies::skipped("TkAttentionViaCache");
    }
    if NUM_Q_HEADS % NCW != 0 {
        return RoleBodies::skipped("TkAttentionViaCache");
    }
    if NUM_Q_HEADS % NUM_KV_HEADS != 0 {
        return RoleBodies::skipped("TkAttentionViaCache");
    }
    // The prefill `_impl` is correct at M=16 only. M=1 is dispatched
    // to `render_attention_via_cache_decode` by the wrappers
    // (`render_attention_via_cache` / `render_sliding_attention_via_cache`)
    // before reaching this body, so M=1 won't be observed here at
    // codegen time. M=8 and M=64 still need their own renders
    // (M-chunked inner loop / cooperative load split for prefill,
    // or batched-decode shape for M=8 vector-attn). Until those
    // land, M ∉ {1, 16} emits a SKIPPED marker so nvcc compiles
    // the rest of the canonical clean.
    if M != 16 {
        return RoleBodies::skipped("TkAttentionViaCache");
    }

    const Q_HEAD_TILE_ROWS: u32 = 16;

    let kv_dim: u32 = NUM_KV_HEADS * HEAD_DIM;
    let q_dim: u32 = NUM_Q_HEADS * HEAD_DIM;
    let kv_block_bytes: u32 = BLOCK_SIZE * kv_dim * BF16_BYTES;
    let heads_per_warp: u32 = NUM_Q_HEADS / NCW;
    let gqa_group: u32 = NUM_Q_HEADS / NUM_KV_HEADS;

    let loader_phase = storer_phase;
    let q_in_p = page(q_in_page_id);
    let attn_out_p = page(attn_out_page_id);

    // The const-generic shape parameters propagate from the top
    // of this fn through every typed handle and `tk20::*` call
    // below — there is no intermediate runtime u32 storage for
    // any shape. nvcc-validation and E2E coherence on
    // `unsloth/Llama-3.2-1B-Instruct` are the next named phase.

    // -----------------------------------------------------------
    // LOADER role — TMA-load Q from gmem into q_in_page.
    // -----------------------------------------------------------
    let mut loader = CuBlock::new();
    let q_in_ready = page_ready_sem(q_in_p);
    let q_in_consumed = page_consumed_sem(q_in_p);
    let attn_out_consumed = page_consumed_sem(attn_out_p);
    let q_in_gmem = gmem_act_ptr_raw(q_in_act_slot);
    let q_bytes = M * q_dim * BF16_BYTES;
    loader.push(tk20::group_wait::<1>(&q_in_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&attn_out_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes::<1>(&q_in_ready, q_bytes));
    // Q tile shape in the page is `[M, Q_DIM]` where Q_DIM =
    // NUM_Q_HEADS * HEAD_DIM. We load it as a flat [M, Q_DIM] tile.
    loader.push(CuStmt::new(format!(
        "{{ \
         auto& __attn_q_tile = *reinterpret_cast<kittens::st_bf<{m}, {q_dim}>*>(\
         ss.pages[{q_in_id}]); \
         kittens::group<1>::tma::load_async(\
         reinterpret_cast<void*>(&__attn_q_tile), \
         reinterpret_cast<void*>({q_gmem}), \
         {q_bytes}, {q_ready}); \
         }}",
        m = M,
        q_dim = q_dim,
        q_in_id = q_in_page_id,
        q_gmem = q_in_gmem.expr(),
        q_bytes = q_bytes,
        q_ready = q_in_ready.expr(),
    )));

    // -----------------------------------------------------------
    // LAUNCHER role — empty (no per-iter scheduling beyond loader).
    // -----------------------------------------------------------
    let launcher = CuBlock::new();

    // -----------------------------------------------------------
    // CONSUMER role — FlashAttention-2 inner loop.
    // -----------------------------------------------------------
    let mut consumer = CuBlock::new();

    // Local block-scope semaphores for per-block-iter K/V TMA
    // handshake. Single-stage (no double buffering), reused across
    // iterations via phase-parity (`p & 1`).
    consumer.push(tk20::decl_shared_semaphore("__attn_k_arr"));
    consumer.push(tk20::decl_shared_semaphore("__attn_v_arr"));
    // Bars 3 and 4 are unused by other AttentionViaCache canonicals
    // (FQRC lives in a separate Qkv-tier kernel; bars 1-2 are the
    // existing intra-op cross-warp reduce/publish convention).
    consumer.push(tk20::init_semaphore_warp0::<NCW>(
        "__attn_k_arr", 1, /*bar_id=*/ 3,
    ));
    consumer.push(tk20::init_semaphore_warp0::<NCW>(
        "__attn_v_arr", 1, /*bar_id=*/ 4,
    ));

    let k_arr = tk20::local_semaphore_ref("__attn_k_arr");
    let v_arr = tk20::local_semaphore_ref("__attn_v_arr");

    consumer.push(tk20::group_wait::<1>(&q_in_ready, consumer_phase));

    // Bind Q tile typed view (page reinterpret) for downstream
    // subtile slicing.
    consumer.push(CuStmt::new(format!(
        "auto& __attn_q_tile = *reinterpret_cast<kittens::st_bf<{m}, {q_dim}>*>(\
         ss.pages[{q_in_id}]);",
        m = M,
        q_dim = q_dim,
        q_in_id = q_in_page_id,
    )));
    consumer.push(CuStmt::new(format!(
        "auto& __attn_o_tile = *reinterpret_cast<kittens::st_bf<{m}, {q_dim}>*>(\
         ss.pages[{out_id}]);",
        m = M,
        q_dim = q_dim,
        out_id = attn_out_page_id,
    )));

    // K_smem / V_smem typed views (single-stage paged-KV staging).
    consumer.push(CuStmt::new(format!(
        "auto& __attn_k_smem = *reinterpret_cast<kittens::st_bf<{bs}, {kvd}>*>(\
         ss.scratch + {ksoff});",
        bs = BLOCK_SIZE,
        kvd = kv_dim,
        ksoff = k_smem_offset,
    )));
    consumer.push(CuStmt::new(format!(
        "auto& __attn_v_smem = *reinterpret_cast<kittens::st_bf<{bs}, {kvd}>*>(\
         ss.scratch + {vsoff});",
        bs = BLOCK_SIZE,
        kvd = kv_dim,
        vsoff = v_smem_offset,
    )));

    // Per-token paged-KV walk parameters. `seq_lens[0]` is used
    // for the bound; multi-token (M > 1) attention shares the
    // same KV walk because all M tokens belong to the same
    // sequence (vLLM's per-request attention).
    consumer.push(tk20::decl_local_int(
        "__attn_seq_len",
        "static_cast<int>(seq_lens[0])",
    ));
    consumer.push(tk20::decl_local_int(
        "__attn_num_blocks",
        &format!("(__attn_seq_len + {bs} - 1) / {bs}", bs = BLOCK_SIZE),
    ));
    consumer.push(tk20::decl_local_int(
        "__attn_consumer_warp_id",
        "static_cast<int>(kittens::warpid())",
    ));

    // Per-Q-head outer loop — each consumer warp owns
    // `heads_per_warp` Q-heads, indexed by q_head_in_warp ∈
    // [0, heads_per_warp). Global Q-head id is
    // `consumer_warp_id * heads_per_warp + q_head_in_warp`.
    let mut q_head_body = CuBlock::new();
    q_head_body.push(tk20::decl_local_int(
        "__attn_q_head",
        &format!("__attn_consumer_warp_id * {hpw} + __attn_qh", hpw = heads_per_warp),
    ));
    q_head_body.push(tk20::decl_local_int(
        "__attn_kv_head",
        &format!("__attn_q_head / {gg}", gg = gqa_group),
    ));

    // Q register tile: `[Q_HEAD_TILE_ROWS, HEAD_DIM]` (16 rows is
    // TK 2.0's minimum tile-row granularity; M < 16 cases pad).
    let (decl_q_rt, q_rt) =
        tk20::decl_rt_bf_row::<Q_HEAD_TILE_ROWS, HEAD_DIM>("__attn_q_rt");
    q_head_body.push(decl_q_rt);
    // Sub-slice Q tile for this Q-head: column range
    // [q_head * HEAD_DIM, (q_head + 1) * HEAD_DIM).
    q_head_body.push(CuStmt::new(format!(
        "auto __attn_q_sub = __attn_q_tile.template subtile<{m}, {hd}>(\
         int2{{0, __attn_q_head}});",
        m = M.max(Q_HEAD_TILE_ROWS),
        hd = HEAD_DIM,
    )));
    q_head_body.push(CuStmt::new(
        "kittens::warp::load(__attn_q_rt, __attn_q_sub);".to_string(),
    ));

    // Running state: max_vec, sum_vec (rv_fl<Q_HEAD_TILE_ROWS, ortho>
    // — ortho layout matches `rt<row>::col_vec_layout` per
    // `rt_base.cuh:79`, required by row_max/row_sum/sub_row/mul_row/
    // div_row), o_reg (rt_fl<Q_HEAD_TILE_ROWS, HEAD_DIM, row>).
    let (decl_max, max_rv) =
        tk20::decl_rv_fl_neg_infty::<Q_HEAD_TILE_ROWS>("__attn_max");
    let (decl_sum, sum_rv) =
        tk20::decl_rv_fl_ortho::<Q_HEAD_TILE_ROWS>("__attn_sum");
    let (decl_o, o_rt) =
        tk20::decl_rt_fl::<Q_HEAD_TILE_ROWS, HEAD_DIM>("__attn_o");
    q_head_body.push(decl_max);
    q_head_body.push(decl_sum);
    q_head_body.push(tk20::warp_zero_rv::<Q_HEAD_TILE_ROWS, _>(&sum_rv));
    q_head_body.push(decl_o);
    q_head_body.push(tk20::warp_zero_rt::<F32, RtRow, Q_HEAD_TILE_ROWS, HEAD_DIM>(
        &o_rt,
    ));

    // Per-block inner loop.
    let mut p_body = CuBlock::new();
    p_body.push(tk20::decl_local_int(
        "__attn_phase",
        "(static_cast<int>(__attn_p)) & 1",
    ));

    // Issue paged-KV K block TMA load on warp-0 lane-0 only. The
    // TK 2.0 `tma::load_async` helper self-gates on `laneid()
    // == 0`, and the outer `if (kittens::warpid() == 0)` reduces
    // to one warp. Net: single thread issues the TMA.
    let (decl_k_ptr, _k_ptr) = tk20::decl_paged_kv_block_ptr(
        "__attn_k_ptr",
        "key_cache_ptrs",
        layer,
        "block_table[__attn_p]",
        kv_block_bytes,
    );
    let (decl_v_ptr, _v_ptr) = tk20::decl_paged_kv_block_ptr(
        "__attn_v_ptr",
        "value_cache_ptrs",
        layer,
        "block_table[__attn_p]",
        kv_block_bytes,
    );
    p_body.push(decl_k_ptr);
    p_body.push(decl_v_ptr);

    let mut warp0_block = CuBlock::new();
    warp0_block.push(tk20::warp_tma_expect_bytes(
        &k_arr,
        &kv_block_bytes.to_string(),
    ));
    warp0_block.push(CuStmt::new(format!(
        "kittens::tma::load_async(\
         reinterpret_cast<void*>(&__attn_k_smem), \
         reinterpret_cast<void*>(__attn_k_ptr), \
         {bytes}, __attn_k_arr);",
        bytes = kv_block_bytes,
    )));
    p_body.push(tk20::if_else(
        "kittens::warpid() == 0",
        &warp0_block,
        None,
    ));

    p_body.push(tk20::warp_wait_sem(&k_arr, "__attn_phase"));

    // K register tile: row layout, sub-sliced for this kv_head's
    // HEAD_DIM-wide column range within the [BLOCK_SIZE, KV_DIM]
    // K_smem block.
    p_body.push(CuStmt::new(format!(
        "auto __attn_k_sub = __attn_k_smem.template subtile<{bs}, {hd}>(\
         int2{{0, __attn_kv_head}});",
        bs = BLOCK_SIZE,
        hd = HEAD_DIM,
    )));
    let (decl_k_rt, k_rt) =
        tk20::decl_rt_bf_row::<BLOCK_SIZE, HEAD_DIM>("__attn_k_rt");
    p_body.push(decl_k_rt);
    p_body.push(CuStmt::new(
        "kittens::warp::load(__attn_k_rt, __attn_k_sub);".to_string(),
    ));

    // att_block = Q @ K^T; shape `[Q_HEAD_TILE_ROWS, BLOCK_SIZE]`
    // fp32, row layout. C accumulator initialized to zeros (we
    // overwrite, not accumulate, since each block's contribution
    // is its own `Q @ K^T_block`).
    let (decl_att, att_rt) =
        tk20::decl_rt_fl::<Q_HEAD_TILE_ROWS, BLOCK_SIZE>("__attn_att");
    p_body.push(decl_att);
    p_body.push(tk20::warp_zero_rt::<F32, RtRow, Q_HEAD_TILE_ROWS, BLOCK_SIZE>(
        &att_rt,
    ));
    p_body.push(tk20::warp_mma_ABt::<Q_HEAD_TILE_ROWS, HEAD_DIM, BLOCK_SIZE>(
        &att_rt, &q_rt, &k_rt, &att_rt,
    ));

    // att *= attn_scale (runtime f32).
    p_body.push(tk20::warp_mul_rt_scalar::<Q_HEAD_TILE_ROWS, BLOCK_SIZE>(
        &att_rt,
        &att_rt,
        &format!("{:e}f", attn_scale),
    ));

    // Tanh softcap: att = softcap * tanh(att / softcap), iff
    // softcap > 0. Skipped at proc-macro time when softcap == 0
    // (the IR records exactly the f32 value the user set).
    if attn_softcap > 0.0 {
        let cap_lit = format!("{:e}f", attn_softcap);
        let inv_cap_lit = format!("{:e}f", 1.0_f32 / attn_softcap);
        p_body.push(tk20::warp_apply_f32_rt_lambda::<Q_HEAD_TILE_ROWS, BLOCK_SIZE>(
            &att_rt,
            &att_rt,
            &format!("{cap} * tanhf(x * {inv})", cap = cap_lit, inv = inv_cap_lit),
        ));
    }

    // Sliding-window mask: kv_pos = p * BLOCK_SIZE + col;
    // q_pos = seq_len - 1 (decode) or token-relative (prefill).
    // Set att = -INF where kv_pos < q_pos - sliding_window.
    if let Some(w) = sliding_window {
        p_body.push(tk20::warp_apply_f32_rt_lambda::<Q_HEAD_TILE_ROWS, BLOCK_SIZE>(
            &att_rt,
            &att_rt,
            &format!(
                "((static_cast<int>(__attn_p) * {bs} + col) < \
                 (__attn_seq_len - 1 - {w})) ? kittens::base_types::constants<float>::neg_infty() : x",
                bs = BLOCK_SIZE,
                w = w,
            ),
        ));
    }

    // Beyond-seq mask: kv_pos >= seq_len → -INF.
    p_body.push(tk20::warp_apply_f32_rt_lambda::<Q_HEAD_TILE_ROWS, BLOCK_SIZE>(
        &att_rt,
        &att_rt,
        &format!(
            "((static_cast<int>(__attn_p) * {bs} + col) >= __attn_seq_len) \
             ? kittens::base_types::constants<float>::neg_infty() : x",
            bs = BLOCK_SIZE,
        ),
    ));

    // Online softmax update.
    let (decl_new_max, new_max_rv) =
        tk20::decl_rv_fl_ortho::<Q_HEAD_TILE_ROWS>("__attn_new_max");
    p_body.push(decl_new_max);
    p_body.push(tk20::warp_row_max_running::<Q_HEAD_TILE_ROWS, BLOCK_SIZE>(
        &new_max_rv,
        &att_rt,
        &max_rv,
    ));
    // att = exp(att - new_max) (broadcast row sub then exp).
    p_body.push(tk20::warp_sub_row::<Q_HEAD_TILE_ROWS, BLOCK_SIZE>(
        &att_rt,
        &att_rt,
        &new_max_rv,
    ));
    p_body.push(tk20::warp_exp_rt::<Q_HEAD_TILE_ROWS, BLOCK_SIZE>(
        &att_rt, &att_rt,
    ));
    // rescale = exp(max_vec - new_max).
    let (decl_rescale, rescale_rv) =
        tk20::decl_rv_fl_ortho::<Q_HEAD_TILE_ROWS>("__attn_rescale");
    p_body.push(decl_rescale);
    p_body.push(tk20::warp_sub_rv_rv::<Q_HEAD_TILE_ROWS, _>(
        &rescale_rv,
        &max_rv,
        &new_max_rv,
    ));
    p_body.push(tk20::warp_exp_rv::<Q_HEAD_TILE_ROWS, _>(&rescale_rv, &rescale_rv));
    // sum_vec = row_sum(att, rescale * sum_vec).
    p_body.push(tk20::warp_mul_rv_rv::<Q_HEAD_TILE_ROWS, _>(
        &sum_rv,
        &sum_rv,
        &rescale_rv,
    ));
    p_body.push(tk20::warp_row_sum_running::<Q_HEAD_TILE_ROWS, BLOCK_SIZE>(
        &sum_rv, &att_rt, &sum_rv,
    ));
    // o_reg *= rescale (per row).
    p_body.push(tk20::warp_mul_row::<Q_HEAD_TILE_ROWS, HEAD_DIM>(
        &o_rt,
        &o_rt,
        &rescale_rv,
    ));
    // max_vec = new_max (reuse register).
    p_body.push(tk20::warp_copy_rv::<F32, Q_HEAD_TILE_ROWS, _>(
        &max_rv,
        &new_max_rv,
    ));

    // V load — same dance as K.
    let mut v_warp0_block = CuBlock::new();
    v_warp0_block.push(tk20::warp_tma_expect_bytes(
        &v_arr,
        &kv_block_bytes.to_string(),
    ));
    v_warp0_block.push(CuStmt::new(format!(
        "kittens::tma::load_async(\
         reinterpret_cast<void*>(&__attn_v_smem), \
         reinterpret_cast<void*>(__attn_v_ptr), \
         {bytes}, __attn_v_arr);",
        bytes = kv_block_bytes,
    )));
    p_body.push(tk20::if_else(
        "kittens::warpid() == 0",
        &v_warp0_block,
        None,
    ));
    p_body.push(tk20::warp_wait_sem(&v_arr, "__attn_phase"));

    p_body.push(CuStmt::new(format!(
        "auto __attn_v_sub = __attn_v_smem.template subtile<{bs}, {hd}>(\
         int2{{0, __attn_kv_head}});",
        bs = BLOCK_SIZE,
        hd = HEAD_DIM,
    )));
    // V register tile in COL layout — `mma_AB` wants B=col.
    let (decl_v_rt, v_rt) =
        tk20::decl_rt_bf_col::<BLOCK_SIZE, HEAD_DIM>("__attn_v_rt");
    p_body.push(decl_v_rt);
    p_body.push(CuStmt::new(
        "kittens::warp::load(__attn_v_rt, __attn_v_sub);".to_string(),
    ));

    // bf16 cast of att_block for the PV matmul.
    let (decl_att_bf, att_bf_rt) =
        tk20::decl_rt_bf_row::<Q_HEAD_TILE_ROWS, BLOCK_SIZE>("__attn_att_bf");
    p_body.push(decl_att_bf);
    p_body.push(tk20::warp_copy_rt_fl_to_bf::<Q_HEAD_TILE_ROWS, BLOCK_SIZE>(
        &att_bf_rt,
        &att_rt,
    ));

    // o_reg += att_bf @ V.
    p_body.push(tk20::warp_mma_AB::<Q_HEAD_TILE_ROWS, BLOCK_SIZE, HEAD_DIM>(
        &o_rt, &att_bf_rt, &v_rt, &o_rt,
    ));

    q_head_body.push(tk20::for_loop(
        "uint32_t __attn_p = 0; __attn_p < static_cast<uint32_t>(__attn_num_blocks); ++__attn_p",
        &p_body,
    ));

    // Finalize: o_reg /= sum_vec (per row).
    q_head_body.push(tk20::warp_div_row::<Q_HEAD_TILE_ROWS, HEAD_DIM>(
        &o_rt, &o_rt, &sum_rv,
    ));

    // Cast o_reg fp32 → bf16 register tile, then store to
    // attn_out subtile for this Q-head.
    let (decl_o_bf, o_bf_rt) =
        tk20::decl_rt_bf_row::<Q_HEAD_TILE_ROWS, HEAD_DIM>("__attn_o_bf");
    q_head_body.push(decl_o_bf);
    q_head_body.push(tk20::warp_copy_rt_fl_to_bf::<Q_HEAD_TILE_ROWS, HEAD_DIM>(
        &o_bf_rt, &o_rt,
    ));
    q_head_body.push(CuStmt::new(format!(
        "auto __attn_o_sub = __attn_o_tile.template subtile<{m}, {hd}>(\
         int2{{0, __attn_q_head}});",
        m = M.max(Q_HEAD_TILE_ROWS),
        hd = HEAD_DIM,
    )));
    q_head_body.push(CuStmt::new(
        "kittens::warp::store(__attn_o_sub, __attn_o_bf);".to_string(),
    ));

    consumer.push(tk20::for_loop(
        &format!(
            "uint32_t __attn_qh = 0; __attn_qh < {hpw}; ++__attn_qh",
            hpw = heads_per_warp
        ),
        &q_head_body,
    ));

    let attn_out_done = page_done_sem(attn_out_p);
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive::<1>(&attn_out_done),
        tk20::group_arrive::<1>(&q_in_consumed),
    ]));

    // -----------------------------------------------------------
    // STORER role — TMA-store attn_out_page back to gmem.
    // -----------------------------------------------------------
    let mut storer = CuBlock::new();
    let attn_out_gmem = gmem_act_ptr_raw(attn_out_act_slot);
    let attn_out_bytes = M * q_dim * BF16_BYTES;
    let attn_out_consumed = page_consumed_sem(attn_out_p);
    storer.push(tk20::group_wait::<1>(&attn_out_done, storer_phase));
    storer.push(CuStmt::new(format!(
        "{{ \
         auto& __attn_o_tile = *reinterpret_cast<kittens::st_bf<{m}, {q_dim}>*>(\
         ss.pages[{out_id}]); \
         kittens::group<1>::tma::store_async(\
         reinterpret_cast<void*>({out_gmem}), \
         reinterpret_cast<void*>(&__attn_o_tile), \
         {bytes}); \
         }}",
        m = M,
        q_dim = q_dim,
        out_id = attn_out_page_id,
        out_gmem = attn_out_gmem.expr(),
        bytes = attn_out_bytes,
    )));
    storer.push(tk20::group_tma_store_async_wait::<1>());
    storer.push(tk20::group_arrive::<1>(&attn_out_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}

// ============================================================
// AttentionViaCache (decode shape, M=1).
//
// Port of TK 2.0's GQA decode template
// (`third_party/thunderkittens/kernels/attn/demo/gqa_decode/template_gqa_decode_new.cu`)
// adapted to ferrite's single-CTA 4-warp-role substrate.
//
// Algorithmic shape (matches TK partial_template.consumer):
//   * Q is heads-packed in scratch:
//     `st_bf<NUM_Q_HEADS, HEAD_DIM>` — row r holds q_head r's Q vector.
//   * Each consumer warp w owns exactly one kv_head (`kv_head = w`,
//     requires NCW == NUM_KV_HEADS). The GQA_GROUP rows for that
//     kv_head live at rows `[kv_head*GQA, kv_head*GQA + GQA)` of the
//     packed Q tile, inside the 16-row TK window
//     `subtile<16, HEAD_DIM>(int2{(kv_head*GQA)/16, 0})`.
//   * Per-block FA-2 with warp-level `mma_ABt` / `mma_AB`,
//     `kittens::warp::*` operating on `rt_bf<16, HEAD_DIM>` Q,
//     `rt_bf<BLOCK_SIZE, HEAD_DIM>` K, `rt_bf<BLOCK_SIZE, HEAD_DIM>`
//     V (col-layout for the PV mma).
//   * Log-space online softmax: SOFTMAX_TEMPERATURE = scale *
//     log2(e); use exp2 (matches TK's `exp2` / `log2` so the
//     reduction merge across split-position partials remains
//     numerically stable even though we only emit the partial here).
//   * Per-warp output staging: each warp `w` writes its 16-row
//     `rt_bf<16, HEAD_DIM>` o-tile to an UNSWIZZLED
//     `st<bf16, 16, HEAD_DIM, false>` slot in the attn_out page at
//     offset `final_out_bytes + w * 16 * HEAD_DIM * 2`. Unswizzled
//     so the cross-warp compact pass can flat-read row-major.
//   * Cross-warp compact: NCW * 32 threads scatter each warp's
//     valid GQA rows into the final `[NUM_Q_HEADS, HEAD_DIM]` tile
//     at the start of the attn_out page. Source row (kv_head, GQA-
//     local r, c) → dest row (kv_head*GQA + r, c). Source read uses
//     row-major arithmetic on the unswizzled staging; dest write
//     uses row-major arithmetic on the final tile.
//   * Final TMA store of the compacted `[NUM_Q_HEADS, HEAD_DIM]`
//     tile to gmem (`M*NUM_Q_HEADS*HEAD_DIM*BF16_BYTES` bytes via
//     byte-mode `tma::store_async`).
//
// Per-block KV walk:
//   * Loader role TMA-loads Q from gmem into the in-page (the same
//     heads-packed `[1, NUM_Q_HEADS, HEAD_DIM]` byte layout the
//     prefill render writes — for M=1 this is identical to the
//     decode-required `[NUM_Q_HEADS, HEAD_DIM]` flat).
//   * Per FA-2 iter, warp-0-lane-0 issues two paged TMA loads
//     (K and V) for the current logical KV block (full
//     `[BLOCK_SIZE, NUM_KV_HEADS * HEAD_DIM]` — every kv_head's
//     slice loaded together, each warp subtiles to its own
//     kv_head's column range).
//   * `__attn_seq_len = seq_lens[0]`; `num_blocks = ceil(seq_len /
//     BLOCK_SIZE)`. The FINAL block needs right-fill masking to
//     -inf for kv positions ≥ seq_len (TK's `right_fill`
//     equivalent, expressed here as a per-element mask via
//     `warp_apply_f32_rt_lambda`).
//
// Constraints (skip if violated):
//   * M == 1 (decode only; multi-token decode batching is a
//     separate render).
//   * ITERS == 1.
//   * NCW == NUM_KV_HEADS (one warp per kv_head).
//   * NUM_Q_HEADS % NUM_KV_HEADS == 0 (clean GQA group).
//   * NUM_Q_HEADS % 16 == 0 (heads-packed Q tile satisfies TK's
//     row-divisibility; for llama-1B NUM_Q_HEADS=32, 32%16==0 ✓).
//   * HEAD_DIM % 16 == 0 (TILE_COL_DIM constraint).
//   * (NUM_Q_HEADS + NCW * 16) * HEAD_DIM * 2 ≤ PAGE_SIZE — the
//     attn_out page must fit final + per-warp staging. PAGE_SIZE is
//     a TapeBudget constant not visible here; checked at runtime
//     via the same scratch-OOB pattern by the proc-macro builder.
// ============================================================

#[allow(clippy::too_many_arguments)]
pub fn render_attention_via_cache_decode<
    const HEAD_DIM: u32,
    const NUM_Q_HEADS: u32,
    const NUM_KV_HEADS: u32,
    const BLOCK_SIZE: u32,
    const MAX_SK: u32,
    const NCW: u32,
    const NUM_LAYERS: u32,
    const ITERS: u32,
>(
    q_in_page_id: u32,
    attn_out_page_id: u32,
    consumer_phase: u32,
    storer_phase: u32,
    layer: u32,
    q_in_act_slot: u32,
    attn_out_act_slot: u32,
    k_smem_offset: u32,
    v_smem_offset: u32,
    attn_scale: f32,
    attn_softcap: f32,
    sliding_window: Option<u32>,
) -> RoleBodies {
    if ITERS != 1 {
        return RoleBodies::skipped("TkAttentionViaCacheDecode");
    }
    if NUM_Q_HEADS == 0 || NUM_KV_HEADS == 0 || HEAD_DIM == 0 || NCW == 0 {
        return RoleBodies::skipped("TkAttentionViaCacheDecode");
    }
    if NUM_Q_HEADS % NUM_KV_HEADS != 0 {
        return RoleBodies::skipped("TkAttentionViaCacheDecode");
    }
    if NCW != NUM_KV_HEADS {
        return RoleBodies::skipped("TkAttentionViaCacheDecode");
    }
    if NUM_Q_HEADS % 16 != 0 {
        return RoleBodies::skipped("TkAttentionViaCacheDecode");
    }
    if HEAD_DIM % 16 != 0 {
        return RoleBodies::skipped("TkAttentionViaCacheDecode");
    }

    const Q_HEAD_TILE_ROWS: u32 = 16;

    let kv_dim: u32 = NUM_KV_HEADS * HEAD_DIM;
    let q_dim: u32 = NUM_Q_HEADS * HEAD_DIM;
    let kv_block_bytes: u32 = BLOCK_SIZE * kv_dim * BF16_BYTES;
    let gqa_group: u32 = NUM_Q_HEADS / NUM_KV_HEADS;
    let final_out_bytes: u32 = NUM_Q_HEADS * HEAD_DIM * BF16_BYTES;
    let stage_per_warp_bytes: u32 = Q_HEAD_TILE_ROWS * HEAD_DIM * BF16_BYTES;

    let loader_phase = storer_phase;
    let q_in_p = page(q_in_page_id);
    let attn_out_p = page(attn_out_page_id);

    // Suppress unused-bind warnings on params that this render path
    // doesn't consume directly (the bytes / page-fit are proven by
    // the caller's TapeBudget; MAX_SK is metadata for the host
    // launcher, not the kernel body).
    let _ = MAX_SK;

    // -----------------------------------------------------------
    // LOADER role — TMA-load Q from gmem into q_in_page.
    // Layout: [M=1, NUM_Q_HEADS, HEAD_DIM] flat = NUM_Q_HEADS *
    // HEAD_DIM bf16 elements. Reinterpreted in consumer as
    // st_bf<NUM_Q_HEADS, HEAD_DIM> (heads-packed).
    // -----------------------------------------------------------
    let mut loader = CuBlock::new();
    let q_in_ready = page_ready_sem(q_in_p);
    let q_in_consumed = page_consumed_sem(q_in_p);
    let attn_out_consumed = page_consumed_sem(attn_out_p);
    let q_in_gmem = gmem_act_ptr_raw(q_in_act_slot);
    let q_bytes = NUM_Q_HEADS * HEAD_DIM * BF16_BYTES;
    loader.push(tk20::group_wait::<1>(&q_in_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&attn_out_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes::<1>(&q_in_ready, q_bytes));
    loader.push(CuStmt::new(format!(
        "{{ \
         auto& __attn_q_tile = *reinterpret_cast<kittens::st_bf<{nqh}, {hd}>*>(\
         ss.pages[{q_in_id}]); \
         kittens::group<1>::tma::load_async(\
         reinterpret_cast<void*>(&__attn_q_tile), \
         reinterpret_cast<void*>({q_gmem}), \
         {q_bytes}, {q_ready}); \
         }}",
        nqh = NUM_Q_HEADS,
        hd = HEAD_DIM,
        q_in_id = q_in_page_id,
        q_gmem = q_in_gmem.expr(),
        q_bytes = q_bytes,
        q_ready = q_in_ready.expr(),
    )));

    // -----------------------------------------------------------
    // LAUNCHER role — empty.
    // -----------------------------------------------------------
    let launcher = CuBlock::new();

    // -----------------------------------------------------------
    // CONSUMER role — per-warp FA-2 inner loop, then cross-warp
    // compact, then arrive on attn_out_done.
    // -----------------------------------------------------------
    let mut consumer = CuBlock::new();

    // Local block-scope semaphores for per-block-iter K/V TMA
    // handshake (single-stage, phase-toggled).
    consumer.push(tk20::decl_shared_semaphore("__attn_k_arr"));
    consumer.push(tk20::decl_shared_semaphore("__attn_v_arr"));
    consumer.push(tk20::init_semaphore_warp0::<NCW>(
        "__attn_k_arr", 1, /*bar_id=*/ 3,
    ));
    consumer.push(tk20::init_semaphore_warp0::<NCW>(
        "__attn_v_arr", 1, /*bar_id=*/ 4,
    ));

    let k_arr = tk20::local_semaphore_ref("__attn_k_arr");
    let v_arr = tk20::local_semaphore_ref("__attn_v_arr");

    consumer.push(tk20::group_wait::<1>(&q_in_ready, consumer_phase));

    // Heads-packed Q tile view of in-page.
    consumer.push(CuStmt::new(format!(
        "auto& __attn_q_tile = *reinterpret_cast<kittens::st_bf<{nqh}, {hd}>*>(\
         ss.pages[{q_in_id}]);",
        nqh = NUM_Q_HEADS,
        hd = HEAD_DIM,
        q_in_id = q_in_page_id,
    )));

    // K_smem / V_smem typed views for this layer's full-block
    // staging (single-stage paged-KV gather).
    consumer.push(CuStmt::new(format!(
        "auto& __attn_k_smem = *reinterpret_cast<kittens::st_bf<{bs}, {kvd}>*>(\
         ss.scratch + {ksoff});",
        bs = BLOCK_SIZE,
        kvd = kv_dim,
        ksoff = k_smem_offset,
    )));
    consumer.push(CuStmt::new(format!(
        "auto& __attn_v_smem = *reinterpret_cast<kittens::st_bf<{bs}, {kvd}>*>(\
         ss.scratch + {vsoff});",
        bs = BLOCK_SIZE,
        kvd = kv_dim,
        vsoff = v_smem_offset,
    )));

    // Per-token paged-KV walk parameters. M=1: single seq, single
    // block_table row.
    consumer.push(tk20::decl_local_int(
        "__attn_seq_len",
        "static_cast<int>(seq_lens[0])",
    ));
    consumer.push(tk20::decl_local_int(
        "__attn_num_blocks",
        &format!("(__attn_seq_len + {bs} - 1) / {bs}", bs = BLOCK_SIZE),
    ));
    consumer.push(tk20::decl_local_int(
        "__attn_consumer_warp_id",
        "static_cast<int>(kittens::warpid())",
    ));

    // Per-warp kv_head ownership: kv_head == warp_id (NCW ==
    // NUM_KV_HEADS). The GQA_GROUP Q rows for this kv_head live in
    // a 16-row Q-tile window; the per-CTA Q tile is contiguous so
    // window_id and the row offset within that window are fixed
    // const-generic functions of kv_head.
    consumer.push(tk20::decl_local_int(
        "__attn_kv_head",
        "__attn_consumer_warp_id",
    ));
    consumer.push(tk20::decl_local_int(
        "__attn_q_window_id",
        &format!("(__attn_kv_head * {gqa}) / 16", gqa = gqa_group),
    ));
    consumer.push(tk20::decl_local_int(
        "__attn_q_row_in_window",
        &format!("(__attn_kv_head * {gqa}) % 16", gqa = gqa_group),
    ));

    // Q register tile for this kv_head's window: rt_bf<16, HEAD_DIM>
    // row-layout. Subtile<16, HEAD_DIM> at (window_id, 0) of the
    // heads-packed [NUM_Q_HEADS, HEAD_DIM] Q tile.
    let (decl_q_rt, q_rt) =
        tk20::decl_rt_bf_row::<Q_HEAD_TILE_ROWS, HEAD_DIM>("__attn_q_rt");
    consumer.push(decl_q_rt);
    consumer.push(CuStmt::new(format!(
        "auto __attn_q_sub = __attn_q_tile.template subtile<16, {hd}>(\
         int2{{__attn_q_window_id, 0}});",
        hd = HEAD_DIM,
    )));
    consumer.push(CuStmt::new(
        "kittens::warp::load(__attn_q_rt, __attn_q_sub);".to_string(),
    ));

    // Online softmax running state: log-space (TK's exp2 / log2
    // pattern) so the partial outputs are mergeable by a future
    // reduction op without losing precision.
    //
    //   max_vec   ← -inf (per-row max of un-scaled att)
    //   sum_vec   ← 0    (per-row sum of exp2(scaled - max_scaled))
    //   o_reg     ← 0    (per-row accumulator)
    //
    // SOFTMAX_TEMPERATURE = attn_scale * log2(e). Multiplying att
    // by this (instead of by attn_scale) lets us use exp2 instead
    // of expf — same math, exp2 is faster and matches TK.
    let (decl_max, max_rv) =
        tk20::decl_rv_fl_neg_infty::<Q_HEAD_TILE_ROWS>("__attn_max");
    let (decl_sum, sum_rv) =
        tk20::decl_rv_fl_ortho::<Q_HEAD_TILE_ROWS>("__attn_sum");
    let (decl_o, o_rt) =
        tk20::decl_rt_fl::<Q_HEAD_TILE_ROWS, HEAD_DIM>("__attn_o");
    consumer.push(decl_max);
    consumer.push(decl_sum);
    consumer.push(tk20::warp_zero_rv::<Q_HEAD_TILE_ROWS, _>(&sum_rv));
    consumer.push(decl_o);
    consumer.push(tk20::warp_zero_rt::<F32, RtRow, Q_HEAD_TILE_ROWS, HEAD_DIM>(
        &o_rt,
    ));

    let softmax_temperature_lit = format!("{:e}f * 1.44269504089f", attn_scale);
    consumer.push(CuStmt::new(format!(
        "const float __attn_softmax_T = {st};",
        st = softmax_temperature_lit,
    )));

    // -----------------------------------------------------------
    // Per-block FA-2 inner loop.
    // -----------------------------------------------------------
    let mut p_body = CuBlock::new();
    p_body.push(tk20::decl_local_int(
        "__attn_phase",
        "(static_cast<int>(__attn_p)) & 1",
    ));

    // Issue paged-KV K/V block TMA loads on warp-0 lane-0 only.
    let (decl_k_ptr, _k_ptr) = tk20::decl_paged_kv_block_ptr(
        "__attn_k_ptr",
        "key_cache_ptrs",
        layer,
        "block_table[__attn_p]",
        kv_block_bytes,
    );
    let (decl_v_ptr, _v_ptr) = tk20::decl_paged_kv_block_ptr(
        "__attn_v_ptr",
        "value_cache_ptrs",
        layer,
        "block_table[__attn_p]",
        kv_block_bytes,
    );
    p_body.push(decl_k_ptr);
    p_body.push(decl_v_ptr);

    let mut warp0_block = CuBlock::new();
    warp0_block.push(tk20::warp_tma_expect_bytes(
        &k_arr,
        &kv_block_bytes.to_string(),
    ));
    warp0_block.push(CuStmt::new(format!(
        "kittens::tma::load_async(\
         reinterpret_cast<void*>(&__attn_k_smem), \
         reinterpret_cast<void*>(__attn_k_ptr), \
         {bytes}, __attn_k_arr);",
        bytes = kv_block_bytes,
    )));
    p_body.push(tk20::if_else(
        "kittens::warpid() == 0",
        &warp0_block,
        None,
    ));
    p_body.push(tk20::warp_wait_sem(&k_arr, "__attn_phase"));

    // K subtile for this warp's kv_head: BLOCK_SIZE rows ×
    // HEAD_DIM cols at column offset kv_head * HEAD_DIM of the
    // [BLOCK_SIZE, NUM_KV_HEADS * HEAD_DIM] K_smem tile.
    p_body.push(CuStmt::new(format!(
        "auto __attn_k_sub = __attn_k_smem.template subtile<{bs}, {hd}>(\
         int2{{0, __attn_kv_head}});",
        bs = BLOCK_SIZE,
        hd = HEAD_DIM,
    )));
    let (decl_k_rt, k_rt) =
        tk20::decl_rt_bf_row::<BLOCK_SIZE, HEAD_DIM>("__attn_k_rt");
    p_body.push(decl_k_rt);
    p_body.push(CuStmt::new(
        "kittens::warp::load(__attn_k_rt, __attn_k_sub);".to_string(),
    ));

    // att = Q @ K^T : rt_fl<16, BLOCK_SIZE>. C accumulator zeroed
    // (we overwrite, this iter's contribution is its own block).
    let (decl_att, att_rt) =
        tk20::decl_rt_fl::<Q_HEAD_TILE_ROWS, BLOCK_SIZE>("__attn_att");
    p_body.push(decl_att);
    p_body.push(tk20::warp_zero_rt::<F32, RtRow, Q_HEAD_TILE_ROWS, BLOCK_SIZE>(
        &att_rt,
    ));
    p_body.push(tk20::warp_mma_ABt::<Q_HEAD_TILE_ROWS, HEAD_DIM, BLOCK_SIZE>(
        &att_rt, &q_rt, &k_rt, &att_rt,
    ));

    // Softcap (att = softcap * tanh(att / softcap)) BEFORE the
    // softmax temperature scaling. Skipped at proc-macro time when
    // softcap == 0.
    if attn_softcap > 0.0 {
        let cap_lit = format!("{:e}f", attn_softcap);
        let inv_cap_lit = format!("{:e}f", 1.0_f32 / attn_softcap);
        p_body.push(tk20::warp_apply_f32_rt_lambda::<Q_HEAD_TILE_ROWS, BLOCK_SIZE>(
            &att_rt,
            &att_rt,
            &format!("{cap} * tanhf(x * {inv})", cap = cap_lit, inv = inv_cap_lit),
        ));
    }

    // Sliding-window mask: kv_pos = p * BLOCK_SIZE + col;
    // q_pos = seq_len - 1 (decode-only, M=1).
    if let Some(w) = sliding_window {
        p_body.push(tk20::warp_apply_f32_rt_lambda::<Q_HEAD_TILE_ROWS, BLOCK_SIZE>(
            &att_rt,
            &att_rt,
            &format!(
                "((static_cast<int>(__attn_p) * {bs} + col) < \
                 (__attn_seq_len - 1 - {w})) ? -INFINITY : x",
                bs = BLOCK_SIZE,
                w = w,
            ),
        ));
    }

    // Beyond-seq mask (TK's right_fill equivalent for col): att =
    // -inf where kv_pos >= seq_len. Combined with a row mask: rows
    // outside [row_in_window, row_in_window+GQA_GROUP) belong to
    // neighboring kv_heads' Q heads and must not be scored against
    // THIS kv_head's K. Emit `kittens::warp::apply` directly so we
    // can name the row arg (the tk20 helper hides it as `/*row*/`).
    p_body.push(CuStmt::new(format!(
        "kittens::warp::apply(__attn_att, __attn_att, \
         [=] __device__ (int row, int col, float x) {{ \
             return (((static_cast<int>(__attn_p) * {bs} + col) >= __attn_seq_len) || \
                     (row < __attn_q_row_in_window) || \
                     (row >= __attn_q_row_in_window + {gqa})) ? -INFINITY : x; \
         }});",
        bs = BLOCK_SIZE,
        gqa = gqa_group,
    )));

    // Online softmax update — TK's log-space pattern (exp2 of
    // SOFTMAX_TEMPERATURE-scaled values).
    //   att' = att * SOFTMAX_TEMPERATURE
    //   new_max = row_max(att', max_vec)
    //   att' -= new_max ; att' = exp2(att')
    //   rescale = exp2(max_vec - new_max)
    //   sum = rescale * sum + row_sum(att')
    //   o   = rescale * o
    //   max = new_max
    p_body.push(tk20::warp_apply_f32_rt_lambda::<Q_HEAD_TILE_ROWS, BLOCK_SIZE>(
        &att_rt,
        &att_rt,
        "x * __attn_softmax_T",
    ));
    let (decl_new_max, new_max_rv) =
        tk20::decl_rv_fl_ortho::<Q_HEAD_TILE_ROWS>("__attn_new_max");
    p_body.push(decl_new_max);
    p_body.push(tk20::warp_row_max_running::<Q_HEAD_TILE_ROWS, BLOCK_SIZE>(
        &new_max_rv,
        &att_rt,
        &max_rv,
    ));
    p_body.push(tk20::warp_sub_row::<Q_HEAD_TILE_ROWS, BLOCK_SIZE>(
        &att_rt,
        &att_rt,
        &new_max_rv,
    ));
    // exp2 instead of exp (TK pattern from
    // `template_gqa_decode_new.cu:186` — `exp2(att_block_fp32,
    // att_block_fp32)`). `kittens::warp::exp2` lives at
    // `include/ops/group/register/tile/maps.cuh:482`.
    p_body.push(CuStmt::new(
        "kittens::warp::exp2(__attn_att, __attn_att);".to_string(),
    ));

    let (decl_rescale, rescale_rv) =
        tk20::decl_rv_fl_ortho::<Q_HEAD_TILE_ROWS>("__attn_rescale");
    p_body.push(decl_rescale);
    p_body.push(tk20::warp_sub_rv_rv::<Q_HEAD_TILE_ROWS, _>(
        &rescale_rv,
        &max_rv,
        &new_max_rv,
    ));
    // `kittens::warp::exp2` overload for rv (register vector) at
    // `include/ops/group/register/vec/maps.cuh:205`.
    p_body.push(CuStmt::new(
        "kittens::warp::exp2(__attn_rescale, __attn_rescale);".to_string(),
    ));

    p_body.push(tk20::warp_mul_rv_rv::<Q_HEAD_TILE_ROWS, _>(
        &sum_rv,
        &sum_rv,
        &rescale_rv,
    ));
    p_body.push(tk20::warp_row_sum_running::<Q_HEAD_TILE_ROWS, BLOCK_SIZE>(
        &sum_rv, &att_rt, &sum_rv,
    ));
    p_body.push(tk20::warp_mul_row::<Q_HEAD_TILE_ROWS, HEAD_DIM>(
        &o_rt,
        &o_rt,
        &rescale_rv,
    ));
    p_body.push(tk20::warp_copy_rv::<F32, Q_HEAD_TILE_ROWS, _>(
        &max_rv,
        &new_max_rv,
    ));

    // V load (single-stage, paged-TMA, warp-0-only issue).
    let mut v_warp0_block = CuBlock::new();
    v_warp0_block.push(tk20::warp_tma_expect_bytes(
        &v_arr,
        &kv_block_bytes.to_string(),
    ));
    v_warp0_block.push(CuStmt::new(format!(
        "kittens::tma::load_async(\
         reinterpret_cast<void*>(&__attn_v_smem), \
         reinterpret_cast<void*>(__attn_v_ptr), \
         {bytes}, __attn_v_arr);",
        bytes = kv_block_bytes,
    )));
    p_body.push(tk20::if_else(
        "kittens::warpid() == 0",
        &v_warp0_block,
        None,
    ));
    p_body.push(tk20::warp_wait_sem(&v_arr, "__attn_phase"));

    p_body.push(CuStmt::new(format!(
        "auto __attn_v_sub = __attn_v_smem.template subtile<{bs}, {hd}>(\
         int2{{0, __attn_kv_head}});",
        bs = BLOCK_SIZE,
        hd = HEAD_DIM,
    )));
    let (decl_v_rt, v_rt) =
        tk20::decl_rt_bf_col::<BLOCK_SIZE, HEAD_DIM>("__attn_v_rt");
    p_body.push(decl_v_rt);
    p_body.push(CuStmt::new(
        "kittens::warp::load(__attn_v_rt, __attn_v_sub);".to_string(),
    ));

    let (decl_att_bf, att_bf_rt) =
        tk20::decl_rt_bf_row::<Q_HEAD_TILE_ROWS, BLOCK_SIZE>("__attn_att_bf");
    p_body.push(decl_att_bf);
    p_body.push(tk20::warp_copy_rt_fl_to_bf::<Q_HEAD_TILE_ROWS, BLOCK_SIZE>(
        &att_bf_rt,
        &att_rt,
    ));
    p_body.push(tk20::warp_mma_AB::<Q_HEAD_TILE_ROWS, BLOCK_SIZE, HEAD_DIM>(
        &o_rt, &att_bf_rt, &v_rt, &o_rt,
    ));

    consumer.push(tk20::for_loop(
        "uint32_t __attn_p = 0; __attn_p < static_cast<uint32_t>(__attn_num_blocks); ++__attn_p",
        &p_body,
    ));

    // Finalize: o /= sum_vec (per row).
    consumer.push(tk20::warp_div_row::<Q_HEAD_TILE_ROWS, HEAD_DIM>(
        &o_rt, &o_rt, &sum_rv,
    ));

    // Cast o (fp32) → bf16, store to per-warp UNSWIZZLED staging in
    // attn_out page at offset `final_out_bytes + warpid * 16 *
    // HEAD_DIM * 2`. swizzle=false so the cross-warp compact below
    // can flat-read row-major.
    let (decl_o_bf, o_bf_rt) =
        tk20::decl_rt_bf_row::<Q_HEAD_TILE_ROWS, HEAD_DIM>("__attn_o_bf");
    consumer.push(decl_o_bf);
    consumer.push(tk20::warp_copy_rt_fl_to_bf::<Q_HEAD_TILE_ROWS, HEAD_DIM>(
        &o_bf_rt, &o_rt,
    ));
    consumer.push(CuStmt::new(format!(
        "auto& __attn_stage_w = *reinterpret_cast<\
         kittens::st<kittens::bf16, 16, {hd}, false>*>(\
         ss.pages[{out_id}] + {fob} + __attn_consumer_warp_id * {sw});",
        hd = HEAD_DIM,
        out_id = attn_out_page_id,
        fob = final_out_bytes,
        sw = stage_per_warp_bytes,
    )));
    consumer.push(CuStmt::new(
        "kittens::warp::store(__attn_stage_w, __attn_o_bf);".to_string(),
    ));

    // -----------------------------------------------------------
    // Cross-warp barrier — every warp's staging slot must be
    // visible before the compact pass reads it.
    // -----------------------------------------------------------
    consumer.push(CuStmt::new("__syncthreads();".to_string()));

    // -----------------------------------------------------------
    // Cross-warp compact: NCW * 32 threads cooperatively scatter
    // each kv_head's GQA valid rows from staging into the final
    // [NUM_Q_HEADS, HEAD_DIM] tile at the start of the attn_out
    // page. Source row in staging = `(kv_head*GQA + r) % 16` which
    // for GQA dividing 16 evenly is just `(kv_head*GQA) % 16 + r`.
    // -----------------------------------------------------------
    consumer.push(CuStmt::new(format!(
        "{{ \
         int __compact_total = {nqh} * {hd}; \
         int __compact_threads = {ncw} * 32; \
         int __compact_tid = static_cast<int>(threadIdx.x); \
         __nv_bfloat16* __compact_final_base = \
             reinterpret_cast<__nv_bfloat16*>(ss.pages[{out_id}]); \
         uint8_t* __compact_stage_base = ss.pages[{out_id}] + {fob}; \
         for (int __idx = __compact_tid; __idx < __compact_total; __idx += __compact_threads) {{ \
             int __r = __idx / {hd}; \
             int __c = __idx % {hd}; \
             int __kvh = __r / {gqa}; \
             int __local_r = __r % {gqa}; \
             int __row_in_win = ((__kvh * {gqa}) % 16) + __local_r; \
             __nv_bfloat16* __src = \
                 reinterpret_cast<__nv_bfloat16*>(__compact_stage_base + __kvh * {sw}) \
                 + __row_in_win * {hd} + __c; \
             __compact_final_base[__r * {hd} + __c] = *__src; \
         }} \
         }}",
        nqh = NUM_Q_HEADS,
        hd = HEAD_DIM,
        ncw = NCW,
        out_id = attn_out_page_id,
        fob = final_out_bytes,
        gqa = gqa_group,
        sw = stage_per_warp_bytes,
    )));

    consumer.push(CuStmt::new("__syncthreads();".to_string()));

    let attn_out_done = page_done_sem(attn_out_p);
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive::<1>(&attn_out_done),
        tk20::group_arrive::<1>(&q_in_consumed),
    ]));

    // -----------------------------------------------------------
    // STORER role — TMA-store final compacted [NUM_Q_HEADS,
    // HEAD_DIM] tile from offset 0 of the attn_out page to gmem.
    // -----------------------------------------------------------
    let mut storer = CuBlock::new();
    let attn_out_gmem = gmem_act_ptr_raw(attn_out_act_slot);
    let attn_out_consumed = page_consumed_sem(attn_out_p);
    storer.push(tk20::group_wait::<1>(&attn_out_done, storer_phase));
    storer.push(CuStmt::new(format!(
        "{{ \
         __nv_bfloat16* __out_src = reinterpret_cast<__nv_bfloat16*>(\
             ss.pages[{out_id}]); \
         kittens::group<1>::tma::store_async(\
             reinterpret_cast<void*>({out_gmem}), \
             reinterpret_cast<void*>(__out_src), \
             {bytes}); \
         }}",
        out_id = attn_out_page_id,
        out_gmem = attn_out_gmem.expr(),
        bytes = final_out_bytes,
    )));
    storer.push(tk20::group_tma_store_async_wait::<1>());
    storer.push(tk20::group_arrive::<1>(&attn_out_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}
