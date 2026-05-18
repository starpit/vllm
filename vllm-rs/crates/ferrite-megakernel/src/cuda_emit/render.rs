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
    F32, RtRow, gmem_act_ptr_raw, gmem_barrier_slot_ptr, gmem_input_ids, gmem_positions,
    gmem_weight_ptr_raw, gmem_weight_ptr_raw_offset, page_as_byte_ptr, page_as_st_bf,
    page_as_sv_bf, page_consumed_sem, page_done_sem, page_ready_sem, scratch_as,
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

    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &act_rv, &in_smem,
    ));

    consumer.push(tk20::warp_copy_rv::<F32, K_PER_WARP>(&sq_rv, &act_rv));
    consumer.push(tk20::warp_mul_rv_rv::<K_PER_WARP>(&sq_rv, &sq_rv, &sq_rv));
    let (decl_partial, partial_sum_expr) = tk20::decl_local_f32("__rms_partial_sum", "0.0f");
    consumer.push(decl_partial);
    consumer.push(tk20::warp_sum_to_scalar_f32::<K_PER_WARP>(&partial_sum_expr, &sq_rv));

    let (decl_full, full_sum_expr) = tk20::decl_local_f32("__rms_full_sum", "0.0f");
    consumer.push(decl_full);
    consumer.push(tk20::cross_warp_reduce_sum_f32::<NCW>(
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
    consumer.push(decl_scale);

    consumer.push(tk20::warp_mul_rv_scalar_f32::<K_PER_WARP>(&act_rv, &act_rv, &scale_expr));
    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &weight_rv,
        &weight_smem,
    ));
    consumer.push(tk20::warp_mul_rv_rv::<K_PER_WARP>(&act_rv, &act_rv, &weight_rv));

    consumer.push(tk20::group_store_rv_to_sv_f32_to_bf16::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &in_smem, &act_rv,
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
    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &delta_rv, &delta_smem,
    ));
    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &res_rv, &residual_smem,
    ));
    consumer.push(tk20::warp_add_rv_rv::<K_PER_WARP>(&res_rv, &res_rv, &delta_rv));
    consumer.push(tk20::group_store_rv_to_sv_f32_to_bf16::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &residual_smem, &res_rv,
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
    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &act_rv, &in_smem,
    ));
    let scale_lit = CuExpr::new(format!("{:e}f", scale));
    consumer.push(tk20::warp_mul_rv_scalar_f32::<K_PER_WARP>(&act_rv, &act_rv, &scale_lit));
    consumer.push(tk20::group_store_rv_to_sv_f32_to_bf16::<NCW, K_PER_WARP, HIDDEN_DIM>(
        if in_place { &in_smem } else { &out_smem },
        &act_rv,
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
    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &act_rv, &in_smem,
    ));
    let lambda_body = format!("tanhf(x * (1.0f / {cap:e}f)) * {cap:e}f");
    consumer.push(tk20::warp_apply_f32_lambda::<K_PER_WARP>(&act_rv, &act_rv, &lambda_body));
    consumer.push(tk20::group_store_rv_to_sv_f32_to_bf16::<NCW, K_PER_WARP, HIDDEN_DIM>(
        if in_place { &in_smem } else { &out_smem },
        &act_rv,
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

    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &delta_rv, &delta_smem,
    ));
    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &res_rv, &residual_smem,
    ));
    consumer.push(tk20::warp_add_rv_rv::<K_PER_WARP>(&res_rv, &res_rv, &delta_rv));

    consumer.push(tk20::warp_copy_rv::<F32, K_PER_WARP>(&sq_rv, &res_rv));
    consumer.push(tk20::warp_mul_rv_rv::<K_PER_WARP>(&sq_rv, &sq_rv, &sq_rv));
    let (decl_partial, partial_sum_expr) = tk20::decl_local_f32("__farn_partial_sum", "0.0f");
    consumer.push(decl_partial);
    consumer.push(tk20::warp_sum_to_scalar_f32::<K_PER_WARP>(&partial_sum_expr, &sq_rv));

    let (decl_full, full_sum_expr) = tk20::decl_local_f32("__farn_full_sum", "0.0f");
    consumer.push(decl_full);
    consumer.push(tk20::cross_warp_reduce_sum_f32::<NCW>(
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
    consumer.push(decl_scale);
    consumer.push(tk20::warp_mul_rv_scalar_f32::<K_PER_WARP>(&res_rv, &res_rv, &scale_expr));

    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &weight_rv, &weight_smem,
    ));
    consumer.push(tk20::warp_mul_rv_rv::<K_PER_WARP>(&res_rv, &res_rv, &weight_rv));

    consumer.push(tk20::group_store_rv_to_sv_f32_to_bf16::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &residual_smem, &res_rv,
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

    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &act_rv, &in_smem,
    ));

    consumer.push(tk20::warp_copy_rv::<F32, K_PER_WARP>(&sq_rv, &act_rv));
    consumer.push(tk20::warp_mul_rv_rv::<K_PER_WARP>(&sq_rv, &sq_rv, &sq_rv));
    let (decl_partial, partial_sum_expr) = tk20::decl_local_f32("__sors_partial_sum", "0.0f");
    consumer.push(decl_partial);
    consumer.push(tk20::warp_sum_to_scalar_f32::<K_PER_WARP>(&partial_sum_expr, &sq_rv));
    let (decl_full, full_sum_expr) = tk20::decl_local_f32("__sors_full_sum", "0.0f");
    consumer.push(decl_full);
    consumer.push(tk20::cross_warp_reduce_sum_f32::<NCW>(
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
    consumer.push(decl_scale);
    consumer.push(tk20::warp_mul_rv_scalar_f32::<K_PER_WARP>(&act_rv, &act_rv, &scale_expr));

    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &weight_rv, &weight_smem,
    ));
    let offset_lit = CuExpr::new(format!("{:e}f", offset));
    consumer.push(tk20::warp_add_rv_scalar_f32::<K_PER_WARP>(&weight_rv, &weight_rv, &offset_lit));
    consumer.push(tk20::warp_mul_rv_rv::<K_PER_WARP>(&act_rv, &act_rv, &weight_rv));

    consumer.push(tk20::group_store_rv_to_sv_f32_to_bf16::<NCW, K_PER_WARP, HIDDEN_DIM>(
        &in_smem, &act_rv,
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
        return RoleBodies::skipped("Gemm");
    }

    let loader_phase = storer_phase;
    let in_p = page(in_page_id);
    let weight_p = page(weight_page_id);
    let out_p = page(out_page_id);

    let in_smem = page_as_st_bf::<M, K>(in_p);
    let out_smem = page_as_st_bf::<M, N>(out_p);
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

    let act_bytes = M * K * BF16_BYTES;
    let weight_bytes = K * N * BF16_BYTES;
    let out_bytes = M * N * BF16_BYTES;

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait::<1>(&in_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&weight_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&out_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes::<1>(&in_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf::<1, M, K>(
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

    let (decl_a, a_rt) = tk20::decl_rt_bf_row::<M, K>("__gemm_a");
    let (decl_b, b_rt) = tk20::decl_rt_bf_col::<K, TILE_N>("__gemm_b");
    let (decl_acc, acc_rt) = tk20::decl_rt_fl::<M, TILE_N>("__gemm_acc");
    consumer.push(decl_a);
    consumer.push(decl_b);
    consumer.push(decl_acc);

    let warp_idx_expr = "static_cast<int>(kittens::warpid())";
    let (decl_b_sub, b_sub) = tk20::decl_st_bf_subtile::<K, N, K, TILE_N>(
        "__gemm_b_sub", &b_tile, "0", warp_idx_expr,
    );
    let (decl_out_sub, out_sub) = tk20::decl_st_bf_subtile::<M, N, M, TILE_N>(
        "__gemm_out_sub", &out_smem, "0", warp_idx_expr,
    );
    consumer.push(decl_b_sub);
    consumer.push(decl_out_sub);

    consumer.push(tk20::warp_load_rt_from_st_bf::<_, M, K>(&a_rt, &in_smem));
    consumer.push(tk20::warp_load_rt_from_st_bf::<_, K, TILE_N>(&b_rt, &b_sub));

    consumer.push(tk20::warp_zero_rt::<F32, _, M, TILE_N>(&acc_rt));
    consumer.push(tk20::warp_mma_AB::<M, K, TILE_N>(&acc_rt, &a_rt, &b_rt, &acc_rt));

    consumer.push(tk20::warp_store_st_bf_from_rt_fl::<M, TILE_N>(&out_sub, &acc_rt));

    consumer.push(tk20::group_sync_named::<NCW>(bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive::<1>(&out_done),
        tk20::group_arrive::<1>(&in_consumed),
        tk20::group_arrive::<1>(&weight_consumed),
    ]));

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&out_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw_st_bf::<1, M, N>(
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

    let loader_phase = storer_phase;
    let in_p = page(in_page_id);
    let weight_p = page(weight_page_id);
    let residual_p = page(residual_page_id);

    let in_smem = page_as_st_bf::<M, K>(in_p);
    let residual_smem = page_as_st_bf::<M, N>(residual_p);
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

    let act_bytes = M * K * BF16_BYTES;
    let weight_bytes = K * N * BF16_BYTES;
    let residual_bytes = M * N * BF16_BYTES;

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait::<1>(&in_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&weight_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&residual_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes::<1>(&in_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf::<1, M, K>(
        &in_smem, &in_gmem, act_bytes, &in_ready,
    ));
    loader.push(tk20::group_tma_expect_bytes::<1>(&weight_ready, weight_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf::<1, K, N>(
        &b_tile, &weight_gmem, weight_bytes, &weight_ready,
    ));
    loader.push(tk20::group_tma_expect_bytes::<1>(&residual_ready, residual_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf::<1, M, N>(
        &residual_smem, &residual_gmem, residual_bytes, &residual_ready,
    ));

    let launcher = CuBlock::new();

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait::<1>(&in_ready, consumer_phase));
    consumer.push(tk20::group_wait::<1>(&weight_ready, consumer_phase));
    consumer.push(tk20::group_wait::<1>(&residual_ready, consumer_phase));

    let (decl_a, a_rt) = tk20::decl_rt_bf_row::<M, K>("__gemm_a");
    let (decl_b, b_rt) = tk20::decl_rt_bf_col::<K, TILE_N>("__gemm_b");
    let (decl_acc, acc_rt) = tk20::decl_rt_fl::<M, TILE_N>("__gemm_acc");
    consumer.push(decl_a);
    consumer.push(decl_b);
    consumer.push(decl_acc);

    let warp_idx_expr = "static_cast<int>(kittens::warpid())";
    let (decl_b_sub, b_sub) = tk20::decl_st_bf_subtile::<K, N, K, TILE_N>(
        "__gemm_b_sub", &b_tile, "0", warp_idx_expr,
    );
    let (decl_resid_sub, resid_sub) = tk20::decl_st_bf_subtile::<M, N, M, TILE_N>(
        "__gemm_resid_sub", &residual_smem, "0", warp_idx_expr,
    );
    consumer.push(decl_b_sub);
    consumer.push(decl_resid_sub);

    consumer.push(tk20::warp_load_rt_from_st_bf::<_, M, K>(&a_rt, &in_smem));
    consumer.push(tk20::warp_load_rt_from_st_bf::<_, K, TILE_N>(&b_rt, &b_sub));
    consumer.push(tk20::warp_load_rt_fl_from_st_bf::<M, TILE_N>(&acc_rt, &resid_sub));

    consumer.push(tk20::warp_mma_AB::<M, K, TILE_N>(&acc_rt, &a_rt, &b_rt, &acc_rt));

    consumer.push(tk20::warp_store_st_bf_from_rt_fl::<M, TILE_N>(&resid_sub, &acc_rt));

    consumer.push(tk20::group_sync_named::<NCW>(bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive::<1>(&residual_done),
        tk20::group_arrive::<1>(&in_consumed),
        tk20::group_arrive::<1>(&weight_consumed),
    ]));

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&residual_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw_st_bf::<1, M, N>(
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
        return RoleBodies::skipped("FusedGateUpActivateMul");
    }

    let loader_phase = storer_phase;
    let in_p = page(in_page_id);
    let weight_p = page(weight_page_id);
    let out_p = page(out_page_id);

    let in_smem = page_as_st_bf::<M, HIDDEN_DIM>(in_p);
    let out_smem = page_as_st_bf::<M, INTERMEDIATE_DIM>(out_p);
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

    let act_bytes = M * HIDDEN_DIM * BF16_BYTES;
    let out_bytes = M * INTERMEDIATE_DIM * BF16_BYTES;
    let weight_total_bytes = gate_bytes + up_bytes;

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait::<1>(&in_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&weight_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&out_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes::<1>(&in_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf::<1, M, HIDDEN_DIM>(
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

    let (decl_a, a_rt) = tk20::decl_rt_bf_row::<M, HIDDEN_DIM>("__gu_a");
    let (decl_gate_b, gate_b_rt) = tk20::decl_rt_bf_col::<HIDDEN_DIM, TILE_N>("__gu_gate_b");
    let (decl_up_b, up_b_rt) = tk20::decl_rt_bf_col::<HIDDEN_DIM, TILE_N>("__gu_up_b");
    let (decl_gate_acc, gate_acc_rt) = tk20::decl_rt_fl::<M, TILE_N>("__gu_gate_acc");
    let (decl_up_acc, up_acc_rt) = tk20::decl_rt_fl::<M, TILE_N>("__gu_up_acc");
    consumer.push(decl_a);
    consumer.push(decl_gate_b);
    consumer.push(decl_up_b);
    consumer.push(decl_gate_acc);
    consumer.push(decl_up_acc);

    let warp_idx_expr = "static_cast<int>(kittens::warpid())";
    let (decl_gate_b_sub, gate_b_sub) =
        tk20::decl_st_bf_subtile::<HIDDEN_DIM, INTERMEDIATE_DIM, HIDDEN_DIM, TILE_N>(
            "__gu_gate_b_sub", &gate_buf, "0", warp_idx_expr,
        );
    let (decl_up_b_sub, up_b_sub) =
        tk20::decl_st_bf_subtile::<HIDDEN_DIM, INTERMEDIATE_DIM, HIDDEN_DIM, TILE_N>(
            "__gu_up_b_sub", &up_buf, "0", warp_idx_expr,
        );
    let (decl_out_sub, out_sub) =
        tk20::decl_st_bf_subtile::<M, INTERMEDIATE_DIM, M, TILE_N>(
            "__gu_out_sub", &out_smem, "0", warp_idx_expr,
        );
    consumer.push(decl_gate_b_sub);
    consumer.push(decl_up_b_sub);
    consumer.push(decl_out_sub);

    consumer.push(tk20::warp_load_rt_from_st_bf::<_, M, HIDDEN_DIM>(&a_rt, &in_smem));
    consumer.push(tk20::warp_load_rt_from_st_bf::<_, HIDDEN_DIM, TILE_N>(&gate_b_rt, &gate_b_sub));
    consumer.push(tk20::warp_load_rt_from_st_bf::<_, HIDDEN_DIM, TILE_N>(&up_b_rt, &up_b_sub));
    consumer.push(tk20::warp_zero_rt::<F32, _, M, TILE_N>(&gate_acc_rt));
    consumer.push(tk20::warp_zero_rt::<F32, _, M, TILE_N>(&up_acc_rt));
    consumer.push(tk20::warp_mma_AB::<M, HIDDEN_DIM, TILE_N>(
        &gate_acc_rt, &a_rt, &gate_b_rt, &gate_acc_rt,
    ));
    consumer.push(tk20::warp_mma_AB::<M, HIDDEN_DIM, TILE_N>(
        &up_acc_rt, &a_rt, &up_b_rt, &up_acc_rt,
    ));

    let activation_lambda = match activation {
        GateUpActivation::Silu => "x * (1.0f / (1.0f + __expf(-x)))",
        GateUpActivation::Gelu => {
            "0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)))"
        }
    };
    consumer.push(tk20::warp_apply_f32_rt_lambda::<M, TILE_N>(
        &gate_acc_rt, &gate_acc_rt, activation_lambda,
    ));
    consumer.push(tk20::warp_mul_rt_rt::<M, TILE_N>(&gate_acc_rt, &gate_acc_rt, &up_acc_rt));
    consumer.push(tk20::warp_store_st_bf_from_rt_fl::<M, TILE_N>(&out_sub, &gate_acc_rt));

    consumer.push(tk20::group_sync_named::<NCW>(bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive::<1>(&out_done),
        tk20::group_arrive::<1>(&in_consumed),
        tk20::group_arrive::<1>(&weight_consumed),
    ]));

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&out_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw_st_bf::<1, M, INTERMEDIATE_DIM>(
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

    let loader_phase = storer_phase;
    let in_p = page(in_page_id);
    let delta_p_opt = delta_page_id.map(page);
    let norm_p = page(norm_weight_page_id);
    let lin_p = page(linear_weight_page_id);
    let out_p = page(out_page_id);

    let in_sv = page_as_sv_bf::<K>(in_p);
    let in_st = page_as_st_bf::<M, K>(in_p);
    let norm_weight_sv = page_as_sv_bf::<K>(norm_p);
    let out_st = page_as_st_bf::<M, N>(out_p);
    let b_tile = scratch_as_st_bf::<K, N>(off(b_tile_offset));
    let partial = scratch_as::<F32>(off(partial_offset));

    let in_ready = page_ready_sem(in_p);
    let norm_weight_ready = page_ready_sem(norm_p);
    let lin_weight_ready = page_ready_sem(lin_p);
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
    let lin_weight_bytes = K * N * BF16_BYTES;
    let out_bytes = M * N * BF16_BYTES;

    let delta_sv_opt = delta_p_opt.map(page_as_sv_bf::<K>);
    let delta_ready_opt = delta_p_opt.map(page_ready_sem);
    let delta_consumed_opt = delta_p_opt.map(page_consumed_sem);
    let delta_gmem_opt = delta_act_slot.map(gmem_act_ptr_raw);

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

    loader.push(tk20::group_tma_expect_bytes::<1>(&lin_weight_ready, lin_weight_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf::<1, K, N>(
        &b_tile, &lin_weight_gmem, lin_weight_bytes, &lin_weight_ready,
    ));

    let launcher = CuBlock::new();

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait::<1>(&in_ready, consumer_phase));
    if let Some(delta_ready) = delta_ready_opt.as_ref() {
        consumer.push(tk20::group_wait::<1>(delta_ready, consumer_phase));
    }
    consumer.push(tk20::group_wait::<1>(&norm_weight_ready, consumer_phase));
    consumer.push(tk20::group_wait::<1>(&lin_weight_ready, consumer_phase));

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
        consumer.push(tk20::warp_add_rv_rv::<K_PER_WARP>(&act_rv, &act_rv, &delta_rv));
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

    consumer.push(tk20::warp_copy_rv::<F32, K_PER_WARP>(&sq_rv, &act_rv));
    consumer.push(tk20::warp_mul_rv_rv::<K_PER_WARP>(&sq_rv, &sq_rv, &sq_rv));
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
    consumer.push(tk20::warp_mul_rv_rv::<K_PER_WARP>(&act_rv, &act_rv, &weight_rv));

    consumer.push(tk20::group_store_rv_to_sv_f32_to_bf16::<NCW, K_PER_WARP, K>(
        &in_sv, &act_rv,
    ));

    consumer.push(tk20::group_sync_named::<NCW>(bar_publish));

    let (decl_a, a_rt) = tk20::decl_rt_bf_row::<M, K>("__lmh_a");
    let (decl_b, b_rt) = tk20::decl_rt_bf_col::<K, TILE_N>("__lmh_b");
    let (decl_acc, acc_rt) = tk20::decl_rt_fl::<M, TILE_N>("__lmh_acc");
    consumer.push(decl_a);
    consumer.push(decl_b);
    consumer.push(decl_acc);

    let warp_idx_expr = "static_cast<int>(kittens::warpid())";
    let (decl_b_sub, b_sub) = tk20::decl_st_bf_subtile::<K, N, K, TILE_N>(
        "__lmh_b_sub", &b_tile, "0", warp_idx_expr,
    );
    let (decl_out_sub, out_sub) = tk20::decl_st_bf_subtile::<M, N, M, TILE_N>(
        "__lmh_out_sub", &out_st, "0", warp_idx_expr,
    );
    consumer.push(decl_b_sub);
    consumer.push(decl_out_sub);

    consumer.push(tk20::warp_load_rt_from_st_bf::<_, M, K>(&a_rt, &in_st));
    consumer.push(tk20::warp_load_rt_from_st_bf::<_, K, TILE_N>(&b_rt, &b_sub));
    consumer.push(tk20::warp_zero_rt::<F32, _, M, TILE_N>(&acc_rt));
    consumer.push(tk20::warp_mma_AB::<M, K, TILE_N>(&acc_rt, &a_rt, &b_rt, &acc_rt));
    consumer.push(tk20::warp_store_st_bf_from_rt_fl::<M, TILE_N>(&out_sub, &acc_rt));

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

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&out_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw_st_bf::<1, M, N>(
        &out_gmem, &out_st, out_bytes,
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
        return RoleBodies::skipped("FusedQkvRopeCache");
    }
    if HEAD_DIM == 0 || TILE_N % HEAD_DIM != 0 {
        return RoleBodies::skipped("FusedQkvRopeCache");
    }

    let loader_phase = storer_phase;
    let in_p = page(in_page_id);
    let qkv_weight_p = page(qkv_weight_page_id);
    let cos_sin_p = page(cos_sin_page_id);
    let q_out_p = page(q_out_page_id);
    let k_out_p = page(k_out_page_id);
    let v_out_p = page(v_out_page_id);

    let in_smem = page_as_st_bf::<M, HIDDEN_DIM>(in_p);
    let q_out_smem = page_as_st_bf::<M, Q_DIM>(q_out_p);
    let k_out_smem = page_as_st_bf::<M, KV_DIM>(k_out_p);
    let v_out_smem = page_as_st_bf::<M, KV_DIM>(v_out_p);
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

    let act_bytes = M * HIDDEN_DIM * BF16_BYTES;
    let qkv_weight_bytes = HIDDEN_DIM * QKV_N * BF16_BYTES;
    let q_out_bytes = M * Q_DIM * BF16_BYTES;
    let k_out_bytes = M * KV_DIM * BF16_BYTES;
    let v_out_bytes = M * KV_DIM * BF16_BYTES;

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait::<1>(&in_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&qkv_weight_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&cos_sin_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&q_out_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&k_out_consumed, loader_phase));
    loader.push(tk20::group_wait::<1>(&v_out_consumed, loader_phase));

    loader.push(tk20::group_tma_expect_bytes::<1>(&in_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf::<1, M, HIDDEN_DIM>(
        &in_smem, &in_gmem, act_bytes, &in_ready,
    ));

    loader.push(tk20::group_tma_expect_bytes::<1>(&qkv_weight_ready, qkv_weight_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf::<1, HIDDEN_DIM, QKV_N>(
        &qkv_b_tile, &qkv_weight_gmem, qkv_weight_bytes, &qkv_weight_ready,
    ));

    loader.push(tk20::cos_sin_per_token_gather::<HEAD_DIM, M>(
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

    let (decl_a, _a_rt) = tk20::decl_rt_bf_row::<M, HIDDEN_DIM>("__qkv_a");
    consumer.push(decl_a);
    consumer.push(CuStmt::new(format!(
        "kittens::warp::load(__qkv_a, {in_smem});",
        in_smem = in_smem.expr(),
    )));

    // Pre-loop decls: scratch ST refs for Q/K rope staging, raw
    // bf16 ptr for cos_sin gather, warp id, and the unroll pragma
    // for the per-head loop. Each is a single-line CuStmt because
    // none binds to a typed tk20 handle (refs to scratch with
    // arbitrary offset; a bf16* cast of a void page; a builtin).
    consumer.push(CuStmt::new(format!(
        "auto& __qkv_q_rope = *reinterpret_cast<kittens::st_bf<{m}, {q_dim}>*>(\
         ss.scratch + {q_rope_offset});",
        m = M,
        q_dim = Q_DIM,
    )));
    consumer.push(CuStmt::new(format!(
        "auto& __qkv_k_rope = *reinterpret_cast<kittens::st_bf<{m}, {kv_dim}>*>(\
         ss.scratch + {k_rope_offset});",
        m = M,
        kv_dim = KV_DIM,
    )));
    consumer.push(CuStmt::new(format!(
        "__nv_bfloat16* __qkv_cos_sin_ptr = reinterpret_cast<__nv_bfloat16*>(\
         ss.pages[{cos_sin_id}]);",
        cos_sin_id = cos_sin_page_id,
    )));
    consumer.push(CuStmt::new(
        "const int __qkv_warp_id = static_cast<int>(kittens::warpid());".to_string(),
    ));
    consumer.push(CuStmt::new("_Pragma(\"unroll\")".to_string()));

    // Per-head loop body: declare __qkv_col, declare __qkv_b
    // register tile, load via runtime-offset subtile, declare
    // accumulator, zero + mma_AB, then route to Q / K / V via
    // `tk20::if_else`.
    let mut loop_body = CuBlock::new();
    loop_body.push(CuStmt::new(format!(
        "const int __qkv_col = __qkv_warp_id * {tile_n} + __qkv_h * {head_dim};",
        tile_n = TILE_N,
        head_dim = HEAD_DIM,
    )));
    loop_body.push(CuStmt::new(format!(
        "kittens::rt_bf<{hidden_dim}, {head_dim}, kittens::ducks::rt_layout::col> __qkv_b;",
        hidden_dim = HIDDEN_DIM,
        head_dim = HEAD_DIM,
    )));
    // The original C++ wrapped __qkv_b_sub in a brace scope to
    // tighten its lifetime; in a for-loop body each `auto` decl is
    // already iter-local, so we drop the redundant brace.
    loop_body.push(CuStmt::new(format!(
        "auto __qkv_b_sub = ({qkv_b_tile})\
         .template subtile<{hidden_dim}, {head_dim}>(\
         int2{{0, __qkv_col / {head_dim}}});",
        qkv_b_tile = qkv_b_tile.expr(),
        hidden_dim = HIDDEN_DIM,
        head_dim = HEAD_DIM,
    )));
    loop_body.push(CuStmt::new(
        "kittens::warp::load(__qkv_b, __qkv_b_sub);".to_string(),
    ));
    loop_body.push(CuStmt::new(format!(
        "kittens::rt_fl<{m}, {head_dim}> __qkv_acc;",
        m = M,
        head_dim = HEAD_DIM,
    )));
    loop_body.push(CuStmt::new(
        "kittens::warp::zero(__qkv_acc);".to_string(),
    ));
    loop_body.push(CuStmt::new(
        "kittens::warp::mma_AB(__qkv_acc, __qkv_a, __qkv_b, __qkv_acc);".to_string(),
    ));

    // Q rope branch: stage acc → scratch ST, sync, then apply RoPE
    // device lambda reading paired half + cos_sin, store to q_out.
    let make_rope_branch = |stg_ref: &str, out_smem_expr: &str, local_expr: &str| {
        let mut block = CuBlock::new();
        block.push(CuStmt::new(format!(
            "const int __qkv_local = {local_expr};"
        )));
        block.push(CuStmt::new(format!(
            "auto __qkv_stg = {stg_ref}\
             .template subtile<{m}, {head_dim}>(\
             int2{{0, __qkv_local / {head_dim}}});",
            m = M,
            head_dim = HEAD_DIM,
        )));
        block.push(CuStmt::new(
            "kittens::warp::store(__qkv_stg, __qkv_acc);".to_string(),
        ));
        block.push(CuStmt::new("__syncwarp();".to_string()));
        block.push(CuStmt::new(
            "__nv_bfloat16* __qkv_stg_ptr = reinterpret_cast<__nv_bfloat16*>(&__qkv_stg);"
                .to_string(),
        ));
        block.push(CuStmt::new(format!(
            "kittens::rt_fl<{m}, {head_dim}> __qkv_rot;",
            m = M,
            head_dim = HEAD_DIM,
        )));
        block.push(CuStmt::new(format!(
            "kittens::warp::apply(__qkv_rot, __qkv_acc, [=] __device__ \
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
             .template subtile<{m}, {head_dim}>(\
             int2{{0, __qkv_local / {head_dim}}});",
            m = M,
            head_dim = HEAD_DIM,
        )));
        block.push(CuStmt::new(
            "kittens::warp::store(__qkv_out, __qkv_rot);".to_string(),
        ));
        block
    };

    let q_branch = make_rope_branch("__qkv_q_rope", q_out_smem.expr().as_str(), "__qkv_col");
    let k_branch = make_rope_branch(
        "__qkv_k_rope",
        k_out_smem.expr().as_str(),
        &format!("__qkv_col - {}", Q_DIM),
    );

    // V passthrough branch: store acc directly to v_out (no RoPE).
    let mut v_branch = CuBlock::new();
    v_branch.push(CuStmt::new(format!(
        "const int __qkv_local = __qkv_col - {q_dim} - {kv_dim};",
        q_dim = Q_DIM,
        kv_dim = KV_DIM,
    )));
    v_branch.push(CuStmt::new(format!(
        "auto __qkv_out = ({v_out_smem})\
         .template subtile<{m}, {head_dim}>(\
         int2{{0, __qkv_local / {head_dim}}});",
        v_out_smem = v_out_smem.expr(),
        m = M,
        head_dim = HEAD_DIM,
    )));
    v_branch.push(CuStmt::new(
        "kittens::warp::store(__qkv_out, __qkv_acc);".to_string(),
    ));

    // Compose the routing as an n-way if/else if/else chain:
    // Q, K, V fallthrough.
    let q_cond = format!("__qkv_col < {}", Q_DIM);
    let k_cond = format!("__qkv_col < {} + {}", Q_DIM, KV_DIM);
    loop_body.push(tk20::if_chain(
        &[(&q_cond, &q_branch), (&k_cond, &k_branch)],
        Some(&v_branch),
    ));

    consumer.push(tk20::for_loop(
        &format!(
            "int __qkv_h = 0; __qkv_h < {heads_per_warp}; ++__qkv_h",
            heads_per_warp = HEADS_PER_WARP,
        ),
        &loop_body,
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

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait::<1>(&q_out_done, storer_phase));
    storer.push(tk20::group_wait::<1>(&k_out_done, storer_phase));
    storer.push(tk20::group_wait::<1>(&v_out_done, storer_phase));

    storer.push(tk20::group_tma_store_async_raw_st_bf::<1, M, Q_DIM>(
        &q_out_gmem, &q_out_smem, q_out_bytes,
    ));
    storer.push(tk20::group_tma_store_async_raw_st_bf::<1, M, KV_DIM>(
        &k_out_gmem, &k_out_smem, k_out_bytes,
    ));
    storer.push(tk20::group_tma_store_async_raw_st_bf::<1, M, KV_DIM>(
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
        return RoleBodies::skipped("AttentionViaCache");
    }
    if NUM_Q_HEADS == 0 || NUM_KV_HEADS == 0 || HEAD_DIM == 0 || NCW == 0 {
        return RoleBodies::skipped("AttentionViaCache");
    }
    if NUM_Q_HEADS % NCW != 0 {
        return RoleBodies::skipped("AttentionViaCache");
    }
    if NUM_Q_HEADS % NUM_KV_HEADS != 0 {
        return RoleBodies::skipped("AttentionViaCache");
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

    // Running state: max_vec, sum_vec (rv_fl<Q_HEAD_TILE_ROWS>),
    // o_reg (rt_fl<Q_HEAD_TILE_ROWS, HEAD_DIM>).
    let (decl_max, max_rv) =
        tk20::decl_rv_fl_neg_infty::<Q_HEAD_TILE_ROWS>("__attn_max");
    let (decl_sum, sum_rv) = tk20::decl_rv_fl::<Q_HEAD_TILE_ROWS>("__attn_sum");
    let (decl_o, o_rt) =
        tk20::decl_rt_fl::<Q_HEAD_TILE_ROWS, HEAD_DIM>("__attn_o");
    q_head_body.push(decl_max);
    q_head_body.push(decl_sum);
    q_head_body.push(tk20::warp_zero_rv::<Q_HEAD_TILE_ROWS>(&sum_rv));
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
                 (__attn_seq_len - 1 - {w})) ? -CUDART_INF_F : x",
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
             ? -CUDART_INF_F : x",
            bs = BLOCK_SIZE,
        ),
    ));

    // Online softmax update.
    let (decl_new_max, new_max_rv) =
        tk20::decl_rv_fl::<Q_HEAD_TILE_ROWS>("__attn_new_max");
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
        tk20::decl_rv_fl::<Q_HEAD_TILE_ROWS>("__attn_rescale");
    p_body.push(decl_rescale);
    p_body.push(tk20::warp_sub_rv_rv::<Q_HEAD_TILE_ROWS>(
        &rescale_rv,
        &max_rv,
        &new_max_rv,
    ));
    p_body.push(tk20::warp_exp_rv::<Q_HEAD_TILE_ROWS>(&rescale_rv, &rescale_rv));
    // sum_vec = row_sum(att, rescale * sum_vec).
    p_body.push(tk20::warp_mul_rv_rv::<Q_HEAD_TILE_ROWS>(
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
    p_body.push(tk20::warp_copy_rv::<F32, Q_HEAD_TILE_ROWS>(
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
