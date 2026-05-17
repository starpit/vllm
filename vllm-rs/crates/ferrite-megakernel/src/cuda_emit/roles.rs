// SPDX-License-Identifier: Apache-2.0
//! Per-`MegaNode` role-body emit. Sprint 1: RmsNorm only.
//!
//! Each variant's emit fn pulls typed-getter values from the node
//! and assembles four [`CuBlock`](super::cu::CuBlock)s — one per
//! warp role (loader / launcher / consumer / storer). The
//! top-level [`lower_to_cuda`](super::lower_to_cuda) walks the tape
//! and concatenates each variant's role bodies into the per-role
//! sections of the final kernel.
//!
//! Sprint 1 = RmsNorm only; every other variant returns
//! [`RoleBodies::skipped`]. Per-variant emit lands one variant per
//! sprint, each variant's emitted `.cu` must compile against TK
//! 2.0 on the pod before it's "done."

use crate::ir::nodes::{
    Add, BarrierSignal, BarrierWait, Embed, FusedAddRmsNorm, FusedGateUpActivateMul,
    GateUpActivation, Gemm, LmHeadNormKind, MegaNode, RmsNorm, ScalarMul, ScalarOffsetRmsNorm,
    TanhSoftCap, TkFusedGemmAdd, TkFusedNormGemm,
};
use crate::ir::tape::TapeBudget;

use super::cu::{CuBlock, CuExpr};
use super::handles::{
    gmem_act_ptr_raw, gmem_barrier_slot_ptr, gmem_input_ids, gmem_weight_ptr_raw,
    gmem_weight_ptr_raw_offset, page_as_byte_ptr, page_as_st_bf, page_as_sv_bf,
    page_consumed_sem, page_done_sem, page_ready_sem, scratch_as, scratch_as_st_bf,
};
use super::tk20;

/// Four role-body chunks for a single MegaNode.
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

/// Per-`MegaNode` dispatch.
pub fn emit_role_bodies(node: &MegaNode, budget: TapeBudget) -> RoleBodies {
    match node {
        MegaNode::RmsNorm(n) => emit_rms_norm(n, budget),
        MegaNode::FusedQkvRopeCache(_) => RoleBodies::skipped("FusedQkvRopeCache"),
        MegaNode::Add(n) => emit_add(n, budget),
        MegaNode::FusedAddRmsNorm(n) => emit_fused_add_rms_norm(n, budget),
        MegaNode::FusedGateUpActivateMul(n) => emit_fused_gate_up_activate_mul(n, budget),
        MegaNode::Embed(n) => emit_embed(n, budget),
        MegaNode::ScalarMul(n) => emit_scalar_mul(n, budget),
        MegaNode::TanhSoftCap(n) => emit_tanh_soft_cap(n, budget),
        MegaNode::ScalarOffsetRmsNorm(n) => emit_scalar_offset_rms_norm(n, budget),
        MegaNode::Gemm(n) => emit_gemm(n, budget),
        MegaNode::TkFusedGemmAdd(n) => emit_tk_fused_gemm_add(n, budget),
        MegaNode::TkFusedNormGemm(n) => emit_tk_fused_norm_gemm(n, budget),
        MegaNode::AttentionViaCache(_) => RoleBodies::skipped("AttentionViaCache"),
        MegaNode::SpliceMmEmbeds(_) => RoleBodies::skipped("SpliceMmEmbeds"),
        MegaNode::BarrierSignal(n) => emit_barrier_signal(n),
        MegaNode::BarrierWait(n) => emit_barrier_wait(n),
    }
}

// ============================================================
// RmsNorm — in-place per-row normalization.
// ============================================================
//
// Page lifecycle:
//   in_page:     Empty → Filled (loader TMA)
//                      → Produced (consumer warp::store)
//                      → Empty (storer TMA + arrive page_consumed)
//   weight_page: Empty → Filled (loader TMA)
//                      → Empty (consumer arrive page_consumed)
//
// TK 2.0 primitives used (every call cited to its source line in
// `third_party/thunderkittens/include/`):
//   - kittens::group<1>::wait        (sync.cuh:112)
//   - kittens::group<1>::arrive      (sync.cuh:69)
//   - kittens::group<1>::tma::expect_bytes  (util/tma.cuh:18)
//   - kittens::group<1>::tma::load_async (raw)  (util/tma.cuh:72)
//   - kittens::group<1>::tma::store_async (raw) (util/tma.cuh:82)
//   - kittens::group<1>::tma::store_async_wait  (util/tma.cuh:46)
//   - kittens::group<NCW>::load(rv_fl, sv_bf)   (vec/shared_to_register.cuh:14)
//   - kittens::group<NCW>::store(sv_bf, rv_fl)  (vec/shared_to_register.cuh:101)
//   - kittens::group<NCW>::sync(int id)         (group.cuh:33)
//   - kittens::warp::copy(rv, rv)              (vec/maps.cuh:176)
//   - kittens::warp::mul(rv, rv, rv)           (vec/maps.cuh:359)
//   - kittens::warp::mul(rv, rv, scalar)       (vec/maps.cuh:54)
//   - kittens::warp::sum(scalar_out, rv)       (vec/reductions.cuh:129)
//
// Plus ferrite substrate: `SharedState<ConfigT>::pages[N]`,
// `::page_ready[N]`, `::page_done[N]`, `::page_consumed[N]`,
// `::scratch + offset`. No `ferrite::tk::*` helpers — the consumer
// body inlines the RMS-scale computation in pure TK 2.0 + raw
// `rsqrtf` / scratch access.

fn emit_rms_norm(n: &RmsNorm, budget: TapeBudget) -> RoleBodies {
    let in_page = n.in_page();
    let weight_page = n.weight_page();
    let consumer_phase = n.consumer_phase().raw();
    let storer_phase = n.storer_phase().raw();
    let loader_phase = storer_phase; // loader waits on consumed; same parity as storer.
    let layer = n.layer().raw();
    let hidden_dim = n.hidden_dim().raw();
    let num_tokens = n.num_tokens().raw();
    let in_act_slot = n.in_act_slot().raw();
    let out_act_slot = n.out_act_slot().raw();
    let weight_accessor_idx = n.weight_accessor_idx().raw();
    let bar_reduce = n.consumer_bar_reduce().raw();
    let bar_publish = n.consumer_bar_publish().raw();
    let eps_value = n.eps().raw();

    let ncw = budget.num_consumer_warps;
    debug_assert!(
        ncw > 0 && hidden_dim % ncw == 0,
        "emit_rms_norm: HIDDEN_DIM ({hidden_dim}) must be divisible by NCW ({ncw})"
    );
    let k_per_warp = hidden_dim / ncw;
    let num_layers = budget.num_layers.max(1);

    // Substrate handles.
    let in_smem = page_as_sv_bf(in_page, hidden_dim);
    let weight_smem = page_as_sv_bf(weight_page, hidden_dim);
    let in_ready = page_ready_sem(in_page);
    let weight_ready = page_ready_sem(weight_page);
    let in_done = page_done_sem(in_page);
    let in_consumed = page_consumed_sem(in_page);
    let weight_consumed = page_consumed_sem(weight_page);
    let partial = scratch_as::<super::handles::F32>(n.partial_offset());
    let in_gmem = gmem_act_ptr_raw(in_act_slot);
    let out_gmem = gmem_act_ptr_raw(out_act_slot);
    let weight_gmem = gmem_weight_ptr_raw(weight_accessor_idx, layer, num_layers);

    let bf16_size_bytes = 2_u32;
    let act_bytes = hidden_dim * num_tokens * bf16_size_bytes;
    let weight_bytes = hidden_dim * bf16_size_bytes;

    // ---------------- Loader body ----------------
    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait(1, &in_consumed, loader_phase));
    loader.push(tk20::group_wait(1, &weight_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes(1, &in_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw(
        1,
        &in_smem,
        &in_gmem,
        act_bytes,
        &in_ready,
    ));
    loader.push(tk20::group_tma_expect_bytes(1, &weight_ready, weight_bytes));
    loader.push(tk20::group_tma_load_async_raw(
        1,
        &weight_smem,
        &weight_gmem,
        weight_bytes,
        &weight_ready,
    ));

    // ---------------- Launcher body ----------------
    // RmsNorm has no launcher work.
    let launcher = CuBlock::new();

    // ---------------- Consumer body ----------------
    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait(1, &in_ready, consumer_phase));
    consumer.push(tk20::group_wait(1, &weight_ready, consumer_phase));

    // Declare per-warp register vectors.
    let (decl_act, act_rv) = tk20::decl_rv_fl("__rms_act_rv", k_per_warp);
    let (decl_sq, sq_rv) = tk20::decl_rv_fl("__rms_sq_rv", k_per_warp);
    let (decl_weight, weight_rv) = tk20::decl_rv_fl("__rms_weight_rv", k_per_warp);
    consumer.push(decl_act);
    consumer.push(decl_sq);
    consumer.push(decl_weight);

    // Cooperative load: NCW warps each get their own subvec of the
    // full HIDDEN_DIM activation row (TK 2.0 auto-slices when NCW>1).
    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32(ncw, &act_rv, &in_smem));

    // Sum-of-squares: copy + multiply + warp::sum.
    consumer.push(tk20::warp_copy_rv(&sq_rv, &act_rv));
    consumer.push(tk20::warp_mul_rv_rv(&sq_rv, &sq_rv, &sq_rv));
    let (decl_partial, partial_sum_expr) =
        tk20::decl_local_f32("__rms_partial_sum", "0.0f");
    consumer.push(decl_partial);
    consumer.push(tk20::warp_sum_to_scalar_f32(&partial_sum_expr, &sq_rv));

    // Cross-warp accumulate the per-warp partial sums + compute scale.
    let (decl_full, full_sum_expr) =
        tk20::decl_local_f32("__rms_full_sum", "0.0f");
    consumer.push(decl_full);
    consumer.push(tk20::cross_warp_reduce_sum_f32(
        full_sum_expr.as_str(),
        partial_sum_expr.as_str(),
        &partial,
        ncw,
        bar_reduce,
    ));
    let (decl_scale, scale_expr) =
        tk20::decl_rms_scale_local("__rms_scale", full_sum_expr.as_str(), hidden_dim, eps_value);
    consumer.push(decl_scale);

    // Apply scale + weight in registers.
    consumer.push(tk20::warp_mul_rv_scalar_f32(&act_rv, &act_rv, &scale_expr));
    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32(
        ncw,
        &weight_rv,
        &weight_smem,
    ));
    consumer.push(tk20::warp_mul_rv_rv(&act_rv, &act_rv, &weight_rv));

    // Write back to in_page (in-place) — auto-slices per warp.
    consumer.push(tk20::group_store_rv_to_sv_f32_to_bf16(
        ncw,
        &in_smem,
        &act_rv,
    ));

    // Cross-warp publish before warp 0 signals page_done.
    consumer.push(tk20::group_sync_named(ncw, bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive(1, &in_done),
        tk20::group_arrive(1, &weight_consumed),
    ]));

    // ---------------- Storer body ----------------
    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait(1, &in_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw(
        1,
        &out_gmem,
        &in_smem,
        act_bytes,
    ));
    storer.push(tk20::group_tma_store_async_wait(1));
    storer.push(tk20::group_arrive(1, &in_consumed));

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
//
// Page lifecycle:
//   delta_page:    Empty → Filled (loader TMA)
//                        → Empty (consumer arrive page_consumed)
//   residual_page: Empty → Filled (loader TMA)
//                        → Produced (consumer warp::store)
//                        → Empty (storer TMA + arrive page_consumed)
//
// TK 2.0 primitives used (every cited):
//   - kittens::group<1>::wait                    (sync.cuh:112)
//   - kittens::group<1>::arrive                  (sync.cuh:69)
//   - kittens::group<1>::tma::expect_bytes       (util/tma.cuh:18)
//   - kittens::group<1>::tma::load_async (raw)   (util/tma.cuh:72)
//   - kittens::group<1>::tma::store_async (raw)  (util/tma.cuh:82)
//   - kittens::group<1>::tma::store_async_wait   (util/tma.cuh:46)
//   - kittens::group<NCW>::load(rv_fl, sv_bf)    (vec/shared_to_register.cuh:14)
//   - kittens::group<NCW>::store(sv_bf, rv_fl)   (vec/shared_to_register.cuh:101)
//   - kittens::group<NCW>::sync(int id)          (group.cuh:33)
//   - kittens::warp::add(rv, rv, rv)             (vec/maps.cuh:333)
//
// No reduction, no scratch — each warp's slice is independent.
// One named-bar publish before warp 0 arrives on page_done so
// the storer doesn't TMA out a partially-written page.

fn emit_add(n: &Add, budget: TapeBudget) -> RoleBodies {
    let delta_page = n.delta_page();
    let residual_page = n.residual_page();
    let consumer_phase = n.consumer_phase().raw();
    let storer_phase = n.storer_phase().raw();
    let loader_phase = storer_phase;
    let hidden_dim = n.hidden_dim().raw();
    let num_tokens = n.num_tokens().raw();
    let delta_act_slot = n.delta_act_slot().raw();
    let residual_act_slot = n.residual_act_slot().raw();
    let bar_publish = n.consumer_bar_publish().raw();

    let ncw = budget.num_consumer_warps;
    debug_assert!(
        ncw > 0 && hidden_dim % ncw == 0,
        "emit_add: HIDDEN_DIM ({hidden_dim}) must be divisible by NCW ({ncw})"
    );
    let k_per_warp = hidden_dim / ncw;

    let delta_smem = page_as_sv_bf(delta_page, hidden_dim);
    let residual_smem = page_as_sv_bf(residual_page, hidden_dim);
    let delta_ready = page_ready_sem(delta_page);
    let residual_ready = page_ready_sem(residual_page);
    let residual_done = page_done_sem(residual_page);
    let delta_consumed = page_consumed_sem(delta_page);
    let residual_consumed = page_consumed_sem(residual_page);
    let delta_gmem = gmem_act_ptr_raw(delta_act_slot);
    let residual_gmem = gmem_act_ptr_raw(residual_act_slot);

    let bf16_size_bytes = 2_u32;
    let act_bytes = hidden_dim * num_tokens * bf16_size_bytes;

    // Loader: TMA-load delta + residual.
    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait(1, &delta_consumed, loader_phase));
    loader.push(tk20::group_wait(1, &residual_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes(1, &delta_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw(
        1,
        &delta_smem,
        &delta_gmem,
        act_bytes,
        &delta_ready,
    ));
    loader.push(tk20::group_tma_expect_bytes(1, &residual_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw(
        1,
        &residual_smem,
        &residual_gmem,
        act_bytes,
        &residual_ready,
    ));

    let launcher = CuBlock::new();

    // Consumer: load delta+residual into per-warp register vecs,
    // warp::add elementwise, store back to residual page.
    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait(1, &delta_ready, consumer_phase));
    consumer.push(tk20::group_wait(1, &residual_ready, consumer_phase));

    let (decl_delta, delta_rv) = tk20::decl_rv_fl("__add_delta_rv", k_per_warp);
    let (decl_res, res_rv) = tk20::decl_rv_fl("__add_res_rv", k_per_warp);
    consumer.push(decl_delta);
    consumer.push(decl_res);
    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32(
        ncw, &delta_rv, &delta_smem,
    ));
    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32(
        ncw,
        &res_rv,
        &residual_smem,
    ));
    consumer.push(tk20::warp_add_rv_rv(&res_rv, &res_rv, &delta_rv));
    consumer.push(tk20::group_store_rv_to_sv_f32_to_bf16(
        ncw,
        &residual_smem,
        &res_rv,
    ));
    consumer.push(tk20::group_sync_named(ncw, bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive(1, &residual_done),
        tk20::group_arrive(1, &delta_consumed),
    ]));

    // Storer: TMA-store residual back to gmem.
    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait(1, &residual_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw(
        1,
        &residual_gmem,
        &residual_smem,
        act_bytes,
    ));
    storer.push(tk20::group_tma_store_async_wait(1));
    storer.push(tk20::group_arrive(1, &residual_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}

// ============================================================
// ScalarMul — in-place per-row scale (out_smem = in_smem * scale).
// in_page == out_page is valid (gemma2 post-attn `* hidden`).
// ============================================================
//
// TK 2.0 primitives used (every cited):
//   - kittens::group<1>::wait                    (sync.cuh:112)
//   - kittens::group<1>::arrive                  (sync.cuh:69)
//   - kittens::group<1>::tma::expect_bytes       (util/tma.cuh:18)
//   - kittens::group<1>::tma::load_async (raw)   (util/tma.cuh:72)
//   - kittens::group<1>::tma::store_async (raw)  (util/tma.cuh:82)
//   - kittens::group<1>::tma::store_async_wait   (util/tma.cuh:46)
//   - kittens::group<NCW>::load(rv_fl, sv_bf)    (vec/shared_to_register.cuh:14)
//   - kittens::group<NCW>::store(sv_bf, rv_fl)   (vec/shared_to_register.cuh:101)
//   - kittens::group<NCW>::sync(int id)          (group.cuh:33)
//   - kittens::warp::mul(rv, rv, scalar)         (vec/maps.cuh:54-55)

fn emit_scalar_mul(n: &ScalarMul, budget: TapeBudget) -> RoleBodies {
    let in_page = n.in_page();
    let out_page = n.out_page();
    let consumer_phase = n.consumer_phase().raw();
    let storer_phase = n.storer_phase().raw();
    let loader_phase = storer_phase;
    let hidden_dim = n.hidden_dim().raw();
    let num_tokens = n.num_tokens().raw();
    let in_act_slot = n.in_act_slot().raw();
    let out_act_slot = n.out_act_slot().raw();
    let bar_publish = n.consumer_bar_publish().raw();
    let scale_value = n.scale.raw();
    let in_place = in_page.raw() == out_page.raw();

    let ncw = budget.num_consumer_warps;
    debug_assert!(ncw > 0 && hidden_dim % ncw == 0);
    let k_per_warp = hidden_dim / ncw;

    let in_smem = page_as_sv_bf(in_page, hidden_dim);
    let out_smem = page_as_sv_bf(out_page, hidden_dim);
    let in_ready = page_ready_sem(in_page);
    let out_done = page_done_sem(out_page);
    let in_consumed = page_consumed_sem(in_page);
    let out_consumed = page_consumed_sem(out_page);
    let in_gmem = gmem_act_ptr_raw(in_act_slot);
    let out_gmem = gmem_act_ptr_raw(out_act_slot);

    let bf16_size_bytes = 2_u32;
    let act_bytes = hidden_dim * num_tokens * bf16_size_bytes;

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait(1, &in_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes(1, &in_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw(
        1, &in_smem, &in_gmem, act_bytes, &in_ready,
    ));

    let launcher = CuBlock::new();

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait(1, &in_ready, consumer_phase));
    let (decl_act, act_rv) = tk20::decl_rv_fl("__smul_rv", k_per_warp);
    consumer.push(decl_act);
    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32(ncw, &act_rv, &in_smem));
    let scale_lit = CuExpr::new(format!("{:e}f", scale_value));
    consumer.push(tk20::warp_mul_rv_scalar_f32(&act_rv, &act_rv, &scale_lit));
    consumer.push(tk20::group_store_rv_to_sv_f32_to_bf16(
        ncw,
        if in_place { &in_smem } else { &out_smem },
        &act_rv,
    ));
    consumer.push(tk20::group_sync_named(ncw, bar_publish));
    let mut publish_stmts = vec![tk20::group_arrive(1, &out_done)];
    if !in_place {
        publish_stmts.push(tk20::group_arrive(1, &in_consumed));
    }
    consumer.push(tk20::block_warp_zero(&publish_stmts));

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait(1, &out_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw(
        1,
        &out_gmem,
        if in_place { &in_smem } else { &out_smem },
        act_bytes,
    ));
    storer.push(tk20::group_tma_store_async_wait(1));
    storer.push(tk20::group_arrive(1, &out_consumed));

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
// In-place (in_page == out_page) is valid (gemma2 lm_head softcap).
// ============================================================
//
// Same substrate shape as ScalarMul. One per-lane unary map; uses
// `kittens::warp::apply` (vec/maps.cuh:79) to splice a __device__
// lambda doing the tanh.

fn emit_tanh_soft_cap(n: &TanhSoftCap, budget: TapeBudget) -> RoleBodies {
    let in_page = n.in_page();
    let out_page = n.out_page();
    let consumer_phase = n.consumer_phase().raw();
    let storer_phase = n.storer_phase().raw();
    let loader_phase = storer_phase;
    let hidden_dim = n.hidden_dim().raw();
    let num_tokens = n.num_tokens().raw();
    let in_act_slot = n.in_act_slot().raw();
    let out_act_slot = n.out_act_slot().raw();
    let bar_publish = n.consumer_bar_publish().raw();
    let cap_value = n.cap.raw();
    let in_place = in_page.raw() == out_page.raw();

    let ncw = budget.num_consumer_warps;
    debug_assert!(ncw > 0 && hidden_dim % ncw == 0);
    let k_per_warp = hidden_dim / ncw;

    let in_smem = page_as_sv_bf(in_page, hidden_dim);
    let out_smem = page_as_sv_bf(out_page, hidden_dim);
    let in_ready = page_ready_sem(in_page);
    let out_done = page_done_sem(out_page);
    let in_consumed = page_consumed_sem(in_page);
    let out_consumed = page_consumed_sem(out_page);
    let in_gmem = gmem_act_ptr_raw(in_act_slot);
    let out_gmem = gmem_act_ptr_raw(out_act_slot);

    let bf16_size_bytes = 2_u32;
    let act_bytes = hidden_dim * num_tokens * bf16_size_bytes;

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait(1, &in_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes(1, &in_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw(
        1, &in_smem, &in_gmem, act_bytes, &in_ready,
    ));

    let launcher = CuBlock::new();

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait(1, &in_ready, consumer_phase));
    let (decl_act, act_rv) = tk20::decl_rv_fl("__tanh_rv", k_per_warp);
    consumer.push(decl_act);
    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32(ncw, &act_rv, &in_smem));

    // tanhf(x / cap) * cap. cap baked in as a constexpr literal.
    let lambda_body = format!(
        "tanhf(x * (1.0f / {cap:e}f)) * {cap:e}f",
        cap = cap_value
    );
    consumer.push(tk20::warp_apply_f32_lambda(&act_rv, &act_rv, &lambda_body));

    consumer.push(tk20::group_store_rv_to_sv_f32_to_bf16(
        ncw,
        if in_place { &in_smem } else { &out_smem },
        &act_rv,
    ));
    consumer.push(tk20::group_sync_named(ncw, bar_publish));
    let mut publish_stmts = vec![tk20::group_arrive(1, &out_done)];
    if !in_place {
        publish_stmts.push(tk20::group_arrive(1, &in_consumed));
    }
    consumer.push(tk20::block_warp_zero(&publish_stmts));

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait(1, &out_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw(
        1,
        &out_gmem,
        if in_place { &in_smem } else { &out_smem },
        act_bytes,
    ));
    storer.push(tk20::group_tma_store_async_wait(1));
    storer.push(tk20::group_arrive(1, &out_consumed));

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
//
// Page lifecycle:
//   delta_page:    Empty -> Filled (loader) -> Empty (consumer arrive)
//   residual_page: Empty -> Filled (loader) -> Produced (consumer store)
//                                           -> Empty (storer + arrive)
//   weight_page:   Empty -> Filled (loader) -> Empty (consumer arrive)
//
// Same TK 2.0 surface as RmsNorm + extra warp::add for the
// residual fold. No new TK 2.0 primitive.

fn emit_fused_add_rms_norm(n: &FusedAddRmsNorm, budget: TapeBudget) -> RoleBodies {
    let delta_page = n.delta_page();
    let residual_page = n.residual_page();
    let weight_page = n.weight_page();
    let consumer_phase = n.consumer_phase().raw();
    let storer_phase = n.storer_phase().raw();
    let loader_phase = storer_phase;
    let layer = n.layer().raw();
    let hidden_dim = n.hidden_dim().raw();
    let num_tokens = n.num_tokens().raw();
    let delta_act_slot = n.delta_act_slot().raw();
    let residual_act_slot = n.residual_act_slot().raw();
    let weight_accessor_idx = n.weight_accessor_idx().raw();
    let bar_reduce = n.consumer_bar_reduce().raw();
    let bar_publish = n.consumer_bar_publish().raw();
    let eps_value = n.eps().raw();

    let ncw = budget.num_consumer_warps;
    debug_assert!(ncw > 0 && hidden_dim % ncw == 0);
    let k_per_warp = hidden_dim / ncw;
    let num_layers = budget.num_layers.max(1);

    let delta_smem = page_as_sv_bf(delta_page, hidden_dim);
    let residual_smem = page_as_sv_bf(residual_page, hidden_dim);
    let weight_smem = page_as_sv_bf(weight_page, hidden_dim);
    let delta_ready = page_ready_sem(delta_page);
    let residual_ready = page_ready_sem(residual_page);
    let weight_ready = page_ready_sem(weight_page);
    let residual_done = page_done_sem(residual_page);
    let delta_consumed = page_consumed_sem(delta_page);
    let residual_consumed = page_consumed_sem(residual_page);
    let weight_consumed = page_consumed_sem(weight_page);
    let partial = scratch_as::<super::handles::F32>(n.partial_offset());
    let delta_gmem = gmem_act_ptr_raw(delta_act_slot);
    let residual_gmem = gmem_act_ptr_raw(residual_act_slot);
    let weight_gmem = gmem_weight_ptr_raw(weight_accessor_idx, layer, num_layers);

    let bf16_size_bytes = 2_u32;
    let act_bytes = hidden_dim * num_tokens * bf16_size_bytes;
    let weight_bytes = hidden_dim * bf16_size_bytes;

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait(1, &delta_consumed, loader_phase));
    loader.push(tk20::group_wait(1, &residual_consumed, loader_phase));
    loader.push(tk20::group_wait(1, &weight_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes(1, &delta_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw(
        1, &delta_smem, &delta_gmem, act_bytes, &delta_ready,
    ));
    loader.push(tk20::group_tma_expect_bytes(1, &residual_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw(
        1, &residual_smem, &residual_gmem, act_bytes, &residual_ready,
    ));
    loader.push(tk20::group_tma_expect_bytes(1, &weight_ready, weight_bytes));
    loader.push(tk20::group_tma_load_async_raw(
        1, &weight_smem, &weight_gmem, weight_bytes, &weight_ready,
    ));

    let launcher = CuBlock::new();

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait(1, &delta_ready, consumer_phase));
    consumer.push(tk20::group_wait(1, &residual_ready, consumer_phase));
    consumer.push(tk20::group_wait(1, &weight_ready, consumer_phase));

    let (decl_delta, delta_rv) = tk20::decl_rv_fl("__farn_delta_rv", k_per_warp);
    let (decl_res, res_rv) = tk20::decl_rv_fl("__farn_res_rv", k_per_warp);
    let (decl_sq, sq_rv) = tk20::decl_rv_fl("__farn_sq_rv", k_per_warp);
    let (decl_w, weight_rv) = tk20::decl_rv_fl("__farn_weight_rv", k_per_warp);
    consumer.push(decl_delta);
    consumer.push(decl_res);
    consumer.push(decl_sq);
    consumer.push(decl_w);

    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32(ncw, &delta_rv, &delta_smem));
    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32(ncw, &res_rv, &residual_smem));
    consumer.push(tk20::warp_add_rv_rv(&res_rv, &res_rv, &delta_rv));

    consumer.push(tk20::warp_copy_rv(&sq_rv, &res_rv));
    consumer.push(tk20::warp_mul_rv_rv(&sq_rv, &sq_rv, &sq_rv));
    let (decl_partial, partial_sum_expr) =
        tk20::decl_local_f32("__farn_partial_sum", "0.0f");
    consumer.push(decl_partial);
    consumer.push(tk20::warp_sum_to_scalar_f32(&partial_sum_expr, &sq_rv));

    let (decl_full, full_sum_expr) =
        tk20::decl_local_f32("__farn_full_sum", "0.0f");
    consumer.push(decl_full);
    consumer.push(tk20::cross_warp_reduce_sum_f32(
        full_sum_expr.as_str(),
        partial_sum_expr.as_str(),
        &partial,
        ncw,
        bar_reduce,
    ));
    let (decl_scale, scale_expr) = tk20::decl_rms_scale_local(
        "__farn_scale",
        full_sum_expr.as_str(),
        hidden_dim,
        eps_value,
    );
    consumer.push(decl_scale);
    consumer.push(tk20::warp_mul_rv_scalar_f32(&res_rv, &res_rv, &scale_expr));

    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32(ncw, &weight_rv, &weight_smem));
    consumer.push(tk20::warp_mul_rv_rv(&res_rv, &res_rv, &weight_rv));

    consumer.push(tk20::group_store_rv_to_sv_f32_to_bf16(ncw, &residual_smem, &res_rv));

    consumer.push(tk20::group_sync_named(ncw, bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive(1, &residual_done),
        tk20::group_arrive(1, &delta_consumed),
        tk20::group_arrive(1, &weight_consumed),
    ]));

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait(1, &residual_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw(
        1, &residual_gmem, &residual_smem, act_bytes,
    ));
    storer.push(tk20::group_tma_store_async_wait(1));
    storer.push(tk20::group_arrive(1, &residual_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}


// ============================================================
// ScalarOffsetRmsNorm — out = (act * scale) * (weight + offset)
// ============================================================

fn emit_scalar_offset_rms_norm(
    n: &ScalarOffsetRmsNorm,
    budget: TapeBudget,
) -> RoleBodies {
    let in_page = n.in_page();
    let weight_page = n.weight_page();
    let consumer_phase = n.consumer_phase().raw();
    let storer_phase = n.storer_phase().raw();
    let loader_phase = storer_phase;
    let layer = n.layer().raw();
    let hidden_dim = n.hidden_dim().raw();
    let num_tokens = n.num_tokens().raw();
    let in_act_slot = n.in_act_slot().raw();
    let out_act_slot = n.out_act_slot().raw();
    let weight_accessor_idx = n.weight_accessor_idx().raw();
    let bar_reduce = n.consumer_bar_reduce().raw();
    let bar_publish = n.consumer_bar_publish().raw();
    let eps_value = n.eps().raw();
    let offset_value = n.offset.raw();

    let ncw = budget.num_consumer_warps;
    debug_assert!(ncw > 0 && hidden_dim % ncw == 0);
    let k_per_warp = hidden_dim / ncw;
    let num_layers = budget.num_layers.max(1);

    let in_smem = page_as_sv_bf(in_page, hidden_dim);
    let weight_smem = page_as_sv_bf(weight_page, hidden_dim);
    let in_ready = page_ready_sem(in_page);
    let weight_ready = page_ready_sem(weight_page);
    let in_done = page_done_sem(in_page);
    let in_consumed = page_consumed_sem(in_page);
    let weight_consumed = page_consumed_sem(weight_page);
    let partial = scratch_as::<super::handles::F32>(n.partial_offset());
    let in_gmem = gmem_act_ptr_raw(in_act_slot);
    let out_gmem = gmem_act_ptr_raw(out_act_slot);
    let weight_gmem = gmem_weight_ptr_raw(weight_accessor_idx, layer, num_layers);

    let bf16_size_bytes = 2_u32;
    let act_bytes = hidden_dim * num_tokens * bf16_size_bytes;
    let weight_bytes = hidden_dim * bf16_size_bytes;

    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait(1, &in_consumed, loader_phase));
    loader.push(tk20::group_wait(1, &weight_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes(1, &in_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw(
        1, &in_smem, &in_gmem, act_bytes, &in_ready,
    ));
    loader.push(tk20::group_tma_expect_bytes(1, &weight_ready, weight_bytes));
    loader.push(tk20::group_tma_load_async_raw(
        1, &weight_smem, &weight_gmem, weight_bytes, &weight_ready,
    ));

    let launcher = CuBlock::new();

    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait(1, &in_ready, consumer_phase));
    consumer.push(tk20::group_wait(1, &weight_ready, consumer_phase));

    let (decl_act, act_rv) = tk20::decl_rv_fl("__sors_act_rv", k_per_warp);
    let (decl_sq, sq_rv) = tk20::decl_rv_fl("__sors_sq_rv", k_per_warp);
    let (decl_w, weight_rv) = tk20::decl_rv_fl("__sors_weight_rv", k_per_warp);
    consumer.push(decl_act);
    consumer.push(decl_sq);
    consumer.push(decl_w);

    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32(ncw, &act_rv, &in_smem));

    consumer.push(tk20::warp_copy_rv(&sq_rv, &act_rv));
    consumer.push(tk20::warp_mul_rv_rv(&sq_rv, &sq_rv, &sq_rv));
    let (decl_partial, partial_sum_expr) =
        tk20::decl_local_f32("__sors_partial_sum", "0.0f");
    consumer.push(decl_partial);
    consumer.push(tk20::warp_sum_to_scalar_f32(&partial_sum_expr, &sq_rv));
    let (decl_full, full_sum_expr) =
        tk20::decl_local_f32("__sors_full_sum", "0.0f");
    consumer.push(decl_full);
    consumer.push(tk20::cross_warp_reduce_sum_f32(
        full_sum_expr.as_str(),
        partial_sum_expr.as_str(),
        &partial,
        ncw,
        bar_reduce,
    ));
    let (decl_scale, scale_expr) = tk20::decl_rms_scale_local(
        "__sors_scale",
        full_sum_expr.as_str(),
        hidden_dim,
        eps_value,
    );
    consumer.push(decl_scale);
    consumer.push(tk20::warp_mul_rv_scalar_f32(&act_rv, &act_rv, &scale_expr));

    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32(ncw, &weight_rv, &weight_smem));
    let offset_lit = CuExpr::new(format!("{:e}f", offset_value));
    consumer.push(tk20::warp_add_rv_scalar_f32(&weight_rv, &weight_rv, &offset_lit));
    consumer.push(tk20::warp_mul_rv_rv(&act_rv, &act_rv, &weight_rv));

    consumer.push(tk20::group_store_rv_to_sv_f32_to_bf16(ncw, &in_smem, &act_rv));

    consumer.push(tk20::group_sync_named(ncw, bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive(1, &in_done),
        tk20::group_arrive(1, &weight_consumed),
    ]));

    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait(1, &in_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw(
        1, &out_gmem, &in_smem, act_bytes,
    ));
    storer.push(tk20::group_tma_store_async_wait(1));
    storer.push(tk20::group_arrive(1, &in_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}


// ============================================================
// BarrierSignal / BarrierWait — cross-CTA gmem barriers via
// `ferrite::barrier_signal/wait` (`ferrite_barrier.cuh`).
// ============================================================
//
// These ops insert a single line into the LOADER body — the
// canonical placement for cross-CTA sync points (where gmem
// reads cluster). The other 3 role bodies are empty.

fn emit_barrier_signal(n: &BarrierSignal) -> RoleBodies {
    let edge = n.edge().raw();
    let slot_ptr = gmem_barrier_slot_ptr(edge);
    let mut loader = CuBlock::new();
    loader.push(tk20::ferrite_barrier_signal(&slot_ptr, 1));
    RoleBodies {
        loader,
        launcher: CuBlock::new(),
        consumer: CuBlock::new(),
        storer: CuBlock::new(),
        skipped: None,
    }
}

fn emit_barrier_wait(n: &BarrierWait) -> RoleBodies {
    let edge = n.edge().raw();
    let expected = n.expected().raw();
    let slot_ptr = gmem_barrier_slot_ptr(edge);
    let mut loader = CuBlock::new();
    loader.push(tk20::ferrite_barrier_wait(&slot_ptr, expected));
    RoleBodies {
        loader,
        launcher: CuBlock::new(),
        consumer: CuBlock::new(),
        storer: CuBlock::new(),
        skipped: None,
    }
}


// ============================================================
// Embed — per-token vocab table gather.
// ============================================================
//
// out_page holds NUM_TOKENS contiguous rows of HIDDEN_DIM bf16
// elements. Loader does a per-token TMA gather from the embed
// table at row `input_ids[t]`. Consumer is a passthrough (no
// compute). Storer TMA-stores the contiguous rows back to gmem
// at out_act_slot.
//
// embed_weight_page is in the IR but unused by this emit — the
// embed table is read directly from gmem (via weight_ptrs[acc *
// NUM_LAYERS + 0]); no shared-memory landing required.

fn emit_embed(n: &Embed, budget: TapeBudget) -> RoleBodies {
    let out_page = n.out_page();
    let consumer_phase = n.consumer_phase().raw();
    let storer_phase = n.storer_phase().raw();
    let loader_phase = storer_phase;
    let hidden_dim = n.hidden_dim().raw();
    let num_tokens = n.num_tokens().raw();
    let out_act_slot = n.out_act_slot().raw();
    let weight_accessor_idx = n.weight_accessor_idx().raw();

    let num_layers = budget.num_layers.max(1);
    // Embed table is layer-0 only (IR doc: "LAYER is always 0 for
    // Embed").
    let layer = 0_u32;

    let out_byte = page_as_byte_ptr(out_page);
    let out_ready = page_ready_sem(out_page);
    let out_done = page_done_sem(out_page);
    let out_consumed = page_consumed_sem(out_page);
    let embed_table = gmem_weight_ptr_raw(weight_accessor_idx, layer, num_layers);
    let input_ids = gmem_input_ids();
    let out_gmem = gmem_act_ptr_raw(out_act_slot);

    // Loader: wait for page_consumed, expect_bytes for total, then
    // per-token gather.
    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait(1, &out_consumed, loader_phase));
    loader.push(tk20::embed_per_token_gather(
        &out_byte,
        &embed_table,
        &input_ids,
        hidden_dim,
        num_tokens,
        &out_ready,
    ));

    let launcher = CuBlock::new();

    // Consumer: passthrough. Wait, then warp 0 publishes done.
    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait(1, &out_ready, consumer_phase));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive(1, &out_done),
    ]));

    // Storer: per-token TMA-store back to gmem at out_act_slot.
    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait(1, &out_done, storer_phase));
    storer.push(tk20::per_token_tma_store(
        &out_gmem,
        &out_byte,
        hidden_dim,
        num_tokens,
    ));
    storer.push(tk20::group_tma_store_async_wait(1));
    storer.push(tk20::group_arrive(1, &out_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}


// ============================================================
// Gemm — `D = A * B + C` with C zero-init (pure matmul). AlongN
// warp split: each consumer warp owns `[M, TILE_N]` output cols
// of out_smem. ITERS=1 path only (multi-iter b_tile pipelining
// is a future sprint and the handoff acknowledges the IR-side
// design is open). For ITERS != 1 the variant skips emit.
// ============================================================
//
// Page lifecycle:
//   in_page:     Empty -> Filled (loader TMA `[M, K]` bf16)
//                       -> Empty (consumer warp 0 arrive `page_consumed`)
//   weight_page: Empty -> Filled-marker only (loader TMA writes
//                       to scratch b_tile but uses weight_page's
//                       page_ready as the b_tile-ready signal).
//                       -> Empty (consumer warp 0 arrive `page_consumed`)
//   out_page:    Empty -> Produced (consumer per-warp store of
//                       `[M, TILE_N]` accumulator slice)
//                       -> Empty (storer TMA + arrive page_consumed)
//
// TK 2.0 primitives used (every call cited):
//   - kittens::group<1>::wait                       (sync.cuh:112)
//   - kittens::group<1>::arrive                     (sync.cuh:69)
//   - kittens::group<1>::tma::expect_bytes          (util/tma.cuh:18)
//   - kittens::group<1>::tma::load_async (raw)      (util/tma.cuh:72)
//   - kittens::group<1>::tma::store_async (raw)     (util/tma.cuh:82)
//   - kittens::group<1>::tma::store_async_wait      (util/tma.cuh:46)
//   - kittens::group<NCW>::sync(int id)             (group.cuh:33)
//   - kittens::warp::load(rt, st)                   (memory/tile/shared_to_register.cuh:14)
//   - kittens::warp::store(st, rt)                  (memory/tile/shared_to_register.cuh:138)
//   - kittens::warp::zero(rt)                       (register/tile/maps.cuh:421)
//   - kittens::warp::mma_AB(D, A, B, C)             (mma/warp.cuh:583)
//   - st_bf<...>::subtile<R, C>(int2{...})          (shared/st.cuh:152)

fn emit_gemm(node: &Gemm, budget: TapeBudget) -> RoleBodies {
    let in_page = node.in_page();
    let weight_page = node.weight_page();
    let out_page = node.out_page();
    let consumer_phase = node.consumer_phase().raw();
    let storer_phase = node.storer_phase().raw();
    let loader_phase = storer_phase;
    let iters = node.iters().raw();
    let layer = node.layer().raw();
    let n_dim = node.n().raw();
    let k_dim = node.k().raw();
    let m_dim = node.m().raw();
    let tile_n = node.tile_n().raw();
    let chunk_k = node.chunk_k().raw();
    let in_act_slot = node.in_act_slot().raw();
    let out_act_slot = node.out_act_slot().raw();
    let weight_accessor_idx = node.weight_accessor_idx().raw();
    let bar_publish = node.consumer_bar_publish().raw();
    let b_tile_offset = node.b_tile_offset();

    let ncw = budget.num_consumer_warps;
    let num_layers = budget.num_layers.max(1);

    // Sprint 10 limit: single-shot b_tile only. Multi-iter
    // pipelining requires per-iter mbarrier phases that the IR
    // doesn't yet model.
    if iters != 1 {
        return RoleBodies::skipped("Gemm");
    }

    debug_assert_eq!(
        chunk_k, k_dim,
        "emit_gemm: ITERS=1 requires CHUNK_K ({chunk_k}) == K ({k_dim})"
    );
    debug_assert_eq!(
        tile_n * ncw,
        n_dim,
        "emit_gemm: AlongN split requires TILE_N ({tile_n}) * NCW ({ncw}) == N ({n_dim})"
    );
    debug_assert!(
        m_dim % 16 == 0,
        "emit_gemm: TK 2.0 mma_AB requires M ({m_dim}) divisible by 16"
    );
    debug_assert!(
        k_dim % 16 == 0,
        "emit_gemm: TK 2.0 mma_AB requires K ({k_dim}) divisible by 16"
    );
    debug_assert!(
        tile_n % 16 == 0,
        "emit_gemm: TK 2.0 mma_AB requires TILE_N ({tile_n}) divisible by 16"
    );

    // Substrate handles.
    let in_smem = page_as_st_bf(in_page, m_dim, k_dim); // [M, K]
    let out_smem = page_as_st_bf(out_page, m_dim, n_dim); // [M, N]
    let b_tile = scratch_as_st_bf(b_tile_offset, k_dim, n_dim); // [K, N]

    let in_ready = page_ready_sem(in_page);
    let weight_ready = page_ready_sem(weight_page);
    let out_done = page_done_sem(out_page);
    let in_consumed = page_consumed_sem(in_page);
    let weight_consumed = page_consumed_sem(weight_page);
    let out_consumed = page_consumed_sem(out_page);

    let in_gmem = gmem_act_ptr_raw(in_act_slot);
    let out_gmem = gmem_act_ptr_raw(out_act_slot);
    let weight_gmem = gmem_weight_ptr_raw(weight_accessor_idx, layer, num_layers);

    let bf16 = 2_u32;
    let act_bytes = m_dim * k_dim * bf16; // [M, K]
    let weight_bytes = k_dim * n_dim * bf16; // [K, N]
    let out_bytes = m_dim * n_dim * bf16; // [M, N]

    // ---------------- Loader body ----------------
    // TMA-load activation into in_smem and the full weight tile
    // into b_tile (lives in scratch). The weight_page semaphore
    // gates the b_tile-ready handshake.
    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait(1, &in_consumed, loader_phase));
    loader.push(tk20::group_wait(1, &weight_consumed, loader_phase));
    loader.push(tk20::group_wait(1, &out_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes(1, &in_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf(
        1, &in_smem, &in_gmem, act_bytes, &in_ready,
    ));
    loader.push(tk20::group_tma_expect_bytes(1, &weight_ready, weight_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf(
        1,
        &b_tile,
        &weight_gmem,
        weight_bytes,
        &weight_ready,
    ));

    // ---------------- Launcher body ----------------
    // No per-iter pipelining at ITERS=1 — launcher idle.
    let launcher = CuBlock::new();

    // ---------------- Consumer body ----------------
    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait(1, &in_ready, consumer_phase));
    consumer.push(tk20::group_wait(1, &weight_ready, consumer_phase));

    // Per-warp register tiles. AlongN: each warp owns
    // `[M, TILE_N]` output cols. A is loaded full `[M, K]` per
    // warp (every warp reads the same activation); B is the
    // per-warp `[K, TILE_N]` slice from b_tile.
    let (decl_a, a_rt) = tk20::decl_rt_bf_row("__gemm_a", m_dim, k_dim);
    let (decl_b, b_rt) = tk20::decl_rt_bf_col("__gemm_b", k_dim, tile_n);
    let (decl_acc, acc_rt) = tk20::decl_rt_fl("__gemm_acc", m_dim, tile_n);
    consumer.push(decl_a);
    consumer.push(decl_b);
    consumer.push(decl_acc);

    // Per-warp B subtile + OUT subtile (col-direction slice at
    // index = warpid()). Declared as named `auto` locals so the
    // st_subtile materializes as a non-const lvalue —
    // `kittens::warp::store(ST&, ...)` requires it.
    let warp_idx_expr = "static_cast<int>(kittens::warpid())";
    let (decl_b_sub, b_sub) =
        tk20::decl_st_bf_subtile("__gemm_b_sub", &b_tile, k_dim, tile_n, "0", warp_idx_expr);
    let (decl_out_sub, out_sub) = tk20::decl_st_bf_subtile(
        "__gemm_out_sub",
        &out_smem,
        m_dim,
        tile_n,
        "0",
        warp_idx_expr,
    );
    consumer.push(decl_b_sub);
    consumer.push(decl_out_sub);

    // Load operands.
    consumer.push(tk20::warp_load_rt_from_st_bf(&a_rt, &in_smem));
    consumer.push(tk20::warp_load_rt_from_st_bf(&b_rt, &b_sub));

    // Zero accumulator and execute mma.
    consumer.push(tk20::warp_zero_rt(&acc_rt));
    consumer.push(tk20::warp_mma_AB(&acc_rt, &a_rt, &b_rt, &acc_rt));

    // Store accumulator to per-warp slice of out_smem.
    consumer.push(tk20::warp_store_st_bf_from_rt_fl(&out_sub, &acc_rt));

    // Cross-warp publish before warp 0 signals page_done.
    consumer.push(tk20::group_sync_named(ncw, bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive(1, &out_done),
        tk20::group_arrive(1, &in_consumed),
        tk20::group_arrive(1, &weight_consumed),
    ]));

    // ---------------- Storer body ----------------
    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait(1, &out_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw_st_bf(
        1,
        &out_gmem,
        &out_smem,
        out_bytes,
    ));
    storer.push(tk20::group_tma_store_async_wait(1));
    storer.push(tk20::group_arrive(1, &out_consumed));

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
// Mirror of `emit_gemm` (Sprint 10) with the output writing back
// to the residual page instead of a separate out page.
// ============================================================
//
// Page lifecycle:
//   in_page:       Empty → Filled (loader TMA in)
//                        → Empty (consumer arrive page_consumed)
//   weight_page:   Empty → Filled (loader TMA b_tile)
//                        → Empty (consumer arrive page_consumed)
//   residual_page: Empty → Filled (loader TMA residual_smem)
//                        → Produced (consumer warp::store of acc)
//                        → Empty (storer TMA residual back to gmem
//                                + arrive page_consumed)
//
// The residual page is BOTH read AND written: the loader stages
// the pre-add residual, the consumer loads it into the fp32
// accumulator (TK 2.0's internal bf16->fp32 convertor handles the
// dtype conversion), uses it as the `C` operand of `mma_AB` so
// `acc = A*B + residual` in a single mma, then stores `acc` back
// to the same shared slice; the storer TMA-flushes the updated
// residual back to its gmem slot (in-place). No separate out page.

fn emit_tk_fused_gemm_add(node: &TkFusedGemmAdd, budget: TapeBudget) -> RoleBodies {
    let in_page = node.in_page();
    let weight_page = node.weight_page();
    let residual_page = node.residual_page();
    let consumer_phase = node.consumer_phase().raw();
    let storer_phase = node.storer_phase().raw();
    let loader_phase = storer_phase;
    let iters = node.iters().raw();
    let layer = node.layer().raw();
    let n_dim = node.n().raw();
    let k_dim = node.k().raw();
    // M = num_tokens for TkFusedGemmAdd (mirrors S10 Gemm's `m`).
    let m_dim = node.num_tokens().raw();
    let tile_n = node.tile_n().raw();
    let chunk_k = node.chunk_k().raw();
    let in_act_slot = node.in_act_slot().raw();
    let residual_act_slot = node.residual_act_slot().raw();
    let weight_accessor_idx = node.weight_accessor_idx().raw();
    let bar_publish = node.consumer_bar_publish().raw();
    let b_tile_offset = node.b_tile_offset();

    let ncw = budget.num_consumer_warps;
    let num_layers = budget.num_layers.max(1);

    // Sprint 11 limit: single-shot b_tile only. Multi-iter
    // pipelining requires per-iter mbarrier phases that the IR
    // doesn't yet model. Mirrors S10 emit_gemm.
    if iters != 1 {
        return RoleBodies::skipped("TkFusedGemmAdd");
    }

    debug_assert_eq!(
        chunk_k, k_dim,
        "emit_tk_fused_gemm_add: ITERS=1 requires CHUNK_K ({chunk_k}) == K ({k_dim})"
    );
    debug_assert_eq!(
        tile_n * ncw,
        n_dim,
        "emit_tk_fused_gemm_add: AlongN split requires TILE_N ({tile_n}) * NCW ({ncw}) == N ({n_dim})"
    );
    debug_assert!(
        m_dim % 16 == 0,
        "emit_tk_fused_gemm_add: TK 2.0 mma_AB requires M ({m_dim}) divisible by 16"
    );
    debug_assert!(
        k_dim % 16 == 0,
        "emit_tk_fused_gemm_add: TK 2.0 mma_AB requires K ({k_dim}) divisible by 16"
    );
    debug_assert!(
        tile_n % 16 == 0,
        "emit_tk_fused_gemm_add: TK 2.0 mma_AB requires TILE_N ({tile_n}) divisible by 16"
    );

    let in_smem = page_as_st_bf(in_page, m_dim, k_dim); // [M, K]
    let residual_smem = page_as_st_bf(residual_page, m_dim, n_dim); // [M, N]
    let b_tile = scratch_as_st_bf(b_tile_offset, k_dim, n_dim); // [K, N]

    let in_ready = page_ready_sem(in_page);
    let weight_ready = page_ready_sem(weight_page);
    let residual_ready = page_ready_sem(residual_page);
    let residual_done = page_done_sem(residual_page);
    let in_consumed = page_consumed_sem(in_page);
    let weight_consumed = page_consumed_sem(weight_page);
    let residual_consumed = page_consumed_sem(residual_page);

    let in_gmem = gmem_act_ptr_raw(in_act_slot);
    let residual_gmem = gmem_act_ptr_raw(residual_act_slot);
    let weight_gmem = gmem_weight_ptr_raw(weight_accessor_idx, layer, num_layers);

    let bf16 = 2_u32;
    let act_bytes = m_dim * k_dim * bf16; // [M, K]
    let weight_bytes = k_dim * n_dim * bf16; // [K, N]
    let residual_bytes = m_dim * n_dim * bf16; // [M, N]

    // ---------------- Loader body ----------------
    // Three TMA loads: activation, weight tile, residual.
    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait(1, &in_consumed, loader_phase));
    loader.push(tk20::group_wait(1, &weight_consumed, loader_phase));
    loader.push(tk20::group_wait(1, &residual_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes(1, &in_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf(
        1, &in_smem, &in_gmem, act_bytes, &in_ready,
    ));
    loader.push(tk20::group_tma_expect_bytes(1, &weight_ready, weight_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf(
        1,
        &b_tile,
        &weight_gmem,
        weight_bytes,
        &weight_ready,
    ));
    loader.push(tk20::group_tma_expect_bytes(
        1,
        &residual_ready,
        residual_bytes,
    ));
    loader.push(tk20::group_tma_load_async_raw_st_bf(
        1,
        &residual_smem,
        &residual_gmem,
        residual_bytes,
        &residual_ready,
    ));

    // ---------------- Launcher body ----------------
    // No per-iter pipelining at ITERS=1 — launcher idle.
    let launcher = CuBlock::new();

    // ---------------- Consumer body ----------------
    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait(1, &in_ready, consumer_phase));
    consumer.push(tk20::group_wait(1, &weight_ready, consumer_phase));
    consumer.push(tk20::group_wait(1, &residual_ready, consumer_phase));

    let (decl_a, a_rt) = tk20::decl_rt_bf_row("__gemm_a", m_dim, k_dim);
    let (decl_b, b_rt) = tk20::decl_rt_bf_col("__gemm_b", k_dim, tile_n);
    let (decl_acc, acc_rt) = tk20::decl_rt_fl("__gemm_acc", m_dim, tile_n);
    consumer.push(decl_a);
    consumer.push(decl_b);
    consumer.push(decl_acc);

    let warp_idx_expr = "static_cast<int>(kittens::warpid())";
    let (decl_b_sub, b_sub) =
        tk20::decl_st_bf_subtile("__gemm_b_sub", &b_tile, k_dim, tile_n, "0", warp_idx_expr);
    let (decl_resid_sub, resid_sub) = tk20::decl_st_bf_subtile(
        "__gemm_resid_sub",
        &residual_smem,
        m_dim,
        tile_n,
        "0",
        warp_idx_expr,
    );
    consumer.push(decl_b_sub);
    consumer.push(decl_resid_sub);

    // Load A and B (bf16). Load residual directly into the fp32
    // accumulator (TK's bf16->fp32 convertor); residual then
    // enters mma as the C operand → `acc = A*B + residual` in a
    // single mma.
    consumer.push(tk20::warp_load_rt_from_st_bf(&a_rt, &in_smem));
    consumer.push(tk20::warp_load_rt_from_st_bf(&b_rt, &b_sub));
    consumer.push(tk20::warp_load_rt_fl_from_st_bf(&acc_rt, &resid_sub));

    consumer.push(tk20::warp_mma_AB(&acc_rt, &a_rt, &b_rt, &acc_rt));

    // Store accumulator IN PLACE to the per-warp residual slice
    // (overwrites the pre-add residual the loader staged).
    consumer.push(tk20::warp_store_st_bf_from_rt_fl(&resid_sub, &acc_rt));

    consumer.push(tk20::group_sync_named(ncw, bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive(1, &residual_done),
        tk20::group_arrive(1, &in_consumed),
        tk20::group_arrive(1, &weight_consumed),
    ]));

    // ---------------- Storer body ----------------
    // TMA-store the updated residual_smem back to its gmem slot.
    // Arrive on residual_consumed (NOT a separate out_consumed —
    // no out page exists for this variant).
    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait(1, &residual_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw_st_bf(
        1,
        &residual_gmem,
        &residual_smem,
        residual_bytes,
    ));
    storer.push(tk20::group_tma_store_async_wait(1));
    storer.push(tk20::group_arrive(1, &residual_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}

// ============================================================
// FusedGateUpActivateMul — `out = activation(A @ W_gate) *
// (A @ W_up)` where activation ∈ {silu, gelu}. Mirror of S10
// emit_gemm extended with a second matmul + tile-elementwise
// activation + tile-elementwise multiply.
// ============================================================
//
// Page lifecycle:
//   in_page:     Empty → Filled (loader TMA in)
//                      → Empty (consumer arrive page_consumed)
//   weight_page: Empty → Filled (loader TMA gate_buf + up_buf)
//                      → Empty (consumer arrive page_consumed)
//   out_page:    Empty → Filled (consumer warp::store of result)
//                      → Empty (storer TMA + arrive page_consumed)
//
// Substrate uses 3 pages + 2 scratch regions (`gate_buf` and
// `up_buf` under `MlpScope`). The single fused weight tensor is
// `[gate || up]` concatenated; loader TMAs the gate half from
// offset 0 and the up half from offset `gate_bytes`. Each warp
// owns disjoint TILE_N output cols under the AlongN split (no
// cross-warp reduction); a single `bar.sync` publishes before
// warp 0 signals page_done[out].

fn emit_fused_gate_up_activate_mul(
    node: &FusedGateUpActivateMul,
    budget: TapeBudget,
) -> RoleBodies {
    let in_page = node.in_page();
    let weight_page = node.gate_up_weight_page();
    let out_page = node.out_page();
    let consumer_phase = node.consumer_phase().raw();
    let storer_phase = node.storer_phase().raw();
    let loader_phase = storer_phase;
    let iters = node.iters().raw();
    let layer = node.layer().raw();
    let hidden_dim = node.hidden_dim().raw();
    let intermediate_dim = node.intermediate_dim().raw();
    let m_dim = node.num_tokens().raw();
    let tile_n = node.tile_n().raw();
    let in_act_slot = node.in_act_slot().raw();
    let out_act_slot = node.out_act_slot().raw();
    let weight_accessor_idx = node.weight_accessor_idx().raw();
    let bar_publish = node.consumer_bar_publish().raw();
    let gate_offset = node.gate_offset();
    let up_offset = node.up_offset();
    let gate_bytes = node.gate_bytes().raw();
    let up_bytes = node.up_bytes().raw();
    let activation = node.activation;

    let ncw = budget.num_consumer_warps;
    let num_layers = budget.num_layers.max(1);

    // Sprint 12 limit: single-shot b_tiles only. Multi-iter
    // pipelining requires per-iter mbarrier phases the IR doesn't
    // model. Mirrors S10 / S11.
    if iters != 1 {
        return RoleBodies::skipped("FusedGateUpActivateMul");
    }

    debug_assert_eq!(
        tile_n * ncw,
        intermediate_dim,
        "emit_fused_gate_up: AlongN split requires TILE_N ({tile_n}) * NCW ({ncw}) == INTERMEDIATE_DIM ({intermediate_dim})"
    );
    debug_assert!(
        m_dim % 16 == 0,
        "emit_fused_gate_up: TK 2.0 mma_AB requires M ({m_dim}) divisible by 16"
    );
    debug_assert!(
        hidden_dim % 16 == 0,
        "emit_fused_gate_up: TK 2.0 mma_AB requires HIDDEN_DIM ({hidden_dim}) divisible by 16"
    );
    debug_assert!(
        tile_n % 16 == 0,
        "emit_fused_gate_up: TK 2.0 mma_AB requires TILE_N ({tile_n}) divisible by 16"
    );
    debug_assert_eq!(
        gate_bytes,
        hidden_dim * intermediate_dim * 2,
        "emit_fused_gate_up: gate_bytes ({gate_bytes}) must equal HIDDEN_DIM*INTERMEDIATE_DIM*2"
    );
    debug_assert_eq!(
        up_bytes,
        hidden_dim * intermediate_dim * 2,
        "emit_fused_gate_up: up_bytes ({up_bytes}) must equal HIDDEN_DIM*INTERMEDIATE_DIM*2"
    );

    // Substrate handles. M = num_tokens; K = hidden_dim;
    // N = intermediate_dim. Activation lives in in_smem [M, K];
    // gate/up b-tiles live in scratch [K, N]; output in
    // out_smem [M, N].
    let in_smem = page_as_st_bf(in_page, m_dim, hidden_dim);
    let out_smem = page_as_st_bf(out_page, m_dim, intermediate_dim);
    let gate_buf = scratch_as_st_bf(gate_offset, hidden_dim, intermediate_dim);
    let up_buf = scratch_as_st_bf(up_offset, hidden_dim, intermediate_dim);

    let in_ready = page_ready_sem(in_page);
    let weight_ready = page_ready_sem(weight_page);
    let out_done = page_done_sem(out_page);
    let in_consumed = page_consumed_sem(in_page);
    let weight_consumed = page_consumed_sem(weight_page);
    let out_consumed = page_consumed_sem(out_page);

    let in_gmem = gmem_act_ptr_raw(in_act_slot);
    let out_gmem = gmem_act_ptr_raw(out_act_slot);
    let gate_gmem = gmem_weight_ptr_raw(weight_accessor_idx, layer, num_layers);
    let up_gmem =
        gmem_weight_ptr_raw_offset(weight_accessor_idx, layer, num_layers, gate_bytes);

    let bf16 = 2_u32;
    let act_bytes = m_dim * hidden_dim * bf16;
    let out_bytes = m_dim * intermediate_dim * bf16;
    let weight_total_bytes = gate_bytes + up_bytes;

    // ---------------- Loader body ----------------
    // Two semaphores total:
    //   in_ready    — 1 TMA for activation (act_bytes)
    //   weight_ready — 2 TMAs for gate + up (gate_bytes + up_bytes)
    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait(1, &in_consumed, loader_phase));
    loader.push(tk20::group_wait(1, &weight_consumed, loader_phase));
    loader.push(tk20::group_wait(1, &out_consumed, loader_phase));
    loader.push(tk20::group_tma_expect_bytes(1, &in_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw_st_bf(
        1, &in_smem, &in_gmem, act_bytes, &in_ready,
    ));
    loader.push(tk20::group_tma_expect_bytes(
        1,
        &weight_ready,
        weight_total_bytes,
    ));
    loader.push(tk20::group_tma_load_async_raw_st_bf(
        1,
        &gate_buf,
        &gate_gmem,
        gate_bytes,
        &weight_ready,
    ));
    loader.push(tk20::group_tma_load_async_raw_st_bf(
        1,
        &up_buf,
        &up_gmem,
        up_bytes,
        &weight_ready,
    ));

    // ---------------- Launcher body ----------------
    let launcher = CuBlock::new();

    // ---------------- Consumer body ----------------
    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait(1, &in_ready, consumer_phase));
    consumer.push(tk20::group_wait(1, &weight_ready, consumer_phase));

    // Per-warp register tiles. A is loaded full `[M, K]` per
    // warp; gate_b and up_b are per-warp `[K, TILE_N]` slices of
    // their respective scratch buffers; gate_acc and up_acc are
    // per-warp fp32 `[M, TILE_N]` accumulators.
    let (decl_a, a_rt) = tk20::decl_rt_bf_row("__gu_a", m_dim, hidden_dim);
    let (decl_gate_b, gate_b_rt) = tk20::decl_rt_bf_col("__gu_gate_b", hidden_dim, tile_n);
    let (decl_up_b, up_b_rt) = tk20::decl_rt_bf_col("__gu_up_b", hidden_dim, tile_n);
    let (decl_gate_acc, gate_acc_rt) = tk20::decl_rt_fl("__gu_gate_acc", m_dim, tile_n);
    let (decl_up_acc, up_acc_rt) = tk20::decl_rt_fl("__gu_up_acc", m_dim, tile_n);
    consumer.push(decl_a);
    consumer.push(decl_gate_b);
    consumer.push(decl_up_b);
    consumer.push(decl_gate_acc);
    consumer.push(decl_up_acc);

    // Per-warp B subtiles + per-warp OUT subtile.
    let warp_idx_expr = "static_cast<int>(kittens::warpid())";
    let (decl_gate_b_sub, gate_b_sub) = tk20::decl_st_bf_subtile(
        "__gu_gate_b_sub",
        &gate_buf,
        hidden_dim,
        tile_n,
        "0",
        warp_idx_expr,
    );
    let (decl_up_b_sub, up_b_sub) = tk20::decl_st_bf_subtile(
        "__gu_up_b_sub",
        &up_buf,
        hidden_dim,
        tile_n,
        "0",
        warp_idx_expr,
    );
    let (decl_out_sub, out_sub) =
        tk20::decl_st_bf_subtile("__gu_out_sub", &out_smem, m_dim, tile_n, "0", warp_idx_expr);
    consumer.push(decl_gate_b_sub);
    consumer.push(decl_up_b_sub);
    consumer.push(decl_out_sub);

    // Load A; load gate_b, up_b. Zero both accumulators; mma
    // both halves.
    consumer.push(tk20::warp_load_rt_from_st_bf(&a_rt, &in_smem));
    consumer.push(tk20::warp_load_rt_from_st_bf(&gate_b_rt, &gate_b_sub));
    consumer.push(tk20::warp_load_rt_from_st_bf(&up_b_rt, &up_b_sub));
    consumer.push(tk20::warp_zero_rt(&gate_acc_rt));
    consumer.push(tk20::warp_zero_rt(&up_acc_rt));
    consumer.push(tk20::warp_mma_AB(&gate_acc_rt, &a_rt, &gate_b_rt, &gate_acc_rt));
    consumer.push(tk20::warp_mma_AB(&up_acc_rt, &a_rt, &up_b_rt, &up_acc_rt));

    // Activation in-place on gate_acc.
    //   silu(x) = x * sigmoid(x) = x / (1 + expf(-x))
    //   gelu(x) = 0.5 * x * (1 + tanhf(sqrt(2/pi) * (x + 0.044715 * x^3)))
    //             (matches the GeLU-tanh approximation used
    //              elsewhere in the kernel; per-element CUDA
    //              math intrinsics, no TK-named gelu helper.)
    let activation_lambda = match activation {
        GateUpActivation::Silu => "x * (1.0f / (1.0f + __expf(-x)))",
        GateUpActivation::Gelu => "0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)))",
    };
    consumer.push(tk20::warp_apply_f32_rt_lambda(
        &gate_acc_rt,
        &gate_acc_rt,
        activation_lambda,
    ));

    // gate_acc *= up_acc (elementwise).
    consumer.push(tk20::warp_mul_rt_rt(&gate_acc_rt, &gate_acc_rt, &up_acc_rt));

    // Store final activated/multiplied tile to per-warp out
    // subtile. TK's internal type-converter downcasts fp32 →
    // bf16 in the same `kittens::warp::store` overload S10 uses.
    consumer.push(tk20::warp_store_st_bf_from_rt_fl(&out_sub, &gate_acc_rt));

    // Cross-warp publish. AlongN split → disjoint output cols
    // → only need to ensure all warps' stores are visible
    // before warp 0 signals page_done[out].
    consumer.push(tk20::group_sync_named(ncw, bar_publish));
    consumer.push(tk20::block_warp_zero(&[
        tk20::group_arrive(1, &out_done),
        tk20::group_arrive(1, &in_consumed),
        tk20::group_arrive(1, &weight_consumed),
    ]));

    // ---------------- Storer body ----------------
    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait(1, &out_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw_st_bf(
        1,
        &out_gmem,
        &out_smem,
        out_bytes,
    ));
    storer.push(tk20::group_tma_store_async_wait(1));
    storer.push(tk20::group_arrive(1, &out_consumed));

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
// Composition of the RmsNorm / FusedAddRmsNorm body (NCW-sliced
// vector ops + cross-warp sum reduce) and the Gemm body (AlongN
// per-warp [M, TILE_N] mma_AB) on the SAME `in_page`.
// ============================================================
//
// Page lifecycle:
//   in_page:          Empty -> Filled (loader TMA in)
//                            -> Produced (consumer norm writeback)
//                            -> Empty (consumer arrive page_consumed)
//   delta_page (opt): Empty -> Filled (loader TMA delta)
//                            -> Empty (consumer arrive page_consumed)
//   norm_w_page:      Empty -> Filled (loader TMA)
//                            -> Empty (consumer arrive page_consumed)
//   lin_w_page:       Empty -> Filled (loader TMA -> b_tile scratch)
//                            -> Empty (consumer arrive page_consumed)
//   out_page:         Empty -> Filled (consumer warp::store of acc)
//                            -> Empty (storer TMA + arrive page_consumed)
//
// `consumer_bar_publish` is reused TWICE: once after the norm
// writeback so all warps observe the full normalized [M, K] in
// shared memory before any warp reads it as the GEMM A operand,
// and once after the GEMM writeback so warp 0 can safely arrive
// on `page_done[out]`. Named bar.sync resets after all NCW arrive,
// so reuse is sound.
//
// Norm-flavor selection (`node.norm_kind`):
//   - RmsNorm:                  (no delta, no offset)
//   - AddRmsNorm:                (delta fold, no offset)
//   - AddScalarOffsetRmsNorm:    (delta fold, offset added to weight)
//   - MeanSubRmsNorm:            mean-subtract before sum-of-squares
//                                (uses bar_reduce twice -- named bar
//                                resets after each round)

fn emit_tk_fused_norm_gemm(node: &TkFusedNormGemm, budget: TapeBudget) -> RoleBodies {
    let in_page = node.in_page();
    let delta_page_opt = node.delta_page();
    let norm_weight_page = node.norm_weight_page();
    let linear_weight_page = node.linear_weight_page();
    let out_page = node.out_page();
    let consumer_phase = node.consumer_phase().raw();
    let storer_phase = node.storer_phase().raw();
    let loader_phase = storer_phase;
    let iters = node.iters().raw();
    let layer = node.layer().raw();
    let n_dim = node.n().raw();
    let k_dim = node.k().raw();
    let m_dim = node.num_tokens().raw();
    let tile_n = node.tile_n().raw();
    let chunk_k = node.chunk_k().raw();
    let in_act_slot = node.in_act_slot().raw();
    let delta_act_slot_opt = node.delta_act_slot();
    let out_act_slot = node.out_act_slot().raw();
    let norm_weight_accessor_idx = node.norm_weight_accessor_idx().raw();
    let linear_weight_accessor_idx = node.linear_weight_accessor_idx().raw();
    let bar_reduce = node.consumer_bar_reduce().raw();
    let bar_publish = node.consumer_bar_publish().raw();
    let eps_value = node.eps().raw();
    let norm_kind = node.norm_kind;
    let offset_opt = node.offset;
    let b_tile_offset = node.b_tile_offset();
    let partial_offset = node.partial_offset();

    let ncw = budget.num_consumer_warps;
    let num_layers = budget.num_layers.max(1);

    // S13 limit: single-shot b_tile only. Multi-iter pipelining
    // requires per-iter mbarrier phases the IR doesn't model.
    // Mirrors S10 / S11 / S12.
    if iters != 1 {
        return RoleBodies::skipped("TkFusedNormGemm");
    }

    debug_assert!(
        ncw > 0 && k_dim % ncw == 0,
        "emit_tk_fused_norm_gemm: K ({k_dim}) must be divisible by NCW ({ncw})"
    );
    let k_per_warp = k_dim / ncw;
    debug_assert_eq!(
        chunk_k, k_dim,
        "emit_tk_fused_norm_gemm: ITERS=1 requires CHUNK_K ({chunk_k}) == K ({k_dim})"
    );
    debug_assert_eq!(
        tile_n * ncw,
        n_dim,
        "emit_tk_fused_norm_gemm: AlongN split requires TILE_N ({tile_n}) * NCW ({ncw}) == N ({n_dim})"
    );
    debug_assert!(
        m_dim % 16 == 0,
        "emit_tk_fused_norm_gemm: TK 2.0 mma_AB requires M ({m_dim}) divisible by 16"
    );
    debug_assert!(
        k_dim % 16 == 0,
        "emit_tk_fused_norm_gemm: TK 2.0 mma_AB requires K ({k_dim}) divisible by 16"
    );
    debug_assert!(
        tile_n % 16 == 0,
        "emit_tk_fused_norm_gemm: TK 2.0 mma_AB requires TILE_N ({tile_n}) divisible by 16"
    );

    // Two views of the same in_page byte buffer: sv_bf<K> for the
    // norm-phase per-warp NCW-sliced register-vector ops, st_bf<M, K>
    // for the GEMM-phase A operand. Both reinterpret_cast the same
    // shared-memory page; views overlap (sv covers the first K bf16
    // elements = first row at M=1).
    let in_sv = page_as_sv_bf(in_page, k_dim);
    let in_st = page_as_st_bf(in_page, m_dim, k_dim);
    let norm_weight_sv = page_as_sv_bf(norm_weight_page, k_dim);
    let out_st = page_as_st_bf(out_page, m_dim, n_dim);
    let b_tile = scratch_as_st_bf(b_tile_offset, k_dim, n_dim);
    let partial = scratch_as::<super::handles::F32>(partial_offset);

    let in_ready = page_ready_sem(in_page);
    let norm_weight_ready = page_ready_sem(norm_weight_page);
    let lin_weight_ready = page_ready_sem(linear_weight_page);
    let out_done = page_done_sem(out_page);
    let in_consumed = page_consumed_sem(in_page);
    let norm_weight_consumed = page_consumed_sem(norm_weight_page);
    let lin_weight_consumed = page_consumed_sem(linear_weight_page);
    let out_consumed = page_consumed_sem(out_page);

    let in_gmem = gmem_act_ptr_raw(in_act_slot);
    let out_gmem = gmem_act_ptr_raw(out_act_slot);
    let norm_weight_gmem = gmem_weight_ptr_raw(norm_weight_accessor_idx, layer, num_layers);
    let lin_weight_gmem = gmem_weight_ptr_raw(linear_weight_accessor_idx, layer, num_layers);

    let bf16 = 2_u32;
    let act_bytes = m_dim * k_dim * bf16;
    let norm_weight_bytes = k_dim * bf16;
    let lin_weight_bytes = k_dim * n_dim * bf16;
    let out_bytes = m_dim * n_dim * bf16;

    // Optional delta plumbing for AddRmsNorm / AddScalarOffsetRmsNorm.
    let delta_sv_opt = delta_page_opt.map(|p| page_as_sv_bf(p, k_dim));
    let delta_ready_opt = delta_page_opt.map(page_ready_sem);
    let delta_consumed_opt = delta_page_opt.map(page_consumed_sem);
    let delta_gmem_opt = delta_act_slot_opt.map(|s| gmem_act_ptr_raw(s.raw()));

    // ---------------- Loader body ----------------
    let mut loader = CuBlock::new();
    loader.push(tk20::group_wait(1, &in_consumed, loader_phase));
    if let Some(delta_consumed) = delta_consumed_opt.as_ref() {
        loader.push(tk20::group_wait(1, delta_consumed, loader_phase));
    }
    loader.push(tk20::group_wait(1, &norm_weight_consumed, loader_phase));
    loader.push(tk20::group_wait(1, &lin_weight_consumed, loader_phase));
    loader.push(tk20::group_wait(1, &out_consumed, loader_phase));

    loader.push(tk20::group_tma_expect_bytes(1, &in_ready, act_bytes));
    loader.push(tk20::group_tma_load_async_raw(
        1, &in_sv, &in_gmem, act_bytes, &in_ready,
    ));

    if let (Some(delta_sv), Some(delta_ready), Some(delta_gmem)) = (
        delta_sv_opt.as_ref(),
        delta_ready_opt.as_ref(),
        delta_gmem_opt.as_ref(),
    ) {
        loader.push(tk20::group_tma_expect_bytes(1, delta_ready, act_bytes));
        loader.push(tk20::group_tma_load_async_raw(
            1, delta_sv, delta_gmem, act_bytes, delta_ready,
        ));
    }

    loader.push(tk20::group_tma_expect_bytes(
        1,
        &norm_weight_ready,
        norm_weight_bytes,
    ));
    loader.push(tk20::group_tma_load_async_raw(
        1,
        &norm_weight_sv,
        &norm_weight_gmem,
        norm_weight_bytes,
        &norm_weight_ready,
    ));

    loader.push(tk20::group_tma_expect_bytes(
        1,
        &lin_weight_ready,
        lin_weight_bytes,
    ));
    loader.push(tk20::group_tma_load_async_raw_st_bf(
        1,
        &b_tile,
        &lin_weight_gmem,
        lin_weight_bytes,
        &lin_weight_ready,
    ));

    // ---------------- Launcher body ----------------
    let launcher = CuBlock::new();

    // ---------------- Consumer body ----------------
    let mut consumer = CuBlock::new();
    consumer.push(tk20::group_wait(1, &in_ready, consumer_phase));
    if let Some(delta_ready) = delta_ready_opt.as_ref() {
        consumer.push(tk20::group_wait(1, delta_ready, consumer_phase));
    }
    consumer.push(tk20::group_wait(1, &norm_weight_ready, consumer_phase));
    consumer.push(tk20::group_wait(1, &lin_weight_ready, consumer_phase));

    // === Norm phase ===
    let (decl_act, act_rv) = tk20::decl_rv_fl("__lmh_act_rv", k_per_warp);
    let (decl_sq, sq_rv) = tk20::decl_rv_fl("__lmh_sq_rv", k_per_warp);
    let (decl_w, weight_rv) = tk20::decl_rv_fl("__lmh_norm_w_rv", k_per_warp);
    consumer.push(decl_act);
    consumer.push(decl_sq);
    consumer.push(decl_w);

    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32(ncw, &act_rv, &in_sv));

    // Residual fold (AddRmsNorm / AddScalarOffsetRmsNorm).
    if let Some(delta_sv) = delta_sv_opt.as_ref() {
        let (decl_delta, delta_rv) = tk20::decl_rv_fl("__lmh_delta_rv", k_per_warp);
        consumer.push(decl_delta);
        consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32(
            ncw, &delta_rv, delta_sv,
        ));
        consumer.push(tk20::warp_add_rv_rv(&act_rv, &act_rv, &delta_rv));
    }

    // MeanSubRmsNorm: mean reduce + subtract before sum-of-squares.
    // Reuses bar_reduce -- named bar resets after all NCW arrive,
    // so a second cross-warp reduce later in the body is sound.
    if matches!(norm_kind, LmHeadNormKind::MeanSubRmsNorm) {
        let (decl_msum, msum_local) =
            tk20::decl_local_f32("__lmh_mean_partial", "0.0f");
        consumer.push(decl_msum);
        consumer.push(tk20::warp_sum_to_scalar_f32(&msum_local, &act_rv));
        let (decl_full_msum, full_msum) =
            tk20::decl_local_f32("__lmh_mean_full", "0.0f");
        consumer.push(decl_full_msum);
        consumer.push(tk20::cross_warp_reduce_sum_f32(
            full_msum.as_str(),
            msum_local.as_str(),
            &partial,
            ncw,
            bar_reduce,
        ));
        // act -= mean   (mean = full_sum / K).
        let neg_mean = CuExpr::new(format!(
            "-({} * (1.0f / {}.0f))",
            full_msum.as_str(),
            k_dim
        ));
        consumer.push(tk20::warp_add_rv_scalar_f32(&act_rv, &act_rv, &neg_mean));
    }

    // Sum-of-squares.
    consumer.push(tk20::warp_copy_rv(&sq_rv, &act_rv));
    consumer.push(tk20::warp_mul_rv_rv(&sq_rv, &sq_rv, &sq_rv));
    let (decl_partial, partial_sum_expr) =
        tk20::decl_local_f32("__lmh_partial_sum", "0.0f");
    consumer.push(decl_partial);
    consumer.push(tk20::warp_sum_to_scalar_f32(&partial_sum_expr, &sq_rv));

    let (decl_full, full_sum_expr) =
        tk20::decl_local_f32("__lmh_full_sum", "0.0f");
    consumer.push(decl_full);
    consumer.push(tk20::cross_warp_reduce_sum_f32(
        full_sum_expr.as_str(),
        partial_sum_expr.as_str(),
        &partial,
        ncw,
        bar_reduce,
    ));
    let (decl_scale, scale_expr) = tk20::decl_rms_scale_local(
        "__lmh_scale",
        full_sum_expr.as_str(),
        k_dim,
        eps_value,
    );
    consumer.push(decl_scale);

    // act *= scale; load norm_weight; (optional offset on weight); act *= norm_w.
    consumer.push(tk20::warp_mul_rv_scalar_f32(&act_rv, &act_rv, &scale_expr));
    consumer.push(tk20::group_load_sv_to_rv_bf16_to_f32(
        ncw,
        &weight_rv,
        &norm_weight_sv,
    ));
    if let Some(offset_value) = offset_opt {
        let offset_lit = CuExpr::new(format!("{:e}f", offset_value.raw()));
        consumer.push(tk20::warp_add_rv_scalar_f32(
            &weight_rv,
            &weight_rv,
            &offset_lit,
        ));
    }
    consumer.push(tk20::warp_mul_rv_rv(&act_rv, &act_rv, &weight_rv));

    // Write normalized act back to in_smem (NCW-sliced).
    consumer.push(tk20::group_store_rv_to_sv_f32_to_bf16(ncw, &in_sv, &act_rv));

    // Cross-warp barrier: all warps must finish their slice of the
    // norm writeback before any warp reads in_st as the GEMM A
    // operand (each consumer warp loads the FULL [M, K] tile).
    consumer.push(tk20::group_sync_named(ncw, bar_publish));

    // === GEMM phase === (mirror of emit_gemm).
    let (decl_a, a_rt) = tk20::decl_rt_bf_row("__lmh_a", m_dim, k_dim);
    let (decl_b, b_rt) = tk20::decl_rt_bf_col("__lmh_b", k_dim, tile_n);
    let (decl_acc, acc_rt) = tk20::decl_rt_fl("__lmh_acc", m_dim, tile_n);
    consumer.push(decl_a);
    consumer.push(decl_b);
    consumer.push(decl_acc);

    let warp_idx_expr = "static_cast<int>(kittens::warpid())";
    let (decl_b_sub, b_sub) =
        tk20::decl_st_bf_subtile("__lmh_b_sub", &b_tile, k_dim, tile_n, "0", warp_idx_expr);
    let (decl_out_sub, out_sub) = tk20::decl_st_bf_subtile(
        "__lmh_out_sub",
        &out_st,
        m_dim,
        tile_n,
        "0",
        warp_idx_expr,
    );
    consumer.push(decl_b_sub);
    consumer.push(decl_out_sub);

    consumer.push(tk20::warp_load_rt_from_st_bf(&a_rt, &in_st));
    consumer.push(tk20::warp_load_rt_from_st_bf(&b_rt, &b_sub));
    consumer.push(tk20::warp_zero_rt(&acc_rt));
    consumer.push(tk20::warp_mma_AB(&acc_rt, &a_rt, &b_rt, &acc_rt));
    consumer.push(tk20::warp_store_st_bf_from_rt_fl(&out_sub, &acc_rt));

    // Cross-warp publish (reuse bar_publish): all warps' [M, TILE_N]
    // out subtile stores must be visible before warp 0 signals
    // page_done[out].
    consumer.push(tk20::group_sync_named(ncw, bar_publish));

    // Per-warp arrives. Warp 0 publishes page_done[out] + arrives
    // on each input page's consumed sem.
    let mut arrives = vec![
        tk20::group_arrive(1, &out_done),
        tk20::group_arrive(1, &in_consumed),
        tk20::group_arrive(1, &norm_weight_consumed),
        tk20::group_arrive(1, &lin_weight_consumed),
    ];
    if let Some(delta_consumed) = delta_consumed_opt.as_ref() {
        arrives.push(tk20::group_arrive(1, delta_consumed));
    }
    consumer.push(tk20::block_warp_zero(&arrives));

    // ---------------- Storer body ----------------
    let mut storer = CuBlock::new();
    storer.push(tk20::group_wait(1, &out_done, storer_phase));
    storer.push(tk20::group_tma_store_async_raw_st_bf(
        1, &out_gmem, &out_st, out_bytes,
    ));
    storer.push(tk20::group_tma_store_async_wait(1));
    storer.push(tk20::group_arrive(1, &out_consumed));

    RoleBodies {
        loader,
        launcher,
        consumer,
        storer,
        skipped: None,
    }
}
