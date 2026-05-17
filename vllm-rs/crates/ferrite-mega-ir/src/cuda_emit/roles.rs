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

use crate::nodes::{
    Add, BarrierSignal, BarrierWait, Embed, FusedAddRmsNorm, MegaNode, RmsNorm, ScalarMul,
    ScalarOffsetRmsNorm, TanhSoftCap,
};
use crate::tape::TapeBudget;

use super::cu::{CuBlock, CuExpr};
use super::handles::{
    gmem_act_ptr_raw, gmem_barrier_slot_ptr, gmem_input_ids, gmem_weight_ptr_raw,
    page_as_byte_ptr, page_as_sv_bf, page_consumed_sem, page_done_sem, page_ready_sem,
    scratch_as,
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
        MegaNode::FusedGateUpActivateMul(_) => RoleBodies::skipped("FusedGateUpActivateMul"),
        MegaNode::Embed(n) => emit_embed(n, budget),
        MegaNode::ScalarMul(n) => emit_scalar_mul(n, budget),
        MegaNode::TanhSoftCap(n) => emit_tanh_soft_cap(n, budget),
        MegaNode::ScalarOffsetRmsNorm(n) => emit_scalar_offset_rms_norm(n, budget),
        MegaNode::Gemm(_) => RoleBodies::skipped("Gemm"),
        MegaNode::FusedCublasGemmAdd(_) => RoleBodies::skipped("FusedCublasGemmAdd"),
        MegaNode::CutlassFusedNormGemm(_) => RoleBodies::skipped("CutlassFusedNormGemm"),
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
